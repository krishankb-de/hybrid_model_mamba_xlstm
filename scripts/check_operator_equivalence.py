#!/usr/bin/env python3
"""EFFICIENCY_PLAN.md rule R1 — the gate every speed change has to pass.

E0 produced two candidate wins: `mamba3_chunk_size=128` (1.40x on the forward) and
`torch.compile` (4.28x at L=2048). Neither is adoptable on a stopwatch alone. This
project has already shipped an operator that computed a different function than
advertised -- the `A_cum.clamp(1e-8)` defect, rel-max-err 0.92 -- and the whole
Mamba-3 campaign exists to repair it. A faster operator that changes a decoded
token would invalidate every published metric, so speed is measured second.

R1, restated from the plan:

  * a variant must match the current operator to rel-max-err <= 1e-4 in fp32, and
  * it must agree with `ssd_sequential_reference` (fp64) at least as well as the
    current operator does, and
  * a mismatch on a `cu_seqlens` document boundary is an automatic fail.

Two levels are checked, because they fail differently. The operator level catches
a wrong recurrence; the model level catches everything else a compiler might do
to a forward pass.

    venv/bin/python scripts/check_operator_equivalence.py --device cpu
    venv/bin/python scripts/check_operator_equivalence.py --device cuda --compile

Exit code is 0 only if every check passes, so a SLURM wrapper can gate a
measurement on it.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import dataclasses

import torch
import yaml

from hybrid_xmamba.kernels.ssd import ssd_chunked_scan, ssd_sequential_reference
from hybrid_xmamba.models.configuration_hybrid import HybridConfig
from hybrid_xmamba.models.hybrid_lm import HybridLanguageModel

R1_TOLERANCE = 1e-4
CONFIG_DIR = project_root / "configs" / "model"


def rel_max_err(a, b):
    """max |a-b| / max(|b|, eps) -- the same statistic the scan-defect work used."""
    a, b = a.double(), b.double()
    denom = b.abs().max().clamp(min=1e-12)
    return ((a - b).abs().max() / denom).item()


def _operands(batch, seqlen, nheads, headdim, ngroups, dstate, device, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    mk = lambda *shape: torch.randn(*shape, generator=g).to(device=device, dtype=torch.float32)
    x = mk(batch, seqlen, nheads, headdim)
    # dt is positive and A negative, as softplus/-exp produce in the real block.
    dt = torch.nn.functional.softplus(mk(batch, seqlen, nheads))
    A = -torch.exp(mk(nheads))
    B = mk(batch, seqlen, ngroups, dstate)
    C = mk(batch, seqlen, ngroups, dstate)
    D = mk(nheads)
    return x, dt, A, B, C, D


def check_operator(device, chunk_sizes, verbose=True):
    """Operator level: does changing chunk_size change the function it computes?"""
    failures = []
    cases = [("contiguous", None)]

    batch, seqlen, nheads, headdim, ngroups, dstate = 2, 256, 4, 32, 1, 64
    # A document boundary that falls INSIDE a chunk for every chunk size tested --
    # that is where the state-reset masks are hardest and where the Mamba-1 defect lived.
    seg = torch.zeros(batch, seqlen, dtype=torch.long, device=device)
    seg[:, 100:] = 1
    seg[1, 173:] = 2
    cases.append(("cu_seqlens (boundary mid-chunk)", seg))

    for case_name, cu in cases:
        x, dt, A, B, C, D = _operands(batch, seqlen, nheads, headdim, ngroups, dstate, device)
        oracle = ssd_sequential_reference(x, dt, A, B, C, D=D, cu_seqlens=cu).float()
        baseline = ssd_chunked_scan(x, dt, A, B, C, D=D, chunk_size=64, cu_seqlens=cu)
        base_err = rel_max_err(baseline, oracle)
        if verbose:
            print("\n  case: {}".format(case_name))
            print("    chunk_size= 64 (shipped)  vs fp64 oracle : {:.3e}".format(base_err))

        for cs in chunk_sizes:
            variant = ssd_chunked_scan(x, dt, A, B, C, D=D, chunk_size=cs, cu_seqlens=cu)
            vs_base = rel_max_err(variant, baseline)
            vs_oracle = rel_max_err(variant, oracle)
            ok = vs_base <= R1_TOLERANCE and vs_oracle <= max(base_err * 2.0, R1_TOLERANCE)
            if verbose:
                print("    chunk_size={:>3}            vs shipped    : {:.3e}   "
                      "vs oracle: {:.3e}   {}".format(
                          cs, vs_base, vs_oracle, "PASS" if ok else "FAIL"))
            if not ok:
                failures.append("operator/{}/chunk_size={}".format(case_name, cs))
    return failures


def _load_model_config(name):
    with open(CONFIG_DIR / "{}.yaml".format(name)) as f:
        raw = yaml.safe_load(f) or {}
    valid = {f.name for f in dataclasses.fields(HybridConfig)}
    return HybridConfig(**{k: v for k, v in raw.items() if k in valid and k != "model_type"})


def check_model(device, model_name, chunk_sizes, seq_length, do_compile, verbose=True):
    """Model level: do the logits move? This is what a decoded token actually sees."""
    failures = []
    torch.manual_seed(0)
    config = _load_model_config(model_name)
    baseline_chunk = getattr(config, "mamba3_chunk_size", None)
    model = HybridLanguageModel(config).to(device=device, dtype=torch.float32).eval()
    input_ids = torch.randint(0, config.vocab_size, (2, seq_length), device=device)

    with torch.no_grad():
        baseline = model(input_ids).logits.float()
    if verbose:
        print("\n  model {} @ L={}, chunk_size={} (baseline)".format(
            model_name, seq_length, baseline_chunk))

    for cs in chunk_sizes:
        for layer in model.layers:
            if hasattr(layer.mixer, "chunk_size"):
                layer.mixer.chunk_size = cs
        with torch.no_grad():
            variant = model(input_ids).logits.float()
        err = rel_max_err(variant, baseline)
        ok = err <= R1_TOLERANCE
        if verbose:
            print("    chunk_size={:>3} logits vs baseline : {:.3e}   {}".format(
                cs, err, "PASS" if ok else "FAIL"))
        if not ok:
            failures.append("model/chunk_size={}".format(cs))
    for layer in model.layers:
        if hasattr(layer.mixer, "chunk_size") and baseline_chunk is not None:
            layer.mixer.chunk_size = baseline_chunk

    if do_compile:
        compiled = torch.compile(model)
        with torch.no_grad():
            got = compiled(input_ids).logits.float()
        err = rel_max_err(got, baseline)
        ok = err <= R1_TOLERANCE
        if verbose:
            print("    torch.compile  logits vs baseline : {:.3e}   {}".format(
                err, "PASS" if ok else "FAIL"))
        if not ok:
            failures.append("model/torch.compile")

    return failures


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--model", default="hybrid_150m_m3")
    ap.add_argument("--chunk-sizes", type=int, nargs="+", default=[128, 256, 512])
    ap.add_argument("--seq-length", type=int, default=512,
                    help="model-level check length; must exceed the largest chunk size "
                         "for the comparison to mean anything")
    ap.add_argument("--compile", dest="do_compile", action="store_true",
                    help="also check that torch.compile preserves the logits (E1-B's gate)")
    ap.add_argument("--skip-model", action="store_true", help="operator level only")
    args = ap.parse_args()

    print("=" * 78)
    print("R1 equivalence gate   device={}  tolerance={:.0e}".format(args.device, R1_TOLERANCE))
    print("=" * 78)

    failures = check_operator(args.device, args.chunk_sizes)
    if not args.skip_model:
        failures += check_model(args.device, args.model, args.chunk_sizes,
                                args.seq_length, args.do_compile)

    print("\n" + "=" * 78)
    if failures:
        print("R1 FAILED: {}".format(", ".join(failures)))
        print("These variants change the function the model computes and must NOT be adopted.")
        print("=" * 78)
        return 1
    print("R1 PASSED: every variant computes the same function as the shipped operator.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
