#!/usr/bin/env python3
"""The M7 ablation ladder, in one place (MAMBA3_PLAN.md M5 / M7-B).

Arms A2..A6 are not separate yaml files. They are `hybrid_150m_m3.yaml` plus a handful of
`model.mamba3_*=...` overrides, which is why every flag is *declared* in that yaml: Hydra's
strict struct mode rejects an override for a key the config never named. Keeping the ladder as
overrides rather than as five near-identical yamls means the arms cannot drift apart in anything
except the lever under test -- an accidental edit to A5's `d_state` would otherwise be invisible.

The reason this is a module and not a paragraph in the README is FM5, the plan's most expensive
failure mode. On 2026-09-06 the A1 arm was submitted, ran for 36 minutes, and logged
`scan_impl=legacy | dt_init=none`: the config named the levers, the trainer built its
`HybridConfig` from a hand-written kwarg list, and the levers never arrived. The defence is that
exactly one definition of each arm exists and *both* the pre-flight and the submission read it:

    python scripts/mamba3_arms.py list                 # the ladder, one line per arm
    python scripts/mamba3_arms.py env A3               # shell exports for sbatch
    python scripts/mamba3_arms.py verify [--full]      # build each arm, check its fingerprint

Submitting one arm:

    eval "$(python scripts/mamba3_arms.py env A5)"
    sbatch --time=12:00:00 scripts/train_stage0_150m_h100.sh
"""

import argparse
import shlex
import sys
from typing import Any, Dict, List, NamedTuple

SCREEN_STEPS = 12000        # M7-B: short enough to screen 8 arms, long enough for WSD to decay
SCREEN_WARMUP = 500
SCREEN_SEED = 42            # identical across arms -- the screen is a *paired* comparison


class Arm(NamedTuple):
    """One rung of the ladder: a config, the levers it flips, and how to recognise it in a log."""

    config: str
    overrides: Dict[str, Any]
    seed: int
    isolates: str
    expect: List[str]        # substrings the ARCH fingerprint must contain
    walltime: str = "12:00:00"


_M3 = "hybrid_150m_m3"
_M3_BASE = ["mamba3x9", "mlstmx3", "d_state=128", "head_dim=64", "ngroups=1"]

ARMS = {
    "A0": Arm(
        config="hybrid_150m_v2", overrides={}, seed=SCREEN_SEED,
        isolates="control; must reproduce the Phase-5 curve or the harness is wrong",
        expect=["mambax9", "scan_impl=legacy", "tfla_impl=legacy", "dt_init=none"],
    ),
    "A0-seed": Arm(
        config="hybrid_150m_v2", overrides={}, seed=1234,
        isolates="THE NOISE FLOOR -- run before ranking anything (M7-A)",
        expect=["mambax9", "scan_impl=legacy", "dt_init=none"],
    ),
    "A1": Arm(
        config="hybrid_150m_a1", overrides={}, seed=SCREEN_SEED,
        isolates="the defect fix alone: exact scan + dt init + no Delta-norm. Screen-only.",
        expect=["mambax9", "scan_impl=exact", "tfla_impl=exact", "dt_init=mamba",
                "norm_topology=hybrid_bc"],
        # MEASURED: job 2513057 hit a 12 h TIMEOUT unfinished. The exact scan for Mamba-1's
        # (d_inner, dstate) A cannot use the cheap log-segsum -- it flips the parallel axis
        # instead, depth cs + L/cs -- and the plan accepted 3-5x slower because A1 never
        # enters the pipeline. 3-5x of A0's 2:37 training loop is 8-13 h on top of ~5 h of
        # validation, so 12 h was never enough. This is the one arm that needs a long clock.
        walltime="24:00:00",
    ),
    "A2": Arm(
        config=_M3, overrides={}, seed=SCREEN_SEED,
        isolates="SSD + 8x state -- report as a BUNDLE, not as 'SSD is better'",
        expect=_M3_BASE + ["conv=True", "trapezoid=False", "rope=False", "bc_bias=none"],
    ),
    "A3": Arm(
        config=_M3, overrides={"mamba3_use_trapezoid": True}, seed=SCREEN_SEED,
        isolates="exponential-trapezoidal discretization (paper Prop. 1)",
        expect=_M3_BASE + ["trapezoid=True", "rope=False"],
    ),
    "A4": Arm(
        config=_M3, overrides={"mamba3_use_rope": True}, seed=SCREEN_SEED,
        isolates="complex-valued state via the RoPE trick (paper Sec 3.2)",
        expect=_M3_BASE + ["trapezoid=False", "rope=True"],
    ),
    "A5": Arm(
        config=_M3, overrides={"mamba3_use_trapezoid": True, "mamba3_use_rope": True},
        seed=SCREEN_SEED, isolates="Mamba-3 SISO: both mechanisms together",
        expect=_M3_BASE + ["trapezoid=True", "rope=True"],
    ),
    "A6": Arm(
        config=_M3,
        overrides={"mamba3_use_trapezoid": True, "mamba3_use_rope": True,
                   "mamba3_bc_bias": "one_init", "mamba3_use_conv": False},
        seed=SCREEN_SEED,
        isolates="Sec 3.4 refinements on top of A5: B/C biases and no short conv",
        expect=_M3_BASE + ["trapezoid=True", "rope=True", "bc_bias=one_init", "conv=False"],
    ),
}

# --- M7-G: re-test Prop. 3/4 at a rotation rate that is a position code, not a scrambler ----
# The M7-B rope arms did not test complex-valued state; they tested theta_max=1.0, the block
# default, which was the only value reachable because mamba3_theta_max was missing from
# HybridConfig. At dt_limit=1.0 that is 1 rad/token and 81 turns over 512 tokens. These arms
# sweep the rate instead. 0.002 keeps the whole sequence inside ~1/6 of a turn; 0.02 inside
# ~1.6 turns; 0.2 inside ~16 and should start to degrade if the aliasing story is right.
for _tag, _theta in (("lo", 0.002), ("mid", 0.02), ("hi", 0.2)):
    ARMS["A4-{}".format(_tag)] = Arm(
        config=_M3,
        overrides={"mamba3_use_rope": True, "mamba3_theta_max": _theta},
        seed=SCREEN_SEED,
        isolates="RoPE at theta_max={} -- {:.2f} turns over 512 tokens".format(
            _theta, 512 * _theta / (2 * 3.141592653589793)),
        expect=_M3_BASE + ["trapezoid=False", "rope=True"],
    )

# Deliberately absent, so it is not silently re-proposed: `mamba3_mimo_rank > 1` (decision 3,
# +3.2% params leaves the parameter-matched regime), `mamba3_ngroups > 1` and
# `mamba3_d_state > 128` (pre-registered as *scaled* arms only, never the headline comparison).


def hydra_overrides(arm: Arm) -> List[str]:
    """The `model.*=...` arguments this arm adds. Booleans are lowercased for Hydra."""
    out = []
    for key, value in sorted(arm.overrides.items()):
        rendered = str(value).lower() if isinstance(value, bool) else str(value)
        out.append("model.{}={}".format(key, rendered))
    return out


def _get(name: str) -> Arm:
    if name not in ARMS:
        raise SystemExit("unknown arm {!r}; known arms: {}".format(name, ", ".join(ARMS)))
    return ARMS[name]


def cmd_list(_args) -> int:
    width = max(len(n) for n in ARMS)
    for name, arm in ARMS.items():
        extra = " ".join(hydra_overrides(arm)) or "-"
        print("{:<{w}}  {:<18} seed={:<5} time={} {}".format(
            name, arm.config, arm.seed, arm.walltime, extra, w=width))
        print("{:<{w}}  {}".format("", arm.isolates, w=width))
    return 0


def cmd_env(args) -> int:
    """Emit shell exports. `eval "$(... env A5)"` then sbatch the 150M wrapper."""
    arm = _get(args.arm)
    exports = {
        "MODEL_CONFIG": arm.config,
        "SEED": str(arm.seed),
        "MAX_STEPS": str(args.steps),
        "WARMUP": str(args.warmup),
        "SAVE_TOP_K": str(args.save_top_k),
        "EXPERIMENT": "m3_screen_{}_s{}".format(args.arm.replace("-", "_"), arm.seed),
        "EXTRA_OVERRIDES": " ".join(hydra_overrides(arm)),
    }
    for key, value in exports.items():
        print("export {}={}".format(key, shlex.quote(value)))
    print("# suggested walltime for this arm: sbatch --time={}".format(arm.walltime))
    return 0


def cmd_verify(args) -> int:
    """Build every arm and check its ARCH fingerprint against `expect`.

    `--full` builds the real 150M models and additionally checks the parameter-matched band; the
    default shrinks `dim`/`num_layers` so the check runs in seconds. The flag tokens are
    scale-independent, so the fast path still catches a lever that never reached the mixer -- the
    failure this whole module exists to prevent.
    """
    import dataclasses

    import torch
    import yaml

    from hybrid_xmamba.models.configuration_hybrid import HybridConfig
    from hybrid_xmamba.models.hybrid_lm import HybridLanguageModel

    failures = []
    for name, arm in ARMS.items():
        raw = yaml.safe_load(open("configs/model/{}.yaml".format(arm.config)))
        raw.update(arm.overrides)
        if not args.full:
            raw.update(dim=128, num_layers=12, num_heads=4, head_dim=32,
                       vocab_size=1024, max_position_embeddings=128)
        # from_hydra is the call the training entry points make. An earlier version of the
        # pre-flight filtered dataclass fields inline -- a *different* path, which is how it
        # passed while the trainer was silently dropping three fields.
        torch.manual_seed(0)
        model = HybridLanguageModel(HybridConfig.from_hydra(raw))
        fp = model.architecture_fingerprint()
        missing = [tok for tok in arm.expect if tok not in fp]
        n = sum(p.numel() for p in model.parameters())
        print("{:<8} {}".format(name, arm.config))
        print("         {}".format(fp))
        if missing:
            failures.append("{}: fingerprint missing {}".format(name, missing))
        if args.full and not 181e6 < n < 186e6:
            failures.append("{}: {:,} params outside the [181,186]M band".format(name, n))
        del model

    unknown = set()
    fields = {f.name for f in dataclasses.fields(HybridConfig)}
    for name, arm in ARMS.items():
        unknown |= {k for k in arm.overrides if k not in fields}
    if unknown:
        failures.append("overrides name no HybridConfig field: {}".format(sorted(unknown)))

    if failures:
        print("\nFAIL:")
        for f in failures:
            print("  -", f)
        return 1
    scope = "at full 150M scale" if args.full else "at reduced scale (flags only)"
    print("\nOK: all {} arms build the operator they claim, {}.".format(len(ARMS), scope))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="print the ladder").set_defaults(func=cmd_list)

    env = sub.add_parser("env", help="shell exports for one arm")
    env.add_argument("arm")
    env.add_argument("--steps", type=int, default=SCREEN_STEPS)
    env.add_argument("--warmup", type=int, default=SCREEN_WARMUP)
    env.add_argument("--save-top-k", type=int, default=1)
    env.set_defaults(func=cmd_env)

    ver = sub.add_parser("verify", help="build each arm and check its fingerprint")
    ver.add_argument("--full", action="store_true", help="build the real 150M models")
    ver.set_defaults(func=cmd_verify)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
