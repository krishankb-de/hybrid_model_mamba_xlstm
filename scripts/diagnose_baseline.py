"""Phase 2 — Diagnostic probe on Phase 6e checkpoint (read-only).

Probes:
  1. mLSTM gate health  : raw i_gate_proj / f_gate_proj pre-activation max/mean/std
  2. Layer-wise L2 norms: mean hidden-state norm per HybridBlock output
  3. Doc-contamination  : 2-doc synthetic packed sequence; asserts doc-B
                         output is unaffected by doc-A perturbation
  4. STS-B align/unif   : alignment + uniformity (Wang & Isola 2020) on
                         STS-B dev positive pairs; Spearman rho across all pairs
  5. MIMIC cosine hist  : (optional) paired image-text cosine on MIMIC-val cache

Usage:
    python scripts/diagnose_baseline.py \\
        --ckpt output_willi_server/total_loss=2.6274.ckpt \\
        --device cpu \\
        --output-dir outputs/baseline_probe

On Willi (GPU):
    python scripts/diagnose_baseline.py \\
        --ckpt /path/to/phase6e/best.ckpt \\
        --device cuda \\
        --mimic-cache /scratch/bhushkri/mimic_cxr_cache \\
        --output-dir outputs/baseline_probe
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

from hybrid_xmamba.models.configuration_hybrid import HybridConfig
from hybrid_xmamba.models.hybrid_lm import HybridLanguageModel, HybridTextEncoder
from hybrid_xmamba.layers.mlstm_block import mLSTMBlock
from hybrid_xmamba.layers.hybrid_block import HybridBlock


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def _strip_prefixes_for_encoder(state_dict: Dict) -> Dict:
    """Strip Lightning / torch.compile prefixes; keep lm.* and projection_head.* intact."""
    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("model._orig_mod."):
            new_k = k[len("model._orig_mod."):]
        elif k.startswith("_orig_mod.model."):
            new_k = k[len("_orig_mod.model."):]
        elif k.startswith("_orig_mod."):
            new_k = k[len("_orig_mod."):]
        elif k.startswith("model."):
            new_k = k[len("model."):]
        else:
            new_k = k
        cleaned[new_k] = v
    return cleaned


def _patch_transformers_compat() -> None:
    """Inject stubs for classes removed/renamed between transformers versions.

    The Phase 6e ckpt was saved with transformers 4.x which had
    BertSdpaSelfAttention as a distinct class. transformers 5.x merged it into
    BertSelfAttention. We only need the state_dict tensors, so a stub suffices.
    """
    try:
        import transformers.models.bert.modeling_bert as _bert_mod
        if not hasattr(_bert_mod, "BertSdpaSelfAttention"):
            _bert_mod.BertSdpaSelfAttention = _bert_mod.BertSelfAttention
    except Exception:
        pass


def load_encoder(ckpt_path: str, device: str) -> HybridTextEncoder:
    _patch_transformers_compat()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw_sd = ckpt.get("state_dict", ckpt)

    print(f"  Checkpoint keys (first 8): {list(raw_sd.keys())[:8]}")

    sd = _strip_prefixes_for_encoder(raw_sd)

    # Infer config from embedding weight shape
    dim = 512
    for k, v in sd.items():
        if "token_embedding.weight" in k:
            dim = int(v.shape[1])
            break

    num_layers = 0
    import re
    for k in sd.keys():
        m = re.search(r"layers\.(\d+)\.", k)
        if m:
            idx = int(m.group(1))
            if idx + 1 > num_layers:
                num_layers = idx + 1
    if num_layers == 0:
        num_layers = 8

    base = ["mamba", "mamba", "mlstm"]
    layer_pattern = [base[i % len(base)] for i in range(num_layers)]

    cfg = HybridConfig(
        dim=dim,
        num_layers=num_layers,
        layer_pattern=layer_pattern,
        max_position_embeddings=1024,
    )
    model = HybridTextEncoder(cfg, embed_dim=dim)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"  Missing keys : {len(missing)}  {missing[:4] if missing else ''}")
    print(f"  Unexpected   : {len(unexpected)}  {unexpected[:4] if unexpected else ''}")
    model.eval()
    model.to(device)
    return model


# ---------------------------------------------------------------------------
# Probe 1 — mLSTM gate health
# ---------------------------------------------------------------------------

def probe_mlstm_gates(
    model: HybridTextEncoder,
    device: str,
    n_tokens: int = 512,
    batch: int = 4,
) -> Dict:
    """Hook i_gate_proj and f_gate_proj linear outputs (pre-activation) per mLSTM layer."""
    gate_stats: Dict[str, Dict[str, float]] = {}
    hooks = []

    def _make_hook(layer_idx: int, gate_name: str):
        def _hook(_module: nn.Module, _inp: Tuple, out: torch.Tensor):
            key = f"layer_{layer_idx}_{gate_name}"
            gate_stats[key] = {
                "max": float(out.abs().max().item()),
                "mean": float(out.mean().item()),
                "std": float(out.std().item()),
                "frac_gt15": float((out.abs() > 15.0).float().mean().item()),
            }
        return _hook

    mlstm_idx = 0
    for block_idx, block in enumerate(model.lm.layers):
        if isinstance(block, HybridBlock) and isinstance(block.mixer, mLSTMBlock):
            mixer = block.mixer
            hooks.append(mixer.i_gate_proj.register_forward_hook(
                _make_hook(block_idx, "i_gate_raw")))
            hooks.append(mixer.f_gate_proj.register_forward_hook(
                _make_hook(block_idx, "f_gate_raw")))
            mlstm_idx += 1

    dummy = torch.randint(0, 50257, (batch, n_tokens), device=device)
    with torch.no_grad():
        model.lm(dummy)

    for h in hooks:
        h.remove()

    return gate_stats


# ---------------------------------------------------------------------------
# Probe 2 — Layer-wise hidden norms
# ---------------------------------------------------------------------------

def probe_layer_norms(
    model: HybridTextEncoder,
    device: str,
    n_tokens: int = 512,
    batch: int = 4,
) -> Dict:
    """Record mean L2 norm of each HybridBlock's output (post-residual)."""
    norm_stats: Dict[str, float] = {}
    hooks = []

    def _make_hook(block_idx: int):
        def _hook(_module: nn.Module, _inp: Tuple, out: torch.Tensor):
            if isinstance(out, tuple):
                out = out[0]
            norm_stats[f"block_{block_idx}_out_norm"] = float(
                out.norm(dim=-1).mean().item()
            )
        return _hook

    for block_idx, block in enumerate(model.lm.layers):
        if isinstance(block, HybridBlock):
            hooks.append(block.register_forward_hook(_make_hook(block_idx)))

    dummy = torch.randint(0, 50257, (batch, n_tokens), device=device)
    with torch.no_grad():
        model.lm(dummy)

    for h in hooks:
        h.remove()

    return norm_stats


# ---------------------------------------------------------------------------
# Probe 3 — Doc-contamination check
# ---------------------------------------------------------------------------

def probe_doc_contamination(
    model: HybridTextEncoder,
    device: str,
    doc_len: int = 32,
) -> Dict:
    """Check whether doc-B output is independent of doc-A content.

    Packs [doc_A + EOS + doc_B] as a single sequence, runs forward,
    records doc-B region hidden states. Then perturbs doc-A and reruns.
    L2 diff at doc-B positions quantifies cross-document leakage.
    """
    EOS = 50256
    torch.manual_seed(42)
    doc_a = torch.randint(100, 800, (doc_len,))
    doc_b = torch.randint(1000, 5000, (doc_len,))
    eos = torch.tensor([EOS])

    packed = torch.cat([doc_a, eos, doc_b]).unsqueeze(0).to(device)  # (1, 2L+1)
    b_start = doc_len + 1

    hidden_states_orig = []
    hooks = []

    def _capture_hook(_module: nn.Module, _inp: Tuple, out: torch.Tensor):
        if isinstance(out, tuple):
            out = out[0]
        hidden_states_orig.append(out.detach().cpu())

    for block in model.lm.layers:
        if isinstance(block, HybridBlock):
            hooks.append(block.register_forward_hook(_capture_hook))

    with torch.no_grad():
        model.lm(packed)
    for h in hooks:
        h.remove()

    # Perturb doc_A
    doc_a_perturb = torch.randint(10000, 40000, (doc_len,))
    packed_perturb = torch.cat([doc_a_perturb, eos, doc_b]).unsqueeze(0).to(device)

    hidden_states_perturb = []
    hooks = []

    def _capture_hook2(_module: nn.Module, _inp: Tuple, out: torch.Tensor):
        if isinstance(out, tuple):
            out = out[0]
        hidden_states_perturb.append(out.detach().cpu())

    for block in model.lm.layers:
        if isinstance(block, HybridBlock):
            hooks.append(block.register_forward_hook(_capture_hook2))

    with torch.no_grad():
        model.lm(packed_perturb)
    for h in hooks:
        h.remove()

    leak_ratios = []
    for h_orig, h_perturb in zip(hidden_states_orig, hidden_states_perturb):
        diff = (h_orig[:, b_start:, :] - h_perturb[:, b_start:, :]).norm(dim=-1).mean()
        base = h_orig[:, b_start:, :].norm(dim=-1).mean().clamp(min=1e-6)
        leak_ratios.append(float((diff / base).item()))

    mean_leak = float(sum(leak_ratios) / len(leak_ratios)) if leak_ratios else 0.0
    max_leak = float(max(leak_ratios)) if leak_ratios else 0.0

    return {
        "mean_leak_ratio": mean_leak,
        "max_leak_ratio": max_leak,
        "per_layer_leak": {f"block_{i}": v for i, v in enumerate(leak_ratios)},
        "verdict": "LEAK_PRESENT" if mean_leak > 0.01 else "CLEAN",
    }


# ---------------------------------------------------------------------------
# Probe 4 — STS-B alignment / uniformity
# ---------------------------------------------------------------------------

def _spearman_rho(x: List[float], y: List[float]) -> float:
    n = len(x)
    if n < 2:
        return 0.0

    def _rank(vals: List[float]) -> List[float]:
        sorted_idx = sorted(range(n), key=lambda i: vals[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n - 1 and vals[sorted_idx[j + 1]] == vals[sorted_idx[j]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[sorted_idx[k]] = avg
            i = j + 1
        return ranks

    rx = _rank(x)
    ry = _rank(y)
    d2 = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    return 1.0 - 6.0 * d2 / (n * (n * n - 1))


def probe_stsb(
    model: HybridTextEncoder,
    tokenizer: AutoTokenizer,
    device: str,
    max_pairs: int = 500,
    batch_size: int = 32,
    stsb_cache: Optional[str] = None,
) -> Dict:
    """Alignment, uniformity, and Spearman rho on STS-B dev."""
    try:
        from datasets import load_dataset
    except ImportError:
        return {"error": "datasets library not available"}

    ds_kwargs: Dict = {}
    if stsb_cache:
        ds_kwargs["cache_dir"] = stsb_cache
    try:
        ds = load_dataset("glue", "stsb", split="validation", **ds_kwargs)
    except Exception as e:
        return {"error": f"STS-B load failed: {e}"}

    pairs = list(zip(ds["sentence1"], ds["sentence2"], ds["label"]))[:max_pairs]

    def _encode_sentences(sents: List[str]) -> torch.Tensor:
        all_embs = []
        for i in range(0, len(sents), batch_size):
            batch_sents = sents[i: i + batch_size]
            enc = tokenizer(
                batch_sents,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            with torch.no_grad():
                embs = model.encode(input_ids, attention_mask=attention_mask)
            all_embs.append(embs.cpu())
        return torch.cat(all_embs, dim=0)

    sents1 = [p[0] for p in pairs]
    sents2 = [p[1] for p in pairs]
    labels = [p[2] for p in pairs]

    z1 = _encode_sentences(sents1)
    z2 = _encode_sentences(sents2)

    # Spearman rho
    cos_sims = (z1 * z2).sum(dim=-1).tolist()
    rho = _spearman_rho(cos_sims, labels)

    # Alignment / uniformity on positive pairs (score >= 3.5 out of 5)
    pos_mask = [l >= 3.5 for l in labels]
    z1_pos = z1[[i for i, m in enumerate(pos_mask) if m]]
    z2_pos = z2[[i for i, m in enumerate(pos_mask) if m]]

    n_pos = z1_pos.shape[0]
    alignment = (z1_pos - z2_pos).pow(2).sum(dim=-1).mean().item() if n_pos > 1 else None
    uniformity = None
    if n_pos > 1:
        all_pos = torch.cat([z1_pos, z2_pos], dim=0)
        sq_dists = torch.pdist(all_pos, p=2).pow(2)
        uniformity = sq_dists.mul(-2.0).exp().mean().log().item()

    return {
        "n_pairs": len(pairs),
        "n_positive_pairs": n_pos,
        "spearman_rho": rho,
        "alignment": alignment,
        "uniformity": uniformity,
        "cos_sim_mean": float(torch.tensor(cos_sims).mean().item()),
        "cos_sim_std": float(torch.tensor(cos_sims).std().item()),
    }


# ---------------------------------------------------------------------------
# Probe 5 — MIMIC-val cosine histogram (optional)
# ---------------------------------------------------------------------------

def probe_mimic_cosine(
    model: HybridTextEncoder,
    tokenizer: AutoTokenizer,
    device: str,
    mimic_cache: str,
    max_pairs: int = 500,
    batch_size: int = 32,
    output_dir: Optional[str] = None,
) -> Dict:
    """Cosine histogram on MIMIC-val cached pairs (image-text)."""
    cache_path = Path(mimic_cache)
    if not cache_path.exists():
        return {"error": f"MIMIC cache not found: {mimic_cache}"}

    # Expect a file with {"texts": [...], "image_embs": ...} or similar
    pair_file = cache_path / "mimic_val_pairs.json"
    img_emb_file = cache_path / "mimic_val_image_embs.pt"

    if not pair_file.exists():
        return {"error": f"Expected {pair_file} — run cache prep first"}

    with open(pair_file) as f:
        data = json.load(f)

    texts = data["texts"][:max_pairs]
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch_sents = texts[i: i + batch_size]
        enc = tokenizer(
            batch_sents,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        with torch.no_grad():
            embs = model.encode(
                enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device),
            )
        all_embs.append(embs.cpu())
    text_embs = torch.cat(all_embs, dim=0)

    if img_emb_file.exists():
        img_embs = torch.load(img_emb_file, map_location="cpu")[:max_pairs]
        img_embs = F.normalize(img_embs, dim=-1)
        cos_sims = (text_embs * img_embs).sum(dim=-1).tolist()
    else:
        cos_sims = None

    result: Dict = {
        "n_pairs": len(texts),
        "text_emb_norm_mean": float(text_embs.norm(dim=-1).mean().item()),
    }
    if cos_sims is not None:
        import statistics
        result["cos_sim_mean"] = float(statistics.mean(cos_sims))
        result["cos_sim_std"] = float(statistics.stdev(cos_sims) if len(cos_sims) > 1 else 0.0)

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            plt.figure(figsize=(6, 4))
            plt.hist(cos_sims, bins=50, edgecolor="white", linewidth=0.5)
            plt.xlabel("Cosine similarity")
            plt.ylabel("Count")
            plt.title("MIMIC-val paired image-text cosine distribution")
            plt.tight_layout()
            if output_dir:
                plt.savefig(os.path.join(output_dir, "mimic_cosine_hist.png"), dpi=150)
            plt.close()
            result["histogram_saved"] = True
        except ImportError:
            result["histogram_saved"] = False
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _print_section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 diagnostic probe")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="outputs/baseline_probe")
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--stsb-cache", default=None)
    parser.add_argument("--mimic-cache", default=None)
    parser.add_argument("--skip-stsb", action="store_true")
    parser.add_argument("--skip-mimic", action="store_true")
    parser.add_argument("--max-pairs", type=int, default=500)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    report: Dict = {"ckpt": args.ckpt, "device": args.device}

    _print_section("Loading checkpoint")
    model = load_encoder(args.ckpt, args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    _print_section("Probe 1: mLSTM gate health")
    gate_stats = probe_mlstm_gates(model, args.device)
    report["mlstm_gates"] = gate_stats
    for k, v in gate_stats.items():
        flag = " *** HIGH" if v["max"] > 15.0 else ""
        print(f"  {k}: max={v['max']:.3f}  mean={v['mean']:.3f}  "
              f"std={v['std']:.3f}  frac>15={v['frac_gt15']:.4f}{flag}")

    _print_section("Probe 2: Layer-wise hidden norms")
    norm_stats = probe_layer_norms(model, args.device)
    report["layer_norms"] = norm_stats
    for k, v in norm_stats.items():
        print(f"  {k}: {v:.4f}")

    _print_section("Probe 3: Doc-contamination")
    contam = probe_doc_contamination(model, args.device)
    report["doc_contamination"] = contam
    print(f"  Mean leak ratio : {contam['mean_leak_ratio']:.6f}")
    print(f"  Max  leak ratio : {contam['max_leak_ratio']:.6f}")
    print(f"  Verdict         : {contam['verdict']}")

    if not args.skip_stsb:
        _print_section("Probe 4: STS-B alignment / uniformity")
        stsb = probe_stsb(
            model, tokenizer, args.device,
            max_pairs=args.max_pairs,
            stsb_cache=args.stsb_cache,
        )
        report["stsb"] = stsb
        if "error" in stsb:
            print(f"  SKIPPED: {stsb['error']}")
        else:
            print(f"  Pairs           : {stsb['n_pairs']}  "
                  f"(positive: {stsb['n_positive_pairs']})")
            print(f"  Spearman rho    : {stsb['spearman_rho']:.4f}")
            print(f"  Alignment       : {stsb['alignment']}")
            print(f"  Uniformity      : {stsb['uniformity']}")
            print(f"  cos_sim mean/std: {stsb['cos_sim_mean']:.4f} / "
                  f"{stsb['cos_sim_std']:.4f}")

    if not args.skip_mimic and args.mimic_cache:
        _print_section("Probe 5: MIMIC-val cosine histogram")
        mimic = probe_mimic_cosine(
            model, tokenizer, args.device,
            mimic_cache=args.mimic_cache,
            max_pairs=args.max_pairs,
            output_dir=args.output_dir,
        )
        report["mimic"] = mimic
        if "error" in mimic:
            print(f"  SKIPPED: {mimic['error']}")
        else:
            print(f"  Pairs           : {mimic['n_pairs']}")
            if "cos_sim_mean" in mimic:
                print(f"  cos_sim mean/std: {mimic['cos_sim_mean']:.4f} / "
                      f"{mimic['cos_sim_std']:.4f}")
    elif not args.skip_mimic:
        print("\n  Probe 5 (MIMIC) skipped — pass --mimic-cache to enable")

    report_path = os.path.join(args.output_dir, "report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved → {report_path}")


if __name__ == "__main__":
    main()
