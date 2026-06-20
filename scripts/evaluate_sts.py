"""STS Benchmark Evaluation for Biomedical Text Encoder.

Evaluates semantic textual similarity (Spearman ρ on cosine similarity) across:
  - BIOSSES  : 100 biomedical sentence pairs, scores 0–4
  - STS-B    : ~1500 general-domain sentence pairs, scores 0–5 (val / test)
  - MedSTS   : clinical sentence pairs from OHNLP MedSTS challenge (best-effort)

For each dataset, optionally runs PubMedBERT (teacher) on the same pairs and
records a side-by-side comparison.

Usage:
    # Single dataset
    python scripts/evaluate_sts.py \\
        --checkpoint outputs/hybrid_70m_stage1_kd_pubmedbert/checkpoints/last.ckpt \\
        --datasets biosses stsb \\
        --output-dir results/stage1_sts

    # All datasets + PubMedBERT baseline
    python scripts/evaluate_sts.py \\
        --checkpoint outputs/hybrid_70m_stage1_kd_pubmedbert/checkpoints/last.ckpt \\
        --datasets all \\
        --compare-pubmedbert \\
        --output-dir results/stage1_sts

Decision gate:
    BIOSSES Spearman >= 0.50  AND  STS-B >= 0.60  →  accept Stage 1 checkpoint.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from tqdm import tqdm
from scipy.stats import spearmanr

from hybrid_xmamba.models.configuration_hybrid import HybridConfig
from hybrid_xmamba.models.hybrid_lm import HybridTextEncoder


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def _strip_prefixes(state_dict: Dict) -> Dict:
    """Strip Lightning / torch.compile / DDP prefixes in safe order."""
    cleaned = {}
    for k, v in state_dict.items():
        k = k.replace("_orig_mod.", "", 1) if k.startswith("_orig_mod.") else k
        k = k.replace("model.", "", 1) if k.startswith("model.") else k
        k = k.replace("module.", "", 1) if k.startswith("module.") else k
        cleaned[k] = v
    return cleaned


def load_encoder(checkpoint_path: str, device: str = "cuda") -> HybridTextEncoder:
    print(f"Loading encoder from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    raw = ckpt.get("state_dict", ckpt)

    # Count layers
    import re
    num_layers = max(
        (int(m.group(1)) + 1 for k in raw for m in [re.search(r"layers\.(\d+)\.", k)] if m),
        default=8,
    )

    state = _strip_prefixes(raw)

    if not any(k.startswith("projection_head.") for k in state):
        raise RuntimeError(
            "Missing projection_head.* in checkpoint after prefix stripping. "
            f"First 10 keys: {list(state.keys())[:10]}"
        )

    dim = next(
        (int(v.shape[1]) for k, v in state.items() if "token_embedding.weight" in k),
        512,
    )
    # Auto-detect architecture from the checkpoint so v1 AND v2 load exact-match
    # (hardcoding [mamba,mamba,mlstm]+pre_rms mismapped the v2 backbone — same bug
    #  class fixed in evaluate_cxr_retrieval.py). mamba blocks carry mixer.A_log;
    # HybridNorm adds dt_norm/B_norm/C_norm (mamba) + v_norm (mlstm).
    layer_pattern = []
    for i in range(num_layers):
        mixer_keys = [k for k in state if f"lm.layers.{i}.mixer." in k]
        is_mamba = any("A_log" in k or "conv1d" in k for k in mixer_keys)
        layer_pattern.append("mamba" if is_mamba else "mlstm")
    norm_topology = "hybrid" if any(
        (".dt_norm." in k or ".v_norm." in k or ".B_norm." in k or ".C_norm." in k)
        for k in state
    ) else "pre_rms"
    print(f"  detected layer_pattern={layer_pattern}, norm_topology={norm_topology}")
    cfg = HybridConfig(
        dim=dim, num_layers=num_layers,
        layer_pattern=layer_pattern,
        norm_topology=norm_topology,
        vocab_size=50257, max_position_embeddings=1024,
        state_size=16, conv_size=4, expand_factor=2, head_dim=64,
        use_tfla=True, proj_factor=2, slstm_hidden_dim=dim, slstm_num_heads=4,
        norm_type="rms", use_mlp=True, mlp_ratio=4.0, dropout=0.0,
    )
    model = HybridTextEncoder(cfg, embed_dim=512)
    missing, unexpected = model.load_state_dict(state, strict=False)
    critical = [k for k in missing if k.startswith("lm.") or k.startswith("projection_head.")]
    if critical:
        raise RuntimeError(f"Critical keys missing after load: {critical[:5]}")
    if missing:
        print(f"  Non-critical missing keys: {len(missing)}")
    model.to(device).eval()
    print(f"  Encoder loaded: {sum(p.numel() for p in model.parameters()):,} params")
    return model


# ---------------------------------------------------------------------------
# Dataset loaders  (BIOSSES, STS-B, MedSTS)
# ---------------------------------------------------------------------------

def _load_biosses() -> Optional[Tuple[List[Tuple[str, str]], List[float]]]:
    """Try several HF dataset paths for BIOSSES. Returns (pairs, scores) or None."""
    candidates = [
        ("bigbio/biosses", {"name": "biosses_source", "split": "train"}),
        ("biosses", {"split": "train"}),
        ("nguyenthanhdo/biosses", {"split": "train"}),
    ]
    for dataset_id, kwargs in candidates:
        try:
            ds = load_dataset(dataset_id, **kwargs)
            pairs, scores = [], []
            for row in ds:
                s1 = str(row.get("text_1") or row.get("sentence1") or row.get("sentence_1") or "")
                s2 = str(row.get("text_2") or row.get("sentence2") or row.get("sentence_2") or "")
                sc = float(row.get("score") or row.get("label") or 0.0)
                if s1 and s2:
                    pairs.append((s1, s2))
                    scores.append(sc)
            if pairs:
                print(f"  BIOSSES loaded ({dataset_id}): {len(pairs)} pairs")
                return pairs, scores
        except Exception as exc:
            print(f"  BIOSSES {dataset_id} failed: {exc}")
    print("  BIOSSES: all sources failed — skipping")
    return None


def _load_stsb(split: str = "validation") -> Tuple[List[Tuple[str, str]], List[float]]:
    ds = load_dataset("glue", "stsb", split=split)
    pairs, scores = [], []
    for row in ds:
        s1, s2 = str(row.get("sentence1") or ""), str(row.get("sentence2") or "")
        if s1 and s2:
            pairs.append((s1, s2))
            scores.append(float(row.get("label") or 0.0))
    print(f"  STS-B ({split}): {len(pairs)} pairs")
    return pairs, scores


def _load_medstsrel() -> Optional[Tuple[List[Tuple[str, str]], List[float]]]:
    """Try to load MedSTS / MedSTSRel from HuggingFace. Best-effort."""
    candidates = [
        ("bigbio/medstsrel", {"name": "medstsrel_source", "split": "train"}),
        ("bigbio/med_sts", {"name": "med_sts_source", "split": "train"}),
        ("medstsrel", {"split": "train"}),
    ]
    for dataset_id, kwargs in candidates:
        try:
            ds = load_dataset(dataset_id, **kwargs)
            pairs, scores = [], []
            for row in ds:
                s1 = str(row.get("text_1") or row.get("sentence1") or row.get("sent1") or "")
                s2 = str(row.get("text_2") or row.get("sentence2") or row.get("sent2") or "")
                sc = float(row.get("score") or row.get("label") or row.get("similarity") or 0.0)
                if s1 and s2:
                    pairs.append((s1, s2))
                    scores.append(sc)
            if pairs:
                print(f"  MedSTS loaded ({dataset_id}): {len(pairs)} pairs")
                return pairs, scores
        except Exception as exc:
            print(f"  MedSTS {dataset_id} failed: {exc}")
    print("  MedSTS: all sources failed — skipping")
    return None


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_sentences(
    model: HybridTextEncoder,
    tokenizer: AutoTokenizer,
    sentences: List[str],
    batch_size: int = 32,
    max_length: int = 512,
    device: str = "cuda",
) -> torch.Tensor:
    all_embs = []
    for i in tqdm(range(0, len(sentences), batch_size), desc="  Encoding", leave=False):
        enc = tokenizer(
            sentences[i : i + batch_size],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        all_embs.append(model.encode(ids, attention_mask=mask).cpu())
    return torch.cat(all_embs, dim=0)


@torch.no_grad()
def encode_sentences_pubmedbert(
    teacher: AutoModel,
    teacher_tokenizer: AutoTokenizer,
    sentences: List[str],
    batch_size: int = 32,
    max_length: int = 512,
    device: str = "cuda",
) -> torch.Tensor:
    """Encode with PubMedBERT (mean pool CLS token over non-padding)."""
    all_embs = []
    for i in tqdm(range(0, len(sentences), batch_size), desc="  Encoding (PubMedBERT)", leave=False):
        enc = teacher_tokenizer(
            sentences[i : i + batch_size],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        out = teacher(input_ids=ids, attention_mask=mask)
        # Mean pool over non-padding positions
        h = out.last_hidden_state  # (B, L, D)
        m = mask.unsqueeze(-1).to(h.dtype)
        emb = (h * m).sum(1) / m.sum(1).clamp(min=1)
        all_embs.append(F.normalize(emb, dim=-1).cpu())
    return torch.cat(all_embs, dim=0)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run_sts_eval(
    model: HybridTextEncoder,
    tokenizer: AutoTokenizer,
    pairs: List[Tuple[str, str]],
    gold_scores: List[float],
    batch_size: int,
    max_length: int,
    device: str,
    label: str = "hybrid",
) -> float:
    s1 = [p[0] for p in pairs]
    s2 = [p[1] for p in pairs]
    z1 = encode_sentences(model, tokenizer, s1, batch_size, max_length, device)
    z2 = encode_sentences(model, tokenizer, s2, batch_size, max_length, device)
    pred = (z1 * z2).sum(dim=-1).tolist()
    rho, _ = spearmanr(gold_scores, pred)
    return float(rho)


def run_sts_eval_pubmedbert(
    teacher: AutoModel,
    teacher_tok: AutoTokenizer,
    pairs: List[Tuple[str, str]],
    gold_scores: List[float],
    batch_size: int,
    max_length: int,
    device: str,
) -> float:
    s1 = [p[0] for p in pairs]
    s2 = [p[1] for p in pairs]
    z1 = encode_sentences_pubmedbert(teacher, teacher_tok, s1, batch_size, max_length, device)
    z2 = encode_sentences_pubmedbert(teacher, teacher_tok, s2, batch_size, max_length, device)
    pred = (z1 * z2).sum(dim=-1).tolist()
    rho, _ = spearmanr(gold_scores, pred)
    return float(rho)


# ---------------------------------------------------------------------------
# Results recording
# ---------------------------------------------------------------------------

def write_metrics_md(
    results: Dict,
    output_dir: Path,
    checkpoint_path: str,
) -> None:
    """Append STS results to results/stage1_metrics.md."""
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / "stage1_metrics.md"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        f"\n## STS Results — {ts}\n",
        f"Checkpoint: `{checkpoint_path}`\n",
        "\n| Dataset | Hybrid Spearman ρ | PubMedBERT ρ | Δ |\n",
        "|---------|-------------------|--------------|---|\n",
    ]
    for ds_name, row in results.items():
        h = row.get("hybrid", float("nan"))
        p = row.get("pubmedbert", None)
        if p is not None:
            delta = h - p
            lines.append(f"| {ds_name} | {h:.4f} | {p:.4f} | {delta:+.4f} |\n")
        else:
            lines.append(f"| {ds_name} | {h:.4f} | — | — |\n")

    gate_biosses = results.get("BIOSSES", {}).get("hybrid", 0.0)
    gate_stsb = results.get("STS-B (val)", {}).get("hybrid", results.get("STS-B (test)", {}).get("hybrid", 0.0))
    gate_pass = gate_biosses >= 0.50 and gate_stsb >= 0.60
    lines.append(f"\n**Decision gate** (BIOSSES ≥ 0.50 AND STS-B ≥ 0.60): {'PASS ✓' if gate_pass else 'FAIL ✗'}\n")

    mode = "a" if md_path.exists() else "w"
    with open(md_path, mode) as f:
        if mode == "w":
            f.write("# Stage 1 Evaluation Metrics\n")
        f.writelines(lines)
    print(f"\nResults appended to {md_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1 STS evaluation")
    parser.add_argument("--checkpoint", required=True, help="Path to Stage 1 .ckpt")
    parser.add_argument(
        "--datasets", nargs="+", default=["biosses", "stsb"],
        choices=["biosses", "stsb", "medstsrel", "all"],
        help="Datasets to evaluate (use 'all' for all)"
    )
    parser.add_argument("--compare-pubmedbert", action="store_true",
                        help="Run PubMedBERT on same pairs and compare")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--output-dir", type=str, default="results/stage1_sts")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    print(f"Device: {device}\n")

    # Resolve dataset list
    run_all = "all" in args.datasets
    do_biosses = run_all or "biosses" in args.datasets
    do_stsb = run_all or "stsb" in args.datasets
    do_medstsrel = run_all or "medstsrel" in args.datasets

    # Load hybrid model
    model = load_encoder(args.checkpoint, device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Optionally load PubMedBERT baseline
    teacher, teacher_tok = None, None
    if args.compare_pubmedbert:
        teacher_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
        print(f"Loading PubMedBERT teacher: {teacher_name}...")
        teacher = AutoModel.from_pretrained(teacher_name, torch_dtype=torch.bfloat16)
        teacher.to(device).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        teacher_tok = AutoTokenizer.from_pretrained(teacher_name)
        print("  Teacher loaded.")

    results: Dict = {}

    # BIOSSES
    if do_biosses:
        print("\n=== BIOSSES ===")
        data = _load_biosses()
        if data is not None:
            pairs, scores = data
            rho = run_sts_eval(model, tokenizer, pairs, scores, args.batch_size, args.max_length, device)
            print(f"  Hybrid Spearman ρ = {rho:.4f}")
            row: Dict = {"hybrid": rho, "n": len(pairs)}
            if teacher is not None:
                rho_b = run_sts_eval_pubmedbert(teacher, teacher_tok, pairs, scores, args.batch_size, 512, device)
                print(f"  PubMedBERT ρ     = {rho_b:.4f}  (Δ = {rho - rho_b:+.4f})")
                row["pubmedbert"] = rho_b
            results["BIOSSES"] = row

    # STS-B validation
    if do_stsb:
        print("\n=== STS-B (validation) ===")
        pairs, scores = _load_stsb("validation")
        rho = run_sts_eval(model, tokenizer, pairs, scores, args.batch_size, args.max_length, device)
        print(f"  Hybrid Spearman ρ = {rho:.4f}")
        row = {"hybrid": rho, "n": len(pairs)}
        if teacher is not None:
            rho_b = run_sts_eval_pubmedbert(teacher, teacher_tok, pairs, scores, args.batch_size, 512, device)
            print(f"  PubMedBERT ρ     = {rho_b:.4f}  (Δ = {rho - rho_b:+.4f})")
            row["pubmedbert"] = rho_b
        results["STS-B (val)"] = row

    # MedSTS
    if do_medstsrel:
        print("\n=== MedSTS ===")
        data = _load_medstsrel()
        if data is not None:
            pairs, scores = data
            rho = run_sts_eval(model, tokenizer, pairs, scores, args.batch_size, args.max_length, device)
            print(f"  Hybrid Spearman ρ = {rho:.4f}")
            row = {"hybrid": rho, "n": len(pairs)}
            if teacher is not None:
                rho_b = run_sts_eval_pubmedbert(teacher, teacher_tok, pairs, scores, args.batch_size, 512, device)
                print(f"  PubMedBERT ρ     = {rho_b:.4f}  (Δ = {rho - rho_b:+.4f})")
                row["pubmedbert"] = rho_b
            results["MedSTS"] = row

    # Summary
    print("\n" + "=" * 60)
    print("STS EVALUATION SUMMARY")
    print("=" * 60)
    for ds_name, row in results.items():
        h = row["hybrid"]
        p = row.get("pubmedbert")
        suffix = f"  vs PubMedBERT={p:.4f} (Δ={h - p:+.4f})" if p is not None else ""
        print(f"  {ds_name:<20}: {h:.4f}{suffix}")

    # Decision gate
    biosses_rho = results.get("BIOSSES", {}).get("hybrid", 0.0)
    stsb_rho = results.get("STS-B (val)", {}).get("hybrid", 0.0)
    gate = biosses_rho >= 0.50 and stsb_rho >= 0.60
    print(f"\nDecision gate (BIOSSES ≥ 0.50, STS-B ≥ 0.60): {'PASS ✓' if gate else 'FAIL ✗'}")
    if not gate:
        print("  → Iterate Phase 2: lower τ / raise λ_max / more steps")

    # Save JSON + markdown
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "sts_results.json"
    with open(json_path, "w") as f:
        json.dump({
            "checkpoint": args.checkpoint,
            "timestamp": datetime.now().isoformat(),
            "results": results,
            "gate_pass": gate,
        }, f, indent=2)
    print(f"\nJSON saved: {json_path}")

    results_dir = PROJECT_ROOT / "results"
    write_metrics_md(results, results_dir, args.checkpoint)


if __name__ == "__main__":
    main()
