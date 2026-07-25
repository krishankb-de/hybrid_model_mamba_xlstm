"""Phase 6C-3 — duplicate / false-negative audit for MIMIC-CXR. CPU only, no GPU.

Answers two questions that decide whether the Phase 6D-3 multi-positive mask is
worth building, and whether the headline R@10 is partly a measurement artifact:

  (a) EVAL GALLERY — how much of the strict-index R@K ceiling do duplicate
      reports eat? ``evaluate_cxr_retrieval.py`` uses ``gt = np.arange(N)``, so
      when m gallery reports are textually identical the tie-break is arbitrary
      and even a perfect model scores ~min(1, k/m) on those rows.

  (b) TRAIN SPLIT — what fraction of each in-batch InfoNCE negative is actually
      a false negative? ``_nt_xent_loss`` (lightning_module.py:540) uses hard
      one-hot ``arange(B)`` targets, so every off-diagonal pair is pushed apart
      unconditionally — including pairs whose reports are word-for-word equal.
      Reported per batch size {32, 64, 128, 256} both analytically and by
      simulation over the actual split.

Deliberately text-only: the image column is dropped before iteration so nothing
decodes a single PNG. Runs in seconds on a login node.

Usage:
    python scripts/audit_mimic_duplicates.py \
        --cache-dir /sc/home/$USER/dataset/mimic_cxr_cache \
        --output-dir results/phase6c
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from datasets import load_dataset

from scripts.evaluate_cxr_retrieval import (
    MIMIC_REPO,
    group_ids_from_texts,
    normalize_report_text,
)

TEXT_COLUMNS = ["findings", "impression"]


def _load_texts(split: str, cache_dir: str) -> List[str]:
    """Load one split's report texts without ever decoding an image."""
    ds = load_dataset(MIMIC_REPO, split=split, cache_dir=cache_dir)
    keep = [c for c in TEXT_COLUMNS if c in ds.column_names]
    if not keep:
        raise RuntimeError(
            "Neither 'findings' nor 'impression' in columns {}".format(ds.column_names)
        )
    drop = [c for c in ds.column_names if c not in keep]
    if drop:
        ds = ds.remove_columns(drop)

    findings = ds["findings"] if "findings" in keep else [""] * len(ds)
    impression = ds["impression"] if "impression" in keep else [""] * len(ds)

    texts = []
    for f, i in zip(findings, impression):
        f = (f or "").strip()
        i = (i or "").strip()
        # Reproduce the training text exactly (train_contrastive.py:378-382).
        texts.append("Findings: {} Impression: {}".format(f, i).strip() if (f or i) else "")
    return texts


def _group_stats(texts: List[str]) -> Dict:
    groups = group_ids_from_texts(texts)
    n = len(texts)
    uniq, counts = np.unique(groups, return_counts=True)
    sizes = counts[np.searchsorted(uniq, groups)]      # per-position group size

    top = Counter()
    for g, c in zip(uniq.tolist(), counts.tolist()):
        top[g] = c
    largest = top.most_common(10)

    # Representative text for the largest groups (truncated).
    first_idx = {}
    for i, g in enumerate(groups.tolist()):
        if g not in first_idx:
            first_idx[g] = i
    largest_examples = [
        {
            "size": int(c),
            "share_of_split": float(c) / max(n, 1),
            "text": texts[first_idx[g]][:160],
        }
        for g, c in largest
    ]

    empty = int(sum(1 for t in texts if not normalize_report_text(t)))

    return {
        "n": int(n),
        "distinct_reports": int(len(uniq)),
        "duplicated_positions": int(n - len(uniq)),
        "duplicate_fraction": float(n - len(uniq)) / max(n, 1),
        "max_group_size": int(counts.max()) if n else 0,
        "mean_group_size": float(counts.mean()) if n else 0.0,
        "empty_reports": empty,
        "largest_groups": largest_examples,
        "_groups": groups,
        "_sizes": sizes,
        "_counts": counts,
    }


def _oracle_ceiling(sizes: np.ndarray, ks=(1, 5, 10)) -> Dict[str, float]:
    """Strict-index R@k for a PERFECT model, given arbitrary tie-breaking.

    A query whose report is shared by m gallery items ranks all m equally; the
    paired index lands in the top k with probability ~min(1, k/m).
    """
    return {
        "R@{}".format(k): float(np.minimum(1.0, float(k) / sizes).mean())
        for k in ks
    }


def _false_negative_rates(counts: np.ndarray, n: int, batch_sizes: List[int],
                          groups: np.ndarray, n_sim: int, seed: int) -> Dict[str, Dict]:
    """Analytic + simulated false-negative rate per in-batch InfoNCE batch size.

    Analytic: P(two distinct random samples share a group)
              = sum_g n_g(n_g - 1) / (N(N - 1)).
    Simulated: draw batches without replacement from the real split and count
               off-diagonal same-group pairs.
    """
    pair_p = float((counts * (counts - 1)).sum()) / max(n * (n - 1), 1)
    rng = np.random.default_rng(seed)
    out = {}
    for b in batch_sizes:
        if b > n:
            continue
        expected_pairs = pair_p * b * (b - 1)
        sim_pairs = []
        for _ in range(n_sim):
            idx = rng.choice(n, size=b, replace=False)
            g = groups[idx]
            _, c = np.unique(g, return_counts=True)
            sim_pairs.append(float((c * (c - 1)).sum()))
        sim_pairs_arr = np.asarray(sim_pairs)
        out[str(b)] = {
            "analytic_false_neg_pairs_per_batch": expected_pairs,
            "analytic_false_neg_rate_of_matrix": expected_pairs / float(b * b),
            "simulated_false_neg_pairs_per_batch_mean": float(sim_pairs_arr.mean()),
            "simulated_false_neg_pairs_per_batch_p95": float(np.percentile(sim_pairs_arr, 95)),
            "simulated_prob_batch_has_any_false_neg": float((sim_pairs_arr > 0).mean()),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", default=os.environ.get(
        "MIMIC_CACHE_DIR", str(Path.home() / "dataset" / "mimic_cxr_cache")))
    ap.add_argument("--output-dir", default="results/phase6c")
    ap.add_argument("--batch-sizes", default="32,64,128,256")
    ap.add_argument("--n-sim", type=int, default=2000,
                    help="Simulated batches per batch size.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x.strip()]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("Phase 6C-3 — MIMIC duplicate / false-negative audit (CPU, text only)")
    print("=" * 78)

    report = {"timestamp": datetime.now().isoformat(), "repo": MIMIC_REPO, "splits": {}}

    for label, split in (("train", "train[:90%]"), ("eval_gallery", "train[90%:]")):
        print("\n--- {} ({}) ---".format(label, split))
        texts = _load_texts(split, args.cache_dir)
        stats = _group_stats(texts)
        groups = stats.pop("_groups")
        sizes = stats.pop("_sizes")
        counts = stats.pop("_counts")

        print("  pairs                : {}".format(stats["n"]))
        print("  distinct reports     : {}".format(stats["distinct_reports"]))
        print("  duplicated positions : {} ({:.1f}%)".format(
            stats["duplicated_positions"], 100 * stats["duplicate_fraction"]))
        print("  largest group        : {} copies".format(stats["max_group_size"]))
        print("  empty reports        : {}".format(stats["empty_reports"]))
        print("  top duplicate groups :")
        for g in stats["largest_groups"][:5]:
            print("    {:5d} copies ({:4.1f}%)  {!r}".format(
                g["size"], 100 * g["share_of_split"], g["text"][:96]))

        if label == "eval_gallery":
            oracle = _oracle_ceiling(sizes)
            stats["oracle_ceiling_strict_index"] = oracle
            print("\n  ORACLE CEILING (perfect model, strict-index gt, arbitrary ties):")
            print("    R@1 {R@1:.4f} | R@5 {R@5:.4f} | R@10 {R@10:.4f}".format(**oracle))
            print("    Student authoritative i2t R@10 = 0.1113 — headroom to the")
            print("    oracle is {:.1f}pp.".format((oracle["R@10"] - 0.1113) * 100))
        else:
            fn = _false_negative_rates(
                counts, stats["n"], batch_sizes, groups, args.n_sim, args.seed)
            stats["false_negative_rates"] = fn
            print("\n  IN-BATCH FALSE NEGATIVES (hard one-hot arange(B) targets):")
            print("    {:>5s}  {:>12s}  {:>12s}  {:>14s}".format(
                "bs", "E[pairs]", "sim mean", "P(any in batch)"))
            for b in batch_sizes:
                if str(b) not in fn:
                    continue
                r = fn[str(b)]
                print("    {:>5d}  {:>12.2f}  {:>12.2f}  {:>14.3f}".format(
                    b,
                    r["analytic_false_neg_pairs_per_batch"],
                    r["simulated_false_neg_pairs_per_batch_mean"],
                    r["simulated_prob_batch_has_any_false_neg"],
                ))

        report["splits"][label] = stats

    # --- Interpretation ---------------------------------------------------
    ev = report["splits"]["eval_gallery"]
    tr = report["splits"]["train"]
    oracle10 = ev["oracle_ceiling_strict_index"]["R@10"]
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    if oracle10 < 0.60:
        print("  Eval gallery: duplicates cost REAL headroom (oracle R@10 {:.1%}).".format(oracle10))
        print("  Report dedup-aware R@10 alongside the strict number in the writeup.")
    else:
        print("  Eval gallery: oracle R@10 {:.1%} — duplicates do NOT explain the".format(oracle10))
        print("  11% plateau. The model is genuinely far from ceiling.")

    fn64 = tr.get("false_negative_rates", {}).get("64")
    if fn64 is not None:
        p_any = fn64["simulated_prob_batch_has_any_false_neg"]
        mean_pairs = fn64["simulated_false_neg_pairs_per_batch_mean"]
        if p_any > 0.25 or mean_pairs > 1.0:
            print("  Train split: at bs=64, {:.0%} of batches contain at least one".format(p_any))
            print("  false negative ({:.2f} pairs/batch on average). The 6D-3".format(mean_pairs))
            print("  multi-positive mask has real signal to fix — BUILD IT.")
        else:
            print("  Train split: at bs=64 only {:.0%} of batches contain a false".format(p_any))
            print("  negative ({:.2f} pairs/batch). Exact-match multi-positives will".format(mean_pairs))
            print("  do little; 6D-3 should lean on SigLIP, or move to CheXbert-label")
            print("  soft targets if the mask is to be worth it.")
    print("=" * 78)

    out_path = out_dir / "phase6c_duplicate_audit.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print("\nWrote {}".format(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
