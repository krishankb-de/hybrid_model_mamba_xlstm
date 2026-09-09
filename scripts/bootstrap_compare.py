#!/usr/bin/env python
"""Paired bootstrap comparison of two report generators.

H100_SCALING_PLAN.md Phase 14A-6.

WHY THIS EXISTS
---------------
Phase 14A's pre-registered success bar is phrased as "the Transformer does NOT
beat the hybrid by more than the 95% bootstrap CI on the difference". Without an
interval, "matches" is not a testable statement -- and the actual margins are
small enough that it matters: on the official test split the hybrid leads
CheXbert-14-micro 0.4736 vs 0.4590 while the Transformer leads ROUGE-L 0.1936 vs
0.1899, a gap of 0.0037. Declaring either a win without a CI would be reading
noise.

The bootstrap is PAIRED: both systems generated for the same 2663 studies, so
each resample draws the same study indices for both and the difference is
computed within the resample. That controls for "some studies are just harder"
variance, which an unpaired comparison would leave in and which would badly
inflate the interval at this effect size.

Metric functions are IMPORTED from evaluate_report_generation.py rather than
reimplemented, so the point estimates here reconcile exactly with the numbers
already reported. ROUGE-L is a per-sample mean; BLEU is corpus-level and is
therefore recomputed on each resample, which is the correct (if slower) thing.

CheXbert F1 needs per-sample label matrices, which the labeler produces but the
scorer did not previously save. `score_chexbert_standalone.py` now always writes
`chexbert_labels.json` alongside `chexbert_metrics.json`; pass those via
--labels-a/--labels-b. Omit them and only the text metrics are compared, which
is still enough to settle the pre-registered ROUGE-L question.

Usage (via the SLURM wrapper -- the aisc login node refuses direct execution):
    A=results/report_gen_tower13d_test_split \
    B=results/report_gen_transformer_test_split \
    NAME_A=hybrid_13D NAME_B=transformer \
    OUTPUT=analysis/bootstrap_hybrid_vs_transformer.md \
      sbatch scripts/bootstrap_compare_h100.sh
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from evaluate_report_generation import corpus_bleu, rouge_l_score  # noqa: E402


def read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return [line.rstrip("\n") for line in fh]


def per_sample_rouge_l(hyps: Sequence[str], refs: Sequence[str]) -> List[float]:
    """ROUGE-L is a per-sample mean, so the per-sample scores can be cached once."""
    return [rouge_l_score(h.split(), r.split()) for h, r in zip(hyps, refs)]


def _f1_from_counts(tp: float, fp: float, fn: float) -> float:
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom > 0 else 0.0


def chexbert_f1(y_true: Sequence[Sequence[int]], y_pred: Sequence[Sequence[int]],
                average: str, cols: Optional[Sequence[int]] = None) -> float:
    """micro / macro F1 over a binary label matrix, matching sklearn's convention.

    Reimplemented (not imported) because the scorer runs in an isolated venv that
    this script cannot import from -- but it is verified against sklearn in
    tests/test_willi_parity.py so the two cannot silently diverge.
    """
    n_labels = len(y_true[0]) if y_true else 0
    idx = list(cols) if cols is not None else list(range(n_labels))

    if average == "micro":
        tp = fp = fn = 0.0
        for t_row, p_row in zip(y_true, y_pred):
            for j in idx:
                tp += t_row[j] and p_row[j]
                fp += (not t_row[j]) and p_row[j]
                fn += t_row[j] and (not p_row[j])
        return _f1_from_counts(tp, fp, fn)

    scores = []
    for j in idx:
        tp = fp = fn = 0.0
        for t_row, p_row in zip(y_true, y_pred):
            tp += t_row[j] and p_row[j]
            fp += (not t_row[j]) and p_row[j]
            fn += t_row[j] and (not p_row[j])
        scores.append(_f1_from_counts(tp, fp, fn))
    return sum(scores) / len(scores) if scores else 0.0


def evaluate_subset(idx: Sequence[int], cache: Dict) -> Dict[str, float]:
    """All metrics for one system on one resample of study indices."""
    out = {"rouge_l": sum(cache["rouge"][i] for i in idx) / len(idx)}

    hyp_toks = [cache["hyp_toks"][i] for i in idx]
    ref_toks = [cache["ref_toks"][i] for i in idx]
    out["bleu_1"] = corpus_bleu(hyp_toks, ref_toks, max_n=1)
    out["bleu_4"] = corpus_bleu(hyp_toks, ref_toks, max_n=4)

    if cache.get("y_true") is not None:
        y_true = [cache["y_true"][i] for i in idx]
        y_pred = [cache["y_pred"][i] for i in idx]
        five = cache.get("five_idx")
        out["chexbert_14_micro"] = chexbert_f1(y_true, y_pred, "micro")
        out["chexbert_14_macro"] = chexbert_f1(y_true, y_pred, "macro")
        if five:
            out["chexbert_5_micro"] = chexbert_f1(y_true, y_pred, "micro", five)
            out["chexbert_5_macro"] = chexbert_f1(y_true, y_pred, "macro", five)
        # Exact-match label-set accuracy. NOTE THE SUBSET, it is not cosmetic:
        # f1chexbert's reported "accuracy" is accuracy_score over the FIVE-label
        # subset (verified in F1CheXbert.forward: accuracy_score(refs_chexbert_5,
        # hyps_chexbert_5)), NOT all fourteen. An earlier revision here computed
        # the 14-label version and called it "exact_match_accuracy", which read as
        # the same quantity as the 0.2163/0.2306 in the headline table while
        # actually being 0.0349/0.0469. Both are emitted now, named for their
        # subset, so the two can never be conflated again.
        out["exact_match_accuracy_14"] = sum(
            1 for t, p in zip(y_true, y_pred) if list(t) == list(p)
        ) / len(y_true)
        if five:
            out["exact_match_accuracy_5"] = sum(
                1 for t, p in zip(y_true, y_pred)
                if [t[j] for j in five] == [p[j] for j in five]
            ) / len(y_true)
    return out


def build_cache(hyps, refs, labels_path: Optional[str]) -> Dict:
    cache = {
        "rouge": per_sample_rouge_l(hyps, refs),
        "hyp_toks": [h.split() for h in hyps],
        "ref_toks": [r.split() for r in refs],
        "y_true": None,
    }
    if labels_path:
        with open(labels_path) as fh:
            payload = json.load(fh)
        cache["y_true"] = payload["y_true"]
        cache["y_pred"] = payload["y_pred"]
        cache["five_idx"] = payload.get("five_label_indices")
    return cache


def paired_bootstrap(cache_a: Dict, cache_b: Dict, n_samples: int, seed: int,
                     alpha: float = 0.05) -> Tuple[Dict, Dict]:
    """Resample study indices ONCE per draw and score both systems on them.

    Pairing is the whole point: the same studies are hard for both systems, so
    differencing within a resample removes that shared variance instead of
    letting it widen both intervals independently.
    """
    n = len(cache_a["rouge"])
    rng = random.Random(seed)

    point_a = evaluate_subset(range(n), cache_a)
    point_b = evaluate_subset(range(n), cache_b)
    metrics = [m for m in point_a if m in point_b]

    diffs: Dict[str, List[float]] = {m: [] for m in metrics}
    for _ in range(n_samples):
        idx = [rng.randrange(n) for _ in range(n)]
        sub_a = evaluate_subset(idx, cache_a)
        sub_b = evaluate_subset(idx, cache_b)
        for m in metrics:
            diffs[m].append(sub_a[m] - sub_b[m])

    results = {}
    lo_q, hi_q = alpha / 2, 1 - alpha / 2
    for m in metrics:
        d = sorted(diffs[m])
        lo = d[max(0, int(lo_q * len(d)) - 1)]
        hi = d[min(len(d) - 1, int(hi_q * len(d)))]
        # How often a resample reverses the observed direction. Reported
        # alongside the CI because at these effect sizes it is the more
        # intuitive number: "in X% of resamples the other system won".
        observed = point_a[m] - point_b[m]
        if observed > 0:
            n_opposite = sum(1 for x in d if x <= 0)
        elif observed < 0:
            n_opposite = sum(1 for x in d if x >= 0)
        else:
            n_opposite = len(d)
        results[m] = {
            "a": point_a[m], "b": point_b[m], "diff": observed,
            "ci_low": lo, "ci_high": hi,
            "significant": (lo > 0) or (hi < 0),
            "frac_sign_flipped": n_opposite / len(d),
        }
    return results, {"n": n, "bootstrap_samples": n_samples, "seed": seed}


def render(results: Dict, meta: Dict, name_a: str, name_b: str) -> str:
    out = ["# Paired bootstrap: %s vs %s\n" % (name_a, name_b)]
    out.append("\nH100_SCALING_PLAN.md Phase 14A-6. n=%d studies, %d paired bootstrap "
               "resamples, seed %d. Both systems generated for the SAME studies, so each "
               "resample draws one set of indices and scores both on it — the difference "
               "is computed within the resample.\n"
               % (meta["n"], meta["bootstrap_samples"], meta["seed"]))
    out.append("\nA positive difference favours **%s**. A result is called only when the "
               "95%% CI excludes zero.\n" % name_a)

    out.append("\n| metric | %s | %s | diff | 95%% CI | verdict |\n" % (name_a, name_b))
    out.append("|---|---|---|---|---|---|\n")
    for m, r in results.items():
        if r["significant"]:
            verdict = "**%s wins**" % (name_a if r["diff"] > 0 else name_b)
        else:
            verdict = "tie (CI spans 0)"
        out.append("| %s | %.4f | %.4f | %+.4f | [%+.4f, %+.4f] | %s |\n"
                   % (m, r["a"], r["b"], r["diff"], r["ci_low"], r["ci_high"], verdict))

    wins_a = [m for m, r in results.items() if r["significant"] and r["diff"] > 0]
    wins_b = [m for m, r in results.items() if r["significant"] and r["diff"] < 0]
    ties = [m for m, r in results.items() if not r["significant"]]
    out.append("\n## Summary\n")
    out.append("\n- **%s wins (CI excludes 0):** %s\n" % (name_a, ", ".join(wins_a) or "none"))
    out.append("- **%s wins (CI excludes 0):** %s\n" % (name_b, ", ".join(wins_b) or "none"))
    out.append("- **Ties (CI spans 0):** %s\n" % (", ".join(ties) or "none"))
    return "".join(out)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hyps-a", required=True)
    ap.add_argument("--hyps-b", required=True)
    ap.add_argument("--refs", required=True)
    ap.add_argument("--labels-a", default=None,
                    help="chexbert_labels.json from score_chexbert_standalone.py --dump-labels")
    ap.add_argument("--labels-b", default=None)
    ap.add_argument("--name-a", default="A")
    ap.add_argument("--name-b", default="B")
    ap.add_argument("--bootstrap-samples", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default=None)
    args = ap.parse_args(argv)

    hyps_a, hyps_b, refs = (read_lines(args.hyps_a), read_lines(args.hyps_b),
                            read_lines(args.refs))
    if not (len(hyps_a) == len(hyps_b) == len(refs)):
        raise SystemExit(
            "Pairing requires equal lengths: %d / %d / %d. Both systems must have "
            "generated for the SAME studies in the SAME order."
            % (len(hyps_a), len(hyps_b), len(refs)))

    cache_a = build_cache(hyps_a, refs, args.labels_a)
    cache_b = build_cache(hyps_b, refs, args.labels_b)
    results, meta = paired_bootstrap(cache_a, cache_b, args.bootstrap_samples, args.seed)

    report = render(results, meta, args.name_a, args.name_b)
    print(report)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(report, encoding="utf-8")
        print("wrote %s" % args.output, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
