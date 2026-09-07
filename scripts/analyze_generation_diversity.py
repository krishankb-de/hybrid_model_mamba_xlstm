#!/usr/bin/env python
"""Boilerplate / duplicate-template analysis for generated reports.

H100_SCALING_PLAN.md Phase 14B (supervisor review 2026-09-07).

WHY THIS EXISTS
---------------
Phase 11C found that 1055/1433 (73.6%) of generated reports fell into one of 184
exact-duplicate template clusters. That number was measured on the PRE-Phase-13
checkpoint and never re-measured on 13D, the checkpoint every headline number in
`analysis/h100_scaling_results.md` comes from. It sat in
`final_verdict.open_items[0]` as an accepted limitation. The supervisor's point is
that it is the single biggest validity threat to the primary result: if the
generator is mostly emitting templates, "beats the retrieval-NN floor" is hollow,
because emitting a plausible templated report is exactly what the retrieval floor
does too.

THE CONTROLS ARE THE POINT
--------------------------
The original 73.6% was reported with NO CONTROL, which makes it uninterpretable on
its own. MIMIC-CXR reports are themselves heavily templated -- a generator at 73.6%
against a reference corpus at 60% is a completely different finding from one against
a reference corpus at 5%. And the retrieval-NN baseline emits REAL HUMAN REPORTS, so
whatever duplication IT shows is the rate a "perfect" non-generative system exhibits.
This script therefore scores all three corpora the same way and reports them
side by side. Run it with `--refs` and `--baseline` whenever they exist.

Deliberately dependency-free (stdlib only): it has to run on the cluster's eval venv
without pulling anything in. Self-BLEU here is a self-contained implementation used
only to compare corpora scored within the SAME run -- it is not sacrebleu-comparable
and must not be quoted as an absolute BLEU.

Usage: submit via the SLURM wrapper -- the aisc login node refuses ANY script
execution ("This command is not allowed on the login node!", Phase 7E), so this
is never invoked directly on the cluster:

    HYPS=results/report_gen_tower13d_test_split/hyps.txt \
    REFS=results/report_gen_tower13d_test_split/refs.txt \
    BASELINE=results/retrieval_floor_test_split/hyps.txt \
    OUTPUT=analysis/generation_diversity_13d.md \
      sbatch scripts/analyze_diversity_h100.sh

The bare `python scripts/analyze_generation_diversity.py --hyps ... --refs ...`
form works locally, and inside the wrapper.
"""

import argparse
import collections
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


def read_lines(path: str) -> List[str]:
    """Read one generation per line, dropping blanks and normalising whitespace."""
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        lines = [" ".join(line.strip().split()) for line in fh]
    return [line for line in lines if line]


def duplicate_clusters(texts: Sequence[str]) -> Tuple[int, int, float, List[int]]:
    """Exact-duplicate clustering — the same measurement Phase 11C reported.

    Returns:
        (num_clusters, num_texts_in_a_cluster, fraction_in_a_cluster, top_sizes)
        where a "cluster" is a distinct text occurring 2+ times.
    """
    counts = collections.Counter(texts)
    sizes = sorted((c for c in counts.values() if c > 1), reverse=True)
    in_cluster = sum(sizes)
    frac = in_cluster / len(texts) if texts else 0.0
    return len(sizes), in_cluster, frac, sizes[:5]


def distinct_n(texts: Sequence[str], n: int) -> float:
    """Ratio of unique n-grams to total n-grams across the corpus (higher = more diverse)."""
    total, unique = 0, set()
    for text in texts:
        tokens = text.split()
        for i in range(len(tokens) - n + 1):
            unique.add(tuple(tokens[i:i + n]))
            total += 1
    return len(unique) / total if total else 0.0


def type_token_ratio(texts: Sequence[str]) -> float:
    """Unique tokens / total tokens across the corpus."""
    total, vocab = 0, set()
    for text in texts:
        tokens = text.split()
        vocab.update(tokens)
        total += len(tokens)
    return len(vocab) / total if total else 0.0


def _ngram_counts(tokens: Sequence[str], n: int) -> Dict[Tuple[str, ...], int]:
    counts: Dict[Tuple[str, ...], int] = collections.defaultdict(int)
    for i in range(len(tokens) - n + 1):
        counts[tuple(tokens[i:i + n])] += 1
    return counts


def _sentence_bleu4(hyp: Sequence[str], refs: Sequence[Sequence[str]]) -> float:
    """Self-contained BLEU-4 with add-one smoothing on higher-order n-grams.

    Not sacrebleu-comparable. Used only to compare corpora scored identically within
    one run of this script.
    """
    if not hyp:
        return 0.0
    log_precisions = []
    for n in range(1, 5):
        hyp_counts = _ngram_counts(hyp, n)
        if not hyp_counts:
            return 0.0
        max_ref: Dict[Tuple[str, ...], int] = {}
        for ref in refs:
            for gram, count in _ngram_counts(ref, n).items():
                if count > max_ref.get(gram, 0):
                    max_ref[gram] = count
        overlap = sum(min(c, max_ref.get(g, 0)) for g, c in hyp_counts.items())
        total = sum(hyp_counts.values())
        # add-one smoothing for n>1 so a single zero order doesn't zero the score
        if n > 1:
            overlap, total = overlap + 1, total + 1
        if overlap == 0:
            return 0.0
        log_precisions.append(math.log(overlap / total))

    closest = min((abs(len(r) - len(hyp)), len(r)) for r in refs)[1]
    brevity = 1.0 if len(hyp) > closest else math.exp(1 - closest / max(len(hyp), 1))
    return brevity * math.exp(sum(log_precisions) / 4.0)


def self_bleu4(texts: Sequence[str], sample: int = 300, refs_per: int = 40,
               seed: int = 0) -> float:
    """Mean BLEU-4 of each text against a sample of the OTHERS. Higher = more repetitive.

    Exhaustive self-BLEU is O(n^2) (7M pairs at n=2663), so this samples: `sample`
    hypotheses each scored against `refs_per` random others. Sampling is seeded, and
    every corpus in a run gets the same treatment, which is what makes the
    generator/reference/baseline comparison valid.
    """
    if len(texts) < 2:
        return 0.0
    rng = random.Random(seed)
    tokenised = [t.split() for t in texts]
    indices = list(range(len(tokenised)))
    chosen = rng.sample(indices, min(sample, len(indices)))

    scores = []
    for i in chosen:
        pool = [j for j in rng.sample(indices, min(refs_per + 1, len(indices))) if j != i]
        if not pool:
            continue
        scores.append(_sentence_bleu4(tokenised[i], [tokenised[j] for j in pool[:refs_per]]))
    return sum(scores) / len(scores) if scores else 0.0


def analyse(name: str, texts: Sequence[str], seed: int = 0) -> Dict[str, object]:
    n_clusters, in_cluster, frac, top = duplicate_clusters(texts)
    return {
        "name": name,
        "n": len(texts),
        "unique": len(set(texts)),
        "duplicate_clusters": n_clusters,
        "in_duplicate_cluster": in_cluster,
        "pct_in_duplicate_cluster": 100.0 * frac,
        "largest_clusters": top,
        "distinct_1": distinct_n(texts, 1),
        "distinct_2": distinct_n(texts, 2),
        "distinct_3": distinct_n(texts, 3),
        "distinct_4": distinct_n(texts, 4),
        "type_token_ratio": type_token_ratio(texts),
        "self_bleu_4": self_bleu4(texts, seed=seed),
        "mean_tokens": (sum(len(t.split()) for t in texts) / len(texts)) if texts else 0.0,
    }


def render(results: List[Dict[str, object]], phase11c_pct: float = 73.6) -> str:
    """Render the comparison table plus an explicit, pre-registered verdict."""
    out = []
    out.append("# Generation diversity / boilerplate analysis\n")
    out.append("Produced by `scripts/analyze_generation_diversity.py` "
               "(H100_SCALING_PLAN.md Phase 14B).\n")
    out.append("\n**Why the controls matter.** MIMIC-CXR reports are themselves heavily "
               "templated, and the retrieval-NN baseline emits *real human reports*. A "
               "generator's duplication rate is only interpretable against those two "
               "reference points — the original 73.6% figure was reported without them.\n")

    out.append("\n## Headline\n")
    out.append("\n| corpus | n | unique | dup. clusters | % in a dup. cluster | largest clusters |")
    out.append("\n|---|---|---|---|---|---|")
    for r in results:
        out.append("\n| %s | %d | %d | %d | **%.1f%%** | %s |" % (
            r["name"], r["n"], r["unique"], r["duplicate_clusters"],
            r["pct_in_duplicate_cluster"],
            ", ".join(str(s) for s in r["largest_clusters"]) or "—"))

    out.append("\n\n## Lexical diversity (higher = more varied)\n")
    out.append("\n| corpus | distinct-1 | distinct-2 | distinct-3 | distinct-4 | TTR | self-BLEU-4 | mean tokens |")
    out.append("\n|---|---|---|---|---|---|---|---|")
    for r in results:
        out.append("\n| %s | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %.1f |" % (
            r["name"], r["distinct_1"], r["distinct_2"], r["distinct_3"], r["distinct_4"],
            r["type_token_ratio"], r["self_bleu_4"], r["mean_tokens"]))
    out.append("\n\nSelf-BLEU is a self-contained implementation (stdlib only, sampled) and is "
               "**not** sacrebleu-comparable — use it only to compare the rows above, which "
               "were all scored identically in one run.\n")

    gen = next((r for r in results if r["name"].startswith("generated")), results[0])
    controls = [r for r in results if r is not gen]
    out.append("\n## Verdict\n")
    out.append("\n- Phase 11C measured **%.1f%%** on the pre-Phase-13 checkpoint. "
               "This run measures **%.1f%%** — a change of **%+.1f pp**.\n"
               % (phase11c_pct, gen["pct_in_duplicate_cluster"],
                  gen["pct_in_duplicate_cluster"] - phase11c_pct))
    for c in controls:
        delta = gen["pct_in_duplicate_cluster"] - c["pct_in_duplicate_cluster"]
        out.append("- vs **%s** control: %.1f%% vs %.1f%% (**%+.1f pp**).\n"
                   % (c["name"], gen["pct_in_duplicate_cluster"],
                      c["pct_in_duplicate_cluster"], delta))

    # Pre-registered interpretation, declared 2026-09-07 before the measurement.
    above_all = all(gen["pct_in_duplicate_cluster"] > c["pct_in_duplicate_cluster"] + 10.0
                    for c in controls) if controls else False
    if gen["pct_in_duplicate_cluster"] >= 70.0 and above_all:
        out.append("\n**PRE-REGISTERED OUTCOME: QUALIFIER REQUIRED.** The generator is still "
                   ">=70% templated AND materially above every control. Per the bar declared "
                   "on 2026-09-07, §1's \"beats the retrieval floor\" headline must carry an "
                   "explicit qualifier **in the abstract**, not merely a bullet in §4.\n")
    elif controls and gen["pct_in_duplicate_cluster"] <= max(
            c["pct_in_duplicate_cluster"] for c in controls):
        out.append("\n**PRE-REGISTERED OUTCOME: POSITIVE FINDING.** The generator's duplication "
                   "rate is at or below its controls', i.e. no more templated than the corpus "
                   "it models. Per the bar declared on 2026-09-07 this should be stated as a "
                   "positive finding, not buried.\n")
    else:
        out.append("\n**PRE-REGISTERED OUTCOME: INTERMEDIATE.** Neither the qualifier trigger "
                   "(>=70% and materially above every control) nor the clean-positive trigger "
                   "(at or below the controls) fired. Report the numbers plainly in §4 with "
                   "the controls alongside, and do not round the interpretation in either "
                   "direction.\n")

    if not controls:
        out.append("\n⚠️ **No controls were supplied.** Re-run with `--refs` and `--baseline`; "
                   "a bare duplication rate is the exact reporting weakness Phase 14B exists "
                   "to fix.\n")
    return "".join(out)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hyps", required=True, help="generated reports, one per line")
    ap.add_argument("--refs", default=None, help="reference reports (control)")
    ap.add_argument("--baseline", default=None, help="retrieval-NN outputs (control)")
    ap.add_argument("--output", default=None, help="write the markdown report here")
    ap.add_argument("--phase11c-pct", type=float, default=73.6,
                    help="the historical rate to compare against (default: 73.6)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    results = [analyse("generated (model)", read_lines(args.hyps), seed=args.seed)]
    if args.refs:
        results.append(analyse("references (human)", read_lines(args.refs), seed=args.seed))
    if args.baseline:
        results.append(
            analyse("retrieval-NN (real reports)", read_lines(args.baseline), seed=args.seed))

    report = render(results, phase11c_pct=args.phase11c_pct)
    print(report)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(report, encoding="utf-8")
        print("wrote %s" % args.output, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
