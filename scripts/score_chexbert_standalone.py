"""Phase 11B: standalone CheXbert F1 scorer, deliberately DECOUPLED from
evaluate_report_generation.py and the main project .venv.

WHY THIS FILE EXISTS (2026-08-29): f1chexbert==0.0.2 (last released 2023)
calls the legacy tokenizer.encode_plus(...) method internally, which was
removed in transformers>=5.0. The main .venv used for report generation and
training needs a recent transformers (GPT-2 tokenizer, BiomedCLIP text tower,
BioMedLM teacher) -- downgrading it cluster-wide just to satisfy this one old
scoring package is the wrong tradeoff. Instead, this script is meant to run
in a SEPARATE, isolated venv pinned to `transformers<5` (see
score_chexbert_h100.sh), and takes plain hyps.txt/refs.txt files as input
(produced by evaluate_report_generation.py's --dump-dir flag) rather than
importing anything from this repo. It intentionally imports NOTHING from
hybrid_xmamba -- pulling that in would drag along Triton/the custom
Mamba/mLSTM kernels, which have no reason to exist in a venv whose only job
is calling a pretrained BERT classifier.

The f1chexbert call and output dict shape here are a deliberate, minimal
duplication of evaluate_report_generation.py's compute_chexbert_metrics() --
kept in sync by hand since both wrap the same tiny, stable public API
(F1CheXbert()(hyps=[...], refs=[...])). If that API ever changes, fix both.

Usage (inside the isolated venv):
    python scripts/score_chexbert_standalone.py \\
        --hyp-file results/report_gen_full_n1433/hyps.txt \\
        --ref-file results/report_gen_full_n1433/refs.txt \\
        --output-dir results/report_gen_full_n1433
"""

import argparse
import json
from datetime import datetime
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """Separate from main() so tests can construct/parse args without
    importing f1chexbert (deferred to inside main(), see below)."""
    parser = argparse.ArgumentParser(description="Standalone CheXbert F1 scorer (isolated venv)")
    parser.add_argument("--hyp-file", type=str, required=True,
                        help="Path to hypotheses, one per line, aligned with --ref-file")
    parser.add_argument("--ref-file", type=str, required=True,
                        help="Path to references, one per line, aligned with --hyp-file")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="If given, also write chexbert_metrics.json here")
    return parser


def main():
    args = build_parser().parse_args()

    hyps = Path(args.hyp_file).read_text().splitlines()
    refs = Path(args.ref_file).read_text().splitlines()
    if len(hyps) != len(refs):
        raise SystemExit(
            "--hyp-file has {} lines but --ref-file has {}".format(len(hyps), len(refs)))

    # Mirrors evaluate_report_generation.py's compute_chexbert_metrics() —
    # see this file's module docstring for why it's duplicated, not imported.
    from f1chexbert import F1CheXbert
    labeler = F1CheXbert()
    accuracy, accuracy_per_sample, chexbert_all, chexbert_5 = labeler(hyps=hyps, refs=refs)
    results = {
        "accuracy": accuracy,
        "chexbert_14": chexbert_all,
        "chexbert_5": chexbert_5,
        "num_examples": len(hyps),
    }

    sep = "=" * 60
    print("\n" + sep)
    print("  CHEXBERT F1  (n={})".format(results["num_examples"]))
    print(sep)
    print("  CheXbert F1 (14-label) micro/macro : {:.4f} / {:.4f}".format(
        chexbert_all["micro avg"]["f1-score"], chexbert_all["macro avg"]["f1-score"]))
    print("  CheXbert F1 (5-label)  micro/macro : {:.4f} / {:.4f}".format(
        chexbert_5["micro avg"]["f1-score"], chexbert_5["macro avg"]["f1-score"]))

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        results["hyp_file"] = args.hyp_file
        results["ref_file"] = args.ref_file
        results["timestamp"] = datetime.now().isoformat()
        out_path = out_dir / "chexbert_metrics.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print("\n  Results saved to " + str(out_path))

    return results


if __name__ == "__main__":
    main()
