#!/usr/bin/env python
"""Post-hoc repair of a generated-report dump (MAMBA3_PLAN_V2.md V5-D).

WHY THIS EXISTS
---------------
The decoding harness has no stop condition. `beam_search_decode`
(scripts/evaluate_report_generation.py) runs exactly `max_new_tokens=100`
iterations for every study, and the model was never trained to stop: the
report-generation trainer sets `pad_token = eos_token` and then masks every
pad position to -100, so no end-of-report token appears in any supervised
target. Two artefacts follow, both visible throughout the eval logs:

  * past the point where the report is finished, decoding degenerates into
    repeated sentences ("Sternal wires are aligned." five times in one
    report). CheXbert reads those repeats as findings, so they cost precision.
  * reports that would run past 100 tokens are cut mid-phrase ("The size of
    the cardiac"), which costs recall against the long references.

Both are properties of the *protocol*, shared by every arm this project has
trained, so no published comparison between arms is invalidated by them. But
they cap the absolute numbers, and they can be tested for free: the dumps are
already on disk, so the question "how much is the missing stop condition
worth?" is answerable without a GPU and without retraining.

WHAT IT DOES
------------
Two independent, separately switchable edits to the hypothesis text only:

  truncate  drop a trailing sentence that has no terminal punctuation, i.e.
            the fragment the 100-token cap severed.
  dedup     collapse repeated sentences ("consecutive" = only immediate
            repeats, the conservative default; "all" = any sentence already
            emitted in the same report).

References are copied through byte-identically. Nothing is overwritten: the
repaired dump is written to a new directory, so the original stays available
as the control arm of the comparison.

HONEST USE
----------
This is a decode-protocol change, not a model improvement, and it is only a
fair comparison if applied identically to every system being compared --
including the Transformer arm and the retrieval floor. Re-scoring one arm and
citing it against another arm's unrepaired numbers would manufacture a win.
The wrapper prints that warning; this note is the other half of it.

Usage:
    python scripts/repair_generations.py \\
        --dump-dir results/13d_default_operator_n400 \\
        --out-dir  results/13d_default_operator_n400_repaired \\
        --metrics
"""

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Tokens that end in "." without ending a sentence. Deliberately conservative:
# a missed boundary makes `truncate` cut back slightly further than needed,
# while a spurious one leaves a fragment in place. Both are mild, but the set
# stays small so that the rule is auditable rather than clever. "no." is
# excluded on purpose -- it is far more often the word than the abbreviation.
ABBREVIATIONS = frozenset(
    [
        "dr", "drs", "mr", "mrs", "ms", "prof", "vs", "etc", "e.g", "i.e",
        "a.m", "p.m", "approx", "fig", "cf", "st", "jr", "sr", "inc", "ltd",
        "al", "am", "pm",
    ]
)

_TERMINALS = ".?!"
_TRAILING = ".?!\"')]}:;,"

# A trailing section header left dangling once its sentence is truncated away
# ("... Impression:"), which is noise rather than text.
_HEADER_RE = re.compile(r"^[A-Za-z][A-Za-z /-]{0,24}:$")


def _normalise(text: str) -> str:
    """Collapse all whitespace runs to single spaces.

    write_hyps_refs() applies exactly this before writing, so on a real dump
    it is a no-op; applying it again makes the script total over hand-made
    files and guarantees that rejoining tokens with " " is lossless."""
    return " ".join(text.split())


def is_sentence_end(token: str) -> bool:
    """Does this whitespace-delimited token end a sentence?"""
    if not token or token[-1] not in _TERMINALS:
        return False
    if token[-1] in "?!":
        return True
    word = token.rstrip(_TRAILING)
    if not word:
        return False                      # a bare "." or "..." on its own
    if word.isdigit():
        return False                      # list marker: "1.  Interval ..."
    if len(word) == 1 and word.isalpha():
        return False                      # initial: "J. Smith"
    return word.lower() not in ABBREVIATIONS


def split_sentences(text: str) -> List[str]:
    """Split normalised report text into sentences, punctuation attached.

    Token-based rather than regex-based so that decimals ("4.5 cm"), redaction
    placeholders ("___.") and list markers each fall out of one readable rule
    instead of a lookbehind. Rejoining the result with " " reproduces the
    input exactly."""
    sentences: List[str] = []
    current: List[str] = []
    for token in _normalise(text).split(" "):
        if not token:
            continue
        current.append(token)
        if is_sentence_end(token):
            sentences.append(" ".join(current))
            current = []
    if current:
        sentences.append(" ".join(current))
    return sentences


def repair_report(
    text: str, dedup: str = "consecutive", truncate: bool = True
) -> Tuple[str, Dict[str, int]]:
    """Repair one report. Returns (text, per-report counters).

    Never returns an empty string: if the edits would delete everything, the
    original is kept and counted under `fallbacks`, because an empty
    hypothesis is a scoring artefact rather than a measurement."""
    original = _normalise(text)
    sentences = split_sentences(original)
    stats = Counter()

    if truncate:
        while sentences:
            last = sentences[-1]
            if last[-1] in _TERMINALS and not _HEADER_RE.match(last):
                break
            sentences.pop()
            stats["sentences_truncated"] += 1
            stats["tokens_truncated"] += len(last.split())

    if dedup != "none" and sentences:
        kept: List[str] = []
        seen = set()
        previous_key = None
        for sentence in sentences:
            key = sentence.lower()
            duplicate = (key == previous_key) if dedup == "consecutive" else (key in seen)
            if duplicate:
                stats["sentences_deduped"] += 1
                stats["tokens_deduped"] += len(sentence.split())
            else:
                kept.append(sentence)
                seen.add(key)
            previous_key = key
        sentences = kept

    repaired = " ".join(sentences).strip()
    if not repaired:
        stats.clear()
        stats["fallbacks"] = 1
        return original, dict(stats)
    if repaired != original:
        stats["reports_changed"] = 1
    return repaired, dict(stats)


def repair_lines(
    lines: List[str], dedup: str = "consecutive", truncate: bool = True
) -> Tuple[List[str], Dict[str, int]]:
    out: List[str] = []
    totals: Counter = Counter()
    for line in lines:
        repaired, stats = repair_report(line, dedup=dedup, truncate=truncate)
        out.append(repaired)
        totals.update(stats)
    totals["reports"] = len(lines)
    return out, dict(totals)


def _read_lines(path: Path) -> List[str]:
    return path.read_text().splitlines()


def _text_metrics(hyps: List[str], refs: List[str]) -> Optional[Dict]:
    """ROUGE-L / BLEU via the SAME functions the original eval used.

    Imported rather than reimplemented: a second copy of the tokenisation
    would silently drift from the numbers being compared against."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_evalrg", REPO_ROOT / "scripts" / "evaluate_report_generation.py"
    )
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:                                   # pragma: no cover
        print("  [metrics] skipped: cannot import evaluate_report_generation "
              "({}). Run with the main venv, or omit --metrics and score "
              "CheXbert separately.".format(exc))
        return None
    return module.compute_all_metrics(hyps, refs, chexbert=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dump-dir", required=True,
                        help="directory holding hyps.txt/refs.txt from "
                             "evaluate_report_generation.py --dump-dir")
    parser.add_argument("--out-dir", required=True,
                        help="where to write the repaired dump (must not be --dump-dir)")
    parser.add_argument("--dedup", choices=["consecutive", "all", "none"],
                        default="consecutive",
                        help="repeated-sentence policy (default: consecutive)")
    parser.add_argument("--no-truncate", action="store_true",
                        help="keep the severed trailing fragment")
    parser.add_argument("--metrics", action="store_true",
                        help="also report ROUGE-L/BLEU before and after (needs the main venv)")
    args = parser.parse_args()

    dump_dir, out_dir = Path(args.dump_dir), Path(args.out_dir)
    if dump_dir.resolve() == out_dir.resolve():
        raise SystemExit("--out-dir must differ from --dump-dir: the original dump is "
                         "the control arm of this comparison and is never overwritten")
    hyp_path, ref_path = dump_dir / "hyps.txt", dump_dir / "refs.txt"
    for path in (hyp_path, ref_path):
        if not path.is_file():
            raise SystemExit("not found: {}".format(path))

    hyps, refs = _read_lines(hyp_path), _read_lines(ref_path)
    if len(hyps) != len(refs):
        raise SystemExit("hyps/refs are misaligned: {} vs {} lines".format(len(hyps), len(refs)))

    truncate = not args.no_truncate
    repaired, stats = repair_lines(hyps, dedup=args.dedup, truncate=truncate)

    before_tokens = sum(len(h.split()) for h in hyps)
    after_tokens = sum(len(h.split()) for h in repaired)

    print("=== repair_generations: {} -> {} ===".format(dump_dir, out_dir))
    print("  policy            : dedup={} truncate={}".format(args.dedup, truncate))
    print("  reports           : {}".format(stats.get("reports", 0)))
    print("  reports changed   : {}".format(stats.get("reports_changed", 0)))
    print("  fallbacks (empty) : {}".format(stats.get("fallbacks", 0)))
    print("  sentences dropped : {} truncated, {} duplicated".format(
        stats.get("sentences_truncated", 0), stats.get("sentences_deduped", 0)))
    print("  tokens            : {} -> {}  ({:+.1f}%)".format(
        before_tokens, after_tokens,
        100.0 * (after_tokens - before_tokens) / max(before_tokens, 1)))
    print("  mean tokens/report: {:.1f} -> {:.1f}".format(
        before_tokens / max(len(hyps), 1), after_tokens / max(len(repaired), 1)))

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "hyps.txt").write_text("\n".join(repaired) + "\n")
    (out_dir / "refs.txt").write_text("\n".join(refs) + "\n")

    payload: Dict = {
        "source_dump": str(dump_dir),
        "policy": {"dedup": args.dedup, "truncate": truncate},
        "stats": stats,
        "tokens_before": before_tokens,
        "tokens_after": after_tokens,
    }

    if args.metrics:
        before = _text_metrics(hyps, refs)
        after = _text_metrics(repaired, refs)
        if before and after:
            print("\n  metric      before     after      delta")
            for key in ("rouge_l", "bleu_1", "bleu_4"):
                b, a = before[key], after[key]
                print("  {:<10}  {:.4f}    {:.4f}    {:+.4f}".format(key, b, a, a - b))
            payload["text_metrics"] = {"before": before, "after": after}
            print("\n  These are text metrics only. The pre-registered prediction is about")
            print("  CheXbert precision, which needs the isolated venv:")
            print("    DUMP_DIR={} sbatch scripts/score_chexbert_h100.sh".format(out_dir))

    (out_dir / "repair_report.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("\n  wrote {}/hyps.txt, refs.txt, repair_report.json".format(out_dir))
    print("  the original dump is untouched and remains the control arm")


if __name__ == "__main__":
    main()
