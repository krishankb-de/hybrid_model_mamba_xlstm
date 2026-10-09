"""P9-G4 (CHAT_UI_PLAN.md): the numbers the three EOS evaluation jobs print. One module, four subcommands. Each prints
`RESULT {json}` lines and `ERROR ...` lines, and nothing else (R7: anything an agent reads has left the cluster, which the DUA
forbids for MIMIC text, study ids and paths). A RESULT line holds numbers under 10 million, flags, null, and strings only from an
explicit allowlist (ALLOWED_NAMES: the wrapper's own constants, the metric names bootstrap_compare emits, the 14 CheXbert labels),
so nothing read from a log or a file can reach one as text. An ERROR line holds literals and numbers this script has validated,
never a value read out of a log or a file.

    decode   --log eval.log --dump-dir DIR --budget 200 --wall-s S                         job 1, scripts/eval_report_eos_h100.sh
    chexbert --dump-dir DIR --wall-s S                                                      job 2, eval_report_eos_chexbert_h100.sh
    hyps     NAME=hyps.txt [NAME=hyps.txt ...] [--tokenizer gpt2|none]                      job 3, eval_report_eos_compare_h100.sh
    gate     --bootstrap report.md --name-a eos_s42 --name-b m3_s42                         job 3, after bootstrap_compare.py

decode. The counts come from the line evaluate_report_generation.py --stop-at-eos prints,
`EOS stop: k/n reports ended at the end-of-report token, m were cut at max_new_tokens=B`, and the text metrics from the block it
prints after its dump. Both are read from the LAST such line of eval.log, because a requeued job appends to the log of the attempt
it replaces. The line is tied to the dump: B must be the budget the job asked for, k + m must be n, and hyps.txt and refs.txt must
hold n lines each; a log that cannot vouch for its dump is an error, and the wrapper then fails the job so that the chain stops.
    RESULT {"decode":"done","n":N,"ended_by_eos":k,"cut_at_budget":m,"share_cut":x,"budget":B,"wall_s":S,"rouge_l":x,"bleu_1":x,"bleu_4":x}

chexbert. chexbert_metrics.json and chexbert_labels.json, written by score_chexbert_standalone.py, must agree with hyps.txt on n.
    RESULT {"chexbert":"done","n":N,"micro_14":x,"macro_14":x,"micro_5":x,"macro_5":x,"wall_s":S}

hyps. Per system: mean length in words and, when the tokenizer is cached offline, in GPT-2 tokens (re-tokenised from the dump, so
close to the decoder's count and not equal to it), empty reports, repeated sentences per report and the share of reports with any
repeat, and the share whose last sentence is unterminated. Sentences are scripts/repair_generations.py's own, and so are the two
questions: a repeat is what repair_report(dedup="all") drops (a sentence the same report already had, compared in lower case), and
a report is unterminated when repair_report(truncate=True) would cut its last sentence, which is V5-D's measure of a report the
100-token cap severed (301 of 400 for the published run). The comparison job also gives it the references as a third system, `refs`:
the length that BLEU and ROUGE are scored against, and the share of reference reports that lack a final period, which is the
baseline the EOS system's unterminated share is read against (a report that ended by itself can lack one too).
    RESULT {"stats":"eos_s42","n":N,"mean_words":x,"mean_tokens":x|null,"empty":k,"repeats_per_report":x,"share_repeat":x,
            "share_unterminated":x}

gate. bootstrap_compare.py's own markdown report, parsed and not recomputed: its table rows (point estimate of A and of B, A - B,
the 95% CI) become one RESULT line each, and the last line is the verdict. The gate fails when any main-table metric has its CI
entirely below zero, that is, the EOS system is significantly worse than the published run on it. The sign is read from the printed
text, which keeps it for a rounded zero: [-0.0123, -0.0000] excludes zero and [-0.0123, +0.0000] spans it, as `hi < 0` does at full
precision. Per-label rows are printed and do not gate: bootstrap_compare's own summary does not count them either, and fourteen
more one-sided looks would make a gate fire by chance. The verdict names them anyway, for the reader, in `label_worse` (the label
rows whose CI is entirely below zero, by the same sign rule), which never changes `gate`; when both lists do not fit in 300
characters the labels go on a RESULT line of their own just before the verdict, which stays last. A report that does not parse,
whose columns are not A then B (swapped columns would flip every difference), whose A - B does not match its own A and B, whose
row names are not the metrics and labels this job prints, or that lacks the label metrics (it was run without the label matrices)
is an error and prints no gate line at all. A failing gate is a finding, not a crash: exit 0 either way.
    RESULT {"bootstrap":"paired","name_a":..,"name_b":..,"n":N,"samples":S,"seed":s}
    RESULT {"metric":"rouge_l","a":x,"b":x,"diff":x,"lo":x,"hi":x}        one per main-table metric
    RESULT {"label":"Lung Lesion","a":x,"b":x,"diff":x,"lo":x,"hi":x}     one per label
    RESULT {"gate":"pass"|"fail","worse":[metric, ...],"label_worse":[label, ...]}

Exit codes: 0 done; 2 a check failed (an ERROR line says which, in literals and numbers); 1 anything unexpected (the traceback goes
to stderr, which the wrapper keeps in a file on the cluster, and one ERROR line names the exception class).

Standard library only at import, so that .venv_chexbert (transformers<5, no torch) can run the chexbert subcommand.
"""

import argparse
import json
import math
import re
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RESULT_LIMIT = 300                                    # what `chat_remote.sh summary` keeps of a line
MAX_NUMBER = 10 ** 7                                  # 8 digits or more could be an id (study 5xxxxxxx, subject 1xxxxxxx)

# The only strings a RESULT line may hold, as a value or as a list item (R7). A name is printed only if it is in ALLOWED_NAMES, so
# report text, a study or subject id or a path cannot reach a line, whatever file or log it was read from.
WRAPPER_NAMES = ("done", "paired", "pass", "fail", "eos_s42", "m3_s42", "refs")      # step results, verdicts, the systems compared
BOOTSTRAP_METRICS = ("rouge_l", "bleu_1", "bleu_4", "chexbert_14_micro", "chexbert_14_macro", "chexbert_5_micro", "chexbert_5_macro",
                     "exact_match_accuracy_14", "exact_match_accuracy_5")           # bootstrap_compare's main table (a test reads it)
CHEXBERT_14 = ("Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema", "Consolidation", "Pneumonia",
               "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices", "No Finding")
ALLOWED_NAMES = frozenset(WRAPPER_NAMES + BOOTSTRAP_METRICS + CHEXBERT_14)    # CHEXBERT_14 is app/labels.py's (a test ties them)


class ResultError(Exception):
    """A check failed. The message is a literal, or literals and numbers: it is printed as an ERROR line."""


# ── the line shapes ───────────────────────────────────────────────────────────

def _plain(value: Any) -> bool:
    if value is None or isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(value) and abs(value) < MAX_NUMBER
    if isinstance(value, str):
        return value in ALLOWED_NAMES
    if isinstance(value, list):
        return all(_plain(item) for item in value)
    return False


def _render(payload: Dict[str, Any]) -> str:
    if not all(_plain(value) for value in payload.values()):
        raise ValueError("a RESULT value must be a number under {}, a flag, null, an allowlisted name or a list of those".format(
            MAX_NUMBER))
    return "RESULT " + json.dumps(payload, separators=(",", ":"))


def result_line(payload: Dict[str, Any]) -> str:
    """`RESULT {compact json}`. R7: every number is finite and under 10 million, and every string is one of ALLOWED_NAMES, matched
    exactly (the wrapper's constants, the metric names bootstrap_compare emits, the 14 CheXbert labels), so no report text, id or
    path can ride along; and the line stays under what `chat_remote.sh summary` keeps. Anything else is a ValueError."""
    line = _render(payload)
    if len(line) >= RESULT_LIMIT:
        raise ValueError("a RESULT line must stay under {} characters".format(RESULT_LIMIT))
    return line


def count_lines(path: Path) -> int:
    data = Path(path).read_bytes()
    return data.count(b"\n") + (1 if data and not data.endswith(b"\n") else 0)


def _read_text(path: Path) -> str:
    return Path(path).read_text(encoding="utf-8", errors="replace")


# ── decode: the evaluator's log ───────────────────────────────────────────────

EOS_STOP = re.compile(r"^EOS stop: (\d+)/(\d+) reports ended at the end-of-report token, (\d+) were cut at max_new_tokens=(\d+)$")
AGGREGATE = re.compile(r"^=== Aggregate over (\d+) samples ===$")
TEXT_METRICS = ("rouge_l", "bleu_1", "bleu_4", "meteor")


def parse_eos_stop(text: str) -> Optional[Tuple[int, int, int, int]]:
    """(ended by EOS, n, cut at the budget, budget) from the last `EOS stop:` line of the log, or None."""
    found = None
    for line in text.splitlines():
        match = EOS_STOP.match(line)
        if match:
            found = tuple(int(group) for group in match.groups())
    return found


def parse_aggregate(text: str) -> Optional[Dict[str, Any]]:
    """The metrics dict the evaluator prints (json.dumps, indent=2) after its last `=== Aggregate over N samples ===` line."""
    lines = text.splitlines()
    starts = [i for i, line in enumerate(lines) if AGGREGATE.match(line)]
    if not starts:
        return None
    tail = "\n".join(lines[starts[-1] + 1:])
    brace = tail.find("{")
    if brace < 0:
        return None
    try:
        payload, _ = json.JSONDecoder().raw_decode(tail[brace:])
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None


def decode_result(log_text: str, dump_dir: Path, budget: int, wall_s: int) -> Tuple[Dict[str, Any], List[str]]:
    """(the RESULT payload, the === notes that explain what is missing from it). Raises ResultError for a log that cannot
    vouch for the dump beside it."""
    eos = parse_eos_stop(log_text)
    if eos is None:
        raise ResultError("eval.log has no EOS stop line: the decoder did not run with --stop-at-eos")
    ended, n, cut, seen_budget = eos
    if n < 1 or ended + cut != n:
        raise ResultError("the EOS stop counts do not add up: {} ended and {} cut against n={}".format(ended, cut, n))
    if seen_budget != budget:
        raise ResultError("the EOS stop line says budget {}, this job asked for {}".format(seen_budget, budget))
    for name in ("hyps.txt", "refs.txt"):
        path = Path(dump_dir) / name
        if not path.is_file():
            raise ResultError("{} is missing from the dump".format(name))
        lines = count_lines(path)
        if lines != n:
            raise ResultError("{} has {} lines, the EOS stop line says n={}".format(name, lines, n))
    result: Dict[str, Any] = {"decode": "done", "n": n, "ended_by_eos": ended, "cut_at_budget": cut,
                              "share_cut": round(cut / n, 4), "budget": budget, "wall_s": int(wall_s)}
    notes: List[str] = []
    aggregate = parse_aggregate(log_text)
    if aggregate is None:
        notes.append("=== no aggregate metrics in eval.log: rouge_l, bleu_1 and bleu_4 are left out ===")
    else:
        examples = aggregate.get("num_examples")
        if type(examples) is not int:                      # whatever JSON the log holds: not a bool, float or text, and never echoed
            raise ResultError("the aggregate metrics do not hold an integer count of reports")
        if examples != n:
            raise ResultError("the aggregate metrics count a different number of reports than the EOS stop line")
        for key in TEXT_METRICS:
            value = aggregate.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                result[key] = round(float(value), 4)          # METEOR is null without nltk's wordnet and is then left out
    return result, notes


# ── chexbert: the scorer's files ──────────────────────────────────────────────

def chexbert_result(dump_dir: Path, wall_s: int) -> Dict[str, Any]:
    dump_dir = Path(dump_dir)
    try:
        metrics = json.loads(_read_text(dump_dir / "chexbert_metrics.json"))
    except (OSError, ValueError):
        raise ResultError("chexbert_metrics.json is missing or unreadable") from None
    try:
        labels = json.loads(_read_text(dump_dir / "chexbert_labels.json"))
    except (OSError, ValueError):
        raise ResultError("chexbert_labels.json is missing or unreadable: bootstrap_compare needs it") from None
    try:
        n = metrics["num_examples"]
        if type(n) is not int:                             # a number the scorer wrote, not text, a float or a bool: never echoed
            raise ResultError("chexbert_metrics.json does not hold an integer example count")
        found = {"micro_14": metrics["chexbert_14"]["micro avg"]["f1-score"], "macro_14": metrics["chexbert_14"]["macro avg"]["f1-score"],
                 "micro_5": metrics["chexbert_5"]["micro avg"]["f1-score"], "macro_5": metrics["chexbert_5"]["macro avg"]["f1-score"]}
        label_rows = (len(labels["y_true"]), len(labels["y_pred"]))
    except (KeyError, TypeError, ValueError):
        raise ResultError("chexbert_metrics.json or chexbert_labels.json does not have the shape the scorer writes") from None
    try:
        hyps = count_lines(dump_dir / "hyps.txt")
    except OSError:
        raise ResultError("hyps.txt is missing from the dump") from None
    if not (n == hyps == label_rows[0] == label_rows[1]):
        raise ResultError("n differs: hyps.txt has {}, the metrics say {}, the label matrices have {} and {}".format(
            hyps, n, label_rows[0], label_rows[1]))
    result = {"chexbert": "done", "n": n}
    result.update({key: round(float(value), 4) for key, value in found.items()})
    result["wall_s"] = int(wall_s)
    return result


# ── hyps: length, repeats, unterminated endings ───────────────────────────────

def report_features(text: str) -> Tuple[int, int, bool]:
    """(words, repeated sentences, last sentence unterminated) of one report, by scripts/repair_generations.py's rules."""
    from scripts.repair_generations import repair_report
    repeats = repair_report(text, dedup="all", truncate=False)[1].get("sentences_deduped", 0)
    cut = repair_report(text, dedup="none", truncate=True)[1]
    words = len(text.split())
    # When truncation would delete every sentence the original is kept and counted as a fallback, not as a truncation; that is
    # still a last sentence that was cut, unless the report is empty (an empty report has no sentences at all).
    unterminated = bool(cut.get("sentences_truncated")) or (bool(cut.get("fallbacks")) and words > 0)
    return words, repeats, unterminated


def hyp_stats(hyps: Sequence[str], count_tokens: Optional[Callable[[str], int]] = None) -> Dict[str, Any]:
    n = len(hyps)
    if n == 0:
        raise ResultError("there are no reports to count")
    features = [report_features(text) for text in hyps]
    return {
        "n": n,
        "mean_words": round(sum(f[0] for f in features) / n, 2),
        "mean_tokens": None if count_tokens is None else round(sum(count_tokens(text) for text in hyps) / n, 2),
        "empty": sum(1 for f in features if f[0] == 0),
        "repeats_per_report": round(sum(f[1] for f in features) / n, 4),
        "share_repeat": round(sum(1 for f in features if f[1] > 0) / n, 4),
        "share_unterminated": round(sum(1 for f in features if f[2]) / n, 4),
    }


def load_token_counter(name: str) -> Optional[Callable[[str], int]]:
    """len(tokenizer.encode(text)) for the Hugging Face tokenizer `name`; None for `none` and when it cannot be loaded (offline
    without a cached copy, or no transformers): the token mean is then null, not a crash."""
    if name == "none":
        return None
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(name)
    except Exception:
        return None
    return lambda text: len(tokenizer.encode(text))


# ── gate: bootstrap_compare's report ──────────────────────────────────────────

REQUIRED_METRICS = ("rouge_l", "bleu_1", "bleu_4", "chexbert_14_micro", "chexbert_14_macro", "exact_match_accuracy_14")
_META = re.compile(r"n=(\d+) studies, (\d+) paired bootstrap\s+resamples, seed (-?\d+)")
_ROW = re.compile(r"^\| (?P<name>[^|]+?) \| (?P<a>[+-]?\d+\.\d+) \| (?P<b>[+-]?\d+\.\d+) \| (?P<diff>[+-]\d+\.\d+)"
                  r" \| \[(?P<lo>[+-]\d+\.\d+), (?P<hi>[+-]\d+\.\d+)\] \| [^|]+ \|$")
_ROUNDING = 1.6e-4                                    # A - B against the printed A, B and diff: three values rounded to 4 places


def parse_bootstrap(text: str, name_a: str, name_b: str) -> Dict[str, Any]:
    """{"n", "samples", "seed", "metrics": [row], "labels": [row]} from bootstrap_compare.render's markdown, a row being
    {"name", "a", "b", "diff", "lo", "hi"} as printed (4 places). Raises ResultError for anything that is not that report."""
    meta = _META.search(text)
    if meta is None:
        raise ResultError("no bootstrap report: its n, resamples and seed line is missing")
    rows: Dict[str, List[Dict[str, Any]]] = {"metrics": [], "labels": []}
    mode = None
    for line in text.splitlines():
        if line.startswith("| metric |") or line.startswith("| label |"):
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if cells[1:] != [name_a, name_b, "diff", "95% CI", "verdict"]:
                raise ResultError("a table header does not name the two systems as asked, A then B")
            mode = "metrics" if cells[0] == "metric" else "labels"
            continue
        if mode is None:
            continue
        if not line.startswith("|"):
            mode = None
            continue
        if line.startswith("|---"):
            continue
        match = _ROW.match(line)
        if match is None:
            raise ResultError("a table row is not a name, A, B, A - B, an interval and a verdict")
        row = {"name": match.group("name")}
        row.update({key: float(match.group(key)) for key in ("a", "b", "diff", "lo", "hi")})
        if abs(row["diff"] - (row["a"] - row["b"])) > _ROUNDING or row["lo"] > row["hi"]:
            raise ResultError("a table row's difference is not its A minus its B, or its interval is upside down")
        rows[mode].append(row)
    foreign = (sum(1 for row in rows["metrics"] if row["name"] not in BOOTSTRAP_METRICS)
               + sum(1 for row in rows["labels"] if row["name"] not in CHEXBERT_14))
    if foreign:
        raise ResultError("the report has {} row(s) named otherwise than the metrics and labels this job prints".format(foreign))
    names = [row["name"] for row in rows["metrics"]]
    if len(set(names)) != len(names):
        raise ResultError("a metric appears twice in the report")
    missing = [name for name in REQUIRED_METRICS if name not in names]
    if missing:
        raise ResultError("the report lacks {} of the metrics the gate is about, such as {}: were the label matrices passed?".format(
            len(missing), missing[0]))
    return {"n": int(meta.group(1)), "samples": int(meta.group(2)), "seed": int(meta.group(3)),
            "metrics": rows["metrics"], "labels": rows["labels"]}


def worse_metrics(metrics: Sequence[Dict[str, Any]]) -> List[str]:
    """The rows (metrics, or labels) whose 95% CI of (A - B) lies entirely below zero. The sign is read off the rounded text,
    which keeps it: float('-0.0000') is -0.0, a CI that stops just short of zero, and copysign sees it where `< 0` would not."""
    return [row["name"] for row in metrics if math.copysign(1.0, row["hi"]) < 0]


def comparison_lines(parsed: Dict[str, Any], name_a: str, name_b: str) -> List[str]:
    lines = [result_line({"bootstrap": "paired", "name_a": name_a, "name_b": name_b, "n": parsed["n"], "samples": parsed["samples"],
                          "seed": parsed["seed"]})]
    for kind, rows in (("metric", parsed["metrics"]), ("label", parsed["labels"])):
        for row in rows:
            lines.append(result_line({kind: row["name"], "a": row["a"], "b": row["b"], "diff": row["diff"], "lo": row["lo"],
                                      "hi": row["hi"]}))
    worse = worse_metrics(parsed["metrics"])
    verdict = {"gate": "fail" if worse else "pass", "worse": worse}
    together = dict(verdict, label_worse=worse_metrics(parsed["labels"]))     # for the reader: it never changes the gate
    if len(_render(together)) < RESULT_LIMIT:
        lines.append(result_line(together))
    else:                                          # 9 metrics and 14 labels do not fit: the labels get a line, the verdict stays last
        lines.append(result_line({"label_worse": together["label_worse"]}))
        lines.append(result_line(verdict))
    return lines


# ── command line ──────────────────────────────────────────────────────────────

def _cmd_decode(args: argparse.Namespace) -> List[str]:
    result, notes = decode_result(_read_text(args.log), Path(args.dump_dir), args.budget, args.wall_s)
    return [result_line(result)] + notes


def _cmd_chexbert(args: argparse.Namespace) -> List[str]:
    return [result_line(chexbert_result(Path(args.dump_dir), args.wall_s))]


def _cmd_hyps(args: argparse.Namespace) -> List[str]:
    systems = []
    for spec in args.systems:
        name, sep, path = spec.partition("=")
        if not sep or name not in ALLOWED_NAMES:
            raise ResultError("a system is NAME=PATH, with a name this job prints")
        systems.append((name, path))
    counter = load_token_counter(args.tokenizer)
    lines = []
    if counter is None and args.tokenizer != "none":
        lines.append("=== no tokenizer available offline: mean_tokens is null ===")
    for name, path in systems:
        stats = hyp_stats(_read_text(Path(path)).splitlines(), counter)
        lines.append(result_line(dict({"stats": name}, **stats)))
    return lines


def _cmd_gate(args: argparse.Namespace) -> List[str]:
    for name in (args.name_a, args.name_b):
        if name not in ALLOWED_NAMES:
            raise ResultError("a system name is not one this job prints")
    parsed = parse_bootstrap(_read_text(Path(args.bootstrap)), args.name_a, args.name_b)
    return comparison_lines(parsed, args.name_a, args.name_b)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    decode = sub.add_parser("decode", help="the RESULT line of the decode job")
    decode.add_argument("--log", required=True, help="eval.log: the evaluator's raw stdout and stderr")
    decode.add_argument("--dump-dir", required=True, help="the directory holding hyps.txt and refs.txt")
    decode.add_argument("--budget", type=int, required=True, help="the --max-new-tokens the job asked for")
    decode.add_argument("--wall-s", type=int, required=True, help="seconds the decoder ran")
    decode.set_defaults(run=_cmd_decode)
    chexbert = sub.add_parser("chexbert", help="the RESULT line of the CheXbert job")
    chexbert.add_argument("--dump-dir", required=True)
    chexbert.add_argument("--wall-s", type=int, required=True)
    chexbert.set_defaults(run=_cmd_chexbert)
    hyps = sub.add_parser("hyps", help="mean length, repeats and unterminated endings of hyps files")
    hyps.add_argument("systems", nargs="+", metavar="NAME=hyps.txt")
    hyps.add_argument("--tokenizer", default="gpt2", help="a Hugging Face tokenizer cached offline, or none (default: gpt2)")
    hyps.set_defaults(run=_cmd_hyps)
    gate = sub.add_parser("gate", help="the comparison lines and the verdict, from bootstrap_compare's report")
    gate.add_argument("--bootstrap", required=True, help="bootstrap_compare.py's --output file")
    gate.add_argument("--name-a", required=True, help="the EOS system, the first column")
    gate.add_argument("--name-b", required=True, help="the published system, the second column")
    gate.set_defaults(run=_cmd_gate)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        lines = args.run(args)                       # nothing is printed until every line is built: no partial answer
    except ResultError as exc:
        print("ERROR {}".format(exc))
        return 2
    except Exception as exc:                          # the class name is all that is printed; the traceback goes to stderr
        traceback.print_exc()
        print("ERROR report_eos_stats raised {}".format(type(exc).__name__))
        return 1
    for line in lines:
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
