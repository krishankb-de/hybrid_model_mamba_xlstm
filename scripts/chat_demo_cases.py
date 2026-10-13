"""CHAT_UI_PLAN.md P9-C: which test studies to show at the supervisor demo. A CPU job (scripts/chat_demo_cases_h100.sh); it prints numbers only.

    python scripts/chat_demo_cases.py --labels results/report_gen_m3_test_split_s42/chexbert_labels.json --gallery <gallery dir>

The demo shows five test studies, two normal, two with support devices and one with a pneumothorax, chosen from the published labels, so that
nobody picks the ones that happen to look good. This job lists the candidates: for each of the three kinds, up to five rows of the test split, a
fair draw from every row of that kind. The person running the demo takes the first 2, 2 and 1 of the lists, skipping a row that is already picked
(a pneumothorax with a chest tube is in two lists).

What it reads.
  --labels   the CheXbert labels of the published run's test split, `chexbert_labels.json`, as scripts/score_chexbert_standalone.py writes it:
             the keys `y_true`, `y_pred`, `label_names`, `five_label_indices`, `hyp_file` and `ref_file`. Only `y_true` is read, and `label_names`
             when there is one: `y_true` is one row of 14 labels (0 or 1) per reference report, in refs.txt order, which is the test split's
             row order, and `label_names` says which column is which (the columns are put in CHEXBERT_14's order by name; a file without the key,
             the older of the two formats, is read in that order, as P5-C's reader reads it). The labels of the reference reports are what the
             kinds are decided on. The model's own labels (`y_pred`), its reports and its scores are never read, so the draw cannot favour the
             studies the model does well on.
  --gallery  the retrieval gallery the server serves: only the test-split size in its manifest.json. The job refuses when the labels' row
             count is not that size, because a row number must name the same study that the picker shows. (That the gallery's test studies
             are in the labels' order was proved when the gallery's labels were built: P5-C's cross-check of the gallery's test reports and
             labels against refs.txt and y_true found 0 differences in 2,663.)

The kinds, on CheXbert-14 and the reference labels:
  normal            No Finding is 1 and every other label is 0
  support_devices   Support Devices is 1
  pneumothorax      Pneumothorax is 1

The draw. A generator seeded with 42 gives one number to each row of a kind, in row order, and the five rows with the lowest numbers are the
candidates, listed in that order: a uniformly random choice of five rows, in a random order, so that the first two are as fair a pick as the five.
Only random.Random.random() is used, the one thing Python promises to give the same numbers for the same seed in every version (the laptop runs
3.14 and the cluster 3.11), so the rows are a function of the labels file alone. A kind with fewer than five rows lists them all.

Output. Stdout is one line, `RESULT {json}`: demo_cases (`done`), n (the studies), counts (the rows of each kind), candidates (the drawn rows of
each kind) and seed (42). Exit status 0 when it printed that; 1 when it refused, with one `ERROR <code> [name=number ...]` line in its place, the
code one of ERROR_CODES (`labels_unreadable`, `labels_shape rows=R width=W`, `label_names`, `gallery_unreadable`, `count_mismatch labels=N
gallery=M`) or `failed <ExceptionClass>`; 64 for a command line it does not take (`ERROR usage`). Nothing is written on stderr or to a file.

R7 (DUA). The labels file is MIMIC-derived: which test study has which finding. This script prints a row number, which is a number of this
project, and counts, and nothing else of the file: no label name, no study id, no path (the file names its own source files, which are never
printed), no part of an exception (an unexpected one is its class's name). Every write is `say`, which has two callers: the help text and a line
made here of numbers and fixed words.
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, TextIO

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:   # `python scripts/chat_demo_cases.py` puts scripts/ on the path, not the repo's root
    sys.path.insert(0, REPO_ROOT)
from app.labels import CHEXBERT_14  # noqa: E402  (standard library only: the one list of the 14 names)

LABELS_KEY, NAMES_KEY = "y_true", "label_names"   # the two keys of chexbert_labels.json this job reads
SEED = 42
PER_CATEGORY = 5
CATEGORIES = ("normal", "support_devices", "pneumothorax")
N_LABELS = len(CHEXBERT_14)
NO_FINDING, SUPPORT_DEVICES, PNEUMOTHORAX = (CHEXBERT_14.index(name) for name in ("No Finding", "Support Devices", "Pneumothorax"))

EXIT_REFUSED = 1
EXIT_USAGE = 64   # as scripts/label_gallery_reports.py: argparse's own 2 is a refusal elsewhere
# Every reason this job stops with, as the code of an `ERROR <code> [name=number ...]` line. The wrapper's allowlist names exactly these.
ERROR_CODES = ("usage", "labels_unreadable", "labels_shape", "label_names", "gallery_unreadable", "count_mismatch")


class Refused(Exception):
    """A reason to stop that the job log may show: a code from ERROR_CODES and whole numbers, never a path, a name, an id or text."""

    def __init__(self, code: str, **numbers: int):
        assert code in ERROR_CODES, code
        super().__init__(code)
        self.code, self.numbers = code, numbers

    def line(self) -> str:
        return "ERROR " + self.code + "".join(" {}={}".format(key, int(value)) for key, value in self.numbers.items())


def say(stream: TextIO, line: str) -> None:
    """The one place this script writes: a whole line, in one call."""
    stream.write(line + "\n")
    stream.flush()


class _Parser(argparse.ArgumentParser):
    """argparse for a job whose arguments are paths: no abbreviations, and a refusal never quotes what it was given."""

    def error(self, message: str) -> Any:
        raise Refused("usage")

    def parse_args(self, args: Optional[Sequence[str]] = None, namespace: Any = None) -> argparse.Namespace:
        parsed, extras = self.parse_known_args(args, namespace)
        if extras:
            raise Refused("usage")
        return parsed


def build_parser() -> argparse.ArgumentParser:
    p = _Parser(prog="chat_demo_cases.py", allow_abbrev=False, add_help=False,
                description="List up to {} candidate test rows for each of three kinds of study (normal, support devices, pneumothorax), a seeded "
                            "fair draw by the reference labels, and print numbers only: one RESULT line. Exit 0 when it printed it, 1 when it "
                            "refused (one ERROR line), 64 for a command line it does not take.".format(PER_CATEGORY))
    p.add_argument("--labels", metavar="PATH", help="the published run's chexbert_labels.json (its y_true and, if present, label_names are read)")
    p.add_argument("--gallery", metavar="DIR", help="the retrieval gallery the server serves (its manifest.json's test-split size is read)")
    p.add_argument("-h", "--help", action="store_true", help="show this help")
    return p


# ---- what is read -----------------------------------------------------------------------------------------------------------------

def read_labels(path: str) -> List[List[int]]:
    """The reference labels of the test split: one row of 14 ints (0 or 1) per study, columns in CHEXBERT_14's order. Refused as
    `labels_unreadable` (no such file, not JSON, no y_true), `labels_shape` (with the number of rows and the width of the first: y_true is not
    a list of rows of 14 labels, each 0 or 1) or `label_names` (the names are not the 14, once each)."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        y_true = payload[LABELS_KEY]
        has_names = NAMES_KEY in payload
        names = payload.get(NAMES_KEY)
    except (OSError, ValueError, RecursionError, KeyError, TypeError, AttributeError):
        raise Refused("labels_unreadable") from None
    rows = len(y_true) if isinstance(y_true, list) else 0
    width = len(y_true[0]) if rows and isinstance(y_true[0], list) else 0
    if not rows or not all(isinstance(row, list) and len(row) == N_LABELS and all(type(v) is int and v in (0, 1) for v in row) for row in y_true):
        raise Refused("labels_shape", rows=rows, width=width)
    if has_names:
        if not (isinstance(names, list) and len(names) == N_LABELS and all(isinstance(n, str) for n in names) and sorted(names) == sorted(CHEXBERT_14)):
            raise Refused("label_names")
        if names != CHEXBERT_14:   # the columns are those of the dump's own order: put them in the standard one, by name
            order = [names.index(name) for name in CHEXBERT_14]
            y_true = [[row[i] for i in order] for row in y_true]
    return y_true


def read_test_count(gallery: str) -> int:
    """The gallery's test-split size, from its manifest.json: counts.test, which counts.report_rows must equal together with counts.images (the
    layout app/gallery.py opens it by). Refused as `gallery_unreadable` when the manifest is not there, is not JSON or says something else."""
    try:
        manifest = json.loads((Path(gallery) / "manifest.json").read_text(encoding="utf-8"))
        counts = manifest["counts"]
        test, images, report_rows = counts["test"], counts["images"], counts["report_rows"]
    except (OSError, ValueError, RecursionError, KeyError, TypeError, AttributeError):
        raise Refused("gallery_unreadable") from None
    if not all(type(v) is int and v >= 1 for v in (test, images, report_rows)) or report_rows != images + test:
        raise Refused("gallery_unreadable")
    return test


# ---- the kinds and the draw -------------------------------------------------------------------------------------------------------

def categories(rows: Sequence[Sequence[int]]) -> Dict[str, List[int]]:
    """The rows of each kind, in row order: -> {kind: [row numbers]} for the kinds of CATEGORIES. A row can be of two kinds (a pneumothorax with
    a support device), never of `normal` and another: `normal` is No Finding alone."""
    found: Dict[str, List[int]] = {name: [] for name in CATEGORIES}
    for number, row in enumerate(rows):
        if row[NO_FINDING] == 1 and sum(row) == 1:
            found["normal"].append(number)
        if row[SUPPORT_DEVICES] == 1:
            found["support_devices"].append(number)
        if row[PNEUMOTHORAX] == 1:
            found["pneumothorax"].append(number)
    return found


def draw(rows: Sequence[int], seed: int = SEED, k: int = PER_CATEGORY) -> List[int]:
    """Up to k of `rows`, a fair draw: a generator seeded with `seed` gives each row one number, in the order the rows are given, and the k rows
    with the lowest numbers are the draw, listed in that order. That is a uniformly random k of the rows, in a random order, and it uses only the
    generator's random(), which Python keeps the same for a seed in every version."""
    rng = random.Random(seed)
    return [row for _, row in sorted((rng.random(), row) for row in rows)[:k]]


def payload(n: int, found: Dict[str, List[int]]) -> Dict[str, Any]:
    return {"demo_cases": "done", "n": n, "counts": {name: len(found[name]) for name in CATEGORIES},
            "candidates": {name: draw(found[name]) for name in CATEGORIES}, "seed": SEED}


def result_line(content: Dict[str, Any]) -> str:
    """The one line this script prints for a result."""
    return "RESULT " + json.dumps(content, separators=(",", ":"))


def main(argv: Optional[Sequence[str]] = None, stdout: Optional[TextIO] = None) -> int:
    """Run the job; -> the exit status. `stdout` is where its one line goes (the process's own by default)."""
    stdout = sys.stdout if stdout is None else stdout
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        if args.help:
            say(stdout, parser.format_help().rstrip("\n"))
            return 0
        if args.labels is None or args.gallery is None:
            raise Refused("usage")
        labels = read_labels(args.labels)
        studies = read_test_count(args.gallery)
        if len(labels) != studies:
            raise Refused("count_mismatch", labels=len(labels), gallery=studies)
        say(stdout, result_line(payload(len(labels), categories(labels))))
        return 0
    except Refused as exc:
        say(stdout, exc.line())
        return EXIT_USAGE if exc.code == "usage" else EXIT_REFUSED
    except Exception as exc:   # the class, never the text
        say(stdout, "ERROR failed " + type(exc).__name__)
        return EXIT_REFUSED


if __name__ == "__main__":
    sys.exit(main())
