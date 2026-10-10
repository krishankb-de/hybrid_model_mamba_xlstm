"""P5-C (CHAT_UI_PLAN.md): CheXbert-14 labels for the retrieval gallery's report rows (section 6.5 of the plan: labels.npy, label_names.json).

    python scripts/label_gallery_reports.py --gallery <gallery dir> [--reference-dir results/report_gen_m3_test_split_s42]
        [--budget-s 5400] [--canary 1000] [--progress 10000] [--device auto|cpu|cuda]        one job, start to finish
    python scripts/label_gallery_reports.py --gallery <dir> --shard I --of N [the same options]  label reps[I::N] only
    python scripts/label_gallery_reports.py --gallery <dir> --merge --of N [--reference-dir]     combine the N shards, then as one job

Runs in .venv_chexbert (f1chexbert needs transformers<5), so it imports the standard library, numpy and f1chexbert and nothing else:
not the repo (no hybrid_xmamba, app or scripts), which is why the 14 names are a literal copy below.

One get_label per duplicate group, not per row. report_texts.txt has one report per gallery row, and txt_groups.npy puts rows whose
text is the same after lower-casing and collapsing whitespace in one group. The uncased BERT tokenizer behind F1CheXbert sees no
difference between the members of a group, so the labels of a group's first row (its representative) are the labels of every row in
it, and a report that repeats an earlier one is never labelled on its own. That is an argument, and the cross-check at the end
measures it: for the test split's 2,663 reports (some of which take their labels from a train row's text) labels.npy must equal the
y_true of the published dump, to the last label. A zero is evidence for the argument only as far as the reports that take their labels
from another row reach, so the log says how many they are (`test_rows_sharing_a_text`).

Speed is why it is built this way. F1CheXbert on a CPU takes about 0.16 s a report (job 2525606), so the groups are hours of CPU. f1chexbert 0.0.2
(read from its source) takes the GPU itself when torch sees one: `F1CheXbert(refs_filename=None, hyps_filename=None, device=None, **kwargs)`
sets device to `cuda` if torch.cuda.is_available(), and get_label(text) returns a list of 14 ints. (scripts/setup_chexbert_venv_h100.sh installs
a CPU-only torch into .venv_chexbert, where that is never true: the GPU wrapper checks what the venv's torch sees before it labels anything.)
The one job prints the device of its model's first parameter and the constructor's argument names, labels the first --canary groups,
prints the rate and what it projects for all of them, and exits 2 if that is past --budget-s: the wrapper reads exit 2 as "use the sharded CPU
path". --device passes a `device` argument to the constructor (cpu or cuda); auto passes none, so the library's own choice stands.

What it writes, all in the gallery directory and nothing outside it:
  one job and --merge   labels_check.json always; then, only if the cross-check found no difference, labels.npy (uint8, one row of 14
                        per report row), label_names.json and the manifest's labels_status (done) with labels_info. With a difference
                        the labels go to labels_unverified.npy, labels_status stays pending, and the gallery looks unlabelled to
                        whoever opens it.
  --shard I --of N      labels_shard_I_rows.npy (the gallery rows of reps[I::N], int64) and labels_shard_I.npy (their labels, uint8): the
                        second is written last and is the marker of a finished shard, which a later run keeps rather than labelling again.
Before any of it, the gallery must be built and verified (manifest.json, gate_rk.equal true in the manifest and in gate_rk.json) and not
already labelled: a finished labelling is never overwritten (R8), and that is checked again at the end, before anything is written, from a
fresh read of the manifest (a GPU job and a CPU merge can both start on a pending gallery; the one that ends second refuses). Before the
labeller is even built, the 2,663 test reports of the gallery are compared with the published refs.txt: if they differ the job stops at once,
instead of finding out after the hours of labelling.

Exit codes: 0 done, 1 refused or failed (a cross-check with a difference too), 2 the canary projects past the budget, 64 a bad command line
(argparse's own 2 would read as the canary's).

R7. What is printed here could be MIMIC-derived (a report, an id, a path), so none of that is: every line is a `[labels]` line, a
`RESULT {json}` line, an `ERROR <code> [name=number ...]` line or a `=== note ... ===` line of a fixed shape, with counts, booleans,
numbers, key names and file basenames. No path, no id, no report text, no exception message (an unexpected exception prints its class
name; its traceback goes to stderr). The wrappers let only lines of those shapes into the job log (their LABEL_SHAPES, tested against
everything printed here) and keep the rest in files in the gallery.
"""
import argparse
import inspect
import json
import os
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# F1CheXbert.target_names, in its order: the same list as app/labels.py's CHEXBERT_14 and scripts/train_report_generation.py's
# CHEXPERT_14_LABELS. A literal copy because this file imports nothing of the repo; tests/test_label_gallery_reports.py pins it against
# both, and every run asserts it against the labeller's own before a single report is labelled.
CHEXBERT_14 = ["Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema", "Consolidation", "Pneumonia",
               "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices", "No Finding"]
N_LABELS = len(CHEXBERT_14)

DEFAULT_REFERENCE_DIR = "results/report_gen_m3_test_split_s42"
DEFAULT_BUDGET_S = 5400.0        # a 2 h job: loading, the check and the writes need little of it
DEFAULT_CANARY = 1000
DEFAULT_PROGRESS = 10000

EXIT_REFUSED = 1
EXIT_SLOW = 2                    # the canary projects past the budget: the wrapper reads it as "use the sharded CPU path"
EXIT_USAGE = 64

# Every reason this script stops with, as the code of an `ERROR <code> [name=number ...]` line. The wrappers' allowlists name exactly these.
ERROR_CODES = ("no_manifest", "manifest_unreadable", "gate_not_equal", "already_labelled", "inputs_unreadable", "rows_disagree",
               "groups_disagree", "test_rows", "reference_unreadable", "reference_shape", "reference_label_names", "refs_mismatch",
               "no_device_argument", "label_order", "bad_label_row", "shards_missing", "shard_invalid")


# ── what is printed (R7) ──────────────────────────────────────────────────────

def say(message: str) -> None:
    """One [labels] line: counts, booleans, numbers, key names and file basenames, never a path, an id or report text. The wrappers pass a
    line to the job log only if it has one of the shapes in their LABEL_SHAPES, so a new line needs a new shape there and in the tests."""
    print("[labels] " + message, flush=True)


def note(message: str) -> None:
    print("=== note: " + message + " ===", flush=True)


def emit_result(payload: Dict[str, int]) -> None:
    print("RESULT " + json.dumps(payload, separators=(",", ":")), flush=True)


class Refused(Exception):
    """A reason to stop that the job log may show: a code from ERROR_CODES and whole numbers, never a path, a name, an id or text."""

    def __init__(self, code: str, **numbers: int):
        assert code in ERROR_CODES, code
        super().__init__(code)
        self.code, self.numbers = code, numbers

    def line(self) -> str:
        return "ERROR " + self.code + "".join(" {}={}".format(key, int(value)) for key, value in self.numbers.items())


def whole(value: Any) -> int:
    """A count out of a manifest as an int; anything else (a missing key, a float, a string) as 0."""
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


# ── files ─────────────────────────────────────────────────────────────────────

def write_atomic(path: Path, write: Any) -> None:
    """Written whole or not at all: the marker of a finished shard, the manifest and the labels are never seen half-written."""
    tmp = path.with_name(path.name + ".tmp")
    with open(str(tmp), "wb") as handle:
        write(handle)
    os.replace(str(tmp), str(path))


def write_json_atomic(path: Path, obj: Any, indent: Optional[int] = 2) -> None:
    write_atomic(path, lambda handle: handle.write(json.dumps(obj, indent=indent).encode("utf-8")))


def save_npy_atomic(path: Path, array: np.ndarray) -> None:
    write_atomic(path, lambda handle: np.save(handle, array))


def read_lines(path: Path) -> List[str]:
    """The lines of a report file the way score_chexbert_standalone.py reads refs.txt: splitlines. The reports have had their whitespace
    collapsed to single spaces, so no line break can hide inside one."""
    return path.read_text(encoding="utf-8").splitlines()


def read_json_object(path: Path, code: str) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):                  # a UnicodeDecodeError is a ValueError
        raise Refused(code) from None
    if not isinstance(value, dict):
        raise Refused(code)
    return value


# ── the pure pieces ───────────────────────────────────────────────────────────

def representatives(groups: np.ndarray) -> np.ndarray:
    """First row of every duplicate group, in group-id order."""
    _, first = np.unique(groups, return_index=True)
    return first


def broadcast(rep_labels: np.ndarray, groups: np.ndarray, reps: np.ndarray) -> np.ndarray:
    """One row of labels per gallery row: that of its group's representative. `reps` is representatives(groups) and `rep_labels` the
    labels of those rows, in that order."""
    return rep_labels[np.searchsorted(groups[reps], groups)]


def rows_of_test_split(split: np.ndarray, split_row: np.ndarray, n_test: int) -> np.ndarray:
    """The gallery rows of the test reports in test.parquet order: the rows with txt_split == 1, taken in txt_split_row order, which must
    be 0 .. n_test-1 each once (the published refs.txt and chexbert_labels.json are in that order)."""
    rows = np.flatnonzero(split == 1)
    rows = rows[np.argsort(split_row[rows], kind="stable")]
    if len(rows) != n_test or not np.array_equal(split_row[rows], np.arange(n_test)):
        raise Refused("test_rows", found=len(rows), expected=n_test)
    return rows


def checked_row(values: Any, index: int) -> List[int]:
    """One labeller reply as 14 ints, each 0 or 1. Anything else means the labeller's API is not what this was written against, and it is
    refused instead of cast (a 257 would be a 1 in a uint8, a reply of 13 a broadcasting error)."""
    try:
        row = [int(v) for v in values]
    except (TypeError, ValueError):
        raise Refused("bad_label_row", row=index) from None
    if len(row) != N_LABELS or any(v not in (0, 1) for v in row):
        raise Refused("bad_label_row", row=index)
    return row


def label_rows(labeler: Any, texts: List[str], rows: Sequence[int], budget_s: float, canary: int = DEFAULT_CANARY,
               progress: int = DEFAULT_PROGRESS) -> np.ndarray:
    """The 14 labels of texts[r] for each r in rows, one get_label call each, in that order. After `canary` rows it prints the rate and
    what it projects for all of them, and exits 2 if that is past budget_s (a job shorter than the canary has none)."""
    out = np.zeros((len(rows), N_LABELS), dtype=np.uint8)
    t0 = time.perf_counter()
    for i, r in enumerate(rows):
        out[i] = checked_row(labeler.get_label(texts[r]), i)
        if i + 1 == canary:
            elapsed = time.perf_counter() - t0
            projected = elapsed / canary * len(rows)
            say("canary: {:.3f} s/report, projected {:.0f} s for {} rows".format(elapsed / canary, projected, len(rows)))
            if projected > budget_s:
                raise SystemExit(EXIT_SLOW)          # wrapper reads exit 2 as "use the sharded CPU path"
        if progress and (i + 1) % progress == 0:
            say("progress: {} of {} rows".format(i + 1, len(rows)))
    return out


def count_line_mismatches(mine: Sequence[str], theirs: Sequence[str]) -> int:
    """Lines that differ, line for line; a line one side has and the other has not counts as a difference."""
    return sum(a != b for a, b in zip(mine, theirs)) + abs(len(mine) - len(theirs))


def cross_check(texts: List[str], test_rows: np.ndarray, labels: np.ndarray, shared: np.ndarray, reference: Dict[str, Any]) -> Dict[str, Any]:
    """The two counts the plan asks for, over the test reports in test.parquet order: refs_mismatch, the report_texts lines that differ from the
    dump's refs.txt, and labels_mismatch, the reports with any of the 14 labels different from the dump's y_true. Of the second, how many are
    reports that took their labels from another row's text (`shared`) and how many their own: what tells the group argument failing from
    the labeller itself differing between the two runs (a GPU against the CPU the dump was labelled on)."""
    differs = (labels[test_rows] != reference["y_true"]).any(axis=1)
    return {"stage": "final", "reference": reference["name"], "n_test": int(len(test_rows)),
            "refs_mismatch": count_line_mismatches([texts[r] for r in test_rows], reference["refs"]),
            "labels_mismatch": int(differs.sum()),
            "labels_mismatch_own_text": int((differs & ~shared).sum()), "labels_mismatch_shared_text": int((differs & shared).sum()),
            "test_rows_sharing_a_text": int(shared.sum()), "label_names_source": reference["names_source"]}


# ── the gallery and the reference ─────────────────────────────────────────────

def check_gallery(gallery: Path) -> Dict[str, Any]:
    """The manifest of a gallery that may be labelled: built (manifest.json), verified (gate_rk.equal true in the manifest and in gate_rk.json,
    the retrieval chapter's own recalls reproduced) and not labelled yet (R8: a finished labelling is never overwritten)."""
    if not (gallery / "manifest.json").is_file():
        raise Refused("no_manifest")
    manifest = read_json_object(gallery / "manifest.json", "manifest_unreadable")
    gate_file = None
    try:
        gate_file = read_json_object(gallery / "gate_rk.json", "gate_not_equal")
    except Refused:
        pass                                       # missing or unreadable: not verified
    for gate in (manifest.get("gate_rk"), gate_file):
        if not isinstance(gate, dict) or gate.get("equal") is not True:
            raise Refused("gate_not_equal")
    if manifest.get("labels_status") == "done":
        raise Refused("already_labelled")
    return manifest


def refuse_if_done(gallery: Path) -> Dict[str, Any]:
    """The manifest as it is on disk NOW, or already_labelled if the gallery was finished since this run began: a GPU job and a CPU merge, or
    a resubmission, can both pass the guards of a pending gallery, and a finished labelling is never overwritten (R8). What a run writes
    into the manifest is its own fields on a read taken here, never on the copy it took at its start."""
    manifest = read_json_object(gallery / "manifest.json", "manifest_unreadable")
    if manifest.get("labels_status") == "done":
        raise Refused("already_labelled")
    return manifest


def load_inputs(gallery: Path, manifest: Dict[str, Any]) -> Dict[str, Any]:
    """The texts, the duplicate groups, their representatives and the gallery rows of the test reports; every count checked against the manifest's."""
    try:
        texts = read_lines(gallery / "report_texts.txt")
        groups = np.load(str(gallery / "txt_groups.npy"), allow_pickle=False)
        split = np.load(str(gallery / "txt_split.npy"), allow_pickle=False)
        split_row = np.load(str(gallery / "txt_split_row.npy"), allow_pickle=False)
    except (OSError, ValueError, EOFError):          # np.load answers a file of 0 bytes with EOFError, which is neither of the others
        raise Refused("inputs_unreadable") from None
    if groups.ndim != 1 or split.ndim != 1 or split_row.ndim != 1:
        raise Refused("inputs_unreadable")
    counts = manifest.get("counts") if isinstance(manifest.get("counts"), dict) else {}
    n_rows, n_test, n_groups = (whole(counts.get(key)) for key in ("report_rows", "test", "report_groups"))
    if not len(texts) == len(groups) == len(split) == len(split_row) == n_rows:
        raise Refused("rows_disagree", texts=len(texts), groups=len(groups), manifest=n_rows)
    reps = representatives(groups)
    if len(reps) != n_groups:
        raise Refused("groups_disagree", found=len(reps), manifest=n_groups)
    return {"texts": texts, "groups": groups, "reps": reps, "test_rows": rows_of_test_split(split, split_row, n_test)}


def load_reference(directory: Path, n_test: int) -> Dict[str, Any]:
    """The published dump's reference side: refs.txt, and the y_true of chexbert_labels.json as an (n_test, 14) matrix in CHEXBERT_14 order.
    The file comes in two formats, with a `label_names` key (the columns are put in CHEXBERT_14 order by name) and without (CHEXBERT_14
    order is assumed, and a note says so)."""
    try:
        refs = read_lines(directory / "refs.txt")
        payload = json.loads((directory / "chexbert_labels.json").read_text(encoding="utf-8"))
        y_true = np.array(payload["y_true"], dtype=np.int64)
        has_names = "label_names" in payload
        names = payload.get("label_names")
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        raise Refused("reference_unreadable") from None
    rows = int(y_true.shape[0]) if y_true.ndim == 2 else 0
    width = int(y_true.shape[1]) if y_true.ndim == 2 else 0
    if y_true.ndim != 2 or rows != n_test or width != N_LABELS:
        raise Refused("reference_shape", rows=rows, expected=n_test, width=width)
    source = "assumed"
    if not has_names:
        note("chexbert_labels.json has no label_names key: CHEXBERT_14 order assumed")
    else:
        if not (isinstance(names, list) and len(names) == N_LABELS and all(isinstance(n, str) for n in names)
                and sorted(names) == sorted(CHEXBERT_14)):
            raise Refused("reference_label_names")
        if names == CHEXBERT_14:
            source = "dump"
        else:
            y_true = y_true[:, [names.index(n) for n in CHEXBERT_14]]
            source = "reordered"
            note("chexbert_labels.json label_names reordered to the CHEXBERT_14 order")
    return {"name": directory.name, "refs": refs, "y_true": y_true, "names_source": source}


def preflight_refs(gallery: Path, data: Dict[str, Any], reference: Dict[str, Any]) -> None:
    """The text half of the cross-check, ahead of the labelling: report_texts.txt's test lines against refs.txt, line for line. It is pure text,
    so a difference is known in a second instead of after hours of labelling. The check at the end repeats it and adds the labels."""
    mismatch = count_line_mismatches([data["texts"][r] for r in data["test_rows"]], reference["refs"])
    if mismatch:
        write_json_atomic(gallery / "labels_check.json", {"stage": "preflight", "reference": reference["name"], "n_test": int(len(data["test_rows"])),
                                                          "refs_mismatch": mismatch, "labels_mismatch": None})
        raise Refused("refs_mismatch", n=mismatch)


# ── the labeller ──────────────────────────────────────────────────────────────

def init_args(cls: Any) -> List[str]:
    """The argument names of the constructor, without self, kept to a printable shape: names only, never a default's value."""
    try:
        names = list(inspect.signature(cls.__init__).parameters)[1:]
    except (TypeError, ValueError):
        return []
    return [re.sub(r"[^A-Za-z0-9_]", "_", n)[:32] for n in names][:8]


def model_device(labeler: Any) -> str:
    """The device of the first parameter of the labeller's model (`cpu`, `cuda:0`, ...), or `unknown` if it cannot be told."""
    for holder in (getattr(labeler, "model", None), labeler):
        try:
            text = str(next(iter(holder.parameters())).device)
        except (AttributeError, StopIteration, TypeError):
            continue
        return text if re.fullmatch(r"cpu|cuda(:[0-9]{1,2})?", text) else "unknown"
    return "unknown"


def make_labeler(device: str) -> Any:
    """F1CheXbert, with a `device` argument only if one was asked for (cpu or cuda; auto leaves the library's choice, which is the GPU when
    torch sees one). Prints the constructor's argument names and the device the model's first parameter ended up on."""
    from f1chexbert import F1CheXbert
    names = init_args(F1CheXbert)
    if names:
        say("init_args=" + ",".join(names))
    kwargs = {}     # type: Dict[str, str]
    if device != "auto":
        if "device" not in inspect.signature(F1CheXbert.__init__).parameters:
            raise Refused("no_device_argument")
        kwargs["device"] = device
    labeler = F1CheXbert(**kwargs)
    say("device=" + model_device(labeler))
    return labeler


def check_label_names(labeler: Any) -> None:
    """The labeller's own names must be the 14 in this order, before it labels anything: labels.npy's columns mean what the plan says they do."""
    try:
        names = [str(n) for n in labeler.target_names]
    except (AttributeError, TypeError):
        names = []
    if names != CHEXBERT_14:
        raise Refused("label_order")


# ── shards ────────────────────────────────────────────────────────────────────

def shard_paths(gallery: Path, index: int) -> Tuple[Path, Path]:
    """(labels, rows): the labels file is written last, so its presence marks a finished shard."""
    return gallery / "labels_shard_{}.npy".format(index), gallery / "labels_shard_{}_rows.npy".format(index)


def read_shard(gallery: Path, index: int, expected_rows: np.ndarray) -> Tuple[str, Optional[np.ndarray]]:
    """('ok', labels) if both files of the shard are there and are exactly the shard asked for: its rows, int64 and equal to expected_rows, and
    uint8 labels of the right shape holding only 0 and 1; ('missing', None) if a file is absent; ('invalid', None) otherwise."""
    labels_path, rows_path = shard_paths(gallery, index)
    if not labels_path.is_file() or not rows_path.is_file():
        return "missing", None
    try:
        rows = np.load(str(rows_path), allow_pickle=False)
        labels = np.load(str(labels_path), allow_pickle=False)
    except (OSError, ValueError, EOFError):          # a shard cut to 0 bytes is damaged like any other: labelled again, never a crash
        return "invalid", None
    good = (rows.dtype == np.int64 and np.array_equal(rows, expected_rows) and labels.dtype == np.uint8
            and labels.shape == (len(expected_rows), N_LABELS) and bool((labels <= 1).all()))
    return ("ok", labels) if good else ("invalid", None)


def merge_shards(gallery: Path, reps: np.ndarray, of: int) -> np.ndarray:
    """The labels of every representative from the N shards: shard i holds exactly reps[i::N], so it fills rows i, i+N, ... of the result.
    A missing shard is named first (its job can simply be run), then a shard that is not the one it should be."""
    found = [read_shard(gallery, i, reps[i::of]) for i in range(of)]
    missing = [i for i, (status, _) in enumerate(found) if status == "missing"]
    if missing:
        raise Refused("shards_missing", count=len(missing), first=missing[0])
    invalid = [i for i, (status, _) in enumerate(found) if status == "invalid"]
    if invalid:
        raise Refused("shard_invalid", shard=invalid[0])
    rep_labels = np.zeros((len(reps), N_LABELS), dtype=np.uint8)
    for i, (_, labels) in enumerate(found):
        rep_labels[i::of] = labels
    return rep_labels


def run_shard(args: argparse.Namespace, gallery: Path, data: Dict[str, Any]) -> int:
    index, of = args.shard, args.of
    rows = data["reps"][index::of]
    if read_shard(gallery, index, rows)[0] == "ok":
        say("shard {} of {} kept: already labelled".format(index, of))       # a finished shard is never labelled again (a requeue, a rerun)
        return 0
    labeler = make_labeler(args.device)
    check_label_names(labeler)
    started = time.perf_counter()
    labels = label_rows(labeler, data["texts"], rows, args.budget_s, args.canary, args.progress)
    labelling_s = int(round(time.perf_counter() - started))
    labels_path, rows_path = shard_paths(gallery, index)
    save_npy_atomic(rows_path, np.asarray(rows, dtype=np.int64))
    say("wrote " + rows_path.name)
    save_npy_atomic(labels_path, labels)
    say("wrote " + labels_path.name)
    emit_result({"shard": index, "of": of, "groups": int(len(rows)), "labelling_s": labelling_s})
    return 0


# ── finishing: the cross-check, then the files ────────────────────────────────

def package_version(name: str) -> Optional[str]:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None


def finish(gallery: Path, data: Dict[str, Any], rep_labels: np.ndarray, reference: Dict[str, Any],
           mode: str, shards: Optional[int], device: Optional[str], labelling_s: Optional[int]) -> int:
    """Broadcast the groups' labels to every row, run the cross-check, write labels_check.json, and only if the check found no difference
    write labels.npy and label_names.json and flip the manifest's labels_status to done. Otherwise the labels are kept as
    labels_unverified.npy, labels_status stays pending, and the exit code is 1. R8 is checked again here, before anything is written: if
    the gallery was finished while this run labelled, it refuses (already_labelled) and replaces nothing. The manifest is written from a
    fresh read (see refuse_if_done), so that nothing added to it since this run began is lost. A window of milliseconds remains between that
    read and the write; closing it would take a lock file, which a job killed in the middle would leave behind for the next one."""
    refuse_if_done(gallery)
    texts, groups, reps, test_rows = data["texts"], data["groups"], data["reps"], data["test_rows"]
    labels = broadcast(rep_labels, groups, reps)
    shared = reps[np.searchsorted(groups[reps], groups)][test_rows] != test_rows        # labelled from another row's text
    check = cross_check(texts, test_rows, labels, shared, reference)
    write_json_atomic(gallery / "labels_check.json", check)
    say("wrote labels_check.json")
    say("test_rows_sharing_a_text={}".format(check["test_rows_sharing_a_text"]))      # how much of a zero below is evidence for the group argument
    emit_result({"refs_mismatch": check["refs_mismatch"], "labels_mismatch": check["labels_mismatch"]})
    if check["refs_mismatch"] or check["labels_mismatch"]:
        if check["labels_mismatch"]:
            say("mismatch split: own_text={} shared_text={}".format(check["labels_mismatch_own_text"], check["labels_mismatch_shared_text"]))
        save_npy_atomic(gallery / "labels_unverified.npy", labels)
        say("wrote labels_unverified.npy")
        say("labels_status=pending")
        return EXIT_REFUSED
    save_npy_atomic(gallery / "labels.npy", labels)
    say("wrote labels.npy")
    write_json_atomic(gallery / "label_names.json", CHEXBERT_14, indent=None)      # equal to labeler.target_names: every labelling run asserts it
    say("wrote label_names.json")
    done = dict(refuse_if_done(gallery))
    done["labels_status"] = "done"
    done["labels_info"] = {
        "labeler": "f1chexbert", "f1chexbert_version": package_version("f1chexbert"), "mode": mode, "shards": shards, "device": device,
        "groups_labelled": int(len(reps)), "rows": int(len(texts)), "labelling_s": labelling_s, "job_id": os.environ.get("SLURM_JOB_ID"),
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "reference": reference["name"],
        "check": {"n_test": check["n_test"], "refs_mismatch": check["refs_mismatch"], "labels_mismatch": check["labels_mismatch"]}}
    write_json_atomic(gallery / "manifest.json", done)
    say("labels_status=done")
    return 0


# ── command line ──────────────────────────────────────────────────────────────

class UsageParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        """argparse exits 2 on a bad command line, which is the canary's code: the wrapper would call a typo "too slow"."""
        self.print_usage(sys.stderr)
        self.exit(EXIT_USAGE, "{}: error: {}\n".format(self.prog, message))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = UsageParser(description=__doc__.splitlines()[0])
    p.add_argument("--gallery", required=True, help="the gallery directory (CHAT_HOME/gallery/<build id>)")
    p.add_argument("--reference-dir", default=DEFAULT_REFERENCE_DIR,
                   help="the published dump with refs.txt and chexbert_labels.json (default: %(default)s)")
    p.add_argument("--budget-s", type=float, default=DEFAULT_BUDGET_S, help="exit 2 if the canary projects more seconds (default: %(default)s)")
    p.add_argument("--canary", type=int, default=DEFAULT_CANARY, help="reports to label before projecting; 0 for no canary (default: %(default)s)")
    p.add_argument("--progress", type=int, default=DEFAULT_PROGRESS, help="print progress every this many reports; 0 for none (default: %(default)s)")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                   help="auto: pass the labeller no device argument (the library takes the GPU if torch sees one) (default: %(default)s)")
    p.add_argument("--shard", type=int, default=None, help="label only reps[SHARD::OF] (needs --of)")
    p.add_argument("--of", type=int, default=None, help="the number of shards")
    p.add_argument("--merge", action="store_true", help="combine the --of shards (needs --of)")
    args = p.parse_args(argv)
    if args.shard is not None and args.merge:
        p.error("--shard and --merge are separate modes")
    if (args.shard is not None or args.merge) and args.of is None:
        p.error("--shard and --merge need --of")
    if args.of is not None and args.shard is None and not args.merge:
        p.error("--of is for --shard or --merge")
    if args.of is not None and args.of < 1:
        p.error("--of must be at least 1")
    if args.shard is not None and not 0 <= args.shard < args.of:
        p.error("--shard must be from 0 to --of minus 1")
    if args.canary < 0 or args.progress < 0 or args.budget_s < 0:
        p.error("--canary, --progress and --budget-s cannot be negative")
    return args


def run(args: argparse.Namespace) -> int:
    gallery = Path(args.gallery)
    if args.shard is not None:
        say("mode=shard index={} of={}".format(args.shard, args.of))
    elif args.merge:
        say("mode=merge of={}".format(args.of))
    else:
        say("mode=single")
    manifest = check_gallery(gallery)
    data = load_inputs(gallery, manifest)
    say("rows={} groups={} test={}".format(len(data["texts"]), len(data["reps"]), len(data["test_rows"])))
    reference = load_reference(Path(args.reference_dir), len(data["test_rows"]))
    preflight_refs(gallery, data, reference)
    if args.shard is not None:
        return run_shard(args, gallery, data)
    if args.merge:
        rep_labels = merge_shards(gallery, data["reps"], args.of)
        emit_result({"merged": args.of, "groups": int(len(data["reps"])), "rows": int(len(data["texts"]))})
        return finish(gallery, data, rep_labels, reference, "sharded", args.of, None, None)
    labeler = make_labeler(args.device)
    check_label_names(labeler)
    device = model_device(labeler)
    started = time.perf_counter()
    rep_labels = label_rows(labeler, data["texts"], data["reps"], args.budget_s, args.canary, args.progress)
    labelling_s = int(round(time.perf_counter() - started))
    emit_result({"groups": int(len(data["reps"])), "rows": int(len(data["texts"])), "labelling_s": labelling_s})
    return finish(gallery, data, rep_labels, reference, "single", None, device, labelling_s)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except Refused as err:
        print(err.line(), flush=True)
        return EXIT_REFUSED
    except Exception as exc:       # not a reason of ours: a bug, or the labeller failing. The class name is the log's; the message and the traceback may hold text
        traceback.print_exc()
        print("ERROR failed " + type(exc).__name__, flush=True)
        return EXIT_REFUSED


if __name__ == "__main__":
    sys.exit(main())
