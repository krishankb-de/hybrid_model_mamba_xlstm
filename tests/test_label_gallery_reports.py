"""P5-C (CHAT_UI_PLAN.md): CheXbert-14 labels for the retrieval gallery (scripts/label_gallery_reports.py) and its two wrappers:
scripts/label_gallery_reports_h100.sh (one GPU job, with a canary) and scripts/label_gallery_reports_cpu_h100.sh (the sharded CPU
fallback and its merge).

Laptop work: CPU, offline, synthetic data only (R7). The real labeller needs the CheXbert weights, so a FAKE `f1chexbert` module stands
in for it: the constructor signature of f1chexbert 0.0.2, a `get_label` that returns 14 ints which are a pure function of the lower-cased,
whitespace-collapsed text (what the uncased BERT tokenizer makes of the real one), a `model` whose first parameter says which device it
sits on, and knobs in the environment (FAKE_F1_*) for what a test has to provoke: a slow labeller, one that dies, one that returns a bad
row, one with another label order, one that chatters. The gallery is P5-B's `--tiny` one. tests/conftest.py's tiny_gallery fixture
builds a finished (labelled) gallery; the labelling needs an unlabelled one, so a module-level template is built with
`with_labels=False` and its gate marked equal, as `--compare-rk` leaves a verified build.
Tested in layers:
  * the pure pieces: representatives, broadcasting, label_rows and its canary, the order of the test rows;
  * main() in process over the tiny gallery: the guards, the single job, shards and merge, the cross-check's verdicts, what is printed;
  * both wrappers, rehearsed in a temp tree (tests/wrapper_rehearsal.py), with a stub python that records the calls and runs the REAL
    script, which imports the fake f1chexbert from PYTHONPATH.
"""
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import time
import types
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import pytest

from scripts import build_retrieval_gallery as bg
from tests import wrapper_rehearsal as wr
from tests.test_chat_remote import STAMP_RE
from tests.wrapper_rehearsal import STAMP, job_lines, real_stamp, results, snapshot

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "label_gallery_reports.py"
GPU_SH = "label_gallery_reports_h100.sh"
CPU_SH = "label_gallery_reports_cpu_h100.sh"
LINE_OK = wr.safe_line("labels")      # the first words of every line a job may print
REFERENCE_NAME = "report_gen_m3_test_split_s42"
# The label order, written out here once more on purpose: the pin is that every copy of it agrees.
NAMES = ["Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema", "Consolidation", "Pneumonia",
         "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices", "No Finding"]


# ── the fake f1chexbert ───────────────────────────────────────────────────────

FAKE_F1CHEXBERT = '''\
"""A fake f1chexbert (tests/test_label_gallery_reports.py): the constructor and get_label of f1chexbert 0.0.2, with no model."""
import atexit
import hashlib
import os
import sys
import time
from types import SimpleNamespace

TARGET_NAMES = ["Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema", "Consolidation", "Pneumonia",
                "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices", "No Finding"]
CONSTRUCTED = []     # one dict per construction: the device argument it was given
CALLS = []           # the text of every get_label call


def label_of(text):
    """14 ints, a pure function of the lower-cased, whitespace-collapsed text: what the uncased tokenizer makes of the real labeller."""
    text = " ".join(text.split())
    if not os.environ.get("FAKE_F1_CASE_SENSITIVE"):
        text = text.lower()
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [digest[i] & 1 for i in range(14)]


def _mark(line):
    path = os.environ.get("FAKE_F1_MARK")
    if path:
        with open(path, "a") as handle:
            handle.write(line + "\\n")


atexit.register(lambda: _mark("calls=%d" % len(CALLS)))


class _Parameter(object):
    def __init__(self, device):
        self.device = device


class _Base(object):
    def _build(self, device):
        CONSTRUCTED.append({"device": device})
        _mark("constructed device=%s hf_home=%s offline=%s" % (
            device, os.environ.get("HF_HOME", "unset"), os.environ.get("HF_HUB_OFFLINE", "unset")))
        self.device = device if device is not None else os.environ.get("FAKE_F1_DEFAULT_DEVICE", "cpu")
        self.model = SimpleNamespace(parameters=lambda: iter([_Parameter(self.device)]))
        self.target_names = list(TARGET_NAMES)
        if os.environ.get("FAKE_F1_NAMES") == "swapped":
            self.target_names[0], self.target_names[1] = self.target_names[1], self.target_names[0]
        if os.environ.get("FAKE_F1_PROGRESS"):
            # a progress bar, which ends in a carriage return and no newline: the next line printed to the same file continues its physical line
            sys.stderr.write("Loading weights:  50%|#####     | 1/2\\r")
            sys.stderr.flush()
        if os.environ.get("FAKE_F1_JUNK"):
            # what a library prints while it loads: text, an id and a path on lines that look like ours, and a traceback
            print("[labels] study_id=12345678")
            print("[labels] The heart is mildly enlarged and there is a small pleural effusion.")
            print("RESULT {\\"study_id\\":50000001}")
            print("ERROR the heart is mildly enlarged")
            print("=== FAKE MIMIC TEXT /sc/home/someone/images/p10/img.jpg ===")
            print("Findings: FAKE REPORT TEXT study_id=12345678", file=sys.stderr)

    def get_label(self, report, mode="rrg"):
        CALLS.append(report)
        limit = os.environ.get("FAKE_F1_DIE_AFTER")
        if limit is not None and len(CALLS) > int(limit):
            raise RuntimeError("fake labeller died on: " + report)       # a real traceback can hold report text like this
        pause = os.environ.get("FAKE_F1_SLEEP")
        if pause:
            time.sleep(float(pause))
        row = label_of(report)
        bad = os.environ.get("FAKE_F1_BAD_ROW")
        if bad is not None and len(CALLS) == int(bad):
            return row[:13]
        return row


if os.environ.get("FAKE_F1_NO_DEVICE_ARG"):
    class F1CheXbert(_Base):
        def __init__(self, refs_filename=None, hyps_filename=None):
            self._build(None)
else:
    class F1CheXbert(_Base):
        def __init__(self, refs_filename=None, hyps_filename=None, device=None, **kwargs):
            self._build(device)
'''


def load_fake() -> types.ModuleType:
    """A fresh copy of the fake (its record lists start empty). Knobs read at load time: FAKE_F1_NO_DEVICE_ARG."""
    module = types.ModuleType("f1chexbert")
    exec(compile(FAKE_F1CHEXBERT, "<fake f1chexbert>", "exec"), module.__dict__)
    return module


FAKE = load_fake()      # for label_of(): the reference dump's labels are the fake labeller's own


@pytest.fixture
def install_fake(monkeypatch):
    """install_fake(**env) -> the fake module, now what `import f1chexbert` finds, with the environment knobs set first."""
    def install(**env: str) -> types.ModuleType:
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        module = load_fake()
        monkeypatch.setitem(sys.modules, "f1chexbert", module)
        return module
    return install


@pytest.fixture
def fake(install_fake):
    return install_fake()


@pytest.fixture(scope="module")
def lg():
    """scripts/label_gallery_reports.py"""
    from scripts import label_gallery_reports
    return label_gallery_reports


# ── the tiny gallery, the reference dump, and main() in process ───────────────

def verify_gate(gallery: Path, equal: bool = True) -> None:
    """gate_rk.equal in manifest.json and in gate_rk.json, as `build_retrieval_gallery.py --compare-rk` leaves them."""
    for name in ("gate_rk.json", "manifest.json"):
        path = gallery / name
        data = json.loads(path.read_text())
        (data if name == "gate_rk.json" else data["gate_rk"])["equal"] = equal
        path.write_text(json.dumps(data, indent=2))


@pytest.fixture(scope="module")
def template(tmp_path_factory):
    out = tmp_path_factory.mktemp("template") / "gallery"
    bg.build_tiny(out, with_labels=False)
    verify_gate(out)
    return out


def read_texts(gallery: Path) -> List[str]:
    return (gallery / "report_texts.txt").read_text().splitlines()


def load_array(gallery: Path, name: str) -> np.ndarray:
    return np.load(str(gallery / name))


def gallery_rows_of_test_reports(gallery: Path) -> np.ndarray:
    """The gallery rows of the test reports in test.parquet order, from txt_split and txt_split_row."""
    split, split_row = load_array(gallery, "txt_split.npy"), load_array(gallery, "txt_split_row.npy")
    rows = np.flatnonzero(split == 1)
    return rows[np.argsort(split_row[rows], kind="stable")]


def write_reference(directory: Path, gallery: Path, names: bool = True) -> Path:
    """The published dump's two files as the scoring job writes them: refs.txt, and chexbert_labels.json with the y_true that the labeller
    gives those lines (here the fake's)."""
    texts = read_texts(gallery)
    refs = [texts[r] for r in gallery_rows_of_test_reports(gallery)]
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "refs.txt").write_text("\n".join(refs) + "\n")
    y_true = [FAKE.label_of(t) for t in refs]
    payload = {"y_true": y_true, "y_pred": y_true, "five_label_indices": [1, 4, 5, 7, 9], "hyp_file": "hyps.txt", "ref_file": "refs.txt"}
    if names:
        payload["label_names"] = list(NAMES)
    (directory / "chexbert_labels.json").write_text(json.dumps(payload))
    return directory


def read_reference(directory: Path) -> dict:
    return json.loads((directory / "chexbert_labels.json").read_text())


def write_reference_payload(directory: Path, payload: dict) -> None:
    (directory / "chexbert_labels.json").write_text(json.dumps(payload))


class World:
    """An unlabelled, verified tiny gallery of its own, with the reference dump beside it."""

    def __init__(self, root: Path, template: Path):
        root.mkdir(parents=True, exist_ok=True)
        self.gallery = root / "gallery"
        shutil.copytree(str(template), str(self.gallery))
        self.reference = write_reference(root / REFERENCE_NAME, self.gallery)

    def argv(self, *extra: str, device: str = "auto") -> List[str]:
        return ["--gallery", str(self.gallery), "--reference-dir", str(self.reference), "--device", device] + list(extra)

    def manifest(self) -> dict:
        return json.loads((self.gallery / "manifest.json").read_text())

    def groups(self) -> np.ndarray:
        return load_array(self.gallery, "txt_groups.npy")

    def reps(self) -> np.ndarray:
        return np.unique(self.groups(), return_index=True)[1]


@pytest.fixture
def world(tmp_path, template):
    return World(tmp_path / "world", template)


class Ran:
    def __init__(self, rc, out: str, err: str):
        self.rc, self.out, self.err = rc, out, err

    @property
    def lines(self) -> List[str]:
        return self.out.splitlines()


def run_main(lg, capsys, argv: List[str]) -> Ran:
    """main() in process: its exit code (a SystemExit's code too) and what it printed."""
    try:
        rc = lg.main(argv)
    except SystemExit as stop:
        rc = stop.code
    captured = capsys.readouterr()
    return Ran(rc, captured.out, captured.err)


def changed_paths(before: list, after: list) -> Set[str]:
    old, new = {p: (s, m) for p, s, m in before}, {p: (s, m) for p, s, m in after}
    return {p for p in set(old) | set(new) if old.get(p) != new.get(p)}


# ── the pure pieces: representatives, broadcasting, test rows ─────────────────

def test_representatives_are_the_first_row_of_every_group_in_group_id_order(lg):
    groups = np.array([5, 2, 5, 9, 2, 7])
    first = lg.representatives(groups)
    assert first.tolist() == [1, 0, 5, 3]                      # ids 2, 5, 7, 9: their first rows are 1, 0, 5, 3
    assert groups[first].tolist() == [2, 5, 7, 9]


def test_representatives_of_random_groups_are_each_groups_least_row(lg):
    rng = np.random.default_rng(7)
    for _ in range(20):
        groups = rng.integers(0, 40, size=int(rng.integers(1, 300)))
        first = lg.representatives(groups)
        assert len(first) == len(np.unique(groups)) and (np.diff(groups[first]) > 0).all(), "one per group, in group-id order"
        for group, row in zip(groups[first].tolist(), first.tolist()):
            assert row == int(np.flatnonzero(groups == group)[0])


def test_representatives_of_nothing_is_nothing(lg):
    assert lg.representatives(np.zeros(0, dtype=np.int64)).tolist() == []


def test_broadcast_gives_every_row_the_labels_of_its_groups_representative(lg):
    groups = np.array([5, 2, 5, 9, 2, 7])
    reps = lg.representatives(groups)
    rep_labels = np.array([[g] * 14 for g in range(4)], dtype=np.uint8)           # group-id order: ids 2, 5, 7, 9 -> 0, 1, 2, 3
    labels = lg.broadcast(rep_labels, groups, reps)
    assert labels.dtype == np.uint8 and labels.shape == (6, 14)
    assert labels[:, 0].tolist() == [1, 0, 1, 3, 0, 2]


def test_the_test_rows_are_taken_in_txt_split_row_order(lg):
    split = np.array([0, 0, 1, 1, 0, 1])
    split_row = np.array([0, 1, 2, 0, 3, 1])
    assert lg.rows_of_test_split(split, split_row, 3).tolist() == [3, 5, 2]       # test.parquet row 0 is gallery row 3, row 1 is 5, row 2 is 2


@pytest.mark.parametrize("split_row, n", [([0, 1, 2, 0, 3, 1], 4), ([0, 1, 2, 0, 3, 3], 3), ([0, 1, 2, 0, 3, 5], 3)])
def test_test_rows_that_are_not_each_row_of_the_test_split_once_are_refused(lg, split_row, n):
    split = np.array([0, 0, 1, 1, 0, 1])
    with pytest.raises(lg.Refused) as stop:
        lg.rows_of_test_split(split, np.array(split_row), n)
    assert stop.value.line().startswith("ERROR test_rows ")


def test_the_literal_label_names_are_the_apps_and_the_trainers_and_the_labellers(lg, fake):
    from app.labels import CHEXBERT_14
    tree = ast.parse((REPO_ROOT / "scripts" / "train_report_generation.py").read_text())
    trainer = [ast.literal_eval(n.value) for n in ast.walk(tree)
               if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "CHEXPERT_14_LABELS" for t in n.targets)]
    assert len(trainer) == 1
    assert lg.CHEXBERT_14 == CHEXBERT_14 == trainer[0] == NAMES == fake.F1CheXbert().target_names == bg.CHEXBERT_14


def f1chexbert_source() -> str:
    """The text of the installed f1chexbert's own module, read without importing it (importing would pull in torch and transformers, and
    building the class would download the CheXbert weights)."""
    import importlib.util
    spec = importlib.util.find_spec("f1chexbert")
    if spec is None or not spec.origin:
        pytest.skip("f1chexbert is not installed")
    return Path(spec.origin).with_name("f1chexbert.py").read_text()


def parse_quietly(source: str) -> ast.AST:
    """The library writes '\\s+' in a plain string: Python 3.12+ warns about it at parse time, which is the library's to fix, not a test failure."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        return ast.parse(source)


def test_the_literal_label_names_are_f1chexberts_own_read_from_its_source(lg):
    tree = parse_quietly(f1chexbert_source())
    found = [ast.literal_eval(n.value) for n in ast.walk(tree)
             if isinstance(n, ast.Assign) and any(isinstance(t, ast.Attribute) and t.attr == "target_names" for t in n.targets)]
    assert found == [lg.CHEXBERT_14], "the library's target_names, in its order, are the 14 this script asserts the labeller has"


def test_the_f1chexbert_api_the_script_and_the_wrappers_are_built_on_is_what_its_source_says(lg):
    """The device facts the docstrings state, and what make_labeler and label_rows call: a library upgrade that moves any of it fails here."""
    source = f1chexbert_source()
    tree = parse_quietly(source)
    cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "F1CheXbert")
    methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    init = methods["__init__"]
    assert [a.arg for a in init.args.args] == ["self", "refs_filename", "hyps_filename", "device"] and init.args.kwarg.arg == "kwargs"
    assert [ast.literal_eval(d) for d in init.args.defaults] == [None, None, None]
    body = ast.get_source_segment(source, init)
    assert "if device is None:" in body and "torch.cuda.is_available()" in body, "without a device argument the library takes the GPU if torch sees one"
    assert "self.model = " in body and "self.model.to(self.device)" in body, "the model whose first parameter model_device reads"
    get_label = methods["get_label"]
    assert [a.arg for a in get_label.args.args] == ["self", "report", "mode"] and [ast.literal_eval(d) for d in get_label.args.defaults] == ["rrg"]
    reply = ast.get_source_segment(source, get_label)
    assert "v = [1 if (isinstance(l, int) and l > 0) else 0 for l in v]" in reply and "return v" in reply, "a list of ints, 1 or 0, one per head"


def check_the_group_premise(source: str) -> None:
    """The two facts of the library that the group broadcast stands on, read from its own source: every report is tokenised with
    BertTokenizer from the checkpoint 'bert-base-uncased' (an uncased vocabulary, whose tokenizer lower-cases: two reports that differ in
    case have the same tokens), after its whitespace has been collapsed (so two that differ in spacing do too). AssertionError names
    the one that is gone. (That the checkpoint lower-cases is what its name says; its tokenizer_config.json is on the hub, not here.)"""
    tree = parse_quietly(source)
    assert "\nfrom transformers import BertTokenizer\n" in source, "the tokenizer is no longer transformers' BertTokenizer"
    cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "F1CheXbert")
    methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    init = ast.get_source_segment(source, methods["__init__"])
    assert "self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')" in init, "the tokenizer is no longer the uncased BERT one"
    tokenize = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "tokenize")
    body = ast.get_source_segment(source, tokenize)
    for needle, what in (("imp = impressions.str.strip()", "strip first"), ("imp = imp.replace('\\n', ' ', regex=True)", "newlines to spaces"),
                         ("imp = imp.replace('\\s+', ' ', regex=True)", "runs of whitespace to one space"),
                         ("impressions = imp.str.strip()", "strip last"), ("tokenizer.tokenize(impressions.iloc[i])", "the tokenizer is what tokenises")):
        assert needle in body, "tokenize() no longer does this: " + what
    assert "out = tokenize(impressions, self.tokenizer)" in ast.get_source_segment(source, methods["get_label"]), "get_label no longer goes through tokenize()"


def test_the_group_premise_is_in_f1chexberts_source_the_uncased_tokenizer_and_the_whitespace_collapse():
    """A library update that breaks the case or the whitespace argument fails here, loudly, before a labelling of 190k groups rests on it."""
    check_the_group_premise(f1chexbert_source())


@pytest.mark.parametrize("old, new", [
    ("BertTokenizer.from_pretrained('bert-base-uncased')", "BertTokenizer.from_pretrained('bert-base-cased')"),
    ("imp = imp.replace('\\s+', ' ', regex=True)", "imp = imp"),
    ("imp = imp.replace('\\n', ' ', regex=True)", "imp = imp"),
    ("imp = impressions.str.strip()", "imp = impressions"),
    ("tokenizer.tokenize(impressions.iloc[i])", "impressions.iloc[i].split()"),
    ("out = tokenize(impressions, self.tokenizer)", "out = [[0]]"),
    ("\nfrom transformers import BertTokenizer\n", "\nfrom transformers import BertTokenizerFast\n")])
def test_the_group_premise_pin_fails_when_the_library_stops_lowercasing_or_collapsing_whitespace(old, new):
    source = f1chexbert_source()
    assert old in source, "the text this test breaks is no longer in the library: re-read check_the_group_premise"
    with pytest.raises(AssertionError):
        check_the_group_premise(source.replace(old, new))


# ── label_rows and its canary ─────────────────────────────────────────────────

NO_REPLY = object()


class CountingLabeler:
    """The labeller as label_rows sees it: get_label(text) -> 14 ints, and a record of what it was asked. `reply` replaces the third answer."""

    def __init__(self, pause: float = 0.0, reply=NO_REPLY):
        self.asked, self.pause, self.reply = [], pause, reply

    def get_label(self, text, mode="rrg"):
        self.asked.append(text)
        if self.pause:
            time.sleep(self.pause)
        if self.reply is not NO_REPLY and len(self.asked) == 3:
            return self.reply
        return FAKE.label_of(text)


def texts_of(n: int) -> List[str]:
    return ["Findings: report number {}. Impression: none.".format(i) for i in range(n)]


def test_label_rows_asks_once_per_row_asked_for_and_returns_them_in_that_order(lg):
    texts = texts_of(20)
    labeler = CountingLabeler()
    out = lg.label_rows(labeler, texts, [7, 3, 19, 0], budget_s=1e9)
    assert labeler.asked == [texts[r] for r in (7, 3, 19, 0)]
    assert out.dtype == np.uint8 and out.shape == (4, 14)
    assert out.tolist() == [FAKE.label_of(texts[r]) for r in (7, 3, 19, 0)]


def test_the_canary_prints_its_one_line_and_exits_2_when_the_projection_is_over_budget(lg, capsys):
    labeler = CountingLabeler(pause=0.02)
    with pytest.raises(SystemExit) as stop:
        lg.label_rows(labeler, texts_of(60), list(range(60)), budget_s=0.0, canary=10)
    assert stop.value.code == 2
    assert len(labeler.asked) == 10, "it stops at the canary and not a row after it"
    (line,) = capsys.readouterr().out.splitlines()
    found = re.fullmatch(r"\[labels\] canary: ([0-9]+\.[0-9]{3}) s/report, projected ([0-9]+) s for 60 rows", line)
    assert found, line
    assert float(found.group(1)) >= 0.02 and int(found.group(2)) >= 1


def test_a_job_inside_its_budget_goes_on_after_the_canary(lg, capsys):
    labeler = CountingLabeler()
    out = lg.label_rows(labeler, texts_of(60), list(range(60)), budget_s=1e9, canary=10)
    assert len(labeler.asked) == 60 and out.shape == (60, 14)
    assert len([l for l in capsys.readouterr().out.splitlines() if "canary" in l]) == 1
    assert out[:, 0].tolist() == [FAKE.label_of(t)[0] for t in texts_of(60)]


def test_a_job_shorter_than_the_canary_has_none(lg, capsys):
    out = lg.label_rows(CountingLabeler(), texts_of(5), list(range(5)), budget_s=0.0, canary=10)      # a budget of 0 would stop at a canary
    assert out.shape == (5, 14) and capsys.readouterr().out == ""


def test_the_canary_is_the_exact_projection_it_prints(lg, capsys, monkeypatch):
    """Half a second for the first 10 of 40 rows: 0.050 s a report, 2 s for the 40."""
    ticks = iter([100.0])                                    # the start; every later reading is half a second on
    monkeypatch.setattr(lg, "time", types.SimpleNamespace(perf_counter=lambda: next(ticks, 100.5)))
    lg.label_rows(CountingLabeler(), texts_of(40), list(range(40)), budget_s=1e9, canary=10)
    assert capsys.readouterr().out == "[labels] canary: 0.050 s/report, projected 2 s for 40 rows\n"


def test_label_rows_prints_progress_at_its_interval(lg, capsys):
    lg.label_rows(CountingLabeler(), texts_of(25), list(range(25)), budget_s=1e9, canary=0, progress=10)
    assert capsys.readouterr().out.splitlines() == ["[labels] progress: 10 of 25 rows", "[labels] progress: 20 of 25 rows"]


@pytest.mark.parametrize("bad", [[0] * 13, [0] * 15, [0] * 13 + [2], [0] * 13 + [-1], [0] * 13 + [257], "not a row", None, []])
def test_a_reply_that_is_not_fourteen_zeros_and_ones_is_refused_not_cast(lg, bad):
    """A 257 would be a 1 in a uint8, a 13-long reply a broadcasting error: a labeller that changed its API is refused by row."""
    with pytest.raises(lg.Refused) as stop:
        lg.label_rows(CountingLabeler(reply=bad), texts_of(10), list(range(10)), budget_s=1e9)
    assert stop.value.line() == "ERROR bad_label_row row=2"


def test_label_rows_takes_an_integer_reply_in_any_integer_type(lg):
    class Numpy:
        def get_label(self, text, mode="rrg"):
            return np.array(FAKE.label_of(text), dtype=np.int64)
    out = lg.label_rows(Numpy(), texts_of(3), [0, 1, 2], budget_s=1e9)
    assert out.tolist() == [FAKE.label_of(t) for t in texts_of(3)]


# ── the guards: what a gallery must be before it is labelled ──────────────────

def test_a_gallery_without_a_manifest_is_refused_and_nothing_is_built(lg, fake, world, capsys):
    (world.gallery / "manifest.json").unlink()
    before = snapshot(world.gallery)
    ran = run_main(lg, capsys, world.argv())
    assert ran.rc == 1 and ran.lines[-1] == "ERROR no_manifest", ran.out
    assert snapshot(world.gallery) == before and fake.CONSTRUCTED == []


def test_a_manifest_that_is_not_a_json_object_is_refused(lg, fake, world, capsys):
    (world.gallery / "manifest.json").write_text("[1, 2]")
    ran = run_main(lg, capsys, world.argv())
    assert ran.rc == 1 and ran.lines[-1] == "ERROR manifest_unreadable" and fake.CONSTRUCTED == []


def break_gate(gallery: Path, how: str) -> None:
    manifest, gate_file = gallery / "manifest.json", gallery / "gate_rk.json"
    m, g = json.loads(manifest.read_text()), json.loads(gate_file.read_text())
    if how == "missing":
        del m["gate_rk"]["equal"], g["equal"]
    elif how == "false":
        m["gate_rk"]["equal"] = g["equal"] = False
    elif how == "text":
        m["gate_rk"]["equal"] = g["equal"] = "true"
    elif how == "manifest_only":
        del g["equal"]
    elif how == "gate_file_only":
        del m["gate_rk"]["equal"]
    elif how == "no_gate_key":
        del m["gate_rk"]
    if how == "no_gate_file":
        gate_file.unlink()
    else:
        gate_file.write_text(json.dumps(g, indent=2))
    manifest.write_text(json.dumps(m, indent=2))


@pytest.mark.parametrize("how", ["missing", "false", "text", "manifest_only", "gate_file_only", "no_gate_key", "no_gate_file"])
def test_a_gallery_whose_gate_is_not_equal_in_both_files_is_refused(lg, fake, world, capsys, how):
    """A build whose gate failed or never ran must not be labelled (it is not the retrieval chapter's gallery)."""
    break_gate(world.gallery, how)
    before = snapshot(world.gallery)
    ran = run_main(lg, capsys, world.argv())
    assert ran.rc == 1 and ran.lines[-1] == "ERROR gate_not_equal", ran.out
    assert snapshot(world.gallery) == before and fake.CONSTRUCTED == []


def test_a_finished_labelling_is_never_overwritten(lg, fake, tiny_gallery, tmp_path, capsys):
    """The fixture is P5-B's finished tiny gallery: labels.npy, label_names.json and labels_status done."""
    verify_gate(tiny_gallery)
    reference = write_reference(tmp_path / REFERENCE_NAME, tiny_gallery)
    before = snapshot(tiny_gallery)
    ran = run_main(lg, capsys, ["--gallery", str(tiny_gallery), "--reference-dir", str(reference)])
    assert ran.rc == 1 and ran.lines[-1] == "ERROR already_labelled", ran.out
    assert snapshot(tiny_gallery) == before and fake.CONSTRUCTED == []


def test_a_done_status_alone_is_enough_to_refuse(lg, fake, world, capsys):
    manifest = world.manifest()
    manifest["labels_status"] = "done"
    (world.gallery / "manifest.json").write_text(json.dumps(manifest))
    assert not (world.gallery / "labels.npy").exists()
    ran = run_main(lg, capsys, world.argv())
    assert ran.rc == 1 and ran.lines[-1] == "ERROR already_labelled" and fake.CONSTRUCTED == []


def test_a_partial_run_may_be_completed(lg, fake, world, capsys):
    """Shards and an unverified result left by an earlier attempt do not stop the next one: labels_status is still pending."""
    (world.gallery / "labels_shard_0.npy").write_bytes(b"left over")
    (world.gallery / "labels_unverified.npy").write_bytes(b"left over")
    (world.gallery / "labels.log").write_text("earlier attempt\n")
    ran = run_main(lg, capsys, world.argv())
    assert ran.rc == 0, ran.out
    assert (world.gallery / "labels.npy").is_file() and world.manifest()["labels_status"] == "done"


def mark_done_by_another_run(gallery: Path) -> bytes:
    """The manifest a run that finished the gallery first leaves: done, with its own labels_info. Returns its bytes."""
    manifest = json.loads((gallery / "manifest.json").read_text())
    manifest["labels_status"] = "done"
    manifest["labels_info"] = {"job_id": "the other run"}
    (gallery / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return (gallery / "manifest.json").read_bytes()


def simulate_another_run_finishing(gallery: Path) -> Dict[str, bytes]:
    """What a run that began later and ended first leaves: its labels, its names, its check, and a manifest that says done. The bytes
    are sentinels for its files: what matters is that this run replaces none of them. Returns every file with its bytes."""
    left = {"labels.npy": b"the other run's labels", "label_names.json": b'["the other run"]', "labels_check.json": b'{"the other run": true}'}
    for name, blob in left.items():
        (gallery / name).write_bytes(blob)
    left["manifest.json"] = mark_done_by_another_run(gallery)
    return left


@pytest.mark.parametrize("mode", ["single", "merge"])
def test_a_run_that_finds_the_gallery_finished_meanwhile_refuses_and_replaces_nothing(lg, fake, world, capsys, monkeypatch, mode):
    """R8 again at the end of the run, not only at its start: a GPU job and a CPU merge, or a resubmission, can both pass the guards of a
    pending gallery, and the one that finishes second must not replace what the first left."""
    argv, target = world.argv(), "label_rows"
    if mode == "merge":
        for i in range(2):
            assert run_main(lg, capsys, world.argv("--shard", str(i), "--of", "2")).rc == 0
        argv, target = world.argv("--merge", "--of", "2"), "merge_shards"
    real, left = getattr(lg, target), {}

    def overtaken(*args, **kwargs):
        result = real(*args, **kwargs)                          # the slow part is done; meanwhile the other run has finished
        left.update(simulate_another_run_finishing(world.gallery))
        return result
    monkeypatch.setattr(lg, target, overtaken)
    ran = run_main(lg, capsys, argv)
    assert ran.rc == 1 and ran.lines[-1] == "ERROR already_labelled", ran.out
    assert left, "the other run did finish in the middle of this one"
    for name, blob in left.items():
        assert (world.gallery / name).read_bytes() == blob, name + " was replaced"
    assert not (world.gallery / "labels_unverified.npy").exists() and "[labels] wrote labels.npy" not in ran.lines


def test_a_run_overtaken_in_the_last_moment_leaves_the_other_runs_manifest_as_it_is(lg, fake, world, capsys, monkeypatch):
    """The window between the end-of-run check and the manifest write is small but not nothing: the manifest is read once more for the write,
    so a gallery that turned done in that window keeps the other run's manifest (its labels_info is not replaced by this run's)."""
    real, left = lg.write_json_atomic, {}

    def interleaved(path, obj, indent=2):
        real(path, obj, indent)
        if Path(path).name == "label_names.json":               # this run's last file before the manifest
            left["manifest.json"] = mark_done_by_another_run(world.gallery)
    monkeypatch.setattr(lg, "write_json_atomic", interleaved)
    ran = run_main(lg, capsys, world.argv())
    assert ran.rc == 1 and ran.lines[-1] == "ERROR already_labelled", ran.out
    assert left and (world.gallery / "manifest.json").read_bytes() == left["manifest.json"]
    assert world.manifest()["labels_info"] == {"job_id": "the other run"}


def test_the_manifest_is_written_from_a_fresh_read_and_not_from_the_copy_taken_at_the_start(lg, fake, world, capsys, monkeypatch):
    """Something else (a note, a field another tool added) changes manifest.json while the labelling runs; the run's own fields are put on
    what is there when it finishes, so that nothing the start-of-run copy lacked is lost."""
    real = lg.label_rows

    def meanwhile(*args, **kwargs):
        result = real(*args, **kwargs)
        manifest = world.manifest()
        manifest["noted_meanwhile"] = {"by": "someone else"}
        (world.gallery / "manifest.json").write_text(json.dumps(manifest, indent=2))
        return result
    monkeypatch.setattr(lg, "label_rows", meanwhile)
    ran = run_main(lg, capsys, world.argv())
    assert ran.rc == 0, ran.out
    manifest = world.manifest()
    assert manifest["noted_meanwhile"] == {"by": "someone else"}, "the start-of-run copy was written back over it"
    assert manifest["labels_status"] == "done" and manifest["labels_info"]["mode"] == "single"


def test_a_manifest_that_disappears_during_the_run_is_a_refusal_at_the_end(lg, fake, world, capsys, monkeypatch):
    real = lg.label_rows

    def meanwhile(*args, **kwargs):
        result = real(*args, **kwargs)
        (world.gallery / "manifest.json").unlink()
        return result
    monkeypatch.setattr(lg, "label_rows", meanwhile)
    ran = run_main(lg, capsys, world.argv())
    assert ran.rc == 1 and ran.lines[-1] == "ERROR manifest_unreadable" and not (world.gallery / "labels.npy").exists(), ran.out


def test_rows_that_disagree_with_the_manifest_are_refused_by_count(lg, fake, world, capsys):
    texts = read_texts(world.gallery)
    (world.gallery / "report_texts.txt").write_text("\n".join(texts[:-1]) + "\n")
    ran = run_main(lg, capsys, world.argv())
    n = len(texts)
    assert ran.rc == 1 and ran.lines[-1] == "ERROR rows_disagree texts={} groups={} manifest={}".format(n - 1, n, n), ran.out
    assert fake.CONSTRUCTED == []


def test_a_group_count_that_disagrees_with_the_manifest_is_refused(lg, fake, world, capsys):
    manifest = world.manifest()
    manifest["counts"]["report_groups"] += 1
    (world.gallery / "manifest.json").write_text(json.dumps(manifest))
    ran = run_main(lg, capsys, world.argv())
    found = len(world.reps())
    assert ran.rc == 1 and ran.lines[-1] == "ERROR groups_disagree found={} manifest={}".format(found, found + 1), ran.out


def test_a_missing_input_file_is_a_refusal_not_a_traceback(lg, fake, world, capsys):
    (world.gallery / "txt_groups.npy").unlink()
    ran = run_main(lg, capsys, world.argv())
    assert ran.rc == 1 and ran.lines[-1] == "ERROR inputs_unreadable" and ran.err == "" and fake.CONSTRUCTED == []


@pytest.mark.parametrize("damage", ["empty", "cut_in_the_header", "cut_in_the_data", "garbage"])
@pytest.mark.parametrize("name", ["txt_groups.npy", "txt_split.npy", "txt_split_row.npy"])
def test_a_damaged_input_array_is_a_refusal_and_not_a_traceback(lg, fake, world, capsys, name, damage):
    """np.load answers a 0-byte file with EOFError, which is neither an OSError nor a ValueError: the one damage that used to be a crash."""
    path = world.gallery / name
    blob = path.read_bytes()
    path.write_bytes({"empty": b"", "cut_in_the_header": blob[:20], "cut_in_the_data": blob[:-30], "garbage": b"not an array"}[damage])
    before = snapshot(world.gallery)
    ran = run_main(lg, capsys, world.argv())
    assert ran.rc == 1 and ran.lines[-1] == "ERROR inputs_unreadable", ran.out + ran.err
    assert ran.err == "" and fake.CONSTRUCTED == [] and snapshot(world.gallery) == before


def test_an_empty_report_texts_file_is_refused_by_count(lg, fake, world, capsys):
    (world.gallery / "report_texts.txt").write_bytes(b"")
    ran = run_main(lg, capsys, world.argv())
    n = len(world.groups())
    assert ran.rc == 1 and ran.lines[-1] == "ERROR rows_disagree texts=0 groups={} manifest={}".format(n, n), ran.out
    assert ran.err == "" and fake.CONSTRUCTED == []


def test_test_rows_that_are_not_a_permutation_are_refused(lg, fake, world, capsys):
    split_row = load_array(world.gallery, "txt_split_row.npy")
    split_row[-1] = split_row[-2]                                      # two test studies with one row in test.parquet
    np.save(str(world.gallery / "txt_split_row.npy"), split_row)
    ran = run_main(lg, capsys, world.argv())
    assert ran.rc == 1 and ran.lines[-1] == "ERROR test_rows found=40 expected=40", ran.out


def test_an_unreadable_reference_is_refused_before_anything_is_labelled(lg, fake, world, capsys):
    (world.reference / "chexbert_labels.json").write_text("{not json")
    ran = run_main(lg, capsys, world.argv())
    assert ran.rc == 1 and ran.lines[-1] == "ERROR reference_unreadable" and fake.CONSTRUCTED == [] and ran.err == ""
    (world.reference / "chexbert_labels.json").write_bytes(b"")                  # an empty file is not a JSON document either
    ran = run_main(lg, capsys, world.argv())
    assert ran.rc == 1 and ran.lines[-1] == "ERROR reference_unreadable" and fake.CONSTRUCTED == [] and ran.err == ""
    shutil.rmtree(str(world.reference))
    assert run_main(lg, capsys, world.argv()).lines[-1] == "ERROR reference_unreadable"


@pytest.mark.parametrize("damage", ["no_y_true", "ragged", "not_numbers", "not_a_list"])
def test_a_reference_whose_labels_cannot_be_read_as_a_matrix_is_refused(lg, fake, world, capsys, damage):
    payload = read_reference(world.reference)
    if damage == "no_y_true":
        del payload["y_true"]
    elif damage == "ragged":
        payload["y_true"][3] = payload["y_true"][3][:5]
    elif damage == "not_a_list":
        payload["y_true"] = "text"
    else:
        payload["y_true"][3][0] = "x"
    write_reference_payload(world.reference, payload)
    ran = run_main(lg, capsys, world.argv())
    assert ran.rc == 1 and ran.lines[-1] == "ERROR reference_unreadable" and fake.CONSTRUCTED == []


@pytest.mark.parametrize("keep_rows, keep_columns, expected", [
    (39, 14, "ERROR reference_shape rows=39 expected=40 width=14"), (41, 14, "ERROR reference_shape rows=41 expected=40 width=14"),
    (40, 13, "ERROR reference_shape rows=40 expected=40 width=13"), (0, 14, "ERROR reference_shape rows=0 expected=40 width=0")])
def test_a_reference_with_another_number_of_studies_or_labels_is_refused(lg, fake, world, capsys, keep_rows, keep_columns, expected):
    payload = read_reference(world.reference)
    rows = (payload["y_true"] + payload["y_true"][:1])[:keep_rows]
    payload["y_true"] = [row[:keep_columns] for row in rows]
    write_reference_payload(world.reference, payload)
    ran = run_main(lg, capsys, world.argv())
    assert ran.rc == 1 and ran.lines[-1] == expected and fake.CONSTRUCTED == [], ran.out


# ── the single job ────────────────────────────────────────────────────────────

def test_a_single_job_labels_checks_and_finishes_the_gallery(lg, fake, world, capsys):
    before = snapshot(world.gallery)
    manifest_before = world.manifest()
    ran = run_main(lg, capsys, world.argv())
    assert ran.rc == 0, ran.out + ran.err
    texts, groups = read_texts(world.gallery), world.groups()
    labels = load_array(world.gallery, "labels.npy")
    assert labels.dtype == np.uint8 and labels.shape == (len(texts), 14)
    assert labels.tolist() == [FAKE.label_of(t) for t in texts], "a row has the labels its own text would have had: duplicates share them"
    assert json.loads((world.gallery / "label_names.json").read_text()) == NAMES
    check = json.loads((world.gallery / "labels_check.json").read_text())
    assert check["refs_mismatch"] == 0 and check["labels_mismatch"] == 0 and check["n_test"] == 40 and check["stage"] == "final"
    assert check["reference"] == REFERENCE_NAME and check["label_names_source"] == "dump"
    manifest = world.manifest()
    assert manifest["labels_status"] == "done"
    info = manifest["labels_info"]
    assert info["groups_labelled"] == len(np.unique(groups)) and info["rows"] == len(texts) and info["mode"] == "single"
    assert info["labeler"] == "f1chexbert" and info["check"] == {"n_test": 40, "refs_mismatch": 0, "labels_mismatch": 0}
    assert {k: v for k, v in manifest.items() if k not in ("labels_status", "labels_info")} == {
        k: v for k, v in manifest_before.items() if k != "labels_status"}, "nothing else in the manifest moved"
    after = snapshot(world.gallery)
    assert changed_paths(before, after) == {"labels.npy", "label_names.json", "labels_check.json", "manifest.json"}
    assert results(ran.lines)[-1] == {"refs_mismatch": 0, "labels_mismatch": 0}, "the cross-check is the last result it prints"
    assert ran.lines[-1] == "[labels] labels_status=done"


def test_the_labeller_is_asked_once_per_group_in_group_id_order_and_never_once_per_row(lg, fake, world, capsys):
    assert run_main(lg, capsys, world.argv()).rc == 0
    texts = read_texts(world.gallery)
    assert fake.CALLS == [texts[r] for r in world.reps()]
    assert len(fake.CALLS) == world.manifest()["counts"]["report_groups"] < len(texts)
    assert len(fake.CONSTRUCTED) == 1


def test_a_single_job_says_what_it_did_in_lines_of_known_shapes(lg, fake, world, capsys):
    ran = run_main(lg, capsys, world.argv("--canary", "10", "--progress", "30"))
    assert ran.rc == 0, ran.out
    groups = len(world.reps())
    lines = ran.lines
    assert lines[:4] == ["[labels] mode=single", "[labels] rows=240 groups={} test=40".format(groups),
                         "[labels] init_args=refs_filename,hyps_filename,device,kwargs", "[labels] device=cpu"]
    assert any(re.fullmatch(r"\[labels\] canary: [0-9]+\.[0-9]{3} s/report, projected [0-9]+ s for [0-9]+ rows", l) for l in lines)
    assert "[labels] progress: 30 of {} rows".format(groups) in lines
    summary = [r for r in results(lines) if "groups" in r]
    assert len(summary) == 1 and summary[0]["groups"] == groups and summary[0]["rows"] == 240 and isinstance(summary[0]["labelling_s"], int)
    sharing = count_test_reports_sharing_a_text(world.gallery)
    assert lines[-6:] == ["[labels] wrote labels_check.json", "[labels] test_rows_sharing_a_text={}".format(sharing),
                          'RESULT {"refs_mismatch":0,"labels_mismatch":0}', "[labels] wrote labels.npy",
                          "[labels] wrote label_names.json", "[labels] labels_status=done"]
    assert ran.err == ""


def count_test_reports_sharing_a_text(gallery: Path) -> int:
    """How many of the test reports are not the first row of their duplicate group, so that their labels are another row's: computed from
    the groups alone, apart from the script."""
    groups = load_array(gallery, "txt_groups.npy").tolist()
    first = {}
    for row, group in enumerate(groups):
        first.setdefault(group, row)
    return sum(first[groups[r]] != r for r in gallery_rows_of_test_reports(gallery).tolist())


@pytest.mark.parametrize("mode", ["single", "merge"])
def test_the_log_says_how_many_test_reports_share_a_text_so_that_the_zero_has_a_weight(lg, fake, world, capsys, mode):
    """The cross-check's zero is informative only for the test reports that take their labels from another row: the line says how many,
    in numbers only, just before the verdict, and labels_check.json holds the same number."""
    argv = world.argv()
    if mode == "merge":
        for i in range(3):
            assert run_main(lg, capsys, world.argv("--shard", str(i), "--of", "3")).rc == 0
        argv = world.argv("--merge", "--of", "3")
    ran = run_main(lg, capsys, argv)
    assert ran.rc == 0, ran.out
    sharing = count_test_reports_sharing_a_text(world.gallery)
    assert 0 < sharing < 40, "the tiny gallery has test reports that share a text and test reports that do not"
    line = "[labels] test_rows_sharing_a_text={}".format(sharing)
    assert ran.lines.count(line) == 1
    assert ran.lines.index(line) == ran.lines.index('RESULT {"refs_mismatch":0,"labels_mismatch":0}') - 1, "right before the verdict"
    assert json.loads((world.gallery / "labels_check.json").read_text())["test_rows_sharing_a_text"] == sharing


def test_a_failed_cross_check_also_says_how_many_test_reports_share_a_text(lg, fake, world, capsys):
    payload = read_reference(world.reference)
    payload["y_true"][5][3] ^= 1
    write_reference_payload(world.reference, payload)
    ran = run_main(lg, capsys, world.argv())
    assert ran.rc == 1 and "[labels] test_rows_sharing_a_text={}".format(count_test_reports_sharing_a_text(world.gallery)) in ran.lines


def test_a_shard_and_a_refusal_print_no_such_line(lg, fake, world, capsys):
    """It belongs to the cross-check, which a shard does not run and a refusal never reaches."""
    shard = run_main(lg, capsys, world.argv("--shard", "0", "--of", "2"))
    assert shard.rc == 0 and not [l for l in shard.lines if "test_rows_sharing_a_text" in l]
    break_gate(world.gallery, "false")
    refused = run_main(lg, capsys, world.argv())
    assert refused.rc == 1 and not [l for l in refused.lines if "test_rows_sharing_a_text" in l]


@pytest.mark.parametrize("device, passed", [("auto", None), ("cpu", "cpu"), ("cuda", "cuda")])
def test_the_device_argument_is_passed_only_when_asked_for(lg, install_fake, world, capsys, device, passed):
    fake = install_fake()
    ran = run_main(lg, capsys, world.argv(device=device))
    assert ran.rc == 0, ran.out
    assert fake.CONSTRUCTED == [{"device": passed}]
    assert "[labels] device={}".format(passed or "cpu") in ran.lines


def test_the_device_line_is_the_first_parameters_device(lg, install_fake, world, capsys):
    install_fake(FAKE_F1_DEFAULT_DEVICE="cuda:0")
    ran = run_main(lg, capsys, world.argv())
    assert ran.rc == 0 and "[labels] device=cuda:0" in ran.lines


def test_a_labeller_whose_device_cannot_be_read_is_reported_as_unknown(lg, install_fake, world, capsys):
    fake = install_fake()
    original = fake.F1CheXbert.__init__

    def without_model(self, *args, **kwargs):
        original(self, *args, **kwargs)
        del self.model
    fake.F1CheXbert.__init__ = without_model
    ran = run_main(lg, capsys, world.argv())
    assert ran.rc == 0 and "[labels] device=unknown" in ran.lines


def test_asking_for_a_device_the_labeller_cannot_take_is_refused_before_it_is_built(lg, install_fake, tmp_path, template, capsys):
    fake = install_fake(FAKE_F1_NO_DEVICE_ARG="1")
    first, second = World(tmp_path / "first", template), World(tmp_path / "second", template)
    ran = run_main(lg, capsys, first.argv())
    assert ran.rc == 0 and "[labels] init_args=refs_filename,hyps_filename" in ran.lines, "auto passes nothing: no argument is fine"
    fake.CONSTRUCTED.clear()
    ran = run_main(lg, capsys, second.argv(device="cpu"))
    assert ran.rc == 1 and ran.lines[-1] == "ERROR no_device_argument" and fake.CONSTRUCTED == []


def test_a_labeller_with_another_label_order_is_refused_before_it_labels_anything(lg, install_fake, world, capsys):
    fake = install_fake(FAKE_F1_NAMES="swapped")
    ran = run_main(lg, capsys, world.argv())
    assert ran.rc == 1 and ran.lines[-1] == "ERROR label_order", ran.out
    assert fake.CALLS == [] and not (world.gallery / "labels.npy").exists() and world.manifest()["labels_status"] == "pending"


def test_a_canary_over_budget_exits_2_and_leaves_the_gallery_as_it_was(lg, install_fake, world, capsys):
    fake = install_fake(FAKE_F1_SLEEP="0.01")
    before = snapshot(world.gallery)
    ran = run_main(lg, capsys, world.argv("--canary", "5", "--budget-s", "0"))
    assert ran.rc == 2, ran.out
    assert re.fullmatch(r"\[labels\] canary: [0-9]+\.[0-9]{3} s/report, projected [0-9]+ s for [0-9]+ rows", ran.lines[-1]), ran.lines[-1]
    assert len(fake.CALLS) == 5
    assert snapshot(world.gallery) == before, "exit 2 writes nothing: the wrapper's next step is the sharded CPU path"


def test_a_canary_inside_the_budget_lets_the_job_finish(lg, install_fake, world, capsys):
    install_fake(FAKE_F1_SLEEP="0.001")
    ran = run_main(lg, capsys, world.argv("--canary", "5", "--budget-s", "100000"))
    assert ran.rc == 0 and world.manifest()["labels_status"] == "done"


def test_a_labeller_that_dies_leaves_nothing_and_the_job_log_the_class_name_only(lg, install_fake, world, capsys):
    install_fake(FAKE_F1_DIE_AFTER="7")
    before = snapshot(world.gallery)
    ran = run_main(lg, capsys, world.argv())
    assert ran.rc == 1 and ran.lines[-1] == "ERROR failed RuntimeError", ran.out
    assert "fake labeller died on" in ran.err, "the traceback is on stderr, which the wrapper keeps in a file"
    assert "Findings" not in ran.out
    assert snapshot(world.gallery) == before


def test_a_bad_row_from_the_labeller_is_refused_by_its_place_in_the_job(lg, install_fake, world, capsys):
    install_fake(FAKE_F1_BAD_ROW="4")
    ran = run_main(lg, capsys, world.argv())
    assert ran.rc == 1 and ran.lines[-1] == "ERROR bad_label_row row=3" and not (world.gallery / "labels.npy").exists()


@pytest.mark.parametrize("case_sensitive", [False, True])
def test_a_representative_in_capitals_is_the_same_report_only_to_a_labeller_that_does_not_see_case(
        lg, install_fake, tmp_path, template, capsys, monkeypatch, case_sensitive):
    """The premise the whole job rests on, tried against a labeller that has it and one that does not (FAKE_F1_CASE_SENSITIVE). The text
    of a train report that a test report shares its group with is put in capitals: the group is still the group (its key is lower-cased).
    A labeller that does not see case gives the broadcast what labelling row by row would give; one that does gives the group's test
    reports labels unlike their own, and the cross-check says so, as text shared with another row."""
    if case_sensitive:
        monkeypatch.setenv("FAKE_F1_CASE_SENSITIVE", "1")
    world = World(tmp_path / "w", template)                       # its reference is labelled by the labeller as it is set now
    victim, differing = change_a_shared_representative(world, str.upper)
    fake = install_fake()
    ran = run_main(lg, capsys, world.argv())
    check = json.loads((world.gallery / "labels_check.json").read_text())
    assert check["test_rows_sharing_a_text"] > 0, "what follows is about test reports that really share a text with another row"
    per_row = np.array([[int(v) for v in fake.F1CheXbert().get_label(t)] for t in read_texts(world.gallery)], dtype=np.uint8)
    if not case_sensitive:
        assert differing == 0 and ran.rc == 0 and check["labels_mismatch"] == 0, ran.out
        assert np.array_equal(load_array(world.gallery, "labels.npy"), per_row), "the broadcast is labelling row by row, capitals and all"
    else:
        assert differing >= 1 and ran.rc == 1, ran.out
        assert check["labels_mismatch"] == check["labels_mismatch_shared_text"] == differing and check["labels_mismatch_own_text"] == 0
        groups, reps = world.groups(), world.reps()
        unverified = load_array(world.gallery, "labels_unverified.npy")
        assert np.array_equal(unverified, per_row[reps[np.searchsorted(groups[reps], groups)]]), "every row got its representative's labels"
        wrong = set(np.flatnonzero((unverified != per_row).any(axis=1)).tolist())
        assert victim not in wrong, "the representative itself was labelled from its own text"
        assert wrong & set(np.flatnonzero(groups == groups[victim]).tolist()), "the members of its group are off"
        # (other rows are off too: the tiny gallery has a train group spelt two ways. The cross-check only sees the test reports.)


# ── shards and merge ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("of", [1, 3, 8, 1000])
def test_shards_then_merge_equal_the_single_job_byte_for_byte(lg, fake, tmp_path, template, capsys, of):
    single, sharded = World(tmp_path / "single", template), World(tmp_path / "sharded", template)
    assert run_main(lg, capsys, single.argv()).rc == 0
    for i in range(of):
        assert run_main(lg, capsys, sharded.argv("--shard", str(i), "--of", str(of))).rc == 0
        assert not (sharded.gallery / "labels.npy").exists(), "a shard finishes nothing"
    merged = run_main(lg, capsys, sharded.argv("--merge", "--of", str(of)))
    assert merged.rc == 0, merged.out
    for name in ("labels.npy", "label_names.json"):
        assert (sharded.gallery / name).read_bytes() == (single.gallery / name).read_bytes(), name
    assert sharded.manifest()["labels_status"] == "done" and sharded.manifest()["labels_info"]["mode"] == "sharded"
    assert json.loads((sharded.gallery / "labels_check.json").read_text())["labels_mismatch"] == 0


def test_a_shard_labels_only_its_rows_and_writes_only_its_two_files(lg, fake, world, capsys):
    before, manifest_before = snapshot(world.gallery), (world.gallery / "manifest.json").read_bytes()
    ran = run_main(lg, capsys, world.argv("--shard", "1", "--of", "4"))
    assert ran.rc == 0, ran.out
    texts, mine = read_texts(world.gallery), world.reps()[1::4]
    rows = load_array(world.gallery, "labels_shard_1_rows.npy")
    labels = load_array(world.gallery, "labels_shard_1.npy")
    assert rows.dtype == np.int64 and rows.tolist() == mine.tolist()
    assert labels.dtype == np.uint8 and labels.tolist() == [FAKE.label_of(texts[r]) for r in mine]
    assert fake.CALLS == [texts[r] for r in mine]
    assert changed_paths(before, snapshot(world.gallery)) == {"labels_shard_1.npy", "labels_shard_1_rows.npy"}
    assert (world.gallery / "manifest.json").read_bytes() == manifest_before, "a shard leaves the manifest alone"
    assert "[labels] mode=shard index=1 of=4" in ran.lines
    (summary,) = results(ran.lines)
    assert summary["shard"] == 1 and summary["of"] == 4 and summary["groups"] == len(mine)
    assert not [r for r in results(ran.lines) if "refs_mismatch" in r], "the cross-check is for the merge"


def test_a_finished_shard_is_kept_and_not_labelled_again(lg, install_fake, world, capsys):
    fake = install_fake()
    assert run_main(lg, capsys, world.argv("--shard", "2", "--of", "4")).rc == 0
    before = snapshot(world.gallery)
    fake.CONSTRUCTED.clear()
    fake.CALLS.clear()
    ran = run_main(lg, capsys, world.argv("--shard", "2", "--of", "4"))
    assert ran.rc == 0, ran.out
    assert "[labels] shard 2 of 4 kept: already labelled" in ran.lines
    assert fake.CONSTRUCTED == [] and fake.CALLS == [] and snapshot(world.gallery) == before


@pytest.mark.parametrize("damage", ["short_labels", "other_rows", "garbage", "values", "dtype", "no_rows_file", "empty_labels", "empty_rows"])
def test_a_shard_that_is_not_the_shard_asked_for_is_labelled_again(lg, fake, world, capsys, damage):
    """An empty file is the case np.load answers with EOFError, not ValueError: a shard cut to 0 bytes (a full disk, a killed copy) must be
    labelled again like any other damage, since nothing here may delete it (R8)."""
    assert run_main(lg, capsys, world.argv("--shard", "2", "--of", "4")).rc == 0
    good = load_array(world.gallery, "labels_shard_2.npy").copy()
    labels_path, rows_path = world.gallery / "labels_shard_2.npy", world.gallery / "labels_shard_2_rows.npy"
    if damage == "short_labels":
        np.save(str(labels_path), good[:-1])
    elif damage == "other_rows":
        np.save(str(rows_path), load_array(world.gallery, "labels_shard_2_rows.npy") + 1)
    elif damage == "garbage":
        labels_path.write_bytes(b"not an array")
    elif damage == "empty_labels":
        labels_path.write_bytes(b"")
    elif damage == "empty_rows":
        rows_path.write_bytes(b"")
    elif damage == "values":
        bad = good.copy()
        bad[0, 0] = 2
        np.save(str(labels_path), bad)
    elif damage == "dtype":
        np.save(str(labels_path), good.astype(np.int64))
    else:
        rows_path.unlink()
    fake.CALLS.clear()
    ran = run_main(lg, capsys, world.argv("--shard", "2", "--of", "4"))
    assert ran.rc == 0 and fake.CALLS, ran.out
    assert load_array(world.gallery, "labels_shard_2.npy").tolist() == good.tolist()


def test_a_finished_shard_of_another_job_size_is_not_this_shard(lg, fake, world, capsys):
    assert run_main(lg, capsys, world.argv("--shard", "1", "--of", "2")).rc == 0
    ran = run_main(lg, capsys, world.argv("--shard", "1", "--of", "4"))
    assert ran.rc == 0 and "[labels] shard 1 of 4 kept: already labelled" not in ran.lines
    assert load_array(world.gallery, "labels_shard_1_rows.npy").tolist() == world.reps()[1::4].tolist()


def test_the_merge_names_the_shards_that_are_missing_and_writes_nothing(lg, install_fake, world, capsys):
    fake = install_fake()
    for i in (0, 1, 3):
        assert run_main(lg, capsys, world.argv("--shard", str(i), "--of", "4")).rc == 0
    fake.CONSTRUCTED.clear()
    before = snapshot(world.gallery)
    ran = run_main(lg, capsys, world.argv("--merge", "--of", "4"))
    assert ran.rc == 1 and ran.lines[-1] == "ERROR shards_missing count=1 first=2", ran.out
    assert snapshot(world.gallery) == before and fake.CONSTRUCTED == []


@pytest.mark.parametrize("damage", ["other_rows", "short_labels", "values", "garbage", "empty_labels", "empty_rows"])
def test_the_merge_refuses_a_shard_that_is_not_the_one_it_should_be(lg, fake, world, capsys, damage):
    for i in range(4):
        assert run_main(lg, capsys, world.argv("--shard", str(i), "--of", "4")).rc == 0
    labels_path, rows_path = world.gallery / "labels_shard_1.npy", world.gallery / "labels_shard_1_rows.npy"
    good = load_array(world.gallery, "labels_shard_1.npy").copy()
    if damage == "other_rows":
        rows = load_array(world.gallery, "labels_shard_1_rows.npy").copy()
        rows[[0, 1]] = rows[[1, 0]]
        np.save(str(rows_path), rows)
    elif damage == "short_labels":
        np.save(str(labels_path), good[:-1])
    elif damage == "values":
        good[0, 0] = 7
        np.save(str(labels_path), good)
    elif damage == "empty_labels":
        labels_path.write_bytes(b"")
    elif damage == "empty_rows":
        rows_path.write_bytes(b"")
    else:
        labels_path.write_bytes(b"not an array")
    before = snapshot(world.gallery)
    ran = run_main(lg, capsys, world.argv("--merge", "--of", "4"))
    assert ran.rc == 1 and ran.lines[-1] == "ERROR shard_invalid shard=1", ran.out
    assert ran.err == "" and snapshot(world.gallery) == before


def test_the_merge_does_not_build_the_labeller(lg, install_fake, world, capsys):
    fake = install_fake()
    for i in range(2):
        assert run_main(lg, capsys, world.argv("--shard", str(i), "--of", "2")).rc == 0
    fake.CONSTRUCTED.clear()
    asked_before = len(fake.CALLS)
    ran = run_main(lg, capsys, world.argv("--merge", "--of", "2"))
    assert ran.rc == 0 and "[labels] mode=merge of=2" in ran.lines
    assert fake.CONSTRUCTED == [] and len(fake.CALLS) == asked_before, "the merge builds no labeller and labels nothing"
    (summary, final) = results(ran.lines)
    assert summary == {"merged": 2, "groups": len(world.reps()), "rows": 240} and final == {"refs_mismatch": 0, "labels_mismatch": 0}


@pytest.mark.parametrize("argv", [
    [], ["--gallery", "g", "--shard", "1"], ["--gallery", "g", "--of", "4"], ["--gallery", "g", "--merge"],
    ["--gallery", "g", "--shard", "4", "--of", "4"], ["--gallery", "g", "--shard", "-1", "--of", "4"], ["--gallery", "g", "--shard", "0", "--of", "0"],
    ["--gallery", "g", "--shard", "0", "--of", "2", "--merge"], ["--gallery", "g", "--canary", "-1"], ["--gallery", "g", "--device", "tpu"],
    ["--gallery", "g", "--budget-s", "x"], ["--gallery", "g", "--budget-s", "-1"], ["--gallery", "g", "--progress", "-5"], ["--gallery", "g", "--bogus"]])
def test_a_bad_command_line_is_a_usage_error_and_not_the_canarys_exit_2(lg, argv, capsys):
    """argparse exits 2 by itself, which the wrapper would read as 'the canary projects too long': a usage error is 64."""
    with pytest.raises(SystemExit) as stop:
        lg.main(argv)
    assert stop.value.code == 64
    assert "usage" in capsys.readouterr().err


def test_help_exits_0(lg, capsys):
    with pytest.raises(SystemExit) as stop:
        lg.main(["--help"])
    assert stop.value.code == 0 and "--shard" in capsys.readouterr().out


# ── the cross-check ───────────────────────────────────────────────────────────

def test_a_doctored_refs_file_stops_the_job_before_the_labeller_exists_and_leaves_the_numbers(lg, fake, world, capsys):
    refs = world.reference / "refs.txt"
    lines = refs.read_text().splitlines()
    lines[3] += " and one more word"
    lines[9] = lines[9].upper()
    refs.write_text("\n".join(lines) + "\n")
    ran = run_main(lg, capsys, world.argv())
    assert ran.rc == 1 and ran.lines[-1] == "ERROR refs_mismatch n=2", ran.out
    assert fake.CONSTRUCTED == [] and not (world.gallery / "labels.npy").exists()
    check = json.loads((world.gallery / "labels_check.json").read_text())
    assert check["refs_mismatch"] == 2 and check["labels_mismatch"] is None and check["stage"] == "preflight"
    assert world.manifest()["labels_status"] == "pending"


def test_refs_with_a_line_missing_or_added_count_as_mismatches(lg, fake, world, capsys):
    refs = world.reference / "refs.txt"
    lines = refs.read_text().splitlines()
    refs.write_text("\n".join(lines[:-1]) + "\n")
    assert run_main(lg, capsys, world.argv()).lines[-1] == "ERROR refs_mismatch n=1"
    refs.write_text("\n".join(lines + ["one more"]) + "\n")
    assert run_main(lg, capsys, world.argv()).lines[-1] == "ERROR refs_mismatch n=1"


def test_a_doctored_y_true_fails_the_cross_check_and_leaves_the_gallery_unlabelled(lg, fake, world, capsys):
    payload = read_reference(world.reference)
    payload["y_true"][5][3] ^= 1
    write_reference_payload(world.reference, payload)
    ran = run_main(lg, capsys, world.argv())
    assert ran.rc == 1, ran.out
    assert results(ran.lines)[-1] == {"refs_mismatch": 0, "labels_mismatch": 1}
    assert ran.lines[-1] == "[labels] labels_status=pending"
    assert fake.CALLS, "the labelling did happen"
    assert not (world.gallery / "labels.npy").exists() and not (world.gallery / "label_names.json").exists()
    unverified = load_array(world.gallery, "labels_unverified.npy")
    assert unverified.tolist() == [FAKE.label_of(t) for t in read_texts(world.gallery)], "what was labelled is kept for the diagnosis"
    check = json.loads((world.gallery / "labels_check.json").read_text())
    assert check["refs_mismatch"] == 0 and check["labels_mismatch"] == 1 and check["stage"] == "final"
    manifest = world.manifest()
    assert manifest["labels_status"] == "pending" and "labels_info" not in manifest


@pytest.mark.parametrize("column", range(14))
def test_every_one_of_the_14_labels_is_compared_with_the_published_one(lg, fake, world, capsys, column):
    """One label of one test report differs from the published y_true, whichever of the 14 it is: exactly one report differs, and the job fails."""
    payload = read_reference(world.reference)
    payload["y_true"][17][column] ^= 1
    write_reference_payload(world.reference, payload)
    ran = run_main(lg, capsys, world.argv())
    assert ran.rc == 1 and results(ran.lines)[-1] == {"refs_mismatch": 0, "labels_mismatch": 1}, ran.out
    assert json.loads((world.gallery / "labels_check.json").read_text())["labels_mismatch"] == 1


def test_a_failed_cross_check_after_shards_fails_the_merge_the_same_way(lg, fake, world, capsys):
    payload = read_reference(world.reference)
    payload["y_true"][0][0] ^= 1
    write_reference_payload(world.reference, payload)
    for i in range(3):
        assert run_main(lg, capsys, world.argv("--shard", str(i), "--of", "3")).rc == 0
    ran = run_main(lg, capsys, world.argv("--merge", "--of", "3"))
    assert ran.rc == 1 and results(ran.lines)[-1] == {"refs_mismatch": 0, "labels_mismatch": 1}
    assert not (world.gallery / "labels.npy").exists() and world.manifest()["labels_status"] == "pending"


def test_a_rerun_after_a_failed_cross_check_may_try_again(lg, fake, world, capsys):
    good = read_reference(world.reference)
    broken = json.loads(json.dumps(good))
    broken["y_true"][5][3] ^= 1
    write_reference_payload(world.reference, broken)
    assert run_main(lg, capsys, world.argv()).rc == 1
    write_reference_payload(world.reference, good)
    ran = run_main(lg, capsys, world.argv())
    assert ran.rc == 0 and (world.gallery / "labels.npy").is_file() and world.manifest()["labels_status"] == "done"


OTHER_REPORT = "Findings: a different report altogether. Impression: it is not the same."


def change_a_shared_representative(world: World, new_text: Callable[[str], str]) -> Tuple[int, int]:
    """Change the text of a TRAIN row that is the representative of a group a test report belongs to (so that report is labelled from that
    row's text, not its own): new_text(old) is the new one. Returns (that row, how many of the group's test reports then have labels that
    differ from the labels of their own text, under the labeller as FAKE_F1_CASE_SENSITIVE sets it now). The tiny gallery has such a group
    on purpose."""
    reps, groups, texts = world.reps(), world.groups(), read_texts(world.gallery)
    rows = gallery_rows_of_test_reports(world.gallery)
    n_train = world.manifest()["counts"]["images"]
    rep_row_of = reps[np.searchsorted(groups[reps], groups)]
    shared = [j for j, r in enumerate(rows) if rep_row_of[r] != r and rep_row_of[r] < n_train]
    assert shared, "the tiny gallery has a test report that a train report repeats"
    victim = int(rep_row_of[rows[shared[0]]])
    texts[victim] = new_text(texts[victim])
    (world.gallery / "report_texts.txt").write_text("\n".join(texts) + "\n")
    own = (world.reference / "refs.txt").read_text().splitlines()
    differing = sum(FAKE.label_of(texts[victim]) != FAKE.label_of(own[j]) for j, r in enumerate(rows) if groups[r] == groups[victim])
    return victim, differing


def doctor_a_shared_representative(world: World) -> int:
    """Put another report altogether in place of that representative's: how many test reports' labels that changes."""
    return change_a_shared_representative(world, lambda old: OTHER_REPORT)[1]


def test_the_cross_check_catches_a_group_whose_representative_is_not_its_members_text(lg, fake, world, capsys):
    """Why the check exists: a test report that shares its group with an earlier row is labelled from THAT row's text. Doctor the text
    of such a representative and the test report's labels no longer equal the published ones, and the job says the text was shared."""
    expected = doctor_a_shared_representative(world)
    assert expected >= 1
    ran = run_main(lg, capsys, world.argv())
    assert ran.rc == 1 and results(ran.lines)[-1] == {"refs_mismatch": 0, "labels_mismatch": expected}, ran.out
    assert "[labels] mismatch split: own_text=0 shared_text={}".format(expected) in ran.lines
    check = json.loads((world.gallery / "labels_check.json").read_text())
    assert check["labels_mismatch_own_text"] == 0 and check["labels_mismatch_shared_text"] == expected
    assert 0 < check["test_rows_sharing_a_text"] <= 40


def read_texts_of_reference(world: World) -> List[str]:
    return (world.reference / "refs.txt").read_text().splitlines()


def test_the_cross_check_takes_the_test_rows_in_txt_split_row_order(lg, fake, world, capsys):
    """Reverse the test rows' place in test.parquet: the reference (written in that order) can only match if the job reorders."""
    split_row = load_array(world.gallery, "txt_split_row.npy")
    n_train = int((load_array(world.gallery, "txt_split.npy") == 0).sum())
    split_row[n_train:] = split_row[n_train:][::-1]
    np.save(str(world.gallery / "txt_split_row.npy"), split_row)
    write_reference(world.reference, world.gallery)
    reference_lines = read_texts_of_reference(world)
    assert reference_lines == read_texts(world.gallery)[n_train:][::-1]
    ran = run_main(lg, capsys, world.argv())
    assert ran.rc == 0, ran.out
    assert results(ran.lines)[-1] == {"refs_mismatch": 0, "labels_mismatch": 0}


def test_chexbert_labels_json_with_label_names_in_the_labellers_order_needs_no_note(lg, fake, world, capsys):
    ran = run_main(lg, capsys, world.argv())
    assert ran.rc == 0 and not [l for l in ran.lines if l.startswith("=== note")]


def test_chexbert_labels_json_without_label_names_is_read_in_chexbert_14_order_and_says_so(lg, fake, world, capsys):
    payload = read_reference(world.reference)
    del payload["label_names"]
    write_reference_payload(world.reference, payload)
    ran = run_main(lg, capsys, world.argv())
    assert ran.rc == 0, ran.out
    assert "=== note: chexbert_labels.json has no label_names key: CHEXBERT_14 order assumed ===" in ran.lines
    assert json.loads((world.gallery / "labels_check.json").read_text())["label_names_source"] == "assumed"


def test_chexbert_labels_json_with_the_names_in_another_order_is_reordered_by_name(lg, fake, world, capsys):
    payload = read_reference(world.reference)
    perm = [3, 1, 13, 0, 2, 4, 5, 6, 7, 8, 9, 10, 12, 11]
    payload["label_names"] = [NAMES[i] for i in perm]
    payload["y_true"] = [[row[i] for i in perm] for row in payload["y_true"]]
    write_reference_payload(world.reference, payload)
    ran = run_main(lg, capsys, world.argv())
    assert ran.rc == 0, ran.out
    assert "=== note: chexbert_labels.json label_names reordered to the CHEXBERT_14 order ===" in ran.lines
    assert results(ran.lines)[-1] == {"refs_mismatch": 0, "labels_mismatch": 0}
    assert json.loads((world.gallery / "labels_check.json").read_text())["label_names_source"] == "reordered"


@pytest.mark.parametrize("names", [["a", "b"], NAMES[:-1], NAMES[:-1] + ["Something Else"], NAMES + ["Extra"], "text", 5])
def test_chexbert_labels_json_naming_other_labels_is_refused(lg, fake, world, capsys, names):
    payload = read_reference(world.reference)
    payload["label_names"] = names
    write_reference_payload(world.reference, payload)
    ran = run_main(lg, capsys, world.argv())
    assert ran.rc == 1 and ran.lines[-1] == "ERROR reference_label_names" and fake.CONSTRUCTED == []


def test_labels_check_json_holds_numbers_and_basenames_only(lg, fake, world, capsys):
    assert run_main(lg, capsys, world.argv()).rc == 0
    check = json.loads((world.gallery / "labels_check.json").read_text())
    assert set(check) == {"stage", "reference", "n_test", "refs_mismatch", "labels_mismatch", "labels_mismatch_own_text",
                          "labels_mismatch_shared_text", "test_rows_sharing_a_text", "label_names_source"}
    assert check["reference"] == REFERENCE_NAME, "a basename, not the path"
    assert 0 < check["test_rows_sharing_a_text"] <= 40


# ── R7: what the script prints ────────────────────────────────────────────────

ERROR_CODES = [
    "no_manifest", "manifest_unreadable", "gate_not_equal", "already_labelled", "inputs_unreadable", "rows_disagree", "groups_disagree",
    "test_rows", "reference_unreadable", "reference_shape", "reference_label_names", "refs_mismatch", "no_device_argument", "label_order",
    "bad_label_row", "shards_missing", "shard_invalid"]

# The one place the shapes of a line the script prints are written down in the tests. The wrappers' allowlists (LABEL_SHAPES in both
# wrappers, one anchored pattern per line) have to accept exactly these, and everything the script prints has to fit them: digits are
# bounded (a count has at most 7), and nothing in a shape is free text, so an id or a piece of a report cannot ride on a line of the log.
LABEL_SHAPE_LIST = [
    r"\[labels\] mode=single",
    r"\[labels\] mode=shard index=[0-9]{1,3} of=[0-9]{1,3}",
    r"\[labels\] mode=merge of=[0-9]{1,3}",
    r"\[labels\] rows=[0-9]{1,7} groups=[0-9]{1,7} test=[0-9]{1,7}",
    r"\[labels\] init_args=[A-Za-z_][A-Za-z0-9_]{0,31}(,[A-Za-z_][A-Za-z0-9_]{0,31}){0,7}",
    r"\[labels\] device=(cpu|cuda|cuda:[0-9]{1,2}|unknown)",
    r"\[labels\] canary: [0-9]{1,4}\.[0-9]{3} s/report, projected [0-9]{1,9} s for [0-9]{1,7} rows",
    r"\[labels\] progress: [0-9]{1,7} of [0-9]{1,7} rows",
    r"\[labels\] shard [0-9]{1,3} of [0-9]{1,3} kept: already labelled",
    r"\[labels\] wrote (labels\.npy|labels_unverified\.npy|label_names\.json|labels_check\.json|labels_shard_[0-9]{1,3}\.npy|labels_shard_[0-9]{1,3}_rows\.npy)",
    r"\[labels\] mismatch split: own_text=[0-9]{1,7} shared_text=[0-9]{1,7}",
    r"\[labels\] test_rows_sharing_a_text=[0-9]{1,7}",
    r"\[labels\] labels_status=(done|pending)",
    r'RESULT \{"refs_mismatch":[0-9]{1,7},"labels_mismatch":[0-9]{1,7}\}',
    r'RESULT \{"groups":[0-9]{1,7},"rows":[0-9]{1,7},"labelling_s":[0-9]{1,7}\}',
    r'RESULT \{"shard":[0-9]{1,3},"of":[0-9]{1,3},"groups":[0-9]{1,7},"labelling_s":[0-9]{1,7}\}',
    r'RESULT \{"merged":[0-9]{1,3},"groups":[0-9]{1,7},"rows":[0-9]{1,7}\}',
    r"ERROR (" + "|".join(ERROR_CODES) + r")( [a-z_]{1,20}=[0-9]{1,7}){0,3}",
    r"ERROR failed [A-Za-z_][A-Za-z0-9_]{0,59}",
    r"=== note: chexbert_labels\.json has no label_names key: CHEXBERT_14 order assumed ===",
    r"=== note: chexbert_labels\.json label_names reordered to the CHEXBERT_14 order ===",
]
LABEL_SHAPES = re.compile(r"^(" + "|".join(LABEL_SHAPE_LIST) + r")$")


def wrapper_patterns(name: str) -> List[str]:
    """LABEL_SHAPES as a wrapper holds it: one anchored pattern per line of one single-quoted assignment (grep -E reads a newline as 'or')."""
    found = re.search(r"^LABEL_SHAPES='(.*?)'$", (REPO_ROOT / "scripts" / name).read_text(), re.M | re.S)
    assert found, name + " has no LABEL_SHAPES assignment"
    return found.group(1).split("\n")


def grep_passes(lines: List[str], name: str = GPU_SH) -> List[str]:
    """The lines a wrapper's filter lets through: those very patterns, through the very grep, as the job runs them."""
    done = subprocess.run(["grep", "-aE", "\n".join(wrapper_patterns(name))], input="\n".join(lines) + "\n", capture_output=True, text=True)
    assert done.returncode in (0, 1), done.stderr
    return done.stdout.splitlines()


LABEL_PASS = [
    "[labels] mode=single", "[labels] mode=shard index=3 of=8", "[labels] mode=merge of=8", "[labels] rows=194125 groups=163021 test=2663",
    "[labels] init_args=refs_filename,hyps_filename,device,kwargs", "[labels] init_args=device",
    "[labels] device=cpu", "[labels] device=cuda", "[labels] device=cuda:0", "[labels] device=unknown",
    "[labels] canary: 0.012 s/report, projected 2345 s for 163021 rows", "[labels] canary: 0.163 s/report, projected 26573 s for 163021 rows",
    "[labels] progress: 20000 of 163021 rows", "[labels] shard 3 of 8 kept: already labelled",
    "[labels] wrote labels.npy", "[labels] wrote labels_unverified.npy", "[labels] wrote label_names.json", "[labels] wrote labels_check.json",
    "[labels] wrote labels_shard_3.npy", "[labels] wrote labels_shard_3_rows.npy", "[labels] wrote labels_shard_12.npy",
    "[labels] mismatch split: own_text=0 shared_text=2", "[labels] labels_status=done", "[labels] labels_status=pending",
    "[labels] test_rows_sharing_a_text=312", "[labels] test_rows_sharing_a_text=0", "[labels] test_rows_sharing_a_text=2663",
    'RESULT {"refs_mismatch":0,"labels_mismatch":0}', 'RESULT {"refs_mismatch":3,"labels_mismatch":12}',
    'RESULT {"groups":163021,"rows":194125,"labelling_s":2345}', 'RESULT {"shard":3,"of":8,"groups":20378,"labelling_s":3401}',
    'RESULT {"merged":8,"groups":163021,"rows":194125}',
    "ERROR no_manifest", "ERROR gate_not_equal", "ERROR already_labelled", "ERROR label_order", "ERROR reference_unreadable",
    "ERROR rows_disagree texts=194124 groups=194125 manifest=194125", "ERROR shards_missing count=3 first=5", "ERROR refs_mismatch n=7",
    "ERROR shard_invalid shard=1", "ERROR bad_label_row row=2", "ERROR failed RuntimeError", "ERROR failed OutOfMemoryError",
    "=== note: chexbert_labels.json has no label_names key: CHEXBERT_14 order assumed ===",
    "=== note: chexbert_labels.json label_names reordered to the CHEXBERT_14 order ===",
]
LABEL_WITHHELD = [
    "[labels] study_id=12345678", "[labels] The heart is mildly enlarged and there is a small pleural effusion.",
    "[labels] rows=194125 groups=163021 test=2663 study_id=12345678", "[labels] /sc/home/someone/images/p10/leak.jpg",
    "[labels] rows=12345678 groups=1 test=1", "[labels] progress: 12345678 of 1 rows",
    "[labels]  mode=single", " [labels] mode=single", "[labels] mode=single ", "[labels] mode=single\t", "[labels] mode=other",
    "[labels] mode=shard index=3 of=8 extra", "[labels] device=cuda:0 extra", "[labels] device=CUDA", "[labels] device=cuda:123",
    "[labels] init_args=", "[labels] init_args=The heart is mildly enlarged", "[labels] init_args=a,b,c,d,e,f,g,h,i",
    "[labels] init_args=" + "a" * 33, "[labels] init_args=9lives",
    "[labels] canary: 0.012 s/report, projected 2345 s", "[labels] canary: 0.1 s/report, projected 2345 s for 10 rows",
    "[labels] wrote /sc/home/someone/chat_sessions/gallery/g/labels.npy", "[labels] wrote report_texts.txt", "[labels] wrote labels_shard_x.npy",
    "[labels] wrote labels.npy and the report", "[labels] labels_status=done, study 50000001", "[labels] labels_status=partial",
    "[labels] mismatch split: own_text=0", "[labels] shard 3 of 8 kept",
    "[labels] test_rows_sharing_a_text=12345678", "[labels] test_rows_sharing_a_text=-1", "[labels] test_rows_sharing_a_text=",
    "[labels] test_rows_sharing_a_text=x", "[labels] test_rows_sharing_a_text=2663 study_id=1", "[labels] test_rows_sharing_a_text=26.63",
    'RESULT {"refs_mismatch":0}', 'RESULT {"refs_mismatch":0,"labels_mismatch":0,"study_id":50000001}', 'RESULT {"study_id":50000001}',
    'RESULT {"refs_mismatch":-1,"labels_mismatch":0}', 'RESULT {"refs_mismatch":0.5,"labels_mismatch":0}',
    'RESULT {"refs_mismatch": 0,"labels_mismatch":0}', 'RESULT {"refs_mismatch":0,"labels_mismatch":0} ',
    'RESULT {"groups":163021,"rows":194125,"labelling_s":12345678}',
    "ERROR the heart is mildly enlarged", "ERROR no_manifest /sc/home/someone", "ERROR no_manifest study=12345678", "ERROR no_manifest key=effusion",
    "ERROR no_manifest key=1 key=2 key=3 key=4", "ERROR something_else", "ERROR shards_missing count=3 first=5 effusion",
    "ERROR failed", "ERROR failed the heart", "ERROR failed " + "A" * 61, "ERROR", "ERROR ",
    "=== note: chexbert_labels.json has no label_names key: CHEXBERT_14 order assumed === extra",
    "=== FAKE MIMIC TEXT /sc/home/someone/images/p10/img.jpg ===", "=== note: something else ===",
    "gallery] mode=single", "[labels]mode=single", "[Labels] mode=single", "Traceback (most recent call last):", "RuntimeError: x", "",
]


@pytest.mark.parametrize("name", [GPU_SH, CPU_SH])
def test_the_wrappers_allowlist_passes_the_known_lines_and_withholds_everything_else(name):
    assert grep_passes(LABEL_PASS, name) == LABEL_PASS
    assert grep_passes(LABEL_WITHHELD, name) == []
    for line in LABEL_PASS + LABEL_WITHHELD:      # the shapes written in the tests and the patterns in the wrapper say the same
        assert bool(LABEL_SHAPES.match(line)) == (grep_passes([line], name) == [line]), line
    assert [bool(LABEL_SHAPES.match(line)) for line in LABEL_PASS] == [True] * len(LABEL_PASS)


def test_both_wrappers_hold_the_same_allowlist():
    assert wrapper_patterns(GPU_SH) == wrapper_patterns(CPU_SH)


@pytest.mark.parametrize("name", [GPU_SH, CPU_SH])
def test_the_wrappers_allowlist_is_one_anchored_pattern_per_shape_with_nothing_free_in_it(name):
    patterns = wrapper_patterns(name)
    assert patterns == ["^" + shape + "$" for shape in LABEL_SHAPE_LIST], "the wrapper's patterns are the tests' shapes, anchored, in the same order"
    assert "" not in patterns, "an empty pattern matches every line"
    for pattern in patterns:
        assert pattern.startswith(("^\\[labels\\] ", "^RESULT ", "^ERROR ", "^=== note: ")) and pattern.endswith("$"), pattern
        bare = re.sub(r"\[[^\]]*\]", "", re.sub(r"\\.", "", pattern))      # without escaped characters, then bracket expressions
        assert not re.search(r"[.*+?]", bare), "a wildcard, an unbounded repeat or an optional part in " + pattern
        assert not re.search(r"\\[sSwWdD]", pattern), pattern


def test_the_error_codes_of_the_script_the_tests_and_the_wrappers_are_the_same_set(lg):
    assert sorted(lg.ERROR_CODES) == sorted(ERROR_CODES)
    tree = ast.parse(SCRIPT.read_text())
    literals = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    raised = {n.args[0].value for n in ast.walk(tree)
              if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "Refused" and n.args and isinstance(n.args[0], ast.Constant)}
    assert raised <= set(ERROR_CODES), "every raise names a code in the list"
    for code in ERROR_CODES:
        assert literals.count(code) >= 2, "{} is in ERROR_CODES and never used: a code the wrappers allow and nothing prints".format(code)
    for name in (GPU_SH, CPU_SH):
        (line,) = [p for p in wrapper_patterns(name) if p.startswith("^ERROR (")]
        assert sorted(re.match(r"\^ERROR \(([a-z_|]+)\)", line).group(1).split("|")) == sorted(ERROR_CODES), name


def run_scenarios(lg, capsys, install_fake, tmp_path, template) -> List[str]:
    """Everything the script prints, over the runs that matter: one line per line."""
    printed = []

    def go(world, *extra, **env):
        install_fake(**env)
        printed.extend(run_main(lg, capsys, world.argv(*extra)).lines)

    go(World(tmp_path / "single", template), "--canary", "10", "--progress", "30")
    sharded = World(tmp_path / "sharded", template)
    for i in range(2):
        go(sharded, "--shard", str(i), "--of", "2")
    go(sharded, "--shard", "1", "--of", "2")                                      # kept
    go(sharded, "--merge", "--of", "2")
    broken = World(tmp_path / "broken", template)                               # a cross-check that fails, over a shared text
    doctor_a_shared_representative(broken)
    go(broken)
    nameless = World(tmp_path / "nameless", template)
    payload = read_reference(nameless.reference)
    del payload["label_names"]
    write_reference_payload(nameless.reference, payload)
    go(nameless)
    permuted = World(tmp_path / "permuted", template)
    payload = read_reference(permuted.reference)
    perm = list(reversed(range(14)))
    payload["label_names"] = [NAMES[i] for i in perm]
    payload["y_true"] = [[row[i] for i in perm] for row in payload["y_true"]]
    write_reference_payload(permuted.reference, payload)
    go(permuted)
    for how in ("no_gate_file", "false"):                                         # refusals
        refused = World(tmp_path / ("refused_" + how), template)
        break_gate(refused.gallery, how)
        go(refused)
    return printed


def test_everything_the_script_prints_fits_a_shape_of_the_allowlist_and_each_shape_is_exercised(lg, capsys, install_fake, tmp_path, template):
    printed = run_scenarios(lg, capsys, install_fake, tmp_path, template)
    assert printed and all(LABEL_SHAPES.match(l) for l in printed), [l for l in printed if not LABEL_SHAPES.match(l)]
    for name in (GPU_SH, CPU_SH):
        assert grep_passes(printed, name) == printed, [l for l in printed if l not in grep_passes(printed, name)]
    not_exercised = [s for s in LABEL_SHAPE_LIST if not s.startswith(r"ERROR") and not any(re.fullmatch(s, l) for l in printed)]
    # the ones only a failure of the labeller itself prints are exercised in their own tests: a canary, a bad row, a death
    assert not_exercised == [], not_exercised
    # and no byte of any of them could be an id or a report: each is cut by the allowlist at the first thing it does not know. (A name at the
    # end of a line, the constructor's argument names and an exception's class name, may be a longer name: the one place a letter may follow.)
    for line in printed:
        assert grep_passes([line + " study_id=12345678"]) == [] and grep_passes([line + " x"]) == [], line
        if not line.startswith(("[labels] init_args=", "ERROR failed ")):
            assert grep_passes([line + "x"]) == [], line


def test_the_script_prints_no_text_no_id_and_no_path(lg, capsys, install_fake, tmp_path, template):
    """Over every run above: no piece of a report, no study id, none of the temp directory, and no run of 8 digits (an id)."""
    printed = "\n".join(run_scenarios(lg, capsys, install_fake, tmp_path, template))
    assert str(tmp_path) not in printed and not re.search(r"[0-9]{8}", printed)
    for text in read_texts(World(tmp_path / "probe", template).gallery):
        assert text[:30] not in printed
    assert "50000000" not in printed and "gallery" not in printed.replace("[labels]", "")


def test_the_cli_runs_as_the_wrappers_run_it_with_nothing_but_the_fake_on_its_path(tmp_path, template):
    """The script as the wrappers run it: `python scripts/label_gallery_reports.py ...` in a fresh interpreter, whose only import path beyond
    its own venv is the fake f1chexbert (so the script imports nothing of this repo, as it must in .venv_chexbert)."""
    site = tmp_path / "site"
    (site / "f1chexbert").mkdir(parents=True)
    (site / "f1chexbert" / "__init__.py").write_text(FAKE_F1CHEXBERT)
    world = World(tmp_path / "cli", template)
    done = subprocess.run([sys.executable, str(SCRIPT)] + world.argv(), cwd=str(tmp_path), capture_output=True, text=True, timeout=120,
                          env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(site), "PYTHONDONTWRITEBYTECODE": "1"})
    assert done.returncode == 0, done.stdout + done.stderr
    assert all(LABEL_SHAPES.match(l) for l in done.stdout.splitlines()), done.stdout
    assert world.manifest()["labels_status"] == "done"


# ── the wrappers, rehearsed in a temp tree ────────────────────────────────────

LABEL_PYTHON_STUB = """#!/bin/bash
# Stands in for the venv's python: records every call, plays the GPU probe (what the real one prints: the count, then a name or `none`), and
# runs everything else (the wrapper's path helper, the REAL scripts/label_gallery_reports.py, which imports the fake f1chexbert from PYTHONPATH)
# with the test interpreter.
""" + wr.RECORD_CALL + """case "$*" in
  *torch.cuda.device_count*) n="${FAKE_GPUS:-1}"; if [ "$n" = 0 ]; then echo "0 none"; else echo "$n NVIDIA H100 80GB HBM3"; fi; exit 0;;
esac
exec "$REAL_PYTHON" "$@"
"""


class LabelBox(wr.JobBox):
    """Both labelling wrappers run for real in the temp tree of tests/wrapper_rehearsal.py: the REAL script beside them, the fake f1chexbert
    on PYTHONPATH, an unlabelled verified tiny gallery in CHAT_HOME (build g13d_m3_v1, the plan's), and the published dump's refs.txt and
    chexbert_labels.json under results/ (repo/results is a symlink into the thesis checkout, as on the cluster)."""

    WRAPPER = GPU_SH
    SCRIPTS = ("label_gallery_reports.py", CPU_SH)
    VENV = ".venv_chexbert"
    PYTHON_STUB = LABEL_PYTHON_STUB
    SLURM_CPUS = "4"

    def __init__(self, root: Path, template: Path, stamp: Optional[str] = STAMP):
        self.template = template
        super().__init__(root, stamp)

    def populate(self) -> None:
        self.build_id = "g13d_m3_v1"
        self.fake_site = self.root / "fake_site"
        (self.fake_site / "f1chexbert").mkdir(parents=True)
        (self.fake_site / "f1chexbert" / "__init__.py").write_text(FAKE_F1CHEXBERT)
        shutil.copytree(str(self.template), str(self.gallery))
        (self.main / "results").mkdir()
        (self.repo / "results").symlink_to(self.main / "results", target_is_directory=True)
        write_reference(self.main / "results" / REFERENCE_NAME, self.gallery)

    @property
    def gallery(self) -> Path:
        return self.chat / "gallery" / self.build_id

    @property
    def reference(self) -> Path:
        return self.main / "results" / REFERENCE_NAME

    @property
    def mark(self) -> Path:
        return self.stubs / "fake.mark"

    def base_env(self) -> Dict[str, str]:
        env = super().base_env()
        env.update(BUILD_ID=self.build_id, PYTHONPATH=str(self.fake_site), FAKE_F1_MARK=str(self.mark), FAKE_F1_DEFAULT_DEVICE="cuda:0")
        return env

    def run_cpu(self, **extra_env: Optional[str]) -> subprocess.CompletedProcess:
        return self.run_script(CPU_SH, **extra_env)

    def marks(self) -> List[str]:
        return self.mark.read_text().splitlines() if self.mark.exists() else []

    def constructed(self) -> List[str]:
        return [m for m in self.marks() if m.startswith("constructed")]

    def labeller_calls(self) -> int:
        return sum(int(m[len("calls="):]) for m in self.marks() if m.startswith("calls="))

    def steps(self) -> List[List[str]]:
        """The job's own steps as the stub python saw them: the labelling script, with its arguments."""
        return [c for c in self.calls() if c and c[0] == "scripts/label_gallery_reports.py"]

    def ran_nothing(self) -> bool:
        return not self.steps()


@pytest.fixture
def box(tmp_path, template):
    return LabelBox(tmp_path, template)


REFERENCE_ARG = "results/" + REFERENCE_NAME
GPU_ARGV = ["--reference-dir", REFERENCE_ARG, "--budget-s", "5400", "--canary", "1000", "--device", "auto"]


def sync_line(done) -> str:
    return job_lines(done)[0]


def test_a_clean_gpu_run_labels_checks_and_prints_only_safe_lines(box):
    before = {name: snapshot(getattr(box, name)) for name in ("repo", "main", "data", "scratch")}
    gallery_before = snapshot(box.gallery)
    done = box.run(FAKE_F1_JUNK="1")
    lines = job_lines(done)
    assert done.returncode == 0, done.stdout
    # R7: only wrapper- and script-authored lines, and none of what the library printed besides lines of a known shape
    assert [l for l in lines if not LINE_OK.match(l)] == [], "a line that is not ===, [labels], RESULT or ERROR"
    assert not [l for l in lines if "FAKE" in l or "12345678" in l or "/sc/home" in l or "Traceback" in l or "heart" in l or "50000001" in l
                or "study_id" in l], lines
    assert lines[0] == "=== sync 3f2a9c41d7e86b05a1c4e9d3b7f60285ac9e1d47 clean ===", "the provenance comes first"
    groups = len(np.unique(np.load(str(box.gallery / "txt_groups.npy"))))
    labels = [l for l in lines if l.startswith("[labels]")]
    assert labels[:4] == ["[labels] mode=single", "[labels] rows=240 groups={} test=40".format(groups),
                          "[labels] init_args=refs_filename,hyps_filename,device,kwargs", "[labels] device=cuda:0"]
    assert lines.index("=== label lines withheld: 5 ===") > lines.index("[labels] device=cuda:0"), "five junk lines, counted, no more"
    (summary, final) = results(lines)
    assert summary["groups"] == groups and summary["rows"] == 240 and final == {"refs_mismatch": 0, "labels_mismatch": 0}
    assert lines[-1].startswith("=== END labels g13d_m3_v1: ") and "wall_s=" in lines[-1]
    assert all(len(l) <= 300 for l in lines if l.startswith(("RESULT ", "ERROR")))
    # the raw output is in a file in the gallery
    log = (box.gallery / "labels.log").read_text()
    assert "FAKE REPORT TEXT" in log and "[labels] study_id=12345678" in log, "stdout and stderr are in the file"
    # what it made
    texts = read_texts(box.gallery)
    assert np.load(str(box.gallery / "labels.npy")).tolist() == [FAKE.label_of(t) for t in texts]
    assert json.loads((box.gallery / "manifest.json").read_text())["labels_status"] == "done"
    assert json.loads((box.gallery / "labels_check.json").read_text())["labels_mismatch"] == 0
    # R8: nothing outside the gallery was touched, and inside it only the job's own files
    for name, snap in before.items():
        assert snapshot(getattr(box, name)) == snap, name
    assert changed_paths(gallery_before, snapshot(box.gallery)) == {"labels.npy", "label_names.json", "labels_check.json", "manifest.json",
                                                                    "labels.log"}
    assert [p.name for p in box.chat.iterdir()] == ["gallery"] and [p.name for p in (box.chat / "gallery").iterdir()] == ["g13d_m3_v1"]


def test_a_gpu_run_without_a_chattering_labeller_says_zero_withheld(box):
    """A 0 is evidence that the filter ran and found nothing to withhold; silence would not tell it from a filter that never ran."""
    done = box.run()
    lines = job_lines(done)
    assert done.returncode == 0, done.stdout
    assert "=== label lines withheld: 0 ===" in lines


@pytest.mark.parametrize("kind", ["gpu", "cpu"])
def test_a_progress_bar_in_front_of_a_line_does_not_hide_the_line(box, kind):
    """A bar ends in a carriage return, so the line the script prints next sits on the same physical line of the raw log: only a filter that
    splits on the carriage return (the tr in the wrapper) still sees it, and still lets only its shape through."""
    done = box.run(FAKE_F1_PROGRESS="1") if kind == "gpu" else box.run_cpu(SLURM_ARRAY_TASK_ID="0", FAKE_F1_PROGRESS="1")
    lines = job_lines(done)
    assert done.returncode == 0, done.stdout
    raw = (box.gallery / ("labels.log" if kind == "gpu" else "labels_shard_0.log")).read_bytes().decode()      # bytes: read_text() would turn \r into \n
    assert "1/2\r[labels] device=" in raw, "the bar and the line are one physical line in the raw log"
    assert "[labels] device=cuda:0" in lines or "[labels] device=cpu" in lines
    assert not [l for l in lines if "Loading" in l or "1/2" in l]
    assert [l for l in lines if not LINE_OK.match(l)] == []


def test_a_filter_that_cannot_count_says_unknown_and_not_zero(box):
    grep = box.bin / "grep"           # fails only for the counting call: every other grep is the real one
    grep.write_text('#!/bin/bash\ncase " $* " in *" -avcE "*) exit 2;; esac\n'
                    'for g in /usr/bin/grep /bin/grep; do [ -x "$g" ] && exec "$g" "$@"; done\nexit 127\n')
    grep.chmod(0o755)
    done = box.run()
    lines = job_lines(done)
    assert done.returncode == 0, done.stdout
    assert "=== label lines withheld: unknown ===" in lines and "=== label lines withheld: 0 ===" not in lines


def test_the_gpu_job_runs_the_script_with_the_published_defaults(box):
    assert box.run().returncode == 0
    (step,) = box.steps()
    assert step == ["scripts/label_gallery_reports.py", "--gallery", str(box.gallery)] + GPU_ARGV
    assert box.constructed() == ["constructed device=None hf_home=unset offline=0"], "no device argument, no HF_HOME, HF_HUB_OFFLINE=0"
    assert box.labeller_calls() == len(np.unique(np.load(str(box.gallery / "txt_groups.npy"))))


def test_the_chexbert_environment_is_that_of_score_chexbert_h100_and_a_caller_can_still_go_offline(box):
    assert box.run(HF_HUB_OFFLINE="1").returncode == 0
    assert box.constructed() == ["constructed device=None hf_home=unset offline=1"]


def test_the_build_the_reference_and_the_canary_can_be_chosen_on_the_submit_line(box):
    other = box.chat / "gallery" / "v2_13d_1"
    shutil.copytree(str(box.gallery), str(other))
    shutil.copytree(str(box.reference), str(box.main / "results" / "elsewhere"))
    done = box.run(BUILD_ID="v2_13d_1", REFERENCE_DIR="results/elsewhere", BUDGET_S="1234", CANARY="77")
    assert done.returncode == 0, done.stdout
    (step,) = box.steps()
    assert step == ["scripts/label_gallery_reports.py", "--gallery", str(other), "--reference-dir", "results/elsewhere", "--budget-s", "1234",
                    "--canary", "77", "--device", "auto"]
    assert json.loads((other / "manifest.json").read_text())["labels_status"] == "done"
    assert json.loads((box.gallery / "manifest.json").read_text())["labels_status"] == "pending", "the other build was not touched"
    assert [l for l in job_lines(done) if l.startswith("=== P5-C")] == [
        "=== P5-C CheXbert labels for gallery v2_13d_1: one job, F1CheXbert on the GPU if it takes one, canary budget 1234 s ==="]


def test_the_build_id_defaults_to_the_plans_gallery(box):
    done = box.run(BUILD_ID=None)
    assert done.returncode == 0, done.stdout
    assert json.loads((box.gallery / "manifest.json").read_text())["labels_status"] == "done"


def test_a_canary_over_budget_prints_the_two_submit_lines_of_the_cpu_path_and_exits_2(box):
    before = snapshot(box.gallery)
    done = box.run(FAKE_F1_SLEEP="0.01", CANARY="5", BUDGET_S="0")
    lines = job_lines(done)
    assert done.returncode == 2, done.stdout
    assert [l for l in lines if not LINE_OK.match(l)] == []
    assert any(re.fullmatch(r"\[labels\] canary: [0-9]+\.[0-9]{3} s/report, projected [0-9]+ s for [0-9]+ rows", l) for l in lines)
    here = lines.index("=== the canary projects past the 0 s budget: F1CheXbert is too slow here for one job ===")
    assert lines[here + 1:here + 4] == [
        "=== use the sharded CPU path, submit these two in order ===",
        "=== 1. bash scripts/chat_remote.sh submit scripts/label_gallery_reports_cpu_h100.sh BUILD_ID=g13d_m3_v1 -- --array=0-7 ===",
        "=== 2. bash scripts/chat_remote.sh submit scripts/label_gallery_reports_cpu_h100.sh BUILD_ID=g13d_m3_v1 MERGE=1 -- "
        "--dependency=afterok:<the array job id> ==="]
    assert not [l for l in lines if l.startswith("=== END") or l.startswith("ERROR")]
    assert changed_paths(before, snapshot(box.gallery)) == {"labels.log"}, "nothing but the raw log: the manifest is still pending"


@pytest.mark.parametrize("how", ["no_gpu", "canary"])
def test_the_two_submit_lines_of_an_exit_2_are_commands_the_real_chat_remote_submit_turns_into_the_intended_sbatch(tmp_path, template, how):
    """The two lines are for copying. Each, with the markers cut off and the array's job id filled in, goes through the real
    `chat_remote.sh submit` (ssh stubbed: tests/test_chat_remote.py's Sandbox), and what reaches the cluster is the one sbatch command meant:
    the array of 8 with BUILD_ID, then the merge with BUILD_ID, MERGE=1 and a dependency on the array."""
    import shlex
    from tests.test_chat_remote import Sandbox
    box = LabelBox(tmp_path / "job", template)
    done = box.run(FAKE_GPUS="0") if how == "no_gpu" else box.run(FAKE_F1_SLEEP="0.01", CANARY="5", BUDGET_S="0")
    assert done.returncode == 2, done.stdout
    printed = {l[4]: l[len("=== N. "):-len(" ===")] for l in job_lines(done) if re.match(r"=== [12]\. ", l)}
    assert sorted(printed) == ["1", "2"]
    sandbox = Sandbox(tmp_path / "submit")
    shutil.copy(str(REPO_ROOT / "scripts" / CPU_SH), str(sandbox.repo / "scripts" / CPU_SH))
    for number in ("1", "2"):
        words = shlex.split(printed[number].replace("<the array job id>", "2634000"))
        assert words[:3] == ["bash", "scripts/chat_remote.sh", "submit"], words
        sent = sandbox.run(*words[2:])
        assert sent.returncode == 0, sent.stderr
    calls = [c[2] for c in sandbox.calls()]
    assert calls == [
        "cd '/fake/hybrid_chat_ui' && env 'BUILD_ID=g13d_m3_v1' sbatch --parsable '--array=0-7' 'scripts/label_gallery_reports_cpu_h100.sh'",
        "cd '/fake/hybrid_chat_ui' && env 'BUILD_ID=g13d_m3_v1' 'MERGE=1' sbatch --parsable '--dependency=afterok:2634000' "
        "'scripts/label_gallery_reports_cpu_h100.sh'"]


def test_a_labeller_that_dies_prints_its_class_and_its_exit_code_and_no_text(box):
    done = box.run(FAKE_F1_DIE_AFTER="7")
    lines = job_lines(done)
    assert done.returncode == 1, done.stdout
    assert "ERROR failed RuntimeError" in lines and "ERROR labels exit=1" in lines
    assert [l for l in lines if not LINE_OK.match(l)] == [] and not [l for l in lines if "died" in l or "Findings" in l or "Traceback" in l]
    assert "fake labeller died on" in (box.gallery / "labels.log").read_text(), "the traceback is in the file"
    assert json.loads((box.gallery / "manifest.json").read_text())["labels_status"] == "pending"
    assert not any(l.startswith("=== END") for l in lines)


def test_a_failed_cross_check_fails_the_job_with_the_counts_in_the_log(box):
    payload = read_reference(box.reference)
    payload["y_true"][5][3] ^= 1
    write_reference_payload(box.reference, payload)
    done = box.run()
    lines = job_lines(done)
    assert done.returncode == 1, done.stdout
    assert results(lines)[-1] == {"refs_mismatch": 0, "labels_mismatch": 1} and "ERROR labels exit=1" in lines
    assert "[labels] labels_status=pending" in lines and not any(l.startswith("=== END") for l in lines)
    assert not (box.gallery / "labels.npy").exists() and (box.gallery / "labels_unverified.npy").is_file()


def test_a_gallery_whose_gate_is_not_equal_is_refused_by_the_script_before_the_labeller_exists(box):
    break_gate(box.gallery, "false")
    done = box.run()
    lines = job_lines(done)
    assert done.returncode == 1 and "ERROR gate_not_equal" in lines and "ERROR labels exit=1" in lines, lines
    assert box.constructed() == [] and len(box.steps()) == 1


def test_a_finished_gallery_is_refused_by_the_wrapper_before_it_touches_a_file(box):
    manifest = json.loads((box.gallery / "manifest.json").read_text())
    manifest["labels_status"] = "done"
    (box.gallery / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (box.gallery / "labels.npy").write_bytes(b"the finished labels")
    (box.gallery / "labels.log").write_text("the finished job's raw log\n")
    before = snapshot(box.gallery)
    for run in (box.run, box.run_cpu):
        done = run(SLURM_ARRAY_TASK_ID="0")
        assert done.returncode == 1, done.stdout
        assert any(l.startswith("ERROR") and "already labelled" in l for l in job_lines(done)), job_lines(done)
        assert snapshot(box.gallery) == before, "not even the raw log of the finished job was overwritten"
    assert box.ran_nothing()


@pytest.mark.parametrize("env, needle", [
    ({"BUILD_ID": "../escape"}, "BUILD_ID"), ({"BUILD_ID": "a/b"}, "BUILD_ID"), ({"BUILD_ID": ".."}, "BUILD_ID"), ({"BUILD_ID": "-x"}, "BUILD_ID"),
    ({"BUILD_ID": "x y"}, "BUILD_ID"), ({"BUILD_ID": "x;y"}, "BUILD_ID"), ({"BUILD_ID": "x$(id)"}, "BUILD_ID"),
    ({"BUDGET_S": "-1"}, "BUDGET_S"), ({"BUDGET_S": "1e3"}, "BUDGET_S"), ({"BUDGET_S": "1234567"}, "BUDGET_S"),
    ({"CANARY": "x"}, "CANARY"), ({"CANARY": "-5"}, "CANARY"), ({"CANARY": "1 2"}, "CANARY")])
def test_a_lever_that_is_not_what_it_should_be_is_refused_before_anything_runs(box, env, needle):
    done = box.run(**env)
    lines = job_lines(done)
    assert done.returncode == 1, done.stdout
    assert [l for l in lines if l.startswith("ERROR")] and needle in "\n".join(l for l in lines if l.startswith("ERROR"))
    assert box.ran_nothing() and not (box.gallery / "labels.log").exists()
    assert [l for l in lines if not LINE_OK.match(l)] == [], "no bash complaint"


def test_a_missing_chat_home_gallery_manifest_venv_or_reference_stops_the_job_before_anything_runs(box):
    cases = [(box.chat, "CHAT_HOME"), (box.gallery, "manifest.json"), (box.gallery / "manifest.json", "manifest.json"),
             (box.repo / ".venv_chexbert" / "bin" / "activate", "venv"), (box.reference / "refs.txt", "refs.txt"),
             (box.reference / "chexbert_labels.json", "chexbert_labels.json")]
    for path, needle in cases:
        moved = Path(str(path) + ".away")
        path.rename(moved)
        for run in (box.run, box.run_cpu):
            done = run(SLURM_ARRAY_TASK_ID="0")
            errors = [l for l in job_lines(done) if l.startswith("ERROR")]
            assert done.returncode == 1 and errors and needle in errors[0], (path.name, job_lines(done))
            assert box.ran_nothing(), path.name
            assert [l for l in job_lines(done) if not LINE_OK.match(l)] == [], "no bash complaint"
        moved.rename(path)


@pytest.mark.parametrize("kind", ["gpu", "cpu"])
def test_a_tree_without_the_labelling_script_is_refused_before_anything_runs(box, kind):
    """A sync that never happened, or a tree of another branch. Python would say `can't open file` and exit 2, which is also the code
    the wrappers read as a hand-over to the CPU path: the guard names the cause, before anything is opened."""
    (box.repo / "scripts" / "label_gallery_reports.py").unlink()
    before = snapshot(box.gallery)
    done = box.run() if kind == "gpu" else box.run_cpu(SLURM_ARRAY_TASK_ID="0")
    errors = [l for l in job_lines(done) if l.startswith("ERROR")]
    assert done.returncode == 1 and len(errors) == 1, job_lines(done)
    assert "label_gallery_reports.py is missing" in errors[0] and "sync" in errors[0]
    assert box.ran_nothing() and snapshot(box.gallery) == before
    assert [l for l in job_lines(done) if not LINE_OK.match(l)] == []


def replace_the_script(box: LabelBox, body: str) -> None:
    """Put a script of the test's own where the job finds scripts/label_gallery_reports.py (the box's copy of the real one)."""
    (box.repo / "scripts" / "label_gallery_reports.py").write_text(body)


CANARY_BEFORE_EXIT_2 = [
    # what the raw log holds when the script exits 2, and whether that is the script's canary line
    ("[labels] canary: 0.123 s/report, projected 99999 s for 78 rows\n", True),
    ("Loading weights:  50%|#####     | 1/2\r[labels] canary: 0.123 s/report, projected 99999 s for 78 rows\n", True),      # glued behind a bar
    ("[labels] canary: study_id=12345678\n", False),                                    # not the shape of the line
    ("[labels] canary: 0.123 s/report, projected 99999 s for 78 rows and some text\n", False),
    ("canary: 0.123 s/report, projected 99999 s for 78 rows\n", False),                # not a [labels] line
    ("", False)]


@pytest.mark.parametrize("kind", ["gpu", "cpu"])
@pytest.mark.parametrize("printed, is_the_canary", CANARY_BEFORE_EXIT_2)
def test_exit_2_is_the_canarys_only_when_the_raw_log_holds_the_canary_line(box, kind, printed, is_the_canary):
    """Python itself exits 2 when it cannot open a script, and a library may exit 2: only the script's own canary line, of its known shape,
    in the raw log makes exit 2 a hand-over to the CPU path. Anything else is a failure, `ERROR labels exit=2`, and the wrapper itself
    exits 1, so that an exit 2 of this wrapper always means 'submit the CPU path'."""
    replace_the_script(box, "import sys\nsys.stdout.write({!r})\nsys.exit(2)\n".format(printed))
    done = box.run() if kind == "gpu" else box.run_cpu(SLURM_ARRAY_TASK_ID="0")
    lines = job_lines(done)
    assert [l for l in lines if not LINE_OK.match(l)] == [], lines
    handed_over = [l for l in lines if "sharded CPU path" in l or "use more shards" in l or l.startswith(("=== 1. ", "=== 2. "))]
    if is_the_canary:
        assert done.returncode == 2 and handed_over and "ERROR labels exit=2" not in lines, done.stdout
    else:
        assert done.returncode == 1 and "ERROR labels exit=2" in lines and not handed_over, done.stdout


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root can read a file whatever its mode")
@pytest.mark.parametrize("kind", ["gpu", "cpu"])
def test_python_unable_to_open_the_script_exits_2_and_that_is_a_failure_not_a_hand_over(box, kind):
    """The real collision: a script that is there (so the guard passes) and that python cannot open, which it answers with exit 2."""
    script = box.repo / "scripts" / "label_gallery_reports.py"
    script.chmod(0)
    try:
        done = box.run() if kind == "gpu" else box.run_cpu(SLURM_ARRAY_TASK_ID="0")
    finally:
        script.chmod(0o644)
    lines = job_lines(done)
    assert done.returncode == 1 and "ERROR labels exit=2" in lines, done.stdout
    assert not [l for l in lines if "sharded CPU path" in l or "use more shards" in l or "canary projects" in l]
    assert [l for l in lines if not LINE_OK.match(l)] == [] and not [l for l in lines if "Errno" in l or "can't open" in l]
    log = box.gallery / ("labels.log" if kind == "gpu" else "labels_shard_0.log")
    assert "can't open file" in log.read_text(), "python's own message is in the raw log"


def test_when_the_venvs_torch_sees_no_gpu_the_gpu_job_hands_over_the_cpu_path_at_once_and_labels_nothing(box):
    """setup_chexbert_venv_h100.sh installs a CPU-only torch, with which F1CheXbert never takes a GPU: the one-job path would only run the
    canary on the CPU. The job says so, prints the two submit lines and exits 2 (not 1: nothing is wrong), before any step and any file."""
    before = snapshot(box.gallery)
    done = box.run(FAKE_GPUS="0")
    lines = job_lines(done)
    assert done.returncode == 2, done.stdout
    assert "=== gpus=0 (none) ===" in lines
    here = lines.index("=== torch in this venv sees no GPU, so F1CheXbert would run on the CPU: the one-job path is skipped, and nothing was labelled ===")
    assert lines[here + 1:] == [
        "=== use the sharded CPU path, submit these two in order ===",
        "=== 1. bash scripts/chat_remote.sh submit scripts/label_gallery_reports_cpu_h100.sh BUILD_ID=g13d_m3_v1 -- --array=0-7 ===",
        "=== 2. bash scripts/chat_remote.sh submit scripts/label_gallery_reports_cpu_h100.sh BUILD_ID=g13d_m3_v1 MERGE=1 -- "
        "--dependency=afterok:<the array job id> ==="]
    assert [l for l in lines if not LINE_OK.match(l)] == [] and not any(l.startswith("ERROR") for l in lines)
    assert box.ran_nothing() and snapshot(box.gallery) == before, "no step ran and no file, not even the raw log, was made"


def test_a_probe_that_cannot_run_torch_reads_as_no_gpu_and_the_cpu_job_does_not_care_either_way(box):
    """The probe is the venv's torch asked for a count: a venv whose torch will not import gives none, and the answer is the same."""
    probe = box.bin / "python"
    probe.write_text("#!/bin/bash\n" + wr.RECORD_CALL + 'case "$*" in *torch.cuda.device_count*) echo "Traceback FAKE" >&2; exit 1;; esac\n'
                     'exec "$REAL_PYTHON" "$@"\n')
    done = box.run()
    lines = job_lines(done)
    assert done.returncode == 2 and "=== gpus=0 (none) ===" in lines and not any("FAKE" in l for l in lines), done.stdout
    assert box.run_cpu(SLURM_ARRAY_TASK_ID="0").returncode == 0, "the CPU job never asks"


@pytest.mark.parametrize("where", ["inside_outputs", "under_main", "outputs_elsewhere", "symlink_into_main", "symlink_named_outputs",
                                   "symlink_to_outputs_elsewhere"])
def test_a_gallery_in_an_outputs_directory_or_under_the_thesis_checkout_is_refused(box, tmp_path, where):
    """CHAT_HOME is made so that the gallery the build wrote is where the job looks, and the guards of the build wrapper apply."""
    if where == "inside_outputs":
        home = box.main / "outputs" / "chat"
    elif where == "under_main":
        home = box.main / "chat"
    elif where == "outputs_elsewhere":
        home = tmp_path / "elsewhere" / "outputs" / "chat"
    elif where == "symlink_into_main":
        (box.main / "chat").mkdir()
        home = tmp_path / "chat_link"
        home.symlink_to(box.main / "chat", target_is_directory=True)
    elif where == "symlink_named_outputs":
        (tmp_path / "real_place" / "chat").mkdir(parents=True)
        (tmp_path / "outputs").symlink_to(tmp_path / "real_place", target_is_directory=True)
        home = tmp_path / "outputs" / "chat"
    else:
        (tmp_path / "other" / "outputs" / "chat").mkdir(parents=True)
        home = tmp_path / "plain_name"
        home.symlink_to(tmp_path / "other" / "outputs" / "chat", target_is_directory=True)
    if where in ("inside_outputs", "under_main", "outputs_elsewhere"):
        home.mkdir(parents=True, exist_ok=True)
    (home / "gallery").mkdir()
    shutil.copytree(str(box.gallery), str(home / "gallery" / box.build_id))
    before_main = snapshot(box.main)
    before_gallery = snapshot(home / "gallery" / box.build_id)
    for run in (box.run, box.run_cpu):
        done = run(CHAT_HOME=str(home), SLURM_ARRAY_TASK_ID="0")
        errors = [l for l in job_lines(done) if l.startswith("ERROR")]
        assert done.returncode == 1 and any("outputs" in l or "thesis checkout" in l for l in errors), (errors, job_lines(done))
    assert box.ran_nothing()
    assert snapshot(box.main) == before_main and snapshot(home / "gallery" / box.build_id) == before_gallery


def test_a_requeued_job_after_a_death_starts_the_labelling_again_and_replaces_the_raw_log(box):
    first = box.run(FAKE_F1_DIE_AFTER="7")
    assert first.returncode == 1 and (box.gallery / "labels.log").is_file()
    assert "fake labeller died on" in (box.gallery / "labels.log").read_text()
    second = box.run(SLURM_RESTART_COUNT="1")
    lines = job_lines(second)
    assert second.returncode == 0, second.stdout
    assert any("restart=1" in l and l.startswith("=== job=") for l in lines)
    assert "fake labeller died on" not in (box.gallery / "labels.log").read_text(), "the raw log is the latest attempt's"
    assert json.loads((box.gallery / "manifest.json").read_text())["labels_status"] == "done"
    assert np.load(str(box.gallery / "labels.npy")).tolist() == [FAKE.label_of(t) for t in read_texts(box.gallery)]
    assert len(box.steps()) == 2


def test_a_requeue_after_a_failed_cross_check_may_run_again_and_succeed(box):
    good = read_reference(box.reference)
    broken = json.loads(json.dumps(good))
    broken["y_true"][5][3] ^= 1
    write_reference_payload(box.reference, broken)
    assert box.run().returncode == 1
    write_reference_payload(box.reference, good)
    assert box.run(SLURM_RESTART_COUNT="1").returncode == 0
    assert (box.gallery / "labels.npy").is_file() and (box.gallery / "labels_unverified.npy").is_file()


@pytest.mark.parametrize("dirty", [False, True])
def test_the_first_line_of_the_job_log_names_the_commit_and_cleanliness_the_tree_was_synced_with(tmp_path, template, dirty):
    stamp = real_stamp(tmp_path / "sync", dirty)
    assert STAMP_RE.match(stamp), stamp
    _, sha, flag = stamp.split()
    for kind in ("gpu", "cpu"):
        box = LabelBox(tmp_path / ("job_" + kind), template, stamp=stamp)
        done = box.run(FAKE_GPUS="0") if kind == "gpu" else box.run_cpu(BUILD_ID="not a name")          # both stop early: the first line is all this asks
        assert sync_line(done) == "=== sync {} {} ===".format(sha, flag), (kind, job_lines(done)[:2])


def test_a_missing_sync_stamp_is_reported_as_unknown_and_the_job_goes_on(tmp_path, template):
    box = LabelBox(tmp_path, template, stamp=None)
    done = box.run()
    assert done.returncode == 0, done.stdout
    assert sync_line(done) == "=== sync unknown ==="


@pytest.mark.parametrize("stamp", [
    "",
    "rm -rf / ; echo $(whoami) FAKE_STAMP_TEXT",
    "2026-10-09T20:00:00Z 3f2a9c41d7e86b05a1c4e9d3b7f60285ac9e1d4 clean",
    "2026-10-09T20:00:00Z 3F2A9C41D7E86B05A1C4E9D3B7F60285AC9E1D47 clean",
    "2026-10-09T20:00:00Z 3f2a9c41d7e86b05a1c4e9d3b7f60285ac9e1d47 maybe",
    "2026-10-09T20:00:00Z 3f2a9c41d7e86b05a1c4e9d3b7f60285ac9e1d47 clean FAKE_EXTRA"])
def test_a_malformed_sync_stamp_is_reported_as_unknown_and_never_echoed(tmp_path, template, stamp):
    box = LabelBox(tmp_path, template, stamp=stamp)
    lines = job_lines(box.run(FAKE_GPUS="0"))
    assert lines[0] == "=== sync unknown ==="
    assert not [l for l in lines if "FAKE_" in l or "whoami" in l or "rm -rf" in l]


# the CPU wrapper

CPU_SHARD_ARGV = ["--reference-dir", REFERENCE_ARG, "--budget-s", "6000", "--canary", "1000", "--device", "cpu"]


def test_a_cpu_shard_labels_its_rows_writes_its_files_and_prints_only_safe_lines(box):
    before = {name: snapshot(getattr(box, name)) for name in ("repo", "main", "data", "scratch")}
    gallery_before = snapshot(box.gallery)
    done = box.run_cpu(SLURM_ARRAY_TASK_ID="3", FAKE_F1_JUNK="1", FAKE_F1_DEFAULT_DEVICE="cuda:0")
    lines = job_lines(done)
    assert done.returncode == 0, done.stdout
    assert [l for l in lines if not LINE_OK.match(l)] == []
    assert not [l for l in lines if "FAKE" in l or "12345678" in l or "/sc/home" in l or "Traceback" in l or "heart" in l]
    assert lines[0].startswith("=== sync ") and "=== label lines withheld: 5 ===" in lines
    (step,) = box.steps()
    assert step == ["scripts/label_gallery_reports.py", "--gallery", str(box.gallery)] + CPU_SHARD_ARGV + ["--shard", "3", "--of", "8"]
    assert "[labels] device=cpu" in lines and "[labels] mode=shard index=3 of=8" in lines, "the CPU job asks for the CPU, whatever the library prefers"
    assert box.constructed() == ["constructed device=cpu hf_home=unset offline=0"]
    reps = np.unique(np.load(str(box.gallery / "txt_groups.npy")), return_index=True)[1]
    assert np.load(str(box.gallery / "labels_shard_3_rows.npy")).tolist() == reps[3::8].tolist()
    assert box.labeller_calls() == len(reps[3::8])
    (summary,) = results(lines)
    assert summary["shard"] == 3 and summary["of"] == 8 and summary["groups"] == len(reps[3::8])
    assert lines[-1].startswith("=== END labels shard 3 of 8") and "wall_s=" in lines[-1]
    assert "FAKE REPORT TEXT" in (box.gallery / "labels_shard_3.log").read_text()
    for name, snap in before.items():
        assert snapshot(getattr(box, name)) == snap, name
    assert changed_paths(gallery_before, snapshot(box.gallery)) == {"labels_shard_3.npy", "labels_shard_3_rows.npy", "labels_shard_3.log"}


def test_eight_cpu_shards_then_the_merge_make_what_the_gpu_job_makes(tmp_path, template):
    gpu, cpu = LabelBox(tmp_path / "gpu", template), LabelBox(tmp_path / "cpu", template)
    assert gpu.run().returncode == 0
    for i in range(8):
        done = cpu.run_cpu(SLURM_ARRAY_TASK_ID=str(i))
        assert done.returncode == 0, done.stdout
    assert not (cpu.gallery / "labels.npy").exists(), "no shard finishes the gallery"
    merged = cpu.run_cpu(MERGE="1")
    lines = job_lines(merged)
    assert merged.returncode == 0, merged.stdout
    assert [l for l in lines if not LINE_OK.match(l)] == []
    (summary, final) = results(lines)
    assert summary["merged"] == 8 and final == {"refs_mismatch": 0, "labels_mismatch": 0}
    for name in ("labels.npy", "label_names.json"):
        assert (cpu.gallery / name).read_bytes() == (gpu.gallery / name).read_bytes(), name
    assert json.loads((cpu.gallery / "manifest.json").read_text())["labels_status"] == "done"
    assert cpu.steps()[-1] == ["scripts/label_gallery_reports.py", "--gallery", str(cpu.gallery), "--reference-dir", REFERENCE_ARG, "--merge", "--of", "8"]
    assert lines[-1].startswith("=== END labels merge of 8")
    assert "FAKE" not in (cpu.gallery / "labels_merge.log").read_text() and (cpu.gallery / "labels_merge.log").is_file()
    assert cpu.constructed() == ["constructed device=cpu hf_home=unset offline=0"] * 8, "the merge builds no labeller"


def test_a_shard_that_is_already_labelled_is_kept_when_its_job_runs_again(box):
    assert box.run_cpu(SLURM_ARRAY_TASK_ID="5").returncode == 0
    files = {n: (box.gallery / n).read_bytes() for n in ("labels_shard_5.npy", "labels_shard_5_rows.npy")}
    stamps = {n: (box.gallery / n).stat().st_mtime_ns for n in files}
    done = box.run_cpu(SLURM_ARRAY_TASK_ID="5", SLURM_RESTART_COUNT="1")
    lines = job_lines(done)
    assert done.returncode == 0 and "[labels] shard 5 of 8 kept: already labelled" in lines, done.stdout
    assert any("restart=1" in l and l.startswith("=== job=") for l in lines)
    assert len(box.constructed()) == 1, "the second run built no labeller"
    for name, data in files.items():
        assert (box.gallery / name).read_bytes() == data and (box.gallery / name).stat().st_mtime_ns == stamps[name], name


def test_a_shard_that_dies_leaves_no_files_and_its_rerun_completes_the_run(box):
    """The requeue path of a shard that was killed mid-labelling: nothing of it is on disk (the files are written whole, the labels last), so
    the next attempt labels it from the start, and the merge sees a complete set."""
    for i in range(8):
        if i == 4:
            dead = box.run_cpu(SLURM_ARRAY_TASK_ID="4", FAKE_F1_DIE_AFTER="3")
            assert dead.returncode == 1 and "ERROR failed RuntimeError" in job_lines(dead) and "ERROR labels exit=1" in job_lines(dead)
            assert not [p.name for p in box.gallery.iterdir() if p.name.startswith("labels_shard_4") and p.suffix == ".npy"]
            assert not list(box.gallery.glob("*.tmp"))
            assert (box.gallery / "labels_shard_4.log").is_file(), "its raw log, with the traceback, is there for the diagnosis"
        assert box.run_cpu(SLURM_ARRAY_TASK_ID=str(i), SLURM_RESTART_COUNT="1" if i == 4 else "0").returncode == 0
    assert box.run_cpu(MERGE="1").returncode == 0
    assert json.loads((box.gallery / "manifest.json").read_text())["labels_status"] == "done"
    assert "fake labeller died on" not in (box.gallery / "labels_shard_4.log").read_text(), "the rerun's raw log replaced the dead one's"


def test_a_partial_cpu_run_is_completed_by_the_missing_shards_and_the_merge(box):
    for i in range(7):
        assert box.run_cpu(SLURM_ARRAY_TASK_ID=str(i)).returncode == 0
    refused = box.run_cpu(MERGE="1")
    assert refused.returncode == 1 and "ERROR shards_missing count=1 first=7" in job_lines(refused), job_lines(refused)
    assert "ERROR labels exit=1" in job_lines(refused) and not (box.gallery / "labels.npy").exists()
    assert box.run_cpu(SLURM_ARRAY_TASK_ID="7").returncode == 0
    assert box.run_cpu(MERGE="1").returncode == 0
    assert json.loads((box.gallery / "manifest.json").read_text())["labels_status"] == "done"


def test_a_failed_cross_check_in_the_merge_fails_that_job(box):
    payload = read_reference(box.reference)
    payload["y_true"][0][0] ^= 1
    write_reference_payload(box.reference, payload)
    for i in range(8):
        assert box.run_cpu(SLURM_ARRAY_TASK_ID=str(i)).returncode == 0
    done = box.run_cpu(MERGE="1")
    lines = job_lines(done)
    assert done.returncode == 1 and results(lines)[-1] == {"refs_mismatch": 0, "labels_mismatch": 1} and "ERROR labels exit=1" in lines
    assert json.loads((box.gallery / "manifest.json").read_text())["labels_status"] == "pending"


@pytest.mark.parametrize("env, needle", [
    ({"SLURM_ARRAY_TASK_ID": None}, "array task"), ({"SLURM_ARRAY_TASK_ID": "x"}, "array task"), ({"SLURM_ARRAY_TASK_ID": "8"}, "outside"),
    ({"SLURM_ARRAY_TASK_ID": "0", "SHARDS": "0"}, "SHARDS"), ({"SLURM_ARRAY_TASK_ID": "0", "SHARDS": "x"}, "SHARDS"),
    ({"SLURM_ARRAY_TASK_ID": "0", "SHARDS": "1000"}, "SHARDS"), ({"SLURM_ARRAY_TASK_ID": "0", "MERGE": "2"}, "MERGE"),
    ({"SLURM_ARRAY_TASK_ID": "0", "MERGE": "yes"}, "MERGE"), ({"SLURM_ARRAY_TASK_ID": "0", "BUILD_ID": "a/b"}, "BUILD_ID"),
    ({"SLURM_ARRAY_TASK_ID": "0", "BUDGET_S": "x"}, "BUDGET_S"), ({"SLURM_ARRAY_TASK_ID": "0", "CANARY": "x"}, "CANARY")])
def test_the_cpu_job_refuses_an_array_or_a_lever_that_is_not_what_it_should_be(box, env, needle):
    done = box.run_cpu(**env)
    errors = [l for l in job_lines(done) if l.startswith("ERROR")]
    assert done.returncode == 1 and errors and needle in errors[0], job_lines(done)
    assert box.ran_nothing() and [l for l in job_lines(done) if not LINE_OK.match(l)] == []


def test_the_shard_count_is_a_lever_and_a_merge_does_not_need_an_array_task(box):
    for i in range(3):
        assert box.run_cpu(SLURM_ARRAY_TASK_ID=str(i), SHARDS="3").returncode == 0
    done = box.run_cpu(SHARDS="3", MERGE="1")
    assert done.returncode == 0, done.stdout
    assert box.steps()[-1][-3:] == ["--merge", "--of", "3"]


def test_a_cpu_shard_whose_canary_is_over_budget_says_to_use_more_shards_and_exits_2(box):
    done = box.run_cpu(SLURM_ARRAY_TASK_ID="0", FAKE_F1_SLEEP="0.01", CANARY="5", BUDGET_S="0")
    lines = job_lines(done)
    assert done.returncode == 2, done.stdout
    assert "=== the canary projects past the 0 s budget: this shard cannot finish in its time limit, use more shards (SHARDS=16, --array=0-15) ===" in lines
    assert [l for l in lines if not LINE_OK.match(l)] == [] and not any(l.startswith("=== END") for l in lines)
    assert not (box.gallery / "labels_shard_0.npy").exists()


def test_the_cpu_job_names_its_task_in_the_log(box):
    lines = job_lines(box.run_cpu(SLURM_ARRAY_TASK_ID="2"))
    assert any(re.fullmatch(r"=== job=1234567 restart=0 node=\S+ task=2 ===", l) for l in lines), lines
    assert any(re.fullmatch(r"=== P5-C CheXbert labels for gallery g13d_m3_v1: shard 2 of 8 on the CPU ===", l) for l in lines), lines


@pytest.mark.parametrize("name", [GPU_SH, CPU_SH])
def test_the_jobs_never_delete_anything(name):
    src = (REPO_ROOT / "scripts" / name).read_text()
    code = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))
    assert not re.search(r"(^|[\s;&|(])rm(\s|$)", code, re.M) and "--delete" not in code and "ln -sf" not in code
