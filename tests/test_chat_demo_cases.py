"""CHAT_UI_PLAN.md P9-C, part 1: scripts/chat_demo_cases.py, which picks the test studies for the supervisor demo, and its CPU job
scripts/chat_demo_cases_h100.sh.

The job reads the published run's per-sample CheXbert labels (`chexbert_labels.json`, written by scripts/score_chexbert_standalone.py: the
keys `y_true`, `y_pred`, `label_names`, `five_label_indices`, `hyp_file` and `ref_file`), takes the reference labels `y_true` (one row of 14 per
test study, in test-split order), and lists up to five candidate test rows for each of three kinds of study: normal, support devices and
pneumothorax. The draw is seeded and fair: it looks at nothing but `y_true`. It prints row numbers and counts only (rule R7: no MIMIC text,
image or study id ever reaches an agent), and it refuses when the number of studies is not the gallery's test-split count, because a row
number must name the study the picker shows.

Everything here is synthetic: a labels file built in the test from a table of kinds, the tiny gallery of P5-B, and a temp tree standing in
for the cluster. No MIMIC file is read, so an assertion may print whatever it compares.
"""
import ast
import io
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from app.labels import CHEXBERT_14
from scripts import chat_demo_cases as cases
from tests import wrapper_rehearsal as wr
from tests.wrapper_rehearsal import job_lines, safe_line

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "chat_demo_cases.py"
WRAPPER = "chat_demo_cases_h100.sh"
SECRET = "SECRET-MIMIC-text-or-study-id"


def row_of(*names):
    """One row of CHEXBERT_14 labels with the named findings positive."""
    assert set(names) <= set(CHEXBERT_14), names
    return [int(name in names) for name in CHEXBERT_14]


KINDS = {
    "normal": row_of("No Finding"),
    "support": row_of("Support Devices", "Lung Opacity"),
    "pneumo": row_of("Pneumothorax"),
    "both": row_of("Pneumothorax", "Support Devices", "Pleural Effusion"),
    "other": row_of("Cardiomegaly"),
    "nofinding_plus": row_of("No Finding", "Edema"),   # No Finding and something else: not normal
    "nothing": row_of(),                               # no label at all: not normal either
}
CYCLE = ["normal", "support", "pneumo", "both", "other", "nofinding_plus", "nothing"]   # seven kinds, repeated down the rows


def kinds_of(n):
    return [CYCLE[row % len(CYCLE)] for row in range(n)]


def y_true_of(n):
    return [list(KINDS[kind]) for kind in kinds_of(n)]


def expected_rows(n):
    """What a reader of the plan gets by hand from the kinds: the rows of each category, in row order."""
    kinds = kinds_of(n)
    return {"normal": [r for r, k in enumerate(kinds) if k == "normal"],
            "support_devices": [r for r, k in enumerate(kinds) if k in ("support", "both")],
            "pneumothorax": [r for r, k in enumerate(kinds) if k in ("pneumo", "both")]}


def write_labels(path, y_true, **extra):
    """A chexbert_labels.json as scripts/score_chexbert_standalone.py writes it, with `extra` keys replacing or adding to its own."""
    payload = {"y_true": y_true, "y_pred": [list(row) for row in y_true], "label_names": list(CHEXBERT_14), "five_label_indices": [0, 1, 2, 3, 4],
               "hyp_file": "results/synthetic/hyps.txt", "ref_file": "results/synthetic/refs.txt"}
    payload.update(extra)
    path = Path(path)
    path.write_text(json.dumps(payload))
    return path


def write_manifest(directory, test=40, images=200, report_rows=None, **extra):
    """A gallery directory holding only the manifest this job reads."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {"counts": {"images": images, "report_rows": images + test if report_rows is None else report_rows, "test": test,
                           "report_groups": 10}}
    manifest.update(extra)
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return directory


def run_cases(args, expect_code=None):
    """main() with its own stream. -> (code, stdout)."""
    out = io.StringIO()
    code = cases.main(list(args), out)
    if expect_code is not None:
        assert code == expect_code, (code, out.getvalue())
    return code, out.getvalue()


def the_result(stdout):
    lines = stdout.splitlines()
    assert len(lines) == 1 and lines[0].startswith("RESULT "), lines
    return json.loads(lines[0][len("RESULT "):])


@pytest.fixture
def world(tmp_path, tiny_gallery):
    """The tiny gallery (40 test studies) and a labels file of 40 rows, the kinds in turn."""
    n = json.loads((tiny_gallery / "manifest.json").read_text())["counts"]["test"]
    return {"labels": write_labels(tmp_path / "chexbert_labels.json", y_true_of(n)), "gallery": tiny_gallery, "n": n, "tmp": tmp_path}


def args_of(world):
    return ["--labels", str(world["labels"]), "--gallery", str(world["gallery"])]


# ---- what the job reads: the layout of chexbert_labels.json ----------------------------------------------------------------------

def test_the_labels_file_layout_is_the_one_the_scoring_script_writes():
    """The job reads `y_true` and, when there is one, `label_names`: the keys score_chexbert_standalone.py writes beside `y_pred`, the indices of the
    five-label subset and the two file names. The writer's own source is the layout's definition, so the keys are looked up in it."""
    writer = (REPO_ROOT / "scripts" / "score_chexbert_standalone.py").read_text(encoding="utf-8")
    for key in ("y_true", "y_pred", "label_names", "five_label_indices", "hyp_file", "ref_file"):
        assert '"{}"'.format(key) in writer, key
    assert cases.LABELS_KEY == "y_true" and cases.NAMES_KEY == "label_names"
    assert '"y_true": [[int(v) for v in labeler.get_label(r)] for r in refs]' in writer, "one row of 14 ints per reference report, in refs.txt order"


def test_the_fourteen_names_and_the_three_kinds_are_the_apps_and_the_plans():
    assert cases.CHEXBERT_14 is CHEXBERT_14 and len(CHEXBERT_14) == 14
    assert (cases.SEED, cases.PER_CATEGORY) == (42, 5)
    assert cases.CATEGORIES == ("normal", "support_devices", "pneumothorax")
    assert (CHEXBERT_14[13], CHEXBERT_14[12], CHEXBERT_14[8]) == ("No Finding", "Support Devices", "Pneumothorax")


def test_a_labels_file_with_label_names_in_the_standard_order_is_read_as_it_is(tmp_path):
    rows = y_true_of(12)
    assert cases.read_labels(write_labels(tmp_path / "a.json", rows)) == rows


def test_label_names_in_another_order_put_the_columns_in_the_standard_order_by_name(tmp_path):
    """The P5-C reader does this too: a dump whose label_names are not CHEXBERT_14's is mapped by name, so a row's findings keep their meaning."""
    rows = y_true_of(12)
    names = list(reversed(CHEXBERT_14))
    reordered = [list(reversed(row)) for row in rows]
    assert cases.read_labels(write_labels(tmp_path / "a.json", reordered, label_names=names)) == rows


def test_a_labels_file_without_label_names_is_read_in_the_standard_order(tmp_path):
    """The older of the two formats the scoring script has written has no label_names key: CHEXBERT_14's order is assumed, as P5-C's reader assumes it."""
    rows = y_true_of(12)
    path = write_labels(tmp_path / "a.json", rows)
    payload = json.loads(path.read_text())
    del payload["label_names"]
    path.write_text(json.dumps(payload))
    assert cases.read_labels(path) == rows


def test_the_other_keys_are_never_needed_a_file_with_only_y_true_will_do(tmp_path):
    path = tmp_path / "only.json"
    path.write_text(json.dumps({"y_true": y_true_of(12)}))
    assert cases.read_labels(path) == y_true_of(12)


LABEL_REFUSALS = [   # (what is wrong with the file, its text or None for a file that is not there, the ERROR line it is refused with)
    ("a missing file", None, "ERROR labels_unreadable"),
    ("text that is no JSON", SECRET + " {", "ERROR labels_unreadable"),
    ("JSON that is no object", json.dumps([1, 2, 3]), "ERROR labels_unreadable"),
    ("no y_true", json.dumps({"y_pred": [[0] * 14]}), "ERROR labels_unreadable"),
    ("a y_true that is a string", json.dumps({"y_true": SECRET}), "ERROR labels_shape rows=0 width=0"),
    ("an empty y_true", json.dumps({"y_true": []}), "ERROR labels_shape rows=0 width=0"),
    ("rows of 13", json.dumps({"y_true": [[0] * 13, [0] * 13]}), "ERROR labels_shape rows=2 width=13"),
    ("a row of 15 among rows of 14", json.dumps({"y_true": [[0] * 14, [0] * 15]}), "ERROR labels_shape rows=2 width=14"),
    ("a row that is no list", json.dumps({"y_true": [[0] * 14, SECRET]}), "ERROR labels_shape rows=2 width=14"),
    ("a label of 2", json.dumps({"y_true": [[0] * 13 + [2]]}), "ERROR labels_shape rows=1 width=14"),
    ("a label of -1", json.dumps({"y_true": [[-1] + [0] * 13]}), "ERROR labels_shape rows=1 width=14"),
    ("a label that is a boolean", json.dumps({"y_true": [[True] + [0] * 13]}), "ERROR labels_shape rows=1 width=14"),
    ("a label that is a float", json.dumps({"y_true": [[1.0] + [0] * 13]}), "ERROR labels_shape rows=1 width=14"),
    ("a label that is text", json.dumps({"y_true": [[SECRET] + [0] * 13]}), "ERROR labels_shape rows=1 width=14"),
    ("label_names that is no list", json.dumps({"y_true": [[0] * 14], "label_names": SECRET}), "ERROR label_names"),
    ("13 label_names", json.dumps({"y_true": [[0] * 14], "label_names": list(CHEXBERT_14[:13])}), "ERROR label_names"),
    ("a label name that is none of the 14", json.dumps({"y_true": [[0] * 14], "label_names": list(CHEXBERT_14[:13]) + [SECRET]}), "ERROR label_names"),
    ("a label name twice", json.dumps({"y_true": [[0] * 14], "label_names": list(CHEXBERT_14[:13]) + [CHEXBERT_14[0]]}), "ERROR label_names"),
    ("label_names that are not text", json.dumps({"y_true": [[0] * 14], "label_names": list(range(14))}), "ERROR label_names"),
]


@pytest.mark.parametrize("what, text, line", LABEL_REFUSALS, ids=[c[0] for c in LABEL_REFUSALS])
def test_a_labels_file_that_will_not_do_is_refused_with_a_fixed_code_and_numbers_and_never_a_word_of_it(what, text, line, world):
    if text is None:
        world["labels"].unlink()
    else:
        world["labels"].write_text(text)
    code, out = run_cases(args_of(world))
    assert code == 1 and out == line + "\n", (what, out)
    assert SECRET not in out and str(world["tmp"]) not in out


# ---- the gallery's test-split count -----------------------------------------------------------------------------------------------

def test_the_gallerys_test_count_is_read_from_its_manifest_and_the_tiny_gallery_has_40(tiny_gallery):
    assert cases.read_test_count(tiny_gallery) == 40 == json.loads((tiny_gallery / "manifest.json").read_text())["counts"]["test"]


GALLERY_REFUSALS = [
    ("no manifest", lambda d: None),
    ("a manifest that is no JSON", lambda d: (d / "manifest.json").write_text(SECRET)),
    ("a manifest that is no object", lambda d: (d / "manifest.json").write_text("[1]")),
    ("no counts", lambda d: (d / "manifest.json").write_text(json.dumps({"build_id": "x"}))),
    ("counts that are no object", lambda d: (d / "manifest.json").write_text(json.dumps({"counts": SECRET}))),
    ("no test count", lambda d: (d / "manifest.json").write_text(json.dumps({"counts": {"images": 200, "report_rows": 240}}))),
    ("a test count of 0", lambda d: write_manifest(d, test=0, images=200, report_rows=200)),
    ("a test count that is text", lambda d: (d / "manifest.json").write_text(json.dumps({"counts": {"images": 200, "report_rows": 240, "test": "40"}}))),
    ("a test count that is a boolean", lambda d: (d / "manifest.json").write_text(json.dumps({"counts": {"images": 1, "report_rows": 2, "test": True}}))),
    ("report rows that are not the train and test rows", lambda d: write_manifest(d, test=40, images=200, report_rows=241)),
]


@pytest.mark.parametrize("what, make", GALLERY_REFUSALS, ids=[c[0] for c in GALLERY_REFUSALS])
def test_a_gallery_whose_manifest_will_not_do_is_refused_with_a_fixed_code(what, make, tmp_path):
    gallery = tmp_path / "gallery"
    gallery.mkdir()
    make(gallery)
    labels = write_labels(tmp_path / "chexbert_labels.json", y_true_of(40))
    code, out = run_cases(["--labels", str(labels), "--gallery", str(gallery)])
    assert code == 1 and out == "ERROR gallery_unreadable\n", (what, out)
    assert SECRET not in out and str(tmp_path) not in out


def test_a_label_count_that_is_not_the_test_count_is_refused_before_any_row_is_drawn(world):
    """A row number must name the same study the picker shows: with 39 reference rows and a gallery of 40 test studies it would not."""
    write_labels(world["labels"], y_true_of(world["n"] - 1))
    code, out = run_cases(args_of(world))
    assert (code, out) == (1, "ERROR count_mismatch labels=39 gallery=40\n")
    write_labels(world["labels"], y_true_of(world["n"] + 3))
    assert run_cases(args_of(world)) == (1, "ERROR count_mismatch labels=43 gallery=40\n")


# ---- the three kinds of study ------------------------------------------------------------------------------------------------------

def test_the_kinds_are_the_plans_normal_is_no_finding_and_nothing_else_the_other_two_are_one_positive_label(world):
    got = cases.categories(y_true_of(world["n"]))
    assert got == expected_rows(world["n"])
    assert set(got) == {"normal", "support_devices", "pneumothorax"} and list(got) == list(cases.CATEGORIES)
    assert got["normal"] == [0, 7, 14, 21, 28, 35], "the rows that are `No Finding` alone: not the `No Finding` + Edema rows, nor the all-zero ones"
    assert set(got["support_devices"]) & set(got["pneumothorax"]) == {3, 10, 17, 24, 31, 38}, "a pneumothorax with a support device is in both lists"


def test_one_other_positive_label_is_enough_to_make_a_study_not_normal():
    for name in CHEXBERT_14[:-1]:
        assert cases.categories([row_of("No Finding", name)])["normal"] == [], name
    assert cases.categories([row_of("No Finding")])["normal"] == [0]
    assert cases.categories([row_of()])["normal"] == [], "no label at all is not `No Finding`"


# ---- the draw ----------------------------------------------------------------------------------------------------------------------

def test_the_draw_is_a_function_of_the_rows_and_the_seed_and_is_listed_in_the_order_drawn():
    pool = list(range(100, 300))
    first = cases.draw(pool, 42, 5)
    assert first == cases.draw(pool, 42, 5) and len(set(first)) == 5 and set(first) <= set(pool)
    assert cases.draw(pool, 43, 5) != first
    assert first != sorted(first) and first != pool[:5], "neither the lowest rows nor an ascending list: the order is the draw's own"
    assert cases.draw(pool, 42, 3) == first[:3], "the first k of a draw are the draw of k: the controller takes the first 2, 2 and 1"


def test_every_qualifying_row_can_be_drawn_and_each_can_come_first():
    """A fair draw: no row is out of reach, and none is favoured for the first place. Over many seeds each of ten rows must lead a draw (a row
    that never could would be kept out by something other than chance: each leads one draw in ten, so the chance of a miss in 600 is below
    1e-27)."""
    pool = list(range(10, 20))
    leaders = [cases.draw(pool, seed, 5)[0] for seed in range(2000)]
    seen = {row for seed in range(60) for row in cases.draw(pool, seed, 5)}
    assert set(leaders) == set(pool) and seen == set(pool)
    counts = [leaders.count(row) for row in pool]
    assert all(120 <= c <= 280 for c in counts), counts   # 200 each in expectation; a standard deviation of about 13


def test_a_pool_smaller_than_five_is_given_whole_and_an_empty_one_gives_nothing():
    assert sorted(cases.draw([7, 9, 12], 42, 5)) == [7, 9, 12] and len(cases.draw([7, 9, 12], 42, 5)) == 3
    assert cases.draw([], 42, 5) == []


def test_the_draw_uses_only_the_generators_random_numbers_and_the_rows_themselves():
    """random.Random(seed).random() is the one part of the module that Python promises to keep the same across versions (the laptop runs 3.14 and
    the cluster 3.11), so the draw is made of those numbers alone: the k lowest of one number per row."""
    import random
    rng = random.Random(42)
    keys = [(rng.random(), row) for row in range(30, 50)]
    assert cases.draw(list(range(30, 50)), 42, 5) == [row for _, row in sorted(keys)[:5]]


def test_the_candidates_do_not_depend_on_the_models_prediction_or_on_anything_but_y_true(world):
    """The fair draw of the plan: the model's own labels (y_pred) must not decide which studies are shown."""
    base = the_result(run_cases(args_of(world), 0)[1])
    for y_pred in ([[0] * 14 for _ in range(world["n"])], [[1] * 14 for _ in range(world["n"])], "SECRET-not-even-labels", None):
        write_labels(world["labels"], y_true_of(world["n"]), y_pred=y_pred)
        assert the_result(run_cases(args_of(world), 0)[1]) == base
    payload = json.loads(world["labels"].read_text())
    del payload["y_pred"]
    world["labels"].write_text(json.dumps(payload))
    assert the_result(run_cases(args_of(world), 0)[1]) == base, "a file with no y_pred at all gives the same answer"
    strings = {n.value for n in ast.walk(ast.parse(SCRIPT.read_text(encoding="utf-8"))) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert "y_pred" not in strings, "the script never asks for it"


# ---- the RESULT line ---------------------------------------------------------------------------------------------------------------

def test_the_result_line_has_the_plans_keys_in_the_plans_order_and_numbers_only(world):
    code, out = run_cases(args_of(world), 0)
    assert out.count("\n") == 1 and out.startswith("RESULT {") and out.endswith("}\n")
    result = the_result(out)
    assert list(result) == ["demo_cases", "n", "counts", "candidates", "seed"]
    assert result["demo_cases"] == "done" and result["n"] == world["n"] == 40 and result["seed"] == 42
    expected = expected_rows(world["n"])
    assert list(result["counts"]) == list(result["candidates"]) == ["normal", "support_devices", "pneumothorax"]
    assert result["counts"] == {kind: len(rows) for kind, rows in expected.items()} == {"normal": 6, "support_devices": 12, "pneumothorax": 12}
    for kind, rows in expected.items():
        picked = result["candidates"][kind]
        assert len(picked) == min(5, len(rows)) and len(set(picked)) == len(picked) and set(picked) <= set(rows), kind
    assert all(type(row) is int for rows in result["candidates"].values() for row in rows)
    assert out == "RESULT " + json.dumps(result, separators=(",", ":")) + "\n", "compact JSON, one line"


def test_the_draw_for_the_tiny_world_is_pinned_so_that_the_demo_rows_do_not_change_by_accident(world):
    """A regression pin. The rows were worked out apart from the script, from the definition alone (random.Random(42), one random() per
    qualifying row in row order, the five lowest numbers: for the six normal rows 0.639, 0.025, 0.275, 0.223, 0.736, 0.677, which put rows 7, 21,
    14, 0 and 35 first) and they are what the script printed. The rows the controller is given for a labels file are a function of that file
    alone, and a change here is a change of which studies are shown."""
    out = run_cases(args_of(world), 0)[1]
    assert out == ('RESULT {"demo_cases":"done","n":40,"counts":{"normal":6,"support_devices":12,"pneumothorax":12},'
                   '"candidates":{"normal":[7,21,14,0,35],"support_devices":[3,31,24,36,10],"pneumothorax":[3,31,24,37,10]},"seed":42}\n'), out


def test_a_category_with_fewer_than_five_studies_lists_them_all_and_one_with_none_lists_none(tmp_path):
    rows = [list(KINDS["normal"]), list(KINDS["normal"]), list(KINDS["support"]), list(KINDS["other"]), list(KINDS["other"])]
    labels = write_labels(tmp_path / "chexbert_labels.json", rows)
    gallery = write_manifest(tmp_path / "gallery", test=5, images=10)
    code, out = run_cases(["--labels", str(labels), "--gallery", str(gallery)])
    result = the_result(out)
    assert code == 0 and result["n"] == 5
    assert result["counts"] == {"normal": 2, "support_devices": 1, "pneumothorax": 0}
    assert sorted(result["candidates"]["normal"]) == [0, 1] and result["candidates"]["support_devices"] == [2]
    assert result["candidates"]["pneumothorax"] == []


def test_the_same_inputs_give_the_same_line_every_time(world):
    assert run_cases(args_of(world)) == run_cases(args_of(world))


def test_the_longest_line_the_job_can_print_stays_under_the_300_characters_chat_remote_summary_keeps(tmp_path):
    n = 2663   # the official test split
    rows = [list(KINDS["both"]) for _ in range(n)]
    labels = write_labels(tmp_path / "chexbert_labels.json", rows)
    gallery = write_manifest(tmp_path / "gallery", test=n, images=191462)
    out = run_cases(["--labels", str(labels), "--gallery", str(gallery)], 0)[1]
    assert len(out.rstrip("\n")) < 300, len(out)
    result = the_result(out)
    assert result["n"] == 2663 and result["counts"]["support_devices"] == result["counts"]["pneumothorax"] == 2663
    assert all(0 <= row < n for rows in result["candidates"].values() for row in rows)


# ---- R7: what is printed ------------------------------------------------------------------------------------------------------------

def test_nothing_of_the_labels_file_but_the_numbers_it_yields_is_ever_printed(world):
    """Every key the file has, and every value that could carry a word, holds a SECRET; the run prints the RESULT line and nothing else."""
    write_labels(world["labels"], y_true_of(world["n"]), hyp_file="/SECRET/hyps.txt", ref_file="/SECRET/refs.txt",
                 study_ids=[SECRET + "-{}".format(i) for i in range(world["n"])], SECRET_KEY=SECRET, reports=[SECRET] * world["n"])
    code, out = run_cases(args_of(world))
    assert code == 0 and SECRET not in out and "SECRET" not in out and str(world["tmp"]) not in out and "hyps.txt" not in out
    assert re.fullmatch(r'RESULT \{[0-9A-Za-z_":,\[\]{}]*\}\n', out), out


def test_an_unexpected_failure_is_the_name_of_its_class_and_nothing_of_its_text(world, monkeypatch):
    def broken(*args):
        raise RuntimeError(SECRET)
    monkeypatch.setattr(cases, "categories", broken)
    code, out = run_cases(args_of(world))
    assert (code, out) == (1, "ERROR failed RuntimeError\n") and SECRET not in out


@pytest.mark.parametrize("argv", [[], ["--labels", "x"], ["--gallery", "y"], ["--labels", "x", "--gallery", "y", "--seed", "7"],
                                  ["--labels", "x", "--gallery", "y", SECRET], ["--lab", "x", "--gall", "y"], ["--" + SECRET]])
def test_a_bad_command_line_is_one_fixed_line_that_quotes_nothing_it_was_given(argv):
    code, out = run_cases(argv)
    assert (code, out) == (64, "ERROR usage\n"), (argv, out)


def test_help_is_printed_on_the_stream_and_the_exit_status_is_0():
    code, out = run_cases(["--help"])
    assert code == 0 and out.startswith("usage: ")
    for word in ("--labels", "--gallery", "RESULT"):
        assert word in out, word


def test_the_script_runs_as_a_script_from_any_directory_the_way_the_job_starts_it(world, tmp_path):
    """A process of its own, with no PYTHONPATH and a working directory that is not the repository: it puts the repository on its own path (it
    imports the app's label list), prints one RESULT line on stdout and nothing on stderr."""
    cwd = tmp_path / "elsewhere"
    cwd.mkdir()
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    done = subprocess.run([sys.executable, str(SCRIPT)] + args_of(world), cwd=str(cwd), env=env, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0 and done.stderr == "", (done.stdout, done.stderr)
    assert done.stdout == run_cases(args_of(world))[1]
    refused = subprocess.run([sys.executable, str(SCRIPT), "--labels", str(world["tmp"] / "nope.json"), "--gallery", str(world["gallery"])],
                             cwd=str(cwd), env=env, capture_output=True, text=True, timeout=120)
    assert refused.returncode == 1 and refused.stdout == "ERROR labels_unreadable\n" and refused.stderr == ""
    assert list(cwd.iterdir()) == [], "it writes no file"


# ---- the CPU job: the wrapper, rehearsed in a temp tree that stands in for CLUSTER_REPO ------------------------------------------------

LINE_OK = safe_line("cases")
JOB_ID = "1234567"
SYNC_LINE = "=== sync 3f2a9c41d7e86b05a1c4e9d3b7f60285ac9e1d47 clean ==="
BANNER = "=== chat demo cases: job={} node=gx14.test.cluster gallery=g13d_m3_v1 ===".format(JOB_ID)
NOTICE = "=== demo cases: the script's raw output stays in files under CHAT_HOME; its RESULT line of a known shape is shown here ==="
DUMP = "results/report_gen_m3_test_split_s42"
HOSTNAME_STUB = '#!/bin/bash\necho "${STUB_HOSTNAME:-gx14.test.cluster}"\n'

# The script the wrapper runs, as a stub: it records its arguments and what it was started with, prints the lines it is given (real ones, from
# the real script over the temp tree's files), a stray line on stdout that is no RESULT line, and raw text on stderr, which must never reach the
# job log; it exits as it is told.
SCRIPT_STUB = '''
import json, os, sys
stub = os.environ["STUB_DIR"]


def read(name, default=""):
    try:
        return open(os.path.join(stub, name)).read()
    except OSError:
        return default


with open(os.path.join(stub, "script.call"), "w") as f:
    json.dump({"argv": sys.argv[1:], "cwd": os.path.realpath(os.getcwd()), "umask": os.umask(0o77)}, f)
sys.stderr.write("Findings: SECRET REPORT TEXT study_id=12345678 /sc/home/someone/img.jpg\\n")
print("Findings: a stray stdout line with SECRET REPORT TEXT")
for line in read("lines.txt").splitlines():
    print(line)
sys.stdout.flush()
sys.exit(int(read("exit.txt", "0") or 0))
'''


class CasesBox(wr.JobBox):
    """The wrapper with the real script, over synthetic files: a labels file of 40 rows in results/, and a gallery directory with the manifest."""
    WRAPPER = WRAPPER
    SCRIPTS = ("chat_demo_cases.py",)
    SLURM_CPUS = "1"

    def populate(self):
        (self.repo / "app").mkdir()
        for name in ("__init__.py", "labels.py"):   # the script imports the app's list of the 14 names
            shutil.copy(str(REPO_ROOT / "app" / name), str(self.repo / "app" / name))
        (self.repo / DUMP).mkdir(parents=True)
        write_labels(self.repo / DUMP / "chexbert_labels.json", y_true_of(40))
        write_manifest(self.chat / "gallery" / "g13d_m3_v1", test=40, images=200)
        (self.bin / "hostname").write_text(HOSTNAME_STUB)
        (self.bin / "hostname").chmod(0o755)

    def labels(self):
        return self.repo / DUMP / "chexbert_labels.json"

    def gallery(self):
        return self.chat / "gallery" / "g13d_m3_v1"

    def result(self):
        """The line the script itself prints for this box's files."""
        out = io.StringIO()
        assert cases.main(["--labels", str(self.labels()), "--gallery", str(self.gallery())], out) == 0
        return out.getvalue().rstrip("\n")

    def raw(self, suffix):
        return self.chat / "logs" / "demo_cases_{}.{}".format(JOB_ID, suffix)

    def stub_the_script(self, lines=(), exit_code=0):
        (self.repo / "scripts" / "chat_demo_cases.py").write_text(SCRIPT_STUB)
        (self.stubs / "lines.txt").write_text("\n".join(lines) + "\n")
        (self.stubs / "exit.txt").write_text(str(exit_code))

    def call(self):
        return json.loads((self.stubs / "script.call").read_text())


@pytest.fixture
def box(tmp_path):
    return CasesBox(tmp_path)


def test_the_wrapper_prints_the_sync_line_a_banner_the_result_line_and_nothing_else(box):
    done = box.run()
    lines = job_lines(done)
    assert done.returncode == 0, done.stdout
    assert lines == [SYNC_LINE, BANNER, NOTICE, box.result(), "=== demo case lines withheld: 0 ===", "=== END chat demo cases ==="]
    assert lines[3].startswith("RESULT ") and the_result(lines[3] + "\n")["demo_cases"] == "done"
    assert all(LINE_OK.match(l) for l in lines), [l for l in lines if not LINE_OK.match(l)]


def test_the_wrapper_runs_the_script_in_the_repository_with_the_published_dumps_labels_and_the_gallery(box):
    assert box.run().returncode == 0
    calls = box.calls()
    assert calls == [["scripts/chat_demo_cases.py", "--labels", DUMP + "/chexbert_labels.json", "--gallery", str(box.gallery())]], calls


def test_the_default_dump_is_the_one_the_serving_job_defaults_to():
    """Where the serve wrapper looks for the published model's dump, this one looks for its labels: the same directory, whose chexbert_labels.json
    the P5-C cross-check read."""
    serve = (REPO_ROOT / "scripts" / "serve_chat_h100.sh").read_text()
    mine = (REPO_ROOT / "scripts" / WRAPPER).read_text()
    line = 'PUBLISHED_MODEL="${PUBLISHED_MODEL:-results/report_gen_m3_test_split_s42}"'
    assert line in serve and line in mine


def test_the_published_model_and_gallery_levers_name_another_dump_and_another_build(box):
    other = box.repo / "results" / "other_dump"
    other.mkdir(parents=True)
    write_labels(other / "chexbert_labels.json", y_true_of(30))
    build = write_manifest(box.chat / "gallery" / "other_build_v2", test=30, images=100)
    done = box.run(PUBLISHED_MODEL="results/other_dump", GALLERY=str(build))
    lines = job_lines(done)
    assert done.returncode == 0 and "gallery=other_build_v2 ===" in done.stdout and str(box.chat) not in done.stdout and "other_dump" not in done.stdout
    assert the_result([l for l in lines if l.startswith("RESULT ")][0] + "\n")["n"] == 30
    assert box.calls()[-1][1:] == ["--labels", "results/other_dump/chexbert_labels.json", "--gallery", str(build)]


def test_a_gallery_that_does_not_have_the_labels_row_count_is_the_scripts_refusal_shown_and_the_jobs_failure(box):
    write_manifest(box.gallery(), test=39, images=200)
    done = box.run()
    lines = job_lines(done)
    assert done.returncode == 1
    assert "ERROR count_mismatch labels=40 gallery=39" in lines
    assert lines[-1] == "ERROR the demo cases script exited with code 1 (see the logs under CHAT_HOME)" and "END chat demo cases" not in done.stdout
    assert not [l for l in lines if l.startswith("RESULT ")] and all(LINE_OK.match(l) for l in lines)


def test_the_raw_output_of_the_script_goes_to_files_under_chat_home_and_nowhere_else(box):
    box.stub_the_script([])
    done = box.run()
    assert done.returncode != 0
    assert "SECRET REPORT TEXT" in box.raw("err").read_text() and "a stray stdout line" in box.raw("out").read_text()
    assert list(box.repo.rglob("demo_cases_*")) == [], "nothing of it in the code tree"
    assert "SECRET" not in done.stdout and "Findings" not in done.stdout and "study_id" not in done.stdout and "/sc/home" not in done.stdout


def test_the_files_the_job_makes_are_owner_only(box):
    assert box.run().returncode == 0
    for suffix in ("out", "err"):
        assert (box.raw(suffix).stat().st_mode & 0o077) == 0, suffix


def test_the_whole_job_log_comes_through_chat_remote_summary_unchanged(box):
    """The controller reads this job through `chat_remote.sh summary`, which keeps the lines of its fixed shapes, cuts each at 300 characters and
    masks what looks like an id: every line of a normal log must come through whole."""
    from tests.test_chat_remote import _summary_of
    lines = job_lines(box.run())
    assert len(lines) == 6 and max(len(l) for l in lines) < 300
    assert _summary_of(box.root / "summary", lines) == lines


def test_a_stray_line_and_a_line_of_no_known_shape_are_counted_and_never_shown(box):
    real = box.result()
    leaks = ['RESULT {"demo_cases":"done","n":40,"report":"Findings: SECRET REPORT TEXT"}', 'RESULT {"fatal":"Findings: SECRET REPORT TEXT"}',
             "ERROR Findings: SECRET REPORT TEXT"]
    box.stub_the_script(leaks[:2] + [real] + leaks[2:])
    done = box.run()
    lines = job_lines(done)
    assert done.returncode == 0 and "SECRET" not in done.stdout
    assert [l for l in lines if l.startswith("RESULT ")] == [real]
    assert "=== demo case lines withheld: 4 ===" in lines, "the three of known keys with text, and the stray line"
    assert lines[-1] == "=== END chat demo cases ==="


def test_a_script_that_exits_0_with_no_result_line_is_not_a_clean_end(box):
    box.stub_the_script([])
    done = box.run()
    assert done.returncode != 0 and "ERROR the demo cases script printed 0 result lines, not 1" in done.stdout
    assert "END chat demo cases" not in done.stdout


def test_a_script_that_prints_two_result_lines_is_not_a_clean_end_either(box):
    box.stub_the_script([box.result(), box.result()])
    done = box.run()
    assert done.returncode != 0 and "ERROR the demo cases script printed 2 result lines, not 1" in done.stdout


def test_a_script_that_exits_1_after_its_line_has_it_shown_and_the_job_ends_with_one_error_line_and_the_same_status(box):
    box.stub_the_script([box.result()], exit_code=1)
    done = box.run()
    lines = job_lines(done)
    assert done.returncode == 1 and len([l for l in lines if l.startswith("RESULT ")]) == 1
    assert lines[-1] == "ERROR the demo cases script exited with code 1 (see the logs under CHAT_HOME)" and "END chat demo cases" not in done.stdout


def test_the_script_runs_with_files_that_it_makes_owner_only_and_in_the_repository(box):
    box.stub_the_script([box.result()])
    assert box.run().returncode == 0
    call = box.call()
    assert call["cwd"] == os.path.realpath(str(box.repo)) and call["umask"] == 0o77


LEVER_CASES = [   # (what is wrong, the environment, the fixed sentence the job says)
    ("GALLERY that is no build name", {"GALLERY": "{chat}/gallery/bad name SECRET"}, "GALLERY must end in one plain build name"),
    ("GALLERY with no manifest", {"GALLERY": "{chat}/gallery/empty_build"}, "the gallery has no manifest.json"),
    ("PUBLISHED_MODEL with a space", {"PUBLISHED_MODEL": "results/a b SECRET"}, "PUBLISHED_MODEL must be a plain relative path"),
    ("PUBLISHED_MODEL that is a command", {"PUBLISHED_MODEL": "$(echo SECRET)"}, "PUBLISHED_MODEL must be a plain relative path"),
    ("PUBLISHED_MODEL that goes up", {"PUBLISHED_MODEL": "../results/x"}, "PUBLISHED_MODEL must be a plain relative path"),
    ("PUBLISHED_MODEL that is absolute", {"PUBLISHED_MODEL": "/sc/home/x"}, "PUBLISHED_MODEL must be a plain relative path"),
    ("PUBLISHED_MODEL with no labels file", {"PUBLISHED_MODEL": "results/no_labels"}, "the published dump has no chexbert_labels.json"),
    ("CHAT_HOME that is missing", {"CHAT_HOME": "{root}/no/such/home"}, "CHAT_HOME does not exist"),
    ("a venv that is missing", {"VENV_ACTIVATE": "{root}/no/such/activate"}, "the venv's activate script is missing"),
]


@pytest.mark.parametrize("what, env, sentence", LEVER_CASES, ids=[c[0] for c in LEVER_CASES])
def test_a_lever_that_will_not_do_is_refused_in_one_line_that_does_not_echo_it_before_anything_runs(what, env, sentence, box):
    (box.chat / "gallery" / "empty_build").mkdir(exist_ok=True)
    (box.repo / "results" / "no_labels").mkdir(exist_ok=True)
    env = {k: v.format(chat=box.chat, root=box.root) for k, v in env.items()}
    done = box.run(**env)
    lines = job_lines(done)
    errors = [l for l in lines if l.startswith("ERROR ")]
    assert done.returncode != 0 and len(errors) == 1 and sentence in errors[0], (what, done.stdout)
    for value in env.values():
        assert value not in done.stdout and "SECRET" not in done.stdout, what
    assert str(box.chat) not in done.stdout and box.calls() == [], "nothing ran"
    assert all(LINE_OK.match(l) for l in lines), lines


def test_a_missing_script_is_said_as_the_other_wrappers_say_it(box):
    (box.repo / "scripts" / "chat_demo_cases.py").unlink()
    done = box.run()
    assert done.returncode != 0 and "ERROR chat_demo_cases.py is missing from this tree: run chat_remote.sh sync first" in done.stdout
    assert box.calls() == []


def test_the_submit_line_the_header_documents_is_one_chat_remote_sh_takes(tmp_path):
    from tests.test_chat_remote import Sandbox
    sandbox = Sandbox(tmp_path)
    shutil.copy(str(REPO_ROOT / "scripts" / WRAPPER), str(sandbox.repo / "scripts" / WRAPPER))
    done = sandbox.run("submit", "scripts/" + WRAPPER)
    assert done.returncode == 0, done.stderr
    assert " ".join(sandbox.calls()[-1][2].split()) == "cd '/fake/hybrid_chat_ui' && env sbatch --parsable 'scripts/" + WRAPPER + "'"
    header = (REPO_ROOT / "scripts" / WRAPPER).read_text()
    assert "bash scripts/chat_remote.sh sync && bash scripts/chat_remote.sh submit scripts/chat_demo_cases_h100.sh" in header
    assert "bash scripts/chat_remote.sh summary logs/chat_demo_cases_<jobid>.log" in header


# One line of each kind the script prints, as the wrapper's allowlist must take it, and the ways a line of that kind could be wrong or carry text:
# each deviation changes one thing in a valid line, so that it is refused for that and for nothing else.
RESULT_LINE = ('RESULT {"demo_cases":"done","n":2663,"counts":{"normal":1200,"support_devices":900,"pneumothorax":120},'
               '"candidates":{"normal":[7,21,14,0,35],"support_devices":[3,31,24,36,10],"pneumothorax":[3,2662,24,37,10]},"seed":42}')
MISMATCH_LINE = "ERROR count_mismatch labels=2663 gallery=2662"
SHAPE_LINE = "ERROR labels_shape rows=2663 width=13"
FAILED_LINE = "ERROR failed RuntimeError"
DEVIATIONS = [
    (RESULT_LINE, '"demo_cases":"done"', '"demo_cases":"refused"'), (RESULT_LINE, '"n":2663', '"n":"2663"'),
    (RESULT_LINE, '"n":2663', '"n":12345678'), (RESULT_LINE, '"n":2663', '"n":-1'), (RESULT_LINE, '"n":2663', '"n":26.63'),
    (RESULT_LINE, '"normal":1200', '"normal":"many"'), (RESULT_LINE, '"normal":1200', '"normal":true'),
    (RESULT_LINE, '"counts":{"normal":1200,"support_devices":900,"pneumothorax":120}',
     '"counts":{"normal":1200,"pneumothorax":120,"support_devices":900}'),
    (RESULT_LINE, '[7,21,14,0,35]', '[7,21,14,0,35,1]'), (RESULT_LINE, '[7,21,14,0,35]', '["SECRET"]'), (RESULT_LINE, '[7,21,14,0,35]', '[7, 21]'),
    (RESULT_LINE, '[7,21,14,0,35]', '[7,21,14,0,35,]'), (RESULT_LINE, '[3,31,24,36,10]', '[3,31,24,36,x]'),
    (RESULT_LINE, '"seed":42}', '"seed":"42"}'), (RESULT_LINE, '"seed":42}', '"seed":42,"report":"Findings: x"}'),
    (RESULT_LINE, '"pneumothorax":[3,2662,24,37,10]}', '"pneumothorax":[3,2662,24,37,10],"study_ids":["SECRET"]}'),
    (RESULT_LINE, '"seed":42', '"seed":42,"x":1'),
    (MISMATCH_LINE, "labels=2663", "labels=x"), (MISMATCH_LINE, "gallery=2662", "gallery=12345678"), (MISMATCH_LINE, "count_mismatch", "elsewhere"),
    (MISMATCH_LINE, "gallery=2662", "gallery=2662 path=/sc/home/x"), (SHAPE_LINE, "rows=2663", "rows=/sc/home"),
    (FAILED_LINE, "RuntimeError", "Runtime Error"), (FAILED_LINE, "RuntimeError", "Findings: x"), (FAILED_LINE, "RuntimeError", "R" * 61),
]


def shapes_of_the_wrapper():
    """The wrapper's own CASES_SHAPES: the text between its single quotes."""
    text = (REPO_ROOT / "scripts" / WRAPPER).read_text()
    found = re.search(r"^CASES_SHAPES='(.*?)'$", text, re.S | re.M)
    assert found, "CASES_SHAPES is not in the wrapper"
    return found.group(1)


def matches(shapes, line):
    return subprocess.run(["grep", "-aE", shapes], input=line + "\n", capture_output=True, text=True).returncode == 0


def test_the_wrappers_allowlist_takes_every_line_the_script_prints_in_every_state_it_can_end_in(tmp_path, world):
    shapes = shapes_of_the_wrapper()
    assert all(line for line in shapes.splitlines()), "an empty pattern matches every line"
    assert len(shapes.splitlines()) == 3 and "'" not in shapes
    for line in (RESULT_LINE, MISMATCH_LINE, SHAPE_LINE, FAILED_LINE, run_cases(args_of(world))[1].rstrip("\n")):
        assert matches(shapes, line), line
    empty = {name: [] for name in cases.CATEGORIES}
    one = {"normal": [0], "support_devices": [1], "pneumothorax": [2]}
    for found in (empty, one):
        assert matches(shapes, cases.result_line(cases.payload(7, found))), found
    n = 2663   # the longest line: every row of every kind, five candidates each
    longest = write_labels(tmp_path / "all.json", [list(KINDS["both"]) for _ in range(n)])
    gallery = write_manifest(tmp_path / "g", test=n, images=191462)
    line = run_cases(["--labels", str(longest), "--gallery", str(gallery)], 0)[1].rstrip("\n")
    assert matches(shapes, line) and len(line) < 300, line
    for code in cases.ERROR_CODES:   # every code the script has is a line the allowlist takes, with the numbers it carries
        assert matches(shapes, "ERROR " + code), code
    for name in ("labels", "gallery", "rows", "width"):
        assert matches(shapes, "ERROR labels_shape {}=1".format(name)), name


def test_the_wrappers_allowlist_names_exactly_the_error_codes_the_script_has_and_no_others():
    shapes = shapes_of_the_wrapper()
    group = re.search(r"\^ERROR \(([a-z_|]+)\)", shapes).group(1)
    assert set(group.split("|")) == set(cases.ERROR_CODES), group


@pytest.mark.parametrize("template, old, new", DEVIATIONS, ids=["deviation{}".format(i) for i in range(len(DEVIATIONS))])
def test_the_wrappers_allowlist_takes_a_line_with_one_thing_wrong_in_it_for_none_of_the_shapes(template, old, new):
    shapes = shapes_of_the_wrapper()
    assert matches(shapes, template), "the valid line is taken: the deviation below is refused for what it changes"
    assert old in template
    assert not matches(shapes, template.replace(old, new)), new


@pytest.mark.parametrize("leak", [
    'RESULT {"gate": {"ok": true}}', 'RESULT {}', 'RESULT ', 'Findings: SECRET', '=== not a result ===', 'ERROR x', 'ERROR ',
    RESULT_LINE + " trailing", " " + RESULT_LINE, "x" + RESULT_LINE, MISMATCH_LINE + " trailing", "  " + FAILED_LINE, FAILED_LINE + " Findings",
    RESULT_LINE.replace('"seed":42', ""), RESULT_LINE.replace('"n":2663,', ""), RESULT_LINE.replace('"demo_cases":"done",', ""),
])
def test_the_wrappers_allowlist_takes_nothing_else(leak):
    assert not matches(shapes_of_the_wrapper(), leak), leak
