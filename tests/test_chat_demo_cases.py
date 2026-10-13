"""CHAT_UI_PLAN.md P9-C, part 1: scripts/chat_demo_cases.py, which picks the test studies for the supervisor demo, and its CPU job
scripts/chat_demo_cases_h100.sh.

The job reads the published run's per-sample CheXbert labels (`chexbert_labels.json`, written by scripts/score_chexbert_standalone.py: the
keys `y_true`, `y_pred`, `label_names`, `five_label_indices`, `hyp_file` and `ref_file`), takes the reference labels `y_true` (one row of 14 per
test study, in test-split order), and lists up to five candidate test rows for each of three kinds of study: normal, support devices and
pneumothorax. The draw is seeded and fair: it looks at nothing but `y_true`, and each kind has a seed of its own (fix 1). It prints row numbers
and counts only (rule R7: no MIMIC text, image or study id ever reaches an agent), and it refuses when the number of studies is not the
gallery's test-split count, and when the gallery's labels_check.json (P5-C's record) does not say it was checked against this dump over this
many studies with no difference (fix 1), because a row number must name the study the picker shows.

Everything here is synthetic: a labels file built in the test from a table of kinds, the tiny gallery of P5-B with a labels_check.json written
the way P5-C writes it, and a temp tree standing in for the cluster. No MIMIC file is read, so an assertion may print whatever it compares.
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


GONE = object()   # in write_check: a field to leave out of the file
DUMP_NAME = "report_gen_m3_test_split_s42"


def write_check(directory, labels, **changes):
    """The gallery's labels_check.json as P5-C writes it (scripts/label_gallery_reports.py, cross_check) for the labels file `labels`: the final
    stage, the name of the dump's directory as the reference, as many test studies as the file has rows, no difference of either kind, and the other
    fields the real record has (which the job never reads). `changes` replace a field; GONE leaves it out."""
    rows = len(json.loads(Path(labels).read_text())["y_true"])
    record = {"stage": "final", "reference": Path(labels).parent.name, "n_test": rows, "refs_mismatch": 0, "labels_mismatch": 0,
              "labels_mismatch_own_text": 0, "labels_mismatch_shared_text": 0, "test_rows_sharing_a_text": 23, "label_names_source": "dump"}
    record.update(changes)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "labels_check.json").write_text(json.dumps({k: v for k, v in record.items() if v is not GONE}))
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
    """The tiny gallery (40 test studies) and a labels file of 40 rows, the kinds in turn, in a directory named as the published dump is; the
    gallery has the labels_check.json that P5-C leaves, which says it was checked against that dump."""
    n = json.loads((tiny_gallery / "manifest.json").read_text())["counts"]["test"]
    (tmp_path / DUMP_NAME).mkdir()
    labels = write_labels(tmp_path / DUMP_NAME / "chexbert_labels.json", y_true_of(n))
    write_check(tiny_gallery, labels)
    return {"labels": labels, "gallery": tiny_gallery, "n": n, "tmp": tmp_path}


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
    assert matches(shapes_of_the_wrapper(), line), "the wrapper's allowlist lets this line into the job log"


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
    assert matches(shapes_of_the_wrapper(), "ERROR gallery_unreadable")


def test_a_label_count_that_is_not_the_test_count_is_refused_before_any_row_is_drawn(world):
    """A row number must name the same study the picker shows: with 39 reference rows and a gallery of 40 test studies it would not."""
    write_labels(world["labels"], y_true_of(world["n"] - 1))
    code, out = run_cases(args_of(world))
    assert (code, out) == (1, "ERROR count_mismatch labels=39 gallery=40\n")
    write_labels(world["labels"], y_true_of(world["n"] + 3))
    assert run_cases(args_of(world)) == (1, "ERROR count_mismatch labels=43 gallery=40\n")
    assert matches(shapes_of_the_wrapper(), "ERROR count_mismatch labels=43 gallery=40")


# ---- the gallery is bound to the dump: P5-C's labels_check.json ---------------------------------------------------------------------

CHECK_REFUSALS = [   # (what is wrong with an otherwise matching check, the fields it changes, the ERROR line it is refused with)
    ("a stage that is preflight", {"stage": "preflight"}, "ERROR check_stage"),
    ("no stage", {"stage": GONE}, "ERROR check_stage"),
    ("a stage that is text", {"stage": SECRET}, "ERROR check_stage"),
    ("a reference that is another dump", {"reference": "report_gen_full_ext_4gpu_tower13d"}, "ERROR check_reference"),
    ("no reference", {"reference": GONE}, "ERROR check_reference"),
    ("a reference that is no text", {"reference": 7}, "ERROR check_reference"),
    ("an empty reference", {"reference": ""}, "ERROR check_reference"),
    ("a reference that is a SECRET", {"reference": SECRET}, "ERROR check_reference"),
    ("a reference with the dump's name and more", {"reference": DUMP_NAME + "_v2"}, "ERROR check_reference"),
    ("an n_test of 39", {"n_test": 39}, "ERROR check_n_test labels=40 n_test=39"),
    ("an n_test of 41", {"n_test": 41}, "ERROR check_n_test labels=40 n_test=41"),
    ("an n_test of 0", {"n_test": 0}, "ERROR check_n_test labels=40 n_test=0"),
    ("no n_test", {"n_test": GONE}, "ERROR check_n_test labels=40"),
    ("an n_test that is text", {"n_test": "40"}, "ERROR check_n_test labels=40"),
    ("an n_test that is a boolean", {"n_test": True}, "ERROR check_n_test labels=40"),
    ("an n_test that is a float", {"n_test": 40.0}, "ERROR check_n_test labels=40"),
    ("an n_test that is negative", {"n_test": -40}, "ERROR check_n_test labels=40"),
    ("an n_test of a billion", {"n_test": 10 ** 9}, "ERROR check_n_test labels=40"),
    ("an n_test of 400 digits", {"n_test": 10 ** 400}, "ERROR check_n_test labels=40"),
    ("refs that differ", {"refs_mismatch": 3}, "ERROR check_mismatch refs_mismatch=3 labels_mismatch=0"),
    ("labels that differ", {"labels_mismatch": 2}, "ERROR check_mismatch refs_mismatch=0 labels_mismatch=2"),
    ("both that differ", {"refs_mismatch": 1, "labels_mismatch": 5}, "ERROR check_mismatch refs_mismatch=1 labels_mismatch=5"),
    ("a labels_mismatch that is null, as a preflight record has it", {"labels_mismatch": None}, "ERROR check_mismatch refs_mismatch=0"),
    ("no mismatch fields", {"refs_mismatch": GONE, "labels_mismatch": GONE}, "ERROR check_mismatch"),
    ("a mismatch that is false, not 0", {"refs_mismatch": False}, "ERROR check_mismatch labels_mismatch=0"),
    ("a mismatch that is 0.0, not 0", {"labels_mismatch": 0.0}, "ERROR check_mismatch refs_mismatch=0"),
    ("a mismatch that is negative", {"labels_mismatch": -1}, "ERROR check_mismatch refs_mismatch=0"),
    ("a mismatch that is text", {"refs_mismatch": SECRET}, "ERROR check_mismatch labels_mismatch=0"),
    ("a mismatch of a billion", {"labels_mismatch": 10 ** 9}, "ERROR check_mismatch refs_mismatch=0"),
]


def test_a_gallery_whose_check_says_it_was_made_against_this_dump_over_these_studies_with_no_difference_is_taken(world):
    """What P5-C leaves in the gallery (labels_check.json, final stage): the dump it was checked against (its directory's name), the studies it was
    checked over and 0 differences of either kind. The four together say that the labels file and the gallery are one pair, so that a row number
    names the study the picker shows: the count alone does not say it."""
    record = json.loads((world["gallery"] / "labels_check.json").read_text())
    assert (record["stage"], record["reference"], record["n_test"], record["refs_mismatch"], record["labels_mismatch"]) == ("final", DUMP_NAME, 40, 0, 0)
    code, out = run_cases(args_of(world))
    assert code == 0 and the_result(out)["n"] == 40


def test_the_checks_other_fields_are_never_read_or_printed(world):
    write_check(world["gallery"], world["labels"], labels_mismatch_own_text=SECRET, labels_mismatch_shared_text=[SECRET], test_rows_sharing_a_text=SECRET,
                label_names_source=SECRET, extra_key={"nested": SECRET})
    code, out = run_cases(args_of(world))
    assert code == 0 and "SECRET" not in out and re.fullmatch(r'RESULT \{[0-9A-Za-z_":,\[\]{}]*\}\n', out)


@pytest.mark.parametrize("what, changes, line", CHECK_REFUSALS, ids=[c[0] for c in CHECK_REFUSALS])
def test_each_field_of_the_check_that_does_not_say_so_is_refused_with_its_own_code_and_numbers_only(what, changes, line, world):
    write_check(world["gallery"], world["labels"], **changes)
    code, out = run_cases(args_of(world))
    assert code == 1 and out == line + "\n", (what, out)
    assert "SECRET" not in out and str(world["tmp"]) not in out and DUMP_NAME not in out, "no value of the file is quoted"
    assert matches(shapes_of_the_wrapper(), line), "the wrapper's allowlist lets this line into the job log"


def test_a_gallery_with_no_check_file_is_refused_with_a_code_of_its_own(world):
    (world["gallery"] / "labels_check.json").unlink()
    assert run_cases(args_of(world)) == (1, "ERROR check_missing\n")
    assert matches(shapes_of_the_wrapper(), "ERROR check_missing")


@pytest.mark.parametrize("text", [SECRET + " {", "[1, 2]", '"final"', "null", "7", ""])
def test_a_check_file_that_is_no_json_object_is_unreadable(text, world):
    (world["gallery"] / "labels_check.json").write_text(text)
    code, out = run_cases(args_of(world))
    assert (code, out) == (1, "ERROR check_unreadable\n") and SECRET not in out


def test_a_check_that_is_a_directory_is_unreadable_and_not_missing(world):
    path = world["gallery"] / "labels_check.json"
    path.unlink()
    path.mkdir()
    assert run_cases(args_of(world)) == (1, "ERROR check_unreadable\n")


def test_the_first_field_that_fails_is_the_one_named_in_the_order_stage_reference_n_test_mismatch(world):
    changes = {"stage": "preflight", "reference": "x", "n_test": 3, "refs_mismatch": 9, "labels_mismatch": 9}
    for line in ("ERROR check_stage", "ERROR check_reference", "ERROR check_n_test labels=40 n_test=3",
                 "ERROR check_mismatch refs_mismatch=9 labels_mismatch=9"):
        write_check(world["gallery"], world["labels"], **changes)
        assert run_cases(args_of(world)) == (1, line + "\n")
        del changes[next(iter(changes))]   # the field that failed is put right, and the next one is the first to fail


def test_the_count_check_comes_before_the_check_file_is_asked(world):
    write_check(world["gallery"], world["labels"], stage="preflight")
    write_manifest(world["gallery"], test=39, images=200)
    assert run_cases(args_of(world)) == (1, "ERROR count_mismatch labels=40 gallery=39\n")


def test_the_reference_is_the_name_of_the_directory_the_labels_file_is_in_however_the_path_is_spelled(tmp_path, tiny_gallery, monkeypatch):
    dump = tmp_path / "somewhere" / "dump_x"
    dump.mkdir(parents=True)
    labels = write_labels(dump / "chexbert_labels.json", y_true_of(40))
    write_check(tiny_gallery, labels, reference="dump_x")
    monkeypatch.chdir(dump.parent)
    for spelled in (str(labels), "dump_x/chexbert_labels.json", "./dump_x/chexbert_labels.json", "dump_x/../dump_x/chexbert_labels.json",
                    str(dump) + "/./chexbert_labels.json"):
        assert run_cases(["--labels", spelled, "--gallery", str(tiny_gallery)])[0] == 0, spelled
    monkeypatch.chdir(dump)
    assert run_cases(["--labels", "chexbert_labels.json", "--gallery", str(tiny_gallery)])[0] == 0, "a bare file name: the directory it is in"
    for other in ("somewhere", "chexbert_labels.json", "dump_x/chexbert_labels.json", ""):
        write_check(tiny_gallery, labels, reference=other)
        assert run_cases(["--labels", str(labels), "--gallery", str(tiny_gallery)]) == (1, "ERROR check_reference\n"), other


def test_a_copy_of_the_labels_in_a_directory_of_another_name_is_not_the_dump_the_gallery_was_checked_against(world, tmp_path):
    copy_dir = tmp_path / "copy_of_it"
    copy_dir.mkdir()
    (copy_dir / "chexbert_labels.json").write_bytes(world["labels"].read_bytes())
    code, out = run_cases(["--labels", str(copy_dir / "chexbert_labels.json"), "--gallery", str(world["gallery"])])
    assert (code, out) == (1, "ERROR check_reference\n")


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
    """A regression pin. The rows were worked out apart from the script, from the definition alone: each kind has a generator of its own, seeded
    with 42 plus the kind's place in CATEGORIES (random.Random(42), (43) and (44)), one random() per qualifying row in row order, the five lowest
    numbers (for the six normal rows, seed 42: 0.639, 0.025, 0.275, 0.223, 0.736, 0.677, which put rows 7, 21, 14, 0 and 35 first); then the
    script printed the same. The rows the controller is given for a labels file are a function of that file alone, and a change here is a change
    of which studies are shown."""
    out = run_cases(args_of(world), 0)[1]
    assert out == ('RESULT {"demo_cases":"done","n":40,"counts":{"normal":6,"support_devices":12,"pneumothorax":12},'
                   '"candidates":{"normal":[7,21,14,0,35],"support_devices":[29,1,8,36,31],"pneumothorax":[24,17,23,30,10]},"seed":42}\n'), out


def test_each_kind_is_drawn_with_a_seed_of_its_own_42_plus_its_place_in_the_list_of_kinds(world, monkeypatch):
    assert cases.SEED == 42 and cases.CATEGORIES == ("normal", "support_devices", "pneumothorax")
    asked = []
    real = cases.draw

    def spy(rows, seed=cases.SEED, k=cases.PER_CATEGORY):
        asked.append((len(rows), seed, k))
        return real(rows, seed, k)

    monkeypatch.setattr(cases, "draw", spy)
    out = run_cases(args_of(world), 0)[1]
    assert asked == [(6, 42, 5), (12, 43, 5), (12, 44, 5)], "normal, support devices, pneumothorax: each with its own seed"
    assert the_result(out)["seed"] == 42, "the line still says the base seed, the one the plan names"


def test_two_kinds_that_qualify_on_the_same_rows_are_not_drawn_alike(tmp_path, tiny_gallery):
    """A seed shared by the kinds gave the same numbers to the same rows: a study with a pneumothorax and a support device is in both lists, and
    when the two lists are the same rows (here every study is one) a shared seed would make the two draws one. Each kind has a seed of its own,
    so the draws of a pool are as independent as the seeds are."""
    (tmp_path / DUMP_NAME).mkdir()
    labels = write_labels(tmp_path / DUMP_NAME / "chexbert_labels.json", [list(KINDS["both"]) for _ in range(40)])
    write_check(tiny_gallery, labels)
    result = the_result(run_cases(["--labels", str(labels), "--gallery", str(tiny_gallery)], 0)[1])
    assert result["counts"]["support_devices"] == result["counts"]["pneumothorax"] == 40
    support, pneumo = result["candidates"]["support_devices"], result["candidates"]["pneumothorax"]
    assert support != pneumo and len(set(support)) == len(set(pneumo)) == 5
    assert support == cases.draw(list(range(40)), 43) and pneumo == cases.draw(list(range(40)), 44) and result["candidates"]["normal"] == []


def test_a_category_with_fewer_than_five_studies_lists_them_all_and_one_with_none_lists_none(tmp_path):
    rows = [list(KINDS["normal"]), list(KINDS["normal"]), list(KINDS["support"]), list(KINDS["other"]), list(KINDS["other"])]
    labels = write_labels(tmp_path / "chexbert_labels.json", rows)
    gallery = write_check(write_manifest(tmp_path / "gallery", test=5, images=10), labels)
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
    gallery = write_check(write_manifest(tmp_path / "gallery", test=n, images=191462), labels)
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
    assert matches(shapes_of_the_wrapper(), "ERROR failed RuntimeError")


@pytest.mark.parametrize("argv", [[], ["--labels", "x"], ["--gallery", "y"], ["--labels", "x", "--gallery", "y", "--seed", "7"],
                                  ["--labels", "x", "--gallery", "y", SECRET], ["--lab", "x", "--gall", "y"], ["--" + SECRET]])
def test_a_bad_command_line_is_one_fixed_line_that_quotes_nothing_it_was_given(argv):
    code, out = run_cases(argv)
    assert (code, out) == (64, "ERROR usage\n"), (argv, out)
    assert matches(shapes_of_the_wrapper(), "ERROR usage")


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
    """The wrapper with the real script, over synthetic files: a labels file of 40 rows in results/, and a gallery directory with the manifest and
    the labels_check.json that says the gallery was checked against that dump."""
    WRAPPER = WRAPPER
    SCRIPTS = ("chat_demo_cases.py",)
    SLURM_CPUS = "1"

    def populate(self):
        (self.repo / "app").mkdir()
        for name in ("__init__.py", "labels.py"):   # the script imports the app's list of the 14 names
            shutil.copy(str(REPO_ROOT / "app" / name), str(self.repo / "app" / name))
        (self.repo / DUMP).mkdir(parents=True)
        labels = write_labels(self.repo / DUMP / "chexbert_labels.json", y_true_of(40))
        write_check(write_manifest(self.chat / "gallery" / "g13d_m3_v1", test=40, images=200), labels)
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
    labels = write_labels(other / "chexbert_labels.json", y_true_of(30))
    build = write_check(write_manifest(box.chat / "gallery" / "other_build_v2", test=30, images=100), labels)
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


@pytest.mark.parametrize("what, change, shown", [
    ("a check made against another dump", lambda box: write_check(box.gallery(), box.labels(), reference="some_other_dump"), "ERROR check_reference"),
    ("a check of a preflight", lambda box: write_check(box.gallery(), box.labels(), stage="preflight"), "ERROR check_stage"),
    ("a check over 39 studies", lambda box: write_check(box.gallery(), box.labels(), n_test=39), "ERROR check_n_test labels=40 n_test=39"),
    ("a check that found 3 differences", lambda box: write_check(box.gallery(), box.labels(), labels_mismatch=3),
     "ERROR check_mismatch refs_mismatch=0 labels_mismatch=3"),
    ("no check file", lambda box: (box.gallery() / "labels_check.json").unlink(), "ERROR check_missing"),
])
def test_a_gallery_that_is_not_bound_to_the_dump_is_the_scripts_refusal_shown_and_the_jobs_failure(what, change, shown, box):
    change(box)
    done = box.run()
    lines = job_lines(done)
    assert done.returncode == 1 and shown in lines, (what, done.stdout)
    assert lines[-1] == "ERROR the demo cases script exited with code 1 (see the logs under CHAT_HOME)" and "END chat demo cases" not in done.stdout
    assert not [l for l in lines if l.startswith("RESULT ")] and all(LINE_OK.match(l) for l in lines)
    assert "some_other_dump" not in done.stdout and DUMP_NAME not in done.stdout, "no value of the check file is quoted"


def test_the_header_says_what_holds_of_what_reaches_the_job_log_and_not_more():
    """It said "Nothing names a path, a port or an exception text". That holds of the script's lines, which are numbers and fixed words, and not of
    the log: the job's own mkdir or tee, failing under CHAT_HOME, is an error that the shell writes to the same log, and it names a path of CHAT_HOME,
    which holds the user name. What holds is that no MIMIC-derived value reaches the log, and the header says that and says what may name a path."""
    text = (REPO_ROOT / "scripts" / WRAPPER).read_text()
    header = " ".join(" ".join(l.lstrip("#") for l in text.splitlines() if l.startswith("#") and not l.startswith("#SBATCH")).split())
    assert "Nothing names a path" not in header and "Nothing names" not in header
    for phrase in ("No MIMIC-derived value reaches this log", "an infrastructure error", "mkdir", "tee", "may name a path under CHAT_HOME", "the user name",
                   "no refusal of this wrapper quotes the value it refuses", "labels_check.json"):
        assert phrase in header, phrase


def test_an_infrastructure_error_under_chat_home_may_name_its_path_in_the_log_and_nothing_derived_from_the_data(box):
    """What the header allows: the job's own mkdir fails (here CHAT_HOME/logs is a file), the shell says so on the job's one log, and the message
    names the path. It is no MIMIC-derived value, and none of the script ran."""
    (box.chat / "logs").write_text("a file where the log directory goes")
    done = box.run()
    assert done.returncode != 0 and str(box.chat) in done.stdout, done.stdout
    assert box.calls() == [] and not [l for l in job_lines(done) if l.startswith(("RESULT ", "ERROR "))]


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
CHECK_N_LINE = "ERROR check_n_test labels=2663 n_test=2662"
CHECK_MISMATCH_LINE = "ERROR check_mismatch refs_mismatch=3 labels_mismatch=2"
PRINTED = {   # every ERROR line the script can print: the code, and the number keys that come with it in the scenarios of this module
    "usage": [()], "labels_unreadable": [()], "label_names": [()], "gallery_unreadable": [()], "check_missing": [()], "check_unreadable": [()],
    "check_stage": [()], "check_reference": [()],
    "labels_shape": [("rows", "width")], "count_mismatch": [("labels", "gallery")],
    "check_n_test": [("labels",), ("labels", "n_test")],
    "check_mismatch": [(), ("refs_mismatch",), ("labels_mismatch",), ("refs_mismatch", "labels_mismatch")],
}
OTHER_KEYS = ("rows", "width", "labels", "gallery", "n_test", "refs_mismatch", "labels_mismatch", "path", "study_id", "secret", "check", "refs")
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
    # fix 1: a key that is no key of the code (the allowlist once took any `[a-z_]{1,20}=` after any code), another code's key, a key twice
    (MISMATCH_LINE, "labels=2663", "labels_x=2663"), (MISMATCH_LINE, "labels=2663", "rows=2663"), (MISMATCH_LINE, "labels=2663", "secret=2663"),
    (SHAPE_LINE, "width=13", "gallery=13"), (SHAPE_LINE, "width=13", "rows=13"), (CHECK_N_LINE, "n_test=2662", "check=2662"),
    (CHECK_N_LINE, "labels=2663", "labels=SECRET"), (CHECK_N_LINE, "n_test=2662", "n_test=2662 n_test=1"),
    (CHECK_MISMATCH_LINE, "refs_mismatch=3", "refs=3"), (CHECK_MISMATCH_LINE, "labels_mismatch=2", "labels_mismatch=-2"),
    (CHECK_MISMATCH_LINE, "refs_mismatch=3 labels_mismatch=2", "labels_mismatch=2 refs_mismatch=3"),
    ("ERROR usage", "usage", "usage rows=1"), ("ERROR check_stage", "check_stage", "check_stage stage=final"),
    ("ERROR check_missing", "check_missing", "check_missing path=1"), ("ERROR label_names", "label_names", "label_names labels=14"),
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
    assert len(shapes.splitlines()) == 7 and "'" not in shapes, "the RESULT line, five families of ERROR lines and the failure of the script"
    for line in (RESULT_LINE, MISMATCH_LINE, SHAPE_LINE, FAILED_LINE, CHECK_N_LINE, CHECK_MISMATCH_LINE, run_cases(args_of(world))[1].rstrip("\n")):
        assert matches(shapes, line), line
    empty = {name: [] for name in cases.CATEGORIES}
    one = {"normal": [0], "support_devices": [1], "pneumothorax": [2]}
    for found in (empty, one):
        assert matches(shapes, cases.result_line(cases.payload(7, found))), found
    n = 2663   # the longest line: every row of every kind, five candidates each
    longest = write_labels(tmp_path / "all.json", [list(KINDS["both"]) for _ in range(n)])
    gallery = write_check(write_manifest(tmp_path / "g", test=n, images=191462), longest)
    line = run_cases(["--labels", str(longest), "--gallery", str(gallery)], 0)[1].rstrip("\n")
    assert matches(shapes, line) and len(line) < 300, line


def error_line(code, keys):
    return " ".join(["ERROR", code] + ["{}={}".format(key, 7 + i) for i, key in enumerate(keys)])


def test_the_scripts_table_of_error_codes_and_the_keys_that_come_with_each_is_what_the_tests_list_as_printed():
    """ERROR_CODES says, for each code, the number keys its line may carry, in the order it carries them, and Refused refuses any other; PRINTED is the
    list this module's scenarios produce. The two agree: no code or key is in the table that no scenario prints, and none is printed that is not in it."""
    assert set(cases.ERROR_CODES) == set(PRINTED)
    for code, keys in cases.ERROR_CODES.items():
        assert set(keys) == set().union(*map(set, PRINTED[code])), code
        assert all(list(shown) == [k for k in keys if k in shown] for shown in PRINTED[code]), (code, "the keys come in the table's order")
    for code, keys in cases.ERROR_CODES.items():   # and Refused enforces it: a key that is not the code's cannot be printed
        for key in OTHER_KEYS:
            if key not in keys:
                with pytest.raises(AssertionError):
                    cases.Refused(code, **{key: 1})
    with pytest.raises(AssertionError):
        cases.Refused("count_mismatch", gallery=1, labels=2)
    assert cases.Refused("count_mismatch", labels=2, gallery=1).line() == "ERROR count_mismatch labels=2 gallery=1"


@pytest.mark.parametrize("code", sorted(PRINTED))
def test_every_error_line_the_script_can_print_passes_the_wrappers_allowlist(code):
    shapes = shapes_of_the_wrapper()
    for keys in PRINTED[code]:
        assert matches(shapes, error_line(code, keys)), error_line(code, keys)


@pytest.mark.parametrize("code", sorted(PRINTED))
def test_a_key_that_the_script_does_not_print_with_a_code_is_withheld_whatever_the_number_and_wherever_it_stands(code):
    """The allowlist once took any `[a-z_]{1,20}=` key after any code. Now each code takes its own keys, in the script's order, and nothing else."""
    shapes = shapes_of_the_wrapper()
    own = set().union(*map(set, PRINTED[code]))
    longest = max(PRINTED[code], key=len)
    for key in OTHER_KEYS:
        if key in own:
            continue
        assert not matches(shapes, error_line(code, (key,))), (code, key)
        assert not matches(shapes, error_line(code, longest + (key,))), (code, key, "after its own")
        assert not matches(shapes, error_line(code, (key,) + longest)), (code, key, "before its own")
    for key in own:   # a key of its own, twice, or with a number that is no number
        assert not matches(shapes, error_line(code, longest) + " {}=1".format(key)), (code, key)
        assert not matches(shapes, error_line(code, longest) + " {}=x".format(key)), (code, key)
    if longest:
        assert not matches(shapes, error_line(code, tuple(reversed(longest)))) or len(longest) == 1, (code, "wrong order")


def allowlist_codes():
    """The codes the wrapper's allowlist names in its ERROR lines, whether as one alternative group or on a line of their own (not `failed`)."""
    found = set()
    for line in shapes_of_the_wrapper().splitlines():
        parsed = re.match(r"\^ERROR (?:\(([a-z_|]+)\)|([a-z_]+))", line)
        if parsed:
            found |= set((parsed.group(1) or parsed.group(2)).split("|"))
    return found - {"failed"}


def test_the_wrappers_allowlist_names_exactly_the_error_codes_the_script_has_and_no_others():
    assert allowlist_codes() == set(cases.ERROR_CODES), sorted(allowlist_codes() ^ set(cases.ERROR_CODES))


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
    "ERROR count_mismatch", "ERROR count_mismatch labels=1", "ERROR labels_shape", "ERROR labels_shape rows=1", "ERROR check_n_test",
    "ERROR check_n_test n_test=3", "ERROR check_n_test  labels=1", "ERROR check_mismatch  refs_mismatch=1", "ERROR check_stage ", "ERROR failed",
])
def test_the_wrappers_allowlist_takes_nothing_else(leak):
    assert not matches(shapes_of_the_wrapper(), leak), leak
