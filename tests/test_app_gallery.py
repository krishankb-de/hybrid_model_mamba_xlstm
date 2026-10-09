"""CHAT_UI_PLAN.md P5-D: the retrieval gallery's loader and queries (app/gallery.py).

CPU only, offline, synthetic data only (R7). Every gallery here is written by `scripts/build_retrieval_gallery.py --tiny`, as the `tiny_gallery`
fixture of tests/conftest.py does (one per test); a module-scoped twin built the same way serves the read-only tests and copy_of() hands a test its
own copy to doctor. The tiny gallery has 200 train images, 40 test studies and 240 report rows, but every size below is read from
manifest.json["counts"], as the real gallery's will be.

The tiny gallery's duplicate reports are exact ties in a query's similarities on purpose (they are what the dedup-aware rank is for). The chapter's
recall breaks such a tie by argpartition's order and the plan's own_report_rank counts only strictly greater similarities, so they are one number
only where nothing ties: the tests below pin that case exactly (a copy whose test reports are made distinct), pin the dedup-aware rank against the
chapter's dedup-aware recall exactly (it does not care about ties inside a group), and bracket the tied case."""
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.gallery import Gallery, GalleryMismatch, _topk
from app.labels import CHEXBERT_14
from scripts import build_retrieval_gallery as bg

REPO_ROOT = Path(__file__).resolve().parent.parent
SECRET = "secret_dir_7f3a9c"        # stands for a path under CHAT_HOME: it must never come back out in an error message


# ── helpers ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

def unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def read_manifest(root: Path) -> dict:
    return json.loads((root / "manifest.json").read_text())


def write_manifest(root: Path, manifest: dict) -> None:
    (root / "manifest.json").write_text(json.dumps(manifest))


def decide_gate(root: Path) -> Path:
    """What compare_rk does on the cluster once the reference evaluation has run, through its own code: the verdict `equal` goes into gate_rk.json and
    manifest.json. A --tiny build has none, because it has no reference run, so a tiny gallery is undecided until this."""
    gate = json.loads((root / "gate_rk.json").read_text())
    (root / "reference_rk").mkdir(exist_ok=True)
    (root / "reference_rk" / "phase6_mimic_20260101T000000Z.json").write_text(json.dumps({"metrics": gate["app"]}))
    assert bg.compare_rk(root) == 0
    assert read_manifest(root)["gate_rk"]["equal"] is True
    return root


def edit_npy(root: Path, name: str, edit) -> None:
    array = np.load(str(root / name))
    np.save(str(root / name), edit(array))


def edit_parquet(root: Path, name: str, edit) -> None:
    frame = pd.read_parquet(root / name)
    edit(frame).to_parquet(root / name, index=False)


def recall(ranks, k: int) -> float:
    return float(np.mean([r <= k for r in ranks]))


def refused(root: Path, expect_tower_sha256=None) -> str:
    """Open a gallery that must be refused. Returns the message, after checking R7: it names no path and quotes no report text."""
    with pytest.raises(GalleryMismatch) as err:
        Gallery.open(root, expect_tower_sha256)
    message = str(err.value)
    assert str(root) not in message and root.parent.name not in message and SECRET not in message
    texts = root / "report_texts.txt"
    if texts.is_file():
        assert not any(line and line in message for line in texts.read_text(errors="replace").splitlines()[:60])
    return message


@pytest.fixture(scope="module")
def base(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("p5d") / "gallery"
    bg.build_tiny(root)
    return decide_gate(root)


@pytest.fixture(scope="module")
def counts(base) -> dict:
    return read_manifest(base)["counts"]


@pytest.fixture(scope="module")
def gallery(base) -> Gallery:
    return Gallery.open(base, read_manifest(base)["tower_sha256"])


@pytest.fixture
def copy_of(base, tmp_path):
    """A private copy of the decided tiny gallery to doctor (the images stay where the copied parquet files point)."""
    def make(name: str = "g") -> Path:
        target = tmp_path / SECRET / name
        shutil.copytree(str(base), str(target))
        return target
    return make


@pytest.fixture(scope="module")
def queries(gallery, base) -> np.ndarray:
    """Unit queries of every kind the pipeline will send: the test images, then train images 0..19, then 40 random directions."""
    rng = np.random.default_rng(11)
    return np.concatenate([np.load(base / "test_img_emb.npy"), gallery.img_emb[:20],
                           unit(rng.standard_normal((40, gallery.img_emb.shape[1]))).astype(np.float32)])


@pytest.fixture(scope="module")
def ref():
    """The retrieval chapter's own compute_retrieval_metrics (a few seconds to import: datasets, transformers)."""
    pytest.importorskip("datasets")
    from scripts import evaluate_cxr_retrieval
    return evaluate_cxr_retrieval


@pytest.fixture(scope="module")
def spelled(base, tmp_path_factory) -> Path:
    """The decided tiny gallery with its 240 report vectors made pairwise distinct (a jittered copy), so that the copies of one report differ as its
    spellings do in a real build. A group's best member is then not every member, and scoring by it can be told from scoring by another."""
    root = tmp_path_factory.mktemp("p5d_spelled") / "gallery"
    shutil.copytree(str(base), str(root))
    texts = np.load(root / "txt_emb.npy").astype(np.float32)
    np.save(root / "txt_emb.npy", unit(texts + 0.08 * np.random.default_rng(6).standard_normal(texts.shape)).astype(np.float16))
    return root


@pytest.fixture(scope="module")
def distinct(base, tmp_path_factory) -> Path:
    """The decided tiny gallery with its 40 test report vectors made pairwise distinct (a jittered copy): no two similarities of a query tie, which
    is the one case where the strict rank and the chapter's argpartition recall are the same number by construction."""
    root = tmp_path_factory.mktemp("p5d_distinct") / "gallery"
    shutil.copytree(str(base), str(root))
    texts = np.load(root / "txt_emb_test.npy")
    jitter = unit(texts + 0.05 * np.random.default_rng(5).standard_normal(texts.shape)).astype(np.float32)
    np.save(root / "txt_emb_test.npy", jitter)
    return root


# ── open: what is loaded ──────────────────────────────────────────────────────────────────────────────────────────────────────────────

def test_open_reads_every_size_from_the_manifests_counts(gallery, base, counts):
    manifest = read_manifest(base)
    assert gallery.img_emb.shape[0] == counts["images"] and gallery.txt_emb.shape[0] == counts["report_rows"]
    assert gallery.txt_emb_test.shape[0] == counts["test"] and len(gallery.report_texts) == counts["report_rows"]
    assert len(gallery.group_starts) == counts["report_groups"]
    assert gallery.build_id == manifest["build_id"] and gallery.manifest == manifest
    assert gallery.img_emb.shape[1] == gallery.txt_emb.shape[1] == gallery.txt_emb_test.shape[1] == gallery.dim


def test_open_loads_float32_ram_copies_of_the_embeddings(gallery, base):
    for attribute, name in (("img_emb", "img_emb.npy"), ("txt_emb", "txt_emb.npy"), ("txt_emb_test", "txt_emb_test.npy")):
        array = getattr(gallery, attribute)
        on_disk = np.load(str(base / name))
        assert array.dtype == np.float32 and array.flags["C_CONTIGUOUS"], attribute
        assert np.array_equal(array, on_disk.astype(np.float32)), attribute
        assert not isinstance(array, np.memmap) and not isinstance(array.base, np.memmap), "a RAM copy, not a mapped file"
    assert np.load(str(base / "img_emb.npy")).dtype == np.float16, "the files are half precision: the copy is what doubles them"


def test_open_loads_the_int_arrays_the_meta_and_the_texts(gallery, base, counts):
    for name in ("txt_groups", "group_order", "group_starts", "txt_test_groups", "txt_split", "txt_split_row", "img_txt_row"):
        assert np.array_equal(getattr(gallery, name), np.load(str(base / (name + ".npy")))), name
    assert gallery.report_texts == (base / "report_texts.txt").read_text().splitlines()
    assert gallery.n_train == counts["images"]


def test_nbytes_is_the_total_of_the_arrays_held(gallery, counts):
    held = [v for v in vars(gallery).values() if isinstance(v, np.ndarray)]
    assert len(held) >= 11 and gallery.nbytes == sum(a.nbytes for a in held)
    floats = 4 * gallery.dim * (counts["images"] + counts["report_rows"] + counts["test"])
    assert floats < gallery.nbytes < floats + 16 * 8 * counts["report_rows"], "the float32 embeddings dominate; the int maps and labels add a little"


def test_the_arrays_are_read_only_so_a_query_cannot_change_the_gallery(gallery):
    for name in ("img_emb", "txt_emb", "txt_emb_test", "txt_groups", "group_order", "group_starts", "img_txt_row", "labels"):
        assert not getattr(gallery, name).flags.writeable, name


def test_open_accepts_a_path_given_as_text_and_no_tower_to_check(base):
    assert Gallery.open(str(base)).build_id == read_manifest(base)["build_id"]
    assert Gallery.open(base, None).build_id == read_manifest(base)["build_id"]


def test_the_tiny_gallery_fixture_opens_once_its_gate_is_decided(tiny_gallery):
    decide_gate(tiny_gallery)
    manifest = read_manifest(tiny_gallery)
    assert Gallery.open(tiny_gallery, manifest["tower_sha256"]).img_emb.shape[0] == manifest["counts"]["images"]


# ── open: the refusals ────────────────────────────────────────────────────────────────────────────────────────────────────────────────

def test_a_mismatch_is_a_runtime_error(base):
    assert issubclass(GalleryMismatch, RuntimeError)
    with pytest.raises(RuntimeError):
        Gallery.open(base, "0" * 64)


def test_open_refuses_another_tower_and_says_nothing_about_either_hash(base):
    manifest = read_manifest(base)
    message = refused(base, "0" * 64)
    assert "tower" in message and manifest["tower_sha256"] not in message and "0" * 64 not in message
    assert Gallery.open(base, manifest["tower_sha256"]) is not None
    refused(base, "")                                       # given and empty is still given, and differs


def test_the_tower_is_checked_before_the_big_files_are_read(copy_of):
    root = copy_of()
    (root / "img_emb.npy").write_bytes(b"not an array")
    assert "tower" in refused(root, "f" * 64)


def test_open_refuses_a_gallery_whose_build_never_decided_its_gate(tiny_gallery):
    assert "equal" not in read_manifest(tiny_gallery)["gate_rk"]
    message = refused(tiny_gallery)
    assert "gate_rk" in message


@pytest.mark.parametrize("gate", [{"app": {"N": 1}}, {"app": {"N": 1}, "equal": False}, {"equal": "true"}, {"equal": 1}, {"equal": None}, [True], "equal", 7, None],
                         ids=["no-verdict", "unequal", "text-true", "one", "null", "list", "text", "number", "null-gate"])
def test_open_refuses_anything_but_a_gate_that_is_equal_true(copy_of, gate):
    root = copy_of()
    manifest = read_manifest(root)
    manifest["gate_rk"] = gate
    write_manifest(root, manifest)
    assert "gate_rk" in refused(root)


def test_open_refuses_a_manifest_with_no_gate_at_all(copy_of):
    root = copy_of()
    manifest = read_manifest(root)
    del manifest["gate_rk"]
    write_manifest(root, manifest)
    assert "gate_rk" in refused(root)


def test_the_right_tower_does_not_excuse_an_undecided_gate(tiny_gallery):
    assert "gate_rk" in refused(tiny_gallery, read_manifest(tiny_gallery)["tower_sha256"])


REQUIRED = ["manifest.json", "img_emb.npy", "txt_emb.npy", "txt_emb_test.npy", "txt_groups.npy", "group_order.npy", "group_starts.npy",
            "txt_test_groups.npy", "txt_split.npy", "txt_split_row.npy", "img_txt_row.npy", "img_meta.parquet", "test_meta.parquet",
            "report_texts.txt", "labels.npy", "label_names.json"]


@pytest.mark.parametrize("name", REQUIRED)
def test_open_refuses_a_missing_file_and_names_only_its_basename(copy_of, name):
    root = copy_of()
    (root / name).unlink()
    assert name in refused(root)


def test_open_names_every_missing_file_at_once(copy_of):
    root = copy_of()
    for name in ("txt_groups.npy", "report_texts.txt"):
        (root / name).unlink()
    message = refused(root)
    assert "txt_groups.npy" in message and "report_texts.txt" in message


def test_open_refuses_a_directory_that_is_not_there(tmp_path):
    message = refused(tmp_path / SECRET / "nothing")
    assert "manifest.json" in message


@pytest.mark.parametrize("text", ["", "{", "[]", "null", "7"])
def test_open_refuses_a_manifest_that_is_not_a_json_object(copy_of, text):
    root = copy_of()
    (root / "manifest.json").write_text(text)
    assert "manifest.json" in refused(root)


@pytest.mark.parametrize("edit", [lambda c: c.pop("images"), lambda c: c.pop("test"), lambda c: c.update(report_groups=0),
                                  lambda c: c.update(images="200"), lambda c: c.update(test=True), lambda c: c.update(test=-1)],
                         ids=["no-images", "no-test", "zero-groups", "text-count", "bool-count", "negative"])
def test_open_refuses_counts_that_are_not_positive_integers(copy_of, edit):
    root = copy_of()
    manifest = read_manifest(root)
    edit(manifest["counts"])
    write_manifest(root, manifest)
    assert "counts" in refused(root)
    manifest["counts"] = None
    write_manifest(root, manifest)
    assert "counts" in refused(root)


def test_open_refuses_counts_that_do_not_add_up_and_gives_them(copy_of, counts):
    root = copy_of()
    manifest = read_manifest(root)
    manifest["counts"]["report_rows"] += 3
    write_manifest(root, manifest)
    message = refused(root)
    assert str(counts["report_rows"] + 3) in message and str(counts["images"]) in message and str(counts["test"]) in message


@pytest.mark.parametrize("name", ["img_emb.npy", "txt_emb.npy", "txt_emb_test.npy", "txt_groups.npy", "group_order.npy", "group_starts.npy",
                                  "txt_test_groups.npy", "txt_split.npy", "txt_split_row.npy", "img_txt_row.npy", "labels.npy"])
def test_open_refuses_an_array_with_a_row_fewer_than_the_counts_say_and_gives_both_numbers(copy_of, counts, name):
    root = copy_of()
    rows = {"img_emb.npy": counts["images"], "txt_emb.npy": counts["report_rows"], "txt_emb_test.npy": counts["test"],
            "txt_groups.npy": counts["report_rows"], "group_order.npy": counts["report_rows"], "group_starts.npy": counts["report_groups"],
            "txt_test_groups.npy": counts["test"], "txt_split.npy": counts["report_rows"], "txt_split_row.npy": counts["report_rows"],
            "img_txt_row.npy": counts["images"], "labels.npy": counts["report_rows"]}[name]
    edit_npy(root, name, lambda a: a[:-1])
    message = refused(root)
    assert name in message and str(rows) in message and str(rows - 1) in message


def test_open_refuses_embeddings_that_disagree_about_the_dimension(copy_of, gallery):
    root = copy_of()
    edit_npy(root, "txt_emb.npy", lambda a: a[:, :8])
    message = refused(root)
    assert "txt_emb.npy" in message and "8" in message and str(gallery.dim) in message


def test_open_refuses_a_one_dimensional_embedding_file(copy_of):
    root = copy_of()
    edit_npy(root, "img_emb.npy", lambda a: a.reshape(-1))
    assert "img_emb.npy" in refused(root)
    right_length = copy_of("right_length")                          # one number per image: the right number of rows, but not a matrix
    edit_npy(right_length, "img_emb.npy", lambda a: a[:, 0].copy())
    assert "img_emb.npy" in refused(right_length)


def test_open_refuses_an_int_array_that_is_a_column_where_a_vector_is_expected(copy_of):
    root = copy_of()
    edit_npy(root, "txt_groups.npy", lambda a: a[:, None].copy())
    assert "txt_groups.npy" in refused(root)


def test_open_refuses_embeddings_with_no_columns(copy_of):
    root = copy_of()
    for name in ("img_emb.npy", "txt_emb.npy", "txt_emb_test.npy"):
        edit_npy(root, name, lambda a: a[:, :0].copy())
    assert "columns" in refused(root)


@pytest.mark.parametrize("name, dtype", [("img_emb.npy", np.int16), ("txt_emb_test.npy", np.int32), ("group_order.npy", np.float32),
                                         ("txt_split.npy", np.float64), ("labels.npy", np.float32)])
def test_open_refuses_an_array_of_the_wrong_kind_of_number(copy_of, name, dtype):
    root = copy_of()
    edit_npy(root, name, lambda a: a.astype(dtype))
    assert name in refused(root)


def test_open_refuses_an_unreadable_array_and_never_unpickles_one(copy_of, counts):
    root = copy_of()
    (root / "img_emb.npy").write_bytes(b"\x93NUMPY not an array at all")
    assert "img_emb.npy" in refused(root)
    pickled = copy_of("pickled")
    np.save(str(pickled / "txt_emb.npy"), np.array([{"a": 1}] * counts["report_rows"], dtype=object), allow_pickle=True)
    assert "txt_emb.npy" in refused(pickled)


def test_open_refuses_a_text_file_with_the_wrong_number_of_reports_and_gives_the_counts(copy_of, counts):
    root = copy_of()
    lines = (root / "report_texts.txt").read_text().splitlines()
    (root / "report_texts.txt").write_text("\n".join(lines[:-1]) + "\n")
    message = refused(root)
    assert "report_texts.txt" in message and str(counts["report_rows"]) in message and str(counts["report_rows"] - 1) in message
    (root / "report_texts.txt").write_text("\n".join(lines))         # the last line is cut short of its newline: a truncated write
    assert "report_texts.txt" in refused(root)
    (root / "report_texts.txt").write_bytes(b"\xff\xfe not utf-8\n" * counts["report_rows"])
    assert "report_texts.txt" in refused(root)


def test_open_refuses_meta_with_a_row_fewer_than_the_counts_say(copy_of, counts):
    for name, rows in (("img_meta.parquet", counts["images"]), ("test_meta.parquet", counts["test"])):
        root = copy_of(name)
        edit_parquet(root, name, lambda f: f.iloc[:-1])
        message = refused(root)
        assert name in message and str(rows) in message and str(rows - 1) in message


@pytest.mark.parametrize("name, key", [("img_meta.parquet", "row"), ("test_meta.parquet", "test_row")])
def test_open_refuses_meta_whose_rows_are_not_numbered_in_order(copy_of, name, key):
    root = copy_of()
    edit_parquet(root, name, lambda f: f.assign(**{key: f[key].iloc[::-1].to_numpy()}))
    assert name in refused(root)


@pytest.mark.parametrize("name, column", [("img_meta.parquet", "file_sha256"), ("img_meta.parquet", "image"), ("img_meta.parquet", "study_id"),
                                          ("test_meta.parquet", "file_sha256"), ("test_meta.parquet", "view"), ("img_meta.parquet", "row")])
def test_open_refuses_meta_without_a_column_it_needs(copy_of, name, column):
    root = copy_of()
    edit_parquet(root, name, lambda f: f.drop(columns=[column]))
    assert name in refused(root)


def test_open_refuses_a_meta_file_that_is_not_parquet(copy_of):
    root = copy_of()
    (root / "img_meta.parquet").write_bytes(b"this is not parquet")
    assert "img_meta.parquet" in refused(root)


def test_open_refuses_a_group_order_that_is_not_a_permutation(copy_of):
    root = copy_of()
    edit_npy(root, "group_order.npy", lambda a: np.r_[a[1:], a[1]])
    assert "group_order.npy" in refused(root)
    outside = copy_of("outside")
    edit_npy(outside, "group_order.npy", lambda a: np.r_[a[:-1], -1])
    assert "group_order.npy" in refused(outside)
    beyond = copy_of("beyond")
    edit_npy(beyond, "group_order.npy", lambda a: np.r_[a[:-1], len(a)])
    assert "group_order.npy" in refused(beyond)
    wrapped = copy_of("wrapped")                  # every row written as a negative number: numpy would index the same rows, and the queries would then answer with row -5
    edit_npy(wrapped, "group_order.npy", lambda a: a - len(a))
    assert "group_order.npy" in refused(wrapped)


def swap_interior(starts):
    starts = starts.copy()
    starts[1], starts[2] = starts[2], starts[1]
    return starts


@pytest.mark.parametrize("edit", [lambda s: np.r_[1, s[1:]], lambda s: s[::-1].copy(), lambda s: np.r_[s[:-1], s[-2]], lambda s: np.r_[s[:-1], 10 ** 6],
                                  swap_interior],
                         ids=["first-not-zero", "descending", "repeated", "beyond-the-rows", "interior-swapped"])
def test_open_refuses_group_starts_that_do_not_cut_the_order_into_ordered_segments(copy_of, edit):
    root = copy_of()
    edit_npy(root, "group_starts.npy", edit)
    assert "group_starts.npy" in refused(root)


INT_FILES = ["txt_groups.npy", "group_order.npy", "group_starts.npy", "txt_test_groups.npy", "txt_split.npy", "txt_split_row.npy", "img_txt_row.npy"]


def test_unsigned_int_files_are_read_as_int64_so_that_a_falling_start_cannot_wrap_around(copy_of, gallery, queries):
    root = copy_of()
    for name in INT_FILES:
        edit_npy(root, name, lambda a: a.astype(np.uint32))
    g = Gallery.open(root, None)
    assert all(getattr(g, name[:-4]).dtype == np.int64 for name in INT_FILES)
    same_results(g.report_matches(queries[1], 5), gallery.report_matches(queries[1], 5))
    descending = copy_of("descending")                                # as uint32 a falling start would be a rise of four billion
    edit_npy(descending, "group_starts.npy", lambda s: s[::-1].astype(np.uint32))
    assert "group_starts.npy" in refused(descending)


def test_open_refuses_a_group_order_that_lists_one_row_of_a_group_twice_and_misses_another(copy_of):
    """Still sorted by group, so the ids line up; but the row left out could be the group's best, and its score would be too low without a sound."""
    root = copy_of()

    def repeat_within_a_group(order):
        starts = np.load(str(root / "group_starts.npy"))
        group = int(np.flatnonzero(np.diff(np.append(starts, len(order))) >= 2)[0])
        order = order.copy()
        order[starts[group] + 1] = order[starts[group]]
        return order
    edit_npy(root, "group_order.npy", repeat_within_a_group)
    assert "group_order.npy" in refused(root)


def test_open_refuses_group_ids_that_the_order_and_the_starts_do_not_agree_with(copy_of):
    root = copy_of()

    def swap(groups):
        a, b = 0, int(np.flatnonzero(groups != groups[0])[0])
        groups[[a, b]] = groups[[b, a]]
        return groups
    edit_npy(root, "txt_groups.npy", swap)
    assert "txt_groups.npy" in refused(root)


@pytest.mark.parametrize("name, edit", [("txt_split.npy", lambda a: np.r_[a[:-1], 0]), ("txt_split.npy", lambda a: np.r_[1, a[1:]]),
                                        ("txt_split_row.npy", lambda a: np.r_[a[:-1], 0]), ("txt_split_row.npy", lambda a: np.r_[7, a[1:]])],
                         ids=["last-is-train", "first-is-test", "last-test-row-wrong", "first-train-row-wrong"])
def test_open_refuses_split_maps_that_do_not_lay_the_train_rows_before_the_test_rows(copy_of, name, edit):
    root = copy_of()
    edit_npy(root, name, edit)
    assert name in refused(root)


@pytest.mark.parametrize("edit", [lambda a: np.r_[a[:-1], 10 ** 6], lambda a: np.r_[a[:-1], -1], lambda a: a - 10 ** 6], ids=["beyond", "negative", "all-negative"])
def test_open_refuses_an_image_whose_report_row_is_outside_the_gallery(copy_of, edit):
    root = copy_of()
    edit_npy(root, "img_txt_row.npy", edit)
    assert "img_txt_row.npy" in refused(root)


def test_open_refuses_report_rows_written_as_negative_numbers_that_numpy_would_wrap(copy_of, counts):
    root = copy_of()
    edit_npy(root, "img_txt_row.npy", lambda a: a - counts["report_rows"])
    assert "img_txt_row.npy" in refused(root)


# ── open: labels ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

def test_labels_are_the_files_labels_while_the_manifest_says_done(gallery, base):
    assert read_manifest(base)["labels_status"] == "done"
    assert gallery.labels.dtype == np.uint8 and np.array_equal(gallery.labels, np.load(str(base / "labels.npy")))
    assert gallery.label_names == CHEXBERT_14 == json.loads((base / "label_names.json").read_text())


def test_labels_are_none_while_the_build_is_pending_and_no_label_file_is_needed(tmp_path):
    root = tmp_path / "pending"
    manifest = bg.build_tiny(root, with_labels=False)
    assert manifest["labels_status"] == "pending"
    decide_gate(root)
    pending = Gallery.open(root, None)
    assert pending.labels is None and pending.label_names is None
    assert all(n["labels"] is None for n in pending.image_neighbors(pending.img_emb[3], 4))
    assert all(m["labels"] is None for m in pending.report_matches(pending.img_emb[3], 3))


@pytest.mark.parametrize("status", ["pending", "partial", "", None, "DONE", True, 1], ids=["pending", "partial", "empty", "null", "upper", "bool", "one"])
def test_labels_are_none_unless_the_status_is_exactly_done_even_when_the_files_are_there(copy_of, status):
    root = copy_of()
    manifest = read_manifest(root)
    manifest["labels_status"] = status
    write_manifest(root, manifest)
    assert Gallery.open(root, None).labels is None


def test_labels_are_none_when_the_status_key_is_missing(copy_of):
    root = copy_of()
    manifest = read_manifest(root)
    del manifest["labels_status"]
    write_manifest(root, manifest)
    assert Gallery.open(root, None).labels is None


def test_open_refuses_label_names_that_are_not_the_chexbert_14_in_order(copy_of):
    root = copy_of()
    (root / "label_names.json").write_text(json.dumps(list(reversed(CHEXBERT_14))))
    assert "label_names.json" in refused(root)
    short = copy_of("short")
    (short / "label_names.json").write_text(json.dumps(CHEXBERT_14[:13]))
    assert "label_names.json" in refused(short)
    junk = copy_of("junk")
    (junk / "label_names.json").write_text("{")
    assert "label_names.json" in refused(junk)


def test_open_refuses_labels_of_the_wrong_width_or_with_a_value_that_is_not_0_or_1(copy_of):
    root = copy_of()
    edit_npy(root, "labels.npy", lambda a: a[:, :13])
    message = refused(root)
    assert "labels.npy" in message and "13" in message and "14" in message
    values = copy_of("values")
    edit_npy(values, "labels.npy", lambda a: np.where(np.arange(a.size).reshape(a.shape) == 5, 2, a).astype(np.uint8))
    assert "labels.npy" in refused(values)


# ── query hygiene ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

METHODS = {"image_neighbors": lambda g, q: g.image_neighbors(q, 4), "report_matches": lambda g, q: g.report_matches(q, 3),
           "own_report_rank": lambda g, q: g.own_report_rank(q, 3)}


def same_results(a, b) -> None:
    """Equal in every field, the similarities to a float32 rounding."""
    assert type(a) is type(b)
    if isinstance(a, list):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            same_results(x, y)
        return
    assert a.keys() == b.keys()
    for key in a:
        if key == "similarity":
            assert a[key] == pytest.approx(b[key], abs=1e-6)
        else:
            assert a[key] == b[key], key


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("bad", ["short", "long", "empty", "matrix", "column", "scalar", "text", "ragged"])
def test_a_query_of_the_wrong_shape_is_a_value_error_with_the_numbers(gallery, method, bad):
    dim = gallery.dim
    query = {"short": np.ones(dim - 1), "long": np.ones(dim + 1), "empty": np.ones(0), "matrix": np.ones((2, dim)), "column": np.ones((dim, 1)),
             "scalar": np.float32(1.0), "text": ["a"] * dim, "ragged": [[1.0], [1.0, 2.0]]}[bad]
    with pytest.raises(ValueError) as err:
        METHODS[method](gallery, query)
    if bad in ("short", "long", "empty"):
        assert str(dim) in str(err.value) and str(len(query)) in str(err.value)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("poison", [np.nan, np.inf, -np.inf])
def test_a_query_with_a_value_that_is_not_finite_is_a_value_error_that_counts_them(gallery, method, poison):
    query = np.ones(gallery.dim, dtype=np.float32)
    query[2] = query[5] = poison
    with pytest.raises(ValueError) as err:
        METHODS[method](gallery, query)
    assert "2" in str(err.value) and "finite" in str(err.value)


@pytest.mark.parametrize("method", METHODS)
def test_a_query_of_zero_length_is_a_value_error(gallery, method):
    with pytest.raises(ValueError):
        METHODS[method](gallery, np.zeros(gallery.dim, dtype=np.float32))


def test_a_query_too_big_for_float32_is_a_value_error_not_garbage(gallery):
    with pytest.raises(ValueError):
        gallery.image_neighbors(np.full(gallery.dim, 1e300), 3)


def test_a_query_that_is_invalid_is_refused_even_when_k_is_zero(gallery):
    with pytest.raises(ValueError):
        gallery.image_neighbors(np.ones(3), 0)
    with pytest.raises(ValueError):
        gallery.report_matches(np.ones(3), 0)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("scale", [1e-3, 0.25, 7.5, 1e4])
def test_every_method_normalises_the_query_so_a_caller_cannot_pass_one_unnormalised(gallery, queries, method, scale):
    for q in queries[::13]:
        same_results(METHODS[method](gallery, q * np.float32(scale)), METHODS[method](gallery, q))


def test_an_unnormalised_query_still_gets_cosines_not_dot_products(gallery):
    q = unit(np.random.default_rng(2).standard_normal(gallery.dim))
    cosine = gallery.image_neighbors(q.astype(np.float32), 6)
    scaled = gallery.image_neighbors(10 * q, 6)
    assert [n["gallery_row"] for n in scaled] == [n["gallery_row"] for n in cosine]
    assert max(n["similarity"] for n in scaled) <= 1 + 2e-3
    for n in scaled:
        assert n["similarity"] == pytest.approx(float(gallery.img_emb[n["gallery_row"]].astype(np.float64) @ q), abs=1e-5)


def test_queries_may_be_lists_float64_or_float16_and_come_back_the_same(gallery, queries):
    q = queries[7]
    reference = gallery.report_matches(q, 3)
    for variant in (q.tolist(), q.astype(np.float64), tuple(q.tolist())):
        same_results(gallery.report_matches(variant, 3), reference)
    own = gallery.img_emb[5]                             # train image 5: its first neighbour is itself, far ahead of the rest, even at half precision
    assert gallery.image_neighbors(own.astype(np.float16), 1)[0]["gallery_row"] == gallery.image_neighbors(own, 1)[0]["gallery_row"] == 5


def test_the_query_is_never_changed_by_a_call(gallery, queries):
    q = (queries[3] * 5).copy()
    before = q.copy()
    for method in METHODS.values():
        method(gallery, q)
    assert np.array_equal(q, before)


@pytest.mark.parametrize("k, images, reports", [(0, 0, 0), (-1, 0, 0), (-10 ** 6, 0, 0), (1, 1, 1), (7, 7, 7), (10, 10, 10), (11, 11, 10), (12, 12, 10),
                                                (13, 12, 10), (1000, 12, 10)])
def test_k_is_clamped_to_12_images_and_10_reports_and_zero_gives_nothing(gallery, queries, k, images, reports):
    assert len(gallery.image_neighbors(queries[0], k)) == images
    assert len(gallery.report_matches(queries[0], k)) == reports


def test_the_clamped_results_are_the_prefix_of_the_unclamped_ones(gallery, queries):
    for q in queries[:5]:
        everything = gallery.image_neighbors(q, 12)
        assert [n["rank"] for n in everything] == list(range(1, 13))
        same_results(gallery.image_neighbors(q, 5), everything[:5])
        same_results(gallery.image_neighbors(q, 500), everything)
        reports = gallery.report_matches(q, 10)
        same_results(gallery.report_matches(q, 4), reports[:4])
        same_results(gallery.report_matches(q, 500), reports)


def test_k_may_be_a_numpy_integer(gallery, queries):
    assert len(gallery.image_neighbors(queries[0], np.int64(3))) == 3 and len(gallery.report_matches(queries[0], np.int32(2))) == 2


# ── _topk ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

def test_topk_is_the_k_largest_in_descending_order_and_never_more_than_there_are():
    sims = np.random.default_rng(1).standard_normal(500).astype(np.float32)
    order = np.argsort(-sims, kind="stable")
    for k in (1, 2, 7, 499, 500, 900):
        assert _topk(sims, k).tolist() == order[:min(k, 500)].tolist()
    assert _topk(sims[:1], 5).tolist() == [0]


# ── image -> image ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

def test_every_gallery_image_retrieves_itself_first(gallery, base, counts):
    meta = pd.read_parquet(base / "img_meta.parquet")
    for row in range(counts["images"]):
        first = gallery.image_neighbors(gallery.img_emb[row], 4)[0]
        assert (first["rank"], first["gallery_row"]) == (1, row)
        assert first["similarity"] == pytest.approx(1.0, abs=2e-3) and first["study_id"] == meta["study_id"][row]


def test_image_neighbors_are_the_k_most_similar_images_with_their_own_cosines(gallery, queries):
    for q in queries:
        unit_q = (q / np.linalg.norm(q)).astype(np.float64)
        sims = gallery.img_emb.astype(np.float64) @ unit_q
        found = gallery.image_neighbors(q, 12)
        rows = [n["gallery_row"] for n in found]
        assert len(set(rows)) == 12 and [n["rank"] for n in found] == list(range(1, 13))
        reported = np.array([n["similarity"] for n in found])
        assert (np.diff(reported) <= 0).all(), "non-increasing"
        assert reported == pytest.approx(sims[rows], abs=1e-5), "each is the cosine of that image"
        assert np.delete(sims, rows).max() <= reported.min() + 1e-5, "nothing better was left out"


def test_image_neighbors_follow_img_txt_row_and_not_the_row_number(copy_of, counts):
    """In a build one image has one report and img_txt_row is the identity, so a lookup by the wrong index would go unseen; here it is not."""
    root = copy_of()
    edit_npy(root, "img_txt_row.npy", lambda a: a[::-1].copy())
    g = Gallery.open(root, None)
    labels = np.load(str(root / "labels.npy"))
    found = g.image_neighbors(g.img_emb[3], 12)
    for n in found:
        report_row = counts["images"] - 1 - n["gallery_row"]
        assert n["txt_row"] == report_row and list(n["labels"].values()) == labels[report_row].tolist()
    assert any(labels[n["txt_row"]].tolist() != labels[n["gallery_row"]].tolist() for n in found), "the permutation changes at least one label row"


def test_an_image_neighbor_carries_the_row_the_study_the_report_row_the_url_and_the_labels_of_that_report(gallery, base):
    meta = pd.read_parquet(base / "img_meta.parquet")
    own_report = np.load(str(base / "img_txt_row.npy"))
    labels = np.load(str(base / "labels.npy"))
    for n in gallery.image_neighbors(gallery.img_emb[5], 12):
        row = n["gallery_row"]
        assert set(n) == {"rank", "similarity", "gallery_row", "study_id", "txt_row", "image_url", "labels"}
        assert n["study_id"] == meta["study_id"][row] and n["txt_row"] == own_report[row]
        assert n["image_url"] == "/v1/gallery/images/{}".format(row)
        assert list(n["labels"]) == CHEXBERT_14 and list(n["labels"].values()) == labels[own_report[row]].tolist()


# ── image -> report ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────

def check_report_matches(g: Gallery, root: Path, queries: np.ndarray) -> float:
    """Brute force, group by group, over the gallery g opened from root. Returns the widest gap between a returned group's best and worst member."""
    groups = np.load(str(root / "txt_groups.npy"))
    labels = np.load(str(root / "labels.npy"))
    txt = g.txt_emb.astype(np.float64)
    widest = 0.0
    for q in queries:
        sims = txt @ (q / np.linalg.norm(q)).astype(np.float64)
        best = np.array([sims[groups == k].max() for k in range(groups.max() + 1)])           # every group, scored by its best member
        worst = np.array([sims[groups == k].min() for k in range(groups.max() + 1)])
        found = g.report_matches(q, 10)
        ids = [m["group"] for m in found]
        assert len(set(ids)) == 10 and [m["rank"] for m in found] == list(range(1, 11)), "distinct groups"
        reported = np.array([m["similarity"] for m in found])
        assert (np.diff(reported) <= 0).all(), "non-increasing"
        assert reported == pytest.approx(best[ids], abs=1e-5), "each group is scored by its best member"
        assert np.delete(best, ids).max() <= reported.min() + 1e-5, "no better group was left out"
        widest = max(widest, float((best - worst)[ids].max()))
        for m in found:
            members = np.flatnonzero(groups == m["group"])
            assert m["group_size"] == len(members) and groups[m["txt_row"]] == m["group"]
            assert sims[m["txt_row"]] == pytest.approx(best[m["group"]], abs=1e-5), "txt_row is a best member of its group"
            assert m["report"] == g.report_texts[m["txt_row"]] and list(m["labels"].values()) == labels[m["txt_row"]].tolist()
    return widest


def test_report_matches_are_distinct_groups_in_non_increasing_similarity_each_scored_by_its_best_member(gallery, base, queries):
    check_report_matches(gallery, base, queries)


def test_a_group_is_scored_by_its_best_member_when_the_copies_of_a_report_are_not_the_same_vector(spelled, queries):
    """Reports that read the same up to case and spacing are one group but are different token sequences, so different vectors (the tiny
    gallery's duplicates are one vector, where best, worst and mean member are all the same number)."""
    g = Gallery.open(spelled, None)
    assert check_report_matches(g, spelled, queries) > 1e-2, "some returned group has members that differ, or this test shows nothing"


def test_a_report_match_has_exactly_the_fields_the_retrieve_stage_documents(gallery, queries):
    for m in gallery.report_matches(queries[3], 10):
        assert set(m) == {"rank", "similarity", "group", "group_size", "txt_row", "report", "labels"}
        assert m["report"].startswith("Findings: ") and list(m["labels"]) == CHEXBERT_14


def test_a_report_group_of_several_rows_is_reported_once_with_its_size(gallery, base):
    groups = np.load(str(base / "txt_groups.npy"))
    big = int(np.bincount(groups).argmax())
    assert np.bincount(groups)[big] >= 4, "the tiny gallery has a report that several studies share"
    q = gallery.txt_emb[int(np.flatnonzero(groups == big)[0])]
    top = gallery.report_matches(q, 3)[0]
    assert top["group"] == big and top["group_size"] == int(np.bincount(groups)[big]) and top["similarity"] == pytest.approx(1.0, abs=2e-3)
    assert [m["group"] for m in gallery.report_matches(q, 10)].count(big) == 1


# ── the chapter's own-report rank ─────────────────────────────────────────────────────────────────────────────────────────────────────

def test_own_report_rank_is_one_plus_the_number_of_strictly_greater_similarities(distinct, counts):
    """On a gallery whose test reports are distinct, so that the formula is well defined to the last bit."""
    g = Gallery.open(distinct, None)
    test_img = np.load(str(distinct / "test_img_emb.npy"))
    groups = np.load(str(distinct / "txt_test_groups.npy"))
    sims = test_img.astype(np.float64) @ g.txt_emb_test.astype(np.float64).T
    assert np.diff(np.sort(sims, axis=1), axis=1).min() > 1e-6, "no two similarities of a query are close enough to swap under float32 rounding"
    for row in range(counts["test"]):
        got = g.own_report_rank(test_img[row], row)
        best_copy = sims[row][groups == groups[row]].max()                # the best-scored report among those that read the same
        assert got["rank"] == 1 + int((sims[row] > sims[row, row]).sum()), row
        assert got["rank_dedup"] == 1 + int((sims[row] > best_copy).sum()), row
        assert got["of"] == counts["test"] and got["hit_at_10"] is (got["rank"] <= 10)


def test_ranks_on_the_test_split_reproduce_the_chapters_recall_when_no_two_reports_tie(distinct, counts):
    g = Gallery.open(distinct, None)
    test_img = np.load(str(distinct / "test_img_emb.npy"))
    ranks = [g.own_report_rank(test_img[row], row)["rank"] for row in range(counts["test"])]
    chapter = bg._recall_metrics(test_img, g.txt_emb_test)        # P5-B's light copy of compute_retrieval_metrics(groups=None), pinned equal to it
    assert chapter["N"] == counts["test"] and 0 < chapter["i2t_R@1"] < chapter["i2t_R@10"] <= 1
    for k in (1, 5, 10):
        assert recall(ranks, k) == chapter["i2t_R@{}".format(k)], k


def test_ranks_on_the_test_split_reproduce_the_chapters_recall_through_the_real_function_too(distinct, counts, ref):
    g = Gallery.open(distinct, None)
    test_img = np.load(str(distinct / "test_img_emb.npy"))
    ranks = [g.own_report_rank(test_img[row], row)["rank"] for row in range(counts["test"])]
    chapter = ref.compute_retrieval_metrics(test_img, g.txt_emb_test)
    for k in (1, 5, 10):
        assert recall(ranks, k) == chapter["i2t_R@{}".format(k)], k


def test_the_dedup_rank_reproduces_the_chapters_dedup_aware_recall_ties_and_all(gallery, base, counts, ref):
    """A tie between two copies of one report is a tie inside a group, so it cannot move the best member of the group: exact on the tied gallery."""
    test_img = np.load(str(base / "test_img_emb.npy"))
    groups = np.load(str(base / "txt_test_groups.npy"))
    assert np.bincount(groups).max() >= 2, "the tiny test split repeats a report"
    ranks = [gallery.own_report_rank(test_img[row], row)["rank_dedup"] for row in range(counts["test"])]
    chapter = ref.compute_retrieval_metrics(test_img, gallery.txt_emb_test, groups=groups)
    for k in (1, 5, 10):
        assert recall(ranks, k) == chapter["i2t_R@{}".format(k)], k


def test_on_tied_reports_the_strict_rank_is_the_optimistic_end_of_what_the_chapters_recall_can_be(gallery, base, counts):
    """Exact ties (a report several test studies share) are decided in the strict rank's favour: it counts only what is strictly greater. The chapter's
    argpartition recall decides them by position, so its recall lies between the all-ties-lost and all-ties-won ranks, which are the same only without
    ties. The band is a millionth wide, so that float rounding on a duplicate cannot move a row out of it."""
    test_img = np.load(str(base / "test_img_emb.npy"))
    sims = test_img.astype(np.float64) @ gallery.txt_emb_test.astype(np.float64).T
    own = np.diag(sims)[:, None]
    tol = 1e-6
    won = 1 + (sims > own + tol).sum(axis=1)                  # every tie decided in the report's favour
    lost = (sims >= own - tol).sum(axis=1)                    # every tie decided against it: itself and all that tie or beat it
    got = [gallery.own_report_rank(test_img[row], row)["rank"] for row in range(counts["test"])]
    assert (won <= got).all() and (got <= lost).all()
    assert (won < lost).any(), "the tiny test split has ties, or this test shows nothing"
    chapter = bg._recall_metrics(test_img, gallery.txt_emb_test)
    for k in (1, 5, 10):
        assert recall(lost, k) <= chapter["i2t_R@{}".format(k)] <= recall(won, k), k
    assert recall(won, 1) > recall(lost, 1)


def test_the_dedup_rank_is_never_worse_than_the_rank_and_equal_for_a_report_nobody_repeats(gallery, base, counts):
    test_img = np.load(str(base / "test_img_emb.npy"))
    groups = np.load(str(base / "txt_test_groups.npy"))
    singles = np.bincount(groups)[groups] == 1
    assert singles.any() and (~singles).any()
    for row in range(counts["test"]):
        got = gallery.own_report_rank(test_img[row], row)
        assert 1 <= got["rank_dedup"] <= got["rank"] <= counts["test"], row
        if singles[row]:
            assert got["rank_dedup"] == got["rank"], row


def test_the_dedup_rank_is_strictly_better_where_another_copy_of_the_report_scores_higher(distinct, counts):
    """In the tiny gallery the copies of a report are the same vector, so the best copy is the report itself; here they differ, as the spellings of one
    report do in a real build (a different token sequence is a different vector)."""
    g = Gallery.open(distinct, None)
    test_img = np.load(str(distinct / "test_img_emb.npy"))
    better = 0
    for row in range(counts["test"]):
        got = g.own_report_rank(test_img[row], row)
        assert 1 <= got["rank_dedup"] <= got["rank"], row
        better += got["rank_dedup"] < got["rank"]
    assert better > 0


def test_own_report_rank_has_the_fields_the_retrieve_stage_documents_in_plain_types(gallery, base, counts):
    got = gallery.own_report_rank(np.load(str(base / "test_img_emb.npy"))[4], 4)
    assert set(got) == {"rank", "of", "rank_dedup", "hit_at_10", "protocol"}
    assert type(got["rank"]) is int and type(got["of"]) is int and type(got["rank_dedup"]) is int and type(got["hit_at_10"]) is bool
    assert got["of"] == counts["test"] and got["hit_at_10"] == (got["rank"] <= 10)
    assert got["protocol"] == "i2t, official test split, strict pairing (compute_retrieval_metrics, groups=None)"


def test_hit_at_10_is_true_at_rank_10_and_false_at_rank_11(distinct):
    """The tiny test split never ranks a report exactly 10th for its own image, so the boundary is looked for among random queries (seeded). Pairs
    whose own similarity is within 1e-5 of another's are skipped: float32 rounding could decide those."""
    g = Gallery.open(distinct, None)
    reports = g.txt_emb_test.astype(np.float64)
    rng = np.random.default_rng(3)
    seen = set()
    for q in unit(rng.standard_normal((400, g.dim))):
        sims = reports @ q
        for row in rng.integers(0, len(reports), 8):
            others = np.delete(sims, row)
            if np.abs(others - sims[row]).min() < 1e-5:
                continue
            rank = 1 + int((sims > sims[row]).sum())
            got = g.own_report_rank(q.astype(np.float32), int(row))
            assert got["rank"] == rank and got["hit_at_10"] is (rank <= 10)
            seen.add(rank)
    assert {9, 10, 11} <= seen, sorted(seen)


def test_own_report_rank_needs_a_test_row_that_exists(gallery, queries, counts):
    for bad in (-1, counts["test"], counts["test"] + 5, 10 ** 9):
        with pytest.raises(IndexError) as err:
            gallery.own_report_rank(queries[0], bad)
        assert str(counts["test"] - 1) in str(err.value)
    for bad in (1.5, "3", None):
        with pytest.raises(TypeError):
            gallery.own_report_rank(queries[0], bad)
    assert gallery.own_report_rank(queries[0], np.int64(2))["of"] == counts["test"]
    assert gallery.own_report_rank(queries[0], counts["test"] - 1)["of"] == counts["test"]


# ── identical images, test studies ────────────────────────────────────────────────────────────────────────────────────────────────────

def file_hash(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_find_identical_finds_a_train_image_by_the_hash_of_its_file(gallery, base, counts):
    meta = pd.read_parquet(base / "img_meta.parquet")
    for row in (0, 1, 7, counts["images"] // 2, counts["images"] - 1):
        assert gallery.find_identical(file_hash(meta["image"][row])) == {"split": "train", "row": row}


def test_find_identical_finds_a_test_image_and_says_it_is_a_test_image(gallery, base, counts):
    meta = pd.read_parquet(base / "test_meta.parquet")
    for row in (0, 5, counts["test"] - 1):
        assert gallery.find_identical(file_hash(meta["image"][row])) == {"split": "test", "row": row}


def test_find_identical_is_none_for_a_file_that_is_in_neither_split(gallery, base):
    assert gallery.find_identical(hashlib.sha256(b"an upload nobody has seen").hexdigest()) is None
    for junk in ("", " ", "xyz", "0" * 64, None, 123):
        assert gallery.find_identical(junk) is None


def test_find_identical_ignores_case_and_surrounding_space(gallery, base):
    sha = file_hash(pd.read_parquet(base / "img_meta.parquet")["image"][9])
    assert gallery.find_identical(sha.upper()) == {"split": "train", "row": 9} == gallery.find_identical("  " + sha + "\n")


def test_find_identical_ignores_the_case_of_the_stored_hashes_too(copy_of):
    root = copy_of()
    train, test = pd.read_parquet(root / "img_meta.parquet"), pd.read_parquet(root / "test_meta.parquet")
    lower_train, lower_test = train["file_sha256"][4], test["file_sha256"][6]
    train.loc[4, "file_sha256"] = lower_train.upper()
    test.loc[6, "file_sha256"] = " " + lower_test.upper() + " "
    train.to_parquet(root / "img_meta.parquet", index=False)
    test.to_parquet(root / "test_meta.parquet", index=False)
    g = Gallery.open(root, None)
    assert g.find_identical(lower_train) == {"split": "train", "row": 4} and g.find_identical(lower_test) == {"split": "test", "row": 6}


def test_find_identical_gives_the_first_row_when_a_file_is_in_the_gallery_twice(copy_of):
    root = copy_of()
    meta = pd.read_parquet(root / "img_meta.parquet")
    sha = meta["file_sha256"][3]
    meta.loc[5, "file_sha256"] = sha
    meta.loc[8, "file_sha256"] = sha
    meta.to_parquet(root / "img_meta.parquet", index=False)
    assert Gallery.open(root, None).find_identical(sha) == {"split": "train", "row": 3}


def test_find_identical_looks_in_the_train_split_before_the_test_split(copy_of):
    root = copy_of()
    train, test = pd.read_parquet(root / "img_meta.parquet"), pd.read_parquet(root / "test_meta.parquet")
    test.loc[2, "file_sha256"] = train["file_sha256"][11]
    test.to_parquet(root / "test_meta.parquet", index=False)
    assert Gallery.open(root, None).find_identical(train["file_sha256"][11]) == {"split": "train", "row": 11}


def test_image_path_is_the_path_of_the_jpeg_of_that_gallery_row(gallery, base, counts):
    from PIL import Image
    meta = pd.read_parquet(base / "img_meta.parquet")
    for row in (0, 17, counts["images"] - 1):
        path = gallery.image_path(row)
        assert isinstance(path, Path) and path == Path(meta["image"][row]) and path.is_file()
        with Image.open(path) as im:
            assert im.size == (320, 320)
    assert gallery.image_path(np.int64(4)) == Path(meta["image"][4])


def test_image_path_refuses_a_row_outside_the_gallery_and_one_that_is_not_an_integer(gallery, counts):
    for bad in (-1, counts["images"], counts["images"] + 1, 10 ** 12):
        with pytest.raises(IndexError) as err:
            gallery.image_path(bad)
        assert str(counts["images"] - 1) in str(err.value)
    for bad in (2.0, "2", None, [1]):
        with pytest.raises(TypeError):
            gallery.image_path(bad)


def test_test_study_returns_the_image_the_study_id_and_the_reference(gallery, base, counts):
    meta = pd.read_parquet(base / "test_meta.parquet")
    lines = (base / "report_texts.txt").read_text().splitlines()
    for row in (0, 1, 13, counts["test"] - 1):
        study = gallery.test_study(row)
        assert {"image", "study_id", "reference"} <= set(study)
        assert study["image"] == Path(meta["image"][row]) and study["image"].is_file()
        assert study["study_id"] == meta["study_id"][row]
        assert study["reference"] == lines[counts["images"] + row], "the report that follows the train reports, in test.parquet order"


def test_a_test_study_reference_is_the_reference_text_of_that_test_row_not_a_train_report(gallery, base, counts):
    split, split_row = np.load(str(base / "txt_split.npy")), np.load(str(base / "txt_split_row.npy"))
    for row in range(counts["test"]):
        report_row = counts["images"] + row
        assert split[report_row] == 1 and split_row[report_row] == row
        assert gallery.test_study(row)["reference"] == gallery.report_texts[report_row]


def test_test_study_refuses_a_row_outside_the_split_and_one_that_is_not_an_integer(gallery, counts):
    for bad in (-1, counts["test"], 10 ** 9):
        with pytest.raises(IndexError) as err:
            gallery.test_study(bad)
        assert str(counts["test"] - 1) in str(err.value)
    for bad in (0.0, "0", None):
        with pytest.raises(TypeError):
            gallery.test_study(bad)


def test_list_test_studies_lists_the_split_in_row_order_with_what_a_picker_needs(gallery, base, counts):
    meta = pd.read_parquet(base / "test_meta.parquet")
    listed = gallery.list_test_studies()
    assert counts["test"] <= 50 and [s["test_row"] for s in listed] == list(range(counts["test"]))
    assert [s["study_id"] for s in listed] == meta["study_id"].tolist() and [s["view"] for s in listed] == meta["view"].tolist()
    assert all(set(s) == {"test_row", "study_id", "view"} for s in listed)
    json.dumps(listed)


def test_list_test_studies_filters_on_the_study_id_prefix(gallery, base):
    ids = [str(s) for s in pd.read_parquet(base / "test_meta.parquet")["study_id"]]
    prefix = ids[13][:-1]
    expected = [row for row, sid in enumerate(ids) if sid.startswith(prefix)]
    assert 13 in expected and 1 < len(expected) < len(ids), "a prefix that keeps some studies and drops others"
    assert [s["test_row"] for s in gallery.list_test_studies(prefix)] == expected
    assert [s["test_row"] for s in gallery.list_test_studies(ids[13])] == [13]
    assert [s["test_row"] for s in gallery.list_test_studies("  " + prefix + " ")] == expected, "surrounding space is not part of the prefix"
    assert [s["test_row"] for s in gallery.list_test_studies(query=prefix)] == expected
    assert gallery.list_test_studies(ids[13] + "0") == [] and gallery.list_test_studies("1") == [] and gallery.list_test_studies("abc") == []
    assert gallery.list_test_studies(ids[13][1:]) == [], "a prefix, not a substring"
    assert gallery.list_test_studies(None) == gallery.list_test_studies("") and len(gallery.list_test_studies(None)) == len(ids), "no query: all"


def test_list_test_studies_stops_at_the_limit_and_a_limit_of_nothing_lists_nothing(gallery, counts):
    assert [s["test_row"] for s in gallery.list_test_studies("", 5)] == [0, 1, 2, 3, 4]
    assert [s["test_row"] for s in gallery.list_test_studies(limit=1)] == [0]
    assert gallery.list_test_studies("", 0) == [] and gallery.list_test_studies("", -3) == []
    assert len(gallery.list_test_studies("", 10 ** 6)) == counts["test"]


# ── R7: nothing is logged; the shape is the redactor's ────────────────────────────────────────────────────────────────────────────────

def test_open_and_every_query_log_nothing(base, queries, caplog):
    with caplog.at_level(logging.DEBUG):
        g = Gallery.open(base, None)
        g.image_neighbors(queries[0], 4)
        g.report_matches(queries[0], 3)
        g.own_report_rank(queries[0], 1)
        g.find_identical("0" * 64)
        g.test_study(0)
        g.list_test_studies("5")
        g.image_path(0)
        with pytest.raises(ValueError):
            g.image_neighbors(np.ones(3), 2)
    assert caplog.records == []


def test_every_result_is_plain_json_so_that_a_stage_can_store_and_send_it(gallery, queries):
    for q in queries[:4]:
        for found in (gallery.image_neighbors(q, 12), gallery.report_matches(q, 10), [gallery.own_report_rank(q, 3)],
                      gallery.list_test_studies(), [gallery.find_identical("0" * 64)]):
            assert json.loads(json.dumps(found)) == found
    for n in gallery.image_neighbors(queries[0], 12):
        assert type(n["rank"]) is int and type(n["gallery_row"]) is int and type(n["txt_row"]) is int and type(n["similarity"]) is float
        assert type(n["image_url"]) is str and all(type(v) is int for v in n["labels"].values())
    for m in gallery.report_matches(queries[0], 10):
        assert type(m["rank"]) is int and type(m["group"]) is int and type(m["group_size"]) is int and type(m["txt_row"]) is int
        assert type(m["similarity"]) is float and type(m["report"]) is str


def test_a_public_event_built_from_the_galleries_output_keeps_rank_and_similarity_only(gallery, base, queries):
    """R1 end to end on the real shapes: study ids, urls, report text, rows and groups are what app/redact.py has to cut."""
    from app.redact import redact_event
    detail = {"image_neighbors": gallery.image_neighbors(queries[2], 4), "report_matches": gallery.report_matches(queries[2], 3),
              "true_report_rank": gallery.own_report_rank(queries[2], 2), "gallery": {"build_id": gallery.build_id}}
    private = json.dumps(detail)
    public = redact_event("stage_end", {"stage": "retrieve", "ms": 1.0, "detail": detail}, "public")
    assert set(public["detail"]) >= {"image_neighbors", "report_matches"} and "true_report_rank" not in public["detail"]
    for item in public["detail"]["image_neighbors"] + public["detail"]["report_matches"]:
        assert set(item) == {"rank", "similarity"}
    text = json.dumps(public)
    for secret in [str(n["study_id"]) for n in detail["image_neighbors"]] + [m["report"] for m in detail["report_matches"]] + \
            [n["image_url"] for n in detail["image_neighbors"]]:
        assert secret in private and secret not in text


def test_importing_the_gallery_pulls_in_no_model_code():
    code = "import sys; import app.gallery; print(sorted(m for m in ('torch', 'transformers', 'datasets', 'hybrid_xmamba', 'scripts') if m in sys.modules))"
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT))
    out = subprocess.run([sys.executable, "-c", code], cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]", out.stdout + out.stderr
