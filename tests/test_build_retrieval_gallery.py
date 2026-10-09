"""P5-B (CHAT_UI_PLAN.md): the retrieval gallery builder (scripts/build_retrieval_gallery.py) and its wrapper
(scripts/build_retrieval_gallery_h100.sh).

Laptop work: CPU, offline, synthetic data only (R7). The real build() needs the 13D checkpoint, the report model and MIMIC, so it
cannot run here. What can run is tested in layers:
  * the pure pieces: the report text, the group layout, the provenance, the manifest, --compare-rk, the ISBI cross-check;
  * the synthetic gallery (--tiny), which goes through the same writers as the real build;
  * build() itself, end to end, with fakes behind the REAL signatures of the reference script's loaders (a call that would not
    bind to evaluate_cxr_retrieval's own functions fails the test), so the wiring runs if the models cannot;
  * the wrapper, rehearsed in a temp tree with a stub python (the real --compare-rk runs for real inside it).
"""
import ast
import hashlib
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pytest

from scripts import build_retrieval_gallery as bg
from tests.test_chat_remote import STAMP_RE, Sandbox

REPO_ROOT = Path(__file__).resolve().parent.parent
BASH = "/bin/bash" if os.path.exists("/bin/bash") else "bash"     # the Mac's /bin/bash is 3.2: the oldest shell to support


@pytest.fixture(scope="module")
def ref():
    """scripts/evaluate_cxr_retrieval.py, the retrieval chapter's own loaders (a few seconds to import: datasets, transformers)."""
    pytest.importorskip("datasets")
    from scripts import evaluate_cxr_retrieval
    return evaluate_cxr_retrieval


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def load(out: Path, name: str) -> np.ndarray:
    return np.load(str(out / name))


# ── report text and group layout ──────────────────────────────────────────────

ROWS = [
    {"findings": "Heart size is normal.", "impression": "No acute disease."},
    {"findings": "", "impression": "Small effusion."},                          # empty findings
    {"findings": "Clear lungs.", "impression": ""},                             # empty impression
    {"findings": "", "impression": ""},
    {"findings": "Line one.\nLine two.\r\n\tLine  three.", "impression": " padded   and   odd whitespace "},
    {"findings": None, "impression": "x"},                                      # a None prints as the published refs.txt printed it
    {"impression": "no findings key at all"},                                   # row.get("findings", "") gives ""
]


def test_report_text_is_the_published_reference_string_with_the_published_whitespace_collapse():
    row = {"findings": "Heart  size\nnormal. ", "impression": "\tNo acute  disease.\n"}
    assert bg.report_text(row) == "Findings: Heart size normal. Impression: No acute disease."


def test_report_text_equals_the_line_the_published_scripts_write_to_refs_txt(tmp_path):
    """run_checkpoint_inspection builds the reference with an f-string (no `or ""`), write_hyps_refs collapses its whitespace."""
    from scripts.evaluate_report_generation import write_hyps_refs
    published = ["Findings: {} Impression: {}".format(r.get("findings", ""), r.get("impression", "")).strip() for r in ROWS]
    write_hyps_refs(str(tmp_path), ["hyp"] * len(ROWS), published)
    refs = (tmp_path / "refs.txt").read_text().splitlines()
    assert len(refs) == len(ROWS)
    assert [bg.report_text(r) for r in ROWS] == refs
    assert "Findings: None Impression: x" in refs, "a missing value reads as the published refs.txt read it"


def test_report_text_reads_a_pandas_row_like_a_dict():
    row = pd.Series({"findings": "a  b", "impression": "c\nd", "study_id": 1})
    assert bg.report_text(row) == "Findings: a b Impression: c d"


def test_group_layout_sorts_stably_by_group_and_marks_where_each_group_begins():
    groups = np.array([2, 0, 2, 1, 0, 2, 3], dtype=np.int64)
    order, starts = bg.group_layout(groups)
    assert order.dtype == np.int64 and starts.dtype == np.int64
    assert order.tolist() == [1, 4, 3, 0, 2, 5, 6]          # sorted ids 0 0 1 2 2 2 3, rows of a group in their own order
    assert starts.tolist() == [0, 2, 3, 6]


def assert_starts_partition_order(groups: np.ndarray, order: np.ndarray, starts: np.ndarray) -> None:
    """The invariants app/gallery.py reads: np.maximum.reduceat(sims[order], starts) is one value per group, and group g is the
    g-th segment (ids are 0..G-1, each present)."""
    n = len(groups)
    assert sorted(order.tolist()) == list(range(n)), "order is a permutation of the rows"
    assert starts[0] == 0 and (np.diff(starts) > 0).all() and starts[-1] < n
    bounds = starts.tolist() + [n]
    for g in range(len(starts)):
        segment = groups[order[bounds[g]:bounds[g + 1]]]
        assert (segment == g).all(), "segment {} holds exactly the rows of group {}".format(g, g)
    assert len(starts) == groups.max() + 1 == len(np.unique(groups))


def test_group_layout_partitions_the_sorted_rows_for_random_groups():
    rng = np.random.default_rng(0)
    for n, g in ((1, 1), (7, 3), (300, 90), (500, 500)):
        ids = rng.permutation(np.r_[np.arange(g), rng.integers(0, g, n - g)]).astype(np.int64)   # ids 0..g-1, each present
        order, starts = bg.group_layout(ids)
        assert_starts_partition_order(ids, order, starts)


def test_group_layout_of_nothing_is_empty():
    order, starts = bg.group_layout(np.zeros(0, dtype=np.int64))
    assert order.shape == (0,) and starts.shape == (0,) and starts.dtype == np.int64


# ── provenance and manifest ───────────────────────────────────────────────────

def test_towers_are_identical_only_when_the_hashes_match_and_there_is_no_img_proj():
    same = {"tower_sha256": "a" * 64, "decoder_tower_sha256": "a" * 64}
    other = {"tower_sha256": "a" * 64, "decoder_tower_sha256": "b" * 64}
    assert bg.towers_identical(same, img_proj_present=False) is True
    assert bg.towers_identical(same, img_proj_present=True) is False
    assert bg.towers_identical(other, img_proj_present=False) is False


SHA = "ab" * 20
NOW = datetime(2026, 10, 9, 20, 5, 3, tzinfo=timezone.utc)


def make_args(tmp_path: Path, **over) -> SimpleNamespace:
    ck13, dec = tmp_path / "c13.ckpt", tmp_path / "dec.ckpt"
    ck13.write_bytes(b"13d weights")
    dec.write_bytes(b"decoder weights")
    args = dict(checkpoint_13d=str(ck13), decoder_checkpoint=str(dec), decoder_config="hybrid_150m_m3_rrg", data=str(tmp_path / "data"),
                out=str(tmp_path / "gallery" / "20261009_77"), build_id=None, batch_size=32, workers=2, isbi_cache=None)
    args.update(over)
    return SimpleNamespace(**args)


def stamped_tree(tmp_path: Path, flag: str = "dirty") -> Path:
    root = tmp_path / "tree"
    root.mkdir()
    (root / ".sync_stamp").write_text("2026-10-09T20:00:00Z {} {}\n".format(SHA, flag))
    return root


def test_provenance_records_build_id_time_the_synced_commit_job_and_both_checkpoints(tmp_path):
    args = make_args(tmp_path)
    prov = bg.provenance(args, root=stamped_tree(tmp_path), env={"SLURM_JOB_ID": "77", "SLURM_RESTART_COUNT": "1"}, now=NOW)
    assert prov == {
        "build_id": "20261009_77", "created": "2026-10-09T20:05:03Z",
        "git_sha": SHA, "git_dirty": True, "git_source": "sync_stamp",
        "job_id": "77", "restart_count": 1,
        "checkpoint_13d": args.checkpoint_13d, "checkpoint_13d_sha256": hashlib.sha256(b"13d weights").hexdigest(),
        "decoder_checkpoint": args.decoder_checkpoint, "decoder_checkpoint_sha256": hashlib.sha256(b"decoder weights").hexdigest(),
        "decoder_config": "hybrid_150m_m3_rrg"}


def test_provenance_takes_the_commit_and_the_flag_from_the_sync_stamp(tmp_path):
    for flag in ("clean", "dirty"):
        base = tmp_path / flag
        base.mkdir()
        prov = bg.provenance(make_args(base), root=stamped_tree(base, flag), env={}, now=NOW)
        assert (prov["git_sha"], prov["git_dirty"], prov["git_source"]) == (SHA, flag == "dirty", "sync_stamp")


def test_provenance_names_the_build_by_the_flag_else_by_its_directory_and_has_no_job_off_the_cluster(tmp_path):
    args = make_args(tmp_path, build_id="named")
    prov = bg.provenance(args, root=tmp_path, env={}, now=NOW)
    assert prov["build_id"] == "named" and prov["job_id"] is None and prov["restart_count"] is None
    assert bg.provenance(make_args(tmp_path), root=tmp_path, env={}, now=NOW)["build_id"] == "20261009_77"
    # an unstamped tree without git says so instead of inventing a commit
    assert (prov["git_sha"], prov["git_dirty"], prov["git_source"]) == (None, None, None)


def test_the_manifest_has_every_key_of_the_contract_and_derives_towers_identical(tmp_path):
    prov = bg.provenance(make_args(tmp_path), root=stamped_tree(tmp_path), env={"SLURM_JOB_ID": "77"}, now=NOW)
    hashes = {"tower_sha256": "c" * 64, "decoder_tower_sha256": "c" * 64}
    counts = {"images": 5, "report_rows": 8, "report_groups": 6, "test": 3}
    gate = {"app": {"i2t_R@1": 0.5}}
    transform = bg.transform_facts([0.5, 0.5, 0.5], [0.25, 0.25, 0.25], 224)
    manifest = bg.assemble_manifest(hashes, False, counts, gate, prov, transform)
    for key in ("build_id", "created", "git_sha", "job_id", "checkpoint_13d", "checkpoint_13d_sha256", "decoder_checkpoint",
                "decoder_checkpoint_sha256", "tower_sha256", "decoder_tower_sha256", "towers_identical", "img_proj_present",
                "counts", "transform", "tokenizer", "labels_status", "gate_rk"):
        assert key in manifest, key
    assert manifest["towers_identical"] is True and manifest["img_proj_present"] is False
    assert manifest["labels_status"] == "pending", "P5-C writes the labels"
    assert manifest["counts"] == counts and manifest["gate_rk"] == gate
    assert manifest["tokenizer"] == {"name": "gpt2", "max_length": 256, "padding": "max_length", "truncation": True,
                                     "padding_side": "right", "pad_token": "eos"}
    assert manifest["transform"]["mean"] == [0.5, 0.5, 0.5] and "CenterCrop(224)" in manifest["transform"]["pipeline"]
    json.dumps(manifest)                                                         # serialisable as it is
    other = bg.assemble_manifest({"tower_sha256": "c" * 64, "decoder_tower_sha256": "d" * 64}, False, counts, gate, prov, transform)
    assert other["towers_identical"] is False
    assert bg.assemble_manifest(hashes, True, counts, gate, prov, transform)["towers_identical"] is False
    assert bg.assemble_manifest(hashes, False, counts, gate, prov, transform, extra={"synthetic": True})["synthetic"] is True


def test_the_manifest_is_written_whole_or_not_at_all(tmp_path):
    """manifest.json is the wrapper's 'finished build' marker: a half-written one would block the requeue that has to redo it."""
    target = tmp_path / "manifest.json"
    bg.write_json_atomic(target, {"a": 1})
    assert json.loads(target.read_text()) == {"a": 1}
    bg.write_json_atomic(target, {"a": 2})
    assert json.loads(target.read_text()) == {"a": 2}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["manifest.json"], "no temp file is left behind"


# ── --compare-rk ──────────────────────────────────────────────────────────────

METRICS = {"i2t_R@1": 0.0415, "i2t_R@5": 0.1042, "i2t_R@10": 0.1714, "t2i_R@1": 0.0388, "t2i_R@5": 0.0991, "t2i_R@10": 0.1626,
           "mean_R@10": 0.167, "N": 2663}


def write_gate(out: Path, app: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "gate_rk.json").write_text(json.dumps({"app": app}))


def write_reference(out: Path, metrics: dict, stamp: str = "20261009_101500") -> Path:
    directory = out / "reference_rk"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "phase6_mimic_{}.json".format(stamp)
    path.write_text(json.dumps({"timestamp": stamp, "dataset": "mimic", "checkpoint": "x.ckpt", "n_samples": metrics.get("N"),
                                "metrics": metrics, "embedding_stats": {}}))
    return path


def result_lines(text: str) -> List[str]:
    return [line for line in text.splitlines() if line.startswith("RESULT ")]


def test_compare_rk_returns_0_when_every_recall_is_equal_and_records_the_verdict(tmp_path, capsys):
    out = tmp_path / "g"
    write_gate(out, METRICS)
    write_reference(out, dict(METRICS))
    assert bg.main(["--compare-rk", str(out)]) == 0
    gate = json.loads((out / "gate_rk.json").read_text())
    assert gate["app"] == METRICS and gate["reference"] == METRICS and gate["equal"] is True
    assert gate["reference_file"] == "phase6_mimic_20261009_101500.json"
    (line,) = result_lines(capsys.readouterr().out)
    payload = json.loads(line[len("RESULT "):])
    assert payload["gate_rk_equal"] is True and payload["n"] == 2663 and payload["i2t_R@10"] == 0.1714 and "differs" not in payload


def test_compare_rk_returns_1_on_a_doctored_reference_and_names_what_differs(tmp_path, capsys):
    out = tmp_path / "g"
    write_gate(out, METRICS)
    write_reference(out, dict(METRICS, **{"i2t_R@10": 0.1715}))
    assert bg.main(["--compare-rk", str(out)]) == 1
    assert json.loads((out / "gate_rk.json").read_text())["equal"] is False
    payload = json.loads(result_lines(capsys.readouterr().out)[0][len("RESULT "):])
    assert payload["gate_rk_equal"] is False and payload["differs"] == ["i2t_R@10"]


@pytest.mark.parametrize("key", ["i2t_R@1", "i2t_R@5", "i2t_R@10", "t2i_R@1", "t2i_R@5", "t2i_R@10"])
def test_each_of_the_six_recalls_decides_the_gate(tmp_path, key):
    out = tmp_path / "g"
    write_gate(out, METRICS)
    write_reference(out, dict(METRICS, **{key: METRICS[key] + 1e-12}))          # equal to every digit, not to a tolerance
    assert bg.main(["--compare-rk", str(out)]) == 1


def test_a_different_number_of_studies_is_not_the_same_evaluation(tmp_path):
    out = tmp_path / "g"
    write_gate(out, METRICS)
    write_reference(out, dict(METRICS, N=2662))
    assert bg.main(["--compare-rk", str(out)]) == 1


def test_compare_rk_reads_the_newest_reference_file(tmp_path):
    out = tmp_path / "g"
    write_gate(out, METRICS)
    write_reference(out, dict(METRICS, **{"t2i_R@5": 0.5}), stamp="20261009_090000")      # an older, different run
    write_reference(out, dict(METRICS), stamp="20261009_101500")
    assert bg.main(["--compare-rk", str(out)]) == 0
    write_reference(out, dict(METRICS, **{"t2i_R@5": 0.5}), stamp="20261009_111500")      # a newer one that differs
    assert bg.main(["--compare-rk", str(out)]) == 1


def test_compare_rk_without_something_to_compare_is_an_error_not_a_verdict(tmp_path, capsys):
    out = tmp_path / "g"
    assert bg.main(["--compare-rk", str(out)]) == 2                                  # no gate_rk.json
    write_gate(out, METRICS)
    assert bg.main(["--compare-rk", str(out)]) == 2                                  # no reference
    write_reference(out, {k: v for k, v in METRICS.items() if k != "i2t_R@5"})
    assert bg.main(["--compare-rk", str(out)]) == 2                                  # a recall missing from the reference
    assert "equal" not in json.loads((out / "gate_rk.json").read_text()), "no verdict was recorded"
    lines = capsys.readouterr().out.splitlines()
    assert lines and all(line.startswith("ERROR") for line in lines)
    assert not [line for line in lines if "/" in line or str(tmp_path) in line], "R7: no path in an ERROR line"


def test_compare_rk_carries_the_verdict_into_the_manifest_when_there_is_one(tmp_path):
    out = tmp_path / "g"
    write_gate(out, METRICS)
    (out / "manifest.json").write_text(json.dumps({"build_id": "x", "gate_rk": {"app": METRICS}}))
    write_reference(out, dict(METRICS))
    assert bg.main(["--compare-rk", str(out)]) == 0
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["build_id"] == "x" and manifest["gate_rk"]["equal"] is True and manifest["gate_rk"]["reference"] == METRICS


def test_compare_rk_prints_one_short_result_line_of_numbers_only(tmp_path, capsys):
    out = tmp_path / "g"
    write_gate(out, METRICS)
    write_reference(out, dict(METRICS))
    assert bg.main(["--compare-rk", str(out), "--wall-s", "3333"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1 and lines[0].startswith("RESULT {") and len(lines[0]) < 300 and "/" not in lines[0]
    assert json.loads(lines[0][len("RESULT "):])["wall_s"] == 3333


# ── the arguments ─────────────────────────────────────────────────────────────

def test_the_build_needs_its_four_inputs_and_the_modes_are_separate(capsys):
    with pytest.raises(SystemExit):
        bg.parse_args([])
    with pytest.raises(SystemExit):
        bg.parse_args(["--checkpoint-13d", "a", "--decoder-checkpoint", "b", "--data", "c"])          # no --out
    with pytest.raises(SystemExit):
        bg.parse_args(["--tiny", "x", "--compare-rk", "y"])
    args = bg.parse_args(["--checkpoint-13d", "a", "--decoder-checkpoint", "b", "--data", "c", "--out", "d"])
    assert (args.decoder_config, args.batch_size, args.build_id, args.isbi_cache) == ("hybrid_150m_m3_rrg", 32, None, None)
    assert bg.parse_args(["--tiny", "x"]).tiny == "x" and bg.parse_args(["--compare-rk", "y"]).compare_rk == "y"


def test_the_defaults_are_the_reference_scripts_so_that_the_vectors_are_the_chapters(ref):
    """build_dataloader and encode_dataset are the reference's, and its main() calls them at its own --batch-size and --max-length
    defaults: the builder's must be the same numbers, or the gate would compare two different evaluations."""
    source = (REPO_ROOT / "scripts" / "evaluate_cxr_retrieval.py").read_text()
    batch = re.search(r'"--batch-size",\s*type=int,\s*default=(\d+)', source)
    length = re.search(r'"--max-length",\s*type=int,\s*default=(\d+)', source)
    assert batch and length, "the reference script's argument list moved: update this test"
    assert bg.BATCH_SIZE == int(batch.group(1)) == 32 and bg.MAX_LENGTH == int(length.group(1)) == 256
    assert bg.parse_args(["--tiny", "x"]).batch_size == bg.BATCH_SIZE


# ── the ISBI cross-check (step 8 of the brief) ────────────────────────────────

def _save_isbi(path: Path, emb16: np.ndarray, encoder: str = "adapted", n: Optional[int] = None) -> Path:
    """What scripts/isbi_figure_cases.py caches: torch.save({"encoder", "n", "emb": gallery.half()})."""
    import torch
    torch.save({"encoder": encoder, "n": len(emb16) if n is None else n, "emb": torch.from_numpy(emb16)}, str(path))
    return path


def test_isbi_cross_check_reports_the_largest_difference_to_the_figure_jobs_cache(tmp_path):
    pytest.importorskip("torch")
    emb16 = (np.random.default_rng(0).random((10, 8)).astype(np.float32) - 0.5).astype(np.float16)
    isbi = emb16.astype(np.float32)
    isbi[3, 2] += 0.01
    found = bg.isbi_cross_check(emb16, _save_isbi(tmp_path / "isbi.pt", isbi.astype(np.float16)))
    assert found["status"] == "compared" and found["rows"] == 10
    assert found["max_abs_diff"] == pytest.approx(0.01, abs=1e-3)
    identical = bg.isbi_cross_check(emb16, _save_isbi(tmp_path / "same.pt", emb16))
    assert identical["status"] == "compared" and identical["max_abs_diff"] == 0.0


def test_isbi_cross_check_is_optional_and_never_fatal(tmp_path):
    pytest.importorskip("torch")
    emb16 = np.zeros((4, 2), dtype=np.float16)
    assert bg.isbi_cross_check(emb16, None) == {"status": "absent"}
    assert bg.isbi_cross_check(emb16, tmp_path / "missing.pt") == {"status": "absent"}
    assert bg.isbi_cross_check(emb16, _save_isbi(tmp_path / "stock.pt", emb16, encoder="stock"))["status"] == "mismatch"
    assert bg.isbi_cross_check(emb16, _save_isbi(tmp_path / "n.pt", emb16, n=5))["status"] == "mismatch"
    bad = tmp_path / "bad.pt"
    bad.write_bytes(b"not a torch file")
    assert bg.isbi_cross_check(emb16, bad) == {"status": "unreadable"}


# ── the synthetic gallery (--tiny) ────────────────────────────────────────────

CONTRACT_FILES = sorted([
    "img_emb.npy", "test_img_emb.npy", "txt_emb.npy", "txt_emb_test.npy", "txt_groups.npy", "group_order.npy", "group_starts.npy",
    "txt_test_groups.npy", "txt_split.npy", "txt_split_row.npy", "img_txt_row.npy", "img_meta.parquet", "test_meta.parquet",
    "report_texts.txt", "labels.npy", "label_names.json", "manifest.json", "gate_rk.json"])
N_IMG, N_TEST, DIM = 200, 40, 16
N_TXT = N_IMG + N_TEST


def test_tiny_writes_every_file_of_the_contract_and_only_its_images_besides(tiny_gallery):
    assert sorted(p.name for p in tiny_gallery.iterdir() if p.is_file()) == CONTRACT_FILES
    assert [p.name for p in tiny_gallery.iterdir() if p.is_dir()] == ["images"]


def test_tiny_arrays_have_the_contract_dtypes_and_consistent_shapes(tiny_gallery):
    expected = {"img_emb": (np.float16, (N_IMG, DIM)), "test_img_emb": (np.float32, (N_TEST, DIM)),
                "txt_emb": (np.float16, (N_TXT, DIM)), "txt_emb_test": (np.float32, (N_TEST, DIM)),
                "txt_groups": (np.int64, (N_TXT,)), "group_order": (np.int64, (N_TXT,)),
                "txt_test_groups": (np.int64, (N_TEST,)), "txt_split": (np.int8, (N_TXT,)),
                "txt_split_row": (np.int64, (N_TXT,)), "img_txt_row": (np.int64, (N_IMG,)), "labels": (np.uint8, (N_TXT, 14))}
    for name, (dtype, shape) in expected.items():
        array = load(tiny_gallery, name + ".npy")
        assert (array.dtype, array.shape) == (np.dtype(dtype), shape), name
    starts = load(tiny_gallery, "group_starts.npy")
    assert starts.dtype == np.int64 and starts.ndim == 1
    texts = (tiny_gallery / "report_texts.txt").read_text().splitlines()
    assert len(texts) == N_TXT and all(t.startswith("Findings: ") and " Impression: " in t for t in texts)
    counts = json.loads((tiny_gallery / "manifest.json").read_text())["counts"]
    assert counts == {"images": N_IMG, "report_rows": N_TXT, "report_groups": len(starts), "test": N_TEST}


def test_tiny_vectors_match_the_tiny_engines_pooled_width():
    from app.tiny import TINY_POOLED_DIM
    assert bg.TINY_DIM == TINY_POOLED_DIM == DIM, "a tiny engine's query must be able to meet the tiny gallery"


def test_tiny_split_and_row_maps_say_where_each_report_and_image_came_from(tiny_gallery):
    assert load(tiny_gallery, "txt_split.npy").tolist() == [0] * N_IMG + [1] * N_TEST
    assert load(tiny_gallery, "txt_split_row.npy").tolist() == list(range(N_IMG)) + list(range(N_TEST))
    assert load(tiny_gallery, "img_txt_row.npy").tolist() == list(range(N_IMG)), "one image and one report per train study"


def test_tiny_group_starts_partition_group_order(tiny_gallery):
    groups, order, starts = (load(tiny_gallery, n + ".npy") for n in ("txt_groups", "group_order", "group_starts"))
    assert_starts_partition_order(groups, order, starts)
    assert len(starts) == json.loads((tiny_gallery / "manifest.json").read_text())["counts"]["report_groups"] < N_TXT


def test_tiny_groups_are_the_reference_scripts_groups(tiny_gallery, ref):
    texts = (tiny_gallery / "report_texts.txt").read_text().splitlines()
    assert np.array_equal(load(tiny_gallery, "txt_groups.npy"), ref.group_ids_from_texts(texts))
    assert np.array_equal(load(tiny_gallery, "txt_test_groups.npy"), ref.group_ids_from_texts(texts[N_IMG:]))


def test_the_light_group_and_recall_functions_equal_the_reference_ones(ref):
    """--tiny must not import the reference script (3-4 s of datasets and transformers), so it carries light copies of its two pure
    functions; this is what keeps the copies the same."""
    texts = ["Findings: A. Impression: B.", "findings: a.  impression: b.", "FINDINGS: A.\nIMPRESSION: B.", "", "  ", None, "x", "X ", "y"]
    assert np.array_equal(bg._group_ids(texts), ref.group_ids_from_texts(texts))
    assert bg._normalize(" A\tb\n") == ref.normalize_report_text(" A\tb\n") == "a b"
    rng = np.random.default_rng(3)
    img, txt = unit(rng.standard_normal((50, 8))).astype(np.float32), unit(rng.standard_normal((50, 8))).astype(np.float32)
    assert bg._recall_metrics(img, txt) == ref.compute_retrieval_metrics(img, txt)
    tied = np.repeat(img[:5], 4, axis=0)                                   # exact ties: argpartition's own order must be the same call
    assert bg._recall_metrics(tied, tied) == ref.compute_retrieval_metrics(tied, tied)


def test_tiny_has_deliberate_duplicates_inside_train_inside_test_across_them_and_in_two_spellings(tiny_gallery):
    groups = load(tiny_gallery, "txt_groups.npy")
    texts = (tiny_gallery / "report_texts.txt").read_text().splitlines()
    train, test = groups[:N_IMG], groups[N_IMG:]
    assert np.bincount(train).max() >= 4, "a templated report that four train studies share"
    assert len(set(test.tolist())) < N_TEST, "two test reports that are the same report (the dedup-aware rank has something to do)"
    assert set(test.tolist()) & set(train.tolist()), "a test report that a train study repeats"
    spellings = {}
    for text, group in zip(texts, groups.tolist()):
        spellings.setdefault(group, set()).add(text)
    assert any(len(s) > 1 for s in spellings.values()), "one group whose members are spelt differently (case)"
    assert len(np.unique(groups)) < N_TXT * 0.6 and len(np.unique(groups)) > 30


def test_tiny_duplicates_share_their_text_vectors_and_their_labels_as_they_would_in_a_real_build(tiny_gallery):
    groups = load(tiny_gallery, "txt_groups.npy")
    txt, labels = load(tiny_gallery, "txt_emb.npy"), load(tiny_gallery, "labels.npy")
    for g in np.unique(groups):
        members = np.flatnonzero(groups == g)
        assert (txt[members] == txt[members[0]]).all(), "the same text gives the same vector"
        assert (labels[members] == labels[members[0]]).all(), "labels are one set per duplicate group"
    assert set(np.unique(labels).tolist()) == {0, 1}


def test_tiny_vectors_are_unit_length_and_the_test_texts_are_in_both_text_files(tiny_gallery):
    img, test_img = load(tiny_gallery, "img_emb.npy").astype(np.float32), load(tiny_gallery, "test_img_emb.npy")
    txt, txt_test = load(tiny_gallery, "txt_emb.npy").astype(np.float32), load(tiny_gallery, "txt_emb_test.npy")
    for name, array, tol in (("img", img, 2e-3), ("test_img", test_img, 1e-5), ("txt", txt, 2e-3), ("txt_test", txt_test, 1e-5)):
        assert np.abs(np.linalg.norm(array, axis=1) - 1).max() < tol, name
    assert np.abs(txt[N_IMG:] - txt_test).max() < 2e-3, "txt_emb_test is the test rows of txt_emb, kept in float32"


def test_tiny_images_retrieve_themselves_first(tiny_gallery):
    img = load(tiny_gallery, "img_emb.npy").astype(np.float32)
    assert (np.argmax(img @ img.T, axis=1) == np.arange(N_IMG)).all()


def test_tiny_gate_is_the_reference_scripts_recall_on_its_test_split(tiny_gallery, ref):
    test_img, txt_test = load(tiny_gallery, "test_img_emb.npy"), load(tiny_gallery, "txt_emb_test.npy")
    gate = json.loads((tiny_gallery / "gate_rk.json").read_text())
    assert gate == {"app": ref.compute_retrieval_metrics(test_img, txt_test)}
    assert json.loads((tiny_gallery / "manifest.json").read_text())["gate_rk"] == gate
    assert gate["app"]["N"] == N_TEST


def test_tiny_metadata_names_real_gray_jpegs_and_their_file_hashes(tiny_gallery):
    from PIL import Image
    from app.engine import file_sha256
    train, test = (pd.read_parquet(tiny_gallery / n) for n in ("img_meta.parquet", "test_meta.parquet"))
    cols = ["study_id", "subject_id", "dicom_id", "view", "image", "file_sha256"]
    assert list(train.columns) == ["row"] + cols and list(test.columns) == ["test_row"] + cols
    assert train["row"].tolist() == list(range(N_IMG)) and test["test_row"].tolist() == list(range(N_TEST))
    assert train["study_id"].is_unique and test["study_id"].is_unique
    assert not set(train["study_id"]) & set(test["study_id"]) and not set(train["subject_id"]) & set(test["subject_id"])
    paths = train["image"].tolist() + test["image"].tolist()
    hashes = train["file_sha256"].tolist() + test["file_sha256"].tolist()
    assert len(set(hashes)) == len(hashes) == N_IMG + N_TEST, "every image differs, so 'identical to a gallery image' is unambiguous"
    for path, digest in list(zip(paths, hashes))[:: 17]:
        assert os.path.isabs(path) and Path(path).parent == (tiny_gallery / "images").resolve()
        with Image.open(path) as im:
            assert (im.format, im.size, im.mode) == ("JPEG", (320, 320), "L")
        assert digest == file_sha256(Path(path)) == hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_tiny_label_names_are_the_chexbert_14_in_the_trainers_order(tiny_gallery):
    tree = ast.parse((REPO_ROOT / "scripts" / "train_report_generation.py").read_text())
    trainer = [ast.literal_eval(n.value) for n in ast.walk(tree)
               if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "CHEXPERT_14_LABELS" for t in n.targets)]
    assert len(trainer) == 1 and len(trainer[0]) == 14
    assert json.loads((tiny_gallery / "label_names.json").read_text()) == trainer[0] == bg.CHEXBERT_14


def test_tiny_manifest_says_what_it_is(tiny_gallery):
    manifest = json.loads((tiny_gallery / "manifest.json").read_text())
    assert manifest["synthetic"] is True and manifest["labels_status"] == "done" and manifest["img_proj_present"] is False
    assert manifest["towers_identical"] is True and manifest["tower_sha256"] == manifest["decoder_tower_sha256"]
    assert re.fullmatch(r"[0-9a-f]{64}", manifest["tower_sha256"])
    assert manifest["build_id"] == "tiny"


def test_tiny_is_deterministic_for_a_seed_and_differs_between_seeds(tmp_path):
    bg.build_tiny(tmp_path / "a", seed=1)
    bg.build_tiny(tmp_path / "b", seed=1)
    bg.build_tiny(tmp_path / "c", seed=2)
    names = [n for n in CONTRACT_FILES if n.endswith(".npy") or n == "report_texts.txt"]
    for name in names:
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes(), name
    a, b = (pd.read_parquet(tmp_path / d / "img_meta.parquet") for d in ("a", "b"))
    assert a["file_sha256"].tolist() == b["file_sha256"].tolist(), "the same pixels, the same JPEG bytes"
    assert (tmp_path / "a" / "txt_emb.npy").read_bytes() != (tmp_path / "c" / "txt_emb.npy").read_bytes()


def test_tiny_without_labels_leaves_them_to_p5c(tmp_path):
    manifest = bg.build_tiny(tmp_path / "g", with_labels=False)
    assert not (tmp_path / "g" / "labels.npy").exists() and not (tmp_path / "g" / "label_names.json").exists()
    assert manifest["labels_status"] == "pending"
    assert sorted(p.name for p in (tmp_path / "g").iterdir() if p.is_file()) == [
        n for n in CONTRACT_FILES if n not in ("labels.npy", "label_names.json")]


def test_tiny_from_the_command_line_is_fast_and_pulls_in_none_of_the_heavy_modules(tmp_path):
    """No model, no data, and not the 3-4 s of importing the reference script: the CLI, cold, finishes in under 5 s."""
    probe = ("import sys; sys.path.insert(0, {root!r}); from scripts import build_retrieval_gallery as bg; "
             "code = bg.main(['--tiny', {out!r}]); "
             "print('HEAVY', sorted(m for m in ('datasets', 'transformers', 'open_clip', 'torchvision', 'scripts.evaluate_cxr_retrieval', "
             "'app.engine') if m in sys.modules)); sys.exit(code)").format(root=str(REPO_ROOT), out=str(tmp_path / "g"))
    started = time.monotonic()
    done = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, cwd=str(tmp_path), timeout=60,
                          env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"})
    elapsed = time.monotonic() - started
    assert done.returncode == 0, done.stdout + done.stderr
    assert "HEAVY []" in done.stdout, done.stdout
    gallery_lines = [l for l in done.stdout.splitlines() if l.startswith("[gallery]")]
    assert gallery_lines == ["[gallery] tiny: {} images, {} report rows, {} groups, {} test".format(
        N_IMG, N_TXT, json.loads((tmp_path / "g" / "manifest.json").read_text())["counts"]["report_groups"], N_TEST)]
    assert elapsed < 5.0, "--tiny took {:.1f} s".format(elapsed)


# ── build(), end to end, with fakes behind the reference script's real signatures ──

GALLERY_SHAPES = re.compile(
    r"^\[gallery\] ("
    r"device=(cuda|cpu)"
    r"|towers_identical=(True|False) img_proj=(True|False)"
    r"|tower_sha256=[0-9a-f]{64} decoder_tower_sha256=[0-9a-f]{64}"
    r"|(test|train): \d+ rows"
    r"|gate_rk app:( (i2t|t2i)_R@(1|5|10)=[0-9.]+){6} n=\d+"
    r"|norms: img_mean=[0-9.]+ txt_mean=[0-9.]+"
    r"|texts: \d+ rows, \d+ groups"
    r"|hashed \d+ image files"
    r"|isbi cross-check: status=(absent|mismatch|unreadable|compared rows=\d+ max_abs_diff=[0-9.e+-]+)"
    r"|manifest written, labels_status=pending"
    r")$")


def checked(real, fake, calls: list, name: str):
    """`fake` behind the REAL function's signature: a call that would not bind to the real one fails here, so build() cannot drift
    from the reference script unseen. Every call is recorded."""
    signature = inspect.signature(real)

    def call(*args, **kwargs):
        signature.bind(*args, **kwargs)
        calls.append((name, args, kwargs))
        return fake(*args, **kwargs)
    return call


class World:
    """Everything build() needs on the laptop: a dataset directory (the tiny generator's frames and JPEGs), two small
    'checkpoints' (only hashed), and fakes behind the reference loaders' real signatures."""
    N_TRAIN, N_TEST, WIDTH = 30, 10, 8

    def __init__(self, root: Path, ref, decoder_differs: bool = False, img_proj=None, train_rows_in_loader: Optional[int] = None):
        import torch
        from app.engine import file_sha256, tensor_sha256
        from scripts.evaluate_report_generation import load_report_generation_module
        self.root, self.ref, self.calls = root, ref, []
        self.frames, self.pick = bg.tiny_frames(root / "images", seed=5, n_images=self.N_TRAIN, n_test=self.N_TEST)
        self.data = root / "data"
        self.data.mkdir()
        for split in ("train", "test"):
            self.frames[split].to_parquet(self.data / "{}.parquet".format(split), index=False)
        rng = np.random.default_rng(11)
        self.emb = {split: (unit(rng.standard_normal((n, self.WIDTH))).astype(np.float32),
                            unit(rng.standard_normal((n, self.WIDTH))).astype(np.float32))
                    for split, n in (("train", self.N_TRAIN), ("test", self.N_TEST))}
        self.args = make_args(root, data=str(self.data), out=str(root / "gallery" / "20261009_77"), workers=3)
        self.text_enc, self.img_proj = object(), img_proj
        with torch.random.fork_rng(devices=[]):                 # the global random state belongs to the other tests
            torch.manual_seed(0)
            self.tower = torch.nn.Linear(4, 4)
            self.decoder_tower = torch.nn.Linear(4, 4)
        if not decoder_differs:
            self.decoder_tower.load_state_dict(self.tower.state_dict())
        rows = {"train": train_rows_in_loader if train_rows_in_loader is not None else self.N_TRAIN, "test": self.N_TEST}

        def fake_dataloader(dataset_name, cache_dir, tokenizer, max_length=256, batch_size=32, num_workers=4, local_parquet_dir=None,
                            mimic_split="test"):
            assert (dataset_name, tokenizer) == ("mimic", "TOKENIZER")
            return mimic_split, rows[mimic_split]

        def fake_encode(loader, text_enc, img_proj, image_enc, device):
            assert (text_enc, image_enc) == (self.text_enc, self.tower) and img_proj is self.img_proj
            return self.emb[loader]

        c = self.calls
        self.deps = SimpleNamespace(
            device=lambda: "cpu",
            load_models=checked(ref.load_models, lambda ckpt, device: (self.text_enc, self.img_proj, self.tower), c, "load_models"),
            load_decoder=checked(load_report_generation_module,
                                 lambda ckpt, config, device="cpu": SimpleNamespace(image_encoder=self.decoder_tower), c, "load_decoder"),
            make_tokenizer=lambda: "TOKENIZER",
            build_dataloader=checked(ref.build_dataloader, fake_dataloader, c, "build_dataloader"),
            encode_dataset=checked(ref.encode_dataset, fake_encode, c, "encode_dataset"),
            group_ids_from_texts=checked(ref.group_ids_from_texts, ref.group_ids_from_texts, c, "group_ids_from_texts"),
            compute_retrieval_metrics=checked(ref.compute_retrieval_metrics, ref.compute_retrieval_metrics, c, "compute_retrieval_metrics"),
            tensor_sha256=tensor_sha256, file_sha256=file_sha256,
            provenance=partial(bg.provenance, root=stamped_tree(root), env={"SLURM_JOB_ID": "77"}, now=NOW),
            transform=bg.transform_facts(ref.IMAGE_MEAN, ref.IMAGE_STD, ref.IMAGE_SIZE))
        self.out = Path(self.args.out)

    def build(self) -> dict:
        return bg.build(self.args, self.deps)

    def published_refs(self, split: str) -> List[str]:
        """The brief's expression: run_checkpoint_inspection's f-string, then write_hyps_refs's collapse."""
        return [" ".join("Findings: {} Impression: {}".format(r.get("findings", ""), r.get("impression", "")).strip().split())
                for _, r in self.frames[split].iterrows()]


def test_build_writes_the_gallery_through_the_reference_loaders_at_their_own_defaults(tmp_path, ref, capsys):
    w = World(tmp_path, ref)
    manifest = w.build()
    out, n_tr, n_te = w.out, w.N_TRAIN, w.N_TEST
    # the calls: both models, then the test split before the train split, each through build_dataloader and encode_dataset
    names = [c[0] for c in w.calls]
    assert names[:2] == ["load_models", "load_decoder"]
    loaders = [c for c in w.calls if c[0] == "build_dataloader"]
    assert [c[2]["mimic_split"] for c in loaders] == ["test", "train"]
    for _, args, kwargs in loaders:
        assert args == ("mimic", "", "TOKENIZER")
        assert kwargs == {"max_length": 256, "batch_size": 32, "num_workers": 3, "local_parquet_dir": str(w.data),
                          "mimic_split": kwargs["mimic_split"]}
    assert w.calls[0][1] == (w.args.checkpoint_13d, "cpu")
    assert w.calls[1][1] == (w.args.decoder_checkpoint, "hybrid_150m_m3_rrg") and w.calls[1][2] == {"device": "cpu"}
    # the vectors, in the contract's dtypes: train and test images, and the train rows then the test rows of the text encoder
    tr_img, tr_txt = w.emb["train"]
    te_img, te_txt = w.emb["test"]
    assert np.array_equal(load(out, "img_emb.npy"), tr_img.astype(np.float16)) and load(out, "img_emb.npy").dtype == np.float16
    assert np.array_equal(load(out, "txt_emb.npy"), np.concatenate([tr_txt, te_txt]).astype(np.float16))
    assert np.array_equal(load(out, "test_img_emb.npy"), te_img) and load(out, "test_img_emb.npy").dtype == np.float32
    assert np.array_equal(load(out, "txt_emb_test.npy"), te_txt) and load(out, "txt_emb_test.npy").dtype == np.float32
    # the gate is the reference's own function on the test split
    gate = json.loads((out / "gate_rk.json").read_text())
    assert gate == {"app": ref.compute_retrieval_metrics(te_img, te_txt)} and manifest["gate_rk"] == gate
    # the texts: train rows then test rows, the published reference string, and the layout over them
    texts = w.published_refs("train") + w.published_refs("test")
    assert (out / "report_texts.txt").read_text() == "\n".join(texts) + "\n"
    groups = ref.group_ids_from_texts(texts)
    assert np.array_equal(load(out, "txt_groups.npy"), groups)
    assert np.array_equal(load(out, "txt_test_groups.npy"), ref.group_ids_from_texts(texts[n_tr:]))
    assert_starts_partition_order(groups, load(out, "group_order.npy"), load(out, "group_starts.npy"))
    assert load(out, "txt_split.npy").tolist() == [0] * n_tr + [1] * n_te
    assert load(out, "txt_split_row.npy").tolist() == list(range(n_tr)) + list(range(n_te))
    assert load(out, "img_txt_row.npy").tolist() == list(range(n_tr))
    # the metadata, with the hash of every image file
    for split, name, key, count in (("train", "img_meta.parquet", "row", n_tr), ("test", "test_meta.parquet", "test_row", n_te)):
        meta = pd.read_parquet(out / name)
        assert list(meta.columns) == [key, "study_id", "subject_id", "dicom_id", "view", "image", "file_sha256"]
        assert meta[key].tolist() == list(range(count))
        for column in ("study_id", "subject_id", "dicom_id", "view", "image"):
            assert meta[column].tolist() == w.frames[split][column].tolist()
        assert meta["file_sha256"].tolist() == [hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in meta["image"]]
    # no labels yet, and what the manifest says
    assert not (out / "labels.npy").exists()
    assert json.loads((out / "manifest.json").read_text()) == json.loads(json.dumps(manifest))
    assert manifest["counts"] == {"images": n_tr, "report_rows": n_tr + n_te, "report_groups": int(groups.max() + 1), "test": n_te}
    assert manifest["labels_status"] == "pending" and manifest["towers_identical"] is True and manifest["img_proj_present"] is False
    assert manifest["tower_sha256"] == manifest["decoder_tower_sha256"] and manifest["build_id"] == "20261009_77"
    assert (manifest["git_sha"], manifest["git_dirty"], manifest["job_id"]) == (SHA, True, "77")
    assert manifest["checkpoint_13d_sha256"] == hashlib.sha256(b"13d weights").hexdigest()
    assert manifest["transform"] == bg.transform_facts(ref.IMAGE_MEAN, ref.IMAGE_STD, ref.IMAGE_SIZE)
    assert manifest["isbi_cross_check"] == {"status": "absent"}
    assert "synthetic" not in manifest
    assert sorted(p.name for p in out.iterdir()) == sorted(n for n in CONTRACT_FILES if n not in ("labels.npy", "label_names.json"))


def test_build_runs_the_real_reference_loaders_and_keeps_every_row_where_the_parquet_has_it(tmp_path, ref, monkeypatch):
    """The reference script's own build_dataloader (Hugging Face parquet loading, MIMICValDataset, the DataLoader, the GPT-2
    tokenizer) and encode_dataset, with only the two models faked: the tower as a function of the pixels, the text encoder as a
    function of the token ids. Row i of every vector file must be the model's answer for row i of the parquet, the property
    that makes the texts and the metadata line up with the vectors."""
    torch = pytest.importorskip("torch")
    datasets = pytest.importorskip("datasets")
    from PIL import Image
    monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", str(tmp_path / "hf_cache"))
    deps = bg.real_deps()
    try:
        tokenizer = deps.make_tokenizer()
    except Exception:
        pytest.skip("the GPT-2 tokenizer is not in the local Hugging Face cache")
    assert (tokenizer.padding_side, tokenizer.pad_token) == ("right", tokenizer.eos_token), "the reference script's tokenizer setup"

    class FakeTextEncoder:
        def __init__(self):
            self.table = torch.randn(len(tokenizer), 8, generator=torch.Generator().manual_seed(3))

        def encode(self, input_ids, attention_mask=None):
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (self.table[input_ids] * mask).sum(1) / mask.sum(1).clamp(min=1)
            return torch.nn.functional.normalize(pooled, dim=-1)

    class FakeTower(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("w", torch.randn(3 * 4 * 4, 8, generator=torch.Generator().manual_seed(4)))

        def forward(self, pixels):
            return torch.nn.functional.adaptive_avg_pool2d(pixels, 4).flatten(1) @ self.w

    text_enc, tower = FakeTextEncoder(), FakeTower()
    frames, _ = bg.tiny_frames(tmp_path / "images", seed=9, n_images=24, n_test=8)
    data = tmp_path / "data"
    data.mkdir()
    for name, split in (("train", "train"), ("validate", "test"), ("test", "test")):        # the loader reads all three files
        frames[split].to_parquet(data / "{}.parquet".format(name), index=False)
    args = make_args(tmp_path, data=str(data), out=str(tmp_path / "gallery" / "20261009_77"), workers=0)
    deps.device = lambda: "cpu"
    deps.load_models = lambda ckpt, device: (text_enc, None, tower)
    deps.load_decoder = lambda ckpt, config, device="cpu": SimpleNamespace(image_encoder=tower)
    deps.provenance = partial(bg.provenance, root=stamped_tree(tmp_path), env={}, now=NOW)
    assert deps.build_dataloader is ref.build_dataloader and deps.encode_dataset is ref.encode_dataset
    manifest = bg.build(args, deps)

    out = Path(args.out)
    transform = ref._img_transform()
    for split, image_file, text_file in (("train", "img_emb.npy", "txt_emb.npy"), ("test", "test_img_emb.npy", "txt_emb_test.npy")):
        frame = frames[split]
        with torch.no_grad():
            pixels = torch.stack([transform(Image.open(p).convert("RGB")) for p in frame["image"]])
            want_img = torch.nn.functional.normalize(tower(pixels).float(), dim=-1).numpy()
            items = [ref.MIMICValDataset(datasets.Dataset.from_dict(frame.to_dict("list")), tokenizer)[i] for i in range(len(frame))]
            want_txt = text_enc.encode(torch.stack([x["input_ids"] for x in items]),
                                       attention_mask=torch.stack([x["attention_mask"] for x in items])).numpy()
        assert len({tuple(np.round(r, 4)) for r in want_img}) == len(frame), "the images tell the rows apart, so the check has teeth"
        got_img = load(out, image_file).astype(np.float32)
        got_txt = load(out, text_file).astype(np.float32)
        if split == "train":
            got_txt = got_txt[:len(frame)]
        assert np.allclose(got_img, want_img, atol=2e-3), "{}: image vector i is the tower's answer for parquet row i".format(split)
        assert np.allclose(got_txt, want_txt, atol=2e-3), "{}: text vector i is the encoder's answer for parquet row i".format(split)
    assert np.allclose(load(out, "txt_emb.npy")[24:].astype(np.float32), load(out, "txt_emb_test.npy"), atol=2e-3)
    assert manifest["counts"] == {"images": 24, "report_rows": 32, "report_groups": manifest["counts"]["report_groups"], "test": 8}
    assert manifest["gate_rk"] == {"app": ref.compute_retrieval_metrics(load(out, "test_img_emb.npy"), load(out, "txt_emb_test.npy"))}
    assert manifest["towers_identical"] is True


def test_build_prints_only_gallery_lines_of_counts_booleans_and_hashes(tmp_path, ref, capsys):
    World(tmp_path, ref).build()
    lines = capsys.readouterr().out.splitlines()
    assert lines and not [l for l in lines if not GALLERY_SHAPES.match(l)], [l for l in lines if not GALLERY_SHAPES.match(l)]
    assert not [l for l in lines if "/" in l or str(tmp_path) in l], "R7: no path in a [gallery] line"
    text = "\n".join(lines)
    for needle in ("towers_identical=True img_proj=False", "test: 10 rows", "train: 30 rows", "texts: 40 rows,",
                   "hashed 40 image files", "isbi cross-check: status=absent", "manifest written, labels_status=pending"):
        assert needle in text, needle
    order = [text.index(n) for n in ("towers_identical", "test: 10 rows", "gate_rk app", "train: 30 rows", "texts: 40 rows", "hashed 40",
                                     "manifest written")]
    assert order == sorted(order), "the tower verdict first, the test gate before the long train pass, the manifest last"


def test_a_different_decoder_tower_is_recorded_not_hidden(tmp_path, ref, capsys):
    w = World(tmp_path, ref, decoder_differs=True)
    manifest = w.build()
    assert manifest["towers_identical"] is False and manifest["tower_sha256"] != manifest["decoder_tower_sha256"]
    assert "towers_identical=False img_proj=False" in capsys.readouterr().out
    assert (w.out / "manifest.json").exists(), "the build goes on: P5-D then loads the 13D tower as a second module"


def test_an_img_proj_means_the_towers_are_not_identical_even_with_equal_hashes(tmp_path, ref, capsys):
    manifest = World(tmp_path, ref, img_proj=object()).build()
    assert manifest["towers_identical"] is False and manifest["img_proj_present"] is True
    assert "towers_identical=False img_proj=True" in capsys.readouterr().out


def test_build_refuses_a_loader_and_parquet_that_disagree_about_the_rows_before_writing_a_manifest(tmp_path, ref):
    w = World(tmp_path, ref, train_rows_in_loader=29)
    with pytest.raises(RuntimeError) as err:
        w.build()
    assert "train" in str(err.value) and "29" in str(err.value) and str(tmp_path) not in str(err.value)
    assert not (w.out / "manifest.json").exists(), "a build that failed leaves no 'finished' marker"


def test_build_refuses_a_parquet_without_the_columns_it_needs_before_loading_any_model(tmp_path, ref):
    """A missing column is found in a second, not after the hour of encoding."""
    w = World(tmp_path, ref)
    train = pd.read_parquet(w.data / "train.parquet").drop(columns=["dicom_id"])
    train.to_parquet(w.data / "train.parquet", index=False)
    with pytest.raises(RuntimeError) as err:
        w.build()
    assert "train.parquet" in str(err.value) and "dicom_id" in str(err.value) and str(tmp_path) not in str(err.value)
    assert w.calls == [], "refused before any model was loaded or any image encoded"
    assert not (w.out / "manifest.json").exists()


def test_a_requeued_build_overwrites_its_own_partial_files(tmp_path, ref):
    w = World(tmp_path, ref)
    w.out.mkdir(parents=True)
    (w.out / "img_emb.npy").write_bytes(b"partial")
    (w.out / "report_texts.txt").write_text("partial\n")
    w.build()
    assert load(w.out, "img_emb.npy").shape == (w.N_TRAIN, w.WIDTH)
    assert (w.out / "report_texts.txt").read_text() != "partial\n"


def test_build_hashes_files_in_parallel_without_losing_their_order(tmp_path):
    paths = []
    for i in range(25):
        paths.append(tmp_path / "f{}".format(i))
        paths[-1].write_bytes(("content %d" % i).encode())
    got = bg.hash_files([str(p) for p in paths], lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest(), workers=4)
    assert got == [hashlib.sha256(("content %d" % i).encode()).hexdigest() for i in range(25)]
    assert bg.hash_files([], lambda p: "x", workers=0) == []


def test_real_deps_are_the_reference_scripts_own_functions_and_the_fakes_have_the_same_names(tmp_path, ref):
    pytest.importorskip("transformers")
    from app.engine import file_sha256, tensor_sha256
    from scripts.evaluate_report_generation import load_report_generation_module
    real = bg.real_deps()
    assert real.load_models is ref.load_models and real.build_dataloader is ref.build_dataloader
    assert real.encode_dataset is ref.encode_dataset and real.group_ids_from_texts is ref.group_ids_from_texts
    assert real.compute_retrieval_metrics is ref.compute_retrieval_metrics
    assert real.load_decoder is load_report_generation_module
    assert real.tensor_sha256 is tensor_sha256 and real.file_sha256 is file_sha256
    assert real.provenance is bg.provenance and real.device() in ("cpu", "cuda")
    assert real.transform == bg.transform_facts(ref.IMAGE_MEAN, ref.IMAGE_STD, ref.IMAGE_SIZE)
    assert set(vars(real)) == set(vars(World(tmp_path, ref).deps)), "build() sees the same seam either way"


def test_the_manifests_transform_facts_are_the_same_numbers_whichever_path_writes_them(ref):
    from app.imaging import CLIP_MEAN, CLIP_STD
    assert list(ref.IMAGE_MEAN) == list(CLIP_MEAN) and list(ref.IMAGE_STD) == list(CLIP_STD) and ref.IMAGE_SIZE == 224
    assert bg.transform_facts(CLIP_MEAN, CLIP_STD, 224) == bg.transform_facts(ref.IMAGE_MEAN, ref.IMAGE_STD, ref.IMAGE_SIZE)


# ── the wrapper, rehearsed in a temp tree ─────────────────────────────────────

PYTHON_STUB = r"""#!/bin/bash
# Stands in for the venv's python: records every call, plays the GPU probe, the build step and the reference step, and runs
# everything else (the wrapper's path helper, the REAL --compare-rk) with the test interpreter.
{ echo "@@"; for a in "$@"; do printf '%s\n' "$a"; done; } >> "$STUB_DIR/python.calls"
case "$*" in
  *torch.cuda.device_count*) echo "${FAKE_GPUS:-1} NVIDIA H100 80GB HBM3"; exit 0;;
  "scripts/build_retrieval_gallery.py "*--compare-rk*) exec "$REAL_PYTHON" "$@";;
  "scripts/build_retrieval_gallery.py "*|"scripts/evaluate_cxr_retrieval.py "*) ;;
  *) exec "$REAL_PYTHON" "$@";;
esac
out=""; odir=""; prev=""
for a in "$@"; do
  [ "$prev" = "--out" ] && out="$a"
  [ "$prev" = "--output-dir" ] && odir="$a"
  prev="$a"
done
if [ "$1" = scripts/build_retrieval_gallery.py ]; then
  printf 'Encoding:  50%%|#####     | 1/2\r'
  echo "[gallery] towers_identical=True img_proj=False"
  echo "[gallery] /sc/home/someone/images/p10/leak.jpg a path that a gallery line must never carry"
  echo "[gallery] test: 40 rows"
  echo "Findings: FAKE REPORT TEXT study_id=12345678 /sc/home/someone/images/p10/img.jpg"
  echo "Traceback (most recent call last): FAKE MIMIC TEXT in a message" >&2
  [ "$(cat "$STUB_DIR/build.mode")" = fail ] && exit 3
  mkdir -p "$out"
  cp "$STUB_DIR/gate_rk.json" "$out/gate_rk.json"
  cp "$STUB_DIR/manifest.json" "$out/manifest.json"
  exit 0
fi
echo "Loading checkpoint: /sc/home/someone/outputs/x/checkpoints/last.ckpt"
echo "Findings: FAKE REPORT TEXT study_id=12345678 /sc/home/someone/images/p10/img.jpg"
echo "Traceback (most recent call last): FAKE MIMIC TEXT in a message" >&2
[ "$(cat "$STUB_DIR/ref.mode")" = fail ] && exit 4
mkdir -p "$odir"
cp "$STUB_DIR/reference.json" "$odir/phase6_mimic_20261009_101500.json"
exit 0
"""

CLUSTER_USER = "krishankumar.bhushan"
STAMP = "2026-10-09T20:00:00Z 3f2a9c41d7e86b05a1c4e9d3b7f60285ac9e1d47 clean"
LINE_OK = re.compile(r"^(=== |\[gallery\] |RESULT |ERROR)")
CKPT_13D = "./outputs/h100_kd_150m_v2_full_data_lr3e6/checkpoints/last.ckpt"
CKPT_DECODER = "./outputs/h100_report_gen_m3_tower13d_s42/checkpoints/last.ckpt"


def snapshot(root: Path) -> List[tuple]:
    """Every path under `root` with size and mtime: what a job that only READS a tree leaves exactly as it was (R8)."""
    found = []
    for base, dirs, files in os.walk(str(root)):
        for name in dirs + files:
            full = os.path.join(base, name)
            st = os.lstat(full)
            found.append((os.path.relpath(full, str(root)), st.st_size, st.st_mtime_ns))
    return sorted(found)


class JobBox:
    """The wrapper run for real in a temp tree standing in for the cluster: CLUSTER_REPO (repo/, with a copy of the real builder
    so that --compare-rk runs for real), the thesis checkout (main/) behind repo/outputs, CHAT_HOME (chat/), the dataset
    (data/) and a stub python for the GPU probe, the build step and the reference step."""

    def __init__(self, root: Path, stamp: Optional[str] = STAMP):
        self.root = root
        self.repo, self.main, self.chat, self.data = root / "repo", root / "main", root / "chat", root / "data"
        self.stubs, self.bin, self.scratch = root / "stubs", root / "bin", root / "scratch"
        (self.repo / "scripts").mkdir(parents=True)
        for name in ("build_retrieval_gallery_h100.sh", "build_retrieval_gallery.py"):
            shutil.copy(str(REPO_ROOT / "scripts" / name), str(self.repo / "scripts" / name))
        (self.repo / ".venv" / "bin").mkdir(parents=True)
        (self.repo / ".venv" / "bin" / "activate").write_text("")
        (self.repo / "logs").mkdir()
        if stamp is not None:
            (self.repo / ".sync_stamp").write_text(stamp + "\n")
        for rel in ("h100_kd_150m_v2_full_data_lr3e6", "h100_report_gen_m3_tower13d_s42"):
            ckpt = self.main / "outputs" / rel / "checkpoints" / "last.ckpt"
            ckpt.parent.mkdir(parents=True)
            ckpt.write_bytes(b"")
        (self.repo / "outputs").symlink_to(self.main / "outputs", target_is_directory=True)
        for directory in (self.chat, self.data, self.stubs, self.bin, self.scratch):
            directory.mkdir()
        for name in ("train.parquet", "test.parquet"):
            (self.data / name).write_bytes(b"")
        python = self.bin / "python"
        python.write_text(PYTHON_STUB)
        python.chmod(0o755)
        self.build_id = "20261009_1234567"
        self.set_build_mode("ok")
        self.set_ref_mode("ok")
        self.set_metrics(METRICS, METRICS)

    @property
    def out_dir(self) -> Path:
        return self.chat / "gallery" / self.build_id

    def set_build_mode(self, mode: str) -> None:
        (self.stubs / "build.mode").write_text(mode + "\n")

    def set_ref_mode(self, mode: str) -> None:
        (self.stubs / "ref.mode").write_text(mode + "\n")

    def set_metrics(self, app: dict, reference: dict) -> None:
        """What the stub build step leaves as gate_rk.json (the app side) and the stub reference step as its result file."""
        (self.stubs / "gate_rk.json").write_text(json.dumps({"app": app}))
        (self.stubs / "manifest.json").write_text(json.dumps({"build_id": self.build_id, "counts": {"images": 1}, "gate_rk": {"app": app}}))
        (self.stubs / "reference.json").write_text(json.dumps({"timestamp": "20261009_101500", "metrics": reference}))

    def run(self, **extra_env: Optional[str]) -> subprocess.CompletedProcess:
        env = {"PATH": "{}:/usr/bin:/bin".format(self.bin), "HOME": str(self.root / "home"), "USER": CLUSTER_USER,
               "SLURM_SUBMIT_DIR": str(self.repo), "SLURM_JOB_ID": "1234567", "SLURM_CPUS_PER_TASK": "16",
               "CHAT_HOME": str(self.chat), "SCRATCH_ROOT": str(self.scratch), "DATA": str(self.data), "BUILD_ID": self.build_id,
               "STUB_DIR": str(self.stubs), "REAL_PYTHON": sys.executable, "PYTHONDONTWRITEBYTECODE": "1"}
        env.update(extra_env)
        env = {k: v for k, v in env.items() if v is not None}
        # stdout and stderr share one pipe, like the single SLURM log: whatever bash itself complains about counts too.
        return subprocess.run([BASH, str(self.repo / "scripts" / "build_retrieval_gallery_h100.sh")], cwd=str(self.root), env=env,
                              stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=240)

    def calls(self) -> List[List[str]]:
        log = self.stubs / "python.calls"
        if not log.exists():
            return []
        return [rec.splitlines() for rec in log.read_text().split("@@\n") if rec.strip()]

    def steps(self) -> Dict[str, List[List[str]]]:
        """The three job steps as the stub saw them: build, reference, compare."""
        calls = [c for c in self.calls() if c and c[0].startswith("scripts/")]
        return {"build": [c for c in calls if c[0].endswith("build_retrieval_gallery.py") and "--compare-rk" not in c],
                "reference": [c for c in calls if c[0].endswith("evaluate_cxr_retrieval.py")],
                "compare": [c for c in calls if c[0].endswith("build_retrieval_gallery.py") and "--compare-rk" in c]}

    def ran_nothing(self) -> bool:
        return not any(self.steps().values())


def job_lines(done: subprocess.CompletedProcess) -> List[str]:
    return done.stdout.splitlines()


def results(lines: List[str]) -> List[dict]:
    return [json.loads(l[len("RESULT "):]) for l in lines if l.startswith("RESULT ")]


def test_a_clean_run_builds_runs_the_reference_compares_and_prints_only_safe_lines(tmp_path):
    box = JobBox(tmp_path)
    before = {name: snapshot(getattr(box, name)) for name in ("repo", "main", "data")}
    done = box.run()
    lines = job_lines(done)
    assert done.returncode == 0, done.stdout
    # R7: only wrapper- and builder-authored lines, and nothing the python steps printed besides [gallery] lines
    assert [l for l in lines if not LINE_OK.match(l)] == [], "a line that is not ===, [gallery], RESULT or ERROR"
    assert not [l for l in lines if "FAKE" in l or "12345678" in l or "/sc/home" in l or "Traceback" in l or "Encoding" in l]
    assert lines[0] == "=== sync 3f2a9c41d7e86b05a1c4e9d3b7f60285ac9e1d47 clean ===", "the provenance comes first"
    gallery = [l for l in lines if l.startswith("[gallery]")]
    assert gallery == ["[gallery] towers_identical=True img_proj=False", "[gallery] test: 40 rows"], \
        "the builder's lines pass, the one that carries a path and the progress bar's fragment do not"
    (final,) = results(lines)
    assert final["gate_rk_equal"] is True and final["n"] == 2663 and final["i2t_R@10"] == 0.1714
    assert isinstance(final["wall_s"], int) and 0 <= final["wall_s"] < 600
    assert lines[-1].startswith("=== END {}".format(box.build_id)) and "wall_s=" in lines[-1]
    assert all(len(l) <= 300 for l in lines if l.startswith(("RESULT ", "ERROR")))
    # the raw output is in files, under OUT
    assert "FAKE REPORT TEXT" in (box.out_dir / "build.log").read_text()
    assert "Traceback" in (box.out_dir / "build.log").read_text(), "stderr is in the file too"
    assert "FAKE REPORT TEXT" in (box.out_dir / "reference_rk.log").read_text()
    assert (box.out_dir / "reference_rk" / "phase6_mimic_20261009_101500.json").is_file()
    gate = json.loads((box.out_dir / "gate_rk.json").read_text())
    assert gate["equal"] is True and gate["app"] == METRICS and gate["reference"] == METRICS
    assert json.loads((box.out_dir / "manifest.json").read_text())["gate_rk"]["equal"] is True
    # R8: nothing outside OUT was touched
    for name, snap in before.items():
        assert snapshot(getattr(box, name)) == snap, name
    assert [p.name for p in box.chat.iterdir()] == ["gallery"] and [p.name for p in (box.chat / "gallery").iterdir()] == [box.build_id]


def test_the_three_steps_run_in_order_with_the_published_commands(tmp_path):
    box = JobBox(tmp_path)
    assert box.run().returncode == 0
    steps = box.steps()
    (build,), (reference,), (compare,) = steps["build"], steps["reference"], steps["compare"]
    assert build == ["scripts/build_retrieval_gallery.py", "--checkpoint-13d", CKPT_13D, "--decoder-checkpoint", CKPT_DECODER,
                     "--decoder-config", "hybrid_150m_m3_rrg", "--data", str(box.data), "--out", str(box.out_dir),
                     "--build-id", box.build_id, "--workers", "16", "--isbi-cache", str(box.scratch / "isbi_gallery_adapted.pt")]
    assert reference == ["scripts/evaluate_cxr_retrieval.py", "--checkpoint", CKPT_13D, "--dataset", "mimic", "--local-parquet-dir",
                         str(box.data), "--mimic-split", "test", "--output-dir", str(box.out_dir / "reference_rk")]
    assert compare[:3] == ["scripts/build_retrieval_gallery.py", "--compare-rk", str(box.out_dir)] and compare[3] == "--wall-s"
    assert compare[4].isdigit() and len(compare) == 5
    order = [c[0] for c in box.calls() if c[0].startswith("scripts/")]
    assert order == ["scripts/build_retrieval_gallery.py", "scripts/evaluate_cxr_retrieval.py", "scripts/build_retrieval_gallery.py"]
    bg.parse_args(build[1:])                          # the wrapper's flags are the builder's
    bg.parse_args(compare[1:])


def test_the_report_model_its_config_the_checkpoints_and_the_build_name_can_be_chosen_on_the_submit_line(tmp_path):
    """`chat_remote.sh submit <wrapper> NAME=value` is how the other selectable report model (hybrid_150m_v2_rrg, the 13D
    decoder) is built; whatever is chosen is what the builder and the reference script are given."""
    box = JobBox(tmp_path)
    (box.root / "alt").mkdir()
    alt_13d, alt_decoder = box.root / "alt" / "retrieval.ckpt", box.root / "alt" / "report.ckpt"
    alt_13d.write_bytes(b"")
    alt_decoder.write_bytes(b"")
    done = box.run(CKPT_13D=str(alt_13d), CHECKPOINT=str(alt_decoder), MODEL_CONFIG="hybrid_150m_v2_rrg", BUILD_ID="v2_13d_1")
    assert done.returncode == 0, done.stdout
    (build,), (reference,) = box.steps()["build"], box.steps()["reference"]
    assert build[build.index("--checkpoint-13d") + 1] == str(alt_13d)
    assert build[build.index("--decoder-checkpoint") + 1] == str(alt_decoder)
    assert build[build.index("--decoder-config") + 1] == "hybrid_150m_v2_rrg" and build[build.index("--build-id") + 1] == "v2_13d_1"
    assert build[build.index("--out") + 1] == str(box.chat / "gallery" / "v2_13d_1")
    assert reference[reference.index("--checkpoint") + 1] == str(alt_13d), "the gate runs on the retrieval checkpoint that was built from"
    assert (box.chat / "gallery" / "v2_13d_1" / "manifest.json").is_file()
    assert [l for l in job_lines(done) if l.startswith("=== P5-B")] == [
        "=== P5-B gallery v2_13d_1: 13D tower and text encoder, report model hybrid_150m_v2_rrg, official train and test splits, "
        "batch 32, 1 GPU ==="]


def test_every_flag_the_wrapper_gives_the_reference_script_is_one_of_its_own():
    """The brief's reading of the reference CLI, checked against the real parser: `--help` ends main() before any model loads."""
    pytest.importorskip("datasets")
    import contextlib
    import io
    from scripts import evaluate_cxr_retrieval
    buffer = io.StringIO()
    argv = sys.argv
    sys.argv = ["evaluate_cxr_retrieval.py", "--help"]
    try:
        with contextlib.redirect_stdout(buffer), pytest.raises(SystemExit) as stop:
            evaluate_cxr_retrieval.main()
    finally:
        sys.argv = argv
    assert stop.value.code == 0
    help_text = buffer.getvalue()
    for flag in ("--checkpoint", "--dataset", "--local-parquet-dir", "--mimic-split", "--output-dir"):
        assert flag in help_text, flag
    assert re.search(r"--mimic-split \{train,validation,test\}", help_text)
    assert "{indiana,mimic}" in help_text


def test_an_unequal_gate_fails_the_job_with_the_verdict_in_the_log(tmp_path):
    box = JobBox(tmp_path)
    box.set_metrics(METRICS, dict(METRICS, **{"t2i_R@10": 0.2}))
    done = box.run()
    lines = job_lines(done)
    assert done.returncode == 1, done.stdout
    (final,) = results(lines)
    assert final["gate_rk_equal"] is False and final["differs"] == ["t2i_R@10"]
    assert "ERROR compare exit=1" in lines and not [l for l in lines if l.startswith("=== END")]
    assert [l for l in lines if not LINE_OK.match(l)] == []
    assert json.loads((box.out_dir / "gate_rk.json").read_text())["equal"] is False


def test_a_failed_build_prints_only_its_exit_code_and_its_safe_lines_and_goes_no_further(tmp_path):
    box = JobBox(tmp_path)
    box.set_build_mode("fail")
    done = box.run()
    lines = job_lines(done)
    assert done.returncode == 3, done.stdout
    assert "ERROR build exit=3" in lines
    assert "[gallery] towers_identical=True img_proj=False" in lines, "what the builder had printed before it failed still shows"
    assert [l for l in lines if not LINE_OK.match(l)] == [] and not [l for l in lines if "FAKE" in l or "/sc/home" in l]
    assert "FAKE REPORT TEXT" in (box.out_dir / "build.log").read_text()
    assert box.steps()["reference"] == [] and box.steps()["compare"] == []
    assert not (box.out_dir / "manifest.json").exists()


def test_a_failed_reference_run_prints_only_its_exit_code_and_skips_the_comparison(tmp_path):
    box = JobBox(tmp_path)
    box.set_ref_mode("fail")
    done = box.run()
    lines = job_lines(done)
    assert done.returncode == 4, done.stdout
    assert "ERROR reference exit=4" in lines
    assert [l for l in lines if not LINE_OK.match(l)] == [] and not [l for l in lines if "FAKE" in l or "/sc/home" in l]
    assert "Traceback" in (box.out_dir / "reference_rk.log").read_text()
    assert box.steps()["compare"] == []


def test_a_comparison_that_cannot_run_is_an_error_line_not_a_traceback_in_the_job_log(tmp_path):
    box = JobBox(tmp_path)
    (box.repo / "scripts" / "build_retrieval_gallery.py").write_text(
        'import sys\nif "--compare-rk" in sys.argv:\n'
        '    print("partial study_id=12345678 FAKE REPORT TEXT")\n'
        '    raise RuntimeError("cannot read study_id=12345678 /sc/home/someone/images/p10/img.jpg FAKE REPORT TEXT")\n')
    # the stub plays the build step, so the crashing script only runs as the compare step
    done = box.run()
    lines = job_lines(done)
    assert done.returncode == 1, done.stdout
    assert "ERROR compare exit=1" in lines
    assert [l for l in lines if not LINE_OK.match(l)] == []
    assert not [l for l in lines if "Traceback" in l or "12345678" in l or "FAKE" in l or "/sc/home" in l], lines
    assert "FAKE REPORT TEXT" in (box.out_dir / "compare.err").read_text()


def test_a_finished_build_is_never_overwritten(tmp_path):
    box = JobBox(tmp_path)
    box.out_dir.mkdir(parents=True)
    (box.out_dir / "manifest.json").write_text('{"first": "build"}\n')
    before = snapshot(box.chat)
    done = box.run()
    lines = job_lines(done)
    assert done.returncode == 1, done.stdout
    assert [l for l in lines if l.startswith("ERROR")] and box.build_id in "\n".join(lines)
    assert box.ran_nothing()
    assert (box.out_dir / "manifest.json").read_text() == '{"first": "build"}\n' and snapshot(box.chat) == before


@pytest.mark.parametrize("where", ["inside_outputs", "under_main", "outputs_elsewhere", "symlink_into_main", "symlink_named_outputs",
                                   "symlink_to_outputs_elsewhere"])
def test_an_out_dir_in_an_outputs_directory_or_under_the_thesis_checkout_is_refused(tmp_path, where):
    box = JobBox(tmp_path)
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
        # The path NAMES an outputs directory but resolves somewhere that does not: only the check on the path as given refuses it.
        (tmp_path / "real_place" / "chat").mkdir(parents=True)
        (tmp_path / "outputs").symlink_to(tmp_path / "real_place", target_is_directory=True)
        home = tmp_path / "outputs" / "chat"
    else:
        # The reverse: nothing in the path says outputs, but it resolves into an outputs directory.
        (tmp_path / "other" / "outputs" / "chat").mkdir(parents=True)
        home = tmp_path / "plain_name"
        home.symlink_to(tmp_path / "other" / "outputs" / "chat", target_is_directory=True)
    if where in ("inside_outputs", "under_main", "outputs_elsewhere"):
        home.mkdir(parents=True, exist_ok=True)
    before_main = snapshot(box.main)
    done = box.run(CHAT_HOME=str(home))
    errors = [l for l in job_lines(done) if l.startswith("ERROR")]
    assert done.returncode == 1, done.stdout
    assert any("outputs" in l or "thesis checkout" in l for l in errors), errors      # refused for this reason, not for a missing CHAT_HOME
    assert box.ran_nothing()
    assert snapshot(box.main) == before_main, "R8: nothing was created in the thesis checkout"
    assert not (home / "gallery").exists(), "no directory was created before the guards passed"


@pytest.mark.parametrize("build_id", ["../escape", "a/b", ".", "..", "-x", "x y", "x;y", "x$(id)"])
def test_a_build_id_that_is_not_one_plain_name_is_refused(tmp_path, build_id):
    box = JobBox(tmp_path)
    done = box.run(BUILD_ID=build_id)
    assert done.returncode == 1, done.stdout
    assert [l for l in job_lines(done) if l.startswith("ERROR")] and box.ran_nothing()
    assert not (box.chat / "gallery").exists() and not (tmp_path / "escape").exists()


def test_the_default_build_id_is_the_date_and_the_job_id_so_a_requeue_resumes_in_the_same_directory(tmp_path):
    box = JobBox(tmp_path)
    first_day = time.strftime("%Y%m%d")
    done = box.run(BUILD_ID=None)
    last_day = time.strftime("%Y%m%d")                   # midnight may pass between the two clocks
    assert done.returncode == 0, done.stdout
    (directory,) = (box.chat / "gallery").iterdir()
    assert directory.name in (first_day + "_1234567", last_day + "_1234567"), directory.name
    again = box.run(BUILD_ID=None, SLURM_RESTART_COUNT="1")
    assert again.returncode == 1 and "already built" in again.stdout, "same job id, same directory, and it is finished"


def test_a_stray_out_variable_cannot_redirect_the_gallery(tmp_path):
    """sbatch exports the submitting shell, and OUT is a common name: the output directory is CHAT_HOME/gallery/<id>, nothing else."""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    box = JobBox(tmp_path / "box")
    done = box.run(OUT=str(elsewhere), OUT_DIR=str(elsewhere))
    assert done.returncode == 0, done.stdout
    assert (box.out_dir / "manifest.json").is_file() and list(elsewhere.iterdir()) == []


def test_a_missing_chat_home_or_input_stops_the_job_before_anything_runs(tmp_path):
    box = JobBox(tmp_path)
    shutil.rmtree(str(box.chat))
    assert box.run().returncode == 1 and not (tmp_path / "chat").exists(), "CHAT_HOME is never created here"
    box.chat.mkdir()
    things = [box.main / "outputs" / "h100_kd_150m_v2_full_data_lr3e6" / "checkpoints" / "last.ckpt",
              box.main / "outputs" / "h100_report_gen_m3_tower13d_s42" / "checkpoints" / "last.ckpt",
              box.data / "train.parquet", box.data / "test.parquet"]
    for path in things:
        path.rename(str(path) + ".away")
        done = box.run()
        assert done.returncode == 1 and [l for l in job_lines(done) if l.startswith("ERROR")], path.name
        assert box.ran_nothing() and not box.out_dir.exists(), path.name
        Path(str(path) + ".away").rename(path)


def test_without_a_gpu_the_job_stops_before_any_step(tmp_path):
    box = JobBox(tmp_path)
    done = box.run(FAKE_GPUS="0")
    assert done.returncode == 1
    assert any(l.startswith("ERROR") and "0 GPU" in l for l in job_lines(done)), job_lines(done)
    assert box.ran_nothing() and not box.out_dir.exists()


def test_a_requeued_unfinished_build_starts_again_in_its_directory_and_replaces_the_partial_raw_logs(tmp_path):
    box = JobBox(tmp_path)
    box.out_dir.mkdir(parents=True)
    (box.out_dir / "img_emb.npy").write_bytes(b"partial")
    (box.out_dir / "build.log").write_text("earlier attempt line\n")
    done = box.run(SLURM_RESTART_COUNT="1")
    lines = job_lines(done)
    assert done.returncode == 0, done.stdout
    assert any("restart=1" in l and l.startswith("=== job=") for l in lines)
    assert any("earlier attempt" in l and l.startswith("=== ") for l in lines), "the restart is announced"
    assert "earlier attempt line" not in (box.out_dir / "build.log").read_text(), "the raw log is the latest attempt's"
    assert (box.out_dir / "manifest.json").is_file()


def test_the_job_never_deletes_anything(tmp_path):
    src = (REPO_ROOT / "scripts" / "build_retrieval_gallery_h100.sh").read_text()
    code = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))
    assert not re.search(r"(^|[\s;&|(])rm(\s|$)", code, re.M) and "--delete" not in code and "ln -sf" not in code


def real_stamp(root: Path, dirty: bool) -> str:
    """The .sync_stamp that `scripts/chat_remote.sh sync` writes, from the real script in a throwaway git tree with ssh and rsync
    stubbed (tests/test_chat_remote.py's Sandbox): the wrapper has to read what the producer writes."""
    box = Sandbox(root)
    if dirty:
        script = box.repo / "scripts" / "chat_remote.sh"
        script.write_text(script.read_text() + "\n# touched\n")
    done = box.run("sync")
    assert done.returncode == 0, done.stdout + done.stderr
    return (box.repo / ".sync_stamp").read_text().strip()


@pytest.mark.parametrize("dirty", [False, True])
def test_the_first_line_of_the_job_log_names_the_commit_and_cleanliness_the_tree_was_synced_with(tmp_path, dirty):
    stamp = real_stamp(tmp_path / "sync", dirty)
    assert STAMP_RE.match(stamp), stamp
    _, sha, flag = stamp.split()
    assert flag == ("dirty" if dirty else "clean")
    box = JobBox(tmp_path / "job", stamp=stamp)
    done = box.run(FAKE_GPUS="0")                  # the job stops at the GPU check: the first line is all this asks about
    assert job_lines(done)[0] == "=== sync {} {} ===".format(sha, flag), job_lines(done)[:2]


def test_a_missing_sync_stamp_is_reported_as_unknown_and_the_job_goes_on(tmp_path):
    box = JobBox(tmp_path, stamp=None)
    done = box.run()
    assert done.returncode == 0, done.stdout
    assert job_lines(done)[0] == "=== sync unknown ==="


@pytest.mark.parametrize("stamp", [
    "",                                                                              # empty
    "rm -rf / ; echo $(whoami) FAKE_STAMP_TEXT",                                     # not the shape at all
    "2026-10-09T20:00:00Z 3f2a9c41d7e86b05a1c4e9d3b7f60285ac9e1d4 clean",           # 39 hex digits
    "2026-10-09T20:00:00Z 3F2A9C41D7E86B05A1C4E9D3B7F60285AC9E1D47 clean",          # upper case
    "2026-10-09T20:00:00Z 3f2a9c41d7e86b05a1c4e9d3b7f60285ac9e1d47 maybe",          # a flag that is neither
    "2026-10-09T20:00:00Z 3f2a9c41d7e86b05a1c4e9d3b7f60285ac9e1d47 clean FAKE_EXTRA",   # a field too many
])
def test_a_malformed_sync_stamp_is_reported_as_unknown_and_never_echoed(tmp_path, stamp):
    box = JobBox(tmp_path, stamp=stamp)
    done = box.run(FAKE_GPUS="0")
    lines = job_lines(done)
    assert lines[0] == "=== sync unknown ===", lines[:2]
    assert not [l for l in lines if "FAKE_" in l or "whoami" in l or "rm -rf" in l]
