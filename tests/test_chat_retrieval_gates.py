"""P5-F (CHAT_UI_PLAN.md): the live-path gates job, scripts/chat_retrieval_gates.py and its wrapper scripts/chat_retrieval_gates_h100.sh.

Laptop work: CPU, offline, synthetic data only (R7). The real job needs the 2.4 GB checkpoint, the real gallery (MIMIC-derived, Class R)
and the CheXbert weights, so the tiny stack stands in for all three: app.tiny's random-init engine (`--engine tiny`, the one argument the
job never passes), P5-B's `--tiny` gallery re-embedded with that very engine (so that a train image finds itself and the live vector of
a test image is the build's), and the keyword RuleLabeler or, behind the real HTTP surface, a fake f1chexbert.
Tested in layers:
  * the pure pieces: the rows checked, the published labels in both formats;
  * main() in process over the tiny world: the three checks, every refusal, what is printed (R7) and what gates.json holds;
  * the real LabelerClient against a stdlib stand-in for the labeller service, and the CLI in a fresh interpreter;
  * the allowlist of the wrapper (the very patterns, through grep) over everything the script prints;
  * the wrapper, rehearsed in a temp tree (tests/wrapper_rehearsal.py) with the REAL script (tiny engine), the REAL app.labeler served by
    the REAL uvicorn over the fake f1chexbert, and stub pythons that record the calls.
The static pins on the wrapper are in tests/test_willi_parity.py.
"""
import ast
import http.server
import importlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pytest

from app.engine import build_engine
from app.gallery import Gallery
from app.labels import CHEXBERT_14, LabelerUnavailable, RuleLabeler
from scripts import build_retrieval_gallery as bg
from tests import wrapper_rehearsal as wr
from tests.app_helpers import decide_gate
from tests.test_chat_remote import _summary_of
from tests.wrapper_rehearsal import STAMP, job_lines, real_stamp, results, snapshot

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "chat_retrieval_gates.py"
SH = "chat_retrieval_gates_h100.sh"
LINE_OK = wr.safe_line("gates")          # the first words of every line a job may print
BUILD = "g13d_m3_v1"
DUMP_NAME = "report_gen_m3_test_split_s42"
N_IMAGES, N_TEST = 200, 60               # the tiny world: 200 train images, 60 test studies (the job needs 50 of each)
# The label order, written out here once more on purpose: the pin is that every copy of it agrees.
NAMES = ["Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema", "Consolidation", "Pneumonia",
         "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices", "No Finding"]
RESULT_CLEAN = 'RESULT {"self_retrieval":50,"own_rank_equal":50,"own_rank_dedup_equal":50,"labeller_equal":100,"label_names_ok":true}'


@pytest.fixture(scope="module")
def gates():
    """scripts/chat_retrieval_gates.py, imported when a test asks for it, so that its absence fails the tests one by one."""
    return importlib.import_module("scripts.chat_retrieval_gates")


# ── the tiny world: gallery, dataset and published dump that agree with one another ─────────────────────────────────────────────────

def embed(engine: Any, paths: Any) -> np.ndarray:
    """The engine's pooled vector of each image file, through the upload path (file bytes in): float32, one row per file."""
    rows = []
    for path in paths:
        _, prepared = engine.preprocess(Path(path).read_bytes())
        _, encoded = engine.encode(prepared)
        rows.append(encoded.pooled.numpy())
    return np.stack(rows).astype(np.float32)


def build_world(root: Path) -> None:
    """root/gallery (P5-B's --tiny gallery, its gate decided, re-embedded with the tiny engine and carrying its tower hash), root/data
    (train.parquet and test.parquet with the columns the job reads and some it does not), root/dump (hyps.txt, refs.txt and
    chexbert_labels.json as the scoring job writes them, with RuleLabeler's labels). Float32 train vectors: the real build writes
    float16, and the tiny tower's vectors are so alike (cosines of 0.9998) that half precision would make the margin a coin toss."""
    gallery = root / "gallery"
    bg.build_tiny(gallery, n_images=N_IMAGES, n_test=N_TEST)
    decide_gate(gallery)
    engine = build_engine("tiny")
    meta, test_meta = pd.read_parquet(gallery / "img_meta.parquet"), pd.read_parquet(gallery / "test_meta.parquet")
    np.save(gallery / "img_emb.npy", embed(engine, meta["image"]))
    np.save(gallery / "test_img_emb.npy", embed(engine, test_meta["image"]))
    manifest = json.loads((gallery / "manifest.json").read_text())
    manifest.update(tower_sha256=engine.tower_sha256(), decoder_tower_sha256=engine.tower_sha256(), towers_identical=True)
    (gallery / "manifest.json").write_text(json.dumps(manifest, indent=2))
    texts = (gallery / "report_texts.txt").read_text().splitlines()
    data = root / "data"
    data.mkdir()
    for split, frame, rows in (("train", meta, range(N_IMAGES)), ("test", test_meta, range(N_IMAGES, N_IMAGES + N_TEST))):
        pd.DataFrame({"image": frame["image"], "study_id": frame["study_id"], "findings": [texts[r] for r in rows],
                      "impression": ["" for _ in rows]}).to_parquet(data / "{}.parquet".format(split))
    dump = root / "dump"
    dump.mkdir()
    refs = [texts[N_IMAGES + t] for t in range(N_TEST)]
    hyps = [texts[(7 * t) % N_IMAGES] for t in range(N_TEST)]
    (dump / "refs.txt").write_text("\n".join(refs) + "\n")
    (dump / "hyps.txt").write_text("\n".join(hyps) + "\n")
    labeller = RuleLabeler()
    (dump / "chexbert_labels.json").write_text(json.dumps({
        "y_true": labeller.label(refs), "y_pred": labeller.label(hyps), "label_names": list(CHEXBERT_14),
        "five_label_indices": [1, 4, 5, 7, 9], "hyp_file": "hyps.txt", "ref_file": "refs.txt"}))


@pytest.fixture(scope="module")
def template(tmp_path_factory):
    root = tmp_path_factory.mktemp("template") / "world"
    root.mkdir()
    build_world(root)
    return root


class World:
    """A copy of the tiny world of its own, with the command line that points the job at it."""

    def __init__(self, root: Path, template: Path):
        shutil.copytree(str(template), str(root))
        self.root = root
        self.gallery, self.data, self.dump, self.out = root / "gallery", root / "data", root / "dump", root / "out"

    def argv(self, *extra: str, engine: str = "tiny") -> List[str]:
        return ["--checkpoint", "unused.ckpt", "--model-config", "hybrid_150m_m3_rrg", "--gallery", str(self.gallery), "--data", str(self.data),
                "--published-labels", str(self.dump / "chexbert_labels.json"), "--published-hyps", str(self.dump / "hyps.txt"),
                "--published-refs", str(self.dump / "refs.txt"), "--labeler-url", "http://127.0.0.1:9", "--out", str(self.out),
                "--engine", engine] + list(extra)

    def manifest(self) -> dict:
        return json.loads((self.gallery / "manifest.json").read_text())

    def set_manifest(self, **fields: Any) -> None:
        manifest = self.manifest()
        manifest.update(fields)
        (self.gallery / "manifest.json").write_text(json.dumps(manifest, indent=2))

    def load(self, name: str) -> np.ndarray:
        return np.load(str(self.gallery / name))

    def save(self, name: str, array: np.ndarray) -> None:
        np.save(str(self.gallery / name), array)

    def published(self) -> dict:
        return json.loads((self.dump / "chexbert_labels.json").read_text())

    def set_published(self, payload: dict) -> None:
        (self.dump / "chexbert_labels.json").write_text(json.dumps(payload))

    def lines(self, name: str) -> List[str]:
        return (self.dump / name).read_text().splitlines()

    def report_texts(self) -> List[str]:
        return (self.gallery / "report_texts.txt").read_text().splitlines()

    def gates_json(self) -> dict:
        return json.loads((self.out / "gates.json").read_text())

    def rows(self, gates: Any) -> List[int]:
        """The 50 train rows the job checks."""
        return [int(r) for r in gates.sample_rows(N_IMAGES, 50)]

    def moving_rows(self, count: int) -> List[int]:
        """The first `count` of the 50 test rows checked whose rank is not the same for the reverse of their vector: negating the
        build's vector of such a row makes the live rank and the build's differ. (In this world the build's vectors ARE the live ones.)"""
        gallery = Gallery.open(self.gallery, expect_tower_sha256=None)
        vectors = self.load("test_img_emb.npy")
        rows = [t for t in range(50) if gallery.own_report_rank(vectors[t], t)["rank"] != gallery.own_report_rank(-vectors[t], t)["rank"]]
        assert len(rows) >= count
        return rows[:count]

    def negate_test_img(self, rows: List[int]) -> None:
        vectors = self.load("test_img_emb.npy")
        for row in rows:
            vectors[row] = -vectors[row]
        self.save("test_img_emb.npy", vectors)

    def negate_img(self, rows: List[int]) -> None:
        vectors = self.load("img_emb.npy")
        for row in rows:
            vectors[row] = -vectors[row]               # nowhere near the image's own vector any more
        self.save("img_emb.npy", vectors)


@pytest.fixture
def world(tmp_path, template):
    return World(tmp_path / "world", template)


class Ran:
    def __init__(self, rc, out: str, err: str):
        self.rc, self.out, self.err = rc, out, err

    @property
    def lines(self) -> List[str]:
        return self.out.splitlines()


def run_main(gates: Any, capsys: Any, world: World, *extra: str, labeller: Any = "rules") -> Ran:
    """main() in process, over the tiny engine and (by default) RuleLabeler: its exit code and what it printed."""
    try:
        rc = gates.main(world.argv(*extra), labeller=RuleLabeler() if labeller == "rules" else labeller)
    except SystemExit as stop:
        rc = stop.code
    captured = capsys.readouterr()
    return Ran(rc, captured.out, captured.err)


# ── the pure pieces ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────

def test_the_numbers_the_brief_names_are_the_modules_constants(gates):
    assert (gates.N_SELF, gates.N_RANK, gates.N_LABELLED, gates.LABELER_TIMEOUT_S) == (50, 50, 50, 120)


def test_the_label_order_written_out_in_these_tests_is_the_apps(gates):
    """The tests spell the 14 names once more to build reordered and foreign dumps: that copy, the app's and the driver's agree."""
    assert NAMES == CHEXBERT_14 == list(gates.CHEXBERT_14)


@pytest.mark.parametrize("n_rows", [200, 191462, 51, 50, 7])
def test_the_rows_checked_are_fifty_spread_evenly_from_the_first_to_the_last(gates, n_rows):
    rows = gates.sample_rows(n_rows, 50)
    assert rows.tolist() == np.linspace(0, n_rows - 1, 50).astype(int).tolist()
    assert len(rows) == 50 and rows[0] == 0 and rows[-1] == n_rows - 1 and (np.diff(rows) >= 0).all()


def test_the_engine_is_built_for_the_cpu_with_the_published_arguments_and_the_default_is_eight_threads(gates, world, monkeypatch):
    built = []

    def spy(kind, **kw):
        built.append((kind, kw))
        return build_engine("tiny")

    monkeypatch.setattr(gates, "build_engine", spy)
    minimal = ["--checkpoint", "c.ckpt", "--gallery", "g", "--data", "d", "--published-labels", "l", "--published-hyps", "h",
               "--published-refs", "r", "--labeler-url", "u", "--out", "o"]
    defaults = gates.parse_args(minimal)
    assert (defaults.threads, defaults.engine, defaults.model_config) == (8, "real", "hybrid_150m_m3_rrg")
    gates.run(gates.parse_args(world.argv("--threads", "6", engine="real")), labeller=RuleLabeler())      # as the job: real, its arguments
    assert built == [("real", {"checkpoint": "unused.ckpt", "model_config": "hybrid_150m_m3_rrg", "device": "cpu", "threads": 6})]
    built.clear()
    gates.run(gates.parse_args(world.argv()), labeller=RuleLabeler())
    assert built == [("tiny", {})]


def test_a_missing_required_argument_is_a_usage_error(gates, world):
    argv = world.argv()
    del argv[argv.index("--gallery"):argv.index("--gallery") + 2]
    with pytest.raises(SystemExit) as stop:
        gates.parse_args(argv)
    assert stop.value.code not in (0, None)


# ── a clean run ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

def test_a_clean_run_holds_all_three_gates_and_says_what_it_counted(gates, world, capsys):
    ran = run_main(gates, capsys, world)
    assert ran.rc == 0, ran.out + ran.err
    groups = int(world.manifest()["counts"]["report_groups"])
    lines = ran.lines
    assert lines[0].startswith("[gates] engine device=cpu threads=")
    assert lines[1:] == [
        "[gates] gallery images={} report_rows={} report_groups={} towers_identical=true img_proj_present=false labels_status=done".format(
            N_IMAGES, N_IMAGES + N_TEST, groups),
        "[gates] self_retrieval hits=50 of=50 misses=0", "[gates] own_rank equal=50 dedup_equal=50 of=50 max_diff=0",
        "[gates] labeller pred_equal=50 true_equal=50 of=50", RESULT_CLEAN]
    assert ran.err == ""


def test_gates_json_holds_the_numbers_and_indices_the_job_logged_and_nothing_else(gates, world, capsys):
    assert run_main(gates, capsys, world).rc == 0
    found = world.gates_json()
    assert found["self_retrieval"] == 50 and found["misses"] == [] and found["own_rank_equal"] == 50 and found["labeller_equal"] == 100
    assert found["own_rank_dedup_equal"] == 50 and found["label_names_ok"] is True and found["passed"] is True
    assert (found["self_retrieval_of"], found["own_rank_of"], found["labeller_of"]) == (50, 50, 100)
    assert found["label_names_source"] == "dump" and found["label_mismatch_rows"] == {"hyps": [], "refs": []}
    assert found["own_rank_differs"] == [] and found["own_rank_max_diff"] == 0
    assert found["gallery"] == {"images": N_IMAGES, "report_rows": N_IMAGES + N_TEST, "report_groups": world.manifest()["counts"]["report_groups"],
                                "towers_identical": True, "labels_status": "done"}
    assert found["engine"]["device"] == "cpu" and isinstance(found["engine"]["threads"], int)
    assert set(found["seconds"]) == {"self_retrieval", "own_rank", "labeller"} and all(v >= 0 for v in found["seconds"].values())
    assert sorted(p.name for p in world.out.iterdir()) == ["gates.json"], "nothing else is written, and not into the gallery"


def test_the_job_only_reads_the_gallery_the_dataset_and_the_dump(gates, world, capsys):
    before = {name: snapshot(getattr(world, name)) for name in ("gallery", "data", "dump")}
    assert run_main(gates, capsys, world).rc == 0
    for name, snap in before.items():
        assert snapshot(getattr(world, name)) == snap, name + " was touched (R8)"


# ── 1. self-retrieval through the live upload path ────────────────────────────────────────────────────────────────────────────────────

def spying_engine(monkeypatch: Any, gates: Any) -> Dict[str, Any]:
    """gates.build_engine returns a tiny engine whose preprocess and encode record what they were given."""
    engine = build_engine("tiny")
    seen = {"preprocess": [], "encode": [], "engine": engine}
    preprocess, encode = engine.preprocess, engine.encode

    def spy_preprocess(data):
        seen["preprocess"].append(data)
        return preprocess(data)

    def spy_encode(prepared):
        seen["encode"].append(prepared)
        return encode(prepared)

    engine.preprocess, engine.encode = spy_preprocess, spy_encode
    monkeypatch.setattr(gates, "build_engine", lambda kind, **kw: engine)
    return seen


def test_each_train_image_goes_through_preprocess_and_encode_from_its_file_bytes(gates, world, monkeypatch, capsys):
    seen = spying_engine(monkeypatch, gates)
    assert run_main(gates, capsys, world).rc == 0
    train, test = pd.read_parquet(world.data / "train.parquet"), pd.read_parquet(world.data / "test.parquet")
    wanted = [Path(train["image"].iloc[r]).read_bytes() for r in world.rows(gates)] + [Path(test["image"].iloc[t]).read_bytes() for t in range(50)]
    assert all(type(data) is bytes for data in seen["preprocess"]), "the upload path takes bytes, not a path or an image"
    assert seen["preprocess"] == wanted, "50 train images (evenly spread), then the first 50 test studies"
    assert len(seen["encode"]) == 100


def test_a_train_image_that_does_not_find_itself_is_a_miss_with_its_row_what_was_found_and_the_gap(gates, world, capsys):
    row = world.rows(gates)[10]
    world.negate_img([row])
    ran = run_main(gates, capsys, world)
    assert ran.rc == 1, ran.out
    found = world.gates_json()
    assert found["self_retrieval"] == 49 and found["passed"] is False
    (miss,) = found["misses"]
    assert set(miss) == {"row", "got", "gap"} and miss["row"] == row and miss["got"] != row
    train = pd.read_parquet(world.data / "train.parquet")
    top = Gallery.open(world.gallery, expect_tower_sha256=None).image_neighbors(embed(build_engine("tiny"), [train["image"].iloc[row]])[0], 2)
    assert miss["got"] == top[0]["gallery_row"] and miss["gap"] == pytest.approx(top[0]["similarity"] - top[1]["similarity"], abs=1e-6)
    assert miss["gap"] >= 0
    assert "[gates] self_retrieval hits=49 of=50 misses=1" in ran.lines
    assert [l for l in ran.lines if l.startswith("[gates] miss ")] == ["[gates] miss row={} got={} gap={:.6f}".format(row, miss["got"], miss["gap"])]
    result_at = [i for i, l in enumerate(ran.lines) if l.startswith("RESULT ")][0]
    assert ran.lines.index("ERROR gate self_retrieval=49 expected=50") > result_at, "the verdict follows the numbers"


def test_the_miss_lines_are_at_most_twenty_and_gates_json_has_every_miss(gates, world, capsys):
    world.negate_img(world.rows(gates)[:30])
    ran = run_main(gates, capsys, world)
    assert ran.rc == 1
    assert len([l for l in ran.lines if l.startswith("[gates] miss ")]) == 20
    assert world.gates_json()["self_retrieval"] == 20 and len(world.gates_json()["misses"]) == 30


# ── 2. the live own-rank against the build's ──────────────────────────────────────────────────────────────────────────────────────────

def test_the_live_vector_and_the_builds_embedding_are_each_asked_the_same_question_in_that_order(gates, world, monkeypatch, capsys):
    build = world.load("test_img_emb.npy")
    world.save("test_img_emb.npy", -build)                                  # the build's vectors are now told apart from the live ones
    queries, original = [], Gallery.own_report_rank

    def spy(self, query, test_row):
        queries.append((np.array(query), test_row))
        return original(self, query, test_row)

    monkeypatch.setattr(Gallery, "own_report_rank", spy)
    assert run_main(gates, capsys, world).rc == 0
    assert [row for _, row in queries] == [t for t in range(50) for _ in range(2)]
    live = embed(build_engine("tiny"), pd.read_parquet(world.data / "test.parquet")["image"].iloc[:50])
    for t in range(50):
        assert np.allclose(queries[2 * t][0], live[t], atol=1e-6), "first: the live vector of the test image"
        assert np.allclose(queries[2 * t + 1][0], -build[t], atol=1e-6), "then: the build's own embedding of it"


def test_a_rank_that_moves_is_counted_and_recorded_and_does_not_fail_the_job(gates, world, capsys):
    rows = world.moving_rows(2)
    world.negate_test_img(rows)
    ran = run_main(gates, capsys, world)
    assert ran.rc == 0, ran.out
    found = world.gates_json()
    assert found["own_rank_equal"] == 48 and found["passed"] is True
    differs = found["own_rank_differs"]
    assert [d["test_row"] for d in differs] == rows
    assert all(set(d) == {"test_row", "rank", "rank_dedup"} and d["rank"][0] != d["rank"][1] for d in differs)
    assert found["own_rank_max_diff"] == max(abs(d["rank"][0] - d["rank"][1]) for d in differs) > 0
    assert "[gates] own_rank equal=48 dedup_equal={} of=50 max_diff={}".format(found["own_rank_dedup_equal"], found["own_rank_max_diff"]) in ran.lines


def test_rank_and_rank_dedup_are_counted_apart_and_neither_gates(gates, world, monkeypatch, capsys):
    original, calls = Gallery.own_report_rank, []

    def scripted(self, query, test_row):
        result = original(self, query, test_row)
        calls.append(test_row)
        if calls.count(test_row) == 1:                                      # the first call of a row is the live vector's
            if test_row in (3, 4):
                result = dict(result, rank=result["rank"] + 1)              # rank moves, rank_dedup does not
            if test_row in (4, 5, 6):
                result = dict(result, rank_dedup=result["rank_dedup"] + 2)
        return result

    monkeypatch.setattr(Gallery, "own_report_rank", scripted)
    ran = run_main(gates, capsys, world)
    assert ran.rc == 0, ran.out
    found = world.gates_json()
    assert (found["own_rank_equal"], found["own_rank_dedup_equal"], found["own_rank_max_diff"]) == (48, 47, 1), "max_diff is of rank"
    assert "[gates] own_rank equal=48 dedup_equal=47 of=50 max_diff=1" in ran.lines
    differs = {d["test_row"]: d for d in found["own_rank_differs"]}
    assert sorted(differs) == [3, 4, 5, 6], "a row is listed when either of its two ranks moved"
    assert differs[3]["rank"][0] - differs[3]["rank"][1] == 1 and differs[3]["rank_dedup"][0] == differs[3]["rank_dedup"][1], "[live, build]"
    assert differs[5]["rank"][0] == differs[5]["rank"][1] and differs[5]["rank_dedup"][0] - differs[5]["rank_dedup"][1] == 2


# ── 3. the labeller against the published labels ──────────────────────────────────────────────────────────────────────────────────────

class Spy(RuleLabeler):
    def __init__(self):
        self.calls = []

    def label(self, texts):
        self.calls.append(list(texts))
        return super().label(texts)


def test_the_labeller_gets_the_first_fifty_hyps_in_one_call_and_the_first_fifty_refs_in_another(gates, world, capsys):
    spy = Spy()
    assert run_main(gates, capsys, world, labeller=spy).rc == 0
    assert spy.calls == [world.lines("hyps.txt")[:50], world.lines("refs.txt")[:50]]
    assert all(len(call) == 50 for call in spy.calls), "LabelerClient's limit is 64 texts per call"


def test_the_service_client_is_the_apps_with_the_two_minute_timeout_the_brief_names(gates, world, monkeypatch, capsys):
    made = []

    class Client(RuleLabeler):
        def __init__(self, url, timeout=10.0):
            made.append((url, timeout))

    monkeypatch.setattr(gates, "LabelerClient", Client)
    assert run_main(gates, capsys, world, labeller=None).rc == 0
    assert made == [("http://127.0.0.1:9", 120)]


class Flipping(RuleLabeler):
    """RuleLabeler, but one label of one row is wrong in call 0 (the hyps) and in call 1 (the refs)."""

    def __init__(self, hyps_row=None, refs_row=None):
        self.rows, self.calls = (hyps_row, refs_row), 0

    def label(self, texts):
        rows = super().label(texts)
        row = self.rows[self.calls]
        self.calls += 1
        if row is not None:
            rows[row][0] = 1 - rows[row][0]
        return rows


def test_a_label_that_differs_is_counted_by_row_and_kind_and_fails_the_gate(gates, world, capsys):
    ran = run_main(gates, capsys, world, labeller=Flipping(hyps_row=3, refs_row=7))
    assert ran.rc == 1, ran.out
    found = world.gates_json()
    assert found["labeller_equal"] == 98 and found["label_mismatch_rows"] == {"hyps": [3], "refs": [7]} and found["passed"] is False
    assert "[gates] labeller pred_equal=49 true_equal=49 of=50" in ran.lines
    assert "ERROR gate labeller_equal=98 expected=100" in ran.lines
    assert [l for l in ran.lines if l.startswith("RESULT ")] == [
        'RESULT {"self_retrieval":50,"own_rank_equal":50,"own_rank_dedup_equal":50,"labeller_equal":98,"label_names_ok":true}']


def test_every_one_of_the_fourteen_labels_is_compared(gates, world, capsys):
    payload = world.published()
    for column in range(14):
        fresh = json.loads(json.dumps(payload))
        fresh["y_pred"][0][column] = 1 - fresh["y_pred"][0][column]
        world.set_published(fresh)
        ran = run_main(gates, capsys, world)
        assert ran.rc == 1 and world.gates_json()["labeller_equal"] == 99, column


def test_only_the_first_fifty_rows_of_the_dump_are_compared(gates, world, capsys):
    payload = world.published()
    payload["y_pred"][55][0] = 1 - payload["y_pred"][55][0]                  # row 55 of 60 is beyond the fifty
    world.set_published(payload)
    assert run_main(gates, capsys, world).rc == 0


def test_published_labels_with_a_label_names_key_in_the_apps_order_need_no_note(gates, world, capsys):
    ran = run_main(gates, capsys, world)
    assert ran.rc == 0 and not [l for l in ran.lines if l.startswith("===")]
    assert world.gates_json()["label_names_source"] == "dump"


def test_published_labels_without_label_names_are_read_in_chexbert_14_order_and_a_note_says_so(gates, world, capsys):
    payload = world.published()
    del payload["label_names"]
    world.set_published(payload)
    ran = run_main(gates, capsys, world)
    assert ran.rc == 0, ran.out
    assert ran.lines.count("=== note: chexbert_labels.json has no label_names key: CHEXBERT_14 order assumed ===") == 1
    assert world.gates_json()["label_names_source"] == "assumed" and world.gates_json()["label_names_ok"] is True
    assert RESULT_CLEAN in ran.lines


def test_published_labels_naming_the_same_labels_in_another_order_are_reordered_by_name_and_a_note_says_so(gates, world, capsys):
    perm = list(reversed(range(14)))
    payload = world.published()
    payload["label_names"] = [NAMES[i] for i in perm]
    for key in ("y_true", "y_pred"):
        payload[key] = [[row[i] for i in perm] for row in payload[key]]
    world.set_published(payload)
    ran = run_main(gates, capsys, world)
    assert ran.rc == 0, ran.out
    assert "=== note: chexbert_labels.json label_names reordered to the CHEXBERT_14 order ===" in ran.lines
    assert world.gates_json()["label_names_source"] == "reordered" and world.gates_json()["labeller_equal"] == 100
    assert RESULT_CLEAN in ran.lines


@pytest.mark.parametrize("names", [[n + "!" if i == 0 else n for i, n in enumerate(NAMES)], NAMES[:-1], NAMES + ["Extra"], "not a list", None, 7,
                                   [NAMES[0]] * 14])
def test_published_labels_naming_other_labels_fail_the_names_gate_and_say_so(gates, world, capsys, names):
    payload = world.published()
    payload["label_names"] = names
    world.set_published(payload)
    ran = run_main(gates, capsys, world)
    assert ran.rc == 1, ran.out
    assert world.gates_json()["label_names_ok"] is False and world.gates_json()["label_names_source"] == "foreign"
    assert "ERROR gate label_names_ok=false" in ran.lines
    assert [l for l in ran.lines if l.startswith("RESULT ")][0].endswith('"label_names_ok":false}')


DAMAGED_DUMPS = ["not json", "{}", '{"y_true": []}', '{"y_true": [[0]], "y_pred": [[0]]}', '{"y_true": 3, "y_pred": 4}',
                 '{"y_true": [[0, 1], [0]], "y_pred": [[0, 1], [0]]}', "[1, 2]", '{"y_true": [["a"], ["b"]], "y_pred": [["a"], ["b"]]}']


@pytest.mark.parametrize("damage", DAMAGED_DUMPS)
def test_published_labels_that_cannot_be_read_as_two_matrices_of_fourteen_are_a_refusal(gates, world, capsys, damage):
    (world.dump / "chexbert_labels.json").write_text(damage)
    ran = run_main(gates, capsys, world)
    assert ran.rc == 1, ran.out
    assert ran.lines[-1] == "ERROR published unreadable"
    assert not [l for l in ran.lines if l.startswith("RESULT")], "nothing is claimed from a file that was not read"


def test_a_published_file_that_is_missing_is_a_refusal_and_not_a_traceback_in_the_log(gates, world, capsys):
    (world.dump / "chexbert_labels.json").unlink()
    ran = run_main(gates, capsys, world)
    assert ran.rc == 1 and ran.lines[-1] == "ERROR published unreadable"


def test_a_dump_whose_files_disagree_in_length_is_a_refusal_with_the_four_counts(gates, world, capsys):
    (world.dump / "hyps.txt").write_text("\n".join(world.lines("hyps.txt") + ["one more", "and another"]) + "\n")
    ran = run_main(gates, capsys, world)
    assert ran.rc == 1
    assert ran.lines[-1] == "ERROR published shape hyps=62 y_pred=60 refs=60 y_true=60"


class Down(RuleLabeler):
    def __init__(self, message):
        self.message = message

    def label(self, texts):
        raise LabelerUnavailable(self.message)


def test_a_labeller_that_cannot_be_trusted_is_a_refusal_after_the_other_two_checks_have_said_their_numbers(gates, world, capsys):
    ran = run_main(gates, capsys, world, labeller=Down("labeller request failed: URLError"))
    assert ran.rc == 1
    assert ran.lines[-1] == "ERROR labeller unavailable"
    assert "[gates] self_retrieval hits=50 of=50 misses=0" in ran.lines and not [l for l in ran.lines if l.startswith("RESULT")]
    found = world.gates_json()
    assert found["error"] == "labeller unavailable" and found["passed"] is False and found["self_retrieval"] == 50, "what was measured is kept"


def test_a_labeller_with_another_label_order_is_named_as_that(gates, world, capsys):
    ran = run_main(gates, capsys, world, labeller=Down("label order mismatch: the service reported 14 label names, expected 14"))
    assert ran.rc == 1 and ran.lines[-1] == "ERROR labeller order mismatch"


# ── the real client against the service's HTTP surface (a stdlib stand-in, on a loopback port) ──────────────────────────────────────

class FakeService:
    """app/labeler.py's two routes over the standard library, in a thread: /healthz and /label, with what a test has to provoke."""

    def __init__(self, names: Optional[list] = None, status: int = 200, body: Any = None, raw: Optional[bytes] = None):
        service = self
        self.names, self.status, self.body, self.raw, self.requests, self.closed = names, status, body, raw, [], False

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, code, data, kind="application/json"):
                self.send_response(code)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                self._send(200 if self.path == "/healthz" else 404, b'{"status": "ok"}')

            def do_POST(self):
                texts = json.loads(self.rfile.read(int(self.headers["Content-Length"])))["texts"]
                service.requests.append(texts)
                if service.status != 200:
                    return self._send(service.status, b'{"detail": "x"}')
                if service.raw is not None:
                    return self._send(200, service.raw, "text/html")
                payload = service.body if service.body is not None else {"label_names": service.names or list(CHEXBERT_14),
                                                                         "labels": RuleLabeler().label(texts)}
                self._send(200, json.dumps(payload).encode())

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = "http://127.0.0.1:{}".format(self.server.server_address[1])
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self.server.shutdown()
            self.server.server_close()
            self.thread.join(5)


@pytest.fixture
def service():
    made = []

    def make(**kw):
        made.append(FakeService(**kw))
        return made[-1]

    yield make
    for each in made:
        each.close()


def run_against(gates: Any, capsys: Any, world: World, url: str) -> Ran:
    argv = world.argv()
    argv[argv.index("--labeler-url") + 1] = url
    try:
        rc = gates.main(argv)
    except SystemExit as stop:
        rc = stop.code
    captured = capsys.readouterr()
    return Ran(rc, captured.out, captured.err)


def test_over_http_the_service_and_the_published_labels_agree_100_of_100(gates, world, capsys, service):
    fake = service()
    ran = run_against(gates, capsys, world, fake.url)
    assert ran.rc == 0, ran.out + ran.err
    assert RESULT_CLEAN in ran.lines
    assert [len(r) for r in fake.requests] == [50, 50] and fake.requests[0] == world.lines("hyps.txt")[:50]


def test_over_http_a_service_that_names_another_label_order_is_refused_by_the_client_and_named(gates, world, capsys, service):
    swapped = list(CHEXBERT_14)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    ran = run_against(gates, capsys, world, service(names=swapped).url)
    assert ran.rc == 1 and ran.lines[-1] == "ERROR labeller order mismatch"


@pytest.mark.parametrize("kind", ["http_500", "http_422", "html_instead_of_json", "json_that_is_not_an_object", "too_few_rows", "no_server"])
def test_over_http_every_other_way_a_service_can_fail_is_labeller_unavailable_and_never_order(gates, world, capsys, service, kind):
    options = {"http_500": {"status": 500}, "http_422": {"status": 422}, "html_instead_of_json": {"raw": b"<html>oops</html>"},
               "json_that_is_not_an_object": {"body": [1, 2]}, "too_few_rows": {"body": {"label_names": list(CHEXBERT_14), "labels": []}},
               "no_server": {}}[kind]
    fake = service(**options)
    if kind == "no_server":
        fake.close()
    ran = run_against(gates, capsys, world, fake.url)
    assert ran.rc == 1 and ran.lines[-1] == "ERROR labeller unavailable", ran.lines


# ── the gallery ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

def test_the_gallery_is_opened_with_the_engines_own_tower_hash(gates, world, monkeypatch, capsys):
    seen = spying_engine(monkeypatch, gates)
    given, original = [], Gallery.open.__func__

    def spy(cls, root, expect_tower_sha256=None):
        given.append(expect_tower_sha256)
        return original(cls, root, expect_tower_sha256)

    monkeypatch.setattr(Gallery, "open", classmethod(spy))
    assert run_main(gates, capsys, world).rc == 0
    assert given == [seen["engine"].tower_sha256()]


def test_a_gallery_built_with_another_tower_is_refused_by_name_before_a_single_image_is_read(gates, world, monkeypatch, capsys):
    world.set_manifest(tower_sha256="0" * 64)
    seen = spying_engine(monkeypatch, gates)
    ran = run_main(gates, capsys, world)
    assert ran.rc == 1
    assert ran.lines[-1] == "ERROR gallery tower mismatch" and not [l for l in ran.lines if l.startswith("RESULT")]
    assert seen["preprocess"] == [] and seen["encode"] == [], "self-retrieval across embedding spaces is never started"
    assert world.gates_json()["error"] == "gallery tower mismatch" and world.gates_json()["passed"] is False


@pytest.mark.parametrize("how", ["gate_not_equal", "img_proj", "no_manifest", "no_file"])
def test_any_other_refusal_of_gallery_open_is_one_plain_line_that_is_not_the_tower_one(gates, world, capsys, how):
    if how == "gate_not_equal":
        world.set_manifest(gate_rk=dict(world.manifest()["gate_rk"], equal=False))
    elif how == "img_proj":
        world.set_manifest(img_proj_present=True)
    elif how == "no_manifest":
        (world.gallery / "manifest.json").unlink()
    else:
        (world.gallery / "txt_emb.npy").unlink()
    ran = run_main(gates, capsys, world)
    assert ran.rc == 1 and ran.lines[-1] == "ERROR gallery refused", ran.lines


def test_the_refusal_keeps_the_gallerys_own_words_in_stderr_for_the_raw_log_and_never_in_the_job_log(gates, world, capsys):
    world.set_manifest(img_proj_present=True)
    ran = run_main(gates, capsys, world)
    assert "img_proj_present" in ran.err and "img_proj_present" not in ran.out.replace("img_proj_present=false", "")


def test_the_gallery_line_says_towers_identical_false_and_labels_pending_when_the_manifest_does(gates, world, capsys):
    world.set_manifest(towers_identical=False, labels_status="pending")
    ran = run_main(gates, capsys, world)
    assert ran.rc == 0, ran.out
    assert [l for l in ran.lines if l.startswith("[gates] gallery ")][0].endswith("towers_identical=false img_proj_present=false labels_status=pending")
    assert world.gates_json()["gallery"]["labels_status"] == "pending" and world.gates_json()["gallery"]["towers_identical"] is False


# ── the data ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("split, n", [("train", 199), ("test", 59)])
def test_a_dataset_that_does_not_have_the_galleries_rows_is_refused_with_the_four_counts(gates, world, capsys, split, n):
    path = world.data / "{}.parquet".format(split)
    pd.read_parquet(path).iloc[:n].to_parquet(path)
    ran = run_main(gates, capsys, world)
    assert ran.rc == 1
    assert ran.lines[-1] == "ERROR data rows disagree train={} gallery_train={} test={} gallery_test={}".format(
        n if split == "train" else N_IMAGES, N_IMAGES, n if split == "test" else N_TEST, N_TEST)
    assert [l for l in ran.lines if l.startswith("[gates] self_retrieval")] == [], "nothing is compared across misaligned rows"


def test_only_the_image_column_is_read_of_each_parquet_file(gates, world, monkeypatch, capsys):
    seen, original = [], pd.read_parquet

    def spy(path, *args, **kwargs):
        seen.append((Path(path).name, kwargs.get("columns")))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", spy)
    assert run_main(gates, capsys, world).rc == 0
    assert [s for s in seen if s[0] in ("train.parquet", "test.parquet")] == [("train.parquet", ["image"]), ("test.parquet", ["image"])]


def test_a_missing_file_is_a_failure_by_class_name_and_a_traceback_in_the_raw_log(gates, world, capsys):
    (world.gallery / "test_img_emb.npy").unlink()
    ran = run_main(gates, capsys, world)
    assert ran.rc == 1
    assert ran.lines[-1] == "ERROR failed FileNotFoundError"
    assert "Traceback (most recent call last)" in ran.err and "Traceback" not in ran.out
    assert world.gates_json()["error"] == "failed FileNotFoundError"


# ── what is printed (R7) and what gates.json holds ────────────────────────────────────────────────────────────────────────────────────

ERROR_CODES = ["gallery tower mismatch", "gallery refused", "data rows disagree", "published unreadable", "published shape",
               "labeller order mismatch", "labeller unavailable"]

# The one place the shapes of a line the script prints are written down in the tests. The wrapper's allowlist (GATES_SHAPES, one anchored
# pattern per line) has to accept exactly these, and everything the script prints has to fit them: digits are bounded and nothing in a
# shape is free text, so an id or a piece of a report cannot ride on a line of the log.
GATES_SHAPE_LIST = [
    r"\[gates\] engine device=(cpu|cuda|cuda:[0-9]{1,2}) threads=[0-9]{1,3}",
    r"\[gates\] gallery images=[0-9]{1,7} report_rows=[0-9]{1,7} report_groups=[0-9]{1,7} towers_identical=(true|false) img_proj_present=false "
    r"labels_status=(done|pending)",
    r"\[gates\] self_retrieval hits=[0-9]{1,3} of=[0-9]{1,3} misses=[0-9]{1,3}",
    r"\[gates\] miss row=[0-9]{1,7} got=[0-9]{1,7} gap=[0-9]\.[0-9]{6}",
    r"\[gates\] own_rank equal=[0-9]{1,3} dedup_equal=[0-9]{1,3} of=[0-9]{1,3} max_diff=[0-9]{1,7}",
    r"\[gates\] labeller pred_equal=[0-9]{1,3} true_equal=[0-9]{1,3} of=[0-9]{1,3}",
    r'RESULT \{"self_retrieval":[0-9]{1,3},"own_rank_equal":[0-9]{1,3},"own_rank_dedup_equal":[0-9]{1,3},"labeller_equal":[0-9]{1,3},'
    r'"label_names_ok":(true|false)\}',
    r"ERROR (gallery tower mismatch|gallery refused|published unreadable|labeller order mismatch|labeller unavailable)",
    r"ERROR (data rows disagree|published shape)( [a-z_]{1,20}=[0-9]{1,7}){1,4}",
    r"ERROR gate (self_retrieval|labeller_equal)=[0-9]{1,3} expected=[0-9]{1,3}",
    r"ERROR gate label_names_ok=false",
    r"ERROR failed [A-Za-z_][A-Za-z0-9_]{0,59}",
    r"=== note: chexbert_labels\.json has no label_names key: CHEXBERT_14 order assumed ===",
    r"=== note: chexbert_labels\.json label_names reordered to the CHEXBERT_14 order ===",
]
GATES_SHAPES = re.compile(r"^(" + "|".join(GATES_SHAPE_LIST) + r")$")


def wrapper_patterns() -> List[str]:
    """GATES_SHAPES as the wrapper holds it: one anchored pattern per line of one single-quoted assignment (grep -E reads a newline as 'or')."""
    found = re.search(r"^GATES_SHAPES='(.*?)'$", (REPO_ROOT / "scripts" / SH).read_text(), re.M | re.S)
    assert found, SH + " has no GATES_SHAPES assignment"
    return found.group(1).split("\n")


def grep_passes(lines: List[str]) -> List[str]:
    """The lines the wrapper's filter lets through: those very patterns, through the very grep, as the job runs them."""
    done = subprocess.run(["grep", "-aE", "\n".join(wrapper_patterns())], input="\n".join(lines) + "\n", capture_output=True, text=True)
    assert done.returncode in (0, 1), done.stderr
    return done.stdout.splitlines()


GATES_PASS = [
    "[gates] engine device=cpu threads=8", "[gates] engine device=cuda threads=1", "[gates] engine device=cuda:0 threads=128",
    "[gates] gallery images=191462 report_rows=194125 report_groups=163021 towers_identical=true img_proj_present=false labels_status=done",
    "[gates] gallery images=1 report_rows=2 report_groups=1 towers_identical=false img_proj_present=false labels_status=pending",
    "[gates] self_retrieval hits=50 of=50 misses=0", "[gates] self_retrieval hits=0 of=50 misses=50",
    "[gates] miss row=0 got=191461 gap=0.000012", "[gates] miss row=123456 got=7 gap=1.999999",
    "[gates] own_rank equal=48 dedup_equal=50 of=50 max_diff=2", "[gates] own_rank equal=0 dedup_equal=0 of=50 max_diff=2662",
    "[gates] labeller pred_equal=50 true_equal=50 of=50",
    RESULT_CLEAN, 'RESULT {"self_retrieval":48,"own_rank_equal":0,"own_rank_dedup_equal":7,"labeller_equal":97,"label_names_ok":false}',
    "ERROR gallery tower mismatch", "ERROR gallery refused", "ERROR published unreadable", "ERROR labeller order mismatch", "ERROR labeller unavailable",
    "ERROR data rows disagree train=191461 gallery_train=191462 test=2663 gallery_test=2663", "ERROR published shape hyps=62 y_pred=60 refs=60 y_true=60",
    "ERROR published shape y_pred=3", "ERROR gate self_retrieval=49 expected=50", "ERROR gate labeller_equal=98 expected=100",
    "ERROR gate label_names_ok=false", "ERROR failed FileNotFoundError", "ERROR failed OutOfMemoryError",
    "=== note: chexbert_labels.json has no label_names key: CHEXBERT_14 order assumed ===",
    "=== note: chexbert_labels.json label_names reordered to the CHEXBERT_14 order ===",
]
GATES_WITHHELD = [
    "[gates] study_id=12345678", "[gates] The heart is mildly enlarged and there is a small pleural effusion.",
    "[gates] engine device=cpu threads=8 study_id=12345678", "[gates] engine device=tpu threads=8", "[gates] engine device=cuda:123 threads=8",
    "[gates] engine device=cpu threads=", "[gates] engine device=cpu threads=1234", "[gates] /sc/home/someone/images/p10/leak.jpg",
    "[gates] gallery images=12345678 report_rows=2 report_groups=1 towers_identical=true img_proj_present=false labels_status=done",
    "[gates] gallery images=1 report_rows=2 report_groups=1 towers_identical=true img_proj_present=true labels_status=done",
    "[gates] gallery images=1 report_rows=2 report_groups=1 towers_identical=maybe img_proj_present=false labels_status=done",
    "[gates] gallery images=1 report_rows=2 report_groups=1 towers_identical=true img_proj_present=false labels_status=partial",
    "[gates] gallery build_id=g13d_m3_v1 images=1 report_rows=2 report_groups=1 towers_identical=true img_proj_present=false labels_status=done",
    "[gates] self_retrieval hits=50 of=50", "[gates] self_retrieval hits=50 of=50 misses=0 row=1", "[gates] self_retrieval hits=-1 of=50 misses=0",
    "[gates] miss row=0 got=1 gap=12.000000", "[gates] miss row=0 got=1 gap=0.5", "[gates] miss row=0 got=1 gap=nan", "[gates] miss row=0 got=1",
    "[gates] miss row=12345678 got=1 gap=0.000001", "[gates] miss row=0 got=1 gap=0.000001 study_id=1",
    "[gates] own_rank equal=48 dedup_equal=50 of=50", "[gates] own_rank equal=48 dedup_equal=50 of=50 max_diff=-1",
    "[gates] labeller pred_equal=50 true_equal=50", "[gates] labeller pred_equal=50 true_equal=50 of=50 extra",
    "[gates]  self_retrieval hits=50 of=50 misses=0", " [gates] self_retrieval hits=50 of=50 misses=0", "[gates] self_retrieval hits=50 of=50 misses=0 ",
    "[gates] self_retrieval hits=50 of=50 misses=0\t", "[Gates] self_retrieval hits=50 of=50 misses=0", "[labels] self_retrieval hits=50 of=50 misses=0",
    'RESULT {"self_retrieval":50}', RESULT_CLEAN + " ", RESULT_CLEAN.replace(",", ", "), RESULT_CLEAN.replace(":50", ":-50", 1),
    RESULT_CLEAN.replace("true", "yes"), RESULT_CLEAN[:-1] + ',"study_id":50000001}', 'RESULT {"study_id":50000001}',
    'RESULT {"self_retrieval":50,"own_rank_equal":50,"own_rank_dedup_equal":50,"labeller_equal":1000,"label_names_ok":true}',
    "ERROR the heart is mildly enlarged", "ERROR gallery tower mismatch /sc/home/someone", "ERROR gallery tower mismatch study=12345678",
    "ERROR gallery refused: manifest.json is missing", "ERROR data rows disagree", "ERROR data rows disagree train=12345678",
    "ERROR data rows disagree train=1 a=1 b=2 c=3 d=4 e=5", "ERROR published shape effusion=x", "ERROR gate self_retrieval=49",
    "ERROR gate self_retrieval=49 expected=50 effusion", "ERROR gate studies=49 expected=50", "ERROR gate label_names_ok=true",
    "ERROR failed", "ERROR failed the heart", "ERROR failed " + "A" * 61, "ERROR", "ERROR ", "ERROR labeller unavailable reason=http",
    "=== note: chexbert_labels.json has no label_names key: CHEXBERT_14 order assumed === extra",
    "=== FAKE MIMIC TEXT /sc/home/someone/images/p10/img.jpg ===", "=== note: something else ===",
    "Traceback (most recent call last):", "FileNotFoundError: [Errno 2] No such file or directory: '/sc/x/p10/s50000001.jpg'", "",
]


def test_the_wrappers_allowlist_passes_the_known_lines_and_withholds_everything_else():
    assert grep_passes(GATES_PASS) == GATES_PASS
    assert grep_passes(GATES_WITHHELD) == []
    for line in GATES_PASS + GATES_WITHHELD:             # the shapes written in the tests and the patterns in the wrapper say the same
        assert bool(GATES_SHAPES.match(line)) == (grep_passes([line]) == [line]), line
    assert [bool(GATES_SHAPES.match(line)) for line in GATES_PASS] == [True] * len(GATES_PASS)


def test_the_wrappers_allowlist_is_one_anchored_pattern_per_shape_with_nothing_free_in_it():
    patterns = wrapper_patterns()
    assert patterns == ["^" + shape + "$" for shape in GATES_SHAPE_LIST], "the wrapper's patterns are the tests' shapes, anchored, in the same order"
    assert "" not in patterns, "an empty pattern matches every line"
    for pattern in patterns:
        assert pattern.startswith(("^\\[gates\\] ", "^RESULT ", "^ERROR ", "^=== note: ")) and pattern.endswith("$"), pattern
        bare = re.sub(r"\[[^\]]*\]", "", re.sub(r"\\.", "", pattern))        # without escaped characters, then bracket expressions
        assert not re.search(r"[.*+?]", bare), "a wildcard, an unbounded repeat or an optional part in " + pattern
        assert not re.search(r"\\[sSwWdD]", pattern), pattern


def test_the_error_codes_of_the_script_the_tests_and_the_wrapper_are_the_same_set(gates):
    assert sorted(gates.ERROR_CODES) == sorted(ERROR_CODES)
    tree = ast.parse(SCRIPT.read_text())
    raised = {n.args[0].value for n in ast.walk(tree)
              if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "Refused" and n.args and isinstance(n.args[0], ast.Constant)}
    assert raised <= set(ERROR_CODES), "every raise names a code in the list"
    literals = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    for code in ERROR_CODES:
        assert literals.count(code) >= 2, "{} is in ERROR_CODES and never raised: a code the wrapper allows and nothing prints".format(code)
    lines = [p for p in wrapper_patterns() if re.match(r"\^ERROR \([a-z |]+\)", p)]
    named = set()
    for line in lines:
        named |= set(re.match(r"\^ERROR \(([a-z |]+)\)", line).group(1).split("|"))
    assert named == set(ERROR_CODES), "the wrapper allows exactly the codes the script has"


def run_scenarios(gates: Any, capsys: Any, tmp_path: Path, template: Path) -> List[str]:
    """Everything the script prints, over the runs that matter: one line per line."""
    printed = []

    def fresh(name):
        return World(tmp_path / name, template)

    def go(world, *extra, labeller="rules"):
        printed.extend(run_main(gates, capsys, world, *extra, labeller=labeller).lines)

    go(fresh("clean"))
    miss = fresh("miss")
    miss.negate_img(miss.rows(gates)[:25])
    go(miss)
    moved = fresh("moved")
    moved.negate_test_img(moved.moving_rows(1))
    go(moved)
    go(fresh("flip"), labeller=Flipping(hyps_row=3, refs_row=7))
    nameless = fresh("nameless")
    payload = nameless.published()
    del payload["label_names"]
    nameless.set_published(payload)
    go(nameless)
    permuted = fresh("permuted")
    perm = list(reversed(range(14)))
    payload = permuted.published()
    payload["label_names"] = [NAMES[i] for i in perm]
    for key in ("y_true", "y_pred"):
        payload[key] = [[row[i] for i in perm] for row in payload[key]]
    permuted.set_published(payload)
    go(permuted)
    foreign = fresh("foreign")
    payload = foreign.published()
    payload["label_names"] = ["Something else"] * 14
    foreign.set_published(payload)
    go(foreign)
    tower = fresh("tower")
    tower.set_manifest(tower_sha256="0" * 64)
    go(tower)
    refused = fresh("refused")
    refused.set_manifest(img_proj_present=True)
    go(refused)
    rows = fresh("rows")
    pd.read_parquet(rows.data / "train.parquet").iloc[:199].to_parquet(rows.data / "train.parquet")
    go(rows)
    unreadable = fresh("unreadable")
    (unreadable.dump / "chexbert_labels.json").write_text("not json")
    go(unreadable)
    shape = fresh("shape")
    (shape.dump / "hyps.txt").write_text("\n".join(shape.lines("hyps.txt") + ["extra"]) + "\n")
    go(shape)
    go(fresh("down"), labeller=Down("labeller request failed: URLError"))
    go(fresh("order"), labeller=Down("label order mismatch: the service reported 14 label names, expected 14"))
    missing = fresh("missing")
    (missing.gallery / "test_img_emb.npy").unlink()
    go(missing)
    return printed


def test_everything_the_script_prints_fits_a_shape_of_the_allowlist_and_each_shape_is_exercised(gates, capsys, tmp_path, template):
    printed = run_scenarios(gates, capsys, tmp_path, template)
    assert printed and all(GATES_SHAPES.match(l) for l in printed), [l for l in printed if not GATES_SHAPES.match(l)]
    assert grep_passes(printed) == printed, [l for l in printed if l not in grep_passes(printed)]
    assert [s for s in GATES_SHAPE_LIST if not any(re.fullmatch(s, l) for l in printed)] == []
    printed_codes = {l[len("ERROR "):] for l in printed if l.startswith("ERROR ")}
    for code in ERROR_CODES:
        assert any(c == code or c.startswith(code + " ") for c in printed_codes), "never printed: " + code
    # and no byte of any of them could be an id or a report: each is cut by the allowlist at the first thing it does not know. (An
    # exception's class name, at the end of a line, may be a longer name: the one place a letter may follow.)
    for line in printed:
        assert grep_passes([line + " study_id=12345678"]) == [] and grep_passes([line + " x"]) == [], line
        if not line.startswith("ERROR failed "):
            assert grep_passes([line + "x"]) == [], line


def test_the_script_prints_no_text_no_id_and_no_path(gates, capsys, tmp_path, template):
    """Over every run above: no piece of a report, no study id, none of the temp directory, and no run of 8 digits (an id)."""
    printed = "\n".join(run_scenarios(gates, capsys, tmp_path, template))
    assert str(tmp_path) not in printed and not re.search(r"[0-9]{8}", printed)
    for text in World(tmp_path / "probe", template).report_texts():
        assert text[:30] not in printed
    assert "50000" not in printed and "56000" not in printed and "gallery/" not in printed and ".jpg" not in printed


def test_gates_json_holds_only_numbers_booleans_lists_and_a_few_fixed_words(gates, capsys, tmp_path, template):
    """R7: the file may be read where the log may not be, so it carries no id, no text and no path: every string in it is one of the
    fixed words, whatever the run, including the runs that fail."""
    fixed = {"dump", "assumed", "reordered", "foreign", "done", "pending", "cpu"} | set(ERROR_CODES) | {"failed FileNotFoundError"}
    known = {"self_retrieval", "self_retrieval_of", "misses", "own_rank_equal", "own_rank_dedup_equal", "own_rank_of", "own_rank_max_diff",
             "own_rank_differs", "labeller_equal", "labeller_of", "label_mismatch_rows", "label_names_ok", "label_names_source", "gallery",
             "engine", "seconds", "passed", "error",
             "row", "got", "gap",                                   # a miss
             "test_row", "rank", "rank_dedup",                      # a rank that differs
             "hyps", "refs",                                        # label_mismatch_rows
             "images", "report_rows", "report_groups", "towers_identical", "labels_status",     # gallery
             "device", "threads",                                   # engine
             "own_rank", "labeller"}                                # seconds (self_retrieval is above)
    worlds = []
    for name in ("clean", "miss", "moved", "tower", "missing"):
        world = World(tmp_path / name, template)
        if name == "miss":
            world.negate_img(world.rows(gates)[3:4])
        elif name == "moved":
            world.negate_test_img(world.moving_rows(1))
        elif name == "tower":
            world.set_manifest(tower_sha256="0" * 64)
        elif name == "missing":
            (world.gallery / "test_img_emb.npy").unlink()
        gates.main(world.argv(), labeller=RuleLabeler())
        worlds.append(world)
    capsys.readouterr()

    def walk(value, path=""):
        if isinstance(value, dict):
            for key, inner in value.items():
                assert key in known, (path, key)
                yield from walk(inner, path + "/" + key)
        elif isinstance(value, list):
            for inner in value:
                yield from walk(inner, path)
        else:
            yield path, value

    for world in worlds:
        for path, value in walk(world.gates_json()):
            if isinstance(value, str):
                assert value in fixed, (path, value)
            else:
                assert isinstance(value, (bool, int, float)) or value is None, (path, value)
        text = (world.out / "gates.json").read_text()
        assert str(tmp_path) not in text and "50000" not in text and ".jpg" not in text


# ── main(): exit codes, and the CLI in a fresh interpreter ────────────────────────────────────────────────────────────────────────────

def test_main_exits_0_when_the_three_gates_hold_and_1_for_each_one_that_does_not(gates, world, capsys, tmp_path, template):
    assert run_main(gates, capsys, world).rc == 0
    assert run_main(gates, capsys, World(tmp_path / "flipped", template), labeller=Flipping(hyps_row=0)).rc == 1
    foreign = World(tmp_path / "foreign", template)
    payload = foreign.published()
    payload["label_names"] = "x"
    foreign.set_published(payload)
    assert run_main(gates, capsys, foreign).rc == 1
    missed = World(tmp_path / "missed", template)
    missed.negate_img([0])
    assert run_main(gates, capsys, missed).rc == 1


def test_the_output_directory_is_made_when_it_is_not_there_and_gates_json_is_replaced_whole(gates, world, capsys):
    assert not world.out.exists()
    assert run_main(gates, capsys, world).rc == 0
    (world.out / "gates.json").write_text("stale")
    assert run_main(gates, capsys, world).rc == 0
    assert json.loads((world.out / "gates.json").read_text())["self_retrieval"] == 50
    assert not list(world.out.glob("*.tmp")), "written whole or not at all: no temporary file is left"


def test_the_cli_runs_as_the_wrapper_runs_it_in_a_fresh_interpreter_against_a_labeller_service(world, service, tmp_path):
    fake = service()
    argv = world.argv()
    argv[argv.index("--labeler-url") + 1] = fake.url
    done = subprocess.run([sys.executable, str(SCRIPT)] + argv, cwd=str(tmp_path), capture_output=True, text=True, timeout=240,
                          env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "PYTHONDONTWRITEBYTECODE": "1"})
    assert done.returncode == 0, done.stdout + done.stderr
    assert all(GATES_SHAPES.match(l) for l in done.stdout.splitlines()), done.stdout
    assert RESULT_CLEAN in done.stdout.splitlines()
    assert world.gates_json()["passed"] is True


# ── the wrapper, rehearsed in a temp tree ────────────────────────────────────────────────────────────────────────────────────────────

FAKE_F1CHEXBERT = '''\
"""A fake f1chexbert (tests/test_chat_retrieval_gates.py): what app/labeler.py uses of f1chexbert 0.0.2, with no model. The labels are
RuleLabeler's. Knobs in the environment: FAKE_F1_LOAD_SLEEP (seconds the load takes), FAKE_F1_LOAD_FAIL (the load raises)."""
import os
import time

from app.labels import CHEXBERT_14, RuleLabeler


def _mark(line):
    path = os.environ.get("FAKE_F1_MARK")
    if path:
        with open(path, "a") as handle:
            handle.write(line + "\\n")


class F1CheXbert(object):
    def __init__(self, refs_filename=None, hyps_filename=None, device=None, **kwargs):
        _mark("loading hf_home=%s offline=%s" % (os.environ.get("HF_HOME", "unset"), os.environ.get("HF_HUB_OFFLINE", "unset")))
        if os.environ.get("FAKE_F1_LOAD_SLEEP"):
            time.sleep(float(os.environ["FAKE_F1_LOAD_SLEEP"]))
        if os.environ.get("FAKE_F1_LOAD_FAIL"):
            raise RuntimeError("fake CheXbert will not load")
        self.target_names = list(CHEXBERT_14)
        _mark("loaded")

    def get_label(self, report, mode="rrg"):
        return RuleLabeler().label([report])[0]
'''

# The venv's python, as the driver's step sees it: a note of the environment the DRIVER runs with, the tiny engine in place of the
# checkpoint (the one argument the job never passes), and, when asked, what a chattering library prints (text, an id and a path on lines
# that look like ours) or a progress bar. Everything else (the free port, the health probe) is the real python.
GATES_PYTHON_STUB = """#!/bin/bash
""" + wr.RECORD_CALL + """case "$1" in
  scripts/chat_retrieval_gates.py)
    { echo "@@"; echo "PYTHONPATH=${PYTHONPATH-unset}"; echo "HF_HOME=${HF_HOME-unset}"; echo "HF_HUB_OFFLINE=${HF_HUB_OFFLINE-unset}"
      echo "OMP_NUM_THREADS=${OMP_NUM_THREADS-unset}"; echo "NO_PROXY=${NO_PROXY-unset}"; echo "no_proxy=${no_proxy-unset}"
      echo "PWD=${PWD}"; } >> "$STUB_DIR/driver.env"
    if [ -n "${FAKE_DRIVER_JUNK:-}" ]; then
      echo "[gates] study_id=12345678"
      echo "[gates] The heart is mildly enlarged and there is a small pleural effusion."
      echo 'RESULT {"study_id":50000001}'
      echo "ERROR the heart is mildly enlarged"
      echo "=== FAKE MIMIC TEXT /sc/home/someone/images/p10/img.jpg ==="
      echo "Findings: FAKE REPORT TEXT study_id=12345678" >&2
    fi
    if [ -n "${FAKE_DRIVER_PROGRESS:-}" ]; then printf 'Loading weights:  50%%|#####     | 1/2\\r' >&2; fi
    [ -z "${FAKE_DRIVER_EXIT:-}" ] || exit "${FAKE_DRIVER_EXIT}"
    exec "$REAL_PYTHON" "$@" --engine tiny;;
esac
exec "$REAL_PYTHON" "$@"
"""

# .venv_chexbert's python: records how the labeller was started (its pid is the one the wrapper holds), then is the real python, or
# exits at once, or ignores SIGTERM (a uvicorn that is waiting for a model to load will not stop on it either).
LABELLER_PYTHON_STUB = """#!/bin/bash
{ echo "@@"; echo "pid=$$"; for a in "$@"; do printf '%s\\n' "$a"; done; echo "PYTHONPATH=${PYTHONPATH-unset}"; echo "HF_HOME=${HF_HOME-unset}"
  echo "HF_HUB_OFFLINE=${HF_HUB_OFFLINE-unset}"; echo "PWD=${PWD}"; } >> "$STUB_DIR/labeller.calls"
[ -z "${FAKE_LABELLER_EXIT:-}" ] || exit "${FAKE_LABELLER_EXIT}"
if [ -n "${FAKE_LABELLER_IGNORES_TERM:-}" ]; then
  trap '' TERM
  while true; do sleep 1; done
fi
exec "$REAL_PYTHON" "$@"
"""


class GatesBox(wr.JobBox):
    """The wrapper run for real in the temp tree of tests/wrapper_rehearsal.py: the REAL driver beside it (its imports are the working
    tree's own, through symlinks: app, hybrid_xmamba and the other scripts), the tiny world as CHAT_HOME's gallery, DATA and the published
    dump under results/ (repo/results is a symlink into the thesis checkout, as on the cluster), both overlays with their sentinels,
    the fake f1chexbert in the labeller's overlay, and two stub pythons."""

    WRAPPER = SH
    SCRIPTS = ("chat_retrieval_gates.py",)
    PYTHON_STUB = GATES_PYTHON_STUB
    RUN_DIRS = ("h100_report_gen_m3_tower13d_s42",)
    SLURM_CPUS = "8"
    job = "1234567"

    def __init__(self, root: Path, template: Path, stamp: Optional[str] = STAMP):
        self.template = template
        super().__init__(root, stamp)

    def populate(self) -> None:
        shutil.copytree(str(self.template / "gallery"), str(self.gallery))
        shutil.rmtree(str(self.data))
        shutil.copytree(str(self.template / "data"), str(self.data))
        (self.main / "results").mkdir()
        (self.repo / "results").symlink_to(self.main / "results", target_is_directory=True)
        shutil.copytree(str(self.template / "dump"), str(self.dump))
        for overlay in (".chat_deps", ".chat_deps_chexbert"):
            (self.repo / overlay).mkdir()
            (self.repo / overlay / ".setup_ok").write_text("2026-10-10T08:00:00Z\n")
        (self.repo / ".chat_deps_chexbert" / "f1chexbert").mkdir()
        (self.repo / ".chat_deps_chexbert" / "f1chexbert" / "__init__.py").write_text(FAKE_F1CHEXBERT)
        (self.repo / ".venv_chexbert" / "bin").mkdir(parents=True)
        labeller = self.repo / ".venv_chexbert" / "bin" / "python"
        labeller.write_text(LABELLER_PYTHON_STUB)
        labeller.chmod(0o755)
        for name in ("app", "hybrid_xmamba"):
            (self.repo / name).symlink_to(REPO_ROOT / name, target_is_directory=True)
        for path in (REPO_ROOT / "scripts").iterdir():                 # the scripts the driver imports (app.engine imports scripts.*)
            if path.is_file() and not (self.repo / "scripts" / path.name).exists():
                (self.repo / "scripts" / path.name).symlink_to(path)

    @property
    def gallery(self) -> Path:
        return self.chat / "gallery" / BUILD

    @property
    def dump(self) -> Path:
        return self.main / "results" / DUMP_NAME

    @property
    def out(self) -> Path:
        return self.main / "results" / "chat_retrieval_gates_{}".format(self.job)

    def base_env(self) -> Dict[str, str]:
        env = super().base_env()
        env.update(DATA=str(self.data), LABELER_WAIT_S="60", FAKE_F1_MARK=str(self.stubs / "fake.mark"))
        return env

    def driver_calls(self) -> List[List[str]]:
        return [c for c in self.calls() if c and c[0] == "scripts/chat_retrieval_gates.py"]

    def driver_env(self) -> List[str]:
        log = self.stubs / "driver.env"
        return [line for rec in log.read_text().split("@@\n") if rec.strip() for line in rec.splitlines()] if log.exists() else []

    def labeller_starts(self) -> List[List[str]]:
        log = self.stubs / "labeller.calls"
        return [rec.splitlines() for rec in log.read_text().split("@@\n") if rec.strip()] if log.exists() else []

    def labeller_pids(self) -> List[int]:
        return [int(start[0][len("pid="):]) for start in self.labeller_starts()]

    def marks(self) -> List[str]:
        mark = self.stubs / "fake.mark"
        return mark.read_text().splitlines() if mark.exists() else []

    def gates_json(self) -> dict:
        return json.loads((self.out / "gates.json").read_text())


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def gone(pid: int, within: float = 5.0) -> bool:
    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        if not alive(pid):
            return True
        time.sleep(0.05)
    return not alive(pid)


@pytest.fixture
def box(tmp_path, template):
    pytest.importorskip("fastapi")
    pytest.importorskip("uvicorn")
    return GatesBox(tmp_path, template)


def test_a_clean_run_holds_the_three_gates_and_prints_only_safe_lines(box):
    before = {name: snapshot(getattr(box, name)) for name in ("repo", "main", "data", "scratch")}
    gallery_before, chat_before = snapshot(box.gallery), snapshot(box.chat)
    done = box.run()
    lines = job_lines(done)
    assert done.returncode == 0, done.stdout
    assert lines[0] == "=== sync 3f2a9c41d7e86b05a1c4e9d3b7f60285ac9e1d47 clean ===", "the provenance comes first"
    assert [l for l in lines if not LINE_OK.match(l)] == [], "a line that is not ===, [gates], RESULT or ERROR"
    assert not [l for l in lines if "12345678" in l or "/sc/" in l or "Traceback" in l or "study_id" in l or str(box.root) in l], lines
    tagged = [l for l in lines if l.startswith("[gates] ")]
    assert tagged[0].startswith("[gates] engine device=cpu threads=") and tagged[1].startswith("[gates] gallery images=200 report_rows=260 ")
    assert tagged[2:] == ["[gates] self_retrieval hits=50 of=50 misses=0", "[gates] own_rank equal=50 dedup_equal=50 of=50 max_diff=0",
                          "[gates] labeller pred_equal=50 true_equal=50 of=50"]
    assert results(lines) == [{"self_retrieval": 50, "own_rank_equal": 50, "own_rank_dedup_equal": 50, "labeller_equal": 100, "label_names_ok": True}]
    assert "=== gates lines withheld: 0 ===" in lines
    assert lines[-1].startswith("=== END chat retrieval gates: all three gates held, wall_s=")
    assert all(len(l) <= 300 for l in lines if l.startswith(("RESULT ", "ERROR")))
    # what it wrote: one new directory under results/, with the two raw logs and gates.json, and nothing else anywhere (R8)
    assert sorted(p.name for p in box.out.iterdir()) == ["gates.json", "gates.log", "labeler.log"]
    assert box.gates_json()["passed"] is True and box.gates_json()["labeller_equal"] == 100
    new_files = {"results/chat_retrieval_gates_1234567/" + n for n in ("gates.json", "gates.log", "labeler.log")} | {"results/chat_retrieval_gates_1234567"}
    for name, snap in before.items():
        after = snapshot(getattr(box, name))
        assert {p for p, _, _ in after} - {p for p, _, _ in snap} <= new_files, name
        assert {p for p, _, _ in snap} - {p for p, _, _ in after} == set(), name + ": something was removed"
        assert {p for p, s, m in snap if (p, s, m) not in set(after)} <= {"results"}, name + ": a file was changed (only results/ gains an entry)"
    assert snapshot(box.gallery) == gallery_before and snapshot(box.chat) == chat_before, "the gallery is only read"
    assert not any(alive(pid) for pid in box.labeller_pids()), "the labeller is stopped when the job ends"


def test_the_raw_output_of_the_driver_and_of_the_labeller_goes_to_files_and_never_to_the_log(box):
    done = box.run(FAKE_DRIVER_JUNK="1", FAKE_DRIVER_PROGRESS="1")
    lines = job_lines(done)
    assert done.returncode == 0, done.stdout
    assert [l for l in lines if not LINE_OK.match(l)] == []
    assert not [l for l in lines if "FAKE" in l or "12345678" in l or "/sc/" in l or "50000001" in l or "heart" in l or "Loading" in l], lines
    assert "=== gates lines withheld: 5 ===" in lines, "five lines that wear one of our prefixes and have none of the shapes, counted"
    raw = (box.out / "gates.log").read_bytes().decode()
    assert "FAKE REPORT TEXT" in raw and "[gates] study_id=12345678" in raw, "stdout and stderr are both in the file"
    assert "1/2\r[gates] engine" in raw, "a progress bar and the line after it are one physical line of the raw log"
    assert any(l.startswith("[gates] engine device=cpu threads=") for l in lines), "and the filter, which splits on the carriage return, finds it"
    assert "POST /label" in (box.out / "labeler.log").read_text()


def test_a_filter_that_cannot_count_says_unknown_and_not_zero(box):
    grep = box.bin / "grep"           # fails only for the counting call: every other grep is the real one
    grep.write_text('#!/bin/bash\ncase " $* " in *" -avcE "*) exit 2;; esac\n'
                    'for g in /usr/bin/grep /bin/grep; do [ -x "$g" ] && exec "$g" "$@"; done\nexit 127\n')
    grep.chmod(0o755)
    done = box.run()
    lines = job_lines(done)
    assert done.returncode == 0, done.stdout
    assert "=== gates lines withheld: unknown ===" in lines and "=== gates lines withheld: 0 ===" not in lines


def test_the_labeller_is_started_in_the_chexbert_environment_on_a_free_loopback_port_and_the_driver_is_told_where(box):
    done = box.run(HF_HOME="/stray/hf", PYTHONPATH="/stray/overlay", CHEXBERT_HF_HUB_OFFLINE=None)
    assert done.returncode == 0, done.stdout
    (start,) = box.labeller_starts()
    assert start[1:7] == ["-m", "uvicorn", "app.labeler:app", "--host", "127.0.0.1", "--port"]
    port = start[7]
    assert port.isdigit() and 1024 <= int(port) <= 65535
    assert start[8:11] == ["PYTHONPATH=.chat_deps_chexbert", "HF_HOME=unset", "HF_HUB_OFFLINE=0"] and start[11].startswith("PWD=")
    assert os.path.realpath(start[11][len("PWD="):]) == os.path.realpath(str(box.repo)), "run from the repository root"
    (call,) = box.driver_calls()
    assert call[call.index("--labeler-url") + 1] == "http://127.0.0.1:" + port
    assert box.marks()[0] == "loading hf_home=unset offline=0", "D24: the weights are in the default Hugging Face cache, online as score_chexbert_h100.sh"


def test_the_chexbert_environment_can_go_offline_on_the_submit_line_and_the_engine_stays_offline(box):
    assert box.run(CHEXBERT_HF_HUB_OFFLINE="1").returncode == 0
    assert box.marks()[0] == "loading hf_home=unset offline=1"
    assert [l for l in box.driver_env() if l.startswith("HF_HUB_OFFLINE=")] == ["HF_HUB_OFFLINE=1"], "the engine's is not that lever's"


def test_the_driver_runs_in_the_main_venv_with_its_own_overlay_offline_with_eight_threads_and_the_published_defaults(box):
    done = box.run(PYTHONPATH="/stray/overlay", HF_HUB_OFFLINE="0", HF_HOME="/stray/hf")
    assert done.returncode == 0, done.stdout
    (call,) = box.driver_calls()
    dump = "results/" + DUMP_NAME
    assert call == ["scripts/chat_retrieval_gates.py", "--checkpoint", "./outputs/h100_report_gen_m3_tower13d_s42/checkpoints/last.ckpt",
                    "--model-config", "hybrid_150m_m3_rrg", "--gallery", str(box.gallery), "--data", str(box.data),
                    "--published-labels", dump + "/chexbert_labels.json", "--published-hyps", dump + "/hyps.txt",
                    "--published-refs", dump + "/refs.txt", "--labeler-url", call[call.index("--labeler-url") + 1],
                    "--out", "results/chat_retrieval_gates_1234567", "--threads", "8"]
    env = box.driver_env()
    assert env[0:3] == ["PYTHONPATH=.chat_deps", "HF_HOME={}/.hf".format(box.scratch), "HF_HUB_OFFLINE=1"], env
    assert "OMP_NUM_THREADS=8" in env and "NO_PROXY=127.0.0.1,localhost" in env and "no_proxy=127.0.0.1,localhost" in env
    assert os.path.realpath(env[-1][len("PWD="):]) == os.path.realpath(str(box.repo))


def test_the_build_and_the_model_and_the_dump_can_be_chosen_on_the_submit_line(box):
    other = box.chat / "gallery" / "v2_13d_1"
    shutil.copytree(str(box.gallery), str(other))
    shutil.copytree(str(box.dump), str(box.main / "results" / "elsewhere"))
    done = box.run(BUILD_ID="v2_13d_1", REFERENCE_DIR="results/elsewhere", MODEL_CONFIG="hybrid_150m_v2_rrg")
    assert done.returncode == 0, done.stdout
    (call,) = box.driver_calls()
    assert call[call.index("--gallery") + 1] == str(other) and call[call.index("--model-config") + 1] == "hybrid_150m_v2_rrg"
    assert call[call.index("--published-hyps") + 1] == "results/elsewhere/hyps.txt"
    assert [l for l in job_lines(done) if l.startswith("=== P5-F")] == [
        "=== P5-F live-path gates for gallery v2_13d_1: report model hybrid_150m_v2_rrg, CPU, 8 threads ==="]


def test_a_gate_that_fails_ends_the_job_with_exit_1_the_numbers_and_the_labeller_stopped(box):
    payload = json.loads((box.dump / "chexbert_labels.json").read_text())
    payload["y_pred"][4][2] = 1 - payload["y_pred"][4][2]
    (box.dump / "chexbert_labels.json").write_text(json.dumps(payload))
    done = box.run()
    lines = job_lines(done)
    assert done.returncode == 1, done.stdout
    assert results(lines) == [{"self_retrieval": 50, "own_rank_equal": 50, "own_rank_dedup_equal": 50, "labeller_equal": 99, "label_names_ok": True}]
    assert "ERROR gate labeller_equal=99 expected=100" in lines and "ERROR gates exit=1" in lines
    assert not [l for l in lines if l.startswith("=== END")], "no END line for a job whose gate failed"
    assert [l for l in lines if not LINE_OK.match(l)] == []
    assert box.gates_json()["passed"] is False and box.gates_json()["label_mismatch_rows"] == {"hyps": [4], "refs": []}
    assert not any(alive(pid) for pid in box.labeller_pids())


def test_a_driver_that_crashes_leaves_its_traceback_in_the_raw_log_only_and_the_labeller_stopped(box):
    (box.data / "train.parquet").write_text("this is not a parquet file: FAKE REPORT TEXT study_id=12345678")    # past the guards, which look for the file
    done = box.run()
    lines = job_lines(done)
    assert done.returncode == 1, done.stdout
    failed = [l for l in lines if l.startswith("ERROR failed ")]
    assert len(failed) == 1 and re.fullmatch(r"ERROR failed [A-Za-z_][A-Za-z0-9_]*", failed[0]) and "ERROR gates exit=1" in lines, lines
    assert "Traceback" not in done.stdout and "parquet" not in done.stdout and "FAKE" not in done.stdout
    assert "Traceback (most recent call last)" in (box.out / "gates.log").read_text()
    assert box.gates_json()["error"] == failed[0][len("ERROR "):] and box.gates_json()["passed"] is False
    assert not any(alive(pid) for pid in box.labeller_pids())


def test_a_driver_that_exits_0_without_its_result_line_is_a_failure_not_a_pass(box):
    done = box.run(FAKE_DRIVER_EXIT="0")
    lines = job_lines(done)
    assert done.returncode == 1, done.stdout
    assert "ERROR gates printed no RESULT line" in lines and not [l for l in lines if l.startswith("=== END")]


def test_a_labeller_that_never_loads_ends_the_job_after_the_bounded_wait_without_running_the_driver(box):
    started = time.monotonic()
    done = box.run(FAKE_F1_LOAD_FAIL="1", LABELER_WAIT_S="3")
    lines = job_lines(done)
    assert done.returncode == 1, done.stdout
    assert "ERROR labeller not ready after 3 s" in lines
    assert time.monotonic() - started < 60 and box.driver_calls() == [], "bounded, and the driver is never started"
    assert not [l for l in lines if l.startswith("[gates]") or l.startswith("RESULT")]
    assert not any(alive(pid) for pid in box.labeller_pids()), "stopped on this exit path too"
    assert not (box.out / "gates.json").exists()


def test_a_labeller_that_dies_at_once_is_reported_with_its_exit_status_and_the_driver_never_runs(box):
    started = time.monotonic()
    done = box.run(FAKE_LABELLER_EXIT="3", LABELER_WAIT_S="60")
    lines = job_lines(done)
    assert done.returncode == 1, done.stdout
    assert "ERROR labeller exit=3" in lines and box.driver_calls() == []
    assert not [l for l in lines if l.startswith("[gates]")]
    assert time.monotonic() - started < 20, "a labeller that is gone is not waited for until the bound (60 s here)"


def test_a_labeller_that_is_slow_to_load_is_waited_for(box):
    done = box.run(FAKE_F1_LOAD_SLEEP="3")
    assert done.returncode == 0, done.stdout
    assert box.marks()[-1] == "loaded" and RESULT_CLEAN in job_lines(done)
    ready = [l for l in job_lines(done) if l.startswith("=== labeller: ready after ")]
    assert len(ready) == 1 and int(re.fullmatch(r"=== labeller: ready after ([0-9]+) s ===", ready[0]).group(1)) >= 2


def test_a_labeller_that_ignores_sigterm_is_stopped_all_the_same_after_a_grace_period(box):
    """A uvicorn that is waiting for the model to load does not leave on SIGTERM either: the stop escalates, so that the failure paths
    end the job in seconds and not in the time the labeller might take."""
    started = time.monotonic()
    done = box.run(FAKE_LABELLER_IGNORES_TERM="1", LABELER_WAIT_S="2")
    assert done.returncode == 1, done.stdout
    assert "ERROR labeller not ready after 2 s" in job_lines(done)
    (pid,) = box.labeller_pids()
    assert not alive(pid) and time.monotonic() - started < 60


def test_a_sigterm_to_the_job_while_it_waits_stops_the_labeller_too(box):
    proc = subprocess.Popen([wr.BASH, str(box.repo / "scripts" / SH)], cwd=str(box.root), env=dict(box.base_env(), FAKE_F1_LOAD_SLEEP="30"),
                            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        deadline = time.monotonic() + 30
        while not box.labeller_pids() and time.monotonic() < deadline:
            time.sleep(0.05)
        (pid,) = box.labeller_pids()
        time.sleep(1.5)                                                # inside the wait, with a probe in flight
        proc.send_signal(signal.SIGTERM)
        out, _ = proc.communicate(timeout=60)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()
    assert proc.returncode == 143, out
    assert gone(pid, 15), "the labeller outlived the job that started it"


# ── the guards: every refusal is one ERROR line, before the labeller exists ───────────────────────────────────────────────────────────

def refuse(box: GatesBox, **env: Optional[str]) -> List[str]:
    done = box.run(**env)
    lines = job_lines(done)
    assert done.returncode == 1, done.stdout
    assert lines[-1].startswith("ERROR "), lines
    assert box.labeller_starts() == [] and box.driver_calls() == [], "no step starts before every guard has passed"
    assert not box.out.exists() or not any(box.out.iterdir()), "and nothing is written"
    assert [l for l in lines if not LINE_OK.match(l)] == []
    return lines


PLAIN = "must be one plain name: letters, digits, dot, dash, underscore"
WHOLE = "ERROR LABELER_WAIT_S must be a whole number of seconds, at most 5 digits"


@pytest.mark.parametrize("env, message", [
    ({"BUILD_ID": "../x"}, "ERROR BUILD_ID " + PLAIN), ({"BUILD_ID": "a b"}, "ERROR BUILD_ID " + PLAIN),
    ({"MODEL_CONFIG": "x/y"}, "ERROR MODEL_CONFIG " + PLAIN), ({"MODEL_CONFIG": "x y"}, "ERROR MODEL_CONFIG " + PLAIN),
    ({"LABELER_WAIT_S": "soon"}, WHOLE), ({"LABELER_WAIT_S": "123456"}, WHOLE), ({"LABELER_WAIT_S": "-1"}, WHOLE),
    ({"BUILD_ID": "g13d_m3_v2"}, "ERROR the gallery has no manifest.json: build it first (build_retrieval_gallery_h100.sh)"),
])
def test_a_bad_lever_or_a_missing_gallery_is_one_error_line_and_nothing_starts(box, env, message):
    assert refuse(box, **env)[-1] == message


def test_a_bad_lever_is_never_echoed(box):
    lines = refuse(box, BUILD_ID="a b FAKE_TEXT", MODEL_CONFIG="x")
    assert not [l for l in lines if "FAKE_TEXT" in l]


def test_a_missing_chat_home_is_refused(box):
    shutil.rmtree(str(box.chat))
    assert refuse(box)[-1] == "ERROR CHAT_HOME does not exist: run chat_cluster_setup_h100.sh first"


def test_a_gallery_whose_gate_is_not_decided_equal_is_refused(box):
    manifest = json.loads((box.gallery / "manifest.json").read_text())
    manifest["gate_rk"]["equal"] = False
    (box.gallery / "manifest.json").write_text(json.dumps(manifest, indent=2))
    assert refuse(box)[-1] == "ERROR the gallery's R@k gate is not decided equal: it cannot be used"
    del manifest["gate_rk"]["equal"]
    (box.gallery / "manifest.json").write_text(json.dumps(manifest, indent=2))
    assert refuse(box)[-1] == "ERROR the gallery's R@k gate is not decided equal: it cannot be used"


def test_a_gallery_without_the_builds_own_test_embeddings_is_refused(box):
    (box.gallery / "test_img_emb.npy").unlink()
    assert refuse(box)[-1] == "ERROR the gallery has no test_img_emb.npy: it is the build's own embedding of the test images"


def test_a_missing_results_directory_is_refused_and_not_made(box):
    (box.repo / "results").unlink()
    assert refuse(box)[-1] == "ERROR results is missing: run chat_cluster_setup_h100.sh first"
    assert not (box.repo / "results").exists()


@pytest.mark.parametrize("path, message", [
    (".venv/bin/activate", "ERROR the main venv is missing: run chat_cluster_setup_h100.sh first"),
    (".venv_chexbert/bin/python", "ERROR the CheXbert venv is missing: run setup_chexbert_venv_h100.sh first"),
    (".chat_deps/.setup_ok", "ERROR the web overlays are not set up: run chat_cluster_setup_h100.sh first"),
    (".chat_deps_chexbert/.setup_ok", "ERROR the web overlays are not set up: run chat_cluster_setup_h100.sh first"),
    ("scripts/chat_retrieval_gates.py", "ERROR chat_retrieval_gates.py is missing from this tree: run chat_remote.sh sync first"),
])
def test_a_missing_venv_overlay_or_script_is_refused(box, path, message):
    (box.repo / path).unlink()
    assert refuse(box)[-1] == message


@pytest.mark.parametrize("where, name, message", [
    ("main", "outputs/h100_report_gen_m3_tower13d_s42/checkpoints/last.ckpt", "ERROR the report model's checkpoint was not found"),
    ("data", "train.parquet", "ERROR train.parquet was not found in DATA"),
    ("data", "test.parquet", "ERROR test.parquet was not found in DATA"),
    ("dump", "hyps.txt", "ERROR the published dump has no hyps.txt"),
    ("dump", "refs.txt", "ERROR the published dump has no refs.txt"),
    ("dump", "chexbert_labels.json", "ERROR the published dump has no chexbert_labels.json"),
])
def test_a_missing_input_is_refused_by_name(box, where, name, message):
    (getattr(box, where) / name).unlink()
    assert refuse(box)[-1] == message


# ── provenance ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("dirty", [False, True])
def test_the_first_line_of_the_job_log_names_the_commit_and_cleanliness_the_tree_was_synced_with(tmp_path, template, dirty):
    stamp = real_stamp(tmp_path / "sync", dirty)
    box = GatesBox(tmp_path / "job", template, stamp=stamp)
    (box.repo / "results").unlink()                                      # stop at a guard: only the first line is the subject
    done = box.run()
    _, sha, flag = stamp.split()
    assert job_lines(done)[0] == "=== sync {} {} ===".format(sha, flag), job_lines(done)[:2]


def test_a_missing_sync_stamp_is_reported_as_unknown_and_the_job_goes_on_to_its_guards(tmp_path, template):
    box = GatesBox(tmp_path, template, stamp=None)
    (box.repo / "results").unlink()
    lines = job_lines(box.run())
    assert lines[0] == "=== sync unknown ===" and lines[-1] == "ERROR results is missing: run chat_cluster_setup_h100.sh first"


@pytest.mark.parametrize("stamp", [
    "", "rm -rf / ; echo $(whoami) FAKE_STAMP_TEXT", "2026-10-10T08:00:00Z 3f2a9c41d7e86b05a1c4e9d3b7f60285ac9e1d4 clean",
    "2026-10-10T08:00:00Z 3F2A9C41D7E86B05A1C4E9D3B7F60285AC9E1D47 clean", "2026-10-10T08:00:00Z 3f2a9c41d7e86b05a1c4e9d3b7f60285ac9e1d47 maybe",
    "2026-10-10T08:00:00Z 3f2a9c41d7e86b05a1c4e9d3b7f60285ac9e1d47 clean FAKE_EXTRA"])
def test_a_malformed_sync_stamp_is_reported_as_unknown_and_never_echoed(tmp_path, template, stamp):
    box = GatesBox(tmp_path, template, stamp=stamp)
    (box.repo / "results").unlink()
    lines = job_lines(box.run())
    assert lines[0] == "=== sync unknown ===", lines[:2]
    assert not [l for l in lines if "FAKE_" in l or "whoami" in l or "rm -rf" in l]


def test_the_whole_clean_job_log_passes_the_summary_allowlist_and_the_mask_unchanged(box, tmp_path):
    """R7, end to end through chat_remote.sh's own grep pattern and mask: nothing of the log is dropped or blanked."""
    lines = job_lines(box.run())
    comparable = [l for l in lines if not l.startswith("=== job=")]        # that one names the node, whose name is the machine's
    assert len(comparable) == len(lines) - 1 and any(l.startswith("RESULT ") for l in comparable)
    assert _summary_of(tmp_path / "summary", comparable) == comparable
