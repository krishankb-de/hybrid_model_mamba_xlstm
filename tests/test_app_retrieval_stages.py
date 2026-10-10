"""CHAT_UI_PLAN.md P5-E: the retrieve, label and score stages of a turn, their endpoints, and what public mode keeps of them.

Tiny engine + the tiny gallery (scripts/build_retrieval_gallery.py --tiny, its gate decided as the cluster decides it) + RuleLabeler:
synthetic data only (R7), so an assertion may print whatever it compares.
"""
import json
import logging
import threading

import pytest
from fastapi.testclient import TestClient

from app import pipeline as pipeline_module
from app import server
from app.engine import build_engine
from app.gallery import Gallery
from app.labels import CHEXBERT_14, LabelerClient, LabelerUnavailable, RuleLabeler, label_agreement
from app.pipeline import PUBLISHED_MODEL, Pipeline, PublishedDumps, Stop, TurnJob, Worker
from app.schemas import Options
from app.scoring import score_pair
from app.server import create_app
from app.store import Store
from scripts.build_retrieval_gallery import build_tiny
from tests.app_helpers import decide_gate, iter_sse, png_bytes, wait_until

PUBLIC = {"Authorization": "Bearer t", "X-Client-Id": "a"}
STAGES = ["preprocess", "encode", "retrieve", "generate", "label", "score"]
UPLOAD = object()   # _post's default image: png_bytes()
MANIFEST_ONLY = ("checkpoint_13d", "checkpoint_13d_sha256", "decoder_checkpoint", "job_id", "tower_sha256", "decoder_tower_sha256",
                 "counts", "gate_rk", "transform", "tokenizer", "labels_status", "synthetic")   # manifest.json keys: never in an event


@pytest.fixture
def gallery(tiny_gallery):
    return Gallery.open(decide_gate(tiny_gallery), None)


@pytest.fixture
def client(tmp_path, gallery):
    with TestClient(create_app(engine="tiny", home=str(tmp_path / "home"), gallery=gallery, labeler=RuleLabeler())) as c:
        yield c


def _post(c, sid, options=None, image=UPLOAD, text="", headers=None):
    files = None
    if image is not None:
        files = {"image": ("x.png", png_bytes() if image is UPLOAD else image, "image/png")}
    data = {"text": text, "options": json.dumps({"max_new_tokens": 16} if options is None else options)}
    return c.post("/v1/sessions/{}/messages".format(sid), files=files, data=data, headers=headers)


def _session(c, headers=None):
    return c.post("/v1/sessions", json={}, headers=headers).json()["id"]


def _turn(c, options=None, image=UPLOAD, text="", headers=None, sid=None):
    r = _post(c, sid or _session(c, headers), options, image, text, headers)
    assert r.status_code == 200, r.text
    frames = list(iter_sse([r.text]))
    assert [f["data"]["seq"] for f in frames] == list(range(1, len(frames) + 1))
    return frames


def _ends(frames):
    return {f["data"]["stage"]: f["data"] for f in frames if f["event"] == "stage_end"}


def _starts(frames):
    return [f["data"]["stage"] for f in frames if f["event"] == "stage_start"]


def _pooled(c, data):
    """The tiny engine's query vector for these bytes, as the turn computes it."""
    engine = c.app.state.engines["tiny"]
    _, prepared = engine.preprocess(data)
    _, encoded = engine.encode(prepared)
    return encoded.pooled.numpy()


def _named(row):
    return dict(zip(CHEXBERT_14, row))


def _keys_anywhere(obj):
    if isinstance(obj, dict):
        return set(obj).union(*[_keys_anywhere(v) for v in obj.values()]) if obj else set()
    if isinstance(obj, list):
        return set().union(*[_keys_anywhere(v) for v in obj]) if obj else set()
    return set()


def _set_manifest(root, **fields):
    manifest = json.loads((root / "manifest.json").read_text())
    manifest.update(fields)
    (root / "manifest.json").write_text(json.dumps(manifest))


def _fake_real_engines(monkeypatch, other_tower=()):
    """build_engine("real", ...) without checkpoints: a tiny engine named after its model. A model in other_tower gets a tower whose
    weights differ, as a report model whose image tower was trained on would."""
    def fake(kind, **kw):
        engine = build_engine("tiny")
        engine.name = kw["model_config"]
        engine._card.update(name=engine.name, checkpoint=kw["checkpoint"])
        if engine.name in other_tower:
            engine.tower.head.bias.data += 1.0
        return engine

    monkeypatch.setattr(server, "build_engine", fake)


# ---- retrieve ------------------------------------------------------------------------------------------------------------

def test_a_turns_retrieve_detail_has_k_images_neighbours_and_k_reports_groups_with_labels(client, gallery):
    frames = _turn(client, {"max_new_tokens": 16, "k_images": 5, "k_reports": 2})
    assert [f["data"]["stage"] for f in frames if f["event"] == "stage_end"] == STAGES
    assert _starts(frames) == STAGES[:5]   # every stage that ran was started; score had no reference
    detail = _ends(frames)["retrieve"]["detail"]
    query = _pooled(client, png_bytes())
    assert detail["image_neighbors"] == gallery.image_neighbors(query, 5)       # Encoded.pooled is the query (D4)
    assert detail["report_matches"] == gallery.report_matches(query, 2)
    assert [n["rank"] for n in detail["image_neighbors"]] == [1, 2, 3, 4, 5]
    assert len(detail["report_matches"]) == 2 and len({m["group"] for m in detail["report_matches"]}) == 2
    assert all(set(n["labels"]) == set(CHEXBERT_14) for n in detail["image_neighbors"] + detail["report_matches"])
    assert "true_report_rank" not in detail   # an upload that is no test image has no own rank
    assert frames[-1]["data"]["status"] == "done"


def test_the_retrieve_event_carries_the_galleries_facts_and_nothing_of_its_manifest(client, gallery):
    frames = _turn(client)
    assert _ends(frames)["retrieve"]["detail"]["gallery"] == gallery.facts()
    assert set(_ends(frames)["retrieve"]["detail"]["gallery"]) == {"build_id", "images", "report_rows", "report_groups",
                                                                 "towers_identical"}
    stored = client.get("/v1/messages/{}".format(frames[0]["data"]["message_id"])).json()["events"]
    assert not _keys_anywhere([e["data"] for e in stored]) & set(MANIFEST_ONLY)


def test_k_zero_skips_retrieval_and_one_k_of_zero_only_empties_its_list(client):
    frames = _turn(client, {"max_new_tokens": 16, "k_images": 0, "k_reports": 0})
    assert _ends(frames)["retrieve"] == {"stage": "retrieve", "skipped": "k_zero", "seq": _ends(frames)["retrieve"]["seq"]}
    assert "retrieve" not in _starts(frames) and frames[-1]["data"]["status"] == "done"
    assert _ends(frames)["label"]["detail"]["neighbor_agreement"] == []   # labels still run: nothing to agree with
    detail = _ends(_turn(client, {"max_new_tokens": 16, "k_images": 0, "k_reports": 2}))["retrieve"]["detail"]
    assert detail["image_neighbors"] == [] and len(detail["report_matches"]) == 2


def test_without_a_gallery_retrieve_is_skipped_and_the_turn_still_labels(tmp_path):
    with TestClient(create_app(engine="tiny", home=str(tmp_path), labeler=RuleLabeler())) as c:
        frames = _turn(c)
        assert _ends(frames)["retrieve"]["skipped"] == "gallery_unavailable"
        assert _ends(frames)["label"]["detail"]["neighbor_agreement"] == []
        assert frames[-1]["data"]["status"] == "done"
        assert c.get("/v1/models").json()["features"] == {"retrieval": False, "labels": True}


def test_an_upload_identical_to_a_gallery_image_says_so_in_the_preprocess_detail(client, gallery):
    frames = _turn(client, image=gallery.image_path(7).read_bytes())
    assert _ends(frames)["preprocess"]["detail"]["identical_to"] == {"split": "train", "row": 7}
    assert "true_report_rank" not in _ends(frames)["retrieve"]["detail"]   # a train image: no own rank
    assert _ends(frames)["score"]["skipped"] == "no_reference"
    plain = _turn(client)
    assert "identical_to" not in _ends(plain)["preprocess"]["detail"]


def test_an_upload_identical_to_a_test_image_gets_its_own_rank_and_a_score(client, gallery):
    data = gallery.test_study(2)["image"].read_bytes()
    frames = _turn(client, image=data)
    ends = _ends(frames)
    assert ends["preprocess"]["detail"]["identical_to"] == {"split": "test", "row": 2}
    assert ends["retrieve"]["detail"]["true_report_rank"] == gallery.own_report_rank(_pooled(client, data), 2)
    assert ends["score"]["detail"]["reference_source"] == "test_split"


# ---- a test-split study (the picker) ---------------------------------------------------------------------------------------

def test_a_test_row_turn_has_its_own_rank_and_a_score_against_its_reference(client, gallery):
    row = 3
    frames = _turn(client, {"max_new_tokens": 16, "test_row": row}, image=None)
    start, ends = frames[0]["data"], _ends(frames)
    assert start["image"]["source"] == "test_split" and start["options"]["test_row"] == row and start["image"]["filename"] is None
    data = gallery.test_study(row)["image"].read_bytes()
    assert ends["preprocess"]["detail"]["test_row"] == row and ends["preprocess"]["detail"]["source"] == "test_split"
    assert ends["retrieve"]["detail"]["true_report_rank"] == gallery.own_report_rank(_pooled(client, data), row)
    assert _starts(frames) == STAGES
    report, reference = frames[-1]["data"]["report"], gallery.test_study(row)["reference"]
    score = ends["score"]["detail"]
    y_hyp, y_ref = RuleLabeler().label([report, reference])
    assert score == dict(score_pair(report, reference, y_hyp, y_ref), reference_source="test_split",
                         reference_chexbert_14=_named(y_ref))
    assert ends["label"]["detail"]["chexbert_14"] == _named(y_hyp)
    assert "published" not in score   # no dumps on the laptop
    assert frames[-1]["data"]["status"] == "done"
    user = client.get("/v1/sessions/{}".format(start["session_id"])).json()["messages"][0]
    assert user["test_row"] == row and user["image_sha256"] == start["image"]["sha256"]


def test_a_text_only_follow_up_after_a_test_study_reruns_that_study(client, gallery):
    sid = _session(client)
    first = _turn(client, {"max_new_tokens": 16, "test_row": 5}, image=None, sid=sid)
    again = _turn(client, {"max_new_tokens": 16}, image=None, text="beam 2", sid=sid)
    start = again[0]["data"]
    assert start["image"]["source"] == "test_split" and start["image"]["sha256"] == first[0]["data"]["image"]["sha256"]
    assert start["options"]["test_row"] == 5 and start["options"]["beam_size"] == 2
    assert _ends(again)["retrieve"]["detail"]["true_report_rank"] == _ends(first)["retrieve"]["detail"]["true_report_rank"]
    assert _ends(again)["score"]["detail"]["reference_source"] == "test_split"
    users = [m for m in client.get("/v1/sessions/{}".format(sid)).json()["messages"] if m["role"] == "user"]
    assert [u["test_row"] for u in users] == [5, 5]
    question = _turn(client, {"max_new_tokens": 16}, image=None, text="is this normal?", sid=sid)
    assert [f["event"] for f in question] == ["message_start", "warning", "message_stop"]   # a question still runs nothing


def test_an_upload_after_a_test_study_is_what_a_follow_up_reruns(client):
    sid = _session(client)
    _turn(client, {"max_new_tokens": 16, "test_row": 1}, image=None, sid=sid)
    _turn(client, sid=sid)   # an upload: the newest image now
    again = _turn(client, {"max_new_tokens": 16}, image=None, text="beam 2", sid=sid)
    assert again[0]["data"]["image"]["source"] == "previous" and again[0]["data"]["options"]["test_row"] is None


def test_test_row_refusals_come_before_the_stream(client, gallery):
    sid = _session(client)
    n_test = gallery.facts()["report_rows"] - gallery.facts()["images"]
    r = _post(client, sid, {"test_row": n_test}, image=None)
    assert r.status_code == 422 and r.json()["error"]["message"] == "test_row must be below {}.".format(n_test)
    r = _post(client, sid, {"test_row": 0})   # and an upload too
    assert r.status_code == 422 and r.json()["error"]["message"] == "Send an image or a test row, not both."
    assert client.get("/v1/sessions/{}".format(sid)).json()["messages"] == []


def test_a_test_row_whose_image_cannot_be_read_is_refused_with_a_500(client, gallery):
    gallery.test_study(4)["image"].unlink()
    r = _post(client, _session(client), {"test_row": 4}, image=None)
    assert r.status_code == 500 and r.json()["error"] == {"type": "internal_error", "message": "Could not read the test-split image."}


# ---- label --------------------------------------------------------------------------------------------------------------

def test_label_has_one_agreement_per_neighbour_with_label_agreements_counts(client):
    frames = _turn(client, {"max_new_tokens": 16, "k_images": 6})
    neighbours = _ends(frames)["retrieve"]["detail"]["image_neighbors"]
    label = _ends(frames)["label"]["detail"]
    y = RuleLabeler().label([frames[-1]["data"]["report"]])[0]
    assert label["chexbert_14"] == _named(y) and label["positives"] == [n for n, v in zip(CHEXBERT_14, y) if v]
    assert [a["rank"] for a in label["neighbor_agreement"]] == [n["rank"] for n in neighbours] == list(range(1, 7))
    for agreement, neighbour in zip(label["neighbor_agreement"], neighbours):
        assert agreement == dict(label_agreement(y, [neighbour["labels"][k] for k in CHEXBERT_14]), rank=neighbour["rank"])
        assert agreement["of"] == 14


class _Down:
    """A labeller that answers its health check as told and fails every label call."""

    def __init__(self, healthy=True):
        self._healthy, self.calls = healthy, 0

    def healthy(self):
        return self._healthy

    def label(self, texts):
        self.calls += 1
        raise LabelerUnavailable("labeller request failed: ConnectionRefusedError")


@pytest.mark.parametrize("labeler, started", [(_Down(healthy=False), False), (_Down(healthy=True), True),
                                              (LabelerClient("http://127.0.0.1:9", timeout=2.0), False)],
                         ids=["health_check_fails", "label_call_fails", "nothing_listens"])
def test_a_labeller_that_is_down_skips_label_and_the_turn_still_ends_done(tmp_path, gallery, labeler, started):
    with TestClient(create_app(engine="tiny", home=str(tmp_path), gallery=gallery, labeler=labeler)) as c:
        frames = _turn(c, {"max_new_tokens": 16, "reference": "Findings: the heart is enlarged."})
        assert _ends(frames)["label"]["skipped"] == "labeler_unavailable"
        assert ("label" in _starts(frames)) is started     # a probe that fails skips before the stage starts
        assert frames[-1]["data"]["status"] == "done"
        score = _ends(frames)["score"]["detail"]           # the text scores still come; no labels, so no CheXbert part
        assert set(score) == {"rouge_l", "bleu_1", "bleu_4", "reference_source"} and score["reference_source"] == "user"


def _pending_gallery(tmp_path):
    """A tiny gallery whose labels are not built yet (labels_status pending), its gate decided."""
    build_tiny(tmp_path / "unlabelled", with_labels=False)
    pending = Gallery.open(decide_gate(tmp_path / "unlabelled"), None)
    assert pending.labels is None
    return pending


def test_labels_off_skips_label_and_says_so(tmp_path, gallery):
    with TestClient(create_app(engine="tiny", home=str(tmp_path / "a"), gallery=gallery, labeler=RuleLabeler())) as c:
        frames = _turn(c, {"max_new_tokens": 16, "label": False})
        assert _ends(frames)["label"]["skipped"] == "label_off" and frames[-1]["data"]["status"] == "done"
        assert "label" not in _starts(frames)


def test_a_gallery_whose_labels_are_pending_still_labels_the_report_and_marks_the_agreement_pending(tmp_path):
    # Ruling (a), fix round 1: pending gallery labels take only the neighbour agreement away, not the report's labels
    with TestClient(create_app(engine="tiny", home=str(tmp_path / "b"), gallery=_pending_gallery(tmp_path),
                               labeler=RuleLabeler())) as c:
        frames = _turn(c, {"max_new_tokens": 16, "test_row": 0}, image=None)
        ends = _ends(frames)
        assert all(n["labels"] is None for n in ends["retrieve"]["detail"]["image_neighbors"])
        label, report = ends["label"]["detail"], frames[-1]["data"]["report"]
        assert label == {"chexbert_14": _named(RuleLabeler().label([report])[0]), "positives": label["positives"],
                         "neighbor_agreement": [], "neighbor_agreement_pending": True}
        score = ends["score"]["detail"]
        assert "chexbert_14_micro_f1" in score and "exact_match_14" in score and "reference_chexbert_14" in score   # it keeps its CheXbert part
        assert "label" in _starts(frames) and frames[-1]["data"]["status"] == "done"
        assert c.get("/v1/models").json()["features"] == {"retrieval": True, "labels": True}   # true, and the stage really runs


def test_the_pending_agreement_marker_is_server_state_that_public_mode_shows(tmp_path):
    with TestClient(create_app(engine="tiny", home=str(tmp_path / "home"), mode="public", token="t",
                               gallery=_pending_gallery(tmp_path), labeler=RuleLabeler())) as c:
        label = _ends(_turn(c, headers=PUBLIC))["label"]["detail"]
        assert label["neighbor_agreement_pending"] is True and "neighbor_agreement" not in label and label["chexbert_14"]


def test_a_labelled_gallery_puts_no_pending_marker_on_the_label_stage(client):
    assert "neighbor_agreement_pending" not in _ends(_turn(client))["label"]["detail"]


# ---- score --------------------------------------------------------------------------------------------------------------

HYP = "Findings: The heart is mildly enlarged. There is a small left pleural effusion. Impression: Cardiomegaly."
REF = "Findings: Heart size is enlarged. Small left effusion is seen. Impression: Cardiomegaly and effusion."


def test_score_pair_uses_the_thesis_tables_own_functions():
    from scripts.bootstrap_compare import chexbert_f1, per_sample_rouge_l
    from scripts.evaluate_report_generation import corpus_bleu
    plain = score_pair(HYP, REF)
    assert plain["rouge_l"] == per_sample_rouge_l([HYP], [REF])[0]
    assert plain["bleu_1"] == corpus_bleu([HYP.split()], [REF.split()], 1)
    assert plain["bleu_4"] == corpus_bleu([HYP.split()], [REF.split()], 4)
    assert set(plain) == {"rouge_l", "bleu_1", "bleu_4"}
    y_hyp, y_ref = RuleLabeler().label([HYP, REF])
    scored = score_pair(HYP, REF, y_hyp, y_ref)
    assert scored["chexbert_14_micro_f1"] == chexbert_f1([y_ref], [y_hyp], "micro")
    assert scored["exact_match_14"] is (y_hyp == y_ref)
    assert score_pair(HYP, HYP, y_hyp, y_hyp)["exact_match_14"] is True
    assert score_pair(HYP, REF, y_hyp, None) == plain   # both label rows, or no CheXbert part


def test_a_users_reference_scores_the_turn_with_its_labels(client):
    frames = _turn(client, {"max_new_tokens": 16, "reference": "  Findings:  " + REF[10:]})
    score, report = _ends(frames)["score"]["detail"], frames[-1]["data"]["report"]
    reference = " ".join(("  Findings:  " + REF[10:]).split())
    y_hyp, y_ref = RuleLabeler().label([report, reference])
    assert score == dict(score_pair(report, reference, y_hyp, y_ref), reference_source="user", reference_chexbert_14=_named(y_ref))
    assert _starts(frames)[-1] == "score" and not [f for f in frames if f["event"] == "warning"]


def test_a_test_row_turn_with_its_own_reference_typed_in_scores_against_the_typed_one(client):
    frames = _turn(client, {"max_new_tokens": 16, "test_row": 1, "reference": REF}, image=None)
    assert _ends(frames)["score"]["detail"]["reference_source"] == "user"


def test_published_dumps_load_aligned_lines_and_say_none_beyond_them(tmp_path):
    (tmp_path / "m").mkdir()
    (tmp_path / "f").mkdir()
    (tmp_path / "m" / "hyps.txt").write_text("model zero\nmodel one\n")
    (tmp_path / "f" / "hyps.txt").write_text("floor zero\nfloor one\n")
    dumps = PublishedDumps.load(tmp_path / "m", tmp_path / "f")
    assert dumps == PublishedDumps(model_hyps=["model zero", "model one"], floor_hyps=["floor zero", "floor one"])
    assert (dumps.line("model", 1), dumps.line("floor", 0)) == ("model one", "floor zero")
    assert dumps.line("model", 2) is None and dumps.line("floor", -1) is None
    with pytest.raises(ValueError):
        dumps.line("other", 0)


def _dumps(tmp_path, n, model=lambda i: "SYNTHETIC model line {}".format(i)):
    for kind, line in (("model", model), ("floor", lambda i: "SYNTHETIC floor line {}".format(i))):
        (tmp_path / kind).mkdir(exist_ok=True)
        (tmp_path / kind / "hyps.txt").write_text("".join(line(i) + "\n" for i in range(n)))
    return {"model": str(tmp_path / "model"), "floor": str(tmp_path / "floor")}


def test_a_test_row_turn_shows_the_floor_line_and_no_model_line_for_an_engine_the_dumps_are_not_of(tmp_path, gallery):
    n_test = gallery.facts()["report_rows"] - gallery.facts()["images"]
    with TestClient(create_app(engine="tiny", home=str(tmp_path / "home"), gallery=gallery, labeler=RuleLabeler(),
                               published_dirs=_dumps(tmp_path, n_test))) as c:
        assert isinstance(c.app.state.published, PublishedDumps)
        score = _ends(_turn(c, {"max_new_tokens": 16, "test_row": 6}, image=None))["score"]["detail"]
        assert score["published"] == {"model_report": None, "floor_report": "SYNTHETIC floor line 6", "live_equals_published": None}
        assert "published" not in _ends(_turn(c, {"max_new_tokens": 16, "reference": REF}))["score"]["detail"]   # an upload


def test_dumps_that_do_not_match_the_test_split_are_not_used(tmp_path, gallery, capsys):
    with TestClient(create_app(engine="tiny", home=str(tmp_path / "home"), gallery=gallery, labeler=RuleLabeler(),
                               published_dirs=_dumps(tmp_path, 7))) as c:
        assert c.app.state.published is None
        assert "published" not in _ends(_turn(c, {"max_new_tokens": 16, "test_row": 6}, image=None))["score"]["detail"]
    assert "[server] published dumps unavailable" in capsys.readouterr().out


def test_live_equals_published_is_compared_only_under_the_published_protocol(tmp_path, tiny_gallery, monkeypatch):
    _fake_real_engines(monkeypatch)
    root = decide_gate(tiny_gallery)
    _set_manifest(root, tower_sha256=build_engine("tiny").tower_sha256())
    n_test = json.loads((root / "manifest.json").read_text())["counts"]["test"]
    with TestClient(create_app(engine="real", models=(PUBLISHED_MODEL,), home=str(tmp_path / "home"), gallery_dir=str(root),
                               labeler=RuleLabeler(), published_dirs=_dumps(tmp_path, n_test), allow_compile=True)) as c:
        published = _ends(_turn(c, {"test_row": 2}, image=None))   # the server's defaults: beam 3, 100 tokens, no stop on repeat
        live = published["generate"]["detail"]
        assert (live["decode"], live["beam_size"], live["stopped"]) == ("beam", 3, "budget")
        block = published["score"]["detail"]["published"]
        assert block == {"model_report": "SYNTHETIC model line 2", "floor_report": "SYNTHETIC floor line 2",
                         "live_equals_published": False}
        report = _turn(c, {"test_row": 2}, image=None)[-1]["data"]["report"]
        dumps = c.app.state.published
        c.app.state.worker.pipeline.published = PublishedDumps([report if i == 2 else line for i, line in enumerate(dumps.model_hyps)],
                                                               dumps.floor_hyps)
        assert _ends(_turn(c, {"test_row": 2}, image=None))["score"]["detail"]["published"]["live_equals_published"] is True
        for other in ({"stop_on_repeat": True}, {"beam_size": 2}, {"decode": "greedy"}, {"max_new_tokens": 16}, {"compile": True}):
            block = _ends(_turn(c, dict(other, test_row=2), image=None))["score"]["detail"]["published"]
            assert block["model_report"] == report and block["live_equals_published"] is None, other


# ---- public mode (R1, U2) -------------------------------------------------------------------------------------------------

def test_public_turns_keep_retrieval_as_rank_and_similarity_only_and_never_score(tmp_path, gallery):
    with TestClient(create_app(engine="tiny", home=str(tmp_path), mode="public", token="t", gallery=gallery,
                               labeler=RuleLabeler())) as c:
        sid = _session(c, PUBLIC)
        r = _post(c, sid, {"test_row": 0}, image=None, headers=PUBLIC)
        assert r.status_code == 403
        frames = _turn(c, {"max_new_tokens": 16, "reference": REF}, headers=PUBLIC, sid=sid)
        ends = _ends(frames)
        assert [f["data"]["code"] for f in frames if f["event"] == "warning"] == ["reference_ignored_public"]
        assert "score" not in ends and "score" not in _starts(frames)                     # no score event at all
        detail = ends["retrieve"]["detail"]
        assert detail["image_neighbors"] and all(set(n) == {"rank", "similarity"} for n in detail["image_neighbors"])
        assert detail["report_matches"] and all(set(m) == {"rank", "similarity"} for m in detail["report_matches"])
        assert detail["gallery"] == {k: v for k, v in gallery.facts().items() if k != "build_id"}
        assert "neighbor_agreement" not in ends["label"]["detail"] and ends["label"]["detail"]["chexbert_14"]
        identical = _turn(c, image=gallery.test_study(2)["image"].read_bytes(), headers=PUBLIC, sid=sid)   # a test image, uploaded
        assert "identical_to" not in _ends(identical)["preprocess"]["detail"]
        assert "true_report_rank" not in _ends(identical)["retrieve"]["detail"] and "score" not in _ends(identical)
        stored = [e["data"] for f in (frames, identical)
                  for e in c.get("/v1/messages/{}".format(f[0]["data"]["message_id"]), headers=PUBLIC).json()["events"]]
        dumped = json.dumps(stored)
        for text in gallery.report_texts:   # no report of the gallery, and none of its private keys, in what was sent and stored
            assert text not in dumped
        assert not _keys_anywhere(stored) & {"study_id", "gallery_row", "txt_row", "group", "group_size", "image_url", "labels",
                                            "neighbor_agreement", "true_report_rank", "identical_to", "test_row", "reference",
                                            "reference_chexbert_14", "published", "build_id"}


# ---- the endpoints -------------------------------------------------------------------------------------------------------

def _retrieve(c, data=None, headers=None, **form):
    return c.post("/v1/retrieve", files={"image": ("x.png", png_bytes() if data is None else data, "image/png")},
                  data={k: str(v) for k, v in form.items()}, headers=headers)


def test_post_retrieve_returns_the_retrieve_detail_a_turn_would_show(client, gallery):
    r = _retrieve(client, k_images=3, k_reports=2)
    assert r.status_code == 200, r.text
    query = _pooled(client, png_bytes())
    assert r.json() == {"image_neighbors": gallery.image_neighbors(query, 3), "report_matches": gallery.report_matches(query, 2),
                        "gallery": gallery.facts()}
    defaults = _retrieve(client).json()
    assert (len(defaults["image_neighbors"]), len(defaults["report_matches"])) == (4, 3)   # the turn's defaults
    data = gallery.test_study(1)["image"].read_bytes()
    assert _retrieve(client, data).json()["true_report_rank"] == gallery.own_report_rank(_pooled(client, data), 1)
    assert client.app.state.worker.in_flight == 0   # its slot is given back


def test_post_retrieve_refusals(client, monkeypatch, tmp_path):
    assert _retrieve(client, k_images=13).status_code == 422
    assert _retrieve(client, k_reports=-1).status_code == 422
    r = _retrieve(client, b"not an image at all")
    assert r.status_code == 422 and r.json()["error"]["type"] == "validation_error"
    assert _retrieve(client, bytes(server.MAX_UPLOAD_BYTES + 1)).status_code == 413
    with monkeypatch.context() as m:
        m.setattr(client.app.state.worker, "reserve", lambda: False)
        assert _retrieve(client).status_code == 429
    with TestClient(create_app(engine="tiny", home=str(tmp_path / "bare"))) as bare:
        r = _retrieve(bare)
        assert r.status_code == 503 and r.json()["error"] == {"type": "unavailable_error",
                                                              "message": "Retrieval needs the gallery, which this server has not loaded."}


def test_post_retrieve_in_public_mode_is_rank_and_similarity_only(tmp_path, gallery):
    with TestClient(create_app(engine="tiny", home=str(tmp_path), mode="public", token="t", gallery=gallery)) as c:
        body = _retrieve(c, gallery.test_study(1)["image"].read_bytes(), headers=PUBLIC).json()
        assert all(set(n) == {"rank", "similarity"} for n in body["image_neighbors"] + body["report_matches"])
        assert set(body) == {"image_neighbors", "report_matches", "gallery"} and "build_id" not in body["gallery"]


def test_post_label_labels_a_text_and_refuses_without_a_working_labeller(client, tmp_path, gallery):
    r = client.post("/v1/label", json={"text": HYP})
    assert r.status_code == 200 and r.json() == {"chexbert_14": _named(RuleLabeler().label([HYP])[0])}
    for body in ({"text": ""}, {"text": "x" * 20001}, {}, {"text": HYP, "more": 1}):
        assert client.post("/v1/label", json=body).status_code == 422, body
    assert client.post("/v1/label", json={"text": "x" * 20000}).status_code == 200
    with TestClient(create_app(engine="tiny", home=str(tmp_path / "none"))) as c:
        r = c.post("/v1/label", json={"text": HYP})
        assert r.status_code == 503 and r.json()["error"] == {
            "type": "unavailable_error", "message": "Labels need a CheXbert labeller, which this server does not have."}
    with TestClient(create_app(engine="tiny", home=str(tmp_path / "down"), labeler=_Down())) as c:
        r = c.post("/v1/label", json={"text": HYP})
        assert r.status_code == 503 and r.json()["error"]["message"] == "The CheXbert labeller did not answer; try again shortly."


def test_the_label_endpoints_bound_is_the_labeller_services_own():
    from app.labeler import MAX_CHARS
    assert server.MAX_LABEL_CHARS == MAX_CHARS   # a longer text would come back from the service as an HTTP 422, read as "down"


def test_get_test_studies_lists_the_picker_in_private_mode_only(client, gallery, tmp_path):
    assert client.get("/v1/test-studies").json() == {"studies": gallery.list_test_studies("", 50)}
    first = gallery.list_test_studies("", 1)[0]
    prefix = str(first["study_id"])[:-1]
    assert client.get("/v1/test-studies", params={"q": prefix, "limit": 3}).json() == {
        "studies": gallery.list_test_studies(prefix, 3)}
    assert client.get("/v1/test-studies", params={"limit": 0}).status_code == 422
    with TestClient(create_app(engine="tiny", home=str(tmp_path / "pub"), mode="public", token="t", gallery=gallery)) as c:
        for params in ({}, {"limit": 0}):
            r = c.get("/v1/test-studies", params=params, headers=PUBLIC)
            assert r.status_code == 403 and r.json()["error"] == {
                "type": "permission_error", "message": "Test-split studies are not available in public mode."}
    with TestClient(create_app(engine="tiny", home=str(tmp_path / "bare"))) as c:
        r = c.get("/v1/test-studies")
        assert r.status_code == 503 and r.json()["error"]["message"] == \
            "Test-split studies need the gallery, which this server has not loaded."


# ---- wiring --------------------------------------------------------------------------------------------------------------

def test_models_says_retrieval_and_labels_once_a_gallery_and_a_labeller_are_wired(client):
    listed = client.get("/v1/models").json()
    assert listed["features"] == {"retrieval": True, "labels": True}
    assert [m["features"] for m in listed["models"]] == [{"retrieval": True, "labels": True}]


def test_models_says_per_model_whether_each_runs_retrieval(tmp_path, tiny_gallery, monkeypatch):
    # M4, fix round 1: the gallery serves only the models of its tower, so retrieval is a property of each model
    _fake_real_engines(monkeypatch, other_tower=("hybrid_150m_v2_rrg",))
    root = decide_gate(tiny_gallery)
    _set_manifest(root, tower_sha256=build_engine("tiny").tower_sha256())
    with TestClient(create_app(engine="real", models=("hybrid_150m_m3_rrg", "hybrid_150m_v2_rrg"), home=str(tmp_path),
                               gallery_dir=str(root), labeler=RuleLabeler())) as c:
        listed = c.get("/v1/models").json()
        assert {m["name"]: m["features"] for m in listed["models"]} == {
            "hybrid_150m_m3_rrg": {"retrieval": True, "labels": True}, "hybrid_150m_v2_rrg": {"retrieval": False, "labels": True}}
        assert listed["features"] == {"retrieval": True, "labels": True}   # the default model's, as before


def test_tiny_gallery_builds_decides_and_opens_a_synthetic_gallery_with_the_rule_labeller(tmp_path):
    home = tmp_path / "home"
    with TestClient(create_app(engine="tiny", home=str(home), tiny_gallery=True)) as c:
        gallery = c.app.state.gallery
        assert isinstance(gallery, Gallery) and gallery.build_id == "tiny" and gallery.dim == 16   # the tiny engine's pooled width
        assert (home / "gallery" / "tiny" / "manifest.json").is_file()
        assert json.loads((home / "gallery" / "tiny" / "manifest.json").read_text())["gate_rk"]["equal"] is True
        assert isinstance(c.app.state.labeler, RuleLabeler)
        assert c.get("/v1/models").json()["features"] == {"retrieval": True, "labels": True}
        frames = _turn(c, {"max_new_tokens": 16, "test_row": 0}, image=None)
        assert {s: ("skipped" in d) for s, d in _ends(frames).items()} == dict.fromkeys(STAGES, False)
        assert len(_ends(frames)["label"]["detail"]["neighbor_agreement"]) == 4
    with TestClient(create_app(engine="tiny", home=str(home), tiny_gallery=True)) as c:   # a restart reuses the build
        assert c.app.state.gallery.facts() == gallery.facts()


def test_injected_objects_win_and_without_any_the_tiny_server_runs_neither_stage(tmp_path, gallery):
    labeler = RuleLabeler()
    with TestClient(create_app(engine="tiny", home=str(tmp_path / "a"), gallery=gallery, labeler=labeler,
                               labeler_url="http://127.0.0.1:9", tiny_gallery=True)) as c:
        assert c.app.state.gallery is gallery and c.app.state.labeler is labeler
    with TestClient(create_app(engine="tiny", home=str(tmp_path / "b"))) as c:
        assert c.app.state.gallery is None and c.app.state.labeler is None
        assert c.get("/v1/models").json()["features"] == {"retrieval": False, "labels": False}
    with TestClient(create_app(engine="tiny", home=str(tmp_path / "c"), labeler_url="http://127.0.0.1:9")) as c:
        assert isinstance(c.app.state.labeler, LabelerClient)


def test_a_real_engine_opens_the_gallery_with_its_own_tower_hash(tmp_path, tiny_gallery, monkeypatch, capsys):
    _fake_real_engines(monkeypatch)
    seen = []
    real_open = Gallery.open.__func__

    def spy(cls, root, expect_tower_sha256=None):
        seen.append(expect_tower_sha256)
        return real_open(cls, root, expect_tower_sha256)

    monkeypatch.setattr(Gallery, "open", classmethod(spy))
    root = decide_gate(tiny_gallery)
    tower = build_engine("tiny").tower_sha256()
    _set_manifest(root, tower_sha256=tower)
    with TestClient(create_app(engine="real", home=str(tmp_path / "a"), gallery_dir=str(root), labeler=RuleLabeler())) as c:
        assert seen == [tower]                                        # the engine's own hash, never None
        assert c.get("/v1/models").json()["features"]["retrieval"] is True
        assert _ends(_turn(c, {"max_new_tokens": 16, "cached_decode": True}))["retrieve"]["detail"]["image_neighbors"]
    _set_manifest(root, tower_sha256="0" * 64)                        # a gallery of another tower
    with TestClient(create_app(engine="real", home=str(tmp_path / "b"), gallery_dir=str(root), labeler=RuleLabeler())) as c:
        assert c.app.state.gallery is None and c.get("/v1/models").json()["features"]["retrieval"] is False
        assert _ends(_turn(c))["retrieve"]["skipped"] == "gallery_unavailable"
    out = capsys.readouterr().out
    assert "[server] gallery unavailable: manifest.json: tower_sha256 is not the engine's" in out
    assert str(root) not in out   # R7: a basename-only reason, never the path


def test_a_second_model_whose_tower_differs_gets_no_retrieval(tmp_path, tiny_gallery, monkeypatch):
    _fake_real_engines(monkeypatch, other_tower=("hybrid_150m_v2_rrg",))
    root = decide_gate(tiny_gallery)
    _set_manifest(root, tower_sha256=build_engine("tiny").tower_sha256())
    with TestClient(create_app(engine="real", models=("hybrid_150m_m3_rrg", "hybrid_150m_v2_rrg"), home=str(tmp_path),
                               gallery_dir=str(root), labeler=RuleLabeler())) as c:
        assert _ends(_turn(c))["retrieve"]["detail"]["image_neighbors"]
        other = _turn(c, {"max_new_tokens": 16, "model": "hybrid_150m_v2_rrg"})
        assert _ends(other)["retrieve"]["skipped"] == "gallery_unavailable" and other[-1]["data"]["status"] == "done"


def test_a_gallery_dir_that_cannot_be_opened_leaves_the_server_running_without_retrieval(tmp_path, capsys):
    with TestClient(create_app(engine="tiny", home=str(tmp_path / "home"), gallery_dir=str(tmp_path / "missing"))) as c:
        assert c.app.state.gallery is None
        assert _ends(_turn(c))["retrieve"]["skipped"] == "gallery_unavailable"
    assert "[server] gallery unavailable: manifest.json is missing" in capsys.readouterr().out


# ---- fix round 1: the test-split file, the worker's slots, a failed retrieval, a labeller that is down -----------------------

def test_a_follow_up_whose_test_image_has_vanished_is_refused_before_the_stream_and_no_path_is_logged(client, gallery, caplog):
    # M1: the real layout puts subject and study ids in the path
    sid = _session(client)
    _turn(client, {"max_new_tokens": 16, "test_row": 4}, image=None, sid=sid)
    image = gallery.test_study(4)["image"]
    image.rename(image.with_name("moved.jpg"))
    with caplog.at_level(logging.DEBUG):
        r = _post(client, sid, {"max_new_tokens": 16}, image=None, text="beam 2")
    assert r.status_code == 422 and r.json()["error"]["message"] == "Attach an X-ray first."   # as an upload gone from disk
    assert [m["role"] for m in client.get("/v1/sessions/{}".format(sid)).json()["messages"]] == ["user", "assistant"]
    assert str(image) not in caplog.text and image.name not in caplog.text and str(image.parent) not in caplog.text


def test_a_test_image_that_vanishes_once_the_turn_is_accepted_ends_it_with_a_fixed_error_and_no_path_in_the_log(tmp_path, gallery,
                                                                                                                 caplog):
    store = Store(tmp_path / "home")
    try:
        engine = build_engine("tiny")
        pipe = Pipeline({"tiny": engine}, "tiny", store, "private", gallery=gallery, labeler=RuleLabeler())
        session = store.create_session("private")
        uid, mid = store.start_turn(session["id"], "", "private", {"test_row": 4}, "ab" * 32, None, 4)
        image = gallery.test_study(4)["image"]
        image.unlink()   # after the server read and checked it, before preprocess reads it again
        sent = []
        with caplog.at_level(logging.DEBUG):
            pipe.run(TurnJob(session["id"], uid, mid, "", None, None, Options(max_new_tokens=16, test_row=4), mode="private",
                             previous_sha256="ab" * 32), lambda event, data: sent.append((event, data)), Stop())
        assert [e for e, _ in sent][-2:] == ["error", "message_stop"] and sent[-1][1]["status"] == "error"
        assert sent[-2][1]["error"] == {"type": "model_error", "message": "Could not read the test-split image."}
        assert str(image) not in caplog.text and image.name not in caplog.text and str(image.parent) not in caplog.text
        assert "Could not read the test-split image." in caplog.text   # the operator still sees what happened
    finally:
        store.close()


def test_a_cancelled_queued_task_gives_its_slot_back():
    # M2: a future cancelled while it waits never runs, so its slot cannot be given back from inside the task
    worker = Worker(None, cap=4)
    gate = threading.Event()
    try:
        assert worker.reserve()
        blocker = worker.run_task(gate.wait)   # holds the one worker thread
        assert worker.reserve()
        queued = worker.run_task(lambda: "never runs")
        assert worker.in_flight == 2
        assert queued.cancel()
        assert worker.in_flight == 1
        gate.set()
        blocker.result(timeout=5)
        wait_until(lambda: worker.in_flight == 0, timeout=5)
    finally:
        gate.set()
        worker.shutdown()


def test_a_retrieval_that_fails_is_a_500_in_the_envelope_and_logs_the_class_only(client, monkeypatch, caplog):
    # M3
    def boom(*args):
        raise RuntimeError("SECRET /sc/home/user/dataset/p10/p10000032/s50414267/x.jpg")

    monkeypatch.setattr(client.app.state.worker.pipeline, "retrieve_upload", boom)
    with caplog.at_level(logging.DEBUG):
        r = _retrieve(client)
    assert r.status_code == 500 and r.json() == {"type": "error", "error": {"type": "internal_error",
                                                                             "message": "The retrieval could not be run."}}
    assert "RuntimeError" in caplog.text and "SECRET" not in caplog.text
    assert client.app.state.worker.in_flight == 0


class _Probe:
    """A labeller whose health check says what it is told, and counts how often it was asked."""

    def __init__(self, healthy):
        self.up, self.checks = healthy, 0

    def healthy(self):
        self.checks += 1
        return self.up

    def label(self, texts):
        return RuleLabeler().label(texts)


def test_a_labeller_found_down_is_not_asked_again_for_a_while(tmp_path, gallery, monkeypatch):
    # M6: a negative health check is kept for a short time, so a labeller that is down does not stall every turn
    now = [1000.0]
    monkeypatch.setattr(pipeline_module, "_clock", lambda: now[0])
    probe = _Probe(healthy=False)
    with TestClient(create_app(engine="tiny", home=str(tmp_path), gallery=gallery, labeler=probe)) as c:
        assert _ends(_turn(c))["label"]["skipped"] == "labeler_unavailable" and probe.checks == 1
        now[0] += pipeline_module.LABELER_DOWN_TTL_S - 1
        assert _ends(_turn(c))["label"]["skipped"] == "labeler_unavailable" and probe.checks == 1   # not asked again yet
        now[0] += 2
        probe.up = True
        assert "detail" in _ends(_turn(c))["label"] and probe.checks == 2   # asked again once the time is up, and it is back


def test_a_labeller_found_up_is_asked_again_at_every_turn(tmp_path, gallery, monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(pipeline_module, "_clock", lambda: now[0])
    probe = _Probe(healthy=True)
    with TestClient(create_app(engine="tiny", home=str(tmp_path), gallery=gallery, labeler=probe)) as c:
        _turn(c)
        _turn(c)
        assert probe.checks == 2
        probe.up = False
        assert _ends(_turn(c))["label"]["skipped"] == "labeler_unavailable" and probe.checks == 3


def test_a_labeller_that_fails_a_call_is_left_alone_for_a_while_too(tmp_path, gallery, monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(pipeline_module, "_clock", lambda: now[0])
    down = _Down(healthy=True)
    with TestClient(create_app(engine="tiny", home=str(tmp_path), gallery=gallery, labeler=down)) as c:
        assert _ends(_turn(c))["label"]["skipped"] == "labeler_unavailable" and down.calls == 1
        assert _ends(_turn(c))["label"]["skipped"] == "labeler_unavailable" and down.calls == 1   # skipped before the stage
        now[0] += pipeline_module.LABELER_DOWN_TTL_S + 1
        _turn(c)
        assert down.calls == 2
