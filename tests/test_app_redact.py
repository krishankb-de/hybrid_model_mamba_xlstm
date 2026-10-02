"""CHAT_UI_PLAN.md P3-C: nothing MIMIC-derived leaves the cluster in public mode (R1)."""
import copy
import json
import logging
import re
from pathlib import PurePosixPath

import pytest

from app import redact
from app.redact import PUBLIC_DROP, redact_card, redact_event

PRIVATE_RETRIEVE = {"stage": "retrieve", "ms": 4.0, "detail": {
    "image_neighbors": [{"rank": 1, "similarity": 0.91, "gallery_row": 1843, "image_url": "/v1/gallery/images/1843",
                         "study_id": "50414267", "labels": {"Edema": 1}}],
    "report_matches": [{"rank": 1, "similarity": 0.41, "group": 88213, "group_size": 4,
                        "report": "Findings: SECRET MIMIC TEXT", "labels": {"Edema": 1}}],
    "true_report_rank": {"rank": 3, "of": 2663}, "gallery": {"build_id": "g1", "images": 191462}}}


def test_public_retrieve_keeps_similarity_scores_only():   # U2
    out = redact_event("stage_end", copy.deepcopy(PRIVATE_RETRIEVE), "public")
    n, m = out["detail"]["image_neighbors"][0], out["detail"]["report_matches"][0]
    assert set(n) == {"rank", "similarity"}
    assert set(m) == {"rank", "similarity"}
    assert "true_report_rank" not in out["detail"] and "build_id" not in out["detail"]["gallery"]


def test_public_label_stage_drops_neighbour_agreement_but_keeps_the_reports_own_labels():
    data = {"stage": "label", "ms": 3.0, "detail": {"chexbert_14": {"Edema": 1}, "positives": ["Edema"],
                                                    "neighbor_agreement": [{"rank": 1, "agree": 13, "of": 14}]}}
    out = redact_event("stage_end", copy.deepcopy(data), "public")
    assert "neighbor_agreement" not in out["detail"] and out["detail"]["chexbert_14"] == {"Edema": 1}


def test_no_private_string_survives_into_a_public_payload():
    out = json.dumps(redact_event("stage_end", copy.deepcopy(PRIVATE_RETRIEVE), "public"))
    for secret in ("SECRET MIMIC TEXT", "50414267", "/v1/gallery/images/1843"):
        assert secret not in out


def test_score_is_dropped_in_public_and_kept_in_private():
    data = {"stage": "score", "ms": 1.0, "detail": {"rouge_l": 0.2}}
    assert redact_event("stage_end", dict(data), "public") is None
    assert redact_event("stage_end", dict(data), "private") == data


def test_private_mode_is_a_no_op():
    assert redact_event("stage_end", copy.deepcopy(PRIVATE_RETRIEVE), "private") == PRIVATE_RETRIEVE


def test_generated_report_is_not_redacted():   # model output is not MIMIC data
    stop = {"status": "done", "report": "Findings: heart normal.", "display_report": "Findings: heart normal."}
    assert redact_event("message_stop", dict(stop), "public") == stop


# ---- helpers ----------------------------------------------------------------------------------------------------

R1_KEYS = ["image_url", "study_id", "subject_id", "gallery_row", "txt_row", "group", "group_size",
           "neighbor_agreement", "reference", "test_row", "identical_to", "true_report_rank"]


def _keys_anywhere(obj):
    """Every dict key at any depth, through dicts, lists and tuples."""
    if isinstance(obj, dict):
        found = set(obj)
        for value in obj.values():
            found |= _keys_anywhere(value)
        return found
    if isinstance(obj, (list, tuple)):
        found = set()
        for item in obj:
            found |= _keys_anywhere(item)
        return found
    return set()


def _redactions(caplog):
    return [r for r in caplog.records if r.name == "app.redact"]


# ---- the policy: one test per path, built from the policy itself -------------------------------------------------

def test_public_drop_is_the_plans_policy():
    assert PUBLIC_DROP == {
        "message_start": ["options.reference", "options.test_row", "image.urls.original"],
        "stage_end:preprocess": ["detail.test_row", "detail.identical_to"],
        "stage_end:retrieve": ["detail.image_neighbors[].image_url", "detail.image_neighbors[].study_id",
                               "detail.image_neighbors[].gallery_row", "detail.image_neighbors[].txt_row",
                               "detail.image_neighbors[].labels", "detail.report_matches[].report",
                               "detail.report_matches[].group", "detail.report_matches[].group_size",
                               "detail.report_matches[].txt_row", "detail.report_matches[].labels",
                               "detail.true_report_rank", "detail.gallery.build_id"],
        "stage_end:label": ["detail.neighbor_agreement"],
        "stage_end:score": "*",
    }


def test_policy_paths_are_dotted_names_that_end_on_a_plain_key():
    for rule in PUBLIC_DROP.values():
        for path in ([] if rule == "*" else rule):
            assert re.fullmatch(r"[a-z0-9_]+(\[\])?(\.[a-z0-9_]+(\[\])?)*", path), path
            assert not path.endswith("[]"), path


def _payload_at(path, leaf="SECRET-LEAF"):
    """A dict with the leaf at the dotted path ("[]" makes a one-element list) and a sibling that must survive."""
    head, _, rest = path.partition(".")
    walk = head.endswith("[]")
    key = head[:-2] if walk else head
    if not rest:
        return {key: leaf, "sibling": "KEEP-SIBLING"}
    inner = _payload_at(rest, leaf)
    return {key: [inner] if walk else inner}


POLICY_PATHS = [(key, path) for key, rule in PUBLIC_DROP.items() if rule != "*" for path in rule]


@pytest.mark.parametrize("policy_key, path", POLICY_PATHS, ids=["{} {}".format(k, p) for k, p in POLICY_PATHS])
def test_each_policy_path_drops_its_field_without_the_catch_all(monkeypatch, policy_key, path):
    monkeypatch.setattr(redact, "CATCH_ALL_KEYS", frozenset())   # the explicit policy has to do this on its own
    event, _, stage = policy_key.partition(":")
    data = dict(_payload_at(path), **({"stage": stage} if stage else {}))
    dumped = json.dumps(redact_event(event, data, "public"))
    assert "SECRET-LEAF" not in dumped and "KEEP-SIBLING" in dumped


# ---- message_start, preprocess, score ----------------------------------------------------------------------------

MODEL_CARD = {"name": "hybrid_150m_m3_rrg", "checkpoint": "/sc/home/SECRET-USER/repo/outputs/run/checkpoints/last.ckpt",
              "checkpoint_sha256": "cd" * 32, "prefix_k": 32, "layer_pattern": ["mamba3", "mlstm"], "device": "cpu",
              "threads": 8, "drift_note": "", "git_sha": None}
MESSAGE_START = {
    "message_id": "msg_2", "user_message_id": "msg_1", "session_id": "ses_1", "mode": "public", "model": MODEL_CARD,
    "options": {"beam_size": 3, "k_images": 4, "reference": "SECRET reference text", "test_row": 2610},
    "image": {"sha256": "ab" * 32, "filename": "scan.png", "source": "upload",
              "urls": {"original": "/v1/SECRET-URL/original.png", "thumb": "/v1/uploads/thumb.jpg",
                       "model_input": "/v1/uploads/model_input.png"}},
    "seq": 1}


def test_public_message_start_drops_the_private_options_and_the_original_url():
    out = redact_event("message_start", copy.deepcopy(MESSAGE_START), "public")
    assert out["options"] == {"beam_size": 3, "k_images": 4}
    assert out["image"] == {"sha256": "ab" * 32, "filename": "scan.png", "source": "upload",
                            "urls": {"thumb": "/v1/uploads/thumb.jpg", "model_input": "/v1/uploads/model_input.png"}}
    assert {k: out[k] for k in ("message_id", "user_message_id", "session_id", "mode", "seq")} == \
        {k: MESSAGE_START[k] for k in ("message_id", "user_message_id", "session_id", "mode", "seq")}


def test_public_message_start_cuts_the_checkpoint_path_in_the_model_block():
    out = redact_event("message_start", copy.deepcopy(MESSAGE_START), "public")
    assert out["model"] == dict(MODEL_CARD, checkpoint="last.ckpt")
    assert "SECRET" not in json.dumps(out)


def test_private_message_start_is_unchanged_including_the_checkpoint_path():
    assert redact_event("message_start", copy.deepcopy(MESSAGE_START), "private") == MESSAGE_START


@pytest.mark.parametrize("extra", [{}, {"model": None}, {"image": None}, {"options": "x"},
                                   {"image": {"urls": None}}, {"model": {"checkpoint": None}}])
def test_a_message_start_missing_its_blocks_does_not_raise(extra):
    data = {"message_id": "msg_2", "seq": 1}
    data.update(extra)
    assert redact_event("message_start", copy.deepcopy(data), "public") == data


def test_public_preprocess_drops_test_row_and_identical_to_only():
    data = {"stage": "preprocess", "ms": 2.0, "detail": {"format": "PNG", "source": "upload", "test_row": 5,
                                                         "identical_to": {"split": "train", "row": 9}}}
    assert redact_event("stage_end", copy.deepcopy(data), "public") == {
        "stage": "preprocess", "ms": 2.0, "detail": {"format": "PNG", "source": "upload"}}


def test_a_skipped_score_is_dropped_too_but_no_other_score_or_stage_event_is():
    assert redact_event("stage_end", {"stage": "score", "skipped": "no_reference"}, "public") is None
    for event, data in (("stage_start", {"stage": "score", "index": 5}), ("stage_end", {"stage": "label", "ms": 1.0})):
        assert redact_event(event, dict(data), "public") == data


# ---- the model card (ruling 1) -----------------------------------------------------------------------------------

def test_public_card_keeps_only_the_checkpoint_file_name_and_every_other_key():
    out = redact_card(copy.deepcopy(MODEL_CARD), "public")
    assert out == dict(MODEL_CARD, checkpoint="last.ckpt")
    assert "SECRET-USER" not in json.dumps(out)


def test_card_without_a_checkpoint_path_is_left_alone():
    assert redact_card(dict(MODEL_CARD, checkpoint=None), "public") == dict(MODEL_CARD, checkpoint=None)
    no_key = {k: v for k, v in MODEL_CARD.items() if k != "checkpoint"}
    assert redact_card(no_key, "public") == no_key


def test_a_path_object_checkpoint_is_cut_to_its_file_name_too():
    card = dict(MODEL_CARD, checkpoint=PurePosixPath("/sc/home/SECRET-USER/outputs/run/last.ckpt"))
    assert redact_card(card, "public")["checkpoint"] == "last.ckpt"


def test_private_card_is_unchanged():
    assert redact_card(copy.deepcopy(MODEL_CARD), "private") == MODEL_CARD


@pytest.mark.parametrize("mode", ["public", "private"])
def test_redact_card_returns_a_copy_and_never_mutates_the_card(mode):
    card = copy.deepcopy(MODEL_CARD)
    out = redact_card(card, mode)
    assert card == MODEL_CARD
    out["layer_pattern"].append("mutated")
    out["name"] = "mutated"
    assert card == MODEL_CARD


# ---- the catch-all (ruling 2) ------------------------------------------------------------------------------------

UNKNOWN_PAYLOAD = {
    "note": "kept",
    "items": [
        {"keep": 1, "image_url": "SECRET-url", "study_id": "SECRET-study", "subject_id": "SECRET-subject",
         "inner": [{"keep": 2, "gallery_row": 7, "txt_row": 8, "group": 9, "group_size": 10}]},
        {"keep": 3, "deeper": {"rows": [{"keep": 4, "neighbor_agreement": [{"rank": 1, "agree": 13}],
                                         "reference": "SECRET-reference", "test_row": 5,
                                         "identical_to": {"split": "train", "row": 6},
                                         "true_report_rank": {"rank": 3, "of": 2663}}]}},
        "plain", None, 3],
}
UNKNOWN_KEPT = {"note": "kept", "items": [{"keep": 1, "inner": [{"keep": 2}]},
                                          {"keep": 3, "deeper": {"rows": [{"keep": 4}]}}, "plain", None, 3]}


def test_the_catch_all_names_exactly_the_unambiguous_r1_keys():
    assert redact.CATCH_ALL_KEYS == frozenset(R1_KEYS)
    assert not redact.CATCH_ALL_KEYS & {"report", "labels"}   # model output uses these: path-scoped only
    assert set(R1_KEYS) <= _keys_anywhere(UNKNOWN_PAYLOAD)    # the fixture above carries every one of them


def test_catch_all_cleans_an_unknown_event():
    assert redact_event("brand_new_event", copy.deepcopy(UNKNOWN_PAYLOAD), "public") == UNKNOWN_KEPT


def test_catch_all_cleans_an_unknown_stage():
    data = {"stage": "mystery", "ms": 1.0, "detail": copy.deepcopy(UNKNOWN_PAYLOAD)}
    assert redact_event("stage_end", data, "public") == {"stage": "mystery", "ms": 1.0, "detail": UNKNOWN_KEPT}


@pytest.mark.parametrize("event, wrap", [("brand_new_event", lambda detail: {"detail": detail}),
                                         ("stage_end", lambda detail: {"stage": "mystery", "detail": detail})],
                         ids=["unknown_event", "unknown_stage"])
@pytest.mark.parametrize("key", R1_KEYS)
def test_catch_all_removes_each_key_on_its_own(event, wrap, key):
    out = redact_event(event, wrap({"rows": [{"keep": 1, key: "SECRET"}]}), "public")
    assert out == wrap({"rows": [{"keep": 1}]})


def test_report_and_labels_stay_in_unknown_events_because_only_their_policy_paths_drop_them():
    data = {"report": "generated text", "labels": {"Edema": 1}, "detail": {"report": "x", "labels": {"A": 0}}}
    assert redact_event("brand_new_event", copy.deepcopy(data), "public") == data
    staged = dict(data, stage="mystery")
    assert redact_event("stage_end", copy.deepcopy(staged), "public") == staged


def test_the_catch_all_walks_tuples_and_the_policy_paths_walk_them_too():
    data = {"stage": "retrieve", "detail": {"image_neighbors": ({"rank": 1, "labels": {"Edema": 1}},),
                                            "other": ({"keep": 2, "group": 1},)}}
    out = redact_event("stage_end", data, "public")
    assert out["detail"]["image_neighbors"][0] == {"rank": 1} and out["detail"]["other"][0] == {"keep": 2}


@pytest.mark.parametrize("stage", [None, 5, ["retrieve"], {"name": "retrieve"}])
def test_a_stage_that_is_not_a_string_selects_no_stage_rule_but_the_catch_all_still_runs(stage):
    out = redact_event("stage_end", {"stage": stage, "detail": {"study_id": "S", "rank": 1}}, "public")
    assert out == {"stage": stage, "detail": {"rank": 1}}


def test_each_catch_all_removal_is_logged_with_event_and_key_but_never_the_value(caplog):
    rows = [{"study_id": "SECRET-A"}, {"study_id": "SECRET-B", "group": "SECRET-C"}]
    data = {"stage": "mystery", "detail": {"rows": rows}}
    with caplog.at_level(logging.WARNING, logger="app.redact"):
        redact_event("stage_end", data, "public")
    records = _redactions(caplog)
    messages = [r.getMessage() for r in records]
    assert [r.levelno for r in records] == [logging.WARNING] * 3
    assert all("stage_end" in m for m in messages)
    assert sorted("study_id" if "study_id" in m else "group" for m in messages) == ["group", "study_id", "study_id"]
    assert "SECRET" not in caplog.text


def test_a_field_the_policy_already_drops_is_not_a_policy_gap_and_logs_nothing(caplog):
    with caplog.at_level(logging.DEBUG, logger="app.redact"):
        redact_event("stage_end", copy.deepcopy(PRIVATE_RETRIEVE), "public")
        redact_event("stage_end", {"stage": "score", "ms": 1.0}, "public")
    assert _redactions(caplog) == []


def test_private_mode_keeps_unknown_events_unchanged_and_logs_nothing(caplog):
    with caplog.at_level(logging.DEBUG, logger="app.redact"):
        assert redact_event("brand_new_event", copy.deepcopy(UNKNOWN_PAYLOAD), "private") == UNKNOWN_PAYLOAD
        stage = {"stage": "mystery", "detail": copy.deepcopy(UNKNOWN_PAYLOAD)}
        assert redact_event("stage_end", copy.deepcopy(stage), "private") == stage
    assert _redactions(caplog) == []


# ---- dotted paths (ruling 5) -------------------------------------------------------------------------------------

@pytest.mark.parametrize("obj, path, expected", [
    ({"a": {"b": 1, "c": 2}}, "a.b", {"a": {"c": 2}}),
    ({"a": [{"b": 1, "c": 2}, {"b": 3}]}, "a[].b", {"a": [{"c": 2}, {}]}),
    ({"a": [{"b": [{"c": 1, "d": 2}, {"c": 3}]}, {"b": None}]}, "a[].b[].c",
     {"a": [{"b": [{"d": 2}, {}]}, {"b": None}]}),
    ({}, "a.b", {}),
    ({"a": {}}, "a.b", {"a": {}}),
    ({"a": None}, "a.b", {"a": None}),
    ({"a": {"b": None}}, "a.b.c", {"a": {"b": None}}),
    ({"a": 5}, "a.b", {"a": 5}),
    ({"a": "text"}, "a[].b", {"a": "text"}),
    ({"a": {"b": 1}}, "a[].b", {"a": {"b": 1}}),       # a dict is not a list
    ({"a": [{"b": 1}]}, "a.b", {"a": [{"b": 1}]}),     # a list is not a dict
    ({"a": [1, None, "x", [], {"b": 1, "c": 2}]}, "a[].b", {"a": [1, None, "x", [], {"c": 2}]}),
])
def test_drop_walks_dotted_paths_and_skips_what_is_not_there(obj, path, expected):
    redact._drop(obj, path)
    assert obj == expected


@pytest.mark.parametrize("detail", [None, "text", 7, [], {}, {"image_neighbors": None}, {"image_neighbors": "x"},
                                    {"image_neighbors": {"rank": 1}}, {"image_neighbors": [None, 3, "x", []]},
                                    {"gallery": None}, {"gallery": "g"}, {"gallery": [1]}])
def test_a_missing_or_non_container_path_is_skipped_silently_through_redact_event(detail):
    data = {"stage": "retrieve", "ms": 1.0, "detail": detail}
    assert redact_event("stage_end", copy.deepcopy(data), "public") == data
    assert redact_event("stage_end", {"stage": "retrieve", "ms": 1.0}, "public") == {"stage": "retrieve", "ms": 1.0}


def test_list_elements_that_are_not_dicts_are_skipped_and_the_others_still_walked():
    data = {"stage": "retrieve", "detail": {"image_neighbors": [1, None, {"rank": 2, "labels": {"A": 1}}, "x"]}}
    assert redact_event("stage_end", data, "public")["detail"]["image_neighbors"] == [1, None, {"rank": 2}, "x"]


# ---- purity and modes (rulings 3 and 4) --------------------------------------------------------------------------

PRIVATE_TURN = [
    ("message_start", MESSAGE_START),
    ("stage_start", {"stage": "preprocess", "index": 0, "seq": 2}),
    ("stage_end", {"stage": "preprocess", "ms": 3.2, "seq": 3, "detail": {
        "format": "JPEG", "input_px": [2544, 3056], "resized_to": [224, 224], "source": "test_split",
        "test_row": 2610, "identical_to": {"split": "test", "row": 2610}}}),
    ("stage_end", {"stage": "encode", "ms": 1.0, "seq": 4, "detail": {"patch_grid": [197, 768], "pooled_dim": 512}}),
    ("stage_end", dict(PRIVATE_RETRIEVE, seq=5, detail={
        "image_neighbors": [{"rank": 1, "similarity": 0.91, "gallery_row": 1843, "image_url": "/v1/SECRET-URL/1843",
                             "study_id": "SECRET-STUDY", "labels": {"Edema": 1}},
                            {"rank": 2, "similarity": 0.88, "gallery_row": 77, "image_url": "/v1/SECRET-URL/77",
                             "study_id": "SECRET-STUDY-2", "labels": {"Edema": 0}}],
        "report_matches": [{"rank": 1, "similarity": 0.41, "group": 88213, "group_size": 4, "txt_row": 9,
                            "report": "Findings: SECRET MIMIC TEXT", "labels": {"Edema": 1}}],
        "true_report_rank": {"rank": 3, "of": 2663, "rank_dedup": 3, "hit_at_10": True, "protocol": "p"},
        "gallery": {"build_id": "SECRET-BUILD", "images": 191462, "report_rows": 194125, "report_groups": 180000,
                    "towers_identical": True}})),
    ("stage_start", {"stage": "generate", "index": 3, "seq": 6}),
    ("content_block_start", {"index": 0, "content_block": {"type": "report", "text": ""}, "seq": 7}),
    ("content_block_delta", {"index": 0, "delta": {"type": "beam_snapshot", "step": 5, "text": "Heart size is normal."},
                             "seq": 8}),
    ("content_block_stop", {"index": 0, "seq": 9}),
    ("stage_end", {"stage": "generate", "ms": 900.0, "seq": 10,
                   "detail": {"decode": "beam", "beam_size": 3, "tokens": 40}}),
    ("stage_end", {"stage": "label", "ms": 3.0, "seq": 11, "detail": {
        "chexbert_14": {"Edema": 1}, "positives": ["Edema"],
        "neighbor_agreement": [{"rank": 1, "agree": 13, "of": 14, "both_positive": ["Edema"]}]}}),
    ("stage_end", {"stage": "score", "ms": 1.0, "seq": 12, "detail": {"rouge_l": 0.2, "reference_source": "user"}}),
    ("message_stop", {"message_id": "msg_2", "status": "done", "total_ms": 5000.0, "report": "Heart size is normal.",
                      "display_report": "Heart size is normal.", "truncated_mid_sentence": False, "disclaimer": "d",
                      "seq": 13}),
]


@pytest.mark.parametrize("mode", ["public", "private"])
def test_redact_event_never_mutates_its_input_and_returns_a_copy(mode):
    for event, original in PRIVATE_TURN:
        data = copy.deepcopy(original)   # a module-level fixture is never handed over, so one failure cannot spread
        out = redact_event(event, data, mode)
        assert data == original
        if out is not None:
            assert out is not data
            out["MUTATED"] = True
            assert "MUTATED" not in data
    data = copy.deepcopy(PRIVATE_RETRIEVE)
    out = redact_event("stage_end", data, mode)
    out["detail"]["gallery"]["images"] = 0
    out["detail"]["image_neighbors"][0]["similarity"] = 0.0
    assert data == PRIVATE_RETRIEVE


@pytest.mark.parametrize("mode", ["Public", "", None, "both"])
def test_an_unknown_mode_is_refused_rather_than_read_as_private(mode):
    with pytest.raises(ValueError, match="mode"):
        redact_event("stage_end", {"stage": "retrieve"}, mode)
    with pytest.raises(ValueError, match="mode"):
        redact_card({"checkpoint": "/a/b"}, mode)


def test_a_whole_private_turn_is_clean_in_public_mode_and_equal_in_private_mode(caplog):
    with caplog.at_level(logging.DEBUG, logger="app.redact"):
        public = [redact_event(e, copy.deepcopy(d), "public") for e, d in PRIVATE_TURN]
    assert _redactions(caplog) == []   # PUBLIC_DROP alone covers every shape in the contract: no catch-all hit
    assert [d["stage"] for (_, d), out in zip(PRIVATE_TURN, public) if out is None] == ["score"]   # the only drop
    dumped = json.dumps([out for out in public if out is not None])
    assert "SECRET" not in dumped and "rouge_l" not in dumped
    assert not _keys_anywhere(json.loads(dumped)) & (redact.CATCH_ALL_KEYS | {"labels", "build_id"})
    for shown in ("Heart size is normal.", "similarity", "chexbert_14", "towers_identical"):
        assert shown in dumped
    assert [redact_event(e, copy.deepcopy(d), "private") for e, d in PRIVATE_TURN] == [d for _, d in PRIVATE_TURN]
