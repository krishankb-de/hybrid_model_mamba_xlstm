"""CHAT_UI_PLAN.md P3-C: nothing MIMIC-derived leaves the cluster in public mode (R1)."""
import copy
import json
import logging
import re
from pathlib import PurePosixPath

import pytest

from app import redact
from app.redact import PUBLIC_DROP, PUBLIC_ERROR_MESSAGE, redact_card, redact_event

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

PLAN_R1_KEYS = ["image_url", "study_id", "subject_id", "gallery_row", "txt_row", "group", "group_size",
                "neighbor_agreement", "reference", "test_row", "identical_to", "true_report_rank"]
# P5-E: the score stage's reference labels and published lines (a dump line, a retrieved MIMIC report) are R1 data wherever they sit.
P5E_KEYS = ["reference_chexbert_14", "published", "model_report", "floor_report"]
R1_KEYS = PLAN_R1_KEYS + P5E_KEYS


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

PLAN_POLICY = {
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


def test_public_drop_is_the_plans_policy_plus_the_one_ruled_entry():
    # M3 (ruling): public score is always skipped and a skipped stage emits only stage_end, so a public stream must
    # never show stage_start(score) either
    assert PUBLIC_DROP == {**PLAN_POLICY, "stage_start:score": "*"}


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
    monkeypatch.setattr(redact, "CATCH_ALL_KEYS", frozenset())   # the explicit policy has to do this on its own,
    monkeypatch.setattr(redact, "U2_LISTS", frozenset())         # without either invariant behind it
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


def test_a_public_stream_never_shows_the_score_stage_start_or_end_but_every_other_stage_is_shown():   # M3
    start, end = {"stage": "score", "index": 5}, {"stage": "score", "ms": 1.0, "detail": {"rouge_l": 0.2}}
    skipped = {"stage": "score", "skipped": "no_reference"}
    for event, data in (("stage_start", start), ("stage_end", end), ("stage_end", skipped)):
        assert redact_event(event, dict(data), "public") is None
        assert redact_event(event, dict(data), "private") == data   # private mode keeps the whole stage
    shown = [("stage_start", {"stage": "generate", "index": 3}), ("stage_start", {"stage": "label", "index": 4}),
             ("stage_end", {"stage": "label", "ms": 1.0}), ("stage_end", {"stage": "generate", "ms": 9.0})]
    for event, data in shown:
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


NESTED_CARD = dict(MODEL_CARD, retrieval={"name": "13D", "checkpoint": "/sc/home/SECRET-USER/kd/x.ckpt",
                                          "towers": [{"checkpoint": "/sc/home/SECRET-USER/kd/y.ckpt", "dim": 512},
                                                     {"checkpoint": None}]},
                   extra=({"deeper": {"checkpoint": PurePosixPath("/sc/home/SECRET-USER/z.ckpt")}},))


def test_redact_card_cuts_every_checkpoint_key_at_any_depth():   # M4
    out = redact_card(copy.deepcopy(NESTED_CARD), "public")
    assert out["checkpoint"] == "last.ckpt"
    assert out["retrieval"] == {"name": "13D", "checkpoint": "x.ckpt",
                                "towers": [{"checkpoint": "y.ckpt", "dim": 512}, {"checkpoint": None}]}
    assert out["extra"][0]["deeper"]["checkpoint"] == "z.ckpt"
    assert "SECRET-USER" not in json.dumps(out)
    assert {k: v for k, v in out.items() if k not in ("checkpoint", "retrieval", "extra")} == \
        {k: v for k, v in MODEL_CARD.items() if k != "checkpoint"}


def test_the_nested_card_is_unchanged_in_private_mode_and_never_mutated():
    card = copy.deepcopy(NESTED_CARD)
    assert redact_card(card, "private") == NESTED_CARD
    redact_card(card, "public")
    assert card == NESTED_CARD


def test_the_model_block_of_message_start_is_cut_at_any_depth_too():
    data = dict(MESSAGE_START, model=copy.deepcopy(NESTED_CARD))
    out = redact_event("message_start", data, "public")
    assert out["model"]["retrieval"]["checkpoint"] == "x.ckpt"
    assert out["model"]["retrieval"]["towers"][0]["checkpoint"] == "y.ckpt"
    assert "SECRET-USER" not in json.dumps(out)


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
                                         "true_report_rank": {"rank": 3, "of": 2663},
                                         "reference_chexbert_14": {"Edema": 1},
                                         "published": {"model_report": "SECRET-dump", "live_equals_published": True},
                                         "model_report": "SECRET-model-line", "floor_report": "SECRET-floor-line"}]}},
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


def test_the_catch_all_walks_tuples_and_the_policy_paths_walk_them_too(monkeypatch):
    monkeypatch.setattr(redact, "U2_LISTS", frozenset())   # so the explicit path alone has to reach into the tuple
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


# ---- free text in error events (I1) ------------------------------------------------------------------------------

LEAKY = "Findings: SECRET MIMIC TEXT; /sc/home/krishankumar.bhushan/chat_sessions/gallery/g1/labels.npy"
AUTHORED_KINDS = ["validation_error", "overloaded_error", "server_restart"]


def _error(block, **top):
    return dict({"type": "error", "error": block}, **top)


def test_the_fixed_error_message_and_the_authored_kinds_are_the_rulings():
    assert PUBLIC_ERROR_MESSAGE == "The model could not finish this turn."
    assert redact.AUTHORED_ERROR_KINDS == frozenset(AUTHORED_KINDS)


def test_a_model_error_with_report_text_and_a_path_becomes_exactly_the_fixed_message():
    out = redact_event("error", _error({"type": "model_error", "message": LEAKY}), "public")
    assert out == _error({"type": "model_error", "message": PUBLIC_ERROR_MESSAGE})
    assert "SECRET" not in json.dumps(out) and "/sc/home" not in json.dumps(out)


@pytest.mark.parametrize("block", [{"type": "weird_error", "message": LEAKY}, {"message": LEAKY},
                                   {"type": None, "message": LEAKY}, {"type": "", "message": LEAKY},
                                   {"type": "Validation_Error", "message": LEAKY},   # kinds are compared exactly
                                   {"type": ["validation_error"], "message": LEAKY},  # unhashable: must not raise
                                   {"type": {"k": 1}, "message": LEAKY}, {"type": 7, "message": LEAKY}])
def test_an_unknown_or_missing_error_kind_gets_the_fixed_message(block):
    out = redact_event("error", _error(copy.deepcopy(block), seq=9), "public")
    assert out == _error(dict(block, message=PUBLIC_ERROR_MESSAGE), seq=9)   # only the message changed


@pytest.mark.parametrize("kind", AUTHORED_KINDS)
def test_an_authored_error_kind_keeps_its_message_in_public_mode(kind):
    data = _error({"type": kind, "message": "k_images must be at most 12."}, seq=3)
    assert redact_event("error", copy.deepcopy(data), "public") == data


@pytest.mark.parametrize("kind", AUTHORED_KINDS + ["model_error", "weird_error", None])
def test_private_mode_keeps_every_error_message(kind):
    block = {"message": LEAKY} if kind is None else {"type": kind, "message": LEAKY}
    data = _error(block, seq=3)
    assert redact_event("error", copy.deepcopy(data), "private") == data


@pytest.mark.parametrize("data", [{"type": "error"}, _error(None), _error({"type": "model_error"}), _error({})])
def test_an_error_event_with_no_message_to_scrub_passes_through(data):
    assert redact_event("error", copy.deepcopy(data), "public") == data


def test_warning_messages_are_left_alone_because_the_server_authors_them():
    data = {"code": "reference_ignored_public", "message": "A reference is not used in public mode.", "seq": 4}
    assert redact_event("warning", copy.deepcopy(data), "public") == data


# ---- U2: retrieval is rank and similarity only, wherever the lists sit (M2) -------------------------------------

def _neighbour(**extra):
    return dict({"rank": 1, "similarity": 0.9, "dicom_id": "SECRET-dicom", "view": "SECRET-view",
                 "report": "SECRET-report", "labels": {"Edema": 1}, "meta": {"study": "SECRET-study"}}, **extra)


U2_SITES = [   # (event, wrap(name, items)): the plan's place, outside detail, another stage, another event, deep
    ("stage_end", lambda n, v: {"stage": "retrieve", "ms": 1.0, "detail": {n: v}}),
    ("stage_end", lambda n, v: {"stage": "retrieve", "ms": 1.0, n: v}),
    ("stage_end", lambda n, v: {"stage": "label", "ms": 1.0, "detail": {n: v}}),
    ("stage_end", lambda n, v: {"stage": "mystery", "detail": {"deeper": [{"x": {n: v}}]}}),
    ("brand_new_event", lambda n, v: {"payload": {n: v}}),
    ("message_stop", lambda n, v: {"status": "done", n: v}),
    ("message_start", lambda n, v: {"message_id": "msg_2", "image": {"urls": {}}, "options": {n: v}}),
]


@pytest.mark.parametrize("name", ["image_neighbors", "report_matches"])
@pytest.mark.parametrize("event, wrap", U2_SITES, ids=["plan", "outside_detail", "label_stage", "deep", "unknown_event",
                                                       "message_stop", "message_start"])
def test_retrieval_list_elements_are_reduced_to_rank_and_similarity_wherever_the_list_sits(event, wrap, name):
    items = [_neighbour(), "stray", None, 5, [{"rank": 9}], _neighbour(rank=2, similarity=0.5), {"unknown": 1},
             {"similarity": 0.3, "x": 2}, {}]
    out = redact_event(event, wrap(name, items), "public")
    reduced = [{"rank": 1, "similarity": 0.9}, {"rank": 2, "similarity": 0.5}, {}, {"similarity": 0.3}, {}]
    assert out == wrap(name, reduced)
    assert "SECRET" not in json.dumps(out)


def test_u2_names_exactly_the_two_retrieval_lists_and_the_two_kept_keys():
    assert redact.U2_LISTS == frozenset({"image_neighbors", "report_matches"})
    assert redact.U2_KEYS == ("rank", "similarity")


def test_u2_leaves_other_lists_and_empty_or_missing_retrieval_values_alone():
    data = {"neighbors": [{"report": "kept"}], "image_neighbors": [], "report_matches": None}
    assert redact_event("brand_new_event", copy.deepcopy(data), "public") == data


def test_u2_reduces_a_lone_dict_and_a_tuple_in_place_of_a_list():
    out = redact_event("brand_new_event", {"image_neighbors": {"rank": 1, "report": "SECRET"},
                                           "report_matches": ({"rank": 2, "labels": {"A": 1}}, "stray")}, "public")
    assert out == {"image_neighbors": {"rank": 1}, "report_matches": [{"rank": 2}]}


def test_private_mode_keeps_the_retrieval_lists_whole():
    data = {"stage": "retrieve", "image_neighbors": [_neighbour()], "report_matches": [_neighbour(), "stray"]}
    assert redact_event("stage_end", copy.deepcopy(data), "private") == data


def test_each_u2_removal_is_logged_with_key_and_event_but_never_the_value(caplog):
    data = {"stage": "mystery", "image_neighbors": [{"rank": 1, "dicom_id": "SECRET-D"}, "SECRET-stray"]}
    with caplog.at_level(logging.WARNING, logger="app.redact"):
        redact_event("stage_end", data, "public")
    messages = [r.getMessage() for r in _redactions(caplog)]
    assert len(messages) == 2 and all("stage_end" in m for m in messages)
    assert any("dicom_id" in m for m in messages)
    assert "SECRET" not in caplog.text


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
def test_a_missing_or_non_container_path_is_skipped_silently_through_redact_event(monkeypatch, detail):
    monkeypatch.setattr(redact, "U2_LISTS", frozenset())   # the explicit paths alone: U2 would rewrite these shapes
    data = {"stage": "retrieve", "ms": 1.0, "detail": detail}
    assert redact_event("stage_end", copy.deepcopy(data), "public") == data
    assert redact_event("stage_end", {"stage": "retrieve", "ms": 1.0}, "public") == {"stage": "retrieve", "ms": 1.0}


def test_list_elements_that_are_not_dicts_are_skipped_and_the_others_still_walked(monkeypatch):
    monkeypatch.setattr(redact, "U2_LISTS", frozenset())   # the explicit paths alone: U2 would drop the stray elements
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
    ("stage_start", {"stage": "score", "index": 5, "seq": 12}),
    ("stage_end", {"stage": "score", "ms": 1.0, "seq": 13, "detail": {"rouge_l": 0.2, "reference_source": "user"}}),
    ("message_stop", {"message_id": "msg_2", "status": "done", "total_ms": 5000.0, "report": "Heart size is normal.",
                      "display_report": "Heart size is normal.", "truncated_mid_sentence": False, "disclaimer": "d",
                      "seq": 14}),
]
EXTRA_EVENTS = [   # outside a successful turn
    ("warning", {"code": "reference_ignored_public", "message": "A reference is not used in public mode.", "seq": 15}),
    ("error", _error({"type": "validation_error", "message": "k_images must be at most 12."}, seq=16)),
    ("error", _error({"type": "model_error", "message": LEAKY}, seq=17)),
]
ALL_EVENTS = PRIVATE_TURN + EXTRA_EVENTS


def _event_id(event, data):
    return "{}:{}".format(event, data.get("stage") or (data.get("error") or {}).get("type") or "")


def _inject(data):
    """The event's data plus every R1 key at its top level and inside a dict inside a list inside a dict."""
    out = copy.deepcopy(data)
    out.update({key: "SECRET-top-" + key for key in R1_KEYS})
    out["nest"] = [dict({"keep": 1}, **{key: "SECRET-nested-" + key for key in R1_KEYS})]
    return out


@pytest.mark.parametrize("event, data", ALL_EVENTS, ids=[_event_id(e, d) for e, d in ALL_EVENTS])
def test_every_r1_key_is_removed_from_every_event_at_the_top_level_and_nested(event, data):   # M1
    injected = _inject(data)
    clean = redact_event(event, copy.deepcopy(data), "public")
    out = redact_event(event, copy.deepcopy(injected), "public")
    if clean is None:   # the score stage is dropped whole
        assert out is None
    else:
        assert not _keys_anywhere(out) & set(R1_KEYS) and "SECRET" not in json.dumps(out)
        assert out == dict(clean, nest=[{"keep": 1}])   # and nothing else about the event changed
    assert redact_event(event, copy.deepcopy(injected), "private") == injected


def _inject_u2(data):
    """The event's data plus both retrieval lists, with extra fields and stray elements, at its top level and inside
    a dict inside a list inside a dict."""
    out = copy.deepcopy(data)
    out.update(image_neighbors=[_neighbour(), "stray"], report_matches=[_neighbour(rank=2)])
    out["nest"] = [{"keep": 1, "image_neighbors": [_neighbour()], "report_matches": [_neighbour(rank=2), None]}]
    return out


@pytest.mark.parametrize("event, data", ALL_EVENTS, ids=[_event_id(e, d) for e, d in ALL_EVENTS])
def test_every_retrieval_list_is_reduced_in_every_event_at_the_top_level_and_nested(event, data):   # M2
    injected = _inject_u2(data)
    clean = redact_event(event, copy.deepcopy(data), "public")
    out = redact_event(event, copy.deepcopy(injected), "public")
    if clean is None:
        assert out is None
    else:
        one, two = [{"rank": 1, "similarity": 0.9}], [{"rank": 2, "similarity": 0.9}]
        nest = [{"keep": 1, "image_neighbors": one, "report_matches": two}]
        assert out == dict(clean, image_neighbors=one, report_matches=two, nest=nest)
        assert "SECRET" not in json.dumps(out)
    assert redact_event(event, copy.deepcopy(injected), "private") == injected


@pytest.mark.parametrize("mode", ["public", "private"])
def test_redact_event_never_mutates_its_input_and_returns_a_copy(mode):
    for event, original in ALL_EVENTS:
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
    assert [(e, d["stage"]) for (e, d), out in zip(PRIVATE_TURN, public) if out is None] == [
        ("stage_start", "score"), ("stage_end", "score")]   # the only drops: a public stream never shows score
    dumped = json.dumps([out for out in public if out is not None])
    assert "SECRET" not in dumped and "rouge_l" not in dumped
    assert not _keys_anywhere(json.loads(dumped)) & (redact.CATCH_ALL_KEYS | {"labels", "build_id"})
    for shown in ("Heart size is normal.", "similarity", "chexbert_14", "towers_identical"):
        assert shown in dumped
    assert [redact_event(e, copy.deepcopy(d), "private") for e, d in PRIVATE_TURN] == [d for _, d in PRIVATE_TURN]


# ---- P5-E: every field the retrieve, label and score stages add, private and public (R1, U2) ------------------------

NAMES = ["Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema", "Consolidation", "Pneumonia",
         "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices", "No Finding"]
P5E_START = dict(MESSAGE_START, options={"beam_size": 3, "k_images": 4, "k_reports": 3, "label": True, "reference": None,
                                         "test_row": 987605},
                 image={"sha256": "cd" * 32, "filename": None, "source": "test_split",
                        "urls": {"original": "/v1/messages/m_1/image?variant=original", "thumb": "/v1/messages/m_1/image?variant=thumb",
                                 "model_input": "/v1/messages/m_1/image?variant=model_input"}})
P5E_PREPROCESS = {"stage": "preprocess", "ms": 1.0, "seq": 3, "detail": {
    "format": "JPEG", "input_px": [320, 320], "source": "test_split", "test_row": 987605, "identical_to": {"split": "test", "row": 987605}}}
P5E_RETRIEVE = {"stage": "retrieve", "ms": 3.0, "seq": 7, "detail": {
    "image_neighbors": [{"rank": 1, "similarity": 0.83, "gallery_row": 987601, "study_id": 98760001, "txt_row": 987602,
                         "image_url": "/v1/gallery/images/987601", "labels": dict.fromkeys(NAMES, 0)},
                        {"rank": 2, "similarity": 0.81, "gallery_row": 987611, "study_id": 98760011, "txt_row": 987612,
                         "image_url": "/v1/gallery/images/987611", "labels": None}],
    "report_matches": [{"rank": 1, "similarity": 0.52, "group": 987603, "group_size": 987609, "txt_row": 987604,
                        "report": "Findings: SECRET-P5E matched report.", "labels": dict.fromkeys(NAMES, 1)}],
    "true_report_rank": {"rank": 987606, "of": 987607, "rank_dedup": 987608, "n_tied": 987610, "hit_at_10": False,
                         "protocol": "SECRET-protocol"},
    "gallery": {"build_id": "SECRET-20261010_9876", "images": 200, "report_rows": 240, "report_groups": 78, "towers_identical": True}}}
P5E_LABEL = {"stage": "label", "ms": 2.0, "seq": 14, "detail": {
    "chexbert_14": dict(dict.fromkeys(NAMES, 0), Edema=1), "positives": ["Edema"],
    "neighbor_agreement": [{"rank": 1, "agree": 13, "of": 14, "both_positive": [], "neighbor_only": [], "generated_only": ["Edema"]}]}}
# Fix round 1, ruling (a): while the gallery's labels are still being built the report is labelled and the agreement is marked pending.
P5E_LABEL_PENDING = {"stage": "label", "ms": 2.0, "seq": 17, "detail": {
    "chexbert_14": dict(dict.fromkeys(NAMES, 0), Edema=1), "positives": ["Edema"], "neighbor_agreement": [],
    "neighbor_agreement_pending": True}}
P5E_SCORE = {"stage": "score", "ms": 1.0, "seq": 16, "detail": {
    "rouge_l": 0.21, "bleu_1": 0.33, "bleu_4": 0.07, "chexbert_14_micro_f1": 0.5, "exact_match_14": False,
    "reference_source": "test_split", "reference_chexbert_14": dict(dict.fromkeys(NAMES, 0), Cardiomegaly=1),
    "published": {"model_report": "SECRET-published model line", "floor_report": "SECRET-floor report",
                  "live_equals_published": False}}}
P5E_EVENTS = [("message_start", P5E_START), ("stage_end", P5E_PREPROCESS), ("stage_start", {"stage": "retrieve", "index": 2, "seq": 6}),
              ("stage_end", P5E_RETRIEVE), ("stage_start", {"stage": "label", "index": 4, "seq": 13}), ("stage_end", P5E_LABEL),
              ("stage_start", {"stage": "score", "index": 5, "seq": 15}), ("stage_end", P5E_SCORE), ("stage_end", P5E_LABEL_PENDING),
              ("stage_end", {"stage": "retrieve", "skipped": "k_zero", "seq": 18}),
              ("stage_end", {"stage": "label", "skipped": "labeler_unavailable", "seq": 19})]
SCORE_FIELDS = ["rouge_l", "bleu_1", "bleu_4", "chexbert_14_micro_f1", "exact_match_14", "reference_source", "reference_chexbert_14",
                "published.model_report", "published.floor_report", "published.live_equals_published"]
P5E_FIELDS = (   # (event, data, path, kept in public)
    [("message_start", P5E_START, "options.test_row", False), ("message_start", P5E_START, "image.source", True)]
    + [("stage_end", P5E_PREPROCESS, "detail." + k, False) for k in ("test_row", "identical_to")]
    + [("stage_end", P5E_RETRIEVE, "detail.image_neighbors[]." + k, k in ("rank", "similarity"))
       for k in ("rank", "similarity", "gallery_row", "study_id", "txt_row", "image_url", "labels")]
    + [("stage_end", P5E_RETRIEVE, "detail.report_matches[]." + k, k in ("rank", "similarity"))
       for k in ("rank", "similarity", "group", "group_size", "txt_row", "report", "labels")]
    + [("stage_end", P5E_RETRIEVE, "detail.true_report_rank." + k, False)
       for k in ("rank", "of", "rank_dedup", "n_tied", "hit_at_10", "protocol")]
    + [("stage_end", P5E_RETRIEVE, "detail.gallery." + k, k != "build_id")
       for k in ("build_id", "images", "report_rows", "report_groups", "towers_identical")]
    + [("stage_end", P5E_LABEL, "detail." + k, k != "neighbor_agreement") for k in ("chexbert_14", "positives", "neighbor_agreement")]
    + [("stage_end", P5E_LABEL_PENDING, "detail." + k, k != "neighbor_agreement")
       for k in ("chexbert_14", "positives", "neighbor_agreement", "neighbor_agreement_pending")]
    + [("stage_end", P5E_SCORE, "detail." + k, False) for k in SCORE_FIELDS]
    + [("stage_end", data, "skipped", True) for event, data in P5E_EVENTS[-2:]])


def _at(obj, path):
    """Every value at a dotted path ("[]" walks a list); [] when there is none."""
    found = [obj]
    for part in path.split("."):
        walk, key = part.endswith("[]"), part[:-2] if part.endswith("[]") else part
        step = []
        for item in found:
            if isinstance(item, dict) and key in item:
                if not walk:
                    step.append(item[key])
                elif isinstance(item[key], list):
                    step.extend(item[key])
        found = step
    return found


def _field_id(event, data, path, public):
    return "{}:{} {}".format(event, data.get("stage", ""), path)


@pytest.mark.parametrize("event, data, path, public", P5E_FIELDS, ids=[_field_id(*f) for f in P5E_FIELDS])
def test_each_p5e_field_is_kept_whole_in_private_mode(event, data, path, public):
    assert _at(data, path), path   # the fixture really carries the field
    assert _at(redact_event(event, copy.deepcopy(data), "private"), path) == _at(data, path)


@pytest.mark.parametrize("event, data, path, public", P5E_FIELDS, ids=[_field_id(*f) for f in P5E_FIELDS])
def test_each_p5e_field_is_dropped_in_public_mode_unless_r1_lets_it_through(event, data, path, public):
    out = redact_event(event, copy.deepcopy(data), "public")
    if public:
        assert _at(out, path) == _at(data, path)
    else:
        assert out is None or _at(out, path) == []


def test_the_public_score_stage_is_gone_whole_with_its_reference_labels_and_published_lines():
    assert redact_event("stage_end", copy.deepcopy(P5E_SCORE), "public") is None
    assert redact_event("stage_start", {"stage": "score", "index": 5}, "public") is None


def test_no_private_value_of_a_p5e_turn_survives_into_a_public_payload(caplog):
    with caplog.at_level(logging.DEBUG, logger="app.redact"):
        public = [redact_event(e, copy.deepcopy(d), "public") for e, d in P5E_EVENTS]
    assert _redactions(caplog) == []   # PUBLIC_DROP covers every P5-E shape on its own: the catch-all is the second line
    dumped = json.dumps([out for out in public if out is not None])
    assert "SECRET" not in dumped and "98760" not in dumped   # every private string and every private number of the fixture
    assert not _keys_anywhere(json.loads(dumped)) & (redact.CATCH_ALL_KEYS | {"labels", "build_id", "report"})
    assert [redact_event(e, copy.deepcopy(d), "private") for e, d in P5E_EVENTS] == [d for _, d in P5E_EVENTS]


@pytest.mark.parametrize("key", P5E_KEYS)
def test_the_score_fields_are_caught_outside_the_score_stage_too(key):
    data = {"stage": "label", "ms": 1.0, "detail": {"chexbert_14": {"Edema": 1}, key: {"SECRET": 1}}}
    assert redact_event("stage_end", copy.deepcopy(data), "public") == {"stage": "label", "ms": 1.0, "detail": {"chexbert_14": {"Edema": 1}}}
    assert redact_event("stage_end", copy.deepcopy(data), "private") == data


def test_the_pending_agreement_marker_is_server_state_that_both_modes_keep():   # fix round 1, ruling (a)
    for mode in ("public", "private"):
        out = redact_event("stage_end", copy.deepcopy(P5E_LABEL_PENDING), mode)
        assert out["detail"]["neighbor_agreement_pending"] is True and out["detail"]["chexbert_14"] == P5E_LABEL_PENDING["detail"]["chexbert_14"]
        assert ("neighbor_agreement" in out["detail"]) is (mode == "private")   # the agreement itself stays private
    assert "neighbor_agreement_pending" not in redact.CATCH_ALL_KEYS
