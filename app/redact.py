"""Public-mode field policy for the chat app (CHAT_UI_PLAN.md P3-C; rule R1, the DUA by construction).

Every streamed event passes through redact_event before it is stored and sent (None drops it, so no seq is consumed),
and every model card through redact_card. Private mode is a no-op. In public mode nothing MIMIC-derived leaves the
cluster: retrieval is rank and similarity only (U2), there is no reference and so no score stage, a card's checkpoint
paths, which hold the username, are cut to file names, and an error message the server did not author is replaced
(a model error is exception text, which can carry report text and paths).

PUBLIC_DROP is the explicit policy, keyed by event or "event:stage". Behind it two invariants hold for every event and
stage, known or not: the unambiguous R1 key names are removed wherever they sit, and every retrieval list is reduced
to rank and similarity. Each such removal is logged (event and key only, never the value) because it means the policy
has a gap. Inputs are never mutated.
"""
import copy
import logging
import os
from typing import Any, Dict, List, Optional, Union

log = logging.getLogger("app.redact")

MODES = ("private", "public")

PUBLIC_DROP: Dict[str, Union[str, List[str]]] = {   # dotted paths; "[]" walks a list; "*" drops the whole event
    "message_start": ["options.reference", "options.test_row", "image.urls.original"],
    "stage_end:preprocess": ["detail.test_row", "detail.identical_to"],
    # U2 (user, 2026-10-01): public mode shows similarity scores only.
    "stage_end:retrieve": ["detail.image_neighbors[].image_url", "detail.image_neighbors[].study_id",
                           "detail.image_neighbors[].gallery_row", "detail.image_neighbors[].txt_row",
                           "detail.image_neighbors[].labels", "detail.report_matches[].report",
                           "detail.report_matches[].group", "detail.report_matches[].group_size",
                           "detail.report_matches[].txt_row", "detail.report_matches[].labels",
                           "detail.true_report_rank", "detail.gallery.build_id"],
    "stage_end:label": ["detail.neighbor_agreement"],
    "stage_end:score": "*",   # the whole event: there is no reference in public mode
    "stage_start:score": "*",   # a skipped stage emits only stage_end, and public score is always skipped
}

# R1 key names that are never legitimate in a public payload. report and labels are not here: model output uses them,
# so they are dropped only at their PUBLIC_DROP paths.
CATCH_ALL_KEYS = frozenset({"image_url", "study_id", "subject_id", "gallery_row", "txt_row", "group", "group_size",
                            "neighbor_agreement", "reference", "test_row", "identical_to", "true_report_rank"})

# U2: every element of a list stored under one of these names, at any depth, keeps only these keys.
U2_LISTS = frozenset({"image_neighbors", "report_matches"})
U2_KEYS = ("rank", "similarity")

# A public error event keeps its message only for these kinds, whose text the server authors. Any other message is
# replaced by the fixed one. Warning messages are left alone: they are authored literals.
AUTHORED_ERROR_KINDS = frozenset({"validation_error", "overloaded_error", "server_restart"})
PUBLIC_ERROR_MESSAGE = "The model could not finish this turn."


def _check_mode(mode: str) -> None:
    if mode not in MODES:
        raise ValueError("mode must be one of {}, got {!r}".format(MODES, mode))


def _drop(obj: Any, path: str) -> None:
    """Delete the key at a dotted path, in place; "a[].b" walks every element of the list a.

    A path that is not there (a missing or None step, a step that is not a dict or list, a list element that is not a
    dict) is skipped silently. Tuples are walked like lists.
    """
    head, _, rest = path.partition(".")
    walk = head.endswith("[]")
    key = head[:-2] if walk else head
    if not isinstance(obj, dict) or key not in obj:
        return
    if not rest:
        del obj[key]
    elif not walk:
        _drop(obj[key], rest)
    elif isinstance(obj[key], (list, tuple)):
        for item in obj[key]:
            _drop(item, rest)


def _removed(key: Any, where: str) -> None:
    log.warning("public redaction: removed %r from %r; PUBLIC_DROP has no entry for it", key, where)


def _rank_and_similarity(value: Any, where: str) -> Any:
    """U2: each dict element of a retrieval list (a lone dict likewise) keeps only U2_KEYS, in place; an element that
    is not a dict is dropped. Any other value is left as it is."""
    if isinstance(value, dict):
        for key in [k for k in value if k not in U2_KEYS]:
            del value[key]
            _removed(key, where)
        return value
    if isinstance(value, (list, tuple)):
        kept = []
        for element in value:
            if isinstance(element, dict):
                kept.append(_rank_and_similarity(element, where))
            else:
                log.warning("public redaction: dropped a %s element of a retrieval list in %r",
                            type(element).__name__, where)
        return kept
    return value


def _sweep(obj: Any, where: str) -> None:
    """Enforce the catch-all and U2 at every depth, in place, logging each removal (key and where, never the value)."""
    if isinstance(obj, dict):
        for key in [k for k in obj if k in CATCH_ALL_KEYS]:
            del obj[key]
            _removed(key, where)
        for key in [k for k in obj if k in U2_LISTS]:
            obj[key] = _rank_and_similarity(obj[key], where)
        children = list(obj.values())
    elif isinstance(obj, (list, tuple)):
        children = obj
    else:
        return
    for child in children:
        _sweep(child, where)


def _scrub_error(data: Dict[str, Any]) -> None:
    """Replace the message of an error that is not of an authored kind (a missing or odd kind included), in place."""
    block = data.get("error")
    if isinstance(block, dict) and "message" in block:
        kind = block.get("type")
        if not (isinstance(kind, str) and kind in AUTHORED_ERROR_KINDS):
            block["message"] = PUBLIC_ERROR_MESSAGE


def _shorten_checkpoints(obj: Any) -> None:
    """Cut every checkpoint path, at any depth, to its file name, in place. None stays None."""
    if isinstance(obj, dict):
        if isinstance(obj.get("checkpoint"), (str, os.PathLike)):
            obj["checkpoint"] = os.path.basename(obj["checkpoint"])
        children = list(obj.values())
    elif isinstance(obj, (list, tuple)):
        children = obj
    else:
        return
    for child in children:
        _shorten_checkpoints(child)


def redact_card(card: Dict[str, Any], mode: str) -> Dict[str, Any]:
    """A model card as it may leave the cluster: in public mode every checkpoint path at any depth (it holds the
    username) is cut to its file name, None stays None, every other key is kept. Returns a copy."""
    _check_mode(mode)
    out = copy.deepcopy(card)
    if mode == "public":
        _shorten_checkpoints(out)
    return out


def redact_event(event: str, data: Dict[str, Any], mode: str) -> Optional[Dict[str, Any]]:
    """The data of one event as it may be stored and sent in this mode; None drops the event. Returns a copy."""
    _check_mode(mode)
    if mode == "private":
        return copy.deepcopy(data)
    stage = data.get("stage")
    keys = [event, "{}:{}".format(event, stage)] if isinstance(stage, str) else [event]
    if any(PUBLIC_DROP.get(key) == "*" for key in keys):
        return None
    out = copy.deepcopy(data)
    for key in keys:
        for path in PUBLIC_DROP.get(key, ()):
            _drop(out, path)
    if event == "message_start" and isinstance(out.get("model"), dict):
        out["model"] = redact_card(out["model"], mode)
    if event == "error":
        _scrub_error(out)
    _sweep(out, keys[-1])
    return out
