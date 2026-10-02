"""Public-mode field policy for the chat app (CHAT_UI_PLAN.md P3-C; rule R1, the DUA by construction).

Every streamed event passes through redact_event before it is stored and sent (None drops it, so no seq is consumed),
and every model card through redact_card. Private mode is a no-op. In public mode nothing MIMIC-derived leaves the
cluster: retrieval is rank and similarity only (U2), there is no reference and so no score stage, and a card's
checkpoint path, which holds the username, is cut to its file name.

PUBLIC_DROP is the explicit policy, keyed by event or "event:stage". After it a catch-all removes the unambiguous R1
key names wherever they sit, so an event or stage the policy has no entry for cannot carry them. Each such removal is
logged (event and key only, never the value) because it means the policy has a gap. Inputs are never mutated.
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
}

# R1 key names that are never legitimate in a public payload. report and labels are not here: model output uses them,
# so they are dropped only at their PUBLIC_DROP paths.
CATCH_ALL_KEYS = frozenset({"image_url", "study_id", "subject_id", "gallery_row", "txt_row", "group", "group_size",
                            "neighbor_agreement", "reference", "test_row", "identical_to", "true_report_rank"})


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


def _sweep(obj: Any, where: str) -> None:
    """Remove the catch-all keys at every depth, in place, and log each removal (key and where, never the value)."""
    if isinstance(obj, dict):
        for key in [k for k in obj if k in CATCH_ALL_KEYS]:
            del obj[key]
            log.warning("public redaction: removed %r from %r; PUBLIC_DROP has no entry for it", key, where)
        children = list(obj.values())
    elif isinstance(obj, (list, tuple)):
        children = obj
    else:
        return
    for child in children:
        _sweep(child, where)


def redact_card(card: Dict[str, Any], mode: str) -> Dict[str, Any]:
    """A model card as it may leave the cluster: in public mode the checkpoint path (it holds the username) is cut to
    its file name, None stays None, every other key is kept. Returns a copy."""
    _check_mode(mode)
    out = copy.deepcopy(card)
    if mode == "public" and isinstance(out.get("checkpoint"), (str, os.PathLike)):
        out["checkpoint"] = os.path.basename(out["checkpoint"])
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
    _sweep(out, keys[-1])
    return out
