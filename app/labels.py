"""CheXbert-14 labels for the chat app (CHAT_UI_PLAN.md P5-A).

CHEXBERT_14 is the label order F1CheXbert.target_names reports; P5-F checks it against the live
service rather than trusting this list. Standard library only. An error message raised here holds
counts, an HTTP status or an exception class name, never report text or label names (R7).
"""
import http.client
import json
import urllib.error
import urllib.request
from typing import Any, Dict, List, Sequence

CHEXBERT_14 = ["Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema",
               "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
               "Pleural Other", "Fracture", "Support Devices", "No Finding"]


class LabelerUnavailable(RuntimeError):
    """The labeller cannot be reached, or answered something the caller cannot trust as labels."""


def _why(exc: BaseException) -> str:
    """What went wrong with a request, without the exception's message (a peer's own bytes can end up in it): an HTTP status or a
    class name."""
    if isinstance(exc, urllib.error.HTTPError):
        return "HTTP {}".format(exc.code)
    if isinstance(exc, urllib.error.URLError) and isinstance(exc.reason, BaseException):
        exc = exc.reason
    return type(exc).__name__


def _count(value: Any) -> str:
    return str(len(value)) if isinstance(value, list) else "no"


def _trusted_rows(body: Any, n_texts: int) -> List[List[int]]:
    """The reply's label rows, or LabelerUnavailable. The reply must name the 14 labels in CHEXBERT_14 order and hold one row of
    14 ints in {0, 1} per text; a service that has drifted from either would otherwise hand wrong labels on without a word."""
    if not isinstance(body, dict):
        raise LabelerUnavailable("labeller reply is not a JSON object")
    names = body.get("label_names")
    if names != CHEXBERT_14:
        raise LabelerUnavailable("label order mismatch: the service reported {} label names, expected {}".format(
            _count(names), len(CHEXBERT_14)))
    rows = body.get("labels")
    if not isinstance(rows, list) or len(rows) != n_texts:
        raise LabelerUnavailable("the service returned {} label rows for {} texts".format(_count(rows), n_texts))
    for i, row in enumerate(rows):
        if not isinstance(row, list) or len(row) != len(CHEXBERT_14):
            raise LabelerUnavailable("label row {} has {} entries, expected {}".format(i, _count(row), len(CHEXBERT_14)))
        if any(type(v) is not int or v not in (0, 1) for v in row):      # ints only: not bool, float or text
            raise LabelerUnavailable("label row {} holds a value that is not 0 or 1".format(i))
    return rows


class LabelerClient:
    def __init__(self, url: str, timeout: float = 10.0):
        self.url, self.timeout = url.rstrip("/"), timeout

    def label(self, texts: List[str]) -> List[List[int]]:
        """One row of 14 labels (0 or 1, CHEXBERT_14 order) per text; LabelerUnavailable for anything not trustworthy as that."""
        if not texts:
            return []
        data = json.dumps({"texts": texts}).encode()
        try:
            req = urllib.request.Request(self.url + "/label", data=data,
                                         headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read())
        except (urllib.error.URLError, OSError, ValueError, http.client.HTTPException) as exc:
            raise LabelerUnavailable("labeller request failed: {}".format(_why(exc))) from None
        return _trusted_rows(body, len(texts))

    def healthy(self) -> bool:
        try:
            with urllib.request.urlopen(self.url + "/healthz", timeout=2.0) as resp:
                return resp.status == 200
        except (urllib.error.URLError, OSError, ValueError, http.client.HTTPException):
            return False


class RuleLabeler:
    """Laptop stand-in: keyword rules, so every label-dependent code path runs without CheXbert."""
    # substring matches on purpose, and crude ("line" also fires on "baseline"): a stand-in, not a labeller
    RULES = {"Cardiomegaly": ("cardiomegaly", "enlarged"), "Edema": ("edema",),
             "Pleural Effusion": ("effusion",), "Pneumothorax": ("pneumothorax",),
             "Atelectasis": ("atelectasis",), "Consolidation": ("consolidation",),
             "Pneumonia": ("pneumonia",), "Lung Opacity": ("opacity", "opacities"),
             "Fracture": ("fracture",), "Support Devices": ("tube", "line", "catheter", "pacemaker", "wires"),
             "Lung Lesion": ("nodule", "lesion")}

    def label(self, texts: List[str]) -> List[List[int]]:
        out = []
        for t in texts:
            low = t.lower()
            row = [int(any(k in low for k in self.RULES.get(n, ()))) for n in CHEXBERT_14]
            row[CHEXBERT_14.index("No Finding")] = int(not any(row))
            out.append(row)
        return out

    def healthy(self) -> bool:
        return True


def label_agreement(generated: Sequence[int], neighbor: Sequence[int]) -> Dict[str, Any]:
    """The user's "n/14 labels agree", plus the positives behind any disagreement (D14)."""
    if len(generated) != len(CHEXBERT_14) or len(neighbor) != len(CHEXBERT_14):
        raise ValueError("label_agreement needs {} labels on each side, got {} and {}".format(
            len(CHEXBERT_14), len(generated), len(neighbor)))
    pairs = list(zip(CHEXBERT_14, generated, neighbor))
    return {"agree": sum(int(a == b) for _, a, b in pairs), "of": len(pairs),
            "both_positive": [n for n, a, b in pairs if a and b],
            "neighbor_only": [n for n, a, b in pairs if b and not a],
            "generated_only": [n for n, a, b in pairs if a and not b]}
