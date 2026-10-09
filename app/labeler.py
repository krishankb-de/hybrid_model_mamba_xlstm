"""CheXbert-14 microservice (CHAT_UI_PLAN.md P5-A). Runs in .venv_chexbert, which pins
transformers<5 and scikit-learn<1.8 for f1chexbert; like score_chexbert_standalone.py it imports
nothing from hybrid_xmamba. Nor does it import app.labels: the names it reports are F1CheXbert's
own, so that the client's CHEXBERT_14 is checked against something independent.

    .venv_chexbert/bin/uvicorn app.labeler:app --host 127.0.0.1 --port 8001

No reply carries report text or an exception's message (R7): a refused request says where and why
as codes, and a model that will not load gets one fixed sentence.
"""
import logging
import threading
from typing import Annotated, List

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

MAX_TEXTS = 64        # per request: bounds what one request can ask of the CPU
MAX_CHARS = 20000     # per text; the same bound as Options.reference

app = FastAPI(title="CheXbert labeller")
log = logging.getLogger("uvicorn.error")      # uvicorn's own error log, in its own format
_labeler = None
_load_lock = threading.Lock()
_UNAVAILABLE = {"status": "unavailable", "message": "CheXbert failed to load"}


def _get():
    global _labeler
    if _labeler is None:
        with _load_lock:    # /healthz can be polled while the model loads: one load, not one more per poll
            if _labeler is None:
                from f1chexbert import F1CheXbert
                _labeler = F1CheXbert()
    return _labeler


def _ready():
    """The model, or None when it cannot be loaded. The cause goes to the log, not to the caller (R7: loading takes no report text,
    so the log can hold it); the next call tries again."""
    try:
        return _get()
    except Exception as exc:
        log.error("CheXbert failed to load: %s: %s", type(exc).__name__, exc)
        return None


class LabelRequest(BaseModel):
    texts: List[Annotated[str, Field(max_length=MAX_CHARS)]] = Field(max_length=MAX_TEXTS)


@app.exception_handler(RequestValidationError)
async def invalid_request(_: Request, exc: RequestValidationError) -> JSONResponse:
    # FastAPI's own 422 body repeats the offending input, which here is report text: say where and why, in codes.
    return JSONResponse({"detail": [{"loc": list(e["loc"]), "type": e["type"]} for e in exc.errors()]}, status_code=422)


@app.get("/healthz")
def healthz():
    if _ready() is None:
        return JSONResponse(_UNAVAILABLE, status_code=503)
    return {"status": "ok"}


@app.post("/label")
def label(req: LabelRequest):
    lab = _ready()
    if lab is None:
        return JSONResponse(_UNAVAILABLE, status_code=503)
    return {"label_names": [str(x) for x in lab.target_names],
            # whitespace-normalised, exactly like the hyps.txt/refs.txt lines the published labels came from
            "labels": [[int(v) for v in lab.get_label(" ".join(t.split()))] for t in req.texts]}
