"""The chat app's HTTP API (CHAT_UI_PLAN.md P3-D): the app factory, routes, auth, scoping and the SSE bridge.

One process, one worker thread: a turn is checked, queued on the Worker (app/pipeline.py) and streamed back as
Server-Sent Events. Every check that can refuse a turn runs before the stream opens and answers with error_body; once
the stream is open, a failure is an error event. A client that disconnects only stops the forwarding: the turn keeps
running and storing its events, and GET /v1/messages/{id}?after=<seq> picks them up (D7).

Every route carries its own OpenAPI summary, description and refusals (P3-E), so /docs is the API reference.

Development server until P7-B adds the CLI: venv/bin/uvicorn --factory app.server:create_app
"""
import asyncio
import hashlib
import hmac
import html
import io
import ipaddress
import json
import logging
import os
import re
from contextlib import asynccontextmanager, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Sequence, Tuple

from fastapi import Depends, FastAPI, File, Form, Header, Query, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import Headers
from starlette.exceptions import HTTPException
from starlette.requests import Request

from app.commands import COMMAND_HELP, parse_command
from app.engine import REPO_ROOT, Engine, build_engine
from app.gallery import Gallery, GalleryMismatch
from app.imaging import (MAX_UPLOAD_BYTES, MIN_SIDE, TOO_LARGE_MSG, UploadError, load_upload, model_input_image,
                         thumbnail_jpeg)
from app.labels import LabelerClient, LabelerUnavailable, RuleLabeler
from app.pipeline import Pipeline, PublishedDumps, TurnJob, Worker, named_labels
from app.redact import redact_card
from app.schemas import DISCLAIMER, Options, error_body
from app.store import Store

log = logging.getLogger("app.server")

MODEL_CHECKPOINTS = {   # resolved against the repo root (ruling 1)
    "hybrid_150m_m3_rrg": "outputs/h100_report_gen_m3_tower13d_s42/checkpoints/last.ckpt",
    "hybrid_150m_v2_rrg": "outputs/h100_report_gen_full_ext_4gpu_tower13d/checkpoints/last.ckpt",
}
STATIC_DIR = Path(__file__).resolve().parent / "static"   # the page; P4 builds it
MAX_REQUEST_BYTES = MAX_UPLOAD_BYTES + 1024 * 1024        # the image and room for the form's other fields
PING_S = 15.0                                             # a keep-alive comment while no event comes
CLIENT_ID = re.compile(r"[\x21-\x7e]{1,128}")             # visible ASCII: it scopes by equality only
ERROR_KINDS = {400: "invalid_request_error", 401: "authentication_error", 403: "permission_error",
               404: "not_found_error", 413: "validation_error", 422: "validation_error", 429: "overloaded_error",
               500: "internal_error", 503: "unavailable_error"}   # 413 as P8-A's test expects; 403 as every public test-split
                                                                  # refusal (fix-1); 503 when the gallery or the labeller is not there
SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
NO_CACHE = {"Cache-Control": "no-cache"}   # the page and its files: revalidate (ETag, Last-Modified) after a redeploy
NO_IMAGE_MSG = "Attach an X-ray first."
NO_CACHE_MSG = "{} has no O(1) decode cache; set cached_decode=false."   # the engine's own wording (P2-D)
NO_TOKEN_MSG = "This server has no token; connect through loopback."
BUSY_MSG = "The server is busy with {} turns; try again shortly."
PUBLIC_TEST_SPLIT_MSG = "Test-split studies are not available in public mode."
NO_GALLERY_MSG = "Test-split studies need the gallery, which this server has not loaded."
NO_RETRIEVAL_MSG = "Retrieval needs the gallery, which this server has not loaded."
NO_LABELER_MSG = "Labels need a CheXbert labeller, which this server does not have."
LABELER_DOWN_MSG = "The CheXbert labeller did not answer; try again shortly."
TEST_IMAGE_MSG = "Could not read the test-split image."
BOTH_IMAGES_MSG = "Send an image or a test row, not both."
MAX_LABEL_CHARS = 20000   # what the labeller service takes in one request (app/labeler.py answers 422 above it)
TINY_GALLERY_REFERENCE = "phase6_mimic_20260101T000000Z.json"   # the reference result that decides the tiny gallery's gate
PLACEHOLDER = ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><title>CXR Report Chat</title></head>"
               "<body><p>{}</p><p>The chat page is not built yet. The API is under <code>/v1/</code>, and its "
               "reference is at <a href=\"/docs\">/docs</a>.</p></body></html>")

# ---- the OpenAPI reference (P3-E): what /docs and /openapi.json say. Metadata only: no route reads any of it. ------
UPLOAD_MB = MAX_UPLOAD_BYTES // (1024 * 1024)
# The published protocol as the options form field's string. FastAPI drops Form(openapi_examples=...) from the spec (a
# form's fields become one body model), so it is the field's own `examples`; P8-E's README walkthrough copies it.
OPTIONS_EXAMPLE = ('{"decode": "beam", "beam_size": 3, "max_new_tokens": 100, "cached_decode": true, "compile": false, '
                   '"k_images": 4, "k_reports": 3, "label": true, "reference": null, "display_repair": false, '
                   '"stop_on_repeat": false}')
# GET /v1/models as /docs shows it (test_app_openapi pins its keys to the real answer): `features` says which stages this server
# runs, so a client can tell a skipped stage from one it switched off.
MODELS_EXAMPLE = {"default_model": "hybrid_150m_m3_rrg", "mode": "private", "allow_compile": False,
                  "features": {"retrieval": False, "labels": False},
                  "models": [{"name": "hybrid_150m_m3_rrg", "prefix_k": 32, "cached_decode_available": True, "eos_trained": False,
                              "device": "cpu"}]}
# GET /healthz as /docs shows it (test_app_openapi pins its keys to the real answer). code_version and started_at (P4-H A2) tell a server
# that runs older code than the checkout: the dev server of 2026-10-09 had, and its turns ended "Internal error (ImportError)".
HEALTHZ_EXAMPLE = {"status": "ok", "mode": "private", "default_model": "hybrid_150m_m3_rrg", "turns_in_flight": 0, "queue_cap": 4,
                   "code_version": "a8e7efe", "started_at": "2026-10-09T14:24:00+00:00"}


def _option_ranges() -> str:
    """"beam_size 1-8, max_new_tokens 16-200, ...", read from Options so the reference cannot drift from it. A field
    with one bound only reads "up to N" or "from N"; one with none is left out."""
    ranges = []
    for name, p in Options.model_json_schema()["properties"].items():
        low, high = p.get("minimum"), p.get("maximum")
        if low is not None and high is not None:
            ranges.append("{} {}-{}".format(name, low, high))
        elif high is not None:
            ranges.append("{} up to {}".format(name, high))
        elif low is not None:
            ranges.append("{} from {}".format(name, low))
    return ", ".join(ranges)


API_DESCRIPTION = (
    DISCLAIMER + " Attach a chest X-ray to a session: the turn streams its stages and the generated report as "
    "Server-Sent Events, and sessions are stored, replayed and exported.\n\n"
    "**Access.** When the server has a token, every `/v1` route needs `Authorization: Bearer <token>` (401 without "
    "it); `/healthz` stays open. A server with no token answers loopback connections only (403 from anywhere else). "
    "In public mode the session and message routes also need an `X-Client-Id` header (400 without it) and show only "
    "that client's data.\n\n"
    "**Errors.** A refusal is JSON, `{\"type\": \"error\", \"error\": {\"type\": kind, \"message\": text}}`, with the "
    "kind of its status: " + ", ".join("{} {}".format(status, kind) for status, kind in ERROR_KINDS.items()) + ".")
CLIENT_ID_DOC = ("Public mode only: names the caller, 1 to 128 visible ASCII characters, and scopes its sessions and "
                 "messages. A private server ignores it.")
XRAY_DOC = "The chest X-ray: PNG, JPEG or WEBP, at most {} MB, each side at least {} px.".format(UPLOAD_MB, MIN_SIDE)
IMAGE_DOC = XRAY_DOC + (" Leave it out to reuse the session's latest image; a session with no image yet answers 422.")
TEXT_DOC = ("A command that changes this turn's options, or a note kept with the turn. " + COMMAND_HELP + " Sent "
            "without an image, a command (or no text) runs the session's latest image again, and any other text is "
            "answered with the command list and no model runs; a session with no image yet answers a text-only turn "
            "with a 422.")
OPTIONS_DOC = ("The turn's options as a JSON object with the keys " + ", ".join(Options.model_fields) + ". A key left "
               "out takes its default, the published protocol; the ranges are " + _option_ranges() + ". An unknown "
               "key or a value out of range is a 422, and anything that is not a JSON object a 400. `stop_on_repeat` ends "
               "decoding at the first sentence that repeats an earlier one word for word (case and spacing aside): the "
               "generate stage then reports `stopped: repeat`, and `truncated_mid_sentence` is false even when the raw "
               "`report` ends in the start of the next sentence; left out, the whole `max_new_tokens` is decoded, as in "
               "the published protocol. A model whose card says `eos_trained: true` was trained to end its reports: its "
               "decoding also stops where it does, whatever `stop_on_repeat` says, and the generate stage then reports "
               "`stopped: eos`. If `stop_on_repeat` ends the search after a report has already ended, that report is the "
               "answer and the stop is `eos` too. `reference` and `test_row` work in private mode only. A command in `text` "
               "is applied on top.")
REFUSALS = {   # what a status means in the reference; a route that can answer it declares it with _refusals()
    400: "The request is malformed: `options` is not a JSON object, or in public mode `X-Client-Id` is missing or "
         "invalid.",
    403: "A test-split study was asked for in public mode.",
    404: "The session or message does not exist or was deleted, or in public mode it belongs to another client.",
    413: "The image is over the {} MB upload limit.".format(UPLOAD_MB),
    422: "The request failed validation: a parameter, an option, the model or the image cannot be used, or there is "
         "no image to run.",
    429: "The server is busy: its queue of accepted turns is full.",
    500: "The server could not store the image, or read a test-split one.",
    503: "What the route needs is not on this server or does not answer: the retrieval gallery or the CheXbert labeller.",
}
STREAM_RESPONSE = {200: {   # the turn is Server-Sent Events, not JSON
    "description": "The turn's events, one frame each: `event: <name>`, then `data: <one-line JSON>` with a `seq` that "
                   "counts 1, 2, ... without gaps. The names are `message_start`, `stage_start`, `stage_end`, "
                   "`content_block_start`, `content_block_delta`, `content_block_stop`, `warning`, `error` and "
                   "`message_stop`. The stream ends with a `message_stop` whose `status` is `done`, `error` or "
                   "`aborted` (a cancelled turn). A failure after this 200 has opened is no HTTP status: it arrives as "
                   "an `error` event followed by a `message_stop` with status `error`. A `: ping` comment goes out "
                   "every {:g} s while no event comes.".format(PING_S),
    "headers": {"X-Message-Id": {"description": "The new message's id, sent before the first event.",
                                 "schema": {"type": "string"}}},
    "content": {"text/event-stream": {"schema": {"type": "string"}}}}}
EXPORT_RESPONSE = {200: {   # JSON is FastAPI's default; Markdown is the other format
    "description": "The session as a file: JSON (the default) or Markdown.",
    "headers": {"Content-Disposition": {"description": "An attachment named session-<id>.json or session-<id>.md.",
                                        "schema": {"type": "string"}}},
    "content": {"text/markdown": {"schema": {"type": "string"}}}}}


def _refusals(messages: Dict[int, str], token: bool = True, client_id: bool = True) -> Dict[Any, Dict[str, Any]]:
    """responses= for a route: each status it refuses with, as the error envelope carrying the message it really sends,
    and a default for what is answered before the route runs. That is 403 for a tokenless server reached off loopback
    (every route), 401 for a missing token (token: every /v1 route) and 400 for a missing client id in public mode
    (client_id: the session and message routes, not /v1/models). A route with a default is not given FastAPI's own
    422, whose HTTPValidationError body this API never sends."""
    out: Dict[Any, Dict[str, Any]] = {
        status: {"description": "{}: {}".format(ERROR_KINDS[status], REFUSALS[status]),
                 "content": {"application/json": {"example": error_body(ERROR_KINDS[status], message)}}}
        for status, message in messages.items()}
    guards = []
    if token:
        guards.append("401 when the server has a token and the request lacks it")
    if client_id:
        guards.append("400 in public mode without a valid `X-Client-Id`")
    guards.append("403 from a tokenless server to a connection that is not loopback")
    example = (error_body(ERROR_KINDS[401], "Missing or wrong token: send Authorization: Bearer <token>.") if token
               else error_body(ERROR_KINDS[403], NO_TOKEN_MSG))
    out["default"] = {"description": "Any other refusal, in the same envelope: " + ", ".join(guards) + ".",
                      "content": {"application/json": {"example": example}}}
    return out


class NewSession(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = ""


class LabelText(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=MAX_LABEL_CHARS,
                      description="The report to label, 1 to {:,} characters.".format(MAX_LABEL_CHARS))


def sse(event: str, data: Dict[str, Any]) -> str:
    return "event: {}\ndata: {}\n\n".format(event, json.dumps(data, separators=(",", ":"), ensure_ascii=False))


async def _frames(queue: "asyncio.Queue[Optional[str]]") -> AsyncIterator[str]:
    """The response body of a turn. A client that disconnects only stops this generator: the worker keeps running
    and storing events, and GET /v1/messages/{id}?after=<seq> picks them up (D7)."""
    while True:
        try:
            frame = await asyncio.wait_for(queue.get(), timeout=PING_S)
        except asyncio.TimeoutError:
            yield ": ping\n\n"
            continue
        if frame is None:          # the worker finished the turn
            return
        yield frame


def _push(loop: asyncio.AbstractEventLoop, queue: "asyncio.Queue[Optional[str]]", item: Optional[str]) -> None:
    """Hand a frame, or the end-of-turn None, from the worker thread to the request's event loop."""
    try:
        loop.call_soon_threadsafe(queue.put_nowait, item)
    except RuntimeError:   # the loop is closed: the event is stored already, and a poll picks it up
        pass


def _error(status: int, message: str, headers: Optional[Dict[str, str]] = None) -> JSONResponse:
    return JSONResponse(error_body(ERROR_KINDS.get(status, "invalid_request_error"), message), status_code=status,
                        headers=headers)


def _capped(receive: Callable) -> Callable:
    """receive(), refusing a body once more than MAX_REQUEST_BYTES of it have arrived (one sent without a length)."""
    seen = 0

    async def capped() -> Dict[str, Any]:
        nonlocal seen
        message = await receive()
        if message["type"] == "http.request":
            seen += len(message.get("body", b""))
            if seen > MAX_REQUEST_BYTES:
                raise HTTPException(413, TOO_LARGE_MSG)
        return message

    return capped


class _Guard:
    """ASGI middleware ahead of routing and of any body parsing (FastAPI parses a form before its dependencies).

    R6: with no token, only a connection that arrived on a loopback address is served, whatever host the process was
    bound to (`uvicorn --factory --host 0.0.0.0` never tells create_app). D23: with a token set, every /v1 path needs
    Authorization: Bearer <token>. Ruling 4: a request whose Content-Length exceeds MAX_REQUEST_BYTES is refused
    unread, and a body sent without a length is cut off as it arrives.
    """

    def __init__(self, app: Callable, token: Optional[str]):
        self.app, self.token = app, (token or "").encode()

    async def __call__(self, scope: Dict[str, Any], receive: Callable, send: Callable) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        path = scope["path"]
        length = headers.get("content-length", "")
        if not self.token and _arrived_off_loopback(scope):
            response = _error(403, NO_TOKEN_MSG)
        elif self.token and (path == "/v1" or path.startswith("/v1/")) and not self._authorized(headers):
            response = _error(401, "Missing or wrong token: send Authorization: Bearer <token>.",
                              {"WWW-Authenticate": "Bearer"})
        elif length.isdigit() and int(length) > MAX_REQUEST_BYTES:
            response = _error(413, TOO_LARGE_MSG)
        else:
            await self.app(scope, _capped(receive), send)
            return
        await response(scope, receive, send)

    def _authorized(self, headers: Headers) -> bool:
        scheme, _, given = headers.get("authorization", "").partition(" ")
        return scheme.lower() == "bearer" and hmac.compare_digest(given.strip().encode(), self.token)


class _StaticFiles(StaticFiles):
    """app/static/ arrives with P4. Until it exists every /static/ path is a 404, where StaticFiles would answer 500.
    Every file goes out with Cache-Control: no-cache, so a browser asks again after an rsync redeploy and the ETag
    answers 304 when nothing changed."""

    async def check_config(self) -> None:
        return None

    async def get_response(self, path: str, scope: Dict[str, Any]) -> Response:
        response = await super().get_response(path, scope)
        response.headers.update(NO_CACHE)
        return response


def _loopback_ip(ip: Any) -> bool:
    """Loopback, an IPv4-mapped one included (::ffff:127.0.0.1 is not loopback to Python 3.11's ipaddress)."""
    return (getattr(ip, "ipv4_mapped", None) or ip).is_loopback


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return _loopback_ip(ipaddress.ip_address(host))
    except ValueError:   # a host name: it can resolve to anything
        return False


def _arrived_off_loopback(scope: Dict[str, Any]) -> bool:
    """The connection came in on a numeric address that is not loopback (uvicorn reports the socket's own address).
    A host name, a unix socket or no address at all is not judged: only a test client sends a name."""
    server = scope.get("server") or (None,)
    try:
        ip = ipaddress.ip_address(server[0])
    except (TypeError, ValueError):
        return False
    return not _loopback_ip(ip)


def _chat_home(home: Optional[str]) -> Path:
    """CHAT_HOME: the argument, else $CHAT_HOME, else ~/chat_sessions. Never inside this repository, nor inside any
    other checkout of this project (a .git beside hybrid_xmamba/), symlinks resolved first: on the cluster, outputs/
    and results/ link into the thesis checkout (DUA). A home directory that is only a git repository (dotfiles) is
    fine."""
    path = Path(home or os.environ.get("CHAT_HOME") or Path.home() / "chat_sessions").expanduser().resolve()
    if path == REPO_ROOT or REPO_ROOT in path.parents:
        raise RuntimeError("CHAT_HOME must be outside the repository, got {}".format(path))
    for directory in (path,) + tuple(path.parents):
        if (directory / ".git").exists() and (directory / "hybrid_xmamba").is_dir():   # .git: a dir or a file
            raise RuntimeError("CHAT_HOME must be outside any checkout of this project, but {} is one; got {}".format(
                directory, path))
    return path


def _engines(kind: str, models: Sequence[str], tiny_step_delay_s: float, drift_note: str) -> Dict[str, Engine]:
    """Ruling 1: tiny is one engine and ignores models; real is one CPU engine per model, the first the default."""
    if kind == "tiny":
        return {"tiny": build_engine("tiny", step_delay_s=tiny_step_delay_s)}
    return {name: build_engine("real", checkpoint=str(REPO_ROOT / MODEL_CHECKPOINTS[name]), model_config=name,
                               device="cpu", drift_note=drift_note) for name in models}


def _readable(errors: Sequence[Any]) -> str:
    """Pydantic errors on one line, "field: problem; ...", without the values that were sent."""
    parts = []
    for e in errors:
        loc = ".".join(str(p) for p in e.get("loc", ()) if p not in ("body", "query", "path", "header"))
        parts.append("{}: {}".format(loc, e.get("msg")) if loc else str(e.get("msg")))
    return "; ".join(parts)


def _turn_options(raw: str, text: str) -> Options:
    """The drawer's options with a text command on top (spec §4): malformed JSON is a 400, invalid options a 422."""
    try:
        drawer = json.loads(raw or "{}")
    except (ValueError, RecursionError):   # deep nesting overflows the decoder (fix-1 M1)
        drawer = None
    if not isinstance(drawer, dict):
        raise HTTPException(400, "options must be a JSON object.")
    try:
        return Options.model_validate(dict(drawer, **(parse_command(text) or {})))
    except ValidationError as exc:
        raise HTTPException(422, "Invalid options: " + _readable(exc.errors(include_url=False, include_input=False)))


def _checked_upload(data: bytes) -> Tuple[Image.Image, str, str]:
    """load_upload before the stream opens (ruling 4); -> (the image, its file extension, the upload's sha256).
    Its messages are written for users."""
    try:
        img, facts = load_upload(data)
    except UploadError as exc:
        raise HTTPException(422, str(exc)) from None
    return img, facts["format"].lower(), hashlib.sha256(data).hexdigest()


def _save_upload(store: Store, session_id: str, sha256: str, data: bytes, img: Image.Image, ext: str) -> None:
    """The bytes as sent, the thumbnail and the 224x224 model input, stored before any record names the image: a
    turn stopped while queued never reaches preprocess, and its image must still exist (fix-1 I1)."""
    png = io.BytesIO()
    model_input_image(img).save(png, "PNG")   # exactly Prepared.model_input
    store.save_upload(session_id, sha256, data, ext, thumbnail_jpeg(img), png.getvalue())


def _test_count(gallery: Gallery) -> int:
    """The test split's size: the gallery's report rows are its train reports, then its test reports."""
    facts = gallery.facts()
    return facts["report_rows"] - facts["images"]


def _test_image_sha256(gallery: Gallery, row: int) -> str:
    """A test study's image, read and checked as an upload is before the stream opens; -> its sha256. It is never copied:
    preprocess reads it again from the dataset."""
    try:
        data = Path(gallery.test_study(row)["image"]).read_bytes()
    except OSError:
        log.warning("a test-split image could not be read")
        raise HTTPException(500, TEST_IMAGE_MSG) from None
    return _checked_upload(data)[2]


def _previous_image(store: Store, session: Dict[str, Any], gallery: Optional[Gallery] = None) -> Optional[Dict[str, Any]]:
    """The newest image of the session that can still run: an upload whose original is on disk, or a test study the gallery
    still has (pass the gallery in private mode only; a public session stores no test row). What a text-only turn reruns."""
    for message in reversed(session["messages"]):
        sha256, row = message["image_sha256"], message.get("test_row")
        if message["role"] != "user" or not sha256:
            continue
        if row is not None:
            if gallery is not None and row < _test_count(gallery):
                return {"sha256": sha256, "filename": message["image_filename"], "test_row": row}
        elif store.upload_path(session["id"], sha256, "original") is not None:
            return {"sha256": sha256, "filename": message["image_filename"]}
    return None


def _wire_gallery(given: Optional[Gallery], gallery_dir: Optional[str], tiny_gallery: bool, home: Path,
                  engines: Dict[str, Engine], default_model: str) -> Tuple[Optional[Gallery], Optional[List[str]]]:
    """The gallery the pipeline queries: the one passed in; else gallery_dir, opened against the default engine's tower hash
    (P5-D), and served only to the engines whose tower is that one; else, with tiny_gallery, a synthetic one; else none. A
    gallery_dir that cannot be served leaves the server running without retrieval, and says why (counts and file names only,
    R7). -> (the gallery, the engines it serves; None for every engine)."""
    if given is not None:
        return given, None
    if gallery_dir:
        towers = {name: e.tower_sha256() for name, e in engines.items()}
        try:
            gallery = Gallery.open(Path(gallery_dir).expanduser(), expect_tower_sha256=towers[default_model])
        except GalleryMismatch as exc:
            print("[server] gallery unavailable: {}".format(exc), flush=True)
            return None, None
        served = [name for name, tower in towers.items() if tower == towers[default_model]]
        facts = gallery.facts()
        print("[server] gallery: {} images, {} report rows, labels {}".format(
            facts["images"], facts["report_rows"], "pending" if gallery.labels is None else "done"), flush=True)
        if len(served) < len(towers):
            print("[server] gallery: no retrieval for {}: another image tower".format(
                ", ".join(name for name in towers if name not in served)), flush=True)
        return gallery, served
    if tiny_gallery:
        return _tiny_gallery(home), None
    return None, None


def _tiny_gallery(home: Path) -> Gallery:
    """A laptop convenience (tiny_gallery=True): the synthetic gallery of scripts/build_retrieval_gallery.py --tiny under
    <home>/gallery/tiny, built once and kept. Its vectors are 16-d, the tiny engine's pooled width, so a tiny turn's query meets
    them. Its R@k gate is decided as the cluster decides it (a reference result equal to the build's own, then compare_rk), and
    it opens without a tower check: its tower hash is a fixed text's, no engine's. Nothing in it is MIMIC-derived."""
    from scripts.build_retrieval_gallery import build_tiny, compare_rk
    root = Path(home) / "gallery" / "tiny"
    if not (root / "manifest.json").is_file():
        build_tiny(root)
    gate = json.loads((root / "manifest.json").read_text()).get("gate_rk")
    if not (isinstance(gate, dict) and gate.get("equal") is True):
        (root / "reference_rk").mkdir(exist_ok=True)
        app_rk = json.loads((root / "gate_rk.json").read_text())["app"]
        (root / "reference_rk" / TINY_GALLERY_REFERENCE).write_text(json.dumps({"metrics": app_rk}))
        with redirect_stdout(io.StringIO()):   # its RESULT line is for a job log
            compare_rk(root)
    return Gallery.open(root, expect_tower_sha256=None)


def _wire_labeler(given: Any, labeler_url: Optional[str], tiny_gallery: bool) -> Any:
    """The labeller: the one passed in (anything with label and healthy), else the service at labeler_url, else, with
    tiny_gallery, the keyword RuleLabeler; else none."""
    if given is not None:
        return given
    if labeler_url:
        return LabelerClient(labeler_url)
    return RuleLabeler() if tiny_gallery else None


def _wire_published(dirs: Optional[Dict[str, str]], gallery: Optional[Gallery]) -> Optional[PublishedDumps]:
    """The published dumps (D18), from {"model": dir, "floor": dir}; None on the laptop. Dumps that cannot be read, or whose
    lines are not one per test study of the gallery, are not used: a line would sit beside the wrong study."""
    if not dirs:
        return None
    try:
        dumps = PublishedDumps.load(Path(dirs["model"]), Path(dirs["floor"]))
    except (OSError, KeyError, UnicodeDecodeError) as exc:
        print("[server] published dumps unavailable ({})".format(type(exc).__name__), flush=True)
        return None
    if gallery is not None and not len(dumps.model_hyps) == len(dumps.floor_hyps) == _test_count(gallery):
        print("[server] published dumps unavailable: {} and {} lines for {} test studies".format(
            len(dumps.model_hyps), len(dumps.floor_hyps), _test_count(gallery)), flush=True)
        return None
    return dumps


def _clean_filename(name: Optional[str]) -> Optional[str]:
    """The upload's own name: no directory a browser put in front, one line, at most 200 characters."""
    return " ".join(os.path.basename((name or "").replace("\\", "/")).split())[:200] or None


def create_app(engine: str = "tiny", mode: str = "private", home: Optional[str] = None, host: str = "127.0.0.1",
               token: Optional[str] = None, queue_cap: int = 4, gallery_dir: Optional[str] = None,
               labeler_url: Optional[str] = None, models: Sequence[str] = ("hybrid_150m_m3_rrg",),
               allow_compile: bool = False, cors_origins: Sequence[str] = (),
               published_dirs: Optional[Dict[str, str]] = None, tiny_step_delay_s: float = 0.0,
               drift_note: str = "", gallery: Optional[Gallery] = None, labeler: Any = None,
               tiny_gallery: bool = False) -> FastAPI:
    """The app. It refuses to exist without a token on a non-loopback address or in public mode (R6, D9).

    The retrieve, label and score stages (P5-E) get their gallery from `gallery` (an opened Gallery), else from gallery_dir
    (opened against the default engine's tower hash), and their labeller from `labeler` (anything with label and healthy),
    else from labeler_url (a LabelerClient). tiny_gallery is a laptop convenience: with neither given, a synthetic gallery is
    built under <home>/gallery/tiny and the keyword RuleLabeler labels, so the tiny server runs every stage. published_dirs
    ({"model": dir, "floor": dir}, each holding a hyps.txt) puts the published lines beside a test study's score. All of them
    are on app.state, with gallery_dir, labeler_url and published_dirs as given.
    """
    if mode not in ("private", "public"):
        raise ValueError("mode must be 'private' or 'public', got {!r}".format(mode))
    if not token and not _is_loopback(host):
        raise RuntimeError("Refusing to serve on {} without a token (R6).".format(host))
    if not token and mode == "public":
        raise RuntimeError("Public mode needs a token (R6).")
    if engine not in ("tiny", "real"):
        raise ValueError("engine must be 'tiny' or 'real', got {!r}".format(engine))
    if engine == "real" and (not models or any(m not in MODEL_CHECKPOINTS for m in models)):
        raise ValueError("models must be among {}, got {}".format(sorted(MODEL_CHECKPOINTS), list(models)))
    if queue_cap < 1:
        raise ValueError("queue_cap must be at least 1, got {}".format(queue_cap))
    store = Store(_chat_home(home))
    print("[server] sqlite journal_mode={}".format(store.journal_mode), flush=True)
    print("[server] recovered {} running turn(s)".format(store.recover_after_restart()), flush=True)
    try:
        engines = _engines(engine, models, tiny_step_delay_s, drift_note)
        default_model = next(iter(engines))
        gallery, retrieval_models = _wire_gallery(gallery, gallery_dir, tiny_gallery, store.home, engines, default_model)
        labeler = _wire_labeler(labeler, labeler_url, tiny_gallery)
        published = _wire_published(published_dirs, gallery)
    except BaseException:
        store.close()
        raise
    worker = Worker(Pipeline(engines, default_model, store, mode, gallery=gallery, labeler=labeler, published=published,
                             drift_note=drift_note, retrieval_models=retrieval_models), queue_cap)
    static_dir = STATIC_DIR
    # Read once, here: the code this process runs is what it imported at the start, whatever the checkout says later (P4-H A2). The card's
    # provenance: the checkout's HEAD, else the .sync_stamp; "-dirty" when that tree had changes its commit does not hold. /healthz needs
    # no token, so a public server names no commit to whoever asks.
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    provenance = engines[default_model].card()
    git_sha = provenance.get("git_sha")
    code_version = None
    if mode == "private" and isinstance(git_sha, str) and git_sha:
        code_version = git_sha[:7] + ("-dirty" if provenance.get("git_dirty") is True else "")

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        # Queued and running turns stop at their next step and end as a server restart. The wait runs off the event
        # loop. uvicorn reaches this only once open connections close: the serving wrapper sets
        # timeout_graceful_shutdown (P7-B).
        await run_in_threadpool(worker.shutdown)
        store.close()

    app = FastAPI(title="CXR report chat", description=API_DESCRIPTION, lifespan=lifespan)
    app.state.store, app.state.engines, app.state.worker = store, engines, worker
    app.state.mode, app.state.default_model = mode, default_model
    app.state.gallery_dir, app.state.labeler_url, app.state.published_dirs = gallery_dir, labeler_url, published_dirs
    app.state.gallery, app.state.labeler, app.state.published = gallery, labeler, published
    app.add_middleware(_Guard, token=token)
    if cors_origins:   # added last, so it runs first: a preflight is answered before the token check
        app.add_middleware(CORSMiddleware, allow_origins=list(cors_origins), allow_methods=["GET", "POST", "DELETE"],
                           allow_headers=["Authorization", "Content-Type", "X-Client-Id"],
                           expose_headers=["X-Message-Id"])

    @app.exception_handler(HTTPException)
    async def http_error(_: Request, exc: HTTPException) -> JSONResponse:
        return _error(exc.status_code, str(exc.detail), exc.headers)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_: Request, exc: RequestValidationError) -> JSONResponse:
        return _error(422, "Invalid request: " + _readable(exc.errors()))

    def client_scope(x_client_id: Optional[str] = Header(None, description=CLIENT_ID_DOC)) -> Optional[str]:
        """D8: the X-Client-Id that scopes sessions and messages in public mode; None (every session) in private."""
        if mode == "private":
            return None
        if not x_client_id or CLIENT_ID.fullmatch(x_client_id) is None:
            raise HTTPException(400, "Public mode needs an X-Client-Id header (1 to 128 visible characters).")
        return x_client_id

    def visible_session(session_id: str, client_id: Optional[str]) -> Dict[str, Any]:
        session = store.get_session(session_id, client_id)
        if session is None:   # unknown and not visible look the same
            raise HTTPException(404, "Session not found.")
        return session

    def visible_message(message_id: str, client_id: Optional[str]) -> Dict[str, Any]:
        message = store.get_message(message_id, client_id)
        if message is None:
            raise HTTPException(404, "Message not found.")
        return message

    @app.get("/", include_in_schema=False)
    def index() -> Response:
        page = static_dir / "index.html"
        if page.is_file():
            return FileResponse(page, headers=NO_CACHE)
        return HTMLResponse(PLACEHOLDER.format(html.escape(DISCLAIMER)), headers=NO_CACHE)

    app.mount("/static", _StaticFiles(directory=str(static_dir), check_dir=False), name="static")

    @app.get("/healthz", summary="Check the server",
             responses={200: {"description": "The server is up.", "content": {"application/json": {"example": HEALTHZ_EXAMPLE}}},
                        **_refusals({}, token=False, client_id=False)},
             description="The server's mode, default model, turns in flight and queue cap, with `code_version` (the short git sha "
                         "of the code it started from, which it runs until it restarts, with `-dirty` when that tree had changes; "
                         "null when it cannot tell, and in public mode) and `started_at` (ISO 8601, UTC). It needs no token.")
    def healthz():
        return {"status": "ok", "mode": mode, "default_model": default_model, "turns_in_flight": worker.in_flight,
                "queue_cap": queue_cap, "code_version": code_version, "started_at": started_at}

    @app.get("/v1/models", summary="List models",
             responses={200: {"description": "The models, the default, and the stages this server runs.",
                              "content": {"application/json": {"example": MODELS_EXAMPLE}}},
                        **_refusals({}, client_id=False)},
             description="The models this server can run, each with its card (checkpoint, device, architecture "
                         "settings, whether decoding can be cached, and `eos_trained`: whether it was trained to end its "
                         "reports, in which case decoding stops where it does and the generate stage reports "
                         "`stopped: eos`), and the default model, the mode, whether "
                         "`compile` is allowed and, in `features`, whether it runs the retrieval and labels stages. "
                         "In public mode a card names its checkpoint by file name only.")
    def list_models():
        pipeline = worker.pipeline   # a gallery and a labeller are what P5-E gives it: until then it has neither
        return {"default_model": default_model, "mode": mode, "allow_compile": allow_compile,
                "features": {"retrieval": getattr(pipeline, "gallery", None) is not None,
                             "labels": getattr(pipeline, "labeler", None) is not None},
                "models": [redact_card(e.card(), mode) for e in engines.values()]}

    @app.post("/v1/sessions", summary="Create a session",
              description="Starts an empty session in the server's mode; `title` is optional, and an untitled session "
                          "is named by its first turn. Answers 422 for an unknown body field.",
              responses=_refusals({422: "Invalid request: nope: Extra inputs are not permitted"}))
    def create_session(body: Optional[NewSession] = None, client_id: Optional[str] = Depends(client_scope)):
        return store.create_session(mode, client_id, body.title if body else "")

    @app.get("/v1/sessions", summary="List sessions",
             description="The sessions you can see, newest first, `limit` at a time (1 to 200, default 50): pass a "
                         "page's `next_cursor` as `cursor` to get the next page, and a null `next_cursor` ends the "
                         "list. Answers 422 for a `limit` outside 1 to 200.",
             responses=_refusals({422: "Invalid request: limit: Input should be greater than or equal to 1"}))
    def list_sessions(limit: int = Query(50, ge=1, le=200), cursor: Optional[str] = None,
                      client_id: Optional[str] = Depends(client_scope)):
        sessions, next_cursor = store.list_sessions(client_id, limit, cursor)
        return {"sessions": sessions, "next_cursor": next_cursor}

    @app.get("/v1/sessions/{session_id}", summary="Get a session",
             description="One session with its messages in order, user and assistant turns alike; a message's events "
                         "come from `GET /v1/messages/{message_id}`. Answers 404 for an unknown or deleted session, "
                         "or in public mode another client's.",
             responses=_refusals({404: "Session not found."}))
    def get_session(session_id: str, client_id: Optional[str] = Depends(client_scope)):
        return visible_session(session_id, client_id)

    @app.delete("/v1/sessions/{session_id}", status_code=204, summary="Delete a session",
                description="Hides the session from every route and removes its uploaded images at once; its stored "
                            "rows stay until the retention sweep purges them. Answers 204, or 404 for an unknown, "
                            "already deleted or (public mode) another client's session.",
                responses=_refusals({404: "Session not found."}))
    def delete_session(session_id: str, client_id: Optional[str] = Depends(client_scope)) -> Response:
        if not store.delete_session(session_id, client_id):
            raise HTTPException(404, "Session not found.")
        return Response(status_code=204)

    @app.post("/v1/sessions/{session_id}/messages", summary="Run a turn (streamed)", response_class=StreamingResponse,
              description="Runs one turn on the attached X-ray, or on the session's latest image when none is "
                          "attached, and streams it as Server-Sent Events (`text/event-stream`), from "
                          "`message_start` to `message_stop`. The `X-Message-Id` response header carries the new "
                          "message's id before the first event, so a client can cancel the turn or poll `GET "
                          "/v1/messages/{message_id}` at once; every refusal (400, 403, 404, 413, 422, 429, 500) "
                          "comes before the stream opens, as a JSON error envelope.",
              responses={**STREAM_RESPONSE, **_refusals({
                  400: "options must be a JSON object.", 403: PUBLIC_TEST_SPLIT_MSG,
                  404: "Session not found.", 413: TOO_LARGE_MSG,
                  422: "Invalid options: beam_size: Input should be less than or equal to 8",
                  429: BUSY_MSG.format(queue_cap),
                  500: "Could not store the image."})})
    async def post_message(session_id: str, image: Optional[UploadFile] = File(None, description=IMAGE_DOC),
                           text: str = Form("", description=TEXT_DOC),
                           options: str = Form("{}", description=OPTIONS_DOC, examples=[OPTIONS_EXAMPLE]),
                           client_id: Optional[str] = Depends(client_scope)) -> Response:
        """One turn, streamed as Server-Sent Events. Every refusal comes before the stream opens (ruling 3)."""
        session = await run_in_threadpool(visible_session, session_id, client_id)
        opts = _turn_options(options, text)
        model = opts.model or default_model
        if model not in engines:
            raise HTTPException(422, "Unknown model; this server has {}.".format(", ".join(engines)))
        if opts.cached_decode and not engines[model].card().get("cached_decode_available", False):
            raise HTTPException(422, NO_CACHE_MSG.format(engines[model].name))
        if opts.compile and not allow_compile:
            raise HTTPException(422, "compile is not enabled on this server.")
        private = "public" not in (mode, session["mode"])
        gallery = worker.pipeline.gallery
        test_sha256 = None
        if opts.test_row is not None:   # R1: test-split studies are private-mode data
            if not private:   # 403, as every public test-split access (fix-1 (b))
                raise HTTPException(403, PUBLIC_TEST_SPLIT_MSG)
            if gallery is None:
                raise HTTPException(422, NO_GALLERY_MSG)
            if opts.test_row >= _test_count(gallery):
                raise HTTPException(422, "test_row must be below {}.".format(_test_count(gallery)))
            if image is not None:
                raise HTTPException(422, BOTH_IMAGES_MSG)
            test_sha256 = await run_in_threadpool(_test_image_sha256, gallery, opts.test_row)
        upload = sha256 = filename = checked = None
        if image is not None:
            upload = await image.read(MAX_UPLOAD_BYTES + 1)
            if len(upload) > MAX_UPLOAD_BYTES:
                raise HTTPException(413, TOO_LARGE_MSG)
            checked = await run_in_threadpool(_checked_upload, upload)
            sha256, filename = checked[2], _clean_filename(image.filename)
        elif test_sha256 is not None:   # the picker: that study's image, read from the dataset by preprocess
            sha256 = test_sha256
        else:   # a text-only turn reruns the session's newest image still on disk, or the test study it last ran
            previous = await run_in_threadpool(_previous_image, store, session, gallery if private else None)
            if previous is None:
                raise HTTPException(422, NO_IMAGE_MSG)
            question = bool(text.strip()) and parse_command(text) is None
            if not question:   # a question gets the fixed answer: no model runs, so no image is used or recorded
                sha256, filename = previous["sha256"], previous["filename"]
                if previous.get("test_row") is not None:   # the turn runs that study again, as its resolved options say
                    opts = opts.model_copy(update={"test_row": previous["test_row"]})
        if not worker.reserve():
            raise HTTPException(429, BUSY_MSG.format(queue_cap))
        try:
            if checked is not None:   # stored before start_turn records it (fix-1 I1)
                await run_in_threadpool(_save_upload, store, session_id, sha256, upload, checked[0], checked[1])
            user_message_id, message_id = await run_in_threadpool(
                store.start_turn, session_id, text, session["mode"], dict(opts.model_dump(), model=model),
                sha256, filename, opts.test_row)
        except KeyError:   # deleted since it was looked up
            worker.release()
            raise HTTPException(404, "Session not found.") from None
        except OSError as exc:   # a full or failing disk: nothing names the image yet
            worker.release()
            log.warning("could not store an upload in session %s: %s", session_id, type(exc).__name__)
            raise HTTPException(500, "Could not store the image.") from None
        except BaseException:
            worker.release()
            raise
        job = TurnJob(session_id, user_message_id, message_id, text, upload, filename,
                      opts.model_copy(update={"model": model}), mode=session["mode"],
                      previous_sha256=sha256 if upload is None else None)
        loop = asyncio.get_running_loop()
        queue: "asyncio.Queue[Optional[str]]" = asyncio.Queue()
        worker.submit(job, lambda event, data: _push(loop, queue, sse(event, data)), lambda: _push(loop, queue, None))
        # The id comes first in a header, so a client can cancel or poll a turn still waiting in the queue.
        return StreamingResponse(_frames(queue), media_type="text/event-stream",
                                 headers=dict(SSE_HEADERS, **{"X-Message-Id": message_id}))

    @app.get("/v1/messages/{message_id}", summary="Get a message and its events",
             description="The stored message (its status is running, done, error or aborted, and its report is set "
                         "once it is done) with its events whose `seq` is above `after`, so a client whose stream "
                         "dropped replays the turn from the last `seq` it saw. Answers 404 for an unknown or hidden "
                         "message and 422 for a negative or non-integer `after`.",
             responses=_refusals({404: "Message not found.",
                                  422: "Invalid request: after: Input should be greater than or equal to 0"}))
    def get_message(message_id: str, after: int = Query(0, ge=0), client_id: Optional[str] = Depends(client_scope)):
        return dict(visible_message(message_id, client_id), events=store.events_after(message_id, after))

    @app.post("/v1/messages/{message_id}/cancel", summary="Cancel a turn",
              description="Stops a queued or running turn at its next step, so it ends `aborted`; the call is "
                          "idempotent, and a turn that already finished answers its final `status` with "
                          "`cancel_requested` false. Answers 404 for an unknown or hidden message.",
              responses=_refusals({404: "Message not found."}))
    def cancel_message(message_id: str, client_id: Optional[str] = Depends(client_scope)):
        """Idempotent (ruling 10): a finished turn answers its final status with cancel_requested false."""
        message = visible_message(message_id, client_id)
        requested = worker.cancel(message_id)
        status = (store.get_message(message_id, client_id) or message)["status"]   # it may have ended meanwhile
        return {"id": message_id, "status": status, "cancel_requested": requested}

    @app.get("/v1/sessions/{session_id}/export", summary="Export a session",
             description="The whole session with every message and its events, as a download: `format=json` (the "
                         "default) is the stored event log, `format=md` a readable record. Answers 404 for an "
                         "unknown or hidden session and 422 for any other `format`.",
             responses={**EXPORT_RESPONSE, **_refusals({404: "Session not found.", 422: "format must be json or md."})})
    def export_session(session_id: str, fmt: str = Query("json", alias="format"),
                       client_id: Optional[str] = Depends(client_scope)) -> Response:
        try:
            filename, media_type, body = store.export(session_id, fmt, client_id)
        except KeyError:
            raise HTTPException(404, "Session not found.") from None
        except ValueError:
            raise HTTPException(422, "format must be json or md.") from None
        return Response(body, media_type=media_type,
                        headers={"Content-Disposition": 'attachment; filename="{}"'.format(filename)})

    # ---- P5-E: retrieval and labels on their own, and the test-split picker's list ------------------------------------

    def private_only() -> None:
        """Test-split studies are private-mode data (R1): 403 in public mode, before anything else about the request."""
        if mode == "public":
            raise HTTPException(403, PUBLIC_TEST_SPLIT_MSG)

    @app.post("/v1/retrieve", summary="Find similar X-rays and reports",
              description="Runs the X-ray through preprocess and encode on the turn worker and returns what a turn's retrieve stage "
                          "shows: the `k_images` most similar training X-rays and the `k_reports` best-matching report groups, as rank "
                          "and similarity only in public mode. Answers 413 for an image over the limit, 422 for an image that cannot "
                          "be used or a k out of range, 429 when the queue is full and 503 when the server has no retrieval gallery.",
              responses=_refusals({413: TOO_LARGE_MSG,
                                   422: "Invalid request: k_images: Input should be less than or equal to 12",
                                   429: BUSY_MSG.format(queue_cap), 503: NO_RETRIEVAL_MSG}, client_id=False))
    async def retrieve(image: UploadFile = File(..., description=XRAY_DOC),
                       k_images: int = Form(4, ge=0, le=12, description="How many similar training X-rays, 0 to 12."),
                       k_reports: int = Form(3, ge=0, le=10, description="How many matching report groups, 0 to 10.")):
        pipeline = worker.pipeline
        if not pipeline.retrieval_ready(default_model):
            raise HTTPException(503, NO_RETRIEVAL_MSG)
        data = await image.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, TOO_LARGE_MSG)
        await run_in_threadpool(_checked_upload, data)
        if not worker.reserve():
            raise HTTPException(429, BUSY_MSG.format(queue_cap))
        future = worker.run_task(lambda: pipeline.retrieve_upload(data, k_images, k_reports))
        return await asyncio.wrap_future(future)

    @app.post("/v1/label", summary="Label a report",
              description="The 14 CheXbert labels of the text, each 0 or 1, as `chexbert_14`. Answers 422 for a missing, empty or "
                          "too long text (at most {:,} characters) and 503 when the server has no labeller or it does not "
                          "answer.".format(MAX_LABEL_CHARS),
              responses=_refusals({422: "Invalid request: text: String should have at least 1 character",
                                   503: NO_LABELER_MSG}, client_id=False))
    def label_text(body: LabelText):
        labeler = worker.pipeline.labeler
        if labeler is None:
            raise HTTPException(503, NO_LABELER_MSG)
        try:
            row = labeler.label([body.text])[0]
        except LabelerUnavailable:
            raise HTTPException(503, LABELER_DOWN_MSG) from None
        return {"chexbert_14": named_labels(row)}

    @app.get("/v1/test-studies", summary="List test-split studies", dependencies=[Depends(private_only)],
             description="The studies of the official test split whose study id starts with `q`, in test-row order, `limit` at a "
                         "time (1 to 200, default 50), each with the `test_row` a turn's options take. Private mode only: answers "
                         "403 in public mode, 422 for a `limit` outside 1 to 200 and 503 when the server has no retrieval gallery.",
             responses=_refusals({403: PUBLIC_TEST_SPLIT_MSG,
                                  422: "Invalid request: limit: Input should be greater than or equal to 1",
                                  503: NO_GALLERY_MSG}, client_id=False))
    def test_studies(q: str = Query("", max_length=64, description="The start of a study id; empty lists every study."),
                     limit: int = Query(50, ge=1, le=200)):
        gallery = worker.pipeline.gallery
        if gallery is None:
            raise HTTPException(503, NO_GALLERY_MSG)
        return {"studies": gallery.list_test_studies(q, limit)}

    return app
