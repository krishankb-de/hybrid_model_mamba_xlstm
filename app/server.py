"""The chat app's HTTP API (CHAT_UI_PLAN.md P3-D): the app factory, routes, auth, scoping and the SSE bridge.

One process, one worker thread: a turn is checked, queued on the Worker (app/pipeline.py) and streamed back as
Server-Sent Events. Every check that can refuse a turn runs before the stream opens and answers with error_body; once
the stream is open, a failure is an error event. A client that disconnects only stops the forwarding: the turn keeps
running and storing its events, and GET /v1/messages/{id}?after=<seq> picks them up (D7).

Development server until P7-B adds the CLI: venv/bin/uvicorn --factory app.server:create_app
"""
import asyncio
import hashlib
import hmac
import html
import ipaddress
import json
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, Optional, Sequence

from fastapi import Depends, FastAPI, File, Form, Header, Query, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import Headers
from starlette.exceptions import HTTPException
from starlette.requests import Request

from app.commands import parse_command
from app.engine import REPO_ROOT, Engine, build_engine
from app.imaging import MAX_UPLOAD_BYTES, TOO_LARGE_MSG, UploadError, load_upload
from app.pipeline import Pipeline, TurnJob, Worker
from app.redact import redact_card
from app.schemas import DISCLAIMER, Options, error_body
from app.store import Store

MODEL_CHECKPOINTS = {   # resolved against the repo root (ruling 1)
    "hybrid_150m_m3_rrg": "outputs/h100_report_gen_m3_tower13d_s42/checkpoints/last.ckpt",
    "hybrid_150m_v2_rrg": "outputs/h100_report_gen_full_ext_4gpu_tower13d/checkpoints/last.ckpt",
}
STATIC_DIR = Path(__file__).resolve().parent / "static"   # the page; P4 builds it
MAX_REQUEST_BYTES = MAX_UPLOAD_BYTES + 1024 * 1024        # the image and room for the form's other fields
PING_S = 15.0                                             # a keep-alive comment while no event comes
CLIENT_ID = re.compile(r"[\x21-\x7e]{1,128}")             # visible ASCII: it scopes by equality only
ERROR_KINDS = {400: "invalid_request_error", 401: "authentication_error", 404: "not_found_error",
               413: "request_too_large", 422: "validation_error", 429: "overloaded_error"}
SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
NO_IMAGE_MSG = "Attach an X-ray first."
NO_CACHE_MSG = "{} has no O(1) decode cache; set cached_decode=false."   # the engine's own wording (P2-D)
PLACEHOLDER = ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><title>CXR Report Chat</title></head>"
               "<body><p>{}</p><p>The chat page is not built yet. The API is under <code>/v1/</code>, and its "
               "reference is at <a href=\"/docs\">/docs</a>.</p></body></html>")


class NewSession(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = ""


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

    D23: with a token set, every /v1 path needs Authorization: Bearer <token>. Ruling 4: a request whose Content-Length
    exceeds MAX_REQUEST_BYTES is refused unread, and a body sent without a length is cut off as it arrives.
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
        if self.token and (path == "/v1" or path.startswith("/v1/")) and not self._authorized(headers):
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
    """app/static/ arrives with P4. Until it exists every /static/ path is a 404, where StaticFiles would answer 500."""

    async def check_config(self) -> None:
        return None


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:   # a host name: it can resolve to anything
        return False


def _chat_home(home: Optional[str]) -> Path:
    """CHAT_HOME: the argument, else $CHAT_HOME, else ~/chat_sessions. Never inside the repository (DUA)."""
    path = Path(home or os.environ.get("CHAT_HOME") or Path.home() / "chat_sessions").expanduser().resolve()
    if path == REPO_ROOT or REPO_ROOT in path.parents:
        raise RuntimeError("CHAT_HOME must be outside the repository, got {}".format(path))
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
    except ValueError:
        drawer = None
    if not isinstance(drawer, dict):
        raise HTTPException(400, "options must be a JSON object.")
    try:
        return Options.model_validate(dict(drawer, **(parse_command(text) or {})))
    except ValidationError as exc:
        raise HTTPException(422, "Invalid options: " + _readable(exc.errors(include_url=False, include_input=False)))


def _checked_upload(data: bytes) -> str:
    """load_upload before the stream opens (ruling 4); -> the upload's sha256. Its messages are written for users."""
    try:
        load_upload(data)
    except UploadError as exc:
        raise HTTPException(422, str(exc)) from None
    return hashlib.sha256(data).hexdigest()


def _previous_image(store: Store, session_id: str) -> Optional[Dict[str, Any]]:
    """The session's last image while its upload is on disk: what a text-only turn reruns."""
    last = store.last_image(session_id)
    if last is None or not last["sha256"] or store.upload_path(session_id, last["sha256"], "original") is None:
        return None
    return last


def _clean_filename(name: Optional[str]) -> Optional[str]:
    """The upload's own name: no directory a browser put in front, one line, at most 200 characters."""
    return " ".join(os.path.basename((name or "").replace("\\", "/")).split())[:200] or None


def create_app(engine: str = "tiny", mode: str = "private", home: Optional[str] = None, host: str = "127.0.0.1",
               token: Optional[str] = None, queue_cap: int = 4, gallery_dir: Optional[str] = None,
               labeler_url: Optional[str] = None, models: Sequence[str] = ("hybrid_150m_m3_rrg",),
               allow_compile: bool = False, cors_origins: Sequence[str] = (),
               published_dirs: Optional[Dict[str, str]] = None, tiny_step_delay_s: float = 0.0,
               drift_note: str = "") -> FastAPI:
    """The app. It refuses to exist without a token on a non-loopback address or in public mode (R6, D9).

    gallery_dir, labeler_url and published_dirs are kept on app.state for the stages P5-E adds.
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
    except BaseException:
        store.close()
        raise
    default_model = next(iter(engines))
    worker = Worker(Pipeline(engines, default_model, store, mode, drift_note=drift_note), queue_cap)
    static_dir = STATIC_DIR

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        worker.shutdown()   # running and queued turns stop at their next step and store their ending
        store.close()

    app = FastAPI(title="CXR report chat", lifespan=lifespan)
    app.state.store, app.state.engines, app.state.worker = store, engines, worker
    app.state.mode, app.state.default_model = mode, default_model
    app.state.gallery_dir, app.state.labeler_url, app.state.published_dirs = gallery_dir, labeler_url, published_dirs
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

    def client_scope(x_client_id: Optional[str] = Header(None)) -> Optional[str]:
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
            return FileResponse(page)
        return HTMLResponse(PLACEHOLDER.format(html.escape(DISCLAIMER)))

    app.mount("/static", _StaticFiles(directory=str(static_dir), check_dir=False), name="static")

    @app.get("/healthz")
    def healthz():
        return {"status": "ok", "mode": mode, "default_model": default_model, "turns_in_flight": worker.in_flight,
                "queue_cap": queue_cap}

    @app.get("/v1/models")
    def list_models():
        return {"default_model": default_model, "mode": mode, "allow_compile": allow_compile,
                "models": [redact_card(e.card(), mode) for e in engines.values()]}

    @app.post("/v1/sessions")
    def create_session(body: Optional[NewSession] = None, client_id: Optional[str] = Depends(client_scope)):
        return store.create_session(mode, client_id, body.title if body else "")

    @app.get("/v1/sessions")
    def list_sessions(limit: int = Query(50, ge=1, le=200), cursor: Optional[str] = None,
                      client_id: Optional[str] = Depends(client_scope)):
        sessions, next_cursor = store.list_sessions(client_id, limit, cursor)
        return {"sessions": sessions, "next_cursor": next_cursor}

    @app.get("/v1/sessions/{session_id}")
    def get_session(session_id: str, client_id: Optional[str] = Depends(client_scope)):
        return visible_session(session_id, client_id)

    @app.delete("/v1/sessions/{session_id}", status_code=204)
    def delete_session(session_id: str, client_id: Optional[str] = Depends(client_scope)) -> Response:
        if not store.delete_session(session_id, client_id):
            raise HTTPException(404, "Session not found.")
        return Response(status_code=204)

    @app.post("/v1/sessions/{session_id}/messages")
    async def post_message(session_id: str, image: Optional[UploadFile] = File(None), text: str = Form(""),
                           options: str = Form("{}"), client_id: Optional[str] = Depends(client_scope)) -> Response:
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
        if opts.test_row is not None:   # R1: test-split studies are private-mode data
            raise HTTPException(422, "Test-split studies are not available in public mode."
                                if "public" in (mode, session["mode"]) else
                                "Test-split studies need the gallery, which this server has not loaded.")
        upload = sha256 = filename = None
        if image is not None:
            upload = await image.read(MAX_UPLOAD_BYTES + 1)
            if len(upload) > MAX_UPLOAD_BYTES:
                raise HTTPException(413, TOO_LARGE_MSG)
            sha256 = await run_in_threadpool(_checked_upload, upload)
            filename = _clean_filename(image.filename)
        else:   # a text-only turn reruns the session's last image
            previous = await run_in_threadpool(_previous_image, store, session_id)
            if previous is None:
                raise HTTPException(422, NO_IMAGE_MSG)
            question = bool(text.strip()) and parse_command(text) is None
            if not question:   # a question gets the fixed answer: no model runs, so no image is used or recorded
                sha256, filename = previous["sha256"], previous["filename"]
        if not worker.reserve():
            raise HTTPException(429, "The server is busy with {} turns; try again shortly.".format(queue_cap))
        try:
            user_message_id, message_id = await run_in_threadpool(
                store.start_turn, session_id, text, session["mode"], dict(opts.model_dump(), model=model),
                sha256, filename)
        except KeyError:   # deleted since it was looked up
            worker.release()
            raise HTTPException(404, "Session not found.") from None
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

    @app.get("/v1/messages/{message_id}")
    def get_message(message_id: str, after: int = Query(0, ge=0), client_id: Optional[str] = Depends(client_scope)):
        return dict(visible_message(message_id, client_id), events=store.events_after(message_id, after))

    @app.post("/v1/messages/{message_id}/cancel")
    def cancel_message(message_id: str, client_id: Optional[str] = Depends(client_scope)):
        """Idempotent (ruling 10): a finished turn answers its final status with cancel_requested false."""
        message = visible_message(message_id, client_id)
        requested = worker.cancel(message_id)
        status = (store.get_message(message_id, client_id) or message)["status"]   # it may have ended meanwhile
        return {"id": message_id, "status": status, "cancel_requested": requested}

    @app.get("/v1/sessions/{session_id}/export")
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

    return app
