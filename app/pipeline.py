"""The turn runner (CHAT_UI_PLAN.md P3-D): one turn's stages, on the server's single worker thread.

Every event is redacted for the turn's mode, stored, then handed on, in that order, so the stored log is exactly what
was streamed: a reload replays it, and a client whose stream dropped polls it (D7). The worker never dies. Every turn
ends with a stored message_stop (done, error or aborted), and an error event carries only text the server wrote: an
UploadError's message, or "Internal error (<class>)" with the traceback in the server log.

Until P5-E the last three stages end skipped with fixed reasons (P5-E keeps them when it adds the real stages).
"""
import hashlib
import io
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

from app.commands import NOT_A_QA_BOT
from app.engine import Cancelled, Encoded, Engine, Generated, Prepared, StageResult
from app.imaging import UploadError, thumbnail_jpeg
from app.redact import redact_card, redact_event
from app.schemas import DISCLAIMER, Options, error_body
from app.store import Store

log = logging.getLogger("app.pipeline")

STAGES = ("preprocess", "encode", "retrieve", "generate", "label", "score")
URL_VARIANTS = ("original", "thumb", "model_input")
REFERENCE_IGNORED_PUBLIC = "Public mode does not use a reference report, so this turn is not scored."

Emit = Callable[[str, Dict[str, Any]], None]


@dataclass
class TurnJob:
    session_id: str
    user_message_id: str
    message_id: str
    text: str
    upload: Optional[bytes]
    filename: Optional[str]
    options: Options
    mode: Optional[str] = None              # the session's own mode; the stricter of it and the pipeline's applies
    previous_sha256: Optional[str] = None   # with no upload: rerun this earlier upload of the session


def image_urls(user_message_id: str) -> Dict[str, str]:
    """message_start.image.urls: the turn's image variants, served by GET /v1/messages/<id>/image (P6-B)."""
    return {v: "/v1/messages/{}/image?variant={}".format(user_message_id, v) for v in URL_VARIANTS}


class _Gone(Exception):
    """The turn's session was deleted mid-turn, so its upload or its rows can no longer be stored."""


@dataclass
class _Turn:
    job: TurnJob
    mode: str
    emit: Emit
    cancel: threading.Event
    t0: float = field(default_factory=time.perf_counter)
    card: Optional[Dict[str, Any]] = None   # the engine's card, for the provenance; None when no model ran


class Pipeline:
    """The stages of one turn on one of the server's engines (options.model; None is the default)."""

    def __init__(self, engines: Dict[str, Engine], default_model: str, store: Store, mode: str,
                 gallery: Optional[Any] = None, labeler: Optional[Any] = None, published: Optional[Any] = None,
                 drift_note: str = ""):
        self.engines, self.default_model, self.store, self.mode = engines, default_model, store, mode
        self.gallery, self.labeler, self.published, self.drift_note = gallery, labeler, published, drift_note

    def run(self, job: TurnJob, emit: Emit, cancel: threading.Event) -> None:
        """Run one turn and store its ending; emit gets each stored event. Never raises (ruling 8)."""
        # A private server also lists the sessions a public run of the same CHAT_HOME made: redact for public then.
        mode = "public" if "public" in (job.mode, self.mode) else "private"
        turn = _Turn(job, mode, emit, cancel)
        status, gen, error = "done", None, None
        try:
            gen = self._stages(turn)
        except Cancelled:
            status = "aborted"
        except UploadError as exc:   # its messages are written for the user
            status, error = "error", error_body("validation_error", str(exc))
        except _Gone:
            status = "aborted"
            log.warning("turn %s: its session was deleted mid-turn; the turn stops here", job.message_id)
        except Exception as exc:   # never its text in an event: it can hold report text or paths
            log.exception("turn %s failed", job.message_id)
            status, error = "error", error_body("model_error", "Internal error ({})".format(type(exc).__name__))
        try:
            if error is not None:
                self._send(turn, "error", error)
            self._stop(turn, status, gen)
        except Exception:   # the store failed, or the turn's rows were purged: log it; the worker lives on
            log.exception("turn %s: could not store its ending", job.message_id)

    def _stages(self, turn: _Turn) -> Optional[Generated]:
        job, opts = turn.job, turn.job.options
        name = opts.model or self.default_model
        engine = self.engines[name]
        turn.card = engine.card()
        image = self._image(job)
        self._send(turn, "message_start", {
            "message_id": job.message_id, "user_message_id": job.user_message_id, "session_id": job.session_id,
            "mode": turn.mode, "model": turn.card, "options": dict(opts.model_dump(), model=name), "image": image})
        if image is None:   # a question and no image: the one fixed answer (spec §4); no model runs
            turn.card = None
            self._send(turn, "warning", {"code": "not_a_command", "message": NOT_A_QA_BOT})
            return None
        prepared = self._stage(turn, "preprocess", lambda: self._preprocess(turn, engine, image["source"]))
        encoded = self._stage(turn, "encode", lambda: engine.encode(prepared))
        self._skip(turn, "retrieve", "gallery_unavailable")
        gen = self._stage(turn, "generate", lambda: self._generate(turn, engine, encoded))
        self._skip(turn, "label", "labeler_unavailable" if opts.label else "label_off")
        if turn.mode == "public" and (opts.reference or "").strip():
            self._send(turn, "warning", {"code": "reference_ignored_public", "message": REFERENCE_IGNORED_PUBLIC})
        self._skip(turn, "score", "no_reference")
        return gen

    def _image(self, job: TurnJob) -> Optional[Dict[str, Any]]:
        """message_start.image: the upload, the session's earlier image (a text-only turn), or None (a question)."""
        if job.upload is not None:
            sha256, source = hashlib.sha256(job.upload).hexdigest(), "upload"
        elif job.previous_sha256:
            sha256, source = job.previous_sha256, "previous"
        else:
            return None
        return {"sha256": sha256, "filename": job.filename, "source": source, "urls": image_urls(job.user_message_id)}

    def _preprocess(self, turn: _Turn, engine: Engine, source: str) -> Tuple[StageResult, Prepared]:
        """The engine's preprocess. An upload is stored here, once per session and hash: the bytes as sent, the
        thumbnail and the 224x224 model input (P6-B serves them). A text-only turn reads its original back."""
        t0 = time.perf_counter()
        job = turn.job
        if source == "upload":
            data = job.upload
        else:
            path = self.store.upload_path(job.session_id, job.previous_sha256, "original")
            if path is None:   # the server saw the file when it accepted the turn: the session was deleted since
                raise _Gone()
            data = path.read_bytes()
        result, prepared = engine.preprocess(data)
        if source == "upload":
            png = io.BytesIO()
            prepared.model_input.save(png, "PNG")
            try:
                self.store.save_upload(job.session_id, prepared.sha256, data, result.detail["format"].lower(),
                                       thumbnail_jpeg(prepared.image), png.getvalue())
            except KeyError:   # the session was deleted
                raise _Gone() from None
        detail = dict(result.detail, source=source)
        return StageResult(detail, round((time.perf_counter() - t0) * 1000.0, 1)), prepared

    def _generate(self, turn: _Turn, engine: Engine, encoded: Encoded) -> Tuple[StageResult, Generated]:
        self._send(turn, "content_block_start", {"index": 0, "content_block": {"type": "report", "text": ""}})

        def snapshot(step: int, text: str) -> None:
            self._send(turn, "content_block_delta",
                       {"index": 0, "delta": {"type": "beam_snapshot", "step": step, "text": text}})

        result, gen = engine.generate(encoded, turn.job.options, snapshot, turn.cancel)
        self._send(turn, "content_block_stop", {"index": 0})
        return result, gen

    def _stage(self, turn: _Turn, name: str, run: Callable[[], Tuple[StageResult, Any]]) -> Any:
        if turn.cancel.is_set():   # stopped during an earlier stage, or while the turn waited in the queue
            raise Cancelled()
        self._send(turn, "stage_start", {"stage": name, "index": STAGES.index(name)})
        result, value = run()
        self._send(turn, "stage_end", {"stage": name, "ms": result.ms, "detail": result.detail})
        return value

    def _skip(self, turn: _Turn, name: str, reason: str) -> None:
        self._send(turn, "stage_end", {"stage": name, "skipped": reason})

    def _send(self, turn: _Turn, event: str, data: Dict[str, Any]) -> None:
        """Redact, store, hand on (D7). None from redact_event drops the event: it is neither stored nor sent."""
        out = redact_event(event, data, turn.mode)
        if out is None:
            return
        try:
            stored = self.store.append_event(turn.job.message_id, event, out)
        except KeyError:
            raise _Gone() from None
        turn.emit(event, stored)

    def _stop(self, turn: _Turn, status: str, gen: Optional[Generated]) -> None:
        """message_stop, stored, then the turn finished in the store, then sent: a client that acts on message_stop
        reads the final status. The provenance is the redacted card (ruling 13); message_stop carries none."""
        job = turn.job
        total_ms = round((time.perf_counter() - turn.t0) * 1000.0, 1)
        out = redact_event("message_stop", {
            "message_id": job.message_id, "status": status, "total_ms": total_ms,
            "report": gen.report if gen else None, "display_report": gen.display_report if gen else None,
            "truncated_mid_sentence": bool(gen and gen.truncated_mid_sentence), "disclaimer": DISCLAIMER}, turn.mode)
        provenance = None if turn.card is None else redact_card(turn.card, turn.mode)
        try:
            stored = self.store.append_event(job.message_id, "message_stop", out)
            self.store.finish_turn(job.message_id, status, report=out["report"], display_report=out["display_report"],
                                   provenance=provenance, total_ms=total_ms)
        except KeyError:
            raise _Gone() from None
        turn.emit("message_stop", stored)


class Worker:
    """The server's one worker thread (ThreadPoolExecutor, max_workers=1) and its admission count.

    At most `cap` turns are accepted and unfinished at once: reserve() counts a turn in (False means refuse it with
    429), and the worker's finally counts it out once its ending is stored, before done() ends its stream.
    """

    def __init__(self, pipeline: Optional[Pipeline], cap: int):
        self.pipeline, self.cap = pipeline, int(cap)
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="turn")
        self._lock = threading.Lock()
        self._cancels: Dict[str, threading.Event] = {}
        self._count = 0

    @property
    def in_flight(self) -> int:
        with self._lock:
            return self._count

    def reserve(self) -> bool:
        with self._lock:
            if self._count >= self.cap:
                return False
            self._count += 1
            return True

    def release(self) -> None:
        """Give back a reservation whose turn was never submitted."""
        with self._lock:
            self._count -= 1

    def submit(self, job: TurnJob, emit: Emit, done: Callable[[], None]) -> None:
        """Queue a reserved turn; done() runs on the worker after the turn has ended, however it ended."""
        cancel = threading.Event()
        with self._lock:
            self._cancels[job.message_id] = cancel
        try:
            self._pool.submit(self._run, job, emit, cancel, done)
        except RuntimeError:   # the pool is shut down: the server is stopping
            with self._lock:
                self._cancels.pop(job.message_id, None)
                self._count -= 1
            raise

    def _run(self, job: TurnJob, emit: Emit, cancel: threading.Event, done: Callable[[], None]) -> None:
        try:
            self.pipeline.run(job, emit, cancel)
        except Exception:   # run() handles everything itself; this only keeps the count right
            log.exception("turn %s: the runner raised", job.message_id)
        finally:
            with self._lock:
                self._cancels.pop(job.message_id, None)
                self._count -= 1
            done()

    def cancel(self, message_id: str) -> bool:
        """Stop a queued or running turn at its next step (D7); False if the worker no longer has it."""
        with self._lock:
            event = self._cancels.get(message_id)
        if event is None:
            return False
        event.set()
        return True

    def shutdown(self) -> None:
        """Stop every turn at its next step, then wait for the worker: each one stores its ending as aborted."""
        with self._lock:
            events = list(self._cancels.values())
        for event in events:
            event.set()
        self._pool.shutdown(wait=True)
