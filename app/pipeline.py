"""The turn runner (CHAT_UI_PLAN.md P3-D): one turn's stages, on the server's single worker thread.

Every event is redacted for the turn's mode, stored, then handed on, in that order, so the stored log is exactly what
was streamed: a reload replays it, and a client whose stream dropped polls it (D7). The worker never dies. Every turn
ends with a stored message_stop (done, error or aborted), and an error event carries only text the server wrote: an
UploadError's message, the store's restart message, or "Internal error (<class>)" with the traceback in the server log.

The server stores an upload before it accepts the turn (fix-1 I1), so preprocess only reads it. Until P5-E the last
three stages end skipped with fixed reasons (P5-E keeps them when it adds the real stages).
"""
import hashlib
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

from app.commands import NOT_A_QA_BOT
from app.engine import Cancelled, Encoded, Engine, Generated, Prepared, StageResult
from app.imaging import UploadError
from app.redact import redact_card, redact_event
from app.schemas import DISCLAIMER, Options, error_body
from app.store import RESTART_MESSAGE, Store

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
    """The turn's session was deleted mid-turn: the turn ends quietly, without a result."""


class Stop(threading.Event):
    """A turn's stop flag, which the engine checks at every step, and who set it first: "user" (a Stop: the turn ends
    aborted) or "shutdown" (the server is stopping: it ends like crash recovery, error with server_restart)."""

    def __init__(self) -> None:
        super().__init__()
        self._by_lock = threading.Lock()
        self.by: Optional[str] = None

    def stop(self, by: str) -> None:
        with self._by_lock:
            if self.by is None:   # the first reason stands: a Stop pressed before a shutdown still ends aborted
                self.by = by
        self.set()


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
            if isinstance(cancel, Stop) and cancel.by == "shutdown":   # ends as crash recovery would end it
                status, error = "error", error_body("server_restart", RESTART_MESSAGE)
            else:
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
            self._check(turn)
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
        """The engine's preprocess, on the uploaded bytes or, for a text-only turn, on its original read back. The
        server stored the upload before it accepted the turn (fix-1 I1)."""
        t0 = time.perf_counter()
        job = turn.job
        if source == "upload":
            data = job.upload
        else:
            path = self.store.upload_path(job.session_id, job.previous_sha256, "original")
            try:
                if path is None:
                    raise FileNotFoundError("the turn's image is no longer stored")
                data = path.read_bytes()
            except FileNotFoundError:
                # The session was alive at _check(). Deleted since (before the lookup or before the read), it ends
                # the turn quietly; with the session alive, only the file went missing from disk: an error.
                self._require_session(job.message_id)
                raise
        result, prepared = engine.preprocess(data)
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

    def _check(self, turn: _Turn) -> None:
        """Before each stage: a Stop or a shutdown (pressed in an earlier stage, or while the turn waited in the queue)
        ends the turn, and so does the deletion of its session."""
        if turn.cancel.is_set():
            raise Cancelled()
        self._require_session(turn.job.message_id)

    def _require_session(self, message_id: str) -> None:
        """_Gone once the turn's session is deleted: a message is resolved through its session."""
        if self.store.get_message(message_id, None) is None:
            raise _Gone()

    def _stage(self, turn: _Turn, name: str, run: Callable[[], Tuple[StageResult, Any]]) -> Any:
        self._check(turn)
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
    429), and the worker's finally counts it out once its ending is stored, before done() ends its stream. Each turn
    has a Stop that records who stopped it first, so a turn shutdown() stops ends as a server restart while a Stop
    the user pressed earlier still ends its turn aborted (fix-1 (c)).
    """

    def __init__(self, pipeline: Optional[Pipeline], cap: int):
        self.pipeline, self.cap = pipeline, int(cap)
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="turn")
        self._lock = threading.Lock()
        self._stops: Dict[str, Stop] = {}
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
        stop = Stop()
        with self._lock:
            self._stops[job.message_id] = stop
        try:
            self._pool.submit(self._run, job, emit, stop, done)
        except RuntimeError:   # the pool is shut down: the server is stopping
            with self._lock:
                self._stops.pop(job.message_id, None)
                self._count -= 1
            raise

    def _run(self, job: TurnJob, emit: Emit, stop: Stop, done: Callable[[], None]) -> None:
        try:
            self.pipeline.run(job, emit, stop)
        except Exception:   # run() handles everything itself; this only keeps the count right
            log.exception("turn %s: the runner raised", job.message_id)
        finally:
            with self._lock:
                self._stops.pop(job.message_id, None)
                self._count -= 1
            done()

    def cancel(self, message_id: str) -> bool:
        """The user's Stop for a queued or running turn: it ends aborted at its next step (D7). False if the worker no
        longer has the turn."""
        with self._lock:
            stop = self._stops.get(message_id)
        if stop is None:
            return False
        stop.stop("user")
        return True

    def shutdown(self) -> None:
        """Stop every queued and running turn at its next step, as a server restart, then wait for the worker."""
        with self._lock:
            stops = list(self._stops.values())
        for stop in stops:
            stop.stop("shutdown")
        self._pool.shutdown(wait=True)
