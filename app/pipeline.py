"""The turn runner (CHAT_UI_PLAN.md P3-D): one turn's stages, on the server's single worker thread.

Every event is redacted for the turn's mode, stored, then handed on, in that order, so the stored log is exactly what
was streamed: a reload replays it, and a client whose stream dropped polls it (D7). The worker never dies. Every turn
ends with a stored message_stop (done, error or aborted), and an error event carries only text the server wrote: an
UploadError's message, the store's restart message, or "Internal error (<class>)" with the traceback in the server log.

The server stores an upload before it accepts the turn (fix-1 I1), so preprocess only reads it; a test-split study (the
picker's options.test_row) is read where the dataset keeps it and never copied.

The last three stages (P5-E) end skipped, the turn going on to done, when they cannot run: retrieve with no gallery for
the turn's engine (gallery_unavailable) or both k at 0 (k_zero); label when it is off (label_off), when the labeller is
missing or down (labeler_unavailable), or while the gallery's labels are still being built (labels_pending: there would be
no neighbour agreement); score with no reference (no_reference). A test study (picked, or an upload that is one of its
files byte for byte) gets its own report's rank and is scored against its reference, which public mode never does (R1).
"""
import hashlib
import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from app.commands import NOT_A_QA_BOT
from app.engine import Cancelled, Encoded, Engine, Generated, Prepared, StageResult
from app.imaging import UploadError
from app.labels import CHEXBERT_14, LabelerUnavailable, label_agreement
from app.redact import redact_card, redact_event
from app.schemas import DISCLAIMER, Options, error_body
from app.scoring import score_pair
from app.store import RESTART_MESSAGE, Store

log = logging.getLogger("app.pipeline")

STAGES = ("preprocess", "encode", "retrieve", "generate", "label", "score")
URL_VARIANTS = ("original", "thumb", "model_input")
REFERENCE_IGNORED_PUBLIC = "Public mode does not use a reference report, so this turn is not scored."
PUBLISHED_MODEL = "hybrid_150m_m3_rrg"   # the published dumps are this model's: only its turns get a model_report line
# What the published dump decoded with (section 2): live_equals_published is compared only for a turn decoded the same way,
# stop_on_repeat off among it (P4-G).
PUBLISHED_PROTOCOL = {"decode": "beam", "beam_size": 3, "max_new_tokens": 100, "stop_on_repeat": False}

Emit = Callable[[str, Dict[str, Any]], None]


@dataclass
class PublishedDumps:
    """The published per-study lines of the test split (D18): the default model's hyps.txt and the retrieval floor's, each
    line aligned with a row of test.parquet. Private data (R1): they reach a turn's score detail in private mode only."""
    model_hyps: List[str]
    floor_hyps: List[str]

    @classmethod
    def load(cls, model_dir: Path, floor_dir: Path) -> "PublishedDumps":
        """Each directory's hyps.txt, split into lines as write_hyps_refs means them to be read back (.splitlines())."""
        return cls(_hyps(Path(model_dir)), _hyps(Path(floor_dir)))

    def line(self, kind: str, row: int) -> Optional[str]:
        """The "model" or "floor" line of a test row; None beyond the dump. ValueError for any other kind."""
        if kind not in ("model", "floor"):
            raise ValueError("kind must be 'model' or 'floor', got {!r}".format(kind))
        lines = self.model_hyps if kind == "model" else self.floor_hyps
        return lines[row] if 0 <= row < len(lines) else None


def _hyps(directory: Path) -> List[str]:
    return (directory / "hyps.txt").read_text(encoding="utf-8").splitlines()


def named_labels(row: Sequence[int]) -> Dict[str, int]:
    """A label row as {name: 0|1}, in CHEXBERT_14 order."""
    return {name: int(value) for name, value in zip(CHEXBERT_14, row)}


def retrieval_detail(gallery: Any, query: Any, k_images: int, k_reports: int, test_row: Optional[int] = None) -> Dict[str, Any]:
    """The retrieve stage's detail (section 6.3): the image neighbours, the report groups, the test study's own report rank
    when there is one, and the gallery's facts(), never its manifest (R1: that holds paths, the commit and the job)."""
    detail = {"image_neighbors": gallery.image_neighbors(query, k_images), "report_matches": gallery.report_matches(query, k_reports)}
    if test_row is not None:
        detail["true_report_rank"] = gallery.own_report_rank(query, test_row)
    detail["gallery"] = gallery.facts()
    return detail


def _query(encoded: Encoded) -> Any:
    """Encoded.pooled as the gallery's query (D4): the image vector of the report model's own tower."""
    return encoded.pooled.detach().cpu().numpy()


def _labels_pending(gallery: Any) -> bool:
    """A gallery whose CheXbert labels are not built yet (manifest labels_status pending): Gallery.labels is None."""
    return gallery is not None and hasattr(gallery, "labels") and gallery.labels is None


def _healthy(labeler: Any) -> bool:
    """The labeller's own health check; one that raises counts as down."""
    try:
        return bool(labeler.healthy())
    except Exception as exc:
        log.warning("the labeller's health check raised %s; labels are skipped", type(exc).__name__)
        return False


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000.0, 1)


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
    previous_sha256: Optional[str] = None   # with no upload: rerun this earlier upload of the session, or, with options.test_row,
                                            # the sha256 of that test study's image, read when the turn was accepted


def image_urls(user_message_id: str) -> Dict[str, str]:
    """message_start.image.urls: the turn's image variants, served by GET /v1/messages/<id>/image (P6-B)."""
    return {v: "/v1/messages/{}/image?variant={}".format(user_message_id, v) for v in URL_VARIANTS}


class _Gone(Exception):
    """The turn's session was deleted mid-turn: the turn ends quietly, without a result."""


class _Skipped(Exception):
    """A stage that had started found it could not finish (the labeller went down between its health check and its call): it
    ends skipped with this reason, and the turn goes on."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


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
    test_row: Optional[int] = None          # private mode: the test study this turn runs (picked, or its image uploaded)


class Pipeline:
    """The stages of one turn on one of the server's engines (options.model; None is the default).

    gallery is an app.gallery.Gallery (or anything with its queries), labeler anything with label() and healthy() (a
    LabelerClient or a RuleLabeler), published the PublishedDumps; each may be None. retrieval_models names the engines whose
    tower made the gallery's vectors; None is every engine.
    """

    def __init__(self, engines: Dict[str, Engine], default_model: str, store: Store, mode: str,
                 gallery: Optional[Any] = None, labeler: Optional[Any] = None, published: Optional[Any] = None,
                 drift_note: str = "", retrieval_models: Optional[Sequence[str]] = None):
        self.engines, self.default_model, self.store, self.mode = engines, default_model, store, mode
        self.gallery, self.labeler, self.published, self.drift_note = gallery, labeler, published, drift_note
        self.retrieval_models = None if retrieval_models is None else frozenset(retrieval_models)

    def retrieval_ready(self, model: str) -> bool:
        """A gallery is loaded and its vectors are this engine's tower's: a query of another tower would land in another space."""
        return self.gallery is not None and (self.retrieval_models is None or model in self.retrieval_models)

    def retrieve_upload(self, data: bytes, k_images: int, k_reports: int) -> Dict[str, Any]:
        """POST /v1/retrieve: the turn's preprocess, encode and retrieve on the default engine, without a turn, so it runs on the
        worker thread like one. An upload that is a test image byte for byte gets its own rank (private mode only). -> the
        retrieve detail as this server's mode may send it."""
        engine = self.engines[self.default_model]
        _, prepared = engine.preprocess(data)
        _, encoded = engine.encode(prepared)
        test_row = None
        if self.mode == "private":
            found = self.gallery.find_identical(prepared.sha256)
            test_row = found["row"] if found is not None and found["split"] == "test" else None
        detail = retrieval_detail(self.gallery, _query(encoded), k_images, k_reports, test_row)
        return redact_event("stage_end", {"stage": "retrieve", "detail": detail}, self.mode)["detail"]

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
        # R1: a test study is private-mode data, and it is the turn's image only when nothing was uploaded with it
        if turn.mode == "private" and job.upload is None and self.gallery is not None:
            turn.test_row = opts.test_row
        image = self._image(turn)
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
        retrieved = self._retrieve(turn, name, encoded)
        gen = self._stage(turn, "generate", lambda: self._generate(turn, engine, encoded))
        labels = self._label(turn, gen, retrieved)
        if turn.mode == "public" and (opts.reference or "").strip():
            self._send(turn, "warning", {"code": "reference_ignored_public", "message": REFERENCE_IGNORED_PUBLIC})
        self._score(turn, name, gen, labels)
        return gen

    def _image(self, turn: _Turn) -> Optional[Dict[str, Any]]:
        """message_start.image: the upload, a test study, the session's earlier image (a text-only turn), or None (a
        question)."""
        job = turn.job
        if job.upload is not None:
            sha256, source = hashlib.sha256(job.upload).hexdigest(), "upload"
        elif turn.test_row is not None and job.previous_sha256:
            sha256, source = job.previous_sha256, "test_split"
        elif job.previous_sha256:
            sha256, source = job.previous_sha256, "previous"
        else:
            return None
        return {"sha256": sha256, "filename": job.filename, "source": source, "urls": image_urls(job.user_message_id)}

    def _preprocess(self, turn: _Turn, engine: Engine, source: str) -> Tuple[StageResult, Prepared]:
        """The engine's preprocess, on the uploaded bytes, on a test study's image or, for a text-only turn, on its original
        read back. The server stored the upload before it accepted the turn (fix-1 I1). In private mode the detail names the
        test row, or the gallery image the upload is byte for byte, and an upload that is a test image makes its turn that
        study's."""
        t0 = time.perf_counter()
        job = turn.job
        if source == "upload":
            data = job.upload
        elif source == "test_split":
            data = Path(self.gallery.test_study(turn.test_row)["image"]).read_bytes()
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
        if source == "test_split":
            detail["test_row"] = turn.test_row
        elif turn.mode == "private" and self.gallery is not None:
            found = self.gallery.find_identical(prepared.sha256)
            if found is not None:
                detail["identical_to"] = found
                if found["split"] == "test":
                    turn.test_row = found["row"]
        return StageResult(detail, round((time.perf_counter() - t0) * 1000.0, 1)), prepared

    def _retrieve(self, turn: _Turn, model: str, encoded: Encoded) -> Optional[Dict[str, Any]]:
        """The k_images most similar training X-rays and the k_reports best report groups for Encoded.pooled (D4), and a test
        study's own report rank. -> the detail, or None when the stage is skipped."""
        opts = turn.job.options
        if not self.retrieval_ready(model):
            self._skip(turn, "retrieve", "gallery_unavailable")
            return None
        if opts.k_images == 0 and opts.k_reports == 0:
            self._skip(turn, "retrieve", "k_zero")
            return None

        def run() -> Tuple[StageResult, Dict[str, Any]]:
            t0 = time.perf_counter()
            detail = retrieval_detail(self.gallery, _query(encoded), opts.k_images, opts.k_reports, turn.test_row)
            return StageResult(detail, _ms(t0)), detail

        return self._stage(turn, "retrieve", run)

    def _label(self, turn: _Turn, gen: Generated, retrieved: Optional[Dict[str, Any]]) -> Optional[List[int]]:
        """The generated report's 14 labels and, for every image neighbour that has labels, how many of the 14 agree (D14).
        -> the report's label row, or None when the stage is skipped."""
        if not turn.job.options.label:
            reason = "label_off"
        elif self.labeler is None:
            reason = "labeler_unavailable"
        elif _labels_pending(self.gallery):
            reason = "labels_pending"
        elif not _healthy(self.labeler):   # a labeller that is down is skipped before the stage starts
            reason = "labeler_unavailable"
        else:
            return self._stage(turn, "label", lambda: self._labelling(gen, retrieved))
        self._skip(turn, "label", reason)
        return None

    def _labelling(self, gen: Generated, retrieved: Optional[Dict[str, Any]]) -> Tuple[StageResult, List[int]]:
        t0 = time.perf_counter()
        try:
            row = self.labeler.label([gen.report])[0]   # the raw protocol text, as the published CheXbert numbers label it
        except LabelerUnavailable:
            raise _Skipped("labeler_unavailable") from None
        neighbours = (retrieved or {}).get("image_neighbors", [])
        agreement = [dict(rank=n["rank"], **label_agreement(row, [n["labels"][name] for name in CHEXBERT_14]))
                     for n in neighbours if n.get("labels")]
        detail = {"chexbert_14": named_labels(row), "positives": [n for n, v in zip(CHEXBERT_14, row) if v],
                  "neighbor_agreement": agreement}
        return StageResult(detail, _ms(t0)), row

    def _score(self, turn: _Turn, model: str, gen: Generated, labels: Optional[List[int]]) -> None:
        """ROUGE-L and BLEU against the reference, with the thesis tables' functions (app/scoring.py), and the CheXbert parts when
        the report was labelled: then the reference is labelled too, one more call."""
        reference, source = self._reference(turn)
        if reference is None:
            self._skip(turn, "score", "no_reference")
            return
        self._stage(turn, "score", lambda: self._scoring(turn, model, gen, labels, reference, source))

    def _reference(self, turn: _Turn) -> Tuple[Optional[str], Optional[str]]:
        """-> (the text to score against, its source): the user's own (options.reference), else the test study's. None in public
        mode, which has no reference (R1)."""
        if turn.mode == "public":
            return None, None
        typed = " ".join((turn.job.options.reference or "").split())   # whitespace collapsed, as the dumps write a report
        if typed:
            return typed, "user"
        if turn.test_row is not None:
            return self.gallery.test_study(turn.test_row)["reference"], "test_split"
        return None, None

    def _scoring(self, turn: _Turn, model: str, gen: Generated, labels: Optional[List[int]], reference: str,
                 source: str) -> Tuple[StageResult, Dict[str, Any]]:
        t0 = time.perf_counter()
        reference_labels = None
        if labels is not None:
            try:
                reference_labels = self.labeler.label([reference])[0]
            except LabelerUnavailable:
                log.warning("turn %s: the labeller did not label the reference; the score has no CheXbert part", turn.job.message_id)
        detail = score_pair(gen.report, reference, labels, reference_labels)
        detail["reference_source"] = source
        if reference_labels is not None:   # the agree/disagree marks on the label chips (P4-C)
            detail["reference_chexbert_14"] = named_labels(reference_labels)
        if turn.test_row is not None and self.published is not None:
            detail["published"] = self._published(turn, model, gen)
        return StageResult(detail, _ms(t0)), detail

    def _published(self, turn: _Turn, model: str, gen: Generated) -> Dict[str, Any]:
        """The test study's published lines: the model's (only for the model the dump is of) and the retrieval floor's, and
        whether the live report equals the model's. That is compared only for a turn decoded as the dump was (PUBLISHED_PROTOCOL)
        by a model that does not stop at its own EOS; otherwise it is None."""
        row, opts = turn.test_row, turn.job.options
        line = self.published.line("model", row) if model == PUBLISHED_MODEL else None
        comparable = (line is not None and all(getattr(opts, key) == value for key, value in PUBLISHED_PROTOCOL.items())
                      and not (turn.card or {}).get("eos_trained"))
        return {"model_report": line, "floor_report": self.published.line("floor", row),
                "live_equals_published": gen.report == line if comparable else None}

    def _generate(self, turn: _Turn, engine: Engine, encoded: Encoded) -> Tuple[StageResult, Generated]:
        self._send(turn, "content_block_start", {"index": 0, "content_block": {"type": "report", "text": ""}})

        def snapshot(step: int, text: str) -> None:
            # A session deleted mid-generate ends its turn here, at this step, as _check ends one before a stage (P4-H): the store still
            # takes a deleted session's events, so without this the turn decoded on to its budget, and every turn queued behind it waited.
            self._require_session(turn.job.message_id)
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
        """_Gone once the turn's session is deleted: a message is resolved through its session. An existence query, since the generate
        stage asks at every step."""
        if not self.store.message_visible(message_id):
            raise _Gone()

    def _stage(self, turn: _Turn, name: str, run: Callable[[], Tuple[StageResult, Any]]) -> Any:
        self._check(turn)
        self._send(turn, "stage_start", {"stage": name, "index": STAGES.index(name)})
        try:
            result, value = run()
        except _Skipped as skipped:
            self._skip(turn, name, skipped.reason)
            return None
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

    def run_task(self, fn: Callable[[], Any]) -> "Future[Any]":
        """Run fn on the worker thread, behind the turns queued before it (POST /v1/retrieve: the engines are only ever used
        there). The caller reserved a slot, which is given back once fn has ended, however it ended."""
        def task() -> Any:
            try:
                return fn()
            finally:
                self.release()

        try:
            return self._pool.submit(task)
        except RuntimeError:   # the pool is shut down: the server is stopping
            self.release()
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
