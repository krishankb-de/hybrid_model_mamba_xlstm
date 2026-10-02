"""SQLite session store for the chat app (CHAT_UI_PLAN.md P3-B; the schema is section 6.4).

The event log is the source of truth. The pipeline writes every streamed event here, one transaction each, before it
is sent, so a stored session replays exactly what the client saw and a dropped stream is resumed by polling it.

D22: /sc/home is NFS, where WAL's shared-memory index is unsafe. locking_mode=EXCLUSIVE is set before WAL, so one
process owns the file and the wal-index lives in heap memory (there is no -shm file). A second Store on the same
file opens only after the first is closed. Store.journal_mode is the mode SQLite actually granted; the server prints
it at start.

One connection (autocommit, explicit BEGIN IMMEDIATE) is shared by the request handlers and the pipeline worker and
guarded by one lock. Public methods take the lock once; the underscore helpers assume it is held.

client_id=None (a private server) sees every session; otherwise only that client's. Messages and images are resolved
through their session, so the same rule covers them. delete_session soft-deletes, so a turn that is still running
keeps storing its events; sweep purges rows and files for good, deleted sessions included. It removes a session's
upload directory before it commits the deletion of the rows, skips a session whose turn is still running, can be
limited to one mode, and removes upload directories that no live session owns, so uploads cannot outlive retention.
"""
import json
import logging
import os
import re
import secrets
import shutil
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from app.ids import new_id
from app.schemas import DISCLAIMER   # the one copy of the message_stop.disclaimer text

log = logging.getLogger(__name__)

RESTART_MESSAGE = "The server restarted while this turn was running."
MODES = ("private", "public")
FINAL_STATUSES = ("done", "error", "aborted")
UPLOAD_EXTS = ("png", "jpg", "webp")
UPLOAD_NAMES = {"thumb": "thumb.jpg", "model_input": "model_input.png"}   # original.<ext> is found by extension
SAFE_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")   # what new_id produces; no dot or slash, so no path can escape
SHA256 = re.compile(r"[0-9a-f]{64}")
MAX_TITLE = 80
MAX_PAGE = 200
PRIVATE_OPTION_KEYS = ("reference", "test_row")   # Options fields that exist only in private mode (R1)

DDL = """
CREATE TABLE IF NOT EXISTS sessions (
  id TEXT PRIMARY KEY, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
  title TEXT NOT NULL DEFAULT '', mode TEXT NOT NULL CHECK (mode IN ('private','public')),
  client_id TEXT, deleted_at TEXT);
CREATE TABLE IF NOT EXISTS messages (
  id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id),
  seq_in_session INTEGER NOT NULL, role TEXT NOT NULL CHECK (role IN ('user','assistant')),
  created_at TEXT NOT NULL, text TEXT NOT NULL DEFAULT '', mode TEXT NOT NULL,
  image_sha256 TEXT, image_filename TEXT, test_row INTEGER, options_json TEXT,
  status TEXT NOT NULL CHECK (status IN ('running','done','error','aborted')),
  report TEXT, display_report TEXT, provenance_json TEXT, total_ms REAL,
  UNIQUE (session_id, seq_in_session));
CREATE TABLE IF NOT EXISTS events (
  message_id TEXT NOT NULL REFERENCES messages(id), seq INTEGER NOT NULL, ts TEXT NOT NULL,
  event TEXT NOT NULL, data_json TEXT NOT NULL, PRIMARY KEY (message_id, seq));
CREATE TABLE IF NOT EXISTS artifacts (
  message_id TEXT NOT NULL REFERENCES messages(id), kind TEXT NOT NULL,
  path TEXT NOT NULL, restricted INTEGER NOT NULL DEFAULT 0);
"""

_SESSION_SELECT = ("SELECT s.id, s.created_at, s.updated_at, s.title, s.mode, "
                   "(SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id AND m.role = 'user') AS turns "
                   "FROM sessions s ")
_IN_SCOPE = "(? IS NULL OR s.client_id = ?)"   # bound twice with client_id
_EXPIRED = ("created_at < ? AND (? IS NULL OR mode = ?) AND NOT EXISTS "   # bound with (cutoff, mode, mode)
            "(SELECT 1 FROM messages m WHERE m.session_id = sessions.id AND m.status = 'running')")
_PURGE_SQL = (   # one session, in foreign-key order
    "DELETE FROM events WHERE message_id IN (SELECT id FROM messages WHERE session_id = ?)",
    "DELETE FROM artifacts WHERE message_id IN (SELECT id FROM messages WHERE session_id = ?)",
    "DELETE FROM messages WHERE session_id = ?",
    "DELETE FROM sessions WHERE id = ?",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _dumps(obj: Any) -> str:
    """The encoding of an SSE frame's data line, so a replayed frame is the stored string."""
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def _loads(text: Optional[str]) -> Any:
    return None if text is None else json.loads(text)


def _check_mode(mode: str) -> None:
    if mode not in MODES:
        raise ValueError("mode must be one of {}, got {!r}".format(MODES, mode))


def _clean_title(text: Optional[str]) -> str:
    return " ".join((text or "").split())[:MAX_TITLE]


def _text_or_none(value: Any) -> Optional[str]:
    return value if isinstance(value, str) else None


def _float_or_none(value: Any) -> Optional[float]:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _safe_id(value: Any) -> bool:
    return isinstance(value, str) and SAFE_ID.fullmatch(value) is not None


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and SHA256.fullmatch(value) is not None


def _session_dict(row: sqlite3.Row) -> Dict[str, Any]:
    return dict(row)


def _message_dict(row: sqlite3.Row) -> Dict[str, Any]:
    out = dict(row)
    out["options"] = _loads(out.pop("options_json"))
    out["provenance"] = _loads(out.pop("provenance_json"))
    return out


def _remove_tree(path: Path) -> None:
    """Remove a session's upload directory without following symlinks (a link or file in its place is unlinked).

    A missing path is fine; any other failure raises OSError, which sweep treats as "leave the rows for next time".
    """
    try:
        if path.is_symlink() or path.is_file():
            path.unlink()
        else:
            shutil.rmtree(path)
    except FileNotFoundError:
        pass


def _rmdir_quietly(path: Path) -> None:
    try:
        path.rmdir()   # only succeeds on an empty directory
    except OSError:
        pass


class Store:
    """The chat database and upload directory under ``home`` (CHAT_HOME)."""

    def __init__(self, home: Path):
        self.home = Path(home)
        self._uploads = self.home / "uploads"
        self._lock = threading.Lock()
        self.home.mkdir(parents=True, exist_ok=True)
        self._uploads.mkdir(mode=0o700, exist_ok=True)
        db = self.home / "chat.db"
        try:   # DUA: the file can hold MIMIC-derived text in private mode, so it is owner-only from its first byte
            os.close(os.open(str(db), os.O_CREAT | os.O_RDWR, 0o600))
        except OSError:
            pass
        self._con = sqlite3.connect(str(db), check_same_thread=False, isolation_level=None)
        try:
            self._con.row_factory = sqlite3.Row
            self._con.execute("PRAGMA locking_mode=EXCLUSIVE")   # before WAL, so WAL needs no shared memory (D22)
            self.journal_mode = str(self._con.execute("PRAGMA journal_mode=WAL").fetchone()[0])   # as granted
            self._con.execute("PRAGMA synchronous=NORMAL")
            self._con.execute("PRAGMA foreign_keys=ON")
            self._con.executescript(DDL)
        except BaseException:
            self._con.close()
            raise

    # ---- sessions ------------------------------------------------------------------------------------------------

    def create_session(self, mode: str, client_id: Optional[str] = None, title: str = "") -> Dict[str, Any]:
        _check_mode(mode)
        session_id, now, title = new_id("s"), _now(), _clean_title(title)
        with self._lock:
            self._con.execute(
                "INSERT INTO sessions (id, created_at, updated_at, title, mode, client_id) VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, now, now, title, mode, client_id))
        return {"id": session_id, "created_at": now, "updated_at": now, "title": title, "mode": mode, "turns": 0}

    def list_sessions(self, client_id: Optional[str], limit: int = 50,
                      cursor: Optional[str] = None) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """Newest first (ids sort by creation). The cursor is the last id of the page, or None on the last page."""
        limit = max(1, min(int(limit), MAX_PAGE))
        with self._lock:
            rows = self._con.execute(
                _SESSION_SELECT + "WHERE s.deleted_at IS NULL AND " + _IN_SCOPE + " AND (? IS NULL OR s.id < ?) "
                "ORDER BY s.id DESC LIMIT ?", (client_id, client_id, cursor, cursor, limit + 1)).fetchall()
        page = [_session_dict(r) for r in rows[:limit]]
        return page, (page[-1]["id"] if len(rows) > limit else None)

    def get_session(self, session_id: str, client_id: Optional[str]) -> Optional[Dict[str, Any]]:
        """The session with its messages in order (events come from events_after); None if hidden or deleted."""
        with self._lock:
            row = self._session_row(session_id, client_id)
            if row is None:
                return None
            messages = self._con.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY seq_in_session", (session_id,)).fetchall()
        out = _session_dict(row)
        out["messages"] = [_message_dict(m) for m in messages]
        return out

    def delete_session(self, session_id: str, client_id: Optional[str]) -> bool:
        if not _safe_id(session_id):
            return False
        with self._lock:
            cur = self._con.execute(
                "UPDATE sessions SET deleted_at = ? WHERE id = ? AND deleted_at IS NULL "
                "AND (? IS NULL OR client_id = ?)", (_now(), session_id, client_id, client_id))
            deleted = cur.rowcount == 1
        if deleted:
            shutil.rmtree(self._uploads / session_id, ignore_errors=True)
        return deleted

    def last_image(self, session_id: str) -> Optional[Dict[str, Any]]:
        """The image (or test row) of the session's latest turn that carried one; a text-only turn reuses it."""
        with self._lock:
            row = self._con.execute(
                "SELECT image_sha256, image_filename, test_row FROM messages WHERE session_id = ? AND role = 'user' "
                "AND (image_sha256 IS NOT NULL OR test_row IS NOT NULL) ORDER BY seq_in_session DESC LIMIT 1",
                (session_id,)).fetchone()
        if row is None:
            return None
        return {"sha256": row["image_sha256"], "filename": row["image_filename"], "test_row": row["test_row"]}

    # ---- turns and events ----------------------------------------------------------------------------------------

    def start_turn(self, session_id: str, text: str, mode: str, options: Dict[str, Any],
                   image_sha256: Optional[str] = None, image_filename: Optional[str] = None,
                   test_row: Optional[int] = None) -> Tuple[str, str]:
        """Store the user message (done) and the assistant message (running); -> (user_message_id, message_id).

        The user message carries the text and the image; the assistant message carries the options and, later, the
        result. The first turn also titles the session: the text, else the filename, else the test row. ``mode`` must
        be the session's own (ValueError otherwise). A test row and a reference are private-mode data (R1): a public
        session stores neither, not as a column and not inside the options JSON, and titles itself with neither.
        """
        _check_mode(mode)
        if image_sha256 is not None and not _is_sha256(image_sha256):
            raise ValueError("image_sha256 must be 64 lowercase hex characters")
        user_id, message_id, now = new_id("m"), new_id("m"), _now()
        with self._tx() as con:
            row = con.execute("SELECT title, mode FROM sessions WHERE id = ? AND deleted_at IS NULL",
                              (session_id,)).fetchone()
            if row is None:
                raise KeyError(session_id)
            if mode != row["mode"]:
                raise ValueError("mode {!r} is not the session's mode {!r}".format(mode, row["mode"]))
            if row["mode"] != "private":
                test_row = None
                options = {k: v for k, v in (options or {}).items() if k not in PRIVATE_OPTION_KEYS}
            title = (row["title"] or _clean_title(text) or _clean_title(image_filename)
                     or ("test row {}".format(test_row) if test_row is not None else ""))
            last = con.execute("SELECT COALESCE(MAX(seq_in_session), 0) FROM messages WHERE session_id = ?",
                               (session_id,)).fetchone()[0]
            con.execute(
                "INSERT INTO messages (id, session_id, seq_in_session, role, created_at, text, mode, image_sha256, "
                "image_filename, test_row, status) VALUES (?, ?, ?, 'user', ?, ?, ?, ?, ?, ?, 'done')",
                (user_id, session_id, last + 1, now, text or "", mode, image_sha256, image_filename, test_row))
            con.execute(
                "INSERT INTO messages (id, session_id, seq_in_session, role, created_at, mode, options_json, status) "
                "VALUES (?, ?, ?, 'assistant', ?, ?, ?, 'running')",
                (message_id, session_id, last + 2, now, mode, _dumps(options or {})))
            con.execute("UPDATE sessions SET updated_at = ?, title = ? WHERE id = ?", (now, title, session_id))
        return user_id, message_id

    def save_upload(self, session_id: str, sha256: str, original: bytes, ext: str,
                    thumb_jpeg: bytes, model_input_png: bytes) -> None:
        """Write uploads/<session>/<sha256>/{original.<ext>, thumb.jpg, model_input.png}.

        Atomic: the files go to a temp directory that is renamed into place, so a failed write (disk quota) leaves
        nothing behind. Once per session and hash: if all three files are there already, the first copy stands.
        """
        if not _safe_id(session_id) or not _is_sha256(sha256):
            raise ValueError("session_id and sha256 must be plain ids")
        ext = ext.lower().lstrip(".")
        ext = "jpg" if ext == "jpeg" else ext
        if ext not in UPLOAD_EXTS:
            raise ValueError("ext must be one of {}, got {!r}".format(UPLOAD_EXTS, ext))
        with self._lock:
            if not self._session_alive(session_id):
                raise KeyError(session_id)
        final = self._uploads / session_id / sha256
        names = ("original.{}".format(ext), UPLOAD_NAMES["thumb"], UPLOAD_NAMES["model_input"])

        def complete() -> bool:
            return all((final / name).is_file() for name in names)

        if complete():
            return
        tmp = final.parent / ".{}.{}.tmp".format(sha256[:12], secrets.token_hex(4))
        try:
            tmp.mkdir(parents=True, mode=0o700)
            for name, blob in zip(names, (original, thumb_jpeg, model_input_png)):
                (tmp / name).write_bytes(blob)
            if final.exists():
                shutil.rmtree(final, ignore_errors=True)   # an incomplete leftover
            try:
                os.replace(str(tmp), str(final))
            except OSError:
                if not complete():   # else a concurrent save of the same bytes won the rename
                    raise
        finally:
            shutil.rmtree(tmp, ignore_errors=True)   # after a successful rename tmp is gone and this does nothing
            if not final.exists():
                _rmdir_quietly(final.parent)   # a failed first upload leaves no empty session directory either

    def upload_path(self, session_id: str, sha256: str, variant: str) -> Optional[Path]:
        """The file of one variant ("original", "thumb" or "model_input"); None if absent, unsafe or deleted."""
        if not _safe_id(session_id) or not _is_sha256(sha256):
            return None
        with self._lock:
            if not self._session_alive(session_id):
                return None
        base = self._uploads / session_id / sha256
        if variant == "original":
            candidates = [base / "original.{}".format(ext) for ext in UPLOAD_EXTS]
        elif variant in UPLOAD_NAMES:
            candidates = [base / UPLOAD_NAMES[variant]]
        else:
            return None
        return next((p for p in candidates if p.is_file()), None)

    def append_event(self, message_id: str, event: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Store one event in its own transaction; -> data plus its seq (1, 2, ... per message, no gaps).

        Raises KeyError for a message that does not exist (a deleted session's messages still do).
        """
        with self._tx() as con:
            if con.execute("SELECT 1 FROM messages WHERE id = ?", (message_id,)).fetchone() is None:
                raise KeyError(message_id)
            return self._insert_event(con, message_id, event, data)

    def events_after(self, message_id: str, after: int = 0) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._con.execute(
                "SELECT seq, event, data_json FROM events WHERE message_id = ? AND seq > ? ORDER BY seq",
                (message_id, after)).fetchall()
        return [{"seq": r["seq"], "event": r["event"], "data": json.loads(r["data_json"])} for r in rows]

    def get_message(self, message_id: str, client_id: Optional[str]) -> Optional[Dict[str, Any]]:
        """One message, resolved through its session: hidden by scope or by deletion, it is None."""
        with self._lock:
            row = self._con.execute(
                "SELECT m.* FROM messages m JOIN sessions s ON s.id = m.session_id "
                "WHERE m.id = ? AND s.deleted_at IS NULL AND " + _IN_SCOPE,
                (message_id, client_id, client_id)).fetchone()
        return None if row is None else _message_dict(row)

    def finish_turn(self, message_id: str, status: str, report: Optional[str] = None,
                    display_report: Optional[str] = None, provenance: Optional[Dict[str, Any]] = None,
                    total_ms: Optional[float] = None) -> None:
        if status not in FINAL_STATUSES:
            raise ValueError("status must be one of {}, got {!r}".format(FINAL_STATUSES, status))
        with self._tx() as con:
            cur = con.execute(
                "UPDATE messages SET status = ?, report = ?, display_report = ?, provenance_json = ?, total_ms = ? "
                "WHERE id = ? AND role = 'assistant'",
                (status, report, display_report, None if provenance is None else _dumps(provenance),
                 None if total_ms is None else float(total_ms), message_id))
            if cur.rowcount != 1:
                raise KeyError(message_id)
            con.execute("UPDATE sessions SET updated_at = ? WHERE id = (SELECT session_id FROM messages WHERE id = ?)",
                        (_now(), message_id))

    def recover_after_restart(self) -> int:
        """Close every turn a restart left running; -> how many.

        A turn whose log has no message_stop gets the ending a failed turn gets (an error event with type
        "server_restart", then message_stop with status "error"), so a client polling it stops. A turn that did log
        its message_stop but died before finish_turn is finished from that event instead. One transaction.
        """
        with self._tx() as con:
            running = [r[0] for r in con.execute("SELECT id FROM messages WHERE status = 'running' ORDER BY id")]
            for message_id in running:
                row = con.execute("SELECT data_json FROM events WHERE message_id = ? AND event = 'message_stop' "
                                  "ORDER BY seq DESC LIMIT 1", (message_id,)).fetchone()
                if row is None:
                    self._insert_event(con, message_id, "error", {
                        "type": "error", "error": {"type": "server_restart", "message": RESTART_MESSAGE}})
                    stop = self._insert_event(con, message_id, "message_stop", {
                        "message_id": message_id, "status": "error", "total_ms": None, "report": None,
                        "display_report": None, "truncated_mid_sentence": False, "disclaimer": DISCLAIMER})
                else:
                    stop = json.loads(row[0])
                status = stop.get("status") if stop.get("status") in FINAL_STATUSES else "error"
                # the stored event is untrusted at start-up: an odd value must not stop the server from coming up
                con.execute("UPDATE messages SET status = ?, report = ?, display_report = ?, total_ms = ? WHERE id = ?",
                            (status, _text_or_none(stop.get("report")), _text_or_none(stop.get("display_report")),
                             _float_or_none(stop.get("total_ms")), message_id))
        return len(running)

    # ---- export and retention ------------------------------------------------------------------------------------

    def export(self, session_id: str, fmt: str, client_id: Optional[str]) -> Tuple[str, str, bytes]:
        """-> (filename, media type, body) for fmt "json" (the event log) or "md" (the readable record).

        Raises KeyError if the session is hidden by scope or deleted, ValueError for another format.
        """
        if fmt not in ("json", "md"):
            raise ValueError("export format must be 'json' or 'md', got {!r}".format(fmt))
        with self._lock:
            row = self._session_row(session_id, client_id)
            if row is None:
                raise KeyError(session_id)
            messages = self._con.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY seq_in_session", (session_id,)).fetchall()
            events = self._con.execute(
                "SELECT e.message_id, e.seq, e.ts, e.event, e.data_json FROM events e "
                "JOIN messages m ON m.id = e.message_id WHERE m.session_id = ? ORDER BY e.message_id, e.seq",
                (session_id,)).fetchall()
        logs: Dict[str, List[Dict[str, Any]]] = {}
        for e in events:
            logs.setdefault(e["message_id"], []).append(
                {"seq": e["seq"], "ts": e["ts"], "event": e["event"], "data": json.loads(e["data_json"])})
        session = _session_dict(row)
        docs = [dict(_message_dict(m), events=logs.get(m["id"], [])) for m in messages]
        if fmt == "json":
            body = json.dumps({"session": session, "messages": docs}, indent=2, ensure_ascii=False)
            return "session-{}.json".format(session_id), "application/json", body.encode("utf-8")
        return ("session-{}.md".format(session_id), "text/markdown; charset=utf-8",
                _markdown(session, docs).encode("utf-8"))

    def sweep(self, older_than_days: int, mode: Optional[str] = None) -> int:
        """Purge sessions created more than ``older_than_days`` ago; -> how many.

        ``mode`` limits it to sessions created in that mode (the server keeps 45 days for private, 7 for public).
        Keyed on creation time, so "sessions are deleted after N days" holds even for one still in use. Deleted
        sessions are purged too; a session whose turn is still running waits for the next sweep (run
        recover_after_restart first at start-up). Per session the upload directory is removed first and the rows
        are deleted and committed after, so a crash or a failed removal (it is logged) leaves the rows for the
        next sweep to redo. Last, upload directories that no live session owns (a crash, or a save_upload that
        raced a delete) are removed, whatever ``mode`` is.
        """
        if older_than_days < 0:
            raise ValueError("older_than_days must not be negative")
        if mode is not None:
            _check_mode(mode)
        try:
            cutoff: Optional[str] = (datetime.now(timezone.utc) - timedelta(days=older_than_days)).isoformat(
                timespec="microseconds")
        except OverflowError:   # nothing is that old
            cutoff = None
        purged = 0
        if cutoff is not None:
            with self._lock:
                ids = [r[0] for r in self._con.execute(
                    "SELECT id FROM sessions WHERE " + _EXPIRED + " ORDER BY id", (cutoff, mode, mode))]
            purged = sum(self._purge(session_id, cutoff, mode) for session_id in ids)
        self._reclaim_orphan_uploads()
        return purged

    # The two steps of sweep. Each takes the lock itself, one session at a time, so event writes are not held up long.
    def _purge(self, session_id: str, cutoff: str, mode: Optional[str]) -> int:
        """Remove one expired session in one transaction, its upload directory before its rows; -> 1, or 0 if skipped.

        Still expired and still idle is checked again here: a turn may have started since sweep chose the session.
        If the directory cannot be removed the transaction rolls back, so the rows stay and the next sweep retries.
        """
        try:
            with self._tx() as con:
                if con.execute("SELECT 1 FROM sessions WHERE id = ? AND " + _EXPIRED,
                               (session_id, cutoff, mode, mode)).fetchone() is None:
                    return 0
                if _safe_id(session_id):
                    _remove_tree(self._uploads / session_id)
                for sql in _PURGE_SQL:
                    con.execute(sql, (session_id,))
            return 1
        except OSError as exc:
            log.warning("sweep: could not remove uploads/%s (%s); its rows stay for the next sweep", session_id, exc)
            return 0

    def _reclaim_orphan_uploads(self) -> None:
        """Remove uploads/<name> directories that no live session owns (one scandir, symlinks never followed).

        Only names that look like ids and real directories qualify. The listing is taken before the sessions are
        read, so a session created meanwhile is never mistaken for an orphan: its directory can only appear after
        its row. A deleted session counts as not live, which is what reclaims what a save_upload left after a delete.
        """
        try:
            with os.scandir(str(self._uploads)) as entries:
                names = [e.name for e in entries if _safe_id(e.name) and e.is_dir(follow_symlinks=False)]
        except OSError:
            return
        if not names:
            return
        with self._lock:
            live = {r[0] for r in self._con.execute("SELECT id FROM sessions WHERE deleted_at IS NULL")}
        for name in names:
            if name not in live:
                try:
                    shutil.rmtree(self._uploads / name)
                except OSError as exc:
                    log.warning("sweep: could not remove orphaned uploads/%s (%s); the next sweep retries", name, exc)

    def close(self) -> None:
        """Release the EXCLUSIVE lock (D22); the server's lifespan calls this at shutdown."""
        with self._lock:
            self._con.close()

    # ---- helpers (the lock is held by the caller) ----------------------------------------------------------------

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        """One write transaction under the lock: BEGIN IMMEDIATE, COMMIT, or ROLLBACK if the body raises."""
        with self._lock:
            con = self._con
            con.execute("BEGIN IMMEDIATE")
            try:
                yield con
                con.execute("COMMIT")
            except BaseException:
                try:
                    con.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                raise

    def _insert_event(self, con: sqlite3.Connection, message_id: str, event: str,
                      data: Dict[str, Any]) -> Dict[str, Any]:
        seq = con.execute("SELECT COALESCE(MAX(seq), 0) + 1 FROM events WHERE message_id = ?",
                          (message_id,)).fetchone()[0]
        stored = dict(data, seq=seq)
        con.execute("INSERT INTO events (message_id, seq, ts, event, data_json) VALUES (?, ?, ?, ?, ?)",
                    (message_id, seq, _now(), event, _dumps(stored)))
        return stored

    def _session_row(self, session_id: str, client_id: Optional[str]) -> Optional[sqlite3.Row]:
        return self._con.execute(_SESSION_SELECT + "WHERE s.id = ? AND s.deleted_at IS NULL AND " + _IN_SCOPE,
                                 (session_id, client_id, client_id)).fetchone()

    def _session_alive(self, session_id: str) -> bool:
        return self._con.execute("SELECT 1 FROM sessions WHERE id = ? AND deleted_at IS NULL",
                                 (session_id,)).fetchone() is not None


# ---- markdown export -------------------------------------------------------------------------------------------------

def _utc_text(stamp: Any) -> str:
    try:
        t = datetime.fromisoformat(stamp)
    except (TypeError, ValueError):
        return str(stamp)
    if t.tzinfo is not None:
        t = t.astimezone(timezone.utc)
    return t.strftime("%Y-%m-%d %H:%M:%S UTC")


def _oneline(value: Any) -> str:
    if isinstance(value, (dict, list)):
        value = _dumps(value)
    return " ".join(str(value).split())


def _cell(value: Any) -> str:
    return _oneline(value).replace("|", "\\|")


def _quote(text: Any) -> str:
    return "\n".join("> " + line if line else ">" for line in str(text).splitlines() or [""])


def _num(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return "{:.3f}".format(value)
    return _oneline(value)


def _stage_detail(events: List[Dict[str, Any]], stage: str) -> Dict[str, Any]:
    """The detail of the last stage_end event of this stage that has one (a skipped stage has none)."""
    found: Dict[str, Any] = {}
    for e in events:
        data = e["data"]
        if e["event"] == "stage_end" and data.get("stage") == stage and isinstance(data.get("detail"), dict):
            found = data["detail"]
    return found


def _error_message(events: List[Dict[str, Any]]) -> Optional[str]:
    for e in reversed(events):
        if e["event"] == "error" and isinstance(e["data"].get("error"), dict):
            return e["data"]["error"].get("message")
    return None


def _markdown(session: Dict[str, Any], messages: List[Dict[str, Any]]) -> str:
    """Per turn: heading with the time, the image (filename or test row), the report, the 14 labels, neighbours, one
    provenance line. Neighbours (MIMIC-derived) and test rows appear only if the session is private."""
    private = session["mode"] == "private"
    header = ["Mode: {}".format(session["mode"]), "Created: {}".format(_utc_text(session["created_at"]))]
    if session["title"]:
        header.append("Title: {}".format(_oneline(session["title"])))
    out = ["# Session {}".format(session["id"]), "", "_{}_".format(DISCLAIMER), "", " · ".join(header)]
    turns: List[Dict[str, Any]] = []
    for m in messages:
        if m["role"] == "user":
            turns.append({"user": m, "assistant": None})
        elif turns and turns[-1]["assistant"] is None:
            turns[-1]["assistant"] = m
        else:   # an assistant message with no user message before it (start_turn never writes one)
            turns.append({"user": None, "assistant": m})
    for n, turn in enumerate(turns, 1):
        user, asst = turn["user"], turn["assistant"]
        out += ["", "## Turn {} — {}".format(n, _utc_text((user or asst)["created_at"]))]
        if user is not None:
            if user["image_filename"]:
                out += ["", "Image: `{}`".format(_oneline(user["image_filename"]).replace("`", "'"))]
            elif user["test_row"] is not None and private:
                out += ["", "Image: test row {}".format(user["test_row"])]
            if user["text"].strip():
                out += ["", "Text:", "", _quote(user["text"])]
        if asst is None:
            continue
        events = asst["events"]
        if asst["report"]:
            out += ["", "Report:", "", _quote(asst["report"])]
            if asst["display_report"] and asst["display_report"] != asst["report"]:
                out += ["", "Display report (sentence repaired):", "", _quote(asst["display_report"])]
        if asst["status"] != "done":
            error = _error_message(events)
            out += ["", "Status: {}{}".format(asst["status"], " · Error: {}".format(_oneline(error)) if error else "")]
        labels = _stage_detail(events, "label").get("chexbert_14")
        if isinstance(labels, dict) and labels:
            out += ["", "CheXbert-14 labels of the generated report (1 = positive):", "",
                    "| Finding | Label |", "| --- | --- |"]
            out += ["| {} | {} |".format(_cell(k), _cell(v)) for k, v in labels.items()]
        if private:
            found = _stage_detail(events, "retrieve")
            images = [x for x in found.get("image_neighbors") or [] if isinstance(x, dict)]
            reports = [x for x in found.get("report_matches") or [] if isinstance(x, dict)]
            if images:
                out += ["", "Nearest images:", ""]
                out += ["- #{} · similarity {}{}".format(
                    x.get("rank"), _num(x.get("similarity")),
                    " · study {}".format(_oneline(x["study_id"])) if x.get("study_id") else "") for x in images]
            if reports:
                out += ["", "Nearest reports:", ""]
                for x in reports:
                    out.append("- #{} · similarity {}".format(x.get("rank"), _num(x.get("similarity"))))
                    if x.get("report"):
                        out.append("  " + _quote(x["report"]).replace("\n", "\n  "))
        provenance = dict(asst["provenance"] or {})
        if asst["total_ms"] is not None and "total_ms" not in provenance:
            provenance["total_ms"] = asst["total_ms"]
        parts = ["{}={}".format(k, _oneline(v)) for k, v in provenance.items() if v is not None]
        out += ["", "Provenance: " + (" · ".join(parts) if parts else "none recorded")]
    return "\n".join(out) + "\n"
