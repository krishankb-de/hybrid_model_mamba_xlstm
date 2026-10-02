"""CHAT_UI_PLAN.md P3-B: the event log is the source of truth and survives restarts."""
import json
import os
import re
import sqlite3
import stat
import subprocess
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.store import Store

REPO_ROOT = Path(__file__).resolve().parent.parent


def _turn(s, mode="private", client=None):
    sess = s.create_session(mode, client)
    user_id, mid = s.start_turn(sess["id"], "hi", mode, {"beam_size": 3})
    return sess, user_id, mid


def test_events_get_contiguous_seq_and_survive_reopen(tmp_path):
    s = Store(tmp_path)
    _, _, mid = _turn(s)
    assert [s.append_event(mid, "stage_start", {"stage": "x"})["seq"] for _ in range(3)] == [1, 2, 3]
    s.close()   # locking_mode=EXCLUSIVE (D22): one open connection owns the file
    reopened = Store(tmp_path)
    assert [e["seq"] for e in reopened.events_after(mid, 0)] == [1, 2, 3]
    assert [e["seq"] for e in reopened.events_after(mid, 2)] == [3]
    assert reopened.events_after(mid, 0)[0]["data"] == {"stage": "x", "seq": 1}


def test_restart_marks_running_turns_as_error(tmp_path):
    s = Store(tmp_path)
    sess, _, mid = _turn(s)
    s.close()
    restarted = Store(tmp_path)
    assert restarted.recover_after_restart() == 1
    assert restarted.get_message(mid, None)["status"] == "error"
    restarted.close()


def test_public_sessions_are_scoped_to_their_client(tmp_path):   # Review Focus 3
    s = Store(tmp_path)
    a = s.create_session("public", "client-a")
    s.create_session("public", "client-b")
    assert [x["id"] for x in s.list_sessions("client-a")[0]] == [a["id"]]
    assert s.get_session(a["id"], "client-b") is None
    assert s.delete_session(a["id"], "client-b") is False


def test_delete_removes_uploads_and_hides_the_session(tmp_path):
    s = Store(tmp_path)
    sess, _, _ = _turn(s)
    s.save_upload(sess["id"], "ab" * 32, b"png-bytes", "png", b"jpg", b"png")
    assert s.upload_path(sess["id"], "ab" * 32, "original").exists()
    assert s.delete_session(sess["id"], None) is True
    assert s.upload_path(sess["id"], "ab" * 32, "original") is None
    assert s.get_session(sess["id"], None) is None


def test_export_json_is_the_event_log_and_md_has_the_report(tmp_path):
    s = Store(tmp_path)
    sess, _, mid = _turn(s)
    s.append_event(mid, "message_stop", {"status": "done", "report": "Findings: clear."})
    s.finish_turn(mid, "done", report="Findings: clear.", display_report="Findings: clear.", total_ms=12.0)
    name, media, body = s.export(sess["id"], "json", None)
    assert media == "application/json" and json.loads(body)["messages"][1]["events"][0]["event"] == "message_stop"
    name, media, body = s.export(sess["id"], "md", None)
    assert media.startswith("text/markdown") and b"Findings: clear." in body


def test_sweep_deletes_only_old_sessions(tmp_path):
    s = Store(tmp_path)
    old = s.create_session("private")
    new = s.create_session("private")
    s._con.execute("UPDATE sessions SET created_at='2000-01-01T00:00:00+00:00' WHERE id=?", (old["id"],))
    s._con.commit()
    assert s.sweep(older_than_days=30) == 1
    assert s.get_session(new["id"], None) is not None


# --- beyond the brief: pragmas and schema (D22), scoping, turns, uploads, recovery, export, retention ---------------

COLUMNS = {   # CHAT_UI_PLAN.md 6.4, in declaration order
    "sessions": ["id", "created_at", "updated_at", "title", "mode", "client_id", "deleted_at"],
    "messages": ["id", "session_id", "seq_in_session", "role", "created_at", "text", "mode", "image_sha256",
                 "image_filename", "test_row", "options_json", "status", "report", "display_report",
                 "provenance_json", "total_ms"],
    "events": ["message_id", "seq", "ts", "event", "data_json"],
    "artifacts": ["message_id", "kind", "path", "restricted"],
}


def test_open_creates_the_home_the_database_and_the_uploads_dir(tmp_path):
    home = tmp_path / "a" / "chat_sessions"
    Store(home).close()
    assert (home / "chat.db").is_file() and (home / "uploads").is_dir()


def test_schema_is_the_binding_6_4_ddl(tmp_path):
    s = Store(tmp_path)
    assert {t: [r[1] for r in s._con.execute("PRAGMA table_info({})".format(t))] for t in COLUMNS} == COLUMNS
    with pytest.raises(sqlite3.IntegrityError):   # the CHECK on mode is in force
        s._con.execute("INSERT INTO sessions (id, created_at, updated_at, mode) VALUES ('x', 't', 't', 'other')")
    with pytest.raises(sqlite3.IntegrityError):   # foreign_keys=ON
        s._con.execute("INSERT INTO events (message_id, seq, ts, event, data_json) "
                       "VALUES ('m_none', 1, 't', 'e', '{}')")


def test_journal_mode_is_wal_and_the_d22_pragmas_are_in_force(tmp_path):
    s = Store(tmp_path)
    assert s.journal_mode == "wal"   # the mode SQLite granted, which the server prints at start
    pragma = lambda name: s._con.execute("PRAGMA " + name).fetchone()[0]
    assert (pragma("journal_mode"), pragma("locking_mode"), pragma("synchronous"), pragma("foreign_keys")) == \
        ("wal", "exclusive", 1, 1)
    s.append_event(_turn(s)[2], "stage_start", {})
    assert not (tmp_path / "chat.db-shm").exists()   # WAL without a shared-memory index: the point of D22
    s.close()


def test_the_file_is_locked_to_other_connections_until_close(tmp_path):
    s = Store(tmp_path)
    other = sqlite3.connect(str(tmp_path / "chat.db"), timeout=0)
    with pytest.raises(sqlite3.OperationalError, match="locked"):
        other.execute("SELECT COUNT(*) FROM sessions").fetchone()
    s.close()
    assert other.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0
    other.close()


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits")
def test_database_and_uploads_are_owner_only(tmp_path):   # the file can hold MIMIC-derived text in private mode
    Store(tmp_path).close()
    assert stat.S_IMODE((tmp_path / "chat.db").stat().st_mode) & 0o077 == 0
    assert stat.S_IMODE((tmp_path / "uploads").stat().st_mode) & 0o077 == 0


def test_timestamps_are_utc_iso_8601(tmp_path):
    s = Store(tmp_path)
    sess, _, mid = _turn(s)
    s.append_event(mid, "stage_start", {})
    stamps = [sess["created_at"], sess["updated_at"], s.get_message(mid, None)["created_at"],
              s._con.execute("SELECT ts FROM events").fetchone()[0]]
    assert all(datetime.fromisoformat(t).utcoffset() == timedelta(0) for t in stamps)


def test_a_private_server_sees_every_session(tmp_path):
    s = Store(tmp_path)
    ids = {s.create_session("private")["id"], s.create_session("private", "x")["id"],
           s.create_session("public", "y")["id"]}
    assert {x["id"] for x in s.list_sessions(None)[0]} == ids
    assert s.list_sessions("nobody")[0] == []


def test_list_sessions_is_newest_first_and_pages_with_a_cursor(tmp_path):
    s = Store(tmp_path)
    ids = [s.create_session("private")["id"] for _ in range(5)]
    page1, cursor = s.list_sessions(None, limit=2)
    assert [x["id"] for x in page1] == ids[::-1][:2] and cursor == page1[-1]["id"]
    page2, cursor = s.list_sessions(None, limit=2, cursor=cursor)
    page3, end = s.list_sessions(None, limit=2, cursor=cursor)
    assert [x["id"] for x in page2 + page3] == ids[::-1][2:] and end is None
    assert s.list_sessions(None, limit=5)[1] is None   # a full page that is also the last one has no cursor


def test_start_turn_stores_the_user_and_assistant_messages(tmp_path):
    s = Store(tmp_path)
    sess = s.create_session("private")
    sha = "cd" * 32
    uid, mid = s.start_turn(sess["id"], "look at this", "private", {"beam_size": 5}, sha, "chest.png")
    assert sess["id"].startswith("s_") and uid.startswith("m_") and uid < mid   # ids sort by creation
    user, asst = s.get_message(uid, None), s.get_message(mid, None)
    assert (user["role"], user["seq_in_session"], user["status"], user["text"], user["session_id"]) == \
        ("user", 1, "done", "look at this", sess["id"])
    assert (user["image_sha256"], user["image_filename"], user["test_row"]) == (sha, "chest.png", None)
    assert (asst["role"], asst["seq_in_session"], asst["status"], asst["options"], asst["mode"]) == \
        ("assistant", 2, "running", {"beam_size": 5}, "private")
    assert (asst["report"], asst["provenance"], asst["total_ms"]) == (None, None, None)
    got = s.get_session(sess["id"], None)
    assert got["turns"] == 1 and [m["id"] for m in got["messages"]] == [uid, mid]
    assert "client_id" not in got and "client_id" not in s.list_sessions(None)[0][0]
    assert s.get_message(s.start_turn(sess["id"], "again", "private", {})[1], None)["seq_in_session"] == 4


def test_start_turn_needs_a_live_session_and_a_real_hash(tmp_path):
    s = Store(tmp_path)
    sess = s.create_session("private")
    with pytest.raises(KeyError):
        s.start_turn("s_missing", "x", "private", {})
    with pytest.raises(ValueError):
        s.start_turn(sess["id"], "x", "private", {}, image_sha256="../../etc")
    with pytest.raises(ValueError):
        s.start_turn(sess["id"], "x", "elsewhere", {})
    s.delete_session(sess["id"], None)
    with pytest.raises(KeyError):
        s.start_turn(sess["id"], "x", "private", {})


def test_session_title_is_the_first_text_else_the_filename_else_the_test_row(tmp_path):
    s = Store(tmp_path)
    text, name, row = (s.create_session("private") for _ in range(3))
    s.start_turn(text["id"], "  what   about\nthis  ", "private", {}, "ab" * 32, "chest.png")
    s.start_turn(text["id"], "second", "private", {})
    s.start_turn(name["id"], "", "private", {}, "ab" * 32, "chest.png")
    s.start_turn(row["id"], "", "private", {}, test_row=7)
    assert [s.get_session(x["id"], None)["title"] for x in (text, name, row)] == \
        ["what about this", "chest.png", "test row 7"]
    assert s.create_session("private", title="  Mine ")["title"] == "Mine"


def test_last_image_is_the_latest_turn_that_carried_one(tmp_path):
    s = Store(tmp_path)
    sess = s.create_session("private")
    assert s.last_image(sess["id"]) is None
    s.start_turn(sess["id"], "just text", "private", {})
    assert s.last_image(sess["id"]) is None
    s.start_turn(sess["id"], "", "private", {}, "ab" * 32, "a.png")
    s.start_turn(sess["id"], "text only", "private", {})
    assert s.last_image(sess["id"]) == {"sha256": "ab" * 32, "filename": "a.png", "test_row": None}
    s.start_turn(sess["id"], "", "private", {}, test_row=7)
    assert s.last_image(sess["id"]) == {"sha256": None, "filename": None, "test_row": 7}


def test_append_event_returns_what_it_stored_and_a_failure_leaves_no_gap(tmp_path):
    s = Store(tmp_path)
    _, _, mid = _turn(s)
    out = s.append_event(mid, "content_block_delta", {"index": 0, "delta": {"text": "caf\u00e9"}})
    assert out == {"index": 0, "delta": {"text": "caf\u00e9"}, "seq": 1}
    assert s.events_after(mid) == [{"seq": 1, "event": "content_block_delta", "data": out}]
    raw = s._con.execute("SELECT data_json FROM events").fetchone()[0]
    assert raw == '{"index":0,"delta":{"text":"caf\u00e9"},"seq":1}'   # the SSE data line, compact and UTF-8
    with pytest.raises(sqlite3.IntegrityError):   # no such message
        s.append_event("m_missing", "stage_start", {})
    with pytest.raises(TypeError):   # not JSON: nothing stored, the next seq is still 2
        s.append_event(mid, "stage_start", {"x": object()})
    assert s.append_event(mid, "stage_end", {})["seq"] == 2


def test_concurrent_appends_keep_seq_contiguous(tmp_path):
    s = Store(tmp_path)
    _, _, mid = _turn(s)

    def work():
        for _ in range(25):
            s.append_event(mid, "stage_start", {})

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert [e["seq"] for e in s.events_after(mid)] == list(range(1, 201))


def test_messages_follow_their_sessions_scope_and_deletion(tmp_path):
    s = Store(tmp_path)
    sess, user_id, mid = _turn(s, "public", "client-a")
    assert s.get_message(mid, "client-a")["id"] == mid and s.get_message(mid, None)["id"] == mid
    assert s.get_message(mid, "client-b") is None
    assert s.delete_session(sess["id"], "client-a") is True
    assert s.get_message(mid, "client-a") is None and s.get_message(user_id, None) is None
    assert s.append_event(mid, "stage_start", {})["seq"] == 1   # soft delete: a turn still running keeps storing
    assert s.delete_session(sess["id"], "client-a") is False


def test_finish_turn_stores_the_result(tmp_path):
    s = Store(tmp_path)
    sess, _, mid = _turn(s)
    before = s.get_session(sess["id"], None)["updated_at"]
    s.finish_turn(mid, "aborted", report="part", display_report="part.", provenance={"model": "m3"}, total_ms=1.5)
    m = s.get_message(mid, None)
    assert (m["status"], m["report"], m["display_report"], m["provenance"], m["total_ms"]) == \
        ("aborted", "part", "part.", {"model": "m3"}, 1.5)
    assert s.get_session(sess["id"], None)["updated_at"] > before
    with pytest.raises(ValueError):
        s.finish_turn(mid, "running")
    with pytest.raises(KeyError):
        s.finish_turn("m_missing", "done")


def test_uploads_are_laid_out_per_session_and_hash(tmp_path):
    s = Store(tmp_path)
    sess, _, _ = _turn(s)
    sha = "ab" * 32
    s.save_upload(sess["id"], sha, b"jpeg-bytes", "jpeg", b"thumb", b"model")
    base = tmp_path / "uploads" / sess["id"] / sha
    assert sorted(p.name for p in base.iterdir()) == ["model_input.png", "original.jpg", "thumb.jpg"]
    assert s.upload_path(sess["id"], sha, "original") == base / "original.jpg"
    assert s.upload_path(sess["id"], sha, "thumb").read_bytes() == b"thumb"
    assert s.upload_path(sess["id"], sha, "model_input").read_bytes() == b"model"
    assert s.upload_path(sess["id"], sha, "nope") is None
    assert s.upload_path(sess["id"], "cd" * 32, "thumb") is None   # a hash that was never saved
    assert s.upload_path("s_missing", sha, "thumb") is None   # a session that does not exist
    s.save_upload(sess["id"], sha, b"other", "jpg", b"t2", b"m2")   # once per session and hash: the first copy stands
    assert s.upload_path(sess["id"], sha, "thumb").read_bytes() == b"thumb"
    assert sorted(p.name for p in (tmp_path / "uploads" / sess["id"]).iterdir()) == [sha]


@pytest.mark.parametrize("session_id", ["../x", "a/b", "..", "", "s_1\n", "x" * 65, "a b", "a.b", 7])
def test_a_crafted_session_id_cannot_escape_the_uploads_directory(tmp_path, session_id):
    s = Store(tmp_path)
    with pytest.raises(ValueError):
        s.save_upload(session_id, "ab" * 32, b"x", "png", b"x", b"x")
    assert s.upload_path(session_id, "ab" * 32, "original") is None
    assert s.delete_session(session_id, None) is False
    assert not (tmp_path / "x").exists() and list((tmp_path / "uploads").iterdir()) == []


@pytest.mark.parametrize("sha", ["../" + "a" * 61, "AB" * 32, "ab" * 31, "ab" * 33, "ab" * 32 + "\n", "", None])
def test_a_crafted_hash_cannot_escape_the_uploads_directory(tmp_path, sha):
    s = Store(tmp_path)
    sess, _, _ = _turn(s)
    with pytest.raises(ValueError):
        s.save_upload(sess["id"], sha, b"x", "png", b"x", b"x")
    assert s.upload_path(sess["id"], sha, "original") is None
    assert list((tmp_path / "uploads").iterdir()) == []


@pytest.mark.parametrize("ext", ["svg", "", "../x", "png/../../x", "exe"])
def test_an_unknown_extension_is_refused(tmp_path, ext):
    s = Store(tmp_path)
    sess, _, _ = _turn(s)
    with pytest.raises(ValueError):
        s.save_upload(sess["id"], "ab" * 32, b"x", ext, b"x", b"x")
    assert list((tmp_path / "uploads").iterdir()) == []


def test_delete_removes_the_files_and_upload_path_refuses_a_deleted_session_on_its_own(tmp_path):
    s = Store(tmp_path)   # each guard alone: the files go, and a file that survives is still not served
    sess, _, _ = _turn(s)
    sha = "ab" * 32
    s.save_upload(sess["id"], sha, b"o", "png", b"t", b"m")
    assert s.delete_session(sess["id"], None) is True
    assert not (tmp_path / "uploads" / sess["id"]).exists()
    leftover = tmp_path / "uploads" / sess["id"] / sha
    leftover.mkdir(parents=True)   # as if an rmtree on NFS had failed half way
    (leftover / "thumb.jpg").write_bytes(b"x")
    assert s.upload_path(sess["id"], sha, "thumb") is None


def test_save_upload_needs_a_live_session(tmp_path):
    s = Store(tmp_path)
    sess, _, _ = _turn(s)
    with pytest.raises(KeyError):
        s.save_upload("s_missing", "ab" * 32, b"x", "png", b"x", b"x")
    s.delete_session(sess["id"], None)
    with pytest.raises(KeyError):
        s.save_upload(sess["id"], "ab" * 32, b"x", "png", b"x", b"x")
    assert list((tmp_path / "uploads").iterdir()) == []


def test_a_failed_upload_write_leaves_no_partial_files(tmp_path, monkeypatch):   # P8-D: disk quota on upload
    s = Store(tmp_path)
    sess, _, _ = _turn(s)
    real = Path.write_bytes

    def flaky(self, data):
        if self.name == "model_input.png":
            raise OSError(28, "No space left on device")
        return real(self, data)

    monkeypatch.setattr(Path, "write_bytes", flaky)
    with pytest.raises(OSError, match="No space"):
        s.save_upload(sess["id"], "ab" * 32, b"o", "png", b"t", b"m")
    assert s.upload_path(sess["id"], "ab" * 32, "original") is None
    assert list((tmp_path / "uploads").rglob("*")) == []
    monkeypatch.setattr(Path, "write_bytes", real)
    s.save_upload(sess["id"], "ab" * 32, b"o", "png", b"t", b"m")   # and a retry works
    assert s.upload_path(sess["id"], "ab" * 32, "original").read_bytes() == b"o"


def test_recovery_closes_the_log_of_an_interrupted_turn_and_is_idempotent(tmp_path):
    s = Store(tmp_path)
    sess, _, mid = _turn(s)
    s.append_event(mid, "stage_start", {"stage": "preprocess"})
    _, finished = s.start_turn(sess["id"], "again", "private", {})
    s.append_event(finished, "message_stop", {"status": "done", "report": "ok"})
    s.finish_turn(finished, "done", report="ok")
    s.close()
    s = Store(tmp_path)
    assert s.recover_after_restart() == 1
    tail = s.events_after(mid, 1)   # the log now ends the way a failed turn ends (error, then message_stop)
    assert [(e["seq"], e["event"]) for e in tail] == [(2, "error"), (3, "message_stop")]
    assert tail[0]["data"]["type"] == "error" and tail[0]["data"]["error"]["type"] == "server_restart"
    assert (tail[1]["data"]["status"], tail[1]["data"]["message_id"], tail[1]["data"]["report"]) == ("error", mid, None)
    assert s.get_message(mid, None)["status"] == "error" and s.get_message(finished, None)["status"] == "done"
    assert s.recover_after_restart() == 0 and len(s.events_after(mid)) == 3


def test_a_killed_process_leaves_a_log_the_next_start_recovers(tmp_path):   # P7-E, P8-D: requeue, kill -9
    code = ("import os, sys; from pathlib import Path; from app.store import Store; s = Store(Path(sys.argv[1])); "
            "sess = s.create_session('private'); u, m = s.start_turn(sess['id'], 'hi', 'private', {}); "
            "[s.append_event(m, 'stage_start', {'i': i}) for i in range(5)]; print(m, flush=True); os._exit(0)")
    # no close(): the WAL is not checkpointed and the exclusive lock dies with the process
    out = subprocess.run([sys.executable, "-c", code, str(tmp_path)], cwd=str(REPO_ROOT), capture_output=True,
                         text=True, check=True)
    mid = out.stdout.strip()
    s = Store(tmp_path)
    assert [e["data"]["i"] for e in s.events_after(mid)] == [0, 1, 2, 3, 4]
    assert s.recover_after_restart() == 1
    assert s.get_message(mid, None)["status"] == "error"
    assert [e["seq"] for e in s.events_after(mid, 5)] == [6, 7]


def test_recovery_survives_an_odd_stored_message_stop(tmp_path):
    s = Store(tmp_path)
    _, _, mid = _turn(s)
    s.append_event(mid, "message_stop", {"status": "weird", "report": {"x": 1}, "display_report": 3, "total_ms": "n/a"})
    assert s.recover_after_restart() == 1
    m = s.get_message(mid, None)
    assert (m["status"], m["report"], m["display_report"], m["total_ms"]) == ("error", None, None, None)


def test_recovery_trusts_a_message_stop_that_was_logged_before_the_crash(tmp_path):
    s = Store(tmp_path)
    _, _, mid = _turn(s)
    s.append_event(mid, "message_stop", {"status": "done", "report": "R", "display_report": "D", "total_ms": 5.0})
    s.close()   # died between the message_stop and finish_turn
    s = Store(tmp_path)
    assert s.recover_after_restart() == 1
    m = s.get_message(mid, None)
    assert (m["status"], m["report"], m["display_report"], m["total_ms"]) == ("done", "R", "D", 5.0)
    assert [e["event"] for e in s.events_after(mid)] == ["message_stop"]


def _stage(stage, detail):
    return {"stage": stage, "ms": 1.0, "detail": detail}


def test_markdown_export_has_the_turn_report_labels_neighbours_and_provenance(tmp_path):
    s = Store(tmp_path)
    sess = s.create_session("private")
    _, mid = s.start_turn(sess["id"], "a note", "private", {}, "ab" * 32, "chest.png")
    names = ["Label{}".format(i) for i in range(14)]
    s.append_event(mid, "stage_end", _stage("retrieve", {
        "image_neighbors": [{"rank": 1, "similarity": 0.91, "study_id": "S123"}],
        "report_matches": [{"rank": 1, "similarity": 0.4, "report": "Findings: NEIGHBOUR TEXT"}]}))
    s.append_event(mid, "stage_end", _stage("label", {"chexbert_14": {n: int(i == 0) for i, n in enumerate(names)}}))
    s.finish_turn(mid, "done", report="Findings: clear.", provenance={"model": "m3", "beam": 3}, total_ms=12.0)
    _, _, body = s.export(sess["id"], "md", None)
    md = body.decode("utf-8")
    assert md.startswith("# Session {}\n".format(sess["id"]))
    assert re.search(r"^## Turn 1 \u2014 \d{4}-\d\d-\d\d \d\d:\d\d:\d\d UTC$", md, re.M)
    assert "chest.png" in md and "a note" in md and "Findings: clear." in md
    assert sum(1 for line in md.splitlines() if line.startswith("| Label")) == 14 and "| Label0 | 1 |" in md
    assert "NEIGHBOUR TEXT" in md and "S123" in md
    assert [x for x in md.splitlines() if x.startswith("Provenance:")] == \
        ["Provenance: model=m3 \u00b7 beam=3 \u00b7 total_ms=12.0"]
    assert "not for clinical use" in md


def test_markdown_export_numbers_turns_and_marks_an_unfinished_one(tmp_path):
    s = Store(tmp_path)
    sess, _, first = _turn(s)
    s.finish_turn(first, "done", report="R1")
    _, second = s.start_turn(sess["id"], "again", "private", {}, test_row=9)
    md = s.export(sess["id"], "md", None)[2].decode("utf-8")
    assert md.index("## Turn 1 ") < md.index("## Turn 2 ") and "## Turn 3" not in md
    assert "test row 9" in md and "R1" in md and "running" in md
    assert not [x for x in md.splitlines() if x.startswith("| ")]   # no label stage, no table


def test_a_public_session_never_keeps_a_test_row_and_its_export_omits_neighbours(tmp_path):
    s = Store(tmp_path)   # redact.py (R1) keeps these out of a public log; the store does not rely on that alone
    sess = s.create_session("public", "a")
    user_id, mid = s.start_turn(sess["id"], "", "public", {}, test_row=3)
    assert s.get_message(user_id, "a")["test_row"] is None and s.last_image(sess["id"]) is None
    assert s.get_session(sess["id"], "a")["title"] == ""   # not "test row 3"
    s._con.execute("UPDATE messages SET test_row = 3 WHERE id = ?", (user_id,))   # as if a row slipped in
    s.append_event(mid, "stage_end", _stage("retrieve", {
        "image_neighbors": [{"rank": 1, "similarity": 0.9, "study_id": "S999"}],
        "report_matches": [{"rank": 1, "similarity": 0.4, "report": "Findings: SECRET"}]}))
    s.finish_turn(mid, "done", report="Findings: PUBLIC REPORT")
    md = s.export(sess["id"], "md", "a")[2].decode("utf-8")
    assert "PUBLIC REPORT" in md
    assert "SECRET" not in md and "S999" not in md and "test row" not in md


def test_export_is_scoped_validated_and_the_json_is_the_event_log(tmp_path):
    s = Store(tmp_path)
    sess, _, mid = _turn(s, "public", "a")
    s.append_event(mid, "stage_start", {"stage": "encode"})
    with pytest.raises(KeyError):
        s.export(sess["id"], "json", "b")
    with pytest.raises(ValueError):
        s.export(sess["id"], "pdf", "a")
    name, media, body = s.export(sess["id"], "json", "a")
    assert name == "session-{}.json".format(sess["id"]) and media == "application/json"
    doc = json.loads(body)
    assert doc["session"]["id"] == sess["id"] and "client_id" not in doc["session"]
    assert [m["role"] for m in doc["messages"]] == ["user", "assistant"] and doc["messages"][0]["events"] == []
    event = doc["messages"][1]["events"][0]
    assert set(event) == {"seq", "ts", "event", "data"} and event["data"] == {"stage": "encode", "seq": 1}
    assert doc["messages"][1]["options"] == {"beam_size": 3}
    assert s.export(sess["id"], "md", "a")[0] == "session-{}.md".format(sess["id"])
    s.delete_session(sess["id"], "a")
    with pytest.raises(KeyError):
        s.export(sess["id"], "md", "a")


def test_sweep_purges_rows_and_files_of_old_sessions_including_deleted_ones(tmp_path):
    s = Store(tmp_path)
    old, _, mid = _turn(s)
    s.append_event(mid, "stage_start", {})
    s.save_upload(old["id"], "ab" * 32, b"o", "png", b"t", b"m")
    s._con.execute("INSERT INTO artifacts (message_id, kind, path) VALUES (?, 'x', 'p')", (mid,))
    deleted, _, _ = _turn(s)
    s.delete_session(deleted["id"], None)
    keep, _, _ = _turn(s)
    for sid in (old["id"], deleted["id"]):
        s._con.execute("UPDATE sessions SET created_at='2000-01-01T00:00:00+00:00' WHERE id=?", (sid,))
    assert s.sweep(older_than_days=30) == 2
    counts = {t: s._con.execute("SELECT COUNT(*) FROM " + t).fetchone()[0]
              for t in ("sessions", "messages", "events", "artifacts")}
    assert counts == {"sessions": 1, "messages": 2, "events": 0, "artifacts": 0}
    assert not (tmp_path / "uploads" / old["id"]).exists()
    assert s.get_session(keep["id"], None) is not None
    assert s.sweep(older_than_days=30) == 0


def test_sweep_keys_on_creation_time_and_refuses_a_negative_age(tmp_path):
    s = Store(tmp_path)
    sess = s.create_session("private")
    s._con.execute("UPDATE sessions SET created_at=?, updated_at=? WHERE id=?",
                   ((datetime.now(timezone.utc) - timedelta(days=10)).isoformat(),
                    datetime.now(timezone.utc).isoformat(), sess["id"]))
    assert s.sweep(older_than_days=30) == 0
    with pytest.raises(ValueError):
        s.sweep(older_than_days=-1)   # a negative age would purge everything
    assert s.sweep(older_than_days=7) == 1   # a session still being used is deleted N days after it was created
