"""CHAT_UI_PLAN.md P9-C, part 2: `POST /v1/sessions/{id}/export/save`, a session's export written on the server.

The UI's export downloads to the browser, which is the laptop. For the supervisor demo the export of a session that holds MIMIC-derived text
(reports, matched reports, study ids) has to stay on the cluster (rules R1 and R7), so the server writes it itself, under `<CHAT_HOME>/exports/`,
and answers with a file name and a size and never the body. It is Store.save_export behind a private-mode-only route, and Store.save_export takes
the body from Store.export, the one export the GET route sends too (and so the held endings of the P8-D follow-up are in it). Synthetic data only:
the tiny engine, and a marker string put where a report would be.
"""
import json
import logging
import os
import re
import sqlite3
import stat

import pytest
from fastapi.testclient import TestClient

from app import server
from app.server import ERROR_KINDS, create_app
from app.store import Store
from tests.app_helpers import png_bytes

SECRET = "SECRET-export-body-text"
SAVE = "/v1/sessions/{}/export/save"
SAVED_NAME = re.compile(r"session-(s_[0-9a-f]{22})-[0-9]{8}T[0-9]{6}Z(-[0-9]+)?\.(json|md)")
PUBLIC = {"Authorization": "Bearer t", "X-Client-Id": "a"}


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(engine="tiny", home=str(tmp_path))) as c:
        yield c


def exports_dir(client):
    return client.app.state.store.home / "exports"


def make_session(c, title=SECRET, headers=None, text=SECRET + " note"):
    """A session with a title and one finished turn on the tiny engine, both holding the marker. -> its id."""
    sid = c.post("/v1/sessions", json={"title": title}, headers=headers).json()["id"]
    r = c.post("/v1/sessions/{}/messages".format(sid), files={"image": ("x.png", png_bytes(), "image/png")},
               data={"text": text, "options": json.dumps({"max_new_tokens": 16})}, headers=headers)
    assert r.status_code == 200 and '"status":"done"' in r.text, r.text[-300:]
    return sid


def saved_files(c):
    d = exports_dir(c)
    return sorted(p.name for p in d.iterdir()) if d.exists() else []


# ---- the route: what it writes and what it answers --------------------------------------------------------------------------------

def test_a_private_session_is_saved_as_json_and_as_markdown_with_the_bytes_the_export_route_sends(client):
    sid = make_session(client)
    for fmt, kind in (("json", "application/json"), ("md", "text/markdown")):
        sent = client.get("/v1/sessions/{}/export".format(sid), params={"format": fmt})
        assert sent.status_code == 200 and sent.headers["content-type"].startswith(kind)
        saved = client.post(SAVE.format(sid), params={"format": fmt})
        assert saved.status_code == 200, saved.text
        assert (exports_dir(client) / saved.json()["saved"]).read_bytes() == sent.content, fmt
    assert SECRET.encode() in sent.content, "the marker is in the body: the file holds what the download holds"


def test_format_is_json_by_default_as_for_the_export_route(client):
    sid = make_session(client)
    saved = client.post(SAVE.format(sid)).json()["saved"]
    assert saved.endswith(".json") and (exports_dir(client) / saved).read_bytes() == client.get("/v1/sessions/{}/export".format(sid)).content


def test_the_answer_is_the_file_name_and_its_size_and_nothing_of_the_body(client, caplog):
    sid = make_session(client)
    with caplog.at_level(logging.DEBUG):
        saved = client.post(SAVE.format(sid), params={"format": "md"})
    answer = saved.json()
    assert set(answer) == {"saved", "bytes"} and saved.status_code == 200
    assert type(answer["bytes"]) is int and answer["bytes"] == (exports_dir(client) / answer["saved"]).stat().st_size > 0
    assert SAVED_NAME.fullmatch(answer["saved"]) and answer["saved"].startswith("session-{}-".format(sid)) and answer["saved"].endswith(".md")
    assert SECRET not in saved.text and "Findings" not in saved.text and str(client.app.state.store.home) not in saved.text
    assert SECRET not in json.dumps(dict(saved.headers)), "no header carries it either"
    assert saved.headers["content-type"].startswith("application/json") and "content-disposition" not in saved.headers, "a JSON answer, no download"
    assert SECRET not in caplog.text and "Findings" not in caplog.text, "nor does the server's log"


def test_the_file_is_named_as_the_export_names_it_with_the_time_it_was_saved_in_utc(client, monkeypatch):
    from app import store as store_module
    monkeypatch.setattr(store_module, "_stamp", lambda: "20261013T181530Z")
    sid = make_session(client)
    for fmt in ("json", "md"):
        assert client.post(SAVE.format(sid), params={"format": fmt}).json()["saved"] == "session-{}-20261013T181530Z.{}".format(sid, fmt)
    assert saved_files(client) == ["session-{}-20261013T181530Z.json".format(sid), "session-{}-20261013T181530Z.md".format(sid)]


def test_the_stamp_is_the_utc_time_to_the_second(monkeypatch):
    from datetime import datetime, timezone
    from app import store as store_module
    before = datetime.now(timezone.utc).replace(microsecond=0)
    stamp = store_module._stamp()
    after = datetime.now(timezone.utc)
    when = datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    assert before <= when <= after and re.fullmatch(r"[0-9]{8}T[0-9]{6}Z", stamp)


def test_the_directory_is_owner_only_and_so_is_every_file_whatever_the_umask(client):
    sid = make_session(client)
    before = os.umask(0o022)   # a permissive umask must not widen what is written
    try:
        names = [client.post(SAVE.format(sid), params={"format": fmt}).json()["saved"] for fmt in ("json", "md")]
    finally:
        os.umask(before)
    assert stat.S_IMODE(exports_dir(client).stat().st_mode) == 0o700
    for name in names:
        assert stat.S_IMODE((exports_dir(client) / name).stat().st_mode) == 0o600, name


def test_an_exports_directory_that_was_made_wider_is_put_back_to_owner_only(client):
    sid = make_session(client)
    exports_dir(client).mkdir(mode=0o755)
    os.chmod(str(exports_dir(client)), 0o755)
    assert client.post(SAVE.format(sid)).status_code == 200
    assert stat.S_IMODE(exports_dir(client).stat().st_mode) == 0o700


def test_the_exports_directory_is_made_by_the_first_save_and_not_before(client):
    assert not exports_dir(client).exists(), "a server that saves nothing has no such directory"
    sid = make_session(client)
    assert not exports_dir(client).exists()
    client.post(SAVE.format(sid))
    assert exports_dir(client).is_dir()


def test_nothing_is_overwritten_two_saves_in_one_second_are_two_files(client, monkeypatch):
    from app import store as store_module
    monkeypatch.setattr(store_module, "_stamp", lambda: "20261013T181530Z")   # the clock stands still
    sid = make_session(client)
    first = client.post(SAVE.format(sid)).json()["saved"]
    (exports_dir(client) / first).write_bytes(b"KEEP: this is the first file, which a second save must not touch")
    second = client.post(SAVE.format(sid)).json()["saved"]
    third = client.post(SAVE.format(sid)).json()["saved"]
    assert len({first, second, third}) == 3 and SAVED_NAME.fullmatch(second) and SAVED_NAME.fullmatch(third)
    assert (exports_dir(client) / first).read_bytes() == b"KEEP: this is the first file, which a second save must not touch"
    assert (exports_dir(client) / second).read_bytes() == (exports_dir(client) / third).read_bytes() == client.get(
        "/v1/sessions/{}/export".format(sid)).content
    assert saved_files(client) == sorted([first, second, third]), "and no temporary file is left beside them"


def test_a_file_is_complete_when_it_has_its_name_a_temporary_one_never_has_a_name_a_reader_would_take(client, monkeypatch):
    """The body is written to a temporary file in the same directory (a hidden name) and put under its final name in one step: while the write is
    going on, the final name does not exist."""
    sid = make_session(client)
    seen = []
    real_fsync = os.fsync

    def watching(fd):
        seen.append(saved_files(client))   # the directory while the temporary file is written
        return real_fsync(fd)

    monkeypatch.setattr(os, "fsync", watching)
    name = client.post(SAVE.format(sid)).json()["saved"]
    assert seen and all(not SAVED_NAME.fullmatch(n) and n.startswith(".") and n.endswith(".tmp") for names in seen for n in names), seen
    assert saved_files(client) == [name]


# ---- who may save: private mode only, and whose session --------------------------------------------------------------------------

def test_public_mode_answers_403_before_anything_else_and_writes_nothing(tmp_path):
    with TestClient(create_app(engine="tiny", home=str(tmp_path), mode="public", token="t")) as c:
        sid = make_session(c, headers=PUBLIC)
        for url, headers in ((SAVE.format(sid), PUBLIC),                                          # a valid session of this client
                             (SAVE.format(sid) + "?format=md", PUBLIC),
                             (SAVE.format(sid), {"Authorization": "Bearer t"}),                    # no client id: the 403 comes first, not a 400
                             (SAVE.format(sid), dict(PUBLIC, **{"X-Client-Id": "b"})),             # another client's session: not a 404
                             (SAVE.format("s_missing"), PUBLIC),                                   # an unknown session
                             (SAVE.format(sid) + "?format=xml", PUBLIC)):                          # a bad format: not a 422
            r = c.post(url, headers=headers)
            assert r.status_code == 403 and r.json() == {"type": "error", "error": {"type": "permission_error", "message": server.PUBLIC_SAVE_MSG}}, (url, r.text)
        assert not (c.app.state.store.home / "exports").exists(), "nothing was written, not even the directory"
        assert c.post(SAVE.format(sid)).status_code == 401, "and the token is asked for first, as for every /v1 route"


def test_a_server_with_a_token_asks_for_it(tmp_path):
    with TestClient(create_app(engine="tiny", home=str(tmp_path), token="t0k")) as c:
        r = c.post(SAVE.format("s_x"))
        assert r.status_code == 401 and r.json()["error"]["type"] == "authentication_error"
        sid = c.post("/v1/sessions", json={}, headers={"Authorization": "Bearer t0k"}).json()["id"]
        assert c.post(SAVE.format(sid), headers={"Authorization": "Bearer t0k"}).status_code == 200


def test_an_unknown_session_is_a_404_in_the_envelope_and_writes_nothing(client):
    r = client.post(SAVE.format("s_missing"))
    assert r.status_code == 404 and r.json() == {"type": "error", "error": {"type": "not_found_error", "message": "Session not found."}}
    assert r.json() == client.get("/v1/sessions/s_missing/export").json(), "the export route's own answer"
    assert not exports_dir(client).exists()


def test_a_deleted_session_is_a_404_as_it_is_for_the_export_and_writes_nothing(client):
    sid = make_session(client)
    assert client.delete("/v1/sessions/{}".format(sid)).status_code == 204
    gone = client.get("/v1/sessions/{}/export".format(sid))
    assert gone.status_code == 404 and gone.json()["error"]["message"] == "Session not found."
    refused = client.post(SAVE.format(sid))
    assert refused.status_code == 404 and refused.json() == gone.json(), "the session's own refusal, not the 404 of a route that is not there"
    assert not exports_dir(client).exists()


@pytest.mark.parametrize("fmt", ["xml", "pdf", "JSON", "", "json,md", "md "])
def test_any_other_format_is_a_422_as_for_the_export_route_and_writes_nothing(fmt, client):
    sid = make_session(client)
    sent = client.get("/v1/sessions/{}/export".format(sid), params={"format": fmt})
    r = client.post(SAVE.format(sid), params={"format": fmt})
    assert r.status_code == 422 == sent.status_code and r.json() == sent.json()
    assert r.json()["error"] == {"type": "validation_error", "message": "format must be json or md."}
    assert not exports_dir(client).exists()


def test_a_server_that_is_not_private_never_calls_the_store(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(Store, "save_export", lambda *a, **k: calls.append(a))
    with TestClient(create_app(engine="tiny", home=str(tmp_path), mode="public", token="t")) as c:
        assert c.post(SAVE.format("s_x"), headers=PUBLIC).status_code == 403
    assert calls == []


# ---- failures: a fixed 500 that names no path -------------------------------------------------------------------------------------

def fixed_500():
    return {"type": "error", "error": {"type": "internal_error", "message": server.SAVE_EXPORT_FAILED_MSG}}


def boom_sqlite(*args, **kwargs):
    raise sqlite3.OperationalError("database is locked: /sc/home/SECRET-user/chat_sessions/chat.db")


def boom_no_space(*args, **kwargs):
    raise OSError(28, "No space left on device", "/sc/home/SECRET-user/chat_sessions/exports/x.tmp")


def boom_io(*args, **kwargs):
    raise OSError(5, "Input/output error", "/sc/home/SECRET-user/chat_sessions/exports/x.tmp")


FAILURES = [   # (what fails, how it is made to fail)
    ("the database", lambda c, m: m.setattr(c.app.state.store, "export", boom_sqlite)),
    ("the file cannot be put under its name", lambda c, m: m.setattr(os, "replace", boom_no_space)),
    ("the write is not flushed", lambda c, m: m.setattr(os, "fsync", boom_io)),
    ("the exports path is a file", lambda c, m: exports_dir(c).write_text("not a directory")),
    ("the exports directory cannot be written to", lambda c, m: (exports_dir(c).mkdir(mode=0o500), os.chmod(str(exports_dir(c)), 0o500))),
]


@pytest.mark.parametrize("what, fail", FAILURES, ids=[f[0] for f in FAILURES])
def test_a_failure_of_the_store_or_the_disk_is_a_fixed_500_that_names_no_path_and_leaves_no_file(what, fail, client, monkeypatch, caplog):
    if what == "the exports directory cannot be written to" and os.geteuid() == 0:
        pytest.skip("root writes to a directory of mode 0500")
    sid = make_session(client)
    fail(client, monkeypatch)
    with caplog.at_level(logging.DEBUG):
        r = client.post(SAVE.format(sid))
    assert r.status_code == 500 and r.json() == fixed_500(), (what, r.text)
    for text in (r.text, caplog.text):
        assert "SECRET" not in text and "/sc/home" not in text and str(client.app.state.store.home) not in text and "chat_sessions" not in text, (what, text)
    assert r.headers["x-request-id"], "the id is how the log line is found"
    directory = exports_dir(client)
    if directory.is_dir():
        os.chmod(str(directory), 0o700)   # the case that made it read-only: so that it can be listed, and removed with the rest
    leftovers = sorted(p.name for p in directory.iterdir()) if directory.is_dir() else []
    assert leftovers == [], (what, leftovers)


def test_the_log_of_a_failed_save_names_the_class_of_the_failure_and_nothing_else(client, monkeypatch, caplog):
    sid = make_session(client)
    monkeypatch.setattr(os, "replace", boom_no_space)
    with caplog.at_level(logging.WARNING, logger="app.server"):
        assert client.post(SAVE.format(sid)).status_code == 500
    lines = [r.getMessage() for r in caplog.records if r.name == "app.server"]
    assert lines == ["could not save an export: OSError"], lines


def test_the_save_still_works_after_a_failed_one(client, monkeypatch):
    sid = make_session(client)
    with monkeypatch.context() as m:
        m.setattr(os, "replace", boom_no_space)
        assert client.post(SAVE.format(sid)).status_code == 500
    ok = client.post(SAVE.format(sid))
    assert ok.status_code == 200 and saved_files(client) == [ok.json()["saved"]]


# ---- the store: one export, the same one ------------------------------------------------------------------------------------------

def test_the_store_saves_the_body_its_own_export_gives_and_never_builds_a_second(tmp_path, monkeypatch):
    store = Store(tmp_path)
    try:
        sid = store.create_session("private", None, SECRET)["id"]
        store.start_turn(sid, SECRET, "private", {}, None, None, None)
        calls = []
        real = Store.export

        def spy(self, session_id, fmt, client_id):
            calls.append((session_id, fmt, client_id))
            return real(self, session_id, fmt, client_id)

        monkeypatch.setattr(Store, "export", spy)
        name, size = store.save_export(sid, "json", None)
        assert calls == [(sid, "json", None)], "one call of export: the save has no export of its own"
        filename, _, body = real(store, sid, "json", None)
        assert name.startswith(filename[:-len(".json")] + "-") and name.endswith(".json") and size == len(body)
        assert (tmp_path / "exports" / name).read_bytes() == body
    finally:
        store.close()


def test_a_turn_whose_ending_the_database_could_not_store_is_in_the_saved_export_as_it_is_in_the_download(tmp_path):
    """P8-D follow-up: the store keeps in memory the ending it could not write, and every read, the export among them, gives it. The saved file is that
    same export, so the held error and the stop are in it, numbered on from the last stored event."""
    store = Store(tmp_path)
    try:
        sid = store.create_session("private", None, "held")["id"]
        _, mid = store.start_turn(sid, "note", "private", {}, None, None, None)
        store.append_event(mid, "message_start", {"message_id": mid})
        store.hold_ending(mid, [("error", {"type": "error", "error": {"type": "store_failed", "message": "held: SECRET"}}),
                                ("message_stop", {"message_id": mid, "status": "error", "report": None})], "error")
        for fmt in ("json", "md"):
            name, size = store.save_export(sid, fmt, None)
            body = (tmp_path / "exports" / name).read_bytes()
            assert body == store.export(sid, fmt, None)[2] and size == len(body)
        saved = json.loads((tmp_path / "exports" / store.save_export(sid, "json", None)[0]).read_text())
        turn = saved["messages"][1]
        assert turn["status"] == "error" and [e["event"] for e in turn["events"]] == ["message_start", "error", "message_stop"]
        assert [e["seq"] for e in turn["events"]] == [1, 2, 3]
        assert "Status: error" in (tmp_path / "exports" / store.save_export(sid, "md", None)[0]).read_text()
    finally:
        store.close()


def test_the_store_raises_what_export_raises_before_it_makes_a_directory_or_a_file(tmp_path):
    store = Store(tmp_path)
    try:
        sid = store.create_session("private", None, "x")["id"]
        with pytest.raises(KeyError):
            store.save_export("s_missing", "json", None)
        with pytest.raises(ValueError):
            store.save_export(sid, "xml", None)
        with pytest.raises(KeyError):   # hidden by scope: another client's session
            store.save_export(sid, "json", "someone-else")
        assert not (tmp_path / "exports").exists()
    finally:
        store.close()


def test_retention_deletes_a_saved_export_after_the_private_days_by_its_own_time_and_not_with_its_session(tmp_path, monkeypatch):
    """P9-C follow-up (user, 2026-10-14: "delete them after 45 days"), which this test first pinned the other way: that nothing deletes a saved export.
    The server's retention sweep removes an export 45 days after it was saved, by the time in its name. Store.sweep, the sweep of the sessions, does
    not, and a session that is purged leaves the exports it was saved as, until their own days are over."""
    from datetime import datetime, timedelta, timezone
    from app import store as store_module
    store = Store(tmp_path)
    try:
        sid = store.create_session("private", None, "x")["id"]
        with monkeypatch.context() as patched:   # one export is saved 46 days ago, one now
            patched.setattr(store_module, "_stamp", lambda: (datetime.now(timezone.utc) - timedelta(days=46)).strftime("%Y%m%dT%H%M%SZ"))
            old, _ = store.save_export(sid, "json", None)
        young, _ = store.save_export(sid, "json", None)
        assert store.sweep(0) == 1 and store.get_session(sid, None) is None, "the session is purged ..."
        assert (tmp_path / "exports" / old).is_file() and (tmp_path / "exports" / young).is_file(), "... and the sweep of the sessions leaves its exports"
    finally:
        store.close()
    with TestClient(create_app(engine="tiny", home=str(tmp_path))):   # the server's own sweep, run as it starts
        pass
    assert not (tmp_path / "exports" / old).exists(), "the export of 46 days ago is deleted"
    assert (tmp_path / "exports" / young).is_file(), "the one saved now is kept"


# ---- the reference ----------------------------------------------------------------------------------------------------------------

def test_the_route_is_in_the_reference_with_its_refusals_and_an_example_that_has_the_real_keys(client):
    spec = client.get("/openapi.json").json()
    op = spec["paths"]["/v1/sessions/{session_id}/export/save"]["post"]
    assert op["summary"] and "403" in op["description"] and "404" in op["description"] and "422" in op["description"] and "500" in op["description"]
    for word in ("never the export itself", "exports/", "private mode only", "never replac", "owner"):
        assert word in op["description"], word
    declared = {int(s) for s in op["responses"] if s != "default" and int(s) >= 400}
    assert declared == {403, 404, 413, 422, 500}
    example = op["responses"]["200"]["content"]["application/json"]["example"]
    sid = make_session(client)
    live = client.post(SAVE.format(sid)).json()
    assert set(example) == set(live) == {"saved", "bytes"} and type(example["bytes"]) is int and SAVED_NAME.fullmatch(example["saved"])
    for status in (403, 404, 422, 500):
        envelope = op["responses"][str(status)]["content"]["application/json"]["example"]
        assert envelope["error"]["type"] == ERROR_KINDS[status] and envelope["error"]["message"]
    assert op["responses"]["500"]["content"]["application/json"]["example"] == fixed_500()
    assert "400" not in op["responses"]["default"]["description"], "in public mode the 403 comes before any client id is asked for"
    export = spec["paths"]["/v1/sessions/{session_id}/export"]["get"]
    assert set(export["responses"]) >= {"200", "404", "422"}, "the download route is as it was"
