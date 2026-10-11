"""CHAT_UI_PLAN.md P8-A: request limits, request ids and the request log, the error envelope everywhere, and the public-mode turn rate limit.

The three tests of the brief come first, verbatim. Then, in the order of the rulings: the 413 before any parsing, the request id on every
response and the one-line JSON request log (stderr only; a query, a header or a body never reaches it), the envelope on every refusal (an
unknown route, a wrong method, a 422 that echoes nothing, a 500 that says nothing), and the token bucket that limits turns per client in
public mode. Synthetic data only (R7): the study-id-like numbers below are made up.
"""
import asyncio
import datetime
import hashlib
import json

import httpx
import pytest
from fastapi.testclient import TestClient
from starlette.responses import JSONResponse

from app import hardening, server
from app.imaging import MAX_UPLOAD_BYTES, TOO_LARGE_MSG
from app.schemas import error_body
from app.server import create_app
from tests.app_helpers import iter_sse, png_bytes, start_live_server, wait_until

CAP = 21 * 1024 * 1024   # what a request body may hold, before any parsing (ruling 1)
LOG_KEYS = ["ts", "request_id", "method", "path", "status", "ms", "client"]
PUBLIC = {"Authorization": "Bearer t", "X-Client-Id": "a"}


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(engine="tiny", home=str(tmp_path))) as c:
        yield c


def logged(err):
    """The request lines of a captured stderr, parsed: one JSON object per line, the other lines (uvicorn's, a library's) left out."""
    out = []
    for line in err.splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and "request_id" in row:
            out.append(row)
    return out


# ---- the brief ------------------------------------------------------------------------------------------------------------------------

def test_oversize_body_is_413_before_parsing(client):
    sid = client.post("/v1/sessions", json={}).json()["id"]
    r = client.post("/v1/sessions/{}/messages".format(sid), content=b"x" * (21 * 1024 * 1024 + 1),
                    headers={"Content-Type": "multipart/form-data; boundary=zzz"})
    assert r.status_code == 413 and r.json()["error"]["type"] == "validation_error"


def test_request_id_is_echoed_and_generated(client):
    assert client.get("/healthz", headers={"X-Request-Id": "abc"}).headers["X-Request-Id"] == "abc"
    assert len(client.get("/healthz").headers["X-Request-Id"]) >= 16


def test_public_rate_limit_is_per_client(tmp_path):
    with TestClient(create_app(engine="tiny", home=str(tmp_path), mode="public", token="t")) as c:
        h = {"Authorization": "Bearer t", "X-Client-Id": "a"}
        sid = c.post("/v1/sessions", json={}, headers=h).json()["id"]
        codes = [c.post("/v1/sessions/{}/messages".format(sid), headers=h,
                        files={"image": ("x.png", png_bytes(), "image/png")},
                        data={"text": "", "options": json.dumps({"max_new_tokens": 16})}).status_code for _ in range(7)]
        assert codes[:6] == [200] * 6 and codes[6] == 429
        other = dict(h, **{"X-Client-Id": "b"})
        sid_b = c.post("/v1/sessions", json={}, headers=other).json()["id"]
        assert c.post("/v1/sessions/{}/messages".format(sid_b), headers=other,
                      files={"image": ("x.png", png_bytes(), "image/png")},
                      data={"text": "", "options": json.dumps({"max_new_tokens": 16})}).status_code == 200


# ---- ruling 2: the request id on every response ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("given, kept", [
    ("abc", True), ("Req-1.2_3", True), ("0", True), ("x" * 64, True), ("A.b-C_d" * 9, True),   # 63 characters
    ("x" * 65, False), ("a b", False), ("a/b", False), ("a;b", False), ("<script>", False), (b"caf\xe9", False), ("a:b", False), ("a,b", False)])
def test_a_callers_request_id_is_kept_only_when_it_is_1_to_64_of_letters_digits_dot_underscore_and_dash(client, given, kept):
    r = client.get("/healthz", headers={"X-Request-Id": given})   # the bytes are what a client that sends latin-1 puts on the wire
    got = r.headers["X-Request-Id"]
    assert (got == given) is kept, got
    if not kept:   # a new id: a uuid4 in hex, never the caller's text
        assert len(got) == 32 and int(got, 16) >= 0 and got == got.lower()


def test_an_empty_request_id_is_replaced_and_two_requests_never_share_a_generated_one(client):
    empty = client.get("/healthz", headers={"X-Request-Id": ""}).headers["X-Request-Id"]
    assert len(empty) == 32
    ids = {client.get("/healthz").headers["X-Request-Id"] for _ in range(20)}
    assert len(ids) == 20


def _every_answer(tmp_path, monkeypatch):
    """One request of each kind of answer the server gives, as {name: response}: success, every refusal of the guard, the router and
    the validator, a 500, a streamed turn, a static file's 304 and a CORS preflight."""
    origin = "http://127.0.0.1:5173"
    answers = {}
    with TestClient(create_app(engine="tiny", home=str(tmp_path / "open"), cors_origins=(origin,)),
                    raise_server_exceptions=False) as c:
        sid = c.post("/v1/sessions", json={}).json()["id"]
        answers["200"] = c.get("/healthz")
        answers["404 route"] = c.get("/v1/nope")
        answers["404 session"] = c.get("/v1/sessions/s_missing")
        answers["405"] = c.delete("/healthz")
        answers["413"] = c.post("/v1/sessions/{}/messages".format(sid), content=b"x" * (CAP + 1),
                                headers={"Content-Type": "multipart/form-data; boundary=zzz"})
        answers["422"] = c.get("/v1/sessions?limit=0")
        answers["403"] = c.get("http://10.1.2.3:8000/healthz")
        answers["preflight"] = c.options("/v1/sessions", headers={"Origin": origin, "Access-Control-Request-Method": "POST"})
        answers["turn"] = c.post("/v1/sessions/{}/messages".format(sid), files={"image": ("x.png", png_bytes(), "image/png")},
                                 data={"options": json.dumps({"max_new_tokens": 16})})
        css = c.get("/static/styles.css")
        answers["304"] = c.get("/static/styles.css", headers={"If-None-Match": css.headers["etag"]})

        def boom(*args, **kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(c.app.state.store, "list_sessions", boom)
        answers["500"] = c.get("/v1/sessions")
    with TestClient(create_app(engine="tiny", home=str(tmp_path / "token"), token="t")) as c:
        answers["401"] = c.get("/v1/sessions")
    with TestClient(create_app(engine="tiny", home=str(tmp_path / "public"), mode="public", token="t")) as c:
        answers["400"] = c.get("/v1/sessions", headers={"Authorization": "Bearer t"})
    return answers


def test_every_response_carries_a_request_id_errors_included(tmp_path, monkeypatch):
    answers = _every_answer(tmp_path, monkeypatch)
    assert {k: v.status_code for k, v in answers.items()} == {
        "200": 200, "404 route": 404, "404 session": 404, "405": 405, "413": 413, "422": 422, "403": 403, "preflight": 200, "turn": 200,
        "304": 304, "500": 500, "401": 401, "400": 400}
    for name, r in answers.items():
        assert len(r.headers["X-Request-Id"]) == 32, name


def test_a_callers_request_id_comes_back_on_a_refusal_and_on_a_stream_too(tmp_path):
    with TestClient(create_app(engine="tiny", home=str(tmp_path), token="t")) as c:
        h = {"Authorization": "Bearer t", "X-Request-Id": "caller-7"}
        assert c.get("/v1/sessions", headers={"X-Request-Id": "caller-8"}).headers["X-Request-Id"] == "caller-8"   # a 401
        sid = c.post("/v1/sessions", json={}, headers=h).json()["id"]
        turn = c.post("/v1/sessions/{}/messages".format(sid), headers=h, files={"image": ("x.png", png_bytes(), "image/png")})
        assert turn.status_code == 200 and turn.headers["X-Request-Id"] == "caller-7"
        assert turn.headers["X-Message-Id"]   # the other custom header is untouched


def test_a_cross_origin_page_can_send_and_read_the_request_id(tmp_path):
    origin = "http://127.0.0.1:5173"
    with TestClient(create_app(engine="tiny", home=str(tmp_path), cors_origins=(origin,))) as c:
        r = c.options("/v1/sessions", headers={"Origin": origin, "Access-Control-Request-Method": "POST",
                                               "Access-Control-Request-Headers": "x-request-id"})
        assert r.status_code == 200 and "x-request-id" in r.headers["access-control-allow-headers"].lower()
        r = c.get("/healthz", headers={"Origin": origin})
        assert "x-request-id" in r.headers["access-control-expose-headers"].lower()


# ---- ruling 2: one JSON line per request, on stderr, and what never goes into it ------------------------------------------------------

def test_each_request_logs_one_json_line_with_seven_fields_on_stderr_and_none_on_stdout(tmp_path, capsys):
    with TestClient(create_app(engine="tiny", home=str(tmp_path))) as c:
        capsys.readouterr()   # the start-up lines
        before = datetime.datetime.now(datetime.timezone.utc)
        r = c.get("/healthz", headers={"X-Request-Id": "req-1", "X-Client-Id": "browser-7"})
        out, err = capsys.readouterr()
    assert out == ""   # P7-B's JobLogFilter keeps stdout for the [server] lines
    [row] = logged(err)
    assert list(row) == LOG_KEYS
    assert row["request_id"] == "req-1" == r.headers["X-Request-Id"]
    assert (row["method"], row["path"], row["status"], row["client"]) == ("GET", "/healthz", 200, "browser-7")
    assert isinstance(row["ms"], (int, float)) and not isinstance(row["ms"], bool) and 0 <= row["ms"] < 5000
    stamp = datetime.datetime.fromisoformat(row["ts"])
    assert stamp.utcoffset() == datetime.timedelta(0) and before - datetime.timedelta(seconds=1) <= stamp <= before + datetime.timedelta(seconds=30)
    assert [l for l in err.splitlines() if "request_id" in l] == [json.dumps(row, separators=(",", ":"))]   # compact, on one line


def test_a_request_without_a_well_formed_client_id_logs_a_dash(client, capsys):
    capsys.readouterr()
    for headers in ({}, {"X-Client-Id": ""}, {"X-Client-Id": "x" * 129}, {"X-Client-Id": "a b"}, {"X-Client-Id": "x" * 128}):
        client.get("/healthz", headers=headers)
    assert [row["client"] for row in logged(capsys.readouterr().err)] == ["-", "-", "-", "-", "x" * 128]


def test_the_line_has_the_status_the_client_got_for_every_kind_of_answer(tmp_path, monkeypatch, capsys):
    capsys.readouterr()
    answers = _every_answer(tmp_path, monkeypatch)
    rows = {row["request_id"]: row for row in logged(capsys.readouterr().err)}
    for name, r in answers.items():
        row = rows[r.headers["X-Request-Id"]]   # one line per request, found by the id the client saw
        assert row["status"] == r.status_code, name


def test_a_query_never_reaches_the_log(client, capsys):
    study = "50123456"   # a test-split study id looks like this, and the picker's q is the start of one
    capsys.readouterr()
    client.get("/v1/test-studies?q=" + study)
    client.get("/healthz?study_id={}&x=1".format(study))
    client.get("/v1/messages/m_x?after=3&secret=" + study)
    out, err = capsys.readouterr()
    assert study not in out and study not in err
    assert [row["path"] for row in logged(err)] == ["/v1/test-studies", "/healthz", "/v1/messages/m_x"]
    assert "?" not in err and "study_id" not in err


def test_no_header_but_the_client_id_and_no_body_token_or_authorization_reaches_the_log(tmp_path, capsys):
    with TestClient(create_app(engine="tiny", home=str(tmp_path), token="tok-SECRET-1")) as c:
        capsys.readouterr()
        c.post("/v1/sessions", json={"title": "BODY-MARKER-77"},
               headers={"Authorization": "Bearer tok-SECRET-1", "X-Other": "HEADER-MARKER-5", "Cookie": "c=COOKIE-9", "User-Agent": "UA-MARKER-3",
                        "Referer": "http://REFERER-MARKER-4/"})
        c.get("/v1/sessions", headers={"Authorization": "Bearer tok-WRONG-2"})   # a refused one
        out, err = capsys.readouterr()
    assert len(logged(err)) == 2
    for marker in ("tok-SECRET-1", "tok-WRONG-2", "BODY-MARKER-77", "HEADER-MARKER-5", "COOKIE-9", "UA-MARKER-3", "REFERER-MARKER-4",
                   "Bearer", "Authorization"):
        assert marker not in err and marker not in out, marker


def test_a_streamed_turn_logs_one_line_when_its_stream_has_ended(tmp_path, capsys):
    with TestClient(create_app(engine="tiny", home=str(tmp_path), tiny_step_delay_s=0.01)) as c:
        sid = c.post("/v1/sessions", json={}).json()["id"]
        capsys.readouterr()
        r = c.post("/v1/sessions/{}/messages".format(sid), files={"image": ("x.png", png_bytes(), "image/png")},
                   data={"options": json.dumps({"max_new_tokens": 16})}, headers={"X-Client-Id": "c1"})
        rows = logged(capsys.readouterr().err)
    [row] = rows
    assert row["request_id"] == r.headers["X-Request-Id"] and row["status"] == 200 and row["method"] == "POST"
    assert row["path"] == "/v1/sessions/{}/messages".format(sid) and row["ms"] >= 100   # 16 steps of 10 ms: the line is the whole stream's


def test_a_unicode_or_long_path_is_one_line_of_json(client, capsys):
    capsys.readouterr()
    client.get("/v1/é\u2028" + "z" * 500)
    err = capsys.readouterr().err
    [row] = logged(err)
    assert row["status"] == 404 and row["path"].startswith("/v1/é\u2028z") and len(row["path"]) <= 256
    assert len([l for l in err.splitlines() if l.strip()]) == 1   # U+2028 is escaped: no second line


def test_a_live_server_writes_its_request_lines_to_stderr_and_leaves_the_job_log_to_its_four_shapes(tmp_path):
    """The real CLI as a child process (P7-B's filter holds its stdout): the lines are on stderr, which the wrapper keeps in a file, and a query's
    study-id-like number is on neither stream."""
    from tests.test_chat_server_cli import JOB_LOG_LINE, LiveCli
    study = "50987654"
    with LiveCli(tmp_path, "--mode", "private") as live:
        base = live.base()
        refused = httpx.get(base + "/v1/test-studies?q=" + study, headers={"X-Request-Id": "live-1", "X-Client-Id": "c1"})   # 503: no gallery
        health = httpx.get(base + "/healthz")
        assert live.stop() == 0
    out, err = live.out.read_text(), live.err.read_text()
    assert refused.status_code == 503 and refused.headers["x-request-id"] == "live-1" and len(health.headers["x-request-id"]) == 32
    assert out.splitlines() and not [l for l in out.splitlines() if not JOB_LOG_LINE.match(l)], out   # stdout: the job log's lines only
    assert "request_id" not in out and study not in out and study not in err
    rows = {row["request_id"]: row for row in logged(err)}
    assert (rows["live-1"]["method"], rows["live-1"]["path"], rows["live-1"]["status"], rows["live-1"]["client"]) == (
        "GET", "/v1/test-studies", 503, "c1")
    assert rows[health.headers["x-request-id"]]["path"] == "/healthz" and rows[health.headers["x-request-id"]]["client"] == "-"
    assert "uvicorn" not in " ".join(r["path"] for r in rows.values())   # and uvicorn's own access log (it prints the query) stays off


# ---- ruling 3: the error envelope everywhere ----------------------------------------------------------------------------------------------

def test_an_unknown_route_and_a_wrong_method_answer_in_the_envelope(client):
    r = client.get("/v1/nope")
    assert r.status_code == 404 and r.json() == error_body("not_found_error", "Not Found")
    r = client.get("/nope.html")
    assert r.status_code == 404 and r.json()["type"] == "error" and r.headers["content-type"].startswith("application/json")
    r = client.delete("/healthz")
    assert r.status_code == 405 and r.json()["type"] == "error" and r.json()["error"]["type"] == server.ERROR_KINDS[405]
    assert "GET" in r.headers["allow"] and r.json()["error"]["message"] == "Method Not Allowed"


def test_a_422_never_echoes_what_was_sent(client):
    marker = "ECHO-MARKER-31337"
    for r in (client.get("/v1/sessions?limit=" + marker),
              client.post("/v1/sessions", json={"title": {"k": marker}}),
              client.get("/v1/messages/m_x?after=" + marker),
              client.post("/v1/label", json={"text": ["a", marker]})):
        assert r.status_code == 422 and r.json()["error"]["type"] == "validation_error"
        assert marker not in r.text
        assert set(r.json()) == {"type", "error"} and set(r.json()["error"]) == {"type", "message"}   # no `detail`, no `input`


def test_an_unhandled_exception_is_a_500_in_the_envelope_with_neither_traceback_nor_exception_text(tmp_path, monkeypatch, capsys):
    with TestClient(create_app(engine="tiny", home=str(tmp_path)), raise_server_exceptions=False) as c:
        def boom(*args, **kwargs):
            raise RuntimeError("SECRET-detail /sc/home/someone/mimic/p10/s5012")

        monkeypatch.setattr(c.app.state.store, "list_sessions", boom)
        capsys.readouterr()
        r = c.get("/v1/sessions", headers={"X-Request-Id": "rid-500", "X-Client-Id": "c9"})
        err = capsys.readouterr().err
    assert r.status_code == 500 and r.headers["content-type"].startswith("application/json")
    assert r.json() == error_body("internal_error", server.INTERNAL_ERROR_MSG)
    assert r.headers["X-Request-Id"] == "rid-500"
    for needle in ("SECRET", "RuntimeError", "Traceback", "mimic", "boom", "File "):
        assert needle not in r.text, needle
    [row] = [row for row in logged(err) if row["request_id"] == "rid-500"]
    assert row["status"] == 500 and row["path"] == "/v1/sessions" and row["client"] == "c9"


# ---- ruling 1: 413 before any parsing -------------------------------------------------------------------------------------------------

def test_the_cap_is_21_mb_a_megabyte_above_the_image_limit():
    assert server.MAX_REQUEST_BYTES == CAP == MAX_UPLOAD_BYTES + 1024 * 1024


def _run_asgi(app, method, path, headers, chunks, query=b"", gone=None):
    """Call the app's ASGI interface directly with a body that arrives as `chunks`; -> (the messages sent, how many chunks were taken).
    The scope is what uvicorn builds, so the request goes through every middleware and no network. The client leaves with its last chunk:
    a receive after it says http.disconnect. gone (P6 fix 2) keeps it there until gone() says it has left: such a receive waits until then
    (is_disconnected() does not wait, and reads a client still there)."""
    sent, taken = [], []
    pending = list(chunks)

    async def receive():
        if not pending:
            while gone is not None and not gone():
                await asyncio.sleep(0.005)
            return {"type": "http.disconnect"}
        taken.append(1)
        body = pending.pop(0)
        return {"type": "http.request", "body": body, "more_body": bool(pending)}

    async def send(message):
        sent.append(message)

    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": method, "scheme": "http", "path": path,
             "raw_path": path.encode(), "query_string": query, "root_path": "", "headers": [(k.lower().encode(), v.encode()) for k, v in headers],
             "client": ("127.0.0.1", 50000), "server": ("127.0.0.1", 8000), "state": {}}
    asyncio.run(app(scope, receive, send))
    return sent, len(taken)


def _body_of(sent):
    return json.loads(b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body"))


MULTIPART = ("Content-Type", "multipart/form-data; boundary=zzz")


def test_a_declared_length_over_the_cap_is_answered_without_one_byte_of_the_body_being_read(tmp_path):
    app = create_app(engine="tiny", home=str(tmp_path))
    try:
        for declared in (str(CAP + 1), "1000000000000", "9" * 5000):   # the last has more digits than int() takes from a string on Python 3.11+
            sent, taken = _run_asgi(app, "POST", "/v1/sessions/s_x/messages", [MULTIPART, ("Content-Length", declared)], [b"x" * 10])
            assert sent[0]["status"] == 413 and taken == 0, declared
            assert _body_of(sent) == error_body("validation_error", TOO_LARGE_MSG)
    finally:
        app.state.worker.shutdown()
        app.state.store.close()


def test_a_chunked_body_is_counted_as_it_streams_and_the_chunks_after_the_cap_are_never_read(tmp_path):
    app = create_app(engine="tiny", home=str(tmp_path))
    try:
        mb = 1024 * 1024
        chunks = [b'--zzz\r\nContent-Disposition: form-data; name="image"; filename="x.png"\r\n\r\n'] + [bytes(mb)] * 60   # no Content-Length
        sent, taken = _run_asgi(app, "POST", "/v1/sessions/s_x/messages", [MULTIPART], chunks)
        assert sent[0]["status"] == 413 and _body_of(sent) == error_body("validation_error", TOO_LARGE_MSG)
        assert 20 <= taken <= 24, "the 413 came at the cap, not after all 61 chunks: {} were taken".format(taken)
    finally:
        app.state.worker.shutdown()
        app.state.store.close()


def _form(boundary, fields, files):
    """A multipart/form-data body, as a browser's FormData sends it: each field, then each file (name -> (filename, bytes, content type))."""
    parts = [b'--%s\r\nContent-Disposition: form-data; name="%s"\r\n\r\n%s\r\n' % (boundary.encode(), name.encode(), value.encode())
             for name, value in fields.items()]
    parts += [b'--%s\r\nContent-Disposition: form-data; name="%s"; filename="%s"\r\nContent-Type: %s\r\n\r\n' % (
        boundary.encode(), name.encode(), filename.encode(), ctype.encode()) + data + b"\r\n" for name, (filename, data, ctype) in files.items()]
    return b"".join(parts) + b"--%s--\r\n" % boundary.encode()


def test_a_turn_whose_client_is_gone_once_its_body_is_read_starts_nothing_and_is_logged_as_499(tmp_path, capsys):
    """P6 fix 1 (defence in depth): the page's Cancel upload aborts the request, and the server would still start the turn from the whole body it
    has. The turn route asks whether the client is still there once the body is read and checked, before it reserves a slot or stores anything:
    a client already gone starts no turn, holds no slot and leaves no message and no upload, and the request log shows 499. _run_asgi's receive
    says http.disconnect after the last chunk, as uvicorn's does for a client that has closed its connection (a client still there, as every
    TestClient turn of this suite is, starts its turn as before)."""
    app = create_app(engine="tiny", home=str(tmp_path))
    try:
        sid = app.state.store.create_session("private")["id"]
        body = _form("zzz", {"text": "beam 2", "options": json.dumps({"max_new_tokens": 16})}, {"image": ("x.png", png_bytes(), "image/png")})
        capsys.readouterr()
        sent, _ = _run_asgi(app, "POST", "/v1/sessions/{}/messages".format(sid), [MULTIPART, ("X-Request-Id", "rid-gone")], [body])
        assert sent[0]["status"] == 499
        assert app.state.store.get_session(sid, None)["messages"] == []   # no turn was started, nothing stored
        assert app.state.worker.in_flight == 0                            # no slot is held
        assert not (tmp_path / "uploads" / sid).exists()                  # and no upload kept
        [row] = [row for row in logged(capsys.readouterr().err) if row["request_id"] == "rid-gone"]
        assert row["status"] == 499 and row["path"] == "/v1/sessions/{}/messages".format(sid)
    finally:
        app.state.worker.shutdown()
        app.state.store.close()


def test_a_client_gone_while_its_upload_is_saved_starts_no_turn_and_takes_back_only_the_copy_it_wrote(tmp_path, capsys, monkeypatch):
    """P6 fix 2: the save of a large upload takes a while (a 7.8 MP X-ray; longer on the cluster's NFS), and a Cancel upload that lands during
    it comes after the first check. The route asks again right before it records the turn: a client gone by then starts no turn, stores no
    message, gives its slot back, and the request log shows 499. The upload directory this request wrote goes too; a copy of the same bytes
    that an earlier turn of the chat left (uploads are kept once per chat and hash) stays as it was."""
    app = create_app(engine="tiny", home=str(tmp_path))
    left = {"now": False}
    save = server._save_upload

    def saved_then_gone(*args):   # the abort lands while the upload is being saved: after the first check, before the turn is recorded
        wrote = save(*args)
        left["now"] = True
        return wrote

    monkeypatch.setattr(server, "_save_upload", saved_then_gone)
    try:
        store = app.state.store
        fresh, kept = store.create_session("private")["id"], store.create_session("private")["id"]
        data = png_bytes()
        sha = hashlib.sha256(data).hexdigest()
        store.save_upload(kept, sha, data, "png", b"earlier thumb", b"earlier model input")   # an earlier turn's copy of the same bytes
        body = _form("zzz", {"text": "beam 2", "options": json.dumps({"max_new_tokens": 16})}, {"image": ("x.png", data, "image/png")})
        capsys.readouterr()
        for sid in (fresh, kept):
            left["now"] = False
            sent, _ = _run_asgi(app, "POST", "/v1/sessions/{}/messages".format(sid), [MULTIPART, ("X-Request-Id", "rid-late-" + sid)], [body],
                                gone=lambda: left["now"])
            assert sent[0]["status"] == 499, sid
            assert store.get_session(sid, None)["messages"] == [], sid   # no turn was recorded
            assert app.state.worker.in_flight == 0, sid                  # the slot it had reserved is free again
        assert not (tmp_path / "uploads" / fresh).exists()               # the copy this request wrote, and the chat's directory, are gone
        assert (tmp_path / "uploads" / kept / sha / "thumb.jpg").read_bytes() == b"earlier thumb"   # the earlier copy is untouched
        assert sorted(p.name for p in (tmp_path / "uploads" / kept / sha).iterdir()) == ["model_input.png", "original.png", "thumb.jpg"]
        rows = [row for row in logged(capsys.readouterr().err) if row["request_id"].startswith("rid-late-")]
        assert [row["status"] for row in rows] == [499, 499]
    finally:
        app.state.worker.shutdown()
        app.state.store.close()


def test_a_body_exactly_at_the_cap_is_not_refused_by_the_cap(client):
    sid = client.post("/v1/sessions", json={}).json()["id"]
    r = client.post("/v1/sessions/{}/messages".format(sid), content=b"x" * CAP, headers={"Content-Type": "multipart/form-data; boundary=zzz"})
    assert r.status_code != 413   # it is refused for what it is (a body that is no multipart form), not for its size


def test_a_413_is_answered_to_any_route_not_only_to_the_turn_route(client):
    for method, path in [("POST", "/v1/sessions"), ("POST", "/v1/label"), ("GET", "/healthz"), ("POST", "/v1/retrieve")]:
        r = client.request(method, path, content=b"x" * (CAP + 1))
        # P8 follow-up (6): a route that takes an image says the image is too large; the others, that the request is
        want = TOO_LARGE_MSG if (method, path) == ("POST", "/v1/retrieve") else server.TOO_LARGE_REQUEST_MSG
        assert r.status_code == 413 and r.json() == error_body("validation_error", want), (method, path)
        assert len(r.headers["X-Request-Id"]) == 32


def test_the_413_of_a_request_over_the_cap_names_the_request_on_a_route_with_no_image_and_the_image_on_the_two_that_take_one(client):
    """P8 follow-up (6): "Image is larger than 20 MB." was the answer to a 22 MB JSON body for /v1/label too. A declared length and a body
    counted as it streams are answered alike, by route: the turn route and /v1/retrieve keep the image message."""
    assert server.TOO_LARGE_REQUEST_MSG == "The request is larger than {} MB.".format(CAP // (1024 * 1024)) and "image" not in server.TOO_LARGE_REQUEST_MSG.lower()
    sid = client.post("/v1/sessions", json={}).json()["id"]
    for method, path, message in [("POST", "/v1/sessions", server.TOO_LARGE_REQUEST_MSG), ("DELETE", "/v1/sessions/s_x", server.TOO_LARGE_REQUEST_MSG),
                                  ("GET", "/v1/models", server.TOO_LARGE_REQUEST_MSG), ("POST", "/v1/label", server.TOO_LARGE_REQUEST_MSG),
                                  ("GET", "/v1/gallery/images/0", server.TOO_LARGE_REQUEST_MSG), ("GET", "/nowhere", server.TOO_LARGE_REQUEST_MSG),
                                  ("POST", "/v1/sessions/{}/messages".format(sid), TOO_LARGE_MSG), ("POST", "/v1/retrieve", TOO_LARGE_MSG),
                                  ("GET", "/v1/sessions/{}/messages".format(sid), server.TOO_LARGE_REQUEST_MSG)]:   # only a POST takes an image
        r = client.request(method, path, content=b"x" * (CAP + 1))
        assert r.status_code == 413 and r.json() == error_body("validation_error", message), (method, path)

    def chunked():   # a body with no Content-Length, counted as it arrives
        yield b"x" * 1024
        yield bytes(CAP)

    for path, message in [("/v1/sessions", server.TOO_LARGE_REQUEST_MSG), ("/v1/label", server.TOO_LARGE_REQUEST_MSG),
                          ("/v1/sessions/{}/messages".format(sid), TOO_LARGE_MSG), ("/v1/retrieve", TOO_LARGE_MSG)]:
        r = client.post(path, content=chunked(), headers={"Content-Type": "multipart/form-data; boundary=zzz"})
        assert "content-length" not in r.request.headers
        assert r.status_code == 413 and r.json() == error_body("validation_error", message), path


# ---- ruling 5: the page and its files are revalidated, not fetched again -------------------------------------------------------------------

def test_the_page_and_every_static_file_carry_an_etag_and_a_last_modified_and_a_conditional_request_gets_a_304(client):
    files = sorted(p.name for p in server.STATIC_DIR.iterdir() if p.is_file())
    assert {"index.html", "app.js", "api.js", "render.js", "state.js", "styles.css"} <= set(files)
    for path in ["/"] + ["/static/" + name for name in files]:
        first = client.get(path)
        assert first.status_code == 200 and first.headers["etag"] and first.headers["last-modified"], path
        assert first.headers["cache-control"] == "no-cache", path
        by_etag = client.get(path, headers={"If-None-Match": first.headers["etag"]})
        assert by_etag.status_code == 304 and by_etag.content == b"" and by_etag.headers["etag"] == first.headers["etag"], path
        by_date = client.get(path, headers={"If-Modified-Since": first.headers["last-modified"]})
        assert by_date.status_code == 304 and by_date.content == b"", path
        stale = client.get(path, headers={"If-None-Match": '"not-the-etag"'})   # a file that changed since is sent whole
        assert stale.status_code == 200 and stale.content == first.content, path
        assert by_etag.headers["cache-control"] == "no-cache", path   # the 304 keeps the page revalidating, as the 200 does


# ---- ruling 4: the public-mode turn rate limit ------------------------------------------------------------------------------------------

class Clock:
    """A clock a test moves by hand: the limiter reads time through it, so no test waits for ten minutes."""

    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def test_a_bucket_of_6_refills_6_per_600_seconds_one_token_per_100():
    clock = Clock()
    limiter = hardening.TurnLimiter(6, 600.0, clock=clock)
    assert [limiter.take("a") for _ in range(6)] == [0.0] * 6   # 0.0: a token was taken
    assert limiter.take("a") == pytest.approx(100.0)            # else: the seconds until one is free
    clock.advance(99)
    assert limiter.take("a") == pytest.approx(1.0)
    clock.advance(1)
    assert limiter.take("a") == 0.0
    assert limiter.take("a") == pytest.approx(100.0)
    clock.advance(10 ** 6)   # a long pause fills the bucket to its capacity and no further
    assert [limiter.take("a") for _ in range(6)] == [0.0] * 6 and limiter.take("a") > 0


def test_every_client_has_its_own_bucket():
    limiter = hardening.TurnLimiter(6, 600.0, clock=Clock())
    assert [limiter.take("a") for _ in range(6)] == [0.0] * 6 and limiter.take("a") > 0
    assert limiter.take("b") == 0.0 and limiter.take("A") == 0.0   # equality only: no folding of case


def test_a_token_that_is_given_back_can_be_taken_again_but_a_bucket_never_holds_more_than_its_capacity():
    limiter = hardening.TurnLimiter(6, 600.0, clock=Clock())
    limiter.take("a")
    limiter.refund("a")
    assert [limiter.take("a") for _ in range(6)] == [0.0] * 6 and limiter.take("a") > 0   # six, not seven
    limiter.refund("a")
    assert limiter.take("a") == 0.0 and limiter.take("a") > 0
    limiter.refund("never-seen")   # a refund for a client the table forgot is nothing to do


def test_a_client_whose_bucket_is_full_is_not_kept_in_the_table():
    clock = Clock()
    limiter = hardening.TurnLimiter(6, 600.0, clock=clock, max_tracked=100)
    limiter.take("a")
    assert limiter.tracked() == 1
    limiter.refund("a")
    assert limiter.tracked() == 0   # back at capacity: the same as a client never seen
    for i in range(500):   # ids minted one after another, each used once: the table forgets those that refilled
        limiter.take("minted-{}".format(i))
        clock.advance(601)
    assert limiter.tracked() <= 101


def test_the_limiter_rejects_a_capacity_or_window_that_is_no_limit():
    for capacity, window in [(0, 600.0), (-1, 600.0), (6, 0.0), (6, -5.0)]:
        with pytest.raises(ValueError):
            hardening.TurnLimiter(capacity, window)


@pytest.fixture
def public(tmp_path):
    """A public-mode app with the limiter's clock in the test's hands."""
    with TestClient(create_app(engine="tiny", home=str(tmp_path), mode="public", token="t")) as c:
        c.clock = Clock()
        c.app.state.rate_limiter.clock = c.clock
        yield c


def _turn_request(c, sid, headers=PUBLIC, **kwargs):
    kwargs.setdefault("files", {"image": ("x.png", png_bytes(), "image/png")})
    kwargs.setdefault("data", {"text": "", "options": json.dumps({"max_new_tokens": 16})})
    return c.post("/v1/sessions/{}/messages".format(sid), headers=headers, **kwargs)


def _session(c, headers=PUBLIC):
    return c.post("/v1/sessions", json={}, headers=headers).json()["id"]


def test_the_7th_turn_in_ten_minutes_is_a_429_rate_limited_with_retry_after_in_whole_seconds(public):
    sid = _session(public)
    assert [_turn_request(public, sid).status_code for _ in range(6)] == [200] * 6
    wait_until(lambda: public.app.state.worker.in_flight == 0)   # each turn has settled: the queue is not what answers below
    r = _turn_request(public, sid)
    assert r.status_code == 429 and r.headers["Retry-After"] == "100"
    assert r.json() == error_body("rate_limited", hardening.RATE_LIMITED_MSG.format(6, 10, 100))
    assert len(r.headers["X-Request-Id"]) == 32
    public.clock.advance(99)
    assert _turn_request(public, sid).headers["Retry-After"] == "1"   # whole seconds, rounded up: 0.99 tokens are 1 s short
    public.clock.advance(1)
    assert _turn_request(public, sid).status_code == 200              # one token is back
    assert _turn_request(public, sid).headers["Retry-After"] == "100"


def test_a_429_for_the_rate_is_not_the_429_for_a_full_queue(public, monkeypatch):
    sid = _session(public)
    with monkeypatch.context() as m:
        m.setattr(public.app.state.worker, "reserve", lambda: False)
        busy = _turn_request(public, sid)
    assert busy.status_code == 429 and busy.json()["error"]["type"] == "overloaded_error" and "Retry-After" not in busy.headers


def test_the_limit_is_asked_before_the_body_is_parsed(public):
    garbage = {"content": b"this is no multipart form", "headers": dict(PUBLIC, **{"Content-Type": "multipart/form-data; boundary=zzz"})}
    fresh = public.post("/v1/sessions/s_x/messages", **dict(garbage, headers=dict(garbage["headers"], **{"X-Client-Id": "fresh"})))
    assert fresh.status_code in (400, 422), "parsing is what refuses this body for a client with tokens left"
    sid = _session(public)
    for _ in range(6):
        assert _turn_request(public, sid).status_code == 200
    spent = public.post("/v1/sessions/s_x/messages", **garbage)
    assert spent.status_code == 429 and spent.json()["error"]["type"] == "rate_limited"   # not parsed, not even looked up


def test_only_a_turn_that_is_accepted_takes_a_token(public, monkeypatch):
    sid = _session(public)
    options = {"options": json.dumps({"max_new_tokens": 16})}
    refused = [
        public.post("/v1/sessions/s_missing/messages", headers=PUBLIC, files={"image": ("x.png", png_bytes(), "image/png")}).status_code,   # 404
        _turn_request(public, sid, data={"options": "[1, 2]"}).status_code,                                                               # 400
        _turn_request(public, sid, data={"options": json.dumps({"beam_size": 99})}).status_code,                                          # 422
        _turn_request(public, sid, files={"image": ("x.png", png_bytes(32, 32), "image/png")}).status_code,                                # 422
        _turn_request(public, sid, files={"image": ("x.png", b"not an image", "image/png")}).status_code,                                  # 422
        _turn_request(public, sid, files={}, data=dict(options, text="no image yet")).status_code,                                         # 422
    ]
    with monkeypatch.context() as m:
        m.setattr(public.app.state.worker, "reserve", lambda: False)
        refused.append(_turn_request(public, sid).status_code)                                                                              # 429: busy
    with monkeypatch.context() as m:
        m.setattr(public.app.state.store, "save_upload", lambda *a: (_ for _ in ()).throw(OSError(28, "no space")))
        refused.append(_turn_request(public, sid).status_code)                                                                              # 500
    assert refused == [404, 400, 422, 422, 422, 422, 429, 500]
    assert public.app.state.rate_limiter.tracked() == 0, "every one of them gave its token back"
    assert [_turn_request(public, sid).status_code for _ in range(6)] == [200] * 6   # all six are still there
    assert _turn_request(public, sid).status_code == 429


def _note(c, sid, text="Is there pneumonia?", headers=PUBLIC):
    """A text-only turn: no image, and `text` as the note."""
    return _turn_request(c, sid, headers=headers, files={}, data={"text": text, "options": json.dumps({"max_new_tokens": 16})})


def _stages_run(r):
    """The stages a turn's stream says ran (a stage_start each), and its warning codes."""
    frames = list(iter_sse([r.text]))
    return [f["data"]["stage"] for f in frames if f["event"] == "stage_start"], [f["data"]["code"] for f in frames if f["event"] == "warning"]


def test_a_chat_note_that_runs_no_model_takes_no_token_one_image_turn_and_six_notes_leave_five_image_turns(public):
    """P8 follow-up (ruling 3): a public-mode text-only turn that is answered with a warning and runs no model (a question, or a note that is no
    command) must not spend a token, as a refusal does not: the limiter takes the token at the check, before the body is read, and the route
    marks a turn that ran no model so that the limiter gives it back. One image turn and six notes: the six notes cost nothing, so five image
    turns are still accepted (the bucket holds six), and the sixth of those is the seventh token."""
    sid = _session(public)
    assert _turn_request(public, sid).status_code == 200   # the image turn: one token
    wait_until(lambda: public.app.state.worker.in_flight == 0)
    notes = [_note(public, sid) for _ in range(6)]
    assert [r.status_code for r in notes] == [200] * 6, "a sixth note would be the seventh token if each cost one"
    assert [_stages_run(r) for r in notes] == [([], ["not_a_command"])] * 6   # no stage ran: nothing used the model
    wait_until(lambda: public.app.state.worker.in_flight == 0)
    assert [_turn_request(public, sid).status_code for _ in range(5)] == [200] * 5   # the image turn after them is accepted, and four more
    wait_until(lambda: public.app.state.worker.in_flight == 0)
    spent = _turn_request(public, sid)
    assert spent.status_code == 429 and spent.json()["error"]["type"] == "rate_limited"   # six image turns in all: the notes were free
    assert spent.headers["Retry-After"] == "100"


def test_a_text_only_turn_that_runs_the_model_still_takes_its_token(public):
    """A command, or no text at all, re-runs the session's last image: that runs the model, so it is a turn like any other."""
    sid = _session(public)
    assert _turn_request(public, sid).status_code == 200
    wait_until(lambda: public.app.state.worker.in_flight == 0)
    reruns = [_note(public, sid, text) for text in ("beam 2", "", "tokens 20", "greedy", "retrieve 2")]
    assert [r.status_code for r in reruns] == [200] * 5
    assert all("generate" in _stages_run(r)[0] for r in reruns)   # each one ran the model
    wait_until(lambda: public.app.state.worker.in_flight == 0)
    assert _note(public, sid, "beam 2").status_code == 429   # six turns in all, so the seventh is refused
    assert _note(public, sid).status_code == 429             # and a note is refused too while the bucket is empty: the limiter asks before the body is read


def test_a_note_in_a_session_with_no_image_is_a_refusal_and_costs_nothing_as_before(public):
    sid = _session(public)
    assert [_note(public, sid).status_code for _ in range(8)] == [422] * 8   # nothing to run it on: a 422 each, and none took a token
    assert public.app.state.rate_limiter.tracked() == 0


def test_however_many_notes_that_run_no_model_are_sent_the_bucket_holds_what_the_image_turn_left(public):
    sid = _session(public)
    assert _turn_request(public, sid).status_code == 200
    wait_until(lambda: public.app.state.worker.in_flight == 0)
    for _ in range(12):   # twice the bucket, in notes
        assert _note(public, sid).status_code == 200
    assert public.app.state.rate_limiter.tracked() == 1   # one client in the table: the image turn's token is the only one it has spent
    tokens, _counted = public.app.state.rate_limiter._buckets["a"]
    assert tokens == pytest.approx(5.0, abs=0.01)


def test_a_note_is_free_over_a_real_socket_too_where_the_scope_carries_uvicorns_own_state(tmp_path):
    """TestClient and uvicorn build a request's scope differently (uvicorn's holds a copy of the lifespan state): the flag the route sets in
    request.state must reach the limiter through that scope as well, so eight notes after one image turn leave the bucket at five."""
    base, stop = start_live_server(create_app(engine="tiny", home=str(tmp_path), mode="public", token="t"))
    try:
        sid = httpx.post(base + "/v1/sessions", json={}, headers=PUBLIC).json()["id"]
        url = base + "/v1/sessions/{}/messages".format(sid)
        image = httpx.post(url, files={"image": ("x.png", png_bytes(), "image/png")}, data={"options": json.dumps({"max_new_tokens": 16})},
                           headers=PUBLIC, timeout=60)
        assert image.status_code == 200
        notes = [httpx.post(url, data={"text": "Is there pneumonia?"}, headers=PUBLIC, timeout=60) for _ in range(8)]
        assert [r.status_code for r in notes] == [200] * 8
        assert all("not_a_command" in r.text for r in notes)
        again = [httpx.post(url, files={"image": ("x.png", png_bytes(), "image/png")}, data={"options": json.dumps({"max_new_tokens": 16})},
                            headers=PUBLIC, timeout=60).status_code for _ in range(6)]
        assert again == [200] * 5 + [429]   # the image turn, and five more: six in all, whatever the eight notes were
    finally:
        stop()


def test_a_turn_the_route_marks_free_gives_its_token_back_though_it_is_a_200():
    async def run():
        limiter = hardening.TurnLimiter(6, 600.0, clock=Clock())

        async def free_ok(scope, receive, send):
            scope.setdefault("state", {})[hardening.FREE_TURN] = True   # what the route does for a note that runs no model
            await JSONResponse({"ok": True})(scope, receive, send)

        async def paid_ok(scope, receive, send):
            await JSONResponse({"ok": True})(scope, receive, send)

        out = []
        for inner in (free_ok, paid_ok):
            guarded = hardening.TurnRateLimit(inner, limiter, client_id=server.CLIENT_ID)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=guarded), base_url="http://test") as c:
                out.append([(await c.post("/v1/sessions/s_1/messages", headers={"X-Client-Id": "a"})).status_code for _ in range(8)])
        return out

    assert asyncio.run(run()) == [[200] * 8, [200] * 6 + [429] * 2]   # eight free turns cost nothing; six paid ones empty the bucket


def test_the_limiters_own_docstring_says_a_note_that_runs_no_model_costs_nothing():
    text = " ".join(hardening.__doc__.split())
    assert "a note that runs no model" in text and "costs the client nothing" in text
    assert hardening.FREE_TURN == "free_turn"


def test_a_burst_of_turns_started_together_is_admitted_six_at_a_time_and_no_more():
    async def burst():
        async def slow_ok(scope, receive, send):
            await asyncio.sleep(0.05)   # long enough for every request below to pass the check before any is answered
            await JSONResponse({"ok": True})(scope, receive, send)

        guarded = hardening.TurnRateLimit(slow_ok, hardening.TurnLimiter(6, 600.0, clock=Clock()), client_id=server.CLIENT_ID)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=guarded), base_url="http://test") as c:
            replies = await asyncio.gather(*[c.post("/v1/sessions/s_1/messages", headers={"X-Client-Id": "a"}) for _ in range(10)])
        return sorted(r.status_code for r in replies)

    assert asyncio.run(burst()) == [200] * 6 + [429] * 4


def test_only_turn_creation_is_limited_and_a_spent_client_can_still_read_cancel_export_and_delete(public):
    sid = _session(public)
    for _ in range(6):
        mid = _turn_request(public, sid).headers["X-Message-Id"]
    assert _turn_request(public, sid).status_code == 429
    reads = [public.get("/v1/sessions", headers=PUBLIC), public.get("/v1/sessions/{}".format(sid), headers=PUBLIC),
             public.get("/v1/messages/{}".format(mid), headers=PUBLIC), public.post("/v1/messages/{}/cancel".format(mid), headers=PUBLIC),
             public.get("/v1/sessions/{}/export".format(sid), headers=PUBLIC), public.get("/v1/models", headers=PUBLIC),
             public.get("/healthz"), public.post("/v1/sessions", json={}, headers=PUBLIC), public.delete("/v1/sessions/{}".format(sid), headers=PUBLIC)]
    assert [r.status_code for r in reads] == [200] * 8 + [204]
    for _ in range(30):   # and as often as it likes
        assert public.get("/v1/sessions", headers=PUBLIC).status_code == 200


def test_a_request_with_no_usable_client_id_is_not_counted_and_the_route_refuses_it_as_before(public):
    sid = _session(public)
    for headers in ({"Authorization": "Bearer t"}, dict(PUBLIC, **{"X-Client-Id": ""}), dict(PUBLIC, **{"X-Client-Id": "x" * 129})):
        for _ in range(8):
            r = _turn_request(public, sid, headers=headers)
            assert r.status_code == 400 and r.json()["error"]["type"] == "invalid_request_error"
    assert public.app.state.rate_limiter.tracked() == 0


def test_a_request_without_the_token_never_reaches_the_limiter(public):
    sid = _session(public)
    for _ in range(10):
        assert _turn_request(public, sid, headers={"X-Client-Id": "a"}).status_code == 401
        assert _turn_request(public, sid, headers={"X-Client-Id": "a", "Authorization": "Bearer wrong"}).status_code == 401
    assert public.app.state.rate_limiter.tracked() == 0
    assert [_turn_request(public, sid).status_code for _ in range(6)] == [200] * 6   # a stranger could not have spent client a's turns


def test_private_mode_has_no_limit(tmp_path):
    with TestClient(create_app(engine="tiny", home=str(tmp_path))) as c:
        assert c.app.state.rate_limiter is None
        sid = c.post("/v1/sessions", json={}).json()["id"]
        assert [_turn_request(c, sid, headers={"X-Client-Id": "a"}).status_code for _ in range(9)] == [200] * 9


def test_a_client_that_mints_a_new_id_for_every_turn_escapes_the_limit(public):
    """The caveat the module's docstring states, as a test: the limit is by client id, and a client id is whatever the caller sends."""
    for i in range(8):
        headers = dict(PUBLIC, **{"X-Client-Id": "minted-{}".format(i)})
        assert _turn_request(public, _session(public, headers), headers=headers).status_code == 200


def test_the_module_says_so_in_its_docstring_that_a_client_minting_ids_escapes_and_what_still_bounds_the_load():
    text = " ".join(hardening.__doc__.split())
    assert "a client that mints a new X-Client-Id for every turn escapes the limit" in text
    assert "queue cap (4 accepted turns at once" in text

