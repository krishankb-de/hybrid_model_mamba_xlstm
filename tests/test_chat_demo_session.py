"""CHAT_UI_PLAN.md P9-C, part 3: scripts/chat_demo_session.py, the supervisor demo's session, run from the laptop against the chat server through
the SSH tunnel.

It makes one session titled "Supervisor demo", runs one test-split turn per row on the server's default model (the EOS model on the live server),
and asks the server to save the session's export as JSON and as Markdown on its own disk (POST /v1/sessions/{id}/export/save), where the text stays
(rules R1 and R7). It keeps the session so that the user can open it in the page, and prints one RESULT line of numbers, booleans and fixed words.
Here it runs against a real uvicorn over the tiny engine and the tiny gallery (synthetic data only), and against a fake server that puts a SECRET in
every text it sends. Nothing in this module connects to port 8000 (the user's dev server) or 8010 (the tunnel to the live server): the socket guard
below stops it, and the only places that write either number are the guard and the refusal tests that run with a client that cannot connect.
"""
import io
import json
import os
import re
import socket
import subprocess
import sys
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest

from app.server import CLIENT_ID as SERVER_CLIENT_ID, create_app
from scripts import chat_demo_session as demo
from scripts import chat_load_drill as drill
from scripts import chat_realturn_probe as probe
from tests.app_helpers import start_live_server
from tests.test_chat_load_drill import NoNetwork, Unreachable, refusal_line, write_token_file
from tests.test_chat_realturn_probe import private_turn, secreted, sse

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "chat_demo_session.py"
ROWS = [3, 7, 12, 20, 31]   # five of the tiny gallery's 40 test studies
TOKEN = "t0k-for-the-demo"
FORBIDDEN_PORTS = frozenset({8000, 8010})   # the user's dev server, and the tunnel to the live cluster server: nothing in this process connects to either


def is_forbidden(address):
    """Whether a socket address, as connect takes it ((host, port), or an IPv6 four-tuple), names a port of the user's own servers."""
    return isinstance(address, tuple) and len(address) >= 2 and address[1] in FORBIDDEN_PORTS


@pytest.fixture(autouse=True)
def never_the_ports_of_the_users_own_servers(monkeypatch):
    """No test may connect to a port of FORBIDDEN_PORTS, whatever it says in a URL: the socket's own connect is guarded for the whole module, so that a
    test that was wrong about its fake or its server would fail instead. The guard stays in this process: no child process is ever handed either port
    (pinned in tests/test_willi_parity.py)."""
    real_connect, real_connect_ex = socket.socket.connect, socket.socket.connect_ex

    def connect(self, address):
        if is_forbidden(address):
            raise AssertionError("a test tried to connect to a port of the user's own servers")
        return real_connect(self, address)

    def connect_ex(self, address):
        if is_forbidden(address):
            raise AssertionError("a test tried to connect to a port of the user's own servers")
        return real_connect_ex(self, address)
    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)


@pytest.fixture(autouse=True)
def no_ambient_token(monkeypatch):
    """Whoever runs the real driver has CHAT_TOKEN exported in that shell: no test may send it, to a local test server or anywhere."""
    monkeypatch.delenv("CHAT_TOKEN", raising=False)


@pytest.fixture
def offline(monkeypatch):
    """A refusal must come before anything is opened: httpx's client and its own request functions fail the test when called."""
    monkeypatch.setattr(httpx, "Client", NoNetwork)
    for name in ("get", "post", "delete", "request", "stream"):
        monkeypatch.setattr(httpx, name, NoNetwork)


def run_demo(args, expect_code=None):
    """main() with its own two streams, so what it writes is told apart from the in-process server's request log. -> (code, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    code = demo.main(list(args), out, err)
    if expect_code is not None:
        assert code == expect_code, (code, out.getvalue(), err.getvalue())
    return code, out.getvalue(), err.getvalue()


def the_result(stdout):
    """The one RESULT line of a run, parsed."""
    lines = stdout.splitlines()
    assert len(lines) == 1 and lines[0].startswith("RESULT "), lines
    return json.loads(lines[0][len("RESULT "):])


# ---- what the driver asks for ---------------------------------------------------------------------------------------------------

def test_the_session_title_the_client_id_and_the_turn_budget_are_the_plans_and_the_eos_models():
    assert demo.TITLE == "Supervisor demo"
    assert demo.CLIENT_ID == "chat-demo-session" and SERVER_CLIENT_ID.fullmatch(demo.CLIENT_ID)
    assert demo.CLIENT_ID not in (drill.CLIENT_ID, probe.CLIENT_ID)
    assert demo.SAVE_FORMATS == ("json", "md")
    assert demo.DEFAULT_MAX_NEW_TOKENS == probe.DEFAULT_EOS_TOKENS == 200, "the budget the EOS model's registry note was measured at"


def test_a_turn_asks_for_the_servers_default_model_on_the_published_protocol_and_a_test_row():
    """The brief: one test-split turn per row on the server's default model, so no `model` key; the published protocol of the probe and the page's
    retrieval counts, at the EOS model's budget. Options forbids an unknown key, so a typo here is a failure."""
    from app.schemas import Options
    options = demo.turn_options(12, 200)
    assert options == dict(probe.PUBLISHED_PROTOCOL, k_images=4, k_reports=3, max_new_tokens=200, test_row=12) and "model" not in options
    assert Options.model_validate(options).model is None and Options.model_validate(options).test_row == 12
    assert demo.turn_options(0, 16)["max_new_tokens"] == 16 and demo.turn_options(2662, 200)["test_row"] == 2662


def test_the_drills_session_helper_sends_a_title_only_when_it_is_given_one():
    """The one change to the load drill: create_session(client, title) posts {"title": ...}; with no title it posts {} as it always did."""
    class Recorder:
        def __init__(self):
            self.posts = []

        def post(self, path, **kwargs):
            self.posts.append((path, kwargs))
            return httpx.Response(200, json={"id": "s_0123456789abcdef012345"})

    client = Recorder()
    assert drill.create_session(client) == "s_0123456789abcdef012345"
    assert drill.create_session(client, "Supervisor demo") == "s_0123456789abcdef012345"
    assert client.posts == [("/v1/sessions", {"json": {}}), ("/v1/sessions", {"json": {"title": "Supervisor demo"}})]


def test_the_defaults_and_the_help():
    args = demo.build_parser().parse_args(["http://127.0.0.1:18080", "--rows", "1,2"])
    assert args.rows == [1, 2] and args.max_new_tokens == 200 and args.timeout == demo.DEFAULT_TIMEOUT_S == 1800.0 and args.token_file is None
    code, out, err = run_demo(["--help"])
    assert code == 0 and err == "" and out.startswith("usage: ")
    flat = " ".join(out.split())   # argparse wraps the description where it likes
    for word in ("--token-file", "CHAT_TOKEN", "--rows", "--max-new-tokens", "--timeout", "RESULT", 'titled "Supervisor demo"'):
        assert word in flat, word


# ---- what it refuses: it never connects --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("url", ["http://127.0.0.1:8000", "http://127.0.0.1:8000/", "http://localhost:8000", "http://[::1]:8000/",
                                 "127.0.0.1:8000", "localhost:8000", "https://example.org:8000", "HTTP://LOCALHOST:8000",
                                 "http://127.0.0.1:08000", "http://127.0.0.1:+8000", "http://127.0.0.1:８０００", "http://user@127.0.0.1:8000"])
def test_the_script_refuses_port_8000_in_one_line_with_exit_2_and_connects_to_nothing(url, offline):
    code, out, err = run_demo([url, "--rows", "1"])
    assert code == 2 and out == ""
    assert "8000" in refusal_line(err) and "dev server" in err


@pytest.mark.parametrize("url", ["http://127.0.0.1:18000", "http://127.0.0.1:8001", "http://127.0.0.1:80", "http://127.0.0.1:18080",
                                 "http://localhost", "http://127.0.0.1:8010"])
def test_only_port_8000_itself_is_refused_for_being_the_dev_server(url):
    """8010 is the tunnel's own port, the one the controller runs the driver on: it is a loopback address and is taken."""
    assert drill.check_url(url).startswith("http")


@pytest.mark.parametrize("url", ["http://example.org:18080", "https://example.org", "http://10.0.0.5:18080", "http://192.168.1.20", "http://0.0.0.0:18080",
                                 "http://[2001:db8::1]:18080", "http://localhost.example.org:18080", "http://127.1:18080", "http://2130706433:18080",
                                 "http://node-07.internal", "example.org:18080"])
def test_a_url_that_is_not_loopback_is_refused_in_one_line_that_does_not_quote_it_and_connects_to_nothing(url, offline):
    """The driver goes through the tunnel and saves MIMIC-derived text on a server: what it talks to is the tunnel's own end, a loopback address."""
    code, out, err = run_demo([url, "--rows", "1"])
    assert code == 2 and out == "" and refusal_line(err) == demo.LOOPBACK_REFUSAL
    assert "example" not in err and "10.0.0.5" not in err and "internal" not in err and "loopback" in err


@pytest.mark.parametrize("url", ["http://127.0.0.1:18080", "http://localhost:18080", "http://[::1]:18080", "http://127.9.9.9:18080", "http://localhost",
                                 "http://127.0.0.1:8010"])
def test_a_loopback_address_gets_past_the_checks_to_its_first_request_which_nothing_answers(url, monkeypatch):
    monkeypatch.setattr(httpx, "Client", Unreachable)
    code, out, err = run_demo([url, "--rows", "1"])
    result = the_result(out)
    assert code == 1 and err == "" and result["ok"] is False and result["error"] == "session_create_ConnectError" and result["session"] is None


@pytest.mark.parametrize("url", ["", "   ", "http://", "ftp://127.0.0.1:2121", "http://127.0.0.1:0", "http://127.0.0.1:70000",
                                 "http://user:pass@127.0.0.1:1234", "http://tok@127.0.0.1:1234", "http://127.0.0.1:1234/?token=abc",
                                 "http://127.0.0.1:1234/#frag"])
def test_a_url_that_is_not_plain_scheme_host_and_port_is_refused_without_being_quoted(url, offline):
    code, out, err = run_demo([url, "--rows", "1"])
    assert code == 2 and out == ""
    line = refusal_line(err)
    assert "dev server" not in line and "127.0.0.1" not in line and "pass" not in line and "tok@" not in line and "abc" not in line


@pytest.mark.parametrize("argv", [
    ["http://127.0.0.1:18080", "--rows", "1", "--token", "SECRET-token"],
    ["http://127.0.0.1:18080", "--rows", "1", "--token=SECRET-token"],
    ["http://127.0.0.1:18080", "--rows", "1", "--tok", "SECRET-token"],                  # an abbreviation of --token-file must not be taken for it
    ["http://127.0.0.1:18080", "--rows", "1", "--token-fil", "SECRET-token"],
    ["--token", "SECRET-token", "http://127.0.0.1:18080", "--rows", "1"],
    ["http://127.0.0.1:18080", "SECRET-token", "--rows", "1"],                           # a second positional: URL TOKEN
    ["http://127.0.0.1:18080", "--rows", "1", "-t", "SECRET-token"],
    ["http://127.0.0.1:18080", "--rows", "1", "--bearer", "SECRET-token", "--password=SECRET-token"],
])
def test_the_script_refuses_a_token_on_the_command_line_and_never_echoes_it(argv, offline):
    code, out, err = run_demo(argv)
    assert code == 2 and out == ""
    assert refusal_line(err) == drill.ARGV_REFUSAL and "SECRET" not in err


def test_a_token_is_not_sent_over_plain_http_to_a_host_that_is_not_loopback(offline, monkeypatch):
    monkeypatch.setenv("CHAT_TOKEN", "SECRET-token")
    code, out, err = run_demo(["http://example.org:18080", "--rows", "1"])
    assert code == 2 and out == "" and refusal_line(err) == drill.CLEARTEXT_REFUSAL and "SECRET" not in err


def test_a_refused_token_file_says_why_without_naming_the_file_or_what_is_in_it(tmp_path, offline):
    wide = write_token_file(tmp_path, "SECRET-token", 0o644, "wide.token")
    code, out, err = run_demo(["http://127.0.0.1:18080", "--rows", "1", "--token-file", str(wide)])
    assert code == 2 and out == "" and "0600" in refusal_line(err)
    assert "SECRET" not in err and "wide.token" not in err and str(tmp_path) not in err


def test_a_chat_token_that_is_not_visible_ascii_is_refused_without_being_quoted(monkeypatch, offline):
    for bad in ("has a space", "tokén", "tab\there"):
        monkeypatch.setenv("CHAT_TOKEN", bad)
        code, out, err = run_demo(["http://127.0.0.1:18080", "--rows", "1"])
        assert code == 2 and out == "" and "CHAT_TOKEN" in refusal_line(err) and bad not in err


@pytest.mark.parametrize("argv, piece", [
    ([], "URL"), (["--rows", "1,2"], "URL"), (["http://127.0.0.1:18080"], "--rows"), (["http://127.0.0.1:18080", "--timeout", "5"], "--rows"),
])
def test_a_missing_url_or_missing_rows_is_a_usage_refusal_in_one_fixed_line(argv, piece, offline):
    code, out, err = run_demo(argv)
    assert code == 2 and out == "" and piece in refusal_line(err)


@pytest.mark.parametrize("text", ["", " ", "x", "4-1", "-1", "0-", "1,,2", "1,1", "0-3,2", "99999999", "1.5", "0x3", "٣", "1;2", "0-9999999"])
def test_rows_that_are_no_numbers_are_refused_in_one_fixed_sentence_that_quotes_nothing(text, offline):
    code, out, err = run_demo(["http://127.0.0.1:18080", "--rows", text])
    assert code == 2 and out == ""
    assert refusal_line(err) == "argument --rows: " + probe.ROWS_REFUSAL


@pytest.mark.parametrize("flag, value", [("--max-new-tokens", "15"), ("--max-new-tokens", "201"), ("--max-new-tokens", "many"), ("--timeout", "0"),
                                         ("--timeout", "-5"), ("--timeout", "soon"), ("--timeout", "nan"), ("--timeout", "86401")])
def test_a_number_out_of_range_is_refused_in_one_line_before_anything_is_sent(flag, value, offline):
    code, out, err = run_demo(["http://127.0.0.1:18080", "--rows", "1", flag, value])
    assert code == 2 and out == "" and flag in refusal_line(err)
    if not value.lstrip("-").isdigit():
        assert value not in refusal_line(err), "a refusal does not quote the value it was given"


# ---- live: the whole run against a real uvicorn over the tiny engine and the tiny gallery ---------------------------------------------

@dataclass
class Served:
    base: str
    app: Any
    home: Path


def serve(home, **kwargs):
    app = create_app(engine="tiny", home=str(home), tiny_gallery=True, **kwargs)
    base, stop = start_live_server(app)
    assert int(base.rsplit(":", 1)[1]) not in FORBIDDEN_PORTS, "a test server never takes the ports of the user's own servers"
    return Served(base, app, Path(home)), stop


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    server, stop = serve(tmp_path_factory.mktemp("demo_home"))
    yield server
    stop()


@pytest.fixture(scope="module")
def full_run(served):
    """The driver over the five rows, once: the line it printed, and what the server holds afterwards."""
    with pytest.MonkeyPatch.context() as patch:
        patch.delenv("CHAT_TOKEN", raising=False)
        out, err = io.StringIO(), io.StringIO()
        code = demo.main([served.base, "--rows", ",".join(str(r) for r in ROWS)], out, err)
    result = the_result(out.getvalue())
    session = httpx.get(served.base + "/v1/sessions/{}".format(result["session"])).json()
    return {"code": code, "out": out.getvalue(), "err": err.getvalue(), "result": result, "session": session}


def test_a_full_run_prints_one_result_line_and_exits_0(full_run):
    assert full_run["code"] == 0 and full_run["err"] == ""
    assert full_run["out"].count("\n") == 1 and full_run["out"].endswith("\n")
    result = full_run["result"]
    assert list(result) == ["demo", "rows", "turns", "statuses", "tokens", "stopped", "eos_trained", "session", "saved", "ok"], \
        "the plan's keys in its order, and after the tokens the stop word of each turn and whether the model was trained to end its reports"
    assert result["demo"] == "done" and result["ok"] is True
    assert result["rows"] == ROWS and result["turns"] == 5 and result["statuses"] == ["done"] * 5
    assert len(result["tokens"]) == 5 and all(type(t) is int and 0 < t <= 200 for t in result["tokens"])
    assert result["stopped"] == ["budget"] * 5 and result["eos_trained"] is False, "the tiny engine decodes to its budget and is not trained to stop"
    assert re.fullmatch(r"s_[0-9a-f]{22}", result["session"])
    assert len(result["saved"]) == 2 and result["saved"][0].endswith(".json") and result["saved"][1].endswith(".md")


def test_the_session_is_titled_holds_the_five_test_studies_in_order_and_is_kept(full_run, served):
    session = full_run["session"]
    assert session["title"] == "Supervisor demo" and session["mode"] == "private"
    messages = session["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant"] * 5
    assert [m["test_row"] for m in messages if m["role"] == "user"] == ROWS
    assert [m["status"] for m in messages if m["role"] == "assistant"] == ["done"] * 5
    listed = httpx.get(served.base + "/v1/sessions").json()["sessions"]
    assert full_run["result"]["session"] in [s["id"] for s in listed], "the user can open it in the page: it is not deleted"


def test_every_turn_ran_the_published_protocol_at_the_eos_budget_on_the_servers_default_model(full_run, served):
    for message in (m for m in full_run["session"]["messages"] if m["role"] == "assistant"):
        options = message["options"]
        assert options["model"] == served.app.state.default_model, "no model was asked for: the server's own"
        assert (options["decode"], options["beam_size"], options["max_new_tokens"], options["stop_on_repeat"]) == ("beam", 3, 200, False)
        assert (options["k_images"], options["k_reports"], options["cached_decode"], options["compile"]) == (4, 3, True, False)


def test_the_two_files_are_on_the_servers_disk_in_exports_and_hold_what_the_download_holds(full_run, served):
    names = full_run["result"]["saved"]
    session = full_run["result"]["session"]
    for name, fmt in zip(names, ("json", "md")):
        assert re.fullmatch(r"session-{}-[0-9]{{8}}T[0-9]{{6}}Z(-[0-9]+)?\.{}".format(session, fmt), name), name
        body = (served.home / "exports" / name).read_bytes()
        assert body == httpx.get(served.base + "/v1/sessions/{}/export".format(session), params={"format": fmt}).content
    saved_json = json.loads((served.home / "exports" / names[0]).read_text())
    assert [m["test_row"] for m in saved_json["messages"] if m["role"] == "user"] == ROWS
    assert saved_json["session"]["title"] == "Supervisor demo"
    assert "Image: test row 3" in (served.home / "exports" / names[1]).read_text()


def test_the_driver_never_deletes_the_session_it_made(served, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("the driver deleted a session")
    monkeypatch.setattr(drill, "delete_session", forbidden)
    code, out, err = run_demo([served.base, "--rows", "1"], 0)
    session = the_result(out)["session"]
    assert httpx.get(served.base + "/v1/sessions/{}".format(session)).status_code == 200


def test_a_row_the_server_refuses_is_said_by_status_and_word_the_rest_still_run_and_the_exports_are_still_saved(served):
    code, out, err = run_demo([served.base, "--rows", "4,9999,5"])
    result = the_result(out)
    assert code == 1 and err == "" and result["ok"] is False and result["demo"] == "failed"
    assert result["statuses"] == ["done", "refused", "done"] and result["turns"] == 3
    assert result["errors"] == [None, "validation_error", None]
    assert result["tokens"][1] is None and len(result["saved"]) == 2, "the session is saved as it stands"
    assert "error" not in result, "the driver itself went right; one row did not"


def test_a_server_that_cannot_save_ends_in_a_fixed_code_and_the_turns_are_still_reported(served, monkeypatch):
    from app.store import Store

    def full_disk(*args, **kwargs):
        raise OSError(28, "No space left on device", "/sc/home/SECRET-user/chat_sessions/exports/x.tmp")
    monkeypatch.setattr(Store, "save_export", full_disk)
    code, out, err = run_demo([served.base, "--rows", "2"])
    result = the_result(out)
    assert code == 1 and result["statuses"] == ["done"] and result["saved"] == [] and result["error"] == "save_http_500" and result["ok"] is False
    assert "SECRET" not in out and "sc/home" not in out and err == ""


def test_an_export_that_saved_one_format_and_failed_the_other_is_not_ok_and_lists_what_was_saved(served, monkeypatch):
    from app.store import Store
    real = Store.save_export

    def only_json(self, session_id, fmt, client_id):
        if fmt == "md":
            raise OSError(5, "Input/output error")
        return real(self, session_id, fmt, client_id)
    monkeypatch.setattr(Store, "save_export", only_json)
    code, out, err = run_demo([served.base, "--rows", "6"])
    result = the_result(out)
    assert code == 1 and len(result["saved"]) == 1 and result["saved"][0].endswith(".json") and result["error"] == "save_http_500"


def test_a_public_server_refuses_the_rows_and_the_saves_and_nothing_is_written(tmp_path):
    token = write_token_file(tmp_path, TOKEN + "\n")
    server, stop = serve(tmp_path / "home", mode="public", token=TOKEN)
    try:
        code, out, err = run_demo([server.base, "--rows", "1,2", "--token-file", str(token)])
    finally:
        stop()
    result = the_result(out)
    assert code == 1 and err == "" and result["ok"] is False
    assert result["statuses"] == ["refused", "refused"] and result["errors"] == ["permission_error", "permission_error"]
    assert result["saved"] == [] and result["error"] == "save_http_403"
    assert not (tmp_path / "home" / "exports").exists(), "a public server writes no export"


def test_a_server_that_wants_a_token_gets_it_as_a_bearer_on_every_request_the_saves_included(tmp_path, monkeypatch):
    server, stop = serve(tmp_path / "home", token=TOKEN)
    try:
        monkeypatch.setenv("CHAT_TOKEN", TOKEN)
        code, out, err = run_demo([server.base, "--rows", "1"])
        assert code == 0 and the_result(out)["ok"] is True and len(the_result(out)["saved"]) == 2
        monkeypatch.delenv("CHAT_TOKEN")
        code, out, err = run_demo([server.base, "--rows", "1"])
        result = the_result(out)
        assert code == 1 and result["error"] == "session_create_http_401" and result["session"] is None and result["statuses"] == []
    finally:
        stop()


def test_a_run_that_runs_out_of_time_says_timeout_keeps_the_session_and_saves_nothing(tmp_path):
    server, stop = serve(tmp_path / "home", tiny_step_delay_s=0.05)
    try:
        code, out, err = run_demo([server.base, "--rows", "1,2,3", "--timeout", "1", "--max-new-tokens", "200"])
        result = the_result(out)
        kept = httpx.get(server.base + "/v1/sessions/{}".format(result["session"]))   # asked of the server itself, before it goes
    finally:
        stop()
    assert code == 1 and err == "" and result["error"] == "timeout" and result["statuses"] == ["timeout"] and result["errors"] == ["timeout"]
    assert result["saved"] == [] and result["turns"] == 1 and re.fullmatch(r"s_[0-9a-f]{22}", result["session"])
    assert kept.status_code == 200 and kept.json()["id"] == result["session"], "the session is there: the timeout deleted nothing"
    assert kept.json()["title"] == "Supervisor demo" and [m["test_row"] for m in kept.json()["messages"] if m["role"] == "user"] == [1]


def test_an_interrupt_ends_in_the_line_of_what_was_known_the_session_is_kept_and_nothing_is_saved(served, monkeypatch):
    real = probe.run_turn
    calls = []

    def interrupted(client, session_id, spec, deadline, clock=None):
        calls.append(spec.row)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return real(client, session_id, spec, deadline)

    monkeypatch.setattr(probe, "run_turn", interrupted)
    code, out, err = run_demo([served.base, "--rows", "1,2,3"])
    result = the_result(out)
    assert code == 1 and err == "" and result["error"] == "interrupted" and result["statuses"] == ["done"] and result["saved"] == []
    assert httpx.get(served.base + "/v1/sessions/{}".format(result["session"])).status_code == 200


def test_a_bug_in_the_driver_is_the_name_of_its_class_and_nothing_of_its_text(served, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("a thing with SECRET text in it")
    monkeypatch.setattr(probe, "run_turn", broken)
    code, out, err = run_demo([served.base, "--rows", "1"])
    result = the_result(out)
    assert code == 1 and result["error"] == "internal_RuntimeError" and "SECRET" not in out + err and result["saved"] == []


def test_the_script_runs_as_a_script_from_any_directory_and_prints_nothing_but_its_result_line(served, tmp_path):
    """A process of its own, started the way a person starts it: it puts the repository on its own path, prints one RESULT line and nothing on stderr,
    exits 0, and writes no file where it ran. The URL it is given is this module's own server's (an ephemeral port), never the user's."""
    assert int(served.base.rsplit(":", 1)[1]) not in FORBIDDEN_PORTS
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "CHAT_TOKEN")}
    done = subprocess.run([sys.executable, str(SCRIPT), served.base, "--rows", "8,9"], cwd=str(cwd), env=env, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0 and done.stderr == "", (done.stdout, done.stderr[-1500:])
    result = the_result(done.stdout)
    assert result["ok"] is True and result["rows"] == [8, 9] and len(result["saved"]) == 2
    assert list(cwd.iterdir()) == []


# ---- R7: a server that puts a SECRET in every text it sends -----------------------------------------------------------------------------

class FakeServer:
    """The chat server as far as the driver can see it, on a free loopback port of this process: the session, a turn's stream (the frames it is
    given) and the save answers, each with a SECRET in every text and a header on every answer. It writes down every request (method, path, query,
    bearer and client id) and which ones were DELETEs."""

    SESSION = "s_0123456789abcdef012345"
    GOOD_JSON = "session-s_0123456789abcdef012345-20261013T181530Z.json"
    GOOD_MD = "session-s_0123456789abcdef012345-20261013T181530Z.md"

    def __init__(self, frames, saves=None, create=None, refuse_turn=None, per_turn=None):
        """`frames` are the frames every turn is answered with; `per_turn`, a list of frame lists, answers the turns one by one instead (the last
        one goes on answering when there are more turns than lists)."""
        self.frames, self.refuse_turn, self.per_turn, self.turns_seen = frames, refuse_turn, per_turn, 0
        self.saves = saves or {"json": (200, {"saved": self.GOOD_JSON, "bytes": 4096}), "md": (200, {"saved": self.GOOD_MD, "bytes": 2048})}
        self.create = create or (200, {"id": self.SESSION, "title": "SECRET-title", "mode": "private"})
        self.requests = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _answer(self, status, body):
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("X-Secret", "SECRET-header")
                self.end_headers()
                self.wfile.write(data)

            def _note(self, method):
                split = urlsplit(self.path)
                length = int(self.headers.get("Content-Length") or 0)
                if length:
                    self.rfile.read(length)
                owner.requests.append((method, split.path, parse_qs(split.query), self.headers.get("Authorization"), self.headers.get("X-Client-Id")))
                return split

            def do_POST(self):
                split = self._note("POST")
                if split.path == "/v1/sessions":
                    return self._answer(*owner.create)
                if split.path.endswith("/messages"):
                    if owner.refuse_turn is not None:
                        return self._answer(*owner.refuse_turn)
                    frames = owner.per_turn[min(owner.turns_seen, len(owner.per_turn) - 1)] if owner.per_turn else owner.frames
                    owner.turns_seen += 1
                    data = sse(frames).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("X-Message-Id", "m_SECRET-id")
                    self.end_headers()
                    self.wfile.write(data)
                    return None
                if split.path.endswith("/export/save"):
                    return self._answer(*owner.saves[parse_qs(split.query)["format"][0]])
                return self._answer(404, {"type": "error", "error": {"type": "not_found_error", "message": "SECRET not found"}})

            def do_DELETE(self):
                self._note("DELETE")
                self.send_response(204)
                self.end_headers()

            def do_GET(self):
                self._note("GET")
                self._answer(404, {"type": "error", "error": {"type": "not_found_error", "message": "SECRET not found"}})

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        assert self.port not in FORBIDDEN_PORTS, "a test server never takes the ports of the user's own servers"
        threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True).start()

    @property
    def base(self):
        return "http://127.0.0.1:{}".format(self.port)

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def fake():
    made = []

    def make(*args, **kwargs):
        server = FakeServer(*args, **kwargs)
        made.append(server)
        return server

    yield make
    for server in made:
        server.stop()


CLOSED_STOPPED = set(probe.STOPPED) | {"unknown", None}   # the stop words the probe keeps: its fixed list, `unknown` for any other, null for none


def nothing_secret(code_out_err):
    """What every run against the fake server must hold: no SECRET on either stream, nothing on stderr, and, on the RESULT line, a per-turn stop word
    that is one of the probe's closed list (or `unknown` or null) and an `eos_trained` that is a boolean or null, whatever the server said."""
    code, out, err = code_out_err
    assert "SECRET" not in out + err, (out, err)
    assert "s_0123456789abcdef012345" not in err and err == ""
    for line in out.splitlines():
        if line.startswith("RESULT "):
            result = json.loads(line[len("RESULT "):])
            assert all(word in CLOSED_STOPPED for word in result["stopped"]), result["stopped"]
            assert result["eos_trained"] is None or type(result["eos_trained"]) is bool, result["eos_trained"]
    return code, out, err


def test_a_hostile_server_leaves_no_word_of_its_own_on_either_stream_in_a_run_that_goes_right(fake):
    """The turn's frames carry a SECRET in the report and every snapshot, the matched reports, the study ids, the urls, the label names, the reference
    and the warnings (the probe's own private_turn); the session's title, a header on every answer and a field the driver never asks for in the save
    answer carry one too. The line has numbers, booleans and fixed words, and the names the server chose for its files."""
    server = fake(private_turn(), saves={"json": (200, {"saved": FakeServer.GOOD_JSON, "bytes": 4096, "report": "SECRET report in the answer"}),
                                         "md": (200, {"saved": FakeServer.GOOD_MD, "bytes": 2048, "path": "/sc/home/SECRET/exports"})})
    code, out, err = nothing_secret(run_demo([server.base, "--rows", "1,2,3"]))
    result = the_result(out)
    assert code == 0 and result["ok"] is True and result["statuses"] == ["done"] * 3 and result["saved"] == [FakeServer.GOOD_JSON, FakeServer.GOOD_MD]
    assert result["session"] == FakeServer.SESSION and result["tokens"] == [100, 100, 100]
    assert not [r for r in server.requests if r[0] == "DELETE"], "no session is deleted"
    assert [r[1] for r in server.requests].count("/v1/sessions/{}/export/save".format(FakeServer.SESSION)) == 2


def test_every_string_of_every_frame_a_secret_and_the_result_line_has_none_of_them(fake):
    """Each string of each frame is a SECRET: the stage names, the stop word, the statuses, the error kind, the codes. A word that is not one of the
    fixed words is `unknown`, and a stage the driver does not know is not read."""
    frames = [(event, secreted(data)) for event, data in private_turn()]
    frames.append(("error", {"seq": len(frames) + 1, "type": "error", "error": {"type": "SECRET-kind", "message": "SECRET-message"}}))
    server = fake(frames)
    code, out, err = nothing_secret(run_demo([server.base, "--rows", "1"]))
    result = the_result(out)
    assert code == 1 and result["statuses"] == ["unknown"] and result["errors"] == ["unknown"] and result["tokens"] == [None]
    assert result["stopped"] == [None] and result["eos_trained"] is False, "no stage it knows, so no stop word; the flag is a bool, and a bool is kept"
    assert result["ok"] is False and len(result["saved"]) == 2, "the saves are asked for all the same: a turn did not go as it should"


def test_each_turns_stop_word_and_whether_the_model_was_trained_to_end_its_reports_are_on_the_line(fake):
    """Review minor 5: a report that was cut. `stopped` is the generate stage's word per turn (`eos` the model ended it, `repeat` the stop on a
    repeated sentence, `budget` the token budget did), and `eos_trained` says, once, whether the model that ran the turns was trained to end its
    reports: `budget` with `true` is a report that was cut, `budget` with `false` is how that model always ends."""
    server = fake(None, per_turn=[private_turn(stopped="eos", eos=True), private_turn(stopped="budget", eos=True), private_turn(stopped="repeat", eos=True)])
    code, out, err = nothing_secret(run_demo([server.base, "--rows", "1,2,3"]))
    result = the_result(out)
    assert code == 0 and result["stopped"] == ["eos", "budget", "repeat"] and result["eos_trained"] is True
    keys = list(result)
    assert keys[keys.index("tokens") + 1:keys.index("tokens") + 3] == ["stopped", "eos_trained"]
    server = fake(private_turn(stopped="budget", eos=False))
    result = the_result(nothing_secret(run_demo([server.base, "--rows", "1,2"]))[1])
    assert result["stopped"] == ["budget", "budget"] and result["eos_trained"] is False


@pytest.mark.parametrize("stopped, word", [("SECRET-stop", "unknown"), ("EOS", "unknown"), (" eos", "unknown"), ("", "unknown"), (7, "unknown"),
                                           (["eos"], "unknown"), ({"eos": 1}, "unknown"), (True, "unknown"), (None, "unknown"), ("eos", "eos"),
                                           ("budget", "budget"), ("repeat", "repeat")])
def test_the_stop_word_is_one_of_the_probes_closed_words_or_unknown_whatever_the_server_says(stopped, word, fake):
    turn = private_turn()
    next(data for event, data in turn if event == "stage_end" and data["stage"] == "generate")["detail"]["stopped"] = stopped
    server = fake(turn)
    result = the_result(nothing_secret(run_demo([server.base, "--rows", "1"]))[1])
    assert result["stopped"] == [word]


def test_a_turn_with_no_stop_word_says_null_and_not_a_word_of_its_own(fake):
    turn = private_turn()
    del next(data for event, data in turn if event == "stage_end" and data["stage"] == "generate")["detail"]["stopped"]
    server = fake(turn)
    assert the_result(nothing_secret(run_demo([server.base, "--rows", "1"]))[1])["stopped"] == [None]


NO_FLAG = object()   # in the test below: the card has no eos_trained at all


@pytest.mark.parametrize("flag, expected", [(True, True), (False, False), ("SECRET-yes", None), ("true", None), (1, None), (0, None), (None, None),
                                            ([True], None), ({"x": True}, None), (NO_FLAG, None)],
                         ids=["true", "false", "text", "text-true", "one", "zero", "null", "list", "object", "absent"])
def test_eos_trained_is_a_boolean_or_null_whatever_the_server_says(flag, expected, fake):
    turn = private_turn()
    model = next(data for event, data in turn if event == "message_start")["model"]
    if flag is NO_FLAG:
        del model["eos_trained"]
    else:
        model["eos_trained"] = flag
    server = fake(turn)
    result = the_result(nothing_secret(run_demo([server.base, "--rows", "1,2"]))[1])
    assert result["eos_trained"] is expected


def test_turns_that_disagree_about_the_flag_and_a_run_with_no_turn_say_null(fake):
    """The default model ran every turn, so they agree; when they do not, or when no turn ran, the flag is not guessed."""
    differ = [private_turn(eos=True), private_turn(eos=False)]
    server = fake(None, per_turn=differ)
    assert the_result(nothing_secret(run_demo([server.base, "--rows", "1,2"]))[1])["eos_trained"] is None
    server = fake(private_turn(), create=(500, {"message": "SECRET"}))
    result = the_result(nothing_secret(run_demo([server.base, "--rows", "1,2"]))[1])
    assert result["stopped"] == [] and result["eos_trained"] is None


def test_a_refused_turn_has_no_stop_word_and_does_not_make_the_flag_up(fake):
    server = fake(private_turn(), refuse_turn=(422, {"type": "error", "error": {"type": "validation_error", "message": "SECRET"}}))
    result = the_result(nothing_secret(run_demo([server.base, "--rows", "1"]))[1])
    assert result["statuses"] == ["refused"] and result["stopped"] == [None] and result["eos_trained"] is None


def test_a_save_answer_that_is_not_a_file_name_the_server_could_have_given_is_dropped_whatever_it_says(fake):
    bad = [{"saved": "SECRET-name", "bytes": 100}, {"saved": "../../SECRET", "bytes": 100}, {"saved": FakeServer.GOOD_JSON + " SECRET", "bytes": 100},
           {"saved": "session-s_other-20261013T181530Z.json", "bytes": 100},   # another session's name
           {"saved": FakeServer.GOOD_MD, "bytes": 100},                        # an md name for the json asked for
           {"saved": FakeServer.GOOD_JSON, "bytes": "SECRET"}, {"saved": FakeServer.GOOD_JSON, "bytes": -1}, {"saved": FakeServer.GOOD_JSON, "bytes": 0},
           {"saved": FakeServer.GOOD_JSON, "bytes": True}, {"saved": FakeServer.GOOD_JSON, "bytes": 10 ** 400}, {"saved": FakeServer.GOOD_JSON},
           {"saved": 12, "bytes": 100}, {"saved": ["SECRET"], "bytes": 100}, ["SECRET"], "SECRET", None]
    server = fake(private_turn())
    for answer in bad:
        server.saves = {"json": (200, answer), "md": (200, {"saved": FakeServer.GOOD_MD, "bytes": 2048})}
        code, out, err = nothing_secret(run_demo([server.base, "--rows", "1"]))
        result = the_result(out)
        assert code == 1 and result["saved"] == [FakeServer.GOOD_MD] and result["error"] == "save_bad_answer", answer


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422, 500, 503])
def test_a_refused_save_is_its_status_and_nothing_of_its_body(status, fake):
    body = {"type": "error", "error": {"type": "SECRET-kind", "message": "SECRET refusal: /sc/home/SECRET-user/chat_sessions"}}
    server = fake(private_turn(), saves={"json": (status, body), "md": (status, body)})
    code, out, err = nothing_secret(run_demo([server.base, "--rows", "1"]))
    result = the_result(out)
    assert code == 1 and result["saved"] == [] and result["error"] == "save_http_{}".format(status)


def test_a_hostile_refusal_of_the_turn_is_a_status_word_and_a_word_of_the_probes_list(fake):
    server = fake(private_turn(), refuse_turn=(422, {"type": "error", "error": {"type": "SECRET-kind", "message": "SECRET text"}}))
    code, out, err = nothing_secret(run_demo([server.base, "--rows", "1"]))
    result = the_result(out)
    assert result["statuses"] == ["refused"] and result["errors"] == ["unknown"] and result["tokens"] == [None]
    server = fake(private_turn(), refuse_turn=(429, {"type": "error", "error": {"type": "overloaded_error", "message": "SECRET text"}}))
    assert the_result(nothing_secret(run_demo([server.base, "--rows", "1"]))[1])["errors"] == ["overloaded_error"]


@pytest.mark.parametrize("create, code", [((500, {"message": "SECRET"}), "session_create_http_500"), ((200, {"id": "SECRET id with spaces"}), "session_create_bad_answer"),
                                          ((200, {"id": "../SECRET"}), "session_create_bad_answer"), ((200, ["SECRET"]), "session_create_bad_answer"),
                                          ((401, {"type": "error", "error": {"type": "SECRET-kind"}}), "session_create_http_401")])
def test_a_session_that_cannot_be_made_is_a_fixed_code_and_nothing_else_runs(create, code, fake):
    server = fake(private_turn(), create=create)
    exit_code, out, err = nothing_secret(run_demo([server.base, "--rows", "1,2"]))
    result = the_result(out)
    assert exit_code == 1 and result["error"] == code and result["session"] is None and result["turns"] == 0 and result["saved"] == []
    assert [r[1] for r in server.requests] == ["/v1/sessions"], "nothing but the one request"


def test_every_request_carries_the_fixed_client_id_and_the_bearer_goes_to_the_server_it_was_given_and_nowhere_else(fake, tmp_path):
    server = fake(private_turn())
    token = write_token_file(tmp_path, TOKEN + "\n")
    code, out, err = nothing_secret(run_demo([server.base, "--rows", "1", "--token-file", str(token)]))
    assert code == 0 and the_result(out)["ok"] is True
    assert {r[4] for r in server.requests} == {"chat-demo-session"} and {r[3] for r in server.requests} == {"Bearer " + TOKEN}
    assert TOKEN not in out + err


def test_the_save_asks_for_the_two_formats_json_first_by_the_query_the_route_takes(fake):
    server = fake(private_turn())
    run_demo([server.base, "--rows", "1"], 0)
    saves = [r for r in server.requests if r[1].endswith("/export/save")]
    assert [(r[0], r[1], r[2]) for r in saves] == [("POST", "/v1/sessions/{}/export/save".format(FakeServer.SESSION), {"format": ["json"]}),
                                                   ("POST", "/v1/sessions/{}/export/save".format(FakeServer.SESSION), {"format": ["md"]})]
    posts = [r for r in server.requests if r[1].endswith("/messages")]
    assert len(posts) == 1


def test_the_driver_connects_straight_to_the_url_whatever_proxy_the_environment_names(fake, monkeypatch):
    """httpx would send even a loopback request to the proxy of HTTP_PROXY or ALL_PROXY: the token and the stream would pass through it."""
    server = fake(private_turn())
    for name in ("HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy", "HTTPS_PROXY", "https_proxy"):
        monkeypatch.setenv(name, "http://127.0.0.1:9")   # nothing listens there: a request that used it would fail
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    code, out, err = run_demo([server.base, "--rows", "1"])
    assert code == 0 and the_result(out)["ok"] is True
