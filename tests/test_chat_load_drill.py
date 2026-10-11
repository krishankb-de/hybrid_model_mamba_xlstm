"""CHAT_UI_PLAN.md P8-C: the load drill. Six turns at once on a server whose queue holds four: four are accepted and run to their end, two are
refused with 429 `overloaded_error` before their stream opens. The first test pins that the `live` fixture and this file share one server
starter; the second runs the drill against a real uvicorn with the tiny engine; the rest are the tests of scripts/chat_load_drill.py, which runs
the same drill from the laptop against the cluster's server through the tunnel. Synthetic data only: the tiny engine, the tiny gallery, and
pictures the script draws itself.
"""
import ast
import io
import json
import os
import socket
import threading
from pathlib import Path
from unittest import mock

import httpx
import pytest
from PIL import Image

from app import cli
from app.server import CLIENT_ID, create_app
from app.store import Store
from scripts import chat_load_drill as drill
from tests.app_helpers import iter_sse, png_bytes, start_live_server, wait_until

REPO_ROOT = Path(__file__).resolve().parent.parent
FAST = ["--max-new-tokens", "16"]                         # a turn of a tiny engine that paces nothing: well under a second
ONE_TURN = ["--turns", "1", "--queue-cap", "1"] + FAST    # a drill whose outcome does not depend on timing: one turn, one slot


@pytest.fixture(autouse=True)
def no_ambient_token(monkeypatch):
    """Whoever runs the real drill has CHAT_TOKEN exported in that shell: no test may send it, to a local test server or anywhere."""
    monkeypatch.delenv("CHAT_TOKEN", raising=False)


def test_the_live_fixture_starts_its_server_through_the_shared_helper():
    """The P3-D `live` fixture and the drill below start their uvicorn in one place, tests/app_helpers.start_live_server: the fixture holds
    no copy of it (no uvicorn, no socket)."""
    tree = ast.parse((REPO_ROOT / "tests" / "test_app_api.py").read_text(encoding="utf-8"))
    live = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "live")
    called = {n.func.id for n in ast.walk(live) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    names = {n.id for n in ast.walk(live) if isinstance(n, ast.Name)}
    names |= {alias.name for n in ast.walk(live) if isinstance(n, ast.Import) for alias in n.names}
    assert "start_live_server" in called, "the fixture calls the helper"
    assert not names & {"uvicorn", "socket"}, sorted(names & {"uvicorn", "socket"})


def test_six_concurrent_turns_four_accepted_two_refused(tmp_path):
    base, stop = start_live_server(create_app(engine="tiny", home=str(tmp_path), queue_cap=4,
                                              tiny_step_delay_s=0.02))
    try:
        sid = httpx.post(base + "/v1/sessions", json={}).json()["id"]
        url = base + "/v1/sessions/{}/messages".format(sid)
        results = []
        together = threading.Barrier(6)   # the six leave at the same moment, so the queue is full for the last two however loaded the machine is

        def one():
            together.wait(30)
            # The same picture from all six, at once, into one session: the case Store.save_upload used to race on (a 500 "could not store the
            # image", about one drill in ten), so this is also the end-to-end test of its P8-C fix.
            with httpx.stream("POST", url, files={"image": ("x.png", png_bytes(), "image/png")},
                              data={"text": "", "options": json.dumps({"max_new_tokens": 60})}, timeout=60) as r:
                if r.status_code == 200:
                    results.append((200, list(iter_sse(r.iter_text())), None))
                else:
                    r.read()
                    results.append((r.status_code, [], r.json()["error"]["type"]))

        threads = [threading.Thread(target=one) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(120)
        assert not any(t.is_alive() for t in threads)
        assert httpx.get(base + "/healthz").json()["turns_in_flight"] == 0   # every accepted turn is counted out
    finally:
        stop()
    assert sorted(code for code, _, _ in results) == [200, 200, 200, 200, 429, 429]
    # the queue's own refusal (private mode has no rate limit, so no other 429 could stand in for it)
    assert [kind for code, _, kind in results if code == 429] == ["overloaded_error", "overloaded_error"]
    for code, frames, _ in results:
        if code == 200:
            assert frames[0]["event"] == "message_start" and frames[-1]["event"] == "message_stop"
            assert frames[-1]["data"]["status"] == "done"
            assert [f["data"]["seq"] for f in frames] == list(range(1, len(frames) + 1))


# ---- the script: the drill from the laptop ------------------------------------------------------------------------------------------

def run_drill(args, expect_code=None):
    """main() with its own two streams, so what it writes is told apart from the in-process server's request log. -> (code, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    code = drill.main(list(args), out, err)
    if expect_code is not None:
        assert code == expect_code, (code, out.getvalue(), err.getvalue())
    return code, out.getvalue(), err.getvalue()


def the_result(stdout):
    """The one RESULT line of a run, parsed."""
    lines = stdout.splitlines()
    assert len(lines) == 1 and lines[0].startswith("RESULT "), lines
    return json.loads(lines[0][len("RESULT "):])


def test_the_script_runs_the_drill_and_prints_one_result_line_and_exits_0(tmp_path):
    base, stop = start_live_server(create_app(engine="tiny", home=str(tmp_path), queue_cap=4, tiny_step_delay_s=0.02))
    try:
        with mock.patch.object(Store, "create_session", autospec=True, side_effect=Store.create_session) as created:
            code, out, err = run_drill([base, "--max-new-tokens", "60"])
        assert created.call_count == 1, "exactly one session"
        assert httpx.get(base + "/v1/sessions").json()["sessions"] == [], "and it is gone: nothing is left on the server"
        assert httpx.get(base + "/healthz").json()["turns_in_flight"] == 0
    finally:
        stop()
    assert code == 0 and err == ""
    result = the_result(out)
    assert result["requested"] == 6 and result["accepted"] == 4 and result["refused_429"] == 2
    assert result["refused_429_types"] == {"overloaded_error": 2}   # the queue's refusal, not a rate limit's
    assert result["done"] == 4 and result["seq_contiguous"] is True and result["seq_contiguous_turns"] == [True] * 4
    assert result["session_deleted"] is True and result["ok"] is True and "error" not in result
    assert isinstance(result["wall_s"], float) and result["wall_s"] > 0


def test_a_server_that_takes_more_turns_than_the_drill_expects_fails_it_with_exit_1(tmp_path):
    base, stop = start_live_server(create_app(engine="tiny", home=str(tmp_path), queue_cap=6))   # six slots: nothing is refused, whatever the timing
    try:
        code, out, err = run_drill([base] + FAST)
    finally:
        stop()
    result = the_result(out)
    assert code == 1 and err == "" and result["ok"] is False
    assert result["accepted"] == 6 and result["refused_429"] == 0 and result["refused_429_types"] == {}
    assert result["done"] == 6 and result["seq_contiguous"] is True and result["session_deleted"] is True   # all else was fine


def test_the_drill_can_be_told_the_servers_queue_cap_and_how_many_turns_to_send(tmp_path):
    base, stop = start_live_server(create_app(engine="tiny", home=str(tmp_path), queue_cap=6))
    try:
        code, out, _ = run_drill([base, "--turns", "3", "--queue-cap", "6"] + FAST)
    finally:
        stop()
    result = the_result(out)
    assert code == 0 and result["requested"] == 3 and result["accepted"] == 3 and result["refused_429"] == 0 and result["ok"] is True


def outcome(status=200, refusal=None, seqs=(1, 2, 3), last_event="message_stop", final="done", error=None):
    return drill.Outcome(status=status, refusal=refusal, seqs=list(seqs), last_event=last_event, final=final, error=error)


def refused(kind="overloaded_error"):
    return outcome(429, kind, (), None, None)


def judge(outcomes, session_deleted=True, error=None, turns=6, queue_cap=4):
    return drill.result_payload(outcomes, turns, queue_cap, session_deleted, error, 1.234)


GOOD = [outcome() for _ in range(4)] + [refused(), refused()]


def test_the_result_counts_what_happened_and_is_ok_only_when_all_of_the_expectation_holds():
    result = judge(GOOD)
    assert result == {"requested": 6, "accepted": 4, "refused_429": 2, "refused_429_types": {"overloaded_error": 2}, "other_status": {},
                      "failed": {}, "done": 4, "seq_contiguous": True, "seq_contiguous_turns": [True] * 4, "session_deleted": True,
                      "wall_s": 1.23, "ok": True}
    assert list(result)[:3] == ["requested", "accepted", "refused_429"]


@pytest.mark.parametrize("why, outcomes, kwargs", [
    ("five accepted, one refused", [outcome() for _ in range(5)] + [refused()], {}),
    ("three accepted, three refused", [outcome() for _ in range(3)] + [refused() for _ in range(3)], {}),
    ("the refusals are a rate limit's, not the queue's", [outcome() for _ in range(4)] + [refused("rate_limited")] * 2, {}),
    ("a turn ended in error", [outcome(final="error")] + GOOD[1:], {}),
    ("a turn was aborted", [outcome(final="aborted")] + GOOD[1:], {}),
    ("a stream stopped before message_stop", [outcome(last_event="content_block_delta", final=None)] + GOOD[1:], {}),
    ("a gap in a turn's seq", [outcome(seqs=(1, 2, 4))] + GOOD[1:], {}),
    ("a turn's seq starts at 2", [outcome(seqs=(2, 3))] + GOOD[1:], {}),
    ("a frame without a seq", [outcome(seqs=(1, None, 3))] + GOOD[1:], {}),
    ("a turn with no frames at all", [outcome(seqs=(), last_event=None, final=None)] + GOOD[1:], {}),
    ("the session could not be deleted", GOOD, {"session_deleted": False}),
    ("no session was made", GOOD, {"session_deleted": None}),
    ("the drill itself failed", GOOD, {"error": "interrupted"}),
    ("one turn never answered", GOOD[1:] + [outcome(None, None, (), None, None, "ReadTimeout")], {}),
])
def test_the_result_is_not_ok_when_any_part_of_the_expectation_fails(why, outcomes, kwargs):
    assert judge(outcomes, **kwargs)["ok"] is False, why


def test_the_result_names_what_went_wrong_by_status_code_and_exception_class_only():
    result = judge([outcome(401, None, (), None, None) for _ in range(5)] + [outcome(None, None, (), None, None, "ReadTimeout")],
                   session_deleted=None, error="session_create_http_401")
    assert result["accepted"] == 0 and result["refused_429"] == 0
    assert result["other_status"] == {"401": 5} and result["failed"] == {"ReadTimeout": 1}
    assert result["session_deleted"] is None and result["error"] == "session_create_http_401" and result["ok"] is False
    assert result["seq_contiguous"] is False and result["seq_contiguous_turns"] == []   # nothing was accepted, so nothing is contiguous


def test_a_refusal_names_its_error_type_only_when_it_is_a_plain_word():
    """An error type is read from a server's answer: only a short lower-case word is kept, anything else is counted as unknown."""
    assert [drill.error_type({"error": {"type": t}}) for t in ("overloaded_error", "rate_limited")] == ["overloaded_error", "rate_limited"]
    for odd in (None, {}, [], "text", {"error": "x"}, {"error": {"type": 7}}, {"error": {"type": "Has Capitals And Spaces"}},
                {"error": {"type": "x" * 41}}, {"error": {"type": "line\nbreak"}}, {"error": {"type": ""}}):
        assert drill.error_type(odd) == "unknown", odd


def test_read_stream_keeps_the_seq_of_every_frame_the_last_event_and_the_final_status_and_nothing_else():
    text = (": ping\n\n"
            'event: message_start\r\ndata: {"seq":1,"mode":"SECRET-A"}\r\n\r\n'
            'event: content_block_delta\ndata: {"seq":2,"delta":{"text":"SECRET-B"}}\n\n'
            ": ping\n\n"
            'event: message_stop\ndata: {"seq":3,"status":"done","report":"SECRET-C"}\n\n')
    for size in (1, 7, 64, len(text)):   # whatever the chunking: a frame may be cut anywhere
        chunks = [text[i:i + size] for i in range(0, len(text), size)]
        assert drill.read_stream(iter(chunks)) == ([1, 2, 3], "message_stop", "done"), size
    assert drill.read_stream(iter([])) == ([], None, None)


def test_read_stream_survives_a_malformed_frame_and_a_status_that_is_not_one_of_the_three():
    text = ('event: content_block_delta\ndata: {not json}\n\n'
            'event: content_block_delta\ndata: [1, 2]\n\n'
            'event: content_block_delta\ndata: {"seq": true}\n\n'
            'event: message_stop\ndata: {"seq":4,"status":"SECRET free text"}\n\n')
    seqs, last, final = drill.read_stream(iter([text]))
    assert seqs == [None, None, None, 4] and last == "message_stop" and final == "unknown"
    assert drill.read_stream(iter(['event: message_stop\ndata: {"seq":1,"status":"aborted"}\n\n']))[2] == "aborted"


def test_the_synthetic_image_is_a_real_png_the_server_takes():
    data = drill.synthetic_png()
    img = Image.open(io.BytesIO(data))
    img.load()
    assert img.format == "PNG" and img.size == (256, 256) and img.mode == "L" and len(data) < 20 * 1024 * 1024
    assert img.getextrema()[0] != img.getextrema()[1], "not a flat image"
    assert data == drill.synthetic_png(), "the same bytes every time"


def test_every_turn_of_a_drill_can_have_a_picture_of_its_own():
    pictures = [drill.synthetic_png(variant=n) for n in range(drill.MAX_TURNS)]
    assert len(set(pictures)) == drill.MAX_TURNS
    for data in (pictures[0], pictures[-1]):
        img = Image.open(io.BytesIO(data))
        img.load()
        assert img.size == (256, 256) and img.getextrema()[0] != img.getextrema()[1]


class Answered:
    """A response the fake client hands out: the queue's 429, as a context manager."""
    status_code = 429

    def read(self):
        return b""

    def json(self):
        return {"error": {"type": "overloaded_error", "message": "busy"}}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class Recording:
    """In place of httpx.Client for run_turns alone: it writes down what each turn sent, and answers 429."""

    def __init__(self):
        self.sent, self.lock = [], threading.Lock()

    def stream(self, method, path, files=None, data=None):
        with self.lock:
            self.sent.append((method, path, files["image"], data))
        return Answered()


def test_run_turns_sends_each_turn_a_picture_of_its_own_with_the_budget_and_files_every_outcome():
    client, outcomes = Recording(), []
    args = drill.build_parser().parse_args(["http://localhost:18080", "--turns", "5", "--max-new-tokens", "33", "--timeout", "30"])
    assert drill.run_turns(client, "s_abc", args, outcomes) is False
    assert [(m, p) for m, p, _, _ in client.sent] == [("POST", "/v1/sessions/s_abc/messages")] * 5
    assert len({files[1] for _, _, files, _ in client.sent}) == 5, "five different pictures"
    assert {files[0] for _, _, files, _ in client.sent} == {"drill.png"} and {files[2] for _, _, files, _ in client.sent} == {"image/png"}
    assert {json.dumps(data, sort_keys=True) for _, _, _, data in client.sent} == {json.dumps({"options": '{"max_new_tokens": 33}', "text": ""})}
    assert [(o.status, o.refusal, o.error) for o in outcomes] == [(429, "overloaded_error", None)] * 5


def test_run_turns_reports_that_time_ran_out_when_a_stream_does_not_end():
    release = threading.Event()

    class Hanging(Recording):
        def stream(self, method, path, files=None, data=None):
            release.wait(30)   # a turn that never answers
            return Answered()

    outcomes = []
    args = drill.build_parser().parse_args(["http://localhost:18080", "--turns", "2", "--timeout", "0.2"])
    try:
        assert drill.run_turns(Hanging(), "s_abc", args, outcomes) is True
        assert len(outcomes) == 2 and all(o.status is None for o in outcomes)
    finally:
        release.set()   # the daemon threads end now rather than at the interpreter's exit


# ---- what the script refuses: it never connects ------------------------------------------------------------------------------------

class NoNetwork:
    """In place of httpx.Client and httpx's own request functions: a refusal must come before anything is opened."""

    def __init__(self, *args, **kwargs):
        raise AssertionError("the drill opened a connection")


@pytest.fixture
def offline(monkeypatch):
    monkeypatch.setattr(httpx, "Client", NoNetwork)
    for name in ("get", "post", "delete", "request", "stream"):
        monkeypatch.setattr(httpx, name, NoNetwork)


class Unreachable:
    """In place of httpx.Client when a test needs the drill to get past its checks to its first request, which nothing answers. A run that gets
    that far ends in one RESULT line whose error is session_create_ConnectError; a run that was refused before it never gets there."""

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def post(self, *args, **kwargs):
        raise httpx.ConnectError("could not connect to a thing with a secret in its name")


def refusal_line(err):
    """A refusal is exactly one line on stderr: ERROR and a fixed sentence."""
    assert err.count("\n") == 1 and err.endswith("\n") and err.startswith("ERROR "), err
    return err[len("ERROR "):-1]


@pytest.mark.parametrize("url", ["http://127.0.0.1:8000", "http://127.0.0.1:8000/", "http://localhost:8000", "http://[::1]:8000/",
                                 "127.0.0.1:8000", "localhost:8000", "https://example.org:8000", "HTTP://LOCALHOST:8000",
                                 "http://0.0.0.0:8000", "http://127.0.0.1:08000", "http://127.0.0.1:+8000", "http://127.0.0.1:8_000",
                                 "http://127.0.0.1:８０００", "http://127.1:8000", "http://user@127.0.0.1:8000"])
def test_the_script_refuses_port_8000_in_one_line_with_exit_2_and_connects_to_nothing(url, offline):
    code, out, err = run_drill([url])
    assert code == 2 and out == ""
    assert "8000" in refusal_line(err) and "dev server" in err


@pytest.mark.parametrize("url", ["http://127.0.0.1:18000", "http://127.0.0.1:8001", "http://127.0.0.1:0800", "http://127.0.0.1:80",
                                 "http://127.0.0.1:18080", "http://localhost", "https://example.org", "http://127.0.0.1:"])
def test_only_port_8000_itself_is_refused_for_being_the_dev_server(url):
    assert drill.check_url(url).startswith("http")   # no refusal


@pytest.mark.parametrize("url, piece", [
    ("", None), ("   ", None), ("http://", None), ("http://:1234", None), ("http:///x", None), ("http://127.0.0.1 :1234", None),
    ("ftp://127.0.0.1:2121", "ftp://"), ("http://127.0.0.1:0", None), ("http://127.0.0.1:70000", "70000"), ("http://127.0.0.1:80000", "80000"),
    ("http://127.0.0.1:abc", "abc"), ("http://127.0.0.1:8000.5", "8000.5"),
    ("http://user:pass@127.0.0.1:1234", "pass"), ("http://tok@127.0.0.1:1234", "tok@"),
    ("http://127.0.0.1:1234/?token=abc", "token=abc"), ("http://127.0.0.1:1234/#frag", "#frag"),
    # P8 follow-up (7), from the P8-C review: httpx raises no InvalidURL for these two, so each ended as a traceback and exit 1
    ("http://127.0.0.1:1234/\udcff", None),   # a byte of the path that is no UTF-8 (a shell's argv, decoded with surrogateescape): UnicodeEncodeError
    ("http://xn--", None),                    # an internationalised host name that is none (idna.IDNAError, a UnicodeError as well)
])
def test_a_url_that_is_not_plain_scheme_host_and_port_is_refused_without_being_quoted(url, piece, offline):
    code, out, err = run_drill([url])
    assert code == 2 and out == ""
    line = refusal_line(err)
    assert "dev server" not in line and "127.0.0.1" not in line   # the sentence is fixed: it does not quote the address it was given
    if piece is not None:
        assert piece not in line


def test_the_url_is_taken_up_to_its_origin():
    assert drill.check_url("http://127.0.0.1:18080") == "http://127.0.0.1:18080"
    assert drill.check_url("http://127.0.0.1:18080/") == "http://127.0.0.1:18080"
    assert drill.check_url("127.0.0.1:18080") == "http://127.0.0.1:18080"
    assert drill.check_url(" https://example.org ") == "https://example.org"
    assert drill.check_url("http://localhost:18080/some/path") == "http://localhost:18080"
    assert drill.check_url("http://[::1]:18080") == "http://[::1]:18080"
    assert drill.check_url("HTTP://LocalHost:0800") == "http://localhost:800"


SECRET = "s3cr3t-T0KEN-value"


@pytest.mark.parametrize("argv", [
    ["http://127.0.0.1:18080", "--token", SECRET],                  # an option the script does not have
    ["http://127.0.0.1:18080", "--token=" + SECRET],
    ["http://127.0.0.1:18080", "--tok", SECRET],                    # an abbreviation of --token-file must not be taken for it
    ["http://127.0.0.1:18080", "--token-fil", SECRET],
    ["--token", SECRET, "http://127.0.0.1:18080"],                  # the secret in the URL's place
    ["http://127.0.0.1:18080", SECRET],                             # the plan's first sketch: URL TOKEN
    ["http://127.0.0.1:18080", "-t", SECRET],
    ["http://127.0.0.1:18080", "--bearer", SECRET, "--password=" + SECRET],
])
def test_the_script_refuses_a_token_on_the_command_line_and_never_echoes_it(argv, offline):
    code, out, err = run_drill(argv)
    assert code == 2 and out == ""
    assert refusal_line(err) == drill.ARGV_REFUSAL, "one fixed sentence, which names no argument"
    assert "CHAT_TOKEN" in drill.ARGV_REFUSAL and "--token-file" in drill.ARGV_REFUSAL
    assert SECRET not in err


@pytest.mark.parametrize("argv", [[], ["--turns", "6"], ["--timeout", "5", "--queue-cap", "4"]])
def test_a_missing_url_is_a_usage_refusal_in_one_line(argv, offline):
    code, out, err = run_drill(argv)
    assert code == 2 and out == ""
    assert "URL" in refusal_line(err)


@pytest.mark.parametrize("argv", [["--help"], ["-h"], ["http://127.0.0.1:18080", "--help"]])
def test_help_goes_to_stdout_with_exit_0_needs_no_url_and_connects_to_nothing(argv, offline):
    code, out, err = run_drill(argv)
    assert code == 0 and err == "" and out.startswith("usage: ")
    assert "--token-file" in out and "CHAT_TOKEN" in out and "8000" in out and "--queue-cap" in out


@pytest.mark.parametrize("flag, value", [("--turns", "0"), ("--turns", "-1"), ("--turns", "33"), ("--turns", "many"),
                                         ("--queue-cap", "0"), ("--queue-cap", "33"), ("--max-new-tokens", "0"),
                                         ("--max-new-tokens", "5"), ("--max-new-tokens", "201"), ("--timeout", "0"),
                                         ("--timeout", "-5"), ("--timeout", "soon"), ("--timeout", "nan")])
def test_a_number_out_of_range_is_refused_in_one_line_before_anything_is_sent(flag, value, offline):
    code, out, err = run_drill(["http://127.0.0.1:18080", flag, value])
    assert code == 2 and out == ""
    line = refusal_line(err)
    assert flag in line
    if not value.lstrip("-").isdigit():
        assert value not in line, "a refusal does not quote the value it was given"


# ---- the token: CHAT_TOKEN or --token-file, sent as a bearer -------------------------------------------------------------------------

TOKEN = "t0k-for-the-drill"


def write_token_file(tmp_path, text=TOKEN + "\n", mode=0o600, name="chat.token"):
    path = tmp_path / name
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(text if isinstance(text, bytes) else text.encode())
    os.chmod(str(path), mode)
    return path


@pytest.fixture
def protected(tmp_path):
    """A live server that wants a token."""
    home = tmp_path / "home"
    home.mkdir()
    base, stop = start_live_server(create_app(engine="tiny", home=str(home), token=TOKEN, queue_cap=4))
    yield base
    stop()


def test_the_token_in_chat_token_goes_out_as_a_bearer(protected, monkeypatch):
    monkeypatch.setenv("CHAT_TOKEN", TOKEN)
    code, out, err = run_drill([protected] + ONE_TURN)
    assert code == 0 and err == "" and the_result(out)["ok"] is True
    assert TOKEN not in out


def test_the_token_in_a_token_file_goes_out_as_a_bearer_and_wins_over_chat_token(protected, monkeypatch, tmp_path):
    path = write_token_file(tmp_path)
    monkeypatch.setenv("CHAT_TOKEN", "the-wrong-token")
    code, out, err = run_drill([protected, "--token-file", str(path)] + ONE_TURN)
    assert code == 0 and err == "" and the_result(out)["ok"] is True
    monkeypatch.delenv("CHAT_TOKEN")
    assert run_drill([protected, "--token-file", str(path)] + ONE_TURN)[0] == 0


def test_a_server_that_wants_a_token_the_drill_was_not_given_answers_401_and_the_drill_fails_with_exit_1(protected, monkeypatch):
    monkeypatch.delenv("CHAT_TOKEN", raising=False)
    code, out, err = run_drill([protected] + ONE_TURN)
    result = the_result(out)
    assert code == 1 and err == "" and result["ok"] is False and result["accepted"] == 0
    assert result["error"] == "session_create_http_401" and result["session_deleted"] is None   # no session was made, so none is left
    monkeypatch.setenv("CHAT_TOKEN", "the-wrong-token")
    result = the_result(run_drill([protected] + ONE_TURN, expect_code=1)[1])
    assert result["error"] == "session_create_http_401" and "the-wrong-token" not in json.dumps(result)


def test_in_public_mode_the_drill_sends_its_client_id_on_every_request_and_leaves_no_session_of_that_client(tmp_path, monkeypatch):
    base, stop = start_live_server(create_app(engine="tiny", home=str(tmp_path), mode="public", token=TOKEN, queue_cap=4))
    monkeypatch.setenv("CHAT_TOKEN", TOKEN)
    try:
        code, out, _ = run_drill([base, "--turns", "2", "--queue-cap", "2"] + FAST)
        assert code == 0 and the_result(out)["ok"] is True
        mine = {"Authorization": "Bearer " + TOKEN, "X-Client-Id": drill.CLIENT_ID}
        assert httpx.get(base + "/v1/sessions", headers=mine).json()["sessions"] == []
        assert httpx.get(base + "/v1/sessions", headers=dict(mine, **{"X-Client-Id": "someone-else"})).json()["sessions"] == []
    finally:
        stop()


def test_the_drill_connects_straight_to_the_url_whatever_proxy_the_environment_names(tmp_path, monkeypatch):
    """httpx trusts HTTP_PROXY and ALL_PROXY by default, loopback included, and the token and the stream would pass through that proxy. Here the
    environment names a proxy that is not there, and the drill must not notice: it goes straight to the URL it was given."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        dead = "http://127.0.0.1:{}".format(s.getsockname()[1])   # a free port, and nothing listens on it once the socket is closed
    base, stop = start_live_server(create_app(engine="tiny", home=str(tmp_path), queue_cap=4))
    try:
        for name in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
            monkeypatch.setenv(name, dead)
        for name in ("NO_PROXY", "no_proxy"):
            monkeypatch.delenv(name, raising=False)
        code, out, err = run_drill([base] + ONE_TURN)
    finally:
        stop()
    assert code == 0 and err == "" and the_result(out)["ok"] is True


def test_the_client_id_is_one_fixed_value_the_server_takes():
    assert drill.CLIENT_ID == "chat-load-drill" and CLIENT_ID.fullmatch(drill.CLIENT_ID)


TOKEN_FILE_CASES = [
    ("good", b"abc123\n", 0o600), ("no newline", b"abc123", 0o600), ("padded", b"  abc123 \n\n", 0o600),
    ("group readable", b"abc123\n", 0o640), ("world readable", b"abc123\n", 0o644), ("read only", b"abc123\n", 0o400),
    ("owner x", b"abc123\n", 0o700), ("empty", b"", 0o600), ("blank", b" \n\n", 0o600), ("non ascii", "tokén".encode(), 0o600),
    ("space inside", b"ab cd\n", 0o600), ("two lines", b"ab\ncd\n", 0o600), ("too big", b"a" * 5000, 0o600), ("control char", b"ab\x01cd", 0o600),
    ("at the size limit", b"a" * 4096, 0o600),
]


def verdict(read, path):
    try:
        return ("ok", read(str(path)))
    except (drill.Refused, cli.TokenFileError):
        return ("refused", None)


@pytest.mark.parametrize("label, data, mode", TOKEN_FILE_CASES, ids=[c[0] for c in TOKEN_FILE_CASES])
def test_the_token_file_policy_is_the_servers_own(label, data, mode, tmp_path):
    """The script reads a token file as the server's command line does (app/cli.py read_token): a regular file of mode 0600 that you own, visible
    ASCII without spaces, 4096 bytes at most. The two are kept apart (the script imports nothing of the app) and held to one table here."""
    path = write_token_file(tmp_path, data, mode)
    assert verdict(drill.read_token, path) == verdict(cli.read_token, path), label


def test_the_token_file_policy_agrees_on_a_missing_path_a_directory_a_foreign_owner_and_a_home_relative_path(tmp_path, monkeypatch):
    path = write_token_file(tmp_path)
    assert verdict(drill.read_token, tmp_path / "missing") == verdict(cli.read_token, tmp_path / "missing") == ("refused", None)
    assert verdict(drill.read_token, tmp_path) == verdict(cli.read_token, tmp_path) == ("refused", None)
    assert verdict(drill.read_token, path) == verdict(cli.read_token, path) == ("ok", TOKEN)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert drill.read_token("~/chat.token") == cli.read_token("~/chat.token") == TOKEN
    monkeypatch.setattr(os, "getuid", lambda: os.stat(str(path)).st_uid + 1)   # somebody else's file
    assert verdict(drill.read_token, path) == verdict(cli.read_token, path) == ("refused", None)


def test_a_refused_token_file_says_why_without_naming_the_file_or_what_is_in_it(tmp_path, offline):
    wide = write_token_file(tmp_path, SECRET, 0o644, "wide.token")
    code, out, err = run_drill(["http://127.0.0.1:18080", "--token-file", str(wide)])
    assert code == 2 and out == "" and "0600" in refusal_line(err)
    assert SECRET not in err and "wide.token" not in err and str(tmp_path) not in err
    empty = write_token_file(tmp_path, "", 0o600, "empty.token")
    code, out, err = run_drill(["http://127.0.0.1:18080", "--token-file", str(empty)])
    assert code == 2 and "empty" in refusal_line(err) and "empty.token" not in err
    code, out, err = run_drill(["http://127.0.0.1:18080", "--token-file", str(tmp_path / "nope.token")])
    assert code == 2 and refusal_line(err) and "nope.token" not in err


def test_a_chat_token_that_is_not_visible_ascii_is_refused_without_being_quoted(monkeypatch, offline):
    for bad in ("has a space", "tokén", "tab\there", SECRET + " " + SECRET):
        monkeypatch.setenv("CHAT_TOKEN", bad)
        code, out, err = run_drill(["http://127.0.0.1:18080"])
        assert code == 2 and out == "" and "CHAT_TOKEN" in refusal_line(err)
        assert SECRET not in err and bad not in err
    monkeypatch.setenv("CHAT_TOKEN", "")   # empty is the same as unset: no token is sent
    assert drill.find_token(None) is None
    monkeypatch.setenv("CHAT_TOKEN", "  " + TOKEN + "  ")
    assert drill.find_token(None) == TOKEN


# ---- a token is not sent in the clear to another machine (P8-C fix) -----------------------------------------------------------------

@pytest.mark.parametrize("url", ["http://example.org:18080", "http://10.0.0.5:18080", "http://192.168.1.20", "http://[2001:db8::1]:18080",
                                 "http://[::ffff:10.0.0.5]:18080", "http://0.0.0.0:18080", "http://localhost.example.org:18080",
                                 "http://127.0.0.1.example.org:18080", "http://127.1:18080", "http://2130706433:18080", "http://node-07.internal",
                                 "example.org:18080", "http://128.0.0.1:18080", "http://127.0.0.1x:18080"])
def test_a_token_is_not_sent_over_plain_http_to_a_host_that_is_not_loopback(url, offline, monkeypatch):
    """One fixed line, exit 2, nothing sent. What cannot be shown to be loopback is not: a name that happens to resolve there, 127.1 and the integer
    form of 127.0.0.1 included, is refused, because this decides where a bearer token may travel readable."""
    monkeypatch.setenv("CHAT_TOKEN", SECRET)
    code, out, err = run_drill([url])
    assert code == 2 and out == "" and refusal_line(err) == drill.CLEARTEXT_REFUSAL
    assert SECRET not in err and "example" not in err and "10.0.0.5" not in err and "internal" not in err


def test_the_refusal_to_send_a_token_in_the_clear_holds_for_a_token_file_too(tmp_path, offline):
    path = write_token_file(tmp_path)
    code, out, err = run_drill(["http://example.org:18080", "--token-file", str(path)])
    assert code == 2 and out == "" and refusal_line(err) == drill.CLEARTEXT_REFUSAL and TOKEN not in err


@pytest.mark.parametrize("url", ["http://localhost:18080", "http://LOCALHOST", "http://127.0.0.1:18080", "http://127.9.9.9:18080",
                                 "http://[::1]:18080", "http://[::ffff:127.0.0.1]:18080", "https://example.org", "https://10.0.0.5:18080",
                                 "https://[2001:db8::1]:18080", "https://localhost:18080", "HTTPS://Example.ORG:8443"])
def test_a_token_goes_to_loopback_over_http_and_to_any_host_over_https(url, monkeypatch):
    """Past the check, the drill asks for its session, and nothing answers (Unreachable): that is how this test sees it was not refused."""
    monkeypatch.setenv("CHAT_TOKEN", SECRET)
    monkeypatch.setattr(httpx, "Client", Unreachable)
    code, out, err = run_drill([url])
    assert code == 1 and err == "" and the_result(out)["error"] == "session_create_ConnectError"
    assert SECRET not in out


def test_without_a_token_a_plain_http_url_to_any_host_is_not_refused(monkeypatch):
    monkeypatch.setattr(httpx, "Client", Unreachable)
    for url in ("http://example.org:18080", "http://10.0.0.5:18080", "http://[2001:db8::1]"):
        code, out, err = run_drill([url])
        assert code == 1 and err == "" and the_result(out)["error"] == "session_create_ConnectError", url


def test_is_loopback_is_true_for_localhost_127_0_0_0_8_and_ipv6_loopback_only():
    assert [drill.is_loopback(h) for h in ("localhost", "LocalHost", "127.0.0.1", "127.255.255.254", "::1", "::ffff:127.0.0.1", "::ffff:7f00:1")] == [True] * 7
    for host in ("", "localhost.", "127.0.0.1.example.org", "128.0.0.1", "126.255.255.255", "0.0.0.0", "10.0.0.5", "::", "::2", "::ffff:10.0.0.5",
                 "127.1", "2130706433", "0x7f.0.0.1", "127.0.0.01", "example.org", "fe80::1%eth0"):
        assert drill.is_loopback(host) is False, host


# ---- it leaves nothing behind -----------------------------------------------------------------------------------------------------

def test_a_drill_that_fails_inside_still_deletes_its_one_session(tmp_path, monkeypatch):
    base, stop = start_live_server(create_app(engine="tiny", home=str(tmp_path), queue_cap=4))

    def broken(*args, **kwargs):
        raise RuntimeError("boom, with text that must not be printed")
    monkeypatch.setattr(drill, "run_turns", broken)
    try:
        code, out, err = run_drill([base] + FAST)
        sessions = httpx.get(base + "/v1/sessions").json()["sessions"]
    finally:
        stop()
    result = the_result(out)
    assert code == 1 and err == "" and result["ok"] is False and result["session_deleted"] is True
    assert result["error"] == "internal_RuntimeError" and "boom" not in out
    assert sessions == []


def test_an_interrupted_drill_still_deletes_its_one_session_and_exits_1(tmp_path, monkeypatch):
    base, stop = start_live_server(create_app(engine="tiny", home=str(tmp_path), queue_cap=4))

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt
    monkeypatch.setattr(drill, "run_turns", interrupted)
    try:
        code, out, err = run_drill([base] + FAST)
        sessions = httpx.get(base + "/v1/sessions").json()["sessions"]
    finally:
        stop()
    result = the_result(out)
    assert code == 1 and err == "" and result["error"] == "interrupted" and result["session_deleted"] is True and result["ok"] is False
    assert sessions == []


def test_a_drill_that_runs_out_of_time_deletes_its_session_which_stops_the_turns_it_started(tmp_path):
    base, stop = start_live_server(create_app(engine="tiny", home=str(tmp_path), queue_cap=4, tiny_step_delay_s=0.2))
    try:
        code, out, err = run_drill([base, "--turns", "2", "--queue-cap", "4", "--max-new-tokens", "100", "--timeout", "2"])
        assert httpx.get(base + "/v1/sessions").json()["sessions"] == []
        # the server ends a deleted chat's turns at their next step (P4-H), so nothing is left running there
        wait_until(lambda: httpx.get(base + "/healthz").json()["turns_in_flight"] == 0, timeout=15)
    finally:
        stop()
    result = the_result(out)
    assert code == 1 and err == "" and result["error"] == "timeout" and result["session_deleted"] is True and result["ok"] is False
    assert result["accepted"] == 2 and result["done"] == 0


def test_a_session_that_cannot_be_deleted_is_reported_and_fails_the_drill(tmp_path):
    base, stop = start_live_server(create_app(engine="tiny", home=str(tmp_path), queue_cap=4))
    try:
        with mock.patch.object(Store, "delete_session", autospec=True, return_value=False):   # the server answers 404
            code, out, err = run_drill([base] + ONE_TURN)
    finally:
        stop()
    result = the_result(out)
    assert code == 1 and err == "" and result["session_deleted"] is False and result["ok"] is False
    assert result["accepted"] == 1 and result["done"] == 1   # the turn itself was fine


def test_a_server_that_cannot_be_reached_ends_in_one_result_line_with_the_class_of_the_failure_only(monkeypatch):
    monkeypatch.setattr(httpx, "Client", Unreachable)
    code, out, err = run_drill(["http://127.0.0.1:18080"])
    result = the_result(out)
    assert code == 1 and err == "" and result["error"] == "session_create_ConnectError" and "secret" not in out
    assert result["accepted"] == 0 and result["session_deleted"] is None and result["ok"] is False


# ---- R7: nothing of the stream is printed or saved ---------------------------------------------------------------------------------

def leaves(value):
    """Every string in a parsed frame."""
    if isinstance(value, dict):
        for item in value.values():
            yield from leaves(item)
    elif isinstance(value, list):
        for item in value:
            yield from leaves(item)
    elif isinstance(value, str):
        yield value


def test_the_script_prints_and_saves_nothing_of_what_the_server_streams(tmp_path, monkeypatch):
    """The tiny gallery makes the stream carry what a private server's carries: the generated report, the matched reports and the neighbours'
    image urls. The test reads the very bytes the script read, finds every report in them, and looks for each in everything the script wrote
    (stdout and stderr), and in the directory it ran in (nothing was created)."""
    workdir = tmp_path / "cwd"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    base, stop = start_live_server(create_app(engine="tiny", home=str(tmp_path / "home"), tiny_gallery=True, queue_cap=4))
    seen = []
    real = drill.read_stream

    def tee(chunks):
        def copying():
            for chunk in chunks:
                seen.append(chunk)
                yield chunk
        return real(copying())
    monkeypatch.setattr(drill, "read_stream", tee)
    try:
        code, out, err = run_drill([base, "--turns", "2", "--queue-cap", "2"] + FAST)
    finally:
        stop()
    assert code == 0, (out, err)
    frames = list(iter_sse(["".join(seen)]))
    reports = {s for f in frames for s in leaves(f["data"]) if len(s) >= 20 and " " in s}
    reports |= {s for f in frames if f["event"] == "stage_end" for n in f["data"].get("detail", {}).get("image_neighbors", [])
                for s in leaves(n)}
    assert len(reports) >= 4, "the stream carried reports to look for"
    assert any(len(s) >= 40 for s in reports) and any(s.startswith("/v1/gallery/images/") for s in reports)
    everything = out + err
    for secret in reports:
        assert secret not in everything, secret[:30]
    for word in ("Findings", "Impression", "heart", "lungs", "effusion", "image_url", "gallery"):
        assert word not in everything, word
    assert list(workdir.iterdir()) == [], "the script wrote no file"
