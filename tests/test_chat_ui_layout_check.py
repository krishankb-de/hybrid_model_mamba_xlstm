"""CHAT_UI_PLAN.md P4-E: the Chrome harness the two browser checks share (scripts/chat_ui_cdp.py) and the parts of the layout check
(scripts/chat_ui_layout_check.py) that are plain Python. No Chrome and no app: a fake browser is a shell script, or an object
that returns canned measurements.

The harness bugs these pin were found by the P4-A re-review: a Chrome orphaned when the DevTools connection failed after it
started, `display: none` mutants of #stop and #drawer that passed, and a report floor that could be deleted unseen. The P4-E
follow-up (the final review's carried items) adds: a close() that an interrupt cuts short still stops Chrome, the websocket's send
keeps the deadline, the SIGTERM handler is one-shot, what a Chrome leaves in its session when it goes is swept, and the fakes'
clean-up fixture is proven to call its clean-up. P6-F fix 1 adds the refusals of a --url that the three checks share (the dev server's
port and the tunnel's, any host that is not this machine's loopback, an app that is not the tiny one), and each check's main() refusing
with them (tests/chat_ui_url_cases.py).
"""
import importlib
import json
import os
import signal
import socket
import struct
import subprocess
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from scripts import chat_ui_layout_check as layout
from scripts import chat_ui_cdp
from scripts.chat_ui_cdp import Browser, kill_group
from tests import chat_ui_url_cases as url_cases


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:   # someone else's process now: not ours any more
        return False
    return True


def _until(predicate, seconds: float = 5.0) -> bool:
    deadline = time.time() + seconds
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


def _record_killpg(monkeypatch) -> List[Any]:
    """Every os.killpg the harness makes is recorded, not delivered."""
    sent = []   # type: List[Any]
    monkeypatch.setattr(chat_ui_cdp.os, "killpg", lambda pgid, sig: sent.append((pgid, sig)))
    return sent


# ---- the harness ------------------------------------------------------------------------------------------------------------------

def _fake_chrome(tmp_path: Path) -> Path:
    """A "Chrome" that starts, records its pid and profile directory, and never opens a DevTools page."""
    script = tmp_path / "chrome"
    script.write_text('#!/bin/sh\necho $$ > "{pid}"\nfor a in "$@"; do case "$a" in --user-data-dir=*) '
                      'echo "${{a#--user-data-dir=}}" > "{profile}";; esac; done\nexec sleep 30\n'.format(
                          pid=tmp_path / "pid", profile=tmp_path / "profile"))
    script.chmod(0o755)
    return script


def test_a_chrome_that_never_opens_devtools_is_killed_and_its_profile_removed(tmp_path):
    chrome = _fake_chrome(tmp_path)
    with pytest.raises(RuntimeError, match="did not open a DevTools page"):
        Browser(str(chrome), startup_timeout=1.0)
    pid = int((tmp_path / "pid").read_text())
    profile = Path((tmp_path / "profile").read_text().strip())
    assert not _alive(pid), "the Chrome that failed to start is still running"
    assert not profile.exists(), "its profile directory was left behind"


def test_a_devtools_connection_that_fails_after_chrome_started_closes_chrome(tmp_path):
    """The P4-A re-review's reproduction: Chrome is up and lists its page, the websocket to it then fails. The Chrome that started
    must not be left running, nor its profile directory behind."""
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(5)

    def hang_up() -> None:
        while True:
            try:
                connection, _ = listener.accept()
            except OSError:
                return
            connection.close()   # accepts the connection and never answers the handshake

    page = json.dumps([{"type": "page", "webSocketDebuggerUrl": "ws://127.0.0.1:{}/devtools/page/1".format(listener.getsockname()[1])}]).encode()

    class ListPages(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Length", str(len(page)))
            self.end_headers()
            self.wfile.write(page)

        def log_message(self, *args: Any) -> None:
            pass

    http = HTTPServer(("127.0.0.1", 0), ListPages)
    threads = [threading.Thread(target=hang_up, daemon=True), threading.Thread(target=http.serve_forever, daemon=True)]
    for thread in threads:
        thread.start()
    script = tmp_path / "chrome"
    script.write_text('#!/bin/sh\necho $$ > "{pid}"\nfor a in "$@"; do case "$a" in --user-data-dir=*) d="${{a#--user-data-dir=}}"; '
                      'echo "$d" > "{profile}"; echo {port} > "$d/DevToolsActivePort";; esac; done\nexec sleep 30\n'.format(
                          pid=tmp_path / "pid", profile=tmp_path / "profile", port=http.server_address[1]))
    script.chmod(0o755)
    try:
        with pytest.raises(ConnectionError):   # a reset or a clean close, whichever the peer's hang-up turns out to be
            Browser(str(script), startup_timeout=10.0)
        pid = int((tmp_path / "pid").read_text())
        assert not _alive(pid), "the Chrome that started is still running after the connection failed"
        assert not Path((tmp_path / "profile").read_text().strip()).exists()
    finally:
        http.shutdown()
        http.server_close()
        listener.close()


def test_chrome_is_started_without_the_default_apps_extensions_and_background_work_it_does_in_a_fresh_profile(tmp_path):
    """P6-F. A fresh profile is not a quiet one: some seconds after Chrome starts, its log shows its default-app and extension installer and a
    background check-in at work, and a fetch that a page makes around then can fail inside Chrome (TypeError: Failed to fetch, net::ERR_ABORTED,
    canceled; no request in the NetLog, nothing at the server). The features check's picker clicks then (at about 7 s), and failed about one run in
    three with the default flags (10 of 30) and none with these five together (0 of 30, runs alternating); which of the five matters is not
    established, and neither is that the activity causes the failure. The test reads the arguments the process was really started with."""
    script = tmp_path / "chrome"
    script.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "{}"\nexec sleep 30\n'.format(tmp_path / "argv"))
    script.chmod(0o755)
    with pytest.raises(RuntimeError, match="did not open a DevTools page"):   # this Chrome never opens one: it is the arguments that are looked at
        Browser(str(script), startup_timeout=1.0)
    started_with = (tmp_path / "argv").read_text().split("\n")
    for flag in ("--disable-extensions", "--disable-default-apps", "--disable-background-networking", "--disable-component-update", "--disable-sync"):
        assert flag in started_with, "Chrome was started without {}: {}".format(flag, started_with)
    assert "--headless=new" in started_with and "--remote-debugging-port=0" in started_with   # and what it always had
    assert any(a.startswith("--user-data-dir=") for a in started_with) and started_with[-2] == "about:blank"


def test_a_chrome_that_exits_at_once_leaves_no_profile_directory(tmp_path):
    script = tmp_path / "chrome"
    script.write_text('#!/bin/sh\nfor a in "$@"; do case "$a" in --user-data-dir=*) echo "${a#--user-data-dir=}" > "%s";; esac; done\nexit 3\n'
                      % (tmp_path / "profile"))
    script.chmod(0o755)
    with pytest.raises(RuntimeError, match="exited with status 3"):
        Browser(str(script), startup_timeout=5.0)
    assert not Path((tmp_path / "profile").read_text().strip()).exists()


def test_kill_group_takes_the_children_with_it(tmp_path):
    child_file = tmp_path / "child"
    proc = subprocess.Popen(["/bin/sh", "-c", 'sleep 30 & echo $! > "{}"; wait'.format(child_file)], start_new_session=True)
    assert _until(lambda: child_file.exists() and child_file.read_text().strip())
    child = int(child_file.read_text())
    assert _alive(child)
    kill_group(proc)
    assert proc.returncode is not None
    assert _until(lambda: not _alive(child)), "the child of the process outlived it"


def test_kill_group_is_safe_on_nothing_and_on_a_process_that_has_gone(monkeypatch):
    sent = _record_killpg(monkeypatch)
    kill_group(None)
    proc = subprocess.Popen(["/bin/sh", "-c", "exit 0"], start_new_session=True)
    proc.wait()
    kill_group(proc)   # reaped already: no signal goes to a pid that may be someone else's
    kill_group(proc)
    assert sent == [], "a signal went to the group of a process that was already reaped: {}".format(sent)


class ScriptedSocket:
    """The DevTools end of a Browser: what it is sent is recorded; what it receives is scripted, per call, as a list of messages."""

    def __init__(self, replies: Any) -> None:
        self.replies, self.sent = list(replies), []
        self.incoming = []   # type: Any

    def send(self, text: str) -> None:
        message = json.loads(text)
        self.sent.append(message)
        self.incoming = [json.dumps(m(message["id"]) if callable(m) else m) for m in self.replies.pop(0)]

    def recv(self) -> str:
        return self.incoming.pop(0)

    def close(self) -> None:
        self.closed = True


def _browser(replies: Any) -> Browser:
    browser = Browser.__new__(Browser)   # no Chrome: only the call/events plumbing is under test
    browser.ws, browser.next_id = ScriptedSocket(replies), 0
    browser.events_seen = deque(maxlen=chat_ui_cdp.KEEP_EVENTS)
    return browser


def test_events_that_arrive_while_a_call_waits_are_kept_and_given_out_once_by_method():
    log = {"method": "Runtime.consoleAPICalled", "params": {"type": "error"}}
    net = {"method": "Network.loadingFailed", "params": {"requestId": "7", "canceled": True}}
    browser = _browser([[log, net, lambda i: {"id": i, "result": {"x": 1}}], [lambda i: {"id": i, "result": {}}]])
    assert browser.call("Runtime.evaluate") == {"x": 1}
    assert browser.events("Network.loadingFailed", clear=False) == [net["params"]]   # peeked, still there
    assert browser.events("Network.loadingFailed") == [net["params"]]                # taken
    assert browser.events("Network.loadingFailed") == []                             # and gone
    assert browser.events("Runtime.consoleAPICalled") == [log["params"]]             # another method's events were not taken with it
    browser.call("Runtime.evaluate")


def test_an_error_answer_raises_with_the_method_and_the_event_record_is_bounded():
    browser = _browser([[lambda i: {"id": i, "error": {"code": -32000, "message": "no such node"}}]])
    with pytest.raises(RuntimeError, match=r"DOM.querySelector failed: .*no such node"):
        browser.call("DOM.querySelector", selector="#x")
    noise = [{"method": "Network.dataReceived", "params": {"n": n}} for n in range(chat_ui_cdp.KEEP_EVENTS + 50)]
    browser = _browser([noise + [lambda i: {"id": i, "result": {}}]])
    browser.call("Page.enable")
    kept = browser.events("Network.dataReceived")
    assert len(kept) == chat_ui_cdp.KEEP_EVENTS and kept[0] == {"n": 50}   # the oldest went first


# ---- stopping what was started: politely first, SIGKILL last ------------------------------------------------------------------------------------

FAKE_STEPS = 300   # a fake waits this many 50 ms steps and ends by itself: 15 s, about 18 s with the forks (measured). One whose test
                   # never got to stop it, because pytest was itself SIGKILLed and no finalizer ran, goes soon after


def _fake(on_term: str, steps: int = FAKE_STEPS) -> str:
    """A fake Chrome or app, as the text of a shell script. The TERM trap comes first and the marker directory second: a test that waits
    for the marker then knows the trap is in place (a TERM that comes earlier kills a shell that has none, at once). Then it waits, for
    at most `steps` x 50 ms. on_term is the trap's action: a TERM that is handled removes the marker and exits 0, as Chrome removes its
    singleton directory; one that is ignored (an empty action) leaves the marker, as a Chrome that hangs would."""
    return ('#!/bin/sh\nmarker="$1"\ntrap \'{on_term}\' TERM\nmkdir -p "$marker"\n'
            'i=0\nwhile [ "$i" -lt {steps} ]; do sleep 0.05; i=$((i + 1)); done\n').format(on_term=on_term, steps=steps)


POLITE = _fake('rm -rf "$marker"; exit 0')
STUBBORN = _fake('')

_STARTED = []   # type: List[subprocess.Popen]   # what _start has started and the end of its test has not yet seen to
_REAL_KILLPG = os.killpg   # kept from before any test patches os.killpg: the clean-up below must not be one of its victims


def _start(tmp_path: Path, body: str, name: str = "fake", env: Optional[Dict[str, str]] = None, ready: Optional[float] = 5.0) -> Any:
    """Starts a fake (see _fake) in a session of its own and returns it with its marker directory. ready is how long to wait for the
    marker, in seconds (None: do not wait). The fake is registered before anything can fail, so that the clean-up below finds it."""
    script = tmp_path / name
    script.write_text(body)
    script.chmod(0o755)
    marker = tmp_path / (name + ".marker")
    proc = subprocess.Popen(["/bin/sh", str(script), str(marker)], start_new_session=True, env=env)
    _STARTED.append(proc)
    if ready is not None:
        assert _until(marker.exists, ready), "the fake process did not start"
    return proc, marker


def _kill_the_fakes() -> None:
    """SIGKILL, with its group, every fake that is still running, and forget them all."""
    while _STARTED:
        proc = _STARTED.pop()
        if proc.poll() is None:   # not reaped yet, so the pid is still this process's own
            try:
                _REAL_KILLPG(proc.pid, signal.SIGKILL)
            except OSError:
                pass
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass


@pytest.fixture(autouse=True)
def _no_fake_outlives_its_test():
    """The fakes wait, and STUBBORN ignores TERM: a test that failed before it stopped its fake used to leave it running (a shell that
    forks a sleep twenty times a second, for ever). Whatever a test started and did not see to is killed when the test ends."""
    yield
    _kill_the_fakes()


def _bare_browser(proc: Any, profile: Path, ws: Any = None) -> Browser:
    browser = Browser.__new__(Browser)   # no Chrome: close() is under test, not the start
    browser.proc, browser.ws, browser.profile, browser.next_id = proc, ws, str(profile), 0
    browser.events_seen = deque(maxlen=chat_ui_cdp.KEEP_EVENTS)
    profile.mkdir()
    return browser


@pytest.mark.parametrize("stall", [0.0, 1.5], ids=["prompt", "late"])   # seconds the test is held up after it sees the shim, as on a loaded machine
@pytest.mark.parametrize("body, handles_term", [(POLITE, True), (STUBBORN, False)], ids=["polite", "stubborn"])
def test_a_fake_has_its_trap_in_place_before_it_makes_its_marker(tmp_path, body, handles_term, stall):
    """A test waits for the marker before it sends TERM, so the trap has to be installed first. mkdir is slowed here by a shim on PATH
    and the TERM comes while it runs: a fake with its trap in place meets it (polite: handles it and exits 0 once mkdir is done;
    stubborn: ignores it and goes on); one without a trap dies of it at once (-15). The window is held open by the test and not by the
    clock (P4-E follow-up): the shim runs mkdir once the test has sent the TERM, however long that takes it (the "late" cases hold the test
    up for longer than the shim's old fixed second; the shim gives up after 10 s, so a failed test leaves nothing looping)."""
    shim = tmp_path / "bin"
    shim.mkdir()
    entered, released = tmp_path / "entered", tmp_path / "released"
    (shim / "mkdir").write_text('#!/bin/sh\n: > "{}"\ni=0\nwhile [ ! -e "{}" ] && [ "$i" -lt 200 ]; do sleep 0.05; i=$((i + 1)); done\n'
                                'exec /bin/mkdir "$@"\n'.format(entered, released))
    (shim / "mkdir").chmod(0o755)
    proc, marker = _start(tmp_path, body, env=dict(os.environ, PATH=str(shim) + os.pathsep + os.environ["PATH"]), ready=None)
    try:
        assert _until(entered.exists, 15), "the fake never reached mkdir"
        time.sleep(stall)
        assert not marker.exists()   # the marker does not exist yet: that is the window
        proc.send_signal(signal.SIGTERM)
    finally:
        released.touch()   # mkdir may go on, whatever became of the test
    if handles_term:
        assert proc.wait(timeout=15) == 0, "the TERM found no trap"
    else:
        assert _until(marker.exists, 15), "the TERM was fatal to a fake that ignores it: its trap came too late"
        assert proc.poll() is None


def test_a_fake_ends_by_itself(tmp_path):
    """Its wait is bounded: one whose test never got to stop it goes by itself (the finalizer cannot run in a pytest that is SIGKILLed)."""
    assert FAKE_STEPS * 0.05 * 1.5 <= 30   # at most 30 s, even where the forks of the sleeps cost half as much again
    proc, marker = _start(tmp_path, _fake("", steps=4), "short")
    assert proc.wait(timeout=5) == 0   # nobody sent it anything
    assert marker.exists()


def test_the_clean_up_kills_a_fake_that_ignores_term_and_forgets_it(tmp_path):
    proc, _ = _start(tmp_path, STUBBORN)
    _kill_the_fakes()
    assert proc.returncode == -signal.SIGKILL
    assert _STARTED == []


def test_a_fake_that_never_comes_up_is_cleaned_up_too(tmp_path):
    with pytest.raises(AssertionError, match="did not start"):
        _start(tmp_path, "#!/bin/sh\nexec sleep 20\n", "mute", ready=0.2)   # it never makes its marker
    (proc,) = _STARTED   # registered before the wait that failed
    assert proc.poll() is None
    _kill_the_fakes()
    assert proc.returncode == -signal.SIGKILL


def test_the_clean_up_is_wired_in_for_every_test_of_this_file(request):
    assert "_no_fake_outlives_its_test" in request.fixturenames


def test_the_autouse_fixture_calls_kill_the_fakes_when_its_test_ends(monkeypatch):
    """P4-E follow-up. The test above sees the fixture named for every test; this one runs it, so that a fixture that is named and
    does nothing (a deleted call) is caught: nothing is killed while the test runs, and the end of the test kills."""
    fixture = getattr(_no_fake_outlives_its_test, "__wrapped__", _no_fake_outlives_its_test)   # pytest wraps a fixture's function
    called = []
    monkeypatch.setitem(globals(), "_kill_the_fakes", lambda: called.append("killed"))
    steps = fixture()
    next(steps)
    assert called == [], "the fixture killed the fakes before the test had run"
    with pytest.raises(StopIteration):
        next(steps)
    assert called == ["killed"]


def test_stop_group_asks_politely_first(tmp_path):
    proc, marker = _start(tmp_path, POLITE)
    started = time.time()
    chat_ui_cdp.stop_group(proc)
    assert proc.returncode == 0, "it was killed ({}), not asked".format(proc.returncode)   # the trap ran and exited; a SIGKILL says -9
    assert not marker.exists()
    assert time.time() - started < 3


def test_stop_group_kills_what_ignores_term_once_the_wait_is_over(tmp_path):
    proc, marker = _start(tmp_path, STUBBORN)
    started = time.time()
    chat_ui_cdp.stop_group(proc, term_wait=0.5)
    took = time.time() - started
    assert proc.returncode == -signal.SIGKILL
    assert marker.exists()   # it never got to clean up: that is what SIGKILL costs, and why it is the last resort
    assert 0.5 <= took < 4, took


def test_stop_group_is_safe_on_nothing_and_signals_nothing_after_the_process_was_reaped(monkeypatch):
    """Once its leader is reaped a group's pid may be anyone's: nothing may be signalled then, whatever the order of the calls. (A
    stop_group that signals without looking would send TERM and KILL to a stranger's group, and a returncode check cannot see that.)"""
    sent = _record_killpg(monkeypatch)
    chat_ui_cdp.stop_group(None)
    gone = subprocess.Popen(["/bin/sh", "-c", "exit 3"], start_new_session=True)
    gone.wait()
    chat_ui_cdp.stop_group(gone)
    chat_ui_cdp.stop_group(gone, term_wait=0.1)
    assert gone.returncode == 3   # untouched
    assert sent == [], "a signal went to the group of a process that was already reaped: {}".format(sent)


def test_the_killpg_patch_sees_what_stop_group_sends_to_a_process_that_is_still_there(monkeypatch):
    """The control for the two tests that assert `sent == []`: the same patch does record what is sent to a live process, TERM first and
    then, nothing having been delivered, KILL of its group (and the process is finished off by Popen.kill)."""
    sent = _record_killpg(monkeypatch)
    live = subprocess.Popen(["/bin/sh", "-c", "exec sleep 20"], start_new_session=True)
    _STARTED.append(live)
    chat_ui_cdp.stop_group(live, term_wait=0.1)
    assert sent == [(live.pid, signal.SIGTERM), (live.pid, signal.SIGKILL)]
    assert live.returncode == -signal.SIGKILL


def test_closing_a_browser_stops_chrome_politely_first(tmp_path):
    proc, marker = _start(tmp_path, POLITE, "chrome")
    started = time.time()
    _bare_browser(proc, tmp_path / "profile").close()
    assert proc.returncode == 0, "Chrome was killed ({}), not asked".format(proc.returncode)
    assert not marker.exists() and not (tmp_path / "profile").exists()
    assert time.time() - started < 3


def test_closing_a_browser_kills_a_chrome_that_ignores_term_within_the_wait(tmp_path, monkeypatch):
    monkeypatch.setattr(chat_ui_cdp, "TERM_WAIT_S", 0.5)
    proc, marker = _start(tmp_path, STUBBORN, "chrome")
    started = time.time()
    _bare_browser(proc, tmp_path / "profile").close()
    assert proc.returncode == -signal.SIGKILL
    assert not (tmp_path / "profile").exists()   # its profile goes whether or not it left politely
    assert 0.5 <= time.time() - started < 4


def test_closing_a_browser_asks_chrome_over_devtools_before_any_signal(tmp_path, monkeypatch):
    monkeypatch.setattr(chat_ui_cdp, "CLOSE_WAIT_S", 0.2)   # this Chrome does not leave on Browser.close: the signals follow
    proc, marker = _start(tmp_path, POLITE, "chrome")
    ws = ScriptedSocket([[lambda i: {"id": i, "result": {}}]])
    _bare_browser(proc, tmp_path / "profile", ws).close()
    assert [m["method"] for m in ws.sent] == ["Browser.close"]
    assert proc.returncode == 0 and not marker.exists()   # then TERM, not KILL


def test_closing_a_browser_that_drops_the_connection_instead_of_answering_still_closes(tmp_path, monkeypatch):
    monkeypatch.setattr(chat_ui_cdp, "CLOSE_WAIT_S", 0.2)

    class Hangs(ScriptedSocket):
        def recv(self) -> str:
            raise ConnectionError("DevTools closed the connection")

    proc, marker = _start(tmp_path, POLITE, "chrome")
    _bare_browser(proc, tmp_path / "profile", Hangs([[]])).close()
    assert proc.returncode == 0 and not (tmp_path / "profile").exists()


@pytest.mark.parametrize("where, interrupt", [("send", SystemExit(128 + signal.SIGTERM)), ("recv", KeyboardInterrupt())],
                         ids=["sigterm-in-the-request", "ctrl-c-in-the-wait-for-the-answer"])
def test_a_close_that_is_interrupted_in_its_devtools_step_still_stops_chrome_and_removes_the_profile(tmp_path, where, interrupt):
    """P4-E follow-up. The SIGTERM handler raises SystemExit wherever the main thread is, and Ctrl-C raises KeyboardInterrupt: neither is an
    Exception, so the DevTools step's `except Exception` lets it through, and the signals and the profile's removal that come after the step
    were skipped: Chrome was left running, its profile left behind. They are in a finally now; the interrupt still goes on."""
    class Interrupted(ScriptedSocket):
        def send(self, text: str) -> None:
            if where == "send":
                raise interrupt
            super().send(text)

        def recv(self) -> str:
            raise interrupt

    proc, marker = _start(tmp_path, POLITE, "chrome")
    ws = Interrupted([[lambda i: {"id": i, "result": {}}]])
    with pytest.raises(type(interrupt)):
        _bare_browser(proc, tmp_path / "profile", ws).close()
    assert proc.returncode == 0 and not marker.exists(), "Chrome was left running ({})".format(proc.returncode)
    assert not (tmp_path / "profile").exists(), "its profile directory was left behind"
    assert getattr(ws, "closed", False), "the DevTools connection was left open"


# A "Chrome" that leaves a helper in its session, as the real one has renderers and a GPU process, and goes by itself after `lifetime`
# seconds, as it does on a DevTools Browser.close. The helper's pid is written beside the marker, which exists once it is running.
def _leader(helper: str, lifetime: float = 0.3) -> str:
    return '#!/bin/sh\nmarker="$1"\nmkdir -p "$marker"\n{helper} &\necho $! > "$marker/helper"\nsleep {lifetime}\nexit 0\n'.format(helper=helper, lifetime=lifetime)


HELPERS = ["sleep 30", "(trap '' TERM; exec sleep 30)"]   # a helper that dies of TERM, and one that ignores it


def _helper_pid(marker: Path) -> int:
    assert _until(lambda: (marker / "helper").exists() and (marker / "helper").read_text().strip()), "the fake's helper did not start"
    pid = int((marker / "helper").read_text())
    assert _alive(pid)
    return pid


@pytest.mark.parametrize("helper", HELPERS, ids=["terminable", "ignores-term"])
def test_a_chrome_that_leaves_on_request_has_what_it_left_in_its_session_swept(tmp_path, helper):
    """P4-E follow-up. A Chrome that goes on Browser.close was never signalled, and what it left in its session (the helpers that outlive
    their browser) was never looked for: close() saw the browser gone and did nothing more. The session is swept (SIGKILL) once the
    leader has gone, while the leader is not yet reaped, so that its pid, which names the session, is still nobody else's."""
    proc, marker = _start(tmp_path, _leader(helper), "chrome")
    straggler = _helper_pid(marker)
    ws = ScriptedSocket([[lambda i: {"id": i, "result": {}}]])
    _bare_browser(proc, tmp_path / "profile", ws).close()
    assert [m["method"] for m in ws.sent] == ["Browser.close"]
    assert proc.returncode == 0, "the Chrome that went by itself was signalled before it went ({})".format(proc.returncode)
    assert _until(lambda: not _alive(straggler)), "a helper of the Chrome that left is still running"


def test_a_chrome_that_died_on_its_own_before_close_has_its_session_swept_too(tmp_path):
    """The same sweep when no DevTools connection is left to ask: the leader has gone (not reaped: nobody has waited for it), its helper
    has not."""
    if not hasattr(os, "waitid"):
        pytest.skip("os.waitid is what tells a leader that has gone from one that has gone and been reaped")
    proc, marker = _start(tmp_path, _leader("sleep 30", lifetime=0.1), "chrome")
    straggler = _helper_pid(marker)
    assert _until(lambda: os.waitid(os.P_PID, proc.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None), "the leader did not go"
    assert proc.returncode is None   # gone, and not reaped: the state of a Chrome that crashed while the check ran
    _bare_browser(proc, tmp_path / "profile").close()
    assert proc.returncode == 0
    assert _until(lambda: not _alive(straggler)), "a helper of the Chrome that had gone is still running"


def test_a_sweep_never_signals_a_session_whose_leader_was_reaped(tmp_path, monkeypatch):
    """The rule the sweep keeps (it is made before the reap, never after): a Chrome that was reaped, by whoever, is not signalled again."""
    sent = _record_killpg(monkeypatch)
    proc, marker = _start(tmp_path, _leader("sleep 30", lifetime=0.1), "chrome")
    helper = _helper_pid(marker)
    try:
        proc.wait(timeout=10)   # reaped: its pid may be anyone's now
        _bare_browser(proc, tmp_path / "profile").close()
        assert sent == [], "a signal went to the session of a leader that was already reaped: {}".format(sent)
    finally:
        try:
            os.kill(helper, signal.SIGKILL)   # nothing else will: with its leader reaped, the clean-up of the fakes does not look for it
        except ProcessLookupError:
            pass


def _server_frame(text: str, opcode: int = 0x1) -> bytes:
    """One unmasked websocket frame, as DevTools sends it: FIN set, a payload under 64 KiB."""
    payload = text.encode("utf-8")
    n = len(payload)
    head = bytes([0x80 | opcode, n]) if n < 126 else bytes([0x80 | opcode, 126]) + struct.pack(">H", n)
    return head + payload


def _devtools_that_never_answers(peer: socket.socket, stop: threading.Event, kind: str, seconds: float) -> None:
    """kind: "events" (a stream of events), "pings" (a stream of pings, which never make a message) or "silence". It hangs up after
    `seconds` (or when stop is set), so a deadline that is missing fails a test instead of hanging it."""
    end = time.monotonic() + seconds
    try:
        while kind != "silence" and not stop.is_set() and time.monotonic() < end:
            peer.sendall(_server_frame(json.dumps({"method": "Network.dataReceived", "params": {}})) if kind == "events"
                         else _server_frame("ping", opcode=0x9))
            time.sleep(0.02)
        stop.wait(max(0.0, end - time.monotonic()))
    except OSError:   # the other end closed
        pass
    finally:
        peer.close()


@pytest.mark.parametrize("kind", ["events", "pings", "silence"])
def test_the_devtools_step_of_close_ends_at_its_deadline_however_much_the_peer_sends(tmp_path, monkeypatch, kind):
    """A socket timeout is per read, so a peer that keeps sending (events, pings) and never answers Browser.close would hold close() for
    as long as it likes. A real WebSocket on a socketpair (no handshake, no Chrome) and a peer that talks for 6 s: close() must still
    be done within a second or so of CLOSE_WAIT_S, and have asked Chrome to leave by TERM, not KILL."""
    monkeypatch.setattr(chat_ui_cdp, "CLOSE_WAIT_S", 0.3)
    mine, theirs = socket.socketpair()
    ws = chat_ui_cdp.WebSocket.__new__(chat_ui_cdp.WebSocket)
    ws.sock, ws.buffer = mine, b""
    stop = threading.Event()
    peer = threading.Thread(target=_devtools_that_never_answers, args=(theirs, stop, kind, 6.0), daemon=True)
    proc, marker = _start(tmp_path, POLITE, "chrome")
    peer.start()
    started = time.monotonic()
    try:
        _bare_browser(proc, tmp_path / "profile", ws).close()
        took = time.monotonic() - started
    finally:
        stop.set()
        peer.join(timeout=5)
    assert took < 2.0, "close() took {:.1f} s: the peer held DevTools open past CLOSE_WAIT_S".format(took)
    assert proc.returncode == 0 and not marker.exists()


def _socketpair_websocket(deadline: Optional[float]):
    """A real WebSocket on one end of a socketpair (no handshake, no Chrome), with its deadline set; -> (it, the peer's end)."""
    mine, theirs = socket.socketpair()
    ws = chat_ui_cdp.WebSocket.__new__(chat_ui_cdp.WebSocket)
    ws.sock, ws.buffer, ws.deadline = mine, b"", deadline
    return ws, theirs


def test_a_send_after_the_deadline_sends_nothing_and_raises():
    """P4-E follow-up. The deadline of a receive was kept by _arm(); the send side had none, so a close() whose Browser.close request met
    a deadline that had passed still wrote it."""
    ws, theirs = _socketpair_websocket(time.monotonic() - 1.0)
    try:
        with pytest.raises((TimeoutError, socket.timeout)):
            ws.send("never sent")
        theirs.settimeout(0.3)
        with pytest.raises((TimeoutError, socket.timeout)):
            theirs.recv(1)   # not one byte reached the peer
    finally:
        ws.close()
        theirs.close()


def test_a_send_to_a_peer_that_never_reads_ends_at_the_deadline_and_not_at_the_sockets_own_timeout():
    """The request is larger than the socket's buffers and the peer reads none of it: sendall blocks. A socket timeout is per operation (here
    5 s, as a connection's own is 30 s), so only the deadline, applied before the write, can end it when close() says it must."""
    ws, theirs = _socketpair_websocket(time.monotonic() + 0.3)
    ws.sock.settimeout(5.0)
    started = time.monotonic()
    try:
        with pytest.raises((TimeoutError, socket.timeout)):
            ws.send("x" * (8 * 1024 * 1024))
        took = time.monotonic() - started
    finally:
        ws.close()
        theirs.close()
    assert took < 2.0, "the send held on for {:.1f} s: it kept the socket's timeout, not the deadline".format(took)


def test_closing_a_browser_whose_chrome_has_gone_sends_nothing(tmp_path):
    proc = subprocess.Popen(["/bin/sh", "-c", "exit 0"], start_new_session=True)
    proc.wait()
    ws = ScriptedSocket([])
    _bare_browser(proc, tmp_path / "profile", ws).close()
    assert ws.sent == []


def test_stopping_an_app_asks_politely_first_and_removes_the_home_it_made(tmp_path):
    proc, marker = _start(tmp_path, POLITE, "app")
    home = tmp_path / "home"
    home.mkdir()
    app = chat_ui_cdp.App.__new__(chat_ui_cdp.App)
    app.proc, app.home, app.owns_home = proc, str(home), True
    app.stop()
    assert proc.returncode == 0 and not marker.exists()
    assert not home.exists()


def test_a_home_that_was_given_is_left_to_its_owner(tmp_path):
    proc, marker = _start(tmp_path, POLITE, "app")
    home = tmp_path / "home"
    home.mkdir()
    app = chat_ui_cdp.App.__new__(chat_ui_cdp.App)
    app.proc, app.home, app.owns_home = proc, str(home), False
    app.stop()
    assert proc.returncode == 0 and home.exists()


def _tiny_turn(engine: Any, max_new_tokens: int = 16):
    """What a turn hands Engine.generate: the encoded synthetic image, and the options of a short turn."""
    from app.schemas import Options
    from tests.app_helpers import png_bytes
    _, prepared = engine.preprocess(png_bytes(320, 320))
    _, encoded = engine.encode(prepared)
    return encoded, Options(max_new_tokens=max_new_tokens)


def test_a_held_tiny_turn_stops_after_its_first_snapshot_and_goes_on_when_the_hold_is_released(tmp_path, monkeypatch):
    """The reload step of the browser check needs a turn that is still running when the page has been reloaded, whatever the load on the
    machine: the tiny engine runs on its own clock, and a reload that took longer than the turn found it done. While the file exists, the
    turn sends the snapshot it has made and then waits (App.hold); the file's removal lets it go on."""
    from app.engine import TinyEngine, build_engine
    monkeypatch.setattr(TinyEngine, "generate", TinyEngine.generate)   # put back at the end of the test, whatever hold_turns does to it
    hold = tmp_path / "hold"
    chat_ui_cdp.hold_turns(str(hold))
    engine = build_engine("tiny")
    encoded, options = _tiny_turn(engine)
    hold.touch()
    snapshots, finished, cancel = [], [], threading.Event()
    worker = threading.Thread(target=lambda: finished.append(engine.generate(encoded, options, lambda step, text: snapshots.append(step), cancel)))
    worker.start()
    try:
        assert _until(lambda: len(snapshots) >= 1, 30), "the turn made no snapshot"
        time.sleep(0.5)   # 25 steps of the 20 ms engine, had it gone on
        assert snapshots == [0] and worker.is_alive(), "the turn went on while it was held: {}".format(snapshots)
        hold.unlink()
        worker.join(timeout=30)
        assert not worker.is_alive() and snapshots == list(range(16)) and len(finished) == 1, "the released turn did not run to its end"
    finally:
        cancel.set()
        worker.join(timeout=10)


def test_a_stop_ends_a_held_turn_with_the_hold_still_on(tmp_path, monkeypatch):
    """Stop must work on a held turn: a check that fails while it holds one stops the app, and the page's Stop is the user's."""
    from app.engine import Cancelled, TinyEngine, build_engine
    monkeypatch.setattr(TinyEngine, "generate", TinyEngine.generate)
    hold = tmp_path / "hold"
    chat_ui_cdp.hold_turns(str(hold))
    engine = build_engine("tiny")
    encoded, options = _tiny_turn(engine)
    hold.touch()
    snapshots, outcome, cancel = [], [], threading.Event()

    def run() -> None:
        try:
            engine.generate(encoded, options, lambda step, text: snapshots.append(step), cancel)
            outcome.append("finished")
        except Cancelled:
            outcome.append("cancelled")

    worker = threading.Thread(target=run)
    worker.start()
    try:
        assert _until(lambda: len(snapshots) >= 1, 30)
        cancel.set()
        worker.join(timeout=5)
        assert not worker.is_alive() and outcome == ["cancelled"] and hold.exists(), (outcome, hold.exists())
    finally:
        cancel.set()
        worker.join(timeout=10)


def test_an_app_holds_and_releases_by_the_file_its_launcher_watches(tmp_path):
    app = chat_ui_cdp.App.__new__(chat_ui_cdp.App)
    app.home = str(tmp_path)
    app.hold()
    app.hold()   # holding what is held is the same
    assert (tmp_path / chat_ui_cdp.HOLD_FILE).exists()
    app.release()
    app.release()   # and so is releasing what is free
    assert not (tmp_path / chat_ui_cdp.HOLD_FILE).exists()
    assert 'hold_turns(os.path.join(args[0], "{}"))'.format(chat_ui_cdp.HOLD_FILE) in chat_ui_cdp.LAUNCH   # the app process watches the same file


def test_exit_on_sigterm_turns_the_signal_into_a_system_exit_so_that_finally_runs():
    before = signal.getsignal(signal.SIGTERM)
    try:
        chat_ui_cdp.exit_on_sigterm()
        handler = signal.getsignal(signal.SIGTERM)
        assert handler is not before
        with pytest.raises(SystemExit) as raised:
            handler(signal.SIGTERM, None)
        assert raised.value.code == 128 + signal.SIGTERM
    finally:
        signal.signal(signal.SIGTERM, before)


def test_the_sigterm_handler_is_one_shot_so_a_second_term_cannot_cut_the_clean_up_short():
    """P4-E follow-up. The first SIGTERM raises SystemExit and the finally clauses stop Chrome and the app, which takes several seconds (a
    DevTools close, a TERM and its wait, a KILL). A handler that raised on every SIGTERM turned a second one, from a harness that is
    impatient, into a second SystemExit in the middle of that clean-up, which is where the processes are left running. After the first, the
    handler is replaced by one that takes the signal and does nothing (not SIG_IGN, which a process started afterwards would inherit)."""
    before = signal.getsignal(signal.SIGTERM)
    try:
        chat_ui_cdp.exit_on_sigterm()
        first = signal.getsignal(signal.SIGTERM)
        with pytest.raises(SystemExit) as raised:
            first(signal.SIGTERM, None)
        assert raised.value.code == 128 + signal.SIGTERM
        later = signal.getsignal(signal.SIGTERM)
        assert later is not first and callable(later), "the handler is still the one that raises"
        assert later(signal.SIGTERM, None) is None   # a second, a third: nothing is raised
        assert later(signal.SIGTERM, None) is None
        assert signal.getsignal(signal.SIGTERM) is later
    finally:
        signal.signal(signal.SIGTERM, before)


def test_both_checks_install_it_before_they_start_anything(monkeypatch):
    from scripts import chat_ui_browser_check as browser_check
    for module in (layout, browser_check):
        monkeypatch.setattr(module, "find_chrome", lambda: None)   # main returns 2 at once: nothing was started
        before = signal.getsignal(signal.SIGTERM)
        try:
            assert module.main([]) == 2
            handler = signal.getsignal(signal.SIGTERM)
            assert handler is not before, module.__name__
            with pytest.raises(SystemExit):
                handler(signal.SIGTERM, None)
        finally:
            signal.signal(signal.SIGTERM, before)
            signal.alarm(0)


@pytest.mark.parametrize("module_name", ["chat_ui_layout_check", "chat_ui_browser_check"])
@pytest.mark.parametrize("interrupted", ["browser", "app"])
def test_a_check_goes_on_stopping_what_it_started_when_a_step_of_its_clean_up_is_interrupted(monkeypatch, tmp_path, module_name, interrupted):
    """P4-E follow-up. Browser.close() now finishes its own clean-up when an interrupt (Ctrl-C, the SIGTERM handler's SystemExit) cuts it
    short, and then lets the interrupt through. A main() whose finally closed the browser, stopped the app and removed its work directory,
    one statement after the other, never got past the first of them: the app was left running. Each step is in the finally of the one
    before it, so the steps after an interrupted one still run, and the interrupt goes on out of main()."""
    module = importlib.import_module("scripts." + module_name)
    calls = []   # type: List[str]
    work = tmp_path / "work"

    class FakeApp:
        url = "http://127.0.0.1:1/"
        static_dir = None

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            calls.append("app started")

        def stop(self) -> None:
            calls.append("app stopped")
            if interrupted == "app":
                raise KeyboardInterrupt

    class FakeBrowser:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            calls.append("browser started")

        def close(self) -> None:
            calls.append("browser closed")
            if interrupted == "browser":
                raise KeyboardInterrupt

    def make_work(prefix: str = "") -> str:
        work.mkdir()
        return str(work)

    monkeypatch.setattr(module, "find_chrome", lambda: "chrome")
    monkeypatch.setattr(module, "App", FakeApp)
    monkeypatch.setattr(module, "Browser", FakeBrowser)
    if module_name == "chat_ui_browser_check":
        monkeypatch.setattr(module, "warm_up", lambda base: 0.0)
        monkeypatch.setattr(module, "run_all", lambda *args, **kwargs: ([], None))
        monkeypatch.setattr(module, "environment", lambda *args, **kwargs: {})
        monkeypatch.setattr(module.tempfile, "mkdtemp", make_work)
    else:
        monkeypatch.setattr(module, "run_checks", lambda browser, url: (0, []))
    before = signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGALRM)
    try:
        with pytest.raises(KeyboardInterrupt):
            module.main(["--no-evidence"] if module_name == "chat_ui_browser_check" else [])
    finally:
        signal.signal(signal.SIGTERM, before[0])
        signal.signal(signal.SIGALRM, before[1])
        signal.alarm(0)
    assert calls == ["app started", "browser started", "browser closed", "app stopped"], calls
    assert not work.exists(), "the work directory was left behind"


# ---- the refusals of a --url: one implementation, called by the three checks (P6-F fix 1) -----------------------------------------------

def test_the_ports_that_are_refused_are_the_dev_servers_and_the_tunnels():
    assert chat_ui_cdp.REFUSED_PORTS == (8000, 8010)


@pytest.mark.parametrize("url", ["http://127.0.0.1:8000/", "http://127.0.0.1:8000", "http://127.0.0.1:8010/", "http://localhost:8000/",
                                 "http://localhost:8010/chat", "http://[::1]:8010/", "http://[::1]:8000/", "HTTP://127.0.0.1:8000/"])
def test_the_dev_server_and_the_tunnel_are_refused_on_every_spelling_of_this_machine(url):
    problem = chat_ui_cdp.url_problem(url)
    assert problem and ("8000" in problem and "8010" in problem), problem   # it says which two, and why


@pytest.mark.parametrize("url", ["http://example.com:5000/", "http://192.168.1.5:51234/", "http://0.0.0.0:51234/", "https://127.0.0.1:51234/",
                                 "ftp://127.0.0.1:51234/", "file:///tmp/x", "127.0.0.1:51234", "http://127.0.0.1/", "http://127.0.0.1:0/", ""])
def test_a_url_that_is_not_an_http_loopback_address_with_a_port_is_refused(url):
    assert chat_ui_cdp.url_problem(url)


@pytest.mark.parametrize("url", ["http://127.0.0.1:51234/", "http://127.0.0.1:51234", "http://localhost:49999/", "http://[::1]:51234/x"])
def test_a_loopback_url_on_another_port_is_accepted(url):
    assert chat_ui_cdp.url_problem(url) is None


def test_the_models_the_refusal_compares_with_are_the_tiny_engines_own():
    from app.engine import TinyEngine
    assert (chat_ui_cdp.TINY_MODEL, chat_ui_cdp.TINY_NOTE) == (TinyEngine.name, TinyEngine.drift_note)


@pytest.mark.parametrize("gallery", [False, True], ids=["no-gallery", "tiny-gallery"])
def test_the_live_tiny_apps_own_models_are_taken_for_the_tiny_app(gallery):
    assert chat_ui_cdp.models_problem(url_cases.tiny_models(gallery)) is None


@pytest.mark.parametrize("why", url_cases.NOT_TINY_WHY)
def test_what_is_not_the_tiny_engine_without_a_checkpoint_is_refused_with_one_fixed_message(why):
    """The default model and every listed model must be the tiny engine (name, the note it gives of itself) with no checkpoint, no checkpoint
    hash and no training run, each of those fields there and null: a card that leaves one out, or is not an object, is refused as well."""
    assert chat_ui_cdp.models_problem(url_cases.not_tiny_answer(why)) == chat_ui_cdp.NOT_THE_TINY_APP


def test_a_refusal_says_the_same_thing_whatever_the_server_said():
    answer = url_cases.tiny_models(True)
    answer["models"][0].update(checkpoint=url_cases.MARKER, name=url_cases.MARKER)
    answer["default_model"] = url_cases.MARKER
    assert chat_ui_cdp.models_problem(answer) == chat_ui_cdp.NOT_THE_TINY_APP and url_cases.MARKER not in chat_ui_cdp.NOT_THE_TINY_APP
    assert url_cases.MARKER not in chat_ui_cdp.MODELS_UNREADABLE


def test_the_app_is_asked_for_its_models_and_for_nothing_else():
    with url_cases.stub_app(url_cases.tiny_models(True)) as stub:
        assert chat_ui_cdp.tiny_only_problem(stub.url) is None
        assert chat_ui_cdp.tiny_only_problem(stub.url.rstrip("/")) is None   # with or without the slash
        assert stub.requests == [("GET", "/v1/models")] * 2, stub.requests


@pytest.mark.parametrize("url", ["http://127.0.0.1:8000/", "http://localhost:8010/", "http://example.com:5000/", "https://127.0.0.1:51234/", ""])
def test_an_address_that_is_refused_is_not_asked_for_its_models_either(monkeypatch, url):
    """tiny_only_problem is the one function that sends anything to a --url, so it refuses what url_problem refuses before it connects: a caller
    that forgot the first refusal could not send a GET to the dev server's port."""
    def forbidden(*args, **kwargs):
        raise AssertionError("a connection was made for a refused address")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    assert chat_ui_cdp.tiny_only_problem(url) == chat_ui_cdp.url_problem(url) and chat_ui_cdp.url_problem(url)


def test_a_redirect_is_not_followed_to_an_app_that_the_address_did_not_name():
    with url_cases.stub_app(url_cases.tiny_models(True)) as elsewhere, url_cases.stub_app(location=elsewhere.url + "v1/models") as stub:
        assert chat_ui_cdp.tiny_only_problem(stub.url) == chat_ui_cdp.MODELS_UNREADABLE
        assert elsewhere.requests == [], elsewhere.requests


def test_a_proxy_in_the_environment_is_not_used_for_the_app_on_this_machine(monkeypatch):
    for name in ("http_proxy", "HTTP_PROXY"):
        monkeypatch.setenv(name, "http://127.0.0.1:{}/".format(chat_ui_cdp.free_port()))   # nothing answers there
    for name in ("no_proxy", "NO_PROXY"):
        monkeypatch.setenv(name, "")
    with url_cases.stub_app(url_cases.tiny_models()) as stub:
        assert chat_ui_cdp.tiny_only_problem(stub.url) is None
        assert stub.requests == [("GET", "/v1/models")]


def test_an_answer_of_more_than_a_megabyte_is_not_read_to_its_end():
    body = json.dumps(url_cases.tiny_models(True)).encode("utf-8") + b" " * (2 * 1024 * 1024)
    with url_cases.stub_app(body=body) as stub:
        assert chat_ui_cdp.tiny_only_problem(stub.url) == chat_ui_cdp.MODELS_UNREADABLE


def test_a_url_with_no_evidence_is_the_only_url_that_is_taken_where_evidence_is_written(capsys):
    import argparse
    parser = argparse.ArgumentParser(prog="check")
    parser.add_argument("--url")
    parser.add_argument("--no-evidence", action="store_true")
    chat_ui_cdp.url_needs_no_evidence(parser, parser.parse_args([]))
    chat_ui_cdp.url_needs_no_evidence(parser, parser.parse_args(["--no-evidence"]))
    chat_ui_cdp.url_needs_no_evidence(parser, parser.parse_args(["--url", "http://127.0.0.1:51234/", "--no-evidence"]))
    with pytest.raises(SystemExit) as refused:
        chat_ui_cdp.url_needs_no_evidence(parser, parser.parse_args(["--url", "http://127.0.0.1:51234/"]))
    assert refused.value.code == 2
    assert chat_ui_cdp.URL_NEEDS_NO_EVIDENCE in capsys.readouterr().err


@pytest.mark.parametrize("port", [8000, 8010])
def test_the_layout_check_refuses_the_dev_servers_and_the_tunnels_port_before_it_sends_or_starts_anything(monkeypatch, capsys, port):
    url_cases.refuses_a_port(layout, monkeypatch, capsys, port)


def test_the_layout_check_refuses_an_address_that_is_not_this_machines_loopback_before_it_sends_or_starts_anything(monkeypatch, capsys):
    url_cases.refuses_a_host(layout, monkeypatch, capsys)


def test_the_layout_check_refuses_an_app_that_is_not_the_tiny_one_with_a_fixed_line_before_it_starts_anything(monkeypatch, capsys):
    url_cases.refuses_a_server_that_is_not_the_tiny_app(layout, monkeypatch, capsys)


def test_the_layout_check_goes_on_with_the_tiny_app(monkeypatch, capsys):
    url_cases.goes_on_with_the_tiny_app(layout, monkeypatch, capsys)


# ---- the geometry check, on measurements ----------------------------------------------------------------------------------------------

def _measure(width: int = 1280, height: int = 900, stop: bool = False, drawer: bool = False) -> Dict[str, Any]:
    """What MEASURE_JS reports for a page that is in order: nothing overflows, the composer is inside the viewport, the report has
    room, and #stop and #drawer are rendered exactly when the state shows them."""
    drawer_box = [width - 320.0, 40.0, float(width), float(height)] if drawer else None
    return {
        "doc": [width, width], "#conversation": {"sw": width, "cw": width, "sh": 300, "ch": 300},
        "#composer": {"sw": width - 20, "cw": width - 20, "sh": 200, "ch": 200},
        "boxes": {"#composer": [10.0, height - 220.0, width - 10.0, height - 20.0], "#settings": [20.0, height - 80.0, 100.0, height - 40.0],
                  "#send": [110.0, height - 80.0, 170.0, height - 40.0], "#stop": [180.0, height - 80.0, 240.0, height - 40.0] if stop else None},
        "hits": {"#settings": {"ok": True, "at": [60, height - 60], "got": "#settings"}, "#send": {"ok": True, "at": [140, height - 60], "got": "#send"},
                 "#stop": {"ok": True, "at": [210, height - 60], "got": "#stop"} if stop else None},
        "rendered": {"#stop": stop, "#drawer": drawer}, "drawer": drawer_box,
    }


def test_a_page_in_order_has_no_failures_in_any_state():
    for stop in (False, True):
        for drawer in (False, True):
            assert layout.check_state(_measure(stop=stop, drawer=drawer), 1280, 900, False, drawer, stop) == []


def test_a_stop_button_that_is_not_rendered_in_a_case_that_shows_it_fails():
    m = _measure(stop=True)
    m["rendered"]["#stop"] = False   # what a display: none mutant of #stop measures
    m["boxes"]["#stop"] = None
    m["hits"]["#stop"] = None
    failures = layout.check_state(m, 1280, 900, False, False, True)
    assert any("#stop is not rendered" in f for f in failures), failures


def test_a_drawer_that_is_not_rendered_in_a_case_that_opens_it_fails():
    m = _measure(drawer=True)
    m["rendered"]["#drawer"] = False
    m["drawer"] = None
    failures = layout.check_state(m, 1280, 900, False, True, False)
    assert any("#drawer is not rendered" in f for f in failures), failures


def test_a_control_that_is_rendered_in_a_case_that_hides_it_fails():
    failures = layout.check_state(_measure(stop=True, drawer=True), 1280, 900, False, False, False)   # the [hidden] rule is gone
    assert any("#stop is rendered, but the case hides it" in f for f in failures), failures
    assert any("#drawer is rendered, but the case hides it" in f for f in failures), failures


def test_an_open_drawer_that_runs_past_the_viewport_fails():
    m = _measure(width=800, drawer=True)
    m["drawer"] = [500.0, 40.0, 900.0, 900.0]
    failures = layout.check_state(m, 800, 900, False, True, False)
    assert any("#drawer leaves the viewport" in f for f in failures), failures


def _save(width: int = 1280, height: int = 900, drawer: bool = True, in_view: bool = True, covered: bool = False, tall: float = 40.0,
          left: Optional[float] = None) -> Dict[str, Any]:
    """What MEASURE_SAVE_JS reports for a drawer whose Save button is in order: rendered and in sight when the drawer is open, not rendered
    when it is closed, inside the drawer's width, a touch target tall, and nothing over its centre once it is scrolled to."""
    if not drawer:
        return {"present": True, "rendered": False, "drawer": None, "box": None, "in_view": False, "hit": None}
    x = width - 320.0 + 16.0 if left is None else left
    box = [x, height - 80.0, x + 288.0, height - 80.0 + tall]
    return {"present": True, "rendered": True, "drawer": [width - 320.0, 40.0, float(width), float(height)], "box": box, "in_view": in_view,
            "hit": {"ok": not covered, "at": [round(x + 144), round(height - 60.0)], "got": "#send" if covered else "#drawer-save"}}


def test_a_save_button_that_is_rendered_in_sight_and_reachable_in_an_open_drawer_has_no_failures():
    for scroll in (False, True):
        assert layout.check_save(_save(), 1280, 900, scroll, True) == []
        assert layout.check_save(_save(drawer=False), 1280, 900, scroll, False) == []


def test_a_save_button_that_is_not_rendered_in_an_open_drawer_or_is_rendered_in_a_closed_one_fails():
    gone = _save()
    gone["rendered"], gone["box"], gone["hit"] = False, None, None   # what a display: none mutant of the button measures
    assert any("#drawer-save is not rendered" in f for f in layout.check_save(gone, 1280, 900, False, True))
    shown = _save()
    assert any("#drawer-save is rendered, but the case hides the drawer" in f for f in layout.check_save(shown, 1280, 900, False, False))
    missing = {"present": False}   # no Save at all: the drawer is not a form that ends in one
    assert any("no #drawer-save" in f for f in layout.check_save(missing, 1280, 900, False, True))


def test_a_save_button_that_leaves_the_drawer_or_the_viewport_or_is_covered_or_too_small_fails():
    assert any("leaves the drawer" in f for f in layout.check_save(_save(left=1100.0), 1280, 900, False, True))   # runs past its right edge
    assert any("leaves the drawer" in f for f in layout.check_save(_save(left=900.0), 1280, 900, False, True))    # starts left of it
    assert any("is covered" in f for f in layout.check_save(_save(covered=True), 1280, 900, False, True))
    assert any("is 30 px tall" in f for f in layout.check_save(_save(tall=30.0), 1280, 900, False, True))
    assert any("is not in sight" in f for f in layout.check_save(_save(in_view=False), 1280, 900, False, True))   # the sticky bar went static


def test_where_the_page_scrolls_save_need_not_be_in_sight_until_it_is_scrolled_to():
    assert layout.check_save(_save(width=667, height=375, in_view=False), 667, 375, True, True) == []   # it is scrolled to, and hit-tested there
    assert any("is covered" in f for f in layout.check_save(_save(width=667, height=375, covered=True), 667, 375, True, True))


def _tall(height: int = 568, report: int = 120, composer_bottom: Optional[float] = None, scrolls: bool = True, reachable: bool = True) -> Dict[str, Any]:
    return {"doc": [375, 375], "conversation": {"ch": report},
            "composer": {"box": [16.0, 200.0, 359.0, composer_bottom if composer_bottom is not None else float(height) - 16.0],
                         "sh": 900 if scrolls else 300, "ch": 300},
            "hits": {"#settings": reachable, "#send": reachable}}


def test_a_tall_composer_that_keeps_the_report_floor_and_scrolls_inside_passes():
    assert layout.check_tall(_tall(), 568) == []


def test_a_deleted_report_floor_fails_the_tall_composer_case():
    failures = layout.check_tall(_tall(report=40), 568)   # grid-template-rows: minmax(0, 1fr) instead of minmax(120px, 1fr)
    assert failures == ["the report area is 40 px tall, under the 120 px floor"]


def test_a_composer_that_runs_off_the_page_or_cannot_be_scrolled_to_its_buttons_fails():
    assert any("runs past the bottom" in f for f in layout.check_tall(_tall(composer_bottom=900.0), 568))
    assert any("does not scroll inside itself" in f for f in layout.check_tall(_tall(scrolls=False), 568))
    failures = layout.check_tall(_tall(reachable=False), 568)   # overflow: visible, so scrolling the composer moves nothing
    assert any("#send cannot be reached" in f for f in failures) and any("#settings cannot be reached" in f for f in failures)


def _sections(width: int = 1280, sections: Optional[List[str]] = None) -> Dict[str, Any]:
    """What MEASURE_SECTIONS_JS reports for a card whose own sections are in order (check i, P6-B..D): each inside the card and no wider
    inside than it is, its pictures loaded, with a size, inside the card; nothing overflows the page."""
    card = [24.0, 60.0, width - 24.0, 2400.0]
    picture = {"box": [card[0] + 16.0, 320.0, card[0] + 176.0, 480.0], "loaded": True}
    long_ = {"h": 63, "sh": 210, "line": 21.0, "chars": 590, "toggle": True}   # three lines of 21 px, more below them, and Show all
    narrow = width <= 600   # P6 fix 1: a ~170-character report overflows three lines at phone width and fits them on a wide page
    mid = {"h": 63, "sh": 84 if narrow else 42, "line": 21.0, "chars": 170, "toggle": narrow}
    short = {"h": 21, "sh": 21, "line": 21.0, "chars": 40, "toggle": False}

    def one(name: str) -> Dict[str, Any]:
        clamped = [dict(long_)] + ([dict(mid), dict(short)] if name == "section.matches" else []) if name in layout.CLAMPED else []
        return {"box": [card[0] + 16.0, 300.0, card[2] - 16.0, 520.0], "sw": 300, "cw": 300,
                "pictures": [dict(picture)] if name in layout.PICTURED else [], "clamped": clamped}

    return {"doc": [width, width], "conversation": [width, width], "card": card,
            "sections": {name: one(name) for name in (layout.SECTIONS if sections is None else sections)}}


def test_card_sections_in_order_have_no_failures():
    assert layout.check_sections(_sections(), 1280, layout.SECTIONS) == []
    assert layout.SECTIONS == ["section.images", "section.neighbors", "section.published", "section.matches"]   # P6-B, P6-C, P6-D
    assert set(layout.PICTURED) | set(layout.CLAMPED) <= set(layout.SECTIONS)


def test_the_card_check_holds_the_tiny_engines_notice_to_what_it_holds_the_sections_to():   # P9-G5b
    """The synthetic turn is the tiny engine's, so its card carries the notice above the report: it must be there at every viewport, inside the
    card's width, and not wider inside than it is. The sections' own list (and what it pins) is as it was."""
    assert layout.NOTICE == "p.test-notice" and layout.CARD_PARTS[:len(layout.SECTIONS) + 1] == layout.SECTIONS + [layout.NOTICE]
    assert layout.check_sections(_sections(320, layout.CARD_PARTS), 320, layout.CARD_PARTS) == []
    gone = _sections(1280, layout.CARD_PARTS)
    gone["sections"][layout.NOTICE] = None
    assert layout.check_sections(gone, 1280, layout.CARD_PARTS) == ["p.test-notice is not in the card"]
    out = _sections(375, layout.CARD_PARTS)
    out["sections"][layout.NOTICE]["box"] = [8.0, 300.0, 390.0, 340.0]   # wider than the card
    assert any(f.startswith("p.test-notice leaves the card") for f in layout.check_sections(out, 375, layout.CARD_PARTS))
    over = _sections(375, layout.CARD_PARTS)
    over["sections"][layout.NOTICE].update(sw=420, cw=300)
    assert any(f.startswith("p.test-notice overflows inside") for f in layout.check_sections(over, 375, layout.CARD_PARTS))


def test_the_card_check_holds_the_report_card_to_what_it_holds_the_sections_to():   # P6-H
    """The report card a person can read: its summary box, its list of sentences, its notes and its Terms are drawn in the page by render.js from
    the synthetic turn, and held at every viewport (375 and 1280 px among them, in both schemes, the drawer closed and open) to what the sections
    are: there, inside the card's width, and not wider inside than it is."""
    assert layout.REPORT_PARTS == [".report-summary", ".report-body", ".report-key", ".report-terms"]
    assert layout.CARD_PARTS == layout.SECTIONS + [layout.NOTICE] + layout.REPORT_PARTS
    assert layout.check_sections(_sections(375, layout.CARD_PARTS), 375, layout.CARD_PARTS) == []
    for part in layout.REPORT_PARTS:
        gone = _sections(1280, layout.CARD_PARTS)
        gone["sections"][part] = None
        assert layout.check_sections(gone, 1280, layout.CARD_PARTS) == [part + " is not in the card"], part
        out = _sections(375, layout.CARD_PARTS)
        out["sections"][part]["box"] = [8.0, 300.0, 390.0, 340.0]   # wider than the card
        assert any(f.startswith(part + " leaves the card") for f in layout.check_sections(out, 375, layout.CARD_PARTS)), part
        over = _sections(375, layout.CARD_PARTS)
        over["sections"][part].update(sw=420, cw=300)   # a word that does not wrap
        assert any(f.startswith(part + " overflows inside") for f in layout.check_sections(over, 375, layout.CARD_PARTS)), part


def test_the_synthetic_turn_draws_every_part_of_the_report_card():   # P6-H
    """The card of the synthetic turn must have what the parts above stand for: a labelled report (the summary), a sentence about an earlier study
    with a run of underscores in it (the notes), a term (the Terms), and a word as long as the widest a sentence must take."""
    events = layout.section_events()
    stop = next(e["data"] for e in events if e["event"] == "message_stop")
    snapshot = next(e["data"]["delta"]["text"] for e in events if e["event"] == "content_block_delta")
    label = next(e["data"]["detail"] for e in events if e["event"] == "stage_end" and e["data"]["stage"] == "label")
    assert stop["report"] == stop["display_report"] == snapshot == layout.CARD_REPORT
    assert "___" in layout.CARD_REPORT and "Compared with" in layout.CARD_REPORT and "pleural effusion" in layout.CARD_REPORT
    assert layout.LONG_WORD in layout.CARD_REPORT and "Findings:" in layout.CARD_REPORT and "Impression:" in layout.CARD_REPORT
    assert any(label["chexbert_14"].values())   # a positive label: the summary names it


def test_a_clamp_that_shows_more_than_three_lines_or_hides_nothing_fails():   # P6-D
    tall = _sections()
    tall["sections"]["section.matches"]["clamped"][0]["h"] = 84   # four lines: the clamp rule is gone
    assert any("clamped report 1 is 84 px tall, over three lines of 21 px" in f for f in layout.check_sections(tall, 1280, layout.SECTIONS))
    open_ = _sections()
    open_["sections"]["section.published"]["clamped"][0].update(h=210, sh=210)   # nothing hidden, so its Show all reveals nothing
    assert any("Show all that reveals nothing" in f for f in layout.check_sections(open_, 1280, layout.SECTIONS))
    none = _sections()
    none["sections"]["section.matches"]["clamped"] = [r for r in none["sections"]["section.matches"]["clamped"] if r["sh"] <= r["h"]]
    assert "section.matches clamps no report that overflows its three lines" in layout.check_sections(none, 1280, layout.SECTIONS)


def test_show_all_is_there_exactly_when_a_clamped_report_overflows():   # P6 fix 1
    for width in (375, 1280):
        assert layout.check_sections(_sections(width=width), width, layout.SECTIONS) == [], width
    lost = _sections()
    lost["sections"]["section.matches"]["clamped"][0]["toggle"] = False   # it overflows, and there is no way to read the rest
    assert any("overflows its three lines with no Show all" in f for f in layout.check_sections(lost, 1280, layout.SECTIONS))
    empty = _sections()
    empty["sections"]["section.matches"]["clamped"][2]["toggle"] = True   # a short report that fits: its button would reveal nothing
    assert any("Show all that reveals nothing" in f for f in layout.check_sections(empty, 1280, layout.SECTIONS))


def test_the_clamp_is_proven_at_phone_width_on_a_170_character_report_and_at_wide_width_on_a_short_one():   # P6 fix 1
    phone = _sections(width=375)
    phone["sections"]["section.matches"]["clamped"][1].update(sh=63, toggle=False)   # the 170-character report fit three lines at 375 px
    assert any("~170-character report has no Show all at 375 px" in f for f in layout.check_sections(phone, 375, layout.SECTIONS))
    gone = _sections(width=375)
    gone["sections"]["section.matches"]["clamped"] = gone["sections"]["section.matches"]["clamped"][:1]
    assert any("no ~170-character report" in f for f in layout.check_sections(gone, 375, layout.SECTIONS))
    wide = _sections()
    wide["sections"]["section.matches"]["clamped"][2].update(sh=42, toggle=True)   # a 40-character report overflowed at 1280 px
    assert any("short report has Show all at 1280 px" in f for f in layout.check_sections(wide, 1280, layout.SECTIONS))
    # with the drawer open the page is narrower: the proof at those two widths is for a closed drawer
    assert layout.check_sections(_sections(width=1024), 1024, layout.SECTIONS, drawer_open=True) == []


def _picker(width: int = 1280, height: int = 900) -> Dict[str, Any]:
    """What MEASURE_PICKER_JS reports for a page whose picker is open and in order (check j): the button and the list are rendered inside
    the viewport, nothing overflows horizontally, and the last card's header, scrolled into view, is hit at its centre (not the composer)."""
    return {"doc": [width, width], "conversation": [width, width], "composer": [width - 40, width - 40],
            "picker": [16.0, height - 420.0, width - 16.0, height - 200.0], "pick": [20.0, height - 80.0, 170.0, height - 40.0],
            "head": [40.0, 300.0, width - 40.0, 330.0], "hit": {"ok": True, "got": "li"}, "pickHit": {"ok": True, "got": "#pick-study"}}


def test_a_page_with_the_picker_open_in_order_has_no_failures():   # P6 fix 1
    for width, height, scroll in ((1280, 900, False), (375, 568, False), (320, 256, True)):   # a short viewport scrolls the page
        assert layout.check_picker(_picker(width, height), width, height, scroll) == [], (width, height)


def test_the_picker_case_fails_on_overflow_a_covered_header_or_a_picker_that_is_not_rendered():   # P6 fix 1
    wide = _picker(375, 568)
    wide["doc"] = [430, 375]
    assert any("documentElement overflows horizontally" in f for f in layout.check_picker(wide, 375, 568))
    spill = _picker(375, 568)
    spill["picker"][2] = 400.0
    assert any("#picker leaves the viewport's width" in f for f in layout.check_picker(spill, 375, 568))
    covered = _picker()
    covered["hit"] = {"ok": False, "got": "#picker"}
    assert any("the last card's header is covered by #picker" in f for f in layout.check_picker(covered, 1280, 900))
    for part in ("picker", "pick"):
        hidden = _picker()
        hidden[part] = None
        assert any("is not rendered" in f for f in layout.check_picker(hidden, 1280, 900)), part
    headless = _picker()
    headless["head"] = None
    assert layout.check_picker(headless, 1280, 900) == ["no card was drawn"]
    above = _picker()
    above["picker"] = [16.0, -300.0, 1264.0, -20.0]   # thrown out of sight above the page
    assert any("#picker is not inside the 1280x900 viewport" in f for f in layout.check_picker(above, 1280, 900))
    assert layout.check_picker(dict(above, picker=[16.0, 300.0, 304.0, 700.0]), 320, 256, scroll=True) == []   # where the page scrolls it may run on
    low = _picker(375, 568)
    low["pick"] = [20.0, 700.0, 170.0, 740.0]   # below the fold of a composer that scrolls inside: fine, as long as it can be scrolled to
    assert layout.check_picker(low, 375, 568) == []
    low["pickHit"] = {"ok": False, "got": "#composer"}
    assert any("#pick-study cannot be reached by scrolling: its centre hits #composer" in f for f in layout.check_picker(low, 375, 568))


SAVED_SENTENCE = "Settings saved. They apply from your next Send."


def _composer(width: int = 801, height: int = 568, state: str = "hint") -> Dict[str, Any]:
    """What MEASURE_COMPOSER_JS reports for a composer in order (check k, P6-A): the row the case shows (the re-run hint, or the preview, as
    a busy composer shows it too) is rendered inside the composer, the other is not, nothing overflows, and Send lies inside the composer's box
    and the viewport, hit at its centre. A busy one (P6 fix 2; any state of layout.BUSY_STATES) has Stop there too: a row above Send beside the
    drawer (801 px and up), and on Send's row, left of it, on a phone (P9-G5b); it reads "Cancel upload" beside a Send that is off, as the
    page's own Send leaves them (P9-G5b fix 1), and the saved row is showing only where Save came during the upload."""
    left, right = (16.0, width - 16.0) if width <= 800 else (276.0, width - 336.0)
    composer = [left, height - 330.0, right, height - 16.0]
    row = [left + 10, height - 280.0, right - 10, height - 260.0]
    busy = state.startswith("busy")
    stop = [right - 240.0, height - 66.0, right - 100.0, height - 26.0] if width <= 800 else [right - 130.0, height - 116.0, right - 10.0, height - 76.0]
    return {"doc": [width, width],
            "composer": {"box": composer, "sw": right - left - 2, "cw": right - left - 2, "sh": 300, "ch": 300, "st": 0},
            "conversation": 150,
            "rows": {"#rerun-hint": row if state == "hint" else None, "#preview": row if state != "hint" else None},
            "send": [right - 90.0, height - 66.0, right - 10.0, height - 26.0], "hit": {"ok": True, "got": "#send"},
            "stop": stop if busy else None,
            "stop_hit": {"ok": True, "got": "#stop"} if busy else None,
            "stop_text": layout.BUSY_STOP_LABEL if busy else "Stop", "send_disabled": busy,
            "saved": {"text": SAVED_SENTENCE, "shown": True} if state == "busy-save" else {"text": "", "shown": False}}


def test_a_composer_whose_rows_keep_send_in_sight_has_no_failures():   # P6-A (5a)
    for width, height, _drawer in layout.COMPOSER_CASES:
        for state in layout.COMPOSER_STATES:
            assert layout.check_composer(_composer(width, height, state), width, height, state) == [], (width, height, state)


def test_send_below_the_composers_fold_or_covered_fails_the_composer_case():   # P6-A (5a): the carried defect, 801-820 x 568-650, drawer open
    low = _composer()
    low["send"] = [631.0, 560.0, 711.0, 600.0]   # under the composer's bottom (552): only scrolling the composer would show it
    assert any("#send is below the composer's fold" in f for f in layout.check_composer(low, 801, 568, "hint"))
    covered = _composer()
    covered["hit"] = {"ok": False, "got": "#chips"}
    assert any("#send is covered: its centre hits #chips" in f for f in layout.check_composer(covered, 801, 568, "hint"))
    gone = _composer()
    gone["send"], gone["hit"] = None, None
    assert "#send is not rendered" in layout.check_composer(gone, 801, 568, "hint")


def test_the_composer_case_fails_on_a_missing_or_stray_row_overflow_or_a_lost_report_floor():   # P6-A (5a)
    hidden = _composer(state="preview")
    hidden["rows"]["#preview"] = None
    assert "#preview is not rendered, but the case shows it" in layout.check_composer(hidden, 801, 568, "preview")
    both = _composer(state="hint")
    both["rows"]["#preview"] = list(both["rows"]["#rerun-hint"])
    assert any("#preview is rendered with #rerun-hint" in f for f in layout.check_composer(both, 801, 568, "hint"))
    wide = _composer(320, 568, "hint")
    wide["composer"]["sw"] = 400   # a long file name that does not wrap or ellipsize widens the composer
    assert any("#composer overflows horizontally" in f for f in layout.check_composer(wide, 320, 568, "hint"))
    spill = _composer(320, 568, "preview")
    spill["rows"]["#preview"][2] = 330.0
    assert any("#preview leaves the composer's width" in f for f in layout.check_composer(spill, 320, 568, "preview"))
    floor = _composer()
    floor["conversation"] = 90
    assert any("the report area is 90 px tall, under 120" in f for f in layout.check_composer(floor, 801, 568, "hint"))


def test_a_busy_composer_keeps_stop_and_send_in_sight():   # P6 fix 2 (4)
    """While an upload goes up, Stop reads "Cancel upload" (its longest label) beside a Send that is off: both stay inside the composer's box
    and the viewport with nothing scrolled, each hit at its centre, where the composer is narrowest (801-834 x 568-650, the drawer open)."""
    for width, height, _drawer in layout.COMPOSER_BUSY_CASES:
        assert layout.check_composer(_composer(width, height, "busy"), width, height, "busy") == [], (width, height)
    low = _composer(state="busy")
    low["stop"] = [330.0, 579.0, 451.0, 621.0]   # the review's 801x568: "Cancel upload" on a row of its own under the composer's bottom (552)
    failures = layout.check_composer(low, 801, 568, "busy")
    assert any("#stop is below the composer's fold: Stop 579 to 621, composer 238 to 552" in f for f in failures)
    assert any("#stop is not inside the 801x568 viewport" in f for f in failures)
    covered = _composer(state="busy")
    covered["stop_hit"] = {"ok": False, "got": "#chips"}
    assert any("#stop is covered: its centre hits #chips" in f for f in layout.check_composer(covered, 801, 568, "busy"))
    gone = _composer(state="busy")
    gone["stop"], gone["stop_hit"] = None, None
    assert "#stop is not rendered, but the case shows it" in layout.check_composer(gone, 801, 568, "busy")
    hidden = _composer(state="busy")
    hidden["rows"]["#preview"] = None
    assert "#preview is not rendered, but the case shows it" in layout.check_composer(hidden, 801, 568, "busy")   # an upload: its preview stays
    assert not any("#stop" in f for f in layout.check_composer(_composer(state="preview"), 801, 568, "preview"))   # idle: no Stop to measure
    assert layout.COMPOSER_BUSY_CASES == [(801, 568, True), (801, 650, True), (820, 568, True), (820, 650, True), (834, 568, True),
                                          (320, 568, False), (320, 568, True)]   # P9-G5b adds the phone, in both drawer states
    assert layout.BUSY_STOP_LABEL == "Cancel upload"


def test_a_busy_composer_on_a_phone_has_stop_on_sends_row_and_in_sight_and_a_stop_on_a_row_of_its_own_under_the_fold_fails():   # P9-G5b (2c)
    """The re-review of P6 fix 2 measured it at 320 x 568: Stop ("Cancel upload") wrapped onto a row of its own, the composer's last, and showed
    9 px above the composer's fold. The fix puts it on Send's row; the case holds it there, and in sight, as it holds every busy composer."""
    for drawer in (False, True):
        assert (320, 568, drawer) in layout.COMPOSER_BUSY_CASES
    fixed = _composer(320, 568, "busy")
    assert layout.check_composer(fixed, 320, 568, "busy") == []
    assert abs(fixed["stop"][1] - fixed["send"][1]) <= layout.ROW_SLACK_PX   # one row: the fake page has what the real one has
    wrapped = _composer(320, 568, "busy")
    wrapped["stop"] = [145.0, 543.0, 278.0, 585.0]   # the review's measurement: the composer ends at 552, so 9 px of Stop show
    wrapped["stop_hit"] = {"ok": False, "got": "body"}
    failures = layout.check_composer(wrapped, 320, 568, "busy")
    assert "#stop is below the composer's fold: Stop 543 to 585, composer 238 to 552 (it is out of sight until the composer is scrolled)" in failures
    assert any("#stop is not inside the 320x568 viewport" in f for f in failures)
    assert "#stop is covered: its centre hits body" in failures
    assert "#stop is not on Send's row at 320 px: Stop starts at 543, Send at 502" in failures
    row_only = _composer(375, 900, "busy")   # in sight, but on a row of its own: a phone is held to the one row as well
    row_only["stop"] = [row_only["send"][0] - 140.0, row_only["send"][1] - 48.0, row_only["send"][0] - 10.0, row_only["send"][1] - 6.0]
    assert layout.check_composer(row_only, 375, 900, "busy") == ["#stop is not on Send's row at 375 px: Stop starts at {:.0f}, Send at {:.0f}".format(
        row_only["stop"][1], row_only["send"][1])]
    beside_the_drawer = _composer(801, 568, "busy")   # from 801 px the composer is 170 px wide: Stop keeps the row above Send
    assert beside_the_drawer["stop"][1] != beside_the_drawer["send"][1] and layout.check_composer(beside_the_drawer, 801, 568, "busy") == []
    assert (layout.PHONE_MAX_PX, layout.ROW_SLACK_PX) == (800, 2)


def test_the_busy_cases_with_more_in_the_composer_than_its_rows_are_the_phone_with_its_drawer_closed():   # P9-G5b fix 1 (1)
    """The review of 2c: the fix held only while the composer's content exactly fit. One more row (the saved row after a Save, a third row of
    chips) put Stop below the fold again: 49 to 105 px at 320 x 568. The drawer is closed at the start of these, as the Save they do goes through
    the page's own drawer."""
    assert layout.COMPOSER_STRESS_CASES == [(320, 568, False)]
    assert layout.STRESS_STATES == ["busy-after-save", "busy-save", "busy-8-chips"]
    assert layout.BUSY_STATES == ["busy"] + layout.STRESS_STATES
    for case in layout.COMPOSER_STRESS_CASES:
        assert case in layout.COMPOSER_CASES
    assert len(layout.SEVEN_CHIPS) == 7 and len(layout.EIGHT_CHIPS) == 8 and layout.EIGHT_CHIPS == layout.SEVEN_CHIPS + ["compiled"]
    assert "full budget" in layout.SEVEN_CHIPS and "labels off" in layout.SEVEN_CHIPS and "raw text" in layout.SEVEN_CHIPS   # the page's own words


def test_the_busy_state_is_entered_by_the_pages_own_send_with_the_upload_held_and_never_set_by_the_check():   # P9-G5b fix 1 (iii)
    """What the case measures is what the user sees: nothing of the busy state is written into the page (Send off, Stop shown, its label), the
    page's own send() does it, with the turn's upload held (the page's fetch, for the POST of a turn only, never answers)."""
    for source in (layout.INJECT_COMPOSER_JS,):
        assert "disabled" not in source and "stop.hidden" not in source and "textContent = 'Stop'" not in source   # no control's state is written
        assert "args.state === 'busy'" not in source and "args.stop" not in source
    assert "window.fetch" in layout.HOLD_UPLOAD_JS and "'POST'" in layout.HOLD_UPLOAD_JS and "/messages$/" in layout.HOLD_UPLOAD_JS
    assert "abort" in layout.HOLD_UPLOAD_JS   # Cancel upload still works on a held request, as on a slow one
    assert layout.BUSY_STOP_LABEL == "Cancel upload"


def test_a_busy_composer_that_scrolls_may_have_stop_on_a_row_of_its_own_if_the_page_has_scrolled_it_in_sight():   # P9-G5b fix 1 (ii)
    """A composer with more in it than its rows scrolls inside itself; its classic scrollbar takes 15 px of its width, three buttons do not fit
    in what is left, and Stop wraps onto a row of its own. The page scrolls the busy composer to its end, so that row is in sight, which is the
    ruling's other way of keeping Stop in view: the check takes it as it takes the first, and measures the composer where the page left it."""
    m = _composer(320, 568, "busy-8-chips")
    m["composer"].update(sw=251, cw=251, sh=380, ch=300, st=80)
    m["send"] = [225.0, 445.0, 278.0, 487.0]
    m["stop"] = [161.0, 495.0, 278.0, 537.0]   # a row of its own, under Send's: inside the composer, which ends at 552
    assert layout.check_composer(m, 320, 568, "busy-8-chips") == []
    unscrolled = _composer(320, 568, "busy-8-chips")
    unscrolled["composer"].update(sw=251, cw=251, sh=380, ch=300, st=0)
    unscrolled["send"] = [225.0, 509.0, 278.0, 551.0]
    unscrolled["stop"], unscrolled["stop_hit"] = [161.0, 559.0, 278.0, 601.0], {"ok": False, "got": "body"}   # the review's 49 px below the fold
    failures = layout.check_composer(unscrolled, 320, 568, "busy-8-chips")
    assert "#stop is below the composer's fold: Stop 559 to 601, composer 238 to 552 (it is out of sight until the composer is scrolled)" in failures
    assert not any("is not on Send's row" in f for f in failures)   # a composer that scrolls is not held to the one row
    fits = _composer(320, 568, "busy")
    fits["stop"] = [145.0, 495.0, 278.0, 537.0]   # but one that does not scroll still is: Stop beside Send is what the CSS rule is for
    assert any("#stop is not on Send's row at 320 px" in f for f in layout.check_composer(fits, 320, 568, "busy"))


def test_the_saved_row_is_gone_after_send_and_showing_where_save_came_during_the_upload():   # P9-G5b fix 1 (i)
    """The sentence says "from your next Send": a Send makes it stale, so the page takes it down then (a Save, then a Send within 4 s was the
    review's routine case). Where Save came while the upload was on its way, the row is on the page, and the case must be measuring it."""
    after = _composer(320, 568, "busy-after-save")
    assert layout.check_composer(after, 320, 568, "busy-after-save") == []
    after["saved"] = {"text": SAVED_SENTENCE, "shown": True}
    assert any("the saved row is still showing after Send" in f for f in layout.check_composer(after, 320, 568, "busy-after-save"))
    during = _composer(320, 568, "busy-save")
    assert during["saved"]["shown"] and layout.check_composer(during, 320, 568, "busy-save") == []
    during["saved"] = {"text": "", "shown": False}
    assert any("the saved row is not showing" in f for f in layout.check_composer(during, 320, 568, "busy-save"))
    for state in ("busy", "busy-8-chips"):   # the other busy cases say nothing of it
        quiet = _composer(320, 568, state)
        quiet["saved"] = {"text": SAVED_SENTENCE, "shown": True}
        assert layout.check_composer(quiet, 320, 568, state) == [], state


def test_a_busy_case_the_page_did_not_enter_by_its_own_send_fails():   # P9-G5b fix 1 (iii)
    """Stop reads "Cancel upload" and Send is off while the image goes up: a case whose Stop reads another word, or whose Send is on, is not
    measuring the upload."""
    accepted = _composer(320, 568, "busy")
    accepted["stop_text"] = "Stop"
    assert "#stop reads 'Stop', not 'Cancel upload': the case is not measuring the upload" in layout.check_composer(accepted, 320, 568, "busy")
    free = _composer(320, 568, "busy")
    free["send_disabled"] = False
    assert "#send is on while the upload goes up" in layout.check_composer(free, 320, 568, "busy")


def test_the_composer_cases_are_the_carried_widths_with_the_drawer_open_and_320_px():   # P6-A (5a)
    open_ = {(w, h) for w, h, drawer in layout.COMPOSER_CASES if drawer}
    assert {(801, 568), (820, 568), (834, 568), (801, 650), (820, 650)} <= open_
    assert (320, 568) in {(w, h) for w, h, _ in layout.COMPOSER_CASES}
    assert layout.COMPOSER_STATES == ["hint", "preview"] and len(layout.LONG_FILE) > 100 and " " not in layout.LONG_FILE


def _viewer(width: int = 1280, height: int = 900, pair: bool = True) -> Dict[str, Any]:
    """What MEASURE_VIEWER_JS reports for a viewer in order (check l, P6-E): under a 40 px banner, a 50 px bar, then one pane, or two side by
    side (stacked below 700 px), each picture loaded and drawn whole inside its stage, its footer naming its size, and every control hit at its
    centre. On a short page under 700 px (P6 fix 1) each stage is half the viewer's height, and the second pane runs below the fold."""
    top, bar = 40.0, 90.0
    if height <= 480 and width < 700:   # the viewer scrolls inside itself, its panes stacked
        stage = (height - top) / 2
        boxes = [[0.0, bar + k * (stage + 26), float(width), bar + k * (stage + 26) + stage + 24] for k in range(2 if pair else 1)]
    elif pair and width >= 700:
        boxes = [[0.0, bar, width / 2 - 1, float(height)], [width / 2 + 1, bar, float(width), float(height)]]
    elif pair:
        middle = (bar + height) / 2
        boxes = [[0.0, bar, float(width), middle - 1], [0.0, middle + 1, float(width), float(height)]]
    else:
        boxes = [[0.0, bar, float(width), float(height)]]
    panes = [{"box": b, "stage": [b[0], b[1], b[2], b[3] - 24], "img": {"box": [b[0] + 8, b[1] + 8, b[2] - 8, b[3] - 32], "loaded": True},
              "footer": "Your X-ray · 512×640 px"} for b in boxes]
    controls = [{"name": n, "ok": True} for n in ("[out]", "[in]", "[fit]", "[actual]", "[reset]", "[brightness]", "[contrast]", "[close]")]
    controls += [{"name": "viewer-stage", "ok": True} for _ in boxes]
    return {"doc": [width, width], "viewer": {"box": [0.0, top, float(width), float(height)], "sw": width, "cw": width, "ch": height - top},
            "panes": panes, "controls": controls}


def test_a_viewer_in_order_has_no_failures_single_or_side_by_side():   # P6-E
    for width, height in ((1280, 900), (800, 568), (375, 812), (320, 568)):
        for pair in (False, True):
            assert layout.check_viewer(_viewer(width, height, pair), width, height, pair) == [], (width, height, pair)


def test_the_viewer_case_fails_on_overflow_a_control_out_of_reach_or_a_picture_not_drawn():   # P6-E
    wide = _viewer(320, 568)
    wide["doc"] = [400, 320]
    assert any("documentElement overflows horizontally" in f for f in layout.check_viewer(wide, 320, 568, True))
    inside = _viewer(320, 568)
    inside["viewer"]["sw"] = 360
    assert any("#viewer overflows horizontally" in f for f in layout.check_viewer(inside, 320, 568, True))
    lost = _viewer(320, 568)
    lost["controls"][7] = {"name": "[close]", "ok": False, "got": "header"}
    assert any("[close] cannot be reached: its centre hits header" in f for f in layout.check_viewer(lost, 320, 568, True))
    flat = _viewer(320, 568)
    flat["panes"][1]["stage"][3] = flat["panes"][1]["stage"][1] + 10   # a stage 10 px tall: no picture to look at
    assert any("pane 2 is 320x10 px" in f for f in layout.check_viewer(flat, 320, 568, True))
    blank = _viewer()
    blank["panes"][0]["img"]["loaded"] = False
    assert any("pane 1's picture has not loaded" in f for f in layout.check_viewer(blank, 1280, 900, True))
    spill = _viewer()
    spill["panes"][0]["img"]["box"][2] += 400   # drawn past its stage: not whole at the fit
    assert any("pane 1's picture is not whole inside its stage" in f for f in layout.check_viewer(spill, 1280, 900, True))
    unsized = _viewer()
    unsized["panes"][1]["footer"] = "Similar X-ray #1"
    assert any("pane 2's footer gives no pixel size" in f for f in layout.check_viewer(unsized, 1280, 900, True))
    assert layout.check_viewer(dict(_viewer(), viewer=None), 1280, 900, True) == ["the viewer is not open"]
    assert any("1 pane(s), not 2" in f for f in layout.check_viewer(_viewer(pair=False), 1280, 900, True))


def test_two_pictures_are_side_by_side_from_700_px_and_stacked_below():   # P6-E
    stacked = _viewer(1280, 900)
    stacked["panes"] = _viewer(375, 900)["panes"]
    assert any("side by side at 1280 px" in f for f in layout.check_viewer(stacked, 1280, 900, True))
    beside = _viewer(375, 812)
    beside["panes"] = _viewer(1280, 812)["panes"]
    for pane in beside["panes"]:   # within the 375 px page, as two columns would be
        pane["box"] = [pane["box"][0] * 375 / 1280, pane["box"][1], pane["box"][2] * 375 / 1280, pane["box"][3]]
        pane["stage"] = [pane["box"][0], pane["box"][1], pane["box"][2], pane["box"][3] - 24]
        pane["img"]["box"] = [pane["box"][0] + 4, pane["box"][1] + 8, pane["box"][2] - 4, pane["box"][3] - 32]
    assert any("stacked below 700 px" in f for f in layout.check_viewer(beside, 375, 812, True))


def test_where_the_page_scrolls_a_pane_may_run_below_the_viewport_if_it_can_be_scrolled_to():   # P6-E
    tall = _viewer(320, 256)
    pane = tall["panes"][1]   # moved under the fold of a viewer that scrolls inside, as tall as it was
    for box in (pane["box"], pane["stage"], pane["img"]["box"]):
        box[1] += 450.0
        box[3] += 450.0
    assert layout.check_viewer(tall, 320, 256, True, scroll=True) == []
    assert any("leaves the viewport" in f for f in layout.check_viewer(tall, 320, 256, True))


def test_on_a_short_page_a_stacked_viewer_leaves_room_for_a_touch_to_scroll_it():   # P6 fix 1 (8)
    """A stage takes a touch for its picture (pan, pinch). Where the viewer scrolls inside itself with its panes stacked (a phone held
    sideways), a stage that fills most of it leaves a finger little but pictures to land on: each stage takes at most 60% of the viewer's
    height. Side by side (from 700 px) the viewer scrolls only past its bar and footer, and a viewer that does not scroll needs no such room."""
    for pair in (False, True):
        roomy = _viewer(320, 256, pair)   # a 216 px viewer, each stage 108 px
        assert layout.check_viewer(roomy, 320, 256, pair, scroll=True) == [], pair
        edge = _viewer(320, 256, pair)
        edge["panes"][0]["stage"][3] = edge["panes"][0]["stage"][1] + 129   # 59.7% of the viewer
        assert layout.check_viewer(edge, 320, 256, pair, scroll=True) == [], pair
        full = _viewer(320, 256, pair)
        full["panes"][0]["stage"][3] = full["panes"][0]["stage"][1] + 131   # 60.6%
        assert any("pane 1 takes 131 of the 216 px viewer (61%), over 60%: little of it is left for a touch to scroll it" in f
                   for f in layout.check_viewer(full, 320, 256, pair, scroll=True)), pair
        assert not any("touch" in f for f in layout.check_viewer(full, 320, 256, pair)), pair   # a viewer that does not scroll needs no room
    beside = _viewer(800, 400)   # side by side, stages of 286 px in a 360 px viewer that scrolls past its bar and footer
    assert layout.check_viewer(beside, 800, 400, True, scroll=True) == []
    assert layout.VIEWER_TOUCH_VIEWPORTS == [(568, 320)] and layout.MAX_TOUCH_STAGE_SHARE == 0.6   # a landscape phone, beside the short ones


def test_a_card_section_that_is_missing_wide_or_holds_a_bad_picture_fails():
    first = layout.SECTIONS[0]
    gone = _sections()
    gone["sections"][first] = None
    assert layout.check_sections(gone, 1280, layout.SECTIONS) == ["{} is not in the card".format(first)]
    wide = _sections()
    wide["sections"][first]["sw"] = 360   # a long word or a grid that does not wrap
    assert any("overflows inside: scrollWidth 360 > clientWidth 300" in f for f in layout.check_sections(wide, 1280, layout.SECTIONS))
    out = _sections(width=375)
    out["sections"][first]["box"][2] = 400.0
    assert any("leaves the card" in f for f in layout.check_sections(out, 375, layout.SECTIONS))
    unloaded = _sections()
    unloaded["sections"][first]["pictures"][0]["loaded"] = False
    assert any("picture 1 has not loaded" in f for f in layout.check_sections(unloaded, 1280, layout.SECTIONS))
    flat = _sections()
    flat["sections"][first]["pictures"][0]["box"] = [40.0, 320.0, 40.0, 480.0]
    assert any("picture 1 has no size" in f for f in layout.check_sections(flat, 1280, layout.SECTIONS))
    stray = _sections()
    stray["sections"][first]["pictures"][0]["box"] = [1200.0, 320.0, 1300.0, 480.0]
    assert any("picture 1 leaves the card" in f for f in layout.check_sections(stray, 1280, layout.SECTIONS))
    bare = _sections()
    bare["sections"][first]["pictures"] = []
    assert "{} shows no picture".format(first) in layout.check_sections(bare, 1280, layout.SECTIONS)


def test_a_page_that_overflows_or_draws_no_card_fails_the_sections_case():
    m = _sections(width=320)
    m["doc"] = [460, 320]
    assert "documentElement overflows horizontally: scrollWidth 460 > clientWidth 320" in layout.check_sections(m, 320, layout.SECTIONS)
    none = _sections()
    none["card"] = None
    assert layout.check_sections(none, 1280, layout.SECTIONS) == ["no card was drawn"]


def test_the_synthetic_turn_is_a_private_one_with_every_section_and_the_widest_text():
    events = layout.section_events()
    assert [e["data"]["seq"] for e in events] == list(range(1, len(events) + 1))
    assert events[0]["event"] == "message_start" and events[-1]["event"] == "message_stop"
    retrieve = next(e["data"]["detail"] for e in events if e["data"].get("stage") == "retrieve")
    assert len(retrieve["image_neighbors"]) == 12 and len(retrieve["report_matches"]) == 10   # as many as the options allow
    assert any(layout.LONG_WORD in m["report"] for m in retrieve["report_matches"])
    paths = [events[0]["data"]["image"]["urls"][v] for v in ("thumb", "model_input")] + [n["image_url"] for n in retrieve["image_neighbors"]]
    assert all(p.startswith("/v1/") for p in paths)   # same-origin paths: the injected loader answers them, nothing is fetched


# ---- the loop, with a page that answers from a script ------------------------------------------------------------------------------------

class FakePage:
    """Stands in for Browser in run_checks: it follows the hidden attribute of #drawer and #stop that the harness writes, and answers
    each measurement from _measure, so that the loop's own wiring (which case claims what) is what is under test."""

    def __init__(self, stop_renders: bool = True, floor: bool = True, save_renders: bool = True, busy_fails: bool = False) -> None:
        self.stop_renders, self.floor, self.save_renders, self.busy_fails = stop_renders, floor, save_renders, busy_fails
        self.width, self.height = 0, 0
        self.drawer, self.stop = False, False
        self.opened = 0
        self.sections_drawn, self.sections_measured, self.picker_measured = 0, 0, 0
        self.composer_set_up, self.composer_measured = [], 0
        self.flow, self.chips, self.held, self.injected_drawer, self.state = [], 0, False, False, "hint"
        self.busy_measured, self.busy_flows, self.keep_scroll = [], [], []
        self.viewers_opened, self.viewer_measured = [], 0

    def open(self, url: str, width: int, height: int, scheme: str) -> None:
        self.width, self.height, self.drawer, self.stop = width, height, False, False
        self.flow, self.chips, self.held, self.state = [], 0, False, "hint"   # a new page: what the user did in the last one is gone
        self.opened += 1

    def key(self, *args: Any, **kwargs: Any) -> None:
        pass

    def click(self, selector: str) -> None:
        """A real click: the loop's busy cases do what a user does, so the clicks are what a case is (P9-G5b fix 1)."""
        assert selector != "#send" or self.held, "Send was clicked with the upload not yet held: the turn would have been answered"
        self.flow.append(selector)

    def wait_for(self, expression: str, timeout: float = 15.0, what: Optional[str] = None, interval: float = 0.04) -> bool:
        if self.busy_fails:
            raise TimeoutError("waited {:g} s for {}".format(timeout, what or expression))
        return True

    def composer_state(self) -> str:
        """Which case the page is in, told by what was done to it: the busy cases are the clicks that led to them."""
        if not self.flow:
            return self.state
        if self.flow == ["#send"]:
            return "busy-8-chips" if self.chips == len(layout.EIGHT_CHIPS) else "busy"
        if self.flow == ["#settings", "#drawer-save", "#send"]:
            return "busy-after-save"
        if self.flow == ["#send", "#settings", "#drawer-save"]:
            return "busy-save"
        raise AssertionError("a busy case that does what no case is: {}".format(self.flow))

    def evaluate(self, expression: str) -> bool:
        if "#drawer').hidden" in expression:
            self.drawer = expression.endswith("= false")
        if "#stop').hidden" in expression:
            self.stop = expression.endswith("= false")
        return True

    def run(self, function: str, argument: Any) -> Any:
        if function == layout.MEASURE_JS:
            scroll = bool(argument["scroll"])
            m = _measure(self.width, self.height, self.stop and self.stop_renders, self.drawer)
            if scroll:   # a short viewport: the page scrolls, the report keeps its natural height and the banner stays at the top
                m["#conversation"] = {"sw": self.width, "cw": self.width, "sh": 200, "ch": 200}
                m["banner"] = {"scrolled": 0, "top": 0}
            return m
        if function == layout.MEASURE_TALL_JS:
            return _tall(self.height, report=120 if self.floor else 40)
        if function == layout.MEASURE_SAVE_JS:
            m = _save(self.width, self.height, self.drawer)
            if self.drawer and not self.save_renders:
                m["rendered"], m["box"], m["hit"] = False, None, None
            return m
        if function == layout.MEASURE_SECTIONS_JS:
            self.sections_measured += 1
            return _sections(self.width, argument["sections"])
        if function == layout.MEASURE_PICKER_JS:
            self.picker_measured += 1
            return _picker(self.width, self.height)
        if function == layout.INJECT_SECTIONS_JS:
            self.sections_drawn += 1
        if function == layout.INJECT_COMPOSER_JS:
            self.composer_set_up.append((self.width, self.height, argument["drawer"], argument["state"], argument["name"]))
            self.state, self.chips, self.injected_drawer = argument["state"], len(argument["chips"]), argument["drawer"]
        if function == layout.SET_CHIPS_JS:
            self.chips = len(argument["chips"])
        if function == layout.HOLD_UPLOAD_JS:
            self.held = True
        if function == layout.MEASURE_COMPOSER_JS:
            self.composer_measured += 1
            state = self.composer_state()
            if state.startswith("busy"):
                self.busy_measured.append((self.width, self.height, self.injected_drawer, state))
                self.busy_flows.append((state, self.chips, tuple(self.flow)))
            self.keep_scroll.append((state, argument["keep"]))
            return _composer(self.width, self.height, state)
        if function == layout.INJECT_VIEWER_JS:
            self.viewers_opened.append((self.width, self.height, argument["pair"]))
            self.pair = argument["pair"]
        if function == layout.MEASURE_VIEWER_JS:
            self.viewer_measured += 1
            return _viewer(self.width, self.height, self.pair)
        return True


def test_the_loop_runs_every_case_and_passes_on_a_page_in_order():
    page = FakePage()
    cases, failures = layout.run_checks(page, "http://x/")
    viewports = len(layout.WIDTHS) * len(layout.HEIGHTS) + len(layout.SHORT_VIEWPORTS)
    expected = (len(layout.WIDTHS) * len(layout.HEIGHTS) * len(layout.SCHEMES) * 2
                + len(layout.SHORT_VIEWPORTS) * len(layout.SCHEMES) * 2 + len(layout.TALL_WIDTHS) * len(layout.SCHEMES)
                + viewports * len(layout.SCHEMES)    # P4-G: one more case for each viewport in each scheme, for the Save button
                + viewports * len(layout.SCHEMES)    # P6-B: and one for the card's own sections
                + viewports * len(layout.SCHEMES)    # P6 fix 1: and one with the test-split picker open
                + len(layout.COMPOSER_CASES) * len(layout.SCHEMES) * len(layout.COMPOSER_STATES)    # P6-A: the composer's rows
                + viewports * len(layout.SCHEMES) * len(layout.VIEWER_STATES)    # P6-E: the viewer open, one picture and two
                + len(layout.VIEWER_TOUCH_VIEWPORTS) * len(layout.SCHEMES) * len(layout.VIEWER_STATES)    # P6 fix 1: and on a landscape phone
                + len(layout.COMPOSER_BUSY_CASES) * len(layout.SCHEMES)   # P6 fix 2: the composer busy, where it is narrowest
                + len(layout.COMPOSER_STRESS_CASES) * len(layout.STRESS_STATES) * len(layout.SCHEMES))   # P9-G5b fix 1: and with more in it than its rows
    assert cases == expected == 366
    assert failures == []
    busy_loads = (len(layout.COMPOSER_BUSY_CASES) + len(layout.COMPOSER_STRESS_CASES) * len(layout.STRESS_STATES)) * len(layout.SCHEMES)
    assert page.opened == ((len(layout.WIDTHS) * len(layout.HEIGHTS) + len(layout.SHORT_VIEWPORTS)) * len(layout.SCHEMES)
                           + len(layout.TALL_WIDTHS) * len(layout.SCHEMES)
                           + len(layout.COMPOSER_CASES) * len(layout.SCHEMES)
                           + busy_loads   # a busy page is busy for good (an upload held, a chat made): each busy case has a page of its own
                           + len(layout.VIEWER_TOUCH_VIEWPORTS) * len(layout.SCHEMES))   # one load per viewport and scheme, the drawer and Stop are toggled in it
    assert page.sections_drawn == viewports * len(layout.SCHEMES)                  # the card drawn once per viewport and scheme,
    assert page.sections_measured == viewports * len(layout.SCHEMES) * 2           # and measured with the drawer closed and open
    assert page.picker_measured == viewports * len(layout.SCHEMES)
    assert page.composer_measured == len(layout.COMPOSER_CASES) * len(layout.SCHEMES) * len(layout.COMPOSER_STATES) + busy_loads
    assert {(w, h, drawer, state) for w, h, drawer, state, _ in page.composer_set_up} == (
        {(w, h, drawer, state) for w, h, drawer in layout.COMPOSER_CASES for state in layout.COMPOSER_STATES}   # the busy cases attach the image as "preview" does
        | {(w, h, drawer, "preview") for w, h, drawer in layout.COMPOSER_BUSY_CASES + layout.COMPOSER_STRESS_CASES})
    assert {name for *_, name in page.composer_set_up} == {layout.LONG_FILE}   # every case with the long file name
    assert {(w, h, drawer, state) for w, h, drawer, state in page.busy_measured} == (   # P9-G5b fix 1: each busy case, reached by its own clicks
        {(w, h, drawer, "busy") for w, h, drawer in layout.COMPOSER_BUSY_CASES}
        | {(w, h, drawer, state) for w, h, drawer in layout.COMPOSER_STRESS_CASES for state in layout.STRESS_STATES})
    assert sorted(set(page.busy_flows)) == sorted([
        ("busy", len(layout.CHIPS), ("#send",)),
        ("busy-after-save", len(layout.SEVEN_CHIPS), ("#settings", "#drawer-save", "#send")),   # Save, then Send within 4 s: the review's routine case
        ("busy-save", len(layout.CHIPS), ("#send", "#settings", "#drawer-save")),               # Save while the upload goes up
        ("busy-8-chips", len(layout.EIGHT_CHIPS), ("#send",))])
    assert all(keep == state.startswith("busy") for state, keep in page.keep_scroll)   # the page's own scroll is measured as it left it, in busy cases only
    assert page.viewer_measured == (viewports + len(layout.VIEWER_TOUCH_VIEWPORTS)) * len(layout.SCHEMES) * len(layout.VIEWER_STATES)
    assert {(w, h) for w, h, _ in page.viewers_opened} == ({(w, h) for h in layout.HEIGHTS for w in layout.WIDTHS} | set(layout.SHORT_VIEWPORTS)
                                                           | set(layout.VIEWER_TOUCH_VIEWPORTS))
    assert sorted({pair for *_, pair in page.viewers_opened}) == [False, True]


def test_the_loop_reports_a_stop_button_that_never_renders_in_the_cases_that_show_it():
    cases, failures = layout.run_checks(FakePage(stop_renders=False), "http://x/")
    assert failures and all("#stop is not rendered, but the case shows it" in f and "stop=shown" in f for f in failures)
    assert cases == 366


def test_the_loop_reports_a_page_that_never_gets_busy_as_a_failure_of_the_busy_cases_and_does_not_stop():   # P9-G5b fix 1 (iii)
    """The busy cases click, and a page that does not do what a click should (Send never held, the drawer never opening) times the wait out:
    that is a failure of that case, in words, and the sweep goes on with the rest."""
    cases, failures = layout.run_checks(FakePage(busy_fails=True), "http://x/")
    assert cases == 366
    assert failures and all("the page did not get to the busy composer" in f and "waited 15 s for" in f for f in failures), failures
    named = {state for state in layout.BUSY_STATES if any("composer {}:".format(state) in f for f in failures)}
    assert named == set(layout.BUSY_STATES), named   # every busy state is named, and every failure is of one


def test_the_loop_reports_a_save_button_that_never_renders_in_the_save_cases_only():
    cases, failures = layout.run_checks(FakePage(save_renders=False), "http://x/")
    assert cases == 366 and len(failures) == len(layout.WIDTHS) * len(layout.HEIGHTS) + len(layout.SHORT_VIEWPORTS)   # once per viewport: both schemes fail alike
    assert all("light+dark" in f and "save" in f and "#drawer-save is not rendered" in f for f in failures), failures


def test_the_loop_reports_a_deleted_floor_only_in_the_tall_cases():
    cases, failures = layout.run_checks(FakePage(floor=False), "http://x/")
    assert len(failures) == len(layout.TALL_WIDTHS)   # a failure seen in both colour schemes is reported once
    assert all("light+dark tall composer: the report area is 40 px tall, under the 120 px floor" in f for f in failures), failures
