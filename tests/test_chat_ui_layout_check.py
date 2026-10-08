"""CHAT_UI_PLAN.md P4-E: the Chrome harness the two browser checks share (scripts/chat_ui_cdp.py) and the parts of the layout check
(scripts/chat_ui_layout_check.py) that are plain Python. No Chrome and no app: a fake browser is a shell script, or an object
that returns canned measurements.

The harness bugs these pin were found by the P4-A re-review: a Chrome orphaned when the DevTools connection failed after it
started, `display: none` mutants of #stop and #drawer that passed, and a report floor that could be deleted unseen.
"""
import json
import os
import signal
import socket
import subprocess
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

import pytest

from scripts import chat_ui_layout_check as layout
from scripts import chat_ui_cdp
from scripts.chat_ui_cdp import Browser, kill_group


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


# ---- the harness ------------------------------------------------------------------------------------------------------------------

def _fake_chrome(tmp_path: Path) -> Path:
    """A "Chrome" that starts, records its pid and profile directory, and never opens a DevTools page."""
    script = tmp_path / "chrome"
    script.write_text('#!/bin/sh\necho $$ > "{pid}"\nfor a in "$@"; do case "$a" in --user-data-dir=*) '
                      'echo "${{a#--user-data-dir=}}" > "{profile}";; esac; done\nexec sleep 300\n'.format(
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
                      'echo "$d" > "{profile}"; echo {port} > "$d/DevToolsActivePort";; esac; done\nexec sleep 300\n'.format(
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
    proc = subprocess.Popen(["/bin/sh", "-c", 'sleep 300 & echo $! > "{}"; wait'.format(child_file)], start_new_session=True)
    assert _until(lambda: child_file.exists() and child_file.read_text().strip())
    child = int(child_file.read_text())
    assert _alive(child)
    kill_group(proc)
    assert proc.returncode is not None
    assert _until(lambda: not _alive(child)), "the child of the process outlived it"


def test_kill_group_is_safe_on_nothing_and_on_a_process_that_has_gone():
    kill_group(None)
    proc = subprocess.Popen(["/bin/sh", "-c", "exit 0"], start_new_session=True)
    proc.wait()
    kill_group(proc)   # reaped already: no signal goes to a pid that may be someone else's
    kill_group(proc)


class ScriptedSocket:
    """The DevTools end of a Browser: what it is sent is recorded; what it receives is scripted, per call, as a list of messages."""

    def __init__(self, replies: Any) -> None:
        self.replies, self.sent = list(replies), []
        self.incoming = []   # type: Any
        self.sock = SimpleNamespace(settimeout=lambda seconds: None)   # Browser.close shortens the wait for an answer

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

POLITE = '#!/bin/sh\nmarker="$1"\nmkdir -p "$marker"\ntrap \'rm -rf "$marker"; exit 0\' TERM\nwhile :; do sleep 0.05; done\n'
STUBBORN = '#!/bin/sh\nmarker="$1"\nmkdir -p "$marker"\ntrap \'\' TERM\nwhile :; do sleep 0.05; done\n'


def _start(tmp_path: Path, body: str, name: str = "fake") -> Any:
    """A fake Chrome or app: a shell script that makes a marker directory, then waits. A TERM that it handles removes the marker and
    exits 0, as Chrome removes its singleton directory; one that it ignores leaves the marker, as a Chrome that hangs would."""
    script = tmp_path / name
    script.write_text(body)
    script.chmod(0o755)
    marker = tmp_path / (name + ".marker")
    proc = subprocess.Popen(["/bin/sh", str(script), str(marker)], start_new_session=True)
    assert _until(marker.exists), "the fake process did not start"
    return proc, marker


def _bare_browser(proc: Any, profile: Path, ws: Any = None) -> Browser:
    browser = Browser.__new__(Browser)   # no Chrome: close() is under test, not the start
    browser.proc, browser.ws, browser.profile, browser.next_id = proc, ws, str(profile), 0
    browser.events_seen = deque(maxlen=chat_ui_cdp.KEEP_EVENTS)
    profile.mkdir()
    return browser


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


def test_stop_group_is_safe_on_nothing_and_signals_nothing_after_the_process_was_reaped(tmp_path):
    chat_ui_cdp.stop_group(None)
    proc = subprocess.Popen(["/bin/sh", "-c", "exit 3"], start_new_session=True)
    proc.wait()
    chat_ui_cdp.stop_group(proc)
    assert proc.returncode == 3   # untouched: no signal went to a pid that may be someone else's now


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


# ---- the loop, with a page that answers from a script ------------------------------------------------------------------------------------

class FakePage:
    """Stands in for Browser in run_checks: it follows the hidden attribute of #drawer and #stop that the harness writes, and answers
    each measurement from _measure, so that the loop's own wiring (which case claims what) is what is under test."""

    def __init__(self, stop_renders: bool = True, floor: bool = True) -> None:
        self.stop_renders, self.floor = stop_renders, floor
        self.width, self.height = 0, 0
        self.drawer, self.stop = False, False
        self.opened = 0

    def open(self, url: str, width: int, height: int, scheme: str) -> None:
        self.width, self.height, self.drawer, self.stop = width, height, False, False
        self.opened += 1

    def key(self, *args: Any, **kwargs: Any) -> None:
        pass

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
        return True


def test_the_loop_runs_every_case_and_passes_on_a_page_in_order():
    page = FakePage()
    cases, failures = layout.run_checks(page, "http://x/")
    expected = (len(layout.WIDTHS) * len(layout.HEIGHTS) * len(layout.SCHEMES) * 2
                + len(layout.SHORT_VIEWPORTS) * len(layout.SCHEMES) * 2 + len(layout.TALL_WIDTHS) * len(layout.SCHEMES))
    assert cases == expected == 94
    assert failures == []
    assert page.opened == ((len(layout.WIDTHS) * len(layout.HEIGHTS) + len(layout.SHORT_VIEWPORTS)) * len(layout.SCHEMES)
                           + len(layout.TALL_WIDTHS) * len(layout.SCHEMES))   # one load per viewport and scheme, the drawer and Stop are toggled in it


def test_the_loop_reports_a_stop_button_that_never_renders_in_the_cases_that_show_it():
    cases, failures = layout.run_checks(FakePage(stop_renders=False), "http://x/")
    assert failures and all("#stop is not rendered, but the case shows it" in f and "stop=shown" in f for f in failures)
    assert cases == 94


def test_the_loop_reports_a_deleted_floor_only_in_the_tall_cases():
    cases, failures = layout.run_checks(FakePage(floor=False), "http://x/")
    assert len(failures) == len(layout.TALL_WIDTHS)   # a failure seen in both colour schemes is reported once
    assert all("light+dark tall composer: the report area is 40 px tall, under the 120 px floor" in f for f in failures), failures
