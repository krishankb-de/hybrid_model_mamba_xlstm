"""What the Chrome-driven checks of the chat page share (CHAT_UI_PLAN.md P4-A, P4-E, P6-F): a DevTools websocket client, a
private headless Chrome with one page, the tiny app on a free loopback port, and the refusals of a --url. Local development
tools: nothing here is imported by the app or run by validate.sh.

    scripts/chat_ui_layout_check.py    geometry at many viewports (writes nothing)
    scripts/chat_ui_browser_check.py   the browser checklist, with screenshots
    scripts/chat_ui_features_check.py  the five image features, with screenshots

A --url is an app that someone else started, and each check sends it uploads and turns or loads its page, and the two that write
evidence screenshot what it shows. So each refuses one first, before it starts or sends anything, with a fixed line and the status 2:
an address that is not http on this machine's loopback with a port of its own, or that has one of REFUSED_PORTS (url_problem), and an
app whose own /v1/models is not the tiny engine's without a checkpoint (tiny_only_problem, models_problem); the two that write evidence
also refuse a --url without --no-evidence (url_needs_no_evidence). Nothing the server says is in a refusal.

Every process this module starts runs in a session of its own, so stopping it reaches the whole group (Chrome's helpers
included) and nothing outlives a failure path. It is stopped politely first, and killed last: Chrome is asked to close over
DevTools (it has CLOSE_WAIT_S in all, by the clock, to answer and go), then sent SIGTERM, and only a process that is still there
after the wait gets SIGKILL. A Chrome that is killed leaves its singleton directory (com.google.Chrome.*, a SingletonSocket and
a SingletonCookie) in $TMPDIR, one more on every run; one that leaves on request removes it. Browser.close() and App.stop() are
safe to call at any time, on an object whose constructor failed half-way too, and Browser.close() stops Chrome and removes its
profile even when an interrupt (the SIGTERM handler's SystemExit, Ctrl-C) cuts it short.

What a process leaves in its session when it goes (the helpers of a Chrome that went on request, or died) is swept too: with SIGKILL,
once, after every way out, and while the leader is not yet reaped. The wait for an exit is made without reaping (exited, wait_exit: waitid
with WNOWAIT), because a reaped leader's pid is free to be anyone's and the session it names with it, so nothing is signalled after
the reap, whatever the order of the calls. Standard library only: the websocket client is the little of RFC 6455 a DevTools session needs.
"""
import base64
import hashlib
import ipaddress
import json
import os
import shutil
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import time
import urllib.request
from collections import deque
from typing import Any, Deque, Dict, List, Optional
from urllib.parse import urlparse

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAC_CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
KEEP_EVENTS = 5000   # DevTools events kept per Browser (Browser.events): the oldest go first
CLOSE_WAIT_S = 3.0   # how long, in all, Chrome has to answer the DevTools Browser.close and to exit, before it is sent SIGTERM
TERM_WAIT_S = 5.0    # how long a process has to exit after SIGTERM, before it is sent SIGKILL

# What a --url may not be (see url_problem, tiny_only_problem). The two ports are the laptop's dev server and the tunnel to the cluster's private
# server: the checks drive an app's page (two of them send it uploads and turns, and write screenshots of what it shows), and neither may be the
# app. No other script names these numbers.
REFUSED_PORTS = (8000, 8010)
TINY_MODEL = "tiny"                    # app.engine.TinyEngine.name: the one model of the app a check may be given with --url
TINY_NOTE = "tiny random-init model"   # app.engine.TinyEngine.drift_note: what that engine says of itself in its card
MODELS_BYTES = 1024 * 1024             # the most of a /v1/models answer that is read (the tiny app's is under 2 KB)
MODELS_UNREADABLE = ("the app at this address did not answer GET /v1/models with JSON (no answer, an error status, a redirect, or something "
                     "else), so it cannot be shown to be the tiny app; these checks run on the tiny app only")
NOT_THE_TINY_APP = ("the app at this address is not the tiny app: its /v1/models names a model that is not the tiny engine with no checkpoint. "
                    "These checks drive the tiny stack (random weights, no checkpoint, no data) only, and never an app that serves anything else")
URL_NEEDS_NO_EVIDENCE = ("--url needs --no-evidence: the pictures and the checklist of a run are evidence of the tiny stack that the check "
                         "started itself, and an app that someone else started is not that")

# What Chrome is started with, besides its profile, its DevTools port and the blank page. A fresh profile is not a quiet one: some seconds after
# Chrome starts, its own log shows its default-app and extension installer at work (a burst of about 70 ms) and a background check-in to
# android.clients.google.com, and a fetch that a page makes around then can fail inside Chrome: TypeError "Failed to fetch" after 1 ms,
# net::ERR_ABORTED with canceled set, no request in the NetLog and none at the server, while the same document's other fetches work. The failure
# coincides with that activity; that the activity causes it is not shown. The features check's first click fetches about 7 s after Chrome starts,
# and that fetch failed in about one run of three with Chrome's default flags (10 of 30 runs) and in none with these five together (0 of 30, runs
# alternating). Which of the five matters is not established. They are the flags that automation tools start Chrome with.
CHROME_FLAGS = ("--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check", "--disable-extensions", "--disable-default-apps",
                "--disable-background-networking", "--disable-component-update", "--disable-sync")
HOLD_FILE = "hold"   # in the app's home: while it exists, the turns of the tiny engine wait (App.hold, hold_turns)

# The app as the checks run it. The uvicorn CLI cannot pass create_app its arguments (the step delay of the tiny engine is
# one), so this is a short launcher. argv: home, step delay in seconds, port, and optionally a directory of static files;
# --tiny-gallery, anywhere after them, gives the app the synthetic gallery and the keyword labeller (P5-E). It also lets a check hold the
# turns of the app (hold_turns): the file HOLD_FILE in the home is the switch.
LAUNCH = """\
import os
import sys
from pathlib import Path
import uvicorn
import app.server as server
from scripts.chat_ui_cdp import hold_turns
args = [a for a in sys.argv[1:] if a != "--tiny-gallery"]
if len(args) > 3:
    server.STATIC_DIR = Path(args[3])
hold_turns(os.path.join(args[0], "hold"))
uvicorn.run(server.create_app(engine="tiny", home=args[0], tiny_step_delay_s=float(args[1]),
                              tiny_gallery="--tiny-gallery" in sys.argv[1:]),
            host="127.0.0.1", port=int(args[2]), log_level="warning")
"""


def hold_turns(path: str) -> None:
    """Run in the app's process, before it serves (LAUNCH): while the file `path` exists, a turn of the tiny engine waits after each snapshot it
    has sent, and goes on when the file is gone, or at once when the turn is stopped. The turn stays running for as long as it is held, and
    the page can find it so. The tiny engine runs on its own clock (a step every STEP_DELAY_S), and a check that needs a turn to be running
    at a moment that it does not control (a reload, which on a loaded machine takes longer than a short turn) holds it there instead of racing it."""
    import app.engine as engine
    generate = engine.TinyEngine.generate

    def held(self: Any, enc: Any, opts: Any, on_snapshot: Any, cancel: Any) -> Any:
        def snapshot(step: int, text: str) -> None:
            on_snapshot(step, text)
            while os.path.exists(path) and not cancel.is_set():
                cancel.wait(0.02)

        return generate(self, enc, opts, snapshot, cancel)

    engine.TinyEngine.generate = held


class WebSocket:
    """A text-frame websocket client: the handshake, masked sends, and receives with fragments, pings and close. With .deadline set (a
    time.monotonic() value) no wait for the peer's data goes past it, however much the peer sends meanwhile."""

    deadline = None  # type: Optional[float]

    def __init__(self, url: str, timeout: float = 30.0) -> None:
        parts = urlparse(url)
        self.sock = socket.create_connection((parts.hostname, parts.port), timeout=timeout)
        try:
            key = base64.b64encode(os.urandom(16)).decode("ascii")
            request = ("GET {} HTTP/1.1\r\nHost: {}:{}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                       "Sec-WebSocket-Key: {}\r\nSec-WebSocket-Version: 13\r\n\r\n")
            self.sock.sendall(request.format(parts.path or "/", parts.hostname, parts.port, key).encode("ascii"))
            head = b""
            while b"\r\n\r\n" not in head:
                chunk = self.sock.recv(4096)
                if not chunk:
                    raise ConnectionError("DevTools closed the connection during the handshake")
                head += chunk
            head, self.buffer = head.split(b"\r\n\r\n", 1)
            lines = head.decode("latin-1").split("\r\n")
            if " 101 " not in lines[0]:
                raise ConnectionError("DevTools refused the websocket: " + lines[0])
            headers = {k.strip().lower(): v.strip() for k, v in (line.split(":", 1) for line in lines[1:] if ":" in line)}
            accept = base64.b64encode(hashlib.sha1((key + WS_GUID).encode("ascii")).digest()).decode("ascii")
            if headers.get("sec-websocket-accept") != accept:
                raise ConnectionError("DevTools answered with a wrong Sec-WebSocket-Accept")
        except BaseException:
            self.close()
            raise

    def _arm(self) -> None:
        """Before a wait for the peer's data, or a write to it: with a deadline, cut the wait off at it. A socket timeout alone is per
        operation, so a peer that keeps sending (events, pings) and never answers would never time out, and one that reads nothing would
        hold a write for as long as the socket's own timeout."""
        if self.deadline is not None:
            remaining = self.deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("DevTools did not answer in time")
            self.sock.settimeout(remaining)

    def _read(self, n: int) -> bytes:
        while len(self.buffer) < n:
            self._arm()
            chunk = self.sock.recv(65536)
            if not chunk:
                raise ConnectionError("DevTools closed the connection")
            self.buffer += chunk
        data, self.buffer = self.buffer[:n], self.buffer[n:]
        return data

    def _send_frame(self, opcode: int, payload: bytes) -> None:
        n = len(payload)
        head = bytearray([0x80 | opcode])
        if n < 126:
            head.append(0x80 | n)
        elif n < 65536:
            head += bytes([0x80 | 126]) + struct.pack(">H", n)
        else:
            head += bytes([0x80 | 127]) + struct.pack(">Q", n)
        mask = os.urandom(4)
        self.sock.sendall(bytes(head) + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(payload)))

    def send(self, text: str) -> None:
        self._arm()
        self._send_frame(0x1, text.encode("utf-8"))

    def recv(self) -> str:
        message = b""
        while True:
            b0, b1 = self._read(2)
            fin, opcode, n = b0 & 0x80, b0 & 0x0F, b1 & 0x7F
            if n == 126:
                n = struct.unpack(">H", self._read(2))[0]
            elif n == 127:
                n = struct.unpack(">Q", self._read(8))[0]
            mask = self._read(4) if b1 & 0x80 else b""
            payload = self._read(n)
            if mask:
                payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
            if opcode == 0x8:
                raise ConnectionError("DevTools closed the connection")
            if opcode == 0x9:
                self._send_frame(0xA, payload)   # answer a ping
            elif opcode in (0x0, 0x1):
                message += payload
                if fin:
                    return message.decode("utf-8")

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass


def kill_group(proc: Optional["subprocess.Popen"]) -> None:
    """The last resort: SIGKILL a process started with start_new_session=True and everything else in its session, then reap it.
    Safe on None and on a process that is gone. The group is signalled before the process is reaped, while its pid cannot be
    anyone else's, and never after: a reaped leader's pid may belong to someone else by then."""
    if proc is None:
        return
    if proc.returncode is None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)   # a session leader: its pid is its process group
        except OSError:
            pass
        try:
            proc.kill()
        except OSError:
            pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass


def exited(proc: "subprocess.Popen") -> bool:
    """Whether the process has exited, without reaping it. proc.poll() answers the same but waits for the process as it does, and a process
    that has been waited for is gone from the kernel's tables: its pid is free to be anyone's, and so is the id of the process group and
    the session that pid names. Until then the pid is still this one's, so what is left in its group can be signalled safely (kill_group
    does). Where the platform has no waitid, the process is reaped to find out (poll), and nothing is swept after it."""
    if proc.returncode is not None:
        return True
    if not hasattr(os, "waitid"):
        return proc.poll() is not None
    try:
        return os.waitid(os.P_PID, proc.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None
    except ChildProcessError:   # waited for behind Popen's back: poll() learns it and settles returncode, so nothing is signalled after it
        return proc.poll() is not None


def wait_exit(proc: "subprocess.Popen", timeout: float) -> bool:
    """Wait up to `timeout` seconds for the process to exit, without reaping it (see exited); -> whether it has."""
    deadline = time.monotonic() + max(0.0, timeout)
    while not exited(proc):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.02)
    return True


def stop_group(proc: Optional["subprocess.Popen"], term_wait: Optional[float] = None) -> None:
    """Stop a process started with start_new_session=True, and its session, politely first: SIGTERM to the group, a wait of up to
    term_wait seconds (TERM_WAIT_S) for the process to go and clean up after itself, then SIGKILL for what is still there. That last step
    is made after every way out, also of a process that went by itself or on request: it is the sweep of what the process left in its
    session, made while the leader is not yet reaped (the wait above does not reap it), and it is made even when an interrupt cuts the
    wait short. Safe on None and on a process that is gone (nothing is signalled once the process was reaped)."""
    if proc is None:
        return
    try:
        if not exited(proc):
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except OSError:
                pass
            wait_exit(proc, TERM_WAIT_S if term_wait is None else term_wait)
    finally:
        kill_group(proc)


def exit_on_sigterm() -> None:
    """Make SIGTERM raise SystemExit, so that a check's `finally` runs and stops Chrome and the app: left at its default, a SIGTERM
    ends the process at once and orphans both (each runs in a session of its own, which the terminal's signals do not reach).

    The handler is one-shot. The clean-up that the SystemExit starts takes several seconds (the DevTools close, the TERM and its wait,
    the KILL), and a second SIGTERM in that time, from a harness that is impatient, used to raise a second SystemExit in the middle of it,
    which is where the processes were left running. After the first, SIGTERM is taken by a handler that does nothing. It is a handler and
    not SIG_IGN, because an ignored signal is inherited by what the process starts afterwards, which would then ignore its TERM too."""
    def absorb(signum: int, frame: Any) -> None:
        pass

    def handler(signum: int, frame: Any) -> None:
        signal.signal(signal.SIGTERM, absorb)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, handler)


def find_chrome() -> Optional[str]:
    for candidate in (os.environ.get("CHROME"), MAC_CHROME, shutil.which("google-chrome"), shutil.which("chromium"),
                      shutil.which("chromium-browser")):
        if candidate and os.path.exists(candidate):
            return candidate
    return None


# ---- the refusals of a --url: one implementation, called by the three checks -----------------------------------------------------

def url_problem(url: str) -> Optional[str]:
    """Why a --url is refused, or None: it must be an http address on this machine's loopback with a port of its own, and not one of
    REFUSED_PORTS. Pure: nothing is sent to it. The reason names the address or the ports, never what an app says."""
    try:
        parts = urlparse(url)
        port = parts.port
    except ValueError:
        return "not a URL: {!r}".format(url)
    if parts.scheme != "http" or not parts.hostname:
        return "not an http URL: {!r}".format(url)
    host = parts.hostname
    try:
        loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = host == "localhost"
    if not loopback:
        return "{} is not a loopback address: these checks talk to an app on this machine only".format(host)
    if port is None or port == 0:
        return "the URL names no port: give the port of the app, as in http://127.0.0.1:PORT/"
    if port in REFUSED_PORTS:
        return "port {} is refused: {} are the dev server's port and the tunnel's to the cluster's server, and these checks never drive either".format(
            port, " and ".join(str(p) for p in REFUSED_PORTS))
    return None


def models_problem(models: Any) -> Optional[str]:
    """Why a /v1/models answer is not the tiny app's (NOT_THE_TINY_APP, a fixed line), or None. The default model and every model listed must be
    the tiny engine with no checkpoint, which its card says by five fields, each there: name is TINY_MODEL, checkpoint, checkpoint_sha256 and
    train_experiment are null (a real engine has a path or a file name, a hash and its run, and public mode keeps None as None and cuts a path to
    its file name), and drift_note is what the tiny engine says of itself, TINY_NOTE. A field that is missing, a card that is not an object and an
    answer that lists no model are refused: nothing is taken on trust."""
    cards = models.get("models") if isinstance(models, dict) else None
    if not isinstance(cards, list) or not cards or models.get("default_model") != TINY_MODEL:
        return NOT_THE_TINY_APP
    for card in cards:
        if not isinstance(card, dict) or card.get("name") != TINY_MODEL or card.get("drift_note") != TINY_NOTE:
            return NOT_THE_TINY_APP
        if any(key not in card or card[key] is not None for key in ("checkpoint", "checkpoint_sha256", "train_experiment")):
            return NOT_THE_TINY_APP
    return None


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:   # an answer that sends the reader elsewhere is not an answer of this address
        return None


# No proxy (an app on this machine is asked directly, whatever the environment says) and no redirect.
_MODELS_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect)


def tiny_only_problem(url: str, timeout: float = 10.0) -> Optional[str]:
    """Why the app at this --url is not the tiny app, as a fixed line, or None: asks it for GET /v1/models, and nothing else, and holds the answer
    to models_problem. An app that does not answer with status 200 and JSON of at most MODELS_BYTES is refused (MODELS_UNREADABLE) as one that is
    not the tiny app, and so is one whose JSON is not the tiny app's (NOT_THE_TINY_APP). Nothing the app says, and no error it raises, is in the
    line. An address that url_problem refuses is refused here first, with its reason, and nothing is sent to it: the dev server's port is no
    more asked for its models than for a turn."""
    problem = url_problem(url)
    if problem:
        return problem
    base = url if url.endswith("/") else url + "/"
    try:
        with _MODELS_OPENER.open(base + "v1/models", timeout=timeout) as response:
            raw = response.read(MODELS_BYTES + 1)
            if response.status != 200 or len(raw) > MODELS_BYTES:
                return MODELS_UNREADABLE
            models = json.loads(raw)
    except Exception:   # an error status, a refused connection, a timeout, a redirect, text that is not JSON: all the same refusal
        return MODELS_UNREADABLE
    return models_problem(models)


def url_needs_no_evidence(parser: Any, args: Any) -> None:
    """For a check that writes evidence: a --url without --no-evidence is a usage error (the parser's, status 2), the way --only without it is."""
    if args.url and not args.no_evidence:
        parser.error(URL_NEEDS_NO_EVIDENCE)


class Browser:
    """A private headless Chrome and one DevTools session on its page. DevTools events that arrive while a call waits for its
    answer are kept in .events (the last KEEP_EVENTS), for events()."""

    def __init__(self, chrome: str, startup_timeout: float = 30.0, label: str = "chat_ui_chrome") -> None:
        self.profile = tempfile.mkdtemp(prefix=label + "_")
        self.proc = None  # type: Optional[subprocess.Popen]
        self.ws = None  # type: Optional[WebSocket]
        self.next_id = 0
        self.startup_timeout = startup_timeout
        self.events_seen: Deque[Dict[str, Any]] = deque(maxlen=KEEP_EVENTS)
        try:   # a failure from here on closes what was opened: no Chrome left running, no profile directory left behind
            self.proc = subprocess.Popen(
                [chrome, *CHROME_FLAGS, "--remote-debugging-port=0", "--user-data-dir=" + self.profile, "about:blank"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
            self.ws = WebSocket(self._page_url())
            self.call("Page.enable")
        except BaseException:
            self.close()
            raise

    def _page_url(self) -> str:
        assert self.proc is not None
        port_file = os.path.join(self.profile, "DevToolsActivePort")
        deadline = time.time() + self.startup_timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError("Chrome exited with status {}".format(self.proc.returncode))
            try:
                with open(port_file) as handle:
                    port = int(handle.readline())
                with urllib.request.urlopen("http://127.0.0.1:{}/json/list".format(port), timeout=2) as response:
                    pages = [t for t in json.load(response) if t.get("type") == "page"]
                if pages:
                    return pages[0]["webSocketDebuggerUrl"]
            except (OSError, ValueError):
                pass
            time.sleep(0.2)
        raise RuntimeError("Chrome did not open a DevTools page within {:g} s".format(self.startup_timeout))

    def call(self, method: str, **params: Any) -> Dict[str, Any]:
        assert self.ws is not None
        self.next_id += 1
        self.ws.send(json.dumps({"id": self.next_id, "method": method, "params": params}))
        while True:
            message = json.loads(self.ws.recv())
            if message.get("id") == self.next_id:
                if "error" in message:
                    raise RuntimeError("{} failed: {}".format(method, message["error"]))
                return message.get("result", {})
            if "method" in message:
                self.events_seen.append(message)

    def events(self, method: str, clear: bool = True) -> List[Dict[str, Any]]:
        """The params of the kept events of this method, oldest first; clear drops them from the record."""
        found = [m.get("params", {}) for m in self.events_seen if m.get("method") == method]
        if clear:
            kept = [m for m in self.events_seen if m.get("method") != method]
            self.events_seen.clear()
            self.events_seen.extend(kept)
        return found

    def evaluate(self, expression: str) -> Any:
        result = self.call("Runtime.evaluate", expression=expression, returnByValue=True, awaitPromise=True)
        if "exceptionDetails" in result:
            details = result["exceptionDetails"]
            raise RuntimeError("page script failed: {}".format((details.get("exception") or {}).get("description")
                                                                  or details.get("text")))
        return result["result"].get("value")

    def run(self, function: str, argument: Any) -> Any:
        return self.evaluate("({})({})".format(function, json.dumps(argument)))

    def open(self, url: str, width: int, height: int, scheme: str) -> None:
        self.call("Emulation.setDeviceMetricsOverride", width=width, height=height, deviceScaleFactor=1, mobile=False)
        self.call("Emulation.setEmulatedMedia", features=[{"name": "prefers-color-scheme", "value": scheme}])
        self.call("Page.navigate", url=url)
        deadline = time.time() + 15
        marker = url.split("?", 1)[1]
        while time.time() < deadline:   # the query is unique per case, so a stale page cannot pass for the new one
            if self.evaluate("location.search.slice(1) + '|' + document.readyState") == marker + "|complete":
                return
            time.sleep(0.05)
        raise RuntimeError("the page did not finish loading: " + url)

    def key(self, key: str, code: str, keycode: int, text: Optional[str] = None, modifiers: int = 0) -> None:
        """One key press: keyDown then keyUp. text is what the key types (Enter types "\\r"); modifiers: 1 Alt, 2 Ctrl, 4 Meta, 8 Shift."""
        params = dict(key=key, code=code, windowsVirtualKeyCode=keycode, modifiers=modifiers)
        self.call("Input.dispatchKeyEvent", type="keyDown", **(dict(params, text=text) if text is not None else params))
        self.call("Input.dispatchKeyEvent", type="keyUp", **params)

    # ---- what the browser check does with a page: wait, click, type, look ---------------------------------------------------

    def wait_for(self, expression: str, timeout: float = 15.0, what: Optional[str] = None, interval: float = 0.04) -> Any:
        """The first truthy value of the expression, polled in the page; TimeoutError after `timeout` seconds."""
        deadline = time.time() + timeout
        while True:
            value = self.evaluate(expression)
            if value:
                return value
            if time.time() >= deadline:
                raise TimeoutError("waited {:g} s for {}".format(timeout, what or expression))
            time.sleep(interval)

    def point(self, selector: str) -> List[float]:
        """The centre of the element, scrolled into view. RuntimeError when there is none or something else covers it."""
        found = self.evaluate(
            "(() => { const el = document.querySelector(%s); if (!el) return null;"
            " el.scrollIntoView({ block: 'nearest', inline: 'nearest' }); const r = el.getBoundingClientRect();"
            " const x = r.left + r.width / 2, y = r.top + r.height / 2, top = document.elementFromPoint(x, y);"
            " return { x, y, hit: !!top && (top === el || el.contains(top)) }; })()" % json.dumps(selector))
        if found is None:
            raise RuntimeError("no element matches " + selector)
        if not found["hit"]:
            raise RuntimeError("{} is covered by another element at ({:.0f}, {:.0f})".format(selector, found["x"], found["y"]))
        return [found["x"], found["y"]]

    def click(self, selector: str) -> None:
        """A real left click on the centre of the element: the events a user's mouse makes, trusted."""
        x, y = self.point(selector)
        self.call("Input.dispatchMouseEvent", type="mouseMoved", x=x, y=y)
        self.call("Input.dispatchMouseEvent", type="mousePressed", x=x, y=y, button="left", clickCount=1)
        self.call("Input.dispatchMouseEvent", type="mouseReleased", x=x, y=y, button="left", clickCount=1)

    def insert_text(self, text: str) -> None:
        self.call("Input.insertText", text=text)

    def type_text(self, text: str) -> None:
        """One key press per character, for letters, digits and the space; anything else is inserted as text."""
        for ch in text:
            if ch.isalnum() and ch.isascii():
                code = ("Digit" if ch.isdigit() else "Key") + ch.upper()
                self.key(ch, code, ord(ch.upper()), text=ch)
            elif ch == " ":
                self.key(" ", "Space", 32, text=" ")
            else:
                self.insert_text(ch)

    def set_files(self, selector: str, paths: List[str]) -> None:
        """Choose files for an <input type=file>, as the file dialog would: what a script cannot do in a headless Chrome."""
        root = self.call("DOM.getDocument", depth=0)["root"]["nodeId"]
        node = self.call("DOM.querySelector", nodeId=root, selector=selector)["nodeId"]
        self.call("DOM.setFileInputFiles", files=list(paths), nodeId=node)

    def backend_ids(self, selector: str) -> List[int]:
        """The accessibility tree's backendDOMNodeId of every element that matches."""
        root = self.call("DOM.getDocument", depth=0)["root"]["nodeId"]
        ids = self.call("DOM.querySelectorAll", nodeId=root, selector=selector)["nodeIds"]
        return [self.call("DOM.describeNode", nodeId=i)["node"]["backendNodeId"] for i in ids]

    def screenshot(self, path: str) -> int:
        """Page.captureScreenshot as a PNG file; -> its size in bytes."""
        raw = base64.b64decode(self.call("Page.captureScreenshot", format="png")["data"])
        with open(path, "wb") as handle:
            handle.write(raw)
        return len(raw)

    def set_viewport(self, width: int, height: int, scheme: Optional[str] = None) -> None:
        self.call("Emulation.setDeviceMetricsOverride", width=width, height=height, deviceScaleFactor=1, mobile=False)
        if scheme:
            self.call("Emulation.setEmulatedMedia", features=[{"name": "prefers-color-scheme", "value": scheme}])

    def close(self) -> None:
        """Close Chrome as a user would, so that it removes what it made outside its profile: the DevTools Browser.close first (Chrome has
        CLOSE_WAIT_S in all, by the clock, to answer it and to go), then SIGTERM to its group (TERM_WAIT_S), and SIGKILL only for what is
        still there, and for what Chrome left in its session when it went (stop_group). Then the profile directory is removed. Safe on a
        half-built Browser and when called twice. An interrupt that is not an Exception (the SIGTERM handler's SystemExit, Ctrl-C) cuts
        the DevTools step short and goes on as it was raised, but the stopping and the removal still happen (P4-E follow-up)."""
        proc = self.proc
        try:
            if self.ws is not None:
                try:
                    if proc is not None and not exited(proc):
                        deadline = time.monotonic() + CLOSE_WAIT_S
                        self.ws.deadline = deadline   # however much DevTools sends meanwhile, the wait for its answer ends here
                        try:
                            self.call("Browser.close")
                        except Exception:   # it may drop the connection, or never answer (TimeoutError); either way the signals follow
                            pass
                        wait_exit(proc, deadline - time.monotonic())
                finally:
                    self.ws.close()
                    self.ws = None
        finally:
            try:
                stop_group(proc)
            finally:
                shutil.rmtree(self.profile, ignore_errors=True)


def free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


class App:
    """The tiny app (random weights, a toy vocabulary) on a free loopback port, in one uvicorn process of its own. home is its
    CHAT_HOME: a temporary directory removed by stop() unless one is given (then the caller owns it). static_dir serves another
    copy of app/static, for a check of a changed page. step_delay_s paces the tiny engine so that a turn can be watched.
    tiny_gallery runs every stage: the synthetic gallery and the keyword labeller (create_app's laptop convenience, P5-E)."""

    def __init__(self, step_delay_s: float = 0.0, home: Optional[str] = None, static_dir: Optional[str] = None,
                 startup_timeout: float = 90.0, tiny_gallery: bool = False) -> None:
        self.owns_home = home is None
        self.static_dir = static_dir
        self.home = home or tempfile.mkdtemp(prefix="chat_ui_home_")
        self.proc = None  # type: Optional[subprocess.Popen]
        self.log_path = os.path.join(self.home, "server.log")
        self.url = ""
        try:
            port = free_port()
            argv = [sys.executable, "-c", LAUNCH, self.home, repr(float(step_delay_s)), str(port)]
            if static_dir:
                argv.append(static_dir)
            if tiny_gallery:
                argv.append("--tiny-gallery")
            with open(self.log_path, "w") as log:
                self.proc = subprocess.Popen(argv, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            self.url = "http://127.0.0.1:{}/".format(port)
            self._wait_healthy(startup_timeout)
        except BaseException:
            self.stop()
            raise

    def _wait_healthy(self, timeout: float) -> None:
        assert self.proc is not None
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                break
            try:
                with urllib.request.urlopen(self.url + "healthz", timeout=2):
                    return
            except OSError:
                time.sleep(0.3)
        raise RuntimeError("the app did not start: " + self.log_tail())

    def log_tail(self, n: int = 1500) -> str:
        try:
            with open(self.log_path) as handle:
                return handle.read()[-n:]
        except OSError:
            return ""

    def hold(self) -> None:
        """From the next snapshot of every turn on, the turns wait (hold_turns) until release(). A turn that is waiting is running."""
        with open(os.path.join(self.home, HOLD_FILE), "w"):
            pass

    def release(self) -> None:
        """The held turns go on. Nothing to release is no error."""
        try:
            os.remove(os.path.join(self.home, HOLD_FILE))
        except FileNotFoundError:
            pass

    def stop(self) -> None:
        stop_group(self.proc)   # uvicorn leaves on SIGTERM; its store is closed by the app's own shutdown
        if self.owns_home:
            shutil.rmtree(self.home, ignore_errors=True)
