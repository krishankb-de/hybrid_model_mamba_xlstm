"""What the two Chrome-driven checks of the chat page share (CHAT_UI_PLAN.md P4-A, P4-E): a DevTools websocket client, a
private headless Chrome with one page, and the tiny app on a free loopback port. Local development tools: nothing here is
imported by the app or run by validate.sh.

    scripts/chat_ui_layout_check.py    geometry at many viewports
    scripts/chat_ui_browser_check.py   the browser checklist, with screenshots

Every process this module starts runs in a session of its own, so stopping it reaches the whole group (Chrome's helpers
included) and nothing outlives a failure path. It is stopped politely first, and killed last: Chrome is asked to close over
DevTools (it has CLOSE_WAIT_S in all, by the clock, to answer and go), then sent SIGTERM, and only a process that is still there
after the wait gets SIGKILL. A Chrome that is killed leaves its singleton directory (com.google.Chrome.*, a SingletonSocket and
a SingletonCookie) in $TMPDIR, one more on every run; one that leaves on request removes it. Browser.close() and App.stop() are
safe to call at any time, on an object whose constructor failed half-way too. Standard library only: the websocket client is the
little of RFC 6455 a DevTools session needs.
"""
import base64
import hashlib
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

# The app as the checks run it. The uvicorn CLI cannot pass create_app its arguments (the step delay of the tiny engine is
# one), so this is a three-line launcher. argv: home, step delay in seconds, port, and optionally a directory of static files.
LAUNCH = """\
import sys
from pathlib import Path
import uvicorn
import app.server as server
if len(sys.argv) > 4:
    server.STATIC_DIR = Path(sys.argv[4])
uvicorn.run(server.create_app(engine="tiny", home=sys.argv[1], tiny_step_delay_s=float(sys.argv[2])),
            host="127.0.0.1", port=int(sys.argv[3]), log_level="warning")
"""


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
        """Before a wait for the peer's data: with a deadline, cut the wait off at it. A socket timeout alone is per read, so a peer that
        keeps sending (events, pings) and never answers would never time out."""
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


def stop_group(proc: Optional["subprocess.Popen"], term_wait: Optional[float] = None) -> None:
    """Stop a process started with start_new_session=True, and its session, politely first: SIGTERM to the group, a wait of up to
    term_wait seconds (TERM_WAIT_S) for the process to go and clean up after itself, then SIGKILL for what is still there. Safe on
    None and on a process that is gone (nothing is signalled once the process was reaped)."""
    if proc is None:
        return
    if proc.poll() is None:   # poll() reaps a process that has already exited
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except OSError:
            pass
        try:
            proc.wait(timeout=TERM_WAIT_S if term_wait is None else term_wait)
        except subprocess.TimeoutExpired:
            pass
    kill_group(proc)


def exit_on_sigterm() -> None:
    """Make SIGTERM raise SystemExit, so that a check's `finally` runs and stops Chrome and the app: left at its default, a SIGTERM
    ends the process at once and orphans both (each runs in a session of its own, which the terminal's signals do not reach)."""
    def handler(signum: int, frame: Any) -> None:
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, handler)


def find_chrome() -> Optional[str]:
    for candidate in (os.environ.get("CHROME"), MAC_CHROME, shutil.which("google-chrome"), shutil.which("chromium"),
                      shutil.which("chromium-browser")):
        if candidate and os.path.exists(candidate):
            return candidate
    return None


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
                [chrome, "--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
                 "--remote-debugging-port=0", "--user-data-dir=" + self.profile, "about:blank"],
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
        still there. Then the profile directory is removed. Safe on a half-built Browser and when called twice."""
        proc = self.proc
        if self.ws is not None:
            if proc is not None and proc.poll() is None:
                deadline = time.monotonic() + CLOSE_WAIT_S
                self.ws.deadline = deadline   # however much DevTools sends meanwhile, the wait for its answer ends here
                try:
                    self.call("Browser.close")
                except Exception:   # it may drop the connection, or never answer (TimeoutError); either way the signals follow
                    pass
                try:
                    proc.wait(timeout=max(0.0, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    pass
            self.ws.close()
            self.ws = None
        stop_group(proc)
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
    copy of app/static, for a check of a changed page. step_delay_s paces the tiny engine so that a turn can be watched."""

    def __init__(self, step_delay_s: float = 0.0, home: Optional[str] = None, static_dir: Optional[str] = None,
                 startup_timeout: float = 90.0) -> None:
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

    def stop(self) -> None:
        stop_group(self.proc)   # uvicorn leaves on SIGTERM; its store is closed by the app's own shutdown
        if self.owns_home:
            shutil.rmtree(self.home, ignore_errors=True)
