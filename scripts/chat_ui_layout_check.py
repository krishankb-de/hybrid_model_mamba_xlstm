"""CHAT_UI_PLAN.md P4-A fix round 1: a geometry check for the chat page. A local development tool; validate.sh does not
run it.

It starts the app (tiny engine, temporary home, free loopback port), or takes --url, and drives headless Chrome over the
DevTools protocol with exact viewports (Emulation.setDeviceMetricsOverride) and an emulated colour scheme. Each case is
one viewport in one scheme with the drawer closed or open (its `hidden` attribute removed in the page); six chips and a
140-character unbroken word are injected, and Stop is shown and hidden. Checks:

  a  no horizontal overflow on the document, #conversation or #composer (scrollWidth <= clientWidth)
  b  the centre of #settings, #send and (shown) #stop hits the button itself, not an overlay
  c  #composer lies fully inside the viewport
  d  #conversation is at least 120 px tall on a viewport at least 568 px tall
  e  Settings, Send and Stop follow each other in DOM order, row by row and left to right
  f  the focus ring of #sidebar-toggle (shown at 800 px and below) lies inside the viewport

On the short viewports, where the page scrolls instead (667x375, 320x256), c and d give way to: the report keeps its
natural height (no scroller of its own, at least 120 px), the composer is not capped, the banner stays at the top of
the viewport while the page scrolls, and each button is hit-tested after it is scrolled into view.

    venv/bin/python scripts/chat_ui_layout_check.py [--url http://127.0.0.1:8000/]

Prints one `FAIL ...` line per failure, then `RESULT {"cases": N, "failures": [...]}`. Exits 1 on any failure and 2 when
Chrome or the app cannot be started. Chrome is $CHROME, else /Applications/Google Chrome.app/..., else one on PATH.
Standard library only: the websocket client below is the little of RFC 6455 a DevTools session needs.
"""
import argparse
import base64
import hashlib
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time
import urllib.request
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAC_CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
WIDTHS = [320, 375, 800, 801, 820, 834, 900, 925, 1024, 1280]
HEIGHTS = [568, 900]
SHORT_VIEWPORTS = [(667, 375), (320, 256)]   # below 480 px tall the page scrolls instead of pinning the composer
SCHEMES = ["light", "dark"]
MIN_REPORT_PX = 120
MIN_TALL_PX = 568
CHIPS = ["beam 3", "100 tok", "cached", "k 4/3", "label on", "repair off"]
LONG_WORD = "x" * 140
REPORT_TEXT = ("The lungs are clear. There is no focal consolidation, pleural effusion or pneumothorax. The "
               "cardiomediastinal silhouette is within normal limits. No acute osseous abnormality is seen. ") * 3
WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

INJECT_JS = """(args) => {
  const q = (s) => document.querySelector(s);
  q('#chips').replaceChildren(...args.chips.map((t) => { const s = document.createElement('span'); s.textContent = t; return s; }));
  const card = document.createElement('article');
  card.className = 'card';
  for (const t of [args.word, args.text]) { const p = document.createElement('p'); p.textContent = t; card.append(p); }
  q('#conversation').replaceChildren(card);
  return true;
}"""

MEASURE_JS = """(args) => {
  const q = (s) => document.querySelector(s);
  const doc = document.documentElement;
  const shown = (el) => el.getClientRects().length > 0;
  const box = (el) => { const r = el.getBoundingClientRect(); return [r.left, r.top, r.right, r.bottom]; };
  const name = (e) => !e ? null : (e.id ? '#' + e.id : e.tagName.toLowerCase());
  window.scrollTo(0, 0);
  const out = { doc: [doc.scrollWidth, doc.clientWidth], boxes: {}, hits: {} };
  for (const s of ['#conversation', '#composer']) {
    const el = q(s);
    out[s] = { sw: el.scrollWidth, cw: el.clientWidth, sh: el.scrollHeight, ch: el.clientHeight };
  }
  for (const s of ['#composer', '#settings', '#send', '#stop']) out.boxes[s] = shown(q(s)) ? box(q(s)) : null;
  for (const s of ['#settings', '#send', '#stop']) {
    const el = q(s);
    if (!shown(el)) { out.hits[s] = null; continue; }
    if (args.scroll) el.scrollIntoView({ block: 'center', inline: 'center' });
    const b = box(el), x = (b[0] + b[2]) / 2, y = (b[1] + b[3]) / 2, top = document.elementFromPoint(x, y);
    out.hits[s] = { ok: !!top && (top === el || el.contains(top)), at: [Math.round(x), Math.round(y)], got: name(top) };
  }
  const toggle = q('#sidebar-toggle');
  if (shown(toggle)) {
    window.scrollTo(0, 0);
    toggle.focus();
    const cs = getComputedStyle(toggle), b = box(toggle);
    out.toggle = { box: b, ring: cs.outlineStyle === 'none' ? null : parseFloat(cs.outlineWidth) + parseFloat(cs.outlineOffset) };
    toggle.blur();
  }
  if (args.scroll) {   // the page scrolls here: does the banner stay at the top of the viewport?
    const room = doc.scrollHeight - innerHeight;
    window.scrollTo(0, Math.min(150, Math.max(room, 0)));
    out.banner = { scrolled: window.scrollY, top: q('.banner').getBoundingClientRect().top };
    window.scrollTo(0, 0);
  }
  return out;
}"""


class WebSocket:
    """A text-frame websocket client: the handshake, masked sends, and receives with fragments, pings and close."""

    def __init__(self, url: str, timeout: float = 30.0) -> None:
        parts = urlparse(url)
        self.sock = socket.create_connection((parts.hostname, parts.port), timeout=timeout)
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

    def _read(self, n: int) -> bytes:
        while len(self.buffer) < n:
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


class Browser:
    """A private headless Chrome and one DevTools session on its page."""

    def __init__(self, chrome: str) -> None:
        self.profile = tempfile.mkdtemp(prefix="chat_ui_layout_chrome_")
        self.proc = subprocess.Popen(
            [chrome, "--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
             "--remote-debugging-port=0", "--remote-allow-origins=*", "--user-data-dir=" + self.profile, "about:blank"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.ws: Optional[WebSocket] = None
        self.next_id = 0
        self.ws = WebSocket(self._page_url())
        self.call("Page.enable")

    def _page_url(self) -> str:
        port_file = os.path.join(self.profile, "DevToolsActivePort")
        deadline = time.time() + 30
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
        raise RuntimeError("Chrome did not open a DevTools page within 30 s")

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

    def key(self, key: str, code: str, keycode: int) -> None:
        for kind in ("keyDown", "keyUp"):
            self.call("Input.dispatchKeyEvent", type=kind, key=key, code=code, windowsVirtualKeyCode=keycode)

    def close(self) -> None:
        if self.ws is not None:
            self.ws.close()
        self.proc.kill()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        shutil.rmtree(self.profile, ignore_errors=True)


def find_chrome() -> Optional[str]:
    for candidate in (os.environ.get("CHROME"), MAC_CHROME, shutil.which("google-chrome"), shutil.which("chromium"),
                      shutil.which("chromium-browser")):
        if candidate and os.path.exists(candidate):
            return candidate
    return None


def start_app() -> Tuple[subprocess.Popen, str, List[str]]:
    """The app on a free loopback port, tiny engine, a temporary CHAT_HOME. -> (process, page url, paths to remove)."""
    home = tempfile.mkdtemp(prefix="chat_ui_layout_home_")
    log_path = os.path.join(home, "server.log")
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    log = open(log_path, "w")
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "--factory", "app.server:create_app", "--host",
                             "127.0.0.1", "--port", str(port), "--log-level", "warning"],
                            cwd=REPO_ROOT, env=dict(os.environ, CHAT_HOME=home), stdout=log, stderr=subprocess.STDOUT)
    url = "http://127.0.0.1:{}/".format(port)
    deadline = time.time() + 90
    while time.time() < deadline:
        if proc.poll() is not None:
            break
        try:
            with urllib.request.urlopen(url + "healthz", timeout=2):
                return proc, url, [home]
        except OSError:
            time.sleep(0.3)
    proc.kill()
    log.close()
    with open(log_path) as handle:
        tail = handle.read()[-1500:]
    shutil.rmtree(home, ignore_errors=True)
    raise RuntimeError("the app did not start: " + tail)


def check_state(m: Dict[str, Any], width: int, height: int, scroll: bool) -> List[str]:
    """The failures in one measurement of one state: what is wrong, in words."""
    failures = []
    for name, (sw, cw) in (("documentElement", m["doc"]), ("#conversation", (m["#conversation"]["sw"],
                                                                          m["#conversation"]["cw"])),
                           ("#composer", (m["#composer"]["sw"], m["#composer"]["cw"]))):
        if sw > cw:
            failures.append("{} overflows horizontally: scrollWidth {} > clientWidth {}".format(name, sw, cw))
    for selector in ("#settings", "#send", "#stop"):
        hit = m["hits"][selector]
        if hit is None and selector != "#stop":
            failures.append("{} is not rendered".format(selector))
        elif hit is not None and not hit["ok"]:
            failures.append("{} is covered: its centre {} hits {}".format(
                selector, tuple(hit["at"]), hit["got"] or "nothing (outside the viewport)"))
    present = [s for s in ("#settings", "#send", "#stop") if m["boxes"][s] is not None]
    for first, second in zip(present, present[1:]):
        a, b = m["boxes"][first], m["boxes"][second]
        if not (b[1] >= a[3] - 1 or b[0] >= a[2] - 1):   # a row below, or the same row and further right
            failures.append("{} is not after {} in visual order".format(second, first))
    toggle = m.get("toggle")
    if toggle is not None:
        ring = toggle["ring"]
        left, top = toggle["box"][0], toggle["box"][1]
        if ring is None:
            failures.append("#sidebar-toggle shows no focus ring")
        elif left - ring < -0.5 or top - ring < -0.5:
            failures.append("#sidebar-toggle focus ring leaves the viewport (box at {:.0f},{:.0f}, ring {:g}px)".format(
                left, top, ring))
    composer = m["boxes"]["#composer"]
    conversation = m["#conversation"]
    if scroll:   # short viewport: the page scrolls, the report keeps its natural height
        if conversation["sh"] > conversation["ch"] + 1:
            failures.append("the report is clipped: #conversation scrollHeight {} > clientHeight {}".format(
                conversation["sh"], conversation["ch"]))
        if conversation["ch"] < MIN_REPORT_PX:
            failures.append("the report area is {} px tall, under {}".format(conversation["ch"], MIN_REPORT_PX))
        if m["#composer"]["sh"] > m["#composer"]["ch"] + 1:
            failures.append("#composer is capped: scrollHeight {} > clientHeight {}".format(
                m["#composer"]["sh"], m["#composer"]["ch"]))
        banner = m["banner"]
        if abs(banner["top"]) > 0.5:
            failures.append("the banner scrolls away: its top is {:.0f} px after scrolling {} px".format(
                banner["top"], banner["scrolled"]))
        return failures
    if composer is None or composer[0] < -0.5 or composer[1] < -0.5 or composer[2] > width + 0.5 \
            or composer[3] > height + 0.5:
        failures.append("#composer is not inside the {}x{} viewport: {}".format(
            width, height, None if composer is None else [round(v) for v in composer]))
    if height >= MIN_TALL_PX and conversation["ch"] < MIN_REPORT_PX:
        failures.append("the report area is {} px tall, under {}".format(conversation["ch"], MIN_REPORT_PX))
    return failures


def run_checks(browser: Browser, url: str) -> Tuple[int, List[str]]:
    """Every viewport in every scheme, drawer closed and open, Stop hidden and shown. A case is a viewport in a scheme
    with the drawer closed or open. -> (cases, failures); a failure seen in both schemes is reported once."""
    cases = 0
    seen = {}   # type: Dict[Tuple[str, str, str, str], List[str]]   # (viewport, drawer, stop, failure) -> schemes
    viewports = [(w, h, False) for h in HEIGHTS for w in WIDTHS] + [(w, h, True) for w, h in SHORT_VIEWPORTS]
    serial = 0
    for width, height, scroll in viewports:
        for scheme in SCHEMES:
            serial += 1
            browser.open("{}?case={}".format(url, serial), width, height, scheme)
            browser.key("Tab", "Tab", 9)   # keyboard modality, so :focus-visible matches when the toggle is focused
            browser.run(INJECT_JS, {"chips": CHIPS, "word": LONG_WORD, "text": REPORT_TEXT})
            for drawer in ("closed", "open"):
                cases += 1
                browser.evaluate("document.querySelector('#drawer').hidden = {}".format(
                    "false" if drawer == "open" else "true"))
                for stop in ("hidden", "shown"):
                    browser.evaluate("document.querySelector('#stop').hidden = {}".format(
                        "false" if stop == "shown" else "true"))
                    measured = browser.run(MEASURE_JS, {"scroll": scroll})
                    for failure in check_state(measured, width, height, scroll):
                        seen.setdefault(("{}x{}".format(width, height), drawer, stop, failure), []).append(scheme)
    failures = ["{} {} drawer={} stop={}: {}".format(viewport, "+".join(schemes), drawer, stop, failure)
                for (viewport, drawer, stop, failure), schemes in seen.items()]
    return cases, failures


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Geometry check for the chat page, in headless Chrome.")
    parser.add_argument("--url", help="check this running app instead of starting one (its page URL, e.g. "
                                      "http://127.0.0.1:8000/)")
    args = parser.parse_args(argv)
    chrome = find_chrome()
    if chrome is None:
        print("ERROR no Chrome found: set $CHROME to its executable")
        return 2
    app, cleanup = None, []   # type: Tuple[Optional[subprocess.Popen], List[str]]
    browser = None
    try:
        if args.url:
            url = args.url if args.url.endswith("/") else args.url + "/"
        else:
            app, url, cleanup = start_app()
        browser = Browser(chrome)
        cases, failures = run_checks(browser, url)
    except (OSError, RuntimeError) as exc:   # ConnectionError is an OSError
        print("ERROR {}".format(exc))
        return 2
    finally:
        if browser is not None:
            browser.close()
        if app is not None:
            app.terminate()
            try:
                app.wait(timeout=15)
            except subprocess.TimeoutExpired:
                app.kill()
        for path in cleanup:
            shutil.rmtree(path, ignore_errors=True)
    for failure in failures:
        print("FAIL " + failure)
    print("RESULT " + json.dumps({"cases": cases, "failures": failures}))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
