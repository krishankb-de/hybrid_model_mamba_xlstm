"""CHAT_UI_PLAN.md P4-H: the harness of the end-to-end tests of the chat page (tests/e2e/test_ui_playwright.py).

Each test gets a tiny server of its own and pages of its own in one headless Google Chrome:
  * The server is app.server.create_app(engine="tiny") (random weights, a toy vocabulary: no checkpoint and no data) on a free
    loopback port with a temporary CHAT_HOME; a test that parametrizes `ui` with {"tiny_gallery": True} gets one with the synthetic
    gallery and the keyword labeller too (P5-E), so every stage runs, and one that adds "public_token": "t" gets a public-mode server
    with that token (P6-C), whose pages load PUBLIC_SETUP_JS first (TinyServer.setup_script). It runs from this checkout with PYTHONPATH set to it, because the venv's editable
    install maps `scripts` and `hybrid_xmamba` to wherever it was installed from, and in a process group of its own, so that
    stopping it reaches everything it started. A test can kill it (SIGKILL: a crash) and start it again on the same port and home.
  * The browser is Playwright's chromium with channel="chrome": the Google Chrome installed on the machine, nothing downloaded. One
    runs for the whole session; each test has its own contexts (storage, downloads) and pages.
  * Every page is watched: console errors, uncaught page errors, HTTP 4xx and 5xx responses, and failed requests. A test that
    provokes a refusal says so first (UI.expect_refusal), and so does one that takes the server down (UI.server_down); anything else
    is a failure, said in plain words by UI.assert_clean. Every test ends with it, and teardown asks again, after a short settle, for
    every test that passed: a problem that arrives after the test's last assertion fails it too.
The images are synthetic (noise, a renamed text file, a GIF, a 32 px PNG, a padded 21 MB file). CHAT_UI_E2E_EVIDENCE=<dir> also saves
screenshots of the key states there (UI.shot), each under 400 KB. Without the playwright package or without Google Chrome every test
here is skipped; `pip install -r requirements-e2e.txt` provides the first.

    venv/bin/python -m pytest tests/e2e -m e2e -q
"""
import base64
import io
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode, urlparse

import numpy as np
import pytest
from PIL import Image

from scripts.chat_ui_cdp import free_port, kill_group, stop_group

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EVIDENCE_ENV = "CHAT_UI_E2E_EVIDENCE"
MAX_SHOT_BYTES = 400 * 1024
STEP_DELAY_S = 0.05      # between two decoding steps of the tiny engine, as the dev server runs it: a stream that can be watched
FAULT_TOKENS = 17        # a turn with this token budget fails inside the engine (LAUNCH): the page's internal-error card
DESKTOP = (1280, 900)
PUBLIC_CLIENT = "e2e-public"   # the X-Client-Id a public-mode test's pages and its API calls share (PUBLIC_SETUP_JS)
# Run before a page's own scripts on every load of a public-mode test's context: the access token as Settings would hold it, and the
# client id the API calls below send too, so that both see the same sessions.
PUBLIC_SETUP_JS = ("try { localStorage.setItem('cxrchat.settings', JSON.stringify({ token: %s }));"
                   " localStorage.setItem('cxrchat.client', %s); } catch (e) { /* no storage: the page has no token */ }")

# The app as these tests run it: create_app's arguments (the step delay among them) cannot pass through the uvicorn CLI. One fault is
# planted, and only here: a turn whose budget is FAULT_TOKENS raises ImportError inside the engine, the class the stale dev server of
# 2026-10-09 raised at every turn, so that the card for an internal error is seen end to end. The server must send the class name and
# never the exception's text, which goes to its log. argv: home, step delay in seconds, port, then the flags: "tiny_gallery" for the P5-E
# stages, and "public:<token>" for a public-mode server with that token (P6-C: what public mode shows, and never asks for).
LAUNCH = """\
import sys
import uvicorn
import app.engine as engine
import app.server as server

FAULT_TOKENS = %d
_generate = engine.Engine.generate


def generate(self, enc, opts, on_snapshot, cancel):
    if opts.max_new_tokens == FAULT_TOKENS:
        raise ImportError("planted by tests/e2e: the decoding code changed under a running server")
    return _generate(self, enc, opts, on_snapshot, cancel)


engine.Engine.generate = generate
flags = sys.argv[4:]
public = [flag.split(":", 1)[1] for flag in flags if flag.startswith("public:")]
uvicorn.run(server.create_app(engine="tiny", home=sys.argv[1], tiny_step_delay_s=float(sys.argv[2]),
                              tiny_gallery="tiny_gallery" in flags, mode="public" if public else "private",
                              token=public[0] if public else None),
            host="127.0.0.1", port=int(sys.argv[3]), log_level="warning")
""" % FAULT_TOKENS

# The newest card as the user sees it. `shown` is the report put back together from the card's sections ("Findings: ..."), the way the
# browser check reads it; `raw` is the Show raw text when that is on.
CARD_JS = """() => {
  const cards = [...document.querySelectorAll('#conversation article.card')];
  const c = cards[cards.length - 1];
  if (!c) return null;
  const body = c.querySelector('.report-body'), raw = c.querySelector('pre.report-raw'), prov = c.querySelector('.provenance p');
  return {
    n: cards.length, id: c.getAttribute('data-message-id'), status: c.getAttribute('data-status'),
    stages: Object.fromEntries([...c.querySelectorAll('.timeline > li')].map((l) => [l.getAttribute('data-stage'), l.getAttribute('data-state')])),
    stage_text: Object.fromEntries([...c.querySelectorAll('.timeline > li')].map((l) => [l.getAttribute('data-stage'), l.textContent])),
    shown: body ? [...body.querySelectorAll('.report-section')].map((s) => {
      const h = s.querySelector('h3'), p = s.querySelector('p');
      return (h ? h.textContent + ': ' : '') + (p ? p.textContent : '');
    }).join(' ') : '',
    raw: raw ? raw.textContent : null,
    notes: [...c.querySelectorAll('.notes .note')].map((n) => n.textContent),          // warnings, the error and its hint, a stop
    report_notes: [...c.querySelectorAll('.report .note')].map((n) => n.textContent),  // why the report ends where it does
    labels: [...c.querySelectorAll('.labels .note')].map((n) => n.textContent),
    provenance: prov ? prov.textContent : '',
    timeline_hidden: !!c.querySelector('ol.timeline[hidden]'),
    spinning: c.querySelectorAll('.timeline > li[data-state="running"]').length,
  };
}"""

COMPOSER_JS = """() => {
  const shown = (sel) => { const e = document.querySelector(sel); return e && !e.hidden ? e.textContent : null; };
  const notice = document.querySelector('#notice');
  return {
    send_disabled: document.querySelector('#send').disabled, stop_hidden: document.querySelector('#stop').hidden,
    stop_disabled: document.querySelector('#stop').disabled, stop_text: document.querySelector('#stop').textContent,
    rerun: shown('#rerun-hint'), preview: shown('#preview span'),
    notice: notice && !notice.hidden ? notice.querySelector('p').textContent : null,
    chips: [...document.querySelectorAll('#chips span')].map((s) => s.textContent),
    saved: document.querySelector('#saved') ? document.querySelector('#saved').textContent : '',
    prompt: document.querySelector('#prompt').value, hash: location.hash,
    focus: document.activeElement && document.activeElement !== document.body
      ? (document.activeElement.id || document.activeElement.getAttribute('data-setting') || document.activeElement.tagName.toLowerCase()) : 'body',
  };
}"""

DRAWER_JS = """() => {
  const d = document.querySelector('#drawer');
  const field = (k) => document.querySelector('[data-setting=' + k + ']');
  let stored = null;
  try { stored = JSON.parse(localStorage.getItem('cxrchat.settings')); } catch (e) { stored = 'unreadable'; }
  return {
    open: !d.hidden, tokens: field('max_new_tokens').value, beam: field('beam_size').value,
    errors: [...d.querySelectorAll('.field-error')].filter((e) => !e.hidden).map((e) => e.id + ': ' + e.textContent),
    invalid: [...d.querySelectorAll('[aria-invalid="true"]')].map((e) => e.getAttribute('data-setting')),
    stored,
  };
}"""

SESSIONS_JS = """() => [...document.querySelectorAll('#session-list li')].map((li) => ({
  id: li.getAttribute('data-session'), title: li.querySelector('.session-title').textContent,
  meta: li.querySelector('.session-meta').textContent, current: li.querySelector('a').getAttribute('aria-current') === 'page' }))"""

# Every distinct state of the newest card, recorded by the page itself as it changes, so that no frame between two looks is missed:
# its status, its stage states in contract order, and its report while the turn runs.
RECORD_JS = """() => {
  const rec = { frames: [], texts: [] };
  window.__rec = rec;
  const conversation = document.querySelector('#conversation');
  let key = null;
  new MutationObserver(() => {
    const c = [...conversation.querySelectorAll('article.card')].pop();
    if (!c) return;
    const body = c.querySelector('.report-body');
    const frame = { status: c.getAttribute('data-status'), report: body ? body.textContent : '',
                    stages: [...c.querySelectorAll('.timeline > li')].map((l) => l.getAttribute('data-state')) };
    const k = JSON.stringify(frame);
    if (k === key) return;
    key = k;
    rec.frames.push(frame);
    if (frame.status === 'running' && frame.report && !rec.texts.includes(frame.report)) rec.texts.push(frame.report);
  }).observe(conversation, { subtree: true, childList: true, attributes: true, characterData: true });
  return true;
}"""

# A file dropped on (or pasted into) an element, as a user's drag from the desktop delivers it: a DataTransfer holding a File.
DROP_JS = """async ({ b64, name, type, target, kind }) => {
  const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
  const data = new DataTransfer();
  data.items.add(new File([bytes], name, { type }));
  const el = document.querySelector(target);
  if (kind === 'paste') return !el.dispatchEvent(new ClipboardEvent('paste', { bubbles: true, cancelable: true, clipboardData: data }));
  for (const t of ['dragenter', 'dragover']) el.dispatchEvent(new DragEvent(t, { bubbles: true, cancelable: true, dataTransfer: data }));
  return !el.dispatchEvent(new DragEvent('drop', { bubbles: true, cancelable: true, dataTransfer: data }));
}"""


# ---- synthetic images --------------------------------------------------------------------------------------------------------------

def noise_png(w: int, h: int) -> bytes:
    arr = (np.random.default_rng(0).random((h, w)) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture(scope="session")
def images(tmp_path_factory) -> Dict[str, str]:
    """name -> path: three usable X-rays (noise) and the four uploads the page or the server must refuse."""
    d = tmp_path_factory.mktemp("images")
    jpeg = io.BytesIO()
    Image.open(io.BytesIO(noise_png(300, 300))).convert("RGB").save(jpeg, "JPEG", quality=90)
    gif = io.BytesIO()
    Image.new("L", (128, 128), 128).save(gif, "GIF")
    files = {"xray_a.png": noise_png(320, 320), "xray_b.png": noise_png(288, 256), "xray_c.jpg": jpeg.getvalue(),
             "notes.png": b"This is a text file that was renamed to .png.\n" * 20, "scan.gif": gif.getvalue(),
             "tiny_32.png": noise_png(32, 32), "huge.png": noise_png(64, 64) + b"\0" * (21 * 1024 * 1024)}
    paths = {}
    for name, data in files.items():
        (d / name).write_bytes(data)
        paths[name] = str(d / name)
    return paths


# ---- the server -------------------------------------------------------------------------------------------------------------------

class TinyServer:
    """The tiny app in a process group of its own on a free loopback port: started, killed and started again on the same port and
    home, and stopped (SIGTERM, then SIGKILL for what is left)."""

    def __init__(self, home: str, step_delay_s: float = STEP_DELAY_S, tiny_gallery: bool = False,
                 public_token: Optional[str] = None) -> None:
        self.home, self.step_delay_s, self.tiny_gallery = home, float(step_delay_s), bool(tiny_gallery)
        self.public_token = public_token   # a public-mode server with this token (P6-C); None: private, no token
        self.port = free_port()
        self.url = "http://127.0.0.1:{}/".format(self.port)
        self.log_path = os.path.join(home, "server.log")
        self.proc = None  # type: Optional[subprocess.Popen]
        self.start()

    def start(self, timeout: float = 90.0) -> None:
        with open(self.log_path, "a") as log:
            self.proc = subprocess.Popen(
                [sys.executable, "-c", LAUNCH, self.home, repr(self.step_delay_s), str(self.port)]
                + (["tiny_gallery"] if self.tiny_gallery else []) + (["public:" + self.public_token] if self.public_token else []),
                cwd=REPO_ROOT,
                env=dict(os.environ, PYTHONPATH=REPO_ROOT), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.time() + timeout
        while time.time() < deadline and self.proc.poll() is None:
            try:
                with urllib.request.urlopen(self.url + "healthz", timeout=2):
                    return
            except OSError:
                time.sleep(0.2)
        self.stop()   # one that did not come up in time is not left running
        raise RuntimeError("the tiny server did not start:\n" + self.log_tail())

    def kill(self) -> None:
        """A crash: SIGKILL to the whole group, with no shutdown of any kind."""
        kill_group(self.proc)

    def stop(self) -> None:
        stop_group(self.proc)

    def log_tail(self, n: int = 3000) -> str:
        try:
            with open(self.log_path) as handle:
                return handle.read()[-n:]
        except OSError:
            return ""

    def _request(self, path: str, **kwargs: Any) -> urllib.request.Request:
        """A request to the server: a public-mode one carries the token and the client id its pages use (PUBLIC_CLIENT)."""
        headers = {"Authorization": "Bearer " + self.public_token, "X-Client-Id": PUBLIC_CLIENT} if self.public_token else {}
        return urllib.request.Request(self.url + path.lstrip("/"), headers=headers, **kwargs)

    def setup_script(self) -> str:
        """PUBLIC_SETUP_JS for this server's token: add it to a context before its first page loads."""
        return PUBLIC_SETUP_JS % (json.dumps(self.public_token or ""), json.dumps(PUBLIC_CLIENT))

    def get(self, path: str) -> Any:
        """A JSON route, read as an API client reads it: what the server stored."""
        with urllib.request.urlopen(self._request(path), timeout=15) as response:
            return json.load(response)

    def status(self, path: str) -> int:
        try:
            with urllib.request.urlopen(self._request(path), timeout=15) as response:
                return response.status
        except urllib.error.HTTPError as exc:
            return exc.code

    def post_turn(self, session_id: str, options: Dict[str, Any], text: str = "") -> str:
        """A turn with no image, sent as an API client sends it (a command, or a test study by options.test_row), read to its end:
        -> its message id. The page does not see it until it loads the chat again."""
        body = urlencode({"text": text, "options": json.dumps(options)}).encode()
        request = urllib.request.Request(self.url + "v1/sessions/{}/messages".format(session_id), data=body, method="POST")
        with urllib.request.urlopen(request, timeout=120) as response:
            response.read()
            return response.headers["X-Message-Id"]

    def sessions(self) -> List[Dict[str, Any]]:
        return self.get("v1/sessions")["sessions"]

    def session(self, session_id: str) -> Dict[str, Any]:
        return self.get("v1/sessions/" + session_id)

    def message(self, message_id: str) -> Dict[str, Any]:
        return self.get("v1/messages/" + message_id)


def stage_detail(message: Dict[str, Any], stage: str) -> Dict[str, Any]:
    """The detail of a stage's stage_end in a stored message's events ({} when it has none)."""
    for row in message["events"]:
        if row["event"] == "stage_end" and row["data"].get("stage") == stage:
            return row["data"].get("detail") or {}
    return {}


def squeezed(text: Optional[str]) -> str:
    return " ".join((text or "").split())


# ---- the page ---------------------------------------------------------------------------------------------------------------------

class UI:
    """One test's view of the app: its server, its pages, what they reported, and what a user does with them."""

    def __init__(self, browser: Any, server: TinyServer, evidence: Optional[str]) -> None:
        self.browser, self.server, self.evidence = browser, server, evidence
        self.problems = []  # type: List[str]
        self.provoked = []  # type: List[Tuple[int, str, str]]
        self.refusals = []  # type: List[str]
        self.requests = []  # type: List[Tuple[str, str]]   # (method, path) of every request any page made, in order
        self.down = False
        self.contexts = []  # type: List[Any]
        self.page = self.new_page()

    # -- contexts, pages and the watch
    def new_context(self, viewport: Tuple[int, int] = DESKTOP, scheme: str = "light", **options: Any) -> Any:
        """A browser context of its own (storage, downloads); options go to Playwright (has_touch=True, is_mobile=True for a phone)."""
        context = self.browser.new_context(viewport={"width": viewport[0], "height": viewport[1]}, color_scheme=scheme,
                                           accept_downloads=True, **options)
        self.contexts.append(context)
        return context

    def new_page(self, context: Any = None, viewport: Tuple[int, int] = DESKTOP, scheme: str = "light", **options: Any) -> Any:
        page = (context or self.new_context(viewport, scheme, **options)).new_page()
        page.on("console", self._console)
        page.on("pageerror", lambda error: self.problems.append("uncaught page error: {}".format(error)))
        page.on("request", lambda request: self.requests.append((request.method, urlparse(request.url).path)))
        page.on("response", self._response)
        page.on("requestfailed", self._failed)
        return page

    def _console(self, message: Any) -> None:
        if message.type != "error":
            return
        if message.text.startswith("Failed to load resource:"):   # the browser's own line for a request: _response and _failed judge it
            return
        self.problems.append("console.error: {}".format(message.text[:300]))

    def _response(self, response: Any) -> None:
        if response.status < 400:
            return
        method, path = response.request.method, urlparse(response.url).path
        line = "HTTP {} {} {}".format(response.status, method, path)
        if any(status == response.status and verb == method and re.search(pattern, path) for status, verb, pattern in self.provoked):
            self.refusals.append(line)
        else:
            self.problems.append(line)

    def _failed(self, request: Any) -> None:
        failure = request.failure or ""
        # net::ERR_ABORTED is the page's own doing (a stream it left or stopped, a reload) or Chrome's word for a fetch answered 204.
        if "ERR_ABORTED" in failure or self.down:
            return
        self.problems.append("request failed: {} {} {}".format(request.method, urlparse(request.url).path, failure))

    def expect_refusal(self, status: int, method: str, path_pattern: str) -> None:
        """This test provokes this refusal on purpose: it is not a failure."""
        self.provoked.append((status, method, path_pattern))

    def server_down(self, down: bool) -> None:
        """While the server is down its requests fail: expected, not a failure."""
        self.down = down

    def assert_clean(self) -> None:
        assert not self.problems, "the page reported {} problem(s):\n  {}".format(len(self.problems), "\n  ".join(self.problems))

    def settle(self, ms: int = 250) -> None:
        """Give the pages a moment to report what is still on its way (a console error from a timer, a request that fails late). One
        wait is enough: Playwright delivers every page's events while any call waits."""
        for context in self.contexts:
            for page in context.pages:
                if not page.is_closed():
                    page.wait_for_timeout(ms)
                    return

    def close(self) -> None:
        for context in self.contexts:
            try:
                context.close()
            except Exception:   # a context of a browser that is already gone
                pass

    # -- looking
    def card(self, page: Any = None) -> Optional[Dict[str, Any]]:
        return (page or self.page).evaluate(CARD_JS)

    def cards(self, page: Any = None) -> int:
        return (page or self.page).evaluate("document.querySelectorAll('#conversation article.card').length")

    def composer(self, page: Any = None) -> Dict[str, Any]:
        return (page or self.page).evaluate(COMPOSER_JS)

    def drawer(self, page: Any = None) -> Dict[str, Any]:
        return (page or self.page).evaluate(DRAWER_JS)

    def sidebar(self, page: Any = None) -> List[Dict[str, Any]]:
        return (page or self.page).evaluate(SESSIONS_JS)

    def record(self, page: Any = None) -> None:
        """From now on the page records every state of its newest card (RECORD_JS); recorded() reads them."""
        (page or self.page).evaluate(RECORD_JS)

    def recorded(self, page: Any = None) -> Dict[str, Any]:
        return (page or self.page).evaluate("window.__rec")

    def session_id(self, page: Any = None) -> str:
        found = re.fullmatch(r"#/s/([\w-]+)", (page or self.page).evaluate("location.hash"))
        assert found, "the page shows no session: " + (page or self.page).evaluate("location.hash")
        return found.group(1)

    # -- doing
    def open(self, page: Any = None, hash_: str = "") -> Any:
        page = page or self.page
        page.goto(self.server.url + hash_)
        self.wait_loaded(page)
        return page

    def reload(self, page: Any = None) -> None:
        page = page or self.page
        page.reload()
        self.wait_loaded(page)

    def wait_loaded(self, page: Any = None, timeout: float = 30.0) -> None:
        """The app has started and drawn the chat its address names: every stored turn of it, or the empty chat."""
        page = page or self.page
        page.wait_for_function("document.querySelector('#mode-badge').textContent !== '' && location.hash !== ''",
                               timeout=timeout * 1000)
        found = re.fullmatch(r"#/s/([\w-]+)", page.evaluate("location.hash"))
        turns = len([m for m in self.server.session(found.group(1))["messages"] if m["role"] == "assistant"]) if found else 0
        page.wait_for_function(
            "(n) => document.querySelectorAll('#conversation article.card').length === n"
            " && (!document.querySelector('#send').disabled || !document.querySelector('#stop').hidden)", arg=turns,
            timeout=timeout * 1000)

    def attach(self, path: str, page: Any = None) -> None:
        """Choose the file as the file dialog would, and wait for the composer to show it."""
        page = page or self.page
        page.set_input_files("#file", path)
        page.wait_for_function("(name) => !document.querySelector('#preview').hidden"
                               " && document.querySelector('#preview').textContent.includes(name)", arg=os.path.basename(path))

    def drop(self, path: str, page: Any = None, target: str = "#image-well", kind: str = "drop", type_: str = "image/png") -> bool:
        """Drop the file on the target (or paste it into it): -> whether the page took the event (preventDefault)."""
        with open(path, "rb") as handle:
            b64 = base64.b64encode(handle.read()).decode("ascii")
        return (page or self.page).evaluate(DROP_JS, {"b64": b64, "name": os.path.basename(path), "type": type_, "target": target,
                                                      "kind": kind})

    def type_note(self, text: str, page: Any = None) -> None:
        page = page or self.page
        page.click("#prompt")
        page.keyboard.press("ControlOrMeta+a")
        page.keyboard.press("Backspace")
        page.keyboard.type(text)

    def wait_settled(self, before: int, page: Any = None, timeout: float = 60.0) -> Dict[str, Any]:
        """The newest card once there are more than `before` and it no longer runs."""
        page = page or self.page
        page.wait_for_function("(n) => { const c = [...document.querySelectorAll('#conversation article.card')];"
                               " return c.length > n && c[c.length - 1].getAttribute('data-status') !== 'running'; }",
                               arg=before, timeout=timeout * 1000)
        return self.card(page)

    def turn(self, image: Optional[str] = None, note: Optional[str] = None, page: Any = None, timeout: float = 60.0) -> Dict[str, Any]:
        """Attach (when given), type the note (when given), Send, and wait for that turn's card to settle."""
        page = page or self.page
        before = self.cards(page)
        if image:
            self.attach(image, page)
        if note is not None:
            self.type_note(note, page)
        page.click("#send")
        return self.wait_settled(before, page, timeout)

    def wait_running(self, page: Any = None, snapshots: int = 3, timeout: float = 30.0) -> None:
        """A turn runs, Stop can cancel it, and its report has grown over at least `snapshots` words."""
        (page or self.page).wait_for_function(
            "(n) => { const s = document.querySelector('#stop'); const c = [...document.querySelectorAll('#conversation article.card')].pop();"
            " const b = c && c.querySelector('.report-body');"
            " return !s.hidden && !s.disabled && c && c.getAttribute('data-status') === 'running' && b && b.textContent.split(/\\s+/).length >= n; }",
            arg=snapshots, timeout=timeout * 1000)

    def apply_settings(self, page: Any = None, **values: Any) -> None:
        """Open Settings, set each field as a user does (a click into a number selects it, the keys replace it; a switch is clicked when
        it is not as wanted), and Save: the drawer closes."""
        page = page or self.page
        page.click("#settings")
        for key, value in values.items():
            field = page.locator("[data-setting={}]".format(key))
            if isinstance(value, bool):
                if field.is_checked() != value:
                    field.click()
            else:
                field.click()
                page.keyboard.type(str(value))
                assert field.input_value() == str(value), "typing {} into {} gave {!r}".format(value, key, field.input_value())
        page.click("#drawer-save")
        page.wait_for_function("document.querySelector('#drawer').hidden")

    def shot(self, name: str, page: Any = None) -> None:
        """With CHAT_UI_E2E_EVIDENCE set, a screenshot of what the page shows now; each must stay under 400 KB."""
        if not self.evidence:
            return
        os.makedirs(self.evidence, exist_ok=True)
        path = os.path.join(self.evidence, name)
        (page or self.page).screenshot(path=path)
        size = os.path.getsize(path)
        assert size < MAX_SHOT_BYTES, "{} is {} bytes, over the 400 KB evidence limit".format(name, size)


# ---- fixtures -------------------------------------------------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def _sigterm_runs_teardown():
    """A SIGTERM (a timeout, a cancelled run) becomes a KeyboardInterrupt, so pytest still runs every teardown: each server runs in a
    process group of its own, which a signal to pytest does not reach, and would otherwise outlive the run."""
    def handler(signum: int, frame: Any) -> None:
        raise KeyboardInterrupt("SIGTERM")

    previous = signal.signal(signal.SIGTERM, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


@pytest.fixture(scope="session")
def browser():
    """The installed Google Chrome through Playwright, headless, for the whole run. Skips every test without Playwright or Chrome."""
    sync_api = pytest.importorskip("playwright.sync_api", reason="the e2e tests need `pip install -r requirements-e2e.txt`")
    manager = sync_api.sync_playwright().start()
    try:
        try:
            chrome = manager.chromium.launch(channel="chrome", headless=True)
        except sync_api.Error as exc:
            pytest.skip("Playwright cannot start the installed Google Chrome (channel='chrome'): {}".format(
                str(exc).strip().splitlines()[0]))
        try:
            yield chrome
        finally:
            chrome.close()
    finally:
        manager.stop()


@pytest.hookimpl(hookwrapper=True)   # the old form, which every pytest this repo meets understands (collection imports this file)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.when == "call":
        item.rep_call = report   # the ui fixture's teardown asks whether the test itself passed


@pytest.fixture
def ui(request, browser, tmp_path):
    """A fresh tiny server and a fresh page, both gone when the test ends, however it ends. Parametrized indirectly, its
    param is TinyServer's keyword arguments ({"tiny_gallery": True})."""
    home = tmp_path / "chat_home"
    home.mkdir()
    server = TinyServer(str(home), **getattr(request, "param", {}))
    harness = None
    try:
        harness = UI(browser, server, os.environ.get(EVIDENCE_ENV) or None)
        yield harness
        report = getattr(request.node, "rep_call", None)
        if report is not None and report.passed:   # asked again even after the test's own assert_clean: a problem that came later fails it too
            harness.settle()
            harness.assert_clean()
    finally:
        if harness is not None:
            harness.close()
        server.stop()
