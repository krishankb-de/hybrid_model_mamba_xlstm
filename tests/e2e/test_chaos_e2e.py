"""CHAT_UI_PLAN.md P8-D: the chaos checklist, the page side (items 1 to 4 and 8; 5 to 7 have no page of their own and are in tests/test_app_chaos.py,
which also holds the API side of 1, 3, 4 and 8). Each item is run in a real Chrome (Playwright) against the tiny engine on a free loopback port
(never 8000 or 8010), with its expected behaviour asserted on what the user sees and on what the server stored. The harness, the server and the
synthetic images are in conftest.py; three TinyServer subclasses here put the server behind a tunnel that can be cut, start it from the serving
command line with a token file, and give it a database that fills up when the test says (item 8, the P8-D follow-up).

    venv/bin/python -m pytest tests/e2e/test_chaos_e2e.py -m e2e -q
"""
import os
import re
import socket
import stat
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

import pytest

pytest.importorskip("playwright.sync_api", reason="the e2e tests need `pip install -r requirements-e2e.txt`")

from app import cli  # noqa: E402
from app.imaging import MAX_UPLOAD_BYTES  # noqa: E402
from app.pipeline import STORE_FAILED_MSG  # noqa: E402
from tests.e2e.conftest import REPO_ROOT, TinyServer, noise_png, squeezed  # noqa: E402

pytestmark = [pytest.mark.e2e, pytest.mark.slow]

RESTART_ERROR = "Error: The server restarted while this turn was running."
NOT_SAVED_ERROR = "Error: " + STORE_FAILED_MSG   # "Could not save this turn: ...": the card of a turn whose store failed (item 8)
TOO_BIG = "Image is larger than 20 MB."          # the page's own refusal and the server's 413 say the same sentence
ASK_FOR_TOKEN = "Enter the access token in Settings"
SETTLED_STAGES = {"preprocess": "done", "encode": "done", "retrieve": "skipped", "generate": "done", "label": "skipped", "score": "skipped"}
DELTA = b"event: content_block_delta"


def assistants(ui: Any, session_id: str) -> List[Dict[str, Any]]:
    """The stored assistant messages of a session, each with its events."""
    return [ui.server.message(m["id"]) for m in ui.server.session(session_id)["messages"] if m["role"] == "assistant"]


# ---- a tunnel that can be cut -------------------------------------------------------------------------------------------------------

def _hang_up(sock: socket.socket) -> None:
    """Close both directions at once (a FIN each way): what the other end of a connection sees when the tunnel that carried it is gone."""
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    sock.close()


class Tunnel:
    """A TCP relay on a free loopback port in front of one upstream port: what the SSH tunnel is to the page. It reads no request and answers
    nothing itself; bytes go both ways untouched, until cut_next_stream() arms it. Then the next response of content type text/event-stream is
    cut as its n-th content_block_delta frame has been passed to the page: both sockets of that connection are closed, the client side in the
    middle of the stream, with the server's turn still running behind it. With mid_frame the cut falls inside that frame, so that the page has
    half of a frame in its buffer. With down_for the tunnel is then down for that many seconds: every connection that was open is hung up, and
    one made meanwhile is closed at once, as a tunnel that is gone and comes back does."""

    def __init__(self, upstream_port: int) -> None:
        self.upstream_port, self.cuts = upstream_port, 0
        self._armed: Optional[Dict[str, Any]] = None
        self._down_until = 0.0
        self._lock = threading.Lock()
        self._open: List[socket.socket] = []
        self._closed = False
        self._listener = socket.socket()
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listener.bind(("127.0.0.1", 0))
        self._listener.listen(64)
        self._listener.settimeout(0.2)
        self.url = "http://127.0.0.1:{}/".format(self._listener.getsockname()[1])
        threading.Thread(target=self._accept, daemon=True).start()

    def cut_next_stream(self, deltas: int, mid_frame: bool = False, down_for: float = 0.0) -> None:
        with self._lock:
            self._armed = {"deltas": deltas, "mid_frame": mid_frame, "down_for": down_for}

    def close(self) -> None:
        self._closed = True
        self._hang_up_all()
        self._listener.close()

    def _hang_up_all(self) -> None:
        with self._lock:
            sockets, self._open = self._open, []
        for sock in sockets:
            _hang_up(sock)

    def _accept(self) -> None:
        while not self._closed:
            try:
                client, _ = self._listener.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            if time.monotonic() < self._down_until:   # the tunnel is down: the connection is made and is gone at once
                _hang_up(client)
                continue
            try:
                upstream = socket.create_connection(("127.0.0.1", self.upstream_port), timeout=10)
            except OSError:
                client.close()
                continue
            upstream.settimeout(None)
            client.settimeout(None)
            with self._lock:
                self._open += [client, upstream]
            threading.Thread(target=self._to_server, args=(client, upstream), daemon=True).start()
            threading.Thread(target=self._to_client, args=(client, upstream), daemon=True).start()

    def _to_server(self, client: socket.socket, upstream: socket.socket) -> None:
        try:
            while True:
                data = client.recv(65536)
                if not data:
                    break
                upstream.sendall(data)
        except OSError:
            pass
        finally:
            _hang_up(client)
            _hang_up(upstream)

    def _to_client(self, client: socket.socket, upstream: socket.socket) -> None:
        streaming, deltas, tail = False, 0, b""
        try:
            while True:
                data = upstream.recv(65536)
                if not data:
                    break
                if data.startswith(b"HTTP/"):   # the head of a response: is this one an event stream?
                    streaming, deltas, tail = b"text/event-stream" in data.lower(), 0, b""
                cut = None
                if streaming:
                    window = tail + data
                    deltas, tail = deltas + window.count(DELTA), window[-(len(DELTA) - 1):]
                    with self._lock:
                        if self._armed is not None and deltas >= self._armed["deltas"]:
                            cut, self._armed, self.cuts = self._armed, None, self.cuts + 1
                if cut is not None and cut["mid_frame"]:
                    at = data.rfind(DELTA)   # the end of that frame never arrives
                    data = data[:at + len(DELTA) + 12] if at >= 0 else data[:len(data) // 2]
                client.sendall(data)
                if cut is not None:
                    if cut["down_for"]:
                        self._down_until = time.monotonic() + cut["down_for"]
                        self._hang_up_all()
                    break
        except OSError:
            pass
        finally:
            _hang_up(client)
            _hang_up(upstream)


class TunnelServer(TinyServer):
    """The tiny server with a Tunnel in front of it: the page is served through the tunnel, and so are the harness's own reads."""

    def __init__(self, home: str, **kwargs: Any) -> None:
        super().__init__(home, **kwargs)
        self.tunnel = Tunnel(self.port)
        self.url = self.tunnel.url

    def stop(self) -> None:
        tunnel = getattr(self, "tunnel", None)   # stop() also ends a start that failed, before the tunnel was made
        if tunnel is not None:
            tunnel.close()
        super().stop()


# ---- a server that takes its token from a file ----------------------------------------------------------------------------------------

def write_token_file(path: str, token: str) -> None:
    """The token file as --token-file takes it (app/cli.py read_token): a new file of the user's, mode 0600."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.fchmod(fd, 0o600)   # read_token takes 0600 exactly, and the umask can narrow the mode os.open asks for
    with os.fdopen(fd, "w") as handle:
        handle.write(token + "\n")


class TokenFileServer(TinyServer):
    """The serving command line (`python -m app.server`) on the tiny engine, private mode, with --token-file: the server of a tunnel with an access
    token, which is the same one a public demo uses. The harness's own reads carry the right token, as the page does once it has it."""

    TOKEN = "chaos-e2e-token-4c1b9d"

    def __init__(self, home: str, **kwargs: Any) -> None:
        self.token_file = os.path.join(home, "app_token")
        write_token_file(self.token_file, self.TOKEN)
        super().__init__(home, **kwargs)

    def start(self, timeout: float = 90.0) -> None:
        argv = [sys.executable, "-m", "app.server", "--engine", "tiny", "--mode", "private", "--home", self.home, "--port", str(self.port),
                "--token-file", self.token_file]
        with open(self.log_path, "a") as log:
            self.proc = subprocess.Popen(argv, cwd=REPO_ROOT, env=dict(os.environ, PYTHONPATH=REPO_ROOT, PYTHONUNBUFFERED="1"),
                                         stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.time() + timeout
        while time.time() < deadline and self.proc.poll() is None:
            try:
                with urllib.request.urlopen(self.url + "healthz", timeout=2):   # /healthz needs no token
                    return
            except OSError:
                time.sleep(0.2)
        self.stop()
        raise RuntimeError("the token-file server did not start:\n" + self.log_tail())

    def _request(self, path: str, **kwargs: Any) -> urllib.request.Request:
        return urllib.request.Request(self.url + path.lstrip("/"), headers={"Authorization": "Bearer " + self.TOKEN}, **kwargs)

    def status_with(self, path: str, token: Optional[str]) -> int:
        """The status of a GET with this token (None: no Authorization header at all)."""
        request = urllib.request.Request(self.url + path.lstrip("/"), headers={"Authorization": "Bearer " + token} if token else {})
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                return response.status
        except urllib.error.HTTPError as exc:
            return exc.code


# ---- a database that fills up ---------------------------------------------------------------------------------------------------------

# The tiny app, as conftest.py's LAUNCH starts it, with one fault planted: while the file store_full is in its home, the store's event append
# raises sqlite3.OperationalError("database or disk is full"), as a database that has hit its disk quota does. argv: home, step delay, port.
STORE_FULL_FLAG = "store_full"
STORE_FAULT_LAUNCH = """\
import os
import sqlite3
import sys

import uvicorn
import app.server as server
from app.store import Store

FULL = os.path.join(sys.argv[1], %r)
_append_event = Store.append_event


def append_event(self, message_id, event, data):
    if os.path.exists(FULL):
        raise sqlite3.OperationalError("database or disk is full")
    return _append_event(self, message_id, event, data)


Store.append_event = append_event
uvicorn.run(server.create_app(engine="tiny", home=sys.argv[1], tiny_step_delay_s=float(sys.argv[2])),
            host="127.0.0.1", port=int(sys.argv[3]), log_level="warning")
""" % STORE_FULL_FLAG


class StoreFaultServer(TunnelServer):
    """The tiny server behind a Tunnel (TunnelServer) whose database fills up when the test says: fill() puts the file store_full in its home,
    and from then on the store takes no event; free() takes the file away, and the store works again."""

    def start(self, timeout: float = 90.0) -> None:
        with open(self.log_path, "a") as log:
            self.proc = subprocess.Popen([sys.executable, "-c", STORE_FAULT_LAUNCH, self.home, repr(self.step_delay_s), str(self.port)],
                                         cwd=REPO_ROOT, env=dict(os.environ, PYTHONPATH=REPO_ROOT), stdin=subprocess.DEVNULL, stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.time() + timeout
        while time.time() < deadline and self.proc.poll() is None:
            try:
                with urllib.request.urlopen("http://127.0.0.1:{}/healthz".format(self.port), timeout=2):   # the server itself, not the tunnel
                    return
            except OSError:
                time.sleep(0.2)
        self.stop()
        raise RuntimeError("the store-fault server did not start:\n" + self.log_tail())

    def fill(self) -> None:
        open(os.path.join(self.home, STORE_FULL_FLAG), "w").close()

    def free(self) -> None:
        os.remove(os.path.join(self.home, STORE_FULL_FLAG))


@pytest.fixture(scope="session")
def limit_image(tmp_path_factory) -> str:
    """An X-ray of exactly the size the page allows (20 MiB): a PNG of noise, then zero bytes, which a PNG reader never reads. The page takes it."""
    png = noise_png(320, 320)
    path = tmp_path_factory.mktemp("limit") / "limit.png"
    path.write_bytes(png + b"\0" * (MAX_UPLOAD_BYTES - len(png)))
    assert path.stat().st_size == MAX_UPLOAD_BYTES
    return str(path)


# ---- 1: the API is killed mid-turn --------------------------------------------------------------------------------------------------

def test_chaos_1_a_page_opened_after_a_kill_and_a_restart_replays_the_turn_as_an_error_card_and_sends_again(ui, images):
    """Chaos item 1, the API killed mid-turn, the page side (the API side, with a real process, is in tests/test_app_chaos.py). A 200-token turn
    runs on the page and the server is killed with SIGKILL; a new server starts on the same home and port. Expected: the start-up says it
    recovered 1 turn; a page opened after that, which has only what the server stored, shows the turn's card as an error ("The server restarted
    while this turn was running."), with no stage spinning and the report that streamed before the kill left out of it, and the page that was
    watching the turn ends in the same card; the composer is usable and the next Send runs to `done`."""
    page = ui.open()
    ui.apply_settings(max_new_tokens=200, stop_on_repeat=False)
    ui.attach(images["xray_a.png"])
    page.click("#send")
    ui.wait_running(snapshots=4)
    sid = ui.session_id()
    ui.server_down(True)
    ui.server.kill()
    ui.server.start()
    assert "[server] recovered 1 running turn(s)" in ui.server.log_tail(20000)
    stored = assistants(ui, sid)[-1]
    assert stored["status"] == "error" and [r["data"]["error"]["type"] for r in stored["events"] if r["event"] == "error"] == ["server_restart"]

    later = ui.new_page()   # a page that was not there when it happened
    ui.open(later, "#/s/" + sid)
    replayed = ui.card(later)
    assert replayed["status"] == "error" and RESTART_ERROR in replayed["notes"] and replayed["spinning"] == 0 and replayed["n"] == 1
    assert replayed["raw"] is None and ui.composer(later)["stop_hidden"] and not ui.composer(later)["send_disabled"]
    watching = ui.wait_settled(0, page, timeout=40)   # the page that saw it happen polled its way to the same end
    ui.server_down(False)
    assert watching["status"] == "error" and RESTART_ERROR in watching["notes"] and watching["spinning"] == 0
    assert ui.composer(page)["notice"] is None and ui.composer(page)["stop_hidden"] and not ui.composer(page)["send_disabled"]
    card = ui.turn(note="tokens 16", page=later)
    assert card["status"] == "done" and card["n"] == 2
    ui.assert_clean()


# ---- 2: the tunnel drops mid-stream -------------------------------------------------------------------------------------------------

# How the tunnel goes: (the report snapshot after which the page is cut off, how: at the end of a frame, inside one, or with the tunnel then down
# for three seconds before it is back, the fewest times the page asks for the turn, the fewest different reports it shows while the turn runs
# (a tunnel that is down has no new report to show until it is back, which then brings the rest at once)).
CUTS = {"before_any_frame": (0, {}, 1, 3), "between_frames": (3, {}, 1, 3), "mid_frame": (3, {"mid_frame": True}, 1, 3),
        "down_for_3_s": (3, {"down_for": 3.0}, 3, 3)}


@pytest.mark.parametrize("ui", [{"server": TunnelServer}], indirect=True, ids=["tunnel"])
@pytest.mark.parametrize("how", list(CUTS), ids=list(CUTS))
def test_chaos_2_a_tunnel_dropped_mid_stream_leaves_a_card_that_the_pages_polling_completes_with_the_report_of_an_uncut_turn(ui, images, how):
    """Chaos item 2, the tunnel dropped mid-stream. The page is served through a TCP relay (the tunnel). One turn runs uncut. The same X-ray
    with the same settings (60 tokens, three seconds on the tiny engine) is sent again, and the relay closes the client side of its event stream
    with the turn still running behind it: right after the response's head, before any report snapshot (before_any_frame), after the third
    snapshot at the end of a frame (between_frames), inside the third frame, so that the page holds half a frame (mid_frame), or at the end
    of that frame with the tunnel then down for three seconds, so that the first polls find no connection (down_for_3_s). Expected: the page
    falls back to polling `GET /v1/messages/<id>` (the uncut turn needed no poll), and keeps asking while the tunnel is down; the card goes on
    from where the stream stopped and ends `done` with every stage settled, no error and no spinner, and the page shows no notice and takes
    the next Send; its report is the report of the uncut turn, the same words on the card and in the store, from a stored log whose events are
    the same events in the same order with seq 1, 2, ... and no gap; and the page's own chat then reloads to the same two cards."""
    deltas, cut_args, min_polls, min_texts = CUTS[how]
    ui.open()
    ui.apply_settings(max_new_tokens=60, stop_on_repeat=False)
    uncut = ui.turn(images["xray_a.png"])
    sid = ui.session_id()
    assert uncut["status"] == "done" and uncut["stages"] == SETTLED_STAGES and uncut["shown"]

    ui.server.tunnel.cut_next_stream(deltas, **cut_args)
    ui.server_down(True)   # the cut is a failed request of the page, which is the point: the harness is told
    ui.record()
    cut = ui.turn(images["xray_a.png"], timeout=30)
    ui.settle()
    ui.server_down(False)
    assert ui.server.tunnel.cuts == 1, "the stream was cut once"
    assert cut["status"] == "done" and cut["stages"] == SETTLED_STAGES and cut["spinning"] == 0 and cut["n"] == 2
    assert cut["notes"] == [] and cut["shown"] == uncut["shown"], "the same report as the uncut turn, on the card"

    first, second = assistants(ui, sid)
    assert first["status"] == second["status"] == "done" and first["id"] != second["id"]
    assert second["report"] == first["report"] and second["display_report"] == first["display_report"] and second["report"]
    assert squeezed(cut["shown"]) == squeezed(second["display_report"])
    assert [(e["event"], e["data"].get("stage")) for e in second["events"]] == [(e["event"], e["data"].get("stage")) for e in first["events"]]
    assert [e["seq"] for e in second["events"]] == list(range(1, len(second["events"]) + 1))
    polls = [path for method, path in ui.requests if method == "GET" and path == "/v1/messages/" + second["id"]]
    assert len(polls) >= min_polls, "the page followed the cut turn by polling it: {} asks".format(len(polls))
    assert "/v1/messages/" + first["id"] not in [path for _, path in ui.requests], "the uncut turn was never polled"
    rec = ui.recorded()
    states = [f["status"] for f in rec["frames"]]
    assert states[-1] == "done" and "running" in states and "error" not in states
    assert len(rec["texts"]) >= min_texts, "the report kept growing on the card, before the cut and after it: {} different texts".format(len(rec["texts"]))
    composer = ui.composer()
    assert composer["notice"] is None and composer["stop_hidden"] and not composer["send_disabled"], composer
    ui.reload()   # and what the server stored is what the page shows
    assert ui.cards() == 2 and ui.card()["shown"] == cut["shown"]
    ui.assert_clean()


# ---- 3: an oversize upload ----------------------------------------------------------------------------------------------------------

def test_chaos_3_an_oversize_upload_is_a_413_the_page_says_so_and_the_preview_stays_to_remove_or_send_again(ui, images, limit_image):
    """Chaos item 3, an oversize upload, the page side (the API side is pinned by tests/test_app_hardening.py::test_oversize_body_is_413_before_
    parsing and tests/test_app_api.py::test_a_content_length_over_the_cap_is_refused_before_the_body_is_read). The page refuses a file of over 20 MB
    itself, so a 413 reaches it only for a request that is over the server's cap as a whole (21 MB): an X-ray of exactly 20 MiB, which the page
    takes, and a note of 1.3 MB with it. Expected: the server answers 413 (the request log line says so) before it parses anything; the page shows
    the size message ("Image is larger than 20 MB."); the preview and the note stay where they were; no card, no chat and no upload is left; the
    composer is usable, so the user can remove the file, or send again, and a Send with the note cut short goes through to `done`. First line of
    defence, the page's own: a file of over 20 MB chosen while another is attached is refused with the same message, nothing is sent, and the
    file that was attached stays."""
    page = ui.open()
    ui.attach(images["xray_a.png"])
    page.set_input_files("#file", images["huge.png"])   # over 20 MB: the page refuses it, and what was attached stays
    page.wait_for_function("(m) => document.querySelector('#notice p').textContent === m", arg=TOO_BIG)
    assert (ui.composer()["preview"] or "").startswith("xray_a.png") and ui.requests.count(("POST", "/v1/sessions")) == 0
    page.click("#preview button")
    assert ui.composer()["preview"] is None

    ui.expect_refusal(413, "POST", r"/messages$")
    ui.attach(limit_image)
    note = "n" * 1_300_000
    page.fill("#prompt", note)
    with page.expect_response(re.compile(r"/v1/sessions/[\w-]+/messages$")) as sent:
        page.click("#send")
    assert sent.value.status == 413 and sent.value.json()["error"]["message"] == TOO_BIG
    page.wait_for_function("(m) => !document.querySelector('#notice').hidden && document.querySelector('#notice p').textContent === m", arg=TOO_BIG)
    composer = ui.composer()
    assert composer["notice"] == TOO_BIG and (composer["preview"] or "").startswith("limit.png"), "the preview stays: {!r}".format(composer["preview"])
    assert composer["prompt"] == note and not composer["send_disabled"] and composer["stop_hidden"]
    assert ui.cards() == 0 and ui.server.get("healthz")["turns_in_flight"] == 0
    page.wait_for_function("location.hash === '#/new'")   # the chat made for the refused first turn is taken back, which the page does in the background
    for _ in range(100):
        if ui.server.sessions() == []:
            break
        page.wait_for_timeout(50)
    assert ui.server.sessions() == [] and len(ui.refusals) == 1 and ui.refusals[0].startswith("HTTP 413 POST /v1/sessions/")

    page.fill("#prompt", "")   # send again, the note cut short: the same file goes through
    card = ui.turn()
    assert card["status"] == "done" and ui.composer()["preview"] is None
    sid = ui.session_id()
    assert ui.server.session(sid)["messages"][0]["image_filename"] == "limit.png"

    ui.attach(limit_image)   # or remove the file: it goes, and the note stays for what the user wants to say next
    page.fill("#prompt", note)
    with page.expect_response(re.compile(r"/v1/sessions/[\w-]+/messages$")) as again:
        page.click("#send")
    assert again.value.status == 413 and (ui.composer()["preview"] or "").startswith("limit.png")
    page.click("#preview button")
    assert ui.composer()["preview"] is None and ui.composer()["prompt"] == note and len(ui.refusals) == 2
    ui.assert_clean()


# ---- 4: a wrong token ---------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("ui", [{"server": TokenFileServer}], indirect=True, ids=["token_file"])
def test_chaos_4_a_wrong_token_is_a_401_the_page_asks_for_the_token_in_settings_and_the_right_one_makes_the_next_turn_work(ui, images):
    """Chaos item 4, a wrong token. A server started from the serving command line with a token file. Expected: the API refuses a wrong token and
    a missing one with 401 and answers /healthz without one; the page, which has no token, says "Enter the access token in Settings" and the
    drawer has a field for it ("Access token"); a wrong token typed there is refused again (401) with the same sentence, and a Send with it
    creates no chat and keeps the preview, so nothing is lost; the right token typed in Settings takes the notice down (P8-D fix: it used to stay on a page that
    worked again) and makes the next Send run through to `done`, and the chat it made is stored."""
    server = ui.server
    for token, want in ((None, 401), ("chaos-e2e-token-WRONG", 401), (server.TOKEN, 200)):
        assert server.status_with("v1/models", token) == want, "GET /v1/models with the token {!r}".format(token)
    assert server.status_with("healthz", None) == 200

    ui.expect_refusal(401, "GET", r"^/v1/")
    ui.expect_refusal(401, "POST", r"^/v1/")
    page = ui.page
    page.goto(server.url)
    page.wait_for_function("document.querySelector('#mode-badge').textContent !== '' && !document.querySelector('#notice').hidden", timeout=30000)
    assert ui.composer()["notice"] == ASK_FOR_TOKEN   # no token at all
    first_refusals = len(ui.refusals)
    assert first_refusals >= 2   # the models and the chats of the page's own start-up

    def type_token(value: str) -> int:
        """Open Settings, put the value in its Access token field and leave the field (the change that saves it); -> the status of /v1/models."""
        if page.evaluate("document.querySelector('#drawer').hidden"):
            page.click("#settings")
        field = page.get_by_label("Access token")
        field.fill(value)
        with page.expect_response(re.compile(r"/v1/models$")) as asked:
            field.press("Tab")
        return asked.value.status

    assert type_token("chaos-e2e-token-WRONG") == 401   # a wrong token: refused, and said in the same words
    page.wait_for_function("(m) => !document.querySelector('#notice').hidden && document.querySelector('#notice p').textContent === m", arg=ASK_FOR_TOKEN)
    page.click("#drawer-close")
    ui.attach(images["xray_a.png"])
    page.click("#send")
    page.wait_for_function("(m) => document.querySelector('#notice p').textContent === m && !document.querySelector('#send').disabled", arg=ASK_FOR_TOKEN)
    composer = ui.composer()
    assert composer["notice"] == ASK_FOR_TOKEN and (composer["preview"] or "").startswith("xray_a.png") and not composer["send_disabled"]
    assert ui.cards() == 0 and ui.server.sessions() == [] and len(ui.refusals) > first_refusals

    assert type_token(server.TOKEN) == 200   # the right one, typed in Settings
    page.wait_for_function("document.querySelector('#notice').hidden", timeout=5000)   # accepted: the page does not go on asking for what it was just given
    assert ui.composer()["notice"] is None and (ui.composer()["preview"] or "").startswith("xray_a.png")
    page.click("#drawer-close")
    card = ui.turn()
    assert card["status"] == "done" and card["stages"] == SETTLED_STAGES
    [session] = ui.server.sessions()
    assert session["turns"] == 1 and ui.server.session(session["id"])["messages"][0]["image_filename"] == "xray_a.png"
    ui.assert_clean()


def test_the_token_file_of_the_token_file_server_is_mode_0600_whatever_the_umask(tmp_path):
    """read_token takes a file of mode 0600 and no other, and the umask can narrow the mode os.open asks for (0277 leaves 0400): the token file
    TokenFileServer writes is 0600 under such a umask too, and read_token reads it."""
    path = str(tmp_path / "app_token")
    umask = os.umask(0o277)
    try:
        write_token_file(path, "chaos-umask-token")
    finally:
        os.umask(umask)
    assert oct(stat.S_IMODE(os.stat(path).st_mode)) == oct(0o600) and cli.read_token(path) == "chaos-umask-token"


# ---- 8: the store fails while a turn runs ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("ui", [{"server": StoreFaultServer}], indirect=True, ids=["store_fault"])
@pytest.mark.parametrize("how", ["streaming", "reloaded", "stream_cut"])
def test_chaos_8_a_store_that_fills_up_mid_turn_ends_the_card_as_an_error_with_the_fixed_message_and_the_page_stops_asking(ui, images, how):
    """Chaos item 8, the store fails while a turn runs, the page side (the API side is in tests/test_app_chaos.py). A 200-token turn runs on the
    page, and the database fills up (sqlite3.OperationalError "database or disk is full" at every event append from then on) while the page
    follows the turn: on its own stream (streaming), after a reload that found the turn running (reloaded), or by polling once the tunnel cut
    its stream (stream_cut). Expected: the card ends as an error with the fixed sentence ("Error: Could not save this turn: ...") and no other
    note, with no stage spinning, and the composer is usable with no notice; the page that kept its stream got the ending there and never
    asked for the message; a page that polled stops asking: its count of GET /v1/messages/<id> does not grow once the card has ended; a reload
    then replays the same error card from one read of the turn, and does not poll it; the server stored the events before the failure and is
    not left running a turn, and its log names the class and the turn, not the database's own words; and with room on the disk again the next
    Send runs to `done`."""
    server = ui.server
    page = ui.open()
    ui.apply_settings(max_new_tokens=200, stop_on_repeat=False)
    if how == "stream_cut":
        server.tunnel.cut_next_stream(3)
        ui.server_down(True)   # the cut is a failed request of the page, which is the point: the harness is told
    ui.attach(images["xray_a.png"])
    page.click("#send")
    ui.wait_running(snapshots=4)
    mid = ui.card()["id"]

    def asked() -> int:
        return len([path for method, path in ui.requests if method == "GET" and path == "/v1/messages/" + mid])

    if how == "reloaded":
        ui.reload()   # the page that comes back reads the turn, finds it running and polls it
    if how != "streaming":
        least = 2 if how == "reloaded" else 1   # a reload reads the turn once before it polls
        for _ in range(200):
            if asked() >= least:
                break
            page.wait_for_timeout(50)
        assert asked() >= least, "the page follows the turn by polling it before the store fills up"
    server.fill()
    card = ui.wait_settled(0, timeout=30)
    ui.server_down(False)
    assert card["status"] == "error" and card["notes"] == [NOT_SAVED_ERROR] and card["spinning"] == 0, card
    composer = ui.composer()
    assert composer["notice"] is None and composer["stop_hidden"] and not composer["send_disabled"], composer
    ended = asked()
    page.wait_for_timeout(2500)   # five of the page's poll intervals
    assert asked() == ended, "the page stopped asking once the card had ended: {} asks, then {}".format(ended, asked())
    if how == "streaming":
        assert ended == 0, "the ending came on the stream: the page never asked for the message"
    ui.reload()   # the chat as a page that loads it now replays it: the same card, read once and not polled
    page.wait_for_timeout(1500)
    replayed = ui.card()
    assert replayed["status"] == "error" and replayed["notes"] == [NOT_SAVED_ERROR] and replayed["spinning"] == 0, replayed
    assert asked() == ended + 1, "the replay read the turn once and did not poll it: {} asks after {}".format(asked(), ended)

    stored = server.message(mid)
    events = stored["events"]
    assert stored["status"] == "error" and [e["seq"] for e in events] == list(range(1, len(events) + 1))
    assert [e["event"] for e in events[-2:]] == ["error", "message_stop"] and events[-2]["data"]["error"]["type"] == "store_failed"
    assert "content_block_delta" in [e["event"] for e in events], "what was stored before the failure is kept"
    assert server.get("healthz")["turns_in_flight"] == 0
    log = server.log_tail(20000)
    assert "could not store an event of turn {}: OperationalError".format(mid) in log and "database or disk is full" not in log
    server.free()
    card = ui.turn(note="tokens 16")
    assert card["status"] == "done" and card["n"] == 2
    ui.assert_clean()
