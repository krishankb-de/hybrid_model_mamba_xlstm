"""CHAT_UI_PLAN.md P8-D: the chaos checklist, the API side. Each item is run against the tiny engine on a free loopback port (never 8000 or 8010)
with its expected behaviour asserted; the page side of items 1 to 4 is tests/e2e/test_chaos_e2e.py. Synthetic data only (R7).

    1  the API is killed mid-turn              test_chaos_1_*   a real server process, SIGKILL, a second process on the same home
    2  the tunnel drops mid-stream             e2e only         a TCP proxy in front of the server; the page polls the card to its end
    3  an oversize upload                      test_chaos_3_*   the 413 itself is pinned by the tests this one names; the page: e2e
    4  a wrong token                           test_chaos_4_*   no side effects; the page asks for the token in Settings: e2e
    5  the labeller is down                    test_chaos_5_*   refused, hung up on, and stopped while a turn runs
    6  the gallery is missing                  test_chaos_6_*   never there, not openable and allowed, and gone after the start
    7  a disk-quota error while an upload is written    test_chaos_7_*
"""
import errno
import http.server
import json
import logging
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import httpx
import pytest
from fastapi.testclient import TestClient

from app import cli
from app.imaging import TOO_LARGE_MSG
from app.labels import CHEXBERT_14
from app.pipeline import TEST_IMAGE_MSG
from app.server import GALLERY_IMAGE_MSG, create_app
from app.store import RESTART_MESSAGE
from tests.app_helpers import iter_sse, png_bytes, start_live_server, wait_until

REPO_ROOT = Path(__file__).resolve().parent.parent
STAGES = ["preprocess", "encode", "retrieve", "generate", "label", "score"]
STORE_FAILED_MSG = "Could not store the image."
# create_app's arguments (the step delay among them) cannot pass through the serving command line, so the server of item 1 starts through this.
LAUNCH = """\
import sys
import uvicorn
import app.server as server

uvicorn.run(server.create_app(engine="tiny", home=sys.argv[1], tiny_step_delay_s=float(sys.argv[2])),
            host="127.0.0.1", port=int(sys.argv[3]), log_level="warning")
"""


# ---- helpers ------------------------------------------------------------------------------------------------------------------------

def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _post(c, sid, options=None, image=True, text="", headers=None):
    files = {"image": ("x.png", png_bytes(), "image/png")} if image else None
    data = {"text": text, "options": json.dumps({"max_new_tokens": 16} if options is None else options)}
    return c.post("/v1/sessions/{}/messages".format(sid), files=files, data=data, headers=headers)


def _turn(c, options=None, sid=None, image=True, headers=None):
    """One whole turn: -> its frames, which carry seq 1, 2, ... without a gap."""
    r = _post(c, sid or c.post("/v1/sessions", json={}, headers=headers).json()["id"], options, image, headers=headers)
    assert r.status_code == 200, r.text
    frames = list(iter_sse([r.text]))
    assert [f["data"]["seq"] for f in frames] == list(range(1, len(frames) + 1))
    return frames


def _stages(frames):
    """Each stage of the turn as it ended: "ran", or "skipped:<reason>"."""
    return {f["data"]["stage"]: "skipped:" + f["data"]["skipped"] if "skipped" in f["data"] else "ran"
            for f in frames if f["event"] == "stage_end"}


def _started(frames):
    return [f["data"]["stage"] for f in frames if f["event"] == "stage_start"]


class ServerProcess:
    """The tiny server as a process of its own, in a session of its own (a kill reaches all of it), on a free loopback port, with its output in a
    log file. One home, many lives: kill() is a crash (SIGKILL: no handler, no lifespan, no flush) and start() brings the next one up on the same home."""

    def __init__(self, home: Path, log: Path, step_delay_s: float = 0.0):
        self.home, self.log, self.step_delay_s = home, log, step_delay_s
        self.proc: Optional[subprocess.Popen] = None
        self.base = ""

    def start(self, timeout: float = 90.0) -> None:
        port = _free_port()
        with open(str(self.log), "ab") as out:
            self.proc = subprocess.Popen([sys.executable, "-c", LAUNCH, str(self.home), repr(self.step_delay_s), str(port)], cwd=str(REPO_ROOT),
                                         env=dict(os.environ, PYTHONPATH=str(REPO_ROOT), PYTHONUNBUFFERED="1"), stdin=subprocess.DEVNULL,
                                         stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
        self.base = "http://127.0.0.1:{}".format(port)
        deadline = time.time() + timeout
        while time.time() < deadline and self.proc.poll() is None:
            try:
                if httpx.get(self.base + "/healthz", timeout=2).status_code == 200:
                    return
            except httpx.HTTPError:
                time.sleep(0.2)
        self.kill()
        raise RuntimeError("the tiny server did not start:\n" + self.text()[-3000:])

    def kill(self) -> None:
        if self.proc is None:
            return
        if self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)   # a session leader: its pid is its process group
            except OSError:
                pass
        self.proc.wait(timeout=10)

    def text(self) -> str:
        return self.log.read_text(errors="replace") if self.log.exists() else ""


@pytest.fixture
def processes():
    """ServerProcess factory; whatever is still running when the test is over is killed."""
    made = []

    def make(home: Path, log: Path, step_delay_s: float = 0.0) -> ServerProcess:
        made.append(ServerProcess(home, log, step_delay_s))
        return made[-1]

    yield make
    for process in made:
        process.kill()


# ---- 1: the API is killed mid-turn --------------------------------------------------------------------------------------------------

def test_chaos_1_a_server_killed_mid_turn_leaves_a_turn_that_the_next_start_closes_as_error_server_restart(tmp_path, processes):
    """Chaos item 1, the API killed mid-turn. A real server process (tiny engine, 50 ms a step) starts a 200-token turn, and is killed with
    SIGKILL once three report snapshots have streamed. A second process starts on the same home. Expected: the start-up says it recovered 1
    turn; the turn is stored as `error`, its log ends with an `error` event of type `server_restart` and a `message_stop` of status `error`, with
    no report and its seq contiguous and what streamed before the kill kept; the replay (the message, the chat with its upload's picture, its
    Markdown export) shows an error and no running turn; and the second server takes the next turn through to `done`."""
    home = tmp_path / "home"
    first = processes(home, tmp_path / "first.log", step_delay_s=0.05)
    first.start()
    sid = httpx.post(first.base + "/v1/sessions", json={}).json()["id"]
    seen = []
    with httpx.stream("POST", first.base + "/v1/sessions/{}/messages".format(sid), files={"image": ("x.png", png_bytes(), "image/png")},
                      data={"text": "", "options": json.dumps({"max_new_tokens": 200})}, timeout=30) as r:
        message_id = r.headers["X-Message-Id"]
        for frame in iter_sse(r.iter_text()):
            seen.append(frame["event"])
            if seen.count("content_block_delta") == 3:
                first.kill()   # mid-generate: 197 steps, about ten seconds, are still to come
                break
    assert first.proc.returncode == -signal.SIGKILL, "a crash, not a shutdown: the process had no chance to close its turn"

    second = processes(home, tmp_path / "second.log")
    second.start()
    assert "[server] recovered 1 running turn(s)" in second.text(), second.text()
    message = httpx.get(second.base + "/v1/messages/" + message_id).json()
    events = message["events"]
    assert message["status"] == "error" and message["report"] is None and message["display_report"] is None
    assert [e["seq"] for e in events] == list(range(1, len(events) + 1))
    assert [e["event"] for e in events[-2:]] == ["error", "message_stop"]
    assert events[-2]["data"]["error"] == {"type": "server_restart", "message": RESTART_MESSAGE}
    assert events[-1]["data"]["status"] == "error" and events[-1]["data"]["report"] is None
    assert [e["event"] for e in events].count("content_block_delta") >= 3, "what streamed before the kill is kept"
    assert "message_stop" not in [e["event"] for e in events[:-1]]

    chat = httpx.get(second.base + "/v1/sessions/" + sid).json()
    assert [(m["role"], m["status"]) for m in chat["messages"]] == [("user", "done"), ("assistant", "error")]
    thumb = httpx.get(second.base + "/v1/messages/{}/image".format(chat["messages"][0]["id"]), params={"variant": "thumb"})
    assert thumb.status_code == 200, "the killed turn's picture is still there: the replayed card shows it"
    exported = httpx.get(second.base + "/v1/sessions/{}/export".format(sid), params={"format": "md"}).text
    assert "Status: error · Error: " + RESTART_MESSAGE in exported
    assert httpx.get(second.base + "/healthz").json()["turns_in_flight"] == 0
    again = httpx.post(second.base + "/v1/sessions/{}/messages".format(sid), data={"text": "", "options": json.dumps({"max_new_tokens": 16})}, timeout=60)
    assert again.status_code == 200 and list(iter_sse([again.text]))[-1]["data"]["status"] == "done"


# ---- 3: an oversize upload ----------------------------------------------------------------------------------------------------------

def test_chaos_3_the_413_for_a_request_over_21_mb_is_pinned_by_the_hardening_and_api_tests_and_the_message_is_the_size_message():
    """Chaos item 3, an oversize upload, API side. Expected: a request over 21 MB gets 413 before its body is parsed. That is pinned already, by
    tests/test_app_hardening.py::test_oversize_body_is_413_before_parsing (and ::test_a_declared_length_over_the_cap_is_answered_without_one_byte_of_
    the_body_being_read) and by tests/test_app_api.py::test_a_content_length_over_the_cap_is_refused_before_the_body_is_read, over a real socket,
    so none of it is repeated here. This test only keeps those names from rotting, and pins the sentence the page shows for it. The page: the e2e
    test of this item (the 413 reaches the page, the preview stays)."""
    from tests import test_app_api, test_app_hardening
    for module, name in ((test_app_hardening, "test_oversize_body_is_413_before_parsing"),
                         (test_app_hardening, "test_a_declared_length_over_the_cap_is_answered_without_one_byte_of_the_body_being_read"),
                         (test_app_api, "test_a_content_length_over_the_cap_is_refused_before_the_body_is_read")):
        assert callable(getattr(module, name, None)), "{}.{} is the API-level pin of chaos item 3".format(module.__name__, name)
    assert TOO_LARGE_MSG == "Image is larger than 20 MB."


# ---- 4: a wrong token ---------------------------------------------------------------------------------------------------------------

def test_chaos_4_a_wrong_token_is_a_401_before_anything_is_read_and_stores_nothing(tmp_path):
    """Chaos item 4, a wrong token, API side. A server whose token comes from a token file (mode 0600, as app/cli.py reads it). Expected: a wrong
    token, a missing one and another scheme's all get 401 `authentication_error` with `WWW-Authenticate: Bearer`, on a read, on a session
    create and on a turn with a 1 MB image, and the message never repeats what was sent; the turn is refused before its body is parsed, so no
    message, no upload and no slot is taken; /healthz stays open; the right token works, and then the turn is stored as usual. The page asks for
    the token in Settings: the e2e test of this item."""
    token_file = tmp_path / "app_token"
    token_file.write_text("chaos-token-7f3a9c\n")
    token_file.chmod(0o600)
    token = cli.read_token(str(token_file))
    wrong = "chaos-token-WRONG"
    home = tmp_path / "home"
    with TestClient(create_app(engine="tiny", home=str(home), token=token)) as c:
        good = {"Authorization": "Bearer " + token}
        sid = c.post("/v1/sessions", json={}, headers=good).json()["id"]
        refused = 0
        for headers in ({"Authorization": "Bearer " + wrong}, {}, {"Authorization": "Basic " + token}):
            for call in (lambda: c.get("/v1/models", headers=headers), lambda: c.post("/v1/sessions", json={}, headers=headers),
                         lambda: _post(c, sid, headers=headers)):
                r = call()
                refused += 1
                assert r.status_code == 401 and r.headers["www-authenticate"] == "Bearer"
                assert r.json()["error"] == {"type": "authentication_error", "message": "Missing or wrong token: send Authorization: Bearer <token>."}
                assert wrong not in r.text and token not in r.text
        assert refused == 9
        assert c.get("/healthz").status_code == 200, "/healthz stays open"
        chat = c.get("/v1/sessions/" + sid, headers=good).json()
        assert chat["messages"] == [] and [s["id"] for s in c.get("/v1/sessions", headers=good).json()["sessions"]] == [sid]
        assert list((home / "uploads").rglob("*")) == [] and c.get("/healthz").json()["turns_in_flight"] == 0
        frames = _turn(c, sid=sid, headers=good)
        assert frames[-1]["data"]["status"] == "done"
        assert len(list((home / "uploads" / sid).iterdir())) == 1, "the right token's turn stored its upload"


# ---- 5: the labeller is down --------------------------------------------------------------------------------------------------------

class LabellerStub:
    """A CheXbert labeller's two routes on a free loopback port: GET /healthz, and POST /label (a row of 14 zeros per text). With answers=False
    /label hangs up without a reply, as a labeller does that died between the health check and the call. stop() takes all of it down at once, as
    the end of its job does: the port then refuses connections."""

    def __init__(self, answers: bool = True):
        stub = self
        self.answers, self.labelled = answers, 0

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, code, body):
                data = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                self.reply(200 if self.path == "/healthz" else 404, {"status": "ok"})

            def do_POST(self):
                texts = json.loads(self.rfile.read(int(self.headers["Content-Length"])))["texts"]
                if not stub.answers:
                    self.close_connection = True   # no status line, no body: the peer is gone
                    return
                stub.labelled += len(texts)
                self.reply(200, {"label_names": CHEXBERT_14, "labels": [[0] * len(CHEXBERT_14) for _ in texts]})

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = "http://127.0.0.1:{}".format(self.server.server_address[1])
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def labellers():
    made = []

    def make(answers: bool = True) -> LabellerStub:
        made.append(LabellerStub(answers))
        return made[-1]

    yield make
    for stub in made:
        stub.stop()   # one already stopped by its test is stopped again harmlessly


def test_chaos_5_a_labeller_that_refuses_connections_skips_the_label_stage_and_the_turn_ends_done(tmp_path):
    """Chaos item 5, the labeller down: its URL refuses connections (nothing listens on the port). Expected: the label stage is skipped with its
    reason, `labeler_unavailable`, before it starts (no stage_start), the turn ends `done` with its report, and so does the next turn;
    /v1/label answers 503 with the labeller's own sentence."""
    url = "http://127.0.0.1:{}".format(_free_port())
    with TestClient(create_app(engine="tiny", home=str(tmp_path), labeler_url=url)) as c:
        for _ in range(2):
            frames = _turn(c)
            assert _stages(frames)["label"] == "skipped:labeler_unavailable" and "label" not in _started(frames)
            assert _stages(frames)["generate"] == "ran" and frames[-1]["data"]["status"] == "done" and frames[-1]["data"]["report"]
            assert [f["data"]["stage"] for f in frames if f["event"] == "stage_end"] == STAGES
        r = c.post("/v1/label", json={"text": "No effusion."})
        assert r.status_code == 503 and r.json()["error"] == {"type": "unavailable_error",
                                                              "message": "The CheXbert labeller did not answer; try again shortly."}


def test_chaos_5_a_labeller_that_hangs_up_on_the_call_skips_the_stage_after_it_started_and_the_turn_ends_done(tmp_path, labellers):
    """Chaos item 5, the labeller down between its health check and its call (it answers /healthz and hangs up on /label). Expected: the stage
    starts and ends skipped with `labeler_unavailable` (stage_start, then a skipped stage_end), the turn ends `done`; the labeller is then taken
    to be down, so the next turn skips the stage before it starts."""
    stub = labellers(answers=False)
    with TestClient(create_app(engine="tiny", home=str(tmp_path), labeler_url=stub.url)) as c:
        frames = _turn(c)
        assert _stages(frames)["label"] == "skipped:labeler_unavailable" and "label" in _started(frames)
        assert frames[-1]["data"]["status"] == "done" and frames[-1]["data"]["report"]
        nxt = _turn(c)
        assert _stages(nxt)["label"] == "skipped:labeler_unavailable" and "label" not in _started(nxt) and nxt[-1]["data"]["status"] == "done"
        assert stub.labelled == 0


def test_chaos_5_a_labeller_stopped_while_a_turn_runs_skips_the_label_stage_and_the_turn_ends_done(tmp_path, labellers):
    """Chaos item 5, the labeller stopped mid-run. A live server with a labeller that works: one turn labels (the stage runs). The next turn is
    started, and the labeller is stopped once its report has begun to stream. Expected: the label stage of that turn is skipped with
    `labeler_unavailable`, and the turn ends `done` with its whole report."""
    stub = labellers()
    base, stop = start_live_server(create_app(engine="tiny", home=str(tmp_path), labeler_url=stub.url, tiny_step_delay_s=0.02))
    try:
        sid = httpx.post(base + "/v1/sessions", json={}).json()["id"]
        url = base + "/v1/sessions/{}/messages".format(sid)
        data = {"text": "", "options": json.dumps({"max_new_tokens": 60})}
        with httpx.stream("POST", url, files={"image": ("x.png", png_bytes(), "image/png")}, data=data, timeout=60) as r:
            first = list(iter_sse(r.iter_text()))
        assert _stages(first)["label"] == "ran" and first[-1]["data"]["status"] == "done" and stub.labelled == 1
        frames = []
        with httpx.stream("POST", url, files={"image": ("x.png", png_bytes(), "image/png")}, data=data, timeout=60) as r:
            for frame in iter_sse(r.iter_text()):
                frames.append(frame)
                if [f["event"] for f in frames].count("content_block_delta") == 1:
                    stub.stop()   # the labeller's job ends while the report is being decoded
        assert _stages(frames)["label"] == "skipped:labeler_unavailable"
        assert frames[-1]["data"]["status"] == "done" and frames[-1]["data"]["report"]
        assert [f["event"] for f in frames].count("content_block_delta") == [f["event"] for f in first].count("content_block_delta"), \
            "the decoding ran to its end: as many snapshots as the turn before it"
    finally:
        stop()


# ---- 6: the gallery is missing ------------------------------------------------------------------------------------------------------

def test_chaos_6_a_server_with_no_gallery_skips_the_retrieve_stage_and_the_turn_ends_done(tmp_path, capsys):
    """Chaos item 6, the gallery missing: a gallery directory that is not there (nothing was built), which create_app tolerates. Expected: the
    server says why (a count-free line: its manifest is missing), runs without retrieval (/v1/models says so), the retrieve stage is skipped with
    `gallery_unavailable` before it starts, the turn ends `done`; the routes that need the gallery answer 503 (the picker's list, /v1/retrieve),
    and a turn asking for a test-split study is a 422."""
    with TestClient(create_app(engine="tiny", home=str(tmp_path / "home"), gallery_dir=str(tmp_path / "no_such_gallery"))) as c:
        assert "[server] gallery unavailable: manifest.json is missing" in capsys.readouterr().out
        assert c.get("/v1/models").json()["features"]["retrieval"] is False
        frames = _turn(c)
        assert _stages(frames)["retrieve"] == "skipped:gallery_unavailable" and "retrieve" not in _started(frames)
        assert _stages(frames)["generate"] == "ran" and frames[-1]["data"]["status"] == "done" and frames[-1]["data"]["report"]
        assert c.get("/v1/test-studies").status_code == 503
        assert c.post("/v1/retrieve", files={"image": ("x.png", png_bytes(), "image/png")}).status_code == 503
        sid = c.post("/v1/sessions", json={}).json()["id"]
        assert _post(c, sid, {"max_new_tokens": 16, "test_row": 0}, image=False).status_code == 422


def test_chaos_6_the_serving_command_line_refuses_a_gallery_that_will_not_open_and_with_allow_no_gallery_serves_without_retrieval(tmp_path, capsys):
    """Chaos item 6, the serving command line (`python -m app.server`) with --gallery naming a directory that is not there. Expected: without
    --allow-no-gallery the start is refused (exit 1, one ERROR line, and no endpoint file); with it a real server process serves, says so in its
    job log (`[server] gallery unavailable: ...`, `serving: retrieval=off`), and its turn skips the retrieve stage and ends `done`."""
    missing = str(tmp_path / "no_such_gallery")
    base_args = ["--engine", "tiny", "--mode", "private", "--home", str(tmp_path / "refused_home"), "--port", "0", "--gallery", missing]
    assert cli.main(base_args) == 1
    out = capsys.readouterr().out
    assert "ERROR the retrieval gallery could not be opened (see the [server] gallery line)" in out and "serving:" not in out

    endpoint = tmp_path / "endpoint"
    log = tmp_path / "cli_stdout.txt"
    argv = [sys.executable, "-m", "app.server", "--engine", "tiny", "--mode", "private", "--home", str(tmp_path / "home"), "--port", "0",
            "--endpoint-file", str(endpoint), "--gallery", missing, "--allow-no-gallery"]
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
    env.pop("CHAT_HOME", None)
    with open(str(log), "wb") as out_file, open(str(tmp_path / "cli_stderr.txt"), "wb") as err_file:
        proc = subprocess.Popen(argv, cwd=str(REPO_ROOT), env=env, stdin=subprocess.DEVNULL, stdout=out_file, stderr=err_file, start_new_session=True)
        try:
            wait_until(lambda: endpoint.exists() or proc.poll() is not None, timeout=120)
            assert proc.poll() is None, "it ended before it served:\n" + log.read_text()
            base = "http://127.0.0.1:{}".format(json.loads(endpoint.read_text())["port"])
            assert httpx.get(base + "/v1/models").json()["features"]["retrieval"] is False
            sid = httpx.post(base + "/v1/sessions", json={}).json()["id"]
            r = httpx.post(base + "/v1/sessions/{}/messages".format(sid), files={"image": ("x.png", png_bytes(), "image/png")},
                           data={"text": "", "options": json.dumps({"max_new_tokens": 16})}, timeout=60)
            frames = list(iter_sse([r.text]))
            assert _stages(frames)["retrieve"] == "skipped:gallery_unavailable" and frames[-1]["data"]["status"] == "done"
            proc.send_signal(signal.SIGTERM)
            assert proc.wait(30) == 0
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
    lines = log.read_text().splitlines()
    assert "[server] gallery unavailable: manifest.json is missing" in lines
    assert any(l.startswith("[server] serving:") and "retrieval=off" in l for l in lines)


def test_chaos_6_a_gallery_directory_that_disappears_after_the_start_cannot_skip_retrieval_because_the_gallery_is_in_memory(tmp_path, caplog):
    """Chaos item 6, a gallery directory that disappears after the start. The code cannot meet that with a skipped stage: Gallery.open copies
    every array into memory and reads the report texts and the tables then, and a query touches no file, so the retrieve stage keeps running on
    what was loaded. Expected: with the directory gone, a turn's retrieve stage still runs (not skipped) and the turn ends `done`. What the
    directory held beyond that fails by name, with a fixed sentence and no path: the neighbours' pictures (GET /v1/gallery/images/<row>) are a 500
    `Could not read the gallery image.`, and a turn that asks for a test-split study (its image is read from there) is a 500 `Could not read the
    test-split image.`; the server stays up, and a turn on an upload goes on to `done`."""
    home = tmp_path / "home"
    with TestClient(create_app(engine="tiny", home=str(home), tiny_gallery=True)) as c:
        assert c.get("/v1/models").json()["features"]["retrieval"] is True
        sid = c.post("/v1/sessions", json={}).json()["id"]
        before = _turn(c, sid=sid)
        assert _stages(before)["retrieve"] == "ran"
        neighbours = [f for f in before if f["event"] == "stage_end" and f["data"]["stage"] == "retrieve"][0]["data"]["detail"]["image_neighbors"]
        image_url = neighbours[0]["image_url"]
        assert c.get(image_url).status_code == 200
        shutil.rmtree(str(home / "gallery"))
        with caplog.at_level(logging.WARNING):
            after = _turn(c, sid=sid)
            assert _stages(after)["retrieve"] == "ran" and "retrieve" in _started(after) and after[-1]["data"]["status"] == "done"
            retrieved = [f for f in after if f["event"] == "stage_end" and f["data"]["stage"] == "retrieve"][0]["data"]["detail"]
            assert retrieved["image_neighbors"] == neighbours and retrieved["report_matches"], "the same answer, from memory"
            picture = c.get(image_url)
            assert picture.status_code == 500 and picture.json()["error"]["message"] == GALLERY_IMAGE_MSG
            study = _post(c, sid, {"max_new_tokens": 16, "test_row": 0}, image=False)
            assert study.status_code == 500 and study.json()["error"]["message"] == TEST_IMAGE_MSG
        assert str(tmp_path) not in picture.text + study.text + caplog.text, "no path in an answer or in the log"
        assert _turn(c, sid=sid)[-1]["data"]["status"] == "done"


# ---- 7: a disk-quota error while an upload is written -------------------------------------------------------------------------------

@pytest.mark.parametrize("where", ["first_write", "last_write", "rename"])
def test_chaos_7_a_disk_quota_error_in_the_upload_write_is_a_readable_error_without_partial_files_and_the_next_turn_works(
        tmp_path, monkeypatch, caplog, where):
    """Chaos item 7, ENOSPC in the store's upload write, at three places: the first of the upload's three files (nothing is written yet), the
    third (two are), and the rename that puts the finished temporary directory in place (all three are). Each raises OSError(ENOSPC, ..., <a path
    under the home>). Expected: the turn is refused with the error envelope (the body an `error` event carries: type `internal_error`, the message
    `Could not store the image.`), which names no path and no errno text, and neither does the log (the class only); nothing is left under
    uploads/, not a temporary directory and not the session's own; no message is recorded for it; its slot is released (the queue holds one
    turn, and the next one is accepted); and with the disk back, the same upload is stored whole and the turn ends `done`."""
    real_write, real_replace, made = Path.write_bytes, os.replace, []

    def no_space(path):
        return OSError(errno.ENOSPC, "No space left on device", str(path))   # the path is under the home: it must not travel

    def write_bytes(self, data):
        made.append(self.name)
        if self.name == {"first_write": "original.png", "last_write": "model_input.png"}.get(where):
            raise no_space(self)
        return real_write(self, data)

    def replace(src, dst, *args, **kwargs):
        made.append("rename")
        raise no_space(src)

    home = tmp_path / "home"
    with TestClient(create_app(engine="tiny", home=str(home), queue_cap=1)) as c:
        sid = c.post("/v1/sessions", json={}).json()["id"]
        monkeypatch.setattr(Path, "write_bytes", write_bytes)
        if where == "rename":
            monkeypatch.setattr(os, "replace", replace)
        with caplog.at_level(logging.WARNING, logger="app.server"):
            r = _post(c, sid)
        assert r.status_code == 500
        assert r.json() == {"type": "error", "error": {"type": "internal_error", "message": STORE_FAILED_MSG}}
        assert made == {"first_write": ["original.png"], "last_write": ["original.png", "thumb.jpg", "model_input.png"],
                        "rename": ["original.png", "thumb.jpg", "model_input.png", "rename"]}[where], "the write failed where the test said"
        for text in (r.text, caplog.text):
            assert str(tmp_path) not in text and "No space" not in text and "uploads" not in text and "model_input" not in text
        assert [rec.getMessage() for rec in caplog.records if rec.name == "app.server"] == [
            "could not store an upload in session {}: OSError".format(sid)]
        assert list((home / "uploads").rglob("*")) == [], "no partial file, no temporary directory, no empty session directory"
        assert c.get("/v1/sessions/" + sid).json()["messages"] == []
        assert c.get("/healthz").json()["turns_in_flight"] == 0
        monkeypatch.setattr(Path, "write_bytes", real_write)
        monkeypatch.setattr(os, "replace", real_replace)
        frames = _turn(c, sid=sid)   # a queue of one: the slot came back
        assert frames[-1]["data"]["status"] == "done"
        [copy] = list((home / "uploads" / sid).iterdir())
        assert sorted(p.name for p in copy.iterdir()) == ["model_input.png", "original.png", "thumb.jpg"]
