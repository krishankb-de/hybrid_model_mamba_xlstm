"""CHAT_UI_PLAN.md P7-B: the two serving wrappers (scripts/serve_chat_h100.sh on the CPU, scripts/serve_chat_gpu_h100.sh on a GPU),
rehearsed for real under /bin/bash (3.2 on the Mac, the oldest shell they have to work in) in the temp tree of
tests/wrapper_rehearsal.py that stands in for CLUSTER_REPO.

The wrappers start two long-running children, so a rehearsal here is a process that is started, waited for and then stopped with a
signal, not a run to the end. Two stub interpreters play the children: `python -m app.server` is the API stub written below, and the
CheXbert venv's python is a labeller that answers /healthz as its mode says (ok, slow, sick, down). Both record their arguments and
the environment they were given, print raw output that must never reach the job log, and exit 0 on SIGTERM. The last test swaps both stubs
for the real thing: the real CLI on the tiny engine and the real app.labeler (with a fake f1chexbert), behind the real wrapper.

The static pins (directives, the twin, the line shapes, flags against the CLI's own) are in tests/test_willi_parity.py. Synthetic data
only (R7); every port is a free one.
"""
import json
import os
import re
import signal
import socket
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import httpx
import pytest

from app.labels import CHEXBERT_14
from tests import wrapper_rehearsal as wr
from tests.app_helpers import iter_sse, png_bytes, wait_until
from tests.wrapper_rehearsal import job_lines, safe_line

REPO_ROOT = Path(__file__).resolve().parent.parent
LINE_OK = safe_line("server")
JOB_ID = "1234567"
SYNC_LINE = "=== sync 3f2a9c41d7e86b05a1c4e9d3b7f60285ac9e1d47 clean ==="

# `python` as the wrapper runs it after sourcing the venv: it records the call, then plays the API (a stub) or, in the end-to-end test,
# runs the real CLI with `--engine tiny` in place of `--engine real` and the real repository on the path (the rehearsal tree has no app/).
BIN_PYTHON = """#!/bin/bash
""" + wr.RECORD_CALL + """if [ "$1" = "-m" ] && [ "$2" = "app.server" ]; then
  if [ -f "$STUB_DIR/e2e" ]; then
    args=()
    for a in "$@"; do
      case "$a" in real) a=tiny;; esac
      args+=("$a")
    done
    PYTHONPATH="$REAL_REPO${PYTHONPATH:+:$PYTHONPATH}" exec "$REAL_PYTHON" "${args[@]}"
  fi
  exec "$REAL_PYTHON" "$STUB_DIR/api_stub.py" "$@"
fi
exec "$REAL_PYTHON" "$@"
"""
# The CheXbert venv's python: the labeller stub, or in the end-to-end test the real uvicorn on the real app.labeler.
LABELER_SHIM = """#!/bin/bash
{ echo "@@"; for a in "$@"; do printf '%s\\n' "$a"; done; } >> "$STUB_DIR/labeler.calls"
if [ -f "$STUB_DIR/e2e" ]; then
  PYTHONPATH="$REAL_REPO:$PYTHONPATH" exec "$REAL_PYTHON" "$@"
fi
exec "$REAL_PYTHON" "$STUB_DIR/labeler_stub.py" "$@"
"""
PYTHON3_SHIM = """#!/bin/bash
exec "$REAL_PYTHON" "$@"
"""

# The API stub. Raw output on stderr (it must go to the server log), a ready marker, SIGTERM answered with a normal exit like the real CLI.
API_STUB = '''
import json, os, signal, sys, time

stub = os.environ["STUB_DIR"]


def note(name, text):
    with open(os.path.join(stub, name), "w") as f:
        f.write(text)


def mode_of(name, default):
    try:
        return open(os.path.join(stub, name)).read().strip() or default
    except OSError:
        return default


def say(text):
    # one write per line, as the real CLI does: SLURM's SIGTERM reaches the wrapper and the API together, and both write to the job log
    sys.stdout.write(text + "\\n")
    sys.stdout.flush()


if mode_of("api.mode", "ok") == "exit_now":
    sys.exit(3)     # gone, and reaped, before the wrapper can look for it: no pid file, no output
KEYS = ("PYTHONPATH", "HF_HOME", "HF_HUB_OFFLINE", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "PYTHONUNBUFFERED", "CUDA_VISIBLE_DEVICES")
note("api.pid", str(os.getpid()))
note("api.env", json.dumps(dict({k: os.environ.get(k) for k in KEYS}, cwd=os.path.realpath(os.getcwd()))))
note("api.environ", json.dumps(dict(os.environ)))
say("[server] stub api up")
sys.stderr.write("Findings: SECRET REPORT TEXT study_id=12345678 /sc/home/someone/img.jpg\\n")
sys.stderr.flush()
if mode_of("api.mode", "ok") == "crash":
    time.sleep(0.3)
    sys.exit(3)
stopping = []


def on_term(sig, frame):
    say("[server] stub api: SIGTERM")
    note("api.term", "1")
    stopping.append(sig)


signal.signal(signal.SIGTERM, on_term)
note("api.ready", "1")
while not stopping:
    time.sleep(0.02)
time.sleep(0.1)
say("[server] stub api: stopped")
sys.exit(0)
'''
# The labeller stub: /healthz as its mode says. ok: 200 at once; slow: 503 until labeler.slow_s has passed; sick: 503 for good; down: the
# process ends at once. Raw output on stdout (the wrapper keeps both streams of the labeller in a file).
LABELER_STUB = '''
import http.server, json, os, signal, sys, threading, time

stub = os.environ["STUB_DIR"]


def note(name, text):
    with open(os.path.join(stub, name), "w") as f:
        f.write(text)


def mode_of(name, default):
    try:
        return open(os.path.join(stub, name)).read().strip() or default
    except OSError:
        return default


args = sys.argv[1:]
port = int(args[args.index("--port") + 1])
mode = mode_of("labeler.mode", "ok")
KEYS = ("PYTHONPATH", "HF_HOME", "HF_HUB_OFFLINE", "PYTHONUNBUFFERED", "CUDA_VISIBLE_DEVICES")
note("labeler.pid", str(os.getpid()))
note("labeler.env", json.dumps(dict({k: os.environ.get(k) for k in KEYS}, cwd=os.path.realpath(os.getcwd()))))
note("labeler.environ", json.dumps(dict(os.environ)))
print("Downloading CheXbert weights: SECRET HF progress", flush=True)
if mode == "down":
    sys.exit(1)
started = time.time()
slow_s = float(mode_of("labeler.slow_s", "2"))


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        ok = mode == "ok" or (mode == "slow" and time.time() - started >= slow_s)
        body = json.dumps({"status": "ok" if ok else "unavailable"}).encode()
        self.send_response(200 if ok else 503)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


server = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)


def on_term(sig, frame):
    note("labeler.term", "1")
    threading.Thread(target=server.shutdown, daemon=True).start()


signal.signal(signal.SIGTERM, on_term)
note("labeler.ready", "1")
server.serve_forever()
'''


def pairs(argv: List[str]) -> Dict[str, object]:
    """['--a', '1', '--b=2', '--flag'] -> {'--a': '1', '--b': '2', '--flag': True}"""
    found, i = {}, 0
    while i < len(argv):
        arg = argv[i]
        if arg.startswith("--") and "=" in arg:
            key, value = arg.split("=", 1)
            found[key] = value
        elif i + 1 < len(argv) and not argv[i + 1].startswith("--"):
            found[arg] = argv[i + 1]
            i += 1
        else:
            found[arg] = True
        i += 1
    return found


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


class Running:
    def __init__(self, proc, log):
        self.proc, self.log = proc, log

    def signal(self, sig=signal.SIGTERM, group=False):
        (os.killpg if group else os.kill)(self.proc.pid, sig)

    def wait(self, timeout=60):
        return self.proc.wait(timeout)


def reap_group(run: Running, timeout: float = 15.0) -> None:
    """SIGKILL every process of the run's group, the wrapper's own session, and return once there is none. The wrapper is waited for each time
    round (poll): a dead wrapper nobody has waited for is a zombie, and a zombie leader would keep the group alive for ever. A group whose
    members are all gone, zombies reaped, is what ProcessLookupError says."""
    deadline = time.monotonic() + timeout
    while True:
        run.proc.poll()
        try:
            os.killpg(run.proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            return
        if time.monotonic() > deadline:
            raise RuntimeError("a process of the run would not die")
        time.sleep(0.02)


class ServeBox(wr.JobBox):
    """The CPU wrapper in the rehearsal tree (GpuBox is its twin), with the two stub interpreters, an overlay of each kind, the default
    gallery (its gate equal) and a CHAT_HOME. run() runs a wrapper that ends by itself (a refusal); start() starts one that serves until it
    is told to stop. DEVICE is what the wrapper serves on when nobody says otherwise."""

    WRAPPER = "serve_chat_h100.sh"
    DEVICE = "cpu"
    PYTHON_STUB = BIN_PYTHON

    def __init__(self, root, stamp=wr.STAMP):
        self.runs = []
        super().__init__(root, stamp)

    def populate(self) -> None:
        (self.stubs / "api_stub.py").write_text(API_STUB)
        (self.stubs / "labeler_stub.py").write_text(LABELER_STUB)
        self.shim(self.bin / "python3", PYTHON3_SHIM)
        chexbert = self.repo / ".venv_chexbert" / "bin"
        chexbert.mkdir(parents=True)
        self.shim(chexbert / "python", LABELER_SHIM)
        (self.repo / ".chat_deps").mkdir()
        (self.repo / ".chat_deps_chexbert").mkdir()
        self.gallery = self.chat / "gallery" / "g13d_m3_v1"
        self.gallery.mkdir(parents=True)
        self.set_gate(True)

    @staticmethod
    def shim(path: Path, text: str) -> None:
        path.write_text(text)
        path.chmod(0o755)

    def base_env(self) -> Dict[str, str]:
        env = super().base_env()
        env["REAL_REPO"] = str(REPO_ROOT)
        return env

    def set_gate(self, equal) -> None:
        (self.gallery / "manifest.json").write_text(json.dumps({"build_id": "g13d_m3_v1", "gate_rk": {"equal": equal}}))

    def set_mode(self, name: str, value: str) -> None:
        (self.stubs / name).write_text(value + "\n")

    def start(self, script: str = "", **extra_env) -> Running:
        """Start the wrapper (or another script of the tree's scripts directory, for the cleanup's own test) in a session of its own."""
        env = self.base_env()
        env.update(extra_env)
        env = {k: v for k, v in env.items() if v is not None}
        log = open(str(self.root / "job.log"), "ab")
        proc = subprocess.Popen([wr.BASH, str(self.repo / "scripts" / (script or self.WRAPPER))], cwd=str(self.root), env=env,
                                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        run = Running(proc, log)
        self.runs.append(run)
        return run

    def run_bounded(self, timeout: float = 15, **extra_env) -> SimpleNamespace:
        """A wrapper that has to end by itself within `timeout` s. One that is still running then is killed, with its children, and the test
        fails saying so. -> (returncode, stdout), the shape job_lines() reads."""
        run = self.start(**extra_env)
        try:
            run.proc.wait(timeout)
        except subprocess.TimeoutExpired:
            reap_group(run)
            pytest.fail("the wrapper was still running after {} s: it did not refuse, it started serving\n{}".format(timeout, self.job_text()))
        return SimpleNamespace(returncode=run.proc.returncode, stdout=self.job_text())

    def wait_ready(self, run: Running, marker: str = "api.ready", timeout: float = 60) -> None:
        wait_until(lambda: (self.stubs / marker).exists() or run.proc.poll() is not None, timeout=timeout)
        assert (self.stubs / marker).exists(), "the wrapper ended first:\n" + self.job_text() + self.server_log()

    def job_text(self) -> str:
        path = self.root / "job.log"
        return path.read_text() if path.exists() else ""

    def job_lines(self) -> List[str]:
        return self.job_text().splitlines()

    def server_log(self) -> str:
        path = self.chat / "logs" / "server_{}.log".format(JOB_ID)
        return path.read_text() if path.exists() else ""

    def read_json(self, name: str):
        return json.loads((self.stubs / name).read_text())

    def pid(self, who: str) -> int:
        return int((self.stubs / (who + ".pid")).read_text())

    def api_calls(self) -> List[List[str]]:
        return [c for c in self.calls() if c[:2] == ["-m", "app.server"]]

    def labeler_calls(self) -> List[List[str]]:
        path = self.stubs / "labeler.calls"
        return [rec.splitlines() for rec in path.read_text().split("@@\n") if rec.strip()] if path.exists() else []

    def labeler_port(self) -> str:
        (call,) = self.labeler_calls()
        return call[call.index("--port") + 1]

    def stopped_cleanly(self, run: Running) -> None:
        assert run.wait() == 0, self.job_text() + self.server_log()
        for who in ("api", "labeler"):
            if (self.stubs / (who + ".pid")).exists():
                assert not alive(self.pid(who)), who + " is still running after the wrapper ended"

    def cleanup(self) -> None:
        """Whatever a run left, however it ended. Every process of a run is in the run's own session, so its process group is the wrapper's pid,
        and the group outlives the wrapper: it lives while any member does. It is killed until it is empty, so that a wrapper that ended while a
        child was still starting, or never stopped its children (a RED run of a wrapper change), cannot leave one behind. The pids the stubs
        write down are no help for that: a stub that is still starting has written none."""
        for run in self.runs:
            reap_group(run)
            run.log.close()


class GpuBox(ServeBox):
    WRAPPER = "serve_chat_gpu_h100.sh"
    DEVICE = "cuda"


@pytest.fixture(params=[ServeBox, GpuBox], ids=["cpu", "gpu"])
def box(request, tmp_path):
    """Every rehearsal below runs against both wrappers: the twin differs from the CPU wrapper in a few pinned lines, and a rehearsal is the
    only thing that shows the difference is all there is."""
    b = request.param(tmp_path)
    yield b
    b.cleanup()


def token_file(box: ServeBox, text: str = "tok-3c9e5b-wrapper\n", mode: int = 0o600) -> Path:
    path = box.chat / "app_token"
    path.write_text(text)
    path.chmod(mode)
    return path


def refused(box: ServeBox, *needles: str, **env) -> List[str]:
    """The wrapper ends by itself with exit 1 and ONE ERROR line, having started nothing. A refusal that does not happen starts the job, which
    would serve until it was told to stop: that is a failure within seconds here, not a hang."""
    done = box.run_bounded(**env)
    lines = job_lines(done)
    assert done.returncode == 1, done.stdout
    errors = [l for l in lines if l.startswith("ERROR")]
    assert len(errors) == 1, lines
    for needle in needles:
        assert needle in errors[0], (needle, errors[0])
    assert not [l for l in lines if not LINE_OK.match(l)], lines
    assert box.calls() == [] and box.labeler_calls() == [], "something was started before the refusal"
    return lines


# ---- a start and a stop ----------------------------------------------------------------------------------------------------

def test_a_start_and_a_stop_print_only_safe_lines_and_keep_the_raw_output_in_chat_home(box):
    run = box.start()
    box.wait_ready(run)
    run.signal()
    box.stopped_cleanly(run)
    lines = box.job_lines()
    assert lines[0] == SYNC_LINE, "the provenance comes first"
    assert not [l for l in lines if not LINE_OK.match(l)], lines
    node = [l for l in lines if l.startswith("=== chat server: ")]
    assert len(node) == 1 and re.fullmatch(
        r"=== chat server: node=\S+ mode=private bind=127\.0\.0\.1 device=" + box.DEVICE + r" gallery=g13d_m3_v1 job=" + JOB_ID + r" ===",
        node[0]), node
    assert "=== labeller up ===" in lines and "[server] stub api up" in lines
    assert lines.index("=== signal received: stopping ===") < lines.index("[server] stub api: SIGTERM")
    assert lines[-2:] == ["[server] stub api: stopped", "=== chat server stopped ==="]
    text = box.job_text()
    assert "SECRET" not in text and "12345678" not in text and "/sc/home/someone" not in text
    assert "SECRET REPORT TEXT" in box.server_log(), "the API's stderr is in the server log under CHAT_HOME"
    assert "SECRET HF progress" in (box.chat / "logs" / "labeler_{}.log".format(JOB_ID)).read_text(), "so are both streams of the labeller"


def test_the_api_is_started_with_the_plans_arguments_and_the_labeller_url(box):
    run = box.start()
    box.wait_ready(run)
    run.signal()
    box.stopped_cleanly(run)
    (call,) = box.api_calls()
    assert pairs(call[2:]) == {
        "--engine": "real", "--device": box.DEVICE, "--mode": "private", "--home": str(box.chat), "--host": "127.0.0.1", "--port": "0",
        "--endpoint-file": str(box.chat / "endpoint"), "--threads": "16", "--models": "m3", "--drift-note": "",
        "--gallery": str(box.gallery), "--labeler": "http://127.0.0.1:" + box.labeler_port()}


def test_the_submit_line_levers_reach_the_api(box):
    token_file(box)                                  # a bind off loopback needs one (R6, below)
    run = box.start(DRIFT_NOTE="CPU vs GPU: 0/20 differ", MODELS="m3,13d", BIND="0.0.0.0")
    box.wait_ready(run)
    run.signal()
    box.stopped_cleanly(run)
    opts = pairs(box.api_calls()[0][2:])
    assert opts["--drift-note"] == "CPU vs GPU: 0/20 differ" and opts["--models"] == "m3,13d" and opts["--host"] == "0.0.0.0"
    assert opts["--token-file"] == str(box.chat / "app_token")


def test_the_api_runs_offline_on_its_overlay_and_the_labeller_in_the_chexbert_environment(box):
    """sbatch exports the submitting shell: a stray PYTHONPATH, HF_HOME or HF_HUB_OFFLINE=0 left in it must not decide where either half looks
    for its packages or its weights. The labeller's weights are in the default HF cache, not the app's (D24)."""
    run = box.start(PYTHONPATH="/stray/overlay", HF_HOME="/stray/hf", HF_HUB_OFFLINE="0")
    box.wait_ready(run)
    run.signal()
    box.stopped_cleanly(run)
    repo = os.path.realpath(str(box.repo))
    assert box.read_json("api.env") == {
        "PYTHONPATH": ".chat_deps", "HF_HOME": str(box.scratch / ".hf"), "HF_HUB_OFFLINE": "1", "OMP_NUM_THREADS": "16",
        "MKL_NUM_THREADS": "16", "PYTHONUNBUFFERED": "1", "CUDA_VISIBLE_DEVICES": None, "cwd": repo}   # the API is never hidden from a GPU
    # The GPU twin's labeller is pinned to the CPU (CUDA_VISIBLE_DEVICES= hides every GPU from it): the P5-F gates job validates the labeller
    # on the CPU against the published labels, so the labels a turn shows must come from the device they were validated on. The CPU
    # wrapper has no such pin: nothing there to hide.
    assert box.read_json("labeler.env") == {"PYTHONPATH": ".chat_deps_chexbert", "HF_HOME": None, "HF_HUB_OFFLINE": "0",
                                            "PYTHONUNBUFFERED": "1", "CUDA_VISIBLE_DEVICES": "" if box.DEVICE == "cuda" else None, "cwd": repo}
    (call,) = box.labeler_calls()
    assert call[:5] == ["-m", "uvicorn", "app.labeler:app", "--host", "127.0.0.1"] and call[5] == "--port" and call[6] == box.labeler_port()
    assert int(box.labeler_port()) > 0 and len(call) == 7


def test_the_labellers_hub_mode_is_a_lever(box):
    run = box.start(CHEXBERT_HF_HUB_OFFLINE="1")
    box.wait_ready(run)
    run.signal()
    box.stopped_cleanly(run)
    assert box.read_json("labeler.env")["HF_HUB_OFFLINE"] == "1"


def test_the_published_dumps_are_passed_when_both_are_there_and_a_missing_one_is_said_not_fatal(box):
    run = box.start()
    box.wait_ready(run)
    run.signal()
    box.stopped_cleanly(run)
    assert "--published-model" not in pairs(box.api_calls()[0][2:])
    assert "=== published dumps not found: the published line is skipped ===" in box.job_lines()
    for name in ("report_gen_m3_test_split_s42", "retrieval_floor_test_split"):
        (box.main / "results" / name).mkdir(parents=True)
        (box.main / "results" / name / "hyps.txt").write_text("Findings: SECRET\n")
    (box.repo / "results").symlink_to(box.main / "results", target_is_directory=True)
    for marker in ("api.ready", "api.term"):
        (box.stubs / marker).unlink()
    run = box.start()
    box.wait_ready(run)
    run.signal()
    box.stopped_cleanly(run)
    opts = pairs(box.api_calls()[1][2:])
    assert opts["--published-model"] == "results/report_gen_m3_test_split_s42" and opts["--published-floor"] == "results/retrieval_floor_test_split"
    assert "SECRET" not in box.job_text()


# ---- R6: a token, and never on the command line --------------------------------------------------------------------------------

def test_public_mode_without_a_token_is_refused_before_anything_starts(box):
    refused(box, "needs a token", "R6", MODE="public")


def test_a_bind_off_loopback_without_a_token_is_refused_before_anything_starts(box):
    refused(box, "needs a token", "R6", BIND="0.0.0.0")


def test_an_empty_token_file_is_no_token(box):
    token_file(box, text="")
    refused(box, "needs a token", "R6", MODE="public")


@pytest.mark.parametrize("mode", [0o644, 0o640, 0o660, 0o400])
def test_a_token_file_that_is_not_mode_0600_is_refused_even_on_loopback(box, mode):
    token_file(box, mode=mode)
    lines = refused(box, "mode 0600", "R6")
    assert "tok-3c9e5b" not in "\n".join(lines)


def test_a_mode_that_is_not_private_or_public_is_refused(box):
    refused(box, "MODE must be private or public", MODE="demo")


def test_a_public_start_passes_the_token_file_by_path_and_the_token_reaches_no_log_argument_or_environment(box):
    token_file(box)
    run = box.start(MODE="public")
    box.wait_ready(run)
    run.signal()
    box.stopped_cleanly(run)
    (call,) = box.api_calls()
    opts = pairs(call[2:])
    assert opts["--mode"] == "public" and opts["--token-file"] == str(box.chat / "app_token")
    places = [box.job_text(), box.server_log(), json.dumps(call), (box.stubs / "api.environ").read_text(),
              (box.stubs / "labeler.environ").read_text(), json.dumps(box.labeler_calls())]
    assert not [p for p in places if "tok-3c9e5b-wrapper" in p]


def test_a_token_file_that_is_there_is_used_in_private_mode_too(box):
    token_file(box)
    run = box.start()
    box.wait_ready(run)
    run.signal()
    box.stopped_cleanly(run)
    assert pairs(box.api_calls()[0][2:])["--token-file"] == str(box.chat / "app_token")


# ---- the gallery ---------------------------------------------------------------------------------------------------------------

def test_a_gallery_without_a_manifest_is_refused_before_anything_starts(box):
    (box.gallery / "manifest.json").unlink()
    lines = refused(box, "manifest.json")
    assert str(box.chat) not in "\n".join(lines), "no path in the log"


@pytest.mark.parametrize("text", [json.dumps({"gate_rk": {"equal": False}}), json.dumps({"gate_rk": {}}), json.dumps({}),
                                  json.dumps({"gate_rk": {"equal": "true"}}), "not json at all"])
def test_a_gallery_whose_gate_is_not_equal_is_refused_before_anything_starts(box, text):
    (box.gallery / "manifest.json").write_text(text)
    refused(box, "gate")


def test_a_gallery_is_served_without_labels_when_they_are_pending_and_without_a_gallery_only_when_asked(box):
    other = box.chat / "gallery" / "other_build"
    other.mkdir()
    (other / "manifest.json").write_text(json.dumps({"gate_rk": {"equal": True}, "labels_status": "pending"}))
    run = box.start(GALLERY=str(other))
    box.wait_ready(run)
    run.signal()
    box.stopped_cleanly(run)
    assert pairs(box.api_calls()[0][2:])["--gallery"] == str(other) and "gallery=other_build" in " ".join(box.job_lines())
    for marker in ("api.ready", "api.term"):
        (box.stubs / marker).unlink()
    run = box.start(GALLERY="none")
    box.wait_ready(run)
    run.signal()
    box.stopped_cleanly(run)
    assert "--gallery" not in pairs(box.api_calls()[1][2:])
    lines = box.job_lines()
    assert "=== no gallery: retrieval is off ===" in lines and any("gallery=none" in l for l in lines)


# ---- the labeller: never a reason not to serve -----------------------------------------------------------------------------------

def test_a_labeller_that_never_answers_costs_the_labels_not_the_server(box):
    box.set_mode("labeler.mode", "sick")
    run = box.start(LABELER_WAIT_S="2")
    box.wait_ready(run)
    (call,) = box.api_calls()
    assert pairs(call[2:])["--labeler"] == "none"
    lines = box.job_lines()
    assert "=== labeller unavailable: labels skipped ===" in lines and "=== labeller up ===" not in lines
    wait_until(lambda: not alive(box.pid("labeler")), timeout=10)
    run.signal()
    box.stopped_cleanly(run)
    assert not [l for l in box.job_lines() if not LINE_OK.match(l)]


def test_a_labeller_that_dies_at_once_is_not_waited_for(box):
    box.set_mode("labeler.mode", "down")
    began = time.monotonic()
    run = box.start(LABELER_WAIT_S="120")
    box.wait_ready(run)
    assert time.monotonic() - began < 30, "the bound is 120 s; a dead labeller needs none of it"
    assert pairs(box.api_calls()[0][2:])["--labeler"] == "none"
    assert "=== labeller unavailable: labels skipped ===" in box.job_lines()
    run.signal()
    box.stopped_cleanly(run)


def test_a_slow_labeller_is_waited_for_within_the_bound(box):
    box.set_mode("labeler.mode", "slow")
    box.set_mode("labeler.slow_s", "1.5")
    run = box.start(LABELER_WAIT_S="30")
    box.wait_ready(run)
    assert pairs(box.api_calls()[0][2:])["--labeler"] == "http://127.0.0.1:" + box.labeler_port()
    assert "=== labeller up ===" in box.job_lines()
    run.signal()
    box.stopped_cleanly(run)


def test_a_missing_chexbert_venv_is_a_labeller_that_is_down_and_the_log_stays_clean(box):
    (box.repo / ".venv_chexbert" / "bin" / "python").unlink()
    run = box.start(LABELER_WAIT_S="30")
    box.wait_ready(run)
    assert pairs(box.api_calls()[0][2:])["--labeler"] == "none"
    run.signal()
    box.stopped_cleanly(run)
    lines = box.job_lines()
    assert "=== labeller unavailable: labels skipped ===" in lines and not [l for l in lines if not LINE_OK.match(l)], lines


def test_a_missing_venv_activate_script_is_refused_before_anything_starts(box):
    (box.repo / ".venv" / "bin" / "activate").unlink()
    refused(box, "activate")


# ---- signals and the end of the job -----------------------------------------------------------------------------------------------

def test_a_term_to_every_process_at_once_as_slurm_sends_it_ends_the_job_the_same_way(box):
    run = box.start()
    box.wait_ready(run)
    run.signal(group=True)
    box.stopped_cleanly(run)
    lines = box.job_lines()
    assert not [l for l in lines if not LINE_OK.match(l)], lines
    assert lines[-1] in ("=== chat server stopped ===", "=== chat server ended ===")
    assert (box.stubs / "api.term").exists() and (box.stubs / "labeler.term").exists()


def test_an_interrupt_stops_the_job_as_a_term_does(box):
    run = box.start()
    box.wait_ready(run)
    run.signal(signal.SIGINT)
    box.stopped_cleanly(run)
    assert box.job_lines()[-1] == "=== chat server stopped ===" and (box.stubs / "api.term").exists()


def test_a_term_while_the_labeller_is_waited_for_ends_the_job_before_the_api_starts(box):
    box.set_mode("labeler.mode", "sick")
    run = box.start(LABELER_WAIT_S="120")
    box.wait_ready(run, marker="labeler.ready")
    run.signal()
    box.stopped_cleanly(run)
    assert box.api_calls() == [], "no API was started after the signal"
    lines = box.job_lines()
    assert "=== signal received: stopping ===" in lines and lines[-1] == "=== chat server stopped ==="
    assert not [l for l in lines if not LINE_OK.match(l)], lines


def test_a_term_that_comes_while_the_api_is_being_started_is_not_lost(box):
    """The wrapper is inside the venv's activate script (a slow one here) when the signal comes: the trap sets the flag and forwards it to
    the labeller, and there is no API yet to forward it to. The API must then not be started at all, since nothing would ever stop it."""
    (box.repo / ".venv" / "bin" / "activate").write_text("sleep 2\n")
    run = box.start()
    wait_until(lambda: "=== labeller up ===" in box.job_text(), timeout=30)
    run.signal()
    box.stopped_cleanly(run)
    assert box.api_calls() == [], "an API was started after the signal"
    lines = box.job_lines()
    assert "=== signal received: stopping ===" in lines and lines[-1] == "=== chat server stopped ==="
    assert not [l for l in lines if not LINE_OK.match(l)], lines


def test_an_api_that_dies_on_its_own_stops_the_labeller_and_fails_the_job_with_its_exit_code(box):
    box.set_mode("api.mode", "crash")
    done = box.run()
    lines = job_lines(done)
    assert done.returncode == 3, done.stdout
    errors = [l for l in lines if l.startswith("ERROR")]
    assert len(errors) == 1 and "code 3" in errors[0] and "SECRET" not in done.stdout
    assert not [l for l in lines if not LINE_OK.match(l)], lines
    assert not alive(box.pid("labeler")), "the labeller does not outlive the API"


def test_an_api_that_is_killed_hard_is_one_error_line_and_bash_adds_nothing(box):
    run = box.start()
    box.wait_ready(run)
    os.kill(box.pid("api"), signal.SIGKILL)
    assert run.wait() == 137, box.job_text()
    lines = box.job_lines()
    assert not [l for l in lines if not LINE_OK.match(l)], lines
    errors = [l for l in lines if l.startswith("ERROR")]
    assert len(errors) == 1 and "code 137" in errors[0]
    assert not alive(box.pid("labeler"))


def test_a_script_error_ends_in_one_error_line_and_leaves_no_child_behind(box):
    """`set -e` stops the script wherever a command fails; the job log then has the exit code, and nothing is left running."""
    (box.repo / ".venv" / "bin" / "activate").write_text("false\n")
    done = box.run()
    lines = job_lines(done)
    assert done.returncode == 1
    errors = [l for l in lines if l.startswith("ERROR")]
    assert len(errors) == 1 and "exit code 1" in errors[0], lines
    assert not [l for l in lines if not LINE_OK.match(l)], lines
    assert not alive(box.pid("labeler"))


# ---- the device and the models are checked before anything starts -----------------------------------------------------------------------

@pytest.mark.parametrize("value", ["tpu", "CUDA", "cuda:0", "cpu cuda", "cpu;echo SECRET-device", "SECRET-device"])
def test_a_device_that_is_not_cpu_or_cuda_is_refused_before_anything_starts_and_is_not_echoed(box, value):
    lines = refused(box, "DEVICE must be cpu or cuda", DEVICE=value)
    assert "SECRET" not in "\n".join(lines) and value not in "\n".join(lines)


@pytest.mark.parametrize("value", ["m3,13D", "m3 13d", "m3;13d", "SECRET-models", "a" * 33, "m3\n13d"])
def test_models_outside_lowercase_digits_and_commas_are_refused_before_anything_starts_and_are_not_echoed(box, value):
    lines = refused(box, "MODELS must be", MODELS=value)
    assert "SECRET" not in "\n".join(lines) and value not in "\n".join(lines)


def test_the_edges_of_what_the_device_and_models_checks_allow_are_served(box):
    """cpu and cuda are both fine on either wrapper (the GPU twin only defaults to cuda); 32 characters of [a-z0-9,] pass the wrapper, and
    the CLI is what knows the model names."""
    other = "cpu" if box.DEVICE == "cuda" else "cuda"
    run = box.start(DEVICE=other, MODELS="a" * 32)
    box.wait_ready(run)
    run.signal()
    box.stopped_cleanly(run)
    opts = pairs(box.api_calls()[0][2:])
    assert opts["--device"] == other and opts["--models"] == "a" * 32
    assert "device={} ".format(other) in " ".join(box.job_lines())


def test_a_model_name_only_the_cli_knows_is_the_clis_refusal_and_becomes_the_jobs_failure_with_its_status(box):
    """The wrapper judges the shape of MODELS (a-z, 0-9, commas), the CLI the names: `m4` passes the one and the real CLI refuses it, in one
    line that does not quote the value, and the job fails with the CLI's own exit status, 2, whatever the labeller was doing."""
    real_processes(box)
    done = box.run_bounded(timeout=120, MODELS="m4", GALLERY="none")
    lines = job_lines(done)
    assert done.returncode == 2, done.stdout
    errors = [l for l in lines if l.startswith("ERROR")]
    assert errors == ["ERROR --models: unknown model (the models are 13d, m3)", "ERROR the API exited with code 2 (see the server log under CHAT_HOME)"], lines
    assert not [l for l in lines if not LINE_OK.match(l)], lines
    assert "m4" not in "\n".join(l for l in lines if "unknown model" in l), "the refused value is not quoted"


# ---- the rehearsal's own cleanup -----------------------------------------------------------------------------------------------------

def processes_naming(path: Path) -> List[int]:
    """Pids of the processes whose command line names `path`: a stub that outlived its test. pgrep leaves itself out."""
    done = subprocess.run(["pgrep", "-f", str(path)], stdout=subprocess.PIPE, universal_newlines=True)
    return [int(pid) for pid in done.stdout.split()]


def test_the_cleanup_reaps_what_a_wrapper_left_behind_even_a_child_that_is_still_starting(tmp_path):
    """The leak this pins: a RED run of a wrapper change left two labeller stubs, parent 1, holding loopback ports. The wrapper under test had
    ended while its labeller was still starting and had never stopped it, and the cleanup knew only the pids the stubs write down, which a stub
    that is still starting has not written, and killed the run's group only while the wrapper itself was alive. The cleanup now kills the run's
    whole process group, which outlives the wrapper, until it is empty. Here the wrapper ends at once with a `sleep` (that writes no pid file at
    all) and a labeller stub still starting as its orphans."""
    box = ServeBox(tmp_path)
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    (box.repo / "scripts" / "leaky.sh").write_text(
        '#!/bin/bash\nsleep 300 &\necho $! > "$STUB_DIR/orphan.pid"\n'
        '"$REAL_PYTHON" "$STUB_DIR/labeler_stub.py" -m uvicorn app.labeler:app --host 127.0.0.1 --port "$LEAKY_PORT" &\nexit 1\n')
    try:
        run = box.start(script="leaky.sh", LEAKY_PORT=str(port))
        assert run.wait(10) == 1, "the wrapper ends at once, its children just forked"
        orphan = int((box.stubs / "orphan.pid").read_text())
        assert alive(orphan), "the setup: the wrapper left an orphan"
        box.cleanup()
        assert not alive(orphan), "a sleep the wrapper left outlived the cleanup"
        assert processes_naming(box.stubs) == [], "a labeller stub that was still starting outlived the cleanup"
    finally:                                         # a failing run of this test must not leave what it found
        for pid in processes_naming(box.stubs):
            os.kill(pid, signal.SIGKILL)
        if (box.stubs / "orphan.pid").exists() and alive(int((box.stubs / "orphan.pid").read_text())):
            os.kill(int((box.stubs / "orphan.pid").read_text()), signal.SIGKILL)


# ---- the API's exit status is never lost -----------------------------------------------------------------------------------------------

def test_an_api_that_is_gone_before_the_wrapper_first_looks_for_it_still_gives_its_exit_status(box):
    """The API exits 3 at once. A DEBUG trap (BASH_ENV) holds the wrapper for a second at the first command after the API is launched, so
    that the API is gone, and reaped, before the wrapper ever tests whether it is alive. Its status must still be the job's: a status that
    was lost read as a clean end, `=== chat server ended ===` and exit 0, for an API that never served."""
    box.set_mode("api.mode", "exit_now")
    hold = box.root / "hold_after_launch.sh"
    # An `if`, not an && list: the trap's own status is the status of the command it ran in front of, under set -e, and a list that is
    # false returns 1. This one returns 0 when it does nothing.
    hold.write_text('trap \'if [ -n "${API_PID:-}" ] && [ ! -e "${STUB_DIR}/held" ]; then : > "${STUB_DIR}/held"; sleep 1; fi\' DEBUG\n')
    done = box.run(BASH_ENV=str(hold))
    lines = job_lines(done)
    assert (box.stubs / "held").exists(), "the wrapper was never held after the launch: this proves nothing"
    assert done.returncode == 3, done.stdout
    errors = [l for l in lines if l.startswith("ERROR")]
    assert len(errors) == 1 and "ERROR the API exited with code 3" in errors[0], lines
    assert "=== chat server ended ===" not in lines and "=== chat server stopped ===" not in lines
    assert not [l for l in lines if not LINE_OK.match(l)], lines
    assert not alive(box.pid("labeler")), "the labeller does not outlive the API"


# ---- the real CLI and the real labeller app behind the real wrapper ------------------------------------------------------------------

def real_processes(box: ServeBox) -> None:
    """Swap both stubs for the real thing: `python -m app.server` runs for real on the tiny engine (the wrapper's `--engine real` is read as
    tiny), and app.labeler runs under the real uvicorn with a fake f1chexbert, whose every label is `No Finding`."""
    (box.stubs / "e2e").write_text("1\n")
    fake = box.repo / ".chat_deps_chexbert" / "f1chexbert"
    fake.mkdir()
    (fake / "__init__.py").write_text("class F1CheXbert:\n    target_names = {!r}\n\n    def get_label(self, text):\n        return [0] * 13 + [1]\n"
                                      .format(CHEXBERT_14))


def test_the_real_cli_under_the_real_wrapper_in_public_mode_wants_the_bearer_token_and_keeps_it_out_of_every_log(box):
    """R6 across the wrapper-to-CLI seam, which each side's own tests only meet at argv: the token file the wrapper checked is the one the
    CLI reads, the app then answers 401 without the token and 200 with it, and the token is in no log, whichever file."""
    real_processes(box)
    token_file(box, "tok-9b2d1f-public\n")
    run = box.start(GALLERY="none", MODE="public")
    endpoint = box.chat / "endpoint"
    wait_until(lambda: endpoint.exists() or run.proc.poll() is not None, timeout=180)
    assert run.proc.poll() is None, box.job_text() + box.server_log()
    data = json.loads(endpoint.read_text())
    base = "http://127.0.0.1:{}".format(data["port"])
    assert data["mode"] == "public" and httpx.get(base + "/healthz").json()["mode"] == "public"
    assert httpx.get(base + "/v1/models").status_code == 401
    ok = httpx.get(base + "/v1/models", headers={"Authorization": "Bearer tok-9b2d1f-public"})
    assert ok.status_code == 200 and ok.json()["mode"] == "public"
    run.signal()
    assert run.wait(60) == 0, box.job_text() + box.server_log()
    assert not endpoint.exists()
    logs = [box.job_text(), box.server_log(), (box.chat / "logs" / "labeler_{}.log".format(JOB_ID)).read_text()]
    assert not [text for text in logs if "tok-9b2d1f-public" in text]
    assert not [l for l in box.job_lines() if not LINE_OK.match(l)], box.job_lines()


def test_the_real_cli_and_the_real_labeller_app_under_the_real_wrapper_serve_a_turn_with_labels_and_stop_cleanly(box):
    """Every flag the wrapper passes is one the real parser takes, the labeller URL it chose is the one the app calls, the endpoint file is
    where the tunnel reads it, and the job log is nothing but its four shapes."""
    real_processes(box)
    run = box.start(GALLERY="none")
    endpoint = box.chat / "endpoint"
    wait_until(lambda: endpoint.exists() or run.proc.poll() is not None, timeout=180)
    assert run.proc.poll() is None, box.job_text() + box.server_log()
    data = json.loads(endpoint.read_text())
    assert oct(endpoint.stat().st_mode & 0o777) == "0o600" and data["mode"] == "private"
    base = "http://127.0.0.1:{}".format(data["port"])
    assert httpx.get(base + "/healthz").json()["status"] == "ok"
    assert httpx.get(base + "/v1/models").json()["features"] == {"retrieval": False, "labels": True}
    sid = httpx.post(base + "/v1/sessions", json={}).json()["id"]
    with httpx.stream("POST", base + "/v1/sessions/{}/messages".format(sid), files={"image": ("x.png", png_bytes(), "image/png")},
                      data={"text": "", "options": json.dumps({"max_new_tokens": 16})}, timeout=60) as response:
        frames = list(iter_sse(response.iter_text()))
    label = [f["data"] for f in frames if f["event"] == "stage_end" and f["data"].get("stage") == "label"]
    assert len(label) == 1 and label[0]["detail"]["chexbert_14"]["No Finding"] == 1, "the labeller the wrapper started answered the turn"
    assert frames[-1]["data"]["status"] == "done"
    run.signal()
    assert run.wait(60) == 0, box.job_text() + box.server_log()
    assert not endpoint.exists()
    lines = box.job_lines()
    text = box.job_text()
    assert lines[0] == SYNC_LINE and lines[-1] == "=== chat server stopped ===", lines
    assert not [l for l in lines if not LINE_OK.match(l)], lines
    assert str(data["port"]) not in text and "Uvicorn" not in text and str(box.chat) not in text, "no endpoint, no uvicorn log, no path"
    assert "Uvicorn running" in box.server_log()
    assert any(l.startswith("[server] serving: ") and "labels=on" in l for l in lines)
