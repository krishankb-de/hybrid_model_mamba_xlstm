"""CHAT_UI_PLAN.md P7-C: the two laptop-side scripts, app/tunnel/tunnel.sh and app/tunnel/public_demo.sh, rehearsed for real under /bin/bash
(3.2 on the Mac, the oldest shell they have to work in) with a fake `ssh`, `curl` and `cloudflared` first on the PATH.

Nothing here touches a network, a real ssh or a real cloudflared. The fakes record their argv (one `@@` line, then one argument per line,
the format of tests/wrapper_rehearsal.py) and play the cluster from files in a stub directory. LOGIN is fake-login.invalid, a name that
never resolves, and the PATH is the fakes' directory then /usr/bin:/bin, so a fake that was missing could not reach a host either. LOCAL_PORT
is always a free port picked here, never 8000, where a developer's own server may be listening: the harness refuses to start a script
without one, and nothing here binds, probes or connects to 8000.

The endpoint file the server writes is ONE JSON line {host, port, mode, pid, started_at} (app/cli.py write_endpoint), not host:port. It
comes from a shared cluster, so its content is untrusted: the tests feed the tunnel hostile content and check what reaches ssh, and that
the content is never echoed. A looping script is bounded twice: MAX_ATTEMPTS ends its loop, and the harness kills the whole process
group when a rehearsal does not end. Every rehearsal runs in a session of its own, and no rehearsal leaves a process behind.

The static pins (what is on the jump path only, BatchMode on the read, the mode check before cloudflared, no secrets and no cluster
commands, bash 3.2 syntax) are in tests/test_willi_parity.py. Synthetic data only (R7).
"""
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pytest

from tests.wrapper_rehearsal import BASH, RECORD_CALL

REPO_ROOT = Path(__file__).resolve().parent.parent
TUNNEL = REPO_ROOT / "app" / "tunnel" / "tunnel.sh"
PUBLIC_DEMO = REPO_ROOT / "app" / "tunnel" / "public_demo.sh"
LOGIN = "fake-login.invalid"
CLUSTER_USER = "krishankumar.bhushan"          # the script's default
STAMPED = r"^\d\d:\d\d:\d\d "                  # one timestamped line per attempt
READ = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10", LOGIN, "cat", "chat_sessions/endpoint"]
NO_ENDPOINT = r"no endpoint yet; is the job running\? \(ssh fake-login\.invalid squeue --me\)"
UNREADABLE = "endpoint unreadable; retrying"
BAD_PORT = "LOCAL_PORT must be a number from 1 to 65535"


# ---- the fakes ---------------------------------------------------------------------------------------------------------------------

def record(tool: str) -> str:
    """The first thing a fake does: note the call in $STUB_DIR/<tool>.calls (one `@@` line, then one argument per line)."""
    return RECORD_CALL.replace("python.calls", tool + ".calls")


# `ssh`. The forward (`-N`) and the read (`... cat chat_sessions/endpoint`) are told apart by -N. The n-th call of a kind is played by the
# file KIND.<n> in $STUB_DIR, else by the file KIND: a forward's file says how it ends (a status, `sleep` to stay up until it is stopped,
# or `bind` to take LOCAL_PORT as a dev server would, and then end), a read's file is what it prints, with <file>.rc beside it for its status.
FAKE_SSH = "#!/bin/bash\n" + record("ssh") + r"""forward=0
for a in "$@"; do [ "$a" = "-N" ] && forward=1; done
nth() {
  local n=0
  [ -f "$STUB_DIR/$1.count" ] && n=$(cat "$STUB_DIR/$1.count")
  n=$((n + 1))
  echo "$n" > "$STUB_DIR/$1.count"
  if [ -f "$STUB_DIR/$1.$n" ]; then echo "$STUB_DIR/$1.$n"; elif [ -f "$STUB_DIR/$1" ]; then echo "$STUB_DIR/$1"; fi
}
if [ "$forward" = 1 ]; then
  file=$(nth forward)
  how=255
  [ -n "$file" ] && how=$(cat "$file")
  case "$how" in
    sleep) echo $$ > "$STUB_DIR/forward.pid"; exec sleep 60 ;;
    bind)  python3 "$STUB_DIR/binder.py" "$LOCAL_PORT" "$STUB_DIR/binder.ready" &
           for _ in $(seq 1 50); do [ -f "$STUB_DIR/binder.ready" ] && break; sleep 0.1; done
           exit 255 ;;
    *)     exit "$how" ;;
  esac
fi
file=$(nth read)
rc=0
if [ -n "$file" ]; then cat "$file"; else echo "cat: chat_sessions/endpoint: No such file or directory" >&2; rc=1; fi
[ -n "$file" ] && [ -f "$file.rc" ] && rc=$(cat "$file.rc")
exit "$rc"
"""
# What a dev server does with a port: listens on it, says so, and keeps it until it is killed.
BINDER = """import socket, sys, time
s = socket.socket()
s.bind(("127.0.0.1", int(sys.argv[1])))
s.listen(1)
open(sys.argv[2], "w").close()
time.sleep(30)
"""
# `curl`: answers with $STUB_DIR/healthz (and its status from healthz.rc), or, with no such file, as curl does for a refused connection.
FAKE_CURL = "#!/bin/bash\n" + record("curl") + r"""rc=$(cat "$STUB_DIR/healthz.rc" 2>/dev/null)
if [ -f "$STUB_DIR/healthz" ]; then cat "$STUB_DIR/healthz"; exit "${rc:-0}"; fi
exit "${rc:-7}"
"""
# `cloudflared`: records its argv and its pid, prints a line, then ends with the status in cloudflared.rc (default 0), or stays up until it
# is stopped when cloudflared.stay exists.
FAKE_CLOUDFLARED = "#!/bin/bash\n" + record("cloudflared") + r"""echo $$ > "$STUB_DIR/cloudflared.pid"
echo "fake cloudflared is up"
if [ -f "$STUB_DIR/cloudflared.stay" ]; then exec sleep 60; fi
exit "$(cat "$STUB_DIR/cloudflared.rc" 2>/dev/null || echo 0)"
"""
PYTHON3_SHIM = """#!/bin/bash
exec "$REAL_PYTHON" "$@"
"""


# ---- the harness -------------------------------------------------------------------------------------------------------------------

def free_port() -> int:
    """A port nobody listens on right now: the OS picks it and it is closed again at once. Never 8000."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    assert port != 8000
    return port


def alive(pid: int) -> bool:
    """Is there a process with this pid that can still run? macOS answers EPERM, not ESRCH, for a zombie (and for a pid some other user
    has been given since): neither is a process of ours that is running."""
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def wait_for(predicate: Callable[[], object], timeout: float = 10.0) -> object:
    """Poll until predicate() is truthy: a bounded wait."""
    deadline = time.monotonic() + timeout
    while True:
        value = predicate()
        if value:
            return value
        if time.monotonic() >= deadline:
            raise AssertionError("condition not met within {} s".format(timeout))
        time.sleep(0.02)


def spawn(args: List[str], **kwargs) -> subprocess.Popen:
    """Popen, with SIGINT at its default for the child. A pytest that was itself started as a background job has SIGINT ignored, a child
    inherits that, and bash cannot trap a signal that was ignored when it started. (Set around the call, not run in the child between fork
    and exec: that is not safe in a process with threads, which this one is once torch has been imported.)"""
    previous = signal.signal(signal.SIGINT, signal.SIG_DFL)
    try:
        return subprocess.Popen(args, **kwargs)
    finally:
        signal.signal(signal.SIGINT, previous)


class Run:
    """One script, started in a session of its own (so its pid is its group's id and one signal can end every process of it)."""

    def __init__(self, proc: subprocess.Popen, log_path: Path, log):
        self.proc, self.log_path, self.log = proc, log_path, log

    @property
    def out(self) -> str:
        return self.log_path.read_text()

    @property
    def lines(self) -> List[str]:
        return self.out.splitlines()

    @property
    def rc(self) -> int:
        return self.proc.returncode

    def signal(self, sig: int) -> None:
        os.kill(self.proc.pid, sig)

    def wait(self, timeout: float = 30.0) -> int:
        try:
            return self.proc.wait(timeout)
        except subprocess.TimeoutExpired:
            self.kill_group()
            raise AssertionError("the script did not end within {} s; its output:\n{}".format(timeout, self.out))

    def group_alive(self) -> bool:
        """Is any process of the run's group still running? (EPERM is what macOS says when only zombies are left: see alive().)"""
        try:
            os.killpg(self.proc.pid, 0)
        except (ProcessLookupError, PermissionError):
            return False
        return True

    def kill_group(self) -> None:
        try:
            os.killpg(self.proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        self.proc.wait()


class Box:
    """A temp tree in which a script runs: fakes first on the PATH, a HOME of its own, a stub directory the fakes read, a free LOCAL_PORT."""

    def __init__(self, root: Path):
        self.root = root
        self.bin, self.stubs, self.home = root / "bin", root / "stubs", root / "home"
        for directory in (self.bin, self.stubs, self.home):
            directory.mkdir()
        for name, text in (("ssh", FAKE_SSH), ("curl", FAKE_CURL), ("python3", PYTHON3_SHIM)):
            self.install(name, text)
        (self.stubs / "binder.py").write_text(BINDER)
        self.port = free_port()
        self.runs: List[Run] = []

    def install(self, name: str, text: str) -> None:
        (self.bin / name).write_text(text)
        (self.bin / name).chmod(0o755)

    def remove(self, name: str) -> None:
        (self.bin / name).unlink()

    def set(self, name: str, text: str) -> None:
        (self.stubs / name).write_text(text)

    @staticmethod
    def endpoint(host: str = "gx01.example", port: int = 43211, mode: str = "private") -> str:
        """The endpoint file as app/cli.py write_endpoint writes it: one JSON line, exactly these keys."""
        return json.dumps({"host": host, "port": port, "mode": mode, "pid": 4242, "started_at": "2026-10-10T08:00:00+00:00"}) + "\n"

    def read(self, text: str, rc: Optional[int] = None, n: Optional[int] = None) -> None:
        """What the n-th `ssh LOGIN cat chat_sessions/endpoint` prints (every one when n is None), and the status it ends with."""
        name = "read" if n is None else "read.{}".format(n)
        self.set(name, text)
        if rc is not None:
            self.set(name + ".rc", str(rc))

    def forward(self, how: str, n: Optional[int] = None) -> None:
        """How the n-th forward ends (every one when n is None): a status, `sleep` (stays up), or `bind` (a dev server takes the port)."""
        self.set("forward" if n is None else "forward.{}".format(n), how)

    def calls(self, tool: str) -> List[List[str]]:
        path = self.stubs / (tool + ".calls")
        if not path.exists():
            return []
        return [rec.splitlines() for rec in path.read_text().split("@@\n") if rec.strip()]

    def env(self, **extra: Optional[str]) -> Dict[str, str]:
        env = {"PATH": "{}:/usr/bin:/bin".format(self.bin), "HOME": str(self.home), "STUB_DIR": str(self.stubs),
               "REAL_PYTHON": sys.executable, "PYTHONDONTWRITEBYTECODE": "1", "LOGIN": LOGIN, "LOCAL_PORT": str(self.port),
               "RETRY_SLEEP": "0", "MAX_ATTEMPTS": "3"}
        env.update(extra)
        return {k: v for k, v in env.items() if v is not None}

    def checked_env(self, **extra_env: Optional[str]) -> Dict[str, str]:
        """env(), after the ways a rehearsal could touch the real world are refused: a script that would fall back to port 8000 or to the
        real cluster alias (an unset or empty variable is a default for the scripts), and a PATH on which a real ssh could come before the
        fakes."""
        env = self.env(**extra_env)
        assert env.get("LOCAL_PORT") not in (None, "", "8000"), "a rehearsal never meets port 8000"
        assert env.get("LOGIN") not in (None, "", "hpi-hpc"), "a rehearsal never meets the real login alias"
        assert env["PATH"] == str(self.bin) or env["PATH"].startswith(str(self.bin) + ":"), "the fakes come first"
        return env

    def start(self, script: Path, **extra_env: Optional[str]) -> Run:
        env = self.checked_env(**extra_env)
        log_path = self.root / "run{}.log".format(len(self.runs))
        log = open(str(log_path), "wb")
        proc = spawn([BASH, str(script)], cwd=str(self.root), env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                     start_new_session=True)
        run = Run(proc, log_path, log)
        self.runs.append(run)
        return run

    def run(self, script: Path, timeout: float = 30.0, **extra_env: Optional[str]) -> Run:
        """A script that ends by itself, waited for (and killed, with its whole group, if it does not end)."""
        run = self.start(script, **extra_env)
        run.wait(timeout)
        return run

    def cleanup(self) -> None:
        for run in self.runs:
            run.kill_group()
            run.log.close()


@pytest.fixture
def box(tmp_path):
    b = Box(tmp_path)
    yield b
    b.cleanup()


# Runs the script as the foreground job of a pseudo-terminal, and types ^C on that terminal when a byte arrives on its own stdin. It is a
# process of its own because a fork is not safe in a pytest that has imported torch (threads), and it is single-threaded. It writes the
# script's pid (also the id of the script's session and process group) to a file, prints what the script printed, and exits with its status.
PTY_HELPER = r"""import os, pty, sys
bash, script, pidfile = sys.argv[1:4]
pid, fd = pty.fork()
if pid == 0:
    os.execv(bash, [bash, script])
with open(pidfile, "w") as handle:
    handle.write(str(pid))
sys.stdin.buffer.read(1)
os.write(fd, b"\x03")
out = b""
while True:
    try:
        chunk = os.read(fd, 4096)
    except OSError:
        break
    if not chunk:
        break
    out += chunk
_, raw = os.waitpid(pid, 0)
sys.stdout.buffer.write(out)
sys.stdout.flush()
sys.exit(os.WEXITSTATUS(raw) if os.WIFEXITED(raw) else 128 + os.WTERMSIG(raw))
"""


def ctrl_c_on_a_terminal(box: Box, script: Path, ready: Callable[[], object], **extra_env: Optional[str]):
    """The script as the foreground job of a terminal and a ^C typed on that terminal once ready() holds: -> (exit status, what it printed,
    whether a process of its group still ran two seconds after it ended). A real Ctrl-C goes to the whole foreground process group, the ssh
    or the cloudflared in it too, which a signal sent to the script alone does not; and bash starts a background job with SIGINT ignored."""
    env = box.checked_env(**extra_env)
    helper, pidfile = box.root / "pty_helper.py", box.root / "terminal.pid"
    helper.write_text(PTY_HELPER)
    proc = spawn([sys.executable, str(helper), BASH, str(script), str(pidfile)], cwd=str(box.root), env=env, stdin=subprocess.PIPE,
                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
    group = None
    try:
        wait_for(lambda: pidfile.exists() and pidfile.read_text() and ready(), 15)
        group = int(pidfile.read_text())
        proc.stdin.write(b"x")
        proc.stdin.flush()
        try:
            out, _ = proc.communicate(timeout=15)
        except subprocess.TimeoutExpired as expired:
            raise AssertionError("the script did not end after ^C; it printed {!r}".format(expired.output))
        left = True
        deadline = time.monotonic() + 2
        while left and time.monotonic() < deadline:
            try:
                os.killpg(group, 0)
                time.sleep(0.02)
            except (ProcessLookupError, PermissionError):
                left = False
        return proc.returncode, out.decode(errors="replace"), left
    finally:
        for target in (group, proc.pid):
            if target is not None:
                try:
                    os.killpg(target, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
        proc.wait()
        for stream in (proc.stdin, proc.stdout):
            if stream is not None and not stream.closed:
                stream.close()


def jump_forward(box: Box, node: str = "gx01.example", port: int = 43211, user: str = CLUSTER_USER) -> List[str]:
    return ["-N", "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=3",
            "-o", "ConnectTimeout=10", "-o", "GatewayPorts=no", "-o", "StrictHostKeyChecking=accept-new",
            "-o", "UserKnownHostsFile=" + str(box.home / ".ssh" / "known_hosts_hpi_nodes"),
            "-J", LOGIN, "-L", "{}:127.0.0.1:{}".format(box.port, port), "{}@{}".format(user, node)]


def login_forward(box: Box, node: str = "gx01.example", port: int = 43211) -> List[str]:
    return ["-N", "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=3",
            "-o", "ConnectTimeout=10", "-o", "GatewayPorts=no", "-L", "{}:{}:{}".format(box.port, node, port), LOGIN]


# ---- tunnel.sh: the endpoint -------------------------------------------------------------------------------------------------------

def test_without_an_endpoint_the_hint_is_printed_and_the_read_is_retried(box):
    done = box.run(TUNNEL, MAX_ATTEMPTS="3")
    assert done.rc == 0, done.out
    hints = [l for l in done.lines if "no endpoint yet" in l]
    assert len(hints) == 3 and all(re.match(STAMPED + NO_ENDPOINT + "$", l) for l in hints), done.out
    assert box.calls("ssh") == [READ] * 3, "a read each attempt, and no forward without an endpoint"
    assert re.match(STAMPED + r"stopping after 3 attempt\(s\) \(MAX_ATTEMPTS\)$", done.lines[-1]), done.out


def test_the_only_thing_run_on_the_cluster_is_one_batchmode_cat(box):
    box.read(box.endpoint())
    box.forward("0")
    box.run(TUNNEL, MAX_ATTEMPTS="2")
    reads = [c for c in box.calls("ssh") if "-N" not in c]
    assert reads == [READ, READ]
    assert not [c for c in box.calls("ssh") if "squeue" in c], "squeue is hint text on the laptop's terminal, never a remote command"


def test_a_read_that_fails_in_ssh_itself_says_so_on_the_same_line(box):
    box.read("", rc=255)
    done = box.run(TUNNEL, MAX_ATTEMPTS="1")
    (line,) = [l for l in done.lines if "no endpoint yet" in l]
    assert re.match(STAMPED + NO_ENDPOINT + r"; ssh itself failed \(exit 255\)", line), line
    box.read("", rc=1)    # a cat that found no file is the ordinary "no endpoint yet"
    done = box.run(TUNNEL, MAX_ATTEMPTS="1")
    (line,) = [l for l in done.lines if "no endpoint yet" in l]
    assert re.match(STAMPED + NO_ENDPOINT + "$", line) and "ssh itself" not in line, line


def test_a_failing_read_is_no_endpoint_even_when_it_printed_something(box):
    box.read(box.endpoint(), rc=1)
    done = box.run(TUNNEL, MAX_ATTEMPTS="2")
    assert [c for c in box.calls("ssh") if "-N" in c] == [], done.out
    assert len([l for l in done.lines if "no endpoint yet" in l]) == 2


# ---- tunnel.sh: the two paths --------------------------------------------------------------------------------------------------------

def test_a_valid_endpoint_is_forwarded_through_the_login_node_as_a_jump_host(box):
    box.read(box.endpoint(host="gx01.example", port=43211, mode="private"))
    box.forward("0")
    done = box.run(TUNNEL, MAX_ATTEMPTS="1")
    assert done.rc == 0, done.out
    assert box.calls("ssh") == [READ, jump_forward(box)]
    forwarding = [l for l in done.lines if "forwarding" in l]
    assert len(forwarding) == 1 and re.match(
        STAMPED + r"forwarding localhost:{} -> gx01\.example:43211 via jump \(mode private\)$".format(box.port), forwarding[0]), done.out
    assert not [l for l in done.lines if "BIND" in l], "the note is for VIA=login"


def test_via_login_forwards_to_the_node_from_the_login_node_and_says_what_it_needs(box):
    box.read(box.endpoint(host="gx01.example", port=43211, mode="private"))
    box.forward("0")
    done = box.run(TUNNEL, MAX_ATTEMPTS="1", VIA="login")
    assert done.rc == 0, done.out
    assert box.calls("ssh") == [READ, login_forward(box)]
    forward = box.calls("ssh")[1]
    for relaxed in ("StrictHostKeyChecking=accept-new", "-J", "127.0.0.1"):
        assert not [a for a in forward if relaxed in a], "{} belongs to the jump path only".format(relaxed)
    assert not [a for a in forward if "known_hosts" in a]
    notes = [l for l in done.lines if "BIND" in l]
    assert len(notes) == 1 and "127.0.0.1" in notes[0] and "token" in notes[0] and "R6" in notes[0], done.out
    assert [l for l in done.lines if "forwarding" in l and "via login" in l]


def test_the_jump_path_keeps_the_host_key_relaxation_to_itself_and_to_the_nodes_file(box):
    box.read(box.endpoint())
    box.forward("0")
    box.run(TUNNEL, MAX_ATTEMPTS="1")
    forward = box.calls("ssh")[1]
    assert "StrictHostKeyChecking=accept-new" in forward
    assert "UserKnownHostsFile=" + str(box.home / ".ssh" / "known_hosts_hpi_nodes") in forward, "a file of its own, never ~/.ssh/known_hosts"
    assert not [a for a in forward if a.endswith("/.ssh/known_hosts")]


def test_the_cluster_user_can_be_set(box):
    box.read(box.endpoint())
    box.forward("0")
    box.run(TUNNEL, MAX_ATTEMPTS="1", CLUSTER_USER="someone.else")
    assert box.calls("ssh")[1][-1] == "someone.else@gx01.example"


def test_the_forward_is_never_bound_beyond_loopback_whatever_the_ssh_config_says(box):
    """R6: a `GatewayPorts yes` in ~/.ssh/config would bind the forwarded port on every interface; the command line wins over the config."""
    box.read(box.endpoint())
    box.forward("0")
    for via in ("jump", "login"):
        box.run(TUNNEL, MAX_ATTEMPTS="1", VIA=via)
    forwards = [c for c in box.calls("ssh") if "-N" in c]
    assert len(forwards) == 2 and all("GatewayPorts=no" in c for c in forwards)


def test_the_tunnel_reads_the_file_the_server_writes(box, monkeypatch):
    """The consumer against the real producer: the plan's sketch had host:port, the server writes JSON (app/cli.py write_endpoint)."""
    from app import cli
    monkeypatch.setattr(cli.socket, "gethostname", lambda: "gx17.hpc.example")
    for mode in ("private", "public"):
        target = box.root / "chat_sessions" / mode
        cli.write_endpoint(str(target), 43999, mode)
        box.read(target.read_text())
        box.forward("0")
        done = box.run(TUNNEL, MAX_ATTEMPTS="1")
        assert [l for l in done.lines if l.endswith("-> gx17.hpc.example:43999 via jump (mode {})".format(mode))], done.out
        assert box.calls("ssh")[-1] == jump_forward(box, node="gx17.hpc.example", port=43999)


# ---- tunnel.sh: untrusted endpoint content -------------------------------------------------------------------------------------------

# (case, the file's content, a string of it that must never appear in the output)
BAD_ENDPOINTS = [
    ("the old host:port sketch", "gx01:43211\n", "gx01"),
    ("not json at all", "<html>login</html>\n", "html"),
    ("an empty object", "{}", "{}"),
    ("a list", '["gx01", 43211]', "gx01"),
    ("a string", '"gx01"', "gx01"),
    ("a number", "43211", "43211"),
    ("null", "null", "null"),
    ("cut off", '{"host": "gx01", "port": 4321', "gx01"),
    ("no host", '{"port": 43211}', "43211"),
    ("no port", '{"host": "gx01"}', "gx01"),
    ("a space in the host", '{"host": "gx01 evil", "port": 43211}', "evil"),
    ("a semicolon in the host", '{"host": "gx01;touch PWNED", "port": 43211}', "PWNED"),
    ("an ssh option for a host", '{"host": "-oProxyCommand=touch PWNED", "port": 43211}', "PWNED"),
    ("a command substitution", '{"host": "$(touch PWNED)", "port": 43211}', "PWNED"),
    ("backticks", '{"host": "`touch PWNED`", "port": 43211}', "PWNED"),
    ("a newline inside the host", '{"host": "gx01\\nPWNED", "port": 43211}', "PWNED"),
    ("a newline after the host", '{"host": "gx01\\n", "port": 43211}', "gx01"),
    ("an empty host", '{"host": "", "port": 43211}', "43211"),
    ("a host that is a number", '{"host": 5, "port": 43211}', "43211"),
    ("a host of 254 characters", json.dumps({"host": "a" * 254, "port": 43211}), "aaaaaaaa"),
    ("port 0", '{"host": "gx01", "port": 0}', "gx01"),
    ("port 65536", '{"host": "gx01", "port": 65536}', "65536"),
    ("port 70000", '{"host": "gx01", "port": 70000}', "70000"),
    ("a negative port", '{"host": "gx01", "port": -1}', "gx01"),
    ("a port as text", '{"host": "gx01", "port": "43211"}', "gx01"),
    ("a port with a command", '{"host": "gx01", "port": "1; touch PWNED"}', "PWNED"),
    ("a port with a fraction", '{"host": "gx01", "port": 43211.5}', "43211"),
    ("a port that is true", '{"host": "gx01", "port": true}', "gx01"),
    ("a null port", '{"host": "gx01", "port": null}', "gx01"),
    ("an infinite port", '{"host": "gx01", "port": 1e999}', "1e999"),
    ("more than the file can be", '{"host": "gx01", "port": 43211, "note": "' + "x" * 5000 + '"}', "xxxxxxxx"),
]


@pytest.mark.parametrize("content,leak", [(c, l) for _, c, l in BAD_ENDPOINTS], ids=[n for n, _, _ in BAD_ENDPOINTS])
def test_an_endpoint_that_is_not_what_the_server_writes_is_ignored_unprinted_and_read_again(box, content, leak):
    box.read(content)
    done = box.run(TUNNEL, MAX_ATTEMPTS="2")
    assert done.rc == 0, done.out
    unreadable = [l for l in done.lines if "endpoint unreadable" in l]
    assert len(unreadable) == 2 and all(re.match(STAMPED + UNREADABLE + "$", l) for l in unreadable), done.out
    assert box.calls("ssh") == [READ, READ], "nothing from the file reached ssh as an argument"
    assert leak not in done.out, "the content is never echoed"
    assert not (box.root / "PWNED").exists(), "the content is never run"


@pytest.mark.parametrize("host,port", [("a", 1), ("gx01", 65535), ("gx01.hpc.example-site.de", 8080), ("127.0.0.1", 22),
                                       ("a" + "b" * 252, 43211), ("0node", 43211)],
                         ids=["one letter, port 1", "port 65535", "dots and a hyphen", "an address", "253 characters", "a leading digit"])
def test_what_the_validation_allows_is_forwarded(box, host, port):
    box.read(box.endpoint(host=host, port=port))
    box.forward("0")
    done = box.run(TUNNEL, MAX_ATTEMPTS="1")
    assert box.calls("ssh") == [READ, jump_forward(box, node=host, port=port)], done.out


def test_the_mode_is_shown_only_when_it_is_one_of_the_two_and_is_never_run(box):
    box.read('{"host": "gx01", "port": 43211, "mode": "$(touch PWNED)"}')
    box.forward("0")
    done = box.run(TUNNEL, MAX_ATTEMPTS="1")
    (line,) = [l for l in done.lines if "forwarding" in l]
    assert line.endswith("via jump (mode unknown)"), done.out
    assert "PWNED" not in done.out and not (box.root / "PWNED").exists()
    box.read('{"host": "gx01", "port": 43211}')       # a file without a mode
    done = box.run(TUNNEL, MAX_ATTEMPTS="1")
    assert [l for l in done.lines if l.endswith("via jump (mode unknown)")]


# ---- tunnel.sh: a stale endpoint, the local port -------------------------------------------------------------------------------------

def test_a_forward_that_ends_at_once_sends_the_loop_back_to_the_file(box):
    """A node that is gone (or a stale file): ExitOnForwardFailure and ConnectTimeout end the ssh, the loop sleeps, and the file is read
    again, so the forward that follows goes to wherever the job is now."""
    box.read(box.endpoint(host="gx01", port=43001), n=1)
    box.read(box.endpoint(host="gx02", port=43002), n=2)
    box.forward("255")
    done = box.run(TUNNEL, MAX_ATTEMPTS="2")
    assert done.rc == 0, done.out
    assert box.calls("ssh") == [READ, jump_forward(box, node="gx01", port=43001), READ, jump_forward(box, node="gx02", port=43002)]
    assert len([l for l in done.lines if "forwarding" in l]) == 2, "one line per attempt"


def test_the_pause_between_attempts_is_RETRY_SLEEP_and_there_is_none_after_the_last(box):
    started = time.monotonic()
    box.run(TUNNEL, MAX_ATTEMPTS="2", RETRY_SLEEP="1.5")
    elapsed = time.monotonic() - started
    assert 1.5 <= elapsed < 2.9, "one pause of 1.5 s between two attempts, and none after the second: {:.2f} s".format(elapsed)


def test_a_busy_local_port_ends_the_tunnel_with_status_1_before_anything_is_asked(box):
    message = "localhost:{} is already in use (a local dev server?); set LOCAL_PORT to a free port".format(box.port)
    with socket.socket() as dev_server:
        dev_server.bind(("127.0.0.1", box.port))
        dev_server.listen(1)
        box.read(box.endpoint())
        done = box.run(TUNNEL, MAX_ATTEMPTS="0")
        assert done.rc == 1 and done.lines == [message], done.out
        (box.stubs / "read").unlink()                       # and with no endpoint either: the port is checked before the first read
        done = box.run(TUNNEL, MAX_ATTEMPTS="0")
        assert done.rc == 1 and done.lines == [message], done.out
        assert box.calls("ssh") == [], "no ssh at all"


def test_the_dev_server_taking_the_port_between_two_forwards_ends_the_tunnel_instead_of_looping(box):
    box.read(box.endpoint())
    box.forward("bind", n=1)         # the first forward ends, and a dev server has the port by then
    done = box.run(TUNNEL, MAX_ATTEMPTS="0")
    assert done.rc == 1, done.out
    assert done.lines[-1].startswith("localhost:{} is already in use".format(box.port))
    assert box.calls("ssh") == [READ, jump_forward(box), READ], "no second forward: the loop ends rather than retry ExitOnForwardFailure"


def test_a_port_the_last_forward_left_in_time_wait_is_free(box):
    """ssh sets SO_REUSEADDR on its listener, so the bind test must too: without it the connections the last forward closed would read as
    'in use' for a minute after it ended, and a requeue would end the tunnel instead of reconnecting it."""
    listener = socket.socket()
    listener.bind(("127.0.0.1", box.port))
    listener.listen(1)
    client = socket.create_connection(("127.0.0.1", box.port))
    served, _ = listener.accept()
    served.close()                   # the side that closes first keeps the TIME_WAIT
    client.close()
    listener.close()
    box.read(box.endpoint())
    box.forward("0")
    done = box.run(TUNNEL, MAX_ATTEMPTS="1")
    assert done.rc == 0 and box.calls("ssh") == [READ, jump_forward(box)], done.out


# ---- tunnel.sh: settings ---------------------------------------------------------------------------------------------------------------

BAD_SETTINGS = [
    ({"VIA": "tunnel"}, "VIA must be jump or login"),
    ({"VIA": "Jump"}, "VIA must be jump or login"),
    ({"LOCAL_PORT": "abc"}, BAD_PORT),
    ({"LOCAL_PORT": "0"}, BAD_PORT),
    ({"LOCAL_PORT": "65536"}, BAD_PORT),
    ({"LOCAL_PORT": "-1"}, BAD_PORT),
    ({"LOCAL_PORT": "012345"}, BAD_PORT),
    ({"LOCAL_PORT": "99999999999999999999"}, BAD_PORT),
    ({"LOCAL_PORT": "8010;touch PWNED"}, BAD_PORT),
    ({"LOGIN": "-oProxyCommand=touch PWNED"}, "LOGIN must be an ssh host name or alias"),
    ({"LOGIN": "fake login.invalid"}, "LOGIN must be an ssh host name or alias"),
    ({"LOGIN": "fake;touch PWNED"}, "LOGIN must be an ssh host name or alias"),
    ({"CLUSTER_USER": "-x"}, "CLUSTER_USER must be a plain user name"),
    ({"CLUSTER_USER": "a b"}, "CLUSTER_USER must be a plain user name"),
    ({"MAX_ATTEMPTS": "many"}, "MAX_ATTEMPTS must be a whole number"),
    ({"MAX_ATTEMPTS": "-1"}, "MAX_ATTEMPTS must be a whole number"),
    ({"MAX_ATTEMPTS": "1.5"}, "MAX_ATTEMPTS must be a whole number"),
    ({"MAX_ATTEMPTS": "1234567890"}, "MAX_ATTEMPTS must be a whole number"),
    ({"RETRY_SLEEP": "soon"}, "RETRY_SLEEP must be a number of seconds"),
    ({"RETRY_SLEEP": "-1"}, "RETRY_SLEEP must be a number of seconds"),
    ({"RETRY_SLEEP": "1e3"}, "RETRY_SLEEP must be a number of seconds"),
    ({"RETRY_SLEEP": "1.2.3"}, "RETRY_SLEEP must be a number of seconds"),
    ({"RETRY_SLEEP": ".5"}, "RETRY_SLEEP must be a number of seconds"),
    ({"RETRY_SLEEP": "5."}, "RETRY_SLEEP must be a number of seconds"),
]


@pytest.mark.parametrize("setting,message", BAD_SETTINGS, ids=["{}={}".format(*next(iter(s.items()))) for s, _ in BAD_SETTINGS])
def test_a_setting_that_is_not_one_is_refused_with_status_2_before_anything_runs(box, setting, message):
    box.read(box.endpoint())
    done = box.run(TUNNEL, **setting)
    assert done.rc == 2 and len(done.lines) == 1 and done.lines[0].startswith(message), done.out
    assert box.calls("ssh") == [] and not (box.root / "PWNED").exists()


def test_without_python3_the_tunnel_says_so(box):
    box.remove("python3")
    done = box.run(TUNNEL, PATH=str(box.bin))
    assert done.rc == 2 and len(done.lines) == 1 and "python3" in done.lines[0], done.out
    assert box.calls("ssh") == []


# ---- tunnel.sh: stopping ---------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("sig,status", [(signal.SIGTERM, 143), (signal.SIGINT, 130), (signal.SIGHUP, 129)], ids=["TERM", "INT", "HUP"])
def test_a_signal_stops_the_running_forward_and_leaves_no_process_behind(box, sig, status):
    box.read(box.endpoint())
    box.forward("sleep")
    run = box.start(TUNNEL, MAX_ATTEMPTS="0")
    wait_for(lambda: (box.stubs / "forward.pid").exists())
    forward_pid = int((box.stubs / "forward.pid").read_text())
    assert alive(forward_pid)
    run.signal(sig)
    assert run.wait(10) == status, run.out
    assert not alive(forward_pid), "the ssh was stopped with the script"
    wait_for(lambda: not run.group_alive(), 5)
    assert run.lines[-1] == "stopped"


def test_a_signal_during_the_retry_sleep_is_not_held_up_by_it(box):
    run = box.start(TUNNEL, MAX_ATTEMPTS="0", RETRY_SLEEP="30")
    wait_for(lambda: "no endpoint yet" in run.out)
    started = time.monotonic()
    run.signal(signal.SIGTERM)
    assert run.wait(10) == 143, run.out
    assert time.monotonic() - started < 5, "the 30 s sleep did not hold the signal up"
    wait_for(lambda: not run.group_alive(), 5)


def test_by_default_the_loop_never_ends_by_itself(box):
    run = box.start(TUNNEL, MAX_ATTEMPTS=None, RETRY_SLEEP="0.05")        # MAX_ATTEMPTS unset: the default is no limit
    wait_for(lambda: run.out.count("no endpoint yet") >= 6)
    assert run.proc.poll() is None, run.out
    run.signal(signal.SIGTERM)
    assert run.wait(10) == 143, run.out
    assert "stopping after" not in run.out


def test_ctrl_c_on_a_terminal_stops_tunnel_sh_and_the_ssh_it_runs(box):
    box.read(box.endpoint())
    box.forward("sleep")
    status, out, left = ctrl_c_on_a_terminal(box, TUNNEL, lambda: (box.stubs / "forward.pid").exists(), MAX_ATTEMPTS="0")
    assert status == 130 and out.rstrip().endswith("stopped"), (status, out)
    assert not alive(int((box.stubs / "forward.pid").read_text())) and not left


def test_ctrl_c_on_a_terminal_while_it_pauses_between_attempts_ends_it_at_once(box):
    started = time.monotonic()
    status, out, left = ctrl_c_on_a_terminal(box, TUNNEL, lambda: (box.stubs / "ssh.calls").exists(), MAX_ATTEMPTS="0", RETRY_SLEEP="30")
    assert status == 130 and out.rstrip().endswith("stopped") and not left, (status, out)
    assert time.monotonic() - started < 10, "the 30 s pause did not hold ^C up"


# ---- public_demo.sh ----------------------------------------------------------------------------------------------------------------------

def public(box: Box, body: Optional[str], rc: Optional[int] = None) -> None:
    """What the server's /healthz answers (None: nothing is listening, as curl reports for a refused connection)."""
    if body is not None:
        box.set("healthz", body)
    if rc is not None:
        box.set("healthz.rc", str(rc))


CURL = ["-s", "--max-time", "5"]


def healthz_url(box: Box) -> str:
    return "http://localhost:{}/healthz".format(box.port)


def test_cloudflared_runs_only_for_a_server_that_says_it_is_public(box):
    from app.server import HEALTHZ_EXAMPLE
    box.install("cloudflared", FAKE_CLOUDFLARED)
    answers = [json.dumps(dict(HEALTHZ_EXAMPLE, mode="public")),                  # the shape the real server answers with
               '{"status":"ok","mode":"public"}', '{"mode":   "public"}\n',
               '{"mode": "public", "default_model": "private"}',                  # "private" in another field does not matter either
               '{\n  "status": "ok",\n  "mode": "public"\n}\n']
    for body in answers:
        public(box, body)
        done = box.run(PUBLIC_DEMO)
        assert done.rc == 0, (body, done.out)
    assert box.calls("curl") == [CURL + [healthz_url(box)]] * len(answers)
    assert box.calls("cloudflared") == [["tunnel", "--url", "http://localhost:{}".format(box.port)]] * len(answers)


REFUSED = [
    ("private mode", '{"status": "ok", "mode": "private", "default_model": "m"}', 0, "private mode"),
    ("nothing listening", None, 7, "no usable"),
    ("a timeout", None, 28, "no usable"),
    ("an empty answer", "", 0, "no usable"),
    ("not JSON", "<html>502 Bad Gateway</html>", 0, "no usable"),
    ("json that looks like html", '<pre>{"mode": "public"}</pre>', 0, "no usable"),
    ("plain text", 'mode: "public"', 0, "no usable"),
    ("public in another field", '{"status": "ok", "mode": "private", "default_model": "public"}', 0, "private mode"),
    ("public in a text field", '{"mode": "private", "note": "the \\"mode\\": \\"public\\" one"}', 0, "private mode"),
    ("a mode that only starts with public", '{"mode": "publicish"}', 0, "no usable"),
    ("another case", '{"mode": "Public"}', 0, "no usable"),
    ("padding", '{"mode": " public"}', 0, "no usable"),
    ("a list holding public", '{"mode": ["public"]}', 0, "no usable"),
    ("no mode", '{"status": "ok"}', 0, "no usable"),
    ("a null mode", '{"mode": null}', 0, "no usable"),
    ("a json list", '["public"]', 0, "no usable"),
    ("a json string", '"public"', 0, "no usable"),
]


@pytest.mark.parametrize("body,rc,reason", [(b, r, w) for _, b, r, w in REFUSED], ids=[n for n, _, _, _ in REFUSED])
def test_public_demo_refuses_everything_but_json_whose_mode_is_exactly_public(box, body, rc, reason):
    box.install("cloudflared", FAKE_CLOUDFLARED)
    public(box, body, rc)
    done = box.run(PUBLIC_DEMO)
    assert done.rc == 1, done.out
    assert len(done.lines) == 1 and done.lines[0].startswith("refusing: ") and reason in done.lines[0], done.out
    assert box.calls("cloudflared") == [] and not (box.stubs / "cloudflared.pid").exists(), "cloudflared never started"


def test_without_cloudflared_a_public_server_is_still_refused_with_a_pointer_to_the_install_page(box):
    public(box, '{"mode": "public"}')
    done = box.run(PUBLIC_DEMO)
    assert done.rc == 1 and len(done.lines) == 1, done.out
    assert done.lines[0].startswith("refusing: cloudflared is not installed") and "https://developers.cloudflare.com/" in done.lines[0]
    assert box.calls("curl") == [CURL + [healthz_url(box)]] and box.calls("ssh") == [], "the one probe, and nothing else was run"


def test_the_mode_is_checked_before_cloudflared_is_looked_for(box):
    public(box, '{"mode": "private"}')          # no cloudflared either: the refusal that matters is the mode
    done = box.run(PUBLIC_DEMO)
    assert done.rc == 1 and "private mode" in done.out and "cloudflared" not in done.out, done.out


def test_the_reminder_comes_before_cloudflared_and_the_closing_line_after_it(box):
    box.install("cloudflared", FAKE_CLOUDFLARED)
    public(box, '{"mode": "public"}')
    done = box.run(PUBLIC_DEMO)
    assert done.rc == 0, done.out
    text = done.out
    assert "anyone with the link" in text.lower(), text
    assert text.index("Stop it") < text.index("fake cloudflared is up") < text.index("cloudflared stopped"), text


def test_the_status_cloudflared_ends_with_is_the_scripts(box):
    box.install("cloudflared", FAKE_CLOUDFLARED)
    public(box, '{"mode": "public"}')
    box.set("cloudflared.rc", "3")
    done = box.run(PUBLIC_DEMO)
    assert done.rc == 3 and "cloudflared stopped" in done.out, done.out


@pytest.mark.parametrize("sig,status", [(signal.SIGTERM, 143), (signal.SIGINT, 130), (signal.SIGHUP, 129)], ids=["TERM", "INT", "HUP"])
def test_a_signal_stops_cloudflared_and_leaves_no_process_behind(box, sig, status):
    box.install("cloudflared", FAKE_CLOUDFLARED)
    public(box, '{"mode": "public"}')
    box.set("cloudflared.stay", "")
    run = box.start(PUBLIC_DEMO)
    wait_for(lambda: (box.stubs / "cloudflared.pid").exists() and "fake cloudflared is up" in run.out)
    pid = int((box.stubs / "cloudflared.pid").read_text())
    assert alive(pid)
    run.signal(sig)
    assert run.wait(10) == status, run.out
    assert not alive(pid), "cloudflared was stopped with the script"
    wait_for(lambda: not run.group_alive(), 5)
    assert "cloudflared stopped" in run.lines[-1]


def test_ctrl_c_on_a_terminal_stops_public_demo_sh_and_the_cloudflared_it_runs(box):
    box.install("cloudflared", FAKE_CLOUDFLARED)
    public(box, '{"mode": "public"}')
    box.set("cloudflared.stay", "")
    status, out, left = ctrl_c_on_a_terminal(box, PUBLIC_DEMO, lambda: (box.stubs / "cloudflared.pid").exists())
    assert status == 130 and "cloudflared stopped: the public link is closed." in out, (status, out)
    assert not alive(int((box.stubs / "cloudflared.pid").read_text())) and not left


@pytest.mark.parametrize("bad", ["abc", "0", "65536", "8000/evil", "8010;touch PWNED", "012345", "-1"])
def test_public_demo_refuses_a_port_that_is_not_one_before_asking_anything(box, bad):
    done = box.run(PUBLIC_DEMO, LOCAL_PORT=bad)
    assert done.rc == 2 and done.lines == [BAD_PORT], done.out
    assert box.calls("curl") == [] and not (box.root / "PWNED").exists()


def test_the_token_is_never_read_named_or_printed_by_either_script(box):
    """R6/R7: the server's token file is the user's secret. A canary in the two places a stray read could find one (the working directory,
    where the remote path chat_sessions/app_token would resolve, and the home) must reach no output and no argument of ssh, curl or cloudflared."""
    secret = "s3cr3t-canary-0123456789"
    for place in (box.root / "chat_sessions", box.home / "chat_sessions"):
        place.mkdir()
        (place / "app_token").write_text(secret)
    box.install("cloudflared", FAKE_CLOUDFLARED)
    box.read(box.endpoint())
    box.forward("0")
    public(box, '{"mode": "public"}')
    outputs = [box.run(TUNNEL, MAX_ATTEMPTS="1", VIA=via).out for via in ("jump", "login")] + [box.run(PUBLIC_DEMO).out]
    assert not [out for out in outputs if secret in out or "app_token" in out]
    for tool in ("ssh", "curl", "cloudflared"):
        assert box.calls(tool), tool
        assert not [a for call in box.calls(tool) for a in call if "app_token" in a or secret in a], tool


def test_public_demo_without_python3_says_so(box):
    box.remove("python3")
    done = box.run(PUBLIC_DEMO, PATH=str(box.bin))
    assert done.rc == 2 and len(done.lines) == 1 and "python3" in done.lines[0], done.out
    assert box.calls("curl") == []
