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

P7-C fix 1 added the loops that watch a running child: tunnel.sh reads the endpoint again every RECHECK_S seconds while its forward runs,
public_demo.sh asks /healthz again every DEMO_RECHECK_S seconds while cloudflared runs. Both look at their child once a second, so a rehearsal
whose fake forward or cloudflared ends at once still takes up to a second for it, and RECHECK_S=1 / DEMO_RECHECK_S=1 (the smallest the settings
allow) put a re-read or a check in every second of a run. The fake ssh numbers its forwards (forward.<n>.pid, forward.<n>.prev) and the fake
curl its answers (healthz.<n>), so that a test can say what the second read, or the third answer, is.
"""
import json
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

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
# file KIND.<n> in $STUB_DIR, else by the file KIND: a forward's file says how it ends (a status; `sleep` to stay up until it is stopped;
# `after:SECONDS:STATUS` to stay up that long and then end by itself; or `bind` to take LOCAL_PORT as a dev server would, and then end), a
# read's file is what it prints, with <file>.rc beside it for its status. Beside the argv it records, in ssh.stdin, what each call's stdin was
# (/dev/null or something else), and for the n-th forward, in forward.<n>.pid its pid (the forwards that stay up) and in forward.<n>.prev
# whether the (n-1)-th forward was still running when this one started.
FAKE_SSH = "#!/bin/bash\n" + record("ssh") + r"""forward=0
for a in "$@"; do [ "$a" = "-N" ] && forward=1; done
kind=read
[ "$forward" = 1 ] && kind=forward
if [ /dev/stdin -ef /dev/null ]; then echo "$kind devnull" >> "$STUB_DIR/ssh.stdin"; else echo "$kind other" >> "$STUB_DIR/ssh.stdin"; fi
nth() {
  local n=0
  [ -f "$STUB_DIR/$1.count" ] && n=$(cat "$STUB_DIR/$1.count")
  n=$((n + 1))
  echo "$n" > "$STUB_DIR/$1.count"
  if [ -f "$STUB_DIR/$1.$n" ]; then echo "$STUB_DIR/$1.$n"; elif [ -f "$STUB_DIR/$1" ]; then echo "$STUB_DIR/$1"; fi
}
if [ "$forward" = 1 ]; then
  file=$(nth forward)
  n=$(cat "$STUB_DIR/forward.count")
  how=255
  [ -n "$file" ] && how=$(cat "$file")
  prev=$(cat "$STUB_DIR/forward.$((n - 1)).pid" 2>/dev/null)
  if [ -n "$prev" ] && kill -0 "$prev" 2>/dev/null; then echo alive; else echo gone; fi > "$STUB_DIR/forward.$n.prev"
  case "$how" in
    sleep) echo $$ > "$STUB_DIR/forward.pid"; echo $$ > "$STUB_DIR/forward.$n.pid"; exec sleep 60 ;;
    after:*) IFS=: read -r _ secs status <<< "$how"
           echo $$ > "$STUB_DIR/forward.pid"; echo $$ > "$STUB_DIR/forward.$n.pid"
           sleep "$secs"; exit "$status" ;;
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
# `curl`: answers with $STUB_DIR/healthz (and its status from healthz.rc), or, with no such file, as curl does for a refused connection. The
# n-th call is answered by healthz.<n> (and healthz.<n>.rc) when either exists, so that a test can make one answer in a row bad.
FAKE_CURL = "#!/bin/bash\n" + record("curl") + r"""n=0
[ -f "$STUB_DIR/curl.count" ] && n=$(cat "$STUB_DIR/curl.count")
n=$((n + 1))
echo "$n" > "$STUB_DIR/curl.count"
name=healthz
{ [ -f "$STUB_DIR/healthz.$n" ] || [ -f "$STUB_DIR/healthz.$n.rc" ]; } && name="healthz.$n"
rc=$(cat "$STUB_DIR/$name.rc" 2>/dev/null)
if [ -f "$STUB_DIR/$name" ]; then cat "$STUB_DIR/$name"; exit "${rc:-0}"; fi
exit "${rc:-7}"
"""
# `cloudflared`: records its argv and its pid, prints a line, then ends with the status in cloudflared.rc (default 0), or stays up until it
# is stopped when cloudflared.stay exists, or stays up that many seconds (the number in cloudflared.after) and then ends by itself.
FAKE_CLOUDFLARED = "#!/bin/bash\n" + record("cloudflared") + r"""echo $$ > "$STUB_DIR/cloudflared.pid"
echo "fake cloudflared is up"
if [ -f "$STUB_DIR/cloudflared.stay" ]; then exec sleep 60; fi
if [ -f "$STUB_DIR/cloudflared.after" ]; then sleep "$(cat "$STUB_DIR/cloudflared.after")"; fi
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

    def toolbox(self, *names: str) -> Path:
        """A directory of symlinks to exactly these tools and nothing else: a name that is one of the fakes here links to the fake, any other
        to the real tool of that name. A script run with only this directory on its PATH can find nothing more, whatever happens to be
        installed in /usr/bin or /bin on this machine."""
        tools = self.root / "tools"
        tools.mkdir(exist_ok=True)
        for name in names:
            target = self.bin / name
            if not target.exists():
                found = shutil.which(name)
                assert found, "{} is not installed here".format(name)
                target = Path(found)
            (tools / name).symlink_to(target)
        return tools

    def set(self, name: str, text: str) -> None:
        """A stub file the fakes read, written atomically: a fake that reads it while a test changes it sees the old or the new, never half."""
        tmp = self.stubs / (name + ".tmp")
        tmp.write_text(text)
        os.replace(str(tmp), str(self.stubs / name))

    def unset(self, name: str) -> None:
        if (self.stubs / name).exists():
            (self.stubs / name).unlink()

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
        fakes (the fakes' directory first, or a toolbox() directory alone: it holds the fakes and the few real tools a test names)."""
        env = self.env(**extra_env)
        assert env.get("LOCAL_PORT") not in (None, "", "8000"), "a rehearsal never meets port 8000"
        assert env.get("LOGIN") not in (None, "", "hpi-hpc"), "a rehearsal never meets the real login alias"
        assert (env["PATH"] == str(self.bin) or env["PATH"].startswith(str(self.bin) + ":")
                or env["PATH"] == str(self.root / "tools")), "the fakes come first"
        return env

    def start(self, script: Path, stdin=subprocess.DEVNULL, **extra_env: Optional[str]) -> Run:
        """stdin is /dev/null unless a test hands the script something else (subprocess.PIPE: an open pipe nobody writes to)."""
        env = self.checked_env(**extra_env)
        log_path = self.root / "run{}.log".format(len(self.runs))
        log = open(str(log_path), "wb")
        proc = spawn([BASH, str(script)], cwd=str(self.root), env=env, stdin=stdin, stdout=log, stderr=subprocess.STDOUT,
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
            if run.proc.stdin is not None and not run.proc.stdin.closed:
                run.proc.stdin.close()


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


def known_hosts_option(home: Path) -> str:
    """The -o value for the nodes' known_hosts file: the path in double quotes, which ssh_config(5) allows around any argument and which a
    path with a space needs (UserKnownHostsFile takes one or more files, separated by whitespace)."""
    return 'UserKnownHostsFile="{}"'.format(home / ".ssh" / "known_hosts_hpi_nodes")


def jump_forward(box: Box, node: str = "gx01.example", port: int = 43211, user: str = CLUSTER_USER) -> List[str]:
    return ["-N", "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=3",
            "-o", "ConnectTimeout=10", "-o", "GatewayPorts=no", "-o", "StrictHostKeyChecking=accept-new",
            "-o", known_hosts_option(box.home),
            "-J", LOGIN, "-L", "{}:127.0.0.1:{}".format(box.port, port), "{}@{}".format(user, node)]


def login_forward(box: Box, node: str = "gx01.example", port: int = 43211) -> List[str]:
    return ["-N", "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=3",
            "-o", "ConnectTimeout=10", "-o", "GatewayPorts=no", "-L", "{}:{}:{}".format(box.port, node, port), LOGIN]


def reads(box: Box) -> List[List[str]]:
    """The calls of `ssh LOGIN cat chat_sessions/endpoint` so far: the first read of each attempt, and every re-read."""
    return [c for c in box.calls("ssh") if "-N" not in c]


def forwards(box: Box) -> List[List[str]]:
    return [c for c in box.calls("ssh") if "-N" in c]


def stdin_kinds(box: Box, kind: str) -> List[str]:
    """What the stdin of each `kind` ("read" or "forward") call of the fake ssh was: devnull or other."""
    path = box.stubs / "ssh.stdin"
    lines = path.read_text().splitlines() if path.exists() else []
    return [l.split()[1] for l in lines if l.split()[0] == kind]


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


def test_the_endpoint_read_takes_its_stdin_from_dev_null(box):
    """A tunnel started as a background job of an interactive shell is stopped with SIGTTIN when a process of it reads the terminal, and
    ssh reads its stdin when it runs a command. The harness gives the script an open pipe (not /dev/null, not a terminal) and the fake ssh
    records whether its own stdin is /dev/null."""
    box.read(box.endpoint())
    box.forward("0")
    run = box.start(TUNNEL, stdin=subprocess.PIPE, MAX_ATTEMPTS="1")
    assert run.wait(20) == 0, run.out
    assert stdin_kinds(box, "read") == ["devnull"], "the read inherited the script's stdin"


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
    assert known_hosts_option(box.home) in forward, "a file of its own, never ~/.ssh/known_hosts"
    assert not [a for a in forward if a.strip('"').endswith("/.ssh/known_hosts")]


def test_a_home_with_a_space_reaches_ssh_as_one_quoted_known_hosts_path(box):
    """P7-C fix 1, item 7. UserKnownHostsFile takes one or more files separated by whitespace, so `UserKnownHostsFile=/Users/a b/.ssh/x`
    would be two files and the nodes' keys would go to neither. In double quotes (ssh_config(5)) it is one."""
    home = box.root / "home with space"
    box.read(box.endpoint())
    box.forward("0")
    box.run(TUNNEL, MAX_ATTEMPTS="1", HOME=str(home))
    (option,) = [a for a in box.calls("ssh")[1] if a.startswith("UserKnownHostsFile=")]
    path = home / ".ssh" / "known_hosts_hpi_nodes"
    assert option == 'UserKnownHostsFile="{}"'.format(path)
    # how ssh reads a -o value: the keyword, an optional '=', then arguments that whitespace separates unless they are in double quotes
    assert shlex.split(option.replace("=", " ", 1)) == ["UserKnownHostsFile", str(path)], "one file, with its space"


@pytest.mark.parametrize("name", ['a"b', "a%b", "a\\b"], ids=["double quote", "percent", "backslash"])
def test_a_home_that_cannot_be_quoted_for_ssh_is_refused_on_the_jump_path_only(box, name):
    """Inside double quotes a double quote ends the value, a backslash escapes, and ssh expands a % token in a file name."""
    home = str(box.root / ("home" + name))
    box.read(box.endpoint())
    box.forward("0")
    done = box.run(TUNNEL, HOME=home)
    assert done.rc == 2 and len(done.lines) == 1 and done.lines[0].startswith("HOME must not contain"), done.out
    assert box.calls("ssh") == [], "nothing was asked before the refusal"
    done = box.run(TUNNEL, MAX_ATTEMPTS="1", VIA="login", HOME=home)      # the login path never uses HOME
    assert done.rc == 0 and box.calls("ssh") == [READ, login_forward(box)], done.out


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
    """Two attempts, one pause between them. The bug this guards against is a loop that sleeps before it looks at MAX_ATTEMPTS, so that
    a pause follows the last attempt too: that takes 2 x RETRY_SLEEP = 6 s, which the upper bound (a loaded machine has 3 s to spare
    before it) turns away. A loop with no pause at all (a hammering loop) ends in well under the lower bound."""
    started = time.monotonic()
    box.run(TUNNEL, MAX_ATTEMPTS="2", RETRY_SLEEP="3")
    elapsed = time.monotonic() - started
    assert 3 <= elapsed < 6, "one pause of 3 s between two attempts, and none after the second: {:.2f} s".format(elapsed)


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


def busy_line(box: Box) -> str:
    return "localhost:{} is already in use (a local dev server?); set LOCAL_PORT to a free port".format(box.port)


def loopback_free(family: int, address: str, port: int) -> bool:
    """Would a bind to this address with SO_REUSEADDR (as ssh binds) succeed? Nothing connects."""
    with socket.socket(family) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((address, port))
        except OSError:
            return False
    return True


def ipv6_loopback_listener(reuse: bool) -> Tuple[socket.socket, int]:
    """A listener on ::1, on a port that is free on 127.0.0.1, so that only the IPv6 side is busy. Skips the test on a host with no IPv6
    loopback. Never port 8000."""
    if not socket.has_ipv6:
        pytest.skip("this Python was built without IPv6")
    for _ in range(50):
        listener = socket.socket(socket.AF_INET6)
        try:
            if reuse:
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("::1", 0))
            listener.listen(1)
        except OSError as error:
            listener.close()
            pytest.skip("no usable IPv6 loopback on this host: {}".format(error))
        port = listener.getsockname()[1]
        if port != 8000 and loopback_free(socket.AF_INET, "127.0.0.1", port):
            return listener, port
        listener.close()
    pytest.skip("found no port that is free on 127.0.0.1 and can be taken on ::1")


@pytest.mark.parametrize("reuse", [False, True], ids=["plain listener", "SO_REUSEADDR listener"])
def test_a_listener_on_ipv6_loopback_makes_the_port_busy_too(box, reuse):
    """P7-C fix 1, item 2. ssh -L binds localhost, which is 127.0.0.1 and ::1, and a browser may resolve localhost to either one: a dev
    server that listens on ::1 only (node does) is in the tunnel's way as much as one on 127.0.0.1."""
    listener, box.port = ipv6_loopback_listener(reuse)
    with listener:
        box.read(box.endpoint())
        done = box.run(TUNNEL, MAX_ATTEMPTS="2")      # bounded, so that a tunnel that does not see the listener ends the test quickly
        assert done.rc == 1 and done.lines == [busy_line(box)], done.out
        assert box.calls("ssh") == [], "no ssh at all"


def test_a_127_0_0_1_listener_that_set_SO_REUSEADDR_itself_still_makes_the_port_busy(box):
    """uvicorn, like most servers, sets SO_REUSEADDR on its listener, as the bind test does. Two sockets that both set it cannot share one
    address and port while one of them listens (BSD and Linux agree), so the bind test still finds the server. What it cannot find is
    named in the script's header: a wildcard listener on macOS."""
    with socket.socket() as dev_server:
        dev_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        dev_server.bind(("127.0.0.1", box.port))
        dev_server.listen(1)
        box.read(box.endpoint())
        done = box.run(TUNNEL, MAX_ATTEMPTS="2")
        assert done.rc == 1 and done.lines == [busy_line(box)], done.out
        assert box.calls("ssh") == []


# A host without IPv6 (no address family, or no ::1 address on its loopback): Python is made to fail the way the kernel does.
NO_IPV6 = {
    "EAFNOSUPPORT when the socket is made": """import errno, socket
_init = socket.socket.__init__
def _init_no_v6(self, family=-1, *args, **kwargs):
    if family == socket.AF_INET6:
        raise OSError(errno.EAFNOSUPPORT, "Address family not supported by protocol")
    _init(self, family, *args, **kwargs)
socket.socket.__init__ = _init_no_v6
""",
    "EADDRNOTAVAIL when ::1 is bound": """import errno, socket
_bind = socket.socket.bind
def _bind_no_v6(self, address):
    if self.family == socket.AF_INET6:
        raise OSError(errno.EADDRNOTAVAIL, "Can't assign requested address")
    return _bind(self, address)
socket.socket.bind = _bind_no_v6
""",
}


@pytest.mark.parametrize("failure", sorted(NO_IPV6), ids=sorted(NO_IPV6))
def test_a_family_the_host_does_not_have_counts_as_free_and_the_other_is_still_checked(box, failure):
    site = box.root / "site"
    site.mkdir()
    (site / "sitecustomize.py").write_text(NO_IPV6[failure])
    probe = subprocess.run([str(box.bin / "python3"), "-c", "import socket; socket.socket(socket.AF_INET6).bind(('::1', 0))"],
                           env=box.env(PYTHONPATH=str(site)), capture_output=True)
    assert probe.returncode != 0, "the shim must make IPv6 fail, or this test says nothing"
    box.read(box.endpoint())
    box.forward("0")
    done = box.run(TUNNEL, MAX_ATTEMPTS="1", PYTHONPATH=str(site))
    assert done.rc == 0 and box.calls("ssh") == [READ, jump_forward(box)], "no IPv6 here, so nothing on ::1 can be in the way: " + done.out
    with socket.socket() as dev_server:
        dev_server.bind(("127.0.0.1", box.port))
        dev_server.listen(1)
        done = box.run(TUNNEL, MAX_ATTEMPTS="2", PYTHONPATH=str(site))
        assert done.rc == 1 and done.lines == [busy_line(box)], "127.0.0.1 is still checked: " + done.out


def test_a_port_the_last_forward_left_in_time_wait_on_ipv6_is_free(box):
    """The IPv6 twin of the TIME_WAIT test: ssh closes its forward's listener on ::1 as well, and that side's TIME_WAIT must not read busy."""
    listener, box.port = ipv6_loopback_listener(reuse=False)
    with listener:
        client = socket.create_connection(("::1", box.port))
        served, _ = listener.accept()
        served.close()                   # the side that closes first keeps the TIME_WAIT
        client.close()
    box.read(box.endpoint())
    box.forward("0")
    done = box.run(TUNNEL, MAX_ATTEMPTS="1")
    assert done.rc == 0 and box.calls("ssh") == [READ, jump_forward(box)], done.out


# ---- tunnel.sh: the endpoint is read again while a forward runs ---------------------------------------------------------------------------
# P7-C fix 1, item 1. A forward whose node or port the server has left stays up and points at nothing: ExitOnForwardFailure covers only the
# local bind and ServerAlive only a dead ssh connection. RECHECK_S=1 is the smallest the setting allows, so that these runs take seconds.

def forward_pid(box: Box, n: int) -> int:
    return int((box.stubs / "forward.{}.pid".format(n)).read_text())


def when_reads_reach(box: Box, counts: List[int], timeout: float = 30.0) -> Dict[int, float]:
    """The monotonic time at which the n-th read of the endpoint was seen, for each n in counts (polled every 10 ms)."""
    seen: Dict[int, float] = {}
    deadline = time.monotonic() + timeout
    while len(seen) < len(counts) and time.monotonic() < deadline:
        have = len(reads(box))
        for n in counts:
            if have >= n and n not in seen:
                seen[n] = time.monotonic()
        time.sleep(0.01)
    assert len(seen) == len(counts), "only {} reads were made".format(len(reads(box)))
    return seen


CHANGES = [("another node", {"host": "gx02"}), ("another port", {"port": 43002}), ("another mode", {"mode": "public"})]


@pytest.mark.parametrize("change", [c for _, c in CHANGES], ids=[n for n, _ in CHANGES])
def test_an_endpoint_that_changed_stops_the_forward_and_the_next_attempt_starts_at_once(box, change):
    """(a) The server was requeued onto another node (or, with --port 0, another port), or came back in another mode. The first forward is
    stopped and reaped before the second one starts, which goes to the new place, and no RETRY_SLEEP pause comes between them: the 30 s pause
    would outlast the 20 s the run is given."""
    old = {"host": "gx01", "port": 43001, "mode": "private"}
    new = dict(old, **change)
    box.read(box.endpoint(**old), n=1)                # the first read
    box.read(box.endpoint(**new))                     # every later one: the re-read, then the next attempt's own read
    box.forward("sleep", n=1)                         # the first forward stays up until it is stopped
    box.forward("0", n=2)                             # the second one ends at once, so that MAX_ATTEMPTS ends the script
    done = box.run(TUNNEL, timeout=20, RECHECK_S="1", RETRY_SLEEP="30", MAX_ATTEMPTS="2")
    assert done.rc == 0, done.out
    assert box.calls("ssh") == [READ, jump_forward(box, "gx01", 43001), READ, READ, jump_forward(box, new["host"], new["port"])], done.out
    assert not alive(forward_pid(box, 1)), "the first forward was stopped"
    assert (box.stubs / "forward.2.prev").read_text().strip() == "gone", "...and was gone before the second one started: reaped, not orphaned"
    assert [l for l in done.lines if "forwarding" in l and "gx01:43001" in l], done.out
    changed = [l for l in done.lines if "endpoint changed" in l]
    assert len(changed) == 1 and re.match(STAMPED + r"endpoint changed: gx01:43001 \(mode private\) -> {}:{} \(mode {}\); restarting the forward$".format(
        new["host"], new["port"], new["mode"]), changed[0]), done.out
    assert len([l for l in done.lines if "forwarding" in l]) == 2 and re.match(
        STAMPED + r"stopping after 2 attempt\(s\) \(MAX_ATTEMPTS\)$", done.lines[-1]), "the attempt after a change is counted as an attempt"


def test_the_re_check_serves_the_login_path_too(box):
    """The watch does not care which way the forward goes: with VIA=login a changed endpoint is followed by a forward to the new node."""
    box.read(box.endpoint(host="gx01", port=43001), n=1)
    box.read(box.endpoint(host="gx02", port=43001))
    box.forward("sleep", n=1)
    box.forward("0", n=2)
    done = box.run(TUNNEL, timeout=20, VIA="login", RECHECK_S="1", RETRY_SLEEP="30", MAX_ATTEMPTS="2")
    assert done.rc == 0, done.out
    assert box.calls("ssh") == [READ, login_forward(box, "gx01", 43001), READ, READ, login_forward(box, "gx02", 43001)], done.out
    assert not alive(forward_pid(box, 1)) and [l for l in done.lines if "endpoint changed: gx01:43001" in l]


GONE = [("a read that fails in the file", "", 1), ("a read that prints nothing", "", 0),
        ("a failing read that still printed an endpoint", Box.endpoint(host="gx01", port=43001), 1)]


@pytest.mark.parametrize("text,rc", [(t, r) for _, t, r in GONE], ids=[n for n, _, _ in GONE])
def test_an_endpoint_file_that_is_gone_stops_the_forward_prints_the_hint_and_retries_after_the_pause(box, text, rc):
    """(b) A clean stop of the server removes the file first thing, so "gone" is the normal sign of a server that is stopping. The forward
    is stopped, one line says so, then today's path: the hint, then RETRY_SLEEP, then the file is read again."""
    box.read(box.endpoint(host="gx01", port=43001), n=1)
    box.read(text, rc=rc, n=2)                                        # the re-read
    box.read(box.endpoint(host="gx02", port=43002))                   # the retry, after the pause
    box.forward("sleep", n=1)
    box.forward("0", n=2)
    run = box.start(TUNNEL, RECHECK_S="1", RETRY_SLEEP="2", MAX_ATTEMPTS="2")
    seen = when_reads_reach(box, [2, 3])
    assert run.wait(30) == 0, run.out
    assert seen[3] - seen[2] >= 1.8, "RETRY_SLEEP (2 s) lay between the re-read and the retry: {:.2f} s".format(seen[3] - seen[2])
    assert not alive(forward_pid(box, 1)), "the forward was stopped"
    assert box.calls("ssh") == [READ, jump_forward(box, "gx01", 43001), READ, READ, jump_forward(box, "gx02", 43002)]
    stamped = [re.sub(STAMPED, "", l) for l in run.lines]
    assert stamped[1:3] == ["endpoint file is gone; stopped the forward to gx01:43001",
                            "no endpoint yet; is the job running? (ssh fake-login.invalid squeue --me)"], run.out
    assert all(re.match(STAMPED, l) for l in run.lines[:4]), "one timestamped line each"
    assert [l for l in run.lines if "forwarding" in l and "gx02:43002" in l]


UNREADABLE_LATER = [("not json", "<html>login</html>\n", "html"), ("a command in the host", '{"host": "gx01;touch PWNED", "port": 43211}', "PWNED"),
                    ("port 70000", '{"host": "gx01", "port": 70000}', "70000"), ("the old host:port sketch", "gx01:43211\n", "43211")]


@pytest.mark.parametrize("content,leak", [(c, l) for _, c, l in UNREADABLE_LATER], ids=[n for n, _, _ in UNREADABLE_LATER])
def test_an_endpoint_that_becomes_unreadable_stops_the_forward_and_takes_the_unreadable_path(box, content, leak):
    """The file is untrusted at every read: what fails the checks stops the forward, is never printed and never run."""
    box.read(box.endpoint(host="gx01", port=43001), n=1)
    box.read(content, n=2)
    box.read(box.endpoint(host="gx02", port=43002))
    box.forward("sleep", n=1)
    box.forward("0", n=2)
    done = box.run(TUNNEL, timeout=30, RECHECK_S="1", RETRY_SLEEP="0", MAX_ATTEMPTS="2")
    assert done.rc == 0, done.out
    assert not alive(forward_pid(box, 1))
    stamped = [re.sub(STAMPED, "", l) for l in done.lines]
    assert stamped[1:3] == ["endpoint file is unreadable; stopped the forward to gx01:43001", "endpoint unreadable; retrying"], done.out
    assert leak not in done.out and not (box.root / "PWNED").exists()
    assert box.calls("ssh") == [READ, jump_forward(box, "gx01", 43001), READ, READ, jump_forward(box, "gx02", 43002)]


def test_a_re_read_that_fails_in_ssh_itself_keeps_the_forward_and_is_tried_again_at_the_next_interval(box):
    """(c) ssh exit 255 is a connection problem, not news about the file; ServerAlive already ends a forward whose connection is dead."""
    box.read(box.endpoint(), n=1)
    box.read("", rc=255, n=2)                         # two re-reads fail in ssh itself...
    box.read("", rc=255, n=3)
    box.read(box.endpoint())                          # ...and the later ones are fine
    box.forward("sleep")
    run = box.start(TUNNEL, RECHECK_S="1", MAX_ATTEMPTS="0")
    when_reads_reach(box, [5])
    pid = forward_pid(box, 1)
    assert run.proc.poll() is None and alive(pid), "the forward was kept"
    assert len(forwards(box)) == 1, "no second forward"
    failed = [l for l in run.lines if "re-check failed" in l]
    assert len(failed) == 2 and all(re.match(STAMPED + r"endpoint re-check failed \(ssh exit 255\); keeping the forward to gx01\.example:43211$", l)
                                    for l in failed), "one line per failed re-read: " + run.out
    assert not [l for l in run.lines if re.search(r"gone|changed|unreadable|stopped the forward", l)], run.out
    run.signal(signal.SIGTERM)
    assert run.wait(10) == 143 and not alive(pid)


UNCHANGED = [("the same file", lambda: Box.endpoint()),
             ("the same place from a restarted server (another pid and start time)",
              lambda: json.dumps({"host": "gx01.example", "port": 43211, "mode": "private", "pid": 777, "started_at": "2026-10-11T09:00:00+00:00"}) + "\n")]


@pytest.mark.parametrize("again", [a for _, a in UNCHANGED], ids=[n for n, _ in UNCHANGED])
def test_an_endpoint_that_is_unchanged_keeps_the_forward_across_re_reads_and_starts_nothing_else(box, again):
    """(d) Two re-reads and more, one forward, nothing printed about the file, and every read is the same BatchMode read with its stdin on
    /dev/null (the harness hands the script an open pipe, so a read that inherited it would show)."""
    box.read(box.endpoint(), n=1)
    box.read(again())
    box.forward("sleep")
    run = box.start(TUNNEL, stdin=subprocess.PIPE, RECHECK_S="1", MAX_ATTEMPTS="0")
    when_reads_reach(box, [4])                        # the first read and three re-reads
    pid = forward_pid(box, 1)
    assert run.proc.poll() is None and alive(pid)
    assert len(forwards(box)) == 1 and len([l for l in run.lines if "forwarding" in l]) == 1, run.out
    assert all(c == READ for c in reads(box)), "the re-read is the read"
    assert set(stdin_kinds(box, "read")) == {"devnull"}, "the re-read, too, takes its stdin from /dev/null"
    assert [re.sub(STAMPED, "", l) for l in run.lines] == ["forwarding localhost:{} -> gx01.example:43211 via jump (mode private)".format(box.port)]
    run.signal(signal.SIGTERM)
    assert run.wait(10) == 143 and not alive(pid)


def test_re_reads_are_not_attempts_so_MAX_ATTEMPTS_never_ends_a_forward_that_is_still_wanted(box):
    """The count in MAX_ATTEMPTS is of attempts (a read, then its forward for as long as it lives), not of re-reads: with MAX_ATTEMPTS=1
    a live forward is re-read three times and is still there."""
    box.read(box.endpoint())
    box.forward("sleep")
    run = box.start(TUNNEL, RECHECK_S="1", MAX_ATTEMPTS="1")
    when_reads_reach(box, [4])
    assert run.proc.poll() is None and "stopping after" not in run.out
    run.signal(signal.SIGTERM)
    assert run.wait(10) == 143


def test_a_forward_that_dies_between_two_re_reads_is_noticed_within_about_a_second(box):
    """(e) The forward ends by itself 1.5 s after it started. The first re-read is 30 s away, and the loop must not wait for it: the next
    attempt (its own read, no re-read) follows within about a second, and the whole run is done in seconds."""
    box.read(box.endpoint())
    box.forward("after:1.5:255", n=1)
    box.forward("0", n=2)
    started = time.monotonic()
    done = box.run(TUNNEL, timeout=20, RECHECK_S="30", RETRY_SLEEP="0", MAX_ATTEMPTS="2")
    elapsed = time.monotonic() - started
    assert done.rc == 0, done.out
    assert box.calls("ssh") == [READ, jump_forward(box), READ, jump_forward(box)], "a read for each attempt, and no re-read in between"
    assert elapsed < 12, "{:.1f} s".format(elapsed)


@pytest.mark.parametrize("sig,status,recheck", [(signal.SIGTERM, 143, "1"), (signal.SIGINT, 130, "1"), (signal.SIGHUP, 129, "1"),
                                                (signal.SIGTERM, 143, "30")], ids=["TERM", "INT", "HUP", "TERM mid-interval"])
def test_a_signal_during_the_re_check_loop_stops_everything_and_leaves_no_process(box, sig, status, recheck):
    """(f) The first three: well into the cycle (the first read and two re-reads are behind it); the last: in the middle of a 30 s interval."""
    box.read(box.endpoint())
    box.forward("sleep")
    run = box.start(TUNNEL, RECHECK_S=recheck, MAX_ATTEMPTS="0")
    wait_for(lambda: (box.stubs / "forward.1.pid").exists())
    if recheck == "1":
        when_reads_reach(box, [3])
    pid = forward_pid(box, 1)
    assert alive(pid)
    run.signal(sig)
    assert run.wait(10) == status, run.out
    assert not alive(pid), "the ssh was stopped with the script"
    wait_for(lambda: not run.group_alive(), 5)
    assert run.lines[-1] == "stopped"


def test_ctrl_c_on_a_terminal_during_the_re_check_loop_stops_tunnel_sh_and_the_ssh_it_runs(box):
    box.read(box.endpoint())
    box.forward("sleep")
    status, out, left = ctrl_c_on_a_terminal(box, TUNNEL, lambda: len(reads(box)) >= 3, RECHECK_S="1", MAX_ATTEMPTS="0")
    assert status == 130 and out.rstrip().endswith("stopped"), (status, out)
    assert not alive(forward_pid(box, 1)) and not left


# ---- tunnel.sh: settings ---------------------------------------------------------------------------------------------------------------

RECHECK_MESSAGE = "RECHECK_S must be a whole number of seconds from 1 to 99999"
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
    ({"RECHECK_S": "0"}, RECHECK_MESSAGE),
    ({"RECHECK_S": "05"}, RECHECK_MESSAGE),
    ({"RECHECK_S": "-1"}, RECHECK_MESSAGE),
    ({"RECHECK_S": "1.5"}, RECHECK_MESSAGE),
    ({"RECHECK_S": "soon"}, RECHECK_MESSAGE),
    ({"RECHECK_S": "100000"}, RECHECK_MESSAGE),
    ({"RECHECK_S": "1e3"}, RECHECK_MESSAGE),
    ({"RECHECK_S": "99999999999999999999"}, RECHECK_MESSAGE),
    ({"RECHECK_S": "5 6"}, RECHECK_MESSAGE),
    ({"RECHECK_S": " 5"}, RECHECK_MESSAGE),
    ({"RECHECK_S": "5;touch PWNED"}, RECHECK_MESSAGE),
]


@pytest.mark.parametrize("setting,message", BAD_SETTINGS, ids=["{}={}".format(*next(iter(s.items()))) for s, _ in BAD_SETTINGS])
def test_a_setting_that_is_not_one_is_refused_with_status_2_before_anything_runs(box, setting, message):
    box.read(box.endpoint())
    done = box.run(TUNNEL, **setting)
    assert done.rc == 2 and len(done.lines) == 1 and done.lines[0].startswith(message), done.out
    assert box.calls("ssh") == [] and not (box.root / "PWNED").exists()


@pytest.mark.parametrize("value", ["1", "9", "10", "60", "99999"])
def test_RECHECK_S_takes_a_whole_number_from_1_to_99999(box, value):
    done = box.run(TUNNEL, MAX_ATTEMPTS="1", RECHECK_S=value)
    assert done.rc == 0 and [l for l in done.lines if "no endpoint yet" in l], done.out


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
    """What the server's /healthz answers from now on (None: nothing is listening, as curl reports for a refused connection)."""
    if body is None:
        box.unset("healthz")
    else:
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
    """P7-C fix 1, item 4. Not "a PATH on which cloudflared happens to be missing": the PATH is one directory of symlinks to exactly the tools
    the script runs (curl, python3, sleep) and the fake curl runs (cat), so no cloudflared can be found whatever is installed on this machine."""
    tools = box.toolbox("curl", "python3", "sleep", "cat")
    path = str(tools)
    assert shutil.which("cloudflared", path=path) is None, "the test's own PATH must not hold a cloudflared"
    assert sorted(p.name for p in tools.iterdir()) == ["cat", "curl", "python3", "sleep"] and all(p.is_symlink() for p in tools.iterdir())
    public(box, '{"mode": "public"}')
    done = box.run(PUBLIC_DEMO, PATH=path)
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


# ---- public_demo.sh: the watch while the link is up ----------------------------------------------------------------------------------
# P7-C fix 1, item 3. A server that comes back in private mode serves MIMIC-derived content (R6, DUA) and must never sit behind a live public
# link, so the mode is asked again every DEMO_RECHECK_S seconds, and the link is closed on anything but JSON whose mode is exactly public.

DEMO_RECHECK_MESSAGE = "DEMO_RECHECK_S must be a whole number of seconds from 1 to 99999"


def start_public_demo(box: Box, **env: Optional[str]) -> Tuple[Run, int]:
    """The demo, with a cloudflared that stays up until it is stopped and a server that says public, up and running: -> (run, cloudflared's pid)."""
    box.install("cloudflared", FAKE_CLOUDFLARED)
    public(box, '{"mode": "public"}')
    box.set("cloudflared.stay", "")
    run = box.start(PUBLIC_DEMO, **env)
    wait_for(lambda: (box.stubs / "cloudflared.pid").exists() and "fake cloudflared is up" in run.out)
    return run, int((box.stubs / "cloudflared.pid").read_text())


# (case, what /healthz answers after the flip (None: nothing listening), curl's status, a word the one line must say)
CLOSES = [
    ("the mode flips to private", '{"status": "ok", "mode": "private", "default_model": "m"}', 0, "private mode"),
    ("the server becomes unreachable", None, 7, "no usable answer"),
    ("the answer times out", None, 28, "no usable answer"),
    ("the answer is empty", "", 0, "no usable answer"),
    ("the answer is not JSON", "<html>502 Bad Gateway</html>", 0, "no usable answer"),
    ("public is only in another field", '{"mode": "private", "default_model": "public"}', 0, "private mode"),
    ("a mode that only starts with public", '{"mode": "publicish"}', 0, "no usable answer"),
]


@pytest.mark.parametrize("body,rc,word", [(b, r, w) for _, b, r, w in CLOSES], ids=[n for n, _, _, _ in CLOSES])
def test_the_public_link_is_closed_within_about_DEMO_RECHECK_S_of_the_server_ceasing_to_say_public(box, body, rc, word):
    run, pid = start_public_demo(box, DEMO_RECHECK_S="2")
    flipped = time.monotonic()
    public(box, body, rc)
    assert run.wait(15) == 1, run.out
    assert time.monotonic() - flipped < 2 + 6, "within about DEMO_RECHECK_S (2 s)"
    refusing = [l for l in run.lines if l.startswith("refusing:")]
    assert len(refusing) == 1 and run.lines[-1] == refusing[0], "one line, and the last: " + run.out
    assert "the public link is closed" in refusing[0] and word in refusing[0], refusing[0]
    assert "cloudflared stopped:" not in run.out, "the one line says it all"
    assert not alive(pid), "cloudflared was stopped"
    wait_for(lambda: not run.group_alive(), 5)
    assert len(box.calls("cloudflared")) == 1, "and not started again"


def test_a_server_that_stays_public_keeps_the_link_across_checks(box):
    run, pid = start_public_demo(box, DEMO_RECHECK_S="1")
    wait_for(lambda: len(box.calls("curl")) >= 4, 15)             # the first look, then three checks
    assert run.proc.poll() is None and alive(pid), run.out
    assert len(box.calls("cloudflared")) == 1 and not [l for l in run.lines if "refusing" in l or "cloudflared stopped" in l], run.out
    assert all(c == CURL + [healthz_url(box)] for c in box.calls("curl")), "every check is the same probe"
    run.signal(signal.SIGTERM)
    assert run.wait(10) == 143 and not alive(pid)
    assert run.lines[-1] == "cloudflared stopped: the public link is closed."


def test_one_bad_answer_is_enough_the_link_is_closed_even_though_the_next_answer_would_be_public(box):
    """It fails closed: no second chance, no count of bad answers in a row."""
    box.set("healthz.2.rc", "28")           # the first check (the second look at /healthz) times out; every other answer says public
    run, pid = start_public_demo(box, DEMO_RECHECK_S="1")
    assert run.wait(15) == 1, run.out
    assert len(box.calls("curl")) == 2 and not alive(pid), "the third answer, which would have said public, was never asked for"
    assert run.lines[-1].startswith("refusing: ") and "the public link is closed" in run.lines[-1]


def test_a_cloudflared_that_ends_by_itself_is_noticed_within_about_a_second_and_its_status_is_the_scripts(box):
    """The watch looks at cloudflared once a second: one that dies 1.5 s in is not left until the next check, 30 s away."""
    box.install("cloudflared", FAKE_CLOUDFLARED)
    public(box, '{"mode": "public"}')
    box.set("cloudflared.after", "1.5")
    box.set("cloudflared.rc", "3")
    started = time.monotonic()
    done = box.run(PUBLIC_DEMO, timeout=20, DEMO_RECHECK_S="30")
    assert done.rc == 3 and done.lines[-1] == "cloudflared stopped: the public link is closed.", done.out
    assert time.monotonic() - started < 10, "{:.1f} s".format(time.monotonic() - started)
    assert len(box.calls("curl")) == 1, "it ended before the first check"


@pytest.mark.parametrize("sig,status", [(signal.SIGTERM, 143), (signal.SIGINT, 130), (signal.SIGHUP, 129)], ids=["TERM", "INT", "HUP"])
def test_a_signal_during_the_watch_stops_cloudflared_and_leaves_no_process_behind(box, sig, status):
    run, pid = start_public_demo(box, DEMO_RECHECK_S="1")
    wait_for(lambda: len(box.calls("curl")) >= 3, 15)             # well into the watch: the first look and two checks are behind it
    assert alive(pid)
    run.signal(sig)
    assert run.wait(10) == status, run.out
    assert not alive(pid), "cloudflared was stopped with the script"
    wait_for(lambda: not run.group_alive(), 5)
    assert run.lines[-1] == "cloudflared stopped: the public link is closed."


def test_ctrl_c_on_a_terminal_during_the_watch_stops_public_demo_sh_and_the_cloudflared_it_runs(box):
    box.install("cloudflared", FAKE_CLOUDFLARED)
    public(box, '{"mode": "public"}')
    box.set("cloudflared.stay", "")
    status, out, left = ctrl_c_on_a_terminal(box, PUBLIC_DEMO, lambda: len(box.calls("curl")) >= 3, DEMO_RECHECK_S="1")
    assert status == 130 and "cloudflared stopped: the public link is closed." in out, (status, out)
    assert not alive(int((box.stubs / "cloudflared.pid").read_text())) and not left


@pytest.mark.parametrize("bad", ["0", "05", "-1", "1.5", "soon", "100000", "1e3", "99999999999999999999", "5 6", " 5", "5;touch PWNED"])
def test_public_demo_refuses_a_check_interval_that_is_not_one_before_asking_anything(box, bad):
    done = box.run(PUBLIC_DEMO, DEMO_RECHECK_S=bad)
    assert done.rc == 2 and done.lines == [DEMO_RECHECK_MESSAGE], done.out
    assert box.calls("curl") == [] and not (box.root / "PWNED").exists()


@pytest.mark.parametrize("value", ["1", "10", "99999"])
def test_DEMO_RECHECK_S_takes_a_whole_number_from_1_to_99999(box, value):
    box.install("cloudflared", FAKE_CLOUDFLARED)
    public(box, '{"mode": "public"}')
    done = box.run(PUBLIC_DEMO, DEMO_RECHECK_S=value)
    assert done.rc == 0 and "cloudflared stopped" in done.out, done.out
