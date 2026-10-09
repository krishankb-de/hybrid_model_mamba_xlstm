"""The harness that rehearses a chat wrapper for real, in a temp tree standing in for the cluster (CHAT_UI_PLAN.md P5-B, extracted
from tests/test_build_retrieval_gallery.py at the P5-B review's request once a second wrapper, P5-C's, needed it).

A JobBox lays out the pieces a wrapper meets on the cluster: CLUSTER_REPO (repo/, with COPIES of the wrapper and of the scripts it runs,
so the real code runs inside the rehearsal), the thesis checkout (main/) behind repo/outputs, CHAT_HOME (chat/), the dataset (data/), a
`.sync_stamp`, a venv `activate`, and a stub `python` and `df` first on PATH. The stub python records every call to python.calls (one
`@@` line, then one argument per line); what else it does is the subclass's PYTHON_STUB. A subclass names its wrapper and scripts and
fills in what is specific to it (populate, base_env); the tests do the rest. Synthetic data only (R7).

Not a test module: pytest does not collect it (no test_ prefix), and test files import from it as `tests.wrapper_rehearsal`.
"""
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Pattern, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
BASH = "/bin/bash" if os.path.exists("/bin/bash") else "bash"     # the Mac's /bin/bash is 3.2: the oldest shell to support

CLUSTER_USER = "krishankumar.bhushan"
STAMP = "2026-10-09T20:00:00Z 3f2a9c41d7e86b05a1c4e9d3b7f60285ac9e1d47 clean"

# The first thing a stub python does: note the call (one `@@` line, then one argument per line).
RECORD_CALL = "{ echo \"@@\"; for a in \"$@\"; do printf '%s\\n' \"$a\"; done; } >> \"$STUB_DIR/python.calls\"\n"

DF_STUB =('#!/bin/bash\nfor last; do :; done\necho "$last" >> "$STUB_DIR/df.calls"\n'
           '[ -d "$last" ] || { echo "df: $last: No such file or directory" >&2; exit 1; }\n'
           'echo "Filesystem 1024-blocks Used Available Capacity Mounted on"\n'
           'echo "/dev/fake 100000000 1000 ${FAKE_DF_KB:-50000000} 1% /fake"\n')


def safe_line(tag: str) -> Pattern:
    """The first words of every line a wrapper may print (R7): `===`, its own `[tag]`, `RESULT` or `ERROR`."""
    return re.compile(r"^(=== |\[" + re.escape(tag) + r"\] |RESULT |ERROR)")


def snapshot(root: Path) -> List[tuple]:
    """Every path under `root` with size and mtime: what a job that only READS a tree leaves exactly as it was (R8)."""
    found = []
    for base, dirs, files in os.walk(str(root)):
        for name in dirs + files:
            full = os.path.join(base, name)
            st = os.lstat(full)
            found.append((os.path.relpath(full, str(root)), st.st_size, st.st_mtime_ns))
    return sorted(found)


class JobBox:
    """A wrapper run for real in a temp tree standing in for the cluster. Subclasses set WRAPPER and SCRIPTS and, where they differ
    from the default, VENV, PYTHON_STUB, RUN_DIRS and DATA_FILES; populate() adds what only that wrapper needs and base_env() the
    environment it is run with."""

    WRAPPER = ""                  # the wrapper under test: a file name in scripts/
    SCRIPTS = ()                  # the scripts it runs, copied next to it, so that they run for real
    VENV = ".venv"                # the venv directory its VENV_ACTIVATE default names
    PYTHON_STUB = "#!/bin/bash\n" + RECORD_CALL + "exec \"$REAL_PYTHON\" \"$@\"\n"
    RUN_DIRS = ()                 # run directories under main/outputs, each with an empty checkpoints/last.ckpt
    DATA_FILES = ()               # empty files made in data/
    SLURM_CPUS = "16"

    def __init__(self, root: Path, stamp: Optional[str] = STAMP):
        self.root = root
        self.repo, self.main, self.chat, self.data = root / "repo", root / "main", root / "chat", root / "data"
        self.stubs, self.bin, self.scratch = root / "stubs", root / "bin", root / "scratch"
        (self.repo / "scripts").mkdir(parents=True)
        for name in (self.WRAPPER,) + tuple(self.SCRIPTS):
            shutil.copy(str(REPO_ROOT / "scripts" / name), str(self.repo / "scripts" / name))
        (self.repo / self.VENV / "bin").mkdir(parents=True)
        (self.repo / self.VENV / "bin" / "activate").write_text("")
        (self.repo / "logs").mkdir()
        if stamp is not None:
            (self.repo / ".sync_stamp").write_text(stamp + "\n")
        (self.main / "outputs").mkdir(parents=True)
        for rel in self.RUN_DIRS:
            ckpt = self.main / "outputs" / rel / "checkpoints" / "last.ckpt"
            ckpt.parent.mkdir(parents=True)
            ckpt.write_bytes(b"")
        (self.repo / "outputs").symlink_to(self.main / "outputs", target_is_directory=True)
        for directory in (self.chat, self.data, self.stubs, self.bin, self.scratch):
            directory.mkdir()
        for name in self.DATA_FILES:
            (self.data / name).write_bytes(b"")
        python = self.bin / "python"
        python.write_text(self.PYTHON_STUB)
        python.chmod(0o755)
        df = self.bin / "df"       # as the real one: it fails for a directory that does not exist; free space is FAKE_DF_KB (default 50 GB)
        df.write_text(DF_STUB)
        df.chmod(0o755)
        self.populate()

    def populate(self) -> None:
        """What only this wrapper's rehearsal needs, once the common tree is there."""

    def base_env(self) -> Dict[str, str]:
        return {"PATH": "{}:/usr/bin:/bin".format(self.bin), "HOME": str(self.root / "home"), "USER": CLUSTER_USER,
                "SLURM_SUBMIT_DIR": str(self.repo), "SLURM_JOB_ID": "1234567", "SLURM_CPUS_PER_TASK": self.SLURM_CPUS,
                "CHAT_HOME": str(self.chat), "SCRATCH_ROOT": str(self.scratch), "STUB_DIR": str(self.stubs),
                "REAL_PYTHON": sys.executable, "PYTHONDONTWRITEBYTECODE": "1"}

    def run_script(self, name: str, **extra_env: Optional[str]) -> subprocess.CompletedProcess:
        """One wrapper of the box, run the way sbatch runs it: cwd outside the repo, SLURM_SUBMIT_DIR set, an environment of its own. A
        value of None removes a variable."""
        env = self.base_env()
        env.update(extra_env)
        env = {k: v for k, v in env.items() if v is not None}
        # stdout and stderr share one pipe, like the single SLURM log: whatever bash itself complains about counts too.
        return subprocess.run([BASH, str(self.repo / "scripts" / name)], cwd=str(self.root), env=env,
                              stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=240)

    def run(self, **extra_env: Optional[str]) -> subprocess.CompletedProcess:
        return self.run_script(self.WRAPPER, **extra_env)

    def calls(self) -> List[List[str]]:
        """Every call the stub python saw, in order, as [argument, ...]."""
        log = self.stubs / "python.calls"
        if not log.exists():
            return []
        return [rec.splitlines() for rec in log.read_text().split("@@\n") if rec.strip()]


def job_lines(done: subprocess.CompletedProcess) -> List[str]:
    return done.stdout.splitlines()


def results(lines: Sequence[str]) -> List[dict]:
    return [json.loads(l[len("RESULT "):]) for l in lines if l.startswith("RESULT ")]


def real_stamp(root: Path, dirty: bool) -> str:
    """The .sync_stamp that `scripts/chat_remote.sh sync` writes, from the real script in a throwaway git tree with ssh and rsync
    stubbed (tests/test_chat_remote.py's Sandbox): the wrapper has to read what the producer writes."""
    from tests.test_chat_remote import Sandbox
    box = Sandbox(root)
    if dirty:
        script = box.repo / "scripts" / "chat_remote.sh"
        script.write_text(script.read_text() + "\n# touched\n")
    done = box.run("sync")
    assert done.returncode == 0, done.stdout + done.stderr
    return (box.repo / ".sync_stamp").read_text().strip()
