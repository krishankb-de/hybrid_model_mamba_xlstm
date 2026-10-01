"""CHAT_UI_PLAN.md P0-G: the Mac-side door to the cluster (scripts/chat_remote.sh) and its setup job.

Nothing here touches the real cluster. Every test that runs the script runs a COPY of it inside a throwaway
git tree, with `ssh` and `rsync` replaced by stub executables that record their argv and CHAT_CLUSTER_ENV
pointing at a fake env file; the assertions are on what the stubs recorded. Synthetic data only (R7).
"""
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
REMOTE_SH = REPO_ROOT / "scripts" / "chat_remote.sh"
SETUP_SH = REPO_ROOT / "scripts" / "chat_cluster_setup_h100.sh"
ENV_EXAMPLE = REPO_ROOT / "scripts" / "chat_cluster.env.example"
EXCLUDE = REPO_ROOT / ".rsync-exclude-chat"
WRAPPER = "scripts/chat_cluster_setup_h100.sh"

# The Mac's /bin/bash is 3.2: the oldest shell chat_remote.sh has to work in.
BASH = "/bin/bash" if os.path.exists("/bin/bash") else "bash"

# The fixed list of lines `summary` may show (R7). Pinned here on purpose: widening it is a decision.
SUMMARY_PATTERN = (
    r"^(RESULT |\[(probe|golden|gallery|labels|gates|server|setup|compile)\]|=== |ERROR|Traceback"
    r"|[A-Za-z]*Error:|[[:space:]]*(Elapsed \(wall|Maximum resident)|  Missing keys|  prefix_k =)"
)

FAKE_ENV = (
    "CLUSTER_HOST=fakehost\n"
    "CLUSTER_REPO=/fake/hybrid_chat_ui\n"
    "MAIN_REPO=/fake/hybrid_mamba_xlstm\n"
    "SCRATCH_ROOT=/fake/scratch/hybrid_xmamba_h100\n"
)

# One stub serves as both ssh and rsync: it appends "@@ <name>", one argument per line, "@@END" to calls.log,
# then prints $STUB_DIR/<name>.stdout when that file exists, and exits 0.
STUB = """#!/bin/bash
{ echo "@@ __NAME__"; for a in "$@"; do printf '%s\\n' "$a"; done; echo "@@END"; } >> "$STUB_DIR/calls.log"
if [ -f "$STUB_DIR/__NAME__.stdout" ]; then cat "$STUB_DIR/__NAME__.stdout"; fi
exit 0
"""

STAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z [0-9a-f]{40} (clean|dirty)$")


class Sandbox:
    """A throwaway git tree holding copies of the chat scripts, with ssh/rsync stubbed out."""

    def __init__(self, root: Path, env_text: str = FAKE_ENV):
        self.repo = root / "repo"
        self.stubs = root / "stubs"
        self.bin = root / "bin"
        (self.repo / "scripts").mkdir(parents=True)
        self.stubs.mkdir()
        self.bin.mkdir()
        shutil.copy(str(REMOTE_SH), str(self.repo / "scripts" / "chat_remote.sh"))
        shutil.copy(str(SETUP_SH), str(self.repo / WRAPPER))
        shutil.copy(str(EXCLUDE), str(self.repo / ".rsync-exclude-chat"))
        for name in ("ssh", "rsync"):
            stub = self.bin / name
            stub.write_text(STUB.replace("__NAME__", name))
            stub.chmod(0o755)
        self.env_file = root / "cluster.env"
        self.env_file.write_text(env_text)
        self.git("init", "-q")
        self.git("add", "-A")
        self.git("commit", "-q", "-m", "init")

    @staticmethod
    def _clean_env() -> Dict[str, str]:
        # no GIT_* leaking in (a hook running pytest would otherwise point git at the real repo)
        env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
        env["GIT_CONFIG_GLOBAL"] = os.devnull
        env["GIT_CONFIG_NOSYSTEM"] = "1"
        return env

    def git(self, *args: str) -> str:
        cmd = ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", "-c", "commit.gpgsign=false",
               "-c", "core.hooksPath=%s" % os.devnull] + list(args)
        return subprocess.run(cmd, cwd=str(self.repo), env=self._clean_env(), check=True,
                              capture_output=True, text=True).stdout

    def run(self, *args: str, env_file: Optional[str] = None) -> subprocess.CompletedProcess:
        env = self._clean_env()
        env["PATH"] = "%s%s%s" % (self.bin, os.pathsep, env["PATH"])
        env["STUB_DIR"] = str(self.stubs)
        env["CHAT_CLUSTER_ENV"] = env_file if env_file is not None else str(self.env_file)
        return subprocess.run([BASH, str(self.repo / "scripts" / "chat_remote.sh")] + list(args),
                              cwd=str(self.repo), env=env, stdin=subprocess.DEVNULL, timeout=60,
                              capture_output=True, text=True)

    def calls(self) -> List[List[str]]:
        """Every stub invocation so far, in order, as [name, arg1, arg2, ...]."""
        log = self.stubs / "calls.log"
        if not log.exists():
            return []
        calls, current = [], None
        for line in log.read_text().splitlines():
            if line.startswith("@@ "):
                current = [line[3:]]
            elif line == "@@END":
                calls.append(current)
                current = None
            else:
                current.append(line)
        return calls


@pytest.fixture
def sandbox(tmp_path):
    return Sandbox(tmp_path)


def _code(path: Path) -> str:
    """The script minus its comment lines (the header names the flag it refuses to use)."""
    return "\n".join(l for l in path.read_text().splitlines() if not l.lstrip().startswith("#"))


# ── static checks ─────────────────────────────────────────────────────────────

def test_scripts_pass_bash_syntax_check():
    for script in (REMOTE_SH, SETUP_SH):
        done = subprocess.run([BASH, "-n", str(script)], capture_output=True, text=True)
        assert done.returncode == 0, "%s: %s" % (script.name, done.stderr)


def test_chat_remote_never_deletes_or_removes():
    """R8: the cluster side is additive only. No rsync --delete (or --remove-*), no rm."""
    code = _code(REMOTE_SH)
    assert "--delete" not in code and "--remove" not in code
    assert not re.search(r"(^|[\s;&|(])rm(\s|$)", code, re.M)


def test_exclude_file_lists_what_must_not_cross():
    lines = [l for l in EXCLUDE.read_text().splitlines() if l and not l.startswith("#")]
    for entry in ("/outputs/", "/results/", "/.venv/", "/venv/", "/logs/", "/data/", "/.git/",
                  "/ISBI Paper /", "*.parquet"):
        assert entry in lines, entry
    # root directories are anchored: an unanchored `data/` would also drop app/data/ (legal-AI job 2588703)
    assert not [l for l in lines if l in ("outputs/", "results/", ".venv/", "venv/", "logs/", "data/", ".git/")]


def test_local_cluster_files_are_gitignored_and_the_example_is_not():
    ignored = [l.strip() for l in (REPO_ROOT / ".gitignore").read_text().splitlines()]
    assert "scripts/chat_cluster.env" in ignored and ".sync_stamp" in ignored
    if not (REPO_ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    def check_ignore(path: str) -> int:
        return subprocess.run(["git", "check-ignore", "-q", path], cwd=str(REPO_ROOT)).returncode
    assert check_ignore("scripts/chat_cluster.env") == 0
    assert check_ignore(".sync_stamp") == 0
    assert check_ignore("scripts/chat_cluster.env.example") == 1


def test_env_example_defines_every_key_and_keeps_the_chat_dir_out_of_the_thesis_checkout():
    """R8: CLUSTER_REPO is where rsync writes; MAIN_REPO is the user's thesis checkout and is never a target."""
    out = subprocess.run([BASH, "-c", 'source "$1"; printf "%s\\n" "$CLUSTER_HOST" "$CLUSTER_REPO" "$MAIN_REPO" "$SCRATCH_ROOT"',
                          "_", str(ENV_EXAMPLE)], capture_output=True, text=True, check=True).stdout.splitlines()
    host, cluster_repo, main_repo, scratch = out
    assert host == "hpi-hpc"
    assert cluster_repo and main_repo and scratch
    assert cluster_repo != main_repo
    assert not cluster_repo.startswith(main_repo + "/") and not main_repo.startswith(cluster_repo + "/")
    assert scratch.endswith("/hybrid_xmamba_h100")


# ── env file handling ─────────────────────────────────────────────────────────

def test_sync_refuses_without_the_env_file(sandbox):
    done = sandbox.run("sync", env_file="/nonexistent")
    assert done.returncode == 1
    assert "missing /nonexistent" in done.stderr
    assert sandbox.calls() == []
    assert not (sandbox.repo / ".sync_stamp").exists()


def test_an_env_file_missing_a_key_is_refused(sandbox):
    incomplete = sandbox.env_file.parent / "incomplete.env"
    incomplete.write_text("CLUSTER_HOST=fakehost\nCLUSTER_REPO=/fake/hybrid_chat_ui\nMAIN_REPO=/fake/main\n")
    done = sandbox.run("queue", env_file=str(incomplete))
    assert done.returncode != 0
    assert "SCRATCH_ROOT" in done.stderr
    assert sandbox.calls() == []


# ── sync ──────────────────────────────────────────────────────────────────────

def test_sync_makes_the_log_dir_then_rsyncs_without_delete(sandbox):
    done = sandbox.run("sync")
    assert done.returncode == 0, done.stderr
    assert sandbox.calls() == [
        ["ssh", "fakehost", "mkdir -p '/fake/hybrid_chat_ui/logs'"],
        ["rsync", "-az", "--exclude-from=%s/.rsync-exclude-chat" % sandbox.repo, "%s/" % sandbox.repo,
         "fakehost:/fake/hybrid_chat_ui/"],
    ]
    assert done.stdout.startswith("[sync] %s -> fakehost:/fake/hybrid_chat_ui/ (" % sandbox.repo)


def test_sync_stamp_is_time_head_and_cleanliness(sandbox):
    head = sandbox.git("rev-parse", "HEAD").strip()
    stamp_file = sandbox.repo / ".sync_stamp"

    assert sandbox.run("sync").returncode == 0
    stamp = stamp_file.read_text().strip()
    assert STAMP_RE.match(stamp), stamp
    assert stamp.split()[1] == head and stamp.endswith(" clean")

    # an untracked file outside the five source dirs does not make the stamp dirty
    (sandbox.repo / "notes.txt").write_text("scratch\n")
    assert sandbox.run("sync").returncode == 0
    assert stamp_file.read_text().strip().endswith(" clean")

    # one inside scripts/ does
    (sandbox.repo / "scripts" / "extra.sh").write_text("echo hi\n")
    assert sandbox.run("sync").returncode == 0
    dirty = stamp_file.read_text().strip()
    assert STAMP_RE.match(dirty) and dirty.split()[1] == head and dirty.endswith(" dirty")


# ── submit ────────────────────────────────────────────────────────────────────

def test_submit_sends_one_quoted_sbatch_command_and_passes_the_job_id_through(sandbox):
    (sandbox.stubs / "ssh.stdout").write_text("2589358\n")
    done = sandbox.run("submit", WRAPPER, "MAIN_REPO=/x", "--", "--time=00:05:00")
    assert done.returncode == 0, done.stderr
    assert done.stdout == "2589358\n"
    assert sandbox.calls() == [[
        "ssh", "fakehost",
        "cd '/fake/hybrid_chat_ui' && env 'MAIN_REPO=/x' sbatch --parsable '--time=00:05:00' "
        "'scripts/chat_cluster_setup_h100.sh'",
    ]]


def test_submit_quotes_each_value_and_stops_env_parsing_at_double_dash(sandbox):
    done = sandbox.run("submit", WRAPPER, "A=1", "B=two words", "--", "--time=00:05:00", "--job-name=x y")
    assert done.returncode == 0, done.stderr
    assert sandbox.calls()[0][2] == (
        "cd '/fake/hybrid_chat_ui' && env 'A=1' 'B=two words' sbatch --parsable '--time=00:05:00' "
        "'--job-name=x y' 'scripts/chat_cluster_setup_h100.sh'"
    )


def test_submit_with_no_options_survives_empty_arrays_on_bash_3(sandbox):
    """`set -u` plus an empty array is an unbound-variable error in bash 3.2 unless the idiom is right."""
    done = sandbox.run("submit", WRAPPER)
    assert done.returncode == 0, done.stderr
    assert " ".join(sandbox.calls()[0][2].split()) == (
        "cd '/fake/hybrid_chat_ui' && env sbatch --parsable 'scripts/chat_cluster_setup_h100.sh'"
    )


def test_quote_refuses_a_single_quote_and_nothing_reaches_the_cluster(sandbox):
    for args in ((WRAPPER, "A=it's"), (WRAPPER, "--", "--comment=it's")):
        done = sandbox.run("submit", *args)
        assert done.returncode == 2, args
        assert "single quote" in done.stderr, args
    assert sandbox.calls() == []


def test_submit_refuses_a_missing_wrapper_and_stray_arguments(sandbox):
    done = sandbox.run("submit", "scripts/nope.sh")
    assert done.returncode == 2 and "no such wrapper" in done.stderr
    done = sandbox.run("submit", WRAPPER, "positional")
    assert done.returncode == 2 and "unexpected argument" in done.stderr
    done = sandbox.run("submit")
    assert done.returncode != 0 and "usage" in done.stderr
    assert sandbox.calls() == []


# ── state / queue / usage ─────────────────────────────────────────────────────

def test_state_is_one_sacct_line(sandbox):
    assert sandbox.run("state", "2589358").returncode == 0
    assert sandbox.calls() == [[
        "ssh", "fakehost",
        "sacct -j '2589358' --format=JobID,JobName%28,State,Elapsed,MaxRSS,ExitCode,NodeList -P",
    ]]


def test_state_and_summary_refuse_a_missing_or_unquotable_argument_before_calling_ssh(sandbox):
    """quote() runs inside $(...): unless its result is assigned, its `exit 2` (and a missing-argument `:?` error)
    is swallowed, and ssh would be handed `sacct -j  --format=...` or a bare `grep` that waits on stdin."""
    for args, code in ((("state",), 1), (("state", "12'34"), 2), (("summary",), 1), (("summary", "logs/it's.log"), 2)):
        done = sandbox.run(*args)
        assert done.returncode == code, (args, done.stderr)
    assert sandbox.calls() == [], "nothing may reach the cluster"


def test_queue_is_squeue_me(sandbox):
    assert sandbox.run("queue").returncode == 0
    assert sandbox.calls() == [["ssh", "fakehost", "squeue --me"]]


def test_unknown_command_prints_usage_and_exits_2(sandbox):
    done = sandbox.run("bogus")
    assert done.returncode == 2
    for sub in ("sync", "submit", "state", "summary", "queue"):
        assert "chat_remote.sh %s" % sub in done.stdout, sub
    assert sandbox.calls() == []


# ── summary (R7) ──────────────────────────────────────────────────────────────

def test_summary_sends_only_a_grep_of_the_fixed_pattern(sandbox):
    assert sandbox.run("summary", "logs/chat_setup_2589358.log").returncode == 0
    assert sandbox.calls() == [[
        "ssh", "fakehost",
        "grep -aE '%s' '/fake/hybrid_chat_ui/logs/chat_setup_2589358.log' | tail -n 200" % SUMMARY_PATTERN,
    ]]


def test_summary_masks_long_digit_runs_and_ids_but_not_job_ids(sandbox):
    (sandbox.stubs / "ssh.stdout").write_text(
        "[probe] study s50414267 subject 10000032 done\n"
        "=== chat setup: job=2589358 node=cn01 ===\n"
        "[gallery] 0a1b2c3d-1a2b3c4d-5e6f7a8b-9c0d1e2f-3a4b5c6d kept\n"
        "RESULT {\"n\": 12345678901234}\n"
    )
    done = sandbox.run("summary", "logs/x.log")
    assert done.returncode == 0, done.stderr
    assert done.stdout.splitlines() == [
        "[probe] study s<num> subject <num> done",          # 8+ digit runs never reach the terminal
        "=== chat setup: job=2589358 node=cn01 ===",        # a 7-digit job id is left alone
        "[gallery] <id> kept",                              # dicom-style ids are replaced whole
        'RESULT {"n": <num>}',
    ]


def test_summary_cuts_lines_to_300_characters(sandbox):
    (sandbox.stubs / "ssh.stdout").write_text("=== " + "x" * 500 + "\n")
    done = sandbox.run("summary", "logs/x.log")
    assert done.returncode == 0, done.stderr
    assert done.stdout == "=== " + "x" * 296 + "\n"


def _remote_command(sandbox: Sandbox) -> str:
    return sandbox.calls()[-1][2]


def test_summary_pattern_passes_summary_lines_and_drops_everything_else(tmp_path):
    """R7, end to end: run the exact command `summary` sends to the cluster, locally, against a synthetic log."""
    cluster = tmp_path / "cluster"
    (cluster / "logs").mkdir(parents=True)
    keep = [
        "=== chat setup: node=cn01 job=2589358 ===",
        "[setup] ok checkpoints/last.ckpt",
        "[probe] n=4 median_ms=12.5",
        "[golden] pass 3/3",
        "[gallery] rows=1000",
        "[labels] micro_f1=0.47",
        "[gates] ok",
        "[server] listening on 127.0.0.1:8000",
        "[compile] skipped",
        'RESULT {"rouge_l": 0.19}',
        "ERROR missing something",
        "Traceback (most recent call last):",
        "ValueError: bad value",
        "RuntimeError: boom",
        "\tElapsed (wall clock) time (h:mm:ss or m:ss): 0:12.30",
        "\tMaximum resident set size (kbytes): 123456",
        "  Missing keys: []",
        "  prefix_k = 32",
    ]
    drop = [
        "The heart size is normal. No pleural effusion or pneumothorax.",
        "s50414267 PA view, no acute cardiopulmonary process",
        '  File "x.py", line 3, in <module>',
        "[other] some debug line",
        "[setup",
        " [setup] indented",
        "warning: not an error line",
        "a report that says ERROR inside the text",
        "XRESULT {}",
        "Study s50414267 Error: x",
        "",
    ]
    interleaved = []
    for i in range(max(len(keep), len(drop))):
        if i < len(drop):
            interleaved.append(drop[i])
        if i < len(keep):
            interleaved.append(keep[i])
    (cluster / "logs" / "x.log").write_text("\n".join(interleaved) + "\n")

    box = Sandbox(tmp_path / "box", env_text=FAKE_ENV.replace("/fake/hybrid_chat_ui", str(cluster)))
    assert box.run("summary", "logs/x.log").returncode == 0
    shown = subprocess.run([BASH, "-c", _remote_command(box)], capture_output=True, text=True)
    assert shown.returncode == 0, shown.stderr
    assert shown.stdout.splitlines() == keep


def test_summary_is_capped_at_the_last_200_matching_lines(tmp_path):
    cluster = tmp_path / "cluster"
    (cluster / "logs").mkdir(parents=True)
    (cluster / "logs" / "x.log").write_text("".join("[probe] %d\n[other] %d\n" % (i, i) for i in range(250)))
    box = Sandbox(tmp_path / "box", env_text=FAKE_ENV.replace("/fake/hybrid_chat_ui", str(cluster)))
    assert box.run("summary", "logs/x.log").returncode == 0
    shown = subprocess.run([BASH, "-c", _remote_command(box)], capture_output=True, text=True).stdout.splitlines()
    assert shown == ["[probe] %d" % i for i in range(50, 250)]


# ── the exclude file under a real rsync ───────────────────────────────────────

@pytest.mark.skipif(shutil.which("rsync") is None, reason="rsync not installed")
def test_exclude_file_under_a_real_rsync_and_without_touching_the_destination_symlinks(tmp_path):
    """The files that must not cross stay home, nested look-alikes (app/data/) do cross, and rsync neither
    replaces the destination's symlinks into the thesis checkout nor writes through them."""
    src, dst, main = tmp_path / "src", tmp_path / "dst", tmp_path / "main"

    def touch(base: Path, rel: str) -> None:
        path = base / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x\n")

    cross = ["scripts/a.sh", "hybrid_xmamba/m.py", "app/data/keep.json", "app/logs/keep.txt",
             "app/outputs/keep.txt", "configs/c.yaml"]
    stay = [".git/config", "venv/bin/python", ".venv/x", ".venv_chexbert/x", ".chat_deps/x",
            ".chat_deps_chexbert/x", "outputs/run/checkpoints/last.ckpt", "results/r/hyps.txt", "logs/l.log",
            "data/raw.txt", "cluster/c.txt", "output_willi_server/o.txt", "hpi_results_logs/h.txt",
            "ISBI Paper /fig.png", ".superpowers/sdd/x.md", "scripts/chat_cluster.env", "pkg/__pycache__/m.pyc",
            ".pytest_cache/c", "pkg.egg-info/PKG-INFO", "w.ckpt", "w.pt", "w.pth", "d.parquet", "a.npy", "a.npz",
            "s.db", ".DS_Store"]
    for rel in cross + stay:
        touch(src, rel)

    # the destination as the setup job leaves it: symlinks into the thesis checkout, an old log
    for name in ("outputs", "results", ".venv", ".venv_chexbert"):
        (main / name).mkdir(parents=True)
        touch(main, "%s/precious.txt" % name)
    dst.mkdir()
    for name in ("outputs", "results", ".venv", ".venv_chexbert"):
        os.symlink(str(main / name), str(dst / name))
    touch(dst, "logs/old.log")

    done = subprocess.run(["rsync", "-az", "--exclude-from=%s" % EXCLUDE, "%s/" % src, "%s/" % dst],
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr

    arrived = []
    for base, _dirs, files in os.walk(str(dst)):    # followlinks=False: never descends into the symlinks
        for name in files:
            full = os.path.join(base, name)
            if not os.path.islink(full):
                arrived.append(os.path.relpath(full, str(dst)))
    assert sorted(arrived) == sorted(cross + ["logs/old.log"])
    for name in ("outputs", "results", ".venv", ".venv_chexbert"):
        assert (dst / name).is_symlink() and os.readlink(str(dst / name)) == str(main / name)
        assert sorted(p.name for p in (main / name).iterdir()) == ["precious.txt"]
