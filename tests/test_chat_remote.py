"""CHAT_UI_PLAN.md P0-G: the Mac-side door to the cluster (scripts/chat_remote.sh) and its setup job.

Nothing here touches the real cluster. Every test that runs chat_remote.sh runs a COPY of it inside a throwaway
git tree, with `ssh` and `rsync` replaced by stub executables that record their argv and CHAT_CLUSTER_ENV
pointing at a fake env file; the assertions are on what the stubs recorded (the summary-filter tests make the ssh
stub run the command it is sent against a synthetic log instead, to reach the fixed grep and the mask). The setup
wrapper is run for real in a temp tree (fake thesis checkout, stub `uv`, stub venv pythons). Synthetic data only (R7).
"""
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
REMOTE_SH = REPO_ROOT / "scripts" / "chat_remote.sh"
SETUP_SH = REPO_ROOT / "scripts" / "chat_cluster_setup_h100.sh"
ENV_EXAMPLE = REPO_ROOT / "scripts" / "chat_cluster.env.example"
EXCLUDE = REPO_ROOT / ".rsync-exclude-chat"
WRAPPER = "scripts/chat_cluster_setup_h100.sh"

# The Mac's /bin/bash is 3.2: the oldest shell these scripts have to work in.
BASH = "/bin/bash" if os.path.exists("/bin/bash") else "bash"

# The fixed list of lines `summary` may show (R7). Pinned here on purpose: widening it is a decision.
# P1-C widened the exception shape from an undotted `<Name>Error:` to a dotted class name ending in Error or Exception.
SUMMARY_PATTERN = (
    r"^(RESULT |\[(probe|golden|gallery|labels|gates|server|setup|compile)\]|=== |ERROR|Traceback"
    r"|([A-Za-z_][A-Za-z0-9_.]*)?(Error|Exception):|[[:space:]]*(Elapsed \(wall|Maximum resident)|  Missing keys|  prefix_k =)"
)


def fake_env(**overrides: str) -> str:
    """The text of a cluster.env; values are double-quoted so a test can put a single quote in one."""
    values = {
        "CLUSTER_HOST": "fakehost",
        "CLUSTER_REPO": "/fake/hybrid_chat_ui",
        "MAIN_REPO": "/fake/hybrid_mamba_xlstm",
        "SCRATCH_ROOT": "/fake/scratch/hybrid_xmamba_h100",
    }
    values.update(overrides)
    return "".join('%s="%s"\n' % (key, value) for key, value in values.items())


# One stub serves as both ssh and rsync: it appends "@@ <name>", one argument per line, "@@END" to calls.log,
# prints $STUB_DIR/<name>.stdout when that file exists, and exits with the code in <name>.rc (default 0).
STUB = """#!/bin/bash
{ echo "@@ __NAME__"; for a in "$@"; do printf '%s\\n' "$a"; done; echo "@@END"; } >> "$STUB_DIR/calls.log"
if [ -f "$STUB_DIR/__NAME__.stdout" ]; then cat "$STUB_DIR/__NAME__.stdout"; fi
if [ -f "$STUB_DIR/__NAME__.rc" ]; then exit "$(cat "$STUB_DIR/__NAME__.rc")"; fi
exit 0
"""

STAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z [0-9a-f]{40} (clean|dirty)$")


class Sandbox:
    """A throwaway git tree holding copies of the chat scripts, with ssh/rsync stubbed out."""

    def __init__(self, root: Path, env_text: Optional[str] = None):
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
        self.env_file.write_text(env_text if env_text is not None else fake_env())
        self.git("init", "-q")
        self.git("add", "-A")
        self.git("commit", "-q", "-m", "init")

    @staticmethod
    def _clean_env() -> Dict[str, str]:
        # No GIT_* leaking in (a hook running pytest would otherwise point git at the real repo), and none of the
        # settings chat_remote.sh reads: an exported SCRATCH_ROOT would satisfy `: "${SCRATCH_ROOT:?}"`.
        drop = ("CLUSTER_HOST", "CLUSTER_REPO", "MAIN_REPO", "SCRATCH_ROOT", "CHAT_CLUSTER_ENV")
        env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_") and k not in drop}
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


def test_every_quote_call_is_the_right_hand_side_of_an_assignment():
    """quote()'s `exit 2` is lost unless errexit sees the assignment: inline in a larger word it sends an empty argument."""
    lines = [l for l in _code(REMOTE_SH).splitlines() if "$(quote" in l]
    assert len(lines) >= 8
    for line in lines:
        assert re.search(r'\w+\+?=\(?"\$\(quote ', line), line


def test_exclude_file_lists_what_must_not_cross():
    lines = [l for l in EXCLUDE.read_text().splitlines() if l and not l.startswith("#")]
    for entry in ("/outputs/", "/results/", "/.venv/", "/venv/", "/logs/", "/data/", "/.git/",
                  "/ISBI Paper /", "*.parquet", "/.claude/", ".physionet_session", "*.physionet_session"):
        assert entry in lines, entry
    # root directories are anchored: an unanchored `data/` would also drop app/data/ (legal-AI job 2588703)
    assert not [l for l in lines if l in ("outputs/", "results/", ".venv/", "venv/", "logs/", "data/", ".git/",
                                          ".claude/")]


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


def test_an_env_file_missing_a_key_is_refused_whatever_the_callers_shell_exports(sandbox, monkeypatch):
    monkeypatch.setenv("SCRATCH_ROOT", "/exported/by/the/callers/shell")
    incomplete = sandbox.env_file.parent / "incomplete.env"
    incomplete.write_text("CLUSTER_HOST=fakehost\nCLUSTER_REPO=/fake/hybrid_chat_ui\nMAIN_REPO=/fake/main\n")
    done = sandbox.run("queue", env_file=str(incomplete))
    assert done.returncode != 0
    assert "SCRATCH_ROOT" in done.stderr
    assert sandbox.calls() == []


@pytest.mark.parametrize("cluster_repo, main_repo", [
    ("/fake/main", "/fake/main"),                 # the same tree
    ("/fake/main/", "/fake/main"),                # a trailing slash
    ("/fake//main", "/fake/main"),                # a doubled slash
    ("/fake/main/chat", "/fake/main"),            # CLUSTER_REPO inside MAIN_REPO
    ("/fake/chat", "/fake/chat/main"),            # MAIN_REPO inside CLUSTER_REPO
    ("/fake", "/fake/main"),                      # a parent
    ("chat", "/fake/main"),                       # relative: could resolve to anywhere
    ("/fake/x/../main", "/fake/main"),            # dot-dot
    ("/fake/./main", "/fake/main"),               # dot
])
def test_cluster_repo_and_main_repo_must_be_separate_absolute_trees(tmp_path, cluster_repo, main_repo):
    """R8's central invariant: rsync writes into CLUSTER_REPO, so it must never be the thesis checkout or hold it."""
    box = Sandbox(tmp_path, env_text=fake_env(CLUSTER_REPO=cluster_repo, MAIN_REPO=main_repo))
    for args in (("sync",), ("submit", WRAPPER), ("summary", "logs/x.log"), ("queue",)):
        done = box.run(*args)
        assert done.returncode == 1, (args, done.stderr)
        assert "CLUSTER_REPO" in done.stderr and "MAIN_REPO" in done.stderr
    assert box.calls() == [], "refused before any ssh or rsync"
    assert not (box.repo / ".sync_stamp").exists()


@pytest.mark.parametrize("cluster_repo, main_repo", [
    ("/fake/hybrid", "/fake/hybrid_mamba_xlstm"),               # a name prefix is not a path prefix
    ("/fake/hybrid_mamba_xlstm_chat", "/fake/hybrid_mamba_xlstm"),
    ("/fake/chat/", "/fake/main/"),
])
def test_separate_trees_are_accepted(tmp_path, cluster_repo, main_repo):
    box = Sandbox(tmp_path, env_text=fake_env(CLUSTER_REPO=cluster_repo, MAIN_REPO=main_repo))
    assert box.run("queue").returncode == 0


# ── sync ──────────────────────────────────────────────────────────────────────

def test_sync_sends_the_tree_without_the_stamp_and_then_the_stamp_alone_last(sandbox):
    done = sandbox.run("sync")
    assert done.returncode == 0, done.stderr
    repo = sandbox.repo
    assert sandbox.calls() == [
        ["ssh", "fakehost", "mkdir -p '/fake/hybrid_chat_ui/logs'"],
        ["rsync", "-az", "--exclude-from=%s/.rsync-exclude-chat" % repo, "--exclude=/.sync_stamp", "%s/" % repo,
         "fakehost:/fake/hybrid_chat_ui/"],
        ["rsync", "-az", "%s/.sync_stamp" % repo, "fakehost:/fake/hybrid_chat_ui/"],
    ]
    assert done.stdout.startswith("[sync] %s -> fakehost:/fake/hybrid_chat_ui/ (" % repo)


@pytest.mark.skipif(shutil.which("rsync") is None, reason="rsync not installed")
def test_sync_arguments_under_a_real_rsync_ship_the_stamp_only_in_the_last_call(sandbox, tmp_path):
    """Replay the argv the script recorded against a real rsync, with a local directory as the destination."""
    assert sandbox.run("sync").returncode == 0
    dest = tmp_path / "dest"
    dest.mkdir()
    replays = [c[1:] for c in sandbox.calls() if c[0] == "rsync"]
    assert len(replays) == 2
    tops = []
    for args in replays:
        local = [a.replace("fakehost:/fake/hybrid_chat_ui/", "%s/" % dest) for a in args]
        done = subprocess.run(["rsync"] + local, capture_output=True, text=True)
        assert done.returncode == 0, done.stderr
        tops.append(sorted(p.name for p in dest.iterdir()))
    assert ".sync_stamp" not in tops[0] and "scripts" in tops[0] and ".git" not in tops[0]
    assert ".sync_stamp" in tops[1]
    assert (dest / ".sync_stamp").read_text() == (sandbox.repo / ".sync_stamp").read_text()


def test_a_failed_tree_transfer_never_sends_or_rewrites_the_stamp(sandbox):
    """An interrupted transfer must not leave a fresh `clean` stamp over a partial tree."""
    (sandbox.repo / ".sync_stamp").write_text("OLD STAMP\n")
    (sandbox.stubs / "rsync.rc").write_text("23\n")
    done = sandbox.run("sync")
    assert done.returncode == 23
    assert [c[0] for c in sandbox.calls()] == ["ssh", "rsync"], "no second rsync after a failed first one"
    assert (sandbox.repo / ".sync_stamp").read_text() == "OLD STAMP\n"


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


@pytest.mark.parametrize("arg", [
    "--chdir=/tmp", "--split-string=touch x", "-C=/tmp", "-u=HOME",   # env options, which can run commands on lx01
    "1A=x", "A-B=x", "=x", "A B=x",                                   # not a variable name
])
def test_submit_accepts_only_plain_variable_assignments_as_env_arguments(sandbox, arg):
    done = sandbox.run("submit", WRAPPER, arg)
    assert done.returncode == 2, done.stderr
    assert "unexpected argument" in done.stderr
    assert sandbox.calls() == [], "nothing may reach the cluster"


def test_submit_still_accepts_underscored_and_empty_valued_assignments(sandbox):
    done = sandbox.run("submit", WRAPPER, "_A1=x", "B_2=", "MAIN_REPO=/x")
    assert done.returncode == 0, done.stderr
    assert " ".join(sandbox.calls()[0][2].split()) == (
        "cd '/fake/hybrid_chat_ui' && env '_A1=x' 'B_2=' 'MAIN_REPO=/x' sbatch --parsable "
        "'scripts/chat_cluster_setup_h100.sh'"
    )


def test_quote_refuses_a_single_quote_and_nothing_reaches_the_cluster(sandbox):
    for args in ((WRAPPER, "A=it's"), (WRAPPER, "--", "--comment=it's")):
        done = sandbox.run("submit", *args)
        assert done.returncode == 2, args
        assert "single quote" in done.stderr, args
    assert sandbox.calls() == []


def test_a_wrapper_name_with_a_single_quote_is_refused_before_sbatch(sandbox):
    """quote() used inline inside the ssh command string would swallow its refusal and send a bare `sbatch`."""
    (sandbox.repo / "scripts" / "it's.sh").write_text("#!/bin/bash\n")
    done = sandbox.run("submit", "scripts/it's.sh")
    assert done.returncode == 2 and "single quote" in done.stderr
    assert sandbox.calls() == []


@pytest.mark.parametrize("args", [("sync",), ("submit", WRAPPER), ("summary", "logs/x.log")])
def test_a_single_quote_in_cluster_repo_is_refused_by_every_command_that_quotes_it(tmp_path, args):
    box = Sandbox(tmp_path, env_text=fake_env(CLUSTER_REPO="/fake/it's"))
    done = box.run(*args)
    assert done.returncode == 2, done.stderr
    assert "single quote" in done.stderr
    assert box.calls() == []
    assert not (box.repo / ".sync_stamp").exists(), "refused before any side effect"


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


def test_unknown_command_prints_the_whole_usage_header_and_exits_2(sandbox):
    done = sandbox.run("bogus")
    assert done.returncode == 2
    header = []
    for line in REMOTE_SH.read_text().splitlines()[1:]:
        if not line.startswith("#"):
            break
        header.append(re.sub(r"^# ?", "", line))
    assert done.stdout.splitlines() == header, "the whole header comment, not a cut at a fixed line"
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
        "[probe] study s12345678 subject p87654321 done\n"
        "=== chat setup: job=2589358 node=cn01 ===\n"
        "[gallery] 0a1b2c3d-1a2b3c4d-5e6f7a8b-9c0d1e2f-3a4b5c6d kept\n"
        "[gallery] 12345678-12345678-12345678-12345678-12345678 kept\n"
        "RESULT {\"n\": 12345678901234}\n"
    )
    done = sandbox.run("summary", "logs/x.log")
    assert done.returncode == 0, done.stderr
    assert done.stdout.splitlines() == [
        "[probe] study s<num> subject p<num> done",         # 8+ digit runs never reach the terminal
        "=== chat setup: job=2589358 node=cn01 ===",        # a 7-digit job id is left alone
        "[gallery] <id> kept",                              # dicom-style ids are replaced whole
        "[gallery] <id> kept",                              # also when every group happens to be digits
        'RESULT {"n": <num>}',
    ]


def test_summary_keeps_the_digits_of_decimals_and_masks_bare_runs(sandbox):
    """R2 numbers (ROUGE-L, drift, step sizes) carry 8+ fractional digits; only whole digit runs are ids."""
    (sandbox.stubs / "ssh.stdout").write_text(
        'RESULT {"rouge_l": 0.1899234512, "eps": 3.814697265625e-06, "delta": -0.1234567890123}\n'
        "[probe] study s12345678 subject p87654321 job 2589358\n"
        "[probe] ids 12345678 and 123456789012, x=12345678\n"
    )
    done = sandbox.run("summary", "logs/x.log")
    assert done.returncode == 0, done.stderr
    assert done.stdout.splitlines() == [
        'RESULT {"rouge_l": 0.1899234512, "eps": 3.814697265625e-06, "delta": -0.1234567890123}',
        "[probe] study s<num> subject p<num> job 2589358",
        "[probe] ids <num> and <num>, x=<num>",
    ]


def test_summary_blanks_exception_text_but_keeps_the_class_name(sandbox):
    """Exception messages can echo report text. Everything after `<Name>Error:` / `<Name>Exception:` goes."""
    (sandbox.stubs / "ssh.stdout").write_text(
        "ValueError: could not parse 'SYNTHETIC findings'\n"
        "[probe] KeyError: 'SYNTHETIC'\n"
        "json.decoder.JSONDecodeError: Expecting 'SYNTHETIC' at line 1\n"
        "HTTPException: SYNTHETIC detail\n"
        "RuntimeError: first KeyError: second 'SYNTHETIC'\n"
        "Error: SYNTHETIC\n"
        "[probe] rows=4 median_ms=12.5\n"
    )
    done = sandbox.run("summary", "logs/x.log")
    assert done.returncode == 0, done.stderr
    assert done.stdout.splitlines() == [
        "ValueError: <msg>",
        "[probe] KeyError: <msg>",
        "json.decoder.JSONDecodeError: <msg>",
        "HTTPException: <msg>",
        "RuntimeError: <msg>",
        "Error: <msg>",
        "[probe] rows=4 median_ms=12.5",                    # no exception marker: untouched
    ]
    assert "SYNTHETIC" not in done.stdout


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
        "s12345678 PA view, no acute cardiopulmonary process",
        '  File "x.py", line 3, in <module>',
        "[other] some debug line",
        "[setup",
        " [setup] indented",
        "warning: not an error line",
        "a report that says ERROR inside the text",
        "XRESULT {}",
        "Study s12345678 Error: x",
        "",
    ]
    interleaved = []
    for i in range(max(len(keep), len(drop))):
        if i < len(drop):
            interleaved.append(drop[i])
        if i < len(keep):
            interleaved.append(keep[i])
    (cluster / "logs" / "x.log").write_text("\n".join(interleaved) + "\n")

    box = Sandbox(tmp_path / "box", env_text=fake_env(CLUSTER_REPO=str(cluster)))
    assert box.run("summary", "logs/x.log").returncode == 0
    shown = subprocess.run([BASH, "-c", _remote_command(box)], capture_output=True, text=True)
    assert shown.returncode == 0, shown.stderr
    assert shown.stdout.splitlines() == keep


def test_summary_is_capped_at_the_last_200_matching_lines(tmp_path):
    cluster = tmp_path / "cluster"
    (cluster / "logs").mkdir(parents=True)
    (cluster / "logs" / "x.log").write_text("".join("[probe] %d\n[other] %d\n" % (i, i) for i in range(250)))
    box = Sandbox(tmp_path / "box", env_text=fake_env(CLUSTER_REPO=str(cluster)))
    assert box.run("summary", "logs/x.log").returncode == 0
    shown = subprocess.run([BASH, "-c", _remote_command(box)], capture_output=True, text=True).stdout.splitlines()
    assert shown == ["[probe] %d" % i for i in range(50, 250)]


RUNNING_SSH = """#!/bin/bash
# stands in for ssh: $1 is the host, $2 the command line the script sends. Run that line here, so `summary` reaches
# its fixed grep and its mask end to end instead of leaving a recorded argv behind.
exec bash -c "$2"
"""


def _summary_of(tmp_path: Path, log_lines: List[str]) -> List[str]:
    """`summary` end to end over a synthetic log: the script's own grep pattern, then its own mask (R7: synthetic data only)."""
    cluster = tmp_path / "cluster"
    (cluster / "logs").mkdir(parents=True)
    (cluster / "logs" / "x.log").write_text("\n".join(log_lines) + "\n")
    box = Sandbox(tmp_path / "box", env_text=fake_env(CLUSTER_REPO=str(cluster)))
    (box.bin / "ssh").write_text(RUNNING_SSH)
    done = box.run("summary", "logs/x.log")
    assert done.returncode == 0, done.stderr
    return done.stdout.splitlines()


def test_summary_keeps_dotted_exception_names_and_blanks_their_text(tmp_path):
    """A failed job has to show WHICH exception it died of. The filter used to keep an exception line only when the class
    name was undotted at the start of the line, so `sqlite3.OperationalError:` and `urllib.error.URLError:` were dropped
    and the summary showed a bare `Traceback`. Dotted names pass now; the mask still blanks everything after the marker,
    because the message can echo report text."""
    shown = _summary_of(tmp_path, [
        "Traceback (most recent call last):",
        "sqlite3.OperationalError: database is locked SYNTHETIC",
        "urllib.error.URLError: <urlopen error x> SYNTHETIC",
        "torch.cuda.OutOfMemoryError: SYNTHETIC findings",
        "app.errors.ChatException: SYNTHETIC findings",
        "ValueError: SYNTHETIC findings",           # undotted: kept before and after
        "HTTPException: SYNTHETIC findings",
        "Exception: SYNTHETIC findings",
        "Error: SYNTHETIC findings",
    ])
    assert shown == [
        "Traceback (most recent call last):",
        "sqlite3.OperationalError: <msg>",
        "urllib.error.URLError: <msg>",
        "torch.cuda.OutOfMemoryError: <msg>",
        "app.errors.ChatException: <msg>",
        "ValueError: <msg>",
        "HTTPException: <msg>",
        "Exception: <msg>",
        "Error: <msg>",
    ]


@pytest.mark.parametrize("line", [
    "os.path.join: something",                      # dotted, but not an exception name
    "module.error_handler: SYNTHETIC findings",     # `error` in lower case
    "sqlite3.OperationalErrors: SYNTHETIC",         # a plural is not an exception class
    "see urllib.error.URLError: SYNTHETIC",         # not at the start of the line
    "Study s12345678.Error: SYNTHETIC",             # a space inside the name (the marker itself is right)
])
def test_summary_still_drops_dotted_lines_that_are_not_exception_names(tmp_path, line):
    assert _summary_of(tmp_path, ["[probe] before", line, "[probe] after"]) == ["[probe] before", "[probe] after"]


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
             "app/outputs/keep.txt", "app/.claude/keep.md", "configs/c.yaml"]
    stay = [".git/config", "venv/bin/python", ".venv/x", ".venv_chexbert/x", ".chat_deps/x",
            ".chat_deps_chexbert/x", "outputs/run/checkpoints/last.ckpt", "results/r/hyps.txt", "logs/l.log",
            "data/raw.txt", "cluster/c.txt", "output_willi_server/o.txt", "hpi_results_logs/h.txt",
            "ISBI Paper /fig.png", ".superpowers/sdd/x.md", "scripts/chat_cluster.env", "pkg/__pycache__/m.pyc",
            ".pytest_cache/c", "pkg.egg-info/PKG-INFO", "w.ckpt", "w.pt", "w.pth", "d.parquet", "a.npy", "a.npz",
            "s.db", ".DS_Store",
            ".claude/settings.json", ".claude/settings.local.json", ".claude/scheduled_tasks.lock",
            ".physionet_session", "scripts/deep/.physionet_session", "scripts/x.physionet_session"]
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


# ── the setup wrapper, run for real in a temp tree ────────────────────────────

SETUP_INPUTS = [   # what the wrapper checks under MAIN_REPO
    "outputs/h100_report_gen_m3_tower13d_s42/checkpoints/last.ckpt",
    "outputs/h100_report_gen_full_ext_4gpu_tower13d/checkpoints/last.ckpt",
    "outputs/h100_kd_150m_v2_full_data_lr3e6/checkpoints/last.ckpt",
    "results/report_gen_m3_test_split_s42/hyps.txt",
    "results/report_gen_m3_test_split_s42/refs.txt",
    "results/report_gen_m3_test_split_s42/chexbert_labels.json",
    "results/retrieval_floor_test_split/hyps.txt",
]

FAKE_MODULES = {   # what the wrapper's import checks need; the stub pythons put this directory on PYTHONPATH
    "torch.py": '__version__ = "2.0.0+fake"\n',
    "fastapi.py": '__version__ = "0.1.0"\n',
    "uvicorn.py": "",
    "httpx.py": '__version__ = "0.28.0"\n',
    "python_multipart.py": "",
    "sklearn.py": '__version__ = "1.7.2"\n',
    "transformers.py": '__version__ = "4.57.0"\n',
}

PY_STUB = """#!/bin/bash
# stands in for a venv's python: runs the wrapper's own `-c` import check with the test interpreter against fake
# modules. -S keeps the test interpreter's real site-packages out of sight, so a module that is missing is missing.
export PYTHONPATH="${PYTHONPATH:-}:__FAKE__"
exec "__PYTHON__" -S "$@"
"""

UV_STUB = """#!/bin/bash
# stands in for uv: records argv, then plays $STUB_DIR/uv.mode. "ok" fills the --target directory; "fail" leaves what a
# failed `uv pip install --target` leaves behind (the directory holding only .lock) and exits 1.
{ echo "@@"; for a in "$@"; do printf '%s\\n' "$a"; done; } >> "$STUB_DIR/uv.calls"
target=""
while [ $# -gt 0 ]; do if [ "$1" = "--target" ]; then target="$2"; fi; shift; done
mkdir -p "$target"
: > "$target/.lock"
if [ "$(cat "$STUB_DIR/uv.mode")" = "ok" ]; then : > "$target/installed"; exit 0; fi
echo "error: simulated install failure" >&2
exit 1
"""


class SetupBox:
    """The setup wrapper run for real in a temp tree: a fake thesis checkout and dataset, a stub uv, and stub venv
    pythons. HOME points into the tree, so ~/chat_sessions is created there and nowhere else."""

    def __init__(self, root: Path, with_uv: bool = True):
        self.root = root
        self.chat, self.main, self.data = root / "chat", root / "main", root / "data"
        self.home, self.stubs, self.fake = root / "home", root / "stubs", root / "fake_modules"
        for directory in (self.chat, self.data, self.stubs, self.fake, self.home / ".local" / "bin"):
            directory.mkdir(parents=True)
        for rel in SETUP_INPUTS:
            self._write(self.main / rel, "")
        for name in ("train.parquet", "test.parquet"):
            self._write(self.data / name, "")
        for name, body in FAKE_MODULES.items():
            (self.fake / name).write_text(body)
        python = PY_STUB.replace("__FAKE__", str(self.fake)).replace("__PYTHON__", sys.executable)
        for venv in (".venv", ".venv_chexbert"):
            self._write(self.main / venv / "bin" / "python", python, executable=True)
        if with_uv:
            self._write(self.home / ".local" / "bin" / "uv", UV_STUB, executable=True)
        self.set_uv_mode("ok")

    @staticmethod
    def _write(path: Path, text: str, executable: bool = False) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        if executable:
            path.chmod(0o755)

    def set_uv_mode(self, mode: str) -> None:
        (self.stubs / "uv.mode").write_text(mode + "\n")

    def uv_calls(self) -> List[List[str]]:
        log = self.stubs / "uv.calls"
        if not log.exists():
            return []
        return [rec.splitlines() for rec in log.read_text().split("@@\n") if rec.strip()]

    def sentinel(self, overlay: str) -> Path:
        return self.chat / overlay / ".setup_ok"

    def mark_installed(self) -> None:
        for overlay in (".chat_deps", ".chat_deps_chexbert"):
            self._write(self.sentinel(overlay), "done\n")

    def main_listing(self) -> List[tuple]:
        """Every path under the fake thesis checkout with size and mtime: R8, it may only be read."""
        found = []
        for base, dirs, files in os.walk(str(self.main)):
            for name in dirs + files:
                full = os.path.join(base, name)
                st = os.lstat(full)
                found.append((os.path.relpath(full, str(self.main)), st.st_size, st.st_mtime_ns))
        return sorted(found)

    def run(self) -> subprocess.CompletedProcess:
        env = {"PATH": "/usr/bin:/bin", "HOME": str(self.home), "SLURM_SUBMIT_DIR": str(self.chat),
               "SLURM_JOB_ID": "1234567", "MAIN_REPO": str(self.main), "DATA": str(self.data),
               "STUB_DIR": str(self.stubs)}
        return subprocess.run([BASH, str(SETUP_SH)], cwd=str(self.root), env=env, stdin=subprocess.DEVNULL,
                              timeout=120, capture_output=True, text=True)


def _target(call: List[str]) -> str:
    return call[call.index("--target") + 1]


def test_setup_rerun_after_a_failed_install_installs_again_and_writes_the_sentinels(tmp_path):
    """A failed `uv pip install --target` leaves the directory behind (holding only .lock), so the directory proves
    nothing: the install is guarded by a sentinel written only after the install and the import check passed."""
    box = SetupBox(tmp_path)
    main_before = box.main_listing()

    box.set_uv_mode("fail")
    first = box.run()
    assert first.returncode == 1, first.stdout + first.stderr
    assert "[setup] ERROR overlay install failed: main" in first.stdout.splitlines()
    assert (box.chat / ".chat_deps").is_dir() and not box.sentinel(".chat_deps").exists()
    assert not (box.chat / ".chat_deps_chexbert").exists(), "it stops at the first failed install"
    assert len(box.uv_calls()) == 1

    box.set_uv_mode("ok")
    second = box.run()
    assert second.returncode == 0, second.stdout + second.stderr
    assert box.sentinel(".chat_deps").is_file() and box.sentinel(".chat_deps_chexbert").is_file()
    lines = second.stdout.splitlines()
    assert any(l.startswith("[setup] main venv: python ") for l in lines)
    assert any(l.startswith("[setup] chexbert venv: python ") for l in lines)
    assert not [l for l in lines if "ERROR" in l]
    assert lines[-1] == "=== END chat setup ==="
    calls = box.uv_calls()
    assert [_target(c) for c in calls] == [".chat_deps", ".chat_deps", ".chat_deps_chexbert"]
    for call in calls:
        assert call[:2] == ["pip", "install"] and "--python" in call
    assert "python-multipart>=0.0.9" in calls[1] and "httpx>=0.27" in calls[1]
    assert "python-multipart>=0.0.9" not in calls[2] and "httpx>=0.27" not in calls[2]

    third = box.run()
    assert third.returncode == 0, third.stdout + third.stderr
    assert len(box.uv_calls()) == 3, "both sentinels exist: nothing is installed again"
    assert any(l.startswith("[setup] main venv: python ") for l in third.stdout.splitlines()), "still checked"
    assert box.main_listing() == main_before, "R8: the thesis checkout was only read"


def test_setup_needs_uv_only_when_an_install_is_pending(tmp_path):
    if shutil.which("uv", path="/usr/bin:/bin"):
        pytest.skip("a uv sits on the minimal PATH this test relies on")
    pending = SetupBox(tmp_path / "pending", with_uv=False).run()
    assert pending.returncode == 1
    assert "[setup] ERROR uv not found" in pending.stdout.splitlines()

    box = SetupBox(tmp_path / "done", with_uv=False)
    box.mark_installed()
    done = box.run()
    assert done.returncode == 0, done.stdout + done.stderr
    assert "uv not found" not in done.stdout
    assert done.stdout.splitlines()[-1] == "=== END chat setup ==="


def test_setup_exits_nonzero_after_every_check_has_printed_when_an_input_is_missing(tmp_path):
    box = SetupBox(tmp_path)
    box.mark_installed()
    (box.main / "results" / "report_gen_m3_test_split_s42" / "refs.txt").unlink()
    done = box.run()
    assert done.returncode == 1, done.stdout + done.stderr
    lines = done.stdout.splitlines()
    at = lines.index("[setup] ERROR missing report_gen_m3_test_split_s42/refs.txt")
    after = lines[at + 1:]
    assert "[setup] ok report_gen_m3_test_split_s42/chexbert_labels.json" in after, "the later checks still ran"
    assert "[setup] ok data/test.parquet" in after
    assert any(l.startswith("[setup] main venv: python ") for l in after)
    assert any(l.startswith("[setup] chexbert venv: python ") for l in after)
    assert lines[-1] == "=== END chat setup: 1 ERROR line(s) above ==="
    assert "=== END chat setup ===" not in lines


def test_setup_exits_nonzero_when_a_thesis_venv_is_missing(tmp_path):
    box = SetupBox(tmp_path)
    shutil.rmtree(str(box.main / ".venv_chexbert"))
    done = box.run()
    assert done.returncode == 1, done.stdout + done.stderr
    assert "[setup] ERROR .venv_chexbert not found in MAIN_REPO" in done.stdout.splitlines()


@pytest.mark.parametrize("drop, add, passes", [
    (["httpx.py"], {}, False),
    (["python_multipart.py"], {"multipart.py": ""}, True),    # the older package name still counts
    (["python_multipart.py"], {}, False),
])
def test_setup_import_check_covers_httpx_and_python_multipart(tmp_path, drop, add, passes):
    box = SetupBox(tmp_path)
    for name in drop:
        (box.fake / name).unlink()
    for name, body in add.items():
        (box.fake / name).write_text(body)
    done = box.run()
    if passes:
        assert done.returncode == 0, done.stdout + done.stderr
        assert box.sentinel(".chat_deps").is_file()
    else:
        assert done.returncode == 1, done.stdout + done.stderr
        assert "[setup] ERROR overlay check failed: main" in done.stdout.splitlines()
        assert not box.sentinel(".chat_deps").exists(), "the sentinel follows a passing import check, not the install"
        assert not (box.chat / ".chat_deps_chexbert").exists()


def test_setup_chexbert_check_rejects_transformers_5_and_keeps_the_main_sentinel(tmp_path):
    box = SetupBox(tmp_path)
    (box.fake / "transformers.py").write_text('__version__ = "5.0.0"\n')
    done = box.run()
    assert done.returncode == 1, done.stdout + done.stderr
    assert "[setup] ERROR overlay check failed: chexbert" in done.stdout.splitlines()
    assert box.sentinel(".chat_deps").is_file() and not box.sentinel(".chat_deps_chexbert").exists()
