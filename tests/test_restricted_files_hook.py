"""CHAT_UI_PLAN.md P9-A: the pre-commit check that refuses restricted (MIMIC-derived or binary) files, and its installer.

Every test runs the scripts in a throwaway git repository under tmp_path: the check is the real file run with that repository as
its working directory, and the installer and the hook are COPIES of the real files placed in the repository's own scripts/ (the
installer finds its repository from where it sits, and the hook runs the check of the repository being committed). The user's git
configuration and every GIT_* variable (a hook that runs pytest sets GIT_INDEX_FILE) are kept out, so nothing here can reach the
real repository's index or hooks. The one read of the real repository is `git ls-files`. Synthetic names and contents only (R7).
"""
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECK_SH = REPO_ROOT / "scripts" / "check_no_restricted_files.sh"
INSTALL_SH = REPO_ROOT / "scripts" / "install_hooks.sh"

# The Mac's /bin/bash is 3.2: the oldest shell these scripts have to work in.
BASH = "/bin/bash" if os.path.exists("/bin/bash") else "bash"

HEADER = "Refusing to commit restricted or binary artefacts:"

# What the check refuses, one example per rule, and the same names in other cases or places. Each is staged as a regular file.
REFUSED = [
    # a restricted directory, at the root (the first rule) and at any depth
    "outputs/run/metrics.json", "results/chat_x/hyps.txt", "uploads/s1/abc/original.png", "chat_sessions/notes.txt",
    "logs/chat_x_1.log", "hpi_results_logs/a_new_run_1.log",
    "analysis/results/table.md", "docs/logs/readme.md", "app/uploads/x.txt", "a/b/hpi_results_logs/y.log", "a/outputs/z.json",
    "a/chat_sessions/s.txt", "Results/x.md", "Outputs/x.txt", "hpi_results_logs/h100_stage0_150m_2341991.log.bak",
    # a restricted extension
    "a.ckpt", "model/last.ckpt", "d.parquet", "emb.npy", "emb.npz", "chat.db", "w.pt", "w.pth", "w.safetensors", "w.h5",
    "t.arrow", "t.feather",
    "EMB.NPY", "deep/nested/dir/emb.npy", "sp ace/emb.npy", "café/ünï.npy", 'say "hi".npy',
    # fix 1: images and archives (the reviewer's own examples first), pickles, and SQLite's side files
    "a.jpg", "a.jpeg", "a.dcm", "a.zip", "a.csv.gz", "a.tar.gz", "a.tgz", "mimic-cxr-reports.zip", "mimic-cxr-2.0.0-split.csv.gz",
    "a.webp", "a.tar", "a.gz", "a.7z", "a.pkl", "a.pickle", "chat.db-wal", "chat.db-shm", "chat.sqlite", "chat.sqlite3",
    "X.JPG", "SCAN.DCM", "deep/er/slice.Dcm",
    # fix 1: the files that hold MIMIC report text, under any directory
    "refs.txt", "hyps.txt", "chexbert_labels.json", "report_texts.txt", "analysis/x/refs.txt", "deep/er/hyps.txt",
    "a/chexbert_labels.json", "g/report_texts.txt", "HYPS.TXT",
    # fix 1: a restricted name followed by more suffixes, as a download, an rsync temporary, a backup or a compressed copy leaves it
    "last.ckpt.part", ".last.ckpt.Xy12Ab", "emb.npy.gz", "emb.npy.bak", "emb.npy~", "emb.npy-old", "emb.npy_old", "w.pt.tmp",
    "x.parquet.crdownload", "a/.emb.npz.Ab12Cd", "chat.db-journal", "hyps.txt.bak", ".refs.txt.Xy12Ab", "refs.txt~",
    # fix 1: a directory named like a restricted file holds restricted files (a SQLite or parquet dataset is a directory)
    "weights.safetensors/shard-00001", "chat.db/MANIFEST", "d.parquet/part-0",
    # PNG is allowed only as the evidence: the same directory takes nothing else
    "docs/chat_ui/evidence/p9z/shot.jpg", "docs/chat_ui/evidence/p9z/refs.txt", "docs/chat_ui/evidence/p9z/shot.png.gz",
]
# What it lets through. The evidence PNGs are committed on purpose (screenshots of the UI over synthetic data). The lookalikes
# are there to keep the patterns honest: an extension is refused when it ends the name or is followed by something that is not a
# letter or digit, never when it is merely the start of a longer word.
ACCEPTED = [
    "app/server.py", "README.md", "scripts/run.sh", "docs/chat_ui/evidence/p4e/settled_1280x900_light.png",
    "docs/chat_ui/evidence/p9z/a_future_shot.png", "outputs_notes.md", "analysis/results_summary.md", "app/static/logo.png",
    "make.target.yaml", "docs/notes.gzip.md", "scripts/foo.ptx", "docs/ER.dbt.md", "app/zip_utils.py", "app/pt_helper.py",
    "docs/results.md", "docs/logs_notes.md", "analysis/refs_table.md", "my_hyps.txt", "docs/hyps.md", "scripts/chexbert_labels_notes.md",
    "scripts/jpg_tools.py", "docs/sample.pthon",
]
# The names of the restricted directories, for the tests that stage a file or a symlink of exactly that name.
BARE_NAMES = ["outputs", "results", "uploads", "chat_sessions", "logs", "hpi_results_logs"]
# Tracked in this repository before the check existed, and not restricted content: three Stage-0 language-model job logs.
TRACKED_LOGS = ["hpi_results_logs/h100_stage0_150m_2341991.log", "hpi_results_logs/monitor_stage0_2351222.log",
                "hpi_results_logs/verify_handoff_2351231.log"]


class Repo:
    """A throwaway git repository, with copies of the scripts in its scripts/ when asked."""

    def __init__(self, root: Path, with_scripts: bool = False):
        self.root = root
        self.home = root / "home"
        self.repo = root / "repo"
        self.home.mkdir()
        (self.repo / "scripts").mkdir(parents=True)
        if with_scripts:
            for source in (CHECK_SH, INSTALL_SH):
                shutil.copy(str(source), str(self.repo / "scripts" / source.name))
        self.git("init", "-q")

    def env(self) -> Dict[str, str]:
        env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
        env.update({"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1", "HOME": str(self.home),
                    "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
                    "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid"})
        return env

    def git(self, *args: str, check: bool = True, cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
        done = subprocess.run(["git"] + list(args), cwd=str(cwd or self.repo), env=self.env(), stdin=subprocess.DEVNULL,
                              capture_output=True, encoding="utf-8", timeout=120)
        assert not check or done.returncode == 0, (args, done.stdout, done.stderr)
        return done

    def write(self, rel: str, text: str = "x") -> None:
        path = self.repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def stage(self, *rels: str) -> None:
        self.git("add", "--", *rels)

    def commit_unchecked(self, message: str = "c") -> None:
        self.git("commit", "-q", "--no-verify", "-m", message)

    def run(self, script: Path, cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
        return subprocess.run([BASH, str(script)], cwd=str(cwd or self.repo), env=self.env(), stdin=subprocess.DEVNULL,
                              capture_output=True, encoding="utf-8", timeout=120)

    def check(self, cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
        """The real check, run with this repository (or a directory of it) as the working directory."""
        return self.run(CHECK_SH, cwd)

    def install(self) -> subprocess.CompletedProcess:
        """The installer COPY in this repository's scripts/: it installs into the repository it sits in, never into another."""
        installer = self.repo / "scripts" / "install_hooks.sh"
        assert installer.is_file() and self.repo.resolve() in installer.resolve().parents
        return self.run(installer)

    def hooks_dir(self) -> Path:
        out = self.git("rev-parse", "--git-path", "hooks").stdout.strip()
        path = Path(out) if os.path.isabs(out) else self.repo / out
        assert self.root.resolve() in path.resolve().parents, "the hooks directory is outside the throwaway repository"
        return path

    def commit(self, message: str = "c") -> subprocess.CompletedProcess:
        """A commit WITH its hooks."""
        return self.git("commit", "-q", "-m", message, check=False)

    def commits(self) -> int:
        done = self.git("rev-list", "--count", "HEAD", check=False)
        return int(done.stdout.strip()) if done.returncode == 0 else 0


@pytest.fixture
def repo(tmp_path) -> Repo:
    return Repo(tmp_path)


@pytest.fixture
def hooked(tmp_path) -> Repo:
    """A repository with the scripts in it and the hook installed."""
    box = Repo(tmp_path, with_scripts=True)
    done = box.install()
    assert done.returncode == 0, done.stdout + done.stderr
    return box


def lines(done: subprocess.CompletedProcess) -> List[str]:
    return done.stdout.splitlines()


# ── the check ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("path", REFUSED)
def test_a_staged_restricted_file_is_refused_with_exit_1_and_named(repo, path):
    repo.write(path)
    repo.stage(path)
    done = repo.check()
    assert done.returncode == 1, (done.stdout, done.stderr)
    assert lines(done) == [HEADER, path]


@pytest.mark.parametrize("path", ACCEPTED)
def test_an_ordinary_staged_file_passes_with_exit_0_and_silence(repo, path):
    repo.write(path)
    repo.stage(path)
    done = repo.check()
    assert done.returncode == 0, (done.stdout, done.stderr)
    assert done.stdout == "" and done.stderr == ""


@pytest.mark.parametrize("kind", ["file", "symlink"])
@pytest.mark.parametrize("where", ["", "a/b/"])
@pytest.mark.parametrize("name", BARE_NAMES)
def test_a_file_or_symlink_named_exactly_like_a_restricted_directory_is_refused_at_any_depth(repo, name, where, kind):
    """The directory rule is `<name>/...`: a symlink or a plain file called `results` has no slash after it, and the cluster tree
    has exactly such symlinks (outputs, results), so a rule written for directories alone would let one through."""
    rel = where + name
    if kind == "file":
        repo.write(rel)
    else:
        (repo.repo / rel).parent.mkdir(parents=True, exist_ok=True)
        os.symlink("../somewhere/else", str(repo.repo / rel))
    repo.stage(rel)
    assert repo.git("ls-files", "-s", "--", rel).stdout.split()[0] == ("100644" if kind == "file" else "120000")
    done = repo.check()
    assert done.returncode == 1, (done.stdout, done.stderr)
    assert lines(done) == [HEADER, rel]


def test_a_symlink_replaced_by_a_real_file_is_a_type_change_and_is_still_refused(repo):
    """git reports a path whose type changed (symlink to file, or back) as T, which --diff-filter=ACMR leaves out."""
    (repo.repo / "x.npy").symlink_to("target.txt")
    repo.stage("x.npy")
    repo.commit_unchecked("a symlink, committed before the check existed")
    (repo.repo / "x.npy").unlink()
    repo.write("x.npy", "now a real file")
    repo.stage("x.npy")
    assert repo.git("diff", "--cached", "--name-status").stdout.split()[0] == "T", "this test needs git to see a type change"
    done = repo.check()
    assert done.returncode == 1, (done.stdout, done.stderr)
    assert lines(done) == [HEADER, "x.npy"]


def test_a_type_change_of_an_ordinary_file_is_not_refused(repo):
    repo.write("a.py")
    repo.stage("a.py")
    repo.commit_unchecked()
    (repo.repo / "a.py").unlink()
    (repo.repo / "a.py").symlink_to("b.py")
    repo.stage("a.py")
    assert repo.git("diff", "--cached", "--name-status").stdout.split()[0] == "T", "this test needs git to see a type change"
    done = repo.check()
    assert done.returncode == 0, (done.stdout, done.stderr)


def test_diff_relative_in_the_configuration_cannot_hide_a_path_outside_the_directory_the_check_runs_in(repo):
    """With diff.relative=true, `git diff` run from a subdirectory lists only what is under it, named relative to it: a restricted
    file elsewhere in the tree would not be seen. The check asks for every path, named from the repository root."""
    repo.git("config", "diff.relative", "true")
    repo.write("sub/ok.py")
    repo.write("sub/emb2.npy")
    repo.write("outside/emb.npy")
    repo.stage("sub/ok.py", "sub/emb2.npy", "outside/emb.npy")
    seen = repo.git("diff", "--cached", "--name-only", cwd=repo.repo / "sub").stdout.split()
    assert seen == ["emb2.npy", "ok.py"], "this test needs git to hide a path from a subdirectory: {}".format(seen)
    done = repo.check(cwd=repo.repo / "sub")
    assert done.returncode == 1, (done.stdout, done.stderr)
    assert lines(done) == [HEADER, "outside/emb.npy", "sub/emb2.npy"]


def test_the_brief_cases_x_npy_refused_and_a_py_file_passes(repo):
    repo.write("x.npy")
    repo.stage("x.npy")
    assert repo.check().returncode == 1
    repo.git("rm", "-q", "--cached", "x.npy")
    repo.write("x.py")
    repo.stage("x.py")
    assert repo.check().returncode == 0


def test_every_refused_file_among_several_is_listed_and_the_others_are_not(repo):
    for rel in ("a.py", "b/emb.npy", "c.md", "outputs/d.txt"):
        repo.write(rel)
    repo.stage("a.py", "b/emb.npy", "c.md", "outputs/d.txt")
    done = repo.check()
    assert done.returncode == 1
    assert lines(done) == [HEADER, "b/emb.npy", "outputs/d.txt"]


def test_the_message_is_the_header_and_the_paths_and_never_what_the_files_hold(repo):
    repo.write("big.npy", "CONTENT-MARKER-SHOULD-NOT-PRINT")
    repo.write("results/study_12345678.txt", "CONTENT-MARKER-SHOULD-NOT-PRINT")
    repo.stage("big.npy", "results/study_12345678.txt")
    done = repo.check()
    assert done.returncode == 1
    assert done.stdout == "{}\nbig.npy\nresults/study_12345678.txt\n".format(HEADER)
    assert done.stderr == ""
    assert "CONTENT-MARKER" not in done.stdout + done.stderr


def test_nothing_staged_passes_and_an_unstaged_restricted_file_is_not_the_checks_business(repo):
    assert repo.check().returncode == 0                           # an empty repository, nothing staged
    repo.write("emb.npy")
    repo.write("outputs/x.json")
    assert repo.check().returncode == 0                           # on disk, not staged


def test_a_file_renamed_into_outputs_is_refused(repo):
    repo.write("a.py")
    repo.stage("a.py")
    repo.commit_unchecked()
    (repo.repo / "outputs").mkdir()                    # git mv does not make the destination directory
    repo.git("mv", "a.py", "outputs/a.py")
    assert "R" in repo.git("diff", "--cached", "--name-status").stdout.split()[0], "this test needs git to see a rename"
    done = repo.check()
    assert done.returncode == 1
    assert lines(done) == [HEADER, "outputs/a.py"]


def test_a_rename_is_still_refused_when_git_is_told_not_to_detect_renames(repo):
    repo.write("a.py")
    repo.stage("a.py")
    repo.commit_unchecked()
    repo.git("config", "diff.renames", "false")
    (repo.repo / "outputs").mkdir()
    repo.git("mv", "a.py", "outputs/a.py")
    assert "R" not in repo.git("diff", "--cached", "--name-status").stdout.split()[0], "renames were meant to be off"
    done = repo.check()
    assert done.returncode == 1 and lines(done) == [HEADER, "outputs/a.py"]


def test_deleting_a_restricted_file_that_was_committed_is_not_refused(repo):
    repo.write("outputs/old.npy")
    repo.stage("outputs/old.npy")
    repo.commit_unchecked("committed before the check existed")
    repo.git("rm", "-q", "outputs/old.npy")
    done = repo.check()
    assert done.returncode == 0, (done.stdout, done.stderr)


def test_the_tracked_stage0_logs_stay_editable_but_a_new_log_beside_them_is_refused(repo):
    for rel in TRACKED_LOGS:
        repo.write(rel, "v1")
        repo.stage(rel)
    repo.commit_unchecked("the logs this repository tracks")
    for rel in TRACKED_LOGS:
        repo.write(rel, "v2")
        repo.stage(rel)
    done = repo.check()
    assert done.returncode == 0, (done.stdout, done.stderr)
    repo.write("hpi_results_logs/h100_stage0_150m_9999999.log")
    repo.stage("hpi_results_logs/h100_stage0_150m_9999999.log")
    done = repo.check()
    assert done.returncode == 1 and lines(done) == [HEADER, "hpi_results_logs/h100_stage0_150m_9999999.log"]


def test_the_first_commit_of_a_repository_is_checked_too(repo):
    assert repo.commits() == 0
    repo.write("x.npy")
    repo.stage("x.npy")
    assert repo.check().returncode == 1


def test_outside_a_git_repository_the_check_fails_closed(tmp_path):
    done = subprocess.run([BASH, str(CHECK_SH)], cwd=str(tmp_path), stdin=subprocess.DEVNULL, capture_output=True, text=True,
                          env={"PATH": os.environ["PATH"], "HOME": str(tmp_path), "GIT_CEILING_DIRECTORIES": str(tmp_path.parent),
                               "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"})
    assert done.returncode == 2, "neither 'fine' (0) nor 'refused' (1): the check could not look, and says so"
    assert HEADER not in done.stdout and "git" in done.stderr


def tracked_here() -> List[str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    done = subprocess.run(["git", "-C", str(REPO_ROOT), "ls-files", "-z"], env=env, stdin=subprocess.DEVNULL,
                          capture_output=True, timeout=60)
    if done.returncode != 0:
        pytest.skip("not a git checkout (the cluster's rsynced tree has no .git)")
    return [os.fsdecode(p) for p in done.stdout.split(b"\0") if p]


def test_no_file_this_repository_tracks_would_be_refused_when_modified(tmp_path):
    """The audit behind the allow-list: every tracked path, staged as an edit would stage it, passes. A tracked file the check
    refuses would make a legitimate edit impossible; the fix is an exact-path allow-list entry in the script, with a reason."""
    names = tracked_here()
    assert len(names) > 100, "git ls-files gave {} paths".format(len(names))
    box = Repo(tmp_path)
    for rel in names:
        box.write(rel, "")
    box.git("add", "-A")
    staged = [p for p in box.git("ls-files", "-z").stdout.split("\0") if p]
    assert len(staged) == len(names), "the audit staged {} of {} tracked paths".format(len(staged), len(names))
    done = box.check()
    assert done.returncode == 0, "tracked files the check would refuse:\n" + done.stdout
    # A control: the same repository refuses a restricted path the moment one is staged, so the exit 0 above was not a blind check.
    for rel in ("analysis/refs.txt", "weights/last.ckpt.part", "scans/a.dcm", "results"):
        box.write(rel)
        box.stage(rel)
    refused = box.check()
    assert refused.returncode == 1, refused.stdout
    assert lines(refused) == [HEADER, "analysis/refs.txt", "results", "scans/a.dcm", "weights/last.ckpt.part"]


# ── the installer and the hook it installs ────────────────────────────────────

def test_install_writes_an_executable_pre_commit_hook_into_git_hooks(tmp_path):
    box = Repo(tmp_path, with_scripts=True)
    done = box.install()
    assert done.returncode == 0, done.stdout + done.stderr
    hook = box.hooks_dir() / "pre-commit"
    assert hook.is_file() and os.access(str(hook), os.X_OK)
    assert "check_no_restricted_files.sh" in hook.read_text()
    assert str(hook.resolve()) in done.stdout, "it says where it installed the hook"


def test_the_installed_hook_blocks_a_commit_of_a_restricted_file_and_lets_an_ordinary_one_through(hooked):
    hooked.write("x.npy")
    hooked.stage("x.npy")
    done = hooked.commit()
    assert done.returncode != 0, (done.stdout, done.stderr)
    assert HEADER in done.stdout + done.stderr and "x.npy" in done.stdout + done.stderr
    assert hooked.commits() == 0, "nothing was committed"
    hooked.git("rm", "-q", "--cached", "x.npy")
    hooked.write("x.py")
    hooked.stage("x.py")
    done = hooked.commit()
    assert done.returncode == 0, (done.stdout, done.stderr)
    assert hooked.commits() == 1


def test_install_twice_is_harmless_and_leaves_the_hook_as_it_was(hooked):
    hook = hooked.hooks_dir() / "pre-commit"
    before = hook.read_text()
    done = hooked.install()
    assert done.returncode == 0, done.stdout + done.stderr
    assert hook.read_text() == before and os.access(str(hook), os.X_OK)


def test_install_repairs_the_executable_bit_of_its_own_hook(hooked):
    hook = hooked.hooks_dir() / "pre-commit"
    hook.chmod(0o644)
    assert hooked.install().returncode == 0
    assert os.access(str(hook), os.X_OK)


FOREIGN_HOOK = "#!/bin/sh\necho 'a hook somebody wrote'\n"


def test_install_never_overwrites_a_different_hook_and_says_how_to_chain_the_check(tmp_path):
    box = Repo(tmp_path, with_scripts=True)
    hook = box.hooks_dir() / "pre-commit"
    hook.parent.mkdir(parents=True, exist_ok=True)
    hook.write_text(FOREIGN_HOOK)
    hook.chmod(0o755)
    done = box.install()
    assert done.returncode == 1, (done.stdout, done.stderr)
    assert hook.read_text() == FOREIGN_HOOK, "the existing hook was changed"
    text = done.stdout + done.stderr
    assert "check_no_restricted_files.sh" in text and "already exists" in text
    # The advice works: the line it prints, appended to that hook, makes the hook run the check.
    advice = [l.strip() for l in text.splitlines() if l.startswith("  ") and "check_no_restricted_files.sh" in l]
    assert len(advice) == 1, text
    hook.write_text(FOREIGN_HOOK + advice[0] + "\n")
    box.write("x.npy")
    box.stage("x.npy")
    refused = box.commit()
    assert refused.returncode != 0 and HEADER in refused.stdout + refused.stderr
    box.git("rm", "-q", "--cached", "x.npy")
    box.write("x.py")
    box.stage("x.py")
    assert box.commit().returncode == 0


@pytest.mark.parametrize("kind", ["broken_symlink", "directory"])
def test_a_hook_that_is_a_broken_symlink_or_a_directory_is_not_overwritten_either(tmp_path, kind):
    box = Repo(tmp_path, with_scripts=True)
    hook = box.hooks_dir() / "pre-commit"
    hook.parent.mkdir(parents=True, exist_ok=True)
    if kind == "directory":
        hook.mkdir()
    else:
        hook.symlink_to(tmp_path / "nowhere")
    done = box.install()
    assert done.returncode == 1, (done.stdout, done.stderr)
    assert hook.is_dir() if kind == "directory" else hook.is_symlink()


@pytest.mark.parametrize("where", ["relative", "absolute"])
def test_install_honours_core_hooks_path(tmp_path, where):
    box = Repo(tmp_path, with_scripts=True)
    target = ".githooks" if where == "relative" else str(tmp_path / "shared_hooks")
    box.git("config", "core.hooksPath", target)
    done = box.install()
    assert done.returncode == 0, done.stdout + done.stderr
    hook = (box.repo / target) if where == "relative" else Path(target)
    assert hook.is_dir() and (hook / "pre-commit").is_file() and os.access(str(hook / "pre-commit"), os.X_OK)
    assert not (box.repo / ".git" / "hooks" / "pre-commit").exists(), "the default directory is not where git looks now"
    box.write("x.npy")
    box.stage("x.npy")
    refused = box.commit()
    assert refused.returncode != 0 and HEADER in refused.stdout + refused.stderr, "git runs the hook it was pointed at"


def test_a_foreign_hook_in_core_hooks_path_is_protected_too(tmp_path):
    box = Repo(tmp_path, with_scripts=True)
    box.git("config", "core.hooksPath", ".githooks")
    (box.repo / ".githooks").mkdir()
    (box.repo / ".githooks" / "pre-commit").write_text(FOREIGN_HOOK)
    assert box.install().returncode == 1
    assert (box.repo / ".githooks" / "pre-commit").read_text() == FOREIGN_HOOK


def test_install_warns_when_core_hooks_path_is_relative_that_linked_worktrees_will_not_run_the_hook(tmp_path):
    box = Repo(tmp_path, with_scripts=True)
    box.git("config", "core.hooksPath", ".githooks")
    done = box.install()
    out = done.stdout + done.stderr
    assert done.returncode == 0, out
    assert "WARNING" in out and "relative" in out and ".githooks" in out, out
    assert "linked worktrees will not run the hook" in out, out
    assert (box.repo / ".githooks" / "pre-commit").is_file(), "the warning does not stop the install"


def test_the_warning_comes_with_a_refusal_and_with_an_install_that_was_already_done_too(tmp_path):
    box = Repo(tmp_path, with_scripts=True)
    box.git("config", "core.hooksPath", ".githooks")
    assert box.install().returncode == 0
    again = box.install()
    assert again.returncode == 0 and "WARNING" in again.stdout + again.stderr and "already installed" in again.stdout
    (box.repo / ".githooks" / "pre-commit").write_text(FOREIGN_HOOK)
    refused = box.install()
    assert refused.returncode == 1 and "WARNING" in refused.stdout + refused.stderr


@pytest.mark.parametrize("where", ["default", "absolute"])
def test_install_gives_no_warning_when_every_worktree_shares_the_hooks_directory(tmp_path, where):
    box = Repo(tmp_path, with_scripts=True)
    if where == "absolute":
        box.git("config", "core.hooksPath", str(tmp_path / "shared_hooks"))
    done = box.install()
    assert done.returncode == 0, done.stdout + done.stderr
    assert "WARNING" not in done.stdout + done.stderr


@pytest.mark.parametrize("relative", [True, False])
def test_the_warning_is_true_a_linked_worktree_runs_the_default_hooks_directory_but_not_a_relative_hooks_path(tmp_path, relative):
    box = Repo(tmp_path, with_scripts=True)
    if relative:
        box.git("config", "core.hooksPath", ".githooks")
    assert box.install().returncode == 0
    box.stage("scripts/check_no_restricted_files.sh", "scripts/install_hooks.sh")      # a worktree is a checkout: only committed files
    box.commit_unchecked("the scripts")
    worktree = tmp_path / "wt"
    box.git("worktree", "add", "-q", str(worktree), "-b", "other")
    assert (worktree / "scripts" / "check_no_restricted_files.sh").is_file()
    (worktree / "x.npy").write_text("x")
    box.git("add", "x.npy", cwd=worktree)
    done = box.git("commit", "-q", "-m", "c", check=False, cwd=worktree)
    if relative:
        assert done.returncode == 0, "the worktree has no .githooks of its own, so no hook ran: " + done.stdout + done.stderr
    else:
        assert done.returncode != 0 and HEADER in done.stdout + done.stderr, "the worktree shares .git/hooks"


def test_the_installer_header_does_not_claim_that_one_install_covers_every_worktree():
    header = " ".join(l.lstrip("#").strip() for l in INSTALL_SH.read_text().splitlines() if l.startswith("#")).lower()   # one line of prose
    assert "relative core.hookspath" in header and "linked worktree" in header
    assert "share that directory" not in header, "the unqualified claim (it fails for a relative core.hooksPath) is still there"


def test_the_hook_does_nothing_where_the_check_is_absent(hooked):
    """The default hooks directory is shared by every worktree of a repository, and a core.hooksPath by every repository that uses
    it: a checkout whose scripts/ has no check (another branch, another project) is committed to as if there were no hook."""
    (hooked.repo / "scripts" / "check_no_restricted_files.sh").unlink()
    hooked.write("x.npy")
    hooked.stage("x.npy")
    done = hooked.commit()
    assert done.returncode == 0, (done.stdout, done.stderr)
    assert hooked.commits() == 1


def test_the_hook_checks_a_commit_made_from_a_subdirectory_too(hooked):
    hooked.write("sub/x.py")
    hooked.write("sub/emb.npy")
    hooked.stage("sub/x.py", "sub/emb.npy")
    done = subprocess.run(["git", "commit", "-q", "-m", "c"], cwd=str(hooked.repo / "sub"), env=hooked.env(),
                          stdin=subprocess.DEVNULL, capture_output=True, encoding="utf-8", timeout=60)
    assert done.returncode != 0 and "sub/emb.npy" in done.stdout + done.stderr


def test_the_installer_refuses_to_run_without_the_check_beside_it(tmp_path):
    box = Repo(tmp_path, with_scripts=True)
    (box.repo / "scripts" / "check_no_restricted_files.sh").unlink()
    done = box.install()
    assert done.returncode == 1
    assert not (box.hooks_dir() / "pre-commit").exists()
