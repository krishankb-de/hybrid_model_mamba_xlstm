"""CHAT_UI_PLAN.md P7-A: the chat app import smoke job (scripts/chat_app_smoke_h100.sh), rehearsed for real in a temp tree.

The wrapper runs under /bin/bash (3.2 on the Mac, the oldest shell it has to work in) in a temp tree standing in for CLUSTER_REPO:
results/ is a symlink into a stand-in thesis checkout, as on the cluster; the two venvs' pythons are stubs that play each probe;
fake pip and uv sit on PATH to record an install. The last tests put the laptop's own interpreter in place of the stubs and run the
three probes for real over the real app/ package, and run the label-order probe against fake f1chexbert packages. The static pins
(directives, overlays, no installs, line shapes) are in tests/test_willi_parity.py. Synthetic data only (R7).
"""
import importlib.util
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import pytest

from tests.test_chat_remote import STAMP_RE, Sandbox, _summary_of

REPO_ROOT = Path(__file__).resolve().parent.parent
SMOKE_SH = REPO_ROOT / "scripts" / "chat_app_smoke_h100.sh"
BASH = "/bin/bash" if os.path.exists("/bin/bash") else "bash"      # the Mac's /bin/bash is 3.2: the oldest shell to support
JOB_ID = "4242"
# The shape scripts/chat_remote.sh sync writes to .sync_stamp: "<UTC time> <40-hex commit> <clean|dirty>".
STAMP = "2026-10-10T08:00:00Z 3f2a9c41d7e86b05a1c4e9d3b7f60285ac9e1d47 clean"
LINE_OK = re.compile(r"^(=== |\[setup\] |ERROR)")      # what `chat_remote.sh summary` shows of a job log, minus RESULT (none here)
SETUP_LINES = ["[setup] app.server imports; fastapi 9.9.9", "[setup] app.labeler imports; transformers 4.4.4",
               "[setup] chexbert label order equal: true (f1chexbert 0.0.2)"]
# The one shape the label-order probe may print, for either answer: the version is digits and dots only (R7: numbers only).
LABELS_LINE = re.compile(r"^\[setup\] chexbert label order equal: (true|false) \(f1chexbert [0-9]+(\.[0-9]+)*\)$")

# Stands in for a venv's python. It records the call, then plays the probe it was handed (told apart by the module names in its code):
# raw output on both streams that must never reach the job log, then the probe's one [setup] line, or the failure its mode asks for.
PYTHON_STUB = """#!/bin/bash
{ echo "@@"; echo "$0"; echo "${PYTHONPATH-unset}"; echo "$PWD"; echo "${HF_HUB_OFFLINE-unset}"; } >> "$STUB_DIR/python.calls"
case "$1" in -m) echo "python $*" >> "$STUB_DIR/installs.log";; esac
case "$2" in
  *app.server*) probe=server;;
  *app.labeler*) probe=labeler;;
  *app.labels*) probe=labels;;
  *) echo "stub: unknown probe" >&2; exit 99;;
esac
mode="$(cat "$STUB_DIR/$probe.mode" 2>/dev/null || echo ok)"
echo "Findings: SECRET REPORT TEXT study_id=12345678 /sc/home/someone/images/p10/img.jpg ($probe)"
echo "Traceback (most recent call last): SECRET MIMIC TEXT in a message ($probe)" >&2
case "$mode" in fail) exit 7;; silent) exit 0;; esac
case "$probe:$mode" in
  server:ok) echo "[setup] app.server imports; fastapi 9.9.9";;
  labeler:ok) echo "[setup] app.labeler imports; transformers 4.4.4";;
  labels:ok) echo "[setup] chexbert label order equal: true (f1chexbert 0.0.2)";;
  labels:differ) echo "[setup] chexbert label order equal: false (f1chexbert 0.0.2)"; exit 3;;
esac
exit 0
"""
# pip, pip3 and uv: anything that would install records itself.
INSTALLER_STUB = """#!/bin/bash
echo "$(basename "$0") $*" >> "$STUB_DIR/installs.log"
exit 0
"""
# The venv's python for the rehearsals that run the probes for real: the test interpreter.
REAL_PYTHON_WRAPPER = """#!/bin/bash
exec "$REAL_PYTHON" "$@"
"""


def listing(*roots: Path) -> Tuple[dict, set]:
    """({file or symlink: (size, mtime)}, {directory}) of everything under the roots, paths relative to the roots' parent: what a job
    that only READS a tree leaves exactly as it was (R8). A directory's own mtime moves when an entry is added, so it is not recorded."""
    files, dirs = {}, set()
    for root in roots:
        for base, subdirs, names in os.walk(str(root)):
            dirs.update(os.path.relpath(os.path.join(base, d), str(root.parent)) for d in subdirs)
            for name in names:
                full = os.path.join(base, name)
                st = os.lstat(full)
                files[os.path.relpath(full, str(root.parent))] = (st.st_size, st.st_mtime_ns)
    return files, dirs


class JobBox:
    """The smoke wrapper run for real in a temp tree: CLUSTER_REPO (repo/) with a results symlink into the thesis checkout (main/),
    two venvs and two overlays, and stubs."""

    def __init__(self, root: Path, stamp: Optional[str] = STAMP, real: bool = False):
        self.root = root
        self.repo, self.main, self.stubs, self.bin = root / "repo", root / "main", root / "stubs", root / "bin"
        for directory in (self.repo / "scripts", self.repo / "logs", self.repo / ".chat_deps", self.repo / ".chat_deps_chexbert",
                          self.main / "results", self.stubs, self.bin):
            directory.mkdir(parents=True)
        shutil.copy(str(SMOKE_SH), str(self.repo / "scripts" / SMOKE_SH.name))
        (self.repo / "results").symlink_to(self.main / "results", target_is_directory=True)
        for venv in (".venv", ".venv_chexbert"):
            (self.repo / venv / "bin").mkdir(parents=True)
            python = self.repo / venv / "bin" / "python"
            python.write_text(REAL_PYTHON_WRAPPER if real else PYTHON_STUB)
            python.chmod(0o755)
        for name in ("pip", "pip3", "uv"):
            stub = self.bin / name
            stub.write_text(INSTALLER_STUB)
            stub.chmod(0o755)
        if real:                                    # the probes import the app: it is the working tree's own
            for name in ("app", "hybrid_xmamba"):
                (self.repo / name).symlink_to(REPO_ROOT / name, target_is_directory=True)
        if stamp is not None:
            (self.repo / ".sync_stamp").write_text(stamp + "\n")

    @property
    def out_dir(self) -> Path:
        return self.main / "results" / "chat_app_smoke_{}".format(JOB_ID)

    def set_mode(self, probe: str, mode: str) -> None:
        """ok | fail | silent | differ (labels only): what the stub does when it plays `probe`."""
        (self.stubs / (probe + ".mode")).write_text(mode + "\n")

    def run(self, **extra_env: str) -> subprocess.CompletedProcess:
        env = {"PATH": "{}:/usr/bin:/bin".format(self.bin), "HOME": str(self.root / "home"), "SLURM_SUBMIT_DIR": str(self.repo),
               "SLURM_JOB_ID": JOB_ID, "STUB_DIR": str(self.stubs), "REAL_PYTHON": sys.executable}
        env.update(extra_env)
        # stdout and stderr share one pipe, like the single SLURM log: whatever bash itself complains about counts too.
        return subprocess.run([BASH, str(self.repo / "scripts" / SMOKE_SH.name)], cwd=str(self.root), env=env,
                              stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120)

    def calls(self) -> List[Tuple[str, str, str, str]]:
        """(interpreter as invoked, PYTHONPATH, working directory, HF_HUB_OFFLINE) of every python call, in order."""
        log = self.stubs / "python.calls"
        if not log.exists():
            return []
        return [tuple(rec.splitlines()) for rec in log.read_text().split("@@\n") if rec.strip()]

    def installs(self) -> List[str]:
        log = self.stubs / "installs.log"
        return log.read_text().splitlines() if log.exists() else []


def job_lines(done: subprocess.CompletedProcess) -> List[str]:
    return done.stdout.splitlines()


def test_a_clean_run_prints_the_sync_line_one_setup_line_per_probe_and_nothing_else(tmp_path):
    box = JobBox(tmp_path)
    done = box.run()
    lines = job_lines(done)
    assert done.returncode == 0, done.stdout
    assert lines[0] == "=== sync 3f2a9c41d7e86b05a1c4e9d3b7f60285ac9e1d47 clean ==="
    assert [l for l in lines if l.startswith("[setup] ")] == SETUP_LINES
    assert lines[-1] == "=== END chat app smoke ==="
    assert not [l for l in lines if not LINE_OK.match(l)], lines
    assert len(lines) == 6, lines                          # sync, the job line, three probes, the end


def test_the_raw_output_of_every_probe_goes_to_a_file_in_a_new_results_directory_and_never_to_the_log(tmp_path):
    box = JobBox(tmp_path)
    done = box.run()
    assert "SECRET" not in done.stdout and "Traceback" not in done.stdout and "12345678" not in done.stdout
    assert sorted(p.name for p in box.out_dir.iterdir()) == ["labeler.out", "labels.out", "server.out"]
    for probe in ("server", "labeler", "labels"):
        raw = (box.out_dir / (probe + ".out")).read_text()
        assert "SECRET REPORT TEXT" in raw and "SECRET MIMIC TEXT" in raw, "stdout and stderr both land in the file: " + probe
        assert "[setup] " in raw


def test_each_probe_runs_in_its_own_venv_with_its_own_overlay_from_the_repository_root_and_offline(tmp_path):
    """sbatch exports the submitting shell's environment: a PYTHONPATH or HF_HUB_OFFLINE=0 left in it must not reach a probe."""
    box = JobBox(tmp_path)
    assert box.run(PYTHONPATH="/stray/overlay", HF_HUB_OFFLINE="0").returncode == 0
    assert [(c[0], c[1]) for c in box.calls()] == [(".venv/bin/python", ".chat_deps"),
                                                   (".venv_chexbert/bin/python", ".chat_deps_chexbert"),
                                                   (".venv_chexbert/bin/python", ".chat_deps_chexbert")]
    assert {os.path.realpath(c[2]) for c in box.calls()} == {os.path.realpath(str(box.repo))}
    assert [c[3] for c in box.calls()] == ["1", "1", "1"], "compute nodes are offline: no import may wait on the network"


def test_a_missing_interpreter_is_exit_127_of_that_probe_and_the_others_still_run(tmp_path):
    box = JobBox(tmp_path)
    (box.repo / ".venv" / "bin" / "python").unlink()
    done = box.run()
    lines = job_lines(done)
    assert done.returncode == 1, done.stdout
    assert "ERROR server exit=127" in lines
    assert [l for l in lines if l.startswith("[setup] ")] == SETUP_LINES[1:]
    assert lines[-1] == "=== END chat app smoke: 1 of 3 probe(s) failed ==="
    assert not [l for l in lines if not LINE_OK.match(l)], lines


@pytest.mark.parametrize("failing", ["server", "labeler", "labels"])
def test_a_failing_probe_prints_its_name_and_exit_code_and_none_of_its_output_and_the_others_still_run(tmp_path, failing):
    box = JobBox(tmp_path)
    box.set_mode(failing, "fail")
    done = box.run()
    lines = job_lines(done)
    assert done.returncode == 1, done.stdout
    assert "ERROR {} exit=7".format(failing) in lines
    assert "SECRET" not in done.stdout and "Traceback" not in done.stdout
    assert [l for l in lines if l.startswith("[setup] ")] == [l for l, p in zip(SETUP_LINES, ("server", "labeler", "labels")) if p != failing]
    assert len(box.calls()) == 3, "every probe runs, whatever an earlier one did"
    assert lines[-1] == "=== END chat app smoke: 1 of 3 probe(s) failed ==="
    assert not [l for l in lines if not LINE_OK.match(l)], lines


def test_a_label_order_mismatch_prints_equal_false_and_fails_the_job(tmp_path):
    box = JobBox(tmp_path)
    box.set_mode("labels", "differ")
    done = box.run()
    lines = job_lines(done)
    assert done.returncode == 1, done.stdout
    assert lines[-3:] == ["[setup] chexbert label order equal: false (f1chexbert 0.0.2)", "ERROR labels exit=3",
                          "=== END chat app smoke: 1 of 3 probe(s) failed ==="], lines
    assert "SECRET" not in done.stdout


def test_a_probe_that_exits_0_without_its_setup_line_is_a_failure_not_a_pass(tmp_path):
    box = JobBox(tmp_path)
    box.set_mode("labeler", "silent")
    done = box.run()
    assert done.returncode == 1, done.stdout
    assert "ERROR labeler exit=0" in job_lines(done)


def test_every_probe_failing_ends_the_job_with_all_three_errors_and_exit_1(tmp_path):
    box = JobBox(tmp_path)
    for probe in ("server", "labeler", "labels"):
        box.set_mode(probe, "fail")
    done = box.run()
    lines = job_lines(done)
    assert done.returncode == 1
    assert [l for l in lines if l.startswith("ERROR")] == ["ERROR server exit=7", "ERROR labeler exit=7", "ERROR labels exit=7"]
    assert lines[-1] == "=== END chat app smoke: 3 of 3 probe(s) failed ==="


def test_nothing_is_installed_and_nothing_but_the_new_results_directory_is_written(tmp_path):
    box = JobBox(tmp_path)
    files_before, dirs_before = listing(box.repo, box.main)         # the stubs' own bookkeeping (stubs/) is the harness's, not the job's
    done = box.run()
    assert done.returncode == 0, done.stdout
    assert box.installs() == [], "pip, uv or `python -m` was called"
    files_after, dirs_after = listing(box.repo, box.main)
    assert all(files_after.get(path) == stat for path, stat in files_before.items()), "an existing file was changed or removed"
    assert dirs_before <= dirs_after, "a directory was removed"
    new = (set(files_after) - set(files_before)) | (dirs_after - dirs_before)
    run_dir = "main/results/chat_app_smoke_{}".format(JOB_ID)
    assert new == {run_dir} | {"{}/{}.out".format(run_dir, p) for p in ("server", "labeler", "labels")}, sorted(new)


def test_a_missing_results_directory_stops_the_job_before_any_probe_and_makes_nothing(tmp_path):
    box = JobBox(tmp_path)
    (box.repo / "results").unlink()
    done = box.run()
    lines = job_lines(done)
    assert done.returncode == 1, done.stdout
    assert lines[-1] == "ERROR results is missing: run chat_cluster_setup_h100.sh first", lines
    assert box.calls() == [] and not (box.repo / "results").exists()
    assert not [l for l in lines if not LINE_OK.match(l)], lines


def real_stamp(root: Path, dirty: bool) -> str:
    """The .sync_stamp that `scripts/chat_remote.sh sync` writes, from the real script in a throwaway git tree with ssh and rsync
    stubbed (tests/test_chat_remote.py's Sandbox): the wrapper has to read what the producer writes."""
    box = Sandbox(root)
    if dirty:
        script = box.repo / "scripts" / "chat_remote.sh"
        script.write_text(script.read_text() + "\n# touched\n")
    done = box.run("sync")
    assert done.returncode == 0, done.stdout + done.stderr
    return (box.repo / ".sync_stamp").read_text().strip()


@pytest.mark.parametrize("dirty", [False, True])
def test_the_first_line_of_the_job_log_names_the_commit_and_cleanliness_the_tree_was_synced_with(tmp_path, dirty):
    stamp = real_stamp(tmp_path / "sync", dirty)
    assert STAMP_RE.match(stamp), stamp
    _, sha, flag = stamp.split()
    assert flag == ("dirty" if dirty else "clean")
    done = JobBox(tmp_path / "job", stamp=stamp).run()
    assert job_lines(done)[0] == "=== sync {} {} ===".format(sha, flag), job_lines(done)[:2]


def test_a_missing_sync_stamp_is_reported_as_unknown_and_the_job_goes_on(tmp_path):
    box = JobBox(tmp_path, stamp=None)
    assert not (box.repo / ".sync_stamp").exists()
    done = box.run()
    assert done.returncode == 0, done.stdout
    assert job_lines(done)[0] == "=== sync unknown ==="


@pytest.mark.parametrize("stamp", [
    "",                                                                              # empty
    "rm -rf / ; echo $(whoami) FAKE_STAMP_TEXT",                                     # not the shape at all
    "2026-10-10T08:00:00Z 3f2a9c41d7e86b05a1c4e9d3b7f60285ac9e1d4 clean",           # 39 hex digits
    "2026-10-10T08:00:00Z 3F2A9C41D7E86B05A1C4E9D3B7F60285AC9E1D47 clean",          # upper case
    "2026-10-10T08:00:00Z 3f2a9c41d7e86b05a1c4e9d3b7f60285ac9e1d47 maybe",          # a flag that is neither
    "2026-10-10T08:00:00Z 3f2a9c41d7e86b05a1c4e9d3b7f60285ac9e1d47 clean FAKE_EXTRA",   # a field too many
])
def test_a_malformed_sync_stamp_is_reported_as_unknown_and_never_echoed(tmp_path, stamp):
    done = JobBox(tmp_path, stamp=stamp).run()
    lines = job_lines(done)
    assert lines[0] == "=== sync unknown ===", lines[:2]
    assert not [l for l in lines if "FAKE_" in l or "whoami" in l or "rm -rf" in l]


# ── the probes for real ───────────────────────────────────────────────────────

def needs(*modules: str) -> None:
    missing = [m for m in modules if importlib.util.find_spec(m) is None]
    if missing:
        pytest.skip("the test interpreter lacks {}: the cluster's shared venvs get them from the overlays".format(missing))


def test_the_three_probes_run_for_real_over_the_apps_own_imports_and_the_installed_f1chexbert(tmp_path):
    needs("fastapi", "transformers", "f1chexbert")
    box = JobBox(tmp_path, real=True)
    done = box.run()
    lines = job_lines(done)
    assert done.returncode == 0, done.stdout
    setup = [l for l in lines if l.startswith("[setup] ")]
    assert len(setup) == 3, lines
    assert re.match(r"^\[setup\] app\.server imports; fastapi \d+\.\d+", setup[0]), setup[0]
    assert re.match(r"^\[setup\] app\.labeler imports; transformers \d+\.\d+", setup[1]), setup[1]
    assert LABELS_LINE.match(setup[2]) and "equal: true" in setup[2], "CHEXBERT_14 differs from the names the installed f1chexbert reports: " + setup[2]
    assert not [l for l in lines if not LINE_OK.match(l)], lines
    assert _summary_of(tmp_path / "summary", setup) == setup, "R7: the real versions pass the summary allowlist and the mask unchanged"


def probe_code(name: str) -> str:
    match = re.search(r"(?ms)^{}='(.*?)'$".format(name), SMOKE_SH.read_text())
    assert match, name
    assert "'" not in match.group(1), "a single quote would end the bash string"
    return match.group(1)


# The shape of f1chexbert 0.0.2's F1CheXbert: the names are assigned in __init__, after the weights are loaded. Executing this file
# at all would raise at once, and running F1CheXbert() would raise at load_the_weights: the probe may only parse it.
F1_SOURCE = '''
raise RuntimeError("f1chexbert.py was executed")


class F1CheXbert:
    def __init__(self, refs_filename=None, **kwargs):
        self.model = load_the_weights()
        self.target_names = {names!r}
        self.target_names_5 = ["Cardiomegaly"]
'''


def run_labels_probe(tmp_path: Path, names: Optional[List[str]], source: Optional[str] = None,
                     version: Optional[str] = "0.0.2") -> subprocess.CompletedProcess:
    """The label-order probe's own code in a bare interpreter (-S: no site-packages, so the real f1chexbert cannot be found by
    accident) over a fake f1chexbert package whose __init__ ends the process if it is ever imported. Its dist-info says `version`
    (None: no dist-info, as if the package had been put there by hand); neither names nor source: no package at all."""
    env = {"PATH": os.environ.get("PATH", ""), "HOME": str(tmp_path)}
    if names is not None or source is not None:
        site = tmp_path / "site"
        package = site / "f1chexbert"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("raise SystemExit(98)\n")
        (package / "f1chexbert.py").write_text(source if source is not None else F1_SOURCE.format(names=names))
        if version is not None:
            info = site / "f1chexbert-0.0.2.dist-info"           # the directory name is how importlib.metadata finds it; the header is what it reports
            info.mkdir()
            (info / "METADATA").write_text("Metadata-Version: 2.1\nName: f1chexbert\nVersion: {}\n".format(version))
        env["PYTHONPATH"] = str(site)
    return subprocess.run([sys.executable, "-S", "-c", probe_code("LABELS_PROBE")], cwd=str(REPO_ROOT), env=env,
                          stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60)


def chexbert_14() -> List[str]:
    from app.labels import CHEXBERT_14
    return list(CHEXBERT_14)


TRUE_LINE = "[setup] chexbert label order equal: true (f1chexbert 0.0.2)\n"
FALSE_LINE = "[setup] chexbert label order equal: false (f1chexbert 0.0.2)\n"


def test_the_label_order_probe_says_true_for_the_order_f1chexbert_assigns(tmp_path):
    done = run_labels_probe(tmp_path, chexbert_14())
    assert done.returncode == 0, done.stderr
    assert done.stdout == TRUE_LINE


def test_the_label_order_probe_says_false_and_exits_3_for_any_other_order(tmp_path):
    names = chexbert_14()
    names[0], names[1] = names[1], names[0]
    done = run_labels_probe(tmp_path, names)
    assert (done.returncode, done.stdout) == (3, FALSE_LINE), done.stderr


def test_the_label_order_probe_says_false_for_a_list_of_another_length(tmp_path):
    done = run_labels_probe(tmp_path, chexbert_14()[:-1])
    assert (done.returncode, done.stdout) == (3, FALSE_LINE)


def test_the_label_order_probe_parses_f1chexbert_and_never_imports_it_or_loads_weights(tmp_path):
    """Importing the fake package ends the process with 98, executing its module raises, and building an F1CheXbert would fail at
    load_the_weights: a clean `true` proves the source was only parsed and the version only read from the metadata (no import,
    so no weights and no network)."""
    done = run_labels_probe(tmp_path, chexbert_14())
    assert (done.returncode, done.stdout, done.stderr) == (0, TRUE_LINE, "")


def test_the_label_order_probe_exits_4_without_a_setup_line_when_the_source_holds_no_names_list(tmp_path):
    done = run_labels_probe(tmp_path, None, source="class F1CheXbert:\n    def __init__(self):\n        self.target_names_5 = []\n")
    assert (done.returncode, done.stdout) == (4, "")


def test_the_label_order_probe_fails_with_no_setup_line_when_f1chexbert_is_not_installed(tmp_path):
    done = run_labels_probe(tmp_path, None)
    assert done.returncode == 1 and "[setup]" not in done.stdout


# Fix 1. Every source here assigns the RIGHT list somewhere, so a probe that took the first assignment it found would say `true`
# for all of them: only a probe that insists on exactly one plain assignment ends with exit 4. {names!r} is filled with CHEXBERT_14.
NOT_EXACTLY_ONE = {
    "twice in __init__": "class F1CheXbert:\n    def __init__(self):\n        self.target_names = {names!r}\n        self.target_names = {names!r}\n",
    "in __init__ and in another method": ("class F1CheXbert:\n    def __init__(self):\n        self.target_names = {names!r}\n"
                                          "    def reset(self):\n        self.target_names = []\n"),
    "in two classes": ("class A:\n    def __init__(self):\n        self.target_names = {names!r}\n"
                       "class B:\n    def __init__(self):\n        self.target_names = {names!r}\n"),
    "as a class attribute and in __init__": ("class F1CheXbert:\n    target_names = {names!r}\n    def __init__(self):\n"
                                             "        self.target_names = {names!r}\n"),
    "annotated and plain": ("class F1CheXbert:\n    def __init__(self):\n        self.target_names: list = {names!r}\n"
                            "        self.target_names = {names!r}\n"),
    "plain and augmented": ("class F1CheXbert:\n    def __init__(self):\n        self.target_names = {names!r}\n"
                            "        self.target_names += ['No Finding']\n"),
    "two targets in one statement": "class F1CheXbert:\n    def __init__(self):\n        self.target_names = other.target_names = {names!r}\n",
    "an augmented assignment alone": "class F1CheXbert:\n    def __init__(self):\n        self.target_names += {names!r}\n",
}
# One assignment, in whatever form, with look-alikes and keyword arguments around it: read as before.
EXACTLY_ONE = {
    "annotated": "class F1CheXbert:\n    def __init__(self):\n        self.target_names: list = {names!r}\n",
    "a class attribute": "class F1CheXbert:\n    target_names = {names!r}\n",
    "among look-alike names and keyword arguments": (
        "class F1CheXbert:\n    def __init__(self):\n        self.target_names = {names!r}\n        self.target_names_5 = ['Edema']\n"
        "        self.target_names_5_index = list(range(3))\n    def forward(self):\n"
        "        return report(target_names=self.target_names, other=self.target_names_5)\n"),
}


@pytest.mark.parametrize("case", sorted(NOT_EXACTLY_ONE))
def test_the_label_order_probe_needs_exactly_one_plain_target_names_assignment_and_otherwise_exits_4(tmp_path, case):
    done = run_labels_probe(tmp_path, None, source=NOT_EXACTLY_ONE[case].format(names=chexbert_14()))
    assert (done.returncode, done.stdout) == (4, ""), (case, done.stderr)


@pytest.mark.parametrize("case", sorted(EXACTLY_ONE))
def test_one_target_names_assignment_is_read_whatever_its_form_and_whatever_surrounds_it(tmp_path, case):
    done = run_labels_probe(tmp_path, None, source=EXACTLY_ONE[case].format(names=chexbert_14()))
    assert (done.returncode, done.stdout) == (0, TRUE_LINE), (case, done.stderr)


def test_the_probe_prints_the_version_from_the_distribution_metadata_in_digits_and_dots(tmp_path):
    names = chexbert_14()
    for answer, version, order in (("true", "10.20.30.40", names), ("false", "0.0.2", names[::-1])):
        done = run_labels_probe(tmp_path / answer, order, version=version)
        assert done.stdout == "[setup] chexbert label order equal: {} (f1chexbert {})\n".format(answer, version), done.stderr
        assert LABELS_LINE.match(done.stdout.strip())


@pytest.mark.parametrize("version", ["0.0.2rc1", "0.0.2.post1", "0.0.2.dev1", "0.0.2+cpu", "1!0.0.2", "v0.0.2", "0.0.2 text", "0..2",
                                     ".0.2", "0.0.2.", "", "unknown", "0.0.2; echo pwned"])
def test_a_version_that_is_not_digits_and_dots_is_never_printed_and_ends_the_probe_with_exit_4(tmp_path, version):
    """R7: the line may carry numbers only. Either answer is refused before it is printed, true or false."""
    for answer, order in (("true", chexbert_14()), ("false", chexbert_14()[::-1])):
        done = run_labels_probe(tmp_path / answer, order, version=version)
        assert (done.returncode, done.stdout) == (4, ""), (answer, version, done.stderr)


def test_a_package_without_distribution_metadata_ends_the_probe_with_no_setup_line(tmp_path):
    done = run_labels_probe(tmp_path, chexbert_14(), version=None)
    assert done.returncode == 1 and done.stdout == "", done.stderr


def test_the_probes_line_passes_the_summary_allowlist_and_the_mask_unchanged(tmp_path):
    """R7, end to end through chat_remote.sh's own grep pattern and mask: nothing of the line is dropped or blanked, for either
    answer and for a version of several digits."""
    shown_in = []
    for answer, version, order in (("true", "0.0.2", chexbert_14()), ("false", "10.20.30.40", chexbert_14()[::-1])):
        shown_in.append(run_labels_probe(tmp_path / answer, order, version=version).stdout.strip())
    assert all(LABELS_LINE.match(line) for line in shown_in), shown_in
    assert _summary_of(tmp_path / "summary", shown_in) == shown_in


def test_the_whole_clean_job_log_passes_the_summary_allowlist_and_the_mask_unchanged(tmp_path):
    lines = job_lines(JobBox(tmp_path / "job").run())
    comparable = [l for l in lines if not l.startswith("=== P7-A")]          # that one names the node, whose name is the machine's
    assert len(comparable) == len(lines) - 1 and any(l.startswith("[setup] chexbert") for l in comparable)
    assert _summary_of(tmp_path / "summary", comparable) == comparable
