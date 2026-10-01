"""CHAT_UI_PLAN.md P1-C: the CPU decode probe (scripts/chat_cpu_decode_probe_h100.sh), run for real in a temp tree.

The wrapper is bash around `python scripts/evaluate_report_generation.py`. Here that script is a stub that records how it
was called and writes synthetic hyps.txt files, `/usr/bin/time` is a stub that prints the two report lines the wrapper
greps for, and `lscpu` is a stub; nothing touches the cluster. Synthetic data only (R7): the "report text" and the study
path in the stub's traceback are made up, and the tests assert that none of it reaches the job log.
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import pytest

from tests.test_chat_remote import RUNNING_SSH, Sandbox, fake_env

REPO_ROOT = Path(__file__).resolve().parent.parent
PROBE_SH = REPO_ROOT / "scripts" / "chat_cpu_decode_probe_h100.sh"
BASH = "/bin/bash" if os.path.exists("/bin/bash") else "bash"   # the Mac's 3.2 is the oldest shell it has to work in
ARMS = ["warm", "cached_1", "cached_a", "cached_b", "uncached"]

# Stands in for evaluate_report_generation.py: logs its argv as JSON, prints what the real one prints, then either fails
# with the traceback in $STUB_FAIL[arm] or writes hyps.txt ("synthetic report <i>", with $STUB_DRIFT rows altered and
# $STUB_SHORT[arm] lines missing at the end).
EVAL_STUB = r'''
import argparse, json, os, sys
parser = argparse.ArgumentParser()
for flag in ("--checkpoint", "--model-config", "--parquet", "--decode", "--beam-size", "--max-new-tokens",
             "--dump-dir", "--num-samples"):
    parser.add_argument(flag)
parser.add_argument("--cached-decode", action="store_true")
args = parser.parse_args()
arm = os.path.basename(args.dump_dir.rstrip("/"))
with open(os.environ["STUB_CALLS"], "a") as fh:
    fh.write(json.dumps({"arm": arm, "n": int(args.num_samples), "cached": args.cached_decode, "decode": args.decode,
                         "beam": args.beam_size, "tokens": args.max_new_tokens, "config": args.model_config,
                         "ckpt": os.path.basename(args.checkpoint), "parquet": os.path.basename(args.parquet)}) + "\n")
print("  prefix_k = 32")
print("  Missing keys: 0, Unexpected: 0")
print("GENERATED: SYNTHETIC report text")
fail = json.loads(os.environ.get("STUB_FAIL", "{}"))
if arm in fail:
    sys.stderr.write(fail[arm])
    sys.exit(1)
hyps = ["synthetic report %d" % i for i in range(int(args.num_samples))]
for row in json.loads(os.environ.get("STUB_DRIFT", "{}")).get(arm, []):
    hyps[row] += " drift"
hyps = hyps[:len(hyps) - json.loads(os.environ.get("STUB_SHORT", "{}")).get(arm, 0)]
os.makedirs(args.dump_dir, exist_ok=True)
with open(os.path.join(args.dump_dir, "hyps.txt"), "w") as fh:
    fh.write("".join(h + "\n" for h in hyps))
'''

TIME_STUB = """#!/bin/bash
# stands in for GNU `time -v` (the Mac's BSD time has no -v): run the command, then print the two lines the wrapper greps for
[ "$1" = "-v" ] && shift
"$@"; rc=$?
printf '\\tElapsed (wall clock) time (h:mm:ss or m:ss): 0:01.23\\n\\tMaximum resident set size (kbytes): %s\\n' "${STUB_RSS_KB:-1234567}" >&2
exit $rc
"""

LSCPU_STUB = """#!/bin/bash
printf 'Architecture:          x86_64\\nModel name:            Stub CPU 9000 @ 3.00GHz\\nCPU(s):                8\\n'
"""

# What a failed arm leaves in its own log: a study path in the first message, a dotted class name in the second.
STUDY_PATH = "/d/files/p10/p10000032/s50414267/x.jpg"
TRACEBACK = (
    "Traceback (most recent call last):\n"
    '  File "scripts/evaluate_report_generation.py", line 559, in run_checkpoint_inspection\n'
    '    img = Image.open(row["image"]).convert("RGB")\n'
    "FileNotFoundError: [Errno 2] No such file or directory: '%s'\n"
    "\n"
    "During handling of the above exception, another exception occurred:\n"
    "\n"
    "Traceback (most recent call last):\n"
    '  File "scripts/evaluate_report_generation.py", line 600, in <module>\n'
    "huggingface_hub.errors.LocalEntryNotFoundError: SYNTHETIC findings text\n"
) % STUDY_PATH


class ProbeBox:
    """The wrapper copied into a temp tree with a fake workspace around it: checkpoint, a 30-line published dump,
    a test parquet, and stubs for python (the eval script), time and lscpu."""

    JOB = "4242"

    def __init__(self, root: Path, time_present: bool = True):
        self.root = root
        self.repo, self.bin, self.data = root / "repo", root / "bin", root / "data"
        self.calls_log = root / "calls.jsonl"
        self.ckpt = self.repo / "outputs" / "h100_report_gen_m3_tower13d_s42" / "checkpoints" / "last.ckpt"
        self.published = self.repo / "results" / "report_gen_m3_test_split_s42" / "hyps.txt"
        self.parquet = self.data / "test.parquet"
        activate = self.repo / ".venv" / "bin" / "activate"
        eval_script = self.repo / "scripts" / "evaluate_report_generation.py"
        for directory in (self.bin, self.ckpt.parent, self.published.parent, self.data, activate.parent, eval_script.parent):
            directory.mkdir(parents=True, exist_ok=True)
        for path in (self.ckpt, self.parquet, activate):
            path.write_text("")
        self.published.write_text("".join("synthetic report %d\n" % i for i in range(30)))
        eval_script.write_text(EVAL_STUB)
        self._stub("python", '#!/bin/bash\nexec "%s" "$@"\n' % sys.executable)
        self._stub("lscpu", LSCPU_STUB)
        self.time_bin = self.bin / ("time" if time_present else "no_such_time")
        if time_present:
            self._stub("time", TIME_STUB)
        self.wrapper = self.repo / "scripts" / "chat_cpu_decode_probe_h100.sh"
        self.wrapper.write_text(self._patched(PROBE_SH.read_text()))

    def _stub(self, name: str, text: str) -> None:
        (self.bin / name).write_text(text)
        (self.bin / name).chmod(0o755)

    def _patched(self, src: str) -> str:
        """The wrapper with /usr/bin/time (GNU time, absent on a Mac) pointed at the stub. Its message text is left alone."""
        assert src.count("[ -x /usr/bin/time ]") == 1 and src.count("/usr/bin/time -v") == 1
        return (src.replace("[ -x /usr/bin/time ]", "[ -x '%s' ]" % self.time_bin)
                   .replace("/usr/bin/time -v", "'%s' -v" % self.time_bin))

    def run(self, **extra: str) -> subprocess.CompletedProcess:
        """Run it as SLURM does: stdout and stderr in one stream, which is the job log."""
        env = {"PATH": os.pathsep.join([str(self.bin), "/usr/bin", "/bin"]), "HOME": str(self.root), "USER": "tester",
               "SLURM_SUBMIT_DIR": str(self.repo), "SLURM_JOB_ID": self.JOB, "SLURM_CPUS_PER_TASK": "8",
               "DATA": str(self.data), "SCRATCH_ROOT": str(self.root / "scratch"), "STUB_CALLS": str(self.calls_log)}
        env.update(extra)
        return subprocess.run([BASH, str(self.wrapper)], cwd=str(self.repo), env=env, stdin=subprocess.DEVNULL,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, timeout=120)

    def calls(self) -> List[Dict]:
        if not self.calls_log.exists():
            return []
        return [json.loads(line) for line in self.calls_log.read_text().splitlines()]

    @property
    def out(self) -> Path:
        return self.repo / "results" / ("chat_cpu_probe_" + self.JOB)


@pytest.fixture
def box(tmp_path):
    return ProbeBox(tmp_path)


def results(job_log: str) -> List[Dict]:
    """The RESULT lines, parsed: [run health, drift]."""
    return [json.loads(line[len("RESULT "):]) for line in job_log.splitlines() if line.startswith("RESULT ")]


def fail(*arms: str) -> Dict[str, str]:
    return {"STUB_FAIL": json.dumps({arm: TRACEBACK for arm in arms})}


# ── I1: a throwaway warm-up arm first, the published protocol on every arm ────

def test_arms_run_warm_up_first_then_in_order_on_the_published_protocol(box):
    done = box.run()
    assert done.returncode == 0, done.stdout
    protocol = {"decode": "beam", "beam": "3", "tokens": "100", "config": "hybrid_150m_m3_rrg",
                "ckpt": "last.ckpt", "parquet": "test.parquet"}
    calls = box.calls()
    assert [(c["arm"], c["n"], c["cached"]) for c in calls] == [
        ("warm", 1, True), ("cached_1", 1, True), ("cached_a", 20, True), ("cached_b", 20, True), ("uncached", 5, False)]
    for call in calls:
        assert {key: call[key] for key in protocol} == protocol, call


def test_warm_up_arm_is_never_compared(box):
    """It exists to pay the cold page-cache read. Whatever it writes, RESULT does not look at it."""
    done = box.run(STUB_DRIFT=json.dumps({"warm": [0]}), STUB_SHORT=json.dumps({"warm": 1, "cached_1": 1}))
    assert done.returncode == 0, done.stdout
    health, drift = results(done.stdout)
    assert health["arms_failed"] == [] and health["line_counts_ok"] is True
    assert drift["cached_vs_published_gpu_differ"] == 0 and drift["differing_rows"] == []


def test_footer_explains_the_per_report_formulas_and_the_warm_up(box):
    footer = [l for l in box.run().stdout.splitlines() if l.startswith("=== seconds per report")]
    assert len(footer) == 1
    assert "(wall(cached_a) - wall(cached_1)) / (N - 1)" in footer[0]
    assert "(wall(uncached) - wall(cached_1)) / N_UNCACHED" in footer[0]
    assert "warm" in footer[0]


# ── I2: the failure path names the exception, never its message ───────────────

def test_a_failed_arm_shows_traceback_frames_and_exception_names_with_messages_masked(box):
    done = box.run(**fail("cached_a"))
    log = done.stdout.splitlines()
    assert "ERROR: arm cached_a failed; traceback frames and exception names follow, messages masked" in log
    assert "[probe] Traceback (most recent call last):" in log
    assert '[probe]   File "scripts/evaluate_report_generation.py", line 559, in run_checkpoint_inspection' in log
    assert "[probe] FileNotFoundError: <msg>" in log
    assert "[probe] huggingface_hub.errors.LocalEntryNotFoundError: <msg>" in log, "a dotted name used to be skipped"
    for leak in (STUDY_PATH, "p10000032", "s50414267", "x.jpg", "SYNTHETIC", "Errno", "Image.open"):
        assert leak not in done.stdout, leak


# ── the job log as a whole: summary-shaped, nothing else (R7) ─────────────────

def _shown_by_summary(tmp_path: Path, job_log: str) -> List[str]:
    """What `chat_remote.sh summary` shows of this job log: the script's own grep pattern, then its own mask."""
    cluster = tmp_path / "cluster"
    (cluster / "logs").mkdir(parents=True)
    (cluster / "logs" / "job.log").write_text(job_log)
    sandbox = Sandbox(tmp_path / "sandbox", env_text=fake_env(CLUSTER_REPO=str(cluster)))
    (sandbox.bin / "ssh").write_text(RUNNING_SSH)
    done = sandbox.run("summary", "logs/job.log")
    assert done.returncode == 0, done.stderr
    return done.stdout.splitlines()


@pytest.mark.parametrize("failing", [[], ["uncached"], ARMS])
def test_every_line_the_job_prints_survives_chat_remote_summary_unchanged(box, tmp_path, failing):
    """R7 end to end: nothing the wrapper prints is filtered away, nothing is left for the mask to blank, and no
    report text, study path or study id is in it."""
    done = box.run(STUB_RSS_KB="12345678", **fail(*failing))
    log = done.stdout.splitlines()
    assert all(re.match(r"(\[probe\] |RESULT |=== |ERROR)", l) for l in log), log
    assert _shown_by_summary(tmp_path, done.stdout) == log
    for leak in ("SYNTHETIC", "p10000032", "s50414267"):
        assert leak not in done.stdout, leak


# ── M4: peak RSS in MB ─────────────────────────────────────────────────────────

def test_peak_rss_is_printed_in_mb_so_the_mask_never_sees_an_8_digit_run(box):
    done = box.run(STUB_RSS_KB="12345678")           # about 12 GB: 8 digits in KB, which summary would replace by <num>
    log = done.stdout.splitlines()
    for arm in ARMS:
        assert "[probe] %s: peak_rss_mb=12056" % arm in log, arm    # 12345678 // 1024
        assert "[probe] %s: Elapsed (wall clock) time (h:mm:ss or m:ss): 0:01.23" % arm in log, arm
    assert not [l for l in log if "Maximum resident" in l]
    assert not re.search(r"\d{8,}", done.stdout)


# ── M2: observed line counts; M5: a result over whatever finished ─────────────

@pytest.mark.parametrize("drift, expected", [
    ({"cached_a": [3], "cached_b": [3]},
     {"cached_vs_cached_differ": 0, "cached_vs_uncached_cpu_differ": 1, "cached_vs_published_gpu_differ": 1,
      "uncached_vs_published_gpu_differ": 0, "differing_rows": [3]}),
    ({"cached_b": [7, 9]},
     {"cached_vs_cached_differ": 2, "cached_vs_uncached_cpu_differ": 0, "cached_vs_published_gpu_differ": 0,
      "uncached_vs_published_gpu_differ": 0, "differing_rows": []}),
    ({"uncached": [0, 4]},
     {"cached_vs_cached_differ": 0, "cached_vs_uncached_cpu_differ": 2, "cached_vs_published_gpu_differ": 0,
      "uncached_vs_published_gpu_differ": 2, "differing_rows": []}),
])
def test_result_reports_observed_line_counts_and_the_drift_between_arms(box, drift, expected):
    done = box.run(STUB_DRIFT=json.dumps(drift))
    assert done.returncode == 0, done.stdout
    health, drift_line = results(done.stdout)
    assert health == {"n": 20, "n_uncached": 5, "arms_failed": [], "line_counts_ok": True,
                      "lines": {"cached_a": 20, "cached_b": 20, "uncached": 5, "published": 30,
                                "published_n": 20, "published_nu": 5}}
    assert drift_line == expected
    assert json.loads((box.out / "summary.json").read_text()) == dict(health, **drift_line)


def test_a_short_arm_shows_in_the_counts_instead_of_truncating_silently(box):
    done = box.run(STUB_SHORT=json.dumps({"cached_b": 3, "uncached": 1}))
    health, drift = results(done.stdout)
    assert health["lines"] == {"cached_a": 20, "cached_b": 17, "uncached": 4, "published": 30,
                               "published_n": 20, "published_nu": 5}
    assert health["line_counts_ok"] is False
    assert drift["cached_vs_cached_differ"] == 0, "zip() stops at the shorter list: why the counts are printed"


def test_a_published_dump_shorter_than_n_shows_in_the_counts(box):
    box.published.write_text("".join("synthetic report %d\n" % i for i in range(12)))
    health, _ = results(box.run().stdout)
    assert health["lines"]["published"] == 12 and health["lines"]["published_n"] == 12
    assert health["lines"]["published_nu"] == 5 and health["line_counts_ok"] is False


@pytest.mark.parametrize("arm, nulls", [
    ("warm", []),
    ("cached_1", []),
    ("cached_a", ["cached_vs_cached_differ", "cached_vs_uncached_cpu_differ", "cached_vs_published_gpu_differ",
                  "differing_rows"]),
    ("cached_b", ["cached_vs_cached_differ"]),
    ("uncached", ["cached_vs_uncached_cpu_differ", "uncached_vs_published_gpu_differ"]),
])
def test_a_failed_arm_still_gets_a_result_over_the_arms_that_finished_and_the_job_exits_1(box, arm, nulls):
    done = box.run(**fail(arm))
    assert done.returncode == 1, done.stdout
    assert [c["arm"] for c in box.calls()] == ARMS, "the arms after the failed one still ran"
    health, drift = results(done.stdout)
    assert health["arms_failed"] == [arm]
    seen = health["lines"]                  # only cached_a, cached_b and uncached are compared; warm and cached_1 just time
    if arm in seen:
        assert seen[arm] is None
    assert health["line_counts_ok"] is (arm not in seen)
    assert sorted(k for k, v in drift.items() if v is None) == sorted(nulls)
    assert "=== ARMS FAILED: %s ===" % arm in done.stdout.splitlines()
    assert json.loads((box.out / "summary.json").read_text()) == dict(health, **drift)


def test_a_failed_arm_does_not_borrow_a_stale_dump_from_an_earlier_run(box):
    assert box.run().returncode == 0               # leaves hyps.txt for every arm in the same output directory
    done = box.run(**fail("cached_a"))
    health, drift = results(done.stdout)
    assert health["lines"]["cached_a"] is None and drift["cached_vs_published_gpu_differ"] is None


def test_result_lines_stay_under_the_300_characters_summary_keeps(box):
    for extra in ({"STUB_DRIFT": json.dumps({"cached_a": list(range(20))})}, fail(*ARMS)):
        found = [l for l in box.run(**extra).stdout.splitlines() if l.startswith("RESULT ")]
        assert len(found) == 2 and max(len(l) for l in found) <= 300, [len(l) for l in found]


# ── M3 / M1: inputs named by basename; GNU time is required ───────────────────

@pytest.mark.parametrize("attr, shown", [("ckpt", "last.ckpt"), ("published", "hyps.txt"), ("parquet", "test.parquet")])
def test_a_missing_input_is_named_by_basename_only_and_no_arm_runs(box, attr, shown):
    getattr(box, attr).unlink()
    done = box.run()
    assert done.returncode == 1
    assert done.stdout.splitlines() == ["ERROR: not found: %s" % shown]
    assert box.calls() == []


def test_the_wrapper_requires_usr_bin_time_instead_of_running_untimed(tmp_path):
    box = ProbeBox(tmp_path, time_present=False)
    done = box.run()
    assert done.returncode == 1
    assert done.stdout.splitlines() == ["ERROR: /usr/bin/time missing"]
    assert box.calls() == []
