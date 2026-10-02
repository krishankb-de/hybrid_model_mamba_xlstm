"""CHAT_UI_PLAN.md P2-E: the golden driver (scripts/chat_engine_golden.py) and its two SLURM wrappers.

The driver runs in-process over the tiny engine, with synthetic images and a synthetic parquet (nothing here touches a
checkpoint, MIMIC or the cluster). The wrappers run for real in a temp tree: `python` is a stub that execs this
interpreter, the eval script and the driver are stubs that record how they were called and write synthetic hyps.txt
files, and `lscpu`/`hostname` are stubs. Synthetic data only (R7): the "report text" and the study path in a stub's
traceback are made up, and the tests assert that none of it reaches the job log.
"""
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import threading
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest
from PIL import Image

from tests.test_chat_remote import RUNNING_SSH, Sandbox, fake_env

REPO_ROOT = Path(__file__).resolve().parent.parent
DRIVER = REPO_ROOT / "scripts" / "chat_engine_golden.py"
CPU_SH = REPO_ROOT / "scripts" / "chat_engine_golden_h100.sh"
GPU_SH = REPO_ROOT / "scripts" / "chat_engine_golden_gpu_h100.sh"
BASH = "/bin/bash" if os.path.exists("/bin/bash") else "bash"   # the Mac's 3.2 is the oldest shell it has to work in
STUDY_PATH = "/d/files/p10/p10000032/s50414267/x.jpg"            # a made-up MIMIC-shaped path (R7)


# ── the driver, in-process over the tiny engine ──────────────────────────────────────────────────────────────

def _gray_jpeg(kind: str) -> bytes:
    """Four synthetic 96x96 studies the tiny model reports differently on (a constant image would repeat one report)."""
    if kind == "noise":
        arr = np.random.default_rng(0).integers(0, 255, (96, 96))
    elif kind == "checker":
        arr = ((np.indices((96, 96)).sum(0) // 8) % 2) * 255
    else:
        arr = np.full((96, 96), {"white": 255, "mid": 128}[kind])
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8)).save(buf, "JPEG", quality=95)
    return buf.getvalue()


@pytest.fixture(scope="module")
def golden():
    """The driver module, imported by path so this file does not depend on how pytest put the repo on sys.path."""
    spec = importlib.util.spec_from_file_location("chat_engine_golden_under_test", str(DRIVER))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def studies(tmp_path):
    """A synthetic test.parquet of 4 rows whose `image` column holds absolute paths of synthetic JPEG files."""
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    folder = tmp_path / "files" / "p10" / "p10000032" / "s50414267"
    folder.mkdir(parents=True)
    paths = []
    for kind in ("noise", "white", "checker", "mid"):
        path = folder / (kind + ".jpg")
        path.write_bytes(_gray_jpeg(kind))
        paths.append(path)
    parquet = tmp_path / "test.parquet"
    pd.DataFrame({"image": [str(p) for p in paths], "study_id": [50414267 + i for i in range(4)],
                  "findings": ["SYNTHETIC findings"] * 4, "impression": ["SYNTHETIC impression"] * 4}).to_parquet(parquet)
    checkpoint = tmp_path / "last.ckpt"
    checkpoint.write_bytes(b"x")
    return {"paths": paths, "parquet": parquet, "checkpoint": checkpoint, "out": tmp_path / "out"}


def _report_per_study(paths, cached: bool) -> List[str]:
    """What the engine says for these files, decoded independently of the driver (the tiny engine is deterministic)."""
    from app.engine import TinyEngine
    from app.schemas import Options
    eng = TinyEngine()
    reports = []
    for path in paths:
        _, prep = eng.preprocess(path.read_bytes())
        _, enc = eng.encode(prep)
        reports.append(eng.generate(enc, Options(cached_decode=cached), lambda s, t: None, threading.Event())[1].report)
    return reports


@pytest.fixture
def run_driver(golden, studies, monkeypatch, capsys):
    """Run golden.main over the tiny engine; returns what it built, saw and printed."""
    import app.engine as engine_module

    def run(*flags, n: int = 3):
        built, opts_seen, bytes_seen = [], [], []
        real_pre, real_gen = engine_module.TinyEngine.preprocess, engine_module.TinyEngine.generate

        def fake_build(kind, **kw):
            built.append((kind, kw))
            return engine_module.TinyEngine()

        def pre(self, data):
            bytes_seen.append(data)
            return real_pre(self, data)

        def gen(self, enc, opts, on_snapshot, cancel):
            opts_seen.append(opts)
            return real_gen(self, enc, opts, on_snapshot, cancel)

        monkeypatch.setattr(engine_module, "build_engine", fake_build)
        monkeypatch.setattr(engine_module.TinyEngine, "preprocess", pre)
        monkeypatch.setattr(engine_module.TinyEngine, "generate", gen)
        capsys.readouterr()
        golden.main(["--checkpoint", str(studies["checkpoint"]), "--parquet", str(studies["parquet"]), "--n", str(n),
                     "--out", str(studies["out"])] + list(flags))
        return {"built": built, "opts": opts_seen, "bytes": bytes_seen, "stdout": capsys.readouterr().out}

    return run


def test_driver_puts_the_repo_root_on_sys_path_when_run_as_a_script(tmp_path):
    """The cluster .venv has no editable install, and `python scripts/x.py` puts scripts/, not the repo root, on
    sys.path: `from app.engine import ...` and `from scripts.evaluate_report_generation import ...` need the insert.
    `-S` skips site, so no editable-install .pth can hide a missing one."""
    code = ("import importlib.util, sys; s = importlib.util.spec_from_file_location('g', %r); "
            "m = importlib.util.module_from_spec(s); s.loader.exec_module(m); print(sys.path[0])" % str(DRIVER))
    done = subprocess.run([sys.executable, "-S", "-c", code], cwd=str(tmp_path), capture_output=True, text=True,
                          timeout=60)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == str(REPO_ROOT)


def test_driver_builds_the_real_engine_on_the_published_protocol(run_driver, studies):
    ran = run_driver()
    (kind, kw), = ran["built"]
    assert kind == "real"
    assert kw["checkpoint"] == str(studies["checkpoint"]) and kw["model_config"] == "hybrid_150m_m3_rrg"
    assert kw["device"] in ("cpu", "cuda") and kw["threads"] == 8
    assert len(ran["opts"]) == 3
    for opts in ran["opts"]:   # R2: the defaults ARE the published protocol, and the O(1) cache is on
        assert (opts.decode, opts.beam_size, opts.max_new_tokens, opts.cached_decode) == ("beam", 3, 100, True)


def test_driver_passes_model_config_and_threads_through(run_driver):
    ran = run_driver("--model-config", "hybrid_150m_v2_rrg", "--threads", "3", "--uncached", n=1)
    assert ran["built"][0][1]["model_config"] == "hybrid_150m_v2_rrg" and ran["built"][0][1]["threads"] == 3


def test_uncached_flag_selects_the_uncached_path_and_nothing_else(run_driver, studies):
    ran = run_driver("--uncached", n=1)
    assert [(o.decode, o.beam_size, o.max_new_tokens, o.cached_decode) for o in ran["opts"]] == [("beam", 3, 100, False)]
    hyps = (studies["out"] / "hyps.txt").read_text().splitlines()
    assert hyps == _report_per_study(studies["paths"][:1], cached=False)
    assert json.loads((studies["out"] / "timings.json").read_text())["card"]["name"] == "tiny"


def test_driver_decodes_the_first_n_files_from_their_bytes_through_the_engine_stages(run_driver, studies):
    expected = _report_per_study(studies["paths"][:3], cached=True)
    assert len(set(expected)) > 1, "the rows must differ, or this test cannot tell one row from another"
    ran = run_driver(n=3)
    assert ran["bytes"] == [p.read_bytes() for p in studies["paths"][:3]]
    assert all(type(b) is bytes for b in ran["bytes"]), "an upload is bytes, not a path"
    assert (studies["out"] / "hyps.txt").read_text().splitlines() == expected


def test_hyps_file_is_byte_identical_to_what_write_hyps_refs_writes(run_driver, studies, tmp_path, capsys):
    """'sanitised like write_hyps_refs': the same bytes the eval script's --dump-dir would hold for these reports."""
    from scripts.evaluate_report_generation import write_hyps_refs
    expected = _report_per_study(studies["paths"][:3], cached=True)
    run_driver(n=3)
    write_hyps_refs(str(tmp_path / "ref"), expected, expected)
    capsys.readouterr()
    assert (studies["out"] / "hyps.txt").read_bytes() == (tmp_path / "ref" / "hyps.txt").read_bytes()
    assert (studies["out"] / "hyps.txt").read_bytes().endswith(b"\n")


def test_driver_collapses_whitespace_so_one_report_is_one_line(golden, studies, monkeypatch):
    import app.engine as engine_module
    from app.engine import Generated, StageResult
    messy = "Findings:  heart\tnormal.\nImpression:\n\n no   effusion. "
    monkeypatch.setattr(engine_module, "build_engine", lambda kind, **kw: engine_module.TinyEngine())
    monkeypatch.setattr(engine_module.TinyEngine, "generate", lambda self, enc, opts, on_snapshot, cancel: (
        StageResult({"prefill_ms": 1.0, "per_token_ms": 2.0, "tokens": 3}, 4.0), Generated([1, 2, 3], messy, messy, False)))
    golden.main(["--checkpoint", str(studies["checkpoint"]), "--parquet", str(studies["parquet"]), "--n", "2",
                 "--out", str(studies["out"])])
    assert (studies["out"] / "hyps.txt").read_text() == "Findings: heart normal. Impression: no effusion.\n" * 2


def test_timings_json_has_the_device_the_card_and_one_row_per_study(run_driver, studies):
    run_driver(n=3)
    timings = json.loads((studies["out"] / "timings.json").read_text())
    assert set(timings) == {"device", "device_name", "card", "rows"}
    assert timings["device"] in ("cpu", "cuda") and isinstance(timings["device_name"], str)
    assert timings["card"]["git_source"] in ("git", "sync_stamp", None) and "checkpoint_sha256" in timings["card"]
    assert [r["row"] for r in timings["rows"]] == [0, 1, 2]
    for row in timings["rows"]:
        assert set(row) == {"row", "preprocess_ms", "encode_ms", "prefill_ms", "per_token_ms", "generate_ms", "tokens"}
        assert all(isinstance(v, (int, float)) and v >= 0 for v in row.values())
        assert 0 < row["tokens"] <= 100


def test_driver_prints_only_golden_summary_lines_and_never_report_text_or_paths(run_driver, studies):
    ran = run_driver(n=3)
    lines = ran["stdout"].splitlines()
    assert lines and all(l.startswith("[golden] ") for l in lines), lines
    assert [l.split()[2] for l in lines if l.startswith("[golden] row ")] == ["0", "1", "2"]
    summary = [l for l in lines if l.startswith("[golden] summary ")]
    assert len(summary) == 1 and "n=3" in summary[0] and "encode_ms_median=" in summary[0]
    assert "cached=True" in summary[0] and "per_token_ms_median=" in summary[0]
    leaks = _report_per_study(studies["paths"][:3], cached=True) + [str(p) for p in studies["paths"]] + [
        "p10000032", "s50414267", "SYNTHETIC"]
    for leak in leaks:
        assert leak not in ran["stdout"], leak
    assert not re.search(r"\d{8,}", ran["stdout"])


def test_driver_states_what_it_ran_on_in_its_first_lines(run_driver):
    lines = run_driver(n=1)["stdout"].splitlines()
    assert lines[0].startswith("[golden] device=") and "threads=" in lines[0] and "torch=" in lines[0]
    assert lines[1].startswith("[golden] card name=tiny") and "git=" in lines[1] and "source=" in lines[1]
    assert "cached_decode_available=True" in lines[1]


def test_an_n_beyond_the_parquet_decodes_every_row_and_pads_nothing(run_driver, studies):
    run_driver(n=50)
    assert len((studies["out"] / "hyps.txt").read_text().splitlines()) == 4


# ── the wrappers, run for real in a temp tree ────────────────────────────────────────────────────────────────

# Stands in for evaluate_report_generation.py: logs its argv as JSON, prints what the real one prints (including report
# text, which must never reach the job log), then either fails with the traceback in $STUB_FAIL[arm] or writes hyps.txt
# ("synthetic report <i>", with $STUB_DRIFT rows altered and $STUB_SHORT[arm] lines missing at the end).
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
    fh.write(json.dumps({"tool": "eval", "arm": arm, "n": int(args.num_samples), "cached": args.cached_decode,
                         "decode": args.decode, "beam": args.beam_size, "tokens": args.max_new_tokens,
                         "config": args.model_config, "ckpt": os.path.basename(args.checkpoint),
                         "parquet": os.path.basename(args.parquet)}) + "\n")
print("Loaded checkpoint: /sc/home/x/outputs/run/checkpoints/last.ckpt")
print("  Missing keys: 0, Unexpected: 0")
print("  prefix_k = 32")
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

# Stands in for chat_engine_golden.py, the same way. $STUB_SILENT makes it print nothing at all.
ENGINE_STUB = r'''
import argparse, json, os, sys
parser = argparse.ArgumentParser()
for flag in ("--checkpoint", "--model-config", "--parquet", "--n", "--out", "--threads"):
    parser.add_argument(flag)
parser.add_argument("--uncached", action="store_true")
args = parser.parse_args()
arm = os.path.basename(args.out.rstrip("/"))
with open(os.environ["STUB_CALLS"], "a") as fh:
    fh.write(json.dumps({"tool": "engine", "arm": arm, "n": int(args.n), "uncached": args.uncached,
                         "threads": args.threads, "config": args.model_config,
                         "ckpt": os.path.basename(args.checkpoint), "parquet": os.path.basename(args.parquet)}) + "\n")
if not os.environ.get("STUB_SILENT"):
    print("Loaded checkpoint: /sc/home/x/outputs/run/checkpoints/last.ckpt")
    print("  Missing keys: 0, Unexpected: 0")
    print("  prefix_k = 32")
    print("[golden] device=cpu name=cpu torch=2.0 threads=%s" % args.threads, flush=True)
    for i in range(int(args.n)):
        print("[golden] row %d encode_ms=1.0 prefill_ms=2.0 per_token_ms=3.0 generate_ms=4.0" % i, flush=True)
    print("GENERATED: SYNTHETIC report text")
fail = json.loads(os.environ.get("STUB_FAIL", "{}"))
if arm in fail:
    sys.stderr.write(fail[arm])
    sys.exit(1)
hyps = ["synthetic report %d" % i for i in range(int(args.n))]
for row in json.loads(os.environ.get("STUB_DRIFT", "{}")).get(arm, []):
    hyps[row] += " drift"
hyps = hyps[:len(hyps) - json.loads(os.environ.get("STUB_SHORT", "{}")).get(arm, 0)]
os.makedirs(args.out, exist_ok=True)
with open(os.path.join(args.out, "hyps.txt"), "w") as fh:
    fh.write("".join(h + "\n" for h in hyps))
with open(os.path.join(args.out, "timings.json"), "w") as fh:
    json.dump({"device": "cpu", "device_name": "cpu", "card": {}, "rows": []}, fh)
'''

# The GPU wrapper asks `python -c` whether torch sees a CUDA device (and for its name); everything else runs for real.
PYTHON_STUB = """#!/bin/bash
if [ "$1" = "-c" ]; then [ -n "${STUB_NO_CUDA:-}" ] || echo "STUB GPU 9000"; exit 0; fi
exec "%s" "$@"
"""

LSCPU_STUB = """#!/bin/bash
printf 'Architecture:          x86_64\\nModel name:            Stub CPU 9000 @ 3.00GHz\\nCPU(s):                8\\n'
"""

# What a failed step leaves in its own log: a study path in the first message, a dotted class name in the second.
TRACEBACK = (
    "Traceback (most recent call last):\n"
    '  File "scripts/chat_engine_golden.py", line 70, in main\n'
    '    img = Image.open(row["image"]).convert("RGB")\n'
    "FileNotFoundError: [Errno 2] No such file or directory: '%s'\n"
    "\n"
    "During handling of the above exception, another exception occurred:\n"
    "\n"
    "Traceback (most recent call last):\n"
    '  File "scripts/chat_engine_golden.py", line 88, in <module>\n'
    "huggingface_hub.errors.LocalEntryNotFoundError: SYNTHETIC findings text\n"
) % STUDY_PATH


class GoldenBox:
    """One wrapper copied into a temp tree with a fake workspace around it: checkpoint, a 30-line published dump, a test
    parquet, the eval script and the driver as stubs, and stubs for python, lscpu and hostname."""

    JOB = "4242"

    def __init__(self, root: Path, kind: str):
        self.kind, self.root = kind, root
        self.repo, self.bin, self.data = root / "repo", root / "bin", root / "data"
        self.calls_log = root / "calls.jsonl"
        self.ckpt = self.repo / "outputs" / "h100_report_gen_m3_tower13d_s42" / "checkpoints" / "last.ckpt"
        self.published = self.repo / "results" / "report_gen_m3_test_split_s42" / "hyps.txt"
        self.parquet = self.data / "test.parquet"
        activate = self.repo / ".venv" / "bin" / "activate"
        scripts = self.repo / "scripts"
        for directory in (self.bin, self.ckpt.parent, self.published.parent, self.data, activate.parent, scripts):
            directory.mkdir(parents=True, exist_ok=True)
        for path in (self.ckpt, self.parquet, activate):
            path.write_text("")
        self.published.write_text("".join("synthetic report %d\n" % i for i in range(30)))
        (scripts / "evaluate_report_generation.py").write_text(EVAL_STUB)
        (scripts / "chat_engine_golden.py").write_text(ENGINE_STUB)
        self._stub("python", PYTHON_STUB % sys.executable)
        self._stub("lscpu", LSCPU_STUB)
        self._stub("hostname", "#!/bin/bash\necho stubnode\n")
        source = CPU_SH if kind == "cpu" else GPU_SH
        self.wrapper = scripts / source.name
        self.wrapper.write_text(source.read_text())

    def _stub(self, name: str, text: str) -> None:
        (self.bin / name).write_text(text)
        (self.bin / name).chmod(0o755)

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
        return self.repo / "results" / ("chat_golden_" + self.JOB)


@pytest.fixture
def cpu(tmp_path):
    return GoldenBox(tmp_path, "cpu")


@pytest.fixture
def gpu(tmp_path):
    return GoldenBox(tmp_path, "gpu")


@pytest.fixture(params=["cpu", "gpu"])
def box(request, tmp_path):
    return GoldenBox(tmp_path, request.param)


CPU_ARMS = ["script_cached", "engine_cached"]
GPU_ARMS = ["engine_uncached", "engine_cached"]


def results(job_log: str) -> List[Dict]:
    """The RESULT lines, parsed: [the counts, the differing rows]."""
    return [json.loads(line[len("RESULT "):]) for line in job_log.splitlines() if line.startswith("RESULT ")]


def merged(job_log: str) -> Dict:
    out = {}
    for part in results(job_log):
        out.update(part)
    return out


def fail(*arms: str) -> Dict[str, str]:
    return {"STUB_FAIL": json.dumps({arm: TRACEBACK for arm in arms})}


# ── what each wrapper asks SLURM for ─────────────────────────────────────────────────────────────────────────

def _directives(path: Path) -> List[str]:
    return [l for l in path.read_text().splitlines() if l.startswith("#SBATCH")]


def test_cpu_wrapper_asks_for_the_proven_cpu_only_combination():
    directives = _directives(CPU_SH)
    for wanted in ("#SBATCH --partition=pot-hpi-aisc-batch", "#SBATCH --account=aisc", "#SBATCH --qos=aisc",
                   "#SBATCH --cpus-per-task=8", "#SBATCH --time=03:00:00", "#SBATCH --job-name=chat_engine_golden",
                   "#SBATCH --output=logs/%x_%j.log", "#SBATCH --error=logs/%x_%j.log"):
        assert wanted in directives, wanted
    assert any(l.startswith("#SBATCH --exclude=ga03,gx17v1,gx13v1") for l in directives)
    assert not [l for l in directives if "--gpus" in l or "--gres" in l], "CPU-only job"


def test_gpu_wrapper_asks_for_one_gpu_by_gpus_and_no_qos():
    directives = _directives(GPU_SH)
    for wanted in ("#SBATCH --partition=pot-hpi-aisc-batch", "#SBATCH --account=aisc", "#SBATCH --gpus=1",
                   "#SBATCH --time=01:00:00", "#SBATCH --job-name=chat_engine_golden_gpu",
                   "#SBATCH --output=logs/%x_%j.log", "#SBATCH --error=logs/%x_%j.log"):
        assert wanted in directives, wanted
    assert any(l.startswith("#SBATCH --exclude=ga03,gx17v1,gx13v1") for l in directives), "it sources the x86 venv"
    assert not [l for l in directives if "--qos" in l or "--gres" in l], "as every GPU wrapper in the repo"


@pytest.mark.parametrize("path", [CPU_SH, GPU_SH])
def test_wrappers_are_additive_and_read_the_published_dump_only(path):
    """R8: no deletion, nothing written outside the new results/chat_golden_<job> directory. R7: the dump is only compared."""
    src = path.read_text()
    code = [l for l in src.splitlines() if not l.lstrip().startswith("#")]
    assert not [l for l in code if re.search(r"\b(rm|rmdir|mv|cp|ln|truncate)\b", l)], "no deletion, no overwrite"
    assert 'OUT="${OUT:-results/chat_golden_${SLURM_JOB_ID:-local}}"' in src
    assert 'PUBLISHED="${PUBLISHED:-results/report_gen_m3_test_split_s42/hyps.txt}"' in src
    assert 'DATA="${DATA:-/sc/home/$USER/dataset/mimic_full}"' in src
    assert "HF_HUB_OFFLINE" in src and "OMP_NUM_THREADS" in src and "cat \"" not in src


# ── what each wrapper runs ───────────────────────────────────────────────────────────────────────────────────

def test_cpu_wrapper_decodes_with_the_script_then_the_engine_on_the_published_protocol(cpu):
    done = cpu.run()
    assert done.returncode == 0, done.stdout
    protocol = {"config": "hybrid_150m_m3_rrg", "ckpt": "last.ckpt", "parquet": "test.parquet"}
    script, engine = cpu.calls()
    assert (script["tool"], script["arm"], script["n"], script["cached"]) == ("eval", "script_cached", 20, True)
    assert (script["decode"], script["beam"], script["tokens"]) == ("beam", "3", "100")
    assert (engine["tool"], engine["arm"], engine["n"], engine["uncached"], engine["threads"]) == (
        "engine", "engine_cached", 20, False, "8")
    for call in (script, engine):
        assert {key: call[key] for key in protocol} == protocol, call
    assert (cpu.out / "script_cached" / "hyps.txt").exists() and (cpu.out / "engine_cached" / "hyps.txt").exists()


def test_gpu_wrapper_decodes_uncached_then_cached_through_the_engine_only(gpu):
    done = gpu.run()
    assert done.returncode == 0, done.stdout
    uncached, cached = gpu.calls()
    assert [c["tool"] for c in (uncached, cached)] == ["engine", "engine"], "no script arm: the published dump is the reference"
    assert (uncached["arm"], uncached["uncached"], uncached["n"]) == ("engine_uncached", True, 20)
    assert (cached["arm"], cached["uncached"], cached["n"]) == ("engine_cached", False, 20)
    for call in (uncached, cached):
        assert (call["config"], call["ckpt"], call["parquet"], call["threads"]) == (
            "hybrid_150m_m3_rrg", "last.ckpt", "test.parquet", "8")


def test_n_is_overridable_and_reaches_every_arm(box):
    done = box.run(N="7")
    assert done.returncode == 0, done.stdout
    assert [c["n"] for c in box.calls()] == [7, 7]
    assert merged(done.stdout)["n"] == 7 and merged(done.stdout)["lengths"] == [7, 7, 7]


# ── the CPU verdict: engine == script on the same node is the hard gate (R2) ─────────────────────────────────

def test_cpu_identical_arms_pass_and_report_the_counts(cpu):
    done = cpu.run()
    assert done.returncode == 0, done.stdout
    counts, rows = results(done.stdout)
    assert counts == {"n": 20, "lengths": [20, 20, 20], "lines_ok": True, "bytes_equal": True,
                      "engine_vs_script_differ": 0, "engine_vs_published_gpu_differ": 0, "gate_passed": True}
    assert rows == {"rows_vs_script": [], "rows_vs_published": []}
    assert json.loads((cpu.out / "golden.json").read_text()) == dict(counts, **rows)
    assert "=== GOLDEN CPU PASSED: engine == script ===" in done.stdout.splitlines()


def test_cpu_drift_from_the_published_gpu_dump_is_reported_and_is_not_fatal(cpu):
    both = {"script_cached": [3, 8], "engine_cached": [3, 8]}      # the CPU node says something else than the H100 did
    done = cpu.run(STUB_DRIFT=json.dumps(both))
    assert done.returncode == 0, done.stdout
    counts, rows = results(done.stdout)
    assert counts["engine_vs_script_differ"] == 0 and counts["engine_vs_published_gpu_differ"] == 2
    assert counts["gate_passed"] is True and rows == {"rows_vs_script": [], "rows_vs_published": [3, 8]}


def test_cpu_an_engine_that_differs_from_the_script_fails_the_job(cpu):
    done = cpu.run(STUB_DRIFT=json.dumps({"engine_cached": [5]}))
    assert done.returncode == 1, done.stdout
    counts, rows = results(done.stdout)
    assert counts["engine_vs_script_differ"] == 1 and counts["bytes_equal"] is False and counts["gate_passed"] is False
    assert rows["rows_vs_script"] == [5]
    assert "=== GOLDEN CPU FAILED ===" in done.stdout.splitlines()


@pytest.mark.parametrize("short", [{"engine_cached": 1}, {"script_cached": 2}, {"engine_cached": 3, "script_cached": 3}])
def test_cpu_a_short_arm_shows_in_the_lengths_and_fails_the_job(cpu, short):
    """zip() stops at the shorter list: without the observed counts a truncated arm would look identical."""
    done = cpu.run(STUB_SHORT=json.dumps(short))
    counts = merged(done.stdout)
    assert done.returncode == 1, done.stdout
    assert counts["lengths"] == [20 - short.get("engine_cached", 0), 20 - short.get("script_cached", 0), 20]
    assert counts["lines_ok"] is False and counts["gate_passed"] is False


def test_cpu_a_published_dump_shorter_than_n_shows_in_the_lengths_and_fails_the_job(cpu):
    cpu.published.write_text("".join("synthetic report %d\n" % i for i in range(12)))
    done = cpu.run()
    assert done.returncode == 1
    assert merged(done.stdout)["lengths"] == [20, 20, 12] and merged(done.stdout)["lines_ok"] is False


def test_cpu_script_arm_is_a_hard_prerequisite(cpu):
    done = cpu.run(**fail("script_cached"))
    assert done.returncode == 1
    assert [c["arm"] for c in cpu.calls()] == ["script_cached"], "no engine arm, no comparison, after a failed script arm"
    assert not [l for l in done.stdout.splitlines() if l.startswith("RESULT ")]


# ── the GPU verdict: uncached engine == published dump is the hard gate; cached is measured (R2, M6-D at scale) ──

def test_gpu_identical_arms_pass_and_report_the_counts(gpu):
    done = gpu.run()
    assert done.returncode == 0, done.stdout
    counts, rows = results(done.stdout)
    assert counts == {"n": 20, "lengths": [20, 20, 20], "lines_ok": True, "engine_uncached_vs_published_differ": 0,
                      "engine_cached_vs_published_differ": 0, "engine_cached_vs_uncached_differ": 0, "gate_passed": True}
    assert rows == {"rows_uncached_vs_published": [], "rows_cached_vs_published": []}
    assert json.loads((gpu.out / "golden.json").read_text()) == dict(counts, **rows)
    assert "=== GOLDEN GPU PASSED: engine (uncached) == published ===" in done.stdout.splitlines()


def test_gpu_a_cached_arm_that_differs_is_measured_and_not_fatal(gpu):
    done = gpu.run(STUB_DRIFT=json.dumps({"engine_cached": [4, 11]}))
    assert done.returncode == 0, done.stdout
    counts, rows = results(done.stdout)
    assert counts["engine_uncached_vs_published_differ"] == 0 and counts["engine_cached_vs_published_differ"] == 2
    assert counts["engine_cached_vs_uncached_differ"] == 2 and counts["gate_passed"] is True
    assert rows == {"rows_uncached_vs_published": [], "rows_cached_vs_published": [4, 11]}


def test_gpu_an_uncached_arm_that_differs_from_the_published_dump_fails_the_job(gpu):
    done = gpu.run(STUB_DRIFT=json.dumps({"engine_uncached": [2, 9]}))
    assert done.returncode == 1, done.stdout
    counts, rows = results(done.stdout)
    assert counts["engine_uncached_vs_published_differ"] == 2 and counts["engine_cached_vs_uncached_differ"] == 2
    assert counts["gate_passed"] is False and rows["rows_uncached_vs_published"] == [2, 9]
    assert "=== GOLDEN GPU FAILED ===" in done.stdout.splitlines()


@pytest.mark.parametrize("short", [{"engine_uncached": 1}, {"engine_cached": 2}])
def test_gpu_a_short_arm_shows_in_the_lengths_and_fails_the_job(gpu, short):
    done = gpu.run(STUB_SHORT=json.dumps(short))
    assert done.returncode == 1, done.stdout
    assert merged(done.stdout)["lengths"] == [20 - short.get("engine_uncached", 0), 20 - short.get("engine_cached", 0), 20]


def test_gpu_a_failed_uncached_arm_stops_the_job_before_the_cached_arm(gpu):
    done = gpu.run(**fail("engine_uncached"))
    assert done.returncode == 1
    assert [c["arm"] for c in gpu.calls()] == ["engine_uncached"]


def test_gpu_without_a_cuda_device_stops_before_decoding_anything(gpu):
    """A node where torch cannot see the GPU would decode on the CPU, and a CPU-vs-H100 'difference' would be reported
    as the golden failing."""
    done = gpu.run(STUB_NO_CUDA="1")
    assert done.returncode == 1
    assert done.stdout.splitlines() == ["ERROR: CUDA unavailable on this node"]
    assert gpu.calls() == []


def test_gpu_names_the_device_it_will_decode_on(gpu):
    log = gpu.run().stdout.splitlines()
    assert "[golden] gpu STUB GPU 9000 node=stubnode n=20" in log


def test_cpu_names_the_cpu_it_will_decode_on(cpu):
    log = cpu.run().stdout.splitlines()
    assert "[golden] cpu Stub CPU 9000 @ 3.00GHz ncpu=8 node=stubnode n=20" in log


# ── the failure path names the exception, never its message (R7) ─────────────────────────────────────────────

@pytest.mark.parametrize("kind, arm", [("cpu", "script_cached"), ("cpu", "engine_cached"), ("gpu", "engine_uncached"),
                                       ("gpu", "engine_cached")])
def test_a_failed_arm_shows_traceback_frames_and_exception_names_with_messages_masked(tmp_path, kind, arm):
    box = GoldenBox(tmp_path, kind)
    done = box.run(**fail(arm))
    log = done.stdout.splitlines()
    assert done.returncode == 1
    assert "ERROR: %s failed; traceback frames and exception names follow, messages masked" % arm in log
    assert "[golden] Traceback (most recent call last):" in log
    assert '[golden]   File "scripts/chat_engine_golden.py", line 70, in main' in log
    assert "[golden] FileNotFoundError: <msg>" in log
    assert "[golden] huggingface_hub.errors.LocalEntryNotFoundError: <msg>" in log, "a dotted name must show too"
    for leak in (STUDY_PATH, "p10000032", "s50414267", "x.jpg", "SYNTHETIC", "Errno", "Image.open"):
        assert leak not in done.stdout, leak


def test_a_failing_arm_with_no_matching_lines_in_its_log_still_ends_in_an_error_line(box):
    done = box.run(STUB_FAIL=json.dumps({"engine_cached": "boom\n"}), STUB_SILENT="1")
    assert done.returncode == 1
    assert "ERROR: engine_cached failed; traceback frames and exception names follow, messages masked" in done.stdout


def test_a_silent_driver_does_not_kill_the_wrapper_through_grep_exit_status(box):
    """grep exits 1 on no match; with `set -eo pipefail` an unguarded filter would end the job there."""
    done = box.run(STUB_SILENT="1")
    assert done.returncode == 0, done.stdout
    assert merged(done.stdout)["gate_passed"] is True


# ── the job log as a whole: summary-shaped, nothing else (R7) ────────────────────────────────────────────────

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


def _scenarios(kind: str) -> List[Dict[str, str]]:
    arms = CPU_ARMS if kind == "cpu" else GPU_ARMS
    return [{}, {"STUB_DRIFT": json.dumps({arms[1]: list(range(20))})}, fail(arms[0]), fail(arms[1]),
            {"STUB_SHORT": json.dumps({arms[1]: 4})}, {"STUB_SILENT": "1"}]


@pytest.mark.parametrize("kind", ["cpu", "gpu"])
def test_every_line_the_job_prints_survives_chat_remote_summary_unchanged(tmp_path, kind):
    """R7 end to end, over a pass, total drift, each arm failing, a short arm and a silent driver: nothing the wrapper
    prints is filtered away, nothing is left for the mask to blank, and no report text, study path or id is in it."""
    for i, extra in enumerate(_scenarios(kind)):
        box = GoldenBox(tmp_path / ("s%d" % i), kind)
        done = box.run(**extra)
        log = done.stdout.splitlines()
        assert all(re.match(r"(\[golden\] |RESULT |=== |ERROR)", l) for l in log), log
        assert _shown_by_summary(tmp_path / ("s%d" % i), done.stdout) == log, extra
        for leak in ("SYNTHETIC", "p10000032", "s50414267", "GENERATED", "Loaded checkpoint", "/sc/home"):
            assert leak not in done.stdout, leak


@pytest.mark.parametrize("kind, drifting", [("cpu", ["engine_cached"]), ("gpu", ["engine_uncached", "engine_cached"])])
@pytest.mark.parametrize("n", [20, 30])
def test_result_lines_stay_under_the_300_characters_summary_keeps(tmp_path, kind, drifting, n):
    """Worst case: every row of the arm(s) that matter differs, so every list of differing rows is as long as it gets.
    At n=30 the lists are capped at 20 entries while the counts still say 30."""
    box = GoldenBox(tmp_path, kind)
    every_row = {arm: list(range(n)) for arm in drifting}
    done = box.run(N=str(n), STUB_DRIFT=json.dumps(every_row))
    found = [l for l in done.stdout.splitlines() if l.startswith("RESULT ")]
    assert len(found) == 2 and max(len(l) for l in found) <= 300, [len(l) for l in found]
    counts, rows = results(done.stdout)
    assert counts["n"] == n and all(len(v) == 20 if n == 30 else len(v) == n for v in rows.values()), rows
    assert max(v for k, v in counts.items() if k.endswith("_differ")) == n
    full = json.loads((box.out / "golden.json").read_text())
    assert all(len(v) == n for k, v in full.items() if k.startswith("rows_")), "golden.json keeps the full lists"


def test_the_driver_lines_are_echoed_tagged_with_their_arm_and_the_load_facts_come_along(box):
    log = box.run().stdout.splitlines()
    arm = "engine_cached"
    assert "[golden] %s: device=cpu name=cpu torch=2.0 threads=8" % arm in log
    assert "[golden] %s: row 19 encode_ms=1.0 prefill_ms=2.0 per_token_ms=3.0 generate_ms=4.0" % arm in log
    assert "[golden] %s: Missing keys: 0, Unexpected: 0" % arm in log and "[golden] %s: prefix_k = 32" % arm in log
    assert not [l for l in log if "report text" in l.lower()]


def test_the_cpu_wrapper_also_shows_the_script_arms_load_facts(cpu):
    log = cpu.run().stdout.splitlines()
    assert "[golden] script_cached: Missing keys: 0, Unexpected: 0" in log
    assert "[golden] script_cached: prefix_k = 32" in log


# ── inputs named by basename ─────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("attr, shown", [("ckpt", "last.ckpt"), ("published", "hyps.txt"), ("parquet", "test.parquet")])
def test_a_missing_input_is_named_by_basename_only_and_no_arm_runs(box, attr, shown):
    getattr(box, attr).unlink()
    done = box.run()
    assert done.returncode == 1
    assert done.stdout.splitlines() == ["ERROR: not found: %s" % shown]
    assert box.calls() == []


def test_output_lands_in_a_new_results_directory_named_for_the_job(box):
    done = box.run()
    assert done.returncode == 0
    assert (box.out / "golden.json").exists()
    assert (box.out / "engine_cached.log").exists(), "raw driver output stays in a file under the output directory"
    assert "GENERATED" in (box.out / ("script_cached.log" if box.kind == "cpu" else "engine_cached.log")).read_text()
