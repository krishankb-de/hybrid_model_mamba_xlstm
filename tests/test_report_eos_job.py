"""P9-G3 (CHAT_UI_PLAN.md): the EOS training job's result summary (scripts/report_eos_result.py) and its wrapper
(scripts/train_report_eos_h100.sh), rehearsed end to end in a temp tree.

Nothing here touches the cluster or trains anything. The wrapper runs for real, under the oldest bash this repo supports
(the Mac's 3.2), in a throwaway tree that stands in for the cluster repo: the real preflight and result scripts, a copy of
the real configs/, a fake scripts/train_contrastive.py, a fake thesis checkout behind the outputs symlink, and a `python`
stub that plays the GPU probe and the trainer. What is asserted is what the job prints and what it writes.

Synthetic data only (R7): the stub trainer's output carries lines that look like MIMIC text and paths, and the job log,
which is all an agent may read, must carry none of them.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import pytest

from scripts import report_eos_preflight as pre
from scripts import report_eos_result as res
from tests import report_eos_recipe as recipe
from tests.report_eos_recipe import stub_tree, write_events
from tests.test_chat_remote import STAMP_RE, Sandbox

pytest.importorskip("torch.utils.tensorboard")

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULT_SCRIPT = REPO_ROOT / "scripts" / "report_eos_result.py"
BASH = "/bin/bash" if os.path.exists("/bin/bash") else "bash"     # the Mac's /bin/bash is 3.2: the oldest shell to support
VAL = "val/lm_loss"


# ── the result summary ────────────────────────────────────────────────────────

def test_the_last_val_loss_is_the_newest_versions_last_logged_step_and_versions_sort_numerically(tmp_path):
    write_events(tmp_path, {
        "version_2": [(VAL, 249, 3.0), (VAL, 499, 2.9)],
        "version_9": [(VAL, 249, 9.0), (VAL, 749, 9.9)],                 # a decoy: 9 < 10, "version_9" > "version_10" as text
        "version_10": [(VAL, 249, 1.4), (VAL, 749, 1.1), ("train/lm_loss_step", 699, 5.0)],
    })
    summary = res.run_summary(tmp_path)
    assert summary == {"steps": 750, "val_loss": pytest.approx(1.1)}


def test_steps_is_the_largest_step_stamp_of_any_scalar_plus_one_because_stamps_count_from_zero(tmp_path):
    write_events(tmp_path, {"version_0": [(VAL, 11999, 1.03), ("train/lm_loss_step", 12009, 0.9), ("train/lr", 11998, 1e-6)]})
    assert res.run_summary(tmp_path)["steps"] == 12010


def test_a_newest_version_that_never_validated_has_steps_but_no_val_loss(tmp_path):
    write_events(tmp_path, {"version_0": [(VAL, 249, 2.0)], "version_1": [("train/lm_loss_step", 99, 5.0)]})
    assert res.run_summary(tmp_path) == {"steps": 100, "val_loss": None}


def test_no_event_files_gives_no_summary(tmp_path):
    assert res.run_summary(tmp_path / "missing") is None
    (tmp_path / "tensorboard" / "version_0").mkdir(parents=True)
    assert res.run_summary(tmp_path) is None


def test_the_val_loss_is_rounded_to_four_decimals_and_steps_is_an_int(tmp_path):
    write_events(tmp_path / "logs", {"version_0": [(VAL, 249, 1.0345678901)]})
    result, _ = res.build_result(tmp_path, tmp_path / "pub" / "run_metadata.json", 7)
    assert result["val_loss"] == 1.0346 and result["steps"] == 250 and isinstance(result["steps"], int)


def fit_report_generation(root: Path, max_steps: int, val_every: int, log_every: int, steps_per_epoch: int):
    """The real ReportGenerationLightningModule, tiny and on CPU, under the trainer's own Trainer arguments, callbacks and log
    layout (scripts/train_report_generation.py:300-350). Returns (trainer.global_step, the run's log dir)."""
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.loggers import TensorBoardLogger
    import torch
    from torch.utils.data import DataLoader, Dataset
    from hybrid_xmamba.models.configuration_hybrid import HybridConfig
    from hybrid_xmamba.training.lightning_module import ReportGenerationLightningModule

    class Rows(Dataset):
        def __init__(self, n):
            self.n = n

        def __len__(self):
            return self.n

        def __getitem__(self, i):
            g = torch.Generator().manual_seed(i)
            return {"input_ids": torch.randint(0, 100, (8,), generator=g), "attention_mask": torch.ones(8, dtype=torch.long),
                    "patch_grid": torch.randn(5, 16, generator=g)}

    config = HybridConfig(vocab_size=100, dim=32, num_layers=1, layer_pattern=["attention"], num_heads=2,
                          max_position_embeddings=64, use_fast_path=False, use_tfla=False)
    module = ReportGenerationLightningModule(
        decoder_config=config, image_patch_dim=16, prefix_k=2, decoder_lr=1e-3, head_lr=1e-3, weight_decay=0.01,
        warmup_steps=2, max_steps=max_steps, gradient_clip_val=0.5)
    trainer = pl.Trainer(
        accelerator="cpu", devices=1, max_steps=max_steps, val_check_interval=val_every, check_val_every_n_epoch=None,
        log_every_n_steps=log_every, accumulate_grad_batches=1,
        callbacks=[ModelCheckpoint(dirpath=str(root / "checkpoints"), monitor="val/lm_loss", mode="min", save_top_k=0,
                                   save_last=True, filename="report_gen-{step:06d}-{val_lm_loss_ckpt:.4f}"),
                   LearningRateMonitor(logging_interval="step")],
        logger=[TensorBoardLogger(save_dir=str(root / "logs"), name="tensorboard")], enable_checkpointing=True,
        enable_progress_bar=False, enable_model_summary=False, num_sanity_val_steps=2, default_root_dir=str(root))
    trainer.fit(module, train_dataloaders=DataLoader(Rows(steps_per_epoch * 4), batch_size=4, shuffle=True, drop_last=True),
                val_dataloaders=DataLoader(Rows(8), batch_size=4))
    return trainer.global_step, root / "logs"


# Lightning's own advisories about a tiny CPU fit (an unused MPS device, a pytree deprecation, dataloader workers).
@pytest.mark.filterwarnings("ignore:GPU available but not used",
                            "ignore:.isinstance.treespec, LeafSpec.. is deprecated",
                            "ignore:The '.*dataloader' does not have many workers")
@pytest.mark.parametrize("max_steps, val_every, log_every, steps_per_epoch", [
    (500, 100, 20, 61),    # the real run's shape, scaled: validation and logging divide max_steps, the last epoch is cut mid-way
    (130, 50, 20, 61),     # neither the last validation nor the last logged step reaches max_steps: the epoch-end metrics do
])
def test_steps_is_the_number_of_optimizer_steps_a_finished_trainer_ran(tmp_path, max_steps, val_every, log_every,
                                                                      steps_per_epoch):
    """The DONE check compares steps with MAX_STEPS, so it must mean what trainer.global_step means. Lightning stamps a
    step with the number completed BEFORE it (validation runs before the step is counted): a finished 12000-step run ends
    at stamp 11999, and the largest stamp alone would never equal MAX_STEPS. Measured on the installed Lightning."""
    pytest.importorskip("pytorch_lightning")
    global_step, log_dir = fit_report_generation(tmp_path, max_steps, val_every, log_every, steps_per_epoch)
    assert global_step == max_steps
    summary = res.run_summary(log_dir)
    assert summary["steps"] == max_steps, "the event stamps no longer count from zero: the DONE check would refuse good runs"
    assert summary["val_loss"] is not None


def make_run(root: Path, own: Optional[List[Tuple[str, int, float]]], published: Optional[List[Tuple[str, int, float]]],
             ckpt: bool = True) -> Tuple[Path, Path]:
    out_dir, pub_dir = root / "run", root / "pub"
    out_dir.mkdir(), pub_dir.mkdir()
    (pub_dir / "run_metadata.json").write_text("{}")
    if ckpt:
        (out_dir / "checkpoints").mkdir()
        (out_dir / "checkpoints" / "last.ckpt").write_bytes(b"")
    if own is not None:
        write_events(out_dir / "logs", {"version_0": own})
    if published is not None:
        write_events(pub_dir / "logs", {"version_0": published})
    return out_dir, pub_dir / "run_metadata.json"


def test_build_result_has_the_fields_the_ruling_asks_for(tmp_path):
    out_dir, meta = make_run(tmp_path, [(VAL, 11999, 1.0345)], [(VAL, 249, 1.9), (VAL, 11999, 1.0182)])
    result, notes = res.build_result(out_dir, meta, 4680)
    assert result == {"train": "done", "steps": 12000, "wall_s": 4680, "val_loss": 1.0345,
                      "published_val_loss": 1.0182, "ckpt_exists": True}
    assert notes == []


def test_build_result_says_so_when_the_published_run_left_no_events(tmp_path):
    out_dir, meta = make_run(tmp_path, [(VAL, 11999, 1.0345)], None)
    result, notes = res.build_result(out_dir, meta, 5)
    assert result["published_val_loss"] is None and result["val_loss"] == 1.0345
    assert len(notes) == 1 and "published" in notes[0]


def test_build_result_says_so_when_this_run_logged_no_validation(tmp_path):
    out_dir, meta = make_run(tmp_path, [("train/lm_loss_step", 39, 5.0)], [(VAL, 11999, 1.0182)], ckpt=False)
    result, notes = res.build_result(out_dir, meta, 5)
    assert result["val_loss"] is None and result["steps"] == 40 and result["ckpt_exists"] is False
    assert any("this run" in note for note in notes)


def test_without_tensorboard_the_result_names_the_checkpoint_and_nulls_the_losses_and_says_so(tmp_path, monkeypatch):
    out_dir, meta = make_run(tmp_path, [(VAL, 11999, 1.0345)], [(VAL, 11999, 1.0182)])
    monkeypatch.setitem(sys.modules, "tensorboard.backend.event_processing.event_accumulator", None)
    monkeypatch.setitem(sys.modules, "tbparse", None)
    result, notes = res.build_result(out_dir, meta, 4680)
    assert result == {"train": "done", "steps": None, "wall_s": 4680, "val_loss": None, "published_val_loss": None,
                      "ckpt_exists": True, "ckpt": "last.ckpt"}
    assert len(notes) == 1 and "tensorboard" in notes[0]


def test_tbparse_reads_the_events_when_tensorboard_cannot_be_imported(tmp_path, monkeypatch):
    """tbparse is not installed here, so this runs against a stand-in with its documented shape (SummaryReader(path).scalars,
    a long-format frame of step, tag, value) and checks the adapter, not tbparse itself."""
    import types
    import pandas as pd
    out_dir, meta = make_run(tmp_path, [(VAL, 11999, 1.0)], [(VAL, 11999, 1.0)])
    asked = []

    class StandIn:
        def __init__(self, path):
            asked.append(Path(path))
            self.scalars = pd.DataFrame({"step": [249, 11999, 11999],
                                         "tag": [VAL, VAL, "train/lm_loss_step"], "value": [2.0, 1.0345, 0.9]})

    monkeypatch.setitem(sys.modules, "tensorboard.backend.event_processing.event_accumulator", None)
    monkeypatch.setitem(sys.modules, "tbparse", types.SimpleNamespace(SummaryReader=StandIn))
    result, notes = res.build_result(out_dir, meta, 4680)
    assert result == {"train": "done", "steps": 12000, "wall_s": 4680, "val_loss": 1.0345, "published_val_loss": 1.0345,
                      "ckpt_exists": True}
    assert notes == [] and {p.parent.name for p in asked} == {"tensorboard"}, "it was handed the version directories"


def run_result_cli(out_dir: Path, meta: Path, wall_s: str = "4680") -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    return subprocess.run([sys.executable, str(RESULT_SCRIPT), "--out-dir", str(out_dir), "--published", str(meta),
                           "--wall-s", wall_s], cwd=str(REPO_ROOT), env=env, stdin=subprocess.DEVNULL,
                          capture_output=True, text=True, timeout=120)


def test_cli_prints_one_short_result_line_of_numbers_and_flags_only(tmp_path):
    out_dir, meta = make_run(tmp_path, [(VAL, 11999, 1.0345)], [(VAL, 11999, 1.0182)])
    done = run_result_cli(out_dir, meta)
    assert done.returncode == 0, done.stdout + done.stderr
    lines = done.stdout.splitlines()
    assert len(lines) == 1 and lines[0].startswith("RESULT {") and len(lines[0]) < 300
    assert json.loads(lines[0][len("RESULT "):]) == {
        "train": "done", "steps": 12000, "wall_s": 4680, "val_loss": 1.0345, "published_val_loss": 1.0182,
        "ckpt_exists": True}
    assert "/" not in lines[0], "numbers only: no path in the line"


def test_cli_reports_a_missing_checkpoint_instead_of_failing(tmp_path):
    out_dir, meta = make_run(tmp_path, [(VAL, 11999, 1.0345)], [(VAL, 11999, 1.0182)], ckpt=False)
    done = run_result_cli(out_dir, meta)
    assert done.returncode == 0, done.stdout + done.stderr
    assert json.loads(done.stdout.splitlines()[0][len("RESULT "):])["ckpt_exists"] is False


def test_cli_notes_are_equals_lines_without_paths(tmp_path):
    out_dir, meta = make_run(tmp_path, [("train/lm_loss_step", 39, 5.0)], None)
    done = run_result_cli(out_dir, meta)
    lines = done.stdout.splitlines()
    assert lines[0].startswith("RESULT ")
    notes = lines[1:]
    assert notes and all(line.startswith("=== ") and line.endswith(" ===") and "/" not in line for line in notes), notes


# ── the wrapper, rehearsed in a temp tree ─────────────────────────────────────

PYTHON_STUB = """#!/bin/bash
# Stands in for the venv's python. It records every call, plays the GPU probe and the trainer, and runs everything
# else (the real preflight and result scripts, the wrapper's own path helper) with the test interpreter.
{ echo "@@"; for a in "$@"; do printf '%s\\n' "$a"; done; } >> "$STUB_DIR/python.calls"
case "$*" in
  *torch.cuda.device_count*) echo "${FAKE_GPUS:-4} NVIDIA H100 80GB HBM3"; exit 0;;
  scripts/train_report_generation.py*) ;;
  *) exec "$REAL_PYTHON" "$@";;
esac
echo "Findings: FAKE REPORT TEXT study_id=12345678 /sc/home/someone/images/p10/img.jpg"
echo "Traceback (most recent call last): FAKE MIMIC TEXT in a message" >&2
out=""
for a in "$@"; do case "$a" in output_dir=*) out="${a#output_dir=}";; esac; done
mode="$(cat "$STUB_DIR/train.mode")"
tb="$(cat "$STUB_DIR/train.tb")"
# A file this stub writes is stamped 5 s ahead: newer than the wrapper's start marker whatever the timestamp resolution
# (bash 3.2 compares whole seconds, and the stub is quicker than that).
fresh() { "$REAL_PYTHON" -c 'import os, sys, time; t = time.time() + 5; [os.utime(p, (t, t)) for p in sys.argv[1:]]' "$@"; }
[ "$mode" = fail ] && exit 3
mkdir -p "$out/checkpoints" "$out/logs"
[ "$mode" = stale ] && exit 0                       # exits 0 and writes nothing: a last.ckpt that exists is an earlier attempt's
[ "$mode" = nockpt ] || { : > "$out/checkpoints/last.ckpt"; fresh "$out/checkpoints/last.ckpt"; }
[ -d "$STUB_DIR/tb_$tb" ] && cp -R "$STUB_DIR/tb_$tb/." "$out/logs/"
if [ "$mode" = interrupt ]; then : > "$out/checkpoints/interrupt.ckpt"; fresh "$out/checkpoints/interrupt.ckpt"; fi
exit 0
"""

# What TensorBoardLogger leaves for a 12000-step run: Lightning stamps steps from zero, so the last one is 11999.
TB_FIXTURES = {
    "ok": [(VAL, 249, 2.0), (VAL, 11999, 1.0345), ("train/lm_loss_step", 11999, 0.95), ("train/lm_loss_epoch", 11999, 0.94)],
    "short": [(VAL, 249, 2.0), (VAL, 11749, 1.2), ("train/lm_loss_step", 11749, 0.95), ("train/lm_loss_epoch", 11749, 0.94)],
    "over": [(VAL, 249, 2.0), (VAL, 12000, 1.0345), ("train/lm_loss_step", 12000, 0.95)],
}

# The shape scripts/chat_remote.sh sync writes to .sync_stamp: "<UTC time> <40-hex commit> <clean|dirty>".
STAMP = "2026-10-09T20:00:00Z 3f2a9c41d7e86b05a1c4e9d3b7f60285ac9e1d47 clean"

LINE_OK = re.compile(r"^(=== |RESULT |ERROR)")


def age(path: Path, seconds: float) -> None:
    """Back-date a file's modification time."""
    then = time.time() - seconds
    os.utime(str(path), (then, then))


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
    """The EOS wrapper run for real in a temp tree standing in for the cluster: CLUSTER_REPO (repo/), the thesis checkout
    (main/) behind repo/outputs, CHAT_HOME (chat/), and a stub python."""

    def __init__(self, root: Path, with_flag: bool = True, stamp: Optional[str] = STAMP):
        self.root = root
        self.repo, self.main, self.chat = root / "repo", root / "main", root / "chat"
        self.stubs, self.bin = root / "stubs", root / "bin"
        stub_tree(self.repo, with_flag=with_flag)
        for name in ("report_eos_result.py", "train_report_eos_h100.sh"):
            shutil.copy(str(REPO_ROOT / "scripts" / name), str(self.repo / "scripts" / name))
        shutil.copytree(str(REPO_ROOT / "configs"), str(self.repo / "configs"))
        (self.repo / ".venv" / "bin").mkdir(parents=True)
        (self.repo / ".venv" / "bin" / "activate").write_text("")
        (self.repo / "logs").mkdir()
        if stamp is not None:
            (self.repo / ".sync_stamp").write_text(stamp + "\n")
        for rel in ("h100_stage0_150m_m3", "h100_kd_150m_v2_full_data_lr3e6"):
            ckpt = self.main / "outputs" / rel / "checkpoints" / "last.ckpt"
            ckpt.parent.mkdir(parents=True)
            ckpt.write_bytes(b"")
        (self.repo / "outputs").symlink_to(self.main / "outputs", target_is_directory=True)
        self.chat.mkdir()
        for directory in (self.stubs, self.bin):
            directory.mkdir()
        python = self.bin / "python"
        python.write_text(PYTHON_STUB)
        python.chmod(0o755)
        self.published = self.main / "outputs" / recipe.PUBLISHED_EXPERIMENT
        self.write_published()
        self.set_train_mode("ok")
        self.set_train_tb("ok")

    @property
    def out_dir(self) -> Path:
        return self.chat / "models" / recipe.NEW_EXPERIMENT

    def write_published(self, mutate=None) -> None:
        """The published run's metadata (its override list composed by Hydra) and its TensorBoard events."""
        resolved = pre.compose_job_config(pre.split_overrides(recipe.published_job_overrides())[0], REPO_ROOT / "configs")
        if mutate:
            mutate(resolved)
        self.published.mkdir(parents=True, exist_ok=True)
        (self.published / "run_metadata.json").write_text(json.dumps({"resolved_config": resolved}))
        if not (self.published / "logs").exists():
            write_events(self.published / "logs", {"version_0": [(VAL, 249, 1.9), (VAL, 11999, 1.0182)]})

    def set_train_mode(self, mode: str) -> None:
        """ok | fail | interrupt | nockpt | stale: what the stub trainer does to the files, and how it exits."""
        (self.stubs / "train.mode").write_text(mode + "\n")

    def set_train_tb(self, name: str) -> None:
        """The event files the stub trainer leaves: a TB_FIXTURES key, or "none" for no events at all."""
        (self.stubs / "train.tb").write_text(name + "\n")
        if name in TB_FIXTURES and not (self.stubs / ("tb_" + name)).exists():
            write_events(self.stubs / ("tb_" + name), {"version_0": TB_FIXTURES[name]})

    def run(self, **extra_env: str) -> subprocess.CompletedProcess:
        env = {"PATH": "{}:/usr/bin:/bin".format(self.bin), "HOME": str(self.root / "home"), "USER": recipe.CLUSTER_USER,
               "SLURM_SUBMIT_DIR": str(self.repo), "SLURM_JOB_ID": "1234567", "CHAT_HOME": str(self.chat),
               "SCRATCH_ROOT": str(self.root / "scratch"), "STUB_DIR": str(self.stubs), "REAL_PYTHON": sys.executable}
        env.update(extra_env)
        # stdout and stderr share one pipe, like the single SLURM log: whatever bash itself complains about counts too.
        return subprocess.run([BASH, str(self.repo / "scripts" / "train_report_eos_h100.sh")], cwd=str(self.root), env=env,
                              stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                              timeout=240)

    def calls(self) -> List[List[str]]:
        log = self.stubs / "python.calls"
        if not log.exists():
            return []
        return [rec.splitlines() for rec in log.read_text().split("@@\n") if rec.strip()]

    def calls_of(self, script: str) -> List[List[str]]:
        return [c for c in self.calls() if c and c[0] == "scripts/" + script]


def job_lines(done: subprocess.CompletedProcess) -> List[str]:
    return done.stdout.splitlines()


def results(lines: List[str]) -> List[dict]:
    return [json.loads(l[len("RESULT "):]) for l in lines if l.startswith("RESULT ")]


def test_a_clean_run_preflights_trains_summarises_and_marks_the_run_done(tmp_path):
    box = JobBox(tmp_path)
    main_before = snapshot(box.main)
    done = box.run()
    lines = job_lines(done)
    assert done.returncode == 0, done.stdout
    # R7: only wrapper-authored lines, and nothing the trainer printed.
    assert [l for l in lines if not LINE_OK.match(l)] == [], "a line that is not ===, RESULT or ERROR"
    assert not [l for l in lines if "FAKE" in l or "12345678" in l or "/sc/home" in l or "Traceback" in l]
    assert "FAKE REPORT TEXT" in (box.out_dir / "train.log").read_text(), "the trainer's output went to train.log"
    assert all(len(l) <= 300 for l in lines if l.startswith(("RESULT ", "ERROR")))

    assert lines[0] == "=== sync 3f2a9c41d7e86b05a1c4e9d3b7f60285ac9e1d47 clean ===", "the provenance comes first"
    found = results(lines)
    assert found[0] == {"preflight": "code", "eos_flag": True, "module_in_cwd": True}
    assert found[1]["preflight"] == "recipe" and found[1]["ok"] is True and found[1]["added"] == ["dataset.report_eos_target"]
    final = found[2]
    assert set(final) == {"train", "steps", "wall_s", "val_loss", "published_val_loss", "ckpt_exists"}
    assert (final["train"], final["steps"], final["val_loss"], final["published_val_loss"], final["ckpt_exists"]) == (
        "done", 12000, 1.0345, 1.0182, True)
    assert isinstance(final["wall_s"], int) and 0 <= final["wall_s"] < 600

    assert (box.out_dir / "DONE").is_file() and (box.out_dir / "checkpoints" / "last.ckpt").is_file()
    assert lines[-1].startswith("=== END ")
    assert snapshot(box.main) == main_before, "R8: the thesis checkout was only read"
    assert [p.name for p in box.chat.iterdir()] == ["models"], "nothing but the new run directory under CHAT_HOME"


def test_one_override_list_feeds_the_preflight_and_the_trainer_and_is_the_recipes(tmp_path):
    box = JobBox(tmp_path)
    assert box.run().returncode == 0
    (pre_call,), (train_call,) = box.calls_of("report_eos_preflight.py"), box.calls_of("train_report_generation.py")
    expected = recipe.new_job_overrides(str(box.out_dir), chat_home=str(box.chat))
    assert pre_call[pre_call.index("--") + 1:] == expected
    assert train_call == ["scripts/train_report_generation.py", "--config-name", "config"] + expected
    order = [c[0] for c in box.calls() if c[0].startswith("scripts/")]
    assert order == ["scripts/report_eos_preflight.py", "scripts/train_report_generation.py", "scripts/report_eos_result.py"]
    assert "output_dir=" + str(box.out_dir) in expected and "hydra.run.dir={}/hydra".format(box.out_dir) in expected
    assert "+dataset.report_eos_target=true" in expected


def test_a_finished_run_is_never_overwritten(tmp_path):
    box = JobBox(tmp_path)
    box.out_dir.mkdir(parents=True)
    (box.out_dir / "DONE").write_text("first run\n")
    before = snapshot(box.chat)
    done = box.run()
    lines = job_lines(done)
    assert done.returncode == 1
    assert [l for l in lines if l.startswith("ERROR")] and recipe.NEW_EXPERIMENT in "\n".join(lines)
    assert box.calls_of("report_eos_preflight.py") == [] and box.calls_of("train_report_generation.py") == []
    assert (box.out_dir / "DONE").read_text() == "first run\n" and snapshot(box.chat) == before


@pytest.mark.parametrize("where", ["inside_outputs", "under_main", "outputs_elsewhere", "symlink_into_main",
                                   "symlink_named_outputs", "symlink_to_outputs_elsewhere"])
def test_an_out_dir_in_an_outputs_directory_or_under_the_thesis_checkout_is_refused(tmp_path, where):
    box = JobBox(tmp_path)
    if where == "inside_outputs":
        home = box.main / "outputs" / "chat"
    elif where == "under_main":
        home = box.main / "chat"
    elif where == "outputs_elsewhere":
        home = tmp_path / "elsewhere" / "outputs" / "chat"
    elif where == "symlink_into_main":
        (box.main / "chat").mkdir()
        home = tmp_path / "chat_link"
        home.symlink_to(box.main / "chat", target_is_directory=True)
    elif where == "symlink_named_outputs":
        # The path NAMES an outputs directory but resolves somewhere that does not (a link called outputs): only the
        # check on the path as given refuses it, the check on the resolved path sees nothing wrong.
        (tmp_path / "real_place" / "chat").mkdir(parents=True)
        (tmp_path / "outputs").symlink_to(tmp_path / "real_place", target_is_directory=True)
        home = tmp_path / "outputs" / "chat"
    else:
        # The reverse: nothing in the path says outputs and it is outside the thesis checkout, but it resolves into an
        # outputs directory. Only the check on the resolved path refuses it.
        (tmp_path / "other" / "outputs" / "chat").mkdir(parents=True)
        home = tmp_path / "plain_name"
        home.symlink_to(tmp_path / "other" / "outputs" / "chat", target_is_directory=True)
    if where in ("inside_outputs", "under_main", "outputs_elsewhere"):
        home.mkdir(parents=True, exist_ok=True)
    before_main = snapshot(box.main)
    done = box.run(CHAT_HOME=str(home))
    assert done.returncode == 1, done.stdout
    assert [l for l in job_lines(done) if l.startswith("ERROR")]
    assert box.calls_of("report_eos_preflight.py") == [] and box.calls_of("train_report_generation.py") == []
    assert snapshot(box.main) == before_main, "R8: nothing was created in the thesis checkout"


def test_a_missing_chat_home_or_input_checkpoint_stops_the_job_before_anything_runs(tmp_path):
    box = JobBox(tmp_path)
    shutil.rmtree(str(box.chat))
    assert box.run().returncode == 1 and not (tmp_path / "chat").exists(), "CHAT_HOME is never created here"
    box.chat.mkdir()
    for rel in ("h100_stage0_150m_m3", "h100_kd_150m_v2_full_data_lr3e6"):
        ckpt = box.main / "outputs" / rel / "checkpoints" / "last.ckpt"
        ckpt.rename(str(ckpt) + ".away")
        done = box.run()
        assert done.returncode == 1 and [l for l in job_lines(done) if l.startswith("ERROR")], rel
        assert box.calls_of("report_eos_preflight.py") == []
        Path(str(ckpt) + ".away").rename(ckpt)


def test_too_few_gpus_stop_the_job_before_the_preflight(tmp_path):
    box = JobBox(tmp_path)
    done = box.run(FAKE_GPUS="2")
    assert done.returncode == 1
    assert any(l.startswith("ERROR") and "2" in l and "4" in l for l in job_lines(done)), job_lines(done)
    assert box.calls_of("report_eos_preflight.py") == []
    assert not box.out_dir.exists()


def test_a_preflight_failure_on_the_recipe_stops_the_job_before_the_trainer(tmp_path):
    box = JobBox(tmp_path)
    box.write_published(mutate=lambda cfg: cfg["model"].__setitem__("decoder_lr", 2e-05))
    done = box.run()
    lines = job_lines(done)
    assert done.returncode == 1, done.stdout
    recipe_result = [r for r in results(lines) if r.get("preflight") == "recipe"][0]
    assert recipe_result["ok"] is False and recipe_result["changed"][0] == "model.decoder_lr"
    assert "ERROR preflight exit=1, nothing was trained" in lines
    assert box.calls_of("train_report_generation.py") == [] and not (box.out_dir / "DONE").exists()
    assert [l for l in lines if not LINE_OK.match(l)] == []


def test_a_dataset_module_without_the_flag_stops_the_job_before_the_trainer(tmp_path):
    box = JobBox(tmp_path, with_flag=False)
    done = box.run()
    lines = job_lines(done)
    assert done.returncode == 1, done.stdout
    assert results(lines)[0] == {"preflight": "code", "eos_flag": False, "module_in_cwd": True}
    assert box.calls_of("train_report_generation.py") == [] and not (box.out_dir / "DONE").exists()


def test_a_failed_trainer_prints_only_its_exit_code_and_leaves_no_done_marker(tmp_path):
    box = JobBox(tmp_path)
    box.set_train_mode("fail")
    done = box.run()
    lines = job_lines(done)
    assert done.returncode == 3, done.stdout
    assert "ERROR train exit=3" in lines
    assert [l for l in lines if not LINE_OK.match(l)] == [] and not [l for l in lines if "FAKE" in l]
    assert "FAKE REPORT TEXT" in (box.out_dir / "train.log").read_text()
    assert not (box.out_dir / "DONE").exists()
    assert box.calls_of("report_eos_result.py") == []


def test_a_crashing_result_script_leaves_its_text_in_result_err_and_none_of_it_in_the_job_log(tmp_path):
    """R7. The result script's stderr goes to OUT_DIR/result.err, and its stdout is cut down to the three line shapes: a crash
    whose traceback and partial output carry ids and paths must reach neither. The result is a report, not a gate: with the
    events unreadable, DONE rests on the fresh checkpoint alone."""
    box = JobBox(tmp_path)
    (box.repo / "scripts" / "report_eos_result.py").write_text(
        'print("partial study_id=12345678 FAKE REPORT TEXT")\n'
        'raise RuntimeError("cannot read study_id=12345678 /sc/home/someone/images/p10/img.jpg FAKE REPORT TEXT")\n')
    done = box.run()
    lines = job_lines(done)
    assert done.returncode == 0, done.stdout
    assert "ERROR result exit=1" in lines
    assert [l for l in lines if not LINE_OK.match(l)] == []
    assert not [l for l in lines if "Traceback" in l or "12345678" in l or "FAKE" in l or "/sc/home" in l], lines
    err = (box.out_dir / "result.err").read_text()
    assert "Traceback" in err and "12345678" in err and "FAKE REPORT TEXT" in err, "the text landed in the file"
    assert (box.out_dir / "DONE").is_file()


def test_a_requeued_trainer_that_exits_0_over_a_stale_checkpoint_gets_no_done(tmp_path):
    """The earlier attempt left last.ckpt; this one exits 0 without writing anything, so that file is not this run's."""
    box = JobBox(tmp_path)
    ckpt = box.out_dir / "checkpoints" / "last.ckpt"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"earlier attempt")
    age(ckpt, 3600)
    box.set_train_mode("stale")
    box.set_train_tb("none")
    done = box.run()
    lines = job_lines(done)
    assert done.returncode == 1, done.stdout
    assert "ERROR done refused: last.ckpt was not written during this attempt" in lines
    assert results(lines)[-1]["ckpt_exists"] is True, "the result line still reports what it found"
    assert not (box.out_dir / "DONE").exists()


def test_a_run_that_stopped_short_of_max_steps_gets_no_done(tmp_path):
    box = JobBox(tmp_path)
    box.set_train_tb("short")
    done = box.run()
    lines = job_lines(done)
    assert done.returncode == 1, done.stdout
    assert "ERROR done refused: steps=11750 expected=12000" in lines
    assert results(lines)[-1]["steps"] == 11750
    assert (box.out_dir / "checkpoints" / "last.ckpt").is_file() and not (box.out_dir / "DONE").exists()
    assert [l for l in lines if not LINE_OK.match(l)] == []


def test_steps_must_equal_max_steps_exactly_so_one_step_over_is_refused_too(tmp_path):
    box = JobBox(tmp_path)
    box.set_train_tb("over")                       # stamps up to 12000: 12001 steps
    done = box.run()
    assert done.returncode == 1 and "ERROR done refused: steps=12001 expected=12000" in job_lines(done)
    assert not (box.out_dir / "DONE").exists()


def test_unreadable_events_do_not_stop_done_but_the_checkpoint_checks_still_apply(tmp_path):
    box = JobBox(tmp_path)
    box.set_train_tb("none")                       # the trainer left no events: steps is null, only the checkpoint is checked
    done = box.run()
    assert done.returncode == 0, done.stdout
    assert results(job_lines(done))[-1]["steps"] is None
    assert (box.out_dir / "DONE").is_file()


def test_a_trainer_that_exits_0_without_a_checkpoint_is_not_marked_done(tmp_path):
    box = JobBox(tmp_path)
    box.set_train_mode("nockpt")
    done = box.run()
    lines = job_lines(done)
    assert done.returncode == 1, done.stdout
    assert results(lines)[-1]["ckpt_exists"] is False, "the result line still reports what it found"
    assert "ERROR done refused: last.ckpt is missing" in lines
    assert not (box.out_dir / "DONE").exists()


def test_a_run_the_signal_handler_cut_short_is_not_marked_done(tmp_path):
    """SignalCheckpointCallback saves interrupt.ckpt and raises SystemExit(0), so a preempted trainer exits 0."""
    box = JobBox(tmp_path)
    box.set_train_mode("interrupt")
    done = box.run()
    assert done.returncode == 1, done.stdout
    assert any(l.startswith("ERROR done refused:") and "interrupt.ckpt" in l for l in job_lines(done)), job_lines(done)
    assert not (box.out_dir / "DONE").exists(), "DONE would block the requeue that has to finish the run"


def test_a_requeued_unfinished_run_restarts_and_appends_to_the_trainer_log(tmp_path):
    box = JobBox(tmp_path)
    (box.out_dir / "checkpoints").mkdir(parents=True)
    (box.out_dir / "checkpoints" / "last.ckpt").write_bytes(b"earlier attempt")
    stale = box.out_dir / "checkpoints" / "interrupt.ckpt"
    stale.write_bytes(b"")
    age(stale, 3600)                                               # the earlier attempt's own signal save
    (box.out_dir / "train.log").write_text("earlier attempt line\n")
    done = box.run()
    assert done.returncode == 0, done.stdout
    assert any("earlier attempt" in l and l.startswith("=== ") for l in job_lines(done)), "the restart is announced"
    log = (box.out_dir / "train.log").read_text()
    assert log.startswith("earlier attempt line\n") and "FAKE REPORT TEXT" in log, "appended, not overwritten"
    assert (box.out_dir / "DONE").is_file(), "a stale interrupt.ckpt from an earlier attempt does not count against this one"


def test_a_stray_environment_variable_cannot_change_the_recipe(tmp_path):
    """sbatch exports the submitting shell: a SEED or MAX_STEPS left in it must not reach this job's trainer."""
    box = JobBox(tmp_path)
    done = box.run(SEED="7", MAX_STEPS="10", DECODER_LR="9e-9", PREFIX_K="8", EXPERIMENT="other", OUT_DIR="/tmp/elsewhere",
                   NUM_GPUS="1", MODEL_CONFIG="hybrid_150m_v2_rrg")
    assert done.returncode == 0, done.stdout
    (train_call,) = box.calls_of("train_report_generation.py")
    assert train_call[3:] == recipe.new_job_overrides(str(box.out_dir), chat_home=str(box.chat))


def real_stamp(root: Path, dirty: bool) -> str:
    """The .sync_stamp that `scripts/chat_remote.sh sync` writes, from the real script in a throwaway git tree with ssh and
    rsync stubbed (tests/test_chat_remote.py's Sandbox): the wrapper has to read what the producer writes."""
    box = Sandbox(root)
    if dirty:
        script = box.repo / "scripts" / "chat_remote.sh"
        script.write_text(script.read_text() + "\n# touched\n")
    done = box.run("sync")
    assert done.returncode == 0, done.stdout + done.stderr
    return (box.repo / ".sync_stamp").read_text().strip()


def test_the_default_stamp_has_the_shape_the_sync_command_writes():
    assert STAMP_RE.match(STAMP)


@pytest.mark.parametrize("dirty", [False, True])
def test_the_first_line_of_the_job_log_names_the_commit_and_cleanliness_the_tree_was_synced_with(tmp_path, dirty):
    stamp = real_stamp(tmp_path / "sync", dirty)
    assert STAMP_RE.match(stamp), stamp
    _, sha, flag = stamp.split()
    assert flag == ("dirty" if dirty else "clean")
    box = JobBox(tmp_path / "job", stamp=stamp)
    done = box.run(FAKE_GPUS="0")                  # the job stops at the GPU check: the first line is all this asks about
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
    "2026-10-09T20:00:00Z 3f2a9c41d7e86b05a1c4e9d3b7f60285ac9e1d4 clean",           # 39 hex digits
    "2026-10-09T20:00:00Z 3F2A9C41D7E86B05A1C4E9D3B7F60285AC9E1D47 clean",          # upper case
    "2026-10-09T20:00:00Z 3f2a9c41d7e86b05a1c4e9d3b7f60285ac9e1d47 maybe",          # a flag that is neither
    "2026-10-09T20:00:00Z 3f2a9c41d7e86b05a1c4e9d3b7f60285ac9e1d47 clean FAKE_EXTRA",   # a field too many
])
def test_a_malformed_sync_stamp_is_reported_as_unknown_and_never_echoed(tmp_path, stamp):
    box = JobBox(tmp_path, stamp=stamp)
    done = box.run(FAKE_GPUS="0")
    lines = job_lines(done)
    assert lines[0] == "=== sync unknown ===", lines[:2]
    assert not [l for l in lines if "FAKE_" in l or "whoami" in l or "rm -rf" in l]
