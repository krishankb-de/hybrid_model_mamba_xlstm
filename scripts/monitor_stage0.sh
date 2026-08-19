#!/bin/bash
# ============================================================================
# Live progress snapshot for a running (or finished) Stage-0 job — reads the
# TensorBoard event file (written live, unlike the console progress bar which
# only flushes at job end) + lists checkpoints. CPU-only, ~1 min. Safe to run
# repeatedly while training continues.
#
#   sbatch scripts/monitor_stage0.sh                                  # default run dir
#   RUN_DIR=outputs/h100_stage0_150m_v2 sbatch scripts/monitor_stage0.sh
#   cat logs/monitor_stage0_<jobid>.log
# ============================================================================
#SBATCH --partition=aisc-batch
#SBATCH --exclude=ga03   # ARM/Grace node; x86 .venv python -> "cannot execute binary file: Exec format error" (2026-08-19)
#SBATCH --account=aisc
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --job-name=monitor_stage0
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log

set -uo pipefail
VENV_ACTIVATE="${VENV_ACTIVATE:-.venv/bin/activate}"
RUN_DIR="${RUN_DIR:-outputs/h100_stage0_150m_v2}"
# Stage-0 logs under logs/stage0_kd; the contrastive run (train_contrastive.py:829)
# uses TensorBoardLogger(name="tensorboard") -> logs/tensorboard. Override per phase:
#   RUN_DIR=outputs/h100_kd_150m_v2_bs128 LOGGER_SUBDIR=tensorboard sbatch scripts/monitor_stage0.sh
LOGGER_SUBDIR="${LOGGER_SUBDIR:-stage0_kd}"
export PYTHONUNBUFFERED=1

cd "${SLURM_SUBMIT_DIR}/hybrid_model_mamba_xlstm" 2>/dev/null || cd "${SLURM_SUBMIT_DIR:-.}"
source "${VENV_ACTIVATE}"

echo "=== checkpoints in ${RUN_DIR}/checkpoints (newest first) ==="
ls -lt --time-style=+%H:%M "${RUN_DIR}/checkpoints/" 2>/dev/null | head -25

echo ""
echo "=== live val/loss + lr trajectory from tfevents ==="
RUN_DIR="${RUN_DIR}" LOGGER_SUBDIR="${LOGGER_SUBDIR}" python - <<'PY'
import glob, os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
base = os.path.join(os.environ["RUN_DIR"], "logs", os.environ["LOGGER_SUBDIR"])
vers = sorted(glob.glob(base + "/version_*"))
ver = vers[-1] if vers else None
print("reading:", ver)
if ver:
    ea = EventAccumulator(ver, size_guidance={'scalars': 0}); ea.Reload()
    tags = ea.Tags()['scalars']
    print("TAGS:", tags)
    # Broad filter so this works for Stage-0 (loss/ppl/lr/grad_norm) AND the
    # contrastive phase (clip_loss / cos_text_teacher / retrieval R@k).
    keys = ('val/', 'retrieval', 'cos_text', 'clip_loss', 'perplex', 'lr', 'grad_norm')
    for tag in tags:
        if any(k in tag.lower() for k in keys):
            s = ea.Scalars(tag)
            if not s:
                continue
            print(f"\n== {tag}  ({len(s)} pts) ==")
            for e in s[::max(1, len(s)//15)]:
                print(f"  step {e.step:>7}  {e.value:.4f}")
            print(f"  LATEST step {s[-1].step:>7}  {s[-1].value:.4f}")
PY
echo ""
echo "=== done ==="
