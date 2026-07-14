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
export PYTHONUNBUFFERED=1

cd "${SLURM_SUBMIT_DIR}/hybrid_model_mamba_xlstm" 2>/dev/null || cd "${SLURM_SUBMIT_DIR:-.}"
source "${VENV_ACTIVATE}"

echo "=== checkpoints in ${RUN_DIR}/checkpoints (newest first) ==="
ls -lt --time-style=+%H:%M "${RUN_DIR}/checkpoints/" 2>/dev/null | head -25

echo ""
echo "=== live val/loss + lr trajectory from tfevents ==="
RUN_DIR="${RUN_DIR}" python - <<'PY'
import glob, os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
base = os.path.join(os.environ["RUN_DIR"], "logs/stage0_kd")
vers = sorted(glob.glob(base + "/version_*"))
ver = vers[-1] if vers else None
print("reading:", ver)
if ver:
    ea = EventAccumulator(ver, size_guidance={'scalars': 0}); ea.Reload()
    tags = ea.Tags()['scalars']
    for tag in ("val/loss", "val/perplexity", "train/lr", "train/grad_norm"):
        if tag in tags:
            s = ea.Scalars(tag)
            print(f"\n== {tag}  ({len(s)} pts) ==")
            for e in s[::max(1, len(s)//25)]:
                print(f"  step {e.step:>7}  {e.value:.4f}")
            print(f"  LATEST step {s[-1].step:>7}  {s[-1].value:.4f}")
PY
echo ""
echo "=== done ==="
