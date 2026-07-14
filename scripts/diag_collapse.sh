#!/bin/bash
# ============================================================================
# Dense (per-log-step) dump of grad_norm / loss / beta2 across a step window,
# to diagnose the ~step-24000 Stage-0 collapse: was it a gradient SPIKE
# (-> clipping/optimizer fix) or a gradual drift (-> LR fix)?
#
#   WIN_LO=20000 WIN_HI=27000 RUN_DIR=outputs/h100_stage0_150m_v2 \
#     sbatch scripts/diag_collapse.sh
#   cat logs/diag_collapse_<jobid>.log
# ============================================================================
#SBATCH --partition=aisc-batch
#SBATCH --account=aisc
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --job-name=diag_collapse
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log

set -uo pipefail
source "${VENV_ACTIVATE:-.venv/bin/activate}"
cd "${SLURM_SUBMIT_DIR:-.}"
export PYTHONUNBUFFERED=1
export RUN_DIR="${RUN_DIR:-outputs/h100_stage0_150m_v2}"
export WIN_LO="${WIN_LO:-20000}"
export WIN_HI="${WIN_HI:-27000}"

python - <<'PY'
import glob, os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
lo, hi = int(os.environ["WIN_LO"]), int(os.environ["WIN_HI"])
ver = sorted(glob.glob(os.path.join(os.environ["RUN_DIR"], "logs/stage0_kd/version_*")))[-1]
print("reading:", ver, "window:", lo, "-", hi)
ea = EventAccumulator(ver, size_guidance={'scalars': 0}); ea.Reload()
tags = ea.Tags()['scalars']
for tag in ("train/grad_norm", "train/ce_loss_step", "train/total_loss_step",
            "train/perplexity_step", "train/adam_beta2", "train/lr"):
    if tag in tags:
        s = [e for e in ea.Scalars(tag) if lo <= e.step <= hi]
        print(f"\n== {tag}  ({len(s)} pts in window) ==")
        for e in s:
            print(f"  step {e.step:>7}  {e.value:.5f}")
PY
