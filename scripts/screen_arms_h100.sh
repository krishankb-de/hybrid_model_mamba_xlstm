#!/bin/bash
# ============================================================================
# MAMBA3_PLAN.md M7-B0 — the short-run screen, as one SLURM job array.
#
# One submission instead of five; each task takes an H100 as one frees, so the queue
# wait is paid once in parallel rather than five times in series.
#
#     sbatch --array=0-4 scripts/screen_arms_h100.sh          # A2..A6
#     sbatch --array=0-4%2 scripts/screen_arms_h100.sh        # ...at most 2 at a time
#     ARMS="A3 A5" sbatch --array=0-1 scripts/screen_arms_h100.sh
#
# The arm table is NOT written here. It lives in scripts/mamba3_arms.py, which is also
# what the pre-flight verifies -- so an arm cannot be screened with a lever the
# pre-flight never checked. That separation is the FM5 defence: on 2026-09-06 a
# pre-flight with its own private copy of the arm list passed while the trainer was
# silently dropping three config fields, and job 2513007 trained A1 as plain A0.
#
# A0, A0-seed and A1 were early-started under M7-A2 and are NOT in the default set --
# re-running them would waste ~22 GPU-h and, worse, produce a second control.
#
# Recipe is held fixed across every arm on purpose (FM7): the 150M wrapper's LR 4e-4,
# warmup 500, grad-clip 0.5, bs 16 x accum 3, and GRAD_CKPT=true even though SSD makes
# checkpointing unnecessary for the mamba3 arms. An arm that trains faster because it
# was allowed a different recipe is not a measurement.
# ============================================================================
#SBATCH --partition=aisc-batch
#SBATCH --account=aisc
#SBATCH --gpus=1
#SBATCH --exclude=ga03,gx17v1,gx13v1
#SBATCH --mem=160G
#SBATCH --cpus-per-task=12
#SBATCH --time=12:00:00
#SBATCH --job-name=m3_screen
#SBATCH --output=logs/%x_%A_%a.log
#SBATCH --error=logs/%x_%A_%a.log
#SBATCH --requeue

set -euo pipefail

ARMS="${ARMS:-A2 A3 A4 A5 A6}"
STEPS="${STEPS:-12000}"
WARMUP_STEPS="${WARMUP_STEPS:-500}"
SAVE_TOP_K_SCREEN="${SAVE_TOP_K_SCREEN:-1}"   # screen arms are not pipeline inputs; 1 is plenty

cd "${SLURM_SUBMIT_DIR:-.}/hybrid_model_mamba_xlstm" 2>/dev/null || cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs

read -r -a ARM_LIST <<< "${ARMS}"
IDX="${SLURM_ARRAY_TASK_ID:-0}"
if [ "${IDX}" -ge "${#ARM_LIST[@]}" ]; then
  echo "FATAL: array index ${IDX} but only ${#ARM_LIST[@]} arms in ARMS='${ARMS}'."
  echo "       Use --array=0-$(( ${#ARM_LIST[@]} - 1 ))."
  exit 1
fi
ARM="${ARM_LIST[$IDX]}"

echo "=== M7-B screen: arm ${ARM} (array task ${IDX} of ${#ARM_LIST[@]}) ==="
date; hostname
echo "branch: $(git rev-parse --abbrev-ref HEAD) @ $(git rev-parse --short HEAD)"

source "${VENV_ACTIVATE:-.venv/bin/activate}"

# The single definition of the ladder decides config, seed and levers. Nothing here
# hand-writes a `model.mamba3_*=` override, which is the point.
eval "$(python scripts/mamba3_arms.py env "${ARM}" \
          --steps "${STEPS}" --warmup "${WARMUP_STEPS}" --save-top-k "${SAVE_TOP_K_SCREEN}")"
echo "arm ${ARM}: MODEL_CONFIG=${MODEL_CONFIG} SEED=${SEED} EXPERIMENT=${EXPERIMENT}"
echo "arm ${ARM}: EXTRA_OVERRIDES=${EXTRA_OVERRIDES:-<none>}"

# Delegate to the 150M wrapper so the stability settings that took five attempts to find
# are inherited rather than restated (LR 4e-4, grad-clip 0.5, 80GB-safe bs/accum).
bash scripts/train_stage0_150m_h100.sh

echo "=== arm ${ARM} finished ==="
date
