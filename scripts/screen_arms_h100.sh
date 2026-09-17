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
#SBATCH --open-mode=append   # aisc-batch is preemptible: without this a requeue
                             # TRUNCATES the log and the restart leaves no trace
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

# The 150M wrapper resolves ARM itself, on the compute node -- never here, and never
# before sbatch. The aisc login node refuses to execute python, so a pre-submit
# `eval "$(python scripts/mamba3_arms.py env A2)"` silently exports nothing and the job
# runs the wrapper's defaults; that is how job 2513581 became a second A0 at 120,000
# steps on 2026-09-06.
# Every array task must use the arm's canonical experiment name. SLURM propagates the
# submitting environment, and the wrapper now lets a caller-supplied EXPERIMENT win (so a
# probe does not write into a screen run's directory) -- which means a stray EXPERIMENT
# exported in the login shell would silently funnel ALL FIVE arms into one output
# directory, each overwriting the last one's checkpoints. Clear it.
unset EXPERIMENT
export ARM STEPS WARMUP_STEPS
export SAVE_TOP_K="${SAVE_TOP_K_SCREEN}"
bash scripts/train_stage0_150m_h100.sh

echo "=== arm ${ARM} finished ==="
date
