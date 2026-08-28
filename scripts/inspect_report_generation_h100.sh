#!/bin/bash
# ============================================================================
# Phase 11A --checkpoint mode (H100_SCALING_PLAN.md) — qualitative check of a
# trained Phase 10E report-generation checkpoint against real held-out images.
# SLURM wrapper for scripts/evaluate_report_generation.py --checkpoint, since
# the login node refuses this ("resource-intensive... use srun/a Slurm job")
# — same login-node constraint documented in build_mimic_cxr_local.sh, hit
# live 2026-08-23. Mirrors train_report_generation_h100.sh's env-lever /
# --exclude / offline-HF conventions so this is a drop-in sibling.
#
# Lightweight (a handful of forward passes, no training) — deliberately a
# short --time and no --gpus request beyond 1, not the full 16h training cap.
# ============================================================================
#SBATCH --partition=aisc-batch
#SBATCH --account=aisc
#SBATCH --gpus=1
#SBATCH --exclude=ga03,gx17v1,gx13v1   # ga03: ARM node, x86 .venv incompatible;
                                        # gx13v1: faulty GPU (cudaErrorContained, 2026-07-19)
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=06:30:00
#SBATCH --job-name=inspect_report_gen
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log

set -euo pipefail

SCRATCH_ROOT="${SCRATCH_ROOT:-/sc/scratch/$USER/hybrid_xmamba_h100}"
VENV_ACTIVATE="${VENV_ACTIVATE:-.venv/bin/activate}"

CHECKPOINT="${CHECKPOINT:-./outputs/h100_report_gen_arm0/checkpoints/last.ckpt}"
MODEL_CONFIG="${MODEL_CONFIG:-hybrid_150m_v2_rrg}"
# Default to VALIDATION images, not train -- generations on train images look
# artificially good even under genuine overfitting; validation is the honest check.
PARQUET="${PARQUET:-/sc/home/$USER/dataset/mimic_full/arm0/validate.parquet}"
NUM_SAMPLES="${NUM_SAMPLES:-10}"
DECODE="${DECODE:-greedy}"
BEAM_SIZE="${BEAM_SIZE:-3}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-100}"
# Phase 11B (2026-08-28): opt-in CheXbert F1 alongside ROUGE-L/BLEU, off by
# default since it needs f1chexbert + network on first use (UNVERIFIED
# against real weights as of this writing -- see evaluate_report_generation.py's
# label_reports_with_chexbert docstring). CHEXPERT_CSV enables the ground-
# truth labeler-wiring cross-check when set alongside CHEXBERT=true.
CHEXBERT="${CHEXBERT:-false}"
CHEXPERT_CSV="${CHEXPERT_CSV:-/sc/home/$USER/dataset/mimic_full/mimic-cxr-2.0.0-chexpert.csv.gz}"

echo "=== Phase 11A checkpoint inspection: ${CHECKPOINT} ==="
echo "=== parquet=${PARQUET} num_samples=${NUM_SAMPLES} decode=${DECODE} ==="
date; hostname
mkdir -p logs

cd "${SLURM_SUBMIT_DIR}/hybrid_model_mamba_xlstm" 2>/dev/null || cd "${SLURM_SUBMIT_DIR:-.}"

export HF_HOME="${SCRATCH_ROOT}/.hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PYTHONUNBUFFERED=1

source "${VENV_ACTIVATE}"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

if [ ! -f "${CHECKPOINT}" ]; then
  echo "ERROR: Checkpoint not found: ${CHECKPOINT}"
  exit 1
fi

CHEXBERT_ARGS=()
if [ "${CHEXBERT}" = "true" ]; then
  CHEXBERT_ARGS+=(--chexbert --chexpert-csv "${CHEXPERT_CSV}")
fi

python scripts/evaluate_report_generation.py \
  --checkpoint "${CHECKPOINT}" \
  --model-config "${MODEL_CONFIG}" \
  --parquet "${PARQUET}" \
  --num-samples "${NUM_SAMPLES}" \
  --decode "${DECODE}" \
  --beam-size "${BEAM_SIZE}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  ${CHEXBERT_ARGS[@]+"${CHEXBERT_ARGS[@]}"}

echo "=== END ==="
date
