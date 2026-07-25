#!/bin/bash
# ============================================================================
# Phase 6C — measurement block. NO TRAINING. ~1-2 GPU-h total.
#
# Runs, in order:
#   6C-3  audit_mimic_duplicates.py        (CPU, seconds)
#   6C-1  reference_biomedclip_zeroshot.py (stock BiomedCLIP, both towers)
#   6C-2  ... same script with --checkpoint (tower-swap 2x2 grid)
#
# WHY THIS BLOCK EXISTS
# Seven consecutive nulls (Stage-0 PPL 15.62->13.18, 70M->150M, negatives
# 32->128, epochs 23->14, batch 128 vs 64, head_lr 6e-4->4.24e-4 and ->3.0e-4)
# against one positive (ViT unfreeze 0->2, +2.5pp). Before spending more H100
# hours, measure the anchor the whole KD design points at and find out which
# tower actually binds.
#
# Per the 2026-07-25 decision this block does NOT gate Phase 6D — 6D runs
# regardless. It calibrates the writeup, and 6C-2 alone gates Phase 6E.
#
# ENV: CKPT (optional — omit for 6C-1 only), MIMIC_CACHE_DIR, SCRATCH_ROOT,
#      VENV_ACTIVATE, OUTPUT_DIR.
#
# Submit:
#   CKPT=./outputs/h100_kd_150m_v2_bs64_head3.0e-4/checkpoints/<best>.ckpt \
#     sbatch scripts/run_phase6c_measurements.sh
# ============================================================================
#SBATCH --partition=aisc-batch
#SBATCH --account=aisc
#SBATCH --gpus=1
#SBATCH --exclude=ga03,gx17v1,gx13v1   # gx13v1: faulty GPU (cudaErrorContained, 2026-07-19)
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --job-name=h100_phase6c
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log

set -euo pipefail

SCRATCH_ROOT="${SCRATCH_ROOT:-/sc/scratch/$USER/hybrid_xmamba_h100}"
VENV_ACTIVATE="${VENV_ACTIVATE:-.venv/bin/activate}"
MIMIC_CACHE_DIR="${MIMIC_CACHE_DIR:-/sc/home/$USER/dataset/mimic_cxr_cache}"
OUTPUT_DIR="${OUTPUT_DIR:-results/phase6c}"
CKPT="${CKPT:-}"

echo "=== Phase 6C measurement block (no training) ==="
date; hostname
mkdir -p logs "${OUTPUT_DIR}"

cd "${SLURM_SUBMIT_DIR}/hybrid_model_mamba_xlstm" 2>/dev/null || cd "${SLURM_SUBMIT_DIR:-.}"

export HF_HOME="${SCRATCH_ROOT}/.hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
# MIMIC-CXR is a GATED repo — an online load_dataset 401s. Same offline+local-cache
# setup that unblocked the Phase-6/6B runs (see 2026-07-22 notes).
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PYTHONUNBUFFERED=1

source "${VENV_ACTIVATE}"

# ---- 6C-3: duplicate / false-negative audit (CPU) --------------------------
echo ""
echo "############ 6C-3 — duplicate / false-negative audit ############"
python scripts/audit_mimic_duplicates.py \
  --cache-dir "${MIMIC_CACHE_DIR}" \
  --output-dir "${OUTPUT_DIR}"

# ---- 6C-1 (+ 6C-2 when CKPT is given) -------------------------------------
echo ""
if [ -n "${CKPT}" ]; then
  if [ ! -e "${CKPT}" ]; then
    echo "ERROR: CKPT does not exist: ${CKPT}"
    echo "NOTE: contrastive checkpoints are DIRECTORIES on this stack — check the"
    echo "      format before assuming a file path."
    exit 1
  fi
  echo "############ 6C-1 + 6C-2 — teacher ceiling + tower-swap grid ############"
  python scripts/reference_biomedclip_zeroshot.py \
    --checkpoint "${CKPT}" \
    --cache-dir "${MIMIC_CACHE_DIR}" \
    --output-dir "${OUTPUT_DIR}"
else
  echo "############ 6C-1 — teacher ceiling only (no CKPT given) ############"
  echo "Pass CKPT=... to also run the 6C-2 tower-swap grid."
  python scripts/reference_biomedclip_zeroshot.py \
    --cache-dir "${MIMIC_CACHE_DIR}" \
    --output-dir "${OUTPUT_DIR}"
fi

echo ""
echo "=== END: results in ${OUTPUT_DIR}/ ==="
echo "Interpret every gap against SE ~0.57pp at p~0.11, N=3063."
date
