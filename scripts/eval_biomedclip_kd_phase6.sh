#!/bin/bash
#SBATCH --partition=mitarb
#SBATCH --account=mitarb
#SBATCH --gres=gpu:mitarb:1
#SBATCH --mem=24G
#SBATCH --time=2:00:00
#SBATCH --job-name=eval_phase6
#SBATCH --output=/scratch/bhushkri/hybrid_xmamba_a100_70m_40/logs/%x_%j.log
#SBATCH --error=/scratch/bhushkri/hybrid_xmamba_a100_70m_40/logs/%x_%j.log

# Phase 6 eval — no-img_proj + alpha_kd=1.0 checkpoint.
# Runs Indiana (743 pairs) then MIMIC-val (~3063 pairs).
#
# Decision gate (Indiana i2t R@10):
#   >= 0.40   → SUCCESS
#   0.25-0.40 → PARTIAL
#   0.15-0.25 → MARGINAL → investigate further
#   < 0.15    → FAIL
#
# Submit:
#   cd /scratch/bhushkri/hybrid_xmamba_a100_70m_40
#   sbatch hybrid_model_mamba_xlstm/scripts/eval_biomedclip_kd_phase6.sh
#
# Override checkpoint:
#   PHASE6_CKPT=/path/to/ckpt sbatch hybrid_model_mamba_xlstm/scripts/eval_biomedclip_kd_phase6.sh

set -euo pipefail

PHASE6_CKPT="${PHASE6_CKPT:-./outputs/biomedclip_kd_phase6/checkpoints/best.ckpt}"
INDIANA_CACHE="${INDIANA_CACHE:-/scratch/bhushkri/indiana_cxr_cache}"
MIMIC_CACHE="${MIMIC_CACHE:-/scratch/bhushkri/mimic_cxr_cache}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/phase6_eval}"

echo "=== Phase 6 Eval: no img_proj + alpha_kd=1.0 ==="
date
echo "Host: $(hostname)"
echo "Checkpoint : ${PHASE6_CKPT}"
echo ""

mkdir -p /scratch/bhushkri/hybrid_xmamba_a100_70m_40/logs
cd "${SLURM_SUBMIT_DIR}/hybrid_model_mamba_xlstm"

export HF_HOME="$PWD/.hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"

if [ ! -d ".venv" ]; then
    echo "ERROR: .venv not found"
    exit 1
fi
source .venv/bin/activate

if [ ! -f "${PHASE6_CKPT}" ]; then
    echo "Resolving best checkpoint by lowest val/total_loss..."
    PHASE6_CKPT=$(find ./outputs/biomedclip_kd_phase6/checkpoints -name "*.ckpt" \
        ! -name "last*.ckpt" | sort | head -1)
    if [ -z "${PHASE6_CKPT}" ]; then
        echo "ERROR: no checkpoint found in ./outputs/biomedclip_kd_phase6/checkpoints/"
        exit 1
    fi
    echo "Using: ${PHASE6_CKPT}"
fi

mkdir -p "${INDIANA_CACHE}" "${MIMIC_CACHE}" "${OUTPUT_DIR}"

echo ""
echo "--- [1/2] Indiana / IU-Xray (743 pairs, cross-dataset) ---"
python scripts/evaluate_cxr_retrieval.py \
    --checkpoint "${PHASE6_CKPT}" \
    --dataset indiana \
    --cache-dir "${INDIANA_CACHE}" \
    --output-dir "${OUTPUT_DIR}" \
    --batch-size 32 \
    --num-workers 2 \
    --max-length 256

echo ""
echo "--- [2/2] MIMIC-CXR val (~3063 pairs, in-distribution) ---"
python scripts/evaluate_cxr_retrieval.py \
    --checkpoint "${PHASE6_CKPT}" \
    --dataset mimic \
    --cache-dir "${MIMIC_CACHE}" \
    --output-dir "${OUTPUT_DIR}" \
    --batch-size 32 \
    --num-workers 2 \
    --max-length 256

echo ""
echo "=== Eval complete ==="
echo "Results in: ${OUTPUT_DIR}/"
date
