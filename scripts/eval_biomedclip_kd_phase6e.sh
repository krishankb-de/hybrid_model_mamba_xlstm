#!/bin/bash
#SBATCH --partition=mitarb
#SBATCH --account=mitarb
#SBATCH --gres=gpu:mitarb:1
#SBATCH --mem=24G
#SBATCH --time=2:00:00
#SBATCH --job-name=eval_phase6e
#SBATCH --output=/scratch/bhushkri/hybrid_xmamba_a100_70m_40/logs/%x_%j.log
#SBATCH --error=/scratch/bhushkri/hybrid_xmamba_a100_70m_40/logs/%x_%j.log

# Phase 6e eval — no MoCo queue, in-batch CLIP + KD warmup.
# Runs Indiana (743 pairs) then MIMIC-val (~3063 pairs).
#
# Decision gate (MIMIC i2t R@10 after 5000 steps):
#   >= 12% AND > Phase 5c (9.99%) → SUCCESS: KD warmup adds value; run Indiana full eval
#   9–12%                          → PARITY: neutral vs 5c; try alpha_kd_post sweep {0.1,0.3,0.5}
#   < 9%                           → FAIL: in-batch insufficient; restore small queue K=256
#
# Submit:
#   cd /scratch/bhushkri/hybrid_xmamba_a100_70m_40
#   sbatch hybrid_model_mamba_xlstm/scripts/eval_biomedclip_kd_phase6e.sh
#
# Override checkpoint:
#   PHASE6E_CKPT=/path/to/ckpt sbatch hybrid_model_mamba_xlstm/scripts/eval_biomedclip_kd_phase6e.sh

set -euo pipefail

PHASE6E_CKPT="${PHASE6E_CKPT:-./outputs/biomedclip_kd_phase6e/checkpoints/best.ckpt}"
INDIANA_CACHE="${INDIANA_CACHE:-/scratch/bhushkri/indiana_cxr_cache}"
MIMIC_CACHE="${MIMIC_CACHE:-/scratch/bhushkri/mimic_cxr_cache}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/phase6e_eval}"

echo "=== Phase 6e Eval: no MoCo queue, in-batch CLIP + KD warmup ==="
date
echo "Host: $(hostname)"
echo "Checkpoint : ${PHASE6E_CKPT}"
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

if [ ! -f "${PHASE6E_CKPT}" ]; then
    echo "Resolving best checkpoint by lowest val/total_loss..."
    PHASE6E_CKPT=$(find ./outputs/biomedclip_kd_phase6e/checkpoints -name "*.ckpt" \
        ! -name "last*.ckpt" | sort | head -1)
    if [ -z "${PHASE6E_CKPT}" ]; then
        echo "ERROR: no checkpoint found in ./outputs/biomedclip_kd_phase6e/checkpoints/"
        exit 1
    fi
    echo "Using: ${PHASE6E_CKPT}"
fi

mkdir -p "${INDIANA_CACHE}" "${MIMIC_CACHE}" "${OUTPUT_DIR}"

echo ""
echo "--- [1/2] Indiana / IU-Xray (743 pairs, cross-dataset) ---"
python scripts/evaluate_cxr_retrieval.py \
    --checkpoint "${PHASE6E_CKPT}" \
    --dataset indiana \
    --cache-dir "${INDIANA_CACHE}" \
    --output-dir "${OUTPUT_DIR}" \
    --batch-size 32 \
    --num-workers 2 \
    --max-length 256

echo ""
echo "--- [2/2] MIMIC-CXR val (~3063 pairs, in-distribution) ---"
python scripts/evaluate_cxr_retrieval.py \
    --checkpoint "${PHASE6E_CKPT}" \
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
