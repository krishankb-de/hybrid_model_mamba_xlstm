#!/bin/bash
#SBATCH --partition=mitarb
#SBATCH --account=mitarb
#SBATCH --gres=gpu:mitarb:1
#SBATCH --mem=24G
#SBATCH --time=1:30:00
#SBATCH --job-name=eval_mimic_val
#SBATCH --output=/scratch/bhushkri/hybrid_xmamba_a100_70m_40/logs/%x_%j.log
#SBATCH --error=/scratch/bhushkri/hybrid_xmamba_a100_70m_40/logs/%x_%j.log

# Phase 6 — MIMIC-CXR val in-distribution retrieval eval.
# Evaluates the best joint checkpoint on the 10% held-out MIMIC val slice (~3063 pairs).
# This is an in-distribution sanity check; Indiana is the real cross-dataset signal.
#
# Submit from parent dir:
#   cd /scratch/bhushkri/hybrid_xmamba_a100_70m_40
#   sbatch hybrid_model_mamba_xlstm/scripts/eval_joint_mimic_cxr_val.sh
#
# Override checkpoint or cache:
#   JOINT_CKPT=/path/to/ckpt sbatch hybrid_model_mamba_xlstm/scripts/eval_joint_mimic_cxr_val.sh

set -euo pipefail

JOINT_CKPT="${JOINT_CKPT:-./outputs/joint_mimic_cxr/checkpoints/contrastive-step=001915-val/total_loss=1.9140.ckpt}"
MIMIC_CACHE="${MIMIC_CACHE:-/scratch/bhushkri/mimic_cxr_cache}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/phase6_eval}"

echo "=== Phase 6: MIMIC-CXR Val Retrieval Eval ==="
date
echo "Host: $(hostname)"
echo "Checkpoint : ${JOINT_CKPT}"
echo "Cache dir  : ${MIMIC_CACHE}"
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

python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB')
"

if [ ! -f "${JOINT_CKPT}" ]; then
    echo "ERROR: checkpoint not found: ${JOINT_CKPT}"
    exit 1
fi

mkdir -p "${MIMIC_CACHE}" "${OUTPUT_DIR}"

python scripts/evaluate_cxr_retrieval.py \
    --checkpoint "${JOINT_CKPT}" \
    --dataset mimic \
    --cache-dir "${MIMIC_CACHE}" \
    --output-dir "${OUTPUT_DIR}" \
    --batch-size 32 \
    --num-workers 4 \
    --max-length 256

echo ""
echo "=== Eval complete ==="
echo "Results in: ${OUTPUT_DIR}/phase6_mimic_*.json"
date
