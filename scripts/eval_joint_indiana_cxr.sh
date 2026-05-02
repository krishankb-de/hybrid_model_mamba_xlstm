#!/bin/bash
#SBATCH --partition=mitarb
#SBATCH --account=mitarb
#SBATCH --gres=gpu:mitarb:1
#SBATCH --mem=24G
#SBATCH --time=1:00:00
#SBATCH --job-name=eval_indiana_cxr
#SBATCH --output=/scratch/bhushkri/hybrid_xmamba_a100_70m_40/logs/%x_%j.log
#SBATCH --error=/scratch/bhushkri/hybrid_xmamba_a100_70m_40/logs/%x_%j.log

# Phase 6 — Indiana/IU-Xray cross-dataset retrieval eval.
# Evaluates the best joint MIMIC-CXR checkpoint on the 743-sample IU-Xray test set.
# Decision gate: i2t R@10 >= 0.40 → done | [0.25, 0.40) → Phase 7 | < 0.25 → debug
#
# Submit from parent dir:
#   cd /scratch/bhushkri/hybrid_xmamba_a100_70m_40
#   sbatch hybrid_model_mamba_xlstm/scripts/eval_joint_indiana_cxr.sh
#
# Override checkpoint:
#   JOINT_CKPT=/path/to/ckpt sbatch hybrid_model_mamba_xlstm/scripts/eval_joint_indiana_cxr.sh

set -euo pipefail

JOINT_CKPT="${JOINT_CKPT:-./outputs/joint_mimic_cxr/checkpoints/contrastive-step=001915-val/total_loss=1.9140.ckpt}"
INDIANA_CACHE="${INDIANA_CACHE:-/scratch/bhushkri/indiana_cxr_cache}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/phase6_eval}"

echo "=== Phase 6: Indiana/IU-Xray Retrieval Eval ==="
date
echo "Host: $(hostname)"
echo "Checkpoint : ${JOINT_CKPT}"
echo "Cache dir  : ${INDIANA_CACHE}"
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

mkdir -p "${INDIANA_CACHE}" "${OUTPUT_DIR}"

python scripts/evaluate_cxr_retrieval.py \
    --checkpoint "${JOINT_CKPT}" \
    --dataset indiana \
    --cache-dir "${INDIANA_CACHE}" \
    --output-dir "${OUTPUT_DIR}" \
    --batch-size 32 \
    --num-workers 4 \
    --max-length 256

echo ""
echo "=== Eval complete ==="
echo "Results in: ${OUTPUT_DIR}/phase6_indiana_*.json"
date
