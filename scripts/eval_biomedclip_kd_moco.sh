#!/bin/bash
#SBATCH --partition=mitarb
#SBATCH --account=mitarb
#SBATCH --gres=gpu:mitarb:1
#SBATCH --mem=24G
#SBATCH --time=2:00:00
#SBATCH --job-name=eval_moco
#SBATCH --output=/scratch/bhushkri/hybrid_xmamba_a100_70m_40/logs/%x_%j.log
#SBATCH --error=/scratch/bhushkri/hybrid_xmamba_a100_70m_40/logs/%x_%j.log

# Phase 5c eval — BiomedCLIP-KD + MoCo text-queue (job 1291) checkpoint.
# Runs Indiana (743 pairs, cross-dataset) then MIMIC-val (~3063 pairs, in-distribution).
#
# Decision gate (Indiana i2t R@10):
#   >= 0.40          → SUCCESS — done
#   [0.25, 0.40)     → PARTIAL — manuscript quality, no Phase 6 needed
#   [0.15, 0.25)     → MARGINAL → proceed to Phase 6 (R-Drop)
#   < 0.15           → FAIL → investigate
#
# Submit:
#   cd /scratch/bhushkri/hybrid_xmamba_a100_70m_40
#   sbatch hybrid_model_mamba_xlstm/scripts/eval_biomedclip_kd_moco.sh
#
# Override checkpoint:
#   MOCO_CKPT=/path/to/ckpt sbatch hybrid_model_mamba_xlstm/scripts/eval_biomedclip_kd_moco.sh

set -euo pipefail

MOCO_CKPT="${MOCO_CKPT:-./outputs/biomedclip_kd_moco/checkpoints/contrastive-step=001637-val/total_loss=2.4879.ckpt}"
INDIANA_CACHE="${INDIANA_CACHE:-/scratch/bhushkri/indiana_cxr_cache}"
MIMIC_CACHE="${MIMIC_CACHE:-/scratch/bhushkri/mimic_cxr_cache}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/phase5c_eval}"

echo "=== Phase 5c: BiomedCLIP-KD + MoCo Retrieval Eval ==="
date
echo "Host: $(hostname)"
echo "Checkpoint : ${MOCO_CKPT}"
echo "Indiana cache: ${INDIANA_CACHE}"
echo "MIMIC cache  : ${MIMIC_CACHE}"
echo "Output dir   : ${OUTPUT_DIR}"
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

if [ ! -f "${MOCO_CKPT}" ]; then
    echo "ERROR: checkpoint not found: ${MOCO_CKPT}"
    exit 1
fi
echo "Checkpoint verified: ${MOCO_CKPT}"

mkdir -p "${INDIANA_CACHE}" "${MIMIC_CACHE}" "${OUTPUT_DIR}"

echo ""
echo "--- [1/2] Indiana / IU-Xray (743 pairs, cross-dataset) ---"
python scripts/evaluate_cxr_retrieval.py \
    --checkpoint "${MOCO_CKPT}" \
    --dataset indiana \
    --cache-dir "${INDIANA_CACHE}" \
    --output-dir "${OUTPUT_DIR}" \
    --batch-size 32 \
    --num-workers 4 \
    --max-length 256

echo ""
echo "--- [2/2] MIMIC-CXR val (~3063 pairs, in-distribution) ---"
python scripts/evaluate_cxr_retrieval.py \
    --checkpoint "${MOCO_CKPT}" \
    --dataset mimic \
    --cache-dir "${MIMIC_CACHE}" \
    --output-dir "${OUTPUT_DIR}" \
    --batch-size 32 \
    --num-workers 4 \
    --max-length 256

echo ""
echo "=== Eval complete ==="
echo "Results in: ${OUTPUT_DIR}/"
echo ""
echo "Decision gate (Indiana i2t R@10):"
echo "  >= 0.40  → SUCCESS"
echo "  0.25-0.40 → PARTIAL"
echo "  0.15-0.25 → MARGINAL → Phase 6 (R-Drop)"
echo "  < 0.15   → FAIL → investigate"
date
