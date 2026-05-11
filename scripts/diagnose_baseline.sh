#!/bin/bash
#SBATCH --partition=mitarb
#SBATCH --account=mitarb
#SBATCH --gres=gpu:mitarb:1
#SBATCH --mem=24G
#SBATCH --time=1:00:00
#SBATCH --job-name=diagnose_phase6e
#SBATCH --output=/scratch/bhushkri/hybrid_xmamba_a100_70m_40/logs/%x_%j.log
#SBATCH --error=/scratch/bhushkri/hybrid_xmamba_a100_70m_40/logs/%x_%j.log

# Phase 2 — diagnostic probe on Phase 6e best checkpoint.
#
# Submit from willi:
#   cd /scratch/bhushkri/hybrid_xmamba_a100_70m_40
#   sbatch hybrid_model_mamba_xlstm/scripts/diagnose_baseline.sh
#
# Override checkpoint path:
#   PHASE6E_CKPT=/custom/path.ckpt sbatch diagnose_baseline.sh

set -euo pipefail

PHASE6E_CKPT="${PHASE6E_CKPT:-./outputs/biomedclip_kd_phase6e/checkpoints/best.ckpt}"
MIMIC_CACHE="${MIMIC_CACHE:-/scratch/bhushkri/mimic_cxr_cache}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/baseline_probe}"

echo "=== Phase 2: Diagnostic Probe ==="
date
echo "Host     : $(hostname)"
echo "Checkpoint: ${PHASE6E_CKPT}"
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

python scripts/diagnose_baseline.py \
    --ckpt "${PHASE6E_CKPT}" \
    --device cuda \
    --output-dir "${OUTPUT_DIR}" \
    --mimic-cache "${MIMIC_CACHE}" \
    --max-pairs 500

echo ""
echo "=== Probe complete ==="
date
echo "Results: ${OUTPUT_DIR}/report.json"
