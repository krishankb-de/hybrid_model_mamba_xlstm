#!/bin/bash
# TEXTENC_PLAN.md T3-A/B: build the re-ranked retrieval index (train/validate/test) and the two
# copy-the-top-1 floors from the Stage-1 checkpoint of textenc_stage1_h100.sh.
#   TEXTENC_ROOT=$HOME/hybrid_textenc sbatch scripts/textenc_rag_index_h100.sh
#SBATCH --partition=pot-hpi-aisc-batch
#SBATCH --account=aisc
#SBATCH --qos=aisc
#SBATCH --gpus=1
#SBATCH --exclude=ga03,gx17v1,gx13v1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --job-name=textenc_rag_index
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-.}/scripts/textenc_guard.sh"
STAGE1_CKPT="${STAGE1_CKPT:-${ROOT_REAL}/outputs/textenc_stage1_m3/checkpoints/last.ckpt}"
RAG_DIR="${RAG_DIR:-${ROOT_REAL}/results/textenc_rag}"
cd "${ROOT_REAL}"
export HF_HOME="${SCRATCH_ROOT:-/sc/scratch/${USER}/hybrid_xmamba_h100}/.hf" HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
source "${VENV_ACTIVATE:-.venv/bin/activate}"
echo "=== TEXTENC T3 index: ckpt=${STAGE1_CKPT} -> ${RAG_DIR} ==="; date; hostname
python scripts/textenc_rag_index.py --stage1-ckpt "${STAGE1_CKPT}" --data "${TEXTENC_DATA}" \
  --out "${RAG_DIR}" --candidates "${RAG_CANDIDATES:-50}" --k "${RAG_K:-3}"
echo "=== END ==="; date
