#!/bin/bash
# ============================================================================
# Phase 11B (2026-08-29) — runs scripts/score_chexbert_standalone.py against
# hyps.txt/refs.txt dumped by evaluate_report_generation.py's --dump-dir flag,
# using the ISOLATED venv from setup_chexbert_venv_h100.sh (transformers<5,
# no hybrid_xmamba/Triton) -- see that script's and score_chexbert_standalone
# .py's module docstrings for why this is a separate venv, not the main one.
#
# SLURM wrapper for the same login-node-refuses-this-command reason as every
# other eval wrapper in this repo.
# ============================================================================
#SBATCH --partition=aisc-batch
#SBATCH --account=aisc
#SBATCH --qos=aisc
#SBATCH --exclude=ga03,gx17v1,gx13v1   # ga03: ARM node, x86 venv incompatible;
                                        # gx13v1: faulty GPU (cudaErrorContained, 2026-07-19)
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --job-name=score_chexbert
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log

set -euo pipefail

VENV_ACTIVATE="${VENV_ACTIVATE:-.venv_chexbert/bin/activate}"

DUMP_DIR="${DUMP_DIR:?Set DUMP_DIR to the directory containing hyps.txt/refs.txt (from evaluate_report_generation.py --dump-dir)}"
HYP_FILE="${HYP_FILE:-${DUMP_DIR}/hyps.txt}"
REF_FILE="${REF_FILE:-${DUMP_DIR}/refs.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-${DUMP_DIR}}"

echo "=== Phase 11B standalone CheXbert scoring: ${HYP_FILE} / ${REF_FILE} ==="
date; hostname
mkdir -p logs

cd "${SLURM_SUBMIT_DIR}/hybrid_model_mamba_xlstm" 2>/dev/null || cd "${SLURM_SUBMIT_DIR:-.}"

if [ ! -f "${VENV_ACTIVATE}" ]; then
  echo "ERROR: isolated venv not found at ${VENV_ACTIVATE} -- run: sbatch scripts/setup_chexbert_venv_h100.sh"
  exit 1
fi

source "${VENV_ACTIVATE}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"   # first run needs the network to fetch chexbert.pth/bert-base-uncased
export PYTHONUNBUFFERED=1

if [ ! -f "${HYP_FILE}" ] || [ ! -f "${REF_FILE}" ]; then
  echo "ERROR: ${HYP_FILE} or ${REF_FILE} not found -- run evaluate_report_generation.py with --dump-dir first"
  exit 1
fi

python scripts/score_chexbert_standalone.py \
  --hyp-file "${HYP_FILE}" \
  --ref-file "${REF_FILE}" \
  --output-dir "${OUTPUT_DIR}"

echo "=== END ==="
date
