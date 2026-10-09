#!/bin/bash
# ============================================================================
# ISBI_BASELINES_PLAN.md B5-A — cost of each retrieval floor (rule R4: bf16,
# batch 4, 3 warmup + 10 timed iterations, one H100). One process per encoder
# so no encoder inherits another's allocator state.
#
# ENV: ENCODERS (space separated, default all five), OUTPUT_DIR, HF_TOKEN_FILE,
#      HF_HUB_OFFLINE (set 0 on the first run of a new encoder), VENV_ACTIVATE.
# Random pixels and a random index of the real shape: no MIMIC data is read.
# ============================================================================
#SBATCH --partition=pot-hpi-aisc-batch
#SBATCH --account=aisc
#SBATCH --qos=aisc
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --exclude=ga03,gx17v1,gx13v1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=00:45:00
#SBATCH --job-name=isbi_floor_cost
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log

set -euo pipefail

SCRATCH_ROOT="${SCRATCH_ROOT:-/sc/scratch/$USER/hybrid_xmamba_h100}"
VENV_ACTIVATE="${VENV_ACTIVATE:-.venv/bin/activate}"
ENCODERS="${ENCODERS:-biomedclip clip pubmedclip xrayclip medsiglip}"
OUTPUT_DIR="${OUTPUT_DIR:-analysis/efficiency_isbi_short}"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-$HOME/.hf_token}"

cd "${SLURM_SUBMIT_DIR}/hybrid_model_mamba_xlstm" 2>/dev/null || cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs "${OUTPUT_DIR}"

export HF_HOME="${SCRATCH_ROOT}/.hf"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PYTHONUNBUFFERED=1
if [ -f "${HF_TOKEN_FILE}" ]; then
  HF_TOKEN="$(tr -d '[:space:]' < "${HF_TOKEN_FILE}")"
  export HF_TOKEN
  echo "HF token: loaded from ${HF_TOKEN_FILE}"
fi

source "${VENV_ACTIVATE}"
for enc in ${ENCODERS}; do
  echo "--- ${enc} ---"
  python scripts/isbi_floor_cost.py --encoder "${enc}" \
    --output "${OUTPUT_DIR}/floor_cost_${enc}.json" || echo "FAILED: ${enc}"
done
echo "=== END ==="
