#!/bin/bash
# ============================================================================
# ISBI_BASELINES_PLAN.md B7-C: per-token cached decode cost after a context of L tokens, for
# mLMamba (fixed state) and the Transformer (KV cache), beam 3, one point per process.
# ENV: CONTEXTS, REPORTS (space separated), MODELS, OUTPUT_DIR.
# ============================================================================
#SBATCH --partition=pot-hpi-aisc-batch
#SBATCH --account=aisc
#SBATCH --qos=aisc
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --exclude=ga03,gx17v1,gx13v1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --job-name=isbi_decode_bench
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}/hybrid_model_mamba_xlstm" 2>/dev/null || cd "${SLURM_SUBMIT_DIR:-.}"
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
source "${VENV_ACTIVATE:-.venv/bin/activate}"
CONTEXTS="${CONTEXTS:-132 1024 4096 16384}"
REPORTS="${REPORTS:-1 16}"
MODELS="${MODELS:-hybrid_150m_m3_rrg transformer_150m_baseline_rrg}"
OUTPUT_DIR="${OUTPUT_DIR:-analysis/isbi_reviewer/decode}"
mkdir -p "${OUTPUT_DIR}"
for m in ${MODELS}; do for r in ${REPORTS}; do for L in ${CONTEXTS}; do
  echo "--- ${m} reports=${r} context=${L} ---"
  python scripts/isbi_decode_bench.py --model "${m}" --context "${L}" --reports "${r}" \
    --output "${OUTPUT_DIR}/${m}_R${r}_L${L}.json" || echo "FAILED: ${m} R${r} L${L}"
done; done; done
echo "=== END ==="
