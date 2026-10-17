#!/bin/bash
# TEXTENC_PLAN.md T5-C: beam-3 test decode with retrieved reports, sharded as an array job.
#   TEXTENC_ROOT=$HOME/hybrid_textenc DECODER=m3 SEED=42 sbatch --array=0-3 scripts/textenc_rag_eval_h100.sh
# then merge + CheXbert (after the array finishes):
#   TEXTENC_ROOT=$HOME/hybrid_textenc DECODER=m3 SEED=42 MERGE=true sbatch scripts/textenc_rag_eval_h100.sh
#SBATCH --partition=pot-hpi-aisc-batch
#SBATCH --account=aisc
#SBATCH --qos=aisc
#SBATCH --gpus=1
#SBATCH --exclude=ga03,gx17v1,gx13v1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --job-name=textenc_rag_eval
#SBATCH --output=logs/%x_%A_%a.log
#SBATCH --error=logs/%x_%A_%a.log
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-.}/scripts/textenc_guard.sh"
DECODER="${DECODER:-m3}"
case "${DECODER}" in
  m3) MODEL_CONFIG=hybrid_150m_m3_rrg ;; transformer) MODEL_CONFIG=transformer_150m_baseline_rrg ;;
  *) echo "ERROR: unknown DECODER=${DECODER}"; exit 2 ;;
esac
SEED="${SEED:-42}"; RAG_K="${RAG_K:-3}"; NUM_SHARDS="${NUM_SHARDS:-4}"
EXPERIMENT="${EXPERIMENT:-textenc_rag_${DECODER}_k${RAG_K}_s${SEED}}"
DUMP_DIR="${ROOT_REAL}/results/${EXPERIMENT}_test_split"
RAG_DIR="${TEXTENC_RAG_DIR:-${ROOT_REAL}/results/textenc_rag}"
cd "${ROOT_REAL}"
export HF_HOME="${SCRATCH_ROOT:-/sc/scratch/${USER}/hybrid_xmamba_h100}/.hf" HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
source "${VENV_ACTIVATE:-.venv/bin/activate}"
COMMON=(--model-config "${MODEL_CONFIG}" --rag-dir "${RAG_DIR}" --data "${TEXTENC_DATA}"
        --dump-dir "${DUMP_DIR}" --k "${RAG_K}" --report-tokens "${RAG_REPORT_TOKENS:-160}"
        --num-shards "${NUM_SHARDS}")
if [ "${MERGE:-false}" = "true" ]; then
  python scripts/textenc_eval_rag.py "${COMMON[@]}" --merge
  exit 0
fi
SHARD="${SLURM_ARRAY_TASK_ID:-0}"
echo "=== TEXTENC T5 eval: ${EXPERIMENT} shard ${SHARD}/${NUM_SHARDS} ==="; date; hostname
python scripts/textenc_eval_rag.py "${COMMON[@]}" \
  --checkpoint "${ROOT_REAL}/outputs/${EXPERIMENT}/checkpoints/last.ckpt" \
  --shard "${SHARD}" --check-uncached "${CHECK_UNCACHED:-2}"
echo "=== END ==="; date
