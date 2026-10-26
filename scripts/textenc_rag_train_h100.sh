#!/bin/bash
# TEXTENC_PLAN.md T5-A/B: Stage 3 with retrieved reports in the context.
# The published recipe (train_report_generation_h100.sh, the 3-seed tower13d runs: 12k steps, 4 GPUs,
# batch 32/GPU, decoder 1e-5, head 3e-4, clip 0.5, prefix 32, the 13D image tower, Stage-0 init) run
# through textenc_train_rag.py. Outputs go to TEXTENC_ROOT/outputs only.
#   TEXTENC_ROOT=$HOME/hybrid_textenc DECODER=m3 SEED=42 sbatch scripts/textenc_rag_train_h100.sh
#SBATCH --partition=pot-hpi-aisc-batch
#SBATCH --account=aisc
#SBATCH --qos=aisc
#SBATCH --gpus=4
#SBATCH --exclude=ga03,gx17v1,gx13v1
#SBATCH --mem=256G
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --job-name=textenc_rag_train
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-.}/scripts/textenc_guard.sh"
DECODER="${DECODER:-m3}"
case "${DECODER}" in
  m3)          export MODEL_CONFIG=hybrid_150m_m3_rrg
               export DECODER_CKPT="${PUBLISHED_OUTPUTS}/h100_stage0_150m_m3/checkpoints/last.ckpt" ;;
  transformer) export MODEL_CONFIG=transformer_150m_baseline_rrg
               export DECODER_CKPT="${PUBLISHED_OUTPUTS}/h100_stage0_transformer_150m/checkpoints/last.ckpt" ;;
  *) echo "ERROR: unknown DECODER=${DECODER}"; exit 2 ;;
esac
export SEED="${SEED:-42}" NUM_GPUS=4 MAX_STEPS=12000 BATCH_SIZE=32 SAVE_TOP_K=0 AUX_LAMBDA=0.0 PREFIX_K=32
# SMOKE=true: the same path for 20 steps on 1 GPU (submit with --gpus=1 --time=00:40:00), so a
# broken entrypoint fails in minutes instead of after a multi-day queue for 4 GPUs.
if [ "${SMOKE:-false}" = "true" ]; then
  export NUM_GPUS=1 MAX_STEPS=20
  SMOKE_SUFFIX=_smoke
fi
export DECODER_LR=1e-5 HEAD_LR=3e-4 GRAD_CLIP=0.5 VIT_LR=1e-6
export IMAGE_ENCODER_CKPT="${PUBLISHED_OUTPUTS}/h100_kd_150m_v2_full_data_lr3e6/checkpoints/last.ckpt"
export MIMIC_CACHE_DIR="${ROOT_REAL}/cache/mimic_cxr_cache"
export EXPERIMENT="${EXPERIMENT:-textenc_rag_${DECODER}_k${RAG_K:-3}_s${SEED}${SMOKE_SUFFIX:-}}"
export TEXTENC_RAG_DIR="${TEXTENC_RAG_DIR:-${ROOT_REAL}/results/textenc_rag}"
export TEXTENC_DATA RAG_K="${RAG_K:-3}" RAG_REPORT_TOKENS="${RAG_REPORT_TOKENS:-160}"
for f in "${DECODER_CKPT}" "${IMAGE_ENCODER_CKPT}" "${TEXTENC_RAG_DIR}/rag_train.pt" "${TEXTENC_RAG_DIR}/rag_validate.pt"; do
  [ -f "$f" ] || { echo "ERROR: missing $f"; exit 1; }
done
cd "${ROOT_REAL}"
# The published wrapper stays byte-identical; this job runs a copy of it in which exactly one line,
# the python entrypoint, is swapped for textenc_train_rag.py (which calls the published main()).
mkdir -p "${ROOT_REAL}/cache"
RUN_COPY="${ROOT_REAL}/cache/train_report_generation_rag_${SLURM_JOB_ID:-local}.sh"
sed 's#^python scripts/train_report_generation.py \\$#python scripts/textenc_train_rag.py \\#' \
  scripts/train_report_generation_h100.sh > "${RUN_COPY}"
CHANGED=$(diff scripts/train_report_generation_h100.sh "${RUN_COPY}" | grep -c '^>' || true)
if [ "${CHANGED}" != "1" ] || ! grep -q '^python scripts/textenc_train_rag.py \\$' "${RUN_COPY}"; then
  echo "ERROR: expected exactly one swapped entrypoint line in ${RUN_COPY} (changed=${CHANGED})"
  exit 1
fi
echo "=== TEXTENC T5 RAG Stage 3: decoder=${DECODER} seed=${SEED} k=${RAG_K} -> outputs/${EXPERIMENT} ==="
exec bash "${RUN_COPY}"
