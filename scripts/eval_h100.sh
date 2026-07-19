#!/bin/bash
# ============================================================================
# H100 evaluation template (aisc-shortrun). Two modes via MODE env var:
#   MODE=ppl        (default) — PubMed val perplexity via evaluate_lm.py
#                                (Stage-0 gate; port of eval_stage0_lm.sh).
#   MODE=retrieval           — MIMIC/Indiana CXR retrieval via
#                                evaluate_cxr_retrieval.py (Phase 6/7 gate).
#
# evaluate_lm.py / evaluate_cxr_retrieval.py auto-detect layer_pattern +
# norm_topology from the --model-config yaml (refactor fixes), so pass only
# CKPT + MODEL_CONFIG. Retrieval eval also loads the fine-tuned image_encoder.*
# from the checkpoint (do NOT eval a joint ckpt with a fresh ViT).
#
# ENV: CKPT, MODEL_CONFIG, MODE, DATASET (retrieval: mimic_cxr | indiana),
#      SCRATCH_ROOT, VENV_ACTIVATE.
# ============================================================================
#SBATCH --partition=aisc-batch
#SBATCH --account=aisc
#SBATCH --gpus=1
#SBATCH --exclude=ga03,gx17v1,gx13v1   # gx13v1: faulty GPU (cudaErrorContained, 2026-07-19)
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --job-name=h100_eval
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log

set -euo pipefail

SCRATCH_ROOT="${SCRATCH_ROOT:-/sc/scratch/$USER/hybrid_xmamba_h100}"
VENV_ACTIVATE="${VENV_ACTIVATE:-.venv/bin/activate}"
MODE="${MODE:-ppl}"
MODEL_CONFIG="${MODEL_CONFIG:-hybrid_70m_v2}"
CKPT="${CKPT:?set CKPT=/path/to/checkpoint.pt}"
# NOTE: evaluate_cxr_retrieval.py's --dataset choices are exactly {mimic, indiana}
# (NOT "mimic_cxr"), and it does NOT accept --model-config (it auto-detects arch from
# the checkpoint). Its --cache-dir default is the stale willi path, so point it at the
# local HF download cache and run with HF_DATASETS_OFFLINE=1 to avoid the gated-repo 401.
DATASET="${DATASET:-mimic}"
EVAL_CACHE_DIR="${EVAL_CACHE_DIR:-${SCRATCH_ROOT}/mimic_cxr_cache}"
OUTPUT_DIR="${OUTPUT_DIR:-$(dirname "${CKPT}")/eval_results}"

echo "=== H100 eval (MODE=${MODE}) ==="
date; hostname
mkdir -p logs "${OUTPUT_DIR}"

cd "${SLURM_SUBMIT_DIR}/hybrid_model_mamba_xlstm" 2>/dev/null || cd "${SLURM_SUBMIT_DIR:-.}"

export HF_HOME="${SCRATCH_ROOT}/.hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONUNBUFFERED=1   # flush output to the log live (else block-buffered → looks frozen)

source "${VENV_ACTIVATE}"
python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print('GPU:', torch.cuda.get_device_name(0))"

if [ "${MODE}" = "ppl" ]; then
  python scripts/evaluate_lm.py \
    --checkpoint "${CKPT}" \
    --model-config "${MODEL_CONFIG}" \
    --dataset pubmed \
    --split validation \
    --batch-size 16 \
    --max-length 512 \
    --throughput \
    --output-dir "${OUTPUT_DIR}"
elif [ "${MODE}" = "retrieval" ]; then
  python scripts/evaluate_cxr_retrieval.py \
    --checkpoint "${CKPT}" \
    --dataset "${DATASET}" \
    --cache-dir "${EVAL_CACHE_DIR}" \
    --batch-size 32 \
    --output-dir "${OUTPUT_DIR}"
else
  echo "ERROR: unknown MODE='${MODE}' (expected 'ppl' or 'retrieval')"; exit 1
fi

echo "=== END: results in ${OUTPUT_DIR} ==="
date
