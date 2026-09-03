#!/bin/bash
# ============================================================================
# H100 evaluation template (aisc-shortrun). Three modes via MODE env var:
#   MODE=ppl        (default) — PubMed val perplexity via evaluate_lm.py
#                                (Stage-0 gate; port of eval_stage0_lm.sh).
#   MODE=retrieval           — MIMIC/Indiana CXR retrieval via
#                                evaluate_cxr_retrieval.py (Phase 6/7 gate).
#   MODE=sts                 — BIOSSES/STS-B/MedSTS Spearman rho via
#                                evaluate_sts.py (Phase 12A). Needs a
#                                JOINT-trained checkpoint with projection_head.*
#                                keys (train_contrastive.py contrastive_mode=
#                                joint output, e.g. a Phase 6/13C/13D tower
#                                checkpoint) -- a bare Stage-0 LM-only
#                                checkpoint has no projection head and will
#                                fail load_encoder()'s missing-key check.
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
# 2026-07-26: the default pointed at ${SCRATCH_ROOT}/mimic_cxr_cache, which is
# EMPTY — the populated caches live under /sc/home/$USER/dataset/. Combined with
# the missing offline exports below this meant a retrieval eval went online and
# 401'd on the gated MIMIC repo (the same failure that killed job 2357924).
# Default per dataset now, still overridable.
if [ "${DATASET}" = "indiana" ]; then
  EVAL_CACHE_DIR="${EVAL_CACHE_DIR:-/sc/home/$USER/dataset/indiana_cxr_cache}"
else
  EVAL_CACHE_DIR="${EVAL_CACHE_DIR:-/sc/home/$USER/dataset/mimic_cxr_cache}"
fi
OUTPUT_DIR="${OUTPUT_DIR:-$(dirname "${CKPT}")/eval_results}"
# Phase 8: point retrieval eval at the local PhysioNet build instead of the
# gated itsanmolgupta HF mirror. Empty by default so an unmodified invocation
# stays on the legacy mirror (the Arm-0 reproduction control needs this).
LOCAL_PARQUET_DIR="${LOCAL_PARQUET_DIR:-}"
MIMIC_SPLIT="${MIMIC_SPLIT:-test}"

echo "=== H100 eval (MODE=${MODE}) ==="
date; hostname
mkdir -p logs "${OUTPUT_DIR}"

cd "${SLURM_SUBMIT_DIR}/hybrid_model_mamba_xlstm" 2>/dev/null || cd "${SLURM_SUBMIT_DIR:-.}"

export HF_HOME="${SCRATCH_ROOT}/.hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
# MIMIC-CXR is a GATED repo: any online load_dataset 401s with
# DatasetNotFoundError. The header comment has said "run with
# HF_DATASETS_OFFLINE=1" since this script was written, but it never actually
# set it — so every retrieval eval depended on the caller remembering. Bake it
# in, matching train_biomedclip_kd_h100.sh. Override to 0 to (re)download.
# MODE=sts pulls PUBLIC HF datasets (glue/stsb, BIOSSES) that need real network
# access, unlike the gated mimic-cxr repo MODE=retrieval defaults offline to
# avoid -- default sts to online, everything else keeps the existing offline
# default so this change cannot silently affect ppl/retrieval behaviour.
if [ "${MODE}" = "sts" ]; then
  export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}"
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
else
  export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
fi
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
  LOCAL_ARGS=()
  if [ -n "${LOCAL_PARQUET_DIR}" ]; then
    LOCAL_ARGS+=(--local-parquet-dir "${LOCAL_PARQUET_DIR}" --mimic-split "${MIMIC_SPLIT}")
  fi
  python scripts/evaluate_cxr_retrieval.py \
    --checkpoint "${CKPT}" \
    --dataset "${DATASET}" \
    --cache-dir "${EVAL_CACHE_DIR}" \
    --batch-size 32 \
    --output-dir "${OUTPUT_DIR}" \
    ${LOCAL_ARGS[@]+"${LOCAL_ARGS[@]}"}
elif [ "${MODE}" = "sts" ]; then
  python scripts/evaluate_sts.py \
    --checkpoint "${CKPT}" \
    --datasets all \
    --output-dir "${OUTPUT_DIR}"
else
  echo "ERROR: unknown MODE='${MODE}' (expected 'ppl', 'retrieval', or 'sts')"; exit 1
fi

echo "=== END: results in ${OUTPUT_DIR} ==="
date
