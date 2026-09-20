#!/bin/bash
# ============================================================================
# Efficiency curves on H100 — latency / throughput / peak-memory vs sequence
# length, with fitted log-log scaling exponents.
#
# Runs the hybrid backbone against BOTH single-family baselines at identical
# dim/depth so the only difference is layer_pattern. Mamba (selective SSM) and
# mLSTM (TFLA) are both linear in sequence length, so the expected latency and
# memory exponents are ~1.0; softmax attention would be ~2.0 for latency.
#
# The sweep runs well past max_position_embeddings (1024) on purpose. That is
# valid because HybridLanguageModel sets use_pos_embedding=False (hybrid_lm.py:43)
# — there is no absolute position table to index out of. Pinned by
# tests/test_willi_parity.py::test_sequence_sweep_is_valid_past_max_position_embeddings.
#
# No dataset and no checkpoint: inputs are random token ids, weights are freshly
# initialised. Throughput and memory do not depend on weight values, so this is
# a pure architecture measurement and needs no HF cache or gated repo access.
#
# ENV: SCALE (70m|150m), SEQ_LENGTHS, BATCH_SIZE, DTYPE, ITERS, OUTPUT_DIR,
#      SCRATCH_ROOT, VENV_ACTIVATE.
# ============================================================================
#SBATCH --partition=aisc-batch
#SBATCH --account=aisc
#SBATCH --gpus=1
#SBATCH --exclude=ga03,gx17v1,gx13v1   # gx13v1: faulty GPU (cudaErrorContained, 2026-07-19)
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --job-name=h100_effcurve
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log

set -euo pipefail

SCRATCH_ROOT="${SCRATCH_ROOT:-/sc/scratch/$USER/hybrid_xmamba_h100}"
VENV_ACTIVATE="${VENV_ACTIVATE:-.venv/bin/activate}"
SCALE="${SCALE:-150m}"
BATCH_SIZE="${BATCH_SIZE:-4}"
DTYPE="${DTYPE:-bf16}"
ITERS="${ITERS:-10}"
# Past the 1024 training context on purpose — that is the point of the curve.
SEQ_LENGTHS="${SEQ_LENGTHS:-256 512 1024 2048 4096 8192 16384}"

# Phase 14A-7: MODELS is env-overridable so the parameter-matched Transformer
# baseline can join the sweep under the IDENTICAL protocol that produced
# analysis/efficiency_150m/ -- same seq lengths, batch, dtype, iterations.
# This is what finally replaces the writeup's standing §3 caveat ("there is no
# attention/transformer baseline in this repo ... the '~2.0 = quadratic
# attention' reference line is a cited comparison, not a measurement made
# here") with an actual measurement.
#   MODELS="hybrid_150m_v2 mamba_150m_baseline xlstm_150m_baseline transformer_150m_baseline" \
#     OUTPUT_DIR=analysis/efficiency_150m_with_transformer \
#     sbatch scripts/profile_efficiency_h100.sh
if [ -z "${MODELS:-}" ]; then
  if [ "${SCALE}" = "70m" ]; then
    MODELS="hybrid_70m_v2 mamba_70m_baseline xlstm_70m_baseline"
  else
    MODELS="hybrid_150m_v2 mamba_150m_baseline xlstm_150m_baseline"
  fi
fi

OUTPUT_DIR="${OUTPUT_DIR:-analysis/efficiency_${SCALE}}"

echo "=== H100 efficiency curves (SCALE=${SCALE}, dtype=${DTYPE}) ==="
date; hostname
mkdir -p logs "${OUTPUT_DIR}"

cd "${SLURM_SUBMIT_DIR}/hybrid_model_mamba_xlstm" 2>/dev/null || cd "${SLURM_SUBMIT_DIR:-.}"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONUNBUFFERED=1   # flush live (else block-buffered -> looks frozen)

source "${VENV_ACTIVATE}"
python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print('GPU:', torch.cuda.get_device_name(0))"

echo ""
echo "########## INFERENCE (forward only) ##########"
python scripts/performance_profile.py --sweep \
  --models ${MODELS} \
  --seq-lengths ${SEQ_LENGTHS} \
  --batch_size "${BATCH_SIZE}" \
  --num_iterations "${ITERS}" \
  --dtype "${DTYPE}" \
  --output-dir "${OUTPUT_DIR}/inference"

echo ""
echo "########## TRAINING STEP (forward + backward) ##########"
# Activations dominate here, so OOM arrives earlier than in inference. Points
# that do not fit are recorded as oom=True rather than killing the sweep.
python scripts/performance_profile.py --sweep --backward \
  --models ${MODELS} \
  --seq-lengths ${SEQ_LENGTHS} \
  --batch_size "${BATCH_SIZE}" \
  --num_iterations "${ITERS}" \
  --dtype "${DTYPE}" \
  --output-dir "${OUTPUT_DIR}/training"

# --- MAMBA3_PLAN_V2.md V3-F: per-token decode, cached vs full recompute -----------------------
# The two sweeps above time full-sequence forwards, which is the one thing autoregressive
# generation never does. This is where the O(L^2) -> O(L) claim is measured. Off by default so the
# 14A-7 protocol above is reproduced byte-for-byte; the cached path needs `tfla_impl: exact`
# (M6 finding 1), so profile hybrid_150m_m3_rrg rather than hybrid_150m_m3. A model with no
# `step()` -- the Transformer -- simply reports the recompute row.
#   DECODE_CURVE=true MODELS="hybrid_150m_v2 hybrid_150m_m3_rrg transformer_150m_baseline" \
#     sbatch scripts/profile_efficiency_h100.sh
DECODE_CURVE="${DECODE_CURVE:-false}"
PROMPT_LEN="${PROMPT_LEN:-256}"
NEW_TOKENS="${NEW_TOKENS:-64}"
if [ "${DECODE_CURVE}" = "true" ]; then
  echo ""
  echo "########## DECODE (per token, cached vs full recompute) ##########"
  for m in ${MODELS}; do
    echo ""
    echo "--- ${m} (prompt ${PROMPT_LEN}, ${NEW_TOKENS} new tokens) ---"
    python scripts/performance_profile.py --decode \
      --model "${m}" \
      --prompt-len "${PROMPT_LEN}" \
      --new-tokens "${NEW_TOKENS}" \
      --batch_size 1 \
      --dtype "${DTYPE}"
  done
fi

echo "=== END: curves in ${OUTPUT_DIR}/{inference,training}/efficiency_curves.{csv,json} ==="
date
