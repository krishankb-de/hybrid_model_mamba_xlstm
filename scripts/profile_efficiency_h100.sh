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

if [ "${SCALE}" = "70m" ]; then
  MODELS="hybrid_70m_v2 mamba_70m_baseline xlstm_70m_baseline"
else
  MODELS="hybrid_150m_v2 mamba_150m_baseline xlstm_150m_baseline"
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

echo "=== END: curves in ${OUTPUT_DIR}/{inference,training}/efficiency_curves.{csv,json} ==="
date
