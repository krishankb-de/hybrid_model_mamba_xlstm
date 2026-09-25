#!/bin/bash
# ============================================================================
# EFFICIENCY_PLAN.md E0 — where does the 4.14x wall-clock gap actually live?
#
# One job, four measurements, none of which touches a checkpoint or a published
# metric. Random weights and random token ids: throughput and memory do not
# depend on weight values, so this needs no HF cache and no MIMIC data.
#
#   E0-A  per-mixer-type split of the forward pass, and the Amdahl bound that
#         follows. 3 of the 12 layers are mLSTM on pure-PyTorch TFLA and no
#         Mamba kernel touches them; if they are 40% of the time then a PERFECT
#         SSD path caps at 2.5x and never reaches the 4.14x we need. That bound
#         gates E2/E3/E4, which is why this job runs before any kernel work.
#   E0-B  of the Mamba-3 time, how much is inside ssd_chunked_scan.
#   E0-C  mamba3_chunk_size sweep. nc = seqlen/chunk_size is the length of the
#         Python loop at ssd_interface.py:169-190. If latency falls ~linearly in
#         nc the path is launch-bound and E2 is worth its cost; if it is flat,
#         it is bandwidth-bound and E2 is not.
#   E0-D  the Transformer re-timed with the fused SDPA backends DISABLED. This
#         is the measurement that answers the supervisor: it separates the
#         algorithm from the hand-written kernel. Expect OOM at the top of the
#         ladder -- b=4, L=16384 needs ~26 GB for one bf16 attention matrix --
#         and that OOM is itself the FlashAttention memory story as data.
#
# Optional E1 arms (off by default; E0 is what gates everything):
#   CHUNK_ARM=true    run E0-C
#   COMPILE_ARM=true  run the torch.compile inference arm (E1-B)
#
# ENV: MODELS, SEQ_LENGTHS, BATCH_SIZE, DTYPE, ITERS, OUTPUT_DIR, CHUNK_SIZES,
#      CHUNK_AT_LEN, CHUNK_ARM, COMPILE_ARM, SCRATCH_ROOT, VENV_ACTIVATE.
# ============================================================================
#SBATCH --partition=aisc-batch
#SBATCH --account=aisc
#SBATCH --qos=aisc
#SBATCH --gpus=1
#SBATCH --exclude=ga03,gx17v1,gx13v1   # gx13v1: faulty GPU (cudaErrorContained, 2026-07-19)
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --requeue
#SBATCH --job-name=h100_layersplit
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log
#SBATCH --open-mode=append

set -euo pipefail

SCRATCH_ROOT="${SCRATCH_ROOT:-/sc/scratch/$USER/hybrid_xmamba_h100}"
VENV_ACTIVATE="${VENV_ACTIVATE:-.venv/bin/activate}"
BATCH_SIZE="${BATCH_SIZE:-4}"
DTYPE="${DTYPE:-bf16}"
ITERS="${ITERS:-10}"
# Identical ladder to the 14A-7 protocol so every number here is comparable to
# analysis/efficiency_150m_m3/ without re-running that sweep.
SEQ_LENGTHS="${SEQ_LENGTHS:-256 512 1024 2048 4096 8192 16384}"
MODELS="${MODELS:-hybrid_150m_m3 hybrid_150m_v2 transformer_150m_baseline}"
OUTPUT_DIR="${OUTPUT_DIR:-analysis/efficiency_layer_split}"
CHUNK_SIZES="${CHUNK_SIZES:-64 128 256 512}"
CHUNK_AT_LEN="${CHUNK_AT_LEN:-4096 16384}"
CHUNK_ARM="${CHUNK_ARM:-false}"
COMPILE_ARM="${COMPILE_ARM:-false}"

echo "=== E0: layer split + attention-backend control (dtype=${DTYPE}, bs=${BATCH_SIZE}) ==="
date; hostname
mkdir -p logs "${OUTPUT_DIR}"

cd "${SLURM_SUBMIT_DIR}/hybrid_model_mamba_xlstm" 2>/dev/null || cd "${SLURM_SUBMIT_DIR:-.}"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1

source "${VENV_ACTIVATE}"
python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print('GPU:', torch.cuda.get_device_name(0), '| torch', torch.__version__)"

echo ""
echo "########## E0-A / E0-B: per-layer-type split + ssd_chunked_scan share ##########"
python scripts/performance_profile.py --per-layer \
  --models ${MODELS} \
  --seq-lengths ${SEQ_LENGTHS} \
  --batch_size "${BATCH_SIZE}" \
  --num_iterations "${ITERS}" \
  --dtype "${DTYPE}" \
  --output-dir "${OUTPUT_DIR}/per_layer"

echo ""
echo "########## E0-D: Transformer, fused SDPA vs unfused math backend ##########"
echo "--- arm 1: auto (fused; this reproduces the published 14A-7 numbers) ---"
python scripts/performance_profile.py --sweep \
  --models transformer_150m_baseline \
  --seq-lengths ${SEQ_LENGTHS} \
  --batch_size "${BATCH_SIZE}" \
  --num_iterations "${ITERS}" \
  --dtype "${DTYPE}" \
  --attn-backend auto \
  --output-dir "${OUTPUT_DIR}/attn_auto"

echo ""
echo "--- arm 2: math (unfused reference; OOM at the top of the ladder is expected) ---"
python scripts/performance_profile.py --sweep \
  --models transformer_150m_baseline \
  --seq-lengths ${SEQ_LENGTHS} \
  --batch_size "${BATCH_SIZE}" \
  --num_iterations "${ITERS}" \
  --dtype "${DTYPE}" \
  --attn-backend math \
  --output-dir "${OUTPUT_DIR}/attn_math"

if [ "${CHUNK_ARM}" = "true" ]; then
  echo ""
  echo "########## E0-C: mamba3_chunk_size sweep (nc = seqlen / chunk_size) ##########"
  for cs in ${CHUNK_SIZES}; do
    echo ""
    echo "--- chunk_size=${cs} ---"
    python scripts/performance_profile.py --per-layer \
      --model hybrid_150m_m3 \
      --seq-lengths ${CHUNK_AT_LEN} \
      --batch_size "${BATCH_SIZE}" \
      --num_iterations "${ITERS}" \
      --dtype "${DTYPE}" \
      --chunk-size "${cs}" \
      --output-dir "${OUTPUT_DIR}/chunk_${cs}"
  done
fi

if [ "${COMPILE_ARM}" = "true" ]; then
  echo ""
  echo "########## E1-B: torch.compile inference arm ##########"
  echo "NOTE: Dynamo will try to unroll the nc-long Python loop in ssd_chunked_scan."
  echo "      A long compile for a small steady-state gain is a NULL under rule R3."
  python scripts/performance_profile.py --sweep \
    --models hybrid_150m_m3 \
    --seq-lengths ${SEQ_LENGTHS} \
    --batch_size "${BATCH_SIZE}" \
    --num_iterations "${ITERS}" \
    --dtype "${DTYPE}" \
    --compile \
    --output-dir "${OUTPUT_DIR}/compiled"
fi

echo ""
echo "=== END ==="
echo "Amdahl bound (E0-F) is printed per sequence length in the E0-A section above"
echo "and stored as amdahl_bound_if_ssd_free in ${OUTPUT_DIR}/per_layer/layer_split.json."
echo "Nothing in this job changed a checkpoint, a dump, or a published metric."
date
