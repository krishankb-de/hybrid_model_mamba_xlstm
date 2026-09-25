#!/bin/bash
# ============================================================================
# EFFICIENCY_PLAN.md E1-E — re-measure the headline under a protocol that the
# last two jobs proved is necessary, and finish the training reference.
#
# WHY THIS EXISTS. Job 2582482 resolved the 2580198 anomaly and, in doing so,
# invalidated the protocol that produced it:
#
#   * chunk_size DID reach the operator (both arms printed their own value), so
#     it was never a plumbing bug.
#   * With a per-arm Inductor cache the SAME points came out 21-30% FASTER at
#     L=4096 (45.21 -> 35.60 ms at cs=64, 45.25 -> 31.79 at cs=128). A shared
#     cache had served both arms one slow kernel -- which is exactly why they
#     matched to a microsecond.
#   * Within a single process, shapes compiled LATER measure worse: in 2582482
#     the compiled training arm ran 3.04x at L=512 and 1.02x by L=4096, and in
#     2580198 L=4096 was the fifth shape compiled and came out slowest.
#
# Every compiled number this project has is therefore a function of what else
# was compiled beside it. The 1.34x-vs-FlashAttention headline at L=16384 came
# from a shared-cache run and has to be re-measured before it is written down.
#
# THE PROTOCOL. One sequence length per PROCESS, each with its own Inductor
# cache. Slower to run, but it is the only way a compiled number means anything
# on its own. Ordered by value so a third timeout still leaves the headline.
#
# ENV: MODEL, INFER_LENS, TRAIN_LENS, CHUNK, BATCH_SIZE, DTYPE, ITERS,
#      TRAIN_ITERS, OUTPUT_DIR, SKIP_INFER, SKIP_TRAIN, SCRATCH_ROOT,
#      VENV_ACTIVATE.
# ============================================================================
#SBATCH --partition=aisc-batch
#SBATCH --account=aisc
#SBATCH --qos=aisc
#SBATCH --gpus=1
#SBATCH --exclude=ga03,gx17v1,gx13v1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --job-name=h100_e1_confirm
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log
#SBATCH --open-mode=append

set -euo pipefail

SCRATCH_ROOT="${SCRATCH_ROOT:-/sc/scratch/$USER/hybrid_xmamba_h100}"
VENV_ACTIVATE="${VENV_ACTIVATE:-.venv/bin/activate}"
MODEL="${MODEL:-hybrid_150m_m3}"
BATCH_SIZE="${BATCH_SIZE:-4}"
DTYPE="${DTYPE:-bf16}"
ITERS="${ITERS:-10}"
TRAIN_ITERS="${TRAIN_ITERS:-5}"
# The headline length first, then the one below it for the crossover slope.
INFER_LENS="${INFER_LENS:-16384 8192}"
# L=2048 is the published training comparison point. 8192 is dropped: 54 GB and
# 1.3 s per iteration bought nothing the comparison needs, and it is what ran the
# clock out in job 2582482.
TRAIN_LENS="${TRAIN_LENS:-2048 1024}"
CHUNK="${CHUNK:-128}"
OUTPUT_DIR="${OUTPUT_DIR:-analysis/efficiency_e1_confirm}"
SKIP_INFER="${SKIP_INFER:-false}"
SKIP_TRAIN="${SKIP_TRAIN:-false}"

echo "=== E1-E: one shape per process, one cache per shape ==="
date; hostname
mkdir -p logs "${OUTPUT_DIR}"

cd "${SLURM_SUBMIT_DIR}/hybrid_model_mamba_xlstm" 2>/dev/null || cd "${SLURM_SUBMIT_DIR:-.}"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1

source "${VENV_ACTIVATE}"
python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print('GPU:', torch.cuda.get_device_name(0), '| torch', torch.__version__)"

# One point, one process, one cache. The cache is removed first so a requeue
# cannot inherit a kernel compiled beside something else.
point () {   # arm_name, model, seq_len, iters, extra flags...
  local arm="$1" mdl="$2" len="$3" it="$4"; shift 4
  export TORCHINDUCTOR_CACHE_DIR="${SCRATCH_ROOT}/ind_${arm}_${len}"
  rm -rf "${TORCHINDUCTOR_CACHE_DIR}"; mkdir -p "${TORCHINDUCTOR_CACHE_DIR}"
  echo ""
  echo "--- ${arm} @ L=${len} (${mdl}) ---"
  python scripts/performance_profile.py --sweep \
    --models "${mdl}" \
    --seq-lengths "${len}" \
    --batch_size "${BATCH_SIZE}" \
    --num_iterations "${it}" \
    --dtype "${DTYPE}" \
    --output-dir "${OUTPUT_DIR}/${arm}_L${len}" \
    "$@"
}

if [ "${SKIP_INFER}" != "true" ]; then
  echo ""
  echo "########## 1: the headline, re-measured one shape at a time ##########"
  for L in ${INFER_LENS}; do
    point xfmr          transformer_150m_baseline "${L}" "${ITERS}"
    point base          "${MODEL}" "${L}" "${ITERS}"
    point chunk         "${MODEL}" "${L}" "${ITERS}" --chunk-size "${CHUNK}"
    point comp          "${MODEL}" "${L}" "${ITERS}" --compile
    point comp_chunk    "${MODEL}" "${L}" "${ITERS}" --compile --chunk-size "${CHUNK}"
  done
fi

if [ "${SKIP_TRAIN}" != "true" ]; then
  echo ""
  echo "########## 2: the training reference that 2582482 never reached ##########"
  echo "The Transformer row runs FIRST -- without it the training numbers have"
  echo "nothing to be compared against, which is the state they are in today."
  for L in ${TRAIN_LENS}; do
    point train_xfmr       transformer_150m_baseline "${L}" "${TRAIN_ITERS}" --backward
    point train_base       "${MODEL}" "${L}" "${TRAIN_ITERS}" --backward
    point train_chunk      "${MODEL}" "${L}" "${TRAIN_ITERS}" --backward --chunk-size "${CHUNK}"
    point train_comp       "${MODEL}" "${L}" "${TRAIN_ITERS}" --backward --compile
    point train_comp_chunk "${MODEL}" "${L}" "${TRAIN_ITERS}" --backward --compile --chunk-size "${CHUNK}"
  done
fi

echo ""
echo "=== END ==="
echo "Each ${OUTPUT_DIR}/<arm>_L<len>/efficiency_curves.csv holds exactly one point,"
echo "measured in its own process with its own compiler cache. Check"
echo "effective_chunk_size in every row before believing any chunk_size claim."
date
