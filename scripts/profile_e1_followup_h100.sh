#!/bin/bash
# ============================================================================
# EFFICIENCY_PLAN.md E1 follow-up — two loose ends from job 2580198.
#
# A) THE ANOMALY. In 2580198 the compiled cs=64 and cs=128 arms timed identically
#    at L=4096 (45.205 vs 45.250 ms) and L=8192 (88.802 vs 88.803 ms) -- one
#    microsecond apart on an 89 ms measurement -- while the UNCOMPILED arms at the
#    same lengths differed by 45%. That is not a plausible coincidence, and there
#    are two candidate explanations with opposite consequences:
#      1. the chunk_size override stopped reaching the operator under compile
#         (a plumbing bug, and the 1.14x seen at L=16384 would be the real effect
#         appearing only where the cache missed), or
#      2. those lengths genuinely plateau once Inductor removes the loop overhead.
#    Both arms shared one TORCHINDUCTOR_CACHE_DIR in 2580198, so a cache hit keyed
#    too loosely is a live possibility. This job gives every arm its OWN cache and
#    prints the chunk size read back off the built module, which separates the two.
#
# B) THE UNTESTED HALF. Every E0/E1 number so far is forward-only. The other
#    published efficiency figure is TRAINING: 8x slower than the Transformer at
#    L=2048. torch.compile has never been tried on that path. This measures it.
#    Still random weights and random ids -- no checkpoint is trained or touched.
#
# ENV: MODEL, ANOMALY_LENS, TRAIN_LENS, CHUNK, BATCH_SIZE, DTYPE, ITERS,
#      OUTPUT_DIR, SKIP_ANOMALY, SKIP_TRAIN, SCRATCH_ROOT, VENV_ACTIVATE.
# ============================================================================
#SBATCH --partition=aisc-batch
#SBATCH --account=aisc
#SBATCH --qos=aisc
#SBATCH --gpus=1
#SBATCH --exclude=ga03,gx17v1,gx13v1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --job-name=h100_e1_followup
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
ANOMALY_LENS="${ANOMALY_LENS:-4096 8192}"
# Training holds activations, so it OOMs earlier than inference. The published
# training comparison is at L=2048; 8192 is where the corrected model still fits.
TRAIN_LENS="${TRAIN_LENS:-512 1024 2048 4096 8192}"
CHUNK="${CHUNK:-128}"
OUTPUT_DIR="${OUTPUT_DIR:-analysis/efficiency_e1_followup}"
SKIP_ANOMALY="${SKIP_ANOMALY:-false}"
SKIP_TRAIN="${SKIP_TRAIN:-false}"

echo "=== E1 follow-up: chunk_size-under-compile anomaly + the training arm ==="
date; hostname
mkdir -p logs "${OUTPUT_DIR}"

cd "${SLURM_SUBMIT_DIR}/hybrid_model_mamba_xlstm" 2>/dev/null || cd "${SLURM_SUBMIT_DIR:-.}"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1

source "${VENV_ACTIVATE}"
python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print('GPU:', torch.cuda.get_device_name(0), '| torch', torch.__version__)"

# Every arm gets its own Inductor cache. This is the whole point of part A: a
# shared cache is the one mechanism that could make two different chunk sizes
# produce byte-identical kernels.
run_isolated () {   # name, extra flags...
  local name="$1"; shift
  export TORCHINDUCTOR_CACHE_DIR="${SCRATCH_ROOT}/inductor_cache_${name}"
  rm -rf "${TORCHINDUCTOR_CACHE_DIR}"
  mkdir -p "${TORCHINDUCTOR_CACHE_DIR}"
  echo ""
  echo "--- arm: ${name}   (isolated cache: ${TORCHINDUCTOR_CACHE_DIR}) ---"
  python scripts/performance_profile.py --sweep \
    --models "${ARM_MODEL:-${MODEL}}" \
    --batch_size "${BATCH_SIZE}" \
    --num_iterations "${ITERS}" \
    --dtype "${DTYPE}" \
    --output-dir "${OUTPUT_DIR}/${name}" \
    "$@"
}

if [ "${SKIP_ANOMALY}" != "true" ]; then
  echo ""
  echo "########## A: does chunk_size still reach the operator under compile? ##########"
  echo "Watch the 'effective chunk_size on the built module' line in each arm."
  echo "If both say what they were asked for AND still time identically, the plateau"
  echo "is real. If the 128 arm reports 64, it was a plumbing bug all along."
  run_isolated anomaly_compiled_cs64  --seq-lengths ${ANOMALY_LENS} --compile --chunk-size 64
  run_isolated anomaly_compiled_cs${CHUNK} --seq-lengths ${ANOMALY_LENS} --compile --chunk-size "${CHUNK}"
fi

if [ "${SKIP_TRAIN}" != "true" ]; then
  echo ""
  echo "########## B: the training step -- forward + backward ##########"
  echo "The published figure this speaks to is '8x slower than the Transformer at"
  echo "L=2048'. Nothing has ever tested compile on this path."
  run_isolated train_baseline --backward --seq-lengths ${TRAIN_LENS}
  run_isolated train_chunk${CHUNK} --backward --seq-lengths ${TRAIN_LENS} --chunk-size "${CHUNK}"
  run_isolated train_compiled --backward --seq-lengths ${TRAIN_LENS} --compile
  run_isolated train_compiled_chunk${CHUNK} --backward --seq-lengths ${TRAIN_LENS} --compile --chunk-size "${CHUNK}"
  echo ""
  echo "For the Transformer's training row, compare against the published 14A-7"
  echo "training sweep; re-run it here only if the node or torch version changed."
  ARM_MODEL=transformer_150m_baseline run_isolated train_transformer --backward --seq-lengths ${TRAIN_LENS}
fi

echo ""
echo "=== END ==="
echo "Results in ${OUTPUT_DIR}/*/efficiency_curves.csv (column effective_chunk_size is the"
echo "proof that the override reached the operator). No checkpoint was trained or touched."
date
