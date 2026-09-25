#!/bin/bash
# ============================================================================
# EFFICIENCY_PLAN.md E1 — adopt-or-reject the two wins E0 found.
#
# E0 measured `mamba3_chunk_size=128` at 1.40x and `torch.compile` at 4.28x,
# cutting the gap to FlashAttention from 5.35x to 1.25x at L=2048. Neither is
# adoptable on a stopwatch. This job is the gate plus the clean comparison.
#
#   STEP 1 (gate)  rule R1: every variant must compute the same function as the
#                  shipped operator -- against the fp64 oracle, and across a
#                  cu_seqlens document boundary that falls inside a chunk. If it
#                  fails, the job STOPS and no timing is produced, because a
#                  faster operator that changes a decoded token would invalidate
#                  every published metric. This repo has shipped a wrong
#                  operator before; that is why the gate runs first.
#   STEP 2 (arms)  all five arms under the IDENTICAL --sweep protocol in ONE
#                  job, so they are comparable to each other and to the
#                  published 14A-7 numbers without cross-job node variance.
#
# Job 2579642 died at its 1.5h limit while compiling L=16384 (FE4: Dynamo
# unrolls the nc-long loop, so graph build grows with nc). Two mitigations:
# --time=03:00:00, and a persistent Inductor cache in scratch so a re-run does
# not pay the same graph build twice. The risky L=16384 compile runs LAST, so a
# second timeout still leaves every other arm complete.
#
# ENV: MODEL, SEQ_LENGTHS, LONG_LEN, CHUNK, BATCH_SIZE, DTYPE, ITERS,
#      OUTPUT_DIR, SKIP_GATE, SCRATCH_ROOT, VENV_ACTIVATE.
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
#SBATCH --job-name=h100_e1_verify
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
# The ladder minus the top point; L=16384 is run separately and last because it
# is the one that has already timed out once.
SEQ_LENGTHS="${SEQ_LENGTHS:-256 512 1024 2048 4096 8192}"
LONG_LEN="${LONG_LEN:-16384}"
CHUNK="${CHUNK:-128}"          # E0-C's measured optimum
OUTPUT_DIR="${OUTPUT_DIR:-analysis/efficiency_e1}"
SKIP_GATE="${SKIP_GATE:-false}"

echo "=== E1: R1 gate, then adopt-or-reject chunk_size=${CHUNK} and torch.compile ==="
date; hostname
mkdir -p logs "${OUTPUT_DIR}"

cd "${SLURM_SUBMIT_DIR}/hybrid_model_mamba_xlstm" 2>/dev/null || cd "${SLURM_SUBMIT_DIR:-.}"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
# Persist Inductor's compiled artifacts so a requeue or a re-run does not pay the
# same graph build again. This is the mitigation for the 2579642 timeout.
export TORCHINDUCTOR_CACHE_DIR="${SCRATCH_ROOT}/inductor_cache"
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}"

source "${VENV_ACTIVATE}"
python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print('GPU:', torch.cuda.get_device_name(0), '| torch', torch.__version__)"

if [ "${SKIP_GATE}" != "true" ]; then
  echo ""
  echo "########## STEP 1: rule R1 equivalence gate ##########"
  echo "A failure here stops the job. That is the point."
  python scripts/check_operator_equivalence.py \
    --device cuda \
    --model "${MODEL}" \
    --chunk-sizes "${CHUNK}" 256 512 \
    --seq-length 1024 \
    --compile
  echo "R1 gate passed -- timing arms may proceed."
fi

run_arm () {   # name, extra flags...
  local name="$1"; shift
  echo ""
  echo "--- arm: ${name} ---"
  python scripts/performance_profile.py --sweep \
    --models "${MODEL}" \
    --batch_size "${BATCH_SIZE}" \
    --num_iterations "${ITERS}" \
    --dtype "${DTYPE}" \
    --output-dir "${OUTPUT_DIR}/${name}" \
    "$@"
}

echo ""
echo "########## STEP 2: the five arms, one protocol ##########"

# 1. The control. Reproduces the published 14A-7 m3 numbers in THIS job, which is
#    what makes every ratio below a within-job comparison.
run_arm baseline            --seq-lengths ${SEQ_LENGTHS} ${LONG_LEN}

# 2. chunk_size alone (E1-A).
run_arm chunk${CHUNK}       --seq-lengths ${SEQ_LENGTHS} ${LONG_LEN} --chunk-size "${CHUNK}"

# 3. compile alone (E1-B), up to the length already known to build.
run_arm compiled            --seq-lengths ${SEQ_LENGTHS} --compile

# 4. both together -- untested whether they stack or collide.
run_arm compiled_chunk${CHUNK} --seq-lengths ${SEQ_LENGTHS} --compile --chunk-size "${CHUNK}"

# 5. The risky one, deliberately last: if Dynamo blows up again, everything above
#    is already written to disk.
echo ""
echo "NOTE: the arms below are the ones that timed out in job 2579642."
run_arm compiled_long       --seq-lengths "${LONG_LEN}" --compile
run_arm compiled_chunk${CHUNK}_long --seq-lengths "${LONG_LEN}" --compile --chunk-size "${CHUNK}"

echo ""
echo "=== END ==="
echo "Compare ${OUTPUT_DIR}/{baseline,chunk${CHUNK},compiled,compiled_chunk${CHUNK}}/efficiency_curves.csv."
echo "Rule R3: a variant is adopted only if >=1.25x at >=2 lengths and nowhere >5% slower."
echo "Rule R2: reject any arm whose peak_memory_gb exceeds the Transformer's 7.152 GB at L=16384."
date
