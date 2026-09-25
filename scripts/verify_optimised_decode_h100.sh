#!/bin/bash
# ============================================================================
# EFFICIENCY_PLAN.md E6 — does the optimised inference configuration produce the
# same reports?
#
# THE GAP THIS CLOSES. Efficiency is reported at mamba3_chunk_size=128 with
# torch.compile. Every quality number in analysis/mamba3_results.md was decoded
# at chunk_size=64, uncompiled. The R1 gate showed the logits agree to 3.0e-05 --
# which is NOT the same as showing the decoded tokens agree. Beam search can flip
# on an arbitrarily small margin, and this project has already watched that
# happen: V5-A changed 227 of 400 reports under an operator swap that moved no
# metric. So the optimised configuration gets decoded, not assumed.
#
# PROTOCOL: identical to V5-A (jobs 2561504/2561505) so the two are comparable --
# 13D, official test split, first NUM_SAMPLES studies, beam 3 -- with exactly one
# thing changed, CHUNK_SIZE. The eval announces the override in its log the way
# scan_impl does.
#
# PRE-REGISTERED RULE (written before the run, EFFICIENCY_PLAN.md E6):
#   * metrics within the paired-bootstrap CI  -> the optimised configuration is
#     reportable as THE efficiency configuration.
#   * reports change but metrics tie          -> say exactly that. It is the V5-A
#     result and it is honest.
#   * metrics move                            -> the efficiency claim reverts to
#     the DEFAULT configuration (562.22 ms / 7.090 GB at L=16384, which is 3.4x
#     SLOWER than the Transformer) and the compiled number is quoted as headroom.
#
# ENV: CHECKPOINT, MODEL_CONFIG, PARQUET, NUM_SAMPLES, CHUNK_SIZE, REF_DUMP,
#      DUMP_DIR, SWEEP_ARM, SCRATCH_ROOT, VENV_ACTIVATE.
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
#SBATCH --job-name=h100_e6_optdecode
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log
#SBATCH --open-mode=append

set -euo pipefail

SCRATCH_ROOT="${SCRATCH_ROOT:-/sc/scratch/$USER/hybrid_xmamba_h100}"
VENV_ACTIVATE="${VENV_ACTIVATE:-.venv/bin/activate}"
# ⚠ FIXED 2026-09-25 after job 2583277. These defaulted to 13D / hybrid_150m_v2_rrg -- the
# INCUMBENT, which has no mamba3 layer -- so mamba3_chunk_size was set and never read, and the
# run measured the unmodified model. The efficiency numbers are all hybrid_150m_m3, so the
# quality check has to be the Mamba-3 decoder too. evaluate_report_generation.py now refuses a
# chunk_size override on a config with no mamba3 layer, so this cannot recur silently.
CHECKPOINT="${CHECKPOINT:-./outputs/h100_report_gen_m3_tower13d_s42/checkpoints/last.ckpt}"
MODEL_CONFIG="${MODEL_CONFIG:-hybrid_150m_m3_rrg}"
PARQUET="${PARQUET:-/sc/home/$USER/dataset/mimic_full/test.parquet}"
NUM_SAMPLES="${NUM_SAMPLES:-400}"
CHUNK_SIZE="${CHUNK_SIZE:-128}"
DUMP_DIR="${DUMP_DIR:-results/report_gen_m3_s42_chunk${CHUNK_SIZE}_n${NUM_SAMPLES}}"
# The control arm is decoded IN THIS JOB rather than paired against an existing dump, so the
# two differ by exactly one setting and no assumption about study ordering is needed.
REF_DUMP="${REF_DUMP:-results/report_gen_m3_s42_chunk64_n${NUM_SAMPLES}}"
SWEEP_ARM="${SWEEP_ARM:-true}"

echo "=== E6: does chunk_size=${CHUNK_SIZE} change the reports? ==="
date; hostname
mkdir -p logs "${DUMP_DIR}"

cd "${SLURM_SUBMIT_DIR}/hybrid_model_mamba_xlstm" 2>/dev/null || cd "${SLURM_SUBMIT_DIR:-.}"

decode_arm () {   # dump_dir, chunk_size
  local dump="$1" cs="$2"
  echo ""
  echo "--- decoding ${NUM_SAMPLES} studies at chunk_size=${cs} -> ${dump} ---"
  mkdir -p "${dump}"
  CHECKPOINT="${CHECKPOINT}" \
  MODEL_CONFIG="${MODEL_CONFIG}" \
  PARQUET="${PARQUET}" \
  NUM_SAMPLES="${NUM_SAMPLES}" \
  DECODE=beam BEAM_SIZE=3 MAX_NEW_TOKENS=100 \
  CHUNK_SIZE="${cs}" \
  DUMP_DIR="${dump}" \
  bash scripts/inspect_report_generation_h100.sh
}

echo ""
echo "########## E6-A: both arms, same job, one setting apart ##########"
echo "Watch for '[operator] mamba3_chunk_size: 64 -> N'. If the log instead says"
echo "'TRAINED with None' the config has no mamba3 layer and the run measures nothing;"
echo "the eval now raises rather than letting that through (job 2583277)."
decode_arm "${REF_DUMP}" 64
decode_arm "${DUMP_DIR}" "${CHUNK_SIZE}"

echo ""
echo "--- how many of the ${NUM_SAMPLES} reports changed textually ---"
if [ -f "${REF_DUMP}/hyps.txt" ] && [ -f "${DUMP_DIR}/hyps.txt" ]; then
  CHANGED=$(awk 'NR==FNR{a[FNR]=$0;next}{if(a[FNR]!=$0)c++}END{print c+0}' \
            "${REF_DUMP}/hyps.txt" "${DUMP_DIR}/hyps.txt")
  echo "${CHANGED} of ${NUM_SAMPLES} generated reports differ between chunk_size 64 and ${CHUNK_SIZE}."
  echo "(Reports changing while metrics tie is a RESULT -- V5-A found exactly that.)"
fi

if [ "${SWEEP_ARM}" = "true" ]; then
  echo ""
  echo "########## E6-D: chunk_size under COMPILE (E0-C swept it uncompiled only) ##########"
  echo "The free version of the dropped E2: if Inductor shifts the balance between"
  echo "loop overhead and mask work, the optimum may not still be 128."
  source "${VENV_ACTIVATE}"
  for cs in 128 256 512; do
    export TORCHINDUCTOR_CACHE_DIR="${SCRATCH_ROOT}/ind_e6_cs${cs}"
    rm -rf "${TORCHINDUCTOR_CACHE_DIR}"; mkdir -p "${TORCHINDUCTOR_CACHE_DIR}"
    echo ""
    echo "--- compiled, chunk_size=${cs}, L=16384 ---"
    # Inductor failed to generate code for cs=256 at L=16384 in job 2583277
    # ("TypeError: list indices must be integers or slices, not NoneType" inside the
    # SplitScan cumsum codegen), which killed the whole job under `set -e`. A compiler
    # that cannot build an arm is DATA about that arm, not a reason to lose the others.
    if ! python scripts/performance_profile.py --sweep \
      --models hybrid_150m_m3 --seq-lengths 16384 --batch_size 4 \
      --num_iterations 10 --dtype bf16 --compile --chunk-size "${cs}" \
      --output-dir "analysis/efficiency_e6/compiled_cs${cs}"; then
      echo "ARM FAILED: torch.compile could not build chunk_size=${cs} at L=16384."
      echo "Recording as a compiler limitation and continuing."
    fi
  done
fi

echo ""
echo "=== NEXT, on the login node (this job does not run them) ==="
cat <<NEXT
  # 1. CheXbert on the new dump (separate venv):
  DUMP_DIR=${DUMP_DIR} sbatch scripts/score_chexbert_h100.sh

  # 2. Paired bootstrap against the chunk_size=64 dump:
  A=${REF_DUMP} B=${DUMP_DIR} \\
    NAME_A=chunk64 NAME_B=chunk${CHUNK_SIZE} PER_LABEL=true \\
    OUTPUT=analysis/bootstrap_chunk64_vs_chunk${CHUNK_SIZE}.md \\
    sbatch scripts/bootstrap_compare_h100.sh

  # 3. How many of the ${NUM_SAMPLES} reports changed textually at all:
  #    (run in the venv, locally or via srun -- never bare on lx01)
  #    diff <(cat ${REF_DUMP}/hyps.txt) <(cat ${DUMP_DIR}/hyps.txt) | grep -c '^<'
NEXT
echo ""
echo "Reminder: the pre-registered rule is in EFFICIENCY_PLAN.md E6. Reports changing"
echo "while metrics tie is a RESULT, not a failure -- V5-A found exactly that."
date
