#!/bin/bash
# ============================================================================
# EFFICIENCY_PLAN.md E7 — does torch.compile change what the model writes?
#
# THE GAP THIS CLOSES. The efficiency headline (1.34x faster than FlashAttention
# at 16,384 tokens) uses TWO levers: mamba3_chunk_size 64->128 and torch.compile.
# E6 settled the first one on text -- 0 of 400 decoded reports differ. The second
# was only ever checked on LOGITS: 3.0e-05 against a 1e-4 gate. Logit closeness
# is not token identity. Beam search flips on an arbitrarily small margin, and
# this project has watched exactly that happen: V5-A changed 227 of 400 reports
# under an operator swap that moved no metric. So compile gets decoded too, and
# "verified by analogy" becomes "verified".
#
# PROTOCOL. Identical to E6 (job 2583455) so the two are directly comparable:
# the Mamba-3 decoder at chunk_size=128, official test split, first NUM_SAMPLES
# studies, beam 3, both arms in ONE job differing by exactly one setting --
# torch.compile on or off. No pairing against an older dump, no assumption about
# study ordering.
#
# ORDERING IS DELIBERATE: the COMPILED arm runs first. Its cost is the unknown
# one. If the job dies late, the arm that survives is the one we do not already
# have (the eager chunk_size=128 dump exists from job 2583455).
#
# THE FAILURE MODE THIS IS BUILT AGAINST. torch.compile fails OPEN -- a capture
# failure or an exhausted recompile limit silently serves eager, and both arms
# would then agree for the trivial reason that neither compiled. The eval reads
# Dynamo's counters back after decoding and ABORTS if nothing was captured, so
# that cannot be reported as agreement. Same class of bug as job 2583277.
#
# PRE-REGISTERED RULE (written before the run, EFFICIENCY_PLAN.md E7):
#   * 0 reports differ        -> compile is verified on text, not by analogy.
#                                The caveat comes out of EFFICIENCY_NOTE.md 5b.
#   * reports differ, metrics tie within the paired bootstrap -> say exactly
#                                that. It is the V5-A result and it is honest;
#                                the efficiency configuration stays reportable
#                                with the text change stated.
#   * a metric moves          -> the speed claim reverts to what is verified on
#                                text. chunk_size=128 uncompiled is 402.96 ms at
#                                L=16,384, which LOSES to the Transformer's
#                                164.85 ms, so the honest headline would become
#                                memory parity plus the scaling exponent, and
#                                the compiled number is quoted as headroom.
#
# ENV: CHECKPOINT, MODEL_CONFIG, PARQUET, NUM_SAMPLES, CHUNK_SIZE, DUMP_DIR,
#      REF_DUMP, SKIP_CANARY, TIME_BUDGET_S, SCRATCH_ROOT, VENV_ACTIVATE.
# ============================================================================
#SBATCH --partition=aisc-batch
#SBATCH --account=aisc
#SBATCH --qos=aisc
#SBATCH --gpus=1
#SBATCH --exclude=ga03,gx17v1,gx13v1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --job-name=h100_e7_compiled_decode
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log
#SBATCH --open-mode=append

set -euo pipefail

SCRATCH_ROOT="${SCRATCH_ROOT:-/sc/scratch/$USER/hybrid_xmamba_h100}"
VENV_ACTIVATE="${VENV_ACTIVATE:-.venv/bin/activate}"
# The Mamba-3 decoder, NOT the incumbent -- a chunk_size override on a config
# with no mamba3 layer is a silent no-op, which is what invalidated job 2583277.
# The eval now refuses that outright, but the defaults should be right anyway.
CHECKPOINT="${CHECKPOINT:-./outputs/h100_report_gen_m3_tower13d_s42/checkpoints/last.ckpt}"
MODEL_CONFIG="${MODEL_CONFIG:-hybrid_150m_m3_rrg}"
PARQUET="${PARQUET:-/sc/home/$USER/dataset/mimic_full/test.parquet}"
NUM_SAMPLES="${NUM_SAMPLES:-400}"
CHUNK_SIZE="${CHUNK_SIZE:-128}"
DUMP_DIR="${DUMP_DIR:-results/report_gen_m3_s42_chunk${CHUNK_SIZE}_compiled_n${NUM_SAMPLES}}"
REF_DUMP="${REF_DUMP:-results/report_gen_m3_s42_chunk${CHUNK_SIZE}_eager_n${NUM_SAMPLES}}"
SKIP_CANARY="${SKIP_CANARY:-false}"
# Seconds allowed for the compiled arm. The default leaves ~2 h of the 8 h wall
# for the eager arm (which measured 43 min for 400 studies in job 2583455).
TIME_BUDGET_S="${TIME_BUDGET_S:-21600}"

echo "=== E7: does torch.compile change the decoded reports? ==="
date; hostname
mkdir -p logs

cd "${SLURM_SUBMIT_DIR}/hybrid_model_mamba_xlstm" 2>/dev/null || cd "${SLURM_SUBMIT_DIR:-.}"

# One persistent Inductor cache for every compiled invocation in this job. The
# E1 lesson was that sharing a cache across arms with DIFFERENT configs corrupts
# them; here every compiled invocation is the same configuration, and nothing
# timed is being reported -- this run measures text. Sharing only means the full
# arm does not re-pay the canary's graph build, which makes the projection below
# conservative rather than optimistic.
export TORCHINDUCTOR_CACHE_DIR="${SCRATCH_ROOT}/ind_e7_cs${CHUNK_SIZE}"
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}"

decode_arm () {   # dump_dir (may be empty = no dump), n_samples, compile
  local dump="$1" n="$2" comp="$3"
  [ -n "${dump}" ] && mkdir -p "${dump}"
  CHECKPOINT="${CHECKPOINT}" \
  MODEL_CONFIG="${MODEL_CONFIG}" \
  PARQUET="${PARQUET}" \
  NUM_SAMPLES="${n}" \
  DECODE=beam BEAM_SIZE=3 MAX_NEW_TOKENS=100 \
  CHUNK_SIZE="${CHUNK_SIZE}" \
  COMPILE="${comp}" \
  DUMP_DIR="${dump}" \
  bash scripts/inspect_report_generation_h100.sh
}

# ---------------------------------------------------------------------------
# Canary. Compiled beam-search decode has never been run in this project and its
# per-sample cost is unknown -- it could be faster than eager (fused kernels) or
# much slower (guard overhead on ~100 growing shapes). Two points give a slope,
# which separates the one-off graph build from the per-study cost; a single point
# cannot, and would have to be read pessimistically enough to abort a run that
# would have finished.
#
# BOTH TIMED POINTS RUN WITH A WARM INDUCTOR CACHE. A throwaway n=2 run pays the
# cold graph build first. Without it the n=4 point carries a cold compile the
# n=20 point does not, the fitted slope comes out BELOW the true per-study cost,
# and the projection is optimistic -- precisely the direction that loses 8 GPU
# hours. ~12 minutes to avoid that.
# ---------------------------------------------------------------------------
CANARY_LOG="${SCRATCH_ROOT}/e7_canary_$$.log"
run_canary () {   # n -> elapsed seconds on stdout, aborts the job on failure
  local n="$1" t0 t1
  t0=$(date +%s)
  if ! decode_arm "" "${n}" true > "${CANARY_LOG}" 2>&1; then
    echo "CANARY FAILED at n=${n}. The compiled decode does not run; error follows." >&2
    tail -n 60 "${CANARY_LOG}" >&2
    exit 1
  fi
  t1=$(date +%s)
  echo $((t1-t0))
}

if [ "${SKIP_CANARY}" != "true" ]; then
  echo ""
  echo "########## E7-CANARY: is a compiled ${NUM_SAMPLES}-study decode affordable? ##########"
  echo "warming the Inductor cache (n=2, untimed) so both timed points are warm..."
  WARM=$(run_canary 2)
  echo "  cold build + 2 studies: ${WARM}s"
  E4=$(run_canary 4)
  E20=$(run_canary 20)
  PER=$(( (E20 - E4) / 16 ))
  [ "${PER}" -lt 1 ] && PER=1
  FIXED=$(( E4 - 4*PER )); [ "${FIXED}" -lt 0 ] && FIXED=0
  # +15%: a two-point fit on a shared node is not a guarantee.
  PROJ=$(( (FIXED + PER*NUM_SAMPLES) * 115 / 100 ))
  echo "n=4 took ${E4}s; n=20 took ${E20}s (both warm)"
  echo "  => fixed startup ~${FIXED}s, per study ~${PER}s"
  echo "  => projected ${NUM_SAMPLES} studies +15%: ${PROJ}s ($((PROJ/60)) min), budget ${TIME_BUDGET_S}s"
  if [ "${PROJ}" -gt "${TIME_BUDGET_S}" ]; then
    echo ""
    echo "ABORTING BEFORE THE FULL ARM. The projected compiled decode does not fit."
    echo "Re-submit with a larger --time and TIME_BUDGET_S, or a smaller NUM_SAMPLES."
    echo "A smaller n is still a real measurement -- same protocol, fewer studies -- so"
    echo "state n in the writeup rather than quietly reporting 400."
    exit 2
  fi
  rm -f "${CANARY_LOG}"
fi

echo ""
echo "########## E7-A: compiled arm (${NUM_SAMPLES} studies, chunk_size=${CHUNK_SIZE}) ##########"
echo "Watch for two lines: '[operator] mamba3_chunk_size: 64 -> ${CHUNK_SIZE}' and"
echo "'[compile] Dynamo captured N call(s)'. If the second is missing or N is 0 the"
echo "eval aborts -- an accidental eager run must not be reported as agreement."
decode_arm "${DUMP_DIR}" "${NUM_SAMPLES}" true

echo ""
echo "########## E7-B: eager reference arm, identical in every other respect ##########"
decode_arm "${REF_DUMP}" "${NUM_SAMPLES}" false

echo ""
echo "--- how many of the ${NUM_SAMPLES} reports changed textually ---"
if [ -f "${REF_DUMP}/hyps.txt" ] && [ -f "${DUMP_DIR}/hyps.txt" ]; then
  CHANGED=$(awk 'NR==FNR{a[FNR]=$0;next}{if(a[FNR]!=$0)c++}END{print c+0}' \
            "${REF_DUMP}/hyps.txt" "${DUMP_DIR}/hyps.txt")
  echo "${CHANGED} of ${NUM_SAMPLES} generated reports differ between eager and compiled."
  if [ "${CHANGED}" -eq 0 ]; then
    echo "=> torch.compile is now verified ON DECODED TEXT, not by analogy from a logit"
    echo "   tolerance. EFFICIENCY_NOTE.md 5b's caveat can be removed and replaced with"
    echo "   this measurement. CheXbert and the bootstrap below are then degenerate"
    echo "   (identical files give identical labels) and are not worth submitting."
  else
    echo "=> the reports CHANGE. That is a result, not a failure (V5-A found the same"
    echo "   under an operator swap). Run the CheXbert + bootstrap steps below and"
    echo "   report both the text change and whether any metric actually moved."
  fi
else
  echo "One or both hyps.txt are missing -- the comparison did not run. Check above."
fi

echo ""
echo "=== NEXT, on the login node (this job does not run them; only needed if CHANGED > 0) ==="
cat <<NEXT
  DUMP_DIR=${DUMP_DIR} sbatch scripts/score_chexbert_h100.sh
  DUMP_DIR=${REF_DUMP} sbatch scripts/score_chexbert_h100.sh

  A=${REF_DUMP} B=${DUMP_DIR} \\
    NAME_A=eager NAME_B=compiled PER_LABEL=true \\
    OUTPUT=analysis/bootstrap_eager_vs_compiled.md \\
    sbatch scripts/bootstrap_compare_h100.sh
NEXT
echo ""
echo "Reminder: the pre-registered rule is in EFFICIENCY_PLAN.md E7. Reports changing"
echo "while metrics tie is a RESULT. A silent eager run is not -- the eval aborts on it."
date
