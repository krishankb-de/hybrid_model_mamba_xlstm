#!/bin/bash
# ============================================================================
# CHAT_UI_PLAN.md P9-G4, job 3 of 3 (CPU, about 15 min): the EOS dump against the published Mamba-3 s42 dump on the official test
# split. scripts/bootstrap_compare.py, unchanged, with per-label intervals; the mean length, repeats and unterminated endings of
# both dumps; and the gate: no metric significantly worse than the published run.
#
# THE THREE JOBS, CHAINED. The controller submits them, from the Mac, in this order, one afterok on the one before (`submit`
# prints the job id; ${DEC%%;*} drops a ;cluster suffix a multi-cluster sbatch --parsable may add, as submit_v3_chain.sh does).
# Job 1 is the decode (GPU, `-- --time=04:00:00` after it for more margin), job 2 the CheXbert scoring, job 3 this one:
#   bash scripts/chat_remote.sh sync
#   DEC=$(bash scripts/chat_remote.sh submit scripts/eval_report_eos_h100.sh)
#   CHX=$(bash scripts/chat_remote.sh submit scripts/eval_report_eos_chexbert_h100.sh DUMP_DIR=results/chat_report_eos_test_split_s42 -- --dependency=afterok:${DEC%%;*})
#   CMP=$(bash scripts/chat_remote.sh submit scripts/eval_report_eos_compare_h100.sh DUMP_DIR=results/chat_report_eos_test_split_s42 PUBLISHED_DIR=results/report_gen_m3_test_split_s42 -- --dependency=afterok:${CHX%%;*})
# Read each job, when it ends, only through `bash scripts/chat_remote.sh state <jobid>` and
#   bash scripts/chat_remote.sh summary logs/chat_eos_eval_<jobid>.log        (job 1: the EOS share, the wall time, the text metrics)
#   bash scripts/chat_remote.sh summary logs/chat_eos_chexbert_<jobid>.log    (job 2: the four CheXbert F1 headlines)
#   bash scripts/chat_remote.sh summary logs/chat_eos_compare_<jobid>.log     (job 3: this job)
# All three default to the same DUMP_DIR, so the settings above only spell the defaults out. The decode needs the training run's
# DONE marker (P9-G3); this job needs the first two to have finished, and each refuses to run on what the one before did not write.
#
# THE COMPARISON IS FIXED HERE, NOT AN ENVIRONMENT LEVER. A is the EOS dump (eos_s42), B the published Mamba-3 s42 dump, decoded with
# the published protocol (beam 3, 100 tokens, uncached; V3 chain, stages 3 and 4: it already holds hyps, refs and the CheXbert
# label matrices). bootstrap_compare.py runs with the flags bootstrap_compare_h100.sh gives it, including --per-label, and the seed
# and resample count that wrapper defaults to (0 and 1000); tests/test_willi_parity.py parses that wrapper and pins all of it. The
# report is written beside the EOS dump. The two dumps must be the same studies: their refs.txt files must be byte-identical.
#
# WHAT IT PRINTS (RESULT lines, numbers and plain names only; scripts/report_eos_stats.py says what each holds):
#   the stats of A, then of B, then of the references (the baseline: some reference reports lack a final period too, and the length
#   BLEU and ROUGE are scored against): mean words and GPT-2 tokens (null when the tokenizer is not cached offline), empty reports,
#   repeated sentences per report, the share with a repeat, the share whose last sentence is unterminated (V5-D's cut-short measure);
#   then the bootstrap's n, resamples and seed, one line per metric (A, B, A - B and its 95% CI, parsed from bootstrap_compare's own
#   report and never recomputed) and one per label; and last the gate:
#     RESULT {"gate":"pass"|"fail","worse":[...]}      worse = every main-table metric whose CI of (EOS - published) is entirely below 0
# Per-label rows are informational and do not gate (the module docstring says why). A failing gate is a finding, not a crash: the
# job exits 0 either way. A report that cannot be parsed prints an ERROR and NO gate line, and the job exits 1.
# R7: the bootstrap's stdout and stderr go to ${DUMP_DIR}/bootstrap.log, the stats script's stderr to ${DUMP_DIR}/compare.err, and the
# job log carries only === / RESULT / ERROR lines with no path. R8: everything is written under DUMP_DIR; the published dump is read.
# Output, beside the EOS dump: results/chat_report_eos_test_split_s42/{bootstrap_eos_s42_vs_m3_s42.md, bootstrap.log, compare.err}
# ============================================================================
#SBATCH --partition=pot-hpi-aisc-batch
#SBATCH --account=aisc
#SBATCH --qos=aisc
#SBATCH --exclude=ga03,gx17v1,gx13v1   # ga03: ARM node, the x86 .venv cannot execute there; gx13v1: faulty GPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --job-name=chat_eos_compare
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log
#SBATCH --requeue
#SBATCH --open-mode=append

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}/hybrid_model_mamba_xlstm" 2>/dev/null || cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs

# Provenance, before anything else: the commit and cleanliness `scripts/chat_remote.sh sync` last shipped to this tree. It writes
# .sync_stamp as "<UTC time> <40-hex commit> <clean|dirty>" and sends it last, so the stamp names a complete transfer. Only the
# commit and the flag are printed, and only for a line of exactly that shape: anything else reads as unknown.
SYNC="unknown"
if [ -f .sync_stamp ]; then
  { read -r _ sync_sha sync_flag sync_extra < .sync_stamp; } 2>/dev/null || true
  if [[ "${sync_sha:-}" =~ ^[0-9a-f]{40}$ && ( "${sync_flag:-}" == "clean" || "${sync_flag:-}" == "dirty" ) && -z "${sync_extra:-}" ]]; then
    SYNC="${sync_sha} ${sync_flag}"
  fi
fi
echo "=== sync ${SYNC} ==="

# Every line printed below starts with ===, RESULT or ERROR (the shapes `chat_remote.sh summary` shows) and names no path.
fail() { echo "ERROR $*"; exit 1; }

SCRATCH_ROOT="${SCRATCH_ROOT:-/sc/scratch/$USER/hybrid_xmamba_h100}"
VENV_ACTIVATE="${VENV_ACTIVATE:-.venv/bin/activate}"
DUMP_DIR="${DUMP_DIR:-results/chat_report_eos_test_split_s42}"
PUBLISHED_DIR="${PUBLISHED_DIR:-results/report_gen_m3_test_split_s42}"

# --- the comparison (see the header): plain assignments, deliberately not ${VAR:-default} --------------------------------
NAME_A=eos_s42
NAME_B=m3_s42
SAMPLES=1000
SEED=0
OUTPUT="${DUMP_DIR}/bootstrap_${NAME_A}_vs_${NAME_B}.md"

echo "=== P9-G4 compare: ${NAME_A} against ${NAME_B}, paired bootstrap ${SAMPLES} resamples seed ${SEED}, per-label on ==="
echo "=== job=${SLURM_JOB_ID:-local} restart=${SLURM_RESTART_COUNT:-0} node=$(hostname) ==="

export HF_HOME="${SCRATCH_ROOT}/.hf"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

# --- guards: nothing is created and nothing runs before every one of them has passed -------------------------------------
case "${DUMP_DIR}" in
  *..*) fail "DUMP_DIR holds a .. segment" ;;
  results/chat_?*) ;;
  *) fail "DUMP_DIR is not a new chat_ directory under results (R8)" ;;
esac
need() { [ -f "$1" ] || fail "$2"; }
need "${DUMP_DIR}/hyps.txt" "the EOS dump has no hyps.txt: run the decode job first"
need "${DUMP_DIR}/refs.txt" "the EOS dump has no refs.txt: run the decode job first"
need "${DUMP_DIR}/chexbert_labels.json" "the EOS dump has no chexbert_labels.json: run the CheXbert job first"
need "${PUBLISHED_DIR}/hyps.txt" "the published dump has no hyps.txt"
need "${PUBLISHED_DIR}/refs.txt" "the published dump has no refs.txt"
need "${PUBLISHED_DIR}/chexbert_labels.json" "the published dump has no chexbert_labels.json"
cmp -s "${DUMP_DIR}/refs.txt" "${PUBLISHED_DIR}/refs.txt" || fail "the two dumps have different refs.txt: they are not the same studies"
if [ -e "${OUTPUT}" ]; then
  fail "the comparison report already exists: a finished comparison is never overwritten (R8)"
fi
[ -f "${VENV_ACTIVATE}" ] || fail "venv not found"
source "${VENV_ACTIVATE}"

# --- the bootstrap: scripts/bootstrap_compare.py as bootstrap_compare_h100.sh runs it with label matrices and PER_LABEL=true ----
echo "=== bootstrap: its own output goes to bootstrap.log and is never printed ==="
rc=0
python scripts/bootstrap_compare.py \
  --hyps-a "${DUMP_DIR}/hyps.txt" \
  --hyps-b "${PUBLISHED_DIR}/hyps.txt" \
  --refs "${DUMP_DIR}/refs.txt" \
  --name-a "${NAME_A}" \
  --name-b "${NAME_B}" \
  --bootstrap-samples "${SAMPLES}" \
  --seed "${SEED}" \
  --output "${OUTPUT}" \
  --labels-a "${DUMP_DIR}/chexbert_labels.json" \
  --labels-b "${PUBLISHED_DIR}/chexbert_labels.json" \
  --per-label >> "${DUMP_DIR}/bootstrap.log" 2>&1 || rc=$?
if [ "${rc}" -ne 0 ]; then
  echo "ERROR bootstrap exit=${rc}"
  exit "${rc}"
fi

# --- the stats of both dumps, then the comparison and the gate. R7: each script's stdout is cut down to the three line shapes
# before it reaches the log, and its stderr goes to a file. A failure of either fails the job; the gate line is the last RESULT.
rc=0
STATS_OUT="$(python scripts/report_eos_stats.py hyps "${NAME_A}=${DUMP_DIR}/hyps.txt" "${NAME_B}=${PUBLISHED_DIR}/hyps.txt" "refs=${DUMP_DIR}/refs.txt" 2>> "${DUMP_DIR}/compare.err")" || rc=$?
printf '%s\n' "${STATS_OUT}" | grep -aE '^(=== |RESULT |ERROR)' || true
[ "${rc}" -eq 0 ] || fail "stats exit=${rc}"

rc=0
GATE_OUT="$(python scripts/report_eos_stats.py gate --bootstrap "${OUTPUT}" --name-a "${NAME_A}" --name-b "${NAME_B}" 2>> "${DUMP_DIR}/compare.err")" || rc=$?
printf '%s\n' "${GATE_OUT}" | grep -aE '^(=== |RESULT |ERROR)' || true
[ "${rc}" -eq 0 ] || fail "gate exit=${rc}: no verdict"
echo "=== END compare ==="
