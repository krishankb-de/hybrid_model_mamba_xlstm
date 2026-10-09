#!/bin/bash
# ============================================================================
# CHAT_UI_PLAN.md P9-G4, job 2 of 3 (CPU, about 30 min): CheXbert labels for the EOS dump's hyps.txt and refs.txt, written beside
# them (chexbert_metrics.json and chexbert_labels.json, which bootstrap_compare.py reads in job 3). Waits for job 1; the submit
# lines for all three jobs are in the header of scripts/eval_report_eos_compare_h100.sh.
#
# WHY NOT scripts/score_chexbert_h100.sh. Ruling 2 was to reuse it unchanged unless a line `chat_remote.sh summary` would show can
# carry text, ids or paths. It can. Its first line is
#     === Phase 11B standalone CheXbert scoring: ${HYP_FILE} / ${REF_FILE} ===
# a `===` line, which `summary` shows, naming both file paths; its two ERROR lines name paths too; it prints no RESULT line (the
# F1 numbers the scorer prints are lines `summary` hides, so the job could not be read through it at all); and it lets the
# scorer's raw stdout and stderr into the job log. This wrapper runs the SAME script (scripts/score_chexbert_standalone.py) with
# the same three arguments, in the same venv (.venv_chexbert), under the same Hugging Face environment, and prints none of that:
# the scorer's output goes to ${DUMP_DIR}/chexbert.log, and the job log carries the sync line and one RESULT line of numbers.
#   Hugging Face. CHAT_UI_PLAN.md D24: the CheXbert weights live in the DEFAULT Hugging Face cache, so HF_HOME is not
#   overridden here (the other chat jobs set it to the cache under SCRATCH_ROOT, which does not hold them), and HF_HUB_OFFLINE
#   defaults to 0 exactly as in score_chexbert_h100.sh, the wrapper that scored the published dumps. Override it to 1 from
#   outside if you want to be sure no network call is made.
#
# It refuses (an `ERROR ...` line, exit 1) when: DUMP_DIR is not a new results/chat_* directory (R8); hyps.txt or refs.txt is
# missing or they differ in length; the venv is missing; or the scoring is already finished (chexbert_labels.json exists: the
# scorer writes chexbert_metrics.json first and chexbert_labels.json last, so only the second marks a finished scoring; a
# requeued job that was cut short between them runs again and rewrites both). Afterwards scripts/report_eos_stats.py chexbert
# checks that the two files it wrote agree with hyps.txt on n, and prints the four F1 headlines; a scoring that wrote no label
# matrices fails the job, because the comparison cannot run without them.
# R7: the scorer's stdout and stderr (the labeller's warnings, a crash that echoes report text) go to ${DUMP_DIR}/chexbert.log and
# are never printed. Read the job with `bash scripts/chat_remote.sh summary logs/chat_eos_chexbert_<jobid>.log`.
# Output (MIMIC-derived: stays on the cluster, never committed, never copied to the laptop), beside the dump:
#   results/chat_report_eos_test_split_s42/{chexbert_metrics.json, chexbert_labels.json, chexbert.log, chexbert_result.err}
# ============================================================================
#SBATCH --partition=pot-hpi-aisc-batch
#SBATCH --account=aisc
#SBATCH --qos=aisc
#SBATCH --exclude=ga03,gx17v1,gx13v1   # ga03: ARM node, x86 venv incompatible; gx13v1: faulty GPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --job-name=chat_eos_chexbert
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

VENV_ACTIVATE="${VENV_ACTIVATE:-.venv_chexbert/bin/activate}"
DUMP_DIR="${DUMP_DIR:-results/chat_report_eos_test_split_s42}"

echo "=== P9-G4 CheXbert scoring of the EOS dump, the scorer of score_chexbert_h100 in its own venv ==="
echo "=== job=${SLURM_JOB_ID:-local} restart=${SLURM_RESTART_COUNT:-0} node=$(hostname) ==="

# --- guards: nothing is created and nothing runs before every one of them has passed -------------------------------------
case "${DUMP_DIR}" in
  *..*) fail "DUMP_DIR holds a .. segment" ;;
  results/chat_?*) ;;
  *) fail "DUMP_DIR is not a new chat_ directory under results (R8)" ;;
esac
[ -f "${DUMP_DIR}/hyps.txt" ] || fail "the dump has no hyps.txt: run the decode job first"
[ -f "${DUMP_DIR}/refs.txt" ] || fail "the dump has no refs.txt: run the decode job first"
if [ -e "${DUMP_DIR}/chexbert_labels.json" ]; then
  fail "the dump is already scored: a finished scoring is never overwritten (R8)"
fi
[ -f "${VENV_ACTIVATE}" ] || fail "the CheXbert venv is missing: run setup_chexbert_venv_h100.sh first"
LINES_HYPS="$(grep -c '' "${DUMP_DIR}/hyps.txt")" || LINES_HYPS=0
LINES_REFS="$(grep -c '' "${DUMP_DIR}/refs.txt")" || LINES_REFS=0
[ "${LINES_HYPS}" -eq "${LINES_REFS}" ] || fail "hyps.txt has ${LINES_HYPS} lines and refs.txt has ${LINES_REFS}"

# The CheXbert environment of score_chexbert_h100.sh (D24, see the header): HF_HOME is left alone.
source "${VENV_ACTIVATE}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export PYTHONUNBUFFERED=1

# --- the scoring -----------------------------------------------------------------------------------------------------------
echo "=== scoring ${LINES_HYPS} reports: the scorer's own output goes to chexbert.log and is never printed ==="
SECONDS=0
rc=0
python scripts/score_chexbert_standalone.py \
  --hyp-file "${DUMP_DIR}/hyps.txt" \
  --ref-file "${DUMP_DIR}/refs.txt" \
  --output-dir "${DUMP_DIR}" >> "${DUMP_DIR}/chexbert.log" 2>&1 || rc=$?
WALL_S="${SECONDS}"
if [ "${rc}" -ne 0 ]; then
  echo "ERROR chexbert exit=${rc}"
  exit "${rc}"
fi

# --- the result. R7: stdout is cut down to the three line shapes, stderr goes to a file. Its failure fails the job.
rc=0
RESULT_OUT="$(python scripts/report_eos_stats.py chexbert --dump-dir "${DUMP_DIR}" --wall-s "${WALL_S}" 2>> "${DUMP_DIR}/chexbert_result.err")" || rc=$?
printf '%s\n' "${RESULT_OUT}" | grep -aE '^(=== |RESULT |ERROR)' || true
[ "${rc}" -eq 0 ] || fail "result exit=${rc}: the scoring is not usable by the comparison"
echo "=== END chexbert ==="
