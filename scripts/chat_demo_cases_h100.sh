#!/bin/bash
# ============================================================================
# CHAT_UI_PLAN.md P9-C (CPU, 1 CPU, 4 GB, about 1 min): which test studies does the supervisor demo show? A selection job: it trains and decodes nothing.
#   bash scripts/chat_remote.sh sync && bash scripts/chat_remote.sh submit scripts/chat_demo_cases_h100.sh
#   bash scripts/chat_remote.sh summary logs/chat_demo_cases_<jobid>.log
#
# The demo shows five test studies, two normal, two with support devices and one with a pneumothorax, chosen from the published labels. This job
# reads the published run's per-sample CheXbert labels (${PUBLISHED_MODEL}/chexbert_labels.json: its y_true, the labels of the reference reports,
# one row of 14 per test study) and lists, for each of the three kinds, up to five candidate rows of the test split: a seeded fair draw (seed 42 for
# the first kind, 43 and 44 for the others) from every row of that kind, which never looks at what the model predicted or scored. It refuses when
# the number of studies is not the gallery's test-split size (GALLERY's manifest.json), and when the gallery's labels_check.json, the record P5-C
# leaves in it, does not say that the gallery was checked against this very dump (its `reference` is the dump's directory name), over this many
# studies, with no difference: a row number must name the same study the picker shows, which the count alone does not say.
# scripts/chat_demo_cases.py says exactly what it reads, how it draws and what it prints; the controller then takes the first 2, 2 and 1
# candidates, skipping a row already picked.
#
# Output. The job log carries the sync line, a banner, a notice, one RESULT line and two closing lines. A normal log:
#   === sync <40-hex commit> <clean|dirty> ===
#   === chat demo cases: job=<id> node=<node> gallery=<build id> ===
#   === demo cases: the script's raw output stays in files under CHAT_HOME; its RESULT line of a known shape is shown here ===
#   RESULT {"demo_cases":"done","n":2663,"counts":{"normal":N,"support_devices":N,"pneumothorax":N},"candidates":{"normal":[...],"support_devices":[...],
#           "pneumothorax":[...]},"seed":42}                (one line: the candidates are test-split row numbers, at most 5 of each kind)
#   === demo case lines withheld: 0 ===
#   === END chat demo cases ===
# The job exits 0 when the script printed its RESULT line, and with the script's own status otherwise (1: it refused, and its `ERROR <code> ...`
# line, one of a closed list, is shown before the job's own `ERROR the demo cases script exited with code 1`; 64: a bad command line, which the
# levers below cannot make). Exit 0 without exactly one RESULT line is a failure like any other.
#
# R7 (DUA). The labels say which test study has which finding. No MIMIC-derived value reaches this log: the script's stdout and stderr go to files
# under ${CHAT_HOME}/logs (demo_cases_<jobid>.out and .err), and of stdout only a line of one of the shapes of CASES_SHAPES below (an allowlist: a
# report, an id or a path on a RESULT or ERROR line is no known shape, and an ERROR line takes no number key but its own code's) reaches this log;
# `=== demo case lines withheld: N ===` counts the others, and no refusal of this wrapper quotes the value it refuses. What the log may name is
# infrastructure, not data: an infrastructure error, the shell's own message when the job's mkdir or tee fails under CHAT_HOME, is written to the
# same log and may name a path under CHAT_HOME, which holds the user name and nothing of the studies.
#
# Files (R8: additive only). The two files of the job's raw output, owner-only, under ${CHAT_HOME}/logs. Nothing else is written; the labels, the
# gallery's manifest and its labels_check.json are only read, and no model, no image and no report text is touched.
#
# Levers, as NAME=value arguments of `chat_remote.sh submit`: CHAT_HOME (default $HOME/chat_sessions), GALLERY (a build directory, default
# ${CHAT_HOME}/gallery/g13d_m3_v1, the one the server serves), PUBLISHED_MODEL (the published model's dump, a plain path under the repository,
# default results/report_gen_m3_test_split_s42, the serve job's own default).
# ============================================================================
#SBATCH --partition=pot-hpi-aisc-batch
#SBATCH --account=aisc
#SBATCH --qos=aisc
#SBATCH --exclude=ga03,gx17v1,gx13v1   # ga03: ARM node, x86 .venv incompatible; gx13v1: faulty GPU
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --job-name=chat_demo_cases
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log

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

# Every line printed below starts with === or ERROR (or is a RESULT line of the script) and names no path, port or value that was given (R7).
fail() { echo "ERROR $*"; exit 1; }

VENV_ACTIVATE="${VENV_ACTIVATE:-.venv/bin/activate}"
CHAT_HOME="${CHAT_HOME:-$HOME/chat_sessions}"
GALLERY="${GALLERY:-${CHAT_HOME}/gallery/g13d_m3_v1}"
PUBLISHED_MODEL="${PUBLISHED_MODEL:-results/report_gen_m3_test_split_s42}"
JOB="${SLURM_JOB_ID:-local}"

# The lines of the script's stdout that may reach this log, as one anchored pattern per line (grep -E reads a newline as "or"). The keys are fixed
# and so are the error codes, and each code takes the number keys the script gives it (ERROR_CODES in scripts/chat_demo_cases.py) and no other; a
# value is a count or a row number of at most 7 digits, and the candidates are lists of at most five of them. A report, an id or a path on a RESULT
# or ERROR line matches none of them. tests/test_chat_demo_cases.py runs these very patterns, through grep, over everything the script prints and
# over lines that carry text. No line may be empty: an empty pattern matches every line.
CASES_SHAPES='^RESULT \{"demo_cases":"done","n":[0-9]{1,7},"counts":\{"normal":[0-9]{1,7},"support_devices":[0-9]{1,7},"pneumothorax":[0-9]{1,7}\},"candidates":\{"normal":\[([0-9]{1,7}(,[0-9]{1,7}){0,4})?\],"support_devices":\[([0-9]{1,7}(,[0-9]{1,7}){0,4})?\],"pneumothorax":\[([0-9]{1,7}(,[0-9]{1,7}){0,4})?\]\},"seed":[0-9]{1,6}\}$
^ERROR (usage|labels_unreadable|label_names|gallery_unreadable|check_missing|check_unreadable|check_stage|check_reference)$
^ERROR labels_shape rows=[0-9]{1,7} width=[0-9]{1,7}$
^ERROR count_mismatch labels=[0-9]{1,7} gallery=[0-9]{1,7}$
^ERROR check_n_test labels=[0-9]{1,7}( n_test=[0-9]{1,7})?$
^ERROR check_mismatch( refs_mismatch=[0-9]{1,7})?( labels_mismatch=[0-9]{1,7})?$
^ERROR failed [A-Za-z_][A-Za-z0-9_]{0,59}$'

# --- the levers and the tree, before anything runs. A refusal says what is wrong and never the value it was given. -------------------------------
[[ "$(basename "${GALLERY}")" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || fail "GALLERY must end in one plain build name: letters, digits, dot, dash and underscore"
[[ "${PUBLISHED_MODEL}" =~ ^[A-Za-z0-9._-]+(/[A-Za-z0-9._-]+)*$ && "${PUBLISHED_MODEL}" != *..* ]] \
  || fail "PUBLISHED_MODEL must be a plain relative path: letters, digits, dot, dash, underscore and slash, none leading and no .."
[ -d "${CHAT_HOME}" ] || fail "CHAT_HOME does not exist: run chat_cluster_setup_h100.sh first"
[ -f "${GALLERY}/manifest.json" ] || fail "the gallery has no manifest.json: build it first (P5-B)"
[ -f "${PUBLISHED_MODEL}/chexbert_labels.json" ] || fail "the published dump has no chexbert_labels.json"
[ -f "${VENV_ACTIVATE}" ] || fail "the venv's activate script is missing: run chat_cluster_setup_h100.sh first"
[ -f scripts/chat_demo_cases.py ] || fail "chat_demo_cases.py is missing from this tree: run chat_remote.sh sync first"

echo "=== chat demo cases: job=${JOB} node=$(hostname) gallery=$(basename "${GALLERY}") ==="

# The serving environment's interpreter and nothing else: the script is the standard library and the app's list of label names. Files made from
# here on are owner-only: the raw output is numbers, but it sits beside the stores that hold report text.
umask 077
export PYTHONUNBUFFERED=1
source "${VENV_ACTIVATE}"
mkdir -p "${CHAT_HOME}/logs"
OUT="${CHAT_HOME}/logs/demo_cases_${JOB}.out"
ERR="${CHAT_HOME}/logs/demo_cases_${JOB}.err"

echo "=== demo cases: the script's raw output stays in files under CHAT_HOME; its RESULT line of a known shape is shown here ==="
# The script's stdout goes through tee into its file and through grep into this log: only a line of a known shape passes grep, the others are in
# the file and are counted below, not shown. Its stderr goes to its own file. The status of each stage of the pipeline is read straight after it
# (a grep that finds nothing is not an error, and set -e is off for the pipeline alone).
set +e
python scripts/chat_demo_cases.py --labels "${PUBLISHED_MODEL}/chexbert_labels.json" --gallery "${GALLERY}" 2> "${ERR}" | tee "${OUT}" | grep --line-buffered -aE "${CASES_SHAPES}"
STATUSES=("${PIPESTATUS[@]}")
set -e
rc="${STATUSES[0]:-1}"
case "${rc}" in ''|*[!0-9]*) rc=1 ;; esac
[ "${STATUSES[1]:-1}" -eq 0 ] || fail "the demo cases output could not be kept"
WITHHELD="$(grep -avcE "${CASES_SHAPES}" "${OUT}" || true)"
case "${WITHHELD}" in ''|*[!0-9]*) WITHHELD=unknown ;; esac
echo "=== demo case lines withheld: ${WITHHELD} ==="
RESULTS="$(grep -aE "${CASES_SHAPES}" "${OUT}" | grep -c '^RESULT ' || true)"
case "${RESULTS}" in ''|*[!0-9]*) RESULTS=0 ;; esac
if [ "${rc}" -ne 0 ]; then
  echo "ERROR the demo cases script exited with code ${rc} (see the logs under CHAT_HOME)"
  exit "${rc}"
fi
# Exit 0 without exactly one RESULT line is a failure like any other: no candidates were given.
[ "${RESULTS}" -eq 1 ] || fail "the demo cases script printed ${RESULTS} result lines, not 1"
echo "=== END chat demo cases ==="
