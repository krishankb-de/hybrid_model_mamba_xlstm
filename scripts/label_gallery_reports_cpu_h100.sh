#!/bin/bash
# ============================================================================
# CHAT_UI_PLAN.md P5-C, the CPU fallback (4 CPUs, no GPU, at most 2 h per job): CheXbert-14 labels for the report rows of the retrieval gallery,
# in SHARDS array tasks of about 1 h each (shard i labels the groups reps[i::SHARDS] and writes labels_shard_i.npy beside the gallery's other
# files), then one MERGE=1 job that combines them, runs the cross-check against the published labels of the 2,663 test references and finishes
# the gallery (labels.npy, label_names.json, labels_status done). For when scripts/label_gallery_reports_h100.sh exits 2, which it does when the
# venv's torch sees no GPU (scripts/setup_chexbert_venv_h100.sh installs a CPU-only torch) or its canary projects the one job past its budget: either
# way f1chexbert is on the CPU, about 0.16 s a report, 7-8 h for the 150-190k duplicate groups (job 2525606). It can be submitted directly: the GPU job
# would only say the same, after a queue wait.
#   bash scripts/chat_remote.sh sync                       (once: the sync stamp of this tree is what each job prints first)
#   bash scripts/chat_remote.sh submit scripts/label_gallery_reports_cpu_h100.sh BUILD_ID=g13d_m3_v1 -- --array=0-7
#   bash scripts/chat_remote.sh submit scripts/label_gallery_reports_cpu_h100.sh BUILD_ID=g13d_m3_v1 MERGE=1 -- --dependency=afterok:<the array job id>
#   (the array is given on the command line, not in this header, so that the same wrapper serves the merge; SHARDS, default 8, must be the array's
#   size; a finished shard is kept, so a resubmitted array only does what is missing. If one task fails, the merge, which waits for all of them
#   with afterok, stays pending (DependencyNeverSatisfied): scancel it, run the failed shard again with `-- --array=3`, and submit the merge again.
#   The gallery is named by BUILD_ID, default g13d_m3_v1; the published dump by REFERENCE_DIR; the seconds a shard's canary may project, default
#   6000, by BUDGET_S; how many groups it labels first by CANARY, default 1000.)
#
# A shard whose canary projects past BUDGET_S exits 2 and says to use more shards: it could not finish in its 2 h. It does so only with the script's
# canary line in the raw log: an exit 2 without it (python itself exits 2 for a script it cannot open) is a failure like any other, printed
# `ERROR labels exit=2`, and this wrapper's own exit is then 1, so that its exit 2 always means "use more shards". A shard that is already labelled
# (both of its files there, and exactly the rows it should hold: they are compared with the gallery's own groups) is kept and costs a second, so a
# requeue or a resubmission never labels twice. The merge names the shards that are missing, then any that is not the one it should be, and writes
# nothing; the cross-check is the script's, as in the GPU job: on any difference the job fails (exit 1), labels_status stays pending, and the labels
# are kept as labels_unverified.npy.
#   Hugging Face. CHAT_UI_PLAN.md D24: the CheXbert weights live in the DEFAULT Hugging Face cache, so HF_HOME is not overridden here, and HF_HUB_OFFLINE
#   defaults to 0 exactly as in score_chexbert_h100.sh, the wrapper that scored the published dumps. It asks the labeller for the CPU (--device cpu),
#   whichever node it lands on, so that the labels are those of the CPU run that the published labels came from.
#
# It refuses (an `ERROR ...` line, exit 1) when: BUILD_ID is not one plain name; BUDGET_S, CANARY or SHARDS is not a whole number, or MERGE is not 0 or 1; a shard job
# is not an array task or its task id is not below SHARDS; CHAT_HOME, the gallery or its manifest.json is missing; the gallery resolves into an outputs
# directory or under the thesis checkout (R8); the gallery is already labelled (labels_status done, checked before any raw log is opened); the CheXbert venv, the
# labelling script (a sync that did not happen) or the reference dump's refs.txt or chexbert_labels.json is missing. The script then refuses a gallery whose
# gate_rk.equal is not true, and a reference that does not match the gallery, before the labeller is built; and the merge checks R8 again before it writes
# anything, from a fresh read of the manifest (a GPU job and a merge can both start on a pending gallery; the one that ends second refuses).
# R7: as in the GPU wrapper. The job log carries only === lines, [labels], RESULT and ERROR lines of the shapes in LABEL_SHAPES below (the same list; the two
# submit lines the GPU wrapper prints name this wrapper, which is repo code), and `=== label lines withheld: N ===`. The raw stdout and stderr go to
# ${GALLERY}/labels_shard_<i>.log and ${GALLERY}/labels_merge.log and are never printed. Read a job with
# `bash scripts/chat_remote.sh summary logs/chat_labels_cpu_<jobid>.log`.
# Output (MIMIC-derived, Class R: stays on the cluster, never committed, never copied to the laptop), in ${CHAT_HOME}/gallery/<build id>/:
#   a shard: labels_shard_<i>.npy, labels_shard_<i>_rows.npy, labels_shard_<i>.log
#   the merge: labels.npy, label_names.json, labels_check.json, labels_merge.log, and labels_status in manifest.json
# ============================================================================
#SBATCH --partition=pot-hpi-aisc-batch
#SBATCH --account=aisc
#SBATCH --qos=aisc
#SBATCH --exclude=ga03,gx17v1,gx13v1   # ga03: ARM node, x86 venv incompatible; gx13v1: faulty GPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --job-name=chat_labels_cpu
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

# Every line printed below starts with ===, RESULT or ERROR, or is a [labels] line of a known shape (the shapes `chat_remote.sh summary` shows),
# and names no path.
fail() { echo "ERROR $*"; exit 1; }
# `readlink -f` needs the path to exist and is not the same everywhere; python's realpath is.
realpath_py() { python -c 'import os, sys; print(os.path.realpath(sys.argv[1]))' "$1"; }

VENV_ACTIVATE="${VENV_ACTIVATE:-.venv_chexbert/bin/activate}"
CHAT_HOME="${CHAT_HOME:-/sc/home/$USER/chat_sessions}"
BUILD_ID="${BUILD_ID:-g13d_m3_v1}"
REFERENCE_DIR="${REFERENCE_DIR:-results/report_gen_m3_test_split_s42}"
BUDGET_S="${BUDGET_S:-6000}"
CANARY="${CANARY:-1000}"
SHARDS="${SHARDS:-8}"
MERGE="${MERGE:-0}"
# One name, never a path: GALLERY is CHAT_HOME/gallery/<name> and nothing else. The numbers go onto the script's command line.
[[ "${BUILD_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || fail "BUILD_ID must be one plain name: letters, digits, dot, dash, underscore"
[[ "${BUDGET_S}" =~ ^[0-9]{1,6}$ ]] || fail "BUDGET_S must be a whole number of seconds, at most 6 digits"
[[ "${CANARY}" =~ ^[0-9]{1,7}$ ]] || fail "CANARY must be a whole number of groups, at most 7 digits"
[[ "${SHARDS}" =~ ^[1-9][0-9]{0,2}$ ]] || fail "SHARDS must be a whole number from 1 to 999"
case "${MERGE}" in 0|1) ;; *) fail "MERGE must be 0 or 1" ;; esac
if [ "${MERGE}" = 1 ]; then
  MODE=merge
else
  MODE=shard
  [[ "${SLURM_ARRAY_TASK_ID:-}" =~ ^[0-9]{1,3}$ ]] || fail "not an array task: submit with --array=0-7 (SHARDS tasks), or with MERGE=1 for the merge"
  [ "${SLURM_ARRAY_TASK_ID}" -lt "${SHARDS}" ] || fail "array task ${SLURM_ARRAY_TASK_ID} is outside the ${SHARDS} shards"
fi
GALLERY="${CHAT_HOME}/gallery/${BUILD_ID}"

# The lines the script prints that may reach this log, as one anchored pattern per line (grep -E reads a newline as "or"): digits are bounded
# (a count has at most 7), and nothing in a shape is free text, so an id or a piece of a report on a [labels] line, or an exception message on an
# ERROR line, matches none of them, whatever is printed. tests/test_label_gallery_reports.py runs these very patterns, through grep, over everything
# the script prints and over ids, report text and paths; both wrappers carry the same list. No line may be empty: an empty pattern matches every line.
LABEL_SHAPES='^\[labels\] mode=single$
^\[labels\] mode=shard index=[0-9]{1,3} of=[0-9]{1,3}$
^\[labels\] mode=merge of=[0-9]{1,3}$
^\[labels\] rows=[0-9]{1,7} groups=[0-9]{1,7} test=[0-9]{1,7}$
^\[labels\] init_args=[A-Za-z_][A-Za-z0-9_]{0,31}(,[A-Za-z_][A-Za-z0-9_]{0,31}){0,7}$
^\[labels\] device=(cpu|cuda|cuda:[0-9]{1,2}|unknown)$
^\[labels\] canary: [0-9]{1,4}\.[0-9]{3} s/report, projected [0-9]{1,9} s for [0-9]{1,7} rows$
^\[labels\] progress: [0-9]{1,7} of [0-9]{1,7} rows$
^\[labels\] shard [0-9]{1,3} of [0-9]{1,3} kept: already labelled$
^\[labels\] wrote (labels\.npy|labels_unverified\.npy|label_names\.json|labels_check\.json|labels_shard_[0-9]{1,3}\.npy|labels_shard_[0-9]{1,3}_rows\.npy)$
^\[labels\] mismatch split: own_text=[0-9]{1,7} shared_text=[0-9]{1,7}$
^\[labels\] test_rows_sharing_a_text=[0-9]{1,7}$
^\[labels\] labels_status=(done|pending)$
^RESULT \{"refs_mismatch":[0-9]{1,7},"labels_mismatch":[0-9]{1,7}\}$
^RESULT \{"groups":[0-9]{1,7},"rows":[0-9]{1,7},"labelling_s":[0-9]{1,7}\}$
^RESULT \{"shard":[0-9]{1,3},"of":[0-9]{1,3},"groups":[0-9]{1,7},"labelling_s":[0-9]{1,7}\}$
^RESULT \{"merged":[0-9]{1,3},"groups":[0-9]{1,7},"rows":[0-9]{1,7}\}$
^ERROR (no_manifest|manifest_unreadable|gate_not_equal|already_labelled|inputs_unreadable|rows_disagree|groups_disagree|test_rows|reference_unreadable|reference_shape|reference_label_names|refs_mismatch|no_device_argument|label_order|bad_label_row|shards_missing|shard_invalid)( [a-z_]{1,20}=[0-9]{1,7}){0,3}$
^ERROR failed [A-Za-z_][A-Za-z0-9_]{0,59}$
^=== note: chexbert_labels\.json has no label_names key: CHEXBERT_14 order assumed ===$
^=== note: chexbert_labels\.json label_names reordered to the CHEXBERT_14 order ===$'

if [ "${MODE}" = shard ]; then
  echo "=== P5-C CheXbert labels for gallery ${BUILD_ID}: shard ${SLURM_ARRAY_TASK_ID} of ${SHARDS} on the CPU ==="
  echo "=== job=${SLURM_JOB_ID:-local} restart=${SLURM_RESTART_COUNT:-0} node=$(hostname) task=${SLURM_ARRAY_TASK_ID} ==="
else
  echo "=== P5-C CheXbert labels for gallery ${BUILD_ID}: merge of ${SHARDS} shards on the CPU ==="
  echo "=== job=${SLURM_JOB_ID:-local} restart=${SLURM_RESTART_COUNT:-0} node=$(hostname) ==="
fi

# --- guards: no step starts, and the gallery is not touched, before every one of them has passed ------------------------
# CHAT_HOME is made by scripts/chat_cluster_setup_h100.sh, owner-only, and not here; the gallery is the build's, and not made here either.
[ -d "${CHAT_HOME}" ] || fail "CHAT_HOME does not exist: run chat_cluster_setup_h100.sh first"
[ -f "${GALLERY}/manifest.json" ] || fail "the gallery has no manifest.json: build it first (build_retrieval_gallery_h100.sh)"
[ -f "${VENV_ACTIVATE}" ] || fail "the CheXbert venv is missing: run setup_chexbert_venv_h100.sh first"
# Python answers a script it cannot find with exit 2, the code this wrapper reads as "use more shards": say what is wrong instead.
[ -f scripts/label_gallery_reports.py ] || fail "label_gallery_reports.py is missing from this tree: run chat_remote.sh sync first"

# The CheXbert environment of score_chexbert_h100.sh (D24, see the header): HF_HOME is left alone.
source "${VENV_ACTIVATE}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export PYTHONUNBUFFERED=1

MAIN_REAL="$(dirname "$(realpath_py outputs)")"       # the thesis checkout: ./outputs is a symlink into it
GALLERY_REAL="$(realpath_py "${GALLERY}")"
case "${GALLERY}/" in */outputs/*) fail "the gallery is inside an outputs directory" ;; esac
case "${GALLERY_REAL}/" in
  */outputs/*) fail "the gallery resolves into an outputs directory" ;;
  "${MAIN_REAL}"/*) fail "the gallery is under the thesis checkout (R8)" ;;
esac
# A finished labelling is never overwritten, and neither is the raw log of the job that finished it: refused here, before any log is opened.
# (The script checks the same from the manifest's JSON.)
if grep -Eq '"labels_status":[[:space:]]*"done"' "${GALLERY}/manifest.json"; then
  fail "the gallery is already labelled (labels_status done): a finished labelling is never overwritten (R8)"
fi
[ -f "${REFERENCE_DIR}/refs.txt" ] || fail "the reference dump has no refs.txt"
[ -f "${REFERENCE_DIR}/chexbert_labels.json" ] || fail "the reference dump has no chexbert_labels.json"

# --- the labelling -------------------------------------------------------------------------------------------------------
# Only the script's lines of a known shape leave the raw log (a progress bar can sit in front of one on the same physical line, hence the tr);
# the lines that wear one of our prefixes and have none of the shapes are counted, not shown. A requeue replaces the raw log.
echo "=== labelling: the script's own output goes to a log in the gallery and is never printed ==="
SECONDS=0
rc=0
if [ "${MODE}" = shard ]; then
  LOG="${GALLERY}/labels_shard_${SLURM_ARRAY_TASK_ID}.log"
  python scripts/label_gallery_reports.py --gallery "${GALLERY}" --reference-dir "${REFERENCE_DIR}" --budget-s "${BUDGET_S}" \
    --canary "${CANARY}" --device cpu --shard "${SLURM_ARRAY_TASK_ID}" --of "${SHARDS}" > "${LOG}" 2>&1 || rc=$?
else
  LOG="${GALLERY}/labels_merge.log"
  python scripts/label_gallery_reports.py --gallery "${GALLERY}" --reference-dir "${REFERENCE_DIR}" --merge --of "${SHARDS}" > "${LOG}" 2>&1 || rc=$?
fi
tr '\r' '\n' < "${LOG}" | grep -aE "${LABEL_SHAPES}" | tail -n 60 || true
WITHHELD="$(tr '\r' '\n' < "${LOG}" | grep -aE '^(\[labels\] |RESULT |ERROR|=== )' | grep -avcE "${LABEL_SHAPES}" || true)"
case "${WITHHELD}" in ''|*[!0-9]*) WITHHELD=unknown ;; esac
echo "=== label lines withheld: ${WITHHELD} ==="
# Exit 2 is the canary's only if the script's own canary line, of its known shape, is in the raw log: python itself exits 2 when it cannot
# open the script, and a library may too. Without the line it is a failure like any other, and the wrapper's own exit is 1, so that an exit
# 2 of this wrapper always means "this shard needs more shards". (The merge has no canary: for it exit 2 is always a failure.)
CANARY_SEEN="$(tr '\r' '\n' < "${LOG}" | grep -aE "${LABEL_SHAPES}" | grep -c '^\[labels\] canary: ' || true)"
case "${CANARY_SEEN}" in ''|*[!0-9]*) CANARY_SEEN=0 ;; esac
if [ "${rc}" -eq 2 ] && [ "${CANARY_SEEN}" -ge 1 ]; then
  echo "=== the canary projects past the ${BUDGET_S} s budget: this shard cannot finish in its time limit, use more shards (SHARDS=16, --array=0-15) ==="
  exit 2
fi
if [ "${rc}" -ne 0 ]; then
  echo "ERROR labels exit=${rc}"
  if [ "${rc}" -eq 2 ]; then exit 1; fi
  exit "${rc}"
fi
if [ "${MODE}" = shard ]; then
  echo "=== END labels shard ${SLURM_ARRAY_TASK_ID} of ${SHARDS}: labelled or kept, wall_s=${SECONDS} ==="
else
  echo "=== END labels merge of ${SHARDS}: labelled and cross-checked, wall_s=${SECONDS} ==="
fi
