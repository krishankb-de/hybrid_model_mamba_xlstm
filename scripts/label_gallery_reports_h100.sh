#!/bin/bash
# ============================================================================
# CHAT_UI_PLAN.md P5-C (1 x H100, at most 2 h): CheXbert-14 labels for the report rows of the retrieval gallery (plan section 6.5: labels.npy
# and label_names.json in ${CHAT_HOME}/gallery/<build id>), one get_label per duplicate group with the result broadcast to the group's rows,
# then cross-checked against the published labels of the 2,663 test references. The gallery is the one
# scripts/build_retrieval_gallery_h100.sh built; its gate_rk.equal must be true.
#   bash scripts/chat_remote.sh sync && bash scripts/chat_remote.sh submit scripts/label_gallery_reports_h100.sh BUILD_ID=g13d_m3_v1
#   (the gallery is named by BUILD_ID, default g13d_m3_v1, the plan's; the published dump by REFERENCE_DIR, default
#   results/report_gen_m3_test_split_s42; the seconds the canary may project by BUDGET_S, default 5400; how many groups it labels first by
#   CANARY, default 1000: all as NAME=value arguments of `chat_remote.sh submit`)
#
# WHY A GPU JOB WITH A CANARY. CheXbert on a CPU takes about 0.16 s a report (job 2525606): 7-8 h for the 150-190k duplicate groups. f1chexbert
# 0.0.2 takes the GPU itself when torch sees one (read from its source: F1CheXbert(device=None) is cuda when torch.cuda.is_available()), so with a
# CUDA torch this job would take 15-45 min. THIS venv may have none: scripts/setup_chexbert_venv_h100.sh installs torch from the CPU index ("CPU-only
# torch is enough"), and then no GPU, however many the job holds, is ever used. So the job asks the venv's own torch first. With no GPU visible to it,
# it says so, prints the two submit lines of the sharded CPU path (below) and exits 2 within seconds, having labelled nothing and written nothing.
# With one, scripts/label_gallery_reports.py labels the first CANARY groups, prints the rate and what it projects for all of them, and exits 2 if
# that is past BUDGET_S (5400 s of the 2 h: loading and the check need little of the rest). This wrapper reads exit 2 as "use the sharded CPU
# path": it prints the two submit lines for it (8 CPU shards of about 1 h, each its own job of scripts/label_gallery_reports_cpu_h100.sh, then
# the merge) and exits 2. Nothing has been written then but the raw log.
#   Hugging Face. CHAT_UI_PLAN.md D24: the CheXbert weights live in the DEFAULT Hugging Face cache, so HF_HOME is not overridden here (the other chat
#   jobs set it to the cache under SCRATCH_ROOT, which does not hold them), and HF_HUB_OFFLINE defaults to 0 exactly as in score_chexbert_h100.sh,
#   the wrapper that scored the published dumps. Override it to 1 from outside if you want to be sure no network call is made.
#
# It refuses (an `ERROR ...` line, exit 1) when: BUILD_ID is not one plain name, or BUDGET_S or CANARY is not a whole number; CHAT_HOME, the gallery or its
# manifest.json is missing; the gallery resolves into an outputs directory or under the thesis checkout (R8); the gallery is already labelled
# (labels_status done: a finished labelling is never overwritten, and this is checked before the raw log is opened, so a finished job's log is not
# replaced); the CheXbert venv, or the reference dump's refs.txt or chexbert_labels.json, is missing. The script itself then refuses a gallery whose
# gate_rk.equal is not true in both manifest.json and gate_rk.json, and a reference that does not match the gallery, before the labeller is built; and
# its cross-check, run last, fails the job (exit 1, labels_status stays pending, the labels kept as labels_unverified.npy) on any difference. A partial run
# (shards of the CPU path, an earlier cross-check that failed) may be completed: labels_status is then still pending. A requeue starts the labelling
# again and replaces the raw log.
# R7: the job log carries only === lines that name no path (the two submit lines of an exit 2 name the CPU wrapper, which is repo code),
# [labels], RESULT and ERROR lines of the shapes in LABEL_SHAPES below (an allowlist: an id or a piece of a report on a [labels] line is no known shape,
# whatever the script or the libraries print, and `=== label lines withheld: N ===` counts what was not shown), and ERROR lines with a class name or
# an exit code. The raw stdout and stderr (the library's warnings, a traceback that echoes report text) go to ${GALLERY}/labels.log and are never
# printed. Read the job with `bash scripts/chat_remote.sh summary logs/chat_labels_<jobid>.log`.
# Output (MIMIC-derived, Class R: stays on the cluster, never committed, never copied to the laptop), in ${CHAT_HOME}/gallery/<build id>/:
#   labels.npy, label_names.json, labels_check.json, labels.log, and labels_status in manifest.json
# ============================================================================
#SBATCH --partition=pot-hpi-aisc-batch
#SBATCH --account=aisc
#SBATCH --gpus=1
#SBATCH --exclude=ga03,gx17v1,gx13v1   # ga03: ARM node, x86 venv incompatible; gx13v1: faulty GPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --job-name=chat_labels
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
BUDGET_S="${BUDGET_S:-5400}"
CANARY="${CANARY:-1000}"
# One name, never a path: GALLERY is CHAT_HOME/gallery/<name> and nothing else. The two numbers go onto the script's command line.
[[ "${BUILD_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || fail "BUILD_ID must be one plain name: letters, digits, dot, dash, underscore"
[[ "${BUDGET_S}" =~ ^[0-9]{1,6}$ ]] || fail "BUDGET_S must be a whole number of seconds, at most 6 digits"
[[ "${CANARY}" =~ ^[0-9]{1,7}$ ]] || fail "CANARY must be a whole number of groups, at most 7 digits"
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
^\[labels\] labels_status=(done|pending)$
^RESULT \{"refs_mismatch":[0-9]{1,7},"labels_mismatch":[0-9]{1,7}\}$
^RESULT \{"groups":[0-9]{1,7},"rows":[0-9]{1,7},"labelling_s":[0-9]{1,7}\}$
^RESULT \{"shard":[0-9]{1,3},"of":[0-9]{1,3},"groups":[0-9]{1,7},"labelling_s":[0-9]{1,7}\}$
^RESULT \{"merged":[0-9]{1,3},"groups":[0-9]{1,7},"rows":[0-9]{1,7}\}$
^ERROR (no_manifest|manifest_unreadable|gate_not_equal|already_labelled|inputs_unreadable|rows_disagree|groups_disagree|test_rows|reference_unreadable|reference_shape|reference_label_names|refs_mismatch|no_device_argument|label_order|bad_label_row|shards_missing|shard_invalid)( [a-z_]{1,20}=[0-9]{1,7}){0,3}$
^ERROR failed [A-Za-z_][A-Za-z0-9_]{0,59}$
^=== note: chexbert_labels\.json has no label_names key: CHEXBERT_14 order assumed ===$
^=== note: chexbert_labels\.json label_names reordered to the CHEXBERT_14 order ===$'

echo "=== P5-C CheXbert labels for gallery ${BUILD_ID}: one job, F1CheXbert on the GPU if it takes one, canary budget ${BUDGET_S} s ==="
echo "=== job=${SLURM_JOB_ID:-local} restart=${SLURM_RESTART_COUNT:-0} node=$(hostname) ==="

# --- guards: no step starts, and the gallery is not touched, before every one of them has passed ------------------------
# CHAT_HOME is made by scripts/chat_cluster_setup_h100.sh, owner-only, and not here; the gallery is the build's, and not made here either.
[ -d "${CHAT_HOME}" ] || fail "CHAT_HOME does not exist: run chat_cluster_setup_h100.sh first"
[ -f "${GALLERY}/manifest.json" ] || fail "the gallery has no manifest.json: build it first (build_retrieval_gallery_h100.sh)"
[ -f "${VENV_ACTIVATE}" ] || fail "the CheXbert venv is missing: run setup_chexbert_venv_h100.sh first"

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
# A finished labelling is never overwritten, and neither is the raw log of the job that finished it: refused here, before the log is opened.
# (The script checks the same from the manifest's JSON.)
if grep -Eq '"labels_status":[[:space:]]*"done"' "${GALLERY}/manifest.json"; then
  fail "the gallery is already labelled (labels_status done): a finished labelling is never overwritten (R8)"
fi
[ -f "${REFERENCE_DIR}/refs.txt" ] || fail "the reference dump has no refs.txt"
[ -f "${REFERENCE_DIR}/chexbert_labels.json" ] || fail "the reference dump has no chexbert_labels.json"

# F1CheXbert takes the GPU only if THIS venv's torch sees one (see the header): the count is asked of the venv's own torch, not of nvidia-smi.
GPU_INFO="$(python -c 'import torch; n = torch.cuda.device_count(); print(n, torch.cuda.get_device_name(0) if n else "none")' 2>/dev/null)" || GPU_INFO="0 none"
GPUS="${GPU_INFO%% *}"
case "${GPUS}" in ''|*[!0-9]*) GPUS=0 ;; esac
echo "=== gpus=${GPUS} (${GPU_INFO#* }) ==="
# The two submit lines of the sharded CPU path: what an exit 2 leaves for whoever reads the log, whether the canary projected past the budget or
# torch could not have used the GPU at all.
use_the_cpu_path() {
  echo "=== use the sharded CPU path, submit these two in order ==="
  echo "=== 1. bash scripts/chat_remote.sh submit scripts/label_gallery_reports_cpu_h100.sh BUILD_ID=${BUILD_ID} -- --array=0-7 ==="
  echo "=== 2. bash scripts/chat_remote.sh submit scripts/label_gallery_reports_cpu_h100.sh BUILD_ID=${BUILD_ID} MERGE=1 -- --dependency=afterok:<the array job id> ==="
}
if [ "${GPUS}" -lt 1 ]; then
  echo "=== torch in this venv sees no GPU, so F1CheXbert would run on the CPU: the one-job path is skipped, and nothing was labelled ==="
  use_the_cpu_path
  exit 2
fi

# --- the labelling -------------------------------------------------------------------------------------------------------
# Only the script's lines of a known shape leave labels.log (a progress bar can sit in front of one on the same physical line, hence the tr);
# the lines that wear one of our prefixes and have none of the shapes are counted, not shown. A requeue replaces labels.log.
echo "=== labelling: the script's own output goes to labels.log in the gallery and is never printed ==="
SECONDS=0
rc=0
python scripts/label_gallery_reports.py --gallery "${GALLERY}" --reference-dir "${REFERENCE_DIR}" --budget-s "${BUDGET_S}" \
  --canary "${CANARY}" --device auto > "${GALLERY}/labels.log" 2>&1 || rc=$?
tr '\r' '\n' < "${GALLERY}/labels.log" | grep -aE "${LABEL_SHAPES}" | tail -n 60 || true
WITHHELD="$(tr '\r' '\n' < "${GALLERY}/labels.log" | grep -aE '^(\[labels\] |RESULT |ERROR|=== )' | grep -avcE "${LABEL_SHAPES}" || true)"
case "${WITHHELD}" in ''|*[!0-9]*) WITHHELD=unknown ;; esac
echo "=== label lines withheld: ${WITHHELD} ==="
if [ "${rc}" -eq 2 ]; then
  echo "=== the canary projects past the ${BUDGET_S} s budget: F1CheXbert is too slow here for one job ==="
  use_the_cpu_path
  exit 2
fi
if [ "${rc}" -ne 0 ]; then echo "ERROR labels exit=${rc}"; exit "${rc}"; fi
echo "=== END labels ${BUILD_ID}: labelled and cross-checked, wall_s=${SECONDS} ==="
