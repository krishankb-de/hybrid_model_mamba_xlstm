#!/bin/bash
# ============================================================================
# CHAT_UI_PLAN.md P5-B (GPU, 1 x H100, about 1 h): the retrieval gallery the chat UI looks neighbours up in (plan section 6.5),
# built through the retrieval chapter's own loaders, with the R@k gate inside the job.
#   bash scripts/chat_remote.sh sync && bash scripts/chat_remote.sh submit scripts/build_retrieval_gallery_h100.sh
#   (the report model, its config and the build name can be set: MODEL_CONFIG=... CHECKPOINT=... BUILD_ID=... as NAME=value
#   arguments of `chat_remote.sh submit`; the 13D checkpoint with CKPT_13D=..., the dataset directory with DATA=...)
#
# Three steps, one job:
#   1. scripts/build_retrieval_gallery.py encodes the official train (191,462) and test (2,663) splits through
#      scripts/evaluate_cxr_retrieval.py's own load_models, build_dataloader and encode_dataset at batch 32, hashes the 13D tower
#      and the report model's tower (towers_identical), and writes the gallery files and gate_rk.app, that script's own
#      compute_retrieval_metrics on the test split.
#   2. scripts/evaluate_cxr_retrieval.py runs UNCHANGED (R3), on the same 13D checkpoint and the test split: the published protocol.
#   3. scripts/build_retrieval_gallery.py --compare-rk: every i2t and t2i R@k of step 1 must equal step 2's, to every digit. When
#      they do not, the job ends with the verdict in its log and exit 1, and the gallery must not be used.
#
# R8: the gallery goes to ${CHAT_HOME}/gallery/<build id>, nowhere else. OUT is derived, never an environment lever (sbatch exports
# the submitting shell, and OUT is a common name); the build id must be one plain name. ./outputs is a symlink into the thesis
# checkout, so the wrapper refuses an OUT inside an outputs directory or under that checkout. A finished build (its manifest.json,
# which the builder writes last) is never overwritten; a requeued, unfinished one starts again in the same directory (the build id
# defaults to the date and the job id, which a requeue keeps) and replaces its own partial files and raw logs.
# R7: the job log carries only === lines that name no path, [gallery] lines (the builder's counts, booleans and hashes, filtered
# here so that none can carry a path), the one RESULT line of the comparison, and ERROR lines with a step and an exit code. The raw
# stdout and stderr of the builder and of the thesis script, which show paths and can show report text, go to ${OUT}/build.log and
# ${OUT}/reference_rk.log and are never printed. Read the job with
# `bash scripts/chat_remote.sh summary logs/chat_gallery_<jobid>.log`. Its mask blanks every run of 8 or more digits, so the date
# in the default build id, and now and then a stretch of a hash, shows as <num>: towers_identical is the evidence, the hashes are
# in manifest.json, and BUILD_ID=<name> gives a directory name the mask leaves alone.
# Output (MIMIC-derived, Class R: stays on the cluster, never committed, never copied to the laptop):
#   CHAT_HOME/gallery/<build id>/
# ============================================================================
#SBATCH --partition=pot-hpi-aisc-batch
#SBATCH --account=aisc
#SBATCH --gpus=1
#SBATCH --exclude=ga03,gx17v1,gx13v1   # ga03: ARM node, x86 .venv incompatible; gx13v1: faulty GPU
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=chat_gallery
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

# Every line printed below starts with ===, [gallery], RESULT or ERROR (the shapes `chat_remote.sh summary` shows) and names no path.
fail() { echo "ERROR $*"; exit 1; }
# `readlink -f` needs the path to exist and is not the same everywhere; python's realpath is.
realpath_py() { python -c 'import os, sys; print(os.path.realpath(sys.argv[1]))' "$1"; }

SCRATCH_ROOT="${SCRATCH_ROOT:-/sc/scratch/$USER/hybrid_xmamba_h100}"
VENV_ACTIVATE="${VENV_ACTIVATE:-.venv/bin/activate}"
CHAT_HOME="${CHAT_HOME:-/sc/home/$USER/chat_sessions}"
DATA="${DATA:-/sc/home/$USER/dataset/mimic_full}"
CKPT_13D="${CKPT_13D:-./outputs/h100_kd_150m_v2_full_data_lr3e6/checkpoints/last.ckpt}"
CHECKPOINT="${CHECKPOINT:-./outputs/h100_report_gen_m3_tower13d_s42/checkpoints/last.ckpt}"
MODEL_CONFIG="${MODEL_CONFIG:-hybrid_150m_m3_rrg}"
BUILD_ID="${BUILD_ID:-$(date +%Y%m%d)_${SLURM_JOB_ID:-local}}"
# One name, never a path: OUT is CHAT_HOME/gallery/<name> and nothing else.
[[ "${BUILD_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || fail "BUILD_ID must be one plain name: letters, digits, dot, dash, underscore"
OUT="${CHAT_HOME}/gallery/${BUILD_ID}"

echo "=== P5-B gallery ${BUILD_ID}: 13D tower and text encoder, report model ${MODEL_CONFIG}, official train and test splits, batch 32, 1 GPU ==="
echo "=== job=${SLURM_JOB_ID:-local} restart=${SLURM_RESTART_COUNT:-0} node=$(hostname) ==="

export HF_HOME="${SCRATCH_ROOT}/.hf"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
source "${VENV_ACTIVATE}"

# --- guards: no directory is created and no job step starts before the path guards have passed ----------------------
MAIN_REAL="$(dirname "$(realpath_py outputs)")"       # the thesis checkout: ./outputs is a symlink into it
OUT_REAL="$(realpath_py "${OUT}")"
case "${OUT}/" in */outputs/*) fail "OUT is inside an outputs directory" ;; esac
case "${OUT_REAL}/" in
  */outputs/*) fail "OUT resolves into an outputs directory" ;;
  "${MAIN_REAL}"/*) fail "OUT is under the thesis checkout (R8)" ;;
esac
# CHAT_HOME is made by scripts/chat_cluster_setup_h100.sh, owner-only, and not here.
[ -d "${CHAT_HOME}" ] || fail "CHAT_HOME does not exist: run chat_cluster_setup_h100.sh first"
if [ -e "${OUT}/manifest.json" ]; then
  fail "${BUILD_ID} is already built (manifest.json exists): a finished build is never overwritten"
fi

GPU_INFO="$(python -c 'import torch; n = torch.cuda.device_count(); print(n, torch.cuda.get_device_name(0) if n else "none")' 2>/dev/null)" || GPU_INFO="0 none"
GPUS="${GPU_INFO%% *}"
[ "${GPUS}" -ge 1 ] || fail "${GPUS} GPU(s) visible, 1 needed (the sbatch header asks for --gpus=1)"
echo "=== gpus=${GPUS} (${GPU_INFO#* }) ==="
[ -f "${CKPT_13D}" ] || fail "13D checkpoint (the retrieval tower and text encoder) not found"
[ -f "${CHECKPOINT}" ] || fail "decoder checkpoint not found"
[ -f "${DATA}/train.parquet" ] || fail "train.parquet not found in DATA"
[ -f "${DATA}/test.parquet" ] || fail "test.parquet not found in DATA"

if [ -d "${OUT}" ]; then
  echo "=== ${BUILD_ID}: an earlier attempt left files here and did not finish: its partial files are overwritten ==="
fi
mkdir -p "${OUT}"

# --- step 1: the gallery -----------------------------------------------------------------------------------------
# Only the builder's [gallery] lines leave build.log (a progress bar can sit in front of one on the same physical line, hence
# the tr), and only those without a slash: a path in one is dropped here, whatever the builder does.
echo "=== step 1/3: the gallery; raw output goes to build.log and is never printed ==="
SECONDS=0
rc=0
python scripts/build_retrieval_gallery.py \
  --checkpoint-13d "${CKPT_13D}" --decoder-checkpoint "${CHECKPOINT}" --decoder-config "${MODEL_CONFIG}" \
  --data "${DATA}" --out "${OUT}" --build-id "${BUILD_ID}" --workers "${SLURM_CPUS_PER_TASK:-8}" \
  --isbi-cache "${SCRATCH_ROOT}/isbi_gallery_adapted.pt" > "${OUT}/build.log" 2>&1 || rc=$?
tr '\r' '\n' < "${OUT}/build.log" | grep -aE '^\[gallery\] ' | grep -av '/' | tail -n 60 || true
if [ "${rc}" -ne 0 ]; then echo "ERROR build exit=${rc}"; exit "${rc}"; fi

# --- step 2: the reference R@k: the thesis script, unchanged, on the same checkpoint and the official test split --------
echo "=== step 2/3: the reference R@k (scripts/evaluate_cxr_retrieval.py, unchanged, test split); raw output goes to reference_rk.log ==="
rc=0
python scripts/evaluate_cxr_retrieval.py \
  --checkpoint "${CKPT_13D}" --dataset mimic --local-parquet-dir "${DATA}" --mimic-split test \
  --output-dir "${OUT}/reference_rk" > "${OUT}/reference_rk.log" 2>&1 || rc=$?
if [ "${rc}" -ne 0 ]; then echo "ERROR reference exit=${rc}"; exit "${rc}"; fi

# --- step 3: the gate -----------------------------------------------------------------------------------------------
# Its stdout is RESULT and ERROR lines of numbers and key names, cut down to those two shapes anyway; its stderr goes to a file.
echo "=== step 3/3: the gate: every i2t and t2i R@k of step 1 against step 2 ==="
rc=0
CMP_OUT="$(python scripts/build_retrieval_gallery.py --compare-rk "${OUT}" --wall-s "${SECONDS}" 2>> "${OUT}/compare.err")" || rc=$?
printf '%s\n' "${CMP_OUT}" | grep -aE '^(RESULT |ERROR)' || true
if [ "${rc}" -ne 0 ]; then echo "ERROR compare exit=${rc}"; exit "${rc}"; fi
echo "=== END ${BUILD_ID}: gallery built, gate equal, wall_s=${SECONDS} ==="
