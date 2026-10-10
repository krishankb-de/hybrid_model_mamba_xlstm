#!/bin/bash
# ============================================================================
# CHAT_UI_PLAN.md P7-B (GPU, one H100, up to 24 h) — the chat server: the CheXbert labeller and the API in one SLURM job.
# The GPU twin of scripts/serve_chat_h100.sh: the same job with one GPU asked for (the SBATCH lines below) and DEVICE=cuda, so the API
# decodes on the GPU and the labeller stays on the CPU. On a GPU the uncached decode reproduces the published dump exactly (R2).
#   bash scripts/chat_remote.sh sync && bash scripts/chat_remote.sh submit scripts/serve_chat_gpu_h100.sh
#   bash scripts/chat_remote.sh submit scripts/serve_chat_gpu_h100.sh MODE=public # needs app_token in CHAT_HOME, mode 0600 (R6)
#   bash scripts/chat_remote.sh summary logs/chat_server_gpu_<jobid>.log
# Reach it from the laptop with app/tunnel/tunnel.sh: the node and the port are in $CHAT_HOME/endpoint, written once the API is serving.
# Data stays in $CHAT_HOME. Which of the two serves, this one or scripts/serve_chat_h100.sh on the CPU, is P1-D's decision (U7).
#
# Two processes, one job. The labeller is .venv_chexbert's uvicorn on app.labeler (its weights are in the default HF cache, so it runs with
# HF_HUB_OFFLINE=0 and without this job's HF_HOME, as scripts/score_chexbert_h100.sh does: D24), on a free loopback port. The API is
# `python -m app.server` (app/cli.py) in the main venv, with the web overlay .chat_deps on PYTHONPATH and offline Hugging Face. The shared
# venvs are never written to (R8).
#
# Never blocked on CheXbert: the job waits up to LABELER_WAIT_S for the labeller's /healthz (it loads the model on the first call), and when
# it never answers, or never started, the API starts with --labeler none and the log says `=== labeller unavailable: labels skipped ===`.
# A gallery is not skipped that way: GALLERY (default ${CHAT_HOME}/gallery/g13d_m3_v1) must have a manifest.json whose R@k gate is equal,
# or the job refuses with an ERROR line; GALLERY=none serves without retrieval, on purpose. Labels still pending in the gallery are fine.
#
# R6: MODE=public, or a BIND other than 127.0.0.1, needs ${CHAT_HOME}/app_token, a file of mode 0600 that you own. Its text is never read
# here and never printed: the API gets the file's path (--token-file), so the token is in no argument list, no environment and no log.
# A token file that is there is used in private mode too.
#
# SIGTERM (a requeue, a cancel, the time limit; SIGKILL follows after KillWait) and SIGINT: the trap forwards SIGTERM to both processes and
# waits for them. The API stops accepting, ends a running turn as an error with server_restart, removes the endpoint file and exits 0.
# With --requeue and --open-mode=append the job comes back on another node and appends to the same log; the stored turn that was cut off
# stays an error, and the new API writes a new endpoint file.
#
# R7: the job log carries only === lines, [server] lines (the CLI's stdout) and ERROR lines, and none names a path, the endpoint, the token
# or any report text. Everything else goes to files in ${CHAT_HOME}/logs: the API's stderr (uvicorn's log, the loader's prints, a traceback)
# and the shell's own complaints to server_<jobid>.log, both streams of the labeller to labeler_<jobid>.log. Read the job with the
# `summary` line above. A normal log:
#   === sync <40-hex commit> <clean|dirty> ===
#   === chat server: node=<node> mode=<private|public> bind=<address> device=<cpu|cuda> gallery=<build id|none> job=<id> ===
#   === labeller up ===                    (or: === labeller unavailable: labels skipped ===)
#   [server] starting: ...  /  [server] ... lines of the app  /  [server] serving: retrieval=on labels=on published=on
#   === signal received: stopping ===  [server] stopping  [server] stopped  === chat server stopped ===
#
# Levers, as NAME=value arguments of `chat_remote.sh submit`: MODE (private|public), BIND, GALLERY (a build directory, or none), MODELS
# (m3 or m3,13d), DRIFT_NOTE (the P1-D sentence every card carries), CHAT_HOME, LABELER_WAIT_S (default 180), CHEXBERT_HF_HUB_OFFLINE
# (default 0), PUBLISHED_MODEL and PUBLISHED_FLOOR (the published dumps, so that a test study's score shows the published line; both are
# looked for under results/, and when one is missing the line is skipped, which the log says).
# ============================================================================
#SBATCH --partition=pot-hpi-aisc-batch
#SBATCH --account=aisc
#SBATCH --gpus=1
#SBATCH --exclude=ga03,gx17v1,gx13v1   # ga03: ARM node, x86 .venv incompatible; gx13v1: faulty GPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --open-mode=append   # without append, a requeue truncates the log
#SBATCH --job-name=chat_server_gpu
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

# Every line printed below starts with === or ERROR and names no path, port or token (R7). ERROR_SHOWN lets the EXIT trap tell a refusal
# that has said why from a script that stopped without a word.
ERROR_SHOWN=0
say_error() { ERROR_SHOWN=1; echo "ERROR $*"; }
fail() { say_error "$@"; exit 1; }

SCRATCH_ROOT="${SCRATCH_ROOT:-/sc/scratch/$USER/hybrid_xmamba_h100}"
VENV_ACTIVATE="${VENV_ACTIVATE:-.venv/bin/activate}"
CHAT_HOME="${CHAT_HOME:-$HOME/chat_sessions}"
MODE="${MODE:-private}"
BIND="${BIND:-127.0.0.1}"
DEVICE="${DEVICE:-cuda}"
GALLERY="${GALLERY:-${CHAT_HOME}/gallery/g13d_m3_v1}"
TOKEN_FILE="${CHAT_HOME}/app_token"
LABELER_WAIT_S="${LABELER_WAIT_S:-180}"
PUBLISHED_MODEL="${PUBLISHED_MODEL:-results/report_gen_m3_test_split_s42}"
PUBLISHED_FLOOR="${PUBLISHED_FLOOR:-results/retrieval_floor_test_split}"
SERVER_LOG="${CHAT_HOME}/logs/server_${SLURM_JOB_ID:-local}.log"
LABELER_LOG="${CHAT_HOME}/logs/labeler_${SLURM_JOB_ID:-local}.log"
export HF_HOME="${SCRATCH_ROOT}/.hf" HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# Every refusal, before anything is started.
case "${MODE}" in private|public) ;; *) fail "MODE must be private or public" ;; esac
[ -f "${VENV_ACTIVATE}" ] || fail "the venv's activate script is missing: run chat_cluster_setup_h100.sh first"
mkdir -p "${CHAT_HOME}" && chmod 700 "${CHAT_HOME}"
mkdir -p "${CHAT_HOME}/logs"
exec 2>> "${SERVER_LOG}"
HAVE_TOKEN=0
if [ -s "${TOKEN_FILE}" ]; then
  python3 -c 'import os, stat, sys; st = os.stat(sys.argv[1]); sys.exit(0 if stat.S_ISREG(st.st_mode) and stat.S_IMODE(st.st_mode) == 0o600 and st.st_uid == os.getuid() else 1)' "${TOKEN_FILE}" \
    || fail "the token file must be a regular file of mode 0600 that you own (R6)"
  HAVE_TOKEN=1
fi
if { [ "${MODE}" = "public" ] || [ "${BIND}" != "127.0.0.1" ]; } && [ "${HAVE_TOKEN}" -eq 0 ]; then
  fail "public mode or a bind off loopback needs a token: put one in app_token under CHAT_HOME, mode 0600 (R6)"
fi
if [ "${GALLERY}" = "none" ]; then
  GALLERY=""
  GALLERY_NAME="none"
  echo "=== no gallery: retrieval is off ==="
else
  [ -f "${GALLERY}/manifest.json" ] || fail "the gallery has no manifest.json: build it first (P5-B)"
  python3 -c 'import json, sys; g = json.load(open(sys.argv[1])).get("gate_rk"); sys.exit(0 if isinstance(g, dict) and g.get("equal") is True else 1)' "${GALLERY}/manifest.json" 2>/dev/null \
    || fail "the gallery's manifest.json is unreadable or its R@k gate is not equal: it cannot be served"
  GALLERY_NAME="$(basename "${GALLERY}")"
fi
HAVE_PUBLISHED=0
if [ -f "${PUBLISHED_MODEL}/hyps.txt" ] && [ -f "${PUBLISHED_FLOOR}/hyps.txt" ]; then
  HAVE_PUBLISHED=1
else
  echo "=== published dumps not found: the published line is skipped ==="
fi
echo "=== chat server: node=$(hostname) mode=${MODE} bind=${BIND} device=${DEVICE} gallery=${GALLERY_NAME} job=${SLURM_JOB_ID:-?} ==="

# The two children. The trap is in place before the first one starts. A signal forwards SIGTERM to both and the main flow ends the job;
# the EXIT trap does the same for any other end, a `set -e` failure included, and says so in one ERROR line when nothing else has.
STOPPING=0
LABELER_PID=""
API_PID=""
stop_children() {
  local pid
  for pid in "${API_PID}" "${LABELER_PID}"; do
    [ -z "${pid}" ] || kill -TERM "${pid}" 2>/dev/null || true
  done
}
stop_and_wait() {
  local pid
  stop_children
  for pid in "${API_PID}" "${LABELER_PID}"; do
    [ -z "${pid}" ] || { wait "${pid}"; } 2>/dev/null || true
  done
  API_PID=""
  LABELER_PID=""
}
stopped() {
  stop_and_wait
  echo "=== chat server stopped ==="
  exit 0
}
finish() {
  local rc=$?
  stop_and_wait
  if [ "${rc}" -ne 0 ] && [ "${ERROR_SHOWN}" -eq 0 ]; then
    echo "ERROR the serve job ended with exit code ${rc} (see the server log under CHAT_HOME)"
  fi
}
trap 'STOPPING=1; echo "=== signal received: stopping ==="; stop_children' TERM INT
trap finish EXIT

# The labeller: loopback only, on a free port (nodes are shared, so a fixed one may be taken). A missing venv is a labeller that is down.
LABELER_PORT="${LABELER_PORT:-$(python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')}"
LABELER_UP=0
if [ -x .venv_chexbert/bin/python ]; then
  env -u HF_HOME HF_HUB_OFFLINE="${CHEXBERT_HF_HUB_OFFLINE:-0}" PYTHONPATH=.chat_deps_chexbert \
    .venv_chexbert/bin/python -m uvicorn app.labeler:app --host 127.0.0.1 --port "${LABELER_PORT}" >> "${LABELER_LOG}" 2>&1 &
  LABELER_PID=$!
  LABELER_DEADLINE=$((SECONDS + LABELER_WAIT_S))
  while [ "${STOPPING}" -eq 0 ] && [ "${SECONDS}" -lt "${LABELER_DEADLINE}" ] && kill -0 "${LABELER_PID}" 2>/dev/null; do
    if python3 -c 'import sys, urllib.request; sys.exit(0 if urllib.request.urlopen(sys.argv[1], timeout=5).status == 200 else 1)' \
         "http://127.0.0.1:${LABELER_PORT}/healthz" > /dev/null 2>&1; then
      LABELER_UP=1
      break
    fi
    sleep 0.5
  done
fi
[ "${STOPPING}" -eq 0 ] || stopped
if [ "${LABELER_UP}" -eq 1 ]; then
  echo "=== labeller up ==="
else
  echo "=== labeller unavailable: labels skipped ==="
  stop_and_wait
fi

# The API. Its stdout is the job log (the CLI prints only [server] and ERROR lines there), its stderr is the server log.
API_ARGS=(--engine real --device "${DEVICE}" --mode "${MODE}" --home "${CHAT_HOME}" --host "${BIND}" --port 0)
API_ARGS+=(--endpoint-file "${CHAT_HOME}/endpoint" --threads "${SLURM_CPUS_PER_TASK:-8}" --models "${MODELS:-m3}" --drift-note="${DRIFT_NOTE:-}")
[ -z "${GALLERY}" ] || API_ARGS+=(--gallery "${GALLERY}")
[ "${HAVE_PUBLISHED}" -eq 0 ] || API_ARGS+=(--published-model "${PUBLISHED_MODEL}" --published-floor "${PUBLISHED_FLOOR}")
[ "${HAVE_TOKEN}" -eq 0 ] || API_ARGS+=(--token-file "${TOKEN_FILE}")
if [ "${LABELER_UP}" -eq 1 ]; then API_ARGS+=(--labeler "http://127.0.0.1:${LABELER_PORT}"); else API_ARGS+=(--labeler none); fi
export PYTHONPATH=.chat_deps
source "${VENV_ACTIVATE}"
# A signal that came while the arguments were made, or in the venv's activate script, is looked for here: no API is started after it. One
# in the microseconds between this line and API_PID being set finds nothing to forward to (and forwarding at launch would not work: bash
# loses a signal sent to a child it has only just forked), but SLURM signals every process of the job itself and follows with SIGKILL.
[ "${STOPPING}" -eq 0 ] || stopped
python -m app.server "${API_ARGS[@]}" &
API_PID=$!

# Until the API ends. A trapped signal ends a `wait` early, with the API still draining: wait again, until it is really gone. The status
# kept is that of the last wait, which is the API's own.
API_RC=0
while kill -0 "${API_PID}" 2>/dev/null; do
  API_RC=0
  { wait "${API_PID}"; } 2>/dev/null || API_RC=$?
done
API_PID=""
stop_and_wait
[ "${STOPPING}" -eq 0 ] || stopped
if [ "${API_RC}" -eq 0 ]; then
  echo "=== chat server ended ==="
  exit 0
fi
say_error "the API exited with code ${API_RC} (see the server log under CHAT_HOME)"
exit "${API_RC}"
