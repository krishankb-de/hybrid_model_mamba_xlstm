#!/bin/bash
# ============================================================================
# CHAT_UI_PLAN.md P5-F (CPU, 8 CPUs, 32 GB, about 10 min): the gates of the live path, on the real gallery and the real labeller.
#   bash scripts/chat_remote.sh sync && bash scripts/chat_remote.sh submit scripts/chat_retrieval_gates_h100.sh
#   bash scripts/chat_remote.sh summary logs/chat_gates_<jobid>.log
#   (the gallery build, the report model and its config, the dataset and the published dump can be set as NAME=value arguments of
#   `chat_remote.sh submit`: BUILD_ID=..., CHECKPOINT=..., MODEL_CONFIG=..., DATA=..., REFERENCE_DIR=...; how long to wait for the labeller to
#   load, default 600 s, by LABELER_WAIT_S; CHEXBERT_HF_HUB_OFFLINE=1 sends the labeller offline, default 0 as in score_chexbert_h100.sh)
#
# Three checks, one job, one driver (scripts/chat_retrieval_gates.py, which says what each is in its docstring):
#   1. self-retrieval: 50 train images, through the live upload path (Engine.preprocess and encode from the file bytes), must each be
#      the first neighbour of themselves in the gallery;
#   2. live own-rank: the rank of a test study's own report for the live CPU vector against the rank for the build's own (H100)
#      embedding of the image, counted for `rank` and for `rank_dedup`, recorded and not gated (prediction: at least 48 of 50);
#   3. labeller: the labeller SERVICE (app.labeler, which this job starts) against the published labels, on the first 50 hyps and the
#      first 50 refs of the published dump (REFERENCE_DIR: chexbert_labels.json, hyps.txt, refs.txt).
# The job exits 1 unless self_retrieval is 50, labeller_equal is 100 and the label names are right. The gallery is opened against the
# engine's own tower hash, so the vectors it compares are the engine's tower's, or the job says `ERROR gallery tower mismatch`.
#
# The labeller. It is started here, exactly as the serve wrapper (P7-B) starts it: in .venv_chexbert (transformers<5) with its web overlay
# .chat_deps_chexbert, without HF_HOME and with HF_HUB_OFFLINE defaulting to 0, the CheXbert environment of score_chexbert_h100.sh
# (CHAT_UI_PLAN.md D24: the weights live in the default Hugging Face cache), on a free port of the loopback interface only. The first load
# can take minutes, so /healthz is waited for, one request at a time, for at most LABELER_WAIT_S seconds, and a labeller that dies meanwhile
# is reported with its exit status. It is stopped by a trap on every way out of this job (a refusal, a failure, a signal, the normal end):
# asked to leave, given three seconds, then killed. The driver runs in .venv with its overlay .chat_deps, offline, from the scratch Hugging
# Face cache (the engine needs no network, the labeller its own cache), one thread per CPU.
#
# Output (DUA-covered, Class R: stays on the cluster, never committed, never copied to the laptop), a new directory under results/:
#   results/chat_retrieval_gates_<job>/gates.json   numbers and indices only: the three counts, every miss (gallery row, row found, gap),
#                                                   the rows whose rank differs, the rows the labeller got wrong, seconds
#   results/chat_retrieval_gates_<job>/gates.log    the driver's raw stdout and stderr (a traceback can show a path or report text)
#   results/chat_retrieval_gates_<job>/labeler.log  the labeller's raw stdout and stderr (its access log: no text)
# R7: the job log carries only === lines, [gates] lines, RESULT and ERROR lines of the shapes in GATES_SHAPES below (an allowlist: an id,
# a path or a piece of a report on a line of the log is no known shape, whatever the driver or a library prints), and
# `=== gates lines withheld: N ===` counts what was not shown. The raw output is never printed. The job log, in full, when all is well:
#   === sync <40-hex commit> <clean|dirty> ===
#   === P5-F live-path gates for gallery <build id>: report model <config>, CPU, 8 threads ===
#   === job=<id> restart=0 node=<node> ===
#   === labeller: starting on a free loopback port ===
#   === labeller: ready after <n> s ===
#   === gates: the driver's own output goes to gates.log in the results directory and is never printed ===
#   [gates] engine device=cpu threads=8
#   [gates] gallery images=<n> report_rows=<n> report_groups=<n> towers_identical=true img_proj_present=false labels_status=done
#   [gates] self_retrieval hits=50 of=50 misses=0                  (a miss: [gates] miss row=<n> got=<n> gap=<similarity gap>, at most 20)
#   [gates] own_rank equal=<n> dedup_equal=<n> of=50 max_diff=<n>
#   [gates] labeller pred_equal=50 true_equal=50 of=50
#   RESULT {"self_retrieval":50,"own_rank_equal":<n>,"own_rank_dedup_equal":<n>,"labeller_equal":100,"label_names_ok":true}
#   === gates lines withheld: 0 ===
#   === END chat retrieval gates: all three gates held, wall_s=<n> ===
# A published chexbert_labels.json without a label_names key is read in CHEXBERT_14 order, and `=== note: ... assumed ===` says so. A gate
# that does not hold prints `ERROR gate <name>=<n> expected=<n>` after RESULT, then `ERROR gates exit=1`, and ends the job with exit 1.
# Read it with `bash scripts/chat_remote.sh summary logs/chat_gates_<jobid>.log`.
# ============================================================================
#SBATCH --partition=pot-hpi-aisc-batch
#SBATCH --account=aisc
#SBATCH --qos=aisc
#SBATCH --exclude=ga03,gx17v1,gx13v1   # ga03: ARM node, x86 venv incompatible; gx13v1: faulty GPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=chat_gates
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

# Every line printed below starts with ===, [gates], RESULT or ERROR (the shapes `chat_remote.sh summary` shows) and names no path.
fail() { echo "ERROR $*"; exit 1; }

SCRATCH_ROOT="${SCRATCH_ROOT:-/sc/scratch/$USER/hybrid_xmamba_h100}"
VENV_ACTIVATE="${VENV_ACTIVATE:-.venv/bin/activate}"
CHAT_HOME="${CHAT_HOME:-/sc/home/$USER/chat_sessions}"
DATA="${DATA:-/sc/home/$USER/dataset/mimic_full}"
CHECKPOINT="${CHECKPOINT:-./outputs/h100_report_gen_m3_tower13d_s42/checkpoints/last.ckpt}"
MODEL_CONFIG="${MODEL_CONFIG:-hybrid_150m_m3_rrg}"
BUILD_ID="${BUILD_ID:-g13d_m3_v1}"
REFERENCE_DIR="${REFERENCE_DIR:-results/report_gen_m3_test_split_s42}"
LABELER_WAIT_S="${LABELER_WAIT_S:-600}"
# Derived, never levers (sbatch exports the submitting shell, and OUT is a likely name): one name for the build, one directory per job.
[[ "${BUILD_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || fail "BUILD_ID must be one plain name: letters, digits, dot, dash, underscore"
[[ "${MODEL_CONFIG}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || fail "MODEL_CONFIG must be one plain name: letters, digits, dot, dash, underscore"
[[ "${LABELER_WAIT_S}" =~ ^[0-9]{1,5}$ ]] || fail "LABELER_WAIT_S must be a whole number of seconds, at most 5 digits"
GALLERY="${CHAT_HOME}/gallery/${BUILD_ID}"
OUT="results/chat_retrieval_gates_${SLURM_JOB_ID:-local}"
THREADS="${SLURM_CPUS_PER_TASK:-8}"

# The lines the driver prints that may reach this log, as one anchored pattern per line (grep -E reads a newline as "or"): digits are
# bounded, and nothing in a shape is free text, so an id, a path or a piece of a report on a [gates] line, or an exception message on an
# ERROR line, matches none of them, whatever is printed. tests/test_chat_retrieval_gates.py runs these very patterns, through grep, over
# everything the driver prints and over ids, report text and paths. No line may be empty: an empty pattern matches every line.
GATES_SHAPES='^\[gates\] engine device=(cpu|cuda|cuda:[0-9]{1,2}) threads=[0-9]{1,3}$
^\[gates\] gallery images=[0-9]{1,7} report_rows=[0-9]{1,7} report_groups=[0-9]{1,7} towers_identical=(true|false) img_proj_present=false labels_status=(done|pending)$
^\[gates\] self_retrieval hits=[0-9]{1,3} of=[0-9]{1,3} misses=[0-9]{1,3}$
^\[gates\] miss row=[0-9]{1,7} got=[0-9]{1,7} gap=[0-9]\.[0-9]{6}$
^\[gates\] own_rank equal=[0-9]{1,3} dedup_equal=[0-9]{1,3} of=[0-9]{1,3} max_diff=[0-9]{1,7}$
^\[gates\] labeller pred_equal=[0-9]{1,3} true_equal=[0-9]{1,3} of=[0-9]{1,3}$
^RESULT \{"self_retrieval":[0-9]{1,3},"own_rank_equal":[0-9]{1,3},"own_rank_dedup_equal":[0-9]{1,3},"labeller_equal":[0-9]{1,3},"label_names_ok":(true|false)\}$
^ERROR (gallery tower mismatch|gallery refused|published unreadable|labeller order mismatch|labeller unavailable)$
^ERROR (data rows disagree|published shape)( [a-z_]{1,20}=[0-9]{1,7}){1,4}$
^ERROR gate (self_retrieval|labeller_equal)=[0-9]{1,3} expected=[0-9]{1,3}$
^ERROR gate label_names_ok=false$
^ERROR failed [A-Za-z_][A-Za-z0-9_]{0,59}$
^=== note: chexbert_labels\.json has no label_names key: CHEXBERT_14 order assumed ===$
^=== note: chexbert_labels\.json label_names reordered to the CHEXBERT_14 order ===$'

echo "=== P5-F live-path gates for gallery ${BUILD_ID}: report model ${MODEL_CONFIG}, CPU, ${THREADS} threads ==="
echo "=== job=${SLURM_JOB_ID:-local} restart=${SLURM_RESTART_COUNT:-0} node=$(hostname) ==="

# --- guards: no process starts, and nothing is written, before every one of them has passed -----------------------------
# CHAT_HOME is made by scripts/chat_cluster_setup_h100.sh, owner-only, and not here; the gallery is the build's, and is only read.
[ -d "${CHAT_HOME}" ] || fail "CHAT_HOME does not exist: run chat_cluster_setup_h100.sh first"
[ -f "${GALLERY}/manifest.json" ] || fail "the gallery has no manifest.json: build it first (build_retrieval_gallery_h100.sh)"
grep -Eq '"equal":[[:space:]]*true' "${GALLERY}/manifest.json" || fail "the gallery's R@k gate is not decided equal: it cannot be used"
[ -f "${GALLERY}/test_img_emb.npy" ] || fail "the gallery has no test_img_emb.npy: it is the build's own embedding of the test images"
# results is a symlink into the thesis checkout (P0-G): a new results/chat_* directory may be made through it, and results itself never is.
[ -d results ] || fail "results is missing: run chat_cluster_setup_h100.sh first"
[ -f "${VENV_ACTIVATE}" ] || fail "the main venv is missing: run chat_cluster_setup_h100.sh first"
[ -x .venv_chexbert/bin/python ] || fail "the CheXbert venv is missing: run setup_chexbert_venv_h100.sh first"
[ -f .chat_deps/.setup_ok ] && [ -f .chat_deps_chexbert/.setup_ok ] || fail "the web overlays are not set up: run chat_cluster_setup_h100.sh first"
# Python answers a script it cannot find with exit 2: say what is wrong instead.
[ -f scripts/chat_retrieval_gates.py ] || fail "chat_retrieval_gates.py is missing from this tree: run chat_remote.sh sync first"
[ -f "${CHECKPOINT}" ] || fail "the report model's checkpoint was not found"
[ -f "${DATA}/train.parquet" ] || fail "train.parquet was not found in DATA"
[ -f "${DATA}/test.parquet" ] || fail "test.parquet was not found in DATA"
[ -f "${REFERENCE_DIR}/hyps.txt" ] || fail "the published dump has no hyps.txt"
[ -f "${REFERENCE_DIR}/refs.txt" ] || fail "the published dump has no refs.txt"
[ -f "${REFERENCE_DIR}/chexbert_labels.json" ] || fail "the published dump has no chexbert_labels.json"

# The engine's environment: compute nodes are offline, the Hugging Face cache is the scratch one. The labeller is started below without
# HF_HOME and with its own HF_HUB_OFFLINE, and an overlay is put on the PYTHONPATH of the one command that needs it, never on the shell.
source "${VENV_ACTIVATE}"
export HF_HOME="${SCRATCH_ROOT}/.hf" HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${THREADS}" MKL_NUM_THREADS="${THREADS}"
# The labeller is on the loopback interface: no proxy of the cluster's may stand between the driver and it.
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,${NO_PROXY}}" no_proxy="127.0.0.1,localhost${no_proxy:+,${no_proxy}}"
mkdir -p "${OUT}" 2>/dev/null || fail "the results directory cannot be made"

# --- the labeller --------------------------------------------------------------------------------------------------------
# Stopped on every way out: asked to leave, given three seconds, then killed (a uvicorn that is waiting for a model to load does not leave
# on SIGTERM). Safe to call twice.
LABELER_PID=""
stop_labeller() {
  [ -n "${LABELER_PID}" ] || return 0
  kill "${LABELER_PID}" 2>/dev/null || true
  for _ in 1 2 3; do
    kill -0 "${LABELER_PID}" 2>/dev/null || break
    sleep 1
  done
  if kill -0 "${LABELER_PID}" 2>/dev/null; then kill -KILL "${LABELER_PID}" 2>/dev/null || true; fi
  wait "${LABELER_PID}" 2>/dev/null || true
  LABELER_PID=""
}
trap stop_labeller EXIT
trap 'stop_labeller; exit 143' TERM
trap 'stop_labeller; exit 130' INT

LABELER_PORT="$(python -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')" || fail "no free loopback port"
case "${LABELER_PORT}" in ''|*[!0-9]*) fail "no free loopback port" ;; esac
LABELER_URL="http://127.0.0.1:${LABELER_PORT}"
echo "=== labeller: starting on a free loopback port ==="
env -u HF_HOME HF_HUB_OFFLINE="${CHEXBERT_HF_HUB_OFFLINE:-0}" PYTHONPATH=.chat_deps_chexbert .venv_chexbert/bin/python \
  -m uvicorn app.labeler:app --host 127.0.0.1 --port "${LABELER_PORT}" > "${OUT}/labeler.log" 2>&1 &
LABELER_PID=$!

# Wait for /healthz: the first request loads the model and answers when it has (a 503 means it could not load; the next request tries
# again). One request at a time, in the background and waited for, so that a signal is acted on at once; each is cut at the time that is left.
HEALTH_PROBE='import sys, urllib.request; urllib.request.urlopen(sys.argv[1] + "/healthz", timeout=float(sys.argv[2])).read()'
READY=0
WAITED_FROM="${SECONDS}"
DEADLINE=$((SECONDS + LABELER_WAIT_S))
while [ "${SECONDS}" -lt "${DEADLINE}" ]; do
  kill -0 "${LABELER_PID}" 2>/dev/null || break
  LEFT=$((DEADLINE - SECONDS))
  [ "${LEFT}" -le 60 ] || LEFT=60
  python -c "${HEALTH_PROBE}" "${LABELER_URL}" "${LEFT}" > /dev/null 2>&1 &
  PROBE_PID=$!
  if wait "${PROBE_PID}"; then READY=1; break; fi
  sleep 1
done
if [ "${READY}" -ne 1 ]; then
  if kill -0 "${LABELER_PID}" 2>/dev/null; then
    fail "labeller not ready after ${LABELER_WAIT_S} s"
  fi
  LABELER_RC=0
  wait "${LABELER_PID}" || LABELER_RC=$?
  LABELER_PID=""
  fail "labeller exit=${LABELER_RC}"
fi
echo "=== labeller: ready after $((SECONDS - WAITED_FROM)) s ==="

# --- the gates -----------------------------------------------------------------------------------------------------------
# Only the driver's lines of a known shape leave gates.log (a progress bar can sit in front of one on the same physical line, hence the
# tr); the lines that wear one of our prefixes and have none of the shapes are counted, not shown.
echo "=== gates: the driver's own output goes to gates.log in the results directory and is never printed ==="
rc=0
PYTHONPATH=.chat_deps python scripts/chat_retrieval_gates.py \
  --checkpoint "${CHECKPOINT}" --model-config "${MODEL_CONFIG}" --gallery "${GALLERY}" --data "${DATA}" \
  --published-labels "${REFERENCE_DIR}/chexbert_labels.json" --published-hyps "${REFERENCE_DIR}/hyps.txt" \
  --published-refs "${REFERENCE_DIR}/refs.txt" --labeler-url "${LABELER_URL}" --out "${OUT}" --threads "${THREADS}" \
  > "${OUT}/gates.log" 2>&1 || rc=$?
stop_labeller
tr '\r' '\n' < "${OUT}/gates.log" | grep -aE "${GATES_SHAPES}" | tail -n 60 || true
WITHHELD="$(tr '\r' '\n' < "${OUT}/gates.log" | grep -aE '^(\[gates\] |RESULT |ERROR|=== )' | grep -avcE "${GATES_SHAPES}" || true)"
case "${WITHHELD}" in ''|*[!0-9]*) WITHHELD=unknown ;; esac
echo "=== gates lines withheld: ${WITHHELD} ==="
RESULTS="$(tr '\r' '\n' < "${OUT}/gates.log" | grep -aE "${GATES_SHAPES}" | grep -c '^RESULT ' || true)"
case "${RESULTS}" in ''|*[!0-9]*) RESULTS=0 ;; esac
if [ "${rc}" -ne 0 ]; then
  echo "ERROR gates exit=${rc}"
  exit "${rc}"
fi
# Exit 0 without the driver's RESULT line is a failure like any other: nothing was claimed.
if [ "${RESULTS}" -ne 1 ]; then
  fail "gates printed no RESULT line"
fi
echo "=== END chat retrieval gates: all three gates held, wall_s=${SECONDS} ==="
