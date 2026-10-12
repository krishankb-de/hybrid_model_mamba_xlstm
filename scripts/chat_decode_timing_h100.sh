#!/bin/bash
# ============================================================================
# CHAT_UI_PLAN.md P8-F, part 1 (CPU, 8 CPUs, 32 GB, about 10 min): where does a live turn's time go? A timing job: it measures and fixes nothing.
#   bash scripts/chat_remote.sh sync && bash scripts/chat_remote.sh submit scripts/chat_decode_timing_h100.sh [NODE=gx14]
#   bash scripts/chat_remote.sh summary logs/chat_decode_timing_<jobid>.log
#
# Why. The P1-D decision predicted about 4 s per turn on CPU, from job 2590184: 3.05 s per report for a cached beam-3, 100-token decode on gx14
# with 8 CPUs (scripts/chat_cpu_decode_probe_h100.sh). The live server (job 2638789, the same node, the same 8 CPUs) takes 7.4 to 8.1 s for that
# decode in its generate stage: about 45 ms more per step. scripts/chat_decode_timing.py decodes the same five test rows (0 to 4, the picker's)
# seven ways, each adding one part of the live path to the one before, so that the first arm that is slower says where the gap comes from:
#   A        the cached beam search alone (on_step=None), main thread: the P1-C path, decode only
#   B        A with a no-op on_step
#   C        Engine.generate in the main thread
#   D        C in a one-worker thread pool, as the pipeline runs it
#   E        the pipeline's whole turn without HTTP: a real store on a fresh directory of the NFS home, the real worker, the events drained on a loop
#            the way the SSE bridge drains them
#   E-local  E with the store on node-local disk (TMPDIR, else /tmp): E against E-local is the NFS
#   F        E through HTTP: the real create_app, uvicorn on a free loopback port inside this job, httpx and the probe's reader
#   A-end    A again after F, on the same rows and options with no warm-up before it. The seven run in a fixed order, so a neighbour that came
#            or went on the node while they ran would look like a difference between arms: A-end against A shows it. It is not one of the
#            seven the threshold looks at; its ratio to A is on its line and, as a_end_ratio, in the summary.
# Reading it: A to B is the decoder building its callback's argument at every step; B to C the engine's own work (the text decode of the best
# beam at every step, the repair of the finished report); C to D the worker thread; D to E the pipeline's work at every step (the session check,
# the redaction, the SQLite write, the hand-over to the loop); E to E-local the NFS (when the environment line says the local directory is on a
# local disk and the home is on nfs); E to F the HTTP and the server's own loop. F's client is a thread of this job, where the live one is on the
# laptop, so F reads a little high by the client's own wake-ups. The total_ms that E, E-local and F report leaves out the label stage and the
# CheXbert half of the score stage, because the job runs no labeller: it is not the live turn's total, and only its generate stage (ms) is
# comparable with the live turn's. A turn that is still going after 300 s fails its arm with the code turn_deadline and the job goes on to the next
# arm, so that one hung turn costs five minutes and not the hour.
# It builds the engine as the server builds it (the published model, CPU, the job's CPUs as threads), under the published protocol and with the
# cached decode, and runs one warm-up turn that no arm counts. Every arm's report must be the same text: that is compared in memory, and only
# `same_report_all_arms` is printed.
#
# Output. The job log carries the sync line, two banners, and ten RESULT lines of the script (each under the 300 characters `summary` keeps): the
# environment (node, CPU model, the size of the CPU affinity and how many physical cores it covers, the node's load average once the engines
# are loaded, the filesystem of the node-local store directory and of the NFS one as a word each (nfs, tmpfs, xfs, ext4 and so on, never a path),
# torch's thread count in the main thread and in a worker), one line per arm as it finishes (the rows' ms, their mean, the ratio to A's mean,
# and for E, E-local and F the whole turn's ms and its mean), A-end, and the summary (the arm of A to F whose mean first exceeds 1.5 times A's,
# or none; a_end_ratio; same_report_all_arms; and the node's load average before the engines were loaded and after the last arm). The cores and
# the loads are there because 8 CPUs can be 8 cores or 4 cores of 2 hardware threads, and because a node's other jobs share its caches and its
# memory bandwidth: neither shows in a thread count, and either can make the same decode slower on one day than on another. A normal log:
#   === sync <40-hex commit> <clean|dirty> ===
#   === chat decode timing: node=<node> job=<id> threads=<n> gallery=<build id> ===
#   === decode timing: the script's raw output stays in files under CHAT_HOME; its RESULT lines of known shapes are shown here as it prints them ===
#   RESULT {"env":{...}}  RESULT {"arm":"A",...}  ...  RESULT {"arm":"F",...}  RESULT {"arm":"A-end",...}  RESULT {"summary":{...}}
#   === timing lines withheld: 0 ===
#   === END chat decode timing ===
# The job exits 0 when every arm ran and every report was the same, and with the script's own status otherwise (1: an arm failed, a report
# differed, or the setup could not finish: its RESULT line names the class); the RESULT lines are shown in either case.
#
# R7 (DUA). The job decodes MIMIC test images, so what it prints is numbers, booleans, fixed words, test-row indices and the environment line,
# and nothing else: the script's stdout and stderr go to files under ${CHAT_HOME}/logs (decode_timing_<jobid>.out and .err), and of stdout only a
# line of one of the shapes of TIMING_SHAPES below (an allowlist: a report, an id or a path on a RESULT line is no known shape) reaches this log;
# `=== timing lines withheld: N ===` counts the others. Nothing names a path, a port or an exception text. The two filesystem words are of a
# closed list, and the mount table they come from, which names the home's server and its export, is never printed. Read the log with the
# `summary` line above.
#
# Files (R8: additive only). The stores of E and F are made under ${CHAT_HOME}/timing/<jobid>/ (e and f) and left in place: they hold the six
# turns each of this job and are DUA-covered like the rest of CHAT_HOME. E-local's store is a directory of TMPDIR that the script makes and
# removes (a job that is killed leaves it to the node). Nothing else is written; the gallery and the checkpoints are only read.
#
# Levers, as NAME=value arguments of `chat_remote.sh submit`: CHAT_HOME (default $HOME/chat_sessions), GALLERY (a build directory, default
# ${CHAT_HOME}/gallery/g13d_m3_v1: the test rows come from it, and it must open against the engine's own tower), NODE (a node name such as gx14).
# NODE is the node this measurement is for, for instance the one jobs 2590184 and 2638789 ran on. sbatch does not read NODE, so the lever cannot
# place the job by itself: the job looks at where it landed and refuses before it loads anything when that is another node. To place it, give
# sbatch's own option after the wrapper's name and the lever with it:
#   bash scripts/chat_remote.sh submit scripts/chat_decode_timing_h100.sh NODE=gx14 -- --nodelist=gx14
# It has no --requeue: a requeued job would meet the stores of its first run, and the timing directory of a job is never reused.
# ============================================================================
#SBATCH --partition=pot-hpi-aisc-batch
#SBATCH --account=aisc
#SBATCH --qos=aisc
#SBATCH --exclude=ga03,gx17v1,gx13v1   # ga03: ARM node, x86 .venv incompatible; gx13v1: faulty GPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=chat_decode_timing
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

SCRATCH_ROOT="${SCRATCH_ROOT:-/sc/scratch/$USER/hybrid_xmamba_h100}"
VENV_ACTIVATE="${VENV_ACTIVATE:-.venv/bin/activate}"
CHAT_HOME="${CHAT_HOME:-$HOME/chat_sessions}"
GALLERY="${GALLERY:-${CHAT_HOME}/gallery/g13d_m3_v1}"
NODE="${NODE:-}"
THREADS="${SLURM_CPUS_PER_TASK:-8}"
LOCAL_DIR="${TMPDIR:-/tmp}"
JOB="${SLURM_JOB_ID:-local}"

# The lines of the script's stdout that may reach this log, as one anchored pattern per line (grep -E reads a newline as "or"). The keys are
# fixed and so are the arm and step words; a value is a number with a decimal point, a count, a boolean, a word of letters, digits and
# underscores, or, of the environment line, a node name and a CPU model of plain characters and two filesystem words of a closed list. A
# report, an id or a path on a RESULT line matches none of them. tests/test_chat_decode_timing.py runs these very patterns, through grep, over
# everything the script prints and over lines that carry text. No line may be empty: an empty pattern matches every line.
TIMING_SHAPES='^RESULT \{"env":\{"node":"[A-Za-z0-9._-]{1,16}","cpu":(null|"[A-Za-z0-9 ().@+_-]{1,48}"),"affinity":(null|[0-9]{1,4}),"cores":(null|[0-9]{1,4}),"load":(null|\[[0-9]{1,4}\.[0-9]{1,2},[0-9]{1,4}\.[0-9]{1,2},[0-9]{1,4}\.[0-9]{1,2}\]),"fs_local":(null|"(nfs|tmpfs|xfs|ext4|lustre|gpfs|beegfs|other)"),"fs_home":(null|"(nfs|tmpfs|xfs|ext4|lustre|gpfs|beegfs|other)"),"threads_main":[0-9]{1,4},"threads_worker":[0-9]{1,4},"engine":"(real|tiny)","rows":\[[0-9]{1,4}(,[0-9]{1,4}){0,5}\]\}\}$
^RESULT \{"arm":"(A|B|C|D|E|E-local|F|A-end)","ms":\[[0-9]{1,7}\.[0-9]{1,2}(,[0-9]{1,7}\.[0-9]{1,2}){0,5}\],"mean":[0-9]{1,7}\.[0-9]{1,2},"ratio":([0-9]{1,7}\.[0-9]{1,2}|null),"same_report":(true|false)\}$
^RESULT \{"arm":"(E|E-local|F)","ms":\[[0-9]{1,7}\.[0-9]{1,2}(,[0-9]{1,7}\.[0-9]{1,2}){0,5}\],"mean":[0-9]{1,7}\.[0-9]{1,2},"ratio":([0-9]{1,7}\.[0-9]{1,2}|null),"total_ms":\[[0-9]{1,7}\.[0-9]{1,2}(,[0-9]{1,7}\.[0-9]{1,2}){0,5}\],"mean_total_ms":[0-9]{1,7}\.[0-9]{1,2},"same_report":(true|false)\}$
^RESULT \{"arm":"(A|B|C|D|E|E-local|F|A-end)","error":"[A-Za-z_][A-Za-z0-9_]{0,59}","rows_done":[0-9]{1,2},"ms":\[([0-9]{1,7}\.[0-9]{1,2}(,[0-9]{1,7}\.[0-9]{1,2}){0,5})?\]\}$
^RESULT \{"summary":\{"first_over_1_5x":"(A|B|C|D|E|E-local|F|none|unknown)","a_end_ratio":(null|[0-9]{1,7}\.[0-9]{1,2}),"same_report_all_arms":(true|false),"tokens":\[([0-9]{1,5}(,[0-9]{1,5}){0,5})?\],"arms_failed":\[("(A|B|C|D|E|E-local|F|A-end)"(,"(A|B|C|D|E|E-local|F|A-end)"){0,7})?\],"load_before":(null|\[[0-9]{1,4}\.[0-9]{1,2},[0-9]{1,4}\.[0-9]{1,2},[0-9]{1,4}\.[0-9]{1,2}\]),"load_after":(null|\[[0-9]{1,4}\.[0-9]{1,2},[0-9]{1,4}\.[0-9]{1,2},[0-9]{1,4}\.[0-9]{1,2}\])\}\}$
^RESULT \{"fatal":"[A-Za-z_][A-Za-z0-9_]{0,59}","step":"(engine|gallery|rows|timing|warmup|run)"\}$'

# --- the levers and the tree, before anything is loaded or made. A refusal says what is wrong and never the value it was given. ---------
[ -z "${NODE}" ] || [[ "${NODE}" =~ ^[a-z][a-z0-9-]{0,30}$ ]] || fail "NODE must be one node name: lower-case letters, digits and dashes"
[[ "${THREADS}" =~ ^[1-9][0-9]{0,2}$ ]] || fail "SLURM_CPUS_PER_TASK must be a whole number from 1 to 999 without a leading zero"
[[ "$(basename "${GALLERY}")" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || fail "GALLERY must end in one plain build name: letters, digits, dot, dash and underscore"
[ -d "${CHAT_HOME}" ] || fail "CHAT_HOME does not exist: run chat_cluster_setup_h100.sh first"
[ -d "${LOCAL_DIR}" ] || fail "the node-local directory (TMPDIR, else /tmp) does not exist"
[ -f "${GALLERY}/manifest.json" ] || fail "the gallery has no manifest.json: build it first (P5-B)"
python3 -c 'import json, sys; g = json.load(open(sys.argv[1])).get("gate_rk"); sys.exit(0 if isinstance(g, dict) and g.get("equal") is True else 1)' "${GALLERY}/manifest.json" 2>/dev/null \
  || fail "the gallery's manifest.json is unreadable or its R@k gate is not equal: it cannot be used"
[ -f "${VENV_ACTIVATE}" ] || fail "the venv's activate script is missing: run chat_cluster_setup_h100.sh first"
[ -f .chat_deps/.setup_ok ] || fail "the web overlay is not set up: run chat_cluster_setup_h100.sh first"
[ -f scripts/chat_decode_timing.py ] || fail "chat_decode_timing.py is missing from this tree: run chat_remote.sh sync first"
NODE_NOW="$(hostname)"
NODE_NOW="${NODE_NOW%%.*}"
[ -z "${NODE}" ] || [ "${NODE_NOW}" = "${NODE}" ] || fail "the job is not on the requested node: place it with -- --nodelist=<node> after the wrapper's name"

echo "=== chat decode timing: node=$(hostname) job=${JOB} threads=${THREADS} gallery=$(basename "${GALLERY}") ==="

# The environment of the serving job: offline Hugging Face on a compute node, the job's CPUs as OpenMP and MKL threads, the web overlay on
# the path of this command and nowhere else. Files made from here on are owner-only (the stores hold report text).
umask 077
export HF_HOME="${SCRATCH_ROOT}/.hf" HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONPATH=.chat_deps
source "${VENV_ACTIVATE}"
mkdir -p "${CHAT_HOME}/logs"
OUT="${CHAT_HOME}/logs/decode_timing_${JOB}.out"
ERR="${CHAT_HOME}/logs/decode_timing_${JOB}.err"

echo "=== decode timing: the script's raw output stays in files under CHAT_HOME; its RESULT lines of known shapes are shown here as it prints them ==="
# The script prints each arm's line as the arm finishes, and its stdout goes through tee into its file and through grep into this log, line by
# line: what an arm measured is here at once, and stays here when the job is killed at its time limit. Only a line of a known shape passes grep;
# the others are in the file and are counted below, not shown. The script's stderr goes to its own file. The status of each stage of the
# pipeline is read straight after it (a grep that finds nothing is not an error, and set -e is off for the pipeline alone).
set +e
python scripts/chat_decode_timing.py --engine real --home "${CHAT_HOME}" --gallery "${GALLERY}" --job "${JOB}" --threads "${THREADS}" \
  --local-dir "${LOCAL_DIR}" 2> "${ERR}" | tee "${OUT}" | grep --line-buffered -aE "${TIMING_SHAPES}"
STATUSES=("${PIPESTATUS[@]}")
set -e
rc="${STATUSES[0]:-1}"
case "${rc}" in ''|*[!0-9]*) rc=1 ;; esac
[ "${STATUSES[1]:-1}" -eq 0 ] || fail "the timing output could not be kept"
WITHHELD="$(grep -avcE "${TIMING_SHAPES}" "${OUT}" || true)"
case "${WITHHELD}" in ''|*[!0-9]*) WITHHELD=unknown ;; esac
echo "=== timing lines withheld: ${WITHHELD} ==="
RESULTS="$(grep -acE "${TIMING_SHAPES}" "${OUT}" || true)"
case "${RESULTS}" in ''|*[!0-9]*) RESULTS=0 ;; esac
if [ "${rc}" -ne 0 ]; then
  echo "ERROR the timing script exited with code ${rc} (see the logs under CHAT_HOME)"
  exit "${rc}"
fi
# Exit 0 without all ten lines (the environment, the seven arms, A-end and the summary) is a failure like any other: something was not measured.
[ "${RESULTS}" -eq 10 ] || fail "the timing script printed ${RESULTS} result lines, not 10"
echo "=== END chat decode timing ==="
