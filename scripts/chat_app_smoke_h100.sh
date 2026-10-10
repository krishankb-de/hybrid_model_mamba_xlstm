#!/bin/bash
# ============================================================================
# CHAT_UI_PLAN.md P7-A (CPU, about 1 min): the chat app imports on the cluster, each half in its own venv plus its overlay (P0-G),
# and the label order the client assumes is the one f1chexbert reports. Installs nothing, in the shared venvs or anywhere (R8).
#   bash scripts/chat_remote.sh sync && bash scripts/chat_remote.sh submit scripts/chat_app_smoke_h100.sh
#   bash scripts/chat_remote.sh summary logs/chat_app_smoke_<jobid>.log
#
# Three probes, one python process each:
#   server   .venv + .chat_deps                      import fastapi, app.server
#   labeler  .venv_chexbert + .chat_deps_chexbert    import app.labeler, transformers
#   labels   .venv_chexbert + .chat_deps_chexbert    app.labels.CHEXBERT_14 is the list of names F1CheXbert reports
# The job log, in full, when all is well:
#   === sync <40-hex commit> <clean|dirty> ===
#   === P7-A chat app import smoke: job=<id> node=<node> ===
#   [setup] app.server imports; fastapi <version>
#   [setup] app.labeler imports; transformers <version>
#   [setup] chexbert label order equal: true (f1chexbert <version>)
#   === END chat app smoke ===
#
# THE LABEL ORDER is read from f1chexbert's source, not from a live F1CheXbert: f1chexbert 0.0.2 assigns `self.target_names` in
# F1CheXbert.__init__, after it has fetched and loaded the CheXbert weights, so no import yields the names, and constructing the
# object would load those weights (and, on an offline node, try the network). The probe therefore never imports f1chexbert: it finds
# the package with importlib.util.find_spec, parses f1chexbert/f1chexbert.py with ast, and compares the list literal assigned to
# target_names with CHEXBERT_14. It insists on exactly ONE assignment to that name in the source (a second one, or an augmented one,
# would make "the" list ambiguous), and it reads the package's version from its distribution metadata (importlib.metadata, which
# does not import the package) so that the line says which source was read. The version is printed only when it is digits and dots
# (R7: numbers only). No weights are read. P5-F still checks the order against the live service.
#
# R7: the job log carries only ===, [setup] and ERROR lines, and names no path. Each probe's whole output (library warnings, a
# traceback) goes to results/chat_app_smoke_<jobid>/<probe>.out, a new directory (R8: results is a symlink into the thesis checkout
# and only a new results/chat_* directory may be made through it), and only the one [setup] line the probe printed is shown. A probe
# that exits non-zero also gets `ERROR <probe> exit=<rc>`, and nothing else of its output. The job runs every probe, then exits 1 if
# any failed. Exit codes worth knowing: 127 the interpreter is missing; 3 (labels) the order differs, and its [setup] line then says
# `equal: false`; 4 (labels) f1chexbert's source does not hold exactly one plain target_names assignment, or its version is not
# digits and dots (no [setup] line then); 1 an import, a parse or the metadata failed.
# ============================================================================
#SBATCH --partition=pot-hpi-aisc-batch
#SBATCH --account=aisc
#SBATCH --qos=aisc
#SBATCH --exclude=ga03,gx17v1,gx13v1   # ga03: ARM node, x86 venv incompatible; gx13v1: faulty GPU
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:15:00
#SBATCH --job-name=chat_app_smoke
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}/hybrid_model_mamba_xlstm" 2>/dev/null || cd "${SLURM_SUBMIT_DIR:-.}"

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

# Every line printed below starts with ===, [setup] or ERROR (the shapes `chat_remote.sh summary` shows) and names no path.
fail() { echo "ERROR $*"; exit 1; }

echo "=== P7-A chat app import smoke: job=${SLURM_JOB_ID:-local} node=$(hostname) ==="
export HF_HUB_OFFLINE=1      # compute nodes are offline: an import must never wait on the network
[ -d results ] || fail "results is missing: run chat_cluster_setup_h100.sh first"
OUT="results/chat_app_smoke_${SLURM_JOB_ID:-local}"
mkdir -p "${OUT}" 2>/dev/null || fail "the results directory cannot be made"

# The probes. Python code in single quotes, so it holds none: strings use double quotes.
SERVER_PROBE='import fastapi, app.server; print("[setup] app.server imports; fastapi", fastapi.__version__)'
LABELER_PROBE='import app.labeler, transformers; print("[setup] app.labeler imports; transformers", transformers.__version__)'
LABELS_PROBE='import ast, importlib.metadata, importlib.util, pathlib, re
from app.labels import CHEXBERT_14
spec = importlib.util.find_spec("f1chexbert")
tree = ast.parse(pathlib.Path(spec.submodule_search_locations[0], "f1chexbert.py").read_text())
version = importlib.metadata.version("f1chexbert")
def assigned(n):
    return n.targets if isinstance(n, ast.Assign) else [n.target] if isinstance(n, (ast.AnnAssign, ast.AugAssign)) else []
found = [n for n in ast.walk(tree) for t in assigned(n) if getattr(t, "attr", getattr(t, "id", None)) == "target_names"]
if len(found) != 1 or isinstance(found[0], ast.AugAssign) or not re.fullmatch(r"[0-9]+(\.[0-9]+)*", version):
    raise SystemExit(4)
equal = list(CHEXBERT_14) == ast.literal_eval(found[0].value)
print("[setup] chexbert label order equal:", str(equal).lower(), "(f1chexbert " + version + ")")
raise SystemExit(0 if equal else 3)'

FAILED=0
# probe <name> <command ...>: the command is a python with its overlay on PYTHONPATH. All its output goes to ${OUT}/<name>.out; the log
# gets the one [setup] line it printed, and `ERROR <name> exit=<rc>` when it exited non-zero or printed none.
probe() {
  local name="$1" rc=0 line=""
  shift
  "$@" > "${OUT}/${name}.out" 2>&1 || rc=$?
  line="$(sed -n '/^\[setup\] /{p;q;}' "${OUT}/${name}.out" 2>/dev/null)" || line=""
  [ -z "${line}" ] || echo "${line}"
  if [ "${rc}" -ne 0 ] || [ -z "${line}" ]; then
    echo "ERROR ${name} exit=${rc}"
    FAILED=$((FAILED + 1))
  fi
}

probe server env PYTHONPATH=.chat_deps .venv/bin/python -c "${SERVER_PROBE}"
probe labeler env PYTHONPATH=.chat_deps_chexbert .venv_chexbert/bin/python -c "${LABELER_PROBE}"
probe labels env PYTHONPATH=.chat_deps_chexbert .venv_chexbert/bin/python -c "${LABELS_PROBE}"

if [ "${FAILED}" -ne 0 ]; then
  echo "=== END chat app smoke: ${FAILED} of 3 probe(s) failed ==="
  exit 1
fi
echo "=== END chat app smoke ==="
