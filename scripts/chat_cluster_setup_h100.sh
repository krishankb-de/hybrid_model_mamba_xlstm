#!/bin/bash
# ============================================================================
# CHAT_UI_PLAN.md P0-G — set up the chat UI's own cluster directory. Additive
# only (R8): creates symlinks that do not exist yet, installs the web deps into
# overlay directories (never into the shared venvs), checks the inputs exist.
#   bash scripts/chat_remote.sh submit scripts/chat_cluster_setup_h100.sh MAIN_REPO=<thesis checkout>
# Re-runnable. An overlay is installed until its .setup_ok sentinel exists, and the sentinel is written only after
# the install and the import check have both passed: a failed `uv pip install --target` leaves its directory
# behind, so the directory alone proves nothing. Any [setup] ERROR line makes the job exit 1, after every check
# has printed. R7: the log is read through `chat_remote.sh summary`; print names, counts and versions only.
# ============================================================================
#SBATCH --partition=pot-hpi-aisc-batch
#SBATCH --account=aisc
#SBATCH --qos=aisc
#SBATCH --exclude=ga03,gx17v1,gx13v1   # ga03: ARM node, x86 venv incompatible; gx13v1: faulty GPU
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --job-name=chat_setup
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}/hybrid_model_mamba_xlstm" 2>/dev/null || cd "${SLURM_SUBMIT_DIR:-.}"
MAIN_REPO="${MAIN_REPO:?set MAIN_REPO to the thesis checkout}"
DATA="${DATA:-/sc/home/$USER/dataset/mimic_full}"
FAILED=0
problem() { echo "[setup] ERROR $*"; FAILED=$((FAILED + 1)); }   # the job goes on to print every check, then exits 1
echo "=== chat setup: node=$(hostname) job=${SLURM_JOB_ID:-?} ==="
for name in outputs results .venv .venv_chexbert; do
  if [ -e "${name}" ] || [ -L "${name}" ]; then
    echo "[setup] ${name}: present, left alone"
  elif [ -e "${MAIN_REPO}/${name}" ]; then
    ln -s "${MAIN_REPO}/${name}" "${name}"
    echo "[setup] ${name}: linked to the thesis checkout"
  else
    problem "${name} not found in MAIN_REPO"
  fi
done
mkdir -p "${HOME}/chat_sessions" && chmod 700 "${HOME}/chat_sessions"
for f in outputs/h100_report_gen_m3_tower13d_s42/checkpoints/last.ckpt \
         outputs/h100_report_gen_full_ext_4gpu_tower13d/checkpoints/last.ckpt \
         outputs/h100_kd_150m_v2_full_data_lr3e6/checkpoints/last.ckpt \
         results/report_gen_m3_test_split_s42/hyps.txt results/report_gen_m3_test_split_s42/refs.txt \
         results/report_gen_m3_test_split_s42/chexbert_labels.json results/retrieval_floor_test_split/hyps.txt \
         "${DATA}/train.parquet" "${DATA}/test.parquet"; do
  if [ -f "${f}" ]; then echo "[setup] ok $(basename "$(dirname "${f}")")/$(basename "${f}")"
  else problem "missing $(basename "$(dirname "${f}")")/$(basename "${f}")"; fi
done
# The shared venvs were built with `uv venv` and carry no pip (no bin/pip, no site-packages/pip, checked
# 2026-10-01), so `python -m pip` cannot install anything. uv resolves each overlay for its venv's own
# interpreter; --target writes only into the overlay directory and never into the venv. uv is needed only
# while an install is pending.
if [ ! -f .chat_deps/.setup_ok ] || [ ! -f .chat_deps_chexbert/.setup_ok ]; then
  UV="${HOME}/.local/bin/uv"
  [ -x "${UV}" ] || UV="$(command -v uv || true)"
  [ -n "${UV}" ] || { echo "[setup] ERROR uv not found"; exit 1; }
fi
if [ ! -f .chat_deps/.setup_ok ]; then
  "${UV}" pip install --quiet --python .venv/bin/python --target .chat_deps \
    "fastapi>=0.115" "uvicorn>=0.30" "python-multipart>=0.0.9" "httpx>=0.27" \
    || { echo "[setup] ERROR overlay install failed: main"; exit 1; }
fi
PYTHONPATH=.chat_deps .venv/bin/python -c "
import sys, torch, fastapi, uvicorn, httpx
try:
    import python_multipart
except ImportError:
    import multipart
print('[setup] main venv: python', sys.version.split()[0], 'torch', torch.__version__, 'fastapi', fastapi.__version__, 'httpx', httpx.__version__)" \
  || { echo "[setup] ERROR overlay check failed: main"; exit 1; }
[ -f .chat_deps/.setup_ok ] || date -u +%Y-%m-%dT%H:%M:%SZ > .chat_deps/.setup_ok
if [ ! -f .chat_deps_chexbert/.setup_ok ]; then
  "${UV}" pip install --quiet --python .venv_chexbert/bin/python --target .chat_deps_chexbert \
    "fastapi>=0.115" "uvicorn>=0.30" \
    || { echo "[setup] ERROR overlay install failed: chexbert"; exit 1; }
fi
PYTHONPATH=.chat_deps_chexbert .venv_chexbert/bin/python -c "import sys, fastapi, uvicorn, sklearn, transformers; assert int(transformers.__version__.split('.')[0]) < 5; assert tuple(int(x) for x in sklearn.__version__.split('.')[:2]) < (1, 8); print('[setup] chexbert venv: python', sys.version.split()[0], 'transformers', transformers.__version__, 'sklearn', sklearn.__version__, 'fastapi', fastapi.__version__)" \
  || { echo "[setup] ERROR overlay check failed: chexbert"; exit 1; }
[ -f .chat_deps_chexbert/.setup_ok ] || date -u +%Y-%m-%dT%H:%M:%SZ > .chat_deps_chexbert/.setup_ok
if command -v node >/dev/null 2>&1; then echo "[setup] node $(node --version)"; else echo "[setup] node absent"; fi
if [ "${FAILED}" -ne 0 ]; then
  echo "=== END chat setup: ${FAILED} ERROR line(s) above ==="
  exit 1
fi
echo "=== END chat setup ==="
