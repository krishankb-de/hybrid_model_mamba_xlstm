#!/bin/bash
# ============================================================================
# H100 env bootstrap (Phase 2 · task 2D) — creates the py>=3.10 .venv the H100
# train/eval templates source via VENV_ACTIVATE=.venv/bin/activate.
#
# Runs on aisc-batch WITHOUT a GPU (login node forbids python/pip). The .venv
# lands in your home repo, which is on the shared network FS, so it is then
# visible on every compute node — build it once.
#
# Uses uv (fast, installed cluster-wide per the workshop docs). Falls back to
# `python -m venv` if uv is unavailable.
#
# Usage:   sbatch scripts/setup_env_h100.sh
# Watch:   squeue --me   ;   tail -f logs/setup_env_h100_<jobid>.log
# ============================================================================
#SBATCH --partition=aisc-batch
#SBATCH --account=aisc
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --job-name=setup_env_h100
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log

set -euo pipefail

PY_VERSION="${PY_VERSION:-3.11}"
# H100 nodes run a CUDA 12.8 driver. The default PyPI torch wheel is built for
# CUDA 13.0 and fails at runtime ("NVIDIA driver too old", CUDA unavailable), so
# install torch/torchvision from the CUDA-12.8 index BEFORE requirements.txt.
# Bump to cu130/etc if the node drivers are upgraded (check: nvidia-smi).
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

echo "=== H100 env bootstrap (py${PY_VERSION}) ==="
date; hostname
mkdir -p logs

cd "${SLURM_SUBMIT_DIR}/hybrid_model_mamba_xlstm" 2>/dev/null || cd "${SLURM_SUBMIT_DIR:-.}"

# Install uv locally if missing (once per user; home FS is shared).
if ! command -v uv &> /dev/null; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  source "$HOME/.local/bin/env"
fi

if command -v uv &> /dev/null; then
  echo "uv: $(uv --version)"
  uv venv --python "${PY_VERSION}" .venv
  source .venv/bin/activate
  echo "Installing CUDA-12.8 torch from ${TORCH_INDEX_URL}..."
  uv pip install torch torchvision --index-url "${TORCH_INDEX_URL}"
  uv pip install -r requirements.txt
else
  echo "uv unavailable — falling back to python -m venv"
  python${PY_VERSION} -m venv .venv || python3 -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  echo "Installing CUDA-12.8 torch from ${TORCH_INDEX_URL}..."
  pip install torch torchvision --index-url "${TORCH_INDEX_URL}"
  pip install -r requirements.txt
fi

echo ""
echo "=== Import smoke test ==="
python -c "import sys, torch, transformers, pytorch_lightning, hydra, datasets; \
print('python', sys.version.split()[0]); \
print('torch', torch.__version__, '(cuda build', torch.version.cuda, ')'); \
print('transformers', transformers.__version__); \
print('lightning', pytorch_lightning.__version__)"

echo ""
echo "=== DONE. Activate with: source .venv/bin/activate ==="
date
