#!/bin/bash
# ============================================================================
# ISBI B10 (ISBI_BASELINES_PLAN.md) -- one-off environment for the PUBLISHED
# report generators (R2Gen, R2GenCMN, R2Gen-Mamba, CvT2DistilGPT2, CXRMate).
#
# WHY A SEPARATE VENV: the project .venv carries torch 2.11 / transformers 5.x.
# CXRMate's remote code was written against transformers 4.41.2 and
# CvT2DistilGPT2 against 4.35 (its .ckpt pickles pytorch_lightning objects), so they get their own interpreter under
# external/.venv_rrg. The project .venv is never touched.
#
# Also pre-fetches the two Hugging Face repos the adapters load
# (aehrc/cxrmate-single-tf, distilgpt2) into the same HF_HOME the jobs use, so
# the generation jobs can run with HF_HUB_OFFLINE=1.
#
# CPU only. Submit from ~/hybrid_isbi:  sbatch scripts/isbi_published_rrg_setup_h100.sh
# ============================================================================
#SBATCH --partition=pot-hpi-aisc-batch
#SBATCH --account=aisc
#SBATCH --qos=aisc
#SBATCH --exclude=ga03,gx13v1   # ga03: ARM node; gx13v1: faulty GPU
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --job-name=isbi_rrg_setup
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs

EXT_DIR="${EXT_DIR:-external}"
VENV_DIR="${VENV_DIR:-${EXT_DIR}/.venv_rrg}"
export HF_HOME="${HF_HOME:-/sc/scratch/${USER}/hybrid_xmamba_h100/.hf}"

if ! command -v uv &> /dev/null; then
  if [ -x "$HOME/.local/bin/uv" ]; then
    source "$HOME/.local/bin/env"
  else
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.local/bin/env"
  fi
fi
echo "uv: $(uv --version)"

if [ ! -x "${VENV_DIR}/bin/python" ]; then
  uv venv --python 3.11 "${VENV_DIR}"
fi
source "${VENV_DIR}/bin/activate"
uv pip install "torch==2.3.1" "torchvision==0.18.1" --index-url https://download.pytorch.org/whl/cu121
uv pip install "numpy<2" "transformers==4.41.2" "timm==0.6.13" einops yacs pandas "pyarrow<17" pillow \
  opencv-python-headless tqdm scikit-learn "huggingface_hub<0.26" safetensors "pytorch-lightning==2.1.3" "setuptools<70"

python - <<'EOF'
import os
from huggingface_hub import snapshot_download
for rid in ("aehrc/cxrmate-single-tf", "distilgpt2"):
    p = snapshot_download(rid, allow_patterns=["*.json", "*.py", "*.safetensors", "*.txt", "*.model", "README.md"])
    print("cached", rid, "->", p)
import torch, torchvision, transformers, timm
print("torch", torch.__version__, "torchvision", torchvision.__version__,
      "transformers", transformers.__version__, "timm", timm.__version__)
EOF
echo "=== DONE ==="
