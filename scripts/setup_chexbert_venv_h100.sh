#!/bin/bash
# ============================================================================
# Phase 11B (2026-08-29) — builds an ISOLATED venv just for CheXbert scoring,
# separate from the main .venv used by every other train/eval script.
#
# WHY A SEPARATE VENV: f1chexbert==0.0.2 calls the legacy
# tokenizer.encode_plus(...) method, removed in transformers>=5.0. The main
# .venv needs a recent transformers for everything else in this repo (GPT-2
# tokenizer, BiomedCLIP text tower, BioMedLM teacher) -- pinning it back to
# satisfy one old scoring package would risk the rest of the pipeline. This
# venv exists ONLY to run score_chexbert_standalone.py against hyps.txt/
# refs.txt files dumped by evaluate_report_generation.py's --dump-dir flag
# (see that script's module docstring). No hybrid_xmamba/Triton/lightning/
# open_clip here -- score_chexbert_standalone.py deliberately doesn't import
# this repo at all, so none of that is needed.
#
# Mirrors setup_env_h100.sh's uv-first / python-m-venv-fallback pattern.
#
# Usage:   sbatch scripts/setup_chexbert_venv_h100.sh
# Watch:   squeue --me   ;   tail -f logs/setup_chexbert_venv_h100_<jobid>.log
# ============================================================================
#SBATCH --partition=aisc-batch
#SBATCH --account=aisc
#SBATCH --qos=aisc
#SBATCH --exclude=ga03,gx17v1,gx13v1   # ga03: ARM node, x86 venv incompatible;
                                        # gx13v1: faulty GPU (cudaErrorContained, 2026-07-19)
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --job-name=setup_chexbert_venv
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log

set -euo pipefail

PY_VERSION="${PY_VERSION:-3.11}"
VENV_DIR="${VENV_DIR:-.venv_chexbert}"

echo "=== CheXbert isolated venv bootstrap (py${PY_VERSION}, dir=${VENV_DIR}) ==="
date; hostname
mkdir -p logs

cd "${SLURM_SUBMIT_DIR}/hybrid_model_mamba_xlstm" 2>/dev/null || cd "${SLURM_SUBMIT_DIR:-.}"

if ! command -v uv &> /dev/null; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  source "$HOME/.local/bin/env"
fi

if command -v uv &> /dev/null; then
  echo "uv: $(uv --version)"
  uv venv --python "${PY_VERSION}" "${VENV_DIR}"
  source "${VENV_DIR}/bin/activate"
  # CPU-only torch is enough -- this venv only runs BERT-classifier forward
  # passes over report text, no training, no image tower.
  uv pip install torch --index-url https://download.pytorch.org/whl/cpu
  uv pip install "transformers<5" f1chexbert
else
  echo "uv unavailable — falling back to python -m venv"
  python${PY_VERSION} -m venv "${VENV_DIR}" || python3 -m venv "${VENV_DIR}"
  source "${VENV_DIR}/bin/activate"
  pip install --upgrade pip
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  pip install "transformers<5" f1chexbert
fi

echo ""
echo "=== Import smoke test ==="
python -c "import sys, torch, transformers, f1chexbert; \
print('python', sys.version.split()[0]); \
print('torch', torch.__version__); \
print('transformers', transformers.__version__); \
print('f1chexbert OK')"

echo ""
echo "=== DONE. Activate with: source ${VENV_DIR}/bin/activate ==="
date
