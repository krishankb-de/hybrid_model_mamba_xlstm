#!/bin/bash
#SBATCH --partition=students
#SBATCH --gres=gpu:student:1
#SBATCH --mem=20G
#SBATCH --time=01:30:00
#SBATCH --job-name=eval_mamba70m
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log

set -euo pipefail
mkdir -p "${SLURM_SUBMIT_DIR}/logs"

echo "=== JOB START (eval_mamba70m) ==="
date
echo "Host: $(hostname)"
cd "${SLURM_SUBMIT_DIR}/hybrid_model_mamba_xlstm"

source .venv/bin/activate

export HF_HOME="$PWD/.hf"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"

nvidia-smi

# IMPORTANT: layer-pattern must be all-mamba (8 layers) to match training
python scripts/evaluate_lm.py \
  --checkpoint outputs/mamba_70m_wikitext_a100_mig20g/checkpoints/last.ckpt \
  --model-config mamba_baseline \
  --layer-pattern mamba,mamba,mamba,mamba,mamba,mamba,mamba,mamba \
  --dataset wikitext \
  --split test \
  --batch-size 4 \
  --max-length 1024 \
  --num-workers 2 \
  --throughput \
  --output-dir outputs/mamba_70m_wikitext_a100_mig20g/eval_results
echo "=== JOB END (eval_mamba70m) ==="
date
