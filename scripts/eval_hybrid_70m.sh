#!/bin/bash
#SBATCH --partition=students
#SBATCH --gres=gpu:student:1
#SBATCH --mem=20G
#SBATCH --time=01:30:00
#SBATCH --job-name=eval_hybrid70m
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log

set -euo pipefail
mkdir -p "${SLURM_SUBMIT_DIR}/logs"

echo "=== JOB START (eval_hybrid70m) ==="
date
echo "Host: $(hostname)"
cd "${SLURM_SUBMIT_DIR}/hybrid_model_mamba_xlstm"

source .venv/bin/activate

export HF_HOME="$PWD/.hf"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"

nvidia-smi

# IMPORTANT: the eval script is scripts/evaluate_lm.py (not evaluate.py)
# --layer-pattern must match the hybrid_70m training pattern exactly
# --max-length must match training (1024)
python scripts/evaluate_lm.py \
  --checkpoint outputs/hybrid_70m_wikitext_a100_mig20g/checkpoints/last.ckpt \
  --model-config hybrid_70m \
  --layer-pattern mamba,mamba,mlstm,mamba,mamba,mlstm,mamba,mamba \
  --dataset wikitext \
  --split test \
  --batch-size 4 \
  --max-length 1024 \
  --num-workers 2 \
  --throughput \
  --generate \
  --output-dir outputs/hybrid_70m_wikitext_a100_mig20g/eval_results

echo "=== JOB END (eval_hybrid70m) ==="
date
