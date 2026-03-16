#!/bin/bash
#SBATCH --partition=students
#SBATCH --gres=gpu:student:1
#SBATCH --mem=40G
#SBATCH --time=08:00:00
#SBATCH --job-name=hybrid70m
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log

set -euo pipefail

echo "=== JOB START (hybrid70m) ==="
date
echo "Host: $(hostname)"
echo "Submit dir: ${SLURM_SUBMIT_DIR}"

cd "${SLURM_SUBMIT_DIR}/hybrid_model_mamba_xlstm"

export HF_HOME="$PWD/.hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCHINDUCTOR_CACHE_DIR="$PWD/.torchinductor"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

nvidia-smi

# Fit on ~20GB MIG slice:
python scripts/train.py \
  model=hybrid_70m \
  dataset=wikitext \
  trainer=a100_single_gpu \
  trainer.max_steps=10000 \
  dataset.batch_size=4 \
  dataset.eval_batch_size=4 \
  dataset.max_length=1024 \
  trainer.accumulate_grad_batches=8 \
  experiment_name=hybrid_70m_wikitext_a100_mig20g \
  wandb.enabled=false \
  model.gradient_clip_val=0.0 \
  dataset.num_workers=2 \
  dataset.preprocessing_num_workers=2

echo "=== JOB END (hybrid70m) ==="
date