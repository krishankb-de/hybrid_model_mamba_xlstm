#!/bin/bash
#SBATCH --partition=students
#SBATCH --gres=gpu:student:1
#SBATCH --mem=20G
#SBATCH --time=12:00:00
#SBATCH --job-name=simcse_stage1
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log

set -euo pipefail

echo "=== JOB START (Stage 1: SimCSE on PubMed) ==="
date
echo "Host: $(hostname)"
echo "Submit dir: ${SLURM_SUBMIT_DIR}"
echo ""
echo "Configuration:"
echo "  - Dataset: PubMed abstracts"
echo "  - Target tokens: 500M"
echo "  - Mode: SimCSE (text-only contrastive)"
echo "  - Batch size: 32"
echo "  - Max steps: 10,000"
echo "  - Expected runtime: 8-10 hours"
echo ""

cd "${SLURM_SUBMIT_DIR}/hybrid_model_mamba_xlstm"

export HF_HOME="$PWD/.hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCHINDUCTOR_CACHE_DIR="$PWD/.torchinductor"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"
export CUDA_LAUNCH_BLOCKING=0

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    python -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

# Install contrastive learning dependencies
pip install open-clip-torch Pillow

nvidia-smi

python scripts/train_contrastive.py \
  --config-name config_70m \
  dataset=pubmed \
  trainer=a100_single_gpu \
  trainer.max_steps=10000 \
  contrastive_mode=simcse \
  dataset.batch_size=32 \
  dataset.eval_batch_size=32 \
  dataset.num_workers=4 \
  dataset.preprocessing_num_workers=8 \
  trainer.val_check_interval=500 \
  trainer.log_every_n_steps=25 \
  callbacks.checkpoint.every_n_train_steps=1000 \
  callbacks.checkpoint.save_top_k=3 \
  experiment_name=stage1_pubmed_simcse \
  output_dir=./outputs/stage1_pubmed_simcse \
  wandb.enabled=false \
  model.gradient_clip_val=1.0

echo ""
echo "=== JOB END (Stage 1: SimCSE) ==="
echo "Checkpoint saved to: ./outputs/stage1_pubmed_simcse/checkpoints/"
date
