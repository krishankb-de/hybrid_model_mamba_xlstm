#!/bin/bash
#SBATCH --partition=students
#SBATCH --gres=gpu:student:1
#SBATCH --mem=40G  # Using full 40GB RAM allocation
#SBATCH --time=24:00:00
#SBATCH --job-name=hybrid70m_fineweb
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log

set -euo pipefail

echo "=== JOB START (hybrid70m with FineWeb 1.5B tokens, 4 epochs) ==="
date
echo "Host: $(hostname)"
echo "Submit dir: ${SLURM_SUBMIT_DIR}"
echo ""
echo "Configuration:"
echo "  - Dataset: FineWeb sample-10BT"
echo "  - Target tokens: 1.5B per epoch"
echo "  - Epochs: 4 (total 6B tokens)"
echo "  - Batch size: 4 (reduced for 20GB MIG GPU)"
echo "  - Gradient accumulation: 8"
echo "  - Effective batch: 32"
echo "  - RAM: 40GB"
echo "  - Sequence length: 1024 (reduced from 2048)"
echo ""

cd "${SLURM_SUBMIT_DIR}/hybrid_model_mamba_xlstm"

export HF_HOME="$PWD/.hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCHINDUCTOR_CACHE_DIR="$PWD/.torchinductor"
# Memory optimization flags
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

nvidia-smi

# Calculate training steps for 1.5B tokens over 4 epochs
# MIG GPU has only ~20GB VRAM, so we need smaller batches
# batch_size=4, accumulate=8 -> effective_batch=32
# seq_len=1024 -> tokens_per_batch = 32 * 1024 = 32,768
# steps_per_epoch = 1,500,000,000 / 32,768 ≈ 45,776
# total_steps = 45,776 * 4 ≈ 183,104 steps

python scripts/train.py \
  model=hybrid_70m \
  dataset=fineweb \
  trainer=a100_single_gpu \
  trainer.max_epochs=4 \
  trainer.max_steps=-1 \
  dataset.batch_size=4 \
  dataset.eval_batch_size=4 \
  dataset.max_length=1024 \
  dataset.max_seq_length=1024 \
  trainer.accumulate_grad_batches=8 \
  trainer.val_check_interval=2000 \
  trainer.log_every_n_steps=100 \
  callbacks.checkpoint.every_n_train_steps=5000 \
  callbacks.checkpoint.save_top_k=3 \
  experiment_name=hybrid_70m_fineweb_1.5B_4epochs \
  wandb.enabled=false \
  model.gradient_clip_val=1.0 \
  dataset.num_workers=4 \
  dataset.preprocessing_num_workers=8 \
  trainer.compile_model=false

echo "=== JOB END (hybrid70m with FineWeb) ==="
date
