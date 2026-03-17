#!/bin/bash
#SBATCH --partition=students
#SBATCH --gres=gpu:student:1
#SBATCH --mem=40G  # Using full 40GB RAM allocation
#SBATCH --time=30:00:00
#SBATCH --job-name=hybrid70m_fineweb_5ep
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log

set -euo pipefail

echo "=== JOB START (hybrid70m with FineWeb 1.5B tokens, 5 epochs) ==="
date
echo "Host: $(hostname)"
echo "Submit dir: ${SLURM_SUBMIT_DIR}"
echo ""
echo "Configuration:"
echo "  - Dataset: FineWeb sample-10BT"
echo "  - Target tokens: 1.5B per epoch"
echo "  - Epochs: 5 (total 7.5B tokens)"
echo "  - Batch size: 8"
echo "  - Gradient accumulation: 4"
echo "  - Effective batch: 32"
echo "  - RAM: 40GB"
echo ""

cd "${SLURM_SUBMIT_DIR}/hybrid_model_mamba_xlstm"

export HF_HOME="$PWD/.hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCHINDUCTOR_CACHE_DIR="$PWD/.torchinductor"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    python -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

nvidia-smi

# Calculate training steps for 1.5B tokens over 5 epochs
# With 40GB RAM, we can use larger batch size and more workers
# batch_size=8, accumulate=4 -> effective_batch=32
# seq_len=2048 -> tokens_per_batch = 32 * 2048 = 65,536
# steps_per_epoch = 1,500,000,000 / 65,536 ≈ 22,888
# total_steps = 22,888 * 5 ≈ 114,440 steps

python scripts/train.py \
  model=hybrid_70m \
  dataset=fineweb \
  trainer=a100_single_gpu \
  trainer.max_epochs=5 \
  trainer.max_steps=-1 \
  dataset.batch_size=8 \
  dataset.eval_batch_size=8 \
  dataset.max_length=2048 \
  trainer.accumulate_grad_batches=4 \
  trainer.val_check_interval=2000 \
  trainer.log_every_n_steps=100 \
  callbacks.checkpoint.every_n_train_steps=5000 \
  callbacks.checkpoint.save_top_k=5 \
  experiment_name=hybrid_70m_fineweb_1.5B_5epochs \
  wandb.enabled=false \
  model.gradient_clip_val=1.0 \
  dataset.num_workers=8 \
  dataset.preprocessing_num_workers=12

echo "=== JOB END (hybrid70m with FineWeb) ==="
date
