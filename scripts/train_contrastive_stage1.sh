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

# Note: CUDA_VISIBLE_DEVICES is set by SLURM automatically for MIG devices
# Don't override it manually
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"

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

# Verify CUDA is available
echo ""
echo "Verifying CUDA availability:"
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA device count: {torch.cuda.device_count()}')
if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    print(f'Current device: {torch.cuda.current_device()}')
    print(f'Device name: {torch.cuda.get_device_name(0)}')
else:
    print('Current device: N/A')
    print('Device name: N/A')
"
echo ""

if ! python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    echo "ERROR: CUDA is not available to PyTorch!"
    echo "This job requires GPU access."
    exit 1
fi

# Check if device count is 0 despite CUDA being available
DEVICE_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")
if [ "$DEVICE_COUNT" = "0" ]; then
    echo "WARNING: CUDA is available but device count is 0"
    echo "This may be a CUDA_VISIBLE_DEVICES issue with MIG devices"
    echo "Attempting to continue anyway - PyTorch Lightning may handle this..."
fi

nvidia-smi

python scripts/train_contrastive.py \
  --config-name config_70m \
  dataset=pubmed \
  trainer=a100_single_gpu \
  trainer.accelerator=cuda \
  contrastive_mode=simcse \
  trainer.max_steps=10000 \
  dataset.batch_size=4 \
  dataset.eval_batch_size=4 \
  dataset.max_length=256 \
  dataset.max_seq_length=256 \
  dataset.num_workers=2 \
  dataset.preprocessing_num_workers=4 \
  dataset.pin_memory=false \
  trainer.accumulate_grad_batches=16 \
  trainer.val_check_interval=500 \
  trainer.log_every_n_steps=25 \
  callbacks.checkpoint.every_n_train_steps=1000 \
  callbacks.checkpoint.save_top_k=3 \
  experiment_name=stage1_pubmed_simcse \
  output_dir=./outputs/stage1_pubmed_simcse \
  wandb.enabled=false \
  model.gradient_clip_val=0.0

echo ""
echo "=== JOB END (Stage 1: SimCSE) ==="
echo "Checkpoint saved to: ./outputs/stage1_pubmed_simcse/checkpoints/"
date
