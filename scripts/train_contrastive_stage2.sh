#!/bin/bash
#SBATCH --partition=students
#SBATCH --gres=gpu:student:1
#SBATCH --mem=20G
#SBATCH --time=06:00:00
#SBATCH --job-name=clip_stage2
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log

set -euo pipefail

echo "=== JOB START (Stage 2: CLIP Alignment on Indiana CXR) ==="
date
echo "Host: $(hostname)"
echo "Submit dir: ${SLURM_SUBMIT_DIR}"
echo ""
echo "Configuration:"
echo "  - Dataset: Indiana CXR (image-text pairs)"
echo "  - Mode: CLIP (image-text contrastive)"
echo "  - Batch size: 64"
echo "  - Max steps: 5,000"
echo "  - Expected runtime: 3-4 hours"
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

# Activate virtual environment
source .venv/bin/activate

# Install contrastive learning dependencies if needed
pip install open-clip-torch Pillow

# Verify CUDA is available
echo ""
echo "Verifying CUDA availability:"
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device count: {torch.cuda.device_count()}'); print(f'Current device: {torch.cuda.current_device() if torch.cuda.is_available() else \"N/A\"}'); print(f'Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo ""

if ! python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    echo "ERROR: CUDA is not available to PyTorch!"
    echo "This job requires GPU access."
    exit 1
fi

# IMPORTANT: Update this path to your Stage 1 checkpoint
STAGE1_CHECKPOINT="./outputs/stage1_pubmed_simcse/checkpoints/last.ckpt"

# Check if Stage 1 checkpoint exists
if [ ! -f "$STAGE1_CHECKPOINT" ]; then
    echo "ERROR: Stage 1 checkpoint not found at: $STAGE1_CHECKPOINT"
    echo "Please run Stage 1 first or update the checkpoint path."
    echo "Available checkpoints:"
    ls -lh ./outputs/stage1_pubmed_simcse/checkpoints/ 2>/dev/null || echo "  No checkpoints found"
    exit 1
fi

echo "Loading Stage 1 checkpoint from: $STAGE1_CHECKPOINT"
echo ""

nvidia-smi

python scripts/train_contrastive.py \
  --config-name config_70m \
  dataset=indiana_cxr \
  trainer=a100_single_gpu \
  trainer.accelerator=cuda \
  contrastive_mode=clip \
  lm_checkpoint="$STAGE1_CHECKPOINT" \
  trainer.max_steps=5000 \
  dataset.batch_size=8 \
  dataset.eval_batch_size=8 \
  dataset.num_workers=2 \
  dataset.pin_memory=false \
  trainer.accumulate_grad_batches=8 \
  trainer.val_check_interval=250 \
  trainer.log_every_n_steps=25 \
  callbacks.checkpoint.every_n_train_steps=500 \
  callbacks.checkpoint.save_top_k=3 \
  experiment_name=stage2_indiana_clip \
  output_dir=./outputs/stage2_indiana_clip \
  wandb.enabled=false \
  model.gradient_clip_val=1.0

echo ""
echo "=== JOB END (Stage 2: CLIP) ==="
echo "Checkpoint saved to: ./outputs/stage2_indiana_clip/checkpoints/"
date
