#!/bin/bash
#SBATCH --partition=mitarb
#SBATCH --account=mitarb
#SBATCH --gres=gpu:mitarb:1
#SBATCH --mem=40G
#SBATCH --time=48:00:00
#SBATCH --job-name=simcse_fixed
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log

# Fixed Stage 1 Training Script - Prevents Model Collapse
# Key fixes:
# - Higher learning rate (0.0001 vs 0.00005)
# - Proper temperature (0.07 vs default)
# - Gradient clipping enabled (1.0)
# - Better early stopping (monitors validation loss)
# - Longer training (10,000 steps)

set -euo pipefail

echo "=== JOB START (Stage 1: SimCSE FIXED) ==="
date
echo "Host: $(hostname)"
echo "Submit dir: ${SLURM_SUBMIT_DIR}"
echo ""
echo "Configuration (FIXED to prevent collapse):"
echo "  - Dataset: PubMed abstracts"
echo "  - Mode: SimCSE (text-only contrastive)"
echo "  - Batch size: 16 (effective: 256 with grad accum 16)"
echo "  - Sequence length: 256"
echo "  - Learning rate: 0.0001 (INCREASED from 0.00005)"
echo "  - Temperature: 0.07 (proper for contrastive learning)"
echo "  - Gradient clip: 1.0 (ENABLED for stability)"
echo "  - Warmup steps: 500 (INCREASED)"
echo "  - Max steps: 10,000 (INCREASED for better learning)"
echo "  - Early stopping: validation loss (catches collapse)"
echo "  - Expected runtime: 6-8 hours"
echo ""

cd "${SLURM_SUBMIT_DIR}/hybrid_model_mamba_xlstm"

export HF_HOME="$PWD/.hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCHINDUCTOR_CACHE_DIR="$PWD/.torchinductor"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"
export CUDA_LAUNCH_BLOCKING=0

echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"

# Activate virtual environment
if [ ! -d ".venv" ]; then
    echo "ERROR: Virtual environment .venv not found!"
    exit 1
fi

source .venv/bin/activate

# Verify CUDA
echo ""
echo "Verifying CUDA availability:"
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA device count: {torch.cuda.device_count()}')
if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    print(f'Device name: {torch.cuda.get_device_name(0)}')
"
echo ""

if ! python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    echo "ERROR: CUDA is not available!"
    exit 1
fi

nvidia-smi

echo ""
echo "Starting training with FIXED hyperparameters..."
echo ""

python scripts/train_contrastive.py \
  --config-name config_70m \
  dataset=pubmed \
  trainer=a100_single_gpu \
  trainer.accelerator=cuda \
  trainer.max_epochs=-1 \
  trainer.max_steps=10000 \
  contrastive_mode=simcse \
  dataset.batch_size=16 \
  dataset.eval_batch_size=8 \
  dataset.max_length=256 \
  dataset.max_seq_length=256 \
  dataset.num_workers=2 \
  dataset.preprocessing_num_workers=4 \
  dataset.pin_memory=false \
  trainer.accumulate_grad_batches=16 \
  trainer.val_check_interval=500 \
  trainer.log_every_n_steps=25 \
  callbacks.checkpoint.every_n_train_steps=1000 \
  callbacks.checkpoint.save_top_k=5 \
  callbacks.checkpoint.monitor=val/contrastive_loss \
  callbacks.checkpoint.mode=min \
  callbacks.early_stopping.enabled=true \
  callbacks.early_stopping.monitor=val/contrastive_loss \
  callbacks.early_stopping.patience=5 \
  callbacks.early_stopping.min_delta=0.001 \
  callbacks.early_stopping.mode=min \
  experiment_name=stage1_pubmed_simcse_fixed \
  output_dir=./outputs/stage1_pubmed_simcse_fixed \
  wandb.enabled=false \
  model.learning_rate=0.0001 \
  model.warmup_steps=500 \
  model.weight_decay=0.01 \
  model.gradient_clip_val=1.0 \
  model.contrastive_temperature=0.07

echo ""
echo "=== JOB END (Stage 1: SimCSE FIXED) ==="
echo "Checkpoint saved to: ./outputs/stage1_pubmed_simcse_fixed/checkpoints/"
echo ""
echo "Expected validation loss range: 0.05 - 0.15"
echo "If loss drops below 0.02, model may be collapsing!"
date
