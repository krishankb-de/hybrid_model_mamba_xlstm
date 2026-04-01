#!/bin/bash
#SBATCH --partition=mitarb
#SBATCH --account=mitarb
#SBATCH --gres=gpu:mitarb:1
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --job-name=simcse_resume
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log

set -euo pipefail

echo "=== JOB START (Stage 1: SimCSE Resume Training) ==="
date
echo "Host: $(hostname)"
echo "Submit dir: ${SLURM_SUBMIT_DIR}"
echo ""
echo "Configuration:"
echo "  - Resuming from checkpoint"
echo "  - Dataset: PubMed abstracts"
echo "  - Mode: SimCSE (text-only contrastive)"
echo "  - Batch size: 16 (effective: 256 with grad accum 16)"
echo "  - Sequence length: 256"
echo "  - Learning rate: 2e-5 (REDUCED for fine-tuning)"
echo "  - Max steps: 20,000 (INCREASED from 5,000)"
echo "  - Target: val loss < 0.15"
echo "  - Expected additional runtime: ~4-5 hours"
echo ""

cd "${SLURM_SUBMIT_DIR}/hybrid_model_mamba_xlstm"

export HF_HOME="$PWD/.hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCHINDUCTOR_CACHE_DIR="$PWD/.torchinductor"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"
export CUDA_LAUNCH_BLOCKING=0

echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"

source .venv/bin/activate

# Verify CUDA
echo ""
echo "Verifying CUDA availability:"
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    print(f'Device name: {torch.cuda.get_device_name(0)}')
"
echo ""

if ! python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    echo "ERROR: CUDA is not available!"
    exit 1
fi

# Find the latest checkpoint
CHECKPOINT_DIR="./outputs/stage1_pubmed_simcse/checkpoints"
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "ERROR: Checkpoint directory not found: $CHECKPOINT_DIR"
    echo "Please run Stage 1 training first."
    exit 1
fi

# Use last.ckpt if it exists, otherwise find the latest checkpoint
if [ -f "$CHECKPOINT_DIR/last.ckpt" ]; then
    CHECKPOINT="$CHECKPOINT_DIR/last.ckpt"
    echo "Resuming from: last.ckpt"
else
    CHECKPOINT=$(ls -t $CHECKPOINT_DIR/*.ckpt 2>/dev/null | head -1)
    if [ -z "$CHECKPOINT" ]; then
        echo "ERROR: No checkpoint files found in $CHECKPOINT_DIR"
        exit 1
    fi
    echo "Resuming from: $(basename $CHECKPOINT)"
fi

echo "Checkpoint path: $CHECKPOINT"
echo ""

nvidia-smi

python scripts/train_contrastive.py \
  --config-name config_70m \
  dataset=pubmed \
  trainer=a100_single_gpu \
  trainer.accelerator=cuda \
  trainer.max_epochs=-1 \
  trainer.max_steps=20000 \
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
  callbacks.checkpoint.save_top_k=3 \
  callbacks.early_stopping.enabled=true \
  callbacks.early_stopping.monitor=train/contrastive_loss_epoch \
  callbacks.early_stopping.patience=3 \
  callbacks.early_stopping.min_delta=0.05 \
  experiment_name=stage1_pubmed_simcse \
  output_dir=./outputs/stage1_pubmed_simcse \
  wandb.enabled=false \
  model.learning_rate=0.00003 \
  model.warmup_steps=200 \
  model.gradient_clip_val=0.0 \
  lm_checkpoint="$CHECKPOINT"

echo ""
echo "=== JOB END (Stage 1: SimCSE Resume) ==="
echo "Checkpoint saved to: ./outputs/stage1_pubmed_simcse/checkpoints/"
date
