#!/bin/bash
#SBATCH --partition=mitarb
#SBATCH --account=mitarb
#SBATCH --gres=gpu:mitarb:1
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --job-name=stage2_clip_cxr
#SBATCH --output=/scratch/bhushkri/hybrid_xmamba_a100_70m_40/logs/%x_%j.log
#SBATCH --error=/scratch/bhushkri/hybrid_xmamba_a100_70m_40/logs/%x_%j.log
#SBATCH --requeue

# Stage 2: CLIP-style image-text alignment on Indiana CXR.
# Loads the Stage 1 text encoder (SimCSE + PubMedBERT KD).
# Override via: STAGE1_CHECKPOINT=/path/to/ckpt sbatch train_contrastive_stage2.sh

set -euo pipefail

STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-./outputs/hybrid_70m_stage1_kd_pubmedbert/checkpoints/last.ckpt}"

echo "=== JOB START (Stage 2: CLIP on Indiana CXR) ==="
date
echo "Host: $(hostname)"
echo "Submit dir: ${SLURM_SUBMIT_DIR}"
echo "Stage 1 checkpoint: ${STAGE1_CHECKPOINT}"
echo ""

mkdir -p /scratch/bhushkri/hybrid_xmamba_a100_70m_40/logs
cd "${SLURM_SUBMIT_DIR}/hybrid_model_mamba_xlstm"

export HF_HOME="$PWD/.hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCHINDUCTOR_CACHE_DIR="$PWD/.torchinductor"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"
export CUDA_LAUNCH_BLOCKING=0

echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"

if [ ! -d ".venv" ]; then
    echo "ERROR: Virtual environment .venv not found!"
    exit 1
fi
source .venv/bin/activate

echo ""
echo "Verifying CUDA availability:"
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device name: {torch.cuda.get_device_name(0)}')
    mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f'Total VRAM: {mem:.1f} GB')
"
echo ""

nvidia-smi

# Verify Stage 1 checkpoint exists before starting
if [ ! -f "${STAGE1_CHECKPOINT}" ]; then
    echo "ERROR: Stage 1 checkpoint not found at: ${STAGE1_CHECKPOINT}"
    echo "Run train_stage1_distill.sh first, or set STAGE1_CHECKPOINT env variable."
    exit 1
fi
echo "Stage 1 checkpoint verified: ${STAGE1_CHECKPOINT}"

echo ""
echo "Starting Stage 2 CLIP training on Indiana CXR..."

python scripts/train_contrastive.py \
  --config-name config_70m \
  dataset=indiana_cxr \
  trainer=a100_single_gpu \
  contrastive_mode=clip \
  trainer.max_steps=5000 \
  trainer.accumulate_grad_batches=8 \
  trainer.val_check_interval=250 \
  trainer.log_every_n_steps=25 \
  dataset.batch_size=8 \
  dataset.eval_batch_size=8 \
  dataset.num_workers=2 \
  dataset.pin_memory=false \
  lm_checkpoint="${STAGE1_CHECKPOINT}" \
  experiment_name=stage2_indiana_clip \
  output_dir=./outputs/stage2_indiana_clip \
  wandb.enabled=false \
  model.learning_rate=1e-5 \
  model.warmup_steps=200 \
  model.gradient_clip_val=1.0 \
  +model.freeze_text_encoder_steps=500

echo ""
echo "=== JOB END (Stage 2: CLIP complete) ==="
echo "Checkpoint saved to: ./outputs/stage2_indiana_clip/checkpoints/"
date
