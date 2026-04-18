#!/bin/bash
#SBATCH --partition=mitarb
#SBATCH --account=mitarb
#SBATCH --gres=gpu:mitarb:1
#SBATCH --mem=40G
#SBATCH --time=48:00:00
#SBATCH --job-name=stage0_kd_resume
#SBATCH --output=/scratch/bhushkri/hybrid_xmamba_a100_70m_40/logs/%x_%j.log
#SBATCH --error=/scratch/bhushkri/hybrid_xmamba_a100_70m_40/logs/%x_%j.log
#SBATCH --requeue

# Stage 0 RESUME: continue LM pretraining from last checkpoint.
#
# Epoch 0 completed (14,745 optimizer steps, 483M tokens, val/PPL=18.4).
# This job resumes from last.ckpt and trains until max_steps=46000 is reached
# (~2.1 more epochs, ~1.0B additional tokens, ~31K more optimizer steps).
#
# Lightning restores: model weights, optimizer state, LR scheduler state,
# global_step, and epoch count from the checkpoint.  The LR schedule continues
# from where it left off — no cold restart.
#
# Expected total training: ~46K optimizer steps, ~1.5B tokens over ~3 epochs.

set -euo pipefail

mkdir -p /scratch/bhushkri/hybrid_xmamba_a100_70m_40/logs

echo "=== JOB START (Stage 0 RESUME: LM pretrain + BioMedLM KD) ==="
date
echo "Host: $(hostname)"
echo "Submit dir: ${SLURM_SUBMIT_DIR}"
echo ""

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

# Verify checkpoint exists before starting
CKPT_PATH="./outputs/hybrid_70m_stage0_kd_pubmed/checkpoints/last.ckpt"
if [ ! -f "${CKPT_PATH}" ]; then
    echo "ERROR: Checkpoint not found at ${CKPT_PATH}"
    echo "Available checkpoints:"
    ls ./outputs/hybrid_70m_stage0_kd_pubmed/checkpoints/ 2>/dev/null || echo "  (none)"
    exit 1
fi
echo "Resuming from checkpoint: ${CKPT_PATH}"
echo ""

nvidia-smi

# BioMedLM is already cached from the first run — this is a fast no-op
echo "Verifying BioMedLM cache..."
python -c "
from transformers import AutoModelForCausalLM
import torch
print('Loading BioMedLM from cache...')
_ = AutoModelForCausalLM.from_pretrained(
    'stanford-crfm/BioMedLM',
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)
print('BioMedLM ready.')
"

echo ""
echo "Starting Stage 0 distillation resume..."

python scripts/train_stage0_distill_resume.py \
  model=hybrid_70m \
  dataset=pubmed \
  trainer=a100_single_gpu \
  distill=stage0_biomedlm \
  trainer.max_steps=46000 \
  trainer.max_epochs=4 \
  trainer.accumulate_grad_batches=8 \
  trainer.val_check_interval=500 \
  trainer.log_every_n_steps=25 \
  trainer.compile_model=false \
  dataset.batch_size=8 \
  dataset.eval_batch_size=8 \
  dataset.max_length=512 \
  dataset.max_seq_length=512 \
  dataset.num_workers=2 \
  dataset.preprocessing_num_workers=2 \
  dataset.pin_memory=true \
  dataset.target_tokens=1500000000 \
  dataset.cache_dir=/scratch/bhushkri/hybrid_xmamba_a100_70m_40/pubmed_cache \
  callbacks.checkpoint.every_n_train_steps=500 \
  callbacks.checkpoint.save_top_k=3 \
  experiment_name=hybrid_70m_stage0_kd_pubmed \
  output_dir=./outputs/hybrid_70m_stage0_kd_pubmed \
  wandb.enabled=false \
  model.learning_rate=6.0e-4 \
  model.warmup_steps=1000 \
  model.gradient_clip_val=1.0 \
  +ckpt_path="${CKPT_PATH}"

echo ""
echo "=== JOB END (Stage 0 RESUME: KD complete) ==="
echo "Checkpoint saved to: ./outputs/hybrid_70m_stage0_kd_pubmed/checkpoints/"
date
