#!/bin/bash
#SBATCH --partition=mitarb
#SBATCH --account=mitarb
#SBATCH --gres=gpu:mitarb:1
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --job-name=stage1_kd_resume
#SBATCH --output=/scratch/bhushkri/hybrid_xmamba_a100_70m_40/logs/%x_%j.log
#SBATCH --error=/scratch/bhushkri/hybrid_xmamba_a100_70m_40/logs/%x_%j.log
#SBATCH --requeue

# Stage 1 resume: SimCSE + PubMedBERT KD, resuming from interrupt.ckpt or last.ckpt.
# Uses same config as train_stage1_distill.sh.
# Override backbone: LM_CHECKPOINT=/path sbatch train_contrastive_stage1_resume.sh

set -euo pipefail

LM_CHECKPOINT="${LM_CHECKPOINT:-./outputs/hybrid_70m_stage0_kd_pubmed/checkpoints/stage0_model_only.pt}"
CKPT_DIR="./outputs/hybrid_70m_stage1_kd_pubmedbert/checkpoints"

echo "=== JOB START (Stage 1 Resume: SimCSE + PubMedBERT KD) ==="
date
echo "Host: $(hostname)"
echo "Submit dir: ${SLURM_SUBMIT_DIR}"
echo "LM checkpoint: ${LM_CHECKPOINT}"
echo ""

mkdir -p /scratch/bhushkri/hybrid_xmamba_a100_70m_40/logs
cd "${SLURM_SUBMIT_DIR}/hybrid_model_mamba_xlstm"

export HF_HOME="$PWD/.hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
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

# Verify backbone checkpoint
if [ ! -f "${LM_CHECKPOINT}" ]; then
    echo "ERROR: LM checkpoint not found at: ${LM_CHECKPOINT}"
    exit 1
fi

# Find resume checkpoint: prefer interrupt.ckpt → last.ckpt → latest epoch ckpt
if [ -f "${CKPT_DIR}/interrupt.ckpt" ]; then
    RESUME_CKPT="${CKPT_DIR}/interrupt.ckpt"
    echo "Resuming from: interrupt.ckpt"
elif [ -f "${CKPT_DIR}/last.ckpt" ]; then
    RESUME_CKPT="${CKPT_DIR}/last.ckpt"
    echo "Resuming from: last.ckpt"
else
    RESUME_CKPT=$(ls -t "${CKPT_DIR}"/*.ckpt 2>/dev/null | head -1)
    if [ -z "${RESUME_CKPT}" ]; then
        echo "ERROR: No checkpoint found in ${CKPT_DIR}"
        echo "Run train_stage1_distill.sh first."
        exit 1
    fi
    echo "Resuming from: $(basename ${RESUME_CKPT})"
fi
echo "Checkpoint: ${RESUME_CKPT}"

# Pre-cache PubMedBERT
echo ""
echo "Pre-caching PubMedBERT..."
python -c "
from transformers import AutoModel, AutoTokenizer
import torch
name = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
_ = AutoModel.from_pretrained(name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
_ = AutoTokenizer.from_pretrained(name)
print('PubMedBERT cached.')
"

echo ""
echo "Resuming Stage 1 distillation..."

python scripts/train_contrastive.py \
  --config-name config_70m \
  dataset=pubmed \
  trainer=a100_single_gpu \
  +distill=stage1_pubmedbert \
  contrastive_mode=simcse \
  trainer.max_steps=10000 \
  trainer.accumulate_grad_batches=8 \
  trainer.val_check_interval=500 \
  trainer.log_every_n_steps=25 \
  dataset.batch_size=8 \
  dataset.eval_batch_size=8 \
  dataset.max_length=512 \
  dataset.num_workers=4 \
  dataset.pin_memory=true \
  dataset.streaming=false \
  dataset.cache_dir=/scratch/bhushkri/hybrid_xmamba_a100_70m_40/pubmed_cache \
  lm_checkpoint="${LM_CHECKPOINT}" \
  experiment_name=hybrid_70m_stage1_kd_pubmedbert \
  output_dir=./outputs/hybrid_70m_stage1_kd_pubmedbert \
  wandb.enabled=false \
  model.learning_rate=3e-5 \
  model.warmup_steps=500 \
  model.gradient_clip_val=1.0 \
  +resume_from_checkpoint="${RESUME_CKPT}"

echo ""
echo "=== JOB END (Stage 1 Resume: KD complete) ==="
echo "Checkpoint saved to: ./outputs/hybrid_70m_stage1_kd_pubmedbert/checkpoints/"
date
