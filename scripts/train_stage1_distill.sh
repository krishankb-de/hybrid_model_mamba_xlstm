#!/bin/bash
#SBATCH --partition=mitarb
#SBATCH --account=mitarb
#SBATCH --gres=gpu:mitarb:1
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --job-name=stage1_kd_pubmedbert
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log

# Stage 1: SimCSE contrastive training on PubMed with PubMedBERT embedding KD.
#
# Teacher  : PubMedBERT (110M, WordPiece tokenizer)
# Student  : hybrid_70m text encoder (512 dim, [mamba, mamba, mlstm])
# Loss     : L_SimCSE + lambda * (1 - cos(student_pooled, teacher_cls))
#            lambda ramps 0 → 0.3 over steps 500-1000
#
# Requires a Stage 0 checkpoint (baseline or KD variant).
# Set lm_checkpoint below or pass via CLI.

set -euo pipefail

LM_CHECKPOINT="${LM_CHECKPOINT:-./outputs/hybrid_70m_stage0_kd_biomedlm/checkpoints/last.ckpt}"

echo "=== JOB START (Stage 1: SimCSE + PubMedBERT KD) ==="
date
echo "Host: $(hostname)"
echo "LM checkpoint: ${LM_CHECKPOINT}"
echo ""

cd "${SLURM_SUBMIT_DIR}/hybrid_model_mamba_xlstm"

export HF_HOME="$PWD/.hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"

if [ ! -d ".venv" ]; then
    echo "ERROR: Virtual environment .venv not found!"
    exit 1
fi
source .venv/bin/activate

nvidia-smi

# Pre-cache PubMedBERT
echo "Pre-caching PubMedBERT..."
python -c "
from transformers import AutoModel, AutoTokenizer
import torch
name = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
_ = AutoModel.from_pretrained(name, torch_dtype=torch.bfloat16)
_ = AutoTokenizer.from_pretrained(name)
print('PubMedBERT cached.')
"

python scripts/train_contrastive.py \
  --config-name config_70m \
  dataset=pubmed \
  trainer=a100_single_gpu \
  distill=stage1_pubmedbert \
  contrastive_mode=simcse \
  trainer.max_steps=10000 \
  trainer.accumulate_grad_batches=1 \
  trainer.val_check_interval=500 \
  trainer.log_every_n_steps=25 \
  dataset.batch_size=64 \
  dataset.eval_batch_size=64 \
  dataset.max_length=512 \
  dataset.num_workers=4 \
  dataset.pin_memory=true \
  lm_checkpoint="${LM_CHECKPOINT}" \
  experiment_name=hybrid_70m_stage1_kd_pubmedbert \
  output_dir=./outputs/hybrid_70m_stage1_kd_pubmedbert \
  wandb.enabled=false \
  model.learning_rate=1.0e-4 \
  model.warmup_steps=500 \
  model.gradient_clip_val=1.0

echo ""
echo "=== JOB END (Stage 1: KD complete) ==="
echo "Checkpoint: ./outputs/hybrid_70m_stage1_kd_pubmedbert/checkpoints/"
date
