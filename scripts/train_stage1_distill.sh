#!/bin/bash
#SBATCH --partition=mitarb
#SBATCH --account=mitarb
#SBATCH --gres=gpu:mitarb:1
#SBATCH --mem=40G
#SBATCH --time=36:00:00
#SBATCH --job-name=stage1_kd_pubmedbert
#SBATCH --output=/scratch/bhushkri/hybrid_xmamba_a100_70m_40/logs/%x_%j.log
#SBATCH --error=/scratch/bhushkri/hybrid_xmamba_a100_70m_40/logs/%x_%j.log
#SBATCH --requeue

# Stage 1: SimCSE contrastive training on PubMed with PubMedBERT embedding KD.
#
# Teacher  : PubMedBERT (110M, WordPiece tokenizer)
# Student  : hybrid_70m text encoder (512 dim, [mamba, mamba, mlstm])
# Loss     : L = 1 - cos(distill_proj(student), pubmedbert_cls)  [pure KD, no SimCSE]
#            lambda=1.0 from step 0 (no warmup, no ramp)
#
# Memory estimate at bs=16, seq=512, accum=4, bf16 + gradient checkpointing:
#   Student 70M (weights+grads+Adam) : ~1.0 GB
#   PubMedBERT 110M frozen bf16     : ~0.2 GB weights + ~0.4 GB acts (bs=8)
#   Mamba scan intermediates x1 fwd : ~7-8 GB (single pass, no SimCSE doubling)
#   Total peak (estimate)           : ~12-15 GB with checkpointing (safe on 40GB)
#   Effective batch = 16 × 4 = 64
# NOTE: OOM with bs=16 no-ckpt was caused by SimCSE 2x activations. Pure KD
#   uses 1x activations, so bs=16 + gradient checkpointing is safe and fast.
#
# Stage 1 hyperparameters (post run 1209 STS-B decline analysis):
#   Pure KD replaces SimCSE. Stage 0 backbone (PPL=13) already separates PubMed
#     abstracts — InfoNCE loss was ~0.002 from step 1 (no gradient signal).
#   lambda_max=1.0: full KD from step 0. PubMedBERT scores BIOSSES=0.85;
#     expected student BIOSSES: 0.72-0.80.



#
# Requires a Stage 0 checkpoint. Default path matches train_stage0_distill.sh output.
# Override via: LM_CHECKPOINT=/path/to/ckpt sbatch train_stage1_distill.sh

set -euo pipefail

LM_CHECKPOINT="${LM_CHECKPOINT:-./outputs/hybrid_70m_stage0_kd_pubmed/checkpoints/stage0_model_only.pt}"

echo "=== JOB START (Stage 1: SimCSE + PubMedBERT KD) ==="
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

# Verify Stage 0 checkpoint exists before starting
if [ ! -f "${LM_CHECKPOINT}" ]; then
    echo "ERROR: Stage 0 checkpoint not found at: ${LM_CHECKPOINT}"
    echo "Run train_stage0_distill.sh first, or set LM_CHECKPOINT env variable."
    exit 1
fi
echo "Stage 0 checkpoint verified: ${LM_CHECKPOINT}"

# Pre-cache PubMedBERT before training starts to avoid timeout mid-run
echo ""
echo "Pre-caching PubMedBERT..."
python -c "
from transformers import AutoModel, AutoTokenizer
import torch
name = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
print(f'Downloading {name}...')
_ = AutoModel.from_pretrained(name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
_ = AutoTokenizer.from_pretrained(name)
print('PubMedBERT cached successfully.')
"

echo ""
echo "Starting Stage 1 distillation..."

python scripts/train_contrastive.py \
  --config-name config_70m \
  dataset=pubmed \
  trainer=a100_single_gpu \
  +distill=stage1_pubmedbert \
  contrastive_mode=simcse \
  trainer.max_steps=20000 \
  trainer.accumulate_grad_batches=4 \
  trainer.val_check_interval=1000 \
  trainer.log_every_n_steps=25 \
  dataset.batch_size=16 \
  dataset.eval_batch_size=16 \
  dataset.max_length=512 \
  dataset.num_workers=2 \
  dataset.pin_memory=true \
  dataset.streaming=false \
  dataset.cache_dir=/scratch/bhushkri/hybrid_xmamba_a100_70m_40/pubmed_cache \
  lm_checkpoint="${LM_CHECKPOINT}" \
  experiment_name=hybrid_70m_stage1_kd_pubmedbert \
  output_dir=./outputs/hybrid_70m_stage1_kd_pubmedbert \
  wandb.enabled=false \
  model.learning_rate=1e-5 \
  model.warmup_steps=1000 \
  model.gradient_clip_val=1.0 \
  model.use_gradient_checkpointing=true \
  model.proj_head_dropout=0.2

echo ""
echo "=== JOB END (Stage 1: KD complete) ==="
echo "Checkpoint saved to: ./outputs/hybrid_70m_stage1_kd_pubmedbert/checkpoints/"
date
