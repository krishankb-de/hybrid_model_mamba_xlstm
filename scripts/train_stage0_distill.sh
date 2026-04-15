#!/bin/bash
#SBATCH --partition=mitarb
#SBATCH --account=mitarb
#SBATCH --gres=gpu:mitarb:1
#SBATCH --mem=40G
#SBATCH --time=48:00:00
#SBATCH --job-name=stage0_kd_biomedlm
#SBATCH --output=/scratch/bhushkri/hybrid_xmamba_a100_70m_40/logs/%x_%j.log
#SBATCH --error=/scratch/bhushkri/hybrid_xmamba_a100_70m_40/logs/%x_%j.log
#SBATCH --requeue

# Stage 0: LM pretraining on PubMed with BioMedLM knowledge distillation.
#
# Teacher : BioMedLM (2.7B, GPT-2 BPE — same vocab as student)
# Student : hybrid_70m (512 dim, 8 layers, [mamba, mamba, mlstm])
# Loss    : (1-alpha)*CE + alpha*hidden-state cosine KD, alpha=0.5
#
# A100 40GB memory estimate at bs=16, seq=512, compile=off:
#   Student 70M (grads+Adam) : ~0.84 GB
#   Teacher 2.7B frozen bf16 : ~5.4 GB weights + ~10 GB acts
#   Total peak               : ~22-26 GB  (safe on 40 GB)

set -euo pipefail

mkdir -p /scratch/bhushkri/hybrid_xmamba_a100_70m_40/logs

echo "=== JOB START (Stage 0: LM pretrain + BioMedLM KD) ==="
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

nvidia-smi

# Pre-download BioMedLM to shared cache before training starts
# (avoids timeout during the first training step)
echo "Pre-caching BioMedLM..."
python -c "
from transformers import AutoModelForCausalLM
import torch
print('Downloading BioMedLM (2.7B)...')
_ = AutoModelForCausalLM.from_pretrained(
    'stanford-crfm/BioMedLM',
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)
print('BioMedLM cached successfully.')
"

echo ""
echo "Starting Stage 0 distillation..."

python scripts/train_stage0_distill.py \
  model=hybrid_70m \
  dataset=pubmed \
  trainer=a100_single_gpu \
  distill=stage0_biomedlm \
  trainer.max_steps=46000 \
  trainer.accumulate_grad_batches=4 \
  trainer.val_check_interval=500 \
  trainer.log_every_n_steps=25 \
  trainer.compile_model=false \
  dataset.batch_size=16 \
  dataset.eval_batch_size=16 \
  dataset.max_length=512 \
  dataset.max_seq_length=512 \
  dataset.num_workers=4 \
  dataset.preprocessing_num_workers=4 \
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
  model.gradient_clip_val=1.0

echo ""
echo "=== JOB END (Stage 0: KD complete) ==="
echo "Checkpoint saved to: ./outputs/hybrid_70m_stage0_kd_pubmed/checkpoints/"
date
