#!/bin/bash
# Phase 9 — Stage 0 LM re-pretrain on PubMed with BioMedLM KD (hybrid_70m_v2)
#
# Winner from Phase 8C: hybrid_70m_v2 ([m,m,m,L,L,m,m,m]) ratio=0.998 at step 1500.
# All architectural fixes active:
#   Phase 3: mLSTM gate soft-cap=15, i_gate_bias=-10
#   Phase 4: norm_topology=hybrid (baked into hybrid_70m_v2.yaml)
#   Phase 6: cu_seqlens doc-boundary resets (PubMed packing)
#   Phase 7: WSD scheduler (1%/85%/14%) + β2 anneal 0.999→0.974
#
# Walltime estimate (from Phase 8 throughput 0.25 it/s with gc=True):
#   10K training steps × 4 s/step  =  40000 s = 11.1 h
#   10 val rounds (val_check=1000)  ×  10 min = 100 min = 1.7 h
#   BioMedLM pre-cache             ≈ 10 min
#   Total                          ≈ 13 h  → 16 h walltime for margin
#
# PPL gate: val/loss (PubMed) ≤ 13.76 nats  (baseline × 1.05; baseline=13.10)
# OR ≥ 10% improvement → PPL ≤ 11.79
#SBATCH --partition=mitarb
#SBATCH --account=mitarb
#SBATCH --gres=gpu:mitarb:1
#SBATCH --mem=40G
#SBATCH --time=16:00:00
#SBATCH --job-name=phase9_stage0_v2
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log
#SBATCH --requeue

set -euo pipefail
echo "=== Phase 9 — Stage 0 re-pretrain: hybrid_70m_v2 + BioMedLM KD ==="
date; hostname

cd "${SLURM_SUBMIT_DIR}/hybrid_model_mamba_xlstm"

export HF_HOME="$PWD/.hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCHINDUCTOR_CACHE_DIR="$PWD/.torchinductor"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"
export CUDA_LAUNCH_BLOCKING=0

source .venv/bin/activate
python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'"
nvidia-smi

# Pre-cache BioMedLM before training (avoids HF download timeout mid-step)
echo ""
echo "Pre-caching BioMedLM (2.7B)..."
python -c "
from transformers import AutoModelForCausalLM
import torch
_ = AutoModelForCausalLM.from_pretrained(
    'stanford-crfm/BioMedLM',
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)
print('BioMedLM cached.')
"

echo ""
echo "Starting Phase 9 Stage 0 distillation..."

python scripts/train_stage0_distill.py \
  model=hybrid_70m_v2 \
  dataset=pubmed \
  trainer=a100_single_gpu \
  distill=stage0_biomedlm \
  trainer.accelerator=cuda \
  trainer.max_epochs=-1 \
  trainer.max_steps=10000 \
  trainer.accumulate_grad_batches=8 \
  trainer.val_check_interval=1000 \
  trainer.log_every_n_steps=25 \
  trainer.compile_model=false \
  dataset.batch_size=8 \
  dataset.eval_batch_size=8 \
  dataset.max_length=512 \
  dataset.max_seq_length=512 \
  dataset.num_workers=2 \
  dataset.preprocessing_num_workers=4 \
  dataset.pin_memory=true \
  dataset.cache_dir=/scratch/bhushkri/hybrid_xmamba_a100_70m_40/pubmed_cache \
  callbacks.checkpoint.every_n_train_steps=500 \
  callbacks.checkpoint.save_top_k=3 \
  experiment_name=phase9_stage0_arch_v2 \
  output_dir=./outputs/phase9_stage0_arch_v2 \
  wandb.enabled=false \
  model.learning_rate=6.0e-4 \
  model.warmup_steps=100 \
  model.gradient_clip_val=1.0 \
  model.use_gradient_checkpointing=true \
  +model.scheduler_name=wsd \
  +model.beta2_schedule=true \
  +model.beta2_start=0.999 \
  +model.beta2_end=0.974

echo ""
echo "=== Phase 9 Stage 0 END ==="
echo "Checkpoint: ./outputs/phase9_stage0_arch_v2/checkpoints/"
date
