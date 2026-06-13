#!/bin/bash
# Phase 9F-EXT — Stage 0 LM re-pretrain on PubMed, EXTENDED to 120K steps (hybrid_70m_v2).
#
# Why extended: job 1884 (50K steps, config CORRECT — norm_topology=hybrid active) hit
# train-val PPL 23.0 with the curve STILL DESCENDING at step 50K. Gate is 13.76 (baseline
# 13.10 measured under identical protocol: PubMed-val, 1000 samples, max_length 512, batch 16).
# Baseline reached 13.10 with ~117K cumulative steps; 50K-from-scratch is undertrained.
# This run targets ~120K steps to fairly match the baseline and give the gate a real shot.
#
# All architectural fixes active (unchanged from 1884):
#   Phase 3: mLSTM gate soft-cap=15, i_gate_bias=-10
#   Phase 4: norm_topology=hybrid (baked into hybrid_70m_v2.yaml; threading bug fixed in 9F)
#   Phase 6: cu_seqlens doc-boundary resets (PubMed packing)
#   Phase 7: WSD scheduler (warmup=1000 abs / stable / 14% decay) + β2 anneal 0.999→0.974
#            With max_steps=120000: warmup=1000, decay=16800 (starts step 103200), stable=102200.
#
# Walltime: job 1884 ran ~1.11 it/s → 120K steps ≈ 30h training + val. val_check_interval
#   raised 1000→2000 to halve validation overhead on the long run. --time=36h for margin.
#   NOTE: resume-on-requeue is NOT wired (trainer.fit has no ckpt_path) — this must fit in
#   one walltime block. If the mitarb partition caps below ~36h, tell me and I'll wire
#   ckpt_path resume so --requeue can chain.
#
# PPL gate: PubMed val PPL ≤ 13.76 (baseline × 1.05; baseline=13.10). Stretch: ≤ 11.79.
#SBATCH --partition=mitarb
#SBATCH --account=mitarb
#SBATCH --gres=gpu:mitarb:1
#SBATCH --mem=40G
#SBATCH --time=36:00:00
#SBATCH --job-name=phase9_stage0_v2_ext
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log
#SBATCH --requeue

set -euo pipefail
echo "=== Phase 9F-EXT — Stage 0 re-pretrain (120K): hybrid_70m_v2 + BioMedLM KD ==="
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
echo "Starting Phase 9F-EXT Stage 0 distillation (120K steps)..."

python scripts/train_stage0_distill.py \
  model=hybrid_70m_v2 \
  dataset=pubmed \
  trainer=a100_single_gpu \
  distill=stage0_biomedlm \
  trainer.accelerator=cuda \
  trainer.max_epochs=-1 \
  trainer.max_steps=120000 \
  trainer.accumulate_grad_batches=8 \
  trainer.val_check_interval=2000 \
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
  callbacks.checkpoint.every_n_train_steps=2000 \
  callbacks.checkpoint.save_top_k=3 \
  experiment_name=phase9_stage0_arch_v2_ext \
  output_dir=./outputs/phase9_stage0_arch_v2_ext \
  wandb.enabled=false \
  model.learning_rate=6.0e-4 \
  model.warmup_steps=1000 \
  model.gradient_clip_val=1.0 \
  model.use_gradient_checkpointing=true \
  +model.scheduler_name=wsd \
  +model.beta2_schedule=true \
  +model.beta2_start=0.999 \
  +model.beta2_end=0.974

echo ""
echo "=== Phase 9F-EXT Stage 0 END ==="
echo "Checkpoint: ./outputs/phase9_stage0_arch_v2_ext/checkpoints/"
date
