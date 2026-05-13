#!/bin/bash
# Phase 9D Isolation Run A — Phase 3 only (mLSTM stabilization, NO HybridNorm, NO v2 pattern)
#
# Purpose: determine whether the Phase 9 PPL regression (20.38 vs baseline 13.10)
# is caused by the v2 layer pattern (Phase 5) or HybridNorm (Phase 4).
# This run isolates Phase 3 (gate soft-cap + i_gate bias=-10 + LSE m_t) only.
#
# Matches Phase 9 exactly EXCEPT:
#   model=hybrid_70m      → cycling [mamba,mamba,mlstm] (Phase 5 layer pattern OFF)
#   norm_topology=pre_rms → no post-FFN norm, no v_norm, no dt/B/C pre-norms (Phase 4 OFF)
#
# Phase 3 stabilization IS active (baked into mLSTM block code):
#   tanh soft-cap 15 on i/f gates, i_gate_proj.bias=-10, LSE m_t stabilizer
#
# All other settings identical to Phase 9 (Phase 6 cu_seqlens, Phase 7 WSD+β2,
# BioMedLM KD, 50K steps, bs=8/accum=8).
#
# PPL gate: ≤ 13.76 (baseline 13.10 × 1.05)
# If PASS: Phase 4+5 caused regression → keep Phase 3 only, skip to Phase 10
# If FAIL: regression is in Phase 3 or training setup → escalate to Phase 12
#SBATCH --partition=mitarb
#SBATCH --account=mitarb
#SBATCH --gres=gpu:mitarb:1
#SBATCH --mem=40G
#SBATCH --time=16:00:00
#SBATCH --job-name=iso_p3only
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log
#SBATCH --requeue

set -euo pipefail
echo "=== Phase 9D Isolation A — Phase 3 only: hybrid_70m + BioMedLM KD ==="
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
echo "Starting Phase 9D Isolation A (Phase 3 only)..."

python scripts/train_stage0_distill.py \
  model=hybrid_70m \
  dataset=pubmed \
  trainer=a100_single_gpu \
  distill=stage0_biomedlm \
  trainer.accelerator=cuda \
  trainer.max_epochs=-1 \
  trainer.max_steps=50000 \
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
  experiment_name=iso_p3only \
  output_dir=./outputs/iso_p3only \
  wandb.enabled=false \
  model.norm_topology=pre_rms \
  model.learning_rate=6.0e-4 \
  model.max_steps=50000 \
  model.warmup_steps=500 \
  model.gradient_clip_val=1.0 \
  model.use_gradient_checkpointing=true \
  +model.scheduler_name=wsd \
  +model.beta2_schedule=true \
  +model.beta2_start=0.999 \
  +model.beta2_end=0.974

echo ""
echo "=== Phase 9D Isolation A END ==="
echo "Checkpoint: ./outputs/iso_p3only/checkpoints/"
date
