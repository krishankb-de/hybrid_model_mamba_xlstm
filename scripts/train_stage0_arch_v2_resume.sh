#!/bin/bash
# Phase 9-EXT RESUME — continue Stage 0 from the 22K-step checkpoint to 40K WITH LR decay.
#
# Context: job 1896 (ext, max_steps=120000) hit the 36h walltime at global_step=22000.
# Eval = PPL 16.62 (exact-match, hybrid) — down from 20.38 @ ~8K and still descending,
# but the WSD decay (scheduled for step ~103K) NEVER fired: LR sat at full 6e-4 the whole
# run. Throughput ~611 opt-steps/h makes 120K infeasible in one block (~8 days).
#
# This run RESUMES from outputs/phase9_stage0_arch_v2_ext/checkpoints/last.ckpt
# (global_step=22000) and trains to max_steps=40000. WSD is rebuilt for max_steps=40000
# in configure_optimizers, so: warmup=1000, decay=5600 (starts step 34400), stable absorbs
# the rest. At the resumed step 22000 we're mid-stable; LR finally anneals over 34400→40000.
# ~18K added steps ≈ 30h at 611/h → fits the 36h block with margin (incl. final val).
#
# NOTE: Lightning's IterableDataset does not resume mid-epoch — the dataloader restarts
# from epoch start (re-sees some PubMed data). Acceptable for LM pretraining.
#
# PPL gate: PubMed val PPL ≤ 13.76 (baseline 13.10 × 1.05). Trajectory makes this plausible
# now that BOTH more steps and the decay are in play.
#SBATCH --partition=mitarb
#SBATCH --account=mitarb
#SBATCH --gres=gpu:mitarb:1
#SBATCH --mem=40G
#SBATCH --time=36:00:00
#SBATCH --job-name=phase9_stage0_v2_resume
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log
#SBATCH --requeue

set -euo pipefail
echo "=== Phase 9-EXT RESUME — Stage 0 22K→40K with WSD decay (hybrid_70m_v2) ==="
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
echo "Resuming Phase 9-EXT Stage 0 from 22K → 40K (decay enabled)..."

python scripts/train_stage0_distill.py \
  model=hybrid_70m_v2 \
  dataset=pubmed \
  trainer=a100_single_gpu \
  distill=stage0_biomedlm \
  trainer.accelerator=cuda \
  trainer.max_epochs=-1 \
  trainer.max_steps=40000 \
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
  +model.beta2_end=0.974 \
  +resume_from_checkpoint=./outputs/phase9_stage0_arch_v2_ext/checkpoints/last.ckpt

echo ""
echo "=== Phase 9-EXT RESUME Stage 0 END ==="
echo "Checkpoint: ./outputs/phase9_stage0_arch_v2_ext/checkpoints/"
date
