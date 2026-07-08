#!/bin/bash
# ============================================================================
# H100 Stage-0 LM pre-train template (hybrid_70m_v2 + BioMedLM KD).
# H100 port of train_stage0_arch_v2_ext.sh. Phase 5 clones this for 150M
# (train_stage0_150m_h100.sh: model=hybrid_150m_v2, larger max_steps/batch).
#
# H100 changes vs the A100 script:
#   - partition aisc-batch (7-day cap) → the full run finishes in ONE block
#     (no requeue juggling; A100 hit 16-36h walltime kills mid-run).
#   - --gpus=1 (H100 node via --exclude=ga03,gx17v1) ; larger --mem.
#   - trainer=h100_single_gpu.
#   - BATCH_SIZE/ACCUM/GRAD_CKPT are env-overridable. aisc H100 = 80GB (NOT the
#     94/141GB the plan first assumed), so 150M needs GRAD_CKPT=true + accum to
#     fit (see train_stage0_150m_h100.sh). 70M fits larger microbatch bare.
#   - compile_model=false: Stage-0 uses the cu_seqlens doc-boundary segmented
#     scan (data-dependent loop) which graph-breaks under torch.compile.
#
# Adjust the two ENV placeholders for the aisc/H100 box:
#   SCRATCH_ROOT — fast scratch for HF/dataset/checkpoint cache.
#   VENV_ACTIVATE — path to the py>=3.10 env activate script (venv or conda).
#
# PPL gate (informational per plan): PubMed val PPL — target <= 13.76.
# ============================================================================
#SBATCH --partition=aisc-batch
#SBATCH --account=aisc
#SBATCH --gpus=1
#SBATCH --exclude=ga03,gx17v1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH --job-name=h100_stage0_v2
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log
#SBATCH --requeue

set -euo pipefail

SCRATCH_ROOT="${SCRATCH_ROOT:-/sc/scratch/$USER/hybrid_xmamba_h100}"
VENV_ACTIVATE="${VENV_ACTIVATE:-.venv/bin/activate}"
MODEL_CONFIG="${MODEL_CONFIG:-hybrid_70m_v2}"
MAX_STEPS="${MAX_STEPS:-120000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
ACCUM="${ACCUM:-1}"                 # grad-accum: eff batch = BATCH_SIZE*ACCUM
GRAD_CKPT="${GRAD_CKPT:-false}"     # gradient checkpointing (trades compute for VRAM)
EXPERIMENT="${EXPERIMENT:-h100_stage0_${MODEL_CONFIG}}"

echo "=== H100 Stage-0 pre-train: ${MODEL_CONFIG} + BioMedLM KD ==="
date; hostname
mkdir -p logs

cd "${SLURM_SUBMIT_DIR}/hybrid_model_mamba_xlstm" 2>/dev/null || cd "${SLURM_SUBMIT_DIR:-.}"

export HF_HOME="${SCRATCH_ROOT}/.hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCHINDUCTOR_CACHE_DIR="${SCRATCH_ROOT}/.torchinductor"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_LAUNCH_BLOCKING=0
export PYTHONUNBUFFERED=1   # flush progress bar to the log live (else block-buffered → looks frozen)

source "${VENV_ACTIVATE}"
python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print('GPU:', torch.cuda.get_device_name(0), f'{torch.cuda.get_device_properties(0).total_memory/1024**3:.0f}GB')"
nvidia-smi

echo "Pre-caching BioMedLM (2.7B)..."
python -c "
from transformers import AutoModelForCausalLM
import torch
_ = AutoModelForCausalLM.from_pretrained('stanford-crfm/BioMedLM', torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
print('BioMedLM cached.')
"

echo "Starting Stage-0 distillation (${MAX_STEPS} steps, bs=${BATCH_SIZE}, accum=1)..."
python scripts/train_stage0_distill.py \
  model=${MODEL_CONFIG} \
  dataset=pubmed \
  trainer=h100_single_gpu \
  distill=stage0_biomedlm \
  trainer.accelerator=cuda \
  trainer.max_epochs=-1 \
  trainer.max_steps=${MAX_STEPS} \
  trainer.accumulate_grad_batches=${ACCUM} \
  trainer.val_check_interval=2000 \
  trainer.log_every_n_steps=25 \
  trainer.compile_model=false \
  dataset.batch_size=${BATCH_SIZE} \
  dataset.eval_batch_size=${BATCH_SIZE} \
  dataset.max_length=512 \
  dataset.max_seq_length=512 \
  dataset.num_workers=4 \
  dataset.preprocessing_num_workers=8 \
  dataset.pin_memory=true \
  dataset.cache_dir="${SCRATCH_ROOT}/pubmed_cache" \
  callbacks.checkpoint.every_n_train_steps=2000 \
  callbacks.checkpoint.save_top_k=3 \
  experiment_name=${EXPERIMENT} \
  output_dir=./outputs/${EXPERIMENT} \
  wandb.enabled=false \
  model.learning_rate=6.0e-4 \
  model.warmup_steps=1000 \
  model.gradient_clip_val=1.0 \
  model.use_gradient_checkpointing=${GRAD_CKPT} \
  +model.scheduler_name=wsd \
  +model.beta2_schedule=true \
  +model.beta2_start=0.999 \
  +model.beta2_end=0.974

echo "=== END: checkpoints in ./outputs/${EXPERIMENT}/checkpoints/ ==="
date
