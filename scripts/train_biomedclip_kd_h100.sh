#!/bin/bash
# ============================================================================
# H100 joint-contrastive template (canonical recipe on hybrid_70m_v2 backbone).
# H100 port of train_biomedclip_kd_phase15_pure.sh, but with the CANONICAL recipe
# (freq_kd=false, vit_unfreeze_blocks=2). Phase 6 clones this for 150M with a
# batch sweep (train_biomedclip_kd_150m_h100.sh).
#
# THE H100 RETRIEVAL LEVER: dataset.batch_size=128 at accumulate_grad_batches=1.
# The CLIP loss uses in-batch negatives only (no all_gather, moco=0), so the true
# negative count == per-step batch. A100 was capped at 32 (~31 negatives); H100
# 141GB lifts this to 128 (~127 negatives) — the single biggest MIMIC lever, and
# it also cuts epoch count on the 27.5k-pair set (less overfitting). Grad-accum
# does NOT add negatives, so keep accum=1 and scale batch_size instead.
#
# LR is sqrt-scaled for the 4x batch (32->128): backbone 1e-5->2e-5, head 3e-4->6e-4.
#
# Kill gates: cos_text_teacher >= 0.85 by step 1000; val/clip_loss < 3.0 by 1000;
#   MIMIC R@10 >= 0.1045 (current best) by step 3000; ViT group lr == 1e-6.
#
# ENV placeholders (adjust for aisc/H100): SCRATCH_ROOT, VENV_ACTIVATE, STAGE0_CKPT.
# ============================================================================
#SBATCH --partition=aisc-shortrun
#SBATCH --account=aisc
#SBATCH --gpus=1
#SBATCH --exclude=ga03,gx17v1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --time=16:00:00
#SBATCH --job-name=h100_kd_contrastive
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log
#SBATCH --requeue

set -euo pipefail

SCRATCH_ROOT="${SCRATCH_ROOT:-/sc/scratch/$USER/hybrid_xmamba_h100}"
VENV_ACTIVATE="${VENV_ACTIVATE:-.venv/bin/activate}"
MODEL_CONFIG="${MODEL_CONFIG:-hybrid_70m_v2}"
BATCH_SIZE="${BATCH_SIZE:-128}"
STAGE0_CKPT="${STAGE0_CKPT:-./outputs/h100_stage0_${MODEL_CONFIG}/checkpoints/stage0_model_only.pt}"
MIMIC_CACHE_DIR="${MIMIC_CACHE_DIR:-${SCRATCH_ROOT}/mimic_cxr_cache}"
EXPERIMENT="${EXPERIMENT:-h100_kd_${MODEL_CONFIG}_bs${BATCH_SIZE}}"

echo "=== H100 joint contrastive: ${MODEL_CONFIG}, bs=${BATCH_SIZE} (true negatives) ==="
date; hostname
mkdir -p logs "${MIMIC_CACHE_DIR}"

cd "${SLURM_SUBMIT_DIR}/hybrid_model_mamba_xlstm" 2>/dev/null || cd "${SLURM_SUBMIT_DIR:-.}"

export HF_HOME="${SCRATCH_ROOT}/.hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCHINDUCTOR_CACHE_DIR="${SCRATCH_ROOT}/.torchinductor"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_LAUNCH_BLOCKING=0

source "${VENV_ACTIVATE}"
python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print('GPU:', torch.cuda.get_device_name(0), f'{torch.cuda.get_device_properties(0).total_memory/1024**3:.0f}GB')"
nvidia-smi

if [ ! -f "${STAGE0_CKPT}" ]; then
  echo "ERROR: Stage-0 checkpoint not found: ${STAGE0_CKPT}"
  echo "Extract it from the Stage-0 run (strip 'model.' prefix into a *_model_only.pt)."
  exit 1
fi
echo "Stage-0 checkpoint: ${STAGE0_CKPT}"

echo "Starting joint contrastive (canonical: freq_kd=false, vit_unfreeze=2, moco=0)..."
python scripts/train_contrastive.py \
  --config-name config_70m \
  model=${MODEL_CONFIG} \
  dataset=mimic_cxr \
  +distill=biomedclip_kd_joint_v2 \
  distill.freq_kd=false \
  distill.vit_unfreeze_blocks=2 \
  distill.vit_lr=1e-6 \
  distill.backbone_lr=2e-5 \
  distill.head_lr=6e-4 \
  trainer=h100_single_gpu \
  contrastive_mode=joint \
  trainer.max_steps=5000 \
  trainer.accumulate_grad_batches=1 \
  trainer.val_check_interval=250 \
  trainer.log_every_n_steps=25 \
  trainer.compile_model=true \
  dataset.batch_size=${BATCH_SIZE} \
  dataset.eval_batch_size=${BATCH_SIZE} \
  dataset.num_workers=8 \
  dataset.pin_memory=true \
  dataset.cache_dir="${MIMIC_CACHE_DIR}" \
  model.use_gradient_checkpointing=false \
  lm_checkpoint="${STAGE0_CKPT}" \
  experiment_name=${EXPERIMENT} \
  output_dir=./outputs/${EXPERIMENT} \
  wandb.enabled=false

echo "=== END: best ckpt in ./outputs/${EXPERIMENT}/checkpoints/ ==="
date
