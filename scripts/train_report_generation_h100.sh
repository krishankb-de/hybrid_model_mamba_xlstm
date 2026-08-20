#!/bin/bash
# ============================================================================
# Phase 10E (H100_SCALING_PLAN.md) — image-conditioned report generation.
# SLURM wrapper for scripts/train_report_generation.py, mirroring
# train_biomedclip_kd_h100.sh's env-lever / --exclude / offline-HF conventions
# so this is a drop-in sibling, not a new pattern to learn.
#
# STILL BLOCKED as of 2026-08-20: needs (1) the Phase 8 local MIMIC-CXR-JPG
# build to finish (job 2461245) and (2) DECODER_CKPT — the Phase-5/Stage-0
# 150M checkpoint (10D) to init the decoder from. Do NOT sbatch this until
# both exist; it will fail fast on the DECODER_CKPT existence check below,
# which is deliberate (same pattern as STAGE0_CKPT in train_biomedclip_kd_h100.sh).
#
# PREFIX_K is the depth-analogue lever Phase 10B calls out to sweep {8,32,64}.
# ============================================================================
#SBATCH --partition=aisc-batch
#SBATCH --account=aisc
#SBATCH --gpus=1
#SBATCH --exclude=ga03,gx17v1,gx13v1   # ga03: ARM node, x86 .venv incompatible;
                                        # gx13v1: faulty GPU (cudaErrorContained, 2026-07-19)
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --time=16:00:00
#SBATCH --job-name=h100_report_gen
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log
#SBATCH --requeue

set -euo pipefail

SCRATCH_ROOT="${SCRATCH_ROOT:-/sc/scratch/$USER/hybrid_xmamba_h100}"
VENV_ACTIVATE="${VENV_ACTIVATE:-.venv/bin/activate}"

MODEL_CONFIG="${MODEL_CONFIG:-hybrid_150m_v2_rrg}"
DATASET_CONFIG="${DATASET_CONFIG:-cxr_mimic_full}"

BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_STEPS="${MAX_STEPS:-10000}"
DECODER_LR="${DECODER_LR:-1e-5}"
HEAD_LR="${HEAD_LR:-3e-4}"
# 150M is spike-fragile (H100_SCALING_PLAN.md notes); keep 0.5, not 1.0.
GRAD_CLIP="${GRAD_CLIP:-0.5}"

# Phase 10B depth-analogue lever — sweep 8/32/64 across separate runs (one
# model per k, not a single model handling variable k at inference).
PREFIX_K="${PREFIX_K:-32}"

VIT_UNFREEZE="${VIT_UNFREEZE:-0}"   # 0 = frozen image tower (10C default until
                                     # a Phase-9 best contrastive checkpoint exists)
VIT_LR="${VIT_LR:-1e-6}"
GRAD_CKPT="${GRAD_CKPT:-false}"

# Phase 10D — decoder init checkpoint. No safe default; must be supplied.
DECODER_CKPT="${DECODER_CKPT:-./outputs/h100_stage0_150m_v2/checkpoints/stage0_model_only.pt}"

MIMIC_CACHE_DIR="${MIMIC_CACHE_DIR:-/sc/home/$USER/dataset/mimic_cxr_cache}"
EXPERIMENT="${EXPERIMENT:-h100_report_gen_150m_v2_k${PREFIX_K}}"

echo "=== Phase 10E report generation: ${MODEL_CONFIG} on ${DATASET_CONFIG}, prefix_k=${PREFIX_K} ==="
echo "=== LRs: decoder=${DECODER_LR} head=${HEAD_LR} grad_clip=${GRAD_CLIP} | max_steps=${MAX_STEPS} ==="
date; hostname
mkdir -p logs "${MIMIC_CACHE_DIR}"

cd "${SLURM_SUBMIT_DIR}/hybrid_model_mamba_xlstm" 2>/dev/null || cd "${SLURM_SUBMIT_DIR:-.}"

export HF_HOME="${SCRATCH_ROOT}/.hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TORCHINDUCTOR_CACHE_DIR="${SCRATCH_ROOT}/.torchinductor"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONUNBUFFERED=1

source "${VENV_ACTIVATE}"
python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print('GPU:', torch.cuda.get_device_name(0), f'{torch.cuda.get_device_properties(0).total_memory/1024**3:.0f}GB')"
nvidia-smi

if [ ! -f "${DECODER_CKPT}" ]; then
  echo "ERROR: Decoder checkpoint not found: ${DECODER_CKPT}"
  echo "Phase 10D: point DECODER_CKPT at the Stage-0/joint-trained 150M backbone."
  exit 1
fi
echo "Decoder checkpoint: ${DECODER_CKPT}"

echo "Starting report-generation training..."
python scripts/train_report_generation.py \
  --config-name config \
  model=${MODEL_CONFIG} \
  dataset=${DATASET_CONFIG} \
  trainer=h100_single_gpu \
  trainer.max_steps=${MAX_STEPS} \
  trainer.accumulate_grad_batches=1 \
  trainer.val_check_interval=250 \
  trainer.log_every_n_steps=25 \
  dataset.batch_size=${BATCH_SIZE} \
  dataset.eval_batch_size=${BATCH_SIZE} \
  dataset.num_workers=8 \
  dataset.pin_memory=true \
  dataset.cache_dir="${MIMIC_CACHE_DIR}" \
  model.prefix_k=${PREFIX_K} \
  model.decoder_lr=${DECODER_LR} \
  model.head_lr=${HEAD_LR} \
  model.gradient_clip_val=${GRAD_CLIP} \
  model.vit_unfreeze_blocks=${VIT_UNFREEZE} \
  model.vit_lr=${VIT_LR} \
  model.use_gradient_checkpointing=${GRAD_CKPT} \
  decoder_checkpoint="${DECODER_CKPT}" \
  experiment_name=${EXPERIMENT} \
  output_dir=./outputs/${EXPERIMENT} \
  wandb.enabled=false

echo "=== END: best ckpt in ./outputs/${EXPERIMENT}/checkpoints/ ==="
date
