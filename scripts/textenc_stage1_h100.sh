#!/bin/bash
# ============================================================================
# TEXTENC_PLAN.md T1-D / T2 -- Stage 1 with the attention-free mLMamba text encoder.
#
# Recipe = the published 13D tower run (outputs/h100_kd_150m_v2_full_data_lr3e6, argv in its
# run_metadata.json, 2026-09-02) with ONE change: the text encoder. Published wrappers
# (train_biomedclip_kd_150m_h100.sh -> train_biomedclip_kd_h100.sh) are not touched.
#
# Isolation (rule R1): everything this job writes goes under TEXTENC_ROOT (the cluster worktree
# ~/hybrid_textenc, whose outputs/ results/ logs/ are real directories). The Stage-0 checkpoint
# is only read. The script refuses to start without TEXTENC_ROOT or when it points at the
# published checkouts.
#
# TEXT_ENCODER selects the arm:
#   m3          (default) hybrid_150m_m3_textenc, Stage-0 mLMamba init, mean pooling   -> E-m3
#   transformer ABL-T: matched Transformer, Stage-0 Transformer init, mean pooling (attention)
#
# Usage (from ~/hybrid_textenc):
#   TEXTENC_ROOT=$HOME/hybrid_textenc sbatch scripts/textenc_stage1_h100.sh
#   TEXTENC_ROOT=$HOME/hybrid_textenc TEXT_ENCODER=transformer sbatch scripts/textenc_stage1_h100.sh
# ============================================================================
#SBATCH --partition=pot-hpi-aisc-batch
#SBATCH --account=aisc
#SBATCH --qos=aisc
#SBATCH --gpus=1
#SBATCH --exclude=ga03,gx17v1,gx13v1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=12
#SBATCH --time=10:00:00
#SBATCH --job-name=textenc_stage1
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log

set -euo pipefail

# --- R1 guard: before anything else -------------------------------------------------------
if [ -z "${TEXTENC_ROOT:-}" ]; then
  echo "ERROR: TEXTENC_ROOT is not set (TEXTENC_PLAN.md R1: this plan writes only under its own root)"
  exit 2
fi
ROOT_REAL="$(cd "${TEXTENC_ROOT}" 2>/dev/null && pwd -P || echo "${TEXTENC_ROOT}")"
case "${ROOT_REAL}" in
  */hybrid_mamba_xlstm|*/hybrid_mamba_xlstm/*|*/hybrid_isbi|*/hybrid_isbi/*|*/hybrid_model_mamba_xlstm/hybrid_model_mamba_xlstm*)
    echo "ERROR: TEXTENC_ROOT=${ROOT_REAL} is a published checkout; refusing to write there (R1)"
    exit 2 ;;
esac
for d in outputs logs cache; do
  if [ -L "${ROOT_REAL}/${d}" ]; then
    echo "ERROR: ${ROOT_REAL}/${d} is a symlink; R1 needs a real directory"
    exit 2
  fi
done

TEXT_ENCODER="${TEXT_ENCODER:-m3}"
PUBLISHED_OUTPUTS="${PUBLISHED_OUTPUTS:-/sc/home/${USER}/hybrid_mamba_xlstm/outputs}"
case "${TEXT_ENCODER}" in
  m3)
    MODEL_CONFIG=hybrid_150m_m3_textenc
    STAGE0_CKPT="${STAGE0_CKPT:-${PUBLISHED_OUTPUTS}/h100_stage0_150m_m3/checkpoints/last.ckpt}"
    POOLING_OVERRIDE=() ;;
  transformer)
    MODEL_CONFIG=transformer_150m_baseline
    STAGE0_CKPT="${STAGE0_CKPT:-${PUBLISHED_OUTPUTS}/h100_stage0_transformer_150m/checkpoints/last.ckpt}"
    POOLING_OVERRIDE=("++model.pooling_strategy=mean") ;;
  *) echo "ERROR: unknown TEXT_ENCODER=${TEXT_ENCODER}"; exit 2 ;;
esac

# --- the published 13D tower recipe (run_metadata.json argv), unchanged ---------------------
BATCH_SIZE=64
MAX_STEPS="${MAX_STEPS:-6000}"
BACKBONE_LR=1.41e-5
HEAD_LR=4.24e-4
VIT_UNFREEZE=12
VIT_LR=3e-6
VIT_SCOPE=blocks
GAMMA_SIMCSE=0.1
KD_DECAY_STEPS=0
ALPHA_KD_FLOOR=0.0
CLIP_LOSS=infonce
MULTIPOS=false
BIDIRECTIONAL="${BIDIRECTIONAL:-false}"
DATASET_CONFIG=cxr_mimic_full
EXPERIMENT="${EXPERIMENT:-textenc_stage1_${TEXT_ENCODER}}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/sc/scratch/${USER}/hybrid_xmamba_h100}"
MIMIC_CACHE_DIR="${ROOT_REAL}/cache/mimic_cxr_cache"

cd "${ROOT_REAL}"
mkdir -p logs outputs "${MIMIC_CACHE_DIR}"
echo "=== TEXTENC Stage 1: text encoder=${TEXT_ENCODER} (${MODEL_CONFIG}), root=${ROOT_REAL} ==="
echo "=== recipe: bs=${BATCH_SIZE} steps=${MAX_STEPS} lr=${BACKBONE_LR}/${HEAD_LR} vit=${VIT_UNFREEZE}@${VIT_LR} simcse=${GAMMA_SIMCSE} bidirectional=${BIDIRECTIONAL} ==="
date; hostname

if [ ! -f "${STAGE0_CKPT}" ]; then
  echo "ERROR: Stage-0 checkpoint not found: ${STAGE0_CKPT}"
  exit 1
fi
echo "Stage-0 checkpoint (read only): ${STAGE0_CKPT}"

export HF_HOME="${SCRATCH_ROOT}/.hf"
export HF_DATASETS_CACHE="${ROOT_REAL}/cache/hf_datasets"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TORCHINDUCTOR_CACHE_DIR="${ROOT_REAL}/cache/torchinductor"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONUNBUFFERED=1
source "${VENV_ACTIVATE:-.venv/bin/activate}"
python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print('GPU:', torch.cuda.get_device_name(0))"

python scripts/train_contrastive.py \
  --config-name config_70m \
  model=${MODEL_CONFIG} \
  dataset=${DATASET_CONFIG} \
  +distill=biomedclip_kd_joint_v2 \
  distill.freq_kd=false \
  distill.vit_unfreeze_blocks=${VIT_UNFREEZE} \
  distill.vit_lr=${VIT_LR} \
  distill.vit_unfreeze_scope=${VIT_SCOPE} \
  distill.backbone_lr=${BACKBONE_LR} \
  distill.head_lr=${HEAD_LR} \
  distill.gamma_simcse=${GAMMA_SIMCSE} \
  distill.kd_decay_steps=${KD_DECAY_STEPS} \
  distill.alpha_kd_floor=${ALPHA_KD_FLOOR} \
  distill.clip_loss_type=${CLIP_LOSS} \
  distill.use_multipos=${MULTIPOS} \
  ++model.bidirectional_encode=${BIDIRECTIONAL} \
  ${POOLING_OVERRIDE[@]+"${POOLING_OVERRIDE[@]}"} \
  trainer=h100_single_gpu \
  contrastive_mode=joint \
  trainer.max_steps=${MAX_STEPS} \
  trainer.accumulate_grad_batches=1 \
  trainer.val_check_interval=250 \
  trainer.log_every_n_steps=25 \
  trainer.compile_model=false \
  dataset.batch_size=${BATCH_SIZE} \
  dataset.eval_batch_size=${BATCH_SIZE} \
  dataset.num_workers=8 \
  dataset.pin_memory=true \
  dataset.cache_dir="${MIMIC_CACHE_DIR}" \
  model.use_gradient_checkpointing=true \
  lm_checkpoint="${STAGE0_CKPT}" \
  experiment_name=${EXPERIMENT} \
  output_dir="${ROOT_REAL}/outputs/${EXPERIMENT}" \
  wandb.enabled=false
echo "=== END: ${ROOT_REAL}/outputs/${EXPERIMENT}/checkpoints/ ==="
date
