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
#
# Phase 13B — multi-GPU DDP lever (NUM_GPUS). The #SBATCH --gpus=1 default
# below is NOT overridden by the NUM_GPUS env var (SLURM parses #SBATCH
# comment directives statically, before this script's shell runs) -- to
# actually get more GPUs allocated you MUST pass matching sbatch CLI flags,
# e.g.: NUM_GPUS=4 sbatch --gpus=4 --gres=gpu:h100:4 scripts/train_report_generation_h100.sh
# NUM_GPUS only controls which trainer= Hydra config this script selects
# (h100_single_gpu vs h100_multi_ddp); it does not request GPUs by itself.
# This is a plain LM cross-entropy loss (no in-batch-negatives semantics),
# so DDP is a clean throughput win here -- unlike the contrastive/CLIP
# trainer, which needs the still-unbuilt Phase 3 all_gather to get anything
# beyond throughput out of extra GPUs (see h100_multi_ddp.yaml's header).
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

# Phase 9D — train-split-only image augmentation (RandomResizedCrop + mild
# rotation). Default false; set true to test whether it fixes the template-
# memorization pattern confirmed live 2026-08-24 (job 2478647's arm0 checkpoint).
AUGMENT="${AUGMENT:-false}"

# Phase 10D — decoder init checkpoint. No safe default; must be supplied.
DECODER_CKPT="${DECODER_CKPT:-./outputs/h100_stage0_150m_v2/checkpoints/stage0_model_only.pt}"

# Phase 13A — optional fine-tuned image-tower checkpoint (a Phase 9
# contrastive .ckpt). Empty (default) keeps the stock BiomedCLIP tower,
# identical behaviour to before this lever existed.
IMAGE_ENCODER_CKPT="${IMAGE_ENCODER_CKPT:-}"

# Phase 13B — multi-GPU DDP lever. See the header comment above: this alone
# does NOT request GPUs from SLURM, it only picks which trainer= config to
# hand to Hydra once GPUs are allocated. Pair with matching --gpus/--gres
# sbatch CLI flags.
#
# EPOCH-BUDGET WARNING (same discipline as train_biomedclip_kd_h100.sh's
# comment on MAX_STEPS): under DDP, effective global batch = BATCH_SIZE x
# NUM_GPUS -- Lightning's trainer.max_steps counts GLOBAL optimizer steps,
# so raising NUM_GPUS with MAX_STEPS unchanged also raises total samples
# seen (confounds a "more GPUs = faster" test with a "bigger effective
# batch" test). To hold the SAME epoch budget as a single-GPU run, divide
# MAX_STEPS by NUM_GPUS; to deliberately target MORE epochs, compute
# MAX_STEPS from (target_epochs * train_pairs) / (BATCH_SIZE * NUM_GPUS).
NUM_GPUS="${NUM_GPUS:-1}"
TRAINER_CFG="h100_single_gpu"
if [ "${NUM_GPUS}" -gt 1 ]; then
  TRAINER_CFG="h100_multi_ddp"
fi

MIMIC_CACHE_DIR="${MIMIC_CACHE_DIR:-/sc/home/$USER/dataset/mimic_cxr_cache}"
EXPERIMENT="${EXPERIMENT:-h100_report_gen_150m_v2_k${PREFIX_K}}"

echo "=== Phase 10E report generation: ${MODEL_CONFIG} on ${DATASET_CONFIG}, prefix_k=${PREFIX_K} ==="
echo "=== LRs: decoder=${DECODER_LR} head=${HEAD_LR} grad_clip=${GRAD_CLIP} | max_steps=${MAX_STEPS} | num_gpus=${NUM_GPUS} (trainer=${TRAINER_CFG}) ==="
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

AVAIL_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
if [ "${AVAIL_GPUS}" -lt "${NUM_GPUS}" ]; then
  echo "ERROR: NUM_GPUS=${NUM_GPUS} requested but only ${AVAIL_GPUS} GPU(s) allocated to this job."
  echo "Pass matching sbatch flags, e.g.: NUM_GPUS=${NUM_GPUS} sbatch --gpus=${NUM_GPUS} --gres=gpu:h100:${NUM_GPUS} scripts/train_report_generation_h100.sh"
  exit 1
fi

if [ ! -f "${DECODER_CKPT}" ]; then
  echo "ERROR: Decoder checkpoint not found: ${DECODER_CKPT}"
  echo "Phase 10D: point DECODER_CKPT at the Stage-0/joint-trained 150M backbone."
  exit 1
fi
echo "Decoder checkpoint: ${DECODER_CKPT}"
if [ -n "${IMAGE_ENCODER_CKPT}" ] && [ ! -f "${IMAGE_ENCODER_CKPT}" ]; then
  echo "ERROR: IMAGE_ENCODER_CKPT set but not found: ${IMAGE_ENCODER_CKPT}"
  exit 1
fi
echo "Image encoder checkpoint: ${IMAGE_ENCODER_CKPT:-<stock BiomedCLIP, unchanged default>}"

EXTRA_ARGS=()
if [ -n "${IMAGE_ENCODER_CKPT}" ]; then
  EXTRA_ARGS+=("image_encoder_checkpoint=${IMAGE_ENCODER_CKPT}")
fi

echo "Starting report-generation training..."
python scripts/train_report_generation.py \
  --config-name config \
  model=${MODEL_CONFIG} \
  dataset=${DATASET_CONFIG} \
  trainer=${TRAINER_CFG} \
  trainer.max_steps=${MAX_STEPS} \
  trainer.accumulate_grad_batches=1 \
  trainer.val_check_interval=250 \
  trainer.log_every_n_steps=25 \
  dataset.batch_size=${BATCH_SIZE} \
  dataset.eval_batch_size=${BATCH_SIZE} \
  dataset.num_workers=8 \
  dataset.pin_memory=true \
  dataset.cache_dir="${MIMIC_CACHE_DIR}" \
  dataset.use_augmentation=${AUGMENT} \
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
  wandb.enabled=false \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

echo "=== END: best ckpt in ./outputs/${EXPERIMENT}/checkpoints/ ==="
date
