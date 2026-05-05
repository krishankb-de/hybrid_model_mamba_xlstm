#!/bin/bash
#SBATCH --partition=mitarb
#SBATCH --account=mitarb
#SBATCH --gres=gpu:mitarb:1
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --job-name=biomedclip_kd_phase6
#SBATCH --output=/scratch/bhushkri/hybrid_xmamba_a100_70m_40/logs/%x_%j.log
#SBATCH --error=/scratch/bhushkri/hybrid_xmamba_a100_70m_40/logs/%x_%j.log
#SBATCH --requeue

# Phase 6 — Fix: bypass img_proj + higher KD weight
#
# Root cause diagnosis (Phase 5c eval):
#   Indiana i2t R@10 = 3.36%, MIMIC-val = 8.36%, paired cosine = 0.22-0.29
#   Identical to PubMedBERT-era despite BiomedCLIP text KD pivot.
#
# Why: clip_model.visual already includes BiomedCLIP's image projection
#   → output is 512-d in BiomedCLIP's joint space.
#   img_proj (random-init 512→GELU→512) was being applied ON TOP,
#   distorting the correct joint-space embeddings. The CLIP loss (β=1.0)
#   then pulled Mamba text toward the distorted space instead of BiomedCLIP
#   joint space, neutralising the KD signal.
#
# Fix:
#   1. _joint_step: z_img = F.normalize(z_img_raw.float()) — skip img_proj
#   2. alpha_kd: 0.3 → 1.0 — stronger KD signal now that both sides are clean
#
# Submit:
#   cd /scratch/bhushkri/hybrid_xmamba_a100_70m_40
#   sbatch hybrid_model_mamba_xlstm/scripts/train_biomedclip_kd_phase6.sh

set -euo pipefail

STAGE0_CHECKPOINT="${STAGE0_CHECKPOINT:-./outputs/hybrid_70m_stage0_kd_pubmed/checkpoints/stage0_model_only.pt}"
MIMIC_CACHE_DIR="${MIMIC_CACHE_DIR:-/scratch/bhushkri/mimic_cxr_cache}"
SKIP_VERIFY="${SKIP_VERIFY:-1}"

echo "=== JOB START (BiomedCLIP-KD Phase 6: no img_proj + alpha_kd=1.0) ==="
date
echo "Host: $(hostname)"
echo "Submit dir: ${SLURM_SUBMIT_DIR}"
echo "Stage 0 checkpoint: ${STAGE0_CHECKPOINT}"
echo ""

mkdir -p /scratch/bhushkri/hybrid_xmamba_a100_70m_40/logs
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

python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device name: {torch.cuda.get_device_name(0)}')
    mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f'Total VRAM: {mem:.1f} GB')
"

nvidia-smi

if [ ! -f "${STAGE0_CHECKPOINT}" ]; then
    echo "ERROR: Stage 0 checkpoint not found: ${STAGE0_CHECKPOINT}"
    exit 1
fi
echo "Stage 0 checkpoint verified: ${STAGE0_CHECKPOINT}"

mkdir -p "${MIMIC_CACHE_DIR}"
if [ "${SKIP_VERIFY}" = "1" ]; then
    echo "SKIP_VERIFY=1 → MIMIC cache already warm, skipping pre-flight."
else
    echo "=== Pre-flight: MIMIC-CXR verify + precache ==="
    python scripts/verify_mimic_cxr.py \
        --cache-dir "${MIMIC_CACHE_DIR}" \
        --split train \
        --precache
    echo "=== Pre-flight complete ==="
fi

echo ""
echo "Starting BiomedCLIP-KD Phase 6 (no img_proj, alpha_kd=1.0)..."

python scripts/train_contrastive.py \
  --config-name config_70m \
  model=hybrid_70m \
  dataset=mimic_cxr \
  +distill=biomedclip_kd_joint \
  trainer=a100_single_gpu \
  contrastive_mode=joint \
  trainer.max_steps=5000 \
  trainer.accumulate_grad_batches=4 \
  trainer.val_check_interval=250 \
  trainer.log_every_n_steps=25 \
  dataset.batch_size=32 \
  dataset.eval_batch_size=32 \
  dataset.num_workers=4 \
  dataset.pin_memory=true \
  dataset.cache_dir="${MIMIC_CACHE_DIR}" \
  model.use_gradient_checkpointing=true \
  lm_checkpoint="${STAGE0_CHECKPOINT}" \
  experiment_name=biomedclip_kd_phase6 \
  output_dir=./outputs/biomedclip_kd_phase6 \
  wandb.enabled=false

echo ""
echo "=== JOB END (BiomedCLIP-KD Phase 6 complete) ==="
echo "Best checkpoint: ./outputs/biomedclip_kd_phase6/checkpoints/"
date
