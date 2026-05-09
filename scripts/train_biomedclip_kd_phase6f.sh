#!/bin/bash
#SBATCH --partition=mitarb
#SBATCH --account=mitarb
#SBATCH --gres=gpu:mitarb:1
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --job-name=biomedclip_kd_phase6f
#SBATCH --output=/scratch/bhushkri/hybrid_xmamba_a100_70m_40/logs/%x_%j.log
#SBATCH --error=/scratch/bhushkri/hybrid_xmamba_a100_70m_40/logs/%x_%j.log
#SBATCH --requeue

# Phase 6f — small MoCo queue (K=256) + KD warmup.
#
# Experiment history leading here:
#   Phase 5c  (K=16384, warm, no gate):  MIMIC R@10=9.99%, Indiana=3.36%, cos=0.226
#   Phase 6d  (K=16384, cold-start):     MIMIC R@10=3.95%  ← cold-start (512 warm steps)
#   Phase 6e  (K=0, in-batch only):      MIMIC R@10=8.23%, Indiana=4.04%, cos=0.258
#
# Phase 6e analysis: KD warmup (cos_teacher=0.89) improved Indiana (+0.68pp)
# and paired cosine (+0.032) vs Phase 5c, but MIMIC stayed 1.76pp below Phase 5c.
# Root cause: 32 in-batch negatives vs Phase 5c's 16384 warm-queue negatives →
# easier InfoNCE → less discriminative training per step.
#
# Phase 6f fix: K=256 MoCo queue.
#   Cold-start window: K/batch = 256/32 = 8 steps (negligible vs 512 in Phase 6d).
#   Negative pool: 256+32=288 per step (9× harder than Phase 6e's 32).
#   Combined with KD warmup (cos_teacher→0.89), should push MIMIC past 9.99%
#   while maintaining Indiana and paired-cosine gains.
#
# All other Phase 6e settings preserved:
#   - freeze_text_encoder_steps=1000
#   - alpha_kd_warmup=1.0 / alpha_kd_post=0.3
#   - CLIP gate (global_step >= freeze_text_encoder_steps)
#   - Queue reset + momentum encoder hard-resync at unfreeze (Phase 9)
#
# Kill-job rules:
#   - cos_text_teacher < 0.5 by step 800 → KILL.
#   - val/clip_loss at step 1000 >= 2.47 (Phase 5c floor) → INVESTIGATE.
#   - After 5000 steps: MIMIC i2t R@10 < 8.23% (Phase 6e) → K=256 not helping.
#
# Submit:
#   cd /scratch/bhushkri/hybrid_xmamba_a100_70m_40
#   sbatch hybrid_model_mamba_xlstm/scripts/train_biomedclip_kd_phase6f.sh

set -euo pipefail

STAGE0_CHECKPOINT="${STAGE0_CHECKPOINT:-./outputs/hybrid_70m_stage0_kd_pubmed/checkpoints/stage0_model_only.pt}"
MIMIC_CACHE_DIR="${MIMIC_CACHE_DIR:-/scratch/bhushkri/mimic_cxr_cache}"
SKIP_VERIFY="${SKIP_VERIFY:-1}"

echo "=== JOB START (BiomedCLIP-KD Phase 6f: K=256 MoCo queue + KD warmup) ==="
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
echo "Starting BiomedCLIP-KD Phase 6f (K=256 MoCo + 1000-step KD warmup)..."

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
  experiment_name=biomedclip_kd_phase6f \
  output_dir=./outputs/biomedclip_kd_phase6f \
  wandb.enabled=false

echo ""
echo "=== JOB END (BiomedCLIP-KD Phase 6f complete) ==="
echo "Best checkpoint: ./outputs/biomedclip_kd_phase6f/checkpoints/"
date
