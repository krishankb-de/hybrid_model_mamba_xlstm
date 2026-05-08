#!/bin/bash
#SBATCH --partition=mitarb
#SBATCH --account=mitarb
#SBATCH --gres=gpu:mitarb:1
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --job-name=biomedclip_kd_phase6e
#SBATCH --output=/scratch/bhushkri/hybrid_xmamba_a100_70m_40/logs/%x_%j.log
#SBATCH --error=/scratch/bhushkri/hybrid_xmamba_a100_70m_40/logs/%x_%j.log
#SBATCH --requeue

# Phase 6e — drop MoCo queue; KD warmup + in-batch CLIP only.
#
# Phase 6d (job 1313) failure root cause:
#   MoCo K=16384 queue reset at unfreeze (step 1000) fills with 16384 random
#   unit-norm vectors. K/batch=16384/32=512 steps to fully refresh. For those
#   512 steps, ~99% of the InfoNCE denominator is noise → clip_loss starts at
#   3.52 (Phase 5c floor: 2.47) → near-random CLIP gradients partially destroy
#   the KD-trained alignment before the queue warms. MIMIC R@10=3.95% (below
#   Phase 5c's 9.99%).
#
# Phase 6e fix: moco_queue_size=0 → in-batch negatives only (32).
#   No queue → no cold-start. With cos_teacher=0.74 pre-alignment at unfreeze,
#   the positive pair similarity is real from step 1000. Hypothesis: val/clip_loss
#   at step 1000 will be BELOW 2.47 (Phase 5c floor), and MIMIC R@10 will
#   exceed 9.99% within 5000 steps.
#
# All other Phase 6d settings preserved:
#   - 1000-step KD warmup (freeze_text_encoder_steps=1000)
#   - α_kd_warmup=1.0 / α_kd_post=0.3 schedule
#   - CLIP gate (global_step >= freeze_text_encoder_steps)
#   - distill_proj/img_proj deleted (Phase 8)
#   - cos_text_teacher + effective_alpha_kd logging (Phase 10)
#
# Kill-job rules:
#   - cos_text_teacher < 0.5 by step 800 → KILL (same as 6d).
#   - val/clip_loss at step 1000 >= 2.47 → INVESTIGATE (no improvement over 6d).
#   - After 5000 steps: MIMIC R@10 < 9.99% → in-batch insufficient; restore K=256.
#
# Submit:
#   cd /scratch/bhushkri/hybrid_xmamba_a100_70m_40
#   sbatch hybrid_model_mamba_xlstm/scripts/train_biomedclip_kd_phase6e.sh

set -euo pipefail

STAGE0_CHECKPOINT="${STAGE0_CHECKPOINT:-./outputs/hybrid_70m_stage0_kd_pubmed/checkpoints/stage0_model_only.pt}"
MIMIC_CACHE_DIR="${MIMIC_CACHE_DIR:-/scratch/bhushkri/mimic_cxr_cache}"
SKIP_VERIFY="${SKIP_VERIFY:-1}"

echo "=== JOB START (BiomedCLIP-KD Phase 6e: no MoCo queue, in-batch CLIP + KD warmup) ==="
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
echo "Starting BiomedCLIP-KD Phase 6e (moco_queue_size=0, in-batch CLIP only, 1000-step KD warmup)..."

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
  experiment_name=biomedclip_kd_phase6e \
  output_dir=./outputs/biomedclip_kd_phase6e \
  wandb.enabled=false

echo ""
echo "=== JOB END (BiomedCLIP-KD Phase 6e complete) ==="
echo "Best checkpoint: ./outputs/biomedclip_kd_phase6e/checkpoints/"
date
