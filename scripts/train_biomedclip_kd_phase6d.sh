#!/bin/bash
#SBATCH --partition=mitarb
#SBATCH --account=mitarb
#SBATCH --gres=gpu:mitarb:1
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --job-name=biomedclip_kd_phase6d
#SBATCH --output=/scratch/bhushkri/hybrid_xmamba_a100_70m_40/logs/%x_%j.log
#SBATCH --error=/scratch/bhushkri/hybrid_xmamba_a100_70m_40/logs/%x_%j.log
#SBATCH --requeue

# Phase 6d — combined recovery run (Phases 8–10 fixes).
#
# Phase 6c (job 1300) cancelled: val/clip_loss 2.97→3.45, i2t R@10=0.49% (random).
# Re-diagnosis identified three vulnerabilities still present after 6a/6b/6c:
#   (1) CLIP loss ran from step 0 — polluted proj_head before KD stabilised.
#   (2) MoCo text_queue enqueued from step 0 — stale GPT-2-space negatives
#       produced random InfoNCE gradients post-unfreeze.
#   (3) distill_proj / img_proj remained in optimizer as dead modules.
#
# Phase 6d combines four fixes:
#   • Phase 8  — distill_proj + img_proj DELETED (architecture + optimizer + ckpts).
#   • Phase 9  — CLIP loss gated by global_step >= freeze_text_encoder_steps;
#                MoCoQueue.reset() + MomentumEncoder.copy_from() at unfreeze step;
#                training_step gates momentum_encoder.update during warmup.
#   • Phase 10 — freeze_text_encoder_steps 500 → 1000;
#                α_kd_warmup=1.0 (KD owns gated warmup);
#                α_kd_post=0.3 (CLIP coexistence weight, no 6a divergence);
#                cos_text_teacher + effective_alpha_kd diagnostic logs.
#   • Ckpt back-compat — strict=False strips Phase ≤6c img_proj.* / distill_proj.*.
#
# Kill-job rules (monitor live):
#   • train/cos_text_teacher < 0.5 by step 800  → KILL (α_kd_warmup or proj_head LR wrong).
#   • val/clip_loss first reading at step 1000 ≥ 2.47 (Phase 5c floor) → INVESTIGATE.
#   • After 5 epochs i2t R@10 < 12%  → continue but expect MARGINAL.
#
# Submit:
#   cd /scratch/bhushkri/hybrid_xmamba_a100_70m_40
#   sbatch hybrid_model_mamba_xlstm/scripts/train_biomedclip_kd_phase6d.sh

set -euo pipefail

STAGE0_CHECKPOINT="${STAGE0_CHECKPOINT:-./outputs/hybrid_70m_stage0_kd_pubmed/checkpoints/stage0_model_only.pt}"
MIMIC_CACHE_DIR="${MIMIC_CACHE_DIR:-/scratch/bhushkri/mimic_cxr_cache}"
SKIP_VERIFY="${SKIP_VERIFY:-1}"

echo "=== JOB START (BiomedCLIP-KD Phase 6d: Phases 8–10 combined recovery) ==="
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
echo "Starting BiomedCLIP-KD Phase 6d (gated CLIP + α_kd schedule + cold-start queue)..."

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
  experiment_name=biomedclip_kd_phase6d \
  output_dir=./outputs/biomedclip_kd_phase6d \
  wandb.enabled=false

echo ""
echo "=== JOB END (BiomedCLIP-KD Phase 6d complete) ==="
echo "Best checkpoint: ./outputs/biomedclip_kd_phase6d/checkpoints/"
date
