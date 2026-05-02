#!/bin/bash
#SBATCH --partition=mitarb
#SBATCH --account=mitarb
#SBATCH --gres=gpu:mitarb:1
#SBATCH --mem=40G
#SBATCH --time=20:00:00
#SBATCH --job-name=joint_mimic_v2
#SBATCH --output=/scratch/bhushkri/hybrid_xmamba_a100_70m_40/logs/%x_%j.log
#SBATCH --error=/scratch/bhushkri/hybrid_xmamba_a100_70m_40/logs/%x_%j.log
#SBATCH --requeue

# Joint multi-task training v2 — MIMIC-CXR, improved hyperparams.
#
# v1 (job 1241) diagnosis:
#   - batch_size=16 → only 15 in-batch negatives per InfoNCE step
#   - R@10 ceiling at 9.4% across all 30 epochs — structural, not fixable by more steps
#   - train/clip collapsed to 0.151 vs val/clip=1.91 (solved 15-neg problem, not 3063-pool)
#   - alpha_kd=0.3 competed with CLIP for backbone gradients
#
# v2 fixes:
#   - batch_size=32 (was 16): 31 in-batch negatives vs 15 → ~2× discriminative capacity
#   - accumulate_grad_batches=4 (was 8): effective batch stays 128
#   - alpha_kd=0.1 (was 0.3): CLIP gets 3× more relative weight vs KD
#   - max_steps=5000 (was 10000): v1 peaked at step 1915; stop earlier
#   - val_check_interval=250 (was 500): finer granularity to catch peak
#
# Submit from parent dir:
#   cd /scratch/bhushkri/hybrid_xmamba_a100_70m_40
#   sbatch hybrid_model_mamba_xlstm/scripts/train_joint_mimic_v2.sh
#
# Overrides:
#   STAGE0_CHECKPOINT=/path/to/ckpt sbatch ...
#   MIMIC_CACHE_DIR=/path/to/cache sbatch ...   (cache already warm → SKIP_VERIFY=1 default)

set -euo pipefail

STAGE0_CHECKPOINT="${STAGE0_CHECKPOINT:-./outputs/hybrid_70m_stage0_kd_pubmed/checkpoints/stage0_model_only.pt}"
MIMIC_CACHE_DIR="${MIMIC_CACHE_DIR:-/scratch/bhushkri/mimic_cxr_cache}"
SKIP_VERIFY="${SKIP_VERIFY:-1}"

echo "=== JOB START (Joint MIMIC-CXR v2: batch=32, alpha_kd=0.1, max_steps=5000) ==="
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
echo "Starting joint MIMIC-CXR training v2..."

python scripts/train_contrastive.py \
  --config-name config_70m \
  model=hybrid_70m \
  dataset=mimic_cxr \
  +distill=joint_mimic \
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
  distill.alpha_kd=0.1 \
  model.use_gradient_checkpointing=true \
  lm_checkpoint="${STAGE0_CHECKPOINT}" \
  experiment_name=joint_mimic_cxr_v2 \
  output_dir=./outputs/joint_mimic_cxr_v2 \
  wandb.enabled=false

echo ""
echo "=== JOB END (Joint MIMIC-CXR v2 complete) ==="
echo "Checkpoint saved to: ./outputs/joint_mimic_cxr_v2/checkpoints/"
date
