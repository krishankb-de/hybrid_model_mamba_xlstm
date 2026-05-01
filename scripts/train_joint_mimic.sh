#!/bin/bash
#SBATCH --partition=mitarb
#SBATCH --account=mitarb
#SBATCH --gres=gpu:mitarb:1
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --job-name=joint_mimic_cxr
#SBATCH --output=/scratch/bhushkri/hybrid_xmamba_a100_70m_40/logs/%x_%j.log
#SBATCH --error=/scratch/bhushkri/hybrid_xmamba_a100_70m_40/logs/%x_%j.log
#SBATCH --requeue

# Joint multi-task training on MIMIC-CXR.
# Loss = α·KD(PubMedBERT) + β·CLIP(BiomedCLIP) + γ·SimCSE
# Inits from Stage 0 LM checkpoint (PPL=13.10).
#
# Effective batch = 16 × accum8 = 128.
# Live kill gate: scancel this job if R@10 < 0.15 at step 3000 and not rising.
#
# Override checkpoint path:
#   STAGE0_CHECKPOINT=/path/to/ckpt sbatch scripts/train_joint_mimic.sh

set -euo pipefail

STAGE0_CHECKPOINT="${STAGE0_CHECKPOINT:-./outputs/hybrid_70m_stage0_kd_pubmed/checkpoints/stage0_model_only.pt}"

echo "=== JOB START (Joint MIMIC-CXR: KD+CLIP+SimCSE) ==="
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

echo ""
echo "Verifying CUDA availability:"
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device name: {torch.cuda.get_device_name(0)}')
    mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f'Total VRAM: {mem:.1f} GB')
"
echo ""

nvidia-smi

# Verify Stage 0 checkpoint before starting
if [ ! -f "${STAGE0_CHECKPOINT}" ]; then
    echo "ERROR: Stage 0 checkpoint not found at: ${STAGE0_CHECKPOINT}"
    echo "Run eval_stage0_lm.sh to confirm the checkpoint, or set STAGE0_CHECKPOINT."
    exit 1
fi
echo "Stage 0 checkpoint verified: ${STAGE0_CHECKPOINT}"

echo ""
echo "Starting joint MIMIC-CXR training..."

python scripts/train_contrastive.py \
  --config-name config_70m \
  model=hybrid_70m \
  dataset=mimic_cxr \
  distill=joint_mimic \
  trainer=a100_single_gpu \
  contrastive_mode=joint \
  trainer.max_steps=10000 \
  trainer.accumulate_grad_batches=8 \
  trainer.val_check_interval=500 \
  trainer.log_every_n_steps=25 \
  dataset.batch_size=16 \
  dataset.eval_batch_size=16 \
  dataset.num_workers=4 \
  dataset.pin_memory=true \
  dataset.cache_dir=/scratch/bhushkri/mimic_cxr_cache \
  model.use_gradient_checkpointing=true \
  lm_checkpoint="${STAGE0_CHECKPOINT}" \
  experiment_name=joint_mimic_cxr \
  output_dir=./outputs/joint_mimic_cxr \
  wandb.enabled=false

echo ""
echo "=== JOB END (Joint MIMIC-CXR complete) ==="
echo "Checkpoint saved to: ./outputs/joint_mimic_cxr/checkpoints/"
date
