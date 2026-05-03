#!/bin/bash
#SBATCH --partition=mitarb
#SBATCH --account=mitarb
#SBATCH --gres=gpu:mitarb:1
#SBATCH --mem=48G
#SBATCH --time=22:00:00
#SBATCH --job-name=joint_faiss
#SBATCH --output=/scratch/bhushkri/hybrid_xmamba_a100_70m_40/logs/%x_%j.log
#SBATCH --error=/scratch/bhushkri/hybrid_xmamba_a100_70m_40/logs/%x_%j.log
#SBATCH --requeue

# Phase 7 — FAISS hard-negative mining + continued joint training.
#
# v1+v2 diagnosis:
#   - Doubling in-batch negatives (15→31) only bought +0.4pp on MIMIC val R@10
#   - Paired cosine 0.29 (MIMIC) / 0.21 (Indiana) — barely aligned
#   - Random in-batch negatives are semantically too easy (dissimilar reports)
#
# Phase 7 fixes:
#   - Mine top-50 hardest negatives per anchor using v2 text encoder
#   - Inject K=4 hard negs per anchor into InfoNCE text bank (31+4=35 negatives)
#   - Resume training from v2 best ckpt (step=001637, loss=2.4715)
#   - max_steps=5000, same bs=32×accum=4 (effective batch 128)
#
# Submit from parent dir:
#   cd /scratch/bhushkri/hybrid_xmamba_a100_70m_40
#   sbatch hybrid_model_mamba_xlstm/scripts/train_joint_mimic_faiss.sh
#
# Overrides:
#   V2_CKPT=/path/to/ckpt sbatch ...
#   HARD_NEG_FILE=/path/to/index.pt sbatch ...  (skip mining if already exists)

set -euo pipefail

V2_CKPT="${V2_CKPT:-./outputs/joint_mimic_cxr_v2/checkpoints/contrastive-step=001637-val/total_loss=2.4715.ckpt}"
MIMIC_CACHE_DIR="${MIMIC_CACHE_DIR:-/scratch/bhushkri/mimic_cxr_cache}"
HARD_NEG_FILE="${HARD_NEG_FILE:-./outputs/mimic_hard_neg_index.pt}"

echo "=== JOB START (Phase 7: FAISS hard-neg mining + joint training) ==="
date
echo "Host: $(hostname)"
echo "Submit dir: ${SLURM_SUBMIT_DIR}"
echo "v2 checkpoint : ${V2_CKPT}"
echo "Hard neg file : ${HARD_NEG_FILE}"
echo ""

mkdir -p /scratch/bhushkri/hybrid_xmamba_a100_70m_40/logs
cd "${SLURM_SUBMIT_DIR}/hybrid_model_mamba_xlstm"

export HF_HOME="$PWD/.hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCHINDUCTOR_CACHE_DIR="$PWD/.torchinductor"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"
export CUDA_LAUNCH_BLOCKING=0

if [ ! -d ".venv" ]; then
    echo "ERROR: .venv not found"
    exit 1
fi
source .venv/bin/activate

python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB')
"

nvidia-smi

if [ ! -f "${V2_CKPT}" ]; then
    echo "ERROR: v2 checkpoint not found: ${V2_CKPT}"
    exit 1
fi

# -----------------------------------------------------------------------
# Step 1 — Mine hard negatives (skip if index already exists)
# -----------------------------------------------------------------------
if [ -f "${HARD_NEG_FILE}" ]; then
    echo "Hard neg index already exists: ${HARD_NEG_FILE} — skipping mining."
else
    echo ""
    echo "=== Step 1: Mining hard negatives (~20 min) ==="
    python scripts/mine_hard_negatives.py \
        --checkpoint "${V2_CKPT}" \
        --cache-dir "${MIMIC_CACHE_DIR}" \
        --output-file "${HARD_NEG_FILE}" \
        --k 50 \
        --batch-size 128 \
        --max-length 256
    echo "=== Mining complete ==="
fi

# -----------------------------------------------------------------------
# Step 2 — Joint training with hard negatives (resume from v2 ckpt)
# -----------------------------------------------------------------------
echo ""
echo "=== Step 2: Joint training with hard negatives ==="
# Paths with '=' in filenames must be single-quoted for Hydra's override grammar.
# resume_from_checkpoint restores full model + optimizer state; lm_checkpoint
# is NOT set here because it only loads the backbone (loses img_proj/distill_proj).
# max_steps=7000 = 1637 (already trained) + ~5363 new steps.

python scripts/train_contrastive.py \
  --config-name config_70m \
  model=hybrid_70m \
  dataset=mimic_cxr \
  +distill=joint_mimic \
  trainer=a100_single_gpu \
  contrastive_mode=joint \
  trainer.max_steps=4500 \
  trainer.accumulate_grad_batches=4 \
  trainer.val_check_interval=250 \
  trainer.log_every_n_steps=25 \
  dataset.batch_size=32 \
  dataset.eval_batch_size=32 \
  dataset.num_workers=4 \
  dataset.pin_memory=true \
  "dataset.cache_dir=${MIMIC_CACHE_DIR}" \
  "dataset.hard_neg_file=${HARD_NEG_FILE}" \
  dataset.hard_neg_k=4 \
  distill.alpha_kd=0.1 \
  model.use_gradient_checkpointing=true \
  "+resume_from_checkpoint='${V2_CKPT}'" \
  experiment_name=joint_mimic_cxr_faiss \
  output_dir=./outputs/joint_mimic_cxr_faiss \
  wandb.enabled=false

echo ""
echo "=== JOB END (Phase 7 complete) ==="
echo "Checkpoint saved to: ./outputs/joint_mimic_cxr_faiss/checkpoints/"
date
