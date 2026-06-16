#!/bin/bash
#SBATCH --partition=mitarb
#SBATCH --account=mitarb
#SBATCH --gres=gpu:mitarb:1
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --job-name=biomedclip_kd_phase15_v2
#SBATCH --output=/scratch/bhushkri/hybrid_xmamba_a100_70m_40/logs/%x_%j.log
#SBATCH --error=/scratch/bhushkri/hybrid_xmamba_a100_70m_40/logs/%x_%j.log
#SBATCH --requeue

# Phase 11 — joint contrastive on the NEW v2 backbone (HYBRID_ARCH_REFACTOR_PLAN).
#
# Inits from the Phase 9-EXT Stage 0 v2 checkpoint (40K steps, PPL ~16.6,
# norm_topology=hybrid, layer pattern [m,m,m,L,L,m,m,m]). Holds the Phase 6e
# contrastive recipe constant (K=0, KD warmup 1000, α_kd 1.0→0.3) so any
# retrieval uplift is attributable to the v2 backbone + the three v2-distill
# deltas (freq-decoupled KD, ViT-unfreeze=2), not a different recipe.
#
# Differences vs train_biomedclip_kd_phase6e.sh:
#   - model=hybrid_70m_v2  (was hybrid_70m)
#   - +distill=biomedclip_kd_joint_v2  (freq_kd=true, vit_unfreeze_blocks=2,
#     vit_lr=1e-6, moco_queue_size=0)
#   - STAGE0_CHECKPOINT defaults to the ext (40K) stripped model-only .pt
#
# PREREQ: extract the 40K checkpoint first (login node):
#   cd /scratch/bhushkri/hybrid_xmamba_a100_70m_40/hybrid_model_mamba_xlstm
#   python -c "import torch,os; d='./outputs/phase9_stage0_arch_v2_ext/checkpoints'; \
#     ck=torch.load(os.path.join(d,'last.ckpt'),map_location='cpu',weights_only=False); \
#     st={k[6:]:v for k,v in ck['state_dict'].items() if k.startswith('model.')}; \
#     torch.save({'state_dict':st}, os.path.join(d,'stage0_v2_model_only.pt')); print('keys',len(st))"
#
# Kill gates (Phase 11B):
#   - cos_text_teacher >= 0.85 by step 1000 (else KD not aligning).
#   - val/clip_loss at step 1000 < 3.0.
#   - MIMIC R@10 by step 3000 >= 0.0823 (Phase 6e floor; no regression).
#   - optimizer.param_groups[-1]['lr'] == 1e-6 ± 1e-9 every 100 steps (ViT group LR guard).
#
# Submit:
#   cd /scratch/bhushkri/hybrid_xmamba_a100_70m_40
#   sbatch hybrid_model_mamba_xlstm/scripts/train_biomedclip_kd_phase15.sh

set -euo pipefail

STAGE0_CHECKPOINT="${STAGE0_CHECKPOINT:-./outputs/phase9_stage0_arch_v2_ext/checkpoints/stage0_v2_model_only.pt}"
MIMIC_CACHE_DIR="${MIMIC_CACHE_DIR:-/scratch/bhushkri/mimic_cxr_cache}"
SKIP_VERIFY="${SKIP_VERIFY:-1}"

echo "=== JOB START (BiomedCLIP-KD Phase 15 / refactor Phase 11: v2 backbone) ==="
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
    echo "Run the 40K extraction first (see header)."
    exit 1
fi
echo "Stage 0 checkpoint verified: ${STAGE0_CHECKPOINT}"

mkdir -p "${MIMIC_CACHE_DIR}"
if [ "${SKIP_VERIFY}" = "1" ]; then
    echo "SKIP_VERIFY=1 → MIMIC cache already warm, skipping pre-flight."
else
    echo "=== Pre-flight: MIMIC-CXR verify + precache ==="
    python scripts/verify_mimic_cxr.py --cache-dir "${MIMIC_CACHE_DIR}" --split train --precache
    echo "=== Pre-flight complete ==="
fi

echo ""
echo "Starting Phase 11 joint contrastive (v2 backbone + biomedclip_kd_joint_v2)..."

python scripts/train_contrastive.py \
  --config-name config_70m \
  model=hybrid_70m_v2 \
  dataset=mimic_cxr \
  +distill=biomedclip_kd_joint_v2 \
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
  experiment_name=biomedclip_kd_phase15_v2 \
  output_dir=./outputs/biomedclip_kd_phase15_v2 \
  wandb.enabled=false

echo ""
echo "=== JOB END (BiomedCLIP-KD Phase 15 / refactor Phase 11 complete) ==="
echo "Best checkpoint: ./outputs/biomedclip_kd_phase15_v2/checkpoints/"
date
