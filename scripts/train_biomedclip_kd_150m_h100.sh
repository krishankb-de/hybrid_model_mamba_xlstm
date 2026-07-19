#!/bin/bash
# ============================================================================
# Phase 6 — Joint contrastive on the 150M v2 backbone (canonical recipe + the
# H100 batch-scaling lever). Thin wrapper over train_biomedclip_kd_h100.sh with
# 150M defaults. Inits from the Phase-5 150M Stage-0 checkpoint.
#
# THE MIMIC LEVER: BATCH_SIZE=128 (true in-batch negatives; A100 was capped at 32).
# Phase 6 sweeps {64,128,256} via BATCH_SIZE=... at submit time. NOTE: the generic
# template bakes bs=128 LRs (backbone 2e-5 / head 6e-4, √-scaled from 32). For a
# different batch, also √-rescale: add distill.backbone_lr=/distill.head_lr= overrides.
#
# PREREQ: extract the 150M Stage-0 model-only checkpoint first (strip 'model.' prefix):
#   python -c "import torch; d='./outputs/h100_stage0_150m_v2/checkpoints'; \
#     ck=torch.load(d+'/last.ckpt',map_location='cpu',weights_only=False); \
#     st={k[6:]:v for k,v in ck['state_dict'].items() if k.startswith('model.')}; \
#     torch.save({'state_dict':st}, d+'/stage0_model_only.pt'); print('keys',len(st))"
#
# Submit sweep e.g.:  BATCH_SIZE=256 sbatch scripts/train_biomedclip_kd_150m_h100.sh
# ============================================================================
#SBATCH --partition=aisc-batch
#SBATCH --account=aisc
#SBATCH --gpus=1
#SBATCH --exclude=ga03,gx17v1,gx13v1   # gx13v1: faulty GPU (cudaErrorContained, 2026-07-19)
#SBATCH --mem=128G
#SBATCH --cpus-per-task=12
#SBATCH --time=1-00:00:00
#SBATCH --job-name=h100_kd_150m
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log
#SBATCH --requeue

set -euo pipefail

export MODEL_CONFIG="hybrid_150m_v2"
export BATCH_SIZE="${BATCH_SIZE:-128}"
# 2026-07-19: bs=128 OOMs on the 80GB card the moment freeze_text_encoder_steps=1000
# expires and the 184M backbone unfreezes (scan intermediate B*L*(D*expand)*N =
# 128*256*1536*16 ~= 3.2GB EACH in fp32 — the fp32 scan that fixed Stage-0 costs 2x
# memory here). Gradient checkpointing is the same lever that fixed Stage-0; keep
# bs=128 so the in-batch-negative count (the MIMIC lever) is preserved.
# If it still OOMs: BATCH_SIZE=64 (halves negatives — last resort).
export GRAD_CKPT="${GRAD_CKPT:-true}"
export STAGE0_CKPT="${STAGE0_CKPT:-./outputs/h100_stage0_150m_v2/checkpoints/stage0_model_only.pt}"
export EXPERIMENT="${EXPERIMENT:-h100_kd_150m_v2_bs${BATCH_SIZE}}"

cd "${SLURM_SUBMIT_DIR:-.}/hybrid_model_mamba_xlstm" 2>/dev/null || cd "${SLURM_SUBMIT_DIR:-.}"
echo "[wrapper] 150M contrastive → delegating to train_biomedclip_kd_h100.sh (bs=${BATCH_SIZE}, ckpt=${STAGE0_CKPT})"
bash scripts/train_biomedclip_kd_h100.sh
