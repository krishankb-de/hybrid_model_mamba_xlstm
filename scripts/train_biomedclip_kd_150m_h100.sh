#!/bin/bash
# ============================================================================
# Phase 6 — Joint contrastive on the 150M v2 backbone (canonical recipe + the
# H100 batch-scaling lever). Thin wrapper over train_biomedclip_kd_h100.sh with
# 150M defaults. Inits from the Phase-5 150M Stage-0 checkpoint.
#
# PHASE 6 RESULT (null): the batch sweep {64,128} was statistically flat —
# bs=128/23ep 0.1084, bs=128/14ep 0.1090, bs=64/14ep 0.1113, spread 0.29pp vs
# SE ~0.57pp. The in-batch-negative thesis is NOT supported on a 27570-pair set.
#
# PHASE 6B (this script's current default): the LR probe. The sweep above was
# never LR-matched — see the batch-derived defaults below. Best arm was bs=64,
# so bs=64 is now the default batch. Two arms to compare:
#   sbatch scripts/train_biomedclip_kd_150m_h100.sh                  # √-matched 4.24e-4
#   HEAD_LR=3.0e-4 BACKBONE_LR=1.0e-5 sbatch scripts/train_biomedclip_kd_150m_h100.sh
# The second is the conservative probe: canonical A100 LRs, motivated by
# grad_norm ~12.3 against gradient_clip_val=1.0 (~12x clipping every step).
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
export BATCH_SIZE="${BATCH_SIZE:-64}"
# Phase 8/9: DATASET_CONFIG=cxr_mimic_full switches to the local PhysioNet
# build. Not exported here on purpose — inherited from the caller's env if
# set (e.g. `DATASET_CONFIG=cxr_mimic_full sbatch scripts/train_biomedclip_
# kd_150m_h100.sh`), else train_biomedclip_kd_h100.sh's own default
# (mimic_cxr, the legacy mirror / Arm-0 control) applies.

# --- Batch-derived defaults (2026-07-21) -----------------------------------
# Phase-6 post-mortem found TWO operator traps in the batch sweep:
#   (1) LRs were hardcoded at the bs=128 sqrt-scaled values, so the bs=64 arm
#       (the best result, 0.1113) silently trained at ~1.4x its proper LR. The
#       sweep was never LR-matched.
#   (2) MAX_STEPS was fixed, so bigger batches saw MORE epochs, not fewer
#       (bs=128 x 5000 = 23 epochs vs A100's 5.8).
# Both are now DERIVED from BATCH_SIZE so they cannot drift apart again.
# Canonical A100 reference point: bs=32 -> backbone 1e-5 / head 3e-4.
# Epoch budget held at 13.93 epochs (384000 samples over 27570 pairs).
case "${BATCH_SIZE}" in
  32)  DEF_BACKBONE_LR=1.0e-5;  DEF_HEAD_LR=3.0e-4;  DEF_MAX_STEPS=12000 ;;
  64)  DEF_BACKBONE_LR=1.41e-5; DEF_HEAD_LR=4.24e-4; DEF_MAX_STEPS=6000  ;;
  128) DEF_BACKBONE_LR=2.0e-5;  DEF_HEAD_LR=6.0e-4;  DEF_MAX_STEPS=3000  ;;
  *)   echo "ERROR: BATCH_SIZE=${BATCH_SIZE} has no derived LR/step defaults."
       echo "Pass BACKBONE_LR=, HEAD_LR= and MAX_STEPS= explicitly."
       [ -n "${BACKBONE_LR:-}" ] && [ -n "${HEAD_LR:-}" ] && [ -n "${MAX_STEPS:-}" ] || exit 1
       DEF_BACKBONE_LR="${BACKBONE_LR}"; DEF_HEAD_LR="${HEAD_LR}"; DEF_MAX_STEPS="${MAX_STEPS}" ;;
esac
export BACKBONE_LR="${BACKBONE_LR:-$DEF_BACKBONE_LR}"
export HEAD_LR="${HEAD_LR:-$DEF_HEAD_LR}"
export MAX_STEPS="${MAX_STEPS:-$DEF_MAX_STEPS}"
# 2026-07-19: bs=128 OOMs on the 80GB card the moment freeze_text_encoder_steps=1000
# expires and the 184M backbone unfreezes (scan intermediate B*L*(D*expand)*N =
# 128*256*1536*16 ~= 3.2GB EACH in fp32 — the fp32 scan that fixed Stage-0 costs 2x
# memory here). Gradient checkpointing is the same lever that fixed Stage-0; keep
# bs=128 so the in-batch-negative count (the MIMIC lever) is preserved.
# If it still OOMs: BATCH_SIZE=64 (halves negatives — last resort).
export GRAD_CKPT="${GRAD_CKPT:-true}"
export STAGE0_CKPT="${STAGE0_CKPT:-./outputs/h100_stage0_150m_v2/checkpoints/stage0_model_only.pt}"
# Head LR is in the run name: the LR probe runs the SAME bs=64 arm twice, so a
# bs-only name would silently overwrite the previous run's checkpoints.
export EXPERIMENT="${EXPERIMENT:-h100_kd_150m_v2_bs${BATCH_SIZE}_head${HEAD_LR}}"

cd "${SLURM_SUBMIT_DIR:-.}/hybrid_model_mamba_xlstm" 2>/dev/null || cd "${SLURM_SUBMIT_DIR:-.}"
echo "[wrapper] 150M contrastive → delegating to train_biomedclip_kd_h100.sh (bs=${BATCH_SIZE}, ckpt=${STAGE0_CKPT})"
bash scripts/train_biomedclip_kd_h100.sh
