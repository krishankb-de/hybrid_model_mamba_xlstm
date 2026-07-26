#!/bin/bash
# ============================================================================
# H100 joint-contrastive template (canonical recipe on hybrid_70m_v2 backbone).
# H100 port of train_biomedclip_kd_phase15_pure.sh, but with the CANONICAL recipe
# (freq_kd=false, vit_unfreeze_blocks=2). Phase 6 clones this for 150M with a
# batch sweep (train_biomedclip_kd_150m_h100.sh).
#
# THE H100 RETRIEVAL LEVER: dataset.batch_size=128 at accumulate_grad_batches=1.
# The CLIP loss uses in-batch negatives only (no all_gather, moco=0), so the true
# negative count == per-step batch. A100 was capped at 32 (~31 negatives); H100
# 141GB lifts this to 128 (~127 negatives) — the single biggest MIMIC lever, and
# it also cuts epoch count on the 27.5k-pair set (less overfitting). Grad-accum
# does NOT add negatives, so keep accum=1 and scale batch_size instead.
#
# LR defaults are sqrt-scaled for the 4x batch (32->128): backbone 1e-5->2e-5,
# head 3e-4->6e-4. Override per-arm with BACKBONE_LR= / HEAD_LR= (see note below).
#
# Kill gates: cos_text_teacher >= 0.85 by step 1000; val/clip_loss < 3.0 by 1000;
#   MIMIC R@10 >= 0.1045 (current best) by step 3000; ViT group lr == 1e-6.
#
# ENV placeholders (adjust for aisc/H100): SCRATCH_ROOT, VENV_ACTIVATE, STAGE0_CKPT.
# ============================================================================
#SBATCH --partition=aisc-batch
#SBATCH --account=aisc
#SBATCH --gpus=1
#SBATCH --exclude=ga03,gx17v1,gx13v1   # gx13v1: faulty GPU (cudaErrorContained, 2026-07-19)
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --time=16:00:00
#SBATCH --job-name=h100_kd_contrastive
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log
#SBATCH --requeue

set -euo pipefail

SCRATCH_ROOT="${SCRATCH_ROOT:-/sc/scratch/$USER/hybrid_xmamba_h100}"
VENV_ACTIVATE="${VENV_ACTIVATE:-.venv/bin/activate}"
MODEL_CONFIG="${MODEL_CONFIG:-hybrid_70m_v2}"
BATCH_SIZE="${BATCH_SIZE:-128}"
# 2026-07-19: compile OFF by default. With compile_model=true the joint run died in
# sanity-val with "CUDA error: Invalid access of peer GPU memory over nvlink or a
# hardware error" (async — reported at the embedding lookup, faulted earlier). The text
# encoder uses the same custom Mamba/mLSTM Triton kernels that Stage-0 runs with
# compile_model=false for exactly this reason. Set COMPILE=true to re-test.
COMPILE="${COMPILE:-false}"
GRAD_CKPT="${GRAD_CKPT:-false}"   # flip true if bs=128 OOMs on the 80GB card
# Epoch budget matters more than raw steps: MIMIC is only 27570 pairs, so bs=128 x
# 5000 steps = 23 epochs (vs A100 bs=32 x 5000 = 5.8) and val/loss bottomed at ~2750.
# Scale MAX_STEPS DOWN as batch goes UP to keep epochs comparable.
MAX_STEPS="${MAX_STEPS:-5000}"
# 2026-07-21: LRs are now ENV-OVERRIDABLE. They were hardcoded at the bs=128
# sqrt-scaled values (backbone 2e-5 / head 6e-4), so the Phase-6 batch sweep ran
# EVERY arm at bs=128 LRs — including the bs=64 arm that produced the best number
# (0.1113). The sweep was therefore never LR-matched, and grad_norm ran ~12.3
# against gradient_clip_val=1.0 (~12x clipping every step). Canonical A100 values
# at bs=32 are backbone 1e-5 / head 3e-4; sqrt-scale from there for other batches.
BACKBONE_LR="${BACKBONE_LR:-2e-5}"
HEAD_LR="${HEAD_LR:-6e-4}"
STAGE0_CKPT="${STAGE0_CKPT:-./outputs/h100_stage0_${MODEL_CONFIG}/checkpoints/stage0_model_only.pt}"
# 2026-07-22: MIMIC-CXR (itsanmolgupta/mimic-cxr-dataset) is GATED — an online
# load_dataset 401s (DatasetNotFoundError). The 2026-07-19 Phase-6 runs only
# worked OFFLINE against a pre-downloaded local cache. Default to that known-good
# cache dir + offline mode below. Point MIMIC_CACHE_DIR at the dir that actually
# holds the downloaded dataset (verify with `ls`), or export HF_HUB_OFFLINE=0 +
# HF_TOKEN for a first-time online populate.
MIMIC_CACHE_DIR="${MIMIC_CACHE_DIR:-/sc/home/$USER/dataset/mimic_cxr_cache}"
EXPERIMENT="${EXPERIMENT:-h100_kd_${MODEL_CONFIG}_bs${BATCH_SIZE}}"

# --- Phase 6D/6E/6F levers (2026-07-25) -------------------------------------
# ALL defaults below reproduce the Phase-6B recipe EXACTLY, so an unmodified
# invocation is the 6D-0 control. Seven consecutive nulls (PPL, model scale,
# negatives, epochs, batch, and both LR arms) against one positive (ViT unfreeze
# 0->2, +2.5pp) is why these are the levers worth spending H100 hours on.
#
#   6D-1  VIT_UNFREEZE=4|6|12    the only lever with a measured positive
#   6D-2  KD_DECAY_STEPS=2000    decay the KD anchor to ALPHA_KD_FLOOR
#   6D-3  CLIP_LOSS=siglip + MULTIPOS=true
#   6D-5  GAMMA_SIMCSE=0.0       ablate the SimCSE term
#   6E-1  BIDIRECTIONAL=true     second pass over the reversed sequence
#   6F-1  SELECTION_SPLIT=true   disjoint selection split (not selection-on-test)
VIT_UNFREEZE="${VIT_UNFREEZE:-2}"
# 6G-1: vit_lr has been pinned at 1e-6 for the ENTIRE project — three orders of
# magnitude below head_lr — so 6D's winning arm (depth 12, +5.2pp) is one where
# the whole tower is unfrozen but barely moving. Depth is exhausted at 12
# (ViT-B/16 has 12 blocks); LR is the untouched half of the dose.
VIT_LR="${VIT_LR:-1e-6}"
# 6G-2: "blocks" (default) unfreezes transformer blocks only — patch_embed,
# cls_token, pos_embed, the final norm and the visual projection stay frozen even
# at depth 12. "all" unfreezes the entire image tower.
VIT_SCOPE="${VIT_SCOPE:-blocks}"
KD_DECAY_STEPS="${KD_DECAY_STEPS:-0}"
ALPHA_KD_FLOOR="${ALPHA_KD_FLOOR:-0.0}"
CLIP_LOSS="${CLIP_LOSS:-infonce}"
MULTIPOS="${MULTIPOS:-false}"
GAMMA_SIMCSE="${GAMMA_SIMCSE:-0.1}"
BIDIRECTIONAL="${BIDIRECTIONAL:-false}"
SELECTION_SPLIT="${SELECTION_SPLIT:-false}"

# 6F: point the val loader at a slice disjoint from the test gallery. The
# authoritative eval hardcodes train[90%:], so only the training/val slices move.
#
# TWO TRAPS, both hit on 2026-07-26 (jobs 2372273-5 died in argument parsing):
#  (1) HF slice syntax contains '[', which is a Hydra override-grammar
#      metacharacter. The shell strips "..." before exec, so Hydra receives a
#      bare train[:90%] and rejects it with "mismatched input '['". The value
#      must arrive at Hydra STILL QUOTED, hence the embedded single quotes.
#  (2) These overrides were emitted unconditionally, so the bug fired on every
#      arm even though SELECTION_SPLIT=false resolves to the yaml's own values.
#      Now nothing is passed unless 6F is actually requested — the default path
#      keeps the exact argv the Phase-6/6B runs used, which is what "6D-0 is
#      bit-identical to the control" is supposed to mean.
SPLIT_OVERRIDES=()
if [ "${SELECTION_SPLIT}" = "true" ]; then
  SPLIT_OVERRIDES=(
    "dataset.train_split='train[:85%]'"
    "dataset.validation_split='train[85%:90%]'"
  )
  echo "=== 6F: disjoint selection split ON (train[:85%] / select train[85%:90%] / test train[90%:]) ==="
fi

echo "=== H100 joint contrastive: ${MODEL_CONFIG}, bs=${BATCH_SIZE} (true negatives) ==="
echo "=== LRs: backbone=${BACKBONE_LR} head=${HEAD_LR} | max_steps=${MAX_STEPS} ==="
echo "=== 6D levers: vit_unfreeze=${VIT_UNFREEZE} vit_lr=${VIT_LR} vit_scope=${VIT_SCOPE} kd_decay=${KD_DECAY_STEPS}->${ALPHA_KD_FLOOR} clip_loss=${CLIP_LOSS} multipos=${MULTIPOS} gamma_simcse=${GAMMA_SIMCSE} bidirectional=${BIDIRECTIONAL} ==="
date; hostname
mkdir -p logs "${MIMIC_CACHE_DIR}"

cd "${SLURM_SUBMIT_DIR}/hybrid_model_mamba_xlstm" 2>/dev/null || cd "${SLURM_SUBMIT_DIR:-.}"

export HF_HOME="${SCRATCH_ROOT}/.hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
# Gated MIMIC-CXR: load from local cache, never hit the Hub (avoids the 401 that
# killed job 2357924). BiomedCLIP + gpt2 snapshots are already in HF_HOME cache,
# so offline resolves them locally too. Override to 0 only to (re)download.
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TORCHINDUCTOR_CACHE_DIR="${SCRATCH_ROOT}/.torchinductor"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# Set CUDA_LAUNCH_BLOCKING=1 to make CUDA errors report at the REAL faulting kernel
# (async reporting otherwise blames a later op, e.g. the embedding lookup). Slow — debug only.
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"
export PYTHONUNBUFFERED=1   # flush progress bar to the log live (else block-buffered → looks frozen)

source "${VENV_ACTIVATE}"
python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print('GPU:', torch.cuda.get_device_name(0), f'{torch.cuda.get_device_properties(0).total_memory/1024**3:.0f}GB')"
nvidia-smi

if [ ! -f "${STAGE0_CKPT}" ]; then
  echo "ERROR: Stage-0 checkpoint not found: ${STAGE0_CKPT}"
  echo "Extract it from the Stage-0 run (strip 'model.' prefix into a *_model_only.pt)."
  exit 1
fi
echo "Stage-0 checkpoint: ${STAGE0_CKPT}"

echo "Starting joint contrastive (canonical: freq_kd=false, moco=0)..."
python scripts/train_contrastive.py \
  --config-name config_70m \
  model=${MODEL_CONFIG} \
  dataset=mimic_cxr \
  +distill=biomedclip_kd_joint_v2 \
  distill.freq_kd=false \
  distill.vit_unfreeze_blocks=${VIT_UNFREEZE} \
  distill.vit_lr=${VIT_LR} \
  distill.vit_unfreeze_scope=${VIT_SCOPE} \
  distill.backbone_lr=${BACKBONE_LR} \
  distill.head_lr=${HEAD_LR} \
  distill.gamma_simcse=${GAMMA_SIMCSE} \
  distill.kd_decay_steps=${KD_DECAY_STEPS} \
  distill.alpha_kd_floor=${ALPHA_KD_FLOOR} \
  distill.clip_loss_type=${CLIP_LOSS} \
  distill.use_multipos=${MULTIPOS} \
  ++model.bidirectional_encode=${BIDIRECTIONAL} \
  trainer=h100_single_gpu \
  contrastive_mode=joint \
  trainer.max_steps=${MAX_STEPS} \
  trainer.accumulate_grad_batches=1 \
  trainer.val_check_interval=250 \
  trainer.log_every_n_steps=25 \
  trainer.compile_model=${COMPILE} \
  dataset.batch_size=${BATCH_SIZE} \
  dataset.eval_batch_size=${BATCH_SIZE} \
  dataset.num_workers=8 \
  dataset.pin_memory=true \
  dataset.cache_dir="${MIMIC_CACHE_DIR}" \
  ${SPLIT_OVERRIDES[@]+"${SPLIT_OVERRIDES[@]}"} \
  model.use_gradient_checkpointing=${GRAD_CKPT} \
  lm_checkpoint="${STAGE0_CKPT}" \
  experiment_name=${EXPERIMENT} \
  output_dir=./outputs/${EXPERIMENT} \
  wandb.enabled=false

echo "=== END: best ckpt in ./outputs/${EXPERIMENT}/checkpoints/ ==="
date
