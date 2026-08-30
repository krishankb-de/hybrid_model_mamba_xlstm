#!/bin/bash
# ============================================================================
# Phase 11C (H100_SCALING_PLAN.md) — retrieval-NN baseline: retrieve the
# nearest TRAINING report by stock BiomedCLIP pooled-image cosine similarity
# and emit it verbatim. "This baseline is the real floor; a generator that
# does not beat it has contributed nothing." Must run BEFORE trusting any
# --checkpoint mode generator number (see job 2478647/2482543's confirmed
# template-memorization result, 2026-08-24).
#
# SLURM wrapper for scripts/evaluate_report_generation.py --retrieval-baseline,
# same login-node-refuses-this-command constraint as inspect_report_generation_
# h100.sh. Mirrors that script's conventions.
#
# No Phase-9 best contrastive checkpoint exists yet -- this embeds with STOCK
# BiomedCLIP (matches 10C's current frozen-tower default); swap in a
# fine-tuned checkpoint later without changing evaluate_report_generation.py.
#
# Embeds the WHOLE gallery (up to MAX_GALLERY) every run -- no caching yet.
# For arm0 (~19881 images) this is a few minutes on H100; --time is generous.
# ============================================================================
#SBATCH --partition=aisc-batch
#SBATCH --account=aisc
#SBATCH --gpus=1
#SBATCH --exclude=ga03,gx17v1,gx13v1   # ga03: ARM node, x86 .venv incompatible;
                                        # gx13v1: faulty GPU (cudaErrorContained, 2026-07-19)
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --job-name=retrieval_baseline
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log

set -euo pipefail

SCRATCH_ROOT="${SCRATCH_ROOT:-/sc/scratch/$USER/hybrid_xmamba_h100}"
VENV_ACTIVATE="${VENV_ACTIVATE:-.venv/bin/activate}"

# 2026-08-30: these defaulted to the arm0/ subset (19881-image gallery) long
# after full-data Phase 8 pack + Phase 10E training (job 2491338,
# EXPERIMENT=h100_report_gen_full) landed, silently making the DEFAULT
# invocation of this script compare against the wrong, much-smaller arm0
# floor -- caught live when job 2494817 (run with only DUMP_DIR set, no
# TRAIN_PARQUET/PARQUET override) reproduced the arm0 retrieval numbers
# (rouge_l 0.369) instead of the intended full-data ones (rouge_l 0.188,
# jobs 2491600/2491687). arm0 is CLOSED/historical per CLAUDE.md's Phase 9
# note -- full data is the active target, so the default should point there.
# Override back to arm0/... explicitly if an arm0 comparison is ever needed.
TRAIN_PARQUET="${TRAIN_PARQUET:-/sc/home/$USER/dataset/mimic_full/train.parquet}"
# Default to VALIDATION images, not train — same honesty rule as the
# --checkpoint inspection wrapper (a query retrieving itself from its own
# gallery would trivially score 1.0 and mean nothing).
PARQUET="${PARQUET:-/sc/home/$USER/dataset/mimic_full/validate.parquet}"
NUM_SAMPLES="${NUM_SAMPLES:-10}"
MAX_GALLERY="${MAX_GALLERY:-0}"   # 0 = full gallery
# Phase 11B (2026-08-28, fixed 2026-08-29 against the real f1chexbert API):
# opt-in CheXbert F1 alongside ROUGE-L/BLEU, off by default -- needs the
# f1chexbert package (pip install f1chexbert) + network for chexbert.pth.
CHEXBERT="${CHEXBERT:-false}"
# Phase 11B (2026-08-29): dump hyps.txt/refs.txt for later CheXbert scoring in
# the isolated venv (see score_chexbert_h100.sh) instead of/in addition to
# --chexbert above. Off by default.
DUMP_DIR="${DUMP_DIR:-}"

echo "=== Phase 11C retrieval-NN baseline: gallery=${TRAIN_PARQUET} query=${PARQUET} ==="
echo "=== num_samples=${NUM_SAMPLES} max_gallery=${MAX_GALLERY} ==="
date; hostname
mkdir -p logs

cd "${SLURM_SUBMIT_DIR}/hybrid_model_mamba_xlstm" 2>/dev/null || cd "${SLURM_SUBMIT_DIR:-.}"

export HF_HOME="${SCRATCH_ROOT}/.hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PYTHONUNBUFFERED=1

source "${VENV_ACTIVATE}"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

CHEXBERT_ARGS=()
if [ "${CHEXBERT}" = "true" ]; then
  CHEXBERT_ARGS+=(--chexbert)
fi
if [ -n "${DUMP_DIR}" ]; then
  CHEXBERT_ARGS+=(--dump-dir "${DUMP_DIR}")
fi

python scripts/evaluate_report_generation.py \
  --retrieval-baseline \
  --train-parquet "${TRAIN_PARQUET}" \
  --parquet "${PARQUET}" \
  --num-samples "${NUM_SAMPLES}" \
  --max-gallery "${MAX_GALLERY}" \
  ${CHEXBERT_ARGS[@]+"${CHEXBERT_ARGS[@]}"}

echo "=== END ==="
date
