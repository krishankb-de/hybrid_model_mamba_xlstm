#!/bin/bash
# ============================================================================
# Phase 9A prep — dump legacy-mirror report hashes for BOTH:
#   legacy_gallery_hashes.txt  (train[90%:], N~3063)  -- Phase 8D leakage guard input
#   legacy_training_hashes.txt (train[:90%], N~27570) -- Arm-0 reproduction subset input
#
# Runs OFFLINE against the itsanmolgupta/mimic-cxr-dataset HF mirror's
# ALREADY-POPULATED local cache (the same one Phase 6 used, via
# HF_DATASETS_OFFLINE=1 + MIMIC_CACHE_DIR) -- no live network needed beyond
# what is already cached. Both splits live in that same cached "train"
# config; only the in-memory slice differs, so this should not need a fresh
# download regardless of which slice was originally cached.
#
# WHY THIS EXISTS (2026-08-17): Arm-0's reproduction check (rerun the D1c
# recipe on the SAME ~27,570 studies the original experiments used, confirm
# retrieval reproduces 0.1459 +/- 1.1pp before trusting anything downstream)
# needs those specific studies present. The main `fetch` processes all
# 218,131 images in plain sequential order, scattering that subset across
# nearly all ~110 chunks -- meaning Arm-0 would not actually be checkable
# until the ~10-day full fetch is nearly done, defeating the point of a
# cheap early sanity check. This produces the input to fetch a SEPARATE,
# small (~27.5k image), OUT-isolated copy of just that subset instead --
# see scripts/build_mimic_cxr_local.sh's STUDY_HASHES lever.
#
# Same login-node restriction as everything else on this cluster applies --
# this must run via sbatch, not directly on lx01.
#
# Usage:
#   sbatch scripts/dump_legacy_hashes.sh
#   # then: wc -l legacy_gallery_hashes.txt legacy_training_hashes.txt
# ============================================================================
#SBATCH --partition=aisc-batch
#SBATCH --account=aisc
#SBATCH --qos=aisc
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --job-name=dump_legacy_hashes
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log

set -euo pipefail

VENV_ACTIVATE="${VENV_ACTIVATE:-.venv/bin/activate}"
MIMIC_CACHE_DIR="${MIMIC_CACHE_DIR:-/sc/home/$USER/dataset/mimic_cxr_cache}"

echo "=== dump_legacy_hashes ==="
date; hostname
mkdir -p logs

cd "${SLURM_SUBMIT_DIR}/hybrid_model_mamba_xlstm" 2>/dev/null || cd "${SLURM_SUBMIT_DIR:-.}"

export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

source "${VENV_ACTIVATE}"

echo "--- legacy TEST gallery (train[90%:], N~3063) -> legacy_gallery_hashes.txt ---"
python scripts/dump_legacy_gallery_hashes.py --cache-dir "${MIMIC_CACHE_DIR}" \
  --split "train[90%:]" --out legacy_gallery_hashes.txt

echo "--- legacy TRAINING set (train[:90%], N~27570) -> legacy_training_hashes.txt ---"
python scripts/dump_legacy_gallery_hashes.py --cache-dir "${MIMIC_CACHE_DIR}" \
  --split "train[:90%]" --out legacy_training_hashes.txt

echo "=== done ==="
wc -l legacy_gallery_hashes.txt legacy_training_hashes.txt
date
