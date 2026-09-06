#!/bin/bash
# ============================================================================
# Phase 5 — Stage-0 pre-train the 150M v2 backbone on H100.
# Thin wrapper over train_stage0_h100.sh with 150M defaults + a 150M-appropriate
# SBATCH header (multi-day walltime; the generic template's 2-day header is too
# tight for a ~120K-step 150M run). aisc-batch 7-day cap → finishes in one block.
#
# 150M ≈ 2x the per-step compute of 70M → ~120K steps ≈ 30-40h on one H100.
#
# MEMORY (aisc H100 = 80GB, NOT the 94/141GB the plan assumed): bs=48/accum=1
# OOMs in the Mamba parallel selective-scan alongside the resident 2.6B teacher.
# 80GB-safe recipe: gradient checkpointing ON + microbatch 16 * accum 3 = eff
# batch 48 (same token math, ~3B tokens; WSD reshapes decay to trainer.max_steps).
# If the log shows spare VRAM, push BATCH_SIZE up / ACCUM down to speed wall-clock.
#
# Override at submit time, e.g.:  BATCH_SIZE=24 ACCUM=2 sbatch scripts/train_stage0_150m_h100.sh
# ============================================================================
#SBATCH --partition=aisc-batch
#SBATCH --account=aisc
#SBATCH --gpus=1
#SBATCH --exclude=ga03,gx17v1,gx13v1   # gx13v1: faulty GPU (cudaErrorContained, 2026-07-19)
#SBATCH --mem=160G
#SBATCH --cpus-per-task=12
#SBATCH --time=4-00:00:00
#SBATCH --job-name=h100_stage0_150m
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log
#SBATCH --requeue

set -euo pipefail

# Overridable so the MAMBA3_PLAN.md screen arms can reuse this wrapper's 150M-tuned SBATCH
# header and stability settings (LR 4e-4, warmup 2000, grad-clip 0.5 -- all load-bearing; see
# the collapse notes above) while swapping only the architecture.
export MODEL_CONFIG="${MODEL_CONFIG:-hybrid_150m_v2}"
export MAX_STEPS="${MAX_STEPS:-120000}"     # ~3B tokens Chinchilla for 150M (eff batch 48)
export BATCH_SIZE="${BATCH_SIZE:-16}"       # 80GB-safe microbatch
export ACCUM="${ACCUM:-3}"                  # 16*3 = eff batch 48
export GRAD_CKPT="${GRAD_CKPT:-true}"       # required to fit 150M+2.6B teacher in 80GB
# 2026-07-11: the FIRST run collapsed (val PPL 1165) because the generic template
# forced 70M's LR=6e-4/warmup=1000 onto the 150M — too hot, model collapsed to
# unigram entropy (val loss 4.27 @ step666 -> 7.05 forever) as LR hit 6e-4.
# Restore the 150M's stability-tuned values (plan resolved LR=4.0e-4, warmup=2000).
export LR="${LR:-4.0e-4}"
export WARMUP="${WARMUP:-2000}"
# 2026-07-14: the 4e-4 run trained HEALTHILY to PPL 18.75 @ step 23.6k, then a single
# gradient SPIKE at step 24749 (norm 1.59 vs ~0.23 baseline) — clipped only to the loose
# max_norm=1.0 (~4x baseline) — knocked the 150M into irreversible collapse. Tighten the
# clip so spikes are bounded near the baseline. β2 (flat 0.999) and LR (flat) were NOT the cause.
export GRAD_CLIP="${GRAD_CLIP:-0.5}"
export EXPERIMENT="${EXPERIMENT:-h100_stage0_150m_v2}"
# The screen arms flip their levers here (see scripts/mamba3_arms.py); empty for A0/A1.
export EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"

cd "${SLURM_SUBMIT_DIR:-.}/hybrid_model_mamba_xlstm" 2>/dev/null || cd "${SLURM_SUBMIT_DIR:-.}"
echo "[wrapper] 150M Stage-0 → delegating to train_stage0_h100.sh (MODEL_CONFIG=${MODEL_CONFIG}, MAX_STEPS=${MAX_STEPS}, BATCH_SIZE=${BATCH_SIZE})"
bash scripts/train_stage0_h100.sh
