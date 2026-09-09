#!/bin/bash
# ============================================================================
# Phase 14B (H100_SCALING_PLAN.md, supervisor review 2026-09-07) — SLURM
# wrapper for scripts/analyze_generation_diversity.py.
#
# WHY THIS SCRIPT EXISTS
# The login node (lx01) refuses ANY script execution, not just heavy ones —
# confirmed live in Phase 7E: `python build_mimic_cxr_local.py meta` was
# rejected with "This command is not allowed on the login node!" before it
# made a single request. Every runnable thing in this repo therefore goes
# through sbatch, including trivially cheap CPU work like this.
#
# This is a CPU-only, no-GPU job on purpose. Do not add --gpus. The analysis
# is pure-stdlib Python over text files (exact-duplicate clustering,
# distinct-n, type-token ratio, sampled self-BLEU-4) and finishes in well
# under a minute at n=2663 — the queue wait will dominate the runtime.
#
# WHAT IT MEASURES AND WHY THE CONTROLS MATTER
# Phase 11C found 1055/1433 (73.6%) of generations fell into one of 184
# exact-duplicate template clusters, on the PRE-Phase-13 checkpoint, and it
# was never re-measured on 13D — the checkpoint every headline number comes
# from. It is the single biggest validity threat to the primary result: if
# the generator is mostly emitting templates, "beats the retrieval-NN floor"
# is hollow, because emitting a plausible templated report is exactly what
# the retrieval floor does too.
#
# The original 73.6% was reported with NO CONTROL, which makes it
# uninterpretable alone: MIMIC-CXR reports are themselves heavily templated,
# and the retrieval-NN baseline emits REAL HUMAN REPORTS, so its duplication
# rate is what a "perfect" non-generative system scores. Always pass REFS and
# BASELINE when they exist — the script says so loudly in its output if you
# do not.
#
# ENV: HYPS (required), REFS, BASELINE, OUTPUT, PHASE11C_PCT, SEED,
#      VENV_ACTIVATE.
#
# Usage (the 13D official-test-split run this phase exists for):
#   HYPS=results/report_gen_tower13d_test_split/hyps.txt \
#   REFS=results/report_gen_tower13d_test_split/refs.txt \
#   BASELINE=results/retrieval_floor_test_split/hyps.txt \
#   OUTPUT=analysis/generation_diversity_13d.md \
#     sbatch scripts/analyze_diversity_h100.sh
#
#   # validate-split comparison against the historical 73.6% measurement:
#   HYPS=results/report_gen_tower13d_n1433/hyps.txt \
#   REFS=results/report_gen_tower13d_n1433/refs.txt \
#   BASELINE=results/retrieval_floor_n1433/hyps.txt \
#   OUTPUT=analysis/generation_diversity_13d_n1433.md \
#     sbatch scripts/analyze_diversity_h100.sh
# ============================================================================
#SBATCH --partition=aisc-batch
#SBATCH --account=aisc
#SBATCH --qos=aisc
#SBATCH --exclude=ga03,gx13v1           # ga03: ARM node, x86 .venv incompatible
                                        # gx13v1: faulty GPU (cudaErrorContained, 2026-07-19)
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --job-name=analyze_diversity
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log

set -euo pipefail

# NOTE (corrected 2026-09-09): an earlier revision left ga03 (ARM) unexcluded on
# the reasoning that this analysis is pure-stdlib and so runs anywhere. That was
# wrong in a way that only shows up on ARM: the script still SOURCES the x86
# .venv below, and once activated `python3` resolves to .venv/bin/python3, an
# x86 binary. On ga03 that dies with "cannot execute binary file: Exec format
# error" -- exactly what happened to the sibling bootstrap wrapper (job 2525864).
# The stdlib-only fallback only helps when the venv is ABSENT, not when it is
# present and wrong for the architecture.

HYPS="${HYPS:?Set HYPS to the generated reports file (hyps.txt from evaluate_report_generation.py --dump-dir)}"
REFS="${REFS:-}"
BASELINE="${BASELINE:-}"
OUTPUT="${OUTPUT:-analysis/generation_diversity.md}"
PHASE11C_PCT="${PHASE11C_PCT:-73.6}"
SEED="${SEED:-0}"
VENV_ACTIVATE="${VENV_ACTIVATE:-.venv/bin/activate}"

echo "=== Phase 14B generation-diversity analysis ==="
date; hostname
mkdir -p logs

cd "${SLURM_SUBMIT_DIR}/hybrid_model_mamba_xlstm" 2>/dev/null || cd "${SLURM_SUBMIT_DIR:-.}"

# Fail loudly and early rather than producing an empty or half-controlled
# report. A run that reports success while measuring nothing is this project's
# documented expensive failure mode.
if [ ! -f "${HYPS}" ]; then
  echo "ERROR: HYPS file not found: ${HYPS}"
  echo "       Generate it first with scripts/inspect_report_generation_h100.sh (DUMP_DIR=...)."
  exit 1
fi

EXTRA_ARGS=()
if [ -n "${REFS}" ]; then
  if [ ! -f "${REFS}" ]; then echo "ERROR: REFS file not found: ${REFS}"; exit 1; fi
  EXTRA_ARGS+=(--refs "${REFS}")
else
  echo "WARNING: no REFS control supplied. A bare duplication rate is the exact"
  echo "         reporting weakness Phase 14B exists to fix — pass REFS if it exists."
fi

if [ -n "${BASELINE}" ]; then
  if [ ! -f "${BASELINE}" ]; then echo "ERROR: BASELINE file not found: ${BASELINE}"; exit 1; fi
  EXTRA_ARGS+=(--baseline "${BASELINE}")
else
  echo "WARNING: no BASELINE control supplied. The retrieval-NN baseline emits real"
  echo "         human reports, so its duplication rate is the most informative control."
fi

if [ -f "${VENV_ACTIVATE}" ]; then
  source "${VENV_ACTIVATE}"
else
  echo "NOTE: ${VENV_ACTIVATE} not found; falling back to system python3."
  echo "      That is fine here — this script imports stdlib only."
fi

export PYTHONUNBUFFERED=1
mkdir -p "$(dirname "${OUTPUT}")"

python3 scripts/analyze_generation_diversity.py \
  --hyps "${HYPS}" \
  --output "${OUTPUT}" \
  --phase11c-pct "${PHASE11C_PCT}" \
  --seed "${SEED}" \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

echo "=== wrote ${OUTPUT} ==="
date
