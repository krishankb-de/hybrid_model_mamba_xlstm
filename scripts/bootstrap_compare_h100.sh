#!/bin/bash
# ============================================================================
# Phase 14A-6 (H100_SCALING_PLAN.md) — SLURM wrapper for bootstrap_compare.py.
#
# WHY: Phase 14A's pre-registered bar is stated in terms of a 95% bootstrap CI
# on the DIFFERENCE between the two systems. Without an interval "matches" is
# not a testable claim, and the observed margins are small enough that it
# matters — hybrid leads CheXbert-14-micro 0.4736 vs 0.4590 while the
# Transformer leads ROUGE-L 0.1936 vs 0.1899 (a gap of 0.0037). Calling either
# without a CI would be reading noise.
#
# CPU-only on purpose (no --gpus): it re-scores cached text and label matrices.
# 1000 paired resamples x 2 systems x corpus-BLEU is the slow part; 2h is ample.
#
# The aisc login node refuses direct script execution (Phase 7E), hence sbatch.
#
# ENV: A, B (result dirs containing hyps.txt), REFS (defaults to A/refs.txt),
#      NAME_A, NAME_B, OUTPUT, SAMPLES, SEED, VENV_ACTIVATE.
#
# Usage:
#   A=results/report_gen_tower13d_test_split \
#   B=results/report_gen_transformer_test_split \
#   NAME_A=hybrid_13D NAME_B=transformer \
#   OUTPUT=analysis/bootstrap_hybrid_vs_transformer.md \
#     sbatch scripts/bootstrap_compare_h100.sh
# ============================================================================
#SBATCH --partition=aisc-batch
#SBATCH --account=aisc
#SBATCH --qos=aisc
#SBATCH --exclude=ga03,gx13v1   # ga03: ARM node, the x86 .venv cannot execute
                                # there ("cannot execute binary file: Exec format
                                # error", job 2525864). gx13v1: faulty GPU.
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --job-name=bootstrap_compare
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log

set -euo pipefail

A="${A:?Set A to the first result dir (e.g. results/report_gen_tower13d_test_split)}"
B="${B:?Set B to the second result dir (e.g. results/report_gen_transformer_test_split)}"
REFS="${REFS:-${A}/refs.txt}"
NAME_A="${NAME_A:-A}"
NAME_B="${NAME_B:-B}"
OUTPUT="${OUTPUT:-analysis/bootstrap_compare.md}"
SAMPLES="${SAMPLES:-1000}"
SEED="${SEED:-0}"
VENV_ACTIVATE="${VENV_ACTIVATE:-.venv/bin/activate}"

echo "=== Phase 14A-6 paired bootstrap: ${NAME_A} vs ${NAME_B} ==="
date; hostname
mkdir -p logs

cd "${SLURM_SUBMIT_DIR}/hybrid_model_mamba_xlstm" 2>/dev/null || cd "${SLURM_SUBMIT_DIR:-.}"

for f in "${A}/hyps.txt" "${B}/hyps.txt" "${REFS}"; do
  if [ ! -f "${f}" ]; then
    echo "ERROR: required file not found: ${f}"
    echo "       Both systems must have been evaluated with the same DUMP_DIR convention."
    exit 1
  fi
done

# CheXbert label matrices are optional: without them only the text metrics are
# compared, which still settles the pre-registered ROUGE-L question. Warn rather
# than fail, but say exactly what is missing and how to get it.
EXTRA_ARGS=()
if [ -f "${A}/chexbert_labels.json" ] && [ -f "${B}/chexbert_labels.json" ]; then
  EXTRA_ARGS+=(--labels-a "${A}/chexbert_labels.json" --labels-b "${B}/chexbert_labels.json")
  echo "CheXbert label matrices found -- CheXbert F1 will get confidence intervals too."
else
  echo "WARNING: chexbert_labels.json missing in ${A} and/or ${B}."
  echo "         Only ROUGE-L/BLEU will get intervals. To include CheXbert F1, re-run"
  echo "         score_chexbert_h100.sh for both (it now writes the label matrices)."
fi

source "${VENV_ACTIVATE}"
export PYTHONUNBUFFERED=1
mkdir -p "$(dirname "${OUTPUT}")"

python3 scripts/bootstrap_compare.py \
  --hyps-a "${A}/hyps.txt" \
  --hyps-b "${B}/hyps.txt" \
  --refs "${REFS}" \
  --name-a "${NAME_A}" \
  --name-b "${NAME_B}" \
  --bootstrap-samples "${SAMPLES}" \
  --seed "${SEED}" \
  --output "${OUTPUT}" \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

echo "=== wrote ${OUTPUT} ==="
date
