#!/bin/bash
# ============================================================================
# MAMBA3_PLAN_V2.md V5-D — SLURM wrapper for scripts/repair_generations.py.
#
# WHY: the decoding harness has no stop condition. beam_search_decode runs a
# fixed max_new_tokens=100 for every study, and the model was never trained to
# emit an end-of-report token (pad_token = eos_token, and every pad position
# is masked to -100). So finished reports run on into repeated sentences, which
# CheXbert scores as findings, and long reports are cut mid-phrase. This job
# repairs the dumps ALREADY ON DISK so the cost of the missing stop condition
# can be measured before anything is re-decoded or retrained.
#
# CPU-only on purpose (no --gpus): it rewrites cached text. Minutes, not hours.
#
# The aisc login node refuses direct script execution (Phase 7E), hence sbatch.
#
# ⚠ A decode-protocol change is only a fair comparison when it is applied to
#   EVERY system being compared -- the Transformer arm and the retrieval floor
#   included. Re-scoring one arm and citing it against another arm's unrepaired
#   numbers manufactures a win. Repair all of them, or cite none of them.
#
# ENV: DUMP_DIR (required), OUT_DIR, DEDUP (consecutive|all|none), TRUNCATE,
#      METRICS, VENV_ACTIVATE.
#
# Usage:
#   DUMP_DIR=results/13d_default_operator_n400 \
#     sbatch scripts/repair_generations_h100.sh
# ============================================================================
#SBATCH --partition=aisc-batch
#SBATCH --account=aisc
#SBATCH --qos=aisc
#SBATCH --exclude=ga03,gx17v1,gx13v1   # ga03: ARM node, the x86 .venv cannot
                                        # execute there. gx13v1: faulty GPU.
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --job-name=repair_generations
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log

set -euo pipefail

DUMP_DIR="${DUMP_DIR:?Set DUMP_DIR to a directory containing hyps.txt/refs.txt}"
OUT_DIR="${OUT_DIR:-${DUMP_DIR}_repaired}"
DEDUP="${DEDUP:-consecutive}"
TRUNCATE="${TRUNCATE:-true}"
METRICS="${METRICS:-true}"
VENV_ACTIVATE="${VENV_ACTIVATE:-.venv/bin/activate}"

echo "=== V5-D repair: ${DUMP_DIR} -> ${OUT_DIR} (dedup=${DEDUP} truncate=${TRUNCATE}) ==="
date; hostname
mkdir -p logs

cd "${SLURM_SUBMIT_DIR}/hybrid_model_mamba_xlstm" 2>/dev/null || cd "${SLURM_SUBMIT_DIR:-.}"

if [ ! -f "${DUMP_DIR}/hyps.txt" ] || [ ! -f "${DUMP_DIR}/refs.txt" ]; then
  echo "ERROR: ${DUMP_DIR} must contain both hyps.txt and refs.txt"
  echo "       Produce them with evaluate_report_generation.py --dump-dir."
  exit 1
fi

if [ ! -f "${VENV_ACTIVATE}" ]; then
  echo "ERROR: venv not found at ${VENV_ACTIVATE}"
  exit 1
fi

source "${VENV_ACTIVATE}"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

ARGS=(--dump-dir "${DUMP_DIR}" --out-dir "${OUT_DIR}" --dedup "${DEDUP}")
[ "${TRUNCATE}" = "false" ] && ARGS+=(--no-truncate)
[ "${METRICS}" = "true" ]  && ARGS+=(--metrics)

python scripts/repair_generations.py "${ARGS[@]}"

echo
echo "=== next ==="
echo "CheXbert on the repaired text (isolated venv, ~2 min):"
echo "    DUMP_DIR=${OUT_DIR} sbatch scripts/score_chexbert_h100.sh"
echo
echo "Then compare repaired against original as two systems, which is exactly"
echo "what the paired bootstrap is for:"
echo "    A=${OUT_DIR} B=${DUMP_DIR} NAME_A=repaired NAME_B=as_decoded \\"
echo "      PER_LABEL=true OUTPUT=analysis/bootstrap_repair_$(basename "${DUMP_DIR}").md \\"
echo "      sbatch scripts/bootstrap_compare_h100.sh"
echo
echo "⚠ Before any headline number moves: repair EVERY arm being compared,"
echo "  including the Transformer and the retrieval floor. One repaired arm"
echo "  against an unrepaired one is not a measurement."
date
