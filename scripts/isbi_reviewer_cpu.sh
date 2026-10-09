#!/bin/bash
# ============================================================================
# ISBI_BASELINES_PLAN.md B7-A + B9-A/B (CPU). Context statistics and the error analysis.
# Aggregates go to analysis/isbi_reviewer/ ; the qualitative file (MIMIC text) goes to
# results/isbi_error_analysis/ and must never leave the cluster (rule R6).
# ENV: SYSTEMS (space separated name=dir), QUAL (name=dir for the qualitative file).
# ============================================================================
#SBATCH --partition=pot-hpi-aisc-batch
#SBATCH --account=aisc
#SBATCH --qos=aisc
#SBATCH --exclude=ga03,gx13v1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --job-name=isbi_reviewer_cpu
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}/hybrid_model_mamba_xlstm" 2>/dev/null || cd "${SLURM_SUBMIT_DIR:-.}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/sc/scratch/$USER/hybrid_xmamba_h100}"
export HF_HOME="${SCRATCH_ROOT}/.hf" HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
source "${VENV_ACTIVATE:-.venv/bin/activate}"
mkdir -p analysis/isbi_reviewer results/isbi_error_analysis
DATA="${DATA:-/sc/home/$USER/dataset/mimic_full}"
SKIP_STATS="${SKIP_STATS:-false}"
if [ "${SKIP_STATS}" != "true" ]; then
  echo "=== B7-A context statistics ==="
  python scripts/isbi_context_stats.py --data "${DATA}" --output analysis/isbi_reviewer/context_stats.json
fi
echo "=== B9 error analysis ==="
ARGS=()
for s in ${SYSTEMS}; do ARGS+=(--system "$s"); done
QARGS=()
if [ -n "${QUAL:-}" ]; then
  QARGS+=(--qualitative "${QUAL%%=*}=results/isbi_error_analysis/qualitative_${QUAL%%=*}.md")
fi
python scripts/isbi_error_analysis.py "${ARGS[@]}" --output "${OUTPUT:-analysis/isbi_reviewer/error_analysis.json}" ${QARGS[@]+"${QARGS[@]}"}
echo "=== END ==="
