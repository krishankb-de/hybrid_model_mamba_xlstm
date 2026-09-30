#!/bin/bash
# ============================================================================
# ISBI Fig. 1 material -- SLURM wrapper for scripts/isbi_figure_cases.py.
# One test X-ray, its 4 nearest training X-rays, and the hybrid decoder's
# published beam-3 report for it. Embeds the 191,462-image training gallery
# once (cached to scratch), so later QUERY_INDEX changes are fast.
#
#   sbatch scripts/isbi_figure_cases_h100.sh
#   QUERY_INDEX=17 sbatch scripts/isbi_figure_cases_h100.sh   # another study
#
# Output (DUA-covered, never commit): results/isbi_fig1_q<QUERY_INDEX>/
# ============================================================================
#SBATCH --partition=pot-hpi-aisc-batch
#SBATCH --account=aisc
#SBATCH --gpus=1
#SBATCH --exclude=ga03,gx17v1,gx13v1   # ga03: ARM node; gx13v1: faulty GPU
#SBATCH --mem=48G
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --job-name=isbi_fig1_cases
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log

set -euo pipefail

SCRATCH_ROOT="${SCRATCH_ROOT:-/sc/scratch/$USER/hybrid_xmamba_h100}"
VENV_ACTIVATE="${VENV_ACTIVATE:-.venv/bin/activate}"
DATA="${DATA:-/sc/home/$USER/dataset/mimic_full}"
SEED="${SEED:-42}"
CHECKPOINT="${CHECKPOINT:-./outputs/h100_report_gen_m3_tower13d_s${SEED}/checkpoints/last.ckpt}"
HYPS="${HYPS:-results/report_gen_m3_test_split_s${SEED}/hyps.txt}"
QUERY_INDEX="${QUERY_INDEX:-0}"
ENCODER="${ENCODER:-adapted}"
OUT_DIR="${OUT_DIR:-results/isbi_fig1_q${QUERY_INDEX}}"

cd "${SLURM_SUBMIT_DIR}/hybrid_model_mamba_xlstm" 2>/dev/null || cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs
export HF_HOME="${SCRATCH_ROOT}/.hf"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PYTHONUNBUFFERED=1
source "${VENV_ACTIVATE}"

for f in "${CHECKPOINT}" "${HYPS}" "${DATA}/train.parquet" "${DATA}/test.parquet"; do
  [ -f "$f" ] || { echo "ERROR: not found: $f"; exit 1; }
done

python scripts/isbi_figure_cases.py \
  --checkpoint "${CHECKPOINT}" \
  --model-config hybrid_150m_m3_rrg \
  --train-parquet "${DATA}/train.parquet" \
  --test-parquet "${DATA}/test.parquet" \
  --hyps "${HYPS}" \
  --query-index "${QUERY_INDEX}" \
  --encoder "${ENCODER}" \
  --cache "${SCRATCH_ROOT}/isbi_gallery_${ENCODER}.pt" \
  --workers "${SLURM_CPUS_PER_TASK:-8}" \
  --out-dir "${OUT_DIR}"

echo "=== END: ${OUT_DIR} ==="
