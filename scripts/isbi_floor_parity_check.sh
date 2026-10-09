#!/bin/bash
# ============================================================================
# ISBI_BASELINES_PLAN.md B3-A, rule R1 — the re-run BiomedCLIP floor must
# reproduce the published floor byte for byte. Exits non-zero otherwise, so
# every new-encoder job submitted with --dependency=afterok on this one never
# starts if the refactored floor code changed the published result.
#
# ENV: NEW (default results/isbi_floor_biomedclip_test_split),
#      OLD (default results/retrieval_floor_test_split).
# ============================================================================
#SBATCH --partition=pot-hpi-aisc-batch
#SBATCH --account=aisc
#SBATCH --qos=aisc
#SBATCH --mem=2G
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --job-name=isbi_floor_parity
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}/hybrid_model_mamba_xlstm" 2>/dev/null || cd "${SLURM_SUBMIT_DIR:-.}"
NEW="${NEW:-results/isbi_floor_biomedclip_test_split}"
OLD="${OLD:-results/retrieval_floor_test_split}"
for f in hyps.txt refs.txt; do
  if cmp -s "${NEW}/${f}" "${OLD}/${f}"; then
    echo "R1 PASS: ${NEW}/${f} is byte-identical to ${OLD}/${f}"
  else
    echo "R1 FAIL: ${NEW}/${f} differs from ${OLD}/${f}"
    exit 1
  fi
done
