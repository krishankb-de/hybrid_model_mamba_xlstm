#!/bin/bash
# TEXTENC_PLAN.md R1 guard, sourced by every textenc_*_h100.sh wrapper except textenc_stage1_h100.sh
# (which carries the same checks inline). Sets ROOT_REAL; exits 2 when this plan would write into a
# published checkout or through a symlinked output directory.
if [ -z "${TEXTENC_ROOT:-}" ]; then
  echo "ERROR: TEXTENC_ROOT is not set (TEXTENC_PLAN.md R1: this plan writes only under its own root)"
  exit 2
fi
ROOT_REAL="$(cd "${TEXTENC_ROOT}" 2>/dev/null && pwd -P || echo "${TEXTENC_ROOT}")"
case "${ROOT_REAL}" in
  */hybrid_mamba_xlstm|*/hybrid_mamba_xlstm/*|*/hybrid_isbi|*/hybrid_isbi/*|*/hybrid_model_mamba_xlstm/hybrid_model_mamba_xlstm*)
    echo "ERROR: TEXTENC_ROOT=${ROOT_REAL} is a published checkout; refusing to write there (R1)"
    exit 2 ;;
esac
for d in outputs results logs cache; do
  if [ -L "${ROOT_REAL}/${d}" ]; then
    echo "ERROR: ${ROOT_REAL}/${d} is a symlink; R1 needs a real directory"
    exit 2
  fi
done
PUBLISHED_OUTPUTS="${PUBLISHED_OUTPUTS:-/sc/home/${USER}/hybrid_mamba_xlstm/outputs}"
TEXTENC_DATA="${TEXTENC_DATA:-/sc/home/${USER}/dataset/mimic_full}"
