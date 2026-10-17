#!/bin/bash
# ============================================================================
# ISBI B10 -- SLURM wrapper for scripts/isbi_published_rrg.py.
#
#   MODE=vocab                       CPU, project .venv: rebuild the R2Gen vocabulary
#   MODE=generate MODEL=<name>       GPU, external/.venv_rrg: decode the test split
#        MODEL in r2gen | r2gencmn | r2genmamba | cvt2distilgpt2 | cxrmate_single_tf
#        OUT defaults to results/published_${MODEL}_test_split; LIMIT=N for a smoke run
#   MODE=protocol_b SYSTEMS="name=dir ..." [HYP_MODE=full]  CPU, project .venv: findings-only dirs
#   MODE=textmetrics DIRS="dir ..."          CPU, project .venv: ROUGE-L / BLEU
#
# GPU only for MODE=generate:  sbatch --gpus=1 scripts/isbi_published_rrg_h100.sh
# Submit from ~/hybrid_isbi. HF_HUB_OFFLINE=1 (setup job pre-fetched the HF repos).
# ============================================================================
#SBATCH --partition=pot-hpi-aisc-batch
#SBATCH --account=aisc
#SBATCH --qos=aisc
#SBATCH --exclude=ga03,gx13v1   # ga03: ARM node; gx13v1: faulty GPU
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --job-name=isbi_published_rrg
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs

MODE="${MODE:?Set MODE to vocab|generate|protocol_b|protocol_imp|protocol_combined|textmetrics}"
export HF_HOME="${HF_HOME:-/sc/scratch/${USER}/hybrid_xmamba_h100/.hf}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
echo "=== ISBI B10 published RRG: MODE=${MODE} MODEL=${MODEL:-} ==="
date; hostname

case "${MODE}" in
  generate)
    source external/.venv_rrg/bin/activate
    MODEL="${MODEL:?Set MODEL}"
    OUT="${OUT:-results/published_${MODEL}_test_split}"
    EXTRA=()
    if [ -n "${LIMIT:-}" ]; then EXTRA+=(--limit "${LIMIT}"); fi
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
    python scripts/isbi_published_rrg.py generate --model "${MODEL}" --out "${OUT}" ${EXTRA[@]+"${EXTRA[@]}"}
    ;;
  vocab)
    source .venv/bin/activate
    python scripts/isbi_published_rrg.py vocab
    ;;
  protocol_b)
    source .venv/bin/activate
    # shellcheck disable=SC2086
    python scripts/isbi_published_rrg.py protocol_b --hyp-mode "${HYP_MODE:-findings}" --systems ${SYSTEMS:?Set SYSTEMS}
    ;;
  protocol_imp|protocol_combined)
    source .venv/bin/activate
    python scripts/isbi_published_rrg.py "${MODE}" --systems ${SYSTEMS:?Set SYSTEMS}
    ;;
  textmetrics)
    source .venv/bin/activate
    # shellcheck disable=SC2086
    python scripts/isbi_published_rrg.py textmetrics --dirs ${DIRS:?Set DIRS}
    ;;
  *) echo "unknown MODE ${MODE}"; exit 1 ;;
esac
echo "=== END ==="; date
