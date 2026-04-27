#!/bin/bash
#SBATCH --partition=mitarb
#SBATCH --account=mitarb
#SBATCH --gres=gpu:mitarb:1
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --job-name=eval_stage1_suite
#SBATCH --output=/scratch/bhushkri/hybrid_xmamba_a100_70m_40/logs/%x_%j.log
#SBATCH --error=/scratch/bhushkri/hybrid_xmamba_a100_70m_40/logs/%x_%j.log

# Phase 3: Stage 1 offline evaluation suite.
#
# Runs STS (BIOSSES, STS-B) + BEIR retrieval (NFCorpus, PubMed pairs) with
# PubMedBERT baseline comparison, then writes results/stage1_metrics.md.
#
# Decision gates (from PHASE_PLAN.md):
#   STS  : BIOSSES Spearman >= 0.50  AND  STS-B   >= 0.60
#   Ret  : PubMed R@10       >= 0.60  OR   NFCorpus nDCG@10 >= 0.25
#
# Usage:
#   # Use default Phase 2 output checkpoint
#   sbatch scripts/eval_stage1_suite.sh
#
#   # Override checkpoint
#   STAGE1_CKPT=/path/to/last.ckpt sbatch scripts/eval_stage1_suite.sh

set -euo pipefail

STAGE1_CKPT="${STAGE1_CKPT:-./outputs/hybrid_70m_stage1_kd_pubmedbert/checkpoints/last.ckpt}"
OUTPUT_ROOT="./outputs/eval_stage1_suite"
BATCH_SIZE=32
NUM_PUBMED_PAIRS=1000

echo "=== JOB START (Phase 3: Stage 1 eval suite) ==="
date
echo "Host: $(hostname)"
echo "Submit dir: ${SLURM_SUBMIT_DIR}"
echo "Checkpoint: ${STAGE1_CKPT}"
echo ""

mkdir -p /scratch/bhushkri/hybrid_xmamba_a100_70m_40/logs
cd "${SLURM_SUBMIT_DIR}/hybrid_model_mamba_xlstm"

export HF_HOME="$PWD/.hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"
export CUDA_LAUNCH_BLOCKING=0

echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"

if [ ! -d ".venv" ]; then
    echo "ERROR: Virtual environment .venv not found!"
    exit 1
fi
source .venv/bin/activate

echo ""
echo "Verifying CUDA availability:"
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device: {torch.cuda.get_device_name(0)}')
    mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f'VRAM: {mem:.1f} GB')
"
echo ""

# Verify Stage 1 checkpoint exists
if [ ! -f "${STAGE1_CKPT}" ]; then
    echo "ERROR: Stage 1 checkpoint not found: ${STAGE1_CKPT}"
    echo "Has train_stage1_distill.sh completed?"
    echo "Available checkpoints:"
    ls -la ./outputs/hybrid_70m_stage1_kd_pubmedbert/checkpoints/ 2>/dev/null || echo "  (directory missing)"
    exit 1
fi
echo "Stage 1 checkpoint verified: ${STAGE1_CKPT}"

nvidia-smi

mkdir -p "${OUTPUT_ROOT}"
mkdir -p results

echo ""
echo "=== Phase 3a: STS evaluation (BIOSSES + STS-B + MedSTS) ==="
python scripts/evaluate_sts.py \
    --checkpoint "${STAGE1_CKPT}" \
    --datasets all \
    --compare-pubmedbert \
    --batch-size ${BATCH_SIZE} \
    --max-length 512 \
    --output-dir "${OUTPUT_ROOT}/sts" \
    || echo "WARNING: STS eval failed — check log above"

echo ""
echo "=== Phase 3b: BEIR retrieval evaluation (NFCorpus + PubMed pairs) ==="
python scripts/evaluate_retrieval.py \
    --checkpoint "${STAGE1_CKPT}" \
    --benchmarks nfcorpus pubmed \
    --compare-pubmedbert \
    --batch-size ${BATCH_SIZE} \
    --max-length 512 \
    --num-pubmed-pairs ${NUM_PUBMED_PAIRS} \
    --output-dir "${OUTPUT_ROOT}/retrieval" \
    || echo "WARNING: retrieval eval failed — check log above"

echo ""
echo "=== Phase 3c: TREC-COVID retrieval (larger corpus, may be slow) ==="
python scripts/evaluate_retrieval.py \
    --checkpoint "${STAGE1_CKPT}" \
    --benchmarks trec-covid \
    --compare-pubmedbert \
    --batch-size ${BATCH_SIZE} \
    --max-length 512 \
    --output-dir "${OUTPUT_ROOT}/retrieval_trec" \
    || echo "WARNING: TREC-COVID eval failed — check log above"

echo ""
echo "=== Results ==="
echo "Summary written to: results/stage1_metrics.md"
cat results/stage1_metrics.md 2>/dev/null || echo "  (results file not generated)"

echo ""
echo "JSON outputs in: ${OUTPUT_ROOT}/"
ls -la "${OUTPUT_ROOT}/" 2>/dev/null

echo ""
echo "=== JOB END (Phase 3: Stage 1 eval suite) ==="
date
