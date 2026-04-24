#!/bin/bash
#SBATCH --partition=mitarb
#SBATCH --account=mitarb
#SBATCH --gres=gpu:mitarb:1
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --job-name=eval_stage1_kd
#SBATCH --output=/scratch/bhushkri/hybrid_xmamba_a100_70m_40/logs/%x_%j.log
#SBATCH --error=/scratch/bhushkri/hybrid_xmamba_a100_70m_40/logs/%x_%j.log

# Stage 1 KD evaluation: STS (BIOSSES + STS-B), PubMed retrieval (R@1/5/10),
# and PubMed perplexity on test split.
# Checkpoint: stage1_model_only.pt (120 keys: lm.*, projection_head.*, logit_scale)
# Override checkpoint: STAGE1_CHECKPOINT=/path sbatch eval_stage1_kd.sh

set -euo pipefail

STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-./outputs/hybrid_70m_stage1_kd_pubmedbert/checkpoints/stage1_model_only.pt}"
OUTPUT_DIR="./outputs/hybrid_70m_stage1_kd_pubmedbert/eval_results"

echo "=== JOB START (Stage 1 KD Evaluation) ==="
date
echo "Host: $(hostname)"
echo "Submit dir: ${SLURM_SUBMIT_DIR}"
echo "Checkpoint: ${STAGE1_CHECKPOINT}"
echo ""

mkdir -p /scratch/bhushkri/hybrid_xmamba_a100_70m_40/logs
cd "${SLURM_SUBMIT_DIR}/hybrid_model_mamba_xlstm"

export HF_HOME="$PWD/.hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"
export CUDA_LAUNCH_BLOCKING=0

if [ ! -d ".venv" ]; then
    echo "ERROR: .venv not found"
    exit 1
fi
source .venv/bin/activate

echo "Verifying environment:"
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"
echo ""
nvidia-smi
echo ""

if [ ! -f "${STAGE1_CHECKPOINT}" ]; then
    echo "ERROR: Checkpoint not found: ${STAGE1_CHECKPOINT}"
    echo "Run extraction first:"
    echo "  python -c \"import torch; ckpt = torch.load('outputs/hybrid_70m_stage1_kd_pubmedbert/checkpoints/last.ckpt', map_location='cpu', weights_only=False); student = {k[len('model.'):]: v for k, v in ckpt['state_dict'].items() if k.startswith('model.')}; torch.save({'state_dict': student}, 'outputs/hybrid_70m_stage1_kd_pubmedbert/checkpoints/stage1_model_only.pt'); print(f'Extracted {len(student)} keys')\""
    exit 1
fi

echo "Checkpoint verified: ${STAGE1_CHECKPOINT} ($(du -h "${STAGE1_CHECKPOINT}" | cut -f1))"
echo ""

mkdir -p "${OUTPUT_DIR}/sts"
mkdir -p "${OUTPUT_DIR}/retrieval"
mkdir -p "${OUTPUT_DIR}/lm"

# ---------------------------------------------------------------------------
# 1. BIOSSES STS (biomedical sentence similarity)
# ---------------------------------------------------------------------------
echo "================================================================================";
echo "=== Step 1/4: STS — BIOSSES (biomedical, 100 pairs) ==="
echo "Expected: Spearman ~0.65–0.80 | Runtime: ~5 min"
echo "================================================================================"

python scripts/evaluate_sts.py \
    --checkpoint "${STAGE1_CHECKPOINT}" \
    --dataset biosses \
    --batch-size 32 \
    --max-length 512 \
    --output-dir "${OUTPUT_DIR}/sts" \
    --device cuda
BIOSSES_EXIT=$?

echo ""

# ---------------------------------------------------------------------------
# 2. STS-B (general-domain STS, ~1379 test pairs)
# ---------------------------------------------------------------------------
echo "================================================================================";
echo "=== Step 2/4: STS — STS-B (general domain) ==="
echo "Expected: Spearman ~0.55–0.70 | Runtime: ~5 min"
echo "================================================================================"

python scripts/evaluate_sts.py \
    --checkpoint "${STAGE1_CHECKPOINT}" \
    --dataset stsb \
    --batch-size 32 \
    --max-length 512 \
    --output-dir "${OUTPUT_DIR}/sts" \
    --device cuda
STSB_EXIT=$?

echo ""

# ---------------------------------------------------------------------------
# 3. PubMed retrieval R@1 / R@5 / R@10
# ---------------------------------------------------------------------------
echo "================================================================================";
echo "=== Step 3/4: Retrieval — PubMed (1000 pairs, R@1/5/10) ==="
echo "Expected: R@1 ~0.50–0.75 | Runtime: ~10 min"
echo "================================================================================"

python scripts/evaluate_retrieval.py \
    --checkpoint "${STAGE1_CHECKPOINT}" \
    --num-pairs 1000 \
    --batch-size 32 \
    --max-length 512 \
    --output-dir "${OUTPUT_DIR}/retrieval" \
    --device cuda
RETRIEVAL_EXIT=$?

echo ""

# ---------------------------------------------------------------------------
# 4. PubMed perplexity — test split
# ---------------------------------------------------------------------------
echo "================================================================================";
echo "=== Step 4/4: Perplexity — PubMed test split ==="
echo "Expected: PPL ~12–14 (same backbone as Stage 0, eval PPL=13.10) | Runtime: ~15 min"
echo "================================================================================"

python scripts/evaluate_lm.py \
    --checkpoint "${STAGE1_CHECKPOINT}" \
    --model-config hybrid_70m \
    --dataset pubmed \
    --split test \
    --batch-size 4 \
    --max-length 512 \
    --num-workers 2 \
    --output-dir "${OUTPUT_DIR}/lm" \
    --device cuda
LM_EXIT=$?

echo ""

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "================================================================================"
echo "=== EVALUATION SUMMARY ==="
echo "================================================================================"
echo "Checkpoint : ${STAGE1_CHECKPOINT}"
echo "Date       : $(date)"
echo ""

if [ -f "${OUTPUT_DIR}/sts/sts_biosses_results.json" ]; then
    python -c "
import json
r = json.load(open('${OUTPUT_DIR}/sts/sts_biosses_results.json'))
print(f'BIOSSES STS  : Spearman r={r.get(\"spearman_correlation\",\"N/A\"):.4f}  (N={r.get(\"num_pairs\",\"?\")})')
"
else
    echo "BIOSSES STS  : FAILED (exit=${BIOSSES_EXIT})"
fi

if [ -f "${OUTPUT_DIR}/sts/sts_stsb_results.json" ]; then
    python -c "
import json
r = json.load(open('${OUTPUT_DIR}/sts/sts_stsb_results.json'))
print(f'STS-B        : Spearman r={r.get(\"spearman_correlation\",\"N/A\"):.4f}  (N={r.get(\"num_pairs\",\"?\")})')
"
else
    echo "STS-B        : FAILED (exit=${STSB_EXIT})"
fi

if [ -f "${OUTPUT_DIR}/retrieval/retrieval_results.json" ]; then
    python -c "
import json
r = json.load(open('${OUTPUT_DIR}/retrieval/retrieval_results.json'))
m = r.get('metrics', {})
print(f'Retrieval    : R@1={m.get(\"R@1\",\"N/A\"):.4f}  R@5={m.get(\"R@5\",\"N/A\"):.4f}  R@10={m.get(\"R@10\",\"N/A\"):.4f}  (N={r.get(\"num_pairs\",\"?\")})')
"
else
    echo "Retrieval    : FAILED (exit=${RETRIEVAL_EXIT})"
fi

if [ -f "${OUTPUT_DIR}/lm/results.json" ]; then
    python -c "
import json
r = json.load(open('${OUTPUT_DIR}/lm/results.json'))
print(f'Perplexity   : PPL={r.get(\"test_perplexity\",r.get(\"perplexity\",\"N/A\")):.2f}  loss={r.get(\"test_loss\",r.get(\"loss\",\"N/A\")):.4f}')
"
else
    echo "Perplexity   : FAILED (exit=${LM_EXIT})"
fi

echo ""
echo "Output dir : ${OUTPUT_DIR}/"
echo ""

OVERALL=0
[ $BIOSSES_EXIT -ne 0 ] && OVERALL=1
[ $RETRIEVAL_EXIT -ne 0 ] && OVERALL=1
[ $LM_EXIT -ne 0 ] && OVERALL=1

echo "=== JOB END (Stage 1 KD Evaluation: exit=${OVERALL}) ==="
date

exit $OVERALL
