#!/bin/bash
#SBATCH --partition=mitarb
#SBATCH --account=mitarb
#SBATCH --gres=gpu:mitarb:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=eval_stage1_complete
#SBATCH --output=logs/eval_stage1_%j.log
#SBATCH --error=logs/eval_stage1_%j.log

set -euo pipefail

echo "================================================================================"
echo "=== JOB START: Stage 1 Complete Evaluation ==="
echo "================================================================================"
date
echo ""
echo "Host: $(hostname)"
echo "Submit dir: ${SLURM_SUBMIT_DIR}"
echo "Job ID: ${SLURM_JOB_ID}"
echo ""

# Navigate to project directory
cd "${SLURM_SUBMIT_DIR}"

# Configuration
CHECKPOINT="outputs/stage1_pubmed_simcse/checkpoints/contrastive-step=008721-val/contrastive_loss=0.0110.ckpt"
OUTPUT_DIR="outputs/eval_stage1"
BATCH_SIZE=32
MAX_LENGTH=256
NUM_RETRIEVAL_PAIRS=1000

echo "Configuration:"
echo "  - Checkpoint: $CHECKPOINT"
echo "  - Output directory: $OUTPUT_DIR"
echo "  - Batch size: $BATCH_SIZE"
echo "  - Max length: $MAX_LENGTH"
echo "  - Retrieval pairs: $NUM_RETRIEVAL_PAIRS"
echo "  - Expected runtime: 30-45 minutes"
echo ""

# Environment setup
export HF_HOME="$PWD/.hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCHINDUCTOR_CACHE_DIR="$PWD/.torchinductor"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"
export CUDA_LAUNCH_BLOCKING=0

echo "Environment:"
echo "  - HF_HOME: $HF_HOME"
echo "  - HF_DATASETS_CACHE: $HF_DATASETS_CACHE"
echo "  - CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
echo ""

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found at $CHECKPOINT"
    echo "Available checkpoints:"
    ls -lh outputs/stage1_pubmed_simcse/checkpoints/*.ckpt 2>/dev/null || echo "  No checkpoints found"
    exit 1
fi

echo "✓ Checkpoint found: $CHECKPOINT"
echo "  Size: $(du -h "$CHECKPOINT" | cut -f1)"
echo ""

# Create output directories
mkdir -p logs
mkdir -p "$OUTPUT_DIR/sts"
mkdir -p "$OUTPUT_DIR/retrieval"
mkdir -p "$OUTPUT_DIR/lm"

# Activate virtual environment
if [ ! -d ".venv" ]; then
    echo "ERROR: Virtual environment .venv not found!"
    echo "Please create it first with: python -m venv .venv"
    exit 1
fi

echo "Activating virtual environment..."
source .venv/bin/activate

# Verify Python and PyTorch
echo ""
echo "Verifying Python environment..."
python -c "
import sys
import torch
print(f'Python version: {sys.version}')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'Device count: {torch.cuda.device_count()}')
    print(f'Device name: {torch.cuda.get_device_name(0)}')
    print(f'Device memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

if [ $? -ne 0 ]; then
    echo "ERROR: Python environment verification failed!"
    exit 1
fi

echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader

echo ""
echo "================================================================================"
echo "=== Step 1: Diagnostic Check ==="
echo "================================================================================"
echo ""

python diagnose_eval_issue.py --checkpoint "$CHECKPOINT"

if [ $? -ne 0 ]; then
    echo ""
    echo "WARNING: Diagnostic check failed or found issues"
    echo "Continuing with evaluation, but results may be invalid"
    echo ""
fi

echo ""
echo "================================================================================"
echo "=== Step 2: STS Evaluation (BIOSSES) ==="
echo "================================================================================"
echo ""
echo "Evaluating semantic textual similarity on BIOSSES dataset..."
echo "Expected time: 5-10 minutes"
echo ""

python scripts/evaluate_sts.py \
    --checkpoint "$CHECKPOINT" \
    --dataset biosses \
    --batch-size $BATCH_SIZE \
    --max-length $MAX_LENGTH \
    --output-dir "$OUTPUT_DIR/sts" \
    --device cuda

STS_EXIT_CODE=$?

if [ $STS_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✓ STS evaluation completed successfully"
    if [ -f "$OUTPUT_DIR/sts/sts_biosses_results.json" ]; then
        echo "Results:"
        python -c "
import json
with open('$OUTPUT_DIR/sts/sts_biosses_results.json') as f:
    results = json.load(f)
    print(f\"  Spearman Correlation: {results.get('spearman_correlation', 'N/A'):.4f}\")
    print(f\"  P-value: {results.get('p_value', 'N/A'):.6f}\")
    print(f\"  Number of pairs: {results.get('num_pairs', 'N/A')}\")
"
    fi
else
    echo ""
    echo "✗ STS evaluation failed with exit code $STS_EXIT_CODE"
fi

echo ""
echo "================================================================================"
echo "=== Step 3: Retrieval Evaluation (PubMed) ==="
echo "================================================================================"
echo ""
echo "Evaluating text retrieval on PubMed abstracts..."
echo "Expected time: 10-15 minutes"
echo ""

python scripts/evaluate_retrieval.py \
    --checkpoint "$CHECKPOINT" \
    --num-pairs $NUM_RETRIEVAL_PAIRS \
    --batch-size $BATCH_SIZE \
    --max-length $MAX_LENGTH \
    --output-dir "$OUTPUT_DIR/retrieval" \
    --device cuda

RETRIEVAL_EXIT_CODE=$?

if [ $RETRIEVAL_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✓ Retrieval evaluation completed successfully"
    if [ -f "$OUTPUT_DIR/retrieval/retrieval_results.json" ]; then
        echo "Results:"
        python -c "
import json
with open('$OUTPUT_DIR/retrieval/retrieval_results.json') as f:
    results = json.load(f)
    metrics = results.get('metrics', {})
    print(f\"  Recall@1:  {metrics.get('R@1', 'N/A'):.4f}\")
    print(f\"  Recall@5:  {metrics.get('R@5', 'N/A'):.4f}\")
    print(f\"  Recall@10: {metrics.get('R@10', 'N/A'):.4f}\")
    print(f\"  Number of pairs: {results.get('num_pairs', 'N/A')}\")
"
    fi
else
    echo ""
    echo "✗ Retrieval evaluation failed with exit code $RETRIEVAL_EXIT_CODE"
fi

echo ""
echo "================================================================================"
echo "=== Step 4: Perplexity Evaluation (Optional) ==="
echo "================================================================================"
echo ""
echo "Evaluating language modeling perplexity on PubMed..."
echo "Expected time: 15-20 minutes"
echo ""

python scripts/evaluate_lm.py \
    --checkpoint "$CHECKPOINT" \
    --dataset pubmed \
    --split validation \
    --batch-size 4 \
    --max-length $MAX_LENGTH \
    --output-dir "$OUTPUT_DIR/lm" \
    --device cuda

LM_EXIT_CODE=$?

if [ $LM_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✓ Perplexity evaluation completed successfully"
    if [ -f "$OUTPUT_DIR/lm/results.json" ]; then
        echo "Results:"
        python -c "
import json
with open('$OUTPUT_DIR/lm/results.json') as f:
    results = json.load(f)
    print(f\"  Perplexity: {results.get('test_perplexity', 'N/A'):.2f}\")
    print(f\"  Loss: {results.get('test_loss', 'N/A'):.4f}\")
    print(f\"  Tokens evaluated: {results.get('num_tokens_evaluated', 'N/A'):,}\")
"
    fi
else
    echo ""
    echo "✗ Perplexity evaluation failed with exit code $LM_EXIT_CODE"
fi

echo ""
echo "================================================================================"
echo "=== Evaluation Summary ==="
echo "================================================================================"
echo ""

# Create summary file
SUMMARY_FILE="$OUTPUT_DIR/EVALUATION_SUMMARY.txt"

cat > "$SUMMARY_FILE" << EOF
Stage 1 Evaluation Summary
==========================

Date: $(date)
Checkpoint: $CHECKPOINT
Job ID: ${SLURM_JOB_ID}

Configuration:
  - Batch size: $BATCH_SIZE
  - Max length: $MAX_LENGTH
  - Retrieval pairs: $NUM_RETRIEVAL_PAIRS

Results:
--------

EOF

# Add STS results
if [ $STS_EXIT_CODE -eq 0 ] && [ -f "$OUTPUT_DIR/sts/sts_biosses_results.json" ]; then
    echo "STS (BIOSSES):" >> "$SUMMARY_FILE"
    python -c "
import json
with open('$OUTPUT_DIR/sts/sts_biosses_results.json') as f:
    results = json.load(f)
    print(f\"  Spearman Correlation: {results.get('spearman_correlation', 'N/A'):.4f}\")
    print(f\"  P-value: {results.get('p_value', 'N/A'):.6f}\")
    print(f\"  Status: {'PASS' if results.get('spearman_correlation', 0) > 0.0 else 'FAIL'}\")
" >> "$SUMMARY_FILE"
else
    echo "STS (BIOSSES): FAILED" >> "$SUMMARY_FILE"
fi

echo "" >> "$SUMMARY_FILE"

# Add Retrieval results
if [ $RETRIEVAL_EXIT_CODE -eq 0 ] && [ -f "$OUTPUT_DIR/retrieval/retrieval_results.json" ]; then
    echo "Retrieval (PubMed):" >> "$SUMMARY_FILE"
    python -c "
import json
with open('$OUTPUT_DIR/retrieval/retrieval_results.json') as f:
    results = json.load(f)
    metrics = results.get('metrics', {})
    print(f\"  Recall@1:  {metrics.get('R@1', 'N/A'):.4f}\")
    print(f\"  Recall@5:  {metrics.get('R@5', 'N/A'):.4f}\")
    print(f\"  Recall@10: {metrics.get('R@10', 'N/A'):.4f}\")
    print(f\"  Status: {'PASS' if metrics.get('R@1', 0) > 0.001 else 'FAIL'}\")
" >> "$SUMMARY_FILE"
else
    echo "Retrieval (PubMed): FAILED" >> "$SUMMARY_FILE"
fi

echo "" >> "$SUMMARY_FILE"

# Add Perplexity results
if [ $LM_EXIT_CODE -eq 0 ] && [ -f "$OUTPUT_DIR/lm/results.json" ]; then
    echo "Perplexity (PubMed):" >> "$SUMMARY_FILE"
    python -c "
import json
with open('$OUTPUT_DIR/lm/results.json') as f:
    results = json.load(f)
    print(f\"  Perplexity: {results.get('test_perplexity', 'N/A'):.2f}\")
    print(f\"  Loss: {results.get('test_loss', 'N/A'):.4f}\")
    print(f\"  Status: PASS\")
" >> "$SUMMARY_FILE"
else
    echo "Perplexity (PubMed): FAILED" >> "$SUMMARY_FILE"
fi

echo "" >> "$SUMMARY_FILE"
echo "Exit Codes:" >> "$SUMMARY_FILE"
echo "  - STS: $STS_EXIT_CODE" >> "$SUMMARY_FILE"
echo "  - Retrieval: $RETRIEVAL_EXIT_CODE" >> "$SUMMARY_FILE"
echo "  - Perplexity: $LM_EXIT_CODE" >> "$SUMMARY_FILE"

# Display summary
cat "$SUMMARY_FILE"

echo ""
echo "Summary saved to: $SUMMARY_FILE"
echo ""

# Overall status
OVERALL_EXIT_CODE=0
if [ $STS_EXIT_CODE -ne 0 ] || [ $RETRIEVAL_EXIT_CODE -ne 0 ]; then
    OVERALL_EXIT_CODE=1
    echo "⚠️  Some evaluations failed. Check logs for details."
else
    echo "✓ All critical evaluations completed successfully!"
fi

echo ""
echo "Output files:"
echo "  - STS results: $OUTPUT_DIR/sts/sts_biosses_results.json"
echo "  - Retrieval results: $OUTPUT_DIR/retrieval/retrieval_results.json"
echo "  - Perplexity results: $OUTPUT_DIR/lm/results.json"
echo "  - Summary: $SUMMARY_FILE"
echo ""

echo "================================================================================"
echo "=== JOB END: Stage 1 Complete Evaluation ==="
echo "================================================================================"
date

exit $OVERALL_EXIT_CODE
