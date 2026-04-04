#!/bin/bash

# Individual evaluation scripts for Stage 1 model
# Run these separately if you want to test individual components

set -e

CHECKPOINT="outputs/stage1_pubmed_simcse/checkpoints/last.ckpt"
OUTPUT_DIR="outputs/eval_stage1"
BATCH_SIZE=32

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found at $CHECKPOINT"
    echo "Please update the CHECKPOINT variable in this script"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Stage 1 Individual Evaluations"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT"
echo ""

# Function to run evaluation with error handling
run_eval() {
    local name=$1
    local cmd=$2
    
    echo ""
    echo "=========================================="
    echo "Running: $name"
    echo "=========================================="
    
    if eval "$cmd"; then
        echo "✓ $name completed successfully"
    else
        echo "✗ $name failed"
        return 1
    fi
}

# 1. STS Benchmark (BIOSSES)
run_eval "STS Benchmark (BIOSSES)" \
    "python scripts/evaluate_sts.py \
        --checkpoint $CHECKPOINT \
        --dataset biosses \
        --batch-size $BATCH_SIZE \
        --output-dir $OUTPUT_DIR/sts"

# 2. STS Benchmark (STS-B) - Optional general domain test
run_eval "STS Benchmark (STS-B)" \
    "python scripts/evaluate_sts.py \
        --checkpoint $CHECKPOINT \
        --dataset stsb \
        --batch-size $BATCH_SIZE \
        --output-dir $OUTPUT_DIR/sts"

# 3. Retrieval Evaluation
run_eval "Retrieval Evaluation" \
    "python scripts/evaluate_retrieval.py \
        --checkpoint $CHECKPOINT \
        --num-pairs 1000 \
        --batch-size $BATCH_SIZE \
        --output-dir $OUTPUT_DIR/retrieval"

# 4. Perplexity Evaluation
run_eval "Perplexity Evaluation" \
    "python scripts/evaluate_lm.py \
        --checkpoint $CHECKPOINT \
        --dataset pubmed \
        --split test \
        --batch-size 8 \
        --max-length 256 \
        --output-dir $OUTPUT_DIR/perplexity"

echo ""
echo "=========================================="
echo "All evaluations completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="
