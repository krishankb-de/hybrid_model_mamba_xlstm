#!/bin/bash
#SBATCH --partition=mitarb
#SBATCH --account=mitarb
#SBATCH --gres=gpu:mitarb:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=eval_stage1_individual
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log

# Individual Stage 1 Evaluation Script
# Run specific evaluation tasks independently
#
# Usage:
#   sbatch --export=EVAL_TYPE=sts scripts/eval_stage1_individual.sh
#   sbatch --export=EVAL_TYPE=retrieval scripts/eval_stage1_individual.sh
#   sbatch --export=EVAL_TYPE=perplexity scripts/eval_stage1_individual.sh

set -euo pipefail

echo "=== JOB START (Stage 1 Individual Evaluation) ==="
date
echo "Host: $(hostname)"
echo "Evaluation Type: ${EVAL_TYPE:-all}"
echo ""

cd "${SLURM_SUBMIT_DIR}/hybrid_model_mamba_xlstm"

export HF_HOME="$PWD/.hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCHINDUCTOR_CACHE_DIR="$PWD/.torchinductor"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"
export CUDA_LAUNCH_BLOCKING=0

# Configuration
CHECKPOINT="outputs/stage1_pubmed_simcse/checkpoints/last.ckpt"
OUTPUT_DIR="outputs/eval_stage1"
BATCH_SIZE=32

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found at $CHECKPOINT"
    exit 1
fi

# Activate virtual environment
if [ ! -d ".venv" ]; then
    echo "ERROR: Virtual environment .venv not found!"
    exit 1
fi

source .venv/bin/activate

# Verify CUDA
echo "Verifying CUDA availability:"
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Device count: {torch.cuda.device_count()}')
"
echo ""

# Install dependencies
echo "Installing evaluation dependencies..."
pip install -q sentence-transformers scipy datasets

nvidia-smi

# Run evaluation based on EVAL_TYPE
case "${EVAL_TYPE:-all}" in
    sts)
        echo ""
        echo "=== Running STS Evaluation ==="
        python scripts/evaluate_sts.py \
            --checkpoint "$CHECKPOINT" \
            --dataset biosses \
            --batch-size $BATCH_SIZE \
            --output-dir "$OUTPUT_DIR/sts"
        ;;
    
    retrieval)
        echo ""
        echo "=== Running Retrieval Evaluation ==="
        python scripts/evaluate_retrieval.py \
            --checkpoint "$CHECKPOINT" \
            --num-pairs 1000 \
            --batch-size $BATCH_SIZE \
            --output-dir "$OUTPUT_DIR/retrieval"
        ;;
    
    perplexity)
        echo ""
        echo "=== Running Perplexity Evaluation ==="
        python scripts/evaluate_lm.py \
            --checkpoint "$CHECKPOINT" \
            --dataset pubmed \
            --split test \
            --batch-size 8 \
            --max-length 256 \
            --output-dir "$OUTPUT_DIR/perplexity"
        ;;
    
    all)
        echo ""
        echo "=== Running All Evaluations ==="
        
        echo ""
        echo "1. STS Evaluation..."
        python scripts/evaluate_sts.py \
            --checkpoint "$CHECKPOINT" \
            --dataset biosses \
            --batch-size $BATCH_SIZE \
            --output-dir "$OUTPUT_DIR/sts"
        
        echo ""
        echo "2. Retrieval Evaluation..."
        python scripts/evaluate_retrieval.py \
            --checkpoint "$CHECKPOINT" \
            --num-pairs 1000 \
            --batch-size $BATCH_SIZE \
            --output-dir "$OUTPUT_DIR/retrieval"
        
        echo ""
        echo "3. Perplexity Evaluation..."
        python scripts/evaluate_lm.py \
            --checkpoint "$CHECKPOINT" \
            --dataset pubmed \
            --split test \
            --batch-size 8 \
            --max-length 256 \
            --output-dir "$OUTPUT_DIR/perplexity"
        ;;
    
    *)
        echo "ERROR: Unknown EVAL_TYPE: ${EVAL_TYPE}"
        echo "Valid options: sts, retrieval, perplexity, all"
        exit 1
        ;;
esac

echo ""
echo "=== JOB END (Stage 1 Individual Evaluation) ==="
echo "Results saved to: $OUTPUT_DIR"
date
