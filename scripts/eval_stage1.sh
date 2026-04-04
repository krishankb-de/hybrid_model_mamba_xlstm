#!/bin/bash
#SBATCH --job-name=eval_stage1
#SBATCH --output=logs/eval_stage1_%j.out
#SBATCH --error=logs/eval_stage1_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Comprehensive Stage 1 Model Evaluation
# Runs STS, Retrieval, and Perplexity evaluations

set -e  # Exit on error

echo "=========================================="
echo "Stage 1 Model Evaluation"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

# Configuration
CHECKPOINT="outputs/stage1_pubmed_simcse/checkpoints/last.ckpt"
OUTPUT_DIR="outputs/eval_stage1"
BATCH_SIZE=32
NUM_RETRIEVAL_PAIRS=1000

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found at $CHECKPOINT"
    echo "Please update the CHECKPOINT variable in this script"
    exit 1
fi

# Create logs directory
mkdir -p logs

# Activate environment (adjust as needed)
# source /path/to/your/venv/bin/activate
# OR
# conda activate your_env

# Install required packages if not already installed
echo "Installing evaluation dependencies..."
pip install -q sentence-transformers scipy datasets

# Run comprehensive evaluation
echo ""
echo "Running comprehensive evaluation..."
echo "Checkpoint: $CHECKPOINT"
echo "Output directory: $OUTPUT_DIR"
echo ""

python scripts/evaluate_stage1_full.py \
    --checkpoint "$CHECKPOINT" \
    --output-dir "$OUTPUT_DIR" \
    --batch-size $BATCH_SIZE \
    --num-retrieval-pairs $NUM_RETRIEVAL_PAIRS

echo ""
echo "=========================================="
echo "Evaluation completed!"
echo "End time: $(date)"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="
