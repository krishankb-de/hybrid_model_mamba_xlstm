#!/bin/bash
# Quick fix script for evaluation errors
# Run this on Willi server

echo "=========================================="
echo "Fixing and Rerunning Stage 1 Evaluation"
echo "=========================================="
echo ""

# Navigate to project
cd hybrid_xmamba_a100_70m_40/hybrid_model_mamba_xlstm || {
    echo "ERROR: Could not find project directory"
    exit 1
}

echo "Step 1: Stashing local changes..."
git stash

echo ""
echo "Step 2: Pulling latest fixes..."
git pull origin a100_70m_baseline

echo ""
echo "Step 3: Verifying checkpoint exists..."
if [ -f "outputs/stage1_pubmed_simcse/checkpoints/last.ckpt" ]; then
    echo "✓ Checkpoint found"
    ls -lh outputs/stage1_pubmed_simcse/checkpoints/last.ckpt
else
    echo "✗ Checkpoint not found!"
    echo "Please check the checkpoint path"
    exit 1
fi

echo ""
echo "Step 4: Submitting evaluation job..."
sbatch scripts/eval_stage1.sh

echo ""
echo "=========================================="
echo "Job submitted! Monitor with:"
echo "  tail -f logs/eval_stage1_*.log"
echo "=========================================="
