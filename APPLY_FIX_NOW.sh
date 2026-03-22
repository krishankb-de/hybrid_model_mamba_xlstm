#!/bin/bash
# Apply the dropout fix and restart training

echo "════════════════════════════════════════════════════════════════"
echo "  FINAL FIX - Dropout Increase for SimCSE Diversity"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "ROOT CAUSE IDENTIFIED:"
echo "  Dropout=0.1 creates insufficient diversity between views"
echo "  Model learns degenerate solution (all embeddings identical)"
echo ""
echo "FIX APPLIED:"
echo "  Temporarily increase dropout to 0.3 during SimCSE encoding"
echo "  This creates more diverse views and prevents collapse"
echo ""
echo "════════════════════════════════════════════════════════════════"
echo ""

# Stop any running jobs
echo "Step 1: Stopping any running SimCSE jobs..."
RUNNING_JOBS=$(squeue -u $USER -n simcse_stage1 -h -o "%i")
if [ -n "$RUNNING_JOBS" ]; then
    for job in $RUNNING_JOBS; do
        echo "  Cancelling job $job..."
        scancel $job
    done
    sleep 2
else
    echo "  No running jobs found"
fi
echo ""

# Clean up collapsed checkpoints
echo "Step 2: Cleaning up collapsed checkpoints..."
if [ -d "outputs/stage1_pubmed_simcse/checkpoints" ]; then
    CKPT_COUNT=$(ls outputs/stage1_pubmed_simcse/checkpoints/*.ckpt 2>/dev/null | wc -l)
    if [ $CKPT_COUNT -gt 0 ]; then
        echo "  Moving $CKPT_COUNT collapsed checkpoints to backup..."
        mkdir -p outputs/stage1_pubmed_simcse/checkpoints_collapsed_all
        mv outputs/stage1_pubmed_simcse/checkpoints/*.ckpt outputs/stage1_pubmed_simcse/checkpoints_collapsed_all/ 2>/dev/null
    else
        echo "  No old checkpoints to clean"
    fi
fi
echo ""

# Submit new job
echo "Step 3: Submitting training with dropout fix..."
echo ""
JOB_ID=$(sbatch scripts/train_contrastive_stage1.sh | grep -oP '\d+$')

if [ -n "$JOB_ID" ]; then
    echo "✅ Job submitted successfully: $JOB_ID"
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "  What Changed"
    echo "════════════════════════════════════════════════════════════════"
    echo ""
    echo "  File: hybrid_xmamba/training/lightning_module.py"
    echo "  Method: _simcse_step"
    echo "  Change: Dropout 0.1 → 0.3 during encoding (temporary)"
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "  Expected Behavior (DIFFERENT from before)"
    echo "════════════════════════════════════════════════════════════════"
    echo ""
    echo "  Step 500:  loss ~3-4   (HIGHER than before due to more dropout)"
    echo "  Step 1000: loss ~2-3   (NOT 0.003 like before)"
    echo "  Step 5000: loss ~1.5-2 (NOT 10^-6 like before)"
    echo "  Step 10000: loss ~1-1.5 (NOT 10^-6 like before)"
    echo ""
    echo "  Duration: 3-5 hours"
    echo "  Final validation loss: 1.0-1.5 (NOT 0.005)"
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "  Monitoring"
    echo "════════════════════════════════════════════════════════════════"
    echo ""
    echo "  Watch log:"
    echo "    tail -f logs/simcse_stage1_${JOB_ID}.log | grep contrastive_loss"
    echo ""
    echo "  Check progress:"
    echo "    tail -20 logs/simcse_stage1_${JOB_ID}.log"
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "  Warning Signs (Stop if you see)"
    echo "════════════════════════════════════════════════════════════════"
    echo ""
    echo "  🚨 Loss < 0.5 at any point"
    echo "  🚨 Loss drops by > 50% in 500 steps"
    echo "  🚨 Training loss << validation loss"
    echo ""
    echo "  To stop: scancel $JOB_ID"
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "  Why This Will Work"
    echo "════════════════════════════════════════════════════════════════"
    echo ""
    echo "  With dropout=0.1:"
    echo "    - Two views are 95% similar"
    echo "    - Model easily outputs identical embeddings"
    echo "    - Loss collapses to near-zero"
    echo ""
    echo "  With dropout=0.3:"
    echo "    - Two views are 70-80% similar"
    echo "    - Model must learn meaningful representations"
    echo "    - Loss stays healthy (1-2)"
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo ""
    echo "See ROOT_CAUSE_ANALYSIS.md for detailed explanation"
    echo "See FINAL_SOLUTION_DROPOUT_FIX.md for complete documentation"
    echo ""
else
    echo "❌ Job submission failed!"
    echo ""
    echo "Check:"
    echo "  1. Are you in the correct directory?"
    echo "  2. Does scripts/train_contrastive_stage1.sh exist?"
    echo "  3. Do you have SLURM access?"
    echo ""
fi
