#!/bin/bash
# Quick restart script for contrastive training

echo "════════════════════════════════════════════════════════"
echo "  Restarting Contrastive Training (Attempt 3)"
echo "════════════════════════════════════════════════════════"
echo ""

# Stop current job if running
echo "Step 1: Stopping any running jobs..."
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

# Clean up failed checkpoints
echo "Step 2: Cleaning up failed checkpoints..."
if [ -d "outputs/stage1_pubmed_simcse/checkpoints" ]; then
    CKPT_COUNT=$(ls outputs/stage1_pubmed_simcse/checkpoints/*.ckpt 2>/dev/null | wc -l)
    if [ $CKPT_COUNT -gt 0 ]; then
        echo "  Moving $CKPT_COUNT old checkpoints to backup..."
        mkdir -p outputs/stage1_pubmed_simcse/checkpoints_failed
        mv outputs/stage1_pubmed_simcse/checkpoints/*.ckpt outputs/stage1_pubmed_simcse/checkpoints_failed/ 2>/dev/null
    else
        echo "  No old checkpoints found"
    fi
else
    echo "  Checkpoint directory doesn't exist yet"
fi
echo ""

# Submit new job
echo "Step 3: Submitting new job with fixed settings..."
echo ""
echo "  Changes from previous attempts:"
echo "    - Learning rate: 0.0003 → 0.0001 (67% reduction)"
echo "    - Batch size: 4 → 8 (more stable)"
echo "    - Early stopping: monitors train loss (catches collapse)"
echo ""

JOB_ID=$(sbatch scripts/train_contrastive_stage1.sh | grep -oP '\d+$')

if [ -n "$JOB_ID" ]; then
    echo "✅ Job submitted successfully: $JOB_ID"
    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  Monitoring Instructions"
    echo "════════════════════════════════════════════════════════"
    echo ""
    echo "Option 1: Automated monitoring"
    echo "  bash monitor_training.sh"
    echo ""
    echo "Option 2: Manual checks"
    echo "  tail -f logs/simcse_stage1_${JOB_ID}.log"
    echo ""
    echo "Option 3: Check loss every 30 min"
    echo "  tail -20 logs/simcse_stage1_${JOB_ID}.log | grep contrastive_loss"
    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  Expected Behavior"
    echo "════════════════════════════════════════════════════════"
    echo ""
    echo "  Duration: 3-5 hours"
    echo "  Steps: ~10,000"
    echo "  Final loss: 1.0-2.0"
    echo ""
    echo "  Loss trajectory:"
    echo "    Step 500:  ~4-5"
    echo "    Step 1000: ~3-4"
    echo "    Step 5000: ~1.5-2.5"
    echo "    Step 10000: ~1.0-2.0"
    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  Warning Signs (Stop if you see)"
    echo "════════════════════════════════════════════════════════"
    echo ""
    echo "  🚨 Training loss < 0.5"
    echo "  🚨 Step count > 11,000"
    echo "  🚨 Loss drops by 50% in 500 steps"
    echo ""
    echo "  To stop: scancel $JOB_ID"
    echo ""
    echo "════════════════════════════════════════════════════════"
else
    echo "❌ Job submission failed!"
    echo ""
    echo "Check:"
    echo "  1. Are you in the correct directory?"
    echo "  2. Does scripts/train_contrastive_stage1.sh exist?"
    echo "  3. Do you have SLURM access?"
    echo ""
fi
