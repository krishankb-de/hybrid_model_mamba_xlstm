#!/bin/bash
# Monitor contrastive training for collapse

LOG_FILE=$(ls -t logs/simcse_stage1_*.log 2>/dev/null | head -1)

if [ -z "$LOG_FILE" ]; then
    echo "No training log found!"
    exit 1
fi

echo "Monitoring: $LOG_FILE"
echo "Press Ctrl+C to stop monitoring"
echo ""
echo "=== Training Health Check ==="
echo ""

# Function to check if training is healthy
check_health() {
    # Get last 20 training losses
    RECENT_LOSSES=$(grep "train/contrastive_loss_step=" "$LOG_FILE" | tail -20 | grep -oP 'train/contrastive_loss_step=\K[0-9.e-]+')
    
    # Get last validation loss
    LAST_VAL_LOSS=$(grep "val/contrastive_loss=" "$LOG_FILE" | tail -1 | grep -oP 'val/contrastive_loss=\K[0-9.]+')
    
    # Get current step
    CURRENT_STEP=$(grep "Epoch 0:" "$LOG_FILE" | tail -1 | grep -oP '\|\s+\K[0-9]+' | head -1)
    
    echo "Current step: ${CURRENT_STEP:-unknown}"
    echo "Last validation loss: ${LAST_VAL_LOSS:-unknown}"
    echo ""
    echo "Recent training losses:"
    echo "$RECENT_LOSSES" | tail -10
    echo ""
    
    # Check for collapse
    if [ -n "$LAST_VAL_LOSS" ]; then
        if (( $(echo "$LAST_VAL_LOSS < 0.3" | bc -l) )); then
            echo "💀 CRITICAL: Validation loss < 0.3 - COMPLETE COLLAPSE!"
            echo "    ACTION: Stop job immediately with: scancel <job_id>"
        elif (( $(echo "$LAST_VAL_LOSS < 0.5" | bc -l) )); then
            echo "🚨 WARNING: Validation loss < 0.5 - Model is collapsing!"
            echo "    ACTION: Monitor closely, consider stopping"
        elif (( $(echo "$LAST_VAL_LOSS < 1.0" | bc -l) )); then
            echo "⚠️  CAUTION: Validation loss < 1.0 - Monitor closely"
        else
            echo "✅ Validation loss looks healthy (> 1.0)"
        fi
    fi
    
    # Check training loss for collapse
    LAST_TRAIN_LOSS=$(echo "$RECENT_LOSSES" | tail -1)
    if [ -n "$LAST_TRAIN_LOSS" ]; then
        if (( $(echo "$LAST_TRAIN_LOSS < 0.1" | bc -l) )); then
            echo "💀 CRITICAL: Training loss < 0.1 - COLLAPSE DETECTED!"
            echo "    ACTION: Stop job NOW with: scancel <job_id>"
        elif (( $(echo "$LAST_TRAIN_LOSS < 0.5" | bc -l) )); then
            echo "🚨 WARNING: Training loss < 0.5 - Collapse starting!"
            echo "    ACTION: Consider stopping job"
        fi
    fi
    
    # Check step count
    if [ -n "$CURRENT_STEP" ]; then
        if [ "$CURRENT_STEP" -gt 12000 ]; then
            echo "🚨 WARNING: Step count > 12,000 - Should have stopped at 10,000!"
        elif [ "$CURRENT_STEP" -gt 10000 ]; then
            echo "⚠️  CAUTION: Step count > 10,000 - Should be finishing soon"
        else
            REMAINING=$((10000 - CURRENT_STEP))
            echo "✅ Step count OK ($REMAINING steps remaining)"
        fi
    fi
    
    echo ""
    echo "---"
    echo ""
}

# Initial check
check_health

# Watch for updates every 60 seconds
while true; do
    sleep 60
    check_health
done
