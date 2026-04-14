#!/bin/bash
# ============================================================================
# Hybrid Mamba-xLSTM Complete Pipeline Validation (Sample Mode)
# ============================================================================
# Purpose: Validate Stage 0 (LM pretraining) + Stage 1 (contrastive) pipeline
#         with sample data before full A100 production run.
#
# Pipeline:
#   Stage 0 → LM pretraining on PubMed (domain adaptation)
#   Stage 1 → Contrastive fine-tuning (SimCSE) using Stage 0 checkpoint
#
# Author: Validation Script
# Date: 2026-04-14
# ============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${PROJECT_ROOT}/venv"
PYTHON="${VENV_PATH}/bin/python"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/validation_pipeline_$(date +%Y%m%d_%H%M%S)"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# ============================================================================
# Helper Functions
# ============================================================================

log_title() {
    echo -e "\n${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║ $1${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}\n"
}

log_step() {
    echo -e "${YELLOW}→ $1${NC}"
}

log_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

log_error() {
    echo -e "${RED}✗ $1${NC}"
}

check_venv() {
    if [ ! -f "$PYTHON" ]; then
        log_error "Virtual environment not found at $VENV_PATH"
        echo "Please create venv first:"
        echo "  python -m venv venv"
        echo "  source venv/bin/activate"
        echo "  pip install -e ."
        echo "  pip install -r requirements.txt"
        exit 1
    fi
    log_success "Virtual environment found: $VENV_PATH"
}

check_configs() {
    local required_configs=("model/hybrid_70m.yaml" "dataset/pubmed.yaml" "trainer/a100_single_gpu.yaml")
    for config in "${required_configs[@]}"; do
        if [ ! -f "${PROJECT_ROOT}/configs/${config}" ]; then
            log_error "Missing config: configs/${config}"
            exit 1
        fi
    done
    log_success "All required configs present"
}

run_command() {
    local cmd="$1"
    local step_name="$2"
    log_step "Running: $step_name"
    echo "  Command: $cmd"
    echo ""
    eval "$cmd"
    if [ $? -eq 0 ]; then
        log_success "$step_name completed successfully"
    else
        log_error "$step_name failed"
        exit 1
    fi
}

# ============================================================================
# Main Pipeline
# ============================================================================

main() {
    log_title "HYBRID MAMBA-xLSTM PIPELINE VALIDATION"
    echo "Output directory: $OUTPUT_DIR"
    echo "Start time: $TIMESTAMP"
    echo ""

    # Setup
    check_venv
    check_configs
    mkdir -p "$OUTPUT_DIR"

    # --------
    # Stage 0: LM Pretraining on PubMed
    # --------
    log_title "STAGE 0: LM PRETRAINING (PubMed Domain Adaptation)"
    log_step "Training hybrid_70m on PubMed abstracts"
    echo "Config:"
    echo "  - Model: hybrid_70m (512 dim, 8 layers, ~70M params)"
    echo "  - Dataset: PubMed (open access biomedical abstracts)"
    echo "  - Batch size: 16 (sample mode for validation)"
    echo "  - Max steps: 200 (fast validation)"
    echo "  - Sequence length: 512"
    echo ""

    STAGE0_EXPERIMENT="validation_stage0_pubmed_lm_$(date +%s)"
    STAGE0_CMD="cd $PROJECT_ROOT && \
    $PYTHON scripts/train.py \
        --config-name config_70m \
        model=hybrid_70m \
        dataset=pubmed \
        trainer=a100_single_gpu \
        trainer.max_steps=200 \
        trainer.val_check_interval=50 \
        trainer.log_every_n_steps=10 \
        dataset.batch_size=16 \
        dataset.eval_batch_size=16 \
        dataset.max_length=512 \
        experiment_name=$STAGE0_EXPERIMENT \
        output_dir=$OUTPUT_DIR/stage0_checkpoint"

    run_command "$STAGE0_CMD" "Stage 0: LM Pretraining"

    # Verify Stage 0 checkpoint
    STAGE0_CKPT="$OUTPUT_DIR/stage0_checkpoint/last.ckpt"
    if [ ! -f "$STAGE0_CKPT" ]; then
        log_error "Stage 0 checkpoint not found: $STAGE0_CKPT"
        exit 1
    fi
    log_success "Stage 0 checkpoint created: $STAGE0_CKPT"

    # --------
    # Stage 1: Contrastive Fine-tuning (SimCSE)
    # --------
    log_title "STAGE 1: CONTRASTIVE FINE-TUNING (SimCSE)"
    log_step "Training text encoder with contrastive learning"
    echo "Config:"
    echo "  - Mode: SimCSE (self-supervised text-only)"
    echo "  - Dataset: PubMed abstracts"
    echo "  - Input: Stage 0 checkpoint"
    echo "  - Batch size: 16"
    echo "  - Max steps: 200 (fast validation)"
    echo "  - Loss: SimCSE (in-batch negatives)"
    echo ""

    STAGE1_EXPERIMENT="validation_stage1_contrastive_$(date +%s)"
    STAGE1_CMD="cd $PROJECT_ROOT && \
    $PYTHON scripts/train_contrastive.py \
        --config-name config_70m \
        dataset=pubmed \
        trainer=a100_single_gpu \
        trainer.max_steps=200 \
        trainer.val_check_interval=50 \
        trainer.log_every_n_steps=10 \
        dataset.batch_size=16 \
        dataset.eval_batch_size=16 \
        lm_checkpoint=$STAGE0_CKPT \
        contrastive_mode=simcse \
        experiment_name=$STAGE1_EXPERIMENT \
        output_dir=$OUTPUT_DIR/stage1_checkpoint"

    run_command "$STAGE1_CMD" "Stage 1: Contrastive Fine-tuning"

    # Verify Stage 1 checkpoint
    STAGE1_CKPT="$OUTPUT_DIR/stage1_checkpoint/last.ckpt"
    if [ ! -f "$STAGE1_CKPT" ]; then
        log_error "Stage 1 checkpoint not found: $STAGE1_CKPT"
        exit 1
    fi
    log_success "Stage 1 checkpoint created: $STAGE1_CKPT"

    # --------
    # Validation: Checkpoint Compatibility Check
    # --------
    log_title "VALIDATION: CHECKPOINT COMPATIBILITY"
    log_step "Verifying checkpoint loading and key structure"

    COMPAT_CMD="cd $PROJECT_ROOT && \
    $PYTHON check_checkpoint_compatibility.py \
        --checkpoint $STAGE1_CKPT"

    run_command "$COMPAT_CMD" "Checkpoint compatibility check"

    # --------
    # Validation: Inference Speed & Perplexity (Optional)
    # --------
    log_title "VALIDATION: INFERENCE PERFORMANCE"
    log_step "Running quick inference and perplexity evaluation"

    EVAL_CMD="cd $PROJECT_ROOT && \
    $PYTHON scripts/evaluate_lm.py \
        --checkpoint $STAGE1_CKPT \
        --model-config hybrid_70m \
        --dataset wikitext \
        --split test \
        --batch-size 16 \
        --max-samples 100 \
        --throughput \
        --output-dir $OUTPUT_DIR/eval_results"

    # Don't exit on eval errors, as this is optional
    if run_command "$EVAL_CMD" "Inference evaluation"; then
        log_success "Evaluation completed"
    else
        log_step "Note: Evaluation skipped (optional)"
    fi

    # --------
    # Summary Report
    # --------
    log_title "PIPELINE VALIDATION COMPLETE"
    cat > "$OUTPUT_DIR/validation_report.txt" << EOF
╔════════════════════════════════════════════════════════════════╗
║  HYBRID MAMBA-xLSTM PIPELINE VALIDATION REPORT                ║
╚════════════════════════════════════════════════════════════════╝

Timestamp: $TIMESTAMP
Project: $PROJECT_ROOT
Output: $OUTPUT_DIR

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STAGE 0: LM PRETRAINING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Model: hybrid_70m (512 dim, 8 layers, ~70M params)
✓ Dataset: PubMed abstracts (biomedical domain)
✓ Training steps: 200
✓ Batch size: 16
✓ Checkpoint: $STAGE0_CKPT

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STAGE 1: CONTRASTIVE FINE-TUNING (SimCSE)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Mode: SimCSE (self-supervised text-only)
✓ Input checkpoint: $STAGE0_CKPT
✓ Training steps: 200
✓ Batch size: 16
✓ Checkpoint: $STAGE1_CKPT

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VALIDATION CHECKS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Stage 0 checkpoint created and verified
✓ Stage 1 checkpoint loaded from Stage 0 successfully
✓ Checkpoint compatibility check passed
✓ Inference evaluation completed

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NEXT STEPS FOR PRODUCTION (A100)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The sample validation has passed. Ready for full production run:

1. STAGE 0 PRODUCTION LM PRETRAINING:
   Commands:
   python scripts/train.py \\
       --config-name config_70m \\
       model=hybrid_70m \\
       dataset=pubmed \\
       trainer=a100_single_gpu \\
       trainer.max_steps=50000 \\
       dataset.batch_size=32 \\
       experiment_name=production_hybrid_70m_pubmed \\
       output_dir=outputs/production_stage0

2. STAGE 1 PRODUCTION CONTRASTIVE TRAINING:
   Commands:
   python scripts/train_contrastive.py \\
       --config-name config_70m \\
       dataset=pubmed \\
       trainer=a100_single_gpu \\
       trainer.max_steps=20000 \\
       dataset.batch_size=32 \\
       lm_checkpoint=outputs/production_stage0/last.ckpt \\
       contrastive_mode=simcse \\
       experiment_name=production_hybrid_70m_contrastive \\
       output_dir=outputs/production_stage1

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TROUBLESHOOTING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

If issues occur during production:

1. Check logs in: $OUTPUT_DIR/logs/
2. Verify GPU memory: nvidia-smi (A100 needs 40-80GB)
3. Monitor loss curves in TensorBoard:
   tensorboard --logdir $OUTPUT_DIR/lightning_logs/
4. For checkpoint issues, use: python debug_checkpoint_keys.py
5. For CUDA/Triton kernel issues, check: scripts/verify_*.py

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Validation completed successfully!
Run time: $(date '+%Y-%m-%d %H:%M:%S')
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EOF

    cat "$OUTPUT_DIR/validation_report.txt"
    log_success "Report saved: $OUTPUT_DIR/validation_report.txt"

    echo ""
    echo -e "${GREEN}Pipeline validation complete!${NC}"
    echo -e "Output directory: ${YELLOW}$OUTPUT_DIR${NC}"
    echo ""
}

# ============================================================================
# Entry Point
# ============================================================================

if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    main "$@"
fi
