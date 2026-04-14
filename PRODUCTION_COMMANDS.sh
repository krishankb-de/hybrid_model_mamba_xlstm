#!/bin/bash
# ============================================================================
# PRODUCTION TRAINING COMMANDS
# Hybrid Mamba-xLSTM 70M Model on A100
# ============================================================================
# Copy and paste these commands directly to production terminal
# Update paths as needed for your environment
# ============================================================================

# ============================================================================
# PRE-FLIGHT CHECKS
# ============================================================================

# Check GPU availability
nvidia-smi

# Activate virtual environment
source venv/bin/activate

# Verify PyTorch installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"

# Verify project structure
ls -la scripts/train.py configs/config_70m.yaml

# ============================================================================
# STAGE 0: LM PRETRAINING (Full Production)
# Duration: ~6-8 hours on A100
# Hardware: Single A100 40GB
# ============================================================================

# Start Stage 0
python scripts/train.py \
    --config-name config_70m \
    model=hybrid_70m \
    dataset=pubmed \
    trainer=a100_single_gpu \
    trainer.max_steps=50000 \
    trainer.val_check_interval=500 \
    trainer.log_every_n_steps=25 \
    dataset.batch_size=32 \
    dataset.eval_batch_size=32 \
    dataset.max_length=512 \
    experiment_name=production_hybrid_70m_pubmed_stage0 \
    output_dir=outputs/production_stage0 \
    2>&1 | tee logs/stage0_training.log

# Expected output checkpoint path:
# outputs/production_stage0/last.ckpt (~280MB)

# Verify Stage 0 checkpoint
python check_checkpoint_compatibility.py \
    --checkpoint outputs/production_stage0/last.ckpt

# ============================================================================
# STAGE 1: CONTRASTIVE TRAINING (Full Production)
# Duration: ~3-4 hours on A100
# Prerequisite: Stage 0 completed successfully
# Hardware: Single A100 40GB
# ============================================================================

# Store Stage 0 checkpoint path
STAGE0_CKPT="outputs/production_stage0/last.ckpt"

# Start Stage 1 (ONLY after Stage 0 completes)
python scripts/train_contrastive.py \
    --config-name config_70m \
    dataset=pubmed \
    trainer=a100_single_gpu \
    trainer.max_steps=20000 \
    trainer.val_check_interval=500 \
    trainer.log_every_n_steps=25 \
    dataset.batch_size=32 \
    dataset.eval_batch_size=32 \
    dataset.max_length=512 \
    lm_checkpoint=$STAGE0_CKPT \
    contrastive_mode=simcse \
    experiment_name=production_hybrid_70m_contrastive_stage1 \
    output_dir=outputs/production_stage1 \
    2>&1 | tee logs/stage1_training.log

# Expected output checkpoint path:
# outputs/production_stage1/last.ckpt (~280MB + projection head)

# Verify Stage 1 checkpoint
python check_checkpoint_compatibility.py \
    --checkpoint outputs/production_stage1/last.ckpt

# ============================================================================
# POST-TRAINING EVALUATION
# ============================================================================

# Quick evaluation on WikiText (sanity check)
python scripts/evaluate_lm.py \
    --checkpoint outputs/production_stage1/last.ckpt \
    --model-config hybrid_70m \
    --dataset wikitext \
    --split test \
    --batch-size 32 \
    --throughput \
    --generate \
    --output-dir outputs/production_evaluation/ \
    2>&1 | tee logs/evaluation.log

# Full retrieval benchmark (optional, requires more time)
# python scripts/evaluate_retrieval.py \
#     --checkpoint outputs/production_stage1/last.ckpt \
#     --model-config hybrid_70m \
#     --output-dir outputs/production_evaluation/

# ============================================================================
# MONITORING IN PARALLEL TERMINALS
# ============================================================================

# Terminal 1: TensorBoard (Stage 0)
# tensorboard --logdir outputs/production_stage0/lightning_logs/ --port 6006

# Terminal 2: TensorBoard (Stage 1) 
# tensorboard --logdir outputs/production_stage1/lightning_logs/ --port 6007

# Terminal 3: GPU monitoring
# watch -n 1 nvidia-smi

# Terminal 4: Log tail
# tail -f logs/stage0_training.log   # During Stage 0
# tail -f logs/stage1_training.log   # During Stage 1

# ============================================================================
# CHECKPOINT MANAGEMENT
# ============================================================================

# Copy final checkpoint to production location
cp outputs/production_stage1/last.ckpt \
   /path/to/production/hybrid_70m_pubmed_final.ckpt

# Create backup
cp outputs/production_stage1/last.ckpt \
   /path/to/backups/hybrid_70m_pubmed_$(date +%Y%m%d_%H%M%S).ckpt

# Archive all logs
tar -czf logs/production_training_$(date +%Y%m%d).tar.gz \
    outputs/production_stage0/lightning_logs/ \
    outputs/production_stage1/lightning_logs/ \
    logs/stage0_training.log \
    logs/stage1_training.log

# ============================================================================
# TROUBLESHOOTING COMMANDS
# ============================================================================

# If CUDA OOM: reduce batch size
# python scripts/train.py ... dataset.batch_size=16

# If Triton kernel fails: disable compiled kernels
# python scripts/train.py ... use_triton_kernels=false

# If gradient explosion: reduce learning rate
# python scripts/train.py ... model.learning_rate=3e-4

# Debug checkpoint keys
python debug_checkpoint_keys.py \
    --checkpoint outputs/production_stage1/last.ckpt

# Check for checkpoint prefix issues
python check_checkpoint_compatibility.py \
    --checkpoint outputs/production_stage1/last.ckpt

# ============================================================================
# USEFUL UTILITIES
# ============================================================================

# Count total parameters of Stage 1 model
python -c "
from hybrid_xmamba.models.hybrid_lm import HybridLanguageModel
from hybrid_xmamba.models.configuration_hybrid import HybridConfig
config = HybridConfig.from_pretrained('hybrid_70m')
model = HybridLanguageModel(config)
total = sum(p.numel() for p in model.parameters())
print(f'Total parameters: {total:,}')
print(f'Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')
"

# Profile GPU memory
python scripts/performance_profile.py \
    --checkpoint outputs/production_stage1/last.ckpt \
    --model-config hybrid_70m \
    --batch-size 32 \
    --max-length 512

# ============================================================================
# NEXT STEPS AFTER PRODUCTION TRAINING
# ============================================================================

# 1. Copy checkpoint to serving infrastructure
# scp outputs/production_stage1/last.ckpt user@server:/production/models/

# 2. Run final validation
# python scripts/evaluate_lm.py --checkpoint outputs/production_stage1/last.ckpt ...

# 3. Archive logs for audit trail
# tar -czf production_logs_backup.tar.gz outputs/production_stage*/

# 4. Update model registry with new checkpoint path
# echo "outputs/production_stage1/last.ckpt" > ACTIVE_MODEL_CHECKPOINT.txt

# ============================================================================
# DOCUMENTATION & REFERENCE
# ============================================================================

# - Pipeline Guide: PIPELINE_GUIDE.md
# - Model Architecture: CLAUDE.md
# - Configuration System: configs/
# - Training Script: scripts/train.py
# - Contrastive Script: scripts/train_contrastive.py
# - Debug Utilities: check_checkpoint_compatibility.py, debug_checkpoint_keys.py

echo "Production training commands ready. See comments for usage."
