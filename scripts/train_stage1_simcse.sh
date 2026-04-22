#!/bin/bash
#SBATCH --job-name=stage1_simcse
#SBATCH --output=/scratch/bhushkri/hybrid_xmamba_a100_70m_40/logs/stage1_simcse_%j.log
#SBATCH --error=/scratch/bhushkri/hybrid_xmamba_a100_70m_40/logs/stage1_simcse_%j.log
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

echo "=== JOB START (Stage 1: SimCSE Contrastive Fine-tuning) ==="
date
echo "Host: $(hostname)"
echo "Submit dir: /scratch/bhushkri/hybrid_xmamba_a100_70m_40"
echo ""

export CUDA_VISIBLE_DEVICES=0
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

echo ""
echo "Verifying CUDA availability:"
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device name: {torch.cuda.get_device_name(0)}')
    print(f'Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

echo ""
nvidia-smi
echo ""

cd /scratch/bhushkri/hybrid_xmamba_a100_70m_40/hybrid_model_mamba_xlstm
source .venv/bin/activate

LM_CHECKPOINT=./outputs/hybrid_70m_stage0_kd_pubmed/checkpoints/stage0_model_only.pt
PUBMED_CACHE=/scratch/bhushkri/hybrid_xmamba_a100_70m_40/pubmed_cache

if [ ! -f "$LM_CHECKPOINT" ]; then
    echo "ERROR: Stage 0 checkpoint not found at $LM_CHECKPOINT"
    echo "Run the checkpoint extraction first:"
    echo "  python -c \"import torch; ckpt=torch.load('outputs/hybrid_70m_stage0_kd_pubmed/checkpoints/last.ckpt', map_location='cpu', weights_only=False); state={k[len('model.'):]:v for k,v in ckpt['state_dict'].items() if k.startswith('model.')}; torch.save({'state_dict':state}, '$LM_CHECKPOINT'); print(f'Extracted {len(state)} keys')\""
    exit 1
fi

echo "Stage 0 checkpoint: $LM_CHECKPOINT"
echo "PubMed cache: $PUBMED_CACHE"
echo ""
echo "Starting Stage 1 SimCSE contrastive fine-tuning..."

python scripts/train_contrastive.py \
    model=hybrid_70m \
    dataset=pubmed \
    trainer=a100_single_gpu \
    contrastive_mode=simcse \
    dataset.batch_size=8 \
    dataset.cache_dir="$PUBMED_CACHE" \
    trainer.accumulate_grad_batches=4 \
    trainer.compile_model=false \
    trainer.max_steps=10000 \
    +lm_checkpoint="$LM_CHECKPOINT" \
    wandb.enabled=false \
    experiment_name=hybrid_70m_stage1_simcse

EXIT_CODE=$?

echo ""
nvidia-smi
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "Stage 1 SimCSE complete."
    echo "Checkpoint saved to: outputs/hybrid_70m_stage1_simcse/checkpoints/"
    echo "=== JOB END (Stage 1 SimCSE: SUCCESS) ==="
else
    echo "=== JOB END (Stage 1 SimCSE: FAILED exit=$EXIT_CODE) ==="
fi
date
