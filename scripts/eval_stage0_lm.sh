#!/bin/bash
#SBATCH --job-name=eval_stage0_lm
#SBATCH --output=/scratch/bhushkri/hybrid_xmamba_a100_70m_40/logs/eval_stage0_lm_%j.log
#SBATCH --error=/scratch/bhushkri/hybrid_xmamba_a100_70m_40/logs/eval_stage0_lm_%j.log
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4

echo "=== JOB START (Stage 0 LM Evaluation) ==="
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

CHECKPOINT=./outputs/hybrid_70m_stage0_kd_pubmed/checkpoints/stage0_model_only.pt
OUTPUT_DIR=./outputs/hybrid_70m_stage0_kd_pubmed/eval_results

echo "Evaluating checkpoint: $CHECKPOINT"
echo ""

python scripts/evaluate_lm.py \
    --checkpoint "$CHECKPOINT" \
    --model-config hybrid_70m \
    --dataset pubmed \
    --split validation \
    --batch-size 16 \
    --max-length 512 \
    --throughput \
    --output-dir "$OUTPUT_DIR"

EXIT_CODE=$?

echo ""
nvidia-smi
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "=== JOB END (Stage 0 LM Evaluation: SUCCESS) ==="
    echo "Results written to: $OUTPUT_DIR/results.json"
else
    echo "=== JOB END (Stage 0 LM Evaluation: FAILED exit=$EXIT_CODE) ==="
fi
date
