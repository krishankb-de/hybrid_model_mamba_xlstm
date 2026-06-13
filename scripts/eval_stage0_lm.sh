#!/bin/bash
#SBATCH --partition=mitarb
#SBATCH --account=mitarb
#SBATCH --gres=gpu:mitarb:1
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --job-name=eval_stage0_lm
#SBATCH --output=/scratch/bhushkri/hybrid_xmamba_a100_70m_40/logs/%x_%j.log
#SBATCH --error=/scratch/bhushkri/hybrid_xmamba_a100_70m_40/logs/%x_%j.log

# Stage 0 evaluation: perplexity + throughput on PubMed validation set.
#
# Loads stage0_model_only.pt (117-key stripped checkpoint, ~166MB, no teacher).
# evaluate_lm.py handles weights_only=False internally via torch.load — safe
# since this is our own checkpoint from the training run.
#
# Expected runtime: ~5 min (501 batches at ~10 it/s on A100).
# Results written to: outputs/hybrid_70m_stage0_kd_pubmed/eval_results/results.json

set -euo pipefail

mkdir -p /scratch/bhushkri/hybrid_xmamba_a100_70m_40/logs

echo "=== JOB START (Stage 0 LM Evaluation) ==="
date
echo "Host: $(hostname)"
echo "Submit dir: ${SLURM_SUBMIT_DIR}"
echo ""

cd "${SLURM_SUBMIT_DIR}/hybrid_model_mamba_xlstm"

export HF_HOME="$PWD/.hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCHINDUCTOR_CACHE_DIR="$PWD/.torchinductor"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"
export CUDA_LAUNCH_BLOCKING=0

echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"

if [ ! -d ".venv" ]; then
    echo "ERROR: Virtual environment .venv not found!"
    exit 1
fi
source .venv/bin/activate

echo ""
echo "Verifying CUDA availability:"
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device name: {torch.cuda.get_device_name(0)}')
    mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f'Total VRAM: {mem:.1f} GB')
"
echo ""

nvidia-smi

CHECKPOINT="./outputs/phase9_stage0_arch_v2/checkpoints/stage0_v2_model_only.pt"
OUTPUT_DIR="./outputs/phase9_stage0_arch_v2/eval_results"

if [ ! -f "${CHECKPOINT}" ]; then
    echo "ERROR: Checkpoint not found at ${CHECKPOINT}"
    echo "Run the extraction first (on login node, CPU-only, takes ~30s):"
    echo "  python -c \""
    echo "  import torch"
    echo "  ckpt = torch.load('/scratch/bhushkri/hybrid_xmamba_a100_70m_40/phase9_stage0_arch_v2/checkpoints/<best-val-loss>.ckpt',"
    echo "                     map_location='cpu', weights_only=False)"
    echo "  state = {k[len('model.'):]: v for k, v in ckpt['state_dict'].items()"
    echo "           if k.startswith('model.')}"
    echo "  torch.save({'state_dict': state}, '${CHECKPOINT}')"
    echo "  print(f'Extracted {len(state)} keys')\""
    exit 1
fi

echo "Evaluating checkpoint: ${CHECKPOINT}"
echo ""

python scripts/evaluate_lm.py \
    --checkpoint "${CHECKPOINT}" \
    --model-config hybrid_70m_v2 \
    --layer-pattern "mamba,mamba,mamba,mlstm,mlstm,mamba,mamba,mamba" \
    --dataset pubmed \
    --split validation \
    --batch-size 16 \
    --max-length 512 \
    --throughput \
    --output-dir "${OUTPUT_DIR}"

EXIT_CODE=$?

echo ""
nvidia-smi
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "Results written to: ${OUTPUT_DIR}/results.json"
    echo "=== JOB END (Stage 0 LM Evaluation: SUCCESS) ==="
else
    echo "=== JOB END (Stage 0 LM Evaluation: FAILED exit=${EXIT_CODE}) ==="
fi
date
