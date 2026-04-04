#!/bin/bash
#SBATCH --partition=mitarb
#SBATCH --account=mitarb
#SBATCH --gres=gpu:mitarb:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --job-name=eval_stage1
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log

set -euo pipefail

echo "=== JOB START (Stage 1 Evaluation) ==="
date
echo "Host: $(hostname)"
echo "Submit dir: ${SLURM_SUBMIT_DIR}"
echo ""
echo "Configuration:"
echo "  - Checkpoint: outputs/stage1_pubmed_simcse/checkpoints/last.ckpt"
echo "  - Evaluations: STS (BIOSSES), Retrieval (PubMed), Perplexity"
echo "  - Batch size: 32"
echo "  - Retrieval pairs: 1,000"
echo "  - Expected runtime: 2-4 hours"
echo ""

cd "${SLURM_SUBMIT_DIR}/hybrid_model_mamba_xlstm"

export HF_HOME="$PWD/.hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCHINDUCTOR_CACHE_DIR="$PWD/.torchinductor"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"
export CUDA_LAUNCH_BLOCKING=0

echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"

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

# Activate virtual environment
if [ ! -d ".venv" ]; then
    echo "ERROR: Virtual environment .venv not found!"
    echo "Please create it first with: python -m venv .venv"
    exit 1
fi

source .venv/bin/activate

# Verify Python and torch are available
if ! python -c "import torch" 2>/dev/null; then
    echo "ERROR: PyTorch not found in virtual environment!"
    echo "Please install requirements: pip install -r requirements.txt"
    exit 1
fi

# Verify CUDA is available
echo ""
echo "Verifying CUDA availability:"
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA device count: {torch.cuda.device_count()}')
if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    print(f'Current device: {torch.cuda.current_device()}')
    print(f'Device name: {torch.cuda.get_device_name(0)}')
"
echo ""

# Install evaluation dependencies if not already installed
echo "Installing evaluation dependencies..."
pip install -q sentence-transformers scipy datasets

nvidia-smi

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
echo "=== JOB END (Stage 1 Evaluation) ==="
echo "Results saved to: $OUTPUT_DIR"
date
