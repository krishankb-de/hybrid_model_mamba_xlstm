#!/bin/bash
#SBATCH --partition=mitarb
#SBATCH --account=mitarb
#SBATCH --gres=gpu:mitarb:1
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --job-name=lm_stage0_pubmed
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log

set -euo pipefail

echo "=== JOB START (Stage 0: LM pretraining on PubMed) ==="
date
echo "Host: $(hostname)"
echo "Submit dir: ${SLURM_SUBMIT_DIR}"
echo ""
echo "Configuration:"
echo "  - Dataset: PubMed (ccdv/pubmed-summarization)"
echo "  - Batch size: 32 (effective 128 w/ grad accum 4)"
echo "  - Sequence length: 512"
echo "  - Learning rate: 6e-4"
echo "  - Warmup steps: 1000"
echo "  - Max steps: 40,000"
echo "  - Grad clip: 1.0"
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

if ! python -c "import torch" 2>/dev/null; then
    echo "ERROR: PyTorch not found in virtual environment!"
    exit 1
fi

echo ""
echo "Verifying CUDA availability:"
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA device count: {torch.cuda.device_count()}')
if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    print(f'Device name: {torch.cuda.get_device_name(0)}')
"
echo ""

if ! python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    echo "ERROR: CUDA is not available to PyTorch!"
    exit 1
fi

nvidia-smi

python scripts/train.py \
  --config-name config_70m \
  dataset=pubmed \
  trainer=a100_single_gpu \
  trainer.accelerator=cuda \
  trainer.max_epochs=-1 \
  trainer.max_steps=40000 \
  trainer.accumulate_grad_batches=4 \
  trainer.val_check_interval=1000 \
  trainer.log_every_n_steps=25 \
  dataset.batch_size=32 \
  dataset.eval_batch_size=32 \
  dataset.max_length=512 \
  dataset.max_seq_length=512 \
  dataset.num_workers=4 \
  dataset.preprocessing_num_workers=4 \
  dataset.pin_memory=true \
  callbacks.checkpoint.every_n_train_steps=2000 \
  callbacks.checkpoint.save_top_k=3 \
  experiment_name=hybrid_70m_lm_pretrain_pubmed \
  output_dir=./outputs/hybrid_70m_lm_pretrain_pubmed \
  wandb.enabled=false \
  model.learning_rate=6.0e-4 \
  model.warmup_steps=1000 \
  model.gradient_clip_val=1.0

echo ""
echo "=== JOB END (Stage 0: LM pretrain) ==="
echo "Checkpoint saved to: ./outputs/hybrid_70m_lm_pretrain_pubmed/checkpoints/"
date
