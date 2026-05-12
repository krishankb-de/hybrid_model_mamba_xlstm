#!/bin/bash
# Phase 8B — willi A100 sanity, v2 (hybrid_70m_v2, [m,m,m,L,L,m,m,m])
# All Phase 3/4/6/7 active. norm_topology=hybrid is now baked into
# hybrid_70m_v2.yaml (no CLI override needed).
#
# OOM fix (Phase 8 retry): same as v1 — use_gradient_checkpointing=true +
# batch_size=16/accumulate=8 to keep effective batch 128.
#SBATCH --partition=mitarb
#SBATCH --account=mitarb
#SBATCH --gres=gpu:mitarb:1
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --job-name=phase8_sanity_v2
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log

set -euo pipefail
echo "=== Phase 8B sanity (retry): hybrid_70m_v2 — 2000 PubMed steps ==="
date; hostname

cd "${SLURM_SUBMIT_DIR}/hybrid_model_mamba_xlstm"

export HF_HOME="$PWD/.hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCHINDUCTOR_CACHE_DIR="$PWD/.torchinductor"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"

source .venv/bin/activate
python -c "import torch; assert torch.cuda.is_available()"
nvidia-smi

python scripts/train.py \
  --config-name config_70m \
  model=hybrid_70m_v2 \
  dataset=pubmed \
  trainer=a100_single_gpu \
  trainer.accelerator=cuda \
  trainer.max_epochs=-1 \
  trainer.max_steps=2000 \
  trainer.accumulate_grad_batches=8 \
  trainer.val_check_interval=500 \
  trainer.log_every_n_steps=25 \
  dataset.batch_size=16 \
  dataset.eval_batch_size=16 \
  dataset.max_length=512 \
  dataset.max_seq_length=512 \
  dataset.num_workers=2 \
  dataset.preprocessing_num_workers=4 \
  dataset.pin_memory=true \
  callbacks.checkpoint.every_n_train_steps=1000 \
  callbacks.checkpoint.save_top_k=1 \
  experiment_name=phase8_sanity_v2 \
  output_dir=./outputs/phase8_sanity_v2 \
  wandb.enabled=false \
  model.learning_rate=6.0e-4 \
  model.warmup_steps=20 \
  model.gradient_clip_val=1.0 \
  model.use_gradient_checkpointing=true \
  +model.scheduler_name=wsd \
  +model.beta2_schedule=true \
  +model.beta2_start=0.999 \
  +model.beta2_end=0.974

echo "=== Phase 8B sanity v2 END ==="; date
