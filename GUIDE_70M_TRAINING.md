# Hybrid Mamba-xLSTM 70M: Complete Training, Testing & Evaluation Guide

> **Model:** Hybrid Mamba-xLSTM 70M Parameters (dim=512, 8 layers)  
> **GPU Target:** NVIDIA A100 40GB (Willi) — also works on A100 80GB, V100, RTX 3090/4090  
> **Dataset:** WikiText-103 (smoke) · PubMed 1.5B tokens (Stage 0 real run)  
> **Estimated Time:** Stage 0 ≈ 25–28h on A100-40GB bf16; Stage 1 SimCSE ≈ 6–8h

---

## Willi A100 40GB operational notes

- `configs/trainer/a100_single_gpu.yaml` defaults to 40GB: `compile_model: false`, `batch_size=32`, `grad_accum=2`.
- **Stage 0 KD** (BioMedLM teacher loaded) automatically drops to `batch_size=16`, `grad_accum=4` via `configs/distill/stage0_biomedlm.yaml`. If you still hit OOM, use `oom_fallback_batch_size=8`, `oom_fallback_grad_accum=8`.
- **No gradient checkpointing is implemented.** If a config reliably OOMs, halve batch size before attempting to add `torch.utils.checkpoint` (it's not currently supported by any block).
- **Checkpoint cadence** is 500 steps (≈17 min). Verify with `cfg.callbacks.checkpoint.every_n_train_steps` at job start.
- **SLURM preemption**: `SignalCheckpointCallback` saves `<ckpt_dir>/interrupt.ckpt` on SIGTERM / SIGUSR1 / uncaught exception. Resume with `ckpt_path=<ckpt_dir>/interrupt.ckpt`. Use `sbatch --requeue --signal=B:SIGUSR1@90` to give the handler 90s before kill.
- **Reproducibility**: every training entrypoint writes `<output_dir>/run_metadata.json` at step 0 (git SHA, branch, dirty flag, argv, resolved Hydra config).
- **Mid-epoch resume is NOT supported** by the default IterableDataset path — a requeued job restarts from the epoch boundary and may see data twice. For 1.5B-token Stage 0 (single epoch) this is a non-issue.
- **First Stage 0 run**: enable `trainer.detect_anomaly=true` for the first 100 steps to surface any NaN in the KD path. Disable for the main run (it's slow).

---

## Table of Contents

0. [Pre-Validation on Google Colab (Before A100)](#0-pre-validation-on-google-colab-before-a100)
1. [Overview](#1-overview)
2. [Prerequisites & Environment Setup](#2-prerequisites--environment-setup)
3. [Project Structure](#3-project-structure)
4. [Quick Start — One Command](#4-quick-start--one-command)
5. [Step-by-Step Manual Process](#5-step-by-step-manual-process)
   - [Step 1: Sanity Check](#step-1-sanity-check)
   - [Step 2: Train the Hybrid 70M Model](#step-2-train-the-hybrid-70m-model)
   - [Step 3: Train Mamba-only 70M Baseline](#step-3-train-mamba-only-70m-baseline)
   - [Step 4: Train xLSTM-only 70M Baseline](#step-4-train-xlstm-only-70m-baseline)
   - [Step 5: Evaluate All Models](#step-5-evaluate-all-models)
   - [Step 6: Compare Results](#step-6-compare-results)
6. [Using the Automated Pipeline](#6-using-the-automated-pipeline)
7. [Configuration Reference](#7-configuration-reference)
8. [Adjusting for Different GPUs](#8-adjusting-for-different-gpus)
9. [Monitoring Training](#9-monitoring-training)
10. [Troubleshooting](#10-troubleshooting)
11. [Scaling Up to 150M](#11-scaling-up-to-150m)

---

## 0. Pre-Validation on Google Colab (Before A100)

> ⚠️ **IMPORTANT:** Always validate the 70M pipeline on Google Colab first  
> before spending A100 compute credits on Lightning AI Studio.

### Why validate on Colab first?

- A100 GPU time is expensive. A single misconfiguration wastes hours of credits.
- Colab's free T4 GPU is sufficient to verify that:
  - ✅ All imports resolve and the model instantiates correctly
  - ✅ The data pipeline (WikiText-103 download → tokenize → pack → DataLoader) works
  - ✅ Forward/backward passes complete without errors
  - ✅ Loss decreases from the initial random value (~10-11 → ~8-9)
  - ✅ Checkpointing writes files to disk
- The 70M model is small enough to run on T4 (15 GB VRAM).

### Colab Validation Steps (using `Colab_Setup.ipynb`)

Open **`Colab_Setup.ipynb`** in Google Colab and run the cells in order. The key cells for 70M validation are:

#### Step 0-A: Check GPU & Install

```python
# Cell 1 — Check GPU
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

```python
# Cell 2 — Clone & install (uses latest code from the a100_70m_baseline branch)
!rm -rf /content/hybrid_model_mamba_xlstm
!git clone -b a100_70m_baseline https://github.com/krishankb-de/hybrid_model_mamba_xlstm.git /content/hybrid_model_mamba_xlstm
%cd /content/hybrid_model_mamba_xlstm
!pip install -e . -q
```

#### Step 0-B: Quick Inference Test (~1 min)

```python
# Tests model creation + single forward pass (no training)
import torch, sys
sys.path.insert(0, '/content/hybrid_model_mamba_xlstm')
from hybrid_xmamba.models.configuration_hybrid import HybridConfig
from hybrid_xmamba.models.hybrid_lm import HybridLanguageModel

config = HybridConfig(dim=512, num_layers=8, vocab_size=50257,
                      state_size=16, conv_size=4, expand_factor=2,
                      head_dim=64, num_heads=8, use_tfla=True, proj_factor=2,
                      slstm_hidden_dim=512, slstm_num_heads=4)
model = HybridLanguageModel(config).eval().cuda()
x = torch.randint(0, 50257, (2, 128)).cuda()
out = model(x)
print(f"✅ Inference OK — output shape: {out.logits.shape}")
print(f"   Parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
```

#### Step 0-C: Training Sanity Check (~20-25 min on T4)

This is the critical test — it runs 50 actual training steps:

```bash
# From a Colab code cell (prefix with !)
!python scripts/train.py \
    model=hybrid_70m \
    dataset=wikitext \
    trainer=colab_single_gpu \
    trainer.max_epochs=1 \
    trainer.num_sanity_val_steps=0 \
    dataset.batch_size=2 \
    dataset.eval_batch_size=2 \
    dataset.num_workers=0 \
    dataset.preprocessing_num_workers=1 \
    +dataset.max_seq_length=128 \
    trainer.accumulate_grad_batches=1 \
    trainer.val_check_interval=1.0 \
    trainer.log_every_n_steps=5 \
    trainer.limit_train_batches=50 \
    trainer.limit_val_batches=0 \
    trainer.precision=32 \
    wandb.enabled=false \
    trainer.enable_checkpointing=false \
    trainer.default_root_dir=/content/outputs
```

**What to verify in the output:**
- Model prints `~70M parameters`
- Loss starts around 10-11 and decreases
- No CUDA OOM errors
- No import or config errors
- Completes all 50 steps

#### Step 0-D: Test Mamba-only & xLSTM-only Baselines (~20-25 min each)

```bash
# Mamba-only 70M baseline — 50 steps
!python scripts/train.py \
    model=mamba_70m_baseline \
    dataset=wikitext \
    trainer=colab_single_gpu \
    trainer.max_epochs=1 \
    trainer.num_sanity_val_steps=0 \
    dataset.batch_size=2 \
    dataset.eval_batch_size=2 \
    dataset.num_workers=0 \
    dataset.preprocessing_num_workers=1 \
    +dataset.max_seq_length=128 \
    trainer.accumulate_grad_batches=1 \
    trainer.log_every_n_steps=5 \
    trainer.limit_train_batches=50 \
    trainer.limit_val_batches=0 \
    trainer.precision=32 \
    wandb.enabled=false \
    trainer.enable_checkpointing=false \
    trainer.default_root_dir=/content/outputs
```

```bash
# xLSTM-only 70M baseline — 50 steps
!python scripts/train.py \
    model=xlstm_70m_baseline \
    dataset=wikitext \
    trainer=colab_single_gpu \
    trainer.max_epochs=1 \
    trainer.num_sanity_val_steps=0 \
    dataset.batch_size=2 \
    dataset.eval_batch_size=2 \
    dataset.num_workers=0 \
    dataset.preprocessing_num_workers=1 \
    +dataset.max_seq_length=128 \
    trainer.accumulate_grad_batches=1 \
    trainer.log_every_n_steps=5 \
    trainer.limit_train_batches=50 \
    trainer.limit_val_batches=0 \
    trainer.precision=32 \
    wandb.enabled=false \
    trainer.enable_checkpointing=false \
    trainer.default_root_dir=/content/outputs
```

#### ✅ Colab Validation Checklist

| Check | Expected Result |
|---|---|
| GPU detected | `Tesla T4` (or V100/A100 if Colab Pro) |
| Model loads | `~70M parameters` printed |
| Inference works | Output shape `[B, seq_len, 50257]` |
| Hybrid 50 steps | Loss decreases, no crashes |
| Mamba-only 50 steps | Loss decreases, no crashes |
| xLSTM-only 50 steps | Loss decreases, no crashes |
| No CUDA OOM | All 3 models fit in T4 15 GB |

> **Once all checks pass → safe to move to Lightning AI A100 for full training.**

### Moving from Colab to Lightning AI Studio (A100)

After Colab validation succeeds:

```bash
# On Lightning AI Studio terminal:
git clone -b a100_70m_baseline https://github.com/krishankb-de/hybrid_model_mamba_xlstm.git
cd hybrid_model_mamba_xlstm
pip install -r requirements.txt
pip install -e .

# Run the full 70M experiment pipeline
python scripts/run_70m_experiments.py --max-steps 10000
```

---

## 1. Overview

This guide walks through the **complete pipeline** to train, test, and evaluate the 70M parameter Hybrid Mamba-xLSTM model and its baselines. The 70M scale is ideal for:

- **Initial validation** — Verify the architecture works before scaling up
- **Hyperparameter tuning** — Fast iteration cycles (~2h per model on A100)
- **Ablation studies** — Compare hybrid vs. pure Mamba vs. pure xLSTM
- **Resource-constrained training** — Fits on consumer GPUs (RTX 3090+)

### Architecture Comparison

| Model | `layer_pattern` | dim | layers | ~Params |
|---|---|---|---|---|
| **Hybrid 70M** | `[mamba, mamba, mlstm]` | 512 | 8 | ~70M |
| **Mamba-only 70M** | `[mamba]` | 512 | 8 | ~70M |
| **xLSTM-only 70M** | `[mlstm]` | 512 | 8 | ~70M |

All three models share the same `dim=512` and `num_layers=8` for a **fair, controlled comparison**.

---

## 2. Prerequisites & Environment Setup

### 2.1 System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| GPU | RTX 3090 (24 GB) | A100 80 GB |
| RAM | 32 GB | 64 GB |
| Disk | 50 GB free | 100 GB free |
| CUDA | 11.8+ | 12.1+ |
| Python | 3.9+ | 3.10 or 3.11 |

### 2.2 Environment Setup (CMD / Terminal)

```bash
# 1. Navigate to the project directory
cd "D:\Thesis Project\Hybrid_Model_Mamba_xLSTM_2\Hybrid_Model_Mamba_xLSTM"

# 2. Create a virtual environment (if you haven't already)
python -m venv venv

# 3. Activate the virtual environment
# On Windows CMD:
venv\Scripts\activate
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
# On Linux/macOS:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Install the project in editable mode
pip install -e .

# 6. Verify GPU is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

### 2.3 For Lightning AI Studio (A100)

```bash
# In a Lightning AI terminal:
cd /teamspace/studios/this_studio
git clone <your-repo-url> hybrid-xmamba
cd hybrid-xmamba
pip install -r requirements.txt
pip install -e .
```

---

## 3. Project Structure

```
Hybrid_Model_Mamba_xLSTM/
├── configs/
│   ├── config.yaml              # Default Hydra config (350M)
│   ├── config_70m.yaml          # 70M preset config
│   ├── model/
│   │   ├── hybrid_70m.yaml      # ★ Hybrid 70M model config
│   │   ├── mamba_70m_baseline.yaml   # ★ Mamba-only 70M baseline
│   │   ├── xlstm_70m_baseline.yaml   # ★ xLSTM-only 70M baseline
│   │   ├── hybrid_150m.yaml     # 150M model config
│   │   └── ...
│   ├── dataset/
│   │   └── wikitext.yaml        # WikiText-103 dataset config
│   └── trainer/
│       ├── a100_single_gpu.yaml # A100-optimized trainer
│       └── single_gpu.yaml      # Generic single-GPU trainer
├── scripts/
│   ├── run_70m_experiments.py   # ★ Automated 70M pipeline
│   ├── run_a100_experiments.py  # Automated 150M pipeline
│   ├── train.py                 # Core training script
│   ├── evaluate_lm.py          # LM evaluation (perplexity)
│   ├── evaluate.py             # General evaluation
│   └── performance_profile.py  # Profiling utility
├── hybrid_xmamba/               # Model source code
│   ├── models/                  # Model definitions
│   ├── layers/                  # Layer implementations
│   ├── kernels/                 # Triton kernels
│   ├── training/                # Lightning module, optimizer
│   └── utils/                   # Utilities
└── outputs/                     # Training outputs (created at runtime)
    ├── hybrid_70m_wikitext/
    ├── mamba_70m_wikitext/
    └── xlstm_70m_wikitext/
```

---

## 4. Quick Start — One Command

To run the **entire pipeline** (sanity check → train 3 models → evaluate → compare):

```bash
# Full pipeline on A100 (default 10k steps, ~6-10 hours)
python scripts/run_70m_experiments.py

# Full pipeline with W&B logging
python scripts/run_70m_experiments.py --wandb

# Paper-quality results (50k steps, ~30 hours on A100)
python scripts/run_70m_experiments.py --max-steps 50000

# Quick verification (500 steps, ~30 min)
python scripts/run_70m_experiments.py --max-steps 500

# See what commands would run, without executing
python scripts/run_70m_experiments.py --dry-run
```

---

## 5. Step-by-Step Manual Process

If you prefer running each step individually (recommended for debugging or when you want more control):

### Step 1: Sanity Check

Run 50 training steps to verify the entire pipeline works:

```bash
python scripts/train.py \
    model=hybrid_70m \
    dataset=wikitext \
    trainer=a100_single_gpu \
    trainer.max_steps=50 \
    trainer.val_check_interval=25 \
    trainer.log_every_n_steps=5 \
    trainer.enable_checkpointing=false \
    trainer.limit_val_batches=5 \
    dataset.batch_size=32 \
    dataset.eval_batch_size=16 \
    dataset.num_workers=4 \
    experiment_name=sanity_check_70m \
    wandb.enabled=false
```

**On Windows CMD** (use `^` for line continuation):

```cmd
python scripts/train.py ^
    model=hybrid_70m ^
    dataset=wikitext ^
    trainer=a100_single_gpu ^
    trainer.max_steps=50 ^
    trainer.val_check_interval=25 ^
    trainer.log_every_n_steps=5 ^
    trainer.enable_checkpointing=false ^
    trainer.limit_val_batches=5 ^
    dataset.batch_size=32 ^
    dataset.eval_batch_size=16 ^
    dataset.num_workers=4 ^
    experiment_name=sanity_check_70m ^
    wandb.enabled=false
```

**On Windows PowerShell** (use `` ` `` for line continuation):

```powershell
python scripts/train.py `
    model=hybrid_70m `
    dataset=wikitext `
    trainer=a100_single_gpu `
    trainer.max_steps=50 `
    trainer.val_check_interval=25 `
    trainer.log_every_n_steps=5 `
    trainer.enable_checkpointing=false `
    trainer.limit_val_batches=5 `
    dataset.batch_size=32 `
    dataset.eval_batch_size=16 `
    dataset.num_workers=4 `
    experiment_name=sanity_check_70m `
    wandb.enabled=false
```

**What to check:**
- ✅ No import errors or crashes
- ✅ Model parameter count should be displayed (~70M)
- ✅ Loss should decrease from ~10-11 to ~8-9 within 50 steps
- ✅ Validation runs without errors

---

### Step 2: Train the Hybrid 70M Model

This is the **main experiment** — the hybrid Mamba-xLSTM architecture:

```bash
python scripts/train.py \
    model=hybrid_70m \
    dataset=wikitext \
    trainer=a100_single_gpu \
    trainer.max_steps=10000 \
    trainer.val_check_interval=500 \
    trainer.log_every_n_steps=25 \
    trainer.accumulate_grad_batches=2 \
    trainer.enable_checkpointing=true \
    dataset.batch_size=32 \
    dataset.eval_batch_size=32 \
    dataset.num_workers=4 \
    dataset.max_length=2048 \
    callbacks.checkpoint.every_n_train_steps=2500 \
    callbacks.checkpoint.save_top_k=3 \
    callbacks.checkpoint.save_last=true \
    experiment_name=hybrid_70m_wikitext \
    wandb.enabled=false
```

**Windows CMD (single line):**

```cmd
python scripts/train.py model=hybrid_70m dataset=wikitext trainer=a100_single_gpu trainer.max_steps=10000 trainer.val_check_interval=500 trainer.log_every_n_steps=25 trainer.accumulate_grad_batches=2 trainer.enable_checkpointing=true dataset.batch_size=32 dataset.eval_batch_size=32 dataset.num_workers=4 dataset.max_length=2048 callbacks.checkpoint.every_n_train_steps=2500 callbacks.checkpoint.save_top_k=3 callbacks.checkpoint.save_last=true experiment_name=hybrid_70m_wikitext wandb.enabled=false
```

**Output location:** `outputs/hybrid_70m_wikitext/`

---

### Step 3: Train Mamba-only 70M Baseline

```bash
python scripts/train.py \
    model=mamba_70m_baseline \
    dataset=wikitext \
    trainer=a100_single_gpu \
    trainer.max_steps=10000 \
    trainer.val_check_interval=500 \
    trainer.log_every_n_steps=25 \
    trainer.accumulate_grad_batches=2 \
    trainer.enable_checkpointing=true \
    dataset.batch_size=32 \
    dataset.eval_batch_size=32 \
    dataset.num_workers=4 \
    dataset.max_length=2048 \
    callbacks.checkpoint.every_n_train_steps=2500 \
    callbacks.checkpoint.save_top_k=3 \
    callbacks.checkpoint.save_last=true \
    experiment_name=mamba_70m_wikitext \
    wandb.enabled=false
```

**Windows CMD (single line):**

```cmd
python scripts/train.py model=mamba_70m_baseline dataset=wikitext trainer=a100_single_gpu trainer.max_steps=10000 trainer.val_check_interval=500 trainer.log_every_n_steps=25 trainer.accumulate_grad_batches=2 trainer.enable_checkpointing=true dataset.batch_size=32 dataset.eval_batch_size=32 dataset.num_workers=4 dataset.max_length=2048 callbacks.checkpoint.every_n_train_steps=2500 callbacks.checkpoint.save_top_k=3 callbacks.checkpoint.save_last=true experiment_name=mamba_70m_wikitext wandb.enabled=false
```

**Output location:** `outputs/mamba_70m_wikitext/`

---

### Step 4: Train xLSTM-only 70M Baseline

```bash
python scripts/train.py \
    model=xlstm_70m_baseline \
    dataset=wikitext \
    trainer=a100_single_gpu \
    trainer.max_steps=10000 \
    trainer.val_check_interval=500 \
    trainer.log_every_n_steps=25 \
    trainer.accumulate_grad_batches=2 \
    trainer.enable_checkpointing=true \
    dataset.batch_size=32 \
    dataset.eval_batch_size=32 \
    dataset.num_workers=4 \
    dataset.max_length=2048 \
    callbacks.checkpoint.every_n_train_steps=2500 \
    callbacks.checkpoint.save_top_k=3 \
    callbacks.checkpoint.save_last=true \
    experiment_name=xlstm_70m_wikitext \
    wandb.enabled=false
```

**Windows CMD (single line):**

```cmd
python scripts/train.py model=xlstm_70m_baseline dataset=wikitext trainer=a100_single_gpu trainer.max_steps=10000 trainer.val_check_interval=500 trainer.log_every_n_steps=25 trainer.accumulate_grad_batches=2 trainer.enable_checkpointing=true dataset.batch_size=32 dataset.eval_batch_size=32 dataset.num_workers=4 dataset.max_length=2048 callbacks.checkpoint.every_n_train_steps=2500 callbacks.checkpoint.save_top_k=3 callbacks.checkpoint.save_last=true experiment_name=xlstm_70m_wikitext wandb.enabled=false
```

**Output location:** `outputs/xlstm_70m_wikitext/`

---

### Step 5: Evaluate All Models

After training, evaluate each model on the WikiText-103 **test set**:

#### 5a. Evaluate Hybrid 70M

```bash
python scripts/evaluate_lm.py \
    --checkpoint=outputs/hybrid_70m_wikitext/checkpoints/last.ckpt \
    --model-config=hybrid_70m \
    --dataset=wikitext \
    --split=test \
    --batch-size=32 \
    --throughput \
    --output-dir=outputs/hybrid_70m_wikitext/eval_results
```

**Windows CMD (single line):**

```cmd
python scripts/evaluate_lm.py --checkpoint=outputs/hybrid_70m_wikitext/checkpoints/last.ckpt --model-config=hybrid_70m --dataset=wikitext --split=test --batch-size=32 --throughput --output-dir=outputs/hybrid_70m_wikitext/eval_results
```

#### 5b. Evaluate Mamba-only 70M

```cmd
python scripts/evaluate_lm.py --checkpoint=outputs/mamba_70m_wikitext/checkpoints/last.ckpt --model-config=mamba_70m_baseline --dataset=wikitext --split=test --batch-size=32 --throughput --output-dir=outputs/mamba_70m_wikitext/eval_results
```

#### 5c. Evaluate xLSTM-only 70M

```cmd
python scripts/evaluate_lm.py --checkpoint=outputs/xlstm_70m_wikitext/checkpoints/last.ckpt --model-config=xlstm_70m_baseline --dataset=wikitext --split=test --batch-size=32 --throughput --output-dir=outputs/xlstm_70m_wikitext/eval_results
```

#### 5d. Evaluate with Text Generation (Optional)

Add `--generate` to see sample text outputs:

```cmd
python scripts/evaluate_lm.py --checkpoint=outputs/hybrid_70m_wikitext/checkpoints/last.ckpt --model-config=hybrid_70m --dataset=wikitext --split=test --batch-size=32 --throughput --generate --output-dir=outputs/hybrid_70m_wikitext/eval_results
```

---

### Step 6: Compare Results

Print a comparison table of all evaluated models:

```bash
python scripts/run_70m_experiments.py --phase 5
```

Or view the individual JSON result files:

```bash
# Windows CMD
type outputs\hybrid_70m_wikitext\eval_results\results.json
type outputs\mamba_70m_wikitext\eval_results\results.json
type outputs\xlstm_70m_wikitext\eval_results\results.json

# Linux/macOS
cat outputs/hybrid_70m_wikitext/eval_results/results.json
```

---

## 6. Using the Automated Pipeline

The `run_70m_experiments.py` script automates all the steps above:

### Command Reference

| Command | Description | ~Time on A100 |
|---|---|---|
| `python scripts/run_70m_experiments.py --phase 0` | Sanity check only | 2-3 min |
| `python scripts/run_70m_experiments.py --phase 1` | Train Hybrid 70M only | 2-3 hours |
| `python scripts/run_70m_experiments.py --phase 2` | Train Mamba-only 70M only | 2-3 hours |
| `python scripts/run_70m_experiments.py --phase 3` | Train xLSTM-only 70M only | 2-3 hours |
| `python scripts/run_70m_experiments.py --phase 4` | Evaluate all models | 15-30 min |
| `python scripts/run_70m_experiments.py --phase 5` | Print comparison table | instant |
| `python scripts/run_70m_experiments.py` | **Run ALL phases** | 6-10 hours |

### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--phase N` | `-1` (all) | Run specific phase (0-5) |
| `--max-steps N` | `10000` | Training steps per model |
| `--batch-size N` | `32` | Batch size per GPU |
| `--wandb` | `false` | Enable W&B logging |
| `--dry-run` | `false` | Show commands only |
| `--generate` | `false` | Generate text during eval |

### Recommended Step Counts

| Purpose | `--max-steps` | Time on A100 | Expected PPL |
|---|---|---|---|
| Quick test | 500 | ~30 min | ~200-500 |
| Development | 5000 | ~3 hours | ~50-100 |
| Standard run | 10000 | ~7 hours | ~30-60 |
| Paper quality | 50000 | ~35 hours | ~20-40 |

---

## 7. Configuration Reference

### 70M Model Architecture

```yaml
# configs/model/hybrid_70m.yaml
dim: 512              # Model dimension
num_layers: 8         # Total layers
layer_pattern: ["mamba", "mamba", "mlstm"]  # 2:1 Mamba:xLSTM ratio
state_size: 16        # SSM state dimension
head_dim: 64          # Attention head dim
num_heads: 8          # 512 / 64 = 8 heads
max_position_embeddings: 2048
```

### A100 Trainer Settings

```yaml
# configs/trainer/a100_single_gpu.yaml
precision: "bf16-mixed"           # A100 Tensor Core bf16
accumulate_grad_batches: 2        # Effective batch = batch_size × 2
max_steps: 10000                  # Overridable via CLI
val_check_interval: 500           # Validate every 500 steps
```

### Key Hydra Overrides

You can override any config value from the command line:

```bash
# Change batch size
python scripts/train.py model=hybrid_70m dataset.batch_size=16

# Change learning rate
python scripts/train.py model=hybrid_70m model.learning_rate=3e-4

# Change sequence length
python scripts/train.py model=hybrid_70m dataset.max_length=1024

# Use different precision
python scripts/train.py model=hybrid_70m trainer.precision=32

# Disable W&B but enable TensorBoard (default)
python scripts/train.py model=hybrid_70m wandb.enabled=false
```

---

## 8. Adjusting for Different GPUs

### A100 80GB (Recommended)

```bash
python scripts/run_70m_experiments.py --batch-size 32
```

### A100 40GB

```bash
python scripts/run_70m_experiments.py --batch-size 16
```

### RTX 4090 / RTX 3090 (24GB)

```bash
python scripts/run_70m_experiments.py --batch-size 8
```

Or manually with fp16 instead of bf16:

```bash
python scripts/train.py \
    model=hybrid_70m \
    dataset=wikitext \
    trainer=single_gpu \
    trainer.precision="16-mixed" \
    trainer.max_steps=10000 \
    dataset.batch_size=8 \
    experiment_name=hybrid_70m_wikitext
```

### V100 (16GB)

```bash
python scripts/train.py \
    model=hybrid_70m \
    dataset=wikitext \
    trainer=single_gpu \
    trainer.precision="16-mixed" \
    trainer.max_steps=10000 \
    dataset.batch_size=4 \
    dataset.max_length=1024 \
    trainer.accumulate_grad_batches=8 \
    experiment_name=hybrid_70m_wikitext
```

### Google Colab (T4 16GB)

```bash
python scripts/train.py \
    model=hybrid_70m \
    dataset=wikitext \
    trainer=colab_single_gpu \
    trainer.max_steps=5000 \
    dataset.batch_size=4 \
    dataset.max_length=1024 \
    experiment_name=hybrid_70m_colab
```

---

## 9. Monitoring Training

### TensorBoard

```bash
# Start TensorBoard (run in a separate terminal)
tensorboard --logdir outputs/ --port 6006

# Then open in browser: http://localhost:6006
```

### Key Metrics to Watch

| Metric | Location | Expected Behavior |
|---|---|---|
| `train/loss` | TensorBoard / console | Should decrease steadily |
| `train/perplexity` | TensorBoard / console | Should decrease steadily |
| `val/loss` | TensorBoard / console | Should decrease, plateau |
| `val/perplexity` | TensorBoard / console | Should decrease, plateau |
| `train/lr` | TensorBoard | Warmup → cosine decay |
| `train/grad_norm` | TensorBoard | Should stay below 1.0 |

### Weights & Biases (Optional)

```bash
# Login to W&B first
wandb login

# Then run with --wandb flag
python scripts/run_70m_experiments.py --wandb
```

---

## 10. Troubleshooting

### Common Issues

| Issue | Solution |
|---|---|
| `CUDA out of memory` | Reduce `batch_size` (e.g., 16 → 8 → 4) |
| `ModuleNotFoundError: hybrid_xmamba` | Run `pip install -e .` from project root |
| `FileNotFoundError: configs/...` | Ensure you're running from the project root directory |
| `tokenizers warning` | Safe to ignore; tokenizer model_max_length is handled |
| `No checkpoint found` | Training must complete before evaluation |
| Training loss stuck/NaN | Try reducing `learning_rate` to `3e-4` |
| Slow data loading | Reduce `num_workers` on Windows, or increase on Linux |

### Checking GPU Status

```bash
# Check GPU usage
nvidia-smi

# Watch GPU usage in real-time (refreshes every 1s)
nvidia-smi -l 1

# or on Windows
nvidia-smi.exe -l 1
```

### Resuming Training from Checkpoint

If training is interrupted, resume from the last checkpoint:

```bash
# Lightning automatically resumes if the checkpoint exists
# Just re-run the same command — it finds last.ckpt automatically

# Or specify explicitly:
python scripts/train.py \
    model=hybrid_70m \
    dataset=wikitext \
    trainer=a100_single_gpu \
    trainer.max_steps=10000 \
    experiment_name=hybrid_70m_wikitext
```

---

## 11. Scaling Up to 150M

Once the 70M experiments validate your setup, scale up to 150M:

```bash
# Automated 150M pipeline (same structure, larger model)
python scripts/run_a100_experiments.py --max-steps 10000

# Or manual training of 150M hybrid
python scripts/train.py \
    model=hybrid_150m \
    dataset=wikitext \
    trainer=a100_single_gpu \
    trainer.max_steps=10000 \
    dataset.batch_size=32 \
    experiment_name=hybrid_150m_wikitext
```

### Model Scale Comparison

| Scale | dim | layers | Params | ~Time/10k steps (A100) |
|---|---|---|---|---|
| **70M** | 512 | 8 | ~70M | ~2-3 hours |
| **150M** | 768 | 12 | ~150M | ~5-6 hours |
| **350M** | 1024 | 24 | ~350M | ~12-15 hours |

---

## Complete Command Reference Cheat Sheet

```bash
# ============================================
# FULL AUTOMATED PIPELINE
# ============================================
python scripts/run_70m_experiments.py                       # All phases
python scripts/run_70m_experiments.py --dry-run              # Preview commands
python scripts/run_70m_experiments.py --max-steps 50000      # Paper quality
python scripts/run_70m_experiments.py --phase 0              # Sanity check
python scripts/run_70m_experiments.py --phase 1              # Train hybrid
python scripts/run_70m_experiments.py --phase 2              # Train mamba baseline
python scripts/run_70m_experiments.py --phase 3              # Train xlstm baseline
python scripts/run_70m_experiments.py --phase 4              # Evaluate all
python scripts/run_70m_experiments.py --phase 5              # Print results

# ============================================
# INDIVIDUAL TRAINING (manual control)
# ============================================
python scripts/train.py model=hybrid_70m dataset=wikitext trainer=a100_single_gpu trainer.max_steps=10000 experiment_name=hybrid_70m_wikitext
python scripts/train.py model=mamba_70m_baseline dataset=wikitext trainer=a100_single_gpu trainer.max_steps=10000 experiment_name=mamba_70m_wikitext
python scripts/train.py model=xlstm_70m_baseline dataset=wikitext trainer=a100_single_gpu trainer.max_steps=10000 experiment_name=xlstm_70m_wikitext

# ============================================
# EVALUATION
# ============================================
python scripts/evaluate_lm.py --checkpoint=outputs/hybrid_70m_wikitext/checkpoints/last.ckpt --model-config=hybrid_70m --dataset=wikitext --split=test --batch-size=32 --throughput --output-dir=outputs/hybrid_70m_wikitext/eval_results
python scripts/evaluate_lm.py --checkpoint=outputs/mamba_70m_wikitext/checkpoints/last.ckpt --model-config=mamba_70m_baseline --dataset=wikitext --split=test --batch-size=32 --throughput --output-dir=outputs/mamba_70m_wikitext/eval_results
python scripts/evaluate_lm.py --checkpoint=outputs/xlstm_70m_wikitext/checkpoints/last.ckpt --model-config=xlstm_70m_baseline --dataset=wikitext --split=test --batch-size=32 --throughput --output-dir=outputs/xlstm_70m_wikitext/eval_results

# ============================================
# PROFILING
# ============================================
python scripts/performance_profile.py --model hybrid_70m --batch_size 8 --seq_length 2048

# ============================================
# MONITORING
# ============================================
tensorboard --logdir outputs/ --port 6006
```

---

## Output Directory Structure (After Training)

```
outputs/
├── hybrid_70m_wikitext/
│   ├── checkpoints/
│   │   ├── last.ckpt                        # Last checkpoint
│   │   ├── checkpoint-epoch00-val_loss_X.ckpt
│   │   └── ...
│   ├── logs/
│   │   └── tensorboard/                     # TensorBoard logs
│   └── eval_results/
│       └── results.json                     # Evaluation metrics
├── mamba_70m_wikitext/
│   ├── checkpoints/
│   ├── logs/
│   └── eval_results/
│       └── results.json
└── xlstm_70m_wikitext/
    ├── checkpoints/
    ├── logs/
    └── eval_results/
        └── results.json
```

---

*Generated for the Hybrid Mamba-xLSTM thesis project*
