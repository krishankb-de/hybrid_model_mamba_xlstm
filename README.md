# Hybrid Mamba-xLSTM Model

A comprehensive implementation of a hybrid architecture combining **Mamba** (Selective SSM) and **xLSTM** (mLSTM with matrix memory) for efficient sequence modeling. This branch contains the **150M parameter** model configurations and A100-optimized training pipeline for research paper experiments.

## Architecture Overview

The hybrid model interleaves Mamba and mLSTM layers in a repeating pattern `[mamba, mamba, mlstm]`:

- **Mamba layers**: Selective State Space Model with input-dependent gating, 1D causal convolution, and SiLU activation. Handles local/sequential patterns via selective scan.
- **mLSTM layers**: Matrix LSTM with exponential gating, matrix-valued cell state ($C_t \in \mathbb{R}^{D \times D}$), and Tiled Flash Linear Attention (TFLA) kernel. Handles long-range associative recall.

| Model | Dim | Layers | Pattern | Params |
|-------|-----|--------|---------|--------|
| `hybrid_70m` | 512 | 8 | [mamba, mamba, mlstm] | ~70M |
| **`hybrid_150m`** | **768** | **12** | **[mamba, mamba, mlstm]** | **~150M** |
| `hybrid_350m` | 1024 | 24 | [mamba, mamba, mlstm] | ~350M |

## Project Structure

```
hybrid-xmamba/
├── configs/                    # Hydra configuration management
│   ├── model/                 # Model configs (hybrid, mamba, xlstm baselines)
│   ├── dataset/               # Dataset configs (wikitext, c4, mqar)
│   └── trainer/               # Trainer configs (single_gpu, a100, fsdp)
├── hybrid_xmamba/             # Core Python package
│   ├── models/                # Model definitions & config dataclass
│   ├── layers/                # Mamba, mLSTM, sLSTM, hybrid blocks
│   ├── kernels/               # Triton kernels (selective scan, TFLA)
│   ├── training/              # Lightning module, optimizer, metrics
│   └── utils/                 # Registry, generation, initialization
├── scripts/                   # Training, evaluation, experiment runner
├── tests/                     # Unit and integration tests
├── requirements.txt           # Dependencies
└── setup.py                   # Installation script
```

## Features

- **Hybrid Architecture**: Flexible interleaving of Mamba, mLSTM, and sLSTM layers
- **Chunk-Parallel Kernels** (no sequential for-loops):
  - **TFLA (mLSTM)**: Chunk-parallel linear attention with inter-chunk recurrence (~32 steps for seq_len=2048 vs 2048 sequential)
  - **Selective Scan (Mamba)**: Chunk-parallel scan with cumulative products (~32 steps instead of 2048)
  - **sLSTM**: Parallel scan via cumulative forget-gate products in log-space
- **A100 Optimized**: bf16 mixed precision, torch.compile, TF32 matmul, fused optimizer
- **Scalable Training**: Support for FSDP, DDP with PyTorch Lightning
- **Hydra Configs**: Modular YAML configuration with CLI overrides
- **Multiple Benchmarks**: WikiText-103, C4, MQAR (Multi-Query Associative Recall)

---

## Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/Hybrid_Model_Mamba_xLSTM.git
cd Hybrid_Model_Mamba_xLSTM
git checkout a100_150m_baseline

# Install dependencies
pip install -e .
pip install -r requirements.txt
```

### Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.1 (with CUDA support)
- PyTorch Lightning ≥ 2.1
- Triton ≥ 2.1 (for GPU kernels, auto-installed with PyTorch)
- NVIDIA GPU with ≥ 40GB VRAM (A100 recommended)

---

## Quick Start (A100 GPU — Lightning AI Studio)

### Step 0: Sanity Check (~5 minutes)

Verify everything works before committing to a long training run:

```bash
python scripts/run_a100_experiments.py --phase 0 --batch-size 32
```

This runs 50 steps of the hybrid model and confirms data loading, model forward/backward, and checkpointing all work.

### Step 1: Train the Hybrid Model (~5–6 hours)

```bash
python scripts/train.py \
    model=hybrid_150m \
    dataset=wikitext \
    trainer=a100_single_gpu \
    trainer.max_steps=10000 \
    dataset.batch_size=32 \
    dataset.max_length=2048 \
    experiment_name=hybrid_150m_wikitext
```

### Step 2: Train the Mamba-Only Baseline (~5–6 hours)

```bash
python scripts/train.py \
    model=mamba_150m_baseline \
    dataset=wikitext \
    trainer=a100_single_gpu \
    trainer.max_steps=10000 \
    dataset.batch_size=32 \
    dataset.max_length=2048 \
    experiment_name=mamba_150m_wikitext
```

### Step 3: Train the xLSTM-Only Baseline (~5–6 hours)

```bash
python scripts/train.py \
    model=xlstm_150m_baseline \
    dataset=wikitext \
    trainer=a100_single_gpu \
    trainer.max_steps=10000 \
    dataset.batch_size=32 \
    dataset.max_length=2048 \
    experiment_name=xlstm_150m_wikitext
```

### Step 4: Evaluate All Models

```bash
# Evaluate each checkpoint on WikiText-103 test set
python scripts/evaluate_lm.py \
    --checkpoint outputs/hybrid_150m_wikitext/checkpoints/last.ckpt \
    --model-config hybrid_150m \
    --dataset wikitext --split test \
    --batch-size 32 --throughput --generate \
    --output-dir outputs/hybrid_150m_wikitext/eval_results

python scripts/evaluate_lm.py \
    --checkpoint outputs/mamba_150m_wikitext/checkpoints/last.ckpt \
    --model-config mamba_150m_baseline \
    --dataset wikitext --split test \
    --batch-size 32 --throughput \
    --output-dir outputs/mamba_150m_wikitext/eval_results

python scripts/evaluate_lm.py \
    --checkpoint outputs/xlstm_150m_wikitext/checkpoints/last.ckpt \
    --model-config xlstm_150m_baseline \
    --dataset wikitext --split test \
    --batch-size 32 --throughput \
    --output-dir outputs/xlstm_150m_wikitext/eval_results
```

### Automated Pipeline (All Phases)

Run all experiments end-to-end with a single command:

```bash
# Full pipeline: sanity check → train 3 models → evaluate → comparison table
python scripts/run_a100_experiments.py --max-steps 10000 --batch-size 32

# With Weights & Biases logging
python scripts/run_a100_experiments.py --max-steps 10000 --batch-size 32 --wandb

# Dry run (print commands without executing)
python scripts/run_a100_experiments.py --dry-run
```

---

## A100 Training Configuration

The `a100_single_gpu` trainer config is optimized for NVIDIA A100 80GB:

| Setting | Value | Rationale |
|---------|-------|-----------|
| Precision | `bf16-mixed` | 4–5× faster matmuls via A100 Tensor Cores |
| Batch size | 32 | Fits in 80GB VRAM with 150M model at seq_len=2048 |
| Grad accumulation | 2 | Effective batch = 64 |
| Sequence length | 2048 | Full context window, text packing (no padding waste) |
| Optimizer | AdamW (fused) | Faster than standard AdamW on CUDA |
| LR schedule | Cosine with warmup (1000 steps) | Standard for LM training |
| Learning rate | 6e-4 | Chinchilla-optimal range for ~150M models |
| `torch.compile` | Enabled | Fuses ops for additional speedup |

### Estimated Training Times (A100 80GB)

| Model | Steps | Est. Time | Tokens Processed |
|-------|-------|-----------|-----------------|
| Hybrid 150M | 10,000 | ~5–6 hours | ~1.3B tokens |
| Mamba 150M | 10,000 | ~5–6 hours | ~1.3B tokens |
| xLSTM 150M | 10,000 | ~5–6 hours | ~1.3B tokens |
| **Total** | **30,000** | **~16–18 hours** | **~4B tokens** |

---

## Model Configs for Fair Comparison

All three 150M models share identical hyperparameters — only the `layer_pattern` differs:

| Config | File | Layer Pattern |
|--------|------|--------------|
| Hybrid | `configs/model/hybrid_150m.yaml` | `[mamba, mamba, mlstm]` |
| Mamba-only | `configs/model/mamba_150m_baseline.yaml` | `[mamba]` |
| xLSTM-only | `configs/model/xlstm_150m_baseline.yaml` | `[mlstm]` |

All use: dim=768, layers=12, vocab=50257, lr=6e-4, weight_decay=0.1, warmup=1000 steps.

---

## Evaluation Metrics

The evaluation script (`scripts/evaluate_lm.py`) computes:

1. **Test Perplexity** — Standard LM benchmark on WikiText-103 test set
2. **Bits-per-Byte (BPB)** — Cross-entropy loss / ln(2)
3. **Inference Throughput** — Tokens/second at various sequence lengths
4. **Peak GPU Memory** — Maximum VRAM during evaluation
5. **Text Generation** — Qualitative samples with top-k/top-p sampling

### Published Baselines (WikiText-103, for reference)

| Model | Params | Test PPL |
|-------|--------|----------|
| GPT-2 | 117M | ~29.4 |
| Transformer-XL | 151M | ~18.3 |
| MEGA | 128M | ~17.3 |
| Standard Transformer ~150M | — | ~24–28 |

> **Note**: Published numbers vary by training setup. The most valid comparison is between your three models trained identically under the same conditions.

---

## Overriding Configuration

Hydra allows any config value to be overridden via CLI:

```bash
# Change learning rate and batch size
python scripts/train.py model=hybrid_150m dataset=wikitext trainer=a100_single_gpu \
    model.learning_rate=3e-4 dataset.batch_size=16

# Use C4 dataset instead of WikiText
python scripts/train.py model=hybrid_150m dataset=c4 trainer=a100_single_gpu

# Train with DDP on multiple GPUs
python scripts/train.py model=hybrid_150m dataset=wikitext trainer=gpu_ddp
```

---

