# Hybrid Mamba-xLSTM Model

A research implementation of a hybrid architecture combining **Mamba** (Selective SSM) and **xLSTM**
(mLSTM with matrix memory) for efficient sequence modeling, applied to chest-X-ray report generation.

---

## 🔬 This branch: `h100_scaling_mamba3` — Mamba-3 backbone upgrade

**Status: M0–M3 complete, 32/62 checkboxes.** Plan of record: [`MAMBA3_PLAN.md`](MAMBA3_PLAN.md);
live state: [`mamba3_state.json`](mamba3_state.json). Branched from `h100_scaling` @ `20a1d27`.
**This branch is never merged without an explicit instruction** — `h100_scaling` keeps the approved
results reproducible.

### Why this branch exists

The work started as "should we adopt [Mamba-3](https://arxiv.org/abs/2603.15569)?" and turned up a
**correctness defect** on the way in. Both chunked recurrences computed
`A_cum * cumsum(Bx / A_cum.clamp(eps))`; wherever the clamp fires, a token's own contribution to the
state is *annihilated* rather than perturbed. Measured against float64 sequential references:

| Operator | Configuration | rel-max-err |
|---|---|---|
| Selective scan | Δ = 0.705 (the value this repo actually initializes to) | **0.92** |
| Selective scan | Δ = 0.1 (`dt_max` of the *correct* init) | 0.39 |
| mLSTM / TFLA | shipped `forget_gate_bias_init=0.0`, `chunk_size=64` | **0.882** |
| Δ at init | `hybrid_150m_v2` canonical vs reference `logU[1e-3, 1e-1]` | 0.807 vs ~0.021 |

All 12 layers of the canonical 150M model were running a recurrence that was not the one specified.
This does **not** invalidate published numbers — training and evaluation used the same operator, so
they are valid measurements of the system as built. The narrower true claim is that the block did not
compute the recurrence it was specified to compute.

### Why SSD, and not just a patch

The exact log-space fix is affordable in Mamba-2/3's parameterization and not in Mamba-1's:

| Form | Exact-mask memory (bs 48, L 1024) | Shape |
|---|---|---|
| Mamba-1, `A` is `(d_inner, d_state)` | **19.3 GB** | elementwise |
| **Mamba-2/3, scalar `A` per head** | **19 MB** | matmul (tensor cores) |

That, rather than any reported quality gain, is the load-bearing argument. On top of it, Mamba-3 SISO
is **parameter-matched to +0.26%** while carrying **8× the SSM state** (B/C are shared across heads):

| Config | Params | SSM state / layer |
|---|---|---|
| `hybrid_150m_v2` (control) | 183,721,824 | 24,576 |
| `hybrid_150m_a1` (defect fix only) | 183,708,000 | 24,576 |
| **`hybrid_150m_m3`** (Mamba-3) | **184,192,200** | **196,608** |

### What's landed

- **`hybrid_xmamba/kernels/ssd/`** — chunked SSD scan, a float64 sequential oracle, and `ssd_step`
  (the same routine the M6 decode cache will call). Exact to ~4e-16 across shapes, chunk sizes and
  packed-document layouts.
- **`hybrid_xmamba/layers/mamba3_block.py`** — `Mamba3Block`, registered as a fourth `layer_pattern`
  type. **With every flag off it is exactly Mamba-2 SSD**; each Mamba-3 feature (trapezoidal
  discretization, complex/RoPE state, B/C biases, conv-drop) sits behind a flag that reduces to that
  baseline, so every ablation arm moves one variable.
- **`scan_impl` / `tfla_impl` flags** (`"legacy"` | `"exact"`), both defaulting to `"legacy"` so every
  pre-2026-09 number stays bit-reproducible. Defaults flip at M9.
- **`tests/test_mamba3_numerics.py`** — float64 oracles for both operators, CPU-collected and
  unconditionally run. The old kernel tests asserted only shape/NaN *and* were CUDA-gated, which is
  how the defect survived.
- **Two live bugs fixed in passing**: sLSTM was silently leaking recurrent state across packed
  documents (`cu_seqlens` was dispatched on a hard-coded layer-name tuple), and Gate 6 of the
  pre-push harness only ever exercised `use_fast_path=False` — the branch that carried an undetected
  copy of the scan defect.

### Running the screen arms

**Locally:**

```bash
git checkout h100_scaling_mamba3
python -m pytest tests/ -m "not cuda and not slow" -q     # 267 passed, 21 xfailed
bash scripts/validate_for_willi.sh                        # 9/9 gates
```

**On the aisc H100 cluster.** The login node refuses to execute anything, `python` included, so the
pre-flight is an sbatch job rather than a one-liner:

```bash
git fetch origin && git checkout h100_scaling_mamba3
sbatch scripts/preflight_mamba3_h100.sh     # ~2 min, CPU only; verifies every arm builds
                                            # the operator it claims and A1's Delta is ~0.02
tail -f logs/mamba3_preflight_*.log

# then the arms — A0 (control), A0-seed (noise floor), A1 (defect fix)
MODEL_CONFIG=hybrid_150m_v2 MAX_STEPS=12000 WARMUP=500 SEED=42 SAVE_TOP_K=1 \
  EXPERIMENT=m3_screen_A0_s42 sbatch --time=12:00:00 scripts/train_stage0_150m_h100.sh
MODEL_CONFIG=hybrid_150m_a1 MAX_STEPS=12000 WARMUP=500 SEED=42 SAVE_TOP_K=1 \
  EXPERIMENT=m3_screen_A1_s42 sbatch --time=12:00:00 scripts/train_stage0_150m_h100.sh

grep ARCH logs/<job>.log   # confirm the operator that is actually training
```

Every model logs an architecture fingerprint at construction, so "is this really Mamba-3?" is
answerable from the log rather than by inference:

```
ARCH layers=[mamba3x9, mlstmx3] | norm_topology=hybrid | scan_impl=legacy | tfla_impl=legacy |
     dt_init=none | mamba3(d_state=128, head_dim=64, ngroups=1, conv=True, trapezoid=False,
     rope=False, bc_bias=none, a_mode=static, mimo_rank=1) | params=184,192,200
```

### Baselines any arm must beat

Official MIMIC-CXR-JPG test split, n=2663, beam=3 (from `analysis/h100_scaling_results.md`):

| Metric | Generator (13D) | Retrieval-NN floor |
|---|---|---|
| ROUGE-L | 0.1899 | 0.1636 |
| CheXbert-14 micro | 0.4736 | 0.4296 |
| CheXbert-14 macro | 0.2800 | **0.3014** |
| Stage-0 val PPL (PubMed) | 13.18 | — |

**Honest caveat carried in the plan:** this project has ten documented text-side nulls, every one
measured against *retrieval*, where the text tower is an encoder anchored to a frozen teacher. The
objective is now *generation*, where the Mamba/mLSTM stack is the generator — so the null record does
not transfer, but neither is a win assumed. That is what the M7 screen measures.

### Progress

| Phase | | Status |
|---|---|---|
| M0 | Branch, plan-of-record, bootstrap | ✅ 6/6 |
| M1 | Pin the defect, then fix it on the legacy path (produces arm A1) | ✅ 9/9 |
| M2 | Mamba3Block = exactly Mamba-2 SSD (+ the sequential oracle) | ✅ 10/10 |
| M3 | Exponential-trapezoidal | ✅ 3/3 |
| M4 | Complex-valued state (RoPE trick) | 🔄 4/5 |
| M5 | Flags folded into arm definitions (no milestone of its own) | ⬜ 0/3 |
| M6 | O(1) recurrent decode cache | ⬜ 0/5 |
| M7 | Timing probe + short-run screen  — H100 | ⬜ 0/8 |
| M8 | Full pipeline on the winner — H100 | ⬜ 0/5 |
| M9 | Cleanup, writeup, reintegration | ⬜ 0/5 |
| M10 | Re-open the mamba/mLSTM ratio | ⬜ 0/3 |

---

## Architecture Overview

The hybrid model interleaves Mamba and mLSTM layers in a repeating pattern `[mamba, mamba, mlstm]`:

- **Mamba layers**: Selective State Space Model with input-dependent gating, 1D causal convolution, and SiLU activation. Handles local/sequential patterns via selective scan.
- **mLSTM layers**: Matrix LSTM with exponential gating, matrix-valued cell state ($C_t \in \mathbb{R}^{D \times D}$), and Tiled Flash Linear Attention (TFLA) kernel. Handles long-range associative recall.

| Model | Dim | Layers | Pattern | Params |
|-------|-----|--------|---------|--------|
| `hybrid_70m` | 512 | 8 | [mamba, mamba, mlstm] | ~70M |
| `hybrid_150m_v2` | 768 | 12 | 9 × mamba + 3 × mlstm (centred) | 183.7M |
| **`hybrid_150m_m3`** | **768** | **12** | **9 × mamba3 + 3 × mlstm** | **184.2M** |
| `hybrid_150m_a1` | 768 | 12 | 9 × mamba (defects repaired) | 183.7M |
| `hybrid_350m` | 1024 | 24 | [mamba, mamba, mlstm] | ~350M |

Valid `layer_pattern` entries: `mamba`, `mamba3`, `mlstm`, `slstm`.

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
│   ├── kernels/               # selective scan, TFLA, SSD — pure PyTorch, not Triton
│   │                          #   (scan_triton.py is dead code; deleted at M9-C)
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

