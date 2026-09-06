# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Session Bootstrap (READ FIRST)

**Branch `h100_scaling_mamba3` (ACTIVE): read `MAMBA3_PLAN.md` + `mamba3_state.json` at repo root FIRST.** They are the active resumable plan-of-record. Resume at `mamba3_state.json["current_phase"]`; the checkboxes in `MAMBA3_PLAN.md` are ground truth. After **every** meaningful change (test written, phase gated, job submitted, job finished, eval scored) tick the checkbox AND update `mamba3_state.json["last_updated"]` + append a one-line `notes` entry + record the evidence (job id, log path, metric) under `phases[<id>].evidence`. Use `python scripts/mamba3_state.py tick|phase|note|readme|show` rather than hand-editing both files, and run `readme` at the end of every phase so `README.md`'s branch section stays current. If `mamba3_state.json` is lost, regenerate it with `scripts/mamba3_state.py sync` from the plan's checkboxes.

**What that plan is.** A verified correctness defect: the live selective scan computes `A_cum * cumsum(Bx / A_cum.clamp(1e-8))`, which annihilates a token's own contribution to the state when the clamp fires — rel-max-err **0.92** at the measured init Δ=0.705 (`scan_interface.py:118-119`, the same bug in `mamba_block.py:_slow_forward`, and the same class in `tfla_interface.py:93-95` where **70.9%** of entries hit the clamp at the shipped forget-gate init). The exact log-space fix costs 19.3 GB in Mamba-1's `A=(d_inner,N)` form but **19 MB** in Mamba-2/3's scalar-`A`-per-head form, so migrating to SSD is what makes correctness affordable; Mamba-3 SISO on top is parameter-matched to **+0.24%** while giving **8× the SSM state**. Screened by Stage-0 PPL, decided by report generation on the official n=2663 test split. **⚠ NO MERGE into `h100_scaling` without an explicit instruction from the user** — not on plan completion, not on a passing gate.

**`H100_SCALING_PLAN.md` + `h100_scaling_state.json` are CLOSED (`current_phase: plan_closed`) and are now the BASELINE, not the active plan.** Do not resume against them. Their results are what any Mamba-3 arm must beat: report-gen on the official test split (n=2663, 13D checkpoint, beam=3) **ROUGE-L 0.1899 / CheXbert-14-micro 0.4736** vs a retrieval-NN floor of 0.1636 / 0.4296; Stage-0 val PPL **13.18**; MIMIC i2t R@10 **0.1459** clean protocol (SE 0.57pp, a non-regression guard only — never a target: the project has 10 documented text-side nulls). Retrieval is a complete, closed supporting chapter — **do not re-open or re-run retrieval arms.** `HYBRID_ARCH_REFACTOR_PLAN.md`, `BIOMEDCLIP_KD_PLAN.md`, `JOINT_TRAINING_PLAN.md` and their state files are historical record only.

**Corrections to older claims in this file:** the Mamba path does **not** use a Triton kernel — `scan_triton.py` is imported and never called; the live scan is the pure-PyTorch chunked one in `scan_interface.py`. `mamba_block_v2.py`, `mlstm_block_v2.py`, `hybrid_layer.py`, `scan_triton.py` and `tfla_triton.py` are dead code (deleted at phase M9-C). `HybridLanguageModel.forward()` **does** accept `inputs_embeds`; image conditioning exists and is prefix-based (`models/prefix_mapper.py`).

## Project Overview

Research implementation of a **Hybrid Mamba-xLSTM Language Model** combining Mamba (Selective SSM) and xLSTM (mLSTM with matrix memory) layers. The active model target is **70M parameters** (dim=512, 8 layers). Supports 150M and 350M variants as well.

## Installation

Use a python virtual 'venv' environement to run or do any testing or any form of installations.

```bash
pip install -e .
pip install -r requirements.txt
```

Requires Python ≥ 3.9, PyTorch ≥ 2.1 with CUDA. The 70M model fits on a T4 (15GB VRAM) for validation; A100 40/80GB recommended for full training runs.



## Common Commands

### Testing
```bash
# Run all tests
pytest tests/ -v

# Run a single test file or class
pytest tests/test_models.py::TestHybridLanguageModel -v

# Skip GPU/slow tests
pytest -m "not slow and not cuda"
```

### Training

```bash
# Sanity check (~5 min, 50 steps)
python scripts/run_70m_experiments.py --phase 0 --batch-size 32

# Train hybrid 70M model
python scripts/train.py \
    model=hybrid_70m dataset=wikitext trainer=a100_single_gpu \
    trainer.max_steps=10000 dataset.batch_size=32 dataset.max_length=1024 \
    experiment_name=hybrid_70m_wikitext

# Train Mamba-only baseline
python scripts/train.py model=mamba_70m_baseline dataset=wikitext trainer=a100_single_gpu \
    trainer.max_steps=10000 dataset.batch_size=32 experiment_name=mamba_70m_wikitext

# Train xLSTM-only baseline
python scripts/train.py model=xlstm_70m_baseline dataset=wikitext trainer=a100_single_gpu \
    trainer.max_steps=10000 dataset.batch_size=32 experiment_name=xlstm_70m_wikitext

# Full automated pipeline (sanity → train 3 models → evaluate → compare table)
python scripts/run_70m_experiments.py --max-steps 10000 --batch-size 32 [--wandb] [--dry-run]

# Colab validation (T4 GPU, 50 steps to verify pipeline before A100)
python scripts/train.py model=hybrid_70m dataset=wikitext trainer=colab_single_gpu \
    trainer.max_steps=50 experiment_name=colab_sanity
```

### Evaluation

```bash
# Evaluate language model (perplexity, throughput, optional generation)
python scripts/evaluate_lm.py \
    --checkpoint outputs/hybrid_70m_wikitext/checkpoints/last.ckpt \
    --model-config hybrid_70m \
    --dataset wikitext --split test \
    --batch-size 32 --throughput --generate \
    --output-dir outputs/hybrid_70m_wikitext/eval_results

# Submit Stage 0 eval via SLURM (use on Willi — never run GPU jobs on login node)
sbatch scripts/eval_stage0_lm.sh

# Submit Stage 1 SimCSE training via SLURM
sbatch scripts/train_stage1_simcse.sh

# Evaluate retrieval benchmarks
python scripts/evaluate_retrieval.py --checkpoint <path> --model-config hybrid_70m

# Evaluate semantic textual similarity
python scripts/evaluate_sts.py --checkpoint <path> --model-config hybrid_70m
```

### Hydra Config Overrides

Any config value can be overridden via CLI:
```bash
python scripts/train.py model=hybrid_70m dataset=c4 trainer=a100_single_gpu \
    model.learning_rate=3e-4 dataset.batch_size=16
```

## Architecture

### Core Package (`hybrid_xmamba/`)

The model is built from three composable layer types interleaved via `layer_pattern`:

- **Mamba block** (`layers/mamba_block.py`): Selective SSM with input-dependent gating, 1D causal convolution, SiLU activation. Uses chunk-parallel selective scan Triton kernel.
- **mLSTM block** (`layers/mlstm_block.py`): Matrix LSTM with exponential gating, matrix-valued cell state (D×D). Uses Tiled Flash Linear Attention (TFLA) Triton kernel for chunk-parallel (~32 steps for seq_len=2048 vs 2048 sequential).
- **sLSTM block** (`layers/slstm_block.py`): Parallel scan via cumulative forget-gate products in log-space.

**Model sizes and patterns:**

| Config | Dim | Layers | `layer_pattern` | Params |
|--------|-----|--------|-----------------|--------|
| **`hybrid_70m`** | **512** | **8** | **`[mamba, mamba, mlstm]`** | **~70M** |
| `hybrid_150m` | 768 | 12 | `[mamba, mamba, mlstm]` | ~150M |
| `hybrid_350m` | 1024 | 24 | `[mamba, mamba, mlstm]` | ~350M |

The 70M model uses `max_position_embeddings=1024` (not 2048) and `num_heads=8`.

**Data flow:**
```
input_ids → Embedding → N × HybridBlock [Pre-norm → Mixer → Residual → MLP → Residual] → RMSNorm → LM Head (logits)
```

**Key files:**
- `hybrid_xmamba/models/hybrid_lm.py` — `HybridLanguageModel`: top-level model with embeddings and LM head
- `hybrid_xmamba/models/configuration_hybrid.py` — `HybridConfig` dataclass (all architecture params)
- `hybrid_xmamba/layers/hybrid_block.py` — `HybridBlock`: factory that dispatches to Mamba/mLSTM/sLSTM
- `hybrid_xmamba/kernels/selective_scan/scan_triton.py` — Triton kernel for Mamba's selective scan
- `hybrid_xmamba/kernels/tfla/tfla_triton.py` — Triton kernel for mLSTM's TFLA
- `hybrid_xmamba/training/lightning_module.py` — PyTorch Lightning training/validation loop

### Configuration System (Hydra)

Configs are composed from four directories under `configs/`:
- `model/` — architecture configs: `hybrid_70m.yaml`, `mamba_70m_baseline.yaml`, `xlstm_70m_baseline.yaml`, etc.
- `dataset/` — `wikitext.yaml`, `c4.yaml`, `fineweb.yaml`, `pubmed.yaml`, `mqar.yaml`
- `trainer/` — `a100_single_gpu.yaml` (bf16, batch=32, grad_accum=2), `colab_single_gpu.yaml` (T4-compatible), `single_gpu.yaml`, `gpu_ddp.yaml`, `gpu_fsdp.yaml`
- `callbacks/` — logging callbacks

For fair comparison, all 70M models share identical hyperparameters (dim=512, layers=8, vocab=50257, lr=6e-4, weight_decay=0.1, warmup=1000 steps, max_position_embeddings=1024) — only `layer_pattern` differs.

The A100 trainer uses bf16-mixed precision, effective batch size 64 (32 × grad_accum=2), `torch.compile`, and fused AdamW.

### Training Entry Points

- `scripts/train.py` — main Hydra training script (calls `pl.Trainer.fit()`)
- `scripts/run_70m_experiments.py` — orchestrator: sanity check → train hybrid/mamba/xlstm 70M → evaluate → print comparison
- `scripts/run_a100_experiments.py` — same for 150M variant
- `scripts/train_contrastive.py` — Stage 1 SimCSE + Stage 2 CLIP contrastive training on top of a pretrained checkpoint

### Checkpoint Compatibility

State dict keys can have prefixes from torch.compile (`_orig_mod.`) or PyTorch Lightning wrapping (`lm.`, `model.`). Evaluation scripts explicitly strip these before loading. When debugging checkpoint loading issues, use `debug_checkpoint_keys.py` and `check_checkpoint_compatibility.py` in the root directory.

### Evaluation Metrics

`scripts/evaluate_lm.py` computes: test perplexity, bits-per-byte, inference throughput (tokens/sec), peak GPU memory, and optional text generation samples.

### Contrastive / Retrieval Pipeline

`scripts/train_contrastive.py` wraps `HybridLanguageModel` in `HybridTextEncoder` with a projection head. Stage 1 uses SimCSE in-batch negatives; Stage 2 uses CLIP-style image-text pairing via BiomedCLIP. Evaluation is via `evaluate_retrieval.py` and `evaluate_sts.py`.


## Production System
The system will run on A100 GPU with max 40 GB of VRAM on the production system when everything is in order and correct. SO the final system would run on the willi server A100 GPU.

---

## Pre-Push Validation Protocol (MANDATORY)

Willi server runs **Python 3.9.23** via conda. Bugs that slip through locally (PEP 604 syntax, wrong type hints, config drift) cost hours of SLURM debugging. Follow this protocol after every edit.

### After ANY edit to `hybrid_xmamba/`, `scripts/`, or `configs/`

```bash
bash scripts/validate_for_willi.sh
```

This runs inside the `willi_parity` conda env (Python 3.9.23) and gates:
1. AST parse of all source files under Python 3.9
2. PEP 604 guard — no `X | Y` union syntax (use `Optional[X]`)
3. PEP 585 guard — no bare `dict[...]`/`list[...]` generics (use `typing.Dict`/`typing.List`)
4. Hydra config invariants for all 70M models (`dim=512`, `num_layers=8`, `max_position_embeddings=1024`)
5. `pytest tests/ -m "not cuda and not slow"` (CPU, no SLURM required)
6. Dry-run training smoke (2 steps, CPU, ~2 min)

**Do not claim an edit is complete or commit it until this script exits 0.**

### If Python 3.9.23 is unavailable locally

Say so explicitly. Do not claim success without running the harness.

### Before pushing to `a100_70m_baseline`

- `bash scripts/validate_for_willi.sh` must be green.
- Last GitHub Actions run on `a100_70m_baseline` must be green (check Actions tab).
- Never push with `git push --no-verify` or skip the harness.

### When adding a new module or config key

Add a corresponding assertion to `tests/test_willi_parity.py`. The parity test file is the living spec of willi compatibility — keep it up to date.

### Common willi-incompatible patterns to avoid

| Wrong (Python ≥ 3.10) | Correct (Python 3.9) |
|---|---|
| `x: dict[str, int]` | `x: Dict[str, int]` (import from `typing`) |
| `def f() -> list[str]` | `def f() -> List[str]` |
| `Optional[X] \| None` or `X \| Y` | `Optional[X]` or `Union[X, Y]` |
| `from __future__ import annotations` + bare generics | Explicit `typing` imports |

### GitHub Actions CI

Every push to `a100_70m_baseline` triggers `.github/workflows/willi_parity.yml`:
- Python 3.9.23, CPU-only, Ubuntu runner
- Same gates as local harness (static checks + unit tests + parity tests + dry-run)
- PRs targeting `a100_70m_baseline` are also gated