# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Session Bootstrap (READ FIRST)

Before any work, read `A100_100GB_TRAINING_PLAN.md` (committed) and `a100_100gb_state.json` (gitignored, local-only) at repo root. They are the resumable plan-of-record for the contrastive training pipeline on the new A100-80GB host. Resume at `a100_100gb_state.json["current_phase"]`; checkboxes in `A100_100GB_TRAINING_PLAN.md` are ground truth for completed work. After every meaningful state change (run submitted, run finished, eval completed), update both the relevant checkbox AND `a100_100gb_state.json["last_updated"]` + append a one-line note to `a100_100gb_state.json["notes"]`. If `a100_100gb_state.json` is missing on a fresh clone, regenerate from `A100_100GB_TRAINING_PLAN.md` checkbox state — it is gitignored on purpose.

`JOINT_TRAINING_PLAN.md` and `joint_training_state.json` are kept on disk as historical record only — do NOT resume against them. The earlier `BIOMEDCLIP_KD_PLAN.md` and `biomedclip_kd_state.json` have been deleted; their pipeline (BiomedCLIP-text-KD + MoCo on Willi A100-40GB) is superseded by the A100-100GB plan.

## Project Overview

Research implementation of a **Hybrid Mamba-xLSTM Language Model** combining Mamba (Selective SSM) and xLSTM (mLSTM with matrix memory) layers. The active model target is **70M parameters** (dim=512, 8 layers). Supports 150M and 350M variants as well.

## Installation

Use a python virtual 'venv' environement to run or do any testing or any form of installations.

```bash
pip install -e .
pip install -r requirements.txt
```

Requires Python ≥ 3.9, PyTorch ≥ 2.1 with CUDA. The 70M model fits on a T4 (15GB VRAM) for validation; A100 40/80GB recommended for full training runs. Production target: **A100-80GB VRAM, 100GB system RAM, no SLURM** (see `A100_100GB_TRAINING_PLAN.md`).

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

## Environment

- **Production host:** A100-80GB VRAM, 100GB system RAM, no SLURM (direct `python` launches via `scripts/launch/`)
- **DATA_CACHE_DIR:** required env var on new machine. Default `./data/cache`. `scripts/setup_data.sh` (Phase 2) creates and populates it (PubMed ~80GB, Indiana ~3GB, MIMIC ~50GB; 150GB budget).
- **Logging:** local-only by default. `TensorBoardLogger` + `run_metadata.json` are always-on. W&B is opt-in via `cfg.wandb.enabled=true` (set `WANDB_API_KEY` first).
- **Conda env:** `hybrid_a100` (Python 3.9.23 to preserve parity guards). The legacy `willi_parity` env is still supported via the `validate_for_willi.sh` alias.

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
- `trainer/` — `a100_single_gpu.yaml` (40GB-tuned), `a100_80gb_single_gpu.yaml` (80GB-tuned, Phase 2), `colab_single_gpu.yaml` (T4-compatible), `single_gpu.yaml`, `gpu_ddp.yaml`, `gpu_fsdp.yaml`. Stage overlays under `configs/trainer/stages/{stage1_simcse,stage2_clip,stage3_joint}.yaml` (Phase 2).
- `callbacks/` — logging callbacks

For fair comparison, all 70M models share identical hyperparameters (dim=512, layers=8, vocab=50257, lr=6e-4, weight_decay=0.1, warmup=1000 steps, max_position_embeddings=1024) — only `layer_pattern` differs.

The A100 trainer uses bf16-mixed precision, effective batch size 64 (32 × grad_accum=2 on 40GB; 64 × grad_accum=1 on 80GB), `torch.compile`, and fused AdamW.

### Training Entry Points

- `scripts/train.py` — main Hydra training script (calls `pl.Trainer.fit()`)
- `scripts/run_70m_experiments.py` — orchestrator: sanity check → train hybrid/mamba/xlstm 70M → evaluate → print comparison
- `scripts/run_a100_experiments.py` — same for 150M variant
- `scripts/train_contrastive.py` — Stage 1 SimCSE + Stage 2 CLIP + Stage 3 Joint contrastive training on top of a pretrained checkpoint
- `scripts/launch/train_stage{1,2,3}.sh` (Phase 2) — non-SLURM wrappers for the three contrastive stages

### Checkpoint Compatibility

State dict keys can have prefixes from `torch.compile` (`_orig_mod.`) or PyTorch Lightning wrapping (`lm.`, `model.`) or DDP (`module.`). Phase 3 consolidates prefix stripping into `hybrid_xmamba/utils/state_dict_loader.py`; until then, evaluation scripts have inline strip logic. When debugging checkpoint loading issues, use `debug_checkpoint_keys.py` and `check_checkpoint_compatibility.py` in the root directory.

### Evaluation Metrics

`scripts/evaluate_lm.py` computes: test perplexity, bits-per-byte, inference throughput (tokens/sec), peak GPU memory, and optional text generation samples.

### Contrastive / Retrieval Pipeline

`scripts/train_contrastive.py` wraps `HybridLanguageModel` in `HybridTextEncoder` with a projection head. Stage 1 uses SimCSE in-batch negatives; Stage 2 uses CLIP-style image-text pairing via BiomedCLIP. Stage 3 (Joint, Phase 10) combines BiomedCLIP-text KD + CLIP + SimCSE + R-Drop with MoCo queue. Evaluation is via `evaluate_retrieval.py`, `evaluate_cxr_retrieval.py`, and `evaluate_sts.py`.

## Production System

The system targets **A100-80GB GPU + 100GB system RAM, no SLURM** as the production environment for the A100_100GB plan. The earlier Willi A100-40GB SLURM environment is preserved as legacy (`scripts/slurm_legacy/` after Phase 2 migration). Training is invoked via `scripts/launch/` direct-python wrappers.

---

## Pre-Push Validation Protocol (MANDATORY)

The validation harness preserves Python 3.9.23 parity (the original Willi env). Bugs that slip through locally (PEP 604 syntax, wrong type hints, config drift) cost hours of debugging. Follow this protocol after every edit.

### After ANY edit to `hybrid_xmamba/`, `scripts/`, or `configs/`

```bash
# Use the new portable harness (default, after Phase 2)
bash scripts/validate.sh

# Or the legacy Willi-named alias (still works)
bash scripts/validate_for_willi.sh
```

By default this runs inside the `hybrid_a100` conda env (Python 3.9.23). Override with `CONDA_ENV=<other> bash scripts/validate.sh`. The harness gates:
1. AST parse of all source files under Python 3.9
2. PEP 604 guard — no `X | Y` union syntax (use `Optional[X]`)
3. PEP 585 guard — no bare `dict[...]`/`list[...]` generics (use `typing.Dict`/`typing.List`)
4. Hydra config invariants for all 70M models (`dim=512`, `num_layers=8`, `max_position_embeddings=1024`)
5. `pytest tests/ -m "not cuda and not slow"` (CPU)
6. Dry-run training smoke (2 steps, CPU, ~2 min)

For plan-only or doc-only changes, add `--ci-static-only` to skip the dry-run smoke.

**Do not claim an edit is complete or commit it until this script exits 0.**

### If Python 3.9.23 is unavailable locally

Say so explicitly. Do not claim success without running the harness.

### Before pushing to `a100_100gb_70m_baseline`

- `bash scripts/validate.sh` must be green.
- Last GitHub Actions run on `a100_100gb_70m_baseline` (or its predecessor `a100_70m_baseline`) must be green (check Actions tab).
- Never push with `git push --no-verify` or skip the harness.

### When adding a new module or config key

Add a corresponding assertion to `tests/test_willi_parity.py`. The parity test file is the living spec of Python 3.9.23 compatibility — keep it up to date.

### Common Python 3.9-incompatible patterns to avoid

| Wrong (Python ≥ 3.10) | Correct (Python 3.9) |
|---|---|
| `x: dict[str, int]` | `x: Dict[str, int]` (import from `typing`) |
| `def f() -> list[str]` | `def f() -> List[str]` |
| `Optional[X] \| None` or `X \| Y` | `Optional[X]` or `Union[X, Y]` |
| `from __future__ import annotations` + bare generics | Explicit `typing` imports |

### GitHub Actions CI

Every push to `a100_100gb_70m_baseline` triggers `.github/workflows/willi_parity.yml`:
- Python 3.9.23, CPU-only, Ubuntu runner
- Same gates as local harness (static checks + unit tests + parity tests + dry-run)
- PRs targeting `a100_100gb_70m_baseline` are also gated
