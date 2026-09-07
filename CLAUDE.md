# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Session Bootstrap (READ FIRST)

Before any work, read `H100_SCALING_PLAN.md` (committed) and `h100_scaling_state.json` (committed; allowlisted in `.gitignore`) at repo root. They are the ACTIVE resumable plan-of-record. **OBJECTIVE PIVOT 2026-08-16: the target is now MEDICAL REPORT GENERATION scored by ROUGE-L / CheXbert F1, not retrieval.** Retrieval (Phases 1–6G) is a **COMPLETE, CLOSED supporting chapter** — final numbers: clean-protocol MIMIC i2t R@10 10.81% → 14.59%; protocol-matched 11.07% → 17.14%; Indiana flat within noise. **Do not re-open or re-run retrieval arms to chase a higher R@10.**

**Where the plan actually stands (updated 2026-09-07):**
- **Phases 1–13 are COMPLETE and CLOSED.** That includes **Phase 7 PhysioNet credentialing — DONE 2026-08-16** (credentialed account, DUAs signed on `mimic-cxr/2.1.0` and `mimic-cxr-jpg/2.1.0`, live download verified) and **Phase 8 — DONE 2026-08-27** (`fetch` 217,999/218,131 = 99.94%, `pack` produced the splits). **Nothing is gated on PhysioNet any more — do not treat it as blocking.** The data lives under `/sc/home/$USER/dataset/mimic_full/` (191,462 train pairs; official subject-disjoint test split n=2663); it is DUA-covered and must never be committed (see the `.gitignore` guards).
- **Image conditioning EXISTS** (Phase 10, shipped): `ImagePrefixMapper` prepends `k=32` prefix tokens to the decoder's input embeddings. The old note that "`HybridLanguageModel.forward()` takes only `input_ids` — no image conditioning exists yet" is **obsolete**.
- **Best checkpoint:** `outputs/h100_report_gen_full_ext_4gpu_tower13d/checkpoints/last.ckpt` (13D), decoded with **beam search, `beam_size=3`** (strictly beats greedy on every metric). Official test split n=2663: ROUGE-L **0.1899**, CheXbert-14-micro **0.4736** vs the retrieval-NN floor's 0.1636 / 0.4296. Writeup: `analysis/h100_scaling_results.md`.
- **ACTIVE WORK: Phase 14** (`current_phase: phase14_supervisor_review`, reopened 2026-09-07 by supervisor review) — **14A** train a parameter-matched Transformer baseline (highest priority; none exists, and the central "attention-free matches attention" claim has never been tested against the architecture it names), **14B** re-measure the boilerplate/template rate on the final 13D checkpoint, **14C** bound the selective-scan `clamp(min=1e-8)` defect with a committed regression test. **⚠ OPERATOR FREEZE: do not change the selective scan while 14A is running** — see Phase 14C for why.
- `MAMBA3_INTEGRATION_PLAN.md` + `mamba3_integration_state.json` own the full selective-scan operator repair (`current_phase: M0_pin_the_defect`, **no code written**). Phase 14 deliberately does **not** activate it.

Full approved plan: `/Users/krish/.claude/plans/i-want-to-implement-twinkling-ullman.md`. Resume at `h100_scaling_state.json["current_phase"]`; checkboxes in `H100_SCALING_PLAN.md` are ground truth for completed work. After every meaningful state change (run submitted, run finished, eval completed), update both the relevant checkbox AND `h100_scaling_state.json["last_updated"]` + append a one-line note to `h100_scaling_state.json["notes"]`. If `h100_scaling_state.json` is missing, regenerate from `H100_SCALING_PLAN.md` checkbox state. `HYBRID_ARCH_REFACTOR_PLAN.md` + `hybrid_arch_refactor_state.json` (**COMPLETE** — broke the MIMIC ceiling 8.23%→10.45% i2t R@10), plus `BIOMEDCLIP_KD_PLAN.md`, `JOINT_TRAINING_PLAN.md` and their state files, are kept as historical record only — do NOT resume against them.

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

- **Mamba block** (`layers/mamba_block.py`): Selective SSM with input-dependent gating, 1D causal convolution, SiLU activation. Uses the **chunk-parallel PyTorch** selective scan (`kernels/selective_scan/scan_interface.py::selective_scan_parallel`), run in fp32 for numerical stability. ⚠️ **Correction (verified 2026-09-07, Phase 14C-5):** `scan_interface.selective_scan()` calls the PyTorch path *unconditionally*; `scan_triton.selective_scan_triton` is imported and **never invoked**. Earlier revisions of this file claimed a Triton kernel is used for Mamba — that is false for the live path, and it also affects how efficiency curves must be described. ⚠️ **Known defect:** the chunk-parallel form divides by the cumulative decay with `A_cum.clamp(min=1e-8)`, which annihilates a token's own state contribution where `A_cum` underflows (rel-max-err ≈0.358 at the model's actual Δ). Being bounded and tested in Phase 14C; the repair is owned by `MAMBA3_INTEGRATION_PLAN.md`.
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