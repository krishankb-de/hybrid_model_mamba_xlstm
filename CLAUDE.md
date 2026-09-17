# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Session Bootstrap (READ FIRST)

**Branch `h100_mamba3_v2` (ACTIVE): read `MAMBA3_PLAN_V2.md` + `mamba3_v2_state.json` at repo root FIRST.** They are the active resumable plan-of-record: the Mamba-3 work from branch `h100_scaling_mamba3` (phases M0–M7 **complete** — defect pinned and fixed behind flags, `Mamba3Block` shipped as a fifth layer type, O(1) decode cache, the 8-arm screen run) merged onto the Phase-14/15 baseline of `h100_scaling`. Resume at `mamba3_v2_state.json["current_phase"]`; the checkboxes in `MAMBA3_PLAN_V2.md` are ground truth. After **every** meaningful change (test written, phase gated, job submitted, job finished, eval scored) tick the checkbox AND update `mamba3_v2_state.json["last_updated"]` + append a one-line `notes` entry + record the evidence (job id, log path, metric) under `phases[<id>].evidence`. Use `python scripts/mamba3_state.py tick|phase|note|readme|show` (locally, in the venv — on the cluster anything scripted goes through `sbatch`/`srun`) rather than hand-editing both files, and run `readme` at the end of every phase. If `mamba3_v2_state.json` is lost, regenerate it with `scripts/mamba3_state.py sync`.

**What that plan is.** A verified correctness defect: the legacy selective scan computes `A_cum * cumsum(Bx / A_cum.clamp(1e-8))`, which annihilates a token's own contribution to the state when the clamp fires — rel-max-err **0.92** at the measured init Δ=0.705 (`scan_interface.py`, the same class in `tfla_interface.py` where **70.9%** of entries hit the clamp at the shipped forget-gate init; pinned independently by Phase 14C-1 in `tests/test_scan_correctness.py`). Both recurrences are fixed behind flags (`scan_impl`, `tfla_impl`; **defaults stay `legacy`** so every published checkpoint reproduces byte-for-byte — new yamls must pin the value explicitly). The exact log-space fix costs 19.3 GB in Mamba-1's `A=(d_inner,N)` form but **19 MB** in Mamba-2/3's scalar-`A`-per-head form, so `Mamba3Block` (SSD, `d_state=128`, parameter-matched to +0.24%) is what makes correctness affordable. **Measured at 12K steps (M7):** fixing the defect on the existing Mamba-1 (A1) gives **−16.0%** val PPL, migrating to SSD (A2) **−13.8%** at **1.94×** the speed — the two are not significantly apart, so *the headline is the defect, not the architecture*. Trapezoid and RoPE were nulls at this scale. Next: screen A2 + exact TFLA, then the full pipeline against the 3-seed bands below. **⚠ NO MERGE into `h100_scaling` without an explicit instruction from the user** — not on plan completion, not on a passing gate.

**`H100_SCALING_PLAN.md` + `h100_scaling_state.json` are the CLOSED BASELINE record** (Phases 1–15 complete; `current_phase: phase15_thesis_strength` is its final state). Do not resume against them; read them for the numbers every Mamba-3 arm is compared to. **OBJECTIVE (pivot 2026-08-16): MEDICAL REPORT GENERATION scored by ROUGE-L / CheXbert F1, not retrieval.** Retrieval (Phases 1–6G) is a **COMPLETE, CLOSED supporting chapter** — clean-protocol MIMIC i2t R@10 10.81% → 14.59%; protocol-matched 11.07% → 17.14%; Indiana flat within noise. **Do not re-open or re-run retrieval arms.**

**Where the baseline stands (Phase 15 closed 2026-09-16):**
- **Phases 1–13 are COMPLETE and CLOSED.** That includes **Phase 7 PhysioNet credentialing — DONE 2026-08-16** (credentialed account, DUAs signed on `mimic-cxr/2.1.0` and `mimic-cxr-jpg/2.1.0`, live download verified) and **Phase 8 — DONE 2026-08-27** (`fetch` 217,999/218,131 = 99.94%, `pack` produced the splits). **Nothing is gated on PhysioNet any more — do not treat it as blocking.** The data lives under `/sc/home/$USER/dataset/mimic_full/` (191,462 train pairs; official subject-disjoint test split n=2663); it is DUA-covered and must never be committed (see the `.gitignore` guards).
- **Image conditioning EXISTS** (Phase 10, shipped): `ImagePrefixMapper` prepends `k=32` prefix tokens to the decoder's input embeddings. The old note that "`HybridLanguageModel.forward()` takes only `input_ids` — no image conditioning exists yet" is **obsolete**.
- **Best checkpoint:** `outputs/h100_report_gen_full_ext_4gpu_tower13d/checkpoints/last.ckpt` (13D), decoded with **beam search, `beam_size=3`** (strictly beats greedy on every metric). Official test split n=2663: ROUGE-L **0.1899**, CheXbert-14-micro **0.4736** vs the retrieval-NN floor's 0.1636 / 0.4296. Writeup: `analysis/h100_scaling_results.md`. **⚠ 2026-09-16: 13D is one seed and the high CheXbert draw — report the 3-seed means instead (ROUGE-L 0.1949 ± 0.0047, CheXbert-14-micro 0.4480 ± 0.0223, 14-macro 0.2660 ± 0.0122, example-F1 0.3790 ± 0.0214). Vs the floor, only the text metrics and exact-match win at all 3 seeds; CheXbert micro ties at 2 of 3 and 14-macro loses at all 3.**
- **Phase 14 is COMPLETE and ACCEPTED** (supervisor review of 2026-09-07; all three items answered with measurements). Primary deliverable: **`analysis/PHASE14_SUPERVISOR_REVIEW.md`** — cite that document, not summary lines elsewhere; it is authoritative where they disagree. Settled result at matched params and matched `prefix_k=32`, official test split n=2663, 95% paired-bootstrap CIs: **hybrid significantly wins 3 CheXbert F1 metrics** (14-micro, 5-micro, 5-macro), **Transformer significantly wins ROUGE-L and exact-match-14**, and **three metrics tie** (BLEU-1, BLEU-4, CheXbert-14-macro). Do **not** write "hybrid wins all four CheXbert" or "Transformer wins all four surface metrics" — both were overturned by the CIs. **🔴 2026-09-16 (15B-4, COMPLETE): that split decision is ONE seed and does NOT replicate in either direction.** Across seeds 42/43/44 the hybrid's three CheXbert wins AND the Transformer's ROUGE-L win are all seed-42-only; 8 of 10 metrics are statistically indistinguishable. Only two differences hold up at all, both small: BLEU-1 (hybrid ahead 2/3 seeds) and exact-match-5/14 (Transformer ahead 3/3). Seed SD is 3–15× the bootstrap half-width. **Cite the 15B-4 seed table in `H100_SCALING_PLAN.md`, never the seed-42 CIs, for any architecture claim.**
- **Phase 15 is COMPLETE (2026-09-16) — all four supervisor items answered with measurements.** **15A** `analysis/PUBLISHED_BASELINES.md` (§4.2 verdict: the macro gap survives and widens to 0.2660 ± 0.0122, but on example-F1 the project is mid-field at 0.3790 ± 0.0214); **15B** 3 seeds × both arms with paired CIs — it overturned Phase 14's split decision in both directions; **15C** the aux CheXpert loss is a **negative result**, stopped by its pre-registered rule (`analysis/PHASE15C_AUX_LOSS.md`); **15D** retrieval framed once, explicitly, as a supporting chapter that trails the best specialised architectures. Remaining work is thesis writing plus archiving before the cluster account expires **2026-09-29**. Original scope below.
- **Phase 15 scope, for reference:** (`current_phase: phase15_thesis_strength`, reopened 2026-09-13 by a **second** supervisor review, scoped "required for thesis strength — needed before submission"). Four items: **15A** a table of published field baselines (R2Gen, CvT2DistilGPT2, RGRG, RA-RRG, Janus-CXR) — none exist in this repo today; **15B** statistical rigor for the generation table (**no seeds, no CIs** — note `configs/config.yaml:13` hardcodes `seed: 42` and no wrapper exposes it, so every report-gen run in project history is one seed); **15C** one designed attempt at the CheXbert-14 macro gap via an auxiliary multi-label CheXpert loss on the image prefix; **15D** retrieval framing. **Ordering is deliberately NOT the supervisor's: 15B's seed band is the measuring instrument for 15C** (13F failed uninterpretably for exactly this lack), and 15A calibrates 15C's target. **🔴 Already computed: 0.40 macro is unreachable by fixing rare findings** — the retrieval floor scores nonzero F1 on all 14 labels and still reaches only 0.3014, and lifting 13D's 3 zero-F1 labels to the floor's own values gives 0.3064. 15C is **one honest attempt, not iterate-to-target**.
- The selective-scan operator repair is owned by `MAMBA3_PLAN_V2.md` on this branch. `MAMBA3_INTEGRATION_PLAN.md` + `mamba3_integration_state.json` were retired at M0-E (their audit is folded into the plan's Context).

Approved plans: Mamba-3 v2 = `/Users/krish/.claude/plans/please-complete-code-base-goofy-mist.md`; the H100 campaign = `/Users/krish/.claude/plans/i-want-to-implement-twinkling-ullman.md`. Only append to `h100_scaling_state.json["notes"]` when a Mamba-3 result bears on a baseline number (plan V4-D). `HYBRID_ARCH_REFACTOR_PLAN.md` + `hybrid_arch_refactor_state.json` (**COMPLETE** — broke the MIMIC ceiling 8.23%→10.45% i2t R@10), plus `BIOMEDCLIP_KD_PLAN.md`, `JOINT_TRAINING_PLAN.md` and their state files, are kept as historical record only — do NOT resume against them.

**Corrections to older claims in this file:** the Mamba path does **not** use a Triton kernel — `scan_triton.py` is imported and never called; the live scan is the pure-PyTorch chunked one in `scan_interface.py`. `mamba_block_v2.py`, `mlstm_block_v2.py`, `hybrid_layer.py`, `scan_triton.py` and `tfla_triton.py` are dead code (deleted at plan phase V4-C). `HybridLanguageModel.forward()` **does** accept `inputs_embeds`; image conditioning exists and is prefix-based (`models/prefix_mapper.py`).

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