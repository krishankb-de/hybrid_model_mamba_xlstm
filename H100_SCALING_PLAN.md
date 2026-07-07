# H100 Scale-Up Plan — Hybrid Mamba-xLSTM CXR Retrieval (Plan-of-Record)

> Resumable plan-of-record. Read this + `h100_scaling_state.json` (gitignored, allowlisted) at session start.
> **Builds on the COMPLETED `HYBRID_ARCH_REFACTOR_PLAN.md`** (broke the MIMIC ceiling 8.23%→10.45% i2t R@10). That plan is finished — historical reference only.
> Full approved plan: `/Users/krish/.claude/plans/i-want-to-implement-twinkling-ullman.md`.
>
> **Current phase: Phase 5 — Stage-0 150M pretrain** (H100 box; local scaffolding through Phase 4 done; Phase 3 deferred post-Phase-6).

## Context

Prior campaign is done on **A100 40GB** (willi/`mitarb`). Canonical model = `hybrid_70m_v2` + `freq_kd=false` + `vit_unfreeze=2` + `moco=0`. Final: **MIMIC i2t R@10 10.45%** (Target tier), **Indiana 3.90%** (intrinsic/data-bound), **Stage-0 PPL 15.62** (undertrained vs baseline 13.10).

User now has **H100 (94/141GB)** + optional 2-4 H100 node. Three A100-era ceilings are now liftable:
1. **Contrastive negatives capped at ~31** — CLIP loss is in-batch only (no `all_gather`, `moco=0`); H100 VRAM fits 128-256 true negatives (`lightning_module.py:512-543,1127-1158`). Biggest MIMIC lever; also cuts epochs on the 27.5k-pair set → less overfitting.
2. **Stage-0 undertrained** — 2.7B frozen teacher forced bs=8/40GB; curve still descending at 40K (needed ~117K). H100 fits bs=32-64 + teacher → finish it.
3. **70M cap** — 150M/350M configs exist but use the OLD `[m,m,mlstm]`+`pre_rms` (no v2 wins). H100 fits 150M v2 training.

Indiana gap is ablation-proven data-bound → only lever is diverse CXR data (user has access).

**Goal:** H100-native infra + 150M-v2 backbone + scaled contrastive negatives + multi-source CXR data → push MIMIC to stretch (≥12%) and recover Indiana (≥floor), with clean per-lever attribution.

## Success bar (tiered)
- **Floor** (no regression): MIMIC i2t R@10 ≥ 10.45%; Indiana i2t ≥ 4.04% (recover); Stage-0 PPL ≤ 15.62.
- **Target**: MIMIC ≥ 12% (old stretch); Indiana ≥ 5.5%; PPL ≤ 13.76.
- **Stretch**: MIMIC ≥ 14%; Indiana ≥ 7%; PPL ≤ 13.10.

## Resolved decisions (from user, 2026-07-07)
- **SLURM**: long training → `--partition=aisc-batch` (7-day cap); eval/smoke → `--partition=aisc-shortrun` (1-day); `--gres=gpu:h100:X` (X=1..8). 7-day cap ⇒ full Stage-0 in ONE block (no requeue juggling).
- **Model**: scale to **150M v2** (port v2 arch; fresh Stage-0).
- **Priority**: backbone + MIMIC (finish Stage-0 → push MIMIC to stretch).
- **Indiana data**: add diverse CXR; **IU-Xray EXCLUDED from training** (it IS the Indiana eval set — zero-leakage).
- **Per-source text**: free-text passthrough (MIMIC) + label-templated pseudo-reports (CheXpert/VinDr) + translated reports (PadChest).
- **150M `max_position_embeddings`**: 1024 (v2 parity; corpora are ≤512).
- **KD teacher**: BioMedLM 2.7B primary; larger-teacher fallback if PPL gate missed.
- **Python**: target py≥3.10 on H100 (verify stack; fall back to 3.9 if a dep breaks); keep 3.9-syntax hygiene (forward-compatible).
- **Multi-GPU (Phase 3)**: secondary — after single-H100 MIMIC win.

## Critical files

| File | Phase | Action |
|---|---|---|
| `H100_SCALING_PLAN.md`, `h100_scaling_state.json`, `.gitignore`, `CLAUDE.md` | 1 | plan/state/bootstrap |
| `configs/trainer/h100_single_gpu.yaml`, `h100_multi_ddp.yaml` (NEW) | 2 | H100 trainers |
| `scripts/train_stage0_h100.sh`, `train_biomedclip_kd_h100.sh`, `eval_h100.sh` (NEW) | 2 | SLURM templates (aisc-batch/shortrun) |
| `hybrid_xmamba/training/lightning_module.py:512-543,1041,1127-1158` | 3 | `all_gather` CLIP negatives |
| `configs/model/hybrid_150m_v2.yaml` (NEW) | 4 | 150M v2 arch |
| `scripts/train_stage0_150m_h100.sh` (NEW) | 5 | 150M Stage-0 |
| `scripts/train_biomedclip_kd_150m_h100.sh` (NEW) | 6 | batch-scaled contrastive |
| `configs/dataset/cxr_multi.yaml` (NEW), `scripts/train_contrastive.py:339-451` | 7 | multi-source CXR + text adapter |
| `analysis/h100_scaling_results.md` (NEW) | 8 | results |
| `tests/test_willi_parity.py`, `test_layers.py` | 2-4 | per-phase asserts |

---

## Phases

### Phase 1 — Plan-of-record + state (NO CODE) ✅ COMPLETE
- [x] **1A** — Write `H100_SCALING_PLAN.md` at repo root (this file).
- [x] **1B** — Write `h100_scaling_state.json` at repo root.
- [x] **1C** — `.gitignore`: allowlist `H100_SCALING_PLAN.md` + `!h100_scaling_state.json`.
- [x] **1D** — `CLAUDE.md` Session Bootstrap: repoint to this plan; `HYBRID_ARCH_REFACTOR_PLAN.md` → completed-historical.
- [x] **1E** — `bash scripts/validate_for_willi.sh` green: 69 passed, 5 skipped, 9/9 gates (doc-only; no regression).
- [x] **1F** — Commit on branch `h100_scaling`.

### Phase 2 — H100 infra enablement (single-GPU primary + 2-4 node file) ✅ LOCAL COMPLETE (2D/2G need H100 box)
Reuse: `torch.set_float32_matmul_precision('high')` (already in train scripts); auto-compile on sm_90 (`train.py:320-334` covers H100).
- [x] **2A** — `configs/trainer/h100_single_gpu.yaml`: `bf16-mixed`, `devices=1`, `compile_model=true`, `accumulate_grad_batches=1`.
- [x] **2B** — `configs/trainer/h100_multi_ddp.yaml` (2-4 node): `strategy=ddp`, `devices=-1`, `find_unused_parameters=true`, `accum=1`. Inert until Phase 3.
- [x] **2C** — SLURM templates `train_stage0_h100.sh` (aisc-batch 7-day, bs=64/accum=1, gc off, compile off for segmented path), `train_biomedclip_kd_h100.sh` (aisc-shortrun, **bs=128 = the in-batch-negative lever**, LR √-scaled), `eval_h100.sh` (ppl/retrieval modes). ENV-parametrized (`SCRATCH_ROOT`/`VENV_ACTIVATE`) — aisc scratch/env paths TBD.
- [ ] **2D** — (H100 box) Python env: full-stack import + 2-step train smoke on py≥3.10; fall back to 3.9 if a dep breaks.
- [x] **2E** — `requirements.txt`: flash-attn documented **opt-in** (NOT forced into base reqs — needs nvcc, breaks CPU/parity harness; ViT-only, marginal).
- [x] **2F** — `tests/test_willi_parity.py::test_h100_trainer_configs_resolve` (2 cases green).
- [ ] **2G** — (H100 box) Smoke `smoke_arch_refactor.py` on H100 (finite fwd/bwd, i_gate<cap, no NaN).
- [x] **2H** — `validate_for_willi.sh` green (9/9 gates; +2 h100 config tests); committed.

### Phase 3 — Distributed CLIP negatives (`all_gather`) — SECONDARY, deferred post-Phase-6
Only makes multi-GPU help retrieval. Single 141GB already gives 128-256 negatives → not on the critical path.
- [ ] **3A** — `lightning_module.py`: grad-preserving `_gather_across_gpus()` (autograd-aware `torch.distributed.nn.functional.all_gather`, OpenCLIP-style). Wire into `_nt_xent_loss` (`:512-543`) + `_joint_step` CLIP (`:1041`). Gate on `dist.is_initialized() and world_size>1` → identity single-GPU.
- [ ] **3B** — Test: world_size=1 numerical invariance; gather shape == world_size×B.
- [ ] **3C** — `validate_for_willi.sh` green; commit.

### Phase 4 — 150M v2 architecture config ✅ COMPLETE
`HybridConfig` supports all knobs; `create_hybrid_blocks` handles any pattern — config-only + tests.
- [x] **4A** — `configs/model/hybrid_150m_v2.yaml`: `dim=768`, `num_layers=12`, `head_dim=64`, `num_heads=12`, `norm_topology=hybrid`, `pooling_strategy=attention`, `max_position_embeddings=1024`, `learning_rate=4.0e-4` (√-width scaled; tunable→5e-4), `warmup_steps=2000`. mLSTM stabilization knobs via HybridConfig defaults (as 70m_v2 does).
- [x] **4B** — `layer_pattern`: `[m,m,m,m,L,L,L,m,m,m,m,m]` (3 mLSTM centered = 25% = v2 parity).
- [x] **4C** — `test_hybrid_150m_v2_config_and_param_count`: **183.72M actual** (nominal "150M"; untied 50k-vocab embeddings dominate — consistent with the 70M config → 83M convention). Tight band [181,186]M guards arch drift.
- [x] **4D** — `validate_for_willi.sh` green (72 passed, 9/9 gates); committed.

### Phase 5 — Stage-0 pretrain 150M v2 on H100 (backbone quality) ⏳ SCRIPT READY — run pending H100 box
- [x] **5A** — `scripts/train_stage0_150m_h100.sh` (wrapper over `train_stage0_h100.sh`; `model=hybrid_150m_v2`, bs=48/accum=1, `max_steps=120000` ~3B tokens, `warmup=2000`, WSD, aisc-batch 4-day).
- [ ] **5B** — (H100) Verify `train_stage0_distill.py` threads `norm_topology` (fixed 9F) + no hardcoded dim=512.
- [ ] **5C** — (H100) Submit. KD teacher BioMedLM 2.7B (primary); fallback larger teacher (BioGPT-Large/Meditron-7B/OpenBioLLM-8B) if PPL misses. Optional KD `alpha` {0.3,0.5} sweep.
- [ ] **5D** — (H100) Gate: `eval_h100.sh MODE=ppl` / `eval_stage0_lm.sh` (auto-detect; locked protocol). Target PPL ≤ 13.76.

### Phase 6 — Contrastive batch scaling (MIMIC-only, isolate lever) → stretch MIMIC ⏳ SCRIPT READY — run pending H100 box
Hold canonical recipe (`biomedclip_kd_joint_v2`: freq_kd=false, vit_unfreeze=2, moco=0). Change ONLY batch + backbone.
- [x] **6A** — `scripts/train_biomedclip_kd_150m_h100.sh` (wrapper; `model=hybrid_150m_v2`, `lm_checkpoint`=Phase-5, `batch_size` sweep {64,128,256} via env, `accum=1`, LR √-scaled for bs=128).
- [ ] **6B** — (H100) LR √-scale per batch: `backbone_lr→~2e-5`, `head_lr→~6e-4` at bs=128 (re-scale for other bs). Log embedding mean/std + cosine histogram (collapse watch).
- [ ] **6C** — (H100) Submit. Kill gates: `cos_text_teacher≥0.85` by 1k; `val/clip_loss<3.0`; MIMIC R@10 ≥ 10.45% by 3k.
- [ ] **6D** — (H100) Gate: MIMIC i2t R@10 (`eval_h100.sh MODE=retrieval` / `evaluate_cxr_retrieval.py`). Target ≥12%.

### Phase 7 — Multi-source CXR data diversification → Indiana lever
Only proven Indiana lever. **IU-Xray EXCLUDED from training (= Indiana eval; zero-leakage).**
- [ ] **7A** — `configs/dataset/cxr_multi.yaml` (NEW): MIMIC + CheXpert + PadChest + VinDr-CXR + per-source weights.
- [ ] **7B** — `scripts/train_contrastive.py`: generalize `MIMICJointDataset`/`load_mimic_cxr` (`:339-451`) → `CXRJointDataset` with per-source **text adapter** (free-text passthrough / label→prompt template / translated-EN). Mixed-source collate; dual-tokenize unchanged.
- [ ] **7C** — Re-run Phase-6 best recipe on `cxr_multi`.
- [ ] **7D** — Gate: Indiana i2t R@10 ≥ 4.04% floor (target 5.5%) while MIMIC holds ≥ Phase-6.

### Phase 8 — Full eval + comparison + writeup
- [ ] **8A** — Authoritative MIMIC + Indiana + STS (BIOSSES/STS-B) + PubMed PPL on best 150M ckpt.
- [ ] **8B** — `analysis/h100_scaling_results.md` (NEW): 70M-A100 vs 150M-H100 table; per-lever ablation; honest limitations.
- [ ] **8C** — Update `h100_scaling_state.json` final verdict + best ckpt path.

---

## Verification (each phase gates on)
1. `bash scripts/validate_for_willi.sh` exits 0 (3.9-syntax hygiene kept, forward-compatible) + H100-env stack smoke on py≥3.10 (Phase 2).
2. New phase test passes.
3. Numerical: forward/backward finite on CPU + H100 smoke; grad-norm bounded; no NaN 50+ steps; i_gate<cap.
4. SLURM kill-gates pre-declared (Phase 5 PPL; Phase 6 cos_text_teacher/clip_loss/R@10).
5. Reconcile in-training vs authoritative eval numbers (refactor caught 4 latent bugs this way).

## Resumability contract
1. Read `H100_SCALING_PLAN.md` + `h100_scaling_state.json` at session start.
2. Resume at `h100_scaling_state.json["current_phase"]`. Checkboxes here = ground truth.
3. After every state change: tick checkbox + update `last_updated` (ISO 8601) + append 1-line `notes` entry.
4. If state JSON missing locally, regenerate from this file's checkbox state (gitignored on purpose).
5. Never re-run a checkpoint-producing phase (5,6,7) without first reading its log + logging a verdict.

## Lessons carried from the completed refactor (do not repeat)
- CLIP negatives are in-batch only — grad-accum does NOT add negatives; per-step batch is the real lever.
- `norm_topology` must be threaded into every `HybridConfig` builder (train/distill/contrastive) — silently dropped = wrong FFN forward.
- Eval must auto-detect `layer_pattern` + `norm_topology` from checkpoint; must load fine-tuned `image_encoder.*` (fresh-ViT load read 1.89% vs true 10.94%).
- MoCo queue post-KD-warmup is harmful → keep `moco_queue_size=0`.
- freq-decoupled KD hurt Indiana → stays off (canonical).
- Always reconcile in-training vs authoritative eval numbers before citing.

## Unresolved questions
- CheXpert/VinDr label→prompt template wording — finalize in Phase 7 (start CheXzero-style; iterate if MIMIC regresses).
- 150M `learning_rate`: **4.0e-4 chosen** (conservative √-width scale; bump to 5e-4 if Phase-5 loss descends slowly).
- Whether willi/A100 remains a target after H100 migration (if retired, drop py3.9 guards + `validate_for_willi.sh`).
