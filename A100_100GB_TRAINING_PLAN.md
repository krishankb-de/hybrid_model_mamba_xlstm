# A100 100GB 70M Hybrid Contrastive Training Plan

**Branch:** `a100_100gb_70m_baseline` · **Plan-of-record:** this file (committed). **State:** `a100_100gb_state.json` (gitignored). Markdown checkboxes are ground truth; JSON is regeneratable.

---

## Context

Old BIOMEDCLIP_KD pipeline (Willi A100-40GB) plateaued at MIMIC i2t R@10 ≈ 9.99% / Indiana ≈ 3.36% with three failure modes:
1. **Overfitting** on small Indiana (6.7K) / MIMIC (27.5K) corpora
2. **Modality gap** — paired cosine 0.21–0.29 never closed
3. **OOM** under joint loss + MoCo queue + frozen ViT teacher near 40GB

Migrating to new host: **A100-80GB VRAM, 100GB system RAM, no SLURM**. Warm-start from `output_willi_server/stage0_model_only.pt` (PPL 12.42 verified) — no LM re-pretrain.

Pipeline: Stage 0 LM → Stage 1 SimCSE (PubMed) → Stage 2 CLIP (Indiana) → Stage 3 Joint MIMIC (BiomedCLIP-text-KD + CLIP + SimCSE + MoCo + R-Drop newly implemented).

---

## Resumability protocol

After every meaningful state change:
1. Flip `[ ] → [x]` checkbox in this file.
2. Update `a100_100gb_state.json`: bump `last_updated` (ISO 8601 UTC), append one-line `notes` entry, update `current_phase` if phase boundary crossed.
3. Commit: `phase N: <one-line update>`.

If state JSON missing on fresh clone → regenerate from checkboxes (gitignored on purpose).

---

## Phase 1 — Plan bootstrap & cleanup (no training)

- [x] Create `A100_100GB_TRAINING_PLAN.md` at repo root (this file, committed)
- [x] Create `a100_100gb_state.json` at repo root (gitignored skeleton)
- [x] Update existing `CLAUDE.md` Session Bootstrap section — remove BiomedCLIP-KD paragraph; replace with A100_100GB plan rules
- [x] `.gitignore`: remove `!BIOMEDCLIP_KD_PLAN.md` allowlist; remove `biomedclip_kd_state.json` ignore line
- [x] `.gitignore`: add `!A100_100GB_TRAINING_PLAN.md` allowlist; add `a100_100gb_state.json` ignore line
- [x] `git rm BIOMEDCLIP_KD_PLAN.md`; `rm biomedclip_kd_state.json` (if present)
- [x] Keep `JOINT_TRAINING_PLAN.md` + `joint_training_state.json` untouched (historical)
- [x] Run `bash scripts/validate_for_willi.sh --ci-static-only` — green
- [x] `git commit -m "phase 1: bootstrap A100-80GB plan; remove BiomedCLIP-KD"`

**Critical files:** `.gitignore`, `CLAUDE.md`, `A100_100GB_TRAINING_PLAN.md` (new), `BIOMEDCLIP_KD_PLAN.md` (delete).

**Verify:** static gates green; `git ls-files | grep -E "A100|CLAUDE|BIOMEDCLIP"` shows expected state.

---

## Phase 2 — Machine + env adapter

- [ ] New `configs/trainer/a100_80gb_single_gpu.yaml` (bf16-mixed, batch=64, accum=1, compile=false, grad_clip=1.0, val_check=500)
- [ ] New per-stage overlays `configs/trainer/stages/{stage1_simcse,stage2_clip,stage3_joint}.yaml`
- [ ] Patch hardcoded `/scratch/...` cache_dir in `configs/dataset/{pubmed,mimic_cxr,indiana_cxr}.yaml` → `${oc.env:DATA_CACHE_DIR,./data/cache}/<dataset>`
- [ ] Patch `scripts/evaluate_cxr_retrieval.py:420`, `scripts/verify_mimic_cxr.py:154` cache-dir defaults
- [ ] New `scripts/validate.sh` (`CONDA_ENV=${CONDA_ENV:-hybrid_a100}`); keep `validate_for_willi.sh` as alias
- [ ] New `scripts/launch/train_stage{1,2,3}.sh` (direct python, no sbatch)
- [ ] Move existing SLURM `*.sh` to `scripts/slurm_legacy/`
- [ ] New `scripts/setup_data.sh` — idempotent, downloads PubMed/Indiana/MIMIC, 150GB budget
- [ ] Confirm logging defaults (TensorBoard + run_metadata.json) — no W&B by default
- [ ] Smoke: 50-step wikitext run < 5 min, peak < 25GB
- [ ] Run `bash scripts/validate.sh` with `CONDA_ENV=hybrid_a100` → green
- [ ] Commit

**Critical files:** `configs/trainer/a100_80gb_single_gpu.yaml`, `configs/trainer/stages/*.yaml`, `configs/dataset/{pubmed,mimic_cxr,indiana_cxr}.yaml`, `scripts/validate.sh`, `scripts/launch/`, `scripts/setup_data.sh`, `scripts/evaluate_cxr_retrieval.py`, `scripts/verify_mimic_cxr.py`.

---

## Phase 3 — Stage 0 ckpt verify + state-dict loader

- [ ] Transfer `output_willi_server/stage0_model_only.pt` to new machine; record SHA256 in state JSON
- [ ] New `hybrid_xmamba/utils/state_dict_loader.py` `load_lm_state_dict(path_or_state, target_module=None)`
- [ ] Handles prefixes in order: `_orig_mod.`, `model._orig_mod.`, `_orig_mod.model.`, `model.`, `lm.`, `module.`
- [ ] Logs first 10 keys before/after; returns `(missing, unexpected)` counts
- [ ] Replace inline logic at `scripts/train_contrastive.py:622-632`
- [ ] Replace inline logic at `scripts/evaluate_lm.py:241-246`
- [ ] Replace inline logic at `scripts/evaluate_sts.py:59`
- [ ] Replace inline logic at `scripts/evaluate_retrieval.py:54`
- [ ] Replace inline logic at `scripts/evaluate_cxr_retrieval.py:76-77`
- [ ] New `tests/test_state_dict_loader.py` covering 8 prefix combinations
- [ ] LM-quality reverify on 1K WikiText test samples → expect PPL within ±0.2 of 12.42
- [ ] Commit

**Critical files:** `hybrid_xmamba/utils/state_dict_loader.py` (new), 5 script callsites, `tests/test_state_dict_loader.py` (new).

---

## Phase 4 — OOM hardening

- [ ] Add per-block `use_gradient_checkpointing` flag in `hybrid_xmamba/layers/hybrid_block.py`
- [ ] New `hybrid_xmamba/training/memory_probe_callback.py` — logs `mem/peak_gb`; warn > 70 GB
- [ ] OOM safety net in `JointMultiTaskLightningModule.training_step` — try/except CUDA OOM, log state, raise (no auto-retry)
- [ ] Sweep batch ∈ {32,48,64} × grad_accum ∈ {1,2}; record peak mem in `docs/results/oom_sweep.md`
- [ ] Pick max-safe (peak ≤ 70 GB); wire into `configs/trainer/stages/stage3_joint.yaml`
- [ ] Drop existing inter-loss `empty_cache()` calls
- [ ] Commit

**Critical files:** `hybrid_xmamba/layers/hybrid_block.py`, `hybrid_xmamba/training/lightning_module.py`, `hybrid_xmamba/training/memory_probe_callback.py` (new), `docs/results/oom_sweep.md` (new), `configs/trainer/stages/stage3_joint.yaml`.

---

## Phase 5 — Overfitting regularization

- [ ] New `hybrid_xmamba/training/ema_callback.py` (decay 0.999, swap-in/out for val)
- [ ] Wire `EarlyStopping` into `scripts/train_contrastive.py` (monitor=val/i2t_r10_ema, patience=3, min_delta=0.005) — Stage 2/3 only
- [ ] Cosine + plateau-restart scheduler option in `configure_optimizers`
- [ ] New `configs/model/overlay_stage3.yaml` (proj_head_dropout=0.15, MLP dropout=0.1)
- [ ] Backbone weight_decay 0.15 via `optimizer.py` `backbone_weight_decay` field
- [ ] Diversity guard in `ContrastiveEvalCallback._log_cosine_stats` — warn if offdiag mean > 0.6
- [ ] Tests: `tests/test_ema_callback.py`
- [ ] Commit

**Critical files:** `hybrid_xmamba/training/ema_callback.py` (new), `scripts/train_contrastive.py`, `hybrid_xmamba/training/lightning_module.py`, `hybrid_xmamba/training/optimizer.py`, `configs/model/overlay_stage3.yaml` (new), `hybrid_xmamba/training/contrastive_eval_callback.py`, `tests/test_ema_callback.py` (new).

---

## Phase 6 — Modality-gap audit + diagnostics

- [ ] Joint-only audit of `JointMultiTaskLightningModule` — ensure no PubMedBERT path runs when `cfg.distill.teacher == "biomedclip_text"` (DistillContrastiveLightningModule's PubMedBERT path is legitimate Stage 1 KD; do not touch)
- [ ] Two-tokenizer assertion in `train_contrastive.py:113-148` and joint dataloader at line 342
- [ ] Three new metrics in `ContrastiveEvalCallback`: `gap/L2`, `paired_cos/text_teacher`, `paired_cos/image_teacher`
- [ ] Optional `--apply-gap-correction` flag in `evaluate_retrieval.py` and `evaluate_cxr_retrieval.py` (default off)
- [ ] Commit

**Critical files:** `hybrid_xmamba/training/lightning_module.py`, `hybrid_xmamba/training/contrastive_eval_callback.py`, `scripts/train_contrastive.py`, `scripts/evaluate_retrieval.py`, `scripts/evaluate_cxr_retrieval.py`.

---

## Phase 7 — R-Drop (new)

- [ ] New `hybrid_xmamba/training/losses.py` `r_drop_loss(emb1, emb2, keys, τ)` — symmetric KL via log_softmax + kl_div(batchmean)
- [ ] Integrate in `JointMultiTaskLightningModule.training_step` only when `r_drop_alpha > 0`; keys from first forward only
- [ ] Tests: non-negativity, zero-when-no-dropout, symmetry, gradient flow (`tests/test_r_drop.py`)
- [ ] Add `r_drop_alpha: 0.5` to `configs/distill/biomedclip_kd_joint.yaml`
- [ ] 100-step smoke on Stage 3 config with R-Drop enabled — `train/r_drop_loss` finite
- [ ] Commit

**Critical files:** `hybrid_xmamba/training/losses.py` (new), `hybrid_xmamba/training/lightning_module.py`, `tests/test_r_drop.py` (new), `configs/distill/biomedclip_kd_joint.yaml`.

---

## Phase 8 — Stage 1 SimCSE (PubMed, 5K steps)

- [ ] Warm-start: `lm_checkpoint=output_willi_server/stage0_model_only.pt`
- [ ] `bash scripts/launch/train_stage1.sh` → mode=simcse, dataset=pubmed, trainer=stages/stage1_simcse, max_steps=5000
- [ ] Logged: alignment/uniformity (Wang & Isola), cos-sim histogram, embedding-norm μ/σ
- [ ] Pass: val/loss decreasing, alignment > 0.6, uniformity > -3, no collapse warning
- [ ] Eval suite: STS-B Spearman ≥ 0.55, BIOSSES ≥ 0.5
- [ ] Save `outputs/stage1_simcse/checkpoints/last.ckpt`
- [ ] Commit results note

---

## Phase 9 — Stage 2 CLIP (Indiana, 3K steps)

- [ ] Warm-start: Stage 1 ckpt
- [ ] Freeze backbone first 1000 steps (reuse `freeze_text_encoder_steps=1000`)
- [ ] `bash scripts/launch/train_stage2.sh` → mode=clip, dataset=indiana_cxr, trainer=stages/stage2_clip, max_steps=3000, batch=16, max_length=256
- [ ] Apply Phase 5 EMA + EarlyStop; Phase 6 gap diagnostics; no R-Drop
- [ ] Pass: val/i2t_r10 ≥ 25%, paired_cos/image_teacher > 0.4, gap/L2 < 0.5
- [ ] Save `outputs/stage2_indiana_clip/checkpoints/last.ckpt`
- [ ] Commit results note

---

## Phase 10 — Stage 3 Joint MIMIC (10K steps)

- [ ] Warm-start: Stage 2 ckpt
- [ ] Loss weights: alpha_kd_warmup=1.0 → alpha_kd_post=0.3; beta_clip=1.0; gamma_simcse=0.1; r_drop_alpha=0.5
- [ ] freeze_text_encoder_steps=1000; MoCo queue cold-start at unfreeze
- [ ] MoCo: K=16384, momentum=0.999, dim=512, text-only queue
- [ ] Trainer: batch from Phase 4 sweep, grad_accum from sweep, max_steps=10000, val_check=500
- [ ] Full Phase 5 regularization + Phase 6 diagnostics + Phase 7 R-Drop enabled
- [ ] **Kill rules** (manual after each val): see state JSON `kill_job_rules`
- [ ] **Pass gates**: MARGINAL = MIMIC R@10 ≥ 15% AND Indiana ≥ 8%; PARTIAL = ≥ 18% / ≥ 12%; SUCCESS = ≥ 25% / ≥ 20%
- [ ] Save `outputs/stage3_joint_mimic/checkpoints/{last,best_i2t_r10}.ckpt`
- [ ] Record tier in state JSON; commit

---

## Phase 11 — Cross-eval + 5 ablations (sequential)

- [ ] Eval matrix {Stage 0, 1, 2, 3} × {STS-B, BIOSSES, NFCorpus, Indiana retrieval, MIMIC retrieval} (skip image-retrieval for Stage 0/1)
- [ ] Ablation 1: −r_drop (`r_drop_alpha=0`)
- [ ] Ablation 2: −moco (`moco_queue_size=0`)
- [ ] Ablation 3: −kd (`alpha_kd_*=0`)
- [ ] Ablation 4: −gap_correction (default eval)
- [ ] Ablation 5: +gap_correction (eval-time `--apply-gap-correction`)
- [ ] New `scripts/run_ablations.py` — orchestrates 4 retraining + 1 eval-only ablation
- [ ] Build `docs/results/a100_100gb_results.md` with comparison vs baseline (MIMIC 9.99%, Indiana 3.36%, paired_cos 0.226)
- [ ] Commit

**Critical files:** `scripts/run_ablations.py` (new), `docs/results/a100_100gb_results.md` (new).

---

## Phase 12 — Writeup, tag, optional PR

- [ ] Finalize `docs/results/a100_100gb_results.md` (pass/fail vs gates, lessons, best config)
- [ ] `git tag v0.1-a100-100gb -m "70M hybrid + BiomedCLIP-text-KD + MoCo + R-Drop, A100-80GB"`
- [ ] (Optional, on user request) open PR `a100_100gb_70m_baseline → main`
- [ ] Update `a100_100gb_state.json["current_phase"] = "12_done"`

---

## Verification gates summary

| Phase | Gate |
|---|---|
| 1 | static gates green; new branch + plan files committed; BIOMEDCLIP_KD_PLAN.md removed |
| 2 | validate.sh green with CONDA_ENV=hybrid_a100; smoke < 5 min, peak < 25 GB |
| 3 | `pytest tests/test_state_dict_loader.py` green; PPL ≈ 12.4 (±0.2) |
| 4 | OOM sweep table committed; chosen Stage-3 config peak ≤ 70 GB |
| 5 | EMA test green; collapse warning fires on synthetic input |
| 6 | Tokenizer assertion fires on broken config; new `gap/*` and `paired_cos/*` metrics in TB/CSV |
| 7 | `pytest tests/test_r_drop.py` green; r_drop_loss finite on 100-step smoke |
| 8 | STS-B ≥ 0.55 |
| 9 | Indiana R@10 ≥ 25% |
| 10 | Tier (MARGINAL/PARTIAL/SUCCESS) recorded in state JSON |
| 11 | Ablation table in `docs/results/a100_100gb_results.md` |
| 12 | `v0.1-a100-100gb` tag pushed; final notes in state JSON |

---

## Resolved decisions

- DATA_CACHE_DIR: fresh on new machine; `scripts/setup_data.sh` downloads PubMed (~80GB) / Indiana (~3GB) / MIMIC (~50GB). 150GB budget.
- Logging: local TensorBoard + `run_metadata.json` (already wired). W&B opt-in via `cfg.wandb.enabled=true`.
- Ablations: sequential, no parallel sweep.
- Gap-correction default: decided post-Phase-11 based on ablation delta.
- Stage 0 ckpt: warm-start from `output_willi_server/stage0_model_only.pt` (no LM re-pretrain).
