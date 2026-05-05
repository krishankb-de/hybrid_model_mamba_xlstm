# BiomedCLIP Text-KD Architectural Pivot — Plan-of-Record

> Supersedes `JOINT_TRAINING_PLAN.md` + `joint_training_state.json`. Resumable. Read this file + `biomedclip_kd_state.json` (gitignored) at session start.
>
> **Current phase: 7 — plan update (Phase 6c failure recovery; Phases 8–12 designed).** Phase 6c (job 1300) failed identically to 6a/6b. Next session executes Phase 8 (delete `distill_proj`/`img_proj`) → Phase 9 (gate CLIP, cold-start MoCo) → Phase 10 (warmup 500→1000, α_kd schedule) → Phase 11 (smoke + sbatch Phase 6d) → Phase 12 (final writeup).

## Experiment history

| Job | Phase | Key change | MIMIC-val i2t R@10 | Indiana i2t R@10 | Paired cos |
|-----|-------|-----------|-------------------|-----------------|------------|
| (old runs) | PubMedBERT teacher, JOINT_TRAINING_PLAN | bs=16→32, FAISS hard-neg | 8.55–8.98% | 3.10–4.17% | 0.21–0.29 |
| 1285 | 4 | BiomedCLIP text KD, in-batch, cancelled ~1700 steps | 8.75% best | — | — |
| 1288 | 5a | + MoCo asymmetric (i2t only) | 5.5% plateau | — | — |
| 1290 | 5b | + MoCo symmetric + img_queue | 0.5% (broken: random img_queue = noise) | — | — |
| 1291 | 5c | + MoCo symmetric, text_queue only | 9.99% best | **3.36%** (eval) | **0.226** |
| 1297 | 6a | bypass img_proj + alpha_kd=1.0 | CANCELLED | 0.49% (random) | clip_loss 2.88→3.47 |
| 1299 | 6b | bypass img_proj + alpha_kd=0.3 | CANCELLED | 0.46% (random) | clip_loss 3.0→3.47 |
| 1300 | 6c | bypass img_proj + direct KD on z_text | CANCELLED | 0.49% (random) | clip_loss 2.97→3.45 |
| TBD | **6d** | **delete dead modules + gate CLIP + cold-start MoCo + warmup 500→1000** | TBD | TBD | TBD |

**Phase 5c root-cause diagnosis (2026-05-05):** `clip_model.visual` already outputs BiomedCLIP-projected 512-d embeddings (joint space). The `img_proj` (random-init `512→GELU→512` MLP) was applied on top, distorting them. The CLIP loss (β=1.0) dominated KD (α=0.3) and pulled Mamba text toward the distorted space — explaining why paired cosine stayed at 0.22-0.29 across ALL runs since Phase 4, identical to the PubMedBERT era.

**Phase 6c root-cause diagnosis (2026-05-05, job 1300):** Direct KD on `z_text` was implemented but failed identically (val/clip_loss 2.97→3.45 across 5 epochs, i2t R@10=0.49%). Three vulnerabilities remained, all unaddressed by 6c: (1) `distill_proj` and `img_proj` are still in the optimizer as dead weights (cosmetic, but future bug surface). (2) **CLIP loss is NOT gated by `freeze_text_encoder_steps`** — it runs from step 0 against a `z_text` still in GPT-2 space, polluting proj_head gradients before KD can stabilise it. (3) **MoCo `text_queue` enqueues from step 0** — by step 500 the 16K queue is full of stale GPT-2-space embeddings, and the post-unfreeze InfoNCE loss against this stale queue produces random gradients (the divergence pattern observed in jobs 1297/1299/1300). Phase 6d combines four complementary fixes: delete dead modules, gate CLIP, cold-start the queue, extend warmup 500→1000 with α_kd schedule.

## Context

Three full training runs on MIMIC-CXR (Phase 5 / 5b / 7 of the OLD plan) plateaued the CLIP retrieval R@10 at a **structural ceiling**:

| Run | Config change | MIMIC val R@10 | Indiana i2t R@10 | Paired cos |
|---|---|---|---|---|
| Phase 5 (v1) | bs=16, α_kd=0.3 | 8.55% | 3.10% | — |
| Phase 5b (v2) | bs=32 (2× negs), α_kd=0.1 | 8.95% | 4.17% | 0.29 / 0.21 |
| Phase 7 (FAISS hard-neg, K=4) | resume v2 + mined negs | 8.98% | — | unchanged |

Doubling negatives bought +0.4 pp; hard-neg mining bought zero. Paired cosine 0.21–0.29 confirms **BiomedCLIP visual features and the GPT-2-tokenized Mamba text embeddings live in fundamentally different spaces**, and the 2-layer MLP `img_proj` cannot bridge them. All in-batch InfoNCE strategies are exhausted.

Audit of the code (after JOINT_TRAINING_PLAN Phase 2) shows three of the five PDF-identified gaps **already resolved**: 2-layer MLP `img_proj` (`lightning_module.py:348-352`), `AttentionPooling` (`hybrid_lm.py:313-349`), `logit_scale` clamped to `[1.0, 100.0]` (`lightning_module.py:464-465`). Two PDF gaps remain open: MoCo dynamic queue + momentum encoder, and R-Drop consistency. The current KD teacher is **PubMedBERT** at the 768-d CLS — a space related to but not coincident with the 512-d joint space BiomedCLIP's image encoder lives in.

The supervisor's Option D — **distil from BiomedCLIP's own text tower into the Mamba text encoder** — is the strongest single fix for the modality gap. BiomedCLIP image and text live by construction in the same 512-d post-projection metric. Pulling Mamba toward that target eliminates the bridging burden currently dumped on `img_proj`.

## Decision

**Primary fix: Option D — distil from BiomedCLIP's text tower at the 512-d post-projection joint embedding.** Replace PubMedBERT teacher with `open_clip` BiomedCLIP's text submodule.

**Secondary fixes (PDF gaps still open):**
- **MoCo dynamic queue + momentum encoder** (Gap 1) — decouples negative-pool size from batch size. Addresses the 8.95 % in-batch ceiling without the bs=64 OOM cost.
- **R-Drop consistency** (Gap 4) — symmetric KL between two dropout-masked text forwards.

**Rejected alternatives.** *Option A — unfreeze BiomedCLIP ViT*: pulls visual side toward Mamba; we want the opposite. *Option B — 3-layer img_proj with LayerNorm*: marginal capacity gain over 2-layer MLP; Phase 7 already proved more bridge capacity does not help when the underlying text space is misaligned. *Cross-attention bridge*: high VRAM, weeks of tuning, deferred.

**Strategy: staged escalation, not single-shot.** Train Option D alone first (Phase 4); if R@10 ≥ target, done. Else add MoCo (Phase 5). Else add R-Drop (Phase 6). Each layer is conditional on the prior failing — staged costs ~3× wall-clock in the worst case but yields a publishable ablation; a bundled run that fails leaves us guessing.

### Before / after data flow

```
BEFORE                             AFTER (Phase 4 — Option D)
──────                             ───────────────────────────
GPT-2 toks ─▶ Mamba ─▶ proj_head   GPT-2 toks ─▶ Mamba ─▶ proj_head
                       │ 512-d                              │ 512-d
                       ├──▶ CLIP loss ◀── img_proj          ├──▶ CLIP loss ◀── img_proj
                       │                  ▲                 │                  ▲
                       │                  │ 512-d           │                  │ 512-d
                       │       BiomedCLIP-vit (frozen)      │       BiomedCLIP-vit (frozen)
                       │                                    │
                       └──▶ distill_proj 512→768            └──▶ distill_proj 512→512
                                          ▲                                    ▲
                          PubMedBERT-toks ─┘                BiomedCLIP-toks ───┘
                          PubMedBERT CLS 768-d              BiomedCLIP encode_text 512-d
                          (different space from img)        (SAME space as img — by construction)
```

## Critical files

| File | Purpose | Phase |
|---|---|---|
| `BIOMEDCLIP_KD_PLAN.md` (NEW, repo root) | Repo-side mirror of this plan | 1 |
| `biomedclip_kd_state.json` (NEW, gitignored) | Resumable state | 1 |
| `.gitignore:84-93` | Allowlist new plan file (`*.md` is globally ignored at line 84); add state file under local-state block | 1 |
| `CLAUDE.md` Session Bootstrap section (lines 5-7) | Replace `JOINT_TRAINING_PLAN.md` + `joint_training_state.json` references with `BIOMEDCLIP_KD_PLAN.md` + `biomedclip_kd_state.json`. **CLAUDE.md exists** (10 KB, project root) — verified. | 1 |
| `JOINT_TRAINING_PLAN.md` | Add deprecation banner; keep file as historical record | 1 |
| `hybrid_xmamba/training/lightning_module.py:684-743` (`JointMultiTaskLightningModule.__init__`) | Swap teacher: PubMedBERT → BiomedCLIP-text submodule; resize `distill_proj` (currently `Sequential(Linear(student,768)→GELU→Linear(768,768))` at line 739) to `Sequential(Linear(student,512)→GELU→Linear(512,512))` | 2 |
| `hybrid_xmamba/training/lightning_module.py:818-863` (`_joint_step` teacher forward) | Replace `t_out.last_hidden_state[:, 0, :]` (line 855) with `teacher.encode_text(input_ids)` | 2 |
| `hybrid_xmamba/training/lightning_module.py:616-617` (parent class `HybridContrastiveLightningModule`) | Has its own `distill_proj = Linear(student_dim, teacher_dim)` and uses `teacher.config.hidden_size`. **Out of scope** for Phase 2 (only joint mode is being retrained), but document so a future bug doesn't surprise. | 2 (note only) |
| `scripts/train_contrastive.py:339-405` (`MIMICJointDataset`) | Keep `teacher_input_ids/attention_mask` keys, but tokenize them with the BiomedCLIP (PubMedBERT) tokenizer instead of an arbitrary HF tokenizer. Student GPT-2 tokenization (`configs/dataset/mimic_cxr.yaml:25`) is **unchanged** | 2 |
| `scripts/train_contrastive.py:633-655` (joint dispatch) | Replace `AutoModel.from_pretrained(distill_cfg.teacher_model)` + `AutoTokenizer` path with a helper that loads BiomedCLIP via `open_clip.create_model_from_pretrained('hf-hub:…')` + `open_clip.get_tokenizer('hf-hub:…')` when `distill_cfg.teacher == "biomedclip_text"` | 2 |
| `configs/distill/biomedclip_kd_joint.yaml` (NEW) | `teacher: biomedclip_text`, α_kd=0.3, β_clip=1.0, γ_simcse=0.1, freeze_text_encoder_steps=500 | 2 |
| `scripts/smoke_test_joint.py:63-101` | Replace `MockPubMedBERT` with `_MockBiomedCLIPText` (returns 512-d via `encode_text`); patch open_clip mock to expose `.encode_text` and `get_tokenizer` | 3 |
| `tests/test_willi_parity.py:640-734` | Update `test_joint_module_all_losses_finite` to use a 512-d `encode_text` stub; assert `distill_proj[-1].out_features == 512`; new `test_biomedclip_kd_config_values` | 2 |
| `scripts/train_biomedclip_kd.sh` (NEW) | SLURM wrapper; copy `train_joint_mimic_v2.sh` shape; 12 h walltime, A100 40 GB | 4 |
| `hybrid_xmamba/training/moco_queue.py` (NEW) | `MoCoQueue` (FIFO embedding bank) + `MomentumEncoder` (EMA wrapper); ~150 LoC | 5 |
| `hybrid_xmamba/training/lightning_module.py` (`_joint_step` line ~879) | **Phase 6**: `z_img = F.normalize(z_img_raw.float(), dim=-1)` — bypass `img_proj`; `clip_model.visual` already projects to joint space | 6 |
| `configs/distill/biomedclip_kd_joint.yaml` | **Phase 6**: `alpha_kd: 1.0` (was 0.3) — KD must dominate now that image side is clean | 6 |
| `scripts/train_biomedclip_kd_phase6.sh` (NEW) | SLURM wrapper, Phase 6 training, `experiment_name=biomedclip_kd_phase6` | 6 |
| `scripts/eval_biomedclip_kd_phase6.sh` (NEW) | SLURM eval for Phase 6 best checkpoint | 6 |
| `hybrid_xmamba/training/lightning_module.py:760-764` | **Phase 8**: delete `self.distill_proj` block | 8 |
| `hybrid_xmamba/training/lightning_module.py:~222` (parent) | **Phase 8**: delete `self.img_proj` and `self.distill_proj`; remove from `configure_optimizers` | 8 |
| `hybrid_xmamba/training/lightning_module.py` ckpt-load path | **Phase 8**: `strict=False` (or pre-strip) for back-compat with Phase 5c checkpoints | 8 |
| `hybrid_xmamba/training/lightning_module.py:873-899` | **Phase 9**: gate CLIP block on `global_step >= freeze_text_encoder_steps`; skip queue enqueue when gated | 9 |
| `hybrid_xmamba/training/lightning_module.py:478-490` | **Phase 9**: at unfreeze step, call `momentum_encoder.copy_from(model)` + `text_queue.reset()` | 9 |
| `hybrid_xmamba/training/moco_queue.py` | **Phase 9**: add `MomentumEncoder.copy_from()` + `MoCoQueue.reset()` | 9 |
| `configs/distill/biomedclip_kd_joint.yaml` | **Phase 10**: `freeze_text_encoder_steps: 500→1000`; add `alpha_kd_warmup: 1.0` + `alpha_kd_post: 0.3` | 10 |
| `hybrid_xmamba/training/lightning_module.py` `_joint_step` | **Phase 10**: effective α_kd schedule + `cos_text_teacher` diagnostic logging | 10 |
| `tests/test_willi_parity.py` | **Phase 8/9/10**: deletion asserts + gating tests + α_kd schedule test | 8/9/10 |
| `scripts/smoke_test_joint.py` | **Phase 11**: assert `l_clip==0` during warmup, non-zero after; queue empty during warmup | 11 |
| `scripts/train_biomedclip_kd_phase6d.sh` (NEW) | **Phase 11**: SLURM wrapper, `experiment_name=biomedclip_kd_phase6d` | 11 |
| `scripts/eval_biomedclip_kd_phase6d.sh` (NEW) | **Phase 11**: SLURM eval for best Phase 6d checkpoint | 11 |

### Verified facts (corrects assumptions in original draft)

1. **Tokenizers are NOT shared.** Student backbone uses GPT-2 (`configs/dataset/mimic_cxr.yaml:25` confirmed: `tokenizer: "gpt2"`). BiomedCLIP's text tower uses PubMedBERT WordPiece. Keep `teacher_input_ids` / `teacher_attention_mask` in the batch — only swap the *tokenizer source* on the teacher side.
2. **Line numbers refreshed** against current `lightning_module.py` (914 lines): `HybridContrastiveLightningModule` at 222, `JointMultiTaskLightningModule` at 684, `_nt_xent_loss` at 442, logit_scale clamp at 464-465, `_joint_step` at 818, joint `distill_proj` at 738-739, joint teacher forward at 855.
3. **Joint `distill_proj` is already a 2-layer MLP** (`Sequential(Linear→GELU→Linear)`, line 739). Change is to resize from 768 (PubMedBERT hidden) → **512** (BiomedCLIP joint dim) and replace the `teacher.config.hidden_size` accessor (line 738) with a constant.
4. **`*.md` is gitignored globally** (`.gitignore:84`) with `JOINT_TRAINING_PLAN.md` allowlisted at line 87. The new plan file MUST be added to the allowlist or it will not commit.
5. **CLAUDE.md DOES exist** at repo root (10 KB, verified) with a `Session Bootstrap` section (lines 5-7) that explicitly references `JOINT_TRAINING_PLAN.md` and `joint_training_state.json`. The Phase 1 CLAUDE.md edit is real and required.

## Phases

### Phase 1 — Bootstrap & deprecate old plan ✅ COMPLETE
- [x] Write `BIOMEDCLIP_KD_PLAN.md` at repo root.
- [x] Write `biomedclip_kd_state.json` at repo root (gitignored).
- [x] Edit `.gitignore`: allowlist `BIOMEDCLIP_KD_PLAN.md`; add `biomedclip_kd_state.json` to local-state block.
- [x] Edit `CLAUDE.md` Session Bootstrap: repoint to `BIOMEDCLIP_KD_PLAN.md` + `biomedclip_kd_state.json`.
- [x] Add deprecation banner to `JOINT_TRAINING_PLAN.md`.
- [x] `bash scripts/validate_for_willi.sh` green (42 passed).

### Phase 2 — BiomedCLIP text teacher wiring ✅ COMPLETE
- [x] **2A** — `_load_biomedclip_text_teacher()` helper in `lightning_module.py`.
- [x] **2B** — `teacher_dim = 512` constant; `distill_proj` resized to `512→GELU→512`.
- [x] **2C** — `_joint_step`: `teacher.encode_text(t_ids)` replaces `last_hidden_state[:,0,:]`.
- [x] **2D** — `MIMICJointDataset`: open_clip tokenizer adapter (`_teacher_is_hf` branch); `teacher_max_length=256`.
- [x] **2E** — `train_contrastive.py` joint dispatch: `teacher=="biomedclip_text"` branch added.
- [x] **2F** — `configs/distill/biomedclip_kd_joint.yaml` created.
- [x] **2G** — `tests/test_willi_parity.py` updated: `encode_text` stub; `distill_proj[-1].out_features==512`; `test_biomedclip_kd_config_values`.
- [x] **2H** — `validate_for_willi.sh` green (43 passed).

### Phase 3 — CPU smoke test ✅ COMPLETE
- [x] `scripts/smoke_test_joint.py`: `_MockBiomedCLIPText` with `encode_text(B,512)`; `get_tokenizer` stub; frozen-teacher grad assertion.
- [x] 3/3 tests pass on CPU.

### Phase 4 — Joint training v3, BiomedCLIP-text-KD only ✅ COMPLETE (job 1285)
- [x] `scripts/train_biomedclip_kd.sh` created and submitted.
- [x] **Verdict:** Cancelled at ~1700 steps. Best `i2t_R@10=8.75%`. BiomedCLIP teacher converges 3× faster than PubMedBERT but hits same ~9% ceiling from 31 in-batch negatives. Advancing to Phase 5 (MoCo).

### Phase 5 — MoCo dynamic queue ✅ COMPLETE (jobs 1288 / 1290 / 1291)

Three iterations to reach correct design:

**5a — Asymmetric MoCo (job 1288):** `i2t`-only queue loss → `t2i > i2t` imbalance, plateau at 5.5%. Fixed.

**5b — Symmetric + img_queue (job 1290):** img_queue seeded with random unit vectors → `t2i` InfoNCE starts at `log(16385)=9.70` (theoretical max) = pure noise gradients. R@10 stuck at 0.5%. Fixed by removing img_queue.

**5c — Symmetric, text_queue only (job 1291):** ✅ Correct design.
- `i2t`: `z_img` queries vs `[z_text_k | text_queue]` (16K+ text negatives)
- `t2i`: `z_text` queries vs `z_img` in-batch only (frozen ViT = deterministic, no queue needed)
- Best MIMIC-val `i2t_R@10 = 9.99%`, `t2i_R@10 = 9.73%`
- **Eval result (2026-05-05):** Indiana `i2t_R@10 = 3.36%`, MIMIC-val `i2t_R@10 = 8.36%`, paired cos = 0.226

- [x] `hybrid_xmamba/training/moco_queue.py` — `MoCoQueue` + `MomentumEncoder`.
- [x] Wired into `JointMultiTaskLightningModule`; `text_queue` + `momentum_encoder`; `_moco_clip_loss_symmetric`.
- [x] Config knobs: `moco_queue_size: 16384`, `moco_momentum: 0.999` in `biomedclip_kd_joint.yaml`.
- [x] Tests: queue shape, EMA delta, symmetric loss grad flow (49 passed).
- [x] `train_biomedclip_kd_moco.sh` + `eval_biomedclip_kd_moco.sh` created.

**Phase 5c root-cause finding:** Paired cosine 0.22-0.29 is IDENTICAL to PubMedBERT era. `clip_model.visual` already outputs BiomedCLIP's projected 512-d joint-space embeddings. The `img_proj` MLP (random-init `512→GELU→512`) distorted them AFTER the fact. CLIP loss (β=1.0) dominated KD (α=0.3) and pulled Mamba text toward the distorted space instead of BiomedCLIP joint space. The BiomedCLIP text KD pivot was correct but neutralised by this bug.

### Phase 6 — Fix img_proj (two iterations)

#### Phase 6a ✗ FAILED (job 1297, cancelled)
- [x] `lightning_module.py` `_joint_step`: bypass img_proj — `z_img = F.normalize(z_img_raw.float(), dim=-1)`.
- [x] `alpha_kd: 1.0` — **WRONG.** KD (→BiomedCLIP text) and CLIP (→BiomedCLIP image) point in different directions even in the joint space (matched-pair cos ~0.5–0.7, not 1.0). Equal α=β=1.0 causes gradient conflict. val/clip_loss diverged from 2.88 → 3.47 over 5 epochs; i2t R@10 = 0.49% (near-random). Cancelled.

#### Phase 6b ✗ FAILED (job 1299, cancelled)
Same failure: clip_loss 3.0→3.47, i2t R@10=0.46%. Root cause was deeper: `distill_proj` absorbs ALL KD gradient — it only teaches `distill_proj` to map `z_text` → BiomedCLIP space, `z_text` itself (used by CLIP) stays in GPT-2 space. Without `img_proj` as bridge, CLIP has zero traction from step 1.

#### Phase 6c ✗ FAILED (job 1300, cancelled 2026-05-05)
**Root cause (then-believed):** `distill_proj` acts as a gradient absorber — KD never reaches `z_text` directly.

```
OLD:  KD = 1 - cos(distill_proj(z_text), BCT)   ← trains distill_proj, not z_text
NEW:  KD = 1 - cos(z_text, BCT)                  ← trains z_text directly
```

During 500-step frozen warm-up, `projection_head` (not `distill_proj`) learns to map Stage-0 Mamba outputs → BiomedCLIP text space. By step 500, `z_text ≈ BCT`. When CLIP engages, `z_text` (≈ BCT) and `z_img` (BCI, bypassing img_proj) are both in BiomedCLIP joint space — CLIP can converge.

- [x] `lightning_module.py` `_joint_step`: KD directly on `z_text` — `l_kd = (1 - cos(z_text, t_emb)).mean()`; `distill_proj` still exists (optimizer compat) but removed from KD path.
- [x] `lightning_module.py` `_joint_step`: img_proj bypass retained.
- [x] `tests/test_willi_parity.py`: `test_joint_module_all_losses_finite` updated — `embed_dim=512`, asserts `projection_head` gets gradient (not `distill_proj`).
- [x] `scripts/train_biomedclip_kd_phase6.sh` updated (Phase 6c comment block).
- [x] `validate_for_willi.sh` green (24 tests, all 6 gates).
- [x] Job 1300 submitted.
- [x] **Verdict:** CANCELLED. val/clip_loss started at 2.97 (above Phase 5c floor 2.47), diverged 2.97→3.02→3.14→3.28→3.45 over 5 epochs. i2t R@10 = 0.49% (near-random). Direct KD on `z_text` was implemented correctly but three vulnerabilities remained: (a) CLIP loss runs from step 0 (no `freeze_text_encoder_steps` gate in `_joint_step`), polluting `proj_head` gradients before KD can stabilise; (b) MoCo `text_queue` enqueues from step 0, filling with stale GPT-2-space embeddings that produce random InfoNCE gradients post-unfreeze; (c) `distill_proj`/`img_proj` still in optimizer as dead weights. Advancing to Phase 8+ (combined fix → Phase 6d run).

### Phase 7 — Plan update (Phase 6c failure recovery design) ⏳ IN PROGRESS
**Scope: only plan files. NO code, NO tests, NO commits in this phase.**
- [x] Refreshed Plan-of-Record header (Current phase line) and experiment history table (job 1300 row, Phase 6d placeholder).
- [x] Added Phase 6c root-cause diagnosis to Context (immediately above this Phases section).
- [x] Folded Phase 6c verdict into the Phase 6 sub-section above.
- [x] Inserted Phases 8 / 9 / 10 / 11 below; renumbered final eval+writeup to **Phase 12**.
- [ ] Update `biomedclip_kd_state.json` (`current_phase` → `7_plan_update_pending`; `phase6.6c_verdict`; pre-create phase8–phase12 keys; append note for job 1300).
- [ ] `bash scripts/validate_for_willi.sh` (sanity, doc-only changes — should remain green).
- [ ] Commit (user-approved): `plan: phase 6c verdict + phases 8-12 for queue/CLIP gating + dead-module deletion`.

### Phase 8 — Architectural cleanup (delete dead modules)
**Goal:** remove `distill_proj` and `img_proj` from the architecture entirely (not just the forward path), eliminating ~500K wasted optimizer-state params and future regression surface.

- [ ] **8A** — Delete `self.distill_proj` block (`lightning_module.py:760-764`) from `JointMultiTaskLightningModule.__init__`.
- [ ] **8B** — In parent `HybridContrastiveLightningModule` (line ~222), delete `self.img_proj` and `self.distill_proj` definitions used by Stage-2 CLIP / Stage-1 KD modes. Joint mode no longer references either.
- [ ] **8C** — `configure_optimizers` (joint + parent): remove `distill_proj.parameters()` and `img_proj.parameters()` from the head_lr param group.
- [ ] **8D** — `grep -rn "distill_proj\|img_proj" hybrid_xmamba/ scripts/` → must be zero in non-test code paths.
- [ ] **8E** — Back-compat: load existing Phase 5c checkpoints with `strict=False` (or pre-strip `distill_proj.*` / `img_proj.*` keys). Add a smoke test that loading a frozen 5c-shaped state-dict succeeds.
- [ ] **8F** — `tests/test_willi_parity.py`: drop `distill_proj.out_features == 512` assertion; add `assert not hasattr(module, "distill_proj")` and `assert not hasattr(module, "img_proj")`.
- [ ] **8G** — `bash scripts/validate_for_willi.sh` green.

### Phase 9 — Curriculum gating (CLIP off + MoCo cold-start)
**Goal:** prevent contamination of `proj_head` and the MoCo queue during the KD-only warm-up window. CLIP loss and queue enqueue are both off until `global_step >= freeze_text_encoder_steps`. At the unfreeze boundary, hard-resync the momentum encoder from the live model and reset the queue.

- [ ] **9A** — `_joint_step` (lines 873–899): wrap CLIP block in `if self.global_step >= self.freeze_text_encoder_steps:`. Gated-off branch sets `l_clip = torch.tensor(0.0, device=z_text.device)` and skips both `momentum_encoder.encode` and `text_queue.enqueue`.
- [ ] **9B** — `hybrid_xmamba/training/moco_queue.py`: add `MomentumEncoder.copy_from(model)` (hard weight copy; reset EMA buffers if any). Add `MoCoQueue.reset()` (zero the buffer, reset pointer).
- [ ] **9C** — `on_train_batch_start` (lines 478–490): on the unfreeze step, additionally call `self.momentum_encoder.copy_from(self.model)` and `self.text_queue.reset()`.
- [ ] **9D** — Tests: `test_clip_loss_gated_during_warmup`, `test_moco_queue_cold_start`, `test_momentum_resync_at_unfreeze`.
- [ ] **9E** — `bash scripts/validate_for_willi.sh` green.

### Phase 10 — Hyperparameter rebalance + alignment diagnostic
**Goal:** give the warm-up enough time AND signal strength to bring `z_text` into BCT space before CLIP turns on. Boost α_kd while CLIP is gated off (no gradient conflict possible), then decay back to the safe post-warmup value.

- [ ] **10A** — `configs/distill/biomedclip_kd_joint.yaml`: `freeze_text_encoder_steps: 500 → 1000`.
- [ ] **10B** — Add `alpha_kd_warmup: 1.0` and `alpha_kd_post: 0.3` to config; thread through `JointMultiTaskLightningModule.__init__`.
- [ ] **10C** — `_joint_step`: `effective_alpha = alpha_kd_warmup if global_step < freeze_text_encoder_steps else alpha_kd_post`.
- [ ] **10D** — Add diagnostic log `train/cos_text_teacher = cos(z_text, t_emb).mean()` (and matching `val/`) every step; this is the kill-job signal.
- [ ] **10E** — Test: assert effective alpha switches at the threshold; assert diagnostic is logged.
- [ ] **10F** — `bash scripts/validate_for_willi.sh` green.

### Phase 11 — Smoke + SLURM scripts → submit Phase 6d run
- [ ] **11A** — `scripts/smoke_test_joint.py`: parametrise `freeze_text_encoder_steps` and assert `l_clip == 0` for steps `< warmup`, then non-zero. Confirm queue is empty during warmup.
- [ ] **11B** — `scripts/train_biomedclip_kd_phase6d.sh` (NEW): copy Phase 6 script, `experiment_name=biomedclip_kd_phase6d`, comment block referencing this plan.
- [ ] **11C** — `scripts/eval_biomedclip_kd_phase6d.sh` (NEW): eval wrapper for best Phase 6d checkpoint.
- [ ] **11D** — Verification gates: `pytest tests/ -m "not cuda and not slow" -v` green; smoke test green; `validate_for_willi.sh` 6/6 green.
- [ ] **11E** — Commit + push.
- [ ] **11F** — `sbatch scripts/train_biomedclip_kd_phase6d.sh` on willi.
- [ ] **11G** — **Monitor (key signals):**
  - **Step 100/500/900 (warm-up phase):** `train/cos_text_teacher` must rise from ~0 to **≥0.7**. If `<0.5` by step 800, KILL the job — α_kd_warmup too low or proj_head LR wrong.
  - **Step 1000 (CLIP turns on):** first `val/clip_loss` reading must be **below 2.47** (Phase 5c floor). If above, `z_text` is still not in BCT space — investigate before letting it run.
  - **After 5 epochs:** `i2t R@10` ≥ 12% (above Phase 5c's 9.99% peak) is the success signal.
- [ ] **11H** — Eval: `sbatch scripts/eval_biomedclip_kd_phase6d.sh`. Record verdict in state JSON.
- [ ] **11I** — **Decision gate (Indiana i2t R@10):**
  - ≥ 40% → SUCCESS
  - 25–40% → PARTIAL — manuscript quality
  - 15–25% → MARGINAL — try R-Drop or unfreeze ViT last 2 blocks
  - < 15% → re-examine warmup length / α_kd_warmup; consider dropping momentum encoder

### Phase 12 — Final eval + writeup (replaces old Phase 7)
- [ ] Cross-checkpoint comparison table: Phase 4 / 5c / 6c / 6d best ckpts × {Indiana, MIMIC-val} × {i2t, t2i} R@1/5/10 + paired cosine.
- [ ] Ablation: which fix moved the needle (queue cold-start vs CLIP gating vs warmup length vs module deletion vs α_kd schedule).
- [ ] If best Indiana R@10 ≥ 15%, write up the recovery story: 6c root-cause → 6d combined fix → result.
- [ ] Update `biomedclip_kd_state.json` with final verdict.

## Verification

Per-phase gates (each must pass before advancing):

1. **Phase 1**: `bash scripts/validate_for_willi.sh` green; `git diff` shows only plan + .gitignore + CLAUDE.md + JOINT_TRAINING_PLAN.md banner.
2. **Phase 2**: `pytest tests/ -m "not cuda and not slow" -v` green (incl. updated joint-loss + new biomedclip_kd config tests); `validate_for_willi.sh` green.
3. **Phase 3**: 5-step smoke test passes locally on CPU; loss-finite + grad-flow assertions hold; teacher params unchanged.
4. **Phase 4**: SLURM job within 12 h walltime, no OOM; eval R@10 measured on both Indiana + MIMIC-val with `evaluate_cxr_retrieval.py`; verdict logged in state JSON.
5. **Phase 5/6**: queue/R-Drop unit tests pass; resumed training shows train_loss continuing to drop (sanity: not NaN, not stuck).
6. **Phase 7**: `BIOMEDCLIP_KD_PLAN.md` + `biomedclip_kd_state.json` updated; `validate_for_willi.sh` still green (no code changes); commit pushed.
7. **Phase 8**: `grep -rn "distill_proj\|img_proj" hybrid_xmamba/` returns zero non-test hits; checkpoint back-compat smoke test green; `validate_for_willi.sh` green.
8. **Phase 9**: `test_clip_loss_gated_during_warmup` green (l_clip==0 at step<warmup); `test_moco_queue_cold_start` green (queue empty during warmup); `test_momentum_resync_at_unfreeze` green.
9. **Phase 10**: effective-alpha switch test green; `train/cos_text_teacher` diagnostic visible in W&B/log output during smoke run.
10. **Phase 11**: 5-step smoke shows `l_clip[0:warmup]==0` then non-zero; `cos_text_teacher` rises monotonically across 5 steps (sign of life); SLURM job survives warmup window without `cos_text_teacher` collapse.

End-to-end smoke before each willi sbatch:
```
python scripts/smoke_test_joint.py --steps 5 --device cpu --config biomedclip_kd_joint
bash scripts/validate_for_willi.sh
```

## Resumability contract

1. Read `BIOMEDCLIP_KD_PLAN.md` + `biomedclip_kd_state.json` at session start.
2. Resume at `biomedclip_kd_state.json["current_phase"]`. Checkboxes in `BIOMEDCLIP_KD_PLAN.md` are ground truth.
3. After every state change: tick the matching checkbox + update `last_updated` (ISO 8601) + append a 1-line entry to `notes`.
4. If `biomedclip_kd_state.json` missing locally, regenerate from this file's checkbox state — gitignored on purpose.
5. Never re-run a checkpoint-producing phase without first reading `output_willi_server/*.log` and logging a verdict in the state JSON.

## Open questions

- **`alpha_kd_post` value:** 0.3 inherited from 6b. If Phase 6d marginal (15–25% Indiana R@10), sweep `{0.1, 0.3, 0.5}` post-warmup.
- **ViT unfreeze:** if Phase 6d MARGINAL, escalate by unfreezing BiomedCLIP ViT last 2 blocks (small image-side adaptation). Out of scope until 6d result.
- **Drop momentum encoder?** If queue cold-start + gating + extended warmup yields the win, the momentum encoder may be unnecessary overhead. Validate by a 6d-NoMomentum ablation if 6d succeeds.
- **Phase 6 key question (still open):** Does `clip_model.visual` for BiomedCLIP (TimmModel backbone) include the linear projection to 512-d joint space, or return raw ViT 768-d features? Verify in Phase 6d by logging `z_img_raw.shape[-1]` once at startup — must be 512. If 768: pass through `clip_model.visual_projection` manually before normalising.

## Resolved questions (archived)

- ~~BiomedCLIP tokenizer max length:~~ Confirmed 256 tokens via `open_clip.get_tokenizer`; `_teacher_is_hf` adapter in `MIMICJointDataset` handles it.
- ~~α_kd=0.3 vs 0.1:~~ Phase 4 (0.3) and old Phase 5b (0.1) both plateau at ~9% MIMIC. Not the bottleneck.
- ~~MoCo queue size:~~ K=16384 confirmed working. Not the bottleneck either — paired cos unchanged with 16K vs 31 negatives.
- ~~α_kd stability with bypass img_proj:~~ α_kd=1.0 broken in 6a; 0.3 broken in 6b/6c. Phase 6d resolves via schedule (1.0 during gated warmup, 0.3 post-warmup) — no gradient conflict possible while CLIP is gated off.
- ~~`distill_proj` / `img_proj` dead-weight:~~ Decided to delete entirely in Phase 8.
