# BiomedCLIP Text-KD Architectural Pivot — Plan-of-Record

> Supersedes `JOINT_TRAINING_PLAN.md` + `joint_training_state.json`. Resumable. Read this file + `biomedclip_kd_state.json` (gitignored) at session start.
>
> **Current phase: 6 — ready to submit.** Push branch, then `sbatch scripts/train_biomedclip_kd_phase6.sh`.

## Experiment history

| Job | Phase | Key change | MIMIC-val i2t R@10 | Indiana i2t R@10 | Paired cos |
|-----|-------|-----------|-------------------|-----------------|------------|
| (old runs) | PubMedBERT teacher, JOINT_TRAINING_PLAN | bs=16→32, FAISS hard-neg | 8.55–8.98% | 3.10–4.17% | 0.21–0.29 |
| 1285 | 4 | BiomedCLIP text KD, in-batch, cancelled ~1700 steps | 8.75% best | — | — |
| 1288 | 5a | + MoCo asymmetric (i2t only) | 5.5% plateau | — | — |
| 1290 | 5b | + MoCo symmetric + img_queue | 0.5% (broken: random img_queue = noise) | — | — |
| 1291 | 5c | + MoCo symmetric, text_queue only | 9.99% best | **3.36%** (eval) | **0.226** |
| 1297 | 6a | bypass img_proj + alpha_kd=1.0 | CANCELLED (diverged) | 0.49% → random | clip_loss 2.88→3.47 |
| TBD | **6b** | **bypass img_proj + alpha_kd=0.3** | TBD | TBD | TBD |

**Phase 5c root-cause diagnosis (2026-05-05):** `clip_model.visual` already outputs BiomedCLIP-projected 512-d embeddings (joint space). The `img_proj` (random-init `512→GELU→512` MLP) was applied on top, distorting them. The CLIP loss (β=1.0) dominated KD (α=0.3) and pulled Mamba text toward the distorted space — explaining why paired cosine stayed at 0.22-0.29 across ALL runs since Phase 4, identical to the PubMedBERT era.

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

#### Phase 6b 🔄 READY TO SUBMIT
**Key insight:** the img_proj bypass is the correct architectural fix. α_kd must stay at 0.3 — the KD and CLIP objectives are not fully aligned even in joint space.

- [x] `lightning_module.py` `_joint_step`: img_proj bypass retained.
- [x] `configs/distill/biomedclip_kd_joint.yaml`: `alpha_kd: 0.3` (reset from 1.0 — isolates architectural fix).
- [x] `tests/test_willi_parity.py`: `test_biomedclip_kd_config_values` asserts `alpha_kd==0.3`.
- [x] `scripts/train_biomedclip_kd_phase6.sh` updated (Phase 6b comment block).
- [x] `scripts/eval_biomedclip_kd_phase6.sh` created.
- [x] `validate_for_willi.sh` green (all 6 gates).
- [ ] **Cancel job 1297:** `scancel 1297`
- [ ] **Submit:** `sbatch hybrid_model_mamba_xlstm/scripts/train_biomedclip_kd_phase6.sh`
- [ ] **Monitor:** val/clip_loss should drop BELOW 2.47 (Phase 5c baseline) within first 3 epochs if fix works. If it starts below 2.47 and continues falling → img_proj bypass is confirmed working.
- [ ] **Eval:** `sbatch hybrid_model_mamba_xlstm/scripts/eval_biomedclip_kd_phase6.sh` after training.
- [ ] **Decision gate (Indiana i2t R@10):**
  - ≥ 40% → SUCCESS
  - 25–40% → PARTIAL — manuscript quality
  - 15–25% → MARGINAL — try R-Drop or unfreeze ViT last 2 blocks
  - < 15% → verify `clip_model.visual` output dim is 512 (print `z_img_raw.shape` in first step); if 512 confirmed, investigate BiomedCLIP paired cosine baseline

### Phase 7 — Final eval + writeup (pending Phase 6 result)
- [ ] Cross-checkpoint comparison table: Phase 4 / 5c / 6 best ckpts × {Indiana, MIMIC-val} × {i2t, t2i} R@1/5/10 + paired cosine.
- [ ] If best run Indiana R@10 ≥ 15%, write ablation: which fix moved the needle, by how much.
- [ ] Update `biomedclip_kd_state.json` with final verdict.

## Verification

Per-phase gates (each must pass before advancing):

1. **Phase 1**: `bash scripts/validate_for_willi.sh` green; `git diff` shows only plan + .gitignore + CLAUDE.md + JOINT_TRAINING_PLAN.md banner.
2. **Phase 2**: `pytest tests/ -m "not cuda and not slow" -v` green (incl. updated joint-loss + new biomedclip_kd config tests); `validate_for_willi.sh` green.
3. **Phase 3**: 5-step smoke test passes locally on CPU; loss-finite + grad-flow assertions hold; teacher params unchanged.
4. **Phase 4**: SLURM job within 12 h walltime, no OOM; eval R@10 measured on both Indiana + MIMIC-val with `evaluate_cxr_retrieval.py`; verdict logged in state JSON.
5. **Phase 5/6**: queue/R-Drop unit tests pass; resumed training shows train_loss continuing to drop (sanity: not NaN, not stuck).

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

- **Phase 6 key question:** Does `clip_model.visual` for BiomedCLIP (TimmModel backbone) include the linear projection to 512-d joint space, or does it return raw ViT 768-d features? If the latter, `z_img_raw` is 768-d and `F.normalize(z_img_raw)` is NOT in joint space. Verify in Phase 6 run by checking `z_img_raw.shape[-1]` early in training — should be 512. If 768: need to pass through `clip_model.visual_projection` manually before normalising.
- **α_kd stability (resolved for 6b):** α_kd=1.0 confirmed broken — gradient conflict with CLIP objective in job 1297. α_kd=0.3 is the safe value for Phase 6b. If 6b plateaus and a sweep is needed, try 0.5 only; do not go above 0.7.
- **img_proj param group:** `img_proj` still exists as a parameter group in the optimizer but receives zero gradient (bypassed in forward). This wastes ~500K params of optimizer state. Not worth fixing for one run, but remove for any future clean implementation.
- **Parent class HybridContrastiveLightningModule** (line ~222): still uses single Linear `distill_proj` and `teacher.config.hidden_size`. Out of scope — only joint mode is being retrained. Note for future standalone Stage 1/2 BiomedCLIP runs.

## Resolved questions (archived)

- ~~BiomedCLIP tokenizer max length:~~ Confirmed 256 tokens via `open_clip.get_tokenizer`; `_teacher_is_hf` adapter in `MIMICJointDataset` handles it.
- ~~α_kd=0.3 vs 0.1:~~ Phase 4 (0.3) and old Phase 5b (0.1) both plateau at ~9% MIMIC. Not the bottleneck.
- ~~MoCo queue size:~~ K=16384 confirmed working. Not the bottleneck either — paired cos unchanged with 16K vs 31 negatives.
