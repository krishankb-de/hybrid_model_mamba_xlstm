# BiomedCLIP Text-KD Architectural Pivot — Plan-of-Record

> Supersedes `JOINT_TRAINING_PLAN.md` + `joint_training_state.json`. Resumable. Read this file + `biomedclip_kd_state.json` (gitignored) at session start.

## Context

Three full training runs on MIMIC-CXR (Phase 5 / 5b / 7) plateaued the CLIP retrieval R@10 at a **structural ceiling**:

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
| `hybrid_xmamba/training/lightning_module.py` (`_joint_step`) | Add `_rdrop_loss` helper; second forward + symmetric KL on text projections | 6 |

### Verified facts (corrects assumptions in original draft)

1. **Tokenizers are NOT shared.** Student backbone uses GPT-2 (`configs/dataset/mimic_cxr.yaml:25` confirmed: `tokenizer: "gpt2"`). BiomedCLIP's text tower uses PubMedBERT WordPiece. Keep `teacher_input_ids` / `teacher_attention_mask` in the batch — only swap the *tokenizer source* on the teacher side.
2. **Line numbers refreshed** against current `lightning_module.py` (914 lines): `HybridContrastiveLightningModule` at 222, `JointMultiTaskLightningModule` at 684, `_nt_xent_loss` at 442, logit_scale clamp at 464-465, `_joint_step` at 818, joint `distill_proj` at 738-739, joint teacher forward at 855.
3. **Joint `distill_proj` is already a 2-layer MLP** (`Sequential(Linear→GELU→Linear)`, line 739). Change is to resize from 768 (PubMedBERT hidden) → **512** (BiomedCLIP joint dim) and replace the `teacher.config.hidden_size` accessor (line 738) with a constant.
4. **`*.md` is gitignored globally** (`.gitignore:84`) with `JOINT_TRAINING_PLAN.md` allowlisted at line 87. The new plan file MUST be added to the allowlist or it will not commit.
5. **CLAUDE.md DOES exist** at repo root (10 KB, verified) with a `Session Bootstrap` section (lines 5-7) that explicitly references `JOINT_TRAINING_PLAN.md` and `joint_training_state.json`. The Phase 1 CLAUDE.md edit is real and required.

## Phases

### Phase 1 — Bootstrap & deprecate old plan (NO training, NO model code edits)
- [ ] Write `BIOMEDCLIP_KD_PLAN.md` at repo root (copy of this plan, repo-relative paths).
- [ ] Write `biomedclip_kd_state.json` at repo root: `{"current_phase": "1", "phase1": {}, …, "last_updated": "<ISO>", "notes": []}`.
- [ ] Edit `.gitignore`: add `!BIOMEDCLIP_KD_PLAN.md` next to existing `!JOINT_TRAINING_PLAN.md` allowlist (line 87); append `biomedclip_kd_state.json` to the local-state block (after line 93).
- [ ] Edit `CLAUDE.md` Session Bootstrap (lines 5-7): repoint to `BIOMEDCLIP_KD_PLAN.md` + `biomedclip_kd_state.json`. Preserve the structural advice (resume at `current_phase`, update `last_updated` after every change). Leave the `/Users/krish/.claude/plans/previously-i-ran-the-drifting-garden.md` historical pointer alone — orthogonal.
- [ ] Add a 5-line deprecation banner to top of `JOINT_TRAINING_PLAN.md` pointing to the new plan; do **not** delete the file.
- [ ] `bash scripts/validate_for_willi.sh` green (no Python edits — should be no-op).
- [ ] Commit: "Pivot to BiomedCLIP text-KD plan; deprecate JOINT_TRAINING_PLAN".

### Phase 2 — Code change: BiomedCLIP text teacher (~2 hours, single subphase)
- [x] **2A** — Add module-level helper `_load_biomedclip_text_teacher()` in `lightning_module.py`: calls `open_clip.create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')`, freezes the whole CLIP wrapper (`requires_grad=False`, `.eval()`), returns it (keep `encode_text` API; don't reach into `.text` directly).
- [x] **2B** — In `JointMultiTaskLightningModule.__init__` (line 738):
  - replace `teacher_dim = teacher.config.hidden_size` with `teacher_dim = 512` (constant; open_clip CLIP wrapper has no `.config.hidden_size`);
  - resize the existing `nn.Sequential(Linear(student,768) → GELU → Linear(768,768))` to `nn.Sequential(Linear(student,512) → GELU → Linear(512,512))` — still no bias;
  - keep `self.teacher = teacher` API (caller now passes a CLIP wrapper instead of `transformers.AutoModel`).
- [x] **2C** — In `_joint_step` (line 855), replace
  ```
  t_cls = F.normalize(t_out.last_hidden_state[:, 0, :].float(), dim=-1)
  ```
  with
  ```
  t_emb = self.teacher.encode_text(t_ids)            # (B, 512)
  t_emb = F.normalize(t_emb.float(), dim=-1)
  ```
  Keep the cosine-distance KD form: `1 - cos(distill_proj(z_text), t_emb)`.
- [x] **2D** — In `MIMICJointDataset` (`train_contrastive.py:339-405`): leave dataset shape (5 keys) and student tokenization untouched. Only swap the teacher tokenizer source — when `teacher_tokenizer` is the BiomedCLIP `open_clip` tokenizer, it's a `Callable[[List[str]], LongTensor]` not a HuggingFace `AutoTokenizer`, so add a small adapter inside the dataset constructor (`def _teacher_tok(text): ids = teacher_tok([text])[0]; return {"input_ids": ids, "attention_mask": (ids != 0).long()}`). teacher_max_length is fixed to 256 (BiomedCLIP context) — drop the cfg lookup.
- [x] **2E** — In `train_contrastive.py:633-655` (`if contrastive_mode == "joint":` block): when `distill_cfg.teacher == "biomedclip_text"`, load the teacher via the new helper and the BiomedCLIP `open_clip` tokenizer. Keep the legacy PubMedBERT branch behind `teacher: pubmedbert` for backwards compat (so `joint_mimic.yaml` still runs unchanged).
- [x] **2F** — Create `configs/distill/biomedclip_kd_joint.yaml`:
  ```yaml
  teacher: "biomedclip_text"   # NEW dispatch key
  alpha_kd: 0.3                # reset to v1 baseline; KD now PUSHES toward CLIP space
  beta_clip: 1.0
  gamma_simcse: 0.1
  backbone_lr: 1.0e-5
  head_lr: 3.0e-4
  freeze_text_encoder_steps: 500
  ```
  (No `teacher_model` / `teacher_dtype` / `teacher_max_length` keys — those are PubMedBERT-specific.)
- [x] **2G** — Extend `tests/test_willi_parity.py`:
  - update `test_joint_module_all_losses_finite` (lines 640-712): stub `encode_text` returning `(B, 512)` instead of `last_hidden_state` stub; assert `mod.distill_proj[-1].out_features == 512`;
  - new `test_biomedclip_kd_config_values` mirroring `test_joint_mimic_config_values`.
- [x] **2H** — `pytest tests/ -m "not cuda and not slow" -v` green; `bash scripts/validate_for_willi.sh` green. Commit.

### Phase 3 — CPU smoke test (~10 min)
- [ ] Adapt `scripts/smoke_test_joint.py`:
  - extend `_install_mock_open_clip` to add `.encode_text` on `_MockClipModel` returning `(B, 512)` and a `get_tokenizer` returning a stub callable;
  - replace `MockPubMedBERT` with a thin wrapper exposing `encode_text(input_ids) -> (B, 512)`;
  - assert: 5 steps, all 3 losses finite; gradient flow into backbone + `attn_pool.q` + `img_proj` + `distill_proj`; **no gradient** into BiomedCLIP teacher params.

### Phase 4 — Joint training v3 on willi A100 40GB (~6–8 h, BiomedCLIP-text-KD only)
- [ ] Create `scripts/train_biomedclip_kd.sh` (SLURM, copy of `train_joint_mimic_v2.sh`): same hyperparams except `+distill=biomedclip_kd_joint`, `experiment_name=biomedclip_kd_v3`, `output_dir=./outputs/biomedclip_kd_v3`. Init from `outputs/hybrid_70m_stage0_kd_pubmed/checkpoints/stage0_model_only.pt`. `bs=32`, `accum=4`, `max_steps=5000`, `val_check_interval=250`.
- [ ] Submit. Live monitor: step 250 `val/clip < 2.0`, step 1000 `R@10 ≥ 0.15`, step 2500 `R@10 ≥ 0.25`. Early-kill gate at step 2000: `R@10 < 0.12` AND not rising → `scancel`.
- [ ] Run `evaluate_cxr_retrieval.py` on Indiana + MIMIC-val using the best checkpoint.
- [ ] **Decision gate:**
  - R@10 ≥ 0.40 (MIMIC val) → SUCCESS, skip Phases 5–6.
  - R@10 ∈ [0.25, 0.40) → PARTIAL, advance to Phase 5 (MoCo).
  - R@10 ∈ [0.15, 0.25) → MARGINAL, advance to Phase 5; reassess after.
  - R@10 < 0.15 → BAD; do **not** proceed to MoCo. Re-examine teacher loading + dim alignment + smoke test before any further training.

### Phase 5 — MoCo dynamic queue (CONDITIONAL, only if Phase 4 R@10 < 0.40)
- [ ] Implement `hybrid_xmamba/training/moco_queue.py`:
  - `MoCoQueue(dim=512, K=16384)`: FIFO ring buffer, registered as buffer, normalized;
  - `MomentumEncoder(query_encoder, m=0.999)`: EMA copy with `requires_grad=False`.
- [ ] Wire into `JointMultiTaskLightningModule`: maintain a momentum copy of the student text encoder; image embeddings from each batch enqueued; CLIP loss denominator now sums over current-batch + queue keys.
- [ ] Add config knobs to `biomedclip_kd_joint.yaml`: `moco_queue_size: 16384`, `moco_momentum: 0.999`.
- [ ] Tests: queue tensor shape `(K, 512)` after K updates; EMA delta correctness on a 2-step toy.
- [ ] Retrain (`train_biomedclip_kd_moco.sh`). Re-eval. Decision gate identical to Phase 4.

### Phase 6 — R-Drop consistency (CONDITIONAL, only if Phase 5 R@10 < 0.40)
- [ ] Add a second forward in `_joint_step` with independent dropout → `z_text₁`, `z_text₂` post-projection.
- [ ] Symmetric KL: `L_rdrop = 0.5 (KL(softmax(z₁·z_img.T) || softmax(z₂·z_img.T)) + reverse)`, scaled by `alpha_rdrop`.
- [ ] Add `alpha_rdrop: 1.0` knob; default `0.0` keeps R-Drop off.
- [ ] Retrain (`train_biomedclip_kd_moco_rdrop.sh`). Re-eval.

### Phase 7 — Final eval + writeup
- [ ] Cross-checkpoint comparison table: Phase 4 / 5 / 6 best ckpts × {Indiana, MIMIC-val} × {i2t, t2i} R@1/5/10 + paired cosine.
- [ ] If best run R@10 ≥ 0.25, write up ablation: which fix moved the needle, by how much.
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

- BiomedCLIP `open_clip` tokenizer max length: confirm `open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')` returns a callable that pads/truncates to 256 tokens (BiomedCLIP context). If it caps at 77 (vanilla CLIP default) instead, MIMIC reports lose context — switch to `transformers.AutoTokenizer.from_pretrained('microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')` for tokenization while still using `open_clip.encode_text` for the forward.
- KD loss form: cosine-distance vs MSE on L2-normalized embeddings. Default to current cosine form; revisit only if KD loss saturates above 0.3.
- α_kd starting value: 0.3 chosen as v1 baseline. With BiomedCLIP teacher, KD and CLIP no longer compete for direction (same target space) — α=0.3 should be safer than v2's 0.1. May still need a sweep [0.3, 0.5, 0.7] in Phase 4 if R@10 sits in marginal band.
- Parent class `HybridContrastiveLightningModule.__init__` (line 616-617) uses single Linear `distill_proj` and `teacher.config.hidden_size`. Out of scope for Phase 2 (joint mode is what's retrained), but a future Stage 1/2 standalone run with a BiomedCLIP teacher would need the same fix there.
