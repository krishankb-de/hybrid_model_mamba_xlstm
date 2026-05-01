# Joint Multi-Task CLIP+KD+SimCSE on MIMIC-CXR — Plan-of-Record

> **Resumable plan-of-record.** Read this file + `joint_training_state.json` (gitignored, repo root) before any work. Resume at `joint_training_state.json["current_phase"]`. Checkboxes here are ground truth.
>
> **Full plan with all detail / rationale:** `/Users/krish/.claude/plans/previously-i-ran-the-drifting-garden.md`

## Why this plan exists

Stage 2 CLIP on Indiana CXR (run 1235) plateaued at `val/i2t_R@10 = 0.207`, `R@1 = 0.0538` (target ≥ 0.40). Severe overfit (train_loss 0.5 vs val_loss 3.13) caused by: (1) Indiana too small (~3.7K pairs); (2) `img_proj = Identity()` not aligning BiomedCLIP image space to text space; (3) mean pooling weighting filler tokens equally with clinical findings.

Per supervisor (Strategies 1-4): collapse Stage 1+Stage 2 into one **joint training** on MIMIC-CXR (~30K pairs, `itsanmolgupta/mimic-cxr-dataset`) with `L = α·KD(PubMedBERT) + β·CLIP + γ·SimCSE`. Reuse Stage 0 LM checkpoint (PPL=13.10) as backbone init. Apply Strategies 3 (img_proj MLP) and 4 (attention pooling) as compounding fixes. Strategy 2 (FAISS hard negatives) only if joint training lands in [0.25, 0.40).

## Critical files

| File | Purpose | Phase |
|---|---|---|
| `.gitignore`, `CLAUDE.md` | Old plan refs purged; CLAUDE.md points here | 0 (done) |
| `JOINT_TRAINING_PLAN.md`, `joint_training_state.json` | This plan + resumable state | 0 |
| `outputs/hybrid_70m_stage0_kd_pubmed/checkpoints/stage0_model_only.pt` | REUSE — Stage 0 backbone | 1 |
| `hybrid_xmamba/training/lightning_module.py:346-349` | Identity → 2-layer MLP `img_proj` | 2A |
| `hybrid_xmamba/models/hybrid_lm.py:313-400` + `configuration_hybrid.py` | Add `AttentionPooling` + `pooling_strategy` field | 2B |
| `hybrid_xmamba/training/lightning_module.py` | New `JointMultiTaskLightningModule` | 2C |
| `configs/dataset/mimic_cxr.yaml` (NEW) | MIMIC-CXR dataset config | 2D |
| `configs/distill/joint_mimic.yaml` (NEW) | α=0.3, β=1.0, γ=0.1, PubMedBERT teacher | 2D |
| `scripts/train_contrastive.py` | Add `contrastive_mode=joint` dispatch | 2C |
| `scripts/train_joint_mimic.sh` (NEW) | SLURM wrapper, 12h, A100 40GB | 2F |
| `tests/test_willi_parity.py` | Add img_proj/pooling/joint-loss assertions | 2E |

## Phases

### Phase 0 — Cleanup + bootstrap (no training)
- [x] Delete `PHASE_PLAN.md`, `phase_state.json`, `STAGE1_RECOVERY_PLAN.md`, `stage1_recovery_state.json`.
- [x] Edit `.gitignore`: drop stale entries, add `joint_training_state.json`, allow `JOINT_TRAINING_PLAN.md`.
- [x] Edit `CLAUDE.md` Session Bootstrap to point here.
- [x] Create `JOINT_TRAINING_PLAN.md` (this file).
- [x] Create `joint_training_state.json`.
- [ ] Commit "Replace 3-stage plan with joint-training plan; delete stale plan files".

### Phase 1 — Verify Stage 0 reusability (read-only, ~15 min)
- [ ] Confirm `outputs/hybrid_70m_stage0_kd_pubmed/checkpoints/stage0_model_only.pt` exists on willi.
- [ ] Re-eval PPL on willi: `sbatch scripts/eval_stage0_lm.sh` → matches 13.10 ±0.5.
- [ ] Verify state-dict keys parse with existing prefix-stripping in `train_contrastive.py:500-510`.
- [ ] Set `joint_training_state.json["stage0"]["verified"] = true`.

### Phase 2 — Code changes (each subphase ends with `bash scripts/validate_for_willi.sh` green)
- [ ] **2A** — Replace `img_proj` Identity/Linear with `Sequential(Linear, GELU, Linear)` (no bias).
- [ ] **2B** — Add `AttentionPooling` to `hybrid_lm.py`; add `pooling_strategy` to `HybridConfig`; default `attention` in `configs/model/hybrid_70m.yaml`.
- [ ] **2C** — Add `JointMultiTaskLightningModule` (KD + CLIP + SimCSE in one forward); wire `contrastive_mode=joint` dispatch in `train_contrastive.py`. Reuses BiomedCLIP loader, `_nt_xent_loss`, `freeze_text_encoder_steps`. Adds frozen PubMedBERT teacher + 2-layer `distill_proj`. 4 param groups: backbone (lr=1e-5), heads (lr=3e-4), ViT-unfrozen (lr=1e-6), no-decay.
- [ ] **2D** — Create `configs/dataset/mimic_cxr.yaml` (`hf_repo_id: itsanmolgupta/mimic-cxr-dataset`, max_length=256, bs=16) and `configs/distill/joint_mimic.yaml` (α=0.3, β=1.0, γ=0.1).
- [ ] **2E** — Extend `tests/test_willi_parity.py` with img_proj/pooling/joint-loss assertions; pytest green.
- [ ] **2F** — Create `scripts/train_joint_mimic.sh`. Final `validate_for_willi.sh` green; commit + push to `a100_70m_baseline`.

### Phase 3 — Local CPU smoke test (~10 min)
- [ ] `scripts/smoke_test_joint.py`: 5 steps on 16-pair Indiana subset → all 3 losses finite, decreasing; gradients flow into img_proj, pooler.q, distill_proj, backbone.

### Phase 4 — MIMIC-CXR data prep on willi (~30 min)
- [ ] Verify `itsanmolgupta/mimic-cxr-dataset` loads; record column names (for `findings_field`/`impression_field`).
- [ ] Pre-cache one full epoch.

### Phase 5 — Joint training on willi A100 40GB (~10–12h)
- [ ] `sbatch scripts/train_joint_mimic.sh` (max_steps=10000, val_check=500, effective batch 128 = bs16×accum8, freeze_text_encoder_steps=500, init from Stage 0 ckpt).
- [ ] Live monitor: step 500 `L_clip < 4.0` & cosine-hist not peaked at 1.0; step 2000 `R@10 > 0.207`; step 5000 `R@10 ≥ 0.30`; step 10000 `R@10 ≥ 0.40`.
- [ ] **Kill gate at step 3000**: `R@10 < 0.15` AND not rising → scancel.

### Phase 6 — Eval + decision gate (~1h)
- [ ] Run `evaluate_retrieval.py` on Indiana CXR test (held out → cross-dataset eval).
- [ ] Run `eval_stage1_suite.sh` on the joint ckpt for STS-B/BIOSSES regression check.
- [ ] **Decision**:
  - Pass: `Indiana i2t R@10 ≥ 0.40 AND STS-B ≥ 0.5` → done.
  - Partial: `R@10 ∈ [0.25, 0.40)` → Phase 7.
  - Fail: `R@10 < 0.25` OR text quality collapsed → debug loss weights, return to Phase 2.

### Phase 7 — FAISS hard-neg mining (CONDITIONAL — only if Phase 6 = Partial)
- [ ] Encode 30K MIMIC reports → FAISS index → mine top-50 hard negatives per anchor (different patient).
- [ ] Inject K=4 mined negatives per anchor into InfoNCE batch.
- [ ] Continue training 5000 steps from Phase 5 ckpt; re-eval. Gate `R@10 ≥ 0.40`.

## Resolved decisions

- Reuse Stage 0 ckpt (PPL=13.10); skip broken Stage 1 (joint training replaces it).
- Loss weights α=0.3 / β=1.0 / γ=0.1 (no early sweep).
- Keep SimCSE term unless OOM (anchors text geometry vs CLIP-only drift).
- Effective batch 128 (256 OOM'd previously).
- Phase 7 gated to `[0.25, 0.40)` only.

## Resumability contract

1. Read this file + `joint_training_state.json`.
2. Resume at `joint_training_state.json["current_phase"]`. Checkboxes here = ground truth.
3. After every state change update: matching checkbox + `last_updated` (ISO 8601) + 1-line `notes` entry.
4. If `joint_training_state.json` missing locally, regenerate from this file's checkboxes (gitignored on purpose).
5. Never re-run a checkpoint-producing phase without first reading the latest `output_willi_server/*.log` and logging a verdict.
