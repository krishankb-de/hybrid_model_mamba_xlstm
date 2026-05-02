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
- [x] Confirm `outputs/hybrid_70m_stage0_kd_pubmed/checkpoints/stage0_model_only.pt` exists on willi.
- [x] Re-eval PPL: PubMed test PPL=12.42 (vs 13.10 baseline, better). Run on Lightning AI A100-80GB (one-off; ckpt exported from willi). Will not repeat — willi is system of record.
- [x] Verify state-dict keys parse: 117 keys, `embeddings.*` + `layers.N.mixer.*` + `final_norm.*`, no `lm.`/`_orig_mod.` prefix → strip logic no-ops cleanly.
- [x] Set `joint_training_state.json["stage0"]["verified"] = true`.

### Phase 2 — Code changes (each subphase ends with `bash scripts/validate_for_willi.sh` green)
- [x] **2A** — Replace `img_proj` Identity/Linear with `Sequential(Linear, GELU, Linear)` (no bias). `lightning_module.py:346-353`.
- [x] **2B** — Add `AttentionPooling` to `hybrid_lm.py`; add `pooling_strategy` to `HybridConfig`; default `attention` in `configs/model/hybrid_70m.yaml`. Baselines stay on `mean` as ablation controls.
- [x] **2C** — Add `JointMultiTaskLightningModule` (KD + CLIP + SimCSE in one forward); wire `contrastive_mode=joint` dispatch in `train_contrastive.py`. Added `MIMICJointDataset` + `load_mimic_cxr`. 4 param groups. 2-layer `distill_proj`. Validates green.
- [x] **2D** — Created `configs/dataset/mimic_cxr.yaml` (hf_repo_id: itsanmolgupta/mimic-cxr-dataset, max_length=256, bs=16) and `configs/distill/joint_mimic.yaml` (α=0.3, β=1.0, γ=0.1).
- [x] **2E** — Extended `tests/test_willi_parity.py` with 4 new tests: img_proj MLP, AttentionPooling, JointModule all-losses, joint_mimic config values. 42 passed.
- [x] **2F** — Created `scripts/train_joint_mimic.sh`. Final `validate_for_willi.sh` green. Committed + pushed.

### Phase 3 — Local CPU smoke test (~10 min)
- [x] `scripts/smoke_test_joint.py`: 5 steps on 16-pair Indiana subset → all 3 losses finite, decreasing; gradients flow into img_proj, pooler.q, distill_proj, backbone.

### Phase 4 — MIMIC-CXR data prep (folded into Phase 5 sbatch)
- [x] Verify `itsanmolgupta/mimic-cxr-dataset` loads; record column names (for `findings_field`/`impression_field`). Verified one-off on Lightning AI A100 (same HF source, same A100 profile): 30633 train rows, columns `['image','findings','impression']`, all images non-None. `findings_field=findings`, `impression_field=impression` match `configs/dataset/mimic_cxr.yaml` defaults.
- [x] Pre-cache one full epoch — **embedded as pre-flight inside `train_joint_mimic.sh`** so willi hits the same HF endpoint into `MIMIC_CACHE_DIR` (default `/scratch/bhushkri/mimic_cxr_cache`) before the trainer reads. Single sbatch covers both verify+precache and training. Set `SKIP_VERIFY=1` to skip on warm-cache resubmits.

### Phase 5 — Joint training on willi A100 40GB (~10–12h)
- [x] From parent dir: `cd /scratch/bhushkri/hybrid_xmamba_a100_70m_40 && sbatch hybrid_model_mamba_xlstm/scripts/train_joint_mimic.sh` (pre-flight verify+precache → max_steps=10000, val_check=500, effective batch 128 = bs16×accum8, freeze_text_encoder_steps=500, init from Stage 0 ckpt).
- [x] Live monitor: peak MIMIC val R@10=9.37% at step~1915, classic overfit onset after step~3010.
- [x] **Kill gate**: job killed past optimum (~step 6657). Best ckpt: `contrastive-step=001915-val/total_loss=1.9140.ckpt`.

### Phase 6 — Eval + decision gate (~1h)
- [ ] Run `eval_joint_indiana_cxr.sh` → Indiana/IU-Xray test (743) cross-dataset eval.
- [ ] Run `eval_joint_mimic_cxr_val.sh` → MIMIC val (3063) in-distribution sanity check.
- [ ] (Optional) Run `eval_stage1_suite.sh` on joint ckpt for STS-B regression check.
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
