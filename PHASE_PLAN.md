# Hybrid Mamba-xLSTM — Multi-Phase Recovery Plan

> **Resumable plan-of-record.** Future sessions: read this file + `phase_state.json` (gitignored, local-only) BEFORE running anything. Update checkboxes here and `phase_state.json["last_updated"]` after each meaningful action. See "Resumability Contract" at bottom.

## Context

Hybrid 70M Mamba-xLSTM (`[mamba, mamba, mlstm]`, dim=512, 8 layers, max_pos=1024) on willi A100 40GB. Pipeline:

```
Stage 0: LM-KD on PubMed (teacher BioMedLM 2.7B)
   ↓
Stage 1: SimCSE + KD-PubMedBERT contrastive training
   ↓
Stage 2: MedicalCLIP image-text alignment (Indiana / MIMIC-CXR / ROCO)
```

**Current state:** Stage 0 ACCEPTED (eval PPL=13.10). Stage 1 FAILED — collapsed (val_loss plateau at 0.006) and walltime-killed. This plan recovers Stage 1 with fixed hyperparameters and adds in-training biomedical eval, then unblocks Stage 2.

---

## Architecture Verdict

The architecture matches the proposal and is correct.

- ✅ `[mamba, mamba, mlstm]` interleave, dim=512, 8 layers, GPT-2 BPE 50257, max_pos=1024 (`configs/model/hybrid_70m.yaml`)
- ✅ Mean pooling with attention_mask (`HybridTextEncoder`) — no last-token-on-pad bug
- ✅ Projection head 512→512→512, GELU, dropout, learnable `logit_scale` init=2.6592 (CLIP)
- ✅ KD loss: `L = L_SimCSE + λ(step) · (1 − cos(student_pooled, teacher_cls))` (`configs/distill/stage1_pubmedbert.yaml`)
- ⚠ Pipeline / hyperparams have bugs (Phase 2) — not the model design.

---

## Diagnosis (from willi logs)

**Stage 0 (`output_willi_server/stage0_kd_resume_1082.log`)**
- Walltime-killed mid-validation; resumed from non-resumable dataloader.
- Final eval PPL=13.10 (target <25). User accepted.

**Stage 1 (`output_willi_server/stage1_kd_pubmedbert_1159.log`) — CRITICAL**
1. **Representation collapse** — `val/total_loss` plateau at ~0.006 (healthy ~0.2–0.5).
2. **λ_max=0.3 too low** — KD signal too weak to anchor against collapse.
3. **batch_size=8** — only 7 in-batch negatives; SimCSE needs more.
4. **No biomedical eval during training** — couldn't see collapse early.
5. **Walltime-killed** mid-epoch 10 (~43% of 20k steps).
6. **`logit_scale` clamp [1, 100]** can drift to 100 → drives loss→0 trivially.

**Verdict: Stage 1 checkpoint UNUSABLE. Re-run required (Phase 2).**

---

## Recovery Roadmap

Each checkbox is the unit of state. Mark `[x]` immediately when done. `phase_state.json["current_phase"]` advances when ALL boxes in a phase are checked.

### Phase 0 — Bootstrap (one-shot)

- [x] Create `PHASE_PLAN.md` at repo root (this file).
- [x] Create `phase_state.json` at repo root (local-only).
- [x] Add `phase_state.json` to `.gitignore`.
- [x] Update `CLAUDE.md` with one paragraph pointing future sessions at `PHASE_PLAN.md`.

### Phase 1 — Stage 0: ACCEPTED (no action)

- [x] Stage 0 eval reviewed (PPL=13.10 in `eval_stage0_lm_1135.log`).
- [x] Foundation for Stage 1: `outputs/hybrid_70m_stage0_kd_pubmed/checkpoints/stage0_model_only.pt`.

### Phase 2 — Fix Stage 1 collapse & re-run (HIGHEST PRIORITY)

**Active sub-plan: `STAGE1_RECOVERY_PLAN.md`** (committed at repo root) + `stage1_recovery_state.json` (gitignored, local-only). After commit `ffc294a` removed SimCSE in favor of pure PubMedBERT KD, run 1217 showed Goodhart collapse           (`student_teacher_cos` ↑ 0.40→0.75 while `train/stsb_spearman` ↓ 0.416→0.286). Recovery: restore SimCSE+KD hybrid, upgrade      
`distill_proj` to 2-layer MLP, λ_max=0.3 with warmup=2000+ramp=2000, last-token pooling, effective batch=128. New sessions: read
sub-plan first, resume at `stage1_recovery_state.json["current_phase"]`. 

**Goal:** non-collapsed biomedical text encoder. **Targets:** BIOSSES Spearman ≥ 0.5, STS-B ≥ 0.6, PubMed retrieval R@10 ≥ 0.6.

**Files to edit:**
- `configs/distill/stage1_pubmedbert.yaml` — `lambda_max`, `ramp_steps`
- `configs/model/hybrid_70m.yaml` — verify, no change expected
- `scripts/train_stage1_distill.sh` — Hydra overrides (bs, max_length, walltime, val_check_interval)
- `hybrid_xmamba/training/lightning_module.py:497–540` — `DistillContrastiveLightningModule` (fixed temperature)
- `hybrid_xmamba/models/hybrid_lm.py` — `HybridTextEncoder.logit_scale` init (or replace with fixed τ)
- NEW: `hybrid_xmamba/training/contrastive_eval_callback.py` — BIOSSES/STS/cosine-hist callback

**Checklist:**

- [x] **Fix temperature**: replace learnable `logit_scale` with fixed τ=0.05 (scale=20) in `_nt_xent_loss` for SimCSE path. Keep learnable `logit_scale` only for CLIP (Stage 2).
- [x] **Increase effective negatives**: `dataset.batch_size=16` (staged; OOM risk at 32), `accumulate_grad_batches=4` (effective 64), `dataset.max_length=512`.
- [x] **Strengthen KD**: `lambda_max: 0.3 → 0.7`; keep `warmup_steps=500`, extend `ramp_steps=500 → 1500`.
- [x] **LR sanity**: keep `1e-5`, warmup=1000, cosine decay.
- [x] **Add eval-during-training callbacks** (new file `hybrid_xmamba/training/contrastive_eval_callback.py`):
  - STS-B dev Spearman every 500 steps (logged as `train/stsb_spearman`)
  - Cosine-similarity mean/std + mean embedding-norm every 500 steps
  - Alignment & uniformity (Wang & Isola 2020) every 1000 steps
  - BIOSSES: skipped (bigbio loading scripts deprecated on willi's datasets version; STS-B covers quality signal)
- [x] **Walltime sanity**: Set SLURM `--time=12:00:00`.
- [x] **Reduce val cost**: `val_check_interval=1000` (was 500).
- [x] **Anomaly detection**: `AnomalyDetectionCallback(max_steps=200)` wired in `train_contrastive.py`.
- [x] **Sanity smoke**: `scripts/smoke_test_distill.py` — 6/6 checks pass (CPU, Python 3.9.23).
- [ ] **Submit**: `sbatch scripts/train_stage1_distill.sh`.
- [ ] **Live monitor**: cosine-hist must NOT peak at 1.0; STS-B must rise from ~0.0 → ≥0.5 within first 5k steps.
- [ ] **Decision gate**: STS-B Spearman ≥ 0.5 at step 10k AND ≥ 0.5 at end ⇒ accept; else iterate Phase 2 (lower bs / change τ / change λ).

### Phase 3 — Stage 1 evaluation suite (offline, post-train)

- [x] `scripts/evaluate_sts.py` — BIOSSES (multi-source fallback), STS-B, MedSTS; `--compare-pubmedbert` flag; writes `results/stage1_metrics.md`.
- [x] `scripts/evaluate_retrieval.py` — BEIR-NFCorpus, BEIR-TREC-COVID, BioASQ (BeIR HF datasets, nDCG@10); PubMed article→abstract R@k fallback; `--compare-pubmedbert` flag; appends to `results/stage1_metrics.md`.
- [x] `scripts/eval_stage1_suite.sh` — SLURM wrapper (mitarb, 4h) that runs both evals and generates `results/stage1_metrics.md`.
- [ ] Run `sbatch scripts/eval_stage1_suite.sh` after Stage 1 training completes.
- [ ] Verify `results/stage1_metrics.md` populated with hybrid vs PubMedBERT comparison.
- [ ] Update `phase_state.json` with metric snapshot.

### Phase 4 — Stage 2 MedicalCLIP

Blocked until Phase 2 BIOSSES gate passes.

- [ ] Freeze text encoder for first N=500 steps.
- [ ] Use BiomedCLIP image encoder; load Indiana / MIMIC-CXR / ROCO image-text pairs.
- [ ] Effective batch ≥ 128 (gradient accumulation as needed for A100 40GB).
- [ ] Eval: image→text and text→image R@1, R@5, R@10 on Indiana-CXR test split.
- [ ] Decision gate: R@10 ≥ 0.4 ⇒ accept; else iterate.

---

## Critical Files Index

| File | Purpose | Edit in Phase |
|---|---|---|
| `configs/model/hybrid_70m.yaml` | Architecture (frozen) | — |
| `configs/distill/stage1_pubmedbert.yaml` | KD weight, ramp | 2 |
| `configs/trainer/a100_single_gpu.yaml` | bf16, batch, grad accum | 2 |
| `scripts/train_stage1_distill.sh` | SLURM wrapper, Hydra overrides | 2 |
| `scripts/train_contrastive.py` | Stage 1 entry point | 2 (if eval cb wiring) |
| `hybrid_xmamba/training/lightning_module.py:497–540` | `DistillContrastiveLightningModule`, NT-Xent | 2 |
| `hybrid_xmamba/models/hybrid_lm.py` | `HybridTextEncoder`, logit_scale init | 2 |
| `tests/test_willi_parity.py` | willi 3.9 parity tests (extend per CLAUDE.md) | 2 |
| `output_willi_server/*.log` | Source-of-truth run history | read-only |

## Verification

- After **Phase 0**: `cat PHASE_PLAN.md && cat phase_state.json` — both exist and parse.
- After **Phase 2**: W&B run shows `val/biosses_spearman ≥ 0.5`, `val/cosine_hist` not peaked at 1.0, `val/embed_norm` stable, no walltime-kill in `slurm-*.log`.
- After **Phase 3**: `results/stage1_metrics.md` populated with vs-PubMedBERT comparison.
- After **Phase 4**: image↔text R@10 ≥ 0.4 on held-out Indiana-CXR.

## Resolved Decisions

- **Stage 0**: accepted; no re-train.
- **Stage 1 fix scope**: aggressive (fixed τ=0.05, bs=32, λ_max=0.7, ramp 1500, eval callbacks).
- **State file**: `phase_state.json` gitignored (local-only); `PHASE_PLAN.md` committed.
- **Max length**: 512 (matches PubMedBERT teacher max).

## Unresolved Questions

1. τ fixed=0.05 vs clamp learnable scale to [10, 20]? (Plan: fixed τ=0.05.)
2. W&B project name / API key configured on willi?
3. BIOSSES/STS-B dataset paths on willi (HF cache or local)?
4. Hard-negative mining now or after first non-collapsed run?

---

## Resumability Contract (READ THIS FIRST in new sessions)

A future session bootstraps as follows:

1. **Read** `PHASE_PLAN.md` (this file) + `phase_state.json` from repo root.
2. **Resume** at `phase_state.json["current_phase"]`. Checkboxes in this file are ground truth for what's done.
3. **Never** re-run a checkpoint-producing phase without first:
   - Reading the latest matching `output_willi_server/*.log`
   - Appending a verdict line to `phase_state.json["notes"]`
4. After every meaningful state change (run submitted, run finished, eval completed), **update**:
   - The relevant checkbox here (`[ ]` → `[x]`)
   - `phase_state.json["last_updated"]` (ISO 8601)
   - `phase_state.json["notes"]` (append one line: `"YYYY-MM-DD: <what changed>"`)
5. If `phase_state.json` is missing locally, **regenerate** from this file's checkbox state (gitignored on purpose; not a bug).
