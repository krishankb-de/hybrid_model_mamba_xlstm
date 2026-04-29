# Stage 1 Recovery — Multi-Phase Plan

> **Sub-plan of `PHASE_PLAN.md` Phase 2.** Resumable across sessions via this file (committed) + `stage1_recovery_state.json` (gitignored, local-only) at repo root. Full plan-of-record: `/Users/krish/.claude/plans/refer-to-the-codebase-replicated-ripple.md`.

## Context

Stage 1 has failed twice. Parent PHASE_PLAN.md Phase 2 (SimCSE+KD with `bs=16+accum=4`, fixed τ=0.05, λ_max=0.7, ramp 1500) was never executed — instead commit `ffc294a` removed SimCSE and switched to pure PubMedBERT KD with `λ=1.0, warmup=0, ramp=0`. Run 1217 shows classic Goodhart collapse: `student_teacher_cos` rises 0.40→0.75 while `train/stsb_spearman` peaks at 0.416 (step 2099) then DROPS to 0.286 (step 4098). The proxy improves; the target degrades.

**Recovery:** restore SimCSE+KD hybrid (pre-`ffc294a`), apply Gemini's accepted improvements (2-layer `distill_proj`, λ warmup/ramp, λ_max=0.3, last-token pooling), keep the existing fixed τ=0.05 SimCSE NT-Xent path.

### Gemini's suggestions — verdict against actual code

| # | Suggestion | Verdict | Notes |
|---|---|---|---|
| A | Last-token non-pad pooling | **Partially correct → adopt as default** | Code already excludes pads (`hybrid_lm.py:390-396`); but mean is suboptimal for *causal* models — last non-pad sees full context |
| B | 2-layer MLP for `distill_proj` | **CORRECT** | `distill_proj` is single `nn.Linear(512, 768, bias=False)` (`lightning_module.py:501`); `projection_head` is already 2-layer |
| C | Distill warmup_steps=2000 | **CORRECT only with SimCSE restored** | pure KD has no other loss to warm up against |
| D | λ_max=0.1 | **Adopted 0.3 (user)** | current 1.0 caused Goodhart collapse; 0.3 splits Gemini's 0.1 and parent plan's 0.7 |
| E | τ=0.05 fixed | **Already implemented** | `lightning_module.py:412 fixed_scale=20.0` — just unused since SimCSE removed |

## Resolved decisions

1. **λ_max = 0.3**
2. **Pooling = `last_token`** (default; mean kept as fallback for Phase 4 ablation)
3. **Effective batch = 128** (`batch_size=16 × accumulate_grad_batches=8`)
4. **`proj_head_dropout` = 0.2** (raise to 0.3 only in Phase 4 if SimCSE collapse recurs)

## Critical files

| File | Purpose | Phase |
|---|---|---|
| `STAGE1_RECOVERY_PLAN.md` (this file, repo root) | sub-plan for session resume | 0 |
| `stage1_recovery_state.json` (repo root, gitignored) | resumable state | 0, every phase |
| `.gitignore` | append `stage1_recovery_state.json` | 0 |
| `PHASE_PLAN.md` | one-paragraph pointer to this file under Phase 2 | 0 |
| `hybrid_xmamba/training/lightning_module.py:456-565` | `DistillContrastiveLightningModule._simcse_step` — restore SimCSE+KD hybrid; upgrade `distill_proj` to 2-layer MLP; defaults `lambda_max=0.3, distill_warmup=2000, distill_ramp=2000` | 1 |
| `hybrid_xmamba/models/hybrid_lm.py:362-399` | `HybridTextEncoder.encode` — add `pooling_strategy` arg (`mean` \| `last_token` \| `weighted_mean`), default `last_token` | 1 |
| `hybrid_xmamba/models/configuration_hybrid.py` | add `pooling_strategy: str = "last_token"` to `HybridConfig` | 1 |
| `configs/distill/stage1_pubmedbert.yaml` | `lambda_max: 1.0 → 0.3`, `warmup_steps: 0 → 2000`, `ramp_steps: 0 → 2000`; rewrite header | 1 |
| `configs/model/hybrid_70m.yaml` | add `pooling_strategy: "last_token"` | 1 |
| `scripts/train_stage1_distill.sh` | `accumulate_grad_batches: 4 → 8` (effective bs=128); update header | 1 |
| `scripts/smoke_test_distill.py` | extend: assert `simcse_loss > 0`, `distill_loss > 0`, lambda(0)=0, lambda(warmup+ramp)=λ_max, last-token pooling shape | 2 |
| `tests/test_willi_parity.py` | assertions for new config keys + Python 3.9 compat | 1 |
| `scripts/validate_for_willi.sh` | run after every Phase 1/2 edit (CLAUDE.md mandate) | 1, 2 |

---

## Phase 0 — Bootstrap (one-shot, no code/config edits)

- [x] Create `STAGE1_RECOVERY_PLAN.md` at repo root (this file).
- [x] Create `stage1_recovery_state.json` at repo root.
- [x] Append `stage1_recovery_state.json` to `.gitignore`.
- [x] Add 1-paragraph pointer in `PHASE_PLAN.md` (under Phase 2) to this sub-plan.
- [x] Verify both files exist and JSON parses.

## Phase 1 — Code fixes (CPU-testable, no SLURM)

After every edit run `bash scripts/validate_for_willi.sh` (Python 3.9 parity gate) — do NOT advance until it exits 0.

- [ ] **Restore SimCSE+KD hybrid loss** in `lightning_module.py` `_simcse_step` (~514-565):
  - Re-introduce two-view dropout SimCSE: `z1 = model.encode(x); z2 = model.encode(x)` (dropout in projection head provides the augmentation)
  - InfoNCE via `_nt_xent_loss(z1, z2, ..., fixed_scale=20.0)` (τ=0.05) — code already exists at line 412
  - KD: `(1 - cos(distill_proj(z1), pubmedbert_cls)).mean()` — keep current
  - `total_loss = simcse_loss + lambda(step) * distill_loss`
  - Log: `train/simcse_loss`, `train/distill_loss`, `train/lambda`, `train/student_teacher_cos`, `val/simcse_loss`, `val/distill_loss`, `val/total_loss`
- [ ] **Upgrade `distill_proj`** in `DistillContrastiveLightningModule.__init__` (line 501):
  ```python
  self.distill_proj = nn.Sequential(
      nn.Linear(student_dim, student_dim, bias=False),
      nn.GELU(),
      nn.Linear(student_dim, teacher_dim, bias=False),
  )
  ```
- [ ] **Add `pooling_strategy` to `HybridTextEncoder.encode`** (`hybrid_lm.py:362-399`):
  - `"mean"` (current behavior — keep as fallback)
  - `"last_token"` (NEW default): for right-padded inputs, `idx = mask.sum(1) - 1`; `seq_repr = last_hidden[torch.arange(B), idx]`. If mask is None, take `last_hidden[:, -1, :]`.
  - `"weighted_mean"` (optional): position-weighted mean
  - Read default from `config.pooling_strategy`
- [ ] **Update `HybridConfig`** (`configuration_hybrid.py`): add `pooling_strategy: str = "last_token"`.
- [ ] **Update `configs/distill/stage1_pubmedbert.yaml`**:
  - `lambda_max: 1.0 → 0.3`
  - `warmup_steps: 0 → 2000`
  - `ramp_steps: 0 → 2000`
  - Rewrite header: SimCSE+KD restored; warmup lets SimCSE organize embedding space first.
- [ ] **Update `configs/model/hybrid_70m.yaml`**: add `pooling_strategy: "last_token"`.
- [ ] **Update `scripts/train_stage1_distill.sh`**: `accumulate_grad_batches: 4 → 8` (effective batch 128 = 127 in-batch negatives, adequate for SimCSE); ensure `contrastive_mode=simcse` retained; rewrite header comment.
- [ ] **Add tests** in `tests/test_willi_parity.py`:
  - `pooling_strategy="last_token"` returns correct shape and uses last non-pad index for right-padded inputs
  - `distill_proj` parameter count = `512*512 + 512*768` = 655,872 (was 393,216)
  - `_get_distill_lambda(step=0)=0`, `(2000)=0`, `(3000)=0.15`, `(4000)=0.3`, `(10000)=0.3`
- [ ] **Run** `bash scripts/validate_for_willi.sh` — green required before commit.
- [ ] Update this file's checkboxes + `stage1_recovery_state.json` (`current_phase: 1 → 2`, append note).

## Phase 2 — Smoke / sanity (CPU)

- [ ] Extend `scripts/smoke_test_distill.py`:
  - 4-sample batch on CPU; instantiate `DistillContrastiveLightningModule` with PubMedBERT teacher (or tiny mock with `hidden_size=768`)
  - At `global_step=0` → assert `simcse_loss > 0.5` (random init), `lambda=0`
  - At `global_step=4000` → assert `distill_loss > 0`, `lambda=0.3`, gradients flow into 2-layer `distill_proj` and into encoder
  - `validation_step` → no NaN, `val/total_loss` finite
- [ ] Re-run `bash scripts/validate_for_willi.sh` (must include the new smoke test).
- [ ] Update plan + state file.

## Phase 3 — Stage 1 SLURM re-run on willi

Goal: non-collapsed encoder; STS-B ≥ 0.45 at step 5k AND non-decreasing through step 20k.

- [ ] Verify clean checkpoint dir (or use `experiment_name=hybrid_70m_stage1_kd_pubmedbert_v2`) to avoid stale resume.
- [ ] `git push` Phase 1+2 commits to `a100_70m_baseline`; confirm GitHub Actions willi-parity green.
- [ ] On willi: `sbatch scripts/train_stage1_distill.sh`.
- [ ] **Live decision gates** (every 1000 steps, monitor `output_willi_server/stage1_kd_pubmedbert_<JOBID>.log`):
  - Step 1000 (pure-SimCSE warmup, λ=0): `train/stsb_spearman ≥ 0.20` AND `val/simcse_loss ∈ [0.5, 4.0]` (NOT collapsed to ~0.006)
  - Step 3000 (mid-ramp, λ≈0.15): STS-B ≥ 0.40
  - Step 5000 (full λ=0.3): STS-B ≥ 0.45
  - Step 10000: STS-B ≥ 0.55 AND non-decreasing vs step 5k
  - Step 20000: STS-B ≥ 0.60
- [ ] If any gate fails → kill job, advance to Phase 4.
- [ ] Update plan + state with run ID and last metrics.

## Phase 4 — Iterate (only if Phase 3 fails)

Pick ONE change per iteration. Record in `stage1_recovery_state.json["decisions"]`.

- [ ] **If `simcse_loss` collapses to ~0** (run 1159 symptom): increase `proj_head_dropout: 0.2 → 0.3`, increase effective batch via `accumulate_grad_batches: 8 → 16`.
- [ ] **If STS-B plateaus < 0.40 with full λ**: try `lambda_max: 0.3 → 0.5`; if still flat, try `0.7` (parent PHASE_PLAN.md value).
- [ ] **If STS-B regresses after λ ramp completes** (Goodhart): reduce `lambda_max: 0.3 → 0.1` and extend `ramp_steps: 2000 → 4000`.
- [ ] **If pooling A/B unclear**: re-run with `pooling_strategy: "mean"` for direct comparison (single ablation).
- [ ] After each iteration: update plan + state, re-submit, re-monitor Phase 3 gates.
- [ ] **Hard cap: 3 iterations.** If none pass, escalate next session — do NOT silently train indefinitely.

## Phase 5 — Eval suite + handoff

- [ ] On Phase 3 gate pass at step 20k: `sbatch scripts/eval_stage1_suite.sh`.
- [ ] Verify `results/stage1_metrics.md` populated with hybrid vs PubMedBERT comparison (BIOSSES, STS-B, MedSTS, BEIR-NFCorpus, BEIR-TREC-COVID, BioASQ, PubMed retrieval).
- [ ] Decision gate (matches PHASE_PLAN.md targets): BIOSSES Spearman ≥ 0.5, STS-B ≥ 0.6, PubMed R@10 ≥ 0.6.
- [ ] Update **parent** `PHASE_PLAN.md` Phase 2 + Phase 3 checkboxes.
- [ ] Mark `stage1_recovery_state.json["completed"]: true`; append final-metrics note.
- [ ] Unblock parent Phase 4 (Stage 2 MedicalCLIP).

---

## Resumability contract (READ FIRST in new sessions)

1. Read `PHASE_PLAN.md` + `STAGE1_RECOVERY_PLAN.md` + `stage1_recovery_state.json` from repo root.
2. Resume at `stage1_recovery_state.json["current_phase"]`. Checkboxes in this file are ground truth.
3. After every meaningful state change (edit committed, smoke pass, sbatch submitted, eval done) update:
   - The matching checkbox here (`[ ]` → `[x]`)
   - `stage1_recovery_state.json["last_updated"]` (ISO 8601)
   - Append one-line note to `["notes"]`: `"YYYY-MM-DD: <what changed>"`
4. If `stage1_recovery_state.json` is missing locally, regenerate from this file's checkbox state — gitignored on purpose.

## Verification

- After Phase 0: both files exist; `python -c "import json; json.load(open('stage1_recovery_state.json'))"` parses.
- After Phase 1: `validate_for_willi.sh` exits 0; new parity tests pass.
- After Phase 2: `python scripts/smoke_test_distill.py` passes; gradients verified through both branches.
- After Phase 3: live log shows `train/stsb_spearman` non-decreasing through gates; no walltime kill.
- After Phase 5: `results/stage1_metrics.md` shows BIOSSES Spearman ≥ 0.5; parent PHASE_PLAN.md Phase 2 boxes checked.

## Unresolved questions

(none — see Resolved decisions above)
