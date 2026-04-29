# Stage 1 Recovery — Multi-Phase Plan (REVISED v2 per supervisor)

> **Sub-plan of `PHASE_PLAN.md` Phase 2.** Resumable across sessions via this file (committed) + `stage1_recovery_state.json` (gitignored, local-only) at repo root. Full plan-of-record: `/Users/krish/.claude/plans/refer-to-the-codebase-replicated-ripple.md`.

## Context (REVISED 2026-04-29 after supervisor review)

Stage 1 has failed twice. Run 1217 (pure PubMedBERT KD, λ=1.0, no warmup, no SimCSE) is **currently running** at submission time. Original plan v1 (this file) called for restoring SimCSE+KD hybrid + last-token pooling + 2-layer `distill_proj`. **Supervisor reviewed and reverted three of four changes:**

| Change | Original v1 plan | Supervisor verdict | Resolution |
|---|---|---|---|
| Restore SimCSE+KD hybrid | Yes | **No** — 6 failed runs is sufficient evidence; backbone is too strong for SimCSE to add gradient signal | **Drop** |
| Switch pooling to `last_token` | Yes | **No** — mean pooling working in 1217; do not change variables during a working run | **Drop** |
| Diagnose 1217 as Goodhart collapse (STS-B 0.416→0.286) | Yes | **No** — n=2 STS-B points is premature; `val/distill_loss` 0.561→0.227 is steadily declining; initial dip is normal recovery | **Concede — diagnosis was overconfident** |
| Upgrade `distill_proj` to 2-layer MLP | Yes | **Yes** | **Keep** |

**Supervisor's revised path:** let 1217 finish to step 20k → next Stage 1 with `lr=3e-6` (down from `1e-5`) + 2-layer `distill_proj` → Stage 2 CXR.

**Three safeguards added to that path** (not in supervisor's original phrasing, justified below):

1. **Step-12k kill gate during 1217.** If at step 12k STS-B < 0.30 AND not rising vs step 8k → kill rather than burn 4h on an unsalvageable run.
2. **Eval-suite gate between Stage 1 finish and Stage 2 launch.** Stage 2 (CLIP on Indiana CXR) requires a usable text encoder. Without `BIOSSES Spearman ≥ 0.5 AND STS-B ≥ 0.6` (PHASE_PLAN.md targets), CLIP inherits a broken encoder and produces another wasted run.
3. **`lr=3e-6` scope.** Apply to backbone only; keep `distill_proj` (and `projection_head`) at `1e-5` so the new MLP head can converge fast while the backbone barely moves.

## Critical files (revised)

| File | Purpose | Phase |
|---|---|---|
| `STAGE1_RECOVERY_PLAN.md` (this file, repo root) | sub-plan for session resume | 0 |
| `stage1_recovery_state.json` (repo root, gitignored) | resumable state | 0, every phase |
| `.gitignore` | append `stage1_recovery_state.json` | 0 (BLOCKED — sandbox EPERM, manual append needed) |
| `PHASE_PLAN.md` | one-paragraph pointer to this file under Phase 2 | 0 (BLOCKED — sandbox EPERM, manual append needed) |
| `output_willi_server/stage1_kd_pubmedbert_<1217 JOBID>.log` | live monitor for kill-gate | 1 (read-only) |
| `scripts/eval_stage1_suite.sh` | eval gate before Stage 2 launch | 2 |
| `hybrid_xmamba/training/lightning_module.py:501` | upgrade `distill_proj` from `nn.Linear` to 2-layer MLP (Linear→GELU→Linear) | 3 |
| `configs/distill/stage1_pubmedbert.yaml` | keep `lambda_max=1.0`, `warmup=0`, `ramp=0` (unchanged from 1217) | 3 |
| `scripts/train_stage1_distill.sh` | add per-param-group LR override: backbone `3e-6`, head `1e-5` (mechanism TBD — see Phase 3) | 3 |
| `scripts/train_contrastive_stage2.sh` | verify chaining via sbatch dependency; address gaps (see "Stage 2 chaining" appendix) | 4 |

---

## Phase 0 — Bootstrap (one-shot, no code/config edits)

- [x] Create `STAGE1_RECOVERY_PLAN.md` at repo root.
- [x] Create `stage1_recovery_state.json` at repo root.
- [ ] Append `stage1_recovery_state.json` to `.gitignore`. **BLOCKED — sandbox EPERM. User to manually add line `stage1_recovery_state.json` after `phase_state.json` in `.gitignore`.**
- [ ] Add 1-paragraph pointer in `PHASE_PLAN.md` (under Phase 2) to this sub-plan. **BLOCKED — sandbox EPERM. User to manually add the paragraph quoted in v1 of this plan.**
- [x] Verify both new files exist and JSON parses.

## Phase 1 — Monitor 1217 to completion (kill-gate at step 12k)

Goal: do not interfere with 1217. Read logs only. Apply step-12k kill gate.

- [ ] Identify 1217 SLURM JOBID from `output_willi_server/` filename.
- [ ] Every ~1h: tail latest log, extract `train/stsb_spearman` and `val/distill_loss` at the most recent val cycle.
- [ ] **Kill gate at step 12k**: if `train/stsb_spearman < 0.30` AND not rising vs step 8k → `scancel <JOBID>` and skip directly to Phase 3.
- [ ] **Else**: let 1217 finish to step 20k.
- [ ] Update `stage1_recovery_state.json` with final 1217 metrics + final checkpoint path.

## Phase 2 — Stage 1 eval gate

Goal: numeric decision on whether 1217's checkpoint is acceptable for Stage 2.

- [ ] On 1217 finish (or kill at Phase 1): submit `sbatch scripts/eval_stage1_suite.sh STAGE1_CHECKPOINT=./outputs/hybrid_70m_stage1_kd_pubmedbert/checkpoints/last.ckpt`.
- [ ] Parse `results/stage1_metrics.md`.
- [ ] **Decision branch:**
  - **Pass** = BIOSSES Spearman ≥ 0.5 AND STS-B ≥ 0.6 → accept 1217, **skip Phase 3**, jump to Phase 4 (Stage 2 launch).
  - **Fail** = below either threshold → proceed to Phase 3 (next Stage 1 run with supervisor's lr-only changes).
- [ ] Record decision + metrics in `stage1_recovery_state.json["decisions"]`.

## Phase 3 — Next Stage 1 run with `lr=3e-6` + 2-layer `distill_proj` (only if Phase 2 fails)

Goal: minimal-variable change from 1217. Slow backbone LR; expand head capacity.

- [ ] **Code change** in `hybrid_xmamba/training/lightning_module.py` line 501: replace `self.distill_proj = nn.Linear(student_dim, teacher_dim, bias=False)` with:
  ```python
  self.distill_proj = nn.Sequential(
      nn.Linear(student_dim, student_dim, bias=False),
      nn.GELU(),
      nn.Linear(student_dim, teacher_dim, bias=False),
  )
  ```
- [ ] **Per-param-group LR.** Decide mechanism (Phase 3 sub-task):
  - Option A: extend `configure_optimizers()` in `HybridContrastiveLightningModule` to split params into `backbone` (lr=3e-6) and `head` groups (`distill_proj` + `projection_head` + `logit_scale`, lr=1e-5).
  - Option B: simpler — global lr=3e-6 (supervisor's literal proposal) and accept slower head convergence.
  - Default: **Option A** (more correct); fall back to Option B if `configure_optimizers` is too tangled.
- [ ] **Update `scripts/train_stage1_distill.sh`**: `model.learning_rate=3e-6` (or pass param-group config). Keep all other 1217 settings: `lambda_max=1.0, warmup=0, ramp=0, batch_size=16, accumulate_grad_batches=4, max_length=512`. Use `experiment_name=hybrid_70m_stage1_kd_pubmedbert_v3` to keep checkpoints separate.
- [ ] **Add unit test** in `tests/test_willi_parity.py`: assert 2-layer `distill_proj` parameter count = `512*512 + 512*768 = 655,872`.
- [ ] **Run** `bash scripts/validate_for_willi.sh` — green required.
- [ ] **Submit** `sbatch scripts/train_stage1_distill.sh`.
- [ ] **Live decision gates** (every 1000 steps):
  - Step 5000 (`val/distill_loss`): should be ≤ 0.30 (1217 hit 0.301 at step 2099 with lr=1e-5; 3.3x slower lr means 5k step ≈ 2k step on 1217 trajectory).
  - Step 12000: STS-B ≥ 0.40.
  - Step 20000: STS-B ≥ 0.55.
- [ ] On finish, return to **Phase 2** (eval-suite gate). If still failing, escalate to next session — do NOT recurse Phase 3 indefinitely.

## Phase 4 — Stage 2 CLIP on Indiana CXR

Triggered only after Phase 2 pass (BIOSSES ≥ 0.5 AND STS-B ≥ 0.6).

**Chaining recommendation:** submit Stage 2 with `--dependency=afterok:<stage1_jobid>` to start automatically when Stage 1 succeeds. Eval gate is being skipped per user direction — Stage 2 inherits whatever Stage 1 produces; if encoder is bad, will see it in image↔text R@10. With `afterok`, a failed/killed Stage 1 does NOT auto-launch Stage 2 (clean abort).

- [x] **Fix `cache_dir` user mismatch** — `configs/dataset/indiana_cxr.yaml:35` changed `/scratch/krishun/...` → `/scratch/bhushkri/...`.
- [x] **Add `freeze_text_encoder_steps` flag** — implemented in `HybridContrastiveLightningModule.__init__` + `on_train_batch_start` (`hybrid_xmamba/training/lightning_module.py`); plumbed through `scripts/train_contrastive.py`; defaulted to `500` in `scripts/train_contrastive_stage2.sh` via `+model.freeze_text_encoder_steps=500`.
- [ ] Eval gate before Stage 2: **SKIPPED per user direction.**
- [ ] Submit: `sbatch --dependency=afterok:<stage1_jobid> scripts/train_contrastive_stage2.sh`.
- [ ] Live monitor: `train/contrastive_loss` should drop from ~3.0 → < 1.5 by step 1000.
- [ ] On finish, run image↔text retrieval eval on Indiana-CXR test split.
- [ ] **Decision gate**: image↔text R@10 ≥ 0.4.
- [ ] Update parent `PHASE_PLAN.md` Phase 2 + Phase 3 + Phase 4 checkboxes.
- [ ] Mark `stage1_recovery_state.json["completed"]: true`.

---

## Stage 2 chaining appendix — script audit

User asked: can Stage 2 be queued via sbatch dependency to start the moment 1217 finishes, with no wasted time? **Yes — with the following caveats.**

### What works ✅

1. **Default checkpoint path resolves correctly.** `STAGE1_CHECKPOINT="./outputs/hybrid_70m_stage1_kd_pubmedbert/checkpoints/last.ckpt"` matches 1217's `experiment_name` and `output_dir`. Path will exist when Stage 1 finishes.
2. **Existence check on line 60-64** gracefully aborts if checkpoint missing.
3. **Compatible with `--dependency=afterok:<stage1_jobid>`.** SLURM holds Stage 2 until Stage 1 returns exit 0; on failure/timeout, Stage 2 never starts (no compute waste).
4. **Compatible with `#SBATCH --requeue` on Stage 1.** Requeued jobs keep the same JOBID, so dependency holds across requeue.
5. **Header settings sane**: `partition=mitarb`, `mem=40G`, `time=12:00:00`, `requeue` set.

### Resolved (this session)

1. ✅ **Text-encoder freeze for first N steps** — implemented. `HybridContrastiveLightningModule` now takes `freeze_text_encoder_steps: int = 0` (Stage 1 default unchanged) and `on_train_batch_start` unfreezes once `global_step >= freeze_text_encoder_steps`. Stage 2 sbatch script passes `+model.freeze_text_encoder_steps=500` via Hydra. Files touched: `hybrid_xmamba/training/lightning_module.py:252-294, 387-401`, `scripts/train_contrastive.py:444-473`, `scripts/train_contrastive_stage2.sh`.
2. ✅ **`cache_dir` user mismatch** — fixed. `configs/dataset/indiana_cxr.yaml:35` is now `/scratch/bhushkri/indiana_cxr_cache`.
3. **`bs=8` overrides yaml's `bs=64`.** Justified by BiomedCLIP image encoder + student + activations on 40GB. Effective batch via `accumulate_grad_batches=8` = 64. Kept; monitor.
4. **`STAGE1_CHECKPOINT=last.ckpt` not best.** With pure-KD `monitor=val/loss`, "best" tracks the proxy. For 1217 trajectory `last.ckpt` ≈ `best.ckpt`. Acceptable.

### Skipped per user direction

5. **No eval gate before Stage 2.** User accepted the risk that Stage 2 inherits whatever Stage 1 produces; quality regression will surface in image↔text R@10 metrics rather than wall-time savings.

### Verdict on user's question

> "if i want to put the process of `scripts/train_contrastive_stage2.sh` through the sbatch so that the stage 2 can be initiated from the checkpoint of stage 1 completion and time is not wasted. is it correct the script and will it run or cause any issues"

**The script will run** when chained via:
```bash
# Get current 1217 JOBID first:
squeue -u $USER  # find stage1 JOBID
# Then chain:
sbatch --dependency=afterok:<1217_jobid> scripts/train_contrastive_stage2.sh
```

**But three issues should be fixed before chaining**:
- (a) **`indiana_cxr.yaml` `cache_dir`** → change to `bhushkri` user (will fail with permission error otherwise on first run).
- (b) **No text-encoder freeze** → either add the flag, or accept higher Stage 2 collapse risk.
- (c) **No eval gate before Stage 2 starts** → if 1217 produces a bad encoder, you waste up to 12h on Stage 2 too. Either run eval manually first, or insert a quick in-script check.

If you accept the risks (a, b, c) for the sake of saving wall time, the chain command above will work as written. **My recommendation**: fix (a) at minimum (it's a 1-line yaml change), accept (b) for now (PHASE_PLAN.md target is image↔text R@10 ≥ 0.4 — if Stage 2 collapses, we'll see it in eval anyway), and instead of (c) eval-gate, set `--dependency=afterany` and add this 4-line guard at the top of the Stage 2 python invocation in the script:
```bash
python -c "
from scripts.evaluate_sts import quick_stsb
score = quick_stsb('${STAGE1_CHECKPOINT}')
print(f'Quick STS-B = {score:.3f}')
import sys; sys.exit(0 if score >= 0.5 else 1)
" || { echo 'Stage 1 STS-B < 0.5 — aborting Stage 2'; exit 0; }
```
(This requires implementing `quick_stsb` helper. Skip if implementation cost > eval-gate manual cost.)

---

## Resumability contract (READ FIRST in new sessions)

1. Read `PHASE_PLAN.md` + `STAGE1_RECOVERY_PLAN.md` + `stage1_recovery_state.json` from repo root.
2. Resume at `stage1_recovery_state.json["current_phase"]`. Checkboxes here are ground truth.
3. After every meaningful state change update: matching checkbox + `["last_updated"]` (ISO 8601) + append to `["notes"]`.
4. If `stage1_recovery_state.json` missing locally, regenerate from this file's checkbox state — gitignored on purpose.

## Verification

- After Phase 0: both files exist; JSON parses.
- After Phase 1: 1217 reached step 20k OR was killed at step 12k with rationale logged.
- After Phase 2: `results/stage1_metrics.md` populated; pass/fail decision recorded.
- After Phase 3 (if triggered): new Stage 1 run finished; loops back to Phase 2.
- After Phase 4: image↔text R@10 ≥ 0.4 on Indiana-CXR test split; parent PHASE_PLAN.md Phase 2/3/4 closed.

## Decision log

| Date | Decision | Rationale |
|---|---|---|
| 2026-04-29 | Drop SimCSE restore | Supervisor: 6 failed runs of SimCSE is sufficient counter-evidence |
| 2026-04-29 | Drop pooling change | Supervisor: don't change variables during working run |
| 2026-04-29 | Concede 1217 collapse diagnosis | n=2 STS-B points is premature; let it finish |
| 2026-04-29 | Keep 2-layer `distill_proj` | Supervisor agreed |
| 2026-04-29 | Add step-12k kill gate | Avoid 4h waste on unsalvageable run |
| 2026-04-29 | Add eval-suite gate before Stage 2 | Don't inherit broken encoder into CLIP |
| 2026-04-29 | Per-param-group LR (3e-6 backbone, 1e-5 head) | Slow backbone protects against Goodhart; head needs full LR to converge |
