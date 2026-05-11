# Hybrid Mamba-xLSTM Architectural Refactor — Plan-of-Record

> Supersedes `BIOMEDCLIP_KD_PLAN.md` + `biomedclip_kd_state.json`. Resumable. Read this file + `hybrid_arch_refactor_state.json` (gitignored) at session start.
>
> **Current phase: Phase 1 — bootstrap.** Approved plan archived at `/Users/krish/.claude/plans/refer-to-the-plan-mellow-peacock.md`.

## Experiment history

| Job | Phase | Key change | PubMed PPL | MIMIC i2t R@10 | Indiana i2t R@10 | Paired cos |
|---|---|---|---|---|---|---|
| — | — | (baseline carried from prior plan) Phase 6e job 1354/1362 — best of BIOMEDCLIP_KD path | 13.10 (Stage 0) | 8.23% | 4.04% | 0.258 |

## Context

The BiomedCLIP-KD pipeline (Phases 4–6f of `BIOMEDCLIP_KD_PLAN.md`) plateaued. Best result (Phase 6e, jobs 1354/1362): **MIMIC i2t R@10 = 8.23%, Indiana = 4.04%, paired cos = 0.258**. Across **9 distinct contrastive recipes** (PubMedBERT KD, BiomedCLIP-text KD, MoCo K ∈ {0, 256, 16384}, KD warmup on/off, img_proj on/off, α_kd schedules) MIMIC R@10 oscillated in the 8–10% band and never broke through. The JOINT_TRAINING_PLAN iterations before that hit the same ~9% ceiling. **The ceiling is structural, not in the contrastive recipe.**

`Research_docs/Hybrid Model Performance Bottleneck Analysis.pdf` identifies six LM/kernel/data-pipeline gaps that no prior plan has touched:

| Gap | Where in repo today |
|---|---|
| 1. Layer ratio + mLSTM placement | `[mamba,mamba,mlstm]` repeats over 8 layers → mLSTM at positions 2, 5 (`configs/model/hybrid_70m.yaml:11`) |
| 2. mLSTM exp-gate overflow | `i_gate = exponential_activation(ĩ_t)` with no LSE stabilizer / soft-cap / negative bias init (`hybrid_xmamba/layers/mlstm_block.py:63-117`) |
| 3. Pre-RMSNorm only, no HybridNorm | `norm1 → mixer → residual; norm2 → mlp → residual` (`hybrid_xmamba/layers/hybrid_block.py:121-132`) |
| 4. Doc-boundary contamination | `group_texts` packs concat'd tokens with no `cu_seqlens` (`scripts/train.py:170-181`); kernels have no boundary reset (`scan_triton.py`, `tfla_interface.py`) |
| 5. Cosine LR + linear warmup; static β2 | `lightning_module.py:419-435`, `:831-848` |
| 6. Cosine KD only; no frequency decoupling / SCCM | `_joint_step:976-978` |

User decisions: **full LM re-pretrain in scope**; **defer Hymba intra-layer parallel fusion**; **tiered target (floor/target/stretch)**. Phase 6e checkpoint is unusable after layer-pattern or normalization changes — Stage 0 must be redone.

## Decision

Stage architectural fixes **from least to most risky**, gate each on numerical sanity (no NaN, grad bounded) and short-run PPL signal before committing to a full A100 re-pretrain. The contrastive stage holds the Phase 6e recipe constant (`K=0`, KD warmup 1000 steps, α_kd 1.0→0.3, no `img_proj`) so any retrieval uplift is attributable to the new backbone, not a different contrastive recipe.

**Success bar (tiered):**
- **Floor** (mandatory; else abort): MIMIC i2t R@10 ≥ 8.23%, Indiana ≥ 4.04%, paired cos ≥ 0.258.
- **Target**: MIMIC ≥ 9.99% (Phase 5c parity), Indiana ≥ 5.5%.
- **Stretch**: MIMIC ≥ 12%, Indiana ≥ 6%.

## Critical files

| File | Phase | Action |
|---|---|---|
| `HYBRID_ARCH_REFACTOR_PLAN.md` (NEW, repo root) | 1 | Plan-of-record (this file) |
| `hybrid_arch_refactor_state.json` (NEW, repo root, gitignored) | 1 | Resumable state |
| `.gitignore` | 1 | Allowlist this plan; add state file to local-state block |
| `CLAUDE.md:5-7` | 1 | Repoint Session Bootstrap |
| `BIOMEDCLIP_KD_PLAN.md` | 1 | Deprecation banner |
| `scripts/diagnose_baseline.py` (NEW) | 2 | Probe: gate health, layer norms, doc-contam, alignment |
| `analysis/baseline_probe_report.md` (NEW) | 2 | Probe report (≤ 1 page) |
| `hybrid_xmamba/layers/mlstm_block.py:117-119` | 3 | tanh soft-cap + bias init −10 |
| `hybrid_xmamba/layers/mlstm_block.py:148-200` | 3 | LSE stabilizer state `m_t` in `_slow_forward` |
| `hybrid_xmamba/kernels/tfla/tfla_interface.py:74-150` | 3, 6 | Stabilizer pass-through + `cu_seqlens` boundary reset |
| `hybrid_xmamba/models/configuration_hybrid.py` | 3, 4, 5 | New config knobs: gate caps, norm topology, layer pattern |
| `hybrid_xmamba/layers/mlstm_block.py:103-110` | 4 | Q/K/V pre-norm |
| `hybrid_xmamba/layers/mamba_block.py:106-115` | 4 | Δ/B/C pre-norm |
| `hybrid_xmamba/layers/hybrid_block.py:38-105, 121-132, 177-188` | 4 | HybridNorm + `is_first_block` |
| `hybrid_xmamba/models/hybrid_lm.py:97-119` | 4, 6 | Thread `norm_topology` + `cu_seqlens` |
| `configs/model/hybrid_70m_v2.yaml` (NEW) | 5 | 1:3 ratio, mLSTM at middle |
| `configs/model/hybrid_70m_v3.yaml` (NEW) | 5 | 1:7 ratio, single mLSTM at midpoint |
| `hybrid_xmamba/kernels/selective_scan/scan_triton.py` (wrapper, not kernel) | 6 | Higher-level per-doc-segment wrapper around the Triton kernel |
| `scripts/train.py:170-181` | 6 | `group_texts`: emit `cu_seqlens` |
| `hybrid_xmamba/training/schedulers.py` (NEW) | 7 | `WSDScheduler` + β2 schedule |
| `hybrid_xmamba/training/lightning_module.py:153-205, 394-436, 793-848` | 7 | Scheduler dispatch in three modules |
| `configs/callbacks/learning_rate.yaml` | 7 | WSD variant |
| `scripts/smoke_arch_refactor.py` (NEW) | 8 | 100-step PubMed CPU smoke + 2K-step A100 sanity |
| `scripts/train_stage0_arch_v2.sh` (NEW) | 9 | Stage 0 re-pretrain wrapper |
| `hybrid_xmamba/training/lightning_module.py:976-978` | 10 | Freq-decoupled KD |
| `configs/distill/biomedclip_kd_joint_v2.yaml` (NEW) | 10 | v2 distill config |
| `scripts/train_biomedclip_kd_phase15.sh` (NEW) | 11 | Joint contrastive on new backbone |
| `scripts/eval_biomedclip_kd_phase15.sh` (NEW) | 11 | Eval wrapper |
| `tests/test_layers.py`, `test_kernels.py`, `test_willi_parity.py` | 3–7, 10 | New assertions per phase |

## Phases

### Phase 1 — Plan-of-record + state file (NO CODE) ⏳ AWAITING COMMIT
**Scope:** plan-file + doc edits only. No `.py` changes, no SLURM jobs.

- [x] **1A** — Write `HYBRID_ARCH_REFACTOR_PLAN.md` at repo root (this file).
- [x] **1B** — Write `hybrid_arch_refactor_state.json` at repo root.
- [x] **1C** — `.gitignore`: allowlist `HYBRID_ARCH_REFACTOR_PLAN.md` (line 89) + `!hybrid_arch_refactor_state.json` (line 94).
- [x] **1D** — `CLAUDE.md:5-7` Session Bootstrap: repointed to this plan + state. BIOMEDCLIP_KD reference kept as deprecated historical.
- [x] **1E** — `BIOMEDCLIP_KD_PLAN.md`: deprecation banner added (2026-05-11).
- [x] **1F** — `bash scripts/validate_for_willi.sh` green: 53 passed, 5 skipped, 6/6 gates green (matches baseline; no regression from doc-only changes).
- [ ] **1G** — Commit: `plan: hybrid architectural refactor (mLSTM stability, HybridNorm, layer-1:5, doc-resets, WSD, freq-KD)`.

### Phase 2 — Diagnostic probe on Phase 6e checkpoint (READ-ONLY)
Locate the ceiling before changing anything.

- [x] **2A** — `scripts/diagnose_baseline.py` (new). Run on Phase 6e best ckpt: mLSTM gate max/mean, layer-wise hidden-norm, doc-contamination probe (2-doc synthetic), alignment/uniformity on STS-B, cosine histogram on MIMIC-val.
- [x] **2B** — `analysis/baseline_probe_report.md` (≤ 1 page): findings + which PDF gaps the data supports / refutes. (STS-B + MIMIC probes pending Willi.)
- [x] **2C** — Re-weight downstream priorities based on probe data. Doc-boundary reset elevated to highest priority. Each gap still implemented.
- [x] **2D** — `validate_for_willi.sh` green (52 passed, 8/8 gates); commit.

### Phase 3 — mLSTM numerical stabilization (PDF gaps 2.1–2.2)
- [x] **3A** — `mlstm_block.py:117-119`: `tanh` soft-cap on raw `ĩ_t`, `f̃_t` pre-activations (cap=15.0, configurable). Init `i_gate_proj.bias = -10`, `f_gate_proj.bias = 0`.
- [x] **3B** — `mlstm_block.py:148-200` `_slow_forward`: LSE stabilizer state `m_t = max(log f_t + m_{t-1}, log i_t)`; rescaled gates `i'_t = exp(ĩ_t − m_t)`, `f'_t = exp(f̃_t + m_{t-1} − m_t)`.
- [x] **3C** — `tfla_interface.tfla_forward_parallel:74-150`: intra-chunk + inter-chunk `m`-state pass-through.
- [x] **3D** — `configuration_hybrid.py`: add `mlstm_gate_soft_cap`, `mlstm_input_gate_bias_init`, `mlstm_forget_gate_bias_init`.
- [x] **3E** — `tests/test_layers.py`: `test_mlstm_no_overflow_at_large_input`, `test_mlstm_input_gate_bias_init`, `test_mlstm_tanh_softcap_applied`.
- [x] **3F** — `tests/test_willi_parity.py`: `test_mlstm_stability_config_present`.
- [x] **3G** — `validate_for_willi.sh` green (57 passed, 6/6 gates); commit.

### Phase 4 — HybridNorm topology (PDF gap 3)
- [x] **4A** — `mlstm_block.py:103-110`: Q/K/V per-projection RMSNorm pre-mixer (extends existing `q_norm`/`k_norm` at line 68).
- [x] **4B** — `mamba_block.py:106-115`: Δ/B/C pre-norm before selective scan.
- [x] **4C** — `hybrid_block.py:38-105`: add `is_first_block` ctor arg.
- [x] **4D** — `hybrid_block.py:121-132` forward: FFN post-norm when `is_first_block=False`; pre-norm when `True`.
- [x] **4E** — `hybrid_block.py:177-188` `create_hybrid_blocks`: pass `is_first_block=(i==0)`; accept `norm_topology` kw.
- [x] **4F** — `hybrid_lm.py:97-119`: thread `norm_topology` to factory.
- [x] **4G** — `configuration_hybrid.py`: `norm_topology: str = "pre_rms"`.
- [x] **4H** — `tests/test_layers.py`: assert FFN normalizes post-residual for block ≥ 1.
- [x] **4I** — `validate_for_willi.sh` green; commit.

### Phase 5 — Layer-pattern restructure (PDF gap 1)
- [ ] **5A** — `configs/model/hybrid_70m_v2.yaml` (NEW): `[mamba, mamba, mamba, mlstm, mlstm, mamba, mamba, mamba]` (1:3 ratio, middle-placed).
- [ ] **5B** — `configs/model/hybrid_70m_v3.yaml` (NEW): `[mamba, mamba, mamba, mamba, mlstm, mamba, mamba, mamba]` (1:7 ratio, single midpoint).
- [ ] **5C** — `tests/test_willi_parity.py:145`: extend parametrize list with `hybrid_70m_v2`, `hybrid_70m_v3`.
- [ ] **5D** — Print `model.get_num_params()` for v2, v3. Adjust `mlp_ratio` only if v3 drops > 5% below 70M target.
- [ ] **5E** — `validate_for_willi.sh` green; commit.

### Phase 6 — Cross-document boundary resets (PDF gap 4)
Design choice: avoid Triton kernel surgery. Higher-level wrapper splits batch into per-document segments via `cu_seqlens` and concatenates. TFLA path is pure PyTorch — patch directly.

- [ ] **6A** — `scripts/train.py:170-181` `group_texts`: emit `cu_seqlens` (cumulative EOS positions). Threaded via dataloader collate.
- [ ] **6B** — `hybrid_lm.py:146-192` forward: accept `cu_seqlens: Optional[Tensor]`, thread through layers.
- [ ] **6C** — `hybrid_block.py:107-133` forward: accept `cu_seqlens`, pass to mixer.
- [ ] **6D** — `mamba_block.py:80-149` forward: per-segment loop wrapper if `cu_seqlens` provided.
- [ ] **6E** — `tfla_interface.tfla_forward_parallel:21-169`: reset `C_state=0`, `n_state=0`, `log_f=-inf` at boundary indices.
- [ ] **6F** — `tests/test_kernels.py`: `test_selective_scan_doc_boundary_reset`, `test_tfla_doc_boundary_reset` — 2-doc packed, assert doc-B output independent of doc-A perturbation.
- [ ] **6G** — `validate_for_willi.sh` green; commit.

### Phase 7 — WSD scheduler + β2 schedule (PDF gap 5)
- [ ] **7A** — `hybrid_xmamba/training/schedulers.py` (NEW): `WSDScheduler` (warmup 1%, stable 85%, decay 14% via `1-sqrt(p)`); β2 schedule 0.999→0.974 during decay.
- [ ] **7B** — `lightning_module.py:153-205, 394-436, 793-848`: dispatch on `self.scheduler_name` (cosine vs `wsd`). β2 dispatch on new `beta2_schedule: bool` ctor kwarg.
- [ ] **7C** — `configs/callbacks/learning_rate.yaml`: add `wsd` variant; default for new training configs.
- [ ] **7D** — `tests/test_willi_parity.py`: `test_wsd_scheduler_shape`.
- [ ] **7E** — `validate_for_willi.sh` green; commit.

### Phase 8 — CPU smoke + 2K-step PubMed sanity (parity gate)
All sanity on **PubMed** (WikiText dropped — does not pack, so doc-boundary code wouldn't fire; PubMed is the actual Stage 0 corpus).

- [ ] **8A** — `scripts/smoke_arch_refactor.py` (NEW): 100-step PubMed CPU run, `model=hybrid_70m_v2`, all Phase 3/4/6/7 active. Assert: loss decreasing, no NaN, grad-norm < 10, `i_gate` pre-cap max < 15, doc-boundary probe passes.
- [ ] **8B** — Submit 2× 1h willi A100 sanity (v1 vs v2 PubMed PPL at 2000 steps).
- [ ] **8C** — Decision gate: new-arch PPL ≤ baseline × 1.05. If > 5% regression: isolate (revert Phase 4 / Phase 6 individually).
- [ ] **8D** — `validate_for_willi.sh` green (gates 1–6, ≥ 53 passed); commit + push.

### Phase 9 — Stage 0 LM re-pretrain on PubMed (~12h A100)
- [ ] **9A** — `scripts/train_stage0_arch_v2.sh` (NEW): reuses `train_stage0_distill.sh` structure (PubMed, BioMedLM KD teacher); `model=hybrid_70m_v2` (or v3 if Phase 8 winner); WSD scheduler; max_steps=10000, val_check=500.
- [ ] **9B** — Submit on willi. Live monitor: WSD plateau visible; `train/mlstm_i_gate_max < 15`; no NaN.
- [ ] **9C** — Eval via `eval_stage0_lm.sh`: PubMed test PPL ≤ 13.76 (5% tolerance) OR ≥ 10% improvement; throughput ≤ 15% regression; BIOSSES sanity.
- [ ] **9D** — Decision gate: PASS → Phase 10. FAIL → isolation re-run (Phase 3 only; Phase 3+4 only).

### Phase 10 — Advanced contrastive head (PDF gap 6)
- [ ] **10A** — Verify `pooling_strategy: attention` active in `hybrid_70m_v2.yaml`; `AttentionPooling` (`hybrid_lm.py:313-349`) gets gradient.
- [ ] **10B** — `lightning_module.py:976-978` `_joint_step`: frequency-decoupled KD — `L_KD_low = MSE(low_band(z_text, t_emb))` (first 32 FFT bins, λ_low=1.0); `L_KD_high = α_high · MSE(high_band(...))` (α_high=0.1); blend `l_kd = λ_low·L_low + α_high·L_high + 0.5·(1-cos)`.
- [ ] **10C** — `configs/distill/biomedclip_kd_joint_v2.yaml` (NEW): clone of v1 + `freq_kd: true`, `freq_kd_low_bins: 32`, `freq_kd_alpha_high: 0.1`. Keep Phase 6e settings (K=0, freeze=1000, α_warmup=1.0, α_post=0.3).
- [ ] **10D** — `tests/test_willi_parity.py`: `test_freq_decoupled_kd_loss_finite`, `test_joint_module_v2_config`.
- [ ] **10E** — `validate_for_willi.sh` green; commit.

### Phase 11 — Joint contrastive re-run on new backbone (~12h A100)
- [ ] **11A** — `scripts/train_biomedclip_kd_phase15.sh` (NEW): init from Phase 9 Stage 0; `distill=biomedclip_kd_joint_v2`; `model=hybrid_70m_v2`; MIMIC-CXR data path same as Phase 6e.
- [ ] **11B** — Submit. Kill gates: `cos_text_teacher` → ≥ 0.85 by step 1000; val/clip_loss at step 1000 < 3.0; MIMIC R@10 by step 3000 ≥ 8.23%.
- [ ] **11C** — `scripts/eval_biomedclip_kd_phase15.sh` (NEW): MIMIC + Indiana + STS-B + BIOSSES on best ckpt.

### Phase 12 — Decision gate
- [ ] **12A** — Stretch hit (MIMIC ≥ 12% AND Indiana ≥ 6%) → Phase 13.
- [ ] **12B** — Target hit (MIMIC ≥ 9.99% AND Indiana ≥ 5.5%) → Phase 13.
- [ ] **12C** — Floor hit only → escalate: SCCM (5-prompt ensemble MSE), KDSP (z-score-filtered teacher KL), v3 swap, 4× longer run.
- [ ] **12D** — Floor missed → halt; per-fix isolation re-run.

### Phase 13 — Full eval + comparison
- [ ] **13A** — `evaluate_cxr_retrieval.py`: external Indiana + MIMIC-val authoritative.
- [ ] **13B** — `evaluate_sts.py`: BIOSSES + STS-B Spearman.
- [ ] **13C** — `evaluate_lm.py`: PubMed PPL on joint ckpt.
- [ ] **13D** — `evaluate_retrieval.py`: BEIR — only if stretch hit.
- [ ] **13E** — Comparison table (rows: Phase 5c, Phase 6e, Phase 11, any Phase 12).

### Phase 14 — Writeup + dissertation
- [ ] **14A** — Update plan + state JSON with final verdict, best ckpt path, per-fix attribution.
- [ ] **14B** — Dissertation table & ablation paragraph; honest reporting of failed paths.
- [ ] **14C** — Reaffirm deprecation banner on `BIOMEDCLIP_KD_PLAN.md`.

## A100 budget (60h total)

| Phase | Hours | Notes |
|---|---|---|
| 8 | 2 | v1 vs v2 PubMed PPL sanity |
| 9 | 12 | Stage 0 LM re-pretrain |
| 11 | 12 | Joint contrastive |
| 13 | 2 | External eval |
| **Mandatory** | **28h** | |
| 12 (cond.) | 12 | Escalation if floor-only |
| 13 ablation | 20 | 2 partial-fix Stage 0 reruns |
| **Total** | **60h** | |

Branch policy: **stretch** → skip 12, full ablation; **target** → 12h escalation + 1 ablation; **floor only** → all 32h on 12 + 1 isolation.

## Verification

Each phase gates on:
1. `bash scripts/validate_for_willi.sh` exits 0 (gates 1–6 all green; passed-count grows per phase from 53 baseline).
2. New phase-specific tests added and passing under Python 3.9.23.
3. Numerical: forward/backward finite on CPU smoke; grad-norm bounded; no NaN over 50+ steps.
4. SLURM monitor against pre-declared kill gates (Phase 9 PPL ≤ 13.76; Phase 11 cos_text_teacher ≥ 0.85 by step 1000, val/clip_loss < 3.0, R@10 ≥ 8.23% by step 3000).

End-to-end smoke before each willi sbatch (Phases 9, 11):

```
python scripts/smoke_arch_refactor.py --steps 100 --device cpu --model hybrid_70m_v2
bash scripts/validate_for_willi.sh
```

## Resumability contract

1. Read `HYBRID_ARCH_REFACTOR_PLAN.md` + `hybrid_arch_refactor_state.json` at session start.
2. Resume at `hybrid_arch_refactor_state.json["current_phase"]`. Checkboxes here = ground truth.
3. After every state change: tick checkbox + update `last_updated` (ISO 8601) + append a 1-line entry to `notes`.
4. If state JSON missing locally, regenerate from this file's checkbox state — gitignored on purpose.
5. Never re-run a checkpoint-producing phase (9, 11) without first reading `output_willi_server/*.log` and logging a verdict.

## Lessons baked in from prior plans (do not repeat)

- JOINT_TRAINING_PLAN ceiling at 9% (bs sweeps, FAISS, more steps) → structural; Phase 11 does not re-explore in-batch count.
- BIOMEDCLIP_KD 6a/6b/6c → KD vs CLIP gradient conflict → Phase 10/11 preserves Phase 6e's gated α_kd schedule.
- 6d/6f → MoCo queue post-KD-warmup is harmful → Phase 11 keeps `moco_queue_size: 0`.
- 5b → never seed a queue with random vectors → Phase 6 boundary reset uses zero (semantic blank), not random.
- JOINT Phase 2 applied PDF gaps 3/4 (img_proj MLP, attention pooling, logit_scale clamp). Gaps 1, 2, 5, 6 untouched — exactly this plan's targets.

## Open questions

- Phase 5 v2 vs v3: resolved by Phase 8 PPL measurement.
- WSD stable-phase length: default 85%; tune in Phase 7 if Phase 9 stalls.
- SCCM/KDSP: deferred to Phase 12 conditional, keeping Phase 11 attributable.
- Per-fix ablation in Phase 13: 2 partial reruns or skip — decide after Phase 11 verdict + budget.
- `f_gate_proj.bias` init = 0 (current) vs +3 (classic LSTM): keep 0 unless Phase 8 PPL regresses.

## Resolved decisions

- **A100 budget**: 60h total.
- **Stage 0 corpus**: PubMed only — matches BiomedCLIP/PubMedBERT/BioMedLM standard. Keeps eval clean; MIMIC report text stays Stage-2-only.
- **PPL regression tolerance**: ≤ +5% vs current Stage 0 (≤ 13.76 absolute, baseline 13.10).
- **No Hymba intra-layer parallel fusion** (user-deferred to follow-up plan).
- **Phase 8 corpus**: PubMed (not WikiText) — distribution match with Phase 9 + exercises Phase 6 doc-boundary reset code path.
