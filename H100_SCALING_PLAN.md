# H100 Plan-of-Record — Hybrid Mamba-xLSTM **CXR Report Generation**

> Resumable plan-of-record. Read this + `h100_scaling_state.json` (gitignored, allowlisted) at session start.
> **Builds on the COMPLETED `HYBRID_ARCH_REFACTOR_PLAN.md`** (broke the MIMIC ceiling 8.23%→10.45% i2t R@10). That plan is finished — historical reference only.
> Full approved plan: `/Users/krish/.claude/plans/i-want-to-implement-twinkling-ullman.md`.
>
> **Current phase: Phase 7 — PhysioNet credentialing (BLOCKING, on the critical path).**
> Phases 1/2/4/5/6/6B/6C/6D/6G are **COMPLETE and CLOSED**. Phase 3 deferred (its lever was measured non-binding).

---

## ⚠️ OBJECTIVE PIVOT — 2026-08-16

**The optimization target is now MEDICAL REPORT GENERATION, not retrieval.**

A generated radiology report scored against ground truth by **ROUGE-L** and **CheXbert F1** is what every decision from here optimizes. Retrieval is **not discarded** — it becomes a **supporting chapter** of the thesis (a complete, closed, well-controlled body of evidence about what does and does not move a CXR image-text joint space), and it remains the pretraining objective that produces the aligned image tower the generator is conditioned on.

**Why the pivot is well-founded on the evidence already in this file:**
- Retrieval is *finished as a research question here*: 10 clean nulls, 1 dominant lever (ViT adaptation depth), and that lever is exhausted (depth saturated at 12/12 blocks; `vit_lr` is an inverted-U with 1e-6 already optimal; scope is null). There is no remaining untested axis with evidence behind it.
- The two live constraints are both **data**, not method: Indiana is flat within noise on every variant (data-bound, 6G-4 + 2026-07-27 correction), and the image tower demonstrably **memorises** the 27,570-pair training set as soon as `vit_lr` rises (6G-1: `train/clip_loss` 1.17→0.04 while `val/clip_loss` 2.585→3.49).
- Report generation is a **stronger MSc contribution** for the same architecture: it exercises the causal Mamba/mLSTM decoder — the actual novel component — as a *generator*, which retrieval never did (retrieval only ever used it as an encoder).

**Retrieval's standing numbers are FINAL. Do not re-run retrieval arms to chase them.** Clean protocol: MIMIC i2t R@10 **10.81% → 14.59%** (+3.78pp, 6.2 SE). Protocol-matched: 11.07% → 17.14%.

---

## Context

Prior campaign is done on **A100 40GB** (willi/`mitarb`). Canonical model = `hybrid_70m_v2` + `freq_kd=false` + `vit_unfreeze=2` + `moco=0`. Final: **MIMIC i2t R@10 10.45%** (Target tier), **Indiana 3.90%** (intrinsic/data-bound), **Stage-0 PPL 15.62** (undertrained vs baseline 13.10).

User now has **H100 (94/141GB)** + optional 2-4 H100 node. Three A100-era ceilings are now liftable:
1. **Contrastive negatives capped at ~31** — CLIP loss is in-batch only (no `all_gather`, `moco=0`); H100 VRAM fits 128-256 true negatives (`lightning_module.py:512-543,1127-1158`). Biggest MIMIC lever; also cuts epochs on the 27.5k-pair set → less overfitting.
2. **Stage-0 undertrained** — 2.7B frozen teacher forced bs=8/40GB; curve still descending at 40K (needed ~117K). H100 fits bs=32-64 + teacher → finish it.
3. **70M cap** — 150M/350M configs exist but use the OLD `[m,m,mlstm]`+`pre_rms` (no v2 wins). H100 fits 150M v2 training.

Indiana gap is ablation-proven data-bound → only lever is diverse CXR data (user has access).

**Goal (superseded 2026-08-16, kept for the record):** H100-native infra + 150M-v2 backbone + scaled contrastive negatives + multi-source CXR data → push MIMIC to stretch (≥12%) and recover Indiana (≥floor), with clean per-lever attribution. **All three of those were delivered or measured null; see the closed phases below.**

**Goal (current):** Use the credentialed full MIMIC-CXR-JPG build to (a) remove the data ceiling that binds both the image tower and Indiana, and (b) train and evaluate an **image-conditioned report generator** on the official subject-disjoint split, scored by ROUGE-L and CheXbert F1.

## Success bar — PRIMARY (report generation) 🎯 NEW

Scored on the **official MIMIC-CXR-JPG test split** (subject-disjoint by construction), generated report vs ground-truth report.

| Tier | ROUGE-L | CheXbert F1 (micro, 14-label) | Rationale |
|---|---|---|---|
| **Floor** | ≥ 0.15 | ≥ 0.25 | beats a retrieval-nearest-neighbour baseline; proves the decoder conditions on the image at all |
| **Target** | ≥ 0.22 | ≥ 0.40 | competitive with R2Gen-class published CNN-LSTM/transformer baselines |
| **Stretch** | ≥ 0.26 | ≥ 0.50 | competitive with strong modern RRG systems |

Secondary/reported-alongside: BLEU-4, METEOR, and a **retrieval-NN baseline** (retrieve the nearest training report with the Phase-6G model and emit it verbatim) — the single most important control, because a strong retrieval system can score deceptively well on n-gram metrics without generating anything.

**Pre-registered before any number is produced:** the retrieval-NN baseline is run FIRST and its ROUGE-L/CheXbert F1 become the real floor. A generator that does not beat its own retrieval baseline has not contributed anything.

## Success bar — SUPPORTING (retrieval) ✅ CLOSED 2026-07-27
- **Floor**: MIMIC i2t R@10 ≥ 10.45%; Indiana i2t ≥ 4.04%; Stage-0 PPL ≤ 15.62.
- **Target**: MIMIC ≥ 12%; Indiana ≥ 5.5%; PPL ≤ 13.76.
- **Stretch**: MIMIC ≥ 14%; Indiana ≥ 7%; PPL ≤ 13.10.

| Metric | Best (`val == test` selection) | Best (clean protocol) | Tier reached |
|---|---|---|---|
| MIMIC i2t R@10 | **0.1714** (D1c, vit_unfreeze=12) | **0.1459** (6G-5) | **STRETCH** ✅ both |
| Indiana i2t R@10 | 0.0485 | 0.0390 | flat within noise (SE 0.76pp) |
| Stage-0 val PPL | **13.18** (Phase 5) | — | Target ✅, ≈stretch (13.10) |

The MIMIC headline is **8.23% → 10.45% (A100 refactor) → 17.14%** on the protocol every prior number in this project used (`val == test` checkpoint selection), or **14.59%** under a fully clean protocol where nothing about the test set touches training or selection. **Quote 14.59% as the thesis headline and 17.14% as the protocol-matched comparison**; the 2.55pp gap between them is itself a reportable measurement of what test-informed selection buys.

The single decisive intervention was **image-tower adaptation depth**; all ten text-side and objective-side levers were null, and the two remaining image-side axes (LR, scope) were already at or near their optimum. **Indiana never moved** on any lever — it is data-bound, which is what Phase 7 exists to address.

## DIAGNOSIS — "is it overfitting?" (read this before proposing a data or regularisation lever)

Three statements in this file look contradictory. They are all true and they are about **different things**. Get this right or the Phase-9 arms will be mis-read.

| Statement | Scope | Evidence |
|---|---|---|
| "overfitting is **not** binding" | **text tower + optimizer** | epochs 23→14 moved R@10 **+0.06pp** (0.1084→0.1090). 6B-3: lower `head_lr` **removed the late-epoch rollover entirely** (arm ends at its max, no decline) and the **plateau height did not move** (0.119 vs 0.122). |
| "the image tower **memorises**" | **ViT at elevated `vit_lr`** | 6G-1: `train/clip_loss` 1.17 → 0.702 → 0.113 → **0.037** as `vit_lr` 1e-6→3e-5, while `val/clip_loss` **rises** 2.585 → 3.487. Overfit onset moves earlier monotonically: best step 4750 → 3250 → 2500 → **1750**. |
| "the binding constraint is the **image representation**" | **the actual diagnosis** | 6C: every text-side and objective-side lever null; both positives ever recorded are image-side. |

**6B-3 is the decisive experiment: it eliminated the overfitting *symptom* and the ceiling did not move.** Whatever sets the ceiling, it is not overfitting.

**The real characterisation.** "Amount of image adaptation" = **depth × LR × scope**, and all three are now measured out: depth is **physically exhausted** (12/12 ViT-B/16 blocks), scope is **null** (0.1704 vs 0.1714), and LR is an **inverted-U with 1e-6 already optimal**. There is no dose left to give. The reason the optimum is pinned at 1e-6 — three orders of magnitude below `head_lr` — is **85.1M trainable image params against 27,570 images**.

> **The model sits at the overfitting knee. It is not broken by overfitting; it is DOSE-LIMITED by it.**
> There is mild, real overfitting at the canonical operating point (best step 4750/6000, mild late rise; the 6G-7 protocol cost of **2.55pp at vit=12 vs 0.26pp at vit=2** is precisely a measurement of how much). But no points are being *lost* to it — the loss is the points that turning the lever up **would have bought**.

**Consequence for Phase 9 — this changes what counts as success:**
- ❌ *"More data fixes overfitting"* → predicts the current number rises on its own.
- ✅ **"More data moves the knee right"** → predicts the `vit_lr` optimum **shifts right of 1e-6 and the peak is higher**. That is 9C's pre-registered prediction and it is the correct test.
- ⇒ **If 9B (data only, recipe unchanged) barely moves, that is NOT a failure** — it is the expected result under the correct mechanism. 9C is where the effect must appear.

**Honest caveat on Indiana:** it is flat within noise on *every* variant tested (0.0390–0.0485, SE 0.76pp). That is a **domain-diversity** problem, not overfitting — and full MIMIC is more of the *same* domain. Expect partial help at best (more patients/scanners/pathologies); do not bank the Indiana gate on it. That is what 9G and VinDr-CXR are for.

**After the pivot the constraint changes identity:** for report generation the blocker is neither overfitting nor data — **there is no image-conditioned decoder at all** (`hybrid_lm.py:147` `forward()` takes only `input_ids`). That is a capability gap, and it is Phase 10.

## Resolved decisions (from user, 2026-08-16) — the pivot
- **Objective**: medical **report generation**, scored by **ROUGE-L + CheXbert F1** against ground-truth reports. Retrieval → supporting chapter.
- **Data**: build the full MIMIC-CXR corpus **from PhysioNet**, not from the third-party HF mirror. **Submit PhysioNet credentialing** (CITI "Data or Specimens Only Research" + DUA) — this is Phase 7 and it **blocks Phases 8–11**.
- **Source project**: **MIMIC-CXR-JPG v2.1.0** (`377,110` JPGs, ~570 GB) — **NOT** MIMIC-CXR DICOM (4.7 TB). Reports come from **MIMIC-CXR v2.1.0** (`mimic-cxr-reports.zip`, ~135 MB). Both need the DUA signed separately under the same credentialing.
- **Storage strategy**: chunked download → downscale in flight → delete originals. **~310–400 GB of network transfer, ~6 GB kept on disk.** Peak disk ~4 GB regardless of corpus size. Fits the 200 GB HPC quota with room to spare.
- **Stored resolution**: **320 px square** (not 224) — costs ~3 GB more and preserves headroom for `RandomResizedCrop(224)`. Square resize is deliberate: it is bit-for-bit what `T.Resize((size,size))` already does, so Arm 0 stays a true reproduction control.
- **Selection**: **frontal only (PA/AP), one image per study.** The report is study-level; pairing laterals to the same text duplicates the text side and mixes two visual distributions.
- **Primary eval split**: the **official `mimic-cxr-2.0.0-split.csv.gz`** (subject-disjoint by construction). The legacy `train[90%:]` N=3063 gallery is retained only as a *continuity* number, and only if the leakage join can be verified (see 8D).

## Resolved decisions (from user, 2026-07-07) — retrieval era, still binding where noted
- **SLURM**: long training → `--partition=aisc-batch` (7-day cap); eval/smoke → `--partition=aisc-shortrun` (1-day); `--gres=gpu:h100:X` (X=1..8). 7-day cap ⇒ full Stage-0 in ONE block (no requeue juggling).
- **Model**: scale to **150M v2** (port v2 arch; fresh Stage-0).
- **Priority**: backbone + MIMIC (finish Stage-0 → push MIMIC to stretch).
- **Indiana data**: add diverse CXR; **IU-Xray EXCLUDED from training** (it IS the Indiana eval set — zero-leakage).
- **Per-source text**: free-text passthrough (MIMIC) + label-templated pseudo-reports (CheXpert/VinDr) + translated reports (PadChest).
- **150M `max_position_embeddings`**: 1024 (v2 parity; corpora are ≤512).
- **KD teacher**: BioMedLM 2.7B primary; larger-teacher fallback if PPL gate missed.
- **Python**: target py≥3.10 on H100 (verify stack; fall back to 3.9 if a dep breaks); keep 3.9-syntax hygiene (forward-compatible).
- **Multi-GPU (Phase 3)**: secondary — after single-H100 MIMIC win.

## Critical files

| File | Phase | Action |
|---|---|---|
| `H100_SCALING_PLAN.md`, `h100_scaling_state.json`, `.gitignore`, `CLAUDE.md` | 1 | plan/state/bootstrap |
| `configs/trainer/h100_single_gpu.yaml`, `h100_multi_ddp.yaml` (NEW) | 2 | H100 trainers |
| `scripts/train_stage0_h100.sh`, `train_biomedclip_kd_h100.sh`, `eval_h100.sh` (NEW) | 2 | SLURM templates (aisc-batch/shortrun) |
| `hybrid_xmamba/training/lightning_module.py:512-543,1041,1127-1158` | 3 | `all_gather` CLIP negatives |
| `configs/model/hybrid_150m_v2.yaml` (NEW) | 4 | 150M v2 arch |
| `scripts/train_stage0_150m_h100.sh` (NEW) | 5 | 150M Stage-0 |
| `scripts/train_biomedclip_kd_150m_h100.sh` (NEW) | 6 | batch-scaled contrastive |
| `configs/dataset/cxr_multi.yaml` (NEW), `scripts/train_contrastive.py:339-451` | 9 | multi-source CXR + text adapter |
| `analysis/h100_scaling_results.md` (NEW) | 12 | results |
| `tests/test_willi_parity.py`, `test_layers.py` | 2-4 | per-phase asserts |
| **`scripts/build_mimic_cxr_local.py` (NEW)** | **8** | **PhysioNet → local 320px parquet build (meta/manifest/fetch/pack)** |
| **`configs/dataset/cxr_mimic_full.yaml` (NEW)** | **8** | **local-parquet dataset config** |
| **`scripts/train_contrastive.py:437-465` (`load_mimic_cxr`)** | **8** | **add `local_parquet_dir` branch** |
| **`scripts/evaluate_cxr_retrieval.py:63-64,284-291,331-337,362`** | **8** | **local branch + `str`→`Image.open` (MISSING today, see 8E)** |
| **`hybrid_xmamba/models/hybrid_lm.py:147,230`** | **10** | **image conditioning: `inputs_embeds` / prefix on `forward` + `generate`** |
| **`scripts/evaluate_report_generation.py` (NEW)** | **11** | **ROUGE-L / BLEU / CheXbert F1 / retrieval-NN baseline** |

---

## Phases

> **PHASES 1–6G BELOW ARE THE SUPPORTING (RETRIEVAL) CHAPTER — COMPLETE AND CLOSED.**
> They are kept verbatim as the experimental record. **Do not re-open, re-run, or re-litigate them.**
> The active work starts at **Phase 7**. Jump there.

### Phase 1 — Plan-of-record + state (NO CODE) ✅ COMPLETE
- [x] **1A** — Write `H100_SCALING_PLAN.md` at repo root (this file).
- [x] **1B** — Write `h100_scaling_state.json` at repo root.
- [x] **1C** — `.gitignore`: allowlist `H100_SCALING_PLAN.md` + `!h100_scaling_state.json`.
- [x] **1D** — `CLAUDE.md` Session Bootstrap: repoint to this plan; `HYBRID_ARCH_REFACTOR_PLAN.md` → completed-historical.
- [x] **1E** — `bash scripts/validate_for_willi.sh` green: 69 passed, 5 skipped, 9/9 gates (doc-only; no regression).
- [x] **1F** — Commit on branch `h100_scaling`.

### Phase 2 — H100 infra enablement (single-GPU primary + 2-4 node file) ✅ LOCAL COMPLETE (2D/2G need H100 box)
Reuse: `torch.set_float32_matmul_precision('high')` (already in train scripts); auto-compile on sm_90 (`train.py:320-334` covers H100).
- [x] **2A** — `configs/trainer/h100_single_gpu.yaml`: `bf16-mixed`, `devices=1`, `compile_model=true`, `accumulate_grad_batches=1`.
- [x] **2B** — `configs/trainer/h100_multi_ddp.yaml` (2-4 node): `strategy=ddp`, `devices=-1`, `find_unused_parameters=true`, `accum=1`. Inert until Phase 3.
- [x] **2C** — SLURM templates `train_stage0_h100.sh` (aisc-batch 7-day, bs=64/accum=1, gc off, compile off for segmented path), `train_biomedclip_kd_h100.sh` (aisc-shortrun, **bs=128 = the in-batch-negative lever**, LR √-scaled), `eval_h100.sh` (ppl/retrieval modes). ENV-parametrized (`SCRATCH_ROOT`/`VENV_ACTIVATE`) — aisc scratch/env paths TBD.
- [ ] **2D** — (H100 box) Python env: full-stack import + 2-step train smoke on py≥3.10; fall back to 3.9 if a dep breaks.
- [x] **2E** — `requirements.txt`: flash-attn documented **opt-in** (NOT forced into base reqs — needs nvcc, breaks CPU/parity harness; ViT-only, marginal).
- [x] **2F** — `tests/test_willi_parity.py::test_h100_trainer_configs_resolve` (2 cases green).
- [ ] **2G** — (H100 box) Smoke `smoke_arch_refactor.py` on H100 (finite fwd/bwd, i_gate<cap, no NaN).
- [x] **2H** — `validate_for_willi.sh` green (9/9 gates; +2 h100 config tests); committed.

### Phase 3 — Distributed CLIP negatives (`all_gather`) — SECONDARY, deferred post-Phase-6
Only makes multi-GPU help retrieval. Single 141GB already gives 128-256 negatives → not on the critical path.
- [ ] **3A** — `lightning_module.py`: grad-preserving `_gather_across_gpus()` (autograd-aware `torch.distributed.nn.functional.all_gather`, OpenCLIP-style). Wire into `_nt_xent_loss` (`:512-543`) + `_joint_step` CLIP (`:1041`). Gate on `dist.is_initialized() and world_size>1` → identity single-GPU.
- [ ] **3B** — Test: world_size=1 numerical invariance; gather shape == world_size×B.
- [ ] **3C** — `validate_for_willi.sh` green; commit.

### Phase 4 — 150M v2 architecture config ✅ COMPLETE
`HybridConfig` supports all knobs; `create_hybrid_blocks` handles any pattern — config-only + tests.
- [x] **4A** — `configs/model/hybrid_150m_v2.yaml`: `dim=768`, `num_layers=12`, `head_dim=64`, `num_heads=12`, `norm_topology=hybrid`, `pooling_strategy=attention`, `max_position_embeddings=1024`, `learning_rate=4.0e-4` (√-width scaled; tunable→5e-4), `warmup_steps=2000`. mLSTM stabilization knobs via HybridConfig defaults (as 70m_v2 does).
- [x] **4B** — `layer_pattern`: `[m,m,m,m,L,L,L,m,m,m,m,m]` (3 mLSTM centered = 25% = v2 parity).
- [x] **4C** — `test_hybrid_150m_v2_config_and_param_count`: **183.72M actual** (nominal "150M"; untied 50k-vocab embeddings dominate — consistent with the 70M config → 83M convention). Tight band [181,186]M guards arch drift.
- [x] **4D** — `validate_for_willi.sh` green (72 passed, 9/9 gates); committed.

### Phase 5 — Stage-0 pretrain 150M v2 on H100 (backbone quality) ⏳ SCRIPT READY — run pending H100 box
- [x] **5A** — `scripts/train_stage0_150m_h100.sh` (wrapper over `train_stage0_h100.sh`; `model=hybrid_150m_v2`, bs=48/accum=1, `max_steps=120000` ~3B tokens, `warmup=2000`, WSD, aisc-batch 4-day).
- [ ] **5B** — (H100) Verify `train_stage0_distill.py` threads `norm_topology` (fixed 9F) + no hardcoded dim=512.
- [ ] **5C** — (H100) Submit. KD teacher BioMedLM 2.7B (primary); fallback larger teacher (BioGPT-Large/Meditron-7B/OpenBioLLM-8B) if PPL misses. Optional KD `alpha` {0.3,0.5} sweep.
- [ ] **5D** — (H100) Gate: `eval_h100.sh MODE=ppl` / `eval_stage0_lm.sh` (auto-detect; locked protocol). Target PPL ≤ 13.76.

### Phase 6 — Contrastive batch scaling (MIMIC-only, isolate lever) → stretch MIMIC ⏳ SCRIPT READY — run pending H100 box
Hold canonical recipe (`biomedclip_kd_joint_v2`: freq_kd=false, vit_unfreeze=2, moco=0). Change ONLY batch + backbone.
- [x] **6A** — `scripts/train_biomedclip_kd_150m_h100.sh` (wrapper; `model=hybrid_150m_v2`, `lm_checkpoint`=Phase-5, `batch_size` sweep {64,128,256} via env, `accum=1`, LR √-scaled for bs=128).
- [ ] **6B** — (H100) LR √-scale per batch: `backbone_lr→~2e-5`, `head_lr→~6e-4` at bs=128 (re-scale for other bs). Log embedding mean/std + cosine histogram (collapse watch).
- [ ] **6C** — (H100) Submit. Kill gates: `cos_text_teacher≥0.85` by 1k; `val/clip_loss<3.0`; MIMIC R@10 ≥ 10.45% by 3k.
- [x] **6D** — (H100) Gate: MIMIC i2t R@10 (`eval_h100.sh MODE=retrieval` / `evaluate_cxr_retrieval.py`). Target ≥12%. **RESULT: NULL.** bs=128/23ep 0.1084, bs=128/14ep 0.1090, bs=64/14ep **0.1113** (best). Floor 0.1045 cleared by all; target 0.12 missed by all; spread 0.29pp vs SE ~0.57pp ⇒ arms statistically indistinguishable.

### Phase 6B — LR-matched rerun (the one supported lever from the 2026-07-21 review) ⏳ SCRIPT READY
Phase-6 post-mortem found the batch sweep was **never LR-matched**: `backbone_lr`/`head_lr` were hardcoded at the bs=128 √-scaled values (`train_biomedclip_kd_h100.sh:90-91`), so the winning bs=64 arm silently trained at ~1.4x its proper LR. Combined with grad_norm ~12.3 against `gradient_clip_val=1.0` (~12x clipping every step), LR is the one untested knob with direct evidence behind it.
- [x] **6B-1** — `BACKBONE_LR`/`HEAD_LR` env-overridable in `train_biomedclip_kd_h100.sh`; 150M wrapper derives LR **and** `MAX_STEPS` from `BATCH_SIZE` (384000-sample / 13.93-epoch budget held across arms) so neither confound can recur. `EXPERIMENT` name now carries head LR so same-batch arms don't overwrite each other.
- [x] **6B-2** — Tests `test_h100_contrastive_lrs_are_overridable` + `test_h100_150m_contrastive_epoch_budget_is_batch_matched` (asserts bs×steps == 384000 and √-scaling off the bs=32 anchor). `validate_for_willi.sh` green 74 passed, 9/9 gates.
- [x] **6B-3** — (H100) Two bs=64 arms run (jobs 2359951 / 2359952). **RESULT: NULL.** √-matched `head_lr=4.24e-4` → in-training i2t R@10 peak 0.120 @ep11, final 0.116. Conservative `head_lr=3.0e-4` → peak/final 0.119. Prior best (head=6e-4) → peak 0.122, final 0.116. Spread 0.1–0.3pp vs SE ~0.57pp ⇒ **LR is not the binding constraint.** Secondary finding: lower LR removed the late-epoch retrieval rollover (arm B ends at its max, no overfit decline) without changing the plateau height — so overfitting is not binding either.
- [ ] **6B-4** — (H100) Authoritative `evaluate_cxr_retrieval.py` on both best-by-`val/total_loss` ckpts, for the record. Expected ~0.108–0.112 given in-training parity. Not a decision point.

---

## 2026-07-25 — Plateau post-mortem: what the seven nulls actually mean

**Lever tally: 7 clean nulls, 1 positive.**

| Lever | Δ i2t R@10 | Verdict |
|---|---|---|
| Stage-0 PPL 15.62 → 13.18 | flat | LM quality does not transfer to retrieval |
| 70M → 150M backbone | flat | capacity not binding |
| Negatives 32 → 128 | flat (bs=64 nominally ahead) | in-batch-negative thesis unsupported |
| Epochs 23 → 14 | +0.06pp | overfitting not binding |
| batch 128 vs 64 (epoch-matched) | 0.29pp spread | indistinguishable |
| head_lr 6e-4 → 4.24e-4 | flat | optimization not binding |
| head_lr 6e-4 → 3.0e-4 | flat | optimization not binding |
| **ViT unfreeze 0 → 2** | **+2.5pp** | **the only live lever — the only trainable thing not anchored to the teacher** |

### The `cos_text_teacher` = 0.57 claim is FALSIFIED — do not build on it

Two external reviews (2026-07-25) independently diagnosed the plateau as a **causal-vs-bidirectional representational ceiling**, evidenced by `cos_text_teacher` pinning at ~0.57 across every configuration. **Our own logs refute this:**

- Steps 0–1000 = KD-only warmup: CLIP is gated off (`lightning_module.py:1012-1016`), `α_kd_warmup=1.0`, and the **LM backbone is frozen** — only `projection_head` + `attn_pool` + `logit_scale` train (15.2M params). In that window `cos_text_teacher` reaches **0.874–0.892** (passes the ≥0.85 gate).
- It then *falls* to 0.60 → 0.57 **after** CLIP switches on at step 1000, and sits at 0.566 / 0.570 / 0.574 / 0.582 across a 2× LR range, both batch sizes, and both model scales.

A frozen causal SSM backbone with a small head hit **0.89** cosine against the bidirectional PubMedBERT teacher. 0.57 is therefore a **loss-weight equilibrium** between KD (α_post=0.3) and CLIP (β=1.0) pulling `z_text` toward targets that are themselves cos~0.5–0.7 apart in the joint space — exactly what the code comment at `:1086-1090` predicted. It is **not** an architecture ceiling, and its invariance to LR/batch/scale is the expected signature of an equilibrium, not of an optimization failure.

**Consequences carried into the phases below:**
1. `cos_text_teacher` must **never** be used as a gate on whether the text tower is architecturally adequate. Any experiment whose kill criterion is "did cos rise above 0.62" would draw a wrong conclusion.
2. **KD-anchor decay (6D-2) is promoted** — it attacks the equilibrium directly and is derived from our own data, not from literature.
3. Bidirectional encode (6E) is retained but **re-motivated**: the honest argument is report structure (the Impression at token ~300 recontextualizes the Findings at token ~40 and a causal encoder cannot propagate that backwards), *not* the cos number. This framing must survive into the writeup.

### Phase 6C — Measurement block (NO TRAINING) ✅ COMPLETE (job 2372055, 86 s wall)
Launch: `CKPT=<best 6B ckpt> sbatch scripts/run_phase6c_measurements.sh`. Ran on the head=3.0e-4 arm's best-by-`val/total_loss` ckpt (step 4750, 2.9465).

**Tower grid, N=3063, strict-index gt (authoritative protocol):**

| image tower | text tower | i2t R@1 | i2t R@5 | **i2t R@10** | t2i R@10 |
|---|---|---|---|---|---|
| stock ViT | BiomedCLIP text | 0.0039 | 0.0186 | **0.0340** | 0.0310 |
| stock ViT | student | 0.0023 | 0.0114 | **0.0232** | 0.0813 |
| fine-tuned ViT | BiomedCLIP text | 0.0065 | 0.0264 | **0.0431** | 0.0189 |
| **fine-tuned ViT** | **student** | **0.0180** | **0.0731** | **0.1172** | **0.1087** |

**Four hypotheses killed, one lever promoted:**

1. **Teacher parity — DEAD.** Stock BiomedCLIP scores **3.40%**; the student scores **11.72%**, i.e. **3.4×** the teacher. The 12% target is not above the anchor and Phase 6 is not a parity result. There is real headroom.
2. **KD anchor is a DRAG, not a ceiling.** BiomedCLIP's text tower is worth 4.31% even on the fine-tuned ViT, while the student is at 11.72%. So `alpha_kd_post=0.3` spends the whole post-warmup run pulling `z_text` toward a representation ~3× worse than the one CLIP is building. This is a stronger and more actionable version of what the external review guessed — **6D-2 is promoted to co-priority with 6D-1.**
3. **False negatives — DEAD (quantitatively).** At bs=64 only **19%** of batches contain a single false negative, mean **0.58 pairs out of a 4096-entry matrix**; at bs=128, 2.32 of 16384. Both external reviews ranked this a top-3 cause and called it "the direct explanation" for the flat negatives lever. It is not. **`MULTIPOS` dropped from the 6D-3 arm.**
4. **Metric artifact — DEAD.** Duplicates are 2.0% of the gallery (largest group 40 of 3063), oracle R@10 = **99.0%**, and dedup-aware R@10 differs from strict by 0.03pp. The templated-report ceiling both reviews warned about does not exist at this scale. Keep the strict metric as the headline.
5. **Text tower is not the weak half.** Swapping BiomedCLIP's text tower **in** costs 7.41pp (0.1172 → 0.0431). Per the pre-registered rule, **Phase 6E is deprioritised.**

**Caveat on 6C-2, state it in the writeup:** the fine-tuned ViT was co-trained with the student text tower, so that pairing is favoured by construction. The mitigating evidence is that fine-tuning still *helped* BiomedCLIP's own text tower (3.40 → 4.31), so the ViT did not drift into a space hostile to the teacher. The defensible claim is "substituting BiomedCLIP's text tower does not help", not "the student text tower is strictly better".

**Logical gap worth being precise about:** 6C-2 answers "is BiomedCLIP's text tower better than ours?" (no). It does **not** answer "would a bidirectional student beat a causal student?" — which is what 6E actually proposes. 6E is therefore *unsupported*, not *refuted*; it drops to an opportunistic cheap test after 6D, not a planned arm.

**Incidental observations for the writeup:**
- `stock ViT × student text` is strongly asymmetric (i2t 0.0232 vs t2i 0.0813) — hubness in the student text space relative to stock-ViT geometry. The co-trained pair is balanced (0.1172 / 0.1087). Evidence of genuine co-adaptation, not of a collapsed text space.
- The 6C load reported `1 missing key: ['logit_bias']` — expected and benign: the checkpoint predates the SigLIP parameter, it loads at its −10 init under `strict=False`, and `encode()` never reads it.
- Fine-tuning 2 ViT blocks lifts the *frozen-text* system 3.40 → 4.31 (+0.91pp), independently corroborating that the image side is where movement lives.

**Net: the binding constraint is the image representation.** Every text-side and optimization lever is null; the only two positives on record are both image-side (ViT unfreeze 0→2 = +2.5pp; ViT unfreeze with a frozen text tower = +0.91pp). Priority order in 6D reflects that.
- [x] **6C-1** — stock BiomedCLIP reference: **3.40%** i2t R@10.
- [x] **6C-2** — tower-swap 2×2 grid (table above).
- [x] **6C-3** — duplicate/false-negative audit: 2.2% train / 2.0% gallery duplicated, oracle R@10 99.0%, 0.58 false-neg pairs per bs=64 batch.
- [x] **6C-4** — R@1/R@5 surfaced; dedup-aware R@10 implemented and shown unnecessary (0.03pp).
Instrumentation and writeup evidence. **Per user decision 2026-07-25, 6D runs regardless of the 6C-1 result** — 6C does not gate 6D, it explains it and calibrates the writeup.
- [ ] **6C-1** — `scripts/reference_biomedclip_zeroshot.py`: stock BiomedCLIP (**its own text tower and image tower**) on the identical `train[90%:]` N=3063 protocol. Report i2t/t2i R@1/5/10 next to the 0.1113 student number. Published anchors put BiomedCLIP zero-shot at ~2–4% on comparable ~2.4k-study galleries, which would put the student at ~3× the teacher and imply real headroom — but the only number that counts is ours, on our protocol.
- [ ] **6C-2** — Tower-swap 2×2 in the same script: {student text, BiomedCLIP text} × {fine-tuned ViT, stock ViT}. Four numbers isolate which tower binds. If substituting BiomedCLIP's text tower barely moves R@10, text-side effort is misallocated and 6E should be dropped.
- [ ] **6C-3** — `scripts/audit_mimic_duplicates.py` (CPU only): exact + whitespace/case-normalised report-text grouping over `train[:90%]` and `train[90%:]`. Outputs (a) the oracle R@10 ceiling on the eval gallery under arbitrary tie-breaking, (b) expected false-negative rate per batch size {32,64,128}. Decides whether 6D-3's multi-positive mask is worth having and whether a dedup-aware R@10 belongs in the headline.
- [ ] **6C-4** — Reporting: surface i2t/t2i **R@1 and R@5** (already computed at `evaluate_cxr_retrieval.py:535-540`, just never carried into the state/writeup). R@1 ≈ 1.7% is far more sensitive to representation quality than R@10 and will show movement when R@10 does not. Add dedup-aware R@10 as a secondary metric.

### Phase 6D — Factorial lever block ✅ ARMS RUN (2026-07-26) — **ViT depth is the lever**

**In-training i2t R@10, N=3063, final @ep13 (peak in parens):**

| Arm | Lever | i2t R@10 | i2t R@1 | t2i R@10 | val/clip_loss | cos_teacher |
|---|---|---|---|---|---|---|
| D0 | control, vit=2 | 0.116 (0.120) | 0.017 | 0.113 | 2.897 | 0.570 |
| D1a | vit=4 | 0.132 (0.135) | 0.020 | 0.131 | 2.774 | 0.559 |
| D1b | vit=6 | 0.150 | 0.029 | 0.151 | 2.687 | 0.555 |
| **D1c** | **vit=12 (whole ViT-B/16)** | **0.168 (0.171)** | **0.030** | 0.165 | **2.585** | 0.544 |
| D2 | KD decay 0.3→0 over 2k | 0.118 (0.120) | 0.016 | 0.114 | 2.934 | **0.194** |
| D3 | SigLIP | 0.119 (0.120) | 0.016 | 0.120 | n/c | 0.546 |
| D5 | γ_simcse=0 | 0.122 | 0.016 | 0.113 | 2.872 | 0.586 |

**6D-1 is a decisive, monotone win.** 0.116 → 0.132 → 0.150 → 0.168 across unfreeze depth 2/4/6/12: **+5.2pp over control, ~9× the SE**. `val/clip_loss` falls monotonically alongside it (2.897 → 2.585) and R@1 nearly doubles, so this is generalization, not a tie-breaking or selection artifact — the eval images are never trained on. No OOM at bs=64 even with all 85.1M ViT params trainable. This corroborates the 6C conclusion exactly: **the image representation was the binding constraint all along.**

**6D-2 falsifies the 6C-derived KD prediction — record this honestly.** From 6C we argued the α_kd=0.3 anchor was dragging `z_text` toward a 4.31%-quality representation and that releasing it should help. The mechanism worked precisely as designed — `cos_text_teacher` collapsed 0.570 → 0.194, so the anchor genuinely released — and retrieval did **not** move (+0.2pp, well inside noise). `val/clip_loss` was marginally *worse* (2.934 vs 2.897), so at α=0.3 the KD term acts as a mild regulariser rather than a drag. **The KD-anchor hypothesis is now dead in both directions** (not a ceiling, not a drag) and should not be revisited.

**6D-3 (SigLIP, +0.3pp) and 6D-5 (γ_simcse=0, +0.6pp) are null.** Both under the 1.1pp bar. Combined with 6C-3 killing the false-negative premise, the entire objective-repair line of both external reviews is now empirically closed.

**Running tally: 10 nulls, 1 dominant lever.** Stage-0 PPL, model scale, negatives, epochs, batch, head_lr ×2, KD decay, SigLIP, SimCSE — all flat. ViT adaptation depth — monotone and large.
- [x] **6D-0** — control reproduces the Phase-6B recipe (0.116, matching the 6B arms).
- [x] **6D-1** — vit_unfreeze {4,6,12}: monotone, **the** result of this phase.
- [x] **6D-2** — KD decay: null; hypothesis retired.
- [x] **6D-3** — SigLIP: null.
- [x] **6D-5** — γ_simcse=0: null.
- [ ] **6D-4** — stack: now redundant. D2/D3/D5 are all null, so the "stack" is just D1c. Fold γ_simcse=0 into the 6G sweep as a free rider rather than running a separate arm.

### Phase 6G — ViT adaptation dose-response, continued ⏳ NEXT
Depth is exhausted at 12 (ViT-B/16 has 12 blocks), but "amount of image adaptation" = depth × LR × scope, and only depth has been swept. `vit_lr` has sat at **1e-6** the entire project — three orders of magnitude below the head LR — so the winning arm is one where the whole tower is unfrozen but barely moving.
- [x] **6G-1** — **`vit_lr` is an INVERTED-U; 1e-6 was already near-optimal.** In-training, N=3063:

  | vit_lr | R@10 final | R@10 peak | val/clip_loss | **train/clip_loss** | cos_teacher |
  |---|---|---|---|---|---|
  | 1e-6 (D1c ref) | 0.168 | 0.171 | **2.585** | 1.17 | 0.544 |
  | 3e-6 | 0.174 | **0.183** @ep9 | 2.721 | 0.702 | 0.540 |
  | 1e-5 | 0.149 | 0.168 | 3.412 | 0.113 | 0.685 |
  | 3e-5 | 0.145 | 0.163 | 3.487 | **0.037** | 0.803 |

  The mechanism is unambiguous overfitting: `train/clip_loss` collapses monotonically with vit_lr (1.17 → 0.70 → 0.11 → 0.04) while `val/clip_loss` rises (2.585 → 3.49). At 3e-5 the 85M-param image tower has effectively memorised the 27,570 training pairs. **Depth was the free lunch; LR is not.** Together with 6D this gives the full statement: image-tower adaptation has an *optimum* — too little (depth 2, 0.1107) and too much (lr 3e-5) both underperform.

  **Mechanistic corroboration for the 6D-2 null:** `cos_text_teacher` *rises* with vit_lr (0.544 → 0.685 → 0.803). At high image-tower LR the ViT moves toward the text encoder rather than the text encoder toward the images, leaving `z_text` nearer its KD anchor. The two towers trade off against each other — which is why releasing the KD anchor (6D-2) changed nothing on its own.
- [x] **6G-2** — **Scope is a NULL.** `all` (87.2M trainable) vs `blocks` (86.0M): R@10 0.170 vs 0.168, `val/clip_loss` 2.580 vs 2.585 — inside noise on both. `patch_embed`, `cls_token`, `pos_embed`, the final norm and the visual projection contribute nothing; the transformer blocks carry the entire adaptation. Keep `vit_unfreeze_scope: blocks` as canonical.
- [x] **6G-3** — **AUTHORITATIVE, FULLY LR-MATCHED DOSE-RESPONSE — STRETCH TIER CLEARED.**

  All four arms at `bs=64`, `head_lr=4.24e-4`, `backbone_lr=1.41e-5`, 6000 steps (13.93 epochs), best-by-`val/total_loss` checkpoint, MIMIC `train[90%:]` N=3063:

  | `vit_unfreeze` | trainable ViT | i2t R@10 | i2t R@1 | i2t R@5 | t2i R@10 | paired cos | tier |
  |---|---|---|---|---|---|---|---|
  | 2 (D0) | 14.2M | 0.1107 | 0.0153 | 0.0637 | 0.1041 | 0.3824 | floor |
  | 4 (D1a) | 28.4M | 0.1319 | 0.0196 | 0.0738 | 0.1166 | 0.3962 | target |
  | 6 (D1b) | 42.5M | 0.1430 | 0.0206 | 0.0937 | 0.1394 | 0.4084 | stretch |
  | **12 (D1c)** | **85.1M** | **0.1714** | **0.0300** | **0.1032** | **0.1538** | **0.4230** | **stretch** |

  **Monotone in every column** — i2t R@1/R@5/R@10, t2i R@10, and paired cosine all rise with depth. **+6.07pp** from depth 2→12, ~10.6× the 0.57pp SE. Floor 0.1045 ✅, target 0.12 ✅, **stretch 0.14 ✅**.

  **Control validated:** D0 (0.1107) reproduces the standing Phase-6 baseline (0.1113) to within 0.06pp, so the curve is a clean single-variable sweep rather than an artefact of the earlier LR/epoch confounds.

  **Per-block returns diminish but do not saturate:** 1.06 pp/block (2→4), 0.56 (4→6), 0.47 (6→12). Depth is exhausted at 12, so the remaining dose axes are `vit_lr` and scope — hence 6G-1/6G-2.

  Historical context: the earlier vit=2 numbers (0.1113 at `head_lr=6e-4`, 0.1172 at `3.0e-4`) are *not* part of this curve — they used different head LRs. Cite the LR-matched table above.

  Note the in-training/authoritative reconciliation *inverted* versus Phase 6: authoritative 0.1714 now slightly **exceeds** the in-training final (0.168). In Phase 6 the val-loss minimum (step 4500) sat well before the retrieval peak (~6000), costing ~1pp at selection time; with the stronger image tower both curves peak together at ~4750, so selecting on `val/total_loss` no longer costs anything.
- [x] **6G-4** — **Indiana: 0.0485 i2t R@10** (t2i 0.0700, R@1 0.0094, paired cos 0.2730, N=743). Floor 0.0404 ✅ (target 0.055 not reached). **The cross-domain risk did not materialise** — unfreezing all 12 ViT blocks on 27.5K in-domain MIMIC pairs improved Indiana too, from the A100 baseline 0.0390 to 0.0485 (+0.95pp). Deep image adaptation is not an in-domain/cross-domain trade here; it is a genuine representation improvement. Phase 7 gate cleared.
- [x] **6G-5 (motivation; result below)** — **Re-run D1c with `SELECTION_SPLIT=true`.** Arm-level comparison used `val/total_loss` on `train[90%:]`, which is the eval gallery — test-set selection at the arm level. The effect is ~10× SE so it is not noise-mining, but the thesis headline should be confirmed under a clean protocol.

  **6G-1 promoted this from formality to necessity.** The `vit_lr=3e-6` arm has **higher retrieval but worse val loss** than D1c (peak 0.183 vs 0.171; `val/total_loss` 2.858 vs 2.721) — retrieval and val-loss have diverged again, exactly the Phase-6 failure mode. Selecting on `val/total_loss` will not find that 0.183 peak, and selecting on retrieval against `train[90%:]` is selection-on-test. **The disjoint selection split is the only legitimate way to exploit a retrieval peak that the loss does not track**, so 6F should become the canonical protocol for any further tuning, and the best config re-run under it.
- [x] **6G-6** — **AUTHORITATIVE 6G RESULTS.** MIMIC `train[90%:]`, N=3063, best-by-`val/total_loss`:

  | Arm | vit_lr | scope | best step | val/total_loss | **i2t R@10** | i2t R@1 | paired cos |
  |---|---|---|---|---|---|---|---|
  | **D1c** | 1e-6 | blocks | 4750 | 2.6680 | **0.1714** | 0.0300 | 0.4230 |
  | scopeall | 1e-6 | all | 4500 | 2.6591 | 0.1704 | 0.0261 | 0.4244 |
  | lr3e6 | 3e-6 | blocks | 3250 | **2.6285** | 0.1632 | 0.0317 | 0.4422 |
  | lr1e5 | 1e-5 | blocks | 2500 | 2.6286 | 0.1606 | 0.0320 | 0.4589 |
  | lr3e5 | 3e-5 | blocks | 1750 | 2.6635 | 0.1456 | 0.0310 | 0.4692 |
  | **cleansplit** | 1e-6 | blocks | 3750 | 2.7438 | **0.1459** | 0.0242 | 0.4301 |

  1. **`vit_lr=1e-6` confirmed optimal.** The 0.183 in-training peak at 3e-6 was *unreachable under honest selection* — its best-by-val-loss checkpoint evals to 0.1632, below D1c. The 6G-1 divergence resolved against the higher LR.
  2. **Overfit onset moves earlier monotonically with LR:** best step 4750 → 3250 → 2500 → 1750 across 1e-6 → 3e-5. A clean measurement of the mechanism.
  3. **`val/total_loss` is a poor CROSS-ARM proxy.** All five arms span 2.6285–2.6680 (0.04 nats) while retrieval spans 2.6pp, and the *lowest*-loss arm (lr3e6) is not the best retriever. Within-arm checkpoint selection on val loss is sound; across-arm selection on it would pick the wrong arm. The depth sweep's agreement between loss and retrieval (2.897→2.585 vs 0.1107→0.1714) was fortunate, not guaranteed.
  4. **`paired cosine` rises with vit_lr (0.4230 → 0.4692) while retrieval falls.** Pulling matched pairs together is not the same as separating them from negatives. Paired cosine is logged as a health metric throughout this project and must not be read as a quality proxy.
  5. **Scope confirmed null** — 0.1704 vs 0.1714, 0.1pp apart.

- [x] **6G-5** — **CLEAN-SPLIT RESULT: MIMIC 0.1459, Indiana 0.0390.** Trained on `train[:85%]` (26,038 pairs), selected on `train[85%:90%]` (N=1532), evaluated on `train[90%:]` (N=3063) — nothing about the test set touched training or selection.

  **MIMIC 0.1459 still clears STRETCH (0.14).** The 2.55pp gap to 0.1714 is **3.76 SE** — a real effect. It is the honest cost of test-informed checkpoint selection, confounded with 5.6% less training data; the two cannot be decomposed without 6G-7.

  **Report both, with the protocol stated.** 0.1714 is comparable to every prior number in this project (all used `val == test`); **0.1459 is the defensible number under a clean protocol** and should be the thesis headline.

**CORRECTION (2026-07-27) — Indiana is FLAT, not improved.** The 6G-4 entry above reported D1c's Indiana 0.0485 as "+0.95pp over the A100 baseline, cross-domain risk did not materialise." That over-read the number. At N=743, p≈0.045, **SE = 0.76pp**, so 0.0485 vs 0.0390 is **1.25 SE — not a difference**, and the clean-split run lands at exactly 0.0390.

  Defensible statement: **Indiana is unchanged within noise across every variant tested** (0.0390–0.0485). Deep ViT adaptation neither helped nor hurt cross-domain — no in-domain/cross-domain trade, but no gain either. Indiana remains as data-bound as the original ablations concluded, which *raises* the priority of Phase 7.

- [x] **6G-7** — **CLEAN-PROTOCOL DEPTH-2 ENDPOINT: 0.1081** (ckpt step 3750, MIMIC N=3063). Completes the 2×2:

  | | vit=2 | vit=12 | **depth effect** |
  |---|---|---|---|
  | `val == test` selection | 0.1107 (D0) | 0.1714 (D1c) | **+6.07pp** (9.7 SE) |
  | clean protocol | 0.1081 (6G-7) | 0.1459 (6G-5) | **+3.78pp** (6.2 SE) |
  | **protocol cost** | **0.26pp** (0.5 SE) | **2.55pp** (3.8 SE) | |

  **1. The depth effect is real but ~40% smaller than the original protocol implied** — 3.78pp, not 6.07pp. Still 6.2 SE, so it holds comfortably, but 6.07pp was inflated and must not be quoted as the effect size.

  **2. Test-informed selection is NOT a constant offset — its value scales with overfitting.** At vit=2 the protocol costs nothing measurable (0.26pp, 0.5 SE); at vit=12 it costs 2.55pp (3.8 SE). Mechanistically clear: with 14.2M trainable image params the val-loss curve is flat and checkpoint choice is near-arbitrary; with 85.1M the model overfits, the curve acquires real structure, and choosing the checkpoint with the test set is worth something.

  **3. Decomposition of the 2.55pp.** Both clean-split runs trained on 5.6% less data. At vit=2 the *combined* cost of less data + honest selection was only 0.26pp, so the data component is ≲0.3pp at both depths (assuming it does not scale with depth, which is reasonable but unproven). **≈2.2 of the 2.55pp at vit=12 is test-informed selection, not lost data.**

  Methodological finding worth stating in the writeup: *the advantage a model gains from selecting checkpoints on its evaluation split grows with how much that model overfits* — here, from nil at 14M trainable image params to 2.2pp at 85M. It is rarely measured.

**FINAL NUMBERS FOR THE WRITEUP.** Quote the clean protocol as primary:
- **MIMIC i2t R@10: 10.81% (vit=2) → 14.59% (vit=12), +3.78pp, 6.2 SE** — clean protocol, stretch tier cleared.
- Protocol-matched to all prior project numbers (`val == test`): 11.07% → 17.14%.
- Indiana: flat within noise throughout (0.0390–0.0485, SE 0.76pp).
Six one-at-a-time nulls have made single-lever probing expensive per bit of information. Run D1–D3 in parallel for attribution **and** D4 stacked for the number. ~3.5 h/arm on one H100. Gate: **>1.1pp over control** (SE ~0.57pp at p≈0.11, n=3063) or it is noise.
Launch: `./scripts/submit_phase6d_arms.sh` (dry run) → `--submit`, or paste its sbatch lines directly. Every lever is env-overridable in `train_biomedclip_kd_h100.sh` and **defaults to the Phase-6B recipe**, so an unmodified invocation *is* 6D-0.

> **Cluster constraint (2026-07-26):** the aisc login node refuses `bash <script>` ("This command is not allowed on the login node!"). Use `./scripts/...`, an `srun --pty bash` session, or paste the `sbatch` lines — `sbatch` itself is allowed on the login node, which is how Phase 6C was submitted.
- [ ] **6D-0** — Control: bs=64, LR-matched (`head_lr=4.24e-4`, `backbone_lr=1.41e-5`), 6000 steps, canonical recipe. Baseline for this block.
- [ ] **6D-1** — `vit_unfreeze_blocks` ∈ {4, 6, 12}. The only lever with a measured positive (+2.5pp at 0→2). Config-only — `_get_vit_blocks()` (`:441-457`) is already generalised to any depth and `configure_optimizers` (`:899-909`) already builds the 4th param group. Watch `vit_lr=1e-6` — consider layer-wise decay only if 12 destabilises.
- [ ] **6D-2** — **CO-PRIORITY after 6C.** KD-anchor decay: linear `alpha_kd_post → alpha_kd_floor` over `kd_decay_steps` post-unfreeze (default floor 0.0, decay 2000). 6C showed the anchor is not holding the student at parity — it is pulling toward a 4.31% representation while CLIP builds an 11.72% one. Watch `pos_cosine_mean` and `val/clip_loss` for space collapse; arm **D2b** is the `alpha_kd_floor=0.05` fallback.
- [ ] **6D-3** — **SigLIP only** — `MULTIPOS` dropped after 6C-3 measured 0.58 false-negative pairs per bs=64 batch (19% of batches contain even one). The multi-positive mask has nothing to fix on this dataset. SigLIP survives on its own rationale (pairwise sigmoid, no global softmax, decoupled from batch size) but with **downgraded expectations** — its headline justification in both external reviews was the false-negative problem that does not exist here. The mask stays implemented and `false_neg_rate` stays logged; both are re-usable if Phase 7 multi-source data changes the duplication profile.
- [ ] **6D-4** — Stack: best-of-6D-1 + 6D-2 + 6D-3. The shot at the 12% target.
- [ ] **6D-5** — Optional cheap ablation: `gamma_simcse=0`. SimCSE pulls the same projection head toward uniformity using two dropout views of one text, competing with CLIP. May be free gain.

### Phase 6E — Bidirectional text encode ⏸ DEPRIORITISED by 6C-2 (code shipped, unscheduled)
The pre-registered gate fired against it: swapping BiomedCLIP's text tower in **costs 7.41pp** (0.1172 → 0.0431), so the student text tower is not the weak half and text-side capacity is not where the plateau lives. Code is implemented, tested and inert behind `BIDIRECTIONAL=false` — run it opportunistically after 6D if image-side levers stall, not as a scheduled arm.
Be precise about what was and was not shown: 6C-2 answers "is BiomedCLIP's text tower better than ours?" (no). It does **not** answer "would a bidirectional student beat a causal student?" — 6E is *unsupported*, not *refuted*. Motivated by report structure, **never** by `cos_text_teacher` (falsified above).
- [ ] **6E-1** — `bidirectional` flag on `HybridTextEncoder.encode`: forward pass + pass over the length-aware reversed sequence (right padding preserved), reverse-pass states gathered back to original positions, averaged before pooling. Costs 2× text-encode FLOPs, trivial next to the ViT. **Checkpoint-compatible — no new parameters, so existing ckpts and `evaluate_cxr_retrieval.py` keep working.**
- [ ] **6E-2** — If 6E-1 wins: the in-layer version (bidirectional scan inside each Mamba/mLSTM block, concatenate directions, project back to `dim`). That is the publishable contribution; the cheap version exists to test the hypothesis before committing to it.

### Phase 6F — Eval-protocol fix (do regardless) ⏳ NOT STARTED
- [ ] **6F-1** — Carve a **disjoint selection split** out of `train[:90%]` (e.g. `train[:85%]` train / `train[85%:90%]` select / `train[90%:]` test). Today `validation_split == test_split == train[90%:]`, so any checkpoint selection is selection-on-test. Currently mitigated by selecting on `val/total_loss` rather than retrieval, but a reviewer will still flag the shared split. Fixing it also legitimises checkpoint-on-retrieval, which is otherwise permanently banned.

**REJECTED — do NOT re-litigate** (2026-07-21 recipe review + 2026-07-25 review of two external plateau analyses):
- `vit_unfreeze_blocks: 0` — already run (jobs 1942/1948/1949): MIMIC **10.45% → 7.97%**, Indiana identical 3.90%. Freezing loses in-domain and recovers nothing cross-domain.
- `freq_kd: true` — already run (jobs 1922/1923 vs 1930/1931): Indiana **3.90% → 2.96%**. Cross-domain regression; attacks the Phase-7 gate.
- Checkpoint/early-stop on `val/retrieval_i2t_R@10` — **selection-on-test** while `validation_split == test_split == train[90%:]`. Unblocked only by 6F-1.
- "The 0.1113 is a *last*-checkpoint artifact; best-val recovers ~0.9pp for free" — **factually wrong.** 0.1113 was measured on `contrastive-step=002750-val/total_loss=3.6083.ckpt`, which *is* the best-by-`val/total_loss` checkpoint (`train_contrastive.py:801-806`, `monitor=val/total_loss`, `mode=min`). The 0.120–0.122 in-training peak is on the same 3063 pairs the eval uses — chasing it is the banned selection-on-test.
- **XBM / cross-batch memory queue** — this is the MoCo queue, already ablated in-repo and found harmful post-KD-warmup; `moco_queue_size=0` is canonical and is in "lessons carried". Re-proposing it with a different name does not make it new evidence.
- **Swapping the image backbone to RAD-DINO / MedSigLIP** — breaks `assert img_out == model.embed_dim` (`:419-422`) and removes the BiomedCLIP joint space that the KD teacher targets. Not a lever; a different project. Also dilutes an MSc contribution that is about the *text* tower.
- **Image resolution 224 → 336/448/512** — BiomedCLIP ViT-B/16 position embeddings are fixed at 224; interpolating them perturbs the frozen joint space the whole design depends on. Poor cost/benefit here.
- **Two-stage text-only distillation to raise `cos_text_teacher`** — optimises a number that already reaches 0.89 under pure KD. Not the bottleneck.
- **"Switch from last-token to mean pooling"** — moot; the v2 configs use attention pooling (`pooling_strategy: attention`), never last-token.
- **`cos_text_teacher` as an architecture-adequacy gate** — falsified above. Any experiment gated on it draws a wrong conclusion.
- **MIMIC-CXR DICOM (4.7 TB)** — wrong project. Use **MIMIC-CXR-JPG** (~570 GB, same 377,110 images, no `pydicom`, no windowing decisions to defend in a viva). Reports still come from MIMIC-CXR (the 135 MB `mimic-cxr-reports.zip`).
- **Storing MIMIC-CXR-JPG at native resolution** — the model sees 224×224 and source images average ~2500×3056, so ~99% of every pixel array is discarded on load. Downscale in flight; that 99% never touches the 200 GB quota.
- **Keeping `itsanmolgupta/mimic-cxr-dataset` as the data source once credentialed** — it is a third-party redistribution of credentialed data, and its provenance (which studies, which views, which section parser, whether it is subject-disjoint across the 90/10 cut) **cannot be stated**. That is a live viva vulnerability. Migrate to the PhysioNet build and do not cite dependence on the mirror.
- **Pairing lateral views to the study-level report** — manufactures guaranteed in-batch false negatives (the thing 6C-3 measured this dataset does *not* currently have: 0.58 pairs per bs=64 batch) and mixes two visual distributions. Frontal-only, one per study.
- **Re-running Stage 0 from scratch (asked 2026-08-16)** — **NO.** Three independent reasons. (1) **Backbone quality was measured not to transfer**: PPL 15.62 → 13.18 moved retrieval **flat** — null #1 of 10. (2) **Stage 0 trains on PubMed abstracts**; the full-MIMIC build changes nothing it sees. The two are orthogonal. (3) The 13.18 checkpoint is **hard-won** — four failed runs (collapse at steps 3k / 24k / 28k) resolved only by the fp32 Mamba-scan + mLSTM-exp-gate fix — and costs **~3 days of H100 wall**. Re-running re-exposes that fragility for zero expected gain.
  ⚠️ **Honest nuance the pivot introduces:** that null was measured on retrieval, which uses the backbone as an **encoder** (pooled embedding). Report generation uses it as a **generator** — autoregressive decoding, exactly what LM pretraining optimises — so the null does **not** automatically carry over. That argues for *keeping* the 13.18 checkpoint, not rebuilding it. The legitimate cheap version of this lever is **10G below**, not a restart.

---

# ACTIVE WORK STARTS HERE

### Phase 7 — PhysioNet credentialing ⏳ **BLOCKING — CRITICAL PATH, NO CODE**

Everything in Phases 8–11 is gated on this. **It is not "a click"** — this project has *never* had PhysioNet credentialing (state note 2026-07-25: *"no PhysioNet credentialing yet, so full-MIMIC scale-up stays out of scope"*). Realistic lead time is **1–4 weeks**, dominated by human review, so it must start before any code is written and the rest of the plan must be sequenced around it.

- [x] **7A/7B/7C** — **DONE (user, 2026-08-16).** CITI "Data or Specimens Only Research" training complete, PhysioNet credentialed account approved, DUA signed on both `mimic-cxr/2.1.0` and `mimic-cxr-jpg/2.1.0`. PhysioNet username: `bhushkri`.
- [x] **7D-auth-mechanism** — **CORRECTED, 2026-08-16: `.netrc`/HTTP Basic Auth does NOT work against PhysioNet for this project.** Verified live through a long diagnostic chain (netrc parsing, password correctness, file formatting, User-Agent filtering all individually ruled out first): `curl -u user https://physionet.org/settings/profile/` returns **302 to `/login/` regardless of credential correctness**, and a `curl -H "Authorization: Basic ..."` against a `/files/` URL returns **403**, while the **identical URL with a session cookie returns 200**. PhysioNet's Django deployment simply does not honour HTTP Basic Auth for this project — the `wget --user --ask-password` recipe printed on PhysioNet project pages (and the earlier proposal that cited it) is **stale for this deployment**.
      **Auth is now session-cookie based.** `build_mimic_cxr_local.py`'s `_get_session()` reads `~/.physionet_session` (chmod 600, contains only the raw `sessionid` cookie value — not the `.netrc` 3-line format) and raises loudly if it's missing, rather than silently falling through to a Basic-Auth path known not to work.
      **Getting the cookie value:** log into physionet.org in a browser, open dev tools → Application/Storage → Cookies → `physionet.org` → copy the `sessionid` value. On the box that runs `meta`/`fetch`:
      ```
      umask 077; printf '%s' 'SESSIONID_VALUE' > ~/.physionet_session
      chmod 600 ~/.physionet_session
      ```
      ⚠️ **Treat this file exactly like a password.** A live session cookie lets anyone holding it act as you on physionet.org until it expires or you log out. Never paste its value into a script, a chat message, argv, or a log line — this project already had one briefly exposed in chat during debugging (2026-08-16); the standing advice is to log out (which typically invalidates the session) and grab a fresh cookie for actual use, same discipline as a rotated password.
      **The expiry trap this also had to guard against:** when the cookie expires mid-`fetch`, PhysioNet 302s to `/login/`, and `requests` follows redirects by default — so it arrives as an ordinary `200` with an HTML login page as the body. Without a guard, that gets written straight into a `.jpg`/`.csv.gz`, and the resume check (`Path.exists()`) then skips the corrupt file forever on every subsequent run — a silent, discovered-weeks-later bug. `_download()` now checks `Content-Type`/`resp.url` **before** streaming any bytes to disk and returns a `SESSION_EXPIRED` sentinel instead of `200`; `stage_fetch` aborts the whole run immediately on seeing it (checked **unconditionally**, not folded into the `ok==0` guard, since a mid-chunk expiry can leave `ok > 0` in the same chunk). Verified with 4 unit tests (mocked responses, no network): cookie-file-missing raises; a login-page-shaped 200 is detected and writes nothing; a genuine 200 still writes correctly; a partial-success-then-expired chunk still hard-aborts.
      ⚠️ **Second bug caught LIVE (2026-08-16), not just in review — atomic-write fix.** A `--time=00:10:00` job timeout killed a `mimic-cxr-reports.zip` download **mid-stream**. `_download()` previously wrote straight to `dest`, so the truncated-but-nonzero-size file remained; the next run's `stage_meta` printed `"[meta] have mimic-cxr-reports.zip"` and skipped re-fetching it (existence + nonzero size ≠ complete), and the corruption only surfaced later as `zipfile.BadZipFile: File is not a zip file` at unzip time. **Fixed properly, not patched around**: `_download()` now streams to a `dest.name + ".part"` sibling and does an atomic `Path.replace()` into `dest` only after the full body is consumed — `dest` is therefore either absent or complete, never partial, for every caller (`stage_meta` and `stage_fetch` alike). Fixing this surfaced a **second, worse latent bug** in the same function: the HTTP status line (200) arrives before the body is streamed, so a connection that died mid-body (same failure class, one step earlier) was returning a bare `200` to the caller even though `dest` was correctly never written — silently reporting success for a download that produced nothing. Restructured so a body-stream failure is caught in its own scope and retried as a fresh request, never falling through to a stale `last_status = 200`. Verified with a 5th unit test simulating a mid-stream `ConnectionError`: confirms the function does **not** return `200` and `dest` is never created.
      ⚠️ **Third issue caught LIVE (2026-08-16) — `/tmp` is node-local, not shared.** Even after the atomic-write fix landed and was confirmed deployed on the cluster (`grep` for `part.replace`/`SESSION_EXPIRED` in the cluster's copy), the *exact same* `zipfile.BadZipFile` recurred on a fresh submission. Cause: every job so far had landed on the same compute node (`gx17v1`), and `/tmp` on a compute node is typically a **separate, node-local disk** from `/tmp` on the login node (`lx01`). `rm -rf /tmp/mimic_smoke_test` run from the login node had **zero effect** on that compute node's own `/tmp` — the corrupt zip from the very first killed job silently persisted through every subsequent "clean" retry, immune to every `rm`. **Do not use `/tmp` for a cluster smoke test.** Use a path under the shared filesystem instead — the same one the real build already targets (`/sc/home/$USER/dataset/...`).
- [x] **7D-verify** — **DONE (2026-08-16), confirmed live end-to-end.** Job 2457693: `[meta] GET` for all 4 small files + `IMAGE_FILENAMES`, `[auth] session cookie loaded`, `[meta] unzipping reports ...`, `[meta] done -> ...`. Auth, submission mechanics, and file-integrity fix all validated against the real server, not just mocks. Took ~30 min wall (13:19→13:49) — confirms the earlier `--time=00:10:00` was the actual problem all along, not anything structural.
      Command used (now the permanent recipe — nothing more to fix here):
      ```
      STAGE=meta OUT=/sc/home/$USER/dataset/mimic_smoke_test sbatch scripts/build_mimic_cxr_local.sh
      ```
      (`--account=aisc --partition=aisc-batch --qos=aisc` baked into the script's `#SBATCH` header — see 7E. Do not add a manual `--time=...` override: the script's own default (1 day) is what let this actually finish. Use a path under `/sc/home/$USER/...`, never `/tmp` — see 7E's node-local warning.)
- [x] **7E** — **ANSWERED (2026-08-16), confirmed live + against official docs.** The login node **rejects every script execution outright** — `python build_mimic_cxr_local.py meta` on `lx01` was refused before making a single request: *"This command is not allowed on the login node!"* This is not the `bash <script>` restriction noted earlier; it is total. Per `docs.sc.hpi.de/cluster/Resources/{Login-Nodes,Data-Transfer,Partitions}` (fetched live): external downloads belong on **compute nodes via Slurm**, not a Run Node (`rx01`/`rx02` — 8h/4-core cap, explicitly *not* for data acquisition).
      ⚠️ **The docs' `cpu-interactive`/`cpu-batch` guidance below turned out not to work for this account** (see the account/QOS correction further down this section) — what actually runs is `aisc-batch`/`aisc-interactive` with `--account=aisc --qos=aisc` explicitly, confirmed live via jobs 2457565/2457693. Kept both descriptions since the general partitions may still be right for a different account/user.
      - **`cpu-interactive`** (8h cap) — intended for the `meta`/`manifest`/`pack` stages (all short). **Superseded for this account by `aisc-interactive`.**
      - **`cpu-batch`** (7-day cap) — intended for the long `fetch` stage. **Superseded for this account by `aisc-batch`.** Resumable design (Phase 8A) means a timeout just needs a resubmit.
      ⚠️ **`/tmp` is node-local, not shared across login/compute nodes — confirmed live 2026-08-16.** Every job so far landed on the same compute node (`gx17v1`); `rm -rf /tmp/mimic_smoke_test` run on the login node had zero effect on that node's own `/tmp`, so a corrupt file from an earlier killed job silently persisted through several "clean" retries. **Never use `/tmp` for a cluster smoke test or any build path** — use `/sc/home/$USER/dataset/...` (the shared filesystem the real build already targets).
      - `scripts/build_mimic_cxr_local.sh` (NEW) wraps all four stages, `STAGE=` env-selected. **Requests no GPU.** Tested: `test_build_mimic_cxr_local_slurm_wrapper_is_cpu_only_on_cpu_batch`.
      - **Account/partition/QOS — three failed defaults before one that works, confirmed live 2026-08-16 via job 2457565** (auth succeeded, 3/4 small files fetched before an unrelated manual `--time` override killed it): a plain `sbatch` with no `--account` fails outright (`"No Slurm account specified"`); `--account=aisc` on `cpu-batch` sits `PD (QOSNotAllowed)` forever (`aisc`'s QOS is scoped to the AISC partitions only); `--account=default` on `cpu-batch` fails with `AssocMaxSubmitJobLimit`. **What actually runs:** `--account=aisc --partition=aisc-batch --qos=aisc` together — now baked into the script.
        ⚠️ **Tradeoff, not a clean fix:** `aisc-batch` is a GPU-capable partition (lands on nodes like `gx17v1` without requesting/using the GPU) and per `docs.sc.hpi.de` AISC partitions are **preempted at any time** — a real risk for the multi-hour `fetch` stage specifically, beyond just wasted GPU-node occupancy. Worth asking `sc-helpdesk@hpi.de` (see the courtesy-contact item above) whether a non-preemptible CPU-only queue is available for this account before committing to the full run.
      ⚠️ **Courtesy step, not yet done — do this before the full `fetch`.** The Data-Transfer doc states verbatim: *"Always contact helpdesk before transferring large datasets."* Email `sc-helpdesk@hpi.de` — ~310–400 GB is transferred even though nothing is kept, and the cluster explicitly polices "flooding the network" / "saturating connection tracking tables." `WORKERS` defaults to 8 concurrent connections for exactly this reason; do not raise it without checking with them first.
      **Correction to the earlier "compute nodes are offline" note**: that was Stage-0/Phase-6's characterization of a specific 401 (gated-dataset auth failure, not a connectivity failure) baked into `HF_DATASETS_OFFLINE=1`. It does not establish that compute nodes lack general internet egress — and the official docs explicitly recommend downloading external data *from* `cpu-interactive`/`cpu-batch`, which would be nonsensical advice if those nodes had no route out. Treat GPU-node connectivity as still unverified/irrelevant; this build never needs a GPU node at all.
- [ ] **7F** — **Quota check**: `quota -s` / `lfs quota` for both **bytes and inodes**. The build creates ~227k report `.txt` files + ~200k JPGs ≈ **430k inodes**, which can trip an inode cap long before the 200 GB byte cap. If inodes are tight, keep the reports zipped and parse from the archive.

**Interim (unblocked, do while waiting):** Phase 10A/10B (decoder architecture + tests) and Phase 11A (metric harness) need **no data** and can be built and unit-tested against the existing 27.5k mirror. Do them during the credentialing wait rather than idling.

### Phase 8 — Local MIMIC-CXR-JPG build (compact, DUA-compliant) ✅ CODE COMPLETE 2026-08-16 — network stages pending Phase 7E
Target: **~190–210k frontal (image, report) pairs at 320 px ≈ 6 GB on disk**, from ~310–400 GB of streamed-and-discarded transfer. Approach reviewed and adopted 2026-08-16; the corrections identified in that review are implemented, not just noted (see 8B).
- [x] **8A** — `scripts/build_mimic_cxr_local.py`, four stages: `meta` (small files ~150 MB) → `manifest` (no network; decides what to build) → `fetch` (chunked download → resize → **delete originals**; resumable) → `pack` (leakage guard + train/validate/test parquet). Emits `build_report.json`. `manifest` and `pack` integration-tested locally against synthetic PhysioNet-shaped fixtures (see 8H) — both run end-to-end correctly, including the hard-fail path.
- [x] **8B** — **Implementation uses Python `requests`, not wget/curl — sidesteps all five originally-identified corrections rather than patching around them:**
      1. No `--cut-dirs` at all (was the off-by-one risk). Every file is fetched by an **absolute URL** (`JPG + "/" + rel_jpg`) to an explicit destination — no directory-stripping arithmetic to get wrong.
      2. No `wget --base=` / `-i` list — same reason.
      3. No `-N` — not applicable.
      4. **Implemented, not just noted**: `stage_fetch` tallies HTTP status codes per chunk and `raise`s if a chunk converts 0 of N (`build_mimic_cxr_local.py:_fetch`, tested by construction — see 8H), rather than silently spinning through empty chunks.
      5. **Superseded (2026-08-16): `.netrc`/HTTP Basic Auth does not work against PhysioNet for this project at all** — see 7D-auth-mechanism for the full diagnosis. `requests` correctly sent `.netrc`-sourced Basic Auth credentials and PhysioNet rejected them regardless (verified via `curl -v`: 403 with Basic Auth, 200 with a session cookie at the identical URL). `_get_session()` now reads a session cookie from `~/.physionet_session` instead. No subprocess, no shell-quoting, the cookie never touches argv or a log line — same discipline as originally intended for the password, just a different credential.
- [x] **8C** — **Official section parser VENDORED VERBATIM**, not reimplemented: `scripts/mimic_cxr_vendor/section_parser.py` is a byte-for-byte copy of `MIT-LCP/mimic-cxr@e8d26fff` `txt/section_parser.py` (fetched 2026-08-16, commit SHA recorded in the file header and in `build_report.json`). `scripts/mimic_cxr_vendor/extract.py` ports the per-study extraction logic (custom index/section-name overrides, last-matching-section lookup) from the companion CLI `MIT-LCP/mimic-cxr@18cdc41c` `txt/create_section_files.py`, refactored into a function so `manifest` can call it per-row instead of shelling out to a batch CLI. **Verified against a synthetic report** (`tests/test_willi_parity.py::test_extract_findings_impression_basic_and_custom_override`): correctly separates FINDINGS/IMPRESSION and correctly honours a `custom_mimic_cxr_rules()` index override.
- [x] **8D** — **Leakage guard implemented as a hard gate, not a warning.** `pack --exclude-hashes <file> [--min-match-frac 0.95]`: hashes `report_hash` (blake2b, **verified byte-identical** to `normalize_report_text` @ `evaluate_cxr_retrieval.py:414` and to the `text_hash` construction @ `train_contrastive.py:419-424` — same normalisation, same digest, so it is not a second drifted scheme), joins to recover `subject_id`, drops every row from a matched subject, and **`raise`s (does not warn) if match rate < 95%** unless `--allow-low-match` is explicitly passed. Verified with a synthetic fixture: a 50% match rate correctly aborts with a clear error; the same fixture with `--min-match-frac 0.4` correctly proceeds and drops exactly the matched subject's row.
      - `scripts/dump_legacy_gallery_hashes.py` (NEW) produces the `--exclude-hashes` input — loads the legacy `train[90%:]` gallery via the existing `MIMIC_REPO` constant and reproduces the identical text construction (`f"Findings: {findings} Impression: {impression}"`) `MIMICValDataset` uses, so the hashes are computed the same way on both sides of the join.
      ⚠️ **The under-match risk from the original review still applies and is not eliminated by any of the above** — it protects against silently trusting a bad join, it does not make the join better. If it fails, drop the legacy gallery comparison; the official split becomes the sole metric.
- [x] **8E** — Code wiring, all verified with parity tests (no network needed for any of these):
      - `train_contrastive.py:437-490` (`load_mimic_cxr`) — `local_parquet_dir` branch added, dispatches to `load_dataset("parquet", data_files={train,validation,test}, split=...)`. `MIMICJointDataset.__getitem__` needed zero changes (already had the `isinstance(img, str)` branch).
      - `evaluate_cxr_retrieval.py` — **both** `IndianaEvalDataset.__getitem__` and `MIMICValDataset.__getitem__` gained `elif isinstance(img, str): img = Image.open(img)` (the crash the original review missed). `build_dataloader` gained `local_parquet_dir` / `mimic_split` params with the same three-file dispatch. `main()` CLI gained `--local-parquet-dir` / `--mimic-split`.
      - `eval_h100.sh` gained `LOCAL_PARQUET_DIR` / `MIMIC_SPLIT` env levers (empty default → legacy mirror, unaffected).
      - `configs/dataset/cxr_mimic_full.yaml` (NEW) + `DATASET_CONFIG` env lever in `train_biomedclip_kd_h100.sh` (default `mimic_cxr` unchanged — **Phase 9A's Arm-0 control is unaffected by this file's existence**).
      - Grayscale (`"L"`) JPEG storage confirmed safe on both read paths (training converts to RGB explicitly; eval has `if img.mode != "RGB": convert("RGB")`).
- [x] **8F** — `manifest` stage emits `findings` and `impression` as **separate untruncated columns**, plus `has_findings` and `has_text` (findings-or-impression) flags, and `build_report.json` reports **both** counts (`with_findings`, `with_findings_or_impression`) plus a per-split breakdown. The findings-only-vs-both choice for Phase 10 training is therefore a config-time decision on already-separate columns, not a re-run.
- [x] **8G** — `.gitignore` guards added: `dataset/mimic_full/`, `mimic_full/`, `*.parquet`, the small PhysioNet metadata files, `legacy_gallery_hashes.txt`. Build path (`/sc/home/krishankumar.bhushan/dataset/mimic_full`, set as the default in `cxr_mimic_full.yaml` — **fixed 2026-08-16**, an earlier version wrongly used the PhysioNet username `bhushkri` instead of the cluster username) is outside the repo regardless.
- [x] **8H** — `bash scripts/validate_for_willi.sh` green: **99 passed** (was 92), 5 skipped, 9/9 gates. 7 new parity tests added covering the hash-join convention, the vendored extractor (incl. the custom-override path), the new config's schema, the `load_mimic_cxr` local-parquet dispatch (mocked `load_dataset`, verifies exact `data_files` paths), the `evaluate_cxr_retrieval.py` str-image fix (real temp JPEG round-tripped through both dataset classes), the env-lever wiring, and the `.gitignore` guard. **Beyond the required parity tests**, `manifest` and `pack` were run end-to-end against synthetic PhysioNet-shaped CSV/report fixtures (not part of the pytest suite — network-shaped integration checks) and produced correct output, including the hard-fail path on a deliberately-low leakage-guard match rate.
      **What is NOT yet tested and cannot be from here**: `stage_meta` and `stage_fetch` need a real PhysioNet connection. Run the `meta` stage first (small, ~150 MB) as the real auth/connectivity smoke test once Phase 7E answers where the job can run.

### Phase 9 — Retrieval on full data (supporting chapter, extended) ⏳ blocked on Phase 8
Single-lever attribution preserved. **Arm 0 runs first and is non-negotiable** — it is 3.5 GPU-h against a confounded multi-day result.
- [ ] **9A** — **Arm 0 — reproduction control.** Rebuild ONLY the same ~27.5k studies through the new pipeline, rerun the D1c recipe. **Gate: reproduce 0.1459 ± 1.1pp.** If it misses, the pipeline changed something (section parser, view selection, the extra resample) and that is found for 3.5 GPU-h instead of inside a confounded result.
      ⚠️ **PROTOCOL TRAP — do not compare a full-MIMIC number on the official split against 0.1459 on the legacy gallery.** The official split is subject-disjoint *by construction*; the legacy `train[90%:]` gallery has **unknown provenance** and may itself leak subjects across its own 90/10 cut. The new protocol is therefore **harder**, and the number can go **down while the model gets better**. Arm 0 exists precisely to separate "the pipeline changed" from "the protocol got harder" from "the model changed". Never quote a cross-protocol delta.
- [ ] **9B** — **Arm 1 — data only.** Full build, D1c recipe unchanged, same 6,000 steps. Note the epoch budget moves 13.93 → ~1.8 at identical GPU-hours: same compute, far less repetition.
- [ ] **9C** — **Arm 2 — the actual hypothesis.** `vit_lr` ∈ {1e-6, 3e-6, 1e-5, 3e-5} on the full set. **Pre-registered prediction: the inverted-U optimum shifts right of 1e-6 and the peak is higher.**
      ⚠️ **State the hypothesis narrowly.** This plan already established that *optimization-side* overfitting is NOT binding (epochs 23→14 = +0.06pp; lower LR removed the rollover without changing plateau height). What 6G-1 measured is that the **image tower specifically** memorises (`train/clip_loss` 1.17→0.04 while val rises). So the claim under test is *"more data relaxes the image-adaptation dose constraint"* — **not** "more data fixes overfitting". If the inverted-U does not move, that is a real and reportable result either way.
- [ ] **9D** — **Image augmentation — the one untested free lever.** **Verified: there is no augmentation anywhere in this repo.** All four transform sites (`train_contrastive.py:186,300,368`, `evaluate_cxr_retrieval.py:250`) are `Resize → [Grayscale(3)] → ToTensor → Normalize`. An 85M-param ViT is being trained on 27,570 images with none — and 6G-1 measured that it memorises them. Add `RandomResizedCrop(224, scale=(0.8,1.0))` + mild rotation as **one parallel arm** (this is why 8 stores at 320 px). Not on the rejected-by-prior-evidence list; attacks exactly the mechanism 6G-1 identified.
- [ ] **9E** — **Free methodological result:** rerun the 6G-7 2×2 (`{vit=2, vit=12} × {val==test, clean}`) on full data. The measured finding was that test-informed selection is worth 2.2pp at 85M trainable image params **because the model overfits**; with ~8× data that number should **shrink**. Measuring how selection-protocol advantage scales with data volume is a clean, rarely-published contribution and costs nothing extra.
- [ ] **9F** — Indiana + official-split evals. Gate: Indiana i2t R@10 ≥ 4.04% floor (target 5.5%) with MIMIC held ≥ Phase-6G.
- [ ] **9G** — *(optional, was Phase 7)* multi-source CheXpert/PadChest/VinDr via `cxr_multi.yaml` + `CXRJointDataset` text adapter. **Deprioritised**: full MIMIC is ~8× the data for a fraction of the engineering, and PhysioNet credentialing also unlocks **VinDr-CXR** directly. Revisit only if Indiana is still flat after 9F. **IU-Xray stays EXCLUDED from training (= Indiana eval; zero-leakage).**

### Phase 10 — Image-conditioned report generator 🎯 NEW — the pivot's core work
**Verified gap:** `HybridLanguageModel.forward()` (`hybrid_lm.py:147`) accepts **only `input_ids`** — no `inputs_embeds`, no `encoder_hidden_states`, no cross-attention. `generate()` (`:230`) likewise. **There is currently no way to condition the decoder on an image.** This is the single largest piece of new work in the pivot and the proposal under review does not mention it.
- [ ] **10A** — `forward(..., inputs_embeds=None)` + `generate(..., prefix_embeds=None)` on `HybridLanguageModel`. Additive and default-`None`, so every existing Stage-0/contrastive checkpoint and eval path is untouched. Pin with a parity test asserting `inputs_embeds=None` is bit-identical to today.
- [ ] **10B** — **Prefix conditioning first** (not cross-attention): project the BiomedCLIP ViT patch grid (197×768) through a small learned mapper to `k` prefix tokens in the decoder's `dim=768` space, prepend, train with cross-entropy on the report. Rationale: Mamba/mLSTM are recurrent — a prefix is absorbed into the state and needs **zero** changes to the SSM/TFLA kernels, whereas cross-attention would mean new per-layer modules and new Triton work. **Sweep `k` ∈ {8, 32, 64} — this is the depth-analogue lever and the most likely place a real effect lives.**
- [ ] **10C** — Initialize the image tower from the **Phase-9 best contrastive checkpoint**, not stock BiomedCLIP. This is exactly where the retrieval chapter pays for itself, and 6C measured the size of that dividend: fine-tuned ViT 0.1172 vs stock 0.0232 with the same text tower. **Run stock-ViT as an ablation arm** — the delta is a headline result linking the two chapters.
- [ ] **10D** — Decoder init from the Phase-5 Stage-0 backbone (val PPL 13.18). Train on `findings`; decide findings-only vs findings+impression per 8F and record it.
- [ ] **10E** — `configs/model/hybrid_150m_v2_rrg.yaml` + `scripts/train_report_generation.py` + SLURM wrapper. Reuse `train_biomedclip_kd_h100.sh` env-lever conventions so an unmodified invocation is the control.
- [ ] **10F** — Numerical gates: finite fwd/bwd, grad-norm bounded, no NaN over 50+ steps. ⚠️ **Stage-0 taught that 150M is spike-fragile** — the fp32 mLSTM-gate / Mamba-scan fix (2026-07-16) is load-bearing; keep `gradient_clip_val=0.5` and monitor `grad_norm` from step 0.
- [ ] **10G** — *(optional arm, replaces "re-run Stage 0")* **Domain-adaptive continuation of the LM on MIMIC report text.** ~190k reports × ~250 tok ≈ **47M tokens** (vs Stage-0's 483M PubMed) — roughly one epoch, **hours not days**. Radiology report style is very distinct from abstract prose, and unlike the retrieval case the backbone here is used as a **generator**, so the Stage-0-quality null does not automatically apply. **Run as an ablation with the untouched 13.18 checkpoint as control** — never as the default. Blocked on Phase 8 (no report corpus until then).

### Phase 11 — Report-generation evaluation 🎯 NEW
- [ ] **11A** — `scripts/evaluate_report_generation.py`: **ROUGE-L** (primary), BLEU-1/4, METEOR. Fixed decoding config (greedy + beam=3, both reported), fixed `max_new_tokens`. **No data needed — build during the Phase-7 wait.**
- [ ] **11B** — **CheXbert F1**: run the CheXbert labeler over generated and ground-truth reports; report micro/macro F1 over the 14 CheXpert labels and the 5-label subset the RRG literature uses. `mimic-cxr-2.0.0-chexpert.csv.gz` (already downloaded in 8A) gives the ground-truth labels as a cross-check on the labeler wiring. Offline weights must be staged — the compute nodes have no internet.
- [ ] **11C** — **Retrieval-NN baseline, run BEFORE the generator.** Use the Phase-9 model to retrieve the nearest training report and emit it verbatim. n-gram metrics reward templated text heavily and MIMIC reports are heavily templated (6C-3: 2% of the gallery is exact-duplicate). **This baseline is the real floor**; a generator that does not beat it has contributed nothing.
- [ ] **11D** — Report on the **official subject-disjoint test split**. Quote the legacy `train[90%:]` gallery only if 8D's recall check passed.
- [ ] **11E** — Human-readable qualitative appendix: N=20 side-by-side generated/ground-truth pairs incl. failure cases. Cheap, and it is what a viva actually asks about.

### Phase 12 — Full eval + comparison + writeup
- [ ] **12A** — Authoritative MIMIC + Indiana retrieval + STS (BIOSSES/STS-B) + PubMed PPL on the best 150M ckpt (the supporting chapter's final table).
- [x] **12A-eff (tooling)** — **Efficiency-curve harness ready.** `scripts/performance_profile.py` gained a `--sweep` mode: latency / throughput / peak-memory vs sequence length across multiple configs, with fitted log-log **scaling exponents** (~1.0 = linear, ~2.0 = softmax attention), CSV + JSON output, per-point peak-memory reset, and OOM points recorded rather than fatal. `--backward` measures the training step. `scripts/profile_efficiency_h100.sh` (NEW) runs hybrid vs both single-family baselines at identical dim/depth, inference + training, L = 256…16384.

  **Bug found and fixed:** the profiler resolved `--model` through `ModelRegistry`, which only ever registers `hybrid_350m/1_3b/7b/mamba_baseline/xlstm_baseline`. **Six of the nine names in its own `--model` choices list — every 70M and 150M config, including the active `hybrid_150m_v2` backbone — raised `ValueError` before a single measurement ran.** Configs now resolve from `configs/model/*.yaml` (the source of truth) with the registry as fallback. Pinned by `test_performance_profile_loads_every_advertised_model_config`.

  Sweeping past `max_position_embeddings` (1024) is valid because `use_pos_embedding = False` (`hybrid_lm.py:43`) — there is no absolute position table to index out of. Pinned by `test_sequence_sweep_is_valid_past_max_position_embeddings` so that re-enabling it fails loudly.

  No dataset or checkpoint needed — random token ids, fresh weights; throughput and memory do not depend on weight values. **Run:** `sbatch scripts/profile_efficiency_h100.sh` (add `SCALE=70m` for the 70M family).
- [x] **12A-eff (measured, 2026-07-28)** — H100 80GB HBM3, gx07, bf16, bs=4, jobs 2382432 (150M) / 2382434 (70M). Curves at `analysis/efficiency_{150m,70m}/{inference,training}/efficiency_curves.{csv,json}` **on the server — not yet pulled into the repo**.
      1. **Linear scaling confirmed** to L=16,384. Asymptotic (L≥4096) exponents ≈1.0 for latency and memory across all three architectures. ⚠️ **The printed full-range exponents understate scaling** (xLSTM 0.652) because the H100 is underutilised at short sequences (150M xlstm: L=2048→92.0ms, L=4096→99.0ms — almost free), dragging the fit down. **Report asymptotic alongside full-range**: latency hybrid 1.012 / mamba 1.011 / xlstm 0.920; memory 0.965 / 0.965 / 0.880. *Not yet implemented in the tool — see open item below.*
      2. **xLSTM is dramatically cheaper than Mamba.** 150M @ L=16384 inference: mamba 1105 ms / 41.99 GB / 59,280 tok/s; hybrid 925 ms / 42.00 GB / 70,882 tok/s; **xlstm 355 ms / 7.12 GB / 184,860 tok/s** — 5.9× less memory, 3.1× faster, *despite more parameters* (159.0M vs 140.5M non-emb).
      3. **The hybrid's win is in TRAINING, not inference.** Inference peak = max over layers, so hybrid ≈ pure Mamba (42.00 vs 41.99 GB). Training peak = *sum* of saved activations, so composition matters: 150M @ L=2048 training — mamba 1348 ms / 67.5 GB vs **hybrid 1078 ms / 54.0 GB (25% faster, 25% less memory)**; xlstm 309 ms / 11.2 GB. Same pattern at 70M. **This is the concrete, defensible justification for the hybrid design over pure Mamba.**
      4. **Training memory is the ceiling**: 150M hybrid and mamba both OOM at L=4096 on 80GB; xlstm reaches 8192. The exponent is clean linear (0.97–0.98) ⇒ constant-factor problem, not a complexity problem.
      5. **Not a bottleneck at this project's lengths.** CXR reports ≤256 tok, PubMed ≤512. At L=256 (150M) the hybrid is the *fastest* of the three (18.08 vs 19.78 / 21.12 ms). **The curves are a separate architectural contribution — they do not explain any retrieval or generation number.**
      6. ⚠️ **Caveat for the writeup:** the "~2.0 = quadratic attention" line in the tool's output is a **reference claim, not a measurement** — there is no attention/transformer baseline in this repo (verified by grep over `configs/model/` and `hybrid_xmamba/layers/`). Either add one at identical dim/depth or state the 2.0 as cited, never as measured.
      7. Incidental: xLSTM shows a throughput regime change at L=4096 (150M 89,042 → 165,436 tok/s; 70M 132,806 → 246,309) — a TFLA kernel efficiency jump.

      **Open tooling items (optional, not blocking):** report asymptotic exponents alongside full-range in `performance_profile.py`; add an attention baseline to actually measure the ~2.0 curve; add an autoregressive decode benchmark (fixed-size recurrent state vs growing KV cache) — that last one is now directly relevant, since Phase 10 makes generation the product.
- [ ] **12B** — `analysis/h100_scaling_results.md` (NEW). Structure the writeup as: **(1) report generation** (primary — ROUGE-L / CheXbert F1 vs the retrieval-NN baseline); **(2) retrieval** (supporting chapter — the 10-nulls-1-lever attribution table, the ViT-depth dose-response, and the selection-protocol methodological finding); **(3) efficiency** (linear scaling; hybrid's training-memory advantage over pure Mamba); **(4) honest limitations.**
- [ ] **12C** — Update `h100_scaling_state.json` final verdict + best ckpt path.

---

## Verification (each phase gates on)
1. `bash scripts/validate_for_willi.sh` exits 0 (3.9-syntax hygiene kept, forward-compatible) + H100-env stack smoke on py≥3.10 (Phase 2).
2. New phase test passes.
3. Numerical: forward/backward finite on CPU + H100 smoke; grad-norm bounded; no NaN 50+ steps; i_gate<cap.
4. SLURM kill-gates pre-declared (Phase 5 PPL; Phase 6 cos_text_teacher/clip_loss/R@10).
5. Reconcile in-training vs authoritative eval numbers (refactor caught 4 latent bugs this way).

## Resumability contract
1. Read `H100_SCALING_PLAN.md` + `h100_scaling_state.json` at session start.
2. Resume at `h100_scaling_state.json["current_phase"]`. Checkboxes here = ground truth.
3. After every state change: tick checkbox + update `last_updated` (ISO 8601) + append 1-line `notes` entry.
4. If state JSON missing locally, regenerate from this file's checkbox state (gitignored on purpose).
5. Never re-run a checkpoint-producing phase (5,6,7) without first reading its log + logging a verdict.

## Lessons carried from the completed refactor (do not repeat)
- CLIP negatives are in-batch only — grad-accum does NOT add negatives; per-step batch is the real lever.
- `norm_topology` must be threaded into every `HybridConfig` builder (train/distill/contrastive) — silently dropped = wrong FFN forward.
- Eval must auto-detect `layer_pattern` + `norm_topology` from checkpoint; must load fine-tuned `image_encoder.*` (fresh-ViT load read 1.89% vs true 10.94%).
- MoCo queue post-KD-warmup is harmful → keep `moco_queue_size=0`.
- freq-decoupled KD hurt Indiana → stays off (canonical).
- Always reconcile in-training vs authoritative eval numbers before citing.
- 150M is **spike-fragile**: the fp32 Mamba-scan / mLSTM-exp-gate fix (2026-07-16) is load-bearing; keep `gradient_clip_val=0.5`. Any new training objective (Phase 10) re-exposes this.
- A run that reports success while doing nothing is the expensive failure mode (`--cut-dirs`, `check=False`). **Every long-running loop needs an assertion that it produced output.**

## Unresolved questions
- PhysioNet credentialing lead time — unknown until 7B is submitted; **everything downstream is gated on it**. Start Phase 10A/10B/11A meanwhile.
- Report-gen text target: **findings-only** (RRG convention) vs findings+impression (what the retrieval chapter used)? Decide at 8F, record in `build_report.json`.
- Prefix length `k` for image conditioning — sweep {8,32,64} at 10B; no prior.
- Cross-attention (10B alternative) — deferred unless prefix conditioning underperforms; it means new per-layer modules and Triton work.
- 8D hash-join recall vs the legacy gallery — if < 95%, the `train[90%:]` continuity number is dropped and the official split becomes the sole metric.
- Where the ~310–400 GB fetch can actually run (7E) — login node under tmux vs transfer node vs egress-capable partition.
- CheXbert labeler weights offline-staging on aisc (11B).
- CheXpert/VinDr label→prompt template wording — only if 9G is revived.
- Whether willi/A100 remains a target after H100 migration (if retired, drop py3.9 guards + `validate_for_willi.sh`).
