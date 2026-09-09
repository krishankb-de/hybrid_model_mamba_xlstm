# H100 Plan-of-Record — Hybrid Mamba-xLSTM **CXR Report Generation**

> Resumable plan-of-record. Read this + `h100_scaling_state.json` (gitignored, allowlisted) at session start.
> **Builds on the COMPLETED `HYBRID_ARCH_REFACTOR_PLAN.md`** (broke the MIMIC ceiling 8.23%→10.45% i2t R@10). That plan is finished — historical reference only.
> Full approved plan: `/Users/krish/.claude/plans/i-want-to-implement-twinkling-ullman.md`.
> Phase 10A/10B-architecture implementation plan (executed 2026-08-20): `/Users/krish/.claude/plans/pure-hatching-stallman.md`.
>
> ### ⚡ PLAN REOPENED 2026-09-07 — PHASE 14 (supervisor review). `current_phase: phase14_supervisor_review`.
> Results were reviewed by the supervisor. Three findings, all about **validity of the central claim**, none about chasing a better number. In the supervisor's own priority order:
> **(1) HIGHEST — there is no trained, parameter-matched Transformer baseline anywhere in the report.** The thesis is *"attention-free hybrid matches/beats attention-based transformers at better efficiency"*, but every comparison in the writeup is against this project's own architecture variants, an off-the-shelf non-fine-tuned model (BiomedCLIP zero-shot), or a naive nearest-neighbour control — none of which is the baseline the claim is about. **Nothing else matters if this isn't in place.**
> **(2) ✅ CLOSED 2026-09-07 (14B): 73.6% → 36.1%** exact-duplicate clustering on 13D, like-for-like on `validate` (n=1433, the split the historical figure came from), against controls at **1.7%** (references) and **6.4%** (retrieval-NN). On the official test split (n=2663) the same measurement gives 29.2% vs 0.2% / 7.3% — duplication is split-dependent, so always quote the split. Pre-registered outcome **INTERMEDIATE**: the "beats the retrieval floor" result is **not** hollow, but the generator is still ~7× less lexically diverse and 2.3× more self-similar than either control, so the exact-duplicate headline flatters it. Original finding: **the boilerplate/duplicate-template rate was never re-measured on the final (13D) checkpoint** — 73.6% of generations fell into 184 duplicate clusters on a *pre-Phase-13* checkpoint. Biggest validity threat to the primary result: if the generator is still mostly copying templates, "beats retrieval baseline" is hollow, since that is exactly what the retrieval baseline does too.
> **(3) The disclosed selective-scan correctness defect is stated but neither fixed nor bounded** — the fp32 guard is in, but the `clamp(min=1e-8)` divide-by-decay approximation is still there and there is still no test against an exact reference recurrence. Fix it, or bound it and report the max deviation.
> ⚡ **14A HEAD-TO-HEAD COMPLETE (2026-09-09) — SPLIT DECISION.** On the official test split (n=2663) the **hybrid wins ALL FOUR CheXbert F1 metrics** (14-micro 0.4736 vs 0.4590, 14-macro 0.2800 vs 0.2774, 5-micro 0.5522 vs 0.5249, 5-macro 0.4487 vs 0.4319) while the **Transformer wins all four surface metrics** (ROUGE-L 0.1936 vs 0.1899, BLEU-1/4, exact-match accuracy). CheXbert F1 asks whether the right *findings* were asserted; ROUGE/BLEU ask whether the *text* matches — and this plan designated CheXbert the more clinically meaningful metric back in Phase 11, when the hybrid was losing on it. The Transformer also wins Stage-0 PPL (11.222 vs 13.18) and **every** efficiency measure (14A-7), though the efficiency benchmark carries a large implementation confound. **Pre-registered quality bar: the CheXbert-14-micro half is CLEARED; the ROUGE-L half turns on a 0.0037 gap and needs the paired bootstrap (tooling shipped, command in 14A-6).**
> ⚠️ **14A-3 RESULT (2026-09-08): the parameter-matched Transformer BEATS the hybrid on Stage-0 LM perplexity, 11.222 vs 13.18 (−14.9%), under a verified single-lever comparison.** First head-to-head of the project, and it went against the thesis. Stage-0 PPL is a text-only metric and this project has measured it not transferring before (null #1 of 10, on retrieval) — but that null used the backbone as an *encoder*, whereas report generation uses it as a *generator*, which is exactly what LM pretraining optimises. Treat it as a genuine warning sign for 14A-5, not a dismissable metric. The pre-registered failure statement for 14A stands as written.
> Full work breakdown, pre-registered success bars, and exact commands: **Phase 14** below. **Phases 1–13 are unchanged and still valid** — Phase 14 adds the missing baseline and the missing validity checks; it does not re-litigate any closed arm. Retrieval stays closed. **⚠ Operator freeze in force: do not change the selective scan while 14A is running (see 14C).**
>
> **PHASE 13 ARC COMPLETE (2026-09-03). Final checkpoint: `outputs/h100_report_gen_full_ext_4gpu_tower13d/checkpoints/last.ckpt` (13D, `vit_lr=3e-6` image tower + extended decoder training + beam decode), confirmed on both validate.parquet and the official test split. The 3-arm `vit_lr` sweep (1e-6→3e-6→1e-5) found `3e-6` is the peak for downstream usefulness — `1e-5` regressed on every CheXbert metric despite the tower's own retrieval R@10 still climbing, so the `3e-5` arm was skipped as not worth the compute. Summary: 13A (free beam decode) + 13B (extended training) cleared milestone 1 outright (CheXbert-14-micro beats the retrieval floor); 13C/13D (image tower on full data) pushed further, peaking at `3e-6` — CheXbert-14-macro's gap to the floor fell 17.5%→10.7%→7.1%, and CheXbert-5-macro now also beats the floor; 13F (rare-label oversampling) was tried honestly and failed, abandoned. **PLAN OF RECORD FULLY CLOSED (2026-09-03).** Phase 12 writeup done (`analysis/h100_scaling_results.md`, `h100_scaling_state.json`'s `final_verdict`); 12A closed (BIOSSES ρ=0.3829, STS-B ρ=0.4472 measured 2026-09-03, job 2505443 — first STS measurement of any kind in this project; PubMed PPL was already covered by Stage-0's existing val PPL 13.18). All of Phases 12 and 13 are checked off. No open items remain except the honestly-flagged limitations in `analysis/h100_scaling_results.md` §4 (unverified boilerplate rate on the final checkpoint, the never-run `vit_lr=3e-5` arm, Indiana never revisited for generation).** 11D (2026-08-30, official test split n=2663) CONFIRMED the same mixed result as 11B/11C below on a fresh split: generator wins ROUGE-L/BLEU-4/accuracy (0.1816 vs 0.1636 rouge_l), retrieval-NN floor wins CheXbert-14-micro decisively (0.4296 vs 0.3326, +29% rel.) — closing Phase 11 with the negative CheXbert result generalized, not a validate-split fluke. Per user decision 2026-08-30, Phase 13 now spends real compute trying to close this gap (staged: decode strategy → decoder training length [+ optional multi-GPU DDP, code shipped] → full-data image tower retrain → imbalance fix if still needed) before Phase 12's writeup is finalized. Phase 11 (below) is the closed historical record of how the gap was first found. Phase 8 (fetch+pack) and the full-data Phase 10E training run (job 2491338, 191,462 pairs) are DONE (2026-08-28). Full n=1433 eval on validate.parquet (11C full, jobs 2491600/2491687) REVERSES the n=10 preliminary read: generator BEATS the retrieval-NN floor on ROUGE-L/BLEU (rouge_l 0.2075 vs 0.1881; bleu_1 0.2706 vs 0.2605; bleu_4 0.0707 vs 0.0465). Caveat: 1055/1433 (73.6%) generated reports fall into one of 184 exact-duplicate template clusters — real signal but still heavily templated. **11B DONE 2026-08-30 (job 2494784): real CheXbert F1 numbers, n=1433 validate.parquet — 14-label micro/macro 0.3097/0.1548, 5-label micro/macro 0.3059/0.1983, exact-match accuracy 0.3531.** Clears the plan's pre-registered Floor tier (ROUGE-L≥0.15 ✅0.2075, CheXbert-14-micro≥0.25 ✅0.3097) but is well below Target (ROUGE-L≥0.22, CheXbert-14-micro≥0.40). Per-label breakdown: high precision / very low recall almost everywhere (e.g. Atelectasis 0.41P/0.04R, Pleural Effusion 0.73P/0.15R), 4 of 14 labels never predicted at all (Lung Lesion/Pneumonia/Pneumothorax/Pleural Other, F1=0) — consistent with the 73.6% boilerplate-template finding above: the generator is conservative and rarely asserts rare findings. Getting here required finding and fixing FIVE bugs in the unmaintained 2023-era `f1chexbert` package/tooling (wrong assumed API; `HF_HUB_OFFLINE` blocking its download; a swallowed `force_filename`-removed exception; `encode_plus` removed in `transformers>=5.0`, worked around with a fully isolated venv; `_check_targets` gaining a 4th return value in `scikit-learn>=1.8.0`) plus two venv-rebuild tooling failures (`uv venv --clear` unsupported then unreliable on this cluster's NFS home, worked around with an explicit `rm -rf`) — full history in the 11B-infra checkbox below. **Two caveats before treating 0.3097 as final:** (1) this is `validate.parquet`, not the official subject-disjoint test split — that's 11D, still open; (2) the retrieval-NN baseline has no CheXbert F1 yet, so per the plan's own pre-registered rule ("the retrieval baseline's CheXbert F1 is the real floor"), the generator is not yet confirmed to beat it on this metric specifically. **A first attempt at (2) (job 2494817/2495070) was INVALID** (stale arm0 defaults in `retrieval_baseline_h100.sh`/`inspect_report_generation_h100.sh`, both now fixed). **The corrected rerun (jobs 2495080/2495164) is VALID and COMPLETES the floor comparison: retrieval-NN CheXbert F1 (14-label) micro/macro = 0.4145/0.3054, (5-label) micro/macro = 0.4624/0.4118 — both well ABOVE the generator's 0.3097/0.1548 and 0.3059/0.1983.** The result is MIXED, not a clean win: the generator beats the retrieval floor on ROUGE-L/BLEU/accuracy (rouge_l 0.2075 vs 0.1881; exact-match accuracy 0.3531 vs 0.3036) but **loses badly on CheXbert F1**, the more clinically meaningful metric (retrieval +34%/+97%/+51%/+108% relative across 14-micro/14-macro/5-micro/5-macro). Per the plan's own pre-registered rule ("a generator that does not beat its own retrieval baseline has not contributed anything"), the generator has **not** cleared this bar on CheXbert F1, only on n-gram overlap and exact-match accuracy. Full writeup in the 11B/11C checkboxes below. Also open, independent: 11D (official test split), 11E (qualitative appendix).
> Phases 1/2/4/5/6/6B/6C/6D/6G/7/8/11 are **COMPLETE and CLOSED**. Phase 3 deferred (its lever was measured non-binding for retrieval; report-gen's decoder DDP lever, Phase 13B, does not need it — see Phase 13). Phase 9's arms 9B (full-data recipe-unchanged retrain) and 9C (vit_lr sweep) — deferred 2026-08-20 in favor of moving straight to Phase 10 — are now **REOPENED as Phase 13C/13D**, run to help close the CheXbert gap rather than to chase retrieval R@10 (do not use their results to reopen the closed retrieval numbers above). See `h100_scaling_state.json` notes for the full rationale.

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

### Phase 7 — PhysioNet credentialing ✅ **COMPLETE (2026-08-16) — no longer blocking anything**

**CLOSED 2026-08-16.** Credentialing, both DUAs, and a verified live download are all done (7A–7E below), and every downstream phase it gated has since run to completion on the real data: Phase 8 `fetch` pulled **217,999 / 218,131 images (99.94%)** and `pack` produced the splits that Phases 9–13 trained and evaluated on (191,462 train pairs; official subject-disjoint test split n=2663). **Nothing in this plan is gated on PhysioNet any more.**

⚠️ **What remains true and load-bearing** (do not delete when trimming this section): the DUA obligations survive the credentialing step. The data stays under `/sc/home/$USER/dataset/mimic_full/` (outside the repo), the `.gitignore` guards from 8G stay in place, and `~/.physionet_session` is still a live credential to be treated like a password. The original framing below (kept verbatim as the historical record) was correct at the time and is *why* this was sequenced first — it was never "a click": *"this project has never had PhysioNet credentialing (state note 2026-07-25); realistic lead time 1–4 weeks, dominated by human review."*

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
- [x] **7F** — **CLOSED 2026-09-07 — resolved empirically by the thing it was guarding.** The concern was that `fetch` writing ~218k individual JPGs might hit an undocumented inode cap. `fetch` **completed** (8I: 217,999/218,131 = 99.94%) and `pack` produced the splits that all of Phases 9–13 trained on, so the failure mode this item existed to pre-empt did not occur — and by the build's own hard-fail design it would have surfaced as a loud error, not silent corruption, if it had. The **byte** side was separately confirmed live by the user's own `du` audit (2026-09-01, home directory approaching the documented 200 GiB cap, which triggered a checkpoint/output cleanup pass). **Honest residual:** an explicit inode *count* was never measured — no documented command for it exists on this cluster — so this is closed as "moot, the run finished", not as "measured". If a future build re-runs `fetch` from scratch, re-read the original text below first.
      ~~Original 7F text, kept for the record:~~ **Quota check — still open, do this in parallel with `fetch` running, not after.** Confirmed live via `docs.sc.hpi.de/cluster/Storage/Quotas/`: home directories have a documented **200 GiB** byte cap, but **no documented command exists for checking inode usage** — that page states only `du -hd 1 . | sort -hr` / `ncdu` for directory *size*, nothing for inode *count*. The build creates ~227k report `.txt` files + ~218k JPGs ≈ **~450k inodes** (updated from the manifest's real `after_one_per_study` count), which can trip an inode cap long before the 200 GiB byte cap even if bytes look fine. **The only reliable way to get an exact inode limit is to ask `sc-helpdesk@hpi.de` directly** — bundle this into the same email as the courtesy large-transfer notice below rather than sending two.
      ⚠️ **`fetch` was already launched (job 2457894) before this and the helpdesk email were done.** Not a mistake to panic over — the hard-fail design means an inode-cap hit would surface as a loud, clear error, not silent corruption — but do this check now, in parallel, rather than finding out at hour 20 of a long run. If inodes turn out tight, keep the reports zipped and parse from the archive (avoids ~227k of the ~450k inode count).
      **Why this didn't matter for `meta`/`manifest`:** those stages are pure local CPU work producing at most a handful of files (a few small CSVs, one `manifest.parquet`) — no per-image files exist yet, so neither bytes nor inodes are a live concern until `fetch` starts writing ~218k individual JPGs. That's also why the courtesy helpdesk email only needed to happen before `fetch`, not before `manifest`: `manifest` never touches the network or writes at meaningful volume, so there is nothing for the cluster's network-flooding policy or a helpdesk heads-up to apply to before that point.

**Interim (historical — this was the wait-time plan, and it is what actually happened):** Phase 10A/10B (decoder architecture + tests) and Phase 11A (metric harness) need **no data** and can be built and unit-tested against the existing 27.5k mirror. Do them during the credentialing wait rather than idling.

### Phase 8 — Local MIMIC-CXR-JPG build (compact, DUA-compliant) ✅ **COMPLETE (code 2026-08-16, network stages 2026-08-27)** — 8A–8I all done
Target: **~190–210k frontal (image, report) pairs at 320 px ≈ 6 GB on disk**, from ~310–400 GB of streamed-and-discarded transfer. Approach reviewed and adopted 2026-08-16; the corrections identified in that review are implemented, not just noted (see 8B).
- [x] **8A** — `scripts/build_mimic_cxr_local.py`, four stages: `meta` (small files ~150 MB) → `manifest` (no network; decides what to build) → `fetch` (chunked download → resize → **delete originals**; resumable) → `pack` (leakage guard + train/validate/test parquet). Emits `build_report.json`. `manifest` and `pack` integration-tested locally against synthetic PhysioNet-shaped fixtures (see 8H) — both run end-to-end correctly, including the hard-fail path.
- [x] **8A-live** — **`meta` + `manifest` run on the FULL real corpus (2026-08-16/17), numbers land almost exactly where predicted:**

  | Metric | Predicted (plan) | Actual (`build_report.json`) |
  |---|---|---|
  | `with_findings` | ~150–160k | **149,496** |
  | `with_findings_or_impression` | ~190k+ | **218,131** |
  | `est_stored_gb` | ~6 GB | **6.54 GB** |
  | `est_transfer_gb` | ~310–400 GB | **322.8 GB** |
  | split | — | train 213,357 / test 3,041 / validate 1,733 |

  `after_frontal_filter` 243,334 → `after_one_per_study` 218,139 (377,110 rows in the raw split CSV). `reports_missing_on_disk: 0`. Confirms the whole pipeline — official section parser, frontal/one-per-study filtering, hash construction — behaves correctly at full scale, not just on synthetic fixtures. `fetch` (218,131 images, ~323 GB transfer) launched 2026-08-16 15:31 (job 2457894).
  ⚠️ **Two more live bugs found and fixed while `fetch` was running** (apply to the *next* invocation — an already-running job keeps its already-loaded code):
  1. **`_get_session()` race condition.** `stage_fetch`'s `ThreadPoolExecutor` calls it from up to `workers` threads concurrently on the first chunk; the original check-then-set wasn't atomic, observed live as **5 duplicate `"[auth] session cookie loaded"` lines** from one invocation. Harmless correctness-wise (every racing thread reads the identical cookie), but wasteful (needless extra `Session` objects splitting the connection pool) and confusing in logs. Fixed with double-checked locking (`threading.Lock`, fast path costs nothing after the first call). Test deliberately slows session construction to force the race open — verified it **fails reliably against the unlocked code** (16/16 racing in a standalone reproduction) and **passes reliably against the fix**, not passing-by-scheduling-luck either way.
  2. **No progress visibility during a chunk's download phase.** A 2000-file chunk at real PhysioNet/shared-uplink throughput can take well over an hour with zero log output in between (the per-chunk summary line only prints after the *entire* chunk — download **and** resize — completes), making "slow but working" indistinguishable from "hung" without shelling out to `du` on the staging directory. Added a progress line roughly every 10% of a chunk (`"[fetch]   downloading N/2000 in this chunk (M ok so far)"`).
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
- [x] **8I** (2026-08-27) — **`fetch` essentially complete: 217,999 / 218,131 (99.94%).** Job 2482266 (latest resubmit after 2457894→2461245 hit the `--time` cap with no `--requeue`) started with 157,597 already done / 60,534 remaining, converted 60,402 of those (`"converted so far: 60402"` is per-invocation, not cumulative — verified by the arithmetic). ~132 images dropped to retryable network blips (`NameResolutionError`/`502` in the log) after exhausting per-chunk retries. Next: one more `STAGE=fetch` resubmit to mop up the stragglers (should finish in minutes), then `STAGE=pack EXCLUDE_HASHES=legacy_gallery_hashes.txt` — not yet run for the full/non-arm0 build — to produce the real `train/validate/test.parquet` at `cxr_mimic_full.yaml`'s `local_parquet_dir`.

### Phase 9 — Retrieval on full data (supporting chapter, extended) ⏳ blocked on Phase 8
Single-lever attribution preserved. **Arm 0 runs first and is non-negotiable** — it is 3.5 GPU-h against a confounded multi-day result.
- [x] **9A-prep** — **Early-validation tooling built (2026-08-17), unblocking Arm 0 from waiting on the full ~10-day `fetch`.** The gap: `fetch` processes all 218,131 images in plain sequential chunk order, scattering the specific ~27,570 legacy-training studies across nearly all ~110 chunks — meaning they would not all be present until the run was nearly done, defeating the point of a cheap early sanity check. Fixed by adding a way to fetch *just* that subset, separately and in isolation:
      - `build_mimic_cxr_local.py`'s `fetch` stage gained `--study-hashes <file>` (env: `STUDY_HASHES`): filters the manifest to only rows whose `report_hash` matches the given hash list, before computing `todo` — reuses every existing chunking/resume/atomic-write/`SESSION_EXPIRED` mechanism unchanged, just a filter on top. Tested (mocked, no network): filter correctly restricts to matching rows only, correctly ignores non-matching hashes without erroring.
      - `scripts/dump_legacy_hashes.sh` (NEW, SLURM-wrapped): dumps hashes for **both** the legacy `train[90%:]` test gallery (already needed for the 8D leakage guard) **and** the legacy `train[:90%]` training set (`legacy_training_hashes.txt`, ~27,570 rows) — `dump_legacy_gallery_hashes.py` already generalised via `--split`, so no code change was needed there, just running it twice. Runs offline against the already-populated HF cache.
      ⚠️ **CORRECTION (found live, 2026-08-17): a separate `OUT` directory does NOT isolate this the way it sounds like it should.** `manifest.parquet`'s `local_jpg` column is an absolute path baked in **at manifest-generation time** (`str(out / "files" / ...)`), not recomputed from whatever `--out` a later `fetch`/`pack` call is pointed at. Copying `manifest.parquet` into a fresh directory does **not** relocate those paths — a `fetch` run there silently reads/writes into the **original** `out`'s `files/` tree instead (confirmed live: an arm0-subset fetch pointed at a fresh `mimic_arm0_subset` directory showed rows already marked "done" at startup — real overlap with what the concurrently-running main `fetch` had already written into `mimic_full/files/...`). This is harmless, not data loss — both jobs converge on the one correct, shared `mimic_full/files/...` tree — but it means the "separate directory" plan doesn't work as an isolation mechanism, only as a filter.
      **Corrected approach: use the SAME `--out` as the main build for both `fetch` and `pack`.** `STUDY_HASHES` on `fetch` prioritises which images get downloaded first within the one shared tree (front-loading the Arm-0-relevant subset). `pack` gained the same `--study-hashes` filter **plus `--out-prefix`** (e.g. `arm0_`) so its output (`arm0_train.parquet` etc.) doesn't clobber the eventual production `train.parquet`/`validate.parquet`/`test.parquet` in that same directory. `pack` already only includes rows whose `local_jpg` exists, so this works correctly even on a **partially**-downloaded subset — no need to wait for 100% completion before packing a usable (if smaller) Arm-0 set.
      **Sequence**: `sbatch scripts/dump_legacy_hashes.sh` → `STAGE=fetch OUT=.../mimic_full STUDY_HASHES=legacy_training_hashes.txt sbatch scripts/build_mimic_cxr_local.sh` (small subset, roughly 1/8th–1/10th the wall time of the full fetch) → `STAGE=pack OUT=.../mimic_full STUDY_HASHES=legacy_training_hashes.txt OUT_PREFIX=arm0_ sbatch scripts/build_mimic_cxr_local.sh` → run the D1c recipe against `arm0_train.parquet`/`arm0_validate.parquet`/`arm0_test.parquet` via the existing `DATASET_CONFIG=cxr_mimic_full` lever (point `local_parquet_dir`'s file lookup at the `arm0_`-prefixed files, or override the config's split filenames).
      ⚠️ **Expect under-matching**, same caveat as the 8D leakage guard: the legacy mirror's report text won't hash-match 1:1 against this pipeline's (official, vendored) parser output, so the resulting subset will likely be smaller than the full 27,570 — report the actual matched count, don't assume exact reproduction of the original set membership. **Observed live**: 20,482 of 26,968 unique requested hashes matched (~76%; the requested count itself is lower than the raw 27,570-line file because ~602 lines are duplicate report hashes — templated reports, consistent with the ~2% duplication rate 6C-3 already measured). Still a substantially-overlapping, statistically meaningful reproduction check, not a bit-exact one.
      ⚠️ **Do not run the Arm-0 subset fetch concurrently with the main fetch at full worker counts.** Observed live: with both jobs scheduled onto the same compute node simultaneously (16 combined concurrent connections to physionet.org from one IP), a wall of `ConnectTimeoutError`s appeared — very likely tripping a connection limit somewhere in the path (PhysioNet-side or the shared campus uplink; exactly the failure class the cluster's own "flooding the network" warning describes). Not data-damaging (failed URLs are simply retried on the next resubmit — the resumable design absorbs this correctly), but it directly informs the still-open `WORKERS` question from 8B/the helpdesk email: this is evidence **against** raising concurrency further, not for it.
- [x] **9A-smoke (150M)** — **Pipeline health smoke test on the Arm-0 subset (job 2471261, 2026-08-20), NOT the 9A gate itself.** Ran the D1c recipe (`vit_unfreeze=12`, `SELECTION_SPLIT=true`) against `cxr_mimic_arm0` (16,899 train / 994 select / 226 test pairs) on the **150M** backbone — the backbone this project is actually carrying forward, per the pivot to report generation. Completed cleanly end-to-end: 6000 steps / 22 epochs, no crashes, checkpointing intact. In-training i2t R@10 on the 994-pair selection split climbed from ~0.03 to a peak of **0.227** (epoch 17), tracking the expected freeze→unfreeze inflection at step 1000 exactly (backbone unfreezes, retrieval jumps from 0.055→0.121 the same epoch); `train/cos_text_teacher` settled at 0.516–0.517, in the same equilibrium band the original Phase-6 runs found (~0.57). No flatlining, no divergence, no near-chance plateau — the qualitative signature of a healthy, non-corrupted dataset (misaligned image/report pairs or a broken section parser would show as retrieval stuck near random throughout).
      ⚠️ **This is NOT a 0.1459 reproduction and cannot be gated against it.** Two axes differ simultaneously from the number the gate is defined against: (1) **backbone** — 0.1459/0.1714 are 70M-v2 numbers; 150M at `vit_unfreeze=12` was never run historically (Phase 6B only tested 150M at the canonical `vit_unfreeze=2`, where it plateaued ~0.116, "capacity not binding" — not comparable to a depth-12 arm), so there is no valid 150M reference point at all; (2) **data volume** — this run used the arm0 subset (76% hash-matched, 16,899 train pairs) vs the legacy 26,038 train pairs 0.1459 was measured on. Also note: 0.227 is the **in-training** metric on the 994-pair **selection** split, not an authoritative eval on a held-out test set with best-by-`val/total_loss` checkpoint selection (the 6F/6G honest protocol) — getting an authoritative number for *this* run specifically would require running `evaluate_cxr_retrieval.py` against the saved checkpoints and `arm0_test.parquet` (226 pairs), which was not done and is not the priority given the above caveats.
      **Net effect**: strong qualitative de-risking of the Phase 8 PhysioNet pipeline (images decode, reports parse, image-report pairs are correctly aligned, training dynamics are sane) on the backbone actually going forward — but the literal 9A gate (below) remains open and untouched by this run.
- [ ] **9A** — **Arm 0 — reproduction control.** Rebuild ONLY the same ~27.5k studies through the new pipeline (now runnable early via 9A-prep, not gated on the full `fetch`), rerun the D1c recipe. **Gate: reproduce 0.1459 ± 1.1pp.** If it misses, the pipeline changed something (section parser, view selection, the extra resample) and that is found for 3.5 GPU-h instead of inside a confounded result. Requires the 70M-v2 Stage-0 backbone (`h100_stage0_hybrid_70m_v2/checkpoints/stage0_model_only.pt`), which is currently MISSING from this checkout (searched 2026-08-19: only the 150M Stage-0 checkpoint and no 70M `last.ckpt` were found) — recover it or accept 9A-smoke above as the substitute given the pivot to 150M/generation.
      ⚠️ **PROTOCOL TRAP — do not compare a full-MIMIC number on the official split against 0.1459 on the legacy gallery.** The official split is subject-disjoint *by construction*; the legacy `train[90%:]` gallery has **unknown provenance** and may itself leak subjects across its own 90/10 cut. The new protocol is therefore **harder**, and the number can go **down while the model gets better**. Arm 0 exists precisely to separate "the pipeline changed" from "the protocol got harder" from "the model changed". Never quote a cross-protocol delta.
- [ ] **9B** — **Arm 1 — data only.** Full build, D1c recipe unchanged, same 6,000 steps. Note the epoch budget moves 13.93 → ~1.8 at identical GPU-hours: same compute, far less repetition.
- [ ] **9C** — **Arm 2 — the actual hypothesis.** `vit_lr` ∈ {1e-6, 3e-6, 1e-5, 3e-5} on the full set. **Pre-registered prediction: the inverted-U optimum shifts right of 1e-6 and the peak is higher.**
      ⚠️ **State the hypothesis narrowly.** This plan already established that *optimization-side* overfitting is NOT binding (epochs 23→14 = +0.06pp; lower LR removed the rollover without changing plateau height). What 6G-1 measured is that the **image tower specifically** memorises (`train/clip_loss` 1.17→0.04 while val rises). So the claim under test is *"more data relaxes the image-adaptation dose constraint"* — **not** "more data fixes overfitting". If the inverted-U does not move, that is a real and reportable result either way.
- [x] **9D-infra** (2026-08-24) — **Image augmentation lever built, applied to 10E first, retrieval still untouched. Experiment (does it actually fix the memorization below) NOT yet run.** (2026-08-24) `train_contrastive.py`'s `build_image_transform(cfg, is_train)` adds `RandomResizedCrop(scale=(0.8,1.0))` + mild rotation, gated behind `cfg.dataset.use_augmentation` (default **false**, declared in both `cxr_mimic_arm0.yaml`/`cxr_mimic_full.yaml`) AND `is_train` (never on val/test). Built after Phase 10E's arm0 checkpoint (job 2478647) was confirmed via `evaluate_report_generation.py --checkpoint` to have **memorized boilerplate templates** — identical generated text across 3 different held-out studies — exactly the mechanism 6G-1 measured. **Retrieval's own arm (this item's original scope) is still open/deprioritized**; `use_augmentation` defaults off everywhere so that closed chapter stays byte-identical. Next: rerun the arm0 report-gen training with `AUGMENT=true` to confirm it fixes the memorization pattern before committing the full-data run to it (user's explicit sequencing choice, 2026-08-24).
- [ ] **9E** — **Free methodological result:** rerun the 6G-7 2×2 (`{vit=2, vit=12} × {val==test, clean}`) on full data. The measured finding was that test-informed selection is worth 2.2pp at 85M trainable image params **because the model overfits**; with ~8× data that number should **shrink**. Measuring how selection-protocol advantage scales with data volume is a clean, rarely-published contribution and costs nothing extra.
- [ ] **9F** — Indiana + official-split evals. Gate: Indiana i2t R@10 ≥ 4.04% floor (target 5.5%) with MIMIC held ≥ Phase-6G.
- [ ] **9G** — *(optional, was Phase 7)* multi-source CheXpert/PadChest/VinDr via `cxr_multi.yaml` + `CXRJointDataset` text adapter. **Deprioritised**: full MIMIC is ~8× the data for a fraction of the engineering, and PhysioNet credentialing also unlocks **VinDr-CXR** directly. Revisit only if Indiana is still flat after 9F. **IU-Xray stays EXCLUDED from training (= Indiana eval; zero-leakage).**

### Phase 10 — Image-conditioned report generator 🎯 NEW — the pivot's core work
**Verified gap:** `HybridLanguageModel.forward()` (`hybrid_lm.py:147`) accepts **only `input_ids`** — no `inputs_embeds`, no `encoder_hidden_states`, no cross-attention. `generate()` (`:230`) likewise. **There is currently no way to condition the decoder on an image.** This is the single largest piece of new work in the pivot and the proposal under review does not mention it.
- [x] **10A** (2026-08-20) — `forward(input_ids=None, inputs_embeds=None, ...)` + `generate(..., prefix_embeds=None)` added to `HybridLanguageModel` (`hybrid_lm.py`). Additive, default-`None` — every existing Stage-0/contrastive checkpoint and call site (21 sites audited) untouched. `forward()` raises `ValueError` unless exactly one of `input_ids`/`inputs_embeds` given. `generate()`'s `prefix_embeds=None` path is byte-for-byte the pre-existing code (separate branch, not a refactor of it); the new branch builds `hidden_states = cat([prefix_embeds, embeddings(input_ids)])` and loops via `forward(inputs_embeds=...)`, re-embedding each sampled token since there is no KV/state cache. Pinned by 5 new parity tests incl. `test_forward_inputs_embeds_matches_token_embedding_path` (bit-identical to token-embedding path) and `test_generate_default_path_unchanged_when_prefix_embeds_none` (bit-identical to pre-10A `generate()`). `bash scripts/validate_for_willi.sh`: 115 passed (was 110), 9/9 gates green.
- [x] **10B-architecture** (2026-08-20) — `hybrid_xmamba/models/prefix_mapper.py`: `ImagePrefixMapper(patch_dim, decoder_dim, k, dropout=0.1)`, `Linear → adaptive_avg_pool1d(N→k) → GELU → Dropout → Linear`, deliberately attention-free (mirrors the prefix-over-cross-attention rationale below — no new Triton machinery even in the connector). Unit-tested with synthetic `(B,197,768)` tensors for `k ∈ {8,32,64}`, gradient flow confirmed. **Training (the `k`-sweep, BiomedCLIP patch-grid wiring via `image_encoder.trunk.forward_features(...)`, actual report data) is NOT done — that's the rest of 10B below plus 10C–10E, blocked on real data/GPU.**
- [ ] **10B (training)** — project the BiomedCLIP ViT patch grid (197×768) through `ImagePrefixMapper` to `k` prefix tokens in the decoder's `dim=768` space, prepend, train with cross-entropy on the report. Rationale: Mamba/mLSTM are recurrent — a prefix is absorbed into the state and needs **zero** changes to the SSM/TFLA kernels, whereas cross-attention would mean new per-layer modules and new Triton work. **Sweep `k` ∈ {8, 32, 64} — this is the depth-analogue lever and the most likely place a real effect lives.**
- [ ] **10C** — Initialize the image tower from the **Phase-9 best contrastive checkpoint**, not stock BiomedCLIP. This is exactly where the retrieval chapter pays for itself, and 6C measured the size of that dividend: fine-tuned ViT 0.1172 vs stock 0.0232 with the same text tower. **Run stock-ViT as an ablation arm** — the delta is a headline result linking the two chapters.
- [ ] **10D** — Decoder init from the Phase-5 Stage-0 backbone (val PPL 13.18). Train on `findings`; decide findings-only vs findings+impression per 8F and record it.
- [x] **10E-infra** (2026-08-20) — `configs/model/hybrid_150m_v2_rrg.yaml` (identical architecture to `hybrid_150m_v2.yaml` + `image_patch_dim`/`prefix_k`/`decoder_lr`/`head_lr` keys), new `ReportGenerationLightningModule` (`hybrid_xmamba/training/lightning_module.py`) wiring decoder + `ImagePrefixMapper` + optional BiomedCLIP `.trunk.forward_features(...)` (the pre-pooling patch-grid path — did not exist anywhere in the codebase before this; every existing CLIP call site used the pooled 512-d output), `scripts/train_report_generation.py` (reuses `train_contrastive.py`'s `load_mimic_cxr`/`ImageTextDataset` directly rather than duplicating), `scripts/train_report_generation_h100.sh` (mirrors `train_biomedclip_kd_h100.sh`'s `--exclude=ga03,...` / env-lever / offline-HF conventions, fails fast if `DECODER_CKPT` is missing). Loss: causal CE over report-token positions only, prefix + pad positions label-masked with `-100` (verified this is honored automatically — `hybrid_lm.py`'s `nn.CrossEntropyLoss()` takes no `ignore_index` arg, so it's relying on PyTorch's `-100` default; pinned by a dedicated regression test). 4 new parity tests (CPU finite-loss+gradient-flow step, the `-100`-masking regression pin, config/param-count check, SLURM-wrapper-conventions check). `validate_for_willi.sh`: 125 passed (was 123), 9/9 gates green; Hydra compose of `model=hybrid_150m_v2_rrg dataset=cxr_mimic_full` verified directly; the training script's imports verified directly (cannot run training itself — no data, no decoder checkpoint). **Still infrastructure, not a runnable recipe** — genuinely blocked on Phase 8 finishing + a Phase 10D decoder checkpoint, per the file's own header docstring.
- [x] **10E-smoke** (2026-08-23) — First-ever live execution of the Phase 10E pipeline (job 2478641, gx10, 50 steps / 17s wall), against the already-fully-fetched Arm-0 subset (`mimic_full/arm0_{train,validate,test}.parquet`, 19881/161/226 pairs) rather than waiting on the still-running Phase 8 full fetch. Found and fixed two Hydra strict-struct bugs surfaced only by an actual `sbatch` submission (neither caught by 10E-infra's own compose check, which never exercised the SLURM wrapper's exact CLI-override list): `model.vit_unfreeze_blocks`/`model.vit_lr` undeclared in `hybrid_150m_v2_rrg.yaml` (job 2478622), then `decoder_checkpoint` undeclared in `configs/config.yaml` — the base config `train_report_generation.py` actually composes against, distinct from `config_70m.yaml` which is why `train_contrastive.py`'s equivalent `lm_checkpoint` never hit this. Both fixed; added `test_train_report_generation_h100_slurm_wrapper_hydra_overrides_compose`, which replays the wrapper's literal python invocation through `hydra.compose()` so this class of bug is caught offline going forward. Smoke run itself: decoder checkpoint loaded 0 missing/0 unexpected keys, BiomedCLIP tower loaded frozen, dataloaders built correctly, `train/lm_loss` finite (5.776) with no crash — validates plumbing, not model quality (only 50 of 500 `warmup_steps`). `validate_for_willi.sh`: 126 passed (was 125), 9/9 gates green throughout. **Distinct from and NOT satisfying 10B(training)/10C/10D/10F** — this is a pipeline-health checkpoint (same status as 9A-smoke was for retrieval), not a real training run; the real arm0 run (10k steps, `EXPERIMENT=h100_report_gen_arm0`) is the natural next step, still on the frozen-ViT/random-schedule 10C/10D defaults since neither of those phases has run yet.
- [ ] **10F** — Numerical gates: finite fwd/bwd, grad-norm bounded, no NaN over 50+ steps. ⚠️ **Stage-0 taught that 150M is spike-fragile** — the fp32 mLSTM-gate / Mamba-scan fix (2026-07-16) is load-bearing; keep `gradient_clip_val=0.5` and monitor `grad_norm` from step 0.
- [ ] **10G** — *(optional arm, replaces "re-run Stage 0")* **Domain-adaptive continuation of the LM on MIMIC report text.** ~190k reports × ~250 tok ≈ **47M tokens** (vs Stage-0's 483M PubMed) — roughly one epoch, **hours not days**. Radiology report style is very distinct from abstract prose, and unlike the retrieval case the backbone here is used as a **generator**, so the Stage-0-quality null does not automatically apply. **Run as an ablation with the untouched 13.18 checkpoint as control** — never as the default. Blocked on Phase 8 (no report corpus until then).

### Phase 11 — Report-generation evaluation 🎯 NEW
- [x] **11A** (2026-08-20) — `scripts/evaluate_report_generation.py` built: **ROUGE-L** (primary, pure-Python LCS-based F), BLEU-1/4 (pure-Python, brevity-penalty corpus BLEU), METEOR (nltk-backed, gracefully reports "skipped" rather than crashing when the wordnet corpus isn't staged locally — was assumed blocked on "compute nodes have no internet," CONFIRMED WRONG 2026-08-24 via sc-helpdesk@hpi.de: "Compute nodes have a 1 Gbit/s internet uplink" — same correction as line 457's earlier one, just not yet propagated here; `nltk.download('wordnet')` should now work directly from a compute job). Fixed decoding config: `greedy_decode()` reuses Phase 10A's `generate(prefix_embeds=..., top_k=1)`; `beam_search_decode(beam_size=3)` is new code built directly on `forward(inputs_embeds=...)` (no beam mode exists in `generate()`). `--smoke-test` CLI mode proves both decoders run end-to-end on a tiny random model + `ImagePrefixMapper` output — no checkpoint or MIMIC data needed. `--hyp-file/--ref-file` CLI mode computes metrics over any precomputed pairs (ready for 11C's retrieval-NN baseline once built). 6 new parity tests incl. a `beam_size=1 == greedy_decode` correctness invariant. `validate_for_willi.sh`: 121 passed (was 115), 9/9 gates green. **Still open:** loading a real checkpoint + running generation over the official MIMIC test split is 11D scope, gated on Phase 8 data + a trained Phase 10 generator.
- [x] **11B-infra** (2026-08-28, API fixed 2026-08-29, package installed 2026-08-29, isolated-venv architecture built 2026-08-29, exercised live + 4th package bug fixed 2026-08-30, venv-rebuild tooling fixed 2026-08-30, REAL NUMBERS PRODUCED 2026-08-30 job 2494784) — **CheXbert F1 harness built and API-correct; f1chexbert package installed; FIVE bugs in the 2023-era package/tooling (not this repo's code) found and worked around; real CheXbert F1 numbers now produced at n=1433 (see 11B checkbox below and the banner above for the numbers).** `scripts/evaluate_report_generation.py` gained an opt-in `--chexbert` flag (usable with `--checkpoint`, `--retrieval-baseline`, or `--hyp-file`/`--ref-file`): `compute_chexbert_metrics(hyps, refs)` wraps `F1CheXbert()(hyps=hyps, refs=refs)`, which returns `(accuracy, accuracy_per_sample, chexbert_all, chexbert_5)` — `chexbert_all`/`chexbert_5` are sklearn `classification_report(output_dict=True)`-style dicts (per-label + `"micro avg"`/`"macro avg"` entries) over the 14 CheXpert labels and the standard 5-label RRG subset (Cardiomegaly/Edema/Consolidation/Atelectasis/Pleural Effusion) respectively — f1chexbert computes labeling AND scoring end-to-end internally, so no separate label-matrix math is needed on this side. **2026-08-28 first attempt used a GUESSED, WRONG API** (`F1CheXbert(device=...)` + an imagined per-report `.get_label()` method, plus a hand-rolled `binary_f1_from_labels`/`load_chexpert_ground_truth_labels` pipeline for a planned ground-truth-CSV cross-check) — live run (job 2492037) correctly degraded to `CheXbert labeling skipped: ModuleNotFoundError: No module named 'f1chexbert'` (proving the graceful-skip contract works) rather than crashing, but would have called the wrong API once the package was actually installed. **2026-08-29: fetched the real PyPI README + vilmedic's own usage of it and rewrote against the confirmed signature.** One casualty: the public API exposes no per-report label vector, so the originally-planned ground-truth-CSV cross-check (comparing our own labeling of the reference text against `mimic-cxr-2.0.0-chexpert.csv.gz`) is **not implementable** against this package and has been dropped, not deferred — `--chexpert-csv` and the CSV-loading/label-math functions were removed along with their tests. Any f1chexbert import/download/API failure still degrades to a printed "skipped" message and `None` (same contract as `meteor_score_corpus`'s nltk/wordnet guard), never crashes the surrounding ROUGE/BLEU. `inspect_report_generation_h100.sh` and `retrieval_baseline_h100.sh` both carry a `CHEXBERT=true` lever (default off, `set -u`-safe empty-array expansion). validate_for_willi.sh: 142 passed, 9/9 gates green. `f1chexbert` remains in `requirements.txt` as a guarded optional dep. **2026-08-29 (later): package installed, real blocker found and confirmed.** Bare `pip install f1chexbert` on the login node hit a cluster-wide custom block ("disabled for users on our cluster"); `python -m pip install` hit a separate generic "not allowed on the login node!" guard; `srun --partition=aisc-batch --pty ...` was rejected outright (aisc-batch is batch-only, no interactive srun); submitting the pip install as its own `sbatch --wrap="...pip install..."` job hit the SAME pip block *inside* the batch job (confirming it's cluster-wide, not login-node-specific); `python -m pip install` inside that job got past the pip block but failed with `No module named pip` (this venv genuinely has no pip module — matches the cluster's own docs mandating conda, with pip tolerated only when a package isn't on conda-forge, which f1chexbert confirmed isn't). **Fix:** `sbatch --wrap="source .venv/bin/activate && uv pip install f1chexbert"` (job 2493754) — `uv` is the same tool `setup_env_h100.sh` already uses to build this venv precisely because bare pip is blocked; installed cleanly (f1chexbert==0.0.2 + appdirs/joblib/narwhals/scikit-learn/threadpoolctl). Rerunning `CHEXBERT=true sbatch scripts/inspect_report_generation_h100.sh` at the full `NUM_SAMPLES=1433` (job 2493764) reproduced ROUGE-L/BLEU bit-for-bit against the earlier runs (no regression), but CheXbert still skipped, now with `OSError: We couldn't connect to 'https://huggingface.co' ... and couldn't find them in the cached files`. **This resolves the previous "open unknown" about HF_HUB_OFFLINE:** both SLURM wrappers export `HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"` by default (correct for the report-gen model, which needs no network) and this also blocks `f1chexbert`'s own first-run download of its underlying HF-hosted checkpoint — nothing is cached yet on any node, and compute nodes do have real internet (proven by the `uv install` succeeding via sbatch minutes earlier), so this is purely an env-var default, not a network wall. **NEXT ACTION FOR THE USER:** override the flag for one run so the download can happen and get cached — cheap test first: `CHECKPOINT=./outputs/h100_report_gen_full/checkpoints/last.ckpt PARQUET=/sc/home/$USER/dataset/mimic_full/validate.parquet NUM_SAMPLES=10 CHEXBERT=true HF_HUB_OFFLINE=0 sbatch scripts/inspect_report_generation_h100.sh`; once that produces real CheXbert F1 numbers (not another skip), rerun at the full `NUM_SAMPLES=1433` the same way. No code change needed — this is a run-config fix only.

**2026-08-29 (later still): two more bugs found in f1chexbert itself, isolated-venv architecture built.** `HF_HUB_OFFLINE=0` fixed the huggingface.co error as predicted (job 2493901 got past it), but hit `FileNotFoundError: ~/.cache/chexbert/chexbert.pth`. Pulled f1chexbert's actual source (the sdist tarball from PyPI — no GitHub repo exists for this package) and found the real bug: its `download_model()` helper calls `hf_hub_download(..., force_filename=f)` inside a bare `except Exception as e: print(e)` that **swallows the failure** — `force_filename` was removed from modern `huggingface_hub`, so the download silently fails and `chexbert.pth` is never written, later surfacing as `torch.load()`'s `FileNotFoundError`. Fixed with a one-off manual download+copy to the exact expected cache path (`sbatch --wrap="... hf_hub_download(repo_id='StanfordAIMI/RRG_scorers', filename='chexbert.pth') ..."`, job 2493899 after job 2493894 failed by landing on ARM node `ga03` with no `--exclude`). The next rerun got past that too, but hit a **third** bug: `AttributeError: BertTokenizer has no attribute encode_plus` — that tokenizer method was removed in `transformers>=5.0`, and `f1chexbert==0.0.2` (2023-era, floor-pinned `transformers>=4.23.1` with no ceiling) is fundamentally incompatible with the modern `transformers` `uv pip install f1chexbert` pulled in for this repo's main pipeline. **Decision (user's choice, offered as an explicit tradeoff):** rather than downgrade the shared main `.venv`'s `transformers` — load-bearing for the GPT-2 tokenizer, BiomedCLIP text tower, and BioMedLM teacher elsewhere in this repo — build a fully **isolated venv** dedicated to CheXbert scoring, decoupled from the main pipeline via plain `hyps.txt`/`refs.txt` files. **Built:** `write_hyps_refs()` (pure I/O; collapses embedded newlines in report text so `hyps.txt`/`refs.txt` stay one-report-per-line-aligned) + a new `--dump-dir` flag wired into both `--checkpoint` and `--retrieval-baseline` modes of `evaluate_report_generation.py`. New `scripts/score_chexbert_standalone.py`: imports nothing from this repo (no `hybrid_xmamba`/Triton), duplicates `compute_chexbert_metrics`'s ~15-line `f1chexbert` call and output-dict shape by hand (kept in sync manually — both wrap the same tiny, stable public API), with the `f1chexbert` import deferred inside `main()` so the file stays parseable/testable in an env that doesn't have it installed. New `scripts/setup_chexbert_venv_h100.sh` (mirrors `setup_env_h100.sh`'s uv-first pattern; builds `.venv_chexbert` with CPU-only torch + `transformers<5` + `f1chexbert`) and `scripts/score_chexbert_h100.sh` (runs the standalone scorer against a `DUMP_DIR`'s `hyps.txt`/`refs.txt` using that isolated venv; required-var guard `${DUMP_DIR:?...}` matches `build_mimic_cxr_local.sh`/`eval_h100.sh`'s existing convention). Both existing wrappers gained a `DUMP_DIR` env lever (default off) alongside `CHEXBERT`. 8 new tests (newline-sanitizing dump round-trip; `--dump-dir` flag presence; standalone script's parser importable without `f1chexbert`; its mismatched-line-count `SystemExit`; both wrappers' `DUMP_DIR` lever; the venv-setup wrapper's `transformers<5` pin; the scoring wrapper's required-`DUMP_DIR` guard). `validate_for_willi.sh`: 150 passed (was 142), 9/9 gates green. **NEXT ACTION FOR THE USER:** (1) `sbatch scripts/setup_chexbert_venv_h100.sh` (one-time); (2) rerun the n=1433 `--checkpoint` inspection with `DUMP_DIR=results/report_gen_full_n1433` (no need for `--chexbert` this time — the dump is separate from scoring) to produce `hyps.txt`/`refs.txt` from the already-working generation run; (3) `DUMP_DIR=results/report_gen_full_n1433 sbatch scripts/score_chexbert_h100.sh` — this is the real integration test for the whole isolated-venv design and should finally produce real CheXbert F1 numbers.

**2026-08-30: all 3 staged steps run — steps 1–2 succeeded, step 3 hit a fourth f1chexbert bug, now fixed.** Step 1 (`setup_chexbert_venv_h100.sh`, job 2493922) built `.venv_chexbert` cleanly (`torch==2.13.0+cpu`, `transformers==4.57.6`, `f1chexbert` import OK). Step 2 (n=1433 `--checkpoint` inspection with `DUMP_DIR=results/report_gen_full_n1433`, no `--chexbert`, job 2493923) succeeded and reproduced the exact same aggregate ROUGE-L/BLEU as every prior run of this n=1433 eval bit-for-bit (`rouge_l 0.20753696229134028`, `bleu_1 0.27059390591927956`, `bleu_4 0.0707177415733251`, `meteor null` — nltk absent, degrades gracefully as designed, `num_examples 1433`), and dumped 1433 hyp/ref pairs to `hyps.txt`/`refs.txt` — confirms `--dump-dir` works correctly end to end without disturbing the existing metrics path. Step 3 (`score_chexbert_h100.sh` against that dump, job 2494015) **failed** with `ValueError: too many values to unpack (expected 3)` at `f1chexbert.py:229` — `y_type, y_true, y_pred = _check_targets(refs_chexbert_5, hyps_chexbert_5)`, a 3-value unpack of the **private** `sklearn.metrics._classification._check_targets` API. **Root-caused (not guessed)** by fetching scikit-learn's own source directly from `raw.githubusercontent.com/scikit-learn/scikit-learn/<tag>/sklearn/metrics/_classification.py` across a sequence of git tags: `_check_targets(y_true, y_pred)` returned exactly `(y_type, y_true, y_pred)` through tag `1.7.2`, then scikit-learn `1.8.0` added a `sample_weight` parameter and return value, making it a 4-tuple. f1chexbert has no scikit-learn upper pin (the same missing-ceiling pattern as its missing `transformers<5` pin), so step 1's unpinned `uv pip install f1chexbert` silently pulled `scikit-learn>=1.8.0` into the isolated venv. **Fixed** (one-line version pin, not a code patch — same remedy class as the `transformers<5` pin): `scripts/setup_chexbert_venv_h100.sh` now installs `"scikit-learn<1.8"` explicitly (both `uv` and `pip`-fallback branches), and its import smoke test now prints the installed scikit-learn version too. New parity test `test_setup_chexbert_venv_h100_slurm_wrapper_pins_scikit_learn_below_1_8` (mirrors the existing `transformers<5` pin test). `validate_for_willi.sh`: 151 passed (was 150), 9/9 gates green. **NEXT ACTION FOR THE USER:** (1) `git pull`; (2) `sbatch scripts/setup_chexbert_venv_h100.sh` again to **rebuild** `.venv_chexbert` with the fixed pin (the existing venv from job 2493922 has the broken scikit-learn baked in and must be rebuilt, not reused); (3) `DUMP_DIR=results/report_gen_full_n1433 sbatch scripts/score_chexbert_h100.sh` again — `hyps.txt`/`refs.txt` from job 2493923 are already on disk, no need to rerun generation. This is the fourth attempt at the actual integration test and should finally produce real CheXbert F1 numbers.

**2026-08-30 (later): step (2) above failed — `setup_chexbert_venv_h100.sh` was not rerunnable in place.** `sbatch scripts/setup_chexbert_venv_h100.sh` (job 2494759) errored immediately: `error: Failed to create virtual environment / Caused by: A virtual environment already exists at: .venv_chexbert`. `uv venv` refuses to overwrite an existing venv directory without `--clear` (or `UV_VENV_CLEAR=1`) — job 2493922 had already created `.venv_chexbert` once, so any rerun of this script (not just this specific pin fix) would hit this, self-defeating the exact "find a bad pin, fix it, rebuild" workflow this script exists for. **Fixed:** added `--clear` to both the `uv venv` invocation and the `python -m venv` fallback branch. New parity test `test_setup_chexbert_venv_h100_slurm_wrapper_is_rerunnable_in_place`. `validate_for_willi.sh`: 152 passed (was 151), 9/9 gates green. **NEXT ACTION FOR THE USER:** `git pull`; `sbatch scripts/setup_chexbert_venv_h100.sh` again (will now actually clear+rebuild); then `DUMP_DIR=results/report_gen_full_n1433 sbatch scripts/score_chexbert_h100.sh`.

**2026-08-30 (later still): `--clear` itself unreliable on this cluster's NFS home filesystem.** The `--clear` fix (job 2494771) FAILED differently: `error: Failed to create virtual environment / Caused by: failed to remove directory .../.venv_chexbert/lib: Directory not empty (os error 39)`. `uv venv --clear`'s own internal directory-removal logic can't reliably wipe the large existing site-packages tree on this cluster's NFS-backed `/sc/home` filesystem. **Fixed** (more robust than trusting either tool's internal `--clear`): the script now does an explicit `rm -rf "${VENV_DIR}"` itself before calling `uv venv`/`python -m venv` with no `--clear` flag at all — matches this repo's existing explicit-cleanup convention (e.g. the `/tmp/mimic_smoke_test` `rm -rf` in the Phase 8 build notes). Updated (not added) the same parity test to assert the `rm -rf` pattern. `validate_for_willi.sh`: 152 passed (unchanged count), 9/9 gates green. **NEXT ACTION FOR THE USER:** `git pull`; `sbatch scripts/setup_chexbert_venv_h100.sh` again (third attempt at just rebuilding the venv); then `DUMP_DIR=results/report_gen_full_n1433 sbatch scripts/score_chexbert_h100.sh`. `hyps.txt`/`refs.txt` from job 2493923 remain valid throughout all of this venv-tooling back-and-forth — no regeneration needed.
- [x] **11B** (2026-08-30, job 2494784) — **REAL CHEXBERT F1 NUMBERS, n=1433 validate.parquet.** `DUMP_DIR=results/report_gen_full_n1433 sbatch scripts/score_chexbert_h100.sh` against `hyps.txt`/`refs.txt` from job 2493923: **CheXbert F1 (14-label) micro/macro = 0.3097/0.1548; CheXbert F1 (5-label) micro/macro = 0.3059/0.1983; exact-match label-set accuracy = 0.3531.** Clears the plan's pre-registered **Floor** tier (ROUGE-L≥0.15 and CheXbert-14-micro≥0.25, both cleared — 0.2075 and 0.3097 respectively), well below **Target** (≥0.22/≥0.40). Full per-label breakdown in `results/report_gen_full_n1433/chexbert_metrics.json`: precision is consistently much higher than recall (e.g. Cardiomegaly 0.45P/0.63R is the one exception with real recall; Atelectasis 0.41P/0.04R, Edema 0.40P/0.08R, Pleural Effusion 0.73P/0.15R are typical) and 4/14 labels are never predicted at all (Lung Lesion, Pneumonia, Pneumothorax, Pleural Other — all F1=0). This is consistent with 11C's finding that 73.6% of generations are boilerplate "No acute cardiopulmonary process"-style templates: the generator plays it safe and rarely asserts rare/serious findings, which caps recall on exactly the labels a real clinical system most needs to catch. **Two caveats before treating this as the final number:** (1) run on `validate.parquet`, not the official subject-disjoint test split (11D, still open); (2) the retrieval-NN baseline (11C) has no CheXbert F1 yet — the plan's own pre-registered rule treats the retrieval baseline's CheXbert F1 as the real floor for this metric, and that comparison hasn't been made. **NEXT ACTION:** score CheXbert on the retrieval-NN baseline too to complete the floor comparison; independently, 11D (official test split) and 11E (qualitative appendix) remain open.

**2026-08-30 (later): first retrieval-floor attempt (jobs 2494817/2495070) was INVALID — misled by this checkbox's own instruction.** Running `DUMP_DIR=results/retrieval_floor_n1433 sbatch scripts/retrieval_baseline_h100.sh` with no other overrides, exactly as instructed above, silently reproduced the **CLOSED arm0** retrieval-NN numbers (`rouge_l 0.3688`/`bleu_1 0.396`/`bleu_4 0.178`, n=10) instead of the intended full-data floor (`rouge_l 0.188`, n=1433, jobs 2491600/2491687). **Root cause:** `retrieval_baseline_h100.sh`'s `TRAIN_PARQUET`/`PARQUET` env-var defaults still pointed at `arm0/{train,validate}.parquet` — stale since before Phase 8's full-data pack landed, never updated when `h100_report_gen_full` (job 2491338) became the active target. The instruction above assumed those defaults already pointed at full data; they didn't. Investigating this surfaced the **same staleness bug** in `inspect_report_generation_h100.sh`'s `CHECKPOINT` (defaulted to `outputs/h100_report_gen_arm0/...`) and `PARQUET` defaults. **Fixed for real** (script defaults, not just a corrected instruction, since this exact mistake could recur on 11D/11E): both scripts' defaults now point at full data (`arm0` still usable via explicit override, since it's referenced/documented as historical, not deleted). 2 new parity tests (`test_inspect_report_generation_h100_slurm_wrapper_defaults_to_full_data_not_arm0`, `test_retrieval_baseline_h100_slurm_wrapper_defaults_to_full_data_not_arm0`). `validate_for_willi.sh`: 154 passed (was 152), 9/9 gates green. **NEXT ACTION FOR THE USER:** `git pull`; `DUMP_DIR=results/retrieval_floor_n1433 NUM_SAMPLES=1433 sbatch scripts/retrieval_baseline_h100.sh` (defaults now correctly point at full data, but `NUM_SAMPLES` must still be set explicitly — its own default stays `10` for cheap sanity checks) — overwrites the invalid n=10 arm0 dump already sitting in that directory; then `DUMP_DIR=results/retrieval_floor_n1433 sbatch scripts/score_chexbert_h100.sh` against the corrected dump. This is the real, valid full-data retrieval-floor CheXbert comparison.

**2026-08-30 (final, VALID): full-data retrieval-floor CheXbert F1 produced and the pre-registered floor comparison is COMPLETE (jobs 2495080/2495164).** `DUMP_DIR=results/retrieval_floor_n1433 NUM_SAMPLES=1433 sbatch scripts/retrieval_baseline_h100.sh` (job 2495080, real 191,462-image train gallery, `validate.parquet` query, n=1433) followed by `DUMP_DIR=results/retrieval_floor_n1433 sbatch scripts/score_chexbert_h100.sh` (job 2495164). **Confirmed valid**: aggregate ROUGE-L/BLEU (`rouge_l 0.18806468556187`, `bleu_1 0.2605135093647304`, `bleu_4 0.0465444119306529`) match jobs 2491600/2491687 to 10+ significant figures. **CheXbert F1 (14-label) micro/macro = 0.4145/0.3054; CheXbert F1 (5-label) micro/macro = 0.4624/0.4118; exact-match label-set accuracy = 0.3036.** Full per-label breakdown in `results/retrieval_floor_n1433/chexbert_metrics.json`. **The floor comparison is mixed, not a clean generator win:**

| metric | generator (11B) | retrieval floor | winner |
|---|---|---|---|
| ROUGE-L | 0.2075 | 0.1881 | generator (+10%) |
| BLEU-1 / BLEU-4 | 0.2706 / 0.0707 | 0.2605 / 0.0465 | generator |
| exact-match accuracy | 0.3531 | 0.3036 | generator (+16%) |
| CheXbert-14 micro F1 | 0.3097 | **0.4145** | retrieval (+34% rel.) |
| CheXbert-14 macro F1 | 0.1548 | **0.3054** | retrieval (+97% rel.) |
| CheXbert-5 micro F1 | 0.3059 | **0.4624** | retrieval (+51% rel.) |
| CheXbert-5 macro F1 | 0.1983 | **0.4118** | retrieval (+108% rel.) |

Notably, the retrieval floor's own CheXbert-14-micro (0.4145) would itself clear the plan's pre-registered **Target** tier (≥0.40) were it the generator's score — the retrieval baseline is a genuinely strong floor on this metric, not a weak strawman. **Interpretation:** consistent with 11B's own per-label finding (4/14 labels never predicted at all, heavy precision-over-recall skew, 73.6% boilerplate-template rate in 11C) — a real, coherent retrieved report from a visually similar patient asserts a fuller, more clinically-plausible set of findings than the generator's conservative, largely-templated output, even though the generator's raw text is closer to the true reference by n-gram overlap. **Per the plan's own pre-registered rule** ("a generator that does not beat its own retrieval baseline has not contributed anything"), **the generator has NOT cleared this bar on CheXbert F1** — only on ROUGE-L/BLEU/exact-match accuracy. This is the honest final floor comparison for the 12B writeup; do not re-run chasing a better number without an actual generator change (e.g. more training steps, an explicit anti-templating objective, or larger/less-templated training data). No code changed for this entry — results/docs only, `validate_for_willi.sh` not required.
- [x] **11C (arm0)** (2026-08-24) — **Retrieval-NN baseline BEATS the arm0 generator on every metric: rouge_l 0.369 vs 0.285(no-aug)/0.284(aug); bleu_1 0.396 vs 0.325/0.337; bleu_4 0.178 vs 0.121/0.103** (job 2483076, stock BiomedCLIP cosine similarity, full 19881-image arm0 gallery, `scripts/retrieval_baseline_h100.sh`). Per this item's own stated bar, the arm0-trained generator **has contributed nothing** — confirms the qualitative `--checkpoint`-mode finding numerically. Not run on full data yet (Phase 8 pending); rerun once a full-data generator checkpoint exists, since the conclusion may not transfer (more data could both improve the generator AND make the retrieval floor harder to beat, per the 11C rationale that templated text is easy for both).
- [x] **11C (full, n=10 preliminary)** (2026-08-28) — job 2491338 (full-data 10k-step run, 191,462 train pairs) generator vs job 2491505 retrieval-NN floor, both n=10: rouge_l 0.197 vs 0.224; bleu_1 0.241 vs 0.252; bleu_4 0.031 vs 0.055. **Generator still below the floor on all 3 metrics** — has not yet cleared the "contributed something" bar. BUT the floor itself dropped hard (0.369->0.224 rouge_l) on the properly deduplicated 191k gallery, and the qualitative `--checkpoint` output (job 2491499) is no longer byte-identical across patients like arm0's was — same ~3 template families reappear but with real per-sample variation, still hallucinating ungrounded content (e.g. an NG-tube template applied to patients whose reference never mentions one). Leading hypothesis: **undertraining, not memorization** — same 10k-step/batch-32 budget now spans only ~1.7 effective epochs of the 9.6x-larger train set (arm0 got ~16). n=10 is too small to trust on its own either way. Next: rerun both tools at `NUM_SAMPLES=200`; check train/val loss trajectory near step 10k for undertraining signal (more steps if still falling steeply).
- [x] **11C (full, n=1433)** (2026-08-28) — user ran both tools at full validate-split scale (`NUM_SAMPLES=1433`, i.e. the entire 1,433-row validate.parquet, not just 200): job 2491600 (`--checkpoint` inspection) vs job 2491687 (retrieval-NN floor, same 191,462-pair train gallery). **REVERSES the n=10 read — generator BEATS the floor on all 3 metrics**: rouge_l 0.2075 vs 0.1881; bleu_1 0.2706 vs 0.2605; bleu_4 0.0707 vs 0.0465. Per 11C's own stated bar, the full-data generator **has contributed something**. Caveat (checked the same way arm0's memorization was caught): of the 1433 generated reports, 1055 (73.6%) are exact-duplicate text shared across one of 184 template clusters (largest clusters: 56, 50, 34, 34, 27 identical generations across different studies/patients) — much less collapsed than arm0 (near-total duplication) but still substantial boilerplate-templating, so part of the win over the retrieval floor is plausibly "safe boilerplate scores okay against a templated reference corpus" rather than full per-patient grounding. Loss-trajectory undertraining check was inconclusive either way: only the last 3 checkpoints' `val_lm_loss` were available (step 9500=1.1095, 9750=1.1046, 10000=1.1007) — a slow, shallow decline at the tail, not a steep one, but too short a window (no full curve from step 0) to rule out undertraining conclusively. Source: `logs/log1.log` (user's pasted cluster session transcript covering jobs 2491577/2491600/2491687).
- [x] **11D** (2026-08-30, jobs 2495184/2495185 generation, 2495772/2495775 CheXbert) — **Official subject-disjoint test split, n=2663 (`test.parquet`), full 191,462-pair train gallery.** `PARQUET=.../test.parquet NUM_SAMPLES=999999 DUMP_DIR=results/report_gen_test_split sbatch scripts/inspect_report_generation_h100.sh` (generator) + same against `retrieval_baseline_h100.sh` (`results/retrieval_floor_test_split`), then `score_chexbert_h100.sh` against each dump. **Confirms the n=1433 validate-split mixed result generalizes to the official held-out test split — not a validate-split artifact:**

| metric | generator (11D) | retrieval floor (11D) | winner |
|---|---|---|---|
| ROUGE-L | 0.1816 | 0.1636 | generator (+11%) |
| BLEU-1 / BLEU-4 | 0.2289 / 0.0448 | 0.2372 / 0.0330 | split (floor wins BLEU-1, generator wins BLEU-4) |
| exact-match accuracy | 0.1926 | 0.1735 | generator (+11%) |
| CheXbert-14 micro/macro F1 | 0.3326 / 0.1671 | **0.4296 / 0.3014** | retrieval (+29%/+80% rel.) |
| CheXbert-5 micro/macro F1 | 0.3454 / 0.2321 | **0.4856 / 0.4284** | retrieval (+41%/+85% rel.) |

Absolute numbers on both arms are a bit lower than validate.parquet's (harder/larger n=2663 test split vs n=1433 validate), but the **shape of the result is identical**: generator wins ROUGE-L/exact-match, retrieval floor wins CheXbert F1 decisively on every sub-metric. Per the plan's own pre-registered rule, **the generator still has not cleared its own retrieval floor on CheXbert F1** — now confirmed on the official test split, closing the last open caveat from 11B/11C. Raw dumps/metrics in `results/report_gen_test_split/chexbert_metrics.json` and `results/retrieval_floor_test_split/chexbert_metrics.json`. Source: `logs/log1.log`.
- [ ] **11E** — Human-readable qualitative appendix: N=20 side-by-side generated/ground-truth pairs incl. failure cases. Cheap, and it is what a viva actually asks about.

### Phase 12 — Full eval + comparison + writeup
- [x] **12A** (2026-09-03, CLOSED) — **NARROWED, one measurement genuinely missing, one already satisfied.** Authoritative MIMIC + Indiana retrieval numbers exist (quoted in 12B's writeup from the closed retrieval chapter). **PubMed PPL is ALREADY MEASURED**: verified `scripts/train_stage0_h100.sh:78` uses `dataset=pubmed`, so the existing "Stage-0 val PPL 13.18" (Phase 5) *is* the PubMed PPL number — it was just never labeled as such. **STS (BIOSSES/STS-B) genuinely never measured** — `scripts/evaluate_sts.py` exists but was never wired into an H100 SLURM wrapper (only pre-H100 `eval_stage1_*.sh` scripts reference it). **Code shipped**: `eval_h100.sh` gained `MODE=sts` (dispatches to `evaluate_sts.py --datasets all`), with a MODE-conditional HF-offline default (sts defaults online — BIOSSES/STS-B are public HF datasets needing real network access, unlike the gated mimic-cxr repo `MODE=retrieval` deliberately stays offline for). 2 new parity tests (mode dispatch presence; a structural guard that the sts online-default can't silently become the global default for ppl/retrieval too). `validate_for_willi.sh`: 164 passed (was 162), 9/9 gates green.

  **IMPORTANT CAVEAT before running**: `evaluate_sts.py`'s `load_encoder()` requires `projection_head.*` keys in the checkpoint (raises `RuntimeError` otherwise) — a bare Stage-0 LM-only checkpoint (`stage0_model_only.pt`) does NOT have one; only a **joint-trained** checkpoint (`train_contrastive.py contrastive_mode=joint` output, e.g. any Phase 6/13C/13D tower checkpoint) does. Staged command, using the final Phase 13D tower as the most relevant "best 150M ckpt" in the project's current state:
  ```
  CKPT=./outputs/h100_kd_150m_v2_full_data_lr3e6/checkpoints/last.ckpt MODE=sts MODEL_CONFIG=hybrid_150m_v2 sbatch scripts/eval_h100.sh
  ```
  **FIRST LIVE RUN (2026-09-03, job 2505439) HIT REAL BUGS IN `evaluate_sts.py` ITSELF, NOW FIXED.** Checkpoint loading worked perfectly (184.7M params, correct architecture auto-detected, `projection_head.*` present — confirms the checkpoint choice above was right). Both dataset sources crashed:
  - **BIOSSES**: all 3 candidates (`bigbio/biosses`, `biosses`, `nguyenthanhdo/biosses`) failed — `"Dataset scripts are no longer supported, but found biosses.py"`. These rely on HF's legacy "dataset loading script" mechanism, which the `datasets` library has since dropped.
  - **STS-B**: `load_dataset("glue", "stsb", split="validation")` crashed the whole job (this call was UNGUARDED — no try/except, unlike BIOSSES's per-candidate loop) with an `hf_file_system.resolve_path` error trying to resolve `glue`'s legacy standalone YAML config.

  **Fixed** (confirmed via live HF dataset-page fetches, not guessed): both now try **`mteb/biosses-sts`** and **`mteb/stsbenchmark-sts`** first — script-free parquet mirrors with a `sentence1`/`sentence2`/`score` schema (BIOSSES: single `test` split, 100 rows, score 0-4; STS-B: train/validation/test splits, score 0-5). Old sources kept as fallbacks only. `_load_stsb` also gained the same per-candidate try/except loop BIOSSES already had, so a first-source failure degrades gracefully instead of crashing the process. 1 new parity test. `validate_for_willi.sh`: 165 passed (was 164), 9/9 gates green.

  **Re-run the same command** (checkpoint loading already confirmed working, no need to change it):
  ```
  CKPT=./outputs/h100_kd_150m_v2_full_data_lr3e6/checkpoints/last.ckpt MODE=sts MODEL_CONFIG=hybrid_150m_v2 sbatch scripts/eval_h100.sh
  ```
  **RESULT (2026-09-03, job 2505443, `13D`'s tower checkpoint) — 12A CLOSED.** BIOSSES Spearman ρ = **0.3829**; STS-B (validation) ρ = **0.4472**. MedSTS gracefully skipped (no working HF mirror exists for it — degraded as designed, not a bug). `evaluate_sts.py` prints `FAIL` against a decision gate (BIOSSES≥0.50, STS-B≥0.60), but **that gate is not meaningful for this checkpoint** — it was written for a dedicated sequential Stage-1 SimCSE-only checkpoint, a pipeline design this project never executed. The actual checkpoint is jointly trained, with SimCSE as one minor auxiliary loss (`gamma_simcse=0.1`) alongside the CLIP/KD objectives the `vit_lr` sweep actually optimized for — report these numbers as a baseline reference, not a failed target. Folded into `analysis/h100_scaling_results.md` §2 alongside the now-correctly-labeled PubMed PPL. **12A is fully closed.**
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
- [x] **12B** (2026-09-03) — `analysis/h100_scaling_results.md` written: (1) report generation [primary — full Phase 11→13 arc, final numbers vs the retrieval-NN floor on the official test split] (2) retrieval [supporting chapter, closed, ten-nulls-one-lever table] (3) efficiency [linear scaling, hybrid's training-memory advantage] (4) honest limitations [8 items, incl. the unverified boilerplate-rate claim and the never-run `3e-5` arm] (5) reproduction commands. Uses only numbers already verified live in this file/state file — no fabricated STS/PPL numbers (see 12A).
- [x] **12C** (2026-09-03) — `h100_scaling_state.json` final verdict recorded: `current_phase` and a dedicated `final_verdict` block with the best checkpoint path, headline numbers, and pointer to `analysis/h100_scaling_results.md`.

---

### Phase 13 — Closing the CheXbert F1 gap 🎯 NEW (2026-08-30)

**Why:** 11D confirmed on the official test split (n=2663) that the generator loses to a trivial retrieval-NN baseline on CheXbert F1 (14-micro 0.33 vs 0.43) despite beating it on ROUGE-L/BLEU/exact-match — the plan's own pre-registered rule says that means the generator "has not contributed anything" on the metric that matters clinically. User decision 2026-08-30: spend real compute closing this gap rather than accept it, milestone 1 = beat the retrieval floor's CheXbert-14-micro (0.4296, official test split), milestone 2 = hit the plan's Target tier (CheXbert-14-micro≥0.40, ROUGE-L≥0.22) — both, floor first.

**Two suspected root causes, diagnosed but never fixed:** (1) the frozen BiomedCLIP image tower conditioning the generator was only ever contrastively trained on the 27.5k-pair Arm-0 subset — Phase 9's full-data retrain (9B–9E) was explicitly deferred 2026-08-20 to move faster to Phase 10; (2) the decoder itself only got ~10k steps / ~1.7 effective epochs over the full 191,462-pair train set (vs ~16 epochs on the 9.6x-smaller arm0 set, which memorized instead of learning).

**Staged, single-lever-attribution order (cheapest/most-diagnostic first, per user decision):**

- [x] **13A** (2026-08-30, jobs 2495934/2496212) — **Decode-strategy diagnostic: beam beats greedy on every metric except exact-match accuracy, at zero training cost.** n=1433 validate.parquet, identical `h100_report_gen_full` checkpoint, `--decode beam --beam-size 3` vs the existing greedy baseline:

  | metric | greedy (11B) | beam (13A) | retrieval floor | beam vs floor |
  |---|---|---|---|---|
  | ROUGE-L | 0.2075 | **0.2138** | 0.1881 | beam wins (+13.7%) |
  | BLEU-1 / BLEU-4 | 0.2706 / 0.0707 | **0.2746 / 0.0708** | 0.2605 / 0.0465 | beam wins both |
  | exact-match accuracy | 0.3531 | 0.3329 (↓5.7%) | 0.3036 | beam still wins |
  | CheXbert-14 micro/macro | 0.3097 / 0.1548 | **0.3665 / 0.1876** (+18.3%/+21.2%) | 0.4145 / 0.3054 | floor wins, gap 34%→**13%** / 97%→**63%** |
  | CheXbert-5 micro/macro | 0.3059 / 0.1983 | **0.3636 / 0.2512** (+18.9%/+26.7%) | 0.4624 / 0.4118 | floor wins, gap 51%→**27%** / 108%→**64%** |

  **Real, free improvement — milestone 1 (beat the floor's CheXbert-14-micro) not yet met, but the gap roughly halved to a third across every CheXbert sub-metric with zero training.** Per-label: beam recovers real recall on *common* findings (Cardiomegaly, Atelectasis, Edema, No Finding, and recovers Pneumonia from a total greedy-zero) but flips Consolidation and Fracture from small-nonzero to exactly zero — still **5/14 labels at F1=0.0** (Lung Lesion, Consolidation, Pneumothorax, Pleural Other, Fracture — all low-support rare findings, 35-110 samples). Confirms 13F's rare-finding-recall problem is real and untouched by decode strategy alone. **Decision: `beam_size=3` is now the standard decode strategy for every future report-gen eval in this project** — greedy is strictly dominated here at zero extra cost. Full per-label breakdown in `results/report_gen_beam_n1433/chexbert_metrics.json`.

- [x] **13B** (2026-08-31, jobs 2496243 training / 2496656 eval) — **MILESTONE 1 CLEARED: the generator beats the retrieval floor on CheXbert-14-micro AND CheXbert-5-micro for the first time in this whole arc.** 4-GPU DDP, `MAX_STEPS=12000` → 8.02 effective epochs (up from 1.7), `train/lm_loss_epoch` 0.919 (down from ~1.12), `val_lm_loss` still slowly declining at the end (1.0514→1.0490→1.0485 across the last 3 checkpoints, no overfitting turn-up at this budget). Eval with beam decode (n=1433 validate.parquet, per 13A's standing decision):

  | metric | greedy (11B) | beam-only (13A) | **13B+beam** | retrieval floor | result |
  |---|---|---|---|---|---|
  | ROUGE-L | 0.2075 | 0.2138 | **0.2102** | 0.1881 | **generator wins** |
  | BLEU-1 / BLEU-4 | 0.2706/0.0707 | 0.2746/0.0708 | **0.2831/0.0710** | 0.2605/0.0465 | **generator wins both** |
  | exact-match accuracy | 0.3531 | 0.3329 | **0.3636** (best yet) | 0.3036 | **generator wins** |
  | CheXbert-14 micro/macro | 0.3097/0.1548 | 0.3665/0.1876 | **0.4382**/0.2631 | 0.4145/0.3054 | **micro: generator wins (+5.7%)** ✅ / macro: floor wins, gap 97%→**14%** |
  | CheXbert-5 micro/macro | 0.3059/0.1983 | 0.3636/0.2512 | **0.4930**/0.3956 | 0.4624/0.4118 | **micro: generator wins (+6.6%)** ✅ / macro: floor wins, gap 108%→**4%** |

  A clean sweep on every metric except the two macro-F1s, and both of those gaps collapsed to near-parity. Per-label: zero-F1 labels dropped from 5/14 (under greedy) to **2/14** (Lung Lesion, Pleural Other — Consolidation, Pneumonia, Pneumothorax, Fracture all recovered to real positive F1). This is broad improvement, not a one-label fluke. **Milestone 2's CheXbert bar (Target tier ≥0.40 14-micro) is ALSO cleared** (0.4382); ROUGE-L's Target bar (≥0.22) is close but not yet met (0.2102).

  **SLURM gotcha hit and fixed live:** the first submit (`sbatch --gpus=4 --gres=gpu:h100:4 ...`, this doc's own earlier recommendation) was **rejected** — `Invalid GRES specification (with and without type identification)`. Combining untyped `--gpus` with a typed `--gres` for the same resource is invalid on this cluster. Fixed to **`--gpus=4` alone** (commit `f982331`), matching every other `*_h100.sh` script's own convention in this repo — none of them actually use `--gres=gpu:h100:N` despite this doc's earlier, unverified assumption that they did.

  **CONFIRMED FINAL 2026-08-31 (jobs 2497023/2497334) — holds on the OFFICIAL test split (n=2663), the harder bar, not just validate.** Same 13B+beam checkpoint, `--decode beam --beam-size 3`:

  | metric | generator (13B+beam) | retrieval floor (11D) | result |
  |---|---|---|---|
  | ROUGE-L | **0.1852** | 0.1636 | generator wins |
  | BLEU-1 | 0.2347 | **0.2372** | floor wins narrowly (same pattern as 11D's greedy read on this split) |
  | BLEU-4 | **0.0513** | 0.0330 | generator wins |
  | exact-match accuracy | **0.2249** | 0.1735 | generator wins (+29.6%) |
  | CheXbert-14 micro | **0.4412** | 0.4296 | **generator wins (+2.7%)** ✅ MILESTONE 1 |
  | CheXbert-14 macro | 0.2487 | **0.3014** | floor wins (-17.5%) — the one real remaining gap |
  | CheXbert-5 micro | **0.5280** | 0.4856 | **generator wins (+8.7%)** ✅ |
  | CheXbert-5 macro | 0.4234 | 0.4284 | near-parity (-1.2%, was -108% under greedy) |

  **Milestone 1 is final, not a validate-split artifact.** Milestone 2's CheXbert Target bar (≥0.40 14-micro) is also cleared (0.4412); ROUGE-L's Target bar (≥0.22) remains short (0.1852). The one real remaining gap is CheXbert-14-macro, concentrated in **3/14 labels still at F1=0.0** (Lung Lesion, Pneumothorax, Pleural Other — 84-177 support out of 2663, all rare). **This is the headline result for the 12B writeup**: after the objective pivot to report generation, the image-conditioned decoder genuinely beats a retrieval-NN baseline on the primary clinical metric, confirmed on the official held-out subject-disjoint split — via two cheap/free levers (beam decode: zero cost; extended training: ~2h23m on 4 GPUs), no architecture change or new loss function needed.

  **DECISION POINT (put to the user 2026-08-31):** continue toward milestone 2 via 13C/13D (image tower retrain — general-quality lever, would help ROUGE-L/broad grounding) or 13F (rare-label oversampling/aux loss — now the more surgically-targeted lever, since the remaining gap is concentrated in exactly 3 never-predicted labels), or stop here and move to Phase 12's writeup with milestone 1 as the headline finding (the plan's own pre-registered rule — "a generator that does not beat its own retrieval baseline has not contributed anything" — is now satisfied).

  ~~Original 13B plan text, kept for the record:~~ Fix undertraining, with an optional multi-GPU speedup (code shipped 2026-08-30). Retrain the decoder from scratch (same `DECODER_CKPT` init, same stock image tower) with `MAX_STEPS` raised to target ~8 effective epochs (comfortably short of arm0's 16-epoch memorization point, well past today's 1.7) — single lever: step budget only.
  - Single-GPU (~9-10h): `MAX_STEPS=${epochs}*191462/32` → **48000** for 8 epochs. `MAX_STEPS=48000 EXPERIMENT=h100_report_gen_full_ext sbatch scripts/train_report_generation_h100.sh`
  - 4-GPU DDP (~2-2.5h, same 8-epoch target — **shipped this session**: `NUM_GPUS` lever added to `train_report_generation_h100.sh`, selects `trainer=h100_multi_ddp` and fails fast if fewer GPUs were actually allocated than requested): effective batch scales to `32×4=128`, so `MAX_STEPS=8*191462/128`≈**12000**. `NUM_GPUS=4 MAX_STEPS=12000 EXPERIMENT=h100_report_gen_full_ext_4gpu sbatch --gpus=4 scripts/train_report_generation_h100.sh` — **`--gpus` alone** (matching every other `*_h100.sh` script's own `#SBATCH --gpus=1` convention in this repo). Do NOT also pass `--gres=gpu:h100:4`: combining untyped `--gpus` with a typed `--gres` for the same resource is rejected by sbatch (`Invalid GRES specification (with and without type identification)`, hit live 2026-08-30 — an earlier version of this line wrongly recommended both flags). `NUM_GPUS` alone does not request GPUs from SLURM, it only picks the trainer config once GPUs exist — the `--gpus` CLI flag is what's load-bearing.
  - **DDP is a clean win here specifically because report-gen is a plain LM cross-entropy loss with no in-batch-negatives semantics** — unlike the contrastive/CLIP trainer (13C below), which needs the still-unbuilt Phase 3 `all_gather` to get anything beyond throughput out of extra GPUs (confirmed via code survey: zero `all_gather`/`torch.distributed` code exists anywhere in this repo, design-doc TODO only).
  - Eval (per 13A's decision, **use beam decode from the start**, not greedy): `DECODE=beam BEAM_SIZE=3 DUMP_DIR=results/report_gen_ext_n1433 CHECKPOINT=./outputs/<experiment>/checkpoints/last.ckpt NUM_SAMPLES=1433 sbatch scripts/inspect_report_generation_h100.sh` then score.

- [x] **13C** (2026-09-01, job 2500827, 3.5h wall — matched the estimate exactly) — **Image tower on full data, same recipe as arm0 (Phase 9B, unblocked). DONE: healthy training, confirms the predicted mechanism, but undertrained.** In-training i2t R@10 climbed to 0.285 (N=1433) at step 6000, still rising, not plateaued. `train/clip_loss_epoch=2.123` vs `val/clip_loss=2.183` — **close together**, unlike every small-data (27.5k-pair) `vit_unfreeze=12` run at the same `vit_lr=1e-6`, which overfit hard (train collapsing to 0.03–1.17 while val rose to 2.5–3.5, per 6G-1). This directly confirms 9C's pre-registered prediction: more data relaxes the overfitting constraint that pinned `vit_lr` at 1e-6 on the small set. Consequence: this tower is **undertrained, not optimized** — arm0's step count was deliberately reused to isolate "does a MIMIC-tuned tower help at all" as a single cheap test, not to find the true optimum. Checkpoint: `outputs/h100_kd_150m_v2_full_data/checkpoints/last.ckpt` (correct to use as-is — no divergence occurred, so final==best). **NEXT: 13E** plugs this into the decoder; if it beats 13B, 13D's vit_lr sweep (push the now-confirmed-undertrained tower further) becomes worth running.

  ~~Original 13C plan text, kept for the record:~~ Image tower on full data, same recipe as arm0 (Phase 9B, unblocked). Rationale: every report-gen checkpoint so far, including 13B (the milestone-1 winner), conditions on **completely stock BiomedCLIP — never fine-tuned on a single MIMIC-CXR image**. 13F's negative result doesn't bear on this lever — that was a decoder-side data-reweighting test; this is a visual-representation-quality lever, a different mechanism, and per the retrieval chapter's own pre-registered prediction (9C) more data should shift the vit_lr optimum right and raise the ceiling — never tested at full scale.

  Confirmed against the actual arm0 launch command (`h100_scaling_state.json` notes, job 2471261: `DATASET_CONFIG=cxr_mimic_arm0 VIT_UNFREEZE=12 SELECTION_SPLIT=true sbatch scripts/train_biomedclip_kd_150m_h100.sh`) and the real script defaults (not re-derived from memory) — `BATCH_SIZE=64` → `BACKBONE_LR=1.41e-5`/`HEAD_LR=4.24e-4`/`MAX_STEPS=6000` are all script defaults at that batch size, `VIT_LR=1e-6` is the script default. Reusing the identical recipe on full data needs only two overrides:
  ```
  DATASET_CONFIG=cxr_mimic_full VIT_UNFREEZE=12 EXPERIMENT=h100_kd_150m_v2_full_data sbatch scripts/train_biomedclip_kd_150m_h100.sh
  ```
  `SELECTION_SPLIT` deliberately left at its default `false` — that lever exists for retrieval-eval protocol integrity (6F), irrelevant here since this run isn't being evaluated for retrieval R@10. **Same step count as arm0 (6000 steps) → wall-clock stays ~3.5-4h** even at ~11x the data — only the epoch count drops (22→~2.0, since MAX_STEPS×BATCH_SIZE is held at a constant 384,000-sample budget by the script's own design). Then **13E** (below) plugs the resulting tower into the decoder and retrains — that step is what actually tells us if this helped.

- [ ] **13D** (2026-09-02, first arm staged, not yet run) — **vit_lr sweep, now evidence-backed rather than conditional-on-ambiguity — 13C/13E's real, confirmed win reopens this.** Starting with the single most information-dense arm rather than the full 4-way sweep serially (~14h): `VIT_LR=3e-6`. On the small 27.5k-pair set (6G-1), `vit_lr=3e-6` was the **actual peak** (R@10 0.183, beating 1e-6's 0.171) — passed over historically only for a val-loss/retrieval-selection-protocol divergence concern, not because it underperformed. 9C's prediction (more data makes higher `vit_lr` safer) reinforces this choice rather than undercutting it.
  ```
  DATASET_CONFIG=cxr_mimic_full VIT_UNFREEZE=12 VIT_LR=3e-6 EXPERIMENT=h100_kd_150m_v2_full_data_lr3e6 sbatch scripts/train_biomedclip_kd_150m_h100.sh
  ```
  Same 6000-step budget as 13C (single lever: only `VIT_LR` changed).

  **ARM1 TRAINING DONE 2026-09-02 (job 2502673, ~3.5h, matched estimate) — real improvement over 13C.** In-training i2t R@10 **0.316** (N=1433) vs 13C's 0.285 (+10.9% rel). Both `train/clip_loss` (2.123→1.982) and `val/clip_loss` (2.183→2.075) dropped vs 13C — not overfitting, genuinely more signal extracted. Confirms `3e-6` (the actual 6G-1 peak on the small dataset) pulls ahead of `1e-6` at full scale, as 9C predicted. Checkpoint: `outputs/h100_kd_150m_v2_full_data_lr3e6/checkpoints/last.ckpt`.

  Then retrain the decoder against it, 13E's exact recipe:
  ```
  NUM_GPUS=4 MAX_STEPS=12000 IMAGE_ENCODER_CKPT=./outputs/h100_kd_150m_v2_full_data_lr3e6/checkpoints/last.ckpt EXPERIMENT=h100_report_gen_full_ext_4gpu_tower13d sbatch --gpus=4 scripts/train_report_generation_h100.sh
  ```
  Eval (beam decode, n=1433 first):
  ```
  DECODE=beam BEAM_SIZE=3 DUMP_DIR=results/report_gen_tower13d_n1433 CHECKPOINT=./outputs/h100_report_gen_full_ext_4gpu_tower13d/checkpoints/last.ckpt NUM_SAMPLES=1433 sbatch scripts/inspect_report_generation_h100.sh
  DUMP_DIR=results/report_gen_tower13d_n1433 sbatch scripts/score_chexbert_h100.sh
  ```
  **DECODER RETRAIN DONE 2026-09-02 (jobs 2502981/2503223/2503391, n=1433 validate.parquet) — another clean, consistent improvement, nothing traded away.**

  | metric | 13B (stock) | 13E (`vit_lr=1e-6`) | 13D (`vit_lr=3e-6`) | retrieval floor |
  |---|---|---|---|---|
  | ROUGE-L | 0.2102 | 0.2165 | 0.2148 (flat) | 0.1881 |
  | BLEU-1 / BLEU-4 | 0.2831/0.0710 | 0.2905/0.0749 | 0.2914/0.0753 (flat/slight up) | 0.2605/0.0465 |
  | exact-match accuracy | 0.3636 | 0.3615 | **0.3859** (+6.8%, best yet) | 0.3036 |
  | CheXbert-14-micro | 0.4382 | 0.4380 | **0.4595** (+4.9%) | 0.4145 |
  | CheXbert-14-macro | 0.2631 | 0.2695 | **0.2869** (+6.5%) | 0.3054 |
  | CheXbert-5-micro | 0.4930 | 0.4963 | **0.5255** (+5.9%) | 0.4624 |
  | CheXbert-5-macro | 0.3956 | 0.4006 | **0.4180** (+4.3%) | 0.4118 |

  Vs the retrieval floor, this is the strongest showing yet: CheXbert-14-micro **+10.9%** over floor (was 13E's +5.7%), CheXbert-5-micro **+13.6%** (was +7.3%), CheXbert-5-macro **+1.5%** (consistent win). **CheXbert-14-macro's gap to the floor has collapsed to 6.1%** (was 11.75% at 13E, 13.85% at 13B) — a clear **monotonic trend across three checkpoints** (13B→13E→13D), not noise. `13D`'s checkpoint (`outputs/h100_report_gen_full_ext_4gpu_tower13d/checkpoints/last.ckpt`) is the best checkpoint by a clear margin.

  **NEXT: confirm on the official test split** (same discipline as every checkpoint before this):
  ```
  DECODE=beam BEAM_SIZE=3 PARQUET=/sc/home/$USER/dataset/mimic_full/test.parquet NUM_SAMPLES=999999 DUMP_DIR=results/report_gen_tower13d_test_split CHECKPOINT=./outputs/h100_report_gen_full_ext_4gpu_tower13d/checkpoints/last.ckpt sbatch scripts/inspect_report_generation_h100.sh
  DUMP_DIR=results/report_gen_tower13d_test_split sbatch scripts/score_chexbert_h100.sh
  ```
  **CONFIRMED FINAL on the official test split 2026-09-03 (n=2663) — the monotonic trend holds, not just validate.**

  | metric | 13B | 13E (`1e-6`) | 13D (`3e-6`) | retrieval floor |
  |---|---|---|---|---|
  | ROUGE-L | 0.1852 | 0.1863 | **0.1899** | 0.1636 |
  | CheXbert-14-micro | 0.4412 | 0.4591 | **0.4736** | 0.4296 |
  | CheXbert-14-macro | 0.2487 | 0.2692 | **0.2800** | 0.3014 |
  | CheXbert-5-micro | 0.5280 | 0.5441 | **0.5522** | 0.4856 |
  | CheXbert-5-macro | 0.4234 | 0.4450 | **0.4487** | 0.4284 |

  CheXbert-14-macro's gap to the floor: **17.5% → 10.7% → 7.1%** across 13B→13E→13D, on the official split — a clean 3-point monotonic trend, not a single lucky result. The other three CheXbert sub-metrics all win by their widest margins yet (14-micro +10.2%, 5-micro +13.7%, 5-macro +4.7%).

  **DECISION (user, 2026-09-03): push to `vit_lr=1e-5`**, the next sweep arm — `3e-6` showed zero overfitting and the trend is still climbing.
  ```
  DATASET_CONFIG=cxr_mimic_full VIT_UNFREEZE=12 VIT_LR=1e-5 EXPERIMENT=h100_kd_150m_v2_full_data_lr1e5 sbatch scripts/train_biomedclip_kd_150m_h100.sh
  ```
  **ARM2 (`vit_lr=1e-5`) TOWER TRAINING DONE 2026-09-03 — no overfitting yet, best of the three arms.**

  | `vit_lr` | i2t R@10 | train/clip_loss | val/clip_loss | train/val gap |
  |---|---|---|---|---|
  | 1e-6 (13C) | 0.285 | 2.123 | 2.183 | 0.060 |
  | 3e-6 (13D) | 0.316 | 1.982 | 2.075 | 0.093 |
  | **1e-5 (arm2)** | **0.350** | **1.752** | **1.910** | 0.158 |

  Val loss is still *decreasing* at `1e-5` (1.910 < 2.075 < 2.183), not turning up — on the small 27.5k-pair set this exact `vit_lr` was already past the overfitting knee (`val_clip_loss` rose to 3.412). Confirms 9C's prediction again: full-data scale pushes the knee further right than even this. **Worth watching, not alarming yet**: the train/val gap is widening monotonically (0.060→0.093→0.158) even as val keeps improving — the earliest visible sign the knee exists somewhere ahead. Checkpoint: `outputs/h100_kd_150m_v2_full_data_lr1e5/checkpoints/last.ckpt`.

  **NEXT: retrain the decoder against it**, same recipe:
  ```
  NUM_GPUS=4 MAX_STEPS=12000 IMAGE_ENCODER_CKPT=./outputs/h100_kd_150m_v2_full_data_lr1e5/checkpoints/last.ckpt EXPERIMENT=h100_report_gen_full_ext_4gpu_tower13d_lr1e5 sbatch --gpus=4 scripts/train_report_generation_h100.sh
  ```
  Eval (beam decode, n=1433 first):
  ```
  DECODE=beam BEAM_SIZE=3 DUMP_DIR=results/report_gen_tower13d_lr1e5_n1433 CHECKPOINT=./outputs/h100_report_gen_full_ext_4gpu_tower13d_lr1e5/checkpoints/last.ckpt NUM_SAMPLES=1433 sbatch scripts/inspect_report_generation_h100.sh
  DUMP_DIR=results/report_gen_tower13d_lr1e5_n1433 sbatch scripts/score_chexbert_h100.sh
  ```
  **ARM2 DECODER RETRAIN DONE 2026-09-03 — first regression in the sweep. SWEEP CONCLUDED, `13D`@`3e-6` is the final checkpoint.**

  | metric | 13D (`3e-6`) | arm2 (`1e-5`) | change |
  |---|---|---|---|
  | ROUGE-L / BLEU-4 | 0.2148 / 0.0753 | **0.2177 / 0.0764** (best yet) | +1.4% / +1.5% |
  | exact-match accuracy | 0.3859 | 0.3636 | **-5.8%** |
  | CheXbert-14-micro / macro | 0.4595 / 0.2869 | 0.4558 / 0.2850 | -0.8% / -0.7% (flat) |
  | CheXbert-5-micro / macro | 0.5255 / **0.4180** ✅ | 0.5138 / 0.4071 ❌ | **-2.2% / -2.6%, CheXbert-5-macro flips back to LOSING the retrieval floor** |

  ROUGE-L/BLEU nudged up, but every CheXbert metric regressed, and CheXbert-5-macro lost the floor-beating milestone `3e-6` had won. **Genuine finding for the writeup**: the tower's own contrastive R@10 kept climbing monotonically through `1e-5` (0.285→0.316→0.350, no overfitting by that metric — see the ARM2 tower-training entry above), but its usefulness for *downstream report-gen conditioning* peaked earlier, at `3e-6`. Retrieval R@10 is not a perfect proxy for what the decoder needs from the image tower.

  **Decision: do not run the `3e-5` arm.** The trend has already inflected on the metric that matters (CheXbert F1), and `3e-5` is expected to push further into overfitting (it was the worst arm on the small dataset too — running it would spend ~7-8h to confirm what's already visible). **`13D` (`vit_lr=3e-6`) is the FINAL checkpoint for the whole Phase 13 arc** — `outputs/h100_report_gen_full_ext_4gpu_tower13d/checkpoints/last.ckpt`, already confirmed on both validate.parquet and the official test split.

  **Phase 13 arc summary**: 13A (free beam decode) + 13B (extended training, ~2.5h) cleared milestone 1 outright; 13C/13D (full-data image tower, 3-arm `vit_lr` sweep) pushed further, peaking at `3e-6`; 13F (rare-label oversampling, 2 doses) was tried and honestly failed, abandoned. **NEXT: Phase 12 writeup**, using 13D as the final/headline checkpoint.

- [x] **13E** (2026-09-01, jobs 2501557 training / 2501876+2501976 eval) — **Retrain the Stage-13B decoder against the Stage-13C tower. DONE (validate split): real, consistent improvement — 13E is now the best checkpoint.** `IMAGE_ENCODER_CKPT` loaded cleanly (`Loaded fine-tuned image encoder... Missing: 0, unexpected: 0` — confirms the checkpoint-loading code shipped weeks earlier works correctly against a real trained tower's key structure, not just the unit test).

  | metric | 13B (stock BiomedCLIP) | 13E (full-data tower) | retrieval floor | verdict |
  |---|---|---|---|---|
  | ROUGE-L | 0.2102 | **0.2165** (+3.0%) | 0.1881 | improved |
  | BLEU-1 / BLEU-4 | 0.2831 / 0.0710 | **0.2905 / 0.0749** (+5.5% BLEU-4) | 0.2605 / 0.0465 | improved |
  | exact-match accuracy | 0.3636 | 0.3615 | 0.3036 | flat (noise) |
  | CheXbert-14-micro | 0.4382 | 0.4380 | 0.4145 | flat, still wins floor |
  | CheXbert-14-macro | 0.2631 | **0.2695** (+2.4%) | 0.3054 | improved, gap to floor 13.9%→**11.8%** |
  | CheXbert-5-micro | 0.4930 | **0.4963** | 0.4624 | improved |
  | CheXbert-5-macro | 0.3956 | **0.4006** (+1.3%) | 0.4118 | improved, gap to floor 3.9%→**2.7%** |

  Unlike 13F's noisy, non-monotonic negative pattern, this is a **consistent** improvement across nearly every metric with **nothing traded away** — the signature of a real effect, not variance. `13E`'s checkpoint (`outputs/h100_report_gen_full_ext_4gpu_tower13c/checkpoints/last.ckpt`) beats 13B on 6/8 metrics tracked here, flat on the other 2.

  **CONFIRMED FINAL 2026-09-02 on the official test split (n=2663) — the win holds and GROWS, plus a new milestone.**

  | metric | 13B (stock) | 13E (full-data tower) | retrieval floor | verdict |
  |---|---|---|---|---|
  | ROUGE-L | 0.1852 | 0.1863 | 0.1636 | slight improvement |
  | BLEU-1 / BLEU-4 | 0.2347 / 0.0513 | 0.2392 / 0.0521 | 0.2372 / 0.0330 | improved |
  | exact-match accuracy | 0.2249 | 0.2227 | 0.1735 | flat (noise) |
  | CheXbert-14-micro | 0.4412 | **0.4591** (+4.1%) | 0.4296 | floor margin grew +2.7%→**+6.9%** |
  | CheXbert-14-macro | 0.2487 | **0.2692** (+8.2%) | 0.3014 | gap to floor 17.5%→**10.7%** |
  | CheXbert-5-micro | 0.5280 | **0.5441** (+3.0%) | 0.4856 | floor margin grew |
  | CheXbert-5-macro | 0.4234 | **0.4450** (+5.1%) | 0.4284 | **NEW: now beats the floor** (was -1.2%, now **+3.9%**) ✅ |

  A new milestone: **CheXbert-5-macro now beats the retrieval floor too** — 3 of 4 CheXbert sub-metrics winning outright, only CheXbert-14-macro still behind (gap nearly halved). Per-label, gains are broad (Cardiomegaly, Edema, Atelectasis, Lung Opacity all up) — **not** concentrated in the 3 still-unsolved rare labels (Lung Lesion/Pleural Other still F1=0, Pneumothorax barely nonzero), confirming this is a genuine general-representation improvement, complementary to (not a fix for) 13F's rare-label problem. **`13E` is the confirmed final/best checkpoint**: `outputs/h100_report_gen_full_ext_4gpu_tower13c/checkpoints/last.ckpt`.

  Since the tower was confirmed *undertrained* (13C's retrieval curve was still rising at step 6000) and this real, positive, non-noisy result came from it anyway, `13D` (vit_lr sweep or extended steps) is now a genuine, evidence-backed follow-up rather than speculation. **DECISION (user, 2026-09-02): push further with 13D** — see below.

  ~~Original 13E plan text, kept for the record:~~ Retrain the Stage-13B decoder against the Stage-13C/13D tower — combines the step-budget fix with the new image tower via the `IMAGE_ENCODER_CKPT` lever (**shipped 2026-08-30**: `ReportGenerationLightningModule.load_image_encoder()` gained an optional `image_encoder_checkpoint` param — loads a Phase-9 contrastive `.ckpt`'s `image_encoder.*`-prefixed weights onto the stock BiomedCLIP architecture, non-strict, via the new module-level `load_image_tower_checkpoint()` helper; `image_encoder_checkpoint` declared in `configs/config.yaml` for Hydra's strict-struct mode; `IMAGE_ENCODER_CKPT` env lever + fail-fast existence check added to `train_report_generation_h100.sh`). Empty/unset keeps stock weights — fully backward compatible.
  Uses 13B's own exact recipe (4-GPU, `MAX_STEPS=12000`, stock `DECODER_CKPT`) with only `IMAGE_ENCODER_CKPT` added — single lever, so any change is attributable to the new tower alone, not confounded with a fresh step-budget or decode-strategy change:
  ```
  NUM_GPUS=4 MAX_STEPS=12000 IMAGE_ENCODER_CKPT=./outputs/h100_kd_150m_v2_full_data/checkpoints/last.ckpt EXPERIMENT=h100_report_gen_full_ext_4gpu_tower13c sbatch --gpus=4 scripts/train_report_generation_h100.sh
  ```
  Eval (beam decode, per 13A's standing decision), n=1433 first:
  ```
  DECODE=beam BEAM_SIZE=3 DUMP_DIR=results/report_gen_tower13c_n1433 CHECKPOINT=./outputs/h100_report_gen_full_ext_4gpu_tower13c/checkpoints/last.ckpt NUM_SAMPLES=1433 sbatch scripts/inspect_report_generation_h100.sh
  DUMP_DIR=results/report_gen_tower13c_n1433 sbatch scripts/score_chexbert_h100.sh
  ```
  Compare against 13B (0.4382/0.4930 14/5-micro, 0.2631/0.3956 14/5-macro) — only rerun on the full n=2663 test split once n=1433 looks promising.

- [x] **13F** (2026-08-31 → 2026-09-01, tried and ABANDONED — negative result at two doses) — **Milestone 1 was cleared by 13B without touching this item; user chose to pursue it anyway to close the one remaining real gap (CheXbert-14-macro, -17.5% vs the floor on the official test split), since it's concentrated in exactly 3/14 labels the 13B checkpoint never predicts (Lung Lesion, Pneumothorax, Pleural Other, all F1=0.0).** Chose (a) from the candidate list below — cheapest, most targeted, no loss-function or architecture change: oversample training reports whose ground-truth CheXpert label is positive for one of these 3 rare findings, via a `WeightedRandomSampler`. **Conclusion (see the two run write-ups below): did not work at either weight tried; abandoned. 13B remains the final checkpoint.**

  **Code shipped:** `scripts/train_report_generation.py` gained `compute_rare_finding_sample_weights()` — a pure function (CPU-testable against a tiny on-disk CSV fixture, no HF Dataset/network needed) that reads `mimic-cxr-2.0.0-chexpert.csv.gz` (U-Zeros convention: 1.0=positive, {0.0,-1.0,NaN}=not-positive — this file was already fetched by Phase 8's `stage_meta()` into `local_parquet_dir` alongside the split/metadata CSVs, but never consumed by the build pipeline until now) and returns per-row weights (`oversample_weight` for any row positive on a target label, else `1.0`; a study_id with no CSV row also gets `1.0` — conservative, never inflates an unknown-label row). `prepare_report_gen_dataloader()` wires this into a `WeightedRandomSampler` when `dataset.oversample_rare_findings=true` (train split only; `shuffle`/`sampler` are mutually exclusive on `DataLoader`, so `shuffle` is forced off when the sampler is active). `configs/dataset/cxr_mimic_full.yaml` (and `cxr_mimic_arm0.yaml`, kept byte-parity per the existing `test_cxr_mimic_arm0_config_is_full_pointed_at_arm0_symlink_dir` invariant — the chexpert CSV is the single official ground-truth file, not arm0-specific, so both configs point at the same path) declare 4 new keys: `oversample_rare_findings: false` (default off, zero behavior change unless enabled), `chexpert_csv` (hardcoded path, mirrors `local_parquet_dir`'s own hardcoding convention), `rare_finding_labels: ["Lung Lesion", "Pneumothorax", "Pleural Other"]`, `oversample_weight: 5.0`. `train_report_generation_h100.sh` gained `OVERSAMPLE_RARE`/`OVERSAMPLE_WEIGHT` env levers (which 3 labels to target stays YAML-only — Hydra CLI-list overrides with spaces in label names are painful, and changing the target set is a rarer edit than tuning the weight). 4 new parity tests (weight-computation correctness incl. out-of-CSV-order study_ids and a study absent from the CSV entirely; SLURM wrapper lever presence; config declared-key presence on both configs). `validate_for_willi.sh`: 162 passed (was 159), 9/9 gates green.

  **RUN 2026-08-31 (jobs 2497357 training, 2497707/2498259 eval) — NEGATIVE RESULT, the flagged overcorrection risk materialized.** n=1433 validate.parquet, beam decode:

  | metric | 13B (no oversample) | 13F (weight=5.0) | retrieval floor | verdict |
  |---|---|---|---|---|
  | ROUGE-L | 0.2102 | 0.2112 | 0.1881 | both still beat floor |
  | exact-match accuracy | 0.3636 | **0.3803** | 0.3036 | 13F slightly better |
  | CheXbert-14-micro | **0.4382** ✅ | 0.3964 | 0.4145 | **13F falls BELOW the floor again** ❌ |
  | CheXbert-5-micro | **0.4930** ✅ | 0.4511 | 0.4624 | **13F falls BELOW the floor again** ❌ |
  | CheXbert-14-macro (the target) | 0.2631 | 0.2599 | 0.3054 | essentially flat — no real win even on its own target |

  Training itself ran cleanly (`Oversample rare findings: true (weight=5.0)` confirmed engaged in the log; `WeightedRandomSampler` wiring worked correctly against real cluster data, not just the CPU unit test — no sampler/shuffle conflict, no crash). **Per-label diagnosis:** 2 of the 3 target labels genuinely improved (Pneumothorax F1 0.027→**0.109**, a 4x gain; Lung Lesion F1 0.0→**0.049**), but Pleural Other never moved off F1=0.0 (support=46, apparently too rare/hard even at 5x weight). The cost: 4+ common, high-support labels got meaningfully **worse** — Edema F1 **-25%** (0.362→0.271), Pleural Effusion **-12%** (0.614→0.542), No Finding **-12%** (0.435→0.384), Support Devices -5%. Since CheXbert-14/5-micro pool over all predictions weighted by support, the high-support losses outweighed the rare-label gains in the aggregate — a real, informative negative result, not noise.

  **`13B`'s checkpoint (`outputs/h100_report_gen_full_ext_4gpu/checkpoints/last.ckpt`) remains the best/reference checkpoint** — it is the one that actually clears milestone 1 on both eval splits. `oversample_weight=5.0` was too aggressive a dose for this lever.

  **SECOND ATTEMPT 2026-09-01 (jobs 2498304 training / 2500656 eval / 2500738 scoring, `OVERSAMPLE_WEIGHT=2.0`, n=1433 validate.parquet) — STILL NEGATIVE, and the pattern points to noise, not a real dose-response.**

  | metric | 13B (no oversample) | 13F @5.0 | 13F @2.0 | retrieval floor |
  |---|---|---|---|---|
  | ROUGE-L | 0.2102 | 0.2112 | **0.2146** | 0.1881 |
  | exact-match accuracy | 0.3636 | **0.3803** | 0.3496 | 0.3036 |
  | CheXbert-14-micro | **0.4382** ✅ | 0.3964 | 0.4026 | 0.4145 |
  | CheXbert-5-micro | **0.4930** ✅ | 0.4511 | 0.4308 | 0.4624 |
  | CheXbert-14-macro (the target) | 0.2631 | 0.2599 | **0.2539 (worst)** | 0.3054 |
  | CheXbert-5-macro | 0.3956 | 0.3589 | 0.3479 (worst) | 0.4118 |

  Weight=2.0 still loses to the retrieval floor on both micro metrics. Critically, it is **not intermediate** between 13B and weight=5.0 the way a clean dose-response would predict — CheXbert-14-macro and 5-macro are the *worst* of all three variants at the *gentler* weight. Per-label: Pneumothorax kept improving with less weight (F1 0.109→**0.148**), but Lung Lesion — which had reached F1=0.049 at weight=5.0 — fell straight back to **exactly 0.0** at weight=2.0, and Pleural Other stayed at F1=0.0 across **all three variants at any weight tried** (support=46 — apparently unlearnable via this lever regardless of dose). This non-monotonic, label-specific scatter across only two data points is the signature of ordinary single-run training variance dominating whatever real effect this lever has — without multi-seed averaging, this project cannot distinguish "oversampling helps a little" from "this run happened to land differently."

  **DECISION: abandon the rare-label oversampling lever. Phase 13 is CLOSED. `13B`'s checkpoint (`outputs/h100_report_gen_full_ext_4gpu/checkpoints/last.ckpt`) is FINAL.** Milestone 1 (the plan's actual pre-registered success bar) is already secured and confirmed on both eval splits by 13B alone — further compute chasing a noisy secondary metric isn't worth it. This is a defensible, honest research narrative for the 12B writeup: two cheap levers (free decode-strategy swap; ~2h23m extended training) closed the primary gap; a third, more invasive lever (data reweighting) was tried at two doses specifically targeting the one remaining secondary-metric gap and honestly did not work, rather than being silently dropped or cherry-picked. **NEXT: Phase 12 writeup**, using 13B as the final report-generation checkpoint and this full 13A–13F arc as the improvement narrative.

  ~~Original 13F plan text, kept for the record:~~ Boilerplate/imbalance fix, conditional (only if milestone 1 still unmet after 13A-13E). Confirmed via code survey: zero existing code anywhere for class-imbalance handling, oversampling, or auxiliary multi-label losses on this dataset. Candidates to choose between once the post-13E per-label CheXbert breakdown is in hand (do not pre-commit): (a) oversample rare-finding reports in the training dataloader (cheapest, loader-only, no loss change); (b) an auxiliary CheXpert multi-label classification loss on top of the LM loss, using the ground-truth `mimic-cxr-2.0.0-chexpert.csv.gz` already used elsewhere in this repo (more invasive, targets the eval metric directly); (c) decode-time repetition penalty against the dominant boilerplate n-grams (cheap, but risks relabeling the boilerplate rather than fixing grounding).

**Code shipped this session (2026-08-30), validated:** `hybrid_xmamba/training/lightning_module.py` (new `load_image_tower_checkpoint()` pure helper + `load_image_encoder(image_encoder_checkpoint=...)` param), `scripts/train_report_generation.py` (threads `image_encoder_checkpoint` through), `configs/config.yaml` (declares the key), `scripts/train_report_generation_h100.sh` (`IMAGE_ENCODER_CKPT` + `NUM_GPUS`/`TRAINER_CFG` levers, fail-fast checks for both). 5 new parity tests (checkpoint-loading helper against a CPU-only stand-in tower, both new SLURM levers, the declared config key, `h100_multi_ddp.yaml`'s key coverage). `validate_for_willi.sh`: 159 passed (was 154), 9/9 gates green.

---

### Phase 14 — Supervisor review response 🎯 NEW (2026-09-07) — **REOPENS THE PLAN**

**Why:** the results were shown to the supervisor (2026-09-07). Three findings came back. All three are fair, and all three are about **validity of the central claim**, not about squeezing out a better number. The priority order below is the supervisor's, not a re-ranking by this plan.

1. **There is no trained, parameter-matched Transformer baseline anywhere in the report.** The thesis is *"attention-free hybrid matches/beats attention-based transformers at better efficiency."* Every comparison currently in `analysis/h100_scaling_results.md` is against (a) this project's own architecture variants (pure-Mamba, pure-xLSTM), (b) an off-the-shelf, non-fine-tuned model (BiomedCLIP zero-shot), or (c) a naive nearest-neighbour retrieval control. **None of those is the baseline the claim is actually about.** §3 of the writeup already concedes exactly this ("*there is no attention/transformer baseline in this repo … the '~2.0 = quadratic attention' reference line is a cited comparison, not a measurement made here*") — the supervisor's point is that a conceded caveat is not a substitute for the experiment. **Nothing else in Phase 14 matters if 14A is not in place.**
2. **The boilerplate/duplicate-template rate was never re-measured on the final checkpoint.** 11C found 1055/1433 (**73.6%**) of generations fell into one of **184** exact-duplicate template clusters — on the *pre-Phase-13* checkpoint. `final_verdict.open_items[0]` admits it was never re-checked on **13D**, the checkpoint every headline number comes from. This is the single largest validity threat to the primary result: **if the generator is still mostly emitting templates, "beats the retrieval-NN floor" is hollow, because emitting a plausible templated report is exactly what the retrieval floor does too.**
3. **The disclosed selective-scan correctness defect is stated but neither fixed nor bounded.** The supervisor checked the branch directly and is correct on every particular: the fp32 guard (2026-07) *is* in (`scan_interface.py:184-200`), but it is a **separate** issue from the underlying `A_cum_safe = A_cum_ci.clamp(min=1e-8)` divide-by-decay approximation (`scan_interface.py:118`, mirrored at `mamba_block.py:229` and `mamba_block_v2.py:343`), which is still there — and `tests/test_kernels.py` has **no** test comparing the chunked scan against an exact reference recurrence (it has `test_selective_scan_doc_boundary_reset` and a TFLA-vs-PyTorch check, neither of which is a correctness bound). Any reviewer told this will ask whether it affects the reported numbers. **Either fix it, or add an explicit error-bound test and report the max deviation.**

**Status of the plan:** `current_phase` moves `plan_closed` → `phase14_supervisor_review`. The Phase 1–13 record below is **unchanged and still valid** — Phase 14 adds the missing baseline and the missing validity checks; it does not re-litigate any closed arm. Retrieval stays closed (do not re-open).

**⚠ OPERATOR FREEZE (load-bearing, read before touching 14C).** The selective-scan operator **must not change** between the incumbent hybrid and the 14A Transformer baseline. The Transformer has no selective scan at all, so applying the 14C fix to the hybrid mid-campaign would make the head-to-head uninterpretable, and would additionally break comparability with every number already in `final_verdict`. **Run 14A on the current, frozen, now-documented operator.** This is also the conservative direction: the defect can only *understate* the hybrid (it annihilates state contributions), so a hybrid win measured on the defective operator is a **lower bound** on the hybrid's true quality. Record this argument in the writeup — it is the answer to "does the bug affect your conclusion?"

**Execution order** (cheapest-and-most-diagnostic first, long pole started as early as possible):
`14C-1` (CPU, free) → `14B` (CPU, free, no regeneration) → `14A-1`/`14A-2` (local code + config) → **`14A-3` Stage-0 launch (the long pole, ~2-4 GPU-days)** → `14C-2`/`14C-3` while Stage-0 runs → `14A-4`…`14A-8`.

---

#### 14A — Parameter-matched Transformer baseline ⏳ **HIGHEST PRIORITY, NOT STARTED**

**Design decision (integration path).** Add `"attention"` as a fourth `layer_pattern` layer type inside the existing `HybridLanguageModel`, **not** a separate model class. A config of `layer_pattern: ["attention"]` then *is* a pure Transformer. Everything downstream — `ImagePrefixMapper` prefix conditioning, `ReportGenerationLightningModule`, `evaluate_report_generation.py`'s beam search, `score_chexbert_h100.sh`, `performance_profile.py` — is architecture-agnostic and needs **zero** changes. This is both the least-effort path and the one that guarantees the baseline goes through the *identical* pipeline, which is the whole point of a matched baseline.

**Parameter match (computed, not estimated).** Instantiated `hybrid_150m_v2` = **183.7218M** params (embeddings 38.5974M + layers 106.5263M + lm_head 38.5974M, `tie_word_embeddings=false`).

| Transformer candidate | layers | mlp_ratio | total | Δ vs hybrid | verdict |
|---|---|---|---|---|---|
| dim=768, **15L**, r=4.0 | 15 | 4.0 | **183.387M** | **−0.18%** | ✅ **PRIMARY** — standard architecture, near-exact param match |
| dim=768, 12L, r=5.5 | 12 | 5.5 | 183.382M | −0.18% | ❌ rejected — matches params only via a non-standard FFN width; a reviewer reads that as a rigged baseline |
| dim=768, 12L, r=4.0 | 12 | 4.0 | 162.149M | −11.74% | ⚪ optional secondary — depth-matched but 11.7% *fewer* params, so it handicaps the baseline |

**Chosen: `dim=768, num_layers=15, num_heads=12 (head_dim=64), mlp_ratio=4.0`.** Param-matching holds parameters constant and lets each architecture pick its own shape; 15L/768 is an ordinary pre-norm decoder (a slightly deeper GPT-2-small), not a contrivance. Use **RoPE**, not learned positional embeddings — the hybrid spends **zero** params on positional encoding (`embeddings` is exactly `50257×768`), so a learned table would be an unmatched +0.79M and an unearned advantage.

- [x] **14A-1** — **DONE 2026-09-07. Code: `"attention"` layer type.** `hybrid_xmamba/layers/attention_block.py` (RoPE, `F.scaled_dot_product_attention(is_causal=True)`, `bias=False` on qkv/out matching the existing MLP, QK-norm under `use_hybrid_norm`, doc-boundary masking from `cu_seqlens`). Wired into `hybrid_block.py` (import + `LayerType` + dispatch + `cu_seqlens` routing) and `configuration_hybrid.py` (`Literal` + `valid_types` + `get_layer_config` branch + two new fields `attn_dropout`/`rope_theta`). `train_stage0_150m_h100.sh:32` made env-overridable. **Verified live:** causality holds (perturbing future positions leaves past outputs bit-identical), doc-boundary masking works AND matters (without it document 1 IS contaminated by editing document 0 — asserted both directions so the test cannot pass vacuously). New `hybrid_xmamba/layers/attention_block.py` — standard pre-norm causal self-attention (RoPE, `F.scaled_dot_product_attention(..., is_causal=True)`, same `dropout`/`initializer_range` conventions as the existing blocks). Wire into: `layers/hybrid_block.py` dispatch (currently imports only `MambaBlock`/`mLSTMBlock`/`sLSTMBlock` at lines 11-13), `models/configuration_hybrid.py:51` `Literal["mamba","mlstm","slstm"]` → add `"attention"`, plus `_validate()`/`get_layer_type()` at lines 146/162. Also make `scripts/train_stage0_150m_h100.sh:32` env-overridable (`export MODEL_CONFIG="${MODEL_CONFIG:-hybrid_150m_v2}"` — currently hardcoded, blocks 14A-3).
  - **Resolve and record:** what `norm_topology: hybrid` (HybridNorm) means for an attention block. HybridNorm normalizes Q/K/V + Δ/B/C; the Q/K/V half maps onto attention but the design is hybrid-specific. **Decision: use `norm_topology: pre_rms` for the baseline** — canonical pre-norm is what "attention-based Transformer baseline" means to a reviewer. Record it as a stated (small) confound rather than silently choosing.
  - **Beam-search note:** `evaluate_report_generation.py:151` `beam_search_decode` re-forwards the full sequence each step (no KV cache), so attention is **correct out of the box** and no decode changes are needed. It is O(L²) per step, which is a *fair* depiction of attention decode cost. If the n=2663 eval becomes intractable, add a KV cache **before** running 14A-7, and say so.
  - Parity tests in `tests/test_willi_parity.py` + `tests/test_layers.py`; `bash scripts/validate_for_willi.sh` must exit 0.
- [x] **14A-2** — **DONE 2026-09-07. Configs.** `configs/model/transformer_150m_baseline.yaml` (Stage-0/LM variant) and `configs/model/transformer_150m_baseline_rrg.yaml` (report-gen variant, mirroring the `hybrid_150m_v2` → `hybrid_150m_v2_rrg` delta exactly: `image_patch_dim: 768`, `prefix_k: 32`, `vit_unfreeze_blocks: 0`, `vit_lr: 1.0e-6`, `decoder_lr: 1.0e-5`, `head_lr: 3.0e-4`, `weight_decay: 0.01`, `warmup_steps: 500`, `max_steps: 10000`, `gradient_clip_val: 0.5`).
  **Every shared hyperparameter is copied verbatim from the hybrid — `learning_rate: 4.0e-4`, `warmup_steps: 2000`, `weight_decay: 0.1`, `max_position_embeddings: 1024`, `dropout: 0.1`, `vocab_size: 50257`, `tie_word_embeddings: false`. Do NOT re-tune the baseline's LR.** Record as an honest limitation: the LR was √-width-scaled *for the hybrid*, so the Transformer runs at a possibly-suboptimal LR. If the Transformer loses on quality, this is the **first** thing an examiner will attack and the first thing to re-test (a 2-arm LR probe at {4e-4, 6e-4} is the pre-agreed remedy).
  **Before launching 14A-3**, instantiate both configs and assert `abs(n_params_transformer / 183_721_800 - 1) < 0.005`; add that assertion as a parity test so it cannot silently drift.
- [x] **14A-3** — **DONE 2026-09-08 (job 2516833, gx09, 23.3 h wall, 120,000 steps). THE TRANSFORMER WINS STAGE-0 BY A CLEAR MARGIN.**

  | model | params | Stage-0 val PPL | |
  |---|---|---|---|
  | `hybrid_150m_v2` (incumbent, Phase 5) | 183,721,824 | **13.18** | |
  | `transformer_150m_baseline` (14A) | 183,386,880 | **11.222** | **−14.9% relative** |

  **Comparability verified before reading anything into it:** same wrapper (`train_stage0_150m_h100.sh`), same corpus (PubMed `ccdv/pubmed-summarization`, 943,706 packed chunks × 512 = 483M tokens), same 120,000 steps, same effective batch 48 (bs 16 × accum 3), same LR 4.0e-4 / warmup 2000 / WSD, same `gradient_clip_val=0.5`, same gradient checkpointing, same BioMedLM 2.6B KD teacher at alpha=0.5. **Only `MODEL_CONFIG` differed** — the single lever held.
  Confirmed live in the log: `Student trainable params: 183,386,880 (183.4M)`, `layer_pattern: [attention]`, `num_layers: 15`, `norm_topology: pre_rms`. `Trainer.fit stopped: max_steps=120000 reached`. Final `val/perplexity: 11.222` (`train/ce_loss 2.153 → train/perplexity 8.681`; note `val/loss 2.356` is the KD-blended objective, not the CE the perplexity comes from — `ln(11.222)=2.418` is the val CE).

  **What this does and does not mean.** It is a real, clean loss on the first head-to-head, and it must be reported as one. But Stage-0 PPL is a **text-only LM metric**, and this project has already measured that it does not automatically transfer: null #1 of 10 records *"Stage-0 PPL 15.62 → 13.18 moved retrieval flat"*. ⚠️ **The plan's own honest nuance cuts against comfort here**, and it is recorded above at line ~427: that null was measured on **retrieval**, which uses the backbone as an *encoder*; **report generation uses it as a *generator* — autoregressive decoding, exactly what LM pretraining optimises — so a PPL advantage is MORE likely to transfer to 14A-5 than it was to retrieval, not less.** Treat 11.222 as a genuine warning sign for the downstream comparison, not as a metric that can be waved away.

  Wall-clock anecdote, **not** evidence: 23.3 h for 120k steps on gx09. Node contention and allocation differ between runs, so this is not a controlled efficiency measurement — **14A-7 is**, and it should now be run.

  ~~Original spec:~~ **Stage-0 pretrain (the long pole).** Identical corpus (PubMed), steps, batch, schedule as the hybrid's Phase-5 Stage-0. Single lever = `MODEL_CONFIG`.

  ```bash
  MODEL_CONFIG=transformer_150m_baseline EXPERIMENT=h100_stage0_transformer_150m sbatch scripts/train_stage0_150m_h100.sh
  ```
  Report val PPL against the hybrid's **13.18**. (This is also the first real head-to-head backbone-quality number in the project.)
- [ ] **14A-4** — **Image tower: REUSE 13D's, unchanged.** `outputs/h100_kd_150m_v2_full_data_lr3e6/checkpoints/last.ckpt`. Do **not** train a Transformer text tower — that adds a second lever and confounds the decoder comparison.
  ⚠ **Honest caveat to record in the writeup:** that tower was contrastively co-trained *with the hybrid text encoder*, so its prefix space is mildly hybrid-favouring. If the Transformer wins anyway, the caveat is moot. If it loses **narrowly**, this is the first confound to question, and the remedy (a per-backbone tower retrain, ~7-8h) is pre-agreed.
- [x] **14A-5** — **DONE 2026-09-09 (job 2522682, gx12, 4×H100, 36.4 min). Report-gen decoder train.**
  Init was clean on both sides: decoder `Missing keys: 0, Unexpected: 1`; image tower `Missing: 0, unexpected: 0` (13D tower reused unchanged, as planned). 184,566,528 trainable. 12,000 steps ≈ 8 effective epochs.

  | | hybrid (13B reference) | transformer (14A-5) |
  |---|---|---|
  | best `val_lm_loss` | 1.0485 | **1.0143** |
  | `train/lm_loss_epoch` | 0.919 | **0.847** |
  | throughput | 1.51 it/s (job 2504565) | **7.96 it/s** (5.3×) |

  ⚠️ **`val_lm_loss` is NOT the paper's metric.** ROUGE-L and CheXbert F1 decide the claim and are unmeasured until 14A-6. Teacher-forced loss and generation quality have come apart before in this project (13A: beam beat greedy on every generation metric at identical loss).
  The 5.3× throughput is worth noting separately: it **independently confirms 14A-7's synthetic prediction (5.1× at L=256 training) on the real task**, with real data and real DDP rather than random weights.

  ~~Original spec:~~ **Report-gen decoder train.** Identical to 13D's winning command in every respect except `MODEL_CONFIG`:

  ⚠️ **`DECODER_CKPT` MUST be overridden — this was a real bug in the first draft of this command.** It defaults to `./outputs/h100_stage0_150m_v2/checkpoints/stage0_model_only.pt` (the **hybrid's** Stage-0 backbone) and the wrapper's existence check passes for it no matter which architecture you are training. Omitting it does **not** fail: the load runs under `strict=False`, matches almost nothing, and the Transformer trains **from random init** — silently, at the cost of a full 4-GPU run, and it would quietly invalidate the entire matched-baseline comparison.
  ```bash
  MODEL_CONFIG=transformer_150m_baseline_rrg \
  DECODER_CKPT=./outputs/h100_stage0_transformer_150m/checkpoints/last.ckpt \
  NUM_GPUS=4 MAX_STEPS=12000 \
    IMAGE_ENCODER_CKPT=./outputs/h100_kd_150m_v2_full_data_lr3e6/checkpoints/last.ckpt \
    EXPERIMENT=h100_report_gen_transformer_tower13d \
    sbatch --gpus=4 scripts/train_report_generation_h100.sh
  ```
  Passing the Lightning `last.ckpt` directly is fine — `train_report_generation.py:180` does `ckpt.get("state_dict", ckpt)` and strips `model.`/`lm.` prefixes, so it handles both the raw `.ckpt` and the stripped `stage0_model_only.pt` the hybrid used. The Stage-0 module's `teacher.*` and `kd_projection.*` keys land in `unexpected` and are ignored, which is correct.
  **Check this line in the log before trusting the run:** `Loaded. Missing keys: N, Unexpected: M`. `N` should be near zero. A guard added 2026-09-08 now hard-fails above 50% missing and warns above 5%, but read the number anyway.
- [x] **14A-6** — **DONE 2026-09-09. SPLIT DECISION — THE HYBRID WINS EVERY CheXbert F1 METRIC.** Checkpoint init clean (`Missing keys: 0, Unexpected: 0`).

  | metric (official test, n=2663) | hybrid 13D | transformer | retrieval floor | winner |
  |---|---|---|---|---|
  | ROUGE-L | 0.1899 | **0.1936** | 0.1636 | transformer (+2.0%) |
  | BLEU-1 | 0.2469 | **0.2496** | 0.2372 | transformer (+1.1%) |
  | BLEU-4 | 0.0542 | **0.0571** | 0.0330 | transformer (+5.3%) |
  | exact-match accuracy | 0.2163 | **0.2306** | 0.1735 | transformer (+6.6%) |
  | **CheXbert-14-micro** | **0.4736** | 0.4590 | 0.4296 | **hybrid (+3.2%)** |
  | **CheXbert-14-macro** | **0.2800** | 0.2774 | 0.3014 | **hybrid (+0.9%)** |
  | **CheXbert-5-micro** | **0.5522** | 0.5249 | 0.4856 | **hybrid (+5.2%)** |
  | **CheXbert-5-macro** | **0.4487** | 0.4319 | 0.4284 | **hybrid (+3.9%)** |

  **The split falls along a meaningful axis.** CheXbert F1 asks *did the report assert the right findings*; ROUGE-L/BLEU ask *does the text look like the reference*. The hybrid wins all four of the former, the Transformer all four of the latter. Both beat the retrieval floor on ROUGE-L and CheXbert-14-micro.
  ⚠️ **This is not post-hoc framing, and that matters.** This plan has called CheXbert F1 "the more clinically meaningful metric" since **Phase 11 (2026-08-30)** — written when the hybrid was *losing* on it by 29% relative. The designation was made when it was inconvenient, which is exactly what makes it usable now. Do not restate it as if it were chosen after seeing this table.

  **PRE-REGISTERED BAR — where it actually stands.** The bar reads: *"the Transformer does NOT beat the hybrid by more than the 95% bootstrap CI on the difference, on CheXbert-14-micro AND ROUGE-L."*
  - **CheXbert-14-micro: CLEARED outright** — the hybrid is ahead, so no interval is needed to settle this half.
  - **ROUGE-L: undecided.** The Transformer leads by **0.0037** (2.0% relative). Whether that clears the bar ("within noise" ⇒ matches) or fails it depends entirely on the CI — which is precisely why 14A-6 specified one rather than eyeballing the gap.

  **Run the bootstrap to settle it** (tooling shipped 2026-09-09; CPU-only, no GPU):
  ```bash
  A=results/report_gen_tower13d_test_split \
  B=results/report_gen_transformer_test_split \
  NAME_A=hybrid_13D NAME_B=transformer \
  OUTPUT=analysis/bootstrap_hybrid_vs_transformer.md \
    sbatch scripts/bootstrap_compare_h100.sh
  ```
  To get CheXbert F1 intervals as well, first re-run `score_chexbert_h100.sh` for **both** dump dirs — it now also writes `chexbert_labels.json` (per-sample `y_true`/`y_pred`), because micro/macro F1 are not decomposable per sample and so cannot be bootstrapped from the aggregate report. Without them the run still settles the ROUGE-L question, which is the one the bar turns on.

  ~~Original spec:~~ **Quality eval, official test split (n=2663), beam_size=3**

  #### Bootstrap result (job 2525951, 2026-09-09) — TEXT METRICS ONLY, CheXbert still pending

  | metric | hybrid 13D | transformer | diff | 95% CI | verdict |
  |---|---|---|---|---|---|
  | ROUGE-L | 0.1899 | 0.1936 | −0.0038 | [−0.0066, −0.0010] | **transformer wins** |
  | BLEU-1 | 0.2469 | 0.2496 | −0.0027 | [−0.0057, +0.0002] | tie (CI spans 0) |
  | BLEU-4 | 0.0542 | 0.0571 | −0.0029 | [−0.0058, +0.0001] | tie (CI spans 0) |

  **The ROUGE-L gap is REAL — the CI excludes zero.** The pre-registered bar required the Transformer *not* to beat the hybrid on CheXbert-14-micro **AND** ROUGE-L. **The ROUGE-L half is FAILED.** Record that plainly; it is the outcome the pre-registered failure statement was written for.

  ⚠️ **A correction to the 14A-6 headline framing.** The earlier reading — "the Transformer wins all four surface metrics" — does not survive the intervals: **BLEU-1 and BLEU-4 are ties.** Only ROUGE-L is a real win. The point estimates all leaned the same way, which is exactly the pattern an interval is supposed to catch, and it did.

  🔴 **THE DECISIVE MEASUREMENT IS STILL MISSING.** Job 2525951 ran at 16:48 but `chexbert_labels.json` was not written until 16:57 — a pure race (the scorer jobs were still running), **not a bug**: both label dumps succeeded (`Per-sample labels saved to ...` in both logs). **Re-run the bootstrap now that the labels exist.** The hybrid leads CheXbert-14-micro by **+0.0146**, nearly 4× the ROUGE-L gap that just proved significant — so there is a real chance that lead is significant too, and it decides which paper this is:
  - CheXbert CI excludes 0 → *"wins clinical correctness, loses text overlap"* — a genuine trade-off result.
  - CheXbert CI spans 0 → *"loses text overlap, ties clinical correctness"* — a materially weaker claim.
  Do not write either sentence until the interval is in hand.

  ```bash
  A=results/report_gen_tower13d_test_split \
  B=results/report_gen_transformer_test_split \
  NAME_A=hybrid_13D NAME_B=transformer \
  OUTPUT=analysis/bootstrap_hybrid_vs_transformer.md \
    sbatch scripts/bootstrap_compare_h100.sh
  ```
  (`exact_match_accuracy` was added to the bootstrap 2026-09-09 so all eight headline metrics get an interval, not seven.)

  #### ✅ FINAL BOOTSTRAP — THE HEAD-TO-HEAD IS SETTLED (job 2526279, 2026-09-09)

  n=2663, 1000 paired resamples, seed 0. CheXbert label matrices present, so all metrics carry intervals.

  | metric | hybrid 13D | transformer | diff | 95% CI | verdict |
  |---|---|---|---|---|---|
  | ROUGE-L | 0.1899 | 0.1936 | −0.0038 | [−0.0066, −0.0010] | **transformer** |
  | BLEU-1 | 0.2469 | 0.2496 | −0.0027 | [−0.0057, +0.0002] | tie |
  | BLEU-4 | 0.0542 | 0.0571 | −0.0029 | [−0.0058, +0.0001] | tie |
  | **CheXbert-14-micro** | **0.4736** | 0.4590 | **+0.0146** | **[+0.0052, +0.0240]** | **hybrid** |
  | CheXbert-14-macro | 0.2800 | 0.2774 | +0.0026 | [−0.0080, +0.0141] | tie |
  | **CheXbert-5-micro** | **0.5522** | 0.5249 | **+0.0273** | **[+0.0147, +0.0410]** | **hybrid** |
  | **CheXbert-5-macro** | **0.4487** | 0.4319 | **+0.0168** | **[+0.0033, +0.0308]** | **hybrid** |
  | exact-match accuracy (5-label) | 0.0349* | 0.0469* | −0.0120 | [−0.0210, −0.0034] | **transformer** |

  **VERDICT: a statistically supported TRADE-OFF, not a loss.**
  - **Hybrid wins 3** (CI excludes 0): CheXbert-14-micro, CheXbert-5-micro, CheXbert-5-macro — the *clinical correctness* family.
  - **Transformer wins 2**: ROUGE-L and exact-match accuracy — *surface overlap* and all-or-nothing label matching.
  - **Ties 3**: BLEU-1, BLEU-4, CheXbert-14-macro.

  **Pre-registered bar, scored honestly.** The bar required the Transformer not to beat the hybrid on CheXbert-14-micro **AND** ROUGE-L. **The CheXbert-14-micro half passes decisively** (the hybrid *wins* it, CI excludes zero). **The ROUGE-L half fails.** As a conjunction the bar is **NOT cleared** — so the pre-registered failure statement applies and the central claim gets rewritten, exactly as declared on 2026-09-07 before any of this ran. It is *not* rewritten to "efficiency only", because the hybrid did win three quality metrics; it is rewritten to the trade-off the data actually shows.

  ⚠️ **`*` The accuracy row is NOT the 0.2163/0.2306 quoted elsewhere in this plan.** `f1chexbert`'s reported "accuracy" is `accuracy_score` over the **5-label** subset (verified in `F1CheXbert.forward`); the bootstrap additionally computes the stricter **14-label** exact match, which is what 0.0349/0.0469 are. Both are now emitted under explicit `_5`/`_14` names and pinned against sklearn, after an earlier revision reported the 14-label figure under the bare name `exact_match_accuracy` — a 6× difference hiding behind an identical-looking label. The *direction* (transformer ahead) is the same for both.

  **The claim this licenses, and its exact scope:** *at matched parameters (183.4M vs 183.7M), the attention-free hybrid produces clinically more accurate reports — significantly higher CheXbert F1 on 3 of 4 variants — while the Transformer produces text with significantly higher surface overlap, at ~5× the hybrid's throughput.* Every clause is measured with an interval. Nothing beyond it is licensed. — the same protocol the 13D headline numbers use:

  ```bash
  DECODE=beam BEAM_SIZE=3 PARQUET=/sc/home/$USER/dataset/mimic_full/test.parquet \
    NUM_SAMPLES=999999 DUMP_DIR=results/report_gen_transformer_test_split \
    CHECKPOINT=./outputs/h100_report_gen_transformer_tower13d/checkpoints/last.ckpt \
    sbatch scripts/inspect_report_generation_h100.sh
  DUMP_DIR=results/report_gen_transformer_test_split sbatch scripts/score_chexbert_h100.sh
  ```
  Compare against the two rows already in `h100_scaling_state.json.final_verdict`: hybrid 13D (rouge_l 0.1899, CheXbert-14-micro 0.4736) and the retrieval-NN floor (0.1636 / 0.4296).
  **Also add paired bootstrap CIs** (resample the n=2663 test set, 1000 draws, report the 95% CI on the hybrid−Transformer *difference* per metric). Without an interval, "matches" is not a testable statement. Cheap, CPU, reuses the dumped `hyps.txt`/`refs.txt`.
- [x] **14A-7** — **DONE 2026-09-08 (job 2521356, gx12, 3.6 min). THE TRANSFORMER WINS ON EFFICIENCY TOO, AT EVERY LENGTH TESTED. The expected quadratic-attention signature never appeared.**

  **Inference (forward, bs=4), latency / peak memory:**

  | L | hybrid_150m_v2 | transformer_150m_baseline | transformer advantage |
  |---|---|---|---|
  | 256 (**the actual task length**) | 24.72 ms / 1.219 GB | **5.03 ms / 0.509 GB** | 4.9× faster, 2.4× less mem |
  | 2048 | 130.79 ms / 5.714 GB | **16.24 ms / 1.244 GB** | 8.1× faster, 4.6× less mem |
  | 16384 | 919.06 ms / 41.999 GB | **163.82 ms / 7.152 GB** | 5.6× faster, 5.9× less mem |

  **Training (forward+backward, bs=4):**

  | L | hybrid_150m_v2 | transformer_150m_baseline | transformer advantage |
  |---|---|---|---|
  | 256 | 76.10 ms / 7.144 GB | **14.84 ms / 1.327 GB** | 5.1× faster, 5.4× less mem |
  | 2048 | 1084.39 ms / 53.963 GB | **52.59 ms / 7.555 GB** | **20.6× faster, 7.1× less mem** |
  | 4096 | **OOM** | 106.63 ms / 14.675 GB | hybrid cannot run at all |
  | 16384 | **OOM** | 538.06 ms / 57.399 GB | hybrid cannot run at all |

  **Scaling exponents (log-log fit):** inference latency hybrid **0.874** vs transformer **0.865** — *statistically indistinguishable, and nowhere near the 2.0 the writeup's cited reference line predicted.* Training latency: hybrid 1.279, transformer **0.885** — the Transformer scales *better*. Memory: transformer 0.644 (inference) / 0.914 (training) vs hybrid 0.863 / 0.973.

  ### ⚠️ WHY, AND THE CONFOUND THAT MAKES THIS NOT YET AN ARCHITECTURE RESULT

  Two things are happening, and only the first is about architecture:

  1. **`F.scaled_dot_product_attention` dispatches to FlashAttention on H100.** FlashAttention is **O(L) memory**, not O(L²) — it never materialises the L×L matrix. The "~2.0 = quadratic attention" reference line this plan and §3 of the writeup have cited throughout describes *naive* attention, which nobody has shipped since 2022. **The premise of the efficiency claim was out of date.** Attention's FLOPs are still O(L²), and it shows at the tail (8192→16384: transformer latency ×2.30 = exponent 1.20, hybrid ×2.00 = exponent 1.00) — so the hybrid *is* asymptotically flatter. But it is 5.6× behind at L=16384, so the crossover sits around **L ≈ 10⁷–10⁸ tokens**. Practically: never.
  2. 🔴 **THE COMPARISON IS UNFAIR TO THE HYBRID, AND THE UNFAIRNESS IS ENTIRELY IN THE IMPLEMENTATION.** Per 14C-5 (verified in-repo): `scan_interface.selective_scan()` **unconditionally** calls the PyTorch `selective_scan_parallel`, and `selective_scan_triton` is imported and **never invoked**. It also runs the scan in **fp32** (the 2026-07 stability guard). So this benchmark pits a hand-written, fused, bf16 CUDA FlashAttention kernel against a **fp32 PyTorch loop over chunks**. That is an *implementation* comparison, not an *architecture* comparison, and it is the single most important caveat on every number above. The xLSTM row is the tell: at L=16384 inference it is 349 ms vs pure-Mamba's 1099 ms with *more* parameters — the difference between a path with a real kernel and one without.

  **What can honestly be claimed today:** *as implemented in this repository*, the hybrid is slower and more memory-hungry than a parameter-matched FlashAttention Transformer at every sequence length tested, including the ≤256 tokens this project's actual task uses. The architectural linear-vs-quadratic argument is **not falsified in principle** — the exponents at the tail still favour the hybrid — but it is **not realised here**, and the honest writeup must say so in those words.

  ~~Original spec:~~ **Efficiency eval — this is the half of the claim that should win.** (It did not.) It profiles *random* weights (the curves measure architecture, not any trained checkpoint), so it does **not** depend on 14A-3's checkpoint or on 14A-5 finishing. Given 14A-3 went against the hybrid, this is the half of the claim most likely to hold, and it is cheap. **Efficiency eval — this is the half of the claim that should win.** Add the Transformer config to `scripts/performance_profile.py` and re-run the *same* protocol that produced `analysis/efficiency_150m/` (H100 80GB, bf16, bs=4, L ∈ {256, 512, 1024, 2048, 4096, 8192, 16384}, forward and forward+backward, latency / peak memory / tok/s):

  Reuse the existing wrapper (`profile_efficiency_h100.sh`, whose `MODELS` was made env-overridable 2026-09-07) rather than a bare `python` call — same login-node restriction as everything else. It runs BOTH the inference and the forward+backward sweep in one job, which is exactly the protocol that produced `analysis/efficiency_150m/`.
  ```bash
  MODELS="hybrid_150m_v2 mamba_150m_baseline xlstm_150m_baseline transformer_150m_baseline" \
  OUTPUT_DIR=analysis/efficiency_150m_with_transformer \
    sbatch scripts/profile_efficiency_h100.sh
  ```
  `performance_profile.py` resolves `--model` from `configs/model/<name>.yaml` first (the yaml is the source of truth, registry only as fallback), so `transformer_150m_baseline` resolves without touching the registry — verified 2026-09-07.
  Expect the measured quadratic exponent (~2.0) that §3 currently only *cites*. Report the crossover length where the hybrid overtakes attention, and the training-peak-memory gap at L=2048 (the hybrid's strongest measured result: 1078ms/54.0GB vs pure-Mamba 1348ms/67.5GB).
- [ ] **14A-8** — **Rewrite `analysis/h100_scaling_results.md` §1 and §3** with the real baseline, and delete the §3 "no attention baseline in this repo" caveat once it is false.

**PRE-REGISTERED SUCCESS BAR FOR 14A — declared 2026-09-07, BEFORE any of it is run.** (Project discipline: the bar is fixed while the outcome is unknown.)
- **Quality — "matches":** the Transformer does **not** beat the hybrid by more than the 95% bootstrap CI on the difference, on CheXbert-14-micro **and** ROUGE-L (official test split, n=2663). "Beats" = hybrid ahead with the CI excluding zero.
- **Efficiency — "at better efficiency":** hybrid strictly better on training peak memory **and** on latency at L ≥ 2048, at matched params.
- **PRE-REGISTERED FAILURE STATEMENT:** if the Transformer beats the hybrid on quality at equal parameters, the central claim gets **rewritten**, not quietly dropped — to the efficiency claim alone ("trades *X* quality for *Y* efficiency at matched parameters"), with the quality gap stated numerically in the abstract. Writing this down now is the point; it is not renegotiable after the numbers land.

---

#### 14B — Re-measure boilerplate/template rate on the final (13D) checkpoint ⏳ **NOT STARTED — cheap, do FIRST**

**Cost: ~zero.** The 13D official-test-split generations already exist on the cluster (`results/report_gen_tower13d_test_split/hyps.txt`, from the 13D eval) — **no GPU, no regeneration**. Same for the retrieval-NN floor's outputs and the references.

- [x] **14B-1** — **DONE 2026-09-07.** `scripts/analyze_generation_diversity.py` (stdlib-only so it runs in the cluster eval venv; exact-duplicate clustering, distinct-1..4, TTR, sampled self-BLEU-4, and a built-in pre-registered verdict that names which of the three declared outcomes fired). Smoke-tested on a synthetic 75%-templated corpus: recovered 74.0% and correctly separated it from a 15% control. Originally specced as: New `scripts/analyze_generation_diversity.py` (CPU, stdlib + existing deps). Reads a `hyps.txt` and reports: exact-duplicate cluster count, % of generations inside a duplicate cluster, the largest cluster sizes, distinct-1/2/3/4, self-BLEU-4, and type-token ratio. Unit tests + `validate_for_willi.sh`.
- [x] **14B-2** — **DONE 2026-09-07 (job 2516811, gx17v1, 4 s wall). Ran on three corpora.**

  | corpus (n=2663, official test split) | unique | dup. clusters | % in a dup. cluster | largest | distinct-2 | distinct-4 | self-BLEU-4 | mean tokens |
  |---|---|---|---|---|---|---|---|---|
  | **generated (13D)** | 2138 | 252 | **29.2%** | 29,10,10,10,10 | 0.0299 | 0.0841 | 0.6854 | 58.2 |
  | references (human) | 2660 | 2 | **0.2%** | 3,2 | 0.2162 | 0.6439 | 0.2974 | 72.3 |
  | retrieval-NN (real reports) | 2561 | 92 | **7.3%** | 4,3,3,3,3 | 0.2116 | 0.6164 | 0.3046 | 64.0 |

  **The controls paid for themselves.** Two findings the bare percentage would have hidden:
  1. **The reference corpus is barely duplicated at all (0.2%)** — this plan's standing claim that "MIMIC reports are heavily templated" is true at the *phrase* level (references' own distinct-2 is only 0.216) but **false at the whole-report level**. §1.2 of the writeup has been corrected.
  2. **The retrieval-NN baseline's 7.3% is a property of retrieval, not of the corpus** — the same gallery report gets returned for several different queries. That is the right reference point for "what a perfect non-generative system scores", and it is 7.3%, not 0%.

  **The exact-duplicate metric flatters the generator.** 29.2% looks like a large win, but on lexical diversity the generator is **~7× less varied** than either control (distinct-2 0.030 vs 0.216/0.212; distinct-4 0.084 vs 0.644/0.616) and **2.3× more self-similar** (self-BLEU-4 0.685 vs 0.297/0.305), and writes shorter reports (58.2 vs 72.3 tokens). Near-duplicates differing by one token do not register as exact duplicates but are still boilerplate — that is where the remaining gap lives, and it is why the diversity columns were included rather than the cluster count alone.

  ⚠️ **Comparability caveat, stated because the headline invites the error:** the 73.6% was measured on `validate.parquet` (n=1433); this is the official test split (n=2663). The −44.4pp change is therefore **split-confounded**. The *direction* is safe — duplicate-cluster rate rises with n, all else equal, so 29.2% on the larger set understates the improvement — but the clean number needs 14B-3.

  ~~Original spec:~~ **Run it on three corpora, not one.** This is the substantive fix to how the number was originally reported:
  1. **13D generations** (the checkpoint under test),
  2. **the reference reports** for the same n=2663 — MIMIC-CXR reports are *themselves* heavily templated, and the plan already measured ~2% exact duplication in the retrieval gallery (6C-3),
  3. **the retrieval-NN baseline's outputs** — these are *real human reports*, so whatever duplication rate they show is the rate a "perfect" non-generative system exhibits.

  The 73.6% figure was reported **with no control**, which makes it uninterpretable on its own. A generator at 73.6% against references at 60% is a very different finding from one against references at 5%.
  ⚠️ **Must go through `sbatch`** — the aisc login node refuses ANY script execution ("This command is not allowed on the login node!", confirmed live in Phase 7E), so `scripts/analyze_diversity_h100.sh` (CPU-only, no GPU, ~20 min wall cap, queue wait dominates) is the entry point. It fail-fasts on a missing input file and warns loudly if either control is omitted.
  ```bash
  HYPS=results/report_gen_tower13d_test_split/hyps.txt \
  REFS=results/report_gen_tower13d_test_split/refs.txt \
  BASELINE=results/retrieval_floor_test_split/hyps.txt \
  OUTPUT=analysis/generation_diversity_13d.md \
    sbatch scripts/analyze_diversity_h100.sh
  ```
- [x] **14B-3** — **DONE 2026-09-07 (job 2516815, 3 s wall). Like-for-like on `validate.parquet` (n=1433), the split the historical 73.6% came from.**

  | corpus (n=1433, validate) | unique | dup. clusters | % in a dup. cluster | largest | distinct-2 | distinct-4 | self-BLEU-4 | mean tokens |
  |---|---|---|---|---|---|---|---|---|
  | **generated (13D)** | 1053 | 137 | **36.1%** | 47,29,29,13,13 | 0.0446 | 0.1121 | 0.7073 | 58.4 |
  | references (human) | 1418 | 10 | **1.7%** | 5,3,3,2,2 | 0.2624 | 0.6681 | 0.3343 | 60.0 |
  | retrieval-NN (real reports) | 1380 | 39 | **6.4%** | 10,6,3,3,2 | 0.2608 | 0.6420 | 0.3549 | 58.1 |

  **CLEAN HEADLINE: 73.6% → 36.1%, −37.5pp, same split, same protocol.** This supersedes the split-confounded −44.4pp from 14B-2 as the number to quote. Both splits are subject-disjoint by construction (official `mimic-cxr-2.0.0-split.csv.gz`), so neither figure is a leakage artifact.

  ⚠️ **A prediction made in 14B-2 was WRONG, and this run falsified it.** 14B-2 argued the direction was safe because "duplicate-cluster rate rises with n, all else equal, so 29.2% on the larger set understates the improvement". Measured, duplication is **lower on the larger split**, for the generator (36.1% → 29.2%) *and* for the references (1.7% → 0.2%). The birthday-collision intuition does not dominate here — **split composition does**. Practical consequence: **always quote the split alongside a duplication rate**; these numbers are not portable between splits.

  **Second correction, to 14B-2's own writeup text:** 14B-2 concluded the generator "writes shorter reports" (58.2 vs 72.3 tokens on test). That is test-split-specific. The generator's length is essentially **constant** across splits (58.4 / 58.2) while the references' is not (60.0 / 72.3) — it matches reference length on validate and is ~19% short on test. The correct statement is that the model writes the same amount regardless of what the case calls for, which is a mild independent signal of the same formulaic behaviour — not that it is uniformly terse.

  The diversity story is unchanged and consistent across both splits: ~6× less varied (distinct-2 0.045 vs 0.262/0.261) and ~2.1× more self-similar (self-BLEU-4 0.707 vs 0.334/0.355). Pre-registered outcome fired **INTERMEDIATE** on this split too.

  ~~Original spec:~~ **A like-for-like re-run, cheap (one job, seconds of compute).** 14B-2 answered the question on the test split; the historical 73.6% is from `validate.parquet` (n=1433), so re-run there to remove the split confound:
  ```bash
  HYPS=results/report_gen_tower13d_n1433/hyps.txt \
  REFS=results/report_gen_tower13d_n1433/refs.txt \
  BASELINE=results/retrieval_floor_n1433/hyps.txt \
  OUTPUT=analysis/generation_diversity_13d_n1433.md \
    sbatch scripts/analyze_diversity_h100.sh
  ```
  ~~Original spec:~~ Compare against the pre-Phase-13 **73.6% / 184 clusters**. Phase 13 changed decode strategy (greedy→beam), decoder training length, and the image tower; any of the three could have moved templating in either direction. Beam search in particular is known to *increase* mode-seeking, so a rise is a live possibility and must be reported if found.

**PRE-REGISTERED INTERPRETATION — declared 2026-09-07:** if 13D is still ≥70% templated **and** materially above both controls, then §1's "beats the retrieval floor" headline gets an **explicit qualifier in the abstract**, not merely a bullet in §4 Limitations. If it is at or below the reference corpus's own rate, that is a genuine positive finding and should be stated as one.

---

#### 14C — Bound the selective-scan defect (and state the fix cost honestly) ⏳ **NOT STARTED**

**This defect is already audited — it just never reached this plan, the writeup, or a test.** `MAMBA3_INTEGRATION_PLAN.md` + `mamba3_integration_state.json` (audit 2026-08-16, all numbers CPU-reproduced in this repo at commit `7104902`, float64 sequential ground truth) contain finding **F3**, `verdict: CONFIRMED — more severe than reported`:

> **Mechanism:** where `A_cum[s]` underflows below the clamp while `A_cum_safe[s] = 1e-8`, the intra-chunk term becomes `h_intra[s] = A_cum[t] · (Bx[s]/1e-8) ≈ 0` — **the current token's own contribution to the state is annihilated.**

| Δ (per-step) | 0.001 | 0.01 | 0.1 | 0.3 | 0.705 | 1.0 |
|---|---|---|---|---|---|---|
| rel max err vs float64 reference | 6.2e-17 | 3e-16 | **0.539** | **0.719** | **1.053** | **1.089** |

At the model's actual (uninitialized) Δ distribution: **16.2% of channels hit the clamp, 29.7% exceed 1% error, overall rel-max-err 0.358.** The worst case is `norm_topology=hybrid` — i.e. the canonical config this project actually trains (audit F1: Δ mean 0.8229, Δ max 4.6536).

**Why the fix is not free (audit F4, and the reason the "or" branch is the right answer here).** The obvious repair — shrink the chunk so `A_cum` never underflows — *is* verified exact (rel err **1.5e-16** at chunk=8, vs 0.358 at chunk=64), **but only once Δ is properly initialized**; at the current uninitialized Δ≈0.705, *even chunk=2 fails* (rel err 0.398). The Δ initialization is a **training-time** change (the correct `_init_dt_proj` already exists as dead code at `mamba_block_v2.py:159-179`), and audit F2 found it is further erased by `dt_norm` on every v2 config unless that is also changed. So a real fix is `M1+M2` **coupled**, which invalidates every existing checkpoint and costs a full Stage-0 re-run (~4 GPU-days) plus the entire downstream chain — and would break comparability with every number in `final_verdict`. The mask-based exact form (Mamba-2/3 segsum) does **not** port: this is Mamba-1-style with `A` of shape `(d_inner, d_state)`, so the decay mask is `(cs,cs,d,n)` — **19.3 GB at chunk=64**, 1.2 GB even at chunk=16 (audit F4).

- [x] **14C-1** — **DONE 2026-09-07. The test the supervisor asked for.** `tests/test_scan_correctness.py` (20 tests, CPU, picked up automatically by `validate_for_willi.sh`) + `analysis/scan_error_bound.md`. **Independently reproduces the 2026-08-16 audit** on a fresh implementation: rel-max-err **0.734 @ Δ=0.3** (audit 0.719), **1.03 @ Δ=0.705** (audit 1.053), **1.08 @ Δ=1.0** (audit 1.089); 92% of chunk entries hit the clamp at the live model's Δ with chunk=64. Also asserts the *positive* half — the chunked scan is exact to fp32 rounding (3e-8…8e-8) wherever the clamp does not fire, so the chunking itself is sound and only the clamped division is not. Original spec:  New `tests/test_scan_correctness.py`: an exact float64 sequential reference recurrence, compared against `selective_scan_parallel` over Δ ∈ {1e-3, 1e-2, 0.1, 0.3, 0.705, 1.0} × A ∈ {−1 … −16} × chunk ∈ {4, 8, 16, 32, 64}. **Assert the documented error envelope, not `<1e-6`** — the operator is known-defective, so the test's job is to be a *regression guard on a measured bound* that fails loudly if the deviation ever grows. CPU-only; wire into `scripts/validate_for_willi.sh`. Emit the table to `analysis/scan_error_bound.md` so the number is citable from the writeup.
- [ ] **14C-2** — **Bound it at the trained checkpoint's real Δ, not just at init.** The audit measured Δ at initialization; a reviewer wants the deviation for the *system as reported*. Hook the live `dt` tensors of the 13D decoder on a real MIMIC batch, dump the empirical Δ distribution, and report the per-channel error distribution + the fraction of channels affected under it. This is a CPU/1-GPU job over one batch.
- [ ] **14C-3** — **Measure the end-to-end effect — the highest-value item in 14C.** ⚠️ **REDESIGNED 2026-09-07: the original `chunk_size=8` design was wrong, and 14C-1 is what caught it.** That design assumed shrinking the chunk gives a clean reference. It does not at this model's Δ: measured rel-max-err at Δ=0.705 is **0.67 at chunk=8** and still **0.43 at chunk=4** — the audit's "chunk=8 → 1.5e-16" figure was quoted under the *proposed* Δ init, which this model does not have. A shrunken chunk is therefore not a reference, just a differently-wrong operator.
  **Corrected design — compare against the *exact* operator instead.** `HYBRID_EXACT_SCAN=1` (shipped 2026-09-07, `scan_interface.selective_scan()`) swaps in `selective_scan_sequential_reference`, the exact float64 recurrence. It is **off by default**, so the ordinary path stays byte-identical to every published run — the 14A operator freeze holds, and a test asserts it. Being O(L) sequential it is far too slow for training, and too slow for a full beam-decoded n=2663 sweep, so run it in two parts:
  - **14C-3a (cheap, full test split, high sensitivity):** teacher-forced LM loss / perplexity on all n=2663, exact vs default. One forward pass per sample, no autoregressive loop. If the next-token distribution is unchanged, that is strong evidence the defect is not doing damage where it would show up first.
  - **14C-3b (the metric that matters, subsample):** beam-decoded generation + CheXbert on n≈300–500 with `HYBRID_EXACT_SCAN=1`, against the same subsample under the default operator, judged against the 14A-6 bootstrap CI. A subsample is the honest cost/sensitivity trade here; report the CI, and do not claim more precision than n supports.
  Together these answer *"does the bug affect your reported numbers?"* by **measurement, not argument** — and unlike the original design, against a reference that is actually exact.
- [ ] **14C-4** — **Record the decision either way. Recommendation: do NOT apply the fix to the trained system** (rationale above: coupled fix, ~4 GPU-days, invalidates every checkpoint and all comparability, and directly conflicts with the 14A operator freeze). Ship instead the statement that survives peer review:
  > *The chunk-parallel scan does not compute the specified state-space recurrence exactly: for `A_cum[s] < 1e-8` the clamp annihilates the token's own state contribution. Training and evaluation used the same operator throughout, so all reported numbers are valid measurements of the system as built. The deviation from the specified recurrence is bounded at **[14C-1 table]**, its magnitude under the trained model's own Δ distribution is **[14C-2]**, and its end-to-end effect on the headline metrics is measured at **[14C-3]**.*

  The full operator repair remains owned by `MAMBA3_INTEGRATION_PLAN.md` (phases M1 + M2, currently `current_phase: M0_pin_the_defect`, `status: PLANNED — no code written`). Phase 14 does **not** activate it. Add `analysis/scan_error_bound.md` to `.gitignore`'s allowlist.
- [x] **14C-5** — **DONE 2026-09-07. Corrected the `CLAUDE.md` claim.** It states Mamba "uses chunk-parallel selective scan Triton kernel"; the audit confirmed `scan_interface.selective_scan()` unconditionally calls the **PyTorch** `selective_scan_parallel` and `selective_scan_triton` is imported but never invoked. This is false for the live path and also bears on how the 14A-7 efficiency curves must be described (they measure the PyTorch scan, not a Triton kernel).

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

**Phase 14 (2026-09-07) — blocking-ish, decide before the noted item:**
- 14A-2: Transformer LR — copy hybrid's 4e-4 verbatim (single lever), or 2-arm probe {4e-4, 6e-4}? Default: verbatim; probe only if it loses on quality. **Decide before 14A-3.**
- 14A-4: reuse 13D's hybrid-co-trained image tower (default, single lever) or retrain one per backbone (+7-8h, removes the confound)? **Decide before 14A-5.**
- 14C-4: accept "bound + report" (recommended), or authorize the coupled M1+M2 operator fix (~4 GPU-days, invalidates every checkpoint, breaks all comparability)? **Decide before writeup.**
- 14A-1: is a KV cache needed before 14A-7, or is uncached O(L²) decode an acceptable (and fair) depiction of attention cost? Measure first.
- 14A: run the optional depth-matched 12L/162M secondary, or param-matched 15L only?

**Pre-existing:**
- PhysioNet credentialing lead time — unknown until 7B is submitted; **everything downstream is gated on it**. Start Phase 10A/10B/11A meanwhile.
- Report-gen text target: **findings-only** (RRG convention) vs findings+impression (what the retrieval chapter used)? Decide at 8F, record in `build_report.json`.
- Prefix length `k` for image conditioning — sweep {8,32,64} at 10B; no prior.
- Cross-attention (10B alternative) — deferred unless prefix conditioning underperforms; it means new per-layer modules and Triton work.
- 8D hash-join recall vs the legacy gallery — if < 95%, the `train[90%:]` continuity number is dropped and the official split becomes the sole metric.
- Where the ~310–400 GB fetch can actually run (7E) — login node under tmux vs transfer node vs egress-capable partition.
- CheXbert labeler weights offline-staging on aisc (11B).
- CheXpert/VinDr label→prompt template wording — only if 9G is revived.
- Whether willi/A100 remains a target after H100 migration (if retired, drop py3.9 guards + `validate_for_willi.sh`).
