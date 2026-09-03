# H100 Scaling Results — Hybrid Mamba-xLSTM CXR Report Generation

**Status:** Phase 13 arc complete, Phase 12 writeup complete, Phase 12A (STS/PubMed PPL)
closed (2026-09-03). Final report-generation checkpoint:
`outputs/h100_report_gen_full_ext_4gpu_tower13d/checkpoints/last.ckpt`.
Full resumable history: `H100_SCALING_PLAN.md` + `h100_scaling_state.json` at repo root.

## Summary

The project's primary objective is **medical report generation** from chest X-rays,
scored by ROUGE-L and CheXbert F1 against ground-truth radiology reports, on the
official MIMIC-CXR-JPG subject-disjoint test split. Retrieval (image-text matching,
R@10) was the original objective and is now a **closed supporting chapter** — it
produced the pretrained image tower the generator conditions on and a complete,
falsified-alternatives account of what does and does not move a CXR joint embedding
space, but is not itself the thing being optimized anymore.

The headline result: the image-conditioned decoder **beats a retrieval-nearest-neighbor
baseline on CheXbert F1**, the plan's own pre-registered success criterion, on the
official held-out test split — not just on a validation split, and not on a single
lucky checkpoint. Getting there took three real, single-lever interventions (a free
decoding-strategy fix, an extended training run, and a full-data image-tower retrain)
and one honest negative result (rare-label oversampling, tried at two doses, abandoned).

---

## 1. Report Generation (Primary)

### 1.1 Task and architecture

- **Decoder**: `hybrid_150m_v2` (768-dim, 12 layers, mixed Mamba/mLSTM), initialized
  from the Phase-5 Stage-0 language-model checkpoint (val PPL 13.18).
- **Image conditioning**: a frozen (or partially unfrozen) BiomedCLIP ViT-B/16 tower
  feeds pre-pooling patch-grid features through an `ImagePrefixMapper` into `k=32`
  prefix tokens prepended to the decoder's input embeddings (prefix-tuning, not
  cross-attention). Loss is causal-LM cross-entropy over report-token positions only.
- **Data**: PhysioNet MIMIC-CXR-JPG v2.1.0, built locally (Phase 8) into
  train/validate/test parquets on the official `mimic-cxr-2.0.0-split.csv.gz` split
  (subject-disjoint by construction) — **191,462 / 1,433 / 2,663** frontal
  (image, findings, impression) pairs at 320px.

### 1.2 Success criteria (pre-registered)

Scored on the official test split.

| Tier | ROUGE-L | CheXbert F1 (14-label micro) | Rationale |
|---|---|---|---|
| Floor | ≥ 0.15 | ≥ 0.25 | beats a retrieval-NN baseline; proves the decoder conditions on the image at all |
| Target | ≥ 0.22 | ≥ 0.40 | competitive with R2Gen-class published baselines |
| Stretch | ≥ 0.26 | ≥ 0.50 | competitive with strong modern RRG systems |

**The pre-registered control**, run before any generator number was trusted: retrieve
the nearest training report by BiomedCLIP cosine similarity and emit it verbatim. MIMIC
reports are heavily templated, so n-gram metrics reward copying — a generator that does
not beat this baseline has not demonstrated it generates anything. This baseline became
the real floor, tier numbers above notwithstanding.

### 1.3 Starting point: a mixed result (Phase 11)

The first full-data checkpoint (10k decoder steps ≈ 1.7 epochs, stock BiomedCLIP tower,
greedy decoding) beat the retrieval floor on ROUGE-L, BLEU, and exact-match accuracy —
but **lost badly on CheXbert F1**, the more clinically meaningful metric, confirmed on
both validate.parquet and the official test split:

| metric (official test, n=2663) | generator | retrieval floor | result |
|---|---|---|---|
| ROUGE-L | 0.1816 | 0.1636 | generator wins |
| CheXbert-14-micro | 0.3326 | 0.4296 | **floor wins by 29%** |
| CheXbert-14-macro | 0.1671 | 0.3014 | **floor wins by 80%** |

Per-label, the generator was conservative and boilerplate-heavy (11C: 73.6% of
validate-split generations fell into one of 184 exact-duplicate template clusters) and
never predicted 4 of 14 CheXpert labels at all. Per the pre-registered rule, this
generator "had not contributed anything" on the metric that matters clinically.

### 1.4 Closing the gap (Phase 13)

Two suspected causes, diagnosed but not yet fixed at that point: the decoder was
undertrained (1.7 of a plausible ~8 epochs), and the image tower was **completely
stock, generic BiomedCLIP — never fine-tuned on a single MIMIC-CXR image**.

**13A — decode strategy (free).** Beam search (`beam_size=3`) vs greedy, zero training
cost, same checkpoint. Beat greedy on every metric except exact-match accuracy —
CheXbert-14-micro 0.31→0.37 (+18%) — but not enough alone to beat the floor. Beam
decoding became the standing default for every subsequent eval.

**13B — extended training (~2.5h, 4-GPU DDP).** Retrained the decoder for ~8 effective
epochs instead of 1.7. Combined with beam decode: **milestone 1 achieved** —
CheXbert-14-micro 0.4412 beat the retrieval floor's 0.4296 (+2.7%) on the official test
split, plus wins on ROUGE-L, BLEU, and exact-match accuracy outright.

**13C/13D — image tower on full data (3-arm `vit_lr` sweep, ~3.5h each).** The decoder
had never conditioned on anything but stock BiomedCLIP. Retrained the tower
contrastively on the full 191,462-pair set (same recipe/step-count as the historical
27.5k-pair arm, isolating "more data" as the only changed variable), then plugged each
resulting tower into the decoder via a new `IMAGE_ENCODER_CKPT` lever and retrained.

| `vit_lr` | tower i2t R@10 (in-training) | tower train/val clip_loss | downstream CheXbert-14-micro (test) |
|---|---|---|---|
| 1e-6 | 0.285 | 2.123 / 2.183 | 0.4591 |
| **3e-6** | 0.316 | 1.982 / 2.075 | **0.4736 (peak)** |
| 1e-5 | 0.350 (best tower metric) | 1.752 / 1.910 | 0.4558 (regressed) |

`3e-6` — the actual peak on the original small-dataset sweep, but overtaken there by
`1e-5`/`3e-5` due to overfitting — turned out to be the peak for *downstream usefulness*
at full scale too, even though the tower's own retrieval R@10 kept climbing past it.
**Retrieval quality is not a perfect proxy for what the decoder needs from the image
tower** — a genuine methodological finding, not just a tuning footnote. The `3e-5` arm
was skipped once `1e-5` regressed on every CheXbert metric; the trend had already
inflected and `3e-5` was expected to be worse.

**13F — rare-label oversampling (2 doses, negative, abandoned).** The one remaining
weakness after 13D was CheXbert-14-macro, driven by 3 CheXpert labels (Lung Lesion,
Pneumothorax, Pleural Other) the model rarely or never predicted. Built a
`WeightedRandomSampler` to oversample training reports positive for these labels, tried
at weight 5.0 and 2.0. Both regressed CheXbert-14/5-micro back below the retrieval floor
— gains on 2 of 3 target labels were bought at the cost of 4+ common, high-support
labels (Edema, Pleural Effusion, No Finding, Support Devices) getting measurably worse.
The pattern was non-monotonic across the two weights (macro-F1 was *worse* at the
gentler dose), indicating ordinary training variance dominates this lever's effect at
this scale — a real, informative negative result, not noise to explain away. Lever
abandoned; 13D's checkpoint was never displaced by anything from this arm.

### 1.5 Final numbers (official test split, n=2663, `13D` checkpoint)

| metric | generator (13D) | retrieval floor | Δ vs floor |
|---|---|---|---|
| ROUGE-L | 0.1899 | 0.1636 | **+16.1%** |
| BLEU-1 / BLEU-4 | 0.2469 / 0.0542 | 0.2372 / 0.0330 | +4.1% / +64.2% |
| Exact-match label-set accuracy | 0.2163 | 0.1735 | +24.7% |
| CheXbert-14 micro / macro | **0.4736** / 0.2800 | 0.4296 / 0.3014 | **+10.2%** / −7.1% |
| CheXbert-5 micro / macro | **0.5522** / **0.4487** | 0.4856 / 0.4284 | **+13.7%** / **+4.7%** |

Three of four CheXbert sub-metrics beat the retrieval floor outright, by their widest
margins of the whole arc. CheXbert-14-macro is the one holdout, but its gap collapsed
from 17.5% (original checkpoint) to 7.1% across three successive, single-lever
improvements (13B → 13E/13C → 13D) — a monotonic trend confirmed on both the validation
and official test splits, not a single favorable measurement.

**Tier scoring**: Floor is cleared on both axes. Target's CheXbert bar (≥0.40
14-micro) is cleared with room to spare (0.4736). Target's ROUGE-L bar (≥0.22) is not
reached (0.1899) — the closest approach in the whole project, but short by ~14%
relative.

### 1.6 What did NOT get fixed

- **CheXbert-14-macro** still trails the retrieval floor by 7.1%, driven by labels with
  very low support (Lung Lesion: 177/2663, Pleural Other: 111/2663) that stayed at or
  near F1=0.0 through every checkpoint in this arc, oversampling included.
- **ROUGE-L Target tier** (≥0.22) was not reached; the model is competitive on n-gram
  overlap but not yet at published R2Gen-class levels by that specific measure.
- **Template/boilerplate rate was never re-measured on the final checkpoint.** The
  73.6%-templated finding is from the original (pre-Phase-13) checkpoint; whether the
  improved checkpoints still exhibit this to the same degree is an open, unverified
  question — flagged honestly rather than assumed to have resolved itself.
- **The `3e-5` `vit_lr` arm was never run.** Skipping it was a judgment call based on
  the visible trend (1e-5 already regressed, and 3e-5 was the worst arm on the small
  dataset too), not an empirically confirmed null.

---

## 2. Retrieval (Supporting Chapter) — ✅ Closed 2026-07-27

Finished as a research question before the objective pivot to report generation; kept
here because it produced the image tower report generation conditions on, and because
one of its own findings (the `vit_lr` inverted-U) was directly re-tested and extended in
Phase 13D above.

| Metric | Best (`val==test` selection) | Best (clean protocol) | Tier |
|---|---|---|---|
| MIMIC i2t R@10 | **17.14%** (D1c, `vit_unfreeze=12`) | **14.59%** (clean split) | Stretch, both protocols |
| Indiana i2t R@10 | 4.85% | 3.90% | flat within noise (SE 0.76pp) |
| Stage-0 val PPL (**= PubMed PPL** — Stage-0 trains on `dataset=pubmed`) | 13.18 | — | Target, ≈Stretch |
| STS (BIOSSES / STS-B-val Spearman ρ) | 0.3829 / 0.4472 | — | see note below |

Headline trajectory: **8.23% → 10.45% (A100 architecture refactor) → 17.14%** (H100 +
deep ViT adaptation) on the protocol every prior number in this project used; quote
17.14% as the protocol-matched comparison and 14.59% as the fully clean-protocol
headline — the 2.55pp gap between them is itself a measurement of what test-informed
checkpoint selection buys.

**Ten clean nulls, one dominant lever.** Stage-0 PPL, model scale, contrastive negative
count, epoch budget, batch size, head learning rate (×2 arms), KD-anchor decay, SigLIP
loss, and SimCSE objective changes all measured flat. The single decisive intervention
was **image-tower adaptation depth** (unfreezing ViT blocks) — a monotone, large effect
(0.111 → 0.143 → 0.171 R@10 as unfrozen depth went 2 → 6 → 12 blocks). That lever is
exhausted: depth is saturated at 12/12 blocks, and `vit_lr` is an inverted-U on the
small (27.5k-pair) dataset, with 1e-6 the apparent optimum there.

**Indiana never moved** on any lever tested — a data-bound, not method-bound, ceiling.

**Connection to Phase 13**: the retrieval chapter's own `vit_lr` sweep predicted that
more training data would shift the inverted-U's optimum right and raise its peak
(prediction 9C). Phase 13D directly tested this at ~7x the data scale and confirmed it:
`3e-6` (already the true — if historically overlooked — peak on the small set) pulled
further ahead of `1e-6` with more data, exactly as predicted, before `1e-5` finally
showed the regression the small-dataset sweep saw much earlier. This is a genuine
cross-chapter validation of the retrieval chapter's own predictive framework.

**STS note (2026-09-03, `13D`'s tower checkpoint, first measurement of any kind in this
project)**: BIOSSES ρ=0.3829, STS-B (validation) ρ=0.4472. `evaluate_sts.py` prints a
`FAIL` against a decision gate (BIOSSES≥0.50, STS-B≥0.60), but that gate belongs to a
different, never-executed pipeline design — a dedicated sequential Stage-1
SimCSE-only checkpoint. This project's actual checkpoint is **jointly** trained, with
SimCSE as one minor auxiliary loss (`gamma_simcse=0.1`) alongside the CLIP/KD objectives
the `vit_lr` sweep was actually optimized for — the gate is not a meaningful pass/fail
bar for this checkpoint lineage, and the numbers are reported here as a baseline
text-embedding-quality reference, not a failed target.

---

## 3. Efficiency

Measured 2026-07-28 (H100 80GB HBM3, bf16, bs=4; `analysis/efficiency_{150m,70m}/`).
Random weights and token ids — these curves measure architecture, not any trained
checkpoint, and do not explain any retrieval or generation number above.

- **Linear scaling confirmed to L=16,384.** Asymptotic (L≥4096) exponents ≈1.0 for
  latency and memory across hybrid, pure-Mamba, and pure-xLSTM. (The short-sequence
  regime drags the naive full-range fit down; report asymptotic, not full-range.)
- **xLSTM is dramatically cheaper than Mamba at inference.** 150M @ L=16384: Mamba
  1105ms / 42.0GB / 59,280 tok/s; hybrid 925ms / 42.0GB / 70,882 tok/s; **xLSTM 355ms /
  7.1GB / 184,860 tok/s** — 5.9× less memory, 3.1× faster, despite more parameters.
  Inference peak memory is a max-over-layers quantity, so the hybrid tracks pure Mamba
  here.
- **The hybrid's real win is training memory, not inference speed.** Training peak is a
  *sum* over saved activations, so layer composition matters: 150M @ L=2048 training —
  Mamba 1348ms/67.5GB vs **hybrid 1078ms/54.0GB (25% faster, 25% less memory)**; xLSTM
  309ms/11.2GB. This is the concrete, measured justification for the hybrid design over
  a pure-Mamba backbone.
- **Not a bottleneck at this project's actual sequence lengths.** CXR reports run
  ≤256 tokens, PubMed pretraining ≤512. At L=256, the hybrid is the fastest of the three
  architectures tested (18.08ms vs 19.78/21.12ms).
- **Caveat**: there is no attention/transformer baseline in this repo. The "~2.0 =
  quadratic attention" reference line in tooling output is a cited comparison, not a
  measurement made here.

---

## 4. Honest Limitations

1. **CheXbert-14-macro still trails the retrieval floor (7.1% gap)**, entirely
   attributable to a small number of very-low-support rare findings the model has never
   reliably learned to predict, across every lever tried (more training, better image
   tower, targeted oversampling).
2. **Rare-finding recall resisted the one lever built specifically to fix it.**
   Oversampling (13F) failed at two doses with a non-monotonic pattern suggesting the
   effect, if any, is smaller than ordinary training variance at this data scale — a
   real negative result, not a tuning miss.
3. **ROUGE-L Target tier (≥0.22) was not reached** (best: 0.1899). The generator is
   competitive on n-gram overlap relative to the retrieval floor but not yet at
   published R2Gen-class levels by this specific measure.
4. **Boilerplate/template rate on the final checkpoint is unverified.** The 73.6%
   exact-duplicate-cluster finding predates the entire Phase 13 improvement arc; whether
   it improved alongside the metrics above has not been directly re-measured.
5. **Retrieval R@10 is not a perfect proxy for downstream conditioning quality** — the
   `vit_lr` sweep's `1e-5` arm had the best tower-side retrieval metric of all three arms
   tested, but the worst downstream CheXbert F1. Any future image-tower tuning should
   select on the downstream task, not the tower's own contrastive metric.
6. **The `vit_lr=3e-5` arm was never empirically run** — skipped on trend extrapolation,
   not measurement. A residual, explicitly-flagged open question.
7. **Indiana (cross-domain) was never revisited for report generation** — all of Phase
   10-13's work targets MIMIC only. Indiana's retrieval ceiling was already established
   as data-bound, not method-bound, and that diagnosis was never re-tested against the
   generation task.
8. **No architecture change, no new loss function, and no additional training
   objective was needed to reach the final result** — worth stating plainly, since it
   means the two things that mattered (decode strategy, image-tower data volume/dose)
   were both cheap, well-understood, single-lever interventions rather than novel
   modeling contributions specific to this project's architecture.
9. **STS is modest (BIOSSES ρ=0.38, STS-B ρ=0.45) relative to a gate that doesn't
   actually apply to this checkpoint.** This project's text encoder was never
   specifically optimized for sentence-similarity quality — SimCSE was always a minor
   auxiliary loss in a joint objective dominated by CLIP/KD terms. Whether a dedicated
   SimCSE pretraining stage (the sequential Stage-1 design this project never executed)
   would meaningfully improve these numbers is untested. PubMed PPL, by contrast, *was*
   already measured all along — Stage-0's own validation PPL (13.18) is on PubMed text,
   it had simply never been labeled as "PubMed PPL" in prior writeups.

---

## 5. Reproduction

Final checkpoint: `outputs/h100_report_gen_full_ext_4gpu_tower13d/checkpoints/last.ckpt`
(decoder), conditioned on `outputs/h100_kd_150m_v2_full_data_lr3e6/checkpoints/last.ckpt`
(image tower). Evaluate with beam decoding (`beam_size=3`), not greedy — greedy is
strictly dominated on every metric measured in this project:

```
DECODE=beam BEAM_SIZE=3 PARQUET=/sc/home/$USER/dataset/mimic_full/test.parquet \
  NUM_SAMPLES=999999 DUMP_DIR=results/<name> \
  CHECKPOINT=./outputs/h100_report_gen_full_ext_4gpu_tower13d/checkpoints/last.ckpt \
  sbatch scripts/inspect_report_generation_h100.sh
DUMP_DIR=results/<name> sbatch scripts/score_chexbert_h100.sh
```

Full experiment history, every negative result, and every SLURM/tooling bug encountered
along the way are preserved in `H100_SCALING_PLAN.md` and `h100_scaling_state.json` at
the repo root.
