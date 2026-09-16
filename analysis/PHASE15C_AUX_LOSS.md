# Phase 15C — Auxiliary multi-label CheXpert loss: a designed attempt that failed

**Status: NEGATIVE RESULT, reported in full.** Written 2026-09-16. Primary deliverable for
supervisor review item 2 of 2026-09-13 (*"Close the CheXbert-14 macro F1 gap … a multi-label
CheXpert-class loss is the right next attempt"*).

This document reports one designed attempt at the CheXbert-14 macro-F1 gap, the pre-registered
rule that stopped it, and what the failure rules out. It is **not** a record of iterating until
something worked — the stopping rule was written down before any arm ran, and it fired.

---

## 1. What was tried, and why that mechanism

| | |
|---|---|
| **Mechanism** | A 14-way linear head on the **mean-pooled image prefix** (`prefix_embeds.mean(1)`, 768→14, 10,752 params), trained with `BCEWithLogitsLoss` against the study's CheXpert-14 vector. |
| **Targets** | `mimic-cxr-2.0.0-chexpert.csv.gz`, **U-Zeros** convention (1.0 → positive; {0.0, −1.0, NaN} → negative), the same convention Phase 13F used, so the two attempts stay directly comparable. Coverage is 100% of both splits. |
| **Class imbalance** | `pos_weight = min((N − n_pos)/n_pos, 10)`, computed on the **train split only**. The cap binds on the six rarest labels; common labels land at 2.0–4.1 (Lung Opacity 3.43, Cardiomegaly 4.11, No Finding 2.03). |
| **Total loss** | `lm_loss + λ · aux_loss`. λ is the **only** lever that moves between arms. |
| **Doses** | λ ∈ {0.1, 0.5}, seed 42, 12000 steps, 4×H100, `prefix_k=32` — otherwise 13D's recipe exactly. |
| **At evaluation** | The head is **discarded**. `evaluate_report_generation.py` loads `strict=False`, so `aux_head.*` lands in `unexpected`; a unit test asserts `missing == []`. The evaluated network is parameter-identical to 13D, so Phase 14A's matched-parameter claim survives intact. |

**Why the image prefix, and not the decoder's pooled state.** Under teacher forcing the decoder's
pooled representation already contains the report, so an auxiliary label task attached there is
solvable from the text alone and exerts no grounding pressure on the image pathway. The prefix is
the component trained by the LM loss *alone*, so a finding that rarely appears in text sends
almost no gradient into the connector — a concrete, nameable bottleneck. It is also the component
this project's own most robust finding identifies as dominant
(`PHASE14_SUPERVISOR_REVIEW.md` §3.5: the conditioning interface matters more than the
sequence-mixing architecture; the k=8→32 swing exceeds every architecture difference measured at
matched k).

---

## 2. The pre-registered rule, and the fact that it fired

Declared in `H100_SCALING_PLAN.md` **before** any arm ran:

> The λ with the higher CheXbert-14-macro **on validate** wins; ties broken by 14-micro. If
> *neither* λ beats 13D's validate macro, that is the reported answer and 15C-4 does not run.

Baseline is **13D itself** on the same split, decode and seed — a paired comparison, not a
re-implementation.

| metric (validate, n=1433) | 13D baseline | λ=0.1 | Δ | λ=0.5 | Δ |
|---|---|---|---|---|---|
| **CheXbert-14-macro** (the target) | **0.2869** | 0.2853 | **−0.0016** | 0.2712 | **−0.0157** |
| CheXbert-14-micro | 0.4595 | 0.4534 | −0.0061 | 0.4460 | −0.0135 |
| CheXbert-5-micro | 0.5255 | 0.5179 | −0.0076 | 0.5026 | −0.0229 |
| CheXbert-5-macro | 0.4180 | 0.4241 | +0.0061 | 0.3971 | −0.0209 |
| exact-match-5 | 0.3859 | 0.3720 | −0.0139 | 0.3740 | −0.0119 |
| example-F1 | 0.3876 | 0.3826 | −0.0050 | 0.3771 | −0.0105 |

**Neither λ wins.** λ=0.1 is a wash: −0.0016 is an eighth of the training-seed SD measured in
15B-4 (0.0122). λ=0.5 is worse on every metric, and the **dose–response is monotone downward**,
which is evidence against the "merely under-dosed" reading — the natural next move (raise λ) is
the one the data argues against.

**15C-4 (three seeds of the winning λ) therefore did not run.** Roughly 7 GPU-hours were returned
to a budget that ends when the cluster account expires.

---

## 3. Test-split confirmation (for the record; no selection performed)

λ=0.1 was decoded and scored once on the official test split, paired with 13D at the same seed.
Nothing was selected on these numbers — the rule had already stopped 15C — but reporting the
attempt only on the split it was tuned against would be its own small dishonesty.

95% paired-bootstrap CIs, 1000 resamples, n=2663 (`analysis/bootstrap_aux_lam0.1_vs_13d_test.md`):

| metric | aux λ=0.1 | 13D | diff | 95% CI | verdict |
|---|---|---|---|---|---|
| ROUGE-L | 0.1874 | 0.1899 | −0.0024 | [−0.0048, −0.0003] | **13D wins** |
| BLEU-1 | 0.2443 | 0.2469 | −0.0026 | [−0.0050, −0.0003] | **13D wins** |
| BLEU-4 | 0.0526 | 0.0542 | −0.0015 | [−0.0038, +0.0006] | tie |
| **CheXbert-14-macro** | 0.2820 | 0.2800 | **+0.0020** | [−0.0072, +0.0110] | tie |
| CheXbert-14-micro | 0.4715 | 0.4736 | −0.0021 | [−0.0104, +0.0051] | tie |
| CheXbert-5-micro | 0.5495 | 0.5522 | −0.0027 | [−0.0132, +0.0074] | tie |
| CheXbert-5-macro | 0.4473 | 0.4487 | −0.0014 | [−0.0134, +0.0099] | tie |
| exact-match-14 | 0.0379 | 0.0349 | +0.0030 | [−0.0045, +0.0105] | tie |
| exact-match-5 | 0.2287 | 0.2163 | +0.0124 | [−0.0019, +0.0252] | tie |

**The target metric moves +0.0020 on test and −0.0016 on validate — opposite signs, each about a
sixth of one training-seed SD.** Two splits agreeing on "nothing happened" is a stronger negative
than either alone. The only significant effects are a small *cost*: ROUGE-L and BLEU-1.

---

## 4. Per-label: the mechanism does the intended thing, ~10× too weakly

Per-label CheXbert-14 F1 on test, sorted by effect size, with the same paired CIs:

| label | support | aux λ=0.1 | 13D | diff | verdict |
|---|---|---|---|---|---|
| Pneumonia | 549 | 0.1975 | 0.1705 | **+0.0270** | tie [−0.0049, +0.0607] |
| Pneumothorax | 84 | 0.0444 | 0.0211 | **+0.0234** | tie [−0.0508, +0.1025] |
| Pleural Other | 111 | 0.0000 | 0.0172 | −0.0172 | tie |
| **Support Devices** | 1192 | 0.6944 | 0.7097 | **−0.0153** | **13D wins** [−0.0304, −0.0004] |
| Lung Lesion | 177 | 0.0109 | 0.0000 | +0.0109 | tie [+0.0000, +0.0345] |
| Fracture | 135 | 0.0321 | 0.0214 | +0.0107 | tie |
| Lung Opacity | 1027 | 0.3005 | 0.2974 | +0.0031 | tie |
| *(Edema, Enl. Card., Atelectasis, Cardiomegaly, Pleural Effusion, Consolidation, No Finding)* | | | | ≤ 0.0071 | tie |

**Read the direction, then the magnitude.** Four of the five largest positive moves are exactly the
rare labels the `pos_weight` cap targets — Pneumonia, Pneumothorax, Lung Lesion, Fracture — and the
one label that moves significantly is a **common** one going *down* (Support Devices, the
highest-support label in the set). That is the intended mechanism, visible in the sign pattern:
weight the rare labels, and mentions shift from common findings towards rare ones.

It is simply far too weak. The rare gains are +0.01 to +0.03 against a macro deficit of −0.044
versus the retrieval floor (15B-4), they do not survive their own CIs, and the common-label loss
cancels most of the gain: net macro +0.0020.

**At λ=0.5, the same mechanism overshoots into 13F's failure shape.** On validate, Lung Lesion and
Pleural Other finally move off zero (0.018 / 0.041) while **Consolidation and Fracture collapse to
0.000** and Lung Opacity — the largest single macro deficit versus the floor — drops 0.257 → 0.238.
Help the rare, hurt the rest. Phase 13F bought that trade with a `WeightedRandomSampler`; 15C buys
it with a loss term. **Two mechanically different interventions producing the same trade-off is the
substantive finding here**, and it is what makes this negative worth a section rather than a footnote.

---

## 5. What is ruled out, and what is not

**Ruled out:** mechanism A — an auxiliary multi-label CheXpert loss on the mean-pooled image prefix
— at λ ∈ {0.1, 0.5} with `pos_weight` cap 10, U-Zeros targets, a 150M decoder and `prefix_k=32`.
Within this budget it does not close the macro gap, and the larger dose is actively harmful.

**Not ruled out, and the honest "what I would try next" line:** *label-conditioned decoding*.
PromptMRG — the nearest parameter-comparable system at ~0.2B, macro 0.381 — does not use an
auxiliary loss. It puts predicted diagnoses **in the decoder's input** as prompts. The distinction
this failure sharpens is that **grounding a representation is not the same as changing what gets
written**: three independent pieces of evidence say the aux loss reshaped the connector without
reshaping the text.

1. Train LM loss is untouched (0.911 at λ=0.1 vs 0.913 at λ=0.5) — this is not an optimisation failure.
2. Text metrics tie at *both* doses on validate (λ=0.5: ROUGE-L +0.0005, BLEU-1 −0.0021, BLEU-4
   −0.0015, all CIs spanning zero) — so λ=0.5's CheXbert damage is not general degradation. The
   model writes equally well and labels worse.
3. The head learns its task while the two labels at maximum `pos_weight` stay at ~zero F1 in the
   generated text (Lung Lesion 0.0109, Pleural Other 0.0000 on test).

A model can be made to *represent* a finding without being made to *mention* it. The mention is
what CheXbert scores.

---

## 6. Why this negative is worth more than 13F's

13F (rare-finding oversampling) failed at two weights and could only report "non-monotonic across
two doses" — with a single seed and no seed band, effect and noise were not separable. 15C is
different in three ways that matter for a viva:

1. **A quantified noise floor exists.** 15B-4 measured the training-seed SD (CheXbert-14-macro
   0.2660 ± 0.0122 over seeds 42/43/44), so "smaller than seed variance" is a measurement, not an
   impression.
2. **The stopping rule was pre-registered and honoured.** The result is not "we tried until we ran
   out of time"; it is "we declared what would count as success, and it did not happen."
3. **The failure is mechanistically localised.** Not "it didn't work", but: the intervention moved
   the intended labels in the intended direction, an order of magnitude too weakly, and the dose
   that strengthens it trades common labels for rare ones without net gain.

---

## 7. Reproduction

```bash
# probe arms (jobs 2548850, 2548851)
for LAM in 0.1 0.5; do
  SEED=42 SAVE_TOP_K=0 AUX_LAMBDA=$LAM NUM_GPUS=4 MAX_STEPS=12000 \
  IMAGE_ENCODER_CKPT=./outputs/h100_kd_150m_v2_full_data_lr3e6/checkpoints/last.ckpt \
  EXPERIMENT=h100_report_gen_aux_lam${LAM}_seed42 \
    sbatch --gpus=4 scripts/train_report_generation_h100.sh
done
```

`AUX_LAMBDA=0.0` (the default) builds no head at all: no `aux_head.*` key enters the checkpoint and
the decoder/prefix_mapper initialisation stays bit-identical to a run from before this code existed
(the head is constructed **last**, because instantiating an `nn.Linear` consumes global-RNG draws).
Both properties are asserted in `tests/test_willi_parity.py`, which is what lets 15B-4's seed band
serve as this attempt's baseline.

| artefact | path / job |
|---|---|
| Probe training | jobs 2548850 (λ=0.1), 2548851 (λ=0.5); GPU smoke 2548455 |
| Validate decode / score | 2550145, 2550153 / 2550637, 2550638 |
| Test decode / score | 2550683 / 2550703 |
| Test bootstrap (per-label) | job 2551233 → `analysis/bootstrap_aux_lam0.1_vs_13d_test.md` |
| Validate bootstraps | jobs 2550663, 2550664 (text metrics only — 13D's validate dump predated the label-matrix dump; re-scored by job 2551261, so these can now be regenerated with CheXbert CIs) |
| Baseline dumps | `results/report_gen_tower13d_n1433` (validate), `results/report_gen_tower13d_test_split` (test) |
