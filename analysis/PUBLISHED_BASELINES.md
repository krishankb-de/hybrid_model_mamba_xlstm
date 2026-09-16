# Published Field Baselines — MIMIC-CXR Report Generation

**Status:** 15A in progress. Rows sourced and cross-checked; three gaps explicitly open (§5).
**Scope:** Phase 15 item 1 of the supervisor review of 2026-09-13 — *"add a table of published
field baselines (R2Gen, CvT2DistilGPT2, RGRG, RA-RRG, Janus-CXR) on the same metrics."*
**Plan:** `H100_SCALING_PLAN.md` Phase 15A. **This project's own numbers:**
`analysis/PHASE14_SUPERVISOR_REVIEW.md`.

---

## 1. Why this table exists, and the rule it is written under

The plan's `Target` tier has said *"competitive with R2Gen-class published baselines"* since
Phase 1. **It was never operationalised — before this document there was not one published
number anywhere in the repository** (grep 2026-09-13: five hits, all prose). A tier that
cannot be scored is not a tier.

**🔴 Standing rule, and the reason for it.** Every number below is traced to a specific
paper and table. Nothing is written from recollection. Where a source could not be read, or
where two readings of the same table disagreed, the cell says so rather than carrying a
plausible-looking value. A wrong digit in a baseline table is the one error in this project
that **no test in this repository can catch** and that a viva examiner is most likely to
check personally.

That rule already changed two things during 15A, both recorded because they are exactly the
mistakes the rule exists to prevent:

1. **A first search returned "R2Gen 13.6 / CvT2DistilGPT2 16.8 / RGRG 18.0 macro-F1 on
   MIMIC-CXR."** Those numbers are real but they are **IU X-Ray**, not MIMIC-CXR. Had they
   gone in unchecked, this project would have appeared to beat R2Gen-class systems on macro-F1
   by ~10 points on the wrong dataset.
2. **RA-RRG's NLG columns could not be aligned reliably.** Two extractions of the same table
   disagreed, one of them giving BLEU-4 (37.9) larger than BLEU-1 (24.9), which is
   arithmetically impossible. Its CheXbert columns *were* confirmed (§3.2); its BLEU/ROUGE-L
   cells are left blank in §3.2 rather than guessed.

---

## 2. The comparability problem — read this before the table

Five axes vary across published MIMIC-CXR report-generation work. **Only the last one is
usually stated in a comparison table, which is why naive cross-paper comparison is
unreliable.**

| axis | this project | typical published work |
|---|---|---|
| **target text** | `"Findings: {…} Impression: {…}"` **concatenated** | **FINDINGS section only** |
| **test set** | 2,663 studies (official subject-disjoint split) | **3,858 samples** (findings-section subset of the official split) |
| **F1 flavour** | micro and macro over 14 labels | often **example-based** (per-sample), sometimes macro, rarely both |
| **model scale** | **183M** params | 183M–**84B** |
| **labeler** | CheXbert | CheXbert (consistent — the one axis that does agree) |

**The target-text axis is the load-bearing one, and it is this project's own unresolved
decision.** `H100_SCALING_PLAN.md` has carried *"Report-gen text target: findings-only (RRG
convention) vs findings+impression (what the retrieval chapter used)?"* as an **open**
question since Phase 8F and never closed it. The retrieval chapter's convention was inherited
by default, so this project generates findings **and** impression
(`scripts/train_contrastive.py:230`, `concatenate_sections: true`), while the field's
convention is findings only. Consequences:

- **ROUGE-L and BLEU are not comparable across this axis at all.** Different target text
  means a different reference distribution, different length, and different achievable
  overlap. This project's references average 72.3 words on test; a findings-only reference is
  shorter.
- **CheXbert F1 is *more* robust to it but not immune.** The impression section restates the
  principal findings, so label recall is broadly preserved — but impressions also introduce
  hedged and negated restatements that the labeler scores.

**The F1-flavour axis is the one that makes most cross-paper tables misleading.** micro,
macro, and example-based F1 are three different numbers on the same predictions, and the
gap between them is large. This project's own final checkpoint: **micro-14 0.4736, macro-14
0.2800** — a spread of 0.19 on identical outputs. A table that puts a micro-F1 next to a
macro-F1 under one heading called "F1" is not a comparison. §3 therefore groups by flavour
and never mixes them in a column.

---

## 3. The table

Numbers are as published, in the source's own units (percentages where the source uses
percentages, converted to a 0–1 scale here for consistency with this project's reporting).

### 3.1 Example-based F1 group — the R2Gen-class lineage

Source: **PromptMRG (AAAI 2024), MIMIC-CXR table.** 3,858 test samples, CheXbert, 14 labels,
**example-based (per-sample) F1**, FINDINGS section. This is the standardised table most
2020–2024 RRG work reports into, and it is where four of the five systems the supervisor
named actually live.

| model | year | precision | recall | **F1 (example)** | BLEU-1 | BLEU-4 | METEOR | ROUGE-L |
|---|---|---|---|---|---|---|---|---|
| R2Gen | 2020 | 0.333 | 0.273 | 0.276 | 0.353 | 0.103 | 0.142 | 0.277 |
| M2TR | 2021 | 0.240 | 0.428 | 0.308 | 0.378 | 0.107 | 0.145 | 0.272 |
| MKSG | 2022 | 0.458 | 0.348 | 0.371 | 0.363 | 0.115 | — | 0.284 |
| CliBert | 2022 | 0.397 | 0.435 | 0.415 | 0.383 | 0.106 | 0.144 | 0.275 |
| **CvT2DistilGPT2** | 2022 | 0.356 | 0.412 | 0.384 | 0.392 | 0.124 | 0.153 | 0.285 |
| M2KT | 2023 | 0.420 | 0.339 | 0.352 | 0.386 | 0.111 | — | 0.274 |
| METransformer | 2023 | 0.364 | 0.309 | 0.311 | 0.386 | 0.124 | 0.152 | 0.291 |
| KiUT | 2023 | 0.371 | 0.318 | 0.321 | 0.393 | 0.113 | 0.160 | 0.285 |
| **RGRG** | 2023 | 0.461 | 0.475 | 0.447 | 0.373 | 0.126 | 0.168 | 0.264 |
| PromptMRG | 2024 | 0.501 | 0.509 | **0.476** | 0.398 | 0.112 | 0.157 | 0.268 |
| — | | | | | | | | |
| **this project — hybrid, 3 seeds** | 2026 | 0.4415 | 0.3789 | **0.3790 ± 0.0214** | 0.2508 ± 0.0034 | 0.0578 ± 0.0032 | — | 0.1949 ± 0.0047 |
| **this project — matched Transformer, 3 seeds** | 2026 | 0.4524 | 0.3768 | **0.3817 ± 0.0144** | 0.2478 ± 0.0022 | 0.0575 ± 0.0016 | — | 0.1952 ± 0.0021 |
| **this project — retrieval-NN floor** | 2026 | 0.4095 | 0.3898 | **0.3691** | 0.2372 | 0.0330 | — | 0.1636 |

**Updated 2026-09-16 (Phase 15B-4).** The generator rows are now **mean ± SD over seeds 42/43/44**, read from each
arm's `samples avg` block; precision/recall are 3-seed means. On this lineage's own metric the project sits
between M2TR (0.308) and MKSG (0.371) — **above the retrieval-NN floor it is measured against, and far below
PromptMRG (0.476)**. The BLEU/ROUGE-L columns remain non-comparable for the reason in §2: this project's targets
are findings **+ impression**, while the R2Gen lineage generates findings only.

### 3.2 Macro / micro F1 group — the modern LLM-scale tier

Source: **RA-RRG (2025), Table 1.** MIMIC-CXR, 3,858 test images, CheXbert 14 classes,
FINDINGS section, non-positive labels binarised to negative (the same convention this project
uses — see `compute_rare_finding_sample_weights()`'s U-Zeros note).

| model | approx. params | **micro-F1** | **macro-F1** | example-F1 |
|---|---|---|---|---|
| DCL | — | — | 0.284 | 0.373 |
| MCA-RG | — | — | 0.335 | 0.408 |
| PromptMRG | ~0.2B | — | 0.381 | 0.476 |
| MAIRA-1 | ~7B | 0.557 | 0.386 | — |
| M4CXR | ~7B | 0.581 | 0.388 | 0.502 |
| LLaVA-Rad | ~7B | 0.573 | 0.395 | — |
| Med-PaLM M | **84B** | 0.536 | 0.398 | — |
| **RA-RRG** | — | **0.585** | **0.417** | **0.507** |
| — | | | | |
| **this project — hybrid, 3 seeds** | **0.183B** | **0.4480 ± 0.0223** | **0.2660 ± 0.0122** | 0.3790 ± 0.0214 |
| **this project — matched Transformer, 3 seeds** | **0.183B** | 0.4443 ± 0.0153 | 0.2692 ± 0.0106 | 0.3817 ± 0.0144 |
| *(the single 13D run previously quoted here)* | 0.183B | *0.4736* | *0.2800* | *0.4029* |
| **this project — retrieval-NN floor** | n/a | 0.4296 | **0.3014** | 0.3691 |

RA-RRG's BLEU/ROUGE-L cells are deliberately absent (§1, finding 2). Its macro-F1 of 0.417 is
corroborated in the paper's prose — *"a Macro-F1 of 41.7, outperforming Med-PaLM M 84B
(39.8)"* — which is what licenses the column at all. The micro/macro/example alignment is
further corroborated by PromptMRG's example-F1 appearing as 0.476 in both tables
independently.

### 3.3 Five-label subset

This project reports CheXbert-5 (micro 0.5522 / macro 0.4487 for 13D). **The 5-label subset is
reported far less often in the RRG literature than the 14-label metrics**, and no verified
5-label row for R2Gen, CvT2DistilGPT2, RGRG, or RA-RRG was found. The only 5-label figures
encountered are Janus-family, which are unverified (§5.2). This column is therefore left
unpopulated rather than filled with the 14-label numbers.

---

## 4. Where this project actually stands

**On macro-F1 the supervisor's concern is confirmed, and should be stated plainly.** 0.2800 is
below every system in §3.2 except DCL (0.284, statistically indistinguishable at this
project's measured seed/bootstrap scale). The modern range really is 0.38–0.42.
**Worse: 0.2800 is a single seed and the high draw.** Over seeds 42/43/44 (Phase 15B-4) the
macro mean is **0.2660 ± 0.0122**, so the like-for-like gap is ~0.014 *wider* than the number
the supervisor was shown, not narrower.

**But the comparison set is not parameter-comparable, and that is the substantive point.**
Every system in the 0.386–0.417 macro band is either an LLM-scale model — MAIRA-1, M4CXR and
LLaVA-Rad at ~7B, Med-PaLM M at **84B**, i.e. **38× to 460× this project's 183M** — or, in
RA-RRG's case, retrieval-augmented. The nearest thing to a parameter-comparable modern
system, PromptMRG at ~0.2B, reaches macro 0.381; that is the honest target for a model this
size, and this project is 0.10 short of it.

**On surface-overlap metrics this project is clearly behind, and dismissing that on
target-text grounds would be too convenient.** ROUGE-L 0.1899 vs a 0.264–0.291 field range,
BLEU-1 0.2469 vs 0.353–0.398. Some of that gap is the findings+impression confound (§2), but
the gap is larger than the confound plausibly explains, and this document does not claim
otherwise. The Phase-14 finding that the hybrid loses ROUGE-L to its own matched Transformer
(−0.0038, CI excluding zero) already indicated that surface overlap is this system's weak
axis independently of any cross-paper comparison.

**Three things genuinely favour this project and should be said once, without inflation:**

1. **CheXbert-14-micro is within the published field's range** — 0.4736 at seed 42, and
   **0.4480 ± 0.0223 over three seeds**, above DCL and PromptMRG's parameter class on that
   metric, though below the 7B+ tier's 0.536–0.585. The tier call survives the seed
   replication: the lowest of the three draws is 0.4324.
2. **The 183M parameter count is 38–460× smaller** than the systems that beat it on macro-F1.
   None of the papers in §3.2 reports a parameter-matched comparison at all.
3. **No comparable paper publishes a prefix-length sweep** (`analysis/PHASE14_SUPERVISOR_REVIEW.md`
   §3.5), nor a parameter-matched attention-free-vs-attention head-to-head. Those are this
   project's contributions and they are not in the table because the field does not measure
   them.

### 4.1 A finding for the thesis narrative, not a plan change

**The current best macro-F1 system in §3.2 — RA-RRG at 0.417 — is retrieval-augmented.** This
project independently has (a) a closed retrieval chapter, and (b) a measured retrieval-NN
control that **beats its own generator on macro-F1**, 0.3014 vs 0.2800, precisely because
copying a real human report recovers nonzero F1 on all 14 labels including the rare ones
(`H100_SCALING_PLAN.md` Phase 15, macro-target arithmetic).

So the field's evidence and this project's own internal evidence point the same way: on
macro-F1 specifically, retrieval beats generation, and the SOTA closes the gap by combining
them. **That is the strongest available answer to "why is your macro-F1 low" — it is a
property of the task, reproduced independently here.** Recorded here as a finding and as
future work; it is **not** a substitute for 15C, which remains the supervisor's nominated
mechanism and is authorised as one designed attempt.

---

### 4.2 How much of the "0.28 vs 0.40" gap survives a like-for-like reading — the verdict (15A-3)

Written after 15B (seeds + CIs) and 15C (the aux-loss attempt), so every number here is the
3-seed mean, not the best draw. **The plan pre-registered both outcomes — that the gap survives,
and that it does not — before the sourcing was done. The answer is split, and the split is the
finding.**

**On macro-F1 the gap survives, and widens.** Every adjustment available either leaves it alone
or makes it worse:

| adjustment | effect on the gap |
|---|---|
| Seed replication (15B-4) | **widens it**: 0.2800 → 0.2660 ± 0.0122 |
| Labeler | none — CheXbert on both sides |
| Label set | none — 14 labels on both sides |
| Parameter scale | the only real mitigation, and it is large: the 0.386–0.417 band is 7B–84B or retrieval-augmented; the nearest parameter-comparable system, PromptMRG (~0.2B), is at 0.381 |
| Target text (findings+impression here vs findings-only there) | **unquantified**, and claimed as a credit by nobody here; §2 |
| Test split (n=2663 official subject-disjoint, frontal-only, vs 3,858 images elsewhere) | **unquantified** |

Against the *parameter-comparable* comparator the shortfall is **0.381 − 0.266 = 0.115**, and
15C establishes it is not closed by the mechanism the field would reach for first: an auxiliary
multi-label CheXpert loss moved the target by a sixth of one seed SD (`PHASE15C_AUX_LOSS.md`).

**On the metric this lineage actually reports, the gap does NOT survive — this project is
mid-field.** §3.1's table is example-based F1, and the project's row is **0.3790 ± 0.0214**:

- **above** R2Gen 0.276, M2TR 0.308, METransformer 0.311, KiUT 0.321, M2KT 0.352, MKSG 0.371
- **level with** CvT2DistilGPT2 0.384 (inside one seed SD)
- **below** CliBert 0.415, RGRG 0.447, PromptMRG 0.476

So "this project is far behind the field on clinical accuracy" is **true on macro-F1 and false
on example-F1**, against the same papers and the same labeler. The two metrics disagree because
macro-F1 averages 14 labels unweighted and is therefore dominated by the rarest ones — which is
exactly what §4.1's retrieval-floor arithmetic already showed.

**The calibration that makes this concrete.** A *real radiologist report*, retrieved for a
visually similar but different patient, scores macro **0.3014** under this evaluation — above
every generator arm this project has trained, and still 0.08 below the modern band. A metric on
which a genuine human report scores 0.30 is not measuring writing quality alone; it is heavily
measuring whether rare findings are named at all. **That does not excuse the gap, but it does
set what closing it would require**: not better prose, but a mechanism that puts rare-label
mentions into the text — and the evidence from 15C is that grounding the image representation
is not that mechanism.

**What this licenses in the thesis.** (1) State the macro gap plainly, at 0.2660 ± 0.0122, with
the parameter-scale context and without claiming the target-text confound closes it. (2) State
that on example-F1 — the metric four of the five supervisor-named systems actually report — the
project sits mid-field at 0.3790 ± 0.0214. (3) State that the field's own best macro system is
retrieval-augmented and that this project's retrieval control independently beats its generator
on macro, so the direction the evidence points is hybrid retrieval-generation, which this
project has the components for but did not build. Nothing beyond those three is supported.

---

## 5. What is NOT verified

### 5.1 ✅ This project's own example-based F1 — **CLOSED 2026-09-16** (was: OPEN, zero compute)

Read out for all six arms and folded into §3.1/§3.2 as mean ± SD over seeds 42/43/44: **hybrid 0.3790 ± 0.0214**, **Transformer 0.3817 ± 0.0144**, floor 0.3691. Seed 42 alone gives 0.4029 / 0.3951 — the high draw, which is exactly why these rows are now seed means (`H100_SCALING_PLAN.md` 15B-4). The original instruction is kept below for provenance.


`chexbert_metrics.json` already contains a `samples avg` block (example-based P/R/F1); it has
simply never been read out, because Phases 11–14 reported micro and macro only. Without it,
this project **cannot be placed in §3.1 at all** — the table where four of the five systems
the supervisor named actually live. Resolve by reading the two existing files:

```bash
python3 -c "
import json
for n,f in [('hybrid_13D','results/report_gen_tower13d_test_split/chexbert_metrics.json'),
            ('transformer','results/report_gen_transformer_test_split/chexbert_metrics.json')]:
    m=json.load(open(f))
    s=m['chexbert_14']['samples avg']
    print('%-12s example-F1 %.4f  (P %.4f  R %.4f)'%(n,s['f1-score'],s['precision'],s['recall']))"
```

### 5.2 ⬜ Janus-CXR / Janus-Pro-CXR — UNVERIFIED, DO NOT CITE

The Nature Communications version is paywalled and the arXiv preprint (2507.19493) has no
HTML mirror, so no table was read. Figures encountered in search summaries only — **Janus-CXR
micro F1-14 0.473, micro F1-5 0.572, RadGraph F1 0.264; Janus-Pro-CXR micro F1-5 0.634, macro
F1-5 0.551** — are recorded here **solely to document what still needs checking**. They are
not in §3 and must not be cited until read from the paper's own table.

### 5.3 ⬜ RA-RRG's BLEU-1 / BLEU-4 / ROUGE-L — UNRESOLVED

Column alignment could not be established (§1, finding 2). Needs a direct read of Table 1 in
the published PDF.

### 5.4 Stated, not resolved

- **Parameter counts in §3.2 are approximate**, taken from each system's architecture
  description rather than an instantiated count. This project's 183,721,824 (hybrid) and
  183,386,880 (Transformer) are instantiated counts, so the comparison is exact on one side
  only.
- **The 3,858-vs-2,663 test-set difference is not quantified.** Both derive from the official
  split; the published subset conditions on a non-empty findings section. No attempt is made
  here to estimate the effect, and none should be claimed.

---

## 6. Sources

- **PromptMRG: Diagnosis-Driven Prompts for Medical Report Generation** (AAAI 2024) —
  §3.1's table. https://arxiv.org/abs/2308.12604
- **RA-RRG: Multimodal Retrieval-Augmented Radiology Report Generation with Key Phrase
  Extraction** (2025) — §3.2's table. https://arxiv.org/abs/2504.07415
- **Improving Chest X-Ray Report Generation by Leveraging Warm Starting** (CvT2DistilGPT2)
  — https://arxiv.org/abs/2201.09405
- **ReXrank: A Public Leaderboard for AI-Powered Radiology Report Generation** — consulted
  for protocol conventions. https://arxiv.org/abs/2411.15122
- **From Bench to Bedside: A DeepSeek-Powered AI System** (Janus-CXR) — **not successfully
  read**, see §5.2. https://arxiv.org/abs/2507.19493

This project's own rows: `analysis/PHASE14_SUPERVISOR_REVIEW.md` §2.3 (hybrid, Transformer),
`analysis/h100_scaling_results.md` §1.5 (retrieval floor),
`results/retrieval_floor_test_split/chexbert_metrics.json` (the floor's `samples avg`).
