# Phase 14 — Supervisor Review: Results and Evidence

**Status:** 14B and 14C complete. 14A complete on the test split; one validation-split
confirmation outstanding (§6.1).
**Scope:** the three items raised in the supervisor review of 2026-09-07, and a detailed
account of where the hybrid architecture wins and where it does not.
**Primary sources:** `H100_SCALING_PLAN.md` (Phase 14), `h100_scaling_state.json`
(`phases.phase14_supervisor_review`). Every number below is reproducible from the commands
in §8.

---

## 1. Executive summary

Three items were raised. All three are answered with measurements rather than argument.

| # | Item | Outcome |
|---|---|---|
| **14A** | No parameter-matched Transformer baseline existed | **Built and run.** Result is a **split decision**: the hybrid significantly wins 3 of 4 CheXbert F1 metrics; the Transformer significantly wins ROUGE-L and exact-match accuracy; 3 metrics tie. All with 95% paired-bootstrap CIs. |
| **14B** | Boilerplate rate never re-measured on the final checkpoint | **Measured, with controls.** 73.6% → **36.1%** like-for-like. The generator is no longer mostly emitting templates, but remains more formulaic than both the reference corpus and the retrieval baseline. |
| **14C** | Scan correctness defect neither fixed nor bounded | **Bounded, tested, and committed.** rel-max-err **≈1.03** at the model's operating point; regression test in `validate_for_willi.sh`; citable table in `analysis/scan_error_bound.md`. Not fixed — the repair is coupled to a training-time change costing a full retraining chain (§5.3). |

**The claim the evidence supports:**

> At matched parameters (183,386,880 vs 183,721,824) and matched image-prefix length
> (k=32 — a value at which the hybrid is at its measured optimum and the Transformer is
> statistically indifferent), the attention-free hybrid produces **clinically more accurate**
> reports — significantly higher CheXbert F1 on 3 of 4 variants — while the Transformer
> produces text with significantly higher **surface overlap**, at roughly **5× the hybrid's
> training throughput**.

Every clause in that sentence has an interval behind it. Nothing beyond it is licensed.

---

## 2. Item 14A — the parameter-matched Transformer baseline

### 2.1 What was missing and why it mattered

The project's thesis is *"attention-free hybrid matches or beats attention-based
transformers at better efficiency."* Before Phase 14 every comparison in the writeup was
against the project's own architecture variants (pure-Mamba, pure-xLSTM), an off-the-shelf
non-fine-tuned model (BiomedCLIP zero-shot), or a nearest-neighbour retrieval control.
**None of those is an attention-based transformer.** §3 of `h100_scaling_results.md` had
conceded this in a caveat; the supervisor's point was that a conceded caveat is not a
substitute for the experiment.

### 2.2 How the baseline was built

Registered `"attention"` as a fourth `layer_pattern` type inside the existing
`HybridLanguageModel`, so `layer_pattern: ["attention"]` *is* a pure Transformer decoder.
Consequence: the baseline traverses the **identical** pipeline — same embeddings, same MLP,
same `ImagePrefixMapper` conditioning, same trainer, same beam search, same CheXbert
scorer. Nothing downstream differs, and nothing downstream *can* accidentally differ.

| | hybrid_150m_v2 | transformer_150m_baseline |
|---|---|---|
| parameters | 183,721,824 | **183,386,880** (−0.18%) |
| dim / layers / heads | 768 / 12 / 12 | 768 / **15** / 12 |
| mixer | `[m,m,m,m,L,L,L,m,m,m,m,m]` | `[attention]` |
| positional encoding | none (recurrence is ordered) | **RoPE** (no parameters) |
| norm topology | HybridNorm | `pre_rms` (canonical) |

**Fairness decisions, each deliberate:**

- **Depth 15, not 12.** Parameter-matching holds parameters constant and lets each
  architecture pick its own shape. Matching at 12 layers would need `mlp_ratio=5.5` — a
  non-standard FFN width that reads as a rigged baseline. A 12-layer/r=4.0 Transformer is
  only 162.1M (−11.7%), which would handicap it.
- **RoPE, not learned position embeddings.** The hybrid spends *exactly zero* parameters on
  positional encoding; a learned table would hand the baseline +0.79M it never got.
- **Every shared hyperparameter copied verbatim** — lr 4.0e-4, warmup 2000, weight decay
  0.1, dropout 0.1, grad clip, batch, steps, KD teacher. Only `MODEL_CONFIG` changed at
  every stage. The image tower (13D, `vit_lr=3e-6`) was reused unchanged for both.
- **`cu_seqlens` doc-boundary masking implemented for attention.** Stage-0 packs multiple
  documents per sequence and the hybrid resets state at each boundary. Without the
  equivalent mask, the baseline would read context the hybrid provably cannot — a silent,
  uncontrolled advantage.

### 2.3 The head-to-head (official test split, n=2663, beam_size=3, k=32)

1000 paired bootstrap resamples. Paired because both systems generated for the same
studies, so each resample draws one index set and scores both on it — removing the
"some studies are just harder" variance that would otherwise swamp effects this size.

| metric | hybrid | transformer | diff | 95% CI | verdict |
|---|---|---|---|---|---|
| ROUGE-L | 0.1899 | 0.1936 | −0.0038 | [−0.0066, −0.0010] | **transformer** |
| BLEU-1 | 0.2469 | 0.2496 | −0.0027 | [−0.0057, +0.0002] | tie |
| BLEU-4 | 0.0542 | 0.0571 | −0.0029 | [−0.0058, +0.0001] | tie |
| **CheXbert-14-micro** | **0.4736** | 0.4590 | **+0.0146** | **[+0.0052, +0.0240]** | **hybrid** |
| CheXbert-14-macro | 0.2800 | 0.2774 | +0.0026 | [−0.0080, +0.0141] | tie |
| **CheXbert-5-micro** | **0.5522** | 0.5249 | **+0.0273** | **[+0.0147, +0.0410]** | **hybrid** |
| **CheXbert-5-macro** | **0.4487** | 0.4319 | **+0.0187** | **[+0.0033, +0.0308]** | **hybrid** |
| exact-match accuracy (14-label) | 0.0349 | 0.0469 | −0.0120 | [−0.0210, −0.0034] | **transformer** |

**Hybrid wins 3. Transformer wins 2. Three ties.** Both beat the retrieval-NN floor.

### 2.4 Where the Transformer wins outright

Two comparisons went to the baseline without qualification, and both are reported here
rather than buried:

- **Stage-0 language modelling.** PubMed validation perplexity **11.222 vs 13.18** — a
  14.9% relative improvement for attention, under a verified single-lever comparison (same
  corpus, 120k steps, effective batch 48, LR schedule, KD teacher; only `MODEL_CONFIG`
  differed).
- **Efficiency, at every sequence length tested** — see §4, including the large
  implementation confound that makes that result *not yet* an architecture claim.

### 2.5 The pre-registered bar, scored honestly

Declared 2026-09-07, before any of this ran:

> *"the Transformer does NOT beat the hybrid by more than the 95% bootstrap CI on the
> difference, on CheXbert-14-micro **AND** ROUGE-L."*

- **CheXbert-14-micro half: passed decisively.** The hybrid does not merely avoid losing
  it — it *wins* it, CI excluding zero.
- **ROUGE-L half: failed.** The gap is real, CI [−0.0066, −0.0010].

**As a conjunction the bar is NOT cleared.** The pre-registered failure statement therefore
applies and the central claim is rewritten. It is **not** rewritten to "efficiency only" —
that clause assumed a quality loss that did not occur, and the hybrid won three quality
metrics. It is rewritten to the trade-off the data shows (§1).

---

## 3. Where the hybrid wins — detailed

### 3.1 The axis of the split is meaningful, not arbitrary

CheXbert F1 asks **did the report assert the right findings**. ROUGE-L and BLEU ask **does
the text look like the reference**. The hybrid wins the former family and loses the latter.

**This framing is not post-hoc.** `H100_SCALING_PLAN.md` has described CheXbert F1 as *"the
more clinically meaningful metric"* since **Phase 11, 2026-08-30** — written when the hybrid
was **losing** on it to a retrieval baseline by 29% relative. The designation was made when
it was inconvenient, which is what makes it usable now. The git history is the evidence, and
it is worth pointing a reviewer at it.

### 3.2 The margins, in relative terms

| metric | absolute gain | relative | what it measures |
|---|---|---|---|
| CheXbert-5-micro | +0.0273 | **+5.2%** | the five clinically weightiest findings: cardiomegaly, edema, consolidation, atelectasis, pleural effusion |
| CheXbert-5-macro | +0.0187 | **+4.3%** | the same five, unweighted by prevalence |
| CheXbert-14-micro | +0.0146 | **+3.2%** | all fourteen CheXpert labels, prevalence-weighted |

The largest and most robust win is on the **5-label subset** — the findings that drive
clinical decisions. It is also the win that survives every framing tested (§3.4).

Per-label breakdowns for both systems are in
`results/report_gen_tower13d_test_split/chexbert_metrics.json` and
`results/report_gen_transformer_test_split/chexbert_metrics.json`.

### 3.3 Both systems beat the pre-registered floor

The plan's standing rule: *"a generator that does not beat its own retrieval baseline has
not contributed anything."* The retrieval-NN baseline retrieves the most visually similar
training image and copies that patient's **real** report — a strong control, not a strawman;
its own CheXbert-14-micro (0.4296) would clear the plan's Target tier.

| | hybrid | transformer | retrieval floor |
|---|---|---|---|
| ROUGE-L | 0.1899 | 0.1936 | 0.1636 |
| CheXbert-14-micro | 0.4736 | 0.4590 | 0.4296 |

Both clear it. The hybrid clears it by the wider margin on CheXbert.

### 3.4 Why k=32 is a principled choice, not a favourable one

The image-prefix length `k` was fixed at 32 in Phase 10 with *"no prior"* and never tuned
for either model. Phase 14 swept it, and the result initially looked alarming — the headline
appeared to **flip** with k. It does not, and the reason is measured:

**CheXbert-14-micro across the sweep:**

| | k=8 | k=32 | k=64 |
|---|---|---|---|
| **hybrid** | 0.4481 | **0.4736** | 0.4488 |
| transformer | — | 0.4590 | 0.4637 |

- The hybrid has a **significant preference for k=32 over both neighbours** — +0.0255
  [+0.0163, +0.0352] vs k=8, and +0.0248 [+0.0160, +0.0344] vs k=64. An optimum **bracketed
  on both sides**.
- The Transformer is **statistically indifferent** between k=32 and k=64: all nine metrics
  tie, not one CI excludes zero. Its apparent k=64 gain (+0.0047) is noise.

**Therefore k=32 disadvantages neither model** — it is the hybrid's measured optimum and a
point indistinguishable from the Transformer's own best. Reporting there is not
cherry-picking; it is the only defensible shared value.

**The apparent flip was mis-diagnosed.** At k=64 the Transformer wins *only because the
hybrid degraded off its optimum*. The Transformer itself did not move at all. The correct
reading is not "the result is fragile to k" but "the hybrid has a genuine optimum that must
be used, and the Transformer does not care."

### 3.5 The finding that survives every framing

> **The conditioning interface matters more than the sequence-mixing architecture.**

The hybrid's k=8→k=32 CheXbert-14-micro swing is **+0.0255**. Every hybrid-vs-Transformer
difference at matched k is *smaller* (+0.0146 at k=32; −0.0149 at k=64). **How the image
reaches the decoder dominates whether that decoder mixes tokens by attention or by a
selective scan.**

This is robust to every framing choice above — unlike the architecture claim, which is
conditional on k. It is also novel for this literature: no comparable report-generation
paper publishes a prefix-length sweep.

---

## 4. Efficiency — and the confound that qualifies it

Measured on H100 80GB, bf16, bs=4, random weights (these curves measure architecture, not
any trained checkpoint).

| | hybrid | transformer | |
|---|---|---|---|
| inference L=256 (**the task's length**) | 24.72 ms / 1.22 GB | **5.03 ms / 0.51 GB** | 4.9× faster |
| inference L=16384 | 919 ms / 42.0 GB | **164 ms / 7.15 GB** | 5.6× faster, 5.9× lighter |
| training L=2048 | 1084 ms / 53.96 GB | **52.6 ms / 7.56 GB** | 20.6× faster, 7.1× lighter |
| training L=4096 | **OOM** | 107 ms / 14.7 GB | hybrid cannot run |

Measured inference-latency scaling exponents: hybrid **0.874**, transformer **0.865** —
indistinguishable, and nowhere near the 2.0 this project's tooling cited as "quadratic
attention". Independently confirmed on the real task: the report-gen decoder trained at
**7.96 it/s** (transformer) vs **1.51 it/s** (hybrid) — 5.3×, against 14A-7's synthetic
prediction of 5.1×.

### 4.1 Two causes, only one architectural

1. **The premise was out of date.** `F.scaled_dot_product_attention` dispatches to
   **FlashAttention**, which is **O(L) memory** — it never materialises the L×L matrix. The
   "quadratic attention" reference line describes *naive* attention, not shipped since 2022.
   Attention's FLOPs remain O(L²) and it shows at the tail (8192→16384: transformer ×2.30
   vs hybrid ×2.00), so the hybrid *is* asymptotically flatter — but from 5.6× behind, the
   crossover sits near L ≈ 10⁷–10⁸ tokens. Never, in practice.

2. 🔴 **The benchmark is unfair to the hybrid, entirely in the implementation.**
   `scan_interface.selective_scan()` unconditionally calls the **PyTorch** chunk-parallel
   scan in **fp32**; `selective_scan_triton` is imported and **never invoked**. This pits a
   fused bf16 CUDA FlashAttention kernel against a fp32 PyTorch loop. That is an
   *implementation* comparison, not an *architecture* one. The xLSTM row is the tell: 349 ms
   at L=16384 vs pure-Mamba's 1099 ms **with more parameters** — the difference between a
   path with a real kernel and one without.

**What can honestly be claimed today:** *as implemented in this repository*, the hybrid is
slower and more memory-hungry than a parameter-matched FlashAttention Transformer at every
sequence length tested, including the ≤256 tokens this task uses. The architectural
linear-vs-quadratic argument is **not falsified in principle** — the tail exponents still
favour the hybrid — but it is **not realised here**, and must be stated in those words.

---

## 5. Item 14B and 14C

### 5.1 14B — boilerplate / duplicate-template rate

Phase 11C found 1055/1433 (**73.6%**) of generations fell into one of 184 exact-duplicate
clusters, on the *pre-Phase-13* checkpoint, and it was never re-checked on 13D. The concern:
if the generator is mostly copying templates, "beats the retrieval baseline" is hollow,
since that is exactly what the retrieval baseline does.

Measured on both splits, **with the controls the original figure lacked**:

| corpus | validate (n=1433) | test (n=2663) | distinct-2 | self-BLEU-4 |
|---|---|---|---|---|
| **generated (13D)** | **36.1%** | 29.2% | 0.0446 / 0.0299 | 0.7073 / 0.6854 |
| references (human) | 1.7% | 0.2% | 0.2624 / 0.2162 | 0.3343 / 0.2974 |
| retrieval-NN (real reports) | 6.4% | 7.3% | 0.2608 / 0.2116 | 0.3549 / 0.3046 |

**Like-for-like: 73.6% → 36.1%** on the split the original figure came from. The specific
worry is **answered and negative** — the generator is not mostly emitting templates.

**But the exact-duplicate metric flatters it.** On lexical diversity the generator is ~6×
less varied than either control and ~2.1× more self-similar. Near-duplicates differing by a
token do not register as exact duplicates but are still boilerplate; that is where the
remaining gap lives.

Two incidental corrections the controls forced:
- The reference corpus duplicates at only **0.2–1.7%**, so this plan's standing claim that
  "MIMIC reports are heavily templated" is true at the *phrase* level but **false at the
  whole-report level**.
- The retrieval baseline's 6.4–7.3% is a property of **retrieval collisions** (the same
  gallery report returned for several queries), not of the corpus.

Pre-registered decision rule returned **INTERMEDIATE** on both splits: neither the
"qualifier in the abstract" trigger (≥70% *and* materially above every control) nor the
clean-positive trigger (at or below the controls) fired. Reported plainly, not rounded.

### 5.2 14C — the selective-scan correctness bound

The defect: `scan_interface.py:118` computes the intra-chunk term as
`A_cum[t] · cumsum(Bx[s] / A_cum[s].clamp(min=1e-8))`. Where `A_cum[s]` underflows past the
clamp, the divisor is too large and `Bx[s]` is under-weighted — and at `s == t` the token's
**own** contribution, which should enter with weight exactly 1, enters with weight
`A_cum[t]/1e-8 ≈ 0` and is **annihilated**.

Measured against an exact float64 sequential reference (`tests/test_scan_correctness.py`,
20 tests, CPU, runs inside `validate_for_willi.sh`; table in
`analysis/scan_error_bound.md`):

| Δ (per-step) | 0.001 | 0.01 | 0.1 | 0.3 | 0.705 | 1.0 |
|---|---|---|---|---|---|---|
| rel max err (chunk=64) | 3.6e-08 | 6.2e-08 | 0.281 | 0.734 | **1.031** | 1.076 |
| fraction of entries clamped | 0% | 0% | 54% | 81% | **92%** | 95% |

The live 150M model runs at Δ≈0.705 — **92% of chunk entries clamped, rel-max-err ≈1.03**.
This independently reproduces the 2026-08-16 audit (which found 0.719 / 1.053 / 1.089) on a
freshly written implementation.

The test also asserts the **positive** half: wherever the clamp does not fire, the chunked
scan is exact to fp32 rounding (3e-8…8e-8). **The chunking itself is sound; only the clamped
division is not.**

### 5.3 Why it was bounded rather than fixed

The obvious repair — shrink the chunk so `A_cum` never underflows — is verified exact
(1.5e-16 at chunk=8) **only once Δ is properly initialised**. At this model's uninitialised
Δ≈0.705, even chunk=2 fails. The Δ initialisation is a **training-time** change, so a real
fix invalidates every checkpoint and costs a full Stage-0 re-run (~23h) plus the entire
downstream chain — and would break comparability with every number in this document. The
mask-based exact form (Mamba-2/3 segsum) does not port: this is Mamba-1-style with `A` of
shape `(d_inner, d_state)`, so the decay mask is `(cs,cs,d,n)` — 19.3 GB at chunk=64.

**The statement that survives peer review:**

> The chunk-parallel scan does not compute the specified state-space recurrence exactly: for
> `A_cum[s] < 1e-8` the clamp annihilates the token's own state contribution. Training and
> evaluation used the same operator throughout, so all reported numbers are valid
> measurements of the system as built. The deviation is bounded at rel-max-err ≈1.03 at the
> model's operating point.

A `HYBRID_EXACT_SCAN=1` toggle ships (off by default; a test asserts the default path stays
bit-identical) so the end-to-end effect can be measured rather than argued. That measurement
(14C-2/14C-3) is **not yet run** — see §6.2.

---

## 6. Honest limitations

### 6.1 Outstanding: k selected on the test split

`k=32` was chosen using **test-split** CheXbert. The selection is far more defensible than a
bare pick — the optimum is large, significant, and bracketed on both sides, and the baseline
is provably indifferent to the same knob — but it is formally test-set selection.

Validate-split evaluations of the hybrid at k=8 and k=64 have been **run** (text metrics
below); **CheXbert scoring on those two dumps is the one step outstanding.** Completing it
converts *"selected on test"* into *"selected on validation, reported on test"*.

| hybrid, validate n=1433 | k=8 | k=32 (13D) | k=64 |
|---|---|---|---|
| ROUGE-L | 0.2153 | 0.2148 | 0.2170 |
| BLEU-1 | 0.2884 | 0.2914 | 0.2901 |
| BLEU-4 | 0.0714 | 0.0753 | 0.0750 |
| **CheXbert-14-micro** | **pending** | 0.4595 | **pending** |

### 6.2 Other known gaps

1. **The efficiency comparison is implementation-bound** (§4.1). Until a real Mamba kernel
   is wired in, it is not an architecture result. This is the single highest-value piece of
   remaining engineering.
2. **The scan defect's end-to-end effect is bounded but not measured** — the
   `HYBRID_EXACT_SCAN=1` probe (14C-2/14C-3) has not been run.
3. **The Transformer was swept at two k values, the hybrid at three.** Its CheXbert curve is
   flat between them, but k=128 was never tested; its "best" is formally a lower bound.
4. **The Transformer's LR was not tuned.** It inherited the hybrid's √-width-scaled 4.0e-4.
   If it had lost on quality this would be the first thing to re-test; it did not lose
   overall, but the asymmetry stands.
5. **The image tower was trained alongside the hybrid text encoder** and reused for both, so
   the prefix space is mildly hybrid-favouring. Removing this confound needs a per-backbone
   tower retrain.
6. **The generator remains ~6× less lexically diverse** than the corpus it models (§5.1).
7. **Rare findings remain near F1=0** — Lung Lesion, Pleural Other, Pneumothorax, Fracture.
   A dedicated oversampling attempt (13F) failed at two doses.
8. **Indiana (cross-domain) was never revisited** for report generation.
9. **`vit_lr=3e-5` was never run** — skipped on trend extrapolation, not measurement.

### 6.3 Process findings worth reporting

Four **silent wrong-configuration traps** were found during Phase 14, each of which loads a
mismatched checkpoint *without error* because this codebase leans on `strict=False`:

| trap | why no error | guard added |
|---|---|---|
| `DECODER_CKPT` defaults to the hybrid's Stage-0 | wrapper only checks the file exists | hard-fail >50% missing keys |
| `--model-config` defaults to the hybrid | architecture not auto-detected | hard-fail >50% missing keys |
| `prefix_k` defaults to 32 | **k has no parameters** — no key-count guard can detect it | resolved from `run_metadata.json`; conflict is a hard error |
| stale code (missing `git pull`) | — | positive `prefix_k = N` line to check for |

The `prefix_k` trap fired live and cost **0.0145 ROUGE-L** — about 4× the entire
hybrid-vs-Transformer gap — which would have read as a large regression and prompted a
needless run. **Standing lesson: `Missing keys: 0` is not evidence the model was built
correctly; in this codebase it only means the parameter shapes agreed.**

Separately, `validate_for_willi.sh` was observed to **degrade silently**: a failed
`torch==2.1.2` install caused a fallback to Python 3.14.3, the version gate dropped from
`PASS` to `WARN`, and the script still printed *"All gates passed."* Always grep for
`PASS] Python version == 3.9.23` specifically.

---

## 7. What changed in the claim, and when

| date | claim | why it changed |
|---|---|---|
| pre-14A | "attention-free hybrid matches/beats transformers at better efficiency" | untested — no baseline existed |
| 14A-3 | quality unknown; Transformer wins Stage-0 PPL | first head-to-head, went against the hybrid |
| 14A-7 | efficiency claim not realised as implemented | FlashAttention + unoptimised scan |
| 14A-6 | **split decision**, hybrid wins clinical correctness | bootstrap with CIs on all metrics |
| k-sweep | briefly appeared k-dependent | hybrid k=64 degrades |
| k-sweep | **resolved**: k=32 principled, headline stands | Transformer proven k-insensitive |

Two corrections were made to this document's own earlier readings, both caught by
confidence intervals and both recorded rather than quietly fixed: "the Transformer wins all
four surface metrics" (BLEU-1/4 are ties), and "prefix bandwidth is not binding" (it is —
on CheXbert, invisible to ROUGE-L).

---

## 8. Reproduction

```bash
# Stage-0 (both; only MODEL_CONFIG differs)
MODEL_CONFIG=transformer_150m_baseline EXPERIMENT=h100_stage0_transformer_150m \
  sbatch scripts/train_stage0_150m_h100.sh

# Report-gen decoder (both; PREFIX_K selects the arm)
MODEL_CONFIG=transformer_150m_baseline_rrg PREFIX_K=32 \
DECODER_CKPT=./outputs/h100_stage0_transformer_150m/checkpoints/last.ckpt \
NUM_GPUS=4 MAX_STEPS=12000 \
IMAGE_ENCODER_CKPT=./outputs/h100_kd_150m_v2_full_data_lr3e6/checkpoints/last.ckpt \
EXPERIMENT=h100_report_gen_transformer_tower13d \
  sbatch --gpus=4 scripts/train_report_generation_h100.sh

# Evaluation (confirm the log prints `prefix_k = N`)
MODEL_CONFIG=transformer_150m_baseline_rrg DECODE=beam BEAM_SIZE=3 \
PARQUET=/sc/home/$USER/dataset/mimic_full/test.parquet NUM_SAMPLES=999999 \
DUMP_DIR=results/report_gen_transformer_test_split \
CHECKPOINT=./outputs/h100_report_gen_transformer_tower13d/checkpoints/last.ckpt \
  sbatch scripts/inspect_report_generation_h100.sh

DUMP_DIR=results/report_gen_transformer_test_split sbatch scripts/score_chexbert_h100.sh

# Paired bootstrap (all nine metrics with 95% CIs)
A=results/report_gen_tower13d_test_split B=results/report_gen_transformer_test_split \
NAME_A=hybrid NAME_B=transformer OUTPUT=analysis/bootstrap_hybrid_vs_transformer.md \
  sbatch scripts/bootstrap_compare_h100.sh

# Efficiency sweep
MODELS="hybrid_150m_v2 mamba_150m_baseline xlstm_150m_baseline transformer_150m_baseline" \
OUTPUT_DIR=analysis/efficiency_150m_with_transformer \
  sbatch scripts/profile_efficiency_h100.sh

# Boilerplate / diversity, with controls
HYPS=results/report_gen_tower13d_test_split/hyps.txt \
REFS=results/report_gen_tower13d_test_split/refs.txt \
BASELINE=results/retrieval_floor_test_split/hyps.txt \
OUTPUT=analysis/generation_diversity_13d.md \
  sbatch scripts/analyze_diversity_h100.sh

# Scan error bound (local, CPU)
python tests/test_scan_correctness.py --emit    # -> analysis/scan_error_bound.md
```

### Artefacts

| file | contents |
|---|---|
| `analysis/bootstrap_hybrid_vs_transformer.md` | the headline comparison, k=32 |
| `analysis/bootstrap_k64_vs_k32.md` | hybrid prefix sweep |
| `analysis/bootstrap_transformer_k64_vs_k32.md` | Transformer k-insensitivity |
| `analysis/bootstrap_best_k_vs_best_k.md` | each model at its own best k |
| `analysis/generation_diversity_13d{,_n1433}.md` | boilerplate, both splits |
| `analysis/scan_error_bound.md` | scan correctness table |
| `analysis/efficiency_150m_with_transformer/` | latency/memory curves |
| `analysis/h100_scaling_results.md` | the Phase 1–13 writeup this supplements |
