# Mamba-3 backbone upgrade — results

> Branch `h100_mamba3_v2`. Plan of record `MAMBA3_PLAN_V2.md`, live state `mamba3_v2_state.json`.
> Every number here was measured on this branch between 2026-09-06 and 2026-09-20. The incumbents it is
> compared against come from `analysis/PHASE14_SUPERVISOR_REVIEW.md` and `H100_SCALING_PLAN.md` §15B-4;
> where a summary line elsewhere disagrees with this document, the seed tables below are the record.
> **This branch is not merged into `h100_scaling`.** The Phase 14/15 results stay reproducible there.

---

## 1. Summary

Two findings, one positive and one null, both measured against a parameter-matched Transformer and the
incumbent hybrid on the identical pipeline and the official MIMIC-CXR test split.

1. **The recurrences this project shipped did not compute the recurrences they specified, and fixing them is
   worth more than any architecture change measured here.** At 12,000 steps, repairing the selective scan and
   the Δ initialisation on the *existing* Mamba-1 block cut validation perplexity by 16.0%. Repairing the
   mLSTM recurrence as well is worth a further 5.2%. At full 120,000-step scale the corrected Mamba-3 backbone
   reaches **11.674** against the incumbent hybrid's **13.18**, closing **77%** of the gap to the matched
   Transformer's **11.222**.
2. **That backbone gain does not reach the task.** On report generation across three seeds, the corrected
   model is **statistically indistinguishable from the matched Transformer on all ten metrics** and from the
   incumbent hybrid on eight of ten. This is the eleventh time this project has measured a text-side
   improvement failing to move the downstream clinical metrics, and it is the best-controlled instance.

A third result is architectural rather than about quality: the corrected backbone **matches FlashAttention's
memory** at every sequence length tested, where the incumbent used 5.9× more at 16,384 tokens, and it trains at
sequence lengths where the incumbent runs out of memory.

---

## 2. The defect

Both chunked recurrences computed `A_cum · cumsum(Bx / A_cum.clamp(ε))`. Where the clamp fires, a token's own
contribution to the state is annihilated rather than perturbed. Measured against float64 sequential references
(`tests/test_mamba3_numerics.py`, reproduced independently by Phase 14C-1 in `tests/test_scan_correctness.py`):

| operator | configuration | rel-max-err |
|---|---|---|
| selective scan | Δ = 0.705, the value this repo initialises to | **0.92** |
| selective scan | Δ = 0.1, the `dt_max` of the *correct* init | 0.39 |
| mLSTM / TFLA | shipped `forget_gate_bias_init=0.0`, chunk 64 | **0.88** |
| Δ at init | `hybrid_150m_v2` canonical vs the reference `logU[1e-3, 1e-1]` | 0.807 vs ~0.021 |

This does not invalidate any published number: training and evaluation used the same operator throughout, so
each is a valid measurement of the system as built. The narrower true statement is that the blocks did not
compute the recurrence they were specified to compute.

**Why the repair motivated the architecture change.** The exact log-space form costs 19.3 GB of mask memory in
Mamba-1's `A = (d_inner, d_state)` parameterisation and 19 MB in Mamba-2/3's scalar-`A`-per-head form. Migrating
to the state-space dual is what makes correctness affordable. That argument was made from arithmetic before any
GPU time was spent and is confirmed by the wall clocks in §6.

---

## 3. Screen: which operator to take forward

12,000 steps, 150M, PubMed with the BioMedLM teacher, identical seed and data order. The pre-registered bar was
**0.642 PPL**, twice the seed standard deviation measured from two runs of the control.

| arm | operator | val PPL (seed 42 / 1234) | wall |
|---|---|---|---|
| A0 | Mamba-1, both recurrences defective | 19.387 / 18.933 | 7:59:00 |
| **A1** | Mamba-1, **both corrected** + Δ init | **16.294** | 4:11:13 |
| A2 | Mamba-3 SSD, `d_state` 128, mLSTM still defective | 16.708 / 16.376 | 4:09:14 |
| A3 | A2 + exponential-trapezoidal | 16.719 | — |
| A4-hi | A2 + complex/RoPE, `theta_max` 0.2 | 16.199 / 18.912 | — |
| **A2x** | **A2 + corrected mLSTM** | **15.566 / 15.788** | 5:45:19 |

- **The trapezoidal rule is a null** at this scale: +0.011 PPL, 1.7% of the bar, in a well-powered paired
  comparison. Reported as a negative result on the paper's Proposition 1.
- **Complex/RoPE is high-variance, not better.** A4-hi led at seed 42 and lost by 2.5 PPL at seed 1234; its
  cross-seed spread is 8.2× A2's. The pre-registered rule advanced the simpler arm and the replication
  vindicated it. A separate controlled experiment *did* reproduce the paper's parity-task capability claim
  (rope off 62-64%, rope on 100%), but only with Δ free to reach ~1; under the reference `dt` init the
  reachable rotation is ~0.06 rad and the mechanism is dormant. Both halves are reported.
- **A2x wins and carries into the pipeline**: −0.865 PPL against A2 on the two-seed mean, 1.35× the bar,
  better at both seeds, and the most seed-stable arm measured (spread 0.222 against A2's 0.332 and A0's 0.454).

⚠ **Correction to an earlier reading of this table.** The screen first concluded "the gain is the defect, not
the architecture" from A1 ≈ A2. That comparison was not like-for-like: A1 had *both* recurrences corrected
while A2 still ran the defective mLSTM. With both corrected on both sides, A2x beats A1 by 0.728 PPL at seed
42 — above the bar, but with a single A1 seed, so it is stated as provisional.

---

## 4. Stage-0: the backbone gate

120,000 steps, one seed per arm, recipe identical across all three (batch 16 × accum 3, LR 4e-4, warmup 2000,
clip 0.5, gradient checkpointing, same corpus, same KD teacher). Job 2553736.

| backbone | val PPL | vs hybrid |
|---|---|---|
| `hybrid_150m_v2`, both recurrences defective | 13.180 | — |
| **`hybrid_150m_m3` (A2x), both corrected** | **11.674** | **−1.506 (−11.4%)** |
| `transformer_150m_baseline`, matched parameters | 11.222 | −1.958 (−14.9%) |

**77% of the hybrid-Transformer gap closes.** The Transformer keeps a 0.452 lead, 4.0% relative.

⚠ **No arm has a Stage-0 seed band** — not this one, not the hybrid, not the Transformer, anywhere in this
project's history. The residual 0.452 is the size of the 12K screen's seed spread (0.33-0.45). "Essentially
matches the Transformer at matched parameters" is supportable; "beats it" is not.

---

## 5. Report generation: the task metrics

Official MIMIC-CXR test split, n=2663, beam 3, `prefix_k=32`, `last.ckpt`, the 13D image tower reused unchanged
so the decoder backbone is the only lever. Three decoder seeds (42/43/44) matched to the incumbents' seeds, so
every difference is paired. Decision rule fixed before the runs: **a claim requires the paired mean to exceed
one baseline seed standard deviation and the sign to hold at ≥2 of 3 seeds.**

| metric | Mamba-3 | hybrid | Transformer | vs hybrid | vs Transformer |
|---|---|---|---|---|---|
| ROUGE-L | .1953 ± .0029 | .1949 ± .0047 | .1952 ± .0021 | tie | tie |
| BLEU-1 | .2484 ± .0005 | .2508 ± .0034 | .2478 ± .0022 | tie | tie |
| BLEU-4 | .0579 ± .0009 | .0578 ± .0032 | .0575 ± .0016 | tie | tie |
| CheXbert-14-micro | .4480 ± .0188 | .4480 ± .0223 | .4443 ± .0153 | tie | tie |
| CheXbert-14-macro | .2715 ± .0121 | .2660 ± .0122 | .2692 ± .0106 | tie | tie |
| CheXbert-5-micro | .5044 ± .0257 | .5086 ± .0382 | .5032 ± .0226 | tie | tie |
| CheXbert-5-macro | .4170 ± .0164 | .4193 ± .0274 | .4170 ± .0165 | tie | tie |
| exact-match-14 | .0452 ± .0028 | .0380 ± .0058 | .0455 ± .0027 | **+.0071** | tie |
| exact-match-5 | .2242 ± .0070 | .2163 ± .0019 | .2244 ± .0057 | **+.0079** | tie |
| example-F1 | .3858 ± .0174 | .3790 ± .0214 | .3817 ± .0144 | tie | tie |

**Against the matched Transformer: indistinguishable on all ten metrics.** Of 27 per-seed interval calls,
exactly one excluded zero (BLEU-1 at seed 44, in Mamba-3's favour). This is the cleanest same-protocol
equivalence the project has measured, and it is the form in which the central thesis claim survives.

**Against the incumbent hybrid: two claims, both exact-match, both small.** Exact-match-5 is positive at 3/3
seeds and exact-match-14 at 2/3. ⚠ The bar is the hybrid's own seed SD as the rule specifies, and for
exact-match-5 that SD is unusually tight (.0019) while Mamba-3's is .0070; against its own spread the claim is
~1.1 SD. The defensible sentence is *"slightly more often reproduces the exact label set"*, nothing stronger.

**The per-seed calls contradict each other**, exactly as Phase 15B found for the hybrid-Transformer pair:

| seed | Mamba-3 wins (CI excludes 0) | hybrid wins |
|---|---|---|
| 42 | BLEU-4, exact-match-14 | CheXbert-5-micro |
| 43 | CheXbert-14-micro, 14-macro, 5-micro | BLEU-1 |
| 44 | exact-match-14 | BLEU-1 |

Any single-seed reading of this table would reach a different and wrong conclusion. That is the finding, not a
caveat to it.

**Against the retrieval-NN floor**, the standing pattern holds for a fourth architecture: text metrics and
exact-match win at 3/3 seeds, **CheXbert-14-macro loses at 3/3** (−.0180 / −.0294 / −.0422) and 5-macro at 2/3.
Per-label, Lung Lesion scores .000 / .022 / .000 and Pleural Other .033 / .017 / .000. The rare-label deficit
that survived rare-finding oversampling (13F) and an auxiliary CheXpert loss (15C) survives a corrected
operator and a new mixer too. It is not an artefact of any one architecture.

**Templating** (job 2560259, seed 42): 26.0% of generations fall in an exact-duplicate cluster, against the
incumbent's 29.2% on the same split, with references at 0.2% and the retrieval floor at 7.3%. Lexical diversity
moves as little: distinct-2 .0350 vs .0299, self-BLEU-4 .6709 vs .6854. The pre-registered outcome is
**INTERMEDIATE**, as it was for the incumbent. Templating therefore explains none of the movement above, which
is what this check exists to rule out.

---

## 6. Efficiency

Random weights, bf16, batch 4, H100 80GB (job 2561023) — the Phase 14A-7 protocol, so the numbers are directly
comparable to that section of the supervisor review.

**Inference, latency and peak memory:**

| L | hybrid | Mamba-3 | Transformer |
|---|---|---|---|
| 256 | 17.6 ms / 1.22 GB | 20.6 ms / **0.51 GB** | 4.9 ms / 0.51 GB |
| 2048 | 120.6 ms / 5.71 GB | 86.2 ms / **1.24 GB** | 16.1 ms / 1.24 GB |
| 16384 | 915.6 ms / 42.00 GB | 678.2 ms / **7.09 GB** | 163.9 ms / 7.15 GB |

**Memory is now at parity with FlashAttention** — 7.09 GB against 7.15 GB at 16,384 tokens, where the incumbent
needed 42 GB. The memory scaling exponent is 0.643 against attention's 0.644; the incumbent's was 0.863. This is
the state-space-dual activation argument realised: the plan predicted ~37× less scan activation memory from
arithmetic, and the end-to-end model measures 5.9× at the longest length.

**Training (forward + backward)** is where it matters most: the incumbent **runs out of memory at L ≥ 4096**,
while the corrected Mamba-3 trains to 8192. At L = 2048 it is **2.6× faster and uses 3.9× less memory**
(416 ms / 13.9 GB against 1096 ms / 54.0 GB). Its latency exponent is 0.886 against the incumbent's 1.285.

**Against attention, the honest position is unchanged from 14A-7 in direction and much narrower in size.** The
Transformer is still faster at every length tested (4.1× at 16,384 inference, 8× at 2048 training), because
`F.scaled_dot_product_attention` dispatches to a fused FlashAttention kernel while the scan is pure PyTorch.
What changed is that the memory half of the efficiency claim now holds rather than fails, and the latency gap
narrowed by 1.35-2.6×. A Triton SSD kernel remains the obvious unexploited headroom.

**Decode, prompt 256, 64 new tokens, batch 1.** The O(1) recurrent cache works at full scale:

| path | s/token | growth, 2nd half / 1st | tok/s |
|---|---|---|---|
| Mamba-3, full recompute | 0.03293 | 0.96× | 30.4 |
| **Mamba-3, cached** | **0.00635** | **1.00×** | **157.6** |

**5.19× per token, and flat in context** — that flat growth *is* the O(L²) → O(L) claim, measured rather than
argued. The documented cost is time-to-first-token: the prefill steps token by token, so it is 4.0× slower
(1.79 s against 0.45 s). Exposing the carried state from the chunked scan would remove that and is recorded as
follow-up. The incumbent hybrid has no cache at all, by construction: its legacy mLSTM operator computes no
recurrence that an O(1) step could reproduce, which is a second, functional argument for the correction.

⚠ **One number in this run is not credible and is excluded:** the Transformer's uncached decode came out at
0.337 s/token, 13× slower than the hybrid in the same loop while being 3.5× *faster* in the sweep. The two are
inconsistent; it is flagged as unverified rather than reported. The Transformer has no KV cache in this repo,
so its decode path is not optimised here either way.

---

## 7. What this licenses, and what it does not

**Licensed:**
- The shipped selective scan and TFLA did not compute their specified recurrences; the deviation is bounded and
  regression-tested at `analysis/scan_error_bound.md` and `tests/test_mamba3_numerics.py`.
- Repairing them is worth −16.0% validation perplexity at 12K steps on the existing architecture, and the
  corrected Mamba-3 backbone reaches 11.674 at 120K against the incumbent's 13.18.
- At matched parameters the corrected attention-free model is statistically indistinguishable from a
  parameter-matched Transformer on every report-generation metric measured, across three seeds.
- It matches attention's memory profile and trains at lengths the incumbent cannot, while remaining slower in
  wall-clock latency against a fused attention kernel.

**Not licensed:**
- "Mamba-3 improves report generation." It does not, measurably, against either incumbent.
- "Mamba-3 matches the Transformer on language modelling." One seed each, 0.452 apart, no seed band.
- Any claim from a single seed. Three of the six per-seed comparisons in §5 contradict their neighbours.
- Anything about the trapezoidal rule or complex state beyond "null at this scale and context length", with
  the dormancy caveat in §3.
- MIMO: plumbed, asserted bit-identical at rank 1, never run.

**Open limitations:** Stage-0 has one seed per arm; A1 has one seed; the image tower was contrastively
co-trained with the legacy hybrid text encoder and reused unchanged for all arms (14A-4); the efficiency
comparison pits a fused attention kernel against a pure-PyTorch scan; the Transformer decode number in §6 is
unverified; and the end-to-end effect of the original defect on the *published* 13D metrics is still unmeasured
(the `HYBRID_EXACT_SCAN=1` probe of Phase 14C-3, which this branch makes cheap but has not run).

---

## 8. Reproduction

All cluster work goes through `sbatch`; the login node executes nothing scripted.

```bash
# screen one arm (the ladder is defined once, in scripts/mamba3_arms.py)
ARMS="A2x A2x-s2" VAL_EVERY=12000 SAVE_TOP_K_SCREEN=0 sbatch --array=0-1 scripts/screen_arms_h100.sh

# the full pipeline: Stage-0 -> decoder x3 seeds -> beam eval -> CheXbert -> 9 paired bootstraps
DRY_RUN=1 source scripts/submit_v3_chain.sh     # prints 19 submissions, runs nothing
source scripts/submit_v3_chain.sh

# efficiency curves and the decode benchmark
DECODE_CURVE=true MODELS="hybrid_150m_v2 hybrid_150m_m3_rrg transformer_150m_baseline" \
  OUTPUT_DIR=analysis/efficiency_150m_m3 sbatch scripts/profile_efficiency_h100.sh
```

Key job ids: screen 2552165, Stage-0 2553736, decoders 2553737/43/49, evals 2553738/44/50, CheXbert
2553739/45/51, bootstraps 2553740-42 / 46-48 / 52-54, diversity 2560259, efficiency 2561023.
