# EFFICIENCY_PLAN — close the wall-clock gap to FlashAttention, or bound why it cannot close

**Branch:** `h100_mamba3_v2` · **State:** `efficiency_state.json` · **Helper:** `venv/bin/python scripts/mamba3_state.py --plan efficiency ...`

**⚠ This plan changes no published number.** Every item is an inference-time or kernel-level change
to operators that are mathematically fixed. No checkpoint is retrained, no metric in
`analysis/mamba3_results.md`, `analysis/PHASE14_SUPERVISOR_REVIEW.md` or `H100_SCALING_PLAN.md` is
touched. Any change that alters a decoded token is out of scope by construction and must fail its
equivalence gate. **No merge into `h100_scaling`.**

---

## 1. Why this plan exists

The supervisor asked, on 2026-09-25:

> "I also find out the throughput of our method is still behind some state of the art model who use
> FlashAttention of transformers. I saw some code you use Triton or something else to empower mamba
> be compatible with FlashAttention attributes. Did it work out?"

The premise about Triton was verified and is false: **no Triton kernel ever executed in this
project**. `scan_triton.py` and `tfla_triton.py` had exactly one call site between them, inside
`mamba_block_v2.py`, which the live `hybrid_block.py` never imports. All of it was deleted in
`37f7964`; `scan_interface.py:28` reads `TRITON_AVAILABLE = False`. The impression comes from three
surviving artefacts, all addressed in E5: the stale root script `test_triton_fix.py`, the unused
`triton>=2.1.0` line in `requirements.txt`, and the name **TFLA** = *Tiled Flash Linear Attention*,
which borrows FlashAttention's tiling idea but is implemented in plain PyTorch.

The measurement behind the supervisor's observation (job 2561023, batch 4, bf16, H100, random
weights, `analysis/efficiency_150m_m3/`):

| seq len | corrected Mamba-3 | Transformer (SDPA) | old hybrid |
|---|---|---|---|
| 2,048 | 86.2 ms / 1.24 GB | 16.1 ms / 1.24 GB | 120.6 ms / 5.71 GB |
| 16,384 | 678.2 ms / 7.09 GB | 163.9 ms / 7.15 GB | 915.6 ms / 42.00 GB |

Latency exponents **0.840** (Mamba-3) vs **0.844** (Transformer); memory exponents 0.643 vs 0.644.

**So the memory half of the claim worked out completely and the speed half did not.** At 16,384
tokens the corrected model matches FlashAttention's memory to within 1% where the previous model
needed 42 GB — but it is 4.14× slower, 5.35× at 2,048, and 8× slower on a training step at 2,048.
Both exponents are flat and roughly equal across the measured range, so **there is no crossover and
no asymptotic speed advantage visible in the data we have**. The writeup attributes the gap to
`F.scaled_dot_product_attention` dispatching to a fused kernel while our scan is interpreted
PyTorch. That attribution is currently an assertion. This plan's first job is to make it a
measurement, and its second is to act on what the measurement says.

## 2. The mechanism, and why the cheap fix is not Triton

`ssd_chunked_scan` (`hybrid_xmamba/kernels/ssd/ssd_interface.py:169-190`) carries state across
chunks with a **Python `for ci in range(nc)` loop**:

```
state = zeros(batch, nheads, headdim, dstate)
for ci in range(nc):                      # nc = ceil(seqlen / chunk_size)
    offsets.append(einsum(state, C_h[:, ci]) * carry_gate)
    state = decay * state + einsum(decay_to_end, coeff, x, B)
```

At the shipped `mamba3_chunk_size: 64`, sequence length 16,384 gives `nc = 256`, and the config runs
**9 Mamba-3 layers**, so one forward pass executes **2,304 sequential Python iterations**, each
issuing several small CUDA kernels whose operands are `(batch=4, cs=64, nheads=24, headdim=64,
dstate=128)`. That is the shape of a launch-latency-bound workload, and it is the structural
difference from the Transformer, whose whole attention is one fused call per layer.

Three consequences drive the phase order:

1. **The loop count is a config knob today.** `mamba3_chunk_size` divides `nc`. The chunked
   decomposition is exact for any chunk size, so this is a pure performance parameter — but it
   changes floating-point association, so it must be gated on numerical equivalence, not assumed.
2. **The loop is removable in pure PyTorch.** The inter-chunk recurrence is first-order linear with
   a *scalar* coefficient per `(batch, head, chunk)`: `state_c = g_c · state_{c-1} + contrib_c`. That
   is the same shape as the intra-chunk term, which this file already solves in parallel with
   `segsum`. **No Triton, no external dependency, no new kernel** — the same trick applied one level
   up. This is E2 and it is the highest-value item in the plan.
3. **A fused kernel is a binding job, not a research job.** The headline arm runs
   `mamba3_use_trapezoid: false`, `mamba3_use_rope: false`, `mamba3_bc_bias: none`, which by the
   config's own comment makes the block *exactly Mamba-2 SSD*, at `headdim 64 / dstate 128 /
   ngroups 1` — the reference kernel's standard shapes. If E3 is reached, it is integration behind a
   third `scan_impl` value, gated by the fp64 oracle this repo already owns.

**Amdahl's ceiling, and why E0 comes first.** 3 of the 12 layers are mLSTM on pure-PyTorch TFLA, and
no Mamba kernel touches them. If mLSTM is 40% of inference time, a *perfect* SSD path caps out at
2.5× and never reaches the 4.14× needed. Nobody has measured that split. Every estimate above this
line is a hypothesis until E0 lands.

## 3. Pre-registered rules (written before any number exists)

- **R1 — Equivalence gate.** No change ships unless, on the same inputs, it matches the current
  operator to `rel-max-err ≤ 1e-4` in fp32 and agrees with `ssd_sequential_reference` (fp64) at
  least as well as the current operator does. Mismatch on a `cu_seqlens` document boundary is an
  automatic fail. A change that survives this cannot alter a decoded token, which is what licenses
  "no published number moves".
- **R2 — Memory is a published claim and outranks speed.** Memory parity with FlashAttention
  (7.09 vs 7.15 GB at 16,384) is a headline result. Any change whose peak memory exceeds the
  Transformer's at *any* measured length is rejected regardless of speedup, unless it is put behind
  an opt-in flag that is off by default.
- **R3 — Adoption bar.** A change becomes the default only if it is ≥ 1.25× faster at ≥ 2 of the
  measured sequence lengths and loses nowhere by more than 5%. Below that it is recorded as a null
  and reverted, not kept "because it might help later".
- **R4 — Predict first.** Each phase writes its expected direction and rough magnitude into this
  file *before* the job is submitted. When the prediction is wrong, the wrongness is recorded next
  to the number. (Three predictions were wrong in direction during V5; this rule is why that is
  visible.)
- **R5 — Honest framing.** Matching attention on wall-clock is engineering, not a research
  contribution. It strengthens the efficiency chapter; it changes no scientific claim. If the
  Amdahl bound from E0 says parity is unreachable, that bound *is* the deliverable and E3 is not
  attempted.

## 4. Phases

### E0 — Measure before optimising: where does the time actually go?

*Pre-registered prediction (R4): the inter-chunk loop is >50% of Mamba-3 layer time at 16,384; the
3 mLSTM layers are 20–40% of total model time; unfused attention is slower than our scan at 16,384.*

- [x] **E0-A** Per-layer-type timing split. Extend `scripts/performance_profile.py` with
  `--per-layer`, using CUDA events around each `HybridBlock` (or `torch.profiler` with
  `record_shapes`), reporting ms and % by mixer type for `hybrid_150m_m3` at the 14A-7 sequence
  ladder. **This is the Amdahl bound and it gates E2/E3/E4.**
- [x] **E0-B** Intra-block split for `Mamba3Block`. **As built (2026-09-25), this is a two-way
  split — `ssd_chunked_scan` versus the rest of the block** (projections, conv, norm), via a
  monkeypatch that restores itself on exit. The finer intra-chunk-vs-inter-chunk breakdown the §2
  hypothesis needs is *not* measured directly by a timer inside the hot path; it is inferred from
  E0-C, where sweeping `chunk_size` moves only the loop length `nc`. If E0-C comes back ambiguous,
  add the in-path timers then and not before — instrumenting the operator is a change to the
  operator.
- [x] **E0-C** Launch-overhead probe: for fixed seqlen 16,384, sweep `mamba3_chunk_size` over
  `{64, 128, 256, 512}` and record latency, peak memory and `nc`. If latency falls ~linearly in
  `nc`, the path is launch-bound and E2 is worth its cost; if flat, it is bandwidth-bound and E2 is
  not.
- [x] **E0-D** **The supervisor's answer, as a measurement.** Re-run the 14A-7 inference sweep for
  `transformer_150m_baseline` with the fused backends disabled (`torch.nn.attention.sdpa_kernel`
  restricted to `MATH`), alongside the existing fused numbers. Add `--attn-backend {auto,math}` to
  the profiler. This separates *algorithm* from *kernel engineering*: if Mamba-3 beats unfused
  attention at long lengths, "the gap is implementation, not algorithm" becomes a number instead of
  a claim.
- [x] **E0-E** Submit E0-A..D as one job (`sbatch scripts/profile_layer_split_h100.sh`, new CPU/GPU
  wrapper following the `profile_efficiency_h100.sh` conventions: `--partition=aisc-batch
  --account=aisc --gpus=1 --exclude=ga03,gx17v1,gx13v1`, no `--gres`). Write results to
  `analysis/efficiency_layer_split/`.
- [x] **E0-F** Record the Amdahl bound explicitly in this file: *max achievable speedup if the
  Mamba-3 path became free* = 1 / (fraction not in Mamba-3). Compare to the 4.14× target and state
  plainly whether parity is reachable at all.

**RESULTS — measured 2026-09-25, jobs 2579631 (E0-A/B/D) and 2579642 (+E0-C, E1-B).** H100 gx14,
batch 4, bf16, random weights. Job 2579642 hit its 1.5 h wall limit inside the E1-B arm at L=16384.

*E0-A / E0-B — the hypothesis in §2 is CONFIRMED.* At L=16384 the Mamba-3 forward is 555 ms, of
which the 9 `mamba3` layers are **77.8%** and `ssd_chunked_scan` alone is **67.5% of the whole
forward** (374.9 ms — i.e. 87% of all Mamba-3 layer time is inside the scan). The 3 mLSTM layers are
only **13.4%** and embed/head/norm **8.8%**. Run-to-run spread across the two jobs is ~4%
(555.29 / 577.07 ms). The incumbent Mamba-1 hybrid is worse still: 86.6% in its `mamba` layers.

*E0-F — the Amdahl bound is **4.50×** at L=16384* (4.59× in the repeat), rising with length from
2.93× at L=2048. The gap to close is 4.14×. **So parity is arithmetically possible but only by
making the scan almost free**: with a free SSD path the model would run at ~123 ms against the
Transformer's 164 ms, and the floor set by mLSTM + embed/head is ~123 ms. E2/E3 are not ruled out by
the ceiling — they are ruled *in*, but with no margin for a half-measure.

*E0-D — this is the supervisor's answer, and it is decisive.* Same Transformer, same protocol, only
the SDPA backend changes:

| L | fused (`auto`) | unfused (`math`) | fused advantage |
|---|---|---|---|
| 2,048 | 16.12 ms / 1.24 GB | 81.93 ms / 2.43 GB | 5.1× |
| 4,096 | 33.32 ms / 2.09 GB | 298.79 ms / 8.11 GB | 9.0× |
| 8,192 | 71.34 ms / 3.78 GB | 1195.95 ms / 30.45 GB | 16.8× |
| 16,384 | 163.86 ms / 7.15 GB | **OOM** | runs vs does not run |

Latency exponent 0.871 fused vs **1.513** unfused; memory exponent 0.644 vs **1.201**. The `auto` arm
reproduces the published 14A-7 number to 0.04 ms (163.86 vs 163.9), so the harness is sound.
**Against the same algorithm without a hand-written kernel, our model wins and the margin grows with
length**: at 8,192 Mamba-3 is ~3× faster than unfused attention, and at 16,384 it runs in 7.09 GB
where unfused attention does not run at all. The 4.14× deficit is the cost of FlashAttention's
kernel engineering, now measured rather than asserted.

*E0-C — `chunk_size` is a real lever, and my pre-registered prediction was wrong (rule R4).* I
predicted 1.5–2.5× for 64→256, monotonic in `nc`. It is a **U-curve with its optimum at 128**:

| chunk_size | L=4,096 forward | L=16,384 forward | scan at 16,384 |
|---|---|---|---|
| 64 (shipped) | 135.6 ms | 557.8 ms | 377.0 ms |
| **128** | **99.4 ms** | **398.6 ms** | **216.7 ms** |
| 256 | 104.6 ms | 411.8 ms | 230.1 ms |
| 512 | 130.7 ms | 512.8 ms | 331.6 ms |

1.40× on the whole forward and 1.74× on the scan, for a one-line config change. Bigger chunks cut
the loop length `nc` but grow the per-chunk mask work as `O(seqlen · chunk_size)`, and 128 is where
those cross. **Unverified under R1 and R2**: `run_layer_split` does not yet record peak memory, and
no equivalence check has been run, so this is not adoptable yet.

*E1-B — `torch.compile` is the biggest single result here, and my prediction was wrong by a factor
of three (rule R4).* I predicted < 1.3× and possible outright failure. Measured, under the same
`--sweep` protocol as the published numbers:

| L | compiled | published uncompiled | Transformer (fused) | remaining gap |
|---|---|---|---|---|
| 2,048 | 20.14 ms / 1.32 GB | 86.2 ms / 1.24 GB | 16.12 ms / 1.24 GB | **1.25×** |
| 4,096 | 44.97 ms / 2.15 GB | — | 33.32 ms / 2.09 GB | 1.35× |
| 8,192 | 88.58 ms / 3.82 GB | — | 71.34 ms / 3.78 GB | 1.24× |
| 16,384 | **compile did not finish** | 678.2 ms / 7.09 GB | 163.86 ms / 7.15 GB | — |

**At L=2,048 that is 4.28×, taking the gap from 5.35× to 1.25×, with memory still at parity.** The
reason compile was pinned off (`MAMBA3_PLAN_V2.md:301`) was a Mamba-1 artifact, and this is the
measurement that retires it for SSD. The failure mode is exactly FE4: Dynamo unrolls the `nc`-long
Python loop, so each new sequence length pays a graph build that grows with `nc`, and L=16,384
(`nc`=256) did not complete inside the job's remaining time. **Nothing here is adoptable until R1
passes** — `torch.compile` is not required to preserve floating-point association, and this project
has already been burned once by an operator that computed a different function than advertised.

*What this does to E2.* The parallel chunk-state rewrite is no longer primarily a speed play —
`torch.compile` already captures much of that win. Its remaining justification is stronger and more
specific: **it removes the unrolled loop that makes compilation blow up**, which is what currently
blocks the compiled path at exactly the sequence lengths the thesis cares about.

**Gate:** E0-F's bound decides what follows. If the bound is < 2×, skip to E4 and E5 and report the
ceiling as the result.

### E1 — Free levers: no new code path

*Pre-registered prediction (R4): `chunk_size` 64→256 gives 1.5–2.5× at 16,384 for < 1.5 GB extra;
`torch.compile` gives < 1.3× and may fail outright by unrolling the `nc` loop into a huge graph.*

- [x] **E1-A** `mamba3_chunk_size` as an inference-time knob. It is already a config key and needs
  no code change. Verify equivalence under R1 across the sweep, then record the
  latency/memory Pareto front. **Note:** the trained checkpoints used 64; R1 is what licenses
  decoding at another value.
- [x] **E1-B** `torch.compile` arm for inference only. Add `--compile` to `performance_profile.py`
  (it has no such flag today). **The documented reason compile is disabled does not apply here:**
  `MAMBA3_PLAN_V2.md:301` attributes `compile_model=false` to `mamba_block.py:248-276`'s Python loop
  over `(row, segment)` in the *Mamba-1* block, and states that SSD "handles boundaries natively …
  One masked-`exp`, fully batched". That decision was never re-evaluated after SSD landed. Measure
  compile time as well as steady-state latency; a 10-minute compile for a 1.1× gain is a null.
- [x] **E1-C** Report both under R3. Anything that fails the bar is written up as a null and
  reverted.
- [ ] **E1-E** **Re-measure the headline under a protocol the last two jobs proved is necessary**
  (`scripts/profile_e1_confirm_h100.sh`). One sequence length per *process*, each with its own
  Inductor cache, Transformer arms first. Also finishes the training reference job 2582482 never
  reached. See the E1 follow-up results below for why this is not optional.
- [x] **E1-D** **The gate, added 2026-09-25 after E0 produced two candidate wins.**
  `scripts/check_operator_equivalence.py` implements R1 at two levels: the operator against the
  fp64 oracle (including a `cu_seqlens` boundary that falls inside a chunk) and the model's logits
  across `chunk_size` and `torch.compile`. `scripts/verify_and_profile_e1_h100.sh` runs it **first**
  and stops the job on failure, then times all five arms — baseline, `chunk_size=128`, compiled,
  compiled+128, and the two L=16384 arms last — under one protocol in one job, so every ratio is a
  within-job comparison. Measured on CPU so far: operator agreement 1e-7, logits 2.4e-5, both far
  inside the 1e-4 tolerance; **`torch.compile` equivalence is still unverified and is the one that
  matters**, since a compiler is under no obligation to preserve floating-point association.
  `--time=03:00:00` and a persistent Inductor cache are the mitigations for the 2579642 timeout.

**RESULTS — measured 2026-09-25, job 2580198 on gx09, torch 2.11.0+cu128.** All five arms in one
job under one protocol, so every ratio below is a within-job comparison.

*R1 PASSED on GPU, including `torch.compile`* — the one check that actually mattered. Operator vs
the fp64 oracle 2.1e-07; `chunk_size` logits 2.3e-05 to 3.4e-05; **`torch.compile` logits 3.0e-05**,
against a 1e-4 tolerance. The `cu_seqlens` boundary case passes at every chunk size. Neither variant
changes the function the model computes.

*Timings (median ms, batch 4, bf16). The Transformer column is the fused-SDPA arm from job 2579631.*

| L | baseline cs=64 | cs=128 | compiled | compiled+128 | Transformer | ours vs Transformer |
|---|---|---|---|---|---|---|
| 1,024 | 51.35 | 39.47 | 13.01 | 11.56 | 8.70 | 0.75× |
| 2,048 | 90.78 | 66.24 | 20.96 | 17.37 | 16.12 | 0.93× |
| 4,096 | 150.12 | 103.53 | 45.20 | 45.25 | 33.32 | 0.74× |
| 8,192 | 286.50 | 199.15 | 88.80 | 88.80 | 71.34 | 0.80× |
| **16,384** | 587.01 | 416.52 | 140.36 | **122.68** | 163.86 | **1.34× faster** |

**There is now a crossover, and the headline of this plan has changed.** At 16,384 tokens the
compiled model at `chunk_size=128` runs in **122.68 ms against FlashAttention's 163.86 ms** — 1.34×
*faster*, at 7.169 GB against 7.152 GB. From 8,192 to 16,384 our latency exponent is **0.47**
(0.66 compiled alone) while attention's rises to **1.20**. The asymptotic advantage the architecture
was supposed to have is visible in measured data for the first time; the plan's §1 statement that
there is "no crossover and no asymptotic speed advantage" was true of the uncompiled path only and
is superseded here.

*Rule R3.* `chunk_size=128` alone: ≥1.25× at 5 of 7 lengths, never slower — **adopt**.
`torch.compile`: 3.2×–5.2× everywhere — **adopt**. Stacking them adds 1.21× at 2,048 and 1.14× at
16,384 but exactly nothing at 4,096 and 8,192 (see the anomaly below).

*Rule R2, resolved rather than waived.* The compiled arm peaks at 7.169 GB against the Transformer's
7.152 GB — 17 MB, 0.24%, over the bar. R2 rejects that **as a default** but explicitly permits it
behind an opt-in flag that is off by default, which is exactly what `torch.compile` already is: the
project pins `compile_model=false` everywhere and this plan changes no training config. So compile
is adopted as an **inference-time opt-in**, and the uncompiled 7.090 GB remains the default-path
number. Compile also *lowers* the memory exponent, 0.544 against 0.643.

*⚠ One anomaly, not yet explained.* The compiled cs=64 and cs=128 arms timed **identically** at
L=4,096 (45.205 vs 45.250 ms) and L=8,192 (88.802 vs 88.803 ms) — one microsecond apart on an 89 ms
measurement — while the uncompiled arms at those same lengths differ by 45%. Two candidates with
opposite consequences: the override stopped reaching the operator under compile (a plumbing bug,
making the 1.14× at 16,384 the real effect showing only where the cache missed), or those lengths
genuinely plateau once Inductor removes the loop overhead. All arms shared one
`TORCHINDUCTOR_CACHE_DIR`, so a loosely-keyed cache hit is live. `profile_e1_followup_h100.sh` part A
gives each arm its own cache and `performance_profile.py` now prints the chunk size read back **off
the built module**, which separates the two. **No stacking claim should be made until that lands.**

*What this does to E2 and E3.* Both were justified by a gap that is now closed. `torch.compile`
already removes the loop overhead E2 was designed to remove — and it compiled L=16,384 fine here in
3.8 s with a warm cache, so even the FE4 graph-explosion argument for E2 is weaker than it looked
after job 2579642's timeout. **Recommendation: do not run E2 or E3.** An external CUDA dependency
(E3) buys nothing once we are ahead of FlashAttention at the length that matters, and it would add a
reproducibility liability to a thesis artefact. The remaining honest work is E1's follow-up, the
untested training path, and E5.

**FOLLOW-UP RESULTS — job 2582482 on gx10, 2026-09-25. It resolved the anomaly and, in doing so,
invalidated the protocol that produced the headline.**

*Part A — the anomaly was a compiler-cache artifact, not a plumbing bug.* Both arms printed their
own `effective chunk_size` off the built module, so the override always reached the operator. But
with a **per-arm** Inductor cache the same points came out 21–30% faster:

| point | shared cache (2580198) | isolated cache (2582482) | |
|---|---|---|---|
| L=4,096, cs=64 | 45.21 ms | **35.60 ms** | 21% faster |
| L=4,096, cs=128 | 45.25 ms | **31.79 ms** | 30% faster |
| L=8,192, cs=64 | 88.80 ms | 87.42 ms | 1.6% |
| L=8,192, cs=128 | 88.80 ms | 88.21 ms | 0.7% |

A shared cache had handed both arms one slow kernel at L=4,096 — which is exactly why they matched
to a microsecond. At L=8,192 the two agree inside 2%, so the plateau *there* is real.

*And a second, larger effect: within one process, shapes compiled later measure worse.* The compiled
training arm ran 3.04× at L=512 and 1.02× by L=4,096; in 2580198, L=4,096 was the fifth shape
compiled and was the slowest. **Every compiled number this project holds is a function of what else
was compiled beside it.** The 1.34×-vs-FlashAttention result at L=16,384 came from a shared-cache
run and is therefore *unconfirmed*. It is likely to improve rather than regress — isolation made
every re-measured point faster — but it must not be written into the thesis until E1-E lands. Both
wrappers now isolate caches per arm; E1-E goes further and gives every length its own process.

*Part B — the training path, measured for the first time, and incomplete.* The job hit its 2 h limit
inside the compiled arm at L=8,192, so `train_compiled_chunk128` and **the Transformer reference row
never ran** — which is the row that makes the rest mean anything.

| L | baseline | cs=128 | compiled | cs=128 speedup | compile speedup | cs=128 memory |
|---|---|---|---|---|---|---|
| 512 | 101.43 ms | 77.02 | 33.42 | 1.32× | 3.04× | +5.2% |
| 1,024 | 168.56 | 119.47 | 56.72 | 1.41× | 2.97× | +5.9% |
| 2,048 | 310.06 | 210.04 | 282.49 | 1.48× | **1.10×** | +6.3% |
| 4,096 | 543.40 | 371.46 | 532.02 | 1.46× | **1.02×** | +6.5% |
| 8,192 | 1337.35 | 884.13 | — | 1.51× | — | +6.6% |

`chunk_size=128` is a steady ~1.45× on the training step but costs 5–7% more memory, which matters
far more here than in inference: training already needs 13.8 GB at L=2,048 and 54.4 GB at 8,192.
Compile's collapse from 3.04× to 1.02× is the same later-shape degradation described above, so it is
a measurement artefact until E1-E re-measures it one shape per process. **No training claim should
be made from this table** — it has no Transformer column.

### E2 — Remove the sequential inter-chunk loop (pure PyTorch, no dependency)

*Pre-registered prediction (R4): 2–4× on the Mamba-3 layers at 16,384; the binding risk is R2, not
correctness.*

The inter-chunk carry is `state_c = g_c · state_{c-1} + contrib_c` with
`g_c = exp(A_cum[:, c, -1]) · carry_ok[:, c, -1]`, scalar per `(batch, head, chunk)`. Computing every
`contrib_c` in one batched einsum and combining them with `exp(segsum(log g))` over the chunk axis
replaces `nc` sequential steps with `O(1)` kernel launches and an `(nc, nc)` matmul per
`(batch, head)` — the identical trick already used for the intra-chunk mask, applied one level up.
Document resets carry over unchanged: `carry_ok = False` sets `log g = -inf`, which `segsum` already
encodes.

- [ ] **E2-A** Test first (TDD). Extend `tests/test_mamba3_numerics.py`: the parallel form must
  match `ssd_chunked_scan` and `ssd_sequential_reference` under R1, including (i) `cu_seqlens` with
  a document boundary inside a chunk, (ii) `seqlen % chunk_size != 0` padding, (iii) `ngroups > 1`,
  (iv) `extra_terms` non-empty (trapezoid/bias arms), (v) `nc == 1`. Tests fail before the
  implementation exists.
- [ ] **E2-B** Implement behind `ssd_impl: {"loop" (default) | "parallel"}`, threaded like the
  existing `scan_impl`/`tfla_impl` flags — parameter-invisible, so the same weights load either way,
  and pinned explicitly in every m3 yaml per the V1-E rule.
- [ ] **E2-C** Measure. Peak memory is the risk: the batched `contrib` tensor is
  `(batch, nc, nheads, headdim, dstate)` ≈ 0.4 GB (bf16) / 0.8 GB (fp32) at batch 4, seqlen 16,384.
  **R2 applies:** if peak exceeds the Transformer's 7.15 GB, `parallel` does not become the default.
- [ ] **E2-D** Fallback if R2 bites: two-level blocking — loop over super-blocks of `k` chunks,
  parallel within each. Cuts iterations by `k` while capping extra memory at `1/k` of E2-C's. Pick
  `k` from the E2-C Pareto front rather than guessing.
- [ ] **E2-E** Confirm the O(1) decode path (`ssd_step`) is untouched, and that
  `tests/test_mamba3_numerics.py`'s cached-vs-uncached token-identity test still passes. Decode is
  where the architecture already wins (5.19× per token); this must not regress it.

### E3 — Fused external kernel (gated on E0-F, not assumed)

*Entered only if E0-F's bound justifies it and E2 has landed. Pre-registered prediction (R4): the
dependency build is the risk that materialises, not the numerics.*

- [ ] **E3-A** Feasibility spike, timeboxed to one day: does `mamba-ssm` build against the cluster's
  CUDA and torch in the `.venv`? Answer on a compute node via `srun`/`sbatch`, never on lx01. If it
  does not build cleanly, stop and record that as the finding.
- [ ] **E3-B** Bind `mamba_chunk_scan_combined` behind `scan_impl: "fused"` for `Mamba3Block` only,
  active only when `use_trapezoid/use_rope/bc_bias` are at their Mamba-2 defaults; every other flag
  combination falls back to the PyTorch path with a warning. Gate on R1 against the fp64 oracle.
- [ ] **E3-C** Re-measure the 14A-7 sweep; report against E0-D's unfused-attention line and the
  fused one.
- [ ] **E3-D** Record the dependency cost honestly: a fused path that only exists on one cluster's
  build is a caveat on every number it produces.

### E4 — The other three layers: mLSTM TFLA

*Entered only if E0-A says mLSTM is a material share.*

- [ ] **E4-A** Apply the E0-B treatment to `mlstm_block.py` / `tfla_interface.py`: where does its
  time go, and does it carry the same sequential-chunk structure?
- [ ] **E4-B** If it does, apply E2's parallel-carry transformation behind a `tfla_impl` value, under
  the same R1/R2/R3 gates. `tfla_impl: exact` must remain byte-compatible.

### E5 — Writeup, and remove the artefacts that caused the question

*No cluster needed.*

- [ ] **E5-A** `analysis/EFFICIENCY_NOTE.md` — supervisor-facing, matching
  `analysis/PHASE14_SUPERVISOR_REVIEW.md` in form: the Triton premise answered factually, the
  memory-parity win, the wall-clock gap, E0-D's fused-vs-unfused separation, and the Amdahl bound.
  **Allowlist it in `.gitignore` in the same commit** (the blanket `*.md` at line 84 silently ate
  `analysis/ARCHIVE_MANIFEST.md` on 2026-09-20).
- [ ] **E5-B** Update `analysis/mamba3_results.md` §6 (Efficiency) with the measured numbers and the
  ceiling. State explicitly that no report-generation metric changes.
- [ ] **E5-C** Delete the misleading artefacts: root `test_triton_fix.py` (stale Colab-era script
  whose docstring says "Test script to verify the Triton kernel fix") and the unused
  `triton>=2.1.0` line in `requirements.txt:8`. Add a parity test asserting no tracked file claims a
  Triton kernel that does not exist.
- [ ] **E5-D** `venv/bin/python scripts/mamba3_state.py --plan efficiency readme`-equivalent
  bookkeeping: final verdict in `efficiency_state.json`, one note appended to
  `mamba3_v2_state.json` if any efficiency number cited there changes. **No merge into
  `h100_scaling`.**

## 5. Budget

| Phase | GPU-h | Wall | Blocking? |
|---|---|---|---|
| E0 | ~2 | 1 job + queue | yes — gates everything |
| E1 | ~2 | 1 job | no |
| E2 | ~3 | ~1–2 d (mostly local TDD) | no |
| E3 | ~4 | ~1 d spike + 1 d, or abandoned at E3-A | gated on E0-F |
| E4 | ~2 | ~1 d | gated on E0-A |
| E5 | 0 | ~0.5 d | no |

Total if everything runs: ~13 GPU-h, ~5 days wall. **E0 + E1 + E5 alone is ~1.5 days and already
answers the supervisor.**

## 6. Risks

- **FE1 — Optimising the wrong thing.** Mitigated by E0 blocking every other phase. The 4.14× gap
  is assumed to be the scan; that is a hypothesis until E0-B.
- **FE2 — Breaking memory parity (R2).** The published memory result is worth more than the speed
  result. E2-C is where this bites; E2-D is the escape.
- **FE3 — Silent numerical drift.** A faster operator that changes a decoded token invalidates
  published metrics. R1 plus the existing fp64 oracle is the guard; `cu_seqlens` boundaries are the
  known-hard case (the clamp defect lived exactly there).
- **FE4 — `torch.compile` graph explosion.** Dynamo will try to unroll `for ci in range(nc)` into
  `nc` copies. Expect long compile times or a recompile storm on shape change. E1-B measures compile
  time, not just steady state.
- **FE5 — Dependency rot (E3).** An external CUDA extension that builds today on one cluster is a
  reproducibility liability. E3-D records it; E3-A can stop the phase.
- **FE6 — Scope creep into training.** Everything here is inference-path. Touching the training path
  would invalidate checkpoints and is out of scope.
- **FE7 — The `.gitignore` `*.md` trap.** Any new markdown deliverable must be allowlisted in the
  same commit; `test_every_analysis_deliverable_can_actually_enter_the_repo` enforces it for
  `analysis/`, and this plan file is allowlisted at `!EFFICIENCY_PLAN.md`.

## 7. State-tracking contract

After **every** meaningful change — test written, phase gated, job submitted, job finished, number
recorded — tick the checkbox and update the state file through the helper, never by hand:

```
venv/bin/python scripts/mamba3_state.py --plan efficiency tick E0-A --note "..." --evidence job=NNNN
venv/bin/python scripts/mamba3_state.py --plan efficiency show E0
venv/bin/python scripts/mamba3_state.py --plan efficiency sync     # rebuild state from checkboxes
```

`efficiency_state.json` is the resumable record; the checkboxes above are ground truth. Local
commands are written `venv/bin/python …` on purpose so they are never pasted into the aisc login
node, which executes nothing scripted — cluster work goes through `sbatch`/`srun`/`source`.

## 9. Cluster runbook (aisc)

The login node `lx01` executes nothing scripted. Everything below is `git`, `sbatch`, `squeue` or
`tail`; no bare `python` line appears in this file for that reason.

**Once, to get onto the branch:**

```bash
cd ~/hybrid_mamba_xlstm
git fetch origin
git checkout -b h100_efficiency origin/h100_efficiency   # first time
# later:  git checkout h100_efficiency && git pull
```

**E0 — the blocking measurement (~30–45 min on one H100, 1.5 h wall limit):**

```bash
sbatch scripts/profile_layer_split_h100.sh
squeue -u $USER
tail -f logs/h100_layersplit_<jobid>.log
```

That job runs E0-A, E0-B and both E0-D arms. Results land in
`analysis/efficiency_layer_split/{per_layer,attn_auto,attn_math}/`.

**E0-C + E1-B, the optional arms (adds ~30 min):**

```bash
CHUNK_ARM=true COMPILE_ARM=true sbatch scripts/profile_layer_split_h100.sh
```

**What to read out of the log, in order:**

1. The `Amdahl:` line under E0-A at L=16384. If the SSD path is under ~50% of the forward, the
   ceiling is below 2× and **E2/E3 are not worth running** — record that and go to E5.
2. `-> of which ssd_chunked_scan` — if the scan is a small share of Mamba-3 time, the §2 hypothesis
   is wrong and the cost is in the projections instead.
3. The two E0-D sweeps side by side. `attn_math` is expected to OOM at the top of the ladder; that
   OOM is the FlashAttention memory story stated as data, not a failed run.
4. Under `CHUNK_ARM`, whether latency falls roughly linearly in `nc`. That is the launch-bound
   signature that justifies E2.

**Nothing here writes to `outputs/`, `results/` or any checkpoint.** The job profiles randomly
initialised weights on random token ids, so it needs no HF cache, no MIMIC data, and cannot touch a
DUA-covered artefact or a published number.

## 8. Unresolved questions

- E0-D: report unfused attention as the headline comparison, or as a footnote to the fused one?
- E2 default: ship `ssd_impl: parallel` as default if it passes R1–R3, or leave opt-in to keep the
  published path byte-identical?
- E3 at all: is an external CUDA dependency acceptable in a thesis artefact?
- E1-A: if a non-64 `chunk_size` wins, re-decode the 13D dump to confirm token identity, or rely on
  R1 alone?
