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

- [ ] **E0-A** Per-layer-type timing split. Extend `scripts/performance_profile.py` with
  `--per-layer`, using CUDA events around each `HybridBlock` (or `torch.profiler` with
  `record_shapes`), reporting ms and % by mixer type for `hybrid_150m_m3` at the 14A-7 sequence
  ladder. **This is the Amdahl bound and it gates E2/E3/E4.**
- [ ] **E0-B** Intra-block split for `Mamba3Block`: in/out projections, causal conv, the
  `ssd_chunked_scan` intra-chunk einsums, and the inter-chunk loop, measured separately. Confirms or
  kills the §2 hypothesis.
- [ ] **E0-C** Launch-overhead probe: for fixed seqlen 16,384, sweep `mamba3_chunk_size` over
  `{64, 128, 256, 512}` and record latency, peak memory and `nc`. If latency falls ~linearly in
  `nc`, the path is launch-bound and E2 is worth its cost; if flat, it is bandwidth-bound and E2 is
  not.
- [ ] **E0-D** **The supervisor's answer, as a measurement.** Re-run the 14A-7 inference sweep for
  `transformer_150m_baseline` with the fused backends disabled (`torch.nn.attention.sdpa_kernel`
  restricted to `MATH`), alongside the existing fused numbers. Add `--attn-backend {auto,math}` to
  the profiler. This separates *algorithm* from *kernel engineering*: if Mamba-3 beats unfused
  attention at long lengths, "the gap is implementation, not algorithm" becomes a number instead of
  a claim.
- [ ] **E0-E** Submit E0-A..D as one job (`sbatch scripts/profile_layer_split_h100.sh`, new CPU/GPU
  wrapper following the `profile_efficiency_h100.sh` conventions: `--partition=aisc-batch
  --account=aisc --gpus=1 --exclude=ga03,gx17v1,gx13v1`, no `--gres`). Write results to
  `analysis/efficiency_layer_split/`.
- [ ] **E0-F** Record the Amdahl bound explicitly in this file: *max achievable speedup if the
  Mamba-3 path became free* = 1 / (fraction not in Mamba-3). Compare to the 4.14× target and state
  plainly whether parity is reachable at all.

**Gate:** E0-F's bound decides what follows. If the bound is < 2×, skip to E4 and E5 and report the
ceiling as the result.

### E1 — Free levers: no new code path

*Pre-registered prediction (R4): `chunk_size` 64→256 gives 1.5–2.5× at 16,384 for < 1.5 GB extra;
`torch.compile` gives < 1.3× and may fail outright by unrolling the `nc` loop into a huge graph.*

- [ ] **E1-A** `mamba3_chunk_size` as an inference-time knob. It is already a config key and needs
  no code change. Verify equivalence under R1 across the sweep, then record the
  latency/memory Pareto front. **Note:** the trained checkpoints used 64; R1 is what licenses
  decoding at another value.
- [ ] **E1-B** `torch.compile` arm for inference only. Add `--compile` to `performance_profile.py`
  (it has no such flag today). **The documented reason compile is disabled does not apply here:**
  `MAMBA3_PLAN_V2.md:301` attributes `compile_model=false` to `mamba_block.py:248-276`'s Python loop
  over `(row, segment)` in the *Mamba-1* block, and states that SSD "handles boundaries natively …
  One masked-`exp`, fully batched". That decision was never re-evaluated after SSD landed. Measure
  compile time as well as steady-state latency; a 10-minute compile for a 1.1× gain is a null.
- [ ] **E1-C** Report both under R3. Anything that fails the bar is written up as a null and
  reverted.

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

## 8. Unresolved questions

- E0-D: report unfused attention as the headline comparison, or as a footnote to the fused one?
- E2 default: ship `ssd_impl: parallel` as default if it passes R1–R3, or leave opt-in to keep the
  published path byte-identical?
- E3 at all: is an external CUDA dependency acceptable in a thesis artefact?
- E1-A: if a non-64 `chunk_size` wins, re-decode the 13D dump to confirm token identity, or rely on
  R1 alone?
