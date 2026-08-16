# Mamba-3 Integration Plan — Hybrid Mamba-xLSTM (Plan-of-Record, subordinate to H100)

> Resumable plan-of-record. Read this + `mamba3_integration_state.json` alongside
> `H100_SCALING_PLAN.md` + `h100_scaling_state.json`.
> **Subordinate to `H100_SCALING_PLAN.md`** — that plan owns the retrieval thesis and stays
> at `current_phase: phase7_data_diversification`. This plan owns the *backbone operator*.
> Source: Mamba-3 (Lahoti, Li, Chen, Wang, Bick, Kolter, Dao, Gu; arXiv:2603.15569).

## Why this plan exists (and what it must NOT claim)

A pre-implementation audit (2026-08-16, all numbers reproduced on CPU in this repo) found a
**correctness defect in the live selective scan**, not merely a tuning opportunity. Fixing it is
justified independently of any metric. Three secondary facts frame everything below:

1. **The live Mamba block is `layers/mamba_block.py`.** `hybrid_block.py:11` imports it.
   `layers/mamba_block_v2.py` and `layers/hybrid_layer.py` are **dead code** — nothing in the
   `HybridLanguageModel` path reaches them. `mamba_block_v2.py:159-179` already contains a
   correct `_init_dt_proj`; it has simply never been wired in.
2. **The Triton selective-scan kernel is also dead.** `scan_interface.selective_scan()`
   unconditionally calls `selective_scan_parallel` (the fp32 PyTorch chunk scan);
   `selective_scan_triton` is imported and never invoked. `CLAUDE.md`'s claim that Mamba
   "uses chunk-parallel selective scan Triton kernel" is false for the live path and must be
   corrected in Phase M7. This also bears on the Phase-8A-eff efficiency curves.
3. **Backbone quality does not move retrieval in this project.** `H100_SCALING_PLAN.md` records
   10 nulls, including *Stage-0 PPL 15.62→13.18 = flat* and *70M→150M = flat*. Every change in
   this plan is a text-backbone change.

**Therefore the success metric for this plan is Stage-0 LM perplexity + operator correctness.**
Retrieval is a **non-regression guard**, never a target. Any phase whose kill-criterion is
"did MIMIC R@10 rise" would repeat the exact error the 2026-07-25 post-mortem documents.

### Honest framing of the existing results

The defect does **not** invalidate the published numbers. Training and evaluation used the same
operator, so `MIMIC 0.1459` (clean) / `0.1714` (`val==test`) are valid measurements of the system
as built. The correct statement is narrower and must survive into the writeup: **the block did not
compute the Mamba recurrence it was specified to compute.** Whether the true recurrence is better
is what Phase M4 measures — it is not assumed here.

---

## Audit results — what was verified, what was refuted

Reproduced with `float64` sequential ground truth on CPU. `A` init = `-[1..16]`, chunk=64.

### Finding 1 — the live block has no `dt` initialization (CONFIRMED)

| config | Δ mean | Δ max | max \|Δ·A\| per step |
|---|---|---|---|
| `norm_topology: pre_rms` | 0.696 | 0.955 | 15.3 |
| **`norm_topology: hybrid` (canonical v2)** | **0.823** | **4.654** | **74.5** |

Reference Mamba initializes Δ ~ `logU[1e-3, 1e-1]`. The canonical config is the **worst** case.

### Finding 2 — `dt_norm` erases the proposed fix (NEW — not in the source analysis)

`mamba_block.py:137-141` applies `RMSNorm` to Δ **before** `softplus`. RMSNorm rescales to unit
RMS across `inner_dim`, discarding the bias offset the Mamba init encodes.

| | Δ mean | Δ max |
|---|---|---|
| what the proposed init asks for | 0.021 | 0.100 |
| what `dt_norm` + softplus delivers | **0.332** | 0.486 |

**Consequence: applying the `dt_proj` init without touching `dt_norm` is a no-op on every v2
config.** Mamba-3 §3.4 specifies BCNorm on **B and C only** — normalizing Δ is this repo's own
addition and is the blocker. M2 resolves this.

### Finding 3 — divide-and-clamp is wrong well inside the intended Δ range (CONFIRMED, worse than reported)

`scan_interface.py:118-119` computes `Bx / A_cum.clamp(min=1e-8)`. The clamp fires when
cumulative decay exceeds `log(1e8)=18.4` — i.e. ~30 orders of magnitude before fp32 underflow.

| Δ | max \|Δ·A\| | current (div+clamp) | segsum |
|---|---|---|---|
| 0.001 | 0.02 | 6.2e-17 | 1.2e-16 |
| 0.01 | 0.2 | 3.0e-16 | 2.9e-16 |
| **0.1** | 1.6 | **5.391e-01** | 7.9e-16 |
| 0.3 | 4.8 | 7.194e-01 | 4.4e-16 |
| **0.705** (measured init) | 11.3 | **1.053e+00** | 3.4e-16 |

Error appears at **Δ=0.1 — the `dt_max` of the standard init**, not only at Δ=0.7.

**Under the proposed `logU[1e-3,1e-1]` init the scan is still wrong**: 16.2% of `(d,n)` channels
hit the clamp inside a 64-chunk, **29.7% of channels exceed 1% relative error**, overall relative
max error **3.6e-01**. The source analysis's claim that the dt init "moves your current scan back
into its correct regime" is **refuted**. #2 is the fix; #1 is necessary but not sufficient.

Mechanism: where `A_cum[s]` underflows to 0 while `A_cum_safe[s]=1e-8`, the diagonal term
`h_intra[s] = A_cum[s] · (Bx[s]/1e-8) = 0` — **the current token's own contribution to the state
is annihilated**, not merely perturbed.

### Finding 4 — the proposed segsum does not port to this repo (NEW)

Mamba-2/3 segsum is cheap because `A_t` is **scalar per head**, giving a `(cs,cs)` mask. This repo
is Mamba-1 style: `A` has shape `(d_inner, d_state)`, so the mask is `(cs,cs,d,n)`.

| chunk | batch | mask size (150M, d_inner=1536, n=16) |
|---|---|---|
| 64 | 48 | 4.83 G elem = **19.3 GB fp32** |
| 32 | 48 | 1.21 G elem = 4.8 GB |
| 16 | 48 | 0.30 G elem = **1.2 GB** |

Verified alternative — **shrinking `chunk_size` fixes it exactly with the existing code**, but only
once Δ is initialized:

| chunk | rel err @ Δ~logU[1e-3,1e-1] | seq. steps @ L=1024 |
|---|---|---|
| 64 | 3.6e-01 | 16 |
| 16 | 6.9e-02 | 64 |
| **8** | **1.5e-16** | 128 |

At the *uninitialized* Δ=0.705, **no chunk size rescues it** (chunk=2 still 4.0e-01) — a single
step already underflows the clamp. **M1 and M2 are therefore coupled and must land together.**

### Finding 5 — why this survived (CONFIRMED)

`tests/test_kernels.py::test_scan_forward` asserts only shape / no-NaN / no-Inf, never numerical
correctness — and is CUDA-gated, so it never runs in `validate_for_willi.sh`. Closed by M0.

### Finding 6 — TFLA is half-fixed already

`tfla_interface.py:149` already uses the log-space difference form for the inter-chunk decay.
`:93-95` still divides by `f_cum.clamp(min=1e-6)` for the intra-chunk term. Only the intra-chunk
half needs work. Scope-limited in M1c.

### Verdict table

| # | Proposed change | Verdict | Phase |
|---|---|---|---|
| 1 | `dt` init, inverse-softplus bias | **Correct, blocked by `dt_norm`, insufficient alone** | M2 |
| 2 | Replace divide-and-clamp | **Correct diagnosis; prescription does not port** (Finding 4) | M1 |
| 3 | `A_floor` + bounded `A` | **Plausible, unverified against reference, no observed failure** — ablate | M3 |
| 4 | Keep BCNorm | **Confirmed present & convergent with §3.4** — but drop the Δ-norm | M2 |
| 5 | Exponential-trapezoidal | Sound; λ=1 gives a bit-identical control | M5 |
| 6 | Learnable B/C biases | Sound, cheap; ablate jointly with #5 per Table 5a | M5 |
| 7 | Drop depthwise conv | Only valid after 5+6 land positive | M6 |

**Not adopted:** complex/RoPE SSM (§3.2) and MIMO (§3.3). Both are large rewrites justified by
state-tracking and decode arithmetic intensity — neither is a bottleneck for a 150M biomedical
retrieval encoder, and MIMO's gain is reported at 1.5B on 100B FineWeb-Edu tokens. Revisit only
if M5 lands positive and time remains. Recorded so it is not silently re-proposed.

---

## Success bar

**Primary — Stage-0 val PPL** (inherits `H100_SCALING_PLAN.md` tiers): floor ≤ 15.62, target
≤ 13.76, stretch ≤ 13.10. Best on record: **13.18** (Phase 5, buggy scan).

**Primary — correctness:** scan relative max error ≤ 1e-6 vs float64 sequential reference across
Δ ∈ {1e-3 … 1.0} and A ∈ {-1 … -16}, on CPU, inside `validate_for_willi.sh`.

**Guard — retrieval non-regression:** MIMIC i2t R@10 ≥ 0.1459 (clean protocol) when the corrected
backbone is re-run through the canonical contrastive recipe. **A null here is the expected
outcome and is not a failure of this plan.** Only a *regression* is.

**Guard — throughput:** ≤ 1.5× Stage-0 step-time regression vs the chunk=64 baseline. M1 trades
sequential steps for correctness; if it exceeds 1.5×, take the M1b masked-segsum branch.

---

## Critical files

| File | Phase | Action |
|---|---|---|
| `tests/test_kernels.py` | M0 | CPU correctness tests vs sequential reference (must fail first) |
| `tests/test_layers.py` | M0/M2 | Δ-at-init probe |
| `hybrid_xmamba/kernels/selective_scan/scan_interface.py` | M1 | replace divide-and-clamp |
| `hybrid_xmamba/kernels/tfla/tfla_interface.py:93-95` | M1c | intra-chunk decay only |
| `hybrid_xmamba/layers/mamba_block.py` | M2/M3/M5/M6 | dt init, `dt_norm`, A bound, trapezoid, B/C bias, conv |
| `hybrid_xmamba/models/configuration_hybrid.py` | M2-M6 | new flags, all defaulting to current behaviour |
| `configs/model/hybrid_150m_v2_m3.yaml` (NEW) | M4 | corrected-operator variant |
| `scripts/train_stage0_150m_h100.sh` | M4 | short-run + full Stage-0 |
| `CLAUDE.md` | M7 | correct the Triton/dead-code claims |
| `analysis/mamba3_integration_results.md` (NEW) | M7 | results + writeup |

**Every new behaviour goes behind a config flag defaulting to today's behaviour**, so each phase
is a single-variable ablation and every existing checkpoint keeps loading.

---

## Phases

### Phase M0 — Pin the defect (tests only, NO behaviour change)
Test-first: these must **fail on `HEAD`** before any fix lands. This is the phase that makes every
later claim checkable.
- [ ] **M0-A** — `test_selective_scan_matches_sequential_reference`: float64 sequential reference;
      parametrized over Δ ∈ {1e-3, 1e-2, 1e-1, 0.3, 0.705, 1.0} × chunk ∈ {8, 64}. **CPU, not
      CUDA-gated.** Expected on HEAD: passes at Δ≤0.01, fails at Δ≥0.1.
- [ ] **M0-B** — `test_delta_at_init_is_in_mamba_range`: assert Δ mean ∈ [1e-3, 1.5e-1] at init for
      **both** `pre_rms` and `hybrid`. Expected on HEAD: fails (0.696 / 0.823).
- [ ] **M0-C** — `test_tfla_intra_chunk_matches_sequential_reference` (same shape, TFLA).
- [ ] **M0-D** — Mark all three `xfail(strict=True)` with a docstring citing this plan, so
      `validate_for_willi.sh` stays green on HEAD and **flips loudly** the moment M1/M2 fix it.
- [ ] **M0-E** — `validate_for_willi.sh` green; commit on branch `mamba3_integration`.

### Phase M1 — Correct the selective scan (Tier-1 #2) — **the load-bearing phase**
Depends on nothing. Blocks everything else.
- [ ] **M1-A** — Implement **M1a: adaptive chunk size.** Pick `chunk_size` so
      `max(Δ·|A|)·chunk ≤ 12` (safe margin under the `log(1e8)` clamp), floor 4, cap 64. Cheapest
      correct fix; no new memory; verified exact at chunk=8. Keeps the existing code path.
- [ ] **M1-B** — Benchmark M1a: Stage-0 step time + peak memory vs chunk=64 baseline, 150M config,
      L=1024, bs=48. **Gate: ≤1.5× step time.** Reuse `scripts/performance_profile.py --sweep`.
- [ ] **M1-C** — If M1-B fails the gate: implement **M1b: masked log-space segsum** at chunk=16
      (`exp(clamp(A_cum[t]-A_cum[s], max=0))`, materialized `(cs,cs,d,n)` = 1.2 GB @ bs=48, no
      reciprocal anywhere). Re-benchmark. Record which branch was taken and why.
- [ ] **M1-D** — TFLA intra-chunk (`tfla_interface.py:93-95`) → same log-space difference form the
      inter-chunk path at `:149` already uses. Scope-limited: do **not** touch `:149`.
- [ ] **M1-E** — Flip M0-A/M0-C off `xfail`. **Gate: rel err ≤ 1e-6 at every Δ.**
- [ ] **M1-F** — Numerical smoke: forward/backward finite, grad-norm bounded, 50 steps no NaN,
      `i_gate < cap` (per `H100_SCALING_PLAN.md` verification item 3).
- [ ] **M1-G** — `validate_for_willi.sh` green; commit.

### Phase M2 — `dt` init + resolve the `dt_norm` blocker (Tier-1 #1, Tier-1 #4 caveat)
**Must land with M1** — M1a is only exact once Δ is in range (Finding 4).
- [ ] **M2-A** — Port `_init_dt_proj` from the dead `mamba_block_v2.py:159-179` into the live
      block. Config: `dt_min=1e-3`, `dt_max=1e-1`, `dt_init="random"`, `dt_scale=1.0`. New flag
      `dt_init_strategy: {none, mamba}`, **default `none`** (today's behaviour).
- [ ] **M2-B** — Add `norm_topology: hybrid_bc` — B/C norm **without** the Δ norm, matching
      Mamba-3 §3.4 exactly. Leave `hybrid` untouched so every existing checkpoint loads unchanged.
- [ ] **M2-C** — Flip M0-B off `xfail` under `dt_init_strategy=mamba` + `norm_topology=hybrid_bc`.
      Assert Δ mean ∈ [1e-3, 1.5e-1]. Also assert `hybrid` + `mamba` init **still fails** the range
      — pinning Finding 2 so it cannot silently regress.
- [ ] **M2-D** — 2×2 CPU probe (200 steps, tiny config, wikitext): `{none, mamba}` ×
      `{hybrid, hybrid_bc}`. Report train loss + Δ distribution. Cheap; decides M4's arm list.
- [ ] **M2-E** — `validate_for_willi.sh` green; commit.

### Phase M3 — Bounded `A` parameterization (Tier-1 #3) — **ablate, do not adopt**
Lowest evidence of the Tier-1 set: no failure observed in this repo, and the "reference bounds it
with a heavy-tail activation" claim could not be verified against Mamba-3 source.
- [ ] **M3-A** — Flag `a_floor: Optional[float] = None` (default off). When set, `A = -exp(A_log)`
      → `A = -(softplus(A_log_raw) + a_floor)` with `a_floor=1e-4`, bounded above.
- [ ] **M3-B** — Assert bit-identical output when `a_floor=None`. Non-negotiable.
- [ ] **M3-C** — Carry as a **free rider** in the M4 short-run only. If it neither helps nor hurts
      PPL, keep it off and record it as a null. Do not spend a dedicated arm on it.

### Phase M4 — Stage-0 re-baseline (**the decision gate**) — requires H100
This is where the plan finds out whether correctness buys anything. Nothing downstream runs first.

**Sequencing constraint:** M1+M2 change the operator, so the Phase-5 Stage-0 checkpoint
(PPL 13.18) is **not** comparable to a corrected-operator model. A re-run is required to make any
PPL claim. Short-run first so a 4-day job is never spent on a guess.
- [ ] **M4-A** — **Short run, 8K steps**, 150M, aisc-shortrun. Arms:
      **(i)** control = current operator (reproduces the Phase-5 curve — validates the harness);
      **(ii)** M1+M2 corrected; **(iii)** M1+M2+M3.
      Compare val-loss curves at matched steps. **Gate: (ii) ≤ (i) at 8K.**
- [ ] **M4-B** — If (ii) is *worse* than (i): **stop and report.** That is a real, publishable
      finding — the buggy operator was acting as an unintended regularizer — and it kills M5/M6.
      Do not tune around it.
- [ ] **M4-C** — If (ii) ≥ (i): full Stage-0, `max_steps=120000`, aisc-batch 4-day, per
      `scripts/train_stage0_150m_h100.sh`. **Gate: PPL ≤ 13.18** (beat the buggy-operator record);
      target ≤ 13.76 tier already cleared by the incumbent, so 13.18 is the real bar.
- [ ] **M4-D** — Record verdict in `mamba3_integration_state.json` **and** as a note on
      `h100_scaling_state.json` — Phase 5's result is superseded either way.

### Phase M5 — Exponential-trapezoidal + B/C biases (Tier-2 #5 + #6) — gated on M4-C
Table 5a shows bias alone is worth 0.19 ppl (16.68→16.49) but bias+trap 0.96 (→15.72), so these
are ablated **jointly**, not separately. Caveat to carry into the writeup: that gain is at 440M on
FineWeb-Edu LM, not 150M biomedical.
- [ ] **M5-A** — `λ_t = sigmoid(lambda_proj(x))`, `+n_heads` columns on `in_proj`; recurrence
      `h_t = α_t h_{t-1} + β_t B_{t-1}x_{t-1} + γ_t B_t x_t` per Prop. 1, implemented as a width-2
      convolution on the state-input **inside** the recurrence (Remark 4 — not an outer conv).
- [ ] **M5-B** — **Bit-identity control: assert λ≡1 reproduces M1 output exactly.** This is the
      property that makes the arm interpretable; without it M5 is uninterpretable.
- [ ] **M5-C** — Learnable head-wise channel-wise B/C biases after BCNorm, init 1.0.
- [ ] **M5-D** — Short-run 2×2: `{trap on/off}` × `{bias on/off}`, 8K steps. **Gate: >0.1 PPL over
      the M4-C control** (below that is noise at this scale).
- [ ] **M5-E** — If gate cleared: full Stage-0 re-run with the winner.

### Phase M6 — Drop the depthwise conv (Tier-2 #7) — gated on M5-D clearing
Per Table 5a, `Mamba-3 + conv` (15.85) is slightly **worse** than without (15.72), so this is a
parameter saving, not a gain. Invalid before M5 lands.
- [ ] **M6-A** — `use_conv: bool = True` flag; assert bit-identical when `True`.
- [ ] **M6-B** — Short-run A/B. **Gate: PPL within +0.05 of M5's winner** — accept only if it does
      not cost quality; the payoff is params + a kernel, not accuracy.

### Phase M7 — Reintegration with the H100 plan + writeup
- [ ] **M7-A** — **Retrieval non-regression.** Best backbone from M4/M5 → canonical contrastive
      recipe (`vit_unfreeze=12`, `vit_lr=1e-6`, `scope=blocks`, `bs=64`, `head_lr=4.24e-4`) under
      the **clean protocol** (`train[:85%]` / `train[85%:90%]` / `train[90%:]`).
      **Gate: MIMIC i2t R@10 ≥ 0.1459.** Expected null per the 10-null record; only a regression
      blocks. Do **not** report a positive here without the ±0.57pp SE attached.
- [ ] **M7-B** — Re-run `sbatch scripts/profile_efficiency_h100.sh`. M1 changes the scan's
      sequential-step count, so every Phase-8A-eff scaling exponent must be re-measured.
- [ ] **M7-C** — `CLAUDE.md` corrections: the live path uses the **PyTorch chunk-parallel** scan,
      not Triton; `mamba_block_v2.py`, `mlstm_block_v2.py`, `hybrid_layer.py` and
      `scan_triton.py` are dead code (delete or document — do not leave them ambiguous).
- [ ] **M7-D** — `analysis/mamba3_integration_results.md`: audit table, per-lever ablation, the
      "operator was not the specified recurrence" framing, and the honest negative results.
- [ ] **M7-E** — Update `H100_SCALING_PLAN.md` + both state files; hand control back to
      `phase7_data_diversification`.

---

## Verification (every phase gates on)
1. `bash scripts/validate_for_willi.sh` exits 0 — py3.9 syntax hygiene, `Optional[X]` not `X | Y`,
   `Dict`/`List` not `dict`/`list`.
2. New phase test passes; the corresponding M0 `xfail` flips.
3. Every new flag asserted **bit-identical** in its default (off) state.
4. Forward/backward finite on CPU + H100 smoke; grad-norm bounded; no NaN 50+ steps.
5. Reconcile in-training vs authoritative eval numbers before citing any figure.

## Resumability contract
1. Read this file + `mamba3_integration_state.json` **and** the two H100 files at session start.
2. Resume at `mamba3_integration_state.json["current_phase"]`. Checkboxes here are ground truth.
3. After every state change: tick the checkbox, update `last_updated` (ISO 8601), append a
   one-line `notes` entry.
4. Never re-run a checkpoint-producing phase (M4, M5-E) without first reading its log and logging
   a verdict.

## Lessons carried in (do not repeat)
- **Never gate a backbone change on retrieval.** 10 documented nulls; Stage-0 PPL 15.62→13.18 was
  flat on MIMIC. PPL is the metric for this plan.
- `norm_topology` must be threaded into **every** `HybridConfig` builder (train / distill /
  contrastive) — silently dropped means a wrong forward pass. `hybrid_bc` inherits this hazard.
- Eval must auto-detect `layer_pattern` + `norm_topology` from the checkpoint; a new topology value
  means `evaluate_cxr_retrieval.py` and `evaluate_lm.py` need the same treatment.
- Shape/NaN assertions are not correctness tests. That is precisely how Finding 3 survived.
- The aisc login node refuses `bash <script>` — use `./scripts/...` or `sbatch` directly.

## Unresolved questions
1. Re-run Stage-0? M4 costs ~4 GPU-days and supersedes Phase 5. Alternative: freeze the operator,
   ship the audit as a documented limitation, keep the H100 plan on Phase 7. **Blocks M4.**
2. Priority vs Phase 7 — serial (M-plan first) or parallel on separate GPUs?
3. Delete the dead code (`mamba_block_v2.py`, `mlstm_block_v2.py`, `hybrid_layer.py`,
   `scan_triton.py`) or keep documented? Deleting risks breaking unaudited imports.
4. Is willi/A100 still a target? If retired, drop the py3.9 guards.
5. Complex/RoPE SSM (§3.2) + MIMO (§3.3) — confirm out of scope, or hold as a stretch phase?
