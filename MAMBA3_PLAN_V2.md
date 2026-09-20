# Mamba-3 Backbone Upgrade — Plan of Record, version 2

> **ACTIVE plan-of-record. Read this + `mamba3_v2_state.json` at session start** (pointed to from `CLAUDE.md`).
> Branch: **`h100_mamba3_v2`**, cut from `h100_scaling` @ `8137221` (Phase 15 closed), with the Mamba-3 branch
> `h100_scaling_mamba3` (24 commits, M0–M7 complete) **merged in at `b191178`**. **No merge into `h100_scaling`
> without an explicit instruction from the user** — the Phase 14/15 results stay reachable and reproducible there.
> Resume at `mamba3_v2_state.json["current_phase"]`; the checkboxes below are ground truth.
> Supersedes `MAMBA3_PLAN.md` + `mamba3_state.json` (renamed to these files; the M0–M7 record is carried below
> verbatim) and, before that, `MAMBA3_INTEGRATION_PLAN.md` + `mamba3_integration_state.json` (retired at M0-E).
> `H100_SCALING_PLAN.md` + `h100_scaling_state.json` are the **closed baseline record** (Phases 1–15).
> Source: Mamba-3, arXiv:2603.15569 + reference impl `state-spaces/mamba` `mamba_ssm/modules/mamba3.py`.
> Tick with `venv/bin/python scripts/mamba3_state.py tick V1-G --note "..." --evidence k=v` (local; on the
> cluster nothing scripted runs outside `sbatch`/`srun`/`source`).

---

## What changed between version 1 and version 2

`MAMBA3_PLAN.md` was cut from `h100_scaling @ 20a1d27` on 2026-09-06. It finished M0–M7 (the defect pinned and
fixed behind flags, `Mamba3Block` as a new layer type, the O(1) decode cache, an 8-arm screen with a winner)
but never started M8. Meanwhile `h100_scaling` ran Phases 14 and 15 (37 commits), which changed what "the
baseline" means. Version 2 ports the finished work onto that branch and re-anchors every gate:

| Version 1 assumed | True on `h100_mamba3_v2` |
|---|---|
| Incumbent = 13D at one seed (ROUGE-L 0.1899 / CheXbert-14-micro 0.4736) | **3-seed bands** (15B-4): hybrid ROUGE-L 0.1949±0.0047, 14-micro 0.4480±0.0223, 14-macro 0.2660±0.0122; 13D is the high CheXbert draw |
| No Transformer baseline | **Parameter-matched Transformer** (`attention` layer type, 15 layers, 183.39M): Stage-0 PPL **11.222** vs hybrid 13.18; report-gen **indistinguishable on 8 of 10 metrics across seeds** |
| No seed lever | `SEED`, `SAVE_TOP_K` on the report-gen wrapper; paired bootstrap CIs and per-label CIs (`bootstrap_compare.py --per-label`) |
| No auxiliary loss | 15C aux CheXpert loss (`AUX_LAMBDA`, default 0.0): **negative, closed**; must stay 0 in every arm |
| Defect pinned on this branch only | Phase 14C-1 pinned it independently (`tests/test_scan_correctness.py`, `HYBRID_EXACT_SCAN=1` fp64 reference); 14C-2/3/4 never ran |
| Old M8-B retrains a contrastive tower | **13D's image tower is reused unchanged** (14A-4 protocol; user decision 2026-09-17) |
| Old M8 runs A2 as screened (`tfla_impl=legacy`) | **A2x = A2 + `tfla_impl=exact` is screened first** (V2), so the mLSTM layers are corrected too and the decode cache can serve the checkpoint (user decision 2026-09-17) |
| Decision 7: flip `scan_impl`/`tfla_impl` defaults at M9 | **Superseded.** Defaults stay `legacy`; every model yaml pins both explicitly (V1-E, parity-tested). A global flip could silently move a published number — the 15B-4 byte-identical re-decode is a property worth keeping |

**Hard constraints.** 200 GiB home quota (`SAVE_TOP_K=0/1` mandatory — 15B lost 3 of 4 arms to it).
The login node executes nothing scripted. Cluster invariants and the login-node rule are in *Verification*.

✅ **Cluster account extension APPROVED (user, 2026-09-19).** No account-expiry deadline governs this plan any
more. Schedule V3–V5 on the science, not the calendar: run the full ladder, take the retry budget, and do not
cut seeds or arms for time. The DUA archiving obligation in V4-E is unchanged (it follows from the DUA, not
from the calendar).

## Locked decisions (version 2)

| # | Decision |
|---|---|
| 1 | **Isolation**: all work on **`h100_mamba3_v2`**. ⚠ **No merge into `h100_scaling`, ever, until the user explicitly instructs one.** |
| 2 | **Scope**: as v1 — `Mamba3Block` (Mamba-2 SSD + trapezoid + RoPE + B/C biases + conv-drop, every flag reducing exactly to Mamba-2) + the O(1) decode cache. All shipped (M2–M6). |
| 3 | **MIMO**: plumbing only, never run (+3.2% params). |
| 4 | **Gate**: staged — the 12K screen decides the arm (done: A2; A2x re-screen pending), the full pipeline decides the claim on the official n=2663 test split **with 3 decoder seeds**, paired to the incumbents' seeds. |
| 5 | **Retrain**: Stage-0 for the winner only; **tower reused from 13D**; decoder × 3 seeds. |
| 6 | `d_state=128`, `headdim=64`, `ngroups=1`, `expand=2` (parameter-matched, +0.26%). |
| 7 | **`scan_impl` / `tfla_impl` defaults stay `legacy` for reproduction. Every yaml with a recurrent layer pins both explicitly** (`test_every_recurrent_model_yaml_pins_the_operator_explicitly`); V2-D decided A2x: `hybrid_150m_m3_rrg` pins `tfla_impl: exact`; `hybrid_150m_m3` stays `legacy` (it defines the M7 arms) and Stage-0 gets `exact` via `ARM=A2x`, with a parity test that the two agree. |
| 8 | `layer_pattern` stays 9 mamba3 + 3 mlstm through V3; the ratio is re-opened only as gated phase V5-C. |
| 9 | py3.9/willi retirement is **deferred to V4-B** (after V3 is submitted) — the harness is the gate mid-campaign and it passes under a real 3.9.23. |
| 10 | **Claim rule for V3** (mirrors 15C-4 as re-anchored): a Mamba-3 win over an incumbent on a metric is claimed only if the paired-by-seed mean exceeds **one baseline seed SD** (15B-4 table) **and** the sign holds at ≥2/3 seeds. Per-seed bootstrap CIs are reported alongside, never substituted. "Indistinguishable across 3 seeds" is the expected and reportable outcome. |
| 11 | **Eval path**: V3's headline numbers come from the incumbents' exact uncached `beam_search_decode`. The M6 cached path is token-identical by test and is used for latency curves only. |

## Baselines every arm is measured against

| Metric (official test split n=2663, beam=3, `prefix_k=32`, `last.ckpt`) | hybrid `hybrid_150m_v2` (seeds 42/43/44) | Transformer `transformer_150m_baseline` (42/43/44) | retrieval-NN floor |
|---|---|---|---|
| ROUGE-L | .1899 / .1991 / .1957 → **0.1949 ± 0.0047** | .1936 / .1976 / .1945 → 0.1952 ± 0.0021 | 0.1636 |
| BLEU-1 | 0.2508 ± 0.0034 | 0.2478 ± 0.0022 | 0.2372 |
| BLEU-4 | 0.0578 ± 0.0032 | 0.0575 ± 0.0016 | 0.0330 |
| CheXbert-14-micro | .4736 / .4380 / .4324 → **0.4480 ± 0.0223** | .4590 / .4455 / .4285 → 0.4443 ± 0.0153 | 0.4296 |
| CheXbert-14-macro | **0.2660 ± 0.0122** | 0.2692 ± 0.0106 | **0.3014** |
| CheXbert-5-micro | 0.5086 ± 0.0382 | 0.5032 ± 0.0227 | 0.4856 |
| CheXbert-5-macro | 0.4193 ± 0.0274 | 0.4170 ± 0.0165 | 0.4284 |
| exact-match-14 / -5 | 0.0380 ± 0.0058 / 0.2163 ± 0.0019 | 0.0455 ± 0.0027 / 0.2244 ± 0.0057 | — / 0.1735 |
| example-F1 | 0.3790 ± 0.0214 | 0.3817 ± 0.0144 | 0.3691 |
| **Stage-0 val PPL** (PubMed, 120K steps, same recipe) | **13.18** | **11.222** | — |
| Stage-0 val PPL at 12K (the screen) | A0 19.387 / 18.933 (s42 / s1234) | — | — |

Per-seed source: `H100_SCALING_PLAN.md` 15B-4 table; incumbents' per-seed dumps live under `results/` on the
cluster (seed 42: `results/report_gen_tower13d_test_split`, `results/report_gen_transformer_test_split`; 43/44:
`results/report_gen_{hybrid,transformer}_seed{43,44}_test_split`, from `h100_scaling_state.json` → `seed_arms`; all six are defaults in `submit_v3_chain.sh`). Pre-registered tiers unchanged:
ROUGE-L Floor 0.15 / Target 0.22 / Stretch 0.26; CheXbert-14-micro 0.25 / 0.40 / 0.50.

## The screen result the pipeline builds on (M7, carried; full record below)

| Arm | val PPL @12K | fit | note |
|---|---|---|---|
| A0 / A0-seed (Mamba-1 as shipped) | 19.387 / 18.933 | 2:36:50 | noise floor 0.454 PPL → bar **0.642** |
| **A1** (Mamba-1 + exact scan + dt init + no Δ-norm) | **16.294** | 4:11:13 | **−16.0%**: the defect fix alone |
| **A2** (Mamba3Block = Mamba-2 SSD, `d_state` 128) | **16.708** / 16.376 (s1234) | **1:20:55** | −13.8%, **1.94× faster**; A1−A2 not significant |
| A3 (+ trapezoid) | 16.719 | 1:28:00 | null (+0.011) |
| A4-hi (+ RoPE, `theta_max` 0.2) / s1234 | 16.199 / 18.912 | 1:31:23 | high-variance; rejected by the pre-registered rule |
| **A2x** (A2 + `tfla_impl=exact`, V2-C) | **15.566 / 15.788** | wall 5:45 / 5:57 | **−0.865 PPL vs A2 on the 2-seed mean; advanced (V2-D)** |

**Winner of M7: A2. Winner of V2: A2x**, which carries into V3.

⚠ **Correction to the M7 headline (2026-09-17).** M7 concluded *"the headline is the defect, not the architecture"* from
A1 − A2 = +0.414 PPL (not significant). That comparison was not like-for-like: **A1 had both recurrences corrected**
(`scan_impl=exact`, `tfla_impl=exact`; its fingerprint in the V2-A preflight says so) while **A2 still ran the defective
mLSTM** (`tfla_impl=legacy`). With both corrected on both sides, A2x s42 **15.566** vs A1 s42 **16.294** = **−0.728 PPL**,
1.13× the bar, with a single A1 seed. The revised reading: correcting the mLSTM is worth ~0.87 PPL on its own (A2 → A2x);
once both operators are correct, SSD + 8× state is ahead of corrected Mamba-1 by ~0.7 PPL — just above the bar, and
unreplicated on A1's side. The defect remains the largest single effect (A0 → A1 −3.09 PPL).

⚠ **Speed claims need re-checking from wall clocks.** The progress bar's elapsed column is not the training time:
A2x shows `1:52:35`, yet 36,000 micro-batches at the bar's own 1.75 it/s is 5.7 h, which matches the job's wall clock
(5:45:18). M7's "A2 1.94× faster than A0" compared that same column. Same-column *rates* are A0 1.36, A2 2.18 (300-step
probe), A2x 1.75 it/s: exact TFLA costs ~20% throughput vs A2, and A2x is ~1.3× A0. Settle it with
`sacct -j 2513005,2513632,2515666 --format=JobID,Elapsed,NodeList` before any speed number reaches the writeup.

---

## Carried from version 1 — Context (written 2026-09-06; every number measured then)

## Context

The user asked whether recent Mamba developments (Mamba-3) should change this architecture. The answer is
yes — but the headline is not the paper. It is a **correctness defect** found on the way in.

**Three things are true, all verified in-repo this session.**

### 1. The live recurrence is not the specified recurrence

`scan_interface.py:118-119` computes `A_cum · cumsum(Bx / A_cum.clamp(min=1e-8))`. Where `A_cum[s]` falls
below `1e-8` the clamp pins the denominator, and the token's own contribution to the state is
**annihilated** — not perturbed. Measured against a float64 sequential reference:

| Δ | 1e-3 | 1e-2 | **1e-1** | 0.3 | **0.705** | 1.0 |
|---|---|---|---|---|---|---|
| rel-max-err (chunk 64) | 2.7e-08 | 3.2e-08 | **3.9e-01** | 7.8e-01 | **9.2e-01** | 9.4e-01 |

Δ at init is **0.70** (`pre_rms`) / **0.82** (`hybrid`, canonical) with `|Δ·A|` up to **74.5**, because
`hybrid_lm.py:138-145` zeroes every bias including `dt_proj.bias` and there is no Mamba `dt` init at all.
The error appears at Δ=0.1 — the `dt_max` of the *correct* init — so a dt fix alone is necessary but **not
sufficient**. `dt_norm` (RMSNorm on Δ before softplus, `mamba_block.py:137-141`) additionally rescales Δ to
unit RMS, discarding any bias offset, so a `dt_proj` init would be a **no-op** on every v2 config.

**The defect appears twice more:**
- `mamba_block.py:_slow_forward` (~`:228`) contains the identical division. `validate_for_willi.sh:310`
  builds its Gate-6 model with `use_fast_path=False`, so **the harness exercises the buggy slow path**.
- `tfla_interface.py:93-95` divides by `f_cum.clamp(min=1e-6)` for the mLSTM intra-chunk term (`:149`, the
  inter-chunk half, is already log-space). Measured: at the shipped `forget_gate_bias_init=0.0`, **70.9% of
  `(t,d)` entries inside a 64-chunk hit the clamp** (36.8% at bias 1.0, 0% at bias ≥ 2.0). The clamp is the
  dominant regime, not an edge case. **Quantified in M1-C** against an fp64 oracle built from the block's
  exact per-dim gating/normalizer convention: at the shipped `forget_gate_bias_init=0.0` and the shipped
  `chunk_size=64`, **rel-max-err = 0.882**. The error tracks the clamp exactly — ~1e-7 wherever
  `min(f_cum)` stays above `1e-6`, 0.48–0.96 wherever it falls below:

  | forget bias | chunk 8 | chunk 32 | chunk 64 | chunk 128 |
  |---|---|---|---|---|
  | **0.0 (shipped)** | 1.1e-07 | 5.0e-01 | **8.8e-01** | 8.8e-01 |
  | 1.0 | 1.2e-07 | 1.7e-07 | 5.7e-01 | 9.6e-01 |
  | 2.0 | 6.5e-08 | 1.4e-07 | 1.8e-07 | 4.8e-01 |
  | 3.0 | 9.1e-08 | 9.1e-08 | 1.3e-07 | 1.4e-07 |

⇒ **All 12 layers of the canonical 150M model run a recurrence that is not the one specified.**

This does **not** invalidate the published numbers — training and eval used the same operator, so
ROUGE-L 0.1899 and MIMIC 0.1459 are valid measurements of the system as built. The narrower true statement,
which must survive into the writeup, is: *the block did not compute the recurrence it was specified to
compute.* Whether the correct one is better is what M7 measures. It is not assumed.

### 2. The exact fix is cheap in Mamba-2/3's parameterization and expensive in Mamba-1's

Log-space segsum (`out[t,s] = exp(Σ_{i=s+1..t} dA_i)`, no division anywhere) is exact — **~1e-16 at every Δ
tested** — in both forms. The cost is not:

| Form | Mask memory @ bs=48, L=1024 | Sequential steps | Shape |
|---|---|---|---|
| Mamba-1, `A=(d_inner,N)=(1536,16)`, cs=64 | **19.3 GB** ❌ | 16 | elementwise |
| Mamba-1, cs=8 | 302 MB | **128** | elementwise |
| **Mamba-2/3, scalar-`A`/head, cs=64** | **19 MB** ✅ | 16 | **matmul (tensor cores)** |

Migrating to SSD is therefore not paper-chasing — **it is what makes the correct operator affordable.**
That argument stands independently of any reported quality gain.

Activation memory follows the same pattern. At the Stage-0 shape (bs=16, L=512, d_inner=1536) the current
Mamba-1 chunk scan materializes six `(B, nc, cs, D, N)` tensors of 201.3M elements (~805 MB each ≈ 4.8 GB)
plus a comparable amount saved for backward — ~9.7 GB **per mamba layer**, which is exactly why
`GRAD_CKPT=true` is mandatory today. SSD at `d_state=128` materializes ~260 MB per layer: a **~37× reduction**.

### 3. Mamba-3 costs ~nothing in parameters and buys 8× the state

Computed and then verified by building the real model (`183,721,824` params, 9 mamba + 3 mlstm,
per-mixer `3,772,448`):

| Block (dim=768, expand=2) | params | Δ/block | 150M total | band `[181,186]M` |
|---|---|---|---|---|
| Mamba-1, `d_state=16` (current) | 3,772,448 | — | 183.72M | ✅ |
| **Mamba-3 SISO, `d_state=128`, `hd=64`, `ngroups=1`, no conv** | **3,821,872** | **+49,424** | **184.17M (+0.24%)** | ✅ |
| Mamba-3, `d_state=64` | 3,708,080 | −64,368 | 183.14M | ✅ |
| Mamba-3, `ngroups=2` | 4,043,056 | +270,608 | 186.16M | ❌ out by 157k |
| Mamba-3, `d_state=256` | 4,049,456 | +277,008 | 186.21M | ❌ out by 215k |
| Mamba-3, `mimo_rank=4` | 4,430,128 | +657,680 | 189.64M | ❌ **out by 3.6M** |

Dropping `x_proj` (122,880) + `dt_proj` (75,264) + `A_log` (24,576) = 222,720 nearly cancels the 276,480
added to `in_proj` for the 8× wider B/C. Because B/C are shared across heads (multi-value attention), the
**SSM state grows 8×** (1536×16 = 24,576 → 24×64×128 = 196,608 elements/layer) at **+0.24% parameters**.
That reallocation is itself a reportable result.

### The honest counter-argument, and why it does not block

This project has 10 clean nulls, every one text-side: Stage-0 PPL 15.62→13.18 *flat* on retrieval;
70M→150M *flat*; the closing verdict says "no architecture change was needed." But **every null was measured
against retrieval**, where the text tower is an *encoder* dragged toward a frozen teacher by KD — and all of
them were measured with a broken operator. The objective is now *generation*, where the Mamba/mLSTM stack
**is** the generator. No decoder-architecture arm has ever been run against ROUGE-L / CheXbert.

That is an argument from absence, not evidence, and the mechanism partly cuts the other way: Stage-0 trains
on PubMed abstracts (~250 tokens) at `max_length=512` **with doc-boundary resets**, and trapezoid/RoPE are
argued to help long-range state tracking. At 250-token effective context the PPL delta may sit under the
seed-noise floor. **M7-A measures that noise floor before anything is ranked** (see FM6).

### Intended outcome

A correct, cacheable, parameter-matched Mamba-3 mixer available as a new layer type on an isolated branch;
an attributable ablation ladder screened by Stage-0 PPL and decided by report generation on the official
subject-disjoint test split; and a publishable finding either way — "the corrected operator did not help" is
as reportable as "it did."

---

---

## Design

### Prototype verification (run on CPU this session, fp64, before any code is written)

The full M2+M3+M4 math was prototyped and checked against a sequential float64 reference. **All of it is
already verified — the implementation phases are engineering, not research:**

| Configuration | rel-max-err vs sequential fp64 |
|---|---|
| Chunked SSD, log-space segsum, scalar-`A`/head | 4.4e-16 |
| + complex/RoPE (rotate B, C by `cumsum(Δ·θ)` outside the scan) | 4.5e-16 |
| + exponential-trapezoidal (two passes, shared decay mask) | 5.3e-16 |
| + trapezoid **and** RoPE together (rotate-then-shift) | 5.3e-16 |
| **Bit-identity control: `λ ≡ 1` (trap on) vs trap off** | **0.00e+00 exactly** |

### New files

| Path | Contents |
|---|---|
| `MAMBA3_PLAN.md`, `mamba3_state.json` | Plan-of-record + resumable state, repo root (M0) |
| `hybrid_xmamba/kernels/ssd/ssd_interface.py` | Chunked SSD scan: log-space `segsum`, **no division**, native `cu_seqlens` masking, fp32 policy |
| `hybrid_xmamba/kernels/ssd/ssd_reference.py` | float64 sequential oracle **and** the single-token `step()` used by the decode cache |
| `hybrid_xmamba/layers/mamba3_block.py` | `Mamba3Block` |
| `hybrid_xmamba/layers/rotary.py` | Data-dependent RoPE: fp64 cumulative angle, `remainder(·, 2π)`, `apply_rotary` |
| `hybrid_xmamba/training/spike_guard.py` | Skip-step-on-spike callback (FM2) |
| `configs/model/hybrid_150m_m3.yaml`, `hybrid_150m_m3_rrg.yaml` | Arch + report-gen configs |
| `scripts/mamba3_arms.py` | **The one definition of the A0–A6 ladder** — config, overrides, seed, expected ARCH tokens; read by the pre-flight and by every submission (M5) |
| `analysis/mamba3_results.md` | Ablation table + writeup |

### The three mechanisms

**1. Exponential-trapezoidal (Prop. 1).** `h_t = α_t h_{t-1} + β_t B_{t-1}x_{t-1} + γ_t B_t x_t`,
`α_t = exp(Δ_t A_t)`, `β_t = (1−λ_t)Δ_t α_t`, `γ_t = λ_t Δ_t`, `λ_t = sigmoid(trap_proj(x))` per head.
The recurrence is **linear in the input**, so this is two SSD passes sharing one decay mask:

```
Y_intra = (L ⊙ C·Bᵀ)·(γ⊙X)  +  (L ⊙ C·shift(B)ᵀ)·(β⊙shift(X))
```

with the matching second term in the inter-chunk state update. ~2× intra-chunk matmul; mask unchanged.
Needs a 1-step `(B, x)` carry across chunk boundaries, and **`β` forced to 0 at each document start** or the
previous document's last token leaks in.

**2. Complex state via the RoPE trick (Prop. 3/4).** Rotate `B_t` and `C_t` by `Θ_t = cumsum(Δ_t·θ_t)`
**outside** the scan — the SSD kernel is untouched. `C_tᵀB_s` then carries `R(Θ_s − Θ_t)`, a relative
rotation. **Rotate first, then shift**, so the `β` term's `B_{t-1}` carries its own `Θ_{t-1}` (Prop. 4).
`rope_fraction=0.5` per the reference — half of `d_state` rotated, half pure real decay.
Parameterize `θ = θ_max · tanh(angle_proj)` with near-zero init so RoPE starts ≈identity (clean warm start,
and it makes rope-on/rope-off comparable at step 0).

**3. MIMO — plumbing only, never run.** `mimo_rank` defaults to 1; `R=1` asserted bit-identical. Recorded so
it is not silently re-proposed: rank-`R` B/C/x with learnable per-head `mimo_x/z/o` scaling vectors,
`chunk_size = C/R`. Breaks parameter matching by +3.2%; its motivation is decode arithmetic intensity, which
is unmeasurable here.

### Architectural refinements (§3.4)

- **`dt_bias` init** — `_dt ~ exp(U(log 1e-3, log 1e-1))`, `dt_bias = _dt + log(−expm1(−_dt))`, re-applied
  **after** `self.apply(self._init_weights)` or it is zeroed (`hybrid_lm.py:138-145`).
- **No Δ-norm.** Mamba-3 normalizes B and C only. This is what unblocks the `dt_bias` init.
  ⚠ **Documented asymmetry:** `norm_topology="hybrid"` means different things per type — for `mamba` it
  enables `dt_norm`+`B_norm`+`C_norm`; for `mamba3` it enables `B_norm`/`C_norm` only.
- **BCNorm is already in the baseline** (`mamba_block.py:139-140` under `norm_topology=hybrid`, which the
  canonical config uses). It is **not** a new degree of freedom and is **not** an arm. Say so in the writeup.
- **B/C biases** — `(nheads, d_state)`, three-valued flag `bc_bias ∈ {none, zero_init, one_init}`;
  `one_init` (1.0) is the reference and is a **genuine architectural arm**, not a no-op.
- **Data-dependent `A`** — `A = −heavy_tail(dd_A)` clamped `≤ −A_floor` (`heavy_tail(x)=1+x if x≥0 else 1/(1−x)`).
- **`dt_limit`** — clamp Δ to `(0.0, 1.0)` by default (FM2: `dt_norm` was an accidental stabilizer).
- **Short conv droppable** — `use_conv`, default `True` so M2 is a clean Mamba-2; dropped as an arm in M5
  (Table 5a: `Mamba-3 + conv` 15.85 is *worse* than without, 15.72).

### `cu_seqlens` — fix a live bug, don't inherit a slow path

`hybrid_block.py:140-143` gates on `("mamba","mlstm")`, so **sLSTM blocks silently leak state across packed
documents today**. Fix with a **class attribute contract + signature drift guard**:
`supports_cu_seqlens: bool` on each mixer, resolved once in `HybridBlock.__init__`; a parity test
cross-checks it against `inspect.signature(mixer.forward)` for every `LayerType`; `warnings.warn` once (do
not raise — sLSTM configs exist) when a mixer declines a non-`None` `cu_seqlens`.

Do **not** copy `mamba_block.py:248-276`'s Python loop over (row, segment) into `Mamba3Block` — it is dozens
of tiny kernel launches per layer per step and is why `compile_model=false`. **SSD handles boundaries
natively**: zero the decay inside the segsum mask at each document's first position. One masked-`exp`, fully
batched, and a real speedup on the packed Stage-0 path.

### Checkpoint architecture recovery (metadata first, hardened sniffer as fallback)

Neither alone works — metadata cannot help ~dozen existing checkpoints, and no sniffer can recover
`d_state`/`headdim`/flag settings that have no distinct key name.

- **Write**: override `on_save_checkpoint` in the base `HybridLMModule` (`lightning_module.py`, next to
  `on_load_checkpoint` `:1126`) → `checkpoint["hybrid_arch"] = {"schema": 1, "config": asdict(cfg)}`.
  Reads the config off the **live model**, so no entry point can drop it. Gotchas: unwrap
  `getattr(model, "_orig_mod", model)`; walk wrappers (`HybridTextEncoder`, report-gen decoder) via one
  `_arch_config()` helper that returns `None` rather than raising; the subclass overrides at
  `train_stage0_distill.py:106` and `train_stage0_distill_resume.py:89` **must call `super()`** — pin with a test.
- **Read**: `HybridConfig.from_checkpoint(ckpt)` classmethod — metadata if present, else the shared sniffer,
  using the safe `dataclasses.fields()` filter idiom (`performance_profile.py:107-111`).
- **Harden the sniffer**: ordered `(type, predicate)` list, assert **exactly one** matches, raise on 0 or ≥2.
  Fingerprints: `mamba` → `mixer.dt_proj.weight`; `mamba3` → `mixer.dt_bias` (1-D) or `mixer.B_bias` (3-D);
  `mlstm` → `mixer.i_gate_proj.bias`. The current `"A_log" in k or "conv1d" in k` fails on Mamba-3 twice over
  (conv dropped, `A` data-dependent). **Derive shapes**: `evaluate_sts.py:105-110` hardcodes
  `state_size=16, conv_size=4, expand_factor=2` — derive `state_size` from `A_log.shape[1]` / `B_bias.shape[-1]`
  and `headdim` from `d_inner / dt_bias.numel()`.
- **Harden the loader**: `evaluate_cxr_retrieval.py:140-145` *prints* on missing keys and returns anyway,
  while `evaluate_sts.py:115-117` raises. Make retrieval match STS — a standalone one-line fix.
- Old checkpoints stay loadable: `MambaBlock` is byte-identical, the fallback sniffer is unchanged for
  mamba/mlstm, and `hybrid_arch` is additive (Lightning ignores unknown top-level keys).

### Bit-identity controls

**Bit-identity is achievable only with shared weights, never a shared seed** — any flag that adds/removes an
`in_proj` output slice changes `in_proj.weight.shape` and therefore the RNG draw sequence in `_init_weights`.
Each assertion: build flag-on, project its state_dict onto the flag-off twin, same input, `torch.equal`.

| Flag | Assertion | Achievable |
|---|---|---|
| trapezoid `λ≡1` | zero the `trap` slice of `in_proj.weight`, `trap_bias = +20` (`sigmoid(20)==1.0` exactly in fp32) ⇒ `β=0` | **Yes** — `a + 0.0 == a` in IEEE754; holds at doc boundaries too |
| rope off | zero the `angles` slice ⇒ `cos(0)==1.0`, `sin(0)==0.0` exactly | **Yes** |
| `bc_bias` | `none ≡ zero_init` bitwise | **Yes** for zero-init; **No** for `one_init` — that is a real arm |
| `use_conv` | assert only that graph/shapes are unchanged and `n_params` moves by exactly `9 × 8,960` | **No** (SiLU placement differs) |
| BCNorm | — | **No, and already on in the baseline** — not an arm |
| `mimo_rank=1` | bit-identical to no-MIMO | **Yes** |
| **SSD kernel** | vs float64 sequential oracle, rel-max-err < 1e-12 at Δ ∈ {0.01, 0.1, 0.705, 1.0, 5.0} | **Yes — the assertion that matters most** |

---

## Ablation ladder (OFAT, not factorial — 2⁴ = 16 combinations is unaffordable)

| Arm | Description | Isolates |
|---|---|---|
| **A0** | Mamba-1, as-shipped | control; must reproduce the Phase-5 curve or the harness is wrong |
| **A0-seed** | A0, different seed | **the noise floor** — run before ranking anything |
| **A1** | Mamba-1, exact scan + Δ init + no Δ-norm | **the defect/Δ fix alone** — screen-only, never enters the full pipeline |
| **A2** | `Mamba3Block`, exact Mamba-2 reduction (`λ=1`, no rope, no bias, conv on), `d_state=128` | SSD + 8× state (a **bundle**) |
| **A3** | A2 + exponential-trapezoidal | Prop. 1 |
| **A4** | A2 + complex/RoPE | §3.2 |
| **A5** | A2 + trapezoid + RoPE | **Mamba-3 SISO** |
| **A6** | A5 + `bc_bias=one_init`, conv dropped | §3.4 refinements |

**A1 is the load-bearing control.** Without it, A2 differs from A0 in Δ regime *and* operator *and* scan
correctness simultaneously, and a Mamba-3 win is uninterpretable. If A1 captures most of the gain, the honest
headline is "the Δ init was broken" — still a real, publishable finding.

**Pre-registered:** `ngroups>1`, `d_state≥256` and `mimo_rank≥4` exit the parameter-matched regime and may
only be reported as *scaled* arms, never as the headline comparison.

---


---

## Phases

Every phase ends with the validation harness exiting 0 and a commit on `h100_scaling_mamba3`.

### M0 — Branch, plan-of-record, bootstrap (**no model code**)
- [x] **M0-A** `git checkout -b h100_scaling_mamba3` from `h100_scaling`. Nothing on `h100_scaling` changes.
- [x] **M0-B** Write `MAMBA3_PLAN.md` (this document) and `mamba3_state.json` at repo root.
      State schema: `current_phase`, `last_updated`, `status`, `branch`, `baselines`, `arms`,
      `phases{id → {status, checkboxes, evidence, verdict}}`, `notes[]`, `open_questions`, `decisions`.
- [x] **M0-C** `.gitignore`: allowlist both (mirror the `!h100_scaling_state.json` pattern).
- [x] **M0-D** `CLAUDE.md` Session Bootstrap → read `MAMBA3_PLAN.md` + `mamba3_state.json` **first**;
      `H100_SCALING_PLAN.md` demoted to closed-historical baseline. Record the resumability contract:
      after every meaningful change, tick the checkbox **and** update `last_updated` + append a `notes` line.
- [x] **M0-E** Retire `MAMBA3_INTEGRATION_PLAN.md` + `mamba3_integration_state.json` (its audit is folded in here).
- [x] **M0-F** Commit.

### M1 — Pin the defect, then fix it on the legacy path (produces arm A1)
Test-first: M1-A/B/C must **fail on HEAD**.
- [x] **M1-A** `test_selective_scan_matches_sequential_reference` — float64 oracle, Δ ∈ {1e-3, 1e-2, 1e-1,
      0.3, 0.705, 1.0} × chunk ∈ {8, 64} × **both `use_fast_path` values** (the slow path has the same bug).
      **CPU-collected.** Fix the marker gap first: `@pytest.mark.cuda` is never applied
      (`test_kernels.py:9-15` uses a local `skipif`), so `-m "not cuda"` deselects nothing.
- [x] **M1-B** `test_delta_at_init_is_in_mamba_range` — Δ mean ∈ [1e-3, 1.5e-1] for `pre_rms` and `hybrid`.
- [x] **M1-C** TFLA intra-chunk test; **quantify the mLSTM output error** (open from Context §1).
- [x] **M1-D** Mark all three `xfail(strict=True)` citing this plan, so CI is green on HEAD and **flips loudly**.
- [x] **M1-E** `scan_impl: {"legacy","exact"}`, default `legacy`. For the Mamba-1 form do **not** use the 4-D
      log-segsum (19.3 GB). Instead **flip the parallel axis**: per-chunk states from zero-init in parallel,
      then an `L/cs` sequential combine. Depth `cs + L/cs`, memory unchanged, no mask, no division, exact.
      A1 is screen-only, so a 3-5× slower scan is acceptable — do not optimize it.
- [x] **M1-F** `dt_init_strategy: {none, mamba}` (default `none`) + `norm_topology: hybrid_bc` (B/C norm
      **without** Δ-norm). `hybrid` untouched — every existing checkpoint must load unchanged.
- [x] **M1-G** Fix `_slow_forward` too, or collapse both into one function.
- [x] **M1-H** TFLA intra-chunk log-space fix (`tfla_interface.py:93-95` only; do **not** touch `:149`).
- [x] **M1-I** Flip all three `xfail`s. **Gate: rel-err ≤ 1e-6 at every Δ**, and `legacy` bit-identical to today.

### M2 — `Mamba3Block` = exactly Mamba-2 SSD (+ the sequential oracle)
- [x] **M2-A** `ssd_reference.py`: float64 sequential oracle **and** `step(x_t, state) → y_t, state`.
      Writing the oracle is mandatory for testing anyway, and it *is* the decode step — free de-risking.
- [x] **M2-B** `ssd_interface.ssd_chunked_scan()` — chunked, scalar-`A`/head, log-space mask, matmul-shaped,
      native `cu_seqlens` decay masking. Carry the fp32 policy explicitly (see FM3).
- [x] **M2-C** `Mamba3Block`: `in_proj → [z, x, B, C, dt, A, trap, angles]`, BCNorm, optional conv, SSD scan,
      gate `* silu(z)`, `out_proj`, `D` per head. Contract `forward(x, cache=None, cu_seqlens=None) -> Tensor`,
      **positional order load-bearing** (`hybrid_lm.py:196-199` checkpoints positionally).
- [x] **M2-D** Register the type: `configuration_hybrid.py:51-53` Literal + `:145-151` `valid_types`;
      `hybrid_block.py:17` `LayerType`, `:71-100` dispatch.
- [x] **M2-E** `supports_cu_seqlens` contract + signature drift guard + sLSTM leak warning.
      Extend `TestDocBoundaryReset` to `mamba3`.
- [x] **M2-F** New `HybridConfig` fields, **all defaulting to Mamba-2 behaviour**. Replace the 18-named-kwargs
      call at `hybrid_lm.py:97-120` with a `dataclasses.fields`-filtered pass-through — this kills the
      silent-drop bug *class*, not an instance. Make `hybrid_block.py`'s whitelist **raise** on unknown kwargs
      carrying a recognized prefix (`mamba3_`, `mlstm_`, `slstm_`) instead of silently dropping them.
- [x] **M2-G** Tests: `"mamba3"` in `test_layers.py:129`; update `test_models.py:114`; **structural equality**
      test (embeddings/lm_head/MLPs/mLSTM mixers/norms exactly equal to the control, and
      `total_m3 − total_ctrl == 9 × (mixer_m3 − mixer_m1)`) — far stronger than a band; plus a ±2% band for
      m3 yamls, predicted **184,166,640**. **Leave `test_150m_v2_param_count` byte-identical** — it guards the control.
- [x] **M2-H** Extend Gate 6 to the 4-type pattern `["mamba","mamba3","mlstm","slstm"]` at `dim=64`
      (`d_inner=128`, `nheads=2`, `use_fast_path=False`) — it currently only exercises `["mamba","mamba","mlstm"]`,
      so a dangling mamba3 parameter would slip past "every parameter receives a gradient".
- [x] **M2-I** **Arch fingerprint** logged at `HybridLanguageModel.__init__` (layer_pattern, mixer classes,
      mamba3 flag tuple, total params) into `utils/run_metadata.py` output — eyeballable at step 0 of a 3-day job.
- [x] **M2-J** 20-sample beam-decode timing probe. **Tripwire:** if > 1.5× the mamba1 time, promote M6 to M3.

### M3 — Exponential-trapezoidal
- [x] **M3-A** `trap` head, `λ = sigmoid(trap)`, two SSD passes sharing one mask, 1-step `(B,x)` chunk carry,
      **`β = 0` at document starts**.
- [x] **M3-B** **Bit-identity: `λ ≡ 1` reproduces M2 exactly (`torch.equal`)** via the zeroed-slice /
      `trap_bias=+20` recipe. Without this the arm is uninterpretable.
- [x] **M3-C** vs the fp64 3-term oracle, rel-err ≤ 1e-6.

### M4 — Complex-valued state (RoPE trick)
- [x] **M4-A** `rotary.py`: `Θ = cumsum(Δ·θ)` **in float64** (~2 MB, negligible), `remainder(·, 2π)` before
      fp32 `sin/cos`, **Θ reset per document segment**, `θ = θ_max·tanh(angle_proj)`, `rope_fraction=0.5`.
- [x] **M4-B** **Bit-identity: rope off reproduces M3 exactly.**
- [x] **M4-C** Rotate-then-shift ordering test (Prop. 4).
- [x] **M4-D** **Capability test — the paper's headline claim.** ✅ **Reproduced.** Controlled parity
      experiment, two seeds, identical but for `use_rope`: **rope-off 61.6% / 64.3%, rope-on 100.0% /
      100.0%** (chance 50%). The cleanest standalone contribution in the plan — a capability Mamba-2
      does not have, demonstrated rather than cited.

      ⚠ **Caveat that matters for M7.** The rotation angle is `Δ_t·θ_t`, so a π turn per token needs
      `Δ·θ ≈ π`. Under the reference dt init (`Δ ~ logU[1e-3,1e-1]`) the reachable angle tops out near
      **0.06 rad — fifty times too small**, and parity stays at 57–63% for *every* `theta_max` tried
      {3.2, 32, 320}; raising `theta_max` alone made it **worse** (320 → 57%), because a large θ on a
      tiny Δ is noise, not a half turn. Only with Δ free to reach ~1 does it solve.
      **Consequence: on PubMed with the standard dt init this capability is largely dormant unless Δ
      learns to grow. A null on LM perplexity would therefore NOT be evidence that complex transitions
      do not work — only that the operating point never entered the regime where they can.** Say this
      in the writeup whichever way M7 lands.
- [x] **M4-E** Angle-drift test: fp32 path vs fp64 sequential rotation < 1e-6 at L=1024;
      alarm if `Θ.abs().max() > 1e3` rad.

### M5 — Flags folded into arm definitions (no milestone of its own)
- [x] **M5-A** `bc_bias ∈ {none, zero_init, one_init}`; assert `none ≡ zero_init` bitwise.
- [x] **M5-B** `use_conv=False` path; `n_params` moves by exactly `9 × 8,960`.
- [x] **M5-C** `mimo_rank` plumbing, default 1, asserted bit-identical. **Never run.**

**Measured.** `none ≡ zero_init` is bit-identical on both paths and with documents — worth stating,
because turning the bias on also changes B and C from group-indexed `(b,l,1,n)` to per-head, so two
different tensor shapes reach the same einsums and "adding zero changes nothing" is a claim about the
kernel's contraction order, not only about IEEE754. `one_init` moves the output, as the table
predicted: A6 owns a real capability difference. Conv drop: exactly **80,640 = 9 × 8,960**.

**The gap M5 actually closed.** A3–A6 exist only as `model.mamba3_*=...` overrides on
`hybrid_150m_m3.yaml`, and `train_stage0_h100.sh` had **no way to pass an extra Hydra argument** —
four of the eight arms were unsubmittable. Fixed by `EXTRA_OVERRIDES` plus **one definition of the
ladder** in `scripts/mamba3_arms.py` (config, overrides, seed, and the ARCH tokens each arm must
log). The pre-flight now verifies that module at full 150M scale instead of carrying its own copy of
three arms, and submission reads the same module:

```bash
eval "$(python scripts/mamba3_arms.py env A5)" && sbatch --time=12:00:00 scripts/train_stage0_150m_h100.sh
```

That shared definition is the FM5 defence. The first pre-flight had its own arm list and filtered
dataclass fields inline — a *different* code path from the trainer's — so it passed on 2026-09-06
while `train_stage0_distill.py` was still dropping three fields, and job 2513007 trained A1 with its
defects intact.

**Ladder parameter counts, measured at 150M:**

| A0 / A0-seed | A1 | A2 | A3 | A4 | A5 | A6 |
|---|---|---|---|---|---|---|
| 183,721,824 | 183,708,000 | 184,192,200 | 184,192,416 | 184,192,200 | 184,192,416 | 184,167,072 |

A2→A6 spans **0.014%**; the whole ladder spans 0.26% of the control. A6's two refinements pull in
opposite directions — the B/C biases add `2 × 24 × 128` per layer, the conv removes 8,960 — leaving it
25,128 parameters *below* A2.

### M6 — O(1) recurrent decode cache
Measured on a tiny CPU model, per-token cost rises monotonically (0.0135 → 0.0173 s/tok from 16 → 128 new
tokens; doubling ratios 1.9× → 2.4×): **confirmed super-linear, trending quadratic**, paid 3× under beam=3.
Only pays off if **every** layer is cacheable — TFLA already carries the `m_state` LSE stabilizer across
chunk boundaries (`tfla_interface.py:110-117`), so an exact mLSTM `step()` is derivable.
- [x] **M6-A** `Mamba3Block.step()` + `allocate_inference_cache()` (reuse M2-A's oracle). State:
      `h (nheads, headdim, d_state)` + `angle_state` + `B_prev`/`x_prev` ≈ **7.1 MB fp32** for the whole model at bs=1.
- [x] **M6-B** `mLSTMBlock.step()` + cache (`C`, `n`, `m`).
- [x] **M6-C** Cache plumbing through `HybridBlock.forward`, `generate()`, `beam_search_decode` — including
      the `prefix_embeds` branch.
- [x] **M6-D** **Equivalence: cached decode == full recompute, `atol ≤ 1e-5`**, greedy and beam=3, with and
      without an image prefix.
- [x] **M6-E** Add prefill / per-token decode / TTFT to `performance_profile.py` — the repo has **no**
      decode-latency benchmark (`evaluate_lm.py:170-194` and `performance_profile.py` time full-sequence
      forwards only). Report O(L²) → O(L).

**M6 RESULT (2026-09-07).** The cache is an *equivalence*, and every test says so:

| Check | Result |
|---|---|
| `Mamba3Block.step` vs the chunked forward, 7 flag combinations | ≤ **5e-7** |
| `mLSTMBlock.step` vs `apply_tfla` (`tfla_impl=exact`) | **1.2e-10** |
| Cached beam=3 vs the existing `beam_search_decode`, ± image prefix | **token-identical** |
| Per-token decode (tiny CPU model, prompt 32, 224 new) | **21.5× faster** |
| Growth, first half → second | recompute **1.19×**, cached **0.99×** |

That last row *is* the O(L²) → O(L) claim: the recomputing path gets slower as context grows and
the cached one does not.

⚠ **Finding 1 — the cache cannot serve a legacy-TFLA checkpoint, and A2 is one.** `tfla_impl=
"legacy"` divides by a clamped forget-gate cumulative product and so computes *no* recurrence;
measured, `step` vs legacy `apply_tfla` is **rel 1.01** — the legacy output is noise relative to a
correct one. No O(1) step can reproduce that, by construction. This is a second, *functional*
argument for the M9-A flip: cacheable decode, not merely correctness.

⚠ **Finding 2 — `mLSTMBlock._slow_forward` and `apply_tfla` are different functions.**
`_slow_forward` carries the LSE stabilizer `m` into `C`/`n` and divides by `max(|n·q|, 1)`;
`apply_tfla` computes an `m_state` and never applies it, and clamps the **signed** denominator.
Gap **0.42 max abs** at L=24, identical for `legacy` and `exact`, so structural rather than the M1
clamp defect. `sequential_mlstm_fp64` (the M1 oracle) already documents TFLA's convention as the
reference one, and `use_tfla=True` is what every checkpoint trained on — so the cache matches TFLA
and `_slow_forward` is the outlier. Same class as the Mamba-1 `_slow_forward` divergence M1 found.

⚠ **Limitation, deliberate.** `prefill` steps token by token, so **TTFT is 1.7–8.6× slower** than
the uncached path (prompt-length dependent) while every token after it is ~21× faster.
`ssd_chunked_scan` already computes the carried state in its inter-chunk loop but does not return
it, and padding is masked out of that state, so exposing it correctly is its own change — recorded
as follow-up rather than bolted on under time pressure.

### M7 — Timing probe + short-run screen (**the cheap decision gate**) — H100
- [x] **M7-A** ⚠ **Do this before anything else costs money.** (i) Run **A0 twice with different seeds** at
      screen length; measure |ΔPPL| = the noise floor. (ii) 200-step timing probe **with and without the
      BioMedLM teacher** (2 × 15 min) — the 2.7B teacher is ~10:1 of per-step FLOPs and contributes nothing
      to ranking. If seed noise ≥ the expected 1-3% effect, **the screen cannot rank arms** — change the
      metric before spending 50 GPU-h.
      ⚠ **(ii) is now informational only.** A0, A0-seed and A1 were early-started under M7-A2 *with* the
      teacher, so dropping it for A2–A6 would break the paired comparison the screen depends on. The early
      start bought ~22 GPU-h of overlapped queue time and spent the teacher-off saving; that trade is
      already made. The probe still has value for a future 70M screen, not for this one.
- [x] **M7-A2** **Early-start the Mamba-1 arms.** A0, A0-seed and A1 need only M1's flags, so they can
      queue while M2–M6 are still being built locally — ~22 GPU-h of queue time overlapped with
      development, and it validates the screen harness before the expensive arms exist.
- [x] **M7-B0** `scripts/screen_arms_h100.sh` as a **SLURM job array** (`--array=0-7`), arm selected by
      `$SLURM_ARRAY_TASK_ID` from a table in the script. One submission instead of eight; each task takes a
      GPU as one frees. Assert the arm table in `tests/test_willi_parity.py` the way the other SLURM
      wrappers are asserted.
- [x] **M7-B** Screen A0, A0-seed, A1, A2, A3, A4, A5, A6 — **12,000 steps**, 150M, `aisc-batch`,
      **identical seed and data order** (paired comparison on Δ log-loss, not independent PPL).
      Set `trainer.max_steps` to the screen length so **WSD reshapes its own decay** — a run stopped at 12K of
      a 120K schedule never enters decay, and decay is where models separate. Warmup 500.
      **Hold `GRAD_CKPT=true` fixed across all arms** even though SSD makes it unnecessary (see FM3).
      If M7-A shows the teacher dominates, screen at 70M first (~3× cheaper) then confirm the top 2 at 150M.
      Every arm is submitted from `scripts/mamba3_arms.py` (M5) so seed, step count and levers cannot
      drift between arms: `eval "$(python scripts/mamba3_arms.py env A5)" && sbatch ...`. A0-seed is the
      one arm allowed to differ in seed — that is what it measures.
**M7-A RESULT (2026-09-06, jobs 2513005 / 2513006) — the screen is underpowered, as FM6 warned.**

| A0 seed 42 | A0 seed 1234 | \|Δ\| | Δ log-loss |
|---|---|---|---|
| val PPL **19.387** (loss 2.887) | val PPL **18.933** (loss 2.864) | **0.454 PPL = 2.37%** | 0.023 nats |

The plan's own trigger was "if seed noise ≥ the expected 1-3% effect, the screen cannot rank arms."
It is 2.37%. With n=2 the SD estimate is 0.321 PPL, so the pre-registered *2× SD* bar an arm must
clear is **0.642 PPL (3.35%)**. Consequences, decided before any arm's number exists:

1. **The A2-vs-A0 headline is the weak comparison** and stays weak — the two differ in operator *and*
   in RNG stream, so nothing pairs. |Δ| < 0.45 PPL is a **null on the bundle**, reportable as such.
2. **The per-lever deltas are the strong ones.** A2–A6 share `in_proj`'s shape, so at a fixed seed they
   start from *identical* embeddings, MLPs, norms and projections — only `trap_bias`/`B_bias` differ —
   and they see the same data in the same order. A3−A2, A4−A2, A5−A2 and A6−A5 are therefore near-
   perfectly paired and are the numbers that can actually attribute a mechanism. Report these, not a
   league table of eight absolute PPLs.
3. **The winner gets a second seed before the full pipeline** (~8 GPU-h) — cheap next to committing
   133 GPU-h to a 0.3 PPL gap that a seed could have produced.
4. **M7-E's mechanism diagnostics carry more weight than PPL here**, not less: MQAR and the
   late-position slice probe the claim directly and are not bounded by this floor.

**A2 GPU probe (2026-09-06, job 2513598) — the operator runs, and it is faster.**

`Mamba3Block` had never executed on a GPU before this; every prior check was CPU and fp32. 300
steps, 7m14s wall. Fingerprint exactly A2 (`mamba3x9, d_state=128, conv=True, trapezoid=False,
rope=False, bc_bias=none, params=184,192,200`); loss finite and descending (ce 4.517, ppl 91.6 at
step 300); no NaN, no spike. bf16 autocast, gradient checkpointing and the SSD scan compose fine.

| | optimiser steps/s | s/step |
|---|---|---|
| A0 (Mamba-1, legacy scan) | 1.36 | 0.78 elapsed / **0.66** train-only |
| **A2 (Mamba-2 SSD, `d_state=128`)** | **2.18** | **0.38** |

**~1.76× faster**, and the direction is not in doubt — SSD replaces the 4-D pairwise-decay tensor
with a `(chunk, chunk)` matmul per head. The magnitude is provisional: A2 ran 300 steps with no
validation pass while A0's elapsed includes six, and A2 amortizes start-up over fewer steps.
Confirm on equal footing from the screen, where every arm shares one validation schedule. Note
also that the 2.6B teacher is a fixed cost in **both**, so the student-side speedup is larger than
this end-to-end figure — 8× the SSM state, at 0.24% more parameters, for less wall-clock.

Measured cost per arm: **~8.0 h wall** for 12,000 steps, of which Lightning's train progress bar
reports only **2:36:50**. Lightning times validation on a separate bar, and the arithmetic says that
is where the other ~5 h goes: `val_max_samples: 2000` caps **articles, not chunks**, so 2000 full
PubMed articles pack into **15,724 chunks = 8M tokens**, evaluated six times, each pass running the
2.6B teacher alongside the student. **Validation looks like roughly twice the cost of training in
this screen.** Do not touch it mid-screen — A0 and A0-seed already ran with it — but size it down
for M8 and any future screen; 1M tokens is ample for a stable val PPL.

⚠ **A1 needs a 24 h clock, not 12.** Job 2513057 hit `TIMEOUT` at 12:00:03 with the run unfinished.
Mamba-1's `(d_inner, dstate)` `A` cannot use the cheap log-segsum (19.3 GB), so `scan_impl=exact`
flips the parallel axis instead — accepted as 3-5× slower because A1 never enters the pipeline —
and 3-5× of a 2:37 training loop plus ~5 h of validation does not fit in 12 h. Recorded as
`walltime` in `scripts/mamba3_arms.py`, which `env` now prints.

**M7-B RESULT (2026-09-06, array 2513632, all five arms 12,000 steps, seed 42, paired).**

| Arm | levers | val PPL | val loss | fit time |
|---|---|---|---|---|
| A0 s42 | control, legacy scan | 19.387 | 2.887 | 2:36:50 |
| A0 s1234 | noise-floor twin | 18.933 | 2.864 | 2:36:44 |
| **A2** | **SSD + 8× state** | **16.708** | **2.746** | **1:20:55** |
| A3 | + trapezoid | 16.719 | 2.746 | 1:28:00 |
| A4 | + RoPE | **1166.701** | 7.046 | 1:30:40 |
| A5 | + trapezoid + RoPE | **1166.185** | 7.045 | 1:44:09 |
| A6 | + `bc_bias`, no conv, both | **1168.531** | 7.047 | 1:35:37 |

**1. M7-C passes, decisively.** A2 beats A0 by **2.679 PPL paired (−13.8%)**, and by 2.225 even
against the luckier A0 seed — **4.2× the pre-registered 0.642 PPL bar**. The corrected operator is
not merely not-worse; it is a large win. Report it as a **bundle** (M7-F): SSD parameterization,
`d_state` 16→128, and a recurrence without the divide-and-clamp, all at once. A1 separates the last
of those and is re-running with a 24 h clock.

**2. Speed confirmed on equal footing.** A0 2:36:50 → A2 **1:20:55 = 1.94×**, same validation
schedule, same node class. The 300-step probe's 1.76× was conservative.

**3. The trapezoid is a null.** A3 − A2 = **+0.011 PPL — 1.7% of the bar**, in the *well-powered*
paired comparison (identical seed, data order, and `in_proj` shape, so the two runs start from the
same weights but for `trap_bias`). This is a real, reportable negative result on Prop. 1 at this
scale and context length.

**4. Every rope-on arm collapsed to unigram — and the cause was mine, not the paper's.**
7.05 nats against `ln(50257) = 10.82` for uniform: the models learned token frequencies and nothing
else. `mamba3_theta_max` **was never a field on `HybridConfig`**. The block's default of 1.0 was the
only value the campaign could run, and with `dt_limit=1.0` that permits **1 rad per token, 512 rad
over a 512-token sequence — 81 full turns.** A relative rotation `R(Θ_s − Θ_t)` is a position code
only while it stays inside one turn; past that it aliases, and because θ is *data-dependent* the
aliasing follows the content between s and t rather than the distance. The rotation stopped being a
positional encoding and became a scrambler of B and C in nine of twelve layers.

Why nothing caught it: **two** hand-maintained copies of the `mamba3_` forwarding whitelist in
`hybrid_block.py` both omitted `theta_max`, so the yaml round-trip test passed, the strict
unknown-kwarg guard never fired, and the ARCH fingerprint — which does not print `theta_max` —
looked correct. Fixed as a class, not an instance: the whitelist is now derived from
`inspect.signature(Mamba3Block.__init__)`, and a test asserts every block lever has a matching
`mamba3_*` config field. That test immediately found a second unreachable lever, `dt_init_floor`.

⚠ **This means M7-B has not yet tested Prop. 3/4.** A collapse caused by an unreachable
hyperparameter is evidence about the harness, not about complex-valued state. Reported as such.

**M7-B/G COMPLETE (2026-09-07). The headline is not the architecture — it is the defect.**

| Arm | levers | val PPL | fit | vs A0 s42 |
|---|---|---|---|---|
| A0 s42 | Mamba-1, legacy scan | 19.387 | 2:36:50 | — |
| A0 s1234 | noise-floor twin | 18.933 | 2:36:44 | — |
| **A1** | **Mamba-1 + exact scan + dt init + no Δ-norm** | **16.294** | 4:11:13 | **−3.093 (−16.0%)** |
| **A2** | **SSD + 8× state** | **16.708** | **1:20:55** | **−2.679 (−13.8%)** |
| A3 | + trapezoid | 16.719 | 1:28:00 | −2.668 |
| A4-lo | + RoPE, `theta_max` 0.002 (0.16 turns) | 16.431 | 1:24:49 | −2.956 |
| A4-mid | + RoPE, `theta_max` 0.02 (1.6 turns) | 16.534 | 1:31:41 | −2.853 |
| **A4-hi** | + RoPE, `theta_max` 0.2 (16 turns) | **16.199** | 1:31:23 | −3.188 |
| A4/A5/A6 | + RoPE, `theta_max` 1.0 (81 turns) | **1166.7** | — | **collapsed** |

**1. Both routes to a correct operator land in the same place.** `A1 − A2 = +0.414 PPL` — **64% of
the 0.642 bar, not significant**. Fixing the divide-and-clamp and the Δ init on the *existing*
Mamba-1 architecture recovers −3.09 PPL; migrating to SSD with 8× the state recovers −2.68. **The
entire measurable quality gain is the correctness fix, not the architecture.**

**2. What SSD buys is cost, and that is exactly what the plan predicted from arithmetic.** A2 trains
**3.10× faster than A1** and 1.94× faster than the broken A0. The exact scan is affordable in
Mamba-2/3's scalar-`A` form (19 MB) and not in Mamba-1's `(d_inner, dstate)` form (19.3 GB) — the
Context section argued this before a single GPU-hour was spent, and A1's 4:11:13 against A2's
1:20:55 is that argument measured. *That* is the case for Mamba-3 here: not better perplexity, the
same corrected-operator perplexity at a third of the cost and with 8× the state.

**3. The rope collapse was `theta_max`, confirmed.** 1.0 → 1166; 0.2 → 16.199; 0.02 → 16.534;
0.002 → 16.431. **My pre-registered *shape* was wrong**: I predicted `lo`/`mid` would recover and
`hi` would degrade. All three recovered and `hi` was the best arm in the screen. Harm therefore sets
in somewhere between 16 and 81 turns, not gradually from one turn. Recorded as a wrong prediction,
not smoothed over.

**4. A paired sensitivity floor, measured for free.** A4-lo/mid/hi share seed, data order and
`in_proj` shape and differ *only* in `theta_max` across a 100× range — yet they span **0.335 PPL,
non-monotonically**. That is trajectory sensitivity, not a mechanism, and it sets the resolution of
every paired comparison here. The A4-hi−A2 gap (0.509) sits barely above it.

**5. M7-D applied, and the replication flipped the ordering.**

| Arm | seed 42 | seed 1234 | mean | cross-seed spread |
|---|---|---|---|---|
| A0 | 19.387 | 18.933 | 19.160 | 0.454 |
| **A2** | 16.708 | **16.376** | **16.542** | **0.332** |
| A4-hi | **16.199** | 18.912 | 17.556 | **2.713** |

A4-hi led A2 by 0.509 PPL at seed 42 — 79% of the bar, so the rule said *advance the simplest, A2*.
At seed 1234 **A2 wins by 2.536**. On two-seed means A2 is better by **1.014 PPL**, and A4-hi's
cross-seed spread is **8.2× A2's**: the rope arm is not equal-but-different, it is *high-variance*,
and its seed-42 lead was a lucky draw.

**The pre-registered rule earned its keep.** Taking the lowest number on the day would have put a
high-variance arm into a 133 GPU-h pipeline. **Winner: A2.**

**Two M8 blockers found in these logs and fixed.** (i) `ModelCheckpoint(filename=
"stage0_kd-{step:06d}-{val/loss:.4f}")` — the slash in `val/loss` is a **path separator**, so every
save created a *directory* `stage0_kd-step=NNNNNN-val/` holding `loss=N.NNNN.ckpt`. Nothing globbing
`checkpoints/*.ckpt` could see a best checkpoint; only `last.ckpt` was ever visible, which is why
every arm reported zero checkpoints while holding 2.1 GB. (ii) `val_check_interval` was hard-coded
at 2000 — at M8-A's 120,000 steps that is **60 validation passes, ~54 h against ~13.5 h of
training**. Now `VAL_EVERY`, default 2000 for screens and 10000 for M8-A. The val *set* stays at
15,724 chunks so the number remains comparable to the 13.18 baseline.

- [x] **M7-C** **Gate: A2 ≤ A0 at 12K.** If the corrected operator is *worse*, **stop and report** — the buggy
      operator was acting as an unintended regularizer. That is a real finding; do not tune around it.
- [x] **M7-D** Per-lever deltas. **Pre-registered decision rule (written before the numbers exist):** advance
      the arm with lowest val PPL **only if Δ > 2× seed SD**; otherwise advance the **simplest** arm.
- [x] **M7-G** ⚠ **Re-test Prop. 3/4 at a rotation rate that is a position code.** The M7-B rope
      arms measured `theta_max=1.0` — the only value reachable at the time — not the mechanism.
      Three arms, `A4-lo/mid/hi` at `theta_max` ∈ {0.002, 0.02, 0.2} = {0.16, 1.6, 16} turns over
      512 tokens, everything else identical to A4. Pre-registered reading: if `lo`/`mid` recover to
      A2's ~16.7 and `hi` degrades, the aliasing account is confirmed and complex state gets a fair
      null-or-win; if **all three** still collapse, the fault is deeper than the rate and RoPE is
      reported as broken in this implementation, not as a failed mechanism. ~15 GPU-h.
- [ ] **M7-E** Mechanism-sensitive diagnostics (nearly free, and they test the actual claim): synthetic
      MQAR/induction probe + a positions-384-512-only PPL slice.
- [x] **M7-F** If A2 wins, note it as a **bundle** (SSD parameterization + 8× state), not "SSD is better",
      unless a `d_state=16` arm is run.


---

## Phases, version 2

Every phase ends with the validation harness exiting 0 and a commit on `h100_mamba3_v2`.

### V0 — Port the Mamba-3 branch onto the Phase-14/15 baseline (local, no GPU) ✅ COMPLETE 2026-09-17

`git merge h100_scaling_mamba3` (one merge commit, history and the M0–M7 evidence attached). Seven conflicts, all
mechanical; the semantic reconciliation is listed under V0-A/B. Merge commit **`b191178`**.

- [x] **V0-A** Merge and resolve: `configuration_hybrid.py` (`Literal`/`valid_types` = five types; `get_layer_config` now **raises** on an unknown type instead of falling through with `base_config`); `hybrid_block.py` (mamba3's capability dispatch `_mixer_takes_cu_seqlens` kept, hard-coded tuple dropped; `AttentionBlock` declares `supports_cu_seqlens = True`); `scan_interface.py` (**both** the `HYBRID_EXACT_SCAN=1` fp64 reference from 14C and the `scan_impl` legacy/exact dispatch, env hook first; module docstring names the three operators and the never-dispatched Triton import); `train_stage0_150m_h100.sh` (ARM resolver + `EXTRA_OVERRIDES` + v2's `MODEL_CONFIG` comment); `tests/test_willi_parity.py` (EOF blocks concatenated); `.gitignore` (union); `CLAUDE.md` (bootstrap rewritten for this branch).
- [x] **V0-B** Semantic checks: `train_report_generation.py` carries both `HybridConfig.from_hydra` and the intact 15C aux block; `train_contrastive.py` both `from_hydra` and the CheXpert label plumbing; the fingerprint prints `attentionx15` for the Transformer; `MAMBA3_INTEGRATION_PLAN.md` + state deleted by the merge (historical references in `H100_SCALING_PLAN.md` 14C / `analysis/scan_error_bound.md` left as written).
- [x] **V0-C** Gate 6 (`validate_for_willi.sh`) and the CI inline smoke build `["mamba","mamba3","mlstm","slstm","attention"]` — every parameter of every mixer receives a gradient.
- [x] **V0-D** **Pre/post-merge equivalence, measured:** `hybrid_150m_v2`, `hybrid_150m_v2_rrg`, `transformer_150m_baseline` built at `8137221` (worktree) and at the merged tree, seed 0: state-dict keys/shapes equal, **same-seed init identical, forward logits (packed docs, `cu_seqlens`) identical, pre-merge weights loaded into the post-merge model forward max|diff| = 0.000e+00** on all three. Every published number re-decodes byte-for-byte after the port.
- [x] **V0-E** Verification: `pytest -m "not cuda and not slow"` **416 passed / 1 skipped / 21 xfailed**; `validate_for_willi.sh` **9/9 under a real Python 3.9.23**; `evaluate_report_generation.py --smoke-test` OK; `mamba3_arms.py verify --full` 13 arms OK. **Finding, fixed:** `smoke_arch_refactor.py` failed its `max pre-clip grad-norm < 50` gate (73.7). Root cause measured: the fast path is bit-identical, but the slow path's shared scan multiplies `(dt·B)·x` where the deleted private copy did `(dt·x)·B` — a 3e-7 change on the logits that the 100-step tiny-model loop amplifies chaotically; **the unchanged pre-merge code scores 10.2 / 52.3 / 34.6 / 83.4 on data seeds 1–4**, i.e. fails its own gate on two of four. The gate now asserts what an explosion actually changes: loss halves, median pre-clip grad-norm < the clip value (measured 1.8–3.2), ≤ 5 spikes above 50 (measured 0–1). Passes on both paths, all seeds.
- [x] **V0-F** Commit `b191178`. Nothing pushed to `h100_scaling`.

### V1 — Re-baseline the plan and harden the seams (local)

- [x] **V1-A** `git mv MAMBA3_PLAN.md MAMBA3_PLAN_V2.md`, `git mv mamba3_state.json mamba3_v2_state.json`; this document; `scripts/mamba3_state.py` reads the V2 files and accepts `M*`/`V*` ids; `CLAUDE.md` bootstrap + Key files + kernel claims corrected (pure PyTorch; `ssd/`; dead Triton files named as dead); every code citation of `MAMBA3_PLAN.md` repointed (the M-sections are carried verbatim so the citations stay valid).
- [x] **V1-B** `configs/model/hybrid_150m_m3_rrg.yaml` — was missing. Built as `hybrid_150m_m3` + **exactly** the `hybrid_150m_v2 → _rrg` delta; `test_m3_rrg_config_is_m3_plus_exactly_the_rrg_delta` pins it and `test_rrg_model_configs_declare_aux_keys` covers it.
- [x] **V1-C** Arms `A2x` (`tfla_impl=exact`, seed 42) and `A2x-s2` (seed 1234) in `scripts/mamba3_arms.py`, expected fingerprint token `tfla_impl=exact`; `hybrid_150m_m3.yaml` now declares `scan_impl`/`tfla_impl` (it did not — the override would have been rejected by Hydra strict-struct, the `theta_max` incident again); `screen_arms_h100.sh` exports `VAL_EVERY`; ladder + paired-comparison tests extended.
- [x] **V1-D** `hybrid_xmamba/utils/checkpoint_arch.py::infer_architecture` — one owned parameter per mixer family (`dt_bias|B_bias`→mamba3, `dt_proj`→mamba, `i_gate_proj`→mlstm, `gate_proj`→slstm, `qkv_proj`→attention), ambiguity refused, sizes from shapes (`state_size`, `conv_size`, `expand_factor`, `dt_rank`, `mamba3_d_state`, `mamba3_head_dim`), topology aware that Mamba-3's BCNorm is unconditional. Wired into `evaluate_cxr_retrieval.py` (which now **raises** on critical missing keys, matching `evaluate_sts.py`) and `evaluate_sts.py` (no more hard-coded `state_size=16`); `train_contrastive.py` gained the >50%-missing guard the decoder and eval already had; `mamba3_watch.sh` parses `stage0_kd-step*.ckpt`.
- [x] **V1-E** Explicit operator pins: `scan_impl`/`tfla_impl` declared in all 15 yamls with recurrent layers (`legacy` everywhere except `hybrid_150m_a1` = `exact`); `test_every_recurrent_model_yaml_pins_the_operator_explicitly`. Dataclass defaults untouched.
- [x] **V1-F** `scripts/submit_v3_chain.sh` — the V3 chain, **`source`d** (no errexit, no python, only `sbatch --parsable` with `--dependency=afterok`), identical under bash and zsh, `DRY_RUN=1` prints the 16 submissions; per-seed incumbent dumps via `HYBRID_DUMP_<seed>` / `TRANSFORMER_DUMP_<seed>` (seed 42 defaulted; a missing one **skips that comparison loudly**, never silently unpairs). Parity test pins its wrappers and levers. `test_no_plan_command_invokes_a_bare_python_script_on_the_cluster` now scans this whole file — local commands are written `venv/bin/python …`.
- [x] **V1-G** Harness green (`validate_for_willi.sh`, full pytest, smokes), `venv/bin/python scripts/mamba3_state.py readme`, commit.

### V2 — Re-validate on the H100 and screen A2x (~3 h wall + queue)

- [x] **V2-A** `sbatch scripts/preflight_mamba3_h100.sh` (CPU, ~2 min) on the merged code, which has never run on the cluster: fingerprints for A0/A1/A2/A2x at 150M, the parameter band, Δ at init, the CPU suite.
  ⚠ **Attempt 1 FAILED (job 2552094, 2026-09-17, 13 s):** `python scripts/mamba3_arms.py verify` → `ModuleNotFoundError: No module named 'hybrid_xmamba'`. Running a script puts `scripts/` on `sys.path`, not the repo root, and the aisc `.venv` has **no editable install** of the package — only the laptop venv does, which is why every local check passed. `mamba3_arms.py` was the one script importing the package without inserting the repo root (27 others do); `env` never hit it because it imports nothing from the package. Its `afterok` dependant, the probe 2552095, went `DependencyNeverSatisfied`, and the screen array 2552097 waited on it. **Fixed** by the same `sys.path.insert` convention, plus two tests: a static rule over every `scripts/*.py`, and a subprocess reproduction with `python -S` (no `.pth` finder) that fails with the cluster's exact error when the fix is removed. Gate 1 measured at 25 s on 4 threads.
- [x] **V2-B** 300-step GPU probe: `ARM=A2x STEPS=300 EXPERIMENT=m3v2_probe_A2x sbatch --time=00:30:00 scripts/train_stage0_150m_h100.sh`. Log must show `mamba3x9 … tfla_impl=exact`, finite descending loss, no NaN; throughput vs A2's 2.18 steps/s (exact TFLA measured 1.03× on CPU — confirm on GPU).
- [x] **V2-C** Paired screen: `ARMS="A2x A2x-s2" VAL_EVERY=12000 SAVE_TOP_K_SCREEN=0 sbatch --array=0-1 scripts/screen_arms_h100.sh` — 12,000 steps, warmup 500, seeds 42/1234, the M7 val set unchanged (pairs with A2 16.708 / A2-s2 16.376), one validation pass at the end (~2.5 h each, parallel).
- [x] **V2-D** **Pre-registered rule, written before the numbers exist:** A2x advances to V3 **unless** (i) its 2-seed mean val PPL exceeds A2's 2-seed mean **16.542** by more than the M7 bar **0.642 PPL**, or (ii) it spikes/NaNs. Otherwise A2 advances and exact TFLA is reported as a screen-level null with the mLSTM defect left in the trained system (decode cache demonstrated on random weights only). Pin the chosen `tfla_impl` into `hybrid_150m_m3.yaml` + `_rrg` (V1-E test) **before** V3-A launches; record both PPLs in the arms table of the state file.

**V2 RESULT (2026-09-17). Preflight 2552163 passed; probe 2552164 exited 0; screen array 2552165 (both tasks on gx07).**

| Arm | seed 42 | seed 1234 | 2-seed mean | cross-seed spread |
|---|---|---|---|---|
| A0 (Mamba-1, both operators legacy) | 19.387 | 18.933 | 19.160 | 0.454 |
| A1 (Mamba-1, both operators exact) | 16.294 | — | — | — |
| A2 (SSD, TFLA legacy) | 16.708 | 16.376 | 16.542 | 0.332 |
| **A2x (SSD, TFLA exact)** | **15.566** | **15.788** | **15.677** | **0.222** |

- **The rule fired in A2x's favour**: better, not merely not-worse. Paired deltas −1.142 / −0.588; mean **−0.865 PPL
  (−5.2%)**, 1.35× the 0.642 bar; both seeds agree. A2x is also the most seed-stable arm measured.
- **Pairing caveat.** A2x validated with `VAL_EVERY=12000`, A2 with 2000. Validation creates a fresh dataloader
  iterator, which draws from the global RNG, so the two trajectories diverge after the first validation even at the
  same seed. The pairing is therefore weaker than M7's within-screen pairs. It does not affect the decision.
- **Cost.** 5:45:18 and 5:57:14 wall for 12,000 steps (≈1.73 s/step all-in), 1.75 / 1.71 it/s. The V2-C estimate of
  ~2.5 h per arm was wrong (see the speed correction above).
- **Pinned for V3.** `hybrid_150m_m3_rrg.yaml` runs `tfla_impl: "exact"`; `hybrid_150m_m3.yaml` stays `legacy` because it
  defines the M7 arms A2–A6. Stage-0 receives `exact` through `ARM=A2x`. `test_v3_decoder_config_runs_the_operator_its_stage0_arm_trained_with`
  pins that the two routes agree — the flag has no parameters, so a mismatch would load with `Missing keys: 0`.

### V3 — Full pipeline on the winner, matched to the 14A/15B protocol (~5–6 days wall)

`source scripts/submit_v3_chain.sh` the moment V2-D decides (`DRY_RUN=1` first). One Stage-0 at seed 42 (as both
incumbents have); decoder seeds 42/43/44 paired with 15B-3; 13D tower reused.

- [x] **V3-A** Stage-0 150M, 120K steps (`ARM=<A2x|A2> STEPS=120000 WARMUP_STEPS=2000 VAL_EVERY=10000 SAVE_TOP_K=1 EXPERIMENT=h100_stage0_150m_m3`), recipe otherwise identical to Phase 5 / 14A-3 (bs 16×3, LR 4e-4, clip 0.5, `GRAD_CKPT=true` — kept for recipe parity with Phase 5 / 14A-3, not for time). Expect **~58 h wall** at A2x's measured 1.73 s/step all-in (V2-C), inside the wrapper's 4-day limit; the unchanged 15,724-chunk val set is kept so PPL stays comparable. A single preemption restarts from step 0 (FM9) — check `sacct` daily. **Gate: val PPL reported against 13.18 (hybrid) and 11.222 (Transformer)**; there is no Stage-0 seed band for any of the three — say so; the 12K-screen spread (0.33–0.45 PPL) is the only noise estimate. If preempted, resume from `last.ckpt` via `train_stage0_distill_resume.py` rather than restarting (FM9).
- [x] **V3-B** Tower: reuse `outputs/h100_kd_150m_v2_full_data_lr3e6/checkpoints/last.ckpt` unchanged (14A-4 caveat: co-trained with the legacy hybrid text encoder; recorded, not fixed). No run.
- [x] **V3-C** Decoder × 3 (chain stage 2): `MODEL_CONFIG=hybrid_150m_m3_rrg`, `DECODER_CKPT=<V3-A last.ckpt>`, `NUM_GPUS=4 MAX_STEPS=12000 SEED=<s> SAVE_TOP_K=0 AUX_LAMBDA=0.0 PREFIX_K=32`, 13D tower. Read each log: `Missing keys: 0`, `prefix_k = 32`, `Training seed: <s>`, `Aux CheXpert loss: OFF`.
- [x] **V3-D** Eval × 3 → CheXbert × 3 → bootstraps × 9 (chain stages 3–5): official test split n=2663, `DECODE=beam BEAM_SIZE=3`, the incumbents' uncached path; `PER_LABEL=true` bootstraps per seed vs hybrid (same seed), vs Transformer (same seed), vs the floor → `analysis/bootstrap_m3_vs_{hybrid,transformer,floor}_seed{42,43,44}.md`. Apply decision 10; report mean ± SD per metric, never one seed. All six incumbent dump dirs are defaulted in the chain (resolved 2026-09-17 from `h100_scaling_state.json`). The decode runs with `EVAL_TIME=12:00:00` because this decoder's uncached beam speed is unmeasured.

**V3 RESULT (2026-09-20). Stage-0 2553736; decoders 2553737/43/49; evals 2553738/44/50; CheXbert 2553739/45/51;
bootstraps 2553740-42 / 46-48 / 52-54. Nothing preempted, nothing re-run.**

**1. The backbone gate passed, and it is the campaign's headline.** Stage-0 val PPL at 120K steps, one seed each,
identical recipe: hybrid **13.18** → Mamba-3 A2x **11.674** → Transformer **11.222**. Correcting both recurrences
and moving to SSD takes **−1.506 PPL (−11.4%)** off the hybrid and closes **77%** of the gap 14A-3 opened; the
Transformer keeps a 0.452 lead (4.0% relative). ⚠ No arm has a Stage-0 seed band, and 0.452 is the size of the 12K
screen's seed spread — "essentially matches the Transformer" is supportable, "beats" is not.

**2. That gain does not reach report generation.** Official test split, n=2663, beam 3, 3 seeds, decision 10 applied
(paired mean must exceed one baseline seed SD **and** hold sign at ≥2/3 seeds):

| metric | Mamba-3 | hybrid | Transformer | vs hybrid | vs Transformer |
|---|---|---|---|---|---|
| ROUGE-L | .1953 ± .0029 | .1949 ± .0047 | .1952 ± .0021 | tie | tie |
| BLEU-1 | .2484 ± .0005 | .2508 ± .0034 | .2478 ± .0022 | tie | tie |
| BLEU-4 | .0579 ± .0009 | .0578 ± .0032 | .0575 ± .0016 | tie | tie |
| CheXbert-14-micro | .4480 ± .0188 | .4480 ± .0223 | .4443 ± .0153 | tie | tie |
| CheXbert-14-macro | .2715 ± .0121 | .2660 ± .0122 | .2692 ± .0106 | tie | tie |
| CheXbert-5-micro | .5044 ± .0257 | .5086 ± .0382 | .5032 ± .0226 | tie | tie |
| CheXbert-5-macro | .4170 ± .0164 | .4193 ± .0274 | .4170 ± .0165 | tie | tie |
| exact-match-14 | .0452 ± .0028 | .0380 ± .0058 | .0455 ± .0027 | **Mamba-3 +.0071** | tie |
| exact-match-5 | .2242 ± .0070 | .2163 ± .0019 | .2244 ± .0057 | **Mamba-3 +.0079** | tie |
| example-F1 | .3858 ± .0174 | .3790 ± .0214 | .3817 ± .0144 | tie | tie |

- **Against the matched Transformer: indistinguishable on all 10 metrics.** Across 27 per-seed interval calls only
  one excluded zero (BLEU-1 at seed 44). This is the strongest same-protocol equivalence the project has measured.
- **Against the hybrid: two claims, both exact-match**, and both small. Exact-match-5 is positive at 3/3 seeds,
  exact-match-14 at 2/3. ⚠ Honest qualifier: the bar is the *hybrid's* seed SD as decision 10 specifies, and for
  exact-match-5 that SD is unusually tight (.0019) while Mamba-3's own is .0070 — against its own spread the claim
  is ~1.1 SD, i.e. marginal. Report it as "slightly more often reproduces the exact label set", nothing stronger.
- **The per-seed calls contradict each other**, exactly as in 15B-4: seed 42 gives Mamba-3 BLEU-4 and exact-match-14
  while the hybrid takes 5-micro; seed 43 gives Mamba-3 three CheXbert metrics while the hybrid takes BLEU-1;
  seed 44 gives one each. Any single-seed reading of this table would be wrong.
- **Against the retrieval floor** the standing pattern is unchanged and now holds for a fourth architecture:
  text and exact-match win at 3/3 seeds, **CheXbert-14-macro loses at 3/3** (−.0180 / −.0294 / −.0422), 5-macro
  loses at 2/3. Lung Lesion is 0.000 / 0.022 / 0.000 and Pleural Other 0.033 / 0.017 / 0.000 — the rare-label
  deficit that survived 13F and 15C survives a corrected operator and a new mixer too.

**3. What this licenses.** *At matched parameters, with both recurrences computing what they are specified to
compute, the attention-free hybrid is statistically indistinguishable from a parameter-matched Transformer on
every report-generation metric measured, and essentially indistinguishable from the incumbent hybrid despite a
much better language-modelling backbone.* The backbone improvement is real and measured; its non-transfer is the
finding, and it is the eleventh instance of this project's recurring result that text-side gains do not move the
downstream clinical metrics.

**4. Cost, settled from `sacct` wall clocks (12,000 steps each), closing the plan's open speed question.**

| arm | wall | s/step | note |
|---|---|---|---|
| A0, Mamba-1, both operators defective | 7:59:00 | 2.395 | |
| A2, SSD, legacy TFLA | 4:09:14 | 1.246 | **A0/A2 = 1.92×**, so M7's 1.94× from the progress bar was right after all |
| A2x, SSD, exact TFLA | 5:45:19 | 1.727 | **1.39× the cost of A2** — the price of the correct mLSTM recurrence |

Corrected Mamba-3 is **1.39× faster** than the shipped hybrid at Stage-0 (57h23m vs the hybrid's ~74h at 2.22
s/step), and the report-gen decoder trains **1.7× faster** (1:19 vs 2:15–2:30). Generation is **1.75× slower**
(5:16 vs 13D's 3:01 on the same split), because beam search re-runs the model with no state cache; the M6 cache
is the shipped answer to that and is unused by this eval path.

- [x] **V3-E** Diversity re-measure on the seed-42 dump (`analyze_diversity_h100.sh`, controls = references + floor) — the boilerplate confound on any ROUGE-L movement.
**V3-E RESULT (job 2560259).** Seed 42, official test split, same protocol and controls as 14B-2. Duplicate-cluster
rate **26.0%** against 13D's 29.2% on this split, with references at 0.2% and the retrieval floor at 7.3%.
Lexical diversity moves the same way and by about as little: distinct-2 0.0350 vs 13D's 0.0299, self-BLEU-4 0.6709
vs 0.6854, mean length 58.5 vs 58.2 tokens. Pre-registered outcome fires **INTERMEDIATE**, as it did for 13D:
neither the ≥70% qualifier trigger nor the at-or-below-controls clean positive. So the corrected operator writes
marginally less templated text and stays in the same regime — it is **not** an explanation for any metric movement,
which is what this check exists to rule out.

- [x] **V3-F** Efficiency + decode: `MODELS="hybrid_150m_v2 hybrid_150m_m3 transformer_150m_baseline" OUTPUT_DIR=analysis/efficiency_150m_m3 sbatch scripts/profile_efficiency_h100.sh` (the 14A-7 protocol, random weights) plus `performance_profile.py --decode` for the O(L²)→O(L) curve (`tfla_impl=exact`, random weights suffice). Pre-registered: SSD is matmul-shaped so the 14A-7 gap should narrow; report exponents and crossover whichever way it lands.
**V3-F RESULT (job 2561023, after the bf16 fix).** 14A-7 protocol, bf16, bs=4, H100.

| L | hybrid | Mamba-3 | Transformer |
|---|---|---|---|
| inference 256 | 17.6 ms / 1.22 GB | 20.6 ms / **0.51 GB** | 4.9 ms / 0.51 GB |
| inference 2048 | 120.6 ms / 5.71 GB | 86.2 ms / **1.24 GB** | 16.1 ms / 1.24 GB |
| inference 16384 | 915.6 ms / 42.00 GB | 678.2 ms / **7.09 GB** | 163.9 ms / 7.15 GB |
| training 2048 | 1096 ms / 53.96 GB | **416 ms / 13.87 GB** | 52 ms / 7.55 GB |
| training ceiling | **OOM at L≥4096** | OOM at 16384 | none |

- **Memory reaches parity with FlashAttention** (7.09 vs 7.15 GB at 16384; exponent 0.643 vs 0.644) where the
  incumbent needed 42 GB and 0.863. The SSD activation argument, measured end to end at 5.9×.
- **Training is where it pays**: 2.6× faster and 3.9× less memory at L=2048, and it trains at lengths the
  incumbent OOMs on. Latency exponent 0.886 vs the incumbent's 1.285.
- **Attention still wins latency** at every length (4.1× at 16384 inference), fused kernel vs pure-PyTorch
  scan. 14A-7's direction stands; its magnitude shrinks and its memory half now goes the other way.
- **Decode, the O(1) claim measured**: cached 0.00635 s/token vs 0.03293 recompute = **5.19×**, growth
  **1.00×** (flat). TTFT is 4.0× worse (token-by-token prefill, the documented M6 limitation). The incumbent
  has no cache at all — its legacy mLSTM computes no recurrence an O(1) step could reproduce.
- ⚠ **Excluded as not credible**: the Transformer's uncached decode, 0.337 s/token, 13× slower than the hybrid
  in the same loop while 3.5× faster in the sweep. Flagged unverified; re-measure before it is ever cited.

⚠ **V3-F was blocked once and the cause is worth recording.** Job 2560261 died in `ssd_chunked_scan` with
`expected scalar type Float but found BFloat16`: the profiler builds the model with `.to(bfloat16)` and no
autocast, while the scan builds its decay factors in fp32 by policy (FM3), and `einsum` does not promote.
Training never saw it because autocast unifies the operands for us. Fixed by unifying einsum operand dtypes to
the mixer's dtype and returning the scan in the input dtype — the same casts autocast already applied, verified
**bitwise identical** on the fp32 and bf16-autocast forwards, so no trained number moves. Three regression
tests; `DECODE_CURVE` added to the wrapper so the decode curve is reachable through `sbatch`.

- [x] **V3-G** Tick/notes/evidence after every job; never re-run a checkpoint-producing step without reading its log first.

### V4 — Writeup + cleanup (after V3)

- [x] **V4-A** `analysis/mamba3_results.md`: audit table, the OFAT ladder, "the headline is the defect" (A1 −16.0% vs A2 −13.8%, not significantly apart), the A2x screen, Stage-0 vs both incumbents, the 3-seed report-gen table with decision 10 applied, efficiency/decode curves, the M4-D parity capability and its Δ caveat, every null stated plainly. Cross-link from `analysis/PHASE14_SUPERVISOR_REVIEW.md` (limitation #2) and `analysis/h100_scaling_results.md`.
- [ ] **V4-B** Retire py3.9 (old M9-B): drop gates 1–3, target py3.11, `scripts/validate.sh` + a `validate_for_willi.sh` shim, delete the willi CI workflow; keep gates 4–6. Only after V3 is submitted.
- [ ] **V4-C** Dead code (old M9-C): delete `mamba_block_v2.py`, `mlstm_block_v2.py`, `hybrid_layer.py`, `scan_triton.py`, `tfla_triton.py`, root `test_hybrid_implementations.py`; keep `debug_checkpoint_keys.py` / `check_checkpoint_compatibility.py` (documented tools).
- [ ] **V4-D** One note in `h100_scaling_state.json` (the Phase-5 PPL and the 13D/15B headline are *compared against*, not superseded); `mamba3_v2_state.json` verdict; `readme`. **Do not merge into `h100_scaling`.**
- [ ] **V4-E** Archive: V3 checkpoints/dumps are DUA-covered and HOME is deleted 6 months after expiry — add them to the Phase-15 archive manifest.

### V5 — Gated / optional (cluster access confirmed 2026-09-19; the remaining gates are scientific)

- [ ] **V5-A** 14C-2/14C-3 closure on **13D** (the supervisor's open limitation #2): `HYBRID_EXACT_SCAN=1` teacher-forced PPL on n=2663, exact vs default, then beam+CheXbert on a 300–500 subsample. Answers "does the bug affect the reported numbers" by measurement.
- [ ] **V5-B** M7-E mechanism diagnostics (MQAR / late-position PPL slice).
- [ ] **V5-C** Ratio screen (`12/0`, `10/2`, `9/3`, `8/4`) **only if V3-D claims a win** under decision 10; same rule; efficiency trade reported alongside.

---

## Verification (every phase)

1. `bash scripts/validate_for_willi.sh` exits 0 — Hydra invariants, `pytest -m "not cuda and not slow"`, CPU fwd/bwd
   smoke with **no missing gradients over the five-type pattern** `["mamba","mamba3","mlstm","slstm","attention"]`.
   Report any conda degradation explicitly (it happened once, 2026-09-09).
2. `tests/test_mamba3_numerics.py` fully green; `tests/test_scan_correctness.py` still green (the default path is
   unchanged — that test is the machine-enforced operator freeze).
3. Every new flag asserted **bit-identical in its default state** (documented exceptions: `bc_bias=one_init`,
   `use_conv`, BCNorm).
4. `venv/bin/python scripts/evaluate_report_generation.py --smoke-test` — the image→LM conditioning path.
5. `venv/bin/python scripts/smoke_arch_refactor.py` — 100-step CPU loop: loss halves, median pre-clip grad-norm
   below the clip, ≤ 5 spikes (the trajectory-robust gate from V0-E).
6. `venv/bin/python scripts/mamba3_arms.py verify --full` — every arm builds the operator it claims at 150M.
7. On the cluster: every training log shows the expected `ARCH …` fingerprint, `Missing keys: 0`, `prefix_k = 32`,
   `Training seed: N`; reconcile in-training vs authoritative eval numbers before citing any figure.

**Cluster invariants**: `--partition=aisc-batch --account=aisc --gpus=N` (**never `--gres` for GPUs** —
rejected live), `--exclude=ga03,gx17v1,gx13v1`, `--requeue` (preemptible), `--open-mode=append` (a requeue
otherwise truncates the log and the restart-from-step-0 leaves no trace), `torch.compile` OFF for anything
touching custom kernels, `HF_HUB_OFFLINE=1`.

⚠ **The login node executes nothing.** Not `python`, and not `bash script.sh` either — the guard fires
before the script's first line, and its refusal text word-splits into `command not found` noise. Three
incidents in the M7 campaign (the pre-flight instruction; the `eval "$(python …)"` launch that silently became a
second A0 at 120,000 steps, job 2513581; `bash scripts/mamba3_watch.sh`). **Anything scripted runs through
`sbatch`/`srun`, or is designed to be `source`d** (`submit_v3_chain.sh`, `mamba3_watch.sh`). Individual
commands (`squeue`, `sacct`, `grep`, `cat`, `ls`) remain fine interactively.

---

## Compute budget (measured rates: A2 1:20:55 per 12K steps incl. validation; decoder 2h15 on 4×H100)

| Item | GPU-h | Wall |
|---|---|---|
| V2 probe + A2x screen (2 seeds, parallel) | ~6 | ~3 h + queue |
| V3-A Stage-0 (A2x measured 1.73 s/step; ×1.5 retry budget) | ~58–87 | ~2.5–3.5 d |
| V3-C decoder ×3 (4 GPU, ~2 h each, parallel) | ~24 | ~2 h |
| V3-D eval + CheXbert ×3 + bootstraps ×9 | ~20 | ~8 h |
| V3-E/F | ~1 | minutes |
| **Total V3** | **~105–135** | **~4–5 d** wall from submission, absent preemption |

**Cut order if the budget bites:** V5 entirely; V3-F's forward+backward sweep (keep inference + decode); a third
decoder seed (report 2 and say so); never the Stage-0 validation set (comparability with 13.18). With the
extension approved this order is a contingency for queue/preemption trouble only — not a schedule.

---

## Top risks

| # | Risk | Early warning | Mitigation |
|---|---|---|---|
| **FM1** | Δ-init confound — A2 differs from A0 in Δ regime *and* operator at once | per-layer `Δ.mean/max` at steps 0/100/1000 | **measured**: A1 isolates the fix (−16.0%); A1−A2 not significant |
| **FM2** | Spike collapse at 150M (history: 5 attempts; step 24749 was one 1.59 grad-norm). Mamba-3 opens two surfaces: Δ unnormalised, `A` clamped at `A_floor` | `Δ.max()>10`, `\|A\|.min()` pinned for >5% of heads, grad-norm > 3× trailing median | `gradient_clip_val=0.5`, `dt_limit=(0,1)`, first validation at 10K as the tripwire |
| **FM3** | bf16 in the scan | rel-err vs the fp64 oracle | keep fp32 for `dt`, `A`, `Θ`, decay; bf16 only behind a flag defaulting off |
| **FM4** | RoPE angle accumulation | `Θ.abs().max()` > 1e3 rad | fp64 accumulation, per-segment reset, `remainder(·,2π)` — moot for A2/A2x (rope off) |
| **FM5** | **Silent config drop** — highest expected cost, happened twice (`norm_topology`; `theta_max`) | the `ARCH` fingerprint at step 0 of every log | `from_hydra` everywhere; derived mamba3 whitelist; prefixed-unknown kwargs raise; five-type Gate 6; explicit yaml pins; the chain passes `PREFIX_K=32` so a `run_metadata.json` mismatch is a hard error |
| **FM6** | The screen is underpowered by construction | seed twins | measured: bar 0.642; per-lever deltas paired; V2-D judged on 2 seeds |
| **FM7** | Recipe drift across arms | wall-clocks differing without explanation | `GRAD_CKPT=true` throughout; `max_steps` = screen length; one chain script |
| **FM8** | **Home quota (200 GiB)** killed 3 of 4 arms in 15B | `du` before every submission | `SAVE_TOP_K=0` for decoders, `1` for Stage-0; delete probe outputs |
| **FM9** | **Preemption restarts Stage-0 from step 0** (`--requeue` passes no `ckpt_path`) | `sacct` shows REQUEUED; log has two `ARCH` lines | resume via `train_stage0_distill_resume.py` from `last.ckpt`; `--open-mode=append` keeps the evidence |
| ~~FM10~~ | ~~Account expiry~~ — **RETIRED 2026-09-19, extension approved** | — | Archiving DUA-covered outputs off-cluster stays live as V4-E, on DUA grounds |

---

## State-tracking contract (`mamba3_v2_state.json`)

1. Session start: read `MAMBA3_PLAN_V2.md` + `mamba3_v2_state.json` (pointed to from `CLAUDE.md`).
2. Resume at `current_phase`; the checkboxes here are ground truth.
3. After **every** meaningful change: tick the checkbox, update `last_updated`, append a `notes` line, record the
   evidence under `phases[<id>].evidence` — through the helper, never by hand in two files:

   ```bash
   venv/bin/python scripts/mamba3_state.py tick V2-C --note "..." --evidence job=...
   venv/bin/python scripts/mamba3_state.py phase V3_full_pipeline --status "..."
   venv/bin/python scripts/mamba3_state.py readme      # refresh README's status line + progress table
   venv/bin/python scripts/mamba3_state.py show [V2]   # progress at a glance
   ```

   Run `readme` at the end of every phase.
4. Never re-run a checkpoint-producing phase (V2-C, V3-A/C) without first reading its log and logging a verdict.
5. If `mamba3_v2_state.json` is lost, `venv/bin/python scripts/mamba3_state.py sync` regenerates the phase tree.

---

## Unresolved questions

- V5-A (14C-3 on 13D, ~4 GPU-h): run inside V3's window or defer?
- Speed: re-derive M7's A0/A2 wall clocks with `sacct` before quoting any speed-up (see the M7 correction).
- A1 has one seed. If the A2x-vs-A1 gap (−0.728) goes into the writeup as an architecture claim, an `A1-s2` arm (~8 h) is the pre-agreed way to make it two-seed.
