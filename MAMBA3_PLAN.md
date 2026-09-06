# Mamba-3 Backbone Upgrade — Plan of Record

> **ACTIVE plan-of-record. Read this + `mamba3_state.json` at session start** (pointed to from `CLAUDE.md`).
> Branch: **`h100_scaling_mamba3`**, cut from `h100_scaling` @ `20a1d27`. **No merge without an explicit
> instruction from the user** — the approved 13D results stay reachable and reproducible on `h100_scaling`.
> Resume at `mamba3_state.json["current_phase"]`; the checkboxes below are ground truth.
> Supersedes `MAMBA3_INTEGRATION_PLAN.md` + `mamba3_integration_state.json` (`PLANNED — no code written`;
> framing stale — it targets the retrieval era and declines RoPE/MIMO).
> `H100_SCALING_PLAN.md` is `current_phase: plan_closed` and stays closed. Its results are the baseline.
> Source: Mamba-3, arXiv:2603.15569 + reference impl `state-spaces/mamba` `mamba_ssm/modules/mamba3.py`.

---

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

## Locked decisions

| # | Decision |
|---|---|
| 1 | **Isolation**: all work on new branch **`h100_scaling_mamba3`**, cut from **`h100_scaling`**. The approved 13D results and every existing checkpoint stay reachable and reproducible on `h100_scaling`. ⚠ **No merge, ever, until the user explicitly instructs one** — not on plan completion, not on a passing gate. The branch stays parallel by default. |
| 2 | **Scope**: new `Mamba3Block` = Mamba-2 SSD base + exponential-trapezoidal + complex/RoPE + B/C biases + optional conv-drop, every feature behind a flag reducing **exactly** to Mamba-2. Plus an O(1) recurrent decode cache. |
| 3 | **MIMO**: plumbing shipped (`mimo_rank`, default 1), **never run**. It breaks parameter matching by +3.2% (189.6M) and needs its own screen *and* full pipeline (+96 GPU-h). Documented as future work. |
| 4 | **Gate**: staged — short-run Stage-0 PPL screens arms; the winner gets the full pipeline and report-gen on the official n=2663 test split. Retrieval R@10 is a **non-regression guard only**. |
| 5 | **Retrain**: full pipeline for the winner — Stage-0 → contrastive tower → report-gen decoder. |
| 6 | **`d_state = 128`**, `headdim=64`, `ngroups=1`, `expand=2`. Parameter-matched to +0.24%. |
| 7 | **`scan_impl`**: default `legacy` through M7 so the A0 control stays bit-reproducible against published numbers; **flip the default to `exact` at M9**, keeping `legacy` behind the flag solely to reproduce pre-2026-09 results. No silent change mid-campaign. |
| 8 | **`layer_pattern`** stays **9 mamba3 + 3 mlstm**, fixed, for the whole M1-M9 campaign — it is the control, and one variable at a time. **If Mamba-3 wins M8, the ratio IS re-opened** as its own gated phase M10 with its own screen. |
| 9 | **willi/A100 retired.** py3.9 guards dropped, target py3.11 (the aisc `.venv`). Dead code deleted and documented, not left ambiguous. |

---

## Baselines any arm must be measured against

| Metric | Incumbent (13D) | Retrieval-NN floor | Source |
|---|---|---|---|
| Report-gen ROUGE-L (test, n=2663, beam=3) | **0.1899** | 0.1636 | `analysis/h100_scaling_results.md:124` |
| BLEU-4 | 0.0542 | 0.0330 | " |
| CheXbert-14 micro / macro | 0.4736 / 0.2800 | 0.4296 / **0.3014** | " |
| CheXbert-5 micro / macro | 0.5522 / 0.4487 | 0.4856 / 0.4284 | " |
| Stage-0 val PPL (PubMed) | **13.18** | — | Phase 5 |
| MIMIC i2t R@10 (clean protocol) | 0.1459 (SE 0.57pp) | — | Phase 6G-5 |

Pre-registered tiers unchanged: ROUGE-L Floor 0.15 / Target 0.22 / Stretch 0.26; CheXbert-14-micro
0.25 / 0.40 / 0.50.

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
- [ ] **M4-D** **Capability test — the paper's headline claim.** Parity + modular arithmetic on a tiny model
      (Table 5b): Mamba-3 ≈100%, Mamba-2/Mamba-1 ≈ chance. Cheap CPU test, and the cleanest standalone
      contribution in the plan.
- [x] **M4-E** Angle-drift test: fp32 path vs fp64 sequential rotation < 1e-6 at L=1024;
      alarm if `Θ.abs().max() > 1e3` rad.

### M5 — Flags folded into arm definitions (no milestone of its own)
- [ ] **M5-A** `bc_bias ∈ {none, zero_init, one_init}`; assert `none ≡ zero_init` bitwise.
- [ ] **M5-B** `use_conv=False` path; `n_params` moves by exactly `9 × 8,960`.
- [ ] **M5-C** `mimo_rank` plumbing, default 1, asserted bit-identical. **Never run.**

### M6 — O(1) recurrent decode cache
Measured on a tiny CPU model, per-token cost rises monotonically (0.0135 → 0.0173 s/tok from 16 → 128 new
tokens; doubling ratios 1.9× → 2.4×): **confirmed super-linear, trending quadratic**, paid 3× under beam=3.
Only pays off if **every** layer is cacheable — TFLA already carries the `m_state` LSE stabilizer across
chunk boundaries (`tfla_interface.py:110-117`), so an exact mLSTM `step()` is derivable.
- [ ] **M6-A** `Mamba3Block.step()` + `allocate_inference_cache()` (reuse M2-A's oracle). State:
      `h (nheads, headdim, d_state)` + `angle_state` + `B_prev`/`x_prev` ≈ **7.1 MB fp32** for the whole model at bs=1.
- [ ] **M6-B** `mLSTMBlock.step()` + cache (`C`, `n`, `m`).
- [ ] **M6-C** Cache plumbing through `HybridBlock.forward`, `generate()`, `beam_search_decode` — including
      the `prefix_embeds` branch.
- [ ] **M6-D** **Equivalence: cached decode == full recompute, `atol ≤ 1e-5`**, greedy and beam=3, with and
      without an image prefix.
- [ ] **M6-E** Add prefill / per-token decode / TTFT to `performance_profile.py` — the repo has **no**
      decode-latency benchmark (`evaluate_lm.py:170-194` and `performance_profile.py` time full-sequence
      forwards only). Report O(L²) → O(L).

### M7 — Timing probe + short-run screen (**the cheap decision gate**) — H100
- [ ] **M7-A** ⚠ **Do this before anything else costs money.** (i) Run **A0 twice with different seeds** at
      screen length; measure |ΔPPL| = the noise floor. (ii) 200-step timing probe **with and without the
      BioMedLM teacher** (2 × 15 min) — the 2.7B teacher is ~10:1 of per-step FLOPs and contributes nothing
      to ranking. If seed noise ≥ the expected 1-3% effect, **the screen cannot rank arms** — change the
      metric before spending 50 GPU-h.
- [ ] **M7-A2** **Early-start the Mamba-1 arms.** A0, A0-seed and A1 need only M1's flags, so they can
      queue while M2–M6 are still being built locally — ~22 GPU-h of queue time overlapped with
      development, and it validates the screen harness before the expensive arms exist.
- [ ] **M7-B0** `scripts/screen_arms_h100.sh` as a **SLURM job array** (`--array=0-7`), arm selected by
      `$SLURM_ARRAY_TASK_ID` from a table in the script. One submission instead of eight; each task takes a
      GPU as one frees. Assert the arm table in `tests/test_willi_parity.py` the way the other SLURM
      wrappers are asserted.
- [ ] **M7-B** Screen A0, A0-seed, A1, A2, A3, A4, A5, A6 — **12,000 steps**, 150M, `aisc-batch`,
      **identical seed and data order** (paired comparison on Δ log-loss, not independent PPL).
      Set `trainer.max_steps` to the screen length so **WSD reshapes its own decay** — a run stopped at 12K of
      a 120K schedule never enters decay, and decay is where models separate. Warmup 500.
      **Hold `GRAD_CKPT=true` fixed across all arms** even though SSD makes it unnecessary (see FM3).
      If M7-A shows the teacher dominates, screen at 70M first (~3× cheaper) then confirm the top 2 at 150M.
- [ ] **M7-C** **Gate: A2 ≤ A0 at 12K.** If the corrected operator is *worse*, **stop and report** — the buggy
      operator was acting as an unintended regularizer. That is a real finding; do not tune around it.
- [ ] **M7-D** Per-lever deltas. **Pre-registered decision rule (written before the numbers exist):** advance
      the arm with lowest val PPL **only if Δ > 2× seed SD**; otherwise advance the **simplest** arm.
- [ ] **M7-E** Mechanism-sensitive diagnostics (nearly free, and they test the actual claim): synthetic
      MQAR/induction probe + a positions-384-512-only PPL slice.
- [ ] **M7-F** If A2 wins, note it as a **bundle** (SSD parameterization + 8× state), not "SSD is better",
      unless a `d_state=16` arm is run.

### M8 — Full pipeline on the winner — H100
Control is the **existing** 13D result — do not rebuild it.
**Submit the whole chain at once** with `--dependency=afterok:<jobid>`: Stage-0 → tower → decoder → eval →
CheXbert, each firing automatically when its predecessor succeeds. One submission, no babysitting between
stages, and no idle days waiting for a human to notice a job finished. The eval and CheXbert steps (~8.5 h
combined) share a single allocation so their queue time is paid once. `scripts/submit_m8_chain.sh` owns this.
- [ ] **M8-A** Stage-0 150M, 120K steps. Measured **2.22 s/step → 74 h**; budget **1.5× for retries** (the
      150M Stage-0 needed five attempts historically). **Gate: PPL ≤ 13.18.**
      Harvest `GRAD_CKPT=false` here only (1.3-1.5×, numerically equivalent with `use_reentrant=False`) —
      verify the first 2000 steps match before committing.
- [ ] **M8-B** Contrastive tower, canonical recipe (`vit_unfreeze=12`, `vit_lr=3e-6`, `bs=64`, `head_lr=4.24e-4`),
      full data. **Guard: MIMIC i2t R@10 ≥ 0.1459 clean protocol.** A null is *expected* and is not a failure;
      only a regression blocks. Never quote a positive without ±0.57pp.
- [ ] **M8-C** Report-gen decoder, `NUM_GPUS=4 MAX_STEPS=12000`, tower from M8-B.
- [ ] **M8-D** Official test split n=2663, `DECODE=beam BEAM_SIZE=3`, then CheXbert as a second job in
      `.venv_chexbert`. **Primary gate: ROUGE-L > 0.1899 and CheXbert-14-micro > 0.4736**, both also vs the
      retrieval-NN floor (0.1636 / 0.4296). Fix beam, `max_new_tokens` and the split up front — evaluate once.
- [ ] **M8-E** Re-measure the boilerplate/template-duplication rate — an open item on the incumbent and the
      obvious confound on any ROUGE-L movement.

### M9 — Cleanup, writeup, reintegration
- [ ] **M9-A** Flip `scan_impl` default to `exact`; keep `legacy` documented as reproduction-only.
- [ ] **M9-B** Retire py3.9: drop harness gates 1-3 (AST/PEP-604/PEP-585), target py3.11, delete the willi CI
      workflow, rename to `scripts/validate.sh` with a back-compat shim. Keep gates 4-6 (Hydra invariants,
      pytest, fwd/bwd smoke).
- [ ] **M9-C** Delete dead code — `mamba_block_v2.py`, `mlstm_block_v2.py`, `hybrid_layer.py`,
      `scan_triton.py`, `tfla_triton.py`, and the root-level `test_hybrid_implementations.py` (their only
      consumer). Correct `CLAUDE.md`'s false "uses chunk-parallel selective scan Triton kernel" claim and
      state plainly that the kernels are pure PyTorch, with a Triton SSD kernel recorded as future work.
- [ ] **M9-D** `analysis/mamba3_results.md`: the audit table, the OFAT ladder, the "not the specified
      recurrence" framing, decode-latency curves, the parity/state-tracking result, and every null recorded honestly.
- [ ] **M9-E** Update `mamba3_state.json` verdict; append a note to `h100_scaling_state.json` — Phase 5's PPL
      and the 13D headline are superseded either way. ⚠ **Do not merge.** `h100_scaling_mamba3` stays a
      parallel branch indefinitely; a merge happens only on an explicit instruction from the user.

### M10 — Re-open the mamba/mLSTM ratio (**gated: only if M8-D clears its primary gate**)
Held fixed at 9 mamba3 + 3 mlstm for all of M1-M9 so every earlier arm is single-variable. Once the Mamba
side is measurably stronger, the 25%-mLSTM split — inherited from the v2 refactor and never re-derived
against a Mamba-3 mixer — is no longer justified by evidence.
- [ ] **M10-A** Screen ratio arms at 12K steps, holding the winning M8 operator and every other lever fixed:
      `12/0` (pure mamba3), `10/2`, **`9/3` (control)**, `8/4`, and one interleaving-position variant
      (mLSTM centred vs distributed). Parameter counts differ across arms — report them, and use the
      structural-equality test to show only the mixer composition changed.
- [ ] **M10-B** Same pre-registered decision rule as M7-D: advance only if Δ > 2× seed SD, else keep `9/3`.
      **The efficiency trade is part of the decision, not a footnote** — `analysis/h100_scaling_results.md:213-236`
      records xLSTM as ~3× faster and ~6× lighter than Mamba at L=16384, so dropping mLSTM layers costs real
      throughput. Report PPL *and* the M6-E decode curves together.
- [ ] **M10-C** Full pipeline only if M10-A produces a winner that also clears the M7-D bar. Otherwise record
      the null and keep `9/3`.

---

## Verification (every phase)

1. Validation harness exits 0 — Hydra invariants, `pytest -m "not cuda and not slow"`, CPU fwd/bwd smoke with
   **no missing gradients**, now over a 4-type layer pattern.
2. The phase's new test passes and its `xfail` flips.
3. Every new flag asserted **bit-identical in its default state** (documented exceptions: `bc_bias=one_init`,
   `use_conv`, BCNorm).
4. `python scripts/evaluate_report_generation.py --smoke-test` (seconds, CPU) — the image→LM conditioning path.
5. `python scripts/smoke_arch_refactor.py` (~2 min, CPU) — 100-step train loop, loss decreases, grad-norm < 50.
6. Reconcile in-training vs authoritative eval numbers before citing any figure.

**Cluster invariants**: `--partition=aisc-batch --account=aisc --gpus=N` (**never `--gres` for GPUs** —
rejected live), `--exclude=ga03,gx17v1,gx13v1`, `--requeue` (preemptible), login node refuses execution so
everything goes through `sbatch`, `torch.compile` OFF for anything touching custom kernels, `HF_HUB_OFFLINE=1`.

---

## Compute budget (measured: 2.22 s/step at 150M, from `hpi_results_logs/h100_stage0_150m_2341991.log:303`)

| Item | GPU-h | Wall |
|---|---|---|
| M7-A probes (2 seeds + teacher on/off) | ~8 | 1 d |
| M7-B screen, 8 arms × 12K steps × 7.4 h | 59 | 2-3 d |
| 150M confirmation of top 2 (if screened at 70M) | 15 | 1 d |
| M8 full pipeline, **1 winner**, Stage-0 ×1.5 retry | 133 | 4-6 d |
| M9 writeup / re-evals | ~15 | 2 d |
| **Total** | **≈ 230 GPU-h** | **10-13 d** |

**Cut order if the budget bites:** (1) MIMO — already cut; (2) a second full-pipeline winner (+133 GPU-h) —
use the existing 13D result as the control; (3) a full pipeline for A1 — it belongs in the screen only;
(4) screen length beyond 12K (20K costs 98 GPU-h for the same set — if underpowered, fix the *metric*, not
the steps); (5) repeated report-gen evals at 7 GPU-h each.

**The single highest-ROI measurement** is M7-A's teacher probe: if BioMedLM is >50% of per-step time,
screening at 70M with the teacher off drops the 8-arm screen from 59 GPU-h to under 10 — enough to afford a
2-seed replicate on every arm.

---

## Top risks

| # | Risk | Early warning | Mitigation |
|---|---|---|---|
| **FM1** | **Δ-init confound** — A2 differs from A0 in Δ regime *and* operator *and* scan correctness at once, so a win is uninterpretable | log per-layer `Δ.mean/max`, `\|Δ·A\|.max` at steps 0/100/1000 for every arm | the A1 control arm (~7 h of screen compute) |
| **FM2** | **Spike collapse at 150M.** History: 5 attempts, collapses at 3k/24k/28k; step 24749 was one grad-norm 1.59 vs a 0.23 baseline. Mamba-3 opens **two new surfaces**: Δ is no longer normalized (`dt_norm` was an accidental stabilizer) and unbounded above; `A` clamped at `A_floor` gives a dead unit *and* `exp(ΔA)≈1` (no forgetting) | alarm on `Δ.max()>10`, `\|A\|.min()` pinned at `A_floor` for >5% of heads, or grad-norm > 3× the trailing 500-step median | keep `gradient_clip_val=0.5` (load-bearing); `dt_limit=(0,1)`; **skip-step-on-spike callback** next to `signal_callbacks.py` — that alone would have saved the 24749 run |
| **FM3** | **bf16 in the scan.** The fp32 force-cast is documented as motivated by the division — which SSD removes. Tempting to flip | rel-err vs the fp64 oracle | **Do not flip.** SSD still exponentiates a cumsum spanning `exp(0)` to underflow within one chunk; bf16 has 8 mantissa bits. Keep fp32 for `dt`, `A`, `Θ`, segsum/decay; gate bf16 for the two big matmuls behind a flag defaulting **off**. `Mamba3Block` must replicate the cast explicitly or silently inherit bf16 from autocast. Memory is not the reason to want it — SSD already cuts scan activations ~37× |
| **FM4** | **RoPE angle accumulation.** At Δ≈0.7, `Θ_512 ≈ 358 rad` (57 turns); naive fp32 cumsum error ~1.1e-2 rad | log `Θ.abs().max()`, alarm > 1e3 rad | fp64 accumulation (~2 MB), per-segment reset, `remainder(·,2π)`, `θ_max·tanh` with near-zero init |
| **FM5** | **Silent config drop — you train Mamba-2 for three days believing it is Mamba-3.** Highest expected cost. It already happened once here (`test_willi_parity.py:1483-1514` exists because `norm_topology` was silently dropped). 25 `HybridConfig(...)` sites across 18 files; `hybrid_block.py:71-100` silently drops unknown kwargs; Hydra strict-struct rejects undeclared CLI overrides | the M2-I arch fingerprint at step 0 of every log | fingerprint + `dataclasses.fields` pass-through (M2-F) + whitelist **raises** on prefixed unknowns + 4-type Gate 6 |
| **FM6** | **The screen may be underpowered by construction.** PubMed abstracts ≈250 tokens with doc resets; trapezoid/RoPE target long-range behaviour. The effect may sit under seed noise | M7-A's 2-seed control | measure the noise floor first; paired Δ log-loss; MQAR + late-position PPL slice; pre-registered decision rule |
| **FM7** | **Recipe drift across arms** — SSD makes `GRAD_CKPT=false` viable (1.3-1.5×), and WSD never enters decay if `max_steps` isn't set to the screen length | arm wall-clocks differing without explanation | hold `GRAD_CKPT=true` across the screen; harvest only for the winner; set `trainer.max_steps` = screen length |

---

## State-tracking contract (`mamba3_state.json`)

1. Session start: read `MAMBA3_PLAN.md` + `mamba3_state.json` (pointed to from `CLAUDE.md`).
2. Resume at `current_phase`; the checkboxes in `MAMBA3_PLAN.md` are ground truth.
3. After **every** meaningful change (test written, phase gated, job submitted, job finished, eval scored):
   tick the checkbox, update `last_updated` (ISO 8601), append a one-line `notes` entry, and record the
   evidence (job id, log path, metric) under `phases[<id>].evidence`. Use the helper rather than editing
   two files by hand — that is how a plan and its state drift apart:

   ```bash
   python scripts/mamba3_state.py tick M3-A M3-B --note "..." --evidence key=value
   python scripts/mamba3_state.py phase M4_complex_state --status "..."
   python scripts/mamba3_state.py readme      # refresh README's status line + progress table
   python scripts/mamba3_state.py show [M3]   # progress at a glance
   ```

   Run `readme` at the end of every phase. The repo already carries one README that went stale enough to
   assert the Mamba path used a Triton kernel it has never called; regenerating the table is what stops
   this one going the same way.
4. Never re-run a checkpoint-producing phase (M7-B, M8-A/B/C) without first reading its log and logging a verdict.
5. If `mamba3_state.json` is lost, regenerate it from the checkboxes.

---

## Unresolved questions

None. All resolved:
- Merge — **never without an explicit instruction from the user** (decision 1).
- mamba/mLSTM ratio — **re-opened as gated phase M10** if M8 clears (decision 8).
- `d_state` 128, `scan_impl` legacy→exact at M9, py3.9 dropped, dead code deleted (decisions 6, 7, 9).

Budget note: M10 adds ~5 screen arms ≈ 37 GPU-h, plus an optional full pipeline (~133 GPU-h). It is gated on
M8 succeeding, so it is not in the ≈230 GPU-h baseline above.
