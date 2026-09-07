"""Selective-scan correctness bound (H100_SCALING_PLAN.md Phase 14C-1).

WHY THIS FILE EXISTS
--------------------
`scan_interface.selective_scan_parallel` computes the intra-chunk term as

    h_intra[t] = A_cum[t] * cumsum_s( Bx[s] / A_cum[s].clamp(min=1e-8) )

which is algebraically exact (the ratio A_cum[t]/A_cum[s] is the correct
transition weight from s to t) but numerically wrong once `A_cum[s]` underflows
below the clamp: the divisor is then 1e-8 instead of the true, smaller value, so
`Bx[s]` is under-weighted -- and at `s == t` the token's own contribution, which
should enter with weight exactly 1, enters with weight `A_cum[t]/1e-8 ~ 0` and is
annihilated.

This was audited on 2026-08-16 (`MAMBA3_INTEGRATION_PLAN.md` finding F3) but the
audit never landed as a runnable test, so nothing guarded against it getting
worse and no number was citable from the writeup. A supervisor review on
2026-09-07 asked for exactly this: "either fix it, or add an explicit error-bound
test and report the max deviation."

WHAT THIS FILE ASSERTS
----------------------
1. In the regime where the clamp never fires (small dt), the chunked scan IS
   exact -- a genuine correctness assertion, not a rubber stamp.
2. Shrinking the chunk restores exactness at fixed dt (audit F4), which is the
   mechanism behind the 14C-3 end-to-end probe.
3. The error in the defective regime does not GROW. The bounds below are
   regression guards, deliberately one-sided: if someone repairs the operator the
   measured error drops and these still pass. They fail only if the deviation
   gets worse, or if the exact regimes stop being exact.

Deliberately NOT asserted: a `<= 1e-6` bound over the whole grid. The operator is
known-defective; a test asserting it is correct would simply fail, and deleting
the defective rows would hide the finding. The point is to pin and publish the
envelope.

Run `python tests/test_scan_correctness.py --emit` to regenerate
`analysis/scan_error_bound.md`.
"""

import pytest
import torch

from hybrid_xmamba.kernels.selective_scan.scan_interface import selective_scan_parallel


# The Delta grid from the 2026-08-16 audit, plus the two regimes that bracket it.
# 0.705 is the measured mean Delta of the live (uninitialized) 150M model;
# 0.8229 is the measured mean under `norm_topology=hybrid`, the canonical config.
AUDIT_DELTAS = (1e-3, 1e-2, 0.1, 0.3, 0.705, 1.0)
AUDIT_CHUNKS = (4, 8, 16, 32, 64)

# Documented envelope, measured by this file on CPU (see analysis/scan_error_bound.md).
# One-sided regression guards with headroom -- see module docstring.
#
# EXACT_TOL is set for float32, not float64: the live path casts to .float() before
# calling selective_scan_parallel, so "exact" here means "exact to float32 rounding".
# Measured error in the clamp-free regime is 3e-8..8e-8, i.e. ~fp32 epsilon (1.2e-7);
# 1e-5 sits two orders above that noise floor and three-to-four orders BELOW the
# smallest defect signal on the grid (5.6e-2), so the two regimes cannot be confused.
EXACT_TOL = 1e-5
DEFECTIVE_CEILING = 2.0     # the defect must not exceed this; it is ~1.08 today


def reference_selective_scan_sequential(x, dt, A, B, C, D):
    """Exact selective scan: the recurrence the Mamba block is SPECIFIED to compute.

    Sequential, float64, no chunking, no division, no clamp -- so there is nothing
    in here to be wrong in the same way the fast path is wrong.

        A_disc[t] = exp(dt[t] * A)
        Bx[t]     = dt[t] * B[t] * x[t]
        h[t]      = A_disc[t] * h[t-1] + Bx[t],     h[-1] = 0
        y[t]      = sum_n C[t,n] * h[t,:,n] + D * x[t]

    Args:
        x:  (B, L, D)   dt: (B, L, D)   A: (D, N)
        B:  (B, L, N)   C:  (B, L, N)   D: (D,)

    Returns:
        (B, L, D) float64
    """
    x, dt = x.double(), dt.double()
    A, B, C, D = A.double(), B.double(), C.double(), D.double()

    batch, seq_len, dim = x.shape
    state_size = A.shape[1]

    h = torch.zeros(batch, dim, state_size, dtype=torch.float64, device=x.device)
    outputs = []
    for t in range(seq_len):
        # (B, D, N): exp(dt[t] (B,D) x A (D,N))
        a_disc = torch.exp(dt[:, t].unsqueeze(-1) * A.unsqueeze(0))
        # (B, D, N): dt[t] (B,D) * B[t] (B,N) * x[t] (B,D)
        bx = dt[:, t].unsqueeze(-1) * B[:, t].unsqueeze(-2) * x[:, t].unsqueeze(-1)
        h = a_disc * h + bx
        y_t = torch.einsum("bdn,bn->bd", h, C[:, t]) + D.unsqueeze(0) * x[:, t]
        outputs.append(y_t)

    return torch.stack(outputs, dim=1)


def _make_inputs(delta, batch=2, seq_len=128, dim=8, state_size=16, seed=0):
    """Fixed-Delta inputs matching the audit's protocol (A spanning -1..-state_size)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, seq_len, dim, generator=g)
    dt = torch.full((batch, seq_len, dim), float(delta))
    # S4D-real style: A[d, n] = -(n + 1), i.e. -1 .. -state_size for every channel.
    A = -torch.arange(1, state_size + 1, dtype=torch.float32).repeat(dim, 1)
    B = torch.randn(batch, seq_len, state_size, generator=g)
    C = torch.randn(batch, seq_len, state_size, generator=g)
    D = torch.randn(dim, generator=g)
    return x, dt, A, B, C, D


def relative_max_error(y_test, y_ref):
    """max|y_test - y_ref| normalised by the reference's own scale.

    Normalising by max|y_ref| rather than per-element keeps the metric finite and
    comparable across Delta regimes (per-element relative error explodes wherever
    the reference happens to pass near zero, which says nothing about the operator).
    """
    y_test, y_ref = y_test.double(), y_ref.double()
    denom = y_ref.abs().max().clamp(min=1e-30)
    return (y_test - y_ref).abs().max().item() / denom.item()


def measure(delta, chunk_size, **kw):
    """Relative max error of the live fp32 chunked scan vs the float64 reference."""
    x, dt, A, B, C, D = _make_inputs(delta, **kw)
    y_ref = reference_selective_scan_sequential(x, dt, A, B, C, D)
    # fp32 is what the live path uses: selective_scan() casts to .float() before
    # calling selective_scan_parallel (the 2026-07 stability guard).
    y_fast = selective_scan_parallel(x, dt, A, B, C, D, chunk_size=chunk_size)
    return relative_max_error(y_fast, y_ref)


def clamp_fire_fraction(delta, chunk_size, dim=8, state_size=16):
    """Fraction of (position, channel, state) entries where the 1e-8 clamp fires.

    This is the mechanism itself, measured directly rather than inferred from the
    output error: A_cum = exp(cumsum(dt * A)) within a chunk, and any entry below
    1e-8 gets its divisor replaced by 1e-8.
    """
    A = -torch.arange(1, state_size + 1, dtype=torch.float32).repeat(dim, 1)
    dt_col = torch.full((chunk_size, dim), float(delta))
    log_a_cum = torch.cumsum(dt_col.unsqueeze(-1) * A.unsqueeze(0), dim=0)
    return (torch.exp(log_a_cum) < 1e-8).double().mean().item()


class TestScanExactWhereClampNeverFires:
    """The chunked scan IS exact when A_cum stays above the clamp -- a real assertion."""

    @pytest.mark.parametrize("delta", [1e-3, 1e-2])
    @pytest.mark.parametrize("chunk_size", list(AUDIT_CHUNKS))
    def test_small_delta_is_exact_at_every_chunk_size(self, delta, chunk_size):
        assert clamp_fire_fraction(delta, chunk_size) == 0.0, "precondition: clamp must not fire"
        err = measure(delta, chunk_size)
        assert err < EXACT_TOL, (
            "chunked scan diverged from the exact recurrence in a regime where the "
            "clamp never fires (delta=%g, chunk=%d): rel max err %.3e >= %.0e. This is "
            "a genuine correctness regression, NOT the known clamp defect."
            % (delta, chunk_size, err, EXACT_TOL)
        )

    def test_shrinking_the_chunk_restores_exactness(self):
        """Audit F4's mechanism: the error is driven by intra-chunk dynamic range.

        At delta=0.1 a 64-wide chunk decays A_cum far below 1e-8, but a 4-wide one
        does not. This is what the 14C-3 end-to-end probe exploits -- chunk_size is
        an inference-time knob, so it can be changed without touching any weights.
        """
        err_wide = measure(0.1, chunk_size=64)
        err_narrow = measure(0.1, chunk_size=4)
        assert clamp_fire_fraction(0.1, 64) > 0.0
        assert clamp_fire_fraction(0.1, 4) == 0.0
        assert err_narrow < EXACT_TOL
        assert err_wide > err_narrow


class TestScanErrorBoundInDefectiveRegime:
    """Regression guards on the known defect. One-sided: a repair still passes."""

    @pytest.mark.parametrize("delta", [0.1, 0.3, 0.705, 1.0])
    def test_error_does_not_exceed_documented_ceiling(self, delta):
        err = measure(delta, chunk_size=64)
        assert err <= DEFECTIVE_CEILING, (
            "selective-scan deviation at delta=%g, chunk=64 is %.4f, above the "
            "documented ceiling %.1f. The operator got WORSE -- re-run "
            "`python tests/test_scan_correctness.py --emit` and update "
            "analysis/scan_error_bound.md before changing this bound."
            % (delta, err, DEFECTIVE_CEILING)
        )

    def test_defect_is_monotone_in_delta_at_fixed_chunk(self):
        """Larger Delta decays A_cum faster, so more of the chunk hits the clamp."""
        errs = [measure(d, chunk_size=64) for d in (0.1, 0.3, 1.0)]
        assert errs == sorted(errs), (
            "error vs Delta is no longer monotone (%s) -- the failure mechanism is "
            "not the one this test documents; re-audit before trusting the bound."
            % errs
        )

    def test_clamp_actually_fires_at_the_live_models_delta(self):
        """Guards the premise. The live 150M runs at Delta ~0.705 (audit F1)."""
        frac = clamp_fire_fraction(0.705, chunk_size=64)
        assert frac > 0.5, (
            "the 1e-8 clamp no longer fires at the live model's measured Delta "
            "(fraction=%.3f) -- if the operator or its init changed, this whole "
            "bound needs re-deriving." % frac
        )


def _emit_report(path="analysis/scan_error_bound.md"):
    """Regenerate the citable error-bound table (Phase 14C-1 deliverable)."""
    import datetime
    import io

    rows = []
    for delta in AUDIT_DELTAS:
        rows.append((delta, [measure(delta, c) for c in AUDIT_CHUNKS],
                     [clamp_fire_fraction(delta, c) for c in AUDIT_CHUNKS]))

    out = io.StringIO()
    out.write("# Selective-scan error bound\n\n")
    out.write("Generated by `python tests/test_scan_correctness.py --emit` on %s.\n"
              % datetime.date.today().isoformat())
    out.write("Guarded by `tests/test_scan_correctness.py` (CPU, runs in "
              "`validate_for_willi.sh`).\n\n")
    out.write("**What is measured.** The live chunk-parallel scan "
              "(`selective_scan_parallel`, fp32 -- the dtype the live path uses) against an "
              "exact float64 sequential implementation of the recurrence the Mamba block is "
              "specified to compute. Metric: `max|y_fast - y_ref| / max|y_ref|`. Inputs: "
              "batch=2, L=128, d_inner=8, d_state=16, constant Delta, A spanning -1..-16.\n\n")
    out.write("**The defect.** The intra-chunk term divides by "
              "`A_cum[s].clamp(min=1e-8)`. Where `A_cum[s]` underflows past the clamp the "
              "divisor is too large, so `Bx[s]` is under-weighted -- and at `s == t` the "
              "token's own contribution, which should enter with weight exactly 1, enters "
              "with weight `A_cum[t]/1e-8 ~ 0` and is annihilated.\n\n")

    out.write("## Relative max error vs the exact recurrence\n\n")
    out.write("| Delta | " + " | ".join("chunk=%d" % c for c in AUDIT_CHUNKS) + " |\n")
    out.write("|---" * (len(AUDIT_CHUNKS) + 1) + "|\n")
    for delta, errs, _ in rows:
        out.write("| %g | " % delta + " | ".join("%.3e" % e for e in errs) + " |\n")

    out.write("\n## Fraction of chunk entries where the 1e-8 clamp fires\n\n")
    out.write("| Delta | " + " | ".join("chunk=%d" % c for c in AUDIT_CHUNKS) + " |\n")
    out.write("|---" * (len(AUDIT_CHUNKS) + 1) + "|\n")
    for delta, _, fracs in rows:
        out.write("| %g | " % delta + " | ".join("%.1f%%" % (f * 100) for f in fracs) + " |\n")

    live = measure(0.705, 64)
    out.write("\n## Reading this\n\n")
    out.write("- **Delta <= 0.01: exact at every chunk size.** The clamp never fires, and the "
              "chunked scan reproduces the sequential recurrence to float precision. The "
              "chunking itself is correct; only the clamped division is not.\n")
    out.write("- **The live 150M model runs at Delta ~0.705** (audit F1, `MAMBA3_INTEGRATION_PLAN.md`; "
              "mean 0.8229 under the canonical `norm_topology=hybrid`), i.e. deep inside the "
              "defective regime: **rel max err %.4f at the production chunk size of 64**.\n" % live)
    out.write("- **Shrinking the chunk restores exactness** at fixed Delta, because the error is "
              "driven by intra-chunk dynamic range. `chunk_size` is an inference-time knob, "
              "which is what Phase 14C-3 exploits to bound the end-to-end effect on the "
              "reported metrics without retraining anything.\n")
    out.write("\n## What this does and does not license\n\n")
    out.write("The numbers above bound the *operator's* deviation from its specification. They do "
              "**not** invalidate any reported result: training and evaluation both used this same "
              "operator throughout, so every published number is a valid measurement of the system "
              "as built. The correct claim is narrower -- **the block did not compute the Mamba "
              "recurrence it was specified to compute** -- and the end-to-end consequence for the "
              "headline metrics is measured separately in Phase 14C-3, not argued from this table.\n")
    out.write("\nA full repair is **not** free: audit F4 shows chunk-shrinking is only exact once "
              "Delta is properly initialised (at the current uninitialised Delta ~0.705 even chunk=2 "
              "fails), and the Delta init is a training-time change. That repair is owned by "
              "`MAMBA3_INTEGRATION_PLAN.md` (M1+M2), which Phase 14 deliberately does not activate.\n")

    with open(path, "w") as fh:
        fh.write(out.getvalue())
    return path, rows


if __name__ == "__main__":
    import sys

    if "--emit" in sys.argv:
        path, rows = _emit_report()
        print("wrote %s" % path)
        for delta, errs, fracs in rows:
            print("  delta=%-6g " % delta + "  ".join(
                "c%d: %.2e (%.0f%% clamped)" % (c, e, f * 100)
                for c, e, f in zip(AUDIT_CHUNKS, errs, fracs)))
    else:
        print("usage: python tests/test_scan_correctness.py --emit")


class TestExactScanToggle:
    """Phase 14C-3's measurement path: an opt-in exact operator, OFF by default."""

    def test_runtime_reference_agrees_with_this_files_independent_implementation(self):
        """Two independently written exact implementations must agree.

        `scan_interface.selective_scan_sequential_reference` is the runtime one (used
        by HYBRID_EXACT_SCAN=1); `reference_selective_scan_sequential` above is this
        file's own. Keeping both and asserting they match catches drift in either.
        """
        from hybrid_xmamba.kernels.selective_scan.scan_interface import (
            selective_scan_sequential_reference,
        )

        x, dt, A, B, C, D = _make_inputs(0.705)
        mine = reference_selective_scan_sequential(x, dt, A, B, C, D)
        runtime = selective_scan_sequential_reference(x, dt, A, B, C, D)
        assert relative_max_error(runtime, mine) < EXACT_TOL

    def test_toggle_is_off_by_default_so_the_operator_freeze_holds(self, monkeypatch):
        """Default behaviour must stay byte-identical to every published run."""
        from hybrid_xmamba.kernels.selective_scan import scan_interface

        monkeypatch.delenv("HYBRID_EXACT_SCAN", raising=False)
        x, dt, A, B, C, D = _make_inputs(0.705)
        default = scan_interface.selective_scan(x, dt, A, B, C, D)
        # selective_scan picks chunk_size from seq_len via its own size ladder:
        # L<=128 -> 32. Mirror that here rather than hardcoding 64.
        chunked = scan_interface.selective_scan_parallel(
            x.float(), dt.float(), A.float(), B.float(), C.float(), D.float(),
            chunk_size=32,
        )
        assert torch.equal(default, chunked), (
            "the default selective_scan path is no longer the plain chunked scan -- "
            "the Phase 14A operator freeze has been broken"
        )

    def test_toggle_on_selects_the_exact_operator(self, monkeypatch):
        from hybrid_xmamba.kernels.selective_scan import scan_interface

        monkeypatch.setenv("HYBRID_EXACT_SCAN", "1")
        x, dt, A, B, C, D = _make_inputs(0.705)
        toggled = scan_interface.selective_scan(x, dt, A, B, C, D)
        exact = reference_selective_scan_sequential(x, dt, A, B, C, D)
        assert relative_max_error(toggled, exact) < EXACT_TOL

        # And it must differ measurably from the default -- otherwise the 14C-3
        # probe would be incapable of detecting anything.
        monkeypatch.delenv("HYBRID_EXACT_SCAN")
        default = scan_interface.selective_scan(x, dt, A, B, C, D)
        assert relative_max_error(default, exact) > 0.1
