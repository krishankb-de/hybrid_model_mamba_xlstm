"""Numerical-correctness tests for the recurrent operators (MAMBA3_PLAN.md, phase M1).

These pin a defect, they are not regression tests for working code. `tests/test_kernels.py`
asserts only shape / no-NaN / no-Inf on the selective scan, and gates the whole class behind
`torch.cuda.is_available()`, so it never runs in CI -- which is exactly how the defect below
survived. Everything here is CPU-collected and unconditionally run.

The defect (MAMBA3_PLAN.md, Context 1). Both chunked scans compute

    h_intra[t] = A_cum[t] * cumsum(Bx / A_cum.clamp(min=1e-8))[t]

Where `A_cum[s]` falls under 1e-8 the clamp pins the denominator, so the ratio
`A_cum[t] / max(A_cum[s], 1e-8)` collapses to ~0 even when the true ratio is O(1) for `s` near
`t`. The token's own contribution to the state is annihilated, not merely perturbed.
"""

from typing import List

import pytest
import torch

from hybrid_xmamba.kernels.selective_scan.scan_interface import (
    selective_scan,
    selective_scan_exact,
    selective_scan_parallel,
)
from hybrid_xmamba.layers.mamba_block import MambaBlock

# Deltas span the reference Mamba init range logU[1e-3, 1e-1] and the range this repo actually
# operates in: Delta ~ 0.70 (pre_rms) / 0.82 (hybrid, canonical), measured in MAMBA3_PLAN.md.
DELTAS: List[float] = [1e-3, 1e-2, 1e-1, 0.3, 0.705, 1.0]
CHUNKS: List[int] = [8, 64]
TOL = 1e-6

_FLIPS = "strict=True, so it fails loudly the moment it starts passing (MAMBA3_PLAN.md M1-I)."
_DEFECT_SCAN = f"MAMBA3_PLAN.md M1: divide-and-clamp in the selective scan. Fixed by M1-E/M1-G. {_FLIPS}"
_DEFECT_TFLA = f"MAMBA3_PLAN.md M1: divide-and-clamp in the TFLA intra-chunk term. Fixed by M1-H. {_FLIPS}"
_DEFECT_DELTA = f"MAMBA3_PLAN.md M1: no Mamba dt init, and dt_norm would erase one. Fixed by M1-F. {_FLIPS}"


def _xfail_if(condition: bool, reason: str = _DEFECT_SCAN):
    """Mark a parametrized case as a known defect. Empty list == expected to pass today."""
    return [pytest.mark.xfail(strict=True, reason=reason)] if condition else []


# "legacy" reproduces the pre-2026-09 numerics on purpose (MAMBA3_PLAN.md decision 7), so its
# failures are permanent xfails, not a TODO. "exact" must pass everywhere -- those are the cases
# that make M1 a fix rather than a description.
SCAN_IMPLS = ("legacy", "exact")


def _scan_cases():
    """Measured on HEAD: chunk=8 rescues delta=0.1 but nothing rescues delta>=0.3."""
    for impl in SCAN_IMPLS:
        for delta in DELTAS:
            for chunk in CHUNKS:
                broken = impl == "legacy" and (delta >= 0.3 or (delta >= 0.1 and chunk >= 64))
                yield pytest.param(impl, delta, chunk, marks=_xfail_if(broken))


def _delta_cases():
    for impl in SCAN_IMPLS:
        for delta in DELTAS:
            yield pytest.param(impl, delta, marks=_xfail_if(impl == "legacy" and delta >= 0.1))


def sequential_selective_scan_fp64(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
) -> torch.Tensor:
    """float64 sequential ground truth for the Mamba-1 (S6) recurrence.

    h_t = exp(dt_t * A) . h_{t-1} + (dt_t * B_t) x_t        h: (batch, dim, state)
    y_t = <C_t, h_t> + D * x_t

    Deliberately a naive Python loop: it is the oracle, so it must be obviously correct
    rather than fast.
    """
    x64, dt64 = x.double(), dt.double()
    A64, B64, C64, D64 = A.double(), B.double(), C.double(), D.double()
    batch, seq_len, dim = x64.shape
    state = A64.shape[1]

    h = torch.zeros(batch, dim, state, dtype=torch.float64)
    ys = []
    for t in range(seq_len):
        decay = torch.exp(dt64[:, t].unsqueeze(-1) * A64)              # (batch, dim, state)
        inp = (dt64[:, t].unsqueeze(-1) * B64[:, t].unsqueeze(1)) * x64[:, t].unsqueeze(-1)
        h = decay * h + inp
        ys.append(torch.einsum("bdn,bn->bd", h, C64[:, t]) + D64 * x64[:, t])
    return torch.stack(ys, dim=1)


def rel_max_err(got: torch.Tensor, want: torch.Tensor) -> float:
    return (got.double() - want).abs().max().item() / want.abs().max().item()


def _fixture(delta: float, seq_len: int = 128, dim: int = 8, state: int = 16):
    """Inputs matching the shipped init: A = -[1..N] repeated per channel (mamba_block.py:82)."""
    torch.manual_seed(0)
    A = -torch.arange(1.0, state + 1).repeat(dim, 1)
    return (
        torch.randn(1, seq_len, dim),                    # x
        torch.full((1, seq_len, dim), delta),            # dt
        A,
        torch.randn(1, seq_len, state),                  # B
        torch.randn(1, seq_len, state),                  # C
        torch.ones(dim),                                 # D
    )


@pytest.mark.parametrize("scan_impl,delta,chunk_size", list(_scan_cases()))
def test_selective_scan_parallel_matches_sequential_reference(scan_impl, delta, chunk_size):
    """M1-A/M1-E: the chunked scan must equal the sequential recurrence it claims to compute."""
    x, dt, A, B, C, D = _fixture(delta)
    want = sequential_selective_scan_fp64(x, dt, A, B, C, D)
    impl = selective_scan_parallel if scan_impl == "legacy" else selective_scan_exact
    err = rel_max_err(impl(x, dt, A, B, C, D, chunk_size=chunk_size), want)
    assert err <= TOL, (
        f"scan_impl={scan_impl} delta={delta} chunk={chunk_size}: "
        f"rel-max-err {err:.3e} > {TOL:.0e}"
    )


@pytest.mark.parametrize("scan_impl,delta", list(_delta_cases()))
def test_selective_scan_public_api_matches_sequential_reference(scan_impl, delta):
    """M1-A/M1-E: same claim for the public entry point, at whatever chunk size it picks."""
    x, dt, A, B, C, D = _fixture(delta)
    want = sequential_selective_scan_fp64(x, dt, A, B, C, D)
    err = rel_max_err(selective_scan(x, dt, A, B, C, D, scan_impl=scan_impl), want)
    assert err <= TOL, f"scan_impl={scan_impl} delta={delta}: rel-max-err {err:.3e} > {TOL:.0e}"


@pytest.mark.parametrize("scan_impl,delta", list(_delta_cases()))
def test_mamba_block_slow_forward_matches_sequential_reference(scan_impl, delta):
    """M1-A: `use_fast_path=False` carries an identical copy of the defect.

    This path is not a curiosity -- `scripts/validate_for_willi.sh` builds its Gate 6 model with
    `use_fast_path=False`, so the pre-push harness exercises the buggy branch, not the fast one.
    """
    dim, state = 8, 16
    x, dt, A, B, C, D = _fixture(delta, dim=dim, state=state)
    block = MambaBlock(dim=dim, state_size=state, expand_factor=1, scan_impl=scan_impl).eval()
    with torch.no_grad():
        block.D.copy_(D)
        got = block._slow_forward(x, dt, A, B, C)
    err = rel_max_err(got, sequential_selective_scan_fp64(x, dt, A, B, C, D))
    assert err <= TOL, f"scan_impl={scan_impl} delta={delta}: rel-max-err {err:.3e} > {TOL:.0e}"


# ---------------------------------------------------------------------------
# M1-B: Delta at initialization
# ---------------------------------------------------------------------------
# Reference Mamba draws Delta ~ logU[1e-3, 1e-1] via an inverse-softplus bias init. This repo
# has no such init at all, and `HybridLanguageModel._init_weights` (hybrid_lm.py:138-145) zeroes
# every bias -- including `dt_proj.bias`. Worse, `norm_topology="hybrid"` (the canonical setting)
# RMSNorms Delta before the softplus, rescaling it to unit RMS and discarding any bias offset, so
# adding the reference init without also removing that norm would be a no-op. Both facts are
# pinned here so neither can regress silently.

DELTA_INIT_RANGE = (1e-3, 1.5e-1)


def _delta_at_init(model, seq_len: int = 128, batch: int = 4) -> torch.Tensor:
    """Reproduce a mamba mixer's Delta on realistic input, mirroring mamba_block.forward:117-141."""
    torch.manual_seed(0)
    block = next(layer for layer in model.layers if layer.layer_type == "mamba")
    mixer = block.mixer
    input_ids = torch.randint(0, model.config.vocab_size, (batch, seq_len))
    with torch.no_grad():
        x = block.norm1(model.embeddings(input_ids))
        x_inner, _ = mixer.in_proj(x).chunk(2, dim=-1)
        x_conv = mixer.activation(
            mixer.conv1d(x_inner.transpose(1, 2))[..., :seq_len].transpose(1, 2)
        )
        dt = mixer.x_proj(x_conv)[..., : mixer.dt_rank]
        dt = mixer.dt_proj(dt)
        if mixer.dt_norm is not None:
            dt = mixer.dt_norm(dt)
        return torch.nn.functional.softplus(dt)


seq_cap = 128


def _tiny_model(norm_topology: str, dt_init_strategy: str = "none"):
    from hybrid_xmamba.models.configuration_hybrid import HybridConfig
    from hybrid_xmamba.models.hybrid_lm import HybridLanguageModel

    torch.manual_seed(0)
    return HybridLanguageModel(
        HybridConfig(
            vocab_size=512, dim=128, num_layers=2, layer_pattern=["mamba", "mlstm"],
            state_size=16, head_dim=32, num_heads=4, max_position_embeddings=seq_cap,
            norm_topology=norm_topology, dt_init_strategy=dt_init_strategy,
        )
    )


# The full 3x2 grid. Only two cells land in range, and *which* two is the whole finding: the
# init alone is not enough, because dt_norm rescales Delta to unit RMS and throws the bias away.
#
#   norm_topology  dt_init  Delta mean   in range
#   pre_rms        none     0.6932       no
#   pre_rms        mamba    0.0211       YES
#   hybrid         none     0.7987       no
#   hybrid         mamba    0.3320       no    <-- dt_norm erases the fix
#   hybrid_bc      none     0.6932       no
#   hybrid_bc      mamba    0.0211       YES
_DELTA_GRID = [
    ("pre_rms", "none", True),
    ("pre_rms", "mamba", False),
    ("hybrid", "none", True),
    ("hybrid", "mamba", True),      # Finding 2: pinned as still-broken, deliberately
    ("hybrid_bc", "none", True),
    ("hybrid_bc", "mamba", False),
]


@pytest.mark.parametrize(
    "norm_topology,dt_init_strategy",
    [
        pytest.param(tp, di, marks=_xfail_if(broken, _DEFECT_DELTA))
        for tp, di, broken in _DELTA_GRID
    ],
)
def test_delta_at_init_is_in_mamba_range(norm_topology, dt_init_strategy):
    """M1-B/M1-F: Delta at init must sit in the reference range logU[1e-3, 1e-1].

    The `hybrid` + `mamba` cell is xfailed *on purpose* and must stay that way. It is the pin on
    Finding 2: applying the reference dt init while dt_norm is still active is a no-op (Delta
    lands at 0.33, not 0.02), so a future refactor that "helpfully" enables the init under
    `hybrid` would be silently ineffective. If that cell ever starts passing, strict=True turns
    it into a failure and someone has to explain why.
    """
    lo, hi = DELTA_INIT_RANGE
    delta = _delta_at_init(_tiny_model(norm_topology, dt_init_strategy))
    mean = delta.mean().item()
    assert lo <= mean <= hi, (
        f"norm_topology={norm_topology} dt_init={dt_init_strategy}: Delta mean {mean:.4f} "
        f"outside [{lo}, {hi}]; max {delta.max().item():.4f}."
    )


def test_dt_norm_erases_the_dt_init():
    """M1-F: state Finding 2 as a direct, positive claim rather than only as an xfail."""
    with_norm = _delta_at_init(_tiny_model("hybrid", "mamba")).mean().item()
    without_norm = _delta_at_init(_tiny_model("hybrid_bc", "mamba")).mean().item()
    assert without_norm < 0.05 < with_norm, (
        f"expected dt_norm to inflate Delta (got {with_norm:.4f} with, {without_norm:.4f} "
        "without). If this flips, the hybrid_bc topology is no longer doing its job."
    )
    assert with_norm / without_norm > 5.0, "dt_norm's effect on Delta has shrunk unexpectedly"


def test_dt_proj_bias_is_zeroed_by_model_init():
    """M1-B: pins the *mechanism* -- `_init_weights` zeroes the bias the Mamba init would set.

    Not xfailed: this documents current behaviour. When M1-F adds `dt_init_strategy="mamba"` this
    test keeps guarding the default (`"none"`), so the audit finding cannot silently disappear.
    """
    mixer = next(l for l in _tiny_model("hybrid").layers if l.layer_type == "mamba").mixer
    assert torch.equal(mixer.dt_proj.bias, torch.zeros_like(mixer.dt_proj.bias)), (
        "dt_proj.bias is no longer zeroed at init -- if a dt init was added, update M1-B/M1-F."
    )


# ---------------------------------------------------------------------------
# M1-C: the same defect class in the mLSTM (TFLA) intra-chunk term
# ---------------------------------------------------------------------------
# `tfla_interface.py:93-95` computes `k_weighted = k * i / f_cum.clamp(min=1e-6)`. The
# inter-chunk half (`:149`) already uses the log-space difference form and is correct, so only
# the intra-chunk half is affected -- but the canonical 150M model has 3 mLSTM layers, so
# together with the Mamba defect all 12 layers run a recurrence that is not the specified one.

FORGET_BIASES: List[float] = [0.0, 1.0, 2.0, 3.0]  # 0.0 is the shipped forget_gate_bias_init


def sequential_mlstm_fp64(q, k, v, i_gate, f_gate) -> torch.Tensor:
    """float64 sequential ground truth for the recurrence tfla_forward_parallel approximates.

    Convention read off tfla_interface.py:65-175 -- per-dimension multiplicative gates, matrix
    memory C, normalizer n, and a joint denominator clamped at 1.0 (not abs-clamped):

        C_t[d,e] = f_t[d] C_{t-1}[d,e] + i_t[d] k_t[d] v_t[e]
        n_t[d]   = f_t[d] n_{t-1}[d]   + i_t[d] k_t[d]
        y_t[e]   = (sum_d q_t[d] C_t[d,e]) / max(sum_d q_t[d] n_t[d], 1)

    `f` is clamped at 1e-6 exactly as the chunked path does (`:74`), so the comparison isolates
    the intra-chunk division and not that clamp.
    """
    q, k, v, i_gate = (t.double() for t in (q, k, v, i_gate))
    f = f_gate.double().clamp(min=1e-6)
    batch, heads, seq_len, dim = q.shape

    C = torch.zeros(batch, heads, dim, dim, dtype=torch.float64)
    n = torch.zeros(batch, heads, dim, dtype=torch.float64)
    ys = []
    for t in range(seq_len):
        ki = k[:, :, t] * i_gate[:, :, t]
        C = f[:, :, t].unsqueeze(-1) * C + torch.einsum("bhd,bhe->bhde", ki, v[:, :, t])
        n = f[:, :, t] * n + ki
        num = torch.einsum("bhd,bhde->bhe", q[:, :, t], C)
        den = torch.einsum("bhd,bhd->bh", q[:, :, t], n).unsqueeze(-1).clamp(min=1.0)
        ys.append(num / den)
    return torch.stack(ys, dim=2)


def _tfla_fixture(forget_bias: float, seq_len: int = 128, heads: int = 2, dim: int = 8):
    torch.manual_seed(0)
    shape = (1, heads, seq_len, dim)
    return (
        torch.randn(shape),                                            # q
        torch.randn(shape) / dim ** 0.5,                               # k
        torch.randn(shape),                                            # v
        torch.sigmoid(torch.randn(shape) - 10.0),                      # i_gate (shipped bias -10)
        torch.sigmoid(torch.randn(shape) * 0.5 + forget_bias),         # f_gate
    )


@pytest.mark.parametrize(
    "tfla_impl,forget_bias,chunk_size",
    [
        pytest.param(
            impl, fb, cs,
            marks=_xfail_if(impl == "legacy" and fb <= 1.0 and cs >= 64, _DEFECT_TFLA),
        )
        for impl in ("legacy", "exact")
        for fb in FORGET_BIASES
        for cs in CHUNKS
    ],
)
def test_tfla_intra_chunk_matches_sequential_reference(tfla_impl, forget_bias, chunk_size):
    """M1-C: the error is governed entirely by whether f_cum underflows the 1e-6 clamp.

    Measured rel-max-err against the fp64 oracle (chunk_size=64 is the shipped default):

        forget_bias   chunk=8    chunk=32   chunk=64   chunk=128
        0.0 (shipped) 1.1e-07    5.0e-01    8.8e-01    8.8e-01
        1.0           1.2e-07    1.7e-07    5.7e-01    9.6e-01
        2.0           6.5e-08    1.4e-07    1.8e-07    4.8e-01
        3.0           9.1e-08    9.1e-08    1.3e-07    1.4e-07

    Every entry above 1e-6 is one where min(f_cum) fell below the clamp, and every entry below
    it is one where it did not. So the shipped configuration -- forget_gate_bias_init=0.0 at
    chunk_size=64 -- runs at rel-max-err 0.88. The small-chunk and high-bias cases are the
    control: they show it is the clamp that breaks, not the chunking.
    """
    from hybrid_xmamba.kernels.tfla.tfla_interface import tfla_forward_parallel

    q, k, v, i_gate, f_gate = _tfla_fixture(forget_bias)
    want = sequential_mlstm_fp64(q, k, v, i_gate, f_gate)
    got = tfla_forward_parallel(
        q, k, v, i_gate, f_gate, chunk_size=chunk_size, tfla_impl=tfla_impl
    )
    err = rel_max_err(got, want)
    assert err <= TOL, (
        f"tfla_impl={tfla_impl} forget_bias={forget_bias} chunk={chunk_size}: "
        f"rel-max-err {err:.3e} > {TOL:.0e}"
    )


@pytest.mark.parametrize("forget_bias", FORGET_BIASES)
def test_tfla_clamp_hit_rate_is_documented(forget_bias):
    """M1-C: pins *why* the test above fails, so a fix cannot be mistaken for a chunking change."""
    torch.manual_seed(0)
    f = torch.sigmoid(torch.randn(200_000) * 0.5 + forget_bias)
    f_cum = torch.log(f.clamp(min=1e-6)).view(-1, 64).cumsum(-1).exp()
    hit_rate = (f_cum < 1e-6).double().mean().item()
    expected = {0.0: 0.70, 1.0: 0.36, 2.0: 0.0, 3.0: 0.0}[forget_bias]
    assert abs(hit_rate - expected) < 0.05, (
        f"forget_bias={forget_bias}: clamp hit-rate {hit_rate:.3f}, expected ~{expected}"
    )


# ---------------------------------------------------------------------------
# M1-E / M1-H: guards on the fixed paths themselves
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("forget_bias", [-4.0, -2.0])
@pytest.mark.parametrize("chunk_size", [64, 128])
def test_tfla_exact_survives_extreme_decay(forget_bias, chunk_size):
    """M1-H: the overflow guard. Re-centring alone produced NaN here.

    Re-centring halves the dynamic range but cannot remove it: a chunk whose total log-decay
    exceeds 2 * _EXP_SAFE still overflows fp32 on one side, and because the causal mask is
    applied after the product, an inf meets a 0 and yields NaN. `chunk_size=128` with a
    forget-gate bias near -2 reaches that regime. The guarded fallback scans within the chunk
    instead of factorizing across it, which forms no large exponential at all.
    """
    from hybrid_xmamba.kernels.tfla.tfla_interface import tfla_forward_parallel

    q, k, v, i_gate, f_gate = _tfla_fixture(forget_bias)
    got = tfla_forward_parallel(q, k, v, i_gate, f_gate, chunk_size=chunk_size, tfla_impl="exact")
    assert torch.isfinite(got).all(), "exact TFLA produced NaN/Inf under extreme decay"
    err = rel_max_err(got, sequential_mlstm_fp64(q, k, v, i_gate, f_gate))
    assert err <= TOL, f"forget_bias={forget_bias} chunk={chunk_size}: rel-max-err {err:.3e}"


def test_legacy_scan_path_is_unchanged():
    """M1-I: `scan_impl="legacy"` must still be the original operator, bit for bit.

    Decision 7 in MAMBA3_PLAN.md keeps `legacy` as the default through the screen precisely so
    the A0 control arm and every number published before 2026-09 stay reproducible. If this ever
    drifts, that comparison is void -- so it is asserted, not assumed.
    """
    x, dt, A, B, C, D = _fixture(0.705, seq_len=256)
    # selective_scan picks chunk 64 for 128 < L <= 512; mirror that exactly.
    direct = selective_scan_parallel(
        x.float(), dt.float(), A.float(), B.float(), C.float(), D.float(), chunk_size=64
    )
    assert torch.equal(selective_scan(x, dt, A, B, C, D, scan_impl="legacy"), direct)


@pytest.mark.parametrize("scan_impl", ["legacy", "exact"])
def test_scan_impl_is_threaded_from_config_to_block(scan_impl):
    """M1-E: a lever that silently fails to arrive is the failure mode this project already hit.

    `hybrid_block.py` filters mixer kwargs against a per-type whitelist and silently drops
    anything unrecognized, so a new config field can reach `HybridConfig` and never reach the
    mixer. Assert it actually lands.
    """
    from hybrid_xmamba.models.configuration_hybrid import HybridConfig
    from hybrid_xmamba.models.hybrid_lm import HybridLanguageModel

    model = HybridLanguageModel(
        HybridConfig(
            vocab_size=256, dim=64, num_layers=2, layer_pattern=["mamba", "mlstm"],
            state_size=16, head_dim=32, num_heads=2, max_position_embeddings=64,
            scan_impl=scan_impl, tfla_impl=scan_impl,
        )
    )
    mamba = next(l.mixer for l in model.layers if l.layer_type == "mamba")
    mlstm = next(l.mixer for l in model.layers if l.layer_type == "mlstm")
    assert mamba.scan_impl == scan_impl, "scan_impl did not reach MambaBlock"
    assert mlstm.tfla_impl == scan_impl, "tfla_impl did not reach mLSTMBlock"


def test_hybrid_bc_keeps_bc_norms_but_drops_dt_norm():
    """M1-F: `hybrid_bc` differs from `hybrid` in exactly one thing -- the Delta norm."""
    hybrid = next(l.mixer for l in _tiny_model("hybrid").layers if l.layer_type == "mamba")
    hybrid_bc = next(l.mixer for l in _tiny_model("hybrid_bc").layers if l.layer_type == "mamba")
    assert hybrid.dt_norm is not None and hybrid.B_norm is not None
    assert hybrid_bc.dt_norm is None, "hybrid_bc must not build a Delta norm"
    assert hybrid_bc.B_norm is not None and hybrid_bc.C_norm is not None, (
        "hybrid_bc must keep the B/C norms -- that is the half of HybridNorm Mamba-3 Sec 3.4 keeps"
    )


# ---------------------------------------------------------------------------
# M2-A/M2-B: the SSD scan
# ---------------------------------------------------------------------------
# The point of SSD here is not the paper's quality numbers, it is that correctness becomes
# affordable. Mamba-1's A is (d_inner, dstate), so an exact pairwise decay mask is
# (chunk, chunk, d_inner, dstate) = 19.3 GB at chunk 64 / batch 48. Mamba-2/3's scalar-per-head A
# makes the same mask (chunk, chunk) per head: 19 MB, and shaped as a matmul.

SSD_SHAPES = [
    # (batch, seqlen, nheads, headdim, ngroups, dstate)
    (2, 128, 4, 16, 1, 32),    # ngroups=1: B/C fully shared (Mamba's MVA default)
    (2, 128, 8, 16, 2, 64),
    (1, 100, 4, 8, 4, 16),     # seqlen not a multiple of any chunk size
    (3, 64, 6, 32, 6, 16),     # ngroups == nheads: no sharing at all
]


def _ssd_fixture(shape, seed: int = 0):
    batch, seqlen, nheads, headdim, ngroups, dstate = shape
    torch.manual_seed(seed)
    f64 = torch.float64
    return dict(
        x=torch.randn(batch, seqlen, nheads, headdim, dtype=f64),
        dt=torch.rand(batch, seqlen, nheads, dtype=f64) * 0.1 + 1e-3,
        A=-torch.rand(nheads, dtype=f64) * 8 - 0.1,
        B=torch.randn(batch, seqlen, ngroups, dstate, dtype=f64),
        C=torch.randn(batch, seqlen, ngroups, dstate, dtype=f64),
        D=torch.ones(nheads, dtype=f64),
    )


@pytest.mark.parametrize("chunk_size", [16, 32, 64])
@pytest.mark.parametrize("shape", SSD_SHAPES, ids=lambda s: "x".join(str(v) for v in s))
def test_ssd_chunked_matches_sequential_reference(shape, chunk_size):
    """M2-B: the chunked scan must equal the sequential SSD recurrence exactly."""
    from hybrid_xmamba.kernels.ssd import ssd_chunked_scan, ssd_sequential_reference

    kw = _ssd_fixture(shape)
    want = ssd_sequential_reference(**kw)
    err = rel_max_err(ssd_chunked_scan(chunk_size=chunk_size, **kw), want)
    assert err <= 1e-12, f"shape={shape} chunk={chunk_size}: rel-max-err {err:.3e}"


@pytest.mark.parametrize("boundaries", [(37,), (32,), (20, 55, 80)],
                         ids=["mid-chunk", "on-chunk-edge", "multi-doc"])
@pytest.mark.parametrize("chunk_size", [16, 32, 64])
def test_ssd_document_boundaries_match_reference(boundaries, chunk_size):
    """M2-B/M2-E: state resets at document boundaries, handled inside the mask.

    `mamba_block._forward_segmented` implements this as a Python loop over (row, segment) that
    re-runs the whole block per piece -- dozens of tiny kernel launches per layer per step, and
    the reason `torch.compile` is disabled on that path. SSD needs no loop: three boolean masks
    over the segment ids express every reset, batched.

    The `on-chunk-edge` case matters on its own: a boundary that lands exactly on a chunk start
    is the one an off-by-one in the carry logic would silently pass everything else.
    """
    from hybrid_xmamba.kernels.ssd import ssd_chunked_scan, ssd_sequential_reference

    kw = _ssd_fixture((2, 96, 4, 16, 2, 32))
    ids = torch.zeros(2, 96, dtype=torch.long)
    for k, start in enumerate(boundaries):
        ids[:, start:] = k + 1
    want = ssd_sequential_reference(cu_seqlens=ids, **kw)
    got = ssd_chunked_scan(chunk_size=chunk_size, cu_seqlens=ids, **kw)
    assert rel_max_err(got, want) <= 1e-12


@pytest.mark.parametrize("chunk_size", [16, 64])
def test_ssd_document_isolation_is_bit_exact(chunk_size):
    """M2-E: perturbing document A must leave document B bit-identical, not merely close."""
    from hybrid_xmamba.kernels.ssd import ssd_chunked_scan

    kw = _ssd_fixture((2, 96, 4, 16, 2, 32))
    boundary = 37
    ids = torch.zeros(2, 96, dtype=torch.long)
    ids[:, boundary:] = 1
    ref = ssd_chunked_scan(chunk_size=chunk_size, cu_seqlens=ids, **kw)

    perturbed = dict(kw)
    perturbed["x"] = kw["x"].clone()
    perturbed["x"][:, :boundary] += torch.randn_like(perturbed["x"][:, :boundary]) * 5.0
    out = ssd_chunked_scan(chunk_size=chunk_size, cu_seqlens=ids, **perturbed)

    assert torch.equal(ref[:, boundary:], out[:, boundary:]), "doc B leaked from doc A"
    assert not torch.allclose(ref[:, :boundary], out[:, :boundary]), "doc A should have changed"


def test_ssd_extra_terms_are_linear():
    """M2-B: the `extra_terms` hook the trapezoidal rule (M3) will use.

    The recurrence is linear in its state-input, so splitting one term into two halves must
    reproduce the original bit-for-bit-ish. Verifying that now means M3 adds a coefficient, not
    a new scan.
    """
    from hybrid_xmamba.kernels.ssd import ssd_chunked_scan, ssd_sequential_reference

    kw = _ssd_fixture((2, 96, 4, 16, 2, 32))
    want = ssd_sequential_reference(**kw)
    half = kw["dt"] * 0.5
    got = ssd_chunked_scan(
        chunk_size=32, coeff=half, extra_terms=[(half, kw["B"], kw["x"])],
        **{k: v for k, v in kw.items() if k != "coeff"},
    )
    assert rel_max_err(got, want) <= 1e-12


def test_ssd_step_matches_the_chunked_scan():
    """M2-A: the decode step and the training scan are the same recurrence.

    `ssd_step` is what M6's O(1) decode cache will call. Pinning it against the chunked scan now
    means a divergence shows up as a test failure rather than as a silent inference-only bug.
    """
    from hybrid_xmamba.kernels.ssd import ssd_chunked_scan, ssd_step

    kw = _ssd_fixture((2, 48, 4, 16, 2, 32))
    batch, seqlen, nheads, headdim = kw["x"].shape
    state = torch.zeros(batch, nheads, headdim, kw["B"].shape[-1], dtype=torch.float64)
    ys = []
    for t in range(seqlen):
        y_t, state = ssd_step(
            kw["x"][:, t], kw["dt"][:, t], kw["A"], kw["B"][:, t], kw["C"][:, t], state, kw["D"]
        )
        ys.append(y_t)
    assert rel_max_err(torch.stack(ys, dim=1), ssd_chunked_scan(chunk_size=16, **kw)) <= 1e-12
