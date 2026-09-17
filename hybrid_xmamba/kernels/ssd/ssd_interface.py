"""Chunked SSD scan -- the Mamba-2/3 state space dual (MAMBA3_PLAN.md M2-B).

Why this exists rather than a patch to `selective_scan`. Mamba-1 parameterizes `A` as
`(d_inner, dstate)`, one decay per channel *and* state. Computing the intra-chunk term exactly
needs the pairwise decay `exp(sum_{i=s+1..t} dA_i)`, which under that parameterization is a
`(chunk, chunk, d_inner, dstate)` tensor -- 19.3 GB at chunk 64 / batch 48. Mamba-2/3 make `A`
a scalar per head, so the same tensor is `(chunk, chunk)` per head: 19 MB, and shaped as a
matmul that runs on tensor cores. Correctness is affordable here and is not there. That, rather
than any reported quality gain, is the load-bearing reason to adopt SSD.

The algorithm is the standard chunked decomposition: within a chunk the output is
`(L . C B^T) X` with `L` the lower-triangular decay mask, and across chunks a linear recurrence
carries the state. Nothing is divided by a decay, so there is no clamp and no annihilation
(contrast `selective_scan_parallel`; see tests/test_mamba3_numerics.py).

Shapes follow `ssd_reference`:

    x   (batch, seqlen, nheads, headdim)
    dt  (batch, seqlen, nheads)            sets the decay exp(dt * A)
    A   (nheads,)                          strictly negative
    B   (batch, seqlen, ngroups, dstate)
    C   (batch, seqlen, ngroups, dstate)
    D   (nheads,) or None
"""

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

Term = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]  # (coefficient, B, x)


def segsum(v: torch.Tensor) -> torch.Tensor:
    """Lower-triangular segment sums: `out[..., t, s] = sum_{i=s+1}^{t} v_i` for `s <= t`.

    Entries above the diagonal are `-inf`, so `exp(segsum(v))` is the causal decay mask directly.
    The diagonal is an empty sum, hence `exp(0) = 1`: a token always contributes to its own state
    at full weight, whatever the decay is. That is precisely the property the divide-and-clamp
    formulation loses.
    """
    T = v.size(-1)
    vv = v.unsqueeze(-1).expand(*v.shape, T)                    # [..., d, e] = v[d]
    lower = torch.ones(T, T, dtype=torch.bool, device=v.device).tril(-1)
    vv = vv.masked_fill(~lower, 0)
    out = vv.cumsum(dim=-2)
    causal = torch.ones(T, T, dtype=torch.bool, device=v.device).tril(0)
    return out.masked_fill(~causal, -float("inf"))


def _segment_ids(cu_seqlens: Optional[torch.Tensor], batch: int, seqlen: int,
                 device: torch.device) -> torch.Tensor:
    """Monotone segment index per position; positions sharing a value share a document.

    Document resets are applied as *boolean* masks rather than by folding a large negative number
    into the log-decay. The sentinel trick is tempting and wrong: adding -1e30 to a cumulative sum
    annihilates the finite part (next to 1e30, float64 cannot represent anything below ~1e14), so
    `A_cum[end] - A_cum[t]` comes back as exactly 0 and every within-document decay is destroyed
    along with the cross-document one. Masks are exact and cost a few bool tensors.
    """
    if cu_seqlens is None:
        return torch.zeros(batch, seqlen, dtype=torch.long, device=device)
    starts = torch.zeros_like(cu_seqlens, dtype=torch.long)
    starts[:, 1:] = (cu_seqlens[:, 1:] != cu_seqlens[:, :-1]).long()
    return starts.cumsum(dim=1)


def ssd_chunked_scan(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    chunk_size: int = 64,
    cu_seqlens: Optional[torch.Tensor] = None,
    coeff: Optional[torch.Tensor] = None,
    extra_terms: Optional[Sequence[Term]] = None,
) -> torch.Tensor:
    """Chunk-parallel SSD scan.

        h_t = exp(dt_t A) h_{t-1} + sum_terms coeff_t (B_t x_t^T)
        y_t = C_t^T h_t + D x_t

    Args:
        chunk_size: intra-chunk block. Depth is `seqlen / chunk_size`; the mask is
            `(chunk_size, chunk_size)` per head.
        cu_seqlens: optional (batch, seqlen) segment ids; state resets where the id changes.
        coeff: state-input coefficient, (batch, seqlen, nheads). Defaults to `dt`, which is
            Mamba-2's exponential-Euler rule. Kept separate from `dt` because `dt` also sets the
            decay, and the exponential-trapezoidal rule needs to scale the input by
            `gamma = lambda * dt` while leaving the decay at `exp(dt A)`.
        extra_terms: additional `(coefficient, B, x)` triples summed into the state. The
            recurrence is linear in its input, so each is another pass over the *same* decay
            mask -- which is exactly how the trapezoidal rule's `beta * B_{t-1} x_{t-1}` term is
            added (MAMBA3_PLAN.md M3) without a second mask or a second scan. Unused at M2.

    Returns:
        (batch, seqlen, nheads, headdim)
    """
    batch, seqlen, nheads, headdim = x.shape
    ngroups, dstate = B.shape[-2], B.shape[-1]
    if nheads % ngroups != 0:
        raise ValueError(f"nheads ({nheads}) must be divisible by ngroups ({ngroups})")
    rep = nheads // ngroups
    chunk_size = max(1, min(chunk_size, seqlen))

    terms: List[Term] = [(dt if coeff is None else coeff, B, x)] + list(extra_terms or [])
    dA = dt * A                                                  # (batch, seqlen, nheads)
    seg = _segment_ids(cu_seqlens, batch, seqlen, x.device)

    pad = (chunk_size - seqlen % chunk_size) % chunk_size
    if pad:
        dA = F.pad(dA, (0, 0, 0, pad))
        C = F.pad(C, (0, 0, 0, 0, 0, pad))
        # Padding gets its own segment id so it can never mix with real tokens.
        seg = F.pad(seg, (0, pad), value=int(seg.max().item()) + 1)
        terms = [
            (F.pad(co, (0, 0, 0, pad)),
             F.pad(bb, (0, 0, 0, 0, 0, pad)),
             F.pad(xx, (0, 0, 0, 0, 0, pad)))
            for co, bb, xx in terms
        ]
    padded = dA.shape[1]
    nc = padded // chunk_size

    dA_c = dA.reshape(batch, nc, chunk_size, nheads)
    A_cum = dA_c.cumsum(dim=2)                                   # decay from the chunk's start
    seg_c = seg.reshape(batch, nc, chunk_size)
    C_c = C.reshape(batch, nc, chunk_size, ngroups, dstate)

    # Three boolean masks encode every document reset:
    #   same     -- token s may influence token t at all
    #   carry_ok -- the state carried in from earlier chunks is still valid at t
    #   end_ok   -- token t's contribution survives to the chunk boundary
    same = (seg_c.unsqueeze(-1) == seg_c.unsqueeze(-2))          # (batch, nc, cs, cs)
    prev_seg = torch.full((batch, nc), -1, dtype=seg_c.dtype, device=seg_c.device)
    if nc > 1:
        prev_seg[:, 1:] = seg_c[:, :-1, -1]
    carry_ok = (seg_c == prev_seg.unsqueeze(-1))                 # (batch, nc, cs)
    end_ok = (seg_c == seg_c[:, :, -1:])                         # (batch, nc, cs)

    chunked: List[Term] = [
        (co.reshape(batch, nc, chunk_size, nheads),
         bb.reshape(batch, nc, chunk_size, ngroups, dstate),
         xx.reshape(batch, nc, chunk_size, nheads, headdim))
        for co, bb, xx in terms
    ]

    mask = torch.exp(segsum(dA_c.permute(0, 1, 3, 2)))           # (batch, nc, nheads, cs, cs)
    mask = mask * same.unsqueeze(2)

    # --- intra-chunk: y[t] = sum_{s<=t} mask[t,s] <C_t, B_s> coeff_s x_s -----------------------
    y = torch.zeros(batch, nc, chunk_size, nheads, headdim, device=x.device, dtype=x.dtype)
    for co_c, B_c, x_ct in chunked:
        CB = torch.einsum("bctgn,bcsgn->bcgts", C_c, B_c)        # (batch, nc, ngroups, cs, cs)
        CB = CB.repeat_interleave(rep, dim=2)                    # broadcast groups over heads
        y = y + torch.einsum("bchts,bcsh,bcshp->bcthp", mask * CB, co_c, x_ct)

    # --- inter-chunk: carry the state across chunks (nc sequential steps) ----------------------
    state = torch.zeros(batch, nheads, headdim, dstate, device=x.device, dtype=x.dtype)
    C_h = C_c.repeat_interleave(rep, dim=3)                      # (batch, nc, cs, nheads, dstate)
    offsets = []
    for ci in range(nc):
        A_cum_ci = A_cum[:, ci]                                  # (batch, cs, nheads)
        carry_gate = (torch.exp(A_cum_ci) * carry_ok[:, ci].unsqueeze(-1)).unsqueeze(-1)
        offsets.append(torch.einsum("bhpn,bthn->bthp", state, C_h[:, ci]) * carry_gate)

        keep = end_ok[:, ci].unsqueeze(-1)                       # (batch, cs, 1)
        decay_to_end = torch.exp(A_cum_ci[:, -1:] - A_cum_ci) * keep
        state = (
            torch.exp(A_cum_ci[:, -1]) * carry_ok[:, ci, -1].unsqueeze(-1)
        ).unsqueeze(-1).unsqueeze(-1) * state
        for co_c, B_c, x_ct in chunked:
            state = state + torch.einsum(
                "bth,bth,bthp,bthn->bhpn",
                decay_to_end, co_c[:, ci], x_ct[:, ci],
                B_c[:, ci].repeat_interleave(rep, dim=2),
            )

    y = (y + torch.stack(offsets, dim=1)).reshape(batch, padded, nheads, headdim)
    if D is not None:
        y = y + D.view(1, 1, -1, 1) * (F.pad(x, (0, 0, 0, 0, 0, pad)) if pad else x)
    return y[:, :seqlen] if pad else y
