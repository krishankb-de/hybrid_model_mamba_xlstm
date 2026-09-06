"""Sequential reference for the SSD recurrence, and the single-token decode step.

These are one thing wearing two hats, which is why they live together (MAMBA3_PLAN.md M2-A):

* `ssd_sequential_reference` is the oracle every chunked-scan test is checked against. It runs in
  float64 by default and is a plain Python loop, because an oracle has to be obviously correct
  rather than fast.
* `ssd_step` advances that same recurrence by exactly one token given a carried state. That is
  the decode path (M6), so writing the oracle costs the decode kernel nothing extra -- and any
  divergence between them is a test failure rather than a silent inference bug.

Shape conventions, shared with `ssd_interface` and matching Mamba-2/3:

    x   (batch, seqlen, nheads, headdim)     SSM input, one vector per head
    dt  (batch, seqlen, nheads)              per-head step size, already positive
    A   (nheads,)                            per-head state-transition, strictly negative
    B   (batch, seqlen, ngroups, dstate)     state-input projection, shared across the heads
    C   (batch, seqlen, ngroups, dstate)     state-output projection, shared across the heads
    D   (nheads,) or None                    skip connection
    state (batch, nheads, headdim, dstate)   the recurrent state

`ngroups` is Mamba's multi-value-attention structure: B and C are shared by `nheads // ngroups`
heads each, which is what keeps an 8x larger state nearly free in parameters.
"""

from typing import List, Optional, Sequence, Tuple

import torch


def _heads_per_group(nheads: int, ngroups: int) -> int:
    if nheads % ngroups != 0:
        raise ValueError(f"nheads ({nheads}) must be divisible by ngroups ({ngroups})")
    return nheads // ngroups


def ssd_step(
    x_t: torch.Tensor,
    dt_t: torch.Tensor,
    A: torch.Tensor,
    B_t: torch.Tensor,
    C_t: torch.Tensor,
    state: torch.Tensor,
    D: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Advance the SSD recurrence by one token.

        h_t = exp(dt_t * A) h_{t-1} + dt_t (B_t x_t^T)
        y_t = C_t^T h_t + D * x_t

    Args:
        x_t: (batch, nheads, headdim)
        dt_t: (batch, nheads)
        A: (nheads,)
        B_t, C_t: (batch, ngroups, dstate)
        state: (batch, nheads, headdim, dstate) -- the state after token t-1
        D: (nheads,) or None

    Returns:
        (y_t, new_state) with y_t (batch, nheads, headdim).
    """
    nheads = x_t.shape[1]
    ngroups = B_t.shape[1]
    rep = _heads_per_group(nheads, ngroups)

    decay = torch.exp(dt_t * A).unsqueeze(-1).unsqueeze(-1)          # (batch, nheads, 1, 1)
    B_h = B_t.repeat_interleave(rep, dim=1)                          # (batch, nheads, dstate)
    C_h = C_t.repeat_interleave(rep, dim=1)
    new_state = decay * state + (
        dt_t.unsqueeze(-1).unsqueeze(-1) * x_t.unsqueeze(-1) * B_h.unsqueeze(-2)
    )
    y_t = torch.einsum("bhpn,bhn->bhp", new_state, C_h)
    if D is not None:
        y_t = y_t + D.view(1, -1, 1) * x_t
    return y_t, new_state


def ssd_sequential_reference(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    coeff: Optional[torch.Tensor] = None,
    extra_terms: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Ground-truth SSD scan: a plain loop, in float64 by default.

        h_t = exp(dt_t A) h_{t-1} + sum_terms coeff_t (B_t x_t^T)
        y_t = C_t^T h_t + D x_t

    Args:
        cu_seqlens: optional (batch, seqlen) int tensor of segment ids. The state is reset
            wherever the id changes, matching the repo-wide packed-document convention
            (`mamba_block.py::_forward_segmented`).
        coeff: state-input coefficient, defaulting to `dt`. Separate from `dt` because `dt` also
            sets the decay, and the exponential-trapezoidal rule scales the input by
            `gamma = lambda * dt` while the decay stays `exp(dt A)`.
        extra_terms: additional `(coefficient, B, x)` triples summed into the state -- the
            trapezoidal rule's `beta * B_{t-1} x_{t-1}` term (MAMBA3_PLAN.md M3).

    Returns:
        (batch, seqlen, nheads, headdim)
    """
    x, dt, A, B, C = (t.to(dtype) for t in (x, dt, A, B, C))
    D = None if D is None else D.to(dtype)
    terms: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = [
        ((dt if coeff is None else coeff.to(dtype)), B, x)
    ] + [(co.to(dtype), bb.to(dtype), xx.to(dtype)) for co, bb, xx in (extra_terms or [])]
    batch, seqlen, nheads, headdim = x.shape
    ngroups, dstate = B.shape[-2], B.shape[-1]
    rep = _heads_per_group(nheads, ngroups)

    state = torch.zeros(batch, nheads, headdim, dstate, dtype=dtype, device=x.device)
    ys = []
    for t in range(seqlen):
        if cu_seqlens is not None and t > 0:
            # Zero the carried state for any row whose document changed at this position.
            new_doc = (cu_seqlens[:, t] != cu_seqlens[:, t - 1]).view(batch, 1, 1, 1)
            state = torch.where(new_doc, torch.zeros_like(state), state)
        state = torch.exp(dt[:, t] * A).view(batch, nheads, 1, 1) * state
        for co, bb, xx in terms:
            B_h = bb[:, t].repeat_interleave(rep, dim=1)          # (batch, nheads, dstate)
            state = state + (
                co[:, t].unsqueeze(-1).unsqueeze(-1)
                * xx[:, t].unsqueeze(-1) * B_h.unsqueeze(-2)
            )
        C_h = C[:, t].repeat_interleave(rep, dim=1)
        y_t = torch.einsum("bhpn,bhn->bhp", state, C_h)
        if D is not None:
            y_t = y_t + D.view(1, -1, 1) * x[:, t]
        ys.append(y_t)
    return torch.stack(ys, dim=1)
