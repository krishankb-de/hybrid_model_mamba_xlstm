"""PyTorch autograd interface for TFLA kernel.

This module provides a PyTorch-compatible interface with automatic differentiation
support for the Tiled Flash Linear Attention kernel.

OPTIMIZED VERSION: Replaces sequential for-loop (2048 steps) with chunk-parallel
matmul implementation (~32 steps for seq_len=2048, chunk=64).

Key improvements:
- Forward: O(L/C) sequential steps with O(C^2 * D) parallel work per chunk
- Backward: Standard PyTorch autograd through the parallel ops (no double-pass)
- No custom autograd Function needed - fully differentiable via torch ops
"""

import torch
import torch.nn.functional as F
from typing import Optional
import math


# fp32 overflows at exp(~88). 40 leaves ~20 orders of magnitude of headroom for the q/k
# magnitudes that multiply these factors before the einsum accumulates them.
_EXP_SAFE = 40.0


def _intra_chunk_sequential(q_c, k_gated, v_c, f_c):
    """Exact intra-chunk term: sequential over positions, parallel over every chunk at once.

    Used only when re-centring the decay would overflow fp32 (see `tfla_forward_parallel`).
    Runs the mLSTM recurrence from a zero state inside each chunk -- which is exactly what the
    intra-chunk term is -- so it forms no exponential of a large number and is exact for any
    gate values. Depth is `chunk_size` instead of one matmul; memory is the (D, D) matrix memory
    held for every chunk simultaneously, ~25 MB at the 150M shape.

    Returns (num, den) shaped (B, H, nc, C, D) and (B, H, nc, C, 1).
    """
    batch, heads, num_chunks, chunk_size, head_dim = q_c.shape
    C_state = torch.zeros(
        batch, heads, num_chunks, head_dim, head_dim, device=q_c.device, dtype=q_c.dtype
    )
    n_state = torch.zeros(
        batch, heads, num_chunks, head_dim, device=q_c.device, dtype=q_c.dtype
    )
    nums, dens = [], []
    for t in range(chunk_size):
        f_t = f_c[:, :, :, t]                                   # (B, H, nc, D)
        ki = k_gated[:, :, :, t]                                # (B, H, nc, D)
        C_state = f_t.unsqueeze(-1) * C_state + torch.einsum(
            'bhcd, bhce -> bhcde', ki, v_c[:, :, :, t]
        )
        n_state = f_t * n_state + ki
        q_t = q_c[:, :, :, t]
        nums.append(torch.einsum('bhcd, bhcde -> bhce', q_t, C_state))
        dens.append(torch.einsum('bhcd, bhcd -> bhc', q_t, n_state))
    return torch.stack(nums, dim=3), torch.stack(dens, dim=3).unsqueeze(-1)


def tfla_forward_parallel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    i_gate: torch.Tensor,
    f_gate: torch.Tensor,
    chunk_size: int = 64,
    tfla_impl: str = "legacy",
) -> torch.Tensor:
    """Chunk-parallel implementation of TFLA using batched matmuls.
    
    Instead of looping over seq_len (2048 steps), this divides the sequence
    into chunks and processes each chunk with parallel matrix operations.
    The recurrent state is only maintained across chunks (L/chunk_size steps).
    
    Complexity: O(L/C) sequential steps with O(C^2 * D) parallel work per step
    For L=2048, C=64: only 32 sequential steps instead of 2048.
    
    Args:
        q: Queries (B, H, L, D)
        k: Keys (B, H, L, D)
        v: Values (B, H, L, D)
        i_gate: Input gates (B, H, L, D)
        f_gate: Forget gates (B, H, L, D)
        chunk_size: Size of each chunk for parallel processing
        tfla_impl: "legacy" (default) keeps the pre-2026-09 numerics, defect included, so
            existing checkpoints stay bit-reproducible. "exact" re-centres the intra-chunk decay
            instead of dividing by it. MAMBA3_PLAN.md M1-H.

    Returns:
        Output tensor (B, H, L, D)
    """
    if tfla_impl not in ("legacy", "exact"):
        raise ValueError(f"tfla_impl must be 'legacy' or 'exact', got {tfla_impl!r}")
    batch, num_heads, seq_len, head_dim = q.shape
    device = q.device
    dtype = q.dtype
    
    # Pad sequence length to multiple of chunk_size
    pad_len = (chunk_size - seq_len % chunk_size) % chunk_size
    if pad_len > 0:
        q = F.pad(q, (0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))
        i_gate = F.pad(i_gate, (0, 0, 0, pad_len))
        f_gate = F.pad(f_gate, (0, 0, 0, pad_len), value=1.0)
    
    padded_len = q.shape[2]
    num_chunks = padded_len // chunk_size
    
    # Reshape into chunks: (B, H, num_chunks, chunk_size, D)
    q_c = q.reshape(batch, num_heads, num_chunks, chunk_size, head_dim)
    k_c = k.reshape(batch, num_heads, num_chunks, chunk_size, head_dim)
    v_c = v.reshape(batch, num_heads, num_chunks, chunk_size, head_dim)
    i_c = i_gate.reshape(batch, num_heads, num_chunks, chunk_size, head_dim)
    f_c = f_gate.reshape(batch, num_heads, num_chunks, chunk_size, head_dim)
    
    # Compute cumulative forget gate products within each chunk (log-space)
    # f_cum_c[..., t, :] = prod(f[0..t]) within chunk
    log_f = torch.log(f_c.clamp(min=1e-6))
    log_f_cum = torch.cumsum(log_f, dim=3)  # (B, H, num_chunks, chunk_size, D)
    f_cum = torch.exp(log_f_cum)
    
    # Causal mask: zero-out future positions (used in intra-chunk attention)
    causal_mask = torch.triu(
        torch.ones(chunk_size, chunk_size, device=device, dtype=torch.bool),
        diagonal=1,
    )
    
    # ============================================================
    # INTRA-CHUNK ATTENTION (fully parallel across all chunks)
    # Per-dimension decay absorbed into query/key scaling — no approximation.
    #
    # decay[i,j,d] = f_cum[i,d] / f_cum[j,d]  (for i >= j)
    # scores[i,j] = sum_d(q[i,d] * k_gated[j,d] * decay[i,j,d])
    #             = sum_d((q[i,d]*f_cum[i,d]) * (k_gated[j,d]/f_cum[j,d]))
    # ============================================================
    k_gated_intra = k_c * i_c  # (B, H, nc, C, D) per-dimension input gating
    if tfla_impl == "legacy":
        # MAMBA3_PLAN.md M1-C: `f_cum` underflows 1e-6 within a chunk for any realistic forget
        # gate -- 70.9% of (t, d) entries at the shipped forget_gate_bias_init=0.0 -- and the
        # clamp then pins the denominator, so decay[i, j] collapses to ~0 even for j close to i.
        # Measured rel-max-err vs an fp64 sequential reference: 0.882 at the shipped settings.
        f_cum_safe = f_cum.clamp(min=1e-6)
        q_weighted = q_c * f_cum                 # absorb decay into query
        k_weighted = k_gated_intra / f_cum_safe  # absorb inverse decay into key
    else:
        # Same factorization, no reciprocal. Re-centre both sides on half the chunk's total
        # log-decay so neither factor has to span the full dynamic range:
        #   decay[i, j] = exp(log_f_cum[i] - log_f_cum[j])
        #               = exp(log_f_cum[i] - m) * exp(m - log_f_cum[j])
        # This is an identity for any m, and m = log_f_cum[-1]/2 splits the range evenly, so a
        # chunk that decays by exp(-44) puts exp(+/-22) on each side instead of exp(-44) on one
        # and exp(+44) on the other. Nothing is clamped, so no term is annihilated.
        m = 0.5 * log_f_cum[:, :, :, -1:, :]     # (B, H, nc, 1, D)
        half_range = 0.5 * log_f_cum[:, :, :, -1, :].abs().max()
        if half_range < _EXP_SAFE:
            q_weighted = q_c * torch.exp(log_f_cum - m)
            k_weighted = k_gated_intra * torch.exp(m - log_f_cum)
        else:
            # Re-centring halves the dynamic range but cannot remove it. A chunk whose total
            # log-decay exceeds 2 * _EXP_SAFE still overflows fp32 on one side, and the causal
            # mask is applied *after* the product, so an inf meets a 0 and yields NaN -- silently
            # wrong in exactly the way this phase exists to eliminate. Reachable in practice:
            # chunk_size=128 with a forget-gate bias near -2 does it.
            #
            # Fall back to scanning within the chunk instead of factorizing across it: sequential
            # in `chunk_size`, parallel over every chunk at once. No exponential of a large
            # number is ever formed, so this is exact for any gate values.
            q_weighted = k_weighted = None
    
    if q_weighted is None:
        h_intra_num, h_intra_den = _intra_chunk_sequential(q_c, k_gated_intra, v_c, f_c)
    else:
        scores = torch.einsum(
            'bhcid, bhcjd -> bhcij', q_weighted, k_weighted
        )  # (B, H, nc, C, C)
        scores = scores.masked_fill(causal_mask, 0.0)

        # Unnormalized weighted sum of values (normalize later with inter-chunk)
        h_intra_num = torch.einsum(
            'bhcij, bhcjd -> bhcid', scores, v_c
        )  # (B, H, nc, C, D)
        h_intra_den = scores.sum(dim=-1, keepdim=True)  # (B, H, nc, C, 1)
    
    # ============================================================
    # INTER-CHUNK RECURRENCE (sequential across chunks only)
    # m_state: per-dim log-scale running max (LSE stabilizer pass-through, Phase 3C)
    # Tracks max(log_i - log_f_cum) across chunk boundaries, mirroring the
    # per-step m_t = max(f̃ + m_{t-1}, ĩ) recurrence in _slow_forward.
    # ============================================================
    C_state = torch.zeros(batch, num_heads, head_dim, head_dim, device=device, dtype=dtype)
    n_state = torch.zeros(batch, num_heads, head_dim, device=device, dtype=dtype)
    # m_state: (B, H, D) — carries max gate log-signal across chunk boundaries
    m_state = torch.full(
        (batch, num_heads, head_dim), float('-inf'), device=device, dtype=dtype
    )

    recurrent_num_list = []
    recurrent_den_list = []

    for ci in range(num_chunks):
        # Current chunk's cumulative forget: (B, H, chunk_size, D)
        f_cum_ci = f_cum[:, :, ci]
        log_f_cum_ci = log_f_cum[:, :, ci]

        # ----- Recurrent contribution to this chunk -----
        q_f = q_c[:, :, ci] * f_cum_ci  # (B, H, C, D)
        h_rec_num = torch.einsum(
            'bhld, bhde -> bhle', q_f, C_state
        )  # (B, H, C, D)

        h_rec_den = torch.einsum(
            'bhld, bhd -> bhl', q_f, n_state
        ).unsqueeze(-1)  # (B, H, C, 1)

        recurrent_num_list.append(h_rec_num)
        recurrent_den_list.append(h_rec_den)

        # ----- Update recurrent state for next chunk -----
        total_f_last = f_cum_ci[:, :, -1, :]  # (B, H, D)
        C_state = total_f_last.unsqueeze(-1) * C_state  # (B,H,D,1) * (B,H,D,D)
        n_state = total_f_last * n_state  # (B,H,D)

        # decay_to_end[t] = prod(f[t+1..end]) = f_cum[-1] / f_cum[t]
        decay_to_end = torch.exp(
            log_f_cum_ci[:, :, -1:, :] - log_f_cum_ci
        )  # (B, H, C, D)

        # Accumulate new key-value pairs into recurrent state
        k_gated_update = k_c[:, :, ci] * i_c[:, :, ci] * decay_to_end
        C_state = C_state + torch.einsum('bhld, bhle -> bhde', k_gated_update, v_c[:, :, ci])
        n_state = n_state + k_gated_update.sum(dim=2)

        # ----- m_state update (LSE pass-through) -----
        # log_alpha[j] = log(i[j]) - log_f_cum[j]: log-scale of key at position j
        log_i_ci = torch.log(i_c[:, :, ci].clamp(min=1e-30))  # (B, H, C, D)
        log_alpha_ci = log_i_ci - log_f_cum_ci                 # (B, H, C, D)
        # Max alpha in this chunk: (B, H, D)
        m_ci = log_alpha_ci.amax(dim=2)
        # Carry m_state forward: decay by chunk-end forget, then take max with new chunk
        log_f_last = log_f_cum_ci[:, :, -1, :]                # (B, H, D)
        m_state = torch.max(log_f_last + m_state, m_ci)
    
    # Stack recurrent contributions (unnormalized): (B, H, nc, C, D) and (B, H, nc, C, 1)
    h_rec_num_all = torch.stack(recurrent_num_list, dim=2)
    h_rec_den_all = torch.stack(recurrent_den_list, dim=2)
    
    # ============================================================
    # COMBINE with joint normalization
    # output = (intra_num + rec_num) / max(intra_den + rec_den, 1)
    # ============================================================
    total_num = h_intra_num + h_rec_num_all  # (B, H, nc, C, D)
    total_den = (h_intra_den + h_rec_den_all).clamp(min=1.0)  # (B, H, nc, C, 1)
    output = total_num / total_den
    output = output.reshape(batch, num_heads, padded_len, head_dim)
    
    # Remove padding
    if pad_len > 0:
        output = output[:, :, :seq_len, :]
    
    return output


def apply_tfla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    i_gate: torch.Tensor,
    f_gate: torch.Tensor,
    tfla_impl: str = "legacy",
) -> torch.Tensor:
    """Apply Tiled Flash Linear Attention.
    
    Public interface for TFLA. Uses chunk-parallel implementation that is
    fully differentiable through standard PyTorch autograd — no custom
    backward pass needed. This eliminates the 2048-step sequential loop
    and the double-forward-pass backward that caused the 40x slowdown.
    
    Args:
        q: Queries (B, H, L, D)
        k: Keys (B, H, L, D)
        v: Values (B, H, L, D)
        i_gate: Input gates (B, H, L, D)
        f_gate: Forget gates (B, H, L, D)
        tfla_impl: "legacy" (default) | "exact" -- see `tfla_forward_parallel`.

    Returns:
        Output tensor (B, H, L, D)
    """
    seq_len = q.shape[2]
    
    # Tune chunk size to balance parallelism vs. recurrence overhead
    if seq_len <= 128:
        chunk_size = 32
    elif seq_len <= 512:
        chunk_size = 64
    elif seq_len <= 2048:
        chunk_size = 64
    else:
        chunk_size = 128
    
    return tfla_forward_parallel(
        q, k, v, i_gate, f_gate, chunk_size=chunk_size, tfla_impl=tfla_impl
    )
