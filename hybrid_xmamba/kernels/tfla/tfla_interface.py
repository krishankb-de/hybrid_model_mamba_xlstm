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


def tfla_forward_parallel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    i_gate: torch.Tensor,
    f_gate: torch.Tensor,
    chunk_size: int = 64,
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
        
    Returns:
        Output tensor (B, H, L, D)
    """
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
    f_cum_safe = f_cum.clamp(min=1e-6)
    q_weighted = q_c * f_cum               # (B, H, nc, C, D) absorb decay into query
    k_weighted = k_gated_intra / f_cum_safe  # (B, H, nc, C, D) absorb inverse decay into key
    
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
    # ============================================================
    C_state = torch.zeros(batch, num_heads, head_dim, head_dim, device=device, dtype=dtype)
    n_state = torch.zeros(batch, num_heads, head_dim, device=device, dtype=dtype)
    
    recurrent_num_list = []
    recurrent_den_list = []
    
    for ci in range(num_chunks):
        # Current chunk's cumulative forget: (B, H, chunk_size, D)
        f_cum_ci = f_cum[:, :, ci]
        log_f_cum_ci = log_f_cum[:, :, ci]
        
        # ----- Recurrent contribution to this chunk -----
        # Apply forget-gate decay to queries (key dimension) then contract:
        # h_rec[t,d_v] = sum_d_k(q[t,d_k] * f_cum[t,d_k] * C_state[d_k,d_v])
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
    
    return tfla_forward_parallel(q, k, v, i_gate, f_gate, chunk_size=chunk_size)
