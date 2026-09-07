"""PyTorch autograd interface for selective scan kernel.

Provides a PyTorch-compatible interface with automatic differentiation
for the selective scan operation used in Mamba.

OPTIMIZED VERSION:
- Forward: Uses Triton kernel when available, else chunk-parallel PyTorch
- Backward: Uses chunk-parallel PyTorch (no seq_len-length for loop)
  The parallel implementation lets autograd trace through batched matmuls
  instead of re-running a sequential loop.
"""

import os

import torch
import torch.nn.functional as F
from typing import Optional

try:
    from hybrid_xmamba.kernels.selective_scan.scan_triton import selective_scan_triton
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


def selective_scan_parallel(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    chunk_size: int = 64,
) -> torch.Tensor:
    """Chunk-parallel selective scan using batched matmuls.
    
    Instead of iterating over every timestep (2048 steps), we:
    1. Divide the sequence into chunks of size C
    2. Within each chunk, compute all timesteps in parallel using cumulative products
    3. Only propagate the hidden state across chunks (~L/C sequential steps)
    
    For L=2048, C=64: 32 sequential steps instead of 2048 (~64x speedup).
    This is fully differentiable through standard PyTorch autograd.
    
    Args:
        x: Input (B, L, D)
        dt: Delta values (B, L, D) - already softplus'd
        A: State transition (D, N) - typically negative
        B: Input matrix (B, L, N)
        C: Output matrix (B, L, N)
        D: Skip connection (D,)
        chunk_size: Chunk size for parallelism
        
    Returns:
        Output tensor (B, L, D)
    """
    batch, seq_len, dim = x.shape
    _, _, state_size = B.shape
    device = x.device
    dtype = x.dtype
    
    # Pad to multiple of chunk_size
    pad_len = (chunk_size - seq_len % chunk_size) % chunk_size
    if pad_len > 0:
        x = F.pad(x, (0, 0, 0, pad_len))
        dt = F.pad(dt, (0, 0, 0, pad_len))
        B = F.pad(B, (0, 0, 0, pad_len))
        C = F.pad(C, (0, 0, 0, pad_len))
    
    L = x.shape[1]
    num_chunks = L // chunk_size
    
    # Reshape into chunks: (B, num_chunks, chunk_size, ...)
    x_c = x.reshape(batch, num_chunks, chunk_size, dim)
    dt_c = dt.reshape(batch, num_chunks, chunk_size, dim)
    B_c = B.reshape(batch, num_chunks, chunk_size, state_size)
    C_c = C.reshape(batch, num_chunks, chunk_size, state_size)
    
    # Compute discretized A for all positions: A_disc[t] = exp(dt[t] * A)
    # A: (D, N), dt_c: (B, nc, cs, D) -> dA: (B, nc, cs, D, N)
    dA = dt_c.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (B, nc, cs, D, N)
    A_disc = torch.exp(dA)  # (B, nc, cs, D, N)
    
    # Compute B_bar = dt * B (Taylor approximation, stable for small dt*A)
    # dt_c: (B, nc, cs, D), B_c: (B, nc, cs, N) -> dB: (B, nc, cs, D, N)
    dB = dt_c.unsqueeze(-1) * B_c.unsqueeze(-2)  # (B, nc, cs, D, N)
    # Input contribution: dB * x -> (B, nc, cs, D, N)
    Bx = dB * x_c.unsqueeze(-1)  # (B, nc, cs, D, N)
    
    # Within each chunk, compute cumulative product of A_disc (log-space)
    log_A_disc = dA  # since exp(dA) and log brings it back, this is dt*A
    log_A_cum = torch.cumsum(log_A_disc, dim=2)  # (B, nc, cs, D, N)
    A_cum = torch.exp(log_A_cum)
    
    # Initialize recurrent hidden state
    h = torch.zeros(batch, dim, state_size, device=device, dtype=dtype)
    
    all_outputs = []
    
    for ci in range(num_chunks):
        # Current chunk data
        A_disc_ci = A_disc[:, ci]       # (B, cs, D, N)
        A_cum_ci = A_cum[:, ci]         # (B, cs, D, N)
        Bx_ci = Bx[:, ci]              # (B, cs, D, N)
        C_ci = C_c[:, ci]              # (B, cs, N)
        x_ci = x_c[:, ci]             # (B, cs, D)
        
        # ---- Contribution from recurrent state (previous chunks) ----
        # h_from_prev[t] = A_cum[t] * h  (broadcast across chunk positions)
        # h: (B, D, N), A_cum_ci: (B, cs, D, N) -> (B, cs, D, N)
        h_prev_contribution = A_cum_ci * h.unsqueeze(1)
        
        # ---- Intra-chunk contribution (parallel) ----
        # For position t in chunk, intra contribution = sum_{s=0}^{t} A_cum[t]/A_cum[s] * Bx[s]
        # = A_cum[t] * sum_{s=0}^{t} A_cum[s]^{-1} * Bx[s]
        # We compute the weighted cumsum of Bx / A_cum, then multiply by A_cum
        
        # Bx_weighted[s] = Bx[s] / A_cum[s]  (deweight by cumulative A)
        # Clamp to avoid division by zero for very decayed states
        A_cum_safe = A_cum_ci.clamp(min=1e-8)
        Bx_weighted = Bx_ci / A_cum_safe  # (B, cs, D, N)
        
        # Cumulative sum over time dim within chunk
        Bx_cum = torch.cumsum(Bx_weighted, dim=1)  # (B, cs, D, N)
        
        # Re-weight: intra[t] = A_cum[t] * Bx_cum[t]
        h_intra = A_cum_ci * Bx_cum  # (B, cs, D, N)
        
        # ---- Total hidden state for this chunk ----
        h_chunk = h_prev_contribution + h_intra  # (B, cs, D, N)
        
        # ---- Compute output: y[t] = C[t] @ h[t] + D * x[t] ----
        # h_chunk: (B, cs, D, N), C_ci: (B, cs, N) -> y: (B, cs, D)
        y_ci = torch.einsum('btdn, btn -> btd', h_chunk, C_ci)
        y_ci = y_ci + D.unsqueeze(0).unsqueeze(0) * x_ci
        
        all_outputs.append(y_ci)
        
        # ---- Update recurrent state for next chunk ----
        # h = h_chunk at last position: (B, D, N)
        h = h_chunk[:, -1, :, :]  # (B, D, N)
    
    # Concatenate: (B, L, D)
    output = torch.cat(all_outputs, dim=1)
    
    # Remove padding
    if pad_len > 0:
        output = output[:, :seq_len, :]
    
    return output


def selective_scan_sequential_reference(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
) -> torch.Tensor:
    """Exact selective scan — the recurrence the Mamba block is SPECIFIED to compute.

    Sequential, float64, no chunking, no division, no clamp, so it cannot be wrong in
    the way `selective_scan_parallel` is wrong (which divides by
    `A_cum.clamp(min=1e-8)` and annihilates a token's own contribution wherever the
    clamp fires -- see `analysis/scan_error_bound.md`).

        A_disc[t] = exp(dt[t] * A);  Bx[t] = dt[t] * B[t] * x[t]
        h[t] = A_disc[t] * h[t-1] + Bx[t],  h[-1] = 0
        y[t] = sum_n C[t,n] * h[t,:,n] + D * x[t]

    O(L) sequential and far slower than the chunked path -- this exists to MEASURE the
    fast path's end-to-end effect (H100_SCALING_PLAN.md Phase 14C-3), not to train with.

    Args:
        x: (B, L, D)   dt: (B, L, D)   A: (D, N)   B: (B, L, N)   C: (B, L, N)   D: (D,)

    Returns:
        (B, L, D) in x's dtype.
    """
    in_dtype = x.dtype
    xd, dtd = x.double(), dt.double()
    Ad, Bd, Cd, Dd = A.double(), B.double(), C.double(), D.double()

    batch, seq_len, dim = xd.shape
    h = torch.zeros(batch, dim, Ad.shape[1], dtype=torch.float64, device=xd.device)

    outputs = []
    for t in range(seq_len):
        a_disc = torch.exp(dtd[:, t].unsqueeze(-1) * Ad.unsqueeze(0))
        bx = dtd[:, t].unsqueeze(-1) * Bd[:, t].unsqueeze(-2) * xd[:, t].unsqueeze(-1)
        h = a_disc * h + bx
        outputs.append(torch.einsum("bdn,bn->bd", h, Cd[:, t]) + Dd.unsqueeze(0) * xd[:, t])

    return torch.stack(outputs, dim=1).to(in_dtype)


def selective_scan(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    z: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply selective scan operation.
    
    Public interface for selective scan with automatic kernel selection.
    Uses chunk-parallel PyTorch implementation that is fully differentiable
    through standard autograd (no custom backward needed).
    
    On CUDA with Triton available, uses the Triton kernel for forward
    but still relies on PyTorch autograd for backward.
    
    Args:
        x: Input (B, L, D)
        dt: Delta values (B, L, D)
        A: State transition (D, N)
        B: Input matrix (B, L, N)
        C: Output matrix (B, L, N)
        D: Skip connection (D,)
        z: Optional gating tensor (B, L, D)
        
    Returns:
        Output tensor (B, L, D)
    """
    # Use chunk-parallel implementation (differentiable, no custom backward)
    # Choose chunk size based on sequence length
    seq_len = x.shape[1]
    if seq_len <= 128:
        chunk_size = 32
    elif seq_len <= 512:
        chunk_size = 64
    elif seq_len <= 2048:
        chunk_size = 64
    else:
        chunk_size = 128
    
    # --- fp32 numerical guard (H100 150M stability, 2026-07) ---
    # The chunk-parallel scan divides by the cumulative decay A_cum (clamped to 1e-8)
    # and cumsum's log-decays over the chunk. Under bf16 autocast those underflowing
    # divisions/accumulations produce large-magnitude gradients — the spikes that
    # collapse the 150M model (grad_norm 0.23 -> 1.6 -> representation collapse).
    # Reference Mamba keeps this SSM scan in fp32 for exactly this reason. Run it in
    # fp32 and cast the result back to the mixer dtype (interface unchanged).
    in_dtype = x.dtype

    # --- Phase 14C-3: opt-in exact operator, for MEASUREMENT only ---
    # HYBRID_EXACT_SCAN=1 swaps in the exact sequential recurrence so the
    # end-to-end effect of the clamp defect on the reported metrics can be
    # measured rather than argued. It is OFF by default, so the training and
    # evaluation path is byte-identical to every run that produced the numbers
    # in h100_scaling_state.json -- the Phase 14A operator freeze is respected.
    # It is O(L) sequential and MUCH slower; never enable it for training.
    if os.environ.get("HYBRID_EXACT_SCAN", "0") == "1":
        y = selective_scan_sequential_reference(
            x.float(), dt.float(), A.float(), B.float(), C.float(), D.float(),
        ).to(in_dtype)
        if z is not None:
            y = y * z
        return y

    y = selective_scan_parallel(
        x.float(), dt.float(), A.float(), B.float(), C.float(), D.float(),
        chunk_size=chunk_size,
    ).to(in_dtype)

    # Apply gating if provided
    if z is not None:
        y = y * z

    return y
