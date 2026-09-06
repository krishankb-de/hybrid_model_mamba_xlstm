"""PyTorch autograd interface for selective scan kernel.

Provides a PyTorch-compatible interface with automatic differentiation
for the selective scan operation used in Mamba.

OPTIMIZED VERSION:
- Forward: Uses Triton kernel when available, else chunk-parallel PyTorch
- Backward: Uses chunk-parallel PyTorch (no seq_len-length for loop)
  The parallel implementation lets autograd trace through batched matmuls
  instead of re-running a sequential loop.
"""

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


def selective_scan_exact(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    chunk_size: Optional[int] = None,
) -> torch.Tensor:
    """Exact chunked selective scan — no division, no clamp (MAMBA3_PLAN.md M1-E).

    `selective_scan_parallel` computes the intra-chunk term as
    ``A_cum * cumsum(Bx / A_cum.clamp(min=1e-8))``. Wherever ``A_cum[s]`` falls below the clamp
    the ratio ``A_cum[t] / max(A_cum[s], 1e-8)`` collapses to ~0 even though the true ratio is
    O(1) for ``s`` near ``t``, so a token's own contribution to the state is annihilated. Measured
    rel-max-err against a float64 sequential reference: 3.9e-01 at Δ=0.1, 9.2e-01 at Δ=0.705 (the
    Δ this repo actually initializes to). See tests/test_mamba3_numerics.py.

    This routine removes the reciprocal entirely by flipping which axis is parallel:

      1. scan *within* each chunk sequentially from a zero state, with all chunks in parallel
         (``chunk_size`` steps over ``(batch, num_chunks, dim, state)``);
      2. carry the per-chunk end states across chunks (``L / chunk_size`` steps over
         ``(batch, dim, state)``);
      3. add ``A_cum * carry`` back into every position, fully parallel.

    Depth is ``chunk_size + L/chunk_size`` rather than ``L/chunk_size``, minimized near
    ``chunk_size = sqrt(L)``, which is what the default picks. Peak memory is unchanged — no
    ``(cs, cs, dim, state)`` mask is ever materialized, which for this Mamba-1 parameterization
    (``A`` is ``(dim, state)``, not scalar-per-head) would be 19.3 GB at chunk 64 / bs 48.

    ``A_cum`` may still underflow to zero. That is correct here and harmless: it multiplies the
    carried state, driving a genuinely-decayed contribution to zero, and is never a denominator.

    Args/Returns: identical to `selective_scan_parallel`.
    """
    batch, seq_len, dim = x.shape
    state_size = B.shape[-1]
    device, dtype = x.device, x.dtype

    if chunk_size is None:
        # Depth is chunk_size + L/chunk_size, minimized at sqrt(L); snap to a power of two in
        # [8, 64] so the reshape stays cheap and the loop count stays predictable.
        target = max(1.0, float(seq_len)) ** 0.5
        chunk_size = min(64, max(8, 1 << int(target).bit_length() - 1))
    chunk_size = min(chunk_size, seq_len) if seq_len > 0 else chunk_size

    pad_len = (chunk_size - seq_len % chunk_size) % chunk_size
    if pad_len > 0:
        x = F.pad(x, (0, 0, 0, pad_len))
        dt = F.pad(dt, (0, 0, 0, pad_len))
        B = F.pad(B, (0, 0, 0, pad_len))
        C = F.pad(C, (0, 0, 0, pad_len))

    padded_len = x.shape[1]
    num_chunks = padded_len // chunk_size

    x_c = x.reshape(batch, num_chunks, chunk_size, dim)
    dt_c = dt.reshape(batch, num_chunks, chunk_size, dim)
    B_c = B.reshape(batch, num_chunks, chunk_size, state_size)
    C_c = C.reshape(batch, num_chunks, chunk_size, state_size)

    # dA[t] = dt[t] * A is the per-step log-decay; A_disc = exp(dA); A_cum = decay from chunk start.
    dA = dt_c.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (B, nc, cs, D, N)
    A_disc = torch.exp(dA)
    A_cum = torch.exp(torch.cumsum(dA, dim=2))
    dB = dt_c.unsqueeze(-1) * B_c.unsqueeze(-2)
    Bx = dB * x_c.unsqueeze(-1)  # (B, nc, cs, D, N)

    # (1) intra-chunk scan from a zero state, every chunk in parallel.
    h = torch.zeros(batch, num_chunks, dim, state_size, device=device, dtype=dtype)
    h_local_steps = []
    for t in range(chunk_size):
        h = A_disc[:, :, t] * h + Bx[:, :, t]
        h_local_steps.append(h)
    h_local = torch.stack(h_local_steps, dim=2)  # (B, nc, cs, D, N)

    # (2) carry chunk-end states forward. `carry_prev[c]` is the state entering chunk c.
    chunk_decay = A_cum[:, :, -1]     # (B, nc, D, N) — total decay across the chunk
    chunk_end = h_local[:, :, -1]     # (B, nc, D, N) — chunk-local end state
    carry = torch.zeros(batch, dim, state_size, device=device, dtype=dtype)
    carries = []
    for ci in range(num_chunks):
        carries.append(carry)
        carry = chunk_decay[:, ci] * carry + chunk_end[:, ci]
    carry_prev = torch.stack(carries, dim=1)  # (B, nc, D, N)

    # (3) h[t] = A_cum[t] * (state entering the chunk) + (chunk-local scan). Fully parallel.
    h_full = h_local + A_cum * carry_prev.unsqueeze(2)
    y = torch.einsum("bctdn,bctn->bctd", h_full, C_c)
    y = y + D.view(1, 1, 1, -1) * x_c
    y = y.reshape(batch, padded_len, dim)
    return y[:, :seq_len] if pad_len > 0 else y


def selective_scan(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    z: Optional[torch.Tensor] = None,
    scan_impl: str = "legacy",
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
        scan_impl: "legacy" (default) reproduces every number published before 2026-09 —
            including its divide-and-clamp defect, so existing checkpoints and the A0 control
            arm stay bit-reproducible. "exact" selects the division-free scan.
            MAMBA3_PLAN.md M1-E; the default flips to "exact" at M9-A.

    Returns:
        Output tensor (B, L, D)
    """
    if scan_impl not in ("legacy", "exact"):
        raise ValueError(f"scan_impl must be 'legacy' or 'exact', got {scan_impl!r}")
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
    # "exact" keeps the fp32 guard: it removes the reciprocal, but the scan still exponentiates
    # a cumsum that spans exp(0) down to underflow inside one chunk, and bf16 has 8 mantissa bits.
    in_dtype = x.dtype
    impl = selective_scan_parallel if scan_impl == "legacy" else selective_scan_exact
    y = impl(
        x.float(), dt.float(), A.float(), B.float(), C.float(), D.float(),
        chunk_size=chunk_size if scan_impl == "legacy" else None,
    ).to(in_dtype)

    # Apply gating if provided
    if z is not None:
        y = y * z

    return y
