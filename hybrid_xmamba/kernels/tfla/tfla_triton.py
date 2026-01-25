"""Triton kernel implementation for Tiled Flash Linear Attention (TFLA).

This kernel implements the efficient TFLA algorithm for mLSTM using Triton JIT compilation.
The implementation follows the two-level hierarchy strategy:
1. Chunkwise Parallelism: Sequence divided into chunks (128-256 tokens)
2. Intra-Chunk Tiling: Flash Attention-like tiling within each chunk

This enables handling sequences of 32k+ tokens by materializing recurrent state C_t
only at chunk boundaries, reducing memory from O(L) to O(L/C).

Key Performance Features:
- Fused attention computation in SRAM (no materialization of full attention matrix)
- Block-by-block QK^T computation with causal masking
- Exponential decay gates in log-space for numerical stability
- Inter-chunk recurrent contribution from previous chunk state
"""

import torch
import triton
import triton.language as tl


@triton.jit
def tfla_chunk_forward_kernel(
    # Input pointers
    Q, K, V, Gates, RecurrentState,
    # Output pointers
    Output,
    # Dimensions
    batch, num_heads, seq_len, head_dim,
    # Strides for Q
    stride_qb, stride_qh, stride_ql, stride_qd,
    # Strides for K
    stride_kb, stride_kh, stride_kl, stride_kd,
    # Strides for V
    stride_vb, stride_vh, stride_vl, stride_vd,
    # Strides for Gates (log-space)
    stride_gb, stride_gh, stride_gl,
    # Strides for RecurrentState [B, H, num_chunks, D, D]
    stride_sb, stride_sh, stride_sc, stride_sd1, stride_sd2,
    # Strides for Output
    stride_ob, stride_oh, stride_ol, stride_od,
    # Block configuration
    CHUNK_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Tiled Flash Linear Attention forward kernel with chunking strategy.
    
    This is the core kernel that implements the TFLA algorithm. The key insight is to:
    1. Divide sequence into chunks to reduce recurrent state materialization
    2. Within each chunk, use Flash Attention-style tiling to avoid materializing QK^T
    3. Add inter-chunk recurrent contribution from previous chunk boundary
    
    Memory complexity: O(L/C * D^2) instead of O(L * D^2)
    where L is sequence length, C is chunk size, D is head dimension.
    """
    # Program indices
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_chunk = tl.program_id(2)
    pid_block_m = tl.program_id(3)
    
    # Calculate chunk boundaries
    chunk_start = pid_chunk * CHUNK_SIZE
    chunk_end = tl.minimum(chunk_start + CHUNK_SIZE, seq_len)
    chunk_len = chunk_end - chunk_start
    
    # Check if this chunk is valid
    if chunk_start >= seq_len:
        return
    
    # Calculate query block offsets within chunk
    offs_m = pid_block_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_m_global = chunk_start + offs_m
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, head_dim)
    
    # Boundary masks
    mask_m = (offs_m < chunk_len) & (offs_m_global < seq_len)
    
    # Base pointers for current batch and head
    q_base = Q + pid_batch * stride_qb + pid_head * stride_qh
    k_base = K + pid_batch * stride_kb + pid_head * stride_kh
    v_base = V + pid_batch * stride_vb + pid_head * stride_vh
    g_base = Gates + pid_batch * stride_gb + pid_head * stride_gh
    o_base = Output + pid_batch * stride_ob + pid_head * stride_oh
    
    # Load Q tile into SRAM
    q_ptrs = q_base + offs_m_global[:, None] * stride_ql + offs_d[None, :] * stride_qd
    q_tile = tl.load(q_ptrs, mask=mask_m[:, None] & (offs_d[None, :] < head_dim), other=0.0)
    
    # Load gates for this Q block (in log-space for numerical stability)
    g_ptrs = g_base + offs_m_global * stride_gl
    gates_m = tl.load(g_ptrs, mask=mask_m, other=0.0)
    
    # ============================================================
    # PART 1: Intra-Chunk Attention (Flash Attention-like tiling)
    # ============================================================
    # Initialize output accumulator in register/SRAM
    acc = tl.zeros([BLOCK_M, head_dim], dtype=tl.float32)
    
    # Loop over K/V blocks within the current chunk
    # This is the "Flash" part - we never materialize the full attention matrix
    for start_n in range(0, chunk_len, BLOCK_N):
        offs_n_chunk = start_n + offs_n
        offs_n_global = chunk_start + offs_n_chunk
        mask_n = (offs_n_chunk < chunk_len) & (offs_n_global < seq_len)
        
        # Causal mask: only attend to positions <= current position within chunk
        # This is critical for autoregressive models
        causal_mask = offs_n_chunk[None, :] <= offs_m[:, None]
        mask_combined = mask_m[:, None] & mask_n[None, :] & causal_mask
        
        # Load K and V tiles into SRAM
        k_ptrs = k_base + offs_n_global[:, None] * stride_kl + offs_d[None, :] * stride_kd
        v_ptrs = v_base + offs_n_global[:, None] * stride_vl + offs_d[None, :] * stride_vd
        k_tile = tl.load(k_ptrs, mask=mask_n[:, None] & (offs_d[None, :] < head_dim), other=0.0)
        v_tile = tl.load(v_ptrs, mask=mask_n[:, None] & (offs_d[None, :] < head_dim), other=0.0)
        
        # Load gates for K positions
        g_n_ptrs = g_base + offs_n_global * stride_gl
        gates_n = tl.load(g_n_ptrs, mask=mask_n, other=0.0)
        
        # 1. Compute intra-chunk attention scores: Q @ K^T
        # This is computed block-by-block in SRAM, never materialized to HBM
        scores = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for d in range(head_dim):
            scores += q_tile[:, d, None] * k_tile[None, :, d]
        
        # 2. Apply exponential decay gates for linear attention
        # Gates are in log-space: log(exp(gate_q) * exp(gate_k)) = gate_q + gate_k
        # Then exponentiate back for numerical stability
        decay = tl.exp(gates_m[:, None] + gates_n[None, :])
        scores = scores * decay
        
        # 3. Apply causal masking (set future positions to 0)
        scores = tl.where(mask_combined, scores, 0.0)
        
        # 4. Compute output contribution: Scores @ V
        # This accumulates the attention-weighted values
        for d in range(head_dim):
            acc[:, d] += tl.sum(scores * v_tile[None, :, d], axis=1)
    
    # ============================================================
    # PART 2: Inter-Chunk Recurrent Contribution
    # ============================================================
    # Add contribution from previous chunk's recurrent state C_{k-1}
    # This state is only materialized at chunk boundaries, not every position
    if pid_chunk > 0:
        # Load previous chunk's recurrent state from HBM
        # Shape: [D, D] - represents accumulated key-value interaction
        state_base = RecurrentState + pid_batch * stride_sb + pid_head * stride_sh + (pid_chunk - 1) * stride_sc
        
        # Compute Q @ prev_state to get recurrent contribution
        # We do this in tiles to fit in SRAM
        for d_out in range(0, head_dim, BLOCK_N):
            offs_d_out = d_out + tl.arange(0, BLOCK_N)
            mask_d_out = offs_d_out < head_dim
            
            recurrent_contrib_tile = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            
            # Compute q_tile @ prev_state[:, offs_d_out]
            for d_in in range(head_dim):
                state_ptrs = state_base + d_in * stride_sd1 + offs_d_out * stride_sd2
                state_col = tl.load(state_ptrs, mask=mask_d_out, other=0.0)
                recurrent_contrib_tile += q_tile[:, d_in, None] * state_col[None, :]
            
            # Apply exponential decay from chunk boundary to current position
            # Load gate at chunk boundary
            chunk_start_gate_ptr = g_base + chunk_start * stride_gl
            chunk_start_gate = tl.load(chunk_start_gate_ptr)
            decay_from_prev = tl.exp(gates_m - chunk_start_gate)
            
            # Add to accumulator with decay
            for i in range(min(BLOCK_N, head_dim - d_out)):
                acc[:, d_out + i] += recurrent_contrib_tile[:, i] * decay_from_prev
    
    # Store final output to HBM
    o_ptrs = o_base + offs_m_global[:, None] * stride_ol + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc, mask=mask_m[:, None] & (offs_d[None, :] < head_dim))


@triton.jit
def update_recurrent_state_kernel(
    # Input pointers
    K, V, Gates,
    # Output pointer
    RecurrentState,
    # Dimensions
    batch, num_heads, seq_len, head_dim,
    # Strides for K
    stride_kb, stride_kh, stride_kl, stride_kd,
    # Strides for V
    stride_vb, stride_vh, stride_vl, stride_vd,
    # Strides for Gates
    stride_gb, stride_gh, stride_gl,
    # Strides for RecurrentState [B, H, num_chunks, D, D]
    stride_sb, stride_sh, stride_sc, stride_sd1, stride_sd2,
    # Block configuration
    CHUNK_SIZE: tl.constexpr,
):
    """
    Materialize recurrent states at chunk boundaries.
    
    For each chunk, compute: C_chunk = sum_{t in chunk} exp(gate_t) * (k_t @ v_t^T)
    
    This is the key memory optimization - instead of storing states at every position,
    we only materialize them at chunk boundaries, reducing memory from O(L*D^2) to O(L/C*D^2).
    """
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_chunk = tl.program_id(2)
    
    # Calculate chunk boundaries
    chunk_start = pid_chunk * CHUNK_SIZE
    chunk_end = tl.minimum(chunk_start + CHUNK_SIZE, seq_len)
    
    if chunk_start >= seq_len:
        return
    
    # Dimension offsets
    offs_d = tl.arange(0, head_dim)
    mask_d = offs_d < head_dim
    
    # Base pointers
    k_base = K + pid_batch * stride_kb + pid_head * stride_kh
    v_base = V + pid_batch * stride_vb + pid_head * stride_vh
    g_base = Gates + pid_batch * stride_gb + pid_head * stride_gh
    state_base = RecurrentState + pid_batch * stride_sb + pid_head * stride_sh + pid_chunk * stride_sc
    
    # Initialize state accumulator [D, D]
    # This will accumulate the outer products weighted by exponential gates
    state_acc = tl.zeros([head_dim, head_dim], dtype=tl.float32)
    
    # Accumulate contributions from all positions in this chunk
    for t in range(chunk_start, chunk_end):
        # Load k_t, v_t, gate_t for position t
        k_ptrs = k_base + t * stride_kl + offs_d * stride_kd
        v_ptrs = v_base + t * stride_vl + offs_d * stride_vd
        g_ptr = g_base + t * stride_gl
        
        k_t = tl.load(k_ptrs, mask=mask_d, other=0.0)
        v_t = tl.load(v_ptrs, mask=mask_d, other=0.0)
        gate_t = tl.load(g_ptr)
        
        # Compute weighted outer product: exp(gate_t) * (k_t @ v_t^T)
        # This represents the contribution of position t to the recurrent state
        weight = tl.exp(gate_t)
        
        # Compute outer product: k_t[:, None] * v_t[None, :]
        # We do this element-wise to avoid materializing the full D x D matrix at once
        for i in range(head_dim):
            for j in range(head_dim):
                state_acc[i, j] += weight * k_t[i] * v_t[j]
    
    # Store accumulated state for this chunk
    # This state will be used by the forward kernel to add inter-chunk contributions
    for i in range(head_dim):
        for j in range(head_dim):
            state_ptr = state_base + i * stride_sd1 + j * stride_sd2
            tl.store(state_ptr, state_acc[i, j])


def tfla_forward_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gates: torch.Tensor,
) -> torch.Tensor:
    """
    Triton-accelerated TFLA forward pass with chunking strategy.
    
    This is the main entry point that orchestrates the two-level hierarchy:
    1. First materialize recurrent states at chunk boundaries
    2. Then compute attention within each chunk + add inter-chunk recurrent contribution
    
    Args:
        q: Queries [B, H, L, D] - query vectors for attention
        k: Keys [B, H, L, D] - key vectors for attention
        v: Values [B, H, L, D] - value vectors to aggregate
        gates: Log-space exponential gates [B, H, L] - control forgetting and input
        
    Returns:
        Output tensor [B, H, L, D] - attention output
    """
    batch, num_heads, seq_len, head_dim = q.shape
    assert k.shape == v.shape == q.shape, "Q, K, V must have same shape"
    assert gates.shape == (batch, num_heads, seq_len), "Gates shape mismatch"
    
    # Allocate output tensor
    output = torch.empty_like(q)
    
    # Chunking configuration for memory efficiency
    # Larger chunks = less overhead but more memory per chunk
    # Smaller chunks = more overhead but better memory efficiency
    # Optimal size depends on sequence length and available SRAM
    if seq_len <= 2048:
        CHUNK_SIZE = 128
    elif seq_len <= 8192:
        CHUNK_SIZE = 256
    else:
        CHUNK_SIZE = 512
    
    num_chunks = (seq_len + CHUNK_SIZE - 1) // CHUNK_SIZE
    
    # Allocate recurrent state storage [B, H, num_chunks, D, D]
    # This is the key memory optimization - only storing state at chunk boundaries
    recurrent_state = torch.zeros(
        batch, num_heads, num_chunks, head_dim, head_dim,
        dtype=q.dtype, device=q.device
    )
    
    # Step 1: Materialize recurrent states at chunk boundaries
    # This allows us to break the sequential dependency across chunks
    grid_state = (batch, num_heads, num_chunks)
    
    update_recurrent_state_kernel[grid_state](
        k, v, gates, recurrent_state,
        batch, num_heads, seq_len, head_dim,
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        gates.stride(0), gates.stride(1), gates.stride(2),
        recurrent_state.stride(0), recurrent_state.stride(1),
        recurrent_state.stride(2), recurrent_state.stride(3), recurrent_state.stride(4),
        CHUNK_SIZE=CHUNK_SIZE,
    )
    
    # Step 2: Compute TFLA with intra-chunk tiling and inter-chunk recurrence
    # Block sizes for tiling within chunks (Flash Attention-style)
    BLOCK_M = 64  # Query block size
    BLOCK_N = 64  # Key/Value block size
    
    # Calculate number of query blocks per chunk
    num_blocks_per_chunk = (CHUNK_SIZE + BLOCK_M - 1) // BLOCK_M
    
    # Grid: (batch, heads, num_chunks, num_blocks_per_chunk)
    # Each thread block handles one query block within one chunk
    grid = (batch, num_heads, num_chunks, num_blocks_per_chunk)
    
    tfla_chunk_forward_kernel[grid](
        q, k, v, gates, recurrent_state, output,
        batch, num_heads, seq_len, head_dim,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        gates.stride(0), gates.stride(1), gates.stride(2),
        recurrent_state.stride(0), recurrent_state.stride(1),
        recurrent_state.stride(2), recurrent_state.stride(3), recurrent_state.stride(4),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        CHUNK_SIZE=CHUNK_SIZE,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    
    return output
