"""Mamba Block implementation.

This module implements the Mamba architecture with selective SSM (State Space Model).
Based on "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from einops import rearrange

from hybrid_xmamba.kernels.selective_scan import selective_scan
from hybrid_xmamba.layers.normalization import RMSNorm


class MambaBlock(nn.Module):
    """Mamba mixer block with selective scan.
    
    Args:
        dim: Model dimension
        state_size: SSM state dimension (N in paper)
        conv_size: Convolution kernel size (typically 4)
        expand_factor: Expansion factor for inner dimension (typically 2)
        dt_rank: Rank of dt projection (typically 'auto' = ceil(dim / 16))
        use_fast_path: Whether to use optimized kernel path
    """
    
    def __init__(
        self,
        dim: int,
        state_size: int = 16,
        conv_size: int = 4,
        expand_factor: int = 2,
        dt_rank: Optional[int] = None,
        use_fast_path: bool = True,
        use_hybrid_norm: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.state_size = state_size
        self.conv_size = conv_size
        self.expand_factor = expand_factor
        self.inner_dim = dim * expand_factor
        self.use_fast_path = use_fast_path
        self.use_hybrid_norm = use_hybrid_norm
        
        # Determine dt_rank
        if dt_rank is None:
            self.dt_rank = max(1, dim // 16)
        else:
            self.dt_rank = dt_rank
        
        # Input projection (x and z branches)
        self.in_proj = nn.Linear(dim, self.inner_dim * 2, bias=False)
        
        # Depthwise convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.inner_dim,
            out_channels=self.inner_dim,
            kernel_size=conv_size,
            padding=conv_size - 1,
            groups=self.inner_dim,
            bias=True,
        )
        
        # SSM projections
        self.x_proj = nn.Linear(self.inner_dim, self.dt_rank + state_size * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.inner_dim, bias=True)

        # Phase 4B (HybridNorm): per-projection pre-norm on Δ/B/C before selective scan
        if use_hybrid_norm:
            self.dt_norm = RMSNorm(self.inner_dim)
            self.B_norm = RMSNorm(state_size)
            self.C_norm = RMSNorm(state_size)
        else:
            self.dt_norm = None
            self.B_norm = None
            self.C_norm = None
        
        # SSM parameters - A is state transition, D is skip connection
        A = torch.arange(1, state_size + 1, dtype=torch.float32).repeat(self.inner_dim, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.inner_dim))
        
        # Output projection
        self.out_proj = nn.Linear(self.inner_dim, dim, bias=False)
        
        # Activation
        self.activation = nn.SiLU()
    
    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of Mamba block.

        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            cache: Optional cache for inference
            cu_seqlens: Optional (B, L) int tensor of per-position doc-ids.
                When provided, splits each batch row into per-doc segments and
                runs the mixer independently per segment so SSM state does not
                cross document boundaries (Phase 6).

        Returns:
            Output tensor of shape (batch, seq_len, dim)
        """
        # Phase 6D: per-segment wrapper when cu_seqlens provided.
        if cu_seqlens is not None:
            return self._forward_segmented(x, cu_seqlens)
        batch, seq_len, dim = x.shape
        
        # Input projection: split into x and z (gate)
        xz = self.in_proj(x)  # (B, L, 2*inner_dim)
        x_inner, z = xz.chunk(2, dim=-1)  # Each (B, L, inner_dim)
        
        # Depthwise convolution
        x_conv = rearrange(x_inner, 'b l d -> b d l')
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]  # Trim padding
        x_conv = rearrange(x_conv, 'b d l -> b l d')
        x_conv = self.activation(x_conv)
        
        # SSM parameters
        x_proj_out = self.x_proj(x_conv)  # (B, L, dt_rank + 2*N)
        dt, B, C = torch.split(
            x_proj_out, 
            [self.dt_rank, self.state_size, self.state_size], 
            dim=-1
        )
        
        # dt projection and transformation
        dt = self.dt_proj(dt)  # (B, L, inner_dim)
        # Phase 4B: pre-norm Δ/B/C before selective scan (HybridNorm)
        if self.dt_norm is not None:
            dt = self.dt_norm(dt)
            B = self.B_norm(B)
            C = self.C_norm(C)
        dt = F.softplus(dt)
        
        # Get A from log space
        A = -torch.exp(self.A_log.float())  # (inner_dim, N)
        
        # Selective scan
        if self.use_fast_path:
            y = selective_scan(
                x_conv, dt, A, B, C, self.D.float(), z=None
            )
        else:
            y = self._slow_forward(x_conv, dt, A, B, C)
        
        # Gating and output projection
        y = y * self.activation(z)
        output = self.out_proj(y)
        
        return output
    
    def _slow_forward(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        """Parallel reference implementation using chunk-wise scan.
        
        Replaces the sequential for-loop with a chunk-parallel approach.
        Divides sequence into chunks and uses cumulative sums within chunks,
        only iterating across chunks (~L/chunk_size steps).
        
        Args:
            x: Input (B, L, D)
            dt: Delta values (B, L, D)
            A: State transition (D, N)
            B: Input projection (B, L, N)
            C: Output projection (B, L, N)
            
        Returns:
            Output tensor (B, L, D)
        """
        batch, seq_len, dim = x.shape
        _, _, state_size = B.shape
        chunk_size = min(64, seq_len)
        
        # Pad to multiple of chunk_size
        pad_len = (chunk_size - seq_len % chunk_size) % chunk_size
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))
            dt = F.pad(dt, (0, 0, 0, pad_len))
            B = F.pad(B, (0, 0, 0, pad_len))
            C = F.pad(C, (0, 0, 0, pad_len))
        
        L = x.shape[1]
        num_chunks = L // chunk_size
        
        # Reshape into chunks
        x_c = x.reshape(batch, num_chunks, chunk_size, dim)
        dt_c = dt.reshape(batch, num_chunks, chunk_size, dim)
        B_c = B.reshape(batch, num_chunks, chunk_size, state_size)
        C_c = C.reshape(batch, num_chunks, chunk_size, state_size)
        
        # Discretize: dA = dt * A -> A_disc = exp(dA)
        dA = dt_c.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        A_disc = torch.exp(dA)  # (B, nc, cs, D, N)
        
        # Input contribution: Bx = dt * x * B
        Bx = dt_c.unsqueeze(-1) * x_c.unsqueeze(-1) * B_c.unsqueeze(-2)
        
        # Cumulative A within chunk (log-space)
        log_A_cum = torch.cumsum(dA, dim=2)
        A_cum = torch.exp(log_A_cum)
        
        h = torch.zeros(batch, dim, state_size, device=x.device, dtype=x.dtype)
        all_outputs = []
        
        for ci in range(num_chunks):
            A_cum_ci = A_cum[:, ci]  # (B, cs, D, N)
            Bx_ci = Bx[:, ci]       # (B, cs, D, N)
            C_ci = C_c[:, ci]       # (B, cs, N)
            x_ci = x_c[:, ci]      # (B, cs, D)
            
            # Previous state contribution
            h_prev = A_cum_ci * h.unsqueeze(1)  # (B, cs, D, N)
            
            # Intra-chunk via cumulative sum
            A_cum_safe = A_cum_ci.clamp(min=1e-8)
            Bx_weighted = Bx_ci / A_cum_safe
            Bx_cum = torch.cumsum(Bx_weighted, dim=1)
            h_intra = A_cum_ci * Bx_cum
            
            h_chunk = h_prev + h_intra  # (B, cs, D, N)
            
            # Output: y = C @ h + D * x
            y_ci = torch.einsum('btdn, btn -> btd', h_chunk, C_ci)
            y_ci = y_ci + self.D.unsqueeze(0).unsqueeze(0) * x_ci
            all_outputs.append(y_ci)
            
            h = h_chunk[:, -1, :, :]
        
        output = torch.cat(all_outputs, dim=1)
        if pad_len > 0:
            output = output[:, :seq_len, :]
        return output

    def _forward_segmented(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        """Per-(row, segment) wrapper that resets SSM state at doc boundaries.

        Args:
            x: (B, L, D)
            cu_seqlens: (B, L) int tensor; segment id per position.

        Returns:
            Output (B, L, D) with per-segment selective scan applied independently.
        """
        B, L, _ = x.shape
        out_parts = []
        for b in range(B):
            ids = cu_seqlens[b]
            # boundaries: positions where doc id changes (+ L sentinel)
            change = torch.ones(L, dtype=torch.bool, device=ids.device)
            change[1:] = ids[1:] != ids[:-1]
            starts = torch.nonzero(change, as_tuple=False).flatten().tolist()
            starts.append(L)
            row_pieces = []
            for i in range(len(starts) - 1):
                s, e = starts[i], starts[i + 1]
                seg = x[b : b + 1, s:e, :]
                row_pieces.append(self.forward(seg))  # cu_seqlens=None branch
            out_parts.append(torch.cat(row_pieces, dim=1))
        return torch.cat(out_parts, dim=0)
