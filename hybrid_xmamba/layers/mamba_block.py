"""Mamba Block implementation.

This module implements the Mamba architecture with selective SSM (State Space Model).
Based on "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from einops import rearrange

from hybrid_xmamba.kernels.selective_scan import selective_scan
from hybrid_xmamba.kernels.selective_scan.scan_interface import (
    selective_scan_exact,
    selective_scan_parallel,
)
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
        scan_impl: str = "legacy",
        dt_init_strategy: str = "none",
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        use_dt_norm: Optional[bool] = None,
    ):
        super().__init__()
        self.dim = dim
        self.state_size = state_size
        self.conv_size = conv_size
        self.expand_factor = expand_factor
        # MAMBA3_PLAN.md M1-E: "legacy" keeps the pre-2026-09 numerics bit-reproducible;
        # "exact" selects the division-free scan. Default flips at M9-A.
        if scan_impl not in ("legacy", "exact"):
            raise ValueError(f"scan_impl must be 'legacy' or 'exact', got {scan_impl!r}")
        self.scan_impl = scan_impl
        # MAMBA3_PLAN.md M1-F. Reference Mamba draws Delta ~ logU[dt_min, dt_max] via an
        # inverse-softplus bias on dt_proj. This repo has never had that init, and Delta sits at
        # ~0.70 (pre_rms) / ~0.81 (hybrid) instead of ~0.02. Two things must both be true for a
        # fix to take effect, which is why they are one phase:
        #   1. the init has to be re-applied AFTER HybridLanguageModel._init_weights zeroes every
        #      bias -- see post_model_init(), called from hybrid_lm.py;
        #   2. dt_norm has to be off, because RMSNorm rescales Delta to unit RMS and discards the
        #      bias offset entirely. Mamba-3 (paper Sec 3.4) normalizes B and C only.
        if dt_init_strategy not in ("none", "mamba"):
            raise ValueError(
                f"dt_init_strategy must be 'none' or 'mamba', got {dt_init_strategy!r}"
            )
        self.dt_init_strategy = dt_init_strategy
        self.dt_min = dt_min
        self.dt_max = dt_max
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
        # use_dt_norm defaults to use_hybrid_norm (the shipped `hybrid` topology). The
        # `hybrid_bc` topology sets it False: B/C norms without the Delta norm, matching
        # Mamba-3 Sec 3.4 and unblocking the dt init above.
        self._use_dt_norm = use_hybrid_norm if use_dt_norm is None else bool(use_dt_norm)
        if use_hybrid_norm:
            # B/C norms are the half Mamba-3 Sec 3.4 keeps (its "BCNorm"/"QKNorm") and are on for
            # both `hybrid` and `hybrid_bc`. The Delta norm is this repo's own addition and is
            # what erases the dt init, so `hybrid_bc` drops it and nothing else.
            self.dt_norm = RMSNorm(self.inner_dim) if self._use_dt_norm else None
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
    
    def post_model_init(self) -> None:
        """Re-apply the Mamba dt init after the parent model's global weight init.

        `HybridLanguageModel.__init__` ends with `self.apply(self._init_weights)`, which zeroes
        every `nn.Linear` bias -- including `dt_proj.bias`. Any init done in `__init__` is
        therefore erased. This hook runs after that pass; it is a no-op unless
        `dt_init_strategy="mamba"`, so the default keeps today's behaviour exactly.
        """
        if self.dt_init_strategy != "mamba":
            return
        with torch.no_grad():
            # Delta ~ logU[dt_min, dt_max]; invert softplus so softplus(bias) reproduces it.
            dt = torch.exp(
                torch.rand(self.inner_dim)
                * (math.log(self.dt_max) - math.log(self.dt_min))
                + math.log(self.dt_min)
            ).clamp(min=1e-4)
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))

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
        # Gate B/C on their own module, not on dt_norm: under `hybrid_bc` the Delta norm is
        # absent while the B/C norms are still required (Mamba-3 Sec 3.4).
        if self.dt_norm is not None:
            dt = self.dt_norm(dt)
        if self.B_norm is not None:
            B = self.B_norm(B)
            C = self.C_norm(C)
        dt = F.softplus(dt)
        
        # Get A from log space
        A = -torch.exp(self.A_log.float())  # (inner_dim, N)
        
        # Selective scan
        if self.use_fast_path:
            y = selective_scan(
                x_conv, dt, A, B, C, self.D.float(), z=None, scan_impl=self.scan_impl
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
        """Non-fast-path selective scan.

        This used to carry its own copy of the chunk-parallel scan, including an identical copy
        of the divide-and-clamp defect (MAMBA3_PLAN.md M1). That mattered more than it looks:
        `scripts/validate_for_willi.sh` builds its Gate 6 model with `use_fast_path=False`, so
        the pre-push harness only ever exercised the buggy duplicate. Both paths now share one
        implementation, so a fix cannot land on one and miss the other.

        Note the two paths still differ in chunk size — this one uses `min(64, seq_len)` where
        `selective_scan` picks adaptively — so `use_fast_path` remains a real switch, just not a
        second implementation.
        """
        chunk_size = min(64, x.shape[1])
        impl = selective_scan_parallel if self.scan_impl == "legacy" else selective_scan_exact
        return impl(x, dt, A, B, C, self.D.float(), chunk_size=chunk_size)

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
