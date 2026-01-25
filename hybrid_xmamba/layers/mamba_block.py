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
    ):
        super().__init__()
        self.dim = dim
        self.state_size = state_size
        self.conv_size = conv_size
        self.expand_factor = expand_factor
        self.inner_dim = dim * expand_factor
        self.use_fast_path = use_fast_path
        
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
        cache: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass of Mamba block.
        
        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            cache: Optional cache for inference
            
        Returns:
            Output tensor of shape (batch, seq_len, dim)
        """
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
        """Slow reference implementation (for debugging/testing).
        
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
        
        # Initialize state
        h = torch.zeros(batch, dim, state_size, device=x.device, dtype=x.dtype)
        
        outputs = []
        for t in range(seq_len):
            # Discretization
            dt_t = dt[:, t, :]  # (B, D)
            A_discrete = torch.exp(A * dt_t.unsqueeze(-1))  # (B, D, N)
            B_t = B[:, t, :].unsqueeze(1)  # (B, 1, N)
            
            # State update
            h = A_discrete * h + (dt_t.unsqueeze(-1) * x[:, t, :].unsqueeze(-1)) * B_t
            
            # Output
            y_t = torch.sum(h * C[:, t, :].unsqueeze(1), dim=-1)  # (B, D)
            y_t = y_t + self.D * x[:, t, :]
            outputs.append(y_t)
        
        return torch.stack(outputs, dim=1)
