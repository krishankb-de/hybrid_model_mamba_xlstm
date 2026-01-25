"""mLSTM (matrix LSTM) Block implementation.

This module implements the mLSTM variant from xLSTM with exponential gating
and efficient TFLA (Tiled Flash Linear Attention) kernel.
Based on "xLSTM: Extended Long Short-Term Memory"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from einops import rearrange

from hybrid_xmamba.kernels.tfla import apply_tfla
from hybrid_xmamba.layers.normalization import RMSNorm
from hybrid_xmamba.layers.activations import exponential_activation


class mLSTMBlock(nn.Module):
    """mLSTM (matrix LSTM) mixer block.
    
    Uses exponential gating and matrix-valued hidden states for enhanced expressiveness.
    
    Args:
        dim: Model dimension
        head_dim: Dimension per attention head
        num_heads: Number of attention heads
        use_tfla: Whether to use Tiled Flash Linear Attention kernel
        proj_factor: Projection factor for input (typically 2)
    """
    
    def __init__(
        self,
        dim: int,
        head_dim: int = 64,
        num_heads: Optional[int] = None,
        use_tfla: bool = True,
        proj_factor: int = 2,
    ):
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        
        if num_heads is None:
            self.num_heads = max(1, dim // head_dim)
        else:
            self.num_heads = num_heads
        
        self.inner_dim = self.num_heads * head_dim
        self.use_tfla = use_tfla
        self.proj_factor = proj_factor
        
        # Input projections
        self.in_proj = nn.Linear(dim, self.inner_dim * proj_factor, bias=False)
        
        # mLSTM specific projections
        # Query, Key, Value for the linear attention mechanism
        self.q_proj = nn.Linear(self.inner_dim, self.inner_dim, bias=False)
        self.k_proj = nn.Linear(self.inner_dim, self.inner_dim, bias=False)
        self.v_proj = nn.Linear(self.inner_dim, self.inner_dim, bias=False)
        
        # Gates: input, forget, output
        self.i_gate_proj = nn.Linear(self.inner_dim, self.inner_dim, bias=True)
        self.f_gate_proj = nn.Linear(self.inner_dim, self.inner_dim, bias=True)
        self.o_gate_proj = nn.Linear(self.inner_dim, self.inner_dim, bias=True)
        
        # Layer normalization for queries and keys
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)
        
        # Output projection
        self.out_proj = nn.Linear(self.inner_dim, dim, bias=False)
        
        # Learnable parameters for stabilization
        self.register_buffer('eps', torch.tensor(1e-6))
    
    def forward(
        self, 
        x: torch.Tensor,
        cache: Optional[dict] = None
    ) -> torch.Tensor:
        """Forward pass of mLSTM block.
        
        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            cache: Optional cache for inference
            
        Returns:
            Output tensor of shape (batch, seq_len, dim)
        """
        batch, seq_len, dim = x.shape
        
        # Input projection
        x_proj = self.in_proj(x)  # (B, L, inner_dim * proj_factor)
        
        if self.proj_factor == 2:
            x_inner, x_gate = x_proj.chunk(2, dim=-1)
        else:
            x_inner = x_proj
            x_gate = x_inner
        
        # Query, Key, Value projections
        q = self.q_proj(x_inner)  # (B, L, inner_dim)
        k = self.k_proj(x_inner)  # (B, L, inner_dim)
        v = self.v_proj(x_inner)  # (B, L, inner_dim)
        
        # Reshape for multi-head
        q = rearrange(q, 'b l (h d) -> b h l d', h=self.num_heads)
        k = rearrange(k, 'b l (h d) -> b h l d', h=self.num_heads)
        v = rearrange(v, 'b l (h d) -> b h l d', h=self.num_heads)
        
        # Normalize queries and keys
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # Gates with exponential activation
        i_gate = exponential_activation(self.i_gate_proj(x_inner))  # Input gate
        f_gate = torch.sigmoid(self.f_gate_proj(x_inner))  # Forget gate
        o_gate = torch.sigmoid(self.o_gate_proj(x_inner))  # Output gate
        
        # Reshape gates for multi-head
        i_gate = rearrange(i_gate, 'b l (h d) -> b h l d', h=self.num_heads)
        f_gate = rearrange(f_gate, 'b l (h d) -> b h l d', h=self.num_heads)
        o_gate = rearrange(o_gate, 'b l (h d) -> b h l d', h=self.num_heads)
        
        # Apply TFLA or fallback to standard implementation
        if self.use_tfla:
            # Use optimized Triton kernel
            h = apply_tfla(q, k, v, i_gate, f_gate)
        else:
            # Slow reference implementation
            h = self._slow_forward(q, k, v, i_gate, f_gate)
        
        # Apply output gate
        h = h * o_gate
        
        # Reshape back
        h = rearrange(h, 'b h l d -> b l (h d)')
        
        # Gating with input
        h = h * torch.sigmoid(x_gate)
        
        # Output projection
        output = self.out_proj(h)
        
        return output
    
    def _slow_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        i_gate: torch.Tensor,
        f_gate: torch.Tensor,
    ) -> torch.Tensor:
        """Slow reference implementation of mLSTM.
        
        Args:
            q: Queries (B, H, L, D)
            k: Keys (B, H, L, D)
            v: Values (B, H, L, D)
            i_gate: Input gates (B, H, L, D)
            f_gate: Forget gates (B, H, L, D)
            
        Returns:
            Output tensor (B, H, L, D)
        """
        batch, num_heads, seq_len, head_dim = q.shape
        
        # Initialize cell state (matrix-valued)
        C = torch.zeros(batch, num_heads, head_dim, head_dim, 
                       device=q.device, dtype=q.dtype)
        n = torch.zeros(batch, num_heads, head_dim, 1,
                       device=q.device, dtype=q.dtype)
        
        outputs = []
        
        for t in range(seq_len):
            # Get timestep data
            q_t = q[:, :, t, :]  # (B, H, D)
            k_t = k[:, :, t, :]  # (B, H, D)
            v_t = v[:, :, t, :]  # (B, H, D)
            i_t = i_gate[:, :, t, :]  # (B, H, D)
            f_t = f_gate[:, :, t, :]  # (B, H, D)
            
            # Update cell state with forget gate
            C = f_t.unsqueeze(-1) * C
            n = f_t.unsqueeze(-1) * n
            
            # Add new information with input gate
            k_v = torch.einsum('bhd,bhe->bhde', k_t, v_t)
            C = C + i_t.unsqueeze(-1) * k_v
            n = n + i_t.unsqueeze(-1) * k_t.unsqueeze(-1)
            
            # Compute output
            h_t_num = torch.einsum('bhd,bhde->bhe', q_t, C)
            h_t_den = torch.einsum('bhd,bhd->bh', q_t, n.squeeze(-1))
            h_t_den = torch.clamp(h_t_den, min=1.0)  # Stabilization
            
            h_t = h_t_num / h_t_den.unsqueeze(-1)
            outputs.append(h_t)
        
        return torch.stack(outputs, dim=2)
