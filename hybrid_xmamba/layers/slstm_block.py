"""sLSTM (scalar LSTM) Block implementation.

This module implements the sLSTM variant from xLSTM with scalar gates
and enhanced memory capabilities.
Based on "xLSTM: Extended Long Short-Term Memory"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from einops import rearrange

from hybrid_xmamba.layers.normalization import RMSNorm
from hybrid_xmamba.layers.activations import exponential_activation


class sLSTMBlock(nn.Module):
    """sLSTM (scalar LSTM) mixer block.
    
    Enhanced scalar LSTM with exponential gating and memory mixing.
    
    Args:
        dim: Model dimension
        hidden_dim: Hidden state dimension
        num_heads: Number of LSTM heads for parallel processing
        use_exponential_gate: Whether to use exponential gating
    """
    
    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        num_heads: int = 4,
        use_exponential_gate: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim or dim
        self.num_heads = num_heads
        self.use_exponential_gate = use_exponential_gate
        
        # Ensure hidden_dim is divisible by num_heads
        assert self.hidden_dim % num_heads == 0, \
            f"hidden_dim {self.hidden_dim} must be divisible by num_heads {num_heads}"
        
        self.head_dim = self.hidden_dim // num_heads
        
        # Input projection
        self.in_proj = nn.Linear(dim, self.hidden_dim, bias=False)
        
        # LSTM gates: input, forget, output, cell
        self.gate_proj = nn.Linear(self.hidden_dim, self.hidden_dim * 4, bias=True)
        
        # Layer normalization for hidden state
        self.h_norm = RMSNorm(self.head_dim)
        
        # Output projection
        self.out_proj = nn.Linear(self.hidden_dim, dim, bias=False)
        
        # Learnable stabilization parameters
        self.register_buffer('eps', torch.tensor(1e-6))
    
    def forward(
        self, 
        x: torch.Tensor,
        cache: Optional[dict] = None
    ) -> torch.Tensor:
        """Forward pass of sLSTM block.
        
        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            cache: Optional cache for inference containing 'h' and 'c'
            
        Returns:
            Output tensor of shape (batch, seq_len, dim)
        """
        batch, seq_len, dim = x.shape
        
        # Input projection
        x_proj = self.in_proj(x)  # (B, L, hidden_dim)
        
        # Reshape for multi-head processing
        x_proj = rearrange(x_proj, 'b l (h d) -> b h l d', h=self.num_heads)
        
        # Gate projections
        gates = self.gate_proj(rearrange(x_proj, 'b h l d -> b l (h d)'))
        gates = rearrange(gates, 'b l (h d) -> b h l d', h=self.num_heads)
        
        # Split into individual gates
        i_gate, f_gate, o_gate, c_tilde = gates.chunk(4, dim=-1)
        
        # Apply activations
        if self.use_exponential_gate:
            i_gate = exponential_activation(i_gate)
            f_gate = torch.sigmoid(f_gate)
        else:
            i_gate = torch.sigmoid(i_gate)
            f_gate = torch.sigmoid(f_gate)
        
        o_gate = torch.sigmoid(o_gate)
        c_tilde = torch.tanh(c_tilde)
        
        # ============================================================
        # PARALLEL SCAN: Compute cell states for all timesteps at once
        # using cumulative forget gate products and weighted input sums.
        #
        # Cell recurrence: c[t] = f[t]*c[t-1] + i[t]*c_tilde[t]
        # This is a first-order linear recurrence solvable via parallel scan.
        # ============================================================
        
        # Compute cumulative forget gate products in log-space
        # log_f_cum[t] = sum(log(f[0..t]))  -> f_cum[t] = prod(f[0..t])
        log_f = torch.log(f_gate.clamp(min=1e-6))        # (B, H, L, D)
        log_f_cum = torch.cumsum(log_f, dim=2)            # (B, H, L, D)
        f_cum = torch.exp(log_f_cum)                      # (B, H, L, D)
        
        # Input contribution weighted by decay to position t:
        # For position t, contribution from s <= t is:
        #   prod(f[s+1..t]) * i[s] * c_tilde[s]
        # = f_cum[t] / f_cum[s] * i[s] * c_tilde[s]
        # = f_cum[t] * (i[s] * c_tilde[s] / f_cum[s])
        #
        # So: c[t] = f_cum[t] * cumsum(i * c_tilde / f_cum, dim=time)
        # (assuming c[0] = 0, no initial cache)
        
        input_contrib = i_gate * c_tilde  # (B, H, L, D)
        weighted_input = input_contrib / f_cum.clamp(min=1e-8)
        weighted_cumsum = torch.cumsum(weighted_input, dim=2)  # (B, H, L, D)
        c_all = f_cum * weighted_cumsum  # (B, H, L, D)
        
        # Handle optional cache for inference
        if cache is not None and 'c' in cache:
            # Add initial state contribution: c_init * f_cum
            c_init = cache['c'].unsqueeze(2)  # (B, H, 1, D)
            c_all = c_all + f_cum * c_init
        
        # Normalize and compute hidden states for all positions
        c_norm = self.h_norm(c_all)  # (B, H, L, D) -- RMSNorm on last dim
        h_out = o_gate * torch.tanh(c_norm)  # (B, H, L, D)
        
        # Reshape back
        h_out = rearrange(h_out, 'b h l d -> b l (h d)')
        
        # Output projection
        output = self.out_proj(h_out)
        
        # Update cache if provided
        if cache is not None:
            cache['h'] = h_out[:, :, -1, :] if h_out.dim() == 4 else None
            cache['c'] = c_all[:, :, -1, :]
        
        return output
    
    def init_cache(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> dict:
        """Initialize cache for inference.
        
        Args:
            batch_size: Batch size
            device: Device to create cache on
            dtype: Data type for cache
            
        Returns:
            Dictionary containing initialized hidden and cell states
        """
        return {
            'h': torch.zeros(batch_size, self.num_heads, self.head_dim,
                           device=device, dtype=dtype),
            'c': torch.zeros(batch_size, self.num_heads, self.head_dim,
                           device=device, dtype=dtype),
        }
