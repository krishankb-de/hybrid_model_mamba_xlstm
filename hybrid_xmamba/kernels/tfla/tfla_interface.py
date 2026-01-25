"""PyTorch autograd interface for TFLA kernel.

This module provides a PyTorch-compatible interface with automatic differentiation
support for the Tiled Flash Linear Attention kernel.
"""

import torch
import torch.nn.functional as F
from typing import Optional

try:
    from hybrid_xmamba.kernels.tfla.tfla_triton import tfla_forward_triton
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


class TFLAFunction(torch.autograd.Function):
    """Autograd function for TFLA kernel."""
    
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        i_gate: torch.Tensor,
        f_gate: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass using Triton kernel or fallback.
        
        Args:
            ctx: Context for saving tensors
            q: Queries (B, H, L, D)
            k: Keys (B, H, L, D)
            v: Values (B, H, L, D)
            i_gate: Input gates (B, H, L, D)
            f_gate: Forget gates (B, H, L, D)
            
        Returns:
            Output tensor (B, H, L, D)
        """
        # Save for backward
        ctx.save_for_backward(q, k, v, i_gate, f_gate)
        
        if TRITON_AVAILABLE and q.is_cuda:
            # Use optimized Triton kernel
            return tfla_forward_triton(q, k, v, i_gate, f_gate)
        else:
            # Fallback to PyTorch implementation
            return tfla_forward_pytorch(q, k, v, i_gate, f_gate)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Backward pass (simplified - full implementation would be more complex).
        
        For a production implementation, this would include proper gradients
        with respect to all inputs. Here we provide a placeholder that uses
        PyTorch autograd.
        """
        q, k, v, i_gate, f_gate = ctx.saved_tensors
        
        # Enable gradients for backward pass
        with torch.enable_grad():
            q_copy = q.detach().requires_grad_(True)
            k_copy = k.detach().requires_grad_(True)
            v_copy = v.detach().requires_grad_(True)
            i_gate_copy = i_gate.detach().requires_grad_(True)
            f_gate_copy = f_gate.detach().requires_grad_(True)
            
            # Forward pass for gradient computation
            output = tfla_forward_pytorch(q_copy, k_copy, v_copy, i_gate_copy, f_gate_copy)
            
            # Backward pass
            output.backward(grad_output)
            
        return (
            q_copy.grad,
            k_copy.grad,
            v_copy.grad,
            i_gate_copy.grad,
            f_gate_copy.grad,
        )


def tfla_forward_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    i_gate: torch.Tensor,
    f_gate: torch.Tensor,
) -> torch.Tensor:
    """PyTorch reference implementation of TFLA.
    
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
    device = q.device
    dtype = q.dtype
    
    # Initialize cell state
    C = torch.zeros(batch, num_heads, head_dim, head_dim, device=device, dtype=dtype)
    n = torch.zeros(batch, num_heads, head_dim, device=device, dtype=dtype)
    
    outputs = []
    
    for t in range(seq_len):
        # Get timestep data
        q_t = q[:, :, t, :]  # (B, H, D)
        k_t = k[:, :, t, :]  # (B, H, D)
        v_t = v[:, :, t, :]  # (B, H, D)
        i_t = i_gate[:, :, t, :]  # (B, H, D)
        f_t = f_gate[:, :, t, :]  # (B, H, D)
        
        # Apply forget gate
        C = f_t.unsqueeze(-1) * C
        n = f_t * n
        
        # Add new information with input gate
        # Outer product k @ v^T
        kv = torch.einsum('bhd,bhe->bhde', k_t, v_t)
        C = C + i_t.unsqueeze(-1) * kv
        n = n + i_t * k_t
        
        # Compute output
        h_t_num = torch.einsum('bhd,bhde->bhe', q_t, C)
        h_t_den = torch.einsum('bhd,bhd->bh', q_t, n).unsqueeze(-1)
        h_t_den = torch.clamp(h_t_den, min=1.0)
        
        h_t = h_t_num / h_t_den
        outputs.append(h_t)
    
    return torch.stack(outputs, dim=2)


def apply_tfla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    i_gate: torch.Tensor,
    f_gate: torch.Tensor,
) -> torch.Tensor:
    """Apply Tiled Flash Linear Attention.
    
    Public interface for TFLA computation with automatic kernel selection.
    
    Args:
        q: Queries (B, H, L, D)
        k: Keys (B, H, L, D)
        v: Values (B, H, L, D)
        i_gate: Input gates (B, H, L, D)
        f_gate: Forget gates (B, H, L, D)
        
    Returns:
        Output tensor (B, H, L, D)
    """
    return TFLAFunction.apply(q, k, v, i_gate, f_gate)
