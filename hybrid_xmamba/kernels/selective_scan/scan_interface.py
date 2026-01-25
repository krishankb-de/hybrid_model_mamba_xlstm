"""PyTorch autograd interface for selective scan kernel.

Provides a PyTorch-compatible interface with automatic differentiation
for the selective scan operation used in Mamba.
"""

import torch
from typing import Optional

try:
    from hybrid_xmamba.kernels.selective_scan.scan_triton import selective_scan_triton
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


class SelectiveScanFunction(torch.autograd.Function):
    """Autograd function for selective scan."""
    
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for selective scan.
        
        Args:
            ctx: Context for saving tensors
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
        # Save for backward
        ctx.save_for_backward(x, dt, A, B, C, D, z)
        
        if TRITON_AVAILABLE and x.is_cuda:
            # Use optimized Triton kernel
            y = selective_scan_triton(x, dt, A, B, C, D)
        else:
            # Fallback to PyTorch implementation
            y = selective_scan_pytorch(x, dt, A, B, C, D)
        
        # Apply gating if provided
        if z is not None:
            y = y * z
        
        return y
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Backward pass for selective scan.
        
        Note: This is a simplified implementation. A full production version
        would implement custom backward kernels for efficiency.
        """
        x, dt, A, B, C, D, z = ctx.saved_tensors
        
        # Enable gradients for backward computation
        with torch.enable_grad():
            x_copy = x.detach().requires_grad_(True)
            dt_copy = dt.detach().requires_grad_(True)
            A_copy = A.detach().requires_grad_(True)
            B_copy = B.detach().requires_grad_(True)
            C_copy = C.detach().requires_grad_(True)
            D_copy = D.detach().requires_grad_(True)
            
            # Forward pass
            y = selective_scan_pytorch(x_copy, dt_copy, A_copy, B_copy, C_copy, D_copy)
            
            if z is not None:
                z_copy = z.detach().requires_grad_(True)
                y = y * z_copy
            
            # Backward pass
            y.backward(grad_output)
        
        grad_z = z_copy.grad if z is not None else None
        
        return (
            x_copy.grad,
            dt_copy.grad,
            A_copy.grad,
            B_copy.grad,
            C_copy.grad,
            D_copy.grad,
            grad_z,
        )


def selective_scan_pytorch(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
) -> torch.Tensor:
    """PyTorch reference implementation of selective scan.
    
    Args:
        x: Input (B, L, D)
        dt: Delta values (B, L, D)
        A: State transition (D, N)
        B: Input matrix (B, L, N)
        C: Output matrix (B, L, N)
        D: Skip connection (D,)
        
    Returns:
        Output tensor (B, L, D)
    """
    batch, seq_len, dim = x.shape
    _, _, state_size = B.shape
    
    # Initialize hidden state
    h = torch.zeros(batch, dim, state_size, device=x.device, dtype=x.dtype)
    
    outputs = []
    
    for t in range(seq_len):
        # Get values at timestep t
        x_t = x[:, t, :]  # (B, D)
        dt_t = dt[:, t, :]  # (B, D)
        B_t = B[:, t, :]  # (B, N)
        C_t = C[:, t, :]  # (B, N)
        
        # Discretize A: A_discrete = exp(A * dt)
        A_discrete = torch.exp(A.unsqueeze(0) * dt_t.unsqueeze(-1))  # (B, D, N)
        
        # Update hidden state: h = A * h + dt * x * B
        h = A_discrete * h
        h = h + (dt_t.unsqueeze(-1) * x_t.unsqueeze(-1)) * B_t.unsqueeze(1)
        
        # Compute output: y = C^T * h + D * x
        y_t = torch.sum(h * C_t.unsqueeze(1), dim=-1)  # (B, D)
        y_t = y_t + D.unsqueeze(0) * x_t
        
        outputs.append(y_t)
    
    return torch.stack(outputs, dim=1)


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
    return SelectiveScanFunction.apply(x, dt, A, B, C, D, z)
