"""MambaBlock Implementation - Selective State Space Model

This module implements the Mamba mixer as specified in the research documentation.

Key features (as per spec 5.2):
- Input-dependent projections: Δ, B, C are projected from input x
- 1D causal convolution for local context
- SiLU activation for gated branches
- Selective scan kernel integration
- Fused discretization for efficiency

Location: hybrid_xmamba/layers/mamba_block_v2.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

# Import selective scan kernel interface
try:
    from hybrid_xmamba.kernels.selective_scan.scan_interface import SelectiveScan
    from hybrid_xmamba.kernels.selective_scan.scan_triton import selective_scan_triton
    SCAN_AVAILABLE = True
except ImportError:
    SCAN_AVAILABLE = False
    print("Warning: Selective scan kernel not available, using PyTorch fallback")


class MambaBlock(nn.Module):
    """Mamba block with selective state space model.
    
    This implements the Mamba architecture with input-dependent (selective) SSM.
    
    Key components (as specified in 5.2):
    1. Input-dependent projections: Δ, B, C computed from input x via x_proj
    2. Convolution: 1D causal conv for local context
    3. SiLU activation: For gated branches
    4. Selective scan: Hardware-accelerated SSM kernel
    
    The forward pass computes:
        x, z = split(in_proj(input))           # Project and split
        x = conv1d(x)                          # Local context
        x = silu(x)                            # Activation
        Δ, B, C = x_proj(x)                    # Input-dependent parameters
        y = selective_scan(x, Δ, A, B, C, D)   # SSM with fused discretization
        y = y * silu(z)                        # Gated output
        return out_proj(y)
    
    Args:
        d_model: Model dimension (input/output)
        d_state: State dimension (N in SSM, typically 16)
        d_conv: Convolution kernel size (typically 4)
        expand_factor: Expansion factor for internal dimension (typically 2)
        dt_rank: Rank for Δ projection ("auto" or int)
        dt_min: Minimum Δ value (default 0.001)
        dt_max: Maximum Δ value (default 0.1)
        dt_init: Initialization mode for Δ ("random" or "constant")
        dt_scale: Scale for Δ initialization (default 1.0)
        bias: Whether to use bias in projections (default False)
        conv_bias: Whether to use bias in convolution (default True)
        use_kernel: Whether to use optimized scan kernel (default True)
        
    Example:
        >>> block = MambaBlock(d_model=768, d_state=16, expand_factor=2)
        >>> x = torch.randn(2, 128, 768)
        >>> output = block(x)
        >>> output.shape
        torch.Size([2, 128, 768])
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand_factor: int = 2,
        dt_rank: str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        bias: bool = False,
        conv_bias: bool = True,
        use_kernel: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand_factor = expand_factor
        self.use_kernel = use_kernel and SCAN_AVAILABLE
        
        # Internal dimension (expanded)
        self.d_inner = d_model * expand_factor
        
        # Δ projection rank (auto or manual)
        if dt_rank == "auto":
            self.dt_rank = math.ceil(d_model / 16)
        else:
            self.dt_rank = dt_rank
        
        # ============================================================
        # INPUT PROJECTION (as per spec)
        # ============================================================
        # Project input to expanded dimension, with extra for gating (z)
        # Output: d_inner for SSM input (x) + d_inner for gate (z)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)
        
        # ============================================================
        # CONVOLUTION (as per spec 5.2: "A 1D causal convolution")
        # ============================================================
        # Causal convolution for local context
        # Groups = d_inner means depthwise convolution (each channel independent)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,  # Causal padding
            bias=conv_bias,
        )
        
        # ============================================================
        # INPUT-DEPENDENT PROJECTIONS (as per spec 5.2)
        # ============================================================
        # "The parameters Δ, B, C are projected from the input x via x_proj"
        # x_proj projects to: dt_rank (for Δ) + d_state (for B) + d_state (for C)
        self.x_proj = nn.Linear(
            self.d_inner,
            self.dt_rank + d_state * 2,
            bias=False,
        )
        
        # Project from dt_rank to d_inner for Δ
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        
        # ============================================================
        # SSM PARAMETERS
        # ============================================================
        # A: State transition matrix (continuous time)
        # Initialized as negative for stability
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))  # Store in log space
        
        # D: Skip connection parameter
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # ============================================================
        # OUTPUT PROJECTION
        # ============================================================
        # Project from expanded dimension back to model dimension
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)
        
        # Initialize Δ projection with special initialization
        self._init_dt_proj(dt_min, dt_max, dt_init, dt_scale)
        
    def _init_dt_proj(self, dt_min: float, dt_max: float, dt_init: str, dt_scale: float):
        """Initialize Δ (delta/timestep) projection with specific distribution.
        
        Following Mamba paper initialization strategy.
        """
        # Initialize dt_proj.bias to inverse softplus of uniform[dt_min, dt_max]
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_min)
        
        # Inverse softplus: log(exp(x) - 1)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        
        # Initialize dt_proj.weight
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_scale)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_scale, dt_scale)
        
    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass of Mamba block.
        
        Implements the selective SSM as specified:
        1. Project input and split into SSM input (x) and gate (z)
        2. Apply 1D causal convolution for local context
        3. Compute input-dependent Δ, B, C
        4. Run selective scan with fused discretization
        5. Apply gated output
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            cache: Optional cache for inference (not yet implemented)
        
        Returns:
            output: Output tensor of shape (batch, seq_len, d_model)
            cache: Updated cache (currently None)
        """
        batch_size, seq_len, _ = x.shape
        
        # ============================================================
        # STEP 1: INPUT PROJECTION AND SPLIT
        # ============================================================
        # Project to expanded dimension with gating
        xz = self.in_proj(x)  # (batch, seq_len, 2 * d_inner)
        x, z = xz.chunk(2, dim=-1)
        # x: SSM input, z: gate
        
        # ============================================================
        # STEP 2: CONVOLUTION (as per spec: "1D causal convolution")
        # ============================================================
        # Transpose for Conv1d: (batch, seq_len, d_inner) -> (batch, d_inner, seq_len)
        x = x.transpose(1, 2)
        
        # Apply causal convolution
        x = self.conv1d(x)
        
        # Remove extra padding to maintain causality
        if self.d_conv > 1:
            x = x[:, :, :seq_len]
        
        # Transpose back: (batch, d_inner, seq_len) -> (batch, seq_len, d_inner)
        x = x.transpose(1, 2)
        
        # ============================================================
        # STEP 3: ACTIVATION (as per spec: "SiLU activation")
        # ============================================================
        x = F.silu(x)
        
        # ============================================================
        # STEP 4: INPUT-DEPENDENT PROJECTIONS (as per spec)
        # ============================================================
        # "Parameters Δ, B, C are projected from the input x via x_proj"
        x_proj_out = self.x_proj(x)  # (batch, seq_len, dt_rank + 2*d_state)
        
        # Split into Δ (delta), B, C
        delta, B, C = x_proj_out.split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        
        # Project delta to full dimension
        delta = self.dt_proj(delta)  # (batch, seq_len, d_inner)
        
        # Apply softplus to ensure Δ > 0 (required for stability)
        delta = F.softplus(delta)
        
        # ============================================================
        # STEP 5: SELECTIVE SCAN (SSM COMPUTATION)
        # ============================================================
        # Get A (state transition matrix) from log space
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state), negative for stability
        
        # Run selective scan with fused discretization
        if self.use_kernel and self.training:
            try:
                # Use optimized kernel
                y = selective_scan_triton(x, delta, A, B, C, self.D, return_states=False)
            except Exception as e:
                print(f"Warning: Selective scan kernel failed ({e}), using PyTorch fallback")
                y = self._selective_scan_pytorch(x, delta, A, B, C, self.D)
        else:
            # Use PyTorch fallback
            y = self._selective_scan_pytorch(x, delta, A, B, C, self.D)
        
        # ============================================================
        # STEP 6: GATED OUTPUT (as per spec: "SiLU activation for gated branches")
        # ============================================================
        # Multiply by gated branch with SiLU
        y = y * F.silu(z)
        
        # ============================================================
        # STEP 7: OUTPUT PROJECTION
        # ============================================================
        output = self.out_proj(y)
        
        return output, None
    
    def _selective_scan_pytorch(
        self,
        x: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
    ) -> torch.Tensor:
        """PyTorch fallback for selective scan.
        
        Implements the SSM recurrence:
            h_t = exp(Δ*A) * h_{t-1} + (exp(Δ*A) - 1) / A * B * x_t
            y_t = C * h_t + D * x_t
        
        For production, use the optimized Triton kernel.
        
        Args:
            x: Input (batch, seq_len, d_inner)
            delta: Time steps (batch, seq_len, d_inner)
            A: State transition (d_inner, d_state)
            B: Input-to-state (batch, seq_len, d_state)
            C: State-to-output (batch, seq_len, d_state)
            D: Skip connection (d_inner)
        
        Returns:
            y: Output (batch, seq_len, d_inner)
        """
        batch_size, seq_len, d_inner = x.shape
        d_state = A.shape[1]
        
        # Initialize output
        y = torch.zeros_like(x)
        
        # Process each dimension independently
        for i in range(d_inner):
            # Get parameters for this dimension
            A_i = A[i]  # (d_state,)
            D_i = D[i]  # scalar
            
            # Initialize hidden state
            h = torch.zeros(batch_size, d_state, device=x.device, dtype=x.dtype)
            
            # Sequential scan over time
            for t in range(seq_len):
                # Get inputs at time t
                x_t = x[:, t, i]  # (batch,)
                delta_t = delta[:, t, i]  # (batch,)
                B_t = B[:, t, :]  # (batch, d_state)
                C_t = C[:, t, :]  # (batch, d_state)
                
                # Discretization: A_bar = exp(Δ * A)
                delta_A = delta_t.unsqueeze(-1) * A_i  # (batch, d_state)
                A_bar = torch.exp(delta_A)  # (batch, d_state)
                
                # B_bar = (A_bar - 1) / A * B (with numerical stability)
                # For small |Δ*A|, use Taylor approximation: B_bar ≈ Δ * B
                abs_delta_A = delta_A.abs()
                use_taylor = abs_delta_A < 0.01
                
                B_bar_taylor = delta_t.unsqueeze(-1) * B_t
                B_bar_exact = (A_bar - 1.0) / A_i * B_t
                B_bar = torch.where(use_taylor, B_bar_taylor, B_bar_exact)
                
                # Update hidden state: h = A_bar * h + B_bar * x
                h = A_bar * h + B_bar * x_t.unsqueeze(-1)
                
                # Compute output: y = C * h + D * x
                y_t = torch.sum(C_t * h, dim=-1) + D_i * x_t
                y[:, t, i] = y_t
        
        return y


# Export the block
__all__ = ['MambaBlock']
