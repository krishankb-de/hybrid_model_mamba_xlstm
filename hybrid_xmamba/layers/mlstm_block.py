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


def _tanh_soft_cap(x: torch.Tensor, cap: float) -> torch.Tensor:
    """Soft-cap via tanh: maps x -> tanh(x/cap)*cap, range (-cap, cap)."""
    return torch.tanh(x / cap) * cap


class mLSTMBlock(nn.Module):
    """mLSTM (matrix LSTM) mixer block.

    Uses exponential gating and matrix-valued hidden states for enhanced expressiveness.

    Args:
        dim: Model dimension
        head_dim: Dimension per attention head
        num_heads: Number of attention heads
        use_tfla: Whether to use Tiled Flash Linear Attention kernel
        proj_factor: Projection factor for input (typically 2)
        gate_soft_cap: tanh soft-cap applied to raw i/f gate pre-activations
        input_gate_bias_init: Initial bias for i_gate_proj (negative → small initial gates)
        forget_gate_bias_init: Initial bias for f_gate_proj
    """

    def __init__(
        self,
        dim: int,
        head_dim: int = 64,
        num_heads: Optional[int] = None,
        use_tfla: bool = True,
        proj_factor: int = 2,
        gate_soft_cap: float = 15.0,
        input_gate_bias_init: float = -10.0,
        forget_gate_bias_init: float = 0.0,
        use_hybrid_norm: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.gate_soft_cap = gate_soft_cap
        self.use_hybrid_norm = use_hybrid_norm

        if num_heads is None:
            self.num_heads = max(1, dim // head_dim)
        else:
            self.num_heads = num_heads

        self.inner_dim = self.num_heads * head_dim
        self.use_tfla = use_tfla
        self.proj_factor = proj_factor

        # Input projections
        self.in_proj = nn.Linear(dim, self.inner_dim * proj_factor, bias=False)

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
        # Phase 4A (HybridNorm): per-projection V pre-mixer norm
        if use_hybrid_norm:
            self.v_norm = RMSNorm(head_dim)
        else:
            self.v_norm = None

        # Output projection
        self.out_proj = nn.Linear(self.inner_dim, dim, bias=False)

        self.register_buffer('eps', torch.tensor(1e-6))

        # Bias init: negative i_gate bias → small initial input gates (prevents overflow)
        nn.init.constant_(self.i_gate_proj.bias, input_gate_bias_init)
        nn.init.constant_(self.f_gate_proj.bias, forget_gate_bias_init)

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[dict] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of mLSTM block.

        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            cache: Optional cache for inference
            cu_seqlens: Optional (B, L) int tensor of per-position doc-ids.
                When provided, splits each batch row into per-doc segments so
                matrix cell state does not cross document boundaries (Phase 6).

        Returns:
            Output tensor of shape (batch, seq_len, dim)
        """
        # Phase 6E: per-segment wrapper to reset C/n/m state at doc boundaries.
        if cu_seqlens is not None:
            return self._forward_segmented(x, cu_seqlens)
        batch, seq_len, dim = x.shape

        # Input projection
        x_proj = self.in_proj(x)  # (B, L, inner_dim * proj_factor)

        if self.proj_factor == 2:
            x_inner, x_gate = x_proj.chunk(2, dim=-1)
        else:
            x_inner = x_proj
            x_gate = x_inner

        # Query, Key, Value projections
        q = self.q_proj(x_inner)
        k = self.k_proj(x_inner)
        v = self.v_proj(x_inner)

        # Reshape for multi-head
        q = rearrange(q, 'b l (h d) -> b h l d', h=self.num_heads)
        k = rearrange(k, 'b l (h d) -> b h l d', h=self.num_heads)
        v = rearrange(v, 'b l (h d) -> b h l d', h=self.num_heads)

        # Normalize queries and keys (and value when HybridNorm enabled)
        q = self.q_norm(q)
        k = self.k_norm(k)
        if self.v_norm is not None:
            v = self.v_norm(v)

        # --- Phase 3A: tanh soft-cap on raw pre-activations before exp/sigmoid ---
        # Bounds logits to (-cap, cap), preventing exp overflow and sigmoid saturation
        i_logit = _tanh_soft_cap(self.i_gate_proj(x_inner), self.gate_soft_cap)
        f_logit = _tanh_soft_cap(self.f_gate_proj(x_inner), self.gate_soft_cap)

        i_gate = exponential_activation(i_logit)      # exp(capped ĩ_t)
        f_gate = torch.sigmoid(f_logit)               # σ(capped f̃_t)
        o_gate = torch.sigmoid(self.o_gate_proj(x_inner))

        # Reshape gates for multi-head
        i_gate = rearrange(i_gate, 'b l (h d) -> b h l d', h=self.num_heads)
        f_gate = rearrange(f_gate, 'b l (h d) -> b h l d', h=self.num_heads)
        o_gate = rearrange(o_gate, 'b l (h d) -> b h l d', h=self.num_heads)

        if self.use_tfla:
            h = apply_tfla(q, k, v, i_gate, f_gate)
        else:
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
        """Sequential reference mLSTM with LSE stabilizer (Phase 3B).

        Implements Algorithm 1 from Beck et al. 2024 with per-dim gates.
        This is a stable but slow reference path used when use_tfla=False.

        Args:
            q: Queries (B, H, L, D)
            k: Keys (B, H, L, D)
            v: Values (B, H, L, D)
            i_gate: Input gates post-exp (B, H, L, D)
            f_gate: Forget gates post-sigmoid (B, H, L, D)

        Returns:
            Output tensor (B, H, L, D)
        """
        B, H, L, D = q.shape

        log_i = torch.log(i_gate.clamp(min=1e-30))   # (B, H, L, D)
        log_f = torch.log(f_gate.clamp(min=1e-30))   # (B, H, L, D) ≤ 0

        # Running state
        C = torch.zeros(B, H, D, D, device=q.device, dtype=q.dtype)
        n = torch.zeros(B, H, D, device=q.device, dtype=q.dtype)
        # m_t: running max in log-space — initialized to -inf so first step uses log_i
        m = torch.full((B, H, D), float('-inf'), device=q.device, dtype=q.dtype)

        outputs = []

        for t in range(L):
            log_i_t = log_i[:, :, t, :]   # (B, H, D)
            log_f_t = log_f[:, :, t, :]   # (B, H, D)
            k_t = k[:, :, t, :]           # (B, H, D)
            v_t = v[:, :, t, :]           # (B, H, D)
            q_t = q[:, :, t, :]           # (B, H, D)

            # m_t = max(f̃_t + m_{t-1}, ĩ_t)
            m_prev = m
            m = torch.max(log_f_t + m_prev, log_i_t)   # (B, H, D)

            # Stabilized gates: always ≤ 1
            i_stab = torch.exp(log_i_t - m)             # (B, H, D)
            f_stab = torch.exp(log_f_t + m_prev - m)    # (B, H, D), = 0 at t=0

            # Matrix cell state update: C[dk, dv] += i_stab[dk] * k[dk] * v[dv]
            C = f_stab.unsqueeze(-1) * C + torch.einsum(
                'bhd,bhe->bhde', i_stab * k_t, v_t
            )
            # Normalizer vector update
            n = f_stab * n + i_stab * k_t

            # Query the cell: h = C^T q (sum over key-dim)
            h_t = torch.einsum('bhde,bhd->bhe', C, q_t)  # (B, H, D)

            # Normalize: denom = max(|n · q|, 1)
            n_dot = (n * q_t).sum(dim=-1, keepdim=True)  # (B, H, 1)
            denom = torch.clamp(n_dot.abs(), min=1.0)
            h_t = h_t / denom

            outputs.append(h_t)

        return torch.stack(outputs, dim=2)  # (B, H, L, D)

    def _forward_segmented(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        """Per-(row, segment) wrapper that resets mLSTM state at doc boundaries.

        Args:
            x: (B, L, D)
            cu_seqlens: (B, L) int tensor; segment id per position.

        Returns:
            Output (B, L, D) with per-segment mLSTM applied independently.
        """
        B, L, _ = x.shape
        out_parts = []
        for b in range(B):
            ids = cu_seqlens[b]
            change = torch.ones(L, dtype=torch.bool, device=ids.device)
            change[1:] = ids[1:] != ids[:-1]
            starts = torch.nonzero(change, as_tuple=False).flatten().tolist()
            starts.append(L)
            row_pieces = []
            for i in range(len(starts) - 1):
                s, e = starts[i], starts[i + 1]
                seg = x[b : b + 1, s:e, :]
                row_pieces.append(self.forward(seg))
            out_parts.append(torch.cat(row_pieces, dim=1))
        return torch.cat(out_parts, dim=0)
