"""Causal self-attention mixer (H100_SCALING_PLAN.md Phase 14A-1).

WHY THIS FILE EXISTS
--------------------
The project's central claim is that an attention-free hybrid matches or beats
attention-based transformers at better efficiency. Until now the repository
contained no attention-based transformer to test that against -- every comparison
was to another attention-free variant, an off-the-shelf frozen model, or a
nearest-neighbour control. A supervisor review (2026-09-07) flagged this as the
highest-priority gap: "nothing else matters if this isn't in place."

This block is deliberately a MIXER, not a new model class. Registered as the
`"attention"` layer type, `layer_pattern: ["attention"]` turns the existing
`HybridLanguageModel` into an ordinary pre-norm Transformer decoder, which means
the baseline goes through the *identical* pipeline as the hybrid -- same
embeddings, same MLP, same image-prefix conditioning, same trainer, same beam
search, same scorer. Nothing downstream needs to change, and nothing downstream
can accidentally differ.

DESIGN CHOICES AND WHY (all of these are comparison-fairness decisions)
-----------------------------------------------------------------------
- **RoPE, not learned position embeddings.** The hybrid spends exactly zero
  parameters on positional encoding (its recurrence is inherently ordered). A
  learned table would hand the baseline +0.79M parameters the hybrid never got.
- **No biases on qkv/out.** Matches the MLP already used by `HybridBlock`
  (`bias=False`), so the two architectures differ only where they must.
- **Doc-boundary masking via `cu_seqlens`.** Stage-0 packs multiple documents per
  sequence and the hybrid resets its recurrent state at each boundary. The
  attention analogue is refusing to attend across a boundary. Ignoring
  `cu_seqlens` here would let the baseline read context the hybrid cannot -- a
  silent, uncontrolled advantage in exactly the comparison this block exists to
  make fair.
- **`use_hybrid_norm` honoured as QK-norm.** `norm_topology=hybrid` (HybridNorm)
  pre-norms the mixer's internal projections. For attention the natural analogue
  is normalising Q and K. The 150M baseline config uses `pre_rms` (canonical
  pre-norm, which is what "Transformer baseline" means to a reviewer), but the
  kwarg is implemented rather than silently swallowed so the block behaves
  correctly if it is ever set.

Parameter accounting per block, matching `HybridBlock`'s MLP: 4*dim^2 (qkv+out)
+ 8*dim^2 (MLP) + 2*dim (two RMSNorms) = 12*dim^2 + 2*dim.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from hybrid_xmamba.layers.normalization import RMSNorm


def build_rope_cache(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute rotary position embedding cos/sin tables.

    Args:
        head_dim: Per-head dimension. Must be even.
        max_seq_len: Longest sequence the tables must cover.
        theta: RoPE base frequency.
        device: Optional device for the tables.
        dtype: Table dtype.

    Returns:
        (cos, sin), each of shape (max_seq_len, head_dim).
    """
    if head_dim % 2 != 0:
        raise ValueError("RoPE requires an even head_dim, got %d" % head_dim)

    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)              # (L, head_dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)               # (L, head_dim)
    cos, sin = emb.cos().to(dtype), emb.sin().to(dtype)
    if device is not None:
        cos, sin = cos.to(device), sin.to(device)
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings to a (B, H, L, head_dim) tensor.

    Args:
        x: Query or key tensor (B, num_heads, seq_len, head_dim).
        cos: (seq_len, head_dim) cosine table.
        sin: (seq_len, head_dim) sine table.

    Returns:
        Rotated tensor, same shape as x.
    """
    cos = cos.unsqueeze(0).unsqueeze(0).to(x.dtype)       # (1, 1, L, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0).to(x.dtype)
    half = x.shape[-1] // 2
    x_rot = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
    return x * cos + x_rot * sin


def build_doc_boundary_attn_mask(cu_seqlens: torch.Tensor) -> torch.Tensor:
    """Build an additive causal mask that also blocks cross-document attention.

    The hybrid's Mamba/mLSTM mixers reset their recurrent state when `cu_seqlens`
    marks a new document. Attention's equivalent is to disallow attending to any
    position belonging to a different document. Without this, a packed-sequence
    baseline sees context the hybrid provably cannot.

    Args:
        cu_seqlens: (B, L) integer tensor of per-position document ids.

    Returns:
        (B, 1, L, L) float mask, 0.0 where attention is allowed and -inf where it
        is not. Ready to pass as `attn_mask` to `scaled_dot_product_attention`.
    """
    batch, seq_len = cu_seqlens.shape
    device = cu_seqlens.device

    causal = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).tril()
    same_doc = cu_seqlens.unsqueeze(2) == cu_seqlens.unsqueeze(1)      # (B, L, L)
    allowed = same_doc & causal.unsqueeze(0)                            # (B, L, L)

    mask = torch.zeros(batch, 1, seq_len, seq_len, device=device, dtype=torch.float32)
    return mask.masked_fill(~allowed.unsqueeze(1), float("-inf"))


class AttentionBlock(nn.Module):
    """Multi-head causal self-attention with RoPE — the Transformer-baseline mixer.

    Drop-in replacement for `MambaBlock`/`mLSTMBlock` inside `HybridBlock`: same
    call signature `(x, cache=None, cu_seqlens=None)`, same (B, L, dim) in and out.

    Args:
        dim: Model dimension.
        num_heads: Number of attention heads. Defaults to `dim // head_dim`.
        head_dim: Per-head dimension (default 64, matching the hybrid's mLSTM).
        attn_dropout: Dropout on attention weights (train-time only).
        rope_theta: RoPE base frequency.
        max_position_embeddings: Size of the precomputed RoPE tables.
        use_hybrid_norm: Apply RMSNorm to Q and K (the HybridNorm analogue).
    """

    def __init__(
        self,
        dim: int,
        num_heads: Optional[int] = None,
        head_dim: int = 64,
        attn_dropout: float = 0.0,
        rope_theta: float = 10000.0,
        max_position_embeddings: int = 1024,
        use_hybrid_norm: bool = False,
    ):
        super().__init__()

        if num_heads is None:
            num_heads = max(1, dim // head_dim)
        if dim % num_heads != 0:
            raise ValueError(
                "dim (%d) must be divisible by num_heads (%d)" % (dim, num_heads)
            )

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attn_dropout = attn_dropout
        self.max_position_embeddings = max_position_embeddings

        # bias=False throughout, matching HybridBlock's MLP so the two
        # architectures differ only in the mixer itself.
        self.qkv_proj = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        self.use_hybrid_norm = use_hybrid_norm
        if use_hybrid_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        else:
            self.q_norm = None
            self.k_norm = None

        cos, sin = build_rope_cache(self.head_dim, max_position_embeddings, theta=rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def _rope_tables(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return cos/sin covering `seq_len`, extending the cache if needed."""
        if seq_len > self.rope_cos.shape[0]:
            cos, sin = build_rope_cache(
                self.head_dim, seq_len, device=device, dtype=self.rope_cos.dtype
            )
            self.rope_cos, self.rope_sin = cos, sin
        return self.rope_cos[:seq_len].to(device), self.rope_sin[:seq_len].to(device)

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[dict] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Causal self-attention.

        Args:
            x: (B, L, dim) input.
            cache: Unused. Accepted for signature parity with the other mixers;
                the project's `beam_search_decode` re-runs the full forward at
                every step rather than caching, so attention is correct without
                a KV cache (see Phase 14A-1).
            cu_seqlens: Optional (B, L) document ids. When given, attention is
                blocked across document boundaries, mirroring the hybrid's
                recurrent-state reset.

        Returns:
            (B, L, dim) output.
        """
        batch, seq_len, _ = x.shape

        qkv = self.qkv_proj(x)                                            # (B, L, 3*dim)
        qkv = qkv.view(batch, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)                    # each (B, H, L, hd)

        if self.use_hybrid_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        cos, sin = self._rope_tables(seq_len, x.device)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        dropout_p = self.attn_dropout if self.training else 0.0

        if cu_seqlens is None:
            attn = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=True)
        else:
            # Explicit mask carries causality AND the document boundaries, so
            # is_causal must be False here (they are mutually exclusive).
            mask = build_doc_boundary_attn_mask(cu_seqlens).to(q.dtype)
            attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout_p)

        attn = attn.transpose(1, 2).reshape(batch, seq_len, self.dim)
        return self.out_proj(attn)

    def extra_repr(self) -> str:
        return "dim=%d, num_heads=%d, head_dim=%d, hybrid_norm=%s" % (
            self.dim, self.num_heads, self.head_dim, self.use_hybrid_norm
        )
