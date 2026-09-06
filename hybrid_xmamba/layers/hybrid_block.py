"""Hybrid Block - Unified wrapper for all layer types.

This module provides a unified interface for Mamba, mLSTM, and sLSTM blocks,
allowing flexible interleaving and configuration.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Literal

from hybrid_xmamba.layers.mamba3_block import Mamba3Block
from hybrid_xmamba.layers.mamba_block import MambaBlock
from hybrid_xmamba.layers.mlstm_block import mLSTMBlock
from hybrid_xmamba.layers.slstm_block import sLSTMBlock
from hybrid_xmamba.layers.normalization import RMSNorm


LayerType = Literal["mamba", "mamba3", "mlstm", "slstm"]


class HybridBlock(nn.Module):
    """Unified hybrid block that can instantiate any layer type.
    
    This wrapper provides:
    - Consistent interface across different layer types
    - Pre/post normalization
    - Residual connections
    - Optional MLP layer
    
    Args:
        dim: Model dimension
        layer_type: Type of layer ('mamba', 'mlstm', 'slstm')
        norm_type: Type of normalization ('rms', 'layer')
        use_mlp: Whether to include MLP after the mixer
        mlp_ratio: Expansion ratio for MLP
        **layer_kwargs: Additional arguments for specific layer types
    """
    
    def __init__(
        self,
        dim: int,
        layer_type: LayerType = "mamba",
        norm_type: str = "rms",
        use_mlp: bool = True,
        mlp_ratio: float = 4.0,
        norm_topology: str = "pre_rms",
        is_first_block: bool = False,
        **layer_kwargs
    ):
        super().__init__()
        self.dim = dim
        self.layer_type = layer_type.lower()
        self.use_mlp = use_mlp
        # Phase 4 HybridNorm: route per-projection norms in mixer + FFN post-norm post-residual
        # for blocks >= 1. First block stays pure pre-norm to avoid early-training instability.
        self.norm_topology = norm_topology
        self.is_first_block = is_first_block
        # MAMBA3_PLAN.md M1-F: "hybrid_bc" is "hybrid" minus the Delta norm -- B/C norms only,
        # matching Mamba-3 Sec 3.4. Everything else (FFN post-norm placement, mLSTM projection
        # norms) is identical, so the two topologies differ in exactly one variable. "hybrid"
        # itself is untouched, so every existing checkpoint keeps loading.
        self._use_hybrid_norm = norm_topology in ("hybrid", "hybrid_bc")
        self._ffn_post_norm = self._use_hybrid_norm and not is_first_block
        layer_kwargs = dict(layer_kwargs)
        layer_kwargs["use_hybrid_norm"] = self._use_hybrid_norm
        layer_kwargs.setdefault("use_dt_norm", norm_topology != "hybrid_bc")
        
        # Pre-normalization for mixer
        if norm_type.lower() == "rms":
            self.norm1 = RMSNorm(dim)
        else:
            self.norm1 = nn.LayerNorm(dim)
        
        # MAMBA3_PLAN.md M2-F: a kwarg that names a specific mixer family but matches nothing is
        # always a mistake -- a typo, or a field added to HybridConfig and never wired through.
        # Silently dropping it is how a lever "does nothing" for three days. Unprefixed kwargs are
        # still dropped quietly, because the flat bag deliberately carries every type's fields.
        _PREFIXES = ("mamba3_", "mlstm_", "slstm_")
        _KNOWN = {
            "mamba3_": {"d_state", "head_dim", "expand_factor", "ngroups", "chunk_size",
                        "use_conv", "conv_size", "use_trapezoid", "use_rope", "rope_fraction",
                        "bc_bias", "mimo_rank", "a_mode", "a_floor", "dt_min", "dt_max",
                        "dt_limit", "use_outproj_norm", "use_hybrid_norm"},
            "mlstm_": {"gate_soft_cap", "input_gate_bias_init", "forget_gate_bias_init"},
            "slstm_": {"hidden_dim", "num_heads"},
        }
        for key in layer_kwargs:
            for prefix in _PREFIXES:
                if key.startswith(prefix) and key[len(prefix):] not in _KNOWN[prefix]:
                    raise ValueError(
                        f"unknown mixer option {key!r}: {key[len(prefix):]!r} is not a "
                        f"{prefix.rstrip('_')} parameter. Known: "
                        f"{sorted(_KNOWN[prefix])}"
                    )

        # Filter layer_kwargs based on layer type
        filtered_kwargs = {}
        
        if self.layer_type == "mamba":
            # MambaBlock parameters
            mamba_params = {"state_size", "conv_size", "expand_factor", "dt_rank", "use_fast_path",
                            "use_hybrid_norm", "use_dt_norm", "scan_impl", "dt_init_strategy",
                            "dt_min", "dt_max"}
            filtered_kwargs = {k: v for k, v in layer_kwargs.items() if k in mamba_params}
            self.mixer = MambaBlock(dim, **filtered_kwargs)
        elif self.layer_type == "mamba3":
            # `mamba3_*`-prefixed config keys are stripped here so the block's own signature stays
            # readable and does not collide with the Mamba-1 names (`state_size` vs `d_state`).
            mamba3_params = {
                "d_state", "head_dim", "expand_factor", "ngroups", "chunk_size", "use_conv",
                "conv_size", "use_trapezoid", "use_rope", "rope_fraction", "bc_bias",
                "mimo_rank", "a_mode", "a_floor", "dt_min", "dt_max", "dt_limit",
                "use_outproj_norm", "use_hybrid_norm",
            }
            renamed = {}
            for key, value in layer_kwargs.items():
                stripped = key[len("mamba3_"):] if key.startswith("mamba3_") else key
                if stripped in mamba3_params:
                    renamed[stripped] = value
            # `head_dim` is shared with mLSTM in the flat kwargs bag; the mamba3_ form wins.
            if "mamba3_head_dim" in layer_kwargs:
                renamed["head_dim"] = layer_kwargs["mamba3_head_dim"]
            filtered_kwargs = renamed
            self.mixer = Mamba3Block(dim, **filtered_kwargs)
        elif self.layer_type == "mlstm":
            # mLSTMBlock parameters
            mlstm_params = {"head_dim", "num_heads", "use_tfla", "tfla_impl", "proj_factor",
                            "gate_soft_cap", "input_gate_bias_init", "forget_gate_bias_init",
                            "use_hybrid_norm"}
            filtered_kwargs = {k: v for k, v in layer_kwargs.items() if k in mlstm_params}
            self.mixer = mLSTMBlock(dim, **filtered_kwargs)
        elif self.layer_type == "slstm":
            # sLSTMBlock parameters (uses slstm_* prefix in config)
            slstm_params = {"num_heads", "use_exponential_gate"}
            # Map slstm_* config keys to actual parameter names
            filtered_kwargs = {}
            for k, v in layer_kwargs.items():
                if k.startswith("slstm_"):
                    # Map slstm_hidden_dim -> hidden_dim, slstm_num_heads -> num_heads
                    param_name = k.replace("slstm_", "")
                    if param_name == "hidden_dim":
                        filtered_kwargs["hidden_dim"] = v
                    elif param_name in slstm_params:
                        filtered_kwargs[param_name] = v
                elif k in slstm_params:
                    filtered_kwargs[k] = v
            self.mixer = sLSTMBlock(dim, **filtered_kwargs)
        else:
            raise ValueError(f"Unknown layer type: {layer_type}")

        # MAMBA3_PLAN.md M2-E. This used to be `if layer_type in ("mamba", "mlstm")`, which is how
        # sLSTM blocks silently leaked recurrent state across packed documents: sLSTM fell to the
        # else branch and never received cu_seqlens. Dispatch on a capability the mixer declares,
        # so a new layer type cannot inherit that bug by omission. A parity test cross-checks the
        # attribute against the real forward signature, so the two cannot drift.
        self._mixer_takes_cu_seqlens = bool(getattr(self.mixer, "supports_cu_seqlens", False))
        
        # Optional MLP (feed-forward network)
        if use_mlp:
            if norm_type.lower() == "rms":
                self.norm2 = RMSNorm(dim)
            else:
                self.norm2 = nn.LayerNorm(dim)
            
            mlp_dim = int(dim * mlp_ratio)
            self.mlp = nn.Sequential(
                nn.Linear(dim, mlp_dim, bias=False),
                nn.GELU(),
                nn.Linear(mlp_dim, dim, bias=False),
            )
        else:
            self.norm2 = None
            self.mlp = None
    
    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[dict] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with residual connections.

        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            cache: Optional cache for inference
            cu_seqlens: Optional (B, L) int tensor of per-position doc-ids for
                cross-document boundary resets (Phase 6). Only Mamba/mLSTM mixers
                consume it; sLSTM passes through unchanged.

        Returns:
            Output tensor of shape (batch, seq_len, dim)
        """
        # Mixer with residual
        residual = x
        x = self.norm1(x)
        if self._mixer_takes_cu_seqlens:
            x = self.mixer(x, cache=cache, cu_seqlens=cu_seqlens)
        else:
            x = self.mixer(x, cache=cache)
        x = residual + x
        
        # MLP with residual (if enabled).
        # Phase 4D (HybridNorm): when enabled and not first block, FFN normalizes
        # post-residual: x = norm2(x + mlp(x)). First block stays pure pre-norm.
        if self.use_mlp:
            if self._ffn_post_norm:
                x = self.norm2(x + self.mlp(x))
            else:
                residual = x
                x = self.norm2(x)
                x = self.mlp(x)
                x = residual + x

        return x
    
    def get_layer_info(self) -> dict:
        """Get information about this layer.
        
        Returns:
            Dictionary with layer type and configuration
        """
        return {
            "layer_type": self.layer_type,
            "dim": self.dim,
            "use_mlp": self.use_mlp,
            "mixer_class": self.mixer.__class__.__name__,
        }


def create_hybrid_blocks(
    dim: int,
    num_layers: int,
    layer_pattern: List[LayerType],
    norm_topology: str = "pre_rms",
    **kwargs
) -> nn.ModuleList:
    """Factory function to create a sequence of hybrid blocks.
    
    Args:
        dim: Model dimension
        num_layers: Total number of layers
        layer_pattern: Pattern of layer types to repeat (e.g., ['mamba', 'mlstm'])
        **kwargs: Additional arguments passed to HybridBlock
        
    Returns:
        ModuleList of HybridBlock instances
    
    Example:
        >>> blocks = create_hybrid_blocks(
        ...     dim=768,
        ...     num_layers=12,
        ...     layer_pattern=['mamba', 'mamba', 'mlstm'],
        ... )
        # Creates 12 layers: [M, M, mL, M, M, mL, M, M, mL, M, M, mL]
    """
    blocks = nn.ModuleList()
    
    for i in range(num_layers):
        # Cycle through the pattern
        layer_type = layer_pattern[i % len(layer_pattern)]
        
        block = HybridBlock(
            dim=dim,
            layer_type=layer_type,
            norm_topology=norm_topology,
            is_first_block=(i == 0),
            **kwargs
        )
        blocks.append(block)
    
    return blocks
