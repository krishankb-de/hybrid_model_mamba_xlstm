"""Hybrid Block - Unified wrapper for all layer types.

This module provides a unified interface for Mamba, mLSTM, and sLSTM blocks,
allowing flexible interleaving and configuration.
"""

import torch
import torch.nn as nn
from typing import Optional, Literal

from hybrid_xmamba.layers.mamba_block import MambaBlock
from hybrid_xmamba.layers.mlstm_block import mLSTMBlock
from hybrid_xmamba.layers.slstm_block import sLSTMBlock
from hybrid_xmamba.layers.normalization import RMSNorm


LayerType = Literal["mamba", "mlstm", "slstm"]


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
        **layer_kwargs
    ):
        super().__init__()
        self.dim = dim
        self.layer_type = layer_type.lower()
        self.use_mlp = use_mlp
        
        # Pre-normalization for mixer
        if norm_type.lower() == "rms":
            self.norm1 = RMSNorm(dim)
        else:
            self.norm1 = nn.LayerNorm(dim)
        
        # Instantiate the appropriate mixer
        if self.layer_type == "mamba":
            self.mixer = MambaBlock(dim, **layer_kwargs)
        elif self.layer_type == "mlstm":
            self.mixer = mLSTMBlock(dim, **layer_kwargs)
        elif self.layer_type == "slstm":
            self.mixer = sLSTMBlock(dim, **layer_kwargs)
        else:
            raise ValueError(f"Unknown layer type: {layer_type}")
        
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
        cache: Optional[dict] = None
    ) -> torch.Tensor:
        """Forward pass with residual connections.
        
        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            cache: Optional cache for inference
            
        Returns:
            Output tensor of shape (batch, seq_len, dim)
        """
        # Mixer with residual
        residual = x
        x = self.norm1(x)
        x = self.mixer(x, cache=cache)
        x = residual + x
        
        # MLP with residual (if enabled)
        if self.use_mlp:
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
    layer_pattern: list[LayerType],
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
            **kwargs
        )
        blocks.append(block)
    
    return blocks
