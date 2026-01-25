"""HybridLayer - Unified wrapper for Mamba and mLSTM blocks

This module implements the HybridLayer wrapper that dynamically instantiates
either MambaBlock or mLSTMBlock based on the configuration pattern.

As specified in section 5.3:
- Dynamically selects block type based on pattern
- Applies pre-normalization (RMSNorm)
- Implements residual connection
- Applies dropout for regularization

Location: hybrid_xmamba/layers/hybrid_layer.py
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import warnings

# Import block implementations
try:
    from hybrid_xmamba.layers.mamba_block_v2 import MambaBlock
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    warnings.warn("MambaBlock not available")

try:
    from hybrid_xmamba.layers.mlstm_block_v2 import mLSTMBlock
    MLSTM_AVAILABLE = True
except ImportError:
    MLSTM_AVAILABLE = False
    warnings.warn("mLSTMBlock not available")


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.
    
    More efficient than LayerNorm, used in modern architectures.
    
    RMSNorm(x) = x / RMS(x) * weight
    where RMS(x) = sqrt(mean(x^2) + eps)
    
    Args:
        d_model: Model dimension
        eps: Small constant for numerical stability (default: 1e-5)
    """
    
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMSNorm.
        
        Args:
            x: Input tensor of shape (..., d_model)
        
        Returns:
            Normalized tensor of same shape
        """
        # Compute RMS: sqrt(mean(x^2) + eps)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        
        # Normalize and scale
        x_normed = x / rms
        
        return self.weight * x_normed


class HybridLayer(nn.Module):
    """HybridLayer wrapper for unified Mamba/mLSTM blocks.
    
    As specified in section 5.3:
    - Dynamically instantiates either MambaBlock or mLSTMBlock
    - Pattern example: "mamba,mamba,mamba,mlstm"
    - Applies pre-normalization (RMSNorm)
    - Implements residual connection with dropout
    
    The forward pass computes:
        residual = x
        x = norm(x)              # Pre-normalization
        x = mixer(x)             # Mamba or mLSTM block
        return residual + dropout(x)  # Residual + dropout
    
    Args:
        layer_idx: Index of this layer in the model
        config: Configuration object or dict with attributes:
            - model.hybrid_pattern: Comma-separated pattern (e.g., "mamba,mamba,mlstm")
            - d_model: Model dimension
            - dropout: Dropout probability
            - d_state: State dimension (for Mamba)
            - d_conv: Convolution kernel size (for Mamba)
            - expand_factor: Expansion factor
            - num_heads: Number of heads (for mLSTM)
            - Additional block-specific parameters
    
    Example:
        >>> config = {
        ...     'model': {'hybrid_pattern': 'mamba,mamba,mlstm'},
        ...     'd_model': 768,
        ...     'dropout': 0.1,
        ...     'd_state': 16,
        ...     'expand_factor': 2,
        ...     'num_heads': 4,
        ... }
        >>> layer = HybridLayer(layer_idx=0, config=config)
        >>> x = torch.randn(2, 128, 768)
        >>> output = layer(x)
        >>> output.shape
        torch.Size([2, 128, 768])
    """
    
    def __init__(self, layer_idx: int, config: Dict[str, Any]):
        super().__init__()
        self.layer_idx = layer_idx
        
        # Extract configuration
        if hasattr(config, '__dict__'):
            # Config is an object
            config_dict = config.__dict__
        else:
            # Config is already a dict
            config_dict = config
        
        # Get model configuration
        if 'model' in config_dict and hasattr(config_dict['model'], 'hybrid_pattern'):
            # Nested config with model.hybrid_pattern
            hybrid_pattern = config_dict['model'].hybrid_pattern
        elif 'model' in config_dict and isinstance(config_dict['model'], dict):
            hybrid_pattern = config_dict['model']['hybrid_pattern']
        elif 'hybrid_pattern' in config_dict:
            # Direct config.hybrid_pattern
            hybrid_pattern = config_dict['hybrid_pattern']
        else:
            raise ValueError("Config must have 'hybrid_pattern' (either at top level or in 'model')")
        
        # Parse pattern
        # Pattern example: "mamba,mamba,mamba,mlstm"
        pattern = hybrid_pattern.split(",")
        layer_type = pattern[layer_idx % len(pattern)]
        
        # Get d_model
        d_model = config_dict.get('d_model', config_dict.get('model', {}).get('d_model', 768))
        
        # ============================================================
        # INSTANTIATE MIXER (as per spec 5.3)
        # ============================================================
        if layer_type == "mamba":
            if not MAMBA_AVAILABLE:
                raise ImportError("MambaBlock is not available. Ensure mamba_block_v2.py is present.")
            
            # Instantiate MambaBlock
            self.mixer = MambaBlock(
                d_model=d_model,
                d_state=config_dict.get('d_state', 16),
                d_conv=config_dict.get('d_conv', 4),
                expand_factor=config_dict.get('expand_factor', 2),
                dt_rank=config_dict.get('dt_rank', 'auto'),
                dt_min=config_dict.get('dt_min', 0.001),
                dt_max=config_dict.get('dt_max', 0.1),
                dt_init=config_dict.get('dt_init', 'random'),
                dt_scale=config_dict.get('dt_scale', 1.0),
                bias=config_dict.get('bias', False),
                conv_bias=config_dict.get('conv_bias', True),
                use_kernel=config_dict.get('use_kernel', True),
            )
            self.layer_type = "mamba"
            
        elif layer_type == "mlstm":
            if not MLSTM_AVAILABLE:
                raise ImportError("mLSTMBlock is not available. Ensure mlstm_block_v2.py is present.")
            
            # Instantiate mLSTMBlock
            self.mixer = mLSTMBlock(
                d_model=d_model,
                num_heads=config_dict.get('num_heads', 4),
                expand_factor=config_dict.get('expand_factor', 2),
                use_kernel=config_dict.get('use_kernel', True),
                eps=config_dict.get('eps', 1e-5),
                bias=config_dict.get('bias', False),
            )
            self.layer_type = "mlstm"
            
        else:
            raise ValueError(f"Unknown layer type: {layer_type}. Must be 'mamba' or 'mlstm'")
        
        # ============================================================
        # PRE-NORMALIZATION (as per spec 5.3: "self.norm = RMSNorm")
        # ============================================================
        self.norm = RMSNorm(d_model, eps=config_dict.get('eps', 1e-5))
        
        # ============================================================
        # DROPOUT (as per spec 5.3: "self.dropout = nn.Dropout")
        # ============================================================
        dropout_prob = config_dict.get('dropout', 0.1)
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with pre-norm and residual connection.
        
        As specified in section 5.3:
            residual = x
            x = self.norm(x)
            x = self.mixer(x)
            return residual + self.dropout(x)
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            cache: Optional cache for inference (not yet implemented)
        
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        # ============================================================
        # RESIDUAL CONNECTION (as per spec: "residual = x")
        # ============================================================
        residual = x
        
        # ============================================================
        # PRE-NORMALIZATION (as per spec: "x = self.norm(x)")
        # ============================================================
        x = self.norm(x)
        
        # ============================================================
        # MIXER (as per spec: "x = self.mixer(x)")
        # ============================================================
        if self.layer_type == "mamba":
            # MambaBlock returns (output, cache)
            x, _ = self.mixer(x, cache)
        else:
            # mLSTMBlock returns output only
            x = self.mixer(x)
        
        # ============================================================
        # RESIDUAL + DROPOUT (as per spec: "return residual + self.dropout(x)")
        # ============================================================
        return residual + self.dropout(x)
    
    def __repr__(self) -> str:
        """String representation of the layer."""
        return (
            f"HybridLayer(layer_idx={self.layer_idx}, "
            f"layer_type={self.layer_type}, "
            f"mixer={self.mixer.__class__.__name__})"
        )


class HybridBackbone(nn.Module):
    """Full hybrid model backbone with multiple HybridLayers.
    
    This stacks multiple HybridLayer modules according to the pattern.
    
    Args:
        num_layers: Number of layers
        config: Configuration dict or object
    
    Example:
        >>> config = {
        ...     'model': {'hybrid_pattern': 'mamba,mamba,mlstm'},
        ...     'd_model': 768,
        ...     'dropout': 0.1,
        ...     'd_state': 16,
        ...     'expand_factor': 2,
        ...     'num_heads': 4,
        ... }
        >>> model = HybridBackbone(num_layers=6, config=config)
        >>> x = torch.randn(2, 128, 768)
        >>> output = model(x)
        >>> output.shape
        torch.Size([2, 128, 768])
    """
    
    def __init__(self, num_layers: int, config: Dict[str, Any]):
        super().__init__()
        self.num_layers = num_layers
        
        # Create layers
        self.layers = nn.ModuleList([
            HybridLayer(layer_idx=i, config=config)
            for i in range(num_layers)
        ])
        
        # Final normalization
        d_model = config.get('d_model', config.get('model', {}).get('d_model', 768))
        self.final_norm = RMSNorm(d_model, eps=config.get('eps', 1e-5))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through all layers.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
        
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        # Pass through all layers
        for layer in self.layers:
            x = layer(x)
        
        # Final normalization
        x = self.final_norm(x)
        
        return x
    
    def get_layer_types(self) -> list:
        """Get the type of each layer (for inspection)."""
        return [layer.layer_type for layer in self.layers]


# Export the classes
__all__ = ['RMSNorm', 'HybridLayer', 'HybridBackbone']
