"""Configuration class for Hybrid models.

Defines the configuration schema for hybrid Mamba-xLSTM architectures,
compatible with Hugging Face transformers library conventions.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Literal


@dataclass
class HybridConfig:
    """Configuration for Hybrid Mamba-xLSTM models.
    
    Args:
        vocab_size: Size of vocabulary
        dim: Model dimension (hidden size)
        num_layers: Number of transformer-style blocks
        layer_pattern: Repeating pattern of layer types ['mamba', 'mlstm', 'slstm']
        
        # Mamba specific
        state_size: SSM state dimension for Mamba blocks
        conv_size: Convolution kernel size for Mamba
        expand_factor: Expansion factor for Mamba inner dimension
        
        # xLSTM specific
        head_dim: Dimension per attention head for mLSTM
        num_heads: Number of heads for mLSTM (auto-computed if None)
        slstm_hidden_dim: Hidden dimension for sLSTM
        slstm_num_heads: Number of heads for sLSTM
        
        # Shared parameters
        norm_type: Type of normalization ('rms', 'layer')
        use_mlp: Whether to include MLP after each mixer
        mlp_ratio: Expansion ratio for MLP layers
        
        # Training parameters
        max_position_embeddings: Maximum sequence length
        dropout: Dropout probability
        initializer_range: Range for weight initialization
        
        # Generation parameters
        use_cache: Whether to use caching during generation
        tie_word_embeddings: Whether to tie input/output embeddings
    """
    
    # Architecture
    vocab_size: int = 50257
    dim: int = 768
    num_layers: int = 12
    layer_pattern: List[Literal["mamba", "mlstm", "slstm"]] = field(
        default_factory=lambda: ["mamba", "mamba", "mlstm"]
    )
    
    # Mamba parameters
    state_size: int = 16
    conv_size: int = 4
    expand_factor: int = 2
    dt_rank: Optional[int] = None  # Auto if None
    use_fast_path: bool = True
    
    # mLSTM parameters
    head_dim: int = 64
    num_heads: Optional[int] = None  # Auto-computed
    use_tfla: bool = True
    proj_factor: int = 2
    
    # sLSTM parameters
    slstm_hidden_dim: Optional[int] = None  # Defaults to dim
    slstm_num_heads: int = 4
    use_exponential_gate: bool = True
    
    # Shared parameters
    norm_type: str = "rms"
    use_mlp: bool = True
    mlp_ratio: float = 4.0
    
    # Training parameters
    max_position_embeddings: int = 2048
    dropout: float = 0.1
    initializer_range: float = 0.02
    
    # Generation parameters
    use_cache: bool = True
    tie_word_embeddings: bool = False
    
    # Model type identifier
    model_type: str = "hybrid_xmamba"
    
    def __post_init__(self):
        """Validate and auto-compute derived parameters."""
        # Auto-compute num_heads for mLSTM if not specified
        if self.num_heads is None:
            self.num_heads = max(1, self.dim // self.head_dim)
        
        # Auto-compute dt_rank for Mamba if not specified
        if self.dt_rank is None:
            self.dt_rank = max(1, self.dim // 16)
        
        # Set sLSTM hidden dim if not specified
        if self.slstm_hidden_dim is None:
            self.slstm_hidden_dim = self.dim
        
        # Validate layer pattern
        valid_types = {"mamba", "mlstm", "slstm"}
        for layer_type in self.layer_pattern:
            if layer_type not in valid_types:
                raise ValueError(
                    f"Invalid layer type '{layer_type}'. "
                    f"Must be one of {valid_types}"
                )
    
    def get_layer_config(self, layer_idx: int) -> dict:
        """Get configuration for a specific layer.
        
        Args:
            layer_idx: Index of the layer
            
        Returns:
            Dictionary of configuration parameters for that layer
        """
        layer_type = self.layer_pattern[layer_idx % len(self.layer_pattern)]
        
        base_config = {
            "dim": self.dim,
            "norm_type": self.norm_type,
            "use_mlp": self.use_mlp,
            "mlp_ratio": self.mlp_ratio,
        }
        
        if layer_type == "mamba":
            base_config.update({
                "state_size": self.state_size,
                "conv_size": self.conv_size,
                "expand_factor": self.expand_factor,
                "dt_rank": self.dt_rank,
                "use_fast_path": self.use_fast_path,
            })
        elif layer_type == "mlstm":
            base_config.update({
                "head_dim": self.head_dim,
                "num_heads": self.num_heads,
                "use_tfla": self.use_tfla,
                "proj_factor": self.proj_factor,
            })
        elif layer_type == "slstm":
            base_config.update({
                "hidden_dim": self.slstm_hidden_dim,
                "num_heads": self.slstm_num_heads,
                "use_exponential_gate": self.use_exponential_gate,
            })
        
        return base_config
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "HybridConfig":
        """Create config from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_pretrained(cls, model_name_or_path: str) -> "HybridConfig":
        """Load config from pretrained model (placeholder for HF integration)."""
        # This would integrate with Hugging Face Hub
        raise NotImplementedError("Pretrained model loading not yet implemented")
    
    def save_pretrained(self, save_directory: str):
        """Save config to directory (placeholder for HF integration)."""
        import json
        import os
        
        os.makedirs(save_directory, exist_ok=True)
        config_path = os.path.join(save_directory, "config.json")
        
        with open(config_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
