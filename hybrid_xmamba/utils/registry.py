"""Model registry for dynamic model loading and management.

Provides a centralized registry for different model configurations
and enables easy model creation from config files.
"""

import torch
import torch.nn as nn
from typing import Dict, Type, Optional, Any
from pathlib import Path
import json

from hybrid_xmamba.models.configuration_hybrid import HybridConfig
from hybrid_xmamba.models.hybrid_lm import HybridLanguageModel
from hybrid_xmamba.models.vision_hybrid import VisionHybridModel


class ModelRegistry:
    """Registry for managing model architectures and configurations."""
    
    _models: Dict[str, Type[nn.Module]] = {}
    _configs: Dict[str, HybridConfig] = {}
    
    @classmethod
    def register_model(cls, name: str, model_class: Type[nn.Module]):
        """Register a model class.
        
        Args:
            name: Name identifier for the model
            model_class: Model class to register
        """
        cls._models[name] = model_class
    
    @classmethod
    def register_config(cls, name: str, config: HybridConfig):
        """Register a model configuration.
        
        Args:
            name: Name identifier for the config
            config: Configuration object
        """
        cls._configs[name] = config
    
    @classmethod
    def get_model_class(cls, name: str) -> Type[nn.Module]:
        """Get registered model class.
        
        Args:
            name: Name of the model
            
        Returns:
            Model class
        """
        if name not in cls._models:
            raise ValueError(f"Model '{name}' not found in registry. Available: {list(cls._models.keys())}")
        return cls._models[name]
    
    @classmethod
    def get_config(cls, name: str) -> HybridConfig:
        """Get registered configuration.
        
        Args:
            name: Name of the config
            
        Returns:
            Configuration object
        """
        if name not in cls._configs:
            raise ValueError(f"Config '{name}' not found in registry. Available: {list(cls._configs.keys())}")
        return cls._configs[name]
    
    @classmethod
    def create_model(
        cls, 
        model_name: str,
        config_name: Optional[str] = None,
        config: Optional[HybridConfig] = None,
        **kwargs
    ) -> nn.Module:
        """Create a model instance.
        
        Args:
            model_name: Name of the model class
            config_name: Name of registered config (optional)
            config: Config object (optional, overrides config_name)
            **kwargs: Additional arguments for model initialization
            
        Returns:
            Instantiated model
        """
        model_class = cls.get_model_class(model_name)
        
        if config is None:
            if config_name is not None:
                config = cls.get_config(config_name)
            else:
                config = HybridConfig()
        
        return model_class(config, **kwargs)
    
    @classmethod
    def list_models(cls) -> list:
        """List all registered models."""
        return list(cls._models.keys())
    
    @classmethod
    def list_configs(cls) -> list:
        """List all registered configs."""
        return list(cls._configs.keys())
    
    @classmethod
    def save_config(cls, name: str, path: str):
        """Save a registered config to file.
        
        Args:
            name: Name of the config
            path: Path to save to
        """
        config = cls.get_config(name)
        config.save_pretrained(path)
    
    @classmethod
    def load_config(cls, path: str, name: Optional[str] = None) -> HybridConfig:
        """Load config from file and optionally register it.
        
        Args:
            path: Path to config file
            name: Optional name to register under
            
        Returns:
            Loaded configuration
        """
        config_path = Path(path)
        
        if config_path.is_dir():
            config_path = config_path / "config.json"
        
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        config = HybridConfig.from_dict(config_dict)
        
        if name is not None:
            cls.register_config(name, config)
        
        return config


# Register default models
ModelRegistry.register_model("hybrid_lm", HybridLanguageModel)
ModelRegistry.register_model("vision_hybrid", VisionHybridModel)


# Register default configurations
def create_default_configs():
    """Create and register default model configurations."""
    
    # Small config for debugging (350M params)
    config_350m = HybridConfig(
        vocab_size=50257,
        dim=1024,
        num_layers=24,
        layer_pattern=["mamba", "mamba", "mlstm"],
        state_size=16,
        head_dim=64,
        num_heads=16,
        max_position_embeddings=2048,
    )
    ModelRegistry.register_config("hybrid_350m", config_350m)
    
    # Medium config (1.3B params)
    config_1_3b = HybridConfig(
        vocab_size=50257,
        dim=2048,
        num_layers=24,
        layer_pattern=["mamba", "mamba", "mlstm"],
        state_size=16,
        head_dim=64,
        num_heads=32,
        max_position_embeddings=4096,
    )
    ModelRegistry.register_config("hybrid_1_3b", config_1_3b)
    
    # Large config (7B params)
    config_7b = HybridConfig(
        vocab_size=50257,
        dim=4096,
        num_layers=32,
        layer_pattern=["mamba", "mamba", "mlstm"],
        state_size=16,
        head_dim=128,
        num_heads=32,
        max_position_embeddings=4096,
    )
    ModelRegistry.register_config("hybrid_7b", config_7b)
    
    # Mamba baseline
    config_mamba = HybridConfig(
        vocab_size=50257,
        dim=2048,
        num_layers=48,
        layer_pattern=["mamba"],  # Pure Mamba
        state_size=16,
        max_position_embeddings=4096,
    )
    ModelRegistry.register_config("mamba_baseline", config_mamba)
    
    # xLSTM baseline (pure mLSTM)
    config_xlstm = HybridConfig(
        vocab_size=50257,
        dim=2048,
        num_layers=48,
        layer_pattern=["mlstm"],  # Pure mLSTM
        head_dim=64,
        num_heads=32,
        max_position_embeddings=4096,
    )
    ModelRegistry.register_config("xlstm_baseline", config_xlstm)


# Initialize default configs
create_default_configs()


def get_model_info(model: nn.Module) -> Dict[str, Any]:
    """Get information about a model.
    
    Args:
        model: Model instance
        
    Returns:
        Dictionary with model information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    info = {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "total_parameters_millions": total_params / 1e6,
        "model_class": model.__class__.__name__,
    }
    
    # Add layer information if available
    if hasattr(model, 'get_layer_types'):
        info["layer_types"] = model.get_layer_types()
    
    if hasattr(model, 'config'):
        info["config"] = model.config.to_dict()
    
    return info
