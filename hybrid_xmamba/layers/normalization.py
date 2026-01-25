"""Normalization layers for the hybrid model."""

import torch
import torch.nn as nn
from typing import Optional


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.
    
    Used in Mamba and many modern LLMs for efficient normalization.
    
    Args:
        dim: Dimension of the input features
        eps: Small constant for numerical stability
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMSNorm.
        
        Args:
            x: Input tensor of shape (..., dim)
            
        Returns:
            Normalized tensor of same shape
        """
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x_norm = x / rms
        return self.weight * x_norm


class LayerNormWrapper(nn.Module):
    """Wrapper for PyTorch LayerNorm with consistent interface.
    
    Args:
        dim: Dimension of the input features
        eps: Small constant for numerical stability
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class GroupNormWrapper(nn.Module):
    """Wrapper for PyTorch GroupNorm.
    
    Args:
        num_groups: Number of groups to separate the channels into
        dim: Number of channels
        eps: Small constant for numerical stability
    """
    
    def __init__(self, num_groups: int, dim: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, dim, eps=eps)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply GroupNorm.
        
        Note: Expects input of shape (B, C, ...) where C is dim.
        """
        return self.norm(x)


def get_norm_layer(norm_type: str, dim: int, **kwargs) -> nn.Module:
    """Factory function to get normalization layer.
    
    Args:
        norm_type: Type of normalization ('rms', 'layer', 'group')
        dim: Dimension of features
        **kwargs: Additional arguments for specific norm layers
        
    Returns:
        Normalization module
    """
    norm_type = norm_type.lower()
    
    if norm_type == 'rms' or norm_type == 'rmsnorm':
        return RMSNorm(dim, eps=kwargs.get('eps', 1e-6))
    elif norm_type == 'layer' or norm_type == 'layernorm':
        return LayerNormWrapper(dim, eps=kwargs.get('eps', 1e-6))
    elif norm_type == 'group' or norm_type == 'groupnorm':
        num_groups = kwargs.get('num_groups', 32)
        return GroupNormWrapper(num_groups, dim, eps=kwargs.get('eps', 1e-6))
    else:
        raise ValueError(f"Unknown normalization type: {norm_type}")
