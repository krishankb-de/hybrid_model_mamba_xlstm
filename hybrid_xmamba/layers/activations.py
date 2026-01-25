"""Custom activation functions for the hybrid model."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def exponential_activation(x: torch.Tensor) -> torch.Tensor:
    """Exponential activation used in xLSTM (mLSTM).
    
    Args:
        x: Input tensor
        
    Returns:
        Exponentially activated tensor
    """
    return torch.exp(x)


def silu_activation(x: torch.Tensor) -> torch.Tensor:
    """SiLU (Swish) activation function.
    
    Args:
        x: Input tensor
        
    Returns:
        SiLU activated tensor
    """
    return F.silu(x)


def swish_activation(x: torch.Tensor) -> torch.Tensor:
    """Swish activation (alias for SiLU).
    
    Args:
        x: Input tensor
        
    Returns:
        Swish activated tensor
    """
    return x * torch.sigmoid(x)


class ExponentialActivation(nn.Module):
    """Exponential activation module."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return exponential_activation(x)


class SiluActivation(nn.Module):
    """SiLU activation module."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return silu_activation(x)


class SwishActivation(nn.Module):
    """Swish activation module."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return swish_activation(x)
