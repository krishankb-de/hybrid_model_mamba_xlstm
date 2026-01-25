"""Weight initialization utilities.

Provides initialization schemes tailored for Mamba and xLSTM architectures.
"""

import torch
import torch.nn as nn
import math
from typing import Optional


def init_mamba_weights(module: nn.Module, initializer_range: float = 0.02):
    """Initialize weights for Mamba blocks.
    
    Uses specialized initialization for SSM parameters as described in the Mamba paper.
    
    Args:
        module: Module to initialize
        initializer_range: Standard deviation for normal initialization
    """
    if isinstance(module, nn.Linear):
        # Standard linear layer initialization
        nn.init.normal_(module.weight, mean=0.0, std=initializer_range)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    
    elif isinstance(module, nn.Conv1d):
        # Convolution initialization
        nn.init.normal_(module.weight, mean=0.0, std=initializer_range)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=initializer_range)


def init_xlstm_weights(module: nn.Module, initializer_range: float = 0.02):
    """Initialize weights for xLSTM blocks (mLSTM/sLSTM).
    
    Uses initialization schemes from the xLSTM paper.
    
    Args:
        module: Module to initialize
        initializer_range: Standard deviation for normal initialization
    """
    if isinstance(module, nn.Linear):
        # Xavier/Glorot initialization for xLSTM
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=initializer_range)


def init_weights(
    model: nn.Module,
    method: str = "hybrid",
    initializer_range: float = 0.02,
):
    """Initialize model weights based on layer type.
    
    Args:
        model: The model to initialize
        method: Initialization method ('hybrid', 'mamba', 'xlstm', 'standard')
        initializer_range: Standard deviation for normal initialization
    """
    def _init_module(module):
        # Get module's layer type if it's a hybrid block
        layer_type = None
        if hasattr(module, 'layer_type'):
            layer_type = module.layer_type
        
        # Apply initialization based on method and layer type
        if method == "hybrid":
            if layer_type == "mamba":
                init_mamba_weights(module, initializer_range)
            elif layer_type in ["mlstm", "slstm"]:
                init_xlstm_weights(module, initializer_range)
            else:
                # Default initialization
                standard_init(module, initializer_range)
        
        elif method == "mamba":
            init_mamba_weights(module, initializer_range)
        
        elif method == "xlstm":
            init_xlstm_weights(module, initializer_range)
        
        else:  # standard
            standard_init(module, initializer_range)
    
    model.apply(_init_module)


def standard_init(module: nn.Module, initializer_range: float = 0.02):
    """Standard initialization for neural networks.
    
    Args:
        module: Module to initialize
        initializer_range: Standard deviation for normal initialization
    """
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=initializer_range)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=initializer_range)
    
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    
    elif isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def init_ssm_parameters(
    A: nn.Parameter,
    dt_proj_bias: Optional[nn.Parameter] = None,
    dt_init: str = "random",
    dt_scale: float = 1.0,
    dt_min: float = 0.001,
    dt_max: float = 0.1,
):
    """Initialize SSM-specific parameters for Mamba.
    
    Args:
        A: State transition parameter
        dt_proj_bias: Delta projection bias
        dt_init: Initialization method for dt ('random', 'constant')
        dt_scale: Scaling factor for dt
        dt_min: Minimum dt value
        dt_max: Maximum dt value
    """
    # Initialize A (typically log-uniform in state space)
    with torch.no_grad():
        # A should be negative for stability
        A.uniform_(-4.0, -1.0)
    
    # Initialize dt projection bias if provided
    if dt_proj_bias is not None:
        with torch.no_grad():
            if dt_init == "constant":
                dt = torch.exp(
                    torch.rand(dt_proj_bias.shape) * (math.log(dt_max) - math.log(dt_min))
                    + math.log(dt_min)
                )
            else:
                dt = torch.clamp(torch.rand(dt_proj_bias.shape) * dt_scale, dt_min, dt_max)
            
            # Inverse of softplus
            dt_proj_bias.copy_(torch.log(torch.exp(dt) - 1))


def init_lstm_gates(
    gate_weights: nn.Parameter,
    gate_type: str = "forget",
    init_value: float = 1.0,
):
    """Initialize LSTM gate biases to specific values.
    
    Common practice: Initialize forget gate bias to 1.0 to help learning.
    
    Args:
        gate_weights: Gate weight parameter
        gate_type: Type of gate ('input', 'forget', 'output', 'cell')
        init_value: Initial bias value
    """
    with torch.no_grad():
        if gate_type == "forget":
            # Forget gate bias initialized to positive value
            gate_weights.fill_(init_value)
        else:
            # Other gates initialized to zero
            gate_weights.zero_()


def reinitialize_layer(layer: nn.Module, method: str = "standard"):
    """Reinitialize a specific layer.
    
    Useful for fine-tuning or curriculum learning strategies.
    
    Args:
        layer: Layer to reinitialize
        method: Initialization method
    """
    init_weights(layer, method=method)
