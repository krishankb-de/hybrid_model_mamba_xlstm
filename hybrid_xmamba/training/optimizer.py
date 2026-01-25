"""Optimizer configuration utilities.

Provides functions to configure optimizers with proper parameter grouping,
weight decay exclusions, and learning rate scaling.
"""

import torch
from torch.optim import Optimizer, Adam, AdamW, SGD
from typing import Optional, List, Dict, Any
import re


def get_parameter_groups(
    model: torch.nn.Module,
    weight_decay: float = 0.1,
    no_decay_bias: bool = True,
    no_decay_norm: bool = True,
    no_decay_embedding: bool = True,
) -> List[Dict[str, Any]]:
    """Organize model parameters into groups with different weight decay settings.
    
    Args:
        model: The model
        weight_decay: Weight decay value
        no_decay_bias: Don't apply weight decay to bias terms
        no_decay_norm: Don't apply weight decay to normalization layers
        no_decay_embedding: Don't apply weight decay to embeddings
        
    Returns:
        List of parameter groups
    """
    # Parameters that should not have weight decay
    no_decay_names = set()
    
    if no_decay_bias:
        no_decay_names.update(['bias'])
    
    if no_decay_norm:
        no_decay_names.update(['norm', 'ln', 'layernorm', 'layer_norm', 'rmsnorm'])
    
    if no_decay_embedding:
        no_decay_names.update(['embedding', 'embed', 'pos_embed', 'token_embedding'])
    
    # Organize parameters
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Check if parameter should have weight decay
        should_decay = True
        name_lower = name.lower()
        
        for no_decay_name in no_decay_names:
            if no_decay_name in name_lower:
                should_decay = False
                break
        
        if should_decay:
            decay_params.append(param)
        else:
            no_decay_params.append(param)
    
    param_groups = [
        {
            'params': decay_params,
            'weight_decay': weight_decay,
        },
        {
            'params': no_decay_params,
            'weight_decay': 0.0,
        }
    ]
    
    return param_groups


def configure_optimizer(
    model: torch.nn.Module,
    optimizer_name: str = "adamw",
    learning_rate: float = 3e-4,
    weight_decay: float = 0.1,
    betas: tuple = (0.9, 0.999),
    eps: float = 1e-8,
    momentum: float = 0.9,
    fused: bool = False,
    **kwargs
) -> Optimizer:
    """Configure optimizer with proper parameter grouping.
    
    Args:
        model: The model to optimize
        optimizer_name: Name of optimizer ('adamw', 'adam', 'sgd')
        learning_rate: Learning rate
        weight_decay: Weight decay
        betas: Beta parameters for Adam-based optimizers
        eps: Epsilon for numerical stability
        momentum: Momentum for SGD
        fused: Use fused optimizer implementation if available
        **kwargs: Additional optimizer arguments
        
    Returns:
        Configured optimizer
    """
    # Get parameter groups
    param_groups = get_parameter_groups(model, weight_decay=weight_decay)
    
    # Log parameter counts
    decay_params = sum(p.numel() for p in param_groups[0]['params'])
    no_decay_params = sum(p.numel() for p in param_groups[1]['params'])
    print(f"Parameters with weight decay: {decay_params:,}")
    print(f"Parameters without weight decay: {no_decay_params:,}")
    
    # Create optimizer
    optimizer_name = optimizer_name.lower()
    
    if optimizer_name == "adamw":
        optimizer = AdamW(
            param_groups,
            lr=learning_rate,
            betas=betas,
            eps=eps,
            fused=fused and torch.cuda.is_available(),
            **kwargs
        )
    
    elif optimizer_name == "adam":
        optimizer = Adam(
            param_groups,
            lr=learning_rate,
            betas=betas,
            eps=eps,
            fused=fused and torch.cuda.is_available(),
            **kwargs
        )
    
    elif optimizer_name == "sgd":
        optimizer = SGD(
            param_groups,
            lr=learning_rate,
            momentum=momentum,
            **kwargs
        )
    
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    return optimizer


def scale_learning_rate(
    base_lr: float,
    batch_size: int,
    base_batch_size: int = 256,
    scaling_rule: str = "linear",
) -> float:
    """Scale learning rate based on batch size.
    
    Common practice when using larger batch sizes for distributed training.
    
    Args:
        base_lr: Base learning rate (for base_batch_size)
        batch_size: Actual batch size
        base_batch_size: Base batch size that base_lr was tuned for
        scaling_rule: How to scale ('linear', 'sqrt')
        
    Returns:
        Scaled learning rate
    """
    ratio = batch_size / base_batch_size
    
    if scaling_rule == "linear":
        return base_lr * ratio
    elif scaling_rule == "sqrt":
        return base_lr * (ratio ** 0.5)
    else:
        return base_lr


def get_layer_wise_lr_decay(
    model: torch.nn.Module,
    base_lr: float,
    decay_rate: float = 0.9,
    num_layers: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Get parameter groups with layer-wise learning rate decay.
    
    Lower layers get smaller learning rates (useful for fine-tuning).
    
    Args:
        model: The model
        base_lr: Base learning rate for the top layer
        decay_rate: Decay rate per layer
        num_layers: Number of layers (auto-detected if None)
        
    Returns:
        List of parameter groups with different learning rates
    """
    if num_layers is None:
        if hasattr(model, 'layers'):
            num_layers = len(model.layers)
        else:
            num_layers = 12  # Default
    
    # Group parameters by layer
    layer_params = {i: [] for i in range(num_layers)}
    other_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Try to extract layer number from parameter name
        match = re.search(r'layers?\.(\d+)', name)
        if match:
            layer_idx = int(match.group(1))
            if layer_idx < num_layers:
                layer_params[layer_idx].append(param)
            else:
                other_params.append(param)
        else:
            other_params.append(param)
    
    # Create parameter groups with decayed learning rates
    param_groups = []
    
    for layer_idx in range(num_layers):
        if layer_params[layer_idx]:
            lr = base_lr * (decay_rate ** (num_layers - layer_idx - 1))
            param_groups.append({
                'params': layer_params[layer_idx],
                'lr': lr,
            })
    
    # Other parameters use base learning rate
    if other_params:
        param_groups.append({
            'params': other_params,
            'lr': base_lr,
        })
    
    return param_groups


def configure_8bit_optimizer(
    model: torch.nn.Module,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.1,
) -> Optimizer:
    """Configure 8-bit AdamW optimizer for memory efficiency.
    
    Requires: pip install bitsandbytes
    
    Args:
        model: The model to optimize
        learning_rate: Learning rate
        weight_decay: Weight decay
        
    Returns:
        8-bit optimizer
    """
    try:
        import bitsandbytes as bnb
    except ImportError:
        raise ImportError("bitsandbytes not installed. Install with: pip install bitsandbytes")
    
    param_groups = get_parameter_groups(model, weight_decay=weight_decay)
    
    optimizer = bnb.optim.AdamW8bit(
        param_groups,
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    
    return optimizer
