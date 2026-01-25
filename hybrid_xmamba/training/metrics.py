"""Metrics for evaluating hybrid models.

Provides computation of various metrics including perplexity,
MQAR accuracy, and other evaluation metrics.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple
import math


def compute_perplexity(loss: torch.Tensor) -> torch.Tensor:
    """Compute perplexity from cross-entropy loss.
    
    Perplexity = exp(loss)
    
    Args:
        loss: Cross-entropy loss
        
    Returns:
        Perplexity value
    """
    return torch.exp(loss)


def compute_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Compute token-level accuracy.
    
    Args:
        logits: Model logits (B, L, vocab_size)
        labels: Target labels (B, L)
        ignore_index: Index to ignore in accuracy computation
        
    Returns:
        Accuracy as a float tensor
    """
    predictions = torch.argmax(logits, dim=-1)
    
    # Create mask for valid tokens
    mask = labels != ignore_index
    
    # Compute accuracy
    correct = (predictions == labels) & mask
    accuracy = correct.sum().float() / mask.sum().float()
    
    return accuracy


def compute_top_k_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    k: int = 5,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Compute top-k accuracy.
    
    Args:
        logits: Model logits (B, L, vocab_size)
        labels: Target labels (B, L)
        k: Number of top predictions to consider
        ignore_index: Index to ignore in accuracy computation
        
    Returns:
        Top-k accuracy as a float tensor
    """
    # Get top-k predictions
    top_k_preds = torch.topk(logits, k, dim=-1).indices  # (B, L, k)
    
    # Expand labels to match
    labels_expanded = labels.unsqueeze(-1).expand_as(top_k_preds)
    
    # Create mask for valid tokens
    mask = labels != ignore_index
    
    # Check if true label is in top-k
    correct = (top_k_preds == labels_expanded).any(dim=-1) & mask
    accuracy = correct.sum().float() / mask.sum().float()
    
    return accuracy


def compute_mqar_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    query_positions: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute accuracy specifically for MQAR (Multi-Query Associative Recall) task.
    
    MQAR tests the model's ability to recall associations from earlier in the sequence.
    We only evaluate accuracy at query positions.
    
    Args:
        logits: Model logits (B, L, vocab_size)
        labels: Target labels (B, L)
        query_positions: Boolean mask indicating query positions (B, L)
        
    Returns:
        MQAR accuracy as a float tensor
    """
    predictions = torch.argmax(logits, dim=-1)
    
    if query_positions is None:
        # If no query positions provided, use all positions
        query_positions = torch.ones_like(labels, dtype=torch.bool)
    
    # Compute accuracy only at query positions
    correct = (predictions == labels) & query_positions
    accuracy = correct.sum().float() / query_positions.sum().float()
    
    return accuracy


def compute_bits_per_byte(loss: torch.Tensor) -> torch.Tensor:
    """Compute bits per byte (BPB) metric.
    
    Used for evaluating compression and language modeling.
    BPB = loss / ln(2)
    
    Args:
        loss: Cross-entropy loss
        
    Returns:
        Bits per byte
    """
    return loss / math.log(2)


def compute_sequence_level_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Compute sequence-level accuracy (all tokens must be correct).
    
    Args:
        logits: Model logits (B, L, vocab_size)
        labels: Target labels (B, L)
        ignore_index: Index to ignore
        
    Returns:
        Sequence-level accuracy
    """
    predictions = torch.argmax(logits, dim=-1)
    
    # Create mask for valid tokens
    mask = labels != ignore_index
    
    # Check if all tokens in each sequence are correct
    correct_tokens = (predictions == labels) | ~mask
    all_correct = correct_tokens.all(dim=-1)
    
    accuracy = all_correct.float().mean()
    
    return accuracy


class MetricsTracker:
    """Track and aggregate metrics over multiple batches."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all tracked metrics."""
        self.metrics = {}
        self.counts = {}
    
    def update(self, metrics: dict, batch_size: int = 1):
        """Update tracked metrics with new values.
        
        Args:
            metrics: Dictionary of metric names to values
            batch_size: Size of the batch (for averaging)
        """
        for name, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            
            if name not in self.metrics:
                self.metrics[name] = 0.0
                self.counts[name] = 0
            
            self.metrics[name] += value * batch_size
            self.counts[name] += batch_size
    
    def compute(self) -> dict:
        """Compute average of all tracked metrics.
        
        Returns:
            Dictionary of averaged metrics
        """
        return {
            name: self.metrics[name] / self.counts[name]
            for name in self.metrics.keys()
        }
    
    def get(self, name: str) -> float:
        """Get a specific metric value.
        
        Args:
            name: Name of the metric
            
        Returns:
            Averaged metric value
        """
        if name not in self.metrics:
            raise KeyError(f"Metric '{name}' not found")
        return self.metrics[name] / self.counts[name]


def compute_gradient_norm(model: torch.nn.Module) -> float:
    """Compute the L2 norm of model gradients.
    
    Useful for monitoring training stability.
    
    Args:
        model: The model
        
    Returns:
        Gradient norm
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def compute_parameter_norm(model: torch.nn.Module) -> float:
    """Compute the L2 norm of model parameters.
    
    Args:
        model: The model
        
    Returns:
        Parameter norm
    """
    total_norm = 0.0
    for p in model.parameters():
        param_norm = p.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm
