"""Generation utilities for autoregressive text generation.

Provides efficient text generation with caching support for the hybrid model.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Callable, List
from dataclasses import dataclass


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_new_tokens: int = 100
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    repetition_penalty: float = 1.0
    do_sample: bool = True
    num_beams: int = 1
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None


def top_k_filtering(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Filter logits to keep only top-k values.
    
    Args:
        logits: Logits tensor (B, vocab_size)
        top_k: Number of top values to keep
        
    Returns:
        Filtered logits
    """
    if top_k <= 0:
        return logits
    
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits[indices_to_remove] = float('-inf')
    return logits


def top_p_filtering(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Nucleus filtering: keep tokens with cumulative probability >= top_p.
    
    Args:
        logits: Logits tensor (B, vocab_size)
        top_p: Cumulative probability threshold
        
    Returns:
        Filtered logits
    """
    if top_p >= 1.0:
        return logits
    
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
    # Remove tokens with cumulative probability above threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    # Keep at least one token
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    # Scatter to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    logits[indices_to_remove] = float('-inf')
    
    return logits


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_tokens: torch.Tensor,
    penalty: float
) -> torch.Tensor:
    """Apply repetition penalty to discourage repeated tokens.
    
    Args:
        logits: Current logits (B, vocab_size)
        generated_tokens: Previously generated tokens (B, L)
        penalty: Repetition penalty factor (> 1.0 to discourage)
        
    Returns:
        Penalized logits
    """
    if penalty == 1.0:
        return logits
    
    # Get unique tokens that have been generated
    batch_size = logits.shape[0]
    
    for i in range(batch_size):
        for token in generated_tokens[i].unique():
            # If score < 0, multiply by penalty (make more negative)
            # If score >= 0, divide by penalty (make less positive)
            if logits[i, token] < 0:
                logits[i, token] *= penalty
            else:
                logits[i, token] /= penalty
    
    return logits


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    config: Optional[GenerationConfig] = None,
    stopping_criteria: Optional[Callable] = None,
) -> torch.Tensor:
    """Generate text autoregressively from a prompt.
    
    Args:
        model: The language model
        input_ids: Starting token IDs (B, L)
        config: Generation configuration
        stopping_criteria: Optional function(input_ids) -> bool to stop generation
        
    Returns:
        Generated token IDs (B, L + max_new_tokens)
    """
    if config is None:
        config = GenerationConfig()
    
    model.eval()
    device = input_ids.device
    batch_size = input_ids.shape[0]
    
    # Track generation
    generated = input_ids.clone()
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=device)
    
    for step in range(config.max_new_tokens):
        # Forward pass
        outputs = model(generated, return_dict=True)
        next_token_logits = outputs.logits[:, -1, :]
        
        # Apply temperature
        if config.temperature != 1.0:
            next_token_logits = next_token_logits / config.temperature
        
        # Apply repetition penalty
        if config.repetition_penalty != 1.0:
            next_token_logits = apply_repetition_penalty(
                next_token_logits,
                generated,
                config.repetition_penalty
            )
        
        # Apply filtering
        if config.top_k is not None:
            next_token_logits = top_k_filtering(next_token_logits, config.top_k)
        
        if config.top_p is not None:
            next_token_logits = top_p_filtering(next_token_logits, config.top_p)
        
        # Sample or greedy decode
        if config.do_sample:
            probs = F.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)
        else:
            next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        
        # Handle EOS token
        if config.eos_token_id is not None:
            next_tokens = next_tokens * unfinished_sequences.unsqueeze(-1)
            next_tokens = next_tokens + (1 - unfinished_sequences.unsqueeze(-1)) * config.pad_token_id
            
            # Mark finished sequences
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.squeeze(-1) != config.eos_token_id
            )
        
        # Append to sequence
        generated = torch.cat([generated, next_tokens], dim=-1)
        
        # Check stopping criteria
        if stopping_criteria is not None and stopping_criteria(generated):
            break
        
        # Check if all sequences are finished
        if config.eos_token_id is not None and unfinished_sequences.max() == 0:
            break
    
    return generated


@torch.no_grad()
def generate_with_cache(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    config: Optional[GenerationConfig] = None,
) -> torch.Tensor:
    """Generate with KV caching for faster inference.
    
    Note: This requires the model to support caching, which would need
    additional implementation in the layer modules.
    
    Args:
        model: The language model
        input_ids: Starting token IDs (B, L)
        config: Generation configuration
        
    Returns:
        Generated token IDs (B, L + max_new_tokens)
    """
    # This is a placeholder for cached generation
    # Full implementation would require cache management in each layer
    return generate(model, input_ids, config)


def beam_search(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    num_beams: int = 4,
    max_length: int = 100,
    length_penalty: float = 1.0,
    eos_token_id: Optional[int] = None,
) -> torch.Tensor:
    """Beam search decoding.
    
    Args:
        model: The language model
        input_ids: Starting token IDs (B, L)
        num_beams: Number of beams
        max_length: Maximum generation length
        length_penalty: Length penalty factor
        eos_token_id: End-of-sequence token ID
        
    Returns:
        Best sequence from beam search (B, L')
    """
    # Placeholder for beam search implementation
    # Full implementation is complex and would require significant additional code
    raise NotImplementedError("Beam search not yet implemented")
