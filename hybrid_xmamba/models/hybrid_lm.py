"""Hybrid Language Model implementation.

Main model class that assembles the hybrid architecture into a
causal language modeling backbone.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from dataclasses import dataclass

from hybrid_xmamba.models.configuration_hybrid import HybridConfig
from hybrid_xmamba.layers.hybrid_block import create_hybrid_blocks
from hybrid_xmamba.layers.normalization import RMSNorm


@dataclass
class CausalLMOutput:
    """Output type for causal language models."""
    loss: Optional[torch.Tensor] = None
    logits: torch.Tensor = None
    hidden_states: Optional[Tuple[torch.Tensor]] = None
    attentions: Optional[Tuple[torch.Tensor]] = None


class HybridEmbedding(nn.Module):
    """Embedding layer with optional positional encoding.
    
    Args:
        config: Model configuration
    """
    
    def __init__(self, config: HybridConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.dim)
        
        # Optional learned positional embeddings
        # Note: Some architectures (like Mamba) don't need explicit positional embeddings
        self.use_pos_embedding = False
        if self.use_pos_embedding:
            self.position_embedding = nn.Embedding(
                config.max_position_embeddings, 
                config.dim
            )
        
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass for embeddings.
        
        Args:
            input_ids: Token IDs (B, L)
            
        Returns:
            Embedded representations (B, L, D)
        """
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        embeddings = self.token_embedding(input_ids)
        
        # Add positional embeddings if used
        if self.use_pos_embedding:
            position_ids = torch.arange(
                seq_len, 
                dtype=torch.long, 
                device=input_ids.device
            ).unsqueeze(0).expand(batch_size, -1)
            position_embeddings = self.position_embedding(position_ids)
            embeddings = embeddings + position_embeddings
        
        return self.dropout(embeddings)


class HybridLanguageModel(nn.Module):
    """Hybrid Mamba-xLSTM Language Model.
    
    This model combines Mamba and xLSTM layers in a flexible pattern
    for causal language modeling.
    
    Args:
        config: Model configuration
    """
    
    def __init__(self, config: HybridConfig):
        super().__init__()
        self.config = config
        
        # Embedding layer
        self.embeddings = HybridEmbedding(config)
        
        # Create hybrid blocks based on layer pattern
        self.layers = create_hybrid_blocks(
            dim=config.dim,
            num_layers=config.num_layers,
            layer_pattern=config.layer_pattern,
            norm_type=config.norm_type,
            use_mlp=config.use_mlp,
            mlp_ratio=config.mlp_ratio,
            # Mamba params
            state_size=config.state_size,
            conv_size=config.conv_size,
            expand_factor=config.expand_factor,
            dt_rank=config.dt_rank,
            use_fast_path=config.use_fast_path,
            # mLSTM params
            head_dim=config.head_dim,
            num_heads=config.num_heads,
            use_tfla=config.use_tfla,
            proj_factor=config.proj_factor,
            # sLSTM params
            hidden_dim=config.slstm_hidden_dim,
            slstm_num_heads=config.slstm_num_heads,
            use_exponential_gate=config.use_exponential_gate,
        )
        
        # Final normalization
        if config.norm_type.lower() == "rms":
            self.final_norm = RMSNorm(config.dim)
        else:
            self.final_norm = nn.LayerNorm(config.dim)
        
        # Language modeling head
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        
        # Optionally tie weights with embedding
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embeddings.token_embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[CausalLMOutput, Tuple[torch.Tensor, ...]]:
        """Forward pass for language modeling.
        
        Args:
            input_ids: Token IDs (B, L)
            labels: Target token IDs for loss computation (B, L)
            output_hidden_states: Whether to return all hidden states
            return_dict: Whether to return CausalLMOutput or tuple
            
        Returns:
            CausalLMOutput or tuple of (loss, logits, hidden_states)
        """
        # Embedding
        hidden_states = self.embeddings(input_ids)
        
        # Store all hidden states if requested
        all_hidden_states = () if output_hidden_states else None
        
        # Pass through all layers
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            hidden_states = layer(hidden_states)
        
        # Final normalization
        hidden_states = self.final_norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        # Language modeling head
        logits = self.lm_head(hidden_states)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1)
            )
        
        if not return_dict:
            output = (logits,)
            if output_hidden_states:
                output = output + (all_hidden_states,)
            return ((loss,) + output) if loss is not None else output
        
        return CausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=all_hidden_states,
        )
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """Generate text autoregressively.
        
        Args:
            input_ids: Starting token IDs (B, L)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            
        Returns:
            Generated token IDs (B, L + max_new_tokens)
        """
        self.eval()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Forward pass
                outputs = self.forward(input_ids, return_dict=True)
                logits = outputs.logits
                
                # Get logits for last token
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k is not None:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Apply nucleus (top-p) filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample from distribution
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids
    
    def get_num_params(self, non_embedding: bool = True) -> int:
        """Get number of parameters.
        
        Args:
            non_embedding: Whether to exclude embedding parameters
            
        Returns:
            Number of parameters
        """
        n_params = sum(p.numel() for p in self.parameters())
        
        if non_embedding:
            n_params -= self.embeddings.token_embedding.weight.numel()
            if hasattr(self.embeddings, 'position_embedding'):
                n_params -= self.embeddings.position_embedding.weight.numel()
        
        return n_params
    
    def get_layer_types(self) -> list:
        """Get the sequence of layer types in the model.
        
        Returns:
            List of layer type strings
        """
        return [layer.layer_type for layer in self.layers]
