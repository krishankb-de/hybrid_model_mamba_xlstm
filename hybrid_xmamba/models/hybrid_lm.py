"""Hybrid Language Model implementation.

Main model class that assembles the hybrid architecture into a
causal language modeling backbone.
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint
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
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[CausalLMOutput, Tuple[torch.Tensor, ...]]:
        """Forward pass for language modeling.

        Args:
            input_ids: Token IDs (B, L)
            labels: Target token IDs for loss computation (B, L)
            attention_mask: (B, L) — 1 for real tokens, 0 for padding. When
                provided, padded positions are zero-masked at the embedding
                layer so SSM/mLSTM state does not absorb padding. Mamba and
                mLSTM blocks are causal; zero-masking the input is sufficient
                for right-padded sequences (the default).
            output_hidden_states: Whether to return all hidden states
            return_dict: Whether to return CausalLMOutput or tuple

        Returns:
            CausalLMOutput or tuple of (loss, logits, hidden_states)
        """
        # Embedding
        hidden_states = self.embeddings(input_ids)

        # Zero out padded positions so recurrent state stays clean
        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask.to(hidden_states.dtype).unsqueeze(-1)
        
        # Store all hidden states if requested
        all_hidden_states = () if output_hidden_states else None
        
        # Pass through all layers
        use_ckpt = self.config.use_gradient_checkpointing and self.training
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if use_ckpt:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    layer, hidden_states, use_reentrant=False
                )
            else:
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


class HybridTextEncoder(nn.Module):
    """Hybrid Mamba-xLSTM text encoder for contrastive / retrieval tasks.

    Wraps HybridLanguageModel and adds a projection head that maps the
    last-token hidden state to a normalised embedding vector suitable for
    contrastive learning (SimCSE, CLIP-style alignment, etc.).

    The underlying LM head is kept intact so the same checkpoint can be
    used for both language-modelling pretraining and contrastive fine-tuning.

    Args:
        config:         HybridConfig instance (same as HybridLanguageModel).
        embed_dim:      Output embedding dimension (default 512).
        freeze_lm:      If True, freeze all LM backbone weights and only
                        train the projection head (useful for light fine-tuning).
    """

    def __init__(
        self,
        config: HybridConfig,
        embed_dim: int = 512,
        freeze_lm: bool = False,
    ):
        super().__init__()
        self.config = config
        self.embed_dim = embed_dim

        # Full LM backbone — reuse existing class
        self.lm = HybridLanguageModel(config)

        # Projection head: hidden_dim → embed_dim (no bias, standard in CLIP).
        # Dropout between layers provides the stochastic augmentation that
        # SimCSE relies on — without it both forward passes produce identical
        # vectors and InfoNCE loss collapses to zero immediately.
        self.projection_head = nn.Sequential(
            nn.Linear(config.dim, config.dim, bias=False),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(config.dim, embed_dim, bias=False),
        )

        # Learnable temperature for contrastive loss (initialised to ln(1/0.07))
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)

        if freeze_lm:
            for param in self.lm.parameters():
                param.requires_grad = False

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode a batch of token sequences into normalised embeddings.

        Uses mean pooling over all non-padding token positions, which gives
        substantially better sentence representations than last-token pooling
        for similarity and retrieval tasks (BIOSSES, STS-B, etc.).

        Args:
            input_ids: Token IDs (B, L)
            attention_mask: (B, L) 1 for real tokens, 0 for padding. If None,
                all positions are treated as real tokens.

        Returns:
            L2-normalised embeddings (B, embed_dim)
        """
        outputs = self.lm(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden = outputs.hidden_states[-1]   # (B, L, dim)

        # Mean pooling over non-padding positions
        if attention_mask is not None:
            mask = attention_mask.to(last_hidden.dtype).unsqueeze(-1)  # (B, L, 1)
        else:
            mask = torch.ones(
                last_hidden.shape[:2], dtype=last_hidden.dtype, device=last_hidden.device
            ).unsqueeze(-1)
        seq_repr = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # (B, dim)

        projected = self.projection_head(seq_repr)
        return nn.functional.normalize(projected, dim=-1, eps=1e-8)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> CausalLMOutput:
        """Forward pass — delegates to the underlying LM (for LM pretraining).

        Call encode() instead when you need contrastive embeddings.
        """
        return self.lm(
            input_ids,
            labels=labels,
            attention_mask=attention_mask,
            return_dict=return_dict,
        )

    def get_num_params(self, non_embedding: bool = True) -> int:
        return self.lm.get_num_params(non_embedding=non_embedding)
