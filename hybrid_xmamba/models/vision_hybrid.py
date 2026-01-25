"""Vision backbone using hybrid architecture.

Implements a vision model using the hybrid Mamba-xLSTM architecture
for image classification or feature extraction.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from einops import rearrange

from hybrid_xmamba.models.configuration_hybrid import HybridConfig
from hybrid_xmamba.layers.hybrid_block import create_hybrid_blocks
from hybrid_xmamba.layers.normalization import RMSNorm


class PatchEmbedding(nn.Module):
    """Convert images to patch embeddings.
    
    Args:
        img_size: Input image size (assumes square)
        patch_size: Size of each patch
        in_channels: Number of input channels (3 for RGB)
        embed_dim: Embedding dimension
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Convolutional projection
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Images (B, C, H, W)
            
        Returns:
            Patch embeddings (B, num_patches, embed_dim)
        """
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x


class VisionHybridModel(nn.Module):
    """Hybrid vision model for image classification.
    
    Uses patch embeddings and processes them through hybrid layers.
    Can be used for classification or as a feature extractor.
    
    Args:
        config: Model configuration
        img_size: Input image size
        patch_size: Patch size for tokenization
        num_classes: Number of output classes (0 for feature extraction)
        in_channels: Number of input image channels
    """
    
    def __init__(
        self,
        config: HybridConfig,
        img_size: int = 224,
        patch_size: int = 16,
        num_classes: int = 1000,
        in_channels: int = 3,
    ):
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=config.dim,
        )
        num_patches = self.patch_embed.num_patches
        
        # Learnable class token (optional)
        self.use_cls_token = True
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.dim))
            num_patches += 1
        
        # Positional embeddings (optional for vision)
        self.use_pos_embed = True
        if self.use_pos_embed:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, config.dim)
            )
        
        self.dropout = nn.Dropout(config.dropout)
        
        # Hybrid blocks
        self.layers = create_hybrid_blocks(
            dim=config.dim,
            num_layers=config.num_layers,
            layer_pattern=config.layer_pattern,
            norm_type=config.norm_type,
            use_mlp=config.use_mlp,
            mlp_ratio=config.mlp_ratio,
            state_size=config.state_size,
            conv_size=config.conv_size,
            expand_factor=config.expand_factor,
            dt_rank=config.dt_rank,
            use_fast_path=config.use_fast_path,
            head_dim=config.head_dim,
            num_heads=config.num_heads,
            use_tfla=config.use_tfla,
            proj_factor=config.proj_factor,
            hidden_dim=config.slstm_hidden_dim,
            slstm_num_heads=config.slstm_num_heads,
            use_exponential_gate=config.use_exponential_gate,
        )
        
        # Final normalization
        if config.norm_type.lower() == "rms":
            self.norm = RMSNorm(config.dim)
        else:
            self.norm = nn.LayerNorm(config.dim)
        
        # Classification head
        if num_classes > 0:
            self.head = nn.Linear(config.dim, num_classes)
        else:
            self.head = nn.Identity()
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            torch.nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm) or isinstance(module, RMSNorm):
            if hasattr(module, 'weight'):
                torch.nn.init.ones_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from images.
        
        Args:
            x: Images (B, C, H, W)
            
        Returns:
            Features (B, num_patches, dim) or (B, dim) if using cls_token
        """
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, dim)
        
        # Add class token if used
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional embeddings
        if self.use_pos_embed:
            x = x + self.pos_embed
        
        x = self.dropout(x)
        
        # Pass through layers
        for layer in self.layers:
            x = layer(x)
        
        # Final norm
        x = self.norm(x)
        
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for classification.
        
        Args:
            x: Images (B, C, H, W)
            
        Returns:
            Class logits (B, num_classes) or features if num_classes=0
        """
        x = self.forward_features(x)
        
        # Use class token for classification if available
        if self.use_cls_token:
            x = x[:, 0]  # Take cls token
        else:
            x = x.mean(dim=1)  # Global average pooling
        
        # Classification head
        x = self.head(x)
        
        return x
    
    def get_num_params(self) -> int:
        """Get number of parameters."""
        return sum(p.numel() for p in self.parameters())
