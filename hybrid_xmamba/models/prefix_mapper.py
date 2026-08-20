"""Image prefix mapper for Phase 10 image-conditioned generation.

Maps a ViT patch grid (e.g. BiomedCLIP's pre-pooling (B, 197, patch_dim)
output) to a fixed-length (B, k, decoder_dim) prefix that is prepended to
token embeddings before HybridLanguageModel.forward()/.generate(). Since
Mamba/mLSTM blocks treat their input as a generic (B, L, D) tensor with no
dependency on how it was produced, this needs no changes to any SSM/TFLA
kernel.

Deliberately attention-free (adaptive average pooling, not a Perceiver
Resampler): the same rationale that favors prefix-conditioning over
cross-attention for the decoder — avoid new attention/Triton machinery —
applies to this connector too.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImagePrefixMapper(nn.Module):
    """Maps a (B, N, patch_dim) patch grid to a (B, k, decoder_dim) prefix.

    Args:
        patch_dim: Input patch embedding dimension (e.g. 768 for ViT-B/16).
        decoder_dim: Output dimension, must match the decoder's config.dim.
        k: Number of prefix tokens to produce. Fixed at construction —
            sweeping k means training separate models, not one model
            handling variable k at inference.
        dropout: Dropout probability in the output projection.
    """

    def __init__(self, patch_dim: int, decoder_dim: int, k: int, dropout: float = 0.1):
        super().__init__()
        self.k = k
        self.token_proj = nn.Linear(patch_dim, decoder_dim, bias=False)
        self.out_proj = nn.Sequential(
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(decoder_dim, decoder_dim, bias=False),
        )

    def forward(self, patch_grid: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_grid: (B, N, patch_dim) — e.g. (B, 197, 768) CLS+patches.

        Returns:
            (B, k, decoder_dim) prefix embeddings.
        """
        x = self.token_proj(patch_grid)   # (B, N, decoder_dim)
        x = x.transpose(1, 2)             # (B, decoder_dim, N)
        x = F.adaptive_avg_pool1d(x, self.k)  # (B, decoder_dim, k)
        x = x.transpose(1, 2)             # (B, k, decoder_dim)
        return self.out_proj(x)
