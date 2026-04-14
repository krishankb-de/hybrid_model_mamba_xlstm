"""Tests for mask-aware pooling in HybridTextEncoder.encode."""

import pytest
import torch

from hybrid_xmamba.models.configuration_hybrid import HybridConfig
from hybrid_xmamba.models.hybrid_lm import HybridTextEncoder


def _small_encoder():
    # Small config — kernels that require CUDA are sidestepped by the tiny dim
    config = HybridConfig(
        vocab_size=1000,
        dim=64,
        num_layers=2,
        layer_pattern=["mamba", "mlstm"],
        max_position_embeddings=128,
        num_heads=2,
        head_dim=32,
        slstm_hidden_dim=64,
        slstm_num_heads=2,
        dropout=0.0,
        use_fast_path=False,
        use_tfla=False,
    )
    return HybridTextEncoder(config, embed_dim=64)


def test_encode_padded_matches_unpadded():
    """encode(padded, mask) should equal encode(unpadded) up to fp tolerance."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _small_encoder().to(device).eval()

    torch.manual_seed(0)
    real_len = 7
    pad_len = 32
    ids_short = torch.randint(1, 999, (1, real_len), device=device)

    pad_token = 0
    ids_padded = torch.full((1, pad_len), pad_token, dtype=torch.long, device=device)
    ids_padded[0, :real_len] = ids_short[0]

    mask_padded = torch.zeros((1, pad_len), dtype=torch.long, device=device)
    mask_padded[0, :real_len] = 1

    with torch.no_grad():
        emb_short = model.encode(ids_short)
        emb_padded = model.encode(ids_padded, attention_mask=mask_padded)

    assert torch.allclose(emb_short, emb_padded, atol=1e-5), (
        f"padded encode differs from unpadded: "
        f"max abs diff {(emb_short - emb_padded).abs().max().item():.3e}"
    )


def test_encode_without_mask_uses_last_position():
    """encode(ids) with no mask falls back to last position (legacy behaviour)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _small_encoder().to(device).eval()

    torch.manual_seed(0)
    ids = torch.randint(1, 999, (2, 10), device=device)
    with torch.no_grad():
        emb = model.encode(ids)
    assert emb.shape == (2, 64)
    norms = emb.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
