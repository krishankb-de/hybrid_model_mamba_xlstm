"""CPU tests for scripts/isbi_figure_cases.py's pure core (the GPU path needs MIMIC and a checkpoint)."""

import pytest
import torch

from scripts.isbi_figure_cases import topk_neighbours


def test_topk_returns_most_similar_first():
    gallery = torch.nn.functional.normalize(torch.randn(50, 8), dim=-1)
    query = gallery[7].clone()
    sims, idx = topk_neighbours(query, gallery, 4)
    assert idx[0].item() == 7
    assert torch.allclose(sims[0], torch.tensor(1.0), atol=1e-5)
    assert torch.all(sims[:-1] >= sims[1:])


def test_topk_rejects_k_larger_than_gallery():
    gallery = torch.nn.functional.normalize(torch.randn(3, 8), dim=-1)
    with pytest.raises(ValueError):
        topk_neighbours(gallery[0], gallery, 4)
