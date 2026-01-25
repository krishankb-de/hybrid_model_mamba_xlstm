"""Hybrid Mamba-xLSTM Package.

This package provides a hybrid architecture combining Mamba and xLSTM layers
with custom CUDA/Triton kernels for efficient sequence modeling.
"""

__version__ = "0.1.0"

from hybrid_xmamba.models.hybrid_lm import HybridLanguageModel
from hybrid_xmamba.models.configuration_hybrid import HybridConfig

__all__ = [
    "HybridLanguageModel",
    "HybridConfig",
]
