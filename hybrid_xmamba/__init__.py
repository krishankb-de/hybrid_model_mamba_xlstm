"""Hybrid Mamba-xLSTM Package.

This package provides a hybrid architecture combining Mamba and xLSTM layers
with chunk-parallel PyTorch kernels for efficient sequence modeling.

There are no hand-written CUDA or Triton kernels in this package: the selective scan,
the SSD scan and the mLSTM TFLA path are all plain PyTorch (see
kernels/selective_scan/scan_interface.py). Triton kernels DO appear at runtime when the
model is run under `torch.compile`, because Inductor generates them -- that is an opt-in
inference path measured in EFFICIENCY_PLAN.md E1, not a kernel written here.
"""

__version__ = "0.1.0"

from hybrid_xmamba.models.hybrid_lm import HybridLanguageModel
from hybrid_xmamba.models.configuration_hybrid import HybridConfig

__all__ = [
    "HybridLanguageModel",
    "HybridConfig",
]
