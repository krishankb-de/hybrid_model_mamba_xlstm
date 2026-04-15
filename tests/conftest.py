"""Test configuration."""

import pytest


def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers",
        "cuda: marks tests that require CUDA"
    )
    config.addinivalue_line(
        "markers",
        "willi_parity: tests that validate Python 3.9.23 / willi server compatibility"
    )
