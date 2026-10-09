"""Test configuration."""

import pytest


@pytest.fixture
def tiny_gallery(tmp_path):
    """CHAT_UI_PLAN.md P5-B: a synthetic gallery of section 6.5, written by `scripts/build_retrieval_gallery.py --tiny` into
    tmp_path/gallery: 200 train images and 240 report rows (the 200 train reports, then 40 test reports) as 16-d vectors, with
    duplicate report groups inside the train rows, inside the test rows and across them, labels, and 320x320 gray JPEGs under
    images/. Returns the directory. P5-C, P5-D and P5-E reuse it; read the sizes from manifest.json["counts"], not from here."""
    from scripts.build_retrieval_gallery import build_tiny
    out = tmp_path / "gallery"
    build_tiny(out)
    return out


def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    # Here and not in pytest.ini: its [tool:pytest] header is a setup.cfg one, so pytest reads nothing from that file.
    config.addinivalue_line(
        "markers",
        "e2e: the chat page in a real Chrome (tests/e2e; needs requirements-e2e.txt); run with -m e2e"
    )
    config.addinivalue_line(
        "markers",
        "cuda: marks tests that require CUDA"
    )
    config.addinivalue_line(
        "markers",
        "willi_parity: tests that validate Python 3.9.23 / willi server compatibility"
    )
