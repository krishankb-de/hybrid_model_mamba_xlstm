"""CHAT_UI_PLAN.md P2-A: the app's three runtime deps are importable in this interpreter."""
import importlib.util


def test_app_dependencies_import():
    import fastapi  # noqa: F401
    import uvicorn  # noqa: F401
    from fastapi.testclient import TestClient  # noqa: F401
    assert importlib.util.find_spec("python_multipart") or importlib.util.find_spec("multipart")


def test_app_is_scanned_for_pep604_and_pep585():
    from tests.test_willi_parity import SCAN_ROOTS, REPO_ROOT
    assert REPO_ROOT / "app" in SCAN_ROOTS
