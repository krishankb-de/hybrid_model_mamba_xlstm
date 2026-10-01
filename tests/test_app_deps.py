"""CHAT_UI_PLAN.md P2-A: the app's runtime deps are importable in this interpreter."""


def test_app_dependencies_import():
    import fastapi  # noqa: F401
    import uvicorn  # noqa: F401
    import httpx2  # noqa: F401
    from fastapi.testclient import TestClient  # noqa: F401
    # Make the multipart check real: import the module that FastAPI requires
    import python_multipart.multipart  # noqa: F401


def test_app_is_scanned_for_pep604_and_pep585():
    from tests.test_willi_parity import SCAN_ROOTS, REPO_ROOT, iter_py_files
    assert REPO_ROOT / "app" in SCAN_ROOTS
    # Verify iter_py_files yields at least one file under app/ (sanity check)
    app_files = [f for f in iter_py_files() if f.is_relative_to(REPO_ROOT / "app")]
    assert len(app_files) > 0, "No Python files found under app/ — iter_py_files may be skipping it"
