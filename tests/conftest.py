# SPDX-License-Identifier: Apache-2.0
"""Pytest configuration and shared fixtures."""

import pytest

_SCRIPT_ONLY_MODULES = {"regression_suite.py"}
"""Files inside ``tests/`` that define ``test_*`` symbols but are
actually standalone scripts invoked by the doctor harness via
subprocess against a live server (see
``vllm_mlx/doctor/checks/api.py``). pytest must not run them as
unit tests — every call would fail with ``URLError`` and the
diff-aware ``targeted_tests`` step in ``scripts/pr_validate``
would flag any newly-added test in such a file as a regression.

The marker lives in conftest (loaded only by pytest) so the
script modules themselves don't take a runtime ``import pytest``
dependency (pytest is dev-only; codex R3 closure)."""


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--server-url",
        action="store",
        default="http://localhost:8000",
        help="URL of the vllm-mlx server for integration tests",
    )
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests that require model loading",
    )


def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "slow: mark test as slow (requires model loading)"
    )
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration test (requires running server)",
    )


def pytest_collection_modifyitems(config, items):
    """Skip slow tests unless --run-slow is passed."""
    if not config.getoption("--run-slow"):
        skip_slow = pytest.mark.skip(reason="Need --run-slow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

    # Skip integration tests unless server URL is explicitly provided
    skip_integration = pytest.mark.skip(reason="Integration tests require --server-url")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)

    # Skip items inside script-only modules (regression_suite.py etc.)
    # — see ``_SCRIPT_ONLY_MODULES`` above. ``pytest_ignore_collect`` is
    # not called when the file is named explicitly on the command line
    # (which is exactly what ``scripts/pr_validate`` does for diff-
    # adjacent files), so the skip has to happen post-collection.
    skip_script_only = pytest.mark.skip(
        reason="Standalone script — runs as subprocess via doctor harness, "
        "not pytest. See tests/conftest.py::_SCRIPT_ONLY_MODULES."
    )
    for item in items:
        if item.path.name in _SCRIPT_ONLY_MODULES:
            item.add_marker(skip_script_only)


@pytest.fixture(scope="session")
def server_url(request):
    """Get server URL from command line."""
    return request.config.getoption("--server-url")
