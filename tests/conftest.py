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


@pytest.fixture(autouse=True)
def _reset_global_parser_state_after_each_test():
    """Keep the process-global parser state hermetic across tests.

    Effective parser resolution reads TWO process-global sources (see
    ``vllm_mlx/routes/models.py`` ``effective_parsers_for``): the
    ``ServerConfig`` singleton (``cfg.tool_call_parser``) AND the
    ``vllm_mlx.server`` module-level ``_tool_call_parser`` fallback. Several
    suites mutate either one directly and never restore it:

    * ``test_orphan_tool_validation`` / ``test_r12_reasoning_sanitizer_required``
      do ``reset_config(); cfg.tool_call_parser = "hermes"``.
    * ``test_capabilities_field`` does ``server._tool_call_parser = "hermes"``
      by *direct assignment* (not ``monkeypatch``), so nothing restores it.

    Either leak bleeds into a later test that reads the resolved parser under a
    given collection order — most visibly
    ``test_routes.py::TestModelsRoutes::test_retrieve_unknown_id_keeps_baseline_shape``,
    whose unknown-id baseline expects the resolved ``tool_call_parser`` to be
    ``None``. Pinning ``cfg.tool_call_parser=None`` in that test is not enough:
    resolution then falls through to the ``server`` module global. That was a
    real order-dependent flake (green in isolation, red in the full suite).

    Reset BOTH sources to their module defaults after every test. Both resets
    are cheap, and every suite that needs specific parser state sets it up at
    the start of each test (or via ``monkeypatch``), so a teardown reset is
    compatible.
    """
    yield

    # Reset only the parser state a test actually loaded. Guarding on
    # ``sys.modules`` (a) skips work for a module no test imported — it cannot
    # have leaked — and (b) avoids importing ``vllm_mlx.server`` here, which
    # pulls ``uvicorn``: the lightweight "no-MLX" CI test job does not install
    # it, so an unconditional import ERRORs every test's teardown.
    import sys

    _config_mod = sys.modules.get("vllm_mlx.config.server_config")
    if _config_mod is not None:
        _config_mod.reset_config()

    _server = sys.modules.get("vllm_mlx.server")
    if _server is not None:
        _server._tool_call_parser = None
        _server._reasoning_parser = None
        _server._reasoning_parser_name = None


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--server-url",
        action="store",
        default="http://localhost:8000",
        help="URL of the Rapid-MLX server for integration tests",
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
    config.addinivalue_line(
        "markers",
        "property: hermetic Hypothesis property-based test (see tests/property/)",
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
