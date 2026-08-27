# SPDX-License-Identifier: Apache-2.0
"""Pytest configuration and shared fixtures."""

import pytest

# Environment variables that point at the host's real, machine-specific HF
# cache. Every non-opted-in test gets these redirected to a fresh ``tmp_path``
# so a developer machine's real ``~/.cache/huggingface`` (possibly hundreds of
# GB, or entirely empty) can never leak into, or be mutated by, a test. This is
# the core of the hermetic-cache guarantee: a host with a 400 GB cache and a
# host with none must produce identical results for every test that did not
# explicitly opt in. See the ``_hermetic_hf_and_config_dirs`` fixture.
#
# ``HF_HUB_OFFLINE`` is deliberately NOT listed here: it is a network-access
# toggle, not a cache-location knob, and ``tests/test_cli_offline_serve.py``
# exercises it exhaustively with its own ``monkeypatch`` calls. Touching it
# here would fight those tests. A test that needs offline semantics sets it
# itself.
_HF_CACHE_ENV_VARS = ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE")

# RAPID_MLX_* env vars that name a config/data directory on the real host.
# Isolating them alongside the HF cache keeps state that lives in the user's
# home (``~/.rapid-mlx/``, community-bench roots, the DDTree draft mirror)
# out of the test run. All are already allowlisted as non-routing config knobs
# in ``tests/test_no_out_of_band_routing.py`` (``ALLOWED_RAPID_MLX_ENV_VARS``);
# this fixture only *points* them at a temp dir, it never selects a route.
#
# Readers, for reference (all read ``os.environ`` at call time, so a run-time
# override takes effect):
#   * ``RAPID_MLX_STATE_DIR``  — ``vllm_mlx/first_run.py::_state_dir``
#   * ``RAPID_MLX_HOME``       — ``vllm_mlx/community_bench/*``
#   * ``RAPID_MLX_DDTREE_PATCH_CACHE`` — ``vllm_mlx/speculative/ddtree/runtime.py``
#   * ``RAPID_MLX_CONFIG_HOME`` — allowlisted; no current reader, kept for parity
_RAPID_MLX_DIR_ENV_VARS = (
    "RAPID_MLX_STATE_DIR",
    "RAPID_MLX_HOME",
    "RAPID_MLX_DDTREE_PATCH_CACHE",
    "RAPID_MLX_CONFIG_HOME",
)


@pytest.fixture(autouse=True)
def _hermetic_hf_and_config_dirs(tmp_path, monkeypatch, request):
    """Isolate every test from the host's real HF cache and config dirs.

    Sets ``HF_HOME`` / ``HF_HUB_CACHE`` / ``TRANSFORMERS_CACHE`` and the
    machine-specific ``RAPID_MLX_*`` data dirs to a fresh per-test ``tmp_path``
    so no test accidentally reads (or writes) the host's real cache or state
    unless it deliberately opts in. This makes the no-MLX unit suite
    deterministic across machines: a box with a 400 GB ``~/.cache/huggingface``
    and a fresh Linux CI runner with none now run every non-opted-in test
    against the same empty cache.

    Opt-out: mark a test ``@pytest.mark.real_hf_cache`` (registered below) when
    it genuinely needs the host's real cache. A handful of integration tests
    legitimately probe the real cache (e.g. scan the repo index for a real
    ``models--*`` layout, or load real weights for a tokenizer/grammar file) —
    those must be marked so they keep working on a host with a real cache
    (Studio). Every OTHER test — including one that used to read the real cache
    (``test_doctor_env_health.py::test_huge_hf_cache_marks_warn``, made hermetic
    by mocking ``_hf_cache_dir`` / ``_dir_size_gb``) — is hermetic by default.
    Opting a test in must be reviewed: it re-introduces host-state dependence
    for that one test.

    Compatibility:
      * Tests that already manipulate these vars via ``monkeypatch`` in their
        own body simply override this fixture (test-body calls win; ``tmp_path``
        and the autouse fixture's values are torn down together).
      * The existing ``scheduler_config_stub`` fixture (NOT autouse) and the
        ``_reset_global_parser_state_after_each_test`` autouse fixture are
        unaffected — they never read HF / RAPID_MLX dirs.
      * Legacy ``HUGGINGFACE_HUB_CACHE`` (pre-``HF_HUB_CACHE`` spelling) is left
        alone; no codebase reader uses it and ``test_release_check_random.py``
        toggles it itself.
    """
    if request.node.get_closest_marker("real_hf_cache"):
        yield
        return

    # NB: do NOT ``mkdir`` the target. Several tests assert their ``tmp_path``
    # is left empty (e.g. ``test_community_bench_upload.py`` asserts
    # ``not list(tmp_path.glob("*"))``), and every reader resolves the cache
    # path lazily and tolerates a non-existent dir. Pointing the env at a
    # not-yet-created path is exactly right for hermeticity.
    #
    # Real huggingface hub layout: ``HF_HOME`` is the base, and the download
    # cache lives at ``<HF_HOME>/hub``. Pointing both at the SAME dir is wrong
    # (the hub looks for ``<HF_HUB_CACHE>/models--*`` nested under it), so we
    # mirror the real ``HF_HOME=<base>`` + ``HF_HUB_CACHE=<base>/hub`` split.
    hf_home = tmp_path / "hf-home"
    hf_hub_cache = hf_home / "hub"
    for var in _HF_CACHE_ENV_VARS:
        if var == "HF_HOME":
            monkeypatch.setenv(var, str(hf_home))
        elif var == "HF_HUB_CACHE":
            monkeypatch.setenv(var, str(hf_hub_cache))
        else:  # TRANSFORMERS_CACHE
            monkeypatch.setenv(var, str(hf_hub_cache))

    for var in _RAPID_MLX_DIR_ENV_VARS:
        monkeypatch.setenv(var, str(tmp_path / var.lower()))

    # Hermeticity for hub *readers*, not just env readers. huggingface_hub
    # snapshots several of these paths into module constants at import time
    # (``huggingface_hub.constants.HF_HUB_CACHE`` / ``HF_HOME`` and
    # ``huggingface_hub.file_download.HF_HUB_CACHE``) and many callers
    # (``scan_cache_dir``, ``snapshot_download``, ``try_to_load_from_cache``,
    # and every default ``cache_dir=...`` path) read those constants — NOT
    # ``os.environ`` — so a bare ``setenv`` still leaks the host cache to them.
    # Patch the constants in place so the whole huggingface_hub surface sees the
    # same temp layout. Guard by ``sys.modules.get`` because ``mock_hf_env`` /
    # network markers import hub lazily and ``huggingface_hub`` may not be
    # loaded yet for a given test.
    import sys

    for modname in (
        "huggingface_hub",
        "huggingface_hub.constants",
        "huggingface_hub.file_download",
    ):
        mod = sys.modules.get(modname)
        if mod is None:
            continue
        for attr, val in (
            ("HF_HOME", str(hf_home)),
            ("HF_HUB_CACHE", str(hf_hub_cache)),
            ("HUGGINGFACE_HUB_CACHE", str(hf_hub_cache)),
        ):
            if hasattr(mod, attr):
                monkeypatch.setattr(mod, attr, val)

    yield


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


@pytest.fixture
def scheduler_config_stub(monkeypatch):
    """Exercise scheduler wiring without importing the Apple-only runtime."""
    import importlib.machinery
    import importlib.util
    import sys
    import types

    turboquant_was_loaded = "vllm_mlx.turboquant" in sys.modules
    if importlib.util.find_spec("mlx") is None:
        # Import the server/tool-parser surface before installing the narrow
        # array shim, otherwise optional-dependency discovery could mistake
        # the shim for a complete MLX runtime.
        import numpy as np

        import vllm_mlx.server  # noqa: F401

        mlx = types.ModuleType("mlx")
        mlx.__path__ = []
        mlx.__spec__ = importlib.machinery.ModuleSpec("mlx", loader=None)
        mlx_core = types.ModuleType("mlx.core")
        mlx_core.__spec__ = importlib.machinery.ModuleSpec("mlx.core", loader=None)
        mlx_core.array = np.array
        mlx_core.float16 = np.float16
        mlx.core = mlx_core
        monkeypatch.setitem(sys.modules, "mlx", mlx)
        monkeypatch.setitem(sys.modules, "mlx.core", mlx_core)

    scheduler = types.ModuleType("vllm_mlx.scheduler")

    class SchedulerConfig:
        def __init__(self, **kwargs):
            self.enable_prefix_cache = True
            self.hybrid_cache_entries = 0
            self.non_trimmable_exact_prefix_reuse = False
            self.enable_mtp = False
            self.spec_decode = "none"
            self.__dict__.update(kwargs)
            if self.enable_mtp:
                self.spec_decode = "mtp"

    scheduler.SchedulerConfig = SchedulerConfig
    monkeypatch.setitem(sys.modules, "vllm_mlx.scheduler", scheduler)
    yield SchedulerConfig
    if not turboquant_was_loaded:
        sys.modules.pop("vllm_mlx.turboquant", None)


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
    config.addinivalue_line(
        "markers",
        "real_hf_cache: opt a test into using the host's real HF cache, "
        "bypassing the hermetic autouse ``_hermetic_hf_and_config_dirs`` "
        "fixture. Use only when a test genuinely needs the real cache; "
        "review every use. See that fixture's docstring.",
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
