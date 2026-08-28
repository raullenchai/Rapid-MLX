# SPDX-License-Identifier: Apache-2.0
"""Pytest configuration and shared fixtures."""

import ipaddress
import os
import socket
import sys

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
# toggle, not a cache-location knob. It is handled separately by the
# network-off half of the same fixture (``_install_hf_offline``), and tests
# such as ``tests/test_cli_offline_serve.py`` that exercise both values still
# win because a test-body ``monkeypatch`` call runs after the autouse fixture.
_HF_CACHE_ENV_VARS = ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE")

# Marker that opts a test back INTO network access (#2518). Every other test
# runs network-off: the Hugging Face hub is pinned offline and any socket
# connect to a non-loopback address fails immediately. ``real_hf_cache`` alone
# does NOT grant network access — a test that needs both marks both.
_NETWORK_OPT_IN_MARKER = "requires_network"


class HermeticNetworkAccessError(RuntimeError):
    """A hermetic (non-``requires_network``) test tried to reach the network.

    Deliberately NOT an ``OSError`` subclass: ``urllib3``/``httpcore``/
    ``asyncio`` retry or translate socket ``OSError``s, which would turn the
    guard into a slow, mislabelled ``ConnectionError``. A ``RuntimeError``
    passes straight through to the test with the offending address in it.
    """


def _is_local_address(address) -> bool:
    """True for loopback / unspecified IPs, ``localhost`` and AF_UNIX paths.

    Loopback is allowed because the unit suite legitimately starts in-process
    servers and connects to them; everything else is off-box by definition.
    """
    if not isinstance(address, tuple) or not address:
        # AF_UNIX path (str/bytes) or an exotic family — local IPC.
        return True
    host = address[0]
    if isinstance(host, bytes):
        host = host.decode("ascii", "replace")
    if not isinstance(host, str):
        return True
    if host == "" or host.lower().rstrip(".") == "localhost":
        return True
    try:
        ip = ipaddress.ip_address(host.split("%", 1)[0])
    except ValueError:
        return False  # any other hostname resolves off-box
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip.is_loopback or ip.is_unspecified


def _install_network_guard(monkeypatch, nodeid: str) -> None:
    """Fail fast on any non-loopback ``socket.connect`` for the current test.

    Same shape as pytest-socket's ``--allow-hosts=127.0.0.1`` mode, kept
    in-tree so the no-MLX CI lane needs no extra plugin. Patching the class
    attribute covers plain sockets, ``ssl.SSLSocket`` (it calls
    ``super().connect``), ``socket.create_connection`` (urllib / urllib3 /
    httpcore) and asyncio's ``sock_connect`` — i.e. every HTTP client the
    product uses, including the model mirror's ``urllib`` round-trips.
    """
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def _refuse(address) -> None:
        where = (
            f"{address[0]}:{address[1]}"
            if isinstance(address, tuple) and len(address) >= 2
            else repr(address)
        )
        raise HermeticNetworkAccessError(
            f"{nodeid} attempted a network connection to {where}. Unit tests "
            "are network-off by default (tests/conftest.py, #2518): stub the "
            "download / HTTP call, or mark the test "
            f"@pytest.mark.{_NETWORK_OPT_IN_MARKER} if it genuinely needs the "
            "network."
        )

    def _guarded_connect(self, address):
        if not _is_local_address(address):
            _refuse(address)
        return real_connect(self, address)

    def _guarded_connect_ex(self, address):
        if not _is_local_address(address):
            _refuse(address)
        return real_connect_ex(self, address)

    _guarded_connect._hermetic_network_guard = True
    _guarded_connect_ex._hermetic_network_guard = True
    monkeypatch.setattr(socket.socket, "connect", _guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", _guarded_connect_ex)


def _hf_offline_from_env() -> bool:
    """``huggingface_hub.constants._is_true`` semantics for the offline switches."""
    raw = os.environ.get("HF_HUB_OFFLINE") or os.environ.get("TRANSFORMERS_OFFLINE")
    return raw is not None and raw.lower() in {"1", "on", "yes", "true"}


def _reset_hf_sessions() -> None:
    """Drop huggingface_hub's cached HTTP session(s), if the hub is loaded.

    huggingface_hub < 1.0 binds its ``OfflineAdapter`` when a (per-thread,
    cached) ``requests`` session is created, so flipping the offline constant
    only takes effect on a fresh session. 1.x checks the constant on every
    request instead and has no ``reset_sessions``; nothing to do there.
    """
    hf_http = sys.modules.get("huggingface_hub.utils._http")
    reset = getattr(hf_http, "reset_sessions", None)
    if reset is not None:
        reset()


def _install_hf_offline(monkeypatch, request) -> None:
    """Pin the Hugging Face hub offline for the current test.

    Two layers, because ``huggingface_hub.constants`` snapshots
    ``HF_HUB_OFFLINE`` from the environment ONCE at import time (see
    ``scripts/kv_quant_quality_gate.py::_enable_hf_offline`` for the same
    fix in the product tooling):

    * ``HF_HUB_OFFLINE=1`` in the environment — read by the product's own
      offline switch (``cli._offline_hub_mode_active``) and by a hub imported
      for the first time during the test.
    * ``huggingface_hub.constants.HF_HUB_OFFLINE = True`` when the hub is
      already imported — ``is_offline_mode()`` reads it at request time, so
      every hub HTTP call raises ``OfflineModeIsEnabled`` and
      ``snapshot_download`` / ``hf_hub_download`` of an uncached repo raise
      ``LocalEntryNotFoundError`` immediately instead of downloading.
    """
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    hf_constants = sys.modules.get("huggingface_hub.constants")
    if hf_constants is not None and hasattr(hf_constants, "HF_HUB_OFFLINE"):
        monkeypatch.setattr(hf_constants, "HF_HUB_OFFLINE", True)
    _reset_hf_sessions()
    # Run before ``monkeypatch`` undoes the constant, so no offline-bound
    # session (hub < 1.0) outlives this test.
    request.addfinalizer(_reset_hf_sessions)


def _sync_hf_offline_with_env(monkeypatch) -> None:
    """For ``requires_network`` tests: make the hub constant mirror the env.

    A previous hermetic test may have been the first to import the hub while
    ``HF_HUB_OFFLINE=1`` was set, freezing the constant to ``True`` for the
    whole session. Re-derive it from the (unpatched) environment so an opted-in
    test sees the host's real offline setting.
    """
    hf_constants = sys.modules.get("huggingface_hub.constants")
    if hf_constants is not None and hasattr(hf_constants, "HF_HUB_OFFLINE"):
        monkeypatch.setattr(hf_constants, "HF_HUB_OFFLINE", _hf_offline_from_env())
    _reset_hf_sessions()


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

    Network-off by default (#2518): an empty per-test cache turned every
    latent ``load_model`` / ``snapshot_download`` into a real download — on
    hosted CI that pulled ``Qwen3.5-9B-4bit`` once per leaking test and kept
    the no-MLX lane busy for ~20 min. So the same fixture also pins the
    Hugging Face hub offline (``HF_HUB_OFFLINE=1`` plus the already-imported
    ``huggingface_hub.constants.HF_HUB_OFFLINE``) and fails any non-loopback
    ``socket.connect`` immediately with ``HermeticNetworkAccessError`` naming
    the test and the address. See ``_install_hf_offline`` /
    ``_install_network_guard``.

    Opt-ins (registered below, both must be reviewed):
      * ``@pytest.mark.real_hf_cache`` — use the host's real cache. A handful
        of integration tests legitimately probe it (scan the repo index for a
        real ``models--*`` layout, load real weights for a tokenizer/grammar
        file). It does NOT grant network access: a cached checkpoint loads
        fine offline, and an uncached one fails fast instead of downloading.
      * ``@pytest.mark.requires_network`` — disarm the network guard. Nothing
        in the unit suite needs it today; a test that genuinely does must
        say so here rather than depend on the runner's connectivity.
    Every OTHER test — including one that used to read the real cache
    (``test_doctor_env_health.py::test_huge_hf_cache_marks_warn``, made hermetic
    by mocking ``_hf_cache_dir`` / ``_dir_size_gb``) — is hermetic by default.

    Compatibility:
      * Tests that already manipulate these vars via ``monkeypatch`` in their
        own body simply override this fixture (test-body calls win; ``tmp_path``
        and the autouse fixture's values are torn down together). That includes
        ``HF_HUB_OFFLINE``: ``tests/test_cli_offline_serve.py`` sets it to
        ``"1"`` / ``""`` per test to drive the product's offline switch either
        way, and a test that mocks the downloader and wants the product's
        ONLINE path simply ``delenv``s it — the socket guard still stands.
      * The existing ``scheduler_config_stub`` fixture (NOT autouse) and the
        ``_reset_global_parser_state_after_each_test`` autouse fixture are
        unaffected — they never read HF / RAPID_MLX dirs.
      * Legacy ``HUGGINGFACE_HUB_CACHE`` (pre-``HF_HUB_CACHE`` spelling) is left
        alone; no codebase reader uses it and ``test_release_check_random.py``
        toggles it itself.
      * Loopback / AF_UNIX sockets stay open: in-process test servers, uvicorn
        on ``127.0.0.1``, asyncio self-pipes and ``HF_ENDPOINT=http://127.0.0.1``
        stubs all keep working.
    """
    if request.node.get_closest_marker(_NETWORK_OPT_IN_MARKER) is not None:
        _sync_hf_offline_with_env(monkeypatch)
    else:
        _install_hf_offline(monkeypatch, request)
        _install_network_guard(monkeypatch, request.node.nodeid)

    # Application state is independent of the HF cache opt-in. A test that
    # reads real cached weights must still never read or mutate the developer's
    # first-run/config/bench state under ~/.rapid-mlx.
    for var in _RAPID_MLX_DIR_ENV_VARS:
        monkeypatch.setenv(var, str(tmp_path / var.lower()))

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


@pytest.fixture
def hub_online_env(monkeypatch):
    """Simulate the product's ONLINE hub mode for a test that mocks the fetch.

    The hermetic default exports ``HF_HUB_OFFLINE=1``, which the product's own
    switch (``cli._offline_hub_mode_active``) honours: ``_ensure_model_downloaded``
    and the ``chat`` / ``/model`` gates then refuse an uncached model up front
    instead of walking the download path. A test that asserts on that path
    (disk gate called, resolved sha pinned, confirm prompt shown, ...) with the
    downloader stubbed requests this fixture so it keeps testing what it says.

    The network guard is NOT disarmed: the hub constant stays offline and
    non-loopback sockets still fail fast, so every real fetch must be stubbed.
    Use ``@pytest.mark.requires_network`` for genuine network access instead.
    """
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)


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
        "bypassing the cache redirection of the hermetic autouse "
        "``_hermetic_hf_and_config_dirs`` fixture (the network guard still "
        "applies). Use only when a test genuinely needs the real cache; "
        "review every use. See that fixture's docstring.",
    )
    config.addinivalue_line(
        "markers",
        f"{_NETWORK_OPT_IN_MARKER}: opt a test into real network access, "
        "disarming the network-off guard of the hermetic autouse "
        "``_hermetic_hf_and_config_dirs`` fixture (HF_HUB_OFFLINE + "
        "non-loopback socket refusal, #2518). Use only for a genuine "
        "integration test; review every use.",
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
