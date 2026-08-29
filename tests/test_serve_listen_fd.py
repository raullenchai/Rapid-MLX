# SPDX-License-Identifier: Apache-2.0
"""Tests for ``rapid-mlx serve --listen-fd`` socket activation (issue #574).

Socket activation is the strongest closure of the bind→auth TOCTOU
window: an external supervisor (launchd, systemd, a parent process)
binds the listening socket, validates env + auth secret, THEN
``execve``'s into ``rapid-mlx serve --listen-fd <N>``. By the time we
run, there is no separate bind step that could race the auth wiring.

These tests pin the public CLI contract:

* ``--listen-fd <N>`` parses and is plumbed to ``uvicorn.run(fd=N, ...)``
  WITHOUT a fresh ``host``/``port`` bind.
* The default path (no ``--listen-fd``) is unchanged: ``host``/``port``
  still flow through.
* Out-of-band fd values (stdio fds, negative, absurdly high) are
  rejected by argparse with rc=2 so a typo doesn't silently take over
  the supervisor's stdout or land somewhere unrelated.
"""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import patch

import pytest

from vllm_mlx import cli

# ---------------------------------------------------------------------------
# Helpers — mirror the chat-command test style: drive ``cli.main()`` and
# capture the resolved ``argparse.Namespace`` via a ``serve_command`` stub.
# ---------------------------------------------------------------------------


def _capture_serve_args(argv: list[str]) -> list:
    """Drive ``cli.main()`` with the given argv and return the captured
    Namespace that ``serve_command`` would have received."""
    captured: list = []
    with (
        patch.object(sys, "argv", argv),
        patch.object(cli, "serve_command", side_effect=captured.append),
    ):
        cli.main()
    return captured


# ---------------------------------------------------------------------------
# Argparse — valid input
# ---------------------------------------------------------------------------


def test_serve_listen_fd_parses_valid_value():
    """``--listen-fd 3`` parses and lands on ``args.listen_fd``."""
    captured = _capture_serve_args(
        ["rapid-mlx", "serve", "qwen3.5-4b-4bit", "--listen-fd", "3"]
    )
    assert len(captured) == 1
    ns = captured[0]
    assert ns.listen_fd == 3


def test_serve_listen_fd_accepts_upper_bound():
    """``--listen-fd 1023`` (SysV soft-limit ceiling) is accepted."""
    captured = _capture_serve_args(
        ["rapid-mlx", "serve", "qwen3.5-4b-4bit", "--listen-fd", "1023"]
    )
    assert captured[0].listen_fd == 1023


def test_serve_listen_fd_default_is_none():
    """Without ``--listen-fd``, ``args.listen_fd`` is ``None`` — preserves
    the existing default behavior of binding ``host``/``port`` fresh."""
    captured = _capture_serve_args(["rapid-mlx", "serve", "qwen3.5-4b-4bit"])
    ns = captured[0]
    assert getattr(ns, "listen_fd", "missing") is None


# ---------------------------------------------------------------------------
# Argparse — invalid input must rc=2 with a friendly message
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_value",
    ["-1", "0", "1", "2", "9999", "65536", "abc", ""],
)
def test_serve_listen_fd_rejects_invalid(capsys, bad_value):
    """Stdio fds (0/1/2), negatives, absurdly high, and non-numeric values
    are rejected with rc=2 — argparse's standard "bad arg" exit code."""
    with (
        patch.object(
            sys,
            "argv",
            ["rapid-mlx", "serve", "qwen3.5-4b-4bit", "--listen-fd", bad_value],
        ),
        pytest.raises(SystemExit) as exc,
    ):
        cli.main()
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "--listen-fd" in err


def test_serve_listen_fd_error_message_is_specific(capsys):
    """The rejection message must explain the accepted range so the user
    immediately knows what to fix — not a bare ``invalid value``."""
    with (
        patch.object(
            sys,
            "argv",
            ["rapid-mlx", "serve", "qwen3.5-4b-4bit", "--listen-fd", "2"],
        ),
        pytest.raises(SystemExit) as exc,
    ):
        cli.main()
    assert exc.value.code == 2
    err = capsys.readouterr().err
    # Either form is acceptable as long as the bounds appear in the message.
    assert "between 3 and 1023" in err


def test_serve_listen_fd_nonnumeric_message(capsys):
    """Non-numeric ``--listen-fd`` rejection mentions integer expectation."""
    with (
        patch.object(
            sys,
            "argv",
            ["rapid-mlx", "serve", "qwen3.5-4b-4bit", "--listen-fd", "fd3"],
        ),
        pytest.raises(SystemExit) as exc,
    ):
        cli.main()
    assert exc.value.code == 2
    assert "must be an integer" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Plumbing — ``serve_command`` must pass ``fd=<N>`` to ``uvicorn.run``
# and must NOT pass host/port in that branch.
# ---------------------------------------------------------------------------


def _minimal_serve_ns(**overrides):
    """Build a Namespace populated with serve defaults via argparse so
    the test stays in sync with the serve subparser. Any field the
    caller wants to override is patched on the resolved Namespace."""
    captured: list = []
    argv = ["rapid-mlx", "serve", "qwen3.5-4b-4bit"]
    for k, v in overrides.items():
        if k == "listen_fd":
            argv += ["--listen-fd", str(v)]
        elif k == "port":
            argv += ["--port", str(v)]
        elif k == "host":
            argv += ["--host", v]
    with (
        patch.object(sys, "argv", argv),
        patch.object(cli, "serve_command", side_effect=captured.append),
    ):
        cli.main()
    return captured[0]


def test_run_uvicorn_passes_fd_when_listen_fd_set(monkeypatch):
    """When ``args.listen_fd`` is an int, ``_run_uvicorn`` MUST invoke
    ``uvicorn.run(app, fd=N, ...)`` and MUST NOT pass ``host``/``port``.

    This is the load-bearing assertion for Leg 2: a regression that
    silently re-introduces ``host=``/``port=`` in the fd branch reopens
    the bind→auth window. The companion bytecode test below pins that
    ``serve_command`` actually invokes this helper so the contract
    can't drift from the call site.

    Round-1 PR #696 codex review: the prior version reimplemented the
    branch inside the test, so it passed even if the production call
    site were deleted. This version exercises the real
    ``cli._run_uvicorn`` and patches ``uvicorn.run`` to capture kwargs.
    """
    captured_kwargs: dict = {}

    def fake_run(app, **kwargs):
        captured_kwargs["app"] = app
        captured_kwargs.update(kwargs)

    import uvicorn

    monkeypatch.setattr(uvicorn, "run", fake_run)

    ns = _minimal_serve_ns(listen_fd=7, port=9000, host="127.0.0.1")
    sentinel_app = object()
    cli._run_uvicorn(sentinel_app, ns, "info")

    assert captured_kwargs.get("app") is sentinel_app
    assert captured_kwargs.get("fd") == 7, (
        f"expected fd=7 in uvicorn.run kwargs, got {captured_kwargs!r}"
    )
    assert "host" not in captured_kwargs, (
        f"host must NOT be passed when fd is set, got {captured_kwargs!r}"
    )
    assert "port" not in captured_kwargs, (
        f"port must NOT be passed when fd is set, got {captured_kwargs!r}"
    )
    assert captured_kwargs.get("log_level") == "info"
    assert captured_kwargs.get("timeout_keep_alive") == 30


def test_run_uvicorn_passes_host_port_when_listen_fd_unset(monkeypatch):
    """The default path is unchanged: ``host``/``port`` flow through and
    ``fd`` is NOT passed. Same anti-regression shape as the listen-fd
    case — exercises the real ``cli._run_uvicorn`` rather than
    reimplementing the branch in the test body.
    """
    captured_kwargs: dict = {}

    def fake_run(app, **kwargs):
        captured_kwargs["app"] = app
        captured_kwargs.update(kwargs)

    import uvicorn

    monkeypatch.setattr(uvicorn, "run", fake_run)

    ns = _minimal_serve_ns(port=9000, host="127.0.0.1")
    assert getattr(ns, "listen_fd", None) is None
    sentinel_app = object()
    cli._run_uvicorn(sentinel_app, ns, "info")

    assert captured_kwargs.get("app") is sentinel_app
    assert captured_kwargs.get("host") == "127.0.0.1"
    assert captured_kwargs.get("port") == 9000
    assert "fd" not in captured_kwargs
    assert captured_kwargs.get("log_level") == "info"
    assert captured_kwargs.get("timeout_keep_alive") == 30


@pytest.fixture
def stub_heavy_serve_deps(monkeypatch):
    """Stub the heavyweight prologue of ``serve_command`` so a behavioral
    test can drive it through to the ``uvicorn.run`` call site without
    actually downloading a model, importing mlx, or booting an engine.

    Each stub is the minimum no-op that lets serve_command's control
    flow reach the ``uvicorn`` dispatch. A new heavy step added to
    serve_command will surface as an ImportError / AttributeError in
    the tests below; extend this fixture rather than working around it
    so the test stays faithful to the real execution path.
    """
    from vllm_mlx import _version_check
    from vllm_mlx import server as server_mod

    monkeypatch.setattr(_version_check, "prompt_upgrade_if_available", lambda: False)
    monkeypatch.setattr(
        _version_check, "print_staleness_warning_if_any", lambda **_kwargs: None
    )
    # Patch the exact module object this test file calls. Earlier suites may
    # deliberately reload ``vllm_mlx.cli`` and replace the package attribute;
    # re-importing it here would patch that new object while the file-level
    # ``cli`` reference below still invokes the old one.
    monkeypatch.setattr(cli, "_ensure_model_downloaded", lambda model: None)
    monkeypatch.setattr(cli, "_check_memory_capacity", lambda *a, **kw: None)
    monkeypatch.setattr(cli, "_check_disk_space", lambda *a, **kw: None)
    monkeypatch.setattr(server_mod, "configure_logging", lambda level: "info")
    monkeypatch.setattr(server_mod, "load_model", lambda *a, **kw: None)
    # ``serve_command`` calls ``server.configure_cors`` which does an
    # ``app.add_middleware``. That fails with "Cannot add middleware
    # after an application has started" if a prior test in the suite
    # has already booted a ``TestClient`` against ``vllm_mlx.server.app``
    # — order-dependent flake. Stub it to a no-op for these tests; the
    # CORS plumbing has its own dedicated tests.
    # ``configure_cors_from_env`` now always calls ``configure_cors`` with
    # keyword args (methods=, headers=, max_age=, allow_credentials=) once
    # the default-wildcard friendly UX landed — accept arbitrary kwargs so
    # the stub keeps matching the real signature.
    monkeypatch.setattr(server_mod, "configure_cors", lambda *a, **kw: None)
    # Some serve_command branches touch the rate-limiter wiring.
    from vllm_mlx.middleware import auth as auth_mod

    monkeypatch.setattr(auth_mod, "configure_rate_limiter", lambda *a, **kw: None)
    # ``install_request_logging_middleware`` also calls ``app.add_middleware``
    # — same "after an application has started" failure as CORS (#1167).
    from vllm_mlx.middleware import request_logging as reqlog_mod

    monkeypatch.setattr(
        reqlog_mod, "install_request_logging_middleware", lambda *a: None
    )
    return monkeypatch


def _free_tcp_port(host: str = "127.0.0.1") -> int:
    """Bind a real socket to an OS-assigned port, then release it. The
    port may race with another listener before the test rebinds, but
    for the few ms between the test's ``socket.bind`` preflight and
    ``uvicorn.run`` stub return that's acceptable.
    """
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def _capture_uvicorn_run(monkeypatch):
    """Patch ``uvicorn.run`` to record kwargs and return without
    actually starting a server. Returns the dict the test asserts on.
    """
    captured: dict = {}

    def fake_run(app, **kwargs):
        captured["app"] = app
        captured.update(kwargs)

    import uvicorn

    monkeypatch.setattr(uvicorn, "run", fake_run)
    return captured


@pytest.mark.requires_mlx
def test_serve_command_dispatches_uvicorn_with_fd_when_listen_fd_set(
    stub_heavy_serve_deps,
):
    """End-to-end behavioral test: with ``--listen-fd N`` set, driving
    the real ``cli.serve_command(ns)`` through its full prologue must
    land on ``uvicorn.run(app, fd=N, ...)`` — no ``host``/``port``.

    Round-5 codex PR #696 review pushed this from a bytecode-name
    pinning into a behavioral pin: a correct refactor that inlines or
    renames the ``_run_uvicorn`` helper while preserving the
    user-visible kwargs must KEEP passing. The only way that's true is
    to assert on the actual ``uvicorn.run`` call.
    """
    captured = _capture_uvicorn_run(stub_heavy_serve_deps)
    ns = _minimal_serve_ns(listen_fd=7)
    cli.serve_command(ns)

    assert captured.get("fd") == 7, (
        f"expected fd=7 in the real uvicorn.run call, got {captured!r}"
    )
    assert "host" not in captured, (
        f"host must NOT be passed in the listen-fd branch, got {captured!r}"
    )
    assert "port" not in captured, (
        f"port must NOT be passed in the listen-fd branch, got {captured!r}"
    )
    # And the Ready-banner source of truth must be wired up.
    from vllm_mlx.config import get_config

    cfg = get_config()
    assert cfg.bind_listen_fd == 7
    assert cfg.bind_host is None
    assert cfg.bind_port is None


def test_dflash_memory_check_receives_original_alias(
    stub_heavy_serve_deps, monkeypatch, scheduler_config_stub
):
    """The DFlash branch sizes the alias, not only its resolved HF path."""
    calls: list[tuple[str, str | None]] = []
    monkeypatch.setattr(
        cli,
        "_check_memory_capacity",
        lambda model, *, alias=None: calls.append((model, alias)),
    )

    dflash_server = ModuleType("vllm_mlx.speculative.dflash.server")
    dflash_server.run_dflash_server = lambda **_kwargs: None
    monkeypatch.setitem(
        sys.modules, "vllm_mlx.speculative.dflash.server", dflash_server
    )

    ns = _minimal_serve_ns()
    ns.enable_dflash = True
    ns._original_alias = "qwen3.5-27b-8bit"
    from vllm_mlx.speculative.dflash import eligibility

    monkeypatch.setattr(eligibility, "have_runtime", lambda: True)

    cli.serve_command(ns)

    assert calls == [(ns.model, ns._original_alias)]


def test_serve_command_threads_auto_detected_hybrid_into_cache_admission(
    stub_heavy_serve_deps, monkeypatch, scheduler_config_stub
):
    """The unified serve entrypoint must consume its one resolved profile."""
    from vllm_mlx import server as server_mod
    from vllm_mlx.model_profile import ModelProfile

    captured = {}
    detected_models = []

    def capture_load_model(*_args, scheduler_config, **_kwargs):
        captured["scheduler_config"] = scheduler_config

    monkeypatch.setattr(server_mod, "load_model", capture_load_model)

    def detect_model_config(model_name):
        detected_models.append(model_name)
        return ModelProfile(
            tool_call_parser="hermes",
            reasoning_parser="qwen3",
            is_hybrid=True,
            is_hybrid_explicit=True,
            supports_spec_decode=False,
        )

    monkeypatch.setattr(
        "vllm_mlx.model_auto_config.detect_model_config", detect_model_config
    )
    _capture_uvicorn_run(monkeypatch)
    ns = _minimal_serve_ns(port=_free_tcp_port())
    ns.model = "publisher/opaque-hybrid-checkpoint"
    ns._original_alias = None

    cli.serve_command(ns)

    scheduler = captured["scheduler_config"]
    assert detected_models == ["publisher/opaque-hybrid-checkpoint"]
    assert scheduler.enable_prefix_cache is True
    assert scheduler.hybrid_cache_entries == 8
    assert scheduler.non_trimmable_exact_prefix_reuse is True


def test_serve_command_cache_is_first_metadata_consumer(
    stub_heavy_serve_deps, monkeypatch, scheduler_config_stub
):
    """Fully pinned adjacent defaults leave cache admission as first reader."""
    from vllm_mlx import server as server_mod
    from vllm_mlx.model_profile import ModelProfile

    captured = {}
    monkeypatch.setattr(
        server_mod,
        "load_model",
        lambda *_args, scheduler_config, **_kwargs: captured.update(
            scheduler_config=scheduler_config
        ),
    )
    monkeypatch.setattr(
        "vllm_mlx.model_auto_config.detect_model_config",
        lambda _name: ModelProfile(is_hybrid=True, is_hybrid_explicit=True),
    )
    _capture_uvicorn_run(monkeypatch)
    ns = _minimal_serve_ns(port=_free_tcp_port())
    ns.model = "publisher/cache-only-metadata"
    ns._original_alias = None
    ns.pflash = "off"
    ns.pflash_keep_ratio = 0.2
    ns.tool_call_parser = "hermes"
    ns.reasoning_parser = "qwen3"
    ns.kv_cache_turboquant = "none"

    cli.serve_command(ns)

    assert captured["scheduler_config"].hybrid_cache_entries == 8


def test_serve_command_missing_auto_config_module_degrades_to_no_default(
    stub_heavy_serve_deps, monkeypatch, scheduler_config_stub
):
    """A partial install keeps the pre-existing optional-default fallback."""
    import builtins

    from vllm_mlx import server as server_mod

    captured = {}
    monkeypatch.setattr(
        server_mod,
        "load_model",
        lambda *_args, scheduler_config, **_kwargs: captured.update(
            scheduler_config=scheduler_config
        ),
    )
    real_import = builtins.__import__

    def block_auto_config(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 1 and name == "model_auto_config":
            raise ImportError("model auto-config unavailable")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", block_auto_config)
    _capture_uvicorn_run(monkeypatch)
    ns = _minimal_serve_ns(port=_free_tcp_port())
    ns.model = "publisher/opaque-checkpoint"
    ns._original_alias = None

    cli.serve_command(ns)

    assert captured["scheduler_config"].hybrid_cache_entries == 0


def test_serve_command_explicit_zero_wins_over_auto_detected_hybrid(
    stub_heavy_serve_deps, monkeypatch, scheduler_config_stub
):
    from vllm_mlx import server as server_mod
    from vllm_mlx.model_profile import ModelProfile

    captured = {}

    def capture_load_model(*_args, scheduler_config, **_kwargs):
        captured["scheduler_config"] = scheduler_config

    monkeypatch.setattr(server_mod, "load_model", capture_load_model)
    monkeypatch.setattr(
        "vllm_mlx.model_auto_config.detect_model_config",
        lambda _name: ModelProfile(
            is_hybrid=True,
            is_hybrid_explicit=True,
            supports_spec_decode=False,
        ),
    )
    _capture_uvicorn_run(monkeypatch)
    ns = _minimal_serve_ns(port=_free_tcp_port())
    ns.model = "publisher/opaque-hybrid-checkpoint"
    ns._original_alias = None
    ns.hybrid_cache_entries = 0

    with patch.object(
        sys,
        "argv",
        [
            "rapid-mlx",
            "serve",
            "publisher/opaque-hybrid-checkpoint",
            "--hybrid-cache-entries",
            "0",
        ],
    ):
        cli.serve_command(ns)

    assert captured["scheduler_config"].hybrid_cache_entries == 0
    assert captured["scheduler_config"].non_trimmable_exact_prefix_reuse is False


def test_serve_command_all_explicit_defaults_skip_model_detection(
    stub_heavy_serve_deps, monkeypatch, scheduler_config_stub
):
    """Explicit operator choices must not make metadata availability fatal."""

    def unexpected_detection(_model_name):
        raise AssertionError("fully explicit serve must not detect model metadata")

    monkeypatch.setattr(
        "vllm_mlx.model_auto_config.detect_model_config", unexpected_detection
    )
    _capture_uvicorn_run(monkeypatch)
    ns = _minimal_serve_ns(port=_free_tcp_port())
    ns.model = "publisher/explicitly-configured-checkpoint"
    ns._original_alias = None
    ns.pflash = "off"
    ns.pflash_keep_ratio = 0.2
    ns.tool_call_parser = "hermes"
    ns.reasoning_parser = "qwen3"
    ns.kv_cache_turboquant = "none"
    ns.hybrid_cache_entries = 0

    with patch.object(
        sys,
        "argv",
        [
            "rapid-mlx",
            "serve",
            "publisher/explicitly-configured-checkpoint",
            "--hybrid-cache-entries",
            "0",
        ],
    ):
        cli.serve_command(ns)


def test_strict_auto_default_retries_failed_nonfatal_parser_detection(
    stub_heavy_serve_deps, monkeypatch, scheduler_config_stub
):
    """A parser probe failure must not suppress a later strict consumer."""
    attempts = []

    def broken_detection(model_name):
        attempts.append(model_name)
        raise RuntimeError("broken checkpoint metadata")

    monkeypatch.setattr(
        "vllm_mlx.model_auto_config.detect_model_config", broken_detection
    )
    _capture_uvicorn_run(monkeypatch)
    ns = _minimal_serve_ns()
    ns.model = "publisher/broken-checkpoint"
    ns._original_alias = None
    ns.pflash = "off"
    ns.pflash_keep_ratio = 0.2  # Make parser detection the first metadata probe.
    ns.kv_cache_turboquant = None  # TurboQuant remains a strict consumer.

    with pytest.raises(RuntimeError, match="broken checkpoint metadata"):
        cli.serve_command(ns)

    assert attempts == [
        "publisher/broken-checkpoint",
        "publisher/broken-checkpoint",
    ]


@pytest.mark.requires_mlx
def test_serve_command_dispatches_uvicorn_with_host_port_when_listen_fd_unset(
    stub_heavy_serve_deps,
):
    """Default path: ``host``/``port`` flow into ``uvicorn.run`` and
    ``fd`` is not passed. Same end-to-end shape as the listen-fd case.
    """
    captured = _capture_uvicorn_run(stub_heavy_serve_deps)
    port = _free_tcp_port()
    ns = _minimal_serve_ns(host="127.0.0.1", port=port)
    cli.serve_command(ns)

    assert captured.get("host") == "127.0.0.1"
    assert captured.get("port") == port
    assert "fd" not in captured

    from vllm_mlx.config import get_config

    cfg = get_config()
    assert cfg.bind_host == "127.0.0.1"
    assert cfg.bind_port == port
    assert cfg.bind_listen_fd is None


@pytest.mark.requires_mlx
def test_serve_command_default_max_tokens_does_not_mutate_args(
    stub_heavy_serve_deps,
):
    """Omitted --max-tokens should stay omitted on args, while load_model
    receives the operational default.
    """
    from vllm_mlx import server as server_mod

    captured_load: dict = {}

    def fake_load_model(*args, **kwargs):
        captured_load["args"] = args
        captured_load["kwargs"] = kwargs

    stub_heavy_serve_deps.setattr(server_mod, "load_model", fake_load_model)
    captured_uvicorn = _capture_uvicorn_run(stub_heavy_serve_deps)
    ns = _minimal_serve_ns(host="127.0.0.1", port=_free_tcp_port())

    assert ns.max_tokens is None

    cli.serve_command(ns)

    assert captured_uvicorn
    assert ns.max_tokens is None
    assert captured_load["kwargs"]["max_tokens"] == 32768
    assert captured_load["kwargs"]["max_tokens_is_explicit"] is False


@pytest.mark.requires_mlx
def test_serve_command_skips_port_preflight_when_listen_fd_set(
    stub_heavy_serve_deps,
):
    """When ``--listen-fd N`` is set, ``serve_command`` MUST skip the
    ``host``/``port`` bind preflight. The supervisor has already bound
    the socket; running our own bind check against the same address
    always collides and would refuse to start.

    Regression: PR #696 introduced ``--listen-fd`` but the preflight at
    the top of ``serve_command`` kept running unconditionally, so any
    real socket-activation launcher hit "Port N is already in use" and
    ``sys.exit(1)`` before ever reaching ``uvicorn.run``.
    """
    import socket

    captured = _capture_uvicorn_run(stub_heavy_serve_deps)

    # Pre-bind the default serve port so the preflight WOULD fail if it
    # ran. Using SO_REUSEADDR matches the production preflight's own
    # socket setup — the preflight only fails when something is actively
    # listening, not when the port is in TIME_WAIT.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as blocker:
        blocker.bind(("127.0.0.1", 0))
        blocker.listen(1)
        blocked_port = blocker.getsockname()[1]

        ns = _minimal_serve_ns(listen_fd=11, host="127.0.0.1", port=blocked_port)
        # If the preflight runs, this raises SystemExit(1) before reaching
        # the uvicorn.run stub. The test passes only when the preflight
        # is correctly skipped in the listen-fd branch.
        cli.serve_command(ns)

    assert captured.get("fd") == 11, (
        f"expected uvicorn.run(fd=11, ...), got {captured!r}"
    )
    assert "host" not in captured
    assert "port" not in captured


@pytest.mark.requires_mlx
def test_serve_command_resets_stale_bind_fields_between_invocations(
    stub_heavy_serve_deps,
):
    """The singleton ``ServerConfig`` persists across in-process
    ``serve_command`` calls (test harnesses, embedded usage). A prior
    host/port stash MUST NOT leak into a subsequent listen-fd run (and
    vice-versa) — otherwise the lifespan banner reports a phantom
    listener.

    Round-4 codex PR #696 review prompted the explicit reset of all
    three fields at the top of the stash block. This test pins that
    behavior end-to-end.
    """
    captured = _capture_uvicorn_run(stub_heavy_serve_deps)

    # First call: host/port.
    port_a = _free_tcp_port()
    cli.serve_command(_minimal_serve_ns(host="127.0.0.1", port=port_a))
    from vllm_mlx.config import get_config

    cfg = get_config()
    assert (cfg.bind_host, cfg.bind_port, cfg.bind_listen_fd) == (
        "127.0.0.1",
        port_a,
        None,
    )

    # Second call: listen-fd. The prior host/port must be cleared.
    captured.clear()
    cli.serve_command(_minimal_serve_ns(listen_fd=11))
    assert (cfg.bind_host, cfg.bind_port, cfg.bind_listen_fd) == (None, None, 11)

    # Third call: back to host/port. The prior fd must be cleared.
    captured.clear()
    port_b = _free_tcp_port("0.0.0.0")
    cli.serve_command(_minimal_serve_ns(host="0.0.0.0", port=port_b))
    # ``host_display`` rewrites 0.0.0.0 → "localhost" for the banner.
    assert (cfg.bind_host, cfg.bind_port, cfg.bind_listen_fd) == (
        "localhost",
        port_b,
        None,
    )


# ---------------------------------------------------------------------------
# Help text — pin the dual-form documentation so future flag-list edits
# don't drop the bind→auth rationale silently.
# ---------------------------------------------------------------------------


def test_serve_listen_fd_help_documents_host_port_ignored(capsys):
    """``rapid-mlx serve --help`` must mention that ``--host``/``--port``
    are ignored when ``--listen-fd`` is set. Operators reading the help
    text need to know the precedence without diving into source."""
    with (
        patch.object(sys, "argv", ["rapid-mlx", "serve", "--help"]),
        pytest.raises(SystemExit) as exc,
    ):
        cli.main()
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "--listen-fd" in help_text
    assert "ignored" in help_text.lower()
