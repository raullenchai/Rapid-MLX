# SPDX-License-Identifier: Apache-2.0
"""Wiring guard for the legacy ``python -m vllm_mlx.server`` entrypoint.

``tests/test_pflash_cli_wiring.py`` proves the ``serve``/``bench`` *commands*
route PFlash resolution through :func:`vllm_mlx.pflash.resolve_pflash_config`.
This file proves the THIRD serving entrypoint — ``vllm_mlx.server.main()`` —
does the same, so the two paths cannot drift.

Regression for #1458: ``server.main()`` used to resolve PFlash with
``resolve_pflash_mode_default`` + ``config_from_args`` directly. That pair
applies the tier-based *mode* default but NOT a per-alias ``pflash_keep_ratio``
override, so a verified alias pinned at a non-default ratio (bonsai-27b-2bit
@0.50, whose mid-prompt needle recall collapses to 1/5 at the 0.20 engine
default) auto-enabled PFlash at the lossy 0.20 here while ``rapid-mlx serve``
used 0.50. This asserts the EFFECTIVE config that reaches the scheduler, so a
revert to the old two-call form (which would capture keep_ratio 0.20) fails.

Fully offline: the resolver runs for real (``detect_model_config`` reads the
bonsai alias profile from the local index); every I/O boundary the entrypoint
hits before the PFlash step is mocked, and the ``validate_model_support`` hook —
called immediately after ``resolve_pflash_config`` with the built config as its
first positional arg — captures that config and short-circuits before any
engine/uvicorn boot.
"""

from __future__ import annotations

import importlib
import os
import sys
from contextlib import nullcontext
from unittest import mock

import pytest


class _StopError(Exception):
    """Short-circuit ``server.main()`` right after the PFlash step."""


def _server_module():
    """Resolve ``vllm_mlx.server`` from ``sys.modules`` at call time.

    NOT a module-level ``import ... as server``: another test in the shared
    process may ``sys.modules.pop("vllm_mlx.cli")`` / reimport (several route
    tests do), which swaps the live module object. A binding captured at import
    time would then be stale, so ``mock.patch.object`` on it would patch a
    different object than the one ``server.main()`` actually calls into — the
    real ``_port_preflight_or_die`` would run and, if port 8000 is taken by an
    earlier test, abort with SystemExit. Fetching fresh (and patching by string
    target below, which mock also resolves through ``sys.modules``) keeps the
    patch and the code-under-test pointed at the same object regardless of
    ordering.
    """
    return importlib.import_module("vllm_mlx.server")


# ``server.main()`` writes serving config into module-level globals (API key,
# timeouts, sampling defaults, and — via alias auto-config — the tool-call /
# reasoning parser names). Driving it here would leak that state into the shared
# test process (e.g. a later ``/models`` route test seeing an auto-detected
# ``tool_call_parser``), so snapshot and restore them around every test.
_SERVER_GLOBALS = (
    "_api_key",
    "_default_timeout",
    "_rate_limiter",
    "_default_temperature",
    "_default_top_p",
    "_default_top_k",
    "_enable_audio_lane",
    "_enable_auto_tool_choice",
    "_tool_call_parser",
    "_enable_tool_logits_bias",
    "_reasoning_parser",
    "_reasoning_parser_name",
)
_MISSING = object()


@pytest.fixture(autouse=True)
def _restore_server_globals():
    server = _server_module()
    saved = {name: getattr(server, name, _MISSING) for name in _SERVER_GLOBALS}
    mcp_saved = os.environ.get("RAPID_MLX_MCP_CONFIG", _MISSING)
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is _MISSING:
                if hasattr(server, name):
                    delattr(server, name)
            else:
                setattr(server, name, value)
        if mcp_saved is _MISSING:
            os.environ.pop("RAPID_MLX_MCP_CONFIG", None)
        else:
            os.environ["RAPID_MLX_MCP_CONFIG"] = mcp_saved


def _run_server_main_capturing_config(argv: list[str], *, lane=(False, False)):
    """Drive the REAL ``server.main()`` and return the ``PFlashConfig`` it
    resolved (captured at the ``validate_model_support`` call that immediately
    follows ``resolve_pflash_config``). ``lane`` is the ``(is_mllm, auto_text)``
    tuple the serving-lane resolver returns."""
    captured: dict = {}

    def _capture_validate(config, *, model_name, is_mllm=False):
        captured["config"] = config
        captured["model_name"] = model_name
        captured["is_mllm"] = is_mllm
        raise _StopError

    server = _server_module()
    # All patch targets are STRING paths so mock resolves each module through
    # ``sys.modules`` at enter time — the same object ``server.main()`` reaches
    # (``_port_preflight_or_die`` via its in-function ``from .cli import``).
    # Using ``mock.patch.object`` on an import-time-captured module would miss
    # after a ``sys.modules`` swap (see ``_server_module``).
    #
    # Only ``_StopError`` is allowed to escape — NOT SystemExit. If a preflight
    # ever runs unpatched (e.g. port 8000 taken) it aborts via SystemExit, and
    # we want that to fail LOUDLY here rather than be swallowed into an empty
    # ``captured`` that then trips a confusing "cfg is None" assertion.
    with (
        mock.patch("vllm_mlx.cli._port_preflight_or_die", lambda *a, **k: None),
        mock.patch("vllm_mlx.server._preflight_vision_runtime", lambda *a, **k: None),
        mock.patch("vllm_mlx.server._ensure_routing_config", lambda *a, **k: None),
        mock.patch("vllm_mlx.server.resolve_serving_lane", lambda name, **kw: lane),
        mock.patch("vllm_mlx.pflash.validate_model_support", _capture_validate),
        mock.patch.object(sys, "argv", ["vllm_mlx.server", *argv]),
        pytest.raises(_StopError),
    ):
        server.main()
    return captured


def _run_server_main_capturing_scheduler(
    argv: list[str],
    *,
    detected_config,
    expected_exception=_StopError,
    block_auto_config_import=False,
):
    """Drive ``server.main()`` through scheduler construction, offline."""
    captured: dict = {}
    detect = (
        mock.Mock(side_effect=detected_config)
        if isinstance(detected_config, BaseException)
        else mock.Mock(return_value=detected_config)
    )

    def _capture_load_model(_model, *, scheduler_config, **_kwargs):
        captured["scheduler_config"] = scheduler_config
        raise _StopError

    server = _server_module()
    real_import = __import__

    def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 1 and name == "model_auto_config":
            raise ImportError("model auto-config unavailable")
        return real_import(name, globals, locals, fromlist, level)

    import_guard = (
        mock.patch("builtins.__import__", blocked_import)
        if block_auto_config_import
        else nullcontext()
    )
    with (
        import_guard,
        mock.patch("vllm_mlx.cli._port_preflight_or_die", lambda *a, **k: None),
        mock.patch("vllm_mlx.server._preflight_vision_runtime", lambda *a, **k: None),
        mock.patch("vllm_mlx.server._ensure_routing_config", lambda *a, **k: None),
        mock.patch(
            "vllm_mlx.server.resolve_serving_lane", lambda _name, **_kw: (False, False)
        ),
        mock.patch(
            "vllm_mlx.model_auto_config.detect_model_config",
            detect,
        ),
        mock.patch("vllm_mlx.pflash.validate_model_support", lambda *a, **k: None),
        mock.patch("vllm_mlx.server.load_model", _capture_load_model),
        mock.patch.object(sys, "argv", ["vllm_mlx.server", *argv]),
        pytest.raises(expected_exception),
    ):
        server.main()
    return captured.get("scheduler_config"), detect.call_count


@pytest.mark.requires_mlx
def test_server_module_applies_per_alias_keep_ratio_override():
    # bonsai-27b-2bit pins pflash_tier=verified + pflash_keep_ratio=0.50.
    # Text lane (is_mllm False) → verified tier auto-enables PFlash "always".
    captured = _run_server_main_capturing_config(["--model", "bonsai-27b-2bit"])
    cfg = captured.get("config")
    assert cfg is not None, "validate_model_support was never reached"
    assert cfg.mode == "always"
    # The whole point of #1458: the alias override reaches this entrypoint too.
    # Old two-call wiring captured 0.20 here; the shared resolver yields 0.50.
    assert cfg.keep_ratio == pytest.approx(0.5)
    assert captured["is_mllm"] is False


@pytest.mark.requires_mlx
def test_server_module_explicit_keep_ratio_flag_still_wins():
    captured = _run_server_main_capturing_config(
        ["--model", "bonsai-27b-2bit", "--pflash-keep-ratio", "0.33"]
    )
    cfg = captured.get("config")
    assert cfg is not None
    assert cfg.keep_ratio == pytest.approx(0.33)


@pytest.mark.requires_mlx
def test_server_module_forwards_multimodal_lane_verdict():
    # MLLM lane → server must forward is_multimodal=True so the verified-tier
    # auto-ON is suppressed (PFlash cannot serve the MLLM lane). If server
    # dropped the lane verdict, the verified alias would still resolve "always".
    captured = _run_server_main_capturing_config(
        ["--model", "bonsai-27b-2bit"], lane=(True, False)
    )
    cfg = captured.get("config")
    assert cfg is not None
    assert captured["is_mllm"] is True
    assert cfg.mode == "off"


def test_server_module_applies_detected_hybrid_cache_default(scheduler_config_stub):
    from vllm_mlx.model_profile import ModelProfile

    scheduler, detect_calls = _run_server_main_capturing_scheduler(
        ["--model", "/models/linear-checkpoint"],
        detected_config=ModelProfile(
            hf_path="/models/linear-checkpoint",
            is_hybrid=True,
            is_hybrid_explicit=True,
        ),
    )

    assert scheduler.enable_prefix_cache is True
    assert scheduler.hybrid_cache_entries == 8
    assert scheduler.non_trimmable_exact_prefix_reuse is True
    assert detect_calls == 1


def test_server_module_keeps_dense_cache_default_zero(scheduler_config_stub):
    from vllm_mlx.model_profile import ModelProfile

    scheduler, detect_calls = _run_server_main_capturing_scheduler(
        ["--model", "/models/dense-checkpoint"],
        detected_config=ModelProfile(
            hf_path="/models/dense-checkpoint",
            is_hybrid=False,
        ),
    )

    assert scheduler.enable_prefix_cache is True
    assert scheduler.hybrid_cache_entries == 0
    assert scheduler.non_trimmable_exact_prefix_reuse is False
    assert detect_calls == 1


@pytest.mark.parametrize(
    "argv",
    [
        ["--tool-call-parser", "hermes", "--reasoning-parser", "qwen3"],
        [
            "--tool-call-parser",
            "hermes",
            "--reasoning-parser",
            "qwen3",
            "--pflash",
            "off",
            "--pflash-keep-ratio",
            "0.2",
        ],
        [
            "--tool-call-parser",
            "hermes",
            "--reasoning-parser",
            "qwen3",
            "--pflash",
            "off",
            "--pflash-keep-ratio",
            "0.2",
            "--kv-cache-turboquant",
            "none",
        ],
    ],
)
def test_server_module_lazy_metadata_consumers_share_one_result(
    scheduler_config_stub, argv
):
    from vllm_mlx.model_profile import ModelProfile

    scheduler, detect_calls = _run_server_main_capturing_scheduler(
        ["--model", "/models/linear-checkpoint", *argv],
        detected_config=ModelProfile(is_hybrid=True, is_hybrid_explicit=True),
    )

    assert scheduler.hybrid_cache_entries == 8
    assert detect_calls == 1


def test_server_module_strict_metadata_failure_propagates(scheduler_config_stub):
    scheduler, detect_calls = _run_server_main_capturing_scheduler(
        ["--model", "/models/broken-checkpoint"],
        detected_config=RuntimeError("broken metadata"),
        expected_exception=RuntimeError,
    )

    assert scheduler is None
    assert detect_calls == 1


def test_server_module_cache_metadata_failure_is_nonfatal(scheduler_config_stub):
    scheduler, detect_calls = _run_server_main_capturing_scheduler(
        [
            "--model",
            "/models/broken-checkpoint",
            "--tool-call-parser",
            "hermes",
            "--reasoning-parser",
            "qwen3",
            "--pflash",
            "off",
            "--pflash-keep-ratio",
            "0.2",
            "--kv-cache-turboquant",
            "none",
        ],
        detected_config=RuntimeError("broken metadata"),
    )

    assert scheduler.hybrid_cache_entries == 0
    assert detect_calls == 1


def test_server_module_missing_auto_config_module_degrades_to_no_default(
    scheduler_config_stub,
):
    scheduler, detect_calls = _run_server_main_capturing_scheduler(
        ["--model", "/models/opaque-checkpoint"],
        detected_config=None,
        block_auto_config_import=True,
    )

    assert scheduler.hybrid_cache_entries == 0
    assert detect_calls == 0
