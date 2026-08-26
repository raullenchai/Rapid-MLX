# SPDX-License-Identifier: Apache-2.0
"""Command-level wiring: ``serve`` and ``bench`` must route PFlash resolution
through :func:`vllm_mlx.pflash.resolve_pflash_config`.

`tests/test_pflash.py` proves the helper resolves mode + keep_ratio correctly.
These tests prove the two COMMANDS actually call it (with the right model name
and multimodal verdict) and propagate its returned config — a mutation that
deletes the ``resolve_pflash_config(...)`` call from either command makes the
corresponding test fail (codex #1458 r2 BLOCKING: the helper-only test stayed
green if a command stopped invoking it).

Fully offline: every I/O boundary the commands hit before the PFlash step is
mocked, and the ``resolve_pflash_config`` stub short-circuits with a sentinel
error so nothing past it (engine boot, uvicorn, weight load) runs.
"""

from __future__ import annotations

import importlib.util
import sys
from unittest import mock

import pytest

import vllm_mlx.cli as cli

# Bind the real engine_core on MLX hosts before any test patches scheduler
# internals. Linux contract lanes deliberately have no MLX runtime.
if importlib.util.find_spec("mlx") is not None:
    import vllm_mlx.engine_core as _engine_core  # noqa: E402,F401


class _StopError(Exception):
    """Raised inside the resolve_pflash_config stub to short-circuit the command
    right after the PFlash wiring point, before any heavy boot."""


def _run_serve_capturing_pflash(argv: list[str], *, lane=(False, False)) -> dict:
    """Drive the REAL ``main()`` → ``serve_command`` and capture the arguments
    the serve path passes to ``resolve_pflash_config``. ``lane`` is the
    ``(is_mllm, auto_text_fallback)`` the serving-lane resolver returns — the
    first element is what serve must forward as ``is_multimodal``."""
    seen: dict = {}

    def _stub(args, *, model_name, is_multimodal=False, _detected_config=None):
        seen["model_name"] = model_name
        seen["is_multimodal"] = is_multimodal
        seen["detected_config"] = _detected_config
        seen["args_is"] = args
        raise _StopError

    with (
        mock.patch.object(cli, "_check_disk_space", lambda *a, **k: None),
        mock.patch.object(cli, "_check_memory_capacity", lambda *a, **k: None),
        mock.patch.object(cli, "_ensure_model_downloaded", lambda *a, **k: None),
        mock.patch(
            "vllm_mlx._version_check.prompt_upgrade_if_available",
            return_value=False,
        ),
        mock.patch("vllm_mlx.api.utils.resolve_serving_lane", lambda name, **kw: lane),
        mock.patch("vllm_mlx.pflash.resolve_pflash_config", _stub),
        mock.patch.object(sys, "argv", ["rapid-mlx", *argv]),
        mock.patch.object(sys.stdin, "isatty", return_value=False),
        pytest.raises((_StopError, SystemExit)),
    ):
        cli.main()
    return seen


def test_serve_command_routes_pflash_through_resolve_pflash_config(
    scheduler_config_stub,
):
    from vllm_mlx.model_aliases import resolve_model

    seen = _run_serve_capturing_pflash(["serve", "bonsai-27b-2bit"])
    # By the PFlash step the command has already resolved the alias to its
    # hf_path — that resolved name is what must reach resolve_pflash_config
    # (detect_model_config resolves the profile back via the hf_path index).
    assert seen.get("model_name") == resolve_model("bonsai-27b-2bit")
    # Text-lane model → serve forwards is_multimodal=False.
    assert seen.get("is_multimodal") is False
    assert seen.get("detected_config") is not None


def test_serve_command_forwards_multimodal_lane_verdict():
    # When the serving-lane resolver flags an MLLM lane, serve must forward
    # is_multimodal=True so resolve_pflash_config can suppress the verified
    # auto-ON (PFlash can't serve the MLLM lane).
    seen = _run_serve_capturing_pflash(["serve", "qwen3.5-4b-4bit"], lane=(True, False))
    assert seen.get("is_multimodal") is True


def _run_bench_capturing_pflash(argv: list[str]) -> dict:
    """Same, for the ``bench`` command. Bench has no MLLM lane, so it always
    forwards ``is_multimodal=False``."""
    seen: dict = {}

    def _stub(args, *, model_name, is_multimodal=False):
        seen["model_name"] = model_name
        seen["is_multimodal"] = is_multimodal
        raise _StopError

    with (
        mock.patch.object(cli, "_check_disk_space", lambda *a, **k: None),
        mock.patch.object(cli, "_check_memory_capacity", lambda *a, **k: None),
        mock.patch.object(cli, "_ensure_model_downloaded", lambda *a, **k: None),
        mock.patch(
            "vllm_mlx.api.utils.resolve_serving_lane",
            lambda name, **kw: (False, False),
        ),
        mock.patch("vllm_mlx.pflash.resolve_pflash_config", _stub),
        mock.patch.object(sys, "argv", ["rapid-mlx", *argv]),
        pytest.raises((_StopError, SystemExit)),
    ):
        cli.main()
    return seen


def test_bench_command_routes_pflash_through_resolve_pflash_config():
    from vllm_mlx.model_aliases import resolve_model

    seen = _run_bench_capturing_pflash(["bench", "bonsai-27b-2bit"])
    assert seen.get("model_name") == resolve_model("bonsai-27b-2bit")
    assert seen.get("is_multimodal") is False
