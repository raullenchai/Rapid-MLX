# SPDX-License-Identifier: Apache-2.0
"""``serve --metal-cap-kv-bytes-per-token`` plumbing.

``Scheduler._infer_kv_dtype_bytes`` documents that quantized-KV
deployments are NOT auto-detected and names
``SchedulerConfig.metal_cap_kv_bytes_per_token`` as the operator escape
hatch. That field existed only on the Python API, so a CLI user running
``--kv-cache-turboquant`` / ``--kv-cache-quantization`` got a D-METAL-CAP
admission projection computed at fp16 regardless of the codec — long
prompts the codec would have fit were rejected with a 503.

These tests pin the CLI surface and the wiring: the parsed value must
reach ``SchedulerConfig``, and the default must stay 0 so the
architecture-aware auto-derivation (and the OOM cliff it prevents) is
untouched for everyone who does not set the flag.
"""

import sys
from unittest import mock

import pytest

import vllm_mlx.cli as cli

# Import the engine module EAGERLY, before ``SchedulerConfig`` is patched.
# Modules that annotate ``SchedulerConfig | None`` evaluate that union at
# import time; if the first import happens while the patch is active, the
# union is built against the fake and raises
# ``TypeError: unsupported operand type(s) for |``. Importing here binds
# the real class first — the patch is then a pure name rebind and no class
# body re-runs under it. Same guard as
# ``tests/test_cli_bench_hybrid_cache_flag.py``.
import vllm_mlx.engine_core as _engine_core  # noqa: E402,F401


class _StopError(Exception):
    """Raised once SchedulerConfig is built, to short-circuit engine boot."""


def _capture_serve_scheduler_config(argv: list[str]) -> dict:
    """Drive the REAL ``main()`` → ``serve_command`` and capture the kwargs
    the serve path passes to ``SchedulerConfig(...)``.

    Mirrors ``tests/test_cli_response_cache_flag.py`` — only the I/O
    boundaries the serve path hits BEFORE construction are mocked, so
    everything between argparse and ``SchedulerConfig(...)`` is real code.
    """
    captured: dict = {}

    def _fake_scheduler_config(*args, **kwargs):
        captured.update(kwargs)
        raise _StopError

    with (
        mock.patch.object(cli, "_check_disk_space", lambda *a, **k: None),
        mock.patch.object(cli, "_check_memory_capacity", lambda *a, **k: None),
        mock.patch.object(cli, "_ensure_model_downloaded", lambda *a, **k: None),
        mock.patch.object(
            cli, "_gather_kv_cache_dtype_inputs", lambda *a, **k: ({}, None)
        ),
        mock.patch(
            "vllm_mlx._version_check.prompt_upgrade_if_available",
            return_value=False,
        ),
        mock.patch(
            "vllm_mlx.utils.tokenizer.load_model_with_fallback",
            return_value=(object(), object()),
        ),
        mock.patch("vllm_mlx.scheduler.SchedulerConfig", _fake_scheduler_config),
        mock.patch.object(sys, "argv", ["rapid-mlx", *argv]),
        mock.patch.object(sys.stdin, "isatty", return_value=False),
        pytest.raises((_StopError, SystemExit)),
    ):
        cli.main()

    assert captured, (
        "SchedulerConfig was never constructed by the serve path — the flow "
        "died before the metal-cap plumbing under test. If this fires after "
        "unrelated serve-path changes, extend the boundary mocks above."
    )
    return captured


def test_serve_metal_cap_kv_bytes_reaches_scheduler_config():
    """MUTATION-KILL: deleting ``metal_cap_kv_bytes_per_token=...`` at the
    serve ``SchedulerConfig(...)`` construction makes this FAIL."""
    captured = _capture_serve_scheduler_config(
        ["serve", "qwen3.5-4b-4bit", "--metal-cap-kv-bytes-per-token", "98304"]
    )
    assert captured.get("metal_cap_kv_bytes_per_token") == 98304


def test_serve_metal_cap_kv_bytes_defaults_to_zero():
    """0 is load-bearing: ``_resolve_kv_bytes_per_token`` only auto-derives
    the architecture-aware figure when the override is falsy. A non-zero
    default would silently disable the sliding-window / KV-sharing
    refinement for every model."""
    captured = _capture_serve_scheduler_config(["serve", "qwen3.5-4b-4bit"])
    assert captured.get("metal_cap_kv_bytes_per_token") == 0


def test_serve_metal_cap_kv_bytes_rejects_negative():
    """Guarded by ``non_negative_int`` — a negative per-token figure would
    make the projection shrink with prompt length and defeat the gate.

    argparse rejects at parse time, i.e. BEFORE SchedulerConfig is
    constructed, so this asserts on the parser directly rather than going
    through ``_capture_serve_scheduler_config`` (whose post-condition is
    that construction was reached).
    """
    with (
        mock.patch.object(
            sys,
            "argv",
            [
                "rapid-mlx",
                "serve",
                "qwen3.5-4b-4bit",
                "--metal-cap-kv-bytes-per-token",
                "-1",
            ],
        ),
        mock.patch.object(sys.stdin, "isatty", return_value=False),
        pytest.raises(SystemExit) as exc,
    ):
        cli.main()
    assert exc.value.code != 0, "a negative per-token figure must not be accepted"
