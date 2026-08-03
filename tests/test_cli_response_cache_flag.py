# SPDX-License-Identifier: Apache-2.0
"""``--response-cache-entries`` end-to-end CLI wiring for ``serve``.

The response cache is a chat/serve feature: its lookup/store logic lives
only in the chat route, and ``bench`` never consumes it — so the flag is
registered on ``serve_parser`` ONLY (the bench flag would have been an
advertised no-op, so it is not registered there).

This test drives the real ``main()`` parser + ``serve_command`` far enough
to intercept the ACTUAL ``SchedulerConfig`` the serve path constructs, and
asserts the parsed ``--response-cache-entries`` value arrives at
``SchedulerConfig(response_cache_entries=N)`` (cli.py serve wiring). An
argparse-only check could not prove the plumbing line was wired at the
construction site — deleting the ``response_cache_entries=...`` kwarg would
leave an argparse-only test green. Fully offline: model load / disk /
network / version-check boundaries are mocked, and construction raises
``_StopError`` before any engine boot.
"""

from __future__ import annotations

import sys
from unittest import mock

import pytest

import vllm_mlx.cli as cli

# Pre-import so any lazy ``from .engine_core import ...`` binds the REAL
# class before a test patches ``vllm_mlx.scheduler.SchedulerConfig``.
import vllm_mlx.engine_core as _engine_core  # noqa: E402,F401


class _StopError(Exception):
    """Raised right after SchedulerConfig is built to short-circuit the
    command before any engine boot."""


def _capture_serve_scheduler_config(argv: list[str]) -> dict:
    """Drive the REAL ``main()`` → ``serve_command`` for a ``serve``
    invocation and capture the kwargs the serve path passes to
    ``SchedulerConfig(...)``.

    Patching ``vllm_mlx.scheduler.SchedulerConfig`` is the correct
    interception point: ``serve_command`` imports it locally via
    ``from .scheduler import SchedulerConfig`` at each call, so the patch
    is picked up at the real construction site (cli.py serve wiring). The
    fake records the kwargs and raises ``_StopError`` so nothing past
    construction (model load, engine boot, uvicorn) runs. Only the
    minimal I/O boundaries the serve path hits BEFORE construction are
    mocked out — everything between parse and ``SchedulerConfig(...)`` is
    the real code.
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
        "died before the response-cache plumbing under test. If this fires "
        "after unrelated serve-path changes, extend the boundary mocks above."
    )
    return captured


# ── serve wiring: the parsed value must reach SchedulerConfig ──────────


def test_serve_response_cache_entries_reaches_scheduler_config():
    """MUTATION-KILL: deleting ``response_cache_entries=...`` at the serve
    ``SchedulerConfig(...)`` construction (cli.py) makes this FAIL —
    ``captured`` would then lack the key and the assertion trips."""
    captured = _capture_serve_scheduler_config(
        ["serve", "qwen3.5-4b-4bit", "--response-cache-entries", "16"]
    )
    assert captured.get("response_cache_entries") == 16


def test_serve_response_cache_entries_defaults_to_zero():
    captured = _capture_serve_scheduler_config(["serve", "qwen3.5-4b-4bit"])
    assert captured.get("response_cache_entries") == 0


# ── serve parser registration (argparse surface) ──────────────────────


def _parse_serve_args(argv: list[str]):
    """Drive the REAL ``main()`` argument parser for a ``serve`` invocation
    and capture the parsed ``args`` namespace by intercepting the dispatch
    to ``serve_command`` — so we exercise the actual serve_parser
    registration, not a reconstructed stand-in. Nothing past parsing runs
    (no model download, no engine boot)."""
    captured = {}

    def _capture(args):
        captured["args"] = args

    with (
        mock.patch.object(cli, "serve_command", _capture),
        mock.patch.object(sys, "argv", ["rapid-mlx", *argv]),
    ):
        cli.main()
    assert "args" in captured, "serve_command dispatch was never reached"
    return captured["args"]


def test_serve_parser_registers_response_cache_entries():
    args = _parse_serve_args(
        ["serve", "qwen3.5-4b-4bit", "--response-cache-entries", "16"]
    )
    assert getattr(args, "response_cache_entries", None) == 16


def test_serve_parser_response_cache_entries_defaults_to_zero():
    args = _parse_serve_args(["serve", "qwen3.5-4b-4bit"])
    assert getattr(args, "response_cache_entries", None) == 0


def test_serve_negative_response_cache_entries_rejected_at_parse_time():
    """A negative ``--response-cache-entries`` is rejected up front by the
    ``non_negative_int`` argparse ``type``, before any model download or
    load — argparse exits 2 with a clear message. This runs the real
    ``main()`` parser, so it fails if the ``type=`` guard is removed."""
    with (
        mock.patch.object(
            sys,
            "argv",
            ["rapid-mlx", "serve", "qwen3.5-4b-4bit", "--response-cache-entries", "-1"],
        ),
        # If the guard were missing, parsing would succeed and dispatch to
        # serve_command; patch it so a regression there does NOT boot a
        # model (it would instead fail this test on the missing SystemExit).
        mock.patch.object(cli, "serve_command", lambda args: None),
        pytest.raises(SystemExit) as excinfo,
    ):
        cli.main()
    assert excinfo.value.code == 2


def test_non_negative_int_helper_rejects_negative_and_non_int():
    """The ``non_negative_int`` argparse ``type`` callable accepts ``>= 0``
    and raises ``ArgumentTypeError`` on a negative or non-integer value."""
    import argparse

    from vllm_mlx.cli import non_negative_int

    assert non_negative_int("0") == 0
    assert non_negative_int("16") == 16
    with pytest.raises(argparse.ArgumentTypeError):
        non_negative_int("-1")
    with pytest.raises(argparse.ArgumentTypeError):
        non_negative_int("abc")


def test_scheduler_config_still_rejects_negative_as_defense_in_depth():
    """The construction-time validation is kept even though argparse now
    rejects negatives earlier — a programmatic caller that bypasses the CLI
    still gets a clear error."""
    from vllm_mlx.scheduler import SchedulerConfig

    with pytest.raises(ValueError, match=r"response_cache_entries must be >= 0"):
        SchedulerConfig(response_cache_entries=-1)


# ── bench must NOT advertise the flag ─────────────────────────────────


def test_bench_does_not_register_response_cache_entries(capsys):
    """The response cache is serve-only; ``bench --response-cache-entries``
    must be REJECTED by argparse (the flag was an advertised no-op and was
    removed).

    Hardened so it cannot pass on the WRONG reason: a bare
    ``pytest.raises(SystemExit(2))`` would also be satisfied by any downstream
    exit-2 (e.g. bench_command later failing on the bogus model). This test
    proves three things:

      1. ``bench_command`` is NEVER reached — patched with a sentinel that
         raises if called, so parsing must have failed before dispatch.
      2. argparse exits with code 2 (its parse-error code).
      3. stderr specifically names the unrecognized flag, so the exit is
         attributable to ``--response-cache-entries`` and not some other
         parse problem.

    Mutation-kill: register ``--response-cache-entries`` on the bench parser
    → argparse accepts it, dispatch reaches the sentinel, and the sentinel's
    ``AssertionError`` (or the missing stderr/exit-code) turns this red.
    """

    def _must_not_run(_args):
        raise AssertionError(
            "bench_command was reached — argparse accepted "
            "--response-cache-entries instead of rejecting it"
        )

    with (
        mock.patch.object(cli, "bench_command", _must_not_run),
        mock.patch.object(
            sys,
            "argv",
            [
                "rapid-mlx",
                "bench",
                "does-not-exist/definitely-not-a-real-model",
                "--response-cache-entries",
                "8",
            ],
        ),
        pytest.raises(SystemExit) as excinfo,
    ):
        cli.main()

    # argparse exits 2 on an unrecognized argument.
    assert excinfo.value.code == 2
    stderr = capsys.readouterr().err
    assert "unrecognized arguments" in stderr, (
        f"expected an argparse unrecognized-argument error, got: {stderr!r}"
    )
    assert "--response-cache-entries" in stderr, (
        "the parse error must name --response-cache-entries so the exit is "
        f"attributable to that flag, got: {stderr!r}"
    )
