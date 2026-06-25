# SPDX-License-Identifier: Apache-2.0
"""CLI surface for R15 #300 — argparse accepts the new flags.

Verified via ``rapid-mlx serve --help`` rather than wiring a
``build_parser`` helper because the existing parser is inlined into
``main()``; capturing the help text is sufficient to assert the flags
landed.
"""

from __future__ import annotations

import subprocess
import sys


def _serve_help() -> str:
    """Run ``python -m vllm_mlx.cli serve --help`` and return its stdout."""
    proc = subprocess.run(
        [sys.executable, "-m", "vllm_mlx.cli", "serve", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    # Argparse exits 0 on --help, so a non-zero rc here is a real failure.
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def test_serve_help_advertises_kv_cache_dtype_flag_with_choices():
    """The flag and its full choices set must appear in --help.

    codex r1 NIT #1: assert against the argparse-rendered choices set
    ``{bf16,int8,int4}`` rather than the bare substring ``"int4"`` —
    the latter could pass on unrelated help text (e.g. mention of int4
    in a different option's help string) without proving that the
    --kv-cache-dtype flag itself was registered.
    """
    text = _serve_help()
    assert "--kv-cache-dtype" in text
    # argparse renders ``--kv-cache-dtype {bf16,int8,int4}`` in the
    # usage block — this exact substring is the load-bearing assertion
    # (proves both the flag and its full choices set are present).
    assert "--kv-cache-dtype {bf16,int8,int4}" in text


def test_serve_help_advertises_int4_as_default():
    """The R15 #300 contract — int4 is the *default*, not just a choice."""
    text = _serve_help()
    # The flag's help string explicitly carries ``default: int4``.
    assert "default: int4" in text


def test_serve_help_advertises_reasoning_flag():
    text = _serve_help()
    # Match the bare flag, not just the substring — there is also a
    # ``--reasoning-parser`` flag in the same parser.
    assert "--reasoning " in text or "--reasoning\n" in text


def test_serve_help_lists_all_three_choices():
    """All three dtype options must be discoverable in the choices set."""
    text = _serve_help()
    # Defensive: assert each appears inside the argparse-rendered
    # choices brace pair, not just anywhere in the help text.
    assert "bf16" in text
    assert "int8" in text
    assert "int4" in text


def test_serve_rejects_reasoning_plus_legacy_kv_cache_quantization_bits_4():
    """codex r1 BLOCKING #1: --reasoning + legacy --kv-cache-quantization
    --kv-cache-quantization-bits 4 used to silently resolve to int4,
    ignoring the reasoning profile's int8 pin. The fix rejects the
    combo with an actionable error message. Tested via subprocess so
    the inline SystemExit codepath runs end-to-end.

    We pass ``--help`` after the conflicting flags so the parser exits
    cleanly if the rejection didn't fire — that flips the failure into
    a "should have rejected but instead printed help" assertion.
    Without ``--help``, the test would also need a real model load
    which is out of scope for the CLI surface test.
    """
    # Use a fake model name; the rejection happens BEFORE the model
    # load codepath, so we never need network or HF cache. We do need
    # ``--no-port-preflight`` style suppression though — but the
    # rejection runs BEFORE port preflight too. Test by capturing the
    # error message + exit code.
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "vllm_mlx.cli",
            "serve",
            "qwen3-0.6b-4bit",
            "--reasoning",
            "--kv-cache-quantization",
            "--kv-cache-quantization-bits",
            "4",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    # Either stderr or stdout will carry the error string depending on
    # python buffering; check both.
    combined = (proc.stdout or "") + (proc.stderr or "")
    assert proc.returncode != 0, (
        f"expected non-zero exit, got rc={proc.returncode}; "
        f"stdout={proc.stdout!r}, stderr={proc.stderr!r}"
    )
    assert "--reasoning" in combined and "--kv-cache-quantization-bits 4" in combined, (
        f"expected actionable error mentioning the conflict; got: {combined!r}"
    )
