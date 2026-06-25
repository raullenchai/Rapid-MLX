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


def test_serve_help_advertises_kv_cache_dtype_flag():
    text = _serve_help()
    assert "--kv-cache-dtype" in text
    # The R15 #300 default must be visible to the operator at --help time.
    assert "int4" in text


def test_serve_help_advertises_reasoning_flag():
    text = _serve_help()
    assert "--reasoning" in text


def test_serve_help_lists_choices():
    """All three dtype options must be discoverable at --help time."""
    text = _serve_help()
    # argparse renders choices as ``{bf16,int8,int4}`` in the help line.
    assert "bf16" in text
    assert "int8" in text
    assert "int4" in text
