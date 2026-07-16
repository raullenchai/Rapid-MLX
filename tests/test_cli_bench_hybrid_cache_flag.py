# SPDX-License-Identifier: Apache-2.0
"""#1103 codex BLOCKING-2: ``--hybrid-cache-entries`` must be honored by the
benchmark path, not only ``serve``.

The bench MemoryCacheConfig assembly reads ``args.hybrid_cache_entries`` (via
``getattr(..., 0)``), but the flag was originally registered ONLY on
``serve_parser``. So ``rapid-mlx bench --hybrid-cache-entries N`` was rejected
by argparse (``unrecognized arguments``) and, had a caller reached the config
assembly, it would have silently fallen back to 0 — meaning bench could never
measure the hybrid-reuse effect the flag exists to expose.

Verified via ``--help`` capture + a real parse attempt (subprocess) because
the argparse parser is inlined into ``main()`` rather than exposed through a
``build_parser`` helper — same approach as ``tests/test_kv_cache_dtype_cli.py``.
"""

from __future__ import annotations

import subprocess
import sys


def _bench_help() -> str:
    proc = subprocess.run(
        [sys.executable, "-m", "vllm_mlx.cli", "bench", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def test_bench_help_advertises_hybrid_cache_entries_flag():
    """The flag must appear in ``rapid-mlx bench --help`` — proof it is
    registered on the benchmark parser, matching serve."""
    text = _bench_help()
    assert "--hybrid-cache-entries" in text


def test_bench_accepts_hybrid_cache_entries_flag():
    """``rapid-mlx bench <model> --hybrid-cache-entries 4`` must PARSE — i.e.
    argparse must not reject the flag with 'unrecognized arguments'.

    We can't run a full benchmark in a unit test (it would download weights),
    so we assert on the negative: whatever the run does downstream, it must
    NOT die at the argparse layer complaining about the flag. Before the fix
    the bench parser rejected it outright.
    """
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "vllm_mlx.cli",
            "bench",
            "does-not-exist/definitely-not-a-real-model",
            "--hybrid-cache-entries",
            "4",
            "--num-prompts",
            "1",
            "--max-tokens",
            "1",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    combined = proc.stdout + proc.stderr
    # The flag itself must be accepted by argparse. Any downstream failure
    # (bad model, no weights) is fine and expected — but NOT an argparse
    # 'unrecognized arguments' rejection of our flag.
    assert "unrecognized arguments" not in combined, combined
    assert "--hybrid-cache-entries" not in _argparse_error_lines(combined), combined


def _argparse_error_lines(text: str) -> str:
    """Return only the lines argparse emits on a usage error, so a legitimate
    mention of the flag elsewhere (e.g. a downstream log) doesn't mask a real
    'unrecognized arguments' rejection."""
    return "\n".join(
        line for line in text.splitlines() if "error:" in line or "usage:" in line
    )
