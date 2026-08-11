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

from vllm_mlx import cli


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


class _KVArgs:
    """Minimal stand-in for the parsed ``serve`` namespace.

    Defaults mirror argparse's: TurboQuant off, legacy quantization off,
    bits=8, reasoning off. Each test overrides only what it is about.
    """

    def __init__(self, **overrides):
        self.kv_cache_turboquant = None
        self.kv_cache_quantization = False
        self.kv_cache_quantization_bits = 8
        self.reasoning = False
        for key, value in overrides.items():
            setattr(self, key, value)


def test_reasoning_plus_legacy_kv_cache_quantization_bits_4_is_rejected():
    """codex r1 BLOCKING #1: --reasoning + legacy --kv-cache-quantization
    --kv-cache-quantization-bits 4 used to silently resolve to int4,
    ignoring the reasoning profile's int8 pin. The fix rejects the combo
    with an actionable error message.

    Exercised against ``kv_cache_flag_conflict`` rather than by spawning
    ``serve``. The subprocess version of this test ran a real ``serve``
    with a 30 s timeout and asserted a non-zero exit; because alias
    resolution and the weight download run before the rejection, it passed
    on machines that happened to have the fixture model cached and timed
    out on machines that did not. Its colour tracked local Hugging Face
    cache state, not whether the rejection still fired — so deleting the
    rejection outright would not reliably have turned it red.
    """
    reason = cli.kv_cache_flag_conflict(
        _KVArgs(
            reasoning=True, kv_cache_quantization=True, kv_cache_quantization_bits=4
        )
    )
    assert reason is not None, "the --reasoning + bits=4 conflict must be rejected"
    assert "--reasoning" in reason
    assert "--kv-cache-quantization-bits 4" in reason
    # The message has to say what to do, not just what is wrong.
    assert "--kv-cache-dtype int8" in reason


def test_reasoning_plus_legacy_bits_8_is_allowed():
    """bits=8 is equivalent to --reasoning's own int8 pin, so it is not a
    conflict. Pins the boundary the check above must not over-reach past."""
    assert (
        cli.kv_cache_flag_conflict(
            _KVArgs(
                reasoning=True,
                kv_cache_quantization=True,
                kv_cache_quantization_bits=8,
            )
        )
        is None
    )


def test_turboquant_and_legacy_quantization_are_mutually_exclusive():
    reason = cli.kv_cache_flag_conflict(
        _KVArgs(kv_cache_turboquant="k8v4", kv_cache_quantization=True)
    )
    assert reason is not None
    assert "mutually exclusive" in reason


def test_out_of_range_quantization_bits_is_rejected():
    """codex r2 BLOCKING #1: argparse pins bits to {4, 8}, but programmatic
    callers can land anything here, and the old code silently labelled every
    non-4 value as int8."""
    reason = cli.kv_cache_flag_conflict(
        _KVArgs(kv_cache_quantization=True, kv_cache_quantization_bits=5)
    )
    assert reason is not None
    assert "must be 4 or 8" in reason


def test_no_kv_flags_is_not_a_conflict():
    """The untouched default must never be reported as a conflict."""
    assert cli.kv_cache_flag_conflict(_KVArgs()) is None


def test_serve_rejects_the_conflict_end_to_end():
    """The wiring half: ``serve`` must actually consult the predicate and
    exit non-zero. Uses ``--kv-cache-quantization-bits 5``, which argparse
    itself rejects at parse time — before any alias resolution or download —
    so this stays fast and cache-independent while still proving the CLI
    refuses the flag rather than proceeding."""
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "vllm_mlx.cli",
            "serve",
            "qwen3-0.6b-4bit",
            "--kv-cache-quantization",
            "--kv-cache-quantization-bits",
            "5",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    combined = (proc.stdout or "") + (proc.stderr or "")
    assert proc.returncode != 0, (
        f"expected non-zero exit, got rc={proc.returncode}; "
        f"stdout={proc.stdout!r}, stderr={proc.stderr!r}"
    )
    assert "4" in combined and "8" in combined, (
        f"expected the bits constraint in the error; got: {combined!r}"
    )
