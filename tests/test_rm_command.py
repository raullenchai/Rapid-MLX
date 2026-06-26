# SPDX-License-Identifier: Apache-2.0
"""Tests for ``rapid-mlx rm <model>``.

Pre-0.9.7 ``rm`` deleted gigabytes of model weights silently — no
confirmation, no freed-space summary. This file pins the new contract:

* default: prompt ``Remove <model> (X.Y GiB)? [y/N]``, default ``N``;
* empty input (just Enter) → ``Aborted.`` and exit 0;
* EOF (non-TTY / ctrl-D) → ``Aborted.`` and exit 0;
* ``-y / --yes`` → no prompt, runs the delete;
* on success a ``Freed X.Y GiB`` line is printed.

The actual HF cache strategy is mocked — these tests must never delete
real files. Size suffix matches ``vllm_mlx.cli._format_bytes`` (GiB).
"""

from __future__ import annotations

import io
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from vllm_mlx import cli

# ----- helpers --------------------------------------------------------------


def _make_repo(
    repo_id: str = "mlx-community/Qwen3.5-9B-MLX-4bit", size_on_disk: int = 6 * 1024**3
) -> MagicMock:
    """Build a fake ``CachedRepoInfo`` matching the surface ``rm_command`` uses."""
    rev = SimpleNamespace(commit_hash="deadbeef")
    repo = MagicMock()
    repo.repo_id = repo_id
    repo.repo_type = "model"
    repo.size_on_disk = size_on_disk
    repo.revisions = [rev]
    return repo


def _patched_scan(repo: MagicMock | None) -> MagicMock:
    """Build a fake ``scan_cache_dir()`` return-value with a no-op strategy.

    ``strategy.execute()`` is what actually deletes files on disk, so the
    mock no-ops it — guaranteeing the test suite can never wipe a real
    HF cache even if the test logic regresses.
    """
    cache = MagicMock()
    cache.repos = [repo] if repo is not None else []
    strategy = MagicMock()
    strategy.expected_freed_size_str = "6.0 GiB"
    cache.delete_revisions.return_value = strategy
    return cache, strategy


def _invoke_rm(
    model: str, yes: bool, stdin_text: str | None
) -> tuple[str, int, MagicMock]:
    """Run ``cli.rm_command`` against a mocked HF cache.

    Returns ``(stdout, exit_code, strategy_mock)``. ``exit_code == 0`` for
    normal returns; on ``sys.exit(N)`` it propagates ``N``. ``stdin_text=None``
    simulates EOF (e.g. piped stdin closed, ctrl-D).
    """
    repo = _make_repo()
    cache, strategy = _patched_scan(repo)

    if stdin_text is None:

        def fake_input(_prompt: str) -> str:
            raise EOFError
    else:

        def fake_input(_prompt: str) -> str:
            return stdin_text

    args = SimpleNamespace(model=model, yes=yes)
    buf = io.StringIO()

    with (
        patch("huggingface_hub.scan_cache_dir", return_value=cache),
        patch("builtins.input", side_effect=fake_input),
        patch.object(sys, "stdout", buf),
    ):
        try:
            cli.rm_command(args)
            code = 0
        except SystemExit as e:
            code = int(e.code) if e.code is not None else 0

    return buf.getvalue(), code, strategy


# ----- prompt path ---------------------------------------------------------


def test_default_prompts_and_n_aborts() -> None:
    """``n`` at the prompt aborts cleanly: exit 0, no delete, ``Aborted.`` printed."""
    out, code, strategy = _invoke_rm(
        "mlx-community/Qwen3.5-9B-MLX-4bit", yes=False, stdin_text="n"
    )
    assert code == 0, f"abort path must exit 0, got {code}"
    assert "Aborted." in out, f"expected 'Aborted.' marker, got:\n{out}"
    strategy.execute.assert_not_called()
    # ``Freed`` summary is the success marker — must NOT appear on abort.
    assert "Freed" not in out


def test_empty_input_aborts() -> None:
    """Pressing Enter without typing accepts the [y/N] default — N — and aborts."""
    out, code, strategy = _invoke_rm(
        "mlx-community/Qwen3.5-9B-MLX-4bit", yes=False, stdin_text=""
    )
    assert code == 0
    assert "Aborted." in out
    strategy.execute.assert_not_called()


def test_eof_aborts() -> None:
    """EOF on stdin (piped / ctrl-D) is treated as cancel, not silent-accept."""
    out, code, strategy = _invoke_rm(
        "mlx-community/Qwen3.5-9B-MLX-4bit", yes=False, stdin_text=None
    )
    assert code == 0
    assert "Aborted." in out
    strategy.execute.assert_not_called()


def test_y_at_prompt_proceeds_and_prints_freed() -> None:
    """Typing ``y`` at the prompt runs the delete and prints ``Freed X.Y GiB``."""
    out, code, strategy = _invoke_rm(
        "mlx-community/Qwen3.5-9B-MLX-4bit", yes=False, stdin_text="y"
    )
    assert code == 0
    strategy.execute.assert_called_once()
    assert "Freed" in out, f"expected 'Freed' summary on success, got:\n{out}"
    assert "GiB" in out, f"expected GiB suffix from _format_bytes, got:\n{out}"


# ----- --yes path ----------------------------------------------------------


def test_yes_flag_skips_prompt() -> None:
    """``--yes`` must never call ``input()`` — proves it via a side-effect that
    would raise on call."""
    repo = _make_repo()
    cache, strategy = _patched_scan(repo)
    args = SimpleNamespace(model="mlx-community/Qwen3.5-9B-MLX-4bit", yes=True)
    buf = io.StringIO()

    def boom(_prompt: str) -> str:
        raise AssertionError("input() must not be called when --yes is set")

    with (
        patch("huggingface_hub.scan_cache_dir", return_value=cache),
        patch("builtins.input", side_effect=boom),
        patch.object(sys, "stdout", buf),
    ):
        cli.rm_command(args)

    strategy.execute.assert_called_once()
    out = buf.getvalue()
    assert "Freed" in out
    assert "GiB" in out


# ----- not-cached path -----------------------------------------------------


def test_missing_model_exits_1_without_prompt() -> None:
    """When the model isn't in the cache there's nothing to confirm — the
    command must short-circuit before ``input()`` and exit 1."""
    cache, _ = _patched_scan(None)  # empty repos
    args = SimpleNamespace(model="mlx-community/Nope", yes=False)
    buf = io.StringIO()

    def boom(_prompt: str) -> str:
        raise AssertionError("input() must not be called when repo is missing")

    with (
        patch("huggingface_hub.scan_cache_dir", return_value=cache),
        patch("builtins.input", side_effect=boom),
        patch.object(sys, "stdout", buf),
        pytest.raises(SystemExit) as exc,
    ):
        cli.rm_command(args)

    assert exc.value.code == 1
    out = buf.getvalue()
    assert "not in the HuggingFace cache" in out


# ----- size formatting ----------------------------------------------------


def test_size_uses_format_bytes_suffix() -> None:
    """The prompt + freed line use the same suffix convention as the rest of
    the CLI (``_format_bytes`` → GiB/MiB/KiB), so users don't see ``5.0 G``
    in one place and ``5.0 GiB`` in another for the same model."""
    repo = _make_repo(size_on_disk=int(6.5 * 1024**3))
    cache, strategy = _patched_scan(repo)
    args = SimpleNamespace(model="mlx-community/Qwen3.5-9B-MLX-4bit", yes=True)
    buf = io.StringIO()

    with (
        patch("huggingface_hub.scan_cache_dir", return_value=cache),
        patch.object(sys, "stdout", buf),
    ):
        cli.rm_command(args)

    out = buf.getvalue()
    # ``_format_bytes`` renders 6.5 GiB as ``6.5 GiB`` (1-decimal IEC).
    assert "6.5 GiB" in out, f"expected '6.5 GiB' in Freed line, got:\n{out}"
