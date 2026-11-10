# SPDX-License-Identifier: Apache-2.0
"""Tests for ``rapid-mlx rm <model>``.

Pre-0.9.7 ``rm`` deleted gigabytes of model weights silently — no
confirmation, no freed-space summary. This file pins the new contract:

* default: prompt ``Remove <model> (X.Y GB)? [y/N]``, default ``N``;
* empty input (just Enter) → ``Aborted.`` and exit 0;
* EOF (non-TTY / ctrl-D) → ``Aborted.`` and exit 0;
* ``-y / --yes`` → no prompt, runs the delete;
* on success a ``Freed X.Y GB`` line is printed.

The prompt tests mock the HF deletion strategy. Variant regressions use
only the temporary HF cache isolated by conftest, with real snapshot symlinks.
Size suffix matches ``rapid_mlx.cli._format_bytes`` (GB).
"""

from __future__ import annotations

import io
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from rapid_mlx import cli

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
    strategy.expected_freed_size_str = "6.0 GB"
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
    """Typing ``y`` at the prompt runs the delete and prints ``Freed X.Y GB``."""
    out, code, strategy = _invoke_rm(
        "mlx-community/Qwen3.5-9B-MLX-4bit", yes=False, stdin_text="y"
    )
    assert code == 0
    strategy.execute.assert_called_once()
    assert "Freed" in out, f"expected 'Freed' summary on success, got:\n{out}"
    assert "GB" in out, f"expected GB suffix from _format_bytes, got:\n{out}"


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
    assert "GB" in out


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
    the CLI (``_format_bytes`` → GB/MB/KB), so users don't see ``5.0 G``
    in one place and ``5.0 GB`` in another for the same model."""
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
    # ``_format_bytes`` renders 6.5 GiB as ``7.0 GB`` (one decimal place).
    assert "7.0 GB" in out, f"expected '7.0 GB' in Freed line, got:\n{out}"


@pytest.fixture
def variant_cache(monkeypatch):
    from pathlib import Path

    from huggingface_hub import scan_cache_dir
    from huggingface_hub.constants import HF_HUB_CACHE

    monkeypatch.setattr(
        "huggingface_hub.scan_cache_dir", lambda: scan_cache_dir(HF_HUB_CACHE)
    )

    root = Path(HF_HUB_CACHE) / "models--LiquidAI--LFM2.5-2.6B-MLX"
    blobs = root / "blobs"
    blobs.mkdir(parents=True)

    def link(revision, name, blob, content):
        import os

        target = blobs / blob
        target.write_bytes(content)
        path = root / "snapshots" / revision / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.symlink_to(os.path.relpath(target, path.parent))
        return path

    link("a" * 40, "config.json", "config", b"{}")
    link("a" * 40, "4bit/model.safetensors", "four", b"4" * 1000)
    link("a" * 40, "8bit/model.safetensors", "eight", b"8" * 2000)
    return root, link


def test_rm_subfolder_keeps_sibling(variant_cache, capsys):
    root, _ = variant_cache
    cli.rm_command(
        SimpleNamespace(
            model="LiquidAI/LFM2.5-2.6B-MLX",
            _original_alias="lfm2.5-2.6b-4bit",
            yes=True,
        )
    )
    assert not (root / "blobs/four").exists()
    assert (root / "blobs/eight").read_bytes() == b"8" * 2000
    assert (root / "blobs/config").exists()
    assert not (root / "snapshots" / ("a" * 40) / "4bit").exists()
    out = capsys.readouterr().out
    assert "4bit" in out and "8bit" in out
    assert f"Freed {cli._format_bytes(1000)}" in out


def test_main_rm_preserves_variant_identity(variant_cache, monkeypatch):
    root, _ = variant_cache
    monkeypatch.setattr(sys, "argv", ["rapid-mlx", "rm", "lfm2.5-2.6b-4bit", "-y"])
    cli.main()
    assert not (root / "blobs/four").exists()
    assert (root / "blobs/eight").exists()


def test_rm_last_variant_removes_repo(variant_cache, capsys):
    root, _ = variant_cache
    import shutil

    shutil.rmtree(root / "snapshots" / ("a" * 40) / "8bit")
    cli.rm_command(SimpleNamespace(model="lfm2.5-2.6b-4bit", yes=True))
    assert not root.exists()
    assert "Freed" in capsys.readouterr().out


@pytest.mark.parametrize("retained_path", ["8bit/shared.json", "tokenizer.json"])
def test_rm_preserves_blobs_referenced_in_other_snapshots(
    variant_cache, retained_path, capsys
):
    root, link = variant_cache
    link("a" * 40, "4bit/shared.json", "shared", b"shared")
    retained = link("b" * 40, retained_path, "shared", b"shared")
    link("b" * 40, "4bit/model.safetensors", "four", b"4" * 1000)
    cli.rm_command(SimpleNamespace(model="lfm2.5-2.6b-4bit", yes=True))
    assert retained.read_bytes() == b"shared"
    assert (root / "blobs/shared").exists()
    assert not (root / "blobs/four").exists()
    assert not (root / "snapshots" / ("b" * 40) / "4bit").exists()
    assert f"Freed {cli._format_bytes(1000)}" in capsys.readouterr().out


def test_rm_uncached_variant_does_not_remove_repo(variant_cache, capsys):
    import shutil

    root, _ = variant_cache
    shutil.rmtree(root / "snapshots" / ("a" * 40) / "4bit")
    with pytest.raises(SystemExit) as exc:
        cli.rm_command(SimpleNamespace(model="lfm2.5-2.6b-4bit", yes=True))
    assert exc.value.code == 1
    assert (root / "blobs/eight").exists()
    assert "4bit is not cached" in capsys.readouterr().out


@pytest.mark.parametrize("response", ["n", "y"])
def test_rm_variant_confirmation(variant_cache, monkeypatch, capsys, response):
    root, _ = variant_cache

    def confirm(prompt):
        assert "LiquidAI/LFM2.5-2.6B-MLX/4bit" in prompt
        assert "keeping: 8bit" in capsys.readouterr().out
        return response

    monkeypatch.setattr("builtins.input", confirm)
    args = SimpleNamespace(model="lfm2.5-2.6b-4bit", yes=False)
    if response == "n":
        with pytest.raises(SystemExit):
            cli.rm_command(args)
        assert (root / "blobs/four").exists()
    else:
        cli.rm_command(args)
        assert not (root / "blobs/four").exists()
    assert (root / "blobs/eight").exists()


def test_rm_variant_includes_regular_files(variant_cache, capsys):
    root, _ = variant_cache
    regular = root / "snapshots" / ("a" * 40) / "4bit/local.json"
    regular.write_bytes(b"local")
    cli.rm_command(SimpleNamespace(model="lfm2.5-2.6b-4bit", yes=True))
    assert not regular.exists()
    assert f"Freed {cli._format_bytes(1005)}" in capsys.readouterr().out


def test_rm_root_checkpoint_removes_repo(variant_cache, capsys):
    import shutil

    root, link = variant_cache
    shutil.rmtree(root / "snapshots")
    link("a" * 40, "config.json", "config", b"{}")
    link("a" * 40, "model.safetensors", "root", b"weights")
    # A repo without a catalog subfolder still uses the existing HF strategy.
    root.rename(root.parent / "models--org--single")
    cli.rm_command(SimpleNamespace(model="org/single", yes=True))
    assert not (root.parent / "models--org--single").exists()
    assert "Freed" in capsys.readouterr().out


def test_rm_only_variant_keeps_root_checkpoint(variant_cache, capsys):
    import shutil

    root, link = variant_cache
    shutil.rmtree(root / "snapshots" / ("a" * 40) / "8bit")
    root_weights = link("b" * 40, "model.safetensors", "root", b"weights")
    cli.rm_command(SimpleNamespace(model="lfm2.5-2.6b-4bit", yes=True))
    assert root_weights.read_bytes() == b"weights"
    assert (root / "blobs/config").exists()
    assert not (root / "blobs/four").exists()
    assert "(repository root)" in capsys.readouterr().out


def test_rm_variant_drops_emptied_snapshot_and_ref(variant_cache, capsys):
    import shutil

    root, link = variant_cache
    shutil.rmtree(root / "snapshots" / ("a" * 40) / "8bit")
    (root / "snapshots" / ("a" * 40) / "config.json").unlink()
    link("b" * 40, "8bit/model.safetensors", "eight", b"8" * 2000)
    refs = root / "refs"
    refs.mkdir()
    (refs / "main").write_text("b" * 40)
    (refs / "old").write_text("a" * 40)
    cli.rm_command(SimpleNamespace(model="lfm2.5-2.6b-4bit", yes=True))
    assert not (root / "snapshots" / ("a" * 40)).exists()
    assert not (refs / "old").exists()
    assert (refs / "main").read_text() == "b" * 40
    assert (root / "blobs/eight").exists()
