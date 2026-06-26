"""End-to-end pins for the `help <typo>` fuzzy-match suggestion.

When a user runs `rapid-mlx help serv`, difflib should nudge them toward
`serve` — but only when there's a plausibly close match.  No close match
falls back to the original "Run `rapid-mlx help` ..." message.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _run_cli(*cli_args: str) -> subprocess.CompletedProcess[str]:
    """Drive `python -m vllm_mlx.cli` against THIS worktree's source.

    Forcing ``PYTHONPATH=.`` keeps the child Python from picking up an
    editable install that lives at the original (non-worktree) checkout.
    """
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    return subprocess.run(
        [sys.executable, "-m", "vllm_mlx.cli", *cli_args],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=env,
    )


def test_help_typo_serv_suggests_serve() -> None:
    result = _run_cli("help", "serv")
    assert result.returncode == 1
    assert "Unknown subcommand: serv" in result.stdout
    assert "Did you mean:" in result.stdout
    assert "serve" in result.stdout
    assert "Run `rapid-mlx help` for the list of subcommands." in result.stdout


def test_help_typo_mdls_suggests_models() -> None:
    result = _run_cli("help", "mdls")
    assert result.returncode == 1
    assert "Unknown subcommand: mdls" in result.stdout
    assert "Did you mean:" in result.stdout
    assert "models" in result.stdout


def test_help_no_close_match_omits_suggestion() -> None:
    result = _run_cli("help", "zzzzz")
    assert result.returncode == 1
    assert "Unknown subcommand: zzzzz" in result.stdout
    assert "Did you mean:" not in result.stdout
    assert "Run `rapid-mlx help` for the list of subcommands." in result.stdout
