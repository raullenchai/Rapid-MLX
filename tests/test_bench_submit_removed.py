# SPDX-License-Identifier: Apache-2.0
"""``rapid-mlx bench --submit`` no longer submits anything.

The legacy board is closed (owner decision 2026-10-02); its old runs are
folded into the community leaderboard. ``--submit`` is now a usage error
(exit 2, stderr) that points to ``benchmark run`` + ``benchmark share`` and
runs nothing: no benchmark, no consent prompt, no network call.
"""

from __future__ import annotations

import argparse
import importlib
import sys

import pytest

cli = importlib.import_module("rapid_mlx.cli")

MESSAGE = "`rapid-mlx bench --submit` no longer submits results"


def _args(**over) -> argparse.Namespace:
    base = dict(
        model="qwen3.5-9b-4bit",
        tier=None,
        submit=True,
        base_url=None,
        sampled=False,
        force_disk_check=False,
        notes=None,
        repo_root=None,
    )
    base.update(over)
    return argparse.Namespace(**base)


def _forbid(monkeypatch: pytest.MonkeyPatch) -> None:
    """Any benchmark, upload or network path reached is a failure."""

    def boom(*args, **kwargs):
        raise AssertionError("bench --submit must not run or send anything")

    monkeypatch.setattr(cli, "_run_submit_flow", boom)
    monkeypatch.setattr(cli, "_run_tier_submit_flow", boom)
    import rapid_mlx._mlx_compat as mlx_compat
    import rapid_mlx._version_check as version_check
    import rapid_mlx.community_bench.upload as upload

    monkeypatch.setattr(upload, "post_submission", boom)
    monkeypatch.setattr(version_check, "print_staleness_warning_if_any", boom)
    monkeypatch.setattr(mlx_compat, "install", boom)


@pytest.mark.parametrize("tier", [None, "all", "speed"])
def test_submit_exits_2_with_the_replacement_and_runs_nothing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tier: str | None,
) -> None:
    _forbid(monkeypatch)
    with pytest.raises(SystemExit) as excinfo:
        cli.bench_command(_args(tier=tier))
    assert excinfo.value.code == 2
    out, err = capsys.readouterr()
    assert out == ""
    assert MESSAGE in err
    assert "nothing was run or sent" in err
    assert "rapid-mlx benchmark run qwen3.5-9b-4bit\n" in err
    assert "rapid-mlx benchmark share <run-id>" in err
    assert "https://rapidmlx.com/leaderboard" in err


def test_message_names_the_alias_the_user_typed_not_the_resolved_repo(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        cli._refuse_bench_submit(
            argparse.Namespace(
                model="mlx-community/Qwen3.5-9B-4bit",
                _original_alias="qwen3.5-9b-4bit",
            )
        )
    err = capsys.readouterr().err
    assert "rapid-mlx benchmark run qwen3.5-9b-4bit\n" in err
    assert "mlx-community/" not in err


def test_main_entry_point_refuses_with_the_typed_alias(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _forbid(monkeypatch)

    def boom(*args, **kwargs):
        raise AssertionError("bench --submit must stop at the parse boundary")

    import rapid_mlx.model_aliases as model_aliases
    import rapid_mlx.telemetry.consent_runtime as consent_runtime

    monkeypatch.setattr(model_aliases, "resolve_model", boom)
    monkeypatch.setattr(consent_runtime, "startup", boom)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(
        sys, "argv", ["rapid-mlx", "bench", "qwen3.5-9b-4bit", "--submit"]
    )
    with pytest.raises(SystemExit) as excinfo:
        cli.main()
    assert excinfo.value.code == 2
    assert "rapid-mlx benchmark run qwen3.5-9b-4bit\n" in capsys.readouterr().err


def test_message_without_a_model_uses_a_placeholder(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        cli._refuse_bench_submit(argparse.Namespace())
    assert "rapid-mlx benchmark run <alias>" in capsys.readouterr().err


def test_plain_bench_is_unaffected(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import rapid_mlx.bench.tier_runner as tier_runner

    monkeypatch.setattr(tier_runner, "run_tier", lambda **kwargs: 0)
    with pytest.raises(SystemExit) as excinfo:
        cli.bench_command(_args(submit=False, tier="smoke"))
    assert excinfo.value.code == 0
    assert MESSAGE not in capsys.readouterr().err


def test_help_marks_submit_removed() -> None:
    parser = cli.build_parser()
    bench = parser._subparsers._group_actions[0].choices["bench"]
    submit = next(a for a in bench._actions if "--submit" in a.option_strings)
    assert submit.help.startswith("Removed: exits with an error")
