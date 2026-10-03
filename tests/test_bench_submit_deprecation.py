# SPDX-License-Identifier: Apache-2.0
"""``rapid-mlx bench --submit`` points users at the community benchmark.

The legacy board is being folded into the community leaderboard. The legacy
upload still works during the grace period, so the CLI only prints a notice
(on stderr, keeping stdout machine-readable) and then runs the same flow.
"""

from __future__ import annotations

import argparse
import importlib

import pytest

cli = importlib.import_module("rapid_mlx.cli")

NOTICE = "`rapid-mlx bench --submit` is deprecated"


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
        json=True,
    )
    base.update(over)
    return argparse.Namespace(**base)


@pytest.mark.parametrize(
    ("tier", "flow"), [(None, "_run_submit_flow"), ("all", "_run_tier_submit_flow")]
)
def test_submit_prints_the_replacement_then_still_runs(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tier: str | None,
    flow: str,
) -> None:
    called = {}

    def _fake(args, **kwargs):
        called["yes"] = True
        return 0

    monkeypatch.setattr(cli, flow, _fake)
    with pytest.raises(SystemExit) as excinfo:
        cli.bench_command(_args(tier=tier))
    assert excinfo.value.code == 0
    assert called == {"yes": True}, "the legacy flow still runs in the grace period"
    out, err = capsys.readouterr()
    assert NOTICE not in out
    assert NOTICE in err
    assert "rapid-mlx benchmark run qwen3.5-9b-4bit" in err
    assert "rapid-mlx benchmark share <run-id>" in err
    assert "https://rapidmlx.com/leaderboard" in err


def test_notice_names_the_alias_the_user_typed_not_the_resolved_repo(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli._print_bench_submit_deprecation(
        argparse.Namespace(
            model="mlx-community/Qwen3.5-9B-4bit", _original_alias="qwen3.5-9b-4bit"
        )
    )
    err = capsys.readouterr().err
    assert "rapid-mlx benchmark run qwen3.5-9b-4bit" in err
    assert "mlx-community/" not in err


def test_main_resolves_the_alias_but_the_notice_keeps_it(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import sys

    seen = {}

    def _fake_submit(args, **kwargs):
        seen["model"] = args.model
        return 0

    monkeypatch.setattr(cli, "_run_submit_flow", _fake_submit)
    monkeypatch.setattr(
        sys, "argv", ["rapid-mlx", "bench", "qwen3.5-9b-4bit", "--submit"]
    )
    with pytest.raises(SystemExit):
        cli.main()
    err = capsys.readouterr().err
    assert "rapid-mlx benchmark run qwen3.5-9b-4bit\n" in err
    assert seen["model"] != "qwen3.5-9b-4bit", (
        "main() resolves the alias for the legacy flow"
    )


def test_notice_comes_before_the_staleness_warning(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import sys

    import rapid_mlx._version_check as version_check

    monkeypatch.setattr(
        version_check,
        "print_staleness_warning_if_any",
        lambda: print("STALENESS-MARKER", file=sys.stderr),
    )
    monkeypatch.setattr(cli, "_run_submit_flow", lambda args, **kw: 0)
    with pytest.raises(SystemExit):
        cli.bench_command(_args(json=False))
    err = capsys.readouterr().err
    assert err.index(NOTICE) < err.index("STALENESS-MARKER")


def test_notice_without_a_model_uses_a_placeholder(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli._print_bench_submit_deprecation(argparse.Namespace())
    assert "rapid-mlx benchmark run <alias>" in capsys.readouterr().err


def test_plain_bench_prints_no_notice(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import rapid_mlx.bench.tier_runner as tier_runner

    monkeypatch.setattr(tier_runner, "run_tier", lambda **kwargs: 0)
    with pytest.raises(SystemExit):
        cli.bench_command(_args(submit=False, tier="smoke"))
    assert NOTICE not in capsys.readouterr().err


def test_help_marks_submit_deprecated() -> None:
    parser = cli.build_parser()
    bench = parser._subparsers._group_actions[0].choices["bench"]
    submit = next(a for a in bench._actions if "--submit" in a.option_strings)
    assert submit.help.startswith("Deprecated: use `rapid-mlx benchmark run`")
