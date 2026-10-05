# SPDX-License-Identifier: Apache-2.0
"""Top-level ``rapid-mlx --help`` is grouped by purpose (rapid_mlx/cli_help.py)."""

from __future__ import annotations

import argparse
import re
import sys

import pytest

import rapid_mlx.cli as cli
from rapid_mlx.cli_help import COMMAND_GROUPS
from rapid_mlx.cli_parser import CLI_IDENTITY, build_parser


def _subparsers(parser: argparse.ArgumentParser) -> argparse._SubParsersAction:
    return next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))


def _primary_names(sp: argparse._SubParsersAction) -> list[str]:
    return [a.dest for a in sp._choices_actions]


@pytest.fixture(scope="module")
def parser() -> argparse.ArgumentParser:
    return build_parser()


@pytest.fixture(scope="module")
def help_text(parser) -> str:
    return parser.format_help()


def test_every_registered_subcommand_is_in_exactly_one_group(parser):
    grouped = [name for _heading, names in COMMAND_GROUPS for name in names]
    assert sorted(grouped) == sorted(_primary_names(_subparsers(parser)))
    assert len(grouped) == len(set(grouped))


def test_every_registered_subcommand_appears_exactly_once_in_help(parser, help_text):
    sp = _subparsers(parser)
    for name in sp.choices:  # primary names AND aliases
        rows = re.findall(
            rf"^  {re.escape(name)}\b|^  \S+ \({re.escape(name)}\)",
            help_text,
            flags=re.MULTILINE,
        )
        assert len(rows) == 1, (name, rows)


def test_groups_are_in_purpose_order(help_text):
    headings = [heading for heading, _names in COMMAND_GROUPS]
    positions = [help_text.index(f"\n{h}\n") for h in headings]
    assert positions == sorted(positions)
    assert headings[0] == "GET STARTED"
    assert headings[-1] == "ADVANCED / EXPERIMENTAL"
    # Core verbs precede niche ones.
    assert help_text.index("  chat (run)") < help_text.index("  system-one")


def test_brace_list_is_gone_and_aliases_are_folded(help_text):
    assert "{" not in help_text.split("options:")[0]
    assert "  chat (run)" in help_text
    assert "  upgrade (update)" in help_text
    assert not re.search(r"^  run\b", help_text, flags=re.MULTILINE)
    assert not re.search(r"^  update\b", help_text, flags=re.MULTILINE)


def test_usage_and_identity(help_text):
    assert help_text.startswith("usage: ")
    assert "<command>" in help_text.splitlines()[0]
    assert " ".join(CLI_IDENTITY.split()) in " ".join(help_text.split())
    assert "Docs: https://rapidmlx.com/docs/" in help_text


def test_invalid_command_still_errors_with_exit_2(parser, capsys):
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["definitely-not-a-command"])
    assert exc.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_help_subcommand_prints_the_grouped_help(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["rapid-mlx", "help"])
    monkeypatch.setattr("rapid_mlx.telemetry.consent_runtime.startup", lambda **_: None)
    monkeypatch.setattr(cli, "_start_v2_lifecycle", lambda _command: None)
    cli.main()
    out = capsys.readouterr().out
    assert "GET STARTED" in out
    assert "ADVANCED / EXPERIMENTAL" in out


def test_top_level_help_flag(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["rapid-mlx", "--help"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert out.startswith("usage: rapid-mlx")
    for heading, _names in COMMAND_GROUPS:
        assert heading in out
