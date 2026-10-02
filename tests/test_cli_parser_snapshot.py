"""Golden snapshot of the full ``rapid-mlx`` argparse surface.

The CLI is a public interface: every subcommand, flag, default, choice,
completer and help string is pinned in
``tests/fixtures/cli_parser_snapshot.json``. Refactors of the parser code must
leave the snapshot byte-identical; an intentional CLI change shows up as a
reviewable diff of the fixture.

Regenerate after an intentional change with::

    UPDATE_CLI_SNAPSHOT=1 python -m pytest tests/test_cli_parser_snapshot.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Any

import pytest

from rapid_mlx import cli

SNAPSHOT = Path(__file__).parent / "fixtures" / "cli_parser_snapshot.json"
_VERSION_PLACEHOLDER = "<version>"
_PY310_BOOL_DEFAULT_SUFFIX = " (default: %(default)s)"


def _callable_name(value: Any) -> str | None:
    if value is None:
        return None
    module = getattr(value, "__module__", None) or type(value).__module__
    name = getattr(value, "__qualname__", None) or type(value).__qualname__
    return f"{module}.{name}"


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if callable(value):
        return {"callable": _callable_name(value)}
    return {"repr": repr(value)}


def _action(action: argparse.Action, version: str) -> dict[str, Any]:
    help_text = action.help
    if (
        isinstance(action, argparse.BooleanOptionalAction)
        and help_text
        and help_text.endswith(_PY310_BOOL_DEFAULT_SUFFIX)
    ):
        # Python 3.10's BooleanOptionalAction appends this itself.
        help_text = help_text[: -len(_PY310_BOOL_DEFAULT_SUFFIX)]
    entry: dict[str, Any] = {
        "class": _callable_name(type(action)),
        "option_strings": list(action.option_strings),
        "dest": action.dest,
        "nargs": _jsonable(action.nargs),
        "const": _jsonable(action.const),
        "default": _jsonable(action.default),
        "type": _callable_name(action.type),
        "help": help_text,
        "metavar": _jsonable(action.metavar),
        "completer": _callable_name(getattr(action, "completer", None)),
    }
    if action.option_strings:
        # Positional ``required`` is derived from nargs, and the derivation
        # changed in Python 3.13 (``nargs="..."`` is no longer required).
        entry["required"] = action.required
    if isinstance(action, argparse._VersionAction):
        entry["version"] = (action.version or "").replace(version, _VERSION_PLACEHOLDER)
    if isinstance(action, argparse._SubParsersAction):
        entry["subcommands"] = {
            name: _parser(sub, version) for name, sub in action.choices.items()
        }
        entry["choices_help"] = {a.dest: a.help for a in action._choices_actions}
    elif action.choices is not None:
        entry["choices"] = _jsonable(list(action.choices))
    return entry


def _parser(parser: argparse.ArgumentParser, version: str) -> dict[str, Any]:
    return {
        "class": _callable_name(type(parser)),
        "prog": parser.prog,
        "description": parser.description,
        "epilog": parser.epilog,
        "formatter_class": _callable_name(parser.formatter_class),
        "allow_abbrev": parser.allow_abbrev,
        "defaults": _jsonable(parser._defaults),
        "groups": [
            {
                "title": group.title,
                "description": group.description,
                "dests": [a.dest for a in group._group_actions],
            }
            for group in parser._action_groups
        ],
        "mutually_exclusive": [
            {"required": g.required, "dests": [a.dest for a in g._group_actions]}
            for g in parser._mutually_exclusive_groups
        ],
        "actions": [_action(a, version) for a in parser._actions],
        # Rendered ``format_help()`` is deliberately not pinned: argparse's
        # layout changes between the supported Python minors (3.10-3.13).
        # Every input to it (help strings, groups, order, metavars) is.
    }


def _snapshot(monkeypatch: pytest.MonkeyPatch) -> str:
    # ``prog`` follows sys.argv[0]; pin it so the runner doesn't leak in.
    monkeypatch.setattr(sys, "argv", ["rapid-mlx"])
    # Normalize versions after the fact instead of monkeypatching the version
    # helper, so the snapshot does not depend on which module defines it.
    data = _parser(cli.build_parser(), cli._resolve_cli_version())
    text = json.dumps(data, indent=1, sort_keys=True, ensure_ascii=False) + "\n"
    # Install hints embed the installed distribution version; a version bump
    # must not churn the snapshot.
    return text.replace(f"=={pkg_version('rapid-mlx')}", f"=={_VERSION_PLACEHOLDER}")


def test_cli_parser_matches_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    actual = _snapshot(monkeypatch)
    if os.environ.get("UPDATE_CLI_SNAPSHOT") == "1":
        SNAPSHOT.write_text(actual)
    expected = SNAPSHOT.read_text()
    assert actual == expected, (
        "rapid-mlx CLI surface changed. If intentional, regenerate with "
        "UPDATE_CLI_SNAPSHOT=1 and review the fixture diff."
    )
