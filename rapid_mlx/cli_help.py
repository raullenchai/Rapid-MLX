"""Top-level ``rapid-mlx --help`` layout: commands grouped by purpose.

argparse lists subcommands in registration order behind a ``{a,b,c,...}``
brace list, which put niche commands (``system-one``, ``cua``) ahead of
``chat`` and printed the 30-name list twice. The top-level parser now hides
argparse's own subcommand rows and renders this ordered table instead. The
one-line description of each command still comes from its ``add_parser(help=)``
string, so the help text has a single source.

``tests/test_cli_help_groups.py`` asserts that every registered subcommand
appears in exactly one group, so a new subcommand cannot silently vanish from
the help.
"""

from __future__ import annotations

import argparse

# (heading, [primary command names]) in display order. Aliases registered via
# ``add_parser(aliases=...)`` are folded onto their primary row.
COMMAND_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("GET STARTED", ("chat", "serve", "pull", "models", "recipe", "import")),
    ("CODING AGENTS", ("agents", "launch", "start", "connect", "share")),
    (
        "MANAGE",
        (
            "ls",
            "rm",
            "alias",
            "ps",
            "info",
            "upgrade",
            "doctor",
            "telemetry",
            "feedback",
            "version",
            "help",
        ),
    ),
    (
        "ADVANCED / EXPERIMENTAL",
        ("bench", "benchmark", "service", "system-one", "cua"),
    ),
)

USAGE = "%(prog)s [options] <command> [<args>]"

EPILOG = """\
Examples:
  rapid-mlx                                   # guided start for this Mac
  rapid-mlx chat qwen3.5-4b-4bit              # chat in the terminal
  rapid-mlx serve qwen3.5-9b-4bit --port 8000 # OpenAI/Anthropic-compatible server
  rapid-mlx launch claude-code --model qwen3.5-9b-4bit
  rapid-mlx recipe                            # best smart + fast models for this Mac

Run `rapid-mlx <command> --help` for a command's options.
Docs: https://rapidmlx.com/docs/
"""


def _aliases_by_primary(
    subparsers: argparse._SubParsersAction,
) -> dict[str, list[str]]:
    """Map each primary subcommand name to its registered aliases."""
    primary_of: dict[int, str] = {}
    for action in subparsers._choices_actions:
        parser = subparsers.choices.get(action.dest)
        if parser is not None:
            primary_of[id(parser)] = action.dest
    aliases: dict[str, list[str]] = {name: [] for name in primary_of.values()}
    for name, parser in subparsers.choices.items():
        primary = primary_of.get(id(parser))
        if primary is not None and name != primary:
            aliases[primary].append(name)
    return aliases


def render_command_groups(subparsers: argparse._SubParsersAction) -> str:
    """Render the grouped command list for the top-level help."""
    helps = {a.dest: (a.help or "") for a in subparsers._choices_actions}
    aliases = _aliases_by_primary(subparsers)
    labels = {
        name: name + (f" ({', '.join(aliases[name])})" if aliases.get(name) else "")
        for name in helps
    }
    width = max(len(label) for label in labels.values()) + 2
    blocks: list[str] = []
    for heading, names in COMMAND_GROUPS:
        rows = [f"  {labels[n]:<{width}}{helps[n]}" for n in names if n in helps]
        blocks.append("\n".join([heading, *rows]))
    return "\n\n".join(blocks)


def apply_grouped_help(
    parser: argparse.ArgumentParser,
    subparsers: argparse._SubParsersAction,
    identity: str,
) -> None:
    """Replace argparse's subcommand listing with the grouped layout.

    Called once every subcommand is registered. Hiding the subparsers action
    from help (``argparse.SUPPRESS``) removes both copies of the brace list;
    the explicit usage string keeps ``<command>`` visible.
    """
    subparsers.help = argparse.SUPPRESS
    parser.usage = USAGE
    parser.description = f"{identity}\n\n{render_command_groups(subparsers)}"
    parser.epilog = EPILOG
