"""Cheetah launch banner for the rapid-mlx CLI.

The rapid-mlx mascot is a cheetah — the fastest land animal, on the fastest
local AI engine for Apple Silicon. This module renders a compact ASCII
cheetah-face emblem (the dark "tear marks" running from the eyes to the
muzzle and the rosette spots are a cheetah's two signatures, and the two
elements survive a monochrome terminal) plus a small ``rapid-mlx`` wordmark
and version line.

Rendering is deliberately pure: it returns a string and takes no I/O or
machine-state probes, so ``main()`` decides when to print it (interactive
TTY only, never for machine-readable output, honoring ``NO_COLOR``).
"""

from __future__ import annotations

_ART = [
    "              ▄▄████████▄▄",
    "          ▄▄███▀▀    ▀▀███▄▄",
    "        ▄████    ●    ●    ████▄",
    "       ███     ●      ●     ███",
    "      ▐██    ▄▄▄▄▄▄▄▄▄▄▄▄    ██▌",
    "      ██   ▄████      ████▄   ██",
    "      ██  ▐████      ████▌   ██",
    "      ██  ▐████      ████▌   ██",
    "      ██   ▀███▄▄▄▄▄▄███▀    ██",
    "       ▀█    ▀██████████▀   ▐█",
    "       ▐█▄     ▀▀▀▀▀▀▀▀    ▄█▌",
    "        ▀██  ● ●  ●  ●   ██▀",
    "          ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀",
]

_WORDMARK = "r a p i d - m l x"
_VERSION_FMT = "Rapid-MLX {version}"

# ANSI 256-color face for the tawny cheetah coat (214) and the rust-brown
# tear marks / rosette spots (130). The wordmark is intentionally left plain
# so "rapid-mlx" stays legible and machine-greppable. When color is disabled
# the plain art lines are emitted verbatim (already monochrome).
_COAT = "\x1b[38;5;214m"  # warm tawny/orange — the cheetah's coat
_DARK = "\x1b[38;5;130m"  # rust-brown — tear marks + rosette spots
_RESET = "\x1b[0m"


def render_banner(version: str, *, color: bool) -> str:
    """Return the full launch banner (art + wordmark + version line).

    ``color`` toggles the tawny/dark ANSI pass on the art; the wordmark is
    always left plain so the "rapid-mlx" text stays legible and machine-
    greppable. No trailing newline.
    """
    # Always copy: _ART is a module-level list, so aliasing it and appending
    # would mutate the shared constant and duplicate the footer on the next
    # call (a pure-function invariant the unit test locks down).
    lines = list(_ART) if not color else [_colorize(line) for line in _ART]
    lines.append("")
    lines.append(_WORDMARK)
    lines.append(_VERSION_FMT.format(version=version))
    return "\n".join(lines)


def _colorize(line: str) -> str:
    """Paint one art row: spots/tear-marks rust-brown, the rest tawny."""
    painted: list[str] = []
    for ch in line:
        if ch in "●":
            painted.append(_DARK + ch + _RESET)
        elif ch in "█▀▄▐▌":
            painted.append(_COAT + ch + _RESET)
        else:
            painted.append(ch)
    return "".join(painted)


# Subcommands whose only job is to print a machine- or script-greppable result
# are kept byte-clean: a decorative banner in front of ``rapid-mlx version``
# would break a wrapper scraping ``rapid-mlx X.Y.Z``. ``--help``/``--version``
# also never reach the gate (argparse exits during parse), but the equivalent
# ``help``/``version`` SUBCOMMANDS do, so they're suppressed here by name.
_BYTE_CLEAN_SUBCOMMANDS = frozenset({"version", "help"})


def should_show_banner(
    *,
    command: str | None,
    json_output: bool,
    no_banner: bool,
    stdout_isatty: bool,
    stdin_isatty: bool,
) -> bool:
    """Decide whether the launch banner should print for a given invocation.

    Pure decision helper so ``cli.py``'s inline gate matches a unit test one
    for one. Rules (see ``main()`` for the full rationale):

      * default NO — any machine-facing or non-terminal path keeps stdout
        byte-clean (scripts parsing ``rapid-mlx`` output must depend on it).
      * ``no_banner`` (the ``--no-banner`` flag) disables it outright, as does
        the ``RAPID_MLX_NO_BANNER`` env var (folded in by the caller).
      * ``json_output`` (``models/recipe/connect --json``) disables it — a
        banner in front of a machine payload would corrupt it.
      * stdout must be a real terminal (pipe/redirect -> no banner).
      * ``version`` / ``help`` subcommands are byte-clean (see
        ``_BYTE_CLEAN_SUBCOMMANDS``), matching the argparse ``--version`` /
        ``-h`` paths that exit before this gate runs.
      * bare ``rapid-mlx`` requires stdin to be a terminal too — the
        nameplate block it precedes only prints when BOTH streams are ttys,
        so ``rapid-mlx </dev/null`` must not get a banner-then-help splice.
    """
    if no_banner or json_output or not stdout_isatty:
        return False
    if command in _BYTE_CLEAN_SUBCOMMANDS:
        return False
    if command is None and not stdin_isatty:
        return False
    return True
