"""Block-letter RAPID-MLX launch banner for the rapid-mlx CLI.

The six-row wordmark uses the brand amber gradient with dim brown shadows,
followed by a plain tagline and version line.

Rendering is deliberately pure: it returns a string and takes no I/O or
machine-state probes, so ``main()`` decides when to print it (interactive
TTY only, never for machine-readable output, honoring ``NO_COLOR``).
"""

from __future__ import annotations

_ART = [
    "██████╗  █████╗ ██████╗ ██╗██████╗       ███╗   ███╗██╗     ██╗  ██╗",
    "██╔══██╗██╔══██╗██╔══██╗██║██╔══██╗      ████╗ ████║██║     ╚██╗██╔╝",
    "██████╔╝███████║██████╔╝██║██║  ██║█████╗██╔████╔██║██║      ╚███╔╝",
    "██╔══██╗██╔══██║██╔═══╝ ██║██║  ██║╚════╝██║╚██╔╝██║██║      ██╔██╗",
    "██║  ██║██║  ██║██║     ██║██████╔╝      ██║ ╚═╝ ██║███████╗██╔╝ ██╗",
    "╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝╚═════╝       ╚═╝     ╚═╝╚══════╝╚═╝  ╚═╝",
]

_TAGLINE = "Local AI for Apple Silicon — serve models and wire them to agents"
_VERSION_FMT = "Rapid-MLX {version}"

# Brand amber (#EFA23A, the README banner and Desktop dark-mode accent)
# shaded from a lighter top (#F6BE63) to the light-mode accent (#C9821F).
# Truecolor terminals get the exact brand values; others get the nearest
# xterm-256 colors. Shadows are a dim brown so the letters stay the focus.
_ROW_RGB = (
    (246, 190, 99),
    (243, 177, 79),
    (239, 162, 58),
    (224, 149, 48),
    (212, 139, 39),
    (201, 130, 31),
)
_ROW_256 = (214, 214, 214, 172, 172, 172)
_SHADOW_RGB = (122, 82, 38)
_SHADOW_256 = 94
_RESET = "\x1b[0m"


def render_banner(version: str, *, color: bool, truecolor: bool = False) -> str:
    """Return the wordmark, plain tagline, and version without a trailing newline.

    ``color`` applies the brand amber gradient to the block letters and dim
    brown to their shadows; ``truecolor`` selects exact 24-bit brand values
    instead of the nearest xterm-256 colors. With color disabled, the art is
    emitted verbatim.
    """
    # Always copy: appending to the module-level artwork would duplicate the
    # footer on subsequent calls and violate the pure-function contract.
    lines = (
        [_colorize(line, row, truecolor) for row, line in enumerate(_ART)]
        if color
        else list(_ART)
    )
    lines.append("")
    lines.append(_TAGLINE)
    lines.append(_VERSION_FMT.format(version=version))
    return "\n".join(lines)


def _sgr(rgb: tuple[int, int, int], xterm: int, truecolor: bool) -> str:
    if truecolor:
        red, green, blue = rgb
        return f"\x1b[38;2;{red};{green};{blue}m"
    return f"\x1b[38;5;{xterm}m"


def _colorize(line: str, row: int, truecolor: bool) -> str:
    """Paint one wordmark row: amber blocks, dim brown shadow strokes."""
    block = _sgr(_ROW_RGB[row], _ROW_256[row], truecolor)
    shadow = _sgr(_SHADOW_RGB, _SHADOW_256, truecolor)
    painted: list[str] = []
    for ch in line:
        if ch == "█":
            painted.append(block + ch + _RESET)
        elif ch in "╗╔╝╚═║":
            painted.append(shadow + ch + _RESET)
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
      * bare ``rapid-mlx`` never shows it: the front door
        (``rapid_mlx/front_door.py``) is a one-screen menu with its own
        one-line header, and a launch banner would push its actions off a
        short terminal.
    """
    del stdin_isatty  # kept in the signature for callers; no rule needs it now
    if no_banner or json_output or not stdout_isatty:
        return False
    if command in _BYTE_CLEAN_SUBCOMMANDS:
        return False
    return command is not None
