"""Bare ``rapid-mlx``: an interactive front door instead of a manual.

* **TTY (stdin AND stdout are terminals, TERM is not ``dumb``, no ``CI``):**
  one screen of at most ~20 rows: header (version · chip · RAM), a one-line
  product statement, the model Enter will chat with (on a cold cache the
  ~3 GB "quick start" model on every Mac; a downloaded recipe pick for this
  Mac wins when there is one) and, labelled separately, "Best for this Mac"
  (``rapid-mlx recipe``'s smart pick, key ``b``). ``c`` connects a detected
  coding agent; ``s`` starts a server; ``m`` picks another model;
  ``q``/Esc/Ctrl-C exits 0. Every action
  echoes the exact command (``→ rapid-mlx chat <model>``) and then runs it
  in-process through the normal CLI path, so the second session needs no menu.
* **Anything else (pipe, CI, ``TERM=dumb``):** a short, deterministic block
  on stderr with the recommended model and copy-paste non-interactive
  commands; the exit status stays 1 so scripts that relied on the old
  help-and-exit-1 keep working.

Rendering is probe-only: no network, no model load, no download and no
server start until the user presses a key. Every probe is fail-silent; if
building the screen fails the caller falls back to the plain help.
"""

from __future__ import annotations

import functools
import os
import re
import socket
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from rapid_mlx.first_run import FIRST_RUN_MODEL

# The quick-start model Enter downloads on a cold cache, on every Mac: small
# (about 3 GB), tool-calling-reliable, and the same starter ``rapid-mlx chat``
# picks with no model. It is labelled "quick start", distinct from the
# "Best for this Mac" recipe pick, so the screen never contradicts recipe.
QUICK_START_MODEL = FIRST_RUN_MODEL
PRODUCT_LINE = "Local OpenAI- and Anthropic-compatible LLM server for Apple Silicon."
DOCS_URL = "https://rapidmlx.com/docs/"
DEFAULT_PORT = 8000
_PORT_SEARCH_SPAN = 100
_PICKER_LIMIT = 5
# Keep in sync with landing/scripts/build_zh_memory_table.py on the site.
# Values are (chat, coding), indexed by minimum RAM in GB.
_MEMORY_LADDER = {
    8: ("qwen3-8b-4bit", "qwen3-8b-4bit"),
    16: ("qwen3.5-9b-4bit", "deepseek-coder-v2-lite-16b-4bit"),
    24: ("qwen3.5-27b-4bit", "qwen3-coder-30b-4bit"),
    32: ("qwen3.8-27b-4bit", "qwen3-coder-30b-4bit"),
    48: ("qwen3.6-35b-4bit", "qwen3-coder-30b-4bit"),
    64: ("qwen3.6-35b-optiq-4bit", "qwen3-coder-30b-4bit"),
    96: ("gpt-oss-120b", "qwen3-coder-next-80b-4bit"),
    128: ("qwen3.5-122b-mxfp4", "qwen3-coder-next-80b-4bit"),
    192: ("minimax-m2.7-mxfp4", "qwen3-coder-next-80b-8bit"),
}
# Measured decode speeds: docs/benchmarks/model-recommendation-measurements.json.
# Newer families precede older ones; quick start is reserved as a fallback.
_MEASURED_PICKER_MODELS = (
    "qwen3.8-27b-4bit",
    "qwen3.6-35b-4bit",
    "qwen3.6-27b-4bit",
    "qwen3.5-35b-4bit",
    "qwen3.5-9b-4bit",
    "gemma-4-26b-4bit",
    "gemma-4-12b-4bit",
    "lfm2.5-8b-a1b-4bit",
    "lfm2.5-2.6b-4bit",
    "lfm2.5-1b-4bit",
    QUICK_START_MODEL,
)
_ROLE_LABELS = {"smart": "Smart", "fast": "Fast"}
_AGENT_LABELS = {
    "claude-code": "Claude Code",
    "cline": "Cline",
    "continue-dev": "Continue",
}


class FrontDoorUnavailableError(Exception):
    """The screen could not be built; the caller shows the plain help."""


# Keys returned by :func:`read_key` (raw characters pass through unchanged).
KEY_ENTER = "enter"
KEY_QUIT = "quit"
KEY_UNKNOWN = "unknown"
# Redraw in place after the picker so the menu stays one screen.
_CLEAR_SCREEN = "\x1b[H\x1b[2J"


@dataclass
class ServerInfo:
    port: int
    model: str
    # False when the server renamed its API model (--served-model-name): the
    # front door cannot name it to ``chat``/``launch``, so it only shows it.
    attachable: bool = True


@dataclass
class FrontDoorState:
    version: str
    chip: str | None
    ram_gb: float
    picks: list[dict]
    cached: list[str] = field(default_factory=list)
    last_used: str | None = None
    server: ServerInfo | None = None
    agent: str | None = None
    selected: str = ""
    quick_start_size: str = "~3 GB"
    local_default: str | None = None
    local_default_cataloged: bool = True

    @property
    def attach_port(self) -> int | None:
        """Port of a running server already serving the selected model."""
        if (
            self.server is not None
            and self.server.attachable
            and self.server.model == self.selected
        ):
            return self.server.port
        return None


# --------------------------------------------------------------------------
# Probes (all fail-silent)
# --------------------------------------------------------------------------
def _chip_label() -> str | None:
    if sys.platform != "darwin":
        return None
    try:
        out = subprocess.run(
            ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=1,
            check=False,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return out or None


def _running_server() -> ServerInfo | None:
    """The lowest-port running ``rapid-mlx serve`` with a plain model name."""
    try:
        from rapid_mlx.cli import _scan_running_servers

        rows = _scan_running_servers()
    except Exception:
        return None
    servers: list[ServerInfo] = []
    for _pid, port, model, _uptime in rows:
        try:
            port_num = int(port)
        except (TypeError, ValueError):
            continue
        # ``ps`` renders a --served-model-name override as "name (model)";
        # the API identity is the leading token.
        name = str(model).split(" (", 1)[0]
        if name and name != "(unknown)":
            servers.append(ServerInfo(port_num, name, " (" not in str(model)))
    return min(servers, key=lambda s: s.port) if servers else None


def _picks_for(ram_gb: float, cache_rows) -> list[dict]:
    from rapid_mlx.cli import _annotate_recipe_picks
    from rapid_mlx.recommendations import recommendation_payload

    payload = recommendation_payload(ram_gb, validate_catalog=False)
    _annotate_recipe_picks(payload, cache_rows)
    return list(payload["picks"])


def quick_start_size_label(cache_rows=None) -> str:
    """``~3 GB``-style one-time download size of the quick-start model."""
    try:
        from rapid_mlx.model_aliases import resolve_profile
        from rapid_mlx.model_sizes import size_bytes

        profile = resolve_profile(QUICK_START_MODEL)
        size = size_bytes(profile.hf_path) if profile is not None else None
    except Exception:
        size = None
    if not size:
        return "~3 GB"
    return f"~{max(1, round(size / float(1 << 30)))} GB"


def best_pick(picks: Sequence[dict]) -> dict:
    """ "Best for this Mac": the recipe tier's smart pick (recipe lists it first)."""
    for pick in picks:
        if pick.get("role") == "smart":
            return pick
    return picks[0]


def _can_chat(model: str) -> bool:
    """False only for a registered alias that is not a text-chat model
    (embedding, image, video); repo ids and paths get the benefit of doubt."""
    try:
        from rapid_mlx.model_aliases import list_profiles

        profile = list_profiles().get(model)
    except Exception:
        return False
    return profile is None or profile.modality == "text"


def _default_model(state: FrontDoorState) -> str:
    """Server model → last used → a cached recipe pick for this Mac (best
    first) → a policy-ranked local model → the quick-start model."""
    if (
        state.server is not None
        and state.server.attachable
        and _can_chat(state.server.model)
    ):
        return state.server.model
    if state.last_used and state.last_used in state.cached:
        return state.last_used
    best = best_pick(state.picks)
    for pick in [best, *(p for p in state.picks if p is not best)]:
        if pick.get("cached") or pick["alias"] in state.cached:
            return str(pick["alias"])
    return (
        state.local_default
        if QUICK_START_MODEL not in state.cached and state.local_default
        else QUICK_START_MODEL
    )


def gather_state(version: str) -> FrontDoorState:
    """Probe this Mac (no network, no model load) for the front door."""
    from rapid_mlx import first_run
    from rapid_mlx.recommendations import physical_ram_gb

    try:
        from rapid_mlx.cli import _scan_hf_cache_models

        cache_rows = _scan_hf_cache_models()
    except Exception:
        cache_rows = []
    ram_gb = physical_ram_gb()
    cached = [
        alias
        for alias, _mtime in first_run.cached_known_aliases(cache_rows)
        if first_run._is_chat_alias(alias)
    ]
    state = FrontDoorState(
        version=version,
        chip=_chip_label(),
        ram_gb=ram_gb,
        picks=_picks_for(ram_gb, cache_rows),
        cached=cached,
        last_used=first_run.last_used_model(),
        server=_running_server(),
        agent=first_run.preferred_agent(),
        quick_start_size=quick_start_size_label(),
    )
    if QUICK_START_MODEL not in cached and _default_model(state) == QUICK_START_MODEL:
        from rapid_mlx.local_models import select_default_local_model

        chosen = select_default_local_model(ram_gb)
        if chosen is not None:
            state.local_default = chosen.model
            state.local_default_cataloged = chosen.cataloged
    state.selected = _default_model(state)
    return state


def first_free_port(
    start: int = DEFAULT_PORT,
    span: int = _PORT_SEARCH_SPAN,
    *,
    is_free: Callable[[int], bool] | None = None,
) -> int | None:
    """First loopback port at or after ``start`` that accepts a bind, or
    ``None`` when the whole range is taken. (``serve`` still reports a clean
    bind error if another process takes the port in between.)"""
    check = is_free or _port_is_free
    for port in range(start, start + span):
        if check(port):
            return port
    return None


def _port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


# --------------------------------------------------------------------------
# Rendering (pure)
# --------------------------------------------------------------------------
def _gb(value: float) -> str:
    return f"{value:g}" if value == int(value) else f"{value:.1f}"


def _header(state: FrontDoorState) -> str:
    parts = [f"rapid-mlx {state.version}"]
    if state.chip:
        parts.append(state.chip)
    if state.ram_gb > 0:
        parts.append(f"{_gb(round(state.ram_gb, 1))} GB")
    return " · ".join(parts)


def _pick_facts(pick: dict) -> str:
    facts = [_ROLE_LABELS.get(pick.get("role", ""), pick.get("role", ""))]
    if pick.get("caveat"):
        facts.append(pick["caveat"])
    if pick.get("tokens_per_sec") is not None:
        facts.append(f"~{round(pick['tokens_per_sec'])} tok/s")
    if pick.get("cached"):
        facts.append("downloaded")
    elif pick.get("download_size_gb") is not None:
        facts.append(f"{pick['download_size_gb']:.1f} GB download")
    if pick.get("disk_fit") is False:
        facts.append("not enough free disk")
    return " · ".join(facts)


def agent_label(agent: str) -> str:
    return _AGENT_LABELS.get(agent, agent)


def _quick_start_facts(state: FrontDoorState) -> str:
    if QUICK_START_MODEL in state.cached or any(
        p["alias"] == QUICK_START_MODEL and p.get("cached") for p in state.picks
    ):
        return "quick start · downloaded"
    return f"quick start · {state.quick_start_size} download"


def _best_facts(pick: dict) -> str:
    facts = [_ROLE_LABELS.get(pick.get("role", ""), pick.get("role", ""))]
    if pick.get("caveat"):
        facts.append(pick["caveat"])
    if pick.get("cached"):
        facts.append("downloaded")
    elif pick.get("download_size_gb") is not None:
        facts.append(f"{pick['download_size_gb']:.1f} GB")
    if pick.get("disk_fit") is False:
        facts.append("not enough free disk")
    return ", ".join(facts)


def render_screen(state: FrontDoorState) -> str:
    """The front-door screen (no ANSI). At most ~20 rows."""
    lines = [_header(state), PRODUCT_LINE]
    if state.server is not None:
        lines.append(f"Server running on :{state.server.port} · {state.server.model}")
    if state.last_used and state.last_used in state.cached:
        lines.append(f"Last used: {state.last_used}")
    lines.append("")

    model = state.selected
    best = best_pick(state.picks)
    if model == QUICK_START_MODEL:
        target = f"{model} ({_quick_start_facts(state)})"
    elif model == state.local_default:
        label = (
            "downloaded"
            if state.local_default_cataloged
            else "on this Mac · not catalog-tested"
        )
        target = f"{model} ({label})"
    elif model in state.cached or (model == best["alias"] and best.get("cached")):
        target = f"{model} (downloaded)"
    else:
        target = model
    if model == best["alias"]:
        target += " · best for this Mac"
    lines.append(f"Ready to chat: {target}")
    if best["alias"] != model:
        lines.append(
            f"Best for this Mac: {best['alias']} ({_best_facts(best)}) — press b"
        )
    lines.append("")

    port = state.attach_port
    if port is not None:
        lines.append(f"  Enter  Chat with {model} on :{port}")
    else:
        lines.append(f"  Enter  Start chatting with {model}")
    if best["alias"] != model:
        lines.append(
            f"  b      Chat with the best model for this Mac ({best['alias']})"
        )
    if state.agent is not None:
        name = agent_label(state.agent)
        if port is not None:
            lines.append(f"  c      Connect {name} to the server on :{port}")
        else:
            lines.append(f"  c      Connect {name} (detected): start a server for it")
    lines.append("  s      Start the server for your own tools (OpenAI/Anthropic API)")
    lines.append("  m      Choose another model (smart / fast / downloaded)")
    lines.append("  q      Quit")
    lines += [
        f"All commands: rapid-mlx --help · Docs: {DOCS_URL}",
        f"Next: rapid-mlx serve {best['alias']}",
    ]
    return "\n".join(lines)


def picker_entries(
    state: FrontDoorState, *, coding: bool = False
) -> list[tuple[str, str]]:
    """Recommend catalog chat models whose weights leave 35% memory headroom."""
    from rapid_mlx.model_aliases import draft_only_conflict, list_profiles
    from rapid_mlx.model_sizes import size_bytes

    candidates = {}
    tool_capable = set()
    for alias, profile in list_profiles().items():
        name = alias.lower()
        if (
            profile.modality != "text"
            or profile.experimental
            or draft_only_conflict(alias)
            or re.search(
                r"(?:^|[-_.])(mtp|draft|assistant|safeguard|embed(?:ding)?|rerank|image|audio|speech|tts|asr)(?:[-_.]|$)",
                name,
            )
        ):
            continue
        size = size_bytes(profile.hf_path)
        if not size or state.ram_gb <= 0 or size > state.ram_gb * 0.65 * (1 << 30):
            continue
        if profile.tool_call_parser and profile.agentic_coverage:
            tool_capable.add(alias)
        elif coding:
            continue
        candidates[alias] = size

    def low_quality(alias):
        return bool(
            re.search(
                r"(?:^|[-_.])(?:[123]bit|ternary|[12](?:\.\d+)?bpw|3(?:\.0+)?bpw)(?:[-_.]|$)",
                alias,
            )
        )

    good = {alias: size for alias, size in candidates.items() if not low_quality(alias)}
    candidates = good or candidates
    tier = max((tier for tier in _MEMORY_LADDER if tier <= state.ram_gb), default=None)
    ladder = _MEMORY_LADDER[tier] if tier is not None else ()
    preferred = list(reversed(ladder)) if coding else list(ladder)
    ranked: list[str] = []
    cached = set(state.cached)
    cached.update(p["alias"] for p in state.picks if p.get("cached"))

    def add(alias):
        if alias in candidates and alias not in ranked and len(ranked) < _PICKER_LIMIT:
            ranked.append(alias)

    if state.last_used in cached:
        add(state.last_used)
    for i, alias in enumerate(preferred):
        if i == 0 or alias in tool_capable:
            add(alias)
    fallback = (
        QUICK_START_MODEL
        if QUICK_START_MODEL in candidates and QUICK_START_MODEL in tool_capable
        else None
    )
    for alias in _MEASURED_PICKER_MODELS:
        if (
            alias in tool_capable
            and alias != fallback
            and len(ranked) < _PICKER_LIMIT - bool(fallback and fallback not in ranked)
        ):
            add(alias)
    if fallback and fallback in tool_capable:
        add(fallback)
    if not good and not ranked:
        for alias in sorted(candidates):
            add(alias)
    entries = []
    for i, alias in enumerate(ranked):
        if alias == state.last_used and alias in cached:
            label = "Last used"
        elif i == 0:
            label = "Best for coding on this Mac" if coding else "Best for this Mac"
        elif alias == fallback:
            label = "Fast, lighter"
        else:
            label = (
                "Coding model"
                if re.search(r"coder|coding|code", alias)
                else "General model"
            )
        facts = f"{label} · {candidates[alias] / (1 << 30):.1f} GB"
        facts += " · downloaded" if alias in cached else " · (not downloaded)"
        entries.append((alias, facts))
    return entries


def render_picker(entries: Sequence[tuple[str, str]]) -> str:
    width = max((len(alias) for alias, _ in entries), default=0) + 3
    lines = ["Choose a model (press its number · Esc to go back)"]
    for index, (alias, facts) in enumerate(entries, start=1):
        lines.append(f"  {index}  {alias:<{width}}{facts}")
    lines.append("  More models: rapid-mlx models")
    return "\n".join(lines)


def render_non_tty(
    version: str, ram_gb: float, picks: Sequence[dict], quick_size: str = "~3 GB"
) -> str:
    """Deterministic text for a bare invocation without a terminal."""
    quick = QUICK_START_MODEL
    best = best_pick(picks) if picks else {"alias": quick}
    ram = f" ({_gb(round(ram_gb, 1))} GB)" if ram_gb > 0 else ""
    recipe = " · ".join(f"{p['alias']} ({p.get('role', '')})" for p in picks)
    return "\n".join(
        [
            f"rapid-mlx {version}: no command given "
            "(the interactive start needs a terminal).",
            f"Quick start: {quick} ({quick_size} download)",
            f"Best for this Mac{ram}: {best['alias']} · recipe picks: {recipe}",
            "Non-interactive use:",
            f"  rapid-mlx pull {quick}",
            f"  rapid-mlx serve {quick} --port {DEFAULT_PORT}",
            f"  rapid-mlx launch claude-code --model {quick}   # or: rapid-mlx agents",
            f"  rapid-mlx chat {quick}",
            "All commands: rapid-mlx --help · "
            "Machine-readable picks: rapid-mlx recipe --json",
        ]
    )


# --------------------------------------------------------------------------
# Actions
# --------------------------------------------------------------------------
@dataclass
class Plan:
    """Commands an action runs in order (each a CLI argv without ``rapid-mlx``)."""

    steps: list[list[str]]
    note: str | None = None
    # Run once the final ``serve`` step answers /health/ready (``c``: the agent
    # is pointed at the server only after it is really up).
    after_ready: list[str] | None = None


def plan_chat(state: FrontDoorState, model: str | None = None) -> Plan:
    if model is not None and model != state.selected:
        state.selected = model
    port = state.attach_port
    argv = ["chat", state.selected]
    if port is not None:
        argv += ["--port", str(port)]
    return Plan([argv])


def plan_serve(state: FrontDoorState, *, port: int) -> Plan:
    argv = ["serve", state.selected]
    if port != DEFAULT_PORT:
        argv += ["--port", str(port)]
    return Plan(
        [argv],
        note=f"OpenAI/Anthropic base URL: http://127.0.0.1:{port}/v1",
    )


def plan_connect(state: FrontDoorState, *, port: int) -> Plan:
    assert state.agent is not None
    attach = state.attach_port
    use_port = attach if attach is not None else port
    launch = [
        "launch",
        state.agent,
        "--model",
        state.selected,
        "--server-url",
        f"http://127.0.0.1:{use_port}",
    ]
    if attach is not None:
        return Plan([launch])
    # Download first (a no-op when cached), then start the server, and only
    # once it answers /health/ready patch the agent's configuration. A
    # declined download or a server that fails to start never touches it.
    return Plan(
        [
            ["pull", state.selected],
            ["serve", state.selected, "--port", str(use_port)],
        ],
        note=(
            f"{agent_label(state.agent)} is configured once the server is "
            "ready; keep this running and open it in another terminal. "
            "Ctrl-C stops the server."
        ),
        after_ready=launch,
    )


def echo_command(argv: Sequence[str]) -> str:
    return "→ rapid-mlx " + " ".join(argv)


# --------------------------------------------------------------------------
# Terminal I/O
# --------------------------------------------------------------------------
def interactive_terminal() -> bool:
    """Whether the bare command may show the interactive screen."""
    try:
        if not (sys.stdin.isatty() and sys.stdout.isatty()):
            return False
    except (AttributeError, ValueError):
        return False
    if os.environ.get("TERM", "") == "dumb":
        return False
    return not os.environ.get("CI")


def read_key() -> str:
    """Read one keypress without waiting for Enter (cbreak mode)."""
    import select
    import termios
    import tty

    fd = sys.stdin.fileno()
    previous = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        raw = os.read(fd, 1)
        if raw == b"\x1b":
            # A lone Esc vs. the start of an arrow-key escape sequence.
            ready, _, _ = select.select([fd], [], [], 0.05)
            if ready:
                os.read(fd, 16)
                return KEY_UNKNOWN
            return KEY_QUIT
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, previous)
    if raw in (b"", b"\x03", b"\x04"):
        return KEY_QUIT
    if raw in (b"\r", b"\n"):
        return KEY_ENTER
    return raw.decode("utf-8", errors="replace").lower()


def _style(text: str, code: str) -> str:
    if os.environ.get("NO_COLOR"):
        return text
    return f"\x1b[{code}m{text}\x1b[0m"


def _fit_width(line: str, columns: int) -> str:
    return line if len(line) <= columns else line[: max(columns - 1, 1)] + "…"


def _styled_screen(text: str, columns: int | None = None) -> str:
    """Bold header; lines truncated (not wrapped) so the menu keeps its rows."""
    if columns is None:
        import shutil

        columns = shutil.get_terminal_size((80, 24)).columns
    lines = [_fit_width(line, columns) for line in text.split("\n")]
    lines[0] = _style(lines[0], "1")
    return "\n".join(lines)


def choose_model(
    state: FrontDoorState,
    *,
    key_reader: Callable[[], str],
    out: Callable[[str], None],
) -> None:
    """Show the picker and update ``state.selected`` on a numbered choice."""
    entries = picker_entries(state)
    out(render_picker(entries))
    while True:
        key = key_reader()
        if key in (KEY_QUIT, "q", KEY_ENTER):
            return
        if key.isdigit() and 1 <= int(key) <= len(entries):
            state.selected = entries[int(key) - 1][0]
            return


def run_interactive(
    state: FrontDoorState,
    *,
    key_reader: Callable[[], str] | None = None,
    out: Callable[[str], None] = print,
    free_port: Callable[[], int | None] = first_free_port,
) -> Plan | None:
    """Run the launcher; actions delegate to the existing CLI paths."""
    import shutil

    from rapid_mlx import first_run
    from rapid_mlx import front_door_tui as tui

    read = key_reader or tui.read_key
    width, height = shutil.get_terminal_size((80, 24))
    saved = first_run.front_door_preferences()
    tools = [
        ("claude-code", "Launch Claude Code", "Anthropic's coding tool"),
        ("opencode", "Launch OpenCode", "An open source coding agent"),
        ("codex", "Launch Codex", "OpenAI's coding agent"),
        ("hermes", "Launch Hermes Agent", "A local AI assistant"),
    ]
    tools.sort(
        key=lambda item: (
            not bool(shutil.which(item[0] if item[0] != "claude-code" else "claude"))
        )
    )
    items = tools + [
        ("chat", "Chat in this terminal", "Talk to a local model right here"),
        ("more", "More...", "Start an API server, other tools, all commands"),
    ]
    more_items = [
        ("serve", "Start an API server", "Serve a model for your own tools"),
        ("other", "Other coding tools", "Goose, aider, pi, qwen-code, crush"),
        ("chat", "Choose a different model to chat", "Talk to a local model"),
        ("help", "All commands", "Show command help"),
        ("back", "Back", "Return to the launcher"),
    ]
    other_items = [
        (n, f"Launch {label}", "Connect a coding tool")
        for n, label in [
            ("goose", "Goose"),
            ("aider", "aider"),
            ("pi", "pi"),
            ("qwen-code", "qwen-code"),
            ("crush", "crush"),
        ]
    ] + [("back", "Back", "Return to More...")]
    main_items = items
    menu = "main"
    cursor = next(
        (i for i, item in enumerate(items) if item[0] == saved.get("selection")), 0
    )
    models = saved.get("models", {})
    if not isinstance(models, dict):
        models = {}

    def draw(text: str) -> None:
        out(_CLEAR_SCREEN + text)

    try:
        if not saved.get("welcome"):
            draw(_styled_screen(tui.welcome(width), width))
            while True:
                key = read()
                if key == KEY_QUIT:
                    return None
                if key == KEY_ENTER:
                    saved["welcome"] = True
                    first_run.save_front_door_preferences(saved)
                    break
        while True:
            hardware = " · ".join(
                x
                for x in [
                    state.chip or "Apple Silicon",
                    f"{_gb(state.ram_gb)} GB" if state.ram_gb else "",
                ]
                if x
            )
            server = (
                f"Server running on :{state.server.port} with {state.server.model}"
                if state.server
                else ""
            )
            draw(
                tui.launcher(
                    f"Rapid-MLX {state.version}",
                    hardware,
                    items,
                    cursor,
                    server,
                    width,
                    height,
                    True,
                )
            )
            key = read()
            if key in (KEY_QUIT, "left"):
                if menu == "main":
                    return None
                menu, items, cursor = (
                    ("more", more_items, 0)
                    if menu == "other"
                    else ("main", main_items, 0)
                )
                continue
            if key in ("up", "down"):
                cursor = (cursor + (-1 if key == "up" else 1)) % len(items)
                continue
            if key not in (KEY_ENTER, "right"):
                continue
            name, label, _ = items[cursor]
            if name in ("more", "other", "back"):
                if name == "more":
                    menu, items = "more", more_items
                elif name == "other":
                    menu, items = "other", other_items
                else:
                    menu, items = (
                        ("more", more_items)
                        if menu == "other"
                        else ("main", main_items)
                    )
                cursor = 0
                continue
            if name == "help":
                return Plan([["--help"]])
            entries = picker_entries(state, coding=name not in ("chat", "serve"))
            query = ""
            selected = models.get(name) or state.last_used or state.selected
            index = next(
                (i for i, entry in enumerate(entries) if entry[0] == selected), 0
            )
            while True:
                filtered = [
                    entry
                    for entry in entries
                    if query.lower() in (entry[0] + " " + entry[1]).lower()
                ]
                filtered = filtered[:_PICKER_LIMIT]
                index = min(index, max(0, len(filtered) - 1))
                draw(
                    tui.picker(
                        label.removeprefix("Launch "),
                        filtered,
                        index,
                        query,
                        width,
                        height,
                        True,
                    )
                )
                k = read()
                if k in ("left", KEY_QUIT):
                    break
                if k in ("up", "down") and filtered:
                    index = (index + (-1 if k == "up" else 1)) % len(filtered)
                elif k == "backspace":
                    query = query[:-1]
                    index = 0
                elif k == KEY_ENTER and filtered:
                    model = filtered[index][0]
                    models[name] = model
                    saved.update(selection=name, models=models)
                    first_run.save_front_door_preferences(saved)
                    if key == "right":
                        break
                    if (
                        model != state.local_default
                        and model not in state.cached
                        and not any(
                            p.get("alias") == model and p.get("cached")
                            for p in state.picks
                        )
                    ):
                        yes = True
                        while True:
                            draw(_styled_screen(tui.confirm(model, yes, width), width))
                            choice = read()
                            if choice in ("left", "right"):
                                yes = not yes
                            elif choice == KEY_QUIT:
                                yes = False
                                break
                            elif choice == KEY_ENTER:
                                break
                        if not yes:
                            break
                        steps = [["pull", model]]
                    else:
                        steps = []
                    if name == "chat":
                        steps.extend(plan_chat(state, model).steps)
                    elif name == "serve":
                        port = free_port()
                        if port is None:
                            out("No free port available")
                            break
                        state.selected = model
                        plan = plan_serve(state, port=port)
                        return Plan(steps + plan.steps, note=plan.note)
                    else:
                        steps.append(["launch", name, "--model", model])
                    return Plan(steps)
                elif len(k) == 1 and k.isprintable():
                    query += k
                    index = 0
    except KeyboardInterrupt:
        return None


def _cli_command() -> list[str]:
    """This install's own ``rapid-mlx`` entry point (never a PATH lookup)."""
    script = os.path.join(os.path.dirname(sys.executable), "rapid-mlx")
    if os.access(script, os.X_OK):
        return [script]
    return [sys.executable, "-m", "rapid_mlx.cli"]


def wait_until_ready(
    port: int,
    proc,
    *,
    timeout_s: float = 900.0,
    probe: Callable[[str], bool] | None = None,
    sleep: Callable[[float], None] | None = None,
) -> bool:
    """Poll ``/health/ready`` until 200 (True) or the server exits/times out."""
    import time

    check = probe or _ready_probe
    pause = sleep or time.sleep
    url = f"http://127.0.0.1:{port}/health/ready"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return False
        if check(url):
            return True
        pause(1.0)
    return False


def _ready_probe(url: str) -> bool:
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(url, timeout=2) as response:  # noqa: S310
            return bool(response.status == 200)
    except (urllib.error.URLError, OSError, ValueError):
        return False


def run_server_process(
    argv: Sequence[str],
    *,
    port: int | None = None,
    on_ready: Callable[[], None] | None = None,
    popen: Callable[..., Any] = subprocess.Popen,
) -> int:
    """Run ``serve`` as a fresh foreground child and return its exit status.

    A server is long-lived and owns its own telemetry lifecycle (surface
    ``server``), so it must not share this short-lived CLI process. The child
    keeps the terminal and its default signal dispositions, so Ctrl-C reaches
    it directly; the parent swallows its own copy while it waits (a handler,
    not ``SIG_IGN``, because an ignored disposition would be inherited).
    ``on_ready`` runs once the server answers ``/health/ready``.
    """
    import signal

    previous = signal.signal(signal.SIGINT, lambda *_args: None)
    try:
        proc = popen([*_cli_command(), *argv])
        if on_ready is not None and port is not None:
            if wait_until_ready(port, proc):
                on_ready()
        return int(proc.wait())
    finally:
        signal.signal(signal.SIGINT, previous)


def run_bare(
    *,
    version: str,
    top_level_flags: Sequence[str],
    dispatch: Callable[[list[str]], object],
    serve: Callable[..., int] = run_server_process,
) -> int:
    """Entry point for bare ``rapid-mlx``. Returns the process exit status.

    ``dispatch`` runs one CLI argv in-process (the normal ``main()`` path).
    ``top_level_flags`` (e.g. ``--no-telemetry``) are forwarded to it; the
    launch banner is suppressed because the front door already printed the
    header.
    """
    try:
        if not interactive_terminal():
            from rapid_mlx.recommendations import physical_ram_gb

            ram_gb = physical_ram_gb()
            text = render_non_tty(
                version, ram_gb, _picks_for(ram_gb, []), quick_start_size_label()
            )
            print(text, file=sys.stderr)
            return 1
        state = gather_state(version)
        render_screen(state)  # surface a rendering failure before any output
    except Exception as exc:
        raise FrontDoorUnavailableError(str(exc)) from exc

    plan = run_interactive(state)
    if plan is None:
        return 0
    if plan.steps and plan.steps[0][0] == "pull" and plan.steps[-1][0] == "launch":
        from rapid_mlx.agents import get_profile
        from rapid_mlx.agents.cli_preflight import missing_agent_cli

        profile = get_profile(plan.steps[-1][1])
        message = missing_agent_cli(profile) if profile else None
        if message:
            print(message, file=sys.stderr)
            return 1
    for index, argv in enumerate(plan.steps):
        print(echo_command(argv))
        if plan.note and index == len(plan.steps) - 1:
            print(f"  {plan.note}")
        full = ["--no-banner", *top_level_flags, *argv]
        if argv[0] == "serve":
            if plan.after_ready is None:
                return serve(full)
            follow = ["--no-banner", *top_level_flags, *plan.after_ready]
            return serve(
                full,
                port=int(argv[argv.index("--port") + 1]),
                on_ready=functools.partial(
                    _run_after_ready, plan.after_ready, follow, dispatch
                ),
            )
        if argv[0] == "pull":
            previous = os.environ.get("RAPID_MLX_FRONT_DOOR_PULL")
            os.environ["RAPID_MLX_FRONT_DOOR_PULL"] = "1"
            try:
                dispatch(full)
            finally:
                if previous is None:
                    os.environ.pop("RAPID_MLX_FRONT_DOOR_PULL", None)
                else:
                    os.environ["RAPID_MLX_FRONT_DOOR_PULL"] = previous
        else:
            dispatch(full)
    return 0


def _run_after_ready(
    shown: Sequence[str], argv: list[str], dispatch: Callable[[list[str]], object]
) -> None:
    """Run the post-ready step; a failure there must not stop the server."""
    print(echo_command(shown))
    try:
        dispatch(argv)
    except SystemExit as exc:
        if exc.code not in (None, 0):
            print(f"  That step failed (exit {exc.code}); the server keeps running.")
    except Exception as exc:  # the foreground server must outlive a bad launch
        print(f"  That step failed ({exc}); the server keeps running.")
