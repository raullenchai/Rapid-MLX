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
_PICKER_LIMIT = 7
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
    first) → the quick-start model."""
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
    return QUICK_START_MODEL


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
    lines += ["", f"All commands: rapid-mlx --help · Docs: {DOCS_URL}"]
    return "\n".join(lines)


def picker_entries(state: FrontDoorState) -> list[tuple[str, str]]:
    """``[(alias, facts), ...]``: the policy picks, then downloaded models."""
    from rapid_mlx.run.cli import _fits_host

    def fit_mark(alias: str) -> str:
        if state.ram_gb <= 0:
            return ""
        try:
            fits = _fits_host(alias, state.ram_gb)
        except Exception:
            return ""
        if fits is None:
            return ""
        return "fits this Mac" if fits else "too big for this Mac"

    entries: list[tuple[str, str]] = []
    for pick in state.picks:
        facts = [_pick_facts(pick)]
        mark = fit_mark(pick["alias"])
        if mark:
            facts.append(mark)
        entries.append((pick["alias"], " · ".join(facts)))
    seen = {alias for alias, _ in entries}
    if QUICK_START_MODEL not in seen:
        seen.add(QUICK_START_MODEL)
        facts = [_quick_start_facts(state)]
        mark = fit_mark(QUICK_START_MODEL)
        if mark:
            facts.append(mark)
        entries.append((QUICK_START_MODEL, " · ".join(facts)))
    for alias in state.cached:
        if alias in seen or len(entries) >= _PICKER_LIMIT:
            continue
        seen.add(alias)
        facts = ["downloaded"]
        mark = fit_mark(alias)
        if mark:
            facts.append(mark)
        entries.append((alias, " · ".join(facts)))
    return entries


def render_picker(entries: Sequence[tuple[str, str]]) -> str:
    width = max(len(alias) for alias, _ in entries) + 3
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
    best = best_pick(picks)
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
    key_reader: Callable[[], str] = read_key,
    out: Callable[[str], None] = print,
    free_port: Callable[[], int | None] = first_free_port,
) -> Plan | None:
    """Show the screen and wait for an action key. ``None`` means quit."""
    out(_styled_screen(render_screen(state)))
    while True:
        try:
            key = key_reader()
        except KeyboardInterrupt:
            return None
        if key in (KEY_QUIT, "q"):
            return None
        if key == KEY_ENTER:
            return plan_chat(state)
        if key == "b":
            return plan_chat(state, best_pick(state.picks)["alias"])
        if key == "s" or (key == "c" and state.agent is not None):
            port = state.attach_port if key == "c" else None
            if port is None:
                port = free_port()
            if port is None:
                out(
                    f"No free port in {DEFAULT_PORT}-"
                    f"{DEFAULT_PORT + _PORT_SEARCH_SPAN - 1}; stop a server "
                    "(rapid-mlx ps) and try again."
                )
                continue
            if key == "s":
                return plan_serve(state, port=port)
            return plan_connect(state, port=port)
        if key == "m":
            try:
                choose_model(state, key_reader=key_reader, out=out)
            except KeyboardInterrupt:
                return None
            out(_CLEAR_SCREEN + _styled_screen(render_screen(state)))


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
