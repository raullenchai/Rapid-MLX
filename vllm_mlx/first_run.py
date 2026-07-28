"""First-run guide helpers — maximize the install → first-token conversion.

Three CLI surfaces share this module (wired in ``vllm_mlx/cli.py``):

  * ``rapid-mlx`` (bare command) → a *nameplate*: hardware line + cached-model
    hint + a "get started" signpost. Non-blocking, prints and exits.
  * ``rapid-mlx chat`` with no model → auto-select the known-good starter, so
    the user's only decision is typing ``chat`` (everything else is a default
    + override).
  * The first successful ``chat`` exit → a one-time "connect your agent" tip.

Design constraints (mirror ``vllm_mlx/telemetry/consent.py``):

  * **TTY-guarded.** The nameplate and notices only render for an interactive
    session; non-TTY (CI, pipes) keeps today's behavior with zero extra output.
    The callers own the TTY check; the pure helpers here never print.
  * **Fail-silent.** Every probe (cache scan, hardware detect, agent detect,
    state file) is wrapped so a failure degrades gracefully and NEVER crashes
    the CLI — a broken HF cache dir must not stop ``rapid-mlx chat``.
"""

from __future__ import annotations

import os
from pathlib import Path

# The starter model for a cold-cache first run. Deliberately NOT RAM-tiered:
# even a 96 GB Mac gets the small 4B first so a brand-new user sees a token in
# ~1-2 minutes instead of waiting on a 60 GB download. Users graduate to the
# bigger models (``rapid-mlx models``) on their own once they're comfortable.
# Must stay a dogfood-tested, tool-calling-reliable alias (0.11.0 headline);
# an experimental low-bit model would damage the first impression.
#
# PARITY: this is the same model install.sh recommends for its lowest RAM tier
# (the 8-23 GB branch, ``RECOMMENDED_MODEL="qwen3.5-4b-4bit"``). If you change
# one, change the other so the installer and the CLI tell one story.
FIRST_RUN_MODEL = "qwen3.5-4b-4bit"

# Human-facing one-time download size for FIRST_RUN_MODEL. A fixed string (not
# a network probe) so the "no model specified" notice is instant. Anchored to
# the actual snapshot size of ``mlx-community/Qwen3.5-4B-MLX-4bit`` — 3.061 GB
# decimal (2.85 GiB), i.e. the ``3061132920`` bytes the download bar counts up
# to. The old "~2.5 GB" undershot the real download by ~0.5 GB (0.11 dogfood).
# If the starter checkpoint is re-quantized, re-probe and update this.
FIRST_RUN_MODEL_SIZE = "~3.1 GB"

# Preference order when several coding agents are detected: claude-code is the
# ICP, so it leads. Others follow in a stable order.
_AGENT_PREFERENCE = ("claude-code", "cursor", "cline", "continue-dev")

_DOCS_URL = "https://rapidmlx.com/docs/"


def _state_dir() -> Path:
    """The ``~/.rapid-mlx/`` state dir (same one telemetry uses; see
    ``telemetry/state.py::_default_telemetry_dir``). Kept independent of the
    telemetry opt-in so the one-time chat tip fires even when telemetry is
    disabled.

    ``RAPID_MLX_STATE_DIR`` overrides the location — used by tests to isolate
    the one-time marker from the real home dir, and available as an escape
    hatch for sandboxed / read-only-home environments.
    """
    override = os.environ.get("RAPID_MLX_STATE_DIR")
    if override:
        return Path(override)
    return Path.home() / ".rapid-mlx"


# --------------------------------------------------------------------------
# Starter model selection (P0-1) + cached-model display (P0-2)
# --------------------------------------------------------------------------
def cached_known_aliases() -> list[tuple[str, float]]:
    """``[(alias, mtime_epoch), ...]`` for cached models that map to a known
    alias, most-recently-modified first.

    Unmapped cache entries (a raw HF repo with no alias in ``aliases.json``)
    are omitted — an unknown profile is unsafe to auto-select as the chat
    default. Returns ``[]`` on a cold cache or any scan error.
    """
    try:
        # Lazy import: cli.py imports this module, so importing it back at
        # module-load time would cycle. By call time cli is fully loaded.
        from vllm_mlx.cli import _scan_hf_cache_models
        from vllm_mlx.model_aliases import list_profiles

        hf_to_alias: dict[str, str] = {}
        for alias, profile in list_profiles().items():
            hf_to_alias.setdefault(profile.hf_path, alias)

        rows: list[tuple[str, float]] = []
        for repo, _size, mtime in _scan_hf_cache_models():
            alias = hf_to_alias.get(repo)
            if alias is not None:
                rows.append((alias, mtime))
        rows.sort(key=lambda r: -r[1])
        return rows
    except Exception:
        return []


def select_chat_default() -> tuple[str, bool]:
    """Pick the model for ``rapid-mlx chat`` / ``run`` when the user gave no
    model. Always the first-run starter (:data:`FIRST_RUN_MODEL`) — a known
    chat/tool-call-capable, dogfood-tested alias.

    Returns ``(alias, already_cached)``, where ``already_cached`` reports only
    whether the starter itself is downloaded (it drives the "already
    downloaded" vs "one-time download" notice — nothing else).

    We deliberately do NOT auto-pick the most-recently-cached alias: it could
    be a non-chat checkpoint (embedding / transcription), and silently
    selecting a model the user never named invites surprise and a hard-to-read
    failure. The bare-command nameplate still LISTS cached models so a
    returning user can pick one explicitly.
    """
    cached_aliases = {alias for alias, _ in cached_known_aliases()}
    return FIRST_RUN_MODEL, FIRST_RUN_MODEL in cached_aliases


# --------------------------------------------------------------------------
# Agent detection (P0-2 nameplate + P0-3 tip)
# --------------------------------------------------------------------------
def _safe_detect(adapter: object) -> bool:
    try:
        return bool(adapter.detect())  # type: ignore[attr-defined]
    except Exception:
        return False


def detected_agents() -> list[str]:
    """Names of coding agents detected on this machine, via the same adapters
    ``rapid-mlx launch`` uses. Ordered by :data:`_AGENT_PREFERENCE` (claude-code
    first). Returns ``[]`` on error."""
    try:
        from vllm_mlx.launch import ADAPTERS

        found = {name for name, a in ADAPTERS.items() if _safe_detect(a)}
    except Exception:
        return []
    ordered = [n for n in _AGENT_PREFERENCE if n in found]
    # Any adapter not in the preference list still gets surfaced, appended
    # in a stable (sorted) order so a new adapter isn't silently dropped.
    ordered += sorted(found - set(ordered))
    return ordered


def preferred_agent() -> str | None:
    """The single agent to name in a signpost, or ``None`` if none detected."""
    agents = detected_agents()
    return agents[0] if agents else None


# --------------------------------------------------------------------------
# Nameplate (P0-2)
# --------------------------------------------------------------------------
def _hardware_line(version: str) -> str:
    try:
        from vllm_mlx.optimizations import detect_hardware

        hw = detect_hardware()
        chip = hw.chip_name if hw.chip_name and hw.chip_name != "Unknown" else None
        mem = f"{hw.total_memory_gb:.0f}GB" if hw.total_memory_gb else None
        if chip and mem:
            return f"Rapid-MLX {version} · {chip} / {mem} detected"
        if mem:
            return f"Rapid-MLX {version} · {mem} detected"
    except Exception:
        pass
    return f"Rapid-MLX {version}"


def _render_get_started(rows: list[tuple[str, str]]) -> list[str]:
    """Render ``(command, comment)`` pairs with the ``#`` comments aligned to a
    common column."""
    width = max((len(cmd) for cmd, _ in rows), default=0)
    out = []
    for cmd, comment in rows:
        out.append(f"  {cmd.ljust(width)}   # {comment}" if comment else f"  {cmd}")
    return out


def build_nameplate(version: str) -> str:
    """Build the bare-command nameplate text (no ANSI, no trailing newline).

    Pure/deterministic given the machine state — the caller decides whether to
    print it (TTY only). Every probe inside is fail-silent, so a broken cache
    or hardware-detect surface still yields a usable signpost.
    """
    lines: list[str] = [_hardware_line(version), ""]

    cached = cached_known_aliases()
    if cached:
        shown = ", ".join(alias for alias, _ in cached[:3])
        lines.append(f"Found in your cache: {shown}")
        lines.append("")

    lines.append("Get started:")

    # The bare-command chat suggestion always points at the known-good starter
    # (FIRST_RUN_MODEL), never whichever alias happens to be cached: the
    # starter is the one model we promise reliable chat + tool-calls, and
    # auto-picking a cached alias would run something the user never named
    # (e.g. a text-diffusion checkpoint, which chats but with very different
    # REPL behavior). Cached models are still listed above so a returning user
    # can name one explicitly.
    starter_cached = any(alias == FIRST_RUN_MODEL for alias, _ in cached)
    rows: list[tuple[str, str]] = []
    if starter_cached:
        rows.append(("rapid-mlx chat", f"{FIRST_RUN_MODEL} — already downloaded"))
    else:
        rows.append(
            (
                "rapid-mlx chat",
                f"{FIRST_RUN_MODEL} ({FIRST_RUN_MODEL_SIZE} one-time download)",
            )
        )
    if cached:
        rows.append(("rapid-mlx chat <model>", "or name a cached model above"))
    rows.append(
        (f"rapid-mlx serve {FIRST_RUN_MODEL}", "OpenAI-compatible API on :8000")
    )

    agent = preferred_agent()
    if agent is not None:
        rows.append((f"rapid-mlx launch {agent}", "connect your agent (detected ✓)"))
    else:
        rows.append(("rapid-mlx launch --all", "detect & connect coding agents"))

    lines += _render_get_started(rows)
    lines.append("")
    lines.append(f"Docs: {_DOCS_URL}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# One-time "connect your agent" tip after the first successful chat (P0-3)
# --------------------------------------------------------------------------
def _chat_tip_marker() -> Path:
    return _state_dir() / "chat_agent_tip_shown"


def claim_chat_agent_tip() -> bool:
    """Atomically claim the one-time chat→agent tip.

    Returns ``True`` for exactly ONE caller per machine: the process that
    creates the marker via exclusive ``O_EXCL`` creation wins and shows the
    tip. Concurrent first-run sessions that lose the race (marker already
    exists) get ``False``. Any other error (read-only / unwritable state
    dir) is also ``False`` — fail-safe toward *never nagging* and never
    crashing.

    Doing the check-and-claim in one atomic OS call closes the
    check→print→mark TOCTOU where two concurrent first sessions could each
    see "not shown" and both print the supposedly once-per-machine tip.
    """
    try:
        marker = _chat_tip_marker()
        marker.parent.mkdir(parents=True, exist_ok=True)
        # Mode "x" is exclusive creation (O_CREAT | O_EXCL): the OS guarantees
        # only one concurrent creator succeeds. The context manager closes the
        # descriptor on every path, so no fd leaks in this long-lived process.
        with open(marker, "x"):
            pass
        return True
    except FileExistsError:
        return False
    except Exception:
        return False


def _session_seen_marker() -> Path:
    return _state_dir() / "session_seen"


def mark_first_session() -> bool:
    """Atomically record that this client has reached a recorded session.

    Returns ``True`` for exactly ONE call per machine -- the first process to
    create the ``session_seen`` marker via exclusive ``O_EXCL`` creation.
    Every later call (and any concurrent racer that loses) gets ``False``.
    This is the client-side "first session" signal for the #1272 activation
    funnel: the client tracks its own first participating session more
    reliably than the server can infer it from a bounded telemetry-retention
    window.

    Semantics note (codex #1273): this marks the first session that is
    actually RECORDED, not necessarily the first-ever binary invocation. The
    caller (``cli.py``) deliberately does NOT reach this on the run that just
    collected first-run consent -- the disclosure promises "nothing from
    before this prompt", so that run emits nothing and is not marked. The
    next (first participating) session therefore carries ``first_session=
    True``, exactly once per client. This is the right funnel semantic:
    "the first session we recorded from this new client." If a client is
    already opted in before its first run (env / prior consent), that first
    run is itself the first recorded session and is marked here.

    The marker is a local empty file; only the derived boolean is ever sent,
    and only when telemetry is enabled.

    Fail-safe toward ``False`` on any error (unwritable state dir, etc.):
    under-reporting a first run is conservative -- it never inflates the
    funnel's new-client count -- and never crashes the session.
    """
    try:
        marker = _session_seen_marker()
        marker.parent.mkdir(parents=True, exist_ok=True)
        with open(marker, "x"):
            pass
        return True
    except FileExistsError:
        return False
    except Exception:
        return False


def chat_agent_tip_text() -> str:
    """The one-line tip shown after a user's first successful chat. Names the
    detected agent (claude-code preferred); falls back to the generic
    ``launch --all`` when none is detected."""
    agent = preferred_agent()
    if agent is not None:
        return f"Tip: connect {agent} to this engine → rapid-mlx launch {agent}"
    return "Tip: connect your coding agent → rapid-mlx launch --all"
