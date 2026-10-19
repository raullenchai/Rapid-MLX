"""First-run guide helpers — maximize the install → first-token conversion.

Three CLI surfaces share this module (wired in ``rapid_mlx/cli.py``):

  * ``rapid-mlx`` (bare command) → the interactive front door in
    ``rapid_mlx/front_door.py``, which uses the probes here (cached aliases,
    detected agents, last-used model).
  * ``rapid-mlx chat`` with no model → prefer a runnable local chat model, so
    the user's only decision is typing ``chat`` (everything else is a default
    + override).
  * The first successful ``chat`` exit → a one-time "connect your agent" tip.

Design constraints (mirror ``rapid_mlx/telemetry/consent.py``):

  * **TTY-guarded.** The front door and notices only render for an interactive
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
_AGENT_PREFERENCE = ("claude-code", "cline", "continue-dev")


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
def cached_known_aliases(
    cache_rows: list[tuple[str, int, float]] | None = None,
) -> list[tuple[str, float]]:
    """``[(alias, mtime_epoch), ...]`` for cached models that map to a known
    alias, most-recently-modified first.

    Unmapped cache entries (a raw HF repo with no alias in ``aliases.json``)
    are omitted; shared local discovery handles those separately. Returns ``[]``
    on a cold cache or scan error; ``cache_rows`` reuses a scan the caller holds.
    """
    try:
        # Lazy import: cli.py imports this module, so importing it back at
        # module-load time would cycle. By call time cli is fully loaded.
        from rapid_mlx.cli import _cache_entry_is_runnable, _scan_hf_cache_models
        from rapid_mlx.model_aliases import list_profiles

        hf_to_alias: dict[str, str] = {}
        for alias, profile in list_profiles().items():
            hf_to_alias.setdefault(profile.hf_path, alias)

        rows: list[tuple[str, float]] = []
        if cache_rows is None:
            cache_rows = _scan_hf_cache_models()
        for repo, _size, mtime in cache_rows:
            alias = hf_to_alias.get(repo)
            if alias is not None and _cache_entry_is_runnable(repo):
                rows.append((alias, mtime))
        rows.sort(key=lambda r: -r[1])
        return rows
    except Exception:
        return []


def select_chat_default() -> tuple[str, bool]:
    """Prefer the cached starter, then a complete, policy-ranked local chat model.

    Return ``(model, already_cached)``; a cold cache retains the starter download.
    Discovery only admits chat-capable models and never accesses the network.
    """
    from rapid_mlx.local_models import catalog_model_ready, select_default_local_model

    cached_aliases = {alias for alias, _ in cached_known_aliases()}
    if FIRST_RUN_MODEL in cached_aliases and catalog_model_ready(FIRST_RUN_MODEL):
        return FIRST_RUN_MODEL, True

    chosen = select_default_local_model()
    return (chosen.model, True) if chosen is not None else (FIRST_RUN_MODEL, False)


# --------------------------------------------------------------------------
# Agent detection (front door + P0-3 tip)
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
        from rapid_mlx.launch import ADAPTERS

        # Cursor cannot use the local default endpoint; it only works when the
        # user explicitly provides a public HTTPS tunnel. Do not recommend a
        # one-shot local launch command that will necessarily be rejected.
        found = {
            name for name, a in ADAPTERS.items() if name != "cursor" and _safe_detect(a)
        }
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
# Last-used chat model (bare-command front door, returning users)
# --------------------------------------------------------------------------
def _last_model_file() -> Path:
    return _state_dir() / "last_chat_model"


def _is_chat_alias(name: str) -> bool:
    """Whether ``name`` is a registered text-chat alias (fail-safe False)."""
    try:
        from rapid_mlx.model_aliases import list_profiles

        profile = list_profiles().get(name)
        return profile is not None and profile.modality == "text"
    except Exception:
        return False


def record_last_model(name: str | None) -> None:
    """Remember the alias of the model a local ``chat`` session started.

    Only registered text aliases are stored (never a path or raw repo id), so
    the front door can offer "continue with <alias>" without guessing whether
    a cached checkpoint can chat. Fail-silent: an unwritable state dir only
    loses the convenience.
    """
    if not name or not _is_chat_alias(name):
        return
    try:
        target = _last_model_file()
        target.parent.mkdir(parents=True, exist_ok=True)
        # Write-then-rename so a concurrent reader never sees a partial file.
        tmp = target.with_name(f"{target.name}.{os.getpid()}.tmp")
        tmp.write_text(name + "\n", encoding="utf-8")
        os.replace(tmp, target)
    except Exception:
        return


def last_used_model() -> str | None:
    """The alias stored by :func:`record_last_model`, if still a chat alias."""
    try:
        name = _last_model_file().read_text(encoding="utf-8").strip()
    except Exception:
        return None
    return name if _is_chat_alias(name) else None


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


def chat_agent_tip_text() -> str:
    """The one-line tip shown after a user's first successful chat.

    It names the detected agent (claude-code preferred) but directs users to
    the stable-server setup help instead of implying that ``launch`` can attach
    to the ephemeral chat server, which has already stopped by this point.
    """
    agent = preferred_agent()
    if agent is not None:
        return (
            f"Tip: set up a separate stable server for {agent} → "
            "rapid-mlx launch --help"
        )
    return "Tip: set up a separate stable agent server → rapid-mlx launch --help"
