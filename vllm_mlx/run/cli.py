# SPDX-License-Identifier: Apache-2.0
"""``rapid-mlx start <agent>`` — one-command agent startup (#150).

``start`` ties the pieces that already exist into a single foreground
verb: resolve an agent profile, pick a compatible model, start the
canonical ``serve`` path as a parent-owned child, wait until the endpoint
is healthy, then print (and optionally apply) the agent's setup config.

Design contract (see ``vllm_mlx/run/__init__.py`` for the full rationale):

* **No second config writer, model selector, or serve entrypoint.** This
  module orchestrates ``vllm_mlx/agents`` (profiles + setup plans),
  ``vllm_mlx/recommendations`` (memory-fit selection),
  ``vllm_mlx/model_aliases`` (alias resolution), ``vllm_mlx/_download_gate``
  (download consent), and the canonical ``serve`` subprocess.
* **Foreground child, parent-owned.** ``start`` keeps the child attached to
  the terminal for stdio but isolates its signal group. The parent is the
  sole SIGINT/SIGTERM relay, exits with the child's status, and prevents a
  terminal Ctrl-C from double-signalling the server. No orphan is left
  (modulo SIGKILL, where the child-side ``RAPID_MLX_WATCHDOG_PPID`` watchdog
  self-terminates).
* **Download consent happens in the parent once.** The serve child is
  spawned with ``RAPID_MLX_CHAT_SPAWN=1`` so its own B2 auto-pull gate
  never re-prompts (identical to the chat REPL spawner).

The verb is ``start`` (not the issue's literal ``run``) because ``run`` is
already a shipped alias for ``chat`` (Ollama muscle-memory); see the plan.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import-time only
    from vllm_mlx.agents.base import AgentProfile


def register(subparsers) -> None:
    """Register the ``start`` subparser (deferred-import from ``cli.py``)."""
    from vllm_mlx._completion import alias_completer  # noqa: PLC0415
    from vllm_mlx.cli import _port_arg  # noqa: PLC0415

    parser = subparsers.add_parser(
        "start",
        help="Start an AI agent with a local model in one command",
        description=(
            "Start an AI agent with a local model in one command.\n\n"
            "Resolves the agent profile, picks a recommended model that "
            "fits this Mac and is already cached (or previews a download), "
            "starts the server in the foreground, and prints the agent's "
            "connection instructions (optionally configuring it)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "profile",
        nargs="?",
        default=None,
        help=(
            "Agent name (e.g. codex, hermes, opencode). Omit for a generic "
            "OpenAI-compatible endpoint."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Model alias or HF repo to serve (default: first recommended "
            "model in the profile that fits this Mac and is already cached; "
            "else a previewed download)."
        ),
    ).completer = alias_completer
    parser.add_argument(
        "--port",
        type=_port_arg,
        default=8000,
        help="Port to serve on (default: 8000)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to serve on (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Refuse to download a model; fail if no recommended model is cached",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the model, port, and config mutations without starting "
        "a server, downloading anything, or writing configuration",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip the confirmation prompt before downloading / writing config",
    )
    parser.add_argument(
        "--no-setup",
        action="store_true",
        help="Do not write agent configuration after the server is ready; "
        "only print instructions",
    )
    parser.add_argument(
        "--ready-timeout",
        type=int,
        default=600,
        help="Seconds to wait for the spawned server to become ready (default: 600)",
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def start_command(args) -> int:
    """Resolve, consent, spawn (or reuse), and wait. Returns an exit code."""
    from vllm_mlx.agents import get_profile

    profile_name = args.profile
    profile = get_profile(profile_name) if profile_name else None
    if profile_name and profile is None:
        _print_unknown_agent(profile_name)
        return 1

    resolved_model = _select_model(
        explicit=args.model,
        profile=profile,
        no_download=args.no_download,
    )
    if resolved_model is None:
        return 1
    # main() expands explicit aliases before dispatch. Downloads and cache
    # probes use that resolved repo, while serve/API/config keep the spelling
    # the user selected, matching a direct ``rapid-mlx serve <alias>`` call.
    served_model = getattr(args, "_original_alias", None) or resolved_model

    # Render the authority through the endpoint SSOT so IPv6 literals are
    # bracketed correctly.  Keep the server root separate from the OpenAI
    # API base: readiness lives at ``/health/ready``, while model discovery
    # and agent configs consume ``/v1``.
    from vllm_mlx.connect import ServerEndpoints

    connect_host = _local_connect_host(args.host)
    base_url = ServerEndpoints(connect_host, args.port, model=served_model).base_url

    # Port already occupied: reuse a compatible healthy server, else refuse.
    if _port_is_busy(args.host, args.port):
        return _reuse_or_refuse(base_url, resolved_model, served_model, profile, args)

    if args.dry_run:
        _print_dry_run(profile_name, served_model, args)
        if profile is not None and not args.no_setup:
            return _attach_and_configure(base_url, served_model, profile, args)
        return 0

    if not _confirm_download(
        resolved_model, no_download=args.no_download, yes=args.yes
    ):
        return 1

    try:
        proc = _spawn_foreground_serve(served_model, args)
    except OSError as exc:
        print(f"  Could not start the server process: {exc}")
        return 1
    try:
        with _foreground_child(proc):
            outcome = _wait_ready(base_url, proc, args.ready_timeout)
            if outcome == "timeout":
                # The child is STILL alive (loading/downloading) but never became
                # ready within --ready-timeout. Waiting longer would hang
                # indefinitely, so terminate it, reap, and fail (HIGH squash: the
                # earlier "Server did not become ready" is the final word).
                _terminate_child(proc)
                return 124
            if outcome == "interrupted":
                _terminate_child(proc)
                return 128 + signal.SIGINT
            if outcome == "exited":
                # The serve child died before /health/ready returned; reap it and
                # surface its (nonzero) status instead of a raw traceback.
                return _wait_child(proc)
            setup_rc = _attach_and_configure(base_url, served_model, profile, args)
            if setup_rc:
                _terminate_child(proc)
                return setup_rc
            return _wait_child(proc)
    except _ForwardedSignalError as exc:
        return 128 + exc.signum


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------


def _select_model(
    *, explicit: str | None, profile: AgentProfile | None, no_download: bool
) -> str | None:
    """Pick the model to serve, applying the documented selection order.

    1. ``explicit`` (from ``--model``) wins, served verbatim. It already
       flowed through ``main()``'s top-level alias->path resolution, so it
       is a concrete alias or HF repo id here.
    2. Otherwise iterate the profile's recommended models; pick the first
       that fits this host's RAM (``is_recommended_alias``) and is already
       cached (``is_repo_cached``).
    3. If none is cached and downloads are allowed, pick the first fitting
       recommended model (the profile's declaration order encodes the
       preferred workhorse first) and let the download-consent step run.
    4. With ``--no-download``, refuse with a clear message rather than
       touching the network.
    """
    from vllm_mlx._download_gate import is_repo_cached
    from vllm_mlx.recommendations import physical_ram_gb

    if explicit:
        return explicit

    if profile is None:
        # No profile and no explicit model: the generic endpoint serves the
        # same known-good first-run starter the bare ``chat``/``run`` use.
        from vllm_mlx.first_run import select_chat_default

        starter, _cached = select_chat_default()
        return starter

    candidates = list(profile.recommended_models)
    if not candidates:
        print("  This agent profile has no recommended models; pass --model.")
        return None

    ram_gb = physical_ram_gb()
    for alias in candidates:
        # ``ram_gb and`` mirrors the pick loop below: a failed RAM probe
        # (0) is treated as "can't judge fit" -> every candidate fits, so a
        # cached recommended model is still returned instead of forcing a
        # re-download of candidates[0].
        if ram_gb and _fits_host(alias, ram_gb) is False:
            continue
        if is_repo_cached(_hf_id(alias)):
            return alias

    if no_download:
        _print_nothing_cached(candidates, ram_gb)
        return None

    # Nothing cached: first recommended model that fits. If RAM is
    # unknowable (0), skip the fit filter so a model can still be picked on
    # machines whose memsize probe failed. The download-consent step runs
    # next and shows the exact transfer size.
    for alias in candidates:
        if ram_gb and _fits_host(alias, ram_gb) is False:
            continue
        return alias

    _print_nothing_fits(candidates, ram_gb)
    return None


def _fits_host(alias: str, ram_gb: float) -> bool | None:
    """Return known fit, known incompatibility, or unknown for an alias."""
    from vllm_mlx.model_aliases import resolve_profile
    from vllm_mlx.recommendations import (
        is_recommended_alias,
        recommendation_footprint_gb,
    )

    # Curated public recommendations encode their supported RAM tiers. For an
    # agent-specific recommendation outside that table, fall back to its
    # explicit minimum-memory contract. Unknown is not the same as "does not
    # fit": the canonical serve path remains the final capacity authority.
    if recommendation_footprint_gb(alias) is not None:
        return is_recommended_alias(alias, ram_gb)
    alias_profile = resolve_profile(alias)
    minimum = alias_profile.min_memory_gb if alias_profile is not None else None
    if minimum is None:
        return None
    return ram_gb >= minimum


def _hf_id(alias: str) -> str:
    """Best-effort alias -> HF repo id, for cache checks."""
    from vllm_mlx.model_aliases import resolve_model

    try:
        return resolve_model(alias)
    except Exception:
        return alias


def _local_connect_host(bind_host: str) -> str:
    """Map wildcard bind addresses to reachable local client addresses."""
    if bind_host in {"", "0.0.0.0"}:
        return "127.0.0.1"
    if bind_host in {"::", "[::]"}:
        return "::1"
    return bind_host


def _print_unknown_agent(name: str) -> None:
    from vllm_mlx.agents import list_profiles

    names = ", ".join(p.name for p in list_profiles())
    print(f"  Unknown agent: {name}")
    print(f"  Known agents: {names}")
    print(
        "  Run 'rapid-mlx agents' for details, or 'rapid-mlx start' "
        "for a generic endpoint."
    )


def _print_nothing_cached(candidates, ram_gb) -> None:
    print("  No recommended model is already cached and --no-download was set.")
    print("  Candidates would have been:")
    for alias in candidates:
        fits = "" if (not ram_gb or _fits_host(alias, ram_gb) is not False) else " ✗"
        print(f"    {alias}{fits}")
    if ram_gb:
        print("  (✗ = does not fit this Mac's RAM)")
    print("  Drop --no-download (or pass --model <cached-alias>) to proceed.")


def _print_nothing_fits(candidates, ram_gb) -> None:
    print(
        f"  None of this agent's recommended models fit this Mac's RAM "
        f"({ram_gb:.1f} GB):"
    )
    for alias in candidates:
        print(f"    {alias}")
    print("  Pass --model <alias> to serve a different model.")


# ---------------------------------------------------------------------------
# Consent + spawn
# ---------------------------------------------------------------------------


def _confirm_download(model: str, *, no_download: bool, yes: bool) -> bool:
    """Consent gate for a first-time (large) download, or a cached pass.

    Reuses ``confirm_or_abort`` from ``vllm_mlx/_download_gate`` for the
    threshold + env-override policy; ``--yes``, ``RAPID_MLX_AUTO_PULL`` and
    non-TTY (CI) all short-circuit to proceed, exactly as the serve path.
    """
    from vllm_mlx._download_gate import (
        confirm_or_abort,
        estimate_download_size_bytes,
        is_repo_cached,
    )

    if os.path.exists(model):
        return True

    hf_id = _hf_id(model)

    if no_download:
        if not is_repo_cached(hf_id):
            print(f"  --no-download set and {model} is not cached; refusing.")
            return False
        return True

    if is_repo_cached(hf_id):
        return True

    env_val = os.environ.get("RAPID_MLX_AUTO_PULL", "").strip().lower()
    if env_val in {"1", "true", "yes"} or yes or not sys.stdin.isatty():
        return True

    size: int | None = None
    try:
        size = estimate_download_size_bytes(hf_id)
    except Exception:
        pass
    return confirm_or_abort(hf_id, size)


def _spawn_foreground_serve(model: str, args) -> subprocess.Popen:
    """Start the canonical ``serve`` path as a foreground, parent-owned child."""
    cmd = [
        sys.executable,
        "-m",
        "vllm_mlx.cli",
        "serve",
        model,
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    child_env = os.environ.copy()
    # The parent already ran the download-consent gate; suppress the child's
    # own B2 re-prompt (chat spawner pattern).
    child_env["RAPID_MLX_CHAT_SPAWN"] = "1"
    # If the start parent is SIGKILLed, the child self-terminates instead of
    # orphan-locking the model + port.
    child_env["RAPID_MLX_WATCHDOG_PPID"] = str(os.getpid())
    if args.no_download:
        # ``--no-download`` is a strict execution contract, not just a
        # parent-side preflight. The canonical serve path already honors both
        # standard offline flags across its Hub/config/tokenizer loaders.
        child_env["HF_HUB_OFFLINE"] = "1"
        child_env["TRANSFORMERS_OFFLINE"] = "1"
    print(f"  Starting {model} on {args.host}:{args.port} (Ctrl-C to stop) ...")
    return subprocess.Popen(  # noqa: S603
        cmd,
        env=child_env,
        # Keep terminal Ctrl-C from reaching both parent and child.  The
        # parent owns signal delivery through ``_foreground_child`` so the
        # server sees one graceful-shutdown signal instead of two.
        start_new_session=True,
    )


class _ForwardedSignalError(Exception):
    """Internal control flow used to leave prompts/waits after a signal."""

    def __init__(self, signum: int):
        self.signum = signum
        super().__init__(signum)


class _foreground_child:
    """Context manager: relay SIGINT/SIGTERM to a spawned serve child.

    Install the relay IMMEDIATELY after ``Popen`` — before the (potentially
    long) download/load readiness phase — so a Ctrl-C or ``kill -INT/-TERM``
    at ANY point is forwarded to the child instead of running with default
    dispositions (which would leave the child to the watchdog and print a raw
    ``KeyboardInterrupt`` traceback in the parent). Restores the previous
    handlers on exit.
    """

    def __init__(self, proc):
        self._proc = proc

    def __enter__(self):
        self._prev_int = signal.signal(signal.SIGINT, self._forward)
        self._prev_term = signal.signal(signal.SIGTERM, self._forward)
        return self

    def __exit__(self, _exc_type, _exc, _tb):
        signal.signal(signal.SIGINT, self._prev_int)
        signal.signal(signal.SIGTERM, self._prev_term)
        if _exc_type is not None and self._proc.poll() is None:
            if _exc_type is _ForwardedSignalError:
                _reap_forwarded_child(self._proc)
            else:
                _terminate_child(self._proc)
        return False

    def _forward(self, signum, _frame):
        try:
            self._proc.send_signal(signum)
        except (ProcessLookupError, PermissionError):
            pass
        # Replacing Python's default handler must not swallow the signal and
        # leave the parent blocked in input()/wait. The surrounding context
        # performs bounded cleanup before start_command maps this to 128+N.
        raise _ForwardedSignalError(signum)


def _terminate_child(proc, *, grace_s: float = 5.0) -> None:
    """Bounded SIGTERM→SIGKILL cleanup for a parent-owned child."""
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
    except (ProcessLookupError, PermissionError):
        return
    try:
        proc.wait(timeout=grace_s)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        proc.kill()
    except (ProcessLookupError, PermissionError):
        return
    try:
        proc.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        # SIGKILL can only remain pending for a process stuck in an
        # uninterruptible kernel wait. Do not turn a readiness timeout into
        # an unbounded parent hang; the child watchdog remains a final guard.
        pass


def _reap_forwarded_child(proc, *, grace_s: float = 5.0) -> None:
    """Reap after a relayed signal without immediately sending a second one."""
    if proc.poll() is not None:
        return
    try:
        proc.wait(timeout=grace_s)
        return
    except subprocess.TimeoutExpired:
        # The graceful signal was ignored. Escalate using the normal bounded
        # SIGTERM -> SIGKILL cleanup contract.
        _terminate_child(proc, grace_s=grace_s)


def _wait_ready(base_url: str, proc, timeout_s: int) -> str:
    """Block until /health/ready returns 200, the child exits, or timeout.

    Returns an outcome so the caller can tell ``exited`` (the serve child
    died before becoming ready — reap it) from ``timeout`` (the child is
    STILL ALIVE but never reported healthy — terminate it, never just wait):
    * ``"ready"`` — /health/ready returned 200.
    * ``"exited"`` — the child exited early (code ``proc.returncode``).
    * ``"timeout"`` — the child is alive but not ready within ``timeout_s``.
    * ``"interrupted"`` — the parent received Ctrl-C while waiting.
    """
    from vllm_mlx.cli import _wait_for_chat_server

    try:
        _wait_for_chat_server(base_url, proc, timeout_s=timeout_s)
    except (RuntimeError, TimeoutError) as exc:
        if proc.poll() is not None:
            print(
                f"  Server exited before becoming ready (code {proc.returncode}): {exc}"
            )
            return "exited"
        print(f"  Server did not become ready in {timeout_s}s; stopping server: {exc}")
        return "timeout"
    except KeyboardInterrupt:
        # Ctrl-C during the (potentially long) download/load readiness phase.
        # The child is alive; the signal relay has already forwarded SIGINT to
        # it, so terminate + reap here for a clean teardown (no raw traceback).
        print("\n  Interrupted during startup; stopping the server.")
        return "interrupted"
    return "ready"


def _port_is_busy(host: str, port: int) -> bool:
    from vllm_mlx.cli import _port_is_busy as probe

    return probe(host, port)


# ---------------------------------------------------------------------------
# Port reuse + attach
# ---------------------------------------------------------------------------


def _reuse_or_refuse(
    base_url: str,
    resolved_model: str,
    served_model: str,
    profile,
    args,
) -> int:
    """Port occupied: reuse a compatible healthy server, else refuse clearly.

    Returns an exit code describing the dispatcher's next action:
    * 0 — reused or dry-run previewed; start_command returns directly.
    * Non-zero — refused; start_command returns the refusal.
    """
    from vllm_mlx.agents.adapter import _fetch_models

    api_base_url = f"{base_url.rstrip('/')}/v1"
    served = {str(m.get("id")) for m in _fetch_models(api_base_url) if m.get("id")}
    if not served:
        print(
            f"  Port {args.port} is occupied but not a healthy rapid-mlx "
            "server; pick another port with --port."
        )
        return 1

    hf_id = _hf_id(resolved_model)
    if served.intersection({resolved_model, served_model, hf_id}):
        if args.dry_run:
            print(
                f"  Dry run — port {args.port} already serves {served_model}; "
                f"start would reuse {base_url}."
            )
            return _attach_and_configure(base_url, served_model, profile, args)
        print(f"  Reusing the running server on {base_url} (serving {served_model}).")
        return _attach_and_configure(base_url, served_model, profile, args)

    print(f"  Port {args.port} already serves {sorted(served)}, not {served_model}.")
    print("  Choose a different port with --port, or stop the other server.")
    return 1


def _attach_and_configure(base_url, model, profile, args) -> int:
    """After the (spawned or reused) server is healthy, print + apply config.

    Uses the same setup-plan machinery as ``rapid-mlx agents --setup`` for
    the first-class profiles (claude-code / continue / deepseek-harness) and
    the generic writer otherwise. Never kills the server. Returns nonzero for
    a genuine setup/render failure; callers decide whether they own the server.
    """
    if profile is None:
        return 0 if _print_instructions(profile, base_url, model) else 1

    api_base_url = f"{base_url.rstrip('/')}/v1"
    if args.no_setup:
        return 0 if _print_instructions(profile, api_base_url, model) else 1

    cfg = profile.get_config_for_version(None)
    needs_context = bool(cfg and cfg.template and "{context_length}" in cfg.template)
    if needs_context and not args.dry_run:
        from vllm_mlx.agents.adapter import fetch_context_window

        try:
            context_length = fetch_context_window(api_base_url, model)
        except Exception:
            context_length = None
    else:
        context_length = _cached_context_window(model) if needs_context else None

    if args.dry_run and needs_context and context_length is None:
        print(
            "  Configuration preview deferred: the model's context window "
            "is not available in the local cache."
        )
        print("  Nothing was written.")
        return 0

    is_first_class = profile.name in {"claude-code", "continue", "deepseek-harness"}
    if cfg and getattr(cfg, "type", None) == "env" and not is_first_class:
        from vllm_mlx.agents.adapter import setup_agent_config

        try:
            instructions = setup_agent_config(
                profile,
                api_base_url,
                model,
                dry_run=True,
                context_length=context_length,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            print(f"  {profile.display_name} setup failed: {exc}")
            _print_instructions(profile, api_base_url, model)
            return 1
        print(f"  {profile.display_name} uses shell environment variables:")
        print(instructions)
        return 0

    if is_first_class:
        supports_reasoning = None
        if profile.name == "deepseek-harness" and not args.dry_run:
            from vllm_mlx.agents.adapter import fetch_reasoning_support

            try:
                supports_reasoning = fetch_reasoning_support(api_base_url, model)
            except (OSError, RuntimeError, TypeError, ValueError):
                supports_reasoning = None
        elif profile.name == "deepseek-harness":
            from vllm_mlx.model_aliases import resolve_profile

            model_profile = resolve_profile(model)
            if model_profile is not None:
                supports_reasoning = model_profile.reasoning_parser is not None

        from vllm_mlx.agents.setup import (
            apply_setup_plan,
            build_setup_plan,
            confirm_plan,
        )

        try:
            plan = build_setup_plan(
                profile.name,
                api_base_url,
                model,
                context_length=context_length,
                supports_reasoning=supports_reasoning,
            )
        except (OSError, ValueError) as exc:
            print(f"  {profile.display_name} setup failed: {exc}")
            _print_instructions(profile, api_base_url, model)
            return 1

        print(f"  {profile.display_name} configuration: {plan.path}")
        if plan.changed:
            print(plan.diff())
        else:
            print("  Already configured; no file changes needed.")

        if args.dry_run:
            print("  Dry run only; nothing was written.")
            return 0

        if plan.changed and not args.yes and not confirm_plan(plan):
            print("  Setup cancelled; nothing was written.")
            _print_instructions(profile, api_base_url, model)
            return 0
        if plan.changed:
            try:
                apply_setup_plan(plan)
            except (OSError, RuntimeError) as exc:
                print(f"  {profile.display_name} setup failed: {exc}")
                _print_instructions(profile, api_base_url, model)
                return 1
            print(f"  Configured {profile.display_name} at {plan.path}.")
    else:
        from vllm_mlx.agents.adapter import setup_agent_config

        try:
            preview = setup_agent_config(
                profile,
                api_base_url,
                model,
                dry_run=True,
                context_length=context_length,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            print(f"  {profile.display_name} setup failed: {exc}")
            _print_instructions(profile, api_base_url, model)
            return 1
        if preview.startswith("Cannot"):
            print(f"  {profile.display_name} setup failed.")
            print(f"  {preview}")
            _print_instructions(profile, api_base_url, model)
            return 1
        print(f"  {profile.display_name} configuration: {preview}")
        if args.dry_run:
            return 0
        if not args.yes and not _confirm_config_write():
            print("  Setup cancelled; nothing was written.")
            _print_instructions(profile, api_base_url, model)
            return 0

        try:
            summary = setup_agent_config(
                profile,
                api_base_url,
                model,
                dry_run=False,
                context_length=context_length,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            print(f"  {profile.display_name} setup failed: {exc}")
            _print_instructions(profile, api_base_url, model)
            return 1
        if summary.startswith("Cannot"):
            print(f"  {profile.display_name} setup failed.")
            print(f"  {summary}")
            _print_instructions(profile, api_base_url, model)
            return 1
        print(f"  {profile.display_name} configured! {summary}")

    _print_next_steps(profile.name, api_base_url, model)
    return 0


def _confirm_config_write() -> bool:
    """Require interactive consent before a generic profile config write."""
    if not sys.stdin.isatty():
        return False
    try:
        return input("Apply this configuration? [y/N] ").strip().lower() in {
            "y",
            "yes",
        }
    except (EOFError, KeyboardInterrupt):
        return False


def _cached_context_window(model: str) -> int | None:
    """Read a cached/local config context limit without network or weights."""
    from vllm_mlx.model_metadata import read_model_metadata

    try:
        metadata = read_model_metadata(_hf_id(model))
    except (OSError, RuntimeError, TypeError, ValueError):
        return None
    config = metadata.config if metadata is not None else None
    if not isinstance(config, dict):
        return None
    candidates = [config.get("max_position_embeddings")]
    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        candidates.append(text_config.get("max_position_embeddings"))
    for candidate in candidates:
        if (
            isinstance(candidate, int)
            and not isinstance(candidate, bool)
            and candidate > 0
        ):
            return candidate
    return None


def _print_instructions(profile, base_url, model) -> bool:
    if profile is None:
        print(f"  OpenAI-compatible endpoint: {base_url}/v1 (model {model})")
        return True
    from vllm_mlx.agents.adapter import get_setup_instructions

    try:
        instructions = get_setup_instructions(profile, base_url, model)
    except Exception as exc:
        print(f"  (Could not render setup instructions: {exc})")
        return False
    else:
        print()
        print(instructions)
        return True


def _print_next_steps(profile_name, base_url, model) -> None:
    print()
    print(f"  {profile_name} is configured to talk to {base_url} (model {model}).")


# ---------------------------------------------------------------------------
# Child lifecycle
# ---------------------------------------------------------------------------


def _wait_child(proc) -> int:
    """Wait for the serve child and return its exit code.

    Signal relay is handled by the enclosing ``_foreground_child`` (active
    from spawn), so this just reaps the child. Signal deaths are mapped to
    the POSIX ``128 + signum`` shell-representable code rather than the raw
    negative returncode.
    """
    import time

    while proc.poll() is None:
        time.sleep(0.1)
    code = proc.returncode
    if code is None:
        return 1
    code = int(code)
    if code < 0:  # died by signal: -SIGKILL/-SIGTERM/etc. -> 128 + signum
        return 128 - code
    return code


def _print_dry_run(profile_name, model, args) -> None:
    from vllm_mlx.recommendations import physical_ram_gb

    print("  Dry run — nothing started, downloaded, or written.")
    print(f"  Agent:  {profile_name or 'generic OpenAI-compatible'}")
    print(f"  Model:  {model}")
    print(f"  Serve:  rapid-mlx serve {model} --host {args.host} --port {args.port}")
    ram_gb = physical_ram_gb()
    if ram_gb:
        print(f"  RAM:    {ram_gb:.1f} GB")
