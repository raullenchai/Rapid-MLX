#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
# SPDX-License-Identifier: Apache-2.0
"""
CLI for rapid-mlx (package name: ``vllm_mlx``).

Commands:
    rapid-mlx serve <model> --port 8000    Start OpenAI-compatible server
    rapid-mlx bench <model>                Run benchmark
    rapid-mlx chat <model>                 Interactive chat REPL

Usage:
    rapid-mlx serve qwen3.5-4b-4bit --port 8000
    rapid-mlx bench qwen3.5-4b-4bit --num-prompts 10
    rapid-mlx chat qwen3.5-4b-4bit
"""

import argparse
import os
import sys

from vllm_mlx._completion import alias_completer

# Project-default mirror for ``RAPID_MLX_MODEL_MIRROR`` (consumed by
# ``_try_mirror_prefetch``). Public Cloudflare Worker → R2 bucket, with
# rate-limit + Range-request passthrough. Override with the env var
# (set to an empty string to disable the mirror and force HF Hub).
MIRROR_DEFAULT = "https://models.rapidmlx.com"

# NOTE: ``argcomplete`` is imported lazily inside ``main()`` instead of
# at module top. Module-level imports of ``vllm_mlx.cli`` (e.g.
# ``tests/test_harmony_parsers.py::TestServeLogLevelFlags``) run in the
# minimal-deps CI lane that doesn't pre-install argcomplete; pulling it
# at top would surface as ``ModuleNotFoundError`` during test collection.
# argcomplete is still a required runtime dep in ``pyproject.toml`` so
# real installs get tab completion out of the box.


def _log_level_choice(value: str) -> str:
    """Argparse ``type`` callable: normalize to upper-case so
    ``--log-level info`` is accepted as ``INFO``. Named (not a lambda)
    so argparse's error messages read sensibly instead of
    ``invalid <lambda> value``.
    """
    return value.upper()


def _auth_feature_str(argv_api_key: str | None) -> str | None:
    """Banner-side renderer for the ``auth: on`` feature line.

    Returns ``"auth: on"`` when the effective API key (argv or env)
    is non-empty, else ``None`` so the banner omits the feature.

    Lives at module scope (not inline in ``serve_command``) so the
    banner gate is directly unit-testable without booting a model.
    Routes through ``server._resolve_api_key`` — the same SSOT the
    server-side enforcement reads — so a refactor of the env-var
    policy cannot drift the banner from the actual auth state.
    Pre-fix the gate was ``if args.api_key`` directly, which printed
    ``auth: off`` for env-only sidecars even though
    ``verify_api_key`` was enforcing (dogfood-v0.8.2 finding #3).
    """
    from vllm_mlx import server as _server

    if _server._resolve_api_key(argv_api_key):
        return "auth: on"
    return None


def _port_arg(value: str) -> int:
    """Argparse ``type`` callable: validate ``--port`` is in [1, 65535].

    Without this, ``rapid-mlx chat --port 99999`` parsed successfully and
    dropped the user into a REPL whose first turn failed with a confusing
    ``Failed to parse: http://127.0.0.1:99999/...``. Validate early so the
    user sees a one-line argparse error instead.
    """
    try:
        port = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"port must be an integer, got {value!r}"
        ) from None
    if not (1 <= port <= 65535):
        raise argparse.ArgumentTypeError(
            f"port must be between 1 and 65535, got {port}"
        )
    return port


def _listen_fd_arg(value: str) -> int:
    """Argparse ``type`` callable: validate ``--listen-fd`` is a sane fd.

    ``--listen-fd`` enables socket activation — the supervisor (launchd,
    systemd, an external parent process) binds the listening socket
    itself and execve's into ``rapid-mlx serve`` with the pre-bound fd.
    This closes the bind→auth TOCTOU window: by the time rapid-mlx
    runs, the socket is already bound but no requests can be accepted
    until ``uvicorn.run`` calls ``accept()`` — at which point the
    FastAPI app (with all route auth dependencies wired) is already
    constructed. See ``vllm_mlx/server.py`` and the regression test
    pinning the bind→auth invariant.

    Accept integers in ``[3, 1023]``:

    * 0/1/2 are stdin/stdout/stderr — never a listening socket.
    * 3 is the conventional "first non-stdio fd" (systemd's
      ``LISTEN_FDS_START`` and launchd both follow this convention).
    * 1023 is the SysV soft-limit ceiling — anything higher is almost
      certainly a typo, not a real fd.
    """
    try:
        fd = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--listen-fd must be an integer, got {value!r}"
        ) from None
    if not (3 <= fd <= 1023):
        raise argparse.ArgumentTypeError(
            f"--listen-fd must be between 3 and 1023, got {fd}"
        )
    return fd


def _apply_body_receive_timeout_env(server_mod, *, logger=None) -> None:
    """Resolve ``RAPID_MLX_BODY_RECEIVE_TIMEOUT_SECONDS`` onto
    ``server_mod._body_receive_timeout_seconds`` (H-14 / F-072
    slow-DoS gate).

    Extracted from ``serve_command`` so the
    ``tests/test_body_receive_timeout.py::test_h14_env_var_override_reduces_timeout``
    case exercises the same code path the production binary runs —
    codex round-2 BLOCKING on PR #786 spotted that an inline-only
    resolver couldn't be unit-tested without duplicating its logic,
    which would silently mask a regression that deleted the wire-up.

    Behaviour:
      * No env var (or empty after strip) → leave the existing
        ``server_mod._body_receive_timeout_seconds`` untouched (the
        module's documented 15 s default).
      * Numeric env value → clamp via ``max(0.0, float(...))`` so
        negative numbers disable the gate without crashing.
      * Non-numeric env value → log a warning and explicitly write
        the 15 s default back to ``server_mod`` (an inherited
        non-default from a prior call would otherwise leak).

    ``server_mod`` is passed in so the test can hand a fresh
    ``vllm_mlx.server`` reference each call without dragging the
    whole import-time CLI prologue along.
    """
    import os

    if logger is None:  # pragma: no cover — tests always pass a logger
        import logging as _logging

        logger = _logging.getLogger(__name__)

    _brt_env_name = "RAPID_MLX_BODY_RECEIVE_TIMEOUT_SECONDS"
    _brt_env = os.environ.get(_brt_env_name, "").strip()
    if not _brt_env:
        return
    try:
        server_mod._body_receive_timeout_seconds = max(0.0, float(_brt_env))
    except ValueError:
        # Interpolate the env-var name via ``%s`` instead of baking it
        # into the format string — same false-positive avoidance
        # pattern as the SSE-keepalive block above.
        logger.warning(
            "%s=%r is not a number; falling back to the 15 s default",
            _brt_env_name,
            _brt_env,
        )
        server_mod._body_receive_timeout_seconds = 15.0


def _wildcard_host_aliases() -> frozenset[str]:
    """Strings that name "bind on every interface" rather than a single
    address. Python's ``socket.bind(("", N))`` and ``socket.bind(("0.0.0.0",
    N))`` are equivalent for IPv4; uvicorn historically treats both the
    empty string and ``0.0.0.0`` the same way. We treat them as a single
    class for the loopback-collision pre-flight (codex round-1 MAJOR on
    PR #848: original gate only matched ``"0.0.0.0"`` so ``--host ""``
    could still re-open the dual-bind ambiguity).

    Kept as a function rather than a module constant so the test suite
    can monkey-patch it in case a future host alias (e.g. ``"::"`` once we
    grow IPv6 pre-flight) needs to land without touching every call site.
    """
    return frozenset({"0.0.0.0", ""})


def _is_ipv6_host(host: str) -> bool:
    """Detect IPv6 literal hosts (``::``, ``::1``, ``2001:db8::1`` ...).

    Codex round-1 MED #6 on PR #855: the IPv4-only preflight always
    created an ``AF_INET`` socket, so any valid uvicorn IPv6 bind
    (``--host ::1``, ``--host ::``, etc.) failed ``socket.bind`` and got
    misreported as "port already in use." Detection is colon-based:
    every IPv6 literal contains at least one ``:``, no IPv4 literal /
    DNS name does (``localhost`` is the canonical non-IPv6 with no
    colon). We deliberately keep this purely lexical — a stricter
    ``ipaddress.ip_address`` parse would reject scoped literals
    (``fe80::1%en0``) that uvicorn happily accepts.
    """
    return ":" in host


def _port_preflight_or_die(host: str, port: int, *, model: str) -> None:
    """Probe ``(host, port)`` AND — when ``host`` is a wildcard alias —
    additionally probe ``("127.0.0.1", port)``. Print a friendly error
    and ``sys.exit(1)`` on the first collision.

    Why both: macOS / Linux let a wildcard listener (``0.0.0.0`` or
    ``""``) coexist with a more-specific loopback listener
    (``127.0.0.1``) on the same port. v0.8.2 dogfood finding #2
    reproduced the resulting PortSweep bypass: ``nc -l 127.0.0.1 11812``
    + ``rapid-mlx serve --port 11812`` BOTH succeed, and
    ``curl 127.0.0.1:11812/healthz`` returns HTTP 000 (kernel routes
    loopback to nc, not rapid-mlx). The fix is to explicitly probe the
    loopback address whenever the requested bind is wider than loopback.

    Extracted from ``serve_command`` so the legacy
    ``python -m vllm_mlx.server`` entrypoint can call it too without
    duplicating the wildcard-alias / probe-loop logic — codex round-1
    MAJOR on PR #848 (the dogfood-CLI fix had to land on both supported
    entrypoints to actually close the bypass).

    ``::1`` is intentionally NOT probed when the user binds an IPv4
    wildcard: macOS treats v4 and v6 loopback as distinct stacks, and
    uvicorn's IPv4 bind never collides with an IPv6 listener. When the
    user EXPLICITLY binds an IPv6 host, we switch the probe family to
    ``AF_INET6`` so the bind doesn't spuriously fail (codex round-1
    MED #6 on PR #855 — pre-fix ``--host ::1`` raised ``OSError`` from
    the ``AF_INET`` socket and was misreported as "port already in use").
    """
    import socket

    wildcards = _wildcard_host_aliases()
    if host in wildcards:
        # Probe the requested wildcard FIRST (so a LAN-side port
        # collision still surfaces the user-supplied host name in the
        # error), then probe 127.0.0.1 to catch the loopback shadow.
        hosts_to_probe: tuple[str, ...] = (host, "127.0.0.1")
    else:
        hosts_to_probe = (host,)

    for probe_host in hosts_to_probe:
        # Pick the address family that matches the host string. IPv6
        # literals (``::``, ``::1``, etc.) need ``AF_INET6`` or the bind
        # raises before we can detect a real collision (codex r1 MED #6
        # on PR #855). Everything else — IPv4 literals, wildcards
        # (``0.0.0.0``, ``""``), the loopback-shadow probe ``127.0.0.1``
        # — stays on ``AF_INET``.
        family = socket.AF_INET6 if _is_ipv6_host(probe_host) else socket.AF_INET
        # ``with`` guarantees the preflight socket is closed on every
        # exit path — including OSError during ``bind``. The previous
        # form called ``_sock.close()`` only on the success branch,
        # which leaked the fd whenever the bind raised (e.g. when
        # running under a test harness that catches ``SystemExit``).
        with socket.socket(family, socket.SOCK_STREAM) as _sock:
            _sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                _sock.bind((probe_host, port))
            except OSError:
                # Surface the host we actually collided on so the user
                # can distinguish "LAN port busy" from "loopback port
                # already claimed by another rapid-mlx / nc / proxy".
                # Use the empty-string-friendly display name so
                # ``--host ""`` shows up as ``0.0.0.0`` rather than a
                # confusing bare quote.
                display_host = probe_host or "0.0.0.0"
                print(f"\n  Error: Port {port} is already in use on {display_host}.")
                print(
                    f"  Try a different port: rapid-mlx serve {model} --port {port + 1}"
                )
                sys.exit(1)


def _run_uvicorn(app, args, log_level: str) -> None:
    """Dispatch into ``uvicorn.run`` with the kwargs that match the
    current ``--listen-fd`` / ``--host``/``--port`` mode.

    Extracted so the call-site contract is unit-testable WITHOUT booting
    the heavy ``serve_command`` prologue (version check, model download,
    server import). The companion bytecode test in
    ``tests/test_serve_listen_fd.py`` pins that ``serve_command``
    actually references this helper so a future refactor that drops the
    dispatch silently is caught — that's the regression-detection codex
    round-1 PR #696 review was after.
    """
    import uvicorn

    listen_fd = getattr(args, "listen_fd", None)
    if listen_fd is not None:
        # ``fd=`` overrides ``host``/``port``: uvicorn skips its own
        # ``socket.bind()`` and adopts the inherited fd directly. This
        # is the close of the bind→auth TOCTOU window — the supervisor
        # bound + validated the auth secret BEFORE execve'ing, and the
        # FastAPI ``app`` (with route auth dependencies) is fully
        # constructed at module load before this call.
        uvicorn.run(
            app,
            fd=listen_fd,
            log_level=log_level,
            timeout_keep_alive=30,
        )
    else:
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level=log_level,
            timeout_keep_alive=30,
        )


def _chat_config_dir() -> str:
    """Directory for first-launch tip markers (and future per-user chat
    state). Honors ``RAPID_MLX_CONFIG_HOME`` override; otherwise falls back
    to ``~/.config/rapid-mlx``. The directory is created lazily by the
    writer; callers don't need to ensure it exists for reads.
    """
    override = os.environ.get("RAPID_MLX_CONFIG_HOME")
    if override:
        return override
    return os.path.join(os.path.expanduser("~"), ".config", "rapid-mlx")


def _seen_tips_path() -> str:
    return os.path.join(_chat_config_dir(), "seen-tips.json")


def _has_seen_tip(key: str) -> bool:
    """Return True iff the marker file records ``key: true``.

    Any IO/parse error is treated as "not seen" — better to show the tip
    one extra time than to hide it forever on a corrupt marker.
    """
    import json

    try:
        with open(_seen_tips_path(), encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return False
    return isinstance(data, dict) and bool(data.get(key))


def _mark_tip_seen(key: str) -> None:
    """Persist ``key: true`` to the seen-tips marker. Best-effort —
    failures are swallowed so a read-only config dir never aborts chat.
    """
    import json

    path = _seen_tips_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except OSError:
        return
    try:
        existing: dict = {}
        try:
            with open(path, encoding="utf-8") as fh:
                loaded = json.load(fh)
            if isinstance(loaded, dict):
                existing = loaded
        except (OSError, ValueError):
            existing = {}
        existing[key] = True
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(existing, fh)
    except OSError:
        return


def _print_unknown_model_help(name: str, *, full_path_example: str) -> None:
    """Print fuzzy suggestions + a curated popular-models hint.

    Replaces the older "Did you mean: X?" + "Run `rapid-mlx models`" pattern
    that left users empty-handed when no close fuzzy match existed
    (e.g. ``rapid-mlx chat gemma4-27b`` returned zero suggestions, told the
    user to run another command, and gave no hint of what was actually
    supported). Now: always show *something* — fuzzy matches when we have
    them, curated popular aliases when we don't.
    """
    from vllm_mlx.model_aliases import POPULAR_ALIASES, list_aliases, suggest_similar

    suggestions = suggest_similar(name)
    if suggestions:
        print(f"  Did you mean: {', '.join(suggestions)}?")
    else:
        print(f"  Try one of: {', '.join(POPULAR_ALIASES)}")
    print(f"  Run `rapid-mlx models` to see all {len(list_aliases())} aliases,")
    print(f"  or pass a full path like: {full_path_example}")


def _embedding_not_found_exception_classes() -> tuple[type[BaseException], ...]:
    """Return the concrete exception classes the embedding loader raises
    for a missing model.

    pr_validate codex r1 NIT: matching ``"not found"`` as a substring of
    the exception text was too loose — a future ``ValueError("config
    field 'x' not found in tensor map")`` from a corrupt model could be
    mis-translated as the alias/HF-id hint, masking the real bug. Bind
    to the actual classes so the wrap-path only fires on the
    well-defined not-found shape.

    Lazy import so the base install (no ``[embeddings]`` extra, no
    ``huggingface_hub`` shadow) stays free of these imports until the
    code path actually runs. Missing classes are silently skipped — the
    caller's tuple-based ``except`` accepts an empty tuple as a no-op,
    so a sparse environment falls back to "re-raise everything", which
    is the safe default.
    """
    classes: list[type[BaseException]] = [FileNotFoundError]
    try:  # mlx_embeddings — installed via the [embeddings] extra
        from mlx_embeddings.utils import ModelNotFoundError

        classes.append(ModelNotFoundError)
    except Exception:  # pragma: no cover — defensive
        pass
    try:  # huggingface_hub — transitive of mlx_embeddings
        from huggingface_hub.errors import (
            EntryNotFoundError,
            RepositoryNotFoundError,
        )

        classes.append(RepositoryNotFoundError)
        classes.append(EntryNotFoundError)
    except Exception:  # pragma: no cover — defensive
        pass
    return tuple(classes)


def _resolve_embedding_alias(name: str) -> tuple[str, bool]:
    """Resolve a ``--embedding-model`` alias through the shared registry.

    D-EMBED-ALIAS: Sarah F-S2-1 — the positional chat-model arg goes
    through ``resolve_model`` at the CLI dispatch (cli.py ~5660), but
    the ``--embedding-model`` flag was passed verbatim to
    ``mlx_embeddings.load`` and crashed with ``ModelNotFoundError`` on
    any alias.

    Returns ``(resolved, did_resolve)``. ``did_resolve`` is True when
    the registry actually mapped ``name`` to a different HF path —
    used by the caller to log the alias hop.
    """
    from .model_aliases import resolve_model

    resolved = resolve_model(name)
    return resolved, resolved != name


def _resolve_audio_model_for_serve(model_name: str):
    """Resolve a model name to an audio registry entry, if it's audio.

    R10-C1: pre-fix ``serve_command`` had a boot guard (rc=2 when
    ``[audio]`` extra missing) but ZERO resolution logic for audio
    aliases. Short aliases like ``kokoro``/``whisper`` then fell into
    ``_ensure_model_downloaded`` and 404'd at HF, while full HF ids
    of audio models (``mlx-community/Kokoro-82M-bf16``) downloaded
    successfully but crashed in ``mlx_lm.load_model`` because they
    have no safetensors. Bo r10-R1: 0/8 audio aliases boot on 0.8.11.

    The fix routes audio names through a SEPARATE serve path that
    skips the text-model loader entirely. This helper returns the
    resolved registry entry (so the dispatcher knows the HF id, type,
    family, voice list) or ``None`` if the name isn't audio. ``None``
    falls through to the legacy text path unchanged — text-model boot
    paths must not regress.
    """
    from .audio.registry import resolve_audio_alias

    return resolve_audio_alias(model_name)


def _serve_audio_mode(args, entry) -> None:
    """Bind the audio-only serve path for a resolved registry entry.

    R10-C1 audio-serve-mode. Pre-fix every ``rapid-mlx serve kokoro``
    crash-looped because the text-model boot path was the ONLY path:

    1. ``_ensure_model_downloaded(args.model)`` queried HF for the
       short alias and 404'd — there's no ``hf.co/kokoro`` repo.
    2. Even when the user supplied a full HF id, ``load_model``
       (text path) called ``mlx_lm.load_model`` which expects
       safetensors. Audio repos ship npz/mlx weights, so the loader
       crashed with "no safetensors found".
    3. ``pflash.validate_model_support`` and the parser auto-detection
       both consult ``args.model`` assuming it's a text-LM alias —
       a wrong tool for audio.

    The audio-serve-mode bypasses all of the above:

    * Print the resolved alias -> HF id banner so the operator sees
      the same alias-resolution UX they get for text models.
    * Stamp the resolved HF id on ``args.model`` so the audio routes
      treat it as a known engine (``STT_MODEL_ALIASES`` /
      ``TTS_MODEL_ALIASES`` map both the short and full forms).
    * Capture the alias on ``server._model_alias`` so ``/v1/models``
      advertises it.
    * Configure server security knobs (api-key, body-size cap, CORS)
      the SAME way the text path does — audio endpoints share the
      same middleware stack.
    * Skip the text-LM loader. The audio engines are loaded LAZILY
      on the first request by the route handlers (``STTEngine.load``
      / ``TTSEngine.load``), so there's nothing to boot at startup —
      and a Kokoro/Whisper weight download mid-boot would only add
      cold-start latency without buying anything.
    * Run uvicorn with the same FastAPI ``app`` text models use; the
      ``/v1/audio/*`` routes are already mounted on it.
    """
    import os
    import sys

    # Late imports — audio mode runs on the lighter base install +
    # ``[audio]`` extra; we don't want the text-LM engine machinery to
    # boot until / unless it's actually needed.
    from . import server
    from .middleware.auth import configure_rate_limiter
    from .server import app

    uvicorn_log_level = server.configure_logging(args.log_level)

    # Stamp the resolved model id so the audio routes find the same
    # alias mapping the registry has. ``server._model_alias`` is read
    # by ``/v1/models`` to surface the operator-facing alias name;
    # ``server._model_name`` / ``server._model_path`` populate
    # ``ServerConfig.model_name`` / ``model_path`` so /v1/models lists
    # the served audio model (codex r1 HIGH #1 follow-up).
    if hasattr(args, "_original_alias") and args._original_alias is not None:
        server._model_alias = args._original_alias
    else:
        # No prior alias hop (e.g. user passed a full HF id). Use the
        # short alias from the registry so /v1/models still shows the
        # friendly name, not the bare HF path.
        server._model_alias = entry.alias
    server._model_name = entry.hf_id
    server._model_path = entry.hf_id

    # Mirror the text path's security configuration. Audio routes use
    # the SAME middleware stack as chat/embeddings — the same env vars
    # and CLI flags govern auth + body-size caps + CORS. Diverging
    # here would silently weaken the deployment posture for anyone who
    # added ``--api-key`` to their ``rapid-mlx serve kokoro`` command.
    server._api_key = server._resolve_api_key(args.api_key)
    server._default_timeout = args.timeout

    _max_body_arg = getattr(args, "max_request_bytes", None)
    if _max_body_arg is not None:
        server._max_request_bytes = max(0, int(_max_body_arg))
    else:
        _env = os.environ.get("RAPID_MLX_MAX_REQUEST_BYTES", "").strip()
        if _env:
            try:
                server._max_request_bytes = max(0, int(_env))
            except ValueError:
                server._max_request_bytes = 8 * 1024 * 1024

    # Body-receive timeout — same env-driven hook the text path uses.
    _apply_body_receive_timeout_env(server)

    # CORS — same friendly default the text path uses.
    server.configure_cors_from_env(args.cors_origins)
    if args.rate_limit > 0:
        server._rate_limiter = configure_rate_limiter(args.rate_limit, enabled=True)

    # CRITICAL: copy the just-set server globals into the
    # ServerConfig singleton the middleware actually reads.
    # ``server.load_model`` does this on the text path (calls
    # ``_sync_config`` after wiring globals); the audio path skips
    # ``load_model`` so we must call it explicitly here. Without this
    # sync the auth middleware reads ``cfg.api_key`` (still ``None``
    # because nothing populated it) instead of ``server._api_key``,
    # so ``rapid-mlx serve kokoro --api-key SECRET`` would silently
    # accept unauthenticated /v1/audio/* requests. Codex r1 HIGH #1.
    server._sync_config()

    # Print the resolution banner so the operator sees what loaded.
    family_tag = f"[audio:{entry.type}]"
    shown_alias = getattr(args, "_original_alias", args.model)
    print()
    print(f"  Audio mode: {shown_alias} → {entry.hf_id} {family_tag}")
    if entry.type == "tts" and entry.default_voice:
        print(f"  Default voice: {entry.default_voice}")
    if entry.type == "stt" and entry.languages:
        print(f"  Languages: {entry.languages}")
    print(
        "  Audio engines load lazily on the first /v1/audio/* request "
        "(no boot-time weight download)."
    )

    # Stamp the bind source-of-truth so the lifespan "Ready:" banner
    # prints the right URL. Mirrors the text-path block.
    host_display = "localhost" if args.host == "0.0.0.0" else args.host
    listen_fd = getattr(args, "listen_fd", None)

    # Port preflight — same friendly "port already in use" probe the
    # text path runs. Skip in --listen-fd mode (the supervisor owns
    # the socket; binding here would race). Mirrors the rationale on
    # the text-path call site.
    if listen_fd is None:
        _port_preflight_or_die(args.host, args.port, model=args.model)

    if listen_fd is not None:
        print(
            f"  Starting server on inherited fd {listen_fd} "
            "(audio routes ready immediately)"
        )
    else:
        print(
            f"  Starting server on http://{host_display}:{args.port} "
            "(audio routes ready immediately)"
        )

    from vllm_mlx._version_check import print_staleness_warning_if_any
    from vllm_mlx.config import get_config

    print_staleness_warning_if_any()
    print()

    _cfg = get_config()
    _cfg.bind_host = None
    _cfg.bind_port = None
    _cfg.bind_listen_fd = None
    if listen_fd is None:
        _cfg.bind_host = host_display
        _cfg.bind_port = args.port
    else:
        _cfg.bind_listen_fd = listen_fd

    # Use sys.stdout.flush so the banner lands before uvicorn's own
    # startup logs interleave — operators expect to see the audio
    # banner FIRST.
    sys.stdout.flush()

    _run_uvicorn(app, args, uvicorn_log_level)


def _load_embedding_model_or_exit(args, load_fn) -> None:
    """Pre-load ``--embedding-model`` with the H-08 install guard and
    the D-EMBED-ALIAS alias-resolution + clean error-wrapping path.

    Lifted out of ``serve_command`` so the dispatch sequence can be
    unit-tested without booting the full engine — the pr_validate
    codex r0 BLOCKING #1 noted that the in-test exercising the
    behaviour at module scope didn't actually invoke the CLI path,
    so a regression that removed the alias resolution would pass.
    Calling this helper directly gives the test surgical coverage.

    ``args`` mirrors the ``argparse.Namespace`` shape — only
    ``embedding_model`` is read and (on alias hit) mutated.
    ``load_fn`` is the embedding-loader callable
    (``vllm_mlx.server.load_embedding_model``) — passed in so tests
    can mock it without monkeypatching the server module.

    Failure modes that exit cleanly:

    * Missing ``[embeddings]`` extra → ``sys.exit(2)`` with install
      hint (H-08, ``require_mlx_embeddings_or_exit``).
    * Loader raises ``ModelNotFoundError`` / ``RepositoryNotFoundError``
      / ``FileNotFoundError`` → ``sys.exit(1)`` with an actionable
      hint pointing at the alias registry and the canonical HF id
      format. Any OTHER ``Exception`` re-raises so unrelated bugs
      surface with their real trace.

    Audio-mode integration (deferred #258 / r11-K coordination): if
    ``_serve_audio_mode`` ever needs to honour ``--embedding-model``
    (e.g. an STT lane that exposes embeddings of the transcript), the
    audio path MUST route through this helper rather than duplicate
    the guard logic. The probe + alias resolve + error-wrap are a
    single source of truth — a second copy in the audio dispatcher
    would drift on the next H-08/H-09/H-13 follow-up. The helper is
    intentionally independent of the text-LM serve path so the audio
    boot path can call it without dragging in the chat-engine
    machinery.
    """
    from .embedding import require_mlx_embeddings_or_exit

    require_mlx_embeddings_or_exit()

    original_embed = args.embedding_model
    resolved_embed, did_resolve = _resolve_embedding_alias(original_embed)
    if did_resolve:
        print(f"  Embedding alias: {original_embed} → {resolved_embed}")
        args.embedding_model = resolved_embed
    print(f"Pre-loading embedding model: {args.embedding_model}")
    # Bind to the concrete not-found classes the loader can raise
    # (mlx_embeddings.utils.ModelNotFoundError +
    # huggingface_hub.errors.RepositoryNotFoundError/EntryNotFoundError
    # + stdlib FileNotFoundError for the local-path branch). Any OTHER
    # exception class falls through unchanged so unrelated bugs (corrupt
    # safetensors mid-load, Metal OOM, schema mismatch) surface with
    # their real trace — pr_validate codex r1 NIT closure (the prior
    # ``"not found"`` substring match was too loose).
    not_found_exc_classes = _embedding_not_found_exception_classes()
    try:
        load_fn(args.embedding_model, lock=True)
    except not_found_exc_classes as exc:
        print(
            f"\n  Error: --embedding-model '{original_embed}' could not "
            f"be loaded ({type(exc).__name__}: {exc})."
        )
        print(
            "  Tip: use a registered embedding alias (see "
            "``rapid-mlx ls`` for the list — e.g. "
            "``embeddinggemma-300m-6bit``) or pass the full "
            "HuggingFace id (e.g. "
            "``mlx-community/embeddinggemma-300m-6bit``).\n"
        )
        sys.exit(1)
    print(f"Embedding model loaded: {args.embedding_model}")


def _check_disk_space(model_name: str, force: bool = False) -> None:
    """Verify there's enough disk space to download the model.

    Queries HuggingFace for the repo size and compares with available space
    on the resolved HF cache filesystem (respects ``HF_HOME`` /
    ``HF_HUB_CACHE`` rather than the hard-coded ``~/.cache/huggingface``).

    Behaviour:

    - Model is already a local path → return.
    - ``config.json`` is in the cache → assume already downloaded → return.
    - HF API call fails (offline, gated repo, etc.) → return silently. The
      loader's 404/auth handlers will surface the real error if there is one.
    - Determined size and disk is insufficient → print actionable error
      and ``sys.exit(1)``. ``force=True`` warns instead of aborting.

    The previous behaviour was to print a soft warning then continue. Users
    burned 30+ minutes downloading a 141 GB model on an 8.8 GB disk before
    HF Hub crashed with ``OSError: No space left on device``.
    """
    # Skip if model is a local path that already exists.
    if os.path.exists(model_name):
        return

    # Skip if model is already in the HF cache.
    try:
        from huggingface_hub import try_to_load_from_cache

        cached = try_to_load_from_cache(model_name, "config.json")
        if isinstance(cached, str) and os.path.exists(cached):
            return
    except Exception:
        pass

    # Query HF for repo size + free space on the actual HF cache filesystem.
    try:
        from huggingface_hub import model_info
        from huggingface_hub.constants import HF_HUB_CACHE

        info = model_info(model_name, files_metadata=True)
        model_size_bytes = sum(
            (s.size or 0)
            for s in (getattr(info, "siblings", None) or [])
            if hasattr(s, "size")
        )
        if model_size_bytes == 0:
            return  # Can't determine size — skip rather than guess.

        # statvfs needs an existing path; HF_HUB_CACHE may not exist yet on
        # a fresh install. Walk up to the first ancestor that does.
        # Resolve to absolute up front so a relative HF_HUB_CACHE doesn't
        # short-circuit to CWD when an ancestor walk hits ".".
        probe = os.path.abspath(HF_HUB_CACHE) if HF_HUB_CACHE else ""
        while probe and not os.path.exists(probe):
            parent = os.path.dirname(probe)
            if parent == probe:
                break
            probe = parent
        if not probe or not os.path.exists(probe):
            probe = os.path.expanduser("~")

        stat = os.statvfs(probe)
        available_bytes = stat.f_bavail * stat.f_frsize

        # ~10% headroom for temp files during xet_get / move-into-place.
        required_bytes = int(model_size_bytes * 1.1)
        if available_bytes >= required_bytes:
            return

        model_size_gb = model_size_bytes / (1024**3)
        available_gb = available_bytes / (1024**3)
        need_to_free_gb = (required_bytes - available_bytes) / (1024**3)

        print()
        print("  Error: Insufficient disk space for download.")
        print(f"    Model size:    {model_size_gb:>7.1f} GB")
        print(f"    Free space:    {available_gb:>7.1f} GB  ({probe})")
        print(f"    Need to free:  {need_to_free_gb:>7.1f} GB")
        print()
        print("  Suggestions:")
        print("    - Free disk space, or set HF_HOME to a drive with more room")
        print("    - Pick a smaller variant: rapid-mlx models")
        if not force:
            print(
                "    - Bypass this check (download will likely fail mid-way): "
                "--force-disk-check"
            )
            print()
            sys.exit(1)
        # ``force=True``: warn loudly, let the user proceed at their own risk.
        print("  --force-disk-check set — proceeding anyway.")
        print()
    except SystemExit:
        raise
    except Exception:
        # Network / auth / etc. failures are non-critical — fall through to
        # the loader's own error handling rather than blocking startup on a
        # flaky HF metadata query.
        pass


def _check_memory_capacity(model_name: str) -> None:
    """Pre-flight memory check — warn loudly if loading this model is
    likely to push unified memory past the danger threshold.

    On low-memory Apple Silicon (especially Mac mini M4 24 GB), loading
    a model that forces unified memory past ~85% of total can trip the
    iBoot AMCC async-abort firmware path and **kernel-panic the entire
    machine** rather than raise a userspace OOM. See issue #324.

    This check is best-effort: it warns the user, never aborts. If we
    can't read the model size (offline / gated repo), or psutil isn't
    importable, fall through silently — the existing loader paths still
    surface real failures.

    Working-set estimate is ``model_size * 1.5`` for a typical short
    chat workload — covers KV cache, activations, and OS reserve.
    Long-context (32k+) or high-concurrency serving pushes the
    multiplier higher; the warning under-predicts in those modes
    rather than over-predicts, so a user who configures aggressively
    may still crash. We err on the side of warning earlier than later.

    **Pressure formula uses already-used memory** rather than just
    ``working / total``. The kernel panic fires on absolute unified-
    memory pressure, so a 10 GB model on a 24 GB Mac that already has
    8 GB used by macOS + Chrome lands at projected ``(8 + 15) / 24``
    = 95.8% — kernel-panic territory. The naive formula would have
    reported only 62.5% and stayed silent.
    """
    try:
        import psutil
    except Exception:
        return

    # Resolve model size in bytes — local path, then HF cache, then HF API.
    model_size_bytes = 0
    try:
        if os.path.isdir(model_name):
            for root, _dirs, files in os.walk(model_name):
                for f in files:
                    try:
                        model_size_bytes += os.path.getsize(os.path.join(root, f))
                    except OSError:
                        continue
        else:
            from huggingface_hub import model_info, try_to_load_from_cache

            cached = try_to_load_from_cache(model_name, "config.json")
            if isinstance(cached, str) and os.path.exists(cached):
                # Already-downloaded model: walk the snapshot directory.
                snapshot_dir = os.path.dirname(cached)
                for root, _dirs, files in os.walk(snapshot_dir):
                    for f in files:
                        try:
                            model_size_bytes += os.path.getsize(os.path.join(root, f))
                        except OSError:
                            continue
            else:
                info = model_info(model_name, files_metadata=True)
                model_size_bytes = sum(
                    (s.size or 0)
                    for s in (getattr(info, "siblings", None) or [])
                    if hasattr(s, "size")
                )
    except Exception:
        return  # Network / auth failure — fall through.

    if model_size_bytes <= 0:
        return

    try:
        vm = psutil.virtual_memory()
        total_ram_bytes = vm.total
        available_ram_bytes = vm.available
    except Exception:
        return

    if total_ram_bytes <= 0:
        return

    # Projected post-load pressure: already-used + estimated working set.
    # ``available`` is psutil's best estimate of "memory we can grab without
    # swapping," which on macOS includes inactive + cached pages that the
    # kernel will reclaim under pressure. ``total - available`` is therefore
    # a tighter "currently-pinned" floor than ``total - free``.
    estimated_working = int(model_size_bytes * 1.5)
    used_ram_bytes = max(0, total_ram_bytes - available_ram_bytes)
    projected_use = used_ram_bytes + estimated_working
    ratio = projected_use / total_ram_bytes
    if ratio < 0.65:
        return  # Comfortable headroom — no warning.

    model_gb = model_size_bytes / (1024**3)
    working_gb = estimated_working / (1024**3)
    used_gb = used_ram_bytes / (1024**3)
    total_gb = total_ram_bytes / (1024**3)

    is_tty = sys.stdout.isatty() and "NO_COLOR" not in os.environ
    YELLOW = "\x1b[33m" if is_tty else ""
    RED = "\x1b[31m" if is_tty else ""
    BOLD = "\x1b[1m" if is_tty else ""
    DIM = "\x1b[2m" if is_tty else ""
    RESET = "\x1b[0m" if is_tty else ""

    print()
    if ratio >= 0.85:
        print(
            f"  {RED}{BOLD}!! Memory pressure warning:{RESET} "
            f"this model is likely too large for your hardware."
        )
        print(
            f"  {DIM}Continuing may trigger a macOS kernel panic "
            f"(see issue #324).{RESET}"
        )
    else:
        print(
            f"  {YELLOW}{BOLD}Memory pressure note:{RESET} "
            f"this model uses a large fraction of system RAM."
        )
    print()
    print(f"    Model on disk:           {model_gb:>6.1f} GB")
    print(
        f"    Est. working set:        {working_gb:>6.1f} GB  "
        f"{DIM}(model x 1.5 — short-chat workload; long-context serving will use more){RESET}"
    )
    print(f"    Currently used by OS:    {used_gb:>6.1f} GB")
    print(
        f"    Total system RAM:        {total_gb:>6.1f} GB  "
        f"({ratio * 100:.0f}% projected utilization)"
    )
    print()
    if ratio >= 0.85:
        print("  Apple Silicon firmware can panic the whole system rather than")
        print("  raise an OOM error when unified-memory pressure exceeds the")
        print("  iBoot AMCC threshold. Recommended actions:")
        print()
        print("    - Close other apps to free RAM, or")
        print("    - Pick a smaller model:    rapid-mlx models")
        print(
            "    - Or lower memory headroom: "
            "rapid-mlx serve <model> --gpu-memory-utilization 0.75"
        )
    else:
        print(
            "  If you see crashes or kernel panics, try: --gpu-memory-utilization 0.85"
        )
    print()


def _try_mirror_prefetch(model_name: str) -> bool:
    """Pre-fetch a HuggingFace repo via R2-first / HF-fallback (per file).

    Delegates to :func:`vllm_mlx._mirror.download_with_mirror_fallback`.
    Returns ``True`` if the snapshot is fully populated (any mix of R2
    and HF). Returns ``False`` if the caller should fall through to the
    plain ``snapshot_download(repo_id)`` path (catalog unavailable for
    catalog-only paths, or one or more files failed both R2 and HF).

    Set ``RAPID_MLX_MODEL_MIRROR=""`` to disable R2 entirely and force
    HuggingFace.

    Codex round-6 BLOCKING #2: the mirror module already returns
    ``False`` on every recoverable network/cache error, so the only
    catch worth doing here is ``ImportError`` (mirror module disabled
    or missing in a minimal install). Programmer errors propagate so
    bugs in the mirror module surface as real stack traces instead of
    silently routing to ``snapshot_download``.
    """
    try:
        from vllm_mlx._mirror import download_with_mirror_fallback
    except ImportError:
        # Mirror module not available (minimal-deps install or
        # deliberately removed). Use the legacy HF path.
        return False
    return download_with_mirror_fallback(model_name)


def _ensure_model_downloaded(model_name: str) -> None:
    """Pre-fetch a model in the foreground so HF's tqdm progress is visible.

    Used by ``rapid-mlx chat``: the chat REPL spawns ``serve`` as a
    subprocess with stdout/stderr redirected to a log file. If the model
    isn't cached, the user sees a silent multi-minute hang while several
    GB downloads behind the log. Calling ``snapshot_download`` here first
    surfaces the standard HF progress bars on the user's terminal, then
    the spawned server starts as a cache hit.

    No-op when the model is already cached, when ``model_name`` is a local
    path, or when the HF lookup fails (let the loader's own error paths
    handle it).
    """
    if os.path.exists(model_name):
        return
    # Reuse the same weight-file-presence probe as ``is_repo_cached``:
    # the older ``try_to_load_from_cache('config.json')`` check
    # short-circuits on a partial cache (metadata downloaded, weight
    # shards still in flight), letting the spawned ``serve`` quietly
    # finish the download inside its logfile. Codex round-3 BLOCKING #2.
    try:
        from vllm_mlx._download_gate import is_repo_cached

        if is_repo_cached(model_name):
            return
    except Exception:
        # Probe failed (filesystem permission error, unexpected layout) —
        # fall through to the heavy snapshot_download path; HF will
        # short-circuit on its own cache check if the repo really is
        # fully present.
        pass

    # Disk-space gate: a 20 GB partial download that fails on the last
    # shard wastes the user's time. ``_check_disk_space`` queries HF for
    # the repo size and aborts with a clear message + exit(1) if there
    # isn't enough room on the resolved HF cache filesystem.
    _check_disk_space(model_name)

    # User-configured mirror path (R2/S3/any HTTP host). When the mirror
    # serves every file the repo declares, populate the HF cache layout
    # ourselves and skip snapshot_download. On any miss we fall through
    # to the normal HuggingFace download below.
    if _try_mirror_prefetch(model_name):
        return

    try:
        from huggingface_hub import model_info, snapshot_download

        size_gb = 0.0
        try:
            info = model_info(model_name, files_metadata=True)
            size_bytes = sum(
                (s.size or 0)
                for s in (getattr(info, "siblings", None) or [])
                if hasattr(s, "size")
            )
            size_gb = size_bytes / (1024**3)
        except Exception:
            pass

        is_tty = sys.stdout.isatty() and "NO_COLOR" not in os.environ
        BOLD = "\x1b[1m" if is_tty else ""
        DIM = "\x1b[2m" if is_tty else ""
        RESET = "\x1b[0m" if is_tty else ""
        if size_gb > 0:
            print(
                f"\n  {BOLD}First-time download{RESET} — "
                f"fetching {model_name} {DIM}(~{size_gb:.1f} GB){RESET} "
                "from HuggingFace ..."
            )
        else:
            print(
                f"\n  {BOLD}First-time download{RESET} — "
                f"fetching {model_name} from HuggingFace ..."
            )

        snapshot_download(model_name)
        print()
    except SystemExit:
        # _check_disk_space aborts via sys.exit(1) — let it through.
        raise
    except Exception as e:
        # Definitive 404s are surfaced so callers (e.g. ``/model bogus``)
        # can refuse fast instead of spawning a doomed serve subprocess
        # that fails after ``--ready-timeout``. Other transient errors
        # (network, auth) fall through silently — the spawned server's
        # own loader will retry and surface a real error if needed.
        from huggingface_hub.utils import RepositoryNotFoundError

        if isinstance(e, RepositoryNotFoundError) or "404" in str(e):
            raise RuntimeError(f"Model {model_name!r} not found on HuggingFace") from e
        print(f"\n  Pre-download skipped ({type(e).__name__}); server will retry.")


def _add_pflash_args(parser) -> None:
    """Attach PFlash long-prompt-compression CLI flags to an argparse parser.

    Used by both ``serve`` and ``bench`` so the flag surface stays in
    sync. The default for ``--pflash`` is intentionally ``None``
    (sentinel for "user passed nothing") so the per-alias resolver in
    ``pflash.resolve_pflash_mode_default`` can switch the engine to
    ``always`` for ``pflash_tier="verified"`` aliases (Qwen3.5 /
    Qwen3.6 family per #287) without breaking the explicit-override
    contract: passing ``--pflash off`` still wins.
    """
    parser.add_argument(
        "--pflash",
        choices=["off", "auto", "always"],
        default=None,
        help="Enable PFlash long-prompt prefill compression "
        "(off, auto, always). Default: 'always' for verified aliases "
        "(Qwen3.5 / Qwen3.6 family per #287), 'off' for everything else.",
    )
    parser.add_argument(
        "--pflash-threshold",
        type=int,
        default=32_768,
        help="Minimum prompt tokens before --pflash auto compresses (default: 32768).",
    )
    parser.add_argument(
        "--pflash-keep-ratio",
        type=float,
        default=0.20,
        help="Fraction of prompt tokens to keep when compressing "
        "(default: 0.20 — matches the bench-validated profile in PR #649: "
        "TTFT 3.87x-8.5x, needle recall 5/5 across tested cells).",
    )
    parser.add_argument(
        "--pflash-min-keep-tokens",
        type=int,
        default=2_048,
        help="Minimum tokens to keep when compressing (default: 2048).",
    )
    parser.add_argument(
        "--pflash-sink-tokens",
        type=int,
        default=256,
        help="Leading prompt tokens always kept by PFlash (default: 256).",
    )
    parser.add_argument(
        "--pflash-tail-tokens",
        type=int,
        default=2_048,
        help="Trailing prompt tokens always kept by PFlash (default: 2048).",
    )
    parser.add_argument(
        "--pflash-block-size",
        type=int,
        default=128,
        help="Middle-token scoring block size (default: 128).",
    )
    parser.add_argument(
        "--pflash-query-window",
        type=int,
        default=512,
        help="Trailing query window used to score middle blocks (default: 512).",
    )
    parser.add_argument(
        "--pflash-stride-blocks",
        type=int,
        default=8,
        help="Keep every Nth middle block as an anchor during scoring "
        "(0 disables anchors, default: 8).",
    )
    parser.add_argument(
        "--pflash-include-tools",
        action="store_true",
        help="Allow PFlash compression on prompts with tool definitions. "
        "By default tool prompts are skipped for tool-call reliability.",
    )


def _build_benchmark_context(target_tokens: int) -> str:
    """Build a deterministic long-context filler for the bench command.

    Used by ``--long-prompt-tokens`` to construct repeatable long
    prompts for TTFT replication runs without depending on a real
    long-context corpus. The block is intentionally generic so the
    measurement targets prefill cost, not semantic difficulty.
    """
    if target_tokens <= 0:
        return ""
    block = (
        "Reference context for long prompt benchmarking. "
        "Rapid MLX evaluates prompt prefill latency, prefix cache behavior, "
        "tool instructions, JSON schema preservation, and model output quality. "
        "The assistant must preserve system instructions and answer only the "
        "final user request after reviewing all reference material. "
    )
    approx_block_tokens = max(1, len(block.split()))
    repeats = max(1, target_tokens // approx_block_tokens)
    return (block * repeats).strip()


def serve_command(args):
    """Start the OpenAI-compatible server."""
    import logging
    import os
    import sys

    _arg_max_tokens = getattr(args, "max_tokens", None)
    _max_tokens_is_explicit = _arg_max_tokens is not None
    effective_max_tokens = _arg_max_tokens if _arg_max_tokens is not None else 32768

    # F-H08-INCOMPLETE: the ``[embeddings]`` extra-required guard MUST
    # fire first thing in ``serve_command`` — before
    # ``prompt_upgrade_if_available`` (which may exit 0 on user
    # decline), before ``_ensure_model_downloaded`` (which can take
    # minutes on a cold cache), and well before the startup banner
    # gets printed. Pre-fix the check lived deeper in the function so
    # the operator saw the alias-resolved log line, the startup banner,
    # the feature list, AND the model id BEFORE the
    # "requires the [embeddings] extra" error and ``sys.exit(2)``,
    # which read as a successful boot followed by a mysterious failure
    # — Diego logged this as a warning-and-fall-through bug because
    # the banner masked the actual exit. Hoisting the probe to the
    # very top of ``serve_command`` puts the error first, with no
    # banner output before it. ``mlx_embeddings`` import stays lazy so
    # the base install (no ``[embeddings]`` extra) keeps booting.
    if getattr(args, "embedding_model", None):
        from .embedding import require_mlx_embeddings_or_exit

        require_mlx_embeddings_or_exit()

    # R-10 (PyPI 0.8.6 dogfood): same boot-guard shape for vision /
    # multimodal aliases. ``mlx-vlm`` lives behind the ``[vision]``
    # extra, but ``rapid-mlx serve ui-tars-1.5-7b-4bit`` on a fresh
    # ``pip install rapid-mlx`` previously fell into the engine load
    # path BEFORE the missing-dep error surfaced (deep ImportError
    # after weight download + alias resolution). Probe here so the
    # operator sees an actionable hint before the long download starts.
    # Honors ``--mllm`` force-on AND the alias-name / HF-path probe
    # used downstream by ``pflash.validate_model_support``. The
    # ``--no-mllm`` escape hatch (force_text in load_model) bypasses
    # this guard, matching the engine-side semantics.
    if not getattr(args, "no_mllm", False):
        from .api.utils import is_mllm_model as _boot_is_mllm

        if getattr(args, "mllm", False) or _boot_is_mllm(args.model):
            from .models.mllm import require_mlx_vlm_or_exit

            require_mlx_vlm_or_exit(args.model)

    # R6-H4 (Eva 0.8.7 dogfood): same boot-guard shape for audio aliases.
    # ``mlx-audio`` lives behind the ``[audio]`` extra; pre-fix
    # ``rapid-mlx serve kokoro`` (or whisper/parakeet/chatterbox/...) on
    # a base install printed the startup banner, opened the port, and
    # only crashed on the first audio request (the in-route lane probe).
    # That looked like "successful boot, broken inference" instead of
    # the obvious "you need the [audio] extra". Probe at flag-parse
    # time so the operator sees an actionable hint with rc=2 before
    # any download / banner output, mirroring r5-C's UI-TARS guard.
    #
    # Recognition is alias-substring based (``whisper``, ``parakeet``,
    # ``kokoro``, ``chatterbox``, ``vibevoice``, ``voxcpm``) so the
    # quantised variants (``kokoro-4bit``) and HF-style ids
    # (``mlx-community/Kokoro-82M-bf16``) trip it the same way bare
    # aliases do. A model name that doesn't match an audio token falls
    # through unchanged — text/vision/embedding models never see this
    # probe.
    from .audio.probe import is_audio_model_alias, require_audio_or_exit

    if is_audio_model_alias(getattr(args, "model", None)):
        require_audio_or_exit(args.model)

    # R10-C1: AUDIO-SERVE-MODE FORK. The boot guard above only checks
    # that the ``[audio]`` extra is installed — it doesn't route the
    # alias anywhere. Pre-R10 every short alias (``kokoro``, ``whisper``,
    # ``parakeet``...) fell through to ``_ensure_model_downloaded``
    # and 404'd at HF, while full HF ids of audio models downloaded
    # successfully but then crashed inside ``mlx_lm.load_model``
    # because audio repos don't ship safetensors. Bo r10-R1: 0/8 audio
    # aliases boot on 0.8.11 (codex r8-A r3 predicted this exact shape).
    #
    # The fix is a clean fork: if the registry resolves the model to
    # an audio entry, route to ``_serve_audio_mode`` (which skips
    # ``_ensure_model_downloaded``, the text loader, pflash, parser
    # detection, etc.) and return. Everything below this block remains
    # untouched for the text path so text-model boot does NOT regress.
    audio_entry = _resolve_audio_model_for_serve(getattr(args, "model", None))
    if audio_entry is not None:
        # Stamp the alias hop so /v1/models, telemetry, and the banner
        # all show the same name pair. ``_original_alias`` is set by
        # the main() alias resolver for text models; we mirror that
        # contract here for audio.
        if not hasattr(args, "_original_alias") or args._original_alias is None:
            args._original_alias = args.model
        # Replace the alias on args.model with the resolved HF id so
        # any downstream code that reads ``args.model`` (eg. session
        # telemetry, ps_command) sees a real repo path. The audio
        # routes still accept both forms because the registry's
        # reverse HF-id index covers full ids too.
        args.model = audio_entry.hf_id
        _serve_audio_mode(args, audio_entry)
        return

    # Interactive auto-upgrade prompt — when serve runs interactively and a
    # newer release is available, ask once before booting the model. Honors
    # RAPID_MLX_DISABLE_VERSION_CHECK, CI=1, and non-TTY stdin. Cached
    # piggy-backs on the existing staleness check's cache (24h TTL).
    from vllm_mlx._version_check import prompt_upgrade_if_available

    if prompt_upgrade_if_available():
        sys.exit(0)

    # Pre-fetch the model via the R2 mirror (with HF fallback) BEFORE the
    # heavy server boot. Without this, ``serve`` falls into
    # ``mlx_lm.load`` → ``huggingface_hub.snapshot_download`` directly and
    # skips the mirror entirely (#651). ``_ensure_model_downloaded`` is a
    # no-op on local paths and on fully-cached repos, so this is free on
    # the warm path.
    _ensure_model_downloaded(args.model)

    # Import unified server
    from . import server
    from .middleware.auth import configure_rate_limiter
    from .scheduler import SchedulerConfig
    from .server import app, load_model

    logger = logging.getLogger(__name__)
    uvicorn_log_level = server.configure_logging(args.log_level)

    # Validate tool calling arguments
    if args.enable_auto_tool_choice and not args.tool_call_parser:
        print("Error: --enable-auto-tool-choice requires --tool-call-parser")
        print("Example: --enable-auto-tool-choice --tool-call-parser mistral")
        sys.exit(1)

    # Validate --tool-call-parser against the live registry (not the
    # stale argparse choices list). v0.6.63 onboarding sweep finding #1.
    if args.tool_call_parser:
        # Narrow the catch: only swallow import-time / attribute access
        # failures (broken install, missing module file). Anything else
        # — a corrupt registry that's loaded but malformed, a TypeError
        # from a buggy parser's __init_subclass__, etc. — is a real bug
        # we want to surface, not paper over with "validation skipped".
        # Codex follow-up to PR #433.
        valid: list[str] | None = None
        try:
            from .tool_parsers import ToolParserManager

            valid = sorted(ToolParserManager.tool_parsers.keys())
        except (ImportError, AttributeError) as e:
            print(
                "warning: --tool-call-parser validation skipped — "
                f"tool_parsers registry unavailable ({type(e).__name__}: {e}). "
                "Proceeding without input check.",
                file=sys.stderr,
            )
        # Treat an empty registry (degenerate install) the same as a
        # failed import — skip validation rather than reject every input.
        # Without this guard, a successful import with zero registered
        # parsers would hard-fail every CLI invocation; DeepSeek
        # follow-up to PR #434.
        if valid and args.tool_call_parser not in valid:
            print(
                f"error: argument --tool-call-parser: invalid choice: "
                f"{args.tool_call_parser!r} "
                f"(choose from: {', '.join(valid)})",
                file=sys.stderr,
            )
            sys.exit(2)

    # Validate gpu-memory-utilization range
    if not (0.0 < args.gpu_memory_utilization <= 1.0):
        print(
            "Error: --gpu-memory-utilization must be between 0.0 (exclusive) and 1.0 (inclusive)"
        )
        sys.exit(1)

    # Validate PFlash config and reject unsupported model combinations
    # at startup. Done here (not lazily in the scheduler) so a typo in
    # --pflash-keep-ratio doesn't surface as a model-load failure
    # after a multi-minute weight download. See #287.
    #
    # ``resolve_pflash_mode_default`` runs before ``config_from_args``
    # so the per-alias default (``"always"`` for verified Qwen3.5 /
    # Qwen3.6 aliases, ``"off"`` everywhere else) is materialized into
    # ``args.pflash``. The resolved value then flows through the same
    # validation path the user-explicit case takes.
    from .api.utils import is_mllm_model
    from .pflash import (
        config_from_args,
        resolve_pflash_mode_default,
        validate_model_support,
    )

    args.pflash = resolve_pflash_mode_default(args, model_name=args.model)
    try:
        pflash_config = config_from_args(args)
        validate_model_support(
            pflash_config,
            model_name=args.model,
            is_mllm=getattr(args, "mllm", False) or is_mllm_model(args.model),
        )
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Auto-detect parser config from model name when not explicitly set.
    # --no-tool-call-parser / --no-reasoning-parser are escape hatches
    # (SOP §10): if the user opts out, do NOT let the AliasProfile auto-
    # populate args.tool_call_parser / args.reasoning_parser. Past
    # incidents: #393-class (auto-detect false positive with no opt-out).
    _opt_out_tool = getattr(args, "no_tool_call_parser", False)
    _opt_out_reasoning = getattr(args, "no_reasoning_parser", False)
    if args.tool_call_parser and _opt_out_tool:
        print(
            "error: --tool-call-parser and --no-tool-call-parser are "
            "mutually exclusive — pick one to override auto-detection.",
            file=sys.stderr,
        )
        sys.exit(2)
    if args.reasoning_parser and _opt_out_reasoning:
        print(
            "error: --reasoning-parser and --no-reasoning-parser are "
            "mutually exclusive — pick one to override auto-detection.",
            file=sys.stderr,
        )
        sys.exit(2)
    # R12-S1: snapshot whether the user explicitly passed
    # ``--tool-call-parser`` BEFORE auto-detect mutates ``args``. The
    # misbind warning below only consults this snapshot — auto-detected
    # bindings are guaranteed to match the model family by construction
    # (the auto path picks the same parser the helper would suggest), so
    # warning on them would be a contradictory "drop the flag" nudge
    # against a flag the user never passed. (Codex r4 NIT — keeps the
    # warning grounded in user intent even if a helper-side regression
    # ever started flagging in-spec cases.)
    _user_explicit_tool_call_parser = bool(args.tool_call_parser)
    if not args.tool_call_parser or not args.reasoning_parser:
        try:
            from .model_auto_config import detect_model_config

            auto_config = detect_model_config(args.model)
            if auto_config:
                if (
                    not args.tool_call_parser
                    and not _opt_out_tool
                    and auto_config.tool_call_parser
                ):
                    args.tool_call_parser = auto_config.tool_call_parser
                    args.enable_auto_tool_choice = True
                    logger.info(
                        f"Auto-configured --tool-call-parser {auto_config.tool_call_parser}"
                    )
                if (
                    not args.reasoning_parser
                    and not _opt_out_reasoning
                    and not args.no_thinking
                    and auto_config.reasoning_parser
                ):
                    args.reasoning_parser = auto_config.reasoning_parser
                    logger.info(
                        f"Auto-configured --reasoning-parser {auto_config.reasoning_parser}"
                    )
        except Exception as e:
            logger.debug(f"Auto-detection failed (non-fatal): {e}")
    if _opt_out_tool:
        logger.info(
            "Tool-call parser auto-detection disabled via --no-tool-call-parser"
        )
    if _opt_out_reasoning:
        logger.info(
            "Reasoning parser auto-detection disabled via --no-reasoning-parser"
        )

    # R12-S1: surface a startup warning when ``args.tool_call_parser``
    # is a ``deepseek_v3`` / ``deepseek_v31`` / ``deepseek_r1_0528``
    # binding but the model can't emit the matching V3 fenced-JSON wire
    # shape. See Sven r12 dogfood HIGH-1: forcing ``--tool-call-parser
    # deepseek_v3`` on ``DeepSeek-R1-Distill-Qwen-1.5B-4bit`` lands tool
    # calls with ``arguments="{}"`` because the parser correctly refuses
    # to parse the non-V3 prose the model emits.
    #
    # Runs on BOTH explicit overrides AND auto-detected bindings,
    # because ``detect_model_config`` still scans the full path — a
    # parent dir like ``/models/DeepSeek-V3/qwen-model`` can fool
    # auto-detect into ``deepseek_v3`` even though the tail model name
    # is out-of-lineage (pr-validate codex r7 BLOCKING). The helper's
    # canonical model-name classification catches that mis-pick that
    # auto-detect missed. Whether the user explicitly bound the flag is
    # tracked in ``_user_explicit_tool_call_parser`` and threaded into
    # the warning so the operator can tell who to blame: a misbound
    # flag (user error) vs. a fooled auto-detect (regex needs
    # tightening — tracked as follow-up so this PR stays scoped).
    try:
        from .model_auto_config import warn_misbound_deepseek_v3_parser

        misbind_warning = warn_misbound_deepseek_v3_parser(
            args.model, args.tool_call_parser
        )
        if misbind_warning:
            # ``logger.warning`` so the message lands in any structured
            # log sink AND surfaces in the terminal at the default
            # ``WARNING`` level (no stderr-print needed).
            # ``stacklevel=2`` so log frameworks attribute the call
            # site to the CLI entry rather than the helper module.
            logger.warning(misbind_warning, stacklevel=2)
            if not _user_explicit_tool_call_parser:
                # Auto-detect mis-pick. Emit a second WARNING line so
                # the operator knows the user didn't bind anything —
                # the ``detect_model_config`` regex was fooled by the
                # path. Forces the user to add an explicit
                # ``--tool-call-parser hermes`` (or similar) to recover
                # tool-call capability, which is the actually-correct
                # action for an out-of-lineage checkpoint.
                logger.warning(
                    "  Auto-detect note: this binding came from "
                    "AUTO-DETECT, not an explicit --tool-call-parser "
                    "flag. The detect_model_config() regex was fooled "
                    "by the path. Override with --tool-call-parser "
                    "hermes (or whatever your checkpoint actually "
                    "emits) to recover tool-call capability."
                )
    except Exception as e:  # noqa: BLE001
        logger.debug(f"deepseek_v3 misbind check failed (non-fatal): {e}")

    # Pass alias info to server (for /v1/models)
    server._model_alias = getattr(args, "_original_alias", None)

    # Configure server security settings. ``RAPID_MLX_API_KEY`` env var
    # is the secret-friendly form ``rapid-mlx share`` uses to avoid
    # exposing the key in argv; inline ``--api-key`` overrides it for
    # backwards-compat with existing scripts. ``_resolve_api_key`` is
    # the single SSOT — both this entrypoint and the ``vllm_mlx.server``
    # ``python -m`` entry call into it, so a future policy tweak (e.g.
    # a deprecation warning when argv is used) lands in one place.
    server._api_key = server._resolve_api_key(args.api_key)
    server._default_timeout = args.timeout

    # Per-request body-size cap. Resolution order:
    #   1. ``--max-request-bytes`` (explicit CLI flag, including 0 to disable)
    #   2. ``RAPID_MLX_MAX_REQUEST_BYTES`` env var
    #   3. ``ServerConfig`` dataclass default (8 MiB)
    # See vllm_mlx/middleware/body_size.py for the DoS rationale.
    _max_body_arg = getattr(args, "max_request_bytes", None)
    if _max_body_arg is not None:
        server._max_request_bytes = max(0, int(_max_body_arg))
    else:
        _env_name = "RAPID_MLX_MAX_REQUEST_BYTES"
        _env = os.environ.get(_env_name, "").strip()
        if _env:
            try:
                server._max_request_bytes = max(0, int(_env))
            except ValueError:
                # Explicit reset (codex round-2 NIT): without this,
                # an in-process callsite that mutated ``_max_request_bytes``
                # before serve_command runs would silently leak a stale
                # value past a malformed env var, which is the worst
                # possible failure shape — bigger cap than the operator
                # intended. Fall back to the documented 8 MiB default
                # explicitly.
                server._max_request_bytes = 8 * 1024 * 1024
                logger.warning(
                    "%s=%r is not an integer; falling back to the 8 MiB default",
                    _env_name,
                    _env,
                )

    # SSE keepalive interval (F-070). Env-only (no CLI flag yet — keep
    # the surface small until operators ask for it). 0 disables. The
    # value lands on ``server._sse_keepalive_seconds`` so ``_sync_config``
    # propagates it into the live ``ServerConfig`` after ``load_model``;
    # writing the config singleton directly here would be clobbered by
    # the subsequent ``_sync_config`` (mirrors the ``_max_request_bytes``
    # pattern just above).
    _sse_env_name = "RAPID_MLX_SSE_KEEPALIVE_SECONDS"
    _sse_env = os.environ.get(_sse_env_name, "").strip()
    if _sse_env:
        try:
            server._sse_keepalive_seconds = max(0.0, float(_sse_env))
        except ValueError:
            # NOTE: the env-var name is interpolated via ``%s`` (not baked
            # into the format string) so the
            # ``tests/test_no_out_of_band_routing.py`` constant scan
            # doesn't see the literal ``RAPID_MLX_…=%r is not a number``
            # as a stand-alone string and false-positive on a routing
            # match. Same pattern the body-receive timeout block below
            # uses.
            logger.warning(
                "%s=%r is not a number; falling back to the 20 s default",
                _sse_env_name,
                _sse_env,
            )
            server._sse_keepalive_seconds = 20.0

    # Body-receive idle timeout (F-072 / H-14 slow-DoS gate). Env-only.
    # 0 disables. Same ``_sync_config``-then-route-handler ordering
    # rationale as the SSE keepalive above. Extracted into
    # :func:`_apply_body_receive_timeout_env` so tests can call the
    # SAME resolver the production binary uses — codex round-2 BLOCKING
    # on PR #786 flagged that an inline-only resolver couldn't be
    # exercised by a unit test without duplicating its logic, which
    # would mask a regression that deleted the wire-up entirely.
    _apply_body_receive_timeout_env(server, logger=logger)

    # Configure CORS (F-090 + F-091). Default: wildcard ``*`` for friendly
    # single-machine UX — rapid-mlx is primarily run locally and a
    # browser frontend at ``http://localhost:3000`` hitting the API at
    # ``http://localhost:8000`` "just works". Operators on multi-tenant /
    # production deployments lock down via
    # ``RAPID_MLX_CORS_ALLOW_ORIGINS=https://your.app,https://other.app``.
    # The full env-var family (METHODS / HEADERS / MAX_AGE /
    # ALLOW_CREDENTIALS) still applies; see
    # ``vllm_mlx/server.py::configure_cors_from_env``.
    #
    # Wildcard + credentials is spec-invalid (Fetch spec rejects the
    # combination), so the resolver forces ``allow_credentials=False``
    # when ``*`` is in the origin list. Operators who need cookie /
    # ``Authorization`` auto-forwarding must pin to specific origins.
    cors_origins = server.configure_cors_from_env(args.cors_origins)
    if args.rate_limit > 0:
        server._rate_limiter = configure_rate_limiter(args.rate_limit, enabled=True)

    # Configure GC control
    gc_control = args.gc_control and not args.no_gc_control
    server._gc_control = gc_control

    # Configure --no-thinking: suppress chain-of-thought in chat template
    server._no_thinking = args.no_thinking

    # Configure system prompt pinning
    server._pin_system_prompt = args.pin_system_prompt

    # Configure tool calling
    if args.enable_auto_tool_choice and args.tool_call_parser:
        server._enable_auto_tool_choice = True
        server._tool_call_parser = args.tool_call_parser
        server._enable_tool_logits_bias = getattr(
            args, "enable_tool_logits_bias", False
        )
    else:
        server._enable_auto_tool_choice = False
        server._tool_call_parser = None
        server._enable_tool_logits_bias = False

    # Configure generation defaults
    if args.default_temperature is not None:
        server._default_temperature = args.default_temperature
    if args.default_top_p is not None:
        server._default_top_p = args.default_top_p
    if args.default_top_k is not None:
        server._default_top_k = args.default_top_k
    if args.default_min_p is not None:
        server._default_min_p = args.default_min_p
    if args.default_repetition_penalty is not None:
        server._default_repetition_penalty = args.default_repetition_penalty
    if args.default_presence_penalty is not None:
        server._default_presence_penalty = args.default_presence_penalty
    if args.default_frequency_penalty is not None:
        server._default_frequency_penalty = args.default_frequency_penalty

    # Configure reasoning parser
    if args.reasoning_parser:
        try:
            from .reasoning import get_parser

            parser_cls = get_parser(args.reasoning_parser)
            server._reasoning_parser = parser_cls()
            server._reasoning_parser_name = args.reasoning_parser
            logger.info(f"Reasoning parser enabled: {args.reasoning_parser}")
        except KeyError as e:
            print(f"Error: {e}")
            sys.exit(1)
        except ImportError as e:
            print(f"Error: Failed to import reasoning module: {e}")
            sys.exit(1)
        except Exception as e:
            print(
                f"Error: Failed to initialize reasoning parser "
                f"'{args.reasoning_parser}': {e}"
            )
            sys.exit(1)
    else:
        server._reasoning_parser = None

    # DFlash mutual-exclusion gate fires BEFORE the startup banner so
    # the user sees a clean error instead of an optimistic "Features:
    # dflash" line immediately followed by an exit. The deeper SchedulerConfig
    # mutex (suffix vs. mtp) stays below since it doesn't involve DFlash.
    if args.enable_dflash and (args.suffix_decoding or args.enable_mtp):
        print(
            "\n  Error: --enable-dflash cannot combine with --suffix-decoding "
            "or --enable-mtp. DFlash runs a dedicated single-user server "
            "that bypasses BatchedEngine; other spec-decode methods only "
            "apply to the BatchedEngine path.\n"
        )
        sys.exit(1)

    # DFlash eligibility gate fires here, BEFORE the startup banner —
    # so the user sees a clean error rather than an optimistic "DFlash
    # enabled" feature line followed by an exit. Cheap (just reads
    # aliases.json + checks the module spec); no model load yet.
    if args.enable_dflash:
        from .model_aliases import resolve_profile
        from .speculative.dflash import DFlashUnavailable, check
        from .speculative.dflash.eligibility import have_runtime

        _alias_name = getattr(args, "_original_alias", None) or args.model
        _profile = resolve_profile(_alias_name)
        if _profile is None:
            print(
                f"\n  Error: --enable-dflash requires a known alias, got "
                f"{_alias_name!r}. DFlash eligibility is recorded per-alias "
                f"in aliases.json; ad-hoc HuggingFace paths can't be "
                f"validated. Try ``rapid-mlx info qwen3.5-27b-8bit``.\n"
            )
            sys.exit(1)
        try:
            check(_profile, alias=_alias_name)
        except DFlashUnavailable as e:
            print(f"\n  Error: {e}\n")
            sys.exit(1)
        if not have_runtime():
            print(
                "\n  Error: --enable-dflash requires mlx-vlm 0.5.0+ for the "
                "DFlash drafter hooks. Install with: "
                "``pip install 'rapid-mlx[dflash]'``.\n"
            )
            sys.exit(1)

        # Warn about flags that BatchedEngine honours but the DFlash
        # server doesn't — better to surface this once at startup than
        # to let users wonder why their tuning has no effect. Inspected
        # against the actual argparse Namespace so we only mention flags
        # the user explicitly set away from their default.
        _GPU_MEM_DEFAULT = 0.90  # keep in sync with the serve_parser default
        _dflash_ignored: list[str] = []
        if getattr(args, "enable_prefix_cache", False):
            _dflash_ignored.append("--enable-prefix-cache")
        if getattr(args, "kv_cache_quantization", None):
            _dflash_ignored.append("--kv-cache-quantization")
        # gpu-memory-utilization defaults to 0.90 (not None) in the serve
        # parser, so an ``is not None`` check would fire on every invocation.
        # Compare to the real default — only warn when the user explicitly
        # tuned it. Tolerate a tiny float-equality slack for safety.
        _gpu_mem = getattr(args, "gpu_memory_utilization", _GPU_MEM_DEFAULT)
        if _gpu_mem is not None and abs(_gpu_mem - _GPU_MEM_DEFAULT) > 1e-6:
            _dflash_ignored.append("--gpu-memory-utilization")
        if getattr(args, "enable_auto_tool_choice", False):
            _dflash_ignored.append("--enable-auto-tool-choice")
        if getattr(args, "tool_call_parser", None):
            _dflash_ignored.append("--tool-call-parser")
        if getattr(args, "reasoning_parser", None):
            _dflash_ignored.append("--reasoning-parser")
        if getattr(args, "embedding_model", None):
            _dflash_ignored.append("--embedding-model")
        if getattr(args, "mcp_config", None):
            _dflash_ignored.append("--mcp-config")
        if _dflash_ignored:
            print(
                "\n  ⚠ The following flags are ignored under --enable-dflash"
                "\n    (DFlash uses a dedicated single-user server that bypasses"
                "\n    BatchedEngine):"
                f"\n      {', '.join(_dflash_ignored)}"
                "\n    Drop them from your serve command, or run without"
                "\n    --enable-dflash if you need them.\n"
            )

    # Startup summary
    print()
    print("  🐆 Rapid-MLX")
    print("  ─────────")
    features = []
    if args.enable_auto_tool_choice:
        bias_info = (
            " + logits bias" if getattr(args, "enable_tool_logits_bias", False) else ""
        )
        features.append(f"tools: {args.tool_call_parser}{bias_info}")
    if args.reasoning_parser:
        features.append(f"reasoning: {args.reasoning_parser}")
    # Banner mirrors the effective auth state via ``_auth_feature_str``
    # so the test can call the same function. Pre-fix the gate said
    # ``if args.api_key`` directly — a sidecar that set env-only saw
    # ``auth: off`` printed even though ``verify_api_key`` was
    # enforcing. ``_auth_feature_str`` keeps the banner and the actual
    # enforcement aligned and is directly unit-testable.
    auth_feature = _auth_feature_str(args.api_key)
    if auth_feature:
        features.append(auth_feature)
    if args.rate_limit > 0:
        features.append(f"rate-limit: {args.rate_limit}/min")
    if args.cloud_model:
        features.append(f"cloud: {args.cloud_model}")
    if gc_control:
        features.append("gc-control")
    if args.pin_system_prompt:
        features.append("pin-system-prompt")
    # Show CORS in the startup banner when CLI flag or env-var-driven
    # config produced an origin list (``configure_cors_from_env`` is what
    # actually resolved it — see the call site earlier in this function).
    if cors_origins:
        features.append(f"cors: {', '.join(cors_origins)}")
    if args.enable_dflash:
        features.append("dflash: single-user")
    if features:
        print(f"  Features: {', '.join(features)}")
    print(f"  Model: {args.model}")
    # Store MCP config path for FastAPI startup
    if args.mcp_config:
        print(f"MCP config: {args.mcp_config}")
        os.environ["RAPID_MLX_MCP_CONFIG"] = args.mcp_config

    # Pre-load embedding model if specified.
    #
    # H-08 install guard + D-EMBED-ALIAS alias-resolution + clean
    # ModelNotFoundError wrapping all live in the shared helper so the
    # standalone ``python -m vllm_mlx.server`` entry behaves identically.
    # See :func:`_load_embedding_model_or_exit` for the full contract;
    # F-H08-INCOMPLETE / D-CAPABILITIES already pre-flighted
    # ``require_mlx_embeddings_or_exit`` at the top of ``serve_command``
    # but the helper re-probes defensively so any caller that
    # synthesizes an ``args`` namespace and jumps into the load path
    # still gets the install-hint exit instead of a raw
    # ``ModuleNotFoundError``.
    if args.embedding_model:
        _load_embedding_model_or_exit(args, server.load_embedding_model)

    # Warn about deprecated flags
    if getattr(args, "simple_engine", False):
        print(
            "\n  ⚠ --simple-engine is deprecated and has no effect."
            "\n    BatchedEngine is now the sole engine — it handles both"
            "\n    single-user and multi-user workloads with equal performance.\n"
        )
    if getattr(args, "kv_bits", None) is not None:
        print(
            "\n  ⚠ --kv-bits is deprecated and has no effect."
            "\n    For prefix cache quantization, use --kv-cache-quantization instead.\n"
        )
    if getattr(args, "draft_model", None):
        print(
            "\n  ⚠ --draft-model is deprecated and has no effect."
            "\n    For DFlash speculative decoding, use --enable-dflash "
            "(requires a DFlash-eligible alias). "
            "For MTP, use --enable-mtp (requires a model with MTP head).\n"
        )
    if getattr(args, "specprefill", False):
        print("\n  ⚠ --specprefill is deprecated and has no effect.\n")

    # Mutual exclusion: turboquant vs standard quantization
    if args.kv_cache_turboquant and args.kv_cache_quantization:
        print(
            "\n  Error: --kv-cache-turboquant and --kv-cache-quantization are "
            "mutually exclusive. Choose one.\n"
        )
        sys.exit(1)

    # Mutual exclusion: only one spec-decode method may wrap _step at a time.
    # (The DFlash-vs-{suffix,mtp} check is upstream, before the banner.)
    if args.suffix_decoding and args.enable_mtp:
        print(
            "\n  Error: --suffix-decoding and --enable-mtp are mutually "
            "exclusive (both monkey-patch the BatchGenerator step). "
            "Pick one.\n"
        )
        sys.exit(1)

    # Build scheduler config
    enable_prefix_cache = args.enable_prefix_cache and not args.disable_prefix_cache

    scheduler_config = SchedulerConfig(
        max_num_seqs=args.max_num_seqs,
        max_concurrent_requests=args.max_concurrent_requests,
        prefill_batch_size=args.prefill_batch_size,
        completion_batch_size=args.completion_batch_size,
        enable_prefix_cache=enable_prefix_cache,
        prefix_cache_size=args.prefix_cache_size,
        # Memory-aware cache options
        use_memory_aware_cache=not args.no_memory_aware_cache,
        cache_memory_mb=args.cache_memory_mb,
        cache_memory_percent=args.cache_memory_percent,
        # Paged cache options
        use_paged_cache=args.use_paged_cache,
        paged_cache_block_size=args.paged_cache_block_size,
        max_cache_blocks=args.max_cache_blocks,
        # Chunked prefill
        chunked_prefill_tokens=args.chunked_prefill_tokens,
        # Prefill step size (chunk size). Must be plumbed here — BatchedEngine
        # reads it off scheduler_config only; the legacy load_model kwarg was
        # accepted but never used. See #400 and the CLI ↔ Config fidelity
        # audit at scripts/audit_cli_config_fidelity.py.
        prefill_step_size=args.prefill_step_size,
        # MTP
        enable_mtp=args.enable_mtp,
        mtp_num_draft_tokens=args.mtp_num_draft_tokens,
        mtp_optimistic=args.mtp_optimistic,
        # SuffixDecoding
        enable_suffix_decoding=args.suffix_decoding,
        suffix_max_draft=args.suffix_max_draft,
        suffix_max_suffix_len=args.suffix_max_suffix_len,
        suffix_min_confidence=args.suffix_min_confidence,
        suffix_min_draft_len=args.suffix_min_draft_len,
        # KV cache quantization
        kv_cache_quantization=args.kv_cache_quantization,
        kv_cache_quantization_bits=args.kv_cache_quantization_bits,
        kv_cache_quantization_group_size=args.kv_cache_quantization_group_size,
        kv_cache_min_quantize_tokens=args.kv_cache_min_quantize_tokens,
        # TurboQuant V-only compression
        kv_cache_turboquant=args.kv_cache_turboquant,
        kv_cache_turboquant_bits=args.kv_cache_turboquant_bits,
        kv_cache_turboquant_group_size=args.kv_cache_turboquant_group_size,
        # PFlash long-prompt compression (#287)
        pflash_config=pflash_config,
        # D-METAL-CAP: thread the user's --gpu-memory-utilization into
        # SchedulerConfig so the admission gate enforces the same cap
        # that ``mx.set_memory_limit`` only treats as a guideline. The
        # CLI ↔ Config fidelity audit blocks merges where this kwarg
        # exists on SchedulerConfig but is missing at the construction
        # site — without this line, ``--gpu-memory-utilization 0.45``
        # would still set the soft Metal hint but the admission-time
        # check would stay disabled (SchedulerConfig default 0.0),
        # silently recreating the D-METAL-CAP regression.
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    print("Mode: Continuous batching (for multiple concurrent users)")
    if args.chunked_prefill_tokens > 0:
        print(f"Chunked prefill: {args.chunked_prefill_tokens} tokens per step")
    if args.enable_mtp:
        print(f"MTP: enabled, draft_tokens={args.mtp_num_draft_tokens}")
    if args.suffix_decoding:
        print(
            f"SuffixDecoding: enabled, max_draft={args.suffix_max_draft}, "
            f"max_suffix={args.suffix_max_suffix_len}, "
            f"min_conf={args.suffix_min_confidence}"
        )
    print(f"Stream interval: {args.stream_interval} tokens")
    if args.use_paged_cache:
        print(
            f"Paged cache: block_size={args.paged_cache_block_size}, max_blocks={args.max_cache_blocks}"
        )
    elif enable_prefix_cache and not args.no_memory_aware_cache:
        cache_info = (
            f"{args.cache_memory_mb}MB"
            if args.cache_memory_mb
            else f"{args.cache_memory_percent * 100:.0f}% of RAM"
        )
        print(f"Memory-aware cache: {cache_info}")
        if args.kv_cache_turboquant:
            bits_str = (
                str(args.kv_cache_turboquant_bits)
                if args.kv_cache_turboquant_bits
                else "auto"
            )
            print(
                f"TurboQuant V-cache: {bits_str}-bit, "
                f"group_size={args.kv_cache_turboquant_group_size} (K stays FP16)"
            )
        elif args.kv_cache_quantization:
            print(
                f"KV cache quantization: {args.kv_cache_quantization_bits}-bit, "
                f"group_size={args.kv_cache_quantization_group_size}"
            )
    elif enable_prefix_cache:
        print(f"Prefix cache: max_entries={args.prefix_cache_size}")

    # Check port availability before loading model (avoid wasting RAM on conflict).
    # Set SO_REUSEADDR to match uvicorn's bind behavior — without it, this
    # preflight fails on a port still in TCP TIME_WAIT (e.g. just after a
    # previous rapid-mlx process exited), even though uvicorn would happily
    # bind it. Caused spurious "port in use" errors for back-to-back server
    # starts in the validation pipeline.
    #
    # Skip in --listen-fd mode: the supervisor has already bound the socket
    # and handed us the fd. There is no host/port for us to check, and any
    # bind we attempt here would race or collide with the inherited socket.
    if getattr(args, "listen_fd", None) is None:
        # Shared helper so the legacy ``python -m vllm_mlx.server``
        # entrypoint (vllm_mlx/server.py) can call the same probe
        # without duplicating the wildcard-alias / loopback-shadow
        # logic. See ``_port_preflight_or_die`` for why we probe both
        # the requested host AND 127.0.0.1 when the requested host is
        # a wildcard alias.
        _port_preflight_or_die(args.host, args.port, model=args.model)

    # Check disk space before downloading model
    _check_disk_space(args.model, force=getattr(args, "force_disk_check", False))

    # Pre-flight memory check — warn (don't abort) if model + working set
    # would push unified memory past the kernel-panic threshold (issue #324).
    _check_memory_capacity(args.model)

    # DFlash fork: when --enable-dflash is set, skip BatchedEngine entirely
    # and run the dedicated DFlash server. The eligibility check above has
    # already validated the alias, so by here we have a known-good profile.
    if args.enable_dflash:
        # DFlash IS a speculative-decode path. The --no-spec-decode escape
        # hatch (SOP §10) must reject it here — otherwise the user thinks
        # they've disabled spec-decode but DFlash silently proceeds via
        # its dedicated server, never touching EngineCore / ModelConfig.
        if getattr(args, "no_spec_decode", False):
            print(
                "error: --enable-dflash and --no-spec-decode are mutually "
                "exclusive — DFlash is a speculative-decode mode.",
                file=sys.stderr,
            )
            sys.exit(2)
        from .model_aliases import resolve_profile
        from .speculative.dflash.server import run_dflash_server

        _alias_name = getattr(args, "_original_alias", None) or args.model
        _profile = resolve_profile(_alias_name)
        # The eligibility check at top of serve_command guarantees this
        # passes — assert to be defensive against future refactors.
        assert _profile is not None and _profile.supports_dflash, (
            f"DFlash profile invariant violated for {_alias_name!r}"
        )
        run_dflash_server(
            main_model_repo=_profile.hf_path,
            drafter_repo=_profile.dflash_draft_model,  # validated non-None by _coerce
            host=args.host,
            port=args.port,
            served_model_name=args.served_model_name or _alias_name,
            default_max_tokens=effective_max_tokens,
            cors_origins=cors_origins,
            uvicorn_log_level=uvicorn_log_level,
            no_thinking=args.no_thinking,
        )
        return

    # Load model with unified server
    if args.mllm and args.no_mllm:
        print(
            "error: --mllm and --no-mllm are mutually exclusive — "
            "pick one to override auto-detection.",
            file=sys.stderr,
        )
        sys.exit(2)
    if getattr(args, "force_hybrid", False) and getattr(args, "no_hybrid", False):
        print(
            "error: --force-hybrid and --no-hybrid are mutually exclusive — "
            "pick one to override auto-detection.",
            file=sys.stderr,
        )
        sys.exit(2)
    if getattr(args, "force_spec_decode", False) and getattr(
        args, "no_spec_decode", False
    ):
        print(
            "error: --force-spec-decode and --no-spec-decode are mutually "
            "exclusive — pick one to override auto-detection.",
            file=sys.stderr,
        )
        sys.exit(2)
    if getattr(args, "force_openai_harmony_streaming", False) and getattr(
        args, "no_openai_harmony_streaming", False
    ):
        print(
            "error: --force-openai-harmony-streaming and "
            "--no-openai-harmony-streaming are mutually exclusive — pick one "
            "to override the HarmonyStreamingRouter auto-upgrade gate (#516).",
            file=sys.stderr,
        )
        sys.exit(2)
    try:
        load_model(
            args.model,
            scheduler_config=scheduler_config,
            stream_interval=args.stream_interval,
            max_tokens=effective_max_tokens,
            max_tokens_is_explicit=_max_tokens_is_explicit,
            force_mllm=args.mllm,
            force_text=args.no_mllm,
            gpu_memory_utilization=args.gpu_memory_utilization,
            cloud_model=args.cloud_model,
            cloud_threshold=args.cloud_threshold,
            cloud_api_base=args.cloud_api_base,
            cloud_api_key=args.cloud_api_key,
            served_model_name=args.served_model_name,
            mtp=args.enable_mtp,
            force_hybrid=getattr(args, "force_hybrid", False),
            no_hybrid=getattr(args, "no_hybrid", False),
            force_spec_decode=getattr(args, "force_spec_decode", False),
            no_spec_decode=getattr(args, "no_spec_decode", False),
            force_openai_harmony_streaming=getattr(
                args, "force_openai_harmony_streaming", False
            ),
            no_openai_harmony_streaming=getattr(
                args, "no_openai_harmony_streaming", False
            ),
        )
    except Exception as e:
        # Show clean error instead of raw traceback. Catch the typed
        # HF exception class for the 404 case; fall back to substring
        # match for legacy callers (older huggingface_hub) and for
        # non-HF errors that still spell out "not found".
        from huggingface_hub.utils import RepositoryNotFoundError

        is_404 = isinstance(e, RepositoryNotFoundError) or (
            "404" in str(e) or "not found" in str(e).lower()
        )
        if is_404:
            shown = getattr(args, "_original_alias", args.model)
            print(f"\n  Error: Model '{shown}' not found on HuggingFace.")
            _print_unknown_model_help(
                shown, full_path_example="mlx-community/Qwen3.5-9B-4bit"
            )
        else:
            print(f"\n  Error loading model: {e}")
        sys.exit(1)

    # Start server
    # Note: Metal shader warmup runs in the FastAPI lifespan hook (server.py).
    # The "Ready:" banner is printed FROM that hook once warmup completes and
    # the port is actually bound — printing it here would lie to users who
    # curl immediately and get connection-refused while shaders compile.
    print()
    host_display = "localhost" if args.host == "0.0.0.0" else args.host
    listen_fd = getattr(args, "listen_fd", None)
    if listen_fd is not None:
        # Socket activation path — supervisor pre-bound the listening
        # socket. We don't know the actual address from the fd without a
        # ``getsockname`` lookup; surfacing fd=<N> in the banner is the
        # honest thing to print here.
        print(
            f"  Starting server on inherited fd {listen_fd} "
            "(warming up — this can take a few seconds)"
        )
    else:
        print(
            f"  Starting server on http://{host_display}:{args.port} (warming up — this can take a few seconds)"
        )
    from vllm_mlx._version_check import print_staleness_warning_if_any

    print_staleness_warning_if_any()
    print()

    # Stash the source of truth for the lifespan "Ready:" banner —
    # which shape depends on the bind mode:
    #
    #   * Default (host+port): stamp ``bind_host``/``bind_port`` so the
    #     banner prints ``Ready: http://host:port/v1``.
    #   * ``--listen-fd``: stamp ``bind_listen_fd`` instead. The
    #     supervisor's ``getsockname`` is the only honest source for the
    #     address — stamping ``args.host``/``args.port`` here would lie
    #     to log readers (the supervisor might have bound to a different
    #     address). Codex rounds 1+3 PR #696 review.
    from vllm_mlx.config import get_config

    # Always reset BOTH source-of-truth fields before stamping the
    # active branch — the singleton config persists across in-process
    # ``serve_command`` invocations (test harnesses, embedded usage), so
    # a prior host/port stash would otherwise take precedence over a
    # subsequent fd stash (and vice-versa) and the Ready banner would
    # lie about which listener is live. Codex round-4 PR #696 review.
    _cfg = get_config()
    _cfg.bind_host = None
    _cfg.bind_port = None
    _cfg.bind_listen_fd = None
    if listen_fd is None:
        _cfg.bind_host = host_display
        _cfg.bind_port = args.port
    else:
        _cfg.bind_listen_fd = listen_fd

    _run_uvicorn(app, args, uvicorn_log_level)


def _run_tier_submit_flow(args) -> int:
    """``rapid-mlx bench <model> --tier <T> --submit`` — PR #5 unification.

    Three-phase pipeline:

    1. Run the requested tier's smoke / harness work through the
       existing HTTP-server-backed dispatcher (``run_tier`` with
       ``return_results=True``). For ``tier='all'`` we pass
       ``skip_speed=True`` because phase 2 will produce the comparable
       speed numbers directly from the engine; running the lightweight
       HTTP-speed probe too would just double-cost the bench AND
       produce a second set of non-comparable numbers next to it.
       For ``tier='speed'`` phase 1 is a no-op — straight to phase 2.
    2. Run the locked B=1 ``run_standardized_bench`` against the same
       model so the schema-required ``buckets`` field carries the
       comparable numbers the community-benchmarks corpus expects.
       This phase IS what plain ``--submit`` (no ``--tier``) has
       always done; the tier kwargs just decorate the payload.
    3. Build the schema-v2 payload and run the standard interactive
       submit flow (consent → write → commit → push → gh pr create).

    Tier-failure handling: if phase 1's smoke probe FAILS, abort
    before phase 2 — there's no point benching a model that can't
    answer "what is 2+2?". A phase 1 harness failure does NOT abort:
    submitting a failure row IS the point of the harness tier (the
    aggregator wants visibility into "this combo doesn't pass the
    gauntlet"), so we proceed and let the payload carry the per-
    adapter failure flags.
    """
    tier = args.tier
    # Validate the tier even though argparse's ``choices=`` should
    # have rejected anything else — a programmatic Namespace (e.g.
    # someone constructing args directly) could bypass argparse, and
    # the previous ``assert`` would be stripped under ``python -O``
    # (Codex PR #623 review NIT-1). Explicit guard returns 2 with a
    # readable error rather than blowing up later inside the submit
    # flow with a less targeted traceback.
    if tier not in ("smoke", "speed", "harness", "all"):
        print(
            f"  Error: unknown tier {tier!r}; expected one of "
            "smoke / speed / harness / all",
            file=sys.stderr,
        )
        return 2

    # Reject --base-url for the --submit combo (Codex PR #623
    # BLOCKING-1). The community-bench corpus aggregates by
    # (chip, model, version) — every submission MUST reflect the
    # contributor's actual hardware booting their actual model. Two
    # gaps if we allowed --base-url:
    #
    # 1. ``smoke_result.boot_time_ms`` is meaningless when the
    #    server was already up (we didn't measure the user's boot);
    #    the producer would have to invent a ``0.0`` placeholder
    #    that downstream consumers can't distinguish from "machine
    #    boots the model in zero ms" — a misleading row in the DB.
    # 2. Phase 2 runs ``run_standardized_bench`` IN PROCESS against
    #    a freshly-loaded engine, so the buckets numbers would NOT
    #    match the server the user pointed at. We'd publish a
    #    payload labelling itself as the user's setup while the
    #    speed numbers came from a separate engine init.
    #
    # The narrow --tier (no --submit) --base-url path is still
    # supported — that's the gauntlet/release_check use case where
    # we WANT to validate against an already-running server.
    # Belt-and-braces: an active ``RAPID_MLX_HARNESS_PROFILES_FILTER``
    # produces a partial harness payload (only the filtered keys), which
    # would fail the schema-v2 ``required`` set at submission time
    # downstream. The G12 gauntlet path only sets this env when calling
    # ``--tier harness --base-url`` (no --submit) — but a future caller
    # combining ``--submit`` with the filter would silently break here.
    # Refuse loudly instead.
    if os.environ.get("RAPID_MLX_HARNESS_PROFILES_FILTER"):
        print(
            "  Error: --submit is incompatible with "
            "RAPID_MLX_HARNESS_PROFILES_FILTER. The filter scopes the "
            "sweep to a subset of harnesses, producing a payload that "
            "would fail the community-bench schema's required-keys check "
            "(all 5 harnesses must be present). Unset the env var or "
            "drop --submit.",
            file=sys.stderr,
        )
        return 2

    if getattr(args, "base_url", None):
        print(
            "  Error: --base-url is incompatible with --submit. "
            "Community-bench submissions must reflect a fresh boot of "
            "your model on your hardware — smoke_result.boot_time_ms "
            "and the standardized B=1 buckets are both measured "
            "in-process. Drop --base-url and let bench --tier "
            "--submit boot the server itself.",
            file=sys.stderr,
        )
        return 2

    # tier='speed' --submit is the historical --submit path with a
    # new ``tier='speed'`` tag on the payload. No phase 1 needed.
    if tier == "speed":
        return _run_submit_flow(args, tier="speed")

    # Phase 1: run the tier dispatcher to capture smoke/harness data.
    # Speed bucket is intentionally skipped (see docstring); ``run_tier``
    # only honours ``skip_speed`` when tier=='all'.
    from .bench.tier_runner import run_tier

    rc, tier_results = run_tier(
        model=args.model,
        tier=tier,
        base_url=getattr(args, "base_url", None),
        sampled=getattr(args, "sampled", False),
        return_results=True,
        skip_speed=True,
    )
    smoke_result = tier_results.get("smoke_result")
    harness_result = tier_results.get("harness_result")

    # Abort gating. The smoke probe is a hard prerequisite for ANY
    # submission: if the model can't say "4" the speed numbers we'd
    # collect in phase 2 would be misleading at best and a fork-and-
    # burn of the user's compute at worst. Harness failures are
    # surfaced THROUGH the payload (the schema's per-adapter
    # ``passed: false`` carries the signal); we DON'T abort there.
    if tier in ("smoke", "all") and smoke_result is not None:
        if not smoke_result.get("first_prompt_ok", False):
            print(
                "\n  Submission aborted: smoke probe failed. The model "
                "couldn't answer the boot prompt cleanly — submitting "
                "speed/harness numbers from this run would be "
                "misleading. Re-check the model + environment with "
                "`rapid-mlx bench <model> --tier smoke` first.",
                file=sys.stderr,
            )
            return 1

    if tier == "smoke" and smoke_result is None:
        # Phase 1 errored before producing smoke_result (e.g. server
        # boot failure). The exit code from ``run_tier`` is already
        # the right thing to return — don't try to phase 2 without
        # the required smoke_result data.
        print(
            "\n  Submission aborted: smoke phase did not produce a "
            "result (server boot likely failed). Nothing was sent.",
            file=sys.stderr,
        )
        return rc or 1
    if tier == "harness" and harness_result is None:
        print(
            "\n  Submission aborted: harness phase did not produce a "
            "result. Nothing was sent.",
            file=sys.stderr,
        )
        return rc or 1
    if tier == "all" and (smoke_result is None or harness_result is None):
        print(
            "\n  Submission aborted: --tier all did not produce both "
            "smoke and harness results. Nothing was sent.",
            file=sys.stderr,
        )
        return rc or 1

    # Phase 2 + 3 reuse the existing standardized + submit path; the
    # tier kwargs decorate the payload built inside ``_run_submit_flow``.
    return _run_submit_flow(
        args,
        tier=tier,
        smoke_result=smoke_result,
        harness_result=harness_result,
    )


def _run_submit_flow(
    args,
    *,
    tier: str | None = None,
    smoke_result: dict | None = None,
    harness_result: dict | None = None,
) -> int:
    """Execute the standardized B=1 community-bench + PR-open flow.

    Routed-to from ``bench_command`` whenever ``--submit`` is set.
    Kept as a separate function so the freeform bench path stays
    completely untouched — the standardized path imports its own
    deps lazily so that users who never touch ``--submit`` don't pay
    the import cost of the community_bench module.

    PR #5 added the schema-v2 tier-tagging kwargs:

    - ``tier`` — string copied verbatim into the ``tier`` field of the
      payload (``"speed"`` | ``"smoke"`` | ``"harness"`` | ``"all"``).
      ``None`` (the default, used by ``--submit`` without ``--tier``)
      omits the field, preserving byte-for-byte equivalence with the
      v1 ``--submit`` payload shape.
    - ``smoke_result`` / ``harness_result`` — schema-v2 sub-objects
      from the tier dispatcher. The builder enforces the
      tier↔result coupling so passing the wrong combo here ``ValueError``s
      at the payload-build line rather than landing a half-shaped row
      in the submissions corpus.
    """
    import asyncio
    from pathlib import Path

    from huggingface_hub.utils import RepositoryNotFoundError
    from mlx_lm import load

    from .community_bench.hardware import collect as collect_hw
    from .community_bench.hardware import is_apple_silicon
    from .community_bench.runner import run_standardized_bench
    from .community_bench.submission import (
        build_submission_payload,
        submit_interactive,
    )
    from .engine_core import AsyncEngineCore, EngineConfig
    from .model_aliases import resolve_profile
    from .scheduler import SchedulerConfig

    if not is_apple_silicon():
        print(
            "  Error: --submit only runs on Apple Silicon (arm64 Darwin). "
            "The community database is Apple-Silicon-specific."
        )
        return 2

    # Whitelist gate. ``model.alias`` in the payload is the bucketing
    # key, so we require the user to type the canonical alias *key*
    # rather than a raw HF path — accepting both forms would let a
    # contributor's typo silently shift their submission into a
    # different bucket via the reverse-lookup. (Codex PR #582 BLOCKING:
    # silent alias coercion bypasses the intended "must be a whitelist
    # key" contract.) The GHA validator re-checks the alias against
    # aliases.json, so this guard is layered. ``args._original_alias``
    # holds the user-typed value when the dispatcher resolved an alias
    # to an HF path; if it's absent (HF path passed directly, or any
    # other no-resolution case) we fall back to ``args.model``, which
    # this guard then re-checks for the ``/`` HF-path signature.
    user_typed = getattr(args, "_original_alias", None) or args.model
    if "/" in user_typed:
        print(
            f"  Error: --submit requires the canonical alias key "
            f"(e.g. 'qwen3.5-9b-4bit'), not the resolved HF path "
            f"'{user_typed}'. Run `rapid-mlx models` for the whitelist."
        )
        return 2
    profile = resolve_profile(user_typed)
    if profile is None:
        print(
            f"  Error: '{user_typed}' is not a registered alias. "
            f"Only models listed in vllm_mlx/aliases.json can be submitted "
            f"(this keeps the comparison apples-to-apples)."
        )
        print("  Run `rapid-mlx models` to see the full whitelist.")
        return 2
    alias = user_typed
    hf_path = profile.hf_path

    notes = args.notes or None
    if notes is not None:
        if len(notes) > 200:
            print("  Error: --notes must be <= 200 chars (schema cap).")
            return 2
        # Reject control characters in --notes. Newlines/CR/terminal
        # escapes would land in the PR body, the JSON file, and any
        # future renderer — the schema's free-form ``notes`` field
        # invites contributor commentary, but it does not invite
        # ``\x1b]0;owned\x07`` terminal-title-set sequences.
        # (Codex PR #582 round-7 NIT.)
        if any(ord(c) < 0x20 or ord(c) == 0x7F for c in notes):
            print(
                "  Error: --notes contains control characters; only "
                "printable ASCII/UTF-8 is permitted."
            )
            return 2

    _check_disk_space(hf_path, force=getattr(args, "force_disk_check", False))
    _check_memory_capacity(hf_path)

    # ``--sampled`` runs a SECOND submission (with sampling="sampled")
    # in addition to the always-on greedy run. The README contract is
    # "two rows when --sampled is set, one row otherwise" — a previous
    # version replaced greedy with sampled, breaking that contract and
    # silently losing the greedy comparison line. (Codex PR #582
    # round-7 BLOCKING.) Greedy goes first so the contributor can
    # still cancel the sampled half during its consent prompt.
    sampling_modes: list[str] = ["greedy"]
    if getattr(args, "sampled", False):
        sampling_modes.append("sampled")

    async def _run() -> int:
        import concurrent.futures

        from .engine_core import _init_mlx_step_thread

        # Load model on the future mlx-step worker thread (#170). mlx-lm
        # 0.31.3+ binds module-level ``generation_stream`` and any
        # auto-default stream to the thread that triggers them. If the
        # model weights or ``mx.compile``-cached graphs are touched on
        # the asyncio loop thread first, every later eval on the step
        # worker raises "There is no Stream(gpu, N) in current thread."
        # Spinning the worker BEFORE load and reusing it for
        # AsyncEngineCore keeps every MLX op on a single owning thread.
        # Mirrors the pattern in ``BatchedEngine._start_llm`` (which is
        # why ``rapid-mlx serve`` works but the unfixed ``bench`` path
        # doesn't).
        print(f"  Loading model {alias} ({hf_path})…")
        model_load_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="mlx-step",
            initializer=_init_mlx_step_thread,
        )
        try:
            model, tokenizer = model_load_executor.submit(load, hf_path).result()
        except (ValueError, ModuleNotFoundError) as e:
            # mlx-lm raises ``ValueError: Model type X not supported`` plus an
            # internal ``ModuleNotFoundError: No module named 'mlx_lm.models.X'``
            # for any architecture it can't import. The Gemma 4 family lives
            # in mlx-vlm (the model classes are vision-aware even for the
            # text-only checkpoints), so a bare ``pip install rapid-mlx``
            # without the ``[vision]`` extras hits this every time. The
            # README still recommends ``gemma-4-*`` aliases so newcomers
            # would otherwise see a raw traceback and conclude the model
            # is broken — translate to an actionable hint. Placed BEFORE
            # the broader ``OSError`` clause so a future maintainer can't
            # accidentally make the broad branch swallow it (Codex PR
            # #600 round-1 BLOCKING).
            msg = str(e)
            needs_vision = (
                "gemma4_unified" in msg
                or "gemma4" in msg
                or "mlx_vlm" in msg
                or "mlx-vlm" in msg
            )
            if needs_vision:
                print()
                print(
                    "  Error: this model needs the vision extras (Gemma 4 "
                    "architecture classes live in mlx-vlm)."
                )
                print("  Install them and re-run:")
                print()
                print("    pip install 'rapid-mlx[vision]'")
                print()
                print(
                    "  Or, if you only need text inference (smaller "
                    "footprint, ~16 MB vs ~450 MB):"
                )
                print("    pip install --no-deps 'mlx-vlm>=0.6.1'")
                print()
            else:
                print(f"  Error loading model: {e}")
            model_load_executor.shutdown(wait=False)
            return 2
        except (RepositoryNotFoundError, OSError) as e:
            print(f"  Error loading model: {e}")
            model_load_executor.shutdown(wait=False)
            return 2

        # Standardized config: B=1, no batching, prefix-cache off so the
        # numbers reflect cold prefill on each round (which is what the
        # tg/pp metrics are supposed to measure).
        scheduler_config = SchedulerConfig(
            max_num_seqs=1,
            max_concurrent_requests=1,
            prefill_batch_size=1,
            completion_batch_size=1,
            enable_prefix_cache=False,
        )
        engine_config = EngineConfig(
            model_name=hf_path,
            scheduler_config=scheduler_config,
        )

        print("  Collecting hardware fingerprint…")
        hardware, software = collect_hw()
        print(
            f"    chip={hardware.chip}, ram={hardware.ram_gb} GB, "
            f"cpu_cores={hardware.cpu_cores}, gpu_cores={hardware.gpu_cores}"
        )
        print(
            f"    macos={software.macos}, rapid_mlx={software.rapid_mlx}, "
            f"mlx={software.mlx}, python={software.python}"
        )

        repo_root = Path(args.repo_root) if args.repo_root else Path.cwd()
        # Pass the EXISTING executor to AsyncEngineCore so the engine
        # loop, BatchGenerator construction, and every forward pass run
        # on the same thread that owns the model weights.
        async with AsyncEngineCore(
            model, tokenizer, engine_config, executor=model_load_executor
        ) as engine:
            for mode in sampling_modes:
                print(
                    f"  Running standardized bench "
                    f"(sampling={mode}, 2 buckets × 5 rounds + 1 warmup)…"
                )
                try:
                    bench = await run_standardized_bench(
                        engine, tokenizer, sampling=mode
                    )
                except RuntimeError as exc:
                    # Friendly surface for the bench's "exactly N tokens"
                    # guard. As of #567's fix this branch is engine-bug
                    # territory (sampling sets ``ignore_eos=True`` so the
                    # model's EOS shouldn't fire); previously it blamed
                    # the user's model alias. Print a clear summary so
                    # contributors aren't dumped into a raw traceback.
                    msg = str(exc)
                    if "standardized bench requires exactly" in msg:
                        print()
                        print(
                            "  Bench round aborted (engine bug — NOT your model's fault):"
                        )
                        for line in msg.split(". "):
                            line = line.strip()
                            if line:
                                print(f"    {line}")
                        print()
                        return 1
                    raise

                print(
                    f"    short: decode={bench.short.decode_stat['median']:.2f} tok/s, "
                    f"prefill={bench.short.prefill_stat['median']:.2f} tok/s, "
                    f"ttft={bench.short.ttft_stat['median']:.1f} ms"
                )
                print(
                    f"    long:  decode={bench.long.decode_stat['median']:.2f} tok/s, "
                    f"prefill={bench.long.prefill_stat['median']:.2f} tok/s, "
                    f"ttft={bench.long.ttft_stat['median']:.1f} ms"
                )

                payload = build_submission_payload(
                    hardware=hardware,
                    software=software,
                    alias=alias,
                    hf_path=hf_path,
                    bench=bench,
                    notes=notes,
                    # v2 tier-tagging: pass through only when the caller
                    # supplied them. The builder validates the tier ↔
                    # smoke_result/harness_result coupling — passing
                    # ``smoke_result`` for ``tier=speed`` would
                    # ``ValueError`` here rather than land a half-shaped
                    # row in the corpus.
                    tier=tier,
                    smoke_result=smoke_result,
                    harness_result=harness_result,
                )
                rc = submit_interactive(payload, repo_root)
                if rc != 0:
                    # Setup error (not a "user said no") — bail out
                    # before kicking off the second submission so the
                    # contributor sees the failure clearly.
                    return rc
        return 0

    return asyncio.run(_run())


def bench_command(args):
    """Run benchmark."""
    import asyncio
    import time

    # Install the MLX hardware-compat shim BEFORE `from mlx_lm import load`.
    # `mlx_lm/__init__.py` re-exports from `mlx_lm.generate`, which captures
    # `mx.new_thread_local_stream(mx.default_device())` at module-import time;
    # on M5 single-stream GPUs that stream is unusable (#404). Bench is a
    # separate entry point from `serve` so it doesn't inherit the
    # scheduler-side install — wire the shim here directly. Idempotent, no-op
    # on hardware where the original API works.
    from . import _mlx_compat as _mlx_compat

    _mlx_compat.install()

    # --tier routes through the user-facing tier dispatcher (PR #2 of
    # the bench-consolidation series). PR #5 unified --tier with
    # --submit: when both flags are set the dispatcher runs the
    # requested smoke/harness work for the schema-v2 sub-objects and
    # ALSO runs the locked B=1 ``run_standardized_bench`` so the
    # required ``buckets`` field carries comparable numbers (the
    # lightweight tier-speed probe is NEVER submitted — its results
    # aren't apples-to-apples with the community DB).
    if getattr(args, "tier", None) and getattr(args, "submit", False):
        sys.exit(_run_tier_submit_flow(args))

    if getattr(args, "tier", None):
        from .bench.tier_runner import run_tier

        sys.exit(
            run_tier(
                model=args.model,
                tier=args.tier,
                base_url=getattr(args, "base_url", None),
                sampled=getattr(args, "sampled", False),
            )
        )

    # --submit routes through the standardized community-bench runner,
    # which locks the comparability knobs the freeform path exposes.
    # Keep the branch high in this function so the rest of bench_command
    # doesn't accidentally read --submit-only args.
    if getattr(args, "submit", False):
        sys.exit(_run_submit_flow(args))

    from mlx_lm import load

    from .api.utils import is_mllm_model as _bench_is_mllm_model
    from .engine_core import AsyncEngineCore, EngineConfig
    from .pflash import config_from_args as _pflash_config_from_args
    from .pflash import resolve_pflash_mode_default as _pflash_resolve_default
    from .pflash import validate_model_support as _bench_pflash_validate
    from .request import SamplingParams
    from .scheduler import SchedulerConfig

    _check_disk_space(args.model, force=getattr(args, "force_disk_check", False))
    _check_memory_capacity(args.model)

    # Handle prefix cache flags
    enable_prefix_cache = args.enable_prefix_cache and not args.disable_prefix_cache

    # PFlash for the bench command — same per-alias default as serve:
    # verified Qwen3.5 / Qwen3.6 aliases switch to ``always``, everything
    # else stays ``off``. Resolves before config_from_args so the
    # validate path sees the final mode, then runs the MLLM-rejection
    # gate ``serve``/``server.py`` already enforce (codex r3 BLOCKING:
    # bench previously skipped this check, so ``rapid-mlx bench
    # --pflash always <mllm-alias>`` would admit a combo PFlash
    # explicitly rejects elsewhere).
    args.pflash = _pflash_resolve_default(args, model_name=args.model)
    try:
        bench_pflash_config = _pflash_config_from_args(args)
        _bench_pflash_validate(
            bench_pflash_config,
            model_name=args.model,
            is_mllm=getattr(args, "mllm", False) or _bench_is_mllm_model(args.model),
        )
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    async def run_benchmark():
        print(f"Loading model: {args.model}")
        try:
            model, tokenizer = load(args.model)
        except Exception as e:
            # Mirror serve_command: clean message instead of a 30-line
            # traceback when the user typed a missing repo / bad alias.
            from huggingface_hub.utils import RepositoryNotFoundError

            is_404 = isinstance(e, RepositoryNotFoundError) or (
                "404" in str(e) or "not found" in str(e).lower()
            )
            if is_404:
                shown = getattr(args, "_original_alias", args.model)
                print(f"\n  Error: Model '{shown}' not found on HuggingFace.")
                _print_unknown_model_help(
                    shown, full_path_example="mlx-community/Qwen3.5-9B-4bit"
                )
            else:
                print(f"\n  Error loading model: {e}")
            sys.exit(1)

        scheduler_config = SchedulerConfig(
            max_num_seqs=args.max_num_seqs,
            max_concurrent_requests=getattr(args, "max_concurrent_requests", 256),
            prefill_batch_size=args.prefill_batch_size,
            completion_batch_size=args.completion_batch_size,
            enable_prefix_cache=enable_prefix_cache,
            prefix_cache_size=args.prefix_cache_size,
            # Memory-aware cache options
            use_memory_aware_cache=not args.no_memory_aware_cache,
            cache_memory_mb=args.cache_memory_mb,
            cache_memory_percent=args.cache_memory_percent,
            # Paged cache options
            use_paged_cache=args.use_paged_cache,
            paged_cache_block_size=args.paged_cache_block_size,
            max_cache_blocks=args.max_cache_blocks,
            # KV cache quantization
            kv_cache_quantization=args.kv_cache_quantization,
            kv_cache_quantization_bits=args.kv_cache_quantization_bits,
            kv_cache_quantization_group_size=args.kv_cache_quantization_group_size,
            kv_cache_min_quantize_tokens=args.kv_cache_min_quantize_tokens,
            # PFlash long-prompt compression (#287)
            pflash_config=bench_pflash_config,
        )
        engine_config = EngineConfig(
            model_name=args.model,
            scheduler_config=scheduler_config,
        )

        if args.use_paged_cache:
            print(
                f"Paged cache: block_size={args.paged_cache_block_size}, max_blocks={args.max_cache_blocks}"
            )

        # Generate prompts
        prompts = [
            f"Write a short poem about {topic}."
            for topic in [
                "nature",
                "love",
                "technology",
                "space",
                "music",
                "art",
                "science",
                "history",
                "food",
                "travel",
            ][: args.num_prompts]
        ]
        # Prepend a deterministic long context when the user asks for
        # one — primarily for PFlash TTFT replication runs (#287).
        long_prompt_tokens = getattr(args, "long_prompt_tokens", 0)
        long_context = _build_benchmark_context(long_prompt_tokens)
        if long_context:
            prompts = [
                f"{long_context}\n\nUser request:\n{prompt}" for prompt in prompts
            ]

        params = SamplingParams(
            max_tokens=args.max_tokens,
            temperature=0.7,
        )

        print(
            f"\nRunning benchmark with {len(prompts)} prompts, max_tokens={args.max_tokens}"
        )
        if long_prompt_tokens > 0:
            print(f"Long prompt target: ~{long_prompt_tokens} tokens")
        print("-" * 50)

        total_prompt_tokens = 0
        total_completion_tokens = 0

        async with AsyncEngineCore(model, tokenizer, engine_config) as engine:
            await asyncio.sleep(0.1)  # Warm up

            start_time = time.perf_counter()

            # Add all requests
            request_ids = []
            for prompt in prompts:
                rid = await engine.add_request(prompt, params)
                request_ids.append(rid)

            # Collect all outputs
            async def get_output(rid):
                async for out in engine.stream_outputs(rid, timeout=120):
                    if out.finished:
                        return out
                return None

            results = await asyncio.gather(*[get_output(r) for r in request_ids])

            total_time = time.perf_counter() - start_time

        # Calculate stats
        for r in results:
            if r:
                total_prompt_tokens += r.prompt_tokens
                total_completion_tokens += r.completion_tokens

        total_tokens = total_prompt_tokens + total_completion_tokens

        print("\nResults:")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Prompts: {len(prompts)}")
        print(f"  Prompts/second: {len(prompts) / total_time:.2f}")
        print(f"  Total prompt tokens: {total_prompt_tokens}")
        print(f"  Total completion tokens: {total_completion_tokens}")
        print(f"  Total tokens: {total_tokens}")
        print(f"  Tokens/second: {total_completion_tokens / total_time:.2f}")
        print(f"  Throughput: {total_tokens / total_time:.2f} tok/s")

    asyncio.run(run_benchmark())


def _format_bytes(n: int) -> str:
    """Render a byte count as a 1-decimal IEC-suffixed string (GiB/MiB/KiB).

    Picks the largest unit where the value is >= 1; falls back to bytes.
    Returns ``"0 B"`` for zero / negative.

    Aligned with ``vllm_mlx._download_gate._format_size`` so the same
    byte count rendered by ``ls --cached`` and by the B2 confirmation
    prompt uses the same suffix convention (Codex/DeepSeek round-3 NIT:
    ``5.0 G`` vs ``5.0 GiB`` for the same model is the kind of paper-
    cut that makes users think two screens are talking about different
    sizes).
    """
    if n <= 0:
        return "0 B"
    for unit, factor in (
        ("GiB", 1024**3),
        ("MiB", 1024**2),
        ("KiB", 1024),
    ):
        if n >= factor:
            return f"{n / factor:.1f} {unit}"
    return f"{n} B"


def _dir_size_bytes(path: str) -> int:
    """Recursive on-disk size of ``path`` (follows blob symlinks).

    HF cache stores model weights as ``blobs/<sha>`` files referenced via
    symlinks under ``snapshots/<rev>/<file>``. ``os.scandir`` recurses
    through both — we follow links so a snapshot's reported size matches
    the user's mental model of "how much disk this model uses".
    """
    total = 0
    try:
        for entry in os.scandir(path):
            try:
                if entry.is_dir(follow_symlinks=False):
                    total += _dir_size_bytes(entry.path)
                else:
                    # follow_symlinks=True so blob symlinks count their
                    # underlying file size, matching ``du -sL``.
                    total += entry.stat(follow_symlinks=True).st_size
            except OSError:
                continue
    except OSError:
        return total
    return total


def _scan_hf_cache_models() -> list[tuple[str, int, float]]:
    """Return ``[(hf_repo, size_bytes, last_modified_epoch), ...]`` for every
    ``models--<org>--<name>`` directory in the HF cache.

    Empty list when the cache dir doesn't exist (fresh install) or has no
    model entries (e.g. only datasets were downloaded). Datasets/spaces
    (``datasets--*``, ``spaces--*``) are deliberately skipped.
    """
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
    except Exception:
        HF_HUB_CACHE = os.path.expanduser("~/.cache/huggingface/hub")
    if not os.path.isdir(HF_HUB_CACHE):
        return []
    out: list[tuple[str, int, float]] = []
    for name in os.listdir(HF_HUB_CACHE):
        if not name.startswith("models--"):
            continue
        # ``models--org--name`` → ``org/name``. Some legacy entries are
        # ``models--name`` (no org) for single-segment repos; pass those
        # through unchanged so the user still sees them in the listing.
        parts = name[len("models--") :].split("--", 1)
        repo = "/".join(parts) if len(parts) == 2 else parts[0]
        full = os.path.join(HF_HUB_CACHE, name)
        try:
            mtime = os.path.getmtime(full)
        except OSError:
            mtime = 0.0
        size = _dir_size_bytes(full)
        out.append((repo, size, mtime))
    return out


def _print_cached_models() -> None:
    """Render the ``--cached`` view: locally-downloaded HF cache entries
    cross-referenced against the alias registry.

    Each row: ``Alias | HF repo | Size on disk | Last modified``. Models
    not in the alias registry are shown with alias=``(unmapped)`` so the
    user still sees what's eating disk space. Empty cache prints a hint
    pointing at ``pull`` / ``chat``.
    """
    import time as _time

    from vllm_mlx.model_aliases import list_profiles

    rows = _scan_hf_cache_models()
    print()
    if not rows:
        print(
            "  No models cached yet. Run 'rapid-mlx pull <alias>' or "
            "'rapid-mlx chat <alias>' to download one."
        )
        print()
        return

    # Reverse-map HF repo path → alias name so the alias column matches the
    # user's mental model (``qwen3.5-4b-4bit`` not ``mlx-community/Qwen3.5-4B...``).
    profiles = list_profiles()
    hf_to_alias: dict[str, str] = {}
    for alias, p in profiles.items():
        hf_to_alias.setdefault(p.hf_path, alias)

    cols = (
        ("Alias", 22),
        ("HF repo", 50),
        ("Size", 9),
        ("Modified", 12),
    )
    width = sum(w for _, w in cols) + len(cols) - 1
    sep = "  " + "─" * width
    header = "  " + " ".join(f"{name:<{w}}" for name, w in cols)
    print(f"  Cached models ({len(rows)} on disk)")
    print(sep)
    print(header)
    print(sep)

    now = _time.time()
    total_bytes = 0
    # Sort by size descending so the biggest-disk-hog row is first — the
    # most useful ordering for "what do I rm to free space?".
    for repo, size, mtime in sorted(rows, key=lambda r: -r[1]):
        total_bytes += size
        alias = hf_to_alias.get(repo, "(unmapped)")
        # Render modified as a human delta: "2 days ago" beats raw epoch.
        if mtime <= 0:
            mod = "?"
        else:
            delta = max(0, int(now - mtime))
            if delta < 3600:
                mod = f"{delta // 60}m ago"
            elif delta < 86400:
                mod = f"{delta // 3600}h ago"
            else:
                mod = f"{delta // 86400}d ago"
        # Truncate over-long HF paths so the row doesn't wrap on a
        # narrow terminal; the alias column carries the canonical name.
        repo_disp = repo if len(repo) <= 50 else (repo[:47] + "...")
        print(f"  {alias:<22} {repo_disp:<50} {_format_bytes(size):<9} {mod:<12}")
    print(sep)
    print(f"  Total: {_format_bytes(total_bytes)}")
    print()
    print("  Tip: `rapid-mlx rm <hf-repo>` to free disk space")
    print()


def models_command(args):
    """List available model aliases with their per-model profile capabilities.

    Default view pulls from ``list_profiles()`` so every alias's
    ``tool_call_parser`` / ``reasoning_parser`` / ``is_hybrid`` /
    ``supports_spec_decode`` / ``suffix_decoding_tier`` shows up — letting
    users pick a model on capabilities, not just on name.

    ``--cached`` swaps to a disk-only view: scans the HuggingFace cache,
    cross-references against the alias registry, and renders
    ``Alias | HF repo | Size on disk | Last modified``. Also reachable as
    the top-level ``rapid-mlx ls`` alias.
    """
    from vllm_mlx._version_check import print_staleness_warning_if_any
    from vllm_mlx.model_aliases import list_profiles

    print_staleness_warning_if_any()

    if getattr(args, "cached", False):
        _print_cached_models()
        return

    profiles = list_profiles()
    print()
    print(f"  Available models ({len(profiles)} aliases)")

    # Widths sized to fit the longest values currently in aliases.json:
    # alias 24 (deepseek-v4-flash-8bit is 22 chars; +2 pad after explicit
    # quant rename), tool 16 (qwen3_coder_xml + 1 pad), reasoning 12
    # (deepseek_r1 + 1 pad), spec 10 ("✗ hybrid"), tier 11, dflash 7.
    cols = (
        ("Alias", 24),
        ("Tools", 16),
        ("Reasoning", 12),
        ("Spec-Decode", 10),
        ("Suffix Tier", 11),
        ("DFlash", 7),
    )
    width = sum(w for _, w in cols) + len(cols) - 1
    sep = "  " + "─" * width
    header = "  " + " ".join(f"{name:<{w}}" for name, w in cols)
    print(sep)
    print(header)
    print(sep)

    for alias in sorted(profiles.keys()):
        p = profiles[alias]
        tools = p.tool_call_parser or "—"
        reasoning = p.reasoning_parser or "—"
        if p.is_hybrid:
            # Hybrid models cannot use spec-decode or suffix-decode regardless
            # of the supports_spec_decode flag (mlx-lm BatchGenerator gate).
            spec = "✗ hybrid"
            tier = "n/a"
        else:
            spec = "✓" if p.supports_spec_decode else "✗"
            tier = p.suffix_decoding_tier
        # DFlash column — eligible aliases show ✓, everything else "—" so
        # the visual scan immediately surfaces what supports it. We don't
        # re-run the eligibility gate here (which would also check that
        # mlx-vlm 0.5.0+ is installed) — that's a runtime concern; the
        # registry column is pure declarative state.
        dflash = "✓" if p.supports_dflash else "—"
        row = (
            f"  {alias:<24} {tools:<16} {reasoning:<12} "
            f"{spec:<10} {tier:<11} {dflash:<7}"
        )
        print(row)

    print(sep)

    # R10-C1: audio alias section. Pre-R10 ``rapid-mlx models`` listed
    # zero audio aliases because they don't live in ``aliases.json``
    # (which only carries text-LM profiles). Users had no in-tool way
    # to discover ``kokoro`` / ``whisper-large-v3`` / ``parakeet`` —
    # they had to read the docs site. Now the audio registry
    # (vllm_mlx/audio/aliases.json) feeds the same table so
    # ``rapid-mlx models`` is the canonical "what can I serve?" view
    # across every lane.
    try:
        from vllm_mlx.audio.registry import list_audio_aliases

        audio_entries = list_audio_aliases()
    except Exception:
        # A malformed audio registry must NOT break the text alias
        # listing — silently degrade by skipping the audio section.
        audio_entries = []

    if audio_entries:
        print()
        print(f"  Audio models ({len(audio_entries)} aliases)")
        audio_sep = "  " + "─" * width
        print(audio_sep)
        audio_header = f"  {'Alias':<24} {'Kind':<10} {'Family':<12} {'HF id':<40}"
        print(audio_header)
        print(audio_sep)
        for entry in audio_entries:
            kind_tag = f"[audio:{entry.type}]"
            print(
                f"  {entry.alias:<24} {kind_tag:<10} "
                f"{entry.family:<12} {entry.hf_id:<40}"
            )
        print(audio_sep)

    print()
    print("  Tip: `rapid-mlx info <alias>` for the full per-model profile")
    print("       `rapid-mlx pull <alias>` to download")
    print("       `rapid-mlx chat <alias>` for an interactive REPL")
    print("       `rapid-mlx serve <alias>` for an OpenAI-compatible server")
    if audio_entries:
        print("       `rapid-mlx serve kokoro|whisper-large-v3|parakeet` for audio")
    print()


def pull_command(args):
    """Download a model to the HuggingFace cache without serving."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import HFValidationError
    from huggingface_hub.utils import RepositoryNotFoundError

    repo_id = args.model  # already alias-resolved by main()

    # R2-first / HuggingFace-fallback per file. Default mirror is
    # ``https://models.rapidmlx.com``; set ``RAPID_MLX_MODEL_MIRROR=""``
    # to force HF only. The function prints its own progress + summary.
    if _try_mirror_prefetch(repo_id):
        from pathlib import Path

        try:
            from huggingface_hub.constants import HF_HUB_CACHE

            cache_root = Path(HF_HUB_CACHE)
        except Exception:
            cache_root = Path.home() / ".cache" / "huggingface" / "hub"
        owner, _, repo = repo_id.partition("/")
        repo_root = cache_root / f"models--{owner}--{repo}"
        try:
            rev = (repo_root / "refs" / "main").read_text().strip()
            print(f"  Cached at: {repo_root / 'snapshots' / rev}")
        except OSError:
            print(f"  Cached at: {repo_root}")
        return
    # Mirror returned False — fall through to plain snapshot_download.
    # Either the catalog was unreachable, the alias isn't catalog-listed,
    # or one or more files failed both R2 and HF in the per-file pool.
    # snapshot_download will retry from HF with its own (more robust)
    # error reporting.
    print(f"\n  Pulling {repo_id} from HuggingFace ...")
    try:
        path = snapshot_download(repo_id)
    except HFValidationError:
        # Malformed HF repo id (e.g. ``foo/bar/baz``) — surface the same
        # friendly "unknown model" hint the alias path uses instead of a
        # raw stack trace.
        shown = getattr(args, "_original_alias", repo_id)
        print(
            f"\n  Error: '{shown}' is not a valid HuggingFace repo id "
            "(expected ``namespace/name``)."
        )
        _print_unknown_model_help(
            shown, full_path_example="mlx-community/Qwen3.5-9B-4bit"
        )
        sys.exit(1)
    except Exception as e:
        is_404 = isinstance(e, RepositoryNotFoundError) or (
            "404" in str(e) or "not found" in str(e).lower()
        )
        if is_404:
            shown = getattr(args, "_original_alias", repo_id)
            print(f"\n  Error: Model '{shown}' not found on HuggingFace.")
            _print_unknown_model_help(
                shown, full_path_example="mlx-community/Qwen3.5-9B-4bit"
            )
            sys.exit(1)
        raise
    print(f"  Cached at: {path}")


def rm_command(args):
    """Remove a model from the HuggingFace cache."""
    from huggingface_hub import scan_cache_dir

    repo_id = args.model
    cache = scan_cache_dir()
    # Filter by repo_type=="model" — same repo_id can refer to a dataset or
    # space, and we don't want ``rapid-mlx rm foo`` deleting a dataset.
    matching = [
        r for r in cache.repos if r.repo_id == repo_id and r.repo_type == "model"
    ]
    if not matching:
        print(f"\n  '{repo_id}' is not in the HuggingFace cache.")
        print("  Nothing to remove.")
        sys.exit(1)

    repo = matching[0]
    revisions = [rev.commit_hash for rev in repo.revisions]
    strategy = cache.delete_revisions(*revisions)
    print(f"\n  Removing {repo_id} ({strategy.expected_freed_size_str}) ...")
    strategy.execute()
    print("  Done.")


def ps_command(_args):
    """List running rapid-mlx servers (process scan)."""
    import time

    import psutil

    rows: list[tuple[int, str, str, str]] = []
    for proc in psutil.process_iter(["pid", "cmdline", "create_time"]):
        try:
            cmd = proc.info["cmdline"] or []
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if not any(
            ("rapid-mlx" in c or "vllm_mlx" in c) and "serve" in cmd for c in cmd
        ):
            continue

        # Extract model arg and --port flag. argparse accepts options
        # before positionals, so the model is the first non-flag token
        # after `serve` whose prior token isn't a value-taking flag.
        # The small list of flags here is conservative; unknown flags
        # are assumed to NOT take a value.
        VALUE_FLAGS = {
            "--host",
            "--port",
            "--api-key",
            "--tool-call-parser",
            "--reasoning-parser",
            "--log-level",
            "--mcp-config",
            "--cors-origins",
            "--cloud-model",
            "--cloud-api-base",
            "--cloud-api-key",
            "--served-model-name",
            "--max-tokens",
            "--gpu-memory-utilization",
        }
        model = "(unknown)"
        port = "8000"  # serve's default
        try:
            i = cmd.index("serve") + 1
            # Pre-PR this loop ``break``ed on the first positional, so a
            # ``rapid-mlx serve qwen3.5-4b-4bit --port 8005`` ended with
            # port="8000" because the positional model token came before
            # ``--port``. Keep scanning for flags after we've captured the
            # model — argparse accepts them on either side.
            model_seen = False
            while i < len(cmd):
                tok = cmd[i]
                if tok.startswith("--"):
                    if "=" in tok:
                        key, val = tok.split("=", 1)
                        if key == "--port":
                            port = val
                        i += 1
                    elif tok in VALUE_FLAGS:
                        if tok == "--port" and i + 1 < len(cmd):
                            port = cmd[i + 1]
                        i += 2
                    else:
                        i += 1
                else:
                    if not model_seen:
                        model = tok
                        model_seen = True
                    i += 1
        except ValueError:
            pass

        uptime_s = max(0, int(time.time() - proc.info["create_time"]))
        h, m = uptime_s // 3600, (uptime_s % 3600) // 60
        uptime = f"{h}h{m:02d}m" if h else f"{m}m{uptime_s % 60:02d}s"
        rows.append((proc.info["pid"], port, model, uptime))

    if not rows:
        print("\n  No rapid-mlx servers running.")
        return

    print()
    print(f"  {'PID':<8}{'PORT':<8}{'MODEL':<40}{'UPTIME':<10}")
    print(f"  {'-' * 66}")
    # Sort numerically by port — string sort would put "10000" before "8000".
    for pid, port, model, uptime in sorted(rows, key=lambda r: int(r[1])):
        print(f"  {pid:<8}{port:<8}{model:<40}{uptime:<10}")
    print()


def _spawn_chat_server(
    model: str,
    log_path: str,
    served_name: str | None = None,
    *,
    register_in: list | None = None,
) -> tuple[object, str]:
    """Spawn a `serve` subprocess on an ephemeral port for chat REPL use.

    Returns (Popen handle, base_url).

    ``register_in`` is an optional list (typically the chat REPL's
    ``_active_procs``). When provided, the new ``Popen`` is appended to it
    *immediately* after construction — narrowing the SIGTERM-orphan race
    that exists between ``Popen()`` returning and the caller registering
    the handle. Caller-side ``register_in.append(proc)`` would still leave
    one Python statement of unprotected window; doing it inside this
    function closes that window for the caller.

    If ``served_name`` is given, it is passed via ``--served-model-name`` so
    the spawned server exposes the alias as the API model name (e.g. user
    typed ``qwen3.5-4b-4bit`` → API requests use ``qwen3.5-4b-4bit`` rather than the
    expanded HF path).
    """
    import socket
    import subprocess

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    base_url = f"http://127.0.0.1:{port}"
    cmd = [
        sys.executable,
        "-m",
        "vllm_mlx.cli",
        "serve",
        model,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--log-level",
        "WARNING",
    ]
    if served_name and served_name != model:
        cmd.extend(["--served-model-name", served_name])
    log = open(log_path, "w")  # noqa: SIM115 — kept open for proc lifetime
    # Tell the child main() that the parent already gated (or that this is
    # an internal spawn, where prompting would deadlock anyway because the
    # child stdin is not a TTY). Without this, the child's B2 gate would
    # see a stdin pipe and re-evaluate against a potentially-stale cache.
    child_env = os.environ.copy()
    child_env["RAPID_MLX_CHAT_SPAWN"] = "1"
    try:
        proc = subprocess.Popen(  # noqa: S603
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=child_env,
        )
    except (OSError, ValueError):
        # Popen raised before constructing the child — the log handle
        # would otherwise leak. Re-raise after closing.
        log.close()
        raise
    # Register first so a SIGTERM landing between here and the caller's
    # next statement still tears the child down.
    if register_in is not None:
        register_in.append(proc)
    # Stash the log handle and path on the proc object so the chat REPL
    # can close+unlink them when the proc is torn down (fixes the file
    # descriptor + tempfile leak across `/model` swaps).
    proc._rapid_mlx_log = log
    proc._rapid_mlx_log_path = log_path
    return proc, base_url


def _wait_for_chat_server(base_url: str, proc, timeout_s: int = 600) -> None:
    """Block until /health/ready returns 200, the proc exits, or timeout.

    On a TTY, draws a spinner + elapsed-seconds counter to stderr so the
    user can see the chat REPL is alive while the spawned server loads
    weights (typically 20-90 s for 4-30 B models on Apple Silicon). The
    line is erased before this function returns so the caller's next
    print lands on a clean line.
    """
    import time

    import requests

    is_tty = sys.stderr.isatty()
    spinner = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    cyan = "\x1b[36m" if is_tty else ""
    dim = "\x1b[2m" if is_tty else ""
    reset = "\x1b[0m" if is_tty else ""
    start = time.monotonic()
    deadline = start + timeout_s
    tick = 0

    def _draw():
        if not is_tty:
            return
        elapsed = int(time.monotonic() - start)
        ch = spinner[tick % len(spinner)]
        sys.stderr.write(
            f"\r  {cyan}{ch}{reset} loading model ... {dim}{elapsed}s{reset}"
        )
        sys.stderr.flush()

    def _clear():
        if not is_tty:
            return
        sys.stderr.write("\r" + " " * 40 + "\r")
        sys.stderr.flush()

    try:
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(
                    f"server exited early (code {proc.returncode}); "
                    "see chat-server.log for details"
                )
            # Animate the spinner at 10 fps; only poll /health once a
            # second to keep the spinner smooth and the network polite.
            if tick % 10 == 0:
                try:
                    r = requests.get(f"{base_url}/health/ready", timeout=2)
                    if r.status_code == 200:
                        return
                except requests.RequestException:
                    pass
            _draw()
            time.sleep(0.1)
            tick += 1
    finally:
        _clear()
    raise TimeoutError(
        f"server did not become ready within {timeout_s}s "
        "(large models can take longer — pass --ready-timeout)"
    )


def _has_short_pattern_dominating_suffix(
    text: str,
    *,
    window: int = 600,
    max_period: int = 300,
) -> bool:
    """Return True if the trailing ``window`` chars of ``text`` are
    periodic with a cycle length ≤``max_period``.

    Catches the degenerate-model cases the rolling whitespace-token
    counter in ``_stream_chat_response`` misses:

    - ``"BarleyBarleyBarley..."`` (no whitespace separator) — the entire
      suffix collapses to a single ``str.split()`` token whose count
      never increments. Real qwen3.5-4b-4bit regression surfaced in the
      0.6.28 onboarding test.
    - Long-cycle phrase loops, e.g. a ~280-char clause that repeats
      verbatim until ``max_tokens``. Surfaced when asked "describe the
      entire history of the Roman Empire in one long unbroken sentence".

    Implementation: compute the KMP failure function over the trailing
    window. The smallest period of the *entire* window is
    ``len(s) - fail[-1]``; a short period (≤``max_period``) means the
    window is dominated by that repetition starting from offset 0.

    Note: KMP itself does NOT detect periods that begin mid-window
    (rotated patterns). Mid-window degeneracy gets caught because this
    helper is invoked after every streaming chunk — once the model has
    been looping long enough to fill the window, the rolling 600-char
    suffix aligns with the pattern and the smallest-period check fires.
    A pure end-of-stream check would miss rotated cases.

    Cost is ``O(window)`` time and memory per call regardless of
    pattern length (the failure-function array is allocated each
    invocation) — much cheaper than the prior
    ``O(window * pattern_max_len)`` anchored scan, and cheap enough
    to run on every streaming chunk.

    The defaults (window=600, max_period=300) leave room for legitimate
    repetitive content like ``[0, 0, 0, ...]`` lists shorter than the
    window. *Long* lists of truly identical values the user explicitly
    asked for will get cut — a user hitting that false positive can
    ``/reset`` and rephrase. The cost of NOT cutting genuine model
    degeneracy (2000+ tokens of garbage) is far higher.
    """
    if len(text) < window:
        return False
    tail = text[-window:]
    n = len(tail)
    # KMP failure function: ``fail[i]`` = longest proper prefix of
    # ``tail[: i + 1]`` that is also a suffix.
    fail = [0] * n
    for i in range(1, n):
        j = fail[i - 1]
        while j > 0 and tail[i] != tail[j]:
            j = fail[j - 1]
        if tail[i] == tail[j]:
            j += 1
        fail[i] = j
    # Smallest period of ``tail``. Always >= 1 (fail[-1] <= n-1, since
    # ``fail`` is the longest *proper* prefix-suffix). ``period == n``
    # means no nontrivial period — the entire window is its own only
    # period and content is aperiodic. Defaults guarantee
    # ``max_period < window`` so this case never trips, but a caller
    # with ``max_period >= window`` would otherwise see aperiodic
    # strings flagged. Explicit ``period < n`` guard locks the contract.
    period = n - fail[-1]
    return period < n and period <= max_period


def _stream_chat_response(
    base_url: str,
    payload: dict,
    timeout_s: int,
    metrics: dict | None = None,
) -> str:
    """POST /v1/chat/completions with stream=True and print tokens as they
    arrive. Returns the full assistant content (concatenated content deltas).

    Reasoning-content deltas (Qwen3, DeepSeek-R1, etc.) are streamed to stdout
    in dim ANSI so the user sees thinking, but excluded from the returned
    string — chat history stores only the final answer, matching the
    OpenAI-compat split between ``content`` and ``reasoning_content``.

    Plain streaming: tokens land directly in the user's terminal as they
    arrive. We deliberately do NOT use ``rich.Live`` + ``Markdown`` here:
    Live re-renders the panel on every refresh and, when the console's
    cursor-overwrite path is unreliable (recordings, some terminal
    multiplexers), each refresh appends rather than overwrites — turning
    a 200-token response into a wall of repeated text. Live markdown
    rendering deserves a separate, more careful effort with explicit
    fallback detection; for now correctness wins over formatting.
    """
    import json

    import requests

    DIM = "\x1b[2m"
    BOLD = "\x1b[1m"
    RESET = "\x1b[0m"
    MAGENTA = "\x1b[35m"
    CYAN = "\x1b[36m"
    is_tty = sys.stdout.isatty() and "NO_COLOR" not in os.environ
    in_reasoning = False
    full = ""

    # ----- Streaming markdown colorer ------------------------------------
    # Body text streams in the terminal's default color (Claude-Code-style
    # — accents only on chrome). Inline coloring handles the markers users
    # see most often: ``\`code\``` (cyan), ``\`\`\`fence\`\`\``` (dim cyan
    # block), ``**bold**`` (ANSI bold), and ATX headers (``#`` … ``####``)
    # at line start. Lists / italic stay raw so the parser stays small.
    HEADING_STYLE = {
        1: BOLD + CYAN,  # `# h1`     — most prominent
        2: BOLD + MAGENTA,  # `## h2`    — secondary
        3: BOLD,  # `### h3`   — bold only
        4: CYAN,  # `#### h4`  — cyan
        5: MAGENTA,  # `##### h5` — magenta
        6: DIM,  # `###### h6`— dim
    }
    _state = {
        "in_fence": False,  # inside a ``` block
        "in_inline_code": False,  # inside a `code` span
        "in_bold": False,  # inside **bold**
        "in_heading": False,  # inside an ATX heading line
        "at_line_start": True,  # cursor is at start of a logical line
        "pending": "",  # buffered chars awaiting lookahead
    }

    def _emit_with_inline_md(piece: str) -> None:
        if not is_tty:
            sys.stdout.write(piece)
            sys.stdout.flush()
            return
        text = _state["pending"] + piece
        _state["pending"] = ""
        out: list[str] = []
        i, n = 0, len(text)
        while i < n:
            c = text[i]
            # Newline closes any line-scoped span (heading) and resets the
            # line-start anchor so the next `#`/`*`/etc. is interpreted in
            # the right context.
            if c == "\n":
                if _state["in_heading"]:
                    out.append(RESET)
                    _state["in_heading"] = False
                out.append("\n")
                _state["at_line_start"] = True
                i += 1
                continue
            # ATX heading: `#`..`######` followed by space at line start.
            # We skip this inside fences (a `#` at line start there is
            # almost always a comment, not a heading).
            if _state["at_line_start"] and c == "#" and not _state["in_fence"]:
                # Count consecutive `#` (1..6).
                j = i
                while j < n and j - i < 6 and text[j] == "#":
                    j += 1
                # Need to see one more char after the hashes to decide
                # heading vs literal "###foo" — buffer if we don't have it.
                if j == n:
                    _state["pending"] = text[i:]
                    break
                hashes = j - i
                if 1 <= hashes <= 6 and text[j] == " ":
                    style = HEADING_STYLE.get(hashes, BOLD)
                    out.append(style)
                    out.append(text[i : j + 1])  # emit "## "
                    _state["in_heading"] = True
                    _state["at_line_start"] = False
                    i = j + 1
                    continue
                # Not a heading — fall through to literal emission below.
            if c == "`":
                # Need 2 chars of lookahead to disambiguate ``` vs `.
                if i + 2 >= n:
                    _state["pending"] = text[i:]
                    break
                if text[i : i + 3] == "```":
                    if _state["in_fence"]:
                        out.append("```" + RESET)
                        _state["in_fence"] = False
                    else:
                        out.append(DIM + CYAN + "```")
                        _state["in_fence"] = True
                    _state["at_line_start"] = False
                    i += 3
                    continue
                # Single backtick.
                if _state["in_fence"]:
                    out.append("`")
                elif _state["in_inline_code"]:
                    out.append("`" + RESET)
                    _state["in_inline_code"] = False
                else:
                    out.append(CYAN + "`")
                    _state["in_inline_code"] = True
                _state["at_line_start"] = False
                i += 1
                continue
            if c == "*" and not _state["in_fence"] and not _state["in_inline_code"]:
                if i + 1 >= n:
                    _state["pending"] = text[i:]
                    break
                if text[i : i + 2] == "**":
                    if _state["in_bold"]:
                        out.append("**" + RESET)
                        _state["in_bold"] = False
                    else:
                        out.append(BOLD + "**")
                        _state["in_bold"] = True
                    _state["at_line_start"] = False
                    i += 2
                    continue
            out.append(c)
            # Whitespace (other than newline, handled above) keeps the
            # line-start anchor true so leading-indent headings still
            # parse — e.g., a list item's child paragraph is rare here.
            if c not in " \t":
                _state["at_line_start"] = False
            i += 1
        sys.stdout.write("".join(out))
        sys.stdout.flush()

    def _close_open_md_spans() -> None:
        if is_tty and (
            _state["in_fence"]
            or _state["in_inline_code"]
            or _state["in_bold"]
            or _state["in_heading"]
        ):
            sys.stdout.write(RESET)
            sys.stdout.flush()
        if _state["pending"]:
            sys.stdout.write(_state["pending"])
            sys.stdout.flush()
            _state["pending"] = ""

    # ----- Repetition guard ----------------------------------------------
    # Models occasionally degenerate into the same token repeated until
    # max_tokens — filling the screen with "Barley Barley Barley...".
    # Two complementary checks run per delta:
    #
    # 1. Whitespace-token-consecutive: the SAME whitespace-split token
    #    repeats ≥``REPEAT_LIMIT`` times in a row. O(1) rolling counter.
    #    Catches the common form ``"Barley Barley Barley..."``. Earlier
    #    guards used "≤2 unique in last 30" but fired on legit content
    #    like ``[0, 0, 0, ...]`` and markdown table separators, so the
    #    bar is now stricter.
    #
    # 2. Character-level pattern check (``_has_short_pattern_dominating_
    #    suffix``): the trailing window is dominated by a short repeating
    #    pattern. Catches the form ``"BarleyBarleyBarley..."`` (no
    #    whitespace separator), where ``piece.split()`` produces one
    #    giant token whose count never increments — this was a real
    #    qwen3.5-4b-4bit regression in 0.6.28 (issue surfaced post-release).
    REPEAT_LIMIT = 25
    repeat_last: str | None = None
    repeat_run = 0
    repetition_aborted = False

    with requests.post(
        f"{base_url}/v1/chat/completions",
        json=payload,
        stream=True,
        timeout=timeout_s,
    ) as resp:
        if resp.status_code != 200:
            # With stream=True the body may still be partial / mid-chunk when
            # the server closed the socket; read defensively so we surface a
            # useful HTTP code instead of a ChunkedEncodingError.
            try:
                body = resp.text[:500]
            except Exception:
                body = "(no body)"
            raise RuntimeError(f"HTTP {resp.status_code}: {body}")
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            # When the caller passes ``stream_options.include_usage``,
            # the server emits a final chunk with empty choices and a
            # populated ``usage`` block. Capture it for the speed line.
            usage = chunk.get("usage")
            if usage and metrics is not None:
                metrics["completion_tokens"] = usage.get("completion_tokens")
                metrics["prompt_tokens"] = usage.get("prompt_tokens")
            # The usage-only final chunk has ``choices=[]``; guard
            # against an IndexError there.
            choices = chunk.get("choices") or []
            delta = choices[0].get("delta", {}) if choices else {}
            # ``finish_reason`` arrives on the last token chunk (after
            # which the server may still emit a usage-only chunk).
            # Capture the most recent non-null value so the caller can
            # surface a "length" warning if the answer was truncated.
            if choices and metrics is not None:
                fr = choices[0].get("finish_reason")
                if fr is not None:
                    metrics["finish_reason"] = fr
            reasoning = delta.get("reasoning_content")
            piece = delta.get("content")
            if reasoning:
                if not in_reasoning:
                    if is_tty:
                        sys.stdout.write(f"{MAGENTA}[thinking]{RESET} {DIM}")
                    else:
                        sys.stdout.write("[thinking] ")
                    in_reasoning = True
                sys.stdout.write(reasoning)
                sys.stdout.flush()
            if piece:
                if in_reasoning:
                    sys.stdout.write(f"{RESET}\n  " if is_tty else "\n")
                    in_reasoning = False
                # Detect repetition BEFORE emitting. If a single coalesced
                # delta contains the cutoff inside it (server batched many
                # repeated tokens into one chunk), find the position and
                # only emit the prefix up to that token — otherwise the
                # user sees the full degenerate dump before the abort
                # message lands.
                #
                # Rolling counter: each new whitespace-separated token in
                # this delta either extends the current consecutive run
                # or resets it. Aborts only on a single token repeated
                # ``REPEAT_LIMIT`` times in a row, not on diverse-but-
                # repetitive content like ``[0, 0, 0, ...]`` or markdown
                # tables.
                cutoff_idx: int | None = None
                tokens = piece.split()
                for i, tok in enumerate(tokens):
                    if tok == repeat_last:
                        repeat_run += 1
                    else:
                        repeat_last = tok
                        repeat_run = 1
                    if repeat_run >= REPEAT_LIMIT:
                        repetition_aborted = True
                        cutoff_idx = i
                        break
                if cutoff_idx is not None:
                    # Find the byte position in ``piece`` corresponding to
                    # the start of the cutoff token, so we can emit only
                    # the prefix. ``str.split()`` collapses runs of
                    # whitespace, so we walk the original text token-by-
                    # token to recover the offset.
                    pos = 0
                    seen = 0
                    while seen < cutoff_idx and pos < len(piece):
                        # Skip leading whitespace.
                        while pos < len(piece) and piece[pos].isspace():
                            pos += 1
                        # Skip the token itself.
                        while pos < len(piece) and not piece[pos].isspace():
                            pos += 1
                        seen += 1
                    prefix = piece[:pos]
                    if prefix:
                        _emit_with_inline_md(prefix)
                        full += prefix
                else:
                    _emit_with_inline_md(piece)
                    full += piece
                # Char-level guard: catches no-whitespace degenerate
                # output like ``"BarleyBarleyBarley..."`` that the
                # whitespace-token counter misses (the entire chunk
                # collapses to one giant token whose consecutive count
                # never climbs). Cheap enough to run on every chunk.
                #
                # Trade-off: runs *after* the chunk is already emitted,
                # so the user sees one extra chunk of garbage before
                # the abort message lands. We accept this — slicing
                # mid-chunk would require re-running KMP per byte (or
                # binary search) on every delta, and degenerate chunks
                # are typically small (≤64 chars) since servers stream
                # token-by-token.
                if not repetition_aborted and _has_short_pattern_dominating_suffix(
                    full
                ):
                    repetition_aborted = True
                if repetition_aborted:
                    break
    _close_open_md_spans()
    if in_reasoning and is_tty:
        sys.stdout.write(RESET)
        sys.stdout.flush()
    if repetition_aborted:
        msg = (
            f"\n\n  {DIM}(response cut: model began repeating itself — "
            f"try /reset or a larger model){RESET}"
            if is_tty
            else "\n\n(response cut: repetition detected)"
        )
        sys.stdout.write(msg)
        sys.stdout.flush()
    return full


def chat_command(args):
    """Interactive REPL chat with a model.

    Spawns a local `serve` on an ephemeral port (or connects to an existing
    server via --base-url / --port), then loops stdin → /v1/chat/completions
    (streaming) → stdout. Maintains multi-turn history; `/reset` clears it.
    Exits cleanly on Ctrl-D, Ctrl-C, or `exit` / `quit`.
    """
    import atexit
    import signal
    import subprocess
    import tempfile

    base_url: str
    proc = None
    log_path: str | None = None
    # Tracks every spawned server (initial + every /model candidate) so
    # the SIGTERM/atexit cleanup tears down in-flight candidates too —
    # not just the bound ``proc``. A SIGTERM landing while a /model
    # swap is mid-spawn would otherwise orphan the candidate server.
    _active_procs: list[subprocess.Popen] = []

    # TTY-gated ANSI palette for the chat UI. NO_COLOR is honoured.
    _is_tty = sys.stdout.isatty() and "NO_COLOR" not in os.environ
    BOLD = "\x1b[1m" if _is_tty else ""
    DIM = "\x1b[2m" if _is_tty else ""
    GREEN = "\x1b[32m" if _is_tty else ""
    CYAN = "\x1b[36m" if _is_tty else ""
    YELLOW = "\x1b[33m" if _is_tty else ""
    RED = "\x1b[31m" if _is_tty else ""
    RESET = "\x1b[0m" if _is_tty else ""

    def _teardown_proc(p) -> None:
        """Terminate a spawned chat server and free its log file.

        Used by `_cleanup` (process exit) and `_switch_model` (mid-
        session swap). Idempotent — safe to call when the proc has
        already exited or never existed. Also reaps the killed child
        with wait(timeout=1) so repeated /model swaps don't leave
        zombies until the parent exits.
        """
        if p is None:
            return
        try:
            if p.poll() is None:
                try:
                    p.terminate()
                    p.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    try:
                        p.kill()
                        # Reap the SIGKILL'd child — without this,
                        # repeated /model swaps stack zombie entries.
                        try:
                            p.wait(timeout=1)
                        except subprocess.TimeoutExpired:
                            pass
                    except (ProcessLookupError, OSError):
                        pass
                except (ProcessLookupError, OSError):
                    pass
        finally:
            # Drop from the tracked set so a subsequent _cleanup walk
            # doesn't double-tear it down.
            try:
                _active_procs.remove(p)
            except ValueError:
                pass
            # Close the log handle and reap the tempfile so /model
            # swaps don't leak FDs. Both attributes set by
            # _spawn_chat_server.
            #
            # Log file unlink policy: zero-byte logs (no server output
            # ever flushed — typical for a clean spawn that never logged
            # a warning) are unlinked; non-empty logs are LEFT IN PLACE
            # so a user investigating a crash or post-mortem error still
            # has the server's stderr to look at. Previously every log
            # was unlinked, which scrubbed useful debugging breadcrumbs
            # along with the noise.
            fh = getattr(p, "_rapid_mlx_log", None)
            if fh is not None:
                try:
                    fh.close()
                except OSError:
                    pass
            lp = getattr(p, "_rapid_mlx_log_path", None)
            if lp:
                try:
                    size = os.path.getsize(lp)
                except OSError:
                    size = -1  # treat unknown as "leave alone"
                if size == 0:
                    try:
                        os.unlink(lp)
                    except FileNotFoundError:
                        pass
                    except OSError:
                        pass

    # Guard against re-entry: ``_cleanup`` is registered once with
    # ``atexit`` AND fired from the SIGTERM handler. Without an idempotent
    # check, a SIGTERM during shutdown would walk ``_active_procs``,
    # _teardown_proc would empty it, then atexit's invocation would walk
    # an empty list — harmless today, but the explicit flag keeps the
    # contract obvious and survives future helpers that read the list
    # before iterating.
    _cleanup_state = {"done": False}

    def _cleanup():
        # Walk every tracked proc — covers the active server and any
        # in-flight /model candidate. Iterate over a snapshot since
        # _teardown_proc mutates _active_procs. Idempotent: a second call
        # short-circuits so atexit + SIGTERM-handler ordering doesn't
        # matter.
        if _cleanup_state["done"]:
            return
        # Mask BOTH SIGTERM and SIGINT for the duration of the loop.
        # Codex round-3 BLOCKING #1: with only SIGTERM masked, a SIGINT
        # landing mid-teardown raises KeyboardInterrupt, unwinds the
        # for-loop, the surrounding ``finally`` issues ``sys.exit(143)``,
        # and atexit's later call sees ``done=True`` (set at function
        # entry, original implementation) → procs after the interrupted
        # one get orphaned. Move the ``done`` flag to AFTER the loop AND
        # mask SIGINT so a Ctrl-C-during-cleanup can't kill the unwind.
        _prev_term = _prev_int = None
        try:
            _prev_term = signal.signal(signal.SIGTERM, signal.SIG_IGN)
        except (ValueError, OSError):
            pass
        try:
            _prev_int = signal.signal(signal.SIGINT, signal.SIG_IGN)
        except (ValueError, OSError):
            pass
        try:
            for p in list(_active_procs):
                _teardown_proc(p)
            _cleanup_state["done"] = True
        finally:
            # Best-effort restore so post-cleanup signals route normally.
            # If restore raises, swallow — we're about to exit anyway.
            for signum, prev in (
                (signal.SIGTERM, _prev_term),
                (signal.SIGINT, _prev_int),
            ):
                if prev is not None:
                    try:
                        signal.signal(signum, prev)
                    except (ValueError, OSError):
                        pass

    # Install SIGTERM handler + atexit BEFORE any spawn. Otherwise a
    # SIGTERM landing in the window between `Popen()` and `signal.signal`
    # uses Python's default handler (calls `_exit`, skips atexit) and
    # orphans the spawned server. SIGINT is *deliberately* left on the
    # default handler so Ctrl-C unblocks ``input()`` via the natural
    # KeyboardInterrupt path, the REPL loop's ``except
    # KeyboardInterrupt: break`` fires, and atexit runs ``_cleanup``.
    # On non-tty stdin (piped input) the SIGINT path is never exercised,
    # so the SIGTERM + atexit pair is what reaps the spawned server.
    #
    # Re-entry: a second SIGTERM landing mid-cleanup (common from process
    # supervisors that escalate after a short grace period) would
    # otherwise call _cleanup again — _teardown_proc's
    # ``proc.terminate() + proc.wait(timeout=5)`` would block while the
    # outer cleanup is still mid-wait, leaving the child orphaned.
    # ``_cleanup`` masks both SIGTERM and SIGINT internally for the
    # duration of its teardown loop (Codex round-3 BLOCKING #1), so the
    # handler here only needs to drive the lifecycle: cleanup → exit.
    # The try/finally guarantees sys.exit fires even if _teardown_proc
    # raises (rare — only on the secondary proc.kill() escalation).
    def _sigterm_handler(*_):
        try:
            _cleanup()
        finally:
            sys.exit(143)

    try:
        signal.signal(signal.SIGTERM, _sigterm_handler)
    except (ValueError, OSError):
        pass
    atexit.register(_cleanup)

    if args.base_url:
        base_url = args.base_url.rstrip("/")
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]
    elif args.port is not None:
        # Pre-flight probe: a valid-range but unbound port previously
        # dropped the user into the REPL and only failed on the first
        # message with a raw HTTPConnectionPool stack trace. Probe once
        # with a 1 s timeout so the failure is friendly + actionable.
        import socket as _socket

        try:
            with _socket.create_connection(("127.0.0.1", args.port), timeout=1):
                pass
        except OSError:
            # OSError covers ConnectionRefusedError + socket.timeout
            # (which is an alias for ``TimeoutError`` in Python 3.10+).
            print(
                f"\n  {RED}Error:{RESET} no rapid-mlx server reachable at "
                f"127.0.0.1:{args.port}."
            )
            print(f"    Start one with: rapid-mlx serve <alias> --port {args.port}")
            print("    Or omit --port to spawn one automatically.")
            sys.exit(1)
        base_url = f"http://127.0.0.1:{args.port}"
    else:
        # Pre-download in the foreground so the HF tqdm progress bar lands
        # in the user's terminal. Otherwise the serve subprocess swallows
        # the bar into the log file and `rapid-mlx chat` looks frozen for
        # several minutes on first run with a fresh model.
        _ensure_model_downloaded(args.model)

        log_path = tempfile.NamedTemporaryFile(
            prefix="rapid-mlx-chat-", suffix=".log", delete=False
        ).name
        print(f"\n  Starting server {DIM}(log: {log_path}){RESET} ...")
        # If main() resolved an alias, expose the alias as the API model name
        # so the chat request body matches what the user typed.
        original = getattr(args, "_original_alias", None)
        proc, base_url = _spawn_chat_server(
            args.model,
            log_path,
            served_name=original,
            register_in=_active_procs,
        )

        try:
            _wait_for_chat_server(base_url, proc, timeout_s=args.ready_timeout)
        except (RuntimeError, TimeoutError) as e:
            print(f"\n  {RED}Failed to start server:{RESET} {e}")
            sys.exit(1)
        print(f"  {GREEN}✓ Ready.{RESET}\n")

    from vllm_mlx._version_check import print_staleness_warning_if_any

    print_staleness_warning_if_any()

    # Resolve ``--max-tokens``. Default is None at the argparse layer so
    # we can distinguish "user did not pass it" from "user passed 2048
    # explicitly". When ``--think`` is set and the user did not supply a
    # value, raise the default from 2048 to 4096 so the reasoning trace +
    # final answer both fit (the round-1 finding: ``chat qwen3.5-4b-4bit
    # --think`` filled the 2048 budget with reasoning and emitted an
    # empty answer with ``finish_reason='length'``).
    user_passed_max_tokens = args.max_tokens is not None
    if args.max_tokens is None:
        args.max_tokens = 4096 if args.think else 2048
    if args.think and not user_passed_max_tokens:
        print(
            f"  {DIM}(--think on; raised --max-tokens to {args.max_tokens} — "
            f"pass --max-tokens to override){RESET}"
        )

    print(
        f"  🐆 {BOLD}Chat{RESET} — "
        f"{DIM}type {RESET}{BOLD}/help{RESET}{DIM} for commands, "
        f"Ctrl-D to exit.{RESET}"
    )
    # First-launch-only banner for the agents/codex tip. The marker-file
    # gate keeps the tip from re-appearing on every chat launch (persona-3
    # finding: irritating by launch #50). Marker logic is skipped entirely
    # when stdout is not a TTY or NO_COLOR is set — pipe/CI runs shouldn't
    # pollute the user's config dir, and the banner is fluff there anyway.
    _is_pipe_or_no_color = (not sys.stdout.isatty()) or ("NO_COLOR" in os.environ)
    if not _is_pipe_or_no_color and not _has_seen_tip("chat_intro_codex"):
        print(
            f"  {DIM}For a Claude Code-like TUI: `rapid-mlx agents codex --setup`, "
            f"then run `codex` in any project.{RESET}\n"
        )
        _mark_tip_seen("chat_intro_codex")
    else:
        # Maintain the existing blank-line spacing the banner used to
        # provide — keeps the prompt layout consistent across runs.
        print()

    served_name = getattr(args, "_original_alias", args.model)
    messages: list[dict] = []
    if args.system:
        messages.append({"role": "system", "content": args.system})

    # The rapid-mlx server's ChatCompletionRequest exposes a top-level
    # ``enable_thinking`` field — ``chat_template_kwargs`` is not a recognized
    # request field and would be silently dropped.
    #
    # Default thinking OFF in the REPL. Reasoning models (Qwen3.5/3.6, etc.)
    # otherwise emit raw chain-of-thought to stdout AND, on the default
    # qwen3.5-4b-4bit model, degenerate into infinite repetition until max-tokens
    # truncates the response — producing zero usable output for a brand-new
    # user. ``--think`` opts back in for users who explicitly want to see
    # reasoning traces; ``--no-think`` is preserved as the legacy form.
    extra: dict = {}
    if not args.think:
        extra["enable_thinking"] = False

    import time

    import requests

    # Importing ``readline`` upgrades the built-in ``input()`` so that
    # the arrow keys recall earlier prompts (and Ctrl-A/E/U/R work).
    # The module is stdlib on macOS/Linux; on Windows it doesn't exist
    # and we fall back to plain input(). When readline IS available we
    # need to wrap the colored prompt's ANSI escapes in \001/\002 so
    # readline's column counter doesn't include the invisible bytes —
    # otherwise long history entries wrap incorrectly and Ctrl-A jumps
    # to the wrong column (especially on libedit-backed Apple system
    # python). The wrappers are no-op on a terminal, so it's safe to
    # always emit them when readline is loaded.
    have_readline = False
    try:
        import readline  # noqa: F401 — side-effect import

        have_readline = True
    except ImportError:
        pass

    def _wrap_invisible(esc: str) -> str:
        if have_readline and esc:
            return "\001" + esc + "\002"
        return esc

    if _is_tty:
        prompt = _wrap_invisible(BOLD + CYAN) + ">" + _wrap_invisible(RESET) + " "
        cont_prompt = _wrap_invisible(DIM) + "…" + _wrap_invisible(RESET) + " "
    else:
        prompt = "> "
        cont_prompt = "… "

    def _print_help():
        print(
            f"\n  {BOLD}Slash commands{RESET}\n"
            f"    {BOLD}/help{RESET}, {BOLD}/?{RESET}          show this help\n"
            f"    {BOLD}/reset{RESET}, {BOLD}/clear{RESET}     clear conversation history\n"
            f"    {BOLD}/model <alias>{RESET}     switch model "
            f"{DIM}(restarts the server, resets history){RESET}\n"
            f"    {BOLD}/save <path>{RESET}       save conversation to a markdown file\n"
            f"    {BOLD}/exit{RESET}, {BOLD}/quit{RESET}, {BOLD}/bye{RESET}    "
            f"exit chat {DIM}(or Ctrl-D){RESET}\n"
            f"\n  {BOLD}Multi-line input{RESET}\n"
            f'    type {BOLD}"""{RESET} on its own line to start, again to end '
            f"{DIM}(paste code blocks){RESET}\n"
            f"\n  {BOLD}Keys{RESET}\n"
            f"    {BOLD}Ctrl-C{RESET}             cancel the current response, "
            f"or exit at empty prompt\n"
            f"    {BOLD}Ctrl-D{RESET}             exit\n"
        )

    def _save_conversation(path_arg: str):
        # Refuse early on an empty conversation — otherwise we create a
        # near-empty file then lock the user out of the same path on
        # the next try (since exclusive-mode open refuses overwrite).
        non_system = [m for m in messages if m.get("role") != "system"]
        if not non_system:
            print(
                f"  {YELLOW}Nothing to save yet.{RESET} "
                f"{DIM}(send a chat turn first){RESET}\n"
            )
            return
        path = os.path.expanduser(path_arg)
        # Auto-create parent directories; otherwise users see a confusing
        # "No such file or directory" for /save logs/2026-05/convo.md.
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            try:
                os.makedirs(parent, exist_ok=True)
            except OSError as exc:
                print(f"  {RED}Save failed:{RESET} cannot create {parent}: {exc}\n")
                return
        try:
            # Mode "x" (O_CREAT | O_EXCL) is atomic — refuses if the path
            # already exists, with no TOCTOU window between exists() and
            # open() that an exists()-then-open("w") check has. Also
            # naturally rejects existing symlinks pointing elsewhere.
            with open(path, "x", encoding="utf-8") as f:
                f.write(f"# rapid-mlx chat — {served_name}\n\n")
                for m in messages:
                    if m["role"] == "system":
                        continue
                    f.write(f"## {m['role'].capitalize()}\n\n{m['content']}\n\n")
            print(f"  {GREEN}✓{RESET} Saved {len(messages)} messages to {path}\n")
        except FileExistsError:
            print(
                f"  {YELLOW}{path} already exists.{RESET} "
                f"{DIM}(/save won't overwrite — pick a different path){RESET}\n"
            )
        except IsADirectoryError:
            print(
                f"  {RED}Save failed:{RESET} {path} is a directory — "
                f"{DIM}give a file path, not a directory{RESET}\n"
            )
        except OSError as exc:
            print(f"  {RED}Save failed:{RESET} {exc}\n")

    def _read_multiline() -> str:
        lines: list[str] = []
        while True:
            try:
                more = input(cont_prompt)
            except (EOFError, KeyboardInterrupt):
                # Tell the user how many lines they're losing — silent
                # discard on Ctrl-C/Ctrl-D mid-paste is hostile.
                if lines:
                    print(
                        f"\n  {YELLOW}(multi-line cancelled — "
                        f"{len(lines)} line{'' if len(lines) == 1 else 's'} "
                        f"discarded){RESET}\n"
                    )
                else:
                    print(f"\n  {YELLOW}(multi-line cancelled){RESET}\n")
                return ""
            if more.rstrip() == '"""':
                # Preserve leading/trailing whitespace verbatim — the
                # heredoc is meant for code paste, where stripping
                # indentation actively corrupts the input.
                return "\n".join(lines)
            lines.append(more)

    def _switch_model(new_alias: str) -> None:
        """Hot-swap the spawned chat server to a new model alias.

        Order matters: validate + pre-download the new model BEFORE
        terminating the old one. If anything fails (bogus alias, disk
        gate, network), the old server stays running and the REPL is
        usable. Only when the new model is on-disk and the new server is
        spawn-ready do we tear down the old proc and rebind.
        """
        nonlocal proc, base_url, log_path, served_name, messages
        if proc is None:
            print(
                f"  {YELLOW}/model is only available when chat spawns its "
                f"own server (not with --base-url / --port).{RESET}\n"
            )
            return
        from vllm_mlx.model_aliases import resolve_model

        resolved = resolve_model(new_alias) or new_alias
        print(f"  {DIM}Preparing {new_alias} → {resolved} ...{RESET}")

        # 1a. Gate before download: the main() entry-point gate only
        #     fires on the CLI invocation, so an uncached /model swap
        #     would otherwise start a 40+ GB pull with no prompt.
        #     Mirror main()'s cheap env/TTY short-circuit so we don't
        #     pay the 5-second HF metadata round-trip on every /model
        #     swap when the user opted into AUTO_PULL or is on non-TTY
        #     stdin. ``confirm_or_abort`` self-skips again internally
        #     but skipping ``estimate_repo_size_bytes`` saves the wait.
        if "/" in resolved and not os.path.exists(resolved):
            _env_val = os.environ.get("RAPID_MLX_AUTO_PULL", "").strip().lower()
            _auto_yes = _env_val in {"1", "true", "yes"}
            _interactive = sys.stdin.isatty()
            if not _auto_yes and _interactive:
                from vllm_mlx._download_gate import (
                    confirm_or_abort,
                    estimate_repo_size_bytes,
                    is_repo_cached,
                )

                if not is_repo_cached(resolved):
                    try:
                        confirm_or_abort(
                            resolved,
                            estimate_repo_size_bytes(resolved),
                        )
                    except SystemExit:
                        # User said no — keep the current server up.
                        print(
                            f"  {YELLOW}Model switch cancelled{RESET} "
                            f"{DIM}(previous server still running).{RESET}\n"
                        )
                        return

        # 1. Pre-download the new model (this also runs the disk-space
        #    gate). The current server keeps running while we do this so
        #    a download failure leaves the user where they were.
        try:
            _ensure_model_downloaded(resolved)
        except SystemExit:
            # Disk gate aborted via sys.exit(1); old server is untouched.
            print(
                f"  {RED}Model switch aborted{RESET} "
                f"{DIM}(disk gate); previous server still running.{RESET}\n"
            )
            return
        except RuntimeError as exc:
            # Definitive 404 from HF; old server stays.
            print(
                f"  {RED}Model switch aborted:{RESET} {exc}  "
                f"{DIM}(previous server still running){RESET}\n"
            )
            return

        # 2. Allocate a new log file and spawn the new server. We don't
        #    tear down the old one yet; we want a working candidate
        #    before we commit.
        new_log_path = tempfile.NamedTemporaryFile(
            prefix="rapid-mlx-chat-", suffix=".log", delete=False
        ).name
        print(f"  Starting server {DIM}(log: {new_log_path}){RESET} ...")
        # ``register_in=_active_procs`` makes the candidate visible to
        # ``_cleanup`` *inside* ``_spawn_chat_server`` — before the
        # readiness wait, before any further Python statement runs in
        # this scope. A SIGTERM/Ctrl-C during the (possibly multi-second)
        # load tears the child down via the cleanup walk.
        new_proc, new_base_url = _spawn_chat_server(
            resolved,
            new_log_path,
            served_name=new_alias,
            register_in=_active_procs,
        )
        try:
            _wait_for_chat_server(new_base_url, new_proc, timeout_s=args.ready_timeout)
        except (RuntimeError, TimeoutError) as exc:
            print(
                f"  {RED}Failed to start new server:{RESET} {exc}  "
                f"{DIM}(previous server still running){RESET}\n"
            )
            # Roll back: tear down the half-spawned new proc + free its
            # log file. The old proc/base_url/log_path stay bound.
            _teardown_proc(new_proc)
            return

        # 3. New server is healthy — commit. Rebind ``proc`` BEFORE
        #    tearing down the old one so a SIGTERM during teardown
        #    walks the new (still-running) proc, not just a freshly
        #    killed corpse.
        old_proc = proc
        proc = new_proc
        base_url = new_base_url
        log_path = new_log_path
        served_name = new_alias
        messages = [{"role": "system", "content": args.system}] if args.system else []
        _teardown_proc(old_proc)
        print(
            f"  {GREEN}✓ Switched to {new_alias}.{RESET} "
            f"{DIM}(history cleared){RESET}\n"
        )

    while True:
        try:
            line = input(prompt).rstrip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        # Heredoc-pasted content must NEVER be dispatched as a slash
        # command — a markdown doc whose first line starts with `/path`
        # or whose content includes `/save` would otherwise be silently
        # eaten by the slash dispatcher. Track the source so we know.
        is_heredoc = False
        if line == '"""':
            line = _read_multiline()
            if not line:
                continue
            is_heredoc = True
        if not is_heredoc:
            # Parse the leading word as the command and dispatch on
            # *exact* match. ``startswith("/save")`` would otherwise treat
            # ``/savefoo`` as ``/save`` (with arg ``foo``), silently
            # writing a file from a typo. Same for ``/modelfoo``.
            # ``str.split(maxsplit=1)`` (no separator arg) splits on any
            # whitespace, so ``/save\tpath.md`` works the same as
            # ``/save path.md``.
            parts = line.split(maxsplit=1)
            cmd = parts[0] if parts else ""
            rest = parts[1].strip() if len(parts) > 1 else ""
            # ``/bye`` is an Ollama-muscle-memory alias for ``/exit`` /
            # ``/quit``. ``/?`` mirrors ``/help`` and was already
            # supported; both alias sets are advertised in ``/help``.
            if cmd in ("exit", "quit", "/exit", "/quit", "/bye"):
                break
            if cmd in ("/help", "/?"):
                _print_help()
                continue
            if cmd in ("/reset", "/clear"):
                messages = (
                    [{"role": "system", "content": args.system}] if args.system else []
                )
                print(f"  {DIM}(history cleared){RESET}\n")
                continue
            if cmd == "/save":
                if not rest:
                    print(f"  {YELLOW}Usage: /save <path>{RESET}\n")
                else:
                    _save_conversation(rest)
                continue
            if cmd == "/model":
                if not rest:
                    print(
                        f"  {YELLOW}Usage: /model <alias>{RESET}  "
                        f"{DIM}(see `rapid-mlx models`){RESET}\n"
                    )
                else:
                    _switch_model(rest)
                continue
            if cmd.startswith("/"):
                print(
                    f"  {YELLOW}Unknown command: {cmd}{RESET}  "
                    f"{DIM}(type /help){RESET}\n"
                )
                continue

        messages.append({"role": "user", "content": line})
        payload = {
            "model": served_name,
            "messages": messages,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
            **extra,
        }
        # Claude-Code-style turn marker: a colored bullet introduces the
        # assistant's response so the user can visually scan turn
        # boundaries when scrolling back through long conversations.
        sys.stdout.write(f"\n  {CYAN}●{RESET} ")
        sys.stdout.flush()
        metrics: dict = {}
        start_t = time.monotonic()
        try:
            assistant = _stream_chat_response(
                base_url,
                payload,
                timeout_s=args.response_timeout,
                metrics=metrics,
            )
        except KeyboardInterrupt:
            print(f"\n  {YELLOW}(response interrupted){RESET}\n")
            messages.pop()
            continue
        except RuntimeError as e:
            print(f"\n  {RED}{e}{RESET}\n")
            messages.pop()
            continue
        except requests.RequestException as e:
            # Connection refused, timeout, dropped midstream — keep the REPL
            # alive and roll back the failed user turn so the next request
            # doesn't carry a dangling user role with no assistant reply.
            print(f"\n  {RED}Request failed:{RESET} {e}\n")
            messages.pop()
            continue
        elapsed = time.monotonic() - start_t
        # Speed line: prefer server-reported usage, fall back to a rough
        # 4-chars-per-token estimate when the server doesn't ship usage
        # in the stream.
        tokens = metrics.get("completion_tokens")
        if not tokens:
            tokens = max(1, len(assistant) // 4)
            tokens_label = f"~{tokens}"
        else:
            tokens_label = str(tokens)
        if assistant and elapsed > 0:
            tps = tokens / elapsed
            print(
                f"\n  {DIM}{tokens_label} tok · {elapsed:.1f}s · "
                f"{tps:.0f} tok/s{RESET}\n"
            )
        else:
            print()
        # Length-cut + empty-content warning. When the server stops
        # because ``finish_reason == "length"`` AND no visible content
        # was streamed (only reasoning), the user otherwise sees an
        # empty bullet and has no signal that the budget was the
        # problem. This is the round-1 ``--think`` regression: 2048-
        # token budget filled by reasoning on small models, zero answer.
        if metrics.get("finish_reason") == "length" and not assistant:
            print(
                f"  {YELLOW}(reasoning consumed the full --max-tokens "
                f"budget; bump --max-tokens for a final answer){RESET}\n"
            )
        if assistant:
            messages.append({"role": "assistant", "content": assistant})
        else:
            messages.pop()


def info_command(args):
    """Print the per-model profile for a model name or alias.

    Stage 1 (regex match) only — does NOT load the model, so this is fast
    and works without weights. Stage 2 (ArraysCache probe) is skipped.
    """
    from vllm_mlx.model_aliases import resolve_model, resolve_profile
    from vllm_mlx.model_auto_config import (
        detect_model_config,
        format_profile_table,
    )

    # ``main()`` (cli.py:~3400) pre-resolves ``args.model`` from alias →
    # HF path before dispatch, stashing the user-typed alias on
    # ``args._original_alias``. Pull from that first so DFlash
    # eligibility (alias-keyed) and the start-command hint render with
    # the alias the user actually typed, not the resolved HF repo.
    original_alias = getattr(args, "_original_alias", None) or args.model
    name = args.model
    resolved = (
        resolve_model(name) if not getattr(args, "_original_alias", None) else None
    )
    if resolved and resolved != name:
        print(f"  alias: {name} → {resolved}")
        name = resolved

    cfg = detect_model_config(name)
    print()
    print(format_profile_table(name, cfg))
    print()

    # DFlash eligibility — render the report so users can see which
    # gates pass/fail without consulting the docs. Skipped for unknown
    # models since AliasProfile is alias-keyed.
    profile = resolve_profile(original_alias)
    if profile is not None:
        _print_dflash_status(original_alias, profile)

    if cfg is None:
        print("  No pattern matched — runtime probe will run when the model loads.")
        print()


def _print_dflash_status(alias: str, profile) -> None:
    """Render a 3-row DFlash status block for ``rapid-mlx info <alias>``.

    Shows each gate (declared support / not MoE / not 4-bit / drafter
    present) so a user who tried ``--enable-dflash`` and got a vague
    error can see exactly which gate they're tripping.
    """
    from vllm_mlx.speculative.dflash.eligibility import (
        _looks_like_4bit,
        have_runtime,
        report,
    )

    r = report(profile, alias=alias)
    inner = 60
    sep = "─" * inner

    def _row(text: str) -> str:
        return f"│ {text:<{inner}} │"

    def _yes(ok: bool, msg_ok: str, msg_no: str) -> str:
        return ("✓ " + msg_ok) if ok else ("✗ " + msg_no)

    rows = [
        (
            "Declared support",
            _yes(profile.supports_dflash, "yes (supports_dflash=true)", "no"),
        ),
        ("Not MoE", _yes(not profile.is_moe, "yes (dense)", "no (MoE)")),
        (
            "Precision ≥8-bit",
            _yes(
                not _looks_like_4bit(profile.hf_path),
                "yes",
                "no (4-bit/mxfp4/nvfp4)",
            ),
        ),
        (
            "Drafter declared",
            _yes(
                bool(profile.dflash_draft_model),
                profile.dflash_draft_model or "yes",
                "no (dflash_draft_model unset)",
            ),
        ),
        (
            "mlx-vlm 0.5.0+",
            _yes(have_runtime(), "installed", "missing (need rapid-mlx[dflash])"),
        ),
    ]

    eligible = not r.reasons and have_runtime()
    summary = "✓ eligible" if eligible else "✗ ineligible"

    top = "┌" + "─" * (inner + 2) + "┐"
    bot = "└" + "─" * (inner + 2) + "┘"

    body = [top, _row(f"DFlash eligibility: {summary}"), _row(sep)]
    for k, v in rows:
        body.append(_row(f"{k:<18}: {v}"))
    body.append(bot)
    print("\n".join(body))
    print()
    if eligible:
        print(f"  Start with: rapid-mlx serve {alias} --enable-dflash")
        print()


def agents_command(args):
    """List, configure, and test agent integrations."""
    from vllm_mlx.agents import get_profile, list_profiles
    from vllm_mlx.agents.adapter import get_setup_instructions, setup_agent_config

    agent_name = args.agent_name
    base_url = args.base_url

    # No agent specified → list all profiles
    if not agent_name:
        profiles = list_profiles()
        print()
        print("  Supported AI Agents")
        print("  " + "─" * 56)
        for p in profiles:
            fc = "FC" if p.needs_function_calling else "  "
            stars = f"{p.stars // 1000}K" if p.stars and p.stars >= 1000 else ""
            if p.recommended_models:
                shown = p.recommended_models[:3]
                models = ", ".join(shown)
                if len(p.recommended_models) > 3:
                    models += f" +{len(p.recommended_models) - 3}"
            else:
                models = ""
            print(f"  {p.name:<15} {p.display_name:<20} {stars:>5}  [{fc}]  {models}")
        print()
        print(f"  {len(profiles)} agents supported")
        print("  Usage: rapid-mlx agents <name>          Show setup guide")
        print("         rapid-mlx agents <name> --setup   Auto-configure")
        print("         rapid-mlx agents <name> --test    Run integration tests")
        print()
        return

    # Get profile
    profile = get_profile(agent_name)
    if not profile:
        print(f"  Unknown agent: {agent_name}")
        print("  Run 'rapid-mlx agents' to see available agents.")
        sys.exit(1)

    # --test: run integration tests
    if args.test:
        from vllm_mlx.agents.testing import AgentTestRunner

        model_id = args.model or None
        runner = AgentTestRunner(
            profile,
            base_url=base_url,
            model_id=model_id,
            agent_version=args.agent_version,
        )
        if not runner._server_available():
            print(f"\n  Server not running at {base_url}")
            print("  Start it first: rapid-mlx serve <model>")
            sys.exit(1)

        report = runner.run()
        success = report.print_summary()
        sys.exit(0 if success else 1)

    # --setup: auto-configure agent
    if args.setup:
        # Detect model from running server
        model_id = args.model or "default"
        if model_id == "default":
            try:
                import httpx

                resp = httpx.get(f"{base_url}/models", timeout=3)
                model_id = resp.json()["data"][0]["id"]
            except Exception:
                pass

        summary = setup_agent_config(
            profile, base_url, model_id, agent_version=args.agent_version
        )
        print(f"\n  {profile.display_name} configured!")
        print(f"  {summary}")
        print()
        return

    # Default: show setup instructions
    # Pass "default" to trigger auto-detection of running model
    model_id = args.model or "default"
    instructions = get_setup_instructions(
        profile, base_url, model_id, agent_version=args.agent_version
    )
    print()
    print(instructions)
    print()


def upgrade_command(args):
    """Detect install method and (optionally) run the right upgrade command."""
    import subprocess

    from vllm_mlx._version_check import (
        _installed_version,
        _parse_version,
        detect_install_method,
        get_latest_version,
    )

    current = _installed_version() or "dev"
    print()
    print(f"  Current:  rapid-mlx {current}")

    latest = get_latest_version(force_refresh=True)
    if latest is None:
        print("  Latest:   (could not reach GitHub — check your network)\n")
        sys.exit(1)
    print(f"  Latest:   rapid-mlx {latest}")

    cur = _parse_version(current)
    lat = _parse_version(latest)
    if cur is not None and lat is not None and cur >= lat:
        print("\n  ✓ Already up to date.\n")
        return

    info = detect_install_method()
    print(f"  Install:  {info.method} ({info.binary_path or 'unknown path'})")
    print(f"  Command:  {info.upgrade_command}")
    print()

    if info.method == "unknown":
        print(
            "  Could not auto-detect install method — run the command above manually.\n"
        )
        return

    if args.yes:
        confirmed = True
    else:
        # Default Y — the user already typed the upgrade command;
        # punishing the Enter key with a no-op skip is bad UX. EOF on
        # stdin is treated as Enter (proceed), mirroring the download
        # gate. Ctrl-C is the only "skip" path — it returns silently
        # without ``sys.exit`` because upgrade is a leaf operation, so
        # there's nothing downstream to abort; cf. the gate, which
        # exits 1 because it's gatekeeping a multi-GB download.
        try:
            answer = input("  Run now? [Y/n] ").strip().lower()
        except EOFError:
            answer = ""
        except KeyboardInterrupt:
            print()
            return
        confirmed = answer not in {"n", "no"}

    if not confirmed:
        print("  Skipped — run the command above when ready.\n")
        return

    print()
    try:
        # Use argv form (shell=False) so paths with spaces in
        # ``sys.executable`` (or any other argv entry) can't be reinterpreted
        # as shell separators. install.sh's pipe is wrapped as ``bash -c``
        # in upgrade_argv, so we still get the pipe semantics it needs.
        result = subprocess.run(info.upgrade_argv, check=False)
    except KeyboardInterrupt:
        print("\n  Interrupted.\n")
        sys.exit(130)
    print()
    sys.exit(result.returncode)


def telemetry_command(args) -> None:
    """Manage anonymous usage telemetry — see Issue #236.

    Five actions: ``status`` / ``enable`` / ``disable`` / ``preview`` /
    ``reset``. Defaults to ``status`` when no action given so users can
    type ``rapid-mlx telemetry`` and immediately see what's set up.
    """
    # Imports kept inside the function so the telemetry package is only
    # loaded when actually needed — keeps `--help` and unrelated
    # subcommands cheap.
    import json

    from vllm_mlx import __version__ as rapid_mlx_version
    from vllm_mlx.telemetry import (
        consent_source,
        get_consent_state,
        get_or_create_client_id,
        is_enabled,
        record_consent,
        reset_state,
    )
    from vllm_mlx.telemetry.schema import sample_preview_payload
    from vllm_mlx.telemetry.state import client_id_path, consent_path

    action = getattr(args, "telemetry_action", None) or "status"
    cli_no = getattr(args, "no_telemetry", False)

    if action == "status":
        state = get_consent_state()
        print()
        print(
            f"  Telemetry: {'ENABLED' if is_enabled(cli_no_telemetry=cli_no) else 'disabled'}"
        )
        print(f"  Source:    {consent_source(cli_no_telemetry=cli_no)}")
        if state is not None:
            print(
                f"  Consent:   {state.consent} (recorded {state.prompted_at}, "
                f"by rapid-mlx {state.prompted_version})"
            )
        else:
            print("  Consent:   never prompted")
        print(f"  Files:     {consent_path()}")
        print(f"             {client_id_path()}")
        print()
        print("  Subcommands:  enable | disable | preview | reset")
        print()
        return

    if action == "enable":
        record_consent(True, rapid_mlx_version=rapid_mlx_version)
        # Generate the client_id eagerly so `preview` immediately after
        # has a real id to show.
        get_or_create_client_id()
        print()
        print("  Telemetry: ENABLED. Thanks for helping us prioritise.")
        print("  Disable anytime with `rapid-mlx telemetry disable`.")
        print("  Preview what we'd send: `rapid-mlx telemetry preview`.")
        print()
        return

    if action == "disable":
        record_consent(False, rapid_mlx_version=rapid_mlx_version)
        print()
        print("  Telemetry: disabled. No data will be sent.")
        print("  Re-enable anytime with `rapid-mlx telemetry enable`.")
        print()
        return

    if action == "preview":
        cid = get_or_create_client_id()
        payload = sample_preview_payload(
            client_id=cid, rapid_mlx_version=rapid_mlx_version
        )
        print()
        print("  Sample payload (this is exactly the shape we send):")
        print()
        print(json.dumps(payload.to_dict(), indent=2))
        print()
        if not is_enabled(cli_no_telemetry=cli_no):
            print("  Telemetry is currently disabled — nothing is actually sent.")
            print()
        return

    if action == "reset":
        reset_state()
        print()
        print("  Removed consent + client-id files. Next interactive run re-prompts.")
        print()
        return

    # Unknown action — argparse choices=[] would have caught this earlier
    # in normal flow; defensive guard for future maintainers.
    print(f"  Unknown telemetry action: {action!r}")
    sys.exit(1)


def main():
    from importlib.metadata import version as pkg_version

    try:
        _version = pkg_version("rapid-mlx")
    except Exception:
        _version = "dev"

    parser = argparse.ArgumentParser(
        description="Rapid-MLX: AI inference for Apple Silicon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  rapid-mlx chat                                      # interactive REPL (defaults to qwen3.5-4b-4bit)
  rapid-mlx chat qwen3.5-9b-4bit --think                   # larger model, surface reasoning
  rapid-mlx serve qwen3.5-9b-4bit --port 8000              # OpenAI-compatible server
  rapid-mlx serve mlx-community/Qwen3.5-9B-4bit       # full HF repo also works
  rapid-mlx models                                    # list all aliases
  rapid-mlx info qwen3.5-9b-4bit                           # show per-alias profile
""",
    )
    parser.add_argument(
        "--version", "-V", action="version", version=f"rapid-mlx {_version}"
    )
    parser.add_argument(
        "--no-telemetry",
        action="store_true",
        help="Disable anonymous usage telemetry for this run "
        "(equivalent to RAPID_MLX_TELEMETRY=0).",
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Serve command. ``allow_abbrev=False`` blocks unique-prefix matches
    # like ``--no-thin`` resolving silently to ``--no-thinking``: with the
    # hidden ``--no-think`` cross-alias added in D4, both flags share the
    # ``--no-thi`` prefix and prefix matching becomes ambiguous (an
    # ambiguity which argparse does NOT report by default for hidden
    # aliases). Force users to type the flag in full.
    serve_parser = subparsers.add_parser(
        "serve",
        help="Start OpenAI-compatible server",
        allow_abbrev=False,
    )
    serve_parser.add_argument(
        "model", type=str, help="Model to serve"
    ).completer = alias_completer
    serve_parser.add_argument(
        "--served-model-name",
        type=str,
        default=None,
        help="The model name used in the API. If not specified, the model argument is used.",
    )
    serve_parser.add_argument(
        "--force-disk-check",
        action="store_true",
        help=(
            "Skip the pre-flight disk-space check that aborts when the model "
            "is larger than free disk. Use only if you know the HF cache lives "
            "on a different filesystem (e.g. external drive via HF_HOME)."
        ),
    )
    serve_parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help=(
            "Host to bind (default: 127.0.0.1, loopback-only). Pass "
            '0.0.0.0 (or "") to expose the server on every '
            "interface (LAN reachable) — only do this once the "
            "bearer-auth posture has been reviewed. The wildcard "
            "bind also widens the PortSweep collision window: macOS "
            "lets a wildcard listener coexist with a more-specific "
            "(127.0.0.1) listener on the same port, so a second "
            "server may start and silently shadow the first on the "
            "loopback path. The pre-flight bind check below probes "
            "127.0.0.1 explicitly whenever --host is a wildcard "
            "alias to keep that bypass closed."
        ),
    )
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    # Socket activation — let an external supervisor (launchd, systemd,
    # parent process) bind the listening socket and execve into
    # ``rapid-mlx`` with the pre-bound fd. This closes the bind→auth
    # TOCTOU window described in issue #574: no co-located process can
    # land an unauthenticated request between socket bind and FastAPI
    # auth dependency registration, because by the time
    # ``rapid-mlx serve`` runs, the app (with auth dependencies wired
    # into chat/embeddings/audio/models routers) is already constructed
    # before ``uvicorn.run`` starts ``accept()``-ing on the fd.
    #
    # When ``--listen-fd`` is set, ``--host``/``--port`` are IGNORED:
    # the supervisor controls the bind address. The "Ready:" banner
    # prints the inherited fd shape (``Ready: inherited fd N``) — NOT
    # the user-supplied host/port, since those don't reflect the
    # supervisor's actual bind. Setting both ``--listen-fd`` and a
    # non-default ``--port`` is allowed but the port has no effect;
    # the active listener is the inherited fd.
    serve_parser.add_argument(
        "--listen-fd",
        type=_listen_fd_arg,
        default=None,
        metavar="FD",
        help=(
            "File descriptor of a pre-bound listening socket (3-1023). "
            "Used for socket activation (launchd/systemd/parent-process "
            "supervision) — supervisor binds the loopback socket, "
            "validates auth secret, then execve's into rapid-mlx. "
            "When set, --host/--port are ignored for binding."
        ),
    )
    serve_parser.add_argument(
        "--log-level",
        type=_log_level_choice,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Log level for Python logging and uvicorn (case-insensitive)",
    )
    serve_parser.add_argument(
        "--max-num-seqs", type=int, default=256, help="Max concurrent sequences"
    )
    serve_parser.add_argument(
        "--max-concurrent-requests",
        type=int,
        default=256,
        help=(
            "Admission cap on in-flight requests (queued + running). When "
            "exceeded, new requests return HTTP 503 with Retry-After. "
            "Default 256; operators on memory-constrained devices may want "
            "to set this near ``--max-num-seqs`` to limit queue depth."
        ),
    )
    serve_parser.add_argument(
        "--prefill-batch-size", type=int, default=8, help="Prefill batch size"
    )
    serve_parser.add_argument(
        "--completion-batch-size", type=int, default=32, help="Completion batch size"
    )
    serve_parser.add_argument(
        "--enable-prefix-cache",
        action="store_true",
        default=True,
        help="Enable prefix caching for repeated prompts (default: enabled)",
    )
    serve_parser.add_argument(
        "--disable-prefix-cache",
        action="store_true",
        help="Disable prefix caching",
    )
    serve_parser.add_argument(
        "--prefix-cache-size",
        type=int,
        default=100,
        help="Max entries in prefix cache (default: 100, legacy mode only)",
    )
    # Memory-aware cache options (recommended for large models)
    serve_parser.add_argument(
        "--cache-memory-mb",
        type=int,
        default=None,
        help="Cache memory limit in MB (default: auto-detect ~20%% of RAM)",
    )
    serve_parser.add_argument(
        "--cache-memory-percent",
        type=float,
        default=0.20,
        help="Fraction of available RAM for cache if auto-detecting (default: 0.20)",
    )
    serve_parser.add_argument(
        "--no-memory-aware-cache",
        action="store_true",
        help="Disable memory-aware cache, use legacy entry-count based cache",
    )
    # KV cache quantization options
    serve_parser.add_argument(
        "--kv-cache-quantization",
        action="store_true",
        help="Quantize stored KV caches to reduce memory (8-bit by default)",
    )
    serve_parser.add_argument(
        "--kv-cache-quantization-bits",
        type=int,
        default=8,
        choices=[4, 8],
        help="Bit width for KV cache quantization (default: 8)",
    )
    serve_parser.add_argument(
        "--kv-cache-quantization-group-size",
        type=int,
        default=64,
        help="Group size for KV cache quantization (default: 64)",
    )
    serve_parser.add_argument(
        "--kv-cache-min-quantize-tokens",
        type=int,
        default=256,
        help="Minimum tokens for quantization to apply (default: 256)",
    )
    # TurboQuant KV cache compression (V-only, experimental)
    serve_parser.add_argument(
        "--kv-cache-turboquant",
        action="store_true",
        help="Enable TurboQuant V-cache compression (3-4 bit, ~86%% prefix cache savings "
        "on dense models). K stays FP16. Experimental — mutually exclusive with "
        "--kv-cache-quantization.",
    )
    serve_parser.add_argument(
        "--kv-cache-turboquant-bits",
        type=int,
        default=None,
        choices=[3, 4],
        help="Bit width for TurboQuant (default: auto-select by head_dim — "
        "3-bit for head_dim>=96, 4-bit for head_dim=64)",
    )
    serve_parser.add_argument(
        "--kv-cache-turboquant-group-size",
        type=int,
        default=32,
        help="Group size for TurboQuant quantization (default: 32)",
    )
    serve_parser.add_argument(
        "--stream-interval",
        type=int,
        default=1,
        help="Tokens to batch before streaming (1=smooth, higher=throughput)",
    )
    serve_parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Default max tokens for generation (default: 32768)",
    )
    serve_parser.add_argument(
        "--continuous-batching",
        action="store_true",
        default=True,
        help="Enable continuous batching (default: on).",
    )
    # Deprecated flags — accepted silently to avoid breaking user scripts
    serve_parser.add_argument(
        "--simple-engine",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )
    serve_parser.add_argument(
        "--kv-bits",
        type=int,
        default=None,
        choices=[4, 8],
        help=argparse.SUPPRESS,
    )
    serve_parser.add_argument(
        "--kv-group-size",
        type=int,
        default=64,
        help=argparse.SUPPRESS,
    )
    serve_parser.add_argument(
        "--draft-model",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    # DFlash — block-diffusion drafter speculative decoding (z-lab / mlx-vlm).
    # Currently single-user serial mode; runs a dedicated DFlash server that
    # bypasses BatchedEngine. Eligible aliases declare ``supports_dflash=true``
    # in aliases.json (dense, ≥8-bit, drafter available — qwen3.5-27b-8bit
    # is the only validated one today). PoC: 1.83–2.18× on Qwen3.5-27B-8bit.
    serve_parser.add_argument(
        "--enable-dflash",
        action="store_true",
        default=False,
        help="Enable DFlash speculative decoding (block-diffusion drafter, "
        "single-user serial mode). Requires a DFlash-eligible alias "
        "(see ``rapid-mlx info <alias>``). Loads the drafter from the "
        "alias's ``dflash_draft_model`` field. Install with "
        "``pip install 'rapid-mlx[dflash]'``.",
    )
    serve_parser.add_argument(
        "--num-draft-tokens",
        type=int,
        default=4,
        help=argparse.SUPPRESS,
    )
    serve_parser.add_argument(
        "--specprefill",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )
    serve_parser.add_argument(
        "--specprefill-threshold",
        type=int,
        default=8192,
        help=argparse.SUPPRESS,
    )
    serve_parser.add_argument(
        "--specprefill-keep-pct",
        type=float,
        default=0.3,
        help=argparse.SUPPRESS,
    )
    serve_parser.add_argument(
        "--specprefill-draft-model",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    serve_parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="Fraction of device memory for Metal allocation limit and emergency "
        "cache clear threshold (0.0-1.0, default: 0.90). Increase to 0.95 for "
        "large models (200GB+) that need more memory headroom.",
    )
    # Paged cache options (experimental)
    serve_parser.add_argument(
        "--use-paged-cache",
        action="store_true",
        help="Use paged KV cache for memory efficiency (experimental)",
    )
    serve_parser.add_argument(
        "--paged-cache-block-size",
        type=int,
        default=64,
        help="Tokens per cache block (default: 64)",
    )
    serve_parser.add_argument(
        "--max-cache-blocks",
        type=int,
        default=1000,
        help="Maximum number of cache blocks (default: 1000)",
    )
    # Chunked prefill
    serve_parser.add_argument(
        "--chunked-prefill-tokens",
        type=int,
        default=0,
        help="Max prefill tokens per scheduler step (0=disabled). "
        "Breaks large prompts into chunks to prevent concurrent requests from starving. "
        "Recommended for Claude Code and agentic workloads with large tool schemas: "
        "--chunked-prefill-tokens 2048",
    )
    # MTP (Multi-Token Prediction)
    serve_parser.add_argument(
        "--enable-mtp",
        action="store_true",
        default=False,
        help="Enable MTP (Multi-Token Prediction) for models with built-in MTP heads. "
        "Uses cache snapshot/restore for speculative generation.",
    )
    serve_parser.add_argument(
        "--mtp-num-draft-tokens",
        type=int,
        default=1,
        help="Number of draft tokens per MTP step (default: 1)",
    )
    serve_parser.add_argument(
        "--mtp-optimistic",
        action="store_true",
        default=False,
        help="Skip MTP acceptance check for maximum speed. "
        "~5-10%% wrong tokens. Best for chat, not for code.",
    )
    # SuffixDecoding — drafter-free spec-decode using a suffix tree over
    # generated tokens. Big wins on agent/tool/JSON workloads (3-5x);
    # ~zero overhead on free-form chat. Pure-attention only.
    serve_parser.add_argument(
        "--suffix-decoding",
        action="store_true",
        default=False,
        help="Enable SuffixDecoding spec-decode (drafter-free, statistical). "
        "Speedup is workload-dependent: 3-5x on tool-call/JSON/code-edit, "
        "~1x on free-form chat. Auto-disabled on hybrid models "
        "(Qwen3.5/3.6, Granite4, Mamba/Jamba/RWKV).",
    )
    serve_parser.add_argument(
        "--suffix-max-draft",
        type=int,
        default=8,
        help="Max draft tokens per verify step (default: 8). "
        "Verify forward cost grows linearly with this.",
    )
    serve_parser.add_argument(
        "--suffix-max-suffix-len",
        type=int,
        default=4,
        help="Max k-gram length indexed for suffix matching (default: 4).",
    )
    serve_parser.add_argument(
        "--suffix-min-confidence",
        type=float,
        default=0.3,
        help="Vote confidence floor for draft truncation (default: 0.3). "
        "Lower → more optimistic drafts; higher → fewer but more reliable.",
    )
    serve_parser.add_argument(
        "--suffix-min-draft-len",
        type=int,
        default=2,
        help="Skip the verify forward when drafter returns fewer than "
        "this many tokens (default: 2). Protects free-form chat from "
        "verify overhead on weak 1-token drafts. Set to 1 to verify "
        "every draft (more aggressive; can regress chat).",
    )
    # Prefill step size
    serve_parser.add_argument(
        "--prefill-step-size",
        type=int,
        default=2048,
        help="Chunk size for prompt prefill processing. Larger values use more memory "
        "but can improve prefill throughput. (default: 2048)",
    )
    # MCP options
    serve_parser.add_argument(
        "--mcp-config",
        type=str,
        default=None,
        help="Path to MCP configuration file (JSON/YAML) for tool integration",
    )
    # Security options
    # ``--api-key`` accepts an inline value OR falls back to the
    # ``RAPID_MLX_API_KEY`` env var. ``rapid-mlx share`` uses the env-var
    # form so the bearer key never lands in argv (visible to ``ps`` for
    # any local user). Inline value still works for backwards-compat
    # with existing scripts; if both are set, the inline value wins.
    serve_parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help=(
            "API key for authentication (if not set, falls back to the "
            "RAPID_MLX_API_KEY env var; if neither, no auth required)"
        ),
    )
    serve_parser.add_argument(
        "--cors-origins",
        type=str,
        nargs="+",
        default=None,
        metavar="ORIGIN",
        help=(
            "Allowed CORS origins (default: * for all origins). "
            "Example: --cors-origins http://localhost:3000 https://myapp.com"
        ),
    )
    serve_parser.add_argument(
        "--rate-limit",
        type=int,
        default=0,
        help="Rate limit requests per minute per client (0 = disabled)",
    )
    # Hard cap on per-request body size — DoS defense.
    # See ``vllm_mlx/middleware/body_size.py`` for the rationale (pre-fix:
    # a 10 MB body silently ran a ~60 s full prefill on a 27B alias before
    # the client timed out; rapid-desktop#273 + #463). Default 8 MiB fits
    # a 128k-token prompt with tool schemas; 0 disables the cap.
    serve_parser.add_argument(
        "--max-request-bytes",
        type=int,
        default=None,
        help=(
            "Maximum HTTP request body size in bytes (default: 8 MiB = "
            "8388608). Requests over this cap are rejected with HTTP 413 "
            "before JSON parsing or tokenization runs. 0 disables the cap. "
            "Falls back to the RAPID_MLX_MAX_REQUEST_BYTES env var if unset."
        ),
    )
    serve_parser.add_argument(
        "--timeout",
        type=float,
        default=1800.0,
        help="Default request timeout in seconds (default: 1800 = 30 min)",
    )
    # Tool calling options
    serve_parser.add_argument(
        "--enable-auto-tool-choice",
        action="store_true",
        help="Enable auto tool choice for supported models. Use --tool-call-parser to specify which parser to use.",
    )
    serve_parser.add_argument(
        "--tool-call-parser",
        type=str,
        default=None,
        # Choices NOT enforced at argparse level — the canonical set is the
        # ToolParserManager registry, which has ~39 entries (canonical
        # names + per-family aliases like ``deepseek_v31``, ``llama4``,
        # ``moonshot`` for kimi, ``nous`` for hermes). The argparse hard-
        # coded list drifted to 19 over multiple releases and rejected
        # legitimate aliases users discovered via ``rapid-mlx info``.
        # Validation now happens post-parse in
        # ``_validate_tool_call_parser_choice`` against the live registry.
        # v0.6.63 onboarding sweep finding #1.
        help=(
            "Select the tool call parser for the model. Canonical options: "
            "auto (auto-detect), mistral, qwen/qwen3/qwen3_xml (reasoning models, "
            "<tool_call>JSON</tool_call> format), qwen3_coder/qwen3_coder_xml "
            "(Coder model, <function=NAME> XML format), llama/llama3/llama4, "
            "hermes/nous, deepseek/deepseek_v3/deepseek_v31, kimi/moonshot/kimi_k2, "
            "granite/granite3, nemotron/nemotron3, xlam, functionary/meetkai, "
            "glm47/glm4, minimax/minimax_m2, harmony/gpt-oss/gpt_oss, "
            "gemma4/gemma_4, seed_oss/seed. "
            "Run `python -c 'from vllm_mlx.tool_parsers import ToolParserManager;"
            "print(sorted(ToolParserManager.tool_parsers))'` for the live list. "
            "Required for --enable-auto-tool-choice."
        ),
    )
    # Tool logits bias (jump-forward decoding for tool call structural tokens)
    serve_parser.add_argument(
        "--enable-tool-logits-bias",
        action="store_true",
        default=False,
        help="Bias logits toward structural tool call tokens for faster generation. "
        "Only active when --tool-call-parser is also set. Currently supports minimax.",
    )
    # Reasoning parser options - choices loaded dynamically from registry
    from .reasoning import list_parsers

    reasoning_choices = list_parsers()
    serve_parser.add_argument(
        "--reasoning-parser",
        type=str,
        default=None,
        choices=reasoning_choices,
        help=(
            "Enable reasoning content extraction with specified parser. "
            "Extracts <think>...</think> tags into reasoning_content field. "
            f"Options: {', '.join(reasoning_choices)}."
        ),
    )
    serve_parser.add_argument(
        "--no-thinking",
        action="store_true",
        default=False,
        help=(
            "Disable reasoning/thinking parser even if auto-detected. "
            "Thinking tokens will appear as regular content. "
            "Useful for faster responses when chain-of-thought is not needed."
        ),
    )
    # Hidden cross-alias mirroring ``chat --no-thinking`` (see the chat
    # parser for the full rationale). ``serve --no-think`` lands on the
    # same ``no_thinking`` destination so users who reach for the shorter
    # name don't get an ``unrecognized arguments`` error.
    serve_parser.add_argument(
        "--no-think",
        dest="no_thinking",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    serve_parser.add_argument(
        "--no-tool-call-parser",
        dest="no_tool_call_parser",
        action="store_true",
        default=False,
        help=(
            "Force-disable tool-call parser auto-detection from the alias "
            "profile. Escape hatch (SOP §10) when AliasProfile's auto-"
            "selected parser misfires for a specific deployment. Mutually "
            "exclusive with --tool-call-parser."
        ),
    )
    serve_parser.add_argument(
        "--no-reasoning-parser",
        dest="no_reasoning_parser",
        action="store_true",
        default=False,
        help=(
            "Force-disable reasoning parser auto-detection from the alias "
            "profile. Distinct from --no-thinking (which also suppresses "
            "the chain-of-thought prompt template) — this flag ONLY skips "
            "the auto-config step. Mutually exclusive with --reasoning-parser."
        ),
    )
    # SOP §10 profile-override escape hatches. Pair every binary
    # auto-routing field with both force-on and force-off CLI flags so
    # users always have an override path when the AliasProfile
    # auto-detection misfires. Registered in
    # tests/test_no_mllm_flag.py::test_auto_routing_flags_have_force_on_and_force_off_pair.
    serve_parser.add_argument(
        "--force-hybrid",
        dest="force_hybrid",
        action="store_true",
        default=False,
        help=(
            "Force-treat the model as a hybrid (linear-attention / Mamba) "
            "architecture even when AliasProfile says otherwise. Disables "
            "spec/suffix decode paths that are unsound on hybrids. "
            "Mutually exclusive with --no-hybrid."
        ),
    )
    serve_parser.add_argument(
        "--no-hybrid",
        dest="no_hybrid",
        action="store_true",
        default=False,
        help=(
            "Force-treat the model as non-hybrid (full attention) even when "
            "AliasProfile says it's hybrid. Use when the profile mis-labels "
            "your model and you want spec/suffix decode enabled. "
            "Mutually exclusive with --force-hybrid."
        ),
    )
    serve_parser.add_argument(
        "--force-spec-decode",
        dest="force_spec_decode",
        action="store_true",
        default=False,
        help=(
            "Force-enable speculative-decode eligibility even when "
            "AliasProfile says the model doesn't support it. Risky on "
            "hybrid models — use only when you've verified the profile "
            "is wrong. Mutually exclusive with --no-spec-decode."
        ),
    )
    serve_parser.add_argument(
        "--no-spec-decode",
        dest="no_spec_decode",
        action="store_true",
        default=False,
        help=(
            "Force-disable speculative-decode eligibility (suffix / MTP / "
            "DFlash) even when AliasProfile says the model supports it. "
            "Mutually exclusive with --force-spec-decode."
        ),
    )
    # #516 — HarmonyStreamingRouter auto-upgrade escape hatches (G11).
    # PR #515 introduced an auto-upgrade from the legacy harmony state
    # machine to openai-harmony's StreamableParser for matched-vocab
    # gpt-oss tokenizers. The auto-detection is conservative (three-layer
    # compat check) but the SOP requires every binary auto-routing
    # decision expose both force-on and force-off CLI flags.
    serve_parser.add_argument(
        "--force-openai-harmony-streaming",
        dest="force_openai_harmony_streaming",
        action="store_true",
        default=False,
        help=(
            "Force-on: construct HarmonyStreamingRouter even when the "
            "compat gate would reject. Use to debug a regression in the "
            "gate itself; production should leave this off. Mutually "
            "exclusive with --no-openai-harmony-streaming."
        ),
    )
    serve_parser.add_argument(
        "--no-openai-harmony-streaming",
        dest="no_openai_harmony_streaming",
        action="store_true",
        default=False,
        help=(
            "Force-off: skip the HarmonyStreamingRouter upgrade and use "
            "the legacy custom harmony state machine even on matched-vocab "
            "gpt-oss tokenizers. Escape hatch for a hypothetical false "
            "positive in the compat gate. Mutually exclusive with "
            "--force-openai-harmony-streaming."
        ),
    )
    # GC control (Tier 0 optimization)
    serve_parser.add_argument(
        "--gc-control",
        action="store_true",
        default=True,
        help="Enable Python GC pausing during generation to avoid latency spikes (default: enabled)",
    )
    serve_parser.add_argument(
        "--no-gc-control",
        action="store_true",
        help="Disable GC control (allow normal Python GC during generation)",
    )
    # Pinned prefix cache (Tier 0 optimization)
    serve_parser.add_argument(
        "--pin-system-prompt",
        action="store_true",
        default=False,
        help="Auto-pin system prompt in prefix cache to prevent eviction under memory pressure",
    )
    # Multimodal option
    serve_parser.add_argument(
        "--mllm",
        action="store_true",
        help="Force load model as multimodal (vision) even if name doesn't match auto-detection patterns",
    )
    serve_parser.add_argument(
        "--no-mllm",
        "--text-only",
        dest="no_mllm",
        action="store_true",
        help="Force load model as text-only LLM even when auto-detection would route it to the multimodal/VLM path. Escape hatch for incomplete vision-tower checkpoints (#393) and text-only forks of multimodal architectures whose config.json still declares vision_config.",
    )
    # Generation defaults
    serve_parser.add_argument(
        "--default-temperature",
        type=float,
        default=None,
        help="Override default temperature for all requests (default: use model default)",
    )
    serve_parser.add_argument(
        "--default-top-p",
        type=float,
        default=None,
        help="Override default top_p for all requests (default: use model default)",
    )
    serve_parser.add_argument(
        "--default-top-k",
        type=int,
        default=None,
        help="Override default top_k for all requests (default: use model default)",
    )
    serve_parser.add_argument(
        "--default-min-p",
        type=float,
        default=None,
        help="Override default min_p for all requests (default: use model default)",
    )
    serve_parser.add_argument(
        "--default-repetition-penalty",
        type=float,
        default=None,
        help="Override default repetition_penalty for all requests (default: use model default)",
    )
    serve_parser.add_argument(
        "--default-presence-penalty",
        type=float,
        default=None,
        help="Override default presence_penalty for all requests (default: use model default)",
    )
    serve_parser.add_argument(
        "--default-frequency-penalty",
        type=float,
        default=None,
        help="Override default frequency_penalty for all requests (default: use model default)",
    )
    # Cloud routing options
    serve_parser.add_argument(
        "--cloud-model",
        type=str,
        default=None,
        help="Cloud model string for litellm (e.g. 'anthropic/claude-sonnet-4-5-20250929'). "
        "When set, large-context requests are routed to the cloud provider.",
    )
    serve_parser.add_argument(
        "--cloud-threshold",
        type=int,
        default=20000,
        help="New token threshold to trigger cloud routing (default: 20000). "
        "Only requests with more new (uncached) tokens than this are routed.",
    )
    serve_parser.add_argument(
        "--cloud-api-base",
        type=str,
        default=None,
        help="Custom API base URL for cloud model (for OpenAI-compatible providers like Zhipu).",
    )
    serve_parser.add_argument(
        "--cloud-api-key",
        type=str,
        default=None,
        help="API key for cloud model (overrides environment variable).",
    )
    # Embedding model option
    serve_parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help=(
            "Pre-load an embedding model at startup (e.g. "
            "mlx-community/embeddinggemma-300m-6bit). Requires the "
            "[embeddings] extra: pip install 'rapid-mlx[embeddings]'."
        ),
    )
    # PFlash long-prompt prefill compression (#287). Off by default; see
    # vllm_mlx/pflash.py for the design and the prefix-cache bypass.
    _add_pflash_args(serve_parser)
    # Bench command
    bench_parser = subparsers.add_parser("bench", help="Run benchmark")
    bench_parser.add_argument(
        "model", type=str, help="Model to benchmark"
    ).completer = alias_completer
    bench_parser.add_argument(
        "--force-disk-check",
        action="store_true",
        help=(
            "Skip the pre-flight disk-space check that aborts when the model "
            "is larger than free disk. Use only if you know the HF cache lives "
            "on a different filesystem (e.g. external drive via HF_HOME)."
        ),
    )
    bench_parser.add_argument(
        "--num-prompts", type=int, default=10, help="Number of prompts"
    )
    bench_parser.add_argument(
        "--max-tokens", type=int, default=100, help="Max tokens per prompt"
    )
    bench_parser.add_argument(
        "--max-num-seqs", type=int, default=32, help="Max concurrent sequences"
    )
    bench_parser.add_argument(
        "--prefill-batch-size", type=int, default=8, help="Prefill batch size"
    )
    bench_parser.add_argument(
        "--completion-batch-size", type=int, default=16, help="Completion batch size"
    )
    bench_parser.add_argument(
        "--enable-prefix-cache",
        action="store_true",
        default=True,
        help="Enable prefix caching (default: enabled)",
    )
    bench_parser.add_argument(
        "--disable-prefix-cache",
        action="store_true",
        help="Disable prefix caching",
    )
    bench_parser.add_argument(
        "--prefix-cache-size",
        type=int,
        default=100,
        help="Max entries in prefix cache (default: 100, legacy mode only)",
    )
    # Memory-aware cache options (recommended for large models)
    bench_parser.add_argument(
        "--cache-memory-mb",
        type=int,
        default=None,
        help="Cache memory limit in MB (default: auto-detect ~20%% of RAM)",
    )
    bench_parser.add_argument(
        "--cache-memory-percent",
        type=float,
        default=0.20,
        help="Fraction of available RAM for cache if auto-detecting (default: 0.20)",
    )
    bench_parser.add_argument(
        "--no-memory-aware-cache",
        action="store_true",
        help="Disable memory-aware cache, use legacy entry-count based cache",
    )
    # KV cache quantization options
    bench_parser.add_argument(
        "--kv-cache-quantization",
        action="store_true",
        help="Quantize stored KV caches to reduce memory (8-bit by default)",
    )
    bench_parser.add_argument(
        "--kv-cache-quantization-bits",
        type=int,
        default=8,
        choices=[4, 8],
        help="Bit width for KV cache quantization (default: 8)",
    )
    bench_parser.add_argument(
        "--kv-cache-quantization-group-size",
        type=int,
        default=64,
        help="Group size for KV cache quantization (default: 64)",
    )
    bench_parser.add_argument(
        "--kv-cache-min-quantize-tokens",
        type=int,
        default=256,
        help="Minimum tokens for quantization to apply (default: 256)",
    )
    # Paged cache options (experimental)
    bench_parser.add_argument(
        "--use-paged-cache",
        action="store_true",
        help="Use paged KV cache for memory efficiency (experimental)",
    )
    bench_parser.add_argument(
        "--paged-cache-block-size",
        type=int,
        default=64,
        help="Tokens per cache block (default: 64)",
    )
    bench_parser.add_argument(
        "--max-cache-blocks",
        type=int,
        default=1000,
        help="Maximum number of cache blocks (default: 1000)",
    )
    # Community benchmark submission. Mutually-exclusive with the
    # freeform bench above — when --submit is set the standardized
    # B=1 runner takes over and every other knob is ignored.
    bench_parser.add_argument(
        "--submit",
        action="store_true",
        help=(
            "Run the standardized B=1 community benchmark and open a PR to "
            "community-benchmarks/. Locks every comparability knob; ignores "
            "the freeform --num-prompts / --max-tokens / --max-num-seqs args."
        ),
    )
    bench_parser.add_argument(
        "--sampled",
        action="store_true",
        help=(
            "With --submit, run the bench at temp=0.7/top_p=0.9 instead of "
            "greedy. Stored as a separate 'sampled' bucket — useful for "
            "comparing against Artificial Analysis-style real-world numbers."
        ),
    )
    bench_parser.add_argument(
        "--notes",
        type=str,
        default=None,
        help=(
            "Optional free-text annotation attached to the submission "
            "(e.g. 'on battery', 'fresh boot'). Max 200 chars."
        ),
    )
    bench_parser.add_argument(
        "--repo-root",
        type=str,
        default=None,
        help=(
            "Path to the Rapid-MLX git checkout. Defaults to the current "
            "working directory. The --submit flow writes the JSON file and "
            "opens the PR from this checkout."
        ),
    )
    # --tier: user-facing tier dispatcher (PR #2). Mutually-exclusive
    # with --submit (PR #3 will consolidate them, but for now the two
    # are independent code paths).
    bench_parser.add_argument(
        "--tier",
        type=str,
        choices=["smoke", "speed", "harness", "all"],
        default=None,
        help=(
            "Run one of the standardized validation tiers: "
            "'smoke' (boot + 1 prompt), "
            "'speed' (B=1 perf probe), "
            "'harness' (5 first-class agent harnesses: "
            "codex/opencode/hermes/aider/langchain), "
            "'all' (smoke → speed → harness sequentially, abort on smoke "
            "fail). Boots the model server exactly once per invocation."
        ),
    )
    bench_parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help=(
            "For --tier: attach to an already-running server at this URL "
            "(e.g. http://localhost:8000) instead of booting one. Used by "
            "release_check_m3.sh G7b to reuse the gauntlet's server."
        ),
    )
    bench_parser.add_argument(
        "--long-prompt-tokens",
        type=int,
        default=0,
        help="Approximate reference-context tokens to prepend to each "
        "benchmark prompt. Used with --pflash auto/always for "
        "long-prompt TTFT replication (#287).",
    )
    _add_pflash_args(bench_parser)

    # Models command. ``ls`` is registered as a top-level alias that
    # defaults to ``models --cached`` (the locally-cached view) — two
    # muscle-memory entry points, one underlying impl.
    models_parser = subparsers.add_parser("models", help="List available model aliases")
    models_parser.add_argument(
        "--cached",
        action="store_true",
        default=False,
        help="Only list models that are downloaded to the local HuggingFace "
        "cache (alias, HF repo, size on disk, last modified).",
    )
    subparsers.add_parser(
        "ls",
        help="List models in the local HuggingFace cache (alias for `models --cached`)",
    )

    # Version + help — utility commands that mirror the existing flags but
    # are scriptable as plain subcommands.
    subparsers.add_parser("version", help="Show version number")
    help_parser = subparsers.add_parser("help", help="Show help for a subcommand")
    help_parser.add_argument(
        "subcommand", nargs="?", help="Subcommand to show help for (omit for top-level)"
    )

    # Pull / rm / ps — Ollama-style cache and process management.
    pull_parser = subparsers.add_parser(
        "pull", help="Download a model to the HuggingFace cache (no server)"
    )
    pull_parser.add_argument(
        "model", help="Model alias (e.g. qwen3.5-4b-4bit) or HF repo (org/name)"
    ).completer = alias_completer
    rm_parser = subparsers.add_parser(
        "rm", help="Remove a cached model from the HuggingFace cache"
    )
    rm_parser.add_argument(
        "model", help="Model alias (e.g. qwen3.5-4b-4bit) or HF repo (org/name)"
    ).completer = alias_completer
    subparsers.add_parser("ps", help="List running rapid-mlx servers")

    # Upgrade — detect install method and run the right upgrade command
    upgrade_parser = subparsers.add_parser(
        "upgrade",
        help="Upgrade rapid-mlx to the latest version (brew / pip / install.sh)",
    )
    upgrade_parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip the confirmation prompt and run the upgrade immediately.",
    )

    # Chat — interactive REPL backed by a (spawned or existing) server.
    # ``run`` is exposed as a subparser alias purely for Ollama-muscle-memory
    # parity (``ollama run <model>``). Both names route to ``chat_command``.
    chat_parser = subparsers.add_parser(
        "chat",
        aliases=["run"],
        help="Interactive chat REPL with a model",
        description=(
            "Interactive chat REPL with a model.\n\n"
            "Note: 'rapid-mlx run' is an alias for 'chat' (Ollama compatibility)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        # See serve_parser for the rationale: ``--think``/``--no-think`` +
        # ``--thinking``/``--no-thinking`` cross-aliases create ambiguous
        # prefixes that argparse silently resolves to whichever flag was
        # added first.
        allow_abbrev=False,
    )
    chat_parser.add_argument(
        "model",
        nargs="?",
        default="qwen3.5-4b-4bit",
        help="Model alias (e.g. qwen3.5-4b-4bit) or HF repo (org/name). "
        "Defaults to qwen3.5-4b-4bit when omitted.",
    ).completer = alias_completer
    chat_parser.add_argument(
        "--system",
        type=str,
        default=None,
        help="System prompt prepended to the conversation",
    )
    chat_parser.add_argument(
        "--think",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable thinking/reasoning mode (default: off in chat REPL — "
        "reasoning models like Qwen3.5 otherwise leak raw chain-of-thought "
        "and can loop until max-tokens). Use --think to surface reasoning, "
        "--no-think is also accepted for back-compat.",
    )
    # Hidden cross-alias for users who picked up the ``--no-thinking`` muscle
    # memory from ``rapid-mlx serve``. ``serve --no-thinking`` and
    # ``chat --no-think`` mean different things internally (server-side
    # parser disable vs. per-request ``enable_thinking=false``), but the
    # flag-name difference trips users. We accept the wrong-side name as
    # an alias for the right-side semantics: ``chat --no-thinking`` simply
    # forwards to the same destination as ``--no-think``.
    chat_parser.add_argument(
        "--no-thinking",
        dest="think",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    chat_parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Max tokens per assistant response (default: 2048; raised to "
        "4096 when --think is set so reasoning + answer fit the budget).",
    )
    chat_parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    chat_parser.add_argument(
        "--port",
        type=_port_arg,
        default=None,
        help="Connect to existing server on 127.0.0.1:<port> instead of spawning",
    )
    chat_parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Connect to existing server URL (e.g. http://host:8000) "
        "instead of spawning. Overrides --port.",
    )
    chat_parser.add_argument(
        "--ready-timeout",
        type=int,
        default=600,
        help="Seconds to wait for the spawned server to become ready (default: 600)",
    )
    chat_parser.add_argument(
        "--response-timeout",
        type=int,
        default=600,
        help="Seconds to wait for a single assistant response (default: 600)",
    )

    # Info command — show the per-model profile (parsers + capability gates)
    info_parser = subparsers.add_parser(
        "info",
        help="Show the per-model profile for a model name or alias",
    )
    info_parser.add_argument(
        "model",
        help="Model alias (e.g. qwen3.5-4b-4bit) or HF repo (e.g. mlx-community/SmolLM3-3B-4bit)",
    ).completer = alias_completer

    # Agents command
    agents_parser = subparsers.add_parser(
        "agents", help="List, configure, and test agent integrations"
    )
    agents_parser.add_argument(
        "agent_name",
        nargs="?",
        default=None,
        help="Agent name (e.g. hermes, goose, aider). Omit to list all.",
    )
    agents_parser.add_argument(
        "--setup",
        action="store_true",
        help="Auto-configure the agent to point at this server",
    )
    agents_parser.add_argument(
        "--test",
        action="store_true",
        help="Run integration tests for this agent",
    )
    agents_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to use (default: auto-detect from running server)",
    ).completer = alias_completer
    agents_parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000/v1",
        help="Rapid-MLX server URL (default: http://localhost:8000/v1)",
    )
    agents_parser.add_argument(
        "--agent-version",
        type=str,
        default=None,
        help="Agent version for version-specific config (e.g. 0.8.5)",
    )

    # Doctor command — pure env-health probe (≤5 s, no model load, no server).
    # Model-validation tiers (smoke/check/full/benchmark) moved to
    # ``rapid-mlx bench --tier ...`` as of v0.7.22.
    #
    # The legacy positional ``tier`` plus ``--model``, ``--models``, and
    # ``--update-baselines`` are intentionally retained (SUPPRESSed from
    # --help) for one release so users hitting the old form
    # ``rapid-mlx doctor check --model qwen3.5-9b-4bit`` get the actionable
    # bench redirect from ``doctor_command`` instead of an argparse
    # ``unrecognized arguments`` wall. Codex review round 1 flagged this:
    # rejecting at argparse-time defeated the redirect. Drop these in a
    # future release once telemetry confirms no one's still calling them.
    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Check environment health (Python, packages, HF cache, network, ...)",
    )
    doctor_parser.add_argument(
        "tier",
        nargs="?",
        default=None,
        choices=["smoke", "check", "full", "benchmark"],
        help=argparse.SUPPRESS,
    )
    doctor_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print the underlying probe detail for each check",
    )
    # Legacy compatibility shims — accepted-but-ignored so the redirect
    # message in ``doctor_command`` can fire (see comment above).
    doctor_parser.add_argument(
        "--model",
        default=None,
        help=argparse.SUPPRESS,
    )
    doctor_parser.add_argument(
        "--models",
        default=None,
        help=argparse.SUPPRESS,
    )
    doctor_parser.add_argument(
        "--update-baselines",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    # Telemetry subcommand — opt-in anonymous usage data (Issue #236).
    # See vllm_mlx/telemetry/ for what we collect / don't collect, and
    # the README "Telemetry" section for the user-facing summary.
    telemetry_parser = subparsers.add_parser(
        "telemetry",
        help="Manage anonymous usage telemetry (opt-in)",
    )
    telemetry_subparsers = telemetry_parser.add_subparsers(
        dest="telemetry_action",
        help="Telemetry actions",
    )
    telemetry_subparsers.add_parser(
        "status", help="Show whether telemetry is enabled and why"
    )
    telemetry_subparsers.add_parser(
        "enable", help="Opt in to anonymous usage telemetry"
    )
    telemetry_subparsers.add_parser(
        "disable", help="Opt out of anonymous usage telemetry"
    )
    telemetry_subparsers.add_parser(
        "preview",
        help="Print a sample payload showing exactly what telemetry would send",
    )
    telemetry_subparsers.add_parser(
        "reset",
        help="Delete the consent + client-id files (next run re-prompts)",
    )

    # Share subcommand — expose a local serve behind a public rapidmlx.com URL.
    from vllm_mlx.share.cli import register as _register_share

    _register_share(subparsers)

    # Launch subcommand — one-shot bootstrap that patches IDE/agent
    # client configs (Cline, Claude Code, Continue, Cursor) to route
    # at the local rapid-mlx server. See GH issue #566 for motivation.
    # Registered AFTER share so the help-text ordering reads
    # serve→…→share→launch, matching the rough "more common first" flow.
    from vllm_mlx.launch.cli import register as _register_launch

    _register_launch(subparsers)

    # Shell tab completion via argcomplete. Must fire before parse_args:
    # when the shell completion handler invokes us with the
    # ``_ARGCOMPLETE`` env var set, this function short-circuits before
    # any heavy import paths or model resolution runs, so the user gets
    # snappy ``rapid-mlx chat gemma-4-<TAB>`` even on a cold shell.
    #
    # ``_action_conflicts`` and ``_seen_non_default_actions`` are
    # populated by argcomplete inside ``IntrospectiveArgumentParser._
    # parse_known_args`` — but option completion (``finders.py:_
    # action_allowed``) reads them before parsing has run on a
    # subparser, raising ``AttributeError`` on the first Tab. We
    # pre-walk the parser tree and seed empty containers so completion
    # works at the very first keystroke. Issue tracked upstream at
    # kislyuk/argcomplete (no mutex groups → conflict set is just
    # empty; this is the documented null-init).
    def _preinit_argcomplete_state(p: argparse.ArgumentParser) -> None:
        if not hasattr(p, "_action_conflicts"):
            p._action_conflicts = {}  # type: ignore[attr-defined]
        if not hasattr(p, "_seen_non_default_actions"):
            p._seen_non_default_actions = set()  # type: ignore[attr-defined]
        if not hasattr(p, "active_actions"):
            p.active_actions = []  # type: ignore[attr-defined]
        for action in p._actions:
            if isinstance(action, argparse._SubParsersAction):
                for sub in action.choices.values():
                    if isinstance(sub, argparse.ArgumentParser):
                        _preinit_argcomplete_state(sub)

    _preinit_argcomplete_state(parser)
    try:
        import argcomplete
    except ModuleNotFoundError as exc:
        # Best-effort: tab completion silently no-ops if argcomplete is
        # missing. Listed as a required dep in ``pyproject.toml`` so
        # this path only fires in minimal test envs or stripped images.
        # Narrow the swallow to the top-level argcomplete package — if a
        # transitive import inside argcomplete is missing we want that
        # to surface, not get mistaken for "argcomplete not installed".
        if exc.name != "argcomplete":
            raise
    else:
        argcomplete.autocomplete(parser)

    args = parser.parse_args()

    # First-run consent prompt — fires at most once per machine, only on
    # interactive subcommands when stdin is a tty. Safe no-op otherwise.
    # Must run *before* heavy subcommand work so the user sees the
    # disclosure before any model load logs scroll past.
    _just_collected_consent = False
    if getattr(args, "command", None) is not None:
        from vllm_mlx.telemetry import maybe_prompt_for_consent
        from vllm_mlx.telemetry.state import set_cli_kill_switch

        # ``--no-telemetry`` is a per-run override; thread it into the
        # process-level kill switch so every emit site sees it without
        # having to plumb the flag through every signature.
        set_cli_kill_switch(getattr(args, "no_telemetry", False))

        _just_collected_consent = maybe_prompt_for_consent(
            args.command,
            cli_no_telemetry=getattr(args, "no_telemetry", False),
        )

    # Telemetry session lifecycle — emit session_start once we know what
    # subcommand we're dispatching, register an atexit hook for
    # session_end so the duration covers the whole interactive run
    # (including ``rapid-mlx chat`` REPLs and ``serve`` processes that
    # only exit on Ctrl-C). emit.* helpers are individually guarded by
    # ``is_enabled()`` — when telemetry is off the calls are cheap
    # no-ops, no payload constructed.
    #
    # The ``telemetry`` subcommand itself is excluded: ``telemetry
    # disable`` / ``reset`` would otherwise queue an event on the way to
    # turning telemetry OFF — a small but ugly "phone home before
    # silencing the phone" surprise that codex round 1 caught. ``status``
    # / ``preview`` / ``enable`` are excluded for consistency; their
    # observability value is near zero.
    #
    # ``_just_collected_consent`` skips the run that JUST collected
    # first-time opt-in (round 3 codex catch): the disclosure copy
    # promises "nothing from before this prompt or from a session you
    # opted out of", and the current invocation's argv was determined
    # BEFORE the user said yes. The next run starts the contract clean.
    #
    # ``_session_models_requested`` is hoisted outside the conditional so
    # the alias-resolution block below can append to it unconditionally
    # without a NameError when telemetry was skipped. The closure
    # passed to ``session_end`` reads the same list, so populate-then-
    # emit is naturally ordered.
    #
    # Round 19 codex catch on the naming: this list captures models
    # the user's invocation REQUESTED -- the alias passed argparse
    # validation -- NOT models the loader confirmed it loaded. A
    # declined auto-pull or a load failure later in the subcommand
    # handler still leaves the entry here, which the lifecycle event
    # surfaces verbatim. Phase 2.2 will replace this with confirmed
    # load events emitted from ``vllm_mlx/engine/loader.py``; until
    # then, the field semantics is "alias the session was for" and the
    # helper docstring spells this out.
    _session_models_requested: list[str] = []
    if (
        getattr(args, "command", None) is not None
        and args.command != "telemetry"
        and not _just_collected_consent
    ):
        import atexit as _atexit
        import sys as _sys
        import time as _time

        from vllm_mlx.telemetry import emit as _telemetry_emit

        _session_subcommand = args.command
        _session_started_at = _time.monotonic()
        # Round 19 codex catch: extract flag names HERE so raw argv
        # tokens (which include flag VALUES) never cross into the
        # telemetry helper signatures. The disclosure promise "values
        # are never even read" is now literally true at the function-
        # call boundary.
        from vllm_mlx.telemetry.redact import (
            hash_flag_names as _telemetry_extract_flag_names,
        )

        _session_flag_names = _telemetry_extract_flag_names(_sys.argv[1:])
        # Round 19 codex NIT: session_start sees an empty IMMUTABLE
        # snapshot of models_loaded so it does not depend on whether
        # ``emit.session_start()`` eagerly copies its input. The closure-
        # captured list keeps mutating until session_end takes its own
        # snapshot below.
        _telemetry_emit.session_start(
            subcommand=_session_subcommand,
            flag_names=_session_flag_names,
            models_loaded=(),
        )

        def _emit_session_end() -> None:
            try:
                # Snapshot the closure-captured list to an immutable
                # tuple so the payload reflects the exact state at this
                # call (round 19 NIT).
                _models_snapshot = tuple(_session_models_requested)
                _telemetry_emit.session_end(
                    subcommand=_session_subcommand,
                    duration_seconds=int(_time.monotonic() - _session_started_at),
                    models_loaded=_models_snapshot,
                )
                # Round 5 codex review caught that the atexit handler
                # for the queue's ``shutdown`` is registered inside
                # ``session_start`` (LIFO → runs after this handler),
                # but relying on that ordering is fragile. Force a
                # synchronous drain here so ``session_end`` actually
                # lands regardless of atexit ordering quirks. Idempotent
                # — the queue's own ``shutdown`` will be a no-op when
                # it runs later.
                #
                # ``session_end`` is best-effort by design (round 7
                # codex catch): the queue's own ``SHUTDOWN_BUDGET_S``
                # (2 s) caps user-visible exit latency. A slow or
                # blackholed collector drops the event — that is the
                # right trade-off, because making the user wait
                # ~12 s on every ``serve`` Ctrl-C just to file a
                # better stat is hostile UX.
                #
                # Round 19 codex review closed the previous round-17
                # SIGTERM gap: ``register_session_end_hook`` is wired
                # below so the FastAPI lifespan shutdown in
                # ``vllm_mlx.server`` calls this same function on
                # SIGTERM (systemd / Docker / Kubernetes graceful
                # stop). The latch inside the emit module makes the
                # second invocation a no-op so the event lands exactly
                # once regardless of which path fires first.
                #
                # ``_queue is None`` (telemetry was disabled, so
                # ``session_end`` no-op'd and never instantiated the
                # singleton) skips ``get_queue()`` — round 7 catch —
                # otherwise we'd spawn a daemon thread during
                # interpreter shutdown for nothing.
                try:
                    if _telemetry_emit._queue is not None:
                        _telemetry_emit._queue.shutdown()
                except BaseException:
                    pass
            except BaseException:
                # atexit handlers are run during interpreter shutdown;
                # anything that fires here — including a stray
                # ``KeyboardInterrupt`` or ``SystemExit`` raised inside
                # redaction / queue code mid-teardown — is purely noise
                # at this point because the process is already exiting.
                # Round 9 codex review caught the previous ``Exception``
                # catch as too narrow for an atexit context.
                return

        # Register the same callable for both teardown paths. The
        # latch in ``fire_session_end_hook`` ensures it runs once
        # regardless of which path (FastAPI lifespan shutdown OR cli
        # atexit fallback) fires first.
        _telemetry_emit.register_session_end_hook(_emit_session_end)
        _atexit.register(_telemetry_emit.fire_session_end_hook)

    # Resolve model aliases before dispatch.
    #
    # The doctor subcommand is exempt for historical reasons (and as a
    # belt-and-suspenders guard now that doctor doesn't take ``--model``):
    # an env-health probe should never trigger an alias→path lookup.
    if (
        hasattr(args, "model")
        and args.model
        and getattr(args, "command", None) != "doctor"
    ):
        from vllm_mlx.model_aliases import resolve_model

        resolved = resolve_model(args.model)
        if resolved != args.model:
            print(f"  Alias: {args.model} → {resolved}")
            args._original_alias = args.model
            args.model = resolved
        elif "/" not in args.model and not os.path.exists(args.model):
            # R8-M5 (Bo 0.8.9 dogfood): short audio aliases (``kokoro``,
            # ``whisper``, ``parakeet``, ``chatterbox``, ``vibevoice``,
            # ``voxcpm``) and their full-form siblings (``kokoro-82m-
            # 8bit``) are NOT in ``aliases.json`` — the resolver returns
            # them unchanged, then this fail-fast branch trips with
            # "is not a known alias or HuggingFace path" BEFORE
            # ``serve_command`` can run the audio boot guard. On a
            # fresh ``pip install rapid-mlx`` (no ``[audio]`` extra)
            # that means the operator sees a generic "unknown alias"
            # instead of the actionable "install rapid-mlx[audio]"
            # hint, and on a healthy install with ``[audio]`` the
            # short alias resolves at request time inside the audio
            # routes (``TTS_MODEL_ALIASES`` / ``STT_MODEL_ALIASES``)
            # but the CLI exits before serve_command ever runs.
            #
            # Skip the fail-fast for audio aliases so:
            #   - missing-extra installs reach the audio boot guard
            #     in ``serve_command`` (rc=2 + install hint).
            #   - healthy installs reach the audio routes' alias
            #     resolution and serve correctly.
            # The substring check matches the same alias surface the
            # serve-command boot guard uses (``_AUDIO_ALIAS_TOKENS``)
            # so a name that trips one trips the other — no risk of a
            # text/vision alias accidentally bypassing the fail-fast.
            from .audio.probe import is_audio_model_alias

            if not is_audio_model_alias(args.model):
                # Not an alias, not a HuggingFace org/name path, not a
                # local directory, not an audio alias — fail fast with
                # suggestions instead of letting the request hit
                # HuggingFace and 404 with a 30-line stack trace.
                print(
                    f"\n  Error: '{args.model}' is not a known alias or HuggingFace path."
                )
                _print_unknown_model_help(
                    args.model, full_path_example="mlx-community/Qwen3.5-9B-4bit"
                )
                sys.exit(1)
        # Round 16 codex catch: record the resolved (or already-canonical)
        # model so ``session_end`` can report what this invocation loaded.
        # ``normalize_model_path`` inside the emit helper redacts local
        # paths to the literal ``<local>`` token, so we don't need to
        # filter here. Captured after the error-fail path so we never
        # record a model that failed validation.
        _session_models_requested.append(args.model)

    # --- BEGIN B2: auto-pull confirmation gate -------------------------
    # For subcommands that may trigger a first-time download of a large
    # repo (chat/run/serve/pull/bench), warn the user before kicking off
    # a multi-GB transfer. Cached repos and small downloads pass through
    # invisibly. Env override: RAPID_MLX_AUTO_PULL=1. See
    # ``vllm_mlx/_download_gate.py`` for the policy.
    #
    # Codex round 1 surfaced two ordering issues:
    #   (a) the chat REPL spawns its own ``serve`` subprocess after the
    #       parent already gated; without RAPID_MLX_CHAT_SPAWN=1 in the
    #       child env, the second main() would re-prompt (or worse,
    #       deadlock on a non-TTY child stdin path that doesn't reach
    #       the early-return).
    #   (b) the env / TTY checks belong *before* the 5-second HF
    #       metadata fetch — otherwise every CI run that sets
    #       RAPID_MLX_AUTO_PULL=1 still pays the network round-trip.
    # Single-use marker: pop the env var as soon as we observe it so a
    # grandchild ``rapid-mlx`` spawn (e.g. a nested invocation from a
    # user hook, a doctor self-probe, or some future hub helper) does
    # NOT inherit the bypass. Codex round-2 BLOCKING #2.
    _chat_spawn_child = os.environ.pop("RAPID_MLX_CHAT_SPAWN", "") == "1"

    _GATED_COMMANDS = {"chat", "run", "serve", "pull", "bench"}
    if (
        getattr(args, "command", None) in _GATED_COMMANDS
        and hasattr(args, "model")
        and args.model
        and "/" in args.model  # only HF-style repo ids; local paths skip
        and not os.path.exists(args.model)
        and not _chat_spawn_child
    ):
        # Cheap checks first: env override and non-TTY both short-circuit
        # without touching the HF API. ``confirm_or_abort`` re-checks
        # both internally; we mirror them here so we can skip the size
        # estimate as well.
        _env_val = os.environ.get("RAPID_MLX_AUTO_PULL", "").strip().lower()
        _auto_yes = _env_val in {"1", "true", "yes"}
        _interactive = sys.stdin.isatty()
        if not _auto_yes and _interactive:
            from vllm_mlx._download_gate import (
                confirm_or_abort,
                estimate_repo_size_bytes,
                is_repo_cached,
            )

            if not is_repo_cached(args.model):
                confirm_or_abort(
                    args.model,
                    estimate_repo_size_bytes(args.model),
                )
    # --- END B2 --------------------------------------------------------

    if args.command == "serve":
        serve_command(args)
    elif args.command == "bench":
        bench_command(args)
    elif args.command == "models":
        models_command(args)
    elif args.command == "ls":
        # `ls` is a top-level alias for `models --cached`. Synthesize the
        # missing flag so models_command's branch fires without having to
        # know which command name it was invoked under.
        args.cached = True
        models_command(args)
    elif args.command == "version":
        print(f"rapid-mlx {_version}")
    elif args.command == "help":
        target = getattr(args, "subcommand", None)
        if not target:
            parser.print_help()
        elif target in subparsers.choices:
            subparsers.choices[target].print_help()
        else:
            print(f"Unknown subcommand: {target}")
            print("Run `rapid-mlx help` for the list of subcommands.")
            sys.exit(1)
    elif args.command == "pull":
        pull_command(args)
    elif args.command == "rm":
        rm_command(args)
    elif args.command == "ps":
        ps_command(args)
    elif args.command == "upgrade":
        upgrade_command(args)
    elif args.command in ("chat", "run"):
        # ``run`` is exposed as a subparser alias for Ollama compatibility;
        # argparse routes via ``aliases=`` but reports the user-typed name
        # on ``args.command``. Both names land here.
        chat_command(args)
    elif args.command == "info":
        info_command(args)
    elif args.command == "agents":
        agents_command(args)
    elif args.command == "doctor":
        from vllm_mlx.doctor.cli import doctor_command

        doctor_command(args)
    elif args.command == "telemetry":
        telemetry_command(args)
    elif args.command == "share":
        from vllm_mlx.share.cli import share_command

        share_command(args)
    elif args.command == "launch":
        from vllm_mlx.launch.cli import launch_command

        launch_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
