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
import shlex
import sys
from collections.abc import Callable

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


def non_negative_int(value: str) -> int:
    """Argparse ``type`` callable: parse a ``>= 0`` integer.

    Rejects a negative value at parse time so a bad ``--response-cache-
    entries -5`` fails immediately with a clear argparse error, before any
    model download or load. ``SchedulerConfig.__post_init__`` also rejects
    negatives, but for ``serve`` that check runs only after the expensive
    download/load, so the early argparse guard gives the user faster,
    clearer feedback. The construction-time check stays as defense in
    depth.
    """
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected a non-negative integer, got {value!r}"
        ) from None
    if n < 0:
        raise argparse.ArgumentTypeError(f"expected a non-negative integer, got {n}")
    return n


def positive_int(value: str) -> int:
    """Argparse ``type`` callable: parse a strictly positive integer."""
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected a positive integer, got {value!r}"
        ) from None
    if n <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {n}")
    return n


def _vision_pixel_bounds_error(min_pixels: int, max_pixels: int) -> str | None:
    if min_pixels and max_pixels and min_pixels > max_pixels:
        return "--vision-min-pixels must not exceed --vision-max-pixels"
    return None


def positive_finite_float(value: str) -> float:
    """Argparse type for positive, finite resource-budget values."""
    import math

    try:
        number = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected a positive finite number, got {value!r}"
        ) from None
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError(
            f"expected a positive finite number, got {value!r}"
        )
    return number


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

    # Validate the port range up front. ``socket.bind()`` raises
    # ``OverflowError`` (NOT an ``OSError`` subclass) for a port outside
    # 0-65535, so the ``except OSError`` collision handler below would let
    # it escape as a raw traceback (dogfood #2125: ``--port 99999`` printed
    # ``OverflowError: bind(): port must be 0-65535`` instead of a friendly
    # message). Catch the typo here — before any probe — and emit the same
    # actionable style as the port-in-use path. ``0`` stays valid: it asks
    # the OS for an ephemeral port, which uvicorn binds normally.
    if not 0 <= port <= 65535:
        print(f"\n  Error: --port {port} is out of range. Ports must be 0-65535.")
        print(f"  Try a valid port: rapid-mlx serve {model} --port 8000")
        sys.exit(1)

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


def _print_port_collision_and_exit(
    host: str, port: int, *, in_listen_fd_mode: bool
) -> None:
    """Print a Sven-style supervisor-friendly EADDRINUSE message to
    stderr and ``sys.exit(1)``. Single SSOT so both the host/port and
    ``--listen-fd`` failure paths emit a consistent operator-facing
    message and the exit code stays non-zero in both.

    In ``--listen-fd`` mode the ``host``/``port`` args don't describe
    the real bind (the supervisor owns it), so we omit the
    port-specific ``lsof -i :N`` hint and reference the inherited fd
    instead — otherwise the operator would chase a port the rapid-mlx
    process never tried to bind (codex round-1 NIT #3).
    """
    if in_listen_fd_mode:
        print(
            "\n  Error: bind() failed on the supervisor-provided "
            "--listen-fd. The inherited socket is unusable. Re-launch "
            "with a fresh socket activation or fall back to --host/--port.",
            file=sys.stderr,
        )
    else:
        display_host = host or "0.0.0.0"
        print(
            f"\n  Error: Port {port} already in use on {display_host}. "
            f"Choose a different --port or stop the existing server "
            f"(lsof -i :{port}).",
            file=sys.stderr,
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

    R13 Sven B1: also the single CLI-side chokepoint that converts a
    uvicorn-side bind failure into the friendly "Port N already in
    use…" message + ``sys.exit(1)`` the operator's supervisor (systemd,
    launchd, k8s) needs to detect failure. Three paths feed in:

      * ``OSError(EADDRINUSE)`` raised through uvicorn (older uvicorns,
        ``--listen-fd`` mode where ``socket.fromfd`` / ``create_server``
        fail before uvicorn's own except arms) — caught directly.
      * ``SystemExit(1)`` from uvicorn>=0.34: ``Server.startup`` catches
        the bind ``OSError``, ``logger.error(exc)``s it (raw
        ``ERROR: [Errno 48] …``), and ``sys.exit(1)``s before our
        ``except OSError`` can fire. The exit code is already non-zero,
        but the friendly hint is missing — so we re-detect by probing
        the same ``(host, port)`` ourselves and, if it's busy, re-emit
        the Sven-style message before propagating the same non-zero
        exit (codex round-1 BLOCKING #2).
      * Any other ``SystemExit`` from uvicorn (clean lifespan shutdown,
        TLS misconfig, etc.) is left untouched.

    ``_port_preflight_or_die`` (run earlier in ``serve_command``)
    handles the common pre-load case at zero cost — this layer is the
    TOCTOU-race / fd-mode safety net.
    """
    import errno

    import uvicorn

    listen_fd = getattr(args, "listen_fd", None)
    try:
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
    except OSError as exc:
        # Direct EADDRINUSE — older uvicorn, ``--listen-fd`` mode bind
        # path. Translate to the friendly message; unrelated OSErrors
        # (e.g. EACCES on a low port) keep their original trace and
        # propagate so the failure is debuggable.
        if exc.errno == errno.EADDRINUSE:
            _print_port_collision_and_exit(
                args.host, args.port, in_listen_fd_mode=listen_fd is not None
            )
        raise
    except SystemExit as exc:
        # uvicorn>=0.34 catches the bind ``OSError`` in ``Server.startup``,
        # ``logger.error(exc)``s it (raw ``[Errno 48]`` line — not the
        # friendly hint a supervisor operator needs), and ``sys.exit(1)``s
        # before our ``except OSError`` can fire. The exit code is
        # already non-zero so the supervisor-failure-detection contract
        # holds, but we re-emit the Sven-style message on top so the
        # operator's grep for "already in use" still hits. Only override
        # the message when a probe confirms the port really IS in use —
        # other ``SystemExit(1)`` paths (TLS, lifespan, etc.) must keep
        # uvicorn's own diagnostic so we don't paper over them.
        #
        # Outer guard: codex round-2 BLOCKING — if the probe itself
        # raises (TypeError from a non-string host, gaierror, etc.) the
        # caller's ``SystemExit`` MUST still propagate. Wrap the
        # discriminator call so any probe-side exception is silently
        # absorbed and the original ``raise`` below re-delivers
        # uvicorn's exit. ``_port_is_busy`` ALSO defends internally,
        # but a future refactor that drops that guard (or a monkeypatch
        # in a test harness) must not corrupt the failure signal.
        if exc.code in (1, "1") and listen_fd is None:
            try:
                busy = _port_is_busy(args.host, args.port)
            except BaseException:
                busy = False
            if busy:
                _print_port_collision_and_exit(
                    args.host, args.port, in_listen_fd_mode=False
                )
        raise


def _port_is_busy(host: str, port: int) -> bool:
    """Best-effort probe: is ``(host, port)`` already bound by another
    process? Used by ``_run_uvicorn`` to disambiguate an uvicorn
    ``SystemExit(1)`` triggered by a bind collision from one triggered
    by an unrelated startup failure (TLS, lifespan, etc.).

    Returns True iff a fresh ``socket.bind`` fails with EADDRINUSE.
    Returns False on ANY other outcome (clean bind, ENETDOWN, EACCES,
    ``gaierror``, ``TypeError`` from a ``None`` host, etc.) so the
    caller's original ``SystemExit`` propagates untouched — codex
    round-2 BLOCKING was that a probe-side ``TypeError`` could replace
    uvicorn's ``SystemExit(1)`` with a misleading traceback. The probe
    is a HEURISTIC: a false-negative is acceptable (operator still
    gets uvicorn's diagnostic + non-zero exit, just no friendly hint),
    a probe-side raise that masks uvicorn's failure is not.
    """
    import errno
    import socket

    if not isinstance(host, str) or not host:
        # uvicorn accepts ``host=""`` as the wildcard alias, but
        # ``socket.bind(("", port))`` works on AF_INET — pre-normalize
        # to avoid a probe-side ``TypeError`` for non-string hosts
        # (sentinel values, configured-via-env edge cases).
        host = "0.0.0.0"

    try:
        family = socket.AF_INET6 if _is_ipv6_host(host) else socket.AF_INET
        with socket.socket(family, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind((host, port))
            except OSError as exc:
                return exc.errno == errno.EADDRINUSE
    except BaseException:
        # Outer guard: ANY probe-side failure (socket constructor,
        # gaierror, host normalization, etc.) MUST NOT mask the caller's
        # ``SystemExit``. Swallow and report "not busy" so the original
        # exception re-raises cleanly. ``BaseException`` is intentional
        # — even a stray ``KeyboardInterrupt`` during the probe should
        # not corrupt the supervisor-facing failure signal; the caller's
        # ``raise`` will re-deliver any interrupt on the next event loop.
        return False
    return False


def _print_unknown_model_help(name: str, *, full_path_example: str) -> None:
    """Print fuzzy suggestions + a curated popular-models hint.

    Replaces the older "Did you mean: X?" + "Run `rapid-mlx models`" pattern
    that left users empty-handed when no close fuzzy match existed
    (e.g. ``rapid-mlx chat gemma4-27b`` returned zero suggestions, told the
    user to run another command, and gave no hint of what was actually
    supported). Now: always show *something* — fuzzy matches when we have
    them, curated popular aliases when we don't.
    """
    from vllm_mlx.model_aliases import POPULAR_ALIASES, suggest_similar

    suggestions = suggest_similar(name)
    if suggestions:
        print(f"  Did you mean: {', '.join(suggestions)}?")
    else:
        print(f"  Try one of: {', '.join(POPULAR_ALIASES)}")
    # No hardcoded alias total here. ``models`` splits the registry into
    # tagged sections (chat / audio / video / image), each with its own
    # accurate per-section count, so a single grand total printed here can
    # only contradict the first header a user lands on (dogfood #2126:
    # this line said "182 aliases" while ``models`` opened with "172").
    # Let ``models`` be the single source of truth for the counts.
    print("  Run `rapid-mlx models` to see all available aliases,")
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


def _resolve_audio_download_alias(command: str | None, model: str) -> str | None:
    """Resolve a short audio alias to its concrete HF id for ``pull``/``rm``.

    #991: the CLI's alias resolver (``resolve_model``) only knows *text*
    aliases (``aliases.json``); audio aliases live in the separate audio
    registry and resolve via :func:`resolve_audio_alias`. ``serve`` keeps
    the short alias and resolves it at request time (see
    :func:`_resolve_audio_model_for_serve` and the route-level
    ``TTS_MODEL_ALIASES`` / ``STT_MODEL_ALIASES`` lookup), but ``pull`` and
    ``rm`` consume ``args.model`` verbatim with no request-time path:

      * ``pull`` — the R2 mirror catalog is keyed by ``hf_path``, so a bare
        alias misses the mirror and then 404s at HF
        (``snapshot_download("whisper")``).
      * ``rm`` — the HF cache is scanned by ``models--<owner>--<repo>``,
        which a bare alias can never match.

    Returns the resolved HF id when ``command`` is ``pull``/``rm`` AND
    ``model`` is a known audio alias; otherwise ``None`` — no rewrite, so
    ``serve``/``chat``/``bench`` keep the short alias and non-audio names
    fall through to the text path unchanged.
    """
    if command not in {"pull", "rm"}:
        return None
    from .audio.registry import resolve_audio_alias

    entry = resolve_audio_alias(model)
    return entry.hf_id if entry is not None else None


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
    # R11-K / task #258: honor ``--served-model-name`` on the audio
    # path, mirroring the text-mode contract at ``server.load_model``
    # (``_model_name = served_model_name or model_name``). Pre-fix the
    # audio dispatcher ignored the flag, so operators wrapping
    # ``rapid-mlx serve kokoro`` behind a gateway with a stable
    # ``model_name`` saw the raw HF id on ``/v1/models`` and the
    # gateway's model-id allowlist 404'd. The underlying HF id stays
    # on ``_model_path`` (cache dir / engine input), and the friendly
    # short alias stays on ``_model_alias`` so ``/v1/models`` lists
    # both the custom name AND the alias — same wire shape as text.
    _served_name = getattr(args, "served_model_name", None)
    server._model_name = _served_name or entry.hf_id
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
    # WH-1: OPT-IN Host-header allowlist (DNS-rebinding hardening).
    server.configure_trusted_hosts(getattr(args, "trusted_hosts", None))
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

    # Task #292: register ``/v1/audio/*`` routes. ``server._model_alias``
    # / ``server._model_name`` were just stamped with the registry-known
    # audio alias above, so the registry-driven branch of
    # :func:`register_audio_routes_if_enabled` is what fires here — the
    # ``--enable-audio`` flag is for the text-mode-with-audio escape
    # hatch, not the audio-mode boot path. Skipping the call would leave
    # text-only behaviour on an audio server, with /v1/audio/* returning
    # 404 (the exact symmetric mistake the unconditional pre-fix made on
    # text-only servers). Idempotent — safe even if a future refactor
    # adds a second call site.
    server.register_audio_routes_if_enabled()

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

    # R11-K / task #258: honor ``--embedding-model`` on the audio
    # path. The shared helper (``_load_embedding_model_or_exit``) is
    # intentionally orthogonal to the text-LM engine — it only goes
    # through ``server.load_embedding_model`` — so audio + embedding
    # compose cleanly: the audio engines stay lazy on /v1/audio/*
    # while the embeddings sidecar serves /v1/embeddings from the
    # same FastAPI app. Mirrors the text-mode call site at
    # ``serve_command`` (post-``load_model``); see the helper's
    # docstring "Audio-mode integration" note (R11-K coordination)
    # — single source of truth for the install + alias + error wrap.
    # Ordered after the banner so the operator sees the audio model
    # banner FIRST (matches the text-mode visual ordering where the
    # ``Model:`` line prints before ``Pre-loading embedding model:``).
    if getattr(args, "embedding_model", None):
        _load_embedding_model_or_exit(args, server.load_embedding_model)

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
    # Validate the embedding input-length setting up front so a bad value
    # is a clean usage error (exit 2), not a mid-load crash (issue #1381).
    from .embedding import normalize_max_length_setting

    try:
        max_length = normalize_max_length_setting(
            getattr(args, "embedding_max_length", "auto")
        )
    except ValueError as exc:
        print(f"error: --embedding-max-length {exc}", file=sys.stderr)
        sys.exit(2)
    overflow_policy = getattr(args, "embedding_overflow_policy", "truncate")

    not_found_exc_classes = _embedding_not_found_exception_classes()
    try:
        load_fn(
            args.embedding_model,
            lock=True,
            max_length=max_length,
            overflow_policy=overflow_policy,
        )
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
    - Files already cached for the current Hub revision are excluded from the
      required download size.
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

    # Which directory inside the repo this alias actually needs. Resolved
    # OUTSIDE the cache probe below: that probe is best-effort and swallows
    # its own failures, and folding the prefix into it meant one flaky
    # ``try_to_load_from_cache`` call silently reverted the size estimate to
    # the whole repo — demanding disk for eight quantizations and refusing a
    # machine with ample room for the one being fetched.
    from vllm_mlx.model_aliases import checkpoint_prefix

    _prefix = checkpoint_prefix(model_name)

    # Keep the historical fast path for ordinary text-model repositories.
    # Their loaders may intentionally fetch only one of several weight formats,
    # so treating every uncached sibling as an impending download would count
    # optional artifacts that the selected loader never requests.
    try:
        from huggingface_hub import try_to_load_from_cache

        cached_config = try_to_load_from_cache(
            model_name,
            f"{_prefix}config.json",
        )
        if isinstance(cached_config, str) and os.path.exists(cached_config):
            return
    except Exception:
        pass

    # Query HF for the current revision and repo size + free space on the
    # actual HF cache filesystem. Component-layout repositories such as mflux
    # have no root config.json, so exclude each file already cached at the
    # current revision instead of assuming the entire snapshot is missing.
    # Only this alias's checkpoint counts — a subfolder-per-quant repo would
    # otherwise demand disk for seven quantizations nobody asked for.
    try:
        from huggingface_hub import model_info, try_to_load_from_cache
        from huggingface_hub.constants import HF_HUB_CACHE

        info = model_info(model_name, files_metadata=True)
        revision = getattr(info, "sha", None)
        siblings = [
            sibling
            for sibling in (getattr(info, "siblings", None) or [])
            if hasattr(sibling, "size")
            and hasattr(sibling, "rfilename")
            and (not _prefix or sibling.rfilename.startswith(_prefix))
        ]
        model_size_bytes = 0
        for sibling in siblings:
            try:
                cached = try_to_load_from_cache(
                    model_name,
                    sibling.rfilename,
                    revision=revision,
                )
                is_cached = isinstance(cached, str) and os.path.exists(cached)
            except Exception:
                # A cache lookup is only an optimization. If it is unreadable,
                # retain the file in the conservative download estimate.
                is_cached = False
            if not is_cached:
                model_size_bytes += sibling.size or 0
        if model_size_bytes == 0:
            return  # Nothing left to download (or size unavailable).

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
        print(f"    Download size: {model_size_gb:>7.1f} GB")
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


def _gather_kv_cache_dtype_inputs(model_name: str) -> tuple[dict | None, dict | None]:
    """Best-effort collect the inputs ``resolve_kv_cache_dtype`` consumes.

    R15 task #300: the safelist that downgrades int4 → bf16 for sliding-
    window and MLA models needs the HF ``config.json`` (``sliding_window``,
    ``q_lora_rank`` / ``kv_lora_rank``) plus any alias-level
    ``sliding_window`` / ``is_mla`` hints. We intentionally avoid
    network fetches here — both signals come from data that's already
    on disk (aliases.json) or that will be downloaded for the model
    load anyway (HF config). If neither is available (offline, gated
    repo, brand-new release), the substring fallback in
    :func:`vllm_mlx.kv_cache_dtype.resolve_kv_cache_dtype` still catches
    the documented families by name.

    Returns:
        A ``(hf_config, alias_metadata)`` pair. Either or both may be
        ``None`` when the inputs aren't reachable.
    """
    hf_cfg: dict | None = None
    alias_meta: dict | None = None

    # Alias metadata — pull straight from the loaded profile so a
    # contributor-curated override (``"sliding_window": true``) wins
    # over the substring heuristic.
    try:
        from .model_aliases import resolve_profile

        profile = resolve_profile(model_name)
        if profile is not None:
            alias_meta = {
                "hf_path": getattr(profile, "hf_path", None),
                # AliasProfile doesn't have ``sliding_window`` / ``is_mla``
                # fields today (R15 #300 intentionally avoids a frozen-
                # dataclass schema bump). The substring fallback covers
                # the in-tree aliases; we leave the hook here so a
                # future closed-key extension picks them up automatically.
                "sliding_window": getattr(profile, "sliding_window", False),
                "is_mla": getattr(profile, "is_mla", False),
            }
    except Exception:
        # Alias resolution must never block server start. The substring
        # fallback covers the documented families even with no profile.
        alias_meta = None

    # HF config — read from the local HF cache only. We're inside the
    # serve preflight path so a network round-trip would be cheap (the
    # model load follows immediately anyway), but staying file-local
    # keeps this helper safe to call in tests and air-gapped installs.
    try:
        import json as _json
        import os as _os

        from huggingface_hub import try_to_load_from_cache as _cache_lookup

        hf_path = (alias_meta or {}).get("hf_path") or model_name
        # Local model directory (e.g. a freshly ``mlx_lm convert``-ed
        # ``-rapid`` build served by path): read its ``config.json``
        # directly. ``try_to_load_from_cache`` only resolves HF repo ids,
        # so without this a local path yields ``hf_cfg=None`` and the MTP
        # eligibility gate wrongly rejects an otherwise-eligible checkpoint.
        _local_cfg = _os.path.join(model_name, "config.json") if model_name else None
        if _local_cfg and _os.path.isfile(_local_cfg):
            with open(_local_cfg) as fh:
                hf_cfg = _json.load(fh)
        elif hf_path:
            cached = _cache_lookup(repo_id=hf_path, filename="config.json")
            if cached and _os.path.exists(cached):
                with open(cached) as fh:
                    hf_cfg = _json.load(fh)
    except Exception:
        hf_cfg = None

    return hf_cfg, alias_meta


def _apply_mtp_cli_model_type_reconciliation(
    *,
    scheduler_config,
    hf_cfg_eligibility,
    logger=None,
) -> None:
    """Thread the eligibility gate's ``model_type`` down into
    ``SchedulerConfig.mtp_model_type``.

    Codex round-I BLOCKING #3 / round-K BLOCKING #2:
    ``serve_command`` reads the HF config on two paths — an early
    best-effort read that populates
    ``scheduler_config.mtp_model_type``, and a second read inside
    the MTP eligibility gate that decides accept/reject. Under a
    transient first-read failure (warm HF cache invalidated,
    filesystem hiccup, etc.), the first read returned ``None`` and
    the second read succeeded. The engine then treated the
    operator's explicit MTP speculative-config request as
    "non-CLI-vetted" and soft-skipped dispatch failures — silently
    degrading MTP to a no-op.

    This helper resolves the skew: it treats the eligibility gate's
    read as the source of truth (the gate accepted the config, so
    it MUST be correct), promotes ``model_type`` into
    ``scheduler_config.mtp_model_type``, and hard-fails at
    ``sys.exit(2)`` if the eligibility read somehow lacks a string
    ``model_type`` (should be unreachable per
    ``detect_mtp_eligibility``'s own gates, but defensive).

    Extracted so tests can drive this helper end-to-end. A
    purely-inline reconciliation block in ``serve_command`` would
    let tests still pass after the block was deleted, because
    replaying the logic inline in a test never exercises the
    production import path (round-K BLOCKING #2).

    Args:
      scheduler_config: The already-constructed ``SchedulerConfig``.
        Its ``mtp_model_type`` field is mutated in place.
      hf_cfg_eligibility: The dict returned by the eligibility
        gate's ``_gather_kv_cache_dtype_inputs`` call.
      logger: Optional ``logging.Logger``. If provided, a warning is
        emitted when the CLI-thread read disagreed with the
        eligibility read. Kept optional so tests don't have to
        conjure a real logger; the ``print`` fallback below covers
        the ``None`` case with the same operator-visible message.
    """
    _eligibility_model_type: str | None = None
    if isinstance(hf_cfg_eligibility, dict):
        _mt_from_eligibility = hf_cfg_eligibility.get("model_type")
        if isinstance(_mt_from_eligibility, str):
            _eligibility_model_type = _mt_from_eligibility
    if _eligibility_model_type is None:
        # Should be unreachable: detect_mtp_eligibility gates on a
        # dict-shaped config with a string model_type. But if some
        # detector refactor ever relaxes that invariant, we MUST
        # hard-fail rather than boot in the silent-skip state.
        print(
            "error: MTP speculative-config eligibility passed but the CLI "
            "could not extract config.json::model_type — this is a "
            "plumbing skew between detect_mtp_eligibility (accepted "
            "the config) and this CLI reconciliation block. Refusing "
            "to boot: the engine's dispatch requires the CLI-vetted "
            "model_type to hard-fail on non-attached dispatch results "
            "(round-E BLOCKER #2 contract). File an issue with the "
            'output of `python -c "import json; '
            "print(json.load(open('config.json')).get('model_type'))\" "
            "run in the model directory.",
            file=sys.stderr,
        )
        sys.exit(2)
    # If the earlier best-effort read succeeded but disagreed with
    # the eligibility read, prefer the eligibility read — it drives
    # the accept/reject decision, so it MUST be the source of truth
    # for the dispatch-side CLI-vetted marker.
    _prior = getattr(scheduler_config, "mtp_model_type", None)
    if _prior is not None and _prior != _eligibility_model_type:
        _msg = (
            "[MTP] CLI-thread config read disagreed with the "
            "eligibility gate on model_type (%r vs %r); using the "
            "eligibility gate's value."
        )
        if logger is not None:
            logger.warning(_msg, _prior, _eligibility_model_type)
        else:
            print(_msg % (_prior, _eligibility_model_type), file=sys.stderr)
    scheduler_config.mtp_model_type = _eligibility_model_type


def _check_alias_min_memory(user_typed: str) -> None:
    """Alias-level unified-memory guard — warn if the user's Mac is
    smaller than the alias's declared ``min_memory_gb`` floor.

    codex #1069 round 3 [NIT #3]: fires alongside (and before) the
    generic model-size / pressure check in ``_check_memory_capacity``.
    That check reads live weights + free RAM, so it can only warn AFTER
    the download completes (or from the HF file-size API, which lags).
    This one fires up front from the alias profile so a user on a 128
    GB Max sees the actionable pointer BEFORE the 166 GB
    ``hy3-preview-4bit`` download starts.

    Best-effort: warns loudly (never aborts) so an operator with a
    borderline machine, headless setup, or unusual memory allocator
    can still opt in. Silent no-op when:
      - The alias has no ``min_memory_gb`` metadata (every model we
        ship under 100 GB weights).
      - The user typed an HF path directly instead of an alias.
      - psutil is unavailable / raises.
    """
    try:
        from .model_aliases import resolve_profile

        profile = resolve_profile(user_typed)
    except Exception:
        return
    if profile is None:
        return
    floor_gb = getattr(profile, "min_memory_gb", None)
    if floor_gb is None or floor_gb <= 0:
        return

    try:
        import psutil

        total_ram_gb = psutil.virtual_memory().total / (1024**3)
    except Exception:
        return
    if total_ram_gb <= 0:
        return
    if total_ram_gb >= floor_gb:
        return  # Machine is big enough — silent.

    is_tty = sys.stdout.isatty() and "NO_COLOR" not in os.environ
    YELLOW = "\x1b[33m" if is_tty else ""
    BOLD = "\x1b[1m" if is_tty else ""
    RESET = "\x1b[0m" if is_tty else ""
    print(
        f"\n{YELLOW}{BOLD}⚠  Ultra-only alias '{user_typed}' declares a "
        f"{floor_gb:.0f} GB unified-memory floor, but this Mac reports "
        f"{total_ram_gb:.1f} GB.{RESET}"
    )
    print(
        f"{YELLOW}   The model weights are large enough to OOM the Metal "
        "allocator (or kernel-panic on macOS < 15.2, issue #324) before "
        f"the first token generates.{RESET}"
    )
    print(
        f"{YELLOW}   Recommended: pick a Tier-1 alias sized for this "
        "machine (`rapid-mlx models` for the full list). "
        f"Proceeding anyway…{RESET}\n"
    )


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

            from vllm_mlx.model_aliases import checkpoint_prefix

            # One repo, one folder per quantization: the working set is
            # this alias's folder, not the eight-checkpoint repo. Sizing
            # the whole repo made a 1.6 GB model project a 28 GB working
            # set and fire the kernel-panic warning on a 32 GB Mac.
            prefix = checkpoint_prefix(model_name)
            cached = try_to_load_from_cache(model_name, f"{prefix}config.json")
            if isinstance(cached, str) and os.path.exists(cached):
                # Already-downloaded model: walk the checkpoint directory.
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
                    and (not prefix or s.rfilename.startswith(prefix))
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


class _StatusSpinner:
    """Animated ``⠋ <label> Ns`` spinner on stderr for a blocking phase.

    Context manager. On a TTY it spawns a daemon thread that redraws the
    label + elapsed seconds at ~10 fps; :meth:`stop` (idempotent) clears the
    line. Non-TTY / ``NO_COLOR`` → fully inert (no thread, no output) so CI
    logs and pipes stay clean. :meth:`stop` is public so a caller can retire
    the spinner mid-``with`` — e.g. from a download's ``on_pull_start`` hook,
    right before the first real progress line — while ``__exit__`` remains a
    belt-and-braces clear on every exit path.

    Mirrors the inline spinner in :func:`_wait_for_chat_server` (same frames /
    stream) so the cold-download "Resolving…" phase looks like the warm
    "loading model…" phase the REPL already shows.
    """

    _FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, label: str, *, stream=None) -> None:
        self._label = label
        self._stream = stream if stream is not None else sys.stderr
        # ``isatty`` can be absent on exotic stream stand-ins; treat missing
        # as non-TTY so we stay inert rather than crash.
        _isatty = getattr(self._stream, "isatty", None)
        self._enabled = bool(_isatty and _isatty()) and "NO_COLOR" not in os.environ
        self._done = False
        self._start = 0.0
        self._thread = None
        import threading

        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        # Serializes the worker's frame writes against ``stop``'s clear so the
        # clear is guaranteed to be the LAST thing written to the stream (no
        # post-clear redraw). Held only around the brief write/flush — never
        # across the inter-frame wait — so ``stop`` can grab it promptly.
        self._draw_lock = threading.Lock()

    def _run(self) -> None:
        import time

        tick = 0
        while not self._stop_event.is_set():
            with self._draw_lock:
                # Re-check under the lock: if ``stop`` set the event while we
                # waited for the lock, do not draw — otherwise this frame would
                # land AFTER stop's clear.
                if self._stop_event.is_set():
                    return
                try:
                    elapsed = int(time.monotonic() - self._start)
                    ch = self._FRAMES[tick % len(self._FRAMES)]
                    self._stream.write(
                        f"\r  \x1b[36m{ch}\x1b[0m {self._label} \x1b[2m{elapsed}s\x1b[0m"
                    )
                    self._stream.flush()
                except (ValueError, OSError):
                    return  # stream closed underneath us — stop quietly.
            tick += 1
            self._stop_event.wait(0.1)

    def __enter__(self) -> "_StatusSpinner":
        if self._enabled:
            import threading
            import time

            self._start = time.monotonic()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def stop(self) -> None:
        """Halt the spinner and clear its line. Idempotent + thread-safe."""
        with self._lock:
            if self._done:
                return
            self._done = True
        self._stop_event.set()
        if self._thread is None:
            return  # inert (non-TTY) or never entered — nothing drawn.
        self._thread.join(timeout=1.0)
        # Clear under the draw lock so the clear is the worker's final write.
        # If the worker is wedged in a blocked ``write()`` and never frees the
        # lock within the timeout, the stream is broken — skip the clear rather
        # than risk a garbled interleave or an unbounded hang (a stray frame on
        # a broken stream won't render anyway). ``acquire`` doubles as the
        # confirmation that no worker write is in flight.
        if self._draw_lock.acquire(timeout=1.0):
            try:
                # Clear the whole line; the label + " Ns" fits well within a
                # generous fixed width (ANSI codes take no display columns).
                self._stream.write("\r" + " " * (len(self._label) + 24) + "\r")
                self._stream.flush()
            except (ValueError, OSError):
                pass
            finally:
                self._draw_lock.release()

    def __exit__(self, *exc: object) -> bool:
        self.stop()
        return False


def _try_mirror_prefetch(
    model_name: str,
    on_pull_start: Callable[[], None] | None = None,
) -> bool:
    """Pre-fetch a HuggingFace repo via R2-first / HF-fallback (per file).

    Delegates to :func:`vllm_mlx._mirror.download_with_mirror_fallback`.
    Returns ``True`` if the snapshot is fully populated (any mix of R2
    and HF). Returns ``False`` if the caller should fall through to the
    plain ``snapshot_download(repo_id)`` path (catalog unavailable for
    catalog-only paths, or one or more files failed both R2 and HF).

    ``on_pull_start`` is forwarded to the mirror and fired once, right before
    the first ``Pulling`` line — used to retire a "Resolving…" spinner.

    Set ``RAPID_MLX_MODEL_MIRROR=""`` to disable R2 entirely and force
    HuggingFace.

    Codex round-6 BLOCKING #2: the mirror module already returns
    ``False`` on every recoverable network/cache error, so the only
    catch worth doing here is ``ImportError`` (mirror module disabled
    or missing in a minimal install). Programmer errors propagate so
    bugs in the mirror module surface as real stack traces instead of
    silently routing to ``snapshot_download``.
    """
    from vllm_mlx.model_aliases import resolve_subfolder

    # The mirror mirrors whole repos; it has no notion of "just this
    # folder". For a repo that ships one checkpoint per quantization that
    # would hydrate every quant, so decline and let the HF path run with
    # ``allow_patterns``. None of these repos is mirrored today — this is
    # here so that mirroring one later can't silently turn a 1.6 GB pull
    # into a 20 GB one.
    if resolve_subfolder(model_name):
        return False
    try:
        from vllm_mlx._mirror import download_with_mirror_fallback
    except ImportError:
        # Mirror module not available (minimal-deps install or
        # deliberately removed). Use the legacy HF path.
        return False
    return download_with_mirror_fallback(model_name, on_pull_start=on_pull_start)


def _ensure_model_downloaded(
    model_name: str, *, force_disk_check: bool = False
) -> None:
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
        from vllm_mlx._download_gate import is_repo_cached, mflux_missing_weights

        if is_repo_cached(model_name):
            return
        # ``is_repo_cached`` probes for root ``model*.safetensors``, which an
        # mflux checkpoint never has — its weights live under ``transformer/``,
        # ``text_encoder/`` and ``vae/``. So a fully-downloaded image model
        # always failed that probe and fell through to the disk-space gate and
        # the mirror below, i.e. every single start of an image model needed
        # the network. On a hostile DNS path that is not a slow start but a
        # hung one: the socket sits in SYN_SENT against a poisoned address
        # while the UI shows "Starting" forever. Verified-complete component
        # weights are just as cached as a verified-complete root checkpoint.
        if mflux_missing_weights(model_name) == []:
            return
    except Exception:
        # Probe failed (filesystem permission error, unexpected layout) —
        # fall through to the heavy snapshot_download path; HF will
        # short-circuit on its own cache check if the repo really is
        # fully present.
        pass

    # Disk-space gate + mirror pull. Both the disk probe (HF ``model_info``)
    # and the mirror's own metadata + ``/api/models`` catalog round-trips run
    # BEFORE the first "Pulling"/"First-time download" line — up to a few
    # seconds of total silence at the most fragile first-run moment, where it
    # reads as a hang. Cover it with a "Resolving…" spinner (TTY-only; inert on
    # CI/pipe) that the mirror retires via ``on_pull_start`` the instant real
    # progress begins, and that ``__exit__`` clears on every other exit path.
    _short = model_name.split("/")[-1]
    spinner = _StatusSpinner(f"Resolving {_short} …")
    with spinner:
        # ``_check_disk_space`` queries HF for the repo size and aborts with a
        # clear message + exit(1) if there isn't enough room on the resolved
        # HF cache filesystem. Clear the spinner first on that fatal path so
        # the abort message and the shell prompt after it land on clean lines.
        try:
            _check_disk_space(model_name, force=force_disk_check)
        except SystemExit:
            spinner.stop()
            raise

        # User-configured mirror path (R2/S3/any HTTP host). When the mirror
        # serves every file the repo declares, populate the HF cache layout
        # ourselves and skip snapshot_download. On any miss we fall through
        # to the normal HuggingFace download below.
        mirror_ok = _try_mirror_prefetch(model_name, on_pull_start=spinner.stop)
    if mirror_ok:
        return

    try:
        from huggingface_hub import model_info, snapshot_download

        from vllm_mlx._download_gate import (
            _HF_RESOLVE_TIMEOUT_SECONDS,
            call_with_deadline,
            pin_main_ref,
        )
        from vllm_mlx.model_aliases import subfolder_allow_patterns

        # Repos that ship one folder per quantization: fetch and measure
        # ONLY this alias's folder. Without the filter a 1.6 GB 4-bit pull
        # reports and downloads the repo's whole ~20 GB quant matrix.
        allow_patterns = subfolder_allow_patterns(model_name)
        _prefix = allow_patterns[0][:-1] if allow_patterns else None

        # This bounded call resolves the revision ahead of the download.
        # ``snapshot_download`` otherwise resolves it through httpx with
        # an explicit ``timeout=None``, which *disables* the client timeout
        # rather than inheriting it, so that call has no deadline and hangs
        # indefinitely on a blackholed route — the desktop then sits at
        # "Starting" until its 30-minute stall window. Proving the Hub answers
        # within a deadline first turns that hang into an error.
        #
        # Pinning the resolved SHA is essential: treating this probe merely as
        # a reachability check leaves a TOCTOU window where the network can go
        # dark before snapshot_download repeats the same unbounded lookup.
        # huggingface_hub does not write refs/main for an explicit SHA, so we
        # publish that ref ourselves, atomically, only after the download wins.
        size_gb = 0.0
        resolved_sha: str | None = None
        try:
            info = call_with_deadline(
                model_info,
                _HF_RESOLVE_TIMEOUT_SECONDS,
                model_name,
                files_metadata=True,
            )
            resolved_sha = getattr(info, "sha", None)
            if not resolved_sha:
                raise RuntimeError("HuggingFace metadata did not include a revision")
            size_bytes = sum(
                (s.size or 0)
                for s in (getattr(info, "siblings", None) or [])
                if hasattr(s, "size")
                and (_prefix is None or s.rfilename.startswith(_prefix))
            )
            size_gb = size_bytes / (1024**3)
        except TimeoutError:
            # The Hub did not answer within the deadline. Falling through to
            # ``snapshot_download`` would re-enter the unbounded lookup and
            # hang; the desktop would sit at "Starting" until its 30-minute
            # stall window. Abort via ``sys.exit`` rather than an exception:
            # the generic handler at the bottom of this function turns any
            # non-404 exception into "Pre-download skipped; server will retry"
            # and lets the serve subprocess start, which would walk straight
            # back into the same hang. ``SystemExit`` is re-raised there, which
            # is the same escape hatch ``_check_disk_space`` already uses.
            print(
                f"\n  Error: could not reach HuggingFace to resolve "
                f"{model_name} within {_HF_RESOLVE_TIMEOUT_SECONDS:.0f}s.\n"
                "  The model is not fully downloaded yet, so it cannot be "
                "started offline.\n"
                "  Check your network or proxy settings and try again.\n",
                file=sys.stderr,
            )
            sys.exit(1)
        except Exception:
            # Any other metadata failure stays best-effort: an outage, a gated
            # repo or a missing token costs us the size quote, and the download
            # proceeds to fail (or succeed) with its own clearer error.
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

        download_kwargs = {"revision": resolved_sha} if resolved_sha else {}
        if allow_patterns:
            snapshot_download(
                model_name, allow_patterns=allow_patterns, **download_kwargs
            )
        else:
            snapshot_download(model_name, **download_kwargs)
        if resolved_sha:
            pin_main_ref(model_name, resolved_sha)
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
        default=None,
        help="Fraction of prompt tokens to keep when compressing. Unset lets "
        "the engine resolve it: a per-alias ``pflash_keep_ratio`` override if "
        "the alias pins one (e.g. 0.50 for a ternary arch), else the default "
        "0.20 (the bench-validated profile in PR #649: TTFT 3.87x-8.5x, needle "
        "recall 5/5). An explicit value here always wins.",
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


def _alias_mtp_declaration(model_name) -> tuple[str | None, int | None]:
    """Return ``(mtp_draft_model, mtp_speculative_tokens)`` declared by an alias.

    ``(None, None)`` when the model is not a known alias, declares no MTP
    sidecar, or the registry cannot be read. Resolution is best-effort by
    design: this only supplies DEFAULTS for a request that already asked for
    MTP, so a registry problem must degrade to "no default" and let the
    injector's own hard-fail speak — never turn a serve into a crash of its
    own (#1998).

    ``mtp_speculative_tokens`` is returned only when it is a positive int. The
    alias schema already rejects the alternatives (``model_aliases`` requires
    ``mtp_draft_model`` alongside it), so this is belt-and-braces against a
    hand-edited registry rather than a live shape.
    """
    if not model_name:
        return None, None
    try:
        from .model_aliases import resolve_profile as _resolve_alias

        profile = _resolve_alias(model_name)
    except Exception:  # noqa: BLE001 — see docstring: never fail the serve here
        return None, None
    if profile is None:
        return None, None
    # Type-check BEFORE ``.strip()``: the value reaches us straight off a
    # profile object, and a non-str there would raise ``AttributeError`` out of
    # this helper — breaking the totality the docstring promises, in the exact
    # hand-edited-registry case it promises it for (codex nit).
    raw_sidecar = getattr(profile, "mtp_draft_model", None)
    sidecar = raw_sidecar.strip() or None if isinstance(raw_sidecar, str) else None
    if sidecar is None:
        # Depth without a sidecar is meaningless — the injector has nothing to
        # load — so don't hand back a lone K either.
        return None, None
    depth = getattr(profile, "mtp_speculative_tokens", None)
    if not isinstance(depth, int) or isinstance(depth, bool) or depth <= 0:
        depth = None
    return sidecar, depth


def _normalize_speculative_config_or_exit(args):
    """Parse ``--speculative-config`` and map methods to runtime fields."""
    import json
    import sys

    from .spec_decode.config import (
        SpeculativeConfigError,
        parse_speculative_config,
        require_migrated_speculative_config,
    )

    raw_config = getattr(args, "speculative_config", None)
    raw_config_was_explicit = raw_config is not None
    config = None

    def _fill_runtime_defaults(*, overwrite: bool) -> None:
        defaults = {
            "enable_ddtree": False,
            "enable_dflash": False,
            "spec_decode": "none",
            "dflash_drafter_path": "",
            "enable_mtp": False,
            "mtp_num_draft_tokens": 1,
            "mtp_optimistic": False,
            "mtp_sidecar": None,
            "mtp_max_k": 1,
            "mtp_disable_auto_k": False,
            "suffix_decoding": False,
        }
        for name, value in defaults.items():
            if overwrite or not hasattr(args, name) or getattr(args, name) is None:
                setattr(args, name, value)

    def _reject_no_spec_decode_runtime_conflicts() -> None:
        if not getattr(args, "no_spec_decode", False):
            return
        conflicts = []
        if getattr(args, "enable_ddtree", False):
            conflicts.append("enable_ddtree")
        if getattr(args, "enable_dflash", False):
            conflicts.append("enable_dflash")
        spec_decode = getattr(args, "spec_decode", "none")
        if spec_decode not in (None, "none"):
            conflicts.append(f"spec_decode={spec_decode}")
        if (getattr(args, "dflash_drafter_path", "") or "").strip():
            conflicts.append("dflash_drafter_path")
        if getattr(args, "enable_mtp", False):
            conflicts.append("enable_mtp")
        if (getattr(args, "mtp_sidecar", None) or "").strip():
            conflicts.append("mtp_sidecar")
        # Idempotency guard: after ``_fill_runtime_defaults(overwrite=True)``
        # a disabled config normalizes to ``mtp_max_k=1``. Only flag the
        # value as a ``--no-spec-decode`` conflict when it diverges from
        # that disabled default (i.e., an explicit non-default was passed).
        if getattr(args, "mtp_max_k", None) not in (None, 1):
            conflicts.append("mtp_max_k")
        if getattr(args, "mtp_num_draft_tokens", 1) != 1:
            conflicts.append("mtp_num_draft_tokens")
        if getattr(args, "mtp_disable_auto_k", False):
            conflicts.append("mtp_disable_auto_k")
        if getattr(args, "mtp_optimistic", False):
            conflicts.append("mtp_optimistic")
        if getattr(args, "suffix_decoding", False):
            conflicts.append("suffix_decoding")
        if getattr(args, "suffix_max_draft", None) is not None:
            conflicts.append("suffix_max_draft")
        if getattr(args, "suffix_max_suffix_len", None) is not None:
            conflicts.append("suffix_max_suffix_len")
        if getattr(args, "suffix_min_confidence", None) is not None:
            conflicts.append("suffix_min_confidence")
        if getattr(args, "suffix_min_draft_len", None) is not None:
            conflicts.append("suffix_min_draft_len")
        if not conflicts:
            return
        joined = ", ".join(conflicts)
        print(
            f"error: --no-spec-decode is mutually exclusive with {joined}.",
            file=sys.stderr,
        )
        sys.exit(2)

    def _legacy_speculative_fields() -> list[str]:
        fields = []
        if getattr(args, "enable_ddtree", False):
            fields.append("enable_ddtree")
        if getattr(args, "enable_dflash", False):
            fields.append("enable_dflash")
        spec_decode = getattr(args, "spec_decode", "none")
        if spec_decode not in (None, "none"):
            fields.append(f"spec_decode={spec_decode}")
        if (getattr(args, "dflash_drafter_path", "") or "").strip():
            fields.append("dflash_drafter_path")
        if getattr(args, "enable_mtp", False):
            fields.append("enable_mtp")
        if (getattr(args, "mtp_sidecar", None) or "").strip():
            fields.append("mtp_sidecar")
        if getattr(args, "mtp_max_k", None) is not None:
            fields.append("mtp_max_k")
        if getattr(args, "mtp_num_draft_tokens", 1) != 1:
            fields.append("mtp_num_draft_tokens")
        if getattr(args, "mtp_disable_auto_k", False):
            fields.append("mtp_disable_auto_k")
        if getattr(args, "mtp_optimistic", False):
            fields.append("mtp_optimistic")
        if getattr(args, "suffix_decoding", False):
            fields.append("suffix_decoding")
        if getattr(args, "suffix_max_draft", None) is not None:
            fields.append("suffix_max_draft")
        if getattr(args, "suffix_max_suffix_len", None) is not None:
            fields.append("suffix_max_suffix_len")
        if getattr(args, "suffix_min_confidence", None) is not None:
            fields.append("suffix_min_confidence")
        if getattr(args, "suffix_min_draft_len", None) is not None:
            fields.append("suffix_min_draft_len")
        return fields

    def _fill_suffix_defaults() -> None:
        if getattr(args, "suffix_max_draft", None) is None:
            args.suffix_max_draft = 8
        if getattr(args, "suffix_max_suffix_len", None) is None:
            args.suffix_max_suffix_len = 4
        if getattr(args, "suffix_min_confidence", None) is None:
            args.suffix_min_confidence = 0.3
        if getattr(args, "suffix_min_draft_len", None) is None:
            args.suffix_min_draft_len = 2

    def _legacy_speculative_config_payload() -> dict | None:
        methods: list[tuple[str, dict]] = []

        def add_method(method: str, payload: dict) -> None:
            methods.append((method, payload))

        def reject_orphan(knob: str, selector: str) -> None:
            print(
                f"error: legacy speculative decoding knob {knob} requires {selector}.",
                file=sys.stderr,
            )
            sys.exit(2)

        spec_decode = getattr(args, "spec_decode", "none")
        dflash_model = (getattr(args, "dflash_drafter_path", "") or "").strip()
        if getattr(args, "enable_ddtree", False):
            add_method("ddtree", {"method": "ddtree"})
        dflash_requested = (
            getattr(args, "enable_dflash", False) or spec_decode == "dflash"
        )
        if dflash_model and not dflash_requested:
            reject_orphan("dflash_drafter_path", "enable_dflash or spec_decode=dflash")
        if dflash_requested:
            payload = {"method": "dflash"}
            if dflash_model:
                payload["model"] = dflash_model
            add_method("dflash", payload)

        mtp_payload = {"method": "mtp"}
        mtp_requested = getattr(args, "enable_mtp", False) or spec_decode == "mtp"
        sidecar = (getattr(args, "mtp_sidecar", None) or "").strip()
        if sidecar:
            if not mtp_requested:
                reject_orphan("mtp_sidecar", "enable_mtp or spec_decode=mtp")
            mtp_payload["model"] = sidecar
        mtp_max_k = getattr(args, "mtp_max_k", None)
        if mtp_max_k is not None:
            if not mtp_requested:
                reject_orphan("mtp_max_k", "enable_mtp or spec_decode=mtp")
            mtp_payload["num_speculative_tokens"] = mtp_max_k
        mtp_num_draft_tokens = getattr(args, "mtp_num_draft_tokens", 1)
        if (
            mtp_max_k is not None
            and mtp_num_draft_tokens != 1
            and mtp_max_k != mtp_num_draft_tokens
        ):
            print(
                "error: legacy MTP aliases mtp_max_k and mtp_num_draft_tokens "
                "conflict; pass only one token-count value.",
                file=sys.stderr,
            )
            sys.exit(2)
        if mtp_num_draft_tokens != 1:
            if not mtp_requested:
                reject_orphan("mtp_num_draft_tokens", "enable_mtp or spec_decode=mtp")
            mtp_payload["num_speculative_tokens"] = mtp_num_draft_tokens
        if getattr(args, "mtp_disable_auto_k", False):
            if not mtp_requested:
                reject_orphan("mtp_disable_auto_k", "enable_mtp or spec_decode=mtp")
            mtp_payload["disable_auto_k"] = True
        if getattr(args, "mtp_optimistic", False):
            if not mtp_requested:
                reject_orphan("mtp_optimistic", "enable_mtp or spec_decode=mtp")
            # Unified spec-decode interface (PR #1050): even legacy
            # ``--enable-mtp`` normalizes to ``spec_decode="mtp"`` and
            # installs the vendored MTP runtime, which does not honour
            # ``mtp_optimistic``. Hard-reject the flag on every entry
            # point so behavior stays consistent (fail loud > silent
            # ignore).
            print(
                "error: legacy speculative decoding knob mtp_optimistic "
                "is not supported under the unified spec-decode "
                "interface; the vendored MTP installer does not "
                "implement optimistic mode. Remove --mtp-optimistic.",
                file=sys.stderr,
            )
            sys.exit(2)
        if mtp_requested:
            add_method("mtp", mtp_payload)

        suffix_payload = {"method": "suffix"}
        suffix_requested = getattr(args, "suffix_decoding", False)
        suffix_fields = (
            ("suffix_max_draft", "num_speculative_tokens"),
            ("suffix_max_suffix_len", "max_suffix_len"),
            ("suffix_min_confidence", "min_confidence"),
            ("suffix_min_draft_len", "min_draft_len"),
        )
        for attr, key in suffix_fields:
            value = getattr(args, attr, None)
            if value is not None:
                if not suffix_requested:
                    reject_orphan(attr, "suffix_decoding")
                suffix_payload[key] = value
        if suffix_requested:
            add_method("suffix", suffix_payload)

        if not methods:
            return None
        distinct = {method for method, _payload in methods}
        if len(distinct) > 1:
            joined = ", ".join(method for method, _payload in methods)
            print(
                "error: legacy speculative decoding aliases select multiple "
                f"methods ({joined}); use one --speculative-config payload.",
                file=sys.stderr,
            )
            sys.exit(2)
        return methods[0][1]

    legacy_payload = None
    legacy_enable_mtp_requested = raw_config is None and getattr(
        args, "enable_mtp", False
    )
    legacy_mtp_optimistic_requested = raw_config is None and getattr(
        args, "mtp_optimistic", False
    )
    if raw_config_was_explicit:
        legacy_fields = _legacy_speculative_fields()
        if legacy_fields:
            joined = ", ".join(legacy_fields)
            print(
                "error: --speculative-config is mutually exclusive with "
                f"legacy speculative decoding aliases: {joined}.",
                file=sys.stderr,
            )
            sys.exit(2)

    if raw_config is None:
        _reject_no_spec_decode_runtime_conflicts()
        legacy_payload = _legacy_speculative_config_payload()
        if legacy_payload is not None:
            raw_config = json.dumps(legacy_payload, separators=(",", ":"))
            args.speculative_config = raw_config

    if raw_config is None:
        _fill_runtime_defaults(overwrite=False)
        args._speculative_config = None
        _fill_suffix_defaults()
        return

    if raw_config is not None:
        try:
            config = parse_speculative_config(raw_config)
            if config is not None:
                require_migrated_speculative_config(config)
        except SpeculativeConfigError as exc:
            print(f"error: {exc}", file=sys.stderr)
            sys.exit(2)
        if getattr(args, "no_spec_decode", False):
            print(
                "error: --speculative-config is mutually exclusive with "
                "--no-spec-decode.",
                file=sys.stderr,
            )
            sys.exit(2)

    if config is None:
        _fill_runtime_defaults(overwrite=True)
        args._speculative_config = None
        _fill_suffix_defaults()
        return

    _fill_runtime_defaults(overwrite=True)
    args._speculative_config = config
    if config.method == "ddtree":
        args.enable_ddtree = True
    elif config.method == "dflash":
        args.enable_dflash = True
        if config.model:
            args.dflash_drafter_path = config.model
    elif config.method == "dspark":
        args.spec_decode = "dspark"
        args.dspark_num_speculative_tokens = config.num_speculative_tokens or 5
    elif config.method == "mtp":
        args.spec_decode = "mtp"
        if legacy_enable_mtp_requested:
            args.enable_mtp = True
            if config.num_speculative_tokens is not None:
                args.mtp_num_draft_tokens = config.num_speculative_tokens
            if legacy_mtp_optimistic_requested:
                args.mtp_optimistic = True
        # #1998: an alias may DECLARE its own MTP sidecar and draft depth
        # (``mtp_draft_model`` / ``mtp_speculative_tokens``). Those were
        # rendered by ``rapid-mlx models`` as ``✓ MTP  MTP@<repo>@<k>`` and
        # then read NOWHERE on the serve path, so the command that listing
        # implies — ``serve <alias> --speculative-config '{"method":"mtp"}'``
        # — reached the injector with ``sidecar=None`` and hard-failed at
        # boot, quoting an unrelated model in the remedy. Fill ONLY what the
        # request left unset; an explicit ``model`` /
        # ``num_speculative_tokens`` in the JSON always wins.
        alias_sidecar, alias_k = _alias_mtp_declaration(getattr(args, "model", None))
        args.mtp_sidecar = config.model or alias_sidecar
        if config.num_speculative_tokens is not None:
            args.mtp_max_k = config.num_speculative_tokens
        elif alias_k is not None:
            # A depth the alias declares for THIS checkpoint beats the generic
            # --force-spec-decode fallback below: it is the more specific fact.
            args.mtp_max_k = alias_k
        elif getattr(args, "force_spec_decode", False):
            # User explicitly opted into spec-decode via --force-spec-decode
            # but didn't pin a draft depth. K=1 chain-of-1 carries draft
            # overhead with no net speedup; default to K=3 (the EV auto-K
            # controller's intended default) so MTP actually accelerates.
            #
            # Only two draft-depth sources exist for the mtp method and
            # neither is set here: (a) ``num_speculative_tokens`` in the JSON
            # is handled by the branch above; (b) the legacy ``--mtp-max-k``
            # flag CANNOT co-occur with ``--speculative-config`` — the
            # mutual-exclusion guard (`_legacy_speculative_fields` lists
            # ``mtp_max_k``) exits with code 2 before we reach this branch.
            # So K=3 here never overwrites a user-pinned depth.
            args.mtp_max_k = 3
        if config.disable_auto_k is not None:
            args.mtp_disable_auto_k = config.disable_auto_k
    elif config.method == "suffix":
        args.suffix_decoding = True
        if config.num_speculative_tokens is not None:
            args.suffix_max_draft = config.num_speculative_tokens
        if config.max_suffix_len is not None:
            args.suffix_max_suffix_len = config.max_suffix_len
        if config.min_confidence is not None:
            args.suffix_min_confidence = config.min_confidence
        if config.min_draft_len is not None:
            args.suffix_min_draft_len = config.min_draft_len

    _fill_suffix_defaults()


def _resolve_dflash_drafter_repo(args, profile) -> str:
    """Return the effective DFlash drafter repo for normalized CLI args."""

    spec_config = getattr(args, "_speculative_config", None)
    if spec_config is not None and spec_config.method == "dflash" and spec_config.model:
        return spec_config.model
    return profile.dflash_draft_model


def _preflight_dflash_mutexes_or_exit(args) -> None:
    """Reject DFlash flag combinations before optional-runtime probes."""

    if not getattr(args, "enable_dflash", False):
        return

    import sys

    spec_config = getattr(args, "_speculative_config", None)
    if spec_config is not None and getattr(spec_config, "method", None) == "dflash":
        if getattr(args, "suffix_decoding", False):
            print(
                "\n  Error: DFlash cannot combine with other spec-decode methods. "
                "DFlash runs a dedicated single-user server that bypasses "
                "BatchedEngine; other spec-decode methods only apply to the "
                "BatchedEngine path.\n"
            )
            sys.exit(1)
        return

    if (
        getattr(args, "suffix_decoding", False)
        or getattr(args, "enable_mtp", False)
        or getattr(args, "spec_decode", "none") != "none"
    ):
        print(
            "\n  Error: DFlash cannot combine with other spec-decode methods. "
            "DFlash runs a dedicated single-user server that bypasses "
            "BatchedEngine; other spec-decode methods only apply to the "
            "BatchedEngine path.\n"
        )
        sys.exit(1)
    if getattr(args, "no_spec_decode", False):
        print(
            "error: DFlash and --no-spec-decode are mutually exclusive "
            "— DFlash is a speculative-decode mode.",
            file=sys.stderr,
        )
        sys.exit(2)


def _preflight_ddtree_or_exit(args):
    """Validate DDTree flag/alias/runtime gates and cache the profile."""
    import sys

    spec_config = getattr(args, "_speculative_config", None)
    if spec_config is None or spec_config.method != "ddtree":
        print(
            'error: DDTree requires --speculative-config \'{"method":"ddtree"}\'.',
            file=sys.stderr,
        )
        sys.exit(2)

    if getattr(args, "enable_dflash", False):
        print(
            "\n  Error: DDTree cannot combine with DFlash. Pick one "
            "block-diffusion speculative-decoding server.\n"
        )
        sys.exit(1)
    if (
        getattr(args, "suffix_decoding", False)
        or getattr(args, "spec_decode", "none") != "none"
    ):
        print(
            "\n  Error: DDTree cannot combine with other spec-decode methods. "
            "DDTree runs a dedicated single-user server that bypasses "
            "BatchedEngine; other spec-decode methods only apply to the "
            "BatchedEngine path.\n"
        )
        sys.exit(1)
    if getattr(args, "no_spec_decode", False):
        print(
            "error: DDTree and --no-spec-decode are mutually "
            "exclusive — DDTree is a speculative-decode mode.",
            file=sys.stderr,
        )
        sys.exit(2)

    from .model_aliases import resolve_profile
    from .speculative.ddtree import DDTreeUnavailable, check
    from .speculative.ddtree.eligibility import have_runtime, runtime_probe_error

    alias_name = getattr(args, "_original_alias", None) or args.model
    profile = resolve_profile(alias_name)
    if profile is None:
        print(
            f"\n  Error: DDTree requires a known alias, got "
            f"{alias_name!r}. DDTree eligibility is recorded per-alias "
            f"in aliases.json; ad-hoc HuggingFace paths can't be "
            f"validated. Try ``rapid-mlx info qwen3.5-9b-8bit``.\n"
        )
        sys.exit(1)
    try:
        check(profile, alias=alias_name)
    except DDTreeUnavailable as e:
        print(f"\n  Error: {e}\n")
        sys.exit(1)
    if not have_runtime():
        probe_error = runtime_probe_error()
        detail = f" Probe failure: {probe_error}." if probe_error else ""
        print(
            "\n  Error: DDTree requires the experimental dtree-mlx "
            "runtime. Install with: ``pip install 'dtree-mlx @ "
            f"git+https://github.com/DrHB/dtree-mlx.git'``.{detail}\n"
        )
        sys.exit(1)

    args._ddtree_alias_name = alias_name
    args._ddtree_profile = profile
    args._ddtree_drafter_repo = spec_config.model or profile.ddtree_draft_model
    args._ddtree_speculative_tokens = (
        spec_config.num_speculative_tokens or profile.ddtree_speculative_tokens
    )
    args._ddtree_tree_budget = spec_config.tree_budget or profile.ddtree_tree_budget
    return alias_name, profile


_DEFAULT_HYBRID_CACHE_ENTRIES = 8


def _resolve_hybrid_cache_entries(
    *,
    enable_prefix_cache: bool,
    explicit_value: int,
    user_set_explicit: bool,
    model_name: str,
) -> int:
    """Return the effective ``hybrid_cache_entries`` value.

    Auto-defaults to 8 when prefix cache is enabled for a hybrid model
    and the user did NOT explicitly pass ``--hybrid-cache-entries``.
    Without this, ``--enable-prefix-cache`` has no effect on hybrid
    models (#1122).
    """
    import logging as _logging

    if not enable_prefix_cache or explicit_value != 0 or user_set_explicit:
        return explicit_value

    needs_bounded_reuse = _needs_bounded_trim_free_reuse(model_name)
    if needs_bounded_reuse:
        _logging.getLogger(__name__).info(
            "Non-trimmable model cache detected with --enable-prefix-cache: "
            "auto-setting --hybrid-cache-entries=%d "
            "(pass --hybrid-cache-entries 0 to disable)",
            _DEFAULT_HYBRID_CACHE_ENTRIES,
        )
        return _DEFAULT_HYBRID_CACHE_ENTRIES
    return explicit_value


def _config_declares_sliding_window(config: dict | None) -> bool:
    """True if the checkpoint config declares sliding-window (local) attention.

    Gemma-2/3/4 and other local-attention families run their sliding layers on
    ``RotatingKVCache``, which reports ``is_trimmable() == False`` once the ring
    has rotated past its window (the front is overwritten, so a trim-then-
    continue would reconstruct wrong KV — see ``memory_cache`` ``_layer_forbids_
    trim``). Such a cache is non-trimmable but still REUSABLE at an exact token
    boundary, so these models need the bounded trim-free snapshot path — exactly
    like the recurrent-state families — or every agentic turn re-prefills the
    whole accumulated context (#2061).

    The signal is read straight off the checkpoint config so it is
    architecture-driven rather than a name list: it covers the whole sliding-
    window family, future additions, and bare local paths that resolve to no
    alias.

    The LANGUAGE backbone's attention config is authoritative — VLM checkpoints
    nest it under ``text_config``, so that is judged first and alone when it
    carries a signal (a top-level ``sliding_window`` scalar on such a checkpoint
    is often a vision/default value unrelated to the LM's attention). Within the
    chosen config, ``layer_types`` (Gemma-3/4's explicit per-layer attention
    kinds) is authoritative: a config listing only ``full_attention`` is NOT
    sliding even if it also carries a leftover ``sliding_window`` value (that
    field is inert without a sliding layer to apply it). A bare positive
    ``sliding_window`` is consulted only as the fallback for older configs that
    omit ``layer_types``. Only if the nested language config carries no signal
    at all does the top level get a look, for checkpoints that place attention
    fields at the root.
    """
    if not isinstance(config, dict):
        return False

    def _level_signal(cfg: object) -> bool | None:
        """True/False if ``cfg`` declares sliding-window state, None if it
        carries no attention signal at all (so the caller may look elsewhere)."""
        if not isinstance(cfg, dict):
            return None
        layer_types = [
            lt for lt in (cfg.get("layer_types") or []) if isinstance(lt, str)
        ]
        if layer_types:
            return any("sliding" in lt for lt in layer_types)
        # Qwen2 / Mistral carry a ``sliding_window`` scalar but gate it behind an
        # explicit ``use_sliding_window`` flag — a positive scalar with the flag
        # off is inert, so honour the disable before the legacy scalar. Treating
        # it as active would auto-allocate bounded snapshots and regress memory.
        if cfg.get("use_sliding_window") is False:
            return False
        sliding_window = cfg.get("sliding_window")
        if isinstance(sliding_window, int):
            return sliding_window > 0
        return None

    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        nested = _level_signal(text_config)
        if nested is not None:
            return nested
    return bool(_level_signal(config))


def _resolve_checkpoint_config(model_name: str, profile) -> dict | None:
    """Read the checkpoint ``config.json`` for ``model_name`` offline.

    Works for a bare local path or HF repo id directly, and for a registered
    alias by resolving it to its ``hf_path`` (the alias name itself is not a
    readable checkpoint dir). Never touches the network — mirrors the offline
    metadata probes ``is_mllm_model`` already relies on.
    """
    from .api.utils import read_model_metadata

    metadata = read_model_metadata(model_name)
    if metadata is not None and metadata.config:
        return metadata.config
    hf_path = getattr(profile, "hf_path", None) if profile is not None else None
    if isinstance(hf_path, str) and hf_path and hf_path != model_name:
        metadata = read_model_metadata(hf_path)
        if metadata is not None and metadata.config:
            return metadata.config
    return None


def _needs_bounded_trim_free_reuse(model_name: str) -> bool:
    """Whether this model's cache can reuse prefixes but not trim exact hits."""
    from .model_aliases import resolve_profile as _resolve_alias
    from .utils.deepseek_v4_0731 import is_deepseek_v4_0731

    profile = _resolve_alias(model_name)
    if profile is not None:
        if profile.is_hybrid:
            return True
        # Dense GatedDeltaNet models (Qwen3.5 / Qwen3.6 4B/9B/27B,
        # Ternary-Bonsai) are deliberately pinned ``is_hybrid=False`` to keep
        # them OFF the hybrid scheduler — the r6-A R6-C1 metal::malloc (499000)
        # wedge fires when their dense recurrent weights hit the hybrid
        # allocation path. But architecturally they DO carry non-trimmable
        # ArraysCache / GatedDeltaNet recurrent layers, so their state can only
        # be reused through the bounded snapshot path (``hybrid_cache_entries``),
        # never the ordinary *trimmable* prefix cache. The
        # ``is_hybrid_explicit=True`` + ``is_hybrid=False`` pinning uniquely
        # identifies that "recurrent but routed non-hybrid" set (the explicit
        # flag is only ever set to suppress the boot-time ArraysCache promotion,
        # i.e. it implies the model has ArraysCache layers). Without this branch
        # #1122's auto-default never fires for them, so ``--enable-prefix-cache``
        # is a silent no-op: every agent turn re-prefills the full accumulated
        # context (measured turn-2 TTFT ~22s with reuse off vs ~0.85s on for
        # qwen3.5-9b-4bit). This does NOT touch routing — is_hybrid stays False,
        # so no throttle, no snapshot path, no wedge.
        if profile.is_hybrid_explicit and not profile.is_hybrid:
            return True

    # Sliding-window attention families (Gemma-2/3/4, …) carry RotatingKVCache
    # local-attention layers that go non-trimmable once the ring rotates past
    # their window, so — like the recurrent families above — they can only
    # reuse a prefix through the bounded snapshot path, never the ordinary
    # trimmable prefix cache. Without this, --enable-prefix-cache is a silent
    # no-op for them and every agentic turn re-prefills the whole accumulated
    # context (#2061). Detected from the checkpoint config (architecture-driven,
    # so it also covers a bare local path that resolves to no alias, which is
    # exactly how #2061 was served). This does NOT touch routing — is_hybrid is
    # untouched, so no throttle and no hybrid allocation path / wedge.
    if _config_declares_sliding_window(_resolve_checkpoint_config(model_name, profile)):
        return True

    return is_deepseek_v4_0731(model_name)


def _serve_will_run_on_mllm_lane(args) -> bool:
    """Whether ``serve`` will actually run this model on the MLLM/VLM
    continuous-batching lane — the ONLY lane that needs the optional
    ``mlx-vlm`` runtime (shipped behind the ``[vision]`` extra).

    Delegates to #1178's :func:`resolve_serving_lane` so the boot-time
    ``[vision]``-required guard agrees with the load-time routing decision.
    A multimodal alias whose LANGUAGE backbone is hybrid/linear-attention
    (Qwen3.6 GatedDeltaNet) auto-downgrades to the text-only mlx-lm lane and
    never touches mlx-vlm, so it must NOT be pushed into a ~1 GB ``[vision]``
    install. A genuine VLM (non-hybrid backbone, e.g. qwen3-vl) stays on the
    MLLM lane and still needs it. ``--mllm`` / ``--no-mllm`` are honoured via
    ``resolve_serving_lane``'s explicit-flag short-circuits.

    The probe reads the cached checkpoint config offline (no network, no
    weight load). On a first-time uncached start the config isn't
    materialized yet, so the hybrid probe answers "not hybrid" and the model
    keeps the SAFE ``[vision]``-required default — the guard's error message
    then points at ``--no-mllm`` for a text-capable backbone.
    """
    from .api.utils import resolve_serving_lane

    is_mllm_lane, _auto_text_fallback = resolve_serving_lane(
        args.model,
        force_mllm=getattr(args, "mllm", False),
        force_text=getattr(args, "no_mllm", False),
    )
    return is_mllm_lane


def kv_cache_flag_conflict(args) -> str | None:
    """Return the operator-facing reason the KV-cache flags conflict, else None.

    Pure predicate over the parsed args — no I/O, no model resolution, no
    process exit. ``serve_command`` prints the returned message and exits 1.

    Extracted from ``serve_command`` because the rejections were only
    reachable by spawning a real ``serve``, and the test that covered them
    did exactly that: it ran ``serve <alias> --reasoning
    --kv-cache-quantization --kv-cache-quantization-bits 4`` in a subprocess
    with a 30 s timeout and asserted a non-zero exit. Alias resolution and
    the weight download run *before* this point, so on a machine without the
    fixture model cached the child spent the whole budget downloading and the
    test timed out — it passed or failed on local Hugging Face cache state
    rather than on whether the rejection still fired. A guard whose colour is
    decided by something other than the behaviour it guards is not a guard.

    The three checks and their messages are unchanged; only their location is.
    """
    # Mutual exclusion: turboquant (any mode) vs standard quantization.
    # The argparse layer normalizes the flag to either ``None`` (off),
    # ``"v4"``, or ``"k8v4"``. Anything truthy means TurboQuant is on.
    if args.kv_cache_turboquant and args.kv_cache_quantization:
        return (
            "--kv-cache-turboquant and --kv-cache-quantization are "
            "mutually exclusive. Choose one."
        )

    if not args.kv_cache_quantization:
        return None

    # codex r1 BLOCKING #1: ``--reasoning`` must override the legacy
    # ``--kv-cache-quantization`` flag too — otherwise ``rapid-mlx serve
    # --reasoning --kv-cache-quantization --kv-cache-quantization-bits 4``
    # silently resolves to int4 and the operator who deliberately asked for
    # the reasoning profile gets the AIME-class quality cliff. Reject the
    # conflicting combo explicitly: silently flipping the legacy bits to 8
    # would hide the misconfiguration. bits=8 is equivalent to
    # ``--reasoning``'s int8 pin and is harmless; only bits=4 conflicts.
    if args.reasoning and args.kv_cache_quantization_bits == 4:
        return (
            "--reasoning is incompatible with --kv-cache-quantization "
            "--kv-cache-quantization-bits 4. The reasoning profile pins KV "
            "cache to int8 because sub-4-bit drops -20pt on AIME-class math. "
            "Either drop --reasoning or drop --kv-cache-quantization-bits 4 "
            "(or both; use --kv-cache-dtype int8 instead)."
        )

    # codex r2 BLOCKING #1: argparse pins ``--kv-cache-quantization-bits`` to
    # ``choices={4,8}``, but programmatic callers (tests, library users that
    # bypass argparse) can land an out-of-range bits value here. The old
    # ``"int4" if bits == 4 else "int8"`` silently labeled every non-4 value
    # as ``int8`` even when KV would actually be quantized at the requested
    # bit width. Fail fast instead so the gauge / banner / SchedulerConfig
    # never lie about the active dtype.
    if args.kv_cache_quantization_bits not in (4, 8):
        return (
            f"--kv-cache-quantization-bits must be 4 or 8 "
            f"(got {args.kv_cache_quantization_bits}). Use --kv-cache-dtype "
            f"for the canonical knob."
        )

    return None


def serve_command(args):
    """Start the OpenAI-compatible server."""
    import logging
    import os
    import sys

    if bounds_error := _vision_pixel_bounds_error(
        getattr(args, "vision_min_pixels", 0),
        getattr(args, "vision_max_pixels", 0),
    ):
        print(f"error: {bounds_error}", file=sys.stderr)
        raise SystemExit(2)

    if os.environ.get("RAPID_PYSAMPLE"):
        from ._pysample import install as _pysample_install

        _pysample_install()

    # Parent-PID watchdog (rapid-desktop issue #449): if the supervisor
    # passed its own PID via ``--watchdog-ppid`` or
    # ``$RAPID_MLX_WATCHDOG_PPID``, spawn a daemon thread that polls
    # ``os.getppid()`` every 2 s. When the parent dies (re-parented to
    # launchd / init), the watchdog sends SIGTERM to ourselves so the
    # FastAPI lifespan can flush + release the model — falling back to
    # SIGKILL after a 5 s grace. Installed at the top of serve_command
    # so it covers BOTH the text-LM path and the audio-mode fork below,
    # AND so it arms before the (potentially multi-minute) model
    # download — an operator who kills the desktop mid-download still
    # gets a clean reap on the sidecar. No-op when no supervisor PID
    # was passed (default).
    from ._parent_watchdog import install_parent_watchdog, resolve_expected_ppid

    install_parent_watchdog(resolve_expected_ppid(getattr(args, "watchdog_ppid", None)))

    _arg_max_tokens = getattr(args, "max_tokens", None)
    _max_tokens_is_explicit = _arg_max_tokens is not None
    effective_max_tokens = _arg_max_tokens if _arg_max_tokens is not None else 32768

    # Video aliases use a dedicated MLX-native runtime. Check the optional
    # package and ffmpeg before version prompts or a 22+ GB model download.
    from .model_aliases import resolve_profile as _resolve_serve_profile

    _serve_profile = _resolve_serve_profile(
        getattr(args, "_original_alias", None) or getattr(args, "model", "")
    )
    _is_wan_video = False
    if _serve_profile is not None and _serve_profile.modality == "video-gen":
        from .runtime.video_lane import require_video_runtime_or_exit
        from .video.wan import is_wan_model

        model_folded = args.model.casefold()
        _is_wan_video = is_wan_model(args.model)
        if "cogvideox" in model_folded or _is_wan_video:
            require_video_runtime_or_exit(args.model)
        else:
            require_video_runtime_or_exit()

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
    #
    # 0.10.16 dogfood follow-up (④): consult the SAME resolved-lane signal
    # the engine uses (#1178 ``resolve_serving_lane`` + ``_auto_text_
    # fallback``) instead of the raw ``is_mllm_model`` classification. A
    # multimodal alias whose LANGUAGE backbone is hybrid/linear-attention
    # (Qwen3.6 GatedDeltaNet) auto-downgrades to the text-only mlx-lm lane
    # and NEVER touches mlx-vlm — forcing a base-wheel user into a ~1 GB
    # ``[vision]`` install for a model that then serves text-only was the
    # dogfood pain point. ``_serve_will_run_on_mllm_lane`` is True only when
    # the model will actually run on the MLLM lane, so:
    #   * genuine VLM (qwen3-vl, non-hybrid backbone) → still requires it,
    #   * hybrid-backbone VLM (qwen3.6) → boots text-only from the base wheel,
    #   * ``--mllm`` force-on / ``--no-mllm`` escape hatch → honoured by
    #     ``resolve_serving_lane``, matching the engine-side semantics.
    # An uncached checkpoint (config not yet materialized) probes "not
    # hybrid" and keeps the SAFE ``[vision]``-required default; the guard's
    # message points at ``--no-mllm`` for a text-capable backbone.
    if _serve_will_run_on_mllm_lane(args):
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

    _normalize_speculative_config_or_exit(args)

    # DDTree has an external experimental runtime and its validated target
    # can be multi-GB. Fail the cheap config/alias/runtime gates before the
    # version prompt and before any model prefetch so a missing dtree-mlx
    # install doesn't start a large download first.
    if getattr(args, "enable_ddtree", False):
        _preflight_ddtree_or_exit(args)

    if getattr(args, "enable_dflash", False):
        _preflight_dflash_mutexes_or_exit(args)

    # DFlash depends on the optional ``mlx-vlm`` bridge that ships in
    # the ``[dflash]`` extra. Pre-0.9.3 the missing-runtime error only
    # surfaced ~50 lines into serve_command, AFTER:
    #   - alias profile resolved (logged twice via pflash)
    #   - tool/reasoning parsers auto-configured
    #   - CORS allow-origin warning printed
    # so the operator saw five INFO lines and a banner before the
    # actionable optional-extra install instructions,
    # matching Diego's earlier ``[embeddings]`` regression shape exactly.
    # Hoist the cheap ``have_runtime()`` probe to the same boot-guard tier
    # as the other extras so the error lands FIRST. ``importlib.util.
    # find_spec("mlx_vlm")`` doesn't trigger a load — safe to run on the
    # hot CLI path.
    _wants_dflash = getattr(args, "enable_dflash", False)
    if _wants_dflash:
        from .speculative.dflash.eligibility import have_runtime

        if not have_runtime():
            print(
                "\n  Error: DFlash speculative decoding "
                '(--speculative-config \'{"method":"dflash"}\') requires '
                "mlx-vlm 0.5.0+ for the DFlash drafter hooks.\n"
                "\n  Install in a Python environment with:\n"
                "    pip install 'rapid-mlx[dflash]'\n"
                "\n  Homebrew installs the text-only package. Homebrew users "
                "can switch to the isolated full install with:\n"
                "    brew uninstall rapid-mlx\n"
                "    uv tool install 'rapid-mlx[dflash]'\n"
            )
            sys.exit(1)

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

    # Finding ⑥ (0.10.16 dogfood): a "weightless stub" cache — config.json
    # present but ``model*.safetensors`` absent (a warm cache commonly holds
    # ~20 Gemma-4 repos in exactly this state, from a metadata-only config
    # probe or an interrupted pull) — LOOKS cached, so ``serve`` eats a
    # surprise multi-GB download with no upfront signal. Surface a one-line
    # notice BEFORE the prefetch. Purely informational and unconditional
    # (fires even in non-TTY / RAPID_MLX_AUTO_PULL=1 runs where the B2
    # confirmation gate self-skips); it does NOT gate or change the download
    # (that stays with ``_ensure_model_downloaded``). Best-effort — a
    # diagnostic must never break serve.
    try:
        from vllm_mlx._download_gate import weightless_stub_notice
        from vllm_mlx.model_aliases import resolve_model as _resolve_model

        # Canonicalize alias → ``org/repo`` BEFORE probing the cache. A
        # shorthand alias (``gemma-4-12b``) has its config-only stub on disk
        # under the RESOLVED HF id, so probing the raw alias string would
        # miss the cache dir and the notice would silently no-op for the
        # common naive-user invocation. ``resolve_model`` is idempotent for
        # already-resolved ids / local paths, and mirrors the resolution the
        # download path (``_ensure_model_downloaded`` on ``args.model``)
        # relies on.
        _probe_model = _resolve_model(args.model)
        _stub_notice = weightless_stub_notice(_probe_model)
        if _stub_notice:
            # stderr + flush: in the exact non-TTY / RAPID_MLX_AUTO_PULL=1
            # case this notice targets, block-buffered stdout may never
            # flush before the multi-GB download + server lifetime, leaving
            # the warning invisible. stderr is line-buffered/unbuffered and
            # is the correct stream for an operational warning anyway.
            print(_stub_notice, file=sys.stderr, flush=True)
    except Exception:
        pass

    # Pre-fetch the model via the R2 mirror (with HF fallback) BEFORE the
    # heavy server boot. Without this, ``serve`` falls into
    # ``mlx_lm.load`` → ``huggingface_hub.snapshot_download`` directly and
    # skips the mirror entirely (#651). ``_ensure_model_downloaded`` is a
    # no-op on local paths and on fully-cached repos, so this is free on
    # the warm path.
    # WanVideoEngine resolves registered repositories at an audited pinned
    # revision (or uses RAPID_MLX_WAN_MODEL_DIR). The generic prefetch has no
    # revision parameter and could otherwise download repository HEAD first,
    # duplicating tens of gigabytes before the pinned snapshot is loaded.
    if not _is_wan_video:
        if getattr(args, "force_disk_check", False):
            _ensure_model_downloaded(args.model, force_disk_check=True)
        else:
            # Keep the historical one-argument call on the default path; a
            # number of embedders/tests replace this hook with a one-argument
            # prefetch function.
            _ensure_model_downloaded(args.model)

    # The prefetch above swallows transport errors on purpose ("server will
    # retry"), which is right for a text model — the loader retries and raises
    # something legible. An mflux checkpoint has no such backstop: it loads
    # whatever shards arrived and renders noise, so a pull that ended early
    # would boot a server whose only possible output is garbage. Refuse here
    # instead, while the operator is still watching the command they typed.
    if _serve_profile is not None and _serve_profile.modality == "image-gen":
        from ._download_gate import mflux_missing_weights

        _missing = mflux_missing_weights(args.model)
        if _missing:
            _shown = getattr(args, "_original_alias", None) or args.model
            print(
                f"\n  Error: {_shown} is only partially downloaded — "
                f"{len(_missing)} required file(s) missing, starting with "
                f"{_missing[0]}.\n"
                f"  Finish the download with `rapid-mlx pull {_shown}`, "
                "then serve again.\n",
                file=sys.stderr,
            )
            sys.exit(1)

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
    if getattr(args, "resident_memory_limit_gb", 0.0) < 0:
        print("Error: --resident-memory-limit-gb must be >= 0")
        sys.exit(1)
    if getattr(args, "resident_model_idle_ttl", 0.0) < 0:
        print("Error: --resident-model-idle-ttl must be >= 0")
        sys.exit(1)
    idle_cache_clear_seconds = getattr(args, "idle_cache_clear_seconds", None)
    if idle_cache_clear_seconds is not None:
        import math

        if not math.isfinite(idle_cache_clear_seconds) or idle_cache_clear_seconds < 0:
            print("Error: --idle-cache-clear-seconds must be finite and >= 0")
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
    from .api.utils import resolve_serving_lane
    from .pflash import (
        resolve_pflash_config,
        validate_model_support,
    )

    # Resolve the FINAL serving lane ONCE (the model is already downloaded by
    # ``_ensure_model_downloaded`` above, so the offline probes have real
    # evidence). PFlash defaulting and ``validate_model_support`` must both see
    # the effective lane, NOT the raw multimodal classification: a hybrid VLM
    # that auto-downgrades to the text-only lane is PFlash-capable there,
    # exactly as an explicit ``--text-only`` run would be (#352 dogfood P1-②).
    if not args.enable_dflash:
        _serve_is_mllm, _ = resolve_serving_lane(
            args.model,
            force_mllm=getattr(args, "mllm", False),
            force_text=getattr(args, "no_mllm", False),
        )
        # Resolve BOTH per-alias PFlash defaults (mode + keep_ratio, e.g.
        # bonsai-27b-2bit → always @ 0.50) and build the config in one shared
        # helper; an explicit --pflash / --pflash-keep-ratio still wins inside.
        try:
            pflash_config = resolve_pflash_config(
                args, model_name=args.model, is_multimodal=_serve_is_mllm
            )
            validate_model_support(
                pflash_config,
                model_name=args.model,
                is_mllm=_serve_is_mllm,
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

    # Task #292: forward the ``--enable-audio`` opt-in to the server
    # module BEFORE ``load_model`` runs — the post-load hook in
    # ``load_model`` calls ``register_audio_routes_if_enabled``, which
    # reads ``server._enable_audio_lane`` to decide whether to mount
    # the audio router on a text-only server. Setting it after
    # ``load_model`` would leave the router unmounted on the very boot
    # that asked for it.
    #
    # Codex r2 NIT #2: assign from the parsed value directly so a second
    # in-process ``serve_command`` call (test harness, embedded usage)
    # without ``--enable-audio`` clears any stale ``True`` from a prior
    # run — the singleton ``server`` module persists across calls in
    # the same process.
    server._enable_audio_lane = bool(getattr(args, "enable_audio", False))

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

    # WH-1: OPT-IN Host-header allowlist (DNS-rebinding hardening).
    server.configure_trusted_hosts(getattr(args, "trusted_hosts", None))

    # Request logging middleware — installed AFTER CORS so it is the
    # outermost layer (Starlette prepends, so last install runs first).
    from vllm_mlx.middleware.request_logging import install_request_logging_middleware

    install_request_logging_middleware(server.app)

    if args.rate_limit > 0:
        server._rate_limiter = configure_rate_limiter(args.rate_limit, enabled=True)

    # Configure GC control
    gc_control = args.gc_control and not args.no_gc_control
    server._gc_control = gc_control

    # Configure --no-thinking: suppress chain-of-thought in chat template
    server._no_thinking = args.no_thinking

    # Configure system prompt pinning
    server._pin_system_prompt = args.pin_system_prompt
    server._relocate_mid_conversation_system = getattr(
        args, "relocate_mid_conversation_system", False
    )

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

    # DFlash / DDTree mutual-exclusion gates fire BEFORE the startup banner so
    # the user sees a clean error instead of an optimistic "Features:
    # dflash" / "ddtree" line immediately followed by an exit. The deeper
    # SchedulerConfig mutex (suffix vs. mtp) stays below since it doesn't
    # involve the dedicated single-user servers.
    if args.enable_dflash and args.suffix_decoding:
        print(
            "\n  Error: DFlash cannot combine with other spec-decode methods. "
            "DFlash runs a dedicated single-user server that bypasses "
            "BatchedEngine; other spec-decode methods only apply to the "
            "BatchedEngine path.\n"
        )
        sys.exit(1)

    # DFlash eligibility gate fires here, BEFORE the startup banner —
    # so the user sees a clean error rather than an optimistic "DFlash
    # enabled" feature line followed by an exit. Cheap (just reads
    # aliases.json + checks the module spec); no model load yet.
    if args.enable_dflash:
        from .model_aliases import resolve_profile
        from .speculative.dflash import DFlashUnavailable, check

        # ``have_runtime()`` validated at the top-of-function boot-guard
        # tier — see the 0.9.2 dogfood comment near the audio probe.
        _alias_name = getattr(args, "_original_alias", None) or args.model
        _profile = resolve_profile(_alias_name)
        if _profile is None:
            print(
                f"\n  Error: DFlash requires a known alias, got "
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
        # ``have_runtime()`` is already validated by the boot-guard tier
        # at the top of ``serve_command`` — see the 0.9.2 dogfood comment
        # there. We keep the import + the deeper DFlashUnavailable / alias
        # check here because they need the resolved profile, but the
        # extras-not-installed branch is unreachable by the time control
        # reaches this point.

        # Warn about flags that BatchedEngine honours but the DFlash
        # server doesn't — better to surface this once at startup than
        # to let users wonder why their tuning has no effect. Inspected
        # against the actual argparse Namespace so we only mention flags
        # the user explicitly set away from their default.
        _GPU_MEM_DEFAULT = 0.90  # keep in sync with the serve_parser default
        _dflash_ignored: list[str] = []
        # Prefix caching is enabled by default in the shared serve parser,
        # so it is not evidence that the user explicitly requested it.
        # DFlash already documents its no-prefix-cache limitation; avoid a
        # warning on every normal DFlash startup.
        if getattr(args, "kv_cache_quantization", None):
            _dflash_ignored.append("--kv-cache-quantization")
        if getattr(args, "kv_cache_turboquant", None):
            _dflash_ignored.append("--kv-cache-turboquant")
        if getattr(args, "pflash", None) not in (None, "auto"):
            _dflash_ignored.append("--pflash")
        # gpu-memory-utilization defaults to 0.90 (not None) in the serve
        # parser, so an ``is not None`` check would fire on every invocation.
        # Compare to the real default — only warn when the user explicitly
        # tuned it. Tolerate a tiny float-equality slack for safety.
        _gpu_mem = getattr(args, "gpu_memory_utilization", _GPU_MEM_DEFAULT)
        if _gpu_mem is not None and abs(_gpu_mem - _GPU_MEM_DEFAULT) > 1e-6:
            _dflash_ignored.append("--gpu-memory-utilization")
        if getattr(args, "enable_tool_logits_bias", False):
            _dflash_ignored.append("--enable-tool-logits-bias")
        if getattr(args, "embedding_model", None):
            _dflash_ignored.append("--embedding-model")
        if getattr(args, "mcp_config", None):
            _dflash_ignored.append("--mcp-config")
        if _dflash_ignored:
            print(
                "\n  ⚠ The following flags are ignored under DFlash"
                "\n    (DFlash uses a dedicated single-user server that bypasses"
                "\n    BatchedEngine):"
                f"\n      {', '.join(_dflash_ignored)}"
                "\n    Drop them from your serve command, or run without DFlash"
                "\n    if you need them.\n"
            )

    # DDTree eligibility gate mirrors DFlash: cheap alias/runtime checks
    # before the startup banner and before any model load.
    if args.enable_ddtree:
        _GPU_MEM_DEFAULT = 0.90
        _ddtree_ignored: list[str] = []
        if getattr(args, "enable_prefix_cache", False):
            _ddtree_ignored.append("--enable-prefix-cache")
        if getattr(args, "kv_cache_quantization", None):
            _ddtree_ignored.append("--kv-cache-quantization")
        _gpu_mem = getattr(args, "gpu_memory_utilization", _GPU_MEM_DEFAULT)
        if _gpu_mem is not None and abs(_gpu_mem - _GPU_MEM_DEFAULT) > 1e-6:
            _ddtree_ignored.append("--gpu-memory-utilization")
        if getattr(args, "enable_auto_tool_choice", False):
            _ddtree_ignored.append("--enable-auto-tool-choice")
        if getattr(args, "tool_call_parser", None):
            _ddtree_ignored.append("--tool-call-parser")
        if getattr(args, "reasoning_parser", None):
            _ddtree_ignored.append("--reasoning-parser")
        if getattr(args, "embedding_model", None):
            _ddtree_ignored.append("--embedding-model")
        if getattr(args, "mcp_config", None):
            _ddtree_ignored.append("--mcp-config")
        if _ddtree_ignored:
            print(
                "\n  ⚠ The following flags are ignored under DDTree mode"
                "\n    (DDTree uses a dedicated experimental single-user "
                "server that bypasses BatchedEngine):"
                f"\n      {', '.join(_ddtree_ignored)}"
                "\n    Drop them from your serve command, or run without"
                "\n    DDTree if you need them.\n"
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
    if gc_control and not args.enable_dflash:
        features.append("gc-control")
    if args.pin_system_prompt and not args.enable_dflash:
        features.append("pin-system-prompt")
    # Show CORS in the startup banner when CLI flag or env-var-driven
    # config produced an origin list (``configure_cors_from_env`` is what
    # actually resolved it — see the call site earlier in this function).
    if cors_origins:
        features.append(f"cors: {', '.join(cors_origins)}")
    if args.enable_dflash:
        features.append("dflash: single-user")
    if args.enable_ddtree:
        features.append("ddtree: experimental single-user")
    if features:
        print(f"  Features: {', '.join(features)}")
    print(f"  Model: {args.model}")
    # Store MCP config path for FastAPI startup
    if args.mcp_config and not args.enable_dflash:
        print(f"MCP config: {args.mcp_config}")
        os.environ["RAPID_MLX_MCP_CONFIG"] = args.mcp_config

    # DFlash owns a dedicated single-user runtime. Fork before constructing
    # BatchedEngine-only cache/TurboQuant/PFlash state so startup output and
    # initialization describe capabilities that actually apply.
    if args.enable_dflash:
        if getattr(args, "no_spec_decode", False):
            print(
                "error: DFlash and --no-spec-decode are mutually "
                "exclusive — DFlash is a speculative-decode mode.",
                file=sys.stderr,
            )
            sys.exit(2)

        from .model_aliases import resolve_profile
        from .speculative.dflash.server import run_dflash_server

        _alias_name = getattr(args, "_original_alias", None) or args.model
        _profile = resolve_profile(_alias_name)
        assert _profile is not None and _profile.supports_dflash, (
            f"DFlash profile invariant violated for {_alias_name!r}"
        )

        _check_disk_space(args.model, force=getattr(args, "force_disk_check", False))
        _check_memory_capacity(args.model)
        server._sync_config()
        run_dflash_server(
            main_model_repo=_profile.hf_path,
            drafter_repo=_resolve_dflash_drafter_repo(args, _profile),
            host=args.host,
            port=args.port,
            served_model_name=args.served_model_name or _alias_name,
            default_max_tokens=effective_max_tokens,
            cors_origins=cors_origins,
            uvicorn_log_level=uvicorn_log_level,
            no_thinking=args.no_thinking,
            api_key=server._api_key,
            rate_limit=args.rate_limit,
            max_request_bytes=server._max_request_bytes,
            body_receive_timeout_seconds=server._body_receive_timeout_seconds,
            default_timeout=server._default_timeout,
            max_concurrent_requests=args.max_concurrent_requests,
            cors_policy=server.get_resolved_cors_policy(),
            tool_call_parser=(
                args.tool_call_parser if args.enable_auto_tool_choice else None
            ),
            reasoning_parser_name=args.reasoning_parser,
        )
        return

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

    # Resolve per-alias TurboQuant default before the mutual-exclusion
    # check below — operator-explicit values still win. The
    # ``turboquant_scheduler_kwargs`` helper is the shared invariant
    # (#969) — ``python -m vllm_mlx.server`` calls the same helper so
    # the two entrypoints can't drift.
    from .turboquant import (
        resolve_turboquant_mode_default,
    )
    from .turboquant import (
        turboquant_scheduler_kwargs as _turboquant_scheduler_kwargs,
    )

    args.kv_cache_turboquant = resolve_turboquant_mode_default(
        args, model_name=args.model
    )

    # Reject conflicting KV-cache flag combinations before anything else in
    # this block reads them. Extracted so the rejection can be tested without
    # spawning ``serve`` — see ``kv_cache_flag_conflict``.
    _kv_conflict = kv_cache_flag_conflict(args)
    if _kv_conflict is not None:
        print(f"\n  Error: {_kv_conflict}\n")
        sys.exit(1)

    # R15 #300: resolve --kv-cache-dtype + --reasoning + safelist BEFORE
    # the legacy --kv-cache-quantization flag wins. When --kv-cache-
    # turboquant is on, leave the kv-cache-dtype path alone — TurboQuant
    # owns the V cache and would conflict with QuantizedKVCache. When
    # the legacy --kv-cache-quantization flag is passed, honor it
    # verbatim for backwards compatibility; the new dtype flag only
    # takes effect on operators who haven't pinned the legacy bool.
    kv_cache_decision = None
    if not args.kv_cache_turboquant and not args.kv_cache_quantization:
        from .kv_cache_dtype import (
            dtype_to_quantization_bits,
            log_kv_cache_decision,
            resolve_kv_cache_dtype,
        )

        hf_cfg, alias_meta = _gather_kv_cache_dtype_inputs(args.model)
        kv_cache_decision = resolve_kv_cache_dtype(
            args.kv_cache_dtype,
            reasoning=args.reasoning,
            model_name=args.model,
            hf_path=(alias_meta or {}).get("hf_path"),
            hf_config=hf_cfg,
            alias_metadata=alias_meta,
        )
        log_kv_cache_decision(kv_cache_decision, model_name=args.model)
        quant, bits = dtype_to_quantization_bits(kv_cache_decision.dtype)
        # Mutate args so the existing SchedulerConfig wiring picks up
        # the resolved values without a second code path.
        args.kv_cache_quantization = quant
        args.kv_cache_quantization_bits = bits
        # Stash on the shared ServerConfig so /metrics surfaces the
        # effective dtype during the pre-engine load window — operator
        # uptime dashboards scrape within ms of process start.
        try:
            from vllm_mlx.config import get_config as _get_config

            _get_config().kv_cache_dtype = kv_cache_decision.dtype
        except Exception:
            # ServerConfig is best-effort observability; never block
            # serve start on a metrics-only side effect.
            pass
    elif args.kv_cache_quantization:
        # Legacy flag took precedence — synthesize a decision so
        # observability still has a single source of truth.
        from .kv_cache_dtype import (
            REASONING_KV_CACHE_DTYPE,
            KVCacheDtypeDecision,
        )

        # The two rejections that used to live here (``--reasoning`` +
        # bits=4, and an out-of-range bits value) moved into
        # ``kv_cache_flag_conflict`` and already fired above, so anything
        # reaching this point has a legal bits value.
        legacy_dtype = "int4" if args.kv_cache_quantization_bits == 4 else "int8"
        # When --reasoning is set alongside the (compatible) bits=8
        # legacy flag, the operator-facing reason should still
        # advertise the reasoning profile so the startup banner is
        # consistent across the two CLI shapes.
        if args.reasoning:
            assert legacy_dtype == REASONING_KV_CACHE_DTYPE  # by the guard above
            reason = (
                f"legacy --kv-cache-quantization flag + --reasoning — "
                f"resolved to {REASONING_KV_CACHE_DTYPE} (reasoning profile "
                f"pin matches legacy bits=8)"
            )
        else:
            reason = (
                f"legacy --kv-cache-quantization flag (bits="
                f"{args.kv_cache_quantization_bits}) — equivalent to "
                f"--kv-cache-dtype {legacy_dtype}"
            )
        kv_cache_decision = KVCacheDtypeDecision(
            dtype=legacy_dtype,
            reason=reason,
            downgraded=False,
            requested=legacy_dtype,
        )
        try:
            from vllm_mlx.config import get_config as _get_config

            _get_config().kv_cache_dtype = legacy_dtype
        except Exception:
            pass

    # Build scheduler config
    enable_prefix_cache = args.enable_prefix_cache and not args.disable_prefix_cache

    # #1122: when prefix cache is enabled for a hybrid model and the user
    # did NOT explicitly pass --hybrid-cache-entries, auto-default to 8 so
    # the cache actually stores entries instead of silently dropping them.
    _hybrid_cache_entries = _resolve_hybrid_cache_entries(
        enable_prefix_cache=enable_prefix_cache,
        explicit_value=getattr(args, "hybrid_cache_entries", 0),
        user_set_explicit="--hybrid-cache-entries" in sys.argv
        or any(a.startswith("--hybrid-cache-entries=") for a in sys.argv),
        model_name=getattr(args, "_original_alias", None) or args.model,
    )

    # 0.9.13 PR-A codex round-E blocker #2: resolve model_type on the
    # CLI's asyncio thread and thread it down through SchedulerConfig
    # so the engine's model-load-executor dispatch step does not need
    # to re-read ``config.json`` (offline HF cache races vs. the CLI's
    # own read were being collapsed into a silent MTP no-op). Only
    # runs when the operator explicitly asked for MTP through
    # ``--speculative-config``; for the "none" path the field stays
    # None and the engine takes the pre-0.9.13 fallback branch that
    # best-effort re-reads the config on the executor.
    _cli_mtp_model_type: str | None = None
    if getattr(args, "spec_decode", "none") == "mtp":
        try:
            _hf_cfg_for_mtype, _ = _gather_kv_cache_dtype_inputs(args.model)
            if isinstance(_hf_cfg_for_mtype, dict):
                _mt = _hf_cfg_for_mtype.get("model_type")
                if isinstance(_mt, str):
                    _cli_mtp_model_type = _mt
        except Exception:  # pragma: no cover — best-effort
            _cli_mtp_model_type = None

    scheduler_config = SchedulerConfig(
        max_num_seqs=args.max_num_seqs,
        max_concurrent_requests=args.max_concurrent_requests,
        prefill_batch_size=args.prefill_batch_size,
        completion_batch_size=args.completion_batch_size,
        enable_prefix_cache=enable_prefix_cache,
        prefix_cache_size=args.prefix_cache_size,
        # R15-P1 (task #303): radix-tree prefix-cache index.
        prefix_cache_index=getattr(args, "prefix_cache_index", "radix"),
        # Memory-aware cache options
        use_memory_aware_cache=not args.no_memory_aware_cache,
        cache_memory_mb=args.cache_memory_mb,
        cache_memory_percent=args.cache_memory_percent,
        idle_cache_clear_seconds=getattr(args, "idle_cache_clear_seconds", None),
        # #1103/#1122: bounded trim-free hybrid (recurrent-state) prefix reuse.
        # Auto-defaulted to 8 for hybrid models when prefix cache is enabled.
        hybrid_cache_entries=_hybrid_cache_entries,
        # Operator override for the D-METAL-CAP projection; 0 keeps the
        # architecture-aware auto-derivation. Needed by quantized-KV
        # deployments, whose real footprint the fp16 estimate over-states.
        metal_cap_kv_bytes_per_token=getattr(args, "metal_cap_kv_bytes_per_token", 0),
        non_trimmable_exact_prefix_reuse=(
            _hybrid_cache_entries > 0
            and _needs_bounded_trim_free_reuse(
                getattr(args, "_original_alias", None) or args.model
            )
        ),
        # Opt-in prompt-deterministic response cache (exact-match short-circuit).
        response_cache_entries=getattr(args, "response_cache_entries", 0),
        # Paged cache options
        use_paged_cache=args.use_paged_cache,
        paged_cache_block_size=args.paged_cache_block_size,
        max_cache_blocks=args.max_cache_blocks,
        # Prefill step size (chunk size). Must be plumbed here — BatchedEngine
        # reads it off scheduler_config only; the legacy load_model kwarg was
        # accepted but never used. See #400 and the CLI ↔ Config fidelity
        # audit at scripts/audit_cli_config_fidelity.py.
        prefill_step_size=args.prefill_step_size,
        vision_min_pixels=getattr(args, "vision_min_pixels", 0),
        vision_max_pixels=getattr(args, "vision_max_pixels", 0),
        # Speculative decoding selection.
        enable_mtp=getattr(args, "enable_mtp", False),
        mtp_num_draft_tokens=getattr(args, "mtp_num_draft_tokens", 1),
        mtp_optimistic=getattr(args, "mtp_optimistic", False),
        spec_decode=getattr(args, "spec_decode", "none"),
        dspark_num_speculative_tokens=getattr(args, "dspark_num_speculative_tokens", 5),
        dflash_drafter_path=getattr(args, "dflash_drafter_path", "") or "",
        # Optional external MTP sidecar path. ``None`` is the "no
        # sidecar; native-MTP path only" sentinel.
        mtp_sidecar=getattr(args, "mtp_sidecar", None),
        # 0.9.13 PR-A codex round-E blocker #2: CLI-resolved
        # ``config.json::model_type`` for the dispatch step. See the
        # ``_cli_mtp_model_type`` block above for why this is
        # resolved on the CLI thread instead of on the executor.
        mtp_model_type=_cli_mtp_model_type,
        # 0.9.13 PR-B: EV depth controller knobs.
        mtp_max_k=getattr(args, "mtp_max_k", 3),
        mtp_disable_auto_k=getattr(args, "mtp_disable_auto_k", False),
        # SuffixDecoding
        enable_suffix_decoding=args.suffix_decoding,
        suffix_max_draft=args.suffix_max_draft,
        suffix_max_suffix_len=args.suffix_max_suffix_len,
        suffix_min_confidence=args.suffix_min_confidence,
        suffix_min_draft_len=args.suffix_min_draft_len,
        # KV cache quantization (R15 #300: dtype string is the canonical
        # observability surface; ``_quantization`` / ``_bits`` are the
        # wire-level toggles that drive ``mlx_lm.QuantizedKVCache``).
        kv_cache_dtype=(
            kv_cache_decision.dtype if kv_cache_decision is not None else "bf16"
        ),
        kv_cache_quantization=args.kv_cache_quantization,
        kv_cache_quantization_bits=args.kv_cache_quantization_bits,
        kv_cache_quantization_group_size=args.kv_cache_quantization_group_size,
        kv_cache_min_quantize_tokens=args.kv_cache_min_quantize_tokens,
        # TurboQuant compression (R15 Phase 4: mode-aware). Shared
        # helper (#969) — ``python -m vllm_mlx.server`` calls the same
        # ``turboquant_scheduler_kwargs`` so both entrypoints stay in
        # lock-step. ``--kv-cache-turboquant`` carries a mode value:
        # ``None`` when off, ``"v4"`` for the legacy V-only path,
        # ``"k8v4"`` for the K-8bit + V-4bit mix. SchedulerConfig keeps
        # the boolean ``kv_cache_turboquant`` for downstream callers;
        # the mode string rides on the dedicated ``_mode`` field.
        **_turboquant_scheduler_kwargs(args),
        # R15-P1 (task #296): disk-backed KV checkpointing. ``0``
        # (default) disables; each snapshot blocks the decode thread for
        # O(context) so the feature is opt-in (#1853). The runtime module
        # guards every hot-path call with ``should_checkpoint`` so the
        # cost when off is one int comparison.
        kv_disk_checkpoint_interval=getattr(args, "kv_disk_checkpoint_interval", 0),
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
    if getattr(args, "spec_decode", "none") == "mtp":
        print(
            "MTP: enabled via --speculative-config, "
            f"max_k={getattr(args, 'mtp_max_k', 1)}"
        )
    if getattr(args, "spec_decode", "none") == "dspark":
        from vllm_mlx.spec_decode.dspark import detect_dspark_metadata

        metadata = detect_dspark_metadata(args.model)
        if metadata is None:
            print(
                "error: DSpark requires a complete DeepSeek V4 DSpark "
                "checkpoint (3 mtp stages plus Markov heads).",
                file=sys.stderr,
            )
            sys.exit(2)
        requested_k = getattr(args, "dspark_num_speculative_tokens", 5)
        if requested_k != metadata.block_size:
            print(
                "error: requested DSpark num_speculative_tokens="
                f"{requested_k} does not match checkpoint block_size="
                f"{metadata.block_size}. Rapid-MLX currently requires the "
                "checkpoint's complete DSpark block.",
                file=sys.stderr,
            )
            sys.exit(2)
        print(
            "DSpark: enabled via --speculative-config, "
            f"max_k={requested_k}, checkpoint_k={metadata.block_size}"
        )
    # Native Qwen3.5/3.6 MTP via vendored mlx-lm PR #990. The
    # config-only entrypoint is ``--speculative-config '{"method":"mtp"}'``.
    # Boot-time eligibility check fires here so misuse bounces with a clear
    # error instead of discovering the mismatch mid-generation.
    if getattr(args, "spec_decode", "none") == "mtp":
        from vllm_mlx.spec_decode.mtp import (
            MTPEligibility,
            detect_mtp_eligibility,
        )

        # ``_gather_kv_cache_dtype_inputs`` already reads
        # ``config.json`` for the same model the operator passed in;
        # reuse it so a side-loaded HF path or alias path both work.
        try:
            hf_cfg_eligibility, _ = _gather_kv_cache_dtype_inputs(args.model)
        except Exception:  # pragma: no cover — best-effort
            hf_cfg_eligibility = None
        has_sidecar = bool(getattr(args, "mtp_sidecar", None))
        eligibility = detect_mtp_eligibility(
            hf_cfg_eligibility, has_external_sidecar=has_sidecar
        )
        if eligibility is MTPEligibility.NONE:
            if has_sidecar:
                print(
                    "error: MTP speculative-config requires either a Qwen3.5 / "
                    "Qwen3.6 checkpoint with mtp_num_hidden_layers >= 1 in "
                    "config.json. Assistant sidecars are reserved for future "
                    "validated support and do not make this model eligible.",
                    file=sys.stderr,
                )
            else:
                print(
                    "error: MTP speculative-config requires a Qwen3.5 / "
                    "Qwen3.6 checkpoint with mtp_num_hidden_layers >= 1 in "
                    "config.json. Assistant sidecars are not currently "
                    "supported. The loaded model does not qualify.",
                    file=sys.stderr,
                )
            sys.exit(2)

        # Codex round-I BLOCKING #3 / round-K BLOCKING #2:
        # reconcile the earlier best-effort CLI-thread config read
        # with the eligibility gate's own read, and thread the
        # resulting model_type down into
        # ``scheduler_config.mtp_model_type``. Extracted into
        # :func:`_apply_mtp_cli_model_type_reconciliation` so a
        # regression test can drive the helper directly instead of
        # reimplementing the logic in the test body (round-K
        # BLOCKING #2 called out that a purely-inline reconciliation
        # block cannot be exercised without spinning up
        # ``serve_command``, which lets the test pass even if the
        # production block is later deleted).
        _apply_mtp_cli_model_type_reconciliation(
            scheduler_config=scheduler_config,
            hf_cfg_eligibility=hf_cfg_eligibility,
            logger=logger,
        )

        sidecar_note = (
            f" +sidecar={getattr(args, 'mtp_sidecar', None)}" if has_sidecar else ""
        )
        print(f"Spec-decode: mtp ({eligibility.value}){sidecar_note}")

    # DFlash is normalized from ``--speculative-config`` near the top of
    # serve_command. By the time we reach here, args.spec_decode is
    # "none" for dflash callers. The speculative.dflash gate at the
    # start of serve_command runs the actual eligibility +
    # drafter-binding checks via the prod bridge.
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
        index_choice = getattr(args, "prefix_cache_index", "radix")
        print(f"Memory-aware cache: {cache_info} (index={index_choice})")
        if args.kv_cache_turboquant:
            mode = args.kv_cache_turboquant
            if mode == "k8v4":
                print(
                    f"TurboQuant K8V4: K=8-bit Walsh-Hadamard, V=4-bit Lloyd-Max, "
                    f"group_size={args.kv_cache_turboquant_group_size}"
                )
            else:
                bits_str = (
                    str(args.kv_cache_turboquant_bits)
                    if args.kv_cache_turboquant_bits
                    else "auto"
                )
                print(
                    f"TurboQuant V-cache ({mode}): {bits_str}-bit, "
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

    # Alias-level unified-memory floor (codex #1069 round 3 [NIT #3]).
    # Fires BEFORE _check_disk_space so the user sees the actionable
    # "your Mac is too small for this Ultra-only alias" hint before we
    # start a 166 GB download.
    _check_alias_min_memory(args.model)

    # Check disk space before downloading model
    _check_disk_space(args.model, force=getattr(args, "force_disk_check", False))

    # Pre-flight memory check — warn (don't abort) if model + working set
    # would push unified memory past the kernel-panic threshold (issue #324).
    _check_memory_capacity(args.model)

    # DDTree fork: same blast-radius boundary as DFlash. It is a
    # speculative-decode mode, but the MVP runs through the external
    # dtree-mlx runtime rather than BatchedEngine.
    if args.enable_ddtree:
        from .speculative.ddtree.server import run_ddtree_server

        _alias_name = getattr(args, "_ddtree_alias_name", None)
        _profile = getattr(args, "_ddtree_profile", None)
        if _alias_name is None or _profile is None:
            _alias_name, _profile = _preflight_ddtree_or_exit(args)
        assert _profile is not None and _profile.supports_ddtree, (
            f"DDTree profile invariant violated for {_alias_name!r}"
        )
        assert _profile.ddtree_draft_model is not None
        assert _profile.ddtree_speculative_tokens is not None
        assert _profile.ddtree_tree_budget is not None
        run_ddtree_server(
            main_model_repo=_profile.hf_path,
            drafter_repo=getattr(args, "_ddtree_drafter_repo", None)
            or _profile.ddtree_draft_model,
            speculative_tokens=getattr(args, "_ddtree_speculative_tokens", None)
            or _profile.ddtree_speculative_tokens,
            tree_budget=getattr(args, "_ddtree_tree_budget", None)
            or _profile.ddtree_tree_budget,
            host=args.host,
            port=args.port,
            served_model_name=args.served_model_name or _alias_name,
            default_max_tokens=args.max_tokens,
            cors_origins=cors_origins,
            uvicorn_log_level=uvicorn_log_level,
            no_thinking=args.no_thinking,
            api_key=server._api_key,
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
    server.configure_model_residency(
        memory_limit_gb=getattr(args, "resident_memory_limit_gb", 0.0),
        idle_ttl_seconds=getattr(args, "resident_model_idle_ttl", 0.0),
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
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
            served_model_name=args.served_model_name,
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
            enable_disk_stream=getattr(args, "disk_stream", False),
            disk_stream_cache_gb=getattr(args, "disk_stream_cache_gb", 1.0),
        )
    except Exception as e:
        # Opt-in telemetry (Phase 2.2 error wiring): record that a model
        # failed to load on the ``serve`` path. The payload carries only a
        # bucketed category + a traceback fingerprint (basename:func:lineno
        # + exception class) — never the model name, message text, or path.
        # ``emit.error`` is ``is_enabled()``-gated and ``@_safe``, so it is a
        # no-op when telemetry is off and can never mask the user-facing
        # error handled just below.
        from vllm_mlx.telemetry import emit as _telemetry_emit

        _telemetry_emit.error(category="model_load_failure", exc=e, phase="startup")
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

    # Task #292 / codex r1 BLOCKING defense-in-depth: ``load_model``
    # already invokes ``register_audio_routes_if_enabled`` at its tail.
    # Calling it AGAIN here makes the wire-up explicit at the CLI
    # surface — a future refactor that moves the hook out of
    # ``load_model`` (e.g. into a lifespan event) won't silently drop
    # ``--enable-audio`` for the ``rapid-mlx serve`` path. The helper
    # is idempotent (app-local sentinel) so the second call is a
    # cheap attribute read.
    server.register_audio_routes_if_enabled()

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
    #     SSOT banner prints ``Ready: http://host:port`` (base URL; the
    #     OpenAI/Anthropic paths are separate rows).
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
    # (Codex PR #623 raised in review). Explicit guard returns 2 with a
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

    # Same gemma4 routing fix as ``bench_command``: ``mlx_lm.load`` cannot
    # construct ``gemma4_unified``, so every ``gemma-4-12b-*`` alias failed
    # to load here and could never be submitted to the community corpus.
    from .utils.tokenizer import load_model_with_fallback as load

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

    # Pre-fetch the model via the R2 mirror (with HF fallback) BEFORE the
    # thread executor spins up. Without this, ``mlx_lm.load`` runs inside
    # the executor and delegates to ``huggingface_hub.snapshot_download``
    # directly, skipping the mirror entirely (bug: --submit diverged from
    # ``serve``/``chat``/``pull`` which all prefetch via the
    # mirror first). Running this in the main thread — before the executor
    # is created — surfaces the mirror's per-file progress lines to the
    # contributor's terminal; if we deferred to the executor, the tqdm
    # bars from a warm cache-miss would race the ``Loading model …`` print
    # below. ``_ensure_model_downloaded`` is a no-op on local paths and on
    # fully-cached repos, and swallows mirror errors gracefully so
    # ``mlx_lm.load`` still falls through to HF when the mirror is
    # unreachable.
    _ensure_model_downloaded(hf_path)

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
            # ``gemma-4-*`` aliases are still in the served catalog (no
            # recommendation surface points at them anymore, but
            # ``rapid-mlx models`` lists them), so anyone picking one on
            # a base install would otherwise see a raw traceback and
            # conclude the model is broken — translate to an actionable
            # hint. Placed BEFORE
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
                # Pinned to ==0.6.3 to match VLM_EXTRA_INSTALL_HINT (0.10.16
                # dogfood ⑤): an unpinned mlx-vlm resolves to the current
                # PyPI latest, which pulls transformers 5.14.x and violates
                # rapid-mlx's own ``transformers<5.13`` core pin.
                print("    pip install --no-deps 'mlx-vlm==0.6.3'")
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
        # The spec-decode arm is the ONE comparability knob --submit lets the
        # caller move, because the whole point of the A/B is to measure it.
        # Everything else stays locked.
        _arm = getattr(args, "spec_decode", "none") or "none"
        scheduler_config = SchedulerConfig(
            max_num_seqs=1,
            max_concurrent_requests=1,
            prefill_batch_size=1,
            completion_batch_size=1,
            enable_prefix_cache=False,
            spec_decode=_arm,
        )
        engine_config = EngineConfig(
            model_name=hf_path,
            scheduler_config=scheduler_config,
        )
        _spec_payload = (
            None
            if _arm == "none"
            else {
                "method": _arm,
                "num_speculative_tokens": getattr(scheduler_config, "mtp_max_k", None),
            }
        )
        import re as _re

        _run_group = getattr(args, "run_group", None) or None
        if _run_group is not None and not _re.fullmatch(r"[0-9a-f]{12}", _run_group):
            print(
                "  Error: --run-group must be exactly 12 lowercase hex chars "
                f"(got {_run_group!r}). Generate one with: "
                "python3 -c 'import secrets;print(secrets.token_hex(6))'"
            )
            return 2

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
                    spec_decode=_spec_payload,
                    run_group=_run_group,
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

    # Use the SAME loader ``serve`` uses, not the bare ``mlx_lm.load``.
    # ``mlx_lm`` has no ``gemma4_unified`` architecture (the model classes
    # live in mlx-vlm), so a bare ``load`` raises "Model type
    # gemma4_unified not supported" for EVERY ``gemma-4-12b-*`` alias and
    # bench can never run them — even though ``serve`` runs them fine.
    # ``load_model_with_fallback`` carries the gemma4 router (plus the
    # vendored-arch and tokenizer-fallback routes) and calls
    # ``validate_local_model_file`` internally.
    from .engine_core import AsyncEngineCore, EngineConfig
    from .pflash import resolve_pflash_config as _pflash_resolve_config
    from .pflash import validate_model_support as _bench_pflash_validate
    from .request import SamplingParams
    from .scheduler import SchedulerConfig
    from .utils.model_file_guard import validate_local_model_file
    from .utils.tokenizer import load_model_with_fallback as load

    _check_disk_space(args.model, force=getattr(args, "force_disk_check", False))
    _check_memory_capacity(args.model)

    # Pre-fetch the model via the R2 mirror (with HF fallback) BEFORE the
    # heavy bench boot. Without this, ``bench`` falls into ``mlx_lm.load``
    # → ``huggingface_hub.snapshot_download`` directly and skips the
    # mirror entirely, wasting the user's bandwidth and hitting HF rate
    # limits (bug: bench diverged from ``serve``/``chat``/``pull``
    # which all prefetch via the mirror first).
    # ``_ensure_model_downloaded`` is a no-op on local paths and on
    # fully-cached repos and swallows mirror errors gracefully (mlx_lm.load
    # falls through to HF), so this is safe on the warm path and when
    # the mirror is unreachable.
    _ensure_model_downloaded(args.model)

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
    # ``bench`` has no MLLM/continuous-batching lane: it ALWAYS loads the text
    # model via ``mlx_lm.load`` and drives ``AsyncEngineCore`` with it (see
    # ``run_benchmark`` below — there is no ``force_text``/``force_mllm`` engine
    # surface here the way ``serve``'s ``BatchedEngine`` has). So the effective
    # bench lane is text for EVERY model, hybrid or not — there is no MLLM lane
    # here for which PFlash would be unavailable. PFlash defaulting AND
    # validation therefore use ``is_mllm=False`` unconditionally: an ordinary
    # (non-hybrid) VLM resolves to ``is_mllm=True`` on the serve path, but here
    # that would wrongly reject PFlash options as if an MLLM lane were in use
    # (#352 dogfood P1-②; codex NIT #1178). We still resolve the lane to surface
    # the hybrid auto-downgrade for lane attribution, and to keep a future
    # reader from "fixing" this by wiring an MLLM lane that would crash on the
    # hybrid backbone (GatedDeltaNet vs BatchKVCache, GH #352).
    from .api.utils import resolve_serving_lane as _bench_resolve_serving_lane

    _, _bench_auto_text_fallback = _bench_resolve_serving_lane(
        args.model,
        force_mllm=getattr(args, "mllm", False),
        force_text=getattr(args, "no_mllm", False),
    )
    if _bench_auto_text_fallback:
        print(
            f"Note: {args.model!r} is a hybrid VLM — benching on the text-only "
            "mlx-lm lane, matching 'serve' auto-downgrade (#352)."
        )
    # Same shared wiring as serve: resolve mode + per-alias keep_ratio override
    # and build the config, so bench measures the SAME effective PFlash config
    # serve would run. is_multimodal=False — bench has no MLLM lane (see above).
    try:
        bench_pflash_config = _pflash_resolve_config(
            args, model_name=args.model, is_multimodal=False
        )
        _bench_pflash_validate(
            bench_pflash_config,
            model_name=args.model,
            is_mllm=False,
        )
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    async def run_benchmark():
        print(f"Loading model: {args.model}")
        try:
            validate_local_model_file(args.model)
            # Load the weights ON the mlx-step worker (created + explained at
            # the ``bench_command`` call site below) and reuse that worker for
            # AsyncEngineCore. mlx-lm 0.31.3+ binds the generation stream to
            # whichever thread first touches MLX, so loading here — on the same
            # worker the engine later steps on — is what keeps the first batch
            # step from raising "There is no Stream(gpu, N) in current thread"
            # (which aborts every request and reports 0.00 tok/s).
            if getattr(args, "disk_stream", False):
                # --disk-stream: load lazily (routed-expert weights never
                # materialized) and patch the MoE blocks to stream them
                # from disk before this model reaches AsyncEngineCore.
                # Same helper `serve` uses, run on this SAME mlx-step
                # worker (#170 stream-ownership requirement above).
                from .engine.batched import _load_lazy_and_install_disk_stream

                model, tokenizer = model_load_executor.submit(
                    _load_lazy_and_install_disk_stream,
                    args.model,
                    {},
                    getattr(args, "disk_stream_cache_gb", 1.0),
                ).result()
            else:
                model, tokenizer = model_load_executor.submit(load, args.model).result()
        except Exception as e:
            # Opt-in telemetry (Phase 2.2 error wiring): mirror the
            # ``serve`` path — record a bucketed model-load failure
            # (category + traceback fingerprint only, no model name /
            # message / path). ``is_enabled()``-gated + ``@_safe``.
            from vllm_mlx.telemetry import emit as _telemetry_emit

            _telemetry_emit.error(category="model_load_failure", exc=e, phase="startup")
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
            # R15-P1 (task #303): radix-tree prefix-cache index. Same
            # default as the main serve path so benches reflect the
            # production index choice.
            prefix_cache_index=getattr(args, "prefix_cache_index", "radix"),
            # Memory-aware cache options
            use_memory_aware_cache=not args.no_memory_aware_cache,
            cache_memory_mb=args.cache_memory_mb,
            cache_memory_percent=args.cache_memory_percent,
            # #1103: bounded trim-free hybrid (recurrent-state) prefix reuse.
            # Bench path mirrors serve so hybrid-reuse effects show up in
            # `rapid-mlx bench` numbers too.
            hybrid_cache_entries=getattr(args, "hybrid_cache_entries", 0),
            # The prompt-deterministic response cache is a chat/serve
            # feature — its lookup/store logic lives only in the chat route
            # (vllm_mlx/routes/chat.py), and `rapid-mlx bench` never consumes
            # it. The bench parser therefore does not expose
            # --response-cache-entries; SchedulerConfig defaults it to 0 here,
            # leaving bench semantics unchanged.
            # Paged cache options
            use_paged_cache=args.use_paged_cache,
            paged_cache_block_size=args.paged_cache_block_size,
            max_cache_blocks=args.max_cache_blocks,
            # KV cache quantization
            kv_cache_quantization=args.kv_cache_quantization,
            kv_cache_quantization_bits=args.kv_cache_quantization_bits,
            kv_cache_quantization_group_size=args.kv_cache_quantization_group_size,
            kv_cache_min_quantize_tokens=args.kv_cache_min_quantize_tokens,
            # R15-P1 (task #296): disk-backed KV checkpointing. Bench
            # path mirrors serve (0 = off by default, #1853) so a
            # regression in the boundary trigger surfaces in
            # `rapid-mlx bench` numbers too.
            kv_disk_checkpoint_interval=getattr(args, "kv_disk_checkpoint_interval", 0),
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

        # Reuse the model-load worker as the engine's mlx-step thread so
        # weights + forward passes + cache state stay on one owning thread
        # (see the load-executor comment above).
        async with AsyncEngineCore(
            model, tokenizer, engine_config, executor=model_load_executor
        ) as engine:
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

    import concurrent.futures

    from .engine_core import _init_mlx_step_thread

    # Create the single mlx-step worker BEFORE loading weights and reuse it
    # for AsyncEngineCore so weights + forward passes + cache state all live
    # on one owning thread (#170). mlx-lm 0.31.3+ binds the module-level
    # generation stream to whichever thread first touches MLX; loading on the
    # main thread and generating on the engine worker raises "There is no
    # Stream(gpu, N) in current thread" on the first batch step → every
    # request aborts → the run silently reports 0.00 tok/s (the app's "Speed
    # on this Mac" card then shows a confident, wrong zero). Mirrors the
    # proven ``bench --submit`` path (``_run``) and ``BatchedEngine._start_llm``
    # — the exact reason ``serve``/``chat`` work and this freeform path did not.
    model_load_executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="mlx-step",
        initializer=_init_mlx_step_thread,
    )
    interrupted = False
    try:
        asyncio.run(run_benchmark())
    except KeyboardInterrupt:
        interrupted = True
        raise
    finally:
        # Caller-supplied executors are NOT owned or shut down by
        # AsyncEngineCore.stop(), so reap the worker here on every exit path.
        # On a normal or load-error (sys.exit) exit the worker is idle — the
        # engine closed its BatchGenerator on it during stop(), or the load
        # future already finished — so this join returns immediately.
        # ``cancel_futures`` drops any still-queued (never-started) work.
        #
        # Ctrl-C landing INSIDE the native ``load`` (an uninterruptible mlx
        # weight read) is a known, inherent limitation, NOT a regression from
        # this change: pre-PR this path loaded on the MAIN thread, equally
        # uninterruptible, so exit already blocked until the read finished.
        # ``wait=False`` here just avoids re-blocking in this ``finally``;
        # concurrent.futures' atexit hook still joins the non-daemon worker at
        # interpreter shutdown, so the wall-clock is unchanged either way and
        # no GPU work ever escapes the process. Truly aborting a native load
        # mid-read would need ``os._exit`` and risk leaving Metal state dirty
        # — not worth it for a benchmark command.
        model_load_executor.shutdown(wait=not interrupted, cancel_futures=True)


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
    """Bytes of the distinct files under ``path``, each counted once.

    The HF cache keeps one copy of every model file under
    ``blobs/<sha>`` and references it from ``snapshots/<rev>/<file>``.
    Counting both sides tallies the same bytes twice, plus once more for
    every extra cached revision, so a 7.9 GB model was reported as
    15.9 GiB in ``ls --cached`` and in the macOS app's Settings → Models
    panel. That number is what tells a user how much disk deleting the
    model would free, so being 2x off is a lie — and it disagreed with
    ``rapid-mlx rm``, which gets its figure from
    ``huggingface_hub``'s own (correct) ``size_on_disk``.

    Two rules keep the count honest:

    * Only regular files are measured. Symlinks found during the walk
      are skipped, not followed: their bytes live somewhere else and a
      snapshot link is just a second name for a blob already counted.
      A dangling link therefore costs nothing and raises nothing, and
      a link cannot be descended into, so the walk neither loops nor
      leaves the tree it was pointed at.
    * Files are deduped by ``(st_dev, st_ino)``, so a hardlinked layout
      (``cp -al`` cache clones, restored CI caches, older hub versions)
      and a blob shared by several revisions each contribute once. When
      the hub genuinely *copies* a blob into a snapshot instead of
      linking it, the two copies have distinct inodes and are counted
      twice — which is right, because they really do occupy twice the
      disk.

    Deduping by identity rather than by the ``blobs``/``snapshots``
    directory names is deliberate: the name-based shortcut breaks on
    hardlinked caches and on directories holding several revisions.
    ``HFCacheByteMonitor.directoryByteCount`` in the macOS app takes the
    same approach via ``fileResourceIdentifierKey``, so the two agree on
    the same directory up to allocation granularity: this sums logical
    ``st_size``, the Swift side sums allocated blocks, so every file
    differs a little through block rounding and sparse or compressed
    files can differ a lot (model weights are neither). ``st_size`` is
    also the quantity ``huggingface_hub`` reports as
    ``CachedRepoInfo.size_on_disk``, which is what ``rapid-mlx rm``
    already prints — matching it makes the two surfaces agree.

    ``path`` itself is resolved through symlinks before the walk starts:
    it names the directory the caller wants measured, so a relocated
    cache entry reports its real contents rather than 0, which would be
    a fresh lie in the same column. Callers pass a
    ``models--<org>--<repo>`` cache root (or any plain directory of real
    files), never a bare ``snapshots/<rev>`` — that subtree owns no
    bytes of its own, and sizing one is ``_snapshot_size_bytes``' job.

    Missing or unreadable paths report 0 rather than raising: this feeds
    a listing, not a decision. For the same reason the walk is not
    hardened against another process swapping a subdirectory for a
    symlink between the moment it is queued and the moment it is
    scanned. Racing that window inflates a number in a table the user
    asked for about their own cache; anyone able to run it can rewrite
    the weights outright. Descriptor-relative traversal
    (``O_DIRECTORY | O_NOFOLLOW`` plus ``dir_fd``) would close it and is
    not worth the fd bookkeeping here.
    """
    total = 0
    seen: set[tuple[int, int]] = set()
    # Resolve once, up front, so the walk starts inside a real directory
    # and every subdirectory below is reached without crossing a link.
    try:
        root = os.path.realpath(path)
    except OSError:
        return 0
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as it:
                entries = list(it)
        except OSError:
            # Vanished mid-walk, or a directory we may not read.
            continue
        for entry in entries:
            try:
                if entry.is_dir(follow_symlinks=False):
                    stack.append(entry.path)
                    continue
                if not entry.is_file(follow_symlinks=False):
                    # Symlink, fifo, socket, device — nothing this
                    # directory owns on disk.
                    continue
                st = entry.stat(follow_symlinks=False)
            except OSError:
                continue
            # A few network/FUSE mounts report ``st_ino == 0`` for every
            # file. Deduping on that would collapse the whole tree into
            # a single file, so fall back to counting each entry.
            if st.st_ino:
                key = (st.st_dev, st.st_ino)
                if key in seen:
                    continue
                seen.add(key)
            total += st.st_size
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


def _external_model_roots() -> list[str]:
    """Directories to scan for models another MLX runtime downloaded.

    Read from ``RAPID_MLX_EXTRA_MODEL_ROOTS`` (``os.pathsep``-separated,
    same convention as ``PATH``). Empty by default: scanning a user's
    disk uninvited is not ours to decide, and a wrong guess costs a slow
    recursive walk on every ``ls``.

    The desktop app populates this from the folder the user picked in
    Settings, which is why the env var is the interface rather than a
    hardcoded list of every MLX tool's default location.
    """
    raw = os.environ.get("RAPID_MLX_EXTRA_MODEL_ROOTS", "").strip()
    if not raw:
        return []
    from vllm_mlx.model_aliases import _external_model_root_values

    roots: list[str] = []
    for part in _external_model_root_values(raw):
        path = os.path.expanduser(part.strip())
        if path and os.path.isdir(path):
            roots.append(os.path.realpath(path))
    return roots


def _scan_external_model_dirs(
    roots: list[str] | None = None,
) -> list[tuple[str, int, float]]:
    """Find MLX-servable models sitting outside the HF hub cache.

    Other MLX runtimes write ``<root>/<publisher>/<repo>/`` rather than
    the hub's ``models--<org>--<name>/snapshots/<sha>/``, so
    :func:`_scan_hf_cache_models` cannot see them and a user who already
    has the weights is asked to download them again — on a machine where
    disk is usually the scarce resource.

    Returns the same ``(repo, size_bytes, mtime)`` triples as the hub
    scanner so both feed one renderer.

    What counts as a model is decided by
    :func:`vllm_mlx._download_gate._snapshot_is_complete`, the same check
    ``serve`` uses. That matters for two reasons: it mirrors mlx-lm's
    actual loader glob (``model*.safetensors``), so we never advertise a
    directory the loader would then refuse; and it excludes GGUF
    structurally rather than by blacklist — mlx-lm can *export* GGUF but
    has no load path for it, so listing one would offer a model that
    fails on start.

    Depth is capped at two levels (``<root>/<a>/<b>``). Every MLX tool
    lays models out as publisher/repo, and an uncapped walk over a
    directory the user pointed at could descend into an entire home
    folder.
    """
    from vllm_mlx._download_gate import _snapshot_is_complete
    from vllm_mlx.model_aliases import (
        _external_model_identifier_parts,
        _external_model_tree_is_contained,
    )

    if roots is None:
        roots = _external_model_roots()

    out: list[tuple[str, int, float]] = []
    seen_paths: set[str] = set()
    seen_repos: set[str] = set()

    def _complete(directory: str) -> bool:
        try:
            return _snapshot_is_complete(directory)
        except OSError:
            return False

    def _looks_like_model_root(directory: str) -> bool:
        try:
            with os.scandir(directory) as entries:
                return any(
                    entry.name == "model.safetensors.index.json"
                    or (
                        entry.name.startswith("model")
                        and entry.name.endswith(".safetensors")
                    )
                    for entry in entries
                )
        except OSError:
            return False

    canonical_roots = [os.path.realpath(root) for root in roots]

    def _record(directory: str, repo: str, canonical_root: str) -> None:
        real = os.path.realpath(directory)
        try:
            if os.path.commonpath((canonical_root, real)) != canonical_root:
                return
        except (OSError, ValueError):
            return
        # Ordered root precedence: the first configured occurrence of a repo
        # wins.  Dedup by both identity and display/launch identifier so two
        # separate roots cannot print duplicate rows or double-count bytes.
        if real in seen_paths or repo in seen_repos:
            return
        try:
            # The loader probe is root-only and bounded. Run it before the
            # recursive symlink audit/size walk so a broad selected folder
            # does not turn every ordinary directory into a deep traversal.
            if not _looks_like_model_root(real):
                return
            if not _external_model_tree_is_contained(real, canonical_roots):
                return
            if not _complete(real):
                return
            mtime = os.path.getmtime(real)
        except OSError:
            return
        try:
            size = _external_tree_size_bytes(real)
        except OSError:
            # External trees are owned by another process and may disappear,
            # lose permission, or contain a broken link while we scan. One
            # racy entry must not take down the entire ``rapid-mlx ls`` view.
            return
        seen_paths.add(real)
        seen_repos.add(repo)
        out.append((repo, size, mtime))

    for root in roots:
        canonical_root = os.path.realpath(root)

        def _contained(path: str, *, _root: str = canonical_root) -> bool:
            try:
                return os.path.commonpath((_root, os.path.realpath(path))) == _root
            except (OSError, ValueError):
                return False

        try:
            first_level = sorted(os.listdir(root))
        except OSError:
            continue
        for publisher in first_level:
            if _external_model_identifier_parts(publisher) is None:
                continue
            # Skip the hub layout: those belong to the hub scanner, and
            # listing them twice would double-count disk usage.
            if publisher.startswith(("models--", "datasets--", "spaces--")):
                continue
            pub_dir = os.path.join(root, publisher)
            if not os.path.isdir(pub_dir):
                continue

            # A model may sit directly at <root>/<name>/ as well as at
            # <root>/<publisher>/<name>/ — accept both.
            if _contained(pub_dir):
                _record(pub_dir, publisher, canonical_root)
                if publisher in seen_repos:
                    continue

            try:
                second_level = sorted(os.listdir(pub_dir))
            except OSError:
                continue
            for name in second_level:
                repo = f"{publisher}/{name}"
                if _external_model_identifier_parts(repo) is None:
                    continue
                model_dir = os.path.join(pub_dir, name)
                if os.path.isdir(model_dir) and _contained(model_dir):
                    _record(model_dir, repo, canonical_root)

    return out


def _cache_entry_is_runnable(repo: str) -> bool:
    """Whether a cache directory contains a complete runnable snapshot.

    A Hugging Face repo directory appears as soon as metadata starts
    downloading.  Treating directory presence as "cached" made interrupted
    downloads (config/tokenizer only, no weights) look ready in ``ls`` and in
    the desktop model picker. Reuse the serve download gate's authoritative
    completeness checks for both text and component-split video layouts.
    """
    try:
        from vllm_mlx._download_gate import (
            _snapshot_is_complete_mflux_model,
            _snapshot_is_complete_split_model,
            _snapshot_is_complete_whisper_model,
            is_repo_cached,
        )
        from vllm_mlx.audio.registry import resolve_audio_alias

        audio_entry = resolve_audio_alias(repo)
        if audio_entry is not None and audio_entry.family == "whisper":
            return _snapshot_is_complete_whisper_model(repo)
        return (
            is_repo_cached(repo)
            or _snapshot_is_complete_split_model(repo)
            or _snapshot_is_complete_mflux_model(repo)
        )
    except Exception:
        return False


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
    external_rows = _scan_external_model_dirs()
    # A RUNNABLE hub copy wins because it is managed by Rapid. Keep an
    # incomplete same-named hub row alongside the external copy: it is a real,
    # independently removable cache entry, and hiding it makes `rm <repo>` look
    # like it targets the read-only external row.
    runnable_hub_repos = {repo for repo, _, _ in rows if _cache_entry_is_runnable(repo)}
    external_rows = [r for r in external_rows if r[0] not in runnable_hub_repos]
    tagged_rows = [(*row, False) for row in rows] + [
        (*row, True) for row in external_rows
    ]
    print()
    if not tagged_rows:
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
        # Width is presentation only. Rows use an explicit two-space
        # delimiter below because the desktop parser splits on 2+ spaces;
        # padding alone cannot guarantee that invariant for a value that
        # exactly fills (or exceeds) its field.
        ("Size", 10),
        ("Modified", 12),
    )
    width = sum(w for _, w in cols) + 2 * (len(cols) - 1)
    sep = "  " + "─" * width
    header = "  " + "  ".join(f"{name:<{w}}" for name, w in cols)
    print(f"  Cached models ({len(tagged_rows)} on disk)")
    print(sep)
    print(header)
    print(sep)

    now = _time.time()
    total_bytes = 0
    # Sort by size descending so the biggest-disk-hog row is first — the
    # most useful ordering for "what do I rm to free space?".
    for repo, size, mtime, is_external_row in sorted(
        tagged_rows, key=lambda row: -row[1]
    ):
        total_bytes += size
        alias = hf_to_alias.get(repo, "(unmapped)")
        # Models found outside the hub cache are listed but never labelled
        # with an alias, and the desktop parser drops every parenthesized
        # alias except ``(unmapped)``. That is deliberate: ``rm`` and the
        # app's delete path both rebuild a target as
        # ``<hub-root>/models--<repo>``, which is not where these live.
        # Advertising one as deletable would either miss (nothing at that
        # path) or, worse, delete an unrelated hub entry that happens to
        # share the name. Read-only is the honest state — we did not
        # download them and we cannot manage them.
        if is_external_row:
            alias = "(external)"
        # Keep partial directories visible for disk cleanup, but never label
        # one as downloaded/runnable. This includes unmapped audio repos: the
        # desktop joins those to its audio registry by HF id, so leaving a
        # partial row as `(unmapped)` gives it a green checkmark and Start.
        # The desktop deliberately rejects `(incomplete)` status rows.
        elif not _cache_entry_is_runnable(repo):
            alias = "(incomplete)"
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
        # External identifiers are machine-consumed by the desktop and must
        # remain byte-for-byte launchable. Hub rows may still be truncated for
        # interactive display because their registered alias is the canonical
        # launch identity; external rows have no independent alias channel.
        repo_disp = repo if is_external_row or len(repo) <= 50 else (repo[:47] + "...")
        print(f"  {alias:<22}  {repo_disp:<50}  {_format_bytes(size):<10}  {mod:<12}")
    print(sep)
    print(f"  Total: {_format_bytes(total_bytes)}")
    if external_rows:
        print(
            "  Note: total is logical model size; shared external weights may repeat."
        )
    print()
    if external_rows:
        print("  Tip: `rapid-mlx rm` only removes Rapid-managed cache entries.")
        print("       Remove external models in the app that downloaded them.")
    else:
        print("  Tip: `rapid-mlx rm <hf-repo>` to free disk space")
    print()


def recipe_command(args) -> None:
    """Recommend exactly two curated models for this Mac's RAM tier.

    Recommendations stay anchored to the shared, curated RAM-tier SSOT. Disk
    pressure is presentation state, not a reason to silently substitute a
    lower-quality model: an unavailable pick remains visible, but we do not
    print a copy-paste ``serve`` command that is known to fail mid-download.
    """
    import json
    import math

    from vllm_mlx.model_aliases import resolve_profile
    from vllm_mlx.model_sizes import size_bytes
    from vllm_mlx.recommendations import physical_ram_gb, recommendation_payload

    ram_gb = (
        float(args.max_ram)
        if getattr(args, "max_ram", None) is not None
        else physical_ram_gb()
    )
    if ram_gb <= 0:
        raise SystemExit("Could not detect physical RAM. Pass --max-ram GB explicitly.")
    payload = recommendation_payload(ram_gb)
    cached_repos = {repo.casefold() for repo, _, _ in _scan_hf_cache_models()}
    free_disk_gb = _recipe_free_disk_gb()
    # Free space rounds DOWN while required space below rounds UP for display.
    # Fit itself still compares the unrounded measurements: presentation must
    # not reject a download that really has enough room. Two directed decimal
    # places ensure a failing boundary cannot render as equal values.
    payload["free_disk_gb"] = (
        None if free_disk_gb is None else math.floor(free_disk_gb * 100) / 100
    )
    for pick in payload["picks"]:
        try:
            profile = resolve_profile(pick["alias"])
            hf_path = profile.hf_path if profile is not None else None
        except ValueError:
            hf_path = None
        pick["hf_path"] = hf_path
        pick["cached"] = bool(
            hf_path
            and hf_path.casefold() in cached_repos
            and _cache_entry_is_runnable(hf_path)
        )
        # ``footprint_gb`` is measured 8K peak RAM, not bytes on disk. Use the
        # checked-in download-size manifest that powers ``rapid-mlx models``;
        # this keeps recipe offline and prevents a 20 GB RAM peak from being
        # misreported as a 20 GB download. Match the live download gate's 10%
        # headroom for xet temporary files and the final cache move.
        download_bytes = 0 if pick["cached"] else size_bytes(hf_path or "")
        required_disk_gb = (
            None if download_bytes is None else (download_bytes * 1.1) / float(1 << 30)
        )
        pick["download_size_gb"] = (
            None
            if download_bytes is None
            else round(download_bytes / float(1 << 30), 1)
        )
        pick["required_disk_gb"] = (
            None
            if required_disk_gb is None
            else math.ceil(required_disk_gb * 100) / 100
        )
        pick["disk_fit"] = (
            None
            if free_disk_gb is None or required_disk_gb is None
            else free_disk_gb >= required_disk_gb
        )

    if getattr(args, "json", False):
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print(
        f"Recommended for this {payload['physical_ram_gb']:.1f} GB Mac "
        f"({payload['tier_floor_gb']} GB tier)"
    )
    labels = {"smart": "Smart", "fast": "Fast"}
    for index, pick in enumerate(payload["picks"], start=1):
        # Label the footprint as RAM: it is measured 8K peak memory, a
        # different axis from the on-disk ``required_disk_gb`` shown in the
        # won't-fit line below. Without the label the two GB numbers read as
        # contradictory (e.g. "20.0 GB" then "needs ~16.72 GB").
        stats = [f"{pick['footprint_gb']:.1f} GB RAM"]
        if pick.get("caveat"):
            stats.append(pick["caveat"])
        else:
            stats.append(f"{pick['capability_pct']}% capability")
        if pick.get("tokens_per_sec") is not None:
            stats.append(f"~{round(pick['tokens_per_sec'])} tok/s")
        cache_badge = " · cached" if pick["cached"] else ""
        print(f"\n{index}. {labels[pick['role']]} — {pick['alias']}{cache_badge}")
        print(f"   {' · '.join(stats)}")
        if pick["disk_fit"] is False:
            print(
                f"   ⚠ won't fit: needs ~{pick['required_disk_gb']:.2f} GB "
                f"including download headroom; {payload['free_disk_gb']:.2f} GB free"
            )
            print("   Free disk space or set HF_HOME/HF_HUB_CACHE to another drive.")
            continue
        command = f"rapid-mlx serve {pick['alias']}"
        if pick["launch_flags"]:
            command += " " + " ".join(pick["launch_flags"])
        print(f"   {command}")


def _recipe_free_disk_gb() -> float | None:
    """Return free GiB on the filesystem that receives HF downloads.

    ``HF_HUB_CACHE`` can name a directory that does not exist yet, including
    one on an external volume. Walk to its nearest existing ancestor before
    probing, matching the real download gate rather than assuming ``$HOME``.
    Unknown disk state is deliberately ``None``: recipe then preserves its
    historical output instead of claiming that a model fits.
    """
    import shutil
    from pathlib import Path

    try:
        from huggingface_hub.constants import HF_HUB_CACHE
    except Exception:
        HF_HUB_CACHE = str(Path.home() / ".cache" / "huggingface" / "hub")
    try:
        probe = Path(HF_HUB_CACHE).expanduser().absolute()
        while not probe.exists() and probe.parent != probe:
            probe = probe.parent
        if not probe.exists():
            return None
        return shutil.disk_usage(probe).free / float(1 << 30)
    except (OSError, TypeError, ValueError):
        return None


def _cached_models_json_payload() -> dict:
    """Structured form of the ``models --cached`` view — the same rows the
    text table renders, with stable keys instead of fixed-width columns.

    Sizes are raw bytes; ``state`` is one of ``ok`` / ``unmapped`` /
    ``incomplete`` / ``external`` mirroring the alias column's parenthesized
    tags; ``alias`` is ``None`` for any non-``ok`` row (those are not
    launchable by alias). Sorted biggest-first, like the table.
    """
    import time as _time

    from vllm_mlx.model_aliases import list_profiles

    rows = _scan_hf_cache_models()
    external_rows = _scan_external_model_dirs()
    runnable_hub_repos = {repo for repo, _, _ in rows if _cache_entry_is_runnable(repo)}
    external_rows = [r for r in external_rows if r[0] not in runnable_hub_repos]

    profiles = list_profiles()
    hf_to_alias: dict[str, str] = {}
    for alias, p in profiles.items():
        hf_to_alias.setdefault(p.hf_path, alias)

    now = _time.time()
    tagged = [(*row, False) for row in rows] + [(*row, True) for row in external_rows]
    models = []
    total_bytes = 0
    for repo, size, mtime, is_external in tagged:
        total_bytes += size
        if is_external:
            alias, state = None, "external"
        elif not _cache_entry_is_runnable(repo):
            alias, state = None, "incomplete"
        else:
            mapped = hf_to_alias.get(repo)
            alias, state = (mapped, "ok") if mapped is not None else (None, "unmapped")
        models.append(
            {
                "alias": alias,
                "repo": repo,
                "size_bytes": int(size),
                "modified_epoch": int(mtime) if mtime and mtime > 0 else None,
                "age_seconds": int(max(0, now - mtime))
                if mtime and mtime > 0
                else None,
                "state": state,
                "external": is_external,
            }
        )
    models.sort(key=lambda m: -m["size_bytes"])
    return {"cached": models, "count": len(models), "total_bytes": int(total_bytes)}


def _available_models_json_payload() -> dict:
    """Structured form of the default ``models`` view: every alias with the
    profile facts the table shows, split by modality (text / audio / video /
    image) exactly as the text sections are. Sizes are download bytes from the
    checked-in manifest (``None`` when unknown); no per-invocation HF I/O.
    """
    from vllm_mlx.model_aliases import list_profiles
    from vllm_mlx.model_sizes import size_bytes

    all_profiles = list_profiles()

    def _modality(p) -> str:
        return getattr(p, "modality", "text") or "text"

    def _profile_dict(alias, p) -> dict:
        raw = None
        try:
            raw = size_bytes(p.hf_path)
        except Exception:
            raw = None
        return {
            "alias": alias,
            "hf_path": p.hf_path,
            "size_bytes": int(raw) if isinstance(raw, int) and raw > 0 else None,
            "tool_call_parser": p.tool_call_parser,
            "reasoning_parser": p.reasoning_parser,
            "is_hybrid": bool(getattr(p, "is_hybrid", False)),
            "is_moe": bool(getattr(p, "is_moe", False)),
            "supports_spec_decode": bool(getattr(p, "supports_spec_decode", False)),
            "modality": _modality(p),
        }

    text, video, image = {}, {}, {}
    for alias, p in all_profiles.items():
        bucket = {"video-gen": video, "image-gen": image}.get(_modality(p), text)
        bucket[alias] = p

    payload = {
        "text": [_profile_dict(a, text[a]) for a in sorted(text)],
        "video": [_profile_dict(a, video[a]) for a in sorted(video)],
        "image": [_profile_dict(a, image[a]) for a in sorted(image)],
        "audio": [],
    }
    try:
        from vllm_mlx.audio.registry import list_audio_aliases

        payload["audio"] = [
            {
                "alias": e.alias,
                "hf_id": e.hf_id,
                "kind": e.type,
                "family": e.family,
                "modality": "audio",
            }
            for e in list_audio_aliases()
        ]
    except Exception:
        payload["audio"] = []
    return payload


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
    from vllm_mlx.model_sizes import format_size

    # JSON mode emits ONLY the payload on stdout — no staleness banner, no
    # table — so a caller can pipe it straight into a parser.
    if getattr(args, "json", False):
        import json as _json

        if getattr(args, "cached", False):
            print(_json.dumps(_cached_models_json_payload()))
        else:
            print(_json.dumps(_available_models_json_payload()))
        return

    print_staleness_warning_if_any()

    if getattr(args, "cached", False):
        _print_cached_models()
        return

    all_profiles = list_profiles()
    # Video-generation aliases are not chat models: they have no
    # tokenizer and no ``stream_chat``, so ``/v1/chat/completions`` on one
    # is an AttributeError, and ``serve`` exits 2 before binding a port
    # when the video extras are absent. Listing them inline in the text
    # table is how a GUI catalog consumer ends up offering a 64 GiB
    # download that can never chat (#1603). Split them into their own
    # tagged section, exactly as audio aliases already are, so a consumer
    # can tell the two kinds apart without hardcoding alias names.
    video_profiles = {
        alias: p
        for alias, p in all_profiles.items()
        if getattr(p, "modality", "text") == "video-gen"
    }
    # Image-generation aliases (mflux FLUX / Qwen-Image) are likewise not chat
    # models — ``/v1/chat/completions`` on one is unrouted. Split them into
    # their own ``[image:gen]``-tagged section for the same catalog-integrity
    # reason as video (#1603): a GUI consumer must be able to tell an image
    # model from a chat model without hardcoding alias names, or it will offer
    # a multi-GB image checkpoint as a chat model that dead-ends on first send.
    image_profiles = {
        alias: p
        for alias, p in all_profiles.items()
        if getattr(p, "modality", "text") == "image-gen"
    }
    profiles = {
        a: p
        for a, p in all_profiles.items()
        if a not in video_profiles and a not in image_profiles
    }
    print()
    print(f"  Available models ({len(profiles)} aliases)")

    # Alias width is computed from the actual registry so new long names
    # (e.g. ``deepseek-coder-v2-lite-16b-4bit``, 31 chars) don't push the
    # rest of their row out of column alignment. 24 is the historical
    # floor — never shrink below it so short rows still feel padded.
    # Other widths sized to fit values currently in aliases.json:
    # tool 16 (qwen3_coder_xml + 1 pad), reasoning 12 (deepseek_r1 + 1),
    # spec 10 ("✗ hybrid"), tier 11, dflash 7, ddtree 7, preset 8.
    alias_width = max(24, max((len(a) for a in profiles), default=0) + 2)
    # Size ("438.3 GiB" is the widest current value) comes right after the
    # alias so the "how big before I pull?" answer is the first thing a user
    # sees next to the name (issue #1286). Values come from the checked-in
    # model_sizes.json manifest — no per-invocation HuggingFace round-trip.
    # Tools and Reasoning carry parser keys whose length is unbounded
    # (``deepseek_r1_distill`` is 19 chars, wider than the old fixed 12).
    # Size them to the data like the alias column so a long value never
    # overflows and shifts every field to its right out of the header's
    # columns (#1999). Preset is last, so its long ``MTP@…`` values can
    # overrun harmlessly and stay unbounded.
    tools_width = max(
        16, max((len(p.tool_call_parser or "—") for p in profiles.values()), default=0)
    )
    reasoning_width = max(
        12,
        max((len(p.reasoning_parser or "—") for p in profiles.values()), default=0),
    )
    cols = (
        ("Alias", alias_width),
        ("Size", 10),
        ("Tools", tools_width),
        ("Reasoning", reasoning_width),
        ("Spec-Decode", 10),
        ("Suffix Tier", 11),
        ("DFlash", 7),
        ("DDTree", 7),
        ("Preset", 8),
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
        if p.mtp_draft_model:
            spec = "✓ MTP"
            tier = "n/a"
            preset = f"MTP@{p.mtp_draft_model}@{p.mtp_speculative_tokens}"
        elif p.is_hybrid:
            # Hybrid models cannot use spec-decode or suffix-decode regardless
            # of the supports_spec_decode flag (mlx-lm BatchGenerator gate).
            spec = "✗ hybrid"
            tier = "n/a"
        else:
            spec = "✓" if p.supports_spec_decode else "✗"
            tier = p.suffix_decoding_tier
            preset = "Suffix" if p.supports_spec_decode else "—"
        if p.is_hybrid and not p.mtp_draft_model:
            preset = "—"
        # DFlash column — eligible aliases show ✓, everything else "—" so
        # the visual scan immediately surfaces what supports it. We don't
        # re-run the eligibility gate here (which would also check that
        # mlx-vlm 0.5.0+ is installed) — that's a runtime concern; the
        # registry column is pure declarative state.
        dflash = "✓" if p.supports_dflash else "—"
        ddtree = "✓" if p.supports_ddtree else "—"
        size = format_size(p.hf_path)
        row = (
            f"  {alias:<{alias_width}} {size:<10} {tools:<{tools_width}} "
            f"{reasoning:<{reasoning_width}} "
            f"{spec:<10} {tier:<11} {dflash:<7} {ddtree:<7} {preset:<8}"
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
        audio_alias_width = max(
            24, max((len(e.alias) for e in audio_entries), default=0) + 2
        )
        print()
        print(f"  Audio models ({len(audio_entries)} aliases)")
        audio_header = (
            f"  {'Alias':<{audio_alias_width}} {'Size':<10} {'Kind':<10} "
            f"{'Family':<12} {'HF id':<40}"
        )
        # Size the rule to THIS table's header, not the text table's width —
        # the text table's Tools/Reasoning columns are data-sized now (#1999),
        # so reusing its width would stretch every secondary rule.
        audio_sep = "  " + "─" * (len(audio_header) - 2)
        print(audio_sep)
        print(audio_header)
        print(audio_sep)
        for entry in audio_entries:
            kind_tag = f"[audio:{entry.type}]"
            print(
                f"  {entry.alias:<{audio_alias_width}} "
                f"{format_size(entry.hf_id):<10} {kind_tag:<10} "
                f"{entry.family:<12} {entry.hf_id:<40}"
            )
        print(audio_sep)

    # Video-generation aliases, in their own tagged section for the same
    # reason audio has one: they are not chat models, and a catalog
    # consumer must be able to tell that from the output rather than by
    # hardcoding names (#1603). The ``[video:gen]`` Kind tag mirrors
    # ``[audio:tts]`` / ``[audio:stt]``.
    if video_profiles:
        video_alias_width = max(
            24, max((len(a) for a in video_profiles), default=0) + 2
        )
        print()
        print(f"  Video models ({len(video_profiles)} aliases)")
        video_header = (
            f"  {'Alias':<{video_alias_width}} {'Size':<10} {'Kind':<11} {'HF id':<40}"
        )
        video_sep = "  " + "─" * (len(video_header) - 2)
        print(video_sep)
        print(video_header)
        print(video_sep)
        for alias in sorted(video_profiles):
            p = video_profiles[alias]
            print(
                f"  {alias:<{video_alias_width}} "
                f"{format_size(p.hf_path):<10} {'[video:gen]':<11} "
                f"{p.hf_path:<40}"
            )
        print(video_sep)

    # Image aliases carry an operation tag: text-to-image checkpoints use
    # ``[image:gen]`` and instruction-edit checkpoints use ``[image:edit]``;
    # FLUX.2 Klein accepts both request shapes and uses ``[image:both]``.
    # Besides keeping both out of chat catalogs, this lets GUI consumers expose
    # the right request shape without guessing capability from the alias.
    if image_profiles:
        image_alias_width = max(
            24, max((len(a) for a in image_profiles), default=0) + 2
        )
        print()
        print(f"  Image models ({len(image_profiles)} aliases)")
        image_header = (
            f"  {'Alias':<{image_alias_width}} {'Size':<10} {'Kind':<11} {'HF id':<40}"
        )
        image_sep = "  " + "─" * (len(image_header) - 2)
        print(image_sep)
        print(image_header)
        print(image_sep)
        for alias in sorted(image_profiles):
            p = image_profiles[alias]
            folded_path = p.hf_path.casefold().replace("_", "-")
            if (
                "flux2" in folded_path
                or "flux.2" in folded_path
                or "klein" in folded_path
            ):
                kind_tag = "[image:both]"
            elif "qwen-image-edit" in folded_path:
                kind_tag = "[image:edit]"
            else:
                kind_tag = "[image:gen]"
            print(
                f"  {alias:<{image_alias_width}} "
                f"{format_size(p.hf_path):<10} {kind_tag:<12} "
                f"{p.hf_path:<40}"
            )
        print(image_sep)

    print()
    print(
        "  Size is an approximate download footprint (weight+tokenizer); "
        "“—” = unknown. The exact size is confirmed at pull time."
    )
    print("  Tip: `rapid-mlx info <alias>` for the full per-model profile")
    print("       `rapid-mlx pull <alias>` to download")
    print("       `rapid-mlx chat <alias>` for an interactive REPL")
    print("       `rapid-mlx serve <alias>` for an OpenAI-compatible server")
    if audio_entries:
        print("       `rapid-mlx serve kokoro|whisper-large-v3|parakeet` for audio")
    # User mappings are rendered explicitly so support output never makes a
    # private nickname look like an immutable catalog alias.
    from vllm_mlx.model_aliases import list_builtin_aliases, user_alias_reserved_names
    from vllm_mlx.user_aliases import validated_user_aliases

    user_aliases = validated_user_aliases(
        list_builtin_aliases(), user_alias_reserved_names()
    )
    if user_aliases:
        print()
        print(f"  User aliases ({len(user_aliases)})")
        for name, target in sorted(user_aliases.items()):
            print(f"  {name} -> {target}")
    print()


def _format_pull_duration(seconds: float) -> str:
    """Render a duration as ``Xs`` (< 60s) or ``Xm Ys`` (>= 60s).

    Sub-minute keeps one decimal so a 4.2s pull doesn't read as ``4s``;
    once we cross a minute the decimals are noise. ``round`` (not
    ``int``) on the whole-second branch means ``119.9s`` reads as
    ``2m 0s`` instead of ``1m 59s``.
    """
    if seconds < 0:
        seconds = 0.0
    if seconds < 60:
        return f"{seconds:.1f}s"
    total = int(round(seconds))
    minutes, secs = divmod(total, 60)
    return f"{minutes}m {secs}s"


def _snapshot_size_bytes(path) -> int:
    """Sum file sizes under ``path`` (recursively, following symlinks).

    The HF cache stores ``snapshots/<rev>/<file>`` as symlinks into
    ``blobs/<sha>``; ``stat()`` follows the link so the byte count is
    the real on-disk weight, matching what the user just downloaded.
    Quietly tolerates partial / missing trees so the summary line is
    a print, not a crash, in degenerate cache states.
    """
    from pathlib import Path

    root = Path(path)
    if not root.exists():
        return 0
    total = 0
    try:
        for entry in root.rglob("*"):
            try:
                if entry.is_file():
                    total += entry.stat().st_size
            except OSError:
                continue
    except OSError:
        pass
    return total


def _external_tree_size_bytes(path: str) -> int:
    """Logical external-tree bytes, following shared files only once."""
    total = 0
    seen: set[tuple[int, int]] = set()

    def inaccessible(error: OSError) -> None:
        raise error

    for current, _directories, files in os.walk(
        path, followlinks=False, onerror=inaccessible
    ):
        for name in files:
            stat = os.stat(os.path.join(current, name), follow_symlinks=True)
            identity = (stat.st_dev, stat.st_ino)
            if identity in seen:
                continue
            seen.add(identity)
            total += stat.st_size
    return total


def _print_pull_summary(repo_id: str, snapshot_dir, elapsed: float) -> None:
    """Emit the one-line ``Downloaded ... — <size> in <duration>`` summary."""
    import os as _os

    from vllm_mlx.model_aliases import resolve_subfolder

    # A filtered pull fetched one folder, but the snapshot root may also
    # hold quant folders left by earlier pulls of a sibling alias. Sizing
    # the root would report those as part of THIS download.
    _sub = resolve_subfolder(repo_id)
    if _sub:
        _candidate = _os.path.join(str(snapshot_dir), _sub)
        if _os.path.isdir(_candidate):
            snapshot_dir = _candidate
    size = _snapshot_size_bytes(snapshot_dir)
    print(
        f"  Downloaded {repo_id} — {_format_bytes(size)} in "
        f"{_format_pull_duration(elapsed)}"
    )

    # Activation funnel (docs/telemetry-activation.md): a successful pull is
    # the ``model_pull`` milestone (an activation, NOT inference-engaged).
    # This helper is only reached on the two success exits of ``pull_command``,
    # so it is the single "pull succeeded" chokepoint. Fired once per install,
    # consent-gated + ``@_safe``, so it is a no-op when telemetry is off and
    # can never affect the pull.
    from vllm_mlx.telemetry import emit as _telemetry_emit
    from vllm_mlx.telemetry.activation_spec import ACTIVATION_MODEL_PULL, SURFACE_CLI

    _telemetry_emit.activation(
        activation_kind=ACTIVATION_MODEL_PULL, surface=SURFACE_CLI
    )


def pull_command(args):
    """Download a model to the HuggingFace cache without serving."""
    import time

    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import HFValidationError
    from huggingface_hub.utils import RepositoryNotFoundError

    repo_id = args.model  # already alias-resolved by main()
    t0 = time.monotonic()

    # Reclaim scratch files stranded by earlier interrupted pulls of THIS repo
    # before adding more. huggingface_hub gives each attempt a uniquely-named
    # ``.incomplete`` blob and removes it while unwinding, which a killed
    # process never gets to do — so a cancel, quit or crash leaves one behind
    # every time and nothing else ever collects them. Best-effort and
    # age-gated; see the helper for why it cannot disturb a live download.
    try:
        from vllm_mlx._download_gate import _format_size, reap_orphan_incomplete_blobs

        _reaped, _bytes = reap_orphan_incomplete_blobs(repo_id)
        if _reaped:
            print(
                f"  Cleaned up {_reaped} abandoned download file(s) "
                f"from earlier interrupted pulls ({_format_size(_bytes)})."
            )
    except Exception:
        pass

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
            snapshot_dir = repo_root / "snapshots" / rev
            print(f"  Cached at: {snapshot_dir}")
        except OSError:
            snapshot_dir = repo_root
            print(f"  Cached at: {repo_root}")
        _print_pull_summary(repo_id, snapshot_dir, time.monotonic() - t0)
        return
    # Mirror returned False — fall through to plain snapshot_download.
    # Either the catalog was unreachable, the alias isn't catalog-listed,
    # or one or more files failed both R2 and HF in the per-file pool.
    # snapshot_download will retry from HF with its own (more robust)
    # error reporting.
    # Say plainly that this path does not resume. The mirror completes a
    # partial ``.part`` with a ranged request, so an interrupted mirror pull
    # picks up where it left off; huggingface_hub gives each attempt its own
    # scratch file and never reuses one, so an interrupted HF pull starts the
    # affected files over from zero. Silently switching between "resumes" and
    # "restarts" is what makes a flaky connection read as "the download is
    # stuck" — the bytes really do go back to the beginning each time.
    print(f"\n  Pulling {repo_id} from HuggingFace ...")
    print("  Note: this path does not resume — interrupting it restarts the")
    print("  files still in flight from the beginning.")
    try:
        from vllm_mlx.model_aliases import resolve_subfolder

        # A repo that ships one folder per quantization holds many times
        # more than any one alias needs — unfiltered, this fetches all
        # eight LFM2.5-2.6B checkpoints (~20 GB) to serve 1.6 GB of them.
        #
        # The narrowing applies even when the operator typed the bare repo
        # id rather than the alias, because every other consumer already
        # keys on the repo id and reaches inside: the download gate sizes
        # the subfolder, ``is_repo_cached`` probes the subfolder, and the
        # loader opens the subfolder. Pulling the whole repo here would
        # make the gate's "1.5 GiB" quote a lie and leave seven
        # checkpoints on disk that nothing can serve. It is announced
        # rather than silent so ``pull <repo-id>`` never quietly does
        # something narrower than it was asked.
        _subfolder = resolve_subfolder(repo_id)
        if _subfolder:
            print(
                f"  This repo ships one checkpoint per quantization; "
                f"fetching only {_subfolder}/ (the folder rapid-mlx serves)."
            )
        _allow = [f"{_subfolder}/*"] if _subfolder else None
        path = (
            snapshot_download(repo_id, allow_patterns=_allow)
            if _allow
            else snapshot_download(repo_id)
        )
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
    _print_pull_summary(repo_id, path, time.monotonic() - t0)


def rm_command(args):
    """Remove a model from the HuggingFace cache.

    Default flow prompts for confirmation, defaulting to N — a real user
    typo (``rapid-mlx rm qwn3.5-9b-4bit`` → matches a 6 GB model) could
    silently nuke gigabytes of weights pre-0.9.7. EOF (non-TTY pipe,
    ctrl-D) also cancels rather than being treated as accept-by-default.
    ``-y/--yes`` skips the prompt for scripts.
    """
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
    size_str = _format_bytes(repo.size_on_disk)

    if not getattr(args, "yes", False):
        try:
            response = input(f"Remove {repo_id} ({size_str})? [y/N] ").strip().lower()
        except EOFError:
            # Non-TTY (piped stdin, ctrl-D) — treat as cancel, never as
            # silent-yes. Matches ``apt`` / ``brew`` muscle memory.
            print("Aborted.")
            sys.exit(0)
        if response not in ("y", "yes"):
            print("Aborted.")
            sys.exit(0)

    revisions = [rev.commit_hash for rev in repo.revisions]
    strategy = cache.delete_revisions(*revisions)
    strategy.execute()
    print(f"Freed {size_str}")


def alias_command(args) -> None:
    """Create, remove, or list user-owned model aliases."""
    from vllm_mlx.model_aliases import list_builtin_aliases, user_alias_reserved_names
    from vllm_mlx.user_aliases import (
        UserAliasError,
        remove_user_alias,
        set_user_alias,
        validated_user_aliases,
    )

    builtins = list_builtin_aliases()
    reserved = user_alias_reserved_names()
    try:
        if args.alias_action == "set":
            set_user_alias(args.name, args.target, builtins, reserved)
            print(f"  User alias: {args.name} -> {args.target}")
        elif args.alias_action == "remove":
            if not remove_user_alias(args.name, builtins, reserved):
                print(f"  User alias {args.name!r} does not exist.", file=sys.stderr)
                raise SystemExit(1)
            print(
                f"  Removed user alias {args.name!r}; cached weights were not changed."
            )
        else:
            aliases = validated_user_aliases(builtins, reserved)
            if not aliases:
                print("  No user aliases configured.")
            else:
                for name, target in sorted(aliases.items()):
                    print(f"  {name} -> {target}")
    except UserAliasError as exc:
        print(f"\n  Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None


def _elide_front(text: str, width: int) -> str:
    """Trim ``text`` to at most ``width`` chars, keeping the TAIL.

    For a served model that is a filesystem path, the tail (the model name) is
    the distinctive part, so the head is what gets replaced by a leading ``…``.
    The result never exceeds ``width`` (widths <= 1 leave no room for the ``…``).
    """
    if len(text) <= width:
        return text
    if width <= 0:
        return ""
    if width == 1:
        return text[-1:]
    return "…" + text[-(width - 1) :]


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

        # 0.9.0 dogfood: ``rapid-mlx serve`` runs under a ``caffeinate
        # -is rapid-mlx serve ...`` wrapper on macOS to prevent sleep.
        # The wrapper's argv carries the same ``rapid-mlx`` / ``serve``
        # tokens as the real server, so the substring match above
        # double-counts it as a second row (same port, different PID).
        # The wrapper is never the actual server — its argv[0] basename
        # is the only reliable way to filter it out without missing the
        # case where caffeinate is launched via an absolute path.
        if cmd and os.path.basename(cmd[0]) == "caffeinate":
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
            "--trusted-hosts",
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
    # Serving a local path (not an alias) is the normal case for a converted
    # model, and those paths routinely exceed the 40-char column — the old
    # code let them run straight into UPTIME with no gap (#1999). Elide from
    # the FRONT so the distinctive tail (the model name) survives, capped at 38
    # so the ``<40`` pad always leaves at least two spaces before UPTIME.
    # Sort numerically by port — string sort would put "10000" before "8000".
    for pid, port, model, uptime in sorted(rows, key=lambda r: int(r[1])):
        print(f"  {pid:<8}{port:<8}{_elide_front(model, 38):<40}{uptime:<10}")
    print()


def _spawn_chat_server(
    model: str,
    log_path: str,
    served_name: str | None = None,
    *,
    register_in: list | None = None,
    log_handle=None,
    disable_prefix_cache: bool = False,
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

    ``log_handle`` is the ``managed_tempfile_path`` context-manager handle
    that owns ``log_path``. When provided, ownership is transferred to the
    proc inside a SIGTERM/SIGINT-masked critical section that also performs
    the ``register_in`` append and the ``_rapid_mlx_log*`` attribute set.
    Without the mask + atomic transfer, a signal landing between
    ``_active_procs.append`` and the caller's later ``handle.release()``
    could fire ``_teardown_proc`` (which intentionally keeps non-empty
    logs for post-mortem) — then the ``with`` block's ``finally`` would
    unlink the kept log anyway, violating the keep-non-empty-log policy
    documented on ``_teardown_proc``. Codex round-1 BLOCKING #1.

    If ``served_name`` is given, it is passed via ``--served-model-name`` so
    the spawned server exposes the alias as the API model name (e.g. user
    typed ``qwen3.5-4b-4bit`` → API requests use ``qwen3.5-4b-4bit`` rather than the
    expanded HF path).
    """
    import signal as _signal
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
    if disable_prefix_cache:
        cmd.append("--disable-prefix-cache")
    log = open(log_path, "w")  # noqa: SIM115 — kept open for proc lifetime
    # Tell the child main() that the parent already gated (or that this is
    # an internal spawn, where prompting would deadlock anyway because the
    # child stdin is not a TTY). Without this, the child's B2 gate would
    # see a stdin pipe and re-evaluate against a potentially-stale cache.
    child_env = os.environ.copy()
    child_env["RAPID_MLX_CHAT_SPAWN"] = "1"
    # Parent-PID watchdog (rapid-desktop #449 sibling fix). The
    # SIGTERM-handler + atexit pair installed below cannot fire under
    # SIGKILL of the chat REPL — the spawned ``serve`` would otherwise
    # outlive ``rapid-mlx chat`` and keep the model + port locked. The
    # watchdog inside the child polls ``os.getppid()`` every 2 s and
    # self-terminates the moment the live PPID stops matching this
    # stamp.
    #
    # Direct assignment (NOT setdefault). Codex r2 MAJOR: if the chat
    # REPL itself was launched under a supervisor that already exported
    # ``RAPID_MLX_WATCHDOG_PPID=<grandparent_pid>``, ``setdefault``
    # would carry the grandparent's PID into the child env. The
    # watchdog would then compare ``os.getppid()`` (= chat REPL's PID,
    # the IMMEDIATE parent) against the grandparent PID, mismatch on
    # first poll, and self-terminate the freshly-booted server. The
    # spawner owns the watchdog relationship for the spawn it just
    # created — overwrite is correct.
    child_env["RAPID_MLX_WATCHDOG_PPID"] = str(os.getpid())
    # Atomic critical section: block SIGTERM/SIGINT delivery around
    # the whole ``Popen()`` + register + attribute-set + ``release()``
    # sequence. We use ``pthread_sigmask(SIG_BLOCK, ...)`` so the
    # parent thread's mask blocks the signals (queued, delivered when
    # restored).
    #
    # POSIX caveat (codex pr_validate round-3 BLOCKING): both the
    # signal mask AND the signal disposition are inherited across
    # ``fork`` + ``execve``. If we Popen() while the mask blocks
    # SIGTERM/SIGINT, the child server inherits the block and won't
    # honour normal shutdown. The fix is a ``preexec_fn`` that
    # explicitly UNBLOCKS the signals in the child between ``fork``
    # and ``exec`` so the child starts with a clean mask.
    #
    # ``preexec_fn`` runs in the child after fork, before exec, and
    # is exactly the right hook for this. There is no async-signal-
    # safety concern because we are still pre-exec; the child has
    # not yet been replaced with a new image.
    #
    # On platforms without ``pthread_sigmask`` (Windows), fall back
    # to the ``SIG_IGN`` shape — Windows ``subprocess`` doesn't have
    # the same fork/exec model, and the chat REPL is not a Windows
    # feature anyway.
    has_pthread_sigmask = hasattr(_signal, "pthread_sigmask")
    sigset = {_signal.SIGTERM, _signal.SIGINT}
    _prev_mask = None
    _prev_term = _prev_int = None

    def _child_unblock_signals():
        """preexec_fn: clear inherited SIGTERM/SIGINT mask in the child
        so it starts with default mask + default disposition.
        """
        try:
            _signal.pthread_sigmask(_signal.SIG_UNBLOCK, sigset)
        except (ValueError, OSError):
            pass

    try:
        if has_pthread_sigmask:
            try:
                _prev_mask = _signal.pthread_sigmask(_signal.SIG_BLOCK, sigset)
            except (ValueError, OSError):
                _prev_mask = None
        else:
            try:
                _prev_term = _signal.signal(_signal.SIGTERM, _signal.SIG_IGN)
            except (ValueError, OSError):
                pass
            try:
                _prev_int = _signal.signal(_signal.SIGINT, _signal.SIG_IGN)
            except (ValueError, OSError):
                pass
        try:
            proc = subprocess.Popen(  # noqa: S603
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=child_env,
                # Codex pr_validate r3 BLOCKING: clear the inherited
                # signal mask in the child so it can be terminated
                # normally. ``preexec_fn`` is the documented hook for
                # post-fork / pre-exec setup; the child cannot reach
                # ``exec`` until this runs.
                preexec_fn=_child_unblock_signals if has_pthread_sigmask else None,
            )
        except (OSError, ValueError):
            # Popen raised before constructing the child — the log handle
            # would otherwise leak. Re-raise after closing. The ``finally``
            # below still restores the signal mask / handlers.
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
        # Hand the tempfile path off to ``_teardown_proc`` BEFORE we
        # leave the masked section. Once released, the ``with`` block's
        # ``finally`` in the caller is a no-op for this path.
        if log_handle is not None:
            log_handle.release()
    finally:
        # Best-effort restore so post-spawn signals route normally. Any
        # SIGTERM/SIGINT that landed while blocked is delivered HERE
        # (kernel-queued, exactly the desired behaviour: the chat's
        # installed handler now sees the proc in ``_active_procs``).
        if has_pthread_sigmask:
            if _prev_mask is not None:
                try:
                    _signal.pthread_sigmask(_signal.SIG_SETMASK, _prev_mask)
                except (ValueError, OSError):
                    pass
        else:
            for signum, prev in (
                (_signal.SIGTERM, _prev_term),
                (_signal.SIGINT, _prev_int),
            ):
                if prev is not None:
                    try:
                        _signal.signal(signum, prev)
                    except (ValueError, OSError):
                        pass
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


def _accumulate_tool_call_deltas(
    parts: dict[int, dict],
    deltas: list,
) -> None:
    """Merge one chunk's ``delta.tool_calls`` into an index-keyed accumulator.

    Streaming tool calls arrive split across chunks: the first carries ``id``
    and ``function.name``, later ones append ``function.arguments`` fragments
    that are only valid JSON once concatenated. ``index`` — not ``id``, which
    later fragments omit — is the field that ties the fragments together, so
    it is the accumulator key.

    A server that streams several calls interleaves their indices, which is
    why fragments are appended per index rather than to a single buffer.
    """

    import json

    for delta in deltas:
        if not isinstance(delta, dict):
            continue
        # Fall back to positional order for servers that omit ``index`` when
        # only one call is in flight.
        index = delta.get("index")
        if not isinstance(index, int):
            index = 0
        part = parts.setdefault(index, {"id": None, "name": None, "arguments": ""})

        if delta.get("id"):
            part["id"] = str(delta["id"])
        function = delta.get("function")
        if not isinstance(function, dict):
            continue
        if function.get("name"):
            part["name"] = str(function["name"])
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            part["arguments"] += arguments
        elif arguments is not None:
            # Some servers send the whole object once instead of fragments.
            part["arguments"] = json.dumps(arguments, ensure_ascii=False)


def _finalize_tool_calls(parts: dict[int, dict]) -> list[dict]:
    """Render the accumulator into OpenAI-shape tool calls, in index order."""

    finalized = []
    for position, index in enumerate(sorted(parts)):
        part = parts[index]
        if not part["name"]:
            # No function name ever arrived — there is nothing to dispatch,
            # and forwarding it would only produce an "unknown tool" round
            # trip.
            continue
        finalized.append(
            {
                "id": part["id"] or f"call_{position}",
                "type": "function",
                "function": {
                    "name": part["name"],
                    "arguments": part["arguments"] or "{}",
                },
            }
        )
    return finalized


def _stream_chat_response(
    base_url: str,
    payload: dict,
    timeout_s: int,
    metrics: dict | None = None,
    tool_calls: list | None = None,
) -> str:
    """POST /v1/chat/completions with stream=True and print tokens as they
    arrive. Returns the full assistant content (concatenated content deltas).

    Reasoning-content deltas (Qwen3, DeepSeek-R1, etc.) are streamed to stdout
    in dim ANSI so the user sees thinking, but excluded from the returned
    string — chat history stores only the final answer, matching the
    OpenAI-compat split between ``content`` and ``reasoning_content``.

    Interactive terminals show a one-line incremental preview, then replace it
    with one correct Rich Markdown render containing structured headings,
    lists, links, tables, and code. Pipes, CI, dumb terminals, and
    ``NO_COLOR`` retain byte-for-byte plain streaming.

    When *tool_calls* is a list, streamed ``delta.tool_calls`` fragments are
    accumulated and the assembled calls are appended to it. This is what lets
    the MCP agent loop stream: the tool round is no longer a reason to fall
    back to a blocking request.
    """
    import json

    import requests

    from .chat_render import StreamingMarkdownRenderer, supports_rich_output

    DIM = "\x1b[2m"
    RESET = "\x1b[0m"
    MAGENTA = "\x1b[35m"
    is_tty = supports_rich_output(sys.stdout)
    in_reasoning = False
    full = ""
    full_reasoning = ""
    tool_call_parts: dict[int, dict] = {}

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

    with (
        requests.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            stream=True,
            timeout=timeout_s,
        ) as resp,
        StreamingMarkdownRenderer() as renderer,
    ):
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
            if tool_calls is not None:
                streamed_calls = delta.get("tool_calls")
                if isinstance(streamed_calls, list):
                    _accumulate_tool_call_deltas(tool_call_parts, streamed_calls)
            if reasoning:
                full_reasoning += reasoning
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
                    sys.stdout.write(f"{RESET}\n" if is_tty else "\n")
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
                        renderer.write(prefix)
                        full += prefix
                else:
                    renderer.write(piece)
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
    if in_reasoning and is_tty:
        sys.stdout.write(RESET)
        sys.stdout.flush()
    if tool_calls is not None:
        tool_calls.extend(_finalize_tool_calls(tool_call_parts))
    if metrics is not None and full_reasoning:
        metrics["reasoning_content"] = full_reasoning
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


def _complete_chat_with_mcp(
    base_url: str,
    payload: dict,
    mcp_runtime,
    timeout_s: int,
    *,
    max_rounds: int = 8,
    on_tool_event=None,
) -> tuple[str, dict]:
    """Run the chat agent loop with MCP tools, streaming every round.

    vLLM and SGLang use the same loop shape: expose MCP tools as ordinary
    function tools, append the assistant tool call and matching tool output,
    then ask the model again.  MCP transport and execution stay inside the
    chat runtime; the inference server only sees standard Chat Completions
    messages.

    Every round streams, including the ones that end in a tool call. The
    blocking variant this replaced left the screen empty for the whole
    multi-round turn, which on a local model is the slowest part of the
    session and the part the user most needs feedback during.
    """
    import json

    messages = payload["messages"]
    request_payload = {
        **payload,
        "stream": True,
        "stream_options": {"include_usage": True},
        "tools": mcp_runtime.tools,
        "tool_choice": "auto",
    }
    total_usage: dict[str, int | float] = {}

    for round_index in range(max_rounds + 1):
        round_metrics: dict = {}
        streamed_calls: list[dict] = []
        content = _stream_chat_response(
            base_url,
            request_payload,
            timeout_s=timeout_s,
            metrics=round_metrics,
            tool_calls=streamed_calls,
        )

        for name in ("prompt_tokens", "completion_tokens"):
            value = round_metrics.get(name)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                total_usage[name] = total_usage.get(name, 0) + value

        if not streamed_calls:
            metrics = dict(total_usage)
            metrics["finish_reason"] = round_metrics.get("finish_reason")
            return content, metrics
        if round_index == max_rounds:
            # Budget exhausted. The tool results already in ``messages`` are
            # real work — the model read files, ran commands — so raising here
            # would send the caller down ``_recover_failed_chat_turn`` and
            # delete all of it. Return instead, with the partial content the
            # model did produce and a finish_reason the caller can report.
            # The assistant message requesting this round's calls is
            # deliberately not appended: unanswered ``tool_calls`` in history
            # make the next request malformed.
            metrics = dict(total_usage)
            metrics["finish_reason"] = "tool_call_limit"
            metrics["tool_rounds_exhausted"] = max_rounds
            return content, metrics

        normalized_calls = []
        for position, tool_call in enumerate(streamed_calls):
            function = tool_call["function"]
            normalized_calls.append(
                {
                    "id": str(tool_call.get("id") or f"call_{round_index}_{position}"),
                    "type": "function",
                    "function": {
                        "name": str(function.get("name") or ""),
                        "arguments": function.get("arguments") or "{}",
                    },
                }
            )

        assistant_message = {
            "role": "assistant",
            "content": content or None,
            "tool_calls": normalized_calls,
        }
        if round_metrics.get("reasoning_content"):
            assistant_message["reasoning_content"] = round_metrics["reasoning_content"]
        messages.append(assistant_message)
        completed_messages = {}

        def _handle_tool_event(event, completed=completed_messages):
            if event.phase == "finish" and event.message is not None:
                completed[event.call_id] = event.message
            if on_tool_event is not None:
                on_tool_event(event)

        try:
            messages.extend(
                mcp_runtime.execute_tool_calls(
                    normalized_calls,
                    on_event=_handle_tool_event,
                )
            )
        except BaseException:
            for pending_call in normalized_calls:
                messages.append(
                    completed_messages.get(pending_call["id"])
                    or {
                        "role": "tool",
                        "tool_call_id": pending_call["id"],
                        "content": json.dumps(
                            {"error": "Tool execution interrupted"},
                            ensure_ascii=False,
                        ),
                    }
                )
            raise

    # Unreachable: the ``round_index == max_rounds`` branch returns on the
    # final iteration. Kept so the function has no implicit ``None`` return.
    raise AssertionError("MCP tool loop exited without a result")


def _recover_failed_chat_turn(messages: list[dict], turn_start: int) -> None:
    """Keep completed tool side effects in history; otherwise roll back."""

    import json

    tool_succeeded = False
    for message in messages[turn_start + 1 :]:
        if message.get("role") != "tool":
            continue
        try:
            result = json.loads(message.get("content") or "")
        except (TypeError, ValueError):
            continue
        if (
            isinstance(result, dict)
            and "error" not in result
            and result.get("isError") is not True
        ):
            tool_succeeded = True
            break

    if tool_succeeded:
        messages.append(
            {
                "role": "assistant",
                "content": "Tool execution completed, but the follow-up response failed.",
            }
        )
    else:
        del messages[turn_start:]


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

    from vllm_mlx._tempfile_safe import managed_tempfile_path
    from vllm_mlx.chat_render import supports_rich_output, terminal_safe_text

    base_url: str
    proc = None
    log_path: str | None = None
    mcp_runtime = None
    # Tracks every spawned server (initial + every /model candidate) so
    # the SIGTERM/atexit cleanup tears down in-flight candidates too —
    # not just the bound ``proc``. A SIGTERM landing while a /model
    # swap is mid-spawn would otherwise orphan the candidate server.
    _active_procs: list[subprocess.Popen] = []

    # ANSI palette only when the output supports interactive formatting.
    _is_tty = supports_rich_output(sys.stdout)
    BOLD = "\x1b[1m" if _is_tty else ""
    DIM = "\x1b[2m" if _is_tty else ""
    GREEN = "\x1b[32m" if _is_tty else ""
    CYAN = "\x1b[36m" if _is_tty else ""
    YELLOW = "\x1b[33m" if _is_tty else ""
    RED = "\x1b[31m" if _is_tty else ""
    RESET = "\x1b[0m" if _is_tty else ""

    # P0-3 (first-run guide): did the user see real value this session — i.e.
    # at least one completed, non-empty response? Gates the one-time agent-
    # connect tip printed on exit, so a load-then-immediate-/exit run neither
    # nags the user nor burns the one-time marker.
    _generated_any = False

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
            mcp_close_error = None
            if mcp_runtime is not None:
                try:
                    mcp_runtime.close()
                except Exception as exc:
                    mcp_close_error = exc
            for p in list(_active_procs):
                _teardown_proc(p)
            if mcp_close_error is not None:
                raise RuntimeError(
                    f"Failed to close MCP runtime: {mcp_close_error}"
                ) from mcp_close_error
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

    attached_to_existing = bool(args.base_url or args.port is not None)

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

        # GH #719: ``NamedTemporaryFile(...).name`` leaked one zero-byte
        # log per invocation if ANYTHING raised between path creation
        # and the proc being appended to ``_active_procs`` (where
        # ``_teardown_proc`` would otherwise reap it). The
        # ``managed_tempfile_path`` helper registers an atexit unlink
        # the moment the path exists, so the race window is closed:
        # cleanup runs on context exit, on ``sys.exit``, or via atexit
        # if the body propagates. The handle is passed through to
        # ``_spawn_chat_server`` which performs the
        # register/attribute-set/release as a single SIGTERM-masked
        # critical section, so ``_teardown_proc``'s keep-non-empty-log
        # policy cannot be undone by a signal during the handoff
        # (codex round-1 BLOCKING #1).
        with managed_tempfile_path(
            prefix="rapid-mlx-chat-", suffix=".log"
        ) as _log_handle:
            log_path = _log_handle.path
            print(f"\n  Starting server {DIM}(log: {log_path}){RESET} ...")
            # If main() resolved an alias, expose the alias as the API model name
            # so the chat request body matches what the user typed.
            original = getattr(args, "_original_alias", None)
            privacy_kwargs = (
                {"disable_prefix_cache": True}
                if getattr(args, "disable_prefix_cache", False)
                else {}
            )
            proc, base_url = _spawn_chat_server(
                args.model,
                log_path,
                served_name=original,
                register_in=_active_procs,
                log_handle=_log_handle,
                **privacy_kwargs,
            )

        try:
            _wait_for_chat_server(base_url, proc, timeout_s=args.ready_timeout)
        except (RuntimeError, TimeoutError) as e:
            print(f"\n  {RED}Failed to start server:{RESET} {e}")
            sys.exit(1)
        print(f"  {GREEN}✓ Ready.{RESET}\n")

    # When attaching without an explicit model, trust the server's advertised
    # model instead of the client's independently configured starter alias.
    # The latter commonly differs from a manually started server and turns the
    # very first request into an avoidable 404. Direct callers predating this
    # marker are treated as explicit to preserve their existing behavior.
    if attached_to_existing and not getattr(args, "_model_was_explicit", True):
        try:
            import requests

            response = requests.get(f"{base_url}/v1/models", timeout=2)
            response.raise_for_status()
            payload = response.json()
            models = payload.get("data", []) if isinstance(payload, dict) else []
            discovered = next(
                (
                    item.get("id")
                    for item in models
                    if isinstance(item, dict)
                    and isinstance(item.get("id"), str)
                    and item["id"].strip()
                ),
                None,
            )
        except (requests.RequestException, ValueError, TypeError):
            discovered = None
        if discovered:
            args.model = discovered
            if hasattr(args, "_original_alias"):
                delattr(args, "_original_alias")
            print(f"  Connected model: {discovered} (discovered from server)")

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
    # A single blank line before the prompt keeps the layout consistent.
    # The "connect your agent" nudge that used to live here (a start-of-chat
    # agents/codex banner) moved to a one-time tip printed after the user's
    # FIRST successful chat exit — see the P0-3 block at the end of this
    # function and ``vllm_mlx/first_run.py``. Nudging once, after value has
    # landed, beats nudging on every cold start.
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
        #    before we commit. ``managed_tempfile_path`` (GH #719)
        #    guarantees the log path is unlinked if the spawn raises
        #    before the proc is registered onto ``_active_procs`` —
        #    the leak window in the original ``NamedTemporaryFile(...).name``
        #    pattern. The handle is passed into ``_spawn_chat_server``
        #    so the register/attribute-set/release happens under one
        #    SIGTERM/SIGINT mask, preserving ``_teardown_proc``'s
        #    keep-non-empty-log policy on signal-during-handoff (codex
        #    round-1 BLOCKING #1).
        with managed_tempfile_path(
            prefix="rapid-mlx-chat-", suffix=".log"
        ) as _new_log_handle:
            new_log_path = _new_log_handle.path
            print(f"  Starting server {DIM}(log: {new_log_path}){RESET} ...")
            # ``register_in=_active_procs`` makes the candidate visible to
            # ``_cleanup`` *inside* ``_spawn_chat_server`` — before the
            # readiness wait, before any further Python statement runs in
            # this scope. A SIGTERM/Ctrl-C during the (possibly multi-second)
            # load tears the child down via the cleanup walk.
            privacy_kwargs = (
                {"disable_prefix_cache": True}
                if getattr(args, "disable_prefix_cache", False)
                else {}
            )
            new_proc, new_base_url = _spawn_chat_server(
                resolved,
                new_log_path,
                served_name=new_alias,
                register_in=_active_procs,
                log_handle=_new_log_handle,
                **privacy_kwargs,
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

    try:
        if getattr(args, "mcp_config", None):
            from vllm_mlx.chat_mcp import ChatMCPRuntime

            try:
                mcp_runtime = ChatMCPRuntime(args.mcp_config)
            except (ImportError, OSError, RuntimeError, ValueError) as exc:
                print(f"\n  {RED}Failed to start MCP:{RESET} {exc}")
                _cleanup()
                sys.exit(1)
            ready_line = (
                f"  {GREEN}✓ MCP ready:{RESET} "
                f"{len(mcp_runtime.tools)} tool(s) from "
                f"{mcp_runtime.server_count} server(s)."
            )
            if mcp_runtime.server_log_path:
                ready_line += (
                    f"\n  {DIM}server logs → {mcp_runtime.server_log_path}{RESET}"
                )
            print(ready_line)
            for server_name, error in sorted(mcp_runtime.connection_errors.items()):
                print(f"  {YELLOW}MCP server {server_name} unavailable:{RESET} {error}")

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
                        [{"role": "system", "content": args.system}]
                        if args.system
                        else []
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

            turn_start = len(messages)
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
            sys.stdout.write(f"\n  {CYAN}●{RESET}\n")
            sys.stdout.flush()
            metrics: dict = {}
            start_t = time.monotonic()
            try:
                if mcp_runtime is None:
                    assistant = _stream_chat_response(
                        base_url,
                        payload,
                        timeout_s=args.response_timeout,
                        metrics=metrics,
                    )
                else:

                    def _show_mcp_tool_event(event) -> None:
                        display_name = terminal_safe_text(event.name).replace(
                            "__", ".", 1
                        )
                        if event.phase == "start":
                            text = f"{DIM}using {display_name}…{RESET}"
                        else:
                            elapsed = event.elapsed_seconds or 0
                            if event.is_error:
                                text = f"{RED}✗ {display_name} ({elapsed:.2f}s){RESET}"
                            else:
                                text = (
                                    f"{GREEN}✓ {display_name} ({elapsed:.2f}s){RESET}"
                                )
                        sys.stdout.write(f"  {text}\n")
                        sys.stdout.flush()

                    assistant, metrics = _complete_chat_with_mcp(
                        base_url,
                        payload,
                        mcp_runtime,
                        timeout_s=args.response_timeout,
                        max_rounds=getattr(args, "mcp_max_rounds", 8),
                        on_tool_event=_show_mcp_tool_event,
                    )
                    # No re-render here: every round streamed its own tokens
                    # (reasoning included) through the same renderer the
                    # non-MCP path uses, so rendering again would print the
                    # final answer twice.
            except KeyboardInterrupt:
                print(f"\n  {YELLOW}(response interrupted){RESET}\n")
                _recover_failed_chat_turn(messages, turn_start)
                continue
            except RuntimeError as e:
                print(f"\n  {RED}{e}{RESET}\n")
                _recover_failed_chat_turn(messages, turn_start)
                continue
            except requests.RequestException as e:
                # Connection refused, timeout, dropped midstream — keep the REPL
                # alive and roll back the failed user turn so the next request
                # doesn't carry a dangling user role with no assistant reply.
                print(f"\n  {RED}Request failed:{RESET} {e}\n")
                _recover_failed_chat_turn(messages, turn_start)
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
            # The turn stopped because the tool budget ran out, not because the
            # model was done. Say so — the tool results are kept in history, so
            # the user can simply ask it to continue.
            if metrics.get("finish_reason") == "tool_call_limit":
                exhausted = metrics.get("tool_rounds_exhausted")
                print(
                    f"  {YELLOW}(stopped after {exhausted} tool rounds; "
                    f"tool results are kept — ask it to continue, or raise "
                    f"--mcp-max-rounds){RESET}\n"
                )
            if assistant:
                messages.append({"role": "assistant", "content": assistant})
                # A non-empty response reached the user → the session
                # delivered value. Arms the one-time exit tip (P0-3).
                _generated_any = True
            else:
                _recover_failed_chat_turn(messages, turn_start)
    finally:
        # Do not defer MCP shutdown to ``atexit``. The official SDK owns helper
        # threads for stdio sessions, and Python waits for non-daemon threads
        # before running atexit callbacks. Closing here avoids that shutdown
        # ordering deadlock and also tears down a spawned model server promptly.
        _cleanup()

    # P0-3 (first-run guide): after the user's FIRST session that actually
    # produced a response, print a one-line nudge to connect their coding
    # agent — the highest-retention next step, offered only once value has
    # landed. Fires at most once per machine (a marker under ~/.rapid-mlx/),
    # interactive terminals only, and never on a crash (an uncaught exception
    # propagates through ``finally`` before reaching here). Fail-silent.
    if _generated_any and _is_tty:
        try:
            from vllm_mlx.first_run import chat_agent_tip_text, claim_chat_agent_tip

            # Build the tip text (which runs agent detection) BEFORE claiming
            # the marker, so a detection error can't burn the one-time chance
            # without ever showing the tip. The atomic claim is last, so only
            # the process that wins the exclusive-create prints — concurrent
            # first sessions can't both show it.
            _tip = chat_agent_tip_text()
            if claim_chat_agent_tip():
                print(f"\n  {DIM}{_tip}{RESET}")
        except Exception:
            pass


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

    # Download footprint (issue #1286). Manifest-first so known aliases print
    # instantly and offline. Fall back to a live 5s-capped HF probe ONLY for a
    # raw hf_path the registry doesn't carry — a manifest entry that is present
    # but null means "already known to be unresolvable", so we must NOT re-probe
    # it live (that would reintroduce the network wait the manifest exists to
    # avoid). ``~`` marks the value as an estimate.
    from vllm_mlx._download_gate import _format_size, estimate_repo_size_bytes
    from vllm_mlx.model_sizes import is_listed, size_bytes

    dl_size = size_bytes(name)
    if dl_size is None and not is_listed(name):
        dl_size = estimate_repo_size_bytes(name)
    if dl_size is not None:
        print(f"  Download size: ~{_format_size(dl_size)}")
        print()

    # DFlash eligibility — render the report so users can see which
    # gates pass/fail without consulting the docs. Skipped for unknown
    # models since AliasProfile is alias-keyed.
    profile = resolve_profile(original_alias)
    if profile is not None:
        _print_dflash_status(original_alias, profile)
        _print_ddtree_status(original_alias, profile)

    if cfg is None:
        print("  No pattern matched — runtime probe will run when the model loads.")
        print()


def _print_dflash_status(alias: str, profile) -> None:
    """Render a 3-row DFlash status block for ``rapid-mlx info <alias>``.

    Shows each gate (declared support / not MoE / not 4-bit / drafter
    present) so a user who tried DFlash and got a vague error can see
    exactly which gate they're tripping.
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
        print(
            f"  Start with: rapid-mlx serve {alias} "
            """--speculative-config '{"method":"dflash"}'"""
        )
        print()


def _print_ddtree_status(alias: str, profile) -> None:
    """Render DDTree status for ``rapid-mlx info <alias>``."""
    from vllm_mlx.speculative.ddtree.eligibility import have_runtime, report
    from vllm_mlx.speculative.dflash.eligibility import _looks_like_4bit

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
            _yes(profile.supports_ddtree, "yes (supports_ddtree=true)", "no"),
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
                bool(profile.ddtree_draft_model),
                profile.ddtree_draft_model or "yes",
                "no (ddtree_draft_model unset)",
            ),
        ),
        (
            "Spec tokens",
            _yes(
                profile.ddtree_speculative_tokens is not None,
                str(profile.ddtree_speculative_tokens),
                "missing",
            ),
        ),
        (
            "Tree budget",
            _yes(
                profile.ddtree_tree_budget is not None,
                str(profile.ddtree_tree_budget),
                "missing",
            ),
        ),
        (
            "dtree-mlx runtime",
            _yes(
                have_runtime(),
                "installed",
                "missing/import-broken",
            ),
        ),
    ]

    eligible = not r.reasons and have_runtime()
    summary = "✓ eligible" if eligible else "✗ ineligible"
    top = "┌" + "─" * (inner + 2) + "┐"
    bot = "└" + "─" * (inner + 2) + "┘"
    body = [top, _row(f"DDTree eligibility: {summary}"), _row(sep)]
    for k, v in rows:
        body.append(_row(f"{k:<18}: {v}"))
    body.append(bot)
    print("\n".join(body))
    print()
    if eligible:
        print(
            f"  Start with: rapid-mlx serve {alias} "
            """--speculative-config '{"method":"ddtree"}'"""
        )
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
        # Size the name column to the widest alias so a name that meets or
        # exceeds the old hardcoded 15 (e.g. "deepseek-harness", 16 chars)
        # can't eat its own separator space and shift every later column
        # right on that one row. Floor at 15 so short rosters keep the
        # familiar layout.
        name_w = max(15, max((len(p.name) for p in profiles), default=15))
        print()
        print("  Supported AI Agents")
        print(
            f"  {'name':<{name_w}} {'client':<20} {'GitHub':>6}  "
            f"{'tools':<5}  recommended models"
        )
        # Grow the divider in step with the name column so it doesn't fall
        # short of the header once a long alias widens the table.
        print("  " + "─" * (78 + name_w - 15))
        for p in profiles:
            tools = "FC" if p.needs_function_calling else "—"
            stars = f"{p.stars // 1000}K" if p.stars and p.stars >= 1000 else ""
            if p.recommended_models:
                shown = p.recommended_models[:3]
                models = ", ".join(shown)
                if len(p.recommended_models) > 3:
                    models += f" +{len(p.recommended_models) - 3}"
            else:
                models = ""
            print(
                f"  {p.name:<{name_w}} {p.display_name:<20} "
                f"{stars:>6}  {tools:<5}  {models}"
            )
        print("  FC = function calling")
        print()
        # Frameworks (langchain, pydanticai, smolagents) are libraries
        # you build agents with, not agents themselves — count them
        # separately so "N agents" stays honest (#2082).
        framework_count = sum(1 for p in profiles if p.kind == "framework")
        agent_count = len(profiles) - framework_count
        agents_label = "agent" if agent_count == 1 else "agents"
        frameworks_label = "framework" if framework_count == 1 else "frameworks"
        if framework_count:
            print(
                f"  {agent_count} {agents_label} + "
                f"{framework_count} {frameworks_label} supported"
            )
        else:
            print(f"  {agent_count} {agents_label} supported")
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
        from vllm_mlx.agents.adapter import (
            _detect_running_model,
            fetch_context_window,
        )

        # Detect model + context window from running server.
        # Only query the server when the profile template uses
        # {context_length} — avoids a 2-second timeout regression
        # for profiles that don't need it.
        model_id = args.model or "default"
        context_length = None
        cfg = profile.get_config_for_version(args.agent_version)
        needs_ctx = cfg.template and "{context_length}" in cfg.template

        if model_id == "default":
            detected_model, detected_ctx = _detect_running_model(base_url)
            if detected_model:
                model_id = detected_model
            if needs_ctx:
                context_length = detected_ctx
        elif needs_ctx:
            # User specified model — look up *that* model's context window
            context_length = fetch_context_window(base_url, model_id)

        # Claude Code, Continue and DSH have first-class setup flows. They
        # preview an exact diff, require consent, back up existing config,
        # write atomically, and verify the server afterwards. The generic
        # profile writer below still lacks the diff/consent/backup half, but
        # it does honour --dry-run, so a preview never writes on either path.
        if profile.name in {"claude-code", "continue", "deepseek-harness"}:
            from vllm_mlx.agents.setup import (
                apply_setup_plan,
                build_setup_plan,
                confirm_plan,
                verify_server,
            )

            # DSH renders a reasoning-effort control from what we write, so
            # it needs the model's real capability, not a blanket claim.
            # Scoped to the one profile that consumes it — the other
            # first-class flows don't, and this is a second HTTP round trip.
            supports_reasoning = None
            if profile.name == "deepseek-harness":
                from vllm_mlx.agents.adapter import fetch_reasoning_support

                supports_reasoning = fetch_reasoning_support(base_url, model_id)

            try:
                plan = build_setup_plan(
                    profile.name,
                    base_url,
                    model_id,
                    context_length=context_length,
                    supports_reasoning=supports_reasoning,
                )
            except (OSError, ValueError) as exc:
                print(f"\n  {profile.display_name} setup failed: {exc}\n")
                sys.exit(1)

            print(f"\n  {profile.display_name} configuration: {plan.path}")
            if plan.changed:
                print(plan.diff())
            else:
                print("  Already configured; no file changes needed.")

            if args.dry_run:
                print("\n  Dry run only; nothing was written.\n")
                return
            if plan.changed and not args.yes and not confirm_plan(plan):
                print("\n  Setup cancelled; nothing was written.\n")
                return
            if plan.changed:
                try:
                    apply_setup_plan(plan)
                except RuntimeError as exc:
                    print(f"\n  {profile.display_name} setup failed: {exc}\n")
                    sys.exit(1)
                print(f"\n  Configured {profile.display_name} at {plan.path}.")
            if not args.no_check:
                try:
                    advertised = verify_server(base_url, model_id)
                except RuntimeError as exc:
                    status = (
                        "Configuration was saved"
                        if plan.changed
                        else "Configuration is unchanged"
                    )
                    print(f"\n  {status}, but the connection check failed: {exc}\n")
                    sys.exit(1)
                print(f"  Connection check passed (model: {advertised}).")
            print()
            return

        summary = setup_agent_config(
            profile,
            base_url,
            model_id,
            agent_version=args.agent_version,
            context_length=context_length,
            dry_run=args.dry_run,
        )
        if summary.startswith("Cannot"):
            print(f"\n  {profile.display_name} setup failed.")
            print(f"  {summary}")
            print()
            sys.exit(1)
        if args.dry_run:
            print(f"\n  {summary}")
            print("\n  Dry run only; nothing was written.\n")
            return
        print(f"\n  {profile.display_name} configured!")
        print(f"  {summary}")
        print()
        return

    # Default: show setup instructions
    # Pass "default" to trigger auto-detection of running model + context
    model_id = args.model or "default"
    instructions = get_setup_instructions(
        profile,
        base_url,
        model_id,
        agent_version=args.agent_version,
    )
    print()
    print(instructions)
    print()


def connect_command(args):
    """Show server connection info and wire up a tool (SSOT-backed).

    Renders from :mod:`vllm_mlx.connect` — the same source the serve
    lifespan banner uses — so ``ready``/``openai``/``anthropic`` and the
    machine form can never drift from what a running server prints.
    """
    from vllm_mlx.connect import probe_server_alive, render_banner, resolve_endpoints

    eps = resolve_endpoints(host=args.host, port=args.port, model=args.model)

    # Per-target "how to connect" cheat sheet.
    if args.target:
        _connect_target(args, eps)
        return

    if args.json:
        print(eps.to_json())
        return

    # Don't announce "Ready:" for a server that isn't there (#1999): probe the
    # target first so a stopped server reads as "no server", not "warming up".
    running = eps.listen_fd is not None or probe_server_alive(eps.host, eps.port)
    print(render_banner(eps, running=running), end="")


def _connect_target(args, eps):
    """Print the exact command / snippet for ``rapid-mlx connect <target>``.

    The ``claude-code`` / ``continue`` entries point at the first-class safe
    setup flow (`rapid-mlx agents <agent> --setup`) rather than re-implementing
    the config writer here — that keeps a single owner for each tool's config
    shape while ``connect`` stays the one place a user learns "how do I point
    this tool at the server?"
    """
    target = args.target

    if target in {"openai", "openai-python", "python"}:
        model = eps.model or "<detected-model>"
        print()
        print(f"  Python (OpenAI SDK)  →  {eps.openai_url}")
        print()
        print("      pip install openai")
        print()
        print("      from openai import OpenAI")
        print(f"      client = OpenAI(base_url={eps.openai_url!r}, api_key='sk-noop')")
        print("      resp = client.chat.completions.create(")
        print(f"          model={model!r},")
        print('          messages=[{"role": "user", "content": "Hello!"}],')
        print("      )")
        print()
        return

    if target in {"claude", "claude-code"}:
        # The setup command is rendered by ``agents`` via ``--base-url``. All
        # agents CLIs uniformly accept the OpenAI-style ``/v1`` base URL (the
        # adapter strips ``/v1`` for Claude's profile), so we always hand it
        # the OpenAI endpoint — never the bare Anthropic base.
        _print_point_command(
            "Claude Code", "agents claude-code --setup", eps.openai_url
        )
        return
    if target in {"continue", "continue-dev"}:
        _print_point_command("Continue.dev", "agents continue --setup", eps.openai_url)
        return

    print(f"  Unknown connect target: {args.target}")
    print("  Supported: claude-code, continue, openai-python")
    sys.exit(1)


def _print_point_command(app: str, setup_verb: str, url: str) -> None:
    """Print the canonical setup command for a first-class agent target.

    ``setup_verb`` is like ``agents claude-code --setup`` and ``url`` is the
    endpoint that tool should target (OpenAI-style ``/v1`` base). We append
    ``--base-url`` so the suggested command carries the requested host/port —
    otherwise the agent would default to localhost:8000 and write a config
    that silently points at the wrong server.
    """
    print()
    print(f"  {app}  →  {url}")
    print()
    print(f"      rapid-mlx {setup_verb} \\")
    print(f"        --base-url {shlex.quote(url)}")
    print()
    print("  This writes your tool's config to point at the running server")
    print("  (previews a diff, requires consent, verifies the connection).")
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

    if getattr(args, "dry_run", False):
        print("  (dry-run — not executed; rerun without --dry-run to apply.)\n")
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
    from vllm_mlx.telemetry.schema import (
        sample_preview_payload,
        sample_request_preview_payload,
    )
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
        session_sample = sample_preview_payload(
            client_id=cid, rapid_mlx_version=rapid_mlx_version
        )
        request_sample = sample_request_preview_payload(
            client_id=cid, rapid_mlx_version=rapid_mlx_version
        )
        print()
        print("  Sample payloads (this is exactly the shape we send):")
        print()
        print("  session event:")
        print(json.dumps(session_sample.to_dict(), indent=2))
        print()
        print("  request event (per completion, sampled) — only bucketed")
        print("  numbers + booleans; never prompt or response text. The")
        print("  output_degenerate flag is computed locally and sent as a")
        print("  bare true/false (#1250):")
        print(json.dumps(request_sample.to_dict(), indent=2))
        print()
        if not is_enabled(cli_no_telemetry=cli_no):
            print("  Telemetry is currently disabled — nothing is actually sent.")
            print()
        return

    if action == "reset":
        try:
            reset_state()
        except OSError as exc:
            print()
            print(f"  Reset incomplete — some files could not be removed: {exc}")
            print("  Telemetry state may still be present; check ~/.rapid-mlx/.")
            print()
            # Non-zero exit so automation (`rapid-mlx telemetry reset` in a
            # script) sees the failure instead of a false success — state may
            # still be on disk and telemetry may still be enabled.
            sys.exit(1)
        print()
        print("  Removed consent + client-id files. Next interactive run re-prompts.")
        print()
        return

    # Unknown action — argparse choices=[] would have caught this earlier
    # in normal flow; defensive guard for future maintainers.
    print(f"  Unknown telemetry action: {action!r}")
    sys.exit(1)


def _parse_args_with_share_passthrough(
    parser: argparse.ArgumentParser, raw_argv: list[str]
) -> argparse.Namespace:
    """Parse ``raw_argv`` with the fully-registered top-level ``parser``,
    applying ``share``'s ``--`` end-of-options passthrough split.

    ``rapid-mlx share <model> -- <serve flags…>`` forwards everything after
    the literal ``--`` verbatim to the ``rapid-mlx serve`` that ``share``
    spawns (stored on ``args._passthrough``). Splitting on ``--`` up front is
    what keeps a value-taking serve flag such as
    ``--speculative-config '{"method":"mtp"}'`` from having its JSON value
    swallowed by share's required ``model`` positional — the passthrough
    tokens never reach share's parser, so option/value grouping is preserved
    exactly as typed. Every other subcommand (and ``share`` with no ``--``)
    keeps argparse's native behavior, including hard errors on unrecognized
    flags and native ``--`` end-of-options handling.

    Factored out of ``main`` so tests can drive the real parser + split with
    representative argv orderings (see tests/test_share_cli.py) — the crux
    being that ``share`` must not corrupt ``model`` or the passthrough list.
    """
    # A ``--`` is only the passthrough separator when it comes AFTER a
    # COMPLETE ``share <model>`` head. A ``--`` positioned BEFORE the model
    # (``share -- MODEL``) or before the subcommand (``-- share MODEL``) is
    # argparse's native end-of-options marker and must keep its native
    # meaning — splitting there would strip the required positional and break
    # those valid forms. So we only split when the tokens to the left of the
    # first ``--`` STRICTLY parse as ``share`` WITH a model.
    #
    # ``normal`` invocations (no ``--`` at all — the overwhelming majority)
    # skip this block entirely and hit the single ``parse_args`` below, so no
    # converter/action runs twice on the common path.
    if "--" in raw_argv:
        import contextlib
        import io

        sep = raw_argv.index("--")
        head_argv, passthrough_argv = raw_argv[:sep], raw_argv[sep + 1 :]
        # Cheap gate before the probe: the passthrough split only ever applies
        # to ``share``, so only probe when ``share`` is the SELECTED subcommand.
        # The selected subcommand is the first non-option token in the head —
        # the top-level parser's only pre-subcommand options (``--version`` /
        # ``-V`` / ``-h`` / ``--no-telemetry``) are all valueless, so no earlier
        # token can be an option *value* masquerading as the command. Checking
        # the command token structurally (rather than ``"share" in head_argv``)
        # excludes a positional value that merely equals "share" — e.g. a model
        # named "share" after another subcommand (``serve share …``) — so NO
        # non-share invocation is ever parsed twice (which would rerun argparse
        # type converters / custom actions). A ``--`` before the command
        # (``-- share MODEL``, empty head → ``cmd_token is None``) also skips the
        # probe and keeps argparse's native end-of-options meaning.
        cmd_token = next((t for t in head_argv if not t.startswith("-")), None)
        if cmd_token == "share":
            # Strict probe: does the head fully resolve to a ``share`` command
            # with a model? A STRICT ``parse_args`` (not ``parse_known_args``)
            # means an incomplete head (``share`` alone, i.e. ``share -- MODEL``)
            # or a typo'd share flag makes the probe exit with a NON-ZERO code —
            # in which case this ``--`` is NOT a passthrough separator and we
            # fall through to native parsing. stderr is muted so the probe's
            # would-be usage error never reaches the user (the fall-through
            # re-parses and either succeeds or emits the real error itself).
            #
            # A ZERO exit is different: the probe already ran a terminal argparse
            # *action* — ``--help`` / ``--version`` printed to stdout and called
            # ``parser.exit(0)``. Swallowing that would make the fall-through
            # parse print the SAME message a second time, so re-raise instead:
            # the text prints exactly once and the process exits cleanly.
            probed = None
            with contextlib.redirect_stderr(io.StringIO()):
                try:
                    probed = parser.parse_args(head_argv)
                except SystemExit as exc:
                    if exc.code not in (None, 0):
                        probed = None
                    else:
                        raise
            if (
                probed is not None
                and getattr(probed, "command", None) == "share"
                and getattr(probed, "model", None) is not None
            ):
                # Head is a complete share command; the tokens after ``--`` are
                # verbatim serve-flag passthrough. ``probed`` already holds
                # share's authoritative parsed args (model + share flags); the
                # denylist in share.cli then vets the passthrough.
                probed._passthrough = passthrough_argv
                return probed

    # Everything else — non-share commands, ``share`` with no passthrough
    # ``--``, and the ``share -- MODEL`` / ``-- share MODEL`` native forms —
    # keeps argparse's native behavior, including native ``--`` handling and
    # hard errors on unrecognized flags.
    args = parser.parse_args(raw_argv)
    args._passthrough = []
    return args


def _resolve_cli_version() -> str:
    from importlib.metadata import version as pkg_version

    try:
        return pkg_version("rapid-mlx")
    except Exception:
        return "dev"


def build_parser() -> argparse.ArgumentParser:
    """Construct the full CLI parser (extracted from ``main`` so tests
    can assert effective flag defaults on the parsed namespace instead
    of scraping source or help text)."""
    _version = _resolve_cli_version()

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
        "model", nargs="?", type=str, help="Model to serve"
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
    # Disk-streaming MoE weight loading (PRD-rapid-mlx-integration.md).
    # Strictly opt-in: default behavior for every existing invocation is
    # unchanged. When set, the model loads lazily (routed-expert weights
    # never materialized) and vllm_mlx.disk_stream_patch.install() patches
    # its MoE blocks to stream selected experts off disk through a
    # byte-budgeted LRU cache instead of holding them resident — lets an
    # operator run a model whose declared min_memory_gb floor
    # (_check_alias_min_memory above) exceeds this Mac's RAM. Does NOT
    # suppress that warning: resident components (attention, KV cache,
    # dense layers, the cache budget itself) still consume real RAM.
    serve_parser.add_argument(
        "--disk-stream",
        action="store_true",
        default=False,
        help=(
            "Stream MoE routed-expert weights from disk instead of holding "
            "them resident (opt-in). Loads the model lazily and installs "
            "vllm_mlx.disk_stream_patch on every MoE layer before serving "
            "starts. Only architectures registered in vllm_mlx.registry "
            "are supported; an unregistered model_type fails at load time."
        ),
    )
    serve_parser.add_argument(
        "--disk-stream-cache-gb",
        type=positive_finite_float,
        default=1.0,
        help=(
            "Byte budget (GB) for the disk-stream expert LRU cache. Only "
            "used when --disk-stream is set. Default: 1.0 GB, matching "
            "vllm_mlx.expert_cache.ExpertCache's default."
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
        "--prefill-batch-size",
        type=int,
        default=8,
        help=(
            "Max prompts prefilled together in one cold wave (default: 8). "
            "Lower it to cut first-token latency under concurrent cold load — "
            "requests start decoding sooner instead of all sharing one "
            "full-wave prefill — at an aggregate-throughput cost on large MoE "
            "models, where staggered rows carry ragged offsets that push "
            "batched attention onto a slower path (see #1861)."
        ),
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
        "--idle-cache-clear-seconds",
        type=float,
        default=None,
        help=(
            "Clear reusable prefix/KV cache after this many seconds with no "
            "active requests, preserving loaded model weights. 0 disables; "
            "default: RAPID_MLX_IDLE_CACHE_CLEAR_SECONDS or disabled."
        ),
    )
    # #1103: bounded trim-free prefix reuse for "non-trimmable" cache entries.
    # Opt-in: the default 0 keeps the #1075 policy of dropping them at store
    # time. Two model families produce non-trimmable layers and both benefit:
    #   * hybrid recurrent-state (GatedDeltaNet / Mamba MoE) — ArraysCache;
    #   * sliding-window attention (Gemma 4, GPT-OSS) — RotatingKVCache once
    #     the ring has rotated (offset >= sliding_window → is_trimmable False).
    # The store gate and the trim-free fetch paths are class-agnostic — they
    # key off is_trimmable() — so lifting the store drop for N>0 recovers
    # within-conversation prefix reuse for BOTH families (generalized from
    # recurrent-state to sliding-window). NOTE: prefix-EXTENSION hits (stable
    # prefix + new suffix) are served trim-free; an EXACT re-request of a
    # rotated sliding-window prompt instead full-prefills, because the
    # scheduler's trim(1) exact-hit compensation is unavailable on a rotated
    # cache (see Scheduler._resolve_exact_hit_tokens).
    serve_parser.add_argument(
        "--hybrid-cache-entries",
        type=int,
        default=0,
        help=(
            "Retain up to N non-trimmable prefix-cache entries for "
            "prefix-extension reuse (a stable prefix + a new suffix each turn); "
            "0 disables (default: 0). Covers both hybrid recurrent-state "
            "(GatedDeltaNet/Mamba) AND sliding-window (Gemma 4, GPT-OSS) "
            "models. Best for stable-system-prompt / long-context agent "
            "workloads. An identical exact re-request of a rotated "
            "sliding-window prompt falls back to a full prefill (byte-equal to "
            "cold)."
        ),
    )
    # Operator override for the D-METAL-CAP admission projection. The
    # auto-derived figure assumes an UNCOMPRESSED fp16 KV cache — see
    # ``Scheduler._infer_kv_dtype_bytes``, which documents that quantized-KV
    # deployments are not auto-detected and names this knob as the escape
    # hatch. It was reachable only from the Python API, so a CLI user running
    # ``--kv-cache-turboquant`` / ``--kv-cache-quantization`` got an admission
    # projection that ignored the codec entirely and 503'd long prompts the
    # codec would have fit. Default 0 preserves auto-derivation exactly.
    serve_parser.add_argument(
        "--metal-cap-kv-bytes-per-token",
        type=non_negative_int,
        default=0,
        metavar="BYTES",
        help=(
            "Override the per-token KV-cache size the D-METAL-CAP admission "
            "gate projects, in bytes. 0 (default) auto-derives an "
            "architecture-aware fp16 figure. Set this when running a "
            "quantized KV cache (--kv-cache-turboquant / "
            "--kv-cache-quantization), whose real footprint the auto-derived "
            "figure over-estimates — an over-estimate only costs you spurious "
            "503s, but on a memory-tight Mac that is the difference between a "
            "long prompt being served and being rejected. UNDER-setting it "
            "risks the OOM cliff the gate exists to prevent: lower it only to "
            "a value you have measured. Overrides the architecture-aware "
            "estimator wholesale (sliding-window and recurrent terms included)."
        ),
    )
    # Opt-in prompt-deterministic RESPONSE CACHE (exact-match short-circuit).
    # Distinct from the prefix/KV cache above: this returns the ENTIRE stored
    # completion for a completely repeated GREEDY request (temperature==0 or
    # top_k==1), doing zero GPU decode. Default 0 = fully disabled.
    serve_parser.add_argument(
        "--response-cache-entries",
        type=non_negative_int,
        default=0,
        help=(
            "Retain up to N fully-computed deterministic (greedy) chat "
            "responses; a completely repeated request returns the stored "
            "completion verbatim with zero GPU decode. 0 disables (default: 0). "
            "Only temperature==0 / top_k==1 requests are cached — sampled "
            "requests are never short-circuited."
        ),
    )
    serve_parser.add_argument(
        "--no-memory-aware-cache",
        action="store_true",
        help="Disable memory-aware cache, use legacy entry-count based cache",
    )
    # R15-P1 (task #303): radix-tree prefix-cache index. Default ``radix``
    # accelerates lookup and accounts for cross-request prefix dedup on
    # shared-system-prompt workloads. ``hash`` is the legacy bisect path,
    # kept as an escape hatch if a regression is found in production.
    serve_parser.add_argument(
        "--prefix-cache-index",
        type=str,
        default="radix",
        choices=("radix", "hash"),
        help=(
            "Prefix-cache lookup index: 'radix' (default, R15-P1) uses a "
            "token trie for O(prefix_len) lookups and surfaces dedup-bytes-"
            "saved on /metrics; 'hash' falls back to the legacy bisect-over-"
            "sorted-keys path."
        ),
    )
    # KV cache quantization options
    # ``--kv-cache-dtype`` (R15 task #300) is the canonical knob. Default
    # is bf16 (#1853): the R15 int4 default was justified by "4×-smaller
    # KV cuts decode bandwidth proportionally", but the live serve path
    # (QuantizedBatchKVCache, #1197) implements quantization as
    # dequant-on-read — it MATERIALIZES full-precision K/V on every
    # decode step, so per-token cost grows with context instead of
    # shrinking. Measured on qwen3.5-4b, 16k context, N=2 (parity
    # server, disk checkpoints off): bf16 134.6 tok/s, int4 98.2
    # (-27%), int8 86.1 (-36%); at 128-ctx: bf16 167, int4 161,
    # int8 160. The #910 numbers that motivated the int4 default were
    # short-context (292-tok prompt), where the regression is invisible.
    # int4/int8 remain available as explicit opt-ins for
    # memory-constrained hosts (KV is 4×/2× smaller); ``--reasoning``
    # still pins to int8.
    serve_parser.add_argument(
        "--kv-cache-dtype",
        type=str,
        default="bf16",
        choices=["bf16", "int8", "int4"],
        help=(
            "KV cache dtype (R15 #300, default: bf16). int8/int4 shrink the "
            "KV cache 2x/4x for memory-constrained hosts, but the live-cache "
            "dequant-on-read costs O(context) per decode step — measured "
            "-27%% (int4) / -36%% (int8) at 16k context (#1853). "
            "Sliding-window (Gemma 3, GPT-OSS) and MLA (DeepSeek V3+, "
            "Kimi K2.5) models auto-downgrade to bf16. Use --reasoning "
            "for AIME / hard math."
        ),
    )
    serve_parser.add_argument(
        "--reasoning",
        action="store_true",
        default=False,
        help=(
            "Reasoning profile: pins --kv-cache-dtype to int8 regardless of "
            "the dtype flag (sub-4-bit drops -20pt on AIME-class math for "
            "Qwen3 thinking variants)."
        ),
    )
    serve_parser.add_argument(
        "--kv-cache-quantization",
        action="store_true",
        help=(
            "[deprecated alias of --kv-cache-dtype int8] Quantize stored "
            "KV caches to reduce memory (8-bit by default). When both "
            "flags are passed, this one wins for backwards compatibility."
        ),
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
    # TurboQuant KV cache compression (experimental, R15 Phase 4).
    #
    # Accepts an optional mode value:
    #   --kv-cache-turboquant              → V-only legacy (v4)
    #   --kv-cache-turboquant v4           → V-only explicit
    #   --kv-cache-turboquant k8v4         → K-8bit + V-4bit mix (R15 Phase 4)
    #   --kv-cache-turboquant none         → explicit off-switch (overrides
    #                                        alias ``turboquant_tier=k8v4_verified``
    #                                        auto-resolution; see
    #                                        ``resolve_turboquant_mode_default``)
    #
    # The bare-flag form preserves PR #157 backward compatibility. Mode
    # is mutually exclusive with --kv-cache-quantization.
    serve_parser.add_argument(
        "--kv-cache-turboquant",
        nargs="?",
        const="v4",
        default=None,
        choices=["v4", "k8v4", "none"],
        help="Enable TurboQuant KV-cache compression. ``v4`` (default when "
        "the flag is bare) is V-only 3-4 bit Lloyd-Max with K in FP16; "
        "``k8v4`` is the R15 Phase 4 mix — K at 8-bit Walsh-Hadamard + V at "
        "4-bit Lloyd-Max (~4.6x KV compression on dense models); ``none`` "
        "is the explicit off-switch — overrides the alias-driven "
        "``turboquant_tier=k8v4_verified`` default so the operator can A/B "
        "the bare FP16 KV path. Experimental — mutually exclusive with "
        "--kv-cache-quantization.",
    )
    serve_parser.add_argument(
        "--kv-cache-turboquant-bits",
        type=int,
        default=None,
        choices=[3, 4],
        help="V-side bit width for TurboQuant (default: auto-select by head_dim — "
        "3-bit for head_dim>=96, 4-bit for head_dim=64). Ignored when "
        "--kv-cache-turboquant=k8v4 (V is pinned to 4-bit there).",
    )
    serve_parser.add_argument(
        "--kv-cache-turboquant-group-size",
        type=int,
        default=32,
        help="Group size for TurboQuant V-side quantization (default: 32)",
    )
    # R15-P1 (task #296): disk-backed KV checkpointing. 0 (default)
    # disables the feature entirely (no scheduler-hot-path cost, no
    # ~/.cache/rapid-mlx/kv_checkpoints/ directory creation). Opt-in
    # only: each snapshot serializes the full KV cache synchronously on
    # the decode thread — O(context) per boundary, which degraded 16k
    # decode by up to 45% when this defaulted to 256 (#1853). When
    # enabling, use a multiple of 256 to match MLX-LM's KVCache.step and
    # LMCache's external-chunk size so the on-disk shape aligns with the
    # in-memory shape on reload.
    serve_parser.add_argument(
        "--kv-disk-checkpoint-interval",
        type=int,
        default=0,
        help=(
            "Token interval at which the scheduler snapshots KV state to "
            "~/.cache/rapid-mlx/kv_checkpoints/ (R15 #296). 0 (default) "
            "disables. Write-only today: no engine path reloads the "
            "snapshots yet, and each one blocks decode for O(context) — "
            "enable only for external tooling that consumes the files "
            "(#1853). Pairs with the RAPID_MLX_KV_CHECKPOINT_MAX_BYTES "
            "env var (default 20 GiB) for the oldest-first disk-cap "
            "eviction policy."
        ),
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
        "--speculative-config",
        dest="speculative_config",
        default=None,
        help=(
            "vLLM-style speculative decoding JSON config. This frontend "
            "parses method/model/num_speculative_tokens now. DFlash "
            "requires the rapid-mlx[dflash] extra and is available with "
            '\'{"method":"dflash"}\', DDTree with '
            '\'{"method":"ddtree"}\', and MTP with '
            '\'{"method":"mtp","num_speculative_tokens":3,'
            '"disable_auto_k":false}\'. SuffixDecoding is an explicit, '
            "workload-specific flag for high prompt/output-overlap traffic "
            "and is available with "
            '\'{"method":"suffix","num_speculative_tokens":8}\'.'
        ),
    )
    # Hidden deprecated aliases. They are intentionally absent from help;
    # normalization folds them into the same SpeculativeConfig path as
    # --speculative-config so old commands do not revive old implementations.
    serve_parser.add_argument(
        "--enable-dflash",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )
    serve_parser.add_argument(
        "--enable-ddtree",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )
    serve_parser.add_argument(
        "--spec-decode",
        dest="spec_decode",
        choices=["none", "dflash", "mtp"],
        default="none",
        help=argparse.SUPPRESS,
    )
    serve_parser.add_argument(
        "--dflash-drafter-path",
        default="",
        help=argparse.SUPPRESS,
    )
    serve_parser.add_argument(
        "--enable-mtp",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )
    serve_parser.add_argument(
        "--mtp-num-draft-tokens",
        type=int,
        default=1,
        help=argparse.SUPPRESS,
    )
    serve_parser.add_argument(
        "--mtp-optimistic",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )
    serve_parser.add_argument(
        "--mtp-sidecar",
        default=None,
        help=argparse.SUPPRESS,
    )
    serve_parser.add_argument(
        "--mtp-max-k",
        dest="mtp_max_k",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    serve_parser.add_argument(
        "--mtp-disable-auto-k",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )
    serve_parser.add_argument(
        "--suffix-decoding",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )
    serve_parser.add_argument(
        "--suffix-max-draft",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    serve_parser.add_argument(
        "--suffix-max-suffix-len",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    serve_parser.add_argument(
        "--suffix-min-confidence",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    serve_parser.add_argument(
        "--suffix-min-draft-len",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    # Deprecated no-op flags — accepted-but-ignored for backward compat.
    # These once controlled removed engine paths (the single BatchedEngine,
    # legacy KV-bit quant, the --draft-model / --num-draft-tokens speculation
    # frontend, the --specprefill prototype, and the legacy chunked-prefill
    # monkey-patch that mlx-lm 0.31+ made unreachable). The implementations are
    # gone, but the launcher must still PARSE these flags without an argparse
    # hard-fail so existing user launch scripts (and older docs) keep booting.
    # They are consumed-and-discarded: stored on ``args`` but never read. Hidden
    # from --help (argparse.SUPPRESS); slated for removal in a future release.
    serve_parser.add_argument(
        "--continuous-batching",
        action="store_true",
        default=True,
        help=argparse.SUPPRESS,
    )
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
        "--chunked-prefill-tokens",
        type=int,
        default=0,
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
    serve_parser.add_argument(
        "--resident-memory-limit-gb",
        type=float,
        default=0.0,
        help=(
            "Process-wide resident model ceiling in GiB. Loading another model "
            "evicts the least-recently-used idle unpinned model first. 0 disables "
            "the ceiling (default: 0)."
        ),
    )
    serve_parser.add_argument(
        "--resident-model-idle-ttl",
        type=float,
        default=0.0,
        help=(
            "Evict idle unpinned secondary models after this many seconds. "
            "0 disables idle eviction (default: 0)."
        ),
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
    # Task #292: opt-in for ``/v1/audio/*`` routes on a text-only server.
    # The audio-mode boot path (``rapid-mlx serve kokoro`` etc.) auto-
    # enables the routes via the registry hit — this flag is the
    # escape hatch for operators who want the audio router mounted
    # alongside a text engine (e.g. side-car deployments that proxy the
    # audio paths to a separate process).
    serve_parser.add_argument(
        "--enable-audio",
        action="store_true",
        default=False,
        help="Mount the ``/v1/audio/*`` routes even when the loaded model "
        "is text-only. Useful for side-car deployments that proxy audio "
        "requests to a separate process. Audio-capable models "
        "(kokoro / whisper / parakeet / chatterbox / vibevoice / voxcpm) "
        "auto-mount the routes — this flag is only needed on text-mode boots.",
    )
    # Prefill step size
    serve_parser.add_argument(
        "--prefill-step-size",
        type=int,
        default=2048,
        help="Chunk size for prompt prefill processing. Larger values use more memory "
        "but can improve prefill throughput. (default: 2048)",
    )
    serve_parser.add_argument(
        "--vision-min-pixels",
        type=non_negative_int,
        default=0,
        help=(
            "Minimum pixels used by dynamic-resolution VLM image processors. "
            "0 keeps the model default (default: 0)."
        ),
    )
    serve_parser.add_argument(
        "--vision-max-pixels",
        type=non_negative_int,
        default=0,
        help=(
            "Maximum pixels used by dynamic-resolution VLM image processors. "
            "Lower values trade image detail for lower TTFT and memory. "
            "0 keeps the model default (default: 0)."
        ),
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
        "--trusted-hosts",
        type=str,
        nargs="+",
        default=None,
        metavar="HOST",
        help=(
            "OPT-IN Host-header allowlist (DNS-rebinding hardening): only "
            "requests whose Host header matches one of these values are "
            "accepted; everything else gets 400. Off by default so "
            "rapid-mlx share and LAN access keep working. Values may be "
            "space- or comma-separated. Example: --trusted-hosts localhost "
            "127.0.0.1 (also settable via "
            "RAPID_MLX_TRUSTED_HOSTS)."
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
            "DFlash / DDTree) even when AliasProfile says the model supports it. "
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
    serve_parser.add_argument(
        "--relocate-mid-conversation-system",
        action="store_true",
        default=False,
        help=(
            "Keep a mid-conversation system message at its position (folded "
            "into the next user turn) instead of hoisting it into the leading "
            "system block. Preserves the prefix cache for clients that inject "
            "reminders mid-session (Claude Code); OFF by default because the "
            "relocated text carries user authority rather than system "
            "authority."
        ),
    )
    # Multimodal option
    serve_parser.add_argument(
        "--mllm",
        action="store_true",
        help="Force load model as multimodal (vision) even if name doesn't match auto-detection patterns. Also DISABLES the automatic text-only fallback: normally a vision-config checkpoint that ships no usable vision tower auto-degrades to text-only serving (#1187); with --mllm it hard-fails instead so a deliberate demand for the vision lane is never silently downgraded.",
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
    # Embedding input-length controls (issue #1381). Prevents silent
    # 512-token truncation: derive a model-aware limit and make overflow
    # observable / configurable.
    serve_parser.add_argument(
        "--embedding-max-length",
        type=str,
        default="auto",
        metavar="TOKENS",
        help=(
            "Max input length (tokens) for --embedding-model. 'auto' "
            "(default) derives it from the model's declared maximum "
            "(config.max_position_embeddings, else tokenizer.model_max_length); "
            "or pass a positive integer to set a lower operational ceiling. "
            "Inputs above the effective limit are handled per "
            "--embedding-overflow-policy (never truncated silently)."
        ),
    )
    serve_parser.add_argument(
        "--embedding-overflow-policy",
        type=str,
        choices=["truncate", "error"],
        default="truncate",
        help=(
            "How to handle embedding inputs longer than "
            "--embedding-max-length: 'truncate' (default) discards the tail "
            "but logs a warning and increments the "
            "rapid_mlx_embedding_truncations_total metric (never silent); "
            "'error' rejects the request with a 400 carrying the observed "
            "and allowed token counts."
        ),
    )
    # Parent-PID watchdog (rapid-desktop issue #449). When set, the
    # sidecar polls ``os.getppid()`` every 2 s and self-terminates if
    # the parent dies (re-parent to launchd / init on macOS/Linux). The
    # supervisor passes its own PID at spawn so a SIGKILL on the desktop
    # cannot leave a 30 GB orphan holding the model + port. ``0`` /
    # negative / unset disables. The ``RAPID_MLX_WATCHDOG_PPID`` env var
    # is honoured as a fallback when the CLI flag is omitted; the flag
    # wins when both are present.
    serve_parser.add_argument(
        "--watchdog-ppid",
        type=int,
        default=None,
        metavar="PID",
        help=(
            "Self-terminate when the parent with this PID dies (defeats "
            "orphan-sidecar after SIGKILL on the supervisor). Honors "
            "$RAPID_MLX_WATCHDOG_PPID as a fallback. Set to 0 / unset to "
            "disable."
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
    # Disk-streaming MoE weight loading — same opt-in flags as `serve`,
    # see the `serve_parser` registration above for the full rationale.
    bench_parser.add_argument(
        "--disk-stream",
        action="store_true",
        default=False,
        help=(
            "Stream MoE routed-expert weights from disk instead of holding "
            "them resident (opt-in). See `rapid-mlx serve --help`."
        ),
    )
    bench_parser.add_argument(
        "--disk-stream-cache-gb",
        type=positive_finite_float,
        default=1.0,
        help="Byte budget (GB) for the disk-stream expert LRU cache.",
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
    # #1103 codex BLOCKING-2: the bench path reads args.hybrid_cache_entries
    # (see the MemoryCacheConfig assembly above) but the flag was only
    # registered on serve_parser, so `rapid-mlx bench --hybrid-cache-entries N`
    # was rejected and the getattr fell back to 0. Register it here too, with
    # the same semantics/default as serve, so bench honors the knob.
    bench_parser.add_argument(
        "--hybrid-cache-entries",
        type=int,
        default=0,
        help=(
            "Retain up to N hybrid (recurrent-state) prefix-cache entries for "
            "exact/prefix-extension reuse; 0 disables (default: 0). Useful for "
            "stable-system-prompt agent workloads on GatedDeltaNet/Mamba models."
        ),
    )
    # --response-cache-entries is intentionally NOT registered on the bench
    # parser. The prompt-deterministic response cache is a chat/serve feature
    # whose lookup/store logic lives only in the chat route; `rapid-mlx bench`
    # never consumes it, so exposing the flag here would advertise a no-op
    # (and wiring bench to the cache would change its measurement semantics).
    # The flag stays serve-only.
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
            "Run the standardized B=1 community benchmark and submit it to "
            "the community board at rapidmlx.com. Asks for consent first; "
            "declining writes and sends nothing. After consent a local copy "
            "is saved before the upload where the filesystem allows it, so "
            "a failed send is usually recoverable; if the copy cannot be "
            "written you are warned before anything is sent. "
            "Locks every comparability knob; "
            "ignores the freeform --num-prompts / --max-tokens / "
            "--max-num-seqs args."
        ),
    )
    bench_parser.add_argument(
        "--spec-decode",
        type=str,
        default="none",
        choices=["none", "mtp"],
        help=(
            "Speculative-decoding arm for --submit. 'none' (default) is the "
            "baseline. Run the same model twice with a shared --run-group to "
            "put a same-machine A/B on the board."
        ),
    )
    bench_parser.add_argument(
        "--run-group",
        type=str,
        default=None,
        metavar="HEX12",
        help=(
            "12 hex chars linking the arms of one A/B. The board only reports "
            "a speedup for two arms that share this AND ran on one machine; "
            "without it the runs are published as independent rows."
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
            "codex/opencode/qwen-code/hermes/aider), "
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
    models_parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Emit the model list as machine-readable JSON instead of the "
        "human table (stable keys; pairs with --cached). Prefer this over "
        "scraping the text columns.",
    )
    recipe_parser = subparsers.add_parser(
        "recipe", help="Recommend the smart and fast models for this Mac"
    )
    recipe_parser.add_argument(
        "--max-ram",
        type=float,
        default=None,
        metavar="GB",
        help="Use this RAM size instead of auto-detecting the current Mac",
    )
    recipe_parser.add_argument(
        "--json", action="store_true", help="Print the recommendation as JSON"
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
    rm_parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip the confirmation prompt and remove the model immediately.",
    )
    alias_parser = subparsers.add_parser(
        "alias", help="Manage user-owned model aliases"
    )
    alias_subparsers = alias_parser.add_subparsers(
        dest="alias_action", required=True, help="Alias action"
    )
    alias_set = alias_subparsers.add_parser("set", help="Create or replace an alias")
    alias_set.add_argument("name", help="Private alias name")
    alias_set.add_argument("target", help="Built-in alias or Hugging Face repo id")
    alias_remove = alias_subparsers.add_parser("remove", help="Remove an alias mapping")
    alias_remove.add_argument("name", help="Private alias name")
    alias_subparsers.add_parser("list", help="List user alias mappings")
    subparsers.add_parser("ps", help="List running rapid-mlx servers")

    # Upgrade — detect install method and run the right upgrade command
    # ``update`` is exposed as a subparser alias purely for muscle-memory
    # parity (``npm update`` / ``brew update`` / ``claude update`` /
    # ``rustup update`` all spell it "update"); both names route to
    # ``upgrade_command``. argparse reports the user-typed name on
    # ``args.command``, so the dispatch below matches both.
    upgrade_parser = subparsers.add_parser(
        "upgrade",
        aliases=["update"],
        help="Upgrade rapid-mlx to the latest version (brew / pip / install.sh)",
        description=(
            "Upgrade rapid-mlx to the latest version.\n\n"
            "Note: 'rapid-mlx update' is an alias for 'upgrade'."
        ),
    )
    upgrade_parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip the confirmation prompt and run the upgrade immediately.",
    )
    upgrade_parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print the detected install method and the upgrade command, "
            "then exit without running it."
        ),
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
        default=None,
        help="Model alias (e.g. qwen3.5-4b-4bit) or HF repo (org/name). "
        "When omitted, defaults to the qwen3.5-4b-4bit starter — a "
        "dogfood-tested, tool-call-reliable model — downloaded once on first "
        "use. See `vllm_mlx.first_run.select_chat_default`.",
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
    chat_parser.add_argument(
        "--mcp-config",
        type=str,
        default=None,
        help="Path to an MCP config file whose tools are available in this chat",
    )
    chat_parser.add_argument(
        "--mcp-max-rounds",
        type=positive_int,
        default=8,
        help=(
            "Maximum tool-call rounds per turn when --mcp-config is set "
            "(default: 8). Multi-step tasks may need more."
        ),
    )
    chat_parser.add_argument(
        "--disable-prefix-cache",
        action="store_true",
        help=(
            "Disable reusable prefix-cache persistence in the server spawned "
            "by chat, so prompt token IDs are not written to disk. Has no "
            "effect with --port or --base-url; configure that server directly."
        ),
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
        help=(
            "Agent name (e.g. codex, opencode, qwen-code, aider; "
            "continue-dev is accepted for continue). Omit to list all."
        ),
    )
    agents_parser.add_argument(
        "--setup",
        action="store_true",
        help="Auto-configure the agent to point at this server",
    )
    agents_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview setup changes without writing configuration",
    )
    agents_parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Apply setup without an interactive confirmation",
    )
    agents_parser.add_argument(
        "--no-check",
        action="store_true",
        help="Skip the post-write server health and model check",
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

    # Connect command — the single place to learn "the server is up, now
    # point a tool at it." Renders from the same SSOT as the serve banner
    # (:mod:`vllm_mlx.connect`) so ``ready``/``openai``/``anthropic`` and the
    # ``--json`` machine form can never drift from what the server prints.
    connect_parser = subparsers.add_parser(
        "connect",
        help="Show the server's connection info and wire up a tool",
    )
    connect_parser.add_argument(
        "target",
        nargs="?",
        default=None,
        help=(
            "Tool to set up: claude-code, continue, or openai-python. "
            "Omit to print the connection banner."
        ),
    )
    connect_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of the rendered banner",
    )
    connect_parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Server host (default: auto-detect or localhost)",
    )
    connect_parser.add_argument(
        "--port",
        type=_port_arg,
        default=None,
        help="Server port 1-65535 (default: auto-detect or 8000)",
    )
    connect_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name to advertise (default: auto-detect from server)",
    ).completer = alias_completer

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

    return parser


def main():
    parser = build_parser()
    _version = _resolve_cli_version()
    # The subcommand help printer below needs the subparsers action;
    # recover it from the parser rather than keeping it as a shared
    # local across the build/parse split.
    subparsers = next(
        a for a in parser._actions if isinstance(a, argparse._SubParsersAction)
    )

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

    # Systematic serve-flag passthrough for ``share`` via the standard ``--``
    # end-of-options separator — see ``_parse_args_with_share_passthrough``.
    args = _parse_args_with_share_passthrough(parser, sys.argv[1:])
    # A missing required positional normally makes argparse print the entire
    # serve help (dozens of expert flags) before its one actionable error.
    # Keep the positional optional at parse time so this first-run mistake gets
    # a short recovery path. Explicit ``serve --help`` still exits from
    # argparse above and retains the complete reference.
    if getattr(args, "command", None) == "serve" and not args.model:
        print("rapid-mlx serve: a model is required.", file=sys.stderr)
        print("  Pick one for this Mac:  rapid-mlx recipe", file=sys.stderr)
        print(
            "  Or start a small one:   rapid-mlx serve qwen3.5-4b-4bit",
            file=sys.stderr,
        )
        sys.exit(2)
    if getattr(args, "command", None) in ("chat", "run"):
        args._model_was_explicit = getattr(args, "model", None) is not None

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
        # #1272 activation-funnel signals, computed HERE (before dispatch)
        # where the argparse result is available. Both are session metadata,
        # never content.
        #   - first_session: claim the one-time local marker. This block is
        #     already skipped on the ``_just_collected_consent`` run (the
        #     disclosure promises "nothing from before this prompt"), so the
        #     marker is claimed on the first RECORDED session -- exactly once
        #     per client -- not the first-ever binary run. That is the funnel
        #     semantic we want ("first session we recorded from this new
        #     client"); see ``mark_first_session`` for the full rationale
        #     (codex #1273). Runs regardless of telemetry on/off within this
        #     block so a later opt-in still sees the marker already set.
        #   - auto_selected: ``chat`` with no positional model (nargs="?"
        #     default None) is exactly the auto-select-the-starter path
        #     (see ``first_run.select_chat_default``), so the wizard's
        #     contribution to activation is attributable.
        from vllm_mlx.first_run import mark_first_session as _mark_first_session

        _first_session = _mark_first_session()
        _auto_selected = (
            _session_subcommand == "chat" and getattr(args, "model", None) is None
        )
        # Round 19 codex NIT: session_start sees an empty IMMUTABLE
        # snapshot of models_loaded so it does not depend on whether
        # ``emit.session_start()`` eagerly copies its input. The closure-
        # captured list keeps mutating until session_end takes its own
        # snapshot below.
        _telemetry_emit.session_start(
            subcommand=_session_subcommand,
            flag_names=_session_flag_names,
            models_loaded=(),
            first_session=_first_session,
            auto_selected=_auto_selected,
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

    # First-run auto-select: ``chat`` / ``run`` invoked with no model arg.
    # Resolve the starter alias HERE — before the alias→path resolution below —
    # so it flows through the normal resolve + download path (and the
    # ``_original_alias`` banner) exactly as if the user had typed it. Always
    # the known-good starter, never an arbitrary cached model (which could be a
    # non-chat checkpoint); the bare-command nameplate lists cached models for
    # explicit selection instead.
    if (
        getattr(args, "command", None) in ("chat", "run")
        and getattr(args, "model", None) is None
    ):
        from vllm_mlx.first_run import FIRST_RUN_MODEL_SIZE, select_chat_default

        _cmd = getattr(args, "command", "chat")
        _sel_alias, _starter_cached = select_chat_default()
        args.model = _sel_alias
        if sys.stdin.isatty() and not (args.base_url or args.port is not None):
            if _starter_cached:
                print(
                    f"  No model specified — using {_sel_alias} (already downloaded)."
                )
            else:
                print(
                    f"  No model specified — using {_sel_alias} "
                    f"({FIRST_RUN_MODEL_SIZE}, one-time download)."
                )
            print(f"  Browse: rapid-mlx models · Override: rapid-mlx {_cmd} <model>")
            print()
        # We intentionally do NOT special-case the download gate for the
        # auto-selected starter. The starter is a small (~2.5 GB) model, well
        # under the gate's 10 GiB confirm threshold, so the gate never prompts
        # for it regardless — deferring to the gate's own authoritative
        # ``is_repo_cached`` + threshold policy is simpler and safer than a
        # bypass flag derived from a fail-silent pre-scan. Non-interactive
        # (CI / pipe): no notice (it is TTY-only); ``main()`` sets the starter
        # and falls through to the same gate a bare ``rapid-mlx chat`` always
        # used, so scripted callers are unchanged (no new exit-1 path).

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
        from vllm_mlx.model_aliases import RetiredModelAliasError, resolve_model
        from vllm_mlx.user_aliases import UserAliasError

        try:
            resolved = resolve_model(args.model)
        except (RetiredModelAliasError, UserAliasError) as exc:
            print(f"\n  Error: {exc}", file=sys.stderr)
            raise SystemExit(1) from None
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
            else:
                # #991: it IS an audio alias. ``serve`` keeps the short
                # alias for request-time resolution, but ``pull``/``rm``
                # consume ``args.model`` directly and would miss the R2
                # mirror (catalog keyed by ``hf_path``) then 404 at HF.
                # Stamp the concrete HF id up front for those two commands
                # only, mirroring the text-alias banner above.
                _audio_hf_id = _resolve_audio_download_alias(
                    getattr(args, "command", None), args.model
                )
                if _audio_hf_id is not None:
                    print(f"  Alias: {args.model} → {_audio_hf_id}")
                    args._original_alias = args.model
                    args.model = _audio_hf_id
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
                # The size estimate is a silent HF ``model_info`` round-trip
                # (up to 5s). Cover it with a "Resolving…" spinner so the
                # first-run cold start doesn't read as a hang here — the same
                # treatment the download prep gets in ``_ensure_model_
                # downloaded``. The spinner clears BEFORE ``confirm_or_abort``
                # so a genuine confirm prompt (large uncached model) lands on a
                # clean line. We keep the real size-based gate for EVERY model,
                # including the auto-selected starter: it is an unpinned HF repo
                # whose declared size we must actually verify, never assume,
                # before waiving consent.
                _short = args.model.split("/")[-1]
                with _StatusSpinner(f"Resolving {_short} …"):
                    _size = estimate_repo_size_bytes(args.model)
                confirm_or_abort(args.model, _size)
    # --- END B2 --------------------------------------------------------

    if args.command == "serve":
        serve_command(args)
    elif args.command == "bench":
        bench_command(args)
    elif args.command == "models":
        models_command(args)
    elif args.command == "recipe":
        recipe_command(args)
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
            import difflib

            print(f"Unknown subcommand: {target}")
            matches = difflib.get_close_matches(
                target, list(subparsers.choices.keys()), n=3, cutoff=0.6
            )
            if matches:
                print(f"  Did you mean: {', '.join(matches)}?")
            print("Run `rapid-mlx help` for the list of subcommands.")
            sys.exit(1)
    elif args.command == "pull":
        pull_command(args)
    elif args.command == "rm":
        rm_command(args)
    elif args.command == "alias":
        alias_command(args)
    elif args.command == "ps":
        ps_command(args)
    elif args.command in ("upgrade", "update"):
        # ``update`` is exposed as a subparser alias for muscle-memory
        # parity; argparse routes via ``aliases=`` but reports the
        # user-typed name on ``args.command``. Both names land here.
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
    elif args.command == "connect":
        connect_command(args)
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
    elif (
        getattr(args, "command", None) is None
        and sys.stdout.isatty()
        and sys.stdin.isatty()
    ):
        # Bare ``rapid-mlx`` in an interactive terminal = a first-run
        # nameplate (hardware + cached-model hint + "get started" signpost),
        # not a wall of argparse help. Non-blocking: it prints and exits 0.
        # Non-interactive invocations (pipe, redirect, CI) fall through to the
        # unchanged help + exit 1 so scripts parsing ``rapid-mlx`` output are
        # unaffected. Fail-silent: any nameplate error also falls through.
        try:
            from vllm_mlx.first_run import build_nameplate

            print(build_nameplate(_version))
            sys.exit(0)
        except SystemExit:
            raise
        except Exception:
            parser.print_help()
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)


def cli_entrypoint() -> None:
    """Console entry point with normal Unix broken-pipe semantics."""
    try:
        main()
    except BrokenPipeError:
        # Unix consumers commonly stop reading early (for example,
        # ``rapid-mlx models | head``).  Treat EPIPE as normal completion
        # instead of printing a traceback from the CLI entry point.
        try:
            sys.stdout.close()
        finally:
            sys.exit(0)


if __name__ == "__main__":
    cli_entrypoint()
