# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
"""Uvicorn startup seam that reports readiness only after socket creation."""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import os
import socket
import sys
from collections.abc import Callable
from typing import Any

import uvicorn
from uvicorn.main import STARTUP_FAILURE
from uvicorn.supervisors import ChangeReload, Multiprocess

ServerAcceptingCallback = Callable[[], None]
logger = logging.getLogger(__name__)


def _port_is_in_use(host: str, port: int) -> bool:
    """Discriminate Uvicorn's bind exit from other startup failures."""

    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    try:
        with socket.socket(family, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind((host, port))
    except OSError as exc:
        return exc.errno in {48, 98}  # macOS/Linux EADDRINUSE
    return False


class _DiagnosedStartupTracebackFilter(logging.Filter):
    """Drop Uvicorn's lifespan traceback once a startup failure was explained."""

    def __init__(self, exc: BaseException) -> None:
        super().__init__()
        self.failure_text = str(exc) or type(exc).__name__
        self.armed = False

    def filter(self, record: logging.LogRecord) -> bool:
        if not self.armed or record.levelno != logging.ERROR:
            return True
        try:
            message = record.getMessage()
        except Exception:
            return True
        if (
            not message.startswith("Traceback (most recent call last):")
            or self.failure_text not in message
        ):
            return True
        self.armed = False
        return False


def suppress_next_lifespan_traceback(exc: BaseException) -> None:
    """Hide the diagnosed lifespan traceback until the next event-loop turn.

    Uvicorn logs the startup failure synchronously as the lifespan unwinds.
    Expire the filter even if an in-process caller catches the exception
    without forwarding it to Uvicorn. Debug logging retains the traceback.
    """
    uvicorn_logger = logging.getLogger("uvicorn.error")
    if uvicorn_logger.isEnabledFor(logging.DEBUG):
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    traceback_filter = _DiagnosedStartupTracebackFilter(exc)
    traceback_filter.armed = True
    uvicorn_logger.addFilter(traceback_filter)
    loop.call_soon(uvicorn_logger.removeFilter, traceback_filter)


class AcceptingConnectionsServer(uvicorn.Server):
    """Run a callback after Uvicorn has created every configured listener."""

    def __init__(
        self,
        config: uvicorn.Config,
        *,
        on_server_accepting: ServerAcceptingCallback | None = None,
    ) -> None:
        super().__init__(config=config)
        self._on_server_accepting = on_server_accepting
        self._accepting_callback_ran = False

    async def startup(self, sockets: list[Any] | None = None) -> None:
        try:
            await self._startup_with_progress(sockets)
        except SystemExit as exc:
            if (
                getattr(self, "lifespan", None) is not None
                and self.lifespan.should_exit
            ):
                # Uvicorn converts any ASGI lifespan startup exception into its
                # startup-failure exit. This is still the pre-bind engine boundary,
                # including warmup after engine.start().
                from rapid_mlx.telemetry.server_start import failed

                failed("engine_start")
            elif (
                self.config.fd is None
                and not self.config.uds
                and _port_is_in_use(self.config.host, self.config.port)
            ):
                host = self.config.host or "0.0.0.0"
                port = self.config.port
                print(
                    f"\n  Error: Port {port} already in use on {host}. "
                    f"Stop the existing server or pass --port <n> "
                    f"(lsof -i :{port}).",
                    file=sys.stderr,
                )
                exc.rapid_mlx_bind_reported = True  # type: ignore[attr-defined]
                # Uvicorn 0.42 changed this path from STARTUP_FAILURE (3) to
                # a generic SystemExit(1). Keep Rapid-MLX's stable process
                # contract for supervisors and the Desktop sidecar.
                exc.code = STARTUP_FAILURE
            raise

        if getattr(self, "lifespan", None) is not None and self.lifespan.should_exit:
            # Uvicorn releases that return here leave ``Server.serve()`` with a
            # successful process status. Preserve a deterministic non-zero
            # startup result for supervisors and the Desktop sidecar.
            from rapid_mlx.telemetry.server_start import failed

            failed("engine_start")
            raise SystemExit(STARTUP_FAILURE)

        listeners = getattr(self, "servers", ())
        listener_created = bool(listeners) and all(
            getattr(server, "sockets", None) for server in listeners
        )
        if self.started and listener_created and not self._accepting_callback_ran:
            self._accepting_callback_ran = True
            from rapid_mlx._signal_observability import ensure_crash_sink
            from rapid_mlx.telemetry.server_start import ready

            ensure_crash_sink()
            ready()
            if self._on_server_accepting is not None:
                try:
                    self._on_server_accepting()
                except Exception:
                    # The listener is already live. A best-effort observer such as
                    # banner output or telemetry must never tear the server down.
                    try:
                        logger.exception("Post-bind server callback failed")
                    except Exception:
                        pass

    async def _startup_with_progress(self, sockets: list[Any] | None) -> None:
        """Keep readiness observable during blocking lifespan model loads.

        The temporary responder shares the eventual listener, so there is no
        unbound-port race at handoff. Specialized transports retain Uvicorn's
        existing startup behavior, including its TLS and inherited-FD handling.
        """
        config = self.config
        lifespan = getattr(self, "lifespan", None)
        if (
            lifespan is None
            or sockets is not None
            or config.fd is not None
            or config.uds
            or config.is_ssl
            or config.workers > 1
            or config.should_reload
        ):
            await super().startup(sockets=sockets)
            return

        # A hostname can resolve to several IPv4/IPv6 listeners. Leave that
        # resolution to Uvicorn rather than silently narrowing its bind scope.
        try:
            ipaddress.ip_address(config.host or "0.0.0.0")
        except ValueError:
            await super().startup(sockets=sockets)
            return

        from ._startup_probe import StartupProbe

        family = socket.AF_INET6 if ":" in (config.host or "") else socket.AF_INET
        listener = socket.socket(family, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if family == socket.AF_INET6:
            # Match asyncio.create_server: an explicit IPv6 listener must not
            # silently accept IPv4 connections or reserve the IPv4 port too.
            listener.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
        probe = None
        original_startup = lifespan.startup
        try:
            try:
                listener.bind((config.host or "0.0.0.0", config.port))
                listener.listen(config.backlog)
            except OSError as exc:
                logger.error(exc)
                raise SystemExit(STARTUP_FAILURE) from exc
            probe = StartupProbe(listener)

            async def startup_then_handoff() -> None:
                try:
                    await original_startup()
                finally:
                    probe.close()

            lifespan.startup = startup_then_handoff
            await super().startup(sockets=[listener])
            if self.started:
                self._log_started_message([listener])
        finally:
            lifespan.startup = original_startup
            if probe is not None:
                probe.close()
            # On success the asyncio server owns the listener. On failure no
            # event-loop transport exists to release this reserved descriptor.
            if not self.started:
                listener.close()


def _run_server(
    app: Any,
    *,
    on_server_accepting: ServerAcceptingCallback | None = None,
    **config_kwargs: Any,
) -> None:
    """Use Uvicorn's runner lifecycle without replacing its public Server class."""
    # Adapted from Uvicorn 0.41.0 main.run; BSD-3-Clause notice in NOTICE.
    app_dir = config_kwargs.pop("app_dir", None)
    if app_dir is not None:
        sys.path.insert(0, app_dir)
    config = uvicorn.Config(app, **config_kwargs)
    server = AcceptingConnectionsServer(config, on_server_accepting=on_server_accepting)
    if (config.reload or config.workers > 1) and not isinstance(app, str):
        logger.warning(
            "You must pass the application as an import string to enable 'reload' or 'workers'."
        )
        raise SystemExit(1)

    try:
        if config.should_reload:
            sock = config.bind_socket()
            ChangeReload(config, target=server.run, sockets=[sock]).run()
        elif config.workers > 1:
            sock = config.bind_socket()
            Multiprocess(config, target=server.run, sockets=[sock]).run()
        else:
            server.run()
    except KeyboardInterrupt:
        pass
    finally:
        if config.uds and os.path.exists(config.uds):
            os.remove(config.uds)

    if not server.started and not config.should_reload and config.workers == 1:
        raise SystemExit(STARTUP_FAILURE)


def run_uvicorn(
    app: Any,
    *,
    on_server_accepting: ServerAcceptingCallback | None = None,
    uvicorn_runner: Callable[..., None] | None = None,
    port_explicit: bool | None = None,
    **config_kwargs: Any,
) -> None:
    """Run the post-bind server without mutating Uvicorn's module globals.

    Lazy imports during lifespan startup (including MCP's SSE dependency)
    must see the real ``uvicorn.main.Server`` and its signal handler.
    ``uvicorn_runner`` is an explicit test seam; production uses _run_server.
    """
    try:
        if uvicorn_runner is None:
            _run_server(app, on_server_accepting=on_server_accepting, **config_kwargs)
        else:
            # Custom runners retain uvicorn.run's keyword contract. The
            # post-bind callback belongs only to our server implementation.
            uvicorn_runner(app, **config_kwargs)
    except BaseException:
        from rapid_mlx.telemetry.server_start import failed

        failed("bind", port_explicit=port_explicit)
        raise
