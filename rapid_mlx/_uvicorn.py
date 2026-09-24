# SPDX-License-Identifier: Apache-2.0
"""Uvicorn startup seam that reports readiness only after socket creation."""

from __future__ import annotations

import importlib
import logging
import socket
import sys
from collections.abc import Callable
from typing import Any

import uvicorn

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
            await super().startup(sockets=sockets)
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
            raise

        if getattr(self, "lifespan", None) is not None and self.lifespan.should_exit:
            # Older Uvicorn releases return instead of raising after a lifespan
            # startup failure. Preserve the same deterministic classification.
            from rapid_mlx.telemetry.server_start import failed

            failed("engine_start")
            return

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


def run_uvicorn(
    app: Any,
    *,
    on_server_accepting: ServerAcceptingCallback | None = None,
    uvicorn_runner: Callable[..., None] | None = None,
    **config_kwargs: Any,
) -> None:
    """Run Uvicorn with its normal runner and the post-bind server subclass."""

    uvicorn_main: Any = importlib.import_module("uvicorn.main")
    original_server = uvicorn_main.Server

    def server_factory(config: uvicorn.Config) -> AcceptingConnectionsServer:
        return AcceptingConnectionsServer(
            config,
            on_server_accepting=on_server_accepting,
        )

    # ``uvicorn.run`` has no server-class injection point. Rapid-MLX starts one
    # server per process, so replacing the constructor for this synchronous
    # call gives us the seam while preserving Uvicorn's fd/UDS, signal, cleanup,
    # and version-specific startup-exit behavior verbatim.
    uvicorn_main.Server = server_factory
    try:
        try:
            (uvicorn_runner or uvicorn.run)(app, **config_kwargs)
        except BaseException:
            from rapid_mlx.telemetry.server_start import failed

            failed("bind")
            raise
    finally:
        uvicorn_main.Server = original_server
