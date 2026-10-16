# SPDX-License-Identifier: Apache-2.0
"""Owned, loopback-only relay checks for upgrade auth and diagnostic privacy."""

import asyncio
import json
import logging
from unittest.mock import patch

import pytest
from websockets.asyncio.client import connect as modern_connect
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosedError
from websockets.legacy.client import connect as legacy_connect

from rapid_mlx.share import ws_tunnel


@pytest.mark.parametrize("connect", [modern_connect, legacy_connect])
@pytest.mark.parametrize("pool", [False, True])
def test_keyed_upgrade_and_close_keep_credentials_out_of_debug_logs(
    connect, pool, caplog
):
    key = "pool-test-secret" if pool else "ab" * 24
    local_key = "local-test-secret" if pool else None
    seen = []

    async def run():
        async def relay(ws):
            seen.append((ws.request.path, ws.request.headers["Authorization"]))
            assert json.loads(await ws.recv()) == {"t": "ready", "v": 1}
            # Even a peer echoing credentials into close/error text is untrusted.
            await ws.close(code=1008, reason=f"echo {key} {local_key or ''}")

        async with serve(
            relay, "127.0.0.1", 0, logger=logging.Logger("test-relay", logging.CRITICAL)
        ) as server:
            port = server.sockets[0].getsockname()[1]
            client = ws_tunnel.TunnelClient(
                local_port=1,
                relay_url=f"ws://127.0.0.1:{port}/up",
                share_key=key,
                override_authorization=local_key,
                inject_stream_usage=pool,
            )
            with (
                patch.object(ws_tunnel.websockets, "connect", connect),
                pytest.raises(ConnectionClosedError),
            ):
                await client.run()
            assert client.close_code == 1008
            assert key not in client.public_url
            assert client.closed_event.is_set()

    with caplog.at_level(logging.DEBUG, logger="websockets.client"):
        asyncio.run(run())
    assert seen[0][1] == f"Bearer {key}"
    assert key not in seen[0][0]
    assert "?id=" in seen[0][0]
    assert "Authorization" in caplog.text  # debug handshake actually ran
    assert key not in caplog.text
    if local_key:
        assert local_key not in caplog.text


def test_tunnel_logger_redacts_exception_tracebacks_without_global_filters(caplog):
    key = "test-handshake-secret"
    client = ws_tunnel.TunnelClient(local_port=1, share_key=key)
    with caplog.at_level(logging.DEBUG, logger="websockets.client"):
        try:
            raise RuntimeError(f"rejected {key}")
        except RuntimeError:
            client._ws_logger.exception("handshake rejected %s", key)
        logging.getLogger("websockets.client").debug("unrelated connection intact")
    assert key not in caplog.text
    assert "RuntimeError: rejected [redacted]" in caplog.text
    assert "handshake rejected [redacted]" in caplog.text
    assert "unrelated connection intact" in caplog.text
    assert not logging.getLogger("websockets.client").filters


def test_tunnel_logger_redacts_overlapping_credentials_longest_first(caplog):
    client = ws_tunnel.TunnelClient(
        local_port=1,
        share_key="shared-secret",
        override_authorization="shared-secret-tail",
    )
    with caplog.at_level(logging.DEBUG, logger="websockets.client"):
        client._ws_logger.debug("%s %s", "shared-secret", "shared-secret-tail")
    assert "shared-secret" not in caplog.text
    assert "-tail" not in caplog.text
    assert "[redacted] [redacted]" in caplog.text


def test_tunnel_logger_redacts_preformatted_exception_and_stack_text(caplog):
    key = "test-stack-secret"
    client = ws_tunnel.TunnelClient(local_port=1, share_key=key)
    record = logging.LogRecord(
        "websockets.client",
        logging.ERROR,
        __file__,
        1,
        "connection diagnostics",
        (),
        None,
        sinfo=f"stack includes {key}",
    )
    record.exc_text = f"cached exception includes {key}"
    with caplog.at_level(logging.ERROR, logger="websockets.client"):
        client._ws_logger.handle(record)
    assert key not in caplog.text
    assert "stack includes [redacted]" in caplog.text
    assert "cached exception includes [redacted]" in caplog.text
