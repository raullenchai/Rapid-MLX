# SPDX-License-Identifier: Apache-2.0
"""Regression: the "Ready:" banner must print only AFTER the listener binds.

Persona A's 16 GB Air onboarding (v0.6.51) found that the banner printed
~6 s before uvicorn actually bound the port, so a user who curled
immediately got connection-refused while GatedDeltaNet kernels compiled.
The CLI prints a "Starting server …" line up-front, stashes bind host/port on
ServerConfig, and the Uvicorn startup seam prints the real banner only after
both lifespan readiness and listener creation complete.

Since the banner is now rendered by the connect SSOT (:mod:`rapid_mlx.connect`),
these tests drive the real lifespan path and assert against the SSOT's output
shape (Ready / OpenAI / Anthropic / Connect), keeping the "banner only after
ready" timing invariant at the source level.
"""

from __future__ import annotations

import inspect
import io
from contextlib import redirect_stdout

import pytest

import rapid_mlx.server as server
from rapid_mlx.config import get_config


@pytest.fixture(autouse=True)
def _isolate_server_state():
    """Snapshot+restore the engine handle and bind fields each test."""
    cfg = get_config()
    saved = (
        server._engine,
        cfg.bind_host,
        cfg.bind_port,
        cfg.bind_listen_fd,
        cfg.ready,
    )
    server._engine = None
    cfg.bind_host = None
    cfg.bind_port = None
    cfg.bind_listen_fd = None
    cfg.ready = False
    yield
    (
        server._engine,
        cfg.bind_host,
        cfg.bind_port,
        cfg.bind_listen_fd,
        cfg.ready,
    ) = saved


async def _enter_then_exit_lifespan() -> str:
    """Drive the lifespan generator through startup, capture stdout, exit."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        agen = server.lifespan(server.app)
        await agen.__anext__()  # startup phase — runs through the yield
        try:
            await agen.__anext__()  # shutdown phase
        except StopAsyncIteration:
            pass
    return buf.getvalue()


async def test_lifespan_does_not_emit_ready_banner_before_bind():
    """Lifespan may mark routes ready, but it must not announce the socket."""
    cfg = get_config()
    cfg.bind_host = "localhost"
    cfg.bind_port = 8765
    cfg.model_alias = "qwen3.6-35b-4bit"

    out = await _enter_then_exit_lifespan()

    assert "Ready:" not in out


def test_ready_banner_emitted_by_post_bind_callback():
    """The startup seam's callback renders the existing SSOT banner shape."""
    cfg = get_config()
    cfg.bind_host = "localhost"
    cfg.bind_port = 8765
    cfg.model_alias = "qwen3.6-35b-4bit"

    buf = io.StringIO()
    with redirect_stdout(buf):
        server.print_ready_banner()
    out = buf.getvalue()

    # Ready is the base URL (no /v1); OpenAI appends /v1, Anthropic does not.
    assert "Ready: http://localhost:8765" in out
    assert "OpenAI:    http://localhost:8765/v1" in out
    assert "Anthropic: http://localhost:8765" in out
    assert "Model:     qwen3.6-35b-4bit" in out
    # Connect section mirrors what a user runs to wire up each tool. The
    # setup commands must carry the real endpoint (--base-url) so a user who
    # copies them connects to the running server, not the localhost default.
    assert (
        "rapid-mlx agents claude-code --setup "
        "--base-url http://localhost:8765/v1" in out
    )
    assert (
        "rapid-mlx agents continue --setup --base-url http://localhost:8765/v1" in out
    )
    assert "rapid-mlx connect openai-python" in out


def test_ready_banner_shows_served_model_name_when_overridden(monkeypatch):
    """Issue #2353: when ``--served-model-name`` is in effect, the banner's
    ``Model:`` line must show the copyable served API name, not the catalog
    alias."""
    cfg = get_config()
    cfg.bind_host = "localhost"
    cfg.bind_port = 8766
    cfg.model_alias = "lfm2.5-1b-4bit"  # the typed alias
    cfg.model_name = "studio-assistant"  # --served-model-name override
    monkeypatch.setattr(server, "_served_model_name_set", True)

    buf = io.StringIO()
    with redirect_stdout(buf):
        server.print_ready_banner()
    out = buf.getvalue()

    assert "Model:     studio-assistant" in out, out
    # The alias is not what the API serves — it must not lead the Model line.
    assert "Model:     lfm2.5-1b-4bit" not in out, out


async def test_ready_banner_suppressed_when_no_bind_info():
    """Embedded usage (uvicorn owned elsewhere) leaves bind_* unset — silent."""
    cfg = get_config()
    cfg.bind_host = None
    cfg.bind_port = None

    out = await _enter_then_exit_lifespan()

    assert "Ready:" not in out
    assert "OpenAI:" not in out


def test_ready_banner_uses_displayed_host_not_zero_bind():
    """CLI translates 0.0.0.0 → localhost before stashing, so the banner
    shows a URL a user can actually curl."""
    cfg = get_config()
    cfg.bind_host = "localhost"  # CLI maps 0.0.0.0 → localhost up-front
    cfg.bind_port = 9999

    buf = io.StringIO()
    with redirect_stdout(buf):
        server.print_ready_banner()
    out = buf.getvalue()

    assert "http://localhost:9999/v1" in out
    assert "0.0.0.0" not in out


def test_lifespan_sets_internal_readiness_without_rendering_banner():
    src = inspect.getsource(server.lifespan)
    assert "_cfg.ready = True" in src
    assert "render_banner" not in src
