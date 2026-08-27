# SPDX-License-Identifier: Apache-2.0
"""Regression: the "Ready:" banner must print only AFTER warmup completes.

Persona A's 16 GB Air onboarding (v0.6.51) found that the banner printed
~6 s before uvicorn actually bound the port, so a user who curled
immediately got connection-refused while GatedDeltaNet kernels compiled.
The CLI now prints a "Starting server …" line up-front, stashes bind
host/port on ServerConfig, and defers the real "Ready:" banner to the
lifespan hook — fires only after `get_config().ready = True`.

Since the banner is now rendered by the connect SSOT (:mod:`vllm_mlx.connect`),
these tests drive the real lifespan path and assert against the SSOT's output
shape (Ready / OpenAI / Anthropic / Connect), keeping the "banner only after
ready" timing invariant at the source level.
"""

from __future__ import annotations

import inspect
import io
from contextlib import redirect_stdout

import pytest

import vllm_mlx.server as server
from vllm_mlx.config import get_config


@pytest.fixture(autouse=True)
def _isolate_server_state():
    """Snapshot+restore the engine handle and bind fields each test."""
    cfg = get_config()
    saved = (
        server._engine,
        cfg.bind_host,
        cfg.bind_port,
        cfg.ready,
    )
    server._engine = None
    cfg.bind_host = None
    cfg.bind_port = None
    cfg.ready = False
    yield
    server._engine, cfg.bind_host, cfg.bind_port, cfg.ready = saved


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


async def test_ready_banner_emitted_when_bind_fields_set():
    """With bind_host/bind_port stashed by CLI, the lifespan prints the full
    Ready / OpenAI / Anthropic / Connect banner from the SSOT."""
    cfg = get_config()
    cfg.bind_host = "localhost"
    cfg.bind_port = 8765
    cfg.model_alias = "qwen3.6-35b-4bit"

    out = await _enter_then_exit_lifespan()

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
    # `ready` flag must be flipped before the banner so /health/ready and
    # the banner agree on the moment of readiness.
    assert cfg.ready is False  # reset on shutdown (second next())


async def test_ready_banner_shows_served_model_name_when_overridden(monkeypatch):
    """Issue #2353: when ``--served-model-name`` is in effect, the banner's
    ``Model:`` line must show the copyable served API name, not the catalog
    alias."""
    cfg = get_config()
    cfg.bind_host = "localhost"
    cfg.bind_port = 8766
    cfg.model_alias = "lfm2.5-1b-4bit"  # the typed alias
    cfg.model_name = "studio-assistant"  # --served-model-name override
    monkeypatch.setattr(server, "_served_model_name_set", True)

    out = await _enter_then_exit_lifespan()

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


async def test_ready_banner_uses_displayed_host_not_zero_bind():
    """CLI translates 0.0.0.0 → localhost before stashing, so the banner
    shows a URL a user can actually curl."""
    cfg = get_config()
    cfg.bind_host = "localhost"  # CLI maps 0.0.0.0 → localhost up-front
    cfg.bind_port = 9999

    out = await _enter_then_exit_lifespan()

    assert "http://localhost:9999/v1" in out
    assert "0.0.0.0" not in out


async def test_ready_banner_fires_after_ready_flag_flip():
    """The banner must come AFTER `get_config().ready = True` in the
    lifespan body — otherwise a client racing /health/ready could see
    the banner before the readiness flag is honored. This is a source-level
    invariant; verifying via execution order is impractical, so assert on
    the source (the SSOT `render_banner` call site).
    """
    src = inspect.getsource(server.lifespan)
    ready_flip_idx = src.index("_cfg.ready = True")
    banner_idx = src.index("render_banner(_ep)")
    assert ready_flip_idx < banner_idx, (
        "Ready banner must print AFTER the readiness flag is set "
        "so the banner and /health/ready agree on the moment of readiness."
    )
