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
import logging
from contextlib import redirect_stdout

import pytest

import rapid_mlx.server as server
from rapid_mlx.config import get_config


@pytest.fixture(autouse=True)
def _isolate_server_state(monkeypatch, tmp_path):
    """Snapshot+restore the engine handle and bind fields each test."""
    cfg = get_config()
    monkeypatch.setenv("RAPID_MLX_STATE_DIR", str(tmp_path / "state"))
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
    assert "rapid-mlx launch codex" in out
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


def test_interactive_ready_banner_ends_with_copyable_next(monkeypatch, capsys):
    cfg = get_config()
    cfg.bind_host = "localhost"
    cfg.bind_port = 8766
    cfg.model_alias = "lfm2.5-1b-4bit"
    cfg.model_name = "studio-assistant"
    cfg.api_key = None
    monkeypatch.setattr(server, "_served_model_name_set", True)
    monkeypatch.setattr(logging.getLogger(), "getEffectiveLevel", lambda: logging.INFO)
    monkeypatch.setattr(server.sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr(
        "rapid_mlx.runtime.optional_runtime._running_in_desktop_sidecar", lambda: False
    )
    server.print_ready_banner()
    out = capsys.readouterr().out
    assert out.index("Ready: http://localhost:8766") < out.index("Next:")
    assert "studio-assistant" in out
    assert (
        "OpenAI SDK base_url: http://localhost:8766/v1; model: studio-assistant" in out
    )
    assert "from openai import OpenAI" in out


def test_interactive_next_step_only_once_per_install(monkeypatch, capsys):
    cfg = get_config()
    cfg.bind_host = "localhost"
    cfg.bind_port = 8766
    cfg.model_alias = "lfm2.5-1b-4bit"
    monkeypatch.setattr(logging.getLogger(), "getEffectiveLevel", lambda: logging.INFO)
    monkeypatch.setattr(server.sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr(
        "rapid_mlx.runtime.optional_runtime._running_in_desktop_sidecar", lambda: False
    )
    server.print_ready_banner()
    assert "Next:" in capsys.readouterr().out
    server.print_ready_banner()
    out = capsys.readouterr().out
    assert "Ready:" in out
    assert "Next:" not in out


def test_interactive_ready_banner_keeps_image_guidance_before_next(monkeypatch, capsys):
    cfg = get_config()
    cfg.bind_host = "localhost"
    cfg.bind_port = 8766
    cfg.model_alias = "lfm2.5-1b-4bit"
    monkeypatch.setattr(logging.getLogger(), "getEffectiveLevel", lambda: logging.INFO)
    monkeypatch.setattr(server.sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr(
        server, "_text_lane_image_note", lambda _cfg: "use a vision model"
    )
    monkeypatch.setattr(
        "rapid_mlx.runtime.optional_runtime._running_in_desktop_sidecar", lambda: False
    )
    server.print_ready_banner()
    out = capsys.readouterr().out
    assert out.index("Images: off. use a vision model") < out.index("Next:")


def test_interactive_ready_banner_uses_configured_key(monkeypatch, capsys):
    cfg = get_config()
    cfg.bind_host = "localhost"
    cfg.bind_port = 8766
    cfg.model_alias = "lfm2.5-1b-4bit"
    monkeypatch.setattr(cfg, "api_key", "test-key")
    monkeypatch.setattr(logging.getLogger(), "getEffectiveLevel", lambda: logging.INFO)
    monkeypatch.setattr(server.sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr(
        "rapid_mlx.runtime.optional_runtime._running_in_desktop_sidecar", lambda: False
    )
    server.print_ready_banner()
    out = capsys.readouterr().out
    assert "Set RAPID_MLX_API_KEY to the server key first." in out
    assert '-H "Authorization: Bearer $RAPID_MLX_API_KEY"' in out
    assert "test-key" not in out


def test_interactive_ready_banner_resolves_agents_before_ready(monkeypatch, capsys):
    import rapid_mlx.agents as agents

    cfg = get_config()
    cfg.bind_host = "localhost"
    cfg.bind_port = 8766
    cfg.model_alias = "lfm2.5-1b-4bit"
    monkeypatch.setattr(logging.getLogger(), "getEffectiveLevel", lambda: logging.INFO)
    monkeypatch.setattr(server.sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr(
        "rapid_mlx.runtime.optional_runtime._running_in_desktop_sidecar", lambda: False
    )
    original_get_profile = agents.get_profile
    first_call = True

    def get_profile(name):
        nonlocal first_call
        if first_call:
            print("INFO:rapid_mlx.agents:Loaded 17 agent profiles")
            first_call = False
        return original_get_profile(name)

    monkeypatch.setattr(agents, "get_profile", get_profile)
    server.print_ready_banner()
    out = capsys.readouterr().out
    assert out.index("Loaded 17 agent profiles") < out.index("Ready:")
    assert out.index("Ready:") < out.index("Next:")


def test_warning_log_level_does_not_add_next(monkeypatch, capsys):
    cfg = get_config()
    cfg.bind_host = "localhost"
    cfg.bind_port = 8766
    cfg.model_alias = "lfm2.5-1b-4bit"
    monkeypatch.setattr(
        logging.getLogger(), "getEffectiveLevel", lambda: logging.WARNING
    )
    monkeypatch.setattr(server.sys.stdout, "isatty", lambda: True)
    server.print_ready_banner()
    out = capsys.readouterr().out
    assert "Connect:" in out
    assert "Next:" not in out


def test_desktop_ready_banner_preserves_existing_shape(monkeypatch, capsys):
    cfg = get_config()
    cfg.bind_host = "localhost"
    cfg.bind_port = 8766
    cfg.model_alias = "lfm2.5-1b-4bit"
    monkeypatch.setattr(logging.getLogger(), "getEffectiveLevel", lambda: logging.INFO)
    monkeypatch.setattr(server.sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr(
        "rapid_mlx.runtime.optional_runtime._running_in_desktop_sidecar", lambda: True
    )
    server.print_ready_banner()
    out = capsys.readouterr().out
    assert "Connect:" in out
    assert "Next:" not in out


def test_audio_ready_banner_has_no_chat_next(monkeypatch, capsys):
    cfg = get_config()
    cfg.bind_host = "localhost"
    cfg.bind_port = 8766
    cfg.model_alias = "kokoro"
    monkeypatch.setattr(logging.getLogger(), "getEffectiveLevel", lambda: logging.INFO)
    monkeypatch.setattr(server.sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr(
        "rapid_mlx.runtime.optional_runtime._running_in_desktop_sidecar", lambda: False
    )
    server.print_ready_banner()
    out = capsys.readouterr().out
    assert "Ready: http://localhost:8766" in out
    assert "Next:" not in out


@pytest.mark.parametrize(
    ("embedding_engine", "rerank_engine"),
    [(object(), None), (None, object()), (object(), object())],
    ids=["embedding", "rerank", "embedding+rerank"],
)
def test_retrieval_only_ready_banner_has_no_chat_next(
    monkeypatch, capsys, embedding_engine, rerank_engine
):
    cfg = get_config()
    cfg.bind_host = "localhost"
    cfg.bind_port = 8766
    monkeypatch.setattr(cfg, "model_name", None)
    monkeypatch.setattr(cfg, "model_alias", None)
    monkeypatch.setattr(cfg, "embedding_engine", embedding_engine)
    monkeypatch.setattr(cfg, "rerank_engine", rerank_engine)
    monkeypatch.setattr(logging.getLogger(), "getEffectiveLevel", lambda: logging.INFO)
    monkeypatch.setattr(server.sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr(
        "rapid_mlx.runtime.optional_runtime._running_in_desktop_sidecar", lambda: False
    )
    server.print_ready_banner()
    out = capsys.readouterr().out
    assert "Ready: http://localhost:8766" in out
    assert "Next:" not in out


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
