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
    monkeypatch.setattr("shutil.which", lambda _name: None)
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
    assert "OpenAI-compatible base URL: http://localhost:8766/v1" in out
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
    assert 'environ["RAPID_MLX_API_KEY"]' in out
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


@pytest.mark.parametrize(
    "tty,level", [(True, logging.INFO), (False, logging.INFO), (True, logging.WARNING)]
)
def test_onboarding_shown_event_only_for_visible_once_tip(
    monkeypatch, capsys, tmp_path, tty, level
):
    cfg = get_config()
    cfg.bind_host = "localhost"
    cfg.bind_port = 8766
    cfg.model_alias = "lfm2.5-1b-4bit"
    monkeypatch.setattr(logging.getLogger(), "getEffectiveLevel", lambda: level)
    monkeypatch.setattr(server.sys.stdout, "isatty", lambda: tty)
    monkeypatch.setattr(
        "rapid_mlx.runtime.optional_runtime._running_in_desktop_sidecar", lambda: False
    )
    events = []
    monkeypatch.setattr(
        "rapid_mlx.onboarding.track_next_step", lambda: events.append("shown")
    )
    server.print_ready_banner()
    server.print_ready_banner()
    visible = tty and level == logging.INFO
    assert events == (["shown"] if visible else [])
    assert capsys.readouterr().out.count("Next:") == int(visible)
    assert (tmp_path / "state" / "first_serve_success").exists()


@pytest.mark.parametrize(
    "model,label,path",
    [
        ("whisper-base", "Transcribe", "/audio/transcriptions"),
        ("kokoro", "Speak", "/audio/speech"),
    ],
)
def test_audio_ready_banner_has_lane_command(monkeypatch, capsys, model, label, path):
    cfg = get_config()
    monkeypatch.setattr(cfg, "bind_host", "127.0.0.1")
    monkeypatch.setattr(cfg, "bind_port", 18732)
    monkeypatch.setattr(cfg, "model_alias", model)
    monkeypatch.setattr(cfg, "model_name", model)
    monkeypatch.setattr(server, "_served_model_name_set", False)
    server.print_ready_banner()
    out = capsys.readouterr().out
    assert label in out and path in out
    assert "launch codex" not in out and "Anthropic:" not in out


def test_audio_ready_banner_resolves_model_with_opaque_served_alias(
    monkeypatch, capsys
):
    cfg = get_config()
    monkeypatch.setattr(cfg, "bind_host", "127.0.0.1")
    monkeypatch.setattr(cfg, "bind_port", 18732)
    monkeypatch.setattr(cfg, "model_alias", "gateway-stt")
    monkeypatch.setattr(cfg, "model_name", "mlx-community/whisper-base-mlx")
    monkeypatch.setattr(server, "_served_model_name_set", False)
    server.print_ready_banner()
    out = capsys.readouterr().out
    assert "Transcribe" in out and "/audio/transcriptions" in out
    assert "model=mlx-community/whisper-base-mlx" in out
    assert "launch codex" not in out


def test_stt_ready_banner_hints_at_configured_api_key(monkeypatch, capsys):
    cfg = get_config()
    monkeypatch.delenv("RAPID_MLX_API_KEY", raising=False)
    monkeypatch.setattr(cfg, "bind_host", "127.0.0.1")
    monkeypatch.setattr(cfg, "bind_port", 18732)
    monkeypatch.setattr(cfg, "model_alias", "whisper-base")
    monkeypatch.setattr(cfg, "model_name", "whisper-base")
    monkeypatch.setattr(cfg, "api_key", "sk-test")
    monkeypatch.setattr(server, "_served_model_name_set", False)
    server.print_ready_banner()
    out = capsys.readouterr().out
    assert "  Auth: authenticated connections require RAPID_MLX_API_KEY." in out
    assert "    export RAPID_MLX_API_KEY='<server-api-key>'" in out
    assert (
        'curl http://127.0.0.1:18732/v1/audio/transcriptions -H "Authorization: Bearer $RAPID_MLX_API_KEY"'
        in out
    )
    assert "sk-test" not in out


def test_retrieval_ready_banner_has_lane_commands(monkeypatch, capsys):
    cfg = get_config()
    monkeypatch.setattr(cfg, "bind_host", "127.0.0.1")
    monkeypatch.setattr(cfg, "bind_port", 18732)
    monkeypatch.setattr(cfg, "model_name", None)
    monkeypatch.setattr(cfg, "embedding_engine", object())
    monkeypatch.setattr(cfg, "rerank_engine", object())
    monkeypatch.setattr(cfg, "embedding_model_locked", "embed-model")
    monkeypatch.setattr(cfg, "rerank_model_locked", "rank-model")
    server.print_ready_banner()
    out = capsys.readouterr().out
    assert "Embeddings" in out and "/embeddings" in out
    assert "Rerank" in out and "/rerank" in out
    assert "launch codex" not in out and "Anthropic:" not in out


@pytest.mark.parametrize(
    "flag,path",
    [
        ("is_image_gen", "/images/generations"),
        ("is_video_gen", "/videos"),
    ],
)
def test_generation_ready_banner_has_authenticated_lane_command(
    monkeypatch, capsys, flag, path
):
    from types import SimpleNamespace

    cfg = get_config()
    monkeypatch.setattr(cfg, "bind_host", "127.0.0.1")
    monkeypatch.setattr(cfg, "bind_port", 18732)
    monkeypatch.setattr(cfg, "model_alias", "model's name")
    monkeypatch.setattr(cfg, "model_name", "model's name")
    monkeypatch.setattr(cfg, "api_key", "secret")
    monkeypatch.setattr(server, "_engine", SimpleNamespace(**{flag: True}))
    monkeypatch.setattr(server, "_served_model_name_set", False)
    server.print_ready_banner()
    out = capsys.readouterr().out
    assert "Generate" in out and path in out
    assert "Authorization: Bearer $RAPID_MLX_API_KEY" in out
    assert "secret" not in out
    assert "launch codex" not in out and "Anthropic:" not in out


def test_rerank_only_ready_banner_without_primary_model(monkeypatch, capsys):
    cfg = get_config()
    monkeypatch.setattr(cfg, "bind_host", "localhost")
    monkeypatch.setattr(cfg, "bind_port", 8765)
    monkeypatch.setattr(cfg, "model_name", None)
    monkeypatch.setattr(cfg, "model_alias", None)
    monkeypatch.setattr(cfg, "embedding_engine", None)
    monkeypatch.setattr(cfg, "rerank_engine", object())
    monkeypatch.setattr(cfg, "rerank_model_locked", "rank-model")
    server.print_ready_banner()
    out = capsys.readouterr().out
    assert "Ready: http://localhost:8765" in out
    assert "Model:     rank-model" in out
    assert "(not yet loaded)" not in out
    assert "curl http://localhost:8765/v1/rerank" in out


def _banner_form(out, path):
    import shlex

    line = next(line for line in out.splitlines() if "curl " in line and path in line)
    words = shlex.split(line[line.index("curl ") :])
    fields = [words[i + 1] for i, word in enumerate(words) if word == "-F"]
    return dict(field.split("=", 1) for field in fields)


def _png_bytes():
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (16, 16), "red").save(buf, format="PNG")
    return buf.getvalue()


def _banner_setup(monkeypatch, engine, model):
    cfg = get_config()
    monkeypatch.setattr(cfg, "bind_host", "127.0.0.1")
    monkeypatch.setattr(cfg, "bind_port", 18732)
    monkeypatch.setattr(cfg, "model_alias", model)
    monkeypatch.setattr(cfg, "model_name", model)
    monkeypatch.setattr(cfg, "engine", engine)
    monkeypatch.setattr(cfg, "model_registry", None)
    monkeypatch.setattr(cfg, "api_key", None)
    monkeypatch.setattr(server, "_engine", engine)
    monkeypatch.setattr(server, "_served_model_name_set", False)


def test_edit_banner_command_passes_route(monkeypatch, capsys):
    from types import SimpleNamespace

    from fastapi.testclient import TestClient

    png = _png_bytes()
    engine = SimpleNamespace(
        is_image_gen=True,
        supports_generation=False,
        supports_editing=True,
        generate=lambda **kwargs: png,
    )
    _banner_setup(monkeypatch, engine, "edit-model")
    server.print_ready_banner()
    out = capsys.readouterr().out
    assert "/v1/images/generations" not in out
    fields = _banner_form(out, "/v1/images/edits")
    assert fields.pop("image") == "@input.png"
    response = TestClient(server.app).post(
        "/v1/images/edits",
        data=fields,
        files={"image": ("input.png", png, "image/png")},
    )
    assert response.status_code == 200, response.text


@pytest.mark.parametrize(
    "family,wan_type",
    [
        ("ltx-2.3", None),
        ("ltx-2.5", None),
        ("wan", "t2v"),
        ("wan", "ti2v"),
        ("wan", "i2v"),
        ("minimax-h3", None),
        ("cogvideox-fun", None),
    ],
)
def test_video_banner_commands_pass_route(
    monkeypatch, capsys, tmp_path, family, wan_type
):
    from fastapi.testclient import TestClient

    from rapid_mlx.routes import video
    from rapid_mlx.runtime.video_lane import VideoEngine, validate_video_request
    from rapid_mlx.video.minimax_h3 import MiniMaxH3VideoEngine
    from rapid_mlx.video.wan import WanVideoEngine

    engine = VideoEngine.__new__(VideoEngine)
    engine.model_name = "test-video-model"
    engine.video_family = family
    engine._wan_engine = None
    engine._cog_engine = None
    engine._minimax_engine = None
    if wan_type:
        wan = WanVideoEngine.__new__(WanVideoEngine)
        wan.config = {"model_type": wan_type, "sample_fps": 24, "max_area": 901120}
        wan.model_name = "Wan2.2"
        engine._wan_engine = wan
    if family == "minimax-h3":
        engine._minimax_engine = MiniMaxH3VideoEngine.__new__(MiniMaxH3VideoEngine)
    _banner_setup(monkeypatch, engine, engine.model_name)
    monkeypatch.setattr(video, "_video_engine", lambda: engine)
    monkeypatch.setattr(video, "_jobs_root", tmp_path)
    monkeypatch.setattr(video, "_jobs", {})
    monkeypatch.setattr(video, "_tasks", {})
    monkeypatch.setattr(video, "_accepting_jobs", True)

    async def no_generation(*args, **kwargs):
        pass

    monkeypatch.setattr(video, "_run_job", no_generation)
    server.print_ready_banner()
    fields = _banner_form(capsys.readouterr().out, "/v1/videos")
    if wan_type == "i2v":
        assert fields.get("input_reference") == "@start.png"
    if family == "cogvideox-fun":
        assert fields.get("seconds") == "1" and fields.get("size") == "672x384"
    files = {}
    if "input_reference" in fields:
        assert fields.pop("input_reference") == "@start.png"
        files["input_reference"] = ("start.png", _png_bytes(), "image/png")
    response = TestClient(server.app).post("/v1/videos", data=fields, files=files)
    assert response.status_code == 200, response.text
    if family == "minimax-h3":
        job = response.json()
        validate_video_request(
            engine, width=768, height=512, num_frames=job["frames"], image=None
        )


def test_audio_banner_custom_alias_uses_resolvable_ids(monkeypatch, capsys):
    import json
    import shlex

    from fastapi import HTTPException

    from rapid_mlx.audio.registry import resolve_audio_alias
    from rapid_mlx.routes.audio import _resolve_stt_model, _resolve_tts_model

    cfg = get_config()
    for alias, catalog, kind in [
        ("gateway-stt", "whisper-base", "stt"),
        ("gateway-tts", "kokoro", "tts"),
    ]:
        entry = resolve_audio_alias(catalog)
        _banner_setup(monkeypatch, None, alias)
        monkeypatch.setattr(cfg, "model_name", entry.hf_id)
        server.print_ready_banner()
        out = capsys.readouterr().out
        if kind == "stt":
            model = _banner_form(out, "/v1/audio/transcriptions")["model"]
            assert model == entry.hf_id
            assert _resolve_stt_model(model) == entry.hf_id
            with pytest.raises(HTTPException) as exc:
                _resolve_stt_model(alias)
            assert exc.value.status_code == 404
        else:
            line = next(
                line
                for line in out.splitlines()
                if "curl " in line and "/audio/speech" in line
            )
            words = shlex.split(line[line.index("curl ") :])
            model = json.loads(words[words.index("-d") + 1])["model"]
            assert model == entry.hf_id
            assert _resolve_tts_model(model) == entry.hf_id


@pytest.mark.parametrize("kind", ["stt", "tts"])
def test_local_audio_banner_keeps_served_name(monkeypatch, capsys, kind):
    import json
    import shlex
    from types import SimpleNamespace

    from rapid_mlx.audio import registry

    entry = SimpleNamespace(
        type=kind,
        notes="local folder",
        hf_id="/local/model",
        family=None,
        alias="local-audio",
    )
    monkeypatch.setattr(registry, "resolve_audio_alias", lambda name: entry)
    _banner_setup(monkeypatch, None, "local-audio")
    server.print_ready_banner()
    out = capsys.readouterr().out
    if kind == "stt":
        assert _banner_form(out, "/v1/audio/transcriptions")["model"] == "local-audio"
    else:
        line = next(
            line
            for line in out.splitlines()
            if "curl " in line and "/audio/speech" in line
        )
        words = shlex.split(line[line.index("curl ") :])
        assert json.loads(words[words.index("-d") + 1])["model"] == "local-audio"


def test_retrieval_only_startup_emits_embedding_banner(monkeypatch, capsys):
    from types import SimpleNamespace

    from rapid_mlx import _uvicorn, _version_check, cli, update_refresh

    cfg = get_config()
    monkeypatch.setattr(server, "configure_logging", lambda level: "info")
    monkeypatch.setattr(
        cli, "_configure_engineless_server_security", lambda *args: None
    )
    monkeypatch.setattr(server, "_sync_config", lambda: None)
    monkeypatch.setattr(cfg, "model_name", None)
    monkeypatch.setattr(cfg, "model_alias", None)
    monkeypatch.setattr(cfg, "api_key", None)
    monkeypatch.setattr(cfg, "rerank_engine", None)
    monkeypatch.setattr(cfg, "rerank_model_locked", None)
    monkeypatch.setattr(cfg, "embedding_engine", object())
    monkeypatch.setattr(cfg, "embedding_model_locked", "embed-model")
    monkeypatch.setattr(cli, "_load_embedding_model_or_exit", lambda args, load: None)
    monkeypatch.setattr(
        _version_check, "print_staleness_warning_if_any", lambda **kw: False
    )
    monkeypatch.setattr(update_refresh, "start", lambda **kw: None)
    monkeypatch.setattr(
        _uvicorn, "run_uvicorn", lambda app, **kw: kw["on_server_accepting"]()
    )
    monkeypatch.setattr(cli, "_hard_exit_after_serve", lambda: None)
    args = SimpleNamespace(
        embedding_model="embed-model",
        host="127.0.0.1",
        port=8123,
        listen_fd=None,
        log_level="info",
    )
    cli._serve_retrieval_only_mode(args)
    assert "curl http://127.0.0.1:8123/v1/embeddings" in capsys.readouterr().out
