# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for ``model_served`` across every serving lane."""

from __future__ import annotations

import http.client
import importlib
import inspect
import json
import os
import sys
import threading
from concurrent.futures import Future
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import FunctionType, ModuleType, SimpleNamespace

import pytest

import rapid_mlx
from rapid_mlx import server
from rapid_mlx.telemetry import (
    consent_runtime,
    model_events,
    posthog_sender,
    state,
    store,
)
from rapid_mlx.telemetry import track as track_module
from rapid_mlx.telemetry.build_gate import ReleaseStamp
from rapid_mlx.telemetry.common_props import PlatformFacts

STAMP = ReleaseStamp(channel="stable", posthog_key="phc_" + "a" * 32)
FACTS = PlatformFacts(
    os="linux",
    os_version="6.8",
    arch="x86_64",
    chip="other",
    memory_gb=16,
    python_version="3.11",
)


class _CaptureHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers["Content-Length"])
        self.server.bodies.append(self.rfile.read(length))  # type: ignore[attr-defined]
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, _format: str, *_args: object) -> None:
        pass


@pytest.fixture
def loopback_model_served(monkeypatch, tmp_path):
    pending_callbacks = []
    sink = HTTPServer(("127.0.0.1", 0), _CaptureHandler)
    sink.bodies = []  # type: ignore[attr-defined]
    sink_thread = threading.Thread(target=sink.serve_forever, daemon=True)
    sink_thread.start()
    url = f"http://127.0.0.1:{sink.server_port}/batch/"
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv(posthog_sender.POSTHOG_URL_ENV, url)
    monkeypatch.setattr(rapid_mlx, "__version__", "0.15.1")
    monkeypatch.setattr(track_module.build_gate, "official_build", lambda: STAMP)
    monkeypatch.setattr(consent_runtime, "upload_allowed", lambda: True)
    monkeypatch.setattr(track_module.common_props, "read_platform_facts", lambda: FACTS)
    monkeypatch.setattr(
        state,
        "get_or_create_client_id",
        lambda: "6f1b1d3e-4a2b-4c9d-8e7f-0a1b2c3d4e5f",
    )
    monkeypatch.setattr(
        state,
        "session_id",
        lambda: "0a1b2c3d-4e5f-6071-8293-a4b5c6d7e8f9",
    )
    monkeypatch.setattr(store, "days_since_first_run_bucket", lambda: "0")
    monkeypatch.setattr(store, "note_model_served", lambda _model: 1)
    monkeypatch.setattr(
        model_events,
        "_submit_model_served",
        lambda callback: pending_callbacks.append(callback) or True,
    )

    def post(_url: str, body: bytes, timeout: float) -> int:
        connection = http.client.HTTPConnection(
            "127.0.0.1", sink.server_port, timeout=timeout
        )
        try:
            connection.request(
                "POST",
                "/batch/",
                body=body,
                headers={"Content-Type": "application/json"},
            )
            return connection.getresponse().status
        finally:
            connection.close()

    sender = posthog_sender.PostHogSender(
        post=post, gate=lambda: STAMP, allowed=lambda: True
    )
    monkeypatch.setattr(posthog_sender, "get_sender", lambda: sender)
    track_module._reset_for_tests()
    server._telemetry_model_served_state = "idle"
    server._telemetry_audio_model_served_state = "idle"
    server._telemetry_embedding_model_served_state = "idle"

    def events() -> list[dict[str, object]]:
        for callback in pending_callbacks:
            callback()
        sender.flush()
        return [
            item
            for body in sink.bodies  # type: ignore[attr-defined]
            for item in json.loads(body)["batch"]
        ]

    yield events
    sender.close()
    sink.shutdown()
    sink_thread.join(timeout=2)
    sink.server_close()
    track_module._reset_for_tests()


@pytest.mark.parametrize(
    ("lane", "model_ref", "expected"),
    [
        ("image", "sdxl-base", "image-gen"),
        ("video", "cogvideox-fun-5b-q4", "video-gen"),
        ("diffusion", "diffusion-gemma-26b-4bit", "text-diffusion"),
        ("audio", "kokoro", "audio"),
        ("embedding", "embeddinggemma-300m-6bit", "embedding"),
    ],
)
def test_every_lane_queues_exactly_one_model_served_on_loopback(
    monkeypatch, loopback_model_served, lane, model_ref, expected
):
    engine = SimpleNamespace(model_name=model_ref)
    if lane in {"image", "video", "diffusion"}:
        monkeypatch.setattr(server, "_model_alias", model_ref)
        server._emit_primary_model_served_once(engine)
        server._emit_primary_model_served_once(engine)
    elif lane == "audio":
        server._emit_audio_model_served_once(engine, model_ref)
        server._emit_audio_model_served_once(engine, model_ref)
    else:
        server._emit_embedding_model_served_once(engine, model_ref)
        server._emit_embedding_model_served_once(engine, model_ref)

    events = loopback_model_served()
    served = [item for item in events if item["event"] == "model_served"]
    assert len(served) == 1
    assert served[0]["properties"]["model_type"] == expected


def test_declined_primary_emission_does_not_latch_and_retries(monkeypatch):
    upload_gate = iter((False, True))
    queued: list[str] = []
    monkeypatch.setattr(track_module, "_upload_allowed", lambda: next(upload_gate))
    monkeypatch.setattr(
        track_module,
        "track",
        lambda event, _props, **_kwargs: queued.append(event) or True,
    )
    monkeypatch.setattr(store, "note_model_served", lambda _model: 1)
    callbacks = []
    monkeypatch.setattr(
        model_events,
        "_submit_model_served",
        lambda callback: callbacks.append(callback) or True,
    )
    monkeypatch.setattr(server, "_model_alias", "sdxl-base")
    monkeypatch.setattr(server, "_telemetry_model_served_state", "idle")

    server._emit_primary_model_served_once(object())
    assert server._telemetry_model_served_state == "pending"
    callbacks.pop(0)()
    assert server._telemetry_model_served_state == "idle"
    server._emit_primary_model_served_once(object())
    callbacks.pop(0)()
    assert server._telemetry_model_served_state == "emitted"
    assert queued == ["model_served"]


def test_slow_model_served_store_never_blocks_request_event_loop(monkeypatch):
    store_started = threading.Event()
    callbacks = []
    monkeypatch.setattr(track_module, "_upload_allowed", lambda: True)

    def slow_note(_model: str) -> int:
        store_started.set()
        return 1

    monkeypatch.setattr(store, "note_model_served", slow_note)
    monkeypatch.setattr(track_module, "track", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        model_events,
        "_submit_model_served",
        lambda callback: callbacks.append(callback) or True,
    )

    assert model_events.emit_model_served(None, "sdxl-base", False) is True
    assert len(callbacks) == 1
    assert not store_started.is_set()

    callbacks[0]()
    assert store_started.is_set()


def test_concurrent_lane_submission_does_not_share_a_blocking_lock(monkeypatch):
    order = []

    class InstrumentedLock:
        def __init__(self, name):
            self.name = name
            self.held = False

        def __enter__(self):
            assert not self.held
            self.held = True
            order.append(f"enter:{self.name}")

        def __exit__(self, *_args):
            order.append(f"exit:{self.name}")
            self.held = False

    primary_lock = InstrumentedLock("primary")
    audio_lock = InstrumentedLock("audio")
    monkeypatch.setattr(
        server,
        "_telemetry_model_served_locks",
        {
            "_telemetry_model_served_state": primary_lock,
            "_telemetry_audio_model_served_state": audio_lock,
            "_telemetry_embedding_model_served_state": InstrumentedLock("embedding"),
        },
    )
    monkeypatch.setattr(server, "_telemetry_model_served_state", "idle", raising=False)
    monkeypatch.setattr(
        server, "_telemetry_audio_model_served_state", "idle", raising=False
    )

    def submit(engine, _model, _auto, **_kwargs):
        lane = "primary" if engine is not None else "audio"
        order.append(f"submit:{lane}")
        if engine is not None:
            assert not primary_lock.held
            assert not audio_lock.held
            server._emit_audio_model_served_once(object(), "kokoro")
        return True

    monkeypatch.setattr(model_events, "emit_model_served", submit)
    server._emit_primary_model_served_once(object())

    assert order == [
        "enter:primary",
        "exit:primary",
        "submit:primary",
        "enter:audio",
        "exit:audio",
        "submit:audio",
    ]


@pytest.mark.parametrize("accepted", [False, True])
def test_inline_model_served_completion_cannot_strand_pending(monkeypatch, accepted):
    """A fast executor may complete before ``emit_model_served`` returns."""

    def complete_inline(_engine, _model, _auto, *, on_complete):
        on_complete(accepted)
        return True

    monkeypatch.setattr(model_events, "emit_model_served", complete_inline)
    monkeypatch.setattr(server, "_telemetry_model_served_state", "idle")

    server._emit_primary_model_served_once(object())

    assert server._telemetry_model_served_state == ("emitted" if accepted else "idle")


def test_rejected_model_served_submission_releases_reservation(monkeypatch):
    monkeypatch.setattr(model_events, "emit_model_served", lambda *_a, **_kw: False)
    monkeypatch.setattr(server, "_telemetry_model_served_state", "idle")

    server._emit_primary_model_served_once(object())

    assert server._telemetry_model_served_state == "idle"


@pytest.mark.parametrize(
    ("helper", "state_name", "model_name", "expected"),
    [
        (
            server._emit_audio_model_served_once,
            "_telemetry_audio_model_served_state",
            "whisper-large-v3",
            "whisper-large-v3",
        ),
        (
            server._emit_audio_model_served_once,
            "_telemetry_audio_model_served_state",
            "mlx-community/whisper-large-v3-mlx",
            "whisper",
        ),
        (
            server._emit_audio_model_served_once,
            "_telemetry_audio_model_served_state",
            "kokoro",
            "kokoro",
        ),
        (
            server._emit_embedding_model_served_once,
            "_telemetry_embedding_model_served_state",
            "embeddinggemma-300m-6bit",
            "embeddinggemma-300m-6bit",
        ),
    ],
)
def test_auxiliary_lanes_use_resolved_model_reference(
    monkeypatch, helper, state_name, model_name, expected
):
    captured = []

    def capture(engine, model_ref, auto_selected, **_kwargs):
        captured.append((engine, model_ref, auto_selected))
        return True

    monkeypatch.setattr(model_events, "emit_model_served", capture)
    monkeypatch.setattr(server, state_name, "idle", raising=False)
    monkeypatch.setattr(
        server, "_telemetry_audio_model_served_emitted", False, raising=False
    )
    monkeypatch.setattr(
        server, "_telemetry_embedding_model_served_emitted", False, raising=False
    )
    helper(object(), model_name)
    assert captured == [(None, model_name, False)]
    assert model_events._serve_props(*captured[0])["model"] == expected


def test_model_served_completion_callback_failure_is_contained(monkeypatch):
    monkeypatch.setattr(track_module, "_upload_allowed", lambda: False)
    model_events._record_model_served(
        None,
        "kokoro",
        False,
        lambda _accepted: (_ for _ in ()).throw(RuntimeError("callback failed")),
    )


def test_diffusion_lane_declares_load_time_telemetry_marker():
    from rapid_mlx.runtime import diffusion_lane

    reloaded = importlib.reload(diffusion_lane)
    assert reloaded.DiffusionEngine.is_text_diffusion is True


def test_cog_backend_signals_after_load_before_generation(monkeypatch, tmp_path):
    from rapid_mlx.video.engine import VideoGenerationEngine

    order = []
    generated = tmp_path / "generated.mp4"
    generated.write_bytes(b"mp4")
    engine = VideoGenerationEngine.__new__(VideoGenerationEngine)
    engine._load_sync = lambda: order.append("load")
    engine._generate_sync = lambda **_kwargs: (
        order.append("inference"),
        generated,
    )[1]

    def submit(function):
        future = Future()
        future.set_result(function())
        return future

    engine._submit = submit
    engine.generate_sync(
        output_path=tmp_path / "output.mp4",
        on_loaded=lambda: order.append("model_served"),
    )
    assert order == ["load", "model_served", "inference"]


@pytest.mark.requires_mlx
@pytest.mark.parametrize("dual_model", [False, True])
def test_wan_pinned_runtime_emits_after_all_loads_before_first_embed(
    monkeypatch, tmp_path, dual_model
):
    wan_generator = pytest.importorskip("mlx_video.generate_wan")
    import mlx.core as mx

    from rapid_mlx.video.wan import WanVideoEngine

    class FirstPostLoadUseError(Exception):
        pass

    order = []
    model_root = tmp_path / "wan"
    model_root.mkdir()
    (model_root / "config.json").write_text(
        json.dumps(
            {
                "model_type": "t2v",
                "model_version": "2.2" if dual_model else "2.1",
                "dual_model": dual_model,
                "sample_guide_scale": 1.0,
                "sample_steps": 1,
                "sample_shift": 1.0,
                "max_area": 64 * 64,
            }
        )
    )

    class Model:
        def __init__(self, name):
            self.name = name

        def embed_text(self, _context):
            order.append(f"embed:{self.name}")
            raise FirstPostLoadUseError

    def load_model(path, *_args, **_kwargs):
        name = {
            "model.safetensors": "single",
            "low_noise_model.safetensors": "low",
            "high_noise_model.safetensors": "high",
        }[Path(path).name]
        order.append(f"load:{name}")
        return Model(name)

    namespace = wan_generator.generate_video.__globals__
    monkeypatch.setitem(namespace, "load_t5_encoder", lambda *_args: object())
    monkeypatch.setitem(namespace, "encode_text", lambda *_args: mx.array([0.0]))
    monkeypatch.setitem(namespace, "load_wan_model", load_model)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda *_args: object())
        ),
    )
    monkeypatch.setattr(
        server,
        "_emit_primary_model_served_once",
        lambda _engine: order.append("model_served"),
    )

    backend = WanVideoEngine(str(model_root))
    backend.steps = 1
    engine = _video_engine(_VideoBackend())
    engine.model_name = str(model_root)
    engine._wan_engine = backend

    with pytest.raises(FirstPostLoadUseError):
        _generate_video(engine, tmp_path / "output.mp4")

    expected_loads = ["load:low", "load:high"] if dual_model else ["load:single"]
    assert order == [
        *expected_loads,
        "model_served",
        f"embed:{'low' if dual_model else 'single'}",
    ]
    assert order.count("model_served") == 1


def test_wan_converted_runtime_executes_materialization_wrapper(monkeypatch, tmp_path):
    from rapid_mlx.video import wan_diffusers

    order = []

    def generate_template(**_kwargs):
        globals()["load_wan_model"]("model", SimpleNamespace(dual_model=False))
        globals()["order"].append("embed")

    namespace = {
        **generate_template.__globals__,
        "order": order,
        "load_wan_model": lambda *_args: order.append("load"),
    }
    generate = FunctionType(generate_template.__code__, namespace)

    @contextmanager
    def runtime(_root, _generator):
        yield tmp_path, generate

    monkeypatch.setattr(wan_diffusers, "is_diffusers_wan21_layout", lambda _root: True)
    monkeypatch.setattr(wan_diffusers, "diffusers_runtime", runtime)
    wan_diffusers.generate_with_runtime(
        tmp_path,
        SimpleNamespace(),
        {},
        on_loaded=lambda: order.append("model_served"),
    )

    plain = lambda: order.append("plain")
    assert wan_diffusers._notify_after_materialization(plain, order.clear) is plain
    plain()
    assert order == ["load", "model_served", "embed", "plain"]


@pytest.mark.parametrize("ready", [False, True])
def test_ltx25_child_failure_emits_only_after_readiness(monkeypatch, tmp_path, ready):
    from rapid_mlx.video import ltx25

    emitted = []

    class Process:
        returncode = 1

        def __init__(self, _command, **kwargs):
            pass_fds = kwargs.get("pass_fds", ())
            self.ready_fd = os.dup(pass_fds[0]) if pass_fds else None

        def communicate(self, *, input, timeout):
            del input, timeout
            if self.ready_fd is not None:
                if ready:
                    os.write(self.ready_fd, b"RMLX_LTX25_READY\n")
                os.close(self.ready_fd)

    monkeypatch.setattr(ltx25, "embedded_ltx25_interpreter", lambda: "/python")
    monkeypatch.setattr(ltx25.subprocess, "Popen", Process)
    monkeypatch.setattr(ltx25.LTX25VideoEngine, "_terminate_process", lambda *_: None)
    engine = _video_engine(_VideoBackend())
    engine.video_family = "ltx-2.5"
    engine._wan_engine = None
    engine._ltx25_engine = ltx25.LTX25VideoEngine("ltx-2.5-mlx-q8")
    monkeypatch.setattr(
        server,
        "_emit_primary_model_served_once",
        lambda loaded: emitted.append(loaded),
    )
    with pytest.raises(Exception, match="exited with code 1"):
        _generate_video(engine, tmp_path / "output.mp4")
    assert emitted == ([engine] if ready else [])


class _ImageBackend:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.loads = 0
        self.generations = 0

    def _ensure_loaded(self, *, for_edit: bool | None = None) -> object:
        del for_edit
        if self.fail:
            raise RuntimeError("load failed")
        self.loads += 1
        return self

    def generate(self, **_kwargs) -> bytes:
        self.generations += 1
        return b"png"

    def generate_with_performance(self, **kwargs):
        return self.generate(**kwargs), {}


def _image_engine(backend: _ImageBackend):
    from rapid_mlx.runtime.image_lane import ImageEngine

    engine = ImageEngine.__new__(ImageEngine)
    engine.model_name = "sdxl-base"
    engine._engine = backend
    return engine


def test_image_backing_load_precedes_inference_and_reload_is_one_shot(monkeypatch):
    order: list[str] = []
    monkeypatch.setattr(
        server,
        "_emit_primary_model_served_once",
        lambda _engine: order.append("model_served"),
    )
    backend = _ImageBackend()
    engine = _image_engine(backend)
    engine.generate(prompt="x")
    order.append("inference")
    assert order == ["model_served", "inference"]


def test_image_failed_backing_load_emits_nothing(monkeypatch):
    emitted: list[object] = []
    monkeypatch.setattr(
        server,
        "_emit_primary_model_served_once",
        lambda engine: emitted.append(engine),
    )
    with pytest.raises(RuntimeError, match="load failed"):
        _image_engine(_ImageBackend(fail=True)).generate(prompt="x")
    assert emitted == []


class _VideoBackend:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail

    def generate(self, **kwargs) -> None:
        if self.fail:
            from rapid_mlx.video.wan import WanBackendError

            raise WanBackendError("load failed")
        kwargs["on_loaded"]()
        Path(kwargs["output_path"]).write_bytes(b"mp4")


def _video_engine(backend: _VideoBackend):
    from rapid_mlx.runtime.video_lane import VideoEngine

    engine = VideoEngine.__new__(VideoEngine)
    engine.model_name = "cogvideox-fun-5b-q4"
    engine.video_family = "wan"
    engine._wan_engine = backend
    engine._ltx25_engine = None
    engine._cog_engine = None
    engine._generation_lock = threading.Lock()
    return engine


def _generate_video(engine, output_path: Path) -> None:
    engine.generate(
        prompt="x",
        output_path=output_path,
        width=64,
        height=64,
        num_frames=5,
        fps=8,
        seed=1,
        image=None,
    )


def test_video_success_emits_before_inference_event(monkeypatch, tmp_path):
    order: list[str] = []
    monkeypatch.setattr(
        server,
        "_emit_primary_model_served_once",
        lambda _engine: order.append("model_served"),
    )

    class Backend(_VideoBackend):
        def generate(self, **kwargs):
            kwargs["on_loaded"]()
            order.append("inference")
            Path(kwargs["output_path"]).write_bytes(b"mp4")

    _generate_video(_video_engine(Backend()), tmp_path / "out.mp4")
    assert order == ["model_served", "inference"]


def test_video_failed_backing_load_emits_nothing(monkeypatch, tmp_path):
    emitted: list[object] = []
    monkeypatch.setattr(
        server,
        "_emit_primary_model_served_once",
        lambda engine: emitted.append(engine),
    )
    with pytest.raises(Exception, match="load failed"):
        _generate_video(_video_engine(_VideoBackend(fail=True)), tmp_path / "out.mp4")
    assert emitted == []


@pytest.fixture
def embedding_mlx_stubs(monkeypatch):
    mlx = ModuleType("mlx")
    mlx_core = ModuleType("mlx.core")
    mlx.core = mlx_core
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", mlx_core)


def test_embedding_emits_only_after_successful_load(monkeypatch, embedding_mlx_stubs):
    order: list[str] = []

    class Engine:
        def __init__(self, model_name, **_kwargs):
            self.model_name = model_name

        def load(self):
            order.append("load")

    monkeypatch.setattr("rapid_mlx.embedding.EmbeddingEngine", Engine)
    monkeypatch.setattr(
        server,
        "_emit_embedding_model_served_once",
        lambda _engine, _name: order.append("model_served"),
    )
    monkeypatch.setattr(server, "_embedding_engine", None)
    server.load_embedding_model("embeddinggemma-300m-6bit", reuse_existing=False)
    assert order == ["load", "model_served"]


def test_embedding_load_failure_emits_nothing(monkeypatch, embedding_mlx_stubs):
    class Engine:
        def __init__(self, model_name, **_kwargs):
            self.model_name = model_name

        def load(self):
            raise RuntimeError("load failed")

    emitted: list[object] = []
    monkeypatch.setattr("rapid_mlx.embedding.EmbeddingEngine", Engine)
    monkeypatch.setattr(
        server,
        "_emit_embedding_model_served_once",
        lambda *args: emitted.append(args),
    )
    with pytest.raises(RuntimeError, match="load failed"):
        server.load_embedding_model("embeddinggemma-300m-6bit", reuse_existing=False)
    assert emitted == []


@pytest.mark.parametrize("lane", ["diffusion", "stt", "tts", "embedding"])
def test_successful_load_always_precedes_lane_inference(monkeypatch, lane):
    order: list[str] = []
    monkeypatch.setattr(
        model_events,
        "emit_model_served",
        lambda *_args, **_kwargs: order.append("model_served") or True,
    )
    server._telemetry_model_served_state = "idle"
    server._telemetry_audio_model_served_state = "idle"
    server._telemetry_embedding_model_served_state = "idle"
    engine = SimpleNamespace()
    if lane == "diffusion":
        server._emit_primary_model_served_once(engine)
    elif lane in {"stt", "tts"}:
        server._emit_audio_model_served_once(engine, "kokoro")
    else:
        server._emit_embedding_model_served_once(engine, "embeddinggemma-300m-6bit")
    order.append("inference")
    assert order == ["model_served", "inference"]


def test_stt_and_tts_share_the_audio_one_shot(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(
        model_events,
        "emit_model_served",
        lambda _engine, model_name, _auto, **_kwargs: calls.append(model_name) or True,
    )
    server._telemetry_audio_model_served_state = "idle"
    server._emit_audio_model_served_once(object(), "whisper-large-v3")
    server._emit_audio_model_served_once(object(), "kokoro")
    assert calls == ["whisper-large-v3"]


@pytest.mark.parametrize(
    ("helper", "latch"),
    [
        (server._emit_audio_model_served_once, "_telemetry_audio_model_served_state"),
        (
            server._emit_embedding_model_served_once,
            "_telemetry_embedding_model_served_state",
        ),
    ],
)
def test_auxiliary_emit_failure_never_latches(monkeypatch, helper, latch):
    monkeypatch.setattr(
        model_events,
        "emit_model_served",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("telemetry failed")),
    )
    monkeypatch.setattr(server, latch, "idle")
    helper(object(), "kokoro")
    assert getattr(server, latch) == "idle"


def test_tts_load_emits_before_inference(monkeypatch):
    from rapid_mlx.routes import audio

    order: list[str] = []

    class Engine:
        def __init__(self, model_name):
            self.model_name = model_name

        def load(self):
            order.append("load")

        def generate(self, _text, **_kwargs):
            order.append("inference")
            return SimpleNamespace(audio=b"raw", sample_rate=24_000, duration=1.0)

        def to_bytes(self, _audio, *, format):
            del format
            return b"wav"

    monkeypatch.setattr("rapid_mlx.audio.tts.TTSEngine", Engine)
    monkeypatch.setattr(
        "rapid_mlx.runtime.audio_worker.run_audio_mlx_sync",
        lambda _lane, _model, _operation, function, *args, **kwargs: function(
            *args, **kwargs
        ),
    )
    monkeypatch.setattr(
        "rapid_mlx.audio.output_format.convert_audio_output",
        lambda *args, **_kwargs: (args[0], args[1], 1),
    )
    monkeypatch.setattr(
        server,
        "_emit_audio_model_served_once",
        lambda _engine, _name: order.append("model_served"),
    )
    monkeypatch.setattr(audio, "_tts_engine", None)
    result = audio._generate_speech_blocking(
        "kokoro", "hello", "wav", {}, None, None, None, None
    )
    assert result == (b"wav", 24_000, 1)
    assert order == ["load", "model_served", "inference"]


def test_cog_video_emits_after_load_before_pipeline(monkeypatch, tmp_path):
    order: list[str] = []

    class Backend:
        def generate_sync(self, *, output_path, **_kwargs):
            _kwargs["on_loaded"]()
            order.append("inference")
            output_path.write_bytes(b"mp4")

    engine = _video_engine(_VideoBackend())
    engine.video_family = "cogvideox-fun"
    engine._wan_engine = None
    engine._cog_engine = Backend()
    monkeypatch.setattr(
        server,
        "_emit_primary_model_served_once",
        lambda _engine: order.append("model_served"),
    )
    _generate_video(engine, tmp_path / "cog.mp4")
    assert order == ["model_served", "inference"]


def test_ltx23_pinned_wrapper_emits_after_generation(monkeypatch, tmp_path):
    order: list[str] = []

    def generate_video_with_audio(*args, **kwargs):
        del args
        order.append("inference")
        Path(kwargs["output_path"]).write_bytes(b"mp4")

    engine = _video_engine(_VideoBackend())
    engine.video_family = "ltx-2.3"
    engine._wan_engine = None
    monkeypatch.setattr(
        "rapid_mlx.runtime.video_lane._resolve_ffmpeg", lambda: "/usr/bin/true"
    )
    monkeypatch.setitem(
        sys.modules,
        "mlx_video",
        SimpleNamespace(generate_video_with_audio=generate_video_with_audio),
    )
    monkeypatch.setattr(
        server,
        "_emit_primary_model_served_once",
        lambda _engine: order.append("model_served"),
    )
    _generate_video(engine, tmp_path / "ltx.mp4")
    assert inspect.signature(generate_video_with_audio) == inspect.Signature(
        parameters=[
            inspect.Parameter("args", inspect.Parameter.VAR_POSITIONAL),
            inspect.Parameter("kwargs", inspect.Parameter.VAR_KEYWORD),
        ]
    )
    assert order == ["inference", "model_served"]


def test_ltx23_future_explicit_hook_emits_before_generation(monkeypatch, tmp_path):
    order: list[str] = []

    def generate_video_with_audio(*, on_loaded, **kwargs):
        on_loaded()
        order.append("inference")
        Path(kwargs["output_path"]).write_bytes(b"mp4")

    engine = _video_engine(_VideoBackend())
    engine.video_family = "ltx-2.3"
    engine._wan_engine = None
    monkeypatch.setattr(
        "rapid_mlx.runtime.video_lane._resolve_ffmpeg", lambda: "/usr/bin/true"
    )
    monkeypatch.setitem(
        sys.modules,
        "mlx_video",
        SimpleNamespace(generate_video_with_audio=generate_video_with_audio),
    )
    monkeypatch.setattr(
        server,
        "_emit_primary_model_served_once",
        lambda _engine: order.append("model_served"),
    )
    _generate_video(engine, tmp_path / "future-ltx.mp4")
    assert order == ["model_served", "inference"]
