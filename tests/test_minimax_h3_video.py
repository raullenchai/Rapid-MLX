import fnmatch
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from rapid_mlx.model_aliases import resolve_profile, subfolder_allow_patterns
from rapid_mlx.video.minimax_h3 import (
    FILES,
    MiniMaxH3BackendError,
    MiniMaxH3RequestError,
    MiniMaxH3VideoEngine,
)
from rapid_mlx.video.minimax_h3 import _merge_turbo_lora as real_merge_turbo_lora

# Tests that exercise the real MLX runtime and/or the real encoder.
_REAL_RUNTIME_TESTS = {
    "test_generation_encodes_with_ffmpeg_binary_when_path_has_no_ffmpeg"
}


@pytest.fixture(autouse=True)
def fake_mlx_cache(monkeypatch, request):
    if request.node.originalname in _REAL_RUNTIME_TESTS:
        return
    # Existing generation tests must not import the real MLX runtime.
    mlx = ModuleType("mlx")
    core = ModuleType("mlx.core")
    core.clear_cache = lambda: None
    mlx.core = core
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    from rapid_mlx import _mirror
    from rapid_mlx.video import encoding, minimax_h3

    if request.node.originalname != "test_invalid_audio_is_rejected":
        monkeypatch.setattr(
            encoding,
            "encode_rgb_video",
            lambda frames, output_path, fps, **kwargs: output_path.write_bytes(b"mp4"),
        )
    monkeypatch.setattr(_mirror, "pinned_snapshot_download", lambda *a, **k: "/fake")
    monkeypatch.setattr(minimax_h3, "_merge_turbo_lora", lambda *a: None)


def fake_result():
    return SimpleNamespace(
        frames=[[[[[0.5, 0.5, 0.5]]]]], fps=24, audio=None, sampling_rate=None
    )


def fake_pipeline():
    return SimpleNamespace(
        transformer=object(), scheduler=SimpleNamespace(set_shift=lambda _: None)
    )


def test_alias_and_download_selection():
    profile = resolve_profile("minimax-h3")
    assert profile.hf_path == "MiniMaxAI/MiniMax-H3"
    assert profile.revision == "b3c7290e66afdf293bef3b9077b7a266ef421f34"
    assert subfolder_allow_patterns(profile.hf_path) == list(FILES)
    for bad in (
        "FL2VA/model.safetensors",
        "Ref2VA/model.safetensors",
        "transformer/model.py",
    ):
        assert not any(fnmatch.fnmatchcase(bad, pattern) for pattern in FILES)


def test_request_validation():
    validate = MiniMaxH3VideoEngine.validate_request
    validate(width=768, height=512, num_frames=124, image=None)
    with pytest.raises(MiniMaxH3RequestError):
        validate(width=768, height=511, num_frames=124, image=None)
    with pytest.raises(MiniMaxH3RequestError):
        validate(width=768, height=512, num_frames=125, image=None)


def test_generation_loads_once_and_maps_workflows(monkeypatch, tmp_path):
    module = ModuleType("mlx_vlm.generate")
    calls = []

    class Request:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    def load(path, workflow):
        calls.append(("load", path, workflow))
        return SimpleNamespace(workflow=workflow, pipeline=fake_pipeline())

    def generate(model, request):
        calls.append((model.workflow, request))
        return fake_result()

    module.VideoGenerationRequest = Request
    module.generate_video = generate
    module.load_video_generation_model = load
    monkeypatch.setitem(sys.modules, "mlx_vlm.generate", module)
    engine = MiniMaxH3VideoEngine(str(tmp_path))
    args = dict(
        prompt="test",
        output_path=tmp_path / "out.mp4",
        width=768,
        height=512,
        num_frames=124,
        fps=24,
        seed=7,
        image=None,
    )
    engine.generate(**args)
    args["image"] = tmp_path / "first.png"
    engine.generate(**args, last_image=tmp_path / "last.png")
    assert [call[0] for call in calls] == ["load", "t2va", "fl2va"]
    assert calls[2][1].last_image == tmp_path / "last.png"
    assert all(call[1].extra == {"drop_adaln_weights": False} for call in calls[1:])
    engine.stop()
    with pytest.raises(MiniMaxH3BackendError, match="cancelled"):
        engine.generate(**args)


def test_generation_encodes_with_ffmpeg_binary_when_path_has_no_ffmpeg(
    monkeypatch, tmp_path
):
    ffmpeg = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
    if not Path(ffmpeg).is_file() or not Path("/usr/bin/afconvert").is_file():
        pytest.skip("ffmpeg or afconvert unavailable")
    ffmpeg = str(Path(ffmpeg).resolve())
    encoders = subprocess.run(
        [ffmpeg, "-hide_banner", "-encoders"],
        capture_output=True,
        text=True,
        check=True,
    )
    if "h264_videotoolbox" not in encoders.stdout:
        pytest.skip("VideoToolbox encoder unavailable")
    monkeypatch.setenv("FFMPEG_BINARY", ffmpeg)
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    assert shutil.which("ffmpeg") is None

    import numpy as np

    mx = pytest.importorskip("mlx.core", exc_type=ImportError)
    module = pytest.importorskip("mlx_vlm.generate", exc_type=ImportError)
    from mlx_vlm.generate.video_generation import VideoGenerationResult

    from rapid_mlx import _mirror
    from rapid_mlx.video import minimax_h3

    monkeypatch.setattr(
        _mirror, "pinned_snapshot_download", lambda *a, **k: str(tmp_path)
    )
    monkeypatch.setattr(minimax_h3, "_merge_turbo_lora", lambda *a: None)

    def generate(request):
        samples = 0.2 * np.sin(2 * np.pi * 440 * np.arange(24000) / 48000)
        return VideoGenerationResult(
            frames=mx.array(np.zeros((1, 12, 64, 96, 3), dtype=np.float32)),
            audio=mx.array(np.stack([samples, samples]).astype(np.float32)),
            fps=24.0,
            sampling_rate=48000,
            seed=request.seed,
            width=96,
            height=64,
            num_frames=12,
            steps=9,
            model="h3",
            family="minimax_h3",
            workflow="t2va",
        )

    monkeypatch.setattr(
        module,
        "load_video_generation_model",
        lambda path, workflow: SimpleNamespace(
            workflow=workflow, pipeline=fake_pipeline(), generate=generate
        ),
    )
    engine = MiniMaxH3VideoEngine(str(tmp_path))
    output = tmp_path / "out.mp4"
    try:
        engine.generate(
            prompt="t",
            output_path=output,
            width=768,
            height=512,
            num_frames=124,
            fps=24,
            seed=7,
            image=None,
        )
    finally:
        engine.stop()
    assert output.is_file() and output.stat().st_size > 0
    probe = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", str(output)], capture_output=True, text=True
    )
    assert "Video: h264" in probe.stderr
    assert "Audio: aac" in probe.stderr


@pytest.mark.parametrize(
    "audio,rate,message",
    [
        ([0.0], None, "sampling rate must be positive"),
        ([], 48000, "invalid audio samples"),
        ([[float("nan")]], 48000, "invalid audio samples"),
        ([[0.0]] * 9, 48000, "invalid audio samples"),
    ],
)
def test_invalid_audio_is_rejected(monkeypatch, tmp_path, audio, rate, message):
    import numpy as np

    from rapid_mlx.runtime import video_lane
    from rapid_mlx.video import encoding
    from rapid_mlx.video.encoding import VideoEncodingError, encode_rgb_video

    monkeypatch.setattr(video_lane, "_resolve_ffmpeg", lambda: "/fake/ffmpeg")
    # Validation must reject the audio before any subprocess is started.
    monkeypatch.setattr(
        encoding.subprocess,
        "run",
        lambda *a, **k: pytest.fail("afconvert must not run for invalid audio"),
    )
    monkeypatch.setattr(
        encoding.subprocess,
        "Popen",
        lambda *a, **k: pytest.fail("ffmpeg must not run for invalid audio"),
    )
    with pytest.raises(VideoEncodingError, match=message):
        encode_rgb_video(
            np.zeros((1, 2, 2, 3), dtype=np.uint8),
            tmp_path / "out.mp4",
            24,
            audio=audio,
            sampling_rate=rate,
        )


def test_runtime_failure_is_mapped(monkeypatch, tmp_path):
    module = ModuleType("mlx_vlm.generate")
    module.VideoGenerationRequest = lambda **kwargs: kwargs
    module.load_video_generation_model = lambda *args, **kwargs: SimpleNamespace(
        workflow="fl2va", pipeline=fake_pipeline()
    )

    def fail(*args, **kwargs):
        raise OSError("private details")

    module.generate_video = fail
    monkeypatch.setitem(sys.modules, "mlx_vlm.generate", module)
    engine = MiniMaxH3VideoEngine(str(tmp_path))
    with pytest.raises(MiniMaxH3BackendError, match="OSError") as exc:
        engine.generate(
            prompt="test",
            output_path=tmp_path / "out.mp4",
            width=768,
            height=512,
            num_frames=124,
            fps=24,
            seed=7,
            image=None,
        )
    assert "private details" not in str(exc.value)


def test_runtime_missing_is_reported(monkeypatch):
    from rapid_mlx.runtime import video_lane

    monkeypatch.setattr(video_lane.importlib.util, "find_spec", lambda name: None)
    issue = video_lane.video_runtime_issue("minimax-h3", provision=False)
    assert issue is not None
    assert "mlx-vlm" in issue.detail
    assert issue.marker_reason == "runtime_dependency_missing"


def test_pinned_component_cache_completeness(monkeypatch, tmp_path):
    import huggingface_hub.constants

    from rapid_mlx._download_gate import is_repo_cached
    from rapid_mlx.video.minimax_h3 import REPO, REVISION

    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path))
    snapshot = tmp_path / "models--MiniMaxAI--MiniMax-H3" / "snapshots" / REVISION
    for name in FILES:
        path = snapshot / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    from rapid_mlx.video.minimax_h3 import TURBO_FILE, TURBO_REPO, TURBO_REVISION

    assert not is_repo_cached(REPO)
    turbo = (
        tmp_path
        / f"models--{TURBO_REPO.replace('/', '--')}"
        / "snapshots"
        / TURBO_REVISION
        / TURBO_FILE
    )
    turbo.parent.mkdir(parents=True)
    turbo.write_bytes(b"x")
    assert is_repo_cached(REPO)
    turbo.unlink()
    assert not is_repo_cached(REPO)
    turbo.write_bytes(b"x")
    (snapshot / FILES[-1]).unlink()
    assert not is_repo_cached(REPO)


def test_load_and_generation_share_one_thread(monkeypatch, tmp_path):
    # MLX streams are thread-local: requests arriving on different threads
    # must still load and generate on the thread that owns the pipeline.
    module = ModuleType("mlx_vlm.generate")
    threads = []

    class Request:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    def load(path, workflow):
        threads.append(threading.current_thread().name)
        return SimpleNamespace(workflow=workflow, pipeline=fake_pipeline())

    def generate(model, request):
        threads.append(threading.current_thread().name)
        return fake_result()

    module.VideoGenerationRequest = Request
    module.generate_video = generate
    module.load_video_generation_model = load
    monkeypatch.setitem(sys.modules, "mlx_vlm.generate", module)
    engine = MiniMaxH3VideoEngine(str(tmp_path))
    args = dict(
        prompt="test",
        output_path=tmp_path / "out.mp4",
        width=768,
        height=512,
        num_frames=124,
        fps=24,
        seed=7,
        image=None,
    )
    errors = []

    def call():
        try:
            engine.generate(**args)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    for _ in range(2):
        caller = threading.Thread(target=call)
        caller.start()
        caller.join()
    engine.stop()
    assert not errors
    assert len(threads) == 3
    assert len(set(threads)) == 1
    assert threads[0].startswith("mlx-minimax-h3")


def test_stop_cancels_queued_job_with_backend_error(monkeypatch, tmp_path):
    # A job still queued behind a running one when stop() shuts the executor
    # down must surface as the engine's cancellation error.
    module = ModuleType("mlx_vlm.generate")
    module.VideoGenerationRequest = lambda **kwargs: SimpleNamespace(**kwargs)
    module.load_video_generation_model = lambda *args, **kwargs: SimpleNamespace(
        workflow="fl2va", pipeline=fake_pipeline()
    )
    running, release = threading.Event(), threading.Event()

    def generate_video(model, request):
        running.set()
        release.wait(5)
        return fake_result()

    module.generate_video = generate_video
    monkeypatch.setitem(sys.modules, "mlx_vlm.generate", module)
    engine = MiniMaxH3VideoEngine(str(tmp_path))
    args = dict(
        prompt="test",
        width=768,
        height=512,
        num_frames=124,
        fps=24,
        seed=7,
        image=None,
    )
    errors = {}

    def run(name):
        try:
            engine.generate(output_path=tmp_path / f"{name}.mp4", **args)
        except BaseException as exc:  # noqa: BLE001 - record the exact type
            errors[name] = exc

    first = threading.Thread(target=run, args=("running",))
    first.start()
    assert running.wait(5)
    second = threading.Thread(target=run, args=("queued",))
    second.start()
    while engine._executor._work_queue.qsize() == 0 and second.is_alive():
        threading.Event().wait(0.01)
    engine.stop()
    release.set()
    second.join(5)
    first.join(5)
    assert isinstance(errors["queued"], MiniMaxH3BackendError)


def test_capabilities():
    from rapid_mlx.routes import video

    caps = video._video_capabilities(
        SimpleNamespace(
            model_name="MiniMaxAI/MiniMax-H3", video_family="minimax-h3", native_fps=24
        )
    )
    assert caps["modes"] == ["text-to-video", "image-to-video"]
    assert caps["limits"]["frames"] == {
        "minimum": 124,
        "maximum": 345,
        "step": 17,
        "offset": 5,
    }
    assert caps["limits"]["seconds"]["default"] == 5
    assert caps["limits"]["size"]["width"]["multiple_of"] == 32


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "overrides,detail",
    [
        ({"fps": 25}, "fixed at 24"),
        ({"seconds": "4"}, "between 5 and 15"),
        ({"negative_prompt": "bad"}, "negative_prompt or guidance_scale"),
        ({"guidance_scale": 2.0}, "negative_prompt or guidance_scale"),
        ({"conditioning_strength": 0.5}, "supported by LTX only"),
        ({"frames": 125}, "17n+5"),
        ({}, "queue is full"),
    ],
)
async def test_route_validation(monkeypatch, tmp_path, overrides, detail):
    from fastapi import HTTPException

    from rapid_mlx.routes import video

    video.configure_video_jobs(tmp_path / "jobs")
    video.start_video_jobs()
    monkeypatch.setattr(video, "_MAX_JOBS", 0)
    monkeypatch.setattr(
        video,
        "_video_engine",
        lambda: SimpleNamespace(
            model_name="MiniMaxAI/MiniMax-H3", video_family="minimax-h3", native_fps=24
        ),
    )
    args = dict(
        prompt="test",
        model="minimax-h3",
        seconds=None,
        size="256x256",
        seed=42,
        fps=None,
        frames=None,
        guidance_scale=None,
        conditioning_strength=None,
        negative_prompt=None,
        input_reference=None,
    )
    args.update(overrides)
    try:
        with pytest.raises(HTTPException) as exc:
            await video.create_video(**args)
    finally:
        video.configure_video_jobs(None)
        video.start_video_jobs()
    assert detail in exc.value.detail


def test_pull_minimax(monkeypatch):
    import argparse

    from rapid_mlx import cli
    from rapid_mlx.runtime import video_lane
    from rapid_mlx.video.minimax_h3 import REVISION

    calls = []
    monkeypatch.setattr(video_lane, "video_runtime_issue", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_pull_repository", lambda *a, **k: calls.append((a, k)))
    cli.pull_command(argparse.Namespace(model="minimax-h3", bits=None, format=None))
    assert calls[0][1] == {
        "allow_patterns_override": list(FILES),
        "revision_override": REVISION,
    }
    from rapid_mlx.video.minimax_h3 import TURBO_FILE, TURBO_REPO, TURBO_REVISION

    assert calls[1][0][0].model == TURBO_REPO
    assert calls[1][1] == {
        "allow_patterns_override": [TURBO_FILE],
        "revision_override": TURBO_REVISION,
    }
    for selector in (dict(bits=4, format=None), dict(bits=None, format="q4")):
        with pytest.raises(ValueError, match="does not support"):
            cli.pull_command(argparse.Namespace(model="minimax-h3", **selector))
    assert len(calls) == 2


def test_runtime_issue_probe_failure_and_missing_ffmpeg(monkeypatch):
    from rapid_mlx.runtime import video_lane

    def fail(_):
        raise ValueError("bad spec")

    monkeypatch.setattr(video_lane.importlib.util, "find_spec", fail)
    assert (
        video_lane.video_runtime_issue("minimax-h3").marker_reason
        == "runtime_dependency_missing"
    )
    spec = SimpleNamespace(submodule_search_locations=["fake"])
    monkeypatch.setattr(video_lane.importlib.util, "find_spec", lambda _: spec)
    monkeypatch.setattr(video_lane.PathFinder, "find_spec", lambda *args: spec)
    monkeypatch.setattr(video_lane, "_resolve_ffmpeg", lambda: None)
    assert "ffmpeg" in video_lane.video_runtime_issue("minimax-h3").detail
    monkeypatch.setattr(video_lane, "_resolve_ffmpeg", lambda: "/fake/ffmpeg")
    assert video_lane.video_runtime_issue("minimax-h3") is None


@pytest.mark.asyncio
async def test_runtime_adapter(monkeypatch, tmp_path):
    from rapid_mlx.runtime import video_lane

    engine = video_lane.VideoEngine("minimax-h3")
    calls = []
    fake = SimpleNamespace(
        validate_request=lambda **kw: calls.append(("validate", kw)),
        generate=lambda **kw: calls.append(("generate", kw)),
        stop=lambda: calls.append(("stop", {})),
    )
    engine._minimax_engine.stop()
    engine._minimax_engine = fake
    args = dict(
        prompt="test",
        output_path=tmp_path / "out.mp4",
        width=768,
        height=512,
        num_frames=124,
        fps=24,
        seed=1,
        image=None,
    )
    video_lane.validate_video_request(
        engine, width=768, height=512, num_frames=124, image=None
    )
    for control in ("negative_prompt", "guidance_scale", "conditioning_strength"):
        with pytest.raises(video_lane.VideoRuntimeError, match="conditioning controls"):
            engine.generate(**args, **{control: 1})
    engine.generate(**args)
    assert calls[0][0] == "validate" and calls[1][0] == "generate"

    def fail(**kwargs):
        raise MiniMaxH3RequestError("invalid")

    fake.generate = fail
    with pytest.raises(video_lane.VideoRuntimeError, match="invalid") as exc:
        engine.generate(**args)
    assert isinstance(exc.value.__cause__, MiniMaxH3RequestError)
    await engine.stop()
    assert calls[-1][0] == "stop"


def test_backend_validation_and_missing_output(monkeypatch, tmp_path):

    module = ModuleType("mlx_vlm.generate")
    module.VideoGenerationRequest = lambda **kwargs: SimpleNamespace(**kwargs)
    module.load_video_generation_model = lambda *a, **k: SimpleNamespace(
        pipeline=fake_pipeline()
    )
    module.generate_video = lambda *a, **k: fake_result()
    from rapid_mlx.video import encoding

    monkeypatch.setattr(encoding, "encode_rgb_video", lambda *a, **k: None)
    monkeypatch.setitem(sys.modules, "mlx_vlm.generate", module)
    cache_calls = []
    mlx = ModuleType("mlx")
    core = ModuleType("mlx.core")
    core.clear_cache = lambda: cache_calls.append(True)
    mlx.core = core
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    engine = MiniMaxH3VideoEngine(str(tmp_path))
    args = dict(
        prompt="test",
        output_path=tmp_path / "out.mp4",
        width=768,
        height=512,
        num_frames=124,
        fps=24,
        seed=7,
        image=None,
    )
    with pytest.raises(MiniMaxH3RequestError, match="24 fps"):
        engine.generate(**(args | {"fps": 12}))
    with pytest.raises(MiniMaxH3RequestError, match="first image"):
        engine.generate(**args, last_image=tmp_path / "last.png")
    loaded = []
    with pytest.raises(MiniMaxH3BackendError, match="without an MP4"):
        engine.generate(**args, on_loaded=lambda: loaded.append(True))
    engine.stop()
    assert loaded == [True] and cache_calls == [True]
    assert not args["output_path"].exists()


def test_clear_cache_without_mlx(monkeypatch):
    from rapid_mlx.video import minimax_h3

    monkeypatch.setitem(sys.modules, "mlx.core", None)
    assert minimax_h3._clear_mlx_cache() is None


@pytest.mark.asyncio
async def test_openai_720p_size_renders_on_the_32_aligned_canvas(monkeypatch):
    import asyncio

    from rapid_mlx.routes import video

    captured = {}

    class FakeEngine:
        model_name = "MiniMaxAI/MiniMax-H3"
        video_family = "minimax-h3"
        native_fps = 24

        def generate(self, *, output_path, **kwargs):
            captured.update(kwargs)
            output_path.write_bytes(b"mp4")

    monkeypatch.setattr(video, "_video_engine", lambda: FakeEngine())
    created = await video.create_video(
        prompt="landscape",
        model="minimax-h3",
        seconds="5",
        size="1280x720",
        seed=1,
        input_reference=None,
    )
    for _ in range(200):
        if (await video.retrieve_video(created["id"]))["status"] == "completed":
            break
        await asyncio.sleep(0.01)
    assert (captured["width"], captured["height"]) == (1280, 736)
    assert (captured["output_width"], captured["output_height"]) == (1280, 720)
    await video.delete_video(created["id"])


def test_turbo_merge_and_metadata(monkeypatch, tmp_path):
    import numpy as np
    from safetensors.numpy import save_file

    from rapid_mlx.video import minimax_h3

    path = tmp_path / "lora.safetensors"
    arrays = {
        f"{name}.lora_{side}.default.weight": np.full(
            (128, 1) if side == "A" else (1, 128), value, dtype=np.float32
        )
        for name in ("one", "two")
        for side, value in (("A", 2), ("B", 3))
    }
    save_file(arrays, str(path), metadata={"alpha": "8"})
    import mlx.core as mx

    mx.load = lambda _: arrays
    mx.float32 = np.float32
    calls = []
    mx.eval = lambda value: calls.append("eval")
    mx.clear_cache = lambda: calls.append("clear")
    utils = ModuleType("mlx.utils")
    utils.tree_flatten = lambda p: list(p.items())
    monkeypatch.setitem(sys.modules, "mlx.utils", utils)
    weights = {
        "one.weight": np.array([[1]], dtype=np.float16),
        "two.weight": np.array([[2]], dtype=np.float16),
    }

    class Transformer:
        def parameters(self):
            return weights

        def load_weights(self, items, strict):
            assert len(items) == 1 and strict is False
            key, value = items[0]
            calls.append(key)
            weights[key] = value

    real_merge_turbo_lora(Transformer(), path)
    assert weights["one.weight"].item() == 49
    assert weights["two.weight"].item() == 50
    assert calls == ["eval", "one.weight", "clear", "eval", "two.weight", "clear"]
    for metadata in ({}, {"alpha": "0"}, {"alpha": "bad"}):
        save_file(arrays, str(path), metadata=metadata)
        with pytest.raises(minimax_h3.MiniMaxH3BackendError, match="alpha"):
            real_merge_turbo_lora(Transformer(), path)
    unrelated = {"other.weight": np.zeros((1, 1), dtype=np.float32)}
    save_file(unrelated, str(path), metadata={"alpha": "8"})
    mx.load = lambda _: unrelated
    with pytest.raises(minimax_h3.MiniMaxH3BackendError, match="no adapter"):
        real_merge_turbo_lora(Transformer(), path)


def test_turbo_load_recipe(monkeypatch, tmp_path):
    from rapid_mlx import _mirror
    from rapid_mlx.video import minimax_h3

    calls = []
    monkeypatch.setattr(
        _mirror,
        "pinned_snapshot_download",
        lambda *a, **k: calls.append((a, k)) or str(tmp_path),
    )
    monkeypatch.setattr(
        minimax_h3, "_merge_turbo_lora", lambda *a: calls.append(("merge", a))
    )
    module = ModuleType("mlx_vlm.generate")
    pipeline = SimpleNamespace(
        transformer=object(),
        scheduler=SimpleNamespace(
            set_shift=lambda value: calls.append(("shift", value))
        ),
    )
    module.load_video_generation_model = lambda *a, **k: SimpleNamespace(
        pipeline=pipeline
    )
    module.VideoGenerationRequest = lambda **kw: SimpleNamespace(**kw)
    module.generate_video = lambda model, request: (
        calls.append(("request", request)) or fake_result()
    )
    monkeypatch.setitem(sys.modules, "mlx_vlm.generate", module)
    engine = MiniMaxH3VideoEngine(str(tmp_path))
    engine.generate(
        prompt="x",
        output_path=tmp_path / "out",
        width=768,
        height=512,
        num_frames=124,
        fps=24,
        seed=1,
        image=None,
    )
    engine.stop()
    assert calls[0] == (
        (minimax_h3.TURBO_REPO, minimax_h3.TURBO_REVISION),
        {"allow_patterns": [minimax_h3.TURBO_FILE]},
    )
    assert calls[1] == (
        "merge",
        (pipeline.transformer, tmp_path / minimax_h3.TURBO_FILE),
    )
    assert calls[2] == ("shift", 6.0)
    assert calls[3][1].steps == 9


def test_turbo_size_payload(monkeypatch):
    import huggingface_hub

    from rapid_mlx.video import minimax_h3 as h3
    from scripts.gen_model_sizes import _size_one

    calls = []

    def info(repo, *, revision, files_metadata, timeout):
        calls.append((repo, revision))
        files = h3.FILES if repo == h3.REPO else (h3.TURBO_FILE,)
        return SimpleNamespace(
            siblings=[SimpleNamespace(rfilename=f, size=2) for f in files]
        )

    monkeypatch.setattr(huggingface_hub, "model_info", info)
    assert _size_one(h3.REPO) == (h3.REPO, 2 * (len(h3.FILES) + 1))
    assert calls == [(h3.REPO, h3.REVISION), (h3.TURBO_REPO, h3.TURBO_REVISION)]
