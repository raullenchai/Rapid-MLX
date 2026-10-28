import fnmatch
import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest

from rapid_mlx.model_aliases import resolve_profile, subfolder_allow_patterns
from rapid_mlx.video.minimax_h3 import (
    FILES,
    MiniMaxH3BackendError,
    MiniMaxH3RequestError,
    MiniMaxH3VideoEngine,
)


@pytest.fixture(autouse=True)
def fake_mlx_cache(monkeypatch):
    # Existing generation tests must not import the real MLX runtime.
    mlx = ModuleType("mlx")
    core = ModuleType("mlx.core")
    core.clear_cache = lambda: None
    mlx.core = core
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", core)


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
        return SimpleNamespace(workflow=workflow)

    def generate(model, request, *, output_path):
        calls.append((model.workflow, request))
        output_path.write_bytes(b"mp4")

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


def test_runtime_failure_is_mapped(monkeypatch, tmp_path):
    module = ModuleType("mlx_vlm.generate")
    module.VideoGenerationRequest = lambda **kwargs: kwargs
    module.load_video_generation_model = lambda *args, **kwargs: SimpleNamespace(
        workflow="fl2va"
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
    assert is_repo_cached(REPO)
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
        return SimpleNamespace(workflow=workflow)

    def generate(model, request, *, output_path):
        threads.append(threading.current_thread().name)
        output_path.write_bytes(b"mp4")

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
        workflow="fl2va"
    )
    running, release = threading.Event(), threading.Event()

    def generate_video(model, request, output_path):
        running.set()
        release.wait(5)
        output_path.write_bytes(b"mp4")

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
    for selector in (dict(bits=4, format=None), dict(bits=None, format="q4")):
        with pytest.raises(ValueError, match="does not support"):
            cli.pull_command(argparse.Namespace(model="minimax-h3", **selector))
    assert len(calls) == 1


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
    module.load_video_generation_model = lambda *a, **k: SimpleNamespace()
    module.generate_video = lambda *a, **k: None
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
