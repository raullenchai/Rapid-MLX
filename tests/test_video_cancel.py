"""Cancellation contracts use fake pipelines and never materialize model weights."""

import asyncio
import json
import subprocess
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from fastapi import HTTPException

from rapid_mlx.routes import video
from rapid_mlx.runtime import video_lane
from rapid_mlx.video import (
    encoding,
    ltx25,
    minimax_h3,
    pika_h3_flex,
    wan,
    wan_diffusers,
)
from rapid_mlx.video.cancel import VideoCancelToken, VideoGenerationCancelled, step_iter
from rapid_mlx.video.engine import VideoGenerationEngine


@pytest.fixture
def cache_clears(monkeypatch):
    calls = []
    monkeypatch.setattr(minimax_h3, "_clear_mlx_cache", lambda: calls.append(True))
    return calls


def test_token():
    token = VideoCancelToken()
    assert not token.cancelled
    assert not token.wait(0)
    assert list(step_iter(range(3), token)) == [0, 1, 2]
    token.cancel()
    token.cancel()
    assert token.cancelled and token.wait(0)
    with pytest.raises(VideoGenerationCancelled):
        token.check()


@pytest.mark.parametrize("family", ["h3", "turbo", "pika-fl2va", "pika-ref2va"])
def test_h3_steps_cleanup_and_reuse(monkeypatch, tmp_path, cache_clears, family):
    token = VideoCancelToken()
    steps = []
    loads = []
    module = ModuleType("mlx_vlm.generate")
    module.VideoGenerationRequest = lambda **kw: SimpleNamespace(**kw)

    def load(*args, **kwargs):
        loads.append(True)
        return SimpleNamespace(pipeline=object())

    def generate(model, request):
        output.write_bytes(b"partial")
        for i in range(10):
            request.progress_callback("denoise", i, 10, None)
            steps.append(i)
            if i == 2:
                token.cancel()
        return SimpleNamespace(
            frames=np.zeros((1, 1, 1, 3)), fps=24, audio=None, sampling_rate=None
        )

    module.load_video_generation_model = load
    module.generate_video = generate
    monkeypatch.setitem(sys.modules, "mlx_vlm.generate", module)
    references_module = ModuleType("mlx_vlm.generate.video_generation")
    references_module.VideoReference = lambda *a: a
    monkeypatch.setitem(sys.modules, references_module.__name__, references_module)
    cls = (
        pika_h3_flex.PikaH3FlexVideoEngine
        if family.startswith("pika")
        else minimax_h3.MiniMaxH3VideoEngine
    )
    engine = cls(str(tmp_path))
    monkeypatch.setattr(engine, "_prepare_model", lambda _: None)
    monkeypatch.setattr(engine, "_prepare_request", lambda _: None)
    monkeypatch.setattr(engine, "_ensure_partition", lambda _: None)
    monkeypatch.setattr(
        encoding,
        "encode_rgb_video",
        lambda frames, path, fps, **kw: path.write_bytes(b"mp4"),
    )
    output = tmp_path / "output.mp4"
    args = dict(
        prompt="test",
        output_path=output,
        width=768,
        height=512,
        num_frames=124,
        fps=24,
        seed=1,
        image=None,
    )
    if family == "pika-ref2va":
        args["references"] = [("image", tmp_path / "image.png")]
    try:
        with pytest.raises(VideoGenerationCancelled):
            engine.generate(**args, cancel_token=token)
        assert steps == [0, 1, 2]
        assert not output.exists()
        assert cache_clears
        engine.generate(**args, cancel_token=VideoCancelToken())
        assert output.read_bytes() == b"mp4"
        assert len(loads) == 1
    finally:
        engine.stop()


@pytest.mark.parametrize("diffusers", [False, True])
def test_wan_scoped_steps_cleanup_and_reuse(
    monkeypatch, tmp_path, cache_clears, diffusers
):
    token = VideoCancelToken()
    steps = []
    module = ModuleType("mlx_video.generate_wan")
    module.__dict__.update(
        tqdm=lambda iterable, **kw: iterable, steps_seen=steps, token=token, Path=Path
    )
    exec(
        "def generate_video(**kw):\n    Path(kw['output_path']).write_bytes(b'partial')\n    for i in tqdm(range(10)):\n        steps_seen.append(i)\n        if i == 2: token.cancel()\n    Path(kw['output_path']).write_bytes(b'mp4')\n",
        module.__dict__,
    )
    original = module.tqdm
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(wan_diffusers, "is_diffusers_wan21_layout", lambda _: diffusers)
    monkeypatch.setattr(
        wan_diffusers, "_identifies_as_diffusers_wan21", lambda _: False
    )
    engine = object.__new__(wan.WanVideoEngine)
    engine.model_path = tmp_path
    engine.config = {"model_type": "t2v"}
    engine.steps = engine.loras = engine.loras_high = engine.loras_low = None
    engine.scheduler = "unipc"
    engine.tiling = "auto"
    output = tmp_path / "output.mp4"
    args = dict(
        prompt="test",
        output_path=output,
        width=256,
        height=256,
        num_frames=25,
        seed=1,
        image=None,
    )
    with pytest.raises(VideoGenerationCancelled):
        engine.generate(**args, cancel_token=token)
    assert steps == [0, 1, 2]
    assert module.tqdm is original
    assert not output.exists() and cache_clears
    engine.generate(**args, cancel_token=VideoCancelToken())
    assert output.read_bytes() == b"mp4"


def test_ltx23_scoped_steps_cleanup_and_reuse(monkeypatch, tmp_path, cache_clears):
    token = VideoCancelToken()
    steps = []
    package = ModuleType("mlx_video")
    av = ModuleType("mlx_video.generate_av")
    other = ModuleType("mlx_video.generate")
    av.tqdm = other.tqdm = lambda iterable, **kw: iterable
    originals = (av.tqdm, other.tqdm)

    def generate(**kw):
        output.write_bytes(b"partial")
        for i in av.tqdm(range(10), disable=True):
            steps.append(i)
            if i == 2:
                token.cancel()
        output.write_bytes(b"mp4")

    package.generate_video_with_audio = generate
    for module in (package, av, other):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(video_lane, "_resolve_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(video_lane.VideoEngine, "_emit_model_served", lambda _: None)
    engine = video_lane.VideoEngine("ltx-2.3-mlx-q4")
    output = tmp_path / "output.mp4"
    args = dict(
        prompt="test",
        output_path=output,
        width=256,
        height=256,
        num_frames=25,
        fps=24,
        seed=1,
        image=None,
    )
    with pytest.raises(VideoGenerationCancelled):
        engine.generate(**args, cancel_token=token)
    assert steps == [0, 1, 2]
    assert (av.tqdm, other.tqdm) == originals
    assert not output.exists() and cache_clears
    engine.generate(**args, cancel_token=VideoCancelToken())
    assert output.read_bytes() == b"mp4"


@pytest.mark.parametrize("extend", [False, True])
def test_ltx25_terminates_child_and_reuses_engine(
    monkeypatch, tmp_path, cache_clears, extend
):
    monkeypatch.setattr(
        ltx25,
        "ltx25_runtime",
        lambda: ltx25.LTX25Runtime(
            state="trusted", source="vendored", executable=sys.executable
        ),
    )
    monkeypatch.setattr(
        "rapid_mlx.video_snapshots.resolve_model_dir", lambda _: tmp_path
    )
    token = VideoCancelToken()
    children = []
    real_popen = subprocess.Popen

    def spawn(command, **kwargs):
        path = command[command.index("--output") + 1]
        code = "import sys,time; from pathlib import Path; sys.stdin.read(); Path(sys.argv[1]).write_bytes(b'mp4'); "
        if not children:
            code += "time.sleep(60)"
        process = real_popen([sys.executable, "-c", code, path], **kwargs)
        children.append(process)
        if len(children) == 1:
            threading.Timer(0.1, token.cancel).start()
        return process

    monkeypatch.setattr(ltx25.subprocess, "Popen", spawn)
    engine = ltx25.LTX25VideoEngine("ltx-2.5-mlx-q8")
    output = tmp_path / "output.mp4"
    args = dict(prompt="test", output_path=output, seed=1)
    if extend:
        function = engine.extend
        args.update(source_video=tmp_path / "source.mp4", extend_frames=8)
    else:
        function = engine.generate
        args.update(width=256, height=256, num_frames=25, fps=24, image=None)
    with pytest.raises(VideoGenerationCancelled):
        function(**args, cancel_token=token)
    assert children[0].poll() is not None
    assert engine._process is None and not engine._stopping
    assert not output.exists() and cache_clears
    assert list(tmp_path.iterdir()) == []
    function(**args, cancel_token=VideoCancelToken())
    assert output.read_bytes() == b"mp4"
    engine.stop()


def test_ref2va_download_progress_cancel(monkeypatch, tmp_path):
    import huggingface_hub

    from rapid_mlx import _mirror, model_downloads

    token = VideoCancelToken()
    updates = []
    monkeypatch.setattr(
        huggingface_hub, "try_to_load_from_cache", lambda *a, **kw: None
    )
    monkeypatch.setattr(model_downloads, "disabled", lambda: False)
    monkeypatch.setattr(_mirror, "_cache_disk_space", lambda: (tmp_path, 10**15))
    monkeypatch.setattr(
        _mirror,
        "download_with_mirror_fallback",
        lambda *a, **kw: False,
    )

    def download(*a, tqdm_class, **kw):
        with tqdm_class(total=10, disable=True) as bar:
            for i in range(10):
                if i == 2:
                    token.cancel()
                bar.update(1)
                updates.append(i)
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    with pytest.raises(VideoGenerationCancelled):
        pika_h3_flex.ensure_ref2va_assets(cancel_token=token)
    assert updates == [0, 1]


def test_cog_steps_cleanup_and_reuse(monkeypatch, tmp_path, cache_clears):
    token = VideoCancelToken()
    steps = []
    core = ModuleType("mlx.core")
    core.zeros, core.ones, core.float32 = np.zeros, np.ones, np.float32
    core.eval = lambda _: None
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    package = ModuleType("mlx")
    package.core = core
    monkeypatch.setitem(sys.modules, "mlx", package)

    def pipe(*, video, cancel_check=None, **kw):
        for i in range(10):
            if cancel_check:
                cancel_check()
            steps.append(i)
            if i == 2:
                token.cancel()
        return video

    engine = VideoGenerationEngine("fake", output_dir=tmp_path)
    monkeypatch.setattr(engine, "_load_sync", lambda: pipe)
    monkeypatch.setattr(
        encoding, "encode_rgb_video", lambda frames, path, fps: path.write_bytes(b"mp4")
    )
    output = tmp_path / "output.mp4"
    args = dict(
        output_path=output, prompt="test", width=32, height=32, frames=5, fps=8, seed=1
    )
    try:
        with pytest.raises(VideoGenerationCancelled):
            engine.generate_sync(**args, cancel_token=token)
        assert steps == [0, 1, 2]
        assert not output.exists() and cache_clears
        assert not list(tmp_path.iterdir())
        engine.generate_sync(**args, cancel_token=VideoCancelToken())
        assert output.read_bytes() == b"mp4"
    finally:
        asyncio.run(engine.close())


@pytest.fixture
def route_store(monkeypatch, tmp_path):
    monkeypatch.setattr(video, "_jobs", {})
    monkeypatch.setattr(video, "_tasks", {})
    monkeypatch.setattr(video, "_cancel_tokens", {})
    monkeypatch.setattr(video, "_jobs_root", tmp_path)
    monkeypatch.setattr(video, "_jobs_are_persistent", False)
    return tmp_path


def start_job(root, engine, name):
    job = video._VideoJob(
        id=name, model="fake", prompt="test", seconds="1", size="256x256"
    )
    (root / name).mkdir()
    video._jobs[name] = job
    task = asyncio.create_task(
        video._run_job(
            job,
            engine=engine,
            width=256,
            height=256,
            num_frames=25,
            fps=24,
            seed=1,
            image_path=None,
            negative_prompt=None,
            guidance_scale=None,
            conditioning_strength=None,
        )
    )
    video._tasks[name] = task
    return job, task


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [VideoGenerationCancelled, RuntimeError])
async def test_running_delete_idempotent_cleanup_gate_and_reuse(route_store, error):
    started, release = threading.Event(), threading.Event()
    calls = []

    class Engine:
        def generate(self, *, output_path, cancel_token, **kwargs):
            calls.append(True)
            output_path.write_bytes(b"partial")
            if len(calls) == 1:
                started.set()
                assert cancel_token.wait(2)
                assert release.wait(2)
                raise error("stop")
            output_path.write_bytes(b"mp4")

    engine = Engine()
    job, task = start_job(route_store, engine, "first")
    assert await asyncio.to_thread(started.wait, 2)
    second, next_task = start_job(route_store, engine, "second")
    for _ in range(2):
        response = await asyncio.wait_for(video.delete_video(job.id), 0.5)
        assert response.status_code == 202
        body = json.loads(response.body)
        assert body["status"] == "in_progress" and body["cancel_requested"] is True
        assert "cancel_token" not in body
    release.set()
    await asyncio.wait_for(asyncio.gather(task, next_task), 3)
    assert job.status == "failed" and job.generation_finished
    assert job.error["code"] == "video_generation_cancelled"
    assert not (route_store / job.id).exists()
    assert (await video.retrieve_video(job.id))["status"] == "failed"
    assert second.status == "completed" and len(calls) == 2
    assert not video._cancel_tokens
    for current in (job, second):
        assert (await video.delete_video(current.id))["deleted"]
        with pytest.raises(HTTPException) as exc:
            await video.retrieve_video(current.id)
        assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_queued_delete_unchanged(route_store):
    gate = video._generation_gate_for_current_loop()
    async with gate:
        job, task = start_job(route_store, object(), "queued")
        await asyncio.sleep(0)
        result = await video.delete_video(job.id)
    assert result["deleted"] and result["error"]["code"] == "video_generation_cancelled"
    assert task.cancelled() and not (route_store / job.id).exists()
    with pytest.raises(HTTPException) as exc:
        await video.retrieve_video(job.id)
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_successful_return_wins_cancel_race(route_store):
    class Engine:
        def generate(self, *, output_path, cancel_token, **kwargs):
            cancel_token.cancel()
            output_path.write_bytes(b"mp4")

    job, task = start_job(route_store, Engine(), "race")
    await task
    assert job.status == "completed" and Path(job.output_path).exists()


@pytest.mark.asyncio
async def test_shutdown_signals_running_job(route_store, monkeypatch):
    monkeypatch.setattr(video, "_accepting_jobs", True)
    started = threading.Event()

    class Engine:
        def generate(self, *, cancel_token, **kwargs):
            started.set()
            assert cancel_token.wait(2)
            cancel_token.check()

    job, task = start_job(route_store, Engine(), "shutdown")
    assert await asyncio.to_thread(started.wait, 2)
    await video.shutdown_video_jobs(timeout=1)
    assert task.done() and job.generation_finished
    assert job.error["code"] == "video_generation_cancelled"
    assert not (route_store / job.id).exists()


def test_cog_pipeline_checks_real_denoise_loop():
    """Execute the pipeline method with array/transformer fakes, without MLX."""
    import ast

    source = Path(
        "videox_fun_mlx/pipeline/pipeline_cogvideox_fun_inpaint.py"
    ).read_text()
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "__call__"
    )
    # The runtime annotations refer to native array types; this test supplies numpy.
    method.returns = None
    for argument in method.args.args:
        argument.annotation = None
    namespace = {
        "mx": SimpleNamespace(
            array=np.array,
            zeros=np.zeros,
            mean=np.mean,
            concatenate=np.concatenate,
            broadcast_to=np.broadcast_to,
            eval=lambda _: None,
            random=SimpleNamespace(normal=lambda shape: np.zeros(shape)),
        )
    }
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), "pipeline", "exec"),
        namespace,
    )
    token = VideoCancelToken()
    steps = []
    decodes = []
    latents = np.zeros((1, 1, 1, 1, 1))

    def forward(**kw):
        steps.append(len(steps))
        if len(steps) == 3:
            token.cancel()
        return latents.transpose(0, 1, 4, 2, 3)

    pipe = SimpleNamespace(
        vae=SimpleNamespace(
            encode=lambda _: SimpleNamespace(mode=lambda: latents),
            scaling_factor=1,
            decode=lambda _: decodes.append(True),
        ),
        scheduler=SimpleNamespace(
            set_timesteps=lambda _: None,
            timesteps=range(10),
            add_noise=lambda *a: latents,
            step=lambda pred, t, current: current,
        ),
        transformer=forward,
        _prepare_rotary_embeddings=lambda *a: None,
    )
    with pytest.raises(VideoGenerationCancelled):
        namespace["__call__"](
            pipe,
            video=np.zeros((1, 1, 1, 1, 1)),
            mask=np.ones((1, 1, 1, 1, 1)),
            prompt_embeds=np.zeros((1, 1, 1)),
            guidance_scale=1,
            cancel_check=token.check,
        )
    assert steps == [0, 1, 2]
    assert not decodes


@pytest.mark.parametrize("family", ["ltx-2.5", "minimax-h3", "wan", "cogvideox-fun"])
def test_lane_passes_token_and_cleans_before_crop(
    monkeypatch, tmp_path, cache_clears, family
):
    token = VideoCancelToken()
    output = tmp_path / "output.mp4"
    calls = []

    def generate(*, cancel_token, **kw):
        assert cancel_token is token
        calls.append(True)
        output.write_bytes(b"mp4")
        token.cancel()

    engine = object.__new__(video_lane.VideoEngine)
    engine._generation_lock = threading.Lock()
    engine._ltx25_engine = engine._minimax_engine = engine._wan_engine = (
        engine._cog_engine
    ) = None
    attr = {
        "ltx-2.5": "_ltx25_engine",
        "minimax-h3": "_minimax_engine",
        "wan": "_wan_engine",
        "cogvideox-fun": "_cog_engine",
    }[family]
    setattr(engine, attr, SimpleNamespace(generate=generate, generate_sync=generate))
    monkeypatch.setattr(
        engine,
        "_crop_generated_output",
        lambda **kw: pytest.fail("cancelled before crop"),
    )
    args = dict(
        prompt="test",
        output_path=output,
        width=256,
        height=256,
        num_frames=25,
        fps=24,
        seed=1,
        image=None,
        cancel_token=token,
    )
    if family == "cogvideox-fun":
        # This backend has no post-generation crop stage: preserve its completed return.
        engine.generate(**args)
        assert output.exists()
    else:
        with pytest.raises(VideoGenerationCancelled):
            engine.generate(**args, output_width=240)
        assert not output.exists() and cache_clears
    assert calls and not engine._generation_lock.locked()


def test_cancel_request_is_transient_in_manifest():
    job = video._VideoJob(
        id="test",
        model="test",
        prompt="test",
        seconds="1",
        size="256x256",
        cancel_requested=True,
    )
    assert job.public()["cancel_requested"] is True
    record = video._completed_job_record(job)
    assert "cancel_requested" not in record
    assert "cancel_token" not in record


@pytest.mark.asyncio
async def test_completion_clears_concurrent_cancel(route_store):
    started = threading.Event()

    class Engine:
        def generate(self, *, output_path, cancel_token, **kwargs):
            started.set()
            assert cancel_token.wait(2)
            output_path.write_bytes(b"mp4")

    job, task = start_job(route_store, Engine(), "winner")
    assert await asyncio.to_thread(started.wait, 2)
    assert (await video.delete_video(job.id)).status_code == 202
    await asyncio.wait_for(task, 3)
    assert job.status == "completed"
    assert not job.cancel_requested
    assert "cancel_requested" not in (await video.retrieve_video(job.id))


@pytest.mark.asyncio
async def test_download_phase_progress_is_monotonic(route_store):
    snapshots = []

    class Engine:
        def generate(self, *, output_path, on_progress, **kwargs):
            for phase, done, total in [
                ("downloading", 0, 100),
                ("downloading", 50, 100),
                ("downloading", 25, 100),
                ("downloading", 100, 100),
                (None, 0, 0),
            ]:
                on_progress(phase, done, total)
                with video._jobs_lock:
                    snapshots.append(video._jobs["progress"].public())
            output_path.write_bytes(b"mp4")

    job, task = start_job(route_store, Engine(), "progress")
    await asyncio.wait_for(task, 3)
    assert [item["progress"] for item in snapshots] == [1, 10, 10, 20, 20]
    assert all(item["status"] == "in_progress" for item in snapshots)
    assert all(item["phase"] == "downloading" for item in snapshots[:-1])
    assert "phase" not in snapshots[-1] and "phase" not in job.public()


@pytest.mark.parametrize("cancel", [False, True])
def test_ref2va_mirror_byte_progress(monkeypatch, tmp_path, cancel):
    import huggingface_hub

    from rapid_mlx import _mirror

    token = VideoCancelToken()
    events = []
    monkeypatch.setattr(
        huggingface_hub, "try_to_load_from_cache", lambda *a, **kw: None
    )
    monkeypatch.setattr(_mirror, "_cache_disk_space", lambda: (tmp_path, 10**15))
    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda *a, **kw: pytest.fail("mirror bypassed"),
    )

    def mirror(repo, *, allow_patterns, on_bytes, **kwargs):
        size = pika_h3_flex.REF2VA_FILE_SIZES[allow_patterns[0]]
        on_bytes(size // 2)
        if cancel:
            token.cancel()
        on_bytes(size - size // 2)
        return True

    monkeypatch.setattr(_mirror, "download_with_mirror_fallback", mirror)
    if cancel:
        with pytest.raises(VideoGenerationCancelled):
            pika_h3_flex.ensure_ref2va_assets(
                cancel_token=token, on_progress=lambda *args: events.append(args)
            )
    else:
        pika_h3_flex.ensure_ref2va_assets(
            cancel_token=token, on_progress=lambda *args: events.append(args)
        )
        assert events[-2] == (
            "downloading",
            pika_h3_flex.REF2VA_EXTRA_BYTES,
            pika_h3_flex.REF2VA_EXTRA_BYTES,
        )
    assert events[0] == ("downloading", 0, pika_h3_flex.REF2VA_EXTRA_BYTES)
    assert events[-1] == (None, 0, 0)


def test_progress_wrapper_delegates_context_attributes_and_updates():
    from rapid_mlx.video.cancel import cancelling_tqdm

    calls = []

    class Bar:
        total = 10

        def __enter__(self):
            calls.append("enter")
            return self

        def __exit__(self, *args):
            calls.append(args)
            return True

        def update(self, n=1):
            calls.append(n)
            return "updated"

    token = VideoCancelToken()
    bar = cancelling_tqdm(Bar, token)()
    assert bar.total == 10
    with bar as entered:
        assert entered is bar
        assert bar.update(3) == "updated"
        raise ValueError("suppressed by underlying bar")
    assert calls[:2] == ["enter", 3]
    assert calls[2][0] is ValueError
    token.cancel()
    with pytest.raises(VideoGenerationCancelled):
        bar.update(1)
    assert len(calls) == 3


def test_scoped_hooks_skip_missing_module_and_restore_after_failure(monkeypatch):
    from rapid_mlx.video.cancel import scoped_step_hooks

    missing = "mlx_video.absent_cancel_test"
    monkeypatch.setitem(sys.modules, missing, None)
    module = ModuleType("fake_video_progress")
    original = module.tqdm = lambda iterable: iterable
    monkeypatch.setitem(sys.modules, module.__name__, module)
    token = VideoCancelToken()
    with (
        pytest.raises(ValueError, match="generation failed"),
        scoped_step_hooks(token, missing, module.__name__),
    ):
        assert module.tqdm is not original
        assert list(module.tqdm(range(2))) == [0, 1]
        raise ValueError("generation failed")
    assert module.tqdm is original


def test_cleanup_normalizes_backend_error_after_cancel(tmp_path, cache_clears):
    from rapid_mlx.video.cancel import cleanup_on_cancel

    token = VideoCancelToken()
    output = tmp_path / "result.mp4"
    error = RuntimeError("backend interrupted")

    @cleanup_on_cancel
    def generate(*, output_path, cancel_token):
        output_path.write_bytes(b"partial")
        output_path.with_suffix(".temp.mp4").write_bytes(b"partial mux")
        cancel_token.cancel()
        raise error

    with pytest.raises(VideoGenerationCancelled) as exc:
        generate(output_path=output, cancel_token=token)
    assert exc.value.__cause__ is error
    assert not list(tmp_path.iterdir())
    assert cache_clears == [True]


def test_lane_extend_checks_and_forwards_token(monkeypatch, tmp_path):
    token = VideoCancelToken()
    calls = []
    engine = object.__new__(video_lane.VideoEngine)
    engine._generation_lock = threading.Lock()
    engine._ltx25_engine = SimpleNamespace(extend=lambda **kw: calls.append(kw))
    output = tmp_path / "extended.mp4"
    engine.extend(
        prompt="continue",
        source_video=tmp_path / "source.mp4",
        output_path=output,
        extend_frames=8,
        seed=1,
        cancel_token=token,
    )
    assert len(calls) == 1
    assert calls[0]["cancel_token"] is token
    assert calls[0]["output_path"] == output
    assert not engine._generation_lock.locked()


@pytest.mark.parametrize("crop", [False, True])
def test_ltx23_post_generation_cancel_checks(monkeypatch, tmp_path, cache_clears, crop):
    package = ModuleType("mlx_video")
    token = VideoCancelToken()
    output = tmp_path / "result.mp4"

    def generate(**kw):
        if crop:
            output.write_bytes(b"mp4")
        token.cancel()

    package.generate_video_with_audio = generate
    monkeypatch.setitem(sys.modules, package.__name__, package)
    # These optional modules need not be installed to use the public entry point.
    for name in ("mlx_video.generate_av", "mlx_video.generate"):
        monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.setattr(video_lane, "_resolve_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(video_lane.VideoEngine, "_emit_model_served", lambda _: None)
    engine = video_lane.VideoEngine("ltx-2.3-mlx-q4")
    monkeypatch.setattr(
        engine,
        "_crop_generated_output",
        lambda **kw: pytest.fail("crop after cancellation"),
    )
    with pytest.raises(VideoGenerationCancelled):
        engine.generate(
            prompt="test",
            output_path=output,
            width=256,
            height=256,
            num_frames=25,
            fps=24,
            seed=1,
            image=None,
            output_width=240 if crop else None,
            cancel_token=token,
        )
    assert not output.exists()
    assert cache_clears == [True]


@pytest.mark.parametrize("cancel_during_load", [False, True])
def test_cog_load_callback_checks_cancel(
    monkeypatch, tmp_path, cache_clears, cancel_during_load
):
    token = VideoCancelToken()
    engine = VideoGenerationEngine("fake", output_dir=tmp_path)
    generated = tmp_path / "worker.mp4"
    output = tmp_path / "result.mp4"
    calls = []

    def load():
        calls.append("load")
        if cancel_during_load:
            token.cancel()

    def generate(**kwargs):
        assert kwargs["cancel_token"] is token
        calls.append("generate")
        generated.write_bytes(b"mp4")
        return generated

    monkeypatch.setattr(engine, "_load_sync", load)
    monkeypatch.setattr(engine, "_generate_sync", generate)
    try:
        kwargs = dict(
            output_path=output,
            cancel_token=token,
            on_loaded=lambda: calls.append("loaded"),
        )
        if cancel_during_load:
            with pytest.raises(VideoGenerationCancelled):
                engine.generate_sync(**kwargs)
            assert calls == ["load"]
            assert not output.exists() and cache_clears
        else:
            engine.generate_sync(**kwargs)
            assert calls == ["load", "loaded", "generate"]
            assert output.read_bytes() == b"mp4"
            assert not generated.exists()
    finally:
        asyncio.run(engine.close())


def test_h3_error_checks_cancel_before_wrapping(monkeypatch, tmp_path, cache_clears):
    token = VideoCancelToken()
    error = RuntimeError("native backend interrupted")
    module = ModuleType("mlx_vlm.generate")
    module.VideoGenerationRequest = lambda **kw: SimpleNamespace(**kw)
    module.load_video_generation_model = lambda *a, **kw: pytest.fail(
        "resident model reloaded"
    )
    output = tmp_path / "partial.mp4"

    def generate(*args):
        output.write_bytes(b"partial")
        token.cancel()
        raise error

    module.generate_video = generate
    monkeypatch.setitem(sys.modules, module.__name__, module)
    engine = minimax_h3.MiniMaxH3VideoEngine(str(tmp_path))
    engine._model = SimpleNamespace()
    monkeypatch.setattr(engine, "_ensure_partition", lambda _: None)
    monkeypatch.setattr(engine, "_prepare_request", lambda _: None)
    try:
        with pytest.raises(VideoGenerationCancelled):
            engine.generate(
                prompt="test",
                output_path=output,
                width=768,
                height=512,
                num_frames=124,
                fps=24,
                seed=1,
                image=None,
                cancel_token=token,
            )
        assert not output.exists() and cache_clears == [True]
        assert engine._cancel_token is None and engine._on_progress is None
    finally:
        engine.stop()


def test_ref2va_download_error_checks_cancel(monkeypatch, tmp_path):
    import huggingface_hub

    from rapid_mlx import _mirror, model_downloads

    token = VideoCancelToken()
    events = []
    monkeypatch.setattr(
        huggingface_hub, "try_to_load_from_cache", lambda *a, **kw: None
    )
    monkeypatch.setattr(model_downloads, "disabled", lambda: False)
    monkeypatch.setattr(_mirror, "_cache_disk_space", lambda: (tmp_path, 10**15))

    def download(*args, **kwargs):
        token.cancel()
        raise OSError("transfer interrupted")

    monkeypatch.setattr(_mirror, "pinned_snapshot_download", download)
    engine = pika_h3_flex.PikaH3FlexVideoEngine("pika-h3-flex")
    engine._cancel_token = token
    engine._on_progress = lambda *args: events.append(args)
    try:
        with pytest.raises(VideoGenerationCancelled):
            engine._ensure_assets("ref2va")
        assert events[0] == ("downloading", 0, pika_h3_flex.REF2VA_EXTRA_BYTES)
        assert events[-1] == (None, 0, 0)
    finally:
        engine.stop()


@pytest.mark.parametrize("diffusers", [False, True])
def test_wan_checks_cancel_after_return_without_output(
    monkeypatch, tmp_path, diffusers
):
    from contextlib import contextmanager

    token = VideoCancelToken()
    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        token.cancel()

    @contextmanager
    def runtime(root, generator):
        try:
            yield tmp_path / "view", generate
        finally:
            calls.append("runtime closed")

    monkeypatch.setattr(wan_diffusers, "is_diffusers_wan21_layout", lambda _: diffusers)
    monkeypatch.setattr(
        wan_diffusers, "_identifies_as_diffusers_wan21", lambda _: False
    )
    monkeypatch.setattr(wan_diffusers, "diffusers_runtime", runtime)
    kwargs = {"model_dir": "original", "output_path": str(tmp_path / "missing.mp4")}
    with pytest.raises(VideoGenerationCancelled):
        wan_diffusers.generate_with_runtime(
            tmp_path,
            SimpleNamespace(generate_video=generate),
            kwargs,
            cancel_token=token,
        )
    expected_model_dir = str(tmp_path / "view") if diffusers else "original"
    assert calls[0]["model_dir"] == expected_model_dir
    assert kwargs["model_dir"] == "original"
    if diffusers:
        assert calls[-1] == "runtime closed"


def test_pull_summary_reports_unknown_size_after_filesystem_error(
    monkeypatch, tmp_path, capsys
):
    from rapid_mlx import cli

    def rglob(self, pattern):
        raise OSError("snapshot unreadable")

    monkeypatch.setattr(Path, "rglob", rglob)
    cli._print_pull_summary("example/video", tmp_path, 2, allow_patterns=["weights/*"])
    output = capsys.readouterr().out
    assert "Downloaded example/video" in output
    assert "unknown size" in output
    assert "Already cached" not in output
