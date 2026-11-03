"""Offline contracts for the Pika H3 Flex adapter."""

import asyncio
import fnmatch
import hashlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from safetensors.numpy import save_file

from rapid_mlx.video import minimax_h3 as h3
from rapid_mlx.video import pika_h3_flex as flex


@pytest.fixture
def mx():
    return pytest.importorskip("mlx.core", exc_type=ImportError)


@pytest.fixture
def array_core(monkeypatch):
    # Exercise sampler arithmetic and load wiring without native GPU dependencies.
    mx = ModuleType("mlx.core")
    mx.array, mx.float32, mx.bfloat16 = np.array, np.float32, np.float16
    mx.float16, mx.expand_dims = np.float16, np.expand_dims
    mx.full, mx.broadcast_to = np.full, np.broadcast_to
    mx.clear_cache = lambda: None
    mx.random = SimpleNamespace(
        key=lambda seed: seed,
        split=lambda key: (key + 1, key + 2),
        normal=lambda shape, dtype, key: (
            np.random.default_rng(key).normal(size=shape).astype(dtype)
        ),
    )
    mlx = ModuleType("mlx")
    mlx.core = mx
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", mx)
    return mx


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    from rapid_mlx import _mirror

    monkeypatch.setattr(
        _mirror,
        "pinned_snapshot_download",
        lambda *a, **k: pytest.fail("unexpected download"),
    )


def test_ladders_and_alignment():
    for shift in (12, 3):
        sigmas = flex.shifted_sigmas(flex.KNOTS, shift)
        expected = [
            shift * (k / 1000) / (1 + (shift - 1) * k / 1000) for k in flex.KNOTS
        ]
        assert sigmas == np.array(expected, dtype=np.float32).tolist()
        assert len(sigmas) == 7
    assert flex.shifted_sigmas(flex.KNOTS, 3)[1] == pytest.approx(0.9)
    for knots in (
        [],
        [1000, 1],
        [999, 0],
        [1000, 500, 500, 0],
        [1000, float("nan"), 0],
    ):
        with pytest.raises(ValueError):
            flex.shifted_sigmas(knots, 12)
    assert [flex.align_frames(s) for s in (5, 6, 10, 14, 15)] == [
        124,
        158,
        243,
        345,
        345,
    ]


def test_renoise_math_and_shared_order(array_core):
    mx = array_core
    draws = []

    def normal(shape):
        draws.append(shape)
        return mx.full(shape, 2.0, dtype=mx.float32)

    noise = SimpleNamespace(normal=normal)
    video, audio = flex.RenoiseScheduler(12, noise), flex.RenoiseScheduler(3, noise)
    video.set_timesteps(99, sigmas=[1, 0])
    assert video.num_inference_steps == 6
    x, v = mx.array([1.0], dtype=mx.float16), mx.array([3.0])
    forwards = 0
    for tv, ta in zip(video.timesteps, audio.timesteps):
        forwards += 1
        for scheduler, t, shape in ((video, tv, (1,)), (audio, ta, (2,))):
            sample = mx.broadcast_to(x, shape)
            sigma = float(scheduler.sigmas[scheduler._step_index + 1])
            x0 = sample.astype(mx.float32) + (1 - t) * v
            result = scheduler.step(v, t, sample)
            expected = x0 if sigma == 0 else (1 - sigma) * x0 + sigma * 2
            assert result.dtype == sample.dtype
            np.testing.assert_array_equal(
                np.array(result), np.array(expected.astype(sample.dtype))
            )
    assert forwards == 6
    assert draws == [(1,), (2,)] * 5
    np.testing.assert_allclose(
        np.array(video.scale_noise(mx.array([4.0]), 0.25, mx.array([2.0]))), [2.5]
    )


def test_real_noise_reproducible(mx):
    stream = flex.RenoiseNoise()
    stream.reset(7)
    first = stream.normal((4,))
    stream.reset(7)
    np.testing.assert_array_equal(np.array(first), np.array(stream.normal((4,))))


def test_merge_and_invalid_adapters(tmp_path, mx):
    import mlx.nn as nn

    class Transformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.to_q = nn.Linear(2, 3, bias=False)
            self.other = nn.Linear(2, 3, bias=False)

    model = Transformer()
    before, untouched = mx.array(model.to_q.weight), mx.array(model.other.weight)
    path = tmp_path / "adapter.safetensors"
    a, b = np.ones((128, 2), np.float32), np.full((3, 128), 2, np.float32)

    def write(prefix="to_q", rank=128, missing_b=False):
        arrays = {f"{prefix}.lora_A.default.weight": a[:rank]}
        if not missing_b:
            arrays[f"{prefix}.lora_B.default.weight"] = b[:, :rank]
        save_file(arrays, str(path), metadata={"alpha": "8"})

    def merge():
        return h3.merge_lora(
            model,
            path,
            scale=1.0,
            allowed_suffixes=flex.TARGET_MODULES,
            expected_rank=128,
        )

    write()
    assert merge() == 1
    np.testing.assert_allclose(np.array(model.to_q.weight), np.array(before + 256))
    np.testing.assert_array_equal(np.array(model.other.weight), np.array(untouched))
    h3._merge_turbo_lora(model, path)
    np.testing.assert_allclose(np.array(model.to_q.weight), np.array(before + 256 + 16))
    for kwargs, message in (
        ({"prefix": "other"}, "target"),
        ({"rank": 127}, "rank"),
        ({"prefix": "missing.to_q"}, "base"),
        ({"missing_b": True}, "lora_B"),
    ):
        write(**kwargs)
        with pytest.raises(h3.MiniMaxH3BackendError, match=message):
            merge()
    save_file({"unrelated": np.zeros((1,), np.float32)}, str(path))
    with pytest.raises(h3.MiniMaxH3BackendError, match="no adapter"):
        merge()


def manifest():
    return dict(
        rank=128,
        alpha=128,
        trunk="transformer",
        numerics_policy="bf16_conditioning_v1",
        target_modules=list(flex.TARGET_MODULES),
        sha256=flex.ADAPTER_SHA256,
        sampling=dict(
            knots=list(flex.KNOTS), video_shift=12.0, audio_shift=3.0, method="renoise"
        ),
    )


def test_adapter_verification(monkeypatch, tmp_path):
    (tmp_path / "fl2va").mkdir()
    (tmp_path / flex.ADAPTER_FILE).write_bytes(b"tiny")
    path = tmp_path / "fl2va/manifest.json"
    data = manifest()
    path.write_text(json.dumps(data))
    with pytest.raises(h3.MiniMaxH3BackendError, match="sha256"):
        flex.verify_adapter(tmp_path)
    monkeypatch.setattr(flex, "ADAPTER_SHA256", hashlib.sha256(b"tiny").hexdigest())
    data["sha256"] = flex.ADAPTER_SHA256
    path.write_text(json.dumps(data))
    flex.verify_adapter(tmp_path)
    for key in data:
        bad = dict(data, **{key: None})
        path.write_text(json.dumps(bad))
        with pytest.raises(h3.MiniMaxH3BackendError):
            flex.verify_adapter(tmp_path)


def test_catalog_and_payloads(monkeypatch, tmp_path):
    import huggingface_hub
    import huggingface_hub.constants

    from rapid_mlx._download_gate import is_repo_cached
    from rapid_mlx.model_aliases import resolve_profile, subfolder_allow_patterns
    from scripts.gen_model_sizes import _size_one

    profile = resolve_profile("pika-h3-flex")
    assert (profile.hf_path, profile.revision, profile.min_memory_gb) == (
        flex.REPO,
        flex.REVISION,
        160,
    )
    assert list(profile.video_modes) == ["text-to-video", "image-to-video"]
    assert subfolder_allow_patterns(flex.REPO) == list(flex.FILES)
    for name in ("pika-h3-flex", flex.REPO, flex.REPO.upper()):
        assert flex.is_pika_h3_flex_model(name) and not h3.is_minimax_h3_model(name)
    for bad in ("ref2va/adapter.safetensors", "comfy/code.py", "transformer/model.py"):
        assert not any(fnmatch.fnmatchcase(bad, pattern) for pattern in flex.FILES)
    root = Path(__file__).resolve().parents[1]
    assert (
        json.loads((root / "rapid_mlx/model_sizes.json").read_text())["sizes"][
            flex.REPO
        ]
        == 146818516457
    )
    assert any(
        e["hf_path"] == flex.REPO
        for e in json.loads((root / "scripts/mirror_unmirrored.json").read_text())[
            "entries"
        ]
    )
    calls = []

    def info(repo, **kwargs):
        calls.append((repo, kwargs["revision"]))
        files = flex.FILES if repo == flex.REPO else h3.FILES
        return SimpleNamespace(
            siblings=[SimpleNamespace(rfilename=f, size=2) for f in files]
        )

    monkeypatch.setattr(huggingface_hub, "model_info", info)
    assert _size_one(flex.REPO) == (flex.REPO, 2 * (len(flex.FILES) + len(h3.FILES)))
    assert calls == [(repo, revision) for repo, revision, _ in flex.PAYLOADS]
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path))
    paths = []
    for repo, revision, files in flex.PAYLOADS:
        for name in files:
            path = (
                tmp_path
                / f"models--{repo.replace('/', '--')}"
                / "snapshots"
                / revision
                / name
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x")
            paths.append(path)
    assert is_repo_cached(flex.REPO)
    for path in (paths[0], paths[-1]):
        path.unlink()
        assert not is_repo_cached(flex.REPO)
        path.write_bytes(b"x")


def test_engine_and_load_recipe(monkeypatch, tmp_path, array_core):
    from rapid_mlx import _mirror
    from rapid_mlx.video import encoding

    mx = array_core
    downloads, merges, requests = [], [], []
    monkeypatch.setattr(
        _mirror,
        "pinned_snapshot_download",
        lambda *a, **k: downloads.append((a, k)) or str(tmp_path),
    )
    monkeypatch.setattr(flex, "verify_adapter", lambda root: None)
    monkeypatch.setattr(flex, "merge_lora", lambda *a, **k: merges.append(k))
    monkeypatch.setattr(
        encoding,
        "encode_rgb_video",
        lambda frames, output, *a, **k: output.write_bytes(b"mp4"),
    )
    pipeline = SimpleNamespace(
        transformer=object(), _normalize_video_latents=lambda x: x
    )
    module = ModuleType("mlx_vlm.generate")
    module.load_video_generation_model = lambda *a, **k: SimpleNamespace(
        pipeline=pipeline
    )
    module.VideoGenerationRequest = lambda **k: SimpleNamespace(**k)

    def generate(model, request):
        requests.append(
            (model.workflow, request, np.array(pipeline.scheduler.noise.normal((2,))))
        )
        return SimpleNamespace(
            frames=np.zeros((1, 1, 1, 3)), fps=24, audio=None, sampling_rate=None
        )

    module.generate_video = generate
    monkeypatch.setitem(sys.modules, "mlx_vlm.generate", module)
    engine = flex.PikaH3FlexVideoEngine("pika-h3-flex")
    args = dict(
        prompt="test",
        output_path=tmp_path / "out.mp4",
        width=832,
        height=480,
        num_frames=124,
        fps=24,
        seed=2,
    )
    try:
        for image, last in (
            (None, None),
            (None, tmp_path / "last"),
            (tmp_path / "first", None),
            (tmp_path / "first", tmp_path / "last"),
        ):
            engine.generate(**args, image=image, last_image=last)
        assert downloads == [
            ((repo, revision), {"allow_patterns": list(files)})
            for repo, revision, files in reversed(flex.PAYLOADS)
        ]
        assert merges == [
            dict(scale=1.0, allowed_suffixes=flex.TARGET_MODULES, expected_rank=128)
        ]
        assert pipeline.scheduler.noise is pipeline.audio_scheduler.noise
        assert [r[0] for r in requests] == ["t2va", "fl2va", "fl2va", "fl2va"]
        assert all(
            r[1].steps == 7 and r[1].extra == {"drop_adaln_weights": False}
            for r in requests
        )
        np.testing.assert_array_equal(requests[0][2], requests[1][2])
        np.testing.assert_array_equal(
            requests[0][2],
            np.random.default_rng(5).normal(size=(2,)).astype(np.float32),
        )
        values = mx.array([1.001], dtype=mx.float32)
        np.testing.assert_array_equal(
            np.array(pipeline._normalize_video_latents(values)),
            np.array(values.astype(mx.bfloat16).astype(mx.float32)),
        )
    finally:
        engine.stop()
    base = h3.MiniMaxH3VideoEngine("minimax-h3")
    try:
        with pytest.raises(h3.MiniMaxH3RequestError, match="first image"):
            base.generate(**args, image=None, last_image=tmp_path / "last")
    finally:
        base.stop()


@pytest.mark.asyncio
async def test_routes_and_lane(monkeypatch, tmp_path):
    from fastapi import HTTPException, UploadFile
    from PIL import Image

    from rapid_mlx.routes import video
    from rapid_mlx.runtime import video_lane

    lane = video_lane.VideoEngine("pika-h3-flex")
    assert isinstance(lane._minimax_engine, flex.PikaH3FlexVideoEngine)
    forwarded = []
    lane._minimax_engine.generate = lambda **kw: forwarded.append(kw)
    lane._crop_generated_output = lambda **kw: None
    lane.generate(
        prompt="test",
        output_path=tmp_path / "out",
        width=832,
        height=480,
        num_frames=158,
        fps=24,
        seed=1,
        image=None,
        last_image=tmp_path / "last",
    )
    assert forwarded[0]["last_image"] == tmp_path / "last"
    await lane.stop()
    for backend in (None, SimpleNamespace(supports_last_frame=False)):
        lane._minimax_engine = backend
        with pytest.raises(
            video_lane.VideoRuntimeError, match="last_frame is not supported"
        ):
            lane.generate(
                prompt="test",
                output_path=tmp_path / "out",
                width=832,
                height=480,
                num_frames=158,
                fps=24,
                seed=1,
                image=None,
                last_image=tmp_path / "last",
            )

    calls = []
    fake = SimpleNamespace(
        model_name=flex.REPO,
        video_family="minimax-h3",
        native_fps=24,
        _minimax_engine=SimpleNamespace(supports_last_frame=True),
    )

    async def run(job, **kw):
        calls.append(kw)

    monkeypatch.setattr(video, "_run_job", run)
    monkeypatch.setattr(video, "_video_engine", lambda: fake)
    video.configure_video_jobs(tmp_path / "jobs")
    video.start_video_jobs()
    path = tmp_path / "last.png"
    Image.new("RGB", (2, 2)).save(path)
    try:
        with path.open("rb") as source:
            await video.create_video(
                prompt="test",
                model="pika-h3-flex",
                seconds="6",
                size="832x480",
                input_reference=None,
                last_frame=UploadFile(source),
            )
            await asyncio.sleep(0)
        assert calls[-1]["num_frames"] == 158
        assert calls[-1]["last_image_path"].read_bytes() == path.read_bytes()
        assert (
            video._video_capabilities(fake)["limits"]["last_frame"]
            == video._video_capabilities(fake)["limits"]["input_reference"]
        )
        fake._minimax_engine.supports_last_frame = False
        with pytest.raises(HTTPException, match="last_frame") as exc:
            await video.create_video(last_frame=object())
        assert exc.value.status_code == 400
        assert "last_frame" not in video._video_capabilities(fake)["limits"]
        fake.model_name = h3.REPO
        await video.create_video(
            prompt="test",
            model="minimax-h3",
            seconds="6",
            size="832x480",
            input_reference=None,
        )
        await asyncio.sleep(0)
        assert calls[-1]["num_frames"] == 141
        fake.video_family = "wan"
        with pytest.raises(HTTPException) as exc:
            await video.create_video(last_frame=object())
        assert exc.value.status_code == 400
    finally:
        video.configure_video_jobs(None)
        video.start_video_jobs()


def test_pull_and_asset_owners(monkeypatch):
    import argparse

    from rapid_mlx import cli
    from rapid_mlx.runtime import video_lane

    calls = []
    monkeypatch.setattr(video_lane, "video_runtime_issue", lambda *a, **k: None)
    monkeypatch.setattr(
        cli, "_pull_repository", lambda a, **k: calls.append((a.model, k))
    )
    cli.pull_command(argparse.Namespace(model="pika-h3-flex", bits=None, format=None))
    assert calls == [
        (repo, dict(allow_patterns_override=list(files), revision_override=rev))
        for repo, rev, files in flex.PAYLOADS
    ]
    assert "pika-h3-flex" in cli._runtime_asset_owners()[h3.REPO.lower()]
    for bits, fmt in ((4, None), (None, "q4")):
        with pytest.raises(ValueError, match="does not support"):
            cli.pull_command(
                argparse.Namespace(model="pika-h3-flex", bits=bits, format=fmt)
            )
