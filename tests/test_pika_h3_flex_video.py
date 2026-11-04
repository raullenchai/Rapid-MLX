"""Offline contracts for the Pika H3 Flex adapter."""

import asyncio
import fnmatch
import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from safetensors.numpy import save_file

from rapid_mlx.video import minimax_h3 as h3
from rapid_mlx.video import pika_h3_flex as flex


class FakeVAEEncoder:
    def __init__(self):
        self.weight = np.array([2.0], dtype=np.float32)

    def parameters(self):
        return {"weight": self.weight}

    def __call__(self, values, *, offset=0):
        return values * self.weight + offset


def test_eager_vae_encoder_preserves_values_and_parameters(monkeypatch, array_core):
    evaluated = []
    monkeypatch.setattr(array_core, "eval", evaluated.append, raising=False)
    encoder = FakeVAEEncoder()
    vae = SimpleNamespace(encoder=encoder)
    values = np.array([1.0, 3.0], dtype=np.float32)
    expected = encoder(values, offset=1)
    parameters = encoder.parameters()
    attributes = dict(vars(encoder))

    flex._install_eager_vae_encoder(vae)
    eager_class = type(encoder)
    assert eager_class is not FakeVAEEncoder
    assert issubclass(eager_class, FakeVAEEncoder)
    assert vae.encoder is encoder
    assert vars(encoder).keys() == attributes.keys()
    assert encoder.parameters().keys() == parameters.keys()
    assert encoder.parameters()["weight"] is parameters["weight"]
    assert encoder.weight is attributes["weight"]
    assert evaluated == []
    flex._install_eager_vae_encoder(vae)
    assert type(encoder) is eager_class
    for count in (1, 2):
        result = encoder(values, offset=1)
        np.testing.assert_array_equal(result, expected)
        assert len(evaluated) == count
        assert evaluated[-1] is result

    second = SimpleNamespace(encoder=FakeVAEEncoder())
    flex._install_eager_vae_encoder(second)
    assert type(second.encoder) is eager_class
    second.encoder(values)
    assert len(evaluated) == 3


@pytest.mark.parametrize(
    "vae", [None, SimpleNamespace(), SimpleNamespace(encoder=None)]
)
def test_eager_vae_encoder_requires_encoder(vae):
    with pytest.raises(h3.MiniMaxH3BackendError, match="Video VAE encoder"):
        flex._install_eager_vae_encoder(vae)


def test_prepare_model_requires_video_vae():
    engine = flex.PikaH3FlexVideoEngine("pika-h3-flex")
    try:
        with pytest.raises(h3.MiniMaxH3BackendError, match="Video VAE encoder"):
            engine._prepare_model(SimpleNamespace(pipeline=SimpleNamespace()))
    finally:
        engine.stop()


@pytest.fixture(scope="module")
def mx():
    # Native MLX initialization can abort when Metal is unavailable in a sandbox.
    probe = subprocess.run(
        [sys.executable, "-c", "import mlx.core as mx; mx.eval(mx.ones((1,)))"],
        capture_output=True,
        timeout=30,
    )
    if probe.returncode:
        pytest.skip("real MLX runtime unavailable on this host")
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
    assert flex.KNOTS == (1000, 750, 500, 250, 100, 40, 0)
    assert flex.VIDEO_SHIFT == 12.0
    assert flex.AUDIO_SHIFT == 3.0
    assert flex.shifted_sigmas(flex.KNOTS, flex.VIDEO_SHIFT) == pytest.approx(
        [1.0, 0.9729730, 0.9230769, 0.8, 0.5714286, 0.3333333, 0.0],
        rel=1e-6,
        abs=1e-6,
    )
    assert flex.shifted_sigmas(flex.KNOTS, flex.AUDIO_SHIFT) == pytest.approx(
        [1.0, 0.9, 0.75, 0.5, 0.25, 0.1111111, 0.0],
        rel=1e-6,
        abs=1e-6,
    )
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

    # Literal reference vectors from the released sampler equations.
    def reference_normal(shape):
        draws.append(shape)
        return mx.array([0.1, -0.2], dtype=mx.float32)

    noise = SimpleNamespace(normal=reference_normal)
    video, audio = flex.RenoiseScheduler(12, noise), flex.RenoiseScheduler(3, noise)
    x = mx.array([1.0, 2.0], dtype=mx.float32)
    v = mx.array([0.5, -1.0], dtype=mx.float32)
    assert video.timesteps[0] == audio.timesteps[0] == 0.0
    np.testing.assert_allclose(
        video.step(v, video.timesteps[0], x),
        [0.1378378, -0.1675676],
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        audio.step(v, audio.timesteps[0], x),
        [0.24, -0.08],
        rtol=1e-6,
        atol=1e-6,
    )
    for t in video.timesteps[1:-1]:
        video.step(v, t, x)
    assert video.timesteps[-1] == pytest.approx(0.6666667)
    draws_before_final = len(draws)
    np.testing.assert_allclose(
        video.step(v, video.timesteps[-1], x),
        [1.1666667, 1.6666667],
        rtol=1e-6,
        atol=1e-6,
    )
    assert len(draws) == draws_before_final


def test_real_noise_reproducible(mx):
    stream = flex.RenoiseNoise()
    stream.reset(7)
    first = stream.normal((4,))
    stream.reset(7)
    np.testing.assert_array_equal(np.array(first), np.array(stream.normal((4,))))


def test_real_scheduler_default_noise(mx):
    scheduler = flex.RenoiseScheduler(12.0)
    expected = flex.RenoiseNoise()
    expected.reset(0)
    assert isinstance(scheduler.noise, flex.RenoiseNoise)
    np.testing.assert_array_equal(
        np.array(scheduler.noise.normal((4,))), np.array(expected.normal((4,)))
    )


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
        192,
    )
    assert list(profile.video_modes) == ["text-to-video", "image-to-video"]
    assert subfolder_allow_patterns(flex.REPO) == list(flex.FILES)
    for name in ("pika-h3-flex", flex.REPO, flex.REPO.upper()):
        assert flex.is_pika_h3_flex_model(name) and not h3.is_minimax_h3_model(name)
    for bad in ("comfy/code.py", "transformer/model.py"):
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
        files = flex.FILES if repo == flex.REPO else flex.BASE_FILES
        return SimpleNamespace(
            siblings=[SimpleNamespace(rfilename=f, size=2) for f in files]
        )

    monkeypatch.setattr(huggingface_hub, "model_info", info)
    assert _size_one(flex.REPO) == (
        flex.REPO,
        2 * (len(flex.FILES) + len(flex.BASE_FILES)),
    )
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
    monkeypatch.setattr(flex, "verify_adapter", lambda root, mode="fl2va": None)
    monkeypatch.setattr(flex, "merge_lora", lambda *a, **k: merges.append(k))
    monkeypatch.setattr(
        encoding,
        "encode_rgb_video",
        lambda frames, output, *a, **k: output.write_bytes(b"mp4"),
    )
    pipeline = SimpleNamespace(
        transformer=object(),
        _normalize_video_latents=lambda x: x,
        video_vae=SimpleNamespace(encoder=FakeVAEEncoder()),
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
        assert type(pipeline.video_vae.encoder) is not FakeVAEEncoder
        assert isinstance(pipeline.video_vae.encoder, FakeVAEEncoder)
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


@pytest.mark.asyncio
@pytest.mark.parametrize("oversized", [True, False], ids=["oversized", "invalid"])
async def test_last_frame_validation(monkeypatch, tmp_path, oversized):
    from fastapi import HTTPException, UploadFile

    from rapid_mlx.routes import video

    fake = SimpleNamespace(
        model_name=flex.REPO,
        video_family="minimax-h3",
        native_fps=24,
        _minimax_engine=SimpleNamespace(supports_last_frame=True),
    )
    monkeypatch.setattr(video, "_video_engine", lambda: fake)
    payload = b"not an image"
    if oversized:
        monkeypatch.setattr(video, "_MAX_REFERENCE_BYTES", len(payload) - 1)
    video.configure_video_jobs(tmp_path / "jobs")
    video.start_video_jobs()
    try:
        with pytest.raises(HTTPException) as exc:
            await video.create_video(
                prompt="test",
                model="pika-h3-flex",
                seconds="6",
                size="832x480",
                input_reference=None,
                last_frame=UploadFile(io.BytesIO(payload), filename="last.png"),
            )
        if oversized:
            assert exc.value.status_code == 413
            assert exc.value.detail == "last_frame exceeds 20 MB"
        else:
            assert exc.value.status_code == 400
            assert exc.value.detail.startswith("invalid last_frame:")
            assert "input_reference" not in exc.value.detail
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


@pytest.mark.parametrize(
    "kinds, message",
    [
        (["image"] * 10, "image"),
        (["video"] * 4, "video"),
        (["audio"] * 4, "audio"),
        (["image"] * 13, "total"),
        (["audio"], "image or video"),
        (["other"], "unsupported"),
    ],
)
def test_reference_counts(kinds, message):
    with pytest.raises(h3.MiniMaxH3RequestError, match=message):
        flex.validate_reference_counts(kinds)
    flex.validate_reference_counts(["image"] * 6 + ["video"] * 3 + ["audio"] * 3)


@pytest.mark.parametrize("kind", ["video", "audio"])
def test_reference_durations(kind):
    for duration in (2 - 1e-6, 2, 15, 15 + 0.5e-6):
        flex.validate_reference_durations([(kind, duration)])
    for duration in (1.9, 15.01, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="duration"):
            flex.validate_reference_durations([(kind, duration)])
    with pytest.raises(ValueError, match="total"):
        flex.validate_reference_durations([(kind, 8), (kind, 8)])
    flex.validate_reference_durations([("video", 15), ("audio", 15)])


def test_reference_layout_math():
    with pytest.raises(ValueError, match="unsupported Flex partition: bogus"):
        flex.adapter_spec("bogus")
    with pytest.raises(ValueError, match="canvas dimensions must be positive"):
        flex.reference_tier(0, 480)
    for (w, h), name in [
        ((832, 480), "p480"),
        ((1344, 768), "p768"),
        ((960, 416), "p480"),
        ((9999, 9999), "p2160"),
    ]:
        assert flex.reference_tier(h, w).name == name
    low = flex.reference_tier(480, 832)
    high = flex.reference_tier(768, 1344)
    assert flex.reference_image_size(1920, 1080, low) == (1024, 1824)
    assert flex.reference_image_size(1920, 1080, high) == (2048, 3648)
    assert flex.reference_video_size(1920, 1080, low) == (480, 832)
    assert flex.reference_audio_max_samples(124) == 165333
    with pytest.raises(ValueError, match="1:4"):
        flex.reference_image_size(100, 500, low)
    with pytest.raises(ValueError, match="reference video must be within 1:4 and 4:1"):
        flex.reference_video_size(2000, 400, low)


def test_ref_adapter_verification(monkeypatch, tmp_path):
    (tmp_path / "ref2va").mkdir()
    (tmp_path / flex.REF_ADAPTER_FILE).write_bytes(b"tiny")
    spec = flex.adapter_spec("ref2va")
    assert (
        spec.sha256
        == "eef684ba1258f0cdbbbda6128188ad2688417f2eb17ff875375e74a32b6d61b5"
    )
    data = dict(
        manifest(),
        trunk="transformer_ref",
        numerics_policy="fp32_conditioning_v1",
        sha256=spec.sha256,
    )
    path = tmp_path / "ref2va/manifest.json"
    path.write_text(json.dumps(data))
    with pytest.raises(h3.MiniMaxH3BackendError, match="sha256"):
        flex.verify_adapter(tmp_path, "ref2va")
    monkeypatch.setattr(flex, "REF_ADAPTER_SHA256", hashlib.sha256(b"tiny").hexdigest())
    data["sha256"] = flex.REF_ADAPTER_SHA256
    path.write_text(json.dumps(data))
    flex.verify_adapter(tmp_path, "ref2va")
    for key in data:
        path.write_text(json.dumps(dict(data, **{key: None})))
        with pytest.raises(h3.MiniMaxH3BackendError):
            flex.verify_adapter(tmp_path, "ref2va")


def test_partition_switch_and_reference_request(monkeypatch, tmp_path, array_core):
    import weakref

    from rapid_mlx.video import reference_media

    installs = []
    monkeypatch.setattr(
        reference_media, "install_reference_decoders", lambda: installs.append(True)
    )
    monkeypatch.setattr(flex, "ensure_ref2va_assets", lambda: None)

    from rapid_mlx import _mirror
    from rapid_mlx.video import encoding

    class Trunk:
        pass

    original = lambda x: x
    pipeline = SimpleNamespace(
        transformer=Trunk(),
        _normalize_video_latents=original,
        video_vae=SimpleNamespace(encoder=FakeVAEEncoder()),
    )
    old = weakref.ref(pipeline.transformer)
    loads, verifications, merges, requests, events = [], [], [], [], []
    monkeypatch.setattr(
        _mirror, "pinned_snapshot_download", lambda *a, **k: str(tmp_path)
    )
    monkeypatch.setattr(
        flex,
        "verify_adapter",
        lambda root, mode: verifications.append((mode, flex.adapter_spec(mode).sha256)),
    )
    monkeypatch.setattr(
        flex, "merge_lora", lambda trunk, path, **kw: merges.append((path, kw))
    )
    monkeypatch.setattr(array_core, "clear_cache", lambda: events.append("clear"))
    weights = ModuleType("mlx_vlm.models.minimax_h3.weights")

    def load(root, *, partition):
        assert pipeline.transformer is None
        assert old() is None
        assert "clear" in events
        loads.append(partition)
        return Trunk()

    weights.load_transformer = load
    monkeypatch.setitem(sys.modules, weights.__name__, weights)
    module = ModuleType("mlx_vlm.generate")

    def first_load(path, *, workflow):
        loads.append(workflow)
        return SimpleNamespace(pipeline=pipeline)

    module.load_video_generation_model = first_load
    module.VideoGenerationRequest = lambda **kw: SimpleNamespace(**kw)

    def generate(model, request):
        requests.append((model.workflow, request))
        return SimpleNamespace(
            frames=np.zeros((1, 1, 1, 3)), fps=24, audio=None, sampling_rate=None
        )

    module.generate_video = generate
    monkeypatch.setitem(sys.modules, module.__name__, module)
    refs_module = ModuleType("mlx_vlm.generate.video_generation")
    refs_module.VideoReference = lambda kind, path: (kind, path)
    monkeypatch.setitem(sys.modules, refs_module.__name__, refs_module)
    monkeypatch.setattr(
        encoding,
        "encode_rgb_video",
        lambda frames, path, *a, **k: path.write_bytes(b"mp4"),
    )
    engine = flex.PikaH3FlexVideoEngine("pika-h3-flex")
    args = dict(
        prompt="test",
        output_path=tmp_path / "out.mp4",
        width=832,
        height=480,
        num_frames=124,
        fps=24,
        seed=1,
        image=None,
    )
    refs = [
        ("audio", tmp_path / "a.wav"),
        ("image", tmp_path / "b.png"),
        ("video", tmp_path / "c.mp4"),
    ]
    try:
        engine._generate(**args)
        assert pipeline._normalize_video_latents is engine._bf16_normalize
        schedulers = (pipeline.scheduler, pipeline.audio_scheduler)
        pipeline._adaln_cache_key, pipeline._adaln_caches = "old", object()
        pipeline._adaln_weights_freed_bytes = 123
        engine._generate(**args, references=refs)
        assert installs == [True]
        assert pipeline._normalize_video_latents is original
        assert pipeline.partition == "ref2va"
        assert pipeline._adaln_cache_key is None and pipeline._adaln_caches is None
        assert pipeline._adaln_weights_freed_bytes == 0
        assert schedulers == (pipeline.scheduler, pipeline.audio_scheduler)
        assert verifications[-1] == ("ref2va", flex.REF_ADAPTER_SHA256)
        assert merges[-1] == (
            tmp_path / flex.REF_ADAPTER_FILE,
            dict(scale=1.0, allowed_suffixes=flex.TARGET_MODULES, expected_rank=128),
        )
        assert requests[-1][0] == "ref2va"
        assert requests[-1][1].references == tuple(refs)
        assert requests[-1][1].image is None and requests[-1][1].last_image is None
        engine._generate(**args, references=refs)
        assert loads == ["fl2va", "ref2va"]
        engine._generate(**args)
        assert loads == ["fl2va", "ref2va", "fl2va"]
        assert pipeline._normalize_video_latents is engine._bf16_normalize
        with pytest.raises(h3.MiniMaxH3RequestError, match="combined"):
            engine._generate(**dict(args, image=tmp_path / "image"), references=refs)
    finally:
        engine.stop()


@pytest.mark.parametrize("recovery_partition", ["fl2va", "ref2va"])
@pytest.mark.parametrize("failure_stage", ["load", "merge"])
def test_failed_partition_switch_self_heals(
    monkeypatch, tmp_path, array_core, recovery_partition, failure_stage
):
    class Trunk:
        pass

    pipeline = SimpleNamespace(
        transformer=Trunk(),
        _normalize_video_latents=lambda x: x,
        video_vae=SimpleNamespace(encoder=FakeVAEEncoder()),
    )
    loads, merges, events = [], [], []
    monkeypatch.setattr(array_core, "clear_cache", lambda: events.append("clear"))
    failure = RuntimeError(f"{failure_stage} failed")
    weights = ModuleType("mlx_vlm.models.minimax_h3.weights")

    def load(root, *, partition):
        assert root == tmp_path
        assert pipeline.transformer is None
        loads.append(partition)
        if failure_stage == "load" and loads == ["ref2va"]:
            events.append("failure")
            raise failure
        return Trunk()

    weights.load_transformer = load
    monkeypatch.setitem(sys.modules, weights.__name__, weights)

    def merge(pipeline, partition):
        assert isinstance(pipeline.transformer, Trunk)
        merges.append(partition)
        if failure_stage == "merge" and merges == ["fl2va", "ref2va"]:
            events.append("failure")
            raise failure

    monkeypatch.setattr(flex, "ensure_ref2va_assets", lambda: None)
    engine = flex.PikaH3FlexVideoEngine("pika-h3-flex")
    monkeypatch.setattr(engine, "_merge_adapter", merge)
    engine._model = SimpleNamespace(pipeline=pipeline)
    engine._model_root = tmp_path
    engine._needed_partition = "fl2va"
    try:
        engine._prepare_model(engine._model)
        assert engine._resident_partition == "fl2va"
        with pytest.raises(RuntimeError, match=f"{failure_stage} failed") as exc:
            engine._ensure_partition("ref2va")
        assert exc.value is failure
        assert pipeline.transformer is None
        assert engine._resident_partition is None
        assert events[-2:] == ["failure", "clear"]

        engine._ensure_partition(recovery_partition)
        assert isinstance(pipeline.transformer, Trunk)
        assert engine._resident_partition == recovery_partition
        assert pipeline.partition == recovery_partition
        assert loads == ["ref2va", recovery_partition]
        expected_merges = ["fl2va"]
        if failure_stage == "merge":
            expected_merges.append("ref2va")
        assert merges == [*expected_merges, recovery_partition]
    finally:
        engine.stop()


@pytest.fixture
def reference_app(monkeypatch, tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from rapid_mlx.routes import video

    fake = SimpleNamespace(
        model_name=flex.REPO,
        video_family="minimax-h3",
        native_fps=24,
        _minimax_engine=SimpleNamespace(
            supports_last_frame=True, supports_references=True
        ),
    )
    calls = []

    async def run(job, **kw):
        calls.append(kw)

    monkeypatch.setattr(
        video, "_reference_decoder_kinds", lambda: frozenset({"video", "audio"})
    )
    monkeypatch.setattr(video, "_run_job", run)
    monkeypatch.setattr(video, "_video_engine", lambda: fake)
    video.configure_video_jobs(tmp_path / "jobs")
    video.start_video_jobs()
    app = FastAPI()
    app.include_router(video.router)
    app.dependency_overrides[video.verify_api_key] = lambda: None
    with TestClient(app) as client:
        yield client, fake, calls
    video.configure_video_jobs(None)
    video.start_video_jobs()


def png_bytes():
    from PIL import Image

    output = io.BytesIO()
    Image.new("RGB", (2, 2)).save(output, format="PNG")
    return output.getvalue()


def post_references(client, kinds, **extra):
    files = [
        (
            "references",
            (
                f"ignored-{i}.bad",
                png_bytes() if kind == "image" else kind.encode(),
                "application/octet-stream",
            ),
        )
        for i, kind in enumerate(kinds)
    ]
    files.extend(extra.get("files", []))
    return client.post(
        "/v1/videos",
        data=dict(prompt="test", model="pika-h3-flex", seconds="5", size="832x480"),
        files=files,
    )


@pytest.mark.parametrize(
    "kinds, message",
    [
        (["image"] * 10, "image"),
        (["video"] * 4, "video"),
        (["audio"] * 4, "audio"),
        (["image"] * 13, "total"),
        (["audio"], "image or video"),
        (["unknown"], "unsupported file type"),
    ],
)
def test_reference_http_limits(reference_app, monkeypatch, kinds, message):
    from fastapi import HTTPException

    from rapid_mlx.routes import video

    client, fake, calls = reference_app

    def probe(path, field):
        kind = path.read_text()
        if kind == "unknown":
            raise HTTPException(
                status_code=400, detail=f"{field}: unsupported file type"
            )
        return kind, "mp4" if kind == "video" else "wav", 2.0

    monkeypatch.setattr(video, "_probe_packed_reference", probe)
    response = post_references(client, kinds)
    assert response.status_code == 400, response.text
    assert message in response.text
    assert not calls
    assert not list(video._jobs_root.glob("video_*"))


def test_reference_http_order_and_capabilities(reference_app, monkeypatch):
    from rapid_mlx.routes import video

    client, fake, calls = reference_app
    monkeypatch.setattr(
        video, "_probe_packed_reference", lambda path, field: ("audio", "wav", 2.0)
    )
    response = post_references(client, ["audio", "image"])
    assert response.status_code == 200, response.text
    assert [kind for kind, path in calls[-1]["references"]] == ["audio", "image"]
    assert [path.name for kind, path in calls[-1]["references"]] == [
        "ref-00.wav",
        "ref-01.img",
    ]
    caps = client.get("/v1/videos/capabilities").json()
    limits = caps["limits"]["references"]
    assert "reference-to-video" in caps["modes"]
    assert limits == dict(
        field="references",
        maximum_total=12,
        image=dict(
            maximum=9, maximum_bytes=20 * 1024 * 1024, formats=["jpeg", "png", "webp"]
        ),
        video=dict(
            maximum=3,
            maximum_bytes=100 * 1024 * 1024,
            formats=["mp4", "mov", "webm", "mkv"],
            duration_seconds=dict(minimum=2, maximum=15),
        ),
        audio=dict(
            maximum=3,
            maximum_bytes=20 * 1024 * 1024,
            formats=["wav", "mp3", "m4a", "aac", "flac", "ogg"],
            duration_seconds=dict(minimum=2, maximum=15),
        ),
        total_duration_seconds_per_kind=15,
        requires_visual=True,
        exclusive_with=["input_reference", "last_frame"],
    )
    fake._minimax_engine.supports_references = False
    caps = client.get("/v1/videos/capabilities").json()
    assert (
        "references" not in caps["limits"] and "reference-to-video" not in caps["modes"]
    )
    response = post_references(client, ["image"])
    assert response.status_code == 400 and "not supported" in response.text


@pytest.mark.parametrize("field", ["input_reference", "last_frame"])
def test_reference_http_exclusive(reference_app, field):
    client, _, _ = reference_app
    response = post_references(
        client, ["image"], files=[(field, ("image.png", png_bytes()))]
    )
    assert response.status_code == 400 and "combined" in response.text


def test_reference_http_oversized_image_skips_validation(reference_app, monkeypatch):
    from rapid_mlx.routes import video

    client, _, _ = reference_app
    calls = []
    monkeypatch.setattr(video, "_MAX_REFERENCE_BYTES", 5)
    monkeypatch.setattr(
        video, "_validate_reference_image", lambda *a, **kw: calls.append(a)
    )
    response = post_references(client, ["image"])
    assert response.status_code == 413
    assert response.json()["detail"] == "references[0] exceeds 20 MB"
    assert calls == []


def test_reference_http_oversized(reference_app, monkeypatch):
    from rapid_mlx.routes import video

    client, _, _ = reference_app
    monkeypatch.setattr(video, "_MAX_REFERENCE_BYTES", 5)
    response = post_references(client, ["image"])
    assert response.status_code == 413
    monkeypatch.setattr(video, "_MAX_REFERENCE_VIDEO_BYTES", 4)
    response = post_references(client, ["video"])
    assert response.status_code == 413


@pytest.mark.parametrize("status,detail", [(400, "references[0]: bad"), (413, "bad")])
def test_reference_http_image_validation_error(
    reference_app, monkeypatch, status, detail
):
    from fastapi import HTTPException

    from rapid_mlx.routes import video

    client, _, calls = reference_app

    def validate(*args, **kwargs):
        raise HTTPException(status_code=status, detail="bad")

    monkeypatch.setattr(video, "_validate_reference_image", validate)
    response = post_references(client, ["image"])
    assert response.status_code == status
    assert response.json()["detail"] == detail
    assert not calls


def test_reference_http_image_decode_error(reference_app, monkeypatch):
    from PIL import Image

    client, _, calls = reference_app

    def open_image(*args, **kwargs):
        raise OSError("corrupt image")

    monkeypatch.setattr(Image, "open", open_image)
    response = post_references(client, ["image"])
    assert response.status_code == 400
    assert response.json()["detail"] == "references[0]: unsupported file type"
    assert not calls


def test_reference_http_oversized_audio(reference_app, monkeypatch):
    from rapid_mlx.routes import video

    client, _, calls = reference_app
    monkeypatch.setattr(video, "_MAX_REFERENCE_BYTES", 4)
    monkeypatch.setattr(
        video, "_probe_packed_reference", lambda *args: ("audio", "wav", 2.0)
    )
    response = post_references(client, ["audio"])
    assert response.status_code == 413
    assert response.json()["detail"] == "references[0] exceeds 20 MB"
    assert not calls


@pytest.mark.parametrize(
    "kind,formats,duration,detail",
    [
        ("subtitle", "mp4", "2", "unsupported file type"),
        ("audio", "matroska", "2", "unsupported file type"),
        ("audio", "wav", "1.0", "audio duration must be 2..15 seconds"),
        ("audio", "wav", "N/A", "audio duration must be 2..15 seconds"),
    ],
)
def test_reference_http_probe_rejection(
    reference_app, monkeypatch, kind, formats, duration, detail
):
    from rapid_mlx.routes import video

    client, _, calls = reference_app
    details = dict(
        streams=[dict(codec_type=kind, codec_name="aac", duration=duration)],
        format=dict(format_name=formats, duration=duration),
    )
    monkeypatch.setattr(video.shutil, "which", lambda name: "/ffprobe")
    monkeypatch.setattr(video, "_ffmpeg_decodable_codecs", lambda: {"h264", "aac"})
    monkeypatch.setattr(
        video.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=json.dumps(details)),
    )
    response = post_references(client, ["audio"])
    assert response.status_code == 400
    assert response.json()["detail"] == f"references[0]: {detail}"
    assert not calls


@pytest.mark.parametrize("kind", ["video", "audio"])
def test_reference_http_total_duration(reference_app, monkeypatch, kind):
    from rapid_mlx.routes import video

    client, _, _ = reference_app
    monkeypatch.setattr(
        video,
        "_probe_packed_reference",
        lambda *a: (kind, "mp4" if kind == "video" else "wav", 8.0),
    )
    response = post_references(client, ["image", kind, kind])
    assert response.status_code == 400 and "total duration" in response.text


@pytest.mark.parametrize(
    "kind,formats,extension",
    [
        ("video", "mov,mp4,m4a,3gp,3g2,mj2", "mp4"),
        ("video", "matroska,webm", "webm"),
        ("audio", "wav", "wav"),
    ],
)
def test_reference_ffprobe(monkeypatch, tmp_path, kind, formats, extension):
    from fastapi import HTTPException

    from rapid_mlx.routes import video

    monkeypatch.setattr(video.shutil, "which", lambda name: "/ffprobe")
    monkeypatch.setattr(video, "_ffmpeg_decodable_codecs", lambda: {"h264", "aac"})
    details = dict(
        streams=[
            dict(
                codec_type=kind,
                codec_name="h264" if kind == "video" else "aac",
                width=1920,
                height=1080,
            )
        ],
        format=dict(format_name=formats, duration="2"),
    )

    def probe(command, **kw):
        assert command[command.index("-protocol_whitelist") + 1] == "file"
        assert "-format_whitelist" in command
        return SimpleNamespace(stdout=json.dumps(details))

    monkeypatch.setattr(video.subprocess, "run", probe)
    assert video._probe_packed_reference(tmp_path / "file", "references[0]") == (
        kind,
        extension,
        2.0,
    )
    for duration in ("1.9", "15.1", "nan"):
        details["format"]["duration"] = duration
        with pytest.raises(HTTPException) as exc:
            video._probe_packed_reference(tmp_path / "file", "references[0]")
        assert exc.value.status_code == 400 and "duration" in exc.value.detail
    monkeypatch.setattr(video.shutil, "which", lambda name: None)
    with pytest.raises(HTTPException) as exc:
        video._probe_packed_reference(tmp_path / "file", "references[0]")
    assert exc.value.status_code == 503


@pytest.mark.parametrize(
    "kind,formats,extension", [("video", "mp4", "mp4"), ("audio", "wav", "wav")]
)
@pytest.mark.parametrize(
    "stream_duration,format_duration,expected",
    [
        ("N/A", "5.0", 5.0),
        ("", "5.0", 5.0),
        ("nan", "5.0", 5.0),
        ("inf", "5.0", 5.0),
        (None, "5.0", 5.0),
        ("3.0", "5.0", 3.0),
        ("N/A", "N/A", None),
        (None, None, None),
    ],
)
def test_reference_ffprobe_duration_fallback(
    monkeypatch,
    tmp_path,
    kind,
    formats,
    extension,
    stream_duration,
    format_duration,
    expected,
):
    from fastapi import HTTPException

    from rapid_mlx.routes import video

    monkeypatch.setattr(video.shutil, "which", lambda name: "/ffprobe")
    monkeypatch.setattr(video, "_ffmpeg_decodable_codecs", lambda: {"h264", "aac"})
    details = dict(
        streams=[
            dict(
                codec_type=kind,
                codec_name="h264" if kind == "video" else "aac",
                width=1920,
                height=1080,
            )
        ],
        format=dict(format_name=formats),
    )
    if stream_duration is not None:
        details["streams"][0]["duration"] = stream_duration
    if format_duration is not None:
        details["format"]["duration"] = format_duration
    monkeypatch.setattr(
        video.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=json.dumps(details)),
    )
    if expected is not None:
        assert video._probe_packed_reference(tmp_path / "file", "references[0]") == (
            kind,
            extension,
            expected,
        )
    else:
        with pytest.raises(HTTPException) as exc:
            video._probe_packed_reference(tmp_path / "file", "references[0]")
        assert exc.value.status_code == 400
        assert (
            exc.value.detail == f"references[0]: {kind} duration must be 2..15 seconds"
        )


@pytest.mark.parametrize(
    "width,height,accepted",
    [
        (3840, 2160, True),
        (2160, 3840, True),
        (7680, 1080, False),
        (1080, 7680, False),
        (1920, 4320, False),
        (4096, 2160, False),
        (2400, 2400, False),
    ],
)
def test_reference_ffprobe_video_resolution(
    monkeypatch, tmp_path, width, height, accepted
):
    from fastapi import HTTPException

    from rapid_mlx.routes import video

    monkeypatch.setattr(video.shutil, "which", lambda name: "/ffprobe")
    monkeypatch.setattr(video, "_ffmpeg_decodable_codecs", lambda: {"h264", "aac"})
    details = dict(
        streams=[
            dict(codec_type="video", codec_name="h264", width=width, height=height)
        ],
        format=dict(format_name="mov,mp4,m4a,3gp,3g2,mj2", duration="2"),
    )
    monkeypatch.setattr(
        video.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=json.dumps(details)),
    )
    if accepted:
        assert (
            video._probe_packed_reference(tmp_path / "file", "references[0]")[0]
            == "video"
        )
    else:
        with pytest.raises(HTTPException) as exc:
            video._probe_packed_reference(tmp_path / "file", "references[0]")
        assert exc.value.status_code == 400
        assert exc.value.detail == "references[0]: video exceeds 3840x2160 pixels"


@pytest.mark.parametrize(
    "width,height,accepted",
    [(1920, 480, True), (1921, 480, False), (480, 1921, False)],
)
def test_reference_ffprobe_video_aspect_ratio(
    monkeypatch, tmp_path, width, height, accepted
):
    from fastapi import HTTPException

    from rapid_mlx.routes import video

    monkeypatch.setattr(video.shutil, "which", lambda name: "/ffprobe")
    monkeypatch.setattr(video, "_ffmpeg_decodable_codecs", lambda: {"h264", "aac"})
    details = dict(
        streams=[
            dict(codec_type="video", codec_name="h264", width=width, height=height)
        ],
        format=dict(format_name="mov,mp4,m4a,3gp,3g2,mj2", duration="2"),
    )
    monkeypatch.setattr(
        video.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=json.dumps(details)),
    )
    if accepted:
        assert video._probe_packed_reference(tmp_path / "file", "references[0]") == (
            "video",
            "mp4",
            2.0,
        )
    else:
        with pytest.raises(HTTPException) as exc:
            video._probe_packed_reference(tmp_path / "file", "references[0]")
        assert exc.value.status_code == 400
        assert (
            exc.value.detail
            == "references[0]: video aspect ratio must be within 1:4 and 4:1"
        )


@pytest.mark.parametrize("initial_mode", ["ref2va", "fl2va"])
def test_first_load_partition(monkeypatch, tmp_path, array_core, initial_mode):
    import huggingface_hub

    from rapid_mlx.video import reference_media

    monkeypatch.setattr(reference_media, "install_reference_decoders", lambda: None)

    from rapid_mlx import _mirror, model_downloads
    from rapid_mlx.video import encoding

    downloads = []
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda *a, **k: None)
    monkeypatch.setattr(model_downloads, "disabled", lambda: False)
    monkeypatch.setattr(_mirror, "_cache_disk_space", lambda: (tmp_path, 10**12))
    loads = []
    module = ModuleType("mlx_vlm.generate")

    def load(path, *, workflow):
        if workflow == "ref2va":
            assert downloads[:2] == [
                ((repo, revision), {"allow_patterns": list(files)})
                for repo, revision, files in flex.REF2VA_PAYLOADS
            ]
        loads.append(workflow)
        return SimpleNamespace(pipeline=SimpleNamespace())

    module.load_video_generation_model = load
    module.VideoGenerationRequest = lambda **kw: SimpleNamespace(**kw)
    module.generate_video = lambda *a: SimpleNamespace(
        frames=np.zeros((1, 1, 1, 3)), fps=24, audio=None, sampling_rate=None
    )
    monkeypatch.setitem(sys.modules, module.__name__, module)
    refs_module = ModuleType("mlx_vlm.generate.video_generation")
    refs_module.VideoReference = lambda kind, path: (kind, path)
    monkeypatch.setitem(sys.modules, refs_module.__name__, refs_module)
    monkeypatch.setattr(
        _mirror,
        "pinned_snapshot_download",
        lambda *a, **k: downloads.append((a, k)) or str(tmp_path),
    )
    monkeypatch.setattr(
        encoding,
        "encode_rgb_video",
        lambda frames, path, *a, **k: path.write_bytes(b"mp4"),
    )
    engine = flex.PikaH3FlexVideoEngine("pika-h3-flex")
    engine._prepare_model = lambda model: setattr(
        engine, "_resident_partition", engine._needed_partition
    )
    engine._prepare_request = lambda seed: None
    try:
        engine._generate(
            prompt="test",
            output_path=tmp_path / "out",
            width=832,
            height=480,
            num_frames=124,
            fps=24,
            seed=1,
            image=None,
            references=[("image", tmp_path / "image")]
            if initial_mode == "ref2va"
            else None,
        )
        assert loads == [initial_mode]
        expected = list(flex.REF2VA_PAYLOADS) if initial_mode == "ref2va" else []
        expected.append((h3.REPO, h3.REVISION, flex.BASE_FILES))
        assert downloads == [
            ((repo, rev), {"allow_patterns": list(files)})
            for repo, rev, files in expected
        ]
        if initial_mode == "fl2va":
            assert not any(
                name.startswith(("transformer_ref/", "ref2va/"))
                for _, kwargs in downloads
                for name in kwargs["allow_patterns"]
            )
    finally:
        engine.stop()


def test_base_engine_rejects_references(tmp_path):
    engine = h3.MiniMaxH3VideoEngine("minimax-h3")
    try:
        assert not engine.supports_references
        with pytest.raises(
            h3.MiniMaxH3RequestError,
            match="references are not supported by this model",
        ):
            engine._generate(
                prompt="test",
                output_path=tmp_path / "out",
                width=832,
                height=480,
                num_frames=124,
                fps=24,
                seed=1,
                image=None,
                references=[("image", tmp_path / "image")],
            )
    finally:
        engine.stop()


def test_reference_preprocessing(monkeypatch, array_core):
    from dataclasses import dataclass

    # Keep the dependency's reference contracts, but fake its media operations
    # and arrays so no decoder, model, or Metal device runs.
    @dataclass
    class Prepared:
        kind: str
        has_audio: bool

    refs_module = ModuleType("mlx_vlm.models.minimax_h3.references")
    refs_module.MiniMaxH3PreparedReference = Prepared
    refs_module.validate_references = lambda refs: None
    monkeypatch.setitem(sys.modules, refs_module.__name__, refs_module)
    media = ModuleType("mlx_vlm.models.minimax_h3.processing")
    media.decode_image = lambda image: image
    media.prepare_reference_image = lambda image, h, w: ("image", h, w)
    media.decode_video = lambda video: (video, 12)
    media.resample_reference_frames = lambda frames, fps: np.repeat(frames, 2, axis=0)
    media.resize_lanczos = lambda frames, h, w: (len(frames), h, w)
    media.decode_audio = lambda audio: (audio, 16000)
    rates = []

    def resample(waveform, rate, target):
        rates.append((waveform.shape, rate, target))
        return np.repeat(waveform, 2, axis=-1)

    media.resample_waveform = resample
    parent = ModuleType("mlx_vlm.models.minimax_h3")
    parent.processing = media
    monkeypatch.setitem(sys.modules, parent.__name__, parent)
    monkeypatch.setitem(sys.modules, media.__name__, media)
    array_core.repeat = np.repeat
    pipeline = SimpleNamespace(
        _validate_duration=lambda frames: frames,
        audio_vae=SimpleNamespace(config=SimpleNamespace(sampling_rate=32000)),
    )
    image = SimpleNamespace(
        kind="image", has_audio=False, image=np.zeros((1080, 1920, 3))
    )
    audio = SimpleNamespace(
        kind="audio", has_audio=True, audio=np.zeros((1, 16000 * 6)), sample_rate=None
    )
    video = SimpleNamespace(
        kind="video",
        has_audio=False,
        video=np.zeros((24, 16, 32, 3)),
        fps=None,
        audio=None,
    )
    prepared, frames = flex.prepare_flex_references(
        pipeline, [image, video, audio], 124, (480, 832)
    )
    assert frames == 124
    assert prepared[0].image == ("image", 1024, 1824)
    assert prepared[1].frames[0] == 48
    assert prepared[1].frames[1:] == (448, 896)
    assert rates == [((2, 96000), 16000, 32000)]
    assert prepared[2].waveform.shape == (2, 165333)
    # The 12-fps source has 2 seconds before truncation, even when resampling
    # and the requested output frame cap change its prepared length.
    prepared, _ = flex.prepare_flex_references(pipeline, [video], 22, (480, 832))
    assert prepared[0].frames[0] == 22
    with pytest.raises(ValueError, match="22 frames"):
        flex.prepare_flex_references(pipeline, [video], 21, (480, 832))
    with pytest.raises(ValueError, match="num_frames"):
        flex.prepare_flex_references(pipeline, [image], None, (480, 832))

    # File-backed video and audio share the soundtrack decoder, including
    # containers with no decodable audio stream.
    source_frames = video.video
    media.decode_video = lambda path: (source_frames, 12)
    decoded = []
    soundtrack = None

    def decode_soundtrack(path):
        decoded.append(path)
        return soundtrack

    media.decode_video_soundtrack = decode_soundtrack
    media.decode_audio = lambda audio: pytest.fail("unexpected fallback audio decoder")
    for path in ("video.mp4", Path("video.mp4")):
        video.video = path
        for soundtrack in (None, (np.ones((1, 32000)), 16000)):
            rates.clear()
            prepared, _ = flex.prepare_flex_references(
                pipeline, [video], 124, (480, 832)
            )
            assert decoded[-1] == path
            assert prepared[0].has_audio == (soundtrack is not None)
            if soundtrack is None:
                assert not hasattr(prepared[0], "waveform")
                assert rates == []
            else:
                assert rates == [((2, 32000), 16000, 32000)]
                np.testing.assert_array_equal(prepared[0].waveform, np.ones((2, 64000)))

    for path in ("audio.wav", Path("audio.wav")):
        audio.audio = path
        soundtrack = (np.ones((2, 32000)), 16000)
        rates.clear()
        prepared, _ = flex.prepare_flex_references(
            pipeline, [image, audio], 124, (480, 832)
        )
        assert decoded[-1] == path
        assert rates == [((2, 32000), 16000, 32000)]
        np.testing.assert_array_equal(prepared[1].waveform, np.ones((2, 64000)))
        soundtrack = None
        with pytest.raises(
            ValueError, match="audio reference has no decodable audio stream"
        ):
            flex.prepare_flex_references(pipeline, [image, audio], 124, (480, 832))
        assert decoded[-1] == path

    for shape in ((3, 32000), (32000,)):
        soundtrack = (np.zeros(shape), 16000)
        with pytest.raises(
            ValueError, match="reference waveform must be mono or stereo"
        ):
            flex.prepare_flex_references(pipeline, [image, audio], 124, (480, 832))


def test_reference_runtime_forwarding(monkeypatch, tmp_path):
    from rapid_mlx.runtime import video_lane

    lane = video_lane.VideoEngine("pika-h3-flex")
    calls = []
    monkeypatch.setattr(lane._minimax_engine, "generate", lambda **kw: calls.append(kw))
    monkeypatch.setattr(lane, "_crop_generated_output", lambda **kw: None)
    refs = [("image", tmp_path / "image")]
    args = dict(
        prompt="test",
        output_path=tmp_path / "out",
        width=832,
        height=480,
        num_frames=124,
        fps=24,
        seed=1,
        image=None,
        references=refs,
    )
    try:
        lane.generate(**args)
        assert calls[-1]["references"] == refs
    finally:
        lane._minimax_engine.stop()
    for backend in (None, SimpleNamespace(supports_references=False)):
        lane._minimax_engine = backend
        with pytest.raises(
            video_lane.VideoRuntimeError, match="references are not supported"
        ):
            lane.generate(**args)


def test_flex_body_limit(monkeypatch):
    from fastapi import FastAPI, Request
    from fastapi.testclient import TestClient

    from rapid_mlx.routes import video

    fake = SimpleNamespace(_minimax_engine=SimpleNamespace(supports_references=True))
    monkeypatch.setattr(video, "_video_engine", lambda: fake)

    async def authorized(scope):
        pass

    monkeypatch.setattr(video, "verify_api_key_for_scope", authorized)
    app = FastAPI()

    @app.post("/v1/videos")
    @app.post("/v1/videos/extend")
    async def receive(request: Request):
        return {"bytes": len(await request.body())}

    video.install_video_body_limit_middleware(app)
    with TestClient(app) as client:
        # A video reference above the legacy 21 MB body cap is admitted for Flex.
        response = client.post(
            "/v1/videos",
            content=b"x",
            headers={"Content-Length": str(30 * 1024 * 1024)},
        )
        assert response.status_code == 200
        response = client.post(
            "/v1/videos",
            content=b"x",
            headers={"Content-Length": str(video._FLEX_REQUEST_BYTES + 1)},
        )
        assert response.status_code == 413
        response = client.post(
            "/v1/videos/extend",
            content=b"x",
            headers={"Content-Length": str(30 * 1024 * 1024)},
        )
        assert response.status_code == 413
        fake._minimax_engine.supports_references = False
        response = client.post(
            "/v1/videos",
            content=b"x",
            headers={"Content-Length": str(30 * 1024 * 1024)},
        )
        assert response.status_code == 413


def test_flex_streaming_body_limit(monkeypatch):
    from fastapi import FastAPI, Request
    from fastapi.testclient import TestClient

    from rapid_mlx.routes import video

    fake = SimpleNamespace(_minimax_engine=SimpleNamespace(supports_references=True))
    monkeypatch.setattr(video, "_video_engine", lambda: fake)
    monkeypatch.setattr(video, "_VIDEO_REQUEST_BYTES", 8)
    monkeypatch.setattr(video, "_FLEX_REQUEST_BYTES", 16)

    async def authorized(scope):
        pass

    monkeypatch.setattr(video, "verify_api_key_for_scope", authorized)
    app = FastAPI()

    @app.post("/v1/videos")
    async def receive(request: Request):
        assert "content-length" not in request.headers
        return {"bytes": len(await request.body())}

    video.install_video_body_limit_middleware(app)
    with TestClient(app) as client:
        response = client.post("/v1/videos", content=iter([b"x" * 8, b"x" * 9]))
        assert "content-length" not in response.request.headers
        assert response.status_code == 413

        response = client.post("/v1/videos", content=iter([b"x" * 8, b"x" * 4]))
        assert "content-length" not in response.request.headers
        assert response.status_code == 200
        assert response.json() == {"bytes": 12}


@pytest.mark.asyncio
async def test_run_job_forwards_references(monkeypatch, tmp_path):
    from rapid_mlx.routes import video

    refs = [("image", tmp_path / "ref-00.img")]
    calls = []

    def generate(**kw):
        calls.append(kw)
        kw["output_path"].write_bytes(b"mp4")

    engine = SimpleNamespace(video_family="minimax-h3", generate=generate)
    monkeypatch.setattr(video, "_persist_completed_job", lambda job: None)
    video.configure_video_jobs(tmp_path / "jobs")
    video.start_video_jobs()
    job = video._VideoJob(
        id="video_" + "1" * 32,
        model=flex.REPO,
        prompt="test",
        seconds="5",
        size="832x480",
    )
    (video._jobs_root / job.id).mkdir()
    try:
        await video._run_job(
            job,
            engine=engine,
            width=832,
            height=480,
            num_frames=124,
            fps=24,
            seed=1,
            image_path=None,
            negative_prompt=None,
            guidance_scale=None,
            conditioning_strength=None,
            references=refs,
        )
        assert calls[-1]["references"] == refs
        assert job.status == "completed"
    finally:
        video.configure_video_jobs(None)
        video.start_video_jobs()


@pytest.mark.parametrize("format", ["GIF", "BMP"])
def test_reference_unsupported_image(reference_app, format):
    from PIL import Image

    client, _, _ = reference_app
    output = io.BytesIO()
    Image.new("RGB", (2, 2)).save(output, format=format)
    response = client.post(
        "/v1/videos",
        data=dict(prompt="test", model="pika-h3-flex", size="832x480"),
        files=[("references", ("wrong.png", output.getvalue()))],
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "references[0]: unsupported file type"


@pytest.mark.parametrize("with_ref2va", [False, True])
def test_pull_optional_ref2va(monkeypatch, with_ref2va):
    import argparse

    from rapid_mlx import cli
    from rapid_mlx.runtime import video_lane

    calls = []
    monkeypatch.setattr(video_lane, "video_runtime_issue", lambda *a, **k: None)
    monkeypatch.setattr(
        cli, "_pull_repository", lambda a, **kw: calls.append((a.model, kw))
    )
    monkeypatch.setattr(cli, "_missing_download_bytes", lambda repo, **kw: 0)
    cli.pull_command(
        argparse.Namespace(
            model="pika-h3-flex", bits=None, format=None, with_ref2va=with_ref2va
        )
    )
    expected = (*flex.PAYLOADS, *(flex.REF2VA_PAYLOADS if with_ref2va else ()))
    assert calls == [
        (repo, dict(allow_patterns_override=list(files), revision_override=rev))
        for repo, rev, files in expected
    ]
    names = [name for _, kw in calls for name in kw["allow_patterns_override"]]
    assert any(name.startswith("transformer_ref/") for name in names) == with_ref2va
    assert any(name.startswith("ref2va/") for name in names) == with_ref2va
    before = len(calls)
    for model in ("minimax-h3", "qwen3.5-4b-4bit"):
        with pytest.raises(ValueError, match="--with-ref2va.*only.*Pika"):
            cli.pull_command(argparse.Namespace(model=model, with_ref2va=True))
    assert len(calls) == before


def test_pull_ref2va_parser():
    from rapid_mlx.cli_parser import build_parser

    args = build_parser().parse_args(["pull", "pika-h3-flex", "--with-ref2va"])
    assert args.with_ref2va
    assert not build_parser().parse_args(["pull", "pika-h3-flex"]).with_ref2va


@pytest.mark.parametrize("enough_space", [False, True])
def test_pull_ref2va_aggregate_disk_preflight(monkeypatch, tmp_path, enough_space):
    from rapid_mlx import _mirror, cli
    from rapid_mlx.cli_parser import build_parser
    from rapid_mlx.runtime import video_lane

    payloads = (*flex.PAYLOADS, *flex.REF2VA_PAYLOADS)
    sizes = {}
    for repo, revision, files in payloads:
        siblings = sizes.setdefault((repo, revision), {})
        siblings.update({name: 10**9 for name in files})
    cached_name = flex.PAYLOADS[0][2][0]
    cached_file = tmp_path / "cached"
    cached_file.write_bytes(b"cached")
    missing = [
        sum(10**9 for name in files if name != cached_name) for _, _, files in payloads
    ]
    required = int(sum(missing) * (1 + cli._DOWNLOAD_DISK_HEADROOM))
    available = required if enough_space else required - 1
    assert available >= max(missing) * (1 + cli._DOWNLOAD_DISK_HEADROOM)

    lookups, downloads, probes = [], [], []

    def model_info(repo, *, revision, files_metadata):
        assert files_metadata
        lookups.append((repo, revision))
        return SimpleNamespace(
            sha=revision,
            siblings=[
                SimpleNamespace(rfilename=name, size=size)
                for name, size in {
                    **sizes[(repo, revision)],
                    "unselected.bin": required * 10,
                }.items()
            ],
        )

    def cache_lookup(repo, filename, *, revision):
        assert (repo, revision) in sizes
        return str(cached_file) if filename == cached_name else None

    def disk_space():
        probes.append(True)
        return tmp_path, available

    def pull(args, **kwargs):
        assert len(lookups) == len(payloads)
        downloads.append((args.model, kwargs))

    monkeypatch.setattr(video_lane, "video_runtime_issue", lambda *a, **k: None)
    monkeypatch.setattr(_mirror, "_packaged_digest_info", lambda *a: None)
    monkeypatch.setattr(_mirror, "_cache_disk_space", disk_space)
    monkeypatch.setattr("huggingface_hub.model_info", model_info)
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", cache_lookup)
    monkeypatch.setattr(cli, "_pull_repository", pull)
    args = build_parser().parse_args(["pull", "pika-h3-flex", "--with-ref2va"])
    if enough_space:
        cli.pull_command(args)
        assert downloads == [
            (repo, dict(revision_override=rev, allow_patterns_override=list(files)))
            for repo, rev, files in payloads
        ]
    else:
        with pytest.raises(SystemExit) as exc:
            cli.pull_command(args)
        assert exc.value.code == 1
        assert downloads == []
    assert lookups == [(repo, rev) for repo, rev, _ in payloads]
    assert probes == [True]


def test_ref2va_missing_bytes_with_cached_base_config(monkeypatch, tmp_path):
    from rapid_mlx import _mirror, cli

    cached_file = tmp_path / "cached"
    cached_file.write_bytes(b"cached")
    cached_names = {"config.json", *flex.BASE_FILES}

    def cache_lookup(repo, filename, *, revision=None):
        assert repo == h3.REPO
        assert revision in (None, h3.REVISION)
        return str(cached_file) if filename in cached_names else None

    def model_info(repo, *, revision, files_metadata):
        assert repo == h3.REPO
        assert revision == h3.REVISION
        assert files_metadata
        return SimpleNamespace(
            sha=revision,
            siblings=[
                SimpleNamespace(rfilename=name, size=size)
                for name, size in {
                    **dict.fromkeys(cached_names, 1),
                    **flex.REF2VA_FILE_SIZES,
                }.items()
            ],
        )

    monkeypatch.setattr(_mirror, "_packaged_digest_info", lambda *a: None)
    monkeypatch.setattr("huggingface_hub.model_info", model_info)
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", cache_lookup)

    # A cached root config must not bypass sizing explicitly selected files.
    assert cli._missing_download_bytes(
        h3.REPO,
        revision_override=h3.REVISION,
        allow_patterns=list(flex.REF2VA_BASE_FILES),
    ) == sum(
        size
        for name, size in flex.REF2VA_FILE_SIZES.items()
        if name.startswith("transformer_ref/")
    )


def test_pull_ref2va_aggregate_disk_preflight_force(monkeypatch, tmp_path):
    from rapid_mlx import _mirror, cli
    from rapid_mlx.cli_parser import build_parser
    from rapid_mlx.runtime import video_lane

    downloads = []
    monkeypatch.setattr(video_lane, "video_runtime_issue", lambda *a, **k: None)
    monkeypatch.setattr(_mirror, "_cache_disk_space", lambda: (tmp_path, 0))
    monkeypatch.setattr(cli, "_missing_download_bytes", lambda *a, **k: 10**9)
    monkeypatch.setattr(
        cli,
        "_pull_repository",
        lambda args, **kwargs: downloads.append((args.model, kwargs)),
    )
    args = build_parser().parse_args(["pull", "pika-h3-flex", "--with-ref2va"])
    args.force_disk_check = True

    cli.pull_command(args)

    assert downloads == [
        (repo, dict(revision_override=rev, allow_patterns_override=list(files)))
        for repo, rev, files in (*flex.PAYLOADS, *flex.REF2VA_PAYLOADS)
    ]


@pytest.fixture
def ref2va_cache(monkeypatch, tmp_path):
    import huggingface_hub

    from rapid_mlx import _mirror, model_downloads

    cached = set()
    present = tmp_path / "cached"
    present.write_bytes(b"x")
    monkeypatch.setattr(
        huggingface_hub,
        "try_to_load_from_cache",
        lambda repo, filename, *, revision: (
            str(present) if (repo, revision, filename) in cached else None
        ),
    )
    monkeypatch.setattr(model_downloads, "disabled", lambda: False)
    monkeypatch.setattr(_mirror, "_cache_disk_space", lambda: (tmp_path, 10**12))
    return cached


def test_local_model_skips_ref2va_assets(monkeypatch, tmp_path):
    local_model = tmp_path / "model"
    local_model.mkdir()
    monkeypatch.setattr(
        flex, "ensure_ref2va_assets", lambda: pytest.fail("local model download")
    )
    engine = flex.PikaH3FlexVideoEngine(str(local_model))
    try:
        engine._ensure_assets("ref2va")
    finally:
        engine.stop()


def test_ref2va_download_and_warm_cache(monkeypatch, tmp_path, ref2va_cache, caplog):
    from rapid_mlx import _mirror, model_downloads

    calls = []

    def download(repo, rev, *, allow_patterns):
        calls.append((repo, rev, tuple(allow_patterns)))
        ref2va_cache.update((repo, rev, name) for name in allow_patterns)

    monkeypatch.setattr(_mirror, "pinned_snapshot_download", download)
    with caplog.at_level("INFO"):
        flex.ensure_ref2va_assets()
    assert calls == list(flex.REF2VA_PAYLOADS)
    assert (
        "Downloading Pika H3 Flex Ref2VA assets (69.0 GB, first Ref2VA request only)"
        in caplog.text
    )
    assert "assets downloaded (69.0 GB)" in caplog.text
    monkeypatch.setattr(model_downloads, "disabled", lambda: True)
    monkeypatch.setattr(
        _mirror, "_cache_disk_space", lambda: pytest.fail("warm disk probe")
    )
    flex.ensure_ref2va_assets()
    assert len(calls) == 2


def test_ref2va_download_failure_is_actionable(monkeypatch, tmp_path, ref2va_cache):
    from rapid_mlx import _mirror, model_downloads

    def download(repo, rev, *, allow_patterns):
        raise OSError("connection reset")

    monkeypatch.setattr(model_downloads, "disabled", lambda: False)
    monkeypatch.setattr(_mirror, "_cache_disk_space", lambda: (tmp_path, 10**15))
    monkeypatch.setattr(_mirror, "pinned_snapshot_download", download)
    with pytest.raises(h3.MiniMaxH3BackendError) as exc:
        flex.ensure_ref2va_assets()
    message = str(exc.value)
    assert "Ref2VA asset download failed" in message
    assert "rapid-mlx pull pika-h3-flex --with-ref2va" in message
    assert "connection reset" in message


@pytest.mark.parametrize("disabled", [False, True])
def test_ref2va_missing_error(monkeypatch, tmp_path, ref2va_cache, disabled):
    from rapid_mlx import _mirror, model_downloads

    monkeypatch.setattr(model_downloads, "disabled", lambda: disabled)
    monkeypatch.setattr(_mirror, "_cache_disk_space", lambda: (tmp_path, 1_000_000_000))
    with pytest.raises(h3.MiniMaxH3BackendError) as exc:
        flex.ensure_ref2va_assets()
    message = str(exc.value)
    assert "rapid-mlx pull pika-h3-flex --with-ref2va" in message
    if disabled:
        assert "downloads are disabled" in message
    else:
        assert "additional 69.0 GB" in message
        assert f"only 1.0 GB free on {tmp_path}" in message


def test_ref2va_disk_check_failure_is_actionable(monkeypatch, ref2va_cache):
    from rapid_mlx import _mirror

    error = OSError("statvfs failed")

    def disk_space():
        raise error

    monkeypatch.setattr(_mirror, "_cache_disk_space", disk_space)
    monkeypatch.setattr(
        _mirror,
        "pinned_snapshot_download",
        lambda *a, **kw: pytest.fail("download attempted after disk check failure"),
    )
    with pytest.raises(h3.MiniMaxH3BackendError) as exc:
        flex.ensure_ref2va_assets()
    assert str(exc.value) == (
        "Could not check free disk space for the Pika H3 Flex Ref2VA download "
        "(statvfs failed). Run `rapid-mlx pull pika-h3-flex --with-ref2va` "
        "to pre-fetch the assets."
    )
    assert exc.value.__cause__ is error


def test_ref2va_disk_counts_only_missing(monkeypatch, ref2va_cache, tmp_path):
    from rapid_mlx import _mirror

    # A cached second transformer leaves just the 2.77 GB adapter + manifest.
    ref2va_cache.update(
        (flex.BASE_REPO, flex.BASE_REVISION, f) for f in flex.REF2VA_BASE_FILES
    )
    monkeypatch.setattr(_mirror, "_cache_disk_space", lambda: (tmp_path, 2_000_000_000))
    with pytest.raises(h3.MiniMaxH3BackendError, match="additional 2.8 GB"):
        flex.ensure_ref2va_assets()
    calls = []
    monkeypatch.setattr(_mirror, "_cache_disk_space", lambda: (tmp_path, 3_100_000_000))
    monkeypatch.setattr(
        _mirror, "pinned_snapshot_download", lambda *a, **kw: calls.append((a, kw))
    )
    flex.ensure_ref2va_assets()
    assert len(calls) == 2


def test_ref2va_disk_failure_keeps_resident_transformer(
    monkeypatch, ref2va_cache, tmp_path
):
    from rapid_mlx import _mirror

    monkeypatch.setattr(_mirror, "_cache_disk_space", lambda: (tmp_path, 0))
    engine = flex.PikaH3FlexVideoEngine("pika-h3-flex")
    trunk = object()
    engine._model = SimpleNamespace(pipeline=SimpleNamespace(transformer=trunk))
    engine._resident_partition = "fl2va"
    try:
        with pytest.raises(h3.MiniMaxH3BackendError, match="69.0 GB"):
            engine._ensure_partition("ref2va")
        assert engine._model.pipeline.transformer is trunk
        assert engine._resident_partition == "fl2va"
    finally:
        engine.stop()


def test_pinned_payload_sizes(monkeypatch):
    import huggingface_hub

    from rapid_mlx import model_sizes
    from scripts.gen_model_sizes import _size_one

    # Metadata-only get_paths_info capture at BASE_REVISION / REVISION.
    sizes = json.loads(
        (Path(__file__).parent / "fixtures/pika_h3_flex_sizes.json").read_text()
    )

    def info(repo, *, revision, **kw):
        assert revision == (flex.REVISION if repo == flex.REPO else flex.BASE_REVISION)
        return SimpleNamespace(
            siblings=[
                SimpleNamespace(rfilename=name, size=size)
                for name, size in sizes[repo].items()
            ]
        )

    monkeypatch.setattr(huggingface_hub, "model_info", info)
    total = sum(sizes[repo][f] for repo, _, files in flex.PAYLOADS for f in files)
    extra = sum(
        sizes[repo][f] for repo, _, files in flex.REF2VA_PAYLOADS for f in files
    )
    assert total == 146_818_516_457
    assert extra == flex.REF2VA_EXTRA_BYTES == 69_047_844_410
    assert total + extra == 215_866_360_867
    assert _size_one(flex.REPO) == (flex.REPO, total)
    assert model_sizes.size_bytes(flex.REPO) == total
    for repo, _, files in flex.REF2VA_PAYLOADS:
        assert {f: sizes[repo][f] for f in files} == {
            f: flex.REF2VA_FILE_SIZES[f] for f in files
        }


@pytest.mark.asyncio
@pytest.mark.parametrize("disabled", [False, True])
async def test_ref2va_asset_error_surfaces_in_failed_job(
    monkeypatch, tmp_path, ref2va_cache, disabled
):
    from rapid_mlx import _mirror, model_downloads
    from rapid_mlx.routes import video
    from rapid_mlx.runtime import video_lane

    monkeypatch.setattr(model_downloads, "disabled", lambda: disabled)
    monkeypatch.setattr(_mirror, "_cache_disk_space", lambda: (tmp_path, 0))
    lane = video_lane.VideoEngine("pika-h3-flex")
    monkeypatch.setattr(
        lane._minimax_engine, "generate", lambda **kw: flex.ensure_ref2va_assets()
    )
    video.configure_video_jobs(tmp_path / "jobs")
    video.start_video_jobs()
    job = video._VideoJob(
        id="video_" + "2" * 32,
        model=flex.REPO,
        prompt="test",
        seconds="5",
        size="832x480",
    )
    (video._jobs_root / job.id).mkdir()
    try:
        await video._run_job(
            job,
            engine=lane,
            width=832,
            height=480,
            num_frames=124,
            fps=24,
            seed=1,
            image_path=None,
            negative_prompt=None,
            guidance_scale=None,
            conditioning_strength=None,
            references=[("image", tmp_path / "ref")],
        )
        assert job.status == "failed"
        assert "rapid-mlx pull pika-h3-flex --with-ref2va" in job.error["message"]
        assert (
            "downloads are disabled" if disabled else "additional 69.0 GB"
        ) in job.error["message"]
    finally:
        lane._minimax_engine.stop()
        video.configure_video_jobs(None)
        video.start_video_jobs()


@pytest.mark.parametrize("mode", ["fl2va", "ref2va"])
def test_adapter_download_is_mode_specific(monkeypatch, tmp_path, mode):
    from rapid_mlx import _mirror

    calls = []
    monkeypatch.setattr(
        _mirror,
        "pinned_snapshot_download",
        lambda *a, **kw: calls.append((a, kw)) or str(tmp_path),
    )
    monkeypatch.setattr(flex, "verify_adapter", lambda *a: None)
    monkeypatch.setattr(flex, "merge_lora", lambda *a, **kw: None)
    engine = flex.PikaH3FlexVideoEngine("pika-h3-flex")
    try:
        engine._merge_adapter(SimpleNamespace(transformer=object()), mode)
        assert calls == [
            (
                (flex.REPO, flex.REVISION),
                {
                    "allow_patterns": list(
                        flex.REF2VA_FILES if mode == "ref2va" else flex.FILES
                    )
                },
            )
        ]
        other = "fl2va/" if mode == "ref2va" else "ref2va/"
        assert not any(name.startswith(other) for name in calls[0][1]["allow_patterns"])
    finally:
        engine.stop()


@pytest.mark.parametrize("field", ["input_reference", "last_frame", "references"])
def test_garbage_upload_errors_are_path_free(monkeypatch, reference_app, field):
    client, _, _ = reference_app
    if field == "references":
        from rapid_mlx.routes import video

        # Exercise the probe failure path without depending on a host ffprobe.
        def reject(*args, **kwargs):
            raise video.subprocess.CalledProcessError(1, ["ffprobe"])

        monkeypatch.setattr(video.shutil, "which", lambda name: "/ffprobe")
        monkeypatch.setattr(video.subprocess, "run", reject)
    response = client.post(
        "/v1/videos",
        data={
            "prompt": "test",
            "model": "pika-h3-flex",
            "seconds": "5",
            "size": "832x480",
        },
        files=[(field, ("reference.img", b"not an image", "image/png"))],
    )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert field in detail
    assert "/" not in detail
    assert "reference.img" not in detail
    assert "tmp" not in detail


def test_ref2va_progress_counts_only_missing_assets(monkeypatch, ref2va_cache):
    from rapid_mlx import _mirror

    repo, rev, files = flex.REF2VA_PAYLOADS[0]
    ref2va_cache.update((repo, rev, name) for name in files)
    expected = sum(flex.REF2VA_FILE_SIZES[name] for name in flex.REF2VA_PAYLOADS[1][2])
    events = []
    requests = []

    def download(repo, rev, *, allow_patterns, on_bytes, **kwargs):
        requests.extend(allow_patterns)
        for name in allow_patterns:
            on_bytes(flex.REF2VA_FILE_SIZES[name])

    monkeypatch.setattr(_mirror, "pinned_snapshot_download", download)
    flex.ensure_ref2va_assets(on_progress=lambda *args: events.append(args))
    assert requests == list(flex.REF2VA_PAYLOADS[1][2])
    assert events[0] == ("downloading", 0, expected)
    assert events[-2] == ("downloading", expected, expected)
    assert events[-1] == (None, 0, 0)


@pytest.mark.parametrize("kind", ["video", "audio"])
@pytest.mark.parametrize("advertised", [False, True])
def test_reference_http_rejects_missing_decoder(
    reference_app, monkeypatch, kind, advertised
):
    from rapid_mlx.routes import video

    client, _, calls = reference_app
    monkeypatch.setattr(
        video,
        "_reference_decoder_kinds",
        lambda: frozenset({kind} if advertised else set()),
    )
    monkeypatch.setattr(
        video,
        "_probe_packed_reference",
        lambda path, field: (kind, "mp4" if kind == "video" else "wav", 2.0),
    )
    response = post_references(client, ["image", kind])
    if advertised:
        assert response.status_code == 200, response.text
        assert [k for k, _ in calls[-1]["references"]] == ["image", kind]
        return
    assert response.status_code == 503
    assert response.json()["detail"] == f"{kind} references require media decoders"
    assert not calls


def test_reference_http_without_ffprobe_keeps_existing_error(
    reference_app, monkeypatch
):
    from rapid_mlx.routes import video

    client, _, calls = reference_app
    monkeypatch.setattr(video, "_reference_decoder_kinds", lambda: frozenset())
    monkeypatch.setattr(video.shutil, "which", lambda name: None)
    response = post_references(client, ["image", "video"])
    assert response.status_code == 503
    assert response.json()["detail"] == "references require ffprobe"
    assert not calls


def test_reference_http_accepts_images_without_timed_decoders(
    reference_app, monkeypatch
):
    from rapid_mlx.routes import video

    client, _, calls = reference_app
    monkeypatch.setattr(video, "_reference_decoder_kinds", lambda: frozenset())
    response = post_references(client, ["image"])
    assert response.status_code == 200, response.text
    assert [kind for kind, _ in calls[-1]["references"]] == ["image"]


@pytest.mark.parametrize("container", ["matroska,webm", "mov,mp4,m4a,3gp,3g2,mj2"])
@pytest.mark.parametrize(
    "video_codec,audio_codec,decodable,rejected",
    [
        ("av1", None, {"h264", "aac"}, "av1 video"),
        ("prores", None, {"h264", "aac"}, "prores video"),
        ("h264", "ac3", {"h264", "aac"}, "ac3 audio"),
        (None, "ac3", {"h264", "aac"}, "ac3 audio"),
        ("av1", None, {"av1"}, None),
        ("prores", "ac3", {"prores", "ac3"}, None),
        (None, "ac3", {"ac3"}, None),
        ("h264", "aac", {"h264", "aac"}, None),
        (None, "aac", {"aac"}, None),
        ("missing", "missing", set(), "unknown video codec"),
        ("", None, {"h264"}, "unknown video codec"),
        ("h264", "missing", {"h264"}, "unknown audio codec"),
        ("h264", "", {"h264"}, "unknown audio codec"),
        (None, "", {"aac"}, "unknown audio codec"),
        (None, "missing", set(), "unknown audio codec"),
    ],
)
def test_reference_probe_checks_runtime_codecs(
    monkeypatch, tmp_path, container, video_codec, audio_codec, decodable, rejected
):
    from fastapi import HTTPException

    from rapid_mlx.routes import video

    streams = []
    if video_codec is not None:
        streams.append(dict(codec_type="video", width=64, height=48))
        if video_codec != "missing":
            streams[-1]["codec_name"] = video_codec
    if audio_codec is not None:
        streams.append(dict(codec_type="audio"))
        if audio_codec != "missing":
            streams[-1]["codec_name"] = audio_codec
    # Audio-only references require an audio-compatible container.
    if video_codec is None:
        container = "mov,mp4,m4a,3gp,3g2,mj2"
    payload = dict(
        streams=streams,
        format=dict(format_name=container, duration="2"),
    )
    monkeypatch.setattr(video.shutil, "which", lambda name: "/bin/" + name)
    monkeypatch.setattr(video, "_ffmpeg_decodable_codecs", lambda: frozenset(decodable))
    monkeypatch.setattr(
        video.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(stdout=json.dumps(payload)),
    )
    if rejected:
        with pytest.raises(HTTPException) as exc:
            video._probe_packed_reference(tmp_path / "clip", "references[0]")
        assert exc.value.status_code == 400
        assert exc.value.detail == (
            f"references[0]: {rejected} is not supported by this runtime"
        )
    else:
        kind, _, duration = video._probe_packed_reference(
            tmp_path / "clip", "references[0]"
        )
        assert kind == ("video" if video_codec is not None else "audio")
        assert duration == 2.0
