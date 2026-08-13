# SPDX-License-Identifier: Apache-2.0
"""Hermetic tests for the mflux image-generation lane.

No network, no disk, no real weights: the mflux model is replaced by a fake
that returns a tiny in-memory PIL image, so these exercise Rapid-MLX's own
validation / dispatch / transport contract rather than the diffusion pipeline.
"""

import base64
import io
import types

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from vllm_mlx.api.models import ImageGenerationRequest
from vllm_mlx.image.engine import (
    ImageGenerationCancelled,
    ImageGenerationEngine,
    ImageRuntimeError,
)
from vllm_mlx.runtime.image_lane import ImageEngine

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


class _FakeGeneratedImage:
    def __init__(self, image):
        self.image = image


class _FakeModel:
    """Stand-in for an mflux Flux1 / QwenImage model."""

    def __init__(self):
        self.calls = []

    def generate_image(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeGeneratedImage(Image.new("RGB", (8, 8), (200, 40, 40)))


# --------------------------------------------------------------------------- #
# Family detection
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "hf_path,expected_family,is_edit",
    [
        # Fast tab defaults — the "klein" / "z-image" tokens must win even
        # though the Klein repo id also contains "flux".
        ("Runpod/FLUX.2-klein-4B-mflux-4bit", "flux2-klein", False),
        ("filipstrand/Z-Image-Turbo-mflux-4bit", "z-image", False),
        ("black-forest-labs/FLUX.1-schnell", "flux-schnell", False),
        ("Qwen/Qwen-Image", "qwen-image", False),
        ("Qwen/Qwen-Image-Edit-2509", "qwen-image-edit", True),
    ],
)
def test_detect_family(hf_path, expected_family, is_edit):
    engine = ImageGenerationEngine(hf_path)
    assert engine.family == expected_family
    assert engine.is_edit is is_edit


def test_flux2_klein_supports_generation_and_editing():
    engine = ImageGenerationEngine("Runpod/FLUX.2-klein-4B-mflux-4bit")
    assert engine.supports_generation is True
    assert engine.supports_editing is True
    assert engine.default_steps == 4
    assert engine.default_edit_steps == 4
    assert engine.default_edit_guidance is None


@pytest.mark.parametrize(
    "hf_path,expected_default_steps",
    [
        ("Runpod/FLUX.2-klein-4B-mflux-4bit", 4),  # distilled turbo
        ("filipstrand/Z-Image-Turbo-mflux-4bit", 8),  # turbo, 8-step sweet spot
        ("black-forest-labs/FLUX.1-schnell", 4),
        ("Qwen/Qwen-Image", 20),  # non-distilled
    ],
)
def test_default_steps_is_family_aware(hf_path, expected_default_steps):
    assert ImageGenerationEngine(hf_path).default_steps == expected_default_steps


def test_unknown_family_raises():
    with pytest.raises(ImageRuntimeError, match="Unsupported image model"):
        ImageGenerationEngine("stabilityai/some-unwired-model")


@pytest.mark.parametrize(
    "hf_path,family",
    [
        # ``<n>bit`` convention — the repos the fast-tab aliases point at
        ("Runpod/FLUX.2-klein-4B-mflux-4bit", "flux2-klein"),
        ("filipstrand/Z-Image-Turbo-mflux-4bit", "z-image"),
        ("dhairyashil/FLUX.1-schnell-mflux-4bit", "flux-schnell"),
        ("OsaurusAI/Qwen-Image-mflux-4bit", "qwen-image"),
        # ``q<n>`` convention — the repo the qwen-image-edit-4bit alias points at
        ("OsaurusAI/Qwen-Image-Edit-mflux-q4", "qwen-image-edit"),
    ],
)
def test_prequantized_repo_disables_onload_quantize(hf_path, family):
    engine = ImageGenerationEngine(hf_path)
    assert engine._prequantized is True
    assert engine._quantize is None  # never re-quantize an already-quantized repo
    assert engine.family == family


@pytest.mark.parametrize(
    "hf_path",
    [
        # Base repos carry no quant tag — the leading "q" of "Qwen" must not be
        # mistaken for a "q<n>" tag, so on-load quantization stays enabled.
        "black-forest-labs/FLUX.1-schnell",
        "Qwen/Qwen-Image",
        "Qwen/Qwen-Image-Edit-2509",
    ],
)
def test_canonical_repo_keeps_onload_quantize(hf_path):
    engine = ImageGenerationEngine(hf_path)
    assert engine._prequantized is False
    assert engine._quantize == 4


# --------------------------------------------------------------------------- #
# ImageGenerationEngine._model_path_for_mflux
# --------------------------------------------------------------------------- #
def test_warm_cache_hands_mflux_a_local_directory(monkeypatch):
    """A verified-complete cache is passed to mflux as a path, not a repo id.

    mflux resolves a repo id through huggingface_hub on every build, and that
    revision lookup has no timeout — on a poisoned-DNS network it hangs a start
    whose weights are already on disk.
    """
    engine = ImageGenerationEngine("Runpod/FLUX.2-klein-4B-mflux-4bit")
    monkeypatch.setattr(
        "vllm_mlx._download_gate.mflux_local_snapshot",
        lambda repo: "/cache/snapshots/abc",
    )

    assert engine._model_path_for_mflux() == "/cache/snapshots/abc"


def test_unresolvable_cache_still_hands_mflux_the_repo_id(monkeypatch):
    """No local verdict → today's behavior, so a cold pull still happens."""
    engine = ImageGenerationEngine("Runpod/FLUX.2-klein-4B-mflux-4bit")
    monkeypatch.setattr(
        "vllm_mlx._download_gate.mflux_local_snapshot", lambda repo: None
    )

    assert engine._model_path_for_mflux() == "Runpod/FLUX.2-klein-4B-mflux-4bit"


def test_canonical_repo_still_defers_to_model_config(monkeypatch):
    """A non-prequantized repo keeps ``model_path=None``.

    mflux selects those weights through ``ModelConfig`` and quantizes on load;
    handing it a path instead would bypass that entirely.
    """
    engine = ImageGenerationEngine("black-forest-labs/FLUX.1-schnell")

    def _unexpected(repo):  # pragma: no cover — must never be consulted
        raise AssertionError("canonical repos must not probe the mflux cache")

    monkeypatch.setattr("vllm_mlx._download_gate.mflux_local_snapshot", _unexpected)

    assert engine._model_path_for_mflux() is None


# --------------------------------------------------------------------------- #
# ImageGenerationEngine.generate
# --------------------------------------------------------------------------- #
def test_generate_returns_png_bytes():
    engine = ImageGenerationEngine("black-forest-labs/FLUX.1-schnell")
    engine._model = _FakeModel()  # bypass lazy load / real weights
    png = engine.generate(prompt="a red fox", width=512, height=512, seed=7)
    assert png.startswith(_PNG_MAGIC)
    # It round-trips as a real PNG.
    assert Image.open(io.BytesIO(png)).size == (8, 8)


def test_generate_is_lazy(monkeypatch):
    engine = ImageGenerationEngine("black-forest-labs/FLUX.1-schnell")
    built = _FakeModel()
    monkeypatch.setattr(engine, "_build_model", lambda: built)
    assert engine._model is None  # not loaded at construction
    engine.generate(prompt="x", seed=1)
    assert engine._model is built  # loaded on first generate
    assert built.calls[0]["prompt"] == "x"


_MFLUX_REPO = "Runpod/FLUX.2-klein-4B-mflux-4bit"


def _seed_mflux_cache(tmp_path, monkeypatch, *, omit=None):
    """Lay down an mflux snapshot in a throwaway HF cache.

    ``omit`` drops one ``(component, shard)`` to model the state an
    interrupted pull leaves behind: indexes and small files present, one
    multi-gigabyte shard absent.
    """
    repo_root = tmp_path / "hf-cache" / "models--Runpod--FLUX.2-klein-4B-mflux-4bit"
    snap = repo_root / "snapshots" / ("e" * 40)
    (snap / "tokenizer").mkdir(parents=True)
    (snap / "tokenizer" / "tokenizer.json").write_text("{}")
    for component in ("transformer", "text_encoder", "vae"):
        component_dir = snap / component
        component_dir.mkdir()
        (component_dir / "model.safetensors.index.json").write_text(
            '{"weight_map": {"a": "0.safetensors", "b": "1.safetensors"}}'
        )
        for shard in ("0.safetensors", "1.safetensors"):
            if omit != (component, shard):
                (component_dir / shard).write_bytes(b"weights")
    (repo_root / "refs").mkdir(parents=True)
    (repo_root / "refs" / "main").write_text("e" * 40)
    monkeypatch.setattr(
        "huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path / "hf-cache")
    )


def test_partial_download_refuses_to_load(tmp_path, monkeypatch):
    """A half-downloaded checkpoint must fail loudly, before mflux touches it.

    mflux globs the component directory and ignores the index beside it, so
    the shards that did arrive would load and the run would render noise with
    no error anywhere. The whole point of the gate is that ``_build_model`` is
    never reached.
    """
    _seed_mflux_cache(tmp_path, monkeypatch, omit=("transformer", "0.safetensors"))
    engine = ImageGenerationEngine(_MFLUX_REPO)

    def _must_not_build():
        raise AssertionError("mflux was handed a partially-downloaded snapshot")

    monkeypatch.setattr(engine, "_build_model", _must_not_build)

    with pytest.raises(ImageRuntimeError, match="transformer/0.safetensors"):
        engine._ensure_loaded()


def test_complete_download_loads(tmp_path, monkeypatch):
    """The gate stays out of the way once every shard is on disk."""
    _seed_mflux_cache(tmp_path, monkeypatch)
    engine = ImageGenerationEngine(_MFLUX_REPO)
    built = _FakeModel()
    monkeypatch.setattr(engine, "_build_model", lambda: built)

    assert engine._ensure_loaded() is built


def test_uncached_model_is_not_treated_as_partial(tmp_path, monkeypatch):
    """No snapshot yet ⇒ no verdict ⇒ mflux does its own first-run download.

    Blocking here would break the cold-start path the gate is meant to
    protect, so "cannot tell" must never harden into "refuse".
    """
    monkeypatch.setattr(
        "huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path / "empty")
    )
    engine = ImageGenerationEngine(_MFLUX_REPO)
    built = _FakeModel()
    monkeypatch.setattr(engine, "_build_model", lambda: built)

    assert engine._ensure_loaded() is built


def test_edit_family_requires_image_paths():
    engine = ImageGenerationEngine("Qwen/Qwen-Image-Edit-2509")
    engine._model = _FakeModel()
    with pytest.raises(ImageRuntimeError, match="requires at least one input image"):
        engine.generate(prompt="make it blue", image_paths=None)


def test_txt2img_family_rejects_image_paths():
    engine = ImageGenerationEngine("Qwen/Qwen-Image")
    engine._model = _FakeModel()
    with pytest.raises(ImageRuntimeError, match="text-to-image only"):
        engine.generate(prompt="a cat", image_paths=["/tmp/x.png"])


def test_flux2_edit_passes_image_and_dimensions_through():
    engine = ImageGenerationEngine("Runpod/FLUX.2-klein-4B-mflux-4bit")
    fake = _FakeModel()
    engine._model = fake
    engine.generate(
        prompt="add a hat",
        image_paths=["/tmp/in.png"],
        width=768,
        height=1024,
        seed=3,
    )
    assert fake.calls[0]["image_paths"] == ["/tmp/in.png"]
    assert fake.calls[0]["width"] == 768
    assert fake.calls[0]["height"] == 1024


def test_flux2_switches_between_generation_and_edit_variants(monkeypatch):
    engine = ImageGenerationEngine("Runpod/FLUX.2-klein-4B-mflux-4bit")
    generation = _FakeModel()
    editing = _FakeModel()
    releases = []
    monkeypatch.setattr(engine, "_build_model", lambda: generation)
    monkeypatch.setattr(engine, "_build_edit_model", lambda: editing)
    monkeypatch.setattr(
        "vllm_mlx.image.engine._release_allocator_cache", lambda: releases.append(True)
    )

    engine.generate(prompt="a fox", seed=1)
    assert engine._model is generation
    assert engine._loaded_mode == "generation"

    engine.generate(prompt="add a hat", image_paths=["/tmp/in.png"], seed=2)
    assert engine._model is editing
    assert engine._loaded_mode == "edit"
    assert editing.calls[0]["image_paths"] == ["/tmp/in.png"]
    assert releases == [True]

    engine.generate(prompt="a dog", seed=3)
    assert engine._model is generation
    assert engine._loaded_mode == "generation"
    assert releases == [True, True]


def test_edit_family_passes_image_paths_through():
    engine = ImageGenerationEngine("Qwen/Qwen-Image-Edit-2509")
    fake = _FakeModel()
    engine._model = fake
    engine.generate(prompt="add a hat", image_paths=["/tmp/in.png"], seed=3)
    assert fake.calls[0]["image_paths"] == ["/tmp/in.png"]


def test_edit_forces_none_dimensions_even_when_size_requested():
    # Regression: mflux edit fixes the conditioning latents to a 1024²-area
    # canvas of the input; forcing a mismatched width/height (e.g. 512×512)
    # desyncs the RoPE ids and yields pure noise. The engine must hand mflux
    # None so it sizes the target to match the conditioning.
    engine = ImageGenerationEngine("Qwen/Qwen-Image-Edit-2509")
    fake = _FakeModel()
    engine._model = fake
    engine.generate(
        prompt="add a hat",
        image_paths=["/tmp/in.png"],
        width=512,
        height=512,
        seed=3,
    )
    assert fake.calls[0]["width"] is None
    assert fake.calls[0]["height"] is None


def test_txt2img_family_honors_requested_dimensions():
    # The noise trap is edit-only: text-to-image must still respect width/height.
    engine = ImageGenerationEngine("Qwen/Qwen-Image")
    fake = _FakeModel()
    engine._model = fake
    engine.generate(prompt="a cat", width=768, height=512, seed=3)
    assert fake.calls[0]["width"] == 768
    assert fake.calls[0]["height"] == 512


def test_progress_reporter_tracks_step_then_cancels():
    from vllm_mlx.image.engine import ImageGenerationCancelled

    engine = ImageGenerationEngine("Runpod/FLUX.2-klein-4B-mflux-4bit")
    engine._progress.update(total=4)
    engine._active_seq = 1  # simulate an in-flight run (set under the lock)

    class _Cfg:
        num_inference_steps = 4

    engine._reporter.call_in_loop(
        t=0, seed=1, prompt="x", latents=None, config=_Cfg(), time_steps=None
    )
    assert engine._progress["step"] == 1
    assert engine._progress["total"] == 4

    # A cancel targets the active run's seq; the next step then aborts by raising.
    engine.request_cancel()
    with pytest.raises(ImageGenerationCancelled):
        engine._reporter.call_in_loop(
            t=1, seed=1, prompt="x", latents=None, config=_Cfg(), time_steps=None
        )


def test_progress_snapshot_shape():
    engine = ImageGenerationEngine("Runpod/FLUX.2-klein-4B-mflux-4bit")
    snap = engine.progress_snapshot()
    assert set(snap) >= {"running", "step", "total", "elapsed_ms", "family"}
    assert snap["running"] is False  # nothing running yet
    assert snap["family"] == "flux2-klein"


def test_generate_resets_progress_and_registers_reporter():
    engine = ImageGenerationEngine("Runpod/FLUX.2-klein-4B-mflux-4bit")

    class _FakeRegistry:
        def __init__(self):
            self.registered = []

        def register(self, callback):
            self.registered.append(callback)

    class _ModelWithCallbacks(_FakeModel):
        def __init__(self):
            super().__init__()
            self.callbacks = _FakeRegistry()

    built = _ModelWithCallbacks()
    engine._build_model = lambda: built  # type: ignore[assignment]
    engine.generate(prompt="a fox", num_inference_steps=4, seed=1)
    # The progress/cancel reporter was registered on the model's mflux registry —
    # without this, live progress and cancellation would silently never fire.
    assert engine._reporter in built.callbacks.registered
    # After a clean run the snapshot is idle but carries the step total.
    assert engine._progress["running"] is False
    assert engine._progress["total"] == 4


def test_klein_drops_unsupported_negative_prompt():
    # Regression: FLUX.2 Klein's generate_image has no negative_prompt param,
    # so the engine must NOT forward it (passing it raises a TypeError).
    engine = ImageGenerationEngine("Runpod/FLUX.2-klein-4B-mflux-4bit")
    fake = _FakeModel()
    engine._model = fake
    engine.generate(prompt="a fox", negative_prompt="blurry, low quality", seed=1)
    assert "negative_prompt" not in fake.calls[0]


def test_supporting_family_keeps_negative_prompt():
    engine = ImageGenerationEngine("filipstrand/Z-Image-Turbo-mflux-4bit")
    fake = _FakeModel()
    engine._model = fake
    engine.generate(prompt="a fox", negative_prompt="blurry", seed=1)
    assert fake.calls[0]["negative_prompt"] == "blurry"


def test_guidance_omitted_when_unset():
    # Unset guidance is dropped so a guidance-distilled model uses its own
    # trained default rather than a forced value.
    engine = ImageGenerationEngine("Runpod/FLUX.2-klein-4B-mflux-4bit")
    fake = _FakeModel()
    engine._model = fake
    engine.generate(prompt="a fox", seed=1)
    assert "guidance" not in fake.calls[0]


def test_backend_failure_becomes_runtime_error():
    engine = ImageGenerationEngine("Qwen/Qwen-Image")

    class _Boom:
        def generate_image(self, **kwargs):
            raise ValueError("metal exploded")

    engine._model = _Boom()
    with pytest.raises(ImageRuntimeError, match="Image generation failed"):
        engine.generate(prompt="x")


# --------------------------------------------------------------------------- #
# ImageEngine adapter
# --------------------------------------------------------------------------- #
def test_image_engine_adapter_is_duck_typed():
    engine = ImageEngine("black-forest-labs/FLUX.1-schnell")
    assert engine.is_image_gen is True
    assert engine._loaded is True
    assert engine.family == "flux-schnell"
    assert engine.is_edit is False


def test_flux2_adapter_advertises_both_operations():
    engine = ImageEngine("Runpod/FLUX.2-klein-4B-mflux-4bit")
    assert engine.supports_generation is True
    assert engine.supports_editing is True
    assert engine.default_edit_steps == 4


def test_flux2_adapter_residency_can_preload_edit_variant(monkeypatch):
    engine = ImageEngine("Runpod/FLUX.2-klein-4B-mflux-4bit")
    modes = []
    monkeypatch.setattr(
        engine._engine,
        "_ensure_loaded",
        lambda *, for_edit=None: modes.append(for_edit),
    )

    engine.ensure_resident(mode="editing")

    assert modes == [True]


def test_image_adapter_residency_without_mode_preserves_family_default(monkeypatch):
    engine = ImageEngine("Qwen/Qwen-Image-Edit-2509")
    modes = []
    monkeypatch.setattr(
        engine._engine,
        "_ensure_loaded",
        lambda *, for_edit=None: modes.append(for_edit),
    )

    engine.ensure_resident()

    assert modes == [None]


# --------------------------------------------------------------------------- #
# Route: /v1/images/generations
# --------------------------------------------------------------------------- #
class _FakeImageEngine:
    is_image_gen = True

    def __init__(self, is_edit=False, default_steps=4):
        self.is_edit = is_edit
        self.default_steps = default_steps
        self.seeds = []
        self.image_paths_seen = []
        self.dims_seen = []
        self.steps_seen = []
        self.cancelled = False

    def progress_snapshot(self):
        return {
            "running": True,
            "step": 2,
            "total": 4,
            "elapsed_ms": 1200,
            "family": "flux2-klein",
        }

    def request_cancel(self):
        self.cancelled = True

    def generate(
        self,
        *,
        prompt,
        num_inference_steps,
        seed,
        guidance,
        negative_prompt,
        width=None,
        height=None,
        image_paths=None,
    ):
        # The edit route omits width/height (the engine derives them from the
        # input image); the generations route always supplies them.
        self.seeds.append(seed)
        self.image_paths_seen.append(image_paths)
        self.dims_seen.append((width, height))
        self.steps_seen.append(num_inference_steps)
        buffer = io.BytesIO()
        Image.new("RGB", (4, 4), (10, 20, 30)).save(buffer, format="PNG")
        return buffer.getvalue()


def _png_upload_bytes():
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), (90, 90, 90)).save(buffer, format="PNG")
    return buffer.getvalue()


def _patch_engine(monkeypatch, engine):
    monkeypatch.setattr(
        "vllm_mlx.config.get_config", lambda: types.SimpleNamespace(engine=engine)
    )


@pytest.fixture
def client():
    from vllm_mlx.server import app

    return TestClient(app)


def test_route_409_when_no_image_model(client, monkeypatch):
    _patch_engine(monkeypatch, None)
    resp = client.post("/v1/images/generations", json={"prompt": "a fox"})
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "image_model_not_loaded"


def test_route_409_when_edit_model_loaded(client, monkeypatch):
    _patch_engine(monkeypatch, _FakeImageEngine(is_edit=True))
    resp = client.post("/v1/images/generations", json={"prompt": "a fox"})
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "wrong_image_endpoint"


def test_route_400_url_response_format(client, monkeypatch):
    _patch_engine(monkeypatch, _FakeImageEngine())
    resp = client.post(
        "/v1/images/generations",
        json={"prompt": "a fox", "response_format": "url"},
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "unsupported_response_format"


def test_route_happy_path_returns_b64(client, monkeypatch):
    _patch_engine(monkeypatch, _FakeImageEngine())
    resp = client.post(
        "/v1/images/generations",
        json={"prompt": "a red fox", "size": "512x512", "seed": 42},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "created" in body and len(body["data"]) == 1
    raw = base64.b64decode(body["data"][0]["b64_json"])
    assert raw.startswith(_PNG_MAGIC)


def test_route_selects_resident_image_engine_by_model(client, monkeypatch):
    from vllm_mlx.runtime.model_registry import ModelEntry, ModelRegistry

    chat = types.SimpleNamespace(is_image_gen=False)
    image = _FakeImageEngine()
    registry = ModelRegistry()
    registry.add(ModelEntry(chat, "chat", "repo/chat"), is_default=True)
    registry.add(ModelEntry(image, "image", "repo/image"))
    monkeypatch.setattr(
        "vllm_mlx.config.get_config",
        lambda: types.SimpleNamespace(
            engine=chat,
            model_registry=registry,
            residency_manager=None,
        ),
    )

    resp = client.post(
        "/v1/images/generations",
        json={"model": "image", "prompt": "a resident fox", "seed": 7},
    )
    assert resp.status_code == 200
    assert image.seeds == [7]


def test_route_multi_image_offsets_seed(client, monkeypatch):
    engine = _FakeImageEngine()
    _patch_engine(monkeypatch, engine)
    resp = client.post(
        "/v1/images/generations",
        json={"prompt": "a fox", "n": 3, "seed": 100},
    )
    assert resp.status_code == 200
    assert len(resp.json()["data"]) == 3
    assert engine.seeds == [100, 101, 102]  # per-index seed offset


def test_progress_endpoint_returns_snapshot(client, monkeypatch):
    _patch_engine(monkeypatch, _FakeImageEngine())
    resp = client.get("/v1/images/progress")
    assert resp.status_code == 200
    body = resp.json()
    assert body["step"] == 2 and body["total"] == 4
    assert body["family"] == "flux2-klein"


def test_progress_endpoint_409_without_image_model(client, monkeypatch):
    _patch_engine(monkeypatch, None)
    assert client.get("/v1/images/progress").status_code == 409


def test_cancel_endpoint_signals_engine(client, monkeypatch):
    engine = _FakeImageEngine()
    _patch_engine(monkeypatch, engine)
    resp = client.post("/v1/images/cancel")
    assert resp.status_code == 200 and resp.json()["ok"] is True
    assert engine.cancelled is True


@pytest.mark.parametrize("bad_size", ["1023x1024", "100x100", "3000x512", "oops"])
def test_route_rejects_bad_size(client, monkeypatch, bad_size):
    _patch_engine(monkeypatch, _FakeImageEngine())
    resp = client.post("/v1/images/generations", json={"prompt": "x", "size": bad_size})
    assert resp.status_code in (400, 422)


@pytest.mark.parametrize("bad_guidance", [float("nan"), float("inf"), float("-inf")])
def test_request_model_rejects_nonfinite_guidance(bad_guidance):
    # NaN/inf fail the ge=0 / le=20 comparisons, so the bounds reject them.
    with pytest.raises(ValueError):
        ImageGenerationRequest(prompt="x", guidance=bad_guidance)


# --------------------------------------------------------------------------- #
# Route: /v1/images/edits
# --------------------------------------------------------------------------- #
def test_edit_409_when_no_image_model(client, monkeypatch):
    _patch_engine(monkeypatch, None)
    resp = client.post(
        "/v1/images/edits",
        files={"image": ("in.png", _png_upload_bytes(), "image/png")},
        data={"prompt": "add a hat"},
    )
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "image_model_not_loaded"


def test_edit_409_when_txt2img_model_loaded(client, monkeypatch):
    _patch_engine(monkeypatch, _FakeImageEngine(is_edit=False))
    resp = client.post(
        "/v1/images/edits",
        files={"image": ("in.png", _png_upload_bytes(), "image/png")},
        data={"prompt": "add a hat"},
    )
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "wrong_image_endpoint"


def test_edit_happy_path_returns_b64_and_passes_image(client, monkeypatch):
    engine = _FakeImageEngine(is_edit=True)
    _patch_engine(monkeypatch, engine)
    resp = client.post(
        "/v1/images/edits",
        files={"image": ("in.png", _png_upload_bytes(), "image/png")},
        data={"prompt": "make the sky blue", "size": "512x512", "seed": "5"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["data"]) == 1
    assert base64.b64decode(body["data"][0]["b64_json"]).startswith(_PNG_MAGIC)
    # The uploaded image was written to a temp file and passed to the engine.
    assert engine.image_paths_seen[0] is not None
    assert len(engine.image_paths_seen[0]) == 1
    # Even though the request carried size=512x512, the edit route does NOT
    # thread dimensions — the engine derives them from the input image.
    assert engine.dims_seen[0] == (None, None)


def test_dual_capability_model_works_on_generation_and_edit_routes(client, monkeypatch):
    engine = _FakeImageEngine(default_steps=4)
    engine.supports_generation = True
    engine.supports_editing = True
    engine.default_edit_steps = 4
    engine.default_edit_guidance = None
    _patch_engine(monkeypatch, engine)

    generated = client.post(
        "/v1/images/generations", json={"prompt": "a fox", "seed": 7}
    )
    edited = client.post(
        "/v1/images/edits",
        files={"image": ("in.png", _png_upload_bytes(), "image/png")},
        data={"prompt": "add a hat", "seed": "8"},
    )

    assert generated.status_code == 200
    assert edited.status_code == 200
    assert engine.steps_seen == [4, 4]


def test_edit_defaults_to_20_steps_when_unspecified(client, monkeypatch):
    # FLUX.1-schnell generation defaults to 4 distilled steps, but a
    # non-distilled edit needs ~20 to resolve structure; the edit route must
    # not inherit the 4-step generation default.
    engine = _FakeImageEngine(is_edit=True)
    _patch_engine(monkeypatch, engine)
    resp = client.post(
        "/v1/images/edits",
        files={"image": ("in.png", _png_upload_bytes(), "image/png")},
        data={"prompt": "add a hat"},
    )
    assert resp.status_code == 200
    assert engine.steps_seen == [20]


def test_edit_honors_explicit_steps(client, monkeypatch):
    engine = _FakeImageEngine(is_edit=True)
    _patch_engine(monkeypatch, engine)
    resp = client.post(
        "/v1/images/edits",
        files={"image": ("in.png", _png_upload_bytes(), "image/png")},
        data={"prompt": "add a hat", "steps": "8"},
    )
    assert resp.status_code == 200
    assert engine.steps_seen == [8]


def test_edit_multi_offsets_seed(client, monkeypatch):
    engine = _FakeImageEngine(is_edit=True)
    _patch_engine(monkeypatch, engine)
    resp = client.post(
        "/v1/images/edits",
        files={"image": ("in.png", _png_upload_bytes(), "image/png")},
        data={"prompt": "x", "n": "2", "seed": "50"},
    )
    assert resp.status_code == 200
    assert len(resp.json()["data"]) == 2
    assert engine.seeds == [50, 51]


def test_edit_400_empty_prompt(client, monkeypatch):
    _patch_engine(monkeypatch, _FakeImageEngine(is_edit=True))
    resp = client.post(
        "/v1/images/edits",
        files={"image": ("in.png", _png_upload_bytes(), "image/png")},
        data={"prompt": "   "},
    )
    assert resp.status_code == 400


def test_edit_400_bad_size(client, monkeypatch):
    _patch_engine(monkeypatch, _FakeImageEngine(is_edit=True))
    resp = client.post(
        "/v1/images/edits",
        files={"image": ("in.png", _png_upload_bytes(), "image/png")},
        data={"prompt": "x", "size": "100x100"},
    )
    assert resp.status_code == 400


def test_edit_400_empty_image(client, monkeypatch):
    _patch_engine(monkeypatch, _FakeImageEngine(is_edit=True))
    resp = client.post(
        "/v1/images/edits",
        files={"image": ("in.png", b"", "image/png")},
        data={"prompt": "x"},
    )
    assert resp.status_code == 400


@pytest.mark.parametrize(
    "field,value",
    [
        ("steps", "0"),
        ("steps", "500"),
        ("guidance", "nan"),
        ("guidance", "999"),
        ("seed", "-1"),
    ],
)
def test_edit_rejects_out_of_bounds_params(client, monkeypatch, field, value):
    # A raw multipart form must not smuggle a negative seed / non-finite
    # guidance / absurd step count past the validated bounds.
    _patch_engine(monkeypatch, _FakeImageEngine(is_edit=True))
    resp = client.post(
        "/v1/images/edits",
        files={"image": ("in.png", _png_upload_bytes(), "image/png")},
        data={"prompt": "x", field: value},
    )
    assert resp.status_code in (400, 422)


def test_edit_rejects_oversized_image(client, monkeypatch):
    _patch_engine(monkeypatch, _FakeImageEngine(is_edit=True))
    oversized = b"\x89PNG\r\n\x1a\n" + b"0" * (26 * 1024 * 1024)  # > 25 MB cap
    resp = client.post(
        "/v1/images/edits",
        files={"image": ("in.png", oversized, "image/png")},
        data={"prompt": "x"},
    )
    assert resp.status_code == 413


def test_edit_rejects_excessive_pixel_dimensions(client, monkeypatch):
    _patch_engine(monkeypatch, _FakeImageEngine(is_edit=True))
    source = io.BytesIO()
    Image.new("1", (8193, 1)).save(source, format="PNG")
    resp = client.post(
        "/v1/images/edits",
        files={"image": ("in.png", source.getvalue(), "image/png")},
        data={"prompt": "x"},
    )
    assert resp.status_code == 413
    assert "8192 px / 40 megapixel" in resp.json()["error"]["message"]


def test_edit_rejects_non_image_bytes(client, monkeypatch):
    _patch_engine(monkeypatch, _FakeImageEngine(is_edit=True))
    resp = client.post(
        "/v1/images/edits",
        files={"image": ("in.png", b"not an image", "image/png")},
        data={"prompt": "x"},
    )
    assert resp.status_code == 400


def test_edit_cancel_returns_cancelled_envelope(client, monkeypatch):
    from vllm_mlx.image.engine import ImageGenerationCancelled

    class _CancelEngine:
        is_image_gen = True
        is_edit = True

        def generate(self, **kwargs):
            raise ImageGenerationCancelled("cancelled")

    _patch_engine(monkeypatch, _CancelEngine())
    resp = client.post(
        "/v1/images/edits",
        files={"image": ("in.png", _png_upload_bytes(), "image/png")},
        data={"prompt": "x"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["cancelled"] is True
    assert body["data"] == []


def test_generate_honors_cancel_after_cold_load():
    # A cancel that lands during the warm-up load must abort before denoising,
    # not be reset away once the model finishes loading.
    engine = ImageGenerationEngine("Runpod/FLUX.2-klein-4B-mflux-4bit")
    built = _FakeModel()

    def _load_then_cancel():
        engine.request_cancel()  # simulate Cancel pressed during the load
        return built

    engine._build_model = _load_then_cancel  # type: ignore[assignment]
    with pytest.raises(ImageGenerationCancelled):
        engine.generate(prompt="x", seed=1)
    # Nothing was generated (aborted before the denoise call).
    assert built.calls == []


@pytest.mark.parametrize("fmt", ["url", "webp", "png"])
def test_edit_rejects_non_b64_response_format(client, monkeypatch, fmt):
    # The edit endpoint must reject any response_format other than b64_json
    # (the local lane has no object store for URLs), matching generations.
    _patch_engine(monkeypatch, _FakeImageEngine(is_edit=True))
    resp = client.post(
        "/v1/images/edits",
        files={"image": ("in.png", _png_upload_bytes(), "image/png")},
        data={"prompt": "x", "response_format": fmt},
    )
    assert resp.status_code == 400


@pytest.mark.parametrize("default_steps", [4, 8, 20])
def test_generations_uses_engine_default_steps(client, monkeypatch, default_steps):
    # A request without an explicit `steps` must inherit the engine's
    # family-specific default (Klein 4, Z-Image 8, Qwen 20), not a hardcoded 4.
    engine = _FakeImageEngine(default_steps=default_steps)
    _patch_engine(monkeypatch, engine)
    resp = client.post("/v1/images/generations", json={"prompt": "a fox"})
    assert resp.status_code == 200
    assert engine.steps_seen == [default_steps]


def test_image_alias_skips_the_mllm_routing_preflight(monkeypatch):
    """An image-gen alias must not run the MLLM-vs-text routing preflight.

    ``_ensure_routing_config`` materializes a checkpoint ``config.json`` so
    ``resolve_serving_lane`` can tell a hybrid VLM from a text model, and
    fails fast when it cannot. mflux-layout checkpoints keep every weight and
    config under ``transformer/`` / ``text_encoder/`` / ``vae/`` and ship no
    ``config.json`` at the checkpoint root, so that preflight can never
    succeed for them — which is how ``serve z-image-turbo`` refused a
    fully-cached 5.5 GB model with an error about hybrid-VLM misrouting, a
    hazard a diffusion model does not have. A diffusion alias branches
    straight to ImageEngine and never asks the question, so the preflight
    must be skipped rather than merely tolerated.
    """
    from vllm_mlx import server
    from vllm_mlx.runtime import image_lane

    def _unmaterializable(name):
        raise RuntimeError(
            f"Could not materialize the checkpoint config for {name!r} "
            "before selecting the serving lane."
        )

    built: dict[str, object] = {}

    class _LazyImageEngine:
        def __init__(self, model_name):
            built["model_name"] = model_name

    monkeypatch.setattr(server, "_ensure_routing_config", _unmaterializable)
    monkeypatch.setattr(image_lane, "ImageEngine", _LazyImageEngine)
    monkeypatch.setattr(
        image_lane, "require_image_runtime_or_exit", lambda *_a, **_kw: None
    )
    monkeypatch.setattr(
        "vllm_mlx.utils.generation_config.load_generation_config_sampling",
        lambda *_a, **_kw: {},
    )

    # ``load_model`` publishes module globals; keep them off sibling tests.
    saved = {
        attr: getattr(server, attr, None)
        for attr in ("_engine", "_model_name", "_model_alias")
    }
    try:
        server.load_model("z-image-turbo")
        assert built.get("model_name") == "filipstrand/Z-Image-Turbo-mflux-4bit", (
            "the image alias never reached ImageEngine — the MLLM routing "
            "preflight ran and killed a lane that has no MLLM question"
        )
    finally:
        for attr, value in saved.items():
            setattr(server, attr, value)
