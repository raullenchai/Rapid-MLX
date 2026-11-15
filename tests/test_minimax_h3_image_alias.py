# SPDX-License-Identifier: Apache-2.0
"""Offline contracts for MiniMax-H3-Image serving and its pinned data assets."""

from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from rapid_mlx import _download_gate as gate
from rapid_mlx.image.engine import (
    ImageGenerationCancelled,
    ImageGenerationEngine,
    ImageRequestError,
    ImageRuntimeError,
    _detect_family,
    image_license_info,
)
from rapid_mlx.model_aliases import resolve_model, resolve_profile
from rapid_mlx.runtime.image_lane import ImageEngine

REPO = "MiniMaxAI/MiniMax-H3-Image"
REVISION = "5f90473023975e6271b0ed1dd84c4e88e4bbc6d3"


@pytest.mark.parametrize(
    "alias,bits,memory,charge",
    [
        ("minimax-h3-image", 4, 64, 44),
        ("minimax-h3-image-8bit", 8, 96, 71),
    ],
)
def test_alias_quantization_and_admission(alias, bits, memory, charge):
    from rapid_mlx.catalog.legacy import build_legacy_catalog_snapshot
    from rapid_mlx.runtime.resident_models import measured_image_estimate_bytes

    profile = resolve_profile(alias)
    assert profile.hf_path == resolve_model(alias) == REPO
    assert profile.revision == REVISION
    assert profile.modality == "image-gen"
    assert profile.image_quantize == bits
    assert profile.min_memory_gb == memory
    lane = ImageEngine(profile.hf_path, image_quantize=profile.image_quantize)
    engine = lane._engine
    assert engine._quantize == bits
    assert engine.supports_generation and engine.supports_editing
    assert engine.max_reference_images == 16
    assert engine.default_steps == engine.default_edit_steps == 6
    assert engine.default_edit_guidance == 1.0
    assert not engine.supports_negative_prompt
    assert measured_image_estimate_bytes(alias) == charge * 1024**3
    catalog = build_legacy_catalog_snapshot()
    row = next(row for row in catalog["aliases"] if row["alias"] == alias)
    assert row["capabilities"]["runtime_adapter"] == "rapid_mlx/minimax_h3_image"
    assert row["capabilities"]["operation_modes"] == ["text_to_image", "image_to_image"]
    assert row["availability"]["desktop"]


@pytest.mark.parametrize(
    "name",
    ["minimax-h3-image", "minimax-h3-image-8bit", REPO, "/models/minimax_h3_image"],
)
def test_family_and_default_quantization(name):
    from rapid_mlx.video.minimax_h3 import is_minimax_h3_model

    assert _detect_family(name) == "minimax-h3-image"
    assert not is_minimax_h3_model(name)
    assert is_minimax_h3_model("minimax-h3")
    assert is_minimax_h3_model("MiniMaxAI/MiniMax-H3")
    assert ImageGenerationEngine(name)._quantize == 4


def test_exact_primary_allowlist_and_shared_auxiliary_pin():
    from rapid_mlx.video.minimax_h3 import FILES
    from rapid_mlx.video.minimax_h3 import REPO as TEXT_REPO
    from rapid_mlx.video.minimax_h3 import REVISION as TEXT_REVISION

    expected = {
        "LICENSE",
        "model_index.json",
        "pdd_config.json",
        "scheduler/scheduler_config.json",
        *(
            f"tokenizer/{name}"
            for name in (
                "merges.txt",
                "tokenizer.json",
                "tokenizer_config.json",
                "vocab.json",
            )
        ),
        *(
            f"processor/{name}"
            for name in (
                "chat_template.json",
                "merges.txt",
                "preprocessor_config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "video_preprocessor_config.json",
                "vocab.json",
            )
        ),
        "transformer_diffusers/config.json",
        "transformer_diffusers/diffusion_pytorch_model.safetensors.index.json",
        *(
            f"transformer_diffusers/diffusion_pytorch_model-{i:05d}-of-00012.safetensors"
            for i in range(1, 13)
        ),
        "vae_diffusers/config.json",
        "vae_diffusers/diffusion_pytorch_model.safetensors",
    }
    assert set(gate.IMAGE_MODEL_DATA_FILES[REPO]) == expected
    assert gate.IMAGE_MODEL_REVISIONS[REPO] == REVISION
    assert not any(
        name.endswith(".py")
        or name.startswith(
            (
                "assets/",
                "examples/",
                "transformer/",
                "vae/",
                "text_encoder/",
                "pdd_fused_heads",
            )
        )
        for name in expected
    )
    ((repo, revision, files),) = gate.image_runtime_assets_for(REPO)
    assert (repo, revision) == (TEXT_REPO, TEXT_REVISION)
    assert revision == "b3c7290e66afdf293bef3b9077b7a266ef421f34"
    assert set(files) == {
        "text_encoder/config.json",
        "text_encoder/model.safetensors.index.json",
        *(f"text_encoder/model-{i:05d}-of-00014.safetensors" for i in range(1, 15)),
    }
    assert set(files) <= set(FILES)
    from scripts.build_model_digests import selections

    selected = selections()
    assert selected["minimax-h3-image"]["allow_patterns"] == list(
        gate.IMAGE_MODEL_DATA_FILES[REPO]
    )
    assert selected["asset/image/" + repo]["revision"] == revision


def test_pinned_snapshots_require_all_data_and_reuse_video_cache(tmp_path, monkeypatch):
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path))
    payloads = [
        (REPO, REVISION, gate.MINIMAX_H3_IMAGE_DATA_FILES),
        *gate.image_runtime_assets_for(REPO),
    ]
    for repo, revision, files in payloads:
        root = (
            tmp_path / ("models--" + repo.replace("/", "--")) / "snapshots" / revision
        )
        for name in files:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"data")
        assert gate.pinned_image_snapshot(repo) == str(root)
        path.unlink()
        assert gate.pinned_image_snapshot(repo) is None
        path.write_bytes(b"data")
        assert gate.pinned_image_snapshot(repo) == str(root)


@pytest.mark.parametrize("bits", [4, 8])
def test_build_dispatch_progress_cancel_and_ordered_references(
    monkeypatch, tmp_path, bits
):
    built = []
    calls = []

    class Runtime:
        def __init__(self, *paths, **kwargs):
            built.append((paths, kwargs))
            self.loaded = False

        def load(self):
            self.loaded = True

        def generate_image(self, prompt, **kwargs):
            assert self.loaded
            calls.append((prompt, kwargs))
            kwargs["cancel_check"]()
            built[0][1]["on_step"](0, 6)
            built[0][1]["on_step"](6, 6)
            return Image.new("RGB", (32, 32))

    import rapid_mlx.image.minimax_h3_image as runtime_package

    # Patch the attribute, not the package: engine validation imports the
    # real ``layout`` submodule.
    monkeypatch.setattr(runtime_package, "MiniMaxH3Image", Runtime, raising=False)
    monkeypatch.setattr(gate, "pinned_image_snapshot", lambda repo: "/text-snapshot")
    engine = ImageGenerationEngine(REPO, quantize=bits)
    monkeypatch.setattr(engine, "_model_path_for_mflux", lambda: "/image-snapshot")
    model = engine._build_model()
    assert built == [
        (
            ("/image-snapshot", "/text-snapshot"),
            {"quantize": bits, "on_step": engine._report_native_step},
        )
    ]
    monkeypatch.setattr(engine, "_ensure_loaded", lambda **kwargs: model)
    paths = []
    for i in range(2):
        path = tmp_path / f"{i}.png"
        Image.new("RGB", (32, 32), (i * 100, 0, 0)).save(path)
        paths.append(str(path))
    assert engine.generate(
        prompt="<Picture 1> next to <Picture 2>",
        num_inference_steps=None,
        image_paths=paths,
    ).startswith(b"\x89PNG")
    prompt, kwargs = calls[0]
    assert prompt == "<Picture 1> next to <Picture 2>"
    assert [im.getpixel((0, 0)) for im in kwargs["references"]] == [
        (0, 0, 0),
        (100, 0, 0),
    ]
    assert kwargs["num_inference_steps"] == 6
    assert kwargs["guidance_scale"] == 1.0
    assert kwargs["min_generation_area"] == 2048**2
    assert engine.progress_snapshot()["step"] == 6
    assert engine.performance_snapshot()["denoise_steps"] == 6
    # Without an explicit size, edits keep the first reference's aspect at ~1 MP.
    assert (kwargs["width"], kwargs["height"]) == (1024, 1024)
    wide = tmp_path / "wide.png"
    Image.new("RGB", (64, 32)).save(wide)
    engine.generate(
        prompt="<Picture 1>",
        width=None,
        height=None,
        num_inference_steps=None,
        image_paths=[str(wide)],
    )
    assert (calls[-1][1]["width"], calls[-1][1]["height"]) == (1440, 736)
    engine.generate(
        prompt="<Picture 1>",
        width=512,
        height=768,
        num_inference_steps=None,
        image_paths=[str(wide)],
    )
    assert (calls[-1][1]["width"], calls[-1][1]["height"]) == (512, 768)
    monkeypatch.setattr(engine, "_is_cancelled", lambda: True)
    with pytest.raises(ImageGenerationCancelled):
        kwargs["cancel_check"]()
    with pytest.raises(ImageGenerationCancelled):
        engine._report_native_step(2, 6)
    monkeypatch.setattr(gate, "pinned_image_snapshot", lambda repo: None)
    with pytest.raises(ImageRuntimeError, match="pinned model and text-encoder"):
        engine._build_model()


def test_same_model_resident_across_generation_and_edit(monkeypatch):
    engine = ImageGenerationEngine(REPO)
    engine._model = object()
    engine._loaded_mode = "generation"
    model = engine._model
    assert engine._ensure_loaded(for_edit=True) is model
    assert engine._ensure_loaded(for_edit=False) is model


INVALID = [
    {"num_inference_steps": 5},
    {"guidance": 2.0},
    {"guidance": float("nan")},
    {"negative_prompt": "blur"},
    {"negative_prompt": ""},
    {"width": 0},
    {"height": -32},
    {"width": 1000},
    {"image_paths": ["x"] * 17},
    {"loras": ["unsupported"]},
]


@pytest.mark.parametrize("overrides", INVALID)
def test_validation_before_load(overrides, monkeypatch):
    engine = ImageGenerationEngine(REPO)
    monkeypatch.setattr(
        engine,
        "_ensure_loaded",
        lambda **kw: pytest.fail("invalid request loaded weights"),
    )
    with pytest.raises(ImageRequestError):
        engine.generate(**({"prompt": "fox", "num_inference_steps": 6} | overrides))


@pytest.fixture
def client(monkeypatch):
    from rapid_mlx.server import app

    engine = ImageGenerationEngine(REPO)
    engine.is_image_gen = True
    calls = []

    class Runtime:
        def generate_image(self, prompt, **kwargs):
            calls.append(kwargs)
            return Image.new("RGB", (32, 32))

    monkeypatch.setattr(engine, "_ensure_loaded", lambda **kw: Runtime())
    monkeypatch.setattr(
        "rapid_mlx.config.get_config", lambda: SimpleNamespace(engine=engine)
    )
    return TestClient(app), calls


def png():
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32)).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.mark.parametrize(
    "payload",
    [
        {"steps": 5},
        {"guidance": 2.0},
        {"negative_prompt": "blur"},
        {"size": "1000x1024"},
        {"loras": [{"name": "anything", "scale": 1.0}]},
    ],
)
def test_generation_invalid_request_is_400(client, payload):
    http, calls = client
    response = http.post("/v1/images/generations", json={"prompt": "fox"} | payload)
    assert response.status_code == 400, response.text
    assert not calls


@pytest.mark.parametrize("editing", [False, True])
def test_response_attribution_and_six_step_defaults(client, editing):
    http, calls = client
    if editing:
        response = http.post(
            "/v1/images/edits",
            data={"prompt": "fox"},
            files=[("image", ("ref.png", png(), "image/png"))] * 16,
        )
    else:
        response = http.post("/v1/images/generations", json={"prompt": "fox"})
    assert response.status_code == 200, response.text
    assert response.json()["attribution"] == "MiniMax-H3-Image"
    assert calls[0]["num_inference_steps"] == 6
    assert len(calls[0]["references"]) == (16 if editing else 0)


def test_too_many_references_is_400(client):
    http, calls = client
    response = http.post(
        "/v1/images/edits",
        data={"prompt": "fox"},
        files=[("image", ("ref.png", png(), "image/png"))] * 17,
    )
    assert response.status_code == 400
    assert not calls


def test_license_metadata_and_cli_listing():
    from rapid_mlx.cli import _available_models_json_payload
    from rapid_mlx.routes.models import _build_model_info

    info = image_license_info(REPO)
    assert info["license"] == "MiniMax-H3-Image Community License"
    assert info["attribution"] == "MiniMax-H3-Image"
    assert (
        info["license_url"]
        == "https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/main/LICENSE"
    )
    assert all(
        s in info["license_notice"]
        for s in ("US$1M", "hosted service", "prominently display")
    )
    payload = _available_models_json_payload()
    rows = {row["alias"]: row for row in payload["image"]}
    for alias in ("minimax-h3-image", "minimax-h3-image-8bit"):
        assert rows[alias]["attribution"] == info["attribution"]
        assert rows[alias]["default_steps"] == 6
        assert rows[alias]["max_reference_images"] == 16
        metadata = _build_model_info(alias)
        assert metadata.attribution == info["attribution"]
        assert metadata.license_notice == info["license_notice"]
        assert metadata.max_reference_images == 16
    sizes = json.loads(
        (Path(__file__).parents[1] / "rapid_mlx/model_sizes.json").read_text()
    )
    assert sizes["sizes"][REPO] > 0


def test_runtime_issue_checks_mlx_vlm(monkeypatch):
    from rapid_mlx.runtime.image_lane import image_runtime_issue

    checked = []
    monkeypatch.setattr("importlib.util.find_spec", lambda name: checked.append(name))
    assert "mlx_vlm" in image_runtime_issue(REPO)
    assert checked == ["mlx_vlm"]


@pytest.mark.parametrize(
    "payload",
    [
        {"steps": "5"},
        {"guidance": "2.0"},
        {"negative_prompt": "blur"},
        {"size": "1008x1024"},
        {"loras": '[{"name":"anything","scale":1.0}]'},
    ],
)
def test_edit_invalid_request_is_400(client, payload):
    http, calls = client
    response = http.post(
        "/v1/images/edits",
        data={"prompt": "fox"} | payload,
        files=[("image", ("ref.png", png(), "image/png"))],
    )
    assert response.status_code == 400, response.text
    assert not calls


@pytest.mark.parametrize(
    "alias,bits", [("minimax-h3-image", 4), ("minimax-h3-image-8bit", 8)]
)
async def test_dynamic_server_carries_alias_quantization(monkeypatch, alias, bits):
    from rapid_mlx import server

    built = []

    class Lane:
        def __init__(self, **kwargs):
            built.append(kwargs)

        def validate_residency_mode(self, mode):
            assert mode == "editing"

    class StopBeforeDownloadError(Exception):
        pass

    monkeypatch.setattr("rapid_mlx.runtime.image_lane.ImageEngine", Lane)

    def require_local(*args):
        raise StopBeforeDownloadError

    monkeypatch.setattr("rapid_mlx.model_downloads.require_local", require_local)
    with pytest.raises(StopBeforeDownloadError):
        await server._load_dynamic_resident_model(alias, None, image_mode="editing")
    assert built == [{"model_name": REPO, "image_quantize": bits}]


def test_quantization_profile_rejects_invalid_values():
    from rapid_mlx.model_aliases import _coerce

    for value in (True, "4", 6, 16):
        with pytest.raises(ValueError, match="image_quantize"):
            _coerce(
                "invalid",
                {
                    "hf_path": REPO,
                    "modality": "image-gen",
                    "supports_spec_decode": False,
                    "image_quantize": value,
                },
            )


@pytest.mark.parametrize(
    "alias,bits", [("minimax-h3-image", 4), ("minimax-h3-image-8bit", 8), (REPO, 4)]
)
def test_primary_server_carries_quantization(
    monkeypatch, isolated_server_model_state, alias, bits
):
    from rapid_mlx import server

    class ConstructedError(Exception):
        pass

    built = []

    class Lane:
        def __init__(self, **kwargs):
            built.append(kwargs)
            raise ConstructedError

    monkeypatch.setattr("rapid_mlx.runtime.image_lane.ImageEngine", Lane)
    monkeypatch.setattr(
        "rapid_mlx.runtime.image_lane.require_image_runtime_or_exit", lambda *a: None
    )
    monkeypatch.setattr(
        "rapid_mlx.utils.generation_config.load_generation_config_sampling",
        lambda *a, **kw: {},
    )
    monkeypatch.setattr(server, "_model_alias", None)
    with pytest.raises(ConstructedError):
        server.load_model(alias)
    assert built == [{"model_name": REPO, "image_quantize": bits}]


def test_cold_primary_uses_exact_allowlist(monkeypatch):
    engine = ImageGenerationEngine(REPO)
    calls = []
    monkeypatch.setattr(gate, "mflux_local_snapshot", lambda repo: None)
    monkeypatch.setattr(engine, "_verify_weights_complete", lambda: None)
    monkeypatch.setattr(
        "rapid_mlx._mirror.pinned_snapshot_download",
        lambda *args, **kwargs: calls.append((args, kwargs)) or "/pinned/image",
    )
    assert engine._resolve_model_path_for_mflux() == "/pinned/image"
    assert calls == [
        ((REPO, REVISION), {"allow_patterns": list(gate.MINIMAX_H3_IMAGE_DATA_FILES)})
    ]


def test_auxiliary_warm_cache_reused_and_cold_pull_pinned(monkeypatch):
    engine = ImageGenerationEngine(REPO)
    calls = []
    monkeypatch.setattr(gate, "pinned_image_snapshot", lambda repo: "/video-cache")
    monkeypatch.setattr(
        "rapid_mlx.model_downloads.check", lambda repo: calls.append(repo)
    )
    monkeypatch.setattr(
        "rapid_mlx._mirror.pinned_snapshot_download",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    engine._ensure_runtime_assets()
    assert not calls
    monkeypatch.setattr(gate, "pinned_image_snapshot", lambda repo: None)
    engine._ensure_runtime_assets()
    ((repo, revision, files),) = gate.image_runtime_assets_for(REPO)
    assert calls == [repo, ((repo, revision), {"allow_patterns": list(files)})]


async def test_unload_releases_native_model():
    lane = ImageEngine(REPO)
    lane._engine._model = object()
    lane._engine._loaded_mode = "generation"
    await lane.stop()
    assert not lane.is_resident
    assert lane._engine._loaded_mode is None


def _png(width, height):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_edit_omitted_size_follows_reference_aspect(client):
    http, calls = client
    response = http.post(
        "/v1/images/edits",
        data={"prompt": "fox"},
        files=[("image", ("ref.png", _png(64, 32), "image/png"))],
    )
    assert response.status_code == 200, response.text
    assert (calls[0]["width"], calls[0]["height"]) == (1440, 736)
    response = http.post(
        "/v1/images/edits",
        data={"prompt": "fox", "size": "512x768"},
        files=[("image", ("ref.png", _png(64, 32), "image/png"))],
    )
    assert response.status_code == 200, response.text
    assert (calls[1]["width"], calls[1]["height"]) == (512, 768)


def test_extreme_reference_aspect_is_400_before_load(client):
    http, calls = client
    response = http.post(
        "/v1/images/edits",
        data={"prompt": "fox"},
        files=[("image", ("ref.png", _png(32, 8192), "image/png"))],
    )
    assert response.status_code == 400, response.text
    assert "4:1" in response.text
    assert not calls
