# SPDX-License-Identifier: Apache-2.0
"""Clef protocol and backend routing without downloading 9B+ checkpoints."""

from __future__ import annotations

import base64
import hashlib
import io
import math
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from rapid_mlx.clef.media import decode_images, decode_media, decode_videos
from rapid_mlx.cli import _resolve_system_one_backend, build_parser
from rapid_mlx.system_one.backends import ClefBackend
from rapid_mlx.system_one.schema import Question, SystemOneRequest
from rapid_mlx.system_one.server import create_app


def test_clef_vendor_source_matches_pinned_release():
    source = Path(__file__).parents[1] / "rapid_mlx/clef/vendor/joint_schema_model.py"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == (
        "0e304cf7c6500e8bb59bef7e2afd2c6373f82596dfb3b57d1aa93c175e2dc3a3"
    )


def _png_data_url() -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), "red").save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def test_clef_cli_selects_existing_system_one_service():
    for model in ("clef", "clef-flash", "Cloudflare/clef", "Cloudflare/clef-flash"):
        assert _resolve_system_one_backend(model, "auto") == "clef"
        assert build_parser().parse_args(["system-one", model]).model == model
    assert _resolve_system_one_backend("org/clefish", "auto") == "laya"


def test_clef_schema_rejects_too_many_video_frames():
    with pytest.raises(ValueError, match="at most 32 frames total"):
        SystemOneRequest(
            state="state",
            questions={"visible": Question(type="noul")},
            videos=[["frame"] * 33],
        )


def test_system_one_keeps_tokenizer_import_lazy(monkeypatch):
    import rapid_mlx.utils as utils

    sentinel = object()
    monkeypatch.setitem(
        sys.modules,
        "rapid_mlx.utils.tokenizer",
        SimpleNamespace(load_model_with_fallback=sentinel),
    )
    assert utils.load_model_with_fallback is sentinel
    missing_name = "not_a_utility"
    with pytest.raises(AttributeError):
        getattr(utils, missing_name)


def test_clef_cli_starts_selected_backend(monkeypatch):
    import rapid_mlx._uvicorn as uvicorn_module
    import rapid_mlx.cli as cli_module
    import rapid_mlx.system_one.backends as backend_module

    observed = {}

    def fake_backend(model, *, device):
        observed.update(model=model, device=device)
        return SimpleNamespace(default_model="clef-flash")

    monkeypatch.setattr(backend_module, "ClefBackend", fake_backend)
    monkeypatch.setattr(cli_module, "_port_preflight_or_die", lambda *a, **k: None)
    monkeypatch.setattr(
        uvicorn_module,
        "run_uvicorn",
        lambda app, **kwargs: observed.update(app=app, kwargs=kwargs),
    )
    args = build_parser().parse_args(["system-one", "clef-flash", "--port", "8701"])
    cli_module.system_one_command(args)
    assert observed["model"] == "clef-flash"
    assert observed["device"] == "gpu"
    assert observed["kwargs"]["port"] == 8701


def test_clef_uses_official_joint_head_result(monkeypatch):
    pytest.importorskip("torch")
    from rapid_mlx.clef import media as clef_media
    from rapid_mlx.clef.vendor import joint_schema_model

    backend = object.__new__(ClefBackend)
    backend.default_model = "clef-flash"
    backend.repo_id = "Cloudflare/clef-flash"
    backend._model = object()
    backend._processor = object()
    backend._lock = threading.Lock()
    observed = {}

    original_decode_media = clef_media.decode_media

    def guarded_decode_media(images, videos):
        assert backend._lock.locked(), "media decoded before model lock"
        return original_decode_media(images, videos)

    monkeypatch.setattr(clef_media, "decode_media", guarded_decode_media)

    def fake_systemone(model, processor, request):
        observed.update(model=model, processor=processor, request=request)
        return {
            "model": request["model"],
            "answers": {"approve": {"type": "noul", "noul": 0.75}},
            "usage": {"input_tokens": 42, "output_tokens": 0},
        }

    monkeypatch.setattr(joint_schema_model, "systemone", fake_systemone)
    client = TestClient(create_app(backend))
    response = client.post(
        "/v1/systemone",
        json={
            "state": {"invoice": "paid"},
            "questions": {"approve": {"type": "noul", "instructions": "Approve?"}},
        },
    )
    assert response.status_code == 200
    assert response.json()["answers"]["approve"]["noul"] == 0.75
    assert response.json()["usage"] == {
        "input_tokens": 42,
        "output_tokens": 0,
        "billing_units": 1,
    }
    assert observed["request"]["questions"]["approve"]["type"] == "noul"
    assert observed["model"] is backend._model

    media_response = client.post(
        "/v1/systemone",
        json={
            "state": "Review this image",
            "images": [_png_data_url()],
            "questions": {"approve": {"type": "noul"}},
        },
    )
    assert media_response.status_code == 200
    assert observed["request"]["images"][0].size == (2, 2)

    video_response = client.post(
        "/v1/systemone",
        json={
            "state": "Review these frames",
            "videos": [[_png_data_url(), _png_data_url()]],
            "questions": {"approve": {"type": "noul"}},
        },
    )
    assert video_response.status_code == 200
    assert observed["request"]["videos"][0][0].shape == (2, 2, 3)

    monkeypatch.setattr("rapid_mlx.clef.media._MAX_TOTAL_PIXELS", 8)
    mixed_response = client.post(
        "/v1/systemone",
        json={
            "state": "Review image and frames",
            "images": [_png_data_url()],
            "videos": [[_png_data_url(), _png_data_url()]],
            "questions": {"approve": {"type": "noul"}},
        },
    )
    assert mixed_response.status_code == 422
    assert "total pixel limit" in mixed_response.text


@pytest.mark.parametrize("device,expected_device", [("cpu", "cpu"), ("gpu", "mps")])
def test_clef_backend_loads_pinned_checkpoint_without_remote_code(
    monkeypatch, device, expected_device
):
    torch = pytest.importorskip("torch")
    import huggingface_hub

    from rapid_mlx.clef.vendor import joint_schema_model

    captured = {}

    def snapshot_download(repo, *, revision):
        captured.update(repo=repo, revision=revision)
        return "/a/local/snapshot"

    def load_release_model(path, *, device):
        captured.update(path=path, device=device)
        return object(), object()

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    monkeypatch.setattr(joint_schema_model, "load_release_model", load_release_model)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    backend = ClefBackend("clef-flash", device=device)
    assert backend.default_model == "clef-flash"
    assert captured == {
        "repo": "Cloudflare/clef-flash",
        "revision": "17f0b0ad64efb65d273590632833508766b2aae6",
        "path": "/a/local/snapshot",
        "device": expected_device,
    }


def test_clef_backend_rejects_bad_setup(monkeypatch):
    import importlib.metadata

    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError, match="unknown Clef model"):
        ClefBackend("Cloudflare/not-clef", device="cpu")
    with pytest.raises(ValueError, match="Clef device"):
        ClefBackend("clef-flash", device="other")

    original_version = importlib.metadata.version

    def missing_torch(package):
        if package == "torch":
            raise importlib.metadata.PackageNotFoundError(package)
        return original_version(package)

    monkeypatch.setattr(importlib.metadata, "version", missing_torch)
    with pytest.raises(RuntimeError, match="optional runtime"):
        ClefBackend("clef-flash", device="cpu")
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda package: "2.10" if package == "torch" else original_version(package),
    )
    with pytest.raises(RuntimeError, match="torch>=2.11"):
        ClefBackend("clef-flash", device="cpu")
    monkeypatch.setattr(importlib.metadata, "version", original_version)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="Apple Metal/MPS"):
        ClefBackend("clef-flash", device="gpu")


@pytest.mark.parametrize("version", ["5.13.0", "5.16.0"])
def test_clef_backend_rejects_unsupported_transformers(monkeypatch, version):
    import importlib.metadata

    original_version = importlib.metadata.version
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda package: (
            "2.14.1"
            if package == "torch"
            else version
            if package == "transformers"
            else original_version(package)
        ),
    )
    with pytest.raises(RuntimeError, match="transformers>=5.10.2,!=5.13.0,<5.16"):
        ClefBackend("clef-flash", device="cpu")


def test_clef_media_decodes_locally_and_rejects_urls():
    image = decode_images([_png_data_url()])[0]
    assert image.size == (2, 2)
    video = decode_videos([[_png_data_url(), _png_data_url()]])
    assert video[0][0].shape == (2, 2, 3)
    with pytest.raises(ValueError, match="at least two frames"):
        decode_videos([[_png_data_url()]])
    try:
        decode_images(["https://example.com/receipt.png"])
    except ValueError as exc:
        assert "data URL" in str(exc)
    else:
        raise AssertionError("remote media URL was accepted")


def test_clef_media_has_one_decoded_pixel_budget_for_images_and_video(monkeypatch):
    png = _png_data_url()
    monkeypatch.setattr("rapid_mlx.clef.media._MAX_TOTAL_PIXELS", 8)
    assert len(decode_images([png, png])) == 2
    assert len(decode_videos([[png, png]])) == 1
    with pytest.raises(ValueError, match="total pixel limit"):
        decode_images([png, png, png])
    with pytest.raises(ValueError, match="total pixel limit"):
        decode_videos([[png, png, png]])
    with pytest.raises(ValueError, match="total pixel limit"):
        decode_media([png], [[png, png]])


def test_clef_media_rejects_invalid_and_oversized_inputs(monkeypatch):
    from PIL import Image

    png = _png_data_url()
    cases = [
        ("data:image/png;base64", "data URL"),
        ("data:image/gif;base64,AAAA", "PNG, JPEG, or WebP"),
        (
            "data:image/png;base64," + "A" * (4 * 1024 * 1024 * 4 // 3 + 5),
            "encoded limit",
        ),
        ("data:image/png;base64,not-base64!", "invalid base64"),
        (
            "data:image/png;base64,"
            + base64.b64encode(b"x" * (4 * 1024 * 1024 + 1)).decode(),
            "encoded limit",
        ),
        (
            "data:image/png;base64," + base64.b64encode(b"not an image").decode(),
            "invalid or oversized",
        ),
    ]
    bitmap = io.BytesIO()
    Image.new("RGB", (2, 2)).save(bitmap, format="BMP")
    cases.append(
        (
            "data:image/png;base64," + base64.b64encode(bitmap.getvalue()).decode(),
            "format does not match",
        )
    )
    jpeg = io.BytesIO()
    Image.new("RGB", (2, 2)).save(jpeg, format="JPEG")
    cases.append(
        (
            "data:image/png;base64," + base64.b64encode(jpeg.getvalue()).decode(),
            "format does not match",
        )
    )
    for value, message in cases:
        with pytest.raises(ValueError, match=message):
            decode_images([value])
    with pytest.raises(ValueError, match="at most 8 images"):
        decode_images([png] * 9)
    with pytest.raises(ValueError, match="at most 2 videos"):
        decode_videos([[png, png]] * 3)
    monkeypatch.setattr("rapid_mlx.clef.media._MAX_PIXELS", 1)
    with pytest.raises(ValueError, match="16 MP pixel limit"):
        decode_images([png])
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1)
    with pytest.raises(ValueError, match="invalid or oversized"):
        decode_images([png])


def test_other_decision_backends_reject_media_before_inference():
    class TextOnlyBackend:
        default_model = "text-only"

        def answer(self, *_args):
            raise AssertionError("text-only backend should not receive media")

    client = TestClient(create_app(TextOnlyBackend()))
    response = client.post(
        "/v1/systemone",
        json={
            "state": "Review this",
            "images": [_png_data_url()],
            "questions": {"clear": {"type": "noul"}},
        },
    )
    assert response.status_code == 422
    assert "does not support media" in response.text


def test_clef_rejects_wrong_model_or_temperature_without_inference():
    backend = object.__new__(ClefBackend)
    backend.default_model = "clef-flash"
    backend.repo_id = "Cloudflare/clef-flash"
    questions = {"yes": Question(type="noul", instructions="Yes?")}
    for model, temperature, expected in (
        ("other", 1.0, KeyError),
        ("clef-flash", 0.5, ValueError),
    ):
        try:
            backend.answer("state", questions, model, temperature)
        except expected:
            pass
        else:
            raise AssertionError("invalid Clef request was accepted")


def test_clef_rank_and_model_catalog(monkeypatch):
    torch = pytest.importorskip("torch")
    from rapid_mlx.clef.vendor import joint_schema_model

    backend = object.__new__(ClefBackend)
    backend.default_model = "clef-flash"
    backend.repo_id = "Cloudflare/clef-flash"
    backend._processor = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0))
    backend._lock = threading.Lock()

    class CloseScoresModel:
        def parameters(self):
            yield torch.nn.Parameter(torch.zeros(1))

        def __call__(self, _batch):
            return [[torch.tensor([0.0, 0.00004])]]

    backend._model = CloseScoresModel()
    monkeypatch.setattr(
        joint_schema_model,
        "encode_record",
        lambda *_args, **_kwargs: SimpleNamespace(
            questions=[SimpleNamespace(option_ids=("0", "1"))]
        ),
    )
    monkeypatch.setattr(joint_schema_model, "collate_records", lambda *_args: {})
    ranked = backend.rank("context", None, ["wrong", "right"], "clef-flash", 1.0)
    assert [item["candidate"] for item in ranked] == ["right", "wrong"]
    assert ranked[0]["prob"] > ranked[1]["prob"]
    assert [round(item["prob"], 4) for item in ranked] == [0.5, 0.5]
    with pytest.raises(KeyError, match="unknown model"):
        backend.rank("context", None, ["wrong", "right"], "other", 1.0)
    with pytest.raises(ValueError, match="temperature=1"):
        backend.rank("context", None, ["wrong", "right"], "clef-flash", 0.5)
    assert backend.models()[0]["hf_id"] == "Cloudflare/clef-flash"


def test_official_joint_head_runs_one_cpu_forward_pass():
    torch = pytest.importorskip("torch")
    from rapid_mlx.clef.vendor.joint_schema_model import (
        ClefModel,
        JointSchemaHead,
        collate_records,
        encode_record,
    )

    class TinyTokenizer:
        pad_token_id = 0

        def __call__(self, value, *, add_special_tokens):
            assert add_special_tokens is False
            return SimpleNamespace(input_ids=[ord(char) % 127 + 1 for char in value])

    class TinyTextModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(128, 8)

        def forward(self, input_ids, **_kwargs):
            return SimpleNamespace(last_hidden_state=self.embedding(input_ids))

    class TinyBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = TinyTextModel()
            self.output = torch.nn.Embedding(128, 8)

        def get_output_embeddings(self):
            return self.output

    tokenizer = TinyTokenizer()
    encoded = encode_record(
        tokenizer,
        {
            "state": "The invoice is paid.",
            "questions": {
                "status": {
                    "type": "choice",
                    "instructions": "What status?",
                    "criteria": {"paid": "Paid", "late": "Overdue"},
                }
            },
        },
    )
    model = ClefModel(
        TinyBackbone(),
        JointSchemaHead(
            hidden_size=8,
            width=8,
            routing_layers=1,
            layers=1,
            heads=2,
            feedforward=16,
        ),
    ).eval()
    with torch.inference_mode():
        logits = model(
            collate_records([encoded], tokenizer.pad_token_id, torch.device("cpu"))
        )
    assert len(logits) == 1 and len(logits[0]) == 1
    assert logits[0][0].shape == (2,)
    assert all(math.isfinite(value) for value in logits[0][0].tolist())
