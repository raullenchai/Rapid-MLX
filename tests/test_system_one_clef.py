# SPDX-License-Identifier: Apache-2.0
"""Clef protocol and backend routing without downloading 9B+ checkpoints."""

from __future__ import annotations

import base64
import io
import math
import threading
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from rapid_mlx.clef.media import decode_images, decode_videos
from rapid_mlx.cli import _resolve_system_one_backend, build_parser
from rapid_mlx.system_one.backends import ClefBackend
from rapid_mlx.system_one.schema import Question
from rapid_mlx.system_one.server import create_app


def _png_data_url() -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), "red").save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def test_clef_cli_selects_existing_system_one_service():
    for model in ("clef", "clef-flash", "Cloudflare/clef", "Cloudflare/clef-flash"):
        assert _resolve_system_one_backend(model, "auto") == "clef"
        assert build_parser().parse_args(["system-one", model]).model == model
    assert _resolve_system_one_backend("org/clefish", "auto") == "laya"


def test_clef_uses_official_joint_head_result(monkeypatch):
    pytest.importorskip("torch")
    from rapid_mlx.clef.vendor import joint_schema_model

    backend = object.__new__(ClefBackend)
    backend.default_model = "clef-flash"
    backend.repo_id = "Cloudflare/clef-flash"
    backend._model = object()
    backend._processor = object()
    backend._lock = threading.Lock()
    observed = {}

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


def test_clef_backend_loads_pinned_checkpoint_without_remote_code(monkeypatch):
    pytest.importorskip("torch")
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
    backend = ClefBackend("clef-flash", device="cpu")
    assert backend.default_model == "clef-flash"
    assert captured == {
        "repo": "Cloudflare/clef-flash",
        "revision": "17f0b0ad64efb65d273590632833508766b2aae6",
        "path": "/a/local/snapshot",
        "device": "cpu",
    }


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
