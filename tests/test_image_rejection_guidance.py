# SPDX-License-Identifier: Apache-2.0
"""Image rejections on a text-only lane say why and what to do next.

A multimodal checkpoint routed to the text lane used to answer every image with
"Model 'X' is serving text-only; image input is unsupported." and no reason,
so clients retried. The rejection now appends one or two sentences built from
the engine's ``serving_lane_reason``: the cause, and either the flag to drop,
the install command, or a catalog vision alias that fits this Mac. The HTTP
status (400) and the machine-readable code (``image_input_unsupported``) stay
exactly as they were because clients key on them.
"""

from __future__ import annotations

import os

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import rapid_mlx.api.utils as api_utils
import rapid_mlx.server as server
from rapid_mlx.api.utils import (
    BANNER_TEXT_LANE_REASONS,
    UnsupportedContentBlockError,
    fitting_vision_alias,
    public_model_label,
    text_lane_image_guidance,
    vision_alias_memory_need_gb,
)
from rapid_mlx.config import reset_config
from rapid_mlx.connect import endpoints_from_bind, render_banner
from rapid_mlx.model_aliases import list_builtin_aliases, resolve_profile
from rapid_mlx.model_profile import ModelProfile
from rapid_mlx.models import mllm
from rapid_mlx.routes.anthropic import router as anthropic_router
from rapid_mlx.routes.chat import router as chat_router
from rapid_mlx.routes.responses import router as responses_router

_BASE = "Model 'qwen3.5-4b-4bit' is serving text-only; image input is unsupported."
_IMAGE_URL = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=="


@pytest.fixture(autouse=True)
def _pin_host(monkeypatch):
    """A 16 GB Mac whose vision runtime can serve hybrid backbones."""
    monkeypatch.setattr(api_utils, "physical_ram_gb", lambda: 16.0)
    monkeypatch.setattr(api_utils, "mllm_hybrid_runtime_supported", lambda: True)
    yield
    reset_config()


class _TextLaneEngine:
    preserve_native_tool_format = False
    is_mllm = False
    supports_guided_generation = False
    tokenizer = None

    def __init__(self, reason):
        # No ``chat``/``stream_chat``: an image request must be rejected at
        # the route boundary, before any generation call could be made.
        self.serving_lane_reason = reason

    def build_prompt(self, messages, tools=None, enable_thinking=None):
        return "PROMPT"


def _client(reason, *, model_name="qwen3.5-4b-4bit"):
    cfg = reset_config()
    engine = _TextLaneEngine(reason)
    cfg.engine = engine
    cfg.model_name = model_name
    cfg.model_registry = None
    cfg.no_thinking = True
    cfg.reasoning_parser = None
    cfg.reasoning_parser_name = None
    cfg.tool_parser = None
    cfg.tool_call_parser = None
    app = FastAPI()
    app.include_router(chat_router)
    app.include_router(anthropic_router)
    app.include_router(responses_router)
    return TestClient(app), engine


def _chat_image_error(reason, **kwargs):
    client, _ = _client(reason, **kwargs)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": kwargs.get("model_name", "qwen3.5-4b-4bit"),
            "max_tokens": 8,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is this"},
                        {"type": "image_url", "image_url": {"url": _IMAGE_URL}},
                    ],
                }
            ],
        },
    )
    assert response.status_code == 400, response.text
    error = response.json()["detail"]["error"]
    assert error["code"] == "image_input_unsupported"
    assert error["type"] == "invalid_request_error"
    assert error["param"] == "messages.content"
    assert error["serving_lane_reason"] == reason
    return error["message"]


# ---------------------------------------------------------------------------
# One test per reason branch, through the real /v1/chat/completions route.
# ---------------------------------------------------------------------------


def test_memory_insufficient_names_floor_ram_and_a_fitting_alias():
    message = _chat_image_error("vision_memory_insufficient")
    alias = fitting_vision_alias(16.0, hybrid_runtime_ok=True)
    assert alias is not None
    assert message == (
        f"{_BASE} Vision for this model needs at least 32 GB of RAM; this Mac "
        f"has 16 GB, so it started text-only. For image input, serve "
        f"'{alias}', a vision model that fits this Mac."
    )


def test_memory_insufficient_without_catalog_floor():
    guidance = text_lane_image_guidance(
        "org/uncatalogued-vlm", "vision_memory_insufficient", ram_gb=8
    )
    assert guidance is not None
    assert guidance.startswith(
        "Vision for this model needs more than this Mac's 8 GB of RAM, so it "
        "started text-only."
    )


def test_memory_insufficient_says_so_when_nothing_fits():
    guidance = text_lane_image_guidance(
        "qwen3.5-4b-4bit", "vision_memory_insufficient", ram_gb=0.5
    )
    assert guidance is not None
    assert guidance.endswith("No catalog vision model fits this Mac's memory.")


def test_hybrid_runtime_unsupported_reuses_the_vision_install_hint():
    message = _chat_image_error("vision_hybrid_runtime_unsupported")
    hint = " ".join(mllm._vision_install_hint(include_paths=False).split())
    assert message == (
        f"{_BASE} The installed vision runtime (mlx-vlm) is missing or too old "
        f"for this model's hybrid backbone, so it started text-only. {hint}"
    )
    assert "'rapid-mlx[vision]'" in message


def test_forced_text_lane_names_the_flag():
    message = _chat_image_error("text_lane_forced")
    assert message == (
        f"{_BASE} Text-only serving was forced with --no-mllm (--text-only); "
        "restart without that flag for image input."
    )


def test_catalog_text_only_pin_suggests_a_vision_alias():
    guidance = text_lane_image_guidance("qwen3.6-35b", "text_lane_forced", ram_gb=64)
    assert resolve_profile("qwen3.6-35b").is_text_only
    alias = fitting_vision_alias(
        64,
        hybrid_runtime_ok=True,
        exclude_hf_path=resolve_profile("qwen3.6-35b").hf_path,
    )
    assert guidance == (
        "Its catalog entry pins it to text-only serving. For image input, "
        f"serve '{alias}', a vision model that fits this Mac."
    )


def test_speculative_decode_names_the_flags():
    message = _chat_image_error("text_lane_speculative_decode")
    assert message == (
        f"{_BASE} Speculative decoding was requested (--spec-decode, "
        "--force-spec-decode or MTP) and only the text lane runs it; restart "
        "without speculative decoding for image input."
    )


@pytest.mark.parametrize(
    ("reason", "cause"),
    [
        ("text_checkpoint", "This checkpoint has no vision tower."),
        (
            "vision_weights_unavailable",
            "This checkpoint's vision weights are missing, so it started text-only.",
        ),
        (
            "vision_architecture_unavailable",
            "The installed vision runtime does not provide this model's vision "
            "architecture, so it started text-only.",
        ),
        (
            "vision_hybrid_cache_unsupported",
            "The vision lane does not support this model's cache layout, so it "
            "started text-only.",
        ),
    ],
)
def test_text_only_checkpoint_reasons_suggest_a_vision_alias(reason, cause):
    message = _chat_image_error(reason, model_name="org/some-text-model")
    alias = fitting_vision_alias(16.0, hybrid_runtime_ok=True)
    assert message == (
        "Model 'org/some-text-model' is serving text-only; image input is "
        f"unsupported. {cause} For image input, serve '{alias}', a vision "
        "model that fits this Mac."
    )


@pytest.mark.parametrize("reason", [None, "not_applicable", 7])
def test_unknown_or_missing_reason_keeps_the_original_message(reason):
    assert text_lane_image_guidance("qwen3.5-4b-4bit", reason) is None
    error = UnsupportedContentBlockError(
        "base.", code="image_input_unsupported", param="p", model_name="m"
    )
    assert error.client_message(reason) == "base."


def test_other_codes_and_modelless_errors_are_untouched():
    other = UnsupportedContentBlockError(
        "video.", code="video_input_unsupported", param="p", model_name="m"
    )
    assert other.client_message("text_lane_forced") == "video."
    modelless = UnsupportedContentBlockError(
        "img.", code="image_input_unsupported", param="p"
    )
    assert modelless.openai_detail(serving_lane_reason="text_lane_forced") == {
        "error": {
            "message": "img.",
            "type": "invalid_request_error",
            "code": "image_input_unsupported",
            "param": "p",
            "serving_lane_reason": "text_lane_forced",
        }
    }


def test_responses_route_carries_the_same_guidance():
    client, _ = _client("text_lane_forced")
    response = client.post(
        "/v1/responses",
        json={
            "model": "qwen3.5-4b-4bit",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "what is this"},
                        {"type": "input_image", "image_url": _IMAGE_URL},
                    ],
                }
            ],
        },
    )
    assert response.status_code == 400, response.text
    body = response.json()
    error = body.get("detail", body)["error"]
    assert error["code"] == "image_input_unsupported"
    assert error["message"].endswith("restart without that flag for image input.")


def _anthropic_image(client):
    return client.post(
        "/v1/messages",
        json={
            "model": "qwen3.5-4b-4bit",
            "max_tokens": 8,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "iVBORw0KGgoAAAANSUhEUg==",
                            },
                        },
                        {"type": "text", "text": "what is this"},
                    ],
                }
            ],
        },
    )


def test_anthropic_route_appends_guidance_and_keeps_400():
    client, _ = _client("text_lane_forced")
    response = _anthropic_image(client)
    assert response.status_code == 400, response.text
    assert response.json()["detail"] == (
        "Model 'qwen3.5-4b-4bit' does not support image inputs. Text-only "
        "serving was forced with --no-mllm (--text-only); restart without that "
        "flag for image input."
    )


def test_anthropic_route_without_reason_keeps_original_detail():
    client, _ = _client(None)
    response = _anthropic_image(client)
    assert response.status_code == 400, response.text
    assert response.json()["detail"] == (
        "Model 'qwen3.5-4b-4bit' does not support image inputs."
    )


# ---------------------------------------------------------------------------
# No filesystem paths reach clients.
# ---------------------------------------------------------------------------


def test_local_checkpoint_paths_are_reduced_to_their_basename():
    message = _chat_image_error(
        "text_checkpoint", model_name="/Users/someone/models/my-vlm"
    )
    assert message.startswith("Model 'my-vlm' is serving text-only;")
    assert "/Users/" not in message
    assert public_model_label("~/models/x/") == "x"
    assert public_model_label("./ckpt") == "ckpt"
    assert public_model_label("/") == "model"
    assert public_model_label(None) == "None"
    assert public_model_label("org/name") == "org/name"


def test_install_hint_without_paths_never_names_the_interpreter(monkeypatch):
    monkeypatch.setattr(mllm, "_managed_desktop_runtime_kind", lambda: None)
    standalone = mllm._vision_install_hint(include_paths=False)
    assert "python -m pip install" in standalone
    assert os.sep + "bin" + os.sep not in standalone

    monkeypatch.setattr(
        mllm, "_managed_desktop_runtime_kind", lambda: "runtime-override"
    )
    monkeypatch.setattr(
        mllm,
        "_managed_desktop_runtime_root",
        lambda: pytest.fail("path-free hint must not resolve the runtime root"),
    )
    override = mllm._vision_install_hint(include_paths=False)
    assert "the Desktop runtime-override folder" in override
    assert "Application Support" not in override


# ---------------------------------------------------------------------------
# The fitting-alias rule.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("hybrid_runtime_ok", [True, False])
@pytest.mark.parametrize(
    "ram_gb", [1, 4, 8, 12, 16, 18, 24, 32, 36, 48, 64, 96, 128, 192, 256, 512]
)
def test_fitting_alias_never_exceeds_machine_memory(ram_gb, hybrid_runtime_ok):
    alias = fitting_vision_alias(ram_gb, hybrid_runtime_ok=hybrid_runtime_ok)
    if alias is None:
        return
    profile = resolve_profile(alias)
    assert profile.supports_image_input and not profile.is_text_only
    assert not profile.experimental and profile.modality == "text"
    assert "-assistant" not in profile.hf_path
    for floor in (profile.vision_min_memory_gb, profile.min_memory_gb):
        assert floor is None or floor <= ram_gb
    need = vision_alias_memory_need_gb(alias, profile)
    assert need is not None and need <= ram_gb
    if not hybrid_runtime_ok:
        assert not profile.is_hybrid and profile.vision_min_memory_gb is None


def test_fitting_alias_is_deterministic_and_picks_the_largest_fit(monkeypatch):
    profiles = {
        "big": ModelProfile(hf_path="o/big", supports_image_input=True),
        "b-tie": ModelProfile(hf_path="o/tie-b", supports_image_input=True),
        "a-tie": ModelProfile(hf_path="o/tie-a", supports_image_input=True),
        "small": ModelProfile(hf_path="o/small", supports_image_input=True),
        "unsized": ModelProfile(hf_path="o/unsized", supports_image_input=True),
        "measured": ModelProfile(hf_path="o/measured", supports_image_input=True),
        "floor": ModelProfile(
            hf_path="o/floor", supports_image_input=True, min_memory_gb=64
        ),
        "hybrid": ModelProfile(
            hf_path="o/hybrid", supports_image_input=True, vision_min_memory_gb=8
        ),
        "experimental": ModelProfile(
            hf_path="o/exp", supports_image_input=True, experimental=True
        ),
        "text": ModelProfile(hf_path="o/text"),
        "drafter": ModelProfile(
            hf_path="o/tiny-it-assistant-bf16", supports_image_input=True
        ),
        "served": ModelProfile(hf_path="o/served", supports_image_input=True),
        "missing": None,
    }
    gib = 1 << 30
    sizes = {
        "o/big": 20 * gib,
        "o/tie-b": 4 * gib,
        "o/tie-a": 4 * gib,
        "o/small": 1 * gib,
        "o/floor": 1 * gib,
        "o/hybrid": 5 * gib,
        "o/exp": 1 * gib,
        "o/tiny-it-assistant-bf16": gib // 4,
        "o/served": 7 * gib,
    }
    monkeypatch.setattr(
        "rapid_mlx.model_aliases.list_builtin_aliases",
        lambda: {name: "x" for name in profiles},
    )
    monkeypatch.setattr(api_utils, "resolve_profile", profiles.get)
    monkeypatch.setattr("rapid_mlx.model_sizes.size_bytes", sizes.get)
    monkeypatch.setattr(
        "rapid_mlx.recommendations.recommendation_footprint_gb",
        lambda alias: 2.0 if alias == "measured" else None,
    )

    # budget = 16 * 0.65 = 10.4 GiB: big (30) is out; served (10.5) is out;
    # hybrid (7.5) is the largest fit; ties fall back to name order.
    assert fitting_vision_alias(16, hybrid_runtime_ok=True) == "hybrid"
    assert fitting_vision_alias(16, hybrid_runtime_ok=False) == "a-tie"
    assert (
        fitting_vision_alias(20, hybrid_runtime_ok=False, exclude_hf_path="o/served")
        == "a-tie"
    )
    assert fitting_vision_alias(20, hybrid_runtime_ok=False) == "served"
    assert fitting_vision_alias(3, hybrid_runtime_ok=True) == "small"
    assert fitting_vision_alias(0, hybrid_runtime_ok=True) is None
    assert fitting_vision_alias(1, hybrid_runtime_ok=True) is None
    assert vision_alias_memory_need_gb("measured", profiles["measured"]) == 2.0


def test_every_suggestion_is_a_real_catalog_alias():
    alias = fitting_vision_alias(16, hybrid_runtime_ok=True)
    assert alias in list_builtin_aliases()


# ---------------------------------------------------------------------------
# Ready banner.
# ---------------------------------------------------------------------------


def test_banner_shows_image_note_under_model_line():
    ep = endpoints_from_bind("127.0.0.1", 8000, model="qwen3.5-4b-4bit")
    text = render_banner(ep, image_note="Why. What.")
    lines = text.splitlines()
    model_at = next(i for i, line in enumerate(lines) if "Model:" in line)
    assert lines[model_at + 1] == "  Images:    off. Why. What."
    assert "Images:" not in render_banner(ep)


class _BannerEngine:
    def __init__(self, is_mllm, reason):
        self.is_mllm = is_mllm
        self.serving_lane_reason = reason


@pytest.mark.parametrize(
    ("engine", "expected"),
    [
        (None, False),
        (_BannerEngine(True, "vision_supported"), False),
        (_BannerEngine(False, "text_lane_forced"), False),
        (_BannerEngine(False, "text_checkpoint"), False),
        (_BannerEngine(False, "vision_memory_insufficient"), True),
        (_BannerEngine(False, "vision_weights_unavailable"), True),
    ],
)
def test_ready_banner_note_only_for_automatic_text_fallback(
    monkeypatch, engine, expected
):
    cfg = reset_config()
    cfg.model_name = "mlx-community/Qwen3.5-4B-MLX-4bit"
    cfg.model_alias = "qwen3.5-4b-4bit"
    monkeypatch.setattr(server, "_engine", engine)
    note = server._text_lane_image_note(cfg)
    if not expected:
        assert note is None
        return
    assert engine.serving_lane_reason in BANNER_TEXT_LANE_REASONS
    assert note == text_lane_image_guidance(
        "qwen3.5-4b-4bit", engine.serving_lane_reason
    )


def test_print_ready_banner_prints_the_note(monkeypatch, capsys):
    cfg = reset_config()
    cfg.model_name = "qwen3.5-4b-4bit"
    cfg.bind_host = "127.0.0.1"
    cfg.bind_port = 8000
    monkeypatch.setattr(
        server, "_engine", _BannerEngine(False, "vision_memory_insufficient")
    )
    server.print_ready_banner()
    out = capsys.readouterr().out
    assert "  Images:    off. Vision for this model needs at least 32 GB" in out
