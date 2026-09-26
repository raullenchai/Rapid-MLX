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
    DESKTOP_HIDDEN_BROKEN_ALIASES,
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


_HOST_PROBES = (
    "_host_ram_gb",
    "_host_hybrid_runtime_ok",
    "_host_vision_runtime_ok",
    "_host_is_desktop",
)
_REAL_PROBES = {name: getattr(api_utils, name) for name in _HOST_PROBES}


def _clear_probe_caches():
    for probe in _REAL_PROBES.values():
        probe.cache_clear()


@pytest.fixture(autouse=True)
def _pin_host(monkeypatch):
    """A 16 GB CLI Mac with a working vision runtime (hybrid-capable)."""
    monkeypatch.setattr(api_utils, "_host_ram_gb", lambda: 16.0)
    monkeypatch.setattr(api_utils, "_host_hybrid_runtime_ok", lambda: True)
    monkeypatch.setattr(api_utils, "_host_vision_runtime_ok", lambda: True)
    monkeypatch.setattr(api_utils, "_host_is_desktop", lambda: False)
    yield
    _clear_probe_caches()
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
        "Vision for this model needs more RAM than this Mac's 8 GB, so it "
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


_FORCED_TEXT = (
    "This server was started on the text-only lane (e.g. with "
    "--no-mllm / --text-only). Restart without it for image input."
)
# A vision model with no RAM floor and a plain attention backbone: dropping a
# text-lane flag really does enable images for it on the pinned 16 GB host.
_UNBLOCKED = "gemma-4-e4b-4bit"
_UNBLOCKED_BASE = (
    f"Model '{_UNBLOCKED}' is serving text-only; image input is unsupported."
)


def test_forced_text_lane_is_worded_neutrally():
    """The engine cannot tell --no-mllm from other forced-text causes (e.g. a
    residency load), so the copy names the flag as an example, not a fact."""
    message = _chat_image_error("text_lane_forced", model_name=_UNBLOCKED)
    assert message == f"{_UNBLOCKED_BASE} {_FORCED_TEXT}"
    assert "was forced with" not in message


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


def test_speculative_decode_names_the_switch_that_turns_it_off():
    """Usually the catalog's MTP default: name --no-spec-decode, not flags the
    user never passed."""
    message = _chat_image_error("text_lane_speculative_decode", model_name=_UNBLOCKED)
    assert message == (
        f"{_UNBLOCKED_BASE} Speculative decoding (MTP) is on, and only the text "
        "lane runs it. Restart with --no-spec-decode (and without any "
        "--spec-decode / --force-spec-decode flag) for image input."
    )
    assert text_lane_image_guidance(
        "org/x", "text_lane_speculative_decode", desktop=True
    ) == (
        "Speculative decoding is on, and only the text lane runs it. Turn it off "
        "in Settings → Performance to add photos."
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
            "The installed vision runtime (mlx-vlm) does not support this "
            "model's vision architecture, so it started text-only.",
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
    client, _ = _client("text_lane_forced", model_name=_UNBLOCKED)
    response = client.post(
        "/v1/responses",
        json={
            "model": _UNBLOCKED,
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
    assert error["message"].endswith(_FORCED_TEXT)


def _anthropic_image(client, model="qwen3.5-4b-4bit"):
    return client.post(
        "/v1/messages",
        json={
            "model": model,
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
    client, _ = _client("text_lane_forced", model_name=_UNBLOCKED)
    response = _anthropic_image(client, _UNBLOCKED)
    assert response.status_code == 400, response.text
    assert response.json()["detail"] == (
        f"Model '{_UNBLOCKED}' does not support image inputs. {_FORCED_TEXT}"
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


_RAM_SWEEP = [n / 2 for n in range(1, 1025)]  # 0.5 GB .. 512 GB


def test_desktop_hidden_list_matches_the_swift_source():
    import re
    from pathlib import Path

    swift = (
        Path(__file__).resolve().parents[1]
        / "apps/rapid-mac/Sources/Rapid/Server/ModelPickerVisibility.swift"
    ).read_text()
    block = re.search(
        r"static let knownBrokenForTextChat: Set<String> = \[(.*?)\]", swift, re.S
    )
    assert block is not None
    assert frozenset(re.findall(r'"([^"]+)"', block.group(1))) == (
        DESKTOP_HIDDEN_BROKEN_ALIASES
    )


@pytest.mark.parametrize("hybrid_runtime_ok", [True, False])
def test_desktop_hidden_family_is_never_suggested(hybrid_runtime_ok):
    suggested = {
        fitting_vision_alias(ram, hybrid_runtime_ok=hybrid_runtime_ok)
        for ram in _RAM_SWEEP
    }
    assert suggested.isdisjoint(DESKTOP_HIDDEN_BROKEN_ALIASES)
    assert not any(a and a.startswith("gemma-4-e2b") for a in suggested)


def test_catalog_assistant_drafters_are_never_suggested():
    drafters = {
        alias
        for alias in list_builtin_aliases()
        if "-assistant" in resolve_profile(alias).hf_path.lower()
        and resolve_profile(alias).supports_image_input
    }
    # The current catalog ships Gemma 4 sidecar drafters; pin that they exist
    # so this test cannot pass vacuously, and that none is ever suggested.
    assert {"gemma-4-e4b-assistant", "gemma-4-31b-assistant"} <= drafters
    for hybrid_runtime_ok in (True, False):
        for ram in _RAM_SWEEP:
            alias = fitting_vision_alias(ram, hybrid_runtime_ok=hybrid_runtime_ok)
            assert alias not in drafters


def test_no_fit_is_said_plainly(monkeypatch):
    monkeypatch.setattr(api_utils, "fitting_vision_alias", lambda *a, **k: None)
    guidance = text_lane_image_guidance(
        "qwen3.5-4b-4bit", "vision_memory_insufficient", ram_gb=8
    )
    assert guidance == (
        "Vision for this model needs at least 32 GB of RAM; this Mac has 8 GB, "
        "so it started text-only. No catalog vision model fits this Mac's memory."
    )


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
        "qwen3.5-4b-4bit", engine.serving_lane_reason, include_paths=True
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


# ---------------------------------------------------------------------------
# Review r1: never-raise, cached probes, served-model identity, missing
# vision runtime, banner paths, unknown RAM, Desktop-only remedies.
# ---------------------------------------------------------------------------


def _boom(*_args, **_kwargs):
    raise RuntimeError("broken catalog")


def test_raising_guidance_keeps_the_original_400_everywhere(monkeypatch, capsys):
    monkeypatch.setattr(api_utils, "text_lane_image_guidance", _boom)
    message = _chat_image_error("text_lane_forced")
    assert message == _BASE

    client, _ = _client("text_lane_forced")
    response = _anthropic_image(client)
    assert response.status_code == 400, response.text
    assert response.json()["detail"] == (
        "Model 'qwen3.5-4b-4bit' does not support image inputs."
    )

    cfg = reset_config()
    cfg.model_name = "qwen3.5-4b-4bit"
    cfg.bind_host = "127.0.0.1"
    cfg.bind_port = 8000
    monkeypatch.setattr(
        server, "_engine", _BannerEngine(False, "vision_memory_insufficient")
    )
    server.print_ready_banner()
    out = capsys.readouterr().out
    assert "Ready: http://127.0.0.1:8000" in out
    assert "Images:" not in out


def test_raising_identity_lookup_is_also_contained(monkeypatch):
    monkeypatch.setattr(api_utils, "served_model_catalog_name", _boom)
    assert (
        api_utils.image_rejection_guidance("text_lane_forced", engine=object()) is None
    )


def test_host_probes_run_once_per_process(monkeypatch):
    import subprocess

    for name, probe in _REAL_PROBES.items():
        monkeypatch.setattr(api_utils, name, probe)
    _clear_probe_caches()
    calls = []

    class _Done:
        stdout = str(16 << 30)

    def fake_run(cmd, **_kwargs):
        calls.append(cmd)
        return _Done()

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr("rapid_mlx.telemetry.track._surface_from_role", lambda: "cli")
    for _ in range(5):
        _chat_image_error("vision_memory_insufficient")
    assert calls == [["sysctl", "-n", "hw.memsize"]]
    assert api_utils._host_ram_gb() == 16.0
    assert api_utils._host_is_desktop() is False
    assert api_utils._host_hybrid_runtime_ok() in (True, False)


@pytest.mark.parametrize(
    ("installed", "dist_version", "expected"),
    [
        (False, None, False),
        (True, None, False),
        (True, "0.0.1", False),
        (True, "ok", True),
    ],
)
def test_vision_runtime_probe_is_import_free(
    monkeypatch, installed, dist_version, expected
):
    from importlib.metadata import PackageNotFoundError

    monkeypatch.setattr(
        api_utils, "_host_vision_runtime_ok", _REAL_PROBES["_host_vision_runtime_ok"]
    )
    _clear_probe_caches()
    monkeypatch.setattr(mllm, "_mlx_vlm_installed", lambda: installed)

    def fake_version(_dist):
        if dist_version is None:
            raise PackageNotFoundError("mlx-vlm")
        return mllm.VALIDATED_MLX_VLM_VERSION if dist_version == "ok" else dist_version

    monkeypatch.setattr(api_utils, "version", fake_version)
    assert api_utils._host_vision_runtime_ok() is expected


def test_served_model_name_uses_the_resolved_alias():
    """``serve qwen3.6-35b --served-model-name gpt-4o``: explain qwen3.6-35b."""
    cfg = reset_config()
    engine = _TextLaneEngine("text_lane_forced")
    cfg.engine = engine
    cfg.model_name = "gpt-4o"
    cfg.model_alias = "qwen3.6-35b"
    assert api_utils.served_model_catalog_name(engine) == "qwen3.6-35b"
    guidance = api_utils.image_rejection_guidance(
        "text_lane_forced", engine=engine, model_name="gpt-4o"
    )
    assert guidance is not None
    assert guidance.startswith("Its catalog entry pins it to text-only serving.")


def test_resident_model_is_explained_not_the_primary():
    from rapid_mlx.runtime.model_registry import ModelEntry, ModelRegistry

    primary = _TextLaneEngine("vision_supported")
    resident = _TextLaneEngine("text_lane_forced")
    registry = ModelRegistry()
    registry.add(
        ModelEntry(engine=primary, model_name="gemma3-12b-4bit", model_path="p"),
        is_default=True,
    )
    registry.add(
        ModelEntry(
            engine=resident,
            model_name="my-resident",
            model_path="mlx-community/unknown-path",
            aliases={"qwen3.6-35b"},
        )
    )
    client, _ = _client("vision_supported", model_name="gemma3-12b-4bit")
    cfg = api_utils_get_config()
    cfg.engine = primary
    cfg.model_registry = registry
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "my-resident",
            "max_tokens": 8,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": _IMAGE_URL}},
                    ],
                }
            ],
        },
    )
    assert response.status_code == 400, response.text
    error = response.json()["detail"]["error"]
    assert error["code"] == "image_input_unsupported"
    assert "Its catalog entry pins it to text-only serving." in error["message"]
    assert "--no-mllm" not in error["message"]


def api_utils_get_config():
    from rapid_mlx.config import get_config

    return get_config()


def test_identity_falls_back_to_first_name_and_to_none():
    from rapid_mlx.runtime.model_registry import ModelEntry, ModelRegistry

    cfg = reset_config()
    stranger = object()
    assert api_utils.served_model_catalog_name(stranger) is None
    registry = ModelRegistry()
    registry.add(ModelEntry(engine=stranger, model_name="custom", model_path="/x/y"))
    cfg.model_registry = registry
    assert api_utils.served_model_catalog_name(stranger) == "custom"


def test_missing_vision_runtime_leads_with_the_install_hint(monkeypatch):
    monkeypatch.setattr(mllm, "_managed_desktop_runtime_kind", lambda: None)
    hint = " ".join(mllm._vision_install_hint(include_paths=False).split())
    alias = fitting_vision_alias(16.0, hybrid_runtime_ok=True)
    for reason in ("text_checkpoint", "vision_memory_insufficient"):
        guidance = text_lane_image_guidance(
            "qwen3.5-4b-4bit", reason, vision_runtime_ok=False
        )
        assert guidance is not None
        # Install first, only then a model to serve.
        assert guidance.endswith(
            "Image input needs the vision runtime (mlx-vlm), which is not usable "
            f"here. {hint} Then for image input, serve '{alias}', a vision model "
            "that fits this Mac."
        )
    nothing_fits = text_lane_image_guidance(
        "org/x", "text_checkpoint", vision_runtime_ok=False, ram_gb=0.5
    )
    assert nothing_fits.endswith(
        f"{hint} Then no catalog vision model fits this Mac's memory."
    )
    assert text_lane_image_guidance(
        "qwen3.5-4b-4bit", "vision_architecture_unavailable", vision_runtime_ok=False
    ) == (
        "The vision runtime (mlx-vlm) is missing or not usable, so this model "
        f"started text-only. {hint}"
    )


def test_banner_keeps_the_interpreter_path_http_does_not(monkeypatch):
    import sys

    monkeypatch.setattr(mllm, "_managed_desktop_runtime_kind", lambda: None)
    cfg = reset_config()
    cfg.model_name = "qwen3.5-4b-4bit"
    monkeypatch.setattr(
        server, "_engine", _BannerEngine(False, "vision_hybrid_runtime_unsupported")
    )
    note = server._text_lane_image_note(cfg)
    assert note is not None and sys.executable in note
    message = _chat_image_error("vision_hybrid_runtime_unsupported")
    assert sys.executable not in message and "python -m pip" in message


def test_unknown_ram_drops_ram_and_fit_wording():
    guidance = text_lane_image_guidance(
        "qwen3.5-4b-4bit", "vision_memory_insufficient", ram_gb=0.0
    )
    assert guidance == (
        "Vision for this model needs at least 32 GB of RAM, so it started "
        "text-only. Serve a vision-capable model for image input."
    )
    assert text_lane_image_guidance(
        "org/x", "vision_memory_insufficient", ram_gb=0.0, desktop=True
    ) == (
        "Vision for this model needs more RAM, so it started text-only. Choose "
        "a vision model in the model picker."
    )


def test_desktop_gets_only_app_actionable_remedies(monkeypatch):
    monkeypatch.setattr(mllm, "_managed_desktop_runtime_kind", lambda: "embedded")
    reasons = [
        "vision_memory_insufficient",
        "vision_hybrid_runtime_unsupported",
        "vision_architecture_unavailable",
        "text_lane_forced",
        "text_lane_speculative_decode",
        "text_checkpoint",
        "vision_weights_unavailable",
        "vision_hybrid_cache_unsupported",
    ]
    alias = fitting_vision_alias(16.0, hybrid_runtime_ok=True)
    for runtime_ok in (True, False):
        for reason in reasons:
            guidance = text_lane_image_guidance(
                "org/x", reason, desktop=True, vision_runtime_ok=runtime_ok
            )
            assert guidance is not None, reason
            assert "--" not in guidance, (reason, guidance)
            assert "serve '" not in guidance and "restart" not in guidance
            assert "pip install" not in guidance and "python" not in guidance
    assert text_lane_image_guidance("org/x", "text_lane_forced", desktop=True) == (
        "This model was started text-only. Choose a vision model in the model "
        f"picker, such as '{alias}', which fits this Mac."
    )
    assert text_lane_image_guidance(
        "org/x", "text_lane_speculative_decode", desktop=True
    ).startswith("Speculative decoding is on, and only the text lane runs it.")


def test_route_detects_desktop_through_the_cached_probe(monkeypatch):
    monkeypatch.setattr(api_utils, "_host_is_desktop", lambda: True)
    message = _chat_image_error("text_lane_forced")
    assert "--no-mllm" not in message
    assert "model picker" in message


# ---------------------------------------------------------------------------
# Review r2.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("hybrid_runtime_ok", [True, False])
def test_suggestion_never_starts_on_the_speculative_text_lane(hybrid_runtime_ok):
    """A default-on MTP alias would serve text-only again after ``serve``."""
    suggested = {
        fitting_vision_alias(ram, hybrid_runtime_ok=hybrid_runtime_ok)
        for ram in _RAM_SWEEP
    } - {None}
    assert suggested
    for alias in suggested:
        profile = resolve_profile(alias)
        assert not (
            profile.mtp_default_enabled
            and (
                profile.supports_native_mtp
                or profile.mtp_continuous_batching_tier == "verified"
            )
        ), alias
    assert fitting_vision_alias(36, hybrid_runtime_ok=True) != "qwen3.8-27b-4bit-fp16"


def test_suggested_aliases_resolve_to_the_vision_lane_under_default_serve():
    """End to end with the CLI's own default-MTP decision."""
    from rapid_mlx import cli

    for ram in (8, 16, 24, 32, 36, 48, 64, 96, 128):
        alias = fitting_vision_alias(ram, hybrid_runtime_ok=True)
        assert alias is not None
        injects = (
            cli._alias_native_mtp_capable(alias)
            or cli._alias_continuous_mtp_tier(alias) == "verified"
        ) and cli._alias_mtp_default_enabled(alias)
        assert not injects, (ram, alias)


def test_evicted_resident_gets_generic_copy_not_the_primarys():
    from rapid_mlx.runtime.model_registry import ModelEntry, ModelRegistry

    cfg = reset_config()
    primary = _TextLaneEngine("text_lane_forced")
    cfg.engine = primary
    cfg.model_name = "qwen3.6-35b"
    cfg.model_alias = "qwen3.6-35b"
    registry = ModelRegistry()
    registry.add(
        ModelEntry(engine=primary, model_name="qwen3.6-35b", model_path="p"),
        is_default=True,
    )
    cfg.model_registry = registry
    evicted = _TextLaneEngine("text_lane_forced")
    assert api_utils.served_model_catalog_name(evicted) is None
    guidance = api_utils.image_rejection_guidance("text_lane_forced", engine=evicted)
    assert guidance == _FORCED_TEXT
    # The routes also pass the (primary's) label; it must not be used either.
    assert (
        api_utils.image_rejection_guidance(
            "text_lane_forced", engine=evicted, model_name="qwen3.6-35b"
        )
        == _FORCED_TEXT
    )
    assert api_utils.served_model_catalog_name(primary) == "qwen3.6-35b"


def test_anthropic_document_block_gets_no_image_guidance():
    client, _ = _client("text_lane_forced")
    response = client.post(
        "/v1/messages",
        json={
            "model": "qwen3.5-4b-4bit",
            "max_tokens": 8,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": "JVBERi0xLjQK",
                            },
                        },
                        {"type": "text", "text": "summarise"},
                    ],
                }
            ],
        },
    )
    assert response.status_code == 400, response.text
    assert response.json()["detail"] == (
        "Model 'qwen3.5-4b-4bit' does not support document inputs."
    )


def test_broken_user_aliases_file_is_never_read_on_rejection(monkeypatch):
    """Guidance uses the built-in catalog only; a malformed user-alias file
    (read by ``resolve_profile`` for non-alias names) must not matter."""
    monkeypatch.setattr(
        "rapid_mlx.user_aliases.validated_user_aliases", _boom, raising=True
    )
    cfg = reset_config()
    engine = _TextLaneEngine("vision_memory_insufficient")
    cfg.engine = engine
    cfg.model_name = resolve_profile("qwen3.5-4b-4bit").hf_path
    guidance = api_utils.image_rejection_guidance(
        "vision_memory_insufficient", engine=engine
    )
    assert guidance is not None
    assert guidance.startswith("Vision for this model needs at least 32 GB of RAM")
    # And the guidance builder itself, handed an HF path directly.
    direct = text_lane_image_guidance(
        resolve_profile("qwen3.5-4b-4bit").hf_path, "vision_memory_insufficient"
    )
    assert direct.startswith("Vision for this model needs at least 32 GB of RAM")


# ---------------------------------------------------------------------------
# Review r3: a text-lane flag can hide a vision blocker. Dropping the flag is
# advised only when it would really enable images.
# ---------------------------------------------------------------------------

_FLAG_REASONS = ("text_lane_forced", "text_lane_speculative_decode")


@pytest.mark.parametrize("reason", _FLAG_REASONS)
def test_flag_copy_names_the_memory_floor_it_would_hit(reason):
    guidance = text_lane_image_guidance("qwen3.5-9b-4bit", reason, ram_gb=16)
    alias = fitting_vision_alias(16, hybrid_runtime_ok=True)
    assert guidance.endswith(
        "Even then, vision for this model needs at least 32 GB of RAM; this Mac "
        f"has 16 GB. For image input, serve '{alias}', a vision model that fits "
        "this Mac."
    )
    assert "Restart" not in guidance and "restart with" not in guidance


@pytest.mark.parametrize("reason", _FLAG_REASONS)
@pytest.mark.parametrize(("vision_ok", "hybrid_ok"), [(False, True), (True, False)])
def test_flag_copy_installs_the_runtime_first(
    monkeypatch, reason, vision_ok, hybrid_ok
):
    monkeypatch.setattr(mllm, "_managed_desktop_runtime_kind", lambda: None)
    hint = " ".join(mllm._vision_install_hint(include_paths=False).split())
    guidance = text_lane_image_guidance(
        "qwen3.5-9b-4bit",
        reason,
        ram_gb=64,
        vision_runtime_ok=vision_ok,
        hybrid_runtime_ok=hybrid_ok,
    )
    assert (
        f"Image input also needs a working vision runtime (mlx-vlm). {hint} Then "
        in guidance
    )
    assert guidance.endswith("for image input.")


@pytest.mark.parametrize("reason", _FLAG_REASONS)
def test_flag_copy_matches_the_lane_the_model_would_get(reason):
    """Sweep: advise dropping the flag exactly when the unflagged lane decision
    (memory floor, hybrid runtime, vision runtime) would be a vision lane."""
    for alias in sorted(list_builtin_aliases()):
        profile = resolve_profile(alias)
        if not profile.supports_image_input or profile.is_text_only:
            continue
        for ram in (8, 16, 24, 32, 64):
            for vision_ok in (True, False):
                for hybrid_ok in (True, False):
                    guidance = text_lane_image_guidance(
                        alias,
                        reason,
                        ram_gb=ram,
                        vision_runtime_ok=vision_ok,
                        hybrid_runtime_ok=hybrid_ok,
                    )
                    hybrid = (
                        profile.is_hybrid or profile.vision_min_memory_gb is not None
                    )
                    floor = profile.vision_min_memory_gb
                    unblocked = (
                        vision_ok
                        and (hybrid_ok or not hybrid)
                        and (floor is None or floor <= ram)
                    )
                    advises_restart_only = guidance.split(". ")[-1].startswith(
                        "Restart"
                    )
                    assert advises_restart_only == unblocked, (
                        alias,
                        ram,
                        vision_ok,
                        hybrid_ok,
                        guidance,
                    )
