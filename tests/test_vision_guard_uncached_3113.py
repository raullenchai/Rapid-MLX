"""#3113: the pre-download ``[vision]`` boot guard must fire from the alias
profile when the cache holds no weight evidence.

``is_mllm_model`` only promotes a checkpoint on positive weight evidence, so
on a fresh install every genuine VLM alias probed "text checkpoint", the guard
stayed silent, and a base-wheel user downloaded 15 GB of Gemma 4 before the
ImportError. These tests pin the profile fallback and its exemptions, plus the
text-diffusion wording of the guard message.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_mlx import cli, model_aliases, model_metadata


def _profile(**overrides):
    base = dict(
        hf_path="org/checkpoint",
        modality="text",
        supports_image_input=True,
        is_hybrid=False,
        is_text_only=False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _patch_profile(monkeypatch, profile):
    monkeypatch.setattr(model_aliases, "resolve_profile", lambda name: profile)


def _patch_weights(monkeypatch, *, metadata, verdict=None):
    monkeypatch.setattr(model_metadata, "read_model_metadata", lambda name: metadata)
    monkeypatch.setattr(
        model_metadata,
        "checkpoint_has_multimodal_weights",
        lambda snapshot_dir, config=None: verdict,
    )


def _needs(model="gemma-4-26b-4bit", **kw):
    kw.setdefault("force_text", False)
    kw.setdefault("requested_spec_decode", "none")
    return cli._alias_needs_vision_runtime_without_weights(model, **kw)


# --- profile fallback -------------------------------------------------------


def test_uncached_genuine_vlm_alias_needs_vision(monkeypatch):
    _patch_profile(monkeypatch, _profile())
    _patch_weights(monkeypatch, metadata=None)
    assert _needs() is True


def test_config_only_snapshot_still_needs_vision(monkeypatch):
    """Config + tokenizer cached, safetensors missing: the weight probe is
    inconclusive, so the profile still decides."""
    _patch_profile(monkeypatch, _profile())
    metadata = SimpleNamespace(config={"vision_config": {}}, snapshot_dir="/snap")
    _patch_weights(monkeypatch, metadata=metadata, verdict=None)
    assert _needs() is True


@pytest.mark.parametrize("verdict", [True, False])
def test_weight_evidence_defers_to_engine_verdict(monkeypatch, verdict):
    """Once the index exists the engine-side ``is_mllm_model`` answered
    already; the fallback must never contradict it."""
    _patch_profile(monkeypatch, _profile())
    metadata = SimpleNamespace(config={"vision_config": {}}, snapshot_dir="/snap")
    _patch_weights(monkeypatch, metadata=metadata, verdict=verdict)
    assert _needs() is False


def test_metadata_without_config_counts_as_no_evidence(monkeypatch):
    _patch_profile(monkeypatch, _profile())
    _patch_weights(
        monkeypatch, metadata=SimpleNamespace(config=None, snapshot_dir=None)
    )
    assert _needs() is True


# --- exemptions -------------------------------------------------------------


def test_unknown_model_never_needs_vision(monkeypatch):
    _patch_profile(monkeypatch, None)
    assert _needs("someone/unknown") is False


def test_text_only_pin_wins(monkeypatch):
    _patch_profile(monkeypatch, _profile(is_text_only=True))
    assert _needs() is False


def test_hybrid_backbone_vlm_boots_from_base_wheel(monkeypatch):
    """Qwen3.6-style hybrid VLMs auto-downgrade to the text lane (#1178)."""
    _patch_profile(monkeypatch, _profile(is_hybrid=True))
    assert _needs() is False


def test_text_alias_without_image_input_is_exempt(monkeypatch):
    _patch_profile(monkeypatch, _profile(supports_image_input=False))
    assert _needs() is False


def test_no_mllm_routes_to_text_lane(monkeypatch):
    _patch_profile(monkeypatch, _profile())
    assert _needs(force_text=True) is False


def test_requested_spec_decode_routes_to_text_lane(monkeypatch):
    _patch_profile(monkeypatch, _profile())
    assert _needs(requested_spec_decode="mtp") is False


def test_text_diffusion_needs_vision_regardless_of_flags(monkeypatch):
    _patch_profile(
        monkeypatch, _profile(modality="text-diffusion", supports_image_input=False)
    )
    assert _needs("diffusion-gemma-26b-4bit") is True
    assert _needs("diffusion-gemma-26b-4bit", force_text=True) is True
    assert _needs("diffusion-gemma-26b-4bit", requested_spec_decode="mtp") is True


# --- wiring into ``_serve_will_run_on_mllm_lane`` -----------------------------


def _args(**overrides):
    base = dict(
        model="mlx-community/gemma-4-26b-a4b-it-4bit",
        mllm=False,
        no_mllm=False,
        spec_decode="none",
        enable_mtp=False,
        force_spec_decode=False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_lane_helper_falls_back_to_profile_when_probe_says_text(monkeypatch):
    from vllm_mlx.api import utils as api_utils

    monkeypatch.setattr(api_utils, "is_mllm_model", lambda name: False)
    _patch_profile(monkeypatch, _profile())
    _patch_weights(monkeypatch, metadata=None)
    assert cli._serve_will_run_on_mllm_lane(_args()) is True
    assert cli._serve_will_run_on_mllm_lane(_args(no_mllm=True)) is False
    assert cli._serve_will_run_on_mllm_lane(_args(enable_mtp=True)) is False


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"no_mllm": True},
        {"spec_decode": "mtp"},
        {"enable_mtp": True},
        {"force_spec_decode": True},
    ],
)
def test_lane_helper_keeps_text_diffusion_on_vision_runtime(monkeypatch, overrides):
    """codex #3127: the flag short-circuits must not let a text-diffusion
    alias skip the mlx-vlm guard — its runtime ignores those flags."""
    from vllm_mlx.api import utils as api_utils

    monkeypatch.setattr(api_utils, "is_mllm_model", lambda name: False)
    _patch_profile(
        monkeypatch, _profile(modality="text-diffusion", supports_image_input=False)
    )
    _patch_weights(monkeypatch, metadata=None)
    args = _args(model="diffusion-gemma-26b-4bit", **overrides)
    assert cli._serve_will_run_on_mllm_lane(args) is True


def test_alias_modality(monkeypatch):
    _patch_profile(monkeypatch, _profile(modality="text-diffusion"))
    assert cli._alias_modality("diffusion-gemma-26b-4bit") == "text-diffusion"
    _patch_profile(monkeypatch, None)
    assert cli._alias_modality("someone/unknown") is None


# --- guard message ------------------------------------------------------------


def _mask_vision_runtime(monkeypatch):
    from vllm_mlx.models import mllm

    monkeypatch.setattr(
        mllm,
        "vision_runtime_status",
        lambda: (mllm.VisionRuntimeStatus.ABSENT, None),
    )


def test_text_diffusion_guard_message_drops_no_mllm_hint(monkeypatch, capsys):
    from vllm_mlx.models.mllm import require_mlx_vlm_or_exit

    _mask_vision_runtime(monkeypatch)
    with pytest.raises(SystemExit) as exc_info:
        require_mlx_vlm_or_exit("diffusion-gemma-26b-4bit", text_diffusion=True)
    assert exc_info.value.code == 2
    err = capsys.readouterr().err
    assert "text-diffusion alias" in err
    assert "rapid-mlx[vision]" in err.replace("'", "")
    assert "--no-mllm" not in err


def test_vision_guard_message_keeps_no_mllm_hint(monkeypatch, capsys):
    from vllm_mlx.models.mllm import require_mlx_vlm_or_exit

    _mask_vision_runtime(monkeypatch)
    with pytest.raises(SystemExit):
        require_mlx_vlm_or_exit("gemma-4-26b-4bit")
    err = capsys.readouterr().err
    assert "vision/multimodal alias" in err
    assert "--no-mllm" in err
