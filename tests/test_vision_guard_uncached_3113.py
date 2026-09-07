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


VLM_CONFIG = {"architectures": ["Gemma4ForConditionalGeneration"], "vision_config": {}}
HYBRID_VLM_CONFIG = {
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "vision_config": {},
    "text_config": {"layer_types": ["linear_attention", "full_attention"]},
}
TEXT_CONFIG = {"architectures": ["Qwen3ForCausalLM"]}


def _patch_weights(monkeypatch, *, metadata, verdict=None, prefetched=None):
    """``metadata`` is what the cache holds before the guard runs;
    ``prefetched`` is what it holds after the config-only prefetch."""
    from vllm_mlx.api import utils as api_utils

    state = {"current": metadata, "prefetches": []}

    def _read(name):
        return state["current"]

    def _prefetch(hf_path):
        state["prefetches"].append(hf_path)
        state["current"] = prefetched

    monkeypatch.setattr(model_metadata, "read_model_metadata", _read)
    monkeypatch.setattr(api_utils, "read_model_metadata", _read)
    monkeypatch.setattr(cli, "_prefetch_config_for_lane_guard", _prefetch)
    monkeypatch.setattr(
        model_metadata,
        "checkpoint_has_multimodal_weights",
        lambda snapshot_dir, config=None: verdict,
    )
    return state


def _meta(config, snapshot_dir="/snap"):
    return SimpleNamespace(config=config, snapshot_dir=snapshot_dir)


def _needs(model="gemma-4-26b-4bit", **kw):
    kw.setdefault("force_text", False)
    kw.setdefault("requested_spec_decode", "none")
    return cli._alias_needs_vision_runtime_without_weights(model, **kw)


# --- profile + config fallback ----------------------------------------------


def test_uncached_genuine_vlm_alias_needs_vision(monkeypatch):
    """Nothing cached: the guard fetches config.json alone and decides."""
    _patch_profile(monkeypatch, _profile())
    state = _patch_weights(monkeypatch, metadata=None, prefetched=_meta(VLM_CONFIG))
    assert _needs() is True
    assert state["prefetches"] == ["org/checkpoint"]


def test_config_only_snapshot_still_needs_vision(monkeypatch):
    """Config + tokenizer cached, safetensors missing: the weight probe is
    inconclusive, so the config decides without a prefetch."""
    _patch_profile(monkeypatch, _profile())
    state = _patch_weights(monkeypatch, metadata=_meta(VLM_CONFIG), verdict=None)
    assert _needs() is True
    assert state["prefetches"] == []


@pytest.mark.parametrize("verdict", [True, False])
def test_weight_evidence_defers_to_engine_verdict(monkeypatch, verdict):
    """Once the index exists the engine-side ``is_mllm_model`` answered
    already; the fallback must never contradict it."""
    _patch_profile(monkeypatch, _profile())
    _patch_weights(monkeypatch, metadata=_meta(VLM_CONFIG), verdict=verdict)
    assert _needs() is False


def test_metadata_without_config_triggers_prefetch(monkeypatch):
    _patch_profile(monkeypatch, _profile())
    state = _patch_weights(
        monkeypatch, metadata=_meta(None, None), prefetched=_meta(VLM_CONFIG)
    )
    assert _needs() is True
    assert state["prefetches"] == ["org/checkpoint"]


def test_unreachable_hub_is_not_evidence(monkeypatch):
    """Offline / Hub down: no config, so the guard stays silent and lets the
    pull report its own error instead of guessing."""
    _patch_profile(monkeypatch, _profile())
    _patch_weights(monkeypatch, metadata=None, prefetched=None)
    assert _needs() is False


def test_hybrid_backbone_config_keeps_base_wheel_boot(monkeypatch):
    """CI on #3127: Qwen3.5 dense aliases carry ``supports_image_input`` and
    ``is_hybrid=False`` yet auto-downgrade to the text lane on the base wheel
    (``vision_hybrid_runtime_unsupported``). The guard must not block them."""
    _patch_profile(monkeypatch, _profile())
    _patch_weights(monkeypatch, metadata=None, prefetched=_meta(HYBRID_VLM_CONFIG))
    assert _needs("qwen3.5-9b-4bit") is False


def test_text_config_behind_vision_flag_is_exempt(monkeypatch):
    """A profile flag cannot outrank a config that declares no vision tower."""
    _patch_profile(monkeypatch, _profile())
    _patch_weights(monkeypatch, metadata=None, prefetched=_meta(TEXT_CONFIG))
    assert _needs() is False


def test_vendored_text_arch_is_exempt(monkeypatch):
    from vllm_mlx.api import utils as api_utils

    _patch_profile(monkeypatch, _profile())
    _patch_weights(monkeypatch, metadata=None, prefetched=_meta(VLM_CONFIG))
    monkeypatch.setattr(
        api_utils, "mllm_arch_unsupported_but_text_vendored", lambda name: True
    )
    assert _needs() is False


def test_prefetch_respects_offline_switch(monkeypatch):
    calls = []
    monkeypatch.setattr(model_metadata, "hub_offline_mode_active", lambda: True)
    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub, "hf_hub_download", lambda *a, **k: calls.append(a)
    )
    cli._prefetch_config_for_lane_guard("org/checkpoint")
    assert calls == []
    monkeypatch.setattr(model_metadata, "hub_offline_mode_active", lambda: False)
    cli._prefetch_config_for_lane_guard("org/checkpoint")
    assert calls == [("org/checkpoint", "config.json")]


def test_prefetch_swallows_hub_errors(monkeypatch):
    import huggingface_hub

    def _boom(*a, **k):
        raise OSError("no network")

    monkeypatch.setattr(model_metadata, "hub_offline_mode_active", lambda: False)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _boom)
    cli._prefetch_config_for_lane_guard("org/checkpoint")  # must not raise


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
    _patch_weights(monkeypatch, metadata=None, prefetched=_meta(VLM_CONFIG))
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
    _patch_weights(monkeypatch, metadata=None, prefetched=None)
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
