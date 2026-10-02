# SPDX-License-Identifier: Apache-2.0
"""Base-wheel text-only degrade for standard-attention VLM checkpoints.

Companion to ``test_basewheel_hybrid_vlm_serve_1017`` (which covers the
#3831 hybrid/arrays-cache degrade). The #3831 contract only auto-downgraded
aliases whose LANGUAGE backbone is hybrid/linear-attention; a curated VLM
alias whose checkpoint has a text-capable standard-attention backbone
(Gemma 4, Qwen3-VL, Gemma 3(n), …) still exited with
``RAPID-MLX-STARTUP-FAILURE: runtime_extra_missing extra=vision`` on a
clean ``pip install rapid-mlx`` even though ``--no-mllm`` serves the same
checkpoint from the base wheel.

These tests pin the extended contract:

* vision runtime ABSENT + a text-lane-loadable backbone (mlx-lm ships the
  arch module, or the vendored Gemma 4 family loader covers it) → the serve
  proceeds on the text lane with one warning naming the lost image/video
  input and the install-method-aware repair command;
* image/video requests get the existing capability-rejected handling;
* a backbone whose ONLY loader lives on the MLLM lane (Bonsai 2's
  ``prism_hadamard_qwen35``), a BROKEN runtime, and an explicit ``--mllm``
  request keep the loud ``[vision]``-required failure;
* the Desktop sidecar keeps its silent-degrade contract (no stderr copy).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest


def _args(
    model: str = "gemma-4-26b-4bit", *, mllm: bool = False, no_mllm: bool = False
):
    return SimpleNamespace(model=model, mllm=mllm, no_mllm=no_mllm)


GEMMA4_VLM_CONFIG = {
    "model_type": "gemma4",
    "architectures": ["Gemma4ForConditionalGeneration"],
    "vision_config": {},
}
PRISM_PACK_CONFIG = {
    "model_type": "prism_hadamard_qwen35",
    "architectures": ["PrismHadamardQwen35ForConditionalGeneration"],
    "vision_config": {},
}
UNKNOWN_VLM_CONFIG = {
    "model_type": "brand_new_vlm_arch",
    "architectures": ["BrandNewVLMForConditionalGeneration"],
    "vision_config": {},
}
TEXT_CONFIG = {
    "model_type": "qwen3",
    "architectures": ["Qwen3ForCausalLM"],
}


def _meta(config):
    return SimpleNamespace(config=config, snapshot_dir=Path("/snap"))


def _patch_lane_probes(monkeypatch, *, is_mllm: bool, cache_mode=None):
    """Stub the offline lane probes ``resolve_serving_lane_decision`` reads.

    ``is_mllm`` simulates the checkpoint-weight evidence the engine classifies
    after the pull; ``cache_mode`` is the backbone cache contract (``None`` =
    ordinary attention, i.e. NOT the #3831 arrays path).
    """
    from rapid_mlx.api import utils as api_utils

    monkeypatch.setattr(api_utils, "is_mllm_model", lambda name: is_mllm)
    monkeypatch.setattr(
        api_utils, "mllm_backbone_cache_mode", lambda name: cache_mode
    )


def _mock_vision_absent(monkeypatch) -> None:
    from rapid_mlx.models import mllm as mllm_mod

    monkeypatch.setattr(
        mllm_mod,
        "vision_runtime_status",
        lambda: (mllm_mod.VisionRuntimeStatus.ABSENT, "mlx_vlm"),
    )


def _mock_vision_broken(monkeypatch) -> None:
    from rapid_mlx.models import mllm as mllm_mod

    monkeypatch.setattr(
        mllm_mod,
        "vision_runtime_status",
        lambda: (mllm_mod.VisionRuntimeStatus.BROKEN, "PIL"),
    )


def _patch_degrade_config(monkeypatch, config) -> None:
    """Feed the text-degrade probe a materialized checkpoint config."""
    from rapid_mlx.api import utils as api_utils

    monkeypatch.setattr(
        api_utils, "read_model_metadata", lambda _name: _meta(config)
    )


def _allow_desktop_warning(monkeypatch) -> None:
    monkeypatch.setattr(
        "rapid_mlx.runtime.optional_runtime._running_in_desktop_sidecar",
        lambda: False,
    )


class _ReachedPastVisionGuardError(Exception):
    """Sentinel proving the vision guard did NOT sys.exit(2)."""


def _stub_post_guard_sentinel(monkeypatch) -> None:
    import rapid_mlx.audio.probe as audio_probe

    def _raise(_name):
        raise _ReachedPastVisionGuardError()

    monkeypatch.setattr(audio_probe, "is_audio_model_alias", _raise)


# ---------------------------------------------------------------------------
# Base install (vision runtime ABSENT): Gemma 4 serves text-only.
# ---------------------------------------------------------------------------


def test_base_install_gemma4_boots_text_only_with_warning(monkeypatch, capsys):
    """A cached Gemma 4 checkpoint on a base wheel degrades to the text lane:
    the boot guard does not exit, and one warning names what is lost plus the
    install-method-aware repair command."""
    from rapid_mlx import cli

    _patch_lane_probes(monkeypatch, is_mllm=True, cache_mode=None)
    _mock_vision_absent(monkeypatch)
    _patch_degrade_config(monkeypatch, GEMMA4_VLM_CONFIG)
    _allow_desktop_warning(monkeypatch)
    _stub_post_guard_sentinel(monkeypatch)

    args = _args("gemma-4-26b-4bit")
    with pytest.raises(_ReachedPastVisionGuardError):
        cli.serve_command(args)

    err = capsys.readouterr().err
    assert err.count("warning: vision runtime absent") == 1
    assert "image and video input" in err
    assert "rapid-mlx[vision]==" in err


def test_base_install_gemma4_fresh_cache_degrades(monkeypatch, capsys):
    """Nothing cached yet: the boot guard prefetches config.json alone, sees a
    text-lane-loadable backbone, and degrades instead of demanding [vision]."""
    from rapid_mlx import cli, model_aliases, model_metadata
    from rapid_mlx.api import utils as api_utils

    state = {"current": None}

    def _read(_name):
        return state["current"]

    def _prefetch(_hf_path):
        state["current"] = _meta(GEMMA4_VLM_CONFIG)

    monkeypatch.setattr(model_metadata, "read_model_metadata", _read)
    monkeypatch.setattr(api_utils, "read_model_metadata", _read)
    monkeypatch.setattr(cli, "_prefetch_config_for_lane_guard", _prefetch)
    _mock_vision_absent(monkeypatch)
    _allow_desktop_warning(monkeypatch)

    args = _args("gemma-4-26b-4bit")
    # Fresh install: no weight evidence, so the guard falls back to the alias
    # profile + config chain — which must answer "degrades", not "needs vision".
    assert cli._serve_will_run_on_mllm_lane(args) is False

    profile = model_aliases.resolve_profile("gemma-4-26b-4bit")
    assert profile is not None
    assert cli._warn_vision_text_only_degrade(profile, args=args) is True
    err = capsys.readouterr().err
    assert err.count("warning: vision runtime absent") == 1
    assert "image and video input" in err


def test_degraded_lane_contract_is_vision_runtime_absent(monkeypatch):
    """The engine-side resolver returns the SAME lane ``--no-mllm`` selects:
    text lane, reason ``vision_runtime_absent``, auto-downgrade flagged so
    diagnostics and the ready banner name the cause."""
    from rapid_mlx.api.utils import resolve_serving_lane_decision

    _patch_lane_probes(monkeypatch, is_mllm=True, cache_mode=None)
    _mock_vision_absent(monkeypatch)
    _patch_degrade_config(monkeypatch, GEMMA4_VLM_CONFIG)

    decision = resolve_serving_lane_decision("gemma-4-26b-4bit")
    assert decision.is_mllm is False
    assert decision.reason == "vision_runtime_absent"
    assert decision.auto_text_fallback is True


def test_degraded_gemma4_rejects_image_and_video_with_capability_event(
    monkeypatch, capsys
):
    """With the degraded lane contract, image/video requests are rejected by
    the existing capability boundary and emit the telemetry event."""
    from rapid_mlx import cli, model_aliases, server
    from rapid_mlx.api.utils import validate_content_blocks_for_capabilities
    from rapid_mlx.models import mllm as mllm_mod

    _patch_lane_probes(monkeypatch, is_mllm=True, cache_mode=None)
    monkeypatch.setattr(
        mllm_mod,
        "vision_runtime_status",
        lambda: (mllm_mod.VisionRuntimeStatus.ABSENT, "mlx_vlm"),
    )
    _patch_degrade_config(monkeypatch, GEMMA4_VLM_CONFIG)
    _allow_desktop_warning(monkeypatch)
    monkeypatch.setattr(server, "_preflight_vision_runtime", lambda *_a, **_kw: None)
    monkeypatch.setattr(server, "_ensure_routing_config", lambda _model: None)
    monkeypatch.setattr(
        "rapid_mlx.utils.tokenizer._resolve_subfolder_checkpoint",
        lambda model: model,
    )
    monkeypatch.setattr(
        "rapid_mlx.model_metadata.read_model_metadata", lambda _model: None
    )
    events = []
    monkeypatch.setattr(
        "rapid_mlx.telemetry.inference.emit_capability_rejected",
        lambda capability, **props: events.append((capability, props)),
    )

    assert cli._warn_vision_text_only_degrade(
        model_aliases.resolve_profile("gemma-4-26b-4bit"), args=_args()
    ) is True
    serving_checkpoint = server._resolve_serving_checkpoint("gemma-4-26b-4bit")
    assert serving_checkpoint.is_mllm is False
    assert serving_checkpoint.lane_reason == "vision_runtime_absent"
    assert serving_checkpoint.auto_text_fallback is True

    with pytest.raises(Exception) as image_caught:
        validate_content_blocks_for_capabilities(
            [{"content": [{"type": "image_url", "image_url": {"url": "x"}}]}],
            model_name="gemma-4-26b-4bit",
            allow_image=serving_checkpoint.is_mllm,
            allow_video=serving_checkpoint.is_mllm,
        )
    assert getattr(image_caught.value, "code", None) == "image_input_unsupported"

    with pytest.raises(ValueError):
        validate_content_blocks_for_capabilities(
            [{"content": [{"type": "video_url", "video_url": {"url": "x"}}]}],
            model_name="gemma-4-26b-4bit",
            allow_image=serving_checkpoint.is_mllm,
            allow_video=serving_checkpoint.is_mllm,
        )
    assert {capability for capability, _props in events} == {
        "image_input_unsupported",
        "video_input_unsupported",
    }
    assert capsys.readouterr().err.count("warning: vision runtime absent") == 1


# ---------------------------------------------------------------------------
# Bonsai 2 (prism_hadamard_qwen35): no text backbone → unchanged failure.
# ---------------------------------------------------------------------------


def test_bonsai2_pack_still_requires_vision_extra(monkeypatch, capsys):
    """A pack whose only loader lives on the MLLM lane (Bonsai 2's
    ``prism_hadamard_qwen35``) is NOT text-capable: the degrade must not fire
    and the boot guard still exits 2 with the [vision] hint."""
    from rapid_mlx import cli
    from rapid_mlx.model_aliases import resolve_profile

    _patch_lane_probes(monkeypatch, is_mllm=True, cache_mode=None)
    _mock_vision_absent(monkeypatch)
    _patch_degrade_config(monkeypatch, PRISM_PACK_CONFIG)
    _allow_desktop_warning(monkeypatch)
    _stub_post_guard_sentinel(monkeypatch)

    args = _args("bonsai2-27b-2bit")
    assert cli._serve_will_run_on_mllm_lane(args) is True
    profile = resolve_profile("bonsai2-27b-2bit")
    assert profile is not None
    assert cli._warn_vision_text_only_degrade(profile, args=args) is False

    with pytest.raises(SystemExit) as exc_info:
        cli.serve_command(args)
    assert exc_info.value.code == 2
    assert "[vision]" in capsys.readouterr().err


def test_degrade_probe_rejects_mllm_only_pack(monkeypatch):
    """The probe itself fails closed on the MLLM-lane-only pack, whatever the
    alias profile claims."""
    from rapid_mlx.api.utils import checkpoint_serves_text_without_vision

    _mock_vision_absent(monkeypatch)
    _patch_degrade_config(monkeypatch, PRISM_PACK_CONFIG)
    assert checkpoint_serves_text_without_vision("bonsai2-27b-2bit") is False


# ---------------------------------------------------------------------------
# BROKEN runtime and explicit vision requests keep the loud guard.
# ---------------------------------------------------------------------------


def test_broken_runtime_never_degrades(monkeypatch, capsys):
    """Only an ABSENT runtime may degrade. A BROKEN install keeps the repair
    guard and prints no degrade warning."""
    from rapid_mlx import cli
    from rapid_mlx.model_aliases import resolve_profile

    _patch_lane_probes(monkeypatch, is_mllm=True, cache_mode=None)
    _mock_vision_broken(monkeypatch)
    _patch_degrade_config(monkeypatch, GEMMA4_VLM_CONFIG)
    _allow_desktop_warning(monkeypatch)

    profile = resolve_profile("gemma-4-26b-4bit")
    assert profile is not None
    assert cli._warn_vision_text_only_degrade(profile, args=_args()) is False
    assert capsys.readouterr().err == ""
    assert cli._serve_will_run_on_mllm_lane(_args()) is True


def test_explicit_mllm_on_gemma4_still_requires_vision_extra(
    monkeypatch, capsys
):
    """``--mllm`` is a deliberate demand for the vision lane: no degrade, no
    warning, and the guard still exits 2 on a base wheel."""
    from rapid_mlx import cli

    _patch_lane_probes(monkeypatch, is_mllm=True, cache_mode=None)
    _mock_vision_absent(monkeypatch)
    _patch_degrade_config(monkeypatch, GEMMA4_VLM_CONFIG)
    _allow_desktop_warning(monkeypatch)
    _stub_post_guard_sentinel(monkeypatch)

    args = _args("gemma-4-26b-4bit", mllm=True)
    assert cli._serve_will_run_on_mllm_lane(args) is True
    with pytest.raises(SystemExit) as exc_info:
        cli.serve_command(args)
    assert exc_info.value.code == 2
    assert capsys.readouterr().err.count("warning: vision runtime absent") == 0


def test_desktop_sidecar_degrades_silently(monkeypatch, capsys):
    """Desktop owns its copy surface: the degrade still applies but the CLI
    warning stays suppressed (the #3831 sidecar contract, unchanged)."""
    from rapid_mlx import cli
    from rapid_mlx.model_aliases import resolve_profile

    _patch_lane_probes(monkeypatch, is_mllm=True, cache_mode=None)
    _mock_vision_absent(monkeypatch)
    _patch_degrade_config(monkeypatch, GEMMA4_VLM_CONFIG)
    monkeypatch.setattr(
        "rapid_mlx.runtime.optional_runtime._running_in_desktop_sidecar",
        lambda: True,
    )

    profile = resolve_profile("gemma-4-26b-4bit")
    assert profile is not None
    assert cli._warn_vision_text_only_degrade(profile, args=_args()) is False
    assert capsys.readouterr().err == ""
    # The lane contract still degrades — Desktop renders its own copy.
    assert cli._serve_will_run_on_mllm_lane(_args()) is False


# ---------------------------------------------------------------------------
# Eligibility probe unit contracts (real mlx_lm import probes, no mocks).
# ---------------------------------------------------------------------------


def test_text_lane_backbone_probe_matches_installed_mlxl_lm():
    """The probe reads the INSTALLED mlx-lm plus the vendored Gemma 4 family:
    vision arches mlx-lm 0.31+ loads (gemma4, qwen3_vl) and the vendored
    gemma4_unified/assistant loaders are text-capable; the Bonsai 2 pack and
    unknown arches are not."""
    from rapid_mlx.api.utils import _text_lane_loads_model_type

    assert _text_lane_loads_model_type("gemma4") is True
    assert _text_lane_loads_model_type("gemma4_unified") is True
    assert _text_lane_loads_model_type("gemma4_assistant") is True
    assert _text_lane_loads_model_type("qwen3_vl") is True
    assert _text_lane_loads_model_type("prism_hadamard_qwen35") is False
    assert _text_lane_loads_model_type("brand_new_vlm_arch") is False


def test_degrade_probe_fails_closed_without_config(monkeypatch):
    """No config (offline / unreachable Hub) is no evidence: keep the safe
    [vision]-required default."""
    from rapid_mlx.api import utils as api_utils
    from rapid_mlx.api.utils import checkpoint_serves_text_without_vision

    _mock_vision_absent(monkeypatch)
    monkeypatch.setattr(api_utils, "read_model_metadata", lambda _name: None)
    monkeypatch.setattr(
        api_utils, "_prefetch_config_for_degrade_probe", lambda _name: None
    )
    assert checkpoint_serves_text_without_vision("gemma-4-26b-4bit") is False


def test_degrade_probe_rejects_text_config(monkeypatch):
    """A config that declares no vision tower never needed the vision lane."""
    from rapid_mlx.api.utils import checkpoint_serves_text_without_vision

    _mock_vision_absent(monkeypatch)
    _patch_degrade_config(monkeypatch, TEXT_CONFIG)
    assert checkpoint_serves_text_without_vision("gemma-4-26b-4bit") is False


def test_degrade_probe_rejects_unknown_arch(monkeypatch):
    """An arch with no text-lane loader fails closed instead of crashing deep
    in mlx-lm after a silent degrade."""
    from rapid_mlx.api.utils import checkpoint_serves_text_without_vision

    _mock_vision_absent(monkeypatch)
    _patch_degrade_config(monkeypatch, UNKNOWN_VLM_CONFIG)
    assert checkpoint_serves_text_without_vision("gemma-4-26b-4bit") is False


def test_degrade_probe_requires_absent_runtime(monkeypatch):
    """A working (OK) vision runtime serves the vision lane — the probe must
    not degrade anything."""
    from rapid_mlx.api.utils import checkpoint_serves_text_without_vision
    from rapid_mlx.models import mllm as mllm_mod

    monkeypatch.setattr(
        mllm_mod,
        "vision_runtime_status",
        lambda: (mllm_mod.VisionRuntimeStatus.OK, None),
    )
    _patch_degrade_config(monkeypatch, GEMMA4_VLM_CONFIG)
    assert checkpoint_serves_text_without_vision("gemma-4-26b-4bit") is False


def test_degrade_probe_keys_on_top_level_model_type(monkeypatch):
    """The top-level ``model_type`` is the loader dispatch key: a nested
    ``text_config`` label cannot smuggle an unloadable pack into a degrade."""
    from rapid_mlx.api.utils import checkpoint_serves_text_without_vision

    _mock_vision_absent(monkeypatch)
    _patch_degrade_config(
        monkeypatch,
        {
            "model_type": "prism_hadamard_qwen35",
            "base_model_type": "qwen3_5",
            "text_config": {"model_type": "qwen3_5"},
            "vision_config": {},
        },
    )
    assert checkpoint_serves_text_without_vision("bonsai2-27b-2bit") is False
