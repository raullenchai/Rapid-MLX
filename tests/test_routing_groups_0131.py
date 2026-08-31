"""Routing-group fixes for 0.13.1 (fix/0131-routing-groups).

FIX 1 — ``qwen4_exp`` (Qwen3.8-Flash-Next) visual-config checkpoints must stay
on the vendored TEXT lane rather than the mlx-vlm MLLM lane, both when served
through a curated ``is_text_only`` alias and when served from an unaliased
local path (via ``_VENDORED_TEXT_FALLBACK_MODEL_TYPES``).

FIX 3 — a requested speculative decoder (``requested_spec_decode != none``) must
slide a vision-capable checkpoint back onto the text lane (reason
``text_lane_speculative_decode``) instead of silently being dropped by the MLLM
lane, which never consumes ``scheduler_config.spec_decode``.

These tests fake the installed mlx-vlm version to >= 0.6.16 (the hybrid-runtime
floor) and assert the resulting lane decision for the Qwen3.8-Flash-Next config
shape returned by the alias artifact's ``config.json`` (``model_type=qwen4_exp``,
``vision_config`` present, ``image_token_id`` present, ``language_model_only==
false``).
"""

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

import pytest

from vllm_mlx.api import utils as utils_mod
from vllm_mlx.api.utils import (
    _VENDORED_TEXT_FALLBACK_MODEL_TYPES,
    mllm_arch_unsupported_but_text_vendored,
    resolve_serving_lane_decision,
)
from vllm_mlx.model_metadata import ModelMetadata
from vllm_mlx.model_profile import ModelProfile


def _flash_next_config() -> dict:
    """The exact config shape the Qwen3.8-Flash-Next-4bit checkpoint ships."""
    return {
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "model_type": "qwen4_exp",
        "vision_config": {"hidden_size": 1024},
        "image_token_id": 248056,
        "language_model_only": False,
    }


def _fake_mlx_vlm_ge_016(monkeypatch):
    """Fake the installed mlx-vlm as a recent version that can drive the
    vision arch — i.e. the scenario where the alias would otherwise be routed
    into the mlx-vlm MLLM lane (the sidecar currently exceeds this floor)."""
    import importlib.metadata as md

    _orig_version = md.version
    monkeypatch.setattr(
        md,
        "version",
        lambda name: "0.6.17" if name == "mlx-vlm" else _orig_version(name),
    )
    # Ensure the MLLM runtime is "supported" under the faked version so the
    # only thing keeping the model off the MLLM lane is our fix.
    monkeypatch.setattr(utils_mod, "mllm_hybrid_runtime_supported", lambda: True)


def _metadata(config: dict, snapshot_dir: Path) -> ModelMetadata:
    return ModelMetadata(
        config=config,
        chat_template=None,
        snapshot_dir=snapshot_dir,
        is_local=False,
    )


def _install_scheduler_stub(monkeypatch) -> None:
    """Let the public serve entry points run in the Linux no-MLX lane."""
    scheduler = ModuleType("vllm_mlx.scheduler")

    class SchedulerConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    scheduler.SchedulerConfig = SchedulerConfig
    monkeypatch.setitem(sys.modules, "vllm_mlx.scheduler", scheduler)


def test_fix1_unaliased_qwen4_exp_vendored_goes_text_lane(monkeypatch, tmp_path):
    """FIX 1: an UNALIASED local qwen4_exp checkpoint (no curated alias profile)
    whose vision config + real vision weights would otherwise route to MLLM must
    stay on the vendored text lane once ``qwen4_exp`` is in
    ``_VENDORED_TEXT_FALLBACK_MODEL_TYPES``."""
    _fake_mlx_vlm_ge_016(monkeypatch)
    assert "qwen4_exp" in _VENDORED_TEXT_FALLBACK_MODEL_TYPES

    # No curated alias for this raw path.
    monkeypatch.setattr(utils_mod, "resolve_profile", lambda name: None)
    monkeypatch.setattr(
        utils_mod,
        "read_model_metadata",
        lambda name: _metadata(_flash_next_config(), tmp_path),
    )
    # Positive multimodal weight evidence — the vendored-text refire is what
    # must pull it back to the text lane, not an inconclusive verdict.
    monkeypatch.setattr(
        utils_mod,
        "checkpoint_has_multimodal_weights",
        lambda snapshot, config: True,
    )
    # This is a missing-runtime-module contract, independent of whichever
    # mlx-vlm release happens to execute the test.  Newer releases may ship
    # qwen4_exp, so model the unavailable architecture explicitly.
    monkeypatch.setattr("importlib.util.find_spec", lambda _name: None)

    # Without the fix the raw mllm model check is True; the simulated missing
    # qwen4_exp module must make the vendored fallback fire.
    assert mllm_arch_unsupported_but_text_vendored("/some/local/qwen4_exp") is True
    decision = resolve_serving_lane_decision("/some/local/qwen4_exp")
    assert decision.is_mllm is False
    assert decision.auto_text_fallback is True
    assert decision.reason == "vision_architecture_unavailable"


def test_fix1_curated_text_alias_goes_text_lane(monkeypatch, tmp_path):
    """FIX 1: the curated qwen3.8-flash-next-4bit alias pins ``is_text_only`` so
    it stays text even against real vision weights and a >= 0.6.16 mlx-vlm."""
    _fake_mlx_vlm_ge_016(monkeypatch)
    curated = ModelProfile(
        hf_path="mlx-community/Qwen3.8-Flash-Next-4bit",
        is_text_only=True,
    )
    monkeypatch.setattr(utils_mod, "resolve_profile", lambda name: curated)
    monkeypatch.setattr(
        utils_mod,
        "read_model_metadata",
        lambda name: _metadata(_flash_next_config(), tmp_path),
    )
    monkeypatch.setattr(
        utils_mod,
        "checkpoint_has_multimodal_weights",
        lambda snapshot, config: True,
    )
    decision = resolve_serving_lane_decision("qwen3.8-flash-next-4bit")
    assert decision.is_mllm is False
    assert decision.reason == "text_checkpoint"


def test_fix2_text_diffusion_resolves_to_assistant_replacement_group():
    """FIX 2: a text-diffusion profile (e.g. diffusion-gemma-26b-4bit) must
    resolve to the SAME replacement group as resident_models._replacement_group
    derives for the text engine ("assistant"), so loading it with a chat model
    resident does not trip the resolved_group != replace_group 409 guard."""
    from vllm_mlx.routes.residency import _resolved_group_for_profile
    from vllm_mlx.runtime.resident_models import ModelEntry, _replacement_group

    # Request-facing profile modality → group, matching the Fix 2 mapping.
    assert _resolved_group_for_profile("text-diffusion") == "assistant"
    assert _resolved_group_for_profile("text") == "assistant"
    assert _resolved_group_for_profile("vision") == "assistant"
    assert _resolved_group_for_profile("image-gen") == "image-gen"
    assert _resolved_group_for_profile("video-gen") == "video-gen"

    # Engine-derivation parity: a text-diffusion engine is a text engine.
    entry = ModelEntry(
        engine=_MockEngine("text"),
        model_name="diffusion-gemma-26b-4bit",
        model_path="diffusion-gemma-26b-4bit",
    )
    assert _replacement_group(entry) == "assistant"


class _MockEngine:
    """Minimal engine stub for resident_models._replacement_group."""

    def __init__(self, modality: str):
        self._modality = modality

    @property
    def is_image_gen(self) -> bool:
        return self._modality == "image-gen"

    @property
    def is_video_gen(self) -> bool:
        return self._modality == "video-gen"

    @property
    def is_mllm(self) -> bool:
        return self._modality == "mllm"


def test_fix3_spec_decode_requested_slides_vision_capable_to_text(
    monkeypatch,
    tmp_path,
):
    """FIX 3: requesting MTP spec-decode on a vision-capable (non-vendored)
    checkpoint must route to the TEXT lane (reason
    ``text_lane_speculative_decode``) so the decoder is honoured instead of
    silently dropped by the MLLM lane, which never consumes
    ``scheduler_config.spec_decode``."""
    _fake_mlx_vlm_ge_016(monkeypatch)
    # A genuinely MLLM-routable vision-capable checkpoint (NOT the qwen4_exp
    # vendored-fallback shape, which Fix 1 already pins to text): the Qwen3.5-4B
    # vision-config shape used by the existing vision-evidence tests.
    monkeypatch.setattr(utils_mod, "resolve_profile", lambda name: None)
    monkeypatch.setattr(
        utils_mod,
        "read_model_metadata",
        lambda name: _metadata(
            {
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "vision_config": {"hidden_size": 1024},
                "image_token_id": 248056,
            },
            tmp_path,
        ),
    )
    monkeypatch.setattr(
        utils_mod,
        "checkpoint_has_multimodal_weights",
        lambda snapshot, config: True,
    )

    # Without a spec-decode request the model is vision-capable -> MLLM lane.
    assert resolve_serving_lane_decision("/local/qwen35-vision").is_mllm is True
    # With a requested decoder -> forced text lane with the fix-3 reason.
    decision = resolve_serving_lane_decision(
        "/local/qwen35-vision", requested_spec_decode="mtp"
    )
    assert decision.is_mllm is False
    assert decision.reason == "text_lane_speculative_decode"
    assert decision.auto_text_fallback is True

    # Spec-decode wins over an explicit --mllm: the MLLM lane can never honour
    # the decoder, so forcing vision must not silently drop it.
    with_mllm = resolve_serving_lane_decision(
        "/local/qwen35-vision",
        force_mllm=True,
        requested_spec_decode="mtp",
    )
    assert with_mllm.is_mllm is False
    assert with_mllm.reason == "text_lane_speculative_decode"

    # An explicit --mllm WITHOUT a speculative request still forces vision.
    vision_only = resolve_serving_lane_decision("/local/qwen35-vision", force_mllm=True)
    assert vision_only.is_mllm is True
    assert vision_only.reason == "vision_lane_forced"


def test_fix3_spec_decode_on_text_checkpoint_keeps_text_lane(monkeypatch):
    monkeypatch.setattr(utils_mod, "is_mllm_model", lambda _name: False)

    decision = resolve_serving_lane_decision(
        "/local/text-checkpoint", requested_spec_decode="mtp"
    )

    assert decision.is_mllm is False
    assert decision.reason == "text_checkpoint"


def test_fix3_cli_helper_forwards_legacy_mtp_as_requested_spec_decode(monkeypatch):
    from vllm_mlx import cli

    seen = {}

    def _resolve(_model, **kwargs):
        seen.update(kwargs)
        return False, True

    monkeypatch.setattr(utils_mod, "resolve_serving_lane", _resolve)

    assert (
        cli._serve_will_run_on_mllm_lane(
            SimpleNamespace(
                model="/local/vision-checkpoint",
                mllm=False,
                no_mllm=False,
                spec_decode="none",
                enable_mtp=True,
            )
        )
        is False
    )
    assert seen["requested_spec_decode"] == "mtp"

    seen.clear()
    assert (
        cli._serve_will_run_on_mllm_lane(
            SimpleNamespace(
                model="/local/vision-checkpoint",
                mllm=False,
                no_mllm=False,
                spec_decode="none",
                enable_mtp=False,
                force_spec_decode=True,
            )
        )
        is False
    )
    assert seen["requested_spec_decode"] == "auto"


def test_cli_explicit_mllm_precedes_automatic_architecture_fallback(monkeypatch):
    """The public serve helper carries ``--mllm`` into the lane SSOT."""
    from vllm_mlx import cli

    monkeypatch.setattr(utils_mod, "is_mllm_model", lambda _name: True)
    monkeypatch.setattr(
        utils_mod,
        "mllm_arch_unsupported_but_text_vendored",
        lambda _name: True,
    )

    assert (
        cli._serve_will_run_on_mllm_lane(
            SimpleNamespace(
                model="/local/unsupported-vision-checkpoint",
                mllm=True,
                no_mllm=False,
                spec_decode="none",
                enable_mtp=False,
                force_spec_decode=False,
            )
        )
        is True
    )


@pytest.mark.parametrize(
    ("flag", "expected"),
    [("--enable-mtp", "mtp"), ("--force-spec-decode", "auto")],
)
def test_fix3_cli_serve_forwards_parsed_spec_decode_to_lane_contract(
    monkeypatch, flag, expected
):
    from vllm_mlx import cli

    class _StopAtPFlashError(Exception):
        pass

    seen = []

    def _resolve(_model, **kwargs):
        seen.append(kwargs)
        return False, True

    def _stop(*_args, **_kwargs):
        raise _StopAtPFlashError

    monkeypatch.setattr(cli, "_check_disk_space", lambda *_a, **_kw: None)
    _install_scheduler_stub(monkeypatch)
    monkeypatch.setattr(cli, "_check_memory_capacity", lambda *_a, **_kw: None)
    monkeypatch.setattr(cli, "_ensure_model_downloaded", lambda *_a, **_kw: None)
    monkeypatch.setattr(utils_mod, "resolve_serving_lane", _resolve)
    monkeypatch.setattr("vllm_mlx.pflash.resolve_pflash_config", _stop)
    monkeypatch.setattr(
        "vllm_mlx._version_check.prompt_upgrade_if_available", lambda: False
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rapid-mlx",
            "serve",
            "qwen3.5-4b-4bit",
            flag,
        ],
    )
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)

    with pytest.raises(_StopAtPFlashError):
        cli.main()

    assert seen
    assert all(call["requested_spec_decode"] == expected for call in seen)


def test_fix3_standalone_server_forwards_parsed_mtp_to_lane_contract(monkeypatch):
    from vllm_mlx import cli, server

    class _StopAtPFlashError(Exception):
        pass

    seen = []

    _install_scheduler_stub(monkeypatch)

    def _resolve(_model, **kwargs):
        seen.append(kwargs)
        return False, True

    monkeypatch.setattr(server, "_ensure_routing_config", lambda _name: None)
    monkeypatch.setattr(server, "resolve_serving_lane", _resolve)
    monkeypatch.setattr(server, "load_model", lambda *_a, **_kw: None)
    monkeypatch.setattr(
        "vllm_mlx.pflash.resolve_pflash_config",
        lambda *_a, **_kw: (_ for _ in ()).throw(_StopAtPFlashError()),
    )
    monkeypatch.setattr(cli, "_port_preflight_or_die", lambda *_a, **_kw: None)
    monkeypatch.setattr("uvicorn.run", lambda *_a, **_kw: None)
    monkeypatch.setattr(
        "vllm_mlx._version_check.prompt_upgrade_if_available", lambda: False
    )
    monkeypatch.setattr(
        "vllm_mlx._version_check.print_staleness_warning_if_any",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "vllm_mlx.server",
            "--model",
            "qwen3.5-4b-4bit",
            "--force-spec-decode",
        ],
    )

    with (
        mock.patch.object(server, "register_audio_routes_if_enabled"),
        pytest.raises(_StopAtPFlashError),
    ):
        server.main()

    assert seen
    assert seen[0]["requested_spec_decode"] == "auto"
