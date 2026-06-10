# SPDX-License-Identifier: Apache-2.0
"""Contract pins for the ``AliasProfile.modality`` field — added on
the ``feat/diffusion-gemma`` skeleton PR. These tests guarantee:

  1. Legacy aliases (no ``modality`` key) keep loading and default to
     ``"text"``. This protects every existing entry in ``aliases.json``
     from a silent routing flip when the field landed.
  2. The accepted modality set is exactly the documented Literal.
     Drift between the Literal and the loader's allow-list has shipped
     silently in this repo before (see ``suffix_decoding_tier`` pre-#283).
  3. Non-text modalities cannot carry AR-only capability gates
     (``supports_spec_decode`` / ``supports_dflash``). Fail loud at
     load instead of misroute at request time.
  4. The diffusion-lane skeleton imports cleanly. The module is
     intentionally not wired into any active code path yet — but it
     must not break ``vllm_mlx`` import.
"""

from __future__ import annotations

import pytest

from vllm_mlx.model_aliases import _VALID_MODALITIES, AliasProfile, _coerce


class TestModalityDefault:
    def test_legacy_string_form_defaults_to_text(self) -> None:
        profile = _coerce("legacy-string", "mlx-community/Qwen3.5-4B-MLX-4bit")
        assert profile.modality == "text"

    def test_dict_without_modality_defaults_to_text(self) -> None:
        profile = _coerce(
            "legacy-dict",
            {"hf_path": "mlx-community/Qwen3.5-4B-MLX-4bit"},
        )
        assert profile.modality == "text"

    def test_dict_with_explicit_text_modality(self) -> None:
        profile = _coerce(
            "explicit-text",
            {"hf_path": "x/y", "modality": "text"},
        )
        assert profile.modality == "text"

    def test_text_diffusion_accepted(self) -> None:
        profile = _coerce(
            "diffusion-gemma-26b",
            {
                "hf_path": "mlx-community/diffusiongemma-26B-A4B-it-4bit",
                "modality": "text-diffusion",
                "is_hybrid": True,
                "is_moe": True,
                "supports_spec_decode": False,
                "supports_dflash": False,
            },
        )
        assert profile.modality == "text-diffusion"
        assert profile.supports_spec_decode is False


class TestModalityValidation:
    def test_unknown_modality_rejected(self) -> None:
        with pytest.raises(ValueError, match="modality must be one of"):
            _coerce(
                "bad",
                {"hf_path": "x/y", "modality": "video"},
            )

    def test_non_string_modality_rejected(self) -> None:
        with pytest.raises(ValueError, match="modality must be one of"):
            _coerce(
                "bad",
                {"hf_path": "x/y", "modality": 1},
            )

    def test_valid_modality_set_pinned(self) -> None:
        # If you add a value here you MUST also update the Literal in
        # model_aliases.py AND the dispatch tables in cli.py /
        # routes/models.py. Failing this assertion is the trigger to
        # do that work.
        assert (
            frozenset({"text", "text-diffusion", "vision", "image-gen"})
            == _VALID_MODALITIES
        )


class TestNonTextLaneRejectsARGates:
    def test_text_diffusion_with_spec_decode_rejected(self) -> None:
        with pytest.raises(ValueError, match="supports_spec_decode must be false"):
            _coerce(
                "bad",
                {
                    "hf_path": "x/y",
                    "modality": "text-diffusion",
                    # supports_spec_decode defaults to True — that's
                    # the trap this guard catches.
                },
            )

    def test_text_diffusion_with_dflash_rejected(self) -> None:
        with pytest.raises(ValueError, match="supports_dflash must be false"):
            _coerce(
                "bad",
                {
                    "hf_path": "x/y",
                    "modality": "text-diffusion",
                    "supports_spec_decode": False,
                    "supports_dflash": True,
                    "dflash_draft_model": "z-lab/whatever",
                },
            )


class TestDiffusionLaneSkeleton:
    def test_module_importable(self) -> None:
        # The whole point of the skeleton: it must import cleanly so
        # alias loaders that branch on modality have a stable symbol
        # to reach for. The bodies stay NotImplementedError until
        # mlx-vlm >= 0.6.3 lands.
        from vllm_mlx.runtime import diffusion_lane

        assert diffusion_lane.DIFFUSION_LANE_VERSION == "0.0-skeleton"
        assert hasattr(diffusion_lane, "DiffusionRunner")
        assert hasattr(diffusion_lane, "load_runner")

    def test_load_runner_raises_with_remediation_hint(self) -> None:
        from vllm_mlx.runtime.diffusion_lane import load_runner

        with pytest.raises(NotImplementedError, match="mlx-vlm >= 0.6.3"):
            load_runner("mlx-community/diffusiongemma-26B-A4B-it-4bit")

    def test_runner_generate_raises_until_wired(self) -> None:
        from vllm_mlx.runtime.diffusion_lane import (
            DiffusionGenerationConfig,
            DiffusionRunner,
        )

        runner = DiffusionRunner(model=object(), tokenizer=object(), hf_path="x/y")
        with pytest.raises(NotImplementedError, match="mlx-vlm >= 0.6.3"):
            runner.generate([1, 2, 3], DiffusionGenerationConfig())


class TestAliasProfileDataclassShape:
    def test_default_modality_when_constructed_directly(self) -> None:
        # Catches the case where a future refactor flips the default
        # in the dataclass but forgets to update the loader. The
        # contract is: AliasProfile(hf_path="x") is a text-lane LLM.
        profile = AliasProfile(hf_path="x/y")
        assert profile.modality == "text"
