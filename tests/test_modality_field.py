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

from vllm_mlx.model_aliases import (
    _RESERVED_MODALITIES,
    _VALID_MODALITIES,
    AliasProfile,
    _coerce,
)


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
            "diffusion-gemma-26b-4bit",
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
        # Implemented lanes — these have working dispatch in
        # ``load_model``. If you add a value here you MUST also
        # update the Literal in model_aliases.py AND the dispatch
        # tables in cli.py / routes/models.py. Failing this assertion
        # is the trigger to do that work.
        assert frozenset({"text", "text-diffusion"}) == _VALID_MODALITIES

    def test_reserved_modality_set_pinned(self) -> None:
        # Reserved lanes — declared in the type alias so routing
        # code can pattern-match once the engine lands, but loading
        # an alias that declares one MUST fail loud right now
        # (pr_validate codex r13 NIT). When you implement one of
        # these, move it from _RESERVED_MODALITIES into
        # _VALID_MODALITIES and update this test.
        assert frozenset({"vision", "image-gen"}) == _RESERVED_MODALITIES

    def test_reserved_modality_rejected_at_load(self) -> None:
        # Loading an alias whose modality is reserved-but-not-routed
        # must fail with a clear "not yet implemented" message.
        for reserved in ("vision", "image-gen"):
            with pytest.raises(ValueError, match="not yet implemented"):
                _coerce(
                    "bad",
                    {
                        "hf_path": "x/y",
                        "modality": reserved,
                        "supports_spec_decode": False,
                        "supports_dflash": False,
                    },
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


class TestDiffusionLaneWired:
    def test_module_importable(self) -> None:
        # The module exposes a working DiffusionEngine that subclasses
        # BaseEngine. The skeleton aliases (DiffusionRunner /
        # load_runner) remain as backward-compat shims so any external
        # caller carried over from PR #551's draft surface still works.
        from vllm_mlx.runtime import diffusion_lane

        assert diffusion_lane.DIFFUSION_LANE_VERSION == "0.1-wired"
        assert hasattr(diffusion_lane, "DiffusionEngine")
        assert hasattr(diffusion_lane, "DiffusionRunner")
        assert hasattr(diffusion_lane, "load_runner")

    def test_engine_inherits_base_engine(self) -> None:
        from vllm_mlx.engine.base import BaseEngine
        from vllm_mlx.runtime.diffusion_lane import DiffusionEngine

        assert issubclass(DiffusionEngine, BaseEngine)

    def test_engine_unloaded_method_calls_raise(self) -> None:
        # Defensive: every public method that needs the loaded model
        # must call _ensure_loaded() so a misconfigured server (the
        # engine was instantiated but start() was never awaited)
        # surfaces a clear error.
        from vllm_mlx.runtime.diffusion_lane import DiffusionEngine

        engine = DiffusionEngine(model_name="mlx-community/whatever")
        with pytest.raises(RuntimeError, match="not loaded"):
            _ = engine.tokenizer
        with pytest.raises(RuntimeError, match="not loaded"):
            engine.build_prompt([{"role": "user", "content": "hi"}])
        with pytest.raises(RuntimeError, match="not loaded"):
            engine.estimate_new_tokens("hi")

    def test_runner_alias_points_at_engine(self) -> None:
        # PR #551 shipped a ``DiffusionRunner`` symbol; we kept it as
        # an alias so the draft branch's tests still pass once it
        # rebases on this work.
        from vllm_mlx.runtime.diffusion_lane import DiffusionEngine, DiffusionRunner

        assert DiffusionRunner is DiffusionEngine


class TestAliasProfileDataclassShape:
    def test_default_modality_when_constructed_directly(self) -> None:
        # Catches the case where a future refactor flips the default
        # in the dataclass but forgets to update the loader. The
        # contract is: AliasProfile(hf_path="x") is a text-lane LLM.
        profile = AliasProfile(hf_path="x/y")
        assert profile.modality == "text"


class TestHfPathReverseLookupRoutesDiffusionLane:
    """pr_validate r5 codex BLOCKING #1 claimed that ``python -m
    vllm_mlx.server --model <hf-path>`` would route the diffusion
    checkpoint into the AR ``BatchedEngine`` because ``_profile is
    None`` for the raw HF path. That claim is FALSE — ``resolve_profile``
    consults the ``_hf_to_alias`` reverse index (model_aliases.py:400)
    so an HF path that matches a registered alias resolves to the
    same profile as the alias name. This test pins that safety net
    so a future refactor that drops the reverse index would NOT
    silently regress the modality dispatch.
    """

    def test_diffusion_4bit_hf_path_resolves_to_text_diffusion_modality(self) -> None:
        from vllm_mlx.model_aliases import resolve_profile

        diffusion_alias_profile = resolve_profile("diffusion-gemma-26b-4bit")
        assert diffusion_alias_profile is not None
        assert diffusion_alias_profile.hf_path == (
            "mlx-community/diffusiongemma-26B-A4B-it-4bit"
        )
        profile = resolve_profile(diffusion_alias_profile.hf_path)
        assert profile is not None, (
            "resolve_profile must reverse-look an HF path that matches "
            "a registered alias — codex r5 BLOCKING #1 false-positive "
            "would have routed this to BatchedEngine"
        )
        assert profile.modality == "text-diffusion"
        assert profile.tool_call_parser == "gemma4"

    def test_diffusion_8bit_hf_path_resolves_to_text_diffusion_modality(self) -> None:
        from vllm_mlx.model_aliases import resolve_profile

        diffusion_alias_profile = resolve_profile("diffusion-gemma-26b-8bit")
        assert diffusion_alias_profile is not None
        assert diffusion_alias_profile.hf_path == (
            "mlx-community/diffusiongemma-26B-A4B-it-8bit"
        )
        profile = resolve_profile(diffusion_alias_profile.hf_path)
        assert profile is not None
        assert profile.modality == "text-diffusion"
        assert profile.tool_call_parser == "gemma4"

    def test_diffusion_bare_alias_is_unregistered(self) -> None:
        """The unqualified ``diffusion-gemma-26b`` alias was removed in
        PR #558 — every diffusion alias must carry an explicit
        ``-Nbit`` quantization suffix to match the rest of
        ``aliases.json`` (see qwen3.5/qwen3.6/deepseek-v4-flash). Pin
        the removal so a future maintainer who adds it back has to
        either bring a written-down reason or delete this test
        deliberately.
        """
        from vllm_mlx.model_aliases import resolve_profile

        assert resolve_profile("diffusion-gemma-26b") is None

    def test_unregistered_hf_path_falls_through_to_none(self) -> None:
        # Sanity: HF paths that AREN'T in aliases.json still return
        # None, which makes server.py default to the text lane —
        # that's the documented behavior for unknown models.
        from vllm_mlx.model_aliases import resolve_profile

        profile = resolve_profile("nobody/this-model-does-not-exist-123")
        assert profile is None
