# SPDX-License-Identifier: Apache-2.0
"""E4B is a text-only Gemma artifact, independent of the Bolt hybrid family."""

import pytest

from rapid_mlx import model_sizes
from rapid_mlx.model_aliases import resolve_profile


@pytest.mark.parametrize(
    "alias,bits,revision,size",
    [
        (
            "nemotron-3.5-bolt-e4b",
            4,
            "4274caa99dbacb12cbbca9bd9595a5063a33880f",
            4_231_241_505,
        ),
        (
            "nemotron-3.5-bolt-e4b-8bit",
            8,
            "210cb1bbd2107dc871cb3a4a23155d914077be94",
            7_962_488_801,
        ),
    ],
)
def test_e4b_serving_contract(alias, bits, revision, size):
    profile = resolve_profile(alias)
    assert profile is not None
    assert profile.hf_path == (
        f"lmstudio-community/NVIDIA-Nemotron-3.5-Bolt-E4B-MLX-{bits}bit"
    )
    assert profile.revision == revision
    assert profile.is_text_only is True
    assert profile.supports_image_input is False
    assert profile.tool_call_parser == "gemma4"
    assert profile.reasoning_parser == "gemma4"
    assert profile.chat_template_id is None
    assert profile.is_hybrid is False
    assert profile.is_moe is False
    assert profile.supports_spec_decode is False
    assert profile.supports_dflash is False
    assert profile.dflash_draft_model is None
    assert dict(profile.recommended_sampling) == {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 64,
    }
    assert model_sizes.size_bytes(profile.hf_path) == size


@pytest.mark.parametrize(
    "model,gemma,expected",
    [
        ("lmstudio-community/NVIDIA-Nemotron-3.5-Bolt-E4B-MLX-4bit", True, False),
        ("nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16", True, False),
        ("rapid-mlx/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-MLX-4bit", False, True),
        ("other/model", False, False),
    ],
)
@pytest.mark.requires_mlx
def test_architecture_takes_precedence_over_publisher_name(
    monkeypatch, model, gemma, expected
):
    from rapid_mlx.models import gemma4_text
    from rapid_mlx.utils.tokenizer import _needs_tokenizer_fallback

    calls = []

    def detect(name):
        calls.append(name)
        return gemma

    monkeypatch.setattr(gemma4_text, "is_gemma4_family_model", detect)
    assert _needs_tokenizer_fallback(model) is expected
    assert calls == ([] if model == "other/model" else [model])


@pytest.mark.parametrize(
    "model",
    [
        "nemotron-3.5-bolt-e4b",
        "nemotron-3.5-bolt-e4b-8bit",
        "nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16",
        "lmstudio-community/NVIDIA-Nemotron-3.5-Bolt-E4B-MLX-4bit",
        "lmstudio-community/NVIDIA-Nemotron-3.5-Bolt-E4B-MLX-8bit",
    ],
)
@pytest.mark.requires_mlx
def test_e4b_real_config_detection(tmp_path, monkeypatch, model):
    import json

    import huggingface_hub

    from rapid_mlx.models.gemma4_text import (
        gemma4_load_plan,
        is_gemma4_family_model,
    )
    from rapid_mlx.utils.tokenizer import _needs_tokenizer_fallback

    # Routing-relevant fields from the pinned E4B source and both quants.
    # Stub only Hub transport: retain the real revision-enforcing download
    # wrapper as well as JSON interpretation and architecture classification.
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "architectures": ["Gemma4ForConditionalGeneration"],
                "model_type": "gemma4",
                "text_config": {
                    "model_type": "gemma4_text",
                    "num_hidden_layers": 42,
                    "num_kv_shared_layers": 18,
                },
            }
        )
    )

    profile = resolve_profile(model)

    def download(repo_id, filename, **kwargs):
        assert repo_id == (profile.hf_path if profile else model)
        assert filename == "config.json"
        assert kwargs.get("revision") == (profile.revision if profile else None)
        return str(config)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    assert is_gemma4_family_model(model) is True
    assert gemma4_load_plan(model) == ("nonunified", True)
    assert _needs_tokenizer_fallback(model) is False
    # The same architecture must work after revision resolution to a snapshot.
    assert is_gemma4_family_model(tmp_path) is True
    assert _needs_tokenizer_fallback(str(tmp_path)) is False
