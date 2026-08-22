"""Architecture-aware prefill chunk defaults for recurrent language models."""

from __future__ import annotations

from types import SimpleNamespace

import vllm_mlx.cli as cli


def _profile(*, hybrid: bool, explicit: bool = False):
    return SimpleNamespace(
        is_hybrid=hybrid,
        is_hybrid_explicit=explicit,
        hf_path="unused",
    )


def test_linear_attention_config_is_detected_at_language_backbone():
    assert cli._config_declares_linear_attention(
        {
            "text_config": {"layer_types": ["linear_attention", "full_attention"]},
            "layer_types": ["full_attention"],
        }
    )


def test_dense_and_sliding_attention_are_not_recurrent():
    assert not cli._config_declares_linear_attention(
        {"layer_types": ["sliding_attention", "full_attention"]}
    )
    assert not cli._config_declares_linear_attention(None)


def test_mamba_and_recurrent_checkpoint_markers_are_detected():
    assert cli._config_declares_linear_attention(
        {"layer_types": ["mamba", "full_attention"]}
    )
    assert cli._config_declares_linear_attention(
        {"text_config": {"model_type": "recurrent_gemma"}}
    )
    assert cli._config_declares_linear_attention({"model_type": "qwen3_next"})


def test_recurrent_alias_auto_defaults_to_512(monkeypatch):
    monkeypatch.setattr(
        "vllm_mlx.model_aliases.resolve_profile",
        lambda _name: _profile(hybrid=False, explicit=True),
    )
    assert (
        cli._resolve_prefill_step_size(
            model_name="qwen3.5-4b-4bit", configured=2048, user_set_explicit=False
        )
        == cli._DEFAULT_RECURRENT_PREFILL_STEP_SIZE
    )


def test_explicit_prefill_step_size_always_wins(monkeypatch):
    monkeypatch.setattr(
        "vllm_mlx.model_aliases.resolve_profile",
        lambda _name: _profile(hybrid=True),
    )
    assert (
        cli._resolve_prefill_step_size(
            model_name="hybrid", configured=2048, user_set_explicit=True
        )
        == 2048
    )


def test_gemma_sliding_window_keeps_dense_default(monkeypatch):
    monkeypatch.setattr(
        "vllm_mlx.model_aliases.resolve_profile",
        lambda _name: _profile(hybrid=False),
    )
    monkeypatch.setattr(
        cli,
        "_resolve_checkpoint_config",
        lambda _name, _profile: {
            "layer_types": ["sliding_attention", "full_attention"]
        },
    )
    assert (
        cli._resolve_prefill_step_size(
            model_name="gemma-4-12b", configured=2048, user_set_explicit=False
        )
        == 2048
    )


def test_bare_linear_attention_checkpoint_is_detected(monkeypatch):
    monkeypatch.setattr("vllm_mlx.model_aliases.resolve_profile", lambda _name: None)
    monkeypatch.setattr(
        cli,
        "_resolve_checkpoint_config",
        lambda _name, _profile: {"layer_types": ["linear_attention"]},
    )
    assert cli._prefers_recurrent_prefill_chunks("/models/recurrent")
