# SPDX-License-Identifier: Apache-2.0
"""Contracts for profile-driven, load-time chat-template resolution."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

from vllm_mlx.model_aliases import list_profiles, resolve_profile
from vllm_mlx.utils.chat_template import apply_chat_template
from vllm_mlx.utils.chat_template_registry import (
    bundled_chat_template,
    resolve_chat_template,
    resolve_profile_chat_template,
)

_CHECKPOINT_TEMPLATE = "checkpoint {{ messages }}"


def _tokenizer(template: str = _CHECKPOINT_TEMPLATE) -> PreTrainedTokenizerFast:
    backend = Tokenizer(models.WordLevel({"<unk>": 0}, unk_token="<unk>"))
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        bos_token="<bos>",
        eos_token="<eos>",
    )
    tokenizer.chat_template = template
    return tokenizer


def _tools() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Execute a command",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "limit": {"type": ["integer", "null"]},
                    },
                    "required": ["command"],
                },
            },
        }
    ]


@pytest.mark.parametrize(
    ("alias", "template_id"),
    [
        ("gemma-4-e2b-4bit", "gemma4_compact"),
        ("gemma-4-e4b-optiq-4bit", "gemma4_compact"),
        ("gemma-4-12b-4bit", "gemma4_full"),
        ("gemma-4-26b-assistant", "gemma4_full"),
        ("gemma-4-31b-qat-8bit", "gemma4_full"),
    ],
)
def test_profile_declares_template_for_alias_and_exact_repo_path(
    alias: str, template_id: str
) -> None:
    profile = resolve_profile(alias)
    assert profile is not None
    assert profile.chat_template_id == template_id
    assert resolve_profile(profile.hf_path) == profile


def test_every_gemma4_parser_profile_declares_a_template() -> None:
    for alias, profile in list_profiles().items():
        if profile.modality == "text" and profile.tool_call_parser == "gemma4":
            assert profile.chat_template_id in {"gemma4_compact", "gemma4_full"}, alias


def test_declared_profile_replaces_checkpoint_template_without_hash_or_name_probe() -> (
    None
):
    tokenizer = _tokenizer("re-exported\n  checkpoint {{ messages }}\n")

    assert resolve_profile_chat_template(tokenizer, "gemma-4-e2b-4bit")
    assert tokenizer.chat_template == bundled_chat_template("gemma4_compact")
    assert not resolve_profile_chat_template(tokenizer, "gemma-4-e2b-4bit")


@pytest.mark.parametrize(
    "explicit",
    [
        "explicit {{ messages }}",
        {
            "default": "explicit default {{ messages }}",
            "tool_use": "explicit tools {{ messages }}",
        },
        [
            {"name": "default", "template": "explicit default {{ messages }}"},
            {"name": "tool_use", "template": "explicit tools {{ messages }}"},
        ],
    ],
)
def test_unknown_checkpoint_and_explicit_override_are_preserved(explicit) -> None:
    unknown = _tokenizer("custom {{ messages }}")
    declared = _tokenizer()

    assert not resolve_profile_chat_template(unknown, "owner/private-checkpoint")
    assert unknown.chat_template == "custom {{ messages }}"
    assert resolve_chat_template(declared, "gemma4_full", explicit_template=explicit)
    assert declared.chat_template == explicit


def test_registry_handles_nested_processor_and_missing_selection() -> None:
    tokenizer = _tokenizer()
    processor = SimpleNamespace(tokenizer=tokenizer)

    assert resolve_chat_template(processor, "gemma4_compact")
    assert tokenizer.chat_template == bundled_chat_template("gemma4_compact")
    assert not resolve_chat_template(SimpleNamespace(), "gemma4_full")
    assert not resolve_chat_template(_tokenizer(), None)


def test_unknown_registry_id_fails_closed() -> None:
    with pytest.raises(ValueError, match="unknown chat template ID"):
        bundled_chat_template("unregistered")


@pytest.mark.parametrize("invalid", [391, "unknown-template"])
def test_alias_schema_rejects_invalid_template_id(invalid) -> None:
    from vllm_mlx.model_aliases import _coerce

    expected = "must be a string" if not isinstance(invalid, str) else "not in"
    with pytest.raises(ValueError, match=expected):
        _coerce(
            "invalid-template",
            {"hf_path": "owner/checkpoint", "chat_template_id": invalid},
        )


def _stub_text_loader(monkeypatch, tokenizer):
    from vllm_mlx.utils import tokenizer as tokenizer_module

    model = object()
    monkeypatch.setattr(
        tokenizer_module, "_resolve_subfolder_checkpoint", lambda name: name
    )
    monkeypatch.setattr(
        tokenizer_module, "_local_snapshot_if_cached", lambda name: name
    )
    monkeypatch.setattr(tokenizer_module, "_resolve_model_path", lambda name: None)
    monkeypatch.setattr(
        tokenizer_module, "validate_local_model_file", lambda name: None
    )
    monkeypatch.setattr(
        tokenizer_module,
        "apply_remote_code_policy",
        lambda config: (config or {}, False),
    )
    monkeypatch.setattr(
        tokenizer_module, "_model_requires_remote_code", lambda name: False
    )
    monkeypatch.setattr(
        tokenizer_module,
        "_load_model_with_fallback_impl",
        lambda name, config: (model, tokenizer),
    )
    monkeypatch.setattr(tokenizer_module, "_post_load_ubc_evict", lambda name: None)
    return tokenizer_module, model


def test_text_loader_resolves_profile_template_once(monkeypatch) -> None:
    tokenizer = _tokenizer()
    tokenizer_module, model = _stub_text_loader(monkeypatch, tokenizer)

    loaded_model, loaded_tokenizer = tokenizer_module.load_model_with_fallback(
        "gemma-4-26b-4bit"
    )

    assert loaded_model is model
    assert loaded_tokenizer is tokenizer
    assert tokenizer.chat_template == bundled_chat_template("gemma4_full")


def test_lazy_text_loader_resolves_profile_template_once(monkeypatch) -> None:
    import mlx_lm

    tokenizer = _tokenizer()
    tokenizer_module, model = _stub_text_loader(monkeypatch, tokenizer)
    monkeypatch.setattr(
        tokenizer_module,
        "_neutralize_unbundled_template_types",
        lambda name, config: config,
    )
    monkeypatch.setattr(tokenizer_module, "_try_inject_mtp_post_load", lambda *a: None)
    monkeypatch.setattr(
        tokenizer_module,
        "augment_eos_token_ids_from_generation_config",
        lambda *a: None,
    )
    monkeypatch.setattr(tokenizer_module, "repair_byte_level_decoder", lambda *a: None)
    monkeypatch.setattr(mlx_lm, "load", lambda *a, **kw: (model, tokenizer))

    loaded_model, loaded_tokenizer = tokenizer_module.load_model_with_fallback(
        "gemma-4-e2b-4bit", lazy=True
    )

    assert loaded_model is model
    assert loaded_tokenizer is tokenizer
    assert tokenizer.chat_template == bundled_chat_template("gemma4_compact")


@pytest.mark.asyncio
async def test_mllm_start_resolves_profile_template_at_processor_load(
    monkeypatch,
) -> None:
    from vllm_mlx import engine_core, mllm_scheduler
    from vllm_mlx.engine import batched as batched_module
    from vllm_mlx.models import mllm as mllm_module

    processor = _tokenizer()

    class FakeMultimodalLM:
        def __init__(self, model_name, trust_remote_code=True):
            self.model = SimpleNamespace()
            self.processor = processor

        def load(self) -> None:
            pass

    class FakeScheduler:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def start(self) -> None:
            pass

    monkeypatch.setattr(engine_core, "_init_mlx_step_thread", lambda: None)
    monkeypatch.setattr(mllm_module, "MLXMultimodalLM", FakeMultimodalLM)
    monkeypatch.setattr(mllm_scheduler, "MLLMScheduler", FakeScheduler)
    monkeypatch.setattr(batched_module, "_probe_mllm_cache_type", lambda model: None)

    engine = object.__new__(batched_module.BatchedEngine)
    engine._model_name = "gemma-4-e2b-4bit"
    engine._trust_remote_code = False
    engine._force_mllm = False
    engine._scheduler_config = None
    engine._model_load_executor = None

    try:
        await engine._start_mllm()
    finally:
        if engine._model_load_executor is not None:
            engine._model_load_executor.shutdown(wait=True)

    assert processor.chat_template == bundled_chat_template("gemma4_compact")


@pytest.mark.parametrize(
    "explicit",
    [
        {
            "default": "explicit default {{ messages }}",
            "tool_use": "explicit tools {{ messages }}",
        },
        [
            {"name": "default", "template": "explicit default {{ messages }}"},
            {"name": "tool_use", "template": "explicit tools {{ messages }}"},
        ],
    ],
)
def test_text_loader_preserves_named_explicit_template_overrides(
    monkeypatch, explicit
) -> None:
    tokenizer = _tokenizer()
    tokenizer_module, _ = _stub_text_loader(monkeypatch, tokenizer)

    _, loaded_tokenizer = tokenizer_module.load_model_with_fallback(
        "gemma-4-26b-4bit",
        tokenizer_config={"chat_template": explicit},
    )

    assert loaded_tokenizer.chat_template == explicit


def test_renderer_does_not_resolve_or_mutate_templates() -> None:
    source = Path("vllm_mlx/utils/chat_template.py").read_text()
    assert "resolve_chat_template" not in source
    assert "upgrade_stale_gemma4_chat_template" not in source


def test_resolved_template_renders_null_and_normalized_openai_arguments() -> None:
    pytest.importorskip("jinja2")
    tokenizer = _tokenizer()
    resolve_chat_template(tokenizer, "gemma4_full")
    messages = [
        {"role": "user", "content": "List /tmp"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": '{"command":"ls /tmp","limit":null}',
                    },
                }
            ],
        },
    ]

    rendered = apply_chat_template(
        tokenizer,
        messages,
        tools=_tools(),
        enable_thinking=False,
        model_name="opaque-model-id",
        add_generation_prompt=False,
    )

    assert 'command:<|"|>ls /tmp<|"|>' in rendered
    assert "limit:null" in rendered
    assert 'type:[<|"|>INTEGER<|"|>,<|"|>NULL<|"|>]' in rendered
    assert "limit:None" not in rendered


def test_resolved_template_restores_thinking_continuation_after_tool_result() -> None:
    pytest.importorskip("jinja2")
    tokenizer = _tokenizer()
    resolve_chat_template(tokenizer, "gemma4_full")
    messages = [
        {"role": "user", "content": "List /tmp"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": {"command": "ls /tmp"},
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "alpha.txt"},
    ]

    rendered = apply_chat_template(
        tokenizer,
        messages,
        tools=_tools(),
        enable_thinking=True,
        model_name="opaque-model-id",
    )

    assert rendered.endswith(
        '<|tool_response>response:bash{value:<|"|>alpha.txt<|"|>}'
        "<tool_response|><|channel>thought\n"
    )
