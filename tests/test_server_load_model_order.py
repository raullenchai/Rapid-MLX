# SPDX-License-Identifier: Apache-2.0
"""Regression for #225 — startup ordering.

`_detect_native_tool_support()` reads `cfg.enable_auto_tool_choice` and
`cfg.tool_call_parser` via `get_config()`. If `_sync_config()` runs
*after* the detection call (the pre-fix layout), those fields are still
at their dataclass defaults (False, None), the guard short-circuits to
False, and `_engine.preserve_native_tool_format` is silently set to
False even though the configured parser supports native format.

Downstream symptom (per the bug report on Qwen3.5-9B-4bit and
Qwen3.6-35B-A3B-4bit-DWQ): assistant tool history gets serialised by
`api/utils.py::process_messages` as
`[Calling tool: name({json})]` text. The model sees prose-format
examples in context and mimics that pattern on subsequent turns —
streaming chunks emit the literal string instead of structured
`tool_calls`. Looks like a model failure but is a startup ordering
bug.
"""

from __future__ import annotations


class _StubEngine:
    is_mllm = False
    preserve_native_tool_format = False
    _tokenizer = None
    _tool_logits_processor_factory = None

    def __init__(self, **kwargs):  # noqa: D401 — match BatchedEngine signature loosely
        self.kwargs = kwargs


def test_load_model_enables_native_tool_format_when_parser_supports_it(monkeypatch):
    """After load_model() returns, the engine MUST reflect the parser's
    native-format support. Pre-fix this asserted False because cfg was
    unsynced when detection ran.
    """
    from vllm_mlx import server
    from vllm_mlx.config import reset_config

    reset_config()

    monkeypatch.setattr(server, "BatchedEngine", _StubEngine)
    monkeypatch.setattr(server, "_engine", None, raising=False)
    monkeypatch.setattr(server, "_enable_auto_tool_choice", True, raising=False)
    monkeypatch.setattr(server, "_tool_call_parser", "hermes", raising=False)
    monkeypatch.setattr(server, "_reasoning_parser_name", None, raising=False)
    monkeypatch.setattr(server, "_reasoning_parser", None, raising=False)
    monkeypatch.setattr(server, "_tool_parser_instance", None, raising=False)
    monkeypatch.setattr(server, "_mcp_manager", None, raising=False)
    monkeypatch.setattr(server, "_enable_tool_logits_bias", False, raising=False)
    monkeypatch.setattr(server, "_model_alias", None, raising=False)

    server.load_model("mlx-community/Qwen3.5-9B-4bit")

    assert server._engine is not None
    # hermes parser sets SUPPORTS_NATIVE_TOOL_FORMAT = True; with the
    # ordering fix, detection sees the synced cfg and propagates that
    # to the engine.
    assert server._engine.preserve_native_tool_format is True


def test_detect_native_tool_support_requires_synced_config(monkeypatch):
    """Contract test for the ordering invariant: detection short-circuits
    to False when cfg has not been synced yet, so callers MUST run
    `_sync_config()` first.
    """
    from vllm_mlx import server
    from vllm_mlx.config import get_config, reset_config

    reset_config()
    monkeypatch.setattr(server, "_enable_auto_tool_choice", True, raising=False)
    monkeypatch.setattr(server, "_tool_call_parser", "hermes", raising=False)
    monkeypatch.setattr(server, "_reasoning_parser", None, raising=False)
    monkeypatch.setattr(server, "_reasoning_parser_name", None, raising=False)
    monkeypatch.setattr(server, "_tool_parser_instance", None, raising=False)
    monkeypatch.setattr(server, "_mcp_manager", None, raising=False)
    monkeypatch.setattr(server, "_enable_tool_logits_bias", False, raising=False)
    monkeypatch.setattr(server, "_engine", None, raising=False)

    cfg = get_config()
    assert cfg.enable_auto_tool_choice is False
    assert cfg.tool_call_parser is None
    assert server._detect_native_tool_support() is False

    server._sync_config()

    cfg = get_config()
    assert cfg.enable_auto_tool_choice is True
    assert cfg.tool_call_parser == "hermes"
    assert server._detect_native_tool_support() is True
