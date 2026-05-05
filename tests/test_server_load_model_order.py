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

import pytest


class _StubEngine:
    """Minimal stand-in for `BatchedEngine` — only the surface `load_model`
    actually accesses between construction and the model-registry add.

    This is intentionally explicit (not `MagicMock`) so that any future
    `load_model` change touching a new attribute fails LOUDLY with
    `AttributeError`, not silently with a fabricated MagicMock value.
    """

    is_mllm = False
    preserve_native_tool_format = False
    _tokenizer = None
    _tool_logits_processor_factory = None

    def __init__(self, *args, **kwargs):
        # Accept positional too in case `BatchedEngine.__init__` ever takes any.
        self.args = args
        self.kwargs = kwargs


@pytest.fixture(autouse=True)
def _reset_cfg_around_each_test():
    """Reset the ServerConfig singleton before AND after every test.

    `monkeypatch.setattr` on module globals is restored automatically, but
    the cfg singleton is a separate process-level object that must be
    explicitly reset on both sides — otherwise a mid-test failure leaks
    cfg state into the next test.
    """
    from vllm_mlx.config import reset_config

    reset_config()
    yield
    reset_config()


def test_load_model_enables_native_tool_format_when_parser_supports_it(monkeypatch):
    """After load_model() returns, the engine MUST reflect the parser's
    native-format support. Pre-fix this asserted False because cfg was
    unsynced when detection ran.
    """
    from vllm_mlx import server

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
    from vllm_mlx.config import get_config

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


def test_sync_config_is_idempotent(monkeypatch):
    """`_sync_config()` is called twice in `load_model` (early before native
    tool detection, late after the model registry add). Both calls must
    leave cfg in the same state — if the function ever grows non-idempotent
    side effects (counter increments, callback fires, cache invalidations),
    the late re-sync becomes a latent bug.
    """
    from vllm_mlx import server
    from vllm_mlx.config import get_config

    monkeypatch.setattr(server, "_enable_auto_tool_choice", True, raising=False)
    monkeypatch.setattr(server, "_tool_call_parser", "hermes", raising=False)
    monkeypatch.setattr(server, "_reasoning_parser", None, raising=False)
    monkeypatch.setattr(server, "_reasoning_parser_name", None, raising=False)
    monkeypatch.setattr(server, "_tool_parser_instance", None, raising=False)
    monkeypatch.setattr(server, "_mcp_manager", None, raising=False)
    monkeypatch.setattr(server, "_enable_tool_logits_bias", False, raising=False)
    monkeypatch.setattr(server, "_engine", None, raising=False)

    server._sync_config()
    cfg = get_config()
    snapshot = {
        "engine": cfg.engine,
        "model_name": cfg.model_name,
        "model_alias": cfg.model_alias,
        "model_path": cfg.model_path,
        "enable_auto_tool_choice": cfg.enable_auto_tool_choice,
        "tool_call_parser": cfg.tool_call_parser,
        "tool_parser_instance": cfg.tool_parser_instance,
        "enable_tool_logits_bias": cfg.enable_tool_logits_bias,
        "reasoning_parser": cfg.reasoning_parser,
        "reasoning_parser_name": cfg.reasoning_parser_name,
        "mcp_manager": cfg.mcp_manager,
        "model_registry": cfg.model_registry,
    }

    server._sync_config()
    cfg2 = get_config()

    for k, v in snapshot.items():
        assert getattr(cfg2, k) == v, f"_sync_config() not idempotent on cfg.{k}"
