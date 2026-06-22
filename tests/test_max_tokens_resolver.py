# SPDX-License-Identifier: Apache-2.0
"""Regression tests for max_tokens cap resolution.

These tests do not load a model. They pin the shared resolver used by
chat, responses, Anthropic, and completions routes.
"""

from __future__ import annotations

import ast
import inspect


def _thinking_cfg(*, default_max_tokens_is_explicit: bool):
    from vllm_mlx.config import reset_config

    cfg = reset_config()
    cfg.default_max_tokens = 128
    cfg.default_max_tokens_is_explicit = default_max_tokens_is_explicit
    cfg.thinking_token_budget = 2048
    cfg.reasoning_parser_name = "qwen3"
    return cfg


def test_request_explicit_max_tokens_is_hard_cap_for_thinking_model():
    from vllm_mlx.service.helpers import _resolve_max_tokens

    _thinking_cfg(default_max_tokens_is_explicit=False)

    assert _resolve_max_tokens(64, enable_thinking=True) == 64


def test_operator_explicit_default_max_tokens_is_hard_cap_for_thinking_model():
    from vllm_mlx.service.helpers import _resolve_max_tokens

    _thinking_cfg(default_max_tokens_is_explicit=True)

    assert _resolve_max_tokens(None, enable_thinking=True) == 128


def test_implicit_default_gets_thinking_headroom_when_request_omits_max_tokens():
    from vllm_mlx.service.helpers import _resolve_max_tokens

    _thinking_cfg(default_max_tokens_is_explicit=False)

    assert _resolve_max_tokens(None, enable_thinking=True) == 128 + 2048


def test_non_thinking_request_does_not_get_implicit_headroom():
    from vllm_mlx.service.helpers import _resolve_max_tokens

    _thinking_cfg(default_max_tokens_is_explicit=False)

    assert _resolve_max_tokens(None, enable_thinking=False) == 128


def test_all_text_generation_routes_use_shared_max_tokens_resolver():
    from vllm_mlx.routes import anthropic, chat, completions, responses

    for module in (anthropic, chat, completions, responses):
        tree = ast.parse(inspect.getsource(module))
        calls_resolver = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_resolve_max_tokens"
            for node in ast.walk(tree)
        )
        assert calls_resolver, f"{module.__name__} does not call _resolve_max_tokens"
