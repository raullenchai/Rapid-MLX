"""No-final-content watchdog regressions for thinking routes."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace


def _load_thinking_processor():
    class _FakeArray:
        pass

    fake_mx = types.SimpleNamespace(array=_FakeArray)
    saved_mlx = sys.modules.get("mlx")
    saved_mlx_core = sys.modules.get("mlx.core")
    sys.modules.setdefault("mlx", types.ModuleType("mlx"))
    sys.modules["mlx.core"] = fake_mx

    try:
        path = (
            Path(__file__).resolve().parents[1]
            / "vllm_mlx/constrained/thinking_processor.py"
        )
        loader = importlib.machinery.SourceFileLoader(
            "thinking_processor_under_test", str(path)
        )
        spec = importlib.util.spec_from_loader("thinking_processor_under_test", loader)
        assert spec is not None
        module = importlib.util.module_from_spec(spec)
        loader.exec_module(module)
        return module
    finally:
        # Leaving the fake in sys.modules breaks any later test whose
        # production code imports mlx.core lazily (prefix/ssd cache tests).
        if saved_mlx is None:
            sys.modules.pop("mlx", None)
        else:
            sys.modules["mlx"] = saved_mlx
        if saved_mlx_core is None:
            sys.modules.pop("mlx.core", None)
        else:
            sys.modules["mlx.core"] = saved_mlx_core


def test_no_final_content_watchdog_forces_transition_before_budget():
    module = _load_thinking_processor()
    proc = module.ThinkingAwareLogitsProcessor(
        start_token_ids=[1],
        end_token_ids=[2],
        thinking_token_budget=100,
        prompt_has_think_tag=True,
        no_final_content_token_limit=3,
    )

    for token_id in [10, 11, 12]:
        proc._advance_with_token(token_id)

    assert proc.state is module.Phase.TRANSITIONING
    assert proc.watchdog_was_enforced is True
    assert proc.thinking_tokens == 3


def test_budget_transition_does_not_mark_watchdog_enforced():
    module = _load_thinking_processor()
    proc = module.ThinkingAwareLogitsProcessor(
        start_token_ids=[1],
        end_token_ids=[2],
        thinking_token_budget=2,
        prompt_has_think_tag=True,
        no_final_content_token_limit=10,
    )

    for token_id in [10, 11]:
        proc._advance_with_token(token_id)

    assert proc.state is module.Phase.TRANSITIONING
    assert proc.watchdog_was_enforced is False


def test_generation_metadata_uses_processor_limit_not_current_env(monkeypatch):
    from vllm_mlx import server

    proc = SimpleNamespace(
        _no_final_content_token_limit=7,
        watchdog_was_enforced=True,
    )

    monkeypatch.setenv("VLLM_MLX_NO_FINAL_CONTENT_TOKEN_LIMIT", "99")

    metadata = server._generation_metadata(proc)

    assert metadata is not None
    assert metadata.no_final_content_watchdog_tokens == 7
    assert metadata.no_final_content_watchdog_enforced is True


def test_generation_metadata_omitted_without_thinking_processor(monkeypatch):
    from vllm_mlx import server

    monkeypatch.setenv("VLLM_MLX_NO_FINAL_CONTENT_TOKEN_LIMIT", "7")

    assert server._generation_metadata(None) is None


def test_build_thinking_processor_warns_when_watchdog_cannot_fire(monkeypatch, caplog):
    from vllm_mlx import server

    class Tokenizer:
        vocab_size = 100

        def encode(self, text, add_special_tokens=False):
            return {"<think>": [1], "</think>": [2]}[text]

    engine = SimpleNamespace(_tokenizer=Tokenizer())
    monkeypatch.setenv("VLLM_MLX_NO_FINAL_CONTENT_TOKEN_LIMIT", "10")

    proc = server._build_thinking_processor(engine, 5)

    assert proc is not None
    assert proc._no_final_content_token_limit == 10
    assert "will not fire" in caplog.text
