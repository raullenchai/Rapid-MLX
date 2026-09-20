from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest

from bench.repro_mtp_forced_k_parity import (
    _first_divergence,
    _format_prompt,
    _loaded_model_type,
    _parse_k_values,
    _token_sha256,
)


def test_parse_k_values_requires_control_first():
    assert _parse_k_values("0,1,2,3") == (0, 1, 2, 3)
    with pytest.raises(argparse.ArgumentTypeError, match="start with the K=0 control"):
        _parse_k_values("1,2,3")
    with pytest.raises(argparse.ArgumentTypeError, match="speculative depth"):
        _parse_k_values("0")


def test_parse_k_values_rejects_duplicates_and_negative_depths():
    with pytest.raises(argparse.ArgumentTypeError, match="duplicates"):
        _parse_k_values("0,1,1")
    with pytest.raises(argparse.ArgumentTypeError, match="non-negative"):
        _parse_k_values("0,-1")
    with pytest.raises(argparse.ArgumentTypeError, match="supported range"):
        _parse_k_values("0,4")


def test_first_divergence_reports_token_flip_and_early_termination():
    assert _first_divergence((10, 20, 30), (10, 21, 30)) == {
        "index": 1,
        "control_token": 20,
        "candidate_token": 21,
    }
    assert _first_divergence((10, 20, 30), (10, 20)) == {
        "index": 2,
        "control_token": 30,
        "candidate_token": None,
    }
    assert _first_divergence((10, 20), (10, 20)) is None


def test_token_hash_is_stable_and_sequence_sensitive():
    assert _token_sha256((10, 20)) == _token_sha256((10, 20))
    assert _token_sha256((10, 20)) != _token_sha256((20, 10))


def test_loaded_model_type_resolves_outer_and_inner_shapes():
    assert _loaded_model_type(SimpleNamespace(model_type="gemma4")) == "gemma4"
    assert (
        _loaded_model_type(
            SimpleNamespace(language_model=SimpleNamespace(model_type="qwen3_5"))
        )
        == "qwen3_5"
    )
    assert (
        _loaded_model_type(SimpleNamespace(config=SimpleNamespace(model_type="hy_v3")))
        == "hy_v3"
    )
    assert _loaded_model_type(SimpleNamespace()) is None


def test_format_prompt_can_mirror_server_chat_template():
    calls = []

    class _Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            calls.append((messages, kwargs))
            return "formatted"

    tokenizer = _Tokenizer()
    assert _format_prompt(tokenizer, "hello", chat_template=False) == "hello"
    assert _format_prompt(tokenizer, "hello", chat_template=True) == "formatted"
    assert calls == [
        (
            [{"role": "user", "content": "hello"}],
            {
                "tokenize": False,
                "add_generation_prompt": True,
                "enable_thinking": False,
            },
        )
    ]
