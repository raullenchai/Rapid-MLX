# SPDX-License-Identifier: Apache-2.0
"""Reasoning-to-schema handoff with a real matcher and CPU mask stand-ins.

No models, downloads, or Metal operations. The MLLM prefill calls processors
with [], then decode appends each current input token before sampling the next
one. Also exercise the text scheduler's prompt-inclusive history.
"""

from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
llguidance = pytest.importorskip("llguidance")
pytest.importorskip("tokenizers")

from llguidance import LLMatcher, LLTokenizer
from llguidance.numpy import (
    allocate_token_bitmask,
    apply_token_bitmask_inplace,
    fill_next_token_bitmask,
)
from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers

from rapid_mlx.api import guided
from rapid_mlx.api import tool_grammar as tg


@pytest.fixture
def setup_processor(monkeypatch):
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {symbol: i for i, symbol in enumerate(alphabet)}
    # Exercise both a merged newline token and one spanning whitespace + JSON.
    vocab.update({"ĊĊ": len(vocab), "ĊĊ{": len(vocab) + 1})
    tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=[("Ċ", "Ċ"), ("ĊĊ", "{")]))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.add_special_tokens(
        [AddedToken(s, special=True) for s in ("<eos>", "<think>", "</think>")]
    )
    lltok = LLTokenizer(tokenizer.to_str(), eos_token=tokenizer.token_to_id("<eos>"))

    def apply(logits, mask):
        result = logits.copy()
        apply_token_bitmask_inplace(result, mask)
        return result

    monkeypatch.setattr(tg, "get_request_matcher", lambda tok, g: LLMatcher(tok, g))
    monkeypatch.setattr(tg, "allocate_token_bitmask", allocate_token_bitmask)
    monkeypatch.setattr(tg, "fill_next_token_bitmask", fill_next_token_bitmask)
    monkeypatch.setattr(tg, "apply_token_bitmask", apply)
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    monkeypatch.setattr(guided, "HAS_LLGUIDANCE", True)
    monkeypatch.setattr(guided, "LLMatcher", LLMatcher)
    monkeypatch.setattr(tg, "get_lltokenizer", lambda _: lltok)
    monkeypatch.setattr(tg, "model_stop_token_ids", lambda _: [])
    end_id = tokenizer.token_to_id("</think>")

    def build(gated=True):
        processor = guided.build_json_schema_logits_processor(
            tokenizer, schema, reasoning_end_id=end_id if gated else None
        )
        assert processor is not None
        return processor

    return SimpleNamespace(
        build=build,
        encode=lambda text: tokenizer.encode(text).ids,
        decode=lambda ids: tokenizer.decode(ids, skip_special_tokens=False),
        end_id=end_id,
        logits=lambda: np.zeros((1, lltok.vocab_size), dtype=np.float32),
    )


@pytest.mark.parametrize("prompt", [[], [17, 18, 19]])
@pytest.mark.parametrize("answer", ['{"answer":"4"}', '\n\n{"answer":"}  4"}'])
def test_cumulative_history_handoff(setup_processor, prompt, answer):
    """MLLM uses []; text TokenBuffer uses prompt; both precede token zero."""
    env = setup_processor
    processor = env.build()
    history = prompt.copy()
    processor(history, env.logits())
    # Each processor call precedes sampling the NEXT token. The current token
    # has already been appended, including the MLLM compute-ahead input.
    for token in env.encode("<think>Reasoning freely."):
        history.append(token)
        assert processor(history, env.logits()).sum() == 0
        assert not processor._reasoning_ended
    history.append(env.end_id)
    processor(history, env.logits())
    assert processor._reasoning_ended
    for token in env.encode(answer):
        masked = processor(history, env.logits())
        assert np.isfinite(masked[0, token])
        history.append(token)
    processor(history, env.logits())
    assert processor._matcher.is_accepting()
    assert not processor._aborted
    assert processor._committed == len(history)


def test_boundary_and_newlines_in_one_update(setup_processor):
    env = setup_processor
    processor = env.build()
    processor([], env.logits())
    history = env.encode('reasoning</think>\n\n{"answer":"4"}')
    processor(history, env.logits())
    assert processor._matcher.is_accepting()
    assert not processor._aborted


def test_reasoning_may_spend_entire_completion_budget(setup_processor):
    env = setup_processor
    processor = env.build()
    history = []
    processor(history, env.logits())
    for token in env.encode("a" * 4096):
        history.append(token)
        assert np.isfinite(processor(history, env.logits())).all()
    assert not processor._reasoning_ended
    assert not processor._aborted


def test_requests_have_independent_gate_and_matcher(setup_processor):
    env = setup_processor
    first, second = env.build(), env.build()
    first([], env.logits())
    second([], env.logits())
    first(env.encode('reasoning</think>{"answer":"4"}'), env.logits())
    assert first._matcher.is_accepting()
    assert not second._matcher.is_accepting()
    assert not second._reasoning_ended


def test_thinking_disabled_constrains_token_zero(setup_processor):
    env = setup_processor
    processor = env.build(gated=False)
    history = []
    assert not np.isfinite(processor(history, env.logits())[0, env.encode("\n")[0]])
    for token in env.encode('{"answer":"4"}'):
        assert np.isfinite(processor(history, env.logits())[0, token])
        history.append(token)
    processor(history, env.logits())
    assert processor._matcher.is_accepting()
    assert not processor._aborted


@pytest.mark.parametrize("separator", ["", "\n\n", " \t\r\n", " " * 32])
def test_post_reasoning_whitespace_is_bounded(setup_processor, separator):
    env = setup_processor
    processor = env.build()
    history = []
    processor(history, env.logits())
    history.append(env.end_id)
    for token in env.encode(separator):
        assert np.isfinite(processor(history, env.logits())[0, token])
        history.append(token)
    masked = processor(history, env.logits())
    if len(separator) == 32:
        assert not np.isfinite(masked[0, env.encode(" ")[0]])
    assert not np.isfinite(masked[0, env.encode("x")[0]])
    assert not processor._matcher.is_accepting()
    for token in env.encode('{"answer":"4"}'):
        assert np.isfinite(processor(history, env.logits())[0, token])
        history.append(token)
    processor(history, env.logits())
    assert processor._matcher.is_accepting()
    assert not processor._aborted


def test_prefix_preserves_required_property_and_value_type(setup_processor):
    env = setup_processor
    processor = env.build()
    processor([], env.logits())
    history = env.encode('reasoning</think>\n\n{"answer":')
    masked = processor(history, env.logits())
    assert not np.isfinite(masked[0, env.encode("4")[0]])
    assert np.isfinite(masked[0, env.encode('"')[0]])
    processor = env.build()
    processor([], env.logits())
    masked = processor(env.encode("reasoning</think>\n\n{"), env.logits())
    assert not np.isfinite(masked[0, env.encode("}")[0]])
