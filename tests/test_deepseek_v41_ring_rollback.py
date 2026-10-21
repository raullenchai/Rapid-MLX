"""Speculative rejection must preserve live KV in a wrapped sliding window."""

from types import SimpleNamespace

import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx

import mlx.core as mx

from rapid_mlx.models.deepseek_v41_native.config import ModelArgs
from rapid_mlx.models.deepseek_v41_native.model import Model
from rapid_mlx.models.deepseek_v41_native.serving import stream_generate


class _Tokenizer:
    eos_token_id = 99
    clean_up_tokenization_spaces = False

    def decode(self, tokens):
        return " ".join(map(str, tokens))


class _OracleDrafter:
    def __init__(self, continuation, prompt_length, wrong_index):
        self.continuation = continuation
        self.prompt_length = prompt_length
        self.wrong_index = wrong_index
        self.position = -1
        self.rejections = 0

    def reset(self):
        self.position = -1

    def observe(self, _hidden, position):
        self.position = position

    def propose(self, seed):
        start = self.position + 1 - self.prompt_length
        proposals = self.continuation[start : start + 5].copy()
        # The seed is supplied by the target, as in the real drafter contract.
        proposals[0] = seed
        if self.wrong_index is not None:
            proposals[self.wrong_index] = (proposals[self.wrong_index] + 1) % 32
            self.rejections += 1
        return proposals, None


# stream_generate keeps at most ``window_size`` proposals per chunk, so a
# corrupted index at or past it would never reach the verifier
@pytest.mark.parametrize(
    "window_size,wrong_index",
    [(w, i) for w in (8, 3) for i in (None, 1, 2, 3, 4) if i is None or i < w],
)
def test_wrapped_window_speculation_matches_plain_greedy(
    wrong_index, window_size, monkeypatch
):
    # Exercise rejected suffixes through index 4, independent of the default.
    monkeypatch.setenv("RAPID_MLX_DSV41_DRAFT_DEPTH", "4")
    mx.random.seed(0)
    args = ModelArgs(
        vocab_size=32,
        dim=32,
        n_layers=2,
        n_heads=2,
        head_dim=32,
        rope_head_dim=8,
        q_lora_rank=16,
        o_lora_rank=16,
        o_groups=1,
        window_size=window_size,
        moe_inter_dim=32,
        n_routed_experts=2,
        n_activated_experts=1,
        hc_mult=2,
        compress_ratios=(0, 0),
        dspark_target_layer_ids=(0, 1),
    )
    model = Model(args)
    mx.eval(model.parameters())
    prompt = list(range(1, 14))
    max_tokens = 32
    cache = model.make_cache(max_seq_len=64)
    logits = model(mx.array([prompt]), cache)
    continuation = []
    for _ in range(max_tokens + 5):
        token = int(mx.argmax(logits[:, -1]))
        continuation.append(token)
        logits = model(mx.array([[token]]), cache)

    # Exercise absent/disabled windows too; these layers own no ring state.
    make_cache = model.make_cache

    def make_mixed_cache(**kwargs):
        result = make_cache(**kwargs)
        result.layers.extend(
            [
                SimpleNamespace(comp_state=None),
                SimpleNamespace(window=0, comp_state=None),
            ]
        )
        return result

    model.make_cache = make_mixed_cache
    draft = _OracleDrafter(continuation, len(prompt), wrong_index)
    tokens = [
        chunk.token
        for chunk in stream_generate(
            model,
            _Tokenizer(),
            prompt,
            runtime=SimpleNamespace(drafter=draft, reset=draft.reset),
            max_tokens=max_tokens,
        )
    ]
    assert tokens == continuation[:max_tokens]
    assert (draft.rejections > 0) == (wrong_index is not None)
