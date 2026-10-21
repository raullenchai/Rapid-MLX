from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

from rapid_mlx.models.deepseek_v41_native import model as native_model  # noqa: E402
from rapid_mlx.models.deepseek_v41_native.attention import Attention  # noqa: E402
from rapid_mlx.models.deepseek_v41_native.config import ModelArgs  # noqa: E402
from rapid_mlx.models.deepseek_v41_native.layers import (  # noqa: E402
    precompute_freqs_cis,
)
from rapid_mlx.models.deepseek_v41_native.model import Model  # noqa: E402


def _args(**overrides) -> ModelArgs:
    base = dict(
        dim=64,
        vocab_size=32,
        n_layers=2,
        n_heads=2,
        head_dim=32,
        rope_head_dim=8,
        q_lora_rank=16,
        o_lora_rank=8,
        o_groups=2,
        hc_mult=2,
        compress_ratios=(0, 0),
        n_routed_experts=4,
        n_activated_experts=2,
        moe_inter_dim=16,
    )
    base.update(overrides)
    return ModelArgs(**base)


@pytest.mark.parametrize("compress_ratios", [(0,), (2,)])
def test_rope_table_grows_geometrically_with_identical_rows(compress_ratios) -> None:
    attention = Attention(0, _args(n_layers=1, compress_ratios=compress_ratios))
    cos, sin = attention._freqs(65)
    assert cos.shape[0] == sin.shape[0] == 128

    # Decode steps inside the table reuse it instead of rebuilding per step.
    for upto in (66, 100, 128):
        assert attention._freqs(upto)[0] is cos

    grown, grown_sin = attention._freqs(129)
    assert grown.shape[0] == 256
    rd, orig_len, theta, factor, beta_fast, beta_slow = attention._rope
    exact_cos, exact_sin = precompute_freqs_cis(
        rd, 129, orig_len, theta, factor, beta_fast, beta_slow
    )
    assert mx.array_equal(grown[:129], exact_cos).item()
    assert mx.array_equal(grown_sin[:129], exact_sin).item()


@pytest.mark.parametrize("compress_ratios", [(0,), (2,)])
def test_long_rope_tables_grow_in_bounded_chunks(compress_ratios) -> None:
    attention = Attention(0, _args(n_layers=1, compress_ratios=compress_ratios))
    assert attention._freqs(4096)[0].shape[0] == 4096
    assert attention._freqs(4097)[0].shape[0] == 8192
    cos, sin = attention._freqs(16385)
    assert cos.shape[0] == 20480  # past one chunk: whole chunks, not doubling
    assert attention._freqs(20480)[0] is cos

    rd, orig_len, theta, factor, beta_fast, beta_slow = attention._rope
    exact_cos, exact_sin = precompute_freqs_cis(
        rd, 16385, orig_len, theta, factor, beta_fast, beta_slow
    )
    assert mx.array_equal(cos[:16385], exact_cos).item()
    assert mx.array_equal(sin[:16385], exact_sin).item()


def test_small_rope_tables_keep_the_minimum_size() -> None:
    attention = Attention(0, _args(n_layers=1, compress_ratios=(0,)))
    assert attention._freqs(1)[0].shape[0] == 64


_blocking_eval = mx.eval  # held before tests patch the shared module


def _decode_trace(model: Model, interval: int) -> list:
    model.eval_interval = interval
    cache = model.make_cache(max_seq_len=40)
    outputs = [model(mx.array([[1, 2, 3, 4, 5]]), cache)]
    for token in range(6, 24):
        outputs.append(model(mx.array([[token % 32]]), cache))
    _blocking_eval(outputs)
    return outputs


def test_async_layer_boundaries_leave_logits_bit_identical(monkeypatch) -> None:
    mx.random.seed(7)
    model = Model(_args(n_layers=4, compress_ratios=(0, 0, 0, 0)))
    mx.eval(model.parameters())
    # The stock path: no boundary at all inside the forward pass.
    reference = _decode_trace(model, 0)

    committed = []
    original = mx.async_eval
    monkeypatch.setattr(
        native_model.mx,
        "async_eval",
        lambda *arrays: committed.append(len(arrays)) or original(*arrays),
    )
    monkeypatch.setattr(
        native_model.mx,
        "eval",
        lambda *_arrays: pytest.fail("a layer boundary blocked the host"),
    )
    for interval, boundaries in ((1, 4), (2, 2), (40, 0)):
        committed.clear()
        actual = _decode_trace(model, interval)
        assert committed == [2] * boundaries * len(actual)
        for expected, got in zip(reference, actual, strict=True):
            assert mx.array_equal(expected, got).item()
