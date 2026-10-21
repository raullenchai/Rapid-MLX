from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

from rapid_mlx.models.deepseek_v41_native import attention  # noqa: E402
from rapid_mlx.models.deepseek_v41_native.config import ModelArgs  # noqa: E402
from rapid_mlx.models.deepseek_v41_native.model import Model  # noqa: E402

DEVICES = [mx.cpu] + ([mx.gpu] if mx.metal.is_available() else [])


def _model(compress_ratios: tuple[int, ...]) -> Model:
    mx.random.seed(7)
    args = ModelArgs(
        dim=64,
        vocab_size=48,
        n_layers=len(compress_ratios),
        n_heads=2,
        head_dim=32,
        rope_head_dim=8,
        q_lora_rank=16,
        o_lora_rank=8,
        o_groups=2,
        window_size=4,
        hc_mult=2,
        hc_sinkhorn_iters=4,
        compress_ratios=compress_ratios,
        kv_source_layers=(0,),
        index_source_layers=(0,),
        index_n_heads=2,
        index_head_dim=32,
        index_topk=3,
        n_routed_experts=4,
        n_activated_experts=2,
        moe_inter_dim=16,
        original_seq_len=0,
    )
    model = Model(args)
    # Non-trivial hyper-connection mixes so every layer contributes.
    for layer in model.layers:
        for name in ("hc_attn_fn", "hc_ffn_fn"):
            setattr(layer, name, mx.random.normal(getattr(layer, name).shape) * 0.1)
        layer.attn.attn_sink = mx.random.normal(layer.attn.attn_sink.shape)
    mx.eval(model.parameters())
    return model


def _decode(
    model: Model, gather: bool, monkeypatch, repeat: int = 1, steps: int = 5
) -> list[mx.array]:
    monkeypatch.setattr(attention, "GATHER_DECODE_ROWS", gather)
    prompt = mx.array([[3, 9, 1, 30, 7, 22, 5, 11, 2, 40, 17, 8] * repeat] * 2)
    prompt[1] = mx.array([4, 4, 12, 6, 19, 33, 2, 1, 9, 9, 27, 3] * repeat)
    cache = model.make_cache(bsz=2, max_seq_len=prompt.shape[1] + steps + 8)
    logits = [model(prompt, cache)]
    tokens = mx.array([[5], [13]])
    for _ in range(steps):
        step = model(tokens, cache)
        logits.append(step)
        tokens = mx.argmax(step[:, -1:], axis=-1).astype(mx.int32)
    mx.eval(logits)
    return logits


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("compress_ratios", [(1, 1), (2, 2), (2, 0, 2)])
def test_decode_row_gather_is_bit_exact_with_full_concat(
    device, compress_ratios, monkeypatch
) -> None:
    previous = mx.default_device()
    mx.set_default_device(device)
    try:
        model = _model(compress_ratios)
        expected = _decode(model, False, monkeypatch)
        actual = _decode(model, True, monkeypatch)
    finally:
        mx.set_default_device(previous)
    for want, got in zip(expected, actual, strict=True):
        assert mx.array_equal(want, got).item()


@pytest.mark.parametrize("device", DEVICES)
def test_decode_row_gather_keeps_masked_selections_exact(device, monkeypatch) -> None:
    # Severing the shared top-k hands consumer layers an all -1 selection.
    previous = mx.default_device()
    mx.set_default_device(device)
    try:
        model = _model((1, 1))
        model._break_sharing = True
        expected = _decode(model, False, monkeypatch)
        actual = _decode(model, True, monkeypatch)
    finally:
        mx.set_default_device(previous)
    for want, got in zip(expected, actual, strict=True):
        assert mx.array_equal(want, got).item()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("compress_ratios", [(1, 1), (4, 4)])
def test_decode_row_gather_is_exact_far_past_the_top_k(
    device, compress_ratios, monkeypatch
) -> None:
    # 120-token prompt: the top-3 selection is a small fraction of the rows.
    previous = mx.default_device()
    mx.set_default_device(device)
    try:
        model = _model(compress_ratios)
        expected = _decode(model, False, monkeypatch, repeat=10, steps=8)
        actual = _decode(model, True, monkeypatch, repeat=10, steps=8)
    finally:
        mx.set_default_device(previous)
    for want, got in zip(expected, actual, strict=True):
        assert mx.array_equal(want, got).item()


def test_decode_row_gather_engages_only_when_selection_is_smaller(monkeypatch) -> None:
    gathers = []
    original = attention.mx.concatenate

    def spy(arrays, axis=0):
        if axis == 1 and len(arrays) == 2:
            gathers.append(int(arrays[1].shape[1]))
        return original(arrays, axis=axis)

    model = _model((1,))
    monkeypatch.setattr(attention.mx, "concatenate", spy)
    cache = model.make_cache(max_seq_len=16)
    model(mx.array([[1, 2, 3, 4, 5, 6]]), cache)  # prefill keeps all 6 rows
    model(mx.array([[7]]), cache)  # decode: top-3 of 7 rows -> 3 gathered
    assert gathers[-1] == 3
    assert 6 in gathers

    monkeypatch.setattr(attention, "GATHER_DECODE_ROWS", False)
    model(mx.array([[8]]), cache)
    assert gathers[-1] == 8
