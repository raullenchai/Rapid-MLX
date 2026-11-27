# SPDX-License-Identifier: Apache-2.0
import subprocess
import sys
import textwrap

import pytest


@pytest.mark.requires_mlx
def test_runner_preserves_standard_norms_with_raw_conv(tmp_path):
    script = textwrap.dedent(
        """
        import json
        import runpy
        import sys
        from pathlib import Path

        import mlx.core as mx
        from mlx.utils import tree_flatten
        from mlx_lm import lora, utils
        from mlx_lm.models import qwen3_5

        checkpoint = Path(sys.argv[1])
        config = {
            "model_type": "qwen3_5",
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "linear_num_value_heads": 1,
            "linear_num_key_heads": 1,
            "linear_key_head_dim": 8,
            "linear_value_head_dim": 8,
            "vocab_size": 32,
        }
        (checkpoint / "config.json").write_text(json.dumps(config))
        model = qwen3_5.Model(qwen3_5.ModelArgs.from_dict(config))
        weights = dict(tree_flatten(model.parameters()))
        conv_key = "language_model.model.layers.0.linear_attn.conv1d.weight"
        conv = weights[conv_key]
        # Keep the initialized unit norm gains and write the conv in raw layout.
        weights[conv_key] = conv.moveaxis(1, 2)
        mx.save_safetensors(str(checkpoint / "model.safetensors"), weights)

        def check_base():
            loaded, _ = lora.load(str(checkpoint))
            loaded_weights = dict(tree_flatten(loaded.parameters()))
            gain = loaded_weights["language_model.model.norm.weight"]
            assert mx.array_equal(gain, mx.ones_like(gain)).item(), gain
            assert mx.array_equal(loaded_weights[conv_key], conv).item()

            from rapid_mlx.patches.qwen3_5_norm_shift import is_installed

            assert is_installed()
            assert qwen3_5._RAPID_MLX_NORM_SHIFT_INSTALLED

        utils.load_tokenizer = lambda *args, **kwargs: None
        lora.main = check_base
        sys.argv = ["runner", "--qwen35", "--train"]
        runpy.run_module("rapid_mlx._finetune_runner", run_name="__main__")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_native_gated_delta_gradients_match_stock(monkeypatch):
    try:
        import mlx.core as mx
    except ImportError as exc:
        pytest.skip(str(exc))
    if not mx.metal.is_available() or not hasattr(mx.fast, "gated_delta_update"):
        pytest.skip("requires the MLX gated-delta Metal kernel")
    pytest.importorskip("mlx_lm")
    from mlx_lm.models import gated_delta, qwen3_5

    from rapid_mlx._finetune_runner import _install_gated_delta

    stock = gated_delta.gated_delta_update
    monkeypatch.setattr(gated_delta, "gated_delta_update", stock)
    monkeypatch.setattr(qwen3_5, "gated_delta_update", qwen3_5.gated_delta_update)
    assert _install_gated_delta()
    native = qwen3_5.gated_delta_update
    assert gated_delta.gated_delta_update is native and native is not stock

    # Use head dimensions supported by the Metal kernel and a short sequence.
    mx.random.seed(0)
    shapes = (
        [(1, 17, 16, 128)] * 2 + [(1, 17, 32, 128)] + [(1, 17, 32)] * 2 + [(32,)] * 2
    )
    inputs = [0.1 * mx.random.normal(shape) for shape in shapes]

    def loss(update):
        def run(*args):
            y, state = update(*args, use_kernel=False)
            return mx.square(y).sum() + mx.square(state).sum()

        return run

    with mx.stream(mx.gpu):
        expected, expected_grads = mx.value_and_grad(
            loss(stock), argnums=list(range(7))
        )(*inputs)
        actual, actual_grads = mx.value_and_grad(loss(native), argnums=list(range(7)))(
            *inputs
        )
        for x, y in zip(
            (actual, *actual_grads), (expected, *expected_grads), strict=True
        ):
            assert (
                mx.max(mx.abs(x - y)).item() <= 0.005 * mx.max(mx.abs(y)).item() + 1e-7
            )
