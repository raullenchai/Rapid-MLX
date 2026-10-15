# SPDX-License-Identifier: Apache-2.0
"""Tiny random-weight contracts for the Nemotron Bolt text backbone."""

import sys

import pytest

pytest.importorskip("mlx.core")
pytestmark = pytest.mark.requires_mlx

import mlx.core as mx  # noqa: E402


def _assert_float32_parity(actual, expected):
    # M5 float32 GEMM uses TF32 (10 fraction bits), whereas GEMV does not.
    # Prefill/decode and different SSM chunks can select different kernels.
    # Allow eight TF32 rounding units across the composed matmuls. Scale by
    # the reference RMS and peak, rather than each element: cancellation can
    # leave individual outputs near zero. Both bounds still reject 1% scaling.
    tolerance = 8 * 2**-10
    error = actual - expected
    assert bool(
        mx.mean(mx.square(error)) <= tolerance**2 * mx.mean(mx.square(expected))
    )
    assert bool(mx.max(mx.abs(error)) <= tolerance * mx.max(mx.abs(expected)))


@pytest.fixture(autouse=True)
def _isolate_puzzle_registration(monkeypatch):
    from rapid_mlx.utils import tokenizer

    name = "mlx_lm.models.nemotron_h_puzzle"
    previous = sys.modules.get(name)
    monkeypatch.setattr(
        tokenizer, "_VENDORED_MODEL_TYPES", set(tokenizer._VENDORED_MODEL_TYPES)
    )
    sys.modules.pop(name, None)
    tokenizer._VENDORED_MODEL_TYPES.discard("nemotron_h_puzzle")
    mx.random.seed(0)
    try:
        yield
    finally:
        sys.modules.pop(name, None)
        if previous is not None:
            sys.modules[name] = previous


class TestNemotronBolt:
    @staticmethod
    def config(**overrides):
        config = dict(
            model_type="nemotron_h_puzzle",
            vocab_size=64,
            hidden_size=32,
            intermediate_size=48,
            num_hidden_layers=4,
            max_position_embeddings=256,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
            attention_bias=False,
            mamba_num_heads=2,
            mamba_head_dim=16,
            mamba_proj_bias=False,
            ssm_state_size=32,
            conv_kernel=4,
            n_groups=1,
            mlp_bias=False,
            layer_norm_epsilon=1e-5,
            use_bias=False,
            use_conv_bias=True,
            moe_latent_size=16,
            moe_shared_expert_intermediate_size=48,
            n_shared_experts=1,
            n_group=1,
            topk_group=1,
            n_routed_experts=512,
            num_experts_per_tok=9,
            norm_topk_prob=True,
            routed_scaling_factor=5.0,
            time_step_limit=[0.0, {"__float__": "Infinity"}],
            block_configs=[
                {"block_type": "mamba"},
                {
                    "block_type": "moe",
                    "n_routed_experts": 4,
                    "moe_intermediate_size": 16,
                    "num_experts_per_tok": 2,
                },
                {"block_type": "attention"},
                {
                    "block_type": "moe",
                    "n_routed_experts": 6,
                    "moe_intermediate_size": 24,
                    "num_experts_per_tok": 3,
                },
            ],
        )
        return config | overrides

    @staticmethod
    def module():
        from rapid_mlx.models import nemotron_h_puzzle

        return nemotron_h_puzzle

    def model(self, **overrides):
        module = self.module()
        return module.Model(module.ModelArgs.from_dict(self.config(**overrides)))

    def test_config_omni_and_blockwise_truth(self):
        from copy import deepcopy

        module = self.module()
        inner = self.config(auto_map={"AutoConfig": "remote.Config"})
        omni = {
            "model_type": "nemotron_h_omni",
            "llm_config": inner,
            "vision_config": {"hidden_size": 999},
        }
        before = deepcopy(omni)
        text = module.text_config(omni)
        assert text["model_type"] == "nemotron_h_puzzle"
        assert "auto_map" not in text and "vision_config" not in text
        args = module.ModelArgs.from_dict(omni)
        assert args.hybrid_override_pattern == ["M", "E", "*", "E"]
        assert args.time_step_limit == (0, float("inf"))
        assert args.for_layer(1).n_routed_experts == 4
        assert args.for_layer(3).n_routed_experts == 6
        assert args.for_layer(1).moe_intermediate_size == 16
        assert args.for_layer(3).num_experts_per_tok == 3
        assert args.n_routed_experts == 512
        assert omni == before
        inner.pop("num_hidden_layers")
        assert module.ModelArgs.from_dict(inner).num_hidden_layers == 4
        with pytest.raises(ValueError, match="Expected"):
            module.text_config({"model_type": "other"})

    def test_config_validation_and_fallbacks(self):
        module = self.module()
        for changes, message in [
            ({"num_hidden_layers": 3}, "one entry"),
            ({"block_configs": [{"block_type": "bad"}] * 4}, "unsupported"),
            ({"block_configs": [{"block_type": "moe"}] * 4}, "missing"),
            ({"hybrid_override_pattern": ["M"] * 4}, "conflicts"),
        ]:
            with pytest.raises(ValueError, match=message):
                module.ModelArgs.from_dict(self.config(**changes))
        config = self.config()
        config["block_configs"][1]["n_routed_experts"] = 1
        with pytest.raises(ValueError, match="top-k"):
            module.ModelArgs.from_dict(config).for_layer(1)
        config = self.config()
        config["block_configs"][1]["num_experts_per_tok"] = None
        with pytest.raises(ValueError, match="top-k"):
            module.ModelArgs.from_dict(config).for_layer(1)
        args = module.ModelArgs.from_dict(
            self.config(
                block_configs=None,
                layers_block_type=["mamba", "attention"],
                num_hidden_layers=None,
                time_step_limit=None,
                time_step_min=0.002,
            )
        )
        assert args.num_hidden_layers == 2 and args.for_layer(0) is args
        assert args.time_step_limit == (0.002, float("inf"))
        # Reference semantics: time_step_max never becomes an inference bound.
        assert module.ModelArgs.from_dict(
            self.config(time_step_limit=None, time_step_min=0.002, time_step_max=0.1)
        ).time_step_limit == (0.002, float("inf"))
        assert (
            module.ModelArgs.from_dict(
                self.config(time_step_limit=None)
            ).time_step_limit[0]
            == 0
        )
        args = module.ModelArgs.from_dict(self.config())
        args.block_configs[1].pop("n_routed_experts")
        assert args.for_layer(1).n_routed_experts == 512

    @pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
    def test_four_layer_forward_and_cached_decode(self, dtype):
        model = self.model()
        model.set_dtype(dtype)
        # Nonzero router weights exercise FP32 matmul rather than tied zero logits.
        for index in (1, 3):
            gate = model.layers[index].mixer.gate
            gate.weight = mx.random.normal(gate.weight.shape).astype(dtype) * 0.1
        tokens = mx.array([[1, 2, 3, 4]])
        expected = model(tokens)
        cache = model.make_cache()
        prefill = model(tokens[:, :3], cache)
        decoded = model(tokens[:, 3:], cache)
        mx.eval(expected, prefill, decoded)
        assert expected.shape == (1, 4, 64)
        assert bool(mx.all(mx.isfinite(expected)))
        if dtype == mx.bfloat16:
            assert bool(mx.allclose(expected[:, :3], prefill, atol=0.08, rtol=0.08))
            assert bool(mx.allclose(expected[:, 3:], decoded, atol=0.08, rtol=0.08))
        else:
            _assert_float32_parity(prefill, expected[:, :3])
            _assert_float32_parity(decoded, expected[:, 3:])
            # Inject a real output-head scaling error; the TF32 budget must
            # remain small enough to detect even a 1% change.
            model.lm_head.weight = model.lm_head.weight * 1.01
            with pytest.raises(AssertionError):
                _assert_float32_parity(model(tokens), expected)
        assert len(cache) == 2
        for index, count, width in [(1, 4, 16), (3, 6, 24)]:
            mixer = model.layers[index].mixer
            assert mixer.gate.weight.shape == (count, 32)
            assert mixer.switch_mlp.fc1.weight.shape == (count, width, 16)
        assert model.cast_predicate("backbone.layers.0.mixer.A_log") is False
        assert (
            model.cast_predicate("backbone.layers.1.mixer.gate.e_score_correction_bias")
            is False
        )
        assert model.cast_predicate("lm_head.weight") is True

    @pytest.mark.parametrize(
        "blocks",
        [
            [{"block_type": "mlp"}],
            [{"block_type": "attention"}],
            [
                {
                    "block_type": "moe",
                    "moe_intermediate_size": 16,
                    "n_routed_experts": 4,
                    "num_experts_per_tok": 2,
                }
            ],
            [{"block_type": "mamba"}],
        ],
    )
    def test_single_kind_schedules(self, blocks):
        model = self.model(
            block_configs=blocks,
            num_hidden_layers=1,
            moe_latent_size=None,
            n_shared_experts=None,
        )
        logits = model(mx.array([[1, 2]]), model.make_cache())
        mx.eval(logits)
        assert logits.shape == (1, 2, 64) and bool(mx.all(mx.isfinite(logits)))

    def test_sanitize_heterogeneous_experts_and_omni(self):
        model = self.model()
        original = {
            "language_model.backbone.embeddings.weight": mx.ones((64, 32)),
            "language_model.backbone.norm_f.weight": mx.ones(32),
            "language_model.lm_head.weight": mx.ones((64, 32)),
            "language_model.mtp.layers.0.weight": mx.ones(1),
            "vision_model.radio_model.weight": mx.ones(1),
            "mlp1.0.weight": mx.ones(1),
            "language_model.backbone.layers.0.mixer.conv1d.weight": mx.ones((96, 1, 4)),
        }
        for index, count, width in [(1, 4, 16), (3, 6, 24)]:
            for expert in range(count):
                prefix = (
                    f"language_model.backbone.layers.{index}.mixer.experts.{expert}"
                )
                original[f"{prefix}.up_proj.weight"] = mx.full((width, 16), expert)
                original[f"{prefix}.down_proj.weight"] = mx.full((16, width), expert)
        result = model.sanitize(original)
        assert len(original) == 27
        assert not any(
            key.startswith(("language_model.", "vision_model.", "mlp1.", "mtp."))
            for key in result
        )
        for index, count, width in [(1, 4, 16), (3, 6, 24)]:
            prefix = f"backbone.layers.{index}.mixer.switch_mlp"
            assert result[f"{prefix}.fc1.weight"].shape == (count, width, 16)
            assert result[f"{prefix}.fc2.weight"].shape == (count, 16, width)
            assert bool(mx.all(result[f"{prefix}.fc1.weight"][-1] == count - 1))
        assert result["backbone.layers.0.mixer.conv1d.weight"].shape == (96, 4, 1)
        twice = model.sanitize(result)
        assert all(twice[key] is value for key, value in result.items())
        assert "backbone.embeddings.weight" in model.sanitize(
            {"model.embeddings.weight": mx.ones(1)}
        )

    def test_norm_router_and_quantization_boundaries(self):
        module = self.module()
        norm = module.NemotronHRMSNorm(32, 1e-5)
        norm.weight = mx.linspace(0.5, 1.5, 32)
        x = mx.linspace(-2, 3, 32).astype(mx.bfloat16)
        reference = (
            x.astype(mx.float32)
            * mx.rsqrt(mx.mean(mx.square(x.astype(mx.float32))) + 1e-5)
            * norm.weight
        ).astype(x.dtype)
        assert bool(mx.array_equal(norm(x), reference))
        gate = self.model().layers[1].mixer.gate
        gate.weight = mx.random.normal(gate.weight.shape).astype(mx.bfloat16)
        gate.e_score_correction_bias = mx.array([0.1, -0.1, 0, 0.2])
        x = mx.random.normal((2, 32)).astype(mx.bfloat16)
        inds, scores = gate(x)
        logits = x.astype(mx.float32) @ gate.weight.astype(mx.float32).T
        raw = mx.sigmoid(logits)
        expected_inds = mx.argsort(-(raw + gate.e_score_correction_bias), axis=-1)[
            :, :2
        ]
        assert bool(mx.array_equal(mx.sort(inds), mx.sort(expected_inds)))
        selected = mx.take_along_axis(raw, inds, axis=-1)
        expected_scores = selected / selected.sum(-1, keepdims=True) * 5
        assert bool(mx.allclose(scores, expected_scores))
        # Missing routed scaling must fail independently of forward tolerances.
        gate.routed_scaling_factor = 1.0
        _, unscaled_scores = gate(x)
        assert not bool(mx.allclose(unscaled_scores, expected_scores))
        predicate = self.model().quant_predicate
        for path in (
            "lm_head",
            "backbone.layers.1.mixer.gate",
            "backbone.layers.1.mixer.fc1_latent_proj",
            "backbone.layers.1.mixer.fc2_latent_proj",
        ):
            assert predicate(path, None) is False
        assert predicate("backbone.layers.1.mixer.switch_mlp.fc1", None) is True

    @pytest.mark.parametrize("normalize", [True, False])
    def test_top_one_router_normalization(self, normalize):
        gate = self.model().layers[1].mixer.gate
        gate.top_k = 1
        gate.norm_topk_prob = normalize
        gate.weight = mx.zeros_like(gate.weight)
        _, weights = gate(mx.ones((2, 32)))
        expected = gate.routed_scaling_factor * (1 if normalize else 0.5)
        assert bool(mx.array_equal(weights, mx.full((2, 1), expected)))

    @pytest.mark.parametrize("use_mask", [True, False])
    @pytest.mark.parametrize("lengths", [(256, 300), (0, 256), (255, 257)])
    def test_right_padded_ssm_matches_independent_sequences(self, lengths, use_mask):
        module = self.module()
        size = 333  # A full 256-token chunk followed by a partial chunk.
        x = mx.random.normal((2, size, 2, 16))
        b = mx.random.normal((2, size, 1, 32))
        c = mx.random.normal((2, size, 1, 32))
        # Slow decay keeps the incoming state measurable across a chunk boundary.
        dt = mx.full((2, size, 2), -5.0)
        initial = mx.random.normal((2, 2, 16, 32))
        a_log, d, bias = mx.zeros(2), mx.ones(2), mx.zeros(2)
        mask = mx.arange(size)[None, :] < mx.array(lengths)[:, None]
        outputs, state = module.ssm_attn(
            x,
            a_log,
            b,
            c,
            d,
            dt,
            bias,
            state=initial,
            mask=mask if use_mask else None,
            lengths=mx.array(lengths),
        )
        for row, length in enumerate(lengths):
            if length == 0:
                assert bool(mx.array_equal(state[row], initial[row]))
                continue
            expected, expected_state = module.ssm_attn(
                x[row : row + 1, :length],
                a_log,
                b[row : row + 1, :length],
                c[row : row + 1, :length],
                d,
                dt[row : row + 1, :length],
                bias,
                state=initial[row : row + 1],
            )
            _assert_float32_parity(outputs[row, :length], expected[0])
            _assert_float32_parity(state[row], expected_state[0])

    @pytest.mark.parametrize("use_mask", [True, False])
    @pytest.mark.parametrize("size,length", [(128, 300), (333, 600)])
    def test_cached_ssm_lengths_exceed_forward(self, size, length, use_mask):
        module = self.module()
        x = mx.random.normal((1, size, 2, 16))
        b = mx.random.normal((1, size, 1, 32))
        c = mx.random.normal((1, size, 1, 32))
        dt = mx.full((1, size, 2), -5.0)
        initial = mx.random.normal((1, 2, 16, 32))
        args = (x, mx.zeros(2), b, c, mx.ones(2), dt, mx.zeros(2))
        # A cached forward consumes only this call's tokens even when the
        # cache's remaining length extends beyond the call or final chunk.
        expected, expected_state = module.ssm_attn(*args, state=initial)
        output, state = module.ssm_attn(
            *args,
            state=initial,
            lengths=mx.array([length]),
            mask=mx.ones((1, size), dtype=mx.bool_) if use_mask else None,
        )
        _assert_float32_parity(output, expected)
        _assert_float32_parity(state, expected_state)

    def test_ssm_native_dt_and_chunked_reference(self):
        module = self.module()
        dt = mx.array([[-1.17, 0.23]], dtype=mx.bfloat16)
        bias = mx.array([0.1, -0.2], dtype=mx.bfloat16)
        assert bool(
            mx.array_equal(
                module.compute_dt_native(dt, bias, (0, 10)),
                mx.clip(mx.logaddexp(0, dt + bias), 0, 10),
            )
        )
        x = mx.random.normal((2, 5, 2, 16))
        b = mx.random.normal((2, 5, 1, 32))
        c = mx.random.normal((2, 5, 1, 32))
        dt = mx.random.normal((2, 5, 2))
        args = (x, mx.zeros(2), b, c, mx.ones(2), dt, mx.zeros(2))
        full, state = module.ssm_attn(*args)
        chunked, chunk_state = module.ssm_attn(*args, step=2)
        _assert_float32_parity(chunked, full)
        _assert_float32_parity(chunk_state, state)
        # Removing the SSM residual is a semantic error, not TF32 roundoff.
        without_residual, _ = module.ssm_attn(
            x, mx.zeros(2), b, c, mx.zeros(2), dt, mx.zeros(2), step=2
        )
        with pytest.raises(AssertionError):
            _assert_float32_parity(without_residual, full)
        mask = mx.array([[True] * 5, [True, True, False, False, False]])
        initial = mx.ones((2, 2, 16, 32))
        result, state = module.ssm_attn(
            *args, state=initial, lengths=mx.array([5, -1]), mask=mask, step=2
        )
        mx.eval(result, state)
        assert bool(mx.array_equal(state[1], initial[1]))

    def test_registration_and_text_loader(self, tmp_path, monkeypatch, caplog):
        import builtins
        import importlib.util
        import json

        from rapid_mlx.utils import tokenizer

        name = "mlx_lm.models.nemotron_h_puzzle"
        monkeypatch.delitem(sys.modules, name, raising=False)
        tokenizer._VENDORED_MODEL_TYPES.discard("nemotron_h_puzzle")
        original_import = builtins.__import__

        def fail_import(name, *args, **kwargs):
            if kwargs.get("fromlist", args[2] if len(args) > 2 else None) == (
                "nemotron_h_puzzle",
            ):
                raise ImportError("injected")
            return original_import(name, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(builtins, "__import__", fail_import)
            tokenizer._register_vendored_archs()
        assert "nemotron_h_puzzle vendored module failed" in caplog.text
        assert "nemotron_h_puzzle" not in tokenizer._VENDORED_MODEL_TYPES
        with monkeypatch.context() as patch:
            original_find = importlib.util.find_spec

            def no_spec(module):
                if module == name:
                    raise ValueError("missing spec")
                return original_find(module)

            patch.setattr(importlib.util, "find_spec", no_spec)
            tokenizer._register_vendored_archs()
        assert sys.modules[name] is self.module()
        tokenizer._register_vendored_archs()
        (tmp_path / "config.json").write_text(
            json.dumps({"model_type": "nemotron_h_puzzle"})
        )
        assert tokenizer._is_vendored_arch_model(str(tmp_path))
        expected = (object(), object())
        monkeypatch.setattr(
            tokenizer, "_load_with_tokenizer_fallback", lambda *a, **kw: expected
        )
        assert tokenizer._load_model_with_fallback_impl(str(tmp_path)) is expected
        monkeypatch.delitem(sys.modules, name)
        with monkeypatch.context() as patch:
            original_find = importlib.util.find_spec
            patch.setattr(
                importlib.util,
                "find_spec",
                lambda module: object() if module == name else original_find(module),
            )
            tokenizer._register_vendored_archs()
        assert name not in sys.modules
        tokenizer._VENDORED_MODEL_TYPES.discard("nemotron_h_puzzle")

    def test_full_size_config_without_allocating_weights(self):
        module = self.module()
        blocks = [{"block_type": "mamba"}] * 40 + [{"block_type": "attention"}] * 8
        counts = (240, 304, 368, 432, 496)
        widths = (1536, 2048, 2304, 2560)
        blocks += [
            {
                "block_type": "moe",
                "n_routed_experts": counts[i % 5],
                "moe_intermediate_size": widths[i % 4],
                "num_experts_per_tok": 9,
            }
            for i in range(40)
        ]
        config = self.config(
            block_configs=blocks,
            num_hidden_layers=None,
            hidden_size=4096,
            moe_latent_size=1024,
        )
        config.pop("n_routed_experts")
        args = module.ModelArgs.from_dict(
            {"model_type": "nemotron_h_omni", "llm_config": config}
        )
        assert args.num_hidden_layers == 88
        assert args.n_routed_experts is None
        assert {args.for_layer(i).n_routed_experts for i in range(48, 88)} == set(
            counts
        )
        assert {args.for_layer(i).moe_intermediate_size for i in range(48, 88)} == set(
            widths
        )
