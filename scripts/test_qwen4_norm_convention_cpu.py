# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-convention regressions; CPU is selected before model imports."""

import json
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest import mock

import mlx.core as mx

mx.set_default_device(mx.cpu)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from vllm_mlx.models.qwen4_exp import (
    Model,
    ModelArgs,
    ZeroCenteredRMSNorm,
)
from vllm_mlx.models.qwen4_norm_convention import (
    normalize_qwen4_checkpoint,
)


def tiny_model():
    return Model(
        ModelArgs.from_dict(
            dict(
                model_type="qwen4_exp",
                text_config=dict(
                    hidden_size=8,
                    num_hidden_layers=2,
                    vocab_size=32,
                    num_attention_heads=2,
                    num_key_value_heads=1,
                    head_dim=4,
                    linear_num_key_heads=1,
                    linear_num_value_heads=3,
                    linear_key_head_dim=4,
                    linear_value_head_dim=4,
                    linear_conv_kernel_dim=3,
                    num_experts=4,
                    num_experts_per_tok=2,
                    moe_intermediate_size=4,
                    shared_expert_intermediate_size=4,
                    hc_count=4,
                    hc_lowrank=3,
                    layer_types=["linear_attention", "full_attention"],
                    indexer_n_heads=2,
                    indexer_kv_heads=1,
                    indexer_head_dim=4,
                    indexer_budget=8,
                    indexer_compress_ratio=2,
                    ple_layer_ids=[],
                    eos_token_id=31,
                    mtp_num_hidden_layers=1,
                ),
            )
        )
    )


def checkpoint(model, *, direct):
    weights = dict(tree_flatten(model.parameters()))
    for path, module in model.named_modules():
        if type(module) is ZeroCenteredRMSNorm:
            weights[f"{path}.weight"] = mx.full(
                module.weight.shape, 1.0 if direct else 0.0, dtype=mx.bfloat16
            )
    return weights


class ConventionTests(unittest.TestCase):
    def test_installed_strict_loader_preserves_converted_fp32_residuals(self):
        from mlx_lm.utils import load_model

        model = tiny_model()
        weights = checkpoint(model, direct=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "config.json").write_text(json.dumps(asdict(model.args)))
            mx.save_safetensors(str(path / "model.safetensors"), weights)
            loaded, _ = load_model(
                path,
                lazy=False,
                strict=True,
                model_config={"model_file": None},
                get_model_classes=lambda config: (Model, ModelArgs),
            )
        self.assertEqual(
            loaded.language_model.norm_convention_receipt["source_convention"],
            "direct_gamma",
        )
        for _path, module in loaded.named_modules():
            if type(module) is ZeroCenteredRMSNorm:
                self.assertEqual(module.weight.dtype, mx.float32)
                np.testing.assert_array_equal(
                    np.array(module.weight), np.zeros(module.weight.shape)
                )

    def test_direct_gamma_checkpoint_recentered_before_strict_load(self):
        model = tiny_model()
        weights = checkpoint(model, direct=True)
        key = "language_model.model.layers.1.self_attn.indexer.q_layernorm.weight"
        gamma = mx.array([0.1, 0.3, 0.75, 1.5], dtype=mx.bfloat16)
        weights[key] = gamma
        untouched = weights["language_model.model.layers.0.linear_attn.norm.weight"]
        sanitized = model.sanitize(weights)
        model.load_weights(list(sanitized.items()), strict=True)
        receipt = model.language_model.norm_convention_receipt
        self.assertEqual(receipt["source_convention"], "direct_gamma")
        self.assertEqual(receipt["anchor_count"], 2)
        self.assertGreater(receipt["recentered_tensors"], 2)
        self.assertIs(
            sanitized["language_model.model.layers.0.linear_attn.norm.weight"],
            untouched,
        )
        self.assertEqual(sanitized[key].dtype, mx.float32)
        np.testing.assert_array_equal(
            np.array(1 + sanitized[key]), np.array(gamma.astype(mx.float32))
        )
        norm = model.model.layers[1].self_attn.indexer.q_layernorm
        x = mx.array([[0.25, -0.5, 0.75, 1.0]], dtype=mx.float32)
        expected = x * mx.rsqrt(
            mx.mean(mx.square(x), axis=-1, keepdims=True) + norm.eps
        )
        expected = expected * gamma.astype(mx.float32)
        np.testing.assert_array_equal(np.array(norm(x)), np.array(expected))

    def test_native_zero_centered_checkpoint_unchanged(self):
        model = tiny_model()
        weights = checkpoint(model, direct=False)
        sanitized = model.sanitize(weights)
        self.assertTrue(all(sanitized[key] is value for key, value in weights.items()))
        receipt = model.language_model.norm_convention_receipt
        self.assertEqual(receipt["source_convention"], "zero_centered")
        self.assertEqual(receipt["recentered_tensors"], 0)
        model.load_weights(list(sanitized.items()), strict=True)

    def test_repeat_canonical_sanitize_cannot_overwrite_source_receipt(self):
        model = tiny_model()
        source = checkpoint(model, direct=True)
        sanitized = model.sanitize(source)
        receipt = model.language_model.norm_convention_receipt
        # A non-norm key-remapping utility call must retain the load decision.
        model.sanitize({"language_model.model.unrelated.weight": mx.zeros((1,))})
        self.assertIs(model.language_model.norm_convention_receipt, receipt)
        with self.assertRaisesRegex(ValueError, "source convention changed"):
            model.sanitize(sanitized)
        self.assertIs(model.language_model.norm_convention_receipt, receipt)
        # Repeating the original source input is harmless and deterministic.
        repeated = model.sanitize(source)
        for key, value in sanitized.items():
            np.testing.assert_array_equal(np.array(repeated[key]), np.array(value))

    def test_mixed_anchor_populations_refused_without_mutating_weights(self):
        model = tiny_model()
        weights = checkpoint(model, direct=True)
        key = "language_model.model.layers.0.attn_hyper_connection.hc_norm.weight"
        weights[key] = mx.zeros_like(weights[key])
        before = weights.copy()
        with self.assertRaisesRegex(ValueError, "ambiguous or mixed"):
            model.sanitize(weights)
        self.assertTrue(all(weights[key] is value for key, value in before.items()))

    def test_ambiguous_anchor_population_refused(self):
        model = tiny_model()
        weights = checkpoint(model, direct=True)
        for key in weights:
            if key.endswith("attn_hyper_connection.hc_norm.weight"):
                weights[key] = mx.full(weights[key].shape, 0.5)
        with self.assertRaisesRegex(ValueError, "ambiguous or mixed"):
            model.sanitize(weights)

    def test_missing_anchor_and_nonfinite_anchor_refused(self):
        for bad in (None, float("nan"), float("inf")):
            with self.subTest(bad=bad):
                model = tiny_model()
                weights = checkpoint(model, direct=True)
                key = (
                    "language_model.model.layers.0.attn_hyper_connection.hc_norm.weight"
                )
                if bad is None:
                    del weights[key]
                else:
                    weights[key] = mx.full(weights[key].shape, bad)
                with self.assertRaises(ValueError):
                    model.sanitize(weights)

    def test_unrepresentable_tiny_or_negative_zero_gain_refused(self):
        for gain in (1e-12, -1e-12, -0.0):
            with self.subTest(gain=gain):
                model = tiny_model()
                weights = checkpoint(model, direct=True)
                key = (
                    "language_model.model.layers.1.self_attn.indexer.q_layernorm.weight"
                )
                weights[key] = mx.full(weights[key].shape, gain)
                with self.assertRaisesRegex(
                    ValueError, "cannot be represented exactly"
                ):
                    model.sanitize(weights)

    def test_raw_anchor_outliers_do_not_require_every_mean_below_half(self):
        # A valid raw counterpart may have trained anchor outliers >0.5;
        # retain the established 90% vote + median admission contract.
        class Anchors(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = []
                for _ in range(48):
                    layer = nn.Module()
                    layer.attn_hyper_connection = nn.Module()
                    layer.attn_hyper_connection.hc_norm = ZeroCenteredRMSNorm(4)
                    self.layers.append(layer)

        for direct in (False, True):
            model = Anchors()
            weights = dict(tree_flatten(model.parameters()))
            for index, key in enumerate(weights):
                mean = 0.765625 if index == 0 else 0.0390625
                weights[key] = mx.full((4,), mean + int(direct))
            receipt = normalize_qwen4_checkpoint(model, weights, ZeroCenteredRMSNorm)
            self.assertEqual(
                receipt["source_convention"],
                "direct_gamma" if direct else "zero_centered",
            )

    def test_mtp_inherits_backbone_convention_not_its_ambiguous_means(self):
        from vllm_mlx.spec_decode.mtp import qwen4_exp_inject as inject

        for direct in (False, True):
            with self.subTest(direct=direct), mock.patch.object(nn, "quantize"):
                model = tiny_model()
                model.sanitize(checkpoint(model, direct=direct))
                mtp = inject._build_mtp(model.language_model)
                weights = checkpoint(mtp, direct=direct)
                # Learned MTP direct gains can be well below the anchor band.
                gamma = mx.full(
                    mtp.pre_fc_norm_embedding.weight.shape, 0.1, dtype=mx.bfloat16
                ).astype(mx.float32)
                weights["pre_fc_norm_embedding.weight"] = gamma if direct else gamma - 1
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "mtp.safetensors"
                    mx.save_safetensors(
                        str(path), {f"mtp.{k}": v for k, v in weights.items()}
                    )
                    self.assertTrue(
                        inject.inject_qwen4_exp_mtp_support(model, mtp_sidecar=path)
                    )
                actual = model.language_model.mtp.pre_fc_norm_embedding.weight
                np.testing.assert_array_equal(np.array(1 + actual), np.array(gamma))

    def test_mtp_without_admitted_backbone_refused(self):
        from vllm_mlx.spec_decode.mtp import qwen4_exp_inject as inject

        with mock.patch.object(nn, "quantize"):
            model = tiny_model()
            mtp = inject._build_mtp(model.language_model)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "mtp.safetensors"
                mx.save_safetensors(
                    str(path),
                    {f"mtp.{k}": v for k, v in tree_flatten(mtp.parameters())},
                )
                with self.assertLogs(inject.logger, level="ERROR"):
                    self.assertFalse(
                        inject.inject_qwen4_exp_mtp_support(model, mtp_sidecar=path)
                    )
                self.assertFalse(hasattr(model.language_model, "mtp"))


if __name__ == "__main__":
    unittest.main()
