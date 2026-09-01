# SPDX-License-Identifier: Apache-2.0
"""Pure mocked/AST checks for model-family continuous-MTP capability seams."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from vllm_mlx.spec_decode.mtp import qwen3_5_inject

ROOT = Path(__file__).resolve().parents[1]
FAMILY_MODULES = (qwen3_5_inject,)


@pytest.mark.parametrize(
    "module,family,dynamic_join",
    [(qwen3_5_inject, "qwen3_5", True)],
)
def test_capability_descriptor_is_explicit_immutable_and_conservative(
    module, family, dynamic_join
):
    capability = module.BATCHED_MTP_CAPABILITY
    assert dict(capability) == {
        "protocol_version": 1,
        "model_family": family,
        "batch_forward": "mtp_batch_forward",
        "recursive_draft_depth": 2,
        "fixed_membership": True,
        "target_return_hidden": True,
        "mtp_return_hidden": True,
        "confirmed_target_forward": True,
        "ragged_rollback": True,
        "atomic_cache_commit": True,
        "dynamic_join": dynamic_join,
        "flash_dynamic_membership_attested": False,
        "quantized_cache": False,
        "windowed_cache": False,
        "xtc": False,
    }
    with pytest.raises(TypeError):
        capability["dynamic_join"] = not dynamic_join


@pytest.mark.parametrize("module", FAMILY_MODULES)
def test_batch_forward_delegates_to_existing_recursive_hidden_path(module):
    calls = []

    class FakeInjectedModel:
        def mtp_forward(self, hidden, tokens, cache, *, return_hidden=False):
            calls.append((hidden, tokens, cache, return_hidden))
            return "logits", "next-hidden"

    result = module._mtp_batch_forward(FakeInjectedModel(), "hidden", "tokens", "cache")
    assert result == ("logits", "next-hidden")
    assert calls == [("hidden", "tokens", "cache", True)]


@pytest.mark.parametrize(
    "relative_path,injected_class",
    [
        ("vllm_mlx/spec_decode/mtp/qwen3_5_inject.py", "_Qwen3_5WithMTP"),
    ],
)
def test_injected_class_exposes_descriptor_seam_and_separate_recursive_depth(
    relative_path, injected_class
):
    tree = ast.parse((ROOT / relative_path).read_text(encoding="utf-8"))
    cls = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == injected_class
    )
    assignments = {
        target.id: node.value
        for node in cls.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }

    assert isinstance(assignments["batched_mtp_capability"], ast.Name)
    assert assignments["batched_mtp_capability"].id == "BATCHED_MTP_CAPABILITY"
    assert isinstance(assignments["mtp_batch_forward"], ast.Name)
    assert assignments["mtp_batch_forward"].id == "_mtp_batch_forward"
    depth = assignments["mtp_recursive_draft_depth"]
    assert isinstance(depth, ast.Constant) and depth.value == 2
    # The new recursive batch depth is deliberately not implemented by
    # changing the legacy single-request cap on the injected class.
    assert "mtp_max_speculative_tokens" not in assignments


def test_qwen35_attests_cycle_boundary_dynamic_join():
    assert qwen3_5_inject.BATCHED_MTP_CAPABILITY["dynamic_join"] is True


def test_validator_requires_the_full_continuous_mtp_surface():
    class _Valid:
        args = object()
        model = object()
        mtp = object()
        batched_mtp_capability = qwen3_5_inject.BATCHED_MTP_CAPABILITY
        mtp_recursive_draft_depth = 2

        def __call__(self, inputs, *, return_hidden=False, n_confirmed=0):
            return inputs, return_hidden, n_confirmed

        def mtp_forward(self, *args, **kwargs):
            return args, kwargs

        def mtp_batch_forward(self, *args, **kwargs):
            return args, kwargs

        def make_mtp_cache(self):
            return []

    assert qwen3_5_inject.validate_mtp_support(_Valid()) is True

    for attribute in (
        "mtp_batch_forward",
        "batched_mtp_capability",
        "mtp_recursive_draft_depth",
    ):
        candidate = _Valid()
        setattr(candidate, attribute, None)
        assert qwen3_5_inject.validate_mtp_support(candidate) is False, attribute
