from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from vllm_mlx.speculative.native_mtp import glm5_compat


def _install_fake_runtime(monkeypatch, released_type):
    mlx = ModuleType("mlx")
    mlx.__path__ = []
    core = ModuleType("mlx.core")
    for name in (
        "arange",
        "array",
        "concatenate",
        "maximum",
        "take_along_axis",
        "where",
        "zeros_like",
    ):
        setattr(core, name, getattr(np, name))
    core.int32 = np.int32
    core.inf = np.inf
    mlx.core = core

    models = ModuleType("mlx_vlm.models")
    models.__path__ = []
    linear_module = ModuleType("mlx_vlm.models.linear")
    linear_module.linear = lambda layer, value: layer(value)

    speculative = ModuleType("mlx_vlm.speculative")
    speculative.__path__ = []
    drafters = ModuleType("mlx_vlm.speculative.drafters")
    drafters.__path__ = []
    package = ModuleType("mlx_vlm.speculative.drafters.glm5_next_mtp")
    package.__path__ = []
    implementation = ModuleType(
        "mlx_vlm.speculative.drafters.glm5_next_mtp.glm5_next_mtp"
    )
    package.Glm5NextMTPDraftModel = released_type
    package.glm5_next_mtp = implementation
    drafters.glm5_next_mtp = package

    for name, module in {
        "mlx": mlx,
        "mlx.core": core,
        "mlx_vlm.models": models,
        "mlx_vlm.models.linear": linear_module,
        "mlx_vlm.speculative": speculative,
        "mlx_vlm.speculative.drafters": drafters,
        "mlx_vlm.speculative.drafters.glm5_next_mtp": package,
        implementation.__name__: implementation,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    return package, implementation


def test_stateless_drafter_detection_covers_marker_signature_and_invalid_call():
    marked = type("Marked", (), {"_RAPID_STATELESS_GLM_MTP": True})

    class Compatible:
        def __call__(self, tokens, hidden, cache, position, target_model):
            pass

    assert glm5_compat._is_stateless_drafter(marked) is True
    assert glm5_compat._is_stateless_drafter(Compatible) is True
    assert (
        glm5_compat._is_stateless_drafter(SimpleNamespace(__call__=object())) is False
    )


def test_install_is_noop_for_future_stateless_upstream(monkeypatch):
    class FutureDrafter:
        def __call__(self, tokens, hidden, cache, position, target_model):
            pass

    package, _ = _install_fake_runtime(monkeypatch, FutureDrafter)
    monkeypatch.setattr(glm5_compat, "_INSTALLED", False)

    assert glm5_compat.install_glm5_mtp_compatibility() is False
    assert package.Glm5NextMTPDraftModel is FutureDrafter
    assert glm5_compat.is_installed() is True


def test_installed_adapter_binds_and_runs_both_output_heads(monkeypatch):
    class ReleasedDrafter:
        def validate_target_compatibility(self, target):
            self.validated = target

    package, implementation = _install_fake_runtime(monkeypatch, ReleasedDrafter)
    monkeypatch.setattr(glm5_compat, "_INSTALLED", False)

    assert glm5_compat.install_glm5_mtp_compatibility() is True
    adapted = package.Glm5NextMTPDraftModel
    assert adapted is implementation.Glm5NextMTPDraftModel
    assert adapted is package.Model
    assert adapted.__name__ == "Glm5NextMTPDraftModel"
    assert glm5_compat.is_installed() is True

    drafter = adapted()
    drafter.args = SimpleNamespace(hidden_size=2)
    drafter.enorm = drafter.hnorm = drafter.shared_head_norm = lambda value: value
    drafter.eh_proj = lambda value: value[..., :2]

    class Attention:
        def __call__(self, value, *, cache, padding_mask):
            assert cache == "draft-cache"
            assert padding_mask.shape == value.shape[:2]
            return np.zeros_like(value), None

    drafter.mtp_block = SimpleNamespace(
        input_layernorm=lambda value: value,
        self_attn=Attention(),
        post_attention_layernorm=lambda value: value,
        mlp=lambda value: np.zeros_like(value),
    )

    class Embeddings:
        def __call__(self, tokens):
            return np.repeat(tokens[..., None], 2, axis=-1).astype(np.float32)

        def as_linear(self, value):
            return value.sum(axis=-1, keepdims=True)

    embeddings = Embeddings()
    target = SimpleNamespace(
        args=SimpleNamespace(tie_word_embeddings=True),
        model=SimpleNamespace(embed_tokens=embeddings),
        lm_head=lambda value: value[..., :1],
    )
    assert drafter.bind(target) is drafter
    assert drafter.validated is target

    tokens = np.array([[1, 2]], dtype=np.int32)
    hidden = np.ones((1, 2, 2), dtype=np.float32)
    tied_logits, tied_hidden = drafter(tokens, hidden, ["draft-cache"], [0], target)
    assert tied_logits.shape == (1, 1, 1)
    assert tied_hidden.shape == hidden.shape

    target.args.tie_word_embeddings = False
    untied_logits, _ = drafter(
        tokens, hidden, ["draft-cache"], [0], target, lengths=[1]
    )
    assert untied_logits.shape == (1, 1, 1)

    with pytest.raises(ValueError, match="hidden states"):
        drafter(tokens, hidden[..., :1], ["draft-cache"], [0], target)
