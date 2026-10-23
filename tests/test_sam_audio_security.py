# SPDX-License-Identifier: Apache-2.0
"""SAM entry and lazy T5 routes must never deserialize legacy pickle weights."""

import json

import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("mlx_audio")
pytestmark = pytest.mark.requires_mlx


class _PickleSentinel:
    def __init__(self, marker):
        self.marker = str(marker)

    def __reduce__(self):
        return eval, (
            f"__import__('pathlib').Path({self.marker!r}).write_text('executed')",
        )


@pytest.mark.parametrize("target", ["sam", "t5"])
@pytest.mark.parametrize("unsafe_default", ["0", "1"])
def test_legacy_checkpoint_rejected_without_deserialization(
    tmp_path, monkeypatch, target, unsafe_default
):
    torch = pytest.importorskip("torch")
    from rapid_mlx.audio.processor import AudioProcessor
    from rapid_mlx.audio.sam_security import _load_t5

    marker = tmp_path / "executed"
    filename = "checkpoint.pt" if target == "sam" else "pytorch_model.bin"
    torch.save(_PickleSentinel(marker), tmp_path / filename)
    (tmp_path / "config.json").write_text("{}")
    monkeypatch.setenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", unsafe_default)
    invoke = (
        AudioProcessor(str(tmp_path)).load
        if target == "sam"
        else lambda: _load_t5(str(tmp_path))
    )
    with pytest.raises((__import__("pickle").UnpicklingError, RuntimeError)):
        invoke()
    assert not marker.exists()


def test_sam_config_redirected_text_encoder_uses_safe_lazy_route(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    import mlx.core as mx
    import mlx_audio.sts
    from mlx_audio.sts.models.sam_audio import model as sam_model
    from mlx_audio.sts.models.sam_audio.text_encoder import T5Encoder

    from rapid_mlx.audio.processor import AudioProcessor

    encoder = tmp_path / "redirected"
    encoder.mkdir()
    marker = encoder / "executed"
    torch.save(_PickleSentinel(marker), encoder / "pytorch_model.bin")
    (encoder / "config.json").write_text("{}")
    (tmp_path / "config.json").write_text(
        json.dumps({"text_encoder": {"name": "attacker/redirected-t5"}})
    )
    mx.save_safetensors(str(tmp_path / "model.safetensors"), {"safe": mx.ones((1,))})

    class FakeSAM:
        sample_rate = 44100

        def __init__(self, config):
            self.config = config
            self.processor = object()

        def post_load_hook(self, _path):
            return self

    downloads = []

    def resolve_snapshot(repo, **kwargs):
        downloads.append((repo, kwargs))
        return str(encoder)

    monkeypatch.setattr("rapid_mlx.model_downloads.snapshot_download", resolve_snapshot)
    monkeypatch.setattr(mlx_audio.sts, "SAMAudio", FakeSAM)
    monkeypatch.setattr(sam_model, "_load_weights", lambda model, weights: model)
    monkeypatch.setattr(
        T5Encoder,
        "from_pretrained",
        lambda *_a, **_kw: pytest.fail("unsafe upstream T5 loader reached"),
    )
    monkeypatch.setenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    processor = AudioProcessor(str(tmp_path))
    processor.load()
    with pytest.raises((__import__("pickle").UnpicklingError, RuntimeError)):
        processor.model.text_encoder._lazy_load()
    assert not marker.exists()
    assert processor.model.text_encoder.model is None
    assert [repo for repo, _kwargs in downloads] == ["attacker/redirected-t5"]
    assert {
        pattern
        for pattern in downloads[0][1]["allow_patterns"]
        if pattern.endswith(".bin")
    } == {"pytorch_model.bin"}


@pytest.mark.parametrize("legacy", [False, True])
def test_safe_t5_tensors_and_builtin_tokenizer_remain_supported(
    tmp_path, monkeypatch, legacy
):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_audio.sts.models.sam_audio.text_encoder import T5Config, T5Encoder
    from transformers import AutoTokenizer

    from rapid_mlx.audio.sam_security import _load_t5

    config = T5Config(
        vocab_size=8, d_model=4, d_kv=2, d_ff=8, num_layers=1, num_heads=2
    )
    from dataclasses import asdict

    (tmp_path / "config.json").write_text(json.dumps(asdict(config)))
    model = T5Encoder(config)
    model.shared.weight = mx.full((config.vocab_size, config.d_model), 0.375)
    mx.save_safetensors(
        str(tmp_path / "model.safetensors"), dict(tree_flatten(model.parameters()))
    )
    if legacy:
        import numpy as np
        import torch

        torch.save(
            {
                name: torch.from_numpy(np.array(value))
                for name, value in tree_flatten(model.parameters())
            },
            tmp_path / "pytorch_model.bin",
        )
        (tmp_path / "model.safetensors").unlink()
    else:
        # A legacy sidecar may coexist; a successful tensor load must ignore it.
        (tmp_path / "pytorch_model.bin").write_bytes(b"not a pickle")
    calls = []
    monkeypatch.setattr(
        AutoTokenizer,
        "from_pretrained",
        lambda *a, **kw: calls.append((a, kw)) or "tokenizer",
    )
    loaded, tokenizer = _load_t5(str(tmp_path))
    assert isinstance(loaded, T5Encoder)
    assert mx.array_equal(loaded.shared.weight, model.shared.weight).item()
    assert tokenizer == "tokenizer"
    assert calls == [
        ((str(tmp_path),), {"trust_remote_code": False, "local_files_only": True})
    ]


def test_corrupt_safetensors_does_not_fall_back_to_pickle(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from rapid_mlx.audio.sam_security import _load_t5

    marker = tmp_path / "executed"
    torch.save(_PickleSentinel(marker), tmp_path / "pytorch_model.bin")
    (tmp_path / "model.safetensors").write_bytes(b"invalid tensor header")
    (tmp_path / "config.json").write_text("{}")
    monkeypatch.setattr(
        torch, "load", lambda *_a, **_kw: pytest.fail("pickle fallback reached")
    )
    with pytest.raises((ValueError, RuntimeError)):
        _load_t5(str(tmp_path))
    assert not marker.exists()


@pytest.mark.parametrize("nested", [False, True])
def test_legacy_numeric_tensor_archives_remain_supported(tmp_path, nested):
    import mlx.core as mx
    import torch

    from rapid_mlx.audio.sam_security import _tensor_weights

    weights = {"weight": torch.arange(4, dtype=torch.float32)}
    path = tmp_path / "checkpoint.pt"
    torch.save({"state_dict": weights} if nested else weights, path)
    loaded = _tensor_weights(path)
    assert mx.array_equal(loaded["weight"], mx.arange(4, dtype=mx.float32)).item()


def test_invalid_config_and_missing_weights_refused(tmp_path):
    from rapid_mlx.audio.sam_security import _config, _require_weights

    (tmp_path / "config.json").write_text("[]")
    with pytest.raises(ValueError, match="JSON object"):
        _config(tmp_path)
    for text in (False, True):
        with pytest.raises(ValueError, match="no supported tensor"):
            _require_weights(tmp_path, text_encoder=text)


def test_non_tensor_legacy_state_refused(tmp_path):
    import torch

    from rapid_mlx.audio.sam_security import _tensor_weights

    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"weight": "not a tensor"}, checkpoint)
    with pytest.raises(ValueError, match="only named tensors"):
        _tensor_weights(checkpoint)


def test_sam_lazy_encoder_loaded_once(tmp_path, monkeypatch):
    import mlx.core as mx
    import mlx.nn as nn
    import mlx_audio.sts
    from mlx_audio.sts.models.sam_audio import model as sam_model

    from rapid_mlx.audio import sam_security

    (tmp_path / "config.json").write_text("{}")
    mx.save_safetensors(str(tmp_path / "model.safetensors"), {"weight": mx.ones((1,))})

    class Model(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.weight = mx.zeros((1,))
            self.processor = object()

        def post_load_hook(self, path):
            return self

        def sanitize(self, weights):
            return weights

    monkeypatch.setattr(mlx_audio.sts, "SAMAudio", Model)
    seen = []
    native_load_weights = sam_model._load_weights

    def load_weights(model, weights):
        seen.append(weights)
        return native_load_weights(model, weights)

    monkeypatch.setattr(sam_model, "_load_weights", load_weights)
    encoder, tokenizer = object(), object()
    calls = []
    monkeypatch.setattr(
        sam_security,
        "_load_t5",
        lambda name: calls.append(name) or (encoder, tokenizer),
    )
    model, processor = sam_security.load_sam_audio(str(tmp_path))
    model.text_encoder._lazy_load()
    model.text_encoder._lazy_load()
    assert model.training is False
    assert model.weight.tolist() == [1.0]
    assert len(calls) == 1
    assert model.text_encoder.model is encoder
    assert model.text_encoder.tokenizer is tokenizer
    assert processor is model.processor
    assert seen[0]["weight"].tolist() == [1.0]


def test_t5_with_real_local_tokenizer(tmp_path):
    from dataclasses import asdict

    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_audio.sts.models.sam_audio.text_encoder import T5Config, T5Encoder
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    from rapid_mlx.audio.sam_security import _load_t5

    config = T5Config(
        vocab_size=8, d_model=4, d_kv=2, d_ff=8, num_layers=1, num_heads=2
    )
    (tmp_path / "config.json").write_text(json.dumps(asdict(config)))
    model = T5Encoder(config)
    mx.save_safetensors(
        str(tmp_path / "model.safetensors"), dict(tree_flatten(model.parameters()))
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel({"<unk>": 0, "hello": 1}, unk_token="<unk>")
        ),
        unk_token="<unk>",
    )
    tokenizer.save_pretrained(tmp_path)
    loaded, tokenizer = _load_t5(str(tmp_path))
    assert isinstance(loaded, T5Encoder)
    assert tokenizer.encode("hello") == [1]


@pytest.mark.parametrize("explicit_activation", [False, True])
def test_native_t5_configuration_and_output_are_preserved(
    tmp_path, monkeypatch, explicit_activation
):
    from dataclasses import asdict

    import huggingface_hub
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_audio.sts.models.sam_audio.text_encoder import T5Config, T5Encoder
    from transformers import AutoTokenizer

    from rapid_mlx.audio.sam_security import _load_t5

    config = T5Config(
        vocab_size=8,
        d_model=4,
        d_kv=2,
        d_ff=8,
        num_layers=1,
        num_heads=2,
        relative_attention_num_buckets=8,
        relative_attention_max_distance=64,
        dropout_rate=0.0,
        is_gated_act=explicit_activation,
        dense_act_fn="gelu_new" if explicit_activation else "relu",
    )
    data = asdict(config)
    if not explicit_activation:
        del data["is_gated_act"], data["dense_act_fn"]
    # The native loader ignores these HF metadata/alias fields; retaining that
    # behavior matters for ordinary checkpoint config files.
    data.update(
        architectures=["T5ForConditionalGeneration"],
        feed_forward_proj="gated-gelu",
        num_decoder_layers=5,
    )
    (tmp_path / "config.json").write_text(json.dumps(data))
    source = T5Encoder(config)
    source.shared.weight = mx.full((config.vocab_size, config.d_model), 0.375)
    mx.save_safetensors(
        str(tmp_path / "model.safetensors"), dict(tree_flatten(source.parameters()))
    )
    requests = []

    def local_file(*, repo_id, filename):
        requests.append((repo_id, filename))
        assert filename in {"config.json", "model.safetensors"}
        return str(tmp_path / filename)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", local_file)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *_a, **_kw: object())
    native = T5Encoder.from_pretrained("native-contract-probe")
    owned, _tokenizer = _load_t5(str(tmp_path))
    assert asdict(owned.config) == asdict(native.config) == asdict(config)
    assert requests == [
        ("native-contract-probe", "config.json"),
        ("native-contract-probe", "model.safetensors"),
    ]
    assert mx.array_equal(owned.shared.weight, source.shared.weight).item()
    tokens = mx.array([[1, 2]])
    assert mx.allclose(owned(tokens), native(tokens)).item()
