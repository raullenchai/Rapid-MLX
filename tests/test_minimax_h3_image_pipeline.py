"""Real image orchestration with tiny local components; no weights or downloads."""

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

if sys.platform != "darwin":
    pytest.skip("MLX parity requires Darwin", allow_module_level=True)
mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_vlm.models.minimax_h3")

from rapid_mlx.image.minimax_h3_image import pipeline
from rapid_mlx.image.minimax_h3_image.layout import SIGMAS
from rapid_mlx.image.minimax_h3_image.vae import MiniMaxH3ImageVAE


@pytest.fixture
def tiny_pipeline(tmp_path, monkeypatch):
    (tmp_path / "pdd_config.json").write_text(
        json.dumps({"num_inference_steps": 6, "sigmas": SIGMAS})
    )
    import transformers

    calls = SimpleNamespace(
        loads=[], tokens=[], encoded=[], vision=[], rows=[], decoded=[], resizes=[]
    )
    resize = Image.Image.resize

    def record_resize(image, size, resample=None, **kwargs):
        calls.resizes.append((size, resample))
        return resize(image, size, resample=resample, **kwargs)

    monkeypatch.setattr(Image.Image, "resize", record_resize)

    class Tokenizer:
        def __call__(self, text, *, add_special_tokens):
            assert not add_special_tokens
            calls.tokens.append(text)
            return {"input_ids": [7, 8]}

        def convert_tokens_to_ids(self, text):
            return {"<|vision_start|>": 10, "<|image_pad|>": 11, "<|vision_end|>": 12}[
                text
            ]

    class Conditioner:
        tokenizer = Tokenizer()

        def _encode(self, ids, tags, **kwargs):
            calls.encoded.append((ids, tags, kwargs))
            return SimpleNamespace(hidden_states=mx.zeros((1, len(ids), 4)))

    class Processor:
        merge_size = 2

        @classmethod
        def from_pretrained(cls, path, *, local_files_only):
            assert path == tmp_path / "processor" and local_files_only
            return cls()

        def __call__(self, *, images, return_tensors):
            assert return_tensors == "np"
            calls.vision.extend(images)
            return {
                "pixel_values": np.zeros((8, 4), np.float32),
                "image_grid_thw": np.array([[1, 2, 2]] * len(images)),
            }

    class VAE:
        config = SimpleNamespace(latent_channels=2, spatial_compression_ratio=16)
        use_tiling = True
        tile_sample_min_height = tile_sample_min_width = 256
        tile_sample_min_overlap_height = tile_sample_min_overlap_width = 64
        _split_tiles = MiniMaxH3ImageVAE._split_tiles
        _stitch_tiles = MiniMaxH3ImageVAE._stitch_tiles
        _blend = staticmethod(MiniMaxH3ImageVAE._blend)
        _tiled = MiniMaxH3ImageVAE._tiled
        decode = MiniMaxH3ImageVAE.decode

        def encode(self, pixels):
            calls.encoded.append(pixels)
            h, w = pixels.shape[-2:]
            shape = (1, 2, h // 16, w // 16)
            return SimpleNamespace(mean=mx.ones(shape), logvar=mx.zeros(shape))

        def normalize(self, values):
            return values * 2

        def denormalize(self, values):
            calls.decoded.append(values)
            return values / 2

        def _decode_tile(self, values):
            h, w = values.shape[-2:]
            return mx.zeros((1, 3, h * 16, w * 16))

    def transformer(rows, text, values, indices, pos, tags, ii, ti, step):
        assert rows.shape[-1] == 8
        assert pos.dtype == mx.float64
        assert text.shape[1] == len(ti)
        assert len(tags) == len(indices) == len(pos)
        assert len(ii) == rows.shape[1]
        clock = np.array(values)[np.array(indices)]
        expected_clock = np.full(len(pos), np.float32(1) - np.float32(SIGMAS[step]))
        expected_clock[np.array(ii)[:-1]] = np.float32(0.999)
        np.testing.assert_allclose(clock, expected_clock, atol=1e-7)
        calls.rows.append(np.array(rows))
        assert step == len(calls.rows) - 1
        return mx.ones_like(rows)

    for name, component in [
        ("load_vae", VAE()),
        ("load_transformer", transformer),
        ("_load_conditioner", Conditioner()),
    ]:

        def load(*args, _name=name, _component=component):
            calls.loads.append((_name, args))
            return _component

        monkeypatch.setattr(pipeline, name, load)
    monkeypatch.setitem(vars(transformers), "Qwen2VLImageProcessor", Processor)
    return pipeline.MiniMaxH3Image(tmp_path, tmp_path / "aux", quantize=4), calls


@pytest.mark.parametrize("edit", [False, True])
def test_pipeline_six_step_generation_and_edit(tiny_pipeline, edit):
    model, calls = tiny_pipeline
    from rapid_mlx.image import minimax_h3_image

    assert minimax_h3_image.MiniMaxH3Image is pipeline.MiniMaxH3Image
    progress, cancellations = [], []
    model.on_step = lambda step, total: progress.append((step, total))
    model.load()
    model.load()
    assert calls.loads == [
        ("load_vae", (model.model_path,)),
        ("_load_conditioner", (model.model_path, model.text_encoder_path, 4)),
        ("load_transformer", (model.model_path, 4)),
    ]
    references = []
    if edit:
        rotated = Image.new("RGB", (64, 32), "red")
        rotated.getexif()[274] = 6
        references = [Image.new("RGB", (1024, 1024), "blue"), rotated]
    result = model.generate_image(
        "a fox",
        width=33,
        height=35,
        seed=19,
        references=references,
        min_generation_area=1024,
        cancel_check=lambda: cancellations.append(True),
    )
    assert result.size == (33, 35)
    assert result.mode == "RGB"
    assert calls.resizes[-1] == ((33, 35), Image.Resampling.LANCZOS)
    assert all(resample == Image.Resampling.LANCZOS for _, resample in calls.resizes)
    # Zero normalized decoder output maps to PIXEL_MEAN, including final resize.
    np.testing.assert_array_equal(np.array(result)[0, 0], [123, 116, 103])
    assert progress == [(step, 6) for step in range(7)]
    assert len(cancellations) == 7
    assert len(calls.rows) == 6
    nref = 3072 if edit else 0
    initial = calls.rows[0][0]
    for step, rows in enumerate(calls.rows):
        np.testing.assert_array_equal(rows[0, :nref], initial[:nref])
        np.testing.assert_allclose(
            rows[0, nref:], initial[nref:] + 1 - SIGMAS[step], atol=1e-6
        )
    final = np.array(calls.decoded[0]).reshape(-1)
    np.testing.assert_allclose(final, (initial[nref:] + 1).reshape(-1), atol=1e-6)
    if edit:
        assert [image.size for image in calls.vision] == [(1024, 1024), (1024, 2048)]
        assert calls.vision[1].getpixel((0, 0)) == (255, 0, 0)
        assert calls.tokens == ["Picture 1: ", "Picture 2: ", "a fox"]
        ids, tags, kwargs = calls.encoded[-1]
        assert ids == [7, 8, 10, 11, 12] * 2 + [7, 8]
        assert tags == [1, 1, 0, 0, 0] * 2 + [1, 1]
        assert kwargs["image_grid_thw"].shape == (2, 3)
        assert [p.shape for p in calls.encoded[:-1]] == [
            (1, 3, 1024, 1024),
            (1, 3, 2048, 1024),
        ]
        # Independently replay the seed draws to check FP16 posterior rounding
        # and the released 0.999 reference augmentation, before patchification.
        key = mx.random.key(19)

        def draw(shape):
            nonlocal key
            key, subkey = mx.random.split(key)
            return np.array(mx.random.normal(shape, key=subkey, dtype=mx.float32))

        posteriors = [
            (1 + draw(shape)).astype(np.float16).astype(np.float32) * 2
            for shape in [(1, 2, 64, 64), (1, 2, 128, 64)]
        ]
        draw((1, 2, 1, 2, 2))  # Target noise precedes reference augmentation.
        expected_rows = []
        for posterior in posteriors:
            t = np.float32(0.999)
            expected = (np.float32(1) - t) * draw(posterior.shape) + t * posterior
            h, w = expected.shape[-2:]
            expected_rows.append(
                expected.reshape(1, 2, h // 2, 2, w // 2, 2)
                .transpose(0, 2, 4, 1, 3, 5)
                .reshape(-1, 8)
            )
        np.testing.assert_allclose(
            initial[:nref], np.concatenate(expected_rows), atol=1e-6
        )

    else:
        assert calls.tokens == ["a fox"]
        assert calls.encoded[-1][2] == {"pixel_values": None, "image_grid_thw": None}


@pytest.mark.parametrize("when", [1, 3])
def test_pipeline_cancellation(tiny_pipeline, when):
    model, calls = tiny_pipeline
    checks = []

    class CancelledError(Exception):
        pass

    def cancel():
        checks.append(True)
        if len(checks) == when:
            raise CancelledError

    with pytest.raises(CancelledError):
        model.generate_image(
            "fox",
            width=32,
            height=32,
            seed=0,
            min_generation_area=1024,
            cancel_check=cancel,
        )
    assert len(calls.rows) == max(0, when - 2)
    assert not calls.decoded
    if when == 1:
        assert not calls.loads


@pytest.mark.parametrize("prompt", ["", None, 42])
def test_pipeline_invalid_prompt(tiny_pipeline, prompt):
    model, calls = tiny_pipeline
    with pytest.raises(ValueError, match="non-empty string"):
        model.generate_image(
            prompt, width=32, height=32, seed=0, min_generation_area=1024
        )
    assert not calls.loads


@pytest.mark.parametrize("override", [{"num_inference_steps": 5}, {"sigmas": [1, 0]}])
def test_pipeline_rejects_incompatible_pdd(tiny_pipeline, override):
    model, _ = tiny_pipeline
    (model.model_path / "pdd_config.json").write_text(
        json.dumps({"num_inference_steps": 6, "sigmas": SIGMAS, **override})
    )
    with pytest.raises(ValueError, match="six-step PDD"):
        model.load()


def test_conditioner_loader(tmp_path, monkeypatch):
    import transformers

    directory = tmp_path / "text_encoder"
    directory.mkdir()
    (directory / "config.json").write_text('{"tiny": true}')
    events = []
    config, qwen, tokenizer, result = object(), object(), object(), object()
    monkeypatch.setattr(
        pipeline,
        "_trim_conditioner_config",
        lambda c, **kw: events.append((c, kw)) or c,
    )
    monkeypatch.setattr(
        pipeline, "QwenConfig", SimpleNamespace(from_dict=lambda c: config)
    )
    monkeypatch.setattr(pipeline, "QwenModel", lambda c: qwen if c is config else None)
    monkeypatch.setattr(
        pipeline,
        "_conditioner_key_selected",
        lambda k, **kw: k == "keep" and kw == {"text_only": False},
    )
    monkeypatch.setattr(
        pipeline,
        "_prepare_conditioner_weights",
        lambda w, **kw: {"selected": w["keep"]} if kw == {"text_only": False} else {},
    )

    def load(model, path, *, bits, select, sanitize):
        assert model is qwen and path == directory and bits == 8
        assert select("keep") and not select("drop")
        assert sanitize({"keep": 1}) == {"selected": 1}

    def tokenize(path, **kwargs):
        assert path == tmp_path / "tokenizer"
        assert kwargs == {"local_files_only": True, "trust_remote_code": False}
        return tokenizer

    def conditioner(model, tokens, *, has_vision):
        assert model is qwen and tokens is tokenizer and has_vision
        return result

    monkeypatch.setattr(pipeline, "stream_load", load)
    monkeypatch.setitem(
        vars(transformers), "AutoTokenizer", SimpleNamespace(from_pretrained=tokenize)
    )
    monkeypatch.setattr(pipeline, "MiniMaxH3Conditioner", conditioner)
    assert pipeline._load_conditioner(tmp_path, tmp_path, 8) is result
    assert events == [({"tiny": True}, {"text_only": False})]
