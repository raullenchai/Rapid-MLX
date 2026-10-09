# SPDX-License-Identifier: Apache-2.0
"""Native MLX Clef head, prompt layout, loader and backend without the checkpoint."""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.requires_mlx
mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
pytest.importorskip("mlx_vlm")

from rapid_mlx.cli import _resolve_system_one_backend, build_parser  # noqa: E402
from rapid_mlx.system_one import clef_mlx  # noqa: E402
from rapid_mlx.system_one.backends import ClefMLXBackend  # noqa: E402
from rapid_mlx.system_one.clef_mlx import (  # noqa: E402
    ClefScorer,
    JointSchemaHead,
    Model,
    ModelConfig,
    _options,
    _render,
    clef_answer,
    load_clef,
)
from rapid_mlx.system_one.schema import Question  # noqa: E402
from rapid_mlx.system_one.server import create_app  # noqa: E402

HIDDEN = 32
HEAD = {
    "hidden_size": HIDDEN,
    "width": 8,
    "routing_layers": 1,
    "layers": 1,
    "heads": 2,
    "feedforward": 16,
}
IMAGE_TOKEN, VIDEO_TOKEN = 60, 61


def test_head_scores_every_option_of_every_question():
    mx.random.seed(0)
    head = JointSchemaHead(**HEAD)
    hidden = mx.random.normal((12, HIDDEN))
    lexical = mx.random.normal((5, HIDDEN))
    logits = head(
        hidden,
        [(1, 3), (5, 6)],
        [(2, 3), (3, 5), (6, 7), (7, 9), (9, 12)],
        lexical,
        mx.array([0, 1]),
        [2, 3],
    )
    assert logits.shape == (5,)
    assert np.isfinite(np.asarray(logits)).all()
    # The head reads option text through the spans: pointing the last option
    # at different tokens changes that question's logits.
    changed = head(
        hidden,
        [(1, 3), (5, 6)],
        [(2, 3), (3, 5), (6, 7), (7, 9), (10, 12)],
        lexical,
        mx.array([0, 1]),
        [2, 3],
    )
    assert not np.allclose(np.asarray(logits[2:]), np.asarray(changed[2:]))


class FakeBackbone:
    def __init__(self):
        mx.random.seed(1)
        self.embed_tokens = nn.Embedding(64, HIDDEN)
        self.calls = []

    def __call__(self, input_ids, inputs_embeds, position_ids):
        self.calls.append((inputs_embeds, position_ids))
        return inputs_embeds


class FakeLanguageModel:
    def __init__(self, quantized):
        self.model = FakeBackbone()
        linear = nn.Linear(HIDDEN, 64, bias=False)
        self.lm_head = (
            nn.QuantizedLinear.from_linear(linear, group_size=32, bits=8)
            if quantized
            else linear
        )
        self.dense_head = linear
        self.quant_predicate = None
        self.rope_calls = []

    def get_rope_index(self, input_ids, image_grid, video_grid):
        self.rope_calls.append((image_grid, video_grid))
        return "positions", None


class FakeVisionTower:
    def __init__(self):
        self.patch_embed = SimpleNamespace(
            proj=SimpleNamespace(weight=mx.zeros((1,), dtype=mx.float32))
        )
        self.calls = []

    def __call__(self, pixels, grid):
        self.calls.append((pixels, grid))
        return mx.full((int(grid[0, 0]), HIDDEN), float(len(self.calls))), None


def _model(monkeypatch, *, quantized=False):
    monkeypatch.setattr(
        clef_mlx.Qwen3_5Model, "__init__", lambda self, config: nn.Module.__init__(self)
    )
    model = Model(SimpleNamespace(head_config=HEAD))
    model.config = SimpleNamespace(
        image_token_index=IMAGE_TOKEN,
        video_token_index=VIDEO_TOKEN,
        text_config=SimpleNamespace(tie_word_embeddings=False),
        vision_config=SimpleNamespace(in_channels=3),
    )
    object.__setattr__(model, "language_model", FakeLanguageModel(quantized))
    object.__setattr__(model, "vision_tower", FakeVisionTower())
    return model


def _call(model, ids, **media):
    return model(
        mx.array([ids]),
        [((1, 3), 2), ((5, 6), 1)],
        [(6, 8), (8, 9), (9, 11)],
        mx.array([0, 1]),
        **media,
    )


def test_model_builds_the_head_and_scores_text(monkeypatch):
    model = _model(monkeypatch)
    assert isinstance(model.head, JointSchemaHead)
    ids = list(range(1, 13))
    seen = {}
    real_head = model.head

    class Recorder:
        hidden_norm = real_head.hidden_norm

        def __call__(self, hidden, question_spans, option_spans, lexical, types, n):
            seen.update(
                question_spans=question_spans, option_spans=option_spans, counts=n
            )
            seen["lexical"] = lexical
            return real_head(hidden, question_spans, option_spans, lexical, types, n)

    object.__setattr__(model, "head", Recorder())
    logits = _call(model, ids)
    assert logits.shape == (3,)
    assert seen["question_spans"] == [(1, 3), (5, 6)]
    assert seen["counts"] == [2, 1]
    # Each option's lexical vector is the mean output embedding of its tokens.
    weight = model.language_model.dense_head.weight
    expected = mx.stack(
        [
            weight[mx.array([7, 8])].mean(axis=0),
            weight[mx.array([9])].mean(axis=0),
            weight[mx.array([10, 11])].mean(axis=0),
        ]
    )
    assert np.allclose(np.asarray(seen["lexical"]), np.asarray(expected), atol=1e-6)
    assert model.language_model.rope_calls == [(None, None)]
    assert model.language_model.model.calls[0][1] == "positions"
    assert model.vision_tower.calls == []


def test_model_reads_option_embeddings_from_a_quantized_output_layer(monkeypatch):
    dense = _model(monkeypatch)
    quantized = _model(monkeypatch, quantized=True)
    ids = list(range(1, 13))
    seen = []

    for model in (dense, quantized):
        real_head = model.head

        class Recorder:
            hidden_norm = real_head.hidden_norm

            def __call__(self, hidden, q, o, lexical, types, n):
                seen.append(np.asarray(lexical))
                return mx.zeros((3,))

        object.__setattr__(model, "head", Recorder())
        _call(model, ids)
    # 8-bit rows dequantize back to nearly the dense rows they came from.
    assert seen[1].shape == (3, HIDDEN)
    assert np.allclose(seen[0], seen[1], atol=2e-2)


def test_model_places_image_and_video_features_on_their_tokens(monkeypatch):
    model = _model(monkeypatch)
    ids = [1, IMAGE_TOKEN, IMAGE_TOKEN, 4, VIDEO_TOKEN, 6, 7, 8, 9, 10, 11, 12]
    logits = _call(
        model,
        ids,
        pixel_values=mx.zeros((2, 4)),
        image_grid_thw=mx.array([[2, 1, 1]]),
        pixel_values_videos=mx.zeros((1, 4)),
        video_grid_thw=mx.array([[1, 1, 1]]),
    )
    assert logits.shape == (3,)
    embeds = np.asarray(model.language_model.model.calls[0][0])[0]
    assert np.allclose(embeds[1], 1.0) and np.allclose(embeds[2], 1.0)
    assert np.allclose(embeds[4], 2.0)
    assert not np.allclose(embeds[3], 1.0)
    image_grid, video_grid = model.language_model.rope_calls[0]
    assert image_grid.tolist() == [[2, 1, 1]] and video_grid.tolist() == [[1, 1, 1]]


def test_sanitize_maps_release_head_names_onto_the_mlx_head(monkeypatch):
    model = _model(monkeypatch)
    width = HEAD["width"]
    weights = {
        "language_model.model.layers.0.note": mx.zeros((2,)),
        "lm_head.weight": mx.zeros((2, 2)),
        "head.evidence_layers.0.attention.in_proj_weight": mx.arange(
            3 * width * width
        ).reshape(3 * width, width),
        "evidence_layers.0.attention.in_proj_bias": mx.arange(3 * width),
        "head.evidence_layers.0.feedforward.3.weight": mx.zeros((1,)),
        "residual_scorer.3.bias": mx.zeros((1,)),
        "head.field_norm.weight": mx.ones((width,)),
    }
    sanitized = model.sanitize(weights)
    attention = "head.evidence_layers.0.attention"
    assert sanitized[f"{attention}.query_proj.weight"].shape == (width, width)
    assert sanitized[f"{attention}.key_proj.bias"].tolist() == list(
        range(width, 2 * width)
    )
    assert sanitized[f"{attention}.value_proj.bias"].tolist() == list(
        range(2 * width, 3 * width)
    )
    assert "head.evidence_layers.0.feedforward.1.weight" in sanitized
    assert "head.residual_scorer.1.bias" in sanitized
    assert "head.field_norm.weight" in sanitized
    assert not any("in_proj" in key or ".3." in key for key in sanitized)
    # Backbone tensors still go through the Qwen3.5 sanitizer.
    assert len([key for key in sanitized if not key.startswith("head.")]) == 2


def test_quantization_never_touches_the_head(monkeypatch):
    model = _model(monkeypatch)
    predicate = model.quant_predicate
    assert predicate("head.field_norm", None) is False
    assert predicate("language_model.model.layers.0", None) is True
    model.language_model.quant_predicate = lambda path, module: {"bits": 8}
    predicate = model.quant_predicate
    assert predicate("language_model.lm_head", None) == {"bits": 8}
    assert predicate("head.layers.0.linear1", None) is False


def test_config_carries_the_head_configuration():
    assert ModelConfig.__dataclass_fields__["head_config"].default_factory() == {}


def test_options_follow_the_release_order_and_defaults():
    assert _render("plain") == "plain"
    assert _render({"b": 1, "a": "é"}) == '{"a":"é","b":1}'
    assert _options({"type": "noul"}) == [
        ("true", "The proposition is true or the answer is yes."),
        ("false", "The proposition is false or the answer is no."),
    ]
    assert _options({"type": "noul", "criteria": {"true": "T"}})[0] == ("true", "T")
    assert _options({"type": "choice", "criteria": {"b": None, "a": "first"}}) == [
        ("a", "first"),
        ("b", None),
    ]
    assert _options({"type": "score", "criteria": ["low", "high"]}) == [
        ("0", "low"),
        ("1", "high"),
    ]


def test_answers_use_the_release_shape_and_rounding():
    assert clef_answer({"type": "noul"}, {"true": 0.123456, "false": 0.876544}) == {
        "type": "noul",
        "noul": 0.1235,
    }
    choice = clef_answer(
        {"type": "choice", "criteria": {"b": None, "a": None}},
        {"a": 0.25004, "b": 0.74996},
    )
    assert choice == {
        "type": "choice",
        "choice": "b",
        "confidence": 0.75,
        "probabilities": {"b": 0.75, "a": 0.25},
    }
    score = clef_answer(
        {"type": "score", "criteria": ["low", "mid", "high"]},
        {"0": 0.1, "1": 0.3, "2": 0.6},
    )
    assert score == {
        "type": "score",
        "score": 1.5,
        "confidence": 0.6,
        "legend": {"0": "low", "1": "mid", "2": "high"},
        "probabilities": {"0": 0.1, "1": 0.3, "2": 0.6},
    }


def _write_config(path: Path, **changes):
    config = {"model_type": "clef", "head_config": dict(HEAD), **changes}
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")


def test_load_clef_validates_checkpoint_metadata(tmp_path):
    with pytest.raises(ValueError, match="no config.json"):
        load_clef(tmp_path)
    _write_config(tmp_path, model_type="qwen3_5")
    with pytest.raises(ValueError, match="model_type 'clef'"):
        load_clef(tmp_path)
    _write_config(tmp_path, head_config=None)
    with pytest.raises(ValueError, match="needs a head_config"):
        load_clef(tmp_path)
    _write_config(tmp_path, head_config={**HEAD, "extra": 1})
    with pytest.raises(ValueError, match="needs a head_config"):
        load_clef(tmp_path)
    for bad in (0, True, 1.5, "8"):
        _write_config(tmp_path, head_config={**HEAD, "width": bad})
        with pytest.raises(ValueError, match="positive integers"):
            load_clef(tmp_path)


def test_load_clef_registers_the_model_type_and_materializes_weights(
    tmp_path, monkeypatch
):
    import mlx_vlm

    _write_config(tmp_path)
    monkeypatch.setitem(sys.modules, "mlx_vlm.models.clef", None)
    patched = []
    monkeypatch.setattr(
        clef_mlx,
        "install_auto_processor_patch",
        lambda model_type, processor: patched.append((model_type, processor)),
    )
    # A sum is lazy until evaluated, and a lazy array is bound to its thread:
    # without the load-time evaluation the worker below raises.
    weight = mx.zeros((2,)) + 1

    def fake_load(path):
        # mlx-vlm resolves the model type through this registry entry.
        assert sys.modules["mlx_vlm.models.clef"] is clef_mlx
        assert path == str(tmp_path)
        return SimpleNamespace(parameters=lambda: {"weight": weight}), "processor"

    monkeypatch.setattr(mlx_vlm, "load", fake_load)
    model, processor = load_clef(str(tmp_path))
    assert processor == "processor"
    assert patched == [("clef", clef_mlx.Qwen3VLProcessor)]
    # Requests run on worker threads; a still-lazy weight would fail there.
    result = []
    thread = threading.Thread(
        target=lambda: result.append(model.parameters()["weight"].tolist())
    )
    thread.start()
    thread.join()
    assert result == [[1.0, 1.0]]


class FakeTokenizer:
    """One token per character, so spans can be read back as text."""

    def __call__(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return {"input_ids": [ord(char) for char in text]}


class FakeProcessor:
    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.video_processor = SimpleNamespace(fps=2, min_frames=2, max_frames=4)
        self.calls = []

    def __call__(self, text, images=None, videos=None, **kwargs):
        self.calls.append({"text": text, "images": images, "videos": videos, **kwargs})
        encoded = {"input_ids": np.array([[901, 902, 903]])}
        if images:
            encoded["pixel_values"] = np.zeros((2, 4), dtype=np.float32)
            encoded["image_grid_thw"] = np.array([[1, 1, 2]])
        if videos:
            encoded["pixel_values_videos"] = np.zeros((1, 4), dtype=np.float32)
            encoded["video_grid_thw"] = np.array([[1, 1, 1]])
        return encoded


class FakeClefModel:
    def __init__(self):
        self.calls = []

    def __call__(self, input_ids, question_spans, option_spans, types, **media):
        self.calls.append(
            {
                "ids": input_ids[0].tolist(),
                "question_spans": question_spans,
                "option_spans": option_spans,
                "types": types.tolist(),
                "media": media,
            }
        )
        return mx.arange(len(option_spans)).astype(mx.float32)


QUESTIONS = {
    "pick": {
        "type": "choice",
        "instructions": "Pick one",
        "criteria": {"b": None, "a": "first"},
    },
    "sure": {"type": "noul"},
    "rate": {"type": "score", "instructions": {"ask": "Rate"}, "criteria": ["l", "h"]},
}


def _text(ids, span):
    return "".join(chr(token) for token in ids[span[0] : span[1]])


def test_scorer_lays_out_the_release_prompt_and_spans():
    model, processor = FakeClefModel(), FakeProcessor()
    scorer = ClefScorer(model, processor)
    answers, tokens = scorer.score({"k": "state"}, QUESTIONS)
    call = model.calls[0]
    ids = call["ids"]
    assert tokens == len(ids)
    prompt = "".join(chr(token) for token in ids)
    assert prompt.startswith("<|im_start|>system\nRead the complete state and schema.")
    assert 'STATE:\n{"k":"state"}\n\nSCHEMA FIELDS:\n\nFIELD 1\nID: pick' in prompt
    assert prompt.endswith("<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:")
    assert call["types"] == [1, 0, 2]
    spans = [span for span, _ in call["question_spans"]]
    # A question without instructions is asked by its own name.
    assert [_text(ids, span) for span in spans] == [
        "Pick one",
        "sure",
        '{"ask":"Rate"}',
    ]
    assert [count for _, count in call["question_spans"]] == [2, 2, 2]
    options = [_text(ids, span) for span in call["option_spans"]]
    assert options[:2] == [
        '{"description":"first","option_id":"a"}',
        '{"option_id":"b"}',
    ]
    assert json.loads(options[2])["option_id"] == "true"
    assert options[-1] == '{"description":"h","option_id":"1"}'
    # Options are scored per question, in the release's option order.
    assert list(answers) == ["pick", "sure", "rate"]
    assert list(answers["pick"]) == ["a", "b"]
    assert list(answers["sure"]) == ["true", "false"]
    for distribution in answers.values():
        assert sum(distribution.values()) == pytest.approx(1.0)
        low, high = distribution.values()
        assert high > low
    assert call["media"] == {}
    assert processor.calls == []


def test_scorer_truncates_the_state_and_rejects_an_oversized_schema(monkeypatch):
    model = FakeClefModel()
    scorer = ClefScorer(model, FakeProcessor())
    _, fixed = scorer.score("", {"sure": {"type": "noul"}})
    monkeypatch.setattr(clef_mlx, "MAX_LENGTH", fixed + 5)
    _, tokens = scorer.score("x" * 50, {"sure": {"type": "noul"}})
    assert tokens == fixed + 5
    call = model.calls[-1]
    prompt = "".join(chr(token) for token in call["ids"])
    assert "STATE:\nxxxxx\n\nSCHEMA FIELDS:" in prompt
    assert _text(call["ids"], call["question_spans"][0][0]) == "sure"
    monkeypatch.setattr(clef_mlx, "MAX_LENGTH", fixed - 1)
    with pytest.raises(ValueError, match="maximum is"):
        scorer.score("x", {"sure": {"type": "noul"}})


def test_scorer_passes_images_and_sampled_video_frames_to_the_processor():
    model, processor = FakeClefModel(), FakeProcessor()
    scorer = ClefScorer(model, processor)
    frames = [f"frame{index}" for index in range(48)]
    short = ["a", "b", "c"]
    original = [list(frames), list(short)]
    scorer.score("s", {"sure": {"type": "noul"}}, ["img1", "img2"], original)
    sent = processor.calls[0]
    assert sent["images"] == ["img1", "img2"]
    assert sent["text"] == [
        "<|vision_start|><|image_pad|><|vision_end|>" * 2
        + "<|vision_start|><|vision_start|><|video_pad|><|vision_end|><|vision_end|>"
        * 2
        + "\n"
    ]
    # 48 frames at 24 fps are two seconds: four frames at the processor's fps,
    # which is also its cap. Three frames stay above its minimum of two.
    assert sent["videos"][0] == ["frame0", "frame16", "frame31", "frame47"]
    assert sent["videos"][1] == ["a", "c"]
    assert sent["video_metadata"] == [
        {"frames_indices": [0, 16, 31, 47], "fps": 24},
        {"frames_indices": [0, 2], "fps": 24},
    ]
    # The caller's frame lists are not edited.
    assert original == [frames, short]
    call = model.calls[0]
    assert set(call["media"]) == {
        "pixel_values",
        "image_grid_thw",
        "pixel_values_videos",
        "video_grid_thw",
    }
    marker = call["ids"].index(901)
    assert call["ids"][marker : marker + 4] == [901, 902, 903, ord("s")]

    scorer.score("s", {"sure": {"type": "noul"}}, ["img"], None)
    assert "video_metadata" not in processor.calls[1]
    assert processor.calls[1]["videos"] is None
    assert set(model.calls[1]["media"]) == {"pixel_values", "image_grid_thw"}


def _backend(scored, tokens=42):
    backend = object.__new__(ClefMLXBackend)
    backend.default_model = "clef-flash-mlx"
    backend.repo_id = "nativ-community/clef-flash-MLX-MXFP4"
    backend._lock = threading.Lock()
    calls = []

    def score(state, questions, images=None, videos=None):
        calls.append((state, questions, images, videos))
        return scored, tokens

    backend._scorer = SimpleNamespace(score=score, calls=calls)
    return backend


def test_backend_answers_in_the_clef_wire_shape(monkeypatch):
    import rapid_mlx.clef.media as media_module

    questions = {
        "pick": Question(
            type="choice", instructions="Pick", criteria={"b": None, "a": None}
        ),
        "sure": Question(type="noul", instructions="Sure?"),
    }
    backend = _backend(
        {"pick": {"a": 0.2, "b": 0.8}, "sure": {"true": 0.7, "false": 0.3}}
    )
    result = backend.answer("state", questions, "clef-flash-mlx", 1.0)
    assert result == {
        "model": "clef-flash-mlx",
        "answers": {
            "pick": {
                "type": "choice",
                "choice": "b",
                "confidence": 0.8,
                "probabilities": {"b": 0.8, "a": 0.2},
            },
            "sure": {"type": "noul", "noul": 0.7},
        },
        "usage": {"billing_units": 2, "input_tokens": 42, "output_tokens": 0},
    }
    state, specs, images, videos = backend._scorer.calls[0]
    assert specs["sure"] == {"type": "noul", "instructions": "Sure?"}
    assert images is None and videos is None

    monkeypatch.setattr(
        media_module,
        "decode_media",
        lambda images, videos: ([f"decoded:{images[0]}"], None),
    )
    backend.answer_media(
        "state", questions, backend.repo_id, 1.0, ["data:image/png;base64,AA"], None
    )
    assert backend._scorer.calls[1][2] == ["decoded:data:image/png;base64,AA"]

    with pytest.raises(KeyError, match="unknown model"):
        backend.answer("s", questions, "other", 1.0)
    with pytest.raises(ValueError, match="temperature=1"):
        backend.answer("s", questions, "clef-flash-mlx", 0.5)


def test_backend_ranks_from_unrounded_probabilities_and_lists_models():
    backend = _backend({"rank": {"0": 0.30001, "1": 0.39998, "2": 0.30002}})
    ranked = backend.rank("ctx", None, ["x", "y", "z"], "clef-flash-mlx", 1.0)
    assert [item["candidate"] for item in ranked] == ["y", "z", "x"]
    assert [item["rank"] for item in ranked] == [1, 2, 3]
    assert backend._scorer.calls[0][1]["rank"] == {
        "type": "choice",
        "instructions": "Choose the best answer.",
        "criteria": {"0": "x", "1": "y", "2": "z"},
    }
    assert backend.models() == [
        {
            "name": "clef-flash-mlx",
            "backend": "clef-mlx",
            "hf_id": "nativ-community/clef-flash-MLX-MXFP4",
            "description": "Cloudflare Clef typed decisions on native MLX",
        }
    ]
    client = TestClient(create_app(backend))
    response = client.post(
        "/v1/rank", json={"context": "c", "question": "q", "answers": ["x", "y", "z"]}
    )
    assert response.status_code == 200
    assert response.json()["ranked"][0]["candidate"] == "y"
    assert backend._scorer.calls[1][1]["rank"]["instructions"] == "q"


def test_backend_resolves_pinned_aliases_and_local_directories(tmp_path, monkeypatch):
    import importlib.util

    import rapid_mlx._mirror as mirror

    loaded = []
    monkeypatch.setattr(
        mirror,
        "pinned_snapshot_download",
        lambda repo, revision: f"/snap/{repo}@{revision}",
    )
    monkeypatch.setattr(
        clef_mlx, "load_clef", lambda path: loaded.append(path) or ("model", "proc")
    )
    monkeypatch.setattr(
        clef_mlx, "ClefScorer", lambda *args: SimpleNamespace(args=args)
    )
    devices = []
    monkeypatch.setattr(mx, "set_default_device", devices.append)

    for name in (
        "clef-flash-mlx",
        "Clef-Flash-MLX",
        "nativ-community/clef-flash-MLX-MXFP4",
    ):
        backend = ClefMLXBackend(name)
        assert backend.default_model == "clef-flash-mlx"
        assert backend.repo_id == "nativ-community/clef-flash-MLX-MXFP4"
        assert backend._scorer.args == ("model", "proc")
    assert loaded[0] == (
        "/snap/nativ-community/clef-flash-MLX-MXFP4"
        "@63a0d4df0213be9843968151609e05c2f6683870"
    )
    large = ClefMLXBackend("clef-mlx")
    assert large.repo_id == "nativ-community/clef-MLX-MXFP4"
    assert loaded[-1].endswith("@b94c97b0d0b80fdda6d7d36c745d289890d1c4e1")
    # A same-named directory in the working directory does not shadow it.
    (tmp_path / "clef-mlx").mkdir()
    monkeypatch.chdir(tmp_path)
    assert ClefMLXBackend("clef-mlx").repo_id == "nativ-community/clef-MLX-MXFP4"
    assert ClefMLXBackend("./clef-mlx").default_model == "clef-mlx"
    assert loaded[-1] == "clef-mlx"
    local = ClefMLXBackend(str(tmp_path), device="cpu")
    assert local.default_model == tmp_path.name
    assert local.repo_id == str(tmp_path)
    assert loaded[-1] == str(tmp_path)
    assert devices == [mx.gpu] * 6 + [mx.cpu]

    # The Torch release names belong to the other backend.
    for unknown in ("clef-flash", "Cloudflare/clef", "someone/clef-flash-mlx"):
        with pytest.raises(ValueError, match="unknown native Clef model"):
            ClefMLXBackend(unknown)
    with pytest.raises(ValueError, match="'gpu' or 'cpu'"):
        ClefMLXBackend("clef-mlx", device="npu")
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    with pytest.raises(RuntimeError, match=r"rapid-mlx\[vision\]"):
        ClefMLXBackend("clef-mlx")


def test_cli_selects_and_starts_the_native_clef_backend(monkeypatch):
    import rapid_mlx._uvicorn as uvicorn_module
    import rapid_mlx.cli as cli_module
    import rapid_mlx.system_one.backends as backend_module

    for model in (
        "clef-mlx",
        "clef-flash-mlx",
        "nativ-community/clef-MLX-MXFP4",
        "nativ-community/clef-flash-MLX-MXFP4",
    ):
        assert _resolve_system_one_backend(model, "auto") == "clef-mlx"
    # The existing names keep their Torch backend.
    assert _resolve_system_one_backend("clef-flash", "auto") == "clef"
    assert _resolve_system_one_backend("Cloudflare/clef", "auto") == "clef"

    observed = {}

    def fake_backend(model, *, device):
        observed.update(model=model, device=device)
        return SimpleNamespace(default_model="clef-flash-mlx")

    monkeypatch.setattr(backend_module, "ClefMLXBackend", fake_backend)
    monkeypatch.setattr(cli_module, "_port_preflight_or_die", lambda *a, **k: None)
    monkeypatch.setattr(
        uvicorn_module,
        "run_uvicorn",
        lambda app, **kwargs: observed.update(kwargs=kwargs),
    )
    args = build_parser().parse_args(
        ["system-one", "/models/clef-8bit", "--backend", "clef-mlx", "--port", "8703"]
    )
    cli_module.system_one_command(args)
    assert observed["model"] == "/models/clef-8bit"
    assert observed["device"] == "gpu"


def test_adapted_code_carries_its_license_and_provenance():
    """clef_mlx.py is adapted from MIT mlx-vlm: the notice travels in the file."""
    source = Path(clef_mlx.__file__).read_text(encoding="utf-8")
    header = source.split('"""', 1)[0]
    assert "Copyright © 2025 Prince Canuma" in header
    assert "Permission is hereby granted, free of charge" in header
    assert 'THE SOFTWARE IS PROVIDED "AS IS"' in header
    assert "fdd94f39552a011e298f5d4160ef001238943c1b" in header
