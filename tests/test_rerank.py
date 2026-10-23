# SPDX-License-Identifier: Apache-2.0
"""``--rerank-model``: engine scoring, ``/v1/rerank`` and serve wiring."""

from __future__ import annotations

import asyncio
import sys
import threading
import types
from argparse import Namespace
from unittest import mock

import pytest

pytestmark = pytest.mark.requires_mlx

RERANK_ID = "mlx-community/Qwen3-Reranker-0.6B-mxfp8"
YES, NO, PAD, MARK = 1, 2, 0, 7
VOCAB = 8


class _FakeTokenizer:
    chat_template = "fake"
    pad_token_id = PAD
    unk_token_id = 3

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["add_generation_prompt"] is True
        return f"<s>{messages[0]['content']}|{messages[1]['content']}<a>"

    # One token per character; "!" is the relevance marker the fake model counts.
    def encode(self, text, add_special_tokens=True):
        if text in ("yes", "no"):
            return [YES if text == "yes" else NO]
        return [MARK if ch == "!" else 4 + ord(ch) % 2 for ch in text]


def _fake_logits(batch):
    import mlx.core as mx

    # Causal: the yes-logit at each position counts markers seen so far.
    markers = mx.cumsum((batch == MARK).astype(mx.float32), axis=1)
    zeros = mx.zeros(markers.shape)
    channels = [markers if i == YES else zeros for i in range(VOCAB)]
    return mx.stack(channels, axis=-1)


class _FullLogitsModel:
    model_type = "fake"

    def __call__(self, batch):
        return _fake_logits(batch)


class _SplitHeadModel:
    model_type = "qwen3"

    def __init__(self):
        self.model = lambda batch: _fake_logits(batch)
        self.lm_head = lambda hidden: hidden


def _engine(model, max_length=8192):
    from rapid_mlx.rerank import FAMILIES, RerankEngine

    engine = RerankEngine("fake/reranker")
    engine._model = model
    engine._tokenizer = _FakeTokenizer()
    engine.family = FAMILIES["qwen3"]
    engine.instruction = engine.family.default_instruction
    engine.prompt = engine.family.default_prompt
    engine._answer_ids, engine._pad_id = [YES, NO], PAD
    engine.max_length = max_length
    return engine


@pytest.mark.parametrize("model", [_FullLogitsModel(), _SplitHeadModel()])
def test_engine_scores_in_input_order_with_right_padding(model):
    import math

    engine = _engine(model)
    docs = ["a", "!!! long document text", "!", "plain but much longer document"]
    result = engine.score("q", docs)
    expected = [1 / (1 + math.exp(-docs[i].count("!"))) for i in range(len(docs))]
    assert result.scores == pytest.approx(expected, rel=1e-5)
    assert result.prompt_tokens > 0


def test_engine_empty_documents_score_nothing():
    assert _engine(_FullLogitsModel()).score("q", []).scores == []


def test_score_lazily_loads_before_handling_empty_documents(monkeypatch):
    from rapid_mlx import rerank

    engine = rerank.RerankEngine("fake/reranker")
    loaded = []

    def load():
        loaded.append(engine.model_name)
        engine._model = _FullLogitsModel()
        engine._tokenizer = _FakeTokenizer()
        engine.family = rerank.FAMILIES["qwen3"]
        engine._answer_ids = [YES, NO]
        engine.max_length = 32

    monkeypatch.setattr(engine, "load", load)
    assert engine.score("q", []).scores == []
    assert loaded == ["fake/reranker"]


def test_score_flushes_full_microbatches(monkeypatch):
    from rapid_mlx import rerank

    engine = _engine(_FullLogitsModel())
    monkeypatch.setattr(rerank, "_MAX_BATCH_SEQUENCES", 1)
    calls = []
    original = engine._score_batch

    def record(sequences):
        calls.append(len(sequences))
        return original(sequences)

    monkeypatch.setattr(engine, "_score_batch", record)
    result = engine.score("q", ["!", "!!"])
    assert result.scores[1] > result.scores[0]
    assert calls == [1, 1]


def test_engine_truncates_long_documents_and_counts_them():
    engine = _engine(_FullLogitsModel())
    head = len(engine._prompt_parts("q", "inst")[0])
    tail = len(engine._prompt_parts("q", "inst")[1])
    engine.max_length = head + tail + 3
    result = engine.score("q", ["!!!!!!", "x"], instruction="inst")
    assert engine.num_truncations == 1
    assert result.prompt_tokens == 2 * (head + tail) + 3 + 2


def test_engine_error_policy_rejects_long_documents():
    from rapid_mlx.rerank import RerankDocumentTooLongError

    engine = _engine(_FullLogitsModel())
    engine.overflow_policy = "error"
    head, tail = (len(part) for part in engine._prompt_parts("q", "inst"))
    engine.max_length = head + tail + 3
    assert engine.score("q", ["ab"], instruction="inst").scores
    with pytest.raises(RerankDocumentTooLongError) as exc_info:
        engine.score("q", ["ab", "abcdef"], instruction="inst")
    assert exc_info.value.index == 1
    assert exc_info.value.observed_tokens == head + tail + 7
    assert engine.num_truncations == 0


def test_engine_rejects_unknown_overflow_policy():
    from rapid_mlx.rerank import RerankEngine

    with pytest.raises(ValueError, match="overflow_policy"):
        RerankEngine("m", overflow_policy="drop")


def test_engine_rejects_unknown_model_family():
    from rapid_mlx.rerank import RerankEngine

    with pytest.raises(ValueError, match="model_family"):
        RerankEngine("m", model_family="other")


def test_qwen3_puts_prompt_in_system_turn_and_instruction_in_user_turn():
    engine = _engine(_FullLogitsModel())
    engine.prompt = "Judge."
    seen = []
    render = engine._tokenizer.apply_chat_template

    def _record(messages, **kwargs):
        seen.append(messages)
        return render(messages, **kwargs)

    engine._tokenizer.apply_chat_template = _record
    engine._prompt_parts("q?", "Find code")
    assert seen[0][0] == {"role": "system", "content": "Judge."}
    assert seen[0][1]["content"].startswith(
        "<Instruct>: Find code\n<Query>: q?\n<Document>:"
    )


def test_engine_instruction_precedence():
    engine = _engine(_FullLogitsModel())
    engine.instruction = "server task"
    seen = []
    original = engine._prompt_parts

    def _record(query, instruction):
        seen.append(instruction)
        return original(query, instruction)

    engine._prompt_parts = _record
    engine.score("q", ["a"])
    engine.score("q", ["a"], instruction="request task")
    assert seen == ["server task", "request task"]


def test_engine_rejects_query_that_leaves_no_document_room():
    from rapid_mlx.rerank import RerankInputTooLongError

    engine = _engine(_FullLogitsModel(), max_length=5)
    with pytest.raises(RerankInputTooLongError) as exc_info:
        engine.score("a long query", ["doc"])
    assert exc_info.value.allowed_tokens == 5


def _causal(model_type, arch, **extra):
    return {"model_type": model_type, "architectures": [arch], **extra}


@pytest.mark.parametrize(
    ("model_name", "config", "family_id"),
    [
        (RERANK_ID, _causal("qwen3", "Qwen3ForCausalLM"), "qwen3"),
        (
            "BAAI/bge-reranker-v2-gemma",
            _causal("gemma", "GemmaForCausalLM"),
            "bge2",
        ),
        (
            "mixedbread-ai/mxbai-rerank-base-v2",
            _causal("qwen2", "Qwen2ForCausalLM"),
            "mxbai2",
        ),
        (
            "/models/converted",
            _causal(
                "qwen2",
                "Qwen2ForCausalLM",
                _name_or_path="mixedbread-ai/mxbai-rerank-large-v2",
            ),
            "mxbai2",
        ),
        (
            "~/models/Qwen3_Reranker_4B",
            _causal("qwen3", "Qwen3ForCausalLM"),
            "qwen3",
        ),
        (
            "BAAI/bge-reranker-v2-m3",
            {
                "model_type": "xlm-roberta",
                "architectures": ["XLMRobertaForSequenceClassification"],
            },
            "roberta",
        ),
        (
            "mlx-community/jina-reranker-v3-4bit-mxfp4",
            _causal("qwen3", "JinaForRanking"),
            "jina3",
        ),
    ],
)
def test_detect_family(model_name, config, family_id):
    from rapid_mlx.rerank import detect_family

    assert detect_family(config, model_name).family_id == family_id


def test_family_choices_match_profiles():
    from rapid_mlx import cli, rerank
    from rapid_mlx.server import _build_parser

    for parser, argv in ((cli.build_parser(), ["serve"]), (_build_parser(), [])):
        for family_id in rerank.FAMILIES:
            args = parser.parse_args([*argv, "--rerank-model-family", family_id])
            assert args.rerank_model_family == family_id
        with pytest.raises(SystemExit):
            parser.parse_args([*argv, "--rerank-model-family", "other"])


def test_detect_family_override_skips_name_match():
    from rapid_mlx.rerank import UnsupportedRerankModelError, detect_family

    config = _causal("qwen2", "Qwen2ForCausalLM")
    with pytest.raises(UnsupportedRerankModelError, match="--rerank-model-family"):
        detect_family(config, "me/my-finetune")
    assert detect_family(config, "me/my-finetune", "mxbai2").family_id == "mxbai2"
    with pytest.raises(UnsupportedRerankModelError, match="model_type=qwen3"):
        detect_family(config, "me/my-finetune", "qwen3")


@pytest.mark.parametrize(
    ("model_name", "config"),
    [
        ("Qwen/Qwen3-8B", _causal("qwen3", "Qwen3ForCausalLM")),
        ("BAAI/bge-m3", _causal("xlm-roberta", "XLMRobertaModel")),
        (
            "mixedbread-ai/mxbai-rerank-large-v1",
            _causal("deberta-v2", "DebertaV2ForSequenceClassification"),
        ),
        (
            "some/qwen3-reranker-gguf",
            _causal("qwen3", "Qwen3ForSequenceClassification"),
        ),
        ("no/config", {}),
    ],
)
def test_detect_family_rejects_other_checkpoints(model_name, config):
    from rapid_mlx.rerank import UnsupportedRerankModelError, detect_family

    with pytest.raises(UnsupportedRerankModelError, match="Supported families"):
        detect_family(config, model_name)


class _CharTokenizer:
    """Reversible tokenizer: one id per character, negative ids for specials."""

    bos_token_id, cls_token_id, sep_token_id, pad_token_id = -1, -2, -3, -4
    _SPECIALS = {-1: "<bos>", -2: "<s>", -3: "</s>", -4: "<pad>"}

    def encode(self, text, add_special_tokens=True):
        return [ord(ch) for ch in text]

    def decode(self, ids):
        return "".join(self._SPECIALS.get(i) or chr(i) for i in ids)


def _format_engine(family_id, instruction=None, prompt=None):
    from rapid_mlx.rerank import FAMILIES, RerankEngine

    engine = RerankEngine("fake/reranker")
    engine._tokenizer = _CharTokenizer()
    engine.family = FAMILIES[family_id]
    engine.instruction = instruction or engine.family.default_instruction
    engine.prompt = prompt or engine.family.default_prompt
    engine.max_length = 8192
    return engine


def test_bge_gemma_format_matches_upstream():
    from rapid_mlx.rerank import FAMILIES

    engine = _format_engine("bge2")
    (sequence,) = engine._build_sequences("q?", ["doc"], engine.instruction)
    prompt = FAMILIES["bge2"].default_instruction
    assert engine._tokenizer.decode(sequence) == f"<bos>A: q?\nB: doc\n{prompt}"


@pytest.mark.parametrize(
    ("instruction", "query_part"),
    [(None, "query: q?"), ("Find code", "instruction: Find code\nquery: q?")],
)
def test_mxbai_format_matches_upstream(instruction, query_part):
    from rapid_mlx import rerank

    engine = _format_engine("mxbai2")
    (sequence,) = engine._build_sequences("q?", ["doc"], instruction)
    assert engine._tokenizer.decode(sequence) == (
        "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a "
        "helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{query_part}\ndocument: doc\n"
        f"{rerank.FAMILIES['mxbai2'].default_prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def test_rerank_prompt_replaces_family_prompt():
    engine = _format_engine("mxbai2", prompt="Judge.")
    (sequence,) = engine._build_sequences("q?", ["doc"], None)
    assert engine._tokenizer.decode(sequence).endswith(
        "document: doc\nJudge.<|im_end|>\n<|im_start|>assistant\n"
    )


def test_cross_encoder_format_matches_pair_tokenizer():
    engine = _format_engine("roberta")
    (sequence,) = engine._build_sequences("q?", ["doc"], None)
    assert engine._tokenizer.decode(sequence) == "<s>q?</s></s>doc</s>"


def test_document_truncation_keeps_the_family_suffix():
    engine = _format_engine("roberta")
    engine.max_length = len("q?") + 4 + 2
    (sequence,) = engine._build_sequences("q?", ["document"], None)
    assert engine._tokenizer.decode(sequence) == "<s>q?</s></s>do</s>"
    assert engine.num_truncations == 1


def test_single_answer_token_scores_its_sigmoid():
    import math

    from rapid_mlx.rerank import FAMILIES

    engine = _engine(_FullLogitsModel())
    engine.family = FAMILIES["bge2"]
    engine._answer_ids = [YES]
    assert engine._score_batch([[MARK, MARK], [4]]) == pytest.approx(
        [1 / (1 + math.exp(-2)), 0.5], rel=1e-5
    )


def test_cross_encoder_scores_with_padding_mask():
    import math

    import mlx.core as mx

    seen = []

    def _model(ids, mask):
        seen.append(mask.tolist())
        return mx.sum(mask, axis=1).astype(mx.float32) - 2

    engine = _format_engine("roberta")
    engine._model, engine._pad_id = _model, 1
    scores = engine._score_batch([[5, 6], [5, 6, 7, 8]])
    assert seen == [[[1, 1, 0, 0], [1, 1, 1, 1]]]
    assert scores == pytest.approx([0.5, 1 / (1 + math.exp(-2))], rel=1e-5)


def test_cross_encoder_rejects_request_instruction():
    from rapid_mlx.rerank import RerankInstructionNotSupportedError

    engine = _format_engine("roberta")
    engine._model = object()
    with pytest.raises(RerankInstructionNotSupportedError):
        engine.score("q", ["d"], instruction="task")


@pytest.mark.parametrize(
    ("model_name", "config", "family_id", "instruction", "max_length", "answers"),
    [
        (
            RERANK_ID,
            _causal("qwen3", "Qwen3ForCausalLM", max_position_embeddings=4096),
            "qwen3",
            "Given a web search query, retrieve relevant passages that answer "
            "the query",
            4096,
            [YES, NO],
        ),
        (
            "BAAI/bge-reranker-v2-gemma",
            _causal("gemma", "GemmaForCausalLM", max_position_embeddings=8192),
            "bge2",
            "Given a query A and a passage B, determine whether the passage "
            "contains an answer to the query by providing a prediction of either "
            "'Yes' or 'No'.",
            1024,
            [ord("Y")],
        ),
        (
            "mixedbread-ai/mxbai-rerank-base-v2",
            _causal("qwen2", "Qwen2ForCausalLM", max_position_embeddings=32768),
            "mxbai2",
            None,
            8192,
            [ord("1"), ord("0")],
        ),
    ],
)
def test_engine_load_applies_family_profile(
    monkeypatch, model_name, config, family_id, instruction, max_length, answers
):
    from rapid_mlx import rerank
    from rapid_mlx.utils import tokenizer as model_loader

    class _Tokenizer(_FakeTokenizer):
        def encode(self, text, add_special_tokens=True):
            if text == "Yes":
                return [ord("Y")]
            if text in ("1", "0"):
                return [ord(text)]
            return super().encode(text, add_special_tokens)

    monkeypatch.setattr(rerank, "_read_model_config", lambda _name: config)
    monkeypatch.setattr(
        model_loader,
        "load_model_with_fallback",
        lambda _name: (_FullLogitsModel(), _Tokenizer()),
    )
    engine = rerank.RerankEngine(model_name)
    engine.load()
    assert engine.family.family_id == family_id
    assert engine.instruction == instruction
    assert engine.prompt == rerank.FAMILIES[family_id].default_prompt
    assert engine.max_length == max_length
    assert engine._answer_ids == answers


def test_engine_load_applies_instruction_and_prompt_settings(monkeypatch):
    from rapid_mlx import rerank
    from rapid_mlx.utils import tokenizer as model_loader

    monkeypatch.setattr(
        rerank, "_read_model_config", lambda _name: _causal("qwen3", "Qwen3ForCausalLM")
    )
    monkeypatch.setattr(
        model_loader,
        "load_model_with_fallback",
        lambda _name: (_FullLogitsModel(), _FakeTokenizer()),
    )
    engine = rerank.RerankEngine(RERANK_ID, instruction="Find code", prompt="Judge.")
    engine.load()
    assert (engine.instruction, engine.prompt) == ("Find code", "Judge.")


def test_engine_load_rejects_instruction_for_cross_encoder(monkeypatch):
    from rapid_mlx import rerank

    monkeypatch.setattr(
        rerank,
        "_read_model_config",
        lambda _name: {
            "model_type": "xlm-roberta",
            "architectures": ["XLMRobertaForSequenceClassification"],
        },
    )
    engine = rerank.RerankEngine("BAAI/bge-reranker-v2-m3", instruction="task")
    with pytest.raises(rerank.UnsupportedRerankModelError, match="no instruction"):
        engine.load()


@pytest.mark.parametrize(
    ("model_name", "config"),
    [
        (
            "BAAI/bge-reranker-v2-m3",
            {
                "model_type": "xlm-roberta",
                "architectures": ["XLMRobertaForSequenceClassification"],
            },
        ),
        ("BAAI/bge-reranker-v2-gemma", _causal("gemma", "GemmaForCausalLM")),
    ],
)
def test_engine_load_rejects_prompt_for_family_without_one(
    monkeypatch, model_name, config
):
    from rapid_mlx import rerank

    monkeypatch.setattr(rerank, "_read_model_config", lambda _name: config)
    engine = rerank.RerankEngine(model_name, prompt="Judge.")
    with pytest.raises(rerank.UnsupportedRerankModelError, match="no prompt"):
        engine.load()


def test_engine_load_rejects_multi_answer_token(monkeypatch):
    from rapid_mlx import rerank
    from rapid_mlx.utils import tokenizer as model_loader

    monkeypatch.setattr(
        rerank,
        "_read_model_config",
        lambda _name: _causal("gemma", "GemmaForCausalLM"),
    )
    monkeypatch.setattr(
        model_loader,
        "load_model_with_fallback",
        lambda _name: (_FullLogitsModel(), _FakeTokenizer()),
    )
    engine = rerank.RerankEngine("BAAI/bge-reranker-v2-gemma")
    with pytest.raises(rerank.UnsupportedRerankModelError, match="'Yes'"):
        engine.load()


def test_engine_load_rejects_qwen_without_chat_template(monkeypatch):
    from rapid_mlx import rerank
    from rapid_mlx.utils import tokenizer as model_loader

    tokenizer = _FakeTokenizer()
    tokenizer.chat_template = None
    monkeypatch.setattr(
        rerank,
        "_read_model_config",
        lambda _name: _causal("qwen3", "Qwen3ForCausalLM"),
    )
    monkeypatch.setattr(
        model_loader, "load_model_with_fallback", lambda _name: (object(), tokenizer)
    )

    with pytest.raises(rerank.UnsupportedRerankModelError, match="no chat template"):
        rerank.RerankEngine(RERANK_ID).load()


def test_engine_loads_cross_encoder_context_and_default_padding(monkeypatch):
    from rapid_mlx import rerank

    class Tokenizer:
        pad_token_id = None
        eos_token_id = None

    config = {
        "model_type": "xlm-roberta",
        "architectures": ["XLMRobertaForSequenceClassification"],
        "max_position_embeddings": 514,
        "pad_token_id": 1,
    }
    model = object()
    monkeypatch.setattr(rerank, "_read_model_config", lambda _name: config)
    monkeypatch.setattr(
        rerank, "_load_cross_encoder", lambda _name, _config: (model, Tokenizer())
    )

    engine = rerank.RerankEngine("BAAI/bge-reranker-v2-m3")
    engine.load()
    assert engine._model is model
    assert engine.max_length == 512
    assert engine._pad_id == 0


_TINY_XLMR = {
    "model_type": "xlm-roberta",
    "architectures": ["XLMRobertaForSequenceClassification"],
    "hidden_size": 8,
    "num_hidden_layers": 1,
    "intermediate_size": 16,
    "num_attention_heads": 2,
    "max_position_embeddings": 20,
    "vocab_size": 30,
    "pad_token_id": 1,
    "id2label": {"0": "LABEL_0"},
}


def test_cross_encoder_loads_classifier_checkpoint(monkeypatch, tmp_path):
    pytest.importorskip("mlx_embeddings")
    import mlx.core as mx
    import transformers
    from mlx.utils import tree_flatten
    from mlx_embeddings.models import xlm_roberta

    from rapid_mlx import rerank

    args = xlm_roberta.ModelArgs.from_dict({**_TINY_XLMR, "add_pooling_layer": False})
    reference = rerank._CrossEncoder(xlm_roberta.Model(args), 8)
    reference.eval()
    weights = dict(tree_flatten(reference.parameters()))
    weights["roberta.embeddings.position_ids"] = mx.arange(20)[None]
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda path, **kwargs: ("tok", path),
    )

    model, tokenizer = rerank._load_cross_encoder(str(tmp_path), _TINY_XLMR)
    assert tokenizer == ("tok", str(tmp_path))
    ids = mx.array([[0, 5, 2, 2, 6, 2]])
    mask = mx.ones(ids.shape)
    assert model(ids, mask).tolist() == pytest.approx(
        reference(ids, mask).tolist(), rel=1e-5
    )

    del weights["classifier.out_proj.weight"]
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    with pytest.raises(rerank.UnsupportedRerankModelError, match="weight layout"):
        rerank._load_cross_encoder(str(tmp_path), _TINY_XLMR)


def test_cross_encoder_requires_embeddings_extra(monkeypatch):
    import importlib.util

    from rapid_mlx import rerank

    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a: None if name == "mlx_embeddings" else real_find_spec(name, *a),
    )
    with pytest.raises(rerank.UnsupportedRerankModelError, match="embeddings"):
        rerank._load_cross_encoder("BAAI/bge-reranker-v2-m3", _TINY_XLMR)


def test_read_model_config_supports_local_and_hub_paths(monkeypatch, tmp_path):
    import json

    from rapid_mlx import model_downloads, rerank

    local = tmp_path / "local"
    local.mkdir()
    (local / "config.json").write_text(json.dumps({"source": "local"}))
    remote = tmp_path / "downloaded-config.json"
    remote.write_text(json.dumps({"source": "hub"}))
    calls = []
    monkeypatch.setattr(
        model_downloads,
        "hf_hub_download",
        lambda model, filename: calls.append((model, filename)) or str(remote),
    )

    assert rerank._read_model_config(str(local)) == {"source": "local"}
    assert rerank._read_model_config("org/reranker") == {"source": "hub"}
    assert calls == [("org/reranker", "config.json")]


def test_cross_encoder_rejects_multilabel_checkpoint(monkeypatch):
    import importlib.util

    from rapid_mlx import rerank

    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: object())
    with pytest.raises(rerank.UnsupportedRerankModelError, match="more than one label"):
        rerank._load_cross_encoder("org/classifier", {**_TINY_XLMR, "num_labels": 2})


def test_cross_encoder_downloads_remote_checkpoint(monkeypatch, tmp_path):
    pytest.importorskip("mlx_embeddings")
    import mlx.core as mx
    import transformers
    from mlx.utils import tree_flatten
    from mlx_embeddings.models import xlm_roberta

    from rapid_mlx import model_downloads, rerank

    args = xlm_roberta.ModelArgs.from_dict({**_TINY_XLMR, "add_pooling_layer": False})
    reference = rerank._CrossEncoder(xlm_roberta.Model(args), args.hidden_size)
    mx.save_safetensors(
        str(tmp_path / "model.safetensors"), dict(tree_flatten(reference.parameters()))
    )
    seen = []
    monkeypatch.setattr(
        model_downloads,
        "snapshot_download",
        lambda model, *, allow_patterns: (
            seen.append((model, allow_patterns)) or str(tmp_path)
        ),
    )

    def load_tokenizer(path, *, trust_remote_code):
        assert trust_remote_code is False
        return "tokenizer", path

    monkeypatch.delenv("RAPID_MLX_TRUST_REMOTE_CODE", raising=False)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", load_tokenizer)

    model, tokenizer = rerank._load_cross_encoder("org/remote-xlm", _TINY_XLMR)
    assert isinstance(model, rerank._CrossEncoder)
    assert tokenizer == ("tokenizer", str(tmp_path))
    assert seen == [("org/remote-xlm", rerank._CROSS_ENCODER_FILES)]


def _install_fake_xlm_roberta(monkeypatch):
    """Stand in for ``mlx_embeddings.models.xlm_roberta`` so the loader runs
    on hosts without the [embeddings] extra."""
    import importlib.util
    import sys
    import types

    import mlx.nn as nn

    class ModelArgs:
        def __init__(self, config):
            self.hidden_size = config["hidden_size"]
            self.vocab_size = config["vocab_size"]

        @classmethod
        def from_dict(cls, config):
            return cls(config)

    class Model(nn.Module):
        def __init__(self, args):
            super().__init__()
            self.embeddings = nn.Embedding(args.vocab_size, args.hidden_size)

        def __call__(self, input_ids, attention_mask=None):
            hidden = self.embeddings(input_ids) * attention_mask[..., None]
            return types.SimpleNamespace(last_hidden_state=hidden)

    xlm_roberta = types.ModuleType("mlx_embeddings.models.xlm_roberta")
    xlm_roberta.ModelArgs = ModelArgs
    xlm_roberta.Model = Model
    models = types.ModuleType("mlx_embeddings.models")
    models.xlm_roberta = xlm_roberta
    package = types.ModuleType("mlx_embeddings")
    package.models = models
    monkeypatch.setitem(sys.modules, "mlx_embeddings", package)
    monkeypatch.setitem(sys.modules, "mlx_embeddings.models", models)
    monkeypatch.setitem(sys.modules, "mlx_embeddings.models.xlm_roberta", xlm_roberta)
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a: (
            object() if name == "mlx_embeddings" else real_find_spec(name, *a)
        ),
    )
    return xlm_roberta


def test_cross_encoder_loader_without_embeddings_extra(monkeypatch, tmp_path):
    import mlx.core as mx
    import transformers
    from mlx.utils import tree_flatten

    from rapid_mlx import model_downloads, rerank

    xlm_roberta = _install_fake_xlm_roberta(monkeypatch)
    args = xlm_roberta.ModelArgs.from_dict(_TINY_XLMR)
    reference = rerank._CrossEncoder(xlm_roberta.Model(args), args.hidden_size)
    weights = dict(tree_flatten(reference.parameters()))
    weights["roberta.embeddings.position_ids"] = mx.arange(20)[None]
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda path, **kwargs: ("tok", path),
    )
    seen = []
    monkeypatch.setattr(
        model_downloads,
        "snapshot_download",
        lambda model, *, allow_patterns: seen.append(model) or str(tmp_path),
    )

    model, tokenizer = rerank._load_cross_encoder("org/remote-xlm", _TINY_XLMR)
    assert seen == ["org/remote-xlm"]
    assert tokenizer == ("tok", str(tmp_path))
    ids = mx.array([[0, 5, 2, 6]])
    mask = mx.ones(ids.shape)
    assert model(ids, mask).tolist() == pytest.approx(
        reference(ids, mask).tolist(), rel=1e-5
    )

    del weights["classifier.out_proj.weight"]
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    with pytest.raises(rerank.UnsupportedRerankModelError, match="weight layout"):
        rerank._load_cross_encoder(str(tmp_path), _TINY_XLMR)


class _MarkerTokenizer(_CharTokenizer):
    """``_CharTokenizer`` that maps the jina marker strings to single ids."""

    DOC, QUERY = -10, -11

    def encode(self, text, add_special_tokens=True):
        import re

        ids = []
        for part in re.split(r"(<\|embed_token\|>|<\|rerank_token\|>)", text):
            if part == "<|embed_token|>":
                ids.append(self.DOC)
            elif part == "<|rerank_token|>":
                ids.append(self.QUERY)
            else:
                ids.extend(ord(ch) for ch in part)
        return ids


class _NonAdditiveMarkerTokenizer(_MarkerTokenizer):
    """Tokenizer with a multi-passage cost singleton estimates cannot predict."""

    def encode(self, text, add_special_tokens=True):
        ids = super().encode(text, add_special_tokens=add_special_tokens)
        if text.count('<passage id="') >= 3:
            ids.extend([ord("x")] * 200)
        return ids


class _FakeJina:
    """Hidden state at each position encodes whether the previous char is a/b."""

    def __init__(self):
        self.calls = 0
        self.prompt_lengths = []
        self.projector = lambda x: x

    def model(self, batch):
        import mlx.core as mx

        self.calls += 1
        self.prompt_lengths.append(batch.shape[1])
        rows, prev = [], 0
        for token in batch[0].tolist():
            rows.append([1.0, float(prev == ord("a")), float(prev == ord("b")), 0.0])
            prev = token
        return mx.array([rows])


def _jina_engine(context_length=100_000, max_length=2048):
    from rapid_mlx.rerank import FAMILIES, RerankEngine

    engine = RerankEngine("mlx-community/jina-reranker-v3-4bit-mxfp4")
    engine.family = FAMILIES["jina3"]
    engine.prompt = engine.family.default_prompt
    engine._tokenizer = _MarkerTokenizer()
    engine._model = _FakeJina()
    engine._marker_ids = (_MarkerTokenizer.DOC, _MarkerTokenizer.QUERY)
    engine._context_length = context_length
    engine.max_length = max_length
    return engine


def test_jina_prompt_matches_upstream():
    from rapid_mlx import rerank

    prompt = _jina_engine()._listwise_prompt(
        "q<|rerank_token|>", ["d1", "d<|embed_token|>2"], "inst"
    )
    assert prompt == (
        f"<|im_start|>system\n{rerank.FAMILIES['jina3'].default_prompt}<|im_end|>\n"
        "<|im_start|>user\nI will provide you with 2 passages, each indicated "
        "by a numerical identifier. Rank the passages based on their relevance to "
        "query: q\n<instruct>\ninst\n</instruct>\n"
        '<passage id="0">\nd1<|embed_token|>\n</passage>\n'
        '<passage id="1">\nd2<|embed_token|>\n</passage>\n'
        "<query>\nq<|rerank_token|>\n</query><|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def test_jina_prompt_replacement_and_omitted_instruction():
    engine = _jina_engine()
    engine.prompt = "Rank."
    prompt = engine._listwise_prompt("q", ["d"], None)
    assert prompt.startswith("<|im_start|>system\nRank.<|im_end|>\n")
    assert "<instruct>" not in prompt


def test_jina_scores_cosine_of_marker_states():
    import math

    engine = _jina_engine()
    result = engine.score("qa", ["xa", "xb", "xc"])
    assert result.scores == pytest.approx(
        [1.0, 0.75, (1 + 1 / math.sqrt(2)) / 2], rel=1e-5
    )
    assert engine._model.calls == 1
    assert result.prompt_tokens > 0


def test_jina_splits_blocks_without_changing_scores():
    single = _jina_engine().score("qa", ["xa", "xb", "xc"]).scores
    engine = _jina_engine(context_length=720, max_length=10)
    assert engine.score("qa", ["xa", "xb", "xc"]).scores == pytest.approx(single)
    assert engine._model.calls == 3


def test_jina_block_prompts_stay_within_context_length():
    engine = _jina_engine(context_length=760)
    documents = ["x" * 20] * 3

    combined_length = len(
        engine._tokenizer.encode(
            engine._listwise_prompt("qa", documents, None),
            add_special_tokens=False,
        )
    )
    # The old raw-token budget admitted this block despite its rendered prompt
    # exceeding the context window.
    assert 2 * len("qa") + sum(map(len, documents)) <= engine._context_length
    assert combined_length > engine._context_length

    engine.score("qa", documents)

    assert engine._model.calls == 3
    assert max(engine._model.prompt_lengths) <= engine._context_length


def test_jina_block_packing_tokenization_is_linear():
    engine = _jina_engine()
    documents = [f"document {index}" for index in range(126)]
    real_encode = engine._tokenizer.encode
    engine._tokenizer.encode = mock.Mock(wraps=real_encode)

    engine.score("qa", documents)

    # One raw-document encode, one singleton prompt, and only a constant number
    # of packing/exact-block encodes per document (plus block inference).
    assert engine._tokenizer.encode.call_count <= (
        3 * len(documents) + engine._model.calls + 4
    )


def test_jina_exact_validation_splits_underestimated_block():
    # A small max_length keeps the upstream raw-budget rule from closing the
    # block first, so only the exact prompt check can split it.
    engine = _jina_engine(context_length=900, max_length=8)
    engine._tokenizer = _NonAdditiveMarkerTokenizer()
    documents = ["xa"] * 4
    assert (
        len(
            engine._tokenizer.encode(
                engine._listwise_prompt("qa", documents, None),
                add_special_tokens=False,
            )
        )
        > engine._context_length
    )

    result = engine.score("qa", documents)

    assert len(result.scores) == len(documents)
    assert engine._model.calls == 2
    assert max(engine._model.prompt_lengths) <= engine._context_length


def test_jina_blocks_keep_upstream_raw_budget_boundary():
    # Upstream closes a block once the raw budget left for documents is at
    # most one max-length document, even when the full prompt would still fit.
    engine = _jina_engine(context_length=100_000, max_length=50_000)
    query_length = len(engine._tokenizer.encode("qa", add_special_tokens=False))
    doc_length = 1000
    budget = engine._context_length - 2 * query_length
    per_block = next(
        n for n in range(1, 200) if budget - n * doc_length <= engine.max_length
    )
    documents = ["x" * doc_length] * (per_block + 1)
    assert len(engine._tokenizer.encode(documents[0], add_special_tokens=False)) == (
        doc_length
    )

    engine.score("qa", documents)

    assert engine._model.calls == 2


def test_jina_context_overflow_policy():
    from rapid_mlx.rerank import (
        RerankDocumentTooLongError,
        RerankInputTooLongError,
    )

    engine = _jina_engine(context_length=710)
    engine.score("qa", ["x" * 20])
    assert engine.num_truncations == 1
    assert engine._model.prompt_lengths == [710]

    engine = _jina_engine(context_length=710)
    engine.overflow_policy = "error"
    with pytest.raises(RerankDocumentTooLongError) as exc_info:
        engine.score("qa", ["x" * 20])
    assert exc_info.value.observed_tokens == 721

    engine = _jina_engine(context_length=700)
    with pytest.raises(RerankInputTooLongError) as exc_info:
        engine.score("qa", ["x"])
    assert exc_info.value.observed_tokens == 701


def test_jina_overflow_policy():
    from rapid_mlx.rerank import (
        RerankDocumentTooLongError,
        RerankInputTooLongError,
    )

    engine = _jina_engine(max_length=3)
    assert engine.score("qa", ["xxxxa", "xb"]).scores[1] == pytest.approx(0.75)
    assert engine.num_truncations == 1

    engine.overflow_policy = "error"
    with pytest.raises(RerankDocumentTooLongError) as exc_info:
        engine.score("qa", ["xb", "xxxxa"])
    assert (exc_info.value.index, exc_info.value.allowed_tokens) == (1, 3)
    with pytest.raises(RerankInputTooLongError):
        engine.score("q" * 513, ["xa"])


def test_jina_loads_transformers_projector_names(monkeypatch, tmp_path):
    import json

    import mlx.core as mx
    import mlx_lm.utils
    from mlx.utils import tree_flatten
    from mlx_lm.models import qwen3

    from rapid_mlx import rerank

    config = {
        "model_type": "qwen3",
        "architectures": ["JinaForRanking"],
        "hidden_size": 8,
        "num_hidden_layers": 1,
        "intermediate_size": 16,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "rms_norm_eps": 1e-6,
        "vocab_size": 16,
        "max_position_embeddings": 64,
        "rope_theta": 10000,
        "tie_word_embeddings": True,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    model_class, _ = rerank._jina_model_classes(config)
    reference = model_class(qwen3.ModelArgs.from_dict(config))
    weights = {
        k.replace("projector.layers.", "projector."): v
        for k, v in tree_flatten(reference.parameters())
    }
    assert "projector.2.weight" in weights
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)

    def load_tokenizer(path, *, tokenizer_config):
        assert tokenizer_config["trust_remote_code"] is False
        return "tok", path

    monkeypatch.delenv("RAPID_MLX_TRUST_REMOTE_CODE", raising=False)
    monkeypatch.setattr(mlx_lm.utils, "load_tokenizer", load_tokenizer)

    model, tokenizer = rerank._load_jina(str(tmp_path))
    assert tokenizer == ("tok", tmp_path)
    assert mx.array_equal(
        model.projector.layers[2].weight, reference.projector.layers[2].weight
    )


def test_jina_remote_download_and_bad_weight_layout(monkeypatch, tmp_path):
    import mlx_lm.utils

    from rapid_mlx import model_downloads, rerank

    seen = []
    monkeypatch.setattr(
        model_downloads,
        "snapshot_download",
        lambda model, *, allow_patterns: (
            seen.append((model, allow_patterns)) or str(tmp_path)
        ),
    )
    monkeypatch.setattr(
        mlx_lm.utils,
        "load_model",
        lambda *_a, **_k: (_ for _ in ()).throw(ValueError("projector mismatch")),
    )

    with pytest.raises(rerank.UnsupportedRerankModelError, match="projector mismatch"):
        rerank._load_jina("org/remote-jina")
    assert seen == [("org/remote-jina", rerank._JINA3_FILES)]


def test_engine_loads_listwise_markers_and_eos_padding(monkeypatch):
    from rapid_mlx import rerank

    class Tokenizer:
        eos_token_id = 9

        def encode(self, token, add_special_tokens=False):
            assert add_special_tokens is False
            return {
                "<|embed_token|>": [10],
                "<|rerank_token|>": [11],
            }[token]

    monkeypatch.setattr(
        rerank,
        "_read_model_config",
        lambda _name: {
            "model_type": "qwen3",
            "architectures": ["JinaForRanking"],
            "max_position_embeddings": 4096,
        },
    )
    monkeypatch.setattr(rerank, "_load_jina", lambda _name: (object(), Tokenizer()))

    engine = rerank.RerankEngine("org/jina")
    engine.load()
    assert engine.family is rerank.FAMILIES["jina3"]
    assert engine._marker_ids == (10, 11)
    assert engine._context_length == 4096
    assert engine._pad_id == 9


def test_engine_rejects_multitoken_listwise_marker(monkeypatch):
    from rapid_mlx import rerank

    class Tokenizer:
        pad_token_id = 0

        def encode(self, _token, add_special_tokens=False):
            assert add_special_tokens is False
            return [10, 11]

    monkeypatch.setattr(
        rerank,
        "_read_model_config",
        lambda _name: {
            "model_type": "qwen3",
            "architectures": ["JinaForRanking"],
            "max_position_embeddings": 4096,
        },
    )
    monkeypatch.setattr(rerank, "_load_jina", lambda _name: (object(), Tokenizer()))

    with pytest.raises(rerank.UnsupportedRerankModelError, match="marker token"):
        rerank.RerankEngine("org/jina").load()


@pytest.mark.parametrize(
    ("setting", "declared", "expected"),
    [
        ("auto", 40960, 8192),
        ("auto", 4096, 4096),
        ("auto", None, 8192),
        ("AUTO", 512, 512),
        (16384, 40960, 16384),
        ("1024", 40960, 1024),
        (100000, 40960, 40960),
        (2048, None, 2048),
    ],
)
def test_engine_resolves_max_length(setting, declared, expected):
    from rapid_mlx.rerank import FAMILIES, RerankEngine

    engine = RerankEngine(RERANK_ID, max_length=setting)
    engine.family = FAMILIES["qwen3"]
    assert engine._resolve_max_length(declared) == expected


@pytest.mark.parametrize("value", ["0", "-1", "abc", "", True, 1.5])
def test_engine_rejects_invalid_max_length(value):
    from rapid_mlx.rerank import RerankEngine

    with pytest.raises(ValueError):
        RerankEngine(RERANK_ID, max_length=value)


# ---------------------------------------------------------------------------
# /v1/rerank
# ---------------------------------------------------------------------------


class _StubEngine:
    def __init__(self, scores, error=None, overflow_policy="truncate"):
        self.scores = scores
        self.error = error
        self.overflow_policy = overflow_policy
        self.calls = []

    def score(self, query, documents, *, instruction=None):
        from rapid_mlx.rerank import RerankScores

        self.calls.append((query, documents, instruction))
        if self.error is not None:
            raise self.error
        return RerankScores(scores=self.scores[: len(documents)], prompt_tokens=42)


@pytest.fixture
def rerank_client(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from rapid_mlx import server
    from rapid_mlx.config import reset_config
    from rapid_mlx.middleware.exception_handlers import install_exception_handlers
    from rapid_mlx.routes.models import router as models_router
    from rapid_mlx.routes.rerank import router

    monkeypatch.setattr(server, "_rerank_engine", None)
    monkeypatch.setattr(server, "_rerank_model_locked", None)
    cfg = reset_config()
    app = FastAPI()
    install_exception_handlers(app)
    app.include_router(router)
    app.include_router(models_router)
    yield types.SimpleNamespace(client=TestClient(app), cfg=cfg)
    reset_config()


def _configure(cfg, engine):
    cfg.rerank_engine = engine
    cfg.rerank_model_locked = RERANK_ID


def test_rerank_without_model_returns_503(rerank_client):
    resp = rerank_client.client.post(
        "/v1/rerank", json={"query": "q", "documents": ["a"]}
    )
    assert resp.status_code == 503
    assert "no_rerank_model" in resp.text


def test_rerank_sorts_truncates_and_returns_documents(rerank_client):
    engine = _StubEngine([0.1, 0.9, 0.5])
    _configure(rerank_client.cfg, engine)
    resp = rerank_client.client.post(
        "/v1/rerank",
        json={
            "model": RERANK_ID,
            "query": "q",
            "documents": ["a", {"text": "b"}, "c"],
            "top_n": 2,
            "instruction": "find it",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["model"] == RERANK_ID
    assert body["id"].startswith("rerank-")
    assert [r["index"] for r in body["results"]] == [1, 2]
    assert body["results"][0] == {
        "index": 1,
        "relevance_score": 0.9,
        "document": {"text": "b"},
    }
    assert body["usage"] == {"prompt_tokens": 42, "total_tokens": 42}
    assert engine.calls == [("q", ["a", "b", "c"], "find it")]


def test_rerank_can_omit_documents_and_defaults_model(rerank_client):
    _configure(rerank_client.cfg, _StubEngine([0.2, 0.3]))
    resp = rerank_client.client.post(
        "/v1/rerank",
        json={"query": "q", "documents": ["a", "b"], "return_documents": False},
    )
    assert resp.status_code == 200
    assert resp.json()["results"] == [
        {"index": 1, "relevance_score": 0.3},
        {"index": 0, "relevance_score": 0.2},
    ]


def test_rerank_rejects_other_model(rerank_client):
    _configure(rerank_client.cfg, _StubEngine([0.2]))
    resp = rerank_client.client.post(
        "/v1/rerank", json={"model": "other/model", "query": "q", "documents": ["a"]}
    )
    assert resp.status_code == 400
    assert "model_not_found" in resp.text


@pytest.mark.parametrize(
    "body",
    [
        {"query": "q", "documents": ["a"], "top_n": -1},
        {"query": "q", "documents": ["a"], "unknown": 1},
        {"query": "q", "documents": [{"content": "a"}]},
        {"documents": ["a"]},
    ],
)
def test_rerank_rejects_malformed_requests(rerank_client, body):
    _configure(rerank_client.cfg, _StubEngine([0.2]))
    assert rerank_client.client.post("/v1/rerank", json=body).status_code in (400, 422)


def test_rerank_maps_document_overflow_to_400(rerank_client):
    from rapid_mlx.rerank import RerankDocumentTooLongError

    error = RerankDocumentTooLongError(index=1, observed_tokens=900, allowed_tokens=512)
    _configure(rerank_client.cfg, _StubEngine([], error=error))
    resp = rerank_client.client.post(
        "/v1/rerank", json={"query": "q", "documents": ["a", "b"]}
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "input_too_long"
    assert resp.json()["error"]["param"] == "documents"


def test_rerank_rejects_instruction_for_cross_encoder(rerank_client):
    from rapid_mlx.rerank import RerankInstructionNotSupportedError

    error = RerankInstructionNotSupportedError(
        "XLM-RoBERTa cross-encoder rerankers take no instruction"
    )
    _configure(rerank_client.cfg, _StubEngine([], error=error))
    resp = rerank_client.client.post(
        "/v1/rerank", json={"query": "q", "documents": ["a"], "instruction": "x"}
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["param"] == "instruction"


def test_rerank_maps_query_overflow_to_400(rerank_client):
    from rapid_mlx.rerank import RerankInputTooLongError

    error = RerankInputTooLongError(observed_tokens=900, allowed_tokens=512)
    _configure(rerank_client.cfg, _StubEngine([], error=error))
    resp = rerank_client.client.post(
        "/v1/rerank", json={"query": "q", "documents": ["a"]}
    )
    assert resp.status_code == 400
    assert "input_too_long" in resp.text


async def test_rerank_scoring_does_not_block_event_loop(monkeypatch):
    import httpx
    from fastapi import FastAPI

    from rapid_mlx import server
    from rapid_mlx.config import reset_config
    from rapid_mlx.routes.rerank import router

    started, release = threading.Event(), threading.Event()

    class _BlockingEngine(_StubEngine):
        def score(self, query, documents, *, instruction=None):
            started.set()
            release.wait(timeout=5)
            return super().score(query, documents, instruction=instruction)

    monkeypatch.setattr(server, "_rerank_engine", None)
    monkeypatch.setattr(server, "_rerank_model_locked", None)
    _configure(reset_config(), _BlockingEngine([0.5]))
    app = FastAPI()
    app.include_router(router)
    app.add_api_route("/ping", lambda: {"ok": True})
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
            rerank = asyncio.create_task(
                c.post("/v1/rerank", json={"query": "q", "documents": ["a"]})
            )
            assert await asyncio.to_thread(started.wait, 5)
            assert (await c.get("/ping")).status_code == 200
            assert not rerank.done()
            release.set()
            assert (await rerank).status_code == 200
    finally:
        release.set()
        reset_config()


class _ChatWorker:
    def __init__(self):
        self.calls = 0

    async def execute_on_model_worker(self, func, *args, **kwargs):
        self.calls += 1
        raise AssertionError("retrieval work used the chat model worker")


def _thread_recording_engine(scores, error=None):
    class _Engine(_StubEngine):
        def score(self, query, documents, *, instruction=None):
            self.threads.append(threading.current_thread().name)
            return super().score(query, documents, instruction=instruction)

    engine = _Engine(scores, error=error)
    engine.threads = []
    return engine


def test_rerank_uses_dedicated_worker_not_chat_worker(rerank_client):
    engine = _thread_recording_engine([0.5])
    _configure(rerank_client.cfg, engine)
    worker = _ChatWorker()
    rerank_client.cfg.engine = worker
    resp = rerank_client.client.post(
        "/v1/rerank", json={"query": "q", "documents": ["a"]}
    )
    assert resp.status_code == 200
    assert worker.calls == 0
    assert len(engine.threads) == 1
    assert engine.threads[0].startswith("mlx-retrieval")


def test_rerank_inference_error_propagates_once(rerank_client):
    engine = _thread_recording_engine([], error=RuntimeError("boom"))
    _configure(rerank_client.cfg, engine)
    worker = _ChatWorker()
    rerank_client.cfg.engine = worker
    resp = rerank_client.client.post(
        "/v1/rerank", json={"query": "q", "documents": ["a"]}
    )
    assert resp.status_code == 500
    assert resp.json() == {
        "error": {
            "message": "Rerank failed.",
            "type": "server_error",
            "code": "rerank_failed",
            "param": None,
        }
    }
    assert "boom" not in resp.text
    assert worker.calls == 0
    assert len(engine.threads) == 1


@pytest.mark.parametrize(
    ("limit", "count", "status"),
    [(2, 3, 400), (3, 3, 200), (0, 50, 200), (-1, 50, 200)],
)
def test_rerank_max_documents_error_policy(rerank_client, limit, count, status):
    engine = _StubEngine([0.5] * 50, overflow_policy="error")
    _configure(rerank_client.cfg, engine)
    rerank_client.cfg.rerank_max_documents = limit
    resp = rerank_client.client.post(
        "/v1/rerank", json={"query": "q", "documents": ["d"] * count}
    )
    assert resp.status_code == status
    if status == 400:
        assert "too_many_documents" in resp.text
        assert engine.calls == []


def test_rerank_max_documents_truncate_policy_rejects_excess(rerank_client):
    engine = _StubEngine([0.1, 0.9, 0.5])
    _configure(rerank_client.cfg, engine)
    rerank_client.cfg.rerank_max_documents = 3
    resp = rerank_client.client.post(
        "/v1/rerank", json={"query": "q", "documents": ["a", "b", "c", "d", "e"]}
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "too_many_documents"
    assert engine.calls == []


def test_rerank_max_documents_defaults_to_1000():
    from rapid_mlx import cli

    args = cli.build_parser().parse_args(["serve", "--rerank-model", RERANK_ID])
    assert args.rerank_max_documents == 1000
    assert args.rerank_max_length == "auto"
    assert args.rerank_overflow_policy == "truncate"
    assert args.rerank_instruction is None
    assert args.rerank_prompt is None
    limited = cli.build_parser().parse_args(
        [
            "serve",
            "--rerank-model",
            RERANK_ID,
            "--rerank-max-documents",
            "1000",
            "--rerank-max-length",
            "4096",
            "--rerank-overflow-policy",
            "error",
            "--rerank-instruction",
            "Find code",
            "--rerank-prompt",
            "Judge.",
        ]
    )
    assert cli._retrieval_only_incompatible_options(limited) == []


def test_models_route_lists_rerank_model(rerank_client):
    _configure(rerank_client.cfg, _StubEngine([]))
    listed = rerank_client.client.get("/v1/models").json()["data"]
    assert [(m["id"], m["capabilities"]) for m in listed] == [(RERANK_ID, ["rerank"])]
    single = rerank_client.client.get(f"/v1/models/{RERANK_ID}")
    assert single.status_code == 200
    assert single.json()["capabilities"] == ["rerank"]


# ---------------------------------------------------------------------------
# serve wiring
# ---------------------------------------------------------------------------


def test_main_dispatches_serve_with_only_rerank_model(capsys):
    from rapid_mlx import cli

    seen: list[Namespace] = []
    with (
        mock.patch.object(
            sys,
            "argv",
            ["rapid-mlx", "--no-banner", "serve", "--rerank-model", RERANK_ID],
        ),
        mock.patch.object(cli, "serve_command", side_effect=seen.append),
    ):
        cli.main()
    assert [(a.model, a.rerank_model) for a in seen] == [(None, RERANK_ID)]
    assert "a model is required" not in capsys.readouterr().err


def test_serve_command_routes_rerank_only_to_retrieval_mode(
    stub_serve_port_resolution,
):
    from rapid_mlx import cli

    args = cli.build_parser().parse_args(["serve", "--rerank-model", RERANK_ID])
    with (
        mock.patch.object(cli, "_serve_retrieval_only_mode") as retrieval_only,
        mock.patch.object(cli, "_ensure_model_downloaded") as download,
    ):
        cli.serve_command(args)
    retrieval_only.assert_called_once_with(args)
    download.assert_not_called()


def test_serve_command_rejects_primary_flags_in_rerank_only_mode(capsys):
    from rapid_mlx import cli

    args = cli.build_parser().parse_args(
        ["serve", "--rerank-model", RERANK_ID, "--lazy-load"]
    )
    with pytest.raises(SystemExit) as exc_info:
        cli.serve_command(args)
    assert exc_info.value.code == 2
    assert "--lazy-load" in capsys.readouterr().err


def test_retrieval_only_mode_loads_both_models(monkeypatch):
    import rapid_mlx
    from rapid_mlx import cli

    events: list[str] = []
    stub_server = types.ModuleType("rapid_mlx.server")
    stub_server.app = object()
    stub_server.configure_logging = lambda _level: "info"
    stub_server._resolve_api_key = lambda key: key
    stub_server.configure_cors_from_env = lambda _origins: None
    stub_server.configure_trusted_hosts = lambda _hosts: None
    stub_server._sync_config = lambda: None
    stub_server.load_embedding_model = object()
    stub_server.load_rerank_model = object()
    monkeypatch.setitem(sys.modules, "rapid_mlx.server", stub_server)
    monkeypatch.setattr(rapid_mlx, "server", stub_server, raising=False)
    monkeypatch.setattr(
        cli,
        "_load_embedding_model_or_exit",
        lambda args, fn: events.append(
            ("embed", fn is stub_server.load_embedding_model)
        ),
    )
    monkeypatch.setattr(
        cli,
        "_load_rerank_model_or_exit",
        lambda args, fn: events.append(("rerank", fn is stub_server.load_rerank_model)),
    )
    monkeypatch.setattr(cli, "_run_uvicorn", lambda *_a, **_k: events.append("run"))
    monkeypatch.setattr(cli, "_hard_exit_after_serve", lambda: None)
    args = types.SimpleNamespace(
        embedding_model="embed",
        rerank_model=RERANK_ID,
        host="127.0.0.1",
        port=8123,
        listen_fd=None,
        log_level="info",
        api_key=None,
        timeout=30,
        max_request_bytes=None,
        rate_limit=0,
        cors_origins=None,
    )
    cli._serve_retrieval_only_mode(args)
    assert events == [("embed", True), ("rerank", True), "run"]


def test_load_rerank_model_resolves_alias_and_wraps_errors(monkeypatch, capsys):
    from rapid_mlx import cli, model_aliases
    from rapid_mlx.rerank import UnsupportedRerankModelError

    monkeypatch.setattr(model_aliases, "resolve_model", lambda name: RERANK_ID)
    loaded: list[tuple] = []
    args = Namespace(
        rerank_model="short-alias",
        rerank_max_documents=1000,
        rerank_max_length="2048",
        rerank_overflow_policy="error",
        rerank_instruction="Find code",
        rerank_prompt="Judge.",
        rerank_model_family="qwen3",
    )
    cli._load_rerank_model_or_exit(
        args, lambda name, **kwargs: loaded.append((name, kwargs))
    )
    assert loaded == [
        (
            RERANK_ID,
            {
                "max_documents": 1000,
                "max_length": 2048,
                "overflow_policy": "error",
                "instruction": "Find code",
                "prompt": "Judge.",
                "model_family": "qwen3",
            },
        )
    ]
    assert args.rerank_model == RERANK_ID

    def _unsupported(_name, **_kwargs):
        raise UnsupportedRerankModelError("'x' is not a supported reranker")

    with pytest.raises(SystemExit) as exc_info:
        cli._load_rerank_model_or_exit(Namespace(rerank_model="x"), _unsupported)
    assert exc_info.value.code == 1
    assert "not a supported reranker" in capsys.readouterr().out

    with pytest.raises(SystemExit) as exc_info:
        cli._load_rerank_model_or_exit(
            Namespace(rerank_model="x", rerank_max_length="0"), _unsupported
        )
    assert exc_info.value.code == 2
    assert "--rerank-max-length" in capsys.readouterr().err


def test_legacy_server_parser_accepts_rerank_settings():
    from rapid_mlx.server import _build_parser

    args = _build_parser().parse_args(
        ["--rerank-instruction", "Find code", "--rerank-prompt", "Judge."]
    )
    assert (args.rerank_instruction, args.rerank_prompt) == ("Find code", "Judge.")
    assert args.rerank_max_documents == 1000
    assert args.embedding_max_inputs == 2048


def test_server_load_rerank_model_configures_engine(monkeypatch):
    import asyncio
    import threading

    from rapid_mlx import rerank, server
    from rapid_mlx.config import get_config, reset_config
    from rapid_mlx.routes._mlx_worker import run_on_mlx_worker

    created = []
    threads = []

    class _Engine:
        def __init__(self, name, **kwargs):
            created.append((name, kwargs))

        def load(self):
            threads.append(threading.current_thread().name)

        def score(self):
            threads.append(threading.current_thread().name)

    monkeypatch.setattr(rerank, "RerankEngine", _Engine)
    monkeypatch.setattr(server, "_emit_rerank_model_served_once", lambda _name: None)
    monkeypatch.setattr(server, "_rerank_engine", None)
    monkeypatch.setattr(server, "_rerank_model_locked", None)
    monkeypatch.setattr(server, "_rerank_max_documents", 0)
    reset_config()
    try:
        server.load_rerank_model(
            RERANK_ID,
            max_documents=5,
            max_length=512,
            overflow_policy="error",
            instruction="Find code",
            prompt="Judge.",
            model_family="qwen3",
        )
        assert created == [
            (
                RERANK_ID,
                {
                    "max_length": 512,
                    "overflow_policy": "error",
                    "instruction": "Find code",
                    "prompt": "Judge.",
                    "model_family": "qwen3",
                },
            )
        ]
        cfg = get_config()
        assert isinstance(cfg.rerank_engine, _Engine)
        assert (cfg.rerank_model_locked, cfg.rerank_max_documents) == (RERANK_ID, 5)
        asyncio.run(run_on_mlx_worker(cfg.rerank_engine.score))
        assert len(threads) == 2
        assert threads[0].startswith("mlx-retrieval")
        assert threads[0] == threads[1]
    finally:
        reset_config()


def test_server_load_rerank_model_propagates_load_error(monkeypatch):
    from rapid_mlx import rerank, server

    error = LookupError("load failed unchanged")

    class _Engine:
        def __init__(self, _name, **_kwargs):
            pass

        def load(self):
            raise error

    monkeypatch.setattr(rerank, "RerankEngine", _Engine)

    with pytest.raises(LookupError, match="load failed unchanged") as exc_info:
        server.load_rerank_model(RERANK_ID)
    assert exc_info.value is error


def test_health_reports_rerank_only_server():
    from rapid_mlx.config import reset_config
    from rapid_mlx.routes import health

    cfg = reset_config()
    try:
        cfg.rerank_engine = object()
        cfg.rerank_model_locked = RERANK_ID
        cfg.ready = True
        full = asyncio.run(health.health())
        assert (full["model_loaded"], full["model_type"]) == (True, "rerank")
        cfg.embedding_engine = object()
        assert asyncio.run(health.health())["model_type"] == "embedding+rerank"
    finally:
        reset_config()


def test_cua_only_rejects_rerank_model():
    from rapid_mlx import cli

    args = Namespace(rerank_model=RERANK_ID)
    assert cli._cua_only_incompatible_options(args) == ["--rerank-model"]


_AUDIO_RERANK_ID = "mlx-community/Qwen3-Reranker-0.6B-mxfp8"


def test_audio_mode_preloads_configured_reranker(monkeypatch):
    # The audio serve path needs the MLX audio stack, so this lives in the
    # MLX-only rerank tests rather than the Linux-run serve tests.
    from rapid_mlx import cli, server
    from tests.test_audio_alias_registry import _make_serve_args

    calls = []
    monkeypatch.setattr(cli, "_run_uvicorn", lambda *_a, **_k: None)
    monkeypatch.setattr(cli, "_port_preflight_or_die", lambda *_a, **_k: None)
    monkeypatch.setattr(cli, "_offline_hub_mode_active", lambda: False)
    monkeypatch.setattr("rapid_mlx.model_downloads.disabled", lambda: False)
    monkeypatch.setattr(
        cli,
        "_load_rerank_model_or_exit",
        lambda args, loader: calls.append((args.rerank_model, loader)),
    )
    args = _make_serve_args("kokoro")
    args.rerank_model = _AUDIO_RERANK_ID

    cli.serve_command(args)

    assert calls == [(_AUDIO_RERANK_ID, server.load_rerank_model)]


def test_jina_zero_projection_scores_stay_finite():
    import math

    import mlx.core as mx

    engine = _jina_engine()
    engine._model.projector = lambda x: mx.zeros_like(x)

    result = engine.score("qa", ["xa", "xb"])

    assert all(math.isfinite(score) for score in result.scores)
    assert all(0.0 <= score <= 1.0 for score in result.scores)


def test_jina_all_opposite_blocks_fall_back_to_unweighted_mean(monkeypatch):
    import math

    import mlx.core as mx

    engine = _jina_engine()

    def opposite_block(query, docs, instruction):
        query_embed = mx.array([1.0, 0.0])
        doc_embeds = mx.array([[-1.0, 0.0]] * len(docs))
        return query_embed, doc_embeds, 1

    monkeypatch.setattr(engine, "_embed_listwise_block", opposite_block)

    result = engine.score("qa", ["xa", "xb"])

    assert all(math.isfinite(score) for score in result.scores)
    assert result.scores == pytest.approx([0.0, 0.0], abs=1e-6)


def test_jina_custom_prompt_strips_reserved_markers():
    engine = _jina_engine()
    engine.prompt = "Rank<|embed_token|> carefully<|rerank_token|>."
    prompt = engine._listwise_prompt("q", ["d"], None)
    assert prompt.startswith("<|im_start|>system\nRank carefully.<|im_end|>\n")
    assert prompt.count("<|embed_token|>") == 1
    assert prompt.count("<|rerank_token|>") == 1


@pytest.mark.parametrize("policy", ["truncate", "error"])
def test_rerank_default_cap_rejects_1001(rerank_client, policy):
    engine = _StubEngine([], overflow_policy=policy)
    _configure(rerank_client.cfg, engine)
    response = rerank_client.client.post(
        "/v1/rerank", json={"query": "q", "documents": ["d"] * 1001}
    )
    assert response.status_code == 400
    error = response.json()["error"]
    assert (error["code"], error["param"]) == ("too_many_documents", "documents")
    assert all(
        s in error["message"] for s in ["1001", "1000", "--rerank-max-documents"]
    )
    assert not engine.calls


def test_rerank_busy_returns_retry_after(rerank_client, monkeypatch):
    from rapid_mlx.routes import rerank
    from rapid_mlx.routes._mlx_worker import RetrievalWorkerBusyError

    _configure(rerank_client.cfg, _StubEngine([]))
    monkeypatch.setattr(
        rerank,
        "run_on_mlx_worker",
        mock.AsyncMock(side_effect=RetrievalWorkerBusyError),
    )
    telemetry = mock.Mock()
    monkeypatch.setattr(
        "rapid_mlx.telemetry.inference.emit_completed_request", telemetry
    )
    response = rerank_client.client.post(
        "/v1/rerank", json={"query": "q", "documents": ["d"]}
    )
    assert response.status_code == 503
    assert response.headers["Retry-After"] == "1"
    assert response.json()["error"]["code"] == "server_busy"
    telemetry.assert_not_called()


@pytest.mark.parametrize("count", [50, 100])
def test_jina_fixed_prompt_tokenization_is_bounded(count):
    engine = _jina_engine()
    encode = engine._tokenizer.encode
    engine._tokenizer.encode = mock.Mock(wraps=encode)
    engine.score("qa", ["xa"] * count)
    full_prompts = [
        c.args[0]
        for c in engine._tokenizer.encode.call_args_list
        if "<|im_start|>system" in c.args[0]
    ]
    # Empty singleton, two empties, exact block validation, block inference.
    assert len(full_prompts) == 4


@pytest.mark.parametrize(
    "document", ["xa", "", "a<|embed_token|>b<|rerank_token|>", "a\nb"]
)
def test_jina_incremental_singleton_length_matches_exact(document):
    engine = _jina_engine()
    encode = lambda text: len(engine._tokenizer.encode(text, add_special_tokens=False))
    base = encode(engine._listwise_prompt("qa", [""], "inst"))
    estimate = (
        base
        - encode(engine._listwise_passage(""))
        + encode(engine._listwise_passage(document))
    )
    assert estimate == encode(engine._listwise_prompt("qa", [document], "inst"))


@pytest.mark.parametrize("policy", ["truncate", "error"])
def test_jina_singleton_exact_validation_handles_underestimate(policy):
    from rapid_mlx.rerank import RerankDocumentTooLongError

    engine = _jina_engine(context_length=750)
    engine.overflow_policy = policy
    original = engine._tokenizer.encode

    def nonadditive(text, **kwargs):
        ids = original(text, **kwargs)
        # Only the full prompt has a large boundary penalty for nonempty docs.
        if "<|im_start|>system" in text and '<passage id="0">\nx' in text:
            ids += [ord(" ")] * 100
        return ids

    engine._tokenizer.encode = nonadditive
    if policy == "error":
        with pytest.raises(RerankDocumentTooLongError):
            engine.score("qa", ["xa"])
    else:
        result = engine.score("qa", ["xa"])
        assert len(result.scores) == 1
        assert engine.num_truncations == 1
        assert max(engine._model.prompt_lengths) <= 750


@pytest.mark.parametrize("max_length", [20, 2048])
def test_jina_singleton_rechecks_truncation_without_double_counting(
    monkeypatch, max_length
):
    engine = _jina_engine(max_length=max_length)
    encode = engine._tokenizer.encode
    decode = engine._tokenizer.decode
    empty_length = len(
        encode(engine._listwise_prompt("qa", [""], None), add_special_tokens=False)
    )
    engine._context_length = empty_length + 105

    def nonadditive(text, **kwargs):
        ids = encode(text, **kwargs)
        # Keep the singleton estimate small so validated_blocks handles overflow.
        if "<|im_start|>system" in text and '<passage id="0">\nx' in text:
            ids += [ord(" ")] * 100
        return ids

    five_token_decodes = []

    def expanding_decode(ids):
        text = decode(ids)
        if len(ids) == 5:
            five_token_decodes.append(text)
            # The search probe fits, but decoding its final result expands it.
            if len(five_token_decodes) == 2:
                return text + "x"
        return text

    monkeypatch.setattr(engine._tokenizer, "encode", nonadditive)
    monkeypatch.setattr(engine._tokenizer, "decode", expanding_decode)

    result = engine.score("qa", ["x" * 40])

    assert len(five_token_decodes) >= 3
    assert len(result.scores) == 1
    assert engine._model.calls == 1
    assert engine._model.prompt_lengths == [engine._context_length]
    assert engine.num_truncations == 1


def test_jina_near_limit_estimate_uses_exact_fitting_singleton():
    engine = _jina_engine(context_length=710)
    result = engine.score("qa", ["xa"])
    assert len(result.scores) == 1
    assert engine.num_truncations == 0


@pytest.mark.requires_mlx
def test_local_jina_refuses_checkpoint_selector_before_model_load(
    tmp_path, monkeypatch
):
    import mlx_lm.utils

    from rapid_mlx.rerank import UnsupportedRerankModelError, _load_jina

    monkeypatch.delenv("RAPID_MLX_TRUST_REMOTE_CODE", raising=False)
    (tmp_path / "config.json").write_text(
        '{"model_type":"qwen3","model_file":"custom.py"}'
    )
    marker = tmp_path / "executed"
    (tmp_path / "custom.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\n"
    )
    monkeypatch.setattr(
        mlx_lm.utils,
        "load_model",
        lambda *a, **k: pytest.fail("model loader must not run"),
    )
    with pytest.raises(
        UnsupportedRerankModelError, match="model_file.*executable Python"
    ):
        _load_jina(str(tmp_path))
    assert not marker.exists()


@pytest.mark.requires_mlx
def test_local_jina_positive_checked_load(tmp_path, monkeypatch):
    import mlx_lm.utils

    from rapid_mlx.rerank import _jina_model_classes, _load_jina

    monkeypatch.delenv("RAPID_MLX_TRUST_REMOTE_CODE", raising=False)
    (tmp_path / "config.json").write_text('{"model_type":"qwen3"}')
    model, tokenizer = object(), object()

    def load(path, **kwargs):
        assert path == tmp_path
        assert kwargs == {"get_model_classes": _jina_model_classes}
        return model, {}

    def load_tokenizer(path, tokenizer_config):
        assert path == tmp_path
        assert tokenizer_config == {"trust_remote_code": False}
        return tokenizer

    monkeypatch.setattr(mlx_lm.utils, "load_model", load)
    monkeypatch.setattr(mlx_lm.utils, "load_tokenizer", load_tokenizer)
    assert _load_jina(str(tmp_path)) == (model, tokenizer)
