# SPDX-License-Identifier: Apache-2.0
"""Gemma stop fallback using config excerpts from cached production tokenizers."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.requires_mlx
pytest.importorskip("mlx_lm")

from mlx_lm.tokenizer_utils import TokenizerWrapper
from tokenizers import Tokenizer, models
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from rapid_mlx.mllm_scheduler import MLLMScheduler
from rapid_mlx.scheduler import Scheduler, SchedulerConfig
from rapid_mlx.utils.tokenizer import augment_eos_token_ids_from_generation_config


def _tokenizer(tmp_path, family):
    fixture = json.loads(
        (
            Path(__file__).parent / "fixtures" / "gemma_stop" / f"{family}.json"
        ).read_text()
    )
    # Keep the actual special-token IDs and config; replace the large ordinary
    # vocabulary with inert placeholders so the regression runs offline.
    vocab = {f"placeholder{i}": i for i in range(107)}
    for entry in fixture["added_tokens"]:
        del vocab[f"placeholder{entry['id']}"]
        vocab[entry["content"]] = entry["id"]
    backend = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    backend.save(str(tmp_path / "tokenizer.json"))
    config = fixture["tokenizer_config"]
    config["added_tokens_decoder"] = {
        str(entry["id"]): {k: v for k, v in entry.items() if k != "id"}
        for entry in fixture["added_tokens"]
    }
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(config))
    return AutoTokenizer.from_pretrained(tmp_path, local_files_only=True)


@pytest.mark.parametrize(
    "family,expected", [("gemma3", {1, 106}), ("gemma4", {1, 50, 106})]
)
@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize(
    "generation", [None, {}, {"eos_token_id": 1}, {"eos_token_id": [1, 42]}]
)
def test_gemma_missing_or_incomplete_generation_eos(
    tmp_path, family, expected, wrapped, generation
):
    tok = _tokenizer(tmp_path, family)
    if family == "gemma3":
        assert "<end_of_turn>" in tok.apply_chat_template(
            [{"role": "user", "content": "Hello"}], tokenize=False
        )
    assert tok.eos_token_id == 1
    if wrapped:
        tok = TokenizerWrapper(tok)
    if generation is not None:
        (tmp_path / "generation_config.json").write_text(json.dumps(generation))
    if generation == {"eos_token_id": [1, 42]}:
        expected = expected | {42}
    for _ in range(2):  # Repeated loads must preserve and deduplicate the union.
        augment_eos_token_ids_from_generation_config(tok, str(tmp_path))
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.tokenizer = scheduler._actual_tokenizer = tok
    scheduler.config = SchedulerConfig()
    assert scheduler._get_stop_tokens() == expected
    mllm = MLLMScheduler.__new__(MLLMScheduler)
    mllm.processor = SimpleNamespace(tokenizer=tok)
    assert mllm._get_stop_tokens() == expected
    assert tok.eos_token_id == 1
    if wrapped:
        assert tok.eos_token_ids == expected


def test_other_family_with_same_vocabulary_is_unchanged(tmp_path):
    gemma = _tokenizer(tmp_path, "gemma3")
    other = PreTrainedTokenizerFast(
        tokenizer_object=gemma.backend_tokenizer, eos_token="<eos>"
    )
    tok = TokenizerWrapper(other)
    augment_eos_token_ids_from_generation_config(tok, str(tmp_path))
    assert tok.eos_token_ids == {1}


def test_gemma_terminators_use_actual_vocab_ids(tmp_path):
    tok = _tokenizer(tmp_path, "gemma4")
    tok.get_vocab = lambda: {"<eos>": 1, "<turn|>": 700, "<|tool_response>": 701}
    wrapped = TokenizerWrapper(tok)
    augment_eos_token_ids_from_generation_config(wrapped, str(tmp_path))
    assert wrapped.eos_token_ids == {1, 700, 701}


def test_gemma_without_turn_markers_does_not_add_unknown_token(tmp_path):
    tok = _tokenizer(tmp_path, "gemma3")
    tok.get_vocab = lambda: {"<eos>": 1, "<unk>": 3}
    wrapped = TokenizerWrapper(tok)
    augment_eos_token_ids_from_generation_config(wrapped, str(tmp_path))
    assert wrapped.eos_token_ids == {1}
