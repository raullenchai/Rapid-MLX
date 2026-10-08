# SPDX-License-Identifier: Apache-2.0
"""Decider prompt layout, scoring and backend routing without the checkpoint."""

from __future__ import annotations

import json
import string
import sys
from types import SimpleNamespace

import mlx.core as mx
import pytest
from fastapi.testclient import TestClient

from rapid_mlx.cli import _resolve_system_one_backend, build_parser
from rapid_mlx.system_one import decider as decider_module
from rapid_mlx.system_one.backends import DeciderBackend
from rapid_mlx.system_one.decider import (
    DeciderScorer,
    load_decider,
    render_question,
    render_state,
)
from rapid_mlx.system_one.schema import Question
from rapid_mlx.system_one.server import create_app

_LABELS = list(string.ascii_uppercase) + [
    a + b for a in string.ascii_uppercase for b in string.ascii_uppercase
]
_LABEL_BASE = 1000


class FakeTokenizer:
    """One token per label when a label stands alone, else one per character."""

    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        if text in _LABELS:
            return [_LABEL_BASE + _LABELS.index(text)]
        return [ord(char) % 900 + 1 for char in text]


class FakeBackbone:
    """Hidden state at a position is the one-hot of that position's token."""

    def __init__(self, vocab: int = 2000):
        self.vocab = vocab
        self.calls: list[tuple[int, int]] = []

    def embed_tokens(self, ids):
        return mx.eye(self.vocab)[ids]

    def __call__(self, ids):
        self.calls.append(tuple(ids.shape))
        return mx.eye(self.vocab)[ids]


def _scorer(settings=None, backbone=None):
    backbone = backbone or FakeBackbone()
    settings = settings or {"temperature": 1.0}
    return DeciderScorer(
        SimpleNamespace(model=backbone), FakeTokenizer(), settings
    ), backbone


def test_render_state_numbers_long_lists_only():
    assert render_state("plain") == "plain"
    assert json.loads(render_state({"a": [1, {"b": 2}]})) == {"a": [1, {"b": 2}]}
    numbered = json.loads(render_state([{"k": i} if i else i for i in range(8)]))
    assert numbered[0] == {"_index": 0, "value": 0}
    assert numbered[3] == {"_index": 3, "k": 3}


def test_render_question_layouts():
    choice = render_question(
        Question(
            type="choice",
            instructions="Pick",
            criteria={"a": "first", "b": None, "c": {"n": 1}},
        )
    )
    assert choice["keys"] == ["a", "b", "c"]
    assert choice["options"] == ["a: first", "b", 'c: {"n": 1}']
    score = render_question(
        Question(type="score", instructions="Rate", criteria=["low", "high"])
    )
    assert score["keys"] == ["0", "1"]
    assert score["options"] == ["0: low", "1: high"]
    assert score["levels"] == ["low", "high"]
    plain = render_question(Question(type="noul", instructions="Sure?"))
    assert plain["options"] == ["no", "yes"]
    described = render_question(
        Question(
            type="noul", instructions="Sure?", criteria={"true": "it is", "false": ""}
        )
    )
    assert described["options"] == ["no", "yes: it is"]
    assert described["keys"] == ["false", "true"]


def test_render_question_rejects_shapes_decider_was_not_trained_on():
    with pytest.raises(ValueError, match="at least two options"):
        render_question(
            Question(type="choice", instructions="Pick", criteria={"only": None})
        )
    with pytest.raises(ValueError, match="at most 10 levels"):
        render_question(
            Question(type="score", instructions="Rate", criteria=list(range(11)))
        )


def test_scorer_requires_distinct_single_token_labels():
    class Collapsing(FakeTokenizer):
        def encode(self, text, add_special_tokens=False):
            return [7] if text in _LABELS else super().encode(text)

    with pytest.raises(ValueError, match="255 distinct label tokens"):
        DeciderScorer(
            SimpleNamespace(model=FakeBackbone()), Collapsing(), {"temperature": 1.0}
        )


def test_prompt_layout_inline_and_tokenized_labels():
    scorer, _ = _scorer()
    tok = FakeTokenizer()
    context = tok.encode("Context:\nstate")
    inline = scorer._prompt(context, "Q", ["x", "y"])
    assert inline == context + tok.encode(
        "\n\nQuestion: Q\nOptions:\n(A) x\n(B) y\nAnswer: ("
    )
    options = [f"o{index}" for index in range(11)]
    wide = scorer._prompt(context, "Q", options)
    expected = context + tok.encode("\n\nQuestion: Q\nOptions:")
    for index, option in enumerate(options):
        expected += tok.encode("\n(") + [_LABEL_BASE + index]
        expected += tok.encode(f") {option}")
    assert wide == expected + tok.encode("\nAnswer: (")


def test_score_reads_label_logits_at_the_last_position():
    # Every prompt ends in "(", which is not a label token, so the one-hot
    # readout is flat over the offered labels: a uniform distribution.
    scorer, backbone = _scorer({"temperature": 2.0, "temperature_by_type": {}})
    questions = {
        "pick": Question(
            type="choice", instructions="Pick", criteria={"a": None, "b": None}
        ),
        "sure": Question(type="noul", instructions="Sure?"),
        "rate": Question(type="score", instructions="Rate", criteria=["l", "m", "h"]),
    }
    answers, tokens = scorer.score({"k": "v"}, questions)
    assert answers["pick"] == (["a", "b"], pytest.approx([0.5, 0.5]))
    assert answers["sure"] == (["false", "true"], pytest.approx([0.5, 0.5]))
    assert answers["rate"][0] == ["0", "1", "2"]
    assert answers["rate"][1] == pytest.approx([1 / 3] * 3)
    # Three rows share one padded batch; the context is counted once.
    assert len(backbone.calls) == 1 and backbone.calls[0][0] == 3
    assert backbone.calls[0][1] % 64 == 0
    context = len(FakeTokenizer().encode("Context:\n" + render_state({"k": "v"})))
    assert context < tokens < 5 * context + 400


def test_score_isolated_levels_normalize_per_level_fits(monkeypatch):
    scorer, backbone = _scorer({"temperature": 1.0, "isolated_levels": True})
    seen = {}

    def fake_rows(prompts, widths, temperatures):
        seen.update(widths=widths, temperatures=temperatures)
        return [[0.9, 0.1], [0.7, 0.3], [0.4, 0.6]]

    monkeypatch.setattr(scorer, "_score_rows", fake_rows)
    answers, _ = scorer.score(
        "s",
        {"rate": Question(type="score", instructions="Rate", criteria=["a", "b", "c"])},
    )
    assert seen["widths"] == [2, 2, 2]
    assert answers["rate"][1] == pytest.approx([0.1, 0.3, 0.6])
    monkeypatch.setattr(scorer, "_score_rows", lambda *a: [[1.0, 0.0]] * 3)
    answers, _ = scorer.score(
        "s",
        {"rate": Question(type="score", instructions="Rate", criteria=["a", "b", "c"])},
    )
    assert answers["rate"][1] == [0.0, 0.0, 0.0]


def test_score_uses_per_type_temperature_and_state_limit(monkeypatch):
    scorer, _ = _scorer(
        {
            "temperature": 1.0,
            "temperature_by_type": {"noul": 3.0},
            "max_state_tokens": 4,
        }
    )
    seen = {}

    def fake_rows(prompts, widths, temperatures):
        seen.update(prompts=prompts, temperatures=temperatures)
        return [[0.5, 0.5], [0.5, 0.5]]

    monkeypatch.setattr(scorer, "_score_rows", fake_rows)
    _, tokens = scorer.score(
        "a long state that is cut",
        {
            "sure": Question(type="noul", instructions="Sure?"),
            "pick": Question(
                type="choice", instructions="Pick", criteria={"a": None, "b": None}
            ),
        },
    )
    assert seen["temperatures"] == [3.0, 1.0]
    assert all(
        prompt[:4] == FakeTokenizer().encode("Cont") for prompt in seen["prompts"]
    )
    # The rows share the state and the "\n\nQuestion: " lead-in; that shared
    # prefix is counted once.
    shared = 4 + len("\n\nQuestion: ")
    assert tokens == shared + sum(len(prompt) - shared for prompt in seen["prompts"])


def test_score_rows_splits_batches_over_the_token_budget(monkeypatch):
    scorer, backbone = _scorer()
    monkeypatch.setattr(decider_module, "_BATCH_TOKEN_BUDGET", 128)
    prompts = [[5] * 60, [5] * 60, [5] * 60]
    results = scorer._score_rows(prompts, [2, 2, 2], [1.0, 1.0, 1.0])
    assert len(results) == 3
    assert [call[0] for call in backbone.calls] == [2, 1]


def test_scorer_built_on_one_thread_scores_on_another():
    # The server builds the backend on the main thread and answers on worker
    # threads. MLX keeps a lazy array bound to the thread that created it.
    import threading

    scorer, _ = _scorer()
    outcome = {}

    def work():
        try:
            outcome["answers"] = scorer.score(
                "state", {"sure": Question(type="noul", instructions="Sure?")}
            )[0]
        except Exception as exc:  # pragma: no cover - only on regression
            outcome["error"] = exc

    worker = threading.Thread(target=work)
    worker.start()
    worker.join()
    assert "error" not in outcome, outcome.get("error")
    assert outcome["answers"]["sure"][1] == pytest.approx([0.5, 0.5])


def test_scorer_defaults_pad_token_when_tokenizer_has_none():
    class NoPad(FakeTokenizer):
        pad_token_id = None

    scorer = DeciderScorer(
        SimpleNamespace(model=FakeBackbone()), NoPad(), {"temperature": 1.0}
    )
    assert scorer._pad_token_id == 0


def test_load_decider_validates_checkpoint_metadata(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="no config.json"):
        load_decider(tmp_path)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"model_type": "qwen3_5"}))
    with pytest.raises(ValueError, match="model_type 'decider2'"):
        load_decider(tmp_path)
    config.write_text(json.dumps({"model_type": "decider2", "decision_config": {}}))
    with pytest.raises(ValueError, match="decision_config.temperature"):
        load_decider(tmp_path)

    config.write_text(
        json.dumps({"model_type": "decider2", "decision_config": {"temperature": 1.1}})
    )
    observed = {}
    language_model = object()

    def fake_load_model(path, get_model_classes):
        from mlx_lm.models import qwen3_5

        observed["path"] = path
        assert get_model_classes(config={}) == (qwen3_5.Model, qwen3_5.ModelArgs)
        return SimpleNamespace(language_model=language_model), {}

    import mlx_lm.utils

    monkeypatch.setattr(mlx_lm.utils, "load_model", fake_load_model)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda path: ("tok", path))
        ),
    )
    model, tokenizer, settings = load_decider(tmp_path)
    assert model is language_model
    assert tokenizer == ("tok", str(tmp_path))
    assert settings == {"temperature": 1.1}
    assert observed["path"] == tmp_path


def _backend(scored, tokens=42):
    backend = object.__new__(DeciderBackend)
    backend.default_model = "decider-2b"
    backend.repo_id = "nativ-community/decider-2b"
    backend._lock = __import__("threading").Lock()
    backend._scorer = SimpleNamespace(score=lambda state, questions: (scored, tokens))
    return backend


def test_decider_backend_answers_in_the_system_one_wire_shape():
    questions = {
        "pick": Question(
            type="choice", instructions="Pick", criteria={"a": None, "b": None}
        ),
        "sure": Question(type="noul", instructions="Sure?"),
        "rate": Question(type="score", instructions="Rate", criteria=["l", "h"]),
    }
    backend = _backend(
        {
            "pick": (["a", "b"], [0.8, 0.2]),
            "sure": (["false", "true"], [0.3, 0.7]),
            "rate": (["0", "1"], [0.25, 0.75]),
        }
    )
    result = backend.answer("state", questions, "decider-2b", 1.0)
    assert result["model"] == "decider-2b"
    assert result["answers"]["pick"] == {
        "type": "choice",
        "choice": "a",
        "confidence": 0.8,
        "probabilities": {"a": 0.8, "b": 0.2},
    }
    assert result["answers"]["sure"] == {"type": "noul", "noul": 0.7}
    assert result["answers"]["rate"]["score"] == 0.75
    assert result["answers"]["rate"]["confidence"] == 0.75
    assert result["answers"]["rate"]["legend"] == {"0": "l", "1": "h"}
    assert result["usage"] == {
        "billing_units": 3,
        "input_tokens": 42,
        "output_tokens": 0,
    }
    # The pinned repository id is accepted as a model name too.
    assert backend.answer("s", questions, "nativ-community/decider-2b", 1.0)
    with pytest.raises(KeyError, match="unknown model"):
        backend.answer("s", questions, "other", 1.0)
    with pytest.raises(ValueError, match="temperature=1"):
        backend.answer("s", questions, "decider-2b", 0.5)


def test_decider_backend_ranks_and_lists_models():
    backend = _backend({"rank": (["0", "1", "2"], [0.2, 0.5, 0.3])})
    ranked = backend.rank("ctx", None, ["x", "y", "z"], "decider-2b", 1.0)
    assert [item["candidate"] for item in ranked] == ["y", "z", "x"]
    assert [item["rank"] for item in ranked] == [1, 2, 3]
    assert backend.models() == [
        {
            "name": "decider-2b",
            "backend": "decider-mlx",
            "hf_id": "nativ-community/decider-2b",
            "description": "Decider typed decisions on native MLX",
        }
    ]
    client = TestClient(create_app(backend))
    response = client.post(
        "/v1/rank", json={"context": "c", "question": "q", "answers": ["x", "y", "z"]}
    )
    assert response.status_code == 200
    assert response.json()["ranked"][0]["candidate"] == "y"


def test_decider_backend_resolves_pinned_alias_and_local_directory(
    tmp_path, monkeypatch
):
    import rapid_mlx._mirror as mirror

    loaded = []
    monkeypatch.setattr(
        mirror,
        "pinned_snapshot_download",
        lambda repo, revision: f"/snap/{repo}@{revision}",
    )
    monkeypatch.setattr(
        decider_module,
        "load_decider",
        lambda path: loaded.append(path) or ("model", "tok", {"temperature": 1.0}),
    )
    monkeypatch.setattr(
        decider_module, "DeciderScorer", lambda *args: SimpleNamespace(args=args)
    )
    devices = []
    monkeypatch.setattr(mx, "set_default_device", devices.append)

    for name in ("decider-2b", "nativ-community/decider-2b"):
        backend = DeciderBackend(name)
        assert backend.default_model == "decider-2b"
        assert backend.repo_id == "nativ-community/decider-2b"
    assert loaded[0] == (
        "/snap/nativ-community/decider-2b@acbae4ecce4dbcc0aea8c5a501c9f54008458a70"
    )
    local = DeciderBackend(str(tmp_path), device="cpu")
    assert local.default_model == tmp_path.name
    assert loaded[-1] == str(tmp_path)
    assert devices == [mx.gpu, mx.gpu, mx.cpu]

    with pytest.raises(ValueError, match="unknown Decider model"):
        DeciderBackend("someone/decider-2b")
    with pytest.raises(ValueError, match="'gpu' or 'cpu'"):
        DeciderBackend("decider-2b", device="npu")


def test_decider_cli_selects_and_starts_the_backend(monkeypatch):
    import rapid_mlx._uvicorn as uvicorn_module
    import rapid_mlx.cli as cli_module
    import rapid_mlx.system_one.backends as backend_module

    for model in ("decider-2b", "nativ-community/decider-2b"):
        assert _resolve_system_one_backend(model, "auto") == "decider"
    assert _resolve_system_one_backend("org/decider-2b-like", "auto") == "laya"

    observed = {}

    def fake_backend(model, *, device):
        observed.update(model=model, device=device)
        return SimpleNamespace(default_model="decider-2b")

    monkeypatch.setattr(backend_module, "DeciderBackend", fake_backend)
    monkeypatch.setattr(cli_module, "_port_preflight_or_die", lambda *a, **k: None)
    monkeypatch.setattr(
        uvicorn_module,
        "run_uvicorn",
        lambda app, **kwargs: observed.update(kwargs=kwargs),
    )
    args = build_parser().parse_args(
        ["system-one", "/models/my-decider", "--backend", "decider", "--port", "8702"]
    )
    cli_module.system_one_command(args)
    assert observed["model"] == "/models/my-decider"
    assert observed["device"] == "gpu"
    assert observed["kwargs"]["port"] == 8702
