# SPDX-License-Identifier: Apache-2.0
"""OpenJev prompt layout, letter readout and backend routing without the checkpoint."""

from __future__ import annotations

import json
import math
import threading
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.requires_mlx
mx = pytest.importorskip("mlx.core")

from rapid_mlx.cli import _resolve_system_one_backend, build_parser
from rapid_mlx.system_one import openjev as openjev_module
from rapid_mlx.system_one.backends import OpenJevBackend
from rapid_mlx.system_one.openjev import (
    LETTERS,
    MAX_PROMPT_TOKENS,
    NOUL_TEMPERATURE,
    READOUT_TEMPERATURE,
    OpenJevScorer,
    choice_confidence,
    compose_groups,
    load_openjev,
    noul_probability,
    openjev_answer,
    option_groups,
    render_prompt,
    render_question,
    render_state,
    score_confidence,
)
from rapid_mlx.system_one.schema import Question
from rapid_mlx.system_one.server import create_app

_LETTER_BASE = 5000
_VOCAB = 5100


class FakeTokenizer:
    """One token per option letter standing alone, else one token per character."""

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        if text in LETTERS:
            return [_LETTER_BASE + LETTERS.index(text)]
        return [ord(char) % 4000 + 1 for char in text]

    def apply_chat_template(
        self, messages, tokenize, add_generation_prompt, enable_thinking
    ):
        assert (tokenize, add_generation_prompt, enable_thinking) == (
            False,
            True,
            False,
        )
        assert [message["role"] for message in messages] == ["user"]
        return "<user>" + messages[0]["content"] + "<assistant>"


class FakeCache:
    def __init__(self):
        self.state = mx.zeros((1,))
        self.seen: list[int] = []


class FakeModel:
    """Letter logits are a fixed function of every token read so far."""

    def __init__(self, logits_for=None):
        self.calls: list[int] = []
        self._logits_for = logits_for or self._default

    @staticmethod
    def _default(seen):
        total = sum((index + 1) * token for index, token in enumerate(seen))
        return [((total * (letter + 3)) % 97) / 10.0 for letter in range(len(LETTERS))]

    def make_cache(self):
        return [FakeCache()]

    def __call__(self, ids, cache):
        tokens = ids[0].tolist()
        self.calls.append(len(tokens))
        cache[0].seen.extend(tokens)
        row = [-50.0] * _VOCAB
        for letter, value in enumerate(self._logits_for(cache[0].seen)):
            row[_LETTER_BASE + letter] = value
        return mx.array([[row] * len(tokens)])


def _scorer(model=None, limit=MAX_PROMPT_TOKENS):
    return OpenJevScorer(model or FakeModel(), FakeTokenizer(), limit)


def _softmax(values, temperature=READOUT_TEMPERATURE):
    weights = [math.exp(value / temperature) for value in values]
    return [weight / sum(weights) for weight in weights]


def test_render_state_keeps_text_and_rejects_images():
    assert render_state("plain text") == "plain text"
    assert render_state(12) == "12"
    assert render_state({"note": "héllo", "screenshot": "short"}) == (
        '{"note": "héllo", "screenshot": "short"}'
    )
    for key, value in (
        ("screenshot", "data:image/png;base64,AAAA"),
        ("image", "A" * 2001),
    ):
        with pytest.raises(ValueError, match=f"state.{key} holds an image"):
            render_state({key: value})


def test_render_question_follows_the_release_layouts():
    choice = Question(
        type="choice",
        instructions={"goal": "open item 3", "done": False},
        criteria={"a": None, "b": "second", "c": {"k": "é"}},
    )
    # Structured instructions are shown as the Python literal the release used.
    assert render_question(choice) == (
        "{'goal': 'open item 3', 'done': False}",
        [("a", ""), ("b", "second"), ("c", '{"k": "é"}')],
    )
    score = Question(type="score", instructions="How urgent?", criteria=["low", None])
    assert render_question(score) == (
        "How urgent? Rate along the ordered levels below (lowest first).",
        [("0", "low"), ("1", "")],
    )
    assert render_question(Question(type="noul", instructions="Angry?")) == (
        "Angry?",
        [("yes", "The statement is true."), ("no", "The statement is false.")],
    )
    described = Question(
        type="noul", instructions="Angry?", criteria={"true": "shouting", "false": ""}
    )
    assert render_question(described)[1] == [
        ("yes", "shouting"),
        ("no", "The statement is false."),
    ]
    with pytest.raises(ValueError, match="need instructions"):
        render_question(Question(type="noul"))


def test_render_prompt_is_the_release_text_lane_prompt():
    assert render_prompt("S", "Q?", [("refund", "wants money"), ("other", "")]) == (
        "State:\nS\n\nQuestion: Q?\nOptions:\n[A] refund: wants money\n[B] other: "
        "\n\nAnswer with the letter of the best option only."
    )


def test_option_groups_split_long_lists_into_near_equal_readouts():
    assert option_groups(52) == [range(52)]
    assert [len(group) for group in option_groups(53)] == [27, 26]
    groups = option_groups(255)
    assert [len(group) for group in groups] == [51] * 5
    assert [index for group in groups for index in group] == list(range(255))
    assert len(option_groups(52 * 52)) == 52
    with pytest.raises(ValueError, match="at least one option"):
        option_groups(0)
    with pytest.raises(ValueError, match="at most 2704 options"):
        option_groups(52 * 52 + 1)


def test_compose_groups_anchors_each_group_on_its_winner():
    merged = compose_groups([[0.75, 0.25], [0.1, 0.9]], [0.2, 0.8])
    raw = [0.2, 0.2 / 3, 0.8 / 9, 0.8]
    assert merged == pytest.approx([value / sum(raw) for value in raw])
    assert sum(merged) == pytest.approx(1.0)


def test_confidence_and_noul_calibration_follow_the_release_formulas():
    assert choice_confidence([1.0]) == 1.0
    assert choice_confidence([0.25] * 4) == 0.0
    assert choice_confidence([0.7, 0.2, 0.1]) == pytest.approx((0.7 - 1 / 3) / (2 / 3))
    assert score_confidence([1.0]) == 1.0
    assert score_confidence([0.0, 1.0, 0.0]) == 1.0
    assert score_confidence([1 / 3] * 3) == pytest.approx(0.0)
    assert noul_probability(0.5) == pytest.approx(0.5)
    expected = 1 / (1 + math.exp(-math.log(0.9 / 0.1) / NOUL_TEMPERATURE))
    assert noul_probability(0.9) == pytest.approx(expected)
    # Golden values from the release helper's formula with its MLX settings
    # (READOUT_NOUL_T=1.829074, READOUT_NOUL_BIAS=0): logit / T, not logit * T.
    assert noul_probability(0.9) == pytest.approx(0.768752, abs=1e-6)
    assert noul_probability(0.2) == pytest.approx(0.319098, abs=1e-6)
    # A saturated readout is clipped before the logit, so it stays finite.
    assert noul_probability(1.0) == pytest.approx(noul_probability(1 - 1e-4))
    assert noul_probability(0.0) == pytest.approx(noul_probability(1e-4))


def test_answers_use_the_release_shape_and_rounding():
    choice = Question(type="choice", instructions="q", criteria={"x": None, "y": None})
    options = [("x", ""), ("y", "")]
    assert openjev_answer(choice, options, [0.123456, 0.876544]) == {
        "type": "choice",
        "choice": "y",
        "probabilities": {"x": 0.1235, "y": 0.8765},
        "confidence": round(choice_confidence([0.123456, 0.876544]), 4),
    }
    score = Question(type="score", instructions="q", criteria=["low", "mid", "high"])
    answer = openjev_answer(
        score, [("0", "low"), ("1", "mid"), ("2", "high")], [0.1, 0.2, 0.7]
    )
    assert answer == {
        "type": "score",
        "score": 1.6,
        "legend": {"0": "low", "1": "mid", "2": "high"},
        "probabilities": {"0": 0.1, "1": 0.2, "2": 0.7},
        "confidence": round(score_confidence([0.1, 0.2, 0.7]), 4),
    }
    noul = Question(type="noul", instructions="q")
    assert openjev_answer(noul, [("yes", ""), ("no", "")], [0.9, 0.1]) == {
        "type": "noul",
        "noul": round(noul_probability(0.9), 4),
    }


def test_scorer_requires_one_distinct_token_per_letter():
    class Colliding(FakeTokenizer):
        def encode(self, text, add_special_tokens=False):
            return [7]

    class Splitting(FakeTokenizer):
        def encode(self, text, add_special_tokens=False):
            return super().encode(text) + ([9] if text == "q" else [])

    for tokenizer in (Colliding(), Splitting()):
        with pytest.raises(ValueError, match="its own single token"):
            OpenJevScorer(FakeModel(), tokenizer)


def test_score_reads_the_letter_logits_with_the_release_temperature():
    model = FakeModel(lambda seen: [2.0, 0.5, 1.0] + [9.0] * 49)
    question = Question(
        type="choice", instructions="Pick", criteria={"x": None, "y": "why", "z": None}
    )
    scored, tokens = _scorer(model).score({"k": 1}, {"q": question})
    options, probabilities = scored["q"]
    assert options == [("x", ""), ("y", "why"), ("z", "")]
    # Only the three offered letters compete; the other letters are ignored.
    assert probabilities == pytest.approx(_softmax([2.0, 0.5, 1.0]))
    prompt = "<user>" + render_prompt('{"k": 1}', "Pick", options) + "<assistant>"
    assert tokens == len(prompt) == sum(model.calls)


def test_shared_prefix_is_read_once_and_matches_uncached_readouts():
    questions = {
        "intent": Question(
            type="choice", instructions="Intent?", criteria={"a": None, "b": None}
        ),
        "angry": Question(type="noul", instructions="Angry?"),
        "level": Question(type="score", instructions="Level?", criteria=["lo", "hi"]),
    }
    state = "the customer wrote a long message " * 4
    shared_model = FakeModel()
    shared, tokens = _scorer(shared_model).score(state, questions)

    plain_model = FakeModel()
    plain = _scorer(plain_model)
    plain._shared_prefix = lambda readouts: None
    assert plain.score(state, questions) == (shared, tokens)

    prefix = len("<user>State:\n" + state + "\n\nQuestion: ")
    assert shared_model.calls[0] == prefix
    assert sum(plain_model.calls) == tokens
    assert sum(shared_model.calls) == tokens - 2 * prefix


def test_single_readout_and_diverging_prompts_skip_the_shared_prefix():
    scorer = _scorer()
    assert scorer._shared_prefix([[1, 2, 3]]) is None
    assert scorer._shared_prefix([[1, 2, 3], [4, 2, 3]]) is None
    # One prompt is a prefix of the other: its last token is left to read.
    prefix, _ = scorer._shared_prefix([[1, 2, 3], [1, 2, 3, 4]])
    assert prefix == [1, 2]


def test_long_prompts_are_read_in_chunks(monkeypatch):
    monkeypatch.setattr(openjev_module, "_PREFILL_CHUNK", 16)
    model = FakeModel()
    question = Question(type="noul", instructions="Angry?")
    chunked, tokens = _scorer(model).score("x" * 40, {"q": question})
    assert model.calls == [16] * (tokens // 16) + [tokens % 16]
    monkeypatch.setattr(openjev_module, "_PREFILL_CHUNK", 4096)
    assert _scorer().score("x" * 40, {"q": question}) == (chunked, tokens)


def test_more_options_than_letters_are_scored_in_groups():
    def logits(seen):
        # Each prompt lists different options, so score by letter position.
        return [float(letter % 5) for letter in range(len(LETTERS))]

    criteria = {f"k{index}": None for index in range(60)}
    question = Question(type="choice", instructions="Pick", criteria=criteria)
    model = FakeModel(logits)
    scored, tokens = _scorer(model).score("s", {"q": question})
    options, probabilities = scored["q"]
    assert len(probabilities) == 60
    assert sum(probabilities) == pytest.approx(1.0)
    assert all(value > 0 for value in probabilities)
    part = _softmax([float(letter % 5) for letter in range(30)])
    final = _softmax([0.0, 1.0])
    assert probabilities == pytest.approx(compose_groups([part, part], final))
    # Two group readouts and one readout over the two group winners.
    winners = [options[4], options[34]]
    lengths = [
        len("<user>" + render_prompt("s", "Pick", shown) + "<assistant>")
        for shown in (options[:30], options[30:], winners)
    ]
    assert tokens == sum(lengths)


def test_score_rejects_oversized_prompts_and_non_finite_scores():
    question = Question(type="noul", instructions="Angry?")
    with pytest.raises(ValueError, match="tokens; the limit is 64"):
        _scorer(limit=64).score("x" * 200, {"q": question})
    broken = FakeModel(lambda seen: [float("nan")] * len(LETTERS))
    with pytest.raises(RuntimeError, match="not finite"):
        _scorer(broken).score("s", {"q": question})


def test_scorer_built_on_one_thread_scores_on_another():
    scorer = _scorer()
    question = Question(type="noul", instructions="Angry?")
    expected = scorer.score("s", {"q": question})
    result = []
    worker = threading.Thread(
        target=lambda: result.append(scorer.score("s", {"q": question}))
    )
    worker.start()
    worker.join(timeout=60)
    assert result == [expected]


def test_load_openjev_materializes_weights_and_bounds_the_prompt(tmp_path, monkeypatch):
    import mlx_lm

    evaluated = []
    model = SimpleNamespace(parameters=lambda: {"w": mx.ones((2,))})
    monkeypatch.setattr(mlx_lm, "load", lambda path: (model, "tok", "extra"))
    monkeypatch.setattr(mx, "eval", lambda *args: evaluated.append(args))

    assert load_openjev(str(tmp_path)) == (model, "tok", MAX_PROMPT_TOKENS)
    assert len(evaluated) == 1
    for config, limit in (
        ({"text_config": {"max_position_embeddings": 4096}}, 4096),
        ({"max_position_embeddings": 262144}, MAX_PROMPT_TOKENS),
        ({"max_position_embeddings": True}, MAX_PROMPT_TOKENS),
    ):
        (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
        assert load_openjev(str(tmp_path))[2] == limit


def _backend(scored, tokens=42):
    backend = object.__new__(OpenJevBackend)
    backend.default_model = "openjev"
    backend.repo_id = "openjev/openjev-MLX"
    backend._lock = threading.Lock()
    backend._scorer = SimpleNamespace(score=lambda state, questions: (scored, tokens))
    return backend


def test_backend_answers_in_the_openjev_wire_shape():
    scored = {
        "intent": ([("refund", ""), ("other", "")], [0.9, 0.1]),
        "angry": ([("yes", ""), ("no", "")], [0.8, 0.2]),
    }
    client = TestClient(create_app(_backend(scored)))
    body = {
        "state": "charged twice",
        "questions": {
            "intent": {
                "type": "choice",
                "instructions": "Intent?",
                "criteria": {"refund": None, "other": None},
            },
            "angry": {"type": "noul", "instructions": "Angry?"},
        },
    }
    response = client.post("/v1/systemone", json=body)
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["model"] == "openjev"
    assert payload["answers"]["intent"] == {
        "type": "choice",
        "choice": "refund",
        "probabilities": {"refund": 0.9, "other": 0.1},
        "confidence": 0.8,
    }
    assert payload["answers"]["angry"] == {
        "type": "noul",
        "noul": round(noul_probability(0.8), 4),
    }
    assert payload["usage"]["input_tokens"] == 42
    assert payload["usage"]["billing_units"] == 2

    by_repo = client.post("/v1/systemone", json=body | {"model": "openjev/openjev-MLX"})
    assert by_repo.status_code == 200
    assert (
        client.post("/v1/systemone", json=body | {"temperature": 0.5}).status_code
        == 422
    )
    assert client.post("/v1/systemone", json=body | {"model": "x"}).status_code == 404
    # The published MLX conversion has no vision tower.
    with_image = body | {"images": ["data:image/png;base64,AAAA"]}
    assert client.post("/v1/systemone", json=with_image).status_code == 422


def test_backend_ranks_from_unrounded_probabilities_and_lists_models():
    scored = {"rank": ([("0", "a"), ("1", "b"), ("2", "c")], [0.2, 0.50004, 0.29996])}
    backend = _backend(scored)
    assert backend.rank("ctx", None, ["a", "b", "c"], "openjev", 1.0) == [
        {"rank": 1, "candidate": "b", "prob": 0.50004},
        {"rank": 2, "candidate": "c", "prob": 0.29996},
        {"rank": 3, "candidate": "a", "prob": 0.2},
    ]
    with pytest.raises(ValueError, match="temperature=1"):
        backend.rank("ctx", "q", ["a"], "openjev", 2.0)
    assert backend.models() == [
        {
            "name": "openjev",
            "backend": "openjev-mlx",
            "hf_id": "openjev/openjev-MLX",
            "license": "CC-BY-NC-4.0",
            "description": "OpenJev typed decisions on native MLX (text only)",
        }
    ]


def test_backend_resolves_the_pinned_alias_and_local_directories(tmp_path, monkeypatch):
    import rapid_mlx._mirror as mirror

    loaded = []
    monkeypatch.setattr(
        mirror,
        "pinned_snapshot_download",
        lambda repo, revision: f"/snap/{repo}@{revision}",
    )
    monkeypatch.setattr(
        openjev_module,
        "load_openjev",
        lambda path: loaded.append(path) or ("model", "tok", 128),
    )
    monkeypatch.setattr(
        openjev_module, "OpenJevScorer", lambda *args: SimpleNamespace(args=args)
    )
    devices = []
    monkeypatch.setattr(mx, "set_default_device", devices.append)

    for name in ("openjev", "openjev/openjev-MLX", "OpenJev/OpenJev-mlx"):
        backend = OpenJevBackend(name)
        assert backend.default_model == "openjev"
        assert backend.repo_id == "openjev/openjev-MLX"
        assert backend._scorer.args == ("model", "tok", 128)
    assert loaded[0] == (
        "/snap/openjev/openjev-MLX@a9dcc20aa827a6c7eae478f6ebb3b255bb135451"
    )
    # A same-named directory in the working directory does not shadow it.
    (tmp_path / "openjev").mkdir()
    monkeypatch.chdir(tmp_path)
    assert OpenJevBackend("openjev").repo_id == "openjev/openjev-MLX"
    assert OpenJevBackend("./openjev").default_model == "openjev"
    local = OpenJevBackend(str(tmp_path), device="cpu")
    assert local.default_model == tmp_path.name
    assert loaded[-1] == str(tmp_path)
    # The server's directory layout stays out of /v1/models.
    assert local.models()[0]["hf_id"] == tmp_path.name
    assert devices == [mx.gpu] * 5 + [mx.cpu]

    with pytest.raises(ValueError, match="unknown OpenJev model"):
        OpenJevBackend("someone/openjev")
    with pytest.raises(ValueError, match="'gpu' or 'cpu'"):
        OpenJevBackend("openjev", device="npu")


def test_cli_selects_the_backend_and_states_the_weight_license(monkeypatch, capsys):
    import rapid_mlx._uvicorn as uvicorn_module
    import rapid_mlx.cli as cli_module
    import rapid_mlx.system_one.backends as backend_module

    for model in ("openjev", "openjev/openjev-MLX"):
        assert _resolve_system_one_backend(model, "auto") == "openjev"
    assert _resolve_system_one_backend("org/openjev-like", "auto") == "laya"

    observed = {}

    class FakeBackend:
        LICENSE_NOTICE = OpenJevBackend.LICENSE_NOTICE
        default_model = "openjev-4bit"

        def __init__(self, model, *, device):
            observed.update(model=model, device=device)

    monkeypatch.setattr(backend_module, "OpenJevBackend", FakeBackend)
    monkeypatch.setattr(cli_module, "_port_preflight_or_die", lambda *a, **k: None)
    monkeypatch.setattr(
        uvicorn_module,
        "run_uvicorn",
        lambda app, **kwargs: observed.update(kwargs=kwargs),
    )
    args = build_parser().parse_args(
        ["system-one", "/models/openjev-4bit", "--backend", "openjev", "--port", "8703"]
    )
    cli_module.system_one_command(args)
    assert observed["model"] == "/models/openjev-4bit"
    assert observed["device"] == "gpu"
    assert observed["kwargs"]["port"] == 8703
    output = capsys.readouterr().out
    assert "CC BY-NC 4.0 (non-commercial use only)" in output
