# SPDX-License-Identifier: Apache-2.0
import json
import math
import random
import re
import shlex
import sys
import weakref
from types import SimpleNamespace

import psutil
import pytest

from rapid_mlx import adapter_eval, cli, prompt_logprobs
from rapid_mlx.lora_adapter import ADAPTER_METADATA
from rapid_mlx.telemetry import consent_runtime
from rapid_mlx.utils import chat_template


def _jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


@pytest.fixture
def session(tmp_path, monkeypatch):
    model = tmp_path / "local model"
    model.mkdir()
    (model / "weights.safetensors").write_bytes(b"weights")
    (model / "config.json").write_text('{"max_position_embeddings": 420}')
    adapter = tmp_path / "my adapter"
    adapter.mkdir()
    (adapter / "adapters.safetensors").touch()
    (adapter / "adapter_config.json").write_text(json.dumps({"model": str(model)}))
    sidecar = adapter / ADAPTER_METADATA
    sidecar.write_text(json.dumps({"base_model": str(model)}))
    data, train, out = (
        tmp_path / name for name in ("test.jsonl", "train.jsonl", "out")
    )
    rows = [
        (f"Private outline {i}", "Private reference " + "long " * 70) for i in range(4)
    ]
    _jsonl(data, [{"messages": adapter_eval._messages([pair])} for pair in rows])
    _jsonl(
        train,
        [{"prompt": f"Training {i}", "completion": f"Example {i}"} for i in range(4)]
        + [{"prompt": rows[0][0], "completion": "Must not leak into few-shot"}],
    )
    calls, loads, judges = [], [], []
    state = SimpleNamespace(
        weights=None,
        vote="first real choice",
        adapter_loss=0.5,
        bad_tokens=False,
        changed_source=False,
        interrupted=False,
    )

    class Model:
        pass

    def encode(prompt, *, add_special_tokens):
        assert not add_special_tokens
        if state.bad_tokens:
            return []
        # Merge the prompt boundary with the first character of the reference.
        return re.findall(r"~.|.", prompt, flags=re.DOTALL)

    def load(name, *, adapter_path, return_source):
        assert name == str(model) and return_source
        assert state.weights is None or state.weights() is None
        loads.append(adapter_path)
        weights = Model()
        weights.adapter = bool(adapter_path)
        state.weights = weakref.ref(weights)
        source = "changed" if state.changed_source and not adapter_path else str(model)
        return weights, SimpleNamespace(encode=encode, model_max_length=512), source

    def render(tok, messages, *, model_name, enable_thinking):
        assert not enable_thinking and model_name == str(model)
        return json.dumps(messages, ensure_ascii=False) + "~"

    def stream_generate(weights, tok, tokens, *, max_tokens, sampler):
        assert sampler == 0
        if state.interrupted:
            raise KeyboardInterrupt
        messages = json.loads("".join(tokens).removesuffix("~"))
        calls.append((weights.adapter, messages, max_tokens))
        assert max_tokens == 512
        text = (
            "Adapter draft."
            if weights.adapter
            else f"Base draft, {len(messages)} turns."
        )
        yield SimpleNamespace(text=text)

    def score(weights, tokens, result):
        assert result.top_k == 0
        prompt = "".join(tokens).split("~")[0]
        messages = json.loads(prompt)
        if tokens[-2] == "~":
            assert not weights.adapter and len(tokens) <= 420
            assert tokens[-1] in {"A", "B"}
            samples = json.loads(messages[0]["content"].splitlines()[1])
            judges.append((samples, tokens[-1]))
            # The judge favors A in both orders, with a larger preference
            # when the adapter occupies that position.
            preferred = state.vote == "first real choice"
            a_score = -0.6 if preferred and samples["A"] == "Adapter draft." else -1.2
            result.logprobs = [a_score if tokens[-1] == "A" else -1.5]
            return
        shots = len(messages) > 1
        loss = state.adapter_loss if weights.adapter else (1.0 if shots else 2.0)
        start = tokens.index("~P")
        result.logprobs = [None] + [0] * (start - 1) + [-loss] * (len(tokens) - start)

    def clear_cache():
        assert state.weights() is None

    core = SimpleNamespace(clear_cache=clear_cache)
    for name, module in {
        "mlx": SimpleNamespace(core=core),
        "mlx.core": core,
        "mlx_lm": SimpleNamespace(stream_generate=stream_generate),
        "mlx_lm.sample_utils": SimpleNamespace(make_sampler=lambda *, temp: temp),
        "rapid_mlx.utils.tokenizer": SimpleNamespace(load_model_with_fallback=load),
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(chat_template, "apply_chat_template", render)
    monkeypatch.setattr(prompt_logprobs, "prefill_prompt_logprobs", score)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=1))

    def unexpected_telemetry(**kwargs):
        pytest.fail("Local eval must bypass telemetry startup")

    monkeypatch.setattr(consent_runtime, "startup", unexpected_telemetry)

    argv = ["eval", str(model), "--adapter", str(adapter), "--data", str(data)]
    return SimpleNamespace(
        argv=argv,
        state=state,
        calls=calls,
        judges=judges,
        loads=loads,
        adapter=adapter,
        rows=rows,
        sidecar=sidecar,
        data=data,
        train=train,
        out=out,
    )


def _run(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["rapid-mlx", *argv])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    return exc.value.code


def test_style_scorecard(session, monkeypatch, capsys):
    s = session
    args = [
        "--few-shot",
        str(s.train),
        "--samples",
        "3",
        "--seed",
        "7",
        "--out",
        str(s.out),
    ]
    assert _run(monkeypatch, s.argv + args) == 0
    assert s.loads == [str(s.adapter), None]
    captured = capsys.readouterr()
    stdout = captured.out
    assert "Warning: low available memory" in captured.err
    assert "Running adapter" in captured.err
    assert stdout.splitlines()[0] == (
        "Scorecard: Perplexity · base 7.389 · few-shot 2.718 · "
        "adapter 1.649 · recommend adapter"
    )
    assert "Elapsed time:" in stdout
    assert "indicative win rate: 100.0%" in stdout
    assert "3/0/0 win/tie/loss" in stdout
    assert (
        shlex.join(["rapid-mlx", "serve", s.argv[1], "--adapter", str(s.adapter)])
        in stdout
    )
    report_text = (s.out / "report.json").read_text()
    assert "Private" not in report_text and "draft" not in report_text
    assert str(s.adapter.parent) not in report_text
    report = json.loads(report_text)
    assert report["recommend_adapter"]
    assert report["indicative_preferences"] == dict.fromkeys(
        ["base", "few-shot"],
        {
            "win": 3,
            "tie": 0,
            "loss": 0,
            "decided": 3,
            "tied_by_margin": 0,
            "invalid": 0,
        },
    )
    assert report["elapsed_seconds"] >= 0
    assert report["notes"] == [adapter_eval.FEW_SHOT_NOTE]
    assert report["arms"]["adapter"]["reference_tokens"] == 3 * len(s.rows[0][1])
    assert report["arms"]["adapter"]["perplexity"] == math.exp(0.5)
    assert [report["arms"][arm]["loss"] for arm in ("base", "few-shot")] == [2.0, 1.0]
    markdown = (s.out / "report.md").read_text()
    assert markdown.splitlines()[0] == stdout.splitlines()[0]
    assert adapter_eval.FEW_SHOT_NOTE in markdown
    assert markdown.count("### Sample") == 3 and "…" in markdown
    assert "LoRA learns how you write" in markdown
    calls = [messages for _, messages, limit in s.calls if limit == 512]
    rng = random.Random(7)
    expected = [outline for outline, _ in rng.sample(s.rows, 3)]
    assert [messages[-1]["content"] for messages in calls] == expected * 3
    shots = rng.sample(range(4), 3)
    assert all(
        [turn["content"] for turn in m[:6:2]] == [f"Training {i}" for i in shots]
        for m in calls[-3:]
    )
    assert [option for _, option in s.judges] == ["A", "B"] * 12
    assert all(a == b for (a, _), (b, _) in zip(s.judges[::2], s.judges[1::2]))
    judges = [samples for samples, _ in s.judges[::2]]
    assert len(judges) == 12
    for left, right in zip(judges[::2], judges[1::2]):
        assert 0 < len(left["Reference"]) < len(s.rows[0][1])
        assert left == {
            "Reference": right["Reference"],
            "A": right["B"],
            "B": right["A"],
        }


@pytest.mark.parametrize(
    ("task", "few_shot", "vote", "adapter_loss", "recommend"),
    [
        ("generic", False, "TIE", 0.5, True),
        ("generic", True, "TIE", 1.5, False),
        ("style", False, "A", 2.0, False),
        ("style", True, "A", 0.5, False),
        ("style", True, "first real choice", 1.5, False),
    ],
)
def test_verdict(
    session, monkeypatch, capsys, task, few_shot, vote, adapter_loss, recommend
):
    s = session
    s.state.vote, s.state.adapter_loss = vote, adapter_loss
    args = ["--task", task, "--samples", "20"]
    if few_shot:
        args.extend(["--few-shot", str(s.train)])
    if task == "generic" and not few_shot:

        def memory_probe_failed():
            raise OSError("memory information unavailable")

        monkeypatch.setattr(psutil, "virtual_memory", memory_probe_failed)
    assert _run(monkeypatch, s.argv + args) == 0
    report = json.loads(
        s.adapter.with_name("my adapter-eval").joinpath("report.json").read_text()
    )
    assert report["samples"] == 4
    assert report["recommend_adapter"] is recommend
    assert bool(report["indicative_preferences"]) is (task == "style")
    captured = capsys.readouterr()
    stdout = captured.out
    assert ("Skipping the three-shot baseline" in captured.err) is not few_shot
    if not few_shot:
        assert "few-shot —" in stdout.splitlines()[0]
        assert not report["notes"]
    assert ("The adapter did not clearly help" in stdout) is not recommend
    assert ("rapid-mlx serve" in stdout) is recommend


@pytest.mark.parametrize(
    ("problem", "message", "code"),
    [
        ("samples", "must be positive", 1),
        ("json", "JSONL line 1", 1),
        ("empty", "at least one JSONL row", 1),
        ("bad_roles", "JSONL line 1", 1),
        ("blank_text", "JSONL line 1", 1),
        ("shots", "three distinct prompts", 1),
        ("sidecar", "adapter base mismatch", 1),
        ("tokens", "scoreable assistant tokens", 1),
        ("loss", "invalid reference log probabilities", 1),
        ("source", "checkpoint changed", 1),
        ("interrupt", "", 130),
    ],
)
def test_errors(session, monkeypatch, capsys, problem, message, code):
    s = session
    args = []
    if problem == "samples":
        args = ["--samples", "0"]
    elif problem == "json":
        s.data.write_text("{")
    elif problem == "empty":
        s.data.write_text("")
    elif problem == "bad_roles":
        _jsonl(s.data, [{"messages": [{"role": "user"}, {"role": "user"}]}])
    elif problem == "blank_text":
        _jsonl(s.data, [{"prompt": "Some text", "completion": " "}])
    elif problem == "shots":
        _jsonl(s.train, [{"prompt": "Same outline", "completion": "Draft"}] * 3)
        args = ["--few-shot", str(s.train)]
    elif problem == "sidecar":
        s.sidecar.write_text('{"base_model": "another/model"}')
    elif problem == "tokens":
        s.state.bad_tokens = True
    elif problem == "loss":
        s.state.adapter_loss = math.inf
    elif problem == "source":
        s.state.changed_source = True
    elif problem == "interrupt":
        s.state.interrupted = True
    assert _run(monkeypatch, s.argv + args + ["--out", str(s.out)]) == code
    assert message in capsys.readouterr().err
    assert not s.out.exists()


def test_style_metrics():
    reference = "And I’ll go — will you?\n- But don't wait."
    profile = adapter_eval._profile(reference)
    assert profile[:10] == pytest.approx(
        (4, 1, 26 / 8, 1, 2 / 8, 2 / 8, 1, 0.5, 1 / len(reference), 2 / len(reference))
    )
    other = adapter_eval._profile("静かな日。")
    scales = adapter_eval._style_scales([profile, other])
    assert all(scale > 0 for scale in scales)
    assert adapter_eval._distance(reference, profile, scales) == 0
    assert adapter_eval._distance("Here!", profile, scales) > 0
    assert adapter_eval._distance(" ", profile, scales) == 1
    assert adapter_eval._distance("静かな日。", other, scales) == 0
    assert scales[3] == 0.001
    assert scales[4] == 0.125


def test_margin_losses_and_invalid_scores():
    scores = iter([-2, -1, -1, -2, -1.375, -1.5, -1.5, -1.5, None, -1, -2, -1])
    counts = adapter_eval._preference(
        lambda ids, option: next(scores),
        lambda messages: [1],
        (2, 3),
        128,
        [("p", "r")] * 4,
        ["Adapter", "", "Adapter", "Adapter"],
        ["Base"] * 4,
    )
    assert counts == {
        "win": 0,
        "tie": 1,
        "loss": 1,
        "decided": 1,
        "tied_by_margin": 1,
        "invalid": 2,
    }


@pytest.mark.parametrize(("options", "token_limit"), [(None, 100), ((2, 3), 1)])
def test_invalid_judge_input(options, token_limit):
    def unexpected_score(*args):
        pytest.fail("Invalid judge input cannot be scored")

    counts = adapter_eval._preference(
        unexpected_score,
        lambda messages: [1],
        options,
        token_limit,
        [("p", "r")],
        ["Adapter"],
        ["Base"],
    )
    assert counts["invalid"] == 1


def test_single_token_options():
    def encode_spaced(text):
        return [ord(text[-1])] if text.startswith(" ") else [1, 2]

    assert adapter_eval._judge_options(encode_spaced) == (ord("A"), ord("B"))
    assert adapter_eval._judge_options(lambda text: [0]) is None
