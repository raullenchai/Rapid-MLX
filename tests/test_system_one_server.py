# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import json
import math
import sys
import threading
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from rapid_mlx.system_one.schema import (
    Question,
    RankRequest,
    SystemOneRequest,
    answer_from_probabilities,
    clm_pairs,
    to_text,
)
from rapid_mlx.system_one.server import create_app


class FakeBackend:
    default_model = "fake-decision"

    def models(self):
        return [{"name": self.default_model, "backend": "fake"}]

    def answer(self, state, questions, model, temperature):
        if model != self.default_model:
            raise KeyError("unknown model")
        return {
            "model": model,
            "answers": {
                key: {"type": value.type, "noul": 0.75}
                for key, value in questions.items()
            },
            "usage": {
                "billing_units": len(questions),
                "input_tokens": 3,
                "output_tokens": 0,
            },
        }

    def rank(self, context, question, answers, model, temperature):
        return [
            {"rank": index + 1, "candidate": answer, "prob": 1.0 / len(answers)}
            for index, answer in enumerate(answers)
        ]


def test_system_one_contract_and_auth():
    client = TestClient(create_app(FakeBackend(), api_key="secret"))
    assert client.get("/health").json() == {"ok": True}
    unauthorized = client.post(
        "/v1/systemone",
        json={
            "state": "ready",
            "questions": {"go": {"type": "noul", "instructions": "Go?"}},
        },
    )
    assert unauthorized.status_code == 401

    response = client.post(
        "/v1/systemone",
        headers={"Authorization": "Bearer secret"},
        json={
            "state": {"status": "ready"},
            "questions": {"go": {"type": "noul", "instructions": "Go?"}},
        },
    )
    assert response.status_code == 200
    assert response.json()["answers"]["go"] == {"type": "noul", "noul": 0.75}
    assert response.json()["usage"]["billing_units"] == 1
    assert float(response.headers["X-Rapid-MLX-Latency-Ms"]) >= 0

    lowercase_scheme = client.get(
        "/v1/models", headers={"Authorization": "bearer secret"}
    )
    assert lowercase_scheme.status_code == 200
    non_ascii_token = client.get(
        "/v1/models", headers=[(b"authorization", b"Bearer \xff")]
    )
    assert non_ascii_token.status_code == 401

    with pytest.raises(ValueError, match="ASCII characters only"):
        create_app(FakeBackend(), api_key="密钥")


def test_system_one_rejects_ambiguous_or_unbounded_questions():
    client = TestClient(create_app(FakeBackend()))
    empty = client.post("/v1/systemone", json={"state": "x", "questions": {}})
    assert empty.status_code == 422
    extra = client.post(
        "/v1/systemone",
        json={
            "state": "x",
            "questions": {"q": {"type": "noul", "instructions": "x", "surprise": True}},
        },
    )
    assert extra.status_code == 422
    too_many = client.post(
        "/v1/systemone",
        json={
            "state": "x",
            "questions": {
                "q": {
                    "type": "choice",
                    "instructions": "x",
                    "criteria": {str(index): str(index) for index in range(256)},
                }
            },
        },
    )
    assert too_many.status_code == 422
    aggregate = client.post(
        "/v1/systemone",
        json={
            "state": "x",
            "questions": {
                "first": {
                    "type": "choice",
                    "instructions": "x",
                    "criteria": {str(index): str(index) for index in range(128)},
                },
                "second": {
                    "type": "choice",
                    "instructions": "x",
                    "criteria": {str(index): str(index) for index in range(128)},
                },
            },
        },
    )
    assert aggregate.status_code == 422
    tiny_temperature = client.post(
        "/v1/systemone",
        json={
            "state": "x",
            "questions": {"q": {"type": "noul", "instructions": "x"}},
            "temperature": 5e-324,
        },
    )
    assert tiny_temperature.status_code == 422


def test_rank_contract():
    client = TestClient(create_app(FakeBackend()))
    response = client.post(
        "/v1/rank",
        json={"context": "question", "answers": ["a", "b"]},
    )
    assert response.status_code == 200
    assert [item["candidate"] for item in response.json()["ranked"]] == ["a", "b"]
    assert client.post("/v1/rank", json={"answers": []}).status_code == 422
    assert client.post("/v1/rank", json={"answers": ["ok", ""]}).status_code == 422
    assert (
        client.post(
            "/v1/rank", json={"answers": ["ok"], "temperature": 5e-324}
        ).status_code
        == 422
    )


def test_server_maps_backend_errors_and_rejects_invalid_capacity(monkeypatch):
    with pytest.raises(ValueError, match="positive"):
        create_app(FakeBackend(), max_concurrent_requests=0)

    client = TestClient(create_app(FakeBackend()))
    unknown = client.post(
        "/v1/systemone",
        json={
            "model": "missing",
            "state": "x",
            "questions": {"q": {"type": "noul", "instructions": "x"}},
        },
    )
    assert unknown.status_code == 404

    class FailingBackend(FakeBackend):
        def answer(self, *args, **kwargs):
            raise ValueError("bad answer")

        def rank(self, *args, **kwargs):
            if kwargs:
                raise AssertionError("backend calls are positional")
            raise KeyError("bad rank model")

    failing = TestClient(create_app(FailingBackend()))
    answer = failing.post(
        "/v1/systemone",
        json={
            "state": "x",
            "questions": {"q": {"type": "noul", "instructions": "x"}},
        },
    )
    assert answer.status_code == 422
    assert answer.json()["detail"] == "invalid request: bad answer"
    rank = failing.post("/v1/rank", json={"answers": ["a"]})
    assert rank.status_code == 404
    assert rank.json()["detail"] == "bad rank model"

    class InvalidRankBackend(FakeBackend):
        def rank(self, *args, **kwargs):
            raise TypeError("bad rank")

    rank = TestClient(create_app(InvalidRankBackend())).post(
        "/v1/rank", json={"answers": ["a"]}
    )
    assert rank.status_code == 422
    assert rank.json()["detail"] == "invalid request: bad rank"

    import rapid_mlx.system_one.server as server_module

    app = create_app(FakeBackend(), max_concurrent_requests=1)
    real_create_task = asyncio.create_task

    def fail_create_task(coroutine):
        coroutine.close()
        raise RuntimeError("task creation failed")

    with monkeypatch.context() as patch:
        patch.setattr(server_module.asyncio, "create_task", fail_create_task)
        with pytest.raises(RuntimeError, match="task creation failed"):
            TestClient(app).post(
                "/v1/systemone",
                json={
                    "state": "x",
                    "questions": {"q": {"type": "noul", "instructions": "x"}},
                },
            )
    assert server_module.asyncio.create_task is real_create_task


def test_clm_rendering_matches_reference_layout():
    question = Question(
        type="choice",
        instructions="Which team?",
        criteria={"billing": "Invoices", "support": "Bugs"},
    )
    pairs = clm_pairs({"customer": "charged twice"}, {"route": question})
    assert pairs["route"] == (
        "customer: charged twice\n\nWhich team?",
        ["billing", "support"],
        ["Invoices", "Bugs"],
    )


def test_schema_validates_every_question_shape_and_finite_temperature():
    invalid_questions = [
        {"type": "choice", "instructions": "x", "criteria": None},
        {"type": "score", "instructions": "x", "criteria": ["low"]},
        {
            "type": "score",
            "instructions": "x",
            "criteria": list(range(256)),
        },
        {"type": "noul", "instructions": "x", "criteria": []},
    ]
    for value in invalid_questions:
        with pytest.raises(ValidationError):
            Question.model_validate(value)

    for request_type in (SystemOneRequest, RankRequest):
        payload = (
            {
                "state": "x",
                "questions": {"q": {"type": "noul", "instructions": "x"}},
            }
            if request_type is SystemOneRequest
            else {"answers": ["a"]}
        )
        payload["temperature"] = float("nan")
        with pytest.raises(ValidationError, match="finite"):
            request_type.model_validate(payload)


def test_schema_renders_structured_values_and_all_answer_types():
    assert to_text(None) == ""
    assert to_text(True) == "true"
    assert to_text(False) == "false"
    assert to_text(3.5) == "3.5"
    assert to_text({"outer": {"inner": 1}, "empty": []}) == (
        "outer:\n  inner: 1\n\nempty: "
    )
    assert to_text([{"key": "value"}, []]) == "-\n  key: value\n- "
    with pytest.raises(TypeError):
        to_text(object())

    questions = {
        "choice": Question(
            type="choice", instructions="", criteria={"named": None, "other": ""}
        ),
        "score": Question(type="score", instructions="Rate", criteria=["low", "high"]),
        "noul": Question(
            type="noul", instructions="Proceed", criteria={"true": "Do it"}
        ),
    }
    pairs = clm_pairs("", questions)
    assert pairs["choice"][2] == ["named", "other"]
    assert pairs["score"] == ("Rate", ["0", "1"], ["low", "high"])
    assert pairs["noul"][2] == [
        "false: No. This is false: Proceed",
        "true: Do it",
    ]

    assert answer_from_probabilities(
        questions["noul"], ["false", "true"], [0.25, 0.75]
    ) == {"type": "noul", "noul": 0.75}
    choice = answer_from_probabilities(
        questions["choice"], ["named", "other"], [0.9, 0.1]
    )
    assert choice["choice"] == "named"
    assert choice["confidence"] == pytest.approx(0.8)
    score = answer_from_probabilities(questions["score"], ["0", "1"], [0.2, 0.8])
    assert score["score"] == pytest.approx(0.8)
    assert score["legend"] == {"0": "low", "1": "high"}


def test_laya_backend_adapter_and_errors(monkeypatch):
    from rapid_mlx.system_one.backends import LayaBackend

    monkeypatch.setitem(sys.modules, "laya_mlx", None)
    with pytest.raises(RuntimeError, match="system-one"):
        LayaBackend("missing")

    calls = []

    class Agent:
        def system_one(self, state, questions):
            calls.append((state, questions))
            return {
                "answers": {
                    "rank": {
                        "probabilities": {"0": 0.2, "1": 0.8},
                    }
                }
            }

    def load(model, **kwargs):
        calls.append((model, kwargs))
        return Agent()

    monkeypatch.setitem(sys.modules, "laya_mlx", SimpleNamespace(load=load))
    backend = LayaBackend("org/laya", device="gpu", dtype="float16", batch_size=4)
    question = Question(type="noul", instructions="Go?")
    result = backend.answer("state", {"q": question}, "laya-rl-agent", 1.0)
    assert result["model"] == "org/laya"
    assert result["usage"]["billing_units"] == 1
    assert calls[-1][1]["q"] == {"type": "noul", "instructions": "Go?"}
    with pytest.raises(KeyError, match="unknown model"):
        backend.answer("state", {"q": question}, "other", 1.0)
    with pytest.raises(ValueError, match="temperature=1"):
        backend.answer("state", {"q": question}, "org/laya", 0.5)
    assert backend.rank("ctx", None, ["a", "b"], "org/laya", 1.0) == [
        {"rank": 1, "candidate": "b", "prob": 0.8},
        {"rank": 2, "candidate": "a", "prob": 0.2},
    ]
    assert backend.models()[0]["backend"] == "laya-mlx"


@pytest.mark.requires_mlx
def test_clm_backend_runs_native_hidden_state_and_reuses_action_cache(
    monkeypatch, tmp_path
):
    from types import SimpleNamespace

    import mlx.core as mx

    import rapid_mlx.utils.tokenizer as tokenizer_module
    from rapid_mlx.system_one.backends import CLMBackend

    config = {
        "width": 4,
        "depth": 2,
        "projection_dim": 2,
        "hidden_size": 4,
        "activation": "gelu",
        "logit_scale": math.log(1000.0),
        "model_name": "clm-test",
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    weights = {}
    for prefix in ("state_head", "action_head"):
        weights[f"{prefix}.inp.weight"] = mx.eye(4)
        weights[f"{prefix}.inp.bias"] = mx.zeros((4,))
        weights[f"{prefix}.out.weight"] = mx.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        weights[f"{prefix}.out.bias"] = mx.zeros((2,))
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)

    calls = []

    class Inner:
        def __call__(self, ids):
            calls.append(ids.tolist())
            value = ids.astype(mx.float32)
            return mx.stack(
                [
                    value,
                    mx.ones_like(value),
                    mx.zeros_like(value),
                    mx.zeros_like(value),
                ],
                axis=-1,
            )

    class Tokenizer:
        eos_token_id = 0

        def encode(self, text, add_special_tokens=True):
            return [{"ctx\n\nPick": 1, "good": 2, "bad": 3}.get(text, 4)]

    model = SimpleNamespace(
        model=Inner(), args=SimpleNamespace(hidden_size=4, model_type="qwen3")
    )
    monkeypatch.setattr(
        tokenizer_module,
        "load_model_with_fallback",
        lambda _: (model, Tokenizer()),
    )
    selected_devices = []
    monkeypatch.setattr(mx, "set_default_device", selected_devices.append)
    backend = CLMBackend(
        "dummy-qwen3",
        str(tmp_path),
        model_name="public-clm",
        device="cpu",
        cache_entries=16,
    )
    assert selected_devices == [mx.cpu]
    assert backend._scale == pytest.approx(100.0)
    backend._tokenizer = SimpleNamespace(
        encode=lambda text, add_special_tokens=True: list(range(10)), eos_token_id=0
    )
    backend._max_tokens = 3
    assert backend._token_ids("short") == [7, 8, 9]
    backend._tokenizer = Tokenizer()
    backend._max_tokens = 2048
    question = Question(
        type="choice",
        instructions="Pick",
        criteria={"a": "good", "b": "bad"},
    )
    first = backend.answer("ctx", {"q": question}, "public-clm", 1.0)
    assert first["model"] == "public-clm"
    assert first["answers"]["q"]["choice"] == "a"
    assert first["usage"]["input_tokens"] == 3
    assert first["usage"]["requested_tokens"] == 3
    assert len(calls) == 3

    second = backend.answer("ctx", {"q": question}, "public-clm", 1.0)
    assert second["usage"]["input_tokens"] == 0
    assert second["usage"]["requested_tokens"] == 3
    assert len(calls) == 3

    backend._project("action", ["good"], [[9]])
    assert len(calls) == 4

    backend._max_work_tokens = 2
    try:
        backend.answer("ctx", {"q": question}, "public-clm", 1.0)
    except ValueError as exc:
        assert "encoder tokens" in str(exc)
    else:
        raise AssertionError("CLM accepted a request over its work-token budget")

    backend._max_work_tokens = 32_768
    backend._max_text_bytes = 4
    calls_before = len(calls)
    with pytest.raises(ValueError, match="state exceeds 4 UTF-8 bytes"):
        backend.answer("oversized", {"q": question}, "public-clm", 1.0)
    assert len(calls) == calls_before

    import mlx.nn as nn

    quantized = nn.QuantizedLinear.from_linear(nn.Linear(32, 4), 32, 4)

    class QuantizedInner(Inner):
        def named_modules(self):
            return [("quantized", quantized)]

    quantized_model = SimpleNamespace(
        model=QuantizedInner(), args=SimpleNamespace(hidden_size=4, model_type="qwen3")
    )
    monkeypatch.setattr(
        tokenizer_module,
        "load_model_with_fallback",
        lambda _: (quantized_model, Tokenizer()),
    )
    with pytest.raises(ValueError, match="quantized encoders"):
        CLMBackend("quantized-qwen3", str(tmp_path))


def test_clm_backend_rejects_non_safetensors_weight_file(tmp_path):
    from rapid_mlx.system_one.backends import CLMBackend

    with pytest.raises(ValueError, match="device"):
        CLMBackend("unused", str(tmp_path), device="ane")

    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    weights = tmp_path / "model.npz"
    weights.write_bytes(b"not used")
    with pytest.raises(ValueError, match="must be a .safetensors file"):
        CLMBackend("unused", str(weights))


@pytest.mark.requires_mlx
def test_clm_projection_rejects_depth_below_two():
    from rapid_mlx.system_one.backends import _ProjectionHead

    with pytest.raises(ValueError, match="depth must be at least 2"):
        _ProjectionHead({"hidden_size": 1, "width": 1, "depth": 1})


@pytest.mark.requires_mlx
def test_clm_projection_hidden_layers_and_validation():
    import mlx.core as mx

    from rapid_mlx.system_one.backends import _ProjectionHead

    with pytest.raises(ValueError, match="unsupported"):
        _ProjectionHead({"hidden_size": 2, "width": 2, "depth": 2, "activation": "bad"})
    for activation in ("gelu", "relu", "silu"):
        head = _ProjectionHead(
            {
                "hidden_size": 2,
                "width": 2,
                "depth": 3,
                "projection_dim": 2,
                "activation": activation,
                "layernorm": True,
                "residual": True,
            }
        )
        value = head(mx.ones((1, 2)))
        mx.eval(value)
        assert value.shape == (1, 2)
        with pytest.raises(ValueError, match="expected 8"):
            head.load_weights({}, "state_head")


@pytest.mark.requires_mlx
def test_clm_backend_validates_artifact_encoder_tokenizer_and_public_methods(
    monkeypatch, tmp_path
):
    import mlx.core as mx

    import rapid_mlx.utils.tokenizer as tokenizer_module
    from rapid_mlx.system_one.backends import CLMBackend

    with pytest.raises(ValueError, match="must be converted"):
        CLMBackend("encoder", str(tmp_path / "head.pt"))
    with pytest.raises(ValueError, match="must contain"):
        CLMBackend("encoder", str(tmp_path / "missing"))

    config = {
        "width": 2,
        "depth": 2,
        "projection_dim": 2,
        "hidden_size": 2,
        "logit_scale": 0.0,
    }
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "config.json").write_text(json.dumps(config), encoding="utf-8")
    weights = {}
    for prefix in ("state_head", "action_head"):
        weights[f"{prefix}.inp.weight"] = mx.eye(2)
        weights[f"{prefix}.inp.bias"] = mx.zeros((2,))
        weights[f"{prefix}.out.weight"] = mx.eye(2)
        weights[f"{prefix}.out.bias"] = mx.zeros((2,))
    mx.save_safetensors(str(artifact / "model.safetensors"), weights)

    class Inner:
        def __call__(self, ids):
            values = ids.astype(mx.float32)
            return mx.stack([values, mx.ones_like(values)], axis=-1)

    class Tokenizer:
        eos_token_id = 7

        def encode(self, text, add_special_tokens=True):
            return [] if text == "empty" else [len(text) or 1]

    def set_model(*, inner=Inner(), model_type="qwen3", hidden_size=2, **args):
        model = SimpleNamespace(
            model=inner,
            args=SimpleNamespace(
                hidden_size=hidden_size, model_type=model_type, **args
            ),
        )
        monkeypatch.setattr(
            tokenizer_module,
            "load_model_with_fallback",
            lambda _: (model, Tokenizer()),
        )

    set_model(inner=None)
    with pytest.raises(ValueError, match="pre-lm-head"):
        CLMBackend("encoder", str(artifact))
    set_model(model_type="other")
    with pytest.raises(ValueError, match="Qwen3"):
        CLMBackend("encoder", str(artifact))
    set_model(hidden_size=3)
    with pytest.raises(ValueError, match="hidden size"):
        CLMBackend("encoder", str(artifact))
    set_model(quantization={"bits": 4})
    with pytest.raises(ValueError, match="quantized"):
        CLMBackend("encoder", str(artifact))

    set_model()
    backend = CLMBackend("encoder", str(artifact), cache_entries=1)
    assert backend._token_ids("empty") == [7]
    backend._tokenizer = SimpleNamespace(
        encode=lambda *args, **kwargs: [], eos_token_id=None
    )
    with pytest.raises(ValueError, match="no eos token"):
        backend._token_ids("empty")
    backend._max_text_bytes = 3
    with pytest.raises(ValueError, match="UTF-8 bytes"):
        backend._token_ids("éé")

    backend._tokenizer = Tokenizer()
    backend._max_text_bytes = 1024
    backend._project("action", ["a"], [[1]])
    backend._project("action", ["bb"], [[2]])
    assert len(backend._cache) == 1
    with pytest.raises(KeyError, match="unknown model"):
        backend.answer(
            "x", {"q": Question(type="noul", instructions="x")}, "wrong", 1.0
        )
    ranked = backend.rank("ctx", None, ["a", "bb"], backend.default_model, 1.0)
    assert {item["candidate"] for item in ranked} == {"a", "bb"}
    assert backend.models() == [
        {
            "name": "clm-latest",
            "backend": "clm-mlx",
            "encoder": "encoder",
            "description": "CLM contrastive state/action model on native MLX",
        }
    ]

    bad_scale = dict(config, logit_scale=float("nan"))
    (artifact / "config.json").write_text(json.dumps(bad_scale), encoding="utf-8")
    set_model()
    with pytest.raises(ValueError, match="must be finite"):
        CLMBackend("encoder", str(artifact))


async def test_system_one_rejects_slow_request_body():
    from rapid_mlx.config import get_config

    app = create_app(FakeBackend())
    body = json.dumps(
        {
            "state": "ready",
            "questions": {"go": {"type": "noul", "instructions": "Go?"}},
        }
    ).encode()
    split = len(body) // 2
    calls = 0

    async def receive():
        nonlocal calls
        calls += 1
        if calls == 1:
            return {"type": "http.request", "body": body[:split], "more_body": True}
        await asyncio.sleep(0.05)
        return {"type": "http.request", "body": body[split:], "more_body": False}

    messages = []

    async def send(message):
        messages.append(message)

    config = get_config()
    previous = config.body_receive_timeout_seconds
    config.body_receive_timeout_seconds = 0.01
    try:
        await app(
            {
                "type": "http",
                "asgi": {"version": "3.0"},
                "http_version": "1.1",
                "method": "POST",
                "scheme": "http",
                "path": "/v1/systemone",
                "raw_path": b"/v1/systemone",
                "query_string": b"",
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
                "client": ("127.0.0.1", 1),
                "server": ("test", 80),
            },
            receive,
            send,
        )
    finally:
        config.body_receive_timeout_seconds = previous
    start = next(
        message for message in messages if message["type"] == "http.response.start"
    )
    assert start["status"] == 408


async def test_system_one_bounds_executor_admission():
    started = threading.Event()
    release = threading.Event()

    class BlockingBackend(FakeBackend):
        def answer(self, state, questions, model, temperature):
            started.set()
            assert release.wait(timeout=2)
            return super().answer(state, questions, model, temperature)

    app = create_app(BlockingBackend(), max_concurrent_requests=1)
    transport = httpx.ASGITransport(app=app)
    request = {
        "state": "ready",
        "questions": {"go": {"type": "noul", "instructions": "Go?"}},
    }
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        first = asyncio.create_task(client.post("/v1/systemone", json=request))
        assert await asyncio.to_thread(started.wait, 1)
        second = await client.post("/v1/systemone", json=request)
        assert second.status_code == 503
        assert second.headers["Retry-After"] == "1"
        release.set()
        assert (await first).status_code == 200


async def test_cancelled_request_keeps_admission_until_worker_exits():
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    class BlockingBackend(FakeBackend):
        def answer(self, state, questions, model, temperature):
            started.set()
            release.wait(timeout=2)
            finished.set()
            return super().answer(state, questions, model, temperature)

    app = create_app(BlockingBackend(), max_concurrent_requests=1)
    transport = httpx.ASGITransport(app=app)
    request = {
        "state": "ready",
        "questions": {"go": {"type": "noul", "instructions": "Go?"}},
    }
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        first = asyncio.create_task(client.post("/v1/systemone", json=request))
        assert await asyncio.to_thread(started.wait, 1)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first

        while_running = await client.post("/v1/systemone", json=request)
        assert while_running.status_code == 503

        release.set()
        assert await asyncio.to_thread(finished.wait, 1)
        await asyncio.sleep(0)
        after_exit = await client.post("/v1/systemone", json=request)
        assert after_exit.status_code == 200
