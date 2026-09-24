# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import json
import math
import threading

import httpx
import pytest
from fastapi.testclient import TestClient

from rapid_mlx.system_one.schema import Question, clm_pairs
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
    backend = CLMBackend(
        "dummy-qwen3", str(tmp_path), model_name="public-clm", cache_entries=16
    )
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


def test_clm_backend_rejects_non_safetensors_weight_file(tmp_path):
    from rapid_mlx.system_one.backends import CLMBackend

    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    weights = tmp_path / "model.npz"
    weights.write_bytes(b"not used")
    with pytest.raises(ValueError, match="must be a .safetensors file"):
        CLMBackend("unused", str(weights))


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
