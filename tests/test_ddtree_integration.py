# SPDX-License-Identifier: Apache-2.0
"""Integration-style tests for the DDTree MVP plumbing."""

from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock


def test_serve_parser_exposes_ddtree_speculative_config() -> None:
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, "-m", "vllm_mlx.cli", "serve", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert out.returncode == 0, out.stderr
    assert "--speculative-config" in out.stdout
    assert "--enable-ddtree" not in out.stdout


def _ddtree_cli_args(**overrides):
    data = {
        "model": "qwen3.5-9b-8bit",
        "_original_alias": None,
        "speculative_config": None,
        "enable_ddtree": False,
        "enable_dflash": False,
        "spec_decode": "none",
        "suffix_decoding": False,
        "no_spec_decode": False,
    }
    data.update(overrides)
    return SimpleNamespace(**data)


def test_speculative_config_ddtree_preflight_uses_config_overrides(
    monkeypatch,
) -> None:
    from vllm_mlx.cli import (
        _normalize_speculative_config_or_exit,
        _preflight_ddtree_or_exit,
    )

    monkeypatch.setattr(
        "vllm_mlx.speculative.ddtree.eligibility.have_runtime",
        lambda: True,
    )
    args = _ddtree_cli_args(
        speculative_config=(
            '{"method":"ddtree","model":"local/draft",'
            '"num_speculative_tokens":8,"tree_budget":12}'
        )
    )

    _normalize_speculative_config_or_exit(args)
    assert args.enable_ddtree is True
    assert args.spec_decode == "none"
    alias, profile = _preflight_ddtree_or_exit(args)

    assert alias == "qwen3.5-9b-8bit"
    assert profile.supports_ddtree is True
    assert args._ddtree_drafter_repo == "local/draft"
    assert args._ddtree_speculative_tokens == 8
    assert args._ddtree_tree_budget == 12


def test_speculative_config_ddtree_preflight_falls_back_to_alias_defaults(
    monkeypatch,
) -> None:
    from vllm_mlx.cli import (
        _normalize_speculative_config_or_exit,
        _preflight_ddtree_or_exit,
    )

    monkeypatch.setattr(
        "vllm_mlx.speculative.ddtree.eligibility.have_runtime",
        lambda: True,
    )
    args = _ddtree_cli_args(speculative_config='{"method":"ddtree"}')

    _normalize_speculative_config_or_exit(args)
    _preflight_ddtree_or_exit(args)

    assert args._ddtree_drafter_repo == "z-lab/Qwen3.5-9B-DFlash"
    assert args._ddtree_speculative_tokens == 16
    assert args._ddtree_tree_budget == 24


def test_info_renders_ddtree_block_for_eligible_alias(capsys, monkeypatch) -> None:
    from vllm_mlx.cli import info_command

    monkeypatch.setattr(
        "vllm_mlx.speculative.ddtree.eligibility.have_runtime",
        lambda: True,
    )
    args = type("Args", (), {"model": "qwen3.5-9b-8bit"})()
    info_command(args)
    captured = capsys.readouterr()
    assert "DDTree eligibility" in captured.out
    assert "Declared support" in captured.out
    assert "z-lab/Qwen3.5-9B-DFlash" in captured.out
    assert "Spec tokens" in captured.out
    assert "Tree budget" in captured.out
    assert (
        'rapid-mlx serve qwen3.5-9b-8bit --speculative-config \'{"method":"ddtree"}\''
        in captured.out
    )


def test_info_ddtree_marks_4bit_alias_ineligible(capsys) -> None:
    from vllm_mlx.cli import info_command

    args = type("Args", (), {"model": "qwen3.5-9b-4bit"})()
    info_command(args)
    captured = capsys.readouterr()
    assert "DDTree eligibility" in captured.out
    assert "ineligible" in captured.out
    assert "4-bit" in captured.out


def test_models_listing_renders_ddtree_column(capsys) -> None:
    from vllm_mlx.cli import models_command

    models_command(None)
    captured = capsys.readouterr()
    assert "DDTree" in captured.out
    lines = captured.out.splitlines()

    def ddtree_cell(row: str) -> str:
        return row.split()[-1]

    eligible_row = next(
        (line for line in lines if line.strip().startswith("qwen3.5-9b-8bit ")),
        None,
    )
    assert eligible_row is not None
    assert ddtree_cell(eligible_row) == "✓", (
        f"DDTree column should be ✓: {eligible_row!r}"
    )
    ineligible_row = next(
        (line for line in lines if line.strip().startswith("qwen3.5-9b-4bit ")),
        None,
    )
    assert ineligible_row is not None
    assert ddtree_cell(ineligible_row) == "—", (
        f"DDTree column should be —: {ineligible_row!r}"
    )


@dataclass
class _FakeResult:
    text: str = "four"
    generated_tokens: list[int] | None = None
    metrics: dict | None = None

    def __post_init__(self) -> None:
        if self.generated_tokens is None:
            self.generated_tokens = [1, 2]
        if self.metrics is None:
            self.metrics = {"num_input_tokens": 5}


def _fake_runtime():
    from vllm_mlx.speculative.ddtree.runtime import DDTreeRuntime

    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = "user: 2+2?\nassistant:"
    tokenizer.encode.return_value = [10, 11, 12]
    target = MagicMock()
    target.tokenizer = tokenizer
    generator = MagicMock()
    generator.target = target
    generator.generate_from_tokens.return_value = _FakeResult()
    return DDTreeRuntime(
        generator=generator,
        main_model_repo="mlx-community/Qwen3.5-9B-8bit",
        drafter_repo="z-lab/Qwen3.5-9B-DFlash",
        speculative_tokens=16,
        tree_budget=24,
    )


def test_build_app_healthz_models_and_completion() -> None:
    from fastapi.testclient import TestClient

    from vllm_mlx.speculative.ddtree.server import _build_app

    runtime = _fake_runtime()
    app = _build_app(
        runtime=runtime,
        served_model_name="qwen3.5-9b-8bit",
        default_max_tokens=64,
        cors_origins=["*"],
    )
    client = TestClient(app)

    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["engine"] == "ddtree"
    assert body["drafter"] == "z-lab/Qwen3.5-9B-DFlash"
    assert body["tree_budget"] == 24

    r = client.get("/v1/models")
    assert r.status_code == 200
    assert r.json()["data"][0]["id"] == "qwen3.5-9b-8bit"

    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen3.5-9b-8bit",
            "messages": [{"role": "user", "content": "2+2?"}],
            "max_tokens": 16,
            "temperature": 0,
        },
    )
    assert r.status_code == 200
    payload = r.json()
    assert payload["choices"][0]["message"]["content"] == "four"
    assert payload["usage"]["prompt_tokens"] == 5
    assert payload["usage"]["completion_tokens"] == 2
    runtime.generator.generate.assert_not_called()
    runtime.generator.generate_from_tokens.assert_called_once()
    call = runtime.generator.generate_from_tokens.call_args
    assert call.kwargs["prompt_tokens"].tolist() == [10, 11, 12]
    assert call.kwargs["skip_special_tokens"] is True


def test_build_app_healthz_works_while_runtime_loads() -> None:
    from fastapi.testclient import TestClient

    from vllm_mlx.speculative.ddtree.server import _build_app

    future: concurrent.futures.Future = concurrent.futures.Future()
    app = _build_app(
        runtime_future=future,
        served_model_name="qwen3.5-9b-8bit",
        default_max_tokens=64,
        cors_origins=["*"],
        drafter_repo="z-lab/Qwen3.5-9B-DFlash",
        speculative_tokens=16,
        tree_budget=24,
    )
    client = TestClient(app)

    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "loading"
    assert body["ready"] is False
    assert body["drafter"] == "z-lab/Qwen3.5-9B-DFlash"

    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen3.5-9b-8bit",
            "messages": [{"role": "user", "content": "2+2?"}],
        },
    )
    assert r.status_code == 503
    assert "still loading" in r.json()["error"]["message"]

    future.set_result(_fake_runtime())
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["ready"] is True


def test_build_app_honors_api_key_and_model_name() -> None:
    from fastapi.testclient import TestClient

    from vllm_mlx.config import get_config, reset_config
    from vllm_mlx.speculative.ddtree.server import _build_app

    reset_config()
    try:
        app = _build_app(
            runtime=_fake_runtime(),
            served_model_name="qwen3.5-9b-8bit",
            default_max_tokens=64,
            cors_origins=["*"],
            api_key="secret",
        )
        client = TestClient(app)

        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3.5-9b-8bit",
                "messages": [{"role": "user", "content": "2+2?"}],
            },
        )
        assert r.status_code == 401

        r = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer secret"},
            json={
                "model": "other-model",
                "messages": [{"role": "user", "content": "2+2?"}],
            },
        )
        assert r.status_code == 404
        assert "other-model" in r.json()["error"]["message"]

        r = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer secret"},
            json={
                "model": "qwen3.5-9b-8bit",
                "messages": [{"role": "user", "content": "2+2?"}],
            },
        )
        assert r.status_code == 200
    finally:
        reset_config()

    reset_config()
    try:
        get_config().api_key = "env-secret"
        app = _build_app(
            runtime=_fake_runtime(),
            served_model_name="qwen3.5-9b-8bit",
            default_max_tokens=64,
            cors_origins=["*"],
        )
        client = TestClient(app)

        r = client.get("/healthz")
        assert r.status_code == 401
        r = client.get("/healthz", headers={"Authorization": "Bearer env-secret"})
        assert r.status_code == 200
    finally:
        reset_config()


def test_build_app_runtime_load_failure_is_sanitized() -> None:
    from fastapi.testclient import TestClient

    from vllm_mlx.speculative.ddtree.server import _build_app

    future: concurrent.futures.Future = concurrent.futures.Future()
    future.set_exception(RuntimeError("secret local path /tmp/model-cache"))
    app = _build_app(
        runtime_future=future,
        served_model_name="qwen3.5-9b-8bit",
        default_max_tokens=64,
        cors_origins=["*"],
    )
    client = TestClient(app)

    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "error"
    assert "secret local path" not in r.text

    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen3.5-9b-8bit",
            "messages": [{"role": "user", "content": "2+2?"}],
        },
    )
    assert r.status_code == 500
    assert "secret local path" not in r.text


def test_chat_completions_rejects_unsupported_ddtree_params() -> None:
    from fastapi.testclient import TestClient

    from vllm_mlx.speculative.ddtree.server import _build_app

    app = _build_app(
        runtime=_fake_runtime(),
        served_model_name="qwen3.5-9b-8bit",
        default_max_tokens=64,
        cors_origins=["*"],
    )
    client = TestClient(app)

    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen3.5-9b-8bit",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [
                {"type": "function", "function": {"name": "x", "parameters": {}}}
            ],
        },
    )
    assert r.status_code == 400
    assert "tool calling" in r.json()["error"]["message"].lower()

    unsupported_cases = [
        ({"stream": True}, "stream=true"),
        ({"stream_options": {"include_usage": True}}, "stream_options"),
        ({"temperature": 0.7}, "temperature"),
        ({"top_p": 0.9}, "top_p"),
        ({"top_k": 8}, "top_k"),
        ({"min_p": 0.1}, "min_p"),
        ({"frequency_penalty": 0.5}, "frequency_penalty"),
        ({"presence_penalty": 0.5}, "presence_penalty"),
        ({"repetition_penalty": 1.1}, "repetition_penalty"),
        ({"seed": 42}, "seed"),
        ({"logit_bias": {"1": 1.0}}, "logit_bias"),
        ({"top_logprobs": 1}, "top_logprobs"),
        ({"tool_choice": "required"}, "tool_choice"),
        ({"function_call": "auto"}, "tool_choice"),
        ({"functions": [{"name": "x"}]}, "Tool calling"),
        ({"reasoning_max_tokens": 8}, "reasoning_max_tokens"),
        ({"reasoning_effort": "low"}, "reasoning_effort"),
        ({"video_fps": 1.0}, "video parameters"),
    ]
    for extra, expected in unsupported_cases:
        payload = {
            "model": "qwen3.5-9b-8bit",
            "messages": [{"role": "user", "content": "hi"}],
        }
        payload.update(extra)
        r = client.post("/v1/chat/completions", json=payload)
        assert r.status_code == 400, extra
        assert expected in r.json()["error"]["message"]

    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen3.5-9b-8bit",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe this"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,abc"},
                        },
                    ],
                }
            ],
        },
    )
    assert r.status_code == 400
    assert "non-text" in r.json()["error"]["message"].lower()


def test_run_ddtree_server_loads_runtime_on_separate_executor(monkeypatch) -> None:
    from vllm_mlx.speculative.ddtree import server

    class RecordingExecutor:
        def __init__(self) -> None:
            self.submitted = []
            self.future: concurrent.futures.Future = concurrent.futures.Future()

        def submit(self, fn, *args, **kwargs):
            self.submitted.append((fn, args, kwargs))
            return self.future

    loader = RecordingExecutor()
    generator = RecordingExecutor()
    uvicorn_run = MagicMock()
    monkeypatch.setattr(server, "have_runtime", lambda: True)
    monkeypatch.setattr(server, "_ddtree_loader_executor", loader)
    monkeypatch.setattr(server, "_ddtree_executor", generator)
    monkeypatch.setattr("uvicorn.run", uvicorn_run)

    server.run_ddtree_server(
        main_model_repo="mlx-community/Qwen3.5-9B-8bit",
        drafter_repo="z-lab/Qwen3.5-9B-DFlash",
        speculative_tokens=16,
        tree_budget=24,
        host="127.0.0.1",
        port=59999,
        served_model_name="qwen3.5-9b-8bit",
        default_max_tokens=64,
        cors_origins=[],
        uvicorn_log_level="warning",
    )

    assert len(loader.submitted) == 1
    assert generator.submitted == []
    uvicorn_run.assert_called_once()
