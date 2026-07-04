# SPDX-License-Identifier: Apache-2.0
"""Integration-style tests for the DDTree MVP plumbing."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock


def test_serve_parser_exposes_enable_ddtree() -> None:
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, "-m", "vllm_mlx.cli", "serve", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert out.returncode == 0, out.stderr
    assert "--enable-ddtree" in out.stdout
    assert "dtree-mlx" in out.stdout


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
    assert "rapid-mlx serve qwen3.5-9b-8bit --enable-ddtree" in captured.out


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
    eligible_row = next(
        (line for line in lines if "qwen3.5-9b-8bit " in line),
        None,
    )
    assert eligible_row is not None
    assert "✓" in eligible_row, f"DDTree column should be ✓: {eligible_row!r}"
    ineligible_row = next(
        (line for line in lines if "qwen3.5-9b-4bit " in line),
        None,
    )
    assert ineligible_row is not None
    assert "—" in ineligible_row, f"DDTree column should be —: {ineligible_row!r}"


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
    target = MagicMock()
    target.tokenizer = tokenizer
    generator = MagicMock()
    generator.target = target
    generator.generate.return_value = _FakeResult()
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

    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen3.5-9b-8bit",
            "messages": [{"role": "user", "content": "hi"}],
            "top_p": 0.9,
        },
    )
    assert r.status_code == 400
    assert "top_p" in r.json()["error"]["message"]
