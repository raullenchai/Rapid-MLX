# SPDX-License-Identifier: Apache-2.0
"""Qualified LFM2.5-VL-3B companion DSpark server contract."""

from __future__ import annotations

import json
import sys
import types
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from rapid_mlx.api.models import (
    ChatCompletionRequest,
    CompanionSpeculativeDecodingInfo,
)
from rapid_mlx.spec_decode.config import SpeculativeConfig
from rapid_mlx.spec_decode.dspark.eligibility import (
    LFM25_VL_3B,
    CompanionDSparkError,
    resolve_companion_dspark_pair,
    validate_companion_artifacts,
)


def _request(**overrides) -> ChatCompletionRequest:
    payload = {
        "model": LFM25_VL_3B.target_repo,
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0,
    }
    payload.update(overrides)
    return ChatCompletionRequest.model_validate(payload)


def _write_configs(tmp_path):
    target = tmp_path / "target"
    draft = tmp_path / "draft"
    target.mkdir()
    draft.mkdir()
    (target / "config.json").write_text(
        json.dumps(
            {
                "model_type": "lfm2_vl",
                "dtype": "bfloat16",
                "text_config": {
                    "num_hidden_layers": 30,
                    "hidden_size": 2048,
                    "vocab_size": 128000,
                },
            }
        )
    )
    (draft / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Lfm2DSparkDraftModel"],
                "dtype": "bfloat16",
                "hidden_size": 2048,
                "vocab_size": 128000,
                "block_size": 9,
                "dflash_config": {
                    "num_target_layers": 30,
                    "target_layer_ids": [2, 9, 17, 21, 27],
                },
                "markov_rank": 256,
            }
        )
    )
    return target, draft


def test_exact_pair_is_the_only_qualified_companion() -> None:
    assert LFM25_VL_3B.target_revision == ("35a118d938ce6d123ac2d371649f24a8efb69058")
    assert LFM25_VL_3B.drafter_revision == ("af77e9306a26e8625fde74d2a3051ab6d21bd955")
    assert LFM25_VL_3B.num_speculative_tokens == 7
    assert LFM25_VL_3B.draft_block_size == 8
    assert (
        resolve_companion_dspark_pair(
            target_repo=LFM25_VL_3B.target_repo,
            drafter_repo=LFM25_VL_3B.drafter_repo,
            num_speculative_tokens=7,
        )
        == LFM25_VL_3B
    )
    for target, draft, k in (
        ("LiquidAI/LFM2.5-VL-3B-4bit", LFM25_VL_3B.drafter_repo, 7),
        (LFM25_VL_3B.target_repo, "other/draft", 7),
        (LFM25_VL_3B.drafter_repo, LFM25_VL_3B.drafter_repo, 7),
        (LFM25_VL_3B.target_repo, LFM25_VL_3B.drafter_repo, 8),
    ):
        with pytest.raises(CompanionDSparkError):
            resolve_companion_dspark_pair(
                target_repo=target,
                drafter_repo=draft,
                num_speculative_tokens=k,
            )


def test_artifact_abi_validation_fails_closed(tmp_path) -> None:
    target, draft = _write_configs(tmp_path)
    validate_companion_artifacts(LFM25_VL_3B, target_path=target, drafter_path=draft)

    config_path = draft / "config.json"
    config = json.loads(config_path.read_text())
    config["markov_rank"] = 128
    config_path.write_text(json.dumps(config))
    with pytest.raises(CompanionDSparkError, match="drafter ABI mismatch"):
        validate_companion_artifacts(
            LFM25_VL_3B, target_path=target, drafter_path=draft
        )


def test_artifact_downloads_pin_both_revisions(monkeypatch, tmp_path) -> None:
    from rapid_mlx.spec_decode.dspark import artifacts

    target, draft = _write_configs(tmp_path)
    calls = []

    def fake_download(repo, *, revision):
        calls.append((repo, revision))
        return str(target if repo == LFM25_VL_3B.target_repo else draft)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_download)
    resolved = artifacts.download_companion_artifacts(LFM25_VL_3B)

    assert resolved.target_path == str(target)
    assert resolved.drafter_path == str(draft)
    assert calls == [
        (LFM25_VL_3B.target_repo, LFM25_VL_3B.target_revision),
        (LFM25_VL_3B.drafter_repo, LFM25_VL_3B.drafter_revision),
    ]


def test_runtime_loads_and_attaches_without_fallback(monkeypatch, tmp_path) -> None:
    from rapid_mlx.spec_decode.dspark import runtime
    from rapid_mlx.spec_decode.dspark.artifacts import CompanionDSparkArtifacts

    target, draft = _write_configs(tmp_path)
    target_model = SimpleNamespace()
    processor = SimpleNamespace()
    drafter = SimpleNamespace(accept_lens=[])
    compatibility_calls = []

    fake_vlm = types.ModuleType("mlx_vlm")
    fake_vlm.load = lambda path: (target_model, processor)
    fake_speculative = types.ModuleType("mlx_vlm.speculative")
    fake_drafters = types.ModuleType("mlx_vlm.speculative.drafters")
    fake_drafters.load_drafter = lambda path, kind=None: (drafter, "dflash")
    fake_drafters.validate_drafter_compatibility = lambda model, draft_model, kind: (
        compatibility_calls.append((model, draft_model, kind))
    )
    monkeypatch.setitem(sys.modules, "mlx_vlm", fake_vlm)
    monkeypatch.setitem(sys.modules, "mlx_vlm.speculative", fake_speculative)
    monkeypatch.setitem(sys.modules, "mlx_vlm.speculative.drafters", fake_drafters)
    monkeypatch.setattr(runtime, "have_runtime", lambda: True)

    model, loaded_processor, handle = runtime.load_runtime(
        LFM25_VL_3B,
        CompanionDSparkArtifacts(str(target), str(draft)),
    )

    assert (model, loaded_processor) == (target_model, processor)
    assert compatibility_calls == [(target_model, drafter, "dflash")]
    assert handle.algorithm == "dspark"
    assert handle.num_speculative_tokens == 7
    assert handle.draft_block_size == 8


def test_cli_preflight_requires_exact_runtime_and_pair(monkeypatch) -> None:
    from rapid_mlx import cli
    from rapid_mlx.spec_decode.dspark import runtime

    args = SimpleNamespace(
        model=LFM25_VL_3B.target_repo,
        _speculative_config=SpeculativeConfig(
            method="dspark",
            model=LFM25_VL_3B.drafter_repo,
            num_speculative_tokens=7,
        ),
        dspark_num_speculative_tokens=7,
        no_mllm=False,
        mcp_config=None,
        embedding_model=None,
        enable_auto_tool_choice=False,
    )
    monkeypatch.setattr(runtime, "have_runtime", lambda: True)
    assert cli._preflight_companion_dspark_or_exit(args) == LFM25_VL_3B

    monkeypatch.setattr(runtime, "have_runtime", lambda: False)
    with pytest.raises(SystemExit) as exc_info:
        cli._preflight_companion_dspark_or_exit(args)
    assert exc_info.value.code == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("temperature", 0.1),
        ("top_p", 0.9),
        ("top_k", 20),
        ("min_p", 0.1),
        ("repetition_penalty", 1.1),
        ("presence_penalty", 0.1),
        ("frequency_penalty", 0.1),
        ("seed", 7),
        ("logprobs", True),
        ("logit_bias", {"1": 1.0}),
        ("response_format", {"type": "json_object"}),
        ("chat_template_kwargs", {"custom": True}),
        ("reasoning_effort", "low"),
        ("enable_thinking", True),
    ],
)
def test_request_contract_rejects_unqualified_processing(field, value) -> None:
    from rapid_mlx.spec_decode.dspark.server import _validate_greedy_request

    with pytest.raises(HTTPException) as exc_info:
        _validate_greedy_request(_request(**{field: value}))
    assert exc_info.value.status_code == 400
    assert field.split("_")[0] in exc_info.value.detail


def test_request_contract_rejects_tools_before_generation() -> None:
    from rapid_mlx.spec_decode.dspark.server import _validate_greedy_request

    request = _request(
        tools=[
            {
                "type": "function",
                "function": {"name": "lookup", "parameters": {}},
            }
        ]
    )
    with pytest.raises(HTTPException) as exc_info:
        _validate_greedy_request(request)
    assert exc_info.value.status_code == 400
    assert "tools" in exc_info.value.detail

    with pytest.raises(HTTPException) as choice_exc:
        _validate_greedy_request(_request(tool_choice="auto"))
    assert choice_exc.value.status_code == 400


def test_multimodal_renderer_preserves_image_content(monkeypatch) -> None:
    from rapid_mlx.models import mllm
    from rapid_mlx.spec_decode.dspark.server import _prepare_multimodal_prompt

    captured = {}
    prompt_utils = types.ModuleType("mlx_vlm.prompt_utils")

    def apply_chat_template(processor, config, messages, **kwargs):
        captured["messages"] = messages
        captured["kwargs"] = kwargs
        return "rendered-image-prompt"

    prompt_utils.apply_chat_template = apply_chat_template
    fake_vlm = types.ModuleType("mlx_vlm")
    fake_vlm.prompt_utils = prompt_utils
    monkeypatch.setitem(sys.modules, "mlx_vlm", fake_vlm)
    monkeypatch.setitem(sys.modules, "mlx_vlm.prompt_utils", prompt_utils)
    monkeypatch.setattr(mllm, "process_image_input", lambda ref: f"local:{ref}")

    request = _request(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.test/cat.png"},
                    },
                ],
            }
        ]
    )
    prepared = _prepare_multimodal_prompt(
        SimpleNamespace(), SimpleNamespace(config={}), request
    )

    assert prepared.prompt == "rendered-image-prompt"
    assert prepared.generation_kwargs == {
        "image": ["local:https://example.test/cat.png"]
    }
    assert captured["kwargs"]["num_images"] == 1
    assert captured["messages"][0]["content"] == [
        {"type": "text", "text": "describe"},
        {"type": "image"},
    ]


def test_server_load_failure_happens_before_listener(monkeypatch, tmp_path) -> None:
    from rapid_mlx.spec_decode.dspark import server
    from rapid_mlx.spec_decode.dspark.artifacts import CompanionDSparkArtifacts

    target, draft = _write_configs(tmp_path)
    listener_called = False

    def fail_load(*_args, **_kwargs):
        raise RuntimeError("attach failed")

    def listen(*_args, **_kwargs):
        nonlocal listener_called
        listener_called = True

    monkeypatch.setattr(server, "load_runtime", fail_load)
    monkeypatch.setattr("rapid_mlx._uvicorn.run_uvicorn", listen)

    with pytest.raises(RuntimeError, match="attach failed"):
        server.run_companion_dspark_server(
            pair=LFM25_VL_3B,
            artifacts=CompanionDSparkArtifacts(str(target), str(draft)),
            host="127.0.0.1",
            port=8000,
            served_model_name="lfm-vl",
            default_max_tokens=16,
            cors_origins=[],
            uvicorn_log_level="warning",
        )
    assert listener_called is False


def test_dflash_shell_preserves_media_and_surfaces_runtime_status() -> None:
    from fastapi.testclient import TestClient

    from rapid_mlx.spec_decode.dspark.runtime import CompanionDSparkRuntime
    from rapid_mlx.spec_decode.dspark.server import _validate_greedy_request
    from rapid_mlx.speculative.dflash.server import PreparedPrompt, _build_app

    drafter = SimpleNamespace(accept_lens=[])
    runtime = CompanionDSparkRuntime(
        drafter=drafter,
        drafter_repo=LFM25_VL_3B.drafter_repo,
        target_revision=LFM25_VL_3B.target_revision,
        drafter_revision=LFM25_VL_3B.drafter_revision,
        num_speculative_tokens=7,
        draft_block_size=8,
    )
    generation_calls = []

    def generate(model, processor, prompt, **kwargs):
        generation_calls.append((prompt, kwargs))
        return SimpleNamespace(text="ok", prompt_tokens=3, generation_tokens=1)

    app = _build_app(
        model=SimpleNamespace(),
        processor=SimpleNamespace(),
        runtime=runtime,
        served_model_name="lfm-vl",
        default_max_tokens=16,
        cors_origins=[],
        render_prompt_fn=lambda *_a, **_kw: PreparedPrompt(
            "rendered", {"image": ["/tmp/image.png"]}
        ),
        generate_fn=generate,
        generation_kwargs_fn=lambda **_kw: {
            "max_tokens": 16,
            "temperature": 0.0,
            "top_p": 1.0,
            "draft_model": drafter,
            "draft_kind": "dflash",
            "draft_block_size": 8,
        },
        validate_request_fn=_validate_greedy_request,
        backend_name="LFM DSpark",
        speculative_info=CompanionSpeculativeDecodingInfo(
            configured=True,
            method="dspark",
            runtime_state="active",
            drafter_model=LFM25_VL_3B.drafter_repo,
            target_revision=LFM25_VL_3B.target_revision,
            drafter_revision=LFM25_VL_3B.drafter_revision,
            num_speculative_tokens=7,
        ),
    )
    client = TestClient(app)

    health = client.get("/healthz")
    assert health.status_code == 200
    assert health.json()["algorithm"] == "dspark"
    assert health.json()["num_speculative_tokens"] == 7
    info = client.get("/v1/models").json()["data"][0]["speculative_decoding"]
    assert info["runtime_state"] == "active"
    assert info["drafter_model"] == LFM25_VL_3B.drafter_repo
    status = client.get("/v1/status")
    assert status.status_code == 200
    assert status.json()["speculative_decoding"]["num_speculative_tokens"] == 7

    rejected = client.post(
        "/v1/chat/completions",
        json={
            "model": "lfm-vl",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0.2,
        },
    )
    assert rejected.status_code == 400
    assert rejected.json()["error"]["type"] == "invalid_request_error"
    assert generation_calls == []

    completed = client.post(
        "/v1/chat/completions",
        json={
            "model": "lfm-vl",
            "messages": [{"role": "user", "content": "describe"}],
            "temperature": 0,
        },
    )
    assert completed.status_code == 200
    assert generation_calls[0][0] == "rendered"
    assert generation_calls[0][1]["image"] == ["/tmp/image.png"]
    assert generation_calls[0][1]["draft_model"] is drafter
    assert generation_calls[0][1]["draft_block_size"] == 8


def test_runtime_generation_failure_is_http_500_without_ar_fallback() -> None:
    from fastapi.testclient import TestClient

    from rapid_mlx.spec_decode.dspark.runtime import CompanionDSparkRuntime
    from rapid_mlx.spec_decode.dspark.server import _validate_greedy_request
    from rapid_mlx.speculative.dflash.server import _build_app

    drafter = SimpleNamespace(accept_lens=[])
    runtime = CompanionDSparkRuntime(
        drafter=drafter,
        drafter_repo=LFM25_VL_3B.drafter_repo,
        target_revision=LFM25_VL_3B.target_revision,
        drafter_revision=LFM25_VL_3B.drafter_revision,
        num_speculative_tokens=7,
        draft_block_size=8,
    )
    calls = 0

    def fail(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("verification failed")

    app = _build_app(
        model=SimpleNamespace(),
        processor=SimpleNamespace(),
        runtime=runtime,
        served_model_name="lfm-vl",
        default_max_tokens=16,
        cors_origins=[],
        render_prompt_fn=lambda *_a, **_kw: "rendered",
        generate_fn=fail,
        generation_kwargs_fn=lambda **_kw: {
            "max_tokens": 16,
            "temperature": 0.0,
            "top_p": 1.0,
            "draft_model": drafter,
            "draft_kind": "dflash",
            "draft_block_size": 8,
        },
        validate_request_fn=_validate_greedy_request,
        backend_name="LFM DSpark",
    )
    response = TestClient(app).post(
        "/v1/chat/completions",
        json={
            "model": "lfm-vl",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0,
        },
    )
    assert response.status_code == 500
    assert "verification failed" in response.json()["error"]["message"]
    assert calls == 1
