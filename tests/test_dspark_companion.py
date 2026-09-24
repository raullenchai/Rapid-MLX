# SPDX-License-Identifier: Apache-2.0
"""Qualified LFM2.5-VL-3B companion DSpark server contract."""

from __future__ import annotations

import json
import socket
import sys
import time
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


def _companion_client(
    *,
    validate_request_fn,
    served_model_name="lfm-vl",
    stream_generate_fn=None,
    model_info=None,
    strict_openai_streaming=False,
    render_prompt_fn=None,
    default_timeout=1800.0,
):
    from fastapi.testclient import TestClient

    from rapid_mlx.spec_decode.dspark.runtime import CompanionDSparkRuntime
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
    render_calls = []
    generation_calls = []

    def render(*args, **kwargs):
        render_calls.append((args, kwargs))
        return "rendered"

    def generate(*args, **kwargs):
        generation_calls.append((args, kwargs))
        return SimpleNamespace(text="unexpected", prompt_tokens=1, generation_tokens=1)

    app = _build_app(
        model=SimpleNamespace(config={}),
        processor=SimpleNamespace(),
        runtime=runtime,
        served_model_name=served_model_name,
        default_max_tokens=16,
        cors_origins=[],
        render_prompt_fn=render_prompt_fn or render,
        generate_fn=generate,
        stream_generate_fn=stream_generate_fn,
        validate_request_fn=validate_request_fn,
        backend_name="LFM DSpark",
        model_info=model_info,
        strict_openai_streaming=strict_openai_streaming,
        default_timeout=default_timeout,
    )
    return TestClient(app), render_calls, generation_calls


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


@pytest.mark.parametrize("contents", ["{", "[]"], ids=["invalid-json", "non-object"])
def test_artifact_config_must_be_readable_object(tmp_path, contents) -> None:
    target, draft = _write_configs(tmp_path)
    (target / "config.json").write_text(contents)

    with pytest.raises(CompanionDSparkError, match="artifact config"):
        validate_companion_artifacts(
            LFM25_VL_3B, target_path=target, drafter_path=draft
        )


def test_artifact_config_must_exist(tmp_path) -> None:
    with pytest.raises(CompanionDSparkError, match="cannot read"):
        validate_companion_artifacts(
            LFM25_VL_3B,
            target_path=tmp_path / "missing-target",
            drafter_path=tmp_path / "missing-draft",
        )


@pytest.mark.parametrize(
    ("target_update", "draft_update", "message"),
    [
        ({"text_config": []}, {}, "target ABI mismatch"),
        ({"dtype": "float16"}, {}, "target ABI mismatch"),
        ({}, {"architectures": "Lfm2DSparkDraftModel"}, "drafter ABI mismatch"),
        ({}, {"dflash_config": []}, "drafter ABI mismatch"),
        ({}, {"block_size": "not-an-int"}, "drafter ABI mismatch"),
        (
            {},
            {"dflash_config": {"num_target_layers": 30, "target_layer_ids": [None]}},
            "drafter ABI mismatch",
        ),
    ],
)
def test_artifact_abi_rejects_malformed_shapes(
    tmp_path, target_update, draft_update, message
) -> None:
    target, draft = _write_configs(tmp_path)
    target_config = json.loads((target / "config.json").read_text())
    target_config.update(target_update)
    (target / "config.json").write_text(json.dumps(target_config))
    draft_config = json.loads((draft / "config.json").read_text())
    draft_config.update(draft_update)
    (draft / "config.json").write_text(json.dumps(draft_config))

    with pytest.raises(CompanionDSparkError, match=message):
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


def test_runtime_version_and_acceptance_metadata(monkeypatch) -> None:
    from importlib.metadata import PackageNotFoundError

    from rapid_mlx.spec_decode.dspark import runtime

    monkeypatch.setattr(runtime, "version", lambda _name: "0.7.2")
    assert runtime.have_runtime() is True
    monkeypatch.setattr(runtime, "version", lambda _name: "0.7.1")
    assert runtime.have_runtime() is False

    def missing(_name):
        raise PackageNotFoundError

    monkeypatch.setattr(runtime, "version", missing)
    assert runtime.have_runtime() is False

    drafter = SimpleNamespace(accept_lens=[1, 2])
    handle = runtime.CompanionDSparkRuntime(
        drafter=drafter,
        drafter_repo="draft",
        target_revision="target-rev",
        drafter_revision="draft-rev",
        num_speculative_tokens=7,
        draft_block_size=8,
    )
    assert handle.accept_lens_snapshot() == [1, 2]
    handle.reset_accept_lens()
    assert handle.accept_lens_snapshot() == []
    drafter.accept_lens = (3, 4)
    handle.reset_accept_lens()
    assert handle.accept_lens_snapshot() == []


def test_runtime_load_fails_without_exact_runtime(monkeypatch, tmp_path) -> None:
    from rapid_mlx.spec_decode.dspark import runtime
    from rapid_mlx.spec_decode.dspark.artifacts import CompanionDSparkArtifacts

    target, draft = _write_configs(tmp_path)
    monkeypatch.setattr(runtime, "have_runtime", lambda: False)
    with pytest.raises(RuntimeError, match="requires exactly mlx-vlm 0.7.2"):
        runtime.load_runtime(
            LFM25_VL_3B,
            CompanionDSparkArtifacts(str(target), str(draft)),
        )


def test_runtime_rejects_wrong_upstream_draft_kind(monkeypatch, tmp_path) -> None:
    from rapid_mlx.spec_decode.dspark import runtime
    from rapid_mlx.spec_decode.dspark.artifacts import CompanionDSparkArtifacts

    target, draft = _write_configs(tmp_path)
    fake_vlm = types.ModuleType("mlx_vlm")
    fake_vlm.load = lambda _path: (SimpleNamespace(), SimpleNamespace())
    fake_speculative = types.ModuleType("mlx_vlm.speculative")
    fake_drafters = types.ModuleType("mlx_vlm.speculative.drafters")
    fake_drafters.load_drafter = lambda _path, kind=None: (SimpleNamespace(), "mtp")
    fake_drafters.validate_drafter_compatibility = lambda *_args: None
    monkeypatch.setitem(sys.modules, "mlx_vlm", fake_vlm)
    monkeypatch.setitem(sys.modules, "mlx_vlm.speculative", fake_speculative)
    monkeypatch.setitem(sys.modules, "mlx_vlm.speculative.drafters", fake_drafters)
    monkeypatch.setattr(runtime, "have_runtime", lambda: True)

    with pytest.raises(RuntimeError, match="runtime mismatch"):
        runtime.load_runtime(
            LFM25_VL_3B,
            CompanionDSparkArtifacts(str(target), str(draft)),
        )


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


def test_cli_preflight_noops_without_companion_config() -> None:
    from rapid_mlx import cli

    args = SimpleNamespace(_speculative_config=None)
    assert cli._preflight_companion_dspark_or_exit(args) is None
    assert args._companion_dspark_pair is None


def test_cli_preflight_rejects_unqualified_pair(capsys) -> None:
    from rapid_mlx import cli

    args = SimpleNamespace(
        model="other/target",
        _speculative_config=SpeculativeConfig(
            method="dspark",
            model=LFM25_VL_3B.drafter_repo,
            num_speculative_tokens=7,
        ),
        dspark_num_speculative_tokens=7,
    )
    with pytest.raises(SystemExit) as exc_info:
        cli._preflight_companion_dspark_or_exit(args)
    assert exc_info.value.code == 2
    assert "qualified only" in capsys.readouterr().err


def test_cli_preflight_rejects_unsupported_server_features(capsys) -> None:
    from rapid_mlx import cli

    args = SimpleNamespace(
        model=LFM25_VL_3B.target_repo,
        _speculative_config=SpeculativeConfig(
            method="dspark",
            model=LFM25_VL_3B.drafter_repo,
            num_speculative_tokens=7,
        ),
        dspark_num_speculative_tokens=7,
        no_mllm=True,
        mcp_config="mcp.json",
        embedding_model="embed/model",
        enable_auto_tool_choice=True,
    )
    with pytest.raises(SystemExit) as exc_info:
        cli._preflight_companion_dspark_or_exit(args)
    assert exc_info.value.code == 2
    error = capsys.readouterr().err
    for option in (
        "--no-mllm",
        "--mcp-config",
        "--embedding-model",
        "--enable-auto-tool-choice",
    ):
        assert option in error


def test_cli_companion_server_dispatches_exact_runtime(monkeypatch) -> None:
    from rapid_mlx import cli
    from rapid_mlx.spec_decode.dspark import server as companion_server

    calls = {"sync": 0, "capacity": [], "run": None}
    monkeypatch.setattr(
        cli,
        "_check_memory_capacity",
        lambda model, alias=None: calls["capacity"].append((model, alias)),
    )
    monkeypatch.setattr(
        companion_server,
        "run_companion_dspark_server",
        lambda **kwargs: calls.__setitem__("run", kwargs),
    )
    server_stub = SimpleNamespace(
        _api_key="secret",
        _max_request_bytes=123,
        _body_receive_timeout_seconds=4.0,
        _default_timeout=5.0,
        _sync_config=lambda: calls.__setitem__("sync", calls["sync"] + 1),
        get_resolved_cors_policy=lambda: "cors-policy",
    )
    artifacts = SimpleNamespace(target_path="target", drafter_path="draft")
    args = SimpleNamespace(
        _companion_dspark_pair=LFM25_VL_3B,
        _companion_dspark_artifacts=artifacts,
        _original_alias="lfm-alias",
        model=LFM25_VL_3B.target_repo,
        host="127.0.0.1",
        port=8765,
        served_model_name=None,
        no_thinking=True,
        rate_limit=7,
        max_concurrent_requests=8,
        reasoning_parser=None,
    )

    assert cli._serve_companion_dspark_if_requested(
        args,
        server_module=server_stub,
        effective_max_tokens=32,
        cors_origins=["http://localhost"],
        uvicorn_log_level="warning",
    )
    assert calls["capacity"] == [(LFM25_VL_3B.target_repo, "lfm-alias")]
    assert calls["sync"] == 1
    assert calls["run"]["pair"] == LFM25_VL_3B
    assert calls["run"]["artifacts"] is artifacts
    assert calls["run"]["served_model_name"] == "lfm-alias"


def test_cli_companion_server_dispatch_noops_without_pair() -> None:
    from rapid_mlx import cli

    assert not cli._serve_companion_dspark_if_requested(
        SimpleNamespace(_companion_dspark_pair=None),
        server_module=SimpleNamespace(),
        effective_max_tokens=32,
        cors_origins=[],
        uvicorn_log_level="warning",
    )


def test_serve_command_downloads_and_dispatches_companion_pair(
    monkeypatch, capsys
) -> None:
    from rapid_mlx import _version_check, cli, model_aliases, model_auto_config, server
    from rapid_mlx.models.deepseek_v41_native import artifacts as v41_artifacts
    from rapid_mlx.spec_decode.dspark import artifacts, runtime

    args = cli.build_parser().parse_args(
        [
            "serve",
            LFM25_VL_3B.target_repo,
            "--speculative-config",
            json.dumps({"method": "dspark", "model": LFM25_VL_3B.drafter_repo}),
        ]
    )
    monkeypatch.setattr("rapid_mlx.routes.video.configure_video_jobs", lambda *_: None)
    monkeypatch.setattr(
        "rapid_mlx._parent_watchdog.install_parent_watchdog", lambda *_: None
    )
    monkeypatch.setattr(model_aliases, "resolve_profile", lambda _name: None)
    monkeypatch.setattr(model_aliases, "resolve_model", lambda name: name)
    monkeypatch.setattr(v41_artifacts, "is_product_target", lambda _name: False)
    monkeypatch.setattr(cli, "_serve_will_run_on_mllm_lane", lambda _args: False)
    monkeypatch.setattr(cli, "_resolve_audio_model_for_serve", lambda _name: None)
    monkeypatch.setattr(_version_check, "prompt_upgrade_if_available", lambda: False)
    monkeypatch.setattr(
        "rapid_mlx._download_gate.weightless_stub_notice", lambda _name: None
    )
    monkeypatch.setattr(runtime, "have_runtime", lambda: True)
    monkeypatch.setattr(model_auto_config, "detect_model_config", lambda _name: None)
    monkeypatch.setattr(
        model_auto_config, "warn_misbound_deepseek_v3_parser", lambda *_args: None
    )
    disk_checks = []
    monkeypatch.setattr(
        cli,
        "_check_disk_space",
        lambda model, **kwargs: disk_checks.append((model, kwargs)),
    )
    resolved_artifacts = SimpleNamespace(target_path="target", drafter_path="draft")
    download_calls = []
    monkeypatch.setattr(
        artifacts,
        "download_companion_artifacts",
        lambda pair: download_calls.append(pair) or resolved_artifacts,
    )
    monkeypatch.setattr(server, "configure_logging", lambda _level: "warning")
    monkeypatch.setattr(server, "_resolve_api_key", lambda value: value)
    monkeypatch.setattr(server, "configure_cors_from_env", lambda _origins: [])
    monkeypatch.setattr(server, "configure_trusted_hosts", lambda _hosts: None)
    monkeypatch.setattr(cli, "_apply_body_receive_timeout_env", lambda *_a, **_k: None)
    # The companion path returns before constructing SchedulerConfig, but
    # serve_command imports that MLX-bound module alongside the unified server.
    # Keep this no-MLX CI contract focused on the import surface the branch
    # actually consumes without mocking any companion behavior below.
    scheduler_stub = types.ModuleType("rapid_mlx.scheduler")
    scheduler_stub.SchedulerConfig = object
    monkeypatch.setitem(sys.modules, "rapid_mlx.scheduler", scheduler_stub)
    for field in (
        "_model_alias",
        "_telemetry_auto_selected",
        "_enable_audio_lane",
        "_api_key",
        "_default_timeout",
        "_gc_control",
        "_no_thinking",
        "_default_reasoning_effort",
        "_pin_system_prompt",
        "_relocate_mid_conversation_system",
        "_enable_auto_tool_choice",
        "_tool_call_parser",
        "_enable_tool_logits_bias",
        "_reasoning_parser",
    ):
        monkeypatch.setattr(server, field, getattr(server, field))
    monkeypatch.setattr(
        "rapid_mlx.middleware.request_logging.install_request_logging_middleware",
        lambda _app: None,
    )
    dispatched = {}
    monkeypatch.setattr(
        cli,
        "_serve_companion_dspark_if_requested",
        lambda dispatched_args, **kwargs: (
            dispatched.update(args=dispatched_args, **kwargs) or True
        ),
    )

    cli.serve_command(args)

    assert download_calls == [LFM25_VL_3B]
    assert disk_checks == [
        (
            LFM25_VL_3B.target_repo,
            {"force": False, "revision_override": LFM25_VL_3B.target_revision},
        ),
        (
            LFM25_VL_3B.drafter_repo,
            {"force": False, "revision_override": LFM25_VL_3B.drafter_revision},
        ),
    ]
    assert args._companion_dspark_artifacts is resolved_artifacts
    assert dispatched["args"] is args
    assert dispatched["effective_max_tokens"] == 32768
    assert "lfm-dspark-7-proposals: single-user" in capsys.readouterr().out


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


@pytest.mark.parametrize(
    "image_url",
    [
        "https://example.test/input.png",
        {"url": "https://example.test/input.png"},
    ],
    ids=["string", "object"],
)
def test_multimodal_renderer_supports_input_image_shapes(
    monkeypatch, image_url
) -> None:
    from rapid_mlx.models import mllm
    from rapid_mlx.spec_decode.dspark.server import _prepare_multimodal_prompt

    captured = {}
    prompt_utils = types.ModuleType("mlx_vlm.prompt_utils")

    def apply_chat_template(_processor, _config, messages, **kwargs):
        captured["messages"] = messages
        captured["kwargs"] = kwargs
        return "rendered"

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
                    {"type": "output_text", "text": "inspect"},
                    {"type": "input_image", "image_url": image_url},
                ],
            }
        ]
    )

    prepared = _prepare_multimodal_prompt(
        SimpleNamespace(), SimpleNamespace(config={}), request
    )
    assert prepared.generation_kwargs == {
        "image": ["local:https://example.test/input.png"]
    }
    assert captured["messages"][0]["content"] == [
        {"type": "text", "text": "inspect"},
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


def test_server_entry_loads_builds_and_binds_qualified_app(monkeypatch, capsys) -> None:
    from rapid_mlx.spec_decode.dspark import server
    from rapid_mlx.spec_decode.dspark.artifacts import CompanionDSparkArtifacts
    from rapid_mlx.spec_decode.dspark.runtime import CompanionDSparkRuntime
    from rapid_mlx.speculative.dflash import server as dflash_server

    artifacts = CompanionDSparkArtifacts("target", "draft")
    drafter = SimpleNamespace(accept_lens=[])
    runtime = CompanionDSparkRuntime(
        drafter=drafter,
        drafter_repo=LFM25_VL_3B.drafter_repo,
        target_revision=LFM25_VL_3B.target_revision,
        drafter_revision=LFM25_VL_3B.drafter_revision,
        num_speculative_tokens=7,
        draft_block_size=8,
    )
    downloads = []
    monkeypatch.setattr(
        server,
        "download_companion_artifacts",
        lambda pair: downloads.append(pair) or artifacts,
    )
    monkeypatch.setattr(
        server,
        "load_runtime",
        lambda pair, resolved: (
            SimpleNamespace(config={}),
            SimpleNamespace(),
            runtime,
        ),
    )

    class ImmediateFuture:
        def __init__(self, fn):
            self.fn = fn

        def result(self):
            return self.fn()

    monkeypatch.setattr(
        dflash_server,
        "_dflash_executor",
        SimpleNamespace(submit=lambda fn: ImmediateFuture(fn)),
    )
    built = {}
    monkeypatch.setattr(
        dflash_server,
        "_build_app",
        lambda **kwargs: built.update(kwargs) or "qualified-app",
    )
    bound = {}

    def run_uvicorn(app, **kwargs):
        bound.update(app=app, **kwargs)
        kwargs["on_server_accepting"]()

    monkeypatch.setattr("rapid_mlx._uvicorn.run_uvicorn", run_uvicorn)

    server.run_companion_dspark_server(
        pair=LFM25_VL_3B,
        artifacts=None,
        host="0.0.0.0",
        port=8123,
        served_model_name="lfm-vl",
        default_max_tokens=24,
        cors_origins=[],
        uvicorn_log_level="warning",
    )

    assert downloads == [LFM25_VL_3B]
    assert built["runtime"] is runtime
    assert built["model_info"].serving_lane_reason == "qualified_companion_dspark"
    assert built["strict_openai_streaming"] is True
    generation_kwargs = built["generation_kwargs_fn"](
        max_tokens=12, temperature=0.0, top_p=1.0
    )
    assert generation_kwargs["draft_model"] is drafter
    assert generation_kwargs["draft_block_size"] == 8
    with pytest.raises(RuntimeError, match="must remain greedy"):
        built["generation_kwargs_fn"](max_tokens=12, temperature=0.1, top_p=1.0)
    built["validate_request_fn"](_request(model="lfm-vl"))
    assert bound["app"] == "qualified-app"
    assert bound["host"] == "0.0.0.0"
    assert "Ready: http://localhost:8123/v1" in capsys.readouterr().out


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
            target_model=LFM25_VL_3B.target_repo,
            drafter_model=LFM25_VL_3B.drafter_repo,
            target_revision=LFM25_VL_3B.target_revision,
            drafter_revision=LFM25_VL_3B.drafter_revision,
            num_speculative_tokens=7,
            draft_block_size=8,
        ),
    )
    client = TestClient(app)

    health = client.get("/healthz")
    assert health.status_code == 200
    assert health.json()["algorithm"] == "dspark"
    assert health.json()["num_speculative_tokens"] == 7
    assert health.json()["target_model"] == LFM25_VL_3B.target_repo
    assert health.json()["draft_block_size"] == 8
    info = client.get("/v1/models").json()["data"][0]["speculative_decoding"]
    assert info["runtime_state"] == "active"
    assert info["target_model"] == LFM25_VL_3B.target_repo
    assert info["drafter_model"] == LFM25_VL_3B.drafter_repo
    assert info["draft_block_size"] == 8
    status = client.get("/v1/status")
    assert status.status_code == 200
    assert status.json()["speculative_decoding"]["num_speculative_tokens"] == 7
    assert status.json()["target_model"] == LFM25_VL_3B.target_repo
    assert status.json()["draft_block_size"] == 8

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


@pytest.mark.parametrize(
    "content_part, media_kind",
    [
        (
            {
                "type": "input_audio",
                "input_audio": {"data": "AAAA", "format": "wav"},
            },
            "audio",
        ),
        (
            {
                "type": "video_url",
                "video_url": {"url": "https://example.test/clip.mp4"},
            },
            "video",
        ),
    ],
    ids=["input-audio", "video-url"],
)
def test_companion_rejects_unsupported_media_before_render_or_generation(
    content_part,
    media_kind,
) -> None:
    from rapid_mlx.spec_decode.dspark.server import _validate_greedy_request

    client, render_calls, generation_calls = _companion_client(
        validate_request_fn=_validate_greedy_request
    )

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "lfm-vl",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "inspect this"},
                        content_part,
                    ],
                }
            ],
            "temperature": 0,
        },
    )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error == {
        "message": f"Model 'lfm-vl' does not support {media_kind} inputs.",
        "type": "invalid_request_error",
        "code": "unsupported_content_type",
        "param": "messages.content",
    }
    assert render_calls == []
    assert generation_calls == []


@pytest.mark.parametrize(
    "content_part",
    [
        {"type": "banana", "banana": "split"},
        {"type": "input_image"},
        {"type": "input_image", "image_url": ""},
    ],
    ids=["unknown-type", "missing-image-url", "empty-image-url"],
)
def test_companion_rejects_malformed_image_blocks_before_render_or_generation(
    content_part,
) -> None:
    from rapid_mlx.spec_decode.dspark.server import _validate_greedy_request

    client, render_calls, generation_calls = _companion_client(
        validate_request_fn=_validate_greedy_request
    )

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "lfm-vl",
            "messages": [{"role": "user", "content": [content_part]}],
            "temperature": 0,
        },
    )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert error["code"] == "invalid_request"
    assert error["param"] == "messages.content"
    assert render_calls == []
    assert generation_calls == []


def test_companion_maps_remote_image_resolution_failure_to_invalid_image(
    monkeypatch,
) -> None:
    from rapid_mlx.models import mllm
    from rapid_mlx.spec_decode.dspark.server import (
        _prepare_multimodal_prompt,
        _validate_greedy_request,
    )

    prompt_utils = types.ModuleType("mlx_vlm.prompt_utils")

    def unexpected_template(*_args, **_kwargs):
        raise AssertionError("image failure must precede template rendering")

    prompt_utils.apply_chat_template = unexpected_template
    fake_vlm = types.ModuleType("mlx_vlm")
    fake_vlm.prompt_utils = prompt_utils
    monkeypatch.setitem(sys.modules, "mlx_vlm", fake_vlm)
    monkeypatch.setitem(sys.modules, "mlx_vlm.prompt_utils", prompt_utils)

    def fail_dns(*_args, **_kwargs):
        raise socket.gaierror("test DNS failure")

    monkeypatch.setattr(mllm.socket, "getaddrinfo", fail_dns)
    client, _, generation_calls = _companion_client(
        validate_request_fn=_validate_greedy_request,
        render_prompt_fn=_prepare_multimodal_prompt,
    )

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "lfm-vl",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "https://does-not-resolve.invalid/image.png"
                            },
                        }
                    ],
                }
            ],
            "temperature": 0,
        },
    )

    assert response.status_code == 400
    assert response.json()["error"] == {
        "message": "Invalid image input.",
        "type": "invalid_request_error",
        "code": "invalid_image",
        "param": "messages.content",
    }
    assert generation_calls == []


def test_companion_maps_remote_image_connection_failure_to_invalid_image(
    monkeypatch,
) -> None:
    import requests

    from rapid_mlx.models import mllm
    from rapid_mlx.spec_decode.dspark.server import (
        _prepare_multimodal_prompt,
        _validate_greedy_request,
    )

    prompt_utils = types.ModuleType("mlx_vlm.prompt_utils")

    def unexpected_template(*_args, **_kwargs):
        raise AssertionError("image failure must precede template rendering")

    prompt_utils.apply_chat_template = unexpected_template
    fake_vlm = types.ModuleType("mlx_vlm")
    fake_vlm.prompt_utils = prompt_utils
    monkeypatch.setitem(sys.modules, "mlx_vlm", fake_vlm)
    monkeypatch.setitem(sys.modules, "mlx_vlm.prompt_utils", prompt_utils)

    def fail_connection(*_args, **_kwargs):
        raise requests.ConnectionError("test connection failure")

    # Exercise the production process_image_input -> download_image path. HEAD
    # retries as GET, and the second RequestException must remain a client-side
    # invalid-image error rather than escaping the render worker as HTTP 500.
    monkeypatch.setattr(mllm, "_guarded_request", fail_connection)
    client, _, generation_calls = _companion_client(
        validate_request_fn=_validate_greedy_request,
        render_prompt_fn=_prepare_multimodal_prompt,
    )

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "lfm-vl",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.test/image.png"},
                        }
                    ],
                }
            ],
            "temperature": 0,
        },
    )

    assert response.status_code == 400
    assert response.json()["error"] == {
        "message": "Invalid image input.",
        "type": "invalid_request_error",
        "code": "invalid_image",
        "param": "messages.content",
    }
    assert generation_calls == []


@pytest.mark.parametrize("failure_stage", ["image-internal", "template"])
def test_multimodal_renderer_does_not_reclassify_internal_errors(
    monkeypatch,
    failure_stage,
) -> None:
    from rapid_mlx.models import mllm
    from rapid_mlx.spec_decode.dspark.server import _prepare_multimodal_prompt

    prompt_utils = types.ModuleType("mlx_vlm.prompt_utils")

    def apply_chat_template(*_args, **_kwargs):
        if failure_stage == "template":
            raise ValueError("internal template failure")
        return "unexpected"

    prompt_utils.apply_chat_template = apply_chat_template
    fake_vlm = types.ModuleType("mlx_vlm")
    fake_vlm.prompt_utils = prompt_utils
    monkeypatch.setitem(sys.modules, "mlx_vlm", fake_vlm)
    monkeypatch.setitem(sys.modules, "mlx_vlm.prompt_utils", prompt_utils)

    if failure_stage == "image-internal":

        def process_image(_ref):
            raise RuntimeError("internal image failure")

    else:

        def process_image(_ref):
            return "/tmp/image.png"

    monkeypatch.setattr(mllm, "process_image_input", process_image)
    request = _request(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.test/image.png"},
                    }
                ],
            }
        ]
    )

    expected = RuntimeError if failure_stage == "image-internal" else ValueError
    with pytest.raises(expected, match="internal .* failure"):
        _prepare_multimodal_prompt(
            SimpleNamespace(),
            SimpleNamespace(config={}),
            request,
        )


def test_companion_rejects_stop_before_render_or_generation() -> None:
    from rapid_mlx.spec_decode.dspark.server import _validate_greedy_request

    client, render_calls, generation_calls = _companion_client(
        validate_request_fn=_validate_greedy_request
    )

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "lfm-vl",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0,
            "stop": "END",
        },
    )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert "stop" in error["message"]
    assert render_calls == []
    assert generation_calls == []


@pytest.mark.parametrize(
    "field, value",
    [("video_fps", 2.0), ("video_max_frames", 8)],
)
def test_companion_rejects_video_controls_before_render_or_generation(
    field,
    value,
) -> None:
    from rapid_mlx.spec_decode.dspark.server import _validate_greedy_request

    client, render_calls, generation_calls = _companion_client(
        validate_request_fn=_validate_greedy_request
    )

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "lfm-vl",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0,
            field: value,
        },
    )

    assert response.status_code == 400
    assert "video parameters" in response.json()["error"]["message"]
    assert render_calls == []
    assert generation_calls == []


def test_companion_rejects_unknown_model_before_render_or_generation() -> None:
    from rapid_mlx.spec_decode.dspark.server import _validate_companion_request

    client, render_calls, generation_calls = _companion_client(
        validate_request_fn=lambda request: _validate_companion_request(
            request,
            served_model_name="lfm-vl",
        ),
    )

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "wrong/model",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0,
        },
    )

    assert response.status_code == 404
    assert response.json()["error"] == {
        "message": "The model `wrong/model` does not exist.",
        "type": "not_found_error",
        "code": "model_not_found",
        "param": "model",
    }
    assert render_calls == []
    assert generation_calls == []


@pytest.mark.parametrize(
    "served_model_name",
    [LFM25_VL_3B.target_repo, "custom-lfm-vision"],
    ids=["exact-target", "custom-served-name"],
)
def test_companion_model_info_is_shared_by_list_and_detail(
    served_model_name,
) -> None:
    from rapid_mlx.spec_decode.dspark.server import (
        _build_companion_model_info,
        _validate_companion_request,
    )

    model_info = _build_companion_model_info(
        pair=LFM25_VL_3B,
        served_model_name=served_model_name,
    )
    client, _, _ = _companion_client(
        served_model_name=served_model_name,
        validate_request_fn=lambda request: _validate_companion_request(
            request,
            served_model_name=served_model_name,
        ),
        model_info=model_info,
    )

    listed = client.get("/v1/models").json()["data"][0]
    detailed = client.get(f"/v1/models/{served_model_name}").json()
    assert listed == detailed
    assert listed["id"] == served_model_name
    assert listed["modality"] == "image"
    assert listed["serving_lane"] == "vision"
    assert listed["serving_lane_reason"] == "qualified_companion_dspark"
    assert listed["capabilities"] == ["text", "vision"]
    speculative = listed["speculative_decoding"]
    assert speculative["target_model"] == LFM25_VL_3B.target_repo
    assert speculative["drafter_model"] == LFM25_VL_3B.drafter_repo
    assert speculative["num_speculative_tokens"] == 7
    assert speculative["draft_block_size"] == 8

    health = client.get("/healthz").json()
    status = client.get("/v1/status").json()
    assert (
        health["target_model"]
        == status["target_model"]
        == (speculative["target_model"])
    )
    assert (
        health["draft_block_size"]
        == status["draft_block_size"]
        == (speculative["draft_block_size"])
    )
    assert status["speculative_decoding"] == speculative


def test_dflash_shell_rejects_inconsistent_model_info() -> None:
    from rapid_mlx.api.models import ModelInfo
    from rapid_mlx.spec_decode.dspark.runtime import CompanionDSparkRuntime
    from rapid_mlx.speculative.dflash.server import _build_app

    runtime = CompanionDSparkRuntime(
        drafter=SimpleNamespace(accept_lens=[]),
        drafter_repo="draft",
        target_revision="target-rev",
        drafter_revision="draft-rev",
        num_speculative_tokens=7,
        draft_block_size=8,
    )
    base = {
        "model": SimpleNamespace(),
        "processor": SimpleNamespace(),
        "runtime": runtime,
        "served_model_name": "lfm-vl",
        "default_max_tokens": 16,
        "cors_origins": [],
    }
    with pytest.raises(ValueError, match="id must match"):
        _build_app(**base, model_info=ModelInfo(id="wrong"))

    first = CompanionSpeculativeDecodingInfo(
        configured=True,
        method="dspark",
        runtime_state="active",
        target_model="target",
        drafter_model="draft",
        target_revision="target-rev",
        drafter_revision="draft-rev",
        num_speculative_tokens=7,
        draft_block_size=8,
    )
    second = first.model_copy(update={"drafter_model": "other-draft"})
    with pytest.raises(ValueError, match="must match speculative_info"):
        _build_app(
            **base,
            model_info=ModelInfo(id="lfm-vl", speculative_decoding=first),
            speculative_info=second,
        )


def test_dflash_shell_detail_attaches_speculative_info_without_model_card() -> None:
    from fastapi.testclient import TestClient

    from rapid_mlx.spec_decode.dspark.runtime import CompanionDSparkRuntime
    from rapid_mlx.speculative.dflash.server import _build_app

    runtime = CompanionDSparkRuntime(
        drafter=SimpleNamespace(accept_lens=[]),
        drafter_repo="draft",
        target_revision="target-rev",
        drafter_revision="draft-rev",
        num_speculative_tokens=7,
        draft_block_size=8,
    )
    info = CompanionSpeculativeDecodingInfo(
        configured=True,
        method="dspark",
        runtime_state="active",
        target_model="target",
        drafter_model="draft",
        target_revision="target-rev",
        drafter_revision="draft-rev",
        num_speculative_tokens=7,
        draft_block_size=8,
    )
    app = _build_app(
        model=SimpleNamespace(),
        processor=SimpleNamespace(),
        runtime=runtime,
        served_model_name="lfm-vl",
        default_max_tokens=16,
        cors_origins=[],
        speculative_info=info,
    )
    detailed = TestClient(app).get("/v1/models/lfm-vl")
    assert detailed.status_code == 200
    assert detailed.json()["speculative_decoding"] == info.model_dump()


def test_dflash_shell_rejects_invalid_renderer_result() -> None:
    from rapid_mlx.spec_decode.dspark.server import _validate_greedy_request

    client, _, generation_calls = _companion_client(
        validate_request_fn=_validate_greedy_request,
        render_prompt_fn=lambda *_args, **_kwargs: object(),
    )
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "lfm-vl",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0,
        },
    )
    assert response.status_code == 500
    assert "renderer returned an invalid result" in response.json()["error"]["message"]
    assert generation_calls == []


def _stream_payloads(response) -> tuple[list[dict], list[str]]:
    data = [
        line.removeprefix("data: ")
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    return [json.loads(item) for item in data if item != "[DONE]"], data


@pytest.mark.parametrize("include_usage", [None, False, True])
def test_companion_stream_usage_follows_openai_include_usage(include_usage) -> None:
    from rapid_mlx.spec_decode.dspark.server import _validate_greedy_request

    def stream_generate(*_args, **_kwargs):
        yield SimpleNamespace(
            text="ok",
            token=7,
            prompt_tokens=3,
            generation_tokens=1,
        )

    client, _, _ = _companion_client(
        validate_request_fn=_validate_greedy_request,
        stream_generate_fn=stream_generate,
        strict_openai_streaming=True,
    )
    payload = {
        "model": "lfm-vl",
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0,
        "stream": True,
    }
    if include_usage is not None:
        payload["stream_options"] = {"include_usage": include_usage}

    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    payloads, data = _stream_payloads(response)
    assert data[-1] == "[DONE]"
    usage_payloads = [item for item in payloads if "usage" in item]
    if include_usage is True:
        assert len(usage_payloads) == 1
        usage_index = payloads.index(usage_payloads[0])
        finish_index = next(
            index
            for index, item in enumerate(payloads)
            if item.get("choices")
            and item["choices"][0].get("finish_reason") is not None
        )
        assert usage_index == finish_index + 1
        assert usage_payloads[0]["choices"] == []
        assert usage_payloads[0]["usage"] == {
            "prompt_tokens": 3,
            "completion_tokens": 1,
            "total_tokens": 4,
        }
    else:
        assert usage_payloads == []


@pytest.mark.parametrize("failure_stage", ["construction", "mid-stream"])
def test_companion_stream_generation_errors_use_canonical_envelope(
    failure_stage,
) -> None:
    from rapid_mlx.spec_decode.dspark.server import _validate_greedy_request

    secret = "/private/model/path/should-not-leak"

    if failure_stage == "construction":

        def stream_generate(*_args, **_kwargs):
            raise RuntimeError(f"construction failed at {secret}")

    else:

        def stream_generate(*_args, **_kwargs):
            yield SimpleNamespace(
                text="partial",
                token=7,
                prompt_tokens=3,
                generation_tokens=1,
            )
            raise RuntimeError(f"mid-stream failed at {secret}")

    client, _, _ = _companion_client(
        validate_request_fn=_validate_greedy_request,
        stream_generate_fn=stream_generate,
        strict_openai_streaming=True,
    )
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "lfm-vl",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    )

    assert response.status_code == 200
    payloads, data = _stream_payloads(response)
    assert data[-1] == "[DONE]"
    error_payloads = [item for item in payloads if "error" in item]
    assert len(error_payloads) == 1
    assert error_payloads[0] == {
        "error": {
            "message": (
                "Inference was interrupted by a transient engine error. "
                "Please try again."
            ),
            "type": "server_error",
            "code": "engine_aborted",
            "param": None,
        }
    }
    assert all("usage" not in item for item in payloads)
    assert not any(
        choice.get("finish_reason") == "length"
        for item in payloads
        for choice in item.get("choices", [])
    )
    assert "dflash_runtime_error" not in response.text
    assert secret not in response.text


def test_companion_stream_timeout_uses_canonical_error_without_usage() -> None:
    import asyncio

    from rapid_mlx.speculative.dflash import server as dflash_server

    class SlowGenerator:
        def __next__(self):
            time.sleep(0.05)
            raise StopIteration

        def close(self):
            pass

    async def exercise() -> None:
        stream = dflash_server._stream_completion(
            prompt="rendered",
            request=_request(stream=True),
            served_model_name="lfm-vl",
            gen_kwargs={"max_tokens": 8},
            model=SimpleNamespace(),
            processor=SimpleNamespace(),
            timeout=0.01,
            stream_generate_fn=lambda *_args, **_kwargs: SlowGenerator(),
            backend_name="LFM DSpark",
            strict_openai_streaming=True,
        )
        body = b"".join([chunk async for chunk in stream]).decode()
        data = [
            line.removeprefix("data: ")
            for line in body.splitlines()
            if line.startswith("data: ")
        ]
        payloads = [json.loads(item) for item in data if item != "[DONE]"]
        assert data[-1] == "[DONE]"
        assert [item for item in payloads if "error" in item] == [
            {
                "error": {
                    "message": "LFM DSpark stream timed out after 0.010 seconds.",
                    "type": "server_error",
                    "code": "request_timeout",
                    "param": None,
                }
            }
        ]
        assert all("usage" not in item for item in payloads)
        await asyncio.sleep(0.1)

    dflash_server._dflash_lock = asyncio.Lock()
    try:
        asyncio.run(exercise())
    finally:
        dflash_server._dflash_lock = asyncio.Lock()


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
