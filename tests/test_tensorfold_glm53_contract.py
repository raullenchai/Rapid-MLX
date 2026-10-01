# SPDX-License-Identifier: Apache-2.0
"""Contracts for the opt-in GLM-5.3 TensorFold product profile."""

import asyncio
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


def test_alias_selects_target_only_tensorfold_mtp() -> None:
    from rapid_mlx import cli

    args = SimpleNamespace(
        model="glm5.3-flash-tensorfold",
        speculative_config=None,
        no_spec_decode=False,
        mllm=False,
    )
    cli._normalize_speculative_config_or_exit(args)
    assert args.speculative_config == '{"method":"mtp","backend":"tensorfold"}'
    assert args._speculative_config.method == "mtp"
    assert args._speculative_config.backend == "tensorfold"


def test_resolved_alias_still_selects_target_only_tensorfold_mtp() -> None:
    """main() resolves aliases before serve_command normalizes acceleration."""
    from rapid_mlx import cli

    args = SimpleNamespace(
        model="Vontra/GLM-5.3-Flash-MLX-4bit-MTP",
        _original_alias="glm5.3-flash-tensorfold",
        speculative_config=None,
        no_spec_decode=False,
        mllm=False,
    )
    cli._normalize_speculative_config_or_exit(args)
    assert args.speculative_config == '{"method":"mtp","backend":"tensorfold"}'
    assert args._speculative_config.backend == "tensorfold"


def test_resolved_alias_explicit_opt_out_keeps_normal_mode() -> None:
    from rapid_mlx import cli

    args = SimpleNamespace(
        model="Vontra/GLM-5.3-Flash-MLX-4bit-MTP",
        _original_alias="glm5.3-flash-tensorfold",
        speculative_config=None,
        no_spec_decode=True,
        mllm=False,
    )
    cli._normalize_speculative_config_or_exit(args)
    assert args.speculative_config is None
    assert args._speculative_config is None


def test_ordinary_glm_alias_keeps_existing_default() -> None:
    from rapid_mlx import cli

    assert cli._tensorfold_mtp_profile("glm5.3-flash-4bit") is None


def test_glm_environment_fails_closed(monkeypatch) -> None:
    from rapid_mlx.speculative import tensorfold_glm53
    from rapid_mlx.speculative.tensorfold_glm53 import (
        TensorFoldUnavailable,
        require_environment,
    )

    monkeypatch.setattr(tensorfold_glm53.sys, "platform", "darwin")
    with pytest.raises(TensorFoldUnavailable, match="256 GB"):
        require_environment(mlx_version="0.32.3", machine="arm64", memory_gb=192)
    with pytest.raises(TensorFoldUnavailable, match="mlx==0.32.3"):
        require_environment(mlx_version="0.32.2", machine="arm64", memory_gb=256)


def test_glm_target_requires_pinned_snapshot(tmp_path: Path) -> None:
    from rapid_mlx.speculative.tensorfold_glm53 import (
        TensorFoldUnavailable,
        validate_target,
    )

    with pytest.raises(TensorFoldUnavailable, match="pinned Hugging Face snapshot"):
        validate_target(tmp_path)


def test_glm_target_download_and_layout_gates(monkeypatch, tmp_path: Path) -> None:
    from rapid_mlx.speculative import tensorfold_glm53 as adapter

    target = (
        tmp_path
        / "models--Vontra--GLM"
        / "snapshots"
        / adapter.SUPPORTED_TARGET_REVISION
    )
    target.mkdir(parents=True)
    config = {
        "model_type": "glm5_next",
        "text_config": {
            "num_hidden_layers": 45,
            "num_nextn_predict_layers": 1,
            "hidden_size": 4096,
        },
        "quantization_config": {"bits": 4, "group_size": 64, "mode": "affine"},
    }
    (target / "config.json").write_text(json.dumps(config))
    (target / "model.safetensors.index.json").write_text(
        json.dumps(
            {"weight_map": {"model.language_model.layers.45.eh_proj.weight": "x"}}
        )
    )
    adapter.validate_target(target)

    seen = []
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda repo, *, revision: (seen.append((repo, revision)), str(target))[1],
    )
    assert adapter.download_qualified_target() == str(target)
    assert seen == [(adapter.SUPPORTED_TARGET, adapter.SUPPORTED_TARGET_REVISION)]

    config["text_config"]["hidden_size"] = 1
    (target / "config.json").write_text(json.dumps(config))
    with pytest.raises(adapter.TensorFoldUnavailable, match="qualified"):
        adapter.validate_target(target)
    (target / "config.json").unlink()
    with pytest.raises(adapter.TensorFoldUnavailable, match="readable"):
        adapter.validate_target(target)


def _qualified_direct_url() -> dict:
    from rapid_mlx.speculative.tensorfold_glm53 import SUPPORTED_RUNTIME_REVISION

    return {
        "url": "https://github.com/ashhart/TensorFold.git",
        "vcs_info": {"vcs": "git", "commit_id": SUPPORTED_RUNTIME_REVISION},
    }


def test_glm_runtime_and_platform_gates(monkeypatch) -> None:
    from importlib.metadata import PackageNotFoundError

    from rapid_mlx.speculative import tensorfold_glm53 as adapter

    adapter.require_runtime("0.6.0", direct_url=_qualified_direct_url())
    monkeypatch.setattr(adapter.importlib.metadata, "version", lambda _name: "0.6.0")
    monkeypatch.setattr(
        adapter.importlib.metadata,
        "distribution",
        lambda _name: SimpleNamespace(
            read_text=lambda filename: (
                json.dumps(_qualified_direct_url())
                if filename == "direct_url.json"
                else None
            )
        ),
    )
    adapter.require_runtime()
    monkeypatch.setattr(
        adapter.importlib.metadata,
        "distribution",
        lambda _name: SimpleNamespace(
            read_text=lambda filename: "[]" if filename == "direct_url.json" else None
        ),
    )
    with pytest.raises(adapter.TensorFoldUnavailable, match="exact qualified"):
        adapter.require_runtime()
    with pytest.raises(adapter.TensorFoldUnavailable, match="found 0.5.0"):
        adapter.require_runtime("0.5.0", direct_url=_qualified_direct_url())
    for provenance in (
        {},
        {"vcs_info": {"vcs": "git", "commit_id": "0" * 40}},
        {
            "vcs_info": {
                "vcs": "git",
                "commit_id": adapter.SUPPORTED_RUNTIME_REVISION,
            },
            "dir_info": {"editable": True},
        },
    ):
        with pytest.raises(adapter.TensorFoldUnavailable, match="exact qualified"):
            adapter.require_runtime("0.6.0", direct_url=provenance)
    monkeypatch.setattr(
        adapter.importlib.metadata,
        "version",
        lambda _name: (_ for _ in ()).throw(PackageNotFoundError()),
    )
    with pytest.raises(adapter.TensorFoldUnavailable, match="tensorfold==0.6.0"):
        adapter.require_runtime()
    monkeypatch.setattr(adapter.sys, "platform", "darwin")
    with pytest.raises(adapter.TensorFoldUnavailable, match="Apple Silicon"):
        adapter.require_environment(
            mlx_version="0.32.3", machine="x86_64", memory_gb=256
        )
    with pytest.raises(adapter.TensorFoldUnavailable, match="mlx==0.32.3"):
        adapter.require_environment(machine="arm64", memory_gb=256)
    monkeypatch.setattr(
        adapter.os,
        "sysconf",
        lambda key: 16_384 if key == "SC_PAGE_SIZE" else 16_777_216,
    )
    adapter.require_environment(mlx_version="0.32.3", machine="arm64")


def test_reasoning_budget_reaches_tensorfold_job() -> None:
    from rapid_mlx.speculative.tensorfold_qwen27_server import generation_kwargs

    request = SimpleNamespace(
        top_k=None,
        min_p=None,
        seed=1,
        stop=None,
        reasoning_max_tokens=256,
    )
    assert (
        generation_kwargs(max_tokens=512, temperature=0.0, top_p=1.0, request=request)[
            "thinking_budget"
        ]
        == 256
    )


@pytest.mark.parametrize("decode_path", ["drafted", "serial"])
def test_glm_parser_sanitizes_tensorfold_implicit_reasoning(decode_path: str) -> None:
    """Both TensorFold paths expose the same raw implicit-think wire shape."""
    from rapid_mlx.reasoning.glm5_parser import Glm5ReasoningParser

    raw = (
        'The user requested exactly "GLM_SMOKE_OK". I should comply exactly.'
        "</think>GLM_SMOKE_OK"
    )
    reasoning, content = Glm5ReasoningParser().extract_reasoning(
        raw,
        prompt_thinking_active=True,
    )
    assert decode_path in {"drafted", "serial"}
    assert (
        reasoning
        == 'The user requested exactly "GLM_SMOKE_OK". I should comply exactly.'
    )
    assert content == "GLM_SMOKE_OK"


def test_glm_stream_matches_nonstream_boundary_trimming() -> None:
    from rapid_mlx.reasoning.glm5_parser import Glm5ReasoningParser

    parser = Glm5ReasoningParser()
    parser.configure_request(prompt_thinking_active=True)
    first = parser.extract_reasoning_streaming(
        "reasoning", "reasoning</think>\n\n```python\n", "</think>\n\n```python\n"
    )
    second = parser.extract_reasoning_streaming(
        "reasoning</think>\n\n```python\n",
        "reasoning</think>\n\n```python\n    return []\n",
        "    return []\n",
    )
    assert first is not None and first.content == "```python\n"
    assert second is not None and second.content == "    return []\n"


@pytest.mark.parametrize("decode_path", ["drafted", "serial"])
def test_non_stream_truncation_never_publishes_implicit_reasoning(
    monkeypatch, decode_path: str
) -> None:
    from rapid_mlx.speculative.dflash import server

    private = "private reasoning truncated before the closing marker"
    monkeypatch.setattr(
        server,
        "get_config",
        lambda: SimpleNamespace(reasoning_parser_name="glm5"),
    )

    async def run():
        return await server._non_stream_completion(
            prompt="prompt ends in an implicit <think> opener",
            request=SimpleNamespace(tools=None),
            served_model_name="glm5.3-flash-tensorfold",
            gen_kwargs={"max_tokens": 1},
            model=None,
            processor=SimpleNamespace(
                chat_template=(
                    "{% if add_generation_prompt and enable_thinking %}"
                    "<think>{% endif %}"
                )
            ),
            enable_thinking=True,
            generate_fn=lambda *_args, **_kwargs: SimpleNamespace(
                text=private,
                generation_tokens=1,
                prompt_tokens=7,
            ),
            backend_name=f"TensorFold {decode_path}",
        )

    response = asyncio.run(run())
    message = response.choices[0].message
    assert message.content is None
    assert message.reasoning_content == private


def test_non_stream_non_implicit_template_keeps_plain_answer_public(
    monkeypatch,
) -> None:
    from rapid_mlx.speculative.dflash import server

    monkeypatch.setattr(
        server,
        "get_config",
        lambda: SimpleNamespace(reasoning_parser_name="qwen3"),
    )

    async def run():
        return await server._non_stream_completion(
            prompt="ordinary assistant prompt",
            request=SimpleNamespace(tools=None),
            served_model_name="qwen-like",
            gen_kwargs={"max_tokens": 1},
            model=None,
            processor=SimpleNamespace(
                chat_template=(
                    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
                )
            ),
            enable_thinking=True,
            generate_fn=lambda *_args, **_kwargs: SimpleNamespace(
                text="plain public answer",
                generation_tokens=1,
                prompt_tokens=4,
            ),
        )

    response = asyncio.run(run())
    message = response.choices[0].message
    assert message.content == "plain public answer"
    assert message.reasoning_content is None


def test_cli_dispatches_qualified_glm_tensorfold_profile(monkeypatch) -> None:
    from rapid_mlx import cli
    from rapid_mlx.speculative import tensorfold_glm53

    captured = {}
    monkeypatch.setattr(cli, "_check_disk_space", lambda *_a, **_k: None)
    monkeypatch.setattr(cli, "_check_memory_capacity", lambda *_a, **_k: None)
    monkeypatch.setattr(cli, "_resolved_serve_port", lambda _args: 8123)
    monkeypatch.setattr(cli, "port_explicit_for", lambda _args: True)
    monkeypatch.setattr(
        tensorfold_glm53,
        "run_tensorfold_glm53_server",
        lambda **kwargs: captured.update(kwargs),
    )
    server = SimpleNamespace(
        _sync_config=lambda: None,
        _api_key=None,
        _max_request_bytes=1024,
        _body_receive_timeout_seconds=1.0,
        _default_timeout=2.0,
        get_resolved_cors_policy=lambda: None,
    )
    args = SimpleNamespace(
        mtp_backend="tensorfold",
        _original_alias="glm5.3-flash-tensorfold",
        model="/pinned/target",
        force_disk_check=False,
        host="127.0.0.1",
        served_model_name=None,
        no_thinking=False,
        rate_limit=0,
        max_concurrent_requests=8,
        reasoning_parser="glm5",
        default_reasoning_effort=None,
    )
    assert cli._serve_tensorfold_mtp_if_requested(
        args,
        server_module=server,
        effective_max_tokens=512,
        cors_origins=[],
        uvicorn_log_level="info",
    )
    assert captured["main_model_repo"] == "/pinned/target"
    assert captured["drafter_repo"] == ""
    assert captured["served_model_name"] == "glm5.3-flash-tensorfold"

    args.mtp_backend = None
    assert not cli._serve_tensorfold_mtp_if_requested(
        args,
        server_module=server,
        effective_max_tokens=512,
        cors_origins=[],
        uvicorn_log_level="info",
    )
    args.mtp_backend = "tensorfold"
    args._original_alias = "glm5.3-flash-4bit"
    with pytest.raises(SystemExit, match="2"):
        cli._serve_tensorfold_mtp_if_requested(
            args,
            server_module=server,
            effective_max_tokens=512,
            cors_origins=[],
            uvicorn_log_level="info",
        )


def test_glm_preflight_and_listen_fd_gate(monkeypatch, capsys) -> None:
    from rapid_mlx import cli
    from rapid_mlx.speculative import tensorfold_glm53

    args = SimpleNamespace(
        model="glm5.3-flash-tensorfold",
        _original_alias="glm5.3-flash-tensorfold",
    )
    monkeypatch.setattr(tensorfold_glm53, "require_runtime", lambda: None)
    monkeypatch.setattr(tensorfold_glm53, "require_environment", lambda: None)
    cli._preflight_tensorfold_qwen27_or_exit(args)
    monkeypatch.setattr(
        tensorfold_glm53,
        "require_runtime",
        lambda: (_ for _ in ()).throw(tensorfold_glm53.TensorFoldUnavailable("bad")),
    )
    with pytest.raises(SystemExit, match="1"):
        cli._preflight_tensorfold_qwen27_or_exit(args)
    assert "GLM-5.3-Flash" in capsys.readouterr().err

    gate = SimpleNamespace(listen_fd=3, mtp_backend="tensorfold")
    with pytest.raises(SystemExit, match="2"):
        cli._reject_unsupported_listen_fd_lane(gate, owns_v41_product_download=False)


def test_glm_loader_uses_family_memory_and_lane_contracts(
    monkeypatch, tmp_path
) -> None:
    from rapid_mlx.speculative import tensorfold_glm53 as adapter

    seen = {}

    class Model:
        def release_rounds(self):
            seen["released"] = True

    model = Model()
    tokenizer = object()

    class Package:
        MLX_ENV = {"RAPID_TEST_TF_GLM": "1"}

        @staticmethod
        def load(path, **kwargs):
            seen.update(load_path=path, load_kwargs=kwargs)
            return model, tokenizer

        @staticmethod
        def engine_settings(_model):
            return {"max_rows": 16, "max_draft": 15}

    family = SimpleNamespace(model_type="glm5_next", package=Package)
    families = ModuleType("tensorfold.families")
    families.detect = lambda _path: family
    monkeypatch.setitem(sys.modules, "tensorfold.families", families)

    mlx = ModuleType("mlx")
    mlx_core = ModuleType("mlx.core")
    mlx.core = mlx_core
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", mlx_core)

    lane = ModuleType("tensorfold.engine.lane_engine")
    lane.LaneEngine = type("LaneEngine", (), {})
    monkeypatch.setitem(sys.modules, "tensorfold.engine.lane_engine", lane)
    prefill = ModuleType("tensorfold.engine.prefill_plan")

    class PrefillPlan:
        def __init__(self, *args):
            seen["prefill"] = args

    prefill.PrefillPlan = PrefillPlan
    prefill.message_markers = lambda _tokenizer: ((1,), (2,))
    monkeypatch.setitem(sys.modules, "tensorfold.engine.prefill_plan", prefill)

    app_module = ModuleType("tensorfold.server.app")

    class ChatApp:
        def __init__(self, loaded_model, loaded_tokenizer, **kwargs):
            seen.update(
                app_model=loaded_model, app_tokenizer=loaded_tokenizer, app=kwargs
            )
            self.scheduler = SimpleNamespace(stop=lambda: None)

    app_module.ChatApp = ChatApp
    monkeypatch.setitem(sys.modules, "tensorfold.server.app", app_module)
    budget = ModuleType("tensorfold.server.memory_budget")
    budget.PROCESS_BYTES = 3 * 1024**3
    budget.configure_mlx = lambda *_a, **_k: 220 * 1024**3
    monkeypatch.setitem(sys.modules, "tensorfold.server.memory_budget", budget)
    residency = ModuleType("tensorfold.server.residency")
    residency.wire_resident = lambda _mx, limit: seen.update(wired=limit)
    monkeypatch.setitem(sys.modules, "tensorfold.server.residency", residency)

    monkeypatch.setattr(adapter, "require_runtime", lambda: None)
    monkeypatch.setattr(adapter, "require_environment", lambda: None)
    monkeypatch.setattr(adapter, "validate_target", lambda _path: None)
    backend = adapter.TensorFoldGLM53Backend.load(
        str(tmp_path), "", served_name="glm", context_window=4096, max_tokens=256
    )
    assert seen["load_kwargs"] == {"mtp_drafts": 3}
    assert seen["released"] is True
    assert seen["app"]["max_rows"] == 16
    assert seen["app"]["max_draft"] == 15
    assert seen["app"]["enable_thinking"] is True
    assert seen["app"]["context_window"] == 4096
    engine_keywords = seen["app"]["engine_factory"].keywords
    assert engine_keywords["prefill_pass"] == 8
    assert engine_keywords["pass_cache"] == 16 * 1024**3
    backend.close()

    family.model_type = "wrong"
    with pytest.raises(adapter.TensorFoldUnavailable, match="did not select"):
        adapter.TensorFoldGLM53Backend.load(str(tmp_path), "", served_name="glm")


def test_glm_server_wrapper_declares_product_metadata(monkeypatch) -> None:
    from rapid_mlx.speculative import tensorfold_glm53 as adapter
    from rapid_mlx.speculative import tensorfold_qwen27_server

    captured = {}
    monkeypatch.setattr(
        tensorfold_qwen27_server,
        "run_tensorfold_qwen27_server",
        lambda **kwargs: captured.update(kwargs),
    )
    adapter.run_tensorfold_glm53_server(main_model_repo="/target")
    assert captured["method"] == "mtp"
    assert captured["paired_repository"] is None
    assert captured["supports_reasoning_budget"] is True
