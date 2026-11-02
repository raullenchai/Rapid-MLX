# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from rapid_mlx.speculative import tensorfold_qwen27_server as server


def test_deepseek_tool_docs_exclude_tensorfold_lane():
    guide = (Path(__file__).parents[1] / "docs/guides/tool-calling.md").read_text()
    row = next(
        line for line in guide.splitlines() if line.startswith("| `deepseek_v4_flash`")
    )
    assert "TensorFold" not in row
    assert "Tool calling is not supported on the DeepSeek TensorFold lane yet" in guide


def test_deepseek_alias_retains_parser_for_ordinary_engine_opt_out():
    from rapid_mlx import cli
    from rapid_mlx.model_auto_config import detect_model_config
    from rapid_mlx.speculative.tensorfold_families import PROFILES

    alias = "deepseek-v4-flash-tensorfold"
    profile = PROFILES[alias]
    assert profile.ordinary_engine
    args = SimpleNamespace(
        model=profile.target,
        _original_alias=alias,
        speculative_config=None,
        no_spec_decode=True,
        mllm=False,
    )
    cli._normalize_speculative_config_or_exit(args)
    cli._require_tensorfold_family_lane_or_exit(args)
    assert args._speculative_config is None
    assert detect_model_config(alias).tool_call_parser == "deepseek_v4_flash"


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize(
    "profile_id,parser",
    [
        ("qwen3.8-27b-tensorfold", "qwen3_coder_xml"),
        ("qwen3.8-27b-tensorfold", None),
        ("deepseek-v4-flash-tensorfold", None),
    ],
)
def test_models_discovery_reports_active_lane_tool_parser(
    monkeypatch, profile_id, parser
):
    from rapid_mlx import _uvicorn

    monkeypatch.delenv("RAPID_MLX_TENSORFOLD_LANES", raising=False)
    monkeypatch.delenv("RAPID_MLX_QUICKSILVER_POOL", raising=False)
    backend = SimpleNamespace(
        _app=SimpleNamespace(tokenizer=object()), close=lambda: None
    )
    provider = SimpleNamespace(
        stream_generate=lambda *a, **k: iter(()), generate=lambda *a, **k: None
    )
    monkeypatch.setattr(
        server, "load_tensorfold_backend", lambda **kwargs: (backend, provider)
    )
    apps = []
    monkeypatch.setattr(_uvicorn, "run_uvicorn", lambda app, **kwargs: apps.append(app))
    server.run_tensorfold_qwen27_server(
        main_model_repo="target",
        main_model_revision=None,
        drafter_repo="draft",
        drafter_revision=None,
        host="127.0.0.1",
        port=0,
        port_explicit=True,
        served_model_name=profile_id,
        default_max_tokens=16,
        cors_origins=[],
        uvicorn_log_level="info",
        profile_id=profile_id,
        tool_call_parser=parser,
        reasoning_parser_name=None,
    )
    with TestClient(apps[0], base_url="http://127.0.0.1") as client:
        response = client.get("/v1/models")
        assert response.status_code == 200
        info = response.json()["data"][0]
        assert info["tool_call_parser"] == parser
        assert info["capabilities"] == [
            "text",
            *(["tools"] if parser else []),
            "experimental",
        ]
        if profile_id == "deepseek-v4-flash-tensorfold":
            rejected = client.post(
                "/v1/chat/completions",
                json={
                    "model": profile_id,
                    "messages": [{"role": "user", "content": "hi"}],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "weather",
                                "parameters": {"type": "object"},
                            },
                        }
                    ],
                },
            )
            assert rejected.status_code == 400
            assert (
                "tool calling is not supported on this tensorfold lane yet"
                in rejected.text.lower()
            )
