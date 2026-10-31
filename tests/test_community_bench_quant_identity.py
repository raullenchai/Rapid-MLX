# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for benchmark runtime identity."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rapid_mlx.community_bench import local_runner as lr
from rapid_mlx.community_bench.acceleration import (
    AccelerationConfigError,
    AccelerationRequest,
    build_serve_args,
    validate_request,
)


def test_alias_wins_over_pulled_variant(monkeypatch, tmp_path):
    import sys

    import huggingface_hub.constants as constants

    monkeypatch.setitem(sys.modules, "llguidance", None)
    monkeypatch.setitem(sys.modules, "mlx_lm", None)
    from rapid_mlx import _mlx_compat

    monkeypatch.setattr(_mlx_compat, "install", lambda: None)
    from rapid_mlx.patches import (
        deepseek_v32_indexer_gate,
        mla_absorbed_verify,
        nemotron_h_ssm,
        qwen3_5_eager_dispatch,
        qwen3_5_norm_shift,
    )

    for module, name in (
        (deepseek_v32_indexer_gate, "install_deepseek_v32_indexer_gate"),
        (mla_absorbed_verify, "install_mla_absorbed_verify"),
        (nemotron_h_ssm, "install_nemotron_h_ssm_fix"),
        (qwen3_5_eager_dispatch, "install_qwen3_5_eager_dispatch"),
        (qwen3_5_norm_shift, "install_qwen3_5_norm_shift_fix"),
    ):
        monkeypatch.setattr(module, name, lambda: None)
    from rapid_mlx import server
    from rapid_mlx._download_gate import persist_pulled_variant
    from rapid_mlx.model_aliases import resolve_profile

    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path))
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
    alias = "lfm2.5-2.6b-4bit"
    repo = "LiquidAI/LFM2.5-2.6B-MLX"
    revision = resolve_profile(alias).revision
    root = tmp_path / "models--LiquidAI--LFM2.5-2.6B-MLX"
    for variant in ("4bit", "8bit"):
        folder = root / "snapshots" / revision / variant
        folder.mkdir(parents=True)
        (folder / "config.json").write_text('{"model_type":"lfm2"}')
        (folder / "model.safetensors").write_bytes(b"stub")
    (root / "refs").mkdir()
    (root / "refs" / "main").write_text(revision)
    assert persist_pulled_variant(repo, "8bit")
    captured = []

    class Engine:
        def __init__(self, **kwargs):
            captured.append(kwargs["model_name"])

        async def start(self):
            pass

        async def stop(self):
            pass

    monkeypatch.setattr(server, "BatchedEngine", Engine)
    args = build_serve_args(alias, AccelerationRequest())
    asyncio.run(lr._load_serving_engine(args.model, None, args))
    assert Path(captured[0]) == root / "snapshots" / revision / "4bit"


def test_pflash_zero_threshold_validates_as_saved():
    from rapid_mlx.community_bench.benchmark_contracts import BenchmarkRunValidator
    from rapid_mlx.community_bench.run_builder import execution_config_digest

    fixture = Path(__file__).parent / "fixtures" / "community_bench_cached_run.json"
    run = json.loads(fixture.read_text())
    run["execution"]["task"]["language"]["pflash"] = {
        "mode": "auto",
        "threshold": 0,
        "keep_ratio_millionths": 1,
    }
    run["execution"]["config_digest"] = execution_config_digest(run["execution"])
    BenchmarkRunValidator().validate(run)


@pytest.mark.parametrize("ratio", [0.0000001, 0.0000006])
def test_pflash_tiny_keep_ratio_refused(ratio):
    request = AccelerationRequest(overrides={"pflash_keep_ratio": ratio})
    with pytest.raises(AccelerationConfigError, match="must be at least 0.000001"):
        validate_request(request, "text_generation", is_fixed_runtime=False)


def test_pflash_smallest_recordable_keep_ratio_accepted():
    request = AccelerationRequest(overrides={"pflash_keep_ratio": 0.000001})
    validate_request(request, "text_generation", is_fixed_runtime=False)


@pytest.mark.parametrize("lane", ["tensorfold_qwen27", "tensorfold_glm53"])
def test_tensorfold_records_runtime_context(lane, tmp_path):
    from rapid_mlx.community_bench.acceleration import dedicated_language
    from rapid_mlx.speculative.serial_benchmark import tensorfold_runtime_handle

    (tmp_path / "config.json").write_text('{"max_position_embeddings":262144}')
    backend = SimpleNamespace(_app=SimpleNamespace(tokenizer=None, context_window=8192))
    handle = tensorfold_runtime_handle(
        backend,
        None,
        lane=lane,
        method="mtp",
        algorithm="test",
        draft_repo=None,
        draft_revision=None,
        draft_path=None,
        target_path=str(tmp_path),
        max_draft_tokens=1,
        lanes=1,
    )
    try:
        assert handle.context_length == 8192
        language = dedicated_language(
            handle,
            context_length=handle.context_length,
            prefill_chunk_size=None,
            draft_identity_digest=None,
        )
        assert language["context_length"] == 8192
    finally:
        handle.executor.shutdown()


def test_same_repo_sidecar_without_root_config(tmp_path):
    from rapid_mlx.community_bench.run_builder import unresolved_model_identity

    snapshot = tmp_path / "models--org--model" / "snapshots" / ("a" * 40)
    for folder in ("4bit", "mtp"):
        (snapshot / folder).mkdir(parents=True)
        (snapshot / folder / "config.json").write_text("{}")
    target = unresolved_model_identity(
        "org/model", "text_generation", "4bit", snapshot_path=str(snapshot / "4bit")
    )
    assert lr._sidecar_in_target_repo(target, "ORG/model", str(snapshot))
    assert not lr._sidecar_in_target_repo(
        target, "org/model", str(snapshot.parent / ("b" * 40))
    )
