# SPDX-License-Identifier: Apache-2.0
"""Linux-lane lifecycle coverage for the Community Benchmark text executor.

``local_runner._text_measurements`` imports ``vllm_mlx.engine_core`` and the
tokenizer loader, so the ordinary no-MLX lane cannot execute it. The inert
MLX seam installed by this folder's conftest permits importing those engine
modules while every tensor operation stays faked — no model is ever loaded.
The measurement-conversion contract itself is identical to the Apple-lane
``requires_mlx`` twin in ``tests/test_community_benchmark_workspace.py``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_mlx.community_bench import local_runner
from vllm_mlx.community_bench.hardware import Hardware, Software
from vllm_mlx.community_bench.runner import BenchResult, BucketResult, RoundResult
from vllm_mlx.community_bench.workspace import LocalRunArchive


def test_text_lane_converts_engine_result_and_reaps_executor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from vllm_mlx import engine_core
    from vllm_mlx.community_bench import runner
    from vllm_mlx.utils import tokenizer as tokenizer_module

    archive = LocalRunArchive(tmp_path)
    shutdown_calls: list[tuple[bool, bool]] = []
    executor_type = local_runner.concurrent.futures.ThreadPoolExecutor

    class RecordingExecutor:
        def __init__(self, *args, **kwargs) -> None:
            self.inner = executor_type(*args, **kwargs)

        def submit(self, *args, **kwargs):
            return self.inner.submit(*args, **kwargs)

        def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
            shutdown_calls.append((wait, cancel_futures))
            self.inner.shutdown(wait=wait, cancel_futures=cancel_futures)

    monkeypatch.setattr(
        local_runner.concurrent.futures, "ThreadPoolExecutor", RecordingExecutor
    )
    monkeypatch.setattr(
        local_runner,
        "plan_for_alias",
        lambda alias: {
            "model": {
                "alias": alias,
                "repo_id": "mlx-community/example-text-model",
                "task_type": "text_generation",
            }
        },
    )
    monkeypatch.setattr(
        local_runner,
        "collect",
        lambda: (
            Hardware("Apple M4 Pro", 24, 12, 16),
            Software("15.6", "0.13.2", "0.32.1", "3.12.1"),
        ),
    )

    class Engine:
        def __init__(self, model, tokenizer, *args, **kwargs) -> None:
            self.engine = SimpleNamespace(_model=model, tokenizer=tokenizer)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args) -> None:
            pass

    async def standardized(
        engine, tokenizer, *, sampling: str, registered_token_ids: bool
    ) -> BenchResult:
        assert sampling == "greedy"
        assert registered_token_ids is True
        short = [RoundResult(100, 200, 10, prompt_tokens=512, output_tokens=128)] * 5
        long = [RoundResult(50, 150, 20, prompt_tokens=2048, output_tokens=512)] * 5
        return BenchResult(
            short=BucketResult(short),
            long=BucketResult(long),
            peak_ram_mb=4096,
            prompt_hash="unused",
            sampling="greedy",
        )

    monkeypatch.setattr(engine_core, "AsyncEngineCore", Engine)
    monkeypatch.setattr(engine_core, "_init_mlx_step_thread", lambda: None)
    monkeypatch.setattr(
        tokenizer_module,
        "load_model_with_fallback",
        lambda repo_id: (
            SimpleNamespace(args=SimpleNamespace(max_position_embeddings=32768)),
            object(),
        ),
    )
    monkeypatch.setattr(runner, "run_standardized_bench", standardized)

    run = local_runner.run_local("example-text", archive=archive)

    assert len(run["measurements"]) == 10
    assert [(row["case_id"], row["round_index"]) for row in run["measurements"]] == [
        *(("pp512-tg128", index) for index in range(1, 6)),
        *(("pp2048-tg512", index) for index in range(1, 6)),
    ]
    assert run["measurements"][0] == {
        "case_id": "pp512-tg128",
        "round_index": 1,
        "total_duration_ms": 1280.0,
        "peak_active_memory_mib": 4096,
        "completed": True,
        "prompt_tokens": 512,
        "output_tokens": 128,
        "ttft_ms": 10,
        "decode_duration_ms": 1270.0,
    }
    assert run["execution"]["task"]["language"]["context_length"] == 32768
    assert archive.get(run["run_id"]) == run
    assert shutdown_calls == [(True, True)]
