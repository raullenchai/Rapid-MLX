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

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_mlx.community_bench import local_runner
from vllm_mlx.community_bench.hardware import Hardware, Software
from vllm_mlx.community_bench.runner import BenchResult, BucketResult, RoundResult
from vllm_mlx.community_bench.workspace import LocalRunArchive


def _missing_module(name: str) -> ModuleNotFoundError:
    exc = ModuleNotFoundError(f"No module named {name!r}")
    exc.name = name
    return exc


def _bench_result() -> BenchResult:
    short = [RoundResult(100, 200, 10, prompt_tokens=512, output_tokens=128)] * 5
    long = [RoundResult(50, 150, 20, prompt_tokens=2048, output_tokens=512)] * 5
    return BenchResult(
        short=BucketResult(short),
        long=BucketResult(long),
        peak_ram_mb=4096,
        prompt_hash="unused",
        sampling="greedy",
    )


def test_serving_adapter_preserves_exact_tokens_and_sampling_contract() -> None:
    from vllm_mlx.engine.base import GenerationOutput
    from vllm_mlx.request import SamplingParams

    observed: dict[str, object] = {}

    class Engine:
        async def stream_generate(self, prompt, **kwargs):
            observed["prompt"] = prompt
            observed["kwargs"] = kwargs
            yield GenerationOutput(
                text="a",
                new_text="a",
                tokens=[7],
                prompt_tokens=3,
                completion_tokens=1,
            )
            yield GenerationOutput(
                text="ab",
                new_text="b",
                tokens=[8],
                prompt_tokens=3,
                completion_tokens=2,
                finished=True,
                finish_reason="length",
            )

    async def exercise():
        adapter = local_runner._ServingBenchmarkAdapter(Engine())
        params = SamplingParams(
            max_tokens=2,
            temperature=0.25,
            top_p=0.8,
            top_k=5,
            min_p=0.1,
            repetition_penalty=1.2,
            presence_penalty=0.3,
            frequency_penalty=0.4,
            ignore_eos=True,
            seed=17,
            stop=["END"],
        )
        request_id = await adapter.add_request([1, 2, 3], params)
        return [item async for item in adapter.stream_outputs(request_id, timeout=1)]

    outputs = asyncio.run(exercise())
    assert observed["prompt"] == [1, 2, 3]
    assert observed["kwargs"]["ignore_eos"] is True
    assert observed["kwargs"]["seed"] == 17
    assert observed["kwargs"]["max_tokens"] == 2
    assert observed["kwargs"]["temperature"] == 0.25
    assert observed["kwargs"]["top_p"] == 0.8
    assert observed["kwargs"]["top_k"] == 5
    assert observed["kwargs"]["min_p"] == 0.1
    assert observed["kwargs"]["repetition_penalty"] == 1.2
    assert observed["kwargs"]["presence_penalty"] == 0.3
    assert observed["kwargs"]["frequency_penalty"] == 0.4
    assert observed["kwargs"]["stop"] == ["END"]
    assert outputs[0].new_token_ids == [7]
    assert outputs[1].output_token_ids == [7, 8]
    assert outputs[1].completion_tokens == 2


def test_serving_adapter_context_and_unbounded_stream() -> None:
    from vllm_mlx.engine.base import GenerationOutput
    from vllm_mlx.request import SamplingParams

    class Engine:
        async def stream_generate(self, _prompt, **_kwargs):
            yield GenerationOutput(text="ok", new_text="ok", tokens=[7], finished=True)

    async def exercise():
        async with local_runner._ServingBenchmarkAdapter(Engine()) as adapter:
            request_id = await adapter.add_request(
                "hello", SamplingParams(max_tokens=1)
            )
            return [item async for item in adapter.stream_outputs(request_id)]

    outputs = asyncio.run(exercise())
    assert [item.output_token_ids for item in outputs] == [[7]]


def test_serving_adapter_zero_timeout_aborts() -> None:
    from vllm_mlx.request import SamplingParams

    events: list[str] = []

    class Engine:
        def abort_request(self, request_id):
            events.append(f"abort:{request_id}")

        async def stream_generate(self, _prompt, **_kwargs):
            yield  # pragma: no cover - the zero deadline prevents iteration

    async def exercise() -> None:
        adapter = local_runner._ServingBenchmarkAdapter(Engine())
        request_id = await adapter.add_request([1], SamplingParams(max_tokens=1))
        async for _ in adapter.stream_outputs(request_id, timeout=0):
            pass

    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(exercise())
    assert events[0].startswith("abort:")


def test_serving_adapter_tolerates_stream_close_error() -> None:
    from vllm_mlx.engine.base import GenerationOutput
    from vllm_mlx.request import SamplingParams

    events: list[str] = []

    class Engine:
        async def stream_generate(self, _prompt, **_kwargs):
            try:
                yield GenerationOutput(text="ok", new_text="ok", tokens=[7])
            finally:
                events.append("close")
                raise RuntimeError("close failure")

    async def exercise() -> None:
        adapter = local_runner._ServingBenchmarkAdapter(Engine())
        request_id = await adapter.add_request([1], SamplingParams(max_tokens=1))
        outputs = adapter.stream_outputs(request_id)
        await anext(outputs)
        await outputs.aclose()

    asyncio.run(exercise())
    assert events == ["close"]


def test_serving_adapter_timeout_is_total_request_budget() -> None:
    from vllm_mlx.engine.base import GenerationOutput
    from vllm_mlx.request import SamplingParams

    events: list[str] = []

    class SlowEngine:
        async def abort_request(self, request_id):
            events.append(f"abort:{request_id}")
            raise RuntimeError("cleanup failure must not mask timeout")

        async def stream_generate(self, _prompt, **_kwargs):
            try:
                for token in (7, 8):
                    await asyncio.sleep(0.03)
                    yield GenerationOutput(text="a", new_text="a", tokens=[token])
            finally:
                events.append("closed")

    async def exercise() -> None:
        adapter = local_runner._ServingBenchmarkAdapter(SlowEngine())
        request_id = await adapter.add_request([1, 2, 3], SamplingParams(max_tokens=2))
        async for _item in adapter.stream_outputs(request_id, timeout=0.05):
            pass

    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(exercise())
    assert any(event.startswith("abort:") for event in events)
    assert "closed" in events


def test_serving_adapter_rejects_unsupported_token_stop_ids() -> None:
    from vllm_mlx.request import SamplingParams

    async def exercise() -> None:
        adapter = local_runner._ServingBenchmarkAdapter(object())
        await adapter.add_request(
            [1, 2, 3], SamplingParams(max_tokens=2, stop_token_ids=[9])
        )

    with pytest.raises(ValueError, match="stop token IDs"):
        asyncio.run(exercise())


def test_benchmark_loader_lane_uses_shared_architecture_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_mlx.api import utils

    observed: list[str] = []
    monkeypatch.setattr(
        utils,
        "is_mllm_model",
        lambda target: observed.append(target) or target.startswith("vision/"),
    )

    assert local_runner._uses_serving_benchmark_engine("vision/model") is True
    assert local_runner._uses_serving_benchmark_engine("text/model") is False
    assert observed == ["vision/model", "text/model"]


def test_serving_wrapper_exposes_real_model_context_length() -> None:
    from vllm_mlx.service.helpers import get_model_max_context

    serving_wrapper = SimpleNamespace(
        _model=SimpleNamespace(args=SimpleNamespace(max_position_embeddings=1_048_576)),
        tokenizer=SimpleNamespace(model_max_length=4096),
    )
    assert get_model_max_context(serving_wrapper) == 1_048_576


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (ValueError("Model type glm5_next not supported."), True),
        (_missing_module("mlx_lm.models.future"), True),
        (_missing_module("sentencepiece"), False),
        (ValueError("corrupt tensor shape"), False),
        (ValueError("processor model type future is not supported"), False),
        (OSError("checkpoint unavailable"), False),
    ],
)
def test_serving_fallback_is_limited_to_architecture_rejection(
    exc: BaseException, expected: bool
) -> None:
    assert local_runner._text_loader_needs_serving_fallback(exc) is expected


def test_deepseek_v4_benchmark_family_keeps_native_text_runtime() -> None:
    """DeepSeek V4 checkpoints use Rapid's vendored text model."""
    from vllm_mlx.model_aliases import resolve_profile
    from vllm_mlx.utils.tokenizer import _VENDORED_MODEL_TYPES

    assert "deepseek_v4" in _VENDORED_MODEL_TYPES
    for alias in (
        "deepseek-v4-flash-2bit",
        "deepseek-v4-flash-4bit",
        "deepseek-v4-flash-8bit",
        "deepseek-v4-flash-0731-mxfp4",
    ):
        profile = resolve_profile(alias)
        assert profile is not None
        assert profile.modality == "text"
        assert profile.supports_image_input is False


def test_unavailable_dedicated_text_runtime_fails_before_model_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_mlx.community_bench import workspace
    from vllm_mlx.utils import tokenizer as tokenizer_module

    monkeypatch.setattr(
        workspace,
        "benchmark_runtime_readiness",
        lambda _alias, _task: {
            "status": "unavailable",
            "message": "dedicated runtime is not benchmarkable",
        },
    )
    monkeypatch.setattr(
        tokenizer_module,
        "load_model_with_fallback",
        lambda *_args, **_kwargs: pytest.fail("model loader must not run"),
    )

    with pytest.raises(RuntimeError, match="dedicated runtime is not benchmarkable"):
        asyncio.run(
            local_runner._text_measurements(
                "deepseek-v41-flash-reap-2bit",
                "rapid-mlx/DeepSeek-V4.1-Flash-REAP-2bit-MLX",
            )
        )


def test_text_lane_uses_serving_runtime_and_tolerates_shutdown_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_mlx.community_bench import runner, workspace
    from vllm_mlx.engine import batched

    events: list[str] = []

    class Engine:
        def __init__(self, target, **kwargs):
            events.append(f"init:{target}")
            assert kwargs["force_mllm"] is True
            self.tokenizer = object()
            self._model = SimpleNamespace(
                args=SimpleNamespace(max_position_embeddings=1_048_576)
            )

        async def start(self):
            events.append("start")

        async def stop(self):
            events.append("stop")
            raise RuntimeError("cleanup failure")

    async def standardized(*_args, **kwargs):
        assert kwargs["registered_token_ids"] is True
        return _bench_result()

    monkeypatch.setattr(local_runner, "_uses_serving_benchmark_engine", lambda _: True)
    monkeypatch.setattr(
        workspace,
        "benchmark_runtime_readiness",
        lambda *_args: {"status": "ready"},
    )
    monkeypatch.setattr(batched, "BatchedEngine", Engine)
    monkeypatch.setattr(runner, "run_standardized_bench", standardized)
    monkeypatch.setattr(
        local_runner, "unresolved_model_identity", lambda *_args, **_kwargs: {}
    )

    measurements, context_length, _identity = asyncio.run(
        local_runner._text_measurements("glm5.3-flash-4bit", "vendor/glm")
    )

    assert len(measurements) == 10
    assert context_length == 1_048_576
    assert events == ["init:vendor/glm", "start", "stop"]


def test_text_lane_reaps_failed_serving_start(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_mlx.community_bench import workspace
    from vllm_mlx.engine import batched

    events: list[str] = []

    class Engine:
        def __init__(self, *_args, **_kwargs):
            pass

        async def start(self):
            events.append("start")
            raise RuntimeError("startup failed")

        async def stop(self):
            events.append("stop")
            raise RuntimeError("cleanup failed")

    monkeypatch.setattr(local_runner, "_uses_serving_benchmark_engine", lambda _: True)
    monkeypatch.setattr(
        workspace,
        "benchmark_runtime_readiness",
        lambda *_args: {"status": "ready"},
    )
    monkeypatch.setattr(batched, "BatchedEngine", Engine)

    with pytest.raises(RuntimeError, match="startup failed"):
        asyncio.run(local_runner._text_measurements("glm", "vendor/glm"))
    assert events == ["start", "stop"]


def test_text_lane_falls_back_only_for_exact_architecture_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_mlx.community_bench import runner, workspace
    from vllm_mlx.engine import batched
    from vllm_mlx.utils import tokenizer as tokenizer_module

    class Engine:
        def __init__(self, *_args, **_kwargs):
            self.tokenizer = object()
            self._model = SimpleNamespace(
                args=SimpleNamespace(max_position_embeddings=32768)
            )

        async def start(self):
            return None

        async def stop(self):
            return None

    async def standardized(*_args, **_kwargs):
        return _bench_result()

    monkeypatch.setattr(local_runner, "_uses_serving_benchmark_engine", lambda _: False)
    monkeypatch.setattr(
        workspace,
        "benchmark_runtime_readiness",
        lambda *_args: {"status": "ready"},
    )
    monkeypatch.setattr(batched, "BatchedEngine", Engine)
    monkeypatch.setattr(runner, "run_standardized_bench", standardized)
    monkeypatch.setattr(
        local_runner, "unresolved_model_identity", lambda *_args, **_kwargs: {}
    )
    monkeypatch.setattr(
        tokenizer_module,
        "load_model_with_fallback",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("Model type future_text not supported.")
        ),
    )

    measurements, context_length, _identity = asyncio.run(
        local_runner._text_measurements("future", "vendor/future")
    )
    assert len(measurements) == 10
    assert context_length == 32768

    monkeypatch.setattr(
        tokenizer_module,
        "load_model_with_fallback",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("corrupt tensor shape")
        ),
    )
    with pytest.raises(ValueError, match="corrupt tensor shape"):
        asyncio.run(local_runner._text_measurements("future", "vendor/future"))


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
            nonlocal engine_open
            engine_open = True
            return self

        async def __aexit__(self, *args) -> None:
            nonlocal engine_open
            engine_open = False

    # The real ``_text_measurements`` hook must snapshot the run conditions
    # after the last measurement and while the engine is still resident.
    events: list[str] = []
    engine_open = False

    def probe() -> dict:
        events.append(
            f"probe(engine_open={engine_open}, shutdowns={len(shutdown_calls)})"
        )
        return {
            "power_source": "ac",
            "low_power_mode": False,
            "thermal_state": "serious" if events[1:] else "nominal",
            "memory_pressure": "normal",
            "available_memory_mib": 1024,
        }

    monkeypatch.setattr(local_runner, "run_conditions", probe)

    async def standardized(
        engine, tokenizer, *, sampling: str, registered_token_ids: bool, on_round=None
    ) -> BenchResult:
        assert sampling == "greedy"
        assert registered_token_ids is True
        events.append("bench")
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
        lambda repo_id, **_: (
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
    assert events == [
        "probe(engine_open=False, shutdowns=0)",
        "bench",
        "probe(engine_open=True, shutdowns=0)",
    ]
    assert run["machine"]["conditions_before"]["thermal_state"] == "nominal"
    assert run["machine"]["conditions_after"]["thermal_state"] == "serious"


def test_text_lane_measures_the_catalog_repo_id_not_a_same_named_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A same-named local directory must not be measured under the catalog
    identity (codex on #3147)."""
    from vllm_mlx import model_aliases
    from vllm_mlx.utils import tokenizer as tokenizer_module

    targets: list[str] = []

    def fake_loader(target, **kwargs):
        assert kwargs.get("return_source") is True
        targets.append(target)
        raise RuntimeError("stop here")

    monkeypatch.setattr(tokenizer_module, "load_model_with_fallback", fake_loader)
    monkeypatch.setattr(model_aliases, "resolve_model", lambda name: "/tmp/same-dir")
    with pytest.raises(RuntimeError, match="stop here"):
        asyncio.run(
            local_runner._text_measurements(
                "qwen3.5-4b-4bit", "mlx-community/Qwen3.5-4B-MLX-4bit"
            )
        )
    assert targets == ["mlx-community/Qwen3.5-4B-MLX-4bit"]
