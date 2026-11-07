"""Request-local execution facts, independent of configured defaults."""

from types import SimpleNamespace


def test_terminal_metrics_preserve_execution_reference():
    from rapid_mlx.service.helpers import _build_response_metrics

    output = SimpleNamespace(
        timing_metrics={"time_to_first_token_ms": 12.0, "mean_itl_ms": 20.0},
        execution_reference={
            "version": 1,
            "status": "unknown",
            "repo_id": None,
            "method": "none",
            "assistant_source": None,
            "backend": "ordinary",
            "max_draft_tokens": None,
            "observed_max_draft_tokens": None,
            "config_key": "none",
            "completion_tokens": 4,
        },
    )
    metrics = _build_response_metrics(output)
    assert metrics.execution_reference.method == "none"
    assert metrics.execution_reference.repo_id is None
    assert metrics.mean_itl_ms == 20.0


def test_catalog_mapping_requires_selected_trusted_snapshot(tmp_path, monkeypatch):
    import huggingface_hub.constants

    from rapid_mlx.execution_reference import freeze_load_receipt

    repo = "mlx-community/Qwen3.5-35B-A3B-8bit"
    hub = tmp_path / "hub"
    selected = hub / ("models--" + repo.replace("/", "--")) / "snapshots" / ("a" * 40)
    selected.mkdir(parents=True)
    head = selected / "mtp"
    head.mkdir()
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(hub))
    receipt = freeze_load_receipt(str(selected), head_container=str(head))
    assert receipt.repo_id == repo
    assert receipt.assistant_source == "embedded_head"
    # A lookalike copied outside the trusted cache is still a local artifact.
    copied = tmp_path / "private" / selected.name
    copied.mkdir(parents=True)
    assert freeze_load_receipt(str(copied)).repo_id is None
    assert freeze_load_receipt("not-a-real-alias").repo_id is None
    assert freeze_load_receipt(str(selected / "missing")).repo_id is None


def test_observation_requires_actual_work_and_keeps_mtp_parking_consistent():
    from rapid_mlx.execution_reference import (
        LoadReceipt,
        observe_mtp_steps,
        observe_plain_steps,
        terminal_reference,
    )
    from rapid_mlx.spec_decode.mtp.accept_counter import MTPAcceptCounter

    repo = "mlx-community/Qwen3.5-35B-A3B-8bit"
    request = SimpleNamespace(_mtp_accept_counter=MTPAcceptCounter())
    owners, uid_map = {}, {}
    gb = SimpleNamespace(uids=[7], _step=lambda: ([11], [0.0]))
    receipt = LoadReceipt(repo, "embedded_head")
    state = observe_plain_steps(
        SimpleNamespace(_generation_batch=gb), owners, uid_map, receipt
    )
    state["method"] = "mtp"  # Configuration alone supplies no observation.
    owners["request"] = request
    uid_map[7] = "request"
    plain_step = gb._step
    phase = "bootstrap"

    def executing_step():
        if phase == "verify":
            request._mtp_accept_counter.record_verify(1, 0)
        if phase == "fallback":
            return plain_step()
        return [12], [0.0]

    gb._step = observe_mtp_steps(executing_step, gb, owners, uid_map, receipt, 2)
    gb._step()
    assert terminal_reference(request, 1, runtime_method="mtp") is None
    phase = "verify"
    gb._step()
    reference = terminal_reference(request, 2, runtime_method="mtp")
    assert reference["status"] == "known"
    assert reference["config_key"] == "mtp:embedded_head:k=2:ordinary"
    assert reference["max_draft_tokens"] == 2
    assert reference["observed_max_draft_tokens"] == 1
    phase = "park"
    gb._step()
    assert terminal_reference(request, 3, runtime_method="mtp")["status"] == "known"
    phase = "fallback"
    gb._step()
    mixed = terminal_reference(request, 4, runtime_method="mtp")
    assert mixed["status"] == mixed["method"] == "mixed"
    assert mixed["config_key"] is None


def test_configured_mtp_plain_fallback_needs_positive_plain_step():
    from rapid_mlx.execution_reference import (
        LoadReceipt,
        observe_plain_steps,
        terminal_reference,
    )

    request = SimpleNamespace(configured_method="mtp")
    gb = SimpleNamespace(uids=[7], _step=lambda: ([11], [0.0]))
    state = observe_plain_steps(
        SimpleNamespace(_generation_batch=gb),
        {"r": request},
        {7: "r"},
        LoadReceipt("mlx-community/Qwen3.5-35B-A3B-8bit"),
    )
    state["method"] = "mtp"
    assert terminal_reference(request, 0, runtime_method="mtp") is None
    gb._step()
    reference = terminal_reference(request, 1, runtime_method="mtp")
    assert reference["method"] == "none"
    assert reference["status"] == "known"
    assert reference["config_key"] == "none"


def test_unattributed_verifier_cannot_become_plain_or_known():
    from rapid_mlx.execution_reference import (
        LoadReceipt,
        observe_plain_steps,
        terminal_reference,
    )
    from rapid_mlx.spec_decode.mtp.accept_counter import MTPAcceptCounter

    request = SimpleNamespace(_mtp_accept_counter=MTPAcceptCounter())
    request._mtp_accept_counter.record_verify(2, 0)
    gb = SimpleNamespace(uids=[7], _step=lambda: ([11], [0.0]))
    state = observe_plain_steps(
        SimpleNamespace(_generation_batch=gb), {"r": request}, {7: "r"}, LoadReceipt()
    )
    state["method"] = "none"
    gb._step()
    value = terminal_reference(request, 1, runtime_method=None)
    assert value["method"] == value["status"] == "unknown"
    assert value["repo_id"] is None
    assert value["config_key"] is None


def test_invalid_private_metadata_is_omitted_without_losing_timing():
    from rapid_mlx.service.helpers import _build_response_metrics

    output = SimpleNamespace(
        timing_metrics={"time_to_first_token_ms": 1.0},
        execution_reference={
            "version": 1,
            "status": "known",
            "repo_id": "/private/user/model",
            "method": "none",
            "backend": "ordinary",
            "config_key": "none",
            "completion_tokens": 2,
        },
    )
    envelope = _build_response_metrics(output)
    assert envelope.execution_reference is None
    assert envelope.time_to_first_token_ms == 1.0
    assert "/private" not in envelope.model_dump_json()
    assert _build_response_metrics(output, include_timing=False) is None


def test_execution_reference_rejects_inconsistent_key_and_extra_fields():
    import pytest
    from pydantic import ValidationError

    from rapid_mlx.api.models import ExecutionReferenceMetrics

    value = dict(
        version=1,
        status="known",
        repo_id="mlx-community/Qwen3.5-35B-A3B-8bit",
        method="none",
        backend="ordinary",
        config_key="none",
        completion_tokens=2,
    )
    assert ExecutionReferenceMetrics.model_validate(value).config_key == "none"
    for invalid in (
        {**value, "config_key": "mtp:embedded_head:k=3:ordinary"},
        {**value, "private_path": "/Users/example"},
        {**value, "method": "mixed"},
        {**value, "completion_tokens": True},
        {**value, "max_draft_tokens": 3},
    ):
        with pytest.raises(ValidationError):
            ExecutionReferenceMetrics.model_validate(invalid)


def test_multi_generation_and_retry_drop_single_generation_reference():
    from rapid_mlx.engine.base import GenerationOutput
    from rapid_mlx.service.helpers import (
        _aggregate_generation_attempts,
        _merge_response_metrics,
    )

    reference = dict(
        version=1,
        status="unknown",
        repo_id=None,
        method="none",
        backend="ordinary",
        config_key="none",
        completion_tokens=2,
    )
    output = GenerationOutput(
        text="ok",
        completion_tokens=2,
        timing_metrics={"time_to_first_token_ms": 10.0, "mean_itl_ms": 20.0},
        execution_reference=reference,
    )
    assert _merge_response_metrics([output]).execution_reference is not None
    assert _merge_response_metrics([output, output]) is None
    aggregated = _aggregate_generation_attempts(output, output)
    assert aggregated.completion_tokens == 4
    assert aggregated.execution_reference is None
    assert aggregated.timing_metrics is None


def test_request_buffer_preserves_terminal_reference():
    import pytest

    pytest.importorskip("mlx.core")
    from rapid_mlx.engine_core import EngineCore
    from rapid_mlx.request import RequestOutput

    reference = dict(
        version=1,
        status="unknown",
        repo_id=None,
        method="none",
        backend="ordinary",
        config_key="none",
        completion_tokens=2,
    )
    output = RequestOutput(
        request_id="r",
        output_text="ok",
        finished=True,
        finish_reason="stop",
        completion_tokens=2,
        execution_reference=reference,
    )
    merged = EngineCore._merge_stream_buffer(None, output)
    assert merged.execution_reference == reference


def test_subfolder_variant_cannot_claim_repo_only_cohort(tmp_path, monkeypatch):
    import huggingface_hub.constants

    from rapid_mlx.execution_reference import (
        freeze_load_receipt,
        observe_plain_steps,
        terminal_reference,
    )

    repo = "LiquidAI/LFM2.5-2.6B-MLX"
    hub = tmp_path / "hub"
    snapshot = hub / ("models--" + repo.replace("/", "--")) / "snapshots" / ("b" * 40)
    selected = snapshot / "4bit"
    selected.mkdir(parents=True)
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(hub))
    receipt = freeze_load_receipt(str(selected))
    assert receipt.repo_id == repo
    assert not receipt.comparable
    request = SimpleNamespace()
    gb = SimpleNamespace(uids=[7], _step=lambda: ([11], [0.0]))
    state = observe_plain_steps(
        SimpleNamespace(_generation_batch=gb), {"r": request}, {7: "r"}, receipt
    )
    state["method"] = "none"
    gb._step()
    reference = terminal_reference(request, 1, runtime_method=None)
    assert reference["status"] == "unknown"
    assert reference["config_key"] is None
    # A sibling directory not selected by the bundled artifact entry is not
    # attributed to that entry just because it shares a repository root.
    sibling = snapshot / "unregistered-quant"
    sibling.mkdir()
    assert freeze_load_receipt(str(sibling)).repo_id is None


def test_nonstream_collector_keeps_terminal_reference():
    from rapid_mlx.output_collector import RequestOutputCollector
    from rapid_mlx.request import RequestOutput

    reference = dict(
        version=1,
        status="unknown",
        method="none",
        backend="ordinary",
        config_key="none",
        completion_tokens=2,
    )
    first = RequestOutput(
        request_id="r", new_text="a", new_token_ids=[1], completion_tokens=1
    )
    terminal = RequestOutput(
        request_id="r",
        new_text="b",
        new_token_ids=[2],
        finished=True,
        finish_reason="stop",
        completion_tokens=2,
        execution_reference=reference,
    )
    merged = RequestOutputCollector()._merge_outputs(first, terminal)
    assert merged.output_text == ""
    assert merged.new_text == "ab"
    assert merged.execution_reference == reference


def test_buffered_replay_keeps_reference_on_finish_before_usage():
    import asyncio
    import json

    from rapid_mlx.api.models import (
        AssistantMessage,
        ChatCompletionChoice,
        ChatCompletionRequest,
        ChatCompletionResponse,
        PerRequestMetrics,
        Usage,
    )
    from rapid_mlx.routes.chat import _stream_buffered_chat_response

    reference = dict(
        version=1,
        status="unknown",
        method="none",
        backend="ordinary",
        config_key="none",
        completion_tokens=2,
    )

    async def run(finish):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "ping",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        request = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
            stream_options={"include_usage": True},
            tools=tools if finish == "tool_calls" else None,
        )
        message = AssistantMessage(content="ok")
        if finish == "tool_calls":
            message = AssistantMessage(
                content=None,
                tool_calls=[
                    {
                        "id": "call_x",
                        "type": "function",
                        "function": {"name": "ping", "arguments": "{}"},
                    }
                ],
            )
        response = ChatCompletionResponse(
            model="test",
            choices=[ChatCompletionChoice(message=message, finish_reason=finish)],
            usage=Usage(prompt_tokens=3, completion_tokens=2, total_tokens=5),
            metrics=PerRequestMetrics(
                execution_reference=reference, time_to_first_token_ms=10, mean_itl_ms=20
            ),
        )
        events = [
            event async for event in _stream_buffered_chat_response(response, request)
        ]
        chunks = [
            json.loads(event[6:])
            for event in events
            if event.startswith("data: ") and not event.startswith("data: [DONE]")
        ]
        terminal_index = next(
            i
            for i, c in enumerate(chunks)
            if c.get("choices") and c["choices"][0].get("finish_reason")
        )
        usage_index = next(
            i for i, c in enumerate(chunks) if c.get("choices") == [] and c.get("usage")
        )
        assert terminal_index < usage_index
        metrics = chunks[terminal_index]["metrics"]
        assert metrics["execution_reference"]["completion_tokens"] == 2
        assert metrics["execution_reference"]["status"] == "unknown"
        assert metrics["mean_itl_ms"] == 20
        assert "metrics" not in chunks[usage_index]

    for finish in ("stop", "tool_calls"):
        asyncio.run(run(finish))


def test_constructed_invalid_reference_is_revalidated_before_publication():
    from rapid_mlx.api.models import ExecutionReferenceMetrics
    from rapid_mlx.service.helpers import _build_response_metrics

    invalid = ExecutionReferenceMetrics.model_construct(
        version=1,
        status="known",
        repo_id="/private/model",
        method="none",
        backend="ordinary",
        config_key="none",
        completion_tokens=2,
    )
    metrics = _build_response_metrics(
        SimpleNamespace(
            execution_reference=invalid, timing_metrics={"time_to_first_token_ms": 1.0}
        )
    )
    assert metrics.execution_reference is None
    assert "/private" not in metrics.model_dump_json()


def test_actual_mtp_installer_freezes_family_ceiling_after_bootstrap(monkeypatch):
    import pytest

    mx = pytest.importorskip("mlx.core")
    from rapid_mlx.execution_reference import LoadReceipt, terminal_reference
    from rapid_mlx.scheduler import _install_mtp_vendored
    from rapid_mlx.spec_decode.mtp import generator as generator_module
    from tests.test_mtp_cli_wiring import _make_batch_gen_with_gb, _StubModel

    captured = {}

    def verifier(*args, **kwargs):
        captured.update(kwargs)
        kwargs["accept_counter"].record_verify(1, 0)
        yield 201, mx.array([0.0]), False

    monkeypatch.setattr(generator_module, "mtp_generate_step", verifier)
    bg, gb = _make_batch_gen_with_gb()
    gb.uids = [41]
    gb._next_tokens = mx.array([500], dtype=mx.uint32)
    gb._next_logprobs = [mx.array([0.0])]
    model = _StubModel()
    model.mtp_max_speculative_tokens = 2
    request = SimpleNamespace(sampling_params=SimpleNamespace(temperature=0.0))
    assert _install_mtp_vendored(
        bg,
        model,
        requests={"r": request},
        uid_to_request_id={41: "r"},
        max_k=3,
        disable_auto_k=True,
        execution_receipt=LoadReceipt(
            "mlx-community/Qwen3.5-35B-A3B-8bit", "embedded_head"
        ),
    )
    gb._step()  # The already-sampled first token does not attest a method.
    assert terminal_reference(request, 1, runtime_method="mtp") is None
    gb._step()
    assert captured["max_k"] == 2
    reference = terminal_reference(request, 2, runtime_method="mtp")
    assert reference["status"] == "known"
    assert reference["config_key"] == "mtp:embedded_head:k=2:ordinary"
    assert reference["observed_max_draft_tokens"] == 1
