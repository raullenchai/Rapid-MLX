import vllm_mlx.request_metrics as request_metrics
from vllm_mlx.request_metrics import RequestRecorder


def test_request_recorder_records_completed_request(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(request_metrics.time, "time", lambda: now[0])

    recorder = RequestRecorder()
    req_id = recorder.start("/v1/chat/completions")

    now[0] += 0.25
    recorder.mark_first_token(req_id)
    recorder.update(req_id, delta_text="hello", generated_tokens=1, prompt_tokens=8)

    now[0] += 0.75
    recorder.finish(
        req_id,
        finish_reason="stop",
        prompt_tokens=8,
        generated_tokens=4,
        engine_gen_tps=12.5,
        engine_ttft=0.25,
    )

    entries = recorder.entries()
    assert recorder.active() is None
    assert len(entries) == 1
    assert entries[0]["surface"] == "/v1/chat/completions"
    assert entries[0]["prompt_tokens"] == 8
    assert entries[0]["generated_tokens"] == 4
    assert entries[0]["decode_tps"] == 12.5
    assert entries[0]["ttft"] == 0.25


def test_request_recorder_active_snapshot(monkeypatch):
    monkeypatch.setattr(request_metrics.time, "time", lambda: 1000.0)

    recorder = RequestRecorder()
    req_id = recorder.start("/v1/completions")
    recorder.update(req_id, delta_text="partial", generated_tokens=2, prompt_tokens=5)

    active = recorder.active()
    assert active is not None
    assert active["request_id"] == req_id
    assert active["surface"] == "/v1/completions"
    assert active["generated_tokens"] == 2
    assert "partial" in active["message_preview"]
