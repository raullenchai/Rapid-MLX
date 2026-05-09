from vllm_mlx.tui import _build_screen, _entry_tokens_per_second


def test_entry_tokens_per_second_prefers_decode_tps():
    assert _entry_tokens_per_second({"decode_tps": 42.5}) == 42.5


def test_entry_tokens_per_second_falls_back_to_elapsed_minus_ttft():
    value = _entry_tokens_per_second(
        {"generated_tokens": 20, "elapsed": 3.0, "ttft": 1.0}
    )
    assert value == 10.0


def test_build_screen_renders_request_metrics():
    screen = _build_screen(
        "http://localhost:8010",
        123,
        1.0,
        {
            "status": "healthy",
            "model_loaded": True,
            "model_name": "local",
            "engine_type": "batched",
        },
        {
            "status": "idle",
            "model": "local",
            "uptime_s": 12,
            "num_running": 0,
            "num_waiting": 0,
            "total_requests_processed": 1,
            "total_prompt_tokens": 10,
            "total_completion_tokens": 20,
            "metal": {},
        },
        {
            "active": None,
            "entries": [
                {
                    "surface": "/v1/chat/completions",
                    "finished_at": 1,
                    "elapsed": 2.0,
                    "prompt_tokens": 10,
                    "generated_tokens": 20,
                    "generation_tps": 10.0,
                    "prompt_tps": 50.0,
                    "finish_reason": "stop",
                }
            ],
        },
        [],
        False,
    )

    assert "Last request" in screen
    assert "Averages so far (1 requests)" in screen
    assert "Recent requests" in screen
    assert "tokens/s" in screen
    assert "10.0" in screen
    assert "n/a" in screen
