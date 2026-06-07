# SPDX-License-Identifier: Apache-2.0
"""Contract pins for ``vllm_mlx.telemetry.emit``.

The emit helpers are the only API call sites should touch. They:
- Refuse to construct a payload when telemetry is disabled.
- Funnel every user-provided string through redaction primitives.
- Catch and swallow non-system exceptions so a telemetry bug cannot
  crash the user's command.
"""

from __future__ import annotations

import importlib

import pytest


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("RAPID_MLX_TELEMETRY", raising=False)

    # Reload state so any caches rebuild under the fresh HOME.
    import vllm_mlx.telemetry.state as state

    importlib.reload(state)

    # Reload emit so it picks up the reloaded state module, and reset
    # its singletons so each test starts clean.
    import vllm_mlx.telemetry.emit as emit

    importlib.reload(emit)
    emit._reset_for_tests()
    return tmp_path


@pytest.fixture
def opted_in(fake_home):
    """Persist a yes-consent so is_enabled() returns True."""
    from vllm_mlx.telemetry.state import record_consent

    record_consent(True, rapid_mlx_version="0.0.0+test")
    return fake_home


@pytest.fixture
def stub_queue(monkeypatch):
    """Replace the singleton queue with an in-memory list capture."""
    from vllm_mlx.telemetry import emit

    captured: list[dict] = []

    class _StubQueue:
        def enqueue(self, payload):
            captured.append(payload)

    monkeypatch.setattr(emit, "get_queue", lambda: _StubQueue())
    return captured


# ---------------------------------------------------------- consent gate


def test_session_start_no_op_when_disabled(fake_home, stub_queue):
    from vllm_mlx.telemetry import emit

    emit.session_start(subcommand="serve")
    assert stub_queue == []


def test_session_end_no_op_when_disabled(fake_home, stub_queue):
    from vllm_mlx.telemetry import emit

    emit.session_end(subcommand="serve", duration_seconds=42)
    assert stub_queue == []


def test_request_no_op_when_disabled(fake_home, stub_queue):
    from vllm_mlx.telemetry import emit

    emit.request(
        endpoint="/v1/chat/completions",
        model_alias="qwen3.5-9b",
        stream=True,
        tool_call_used=False,
        prompt_tokens=100,
        completion_tokens=400,
        ttft_ms=250.0,
        tps=42.0,
        status=200,
    )
    assert stub_queue == []


def test_error_no_op_when_disabled(fake_home, stub_queue):
    from vllm_mlx.telemetry import emit

    emit.error(category="model_load_failure", exc=RuntimeError("x"), phase="startup")
    assert stub_queue == []


# ---------------------------------------------------------- shape when on


def test_session_start_envelope_when_enabled(opted_in, stub_queue):
    from vllm_mlx.telemetry import emit

    emit.session_start(
        subcommand="serve",
        argv=["serve", "qwen3.5-9b", "--host", "0.0.0.0", "--port", "8000"],
        engine="batched",
        models_loaded=["mlx-community/Qwen3.5-9B-4bit"],
    )
    assert len(stub_queue) == 1
    payload = stub_queue[0]

    # Envelope contract.
    assert payload["schema_version"] == 1
    assert payload["event"] == "session_start"
    assert payload["timestamp"].endswith("Z")
    assert payload["platform"]["os"] in {"darwin", "linux", "windows"}
    assert "chip" in payload["platform"]

    # Session payload: flag VALUES must be absent — only NAMES survive
    # redaction. ``0.0.0.0`` and ``8000`` are values, must not appear.
    blob = repr(payload)
    assert "0.0.0.0" not in blob
    assert "8000" not in blob
    # Flag names should be there.
    assert set(payload["session"]["flag_names"]) == {"host", "port"}


def test_session_start_models_loaded_redacted(opted_in, stub_queue):
    """Local paths must collapse to "<local>" — not leak home dirs."""
    from vllm_mlx.telemetry import emit

    emit.session_start(
        subcommand="serve",
        models_loaded=[
            "mlx-community/Qwen3.5-9B-4bit",  # public, passes through
            "/Users/alice/secret-checkout",  # local, redacted
        ],
    )
    loaded = stub_queue[0]["session"]["models_loaded"]
    assert "mlx-community/Qwen3.5-9B-4bit" in loaded
    assert "<local>" in loaded
    assert "alice" not in repr(loaded)


def test_session_start_models_loaded_capped_at_32(opted_in, stub_queue):
    """Don't let a multi-load surface blow up a single payload."""
    from vllm_mlx.telemetry import emit

    emit.session_start(
        subcommand="serve",
        models_loaded=[f"org/model-{i}" for i in range(50)],
    )
    assert len(stub_queue[0]["session"]["models_loaded"]) == 32


def test_request_buckets_not_raw_numbers(opted_in, stub_queue):
    """Bucketed counts only — raw token counts and TTFT must not survive."""
    from vllm_mlx.telemetry import emit

    emit.request(
        endpoint="/v1/chat/completions",
        model_alias="qwen3.5-9b",
        stream=True,
        tool_call_used=False,
        prompt_tokens=137,
        completion_tokens=1729,
        ttft_ms=432.5,
        tps=58.2,
        status=200,
    )
    r = stub_queue[0]["request"]
    # Bucket strings, not raw ints / floats.
    assert isinstance(r["prompt_tokens_bucket"], str)
    assert isinstance(r["ttft_ms_bucket"], str)
    # Specific values must not survive.
    blob = repr(stub_queue[0])
    for raw in ("137", "1729", "432.5", "58.2"):
        assert raw not in blob


def test_error_carries_fingerprint_no_message(opted_in, stub_queue):
    """Crash fingerprint excludes message text and module path."""
    from vllm_mlx.telemetry import emit

    try:
        raise ValueError("/Users/alice/secret.txt: not found")
    except ValueError as exc:
        emit.error(category="model_load_failure", exc=exc, phase="startup")

    err = stub_queue[0]["error"]
    assert len(err["fingerprint"]) == 16
    blob = repr(stub_queue[0])
    assert "/Users/alice/secret.txt" not in blob
    assert "not found" not in blob


# ---------------------------------------------------------- failure suppression


def test_session_start_swallows_internal_bug(opted_in, monkeypatch, stub_queue):
    """An internal redaction bug must not propagate to the caller."""
    from vllm_mlx.telemetry import emit

    def boom(*args, **kwargs):
        raise RuntimeError("synthetic redact failure")

    monkeypatch.setattr(emit, "hash_flag_names", boom)
    # Must not raise:
    emit.session_start(subcommand="serve", argv=["--x"])


def test_emit_does_not_catch_keyboard_interrupt(opted_in, monkeypatch, stub_queue):
    """User intent (Ctrl-C) and SystemExit must propagate."""
    from vllm_mlx.telemetry import emit

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(emit, "platform_info", interrupt)
    with pytest.raises(KeyboardInterrupt):
        emit.session_start(subcommand="serve")
