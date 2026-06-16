# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``rapid-mlx bench <model> --tier smoke``.

Smoke-tier is the cheapest tier: boot the model server, send one
prompt ("Hello, what is 2+2?"), assert the response contains "4",
print PASS/FAIL + TTFT + boot time. These tests stub out the HTTP
client and the doctor.server boot helper so the tier code runs
end-to-end without ever loading a model.
"""

from __future__ import annotations

import contextlib
from unittest.mock import patch

import pytest

from vllm_mlx.bench.tier_runner import (
    HARNESS_PROFILES,
    TierResult,
    _find_free_port_in_range,
    _resolve_base_url,
    run_tier,
)


class _FakeStreamResp:
    """httpx.Client.stream context manager that yields canned SSE lines."""

    def __init__(self, lines: list[str]):
        self._lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def raise_for_status(self):
        return None

    def iter_lines(self):
        yield from self._lines


class _FakeClient:
    """Minimal httpx.Client stand-in for the smoke-tier code path."""

    def __init__(self, *, models_payload: dict, stream_lines: list[str]):
        self._models_payload = models_payload
        self._stream_lines = stream_lines

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get(self, url: str):
        class _R:
            def __init__(self, payload):
                self._payload = payload

            def raise_for_status(self):
                return None

            def json(self):
                return self._payload

        return _R(self._models_payload)

    def stream(self, method: str, url: str, json=None):
        return _FakeStreamResp(self._stream_lines)


@contextlib.contextmanager
def _fake_serve(model, port=None, **kwargs):
    """Drop-in for vllm_mlx.doctor.server.serve — no subprocess."""
    yield {"base_url": f"http://127.0.0.1:{port}/v1", "port": port}


@pytest.fixture
def patch_smoke_environment():
    """Stub the boot path + HTTP client so the tier runs in-process."""

    # Pretend port 8500 is always free.
    def _free_port(lo, hi):
        return 8500

    # Canned SSE stream that produces "2+2 equals 4."
    stream_lines = [
        'data: {"choices":[{"delta":{"content":"2+2"}}]}',
        'data: {"choices":[{"delta":{"content":" equals 4."}}]}',
        "data: [DONE]",
    ]
    models_payload = {"data": [{"id": "test-model"}]}

    def _client_factory(*args, **kwargs):
        return _FakeClient(
            models_payload=models_payload, stream_lines=stream_lines
        )

    with (
        patch(
            "vllm_mlx.bench.tier_runner._find_free_port_in_range",
            side_effect=_free_port,
        ),
        patch("vllm_mlx.doctor.server.serve", _fake_serve),
        patch("httpx.Client", _client_factory),
    ):
        yield


def test_smoke_happy_path_returns_zero(patch_smoke_environment, capsys):
    """Smoke tier with a "4" in the response exits 0 and prints PASS."""
    rc = run_tier(model="qwen3.5-4b-4bit", tier="smoke")
    assert rc == 0, "smoke tier with valid response should exit 0"

    captured = capsys.readouterr()
    assert "[PASS] tier=smoke" in captured.out
    assert "OK: 1/1 tiers passed" in captured.out


def test_smoke_fail_when_no_four_in_response(capsys):
    """Smoke tier fails (rc=1) when response doesn't contain "4"."""

    def _free_port(lo, hi):
        return 8500

    # Response says "I don't know" — no "4" — should FAIL.
    stream_lines = [
        'data: {"choices":[{"delta":{"content":"I don\\u0027t"}}]}',
        'data: {"choices":[{"delta":{"content":" know"}}]}',
        "data: [DONE]",
    ]
    models_payload = {"data": [{"id": "test-model"}]}

    def _client_factory(*args, **kwargs):
        return _FakeClient(
            models_payload=models_payload, stream_lines=stream_lines
        )

    with (
        patch(
            "vllm_mlx.bench.tier_runner._find_free_port_in_range",
            side_effect=_free_port,
        ),
        patch("vllm_mlx.doctor.server.serve", _fake_serve),
        patch("httpx.Client", _client_factory),
    ):
        rc = run_tier(model="qwen3.5-4b-4bit", tier="smoke")

    assert rc == 1, "smoke tier without '4' in response should exit 1"
    captured = capsys.readouterr()
    assert "[FAIL] tier=smoke" in captured.out


def test_smoke_output_shape_contains_required_fields(
    patch_smoke_environment, capsys
):
    """Smoke output must contain model name, tier name, duration, and TTFT."""
    run_tier(model="qwen3.5-4b-4bit", tier="smoke")

    captured = capsys.readouterr()
    # Header + per-tier marker + finalize.
    assert "tier=smoke" in captured.out
    assert "model=qwen3.5-4b-4bit" in captured.out
    # Duration appears on the per-tier marker line.
    assert "duration=" in captured.out
    # TTFT is logged in the detail block.
    assert "ttft=" in captured.out


def test_smoke_rejects_unknown_tier_name(capsys):
    """Bogus tier name should exit with code 2 and a clear error."""
    rc = run_tier(model="qwen3.5-4b-4bit", tier="bogus")
    assert rc == 2
    captured = capsys.readouterr()
    assert "unknown tier" in captured.err.lower()


def test_resolve_base_url_strips_v1_suffix():
    """--base-url accepts both http://host:port and http://host:port/v1."""
    assert _resolve_base_url("http://localhost:8000") == ("localhost", 8000)
    assert _resolve_base_url("http://localhost:8000/v1") == ("localhost", 8000)
    assert _resolve_base_url(None) is None


def test_harness_profiles_list_has_five_in_correct_order():
    """The 5 first-class harnesses must be in the documented order."""
    assert HARNESS_PROFILES == ("codex", "opencode", "hermes", "aider", "langchain")


def test_find_free_port_in_range_uses_band():
    """Free port should land in 8500-8599 or be a positive int fallback."""
    port = _find_free_port_in_range(8500, 8599)
    assert isinstance(port, int) and port > 0


def test_tier_result_dataclass_round_trip():
    """TierResult holds the fields the dispatcher expects."""
    r = TierResult(name="smoke", passed=True, duration_s=1.5, detail="x")
    assert r.name == "smoke"
    assert r.passed is True
    assert r.duration_s == 1.5
    assert r.detail == "x"
