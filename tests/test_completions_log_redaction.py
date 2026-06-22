# SPDX-License-Identifier: Apache-2.0
"""R-05 — ``/v1/completions`` must not leak the request body at INFO level.

PyPI 0.8.6 dogfood (liang-r2 L2-001): a request to ``/v1/completions``
with prompt ``"my secret password is hunter2"`` produced a server log
line at INFO level containing
``prompt_preview="my secret password is hunter2"``. The chat /
anthropic / responses lanes already split metadata at INFO and content
preview at DEBUG; legacy completions skipped the parity, so anyone with
log-aggregator read access could harvest credentials posted to that
route.

This module pins the fix:

* INFO records for ``/v1/completions`` contain only metadata (counts,
  lengths, model id, sampling knobs) — no prompt body.
* The body preview, when emitted, lands at DEBUG (operator opt-in via
  ``--log-level debug``), matching the chat lane's
  ``logger.debug("[REQUEST] last user message preview: ...")``.
* The metadata still includes ``prompt_chars`` (length) so operators
  can still detect oversized prompts without reading the body.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

SENTINEL = "my secret password is hunter2"


@pytest.fixture()
def client_with_completions_route(monkeypatch):
    """Wire just enough of the production stack to hit
    ``create_completion`` and observe its log output.

    We monkeypatch the engine + admission helpers so the route does
    NOT try to load a real model; the goal is purely to drive the
    request-logging path with caplog.
    """
    from vllm_mlx.routes import completions as completions_mod

    # Fake engine that returns a deterministic empty completion. The
    # body-redaction log line runs BEFORE the engine is touched, but
    # the helpers below still need plausible return values so the
    # response path completes without error.
    fake_engine = MagicMock()
    fake_engine.generate = AsyncMock(
        return_value=MagicMock(
            text="",
            finish_reason="stop",
            completion_tokens=0,
            prompt_tokens=0,
            cached_tokens=0,
        )
    )

    monkeypatch.setattr(completions_mod, "get_engine", lambda _name: fake_engine)
    monkeypatch.setattr(completions_mod, "_check_admission_or_503", lambda _eng: None)
    monkeypatch.setattr(
        completions_mod, "_release_admission_unless_committed", lambda *a, **kw: None
    )
    monkeypatch.setattr(
        completions_mod,
        "enforce_context_length_for_prompt",
        lambda *a, **kw: None,
    )
    monkeypatch.setattr(
        completions_mod, "_validate_model_name", lambda _m: None
    )
    monkeypatch.setattr(
        completions_mod, "_resolve_model_name", lambda m: m
    )
    monkeypatch.setattr(
        completions_mod, "_resolve_max_tokens", lambda m: m or 16
    )
    monkeypatch.setattr(
        completions_mod, "_resolve_temperature", lambda t: t
    )
    monkeypatch.setattr(completions_mod, "_resolve_top_p", lambda p: p)
    monkeypatch.setattr(
        completions_mod, "build_extended_sampling_kwargs", lambda _r: {}
    )

    async def _passthrough(coro, *_a, **_kw):
        return await coro

    monkeypatch.setattr(completions_mod, "_wait_with_disconnect", _passthrough)

    # No-op auth + rate limit so the request reaches the body of the
    # handler without hitting either middleware.
    with patch(
        "vllm_mlx.middleware.auth.verify_api_key", new=lambda *a, **kw: None
    ), patch(
        "vllm_mlx.middleware.auth.check_rate_limit", new=lambda *a, **kw: None
    ):
        app = FastAPI()
        app.include_router(completions_mod.router)
        yield TestClient(app)


def _records_at_or_above(caplog, level: int) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.levelno >= level]


def test_info_log_does_not_leak_prompt_body(
    client_with_completions_route, caplog
):
    """R-05 repro: send a prompt containing a 'secret' sentinel,
    assert it does NOT appear in any INFO-or-higher log record."""
    # Capture every level so we can also positively assert the DEBUG
    # preview behaviour later.
    # Note: the runtime log-namespace rebrand (vllm_mlx -> rapid_mlx)
    # rewrites the record.name AFTER emit, but caplog filters by the
    # logger we configure — set both to be safe.
    caplog.set_level(logging.DEBUG)

    resp = client_with_completions_route.post(
        "/v1/completions",
        json={
            "model": "qwen3-0.6b-8bit",
            "prompt": SENTINEL,
            "max_tokens": 8,
        },
    )
    assert resp.status_code == 200, resp.text

    info_records = _records_at_or_above(caplog, logging.INFO)
    leaked = [r for r in info_records if SENTINEL in r.getMessage()]
    assert not leaked, (
        "R-05 regression: /v1/completions INFO log carried the prompt "
        f"body. Offending records: {[r.getMessage() for r in leaked]}"
    )


def test_info_log_carries_metadata_only(
    client_with_completions_route, caplog
):
    """INFO line must still surface request metadata (model id, token
    counts, sampling) so operators can debug without reading bodies."""
    caplog.set_level(logging.INFO)

    resp = client_with_completions_route.post(
        "/v1/completions",
        json={
            "model": "qwen3-0.6b-8bit",
            "prompt": SENTINEL,
            "max_tokens": 8,
        },
    )
    assert resp.status_code == 200, resp.text

    info_msgs = [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.INFO
        and r.name.endswith("routes.completions")
    ]
    request_lines = [m for m in info_msgs if "[REQUEST]" in m]
    assert request_lines, f"no [REQUEST] log line found in {info_msgs!r}"

    line = request_lines[0]
    # Metadata must remain present (no body, but counts/sampling/model).
    assert "/v1/completions" in line
    assert "prompt_chars=" in line
    assert "n_prompts=" in line
    assert "max_tokens=" in line
    # And the secret sentinel must be absent from this specific record.
    assert SENTINEL not in line


def test_debug_log_carries_redacted_preview(
    client_with_completions_route, caplog
):
    """The body preview moved from INFO to DEBUG. Operators who flip
    the log level to DEBUG still get a 300-char preview for local
    debugging — but it's behind an explicit opt-in, not the production
    default."""
    # Note: the runtime log-namespace rebrand (vllm_mlx -> rapid_mlx)
    # rewrites the record.name AFTER emit, but caplog filters by the
    # logger we configure — set both to be safe.
    caplog.set_level(logging.DEBUG)

    resp = client_with_completions_route.post(
        "/v1/completions",
        json={
            "model": "qwen3-0.6b-8bit",
            "prompt": SENTINEL,
            "max_tokens": 8,
        },
    )
    assert resp.status_code == 200, resp.text

    debug_msgs = [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.DEBUG
        and r.name.endswith("routes.completions")
    ]
    preview_lines = [m for m in debug_msgs if "prompt preview" in m]
    assert preview_lines, (
        "DEBUG preview line missing — the body still has to be reachable "
        f"for local debugging, just not at INFO. debug_msgs={debug_msgs!r}"
    )
    # The sentinel CAN appear at DEBUG (that's the whole point of the
    # opt-in preview); the regression we guard against is INFO-level
    # leakage.
    assert SENTINEL in preview_lines[0]
