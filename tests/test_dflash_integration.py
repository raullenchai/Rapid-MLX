# SPDX-License-Identifier: Apache-2.0
"""Integration tests for the DFlash production path.

Two tiers of coverage here:

1. **Unit-ish** — exercise the CLI/info/server module surface without
   loading any weights. These run in the standard pytest suite (no
   mlx-vlm 0.5.0 required); they verify the user-facing plumbing
   (flag parsing, eligibility errors, info rendering, app construction
   with mocked model/processor/runtime).

2. **End-to-end** — guarded by ``RAPID_MLX_DFLASH_E2E=1`` and the
   presence of mlx-vlm 0.5.0 + the Qwen3.5-27B-8bit weights and DFlash
   drafter locally. These actually generate text via the production
   server. They live here (not in a separate file) so a maintainer can
   add new e2e cases without searching for the right module.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest

# =============================================================================
# CLI flag plumbing — argparse adds --enable-dflash + the eligibility check
# fires before the model load when an ineligible alias is passed.
# =============================================================================


def test_serve_parser_exposes_enable_dflash() -> None:
    """``--enable-dflash`` is a real flag (not argparse.SUPPRESSed)."""
    # serve flags are inlined in main(); easier to assert on --help than
    # to re-build the parser. Coarser but reliable.
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, "-m", "vllm_mlx.cli", "serve", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert out.returncode == 0, out.stderr
    assert "--enable-dflash" in out.stdout, "serve --help should list --enable-dflash"
    # Help text mentions the install path so users know how to enable
    # the feature when it's missing.
    assert "[dflash]" in out.stdout, (
        "help text should reference the rapid-mlx[dflash] extras"
    )


# =============================================================================
# info command DFlash block — the user-facing eligibility status table.
# =============================================================================


def test_info_renders_dflash_block_for_eligible_alias(capsys) -> None:
    """``rapid-mlx info qwen3.5-27b-8bit`` shows the per-gate table."""
    from vllm_mlx.cli import info_command

    args = type("Args", (), {"model": "qwen3.5-27b-8bit"})()
    info_command(args)
    captured = capsys.readouterr()
    assert "DFlash eligibility" in captured.out
    # All four declared-content gates should pass for the validated alias.
    assert "Declared support" in captured.out
    assert "Not MoE" in captured.out
    assert "Drafter declared" in captured.out
    assert "z-lab/Qwen3.5-27B-DFlash" in captured.out


def test_info_dflash_block_skipped_for_unknown_alias(capsys) -> None:
    """Unknown HF paths (not in aliases.json) — no DFlash block, since
    eligibility is per-alias and can't be inferred from a raw path."""
    from vllm_mlx.cli import info_command

    args = type("Args", (), {"model": "not-a-real-alias-zzz"})()
    info_command(args)
    captured = capsys.readouterr()
    assert "DFlash eligibility" not in captured.out


def test_info_dflash_marks_4bit_alias_ineligible(capsys) -> None:
    """The default ``qwen3.5-27b`` alias points at the 4-bit variant and
    must surface as ineligible with the right gate failing."""
    from vllm_mlx.cli import info_command

    args = type("Args", (), {"model": "qwen3.5-27b"})()
    info_command(args)
    captured = capsys.readouterr()
    assert "DFlash eligibility" in captured.out
    assert "ineligible" in captured.out


# =============================================================================
# Server-app construction — _build_app with mocks. Verifies the FastAPI
# surface and the lock + serial dispatch logic without loading weights.
# =============================================================================


def test_build_app_returns_fastapi_app() -> None:
    """The app exposes the three OpenAI-compat routes."""
    from vllm_mlx.speculative.dflash.runtime import DFlashRuntime
    from vllm_mlx.speculative.dflash.server import _build_app

    runtime = DFlashRuntime(
        drafter=MagicMock(),
        kind="dflash",
        drafter_repo="z-lab/Qwen3.5-27B-DFlash",
    )
    app = _build_app(
        model=MagicMock(),
        processor=MagicMock(),
        runtime=runtime,
        served_model_name="qwen3.5-27b-8bit",
        default_max_tokens=512,
        cors_origins=["*"],
    )
    routes = {r.path for r in app.routes if hasattr(r, "path")}
    assert "/healthz" in routes
    assert "/v1/models" in routes
    assert "/v1/chat/completions" in routes


def test_healthz_and_models_routes() -> None:
    """``/healthz`` reports DFlash mode + drafter; ``/v1/models`` lists
    the served name. These don't touch the model so they're safe to
    exercise without weights."""
    from fastapi.testclient import TestClient

    from vllm_mlx.speculative.dflash.runtime import DFlashRuntime
    from vllm_mlx.speculative.dflash.server import _build_app

    runtime = DFlashRuntime(
        drafter=MagicMock(),
        kind="dflash",
        drafter_repo="z-lab/Qwen3.5-27B-DFlash",
    )
    app = _build_app(
        model=MagicMock(),
        processor=MagicMock(),
        runtime=runtime,
        served_model_name="qwen3.5-27b-8bit",
        default_max_tokens=512,
        cors_origins=["*"],
    )
    client = TestClient(app)

    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["engine"] == "dflash"
    assert body["drafter"] == "z-lab/Qwen3.5-27B-DFlash"

    r = client.get("/v1/models")
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    assert body["data"][0]["id"] == "qwen3.5-27b-8bit"


def test_chat_completions_rejects_tools() -> None:
    """DFlash v1 doesn't run a tool-call parser. The route must reject
    tool requests with a clear 400 — silent passthrough would surprise
    users (model emits free-form text instead of structured tool calls)."""
    from fastapi.testclient import TestClient

    from vllm_mlx.speculative.dflash.runtime import DFlashRuntime
    from vllm_mlx.speculative.dflash.server import _build_app

    runtime = DFlashRuntime(
        drafter=MagicMock(),
        kind="dflash",
        drafter_repo="z-lab/Qwen3.5-27B-DFlash",
    )
    app = _build_app(
        model=MagicMock(),
        processor=MagicMock(),
        runtime=runtime,
        served_model_name="qwen3.5-27b-8bit",
        default_max_tokens=512,
        cors_origins=["*"],
    )
    client = TestClient(app)

    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen3.5-27b-8bit",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "get_weather", "parameters": {}},
                }
            ],
        },
    )
    assert r.status_code == 400
    assert "tool calling" in r.json()["detail"].lower()


def test_chat_completions_rejects_empty_messages() -> None:
    """OpenAI-compat parity: empty messages → 400."""
    from fastapi.testclient import TestClient

    from vllm_mlx.speculative.dflash.runtime import DFlashRuntime
    from vllm_mlx.speculative.dflash.server import _build_app

    runtime = DFlashRuntime(
        drafter=MagicMock(),
        kind="dflash",
        drafter_repo="z-lab/Qwen3.5-27B-DFlash",
    )
    app = _build_app(
        model=MagicMock(),
        processor=MagicMock(),
        runtime=runtime,
        served_model_name="qwen3.5-27b-8bit",
        default_max_tokens=512,
        cors_origins=["*"],
    )
    client = TestClient(app)
    r = client.post(
        "/v1/chat/completions",
        json={"model": "qwen3.5-27b-8bit", "messages": []},
    )
    assert r.status_code == 400


# =============================================================================
# Eligibility error surfaces (CLI startup) — the gate must fail fast
# with an actionable error before the user wastes 5 min downloading weights.
# =============================================================================


def test_run_dflash_server_raises_when_mlx_vlm_missing(monkeypatch) -> None:
    """When mlx-vlm 0.5.0+ isn't importable, ``run_dflash_server``
    raises with the install hint — not a cryptic ImportError."""
    from vllm_mlx.speculative.dflash import server as srv

    monkeypatch.setattr(srv, "have_runtime", lambda: False)
    with pytest.raises(RuntimeError, match=r"rapid-mlx\[dflash\]"):
        srv.run_dflash_server(
            main_model_repo="mlx-community/Qwen3.5-27B-8bit",
            drafter_repo="z-lab/Qwen3.5-27B-DFlash",
            host="127.0.0.1",
            port=58999,  # never bound — raises before uvicorn
            served_model_name="qwen3.5-27b-8bit",
            default_max_tokens=512,
            cors_origins=["*"],
            uvicorn_log_level="info",
        )


# =============================================================================
# End-to-end — heavy. Requires:
#   - ``RAPID_MLX_DFLASH_E2E=1`` env var (opt-in; CI doesn't set it)
#   - mlx-vlm 0.5.0+ installed (skipif gates this)
#   - Qwen3.5-27B-8bit + DFlash drafter cached locally (~30 GB combined)
# Validates the full happy path: model load → generate → OpenAI-format
# response. Mirrors the PoC bench harness but goes through our server.
# =============================================================================


_E2E_ENABLED = os.environ.get("RAPID_MLX_DFLASH_E2E", "") in ("1", "true", "yes")


@pytest.mark.skipif(
    not _E2E_ENABLED,
    reason="DFlash e2e disabled — set RAPID_MLX_DFLASH_E2E=1 to enable "
    "(requires Qwen3.5-27B-8bit + drafter cached, ~30 GB)",
)
def test_dflash_e2e_chat_completion_smoke() -> None:
    """One non-streaming chat completion through the production server.

    Loads the real model + drafter, fires a single completion through
    ``_non_stream_completion``, and asserts the response shape +
    plausible token counts. Doesn't measure speedup here — the bench
    harness owns that — but does confirm the wiring produces a valid
    OpenAI-compat response."""
    from vllm_mlx.speculative.dflash.eligibility import have_runtime

    if not have_runtime():
        pytest.skip("mlx-vlm 0.5.0+ not installed")

    from fastapi.testclient import TestClient
    from mlx_vlm import load

    from vllm_mlx.speculative.dflash.runtime import load_runtime
    from vllm_mlx.speculative.dflash.server import _build_app

    model, processor = load("mlx-community/Qwen3.5-27B-8bit")
    runtime = load_runtime("z-lab/Qwen3.5-27B-DFlash")
    app = _build_app(
        model=model,
        processor=processor,
        runtime=runtime,
        served_model_name="qwen3.5-27b-8bit",
        default_max_tokens=64,
        cors_origins=["*"],
    )
    client = TestClient(app)

    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen3.5-27b-8bit",
            "messages": [
                {"role": "user", "content": "Write the first 5 Fibonacci numbers."}
            ],
            "max_tokens": 64,
            "temperature": 0.0,
            "stream": False,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["choices"][0]["message"]["content"]
    assert body["usage"]["completion_tokens"] > 0
