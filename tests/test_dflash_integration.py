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

import importlib.util
import os
from unittest.mock import MagicMock

import pytest

# Several "unit-ish" tests below monkey-patch ``mlx_vlm`` symbols
# (stream_generate / prompt_utils.apply_chat_template) to exercise the
# server path without loading any weights. They still require mlx_vlm
# to be importable — i.e. the ``[dflash]`` (or ``[vision]``) extras
# present. Gate them so a minimal install runs the rest of the suite.
_MLX_VLM_AVAILABLE = importlib.util.find_spec("mlx_vlm") is not None
_skip_without_mlx_vlm = pytest.mark.skipif(
    not _MLX_VLM_AVAILABLE,
    reason="mlx_vlm not installed (DFlash test path needs [dflash] extras)",
)

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


def test_info_dflash_start_with_uses_alias_not_hf_path(capsys) -> None:
    """``main()`` resolves alias → HF path before dispatch, stashing the
    user-typed alias on ``args._original_alias``. The ``Start with`` hint
    in the DFlash block must render the *alias*, not the resolved HF
    repo — copy-pasting the resolved path back into ``rapid-mlx serve``
    breaks the alias-keyed eligibility check."""
    from vllm_mlx.cli import info_command

    # Mirror main()'s pre-resolve: model = HF path, _original_alias = alias.
    args = type(
        "Args",
        (),
        {
            "model": "mlx-community/Qwen3.5-27B-8bit",
            "_original_alias": "qwen3.5-27b-8bit",
        },
    )()
    info_command(args)
    captured = capsys.readouterr()
    assert "rapid-mlx serve qwen3.5-27b-8bit --enable-dflash" in captured.out
    # The HF path must not show up in the start-with hint.
    assert "rapid-mlx serve mlx-community/" not in captured.out


def test_models_listing_renders_dflash_column(capsys) -> None:
    """``rapid-mlx models`` must show a ``DFlash`` column so users can
    scan eligibility at a glance. The known-good alias renders ✓; a
    non-DFlash alias renders —."""
    from vllm_mlx.cli import models_command

    models_command(None)
    captured = capsys.readouterr()
    # Header
    assert "DFlash" in captured.out
    # The qwen3.5-27b-8bit row must show ✓ in its DFlash column. We can't
    # anchor on exact column offsets (table widths may shift), so look
    # for the alias and the marker on the same line.
    lines = captured.out.splitlines()
    eligible_row = next(
        (line for line in lines if "qwen3.5-27b-8bit " in line),
        None,
    )
    assert eligible_row is not None, "qwen3.5-27b-8bit row missing"
    assert "✓" in eligible_row, f"DFlash column should be ✓: {eligible_row!r}"

    # A non-DFlash alias renders — in the DFlash column.
    ineligible_row = next(
        (line for line in lines if "qwen3.5-4b " in line),
        None,
    )
    assert ineligible_row is not None, "qwen3.5-4b row missing"
    assert "—" in ineligible_row, f"DFlash column should be —: {ineligible_row!r}"


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


@_skip_without_mlx_vlm
def test_stream_completion_surfaces_generator_exception(monkeypatch) -> None:
    """When ``mlx_vlm.stream_generate`` raises mid-stream, the SSE
    response must finish cleanly with an OpenAI-style error block + a
    final ``[DONE]`` event — never leave the client hanging.

    Regression guard for the DeepSeek-flagged unhandled-exception path
    in ``_next_chunk`` (was only catching ``StopIteration``)."""
    from fastapi.testclient import TestClient

    from vllm_mlx.speculative.dflash import server as srv
    from vllm_mlx.speculative.dflash.runtime import DFlashRuntime

    class _BoomGen:
        """Sync generator that yields once, then raises — mirrors the
        shape of mlx-vlm ``stream_generate`` so the production iter
        loop exercises both the happy and error branches."""

        def __init__(self):
            self.calls = 0

        def __iter__(self):
            return self

        def __next__(self):
            self.calls += 1
            if self.calls == 1:

                class _Chunk:
                    text = "hello"
                    token = 1
                    prompt_tokens = 7
                    generation_tokens = 1

                return _Chunk()
            raise RuntimeError("simulated mlx-vlm failure")

    def _fake_stream_generate(*args, **kwargs):
        return _BoomGen()

    # Patch the symbol where it's looked up — mlx_vlm.stream_generate.
    # The server imports it lazily inside ``_stream_completion``, so we
    # patch the source module not a re-export.
    import mlx_vlm

    monkeypatch.setattr(mlx_vlm, "stream_generate", _fake_stream_generate)

    runtime = DFlashRuntime(
        drafter=MagicMock(),
        kind="dflash",
        drafter_repo="z-lab/Qwen3.5-27B-DFlash",
    )
    # Mock processor / model so _render_prompt doesn't try to invoke
    # mlx-vlm chat templating. apply_chat_template is patched at the
    # source too.
    import mlx_vlm.prompt_utils

    monkeypatch.setattr(
        mlx_vlm.prompt_utils,
        "apply_chat_template",
        lambda *a, **kw: "rendered prompt",
    )

    app = srv._build_app(
        model=MagicMock(),
        processor=MagicMock(),
        runtime=runtime,
        served_model_name="qwen3.5-27b-8bit",
        default_max_tokens=64,
        cors_origins=["*"],
    )
    client = TestClient(app)

    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "qwen3.5-27b-8bit",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    ) as resp:
        assert resp.status_code == 200
        body = b"".join(resp.iter_bytes()).decode()

    # The stream must terminate with a ``[DONE]`` marker — proves the
    # response coroutine didn't crash mid-flight.
    assert "data: [DONE]" in body, (
        "stream must end with [DONE] even when the upstream generator "
        f"raises; body was:\n{body}"
    )
    # The final delta block must carry the error block + finish_reason=error.
    assert '"finish_reason": "error"' in body
    assert "dflash_runtime_error" in body
    assert "simulated mlx-vlm failure" in body
    # And the one happy chunk that *did* arrive before the raise must
    # still appear in the stream.
    assert '"content": "hello"' in body


# =============================================================================
# finish_reason: must report "length" on token-budget hit (OpenAI clients
# distinguish "stop" from "length"; presenting "stop" for a truncated
# reply misleads downstream tools that auto-continue on truncation).
# =============================================================================


@_skip_without_mlx_vlm
def test_stream_completion_surfaces_constructor_exception(monkeypatch) -> None:
    """If ``stream_generate`` raises at *construction* time (before
    yielding the first chunk — e.g. OOM, missing mlx-vlm kernel), the
    SSE response must finish with an error block + ``[DONE]``, not
    propagate out of the async generator and leave the client hanging.
    Regression guard for round-7 review finding."""
    import asyncio

    from vllm_mlx.api.models import ChatCompletionRequest, Message
    from vllm_mlx.speculative.dflash import server as srv

    def _exploding_stream_generate(*a, **kw):
        raise RuntimeError("simulated OOM at generator construction")

    import mlx_vlm as _mlx_vlm

    monkeypatch.setattr(_mlx_vlm, "stream_generate", _exploding_stream_generate)

    req = ChatCompletionRequest(
        model="qwen3.5-27b-8bit",
        messages=[Message(role="user", content="ping")],
        stream=True,
    )
    gen_iter = srv._stream_completion(
        prompt="ping",
        request=req,
        served_model_name="qwen3.5-27b-8bit",
        gen_kwargs={"max_tokens": 4},
        model=MagicMock(),
        processor=MagicMock(),
    )

    async def _drain() -> list[bytes]:
        return [b async for b in gen_iter]

    chunks = asyncio.run(_drain())
    body = b"".join(chunks).decode()
    # Must still get a clean [DONE] terminator + the error block.
    assert "data: [DONE]" in body, (
        f"constructor-time crash must still terminate the stream; got:\n{body}"
    )
    assert '"finish_reason": "error"' in body
    assert "dflash_runtime_error" in body
    assert "simulated OOM at generator construction" in body


@_skip_without_mlx_vlm
def test_stream_completion_reports_length_when_max_tokens_hit(monkeypatch) -> None:
    """When ``generation_tokens >= max_tokens``, the final SSE event must
    carry ``finish_reason="length"``. mlx-vlm's GenerationResult doesn't
    expose finish_reason itself, so the server infers it from token-count
    vs budget. Regression guard for round-5 review finding."""
    import asyncio

    from vllm_mlx.api.models import ChatCompletionRequest, Message
    from vllm_mlx.speculative.dflash import server as srv

    class _Chunk:
        text = "x"
        token = 1
        prompt_tokens = 3
        generation_tokens = 4  # equals max_tokens below

    def _gen():
        yield _Chunk()

    import mlx_vlm as _mlx_vlm

    monkeypatch.setattr(_mlx_vlm, "stream_generate", lambda *a, **kw: _gen())

    req = ChatCompletionRequest(
        model="qwen3.5-27b-8bit",
        messages=[Message(role="user", content="ping")],
        stream=True,
    )
    gen_iter = srv._stream_completion(
        prompt="ping",
        request=req,
        served_model_name="qwen3.5-27b-8bit",
        gen_kwargs={"max_tokens": 4},  # budget hit
        model=MagicMock(),
        processor=MagicMock(),
    )

    async def _drain() -> list[bytes]:
        return [b async for b in gen_iter]

    chunks = asyncio.run(_drain())
    body = b"".join(chunks).decode()
    # The penultimate SSE event carries the final finish_reason — must be
    # "length", not "stop", since we hit the token budget.
    assert '"finish_reason": "length"' in body, (
        f"max_tokens hit should report finish_reason=length; got:\n{body}"
    )
    assert '"finish_reason": "stop"' not in body, (
        f"must not also emit finish_reason=stop on the final event; got:\n{body}"
    )


@_skip_without_mlx_vlm
def test_stream_completion_reports_stop_when_eos_lands_at_max_tokens(
    monkeypatch,
) -> None:
    """Edge case from round-13 review: if the model emits EOS at exactly
    ``max_tokens``, the stop was natural (the model would have stopped
    even with a larger budget). Reporting "length" would mislead clients
    that auto-continue on truncation. The token-id disambiguation must
    correctly classify this as "stop"."""
    import asyncio

    from vllm_mlx.api.models import ChatCompletionRequest, Message
    from vllm_mlx.speculative.dflash import server as srv

    class _Chunk:
        text = "done"
        token = 7  # we'll make this the EOS token id below
        prompt_tokens = 3
        generation_tokens = 4  # equals max_tokens — would otherwise be "length"

    def _gen():
        yield _Chunk()

    import mlx_vlm as _mlx_vlm

    monkeypatch.setattr(_mlx_vlm, "stream_generate", lambda *a, **kw: _gen())

    # Processor's tokenizer reports eos_token_id == 7 — matches the
    # chunk's token, so the heuristic must override "length" → "stop".
    processor = MagicMock()
    processor.tokenizer.eos_token_id = 7

    req = ChatCompletionRequest(
        model="qwen3.5-27b-8bit",
        messages=[Message(role="user", content="ping")],
        stream=True,
    )
    gen_iter = srv._stream_completion(
        prompt="ping",
        request=req,
        served_model_name="qwen3.5-27b-8bit",
        gen_kwargs={"max_tokens": 4},
        model=MagicMock(),
        processor=processor,
    )

    async def _drain() -> list[bytes]:
        return [b async for b in gen_iter]

    chunks = asyncio.run(_drain())
    body = b"".join(chunks).decode()
    assert '"finish_reason": "stop"' in body, (
        "EOS at exactly max_tokens must be classified as stop, not length; "
        f"got:\n{body}"
    )
    assert '"finish_reason": "length"' not in body, (
        f"must not also emit length when last token is EOS; got:\n{body}"
    )


@_skip_without_mlx_vlm
def test_non_stream_completion_reports_length_when_max_tokens_hit(
    monkeypatch,
) -> None:
    """Same length-vs-stop distinction in the non-stream path."""
    import asyncio

    from vllm_mlx.api.models import ChatCompletionRequest, Message
    from vllm_mlx.speculative.dflash import server as srv

    class _Result:
        text = "xxxx"
        prompt_tokens = 3
        generation_tokens = 4  # equals max_tokens below

    import mlx_vlm as _mlx_vlm

    monkeypatch.setattr(_mlx_vlm, "generate", lambda *a, **kw: _Result())

    req = ChatCompletionRequest(
        model="qwen3.5-27b-8bit",
        messages=[Message(role="user", content="ping")],
        stream=False,
    )
    resp = asyncio.run(
        srv._non_stream_completion(
            prompt="ping",
            request=req,
            served_model_name="qwen3.5-27b-8bit",
            gen_kwargs={"max_tokens": 4},
            model=MagicMock(),
            processor=MagicMock(),
        )
    )
    assert resp.choices[0].finish_reason == "length", (
        f"max_tokens hit should report finish_reason=length, "
        f"got {resp.choices[0].finish_reason!r}"
    )


# =============================================================================
# Thread-affinity contract — every mlx-vlm call must land on the dedicated
# single-thread executor. mlx-lm 0.31.3+ keeps GPU Stream in thread-local
# storage; hand-off across worker threads crashes mid-generation with
# "There is no Stream(gpu, N) in current thread". A regression here would
# only surface in production (no Stream error in mock tests), so we pin
# the invariant via the executor's identity.
# =============================================================================


@_skip_without_mlx_vlm
def test_stream_completion_pins_to_dedicated_executor(monkeypatch) -> None:
    """``_stream_completion`` must submit every mlx-vlm call (generator
    construction + each ``next(gen)``) to the module-level single-thread
    ``_dflash_executor`` — never to the default ThreadPoolExecutor
    (which has N workers and would tear apart mlx's thread-local Stream).

    The spy counts submissions on the pinned executor; a regression
    that routed work to the default executor would zero the counter."""
    import asyncio

    from vllm_mlx.api.models import ChatCompletionRequest, Message
    from vllm_mlx.speculative.dflash import server as srv

    # Runtime spy on ``_dflash_executor.submit`` — counts the actual
    # submissions during a real ``_stream_completion`` invocation.
    submit_count = [0]
    real_submit = srv._dflash_executor.submit

    def _count_submit(fn, *args, **kwargs):
        submit_count[0] += 1
        return real_submit(fn, *args, **kwargs)

    monkeypatch.setattr(srv._dflash_executor, "submit", _count_submit)

    class _OneChunk:
        text = "hi"
        generation_tokens = 1
        prompt_tokens = 2

    def _gen():
        yield _OneChunk()

    import mlx_vlm as _mlx_vlm

    monkeypatch.setattr(_mlx_vlm, "stream_generate", lambda *a, **kw: _gen())

    req = ChatCompletionRequest(
        model="qwen3.5-27b-8bit",
        messages=[Message(role="user", content="ping")],
        stream=True,
    )
    gen_iter = srv._stream_completion(
        prompt="ping",
        request=req,
        served_model_name="qwen3.5-27b-8bit",
        gen_kwargs={"max_tokens": 8},
        model=MagicMock(),
        processor=MagicMock(),
    )

    async def _drain() -> None:
        async for _ in gen_iter:
            pass

    asyncio.run(_drain())

    # Expect at least 2 submits: one for ``_make_gen`` (construct the
    # generator on the worker) and at least one for ``_next_chunk``.
    assert submit_count[0] >= 2, (
        f"_dflash_executor.submit only called {submit_count[0]} time(s); "
        "expected ≥2 (one for generator construction + one per next()). "
        "Thread affinity contract violated."
    )


def test_dflashruntime_accept_lens_tolerates_wrong_type(caplog) -> None:
    """If a future mlx-vlm renames ``accept_lens`` or changes its type,
    we must not crash on reset — degrade to a warning + no-op. Verifies
    the isinstance guard added after the round-4 review."""
    import logging

    from vllm_mlx.speculative.dflash.runtime import DFlashRuntime

    drafter = MagicMock()
    drafter.accept_lens = 42  # not a list
    rt = DFlashRuntime(drafter=drafter, kind="dflash", drafter_repo="fake/repo")

    with caplog.at_level(logging.WARNING, logger="vllm_mlx.speculative.dflash.runtime"):
        rt.reset_accept_lens()
    assert any("unexpected type" in rec.message for rec in caplog.records), (
        "reset_accept_lens should warn (not crash) when accept_lens isn't a list"
    )
    # Snapshot also degrades gracefully — empty list, not raise.
    assert rt.accept_lens_snapshot() == []


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


@_skip_without_mlx_vlm
def test_run_dflash_server_loads_models_on_executor_thread(monkeypatch) -> None:
    """Model + drafter MUST load on the ``_dflash_executor`` worker
    thread, never on the main thread.

    Regression guard for the v0.6.36 hotfix: mlx-lm 0.31.3+ keeps the GPU
    Stream in thread-local storage. If ``load()`` runs on the main thread
    but ``generate()`` runs on the executor, mlx-vlm raises ``RuntimeError:
    There is no Stream(gpu, N) in current thread`` on the first request.
    Pinning load to the same executor that owns generate keeps streams
    reachable for the process lifetime.

    Mocks ``load`` / ``load_runtime`` to record which thread they run on,
    and patches ``uvicorn.run`` to a no-op so the test doesn't bind a port.
    """
    import threading

    from vllm_mlx.speculative.dflash import server as srv

    load_thread: dict[str, str | None] = {"load": None, "load_runtime": None}

    def _fake_load(_repo):
        load_thread["load"] = threading.current_thread().name
        return MagicMock(), MagicMock()

    def _fake_load_runtime(_repo):
        load_thread["load_runtime"] = threading.current_thread().name
        return MagicMock()

    # Patch the imports at the point of use inside ``run_dflash_server``.
    import mlx_vlm as _mlx_vlm

    monkeypatch.setattr(_mlx_vlm, "load", _fake_load)
    monkeypatch.setattr(srv, "load_runtime", _fake_load_runtime)

    # No-op uvicorn so we don't bind a port; return immediately after load.
    import uvicorn

    monkeypatch.setattr(uvicorn, "run", lambda *a, **kw: None)

    srv.run_dflash_server(
        main_model_repo="mlx-community/Qwen3.5-27B-8bit",
        drafter_repo="z-lab/Qwen3.5-27B-DFlash",
        host="127.0.0.1",
        port=58998,
        served_model_name="qwen3.5-27b-8bit",
        default_max_tokens=512,
        cors_origins=["*"],
        uvicorn_log_level="info",
    )

    # Both must have run on the dflash worker thread (prefix set when the
    # ThreadPoolExecutor was constructed at module load).
    assert load_thread["load"] is not None, "load() was not called"
    assert load_thread["load_runtime"] is not None, "load_runtime() was not called"
    assert load_thread["load"].startswith("dflash-worker"), (
        f"model load must run on dflash-worker thread, ran on "
        f"{load_thread['load']!r} — Stream(gpu, N) would not be visible "
        f"to generate() on the executor."
    )
    assert load_thread["load_runtime"].startswith("dflash-worker"), (
        f"drafter load must run on dflash-worker thread, ran on "
        f"{load_thread['load_runtime']!r}."
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

    # Cache-presence gate — if a curious dev sets RAPID_MLX_DFLASH_E2E=1
    # but doesn't have the weights, ``mlx_vlm.load`` would silently
    # start a multi-GB HuggingFace download (no progress visible from
    # pytest). Skip with a precise reason so they know how to bring the
    # test into reach instead of waiting on a stuck process.
    from huggingface_hub import try_to_load_from_cache

    _required_repos = (
        "mlx-community/Qwen3.5-27B-8bit",
        "z-lab/Qwen3.5-27B-DFlash",
    )
    for repo in _required_repos:
        cfg = try_to_load_from_cache(repo, "config.json")
        if not cfg:
            pytest.skip(
                f"DFlash e2e: {repo} not cached locally. Run "
                f"`huggingface-cli download {repo}` before re-running "
                "with RAPID_MLX_DFLASH_E2E=1."
            )

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
