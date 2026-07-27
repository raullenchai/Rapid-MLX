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
from types import SimpleNamespace
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


@pytest.fixture(autouse=True)
def _reset_dflash_shared_globals():
    """Isolate the process-global server state that DFlash tests mutate.

    ``_build_app`` / ``run_dflash_server`` mutate two shared singletons that
    outlive a single test:

    * the process-global rate limiter (via ``configure_rate_limiter``) — a
      DFlash test that enables it (``rate_limit>0``) would otherwise leak that
      enabled state into unrelated later tests (e.g.
      ``tests/test_embeddings_timeout_admission.py``), turning their requests
      into surprise 429s; and

    * the shared ``get_config()`` singleton — ``_build_app`` writes ``api_key``,
      ``max_request_bytes``, ``body_receive_timeout_seconds`` and
      ``default_timeout`` into it. codex round-7 #4: snapshot and restore ALL
      of these so tests are not order-dependent under randomized execution
      (e.g. a test setting ``api_key`` must not silently authenticate — or
      reject — a later test that assumes no key).
    """
    from vllm_mlx import server as _main_server
    from vllm_mlx.config import get_config

    cfg = get_config()
    _snapshot = {
        field: getattr(cfg, field)
        for field in (
            "api_key",
            "max_request_bytes",
            "body_receive_timeout_seconds",
            "default_timeout",
            "enable_auto_tool_choice",
            "tool_call_parser",
            "reasoning_parser_name",
        )
    }
    # codex round-8 #3: the CORS tests here call ``configure_cors_from_env``,
    # which writes the process-global ``_last_resolved_cors_policy`` in
    # ``vllm_mlx.server``. Snapshot + restore it too so a resolved policy from
    # one test can't leak into a later one under randomized ordering.
    _cors_snapshot = _main_server._last_resolved_cors_policy
    try:
        yield
    finally:
        from vllm_mlx.middleware.auth import configure_rate_limiter

        configure_rate_limiter(0, enabled=False)
        for field, value in _snapshot.items():
            setattr(cfg, field, value)
        _main_server._last_resolved_cors_policy = _cors_snapshot


# =============================================================================
# CLI flag plumbing — argparse exposes --speculative-config and the eligibility check
# fires before the model load when an ineligible alias is passed.
# =============================================================================


def test_serve_parser_exposes_speculative_config() -> None:
    """DFlash is exposed through the shared speculative-config surface."""
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
    assert "--speculative-config" in out.stdout
    assert "--enable-dflash" not in out.stdout
    assert "--spec-decode" not in out.stdout
    # Help text mentions the install path so users know how to enable
    # the feature when it's missing.
    assert "[dflash]" in out.stdout, (
        "help text should reference the rapid-mlx[dflash] extras"
    )


def _dflash_cli_args(**overrides):
    data = {
        "model": "qwen3.5-27b-8bit",
        "_original_alias": None,
        "speculative_config": None,
        "enable_ddtree": False,
        "enable_dflash": False,
        "enable_mtp": False,
        "spec_decode": "none",
        "suffix_decoding": False,
        "no_spec_decode": False,
        "dflash_drafter_path": "",
    }
    data.update(overrides)
    return SimpleNamespace(**data)


def test_speculative_config_dflash_normalizes_to_legacy_server_flag() -> None:
    from vllm_mlx.cli import (
        _normalize_speculative_config_or_exit,
        _preflight_dflash_mutexes_or_exit,
        _resolve_dflash_drafter_repo,
    )

    args = _dflash_cli_args(
        speculative_config=('{"method":"dflash","model":"z-lab/Qwen3.5-27B-DFlash"}')
    )

    _normalize_speculative_config_or_exit(args)

    assert args.enable_dflash is True
    assert args.spec_decode == "none"
    _preflight_dflash_mutexes_or_exit(args)
    assert args._speculative_config.method == "dflash"
    assert args._speculative_config.model == "z-lab/Qwen3.5-27B-DFlash"
    profile = SimpleNamespace(dflash_draft_model="z-lab/default")
    assert _resolve_dflash_drafter_repo(args, profile) == "z-lab/Qwen3.5-27B-DFlash"


def test_dflash_preflight_rejects_legacy_mtp_alias(capsys) -> None:
    from vllm_mlx.cli import _preflight_dflash_mutexes_or_exit

    args = _dflash_cli_args(enable_dflash=True, enable_mtp=True)

    with pytest.raises(SystemExit) as excinfo:
        _preflight_dflash_mutexes_or_exit(args)

    assert excinfo.value.code == 1
    assert "DFlash cannot combine" in capsys.readouterr().out


def test_dflash_preflight_ignores_compat_marker_for_dflash_config() -> None:
    from vllm_mlx.cli import (
        _normalize_speculative_config_or_exit,
        _preflight_dflash_mutexes_or_exit,
    )

    args = _dflash_cli_args(speculative_config='{"method":"dflash"}')

    _normalize_speculative_config_or_exit(args)
    args.enable_mtp = True

    _preflight_dflash_mutexes_or_exit(args)


def test_dflash_speculative_config_rejects_no_spec_decode(capsys) -> None:
    from vllm_mlx.cli import (
        _normalize_speculative_config_or_exit,
    )

    args = _dflash_cli_args(
        speculative_config='{"method":"dflash"}',
        no_spec_decode=True,
    )

    with pytest.raises(SystemExit) as excinfo:
        _normalize_speculative_config_or_exit(args)

    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert "mutually exclusive" in captured.err


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
    """The default ``qwen3.5-27b-4bit`` alias points at the 4-bit variant and
    must surface as ineligible with the right gate failing."""
    from vllm_mlx.cli import info_command

    args = type("Args", (), {"model": "qwen3.5-27b-4bit"})()
    info_command(args)
    captured = capsys.readouterr()
    assert "DFlash eligibility" in captured.out
    assert "ineligible" in captured.out


def test_info_dflash_start_with_uses_alias_not_hf_path(capsys, monkeypatch) -> None:
    """``main()`` resolves alias → HF path before dispatch, stashing the
    user-typed alias on ``args._original_alias``. The ``Start with`` hint
    in the DFlash block must render the *alias*, not the resolved HF
    repo — copy-pasting the resolved path back into ``rapid-mlx serve``
    breaks the alias-keyed eligibility check.

    The ``Start with:`` hint is gated on ``eligible == True``, which
    requires ``have_runtime()`` (mlx-vlm 0.5.0+) to return True. In
    base / CI installs without the ``[dflash]`` extras the runtime
    check returns False, the hint is suppressed, and the alias-vs-HF
    invariant becomes untestable. Mock ``have_runtime`` so eligibility
    evaluates cleanly and the hint surface remains pinned regardless
    of which extras the test env carries.
    """
    from vllm_mlx.cli import info_command

    # Force eligibility True at the import site that ``_print_dflash_status``
    # uses, otherwise the start-with hint is suppressed.
    monkeypatch.setattr(
        "vllm_mlx.speculative.dflash.eligibility.have_runtime",
        lambda: True,
    )

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
    assert (
        """rapid-mlx serve qwen3.5-27b-8bit --speculative-config '{"method":"dflash"}'"""
        in captured.out
    )
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
        (line for line in lines if "qwen3.5-4b-4bit " in line),
        None,
    )
    assert ineligible_row is not None, "qwen3.5-4b-4bit row missing"
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


def test_run_dflash_server_wires_security_configuration(monkeypatch) -> None:
    """DFlash must enforce the same auth, rate, and body guards as serve.

    The production entry point is exercised with mocked model loading so this
    test covers the actual ``run_dflash_server`` -> ``_build_app`` wiring
    without downloading the DFlash model pair or binding a TCP port.
    """
    import sys
    import types

    from fastapi.testclient import TestClient

    from vllm_mlx.config import get_config
    from vllm_mlx.speculative.dflash import server as srv
    from vllm_mlx.speculative.dflash.runtime import DFlashRuntime

    runtime = DFlashRuntime(
        drafter=MagicMock(),
        kind="dflash",
        drafter_repo="z-lab/Qwen3.5-27B-DFlash",
    )
    monkeypatch.setattr(srv, "have_runtime", lambda: True)
    monkeypatch.setattr(srv, "load_runtime", lambda _repo: runtime)

    fake_mlx_vlm = types.ModuleType("mlx_vlm")
    fake_mlx_vlm.load = lambda _repo: (MagicMock(), MagicMock())
    monkeypatch.setitem(sys.modules, "mlx_vlm", fake_mlx_vlm)

    captured: dict = {}
    import uvicorn

    monkeypatch.setattr(
        uvicorn,
        "run",
        lambda app, **_kwargs: captured.setdefault("app", app),
    )

    srv.run_dflash_server(
        main_model_repo="mlx-community/Qwen3.5-27B-8bit",
        drafter_repo="z-lab/Qwen3.5-27B-DFlash",
        host="127.0.0.1",
        port=58997,
        served_model_name="qwen3.5-27b-8bit",
        default_max_tokens=64,
        cors_origins=[],
        uvicorn_log_level="info",
        api_key="dflash-secret",
        rate_limit=1,
        max_request_bytes=512,
        body_receive_timeout_seconds=7.5,
        default_timeout=12.5,
        max_concurrent_requests=3,
    )

    app = captured["app"]
    client = TestClient(app)
    auth = {"Authorization": "Bearer dflash-secret"}

    # Probe routes remain available to load balancers, but every /v1 route
    # requires the configured bearer key.
    assert client.get("/healthz").status_code == 200
    assert client.get("/v1/models").status_code == 401
    assert client.get("/v1/models", headers=auth).status_code == 200

    # The first request reaches DFlash's normal validation path (empty
    # messages -> 400) and therefore consumes the one-request rate budget;
    # the second must be rejected before it can reach generation.
    invalid_chat = {"model": "qwen3.5-27b-8bit", "messages": []}
    assert (
        client.post("/v1/chat/completions", headers=auth, json=invalid_chat).status_code
        == 400
    )
    assert (
        client.post("/v1/chat/completions", headers=auth, json=invalid_chat).status_code
        == 429
    )

    # The generic ASGI guard runs before FastAPI parses the JSON body.
    oversized = client.post(
        "/v1/chat/completions",
        headers={**auth, "Content-Type": "application/json"},
        content=b"x" * 513,
    )
    assert oversized.status_code == 413
    assert get_config().max_request_bytes == 512
    assert get_config().body_receive_timeout_seconds == 7.5
    assert get_config().default_timeout == 12.5
    assert app.state.dflash_admission._max_concurrent_requests == 3


def test_dflash_cli_forwards_security_and_resource_limits() -> None:
    """The dedicated server must receive every relevant ``serve`` policy.

    Keep this as a structural CLI contract: invoking ``serve_command`` would
    otherwise require a model alias, preflight checks, and an actual listener
    before the DFlash fork is reached.
    """
    import ast
    import inspect

    from vllm_mlx import cli

    tree = ast.parse(inspect.getsource(cli.serve_command))
    call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_dflash_server"
    )
    keywords = {keyword.arg: keyword.value for keyword in call.keywords}

    expected = {
        "api_key": "server._api_key",
        "rate_limit": "args.rate_limit",
        "max_request_bytes": "server._max_request_bytes",
        "body_receive_timeout_seconds": "server._body_receive_timeout_seconds",
        "default_timeout": "server._default_timeout",
        "max_concurrent_requests": "args.max_concurrent_requests",
        "cors_policy": "server.get_resolved_cors_policy()",
        "tool_call_parser": (
            "args.tool_call_parser if args.enable_auto_tool_choice else None"
        ),
        "reasoning_parser_name": "args.reasoning_parser",
    }
    assert {name: ast.unparse(keywords[name]) for name in expected} == expected


def test_dflash_cli_forks_before_batched_engine_startup() -> None:
    """DFlash startup must not advertise BatchedEngine-only features."""
    import inspect

    from vllm_mlx import cli

    source = inspect.getsource(cli.serve_command)
    dflash_call = source.index("run_dflash_server(")
    assert dflash_call < source.index("Mode: Continuous batching")
    assert dflash_call < source.index("Resolve per-alias TurboQuant default")


def test_dflash_admission_cap_rejects_before_prompt_rendering() -> None:
    """DFlash must bound its serial queue before prompt work or generation."""
    from fastapi.testclient import TestClient

    from vllm_mlx.speculative.dflash.runtime import DFlashRuntime
    from vllm_mlx.speculative.dflash.server import _build_app

    app = _build_app(
        model=MagicMock(),
        processor=MagicMock(),
        runtime=DFlashRuntime(
            drafter=MagicMock(),
            kind="dflash",
            drafter_repo="z-lab/Qwen3.5-27B-DFlash",
        ),
        served_model_name="qwen3.5-27b-8bit",
        default_max_tokens=64,
        cors_origins=[],
        max_concurrent_requests=1,
    )
    reservation = app.state.dflash_admission.reserve()
    try:
        response = TestClient(app).post(
            "/v1/chat/completions",
            json={
                "model": "qwen3.5-27b-8bit",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
    finally:
        reservation.release()

    assert response.status_code == 503
    assert response.headers["Retry-After"] == "1"


def test_dflash_nonstream_timeout_keeps_gpu_slot_until_worker_finishes(
    monkeypatch,
) -> None:
    """A 504 must not let another call overlap the still-running worker."""
    import asyncio
    import sys
    import time
    import types

    from fastapi import HTTPException

    from vllm_mlx.speculative.dflash import server as srv

    fake_mlx_vlm = types.ModuleType("mlx_vlm")

    def slow_generate(*_args, **_kwargs):
        time.sleep(0.05)
        return SimpleNamespace(text="done", prompt_tokens=1, generation_tokens=1)

    fake_mlx_vlm.generate = slow_generate
    monkeypatch.setitem(sys.modules, "mlx_vlm", fake_mlx_vlm)

    async def exercise_timeout() -> None:
        admission = srv._DFlashAdmission(max_concurrent_requests=1)
        reservation = admission.reserve()
        with pytest.raises(HTTPException) as excinfo:
            await srv._non_stream_completion(
                prompt="prompt",
                request=MagicMock(),
                served_model_name="qwen3.5-27b-8bit",
                gen_kwargs={},
                model=MagicMock(),
                processor=MagicMock(),
                timeout=0.01,
                admission_reservation=reservation,
            )
        assert excinfo.value.status_code == 504
        assert admission._reservations == 1
        assert srv._dflash_lock.locked()

        # The detached worker owns the lock until ``slow_generate`` returns.
        await asyncio.sleep(0.1)
        assert admission._reservations == 0
        assert not srv._dflash_lock.locked()

    asyncio.run(exercise_timeout())


def test_dflash_timeout_message_reports_original_not_post_render_budget(
    monkeypatch,
) -> None:
    """codex round-5 #6: the 504 message reports the ORIGINAL configured
    timeout, not the render-reduced remaining budget.

    The endpoint charges prompt-render time against one absolute deadline and
    hands the REMAINING budget to the completion helper as ``timeout``. But a
    user who set a 60s timeout and spent 20s rendering should read "timed out
    after 60.0 seconds", not "40.0" — otherwise they chase a phantom shorter
    limit. ``timeout_label`` carries the original for diagnostics.
    """
    import asyncio
    import sys
    import time
    import types

    from fastapi import HTTPException

    from vllm_mlx.speculative.dflash import server as srv

    fake_mlx_vlm = types.ModuleType("mlx_vlm")
    fake_mlx_vlm.generate = lambda *a, **kw: (
        time.sleep(0.05)
        or SimpleNamespace(text="x", prompt_tokens=1, generation_tokens=1)
    )
    monkeypatch.setitem(sys.modules, "mlx_vlm", fake_mlx_vlm)

    async def exercise() -> None:
        admission = srv._DFlashAdmission(max_concurrent_requests=1)
        reservation = admission.reserve()
        with pytest.raises(HTTPException) as excinfo:
            await srv._non_stream_completion(
                prompt="prompt",
                request=MagicMock(),
                served_model_name="qwen3.5-27b-8bit",
                gen_kwargs={},
                model=MagicMock(),
                processor=MagicMock(),
                timeout=0.01,  # enforced remaining budget (tiny → 504)
                timeout_label=60.0,  # ORIGINAL configured timeout
                admission_reservation=reservation,
            )
        assert excinfo.value.status_code == 504
        # Message reports the ORIGINAL 60.0s, NOT the 0.01s remaining budget
        # (which would round to "0.0 seconds").
        assert "60.0 seconds" in excinfo.value.detail, excinfo.value.detail
        assert "after 0.0 seconds" not in excinfo.value.detail
        # Let the detached worker finish so the lock/slot release.
        await asyncio.sleep(0.1)

    srv._dflash_lock = asyncio.Lock()
    try:
        asyncio.run(exercise())
    finally:
        srv._dflash_lock = asyncio.Lock()


def test_dflash_stream_uses_absolute_deadline_over_relative_timeout(
    monkeypatch,
) -> None:
    """codex round-8 #2: an absolute ``deadline`` takes precedence over a
    relative ``timeout``, so the wall-clock spent before the first stream step
    is NOT refunded to the request budget.

    We pass a ``deadline`` that is ALREADY in the past alongside a large
    ``timeout``. If the helper re-based the clock from ``timeout`` (the bug),
    the stream would run to completion; honoring the absolute past deadline, it
    must time out immediately.
    """
    import asyncio
    import sys
    import types

    from vllm_mlx.api.models import ChatCompletionRequest, Message
    from vllm_mlx.speculative.dflash import server as srv

    class _Gen:
        def __iter__(self):
            return self

        def __next__(self):
            return SimpleNamespace(
                text="tok", generation_tokens=1, prompt_tokens=1, token=1
            )

        def close(self):
            pass

    fake_mlx_vlm = types.ModuleType("mlx_vlm")
    fake_mlx_vlm.stream_generate = lambda *a, **kw: _Gen()
    monkeypatch.setitem(sys.modules, "mlx_vlm", fake_mlx_vlm)

    request = ChatCompletionRequest(
        model="qwen3.5-27b-8bit",
        messages=[Message(role="user", content="hi")],
        stream=True,
    )

    async def exercise() -> None:
        admission = srv._DFlashAdmission(max_concurrent_requests=1)
        reservation = admission.reserve()
        loop = asyncio.get_running_loop()
        stream = srv._stream_completion(
            prompt="p",
            request=request,
            served_model_name="qwen3.5-27b-8bit",
            gen_kwargs={"max_tokens": 8},
            model=MagicMock(),
            processor=MagicMock(),
            # A large relative timeout that MUST be ignored in favor of...
            timeout=999.0,
            timeout_label=999.0,
            # ...this already-expired absolute deadline.
            deadline=loop.time() - 1.0,
            admission_reservation=reservation,
        )
        body = b"".join([chunk async for chunk in stream]).decode()
        # Immediate timeout — the past deadline wins over the 999s timeout.
        assert "timed out" in body, (
            "codex #8 regression: relative timeout re-based the clock, "
            "ignoring the absolute (already-expired) deadline"
        )
        assert "data: [DONE]" in body

    srv._dflash_lock = asyncio.Lock()
    try:
        asyncio.run(exercise())
    finally:
        srv._dflash_lock = asyncio.Lock()


def test_dflash_format_timeout_seconds_adaptive_precision() -> None:
    """codex round-6 #5: small positive timeouts keep meaningful digits."""
    from vllm_mlx.speculative.dflash.server import _format_timeout_seconds

    assert _format_timeout_seconds(60.0) == "60.0 seconds"
    assert _format_timeout_seconds(1.5) == "1.5 seconds"
    # Sub-second and sub-100ms: NOT collapsed to "0.0 seconds".
    assert _format_timeout_seconds(0.25) == "0.25 seconds"
    assert _format_timeout_seconds(0.01) == "0.010 seconds"
    assert _format_timeout_seconds(0.005) == "0.005 seconds"
    assert _format_timeout_seconds(0.0) == "0 seconds"
    # codex round-7 #3: sub-millisecond timeouts floor instead of "0.000".
    assert _format_timeout_seconds(0.0001) == "<0.001 seconds"


def test_dflash_stream_timeout_stops_after_the_inflight_worker_step(
    monkeypatch,
) -> None:
    """DFlash honors stream deadlines without overlapping GPU work.

    codex round-2 #2: the timeout SSE must be emitted IMMEDIATELY on
    deadline expiry — the response must NOT block on the non-preemptible
    in-flight worker step. The lock stays held (deferred cleanup) until the
    detached worker actually exits, then is released. Here the worker sleeps
    a long ``_WORKER_SLEEP`` while the deadline is ``_DEADLINE``, so a fix
    that awaited the worker before emitting the timeout would keep the client
    waiting the full worker sleep; the fixed path emits the timeout SSE right
    away and releases the lock only after the worker done-callback runs.

    codex round-4 #6: the elapsed-time assertion uses a threshold with ample
    separation from BOTH the deadline and the worker completion (rather than
    comparing against the worker's exact sleep), so it is not
    scheduler-sensitive on loaded CI hosts.
    """
    import asyncio
    import sys
    import threading
    import time
    import types

    from vllm_mlx.api.models import ChatCompletionRequest, Message
    from vllm_mlx.speculative.dflash import server as srv

    # Wide separation (codex round-4 #6): deadline 10 ms, worker 500 ms,
    # assertion threshold 200 ms. The fixed path returns the timeout in
    # ~10 ms; a regressed (worker-awaiting) path would take ~500 ms. 200 ms
    # sits well clear of both, so ordinary scheduler jitter cannot flip it.
    _DEADLINE = 0.01
    _WORKER_SLEEP = 0.5
    _ELAPSED_THRESHOLD = 0.2

    worker_finished = threading.Event()

    class _SlowGenerator:
        def __next__(self):
            time.sleep(_WORKER_SLEEP)
            worker_finished.set()
            raise StopIteration

        def close(self):
            pass

    fake_mlx_vlm = types.ModuleType("mlx_vlm")
    fake_mlx_vlm.stream_generate = lambda *_args, **_kwargs: _SlowGenerator()
    monkeypatch.setitem(sys.modules, "mlx_vlm", fake_mlx_vlm)

    request = ChatCompletionRequest(
        model="qwen3.5-27b-8bit",
        messages=[Message(role="user", content="hi")],
        stream=True,
    )

    async def exercise() -> None:
        stream = srv._stream_completion(
            prompt="prompt",
            request=request,
            served_model_name="qwen3.5-27b-8bit",
            gen_kwargs={"max_tokens": 8},
            model=MagicMock(),
            processor=MagicMock(),
            timeout=_DEADLINE,
        )
        started = asyncio.get_running_loop().time()
        body = b"".join([chunk async for chunk in stream]).decode()
        elapsed = asyncio.get_running_loop().time() - started

        # The timeout SSE + DONE are emitted without blocking on the worker.
        # codex round-6 #5: adaptive precision renders the 0.01s deadline as
        # "0.010 seconds", not a misleading "0.0 seconds".
        assert "DFlash stream timed out after 0.010 seconds." in body
        assert "data: [DONE]" in body
        # The worker step must NOT gate the response — the fixed path surfaces
        # the timeout well before the worker's (much longer) sleep elapses.
        assert elapsed < _ELAPSED_THRESHOLD, (
            f"timeout SSE blocked on the in-flight worker ({elapsed:.3f}s); "
            "codex round-2 #2 regression"
        )

        # The lock stays held by deferred cleanup until the detached worker
        # exits; then the done-callback releases it. Poll (keeping the loop
        # alive) until that happens.
        assert await asyncio.to_thread(worker_finished.wait, 1)
        for _ in range(200):
            if not srv._dflash_lock.locked():
                break
            await asyncio.sleep(0.01)
        assert not srv._dflash_lock.locked(), (
            "lock not released after the deferred worker finished"
        )

    srv._dflash_lock = asyncio.Lock()
    try:
        asyncio.run(exercise())
    finally:
        srv._dflash_lock = asyncio.Lock()


def test_dflash_stream_timeout_bounds_lock_queue_wait(monkeypatch) -> None:
    """A stream deadline expires while waiting for DFlash's serial lock."""
    import asyncio
    import sys
    import threading
    import types

    from vllm_mlx.api.models import ChatCompletionRequest, Message
    from vllm_mlx.speculative.dflash import server as srv

    started = threading.Event()
    release_first = threading.Event()
    generated = 0

    class _BlockingGenerator:
        def __next__(self):
            started.set()
            assert release_first.wait(timeout=5)
            raise StopIteration

        def close(self):
            pass

    class _EmptyGenerator:
        def __next__(self):
            raise StopIteration

        def close(self):
            pass

    def stream_generate(*_args, **_kwargs):
        nonlocal generated
        generated += 1
        return _BlockingGenerator() if generated == 1 else _EmptyGenerator()

    fake_mlx_vlm = types.ModuleType("mlx_vlm")
    fake_mlx_vlm.stream_generate = stream_generate
    monkeypatch.setitem(sys.modules, "mlx_vlm", fake_mlx_vlm)

    request = ChatCompletionRequest(
        model="qwen3.5-27b-8bit",
        messages=[Message(role="user", content="hi")],
        stream=True,
    )

    async def exercise_queue_timeout() -> None:
        first = srv._stream_completion(
            prompt="first",
            request=request,
            served_model_name="qwen3.5-27b-8bit",
            gen_kwargs={},
            model=MagicMock(),
            processor=MagicMock(),
        )
        await anext(first)
        first_next = asyncio.create_task(anext(first))
        assert await asyncio.to_thread(started.wait, 1)

        second = srv._stream_completion(
            prompt="second",
            request=request,
            served_model_name="qwen3.5-27b-8bit",
            gen_kwargs={},
            model=MagicMock(),
            processor=MagicMock(),
            timeout=0.01,
        )
        await anext(second)
        try:
            timeout_event = await anext(second)
            assert b"DFlash stream timed out" in timeout_event
            assert generated == 1
        finally:
            release_first.set()
            await first_next

    # asyncio.Lock binds to the first loop that has waiters. Production has
    # one long-lived uvicorn loop; tests intentionally create short-lived
    # loops, so leave the module global fresh for the next isolated case.
    srv._dflash_lock = asyncio.Lock()
    try:
        asyncio.run(exercise_queue_timeout())
    finally:
        srv._dflash_lock = asyncio.Lock()


def test_dflash_stream_cancellation_releases_admission_slot() -> None:
    """Client disconnects must not leave DFlash permanently at capacity."""
    import asyncio

    from vllm_mlx.speculative.dflash import server as srv

    closed = False

    async def source():
        nonlocal closed
        try:
            yield b"data: first\n\n"
            await asyncio.Event().wait()
        finally:
            closed = True

    async def cancel_stream() -> None:
        admission = srv._DFlashAdmission(max_concurrent_requests=1)
        reservation = admission.reserve()
        stream = srv._stream_with_admission(source(), reservation)
        assert await anext(stream) == b"data: first\n\n"
        await stream.aclose()
        assert closed
        assert admission._reservations == 0

    asyncio.run(cancel_stream())


def test_dflash_stream_claims_slot_only_when_iteration_begins() -> None:
    """codex round-4 #1: ownership of the admission slot must transfer to the
    stream (``claim``) only once ``_stream_with_admission`` is actually
    iterated — NOT when the endpoint returns the ``StreamingResponse``.

    If the endpoint claimed the slot eagerly and Starlette then failed to send
    the response headers (client vanished during startup), the stream body
    would never run, its ``finally`` release would never fire, and the
    middleware safety net — seeing a *claimed* slot — would decline to release
    it, permanently shrinking capacity. Guarding on ``claimed`` here proves an
    unstarted stream stays unclaimed (middleware reclaims it) and a started
    one claims + releases itself.
    """
    import asyncio

    from vllm_mlx.speculative.dflash import server as srv

    async def exercise() -> None:
        admission = srv._DFlashAdmission(max_concurrent_requests=1)

        # 1. Response-startup failure: the wrapped generator is created but
        #    never iterated (Starlette never got past sending headers). The
        #    slot must remain UNCLAIMED so the middleware's safety net owns
        #    the release.
        never_started = False

        async def unused_source():
            nonlocal never_started
            never_started = True
            yield b"data: x\n\n"

        reservation = admission.reserve()
        stream = srv._stream_with_admission(unused_source(), reservation)
        assert reservation.claimed is False, (
            "codex #1 regression: slot claimed before the stream was iterated"
        )
        assert never_started is False
        # The middleware safety net force-releases an unclaimed slot.
        await stream.aclose()
        if not reservation.claimed:
            reservation.release(force=True)
        assert admission._reservations == 0

        # 2. Normal start: iterating the wrapper transfers ownership (claim),
        #    and closing it releases the slot exactly once.
        async def source():
            yield b"data: first\n\n"

        reservation2 = admission.reserve()
        stream2 = srv._stream_with_admission(source(), reservation2)
        assert await anext(stream2) == b"data: first\n\n"
        assert reservation2.claimed is True, (
            "codex #1 regression: iterating the stream did not claim the slot"
        )
        await stream2.aclose()
        assert admission._reservations == 0

    asyncio.run(exercise())


def test_dflash_stream_admission_released_when_aclose_is_cancelled() -> None:
    """codex round-5 #4: ``_stream_with_admission`` must release the slot even
    if ``aclose()`` raises ``asyncio.CancelledError``.

    ``CancelledError`` subclasses ``BaseException``, not ``Exception``, so the
    pre-fix ``except Exception`` around ``await aclose()`` let it skip the
    mandatory ``reservation.release()`` — permanently leaking a slot whenever a
    client cancellation propagated through generator teardown. The release now
    lives in its own unconditional ``finally``.
    """
    import asyncio

    from vllm_mlx.speculative.dflash import server as srv

    class _CancelOnClose:
        """An async-iterator whose ``aclose`` raises CancelledError."""

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

        async def aclose(self):
            raise asyncio.CancelledError

    async def exercise() -> None:
        admission = srv._DFlashAdmission(max_concurrent_requests=1)
        reservation = admission.reserve()
        wrapped = srv._stream_with_admission(_CancelOnClose(), reservation)
        # Drive the wrapper to completion; its ``finally`` closes the inner
        # stream (which raises CancelledError) and must STILL release the slot.
        with pytest.raises(asyncio.CancelledError):
            async for _ in wrapped:
                pass
        assert admission._reservations == 0, (
            "codex #4 regression: CancelledError in aclose() skipped the "
            "admission release"
        )
        assert reservation.claimed is True

    asyncio.run(exercise())


def test_dflash_stream_cancellation_waits_for_worker_cleanup(monkeypatch) -> None:
    """A cancelled stream cannot let a new generator overtake its close."""
    import asyncio
    import sys
    import threading
    import types

    from vllm_mlx.api.models import ChatCompletionRequest, Message
    from vllm_mlx.speculative.dflash import server as srv

    first_step_started = threading.Event()
    allow_first_step_to_finish = threading.Event()
    first_closed = threading.Event()
    events: list[str] = []
    generated = 0

    class _FirstGenerator:
        def __next__(self):
            first_step_started.set()
            assert allow_first_step_to_finish.wait(timeout=5)
            raise StopIteration

        def close(self):
            events.append("first-close")
            first_closed.set()

    class _SecondGenerator:
        def __next__(self):
            raise StopIteration

        def close(self):
            pass

    def stream_generate(*_args, **_kwargs):
        nonlocal generated
        generated += 1
        if generated == 1:
            events.append("first-generate")
            return _FirstGenerator()
        events.append("second-generate")
        return _SecondGenerator()

    fake_mlx_vlm = types.ModuleType("mlx_vlm")
    fake_mlx_vlm.stream_generate = stream_generate
    monkeypatch.setitem(sys.modules, "mlx_vlm", fake_mlx_vlm)

    request = ChatCompletionRequest(
        model="qwen3.5-27b-8bit",
        messages=[Message(role="user", content="hi")],
        stream=True,
    )

    async def exercise_cancellation() -> None:
        admission = srv._DFlashAdmission(max_concurrent_requests=2)
        first = srv._stream_completion(
            prompt="first",
            request=request,
            served_model_name="qwen3.5-27b-8bit",
            gen_kwargs={},
            model=MagicMock(),
            processor=MagicMock(),
            admission_reservation=admission.reserve(),
        )
        # The role marker is emitted before the GPU worker starts.
        await anext(first)
        first_next = asyncio.create_task(anext(first))
        assert await asyncio.to_thread(first_step_started.wait, 1)

        first_next.cancel()
        # Do not await the cancelled consumer yet: its worker is still in a
        # blocking token step. A new stream must remain outside the serial
        # worker until that step's generator has been closed.
        await asyncio.sleep(0)

        second = srv._stream_completion(
            prompt="second",
            request=request,
            served_model_name="qwen3.5-27b-8bit",
            gen_kwargs={},
            model=MagicMock(),
            processor=MagicMock(),
            admission_reservation=admission.reserve(),
        )
        await anext(second)
        second_next = asyncio.create_task(anext(second))
        await asyncio.sleep(0.01)
        assert "second-generate" not in events

        allow_first_step_to_finish.set()
        assert await asyncio.to_thread(first_closed.wait, 1)
        with pytest.raises(asyncio.CancelledError):
            await first_next
        await second_next
        assert events.index("first-close") < events.index("second-generate")
        await second.aclose()
        assert admission._reservations == 0

    srv._dflash_lock = asyncio.Lock()
    try:
        asyncio.run(exercise_cancellation())
    finally:
        srv._dflash_lock = asyncio.Lock()


def test_dflash_inherits_explicit_cors_policy(monkeypatch) -> None:
    """DFlash must not broaden the operator's resolved CORS settings."""
    from fastapi.testclient import TestClient

    from vllm_mlx import server
    from vllm_mlx.speculative.dflash.runtime import DFlashRuntime
    from vllm_mlx.speculative.dflash.server import _build_app

    monkeypatch.setenv("RAPID_MLX_CORS_ALLOW_METHODS", "POST")
    monkeypatch.setenv("RAPID_MLX_CORS_ALLOW_HEADERS", "Content-Type")
    monkeypatch.setenv("RAPID_MLX_CORS_MAX_AGE", "17")
    monkeypatch.setenv("RAPID_MLX_CORS_ALLOW_CREDENTIALS", "false")
    server.configure_cors_from_env(["https://console.example"])
    policy = server.get_resolved_cors_policy()
    assert policy is not None

    app = _build_app(
        model=MagicMock(),
        processor=MagicMock(),
        runtime=DFlashRuntime(
            drafter=MagicMock(),
            kind="dflash",
            drafter_repo="z-lab/Qwen3.5-27B-DFlash",
        ),
        served_model_name="qwen3.5-27b-8bit",
        default_max_tokens=64,
        cors_origins=list(policy.origins),
        cors_policy=policy,
    )
    client = TestClient(app)
    allowed = {
        "Origin": "https://console.example",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "Content-Type",
    }
    response = client.options("/v1/chat/completions", headers=allowed)
    assert response.status_code == 200
    assert response.headers["Access-Control-Allow-Methods"] == "POST"
    assert response.headers["Access-Control-Max-Age"] == "17"
    assert "Access-Control-Allow-Credentials" not in response.headers

    denied = client.options(
        "/v1/chat/completions",
        headers={**allowed, "Access-Control-Request-Headers": "Authorization"},
    )
    assert denied.status_code == 400


def test_dflash_cors_wildcard_forces_credentials_off(monkeypatch) -> None:
    """A CORS policy snapshot must retain Fetch's wildcard invariant."""
    from fastapi.testclient import TestClient

    from vllm_mlx import server
    from vllm_mlx.speculative.dflash.runtime import DFlashRuntime
    from vllm_mlx.speculative.dflash.server import _build_app

    monkeypatch.setenv("RAPID_MLX_CORS_ALLOW_CREDENTIALS", "true")
    server.configure_cors_from_env(["*"])
    policy = server.get_resolved_cors_policy()
    assert policy is not None
    assert policy.allow_credentials is False

    app = _build_app(
        model=MagicMock(),
        processor=MagicMock(),
        runtime=DFlashRuntime(
            drafter=MagicMock(),
            kind="dflash",
            drafter_repo="z-lab/Qwen3.5-27B-DFlash",
        ),
        served_model_name="qwen3.5-27b-8bit",
        default_max_tokens=64,
        cors_origins=list(policy.origins),
        cors_policy=policy,
    )
    response = TestClient(app).options(
        "/v1/chat/completions",
        headers={
            "Origin": "https://evil.example",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.headers["Access-Control-Allow-Origin"] == "*"
    assert "Access-Control-Allow-Credentials" not in response.headers


@pytest.mark.parametrize(
    ("request_fields", "expected"),
    [
        (
            {
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "get_weather", "parameters": {}},
                    }
                ]
            },
            "tool-call parser",
        ),
        ({"enable_thinking": True}, "reasoning parser"),
    ],
)
def test_chat_completions_rejects_unparsed_structured_output(
    request_fields, expected
) -> None:
    """Explicit parser opt-outs must not turn structured output into raw text."""
    from fastapi.testclient import TestClient

    from vllm_mlx.speculative.dflash.runtime import DFlashRuntime
    from vllm_mlx.speculative.dflash.server import _build_app

    app = _build_app(
        model=MagicMock(),
        processor=MagicMock(),
        runtime=DFlashRuntime(
            drafter=MagicMock(),
            kind="dflash",
            drafter_repo="z-lab/Qwen3.5-27B-DFlash",
        ),
        served_model_name="qwen3.5-27b-8bit",
        default_max_tokens=64,
        cors_origins=["*"],
    )
    response = TestClient(app).post(
        "/v1/chat/completions",
        json={
            "model": "qwen3.5-27b-8bit",
            "messages": [{"role": "user", "content": "hi"}],
            **request_fields,
        },
    )
    assert response.status_code == 400
    assert expected in response.json()["error"]["message"]


@pytest.mark.parametrize(
    "tool_choice",
    [
        "required",
        "none",
        {"type": "function", "function": {"name": "get_weather"}},
    ],
)
def test_chat_completions_rejects_unsupported_tool_choice(tool_choice) -> None:
    """DFlash must not silently claim forced-tool semantics it cannot enforce."""
    from fastapi.testclient import TestClient

    from vllm_mlx.speculative.dflash.runtime import DFlashRuntime
    from vllm_mlx.speculative.dflash.server import _build_app

    app = _build_app(
        model=MagicMock(),
        processor=MagicMock(),
        runtime=DFlashRuntime(
            drafter=MagicMock(),
            kind="dflash",
            drafter_repo="z-lab/Qwen3.5-27B-DFlash",
        ),
        served_model_name="qwen3.5-27b-8bit",
        default_max_tokens=64,
        cors_origins=["*"],
        tool_call_parser="hermes",
    )
    response = TestClient(app).post(
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
            "tool_choice": tool_choice,
        },
    )
    assert response.status_code == 400
    assert "tool_choice='auto' only" in response.json()["error"]["message"]


@_skip_without_mlx_vlm
def test_chat_completions_parses_tool_calls(monkeypatch) -> None:
    """DFlash reuses the standard Hermes parser for OpenAI tool calls."""
    import mlx_vlm
    import mlx_vlm.prompt_utils
    from fastapi.testclient import TestClient

    from vllm_mlx.speculative.dflash.runtime import DFlashRuntime
    from vllm_mlx.speculative.dflash.server import _build_app

    captured_template: dict = {}

    def _render(*args, **kwargs):
        captured_template.update(kwargs)
        return "rendered prompt"

    monkeypatch.setattr(mlx_vlm.prompt_utils, "apply_chat_template", _render)
    monkeypatch.setattr(
        mlx_vlm,
        "generate",
        lambda *args, **kwargs: SimpleNamespace(
            text=(
                '<tool_call>{"name":"get_weather",'
                '"arguments":{"city":"Paris"}}</tool_call>'
            ),
            prompt_tokens=8,
            generation_tokens=6,
        ),
    )

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
        tool_call_parser="hermes",
        reasoning_parser_name="qwen3",
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
                    "function": {
                        "name": "get_weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    },
                }
            ],
        },
    )
    assert r.status_code == 200, r.text
    choice = r.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] == ""
    call = choice["message"]["tool_calls"][0]
    assert call["function"]["name"] == "get_weather"
    assert call["function"]["arguments"] == '{"city": "Paris"}'
    assert captured_template["tools"][0]["function"]["name"] == "get_weather"


@_skip_without_mlx_vlm
def test_chat_completions_streams_tool_calls(monkeypatch) -> None:
    """DFlash streaming emits structured deltas, not raw tool markup."""
    import json

    import mlx_vlm
    import mlx_vlm.prompt_utils
    from fastapi.testclient import TestClient

    from vllm_mlx.speculative.dflash.runtime import DFlashRuntime
    from vllm_mlx.speculative.dflash.server import _build_app

    monkeypatch.setattr(
        mlx_vlm.prompt_utils,
        "apply_chat_template",
        lambda *args, **kwargs: "rendered prompt",
    )

    chunks = (
        "<tool_call>",
        '{"name":"get_weather","arguments":{"city":"Paris"}}',
        "</tool_call>",
    )

    def _stream(*args, **kwargs):
        for index, text in enumerate(chunks, start=1):
            yield SimpleNamespace(
                text=text,
                token=index,
                prompt_tokens=8,
                generation_tokens=index,
            )

    monkeypatch.setattr(mlx_vlm, "stream_generate", _stream)
    app = _build_app(
        model=MagicMock(),
        processor=MagicMock(),
        runtime=DFlashRuntime(
            drafter=MagicMock(),
            kind="dflash",
            drafter_repo="z-lab/Qwen3.5-27B-DFlash",
        ),
        served_model_name="qwen3.5-27b-8bit",
        default_max_tokens=512,
        cors_origins=["*"],
        tool_call_parser="hermes",
        reasoning_parser_name="qwen3",
    )

    with TestClient(app).stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "qwen3.5-27b-8bit",
            "messages": [{"role": "user", "content": "weather in Paris?"}],
            "stream": True,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    },
                }
            ],
        },
    ) as response:
        assert response.status_code == 200
        body = b"".join(response.iter_bytes()).decode()

    payloads = [
        json.loads(line.removeprefix("data: "))
        for line in body.splitlines()
        if line.startswith("data: {")
    ]
    tool_deltas = [
        choice["delta"]["tool_calls"]
        for payload in payloads
        for choice in payload.get("choices", [])
        if choice.get("delta", {}).get("tool_calls")
    ]
    assert tool_deltas[0][0]["function"]["name"] == "get_weather"
    assert '"city": "Paris"' in tool_deltas[0][0]["function"]["arguments"]
    assert payloads[-1]["choices"][0]["finish_reason"] == "tool_calls"
    assert "<tool_call>" not in body


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


def test_chat_completions_rejects_logprobs() -> None:
    """DFlash v1 doesn't surface per-token logprobs. Silent-drop would let
    callers think they got logprobs back. Reject with a 400 instead."""
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
            "logprobs": True,
        },
    )
    assert r.status_code == 400
    # D-ANTHRO-VALIDATION F11: canonical envelope shape — see comment
    # on test_chat_completions_rejects_tools above.
    assert "logprobs" in r.json()["error"]["message"].lower()


def test_chat_completions_rejects_response_format() -> None:
    """DFlash v1 has no structured-output enforcement. Silent-drop would
    mean a JSON-schema request gets free-form text with no surfaced error."""
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
            "response_format": {"type": "json_object"},
        },
    )
    assert r.status_code == 400
    # D-ANTHRO-VALIDATION F11: canonical envelope shape — see comment
    # on test_chat_completions_rejects_tools above.
    assert "response_format" in r.json()["error"]["message"].lower()


def _capture_enable_thinking(monkeypatch, *, no_thinking: bool, request_body: dict):
    """Drive a chat request through ``_build_app`` and capture the
    ``enable_thinking`` kwarg the route passed to ``apply_chat_template``.

    Skip actually running the model — the route short-circuits as soon
    as it tries to build the gen_kwargs, which is fine since we only
    care about the chat-template render path.
    """
    from fastapi.testclient import TestClient

    from vllm_mlx.speculative.dflash.runtime import DFlashRuntime
    from vllm_mlx.speculative.dflash.server import _build_app

    captured: dict = {}

    import mlx_vlm.prompt_utils as _prompt_utils

    def _spy(processor, config, messages, **kw):
        captured.update(kw)
        return "stub prompt"

    monkeypatch.setattr(_prompt_utils, "apply_chat_template", _spy)

    # Stub the streaming generator so the request doesn't try to load
    # weights. We only need the route to reach _render_prompt, which is
    # *before* generation kicks off.
    import mlx_vlm as _mlx_vlm

    def _empty_gen(*a, **kw):
        if False:
            yield None  # pragma: no cover — generator shell

    monkeypatch.setattr(_mlx_vlm, "stream_generate", _empty_gen)

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
        default_max_tokens=64,
        cors_origins=["*"],
        no_thinking=no_thinking,
        reasoning_parser_name="qwen3",
    )
    client = TestClient(app)
    # Stream=True so the request reaches _render_prompt then exits via
    # the (now-empty) generator without needing a real mlx_vlm runtime.
    with client.stream("POST", "/v1/chat/completions", json=request_body) as resp:
        b"".join(resp.iter_bytes())
    return captured


@_skip_without_mlx_vlm
def test_no_thinking_server_flag_forces_enable_thinking_false(monkeypatch) -> None:
    """``--no-thinking`` server-side must force ``enable_thinking=False``
    on the chat template even when the request didn't ask for it. This
    is the v0.6.37 regression fix: DFlash hardcoded True regardless."""
    captured = _capture_enable_thinking(
        monkeypatch,
        no_thinking=True,
        request_body={
            "model": "qwen3.5-27b-8bit",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    )
    assert captured.get("enable_thinking") is False, (
        "--no-thinking must override the chat-template default; got "
        f"enable_thinking={captured.get('enable_thinking')!r}"
    )


@_skip_without_mlx_vlm
def test_request_enable_thinking_false_honored(monkeypatch) -> None:
    """Per-request ``enable_thinking=false`` body field must reach the
    chat template even when the server didn't set ``--no-thinking``."""
    captured = _capture_enable_thinking(
        monkeypatch,
        no_thinking=False,
        request_body={
            "model": "qwen3.5-27b-8bit",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "enable_thinking": False,
        },
    )
    assert captured.get("enable_thinking") is False


@_skip_without_mlx_vlm
def test_enable_thinking_defaults_false_without_reasoning_parser(monkeypatch) -> None:
    """DFlash must not leak unparsed thinking into normal response content."""
    captured = _capture_enable_thinking(
        monkeypatch,
        no_thinking=False,
        request_body={
            "model": "qwen3.5-27b-8bit",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    )
    assert captured.get("enable_thinking") is False


@_skip_without_mlx_vlm
def test_request_enable_thinking_true_honored(monkeypatch) -> None:
    """Callers can still explicitly opt in to the model's thinking output."""
    captured = _capture_enable_thinking(
        monkeypatch,
        no_thinking=False,
        request_body={
            "model": "qwen3.5-27b-8bit",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "enable_thinking": True,
        },
    )
    assert captured.get("enable_thinking") is True


@_skip_without_mlx_vlm
@pytest.mark.parametrize("stream", [False, True])
def test_explicit_thinking_is_parsed_into_reasoning_content(
    monkeypatch, stream: bool
) -> None:
    """DFlash opt-in thinking uses the standard Qwen reasoning parser."""
    import json

    import mlx_vlm
    import mlx_vlm.prompt_utils
    from fastapi.testclient import TestClient

    from vllm_mlx.speculative.dflash.runtime import DFlashRuntime
    from vllm_mlx.speculative.dflash.server import _build_app

    monkeypatch.setattr(
        mlx_vlm.prompt_utils,
        "apply_chat_template",
        lambda *args, **kwargs: "rendered prompt",
    )
    monkeypatch.setattr(
        mlx_vlm,
        "generate",
        lambda *args, **kwargs: SimpleNamespace(
            text="<think>Need one addition.</think>Four.",
            prompt_tokens=5,
            generation_tokens=7,
        ),
    )

    def _stream(*args, **kwargs):
        for index, text in enumerate(
            ("<think>", "Need one addition.", "</think>", "Four."), start=1
        ):
            yield SimpleNamespace(
                text=text,
                token=index,
                prompt_tokens=5,
                generation_tokens=index,
            )

    monkeypatch.setattr(mlx_vlm, "stream_generate", _stream)
    app = _build_app(
        model=MagicMock(),
        processor=MagicMock(),
        runtime=DFlashRuntime(
            drafter=MagicMock(),
            kind="dflash",
            drafter_repo="z-lab/Qwen3.5-27B-DFlash",
        ),
        served_model_name="qwen3.5-27b-8bit",
        default_max_tokens=64,
        cors_origins=["*"],
        reasoning_parser_name="qwen3",
    )
    request_body = {
        "model": "qwen3.5-27b-8bit",
        "messages": [{"role": "user", "content": "What is 2 + 2?"}],
        "enable_thinking": True,
        "stream": stream,
    }

    if not stream:
        response = TestClient(app).post(
            "/v1/chat/completions",
            json=request_body,
        )
        assert response.status_code == 200, response.text
        message = response.json()["choices"][0]["message"]
        assert message["reasoning_content"] == "Need one addition."
        assert message["content"] == "Four."
        return

    with TestClient(app).stream(
        "POST", "/v1/chat/completions", json=request_body
    ) as response:
        assert response.status_code == 200
        body = b"".join(response.iter_bytes()).decode()
    payloads = [
        json.loads(line.removeprefix("data: "))
        for line in body.splitlines()
        if line.startswith("data: {")
    ]
    deltas = [
        choice["delta"] for payload in payloads for choice in payload.get("choices", [])
    ]
    assert "".join(delta.get("reasoning_content", "") for delta in deltas) == (
        "Need one addition."
    )
    assert "".join(delta.get("content", "") for delta in deltas) == "Four."
    assert "<think>" not in body


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
    # The final delta block must carry the error block alongside an
    # OpenAI-spec-compliant finish_reason. ``"error"`` is NOT in the
    # OpenAI ChatCompletion finish_reason literal set; aborts use
    # ``"length"`` so spec-validating clients (openai-python, pydantic-ai)
    # can parse the response. The ``error`` block carries the diagnostic
    # details. v0.6.63 onboarding sweep finding #6.
    assert '"finish_reason": "length"' in body
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
    # finish_reason is OpenAI-spec-compliant ``"length"`` (see the
    # generator-exception test above for full rationale).
    assert "data: [DONE]" in body, (
        f"constructor-time crash must still terminate the stream; got:\n{body}"
    )
    assert '"finish_reason": "length"' in body
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
# Adversarial concurrency / liveness regressions (PR #1109 codex review).
# These pin the five BLOCKING findings so a future refactor cannot silently
# reintroduce a lock leak, an unstarted-but-charged GPU job, a path-dependent
# zero-timeout, or an admission gate that runs after body parse.
# =============================================================================


def test_dflash_stream_cancel_during_construction_releases_lock(monkeypatch) -> None:
    """F3: cancelling while the generator is still being CONSTRUCTED must
    not leak ``_dflash_lock``.

    Reproduces the codex finding: with no deadline, the worker future is
    awaited bare; a client cancellation that lands in the tiny window while
    ``_make_gen`` is in flight used to unwind through ``__aexit__`` →
    ``_capture_generator`` → ``future.result()``, which raised
    ``CancelledError`` (a ``BaseException``, not caught by ``except
    Exception``) and skipped the lock release entirely. After the fix the
    lease's ``__aexit__`` always drives cleanup to completion.
    """
    import asyncio
    import sys
    import threading
    import types

    from vllm_mlx.api.models import ChatCompletionRequest, Message
    from vllm_mlx.speculative.dflash import server as srv

    in_construction = threading.Event()
    allow_construction_to_finish = threading.Event()

    class _EmptyGenerator:
        def __next__(self):
            raise StopIteration

        def close(self):
            pass

    def blocking_stream_generate(*_args, **_kwargs):
        # Block INSIDE generator construction so the cancellation can land
        # while the ``_make_gen`` worker future is still in flight.
        in_construction.set()
        assert allow_construction_to_finish.wait(timeout=5)
        return _EmptyGenerator()

    fake_mlx_vlm = types.ModuleType("mlx_vlm")
    fake_mlx_vlm.stream_generate = blocking_stream_generate
    monkeypatch.setitem(sys.modules, "mlx_vlm", fake_mlx_vlm)

    request = ChatCompletionRequest(
        model="qwen3.5-27b-8bit",
        messages=[Message(role="user", content="hi")],
        stream=True,
    )

    async def exercise() -> None:
        admission = srv._DFlashAdmission(max_concurrent_requests=1)
        reservation = admission.reserve()
        stream = srv._stream_completion(
            prompt="prompt",
            request=request,
            served_model_name="qwen3.5-27b-8bit",
            gen_kwargs={},
            model=MagicMock(),
            processor=MagicMock(),
            timeout=0,  # no deadline -> bare-await construction path
            admission_reservation=reservation,
        )
        # Role marker first; the producer then starts constructing the gen.
        await anext(stream)
        assert await asyncio.to_thread(in_construction.wait, 1)

        # Cancel mid-construction, then let construction finish so the
        # deferred cleanup can run the generator close + lock release.
        await stream.aclose()
        allow_construction_to_finish.set()

        # The lock and the admission slot must both be freed once the
        # detached worker unwinds — poll briefly for the deferred cleanup.
        for _ in range(200):
            if not srv._dflash_lock.locked() and admission._reservations == 0:
                break
            await asyncio.sleep(0.01)
        assert not srv._dflash_lock.locked(), "F3 regression: lock leaked"
        assert admission._reservations == 0, "F3 regression: admission slot leaked"

    srv._dflash_lock = asyncio.Lock()
    try:
        asyncio.run(exercise())
    finally:
        srv._dflash_lock = asyncio.Lock()


def test_dflash_nonstream_expired_deadline_skips_gpu_work(monkeypatch) -> None:
    """F5: a non-stream request whose deadline expired while acquiring the
    serial lock must NOT submit GPU work.

    Deterministic construction (codex round-2 #4): the lock is FREE, so
    acquisition succeeds immediately — this test does NOT rely on the
    lock-acquire ``wait_for`` timing out (which would leave the F5
    post-acquisition recheck untested and pass even if it were deleted).
    Instead a fake monotonic clock jumps PAST the deadline between the
    deadline computation and the post-acquire recheck, so acquisition
    succeeds but the recheck sees an already-expired deadline and must
    raise 504 BEFORE submitting the executor job. ``generate`` must never
    be called.
    """
    import asyncio
    import sys
    import types

    from fastapi import HTTPException

    from vllm_mlx.speculative.dflash import server as srv

    generate_calls = [0]

    def spy_generate(*_args, **_kwargs):
        generate_calls[0] += 1
        return SimpleNamespace(text="x", prompt_tokens=1, generation_tokens=1)

    fake_mlx_vlm = types.ModuleType("mlx_vlm")
    fake_mlx_vlm.generate = spy_generate
    monkeypatch.setitem(sys.modules, "mlx_vlm", fake_mlx_vlm)

    async def exercise() -> None:
        admission = srv._DFlashAdmission(max_concurrent_requests=1)
        reservation = admission.reserve()

        loop = asyncio.get_running_loop()
        real_time = loop.time
        # Deterministic clock: ``loop.time`` returns ``real_now + advance``.
        # The lock is FREE, so acquisition succeeds instantly (this does NOT
        # rely on the acquire ``wait_for`` timing out — the flaw codex #4
        # flagged). We advance the clock PAST the deadline the instant the
        # lock is acquired, via an ``acquire`` wrapper, so that:
        #   * the deadline anchor + pre-acquire remaining-time check both see
        #     ``advance == 0`` (plenty of budget), and
        #   * the POST-acquire remaining-time recheck (the actual F5 guard)
        #     sees ``advance == +10s`` → deadline blown → 504 before submit.
        advance = {"v": 0.0}
        monkeypatch.setattr(loop, "time", lambda: real_time() + advance["v"])

        real_acquire = srv._dflash_lock.acquire

        async def acquire_then_blow_deadline() -> bool:
            got = await real_acquire()
            advance["v"] = 10.0  # deadline (anchor + 5s) is now in the past
            return got

        monkeypatch.setattr(srv._dflash_lock, "acquire", acquire_then_blow_deadline)

        with pytest.raises(HTTPException) as excinfo:
            await srv._non_stream_completion(
                prompt="prompt",
                request=MagicMock(),
                served_model_name="qwen3.5-27b-8bit",
                gen_kwargs={},
                model=MagicMock(),
                processor=MagicMock(),
                timeout=5.0,  # positive deadline; blown right after acquire
                admission_reservation=reservation,
            )
        monkeypatch.setattr(loop, "time", real_time)

        assert excinfo.value.status_code == 504
        # The core F5 assertion: no GPU work started for an expired request.
        assert generate_calls[0] == 0, "F5 regression: GPU work started after deadline"
        # Nothing was submitted, so no detached worker holds the lock.
        assert not srv._dflash_lock.locked()
        reservation.release()
        assert admission._reservations == 0

    srv._dflash_lock = asyncio.Lock()
    try:
        asyncio.run(exercise())
    finally:
        srv._dflash_lock = asyncio.Lock()


def test_dflash_zero_timeout_is_no_deadline_on_both_paths(monkeypatch) -> None:
    """F4: ``timeout <= 0`` means "no deadline" consistently on the stream
    AND non-stream paths — not "unlimited" on one and "immediate 504" on
    the other.
    """
    import asyncio
    import sys
    import types

    from vllm_mlx.api.models import ChatCompletionRequest, Message
    from vllm_mlx.speculative.dflash import server as srv

    class _OneChunk:
        text = "hi"
        generation_tokens = 1
        prompt_tokens = 2
        token = 7

    def _gen():
        yield _OneChunk()

    fake_mlx_vlm = types.ModuleType("mlx_vlm")
    fake_mlx_vlm.stream_generate = lambda *a, **kw: _gen()
    fake_mlx_vlm.generate = lambda *a, **kw: SimpleNamespace(
        text="hi", prompt_tokens=2, generation_tokens=1
    )
    monkeypatch.setitem(sys.modules, "mlx_vlm", fake_mlx_vlm)

    request = ChatCompletionRequest(
        model="qwen3.5-27b-8bit",
        messages=[Message(role="user", content="hi")],
        stream=True,
    )

    async def exercise() -> None:
        # Stream path with timeout=0 must generate normally (no immediate
        # timeout SSE) and finish cleanly.
        stream = srv._stream_completion(
            prompt="p",
            request=request,
            served_model_name="qwen3.5-27b-8bit",
            gen_kwargs={"max_tokens": 8},
            model=MagicMock(),
            processor=MagicMock(),
            timeout=0,
        )
        body = b"".join([chunk async for chunk in stream]).decode()
        assert "DFlash stream timed out" not in body
        assert '"content": "hi"' in body
        assert "data: [DONE]" in body
        assert not srv._dflash_lock.locked()

        # Non-stream path with timeout=0 must NOT return an immediate 504 —
        # it must run to completion (the pre-fix bug returned 504 here).
        resp = await srv._non_stream_completion(
            prompt="p",
            request=MagicMock(),
            served_model_name="qwen3.5-27b-8bit",
            gen_kwargs={"max_tokens": 8},
            model=MagicMock(),
            processor=MagicMock(),
            timeout=0,
        )
        assert resp.choices[0].message.content == "hi"
        assert not srv._dflash_lock.locked()

    srv._dflash_lock = asyncio.Lock()
    try:
        asyncio.run(exercise())
    finally:
        srv._dflash_lock = asyncio.Lock()


def test_dflash_stream_backpressure_does_not_hold_lock(monkeypatch) -> None:
    """F2: a slow/stalled consumer must not pin ``_dflash_lock``.

    Generation runs in a producer task that owns the lease; the consumer
    only shuttles bytes to the socket. So once the (short) generation
    finishes, the lock is released even if the consumer never reads another
    chunk. We prove it by draining exactly the role marker, then — without
    reading any generated chunk — waiting for the lock to free while the
    producer completes independently.
    """
    import asyncio
    import sys
    import types

    from vllm_mlx.api.models import ChatCompletionRequest, Message
    from vllm_mlx.speculative.dflash import server as srv

    class _TwoChunks:
        def __init__(self):
            self._n = 0

        def __iter__(self):
            return self

        def __next__(self):
            self._n += 1
            if self._n > 2:
                raise StopIteration
            return SimpleNamespace(
                text=f"tok{self._n}",
                generation_tokens=self._n,
                prompt_tokens=3,
                token=100 + self._n,
            )

        def close(self):
            pass

    fake_mlx_vlm = types.ModuleType("mlx_vlm")
    fake_mlx_vlm.stream_generate = lambda *a, **kw: _TwoChunks()
    monkeypatch.setitem(sys.modules, "mlx_vlm", fake_mlx_vlm)

    request = ChatCompletionRequest(
        model="qwen3.5-27b-8bit",
        messages=[Message(role="user", content="hi")],
        stream=True,
    )

    async def exercise() -> None:
        admission = srv._DFlashAdmission(max_concurrent_requests=1)
        reservation = admission.reserve()
        stream = srv._stream_completion(
            prompt="p",
            request=request,
            served_model_name="qwen3.5-27b-8bit",
            gen_kwargs={"max_tokens": 8},
            model=MagicMock(),
            processor=MagicMock(),
            timeout=30,
            admission_reservation=reservation,
        )
        # Read ONLY the role marker, then stop reading (simulate a stalled
        # socket). The producer keeps running to completion in the
        # background and must release the lock + slot without us reading
        # any more chunks.
        assert b'"role": "assistant"' in await anext(stream)
        for _ in range(300):
            if not srv._dflash_lock.locked() and admission._reservations == 0:
                break
            await asyncio.sleep(0.01)
        assert not srv._dflash_lock.locked(), (
            "F2 regression: lock held while consumer is not reading"
        )
        assert admission._reservations == 0, (
            "F2 regression: slot held under backpressure"
        )
        # Drain the rest so the generator closes cleanly.
        async for _ in stream:
            pass

    srv._dflash_lock = asyncio.Lock()
    try:
        asyncio.run(exercise())
    finally:
        srv._dflash_lock = asyncio.Lock()


def test_dflash_stream_backpressure_delivers_terminal_notice(monkeypatch) -> None:
    """codex round-4 #3: hitting the server-side backpressure cap must NOT
    truncate the stream silently — the client gets an explicit terminal error
    frame + ``[DONE]``.

    Previously a full queue past ``_STREAM_BACKPRESSURE_TIMEOUT_SECONDS`` raised
    ``_DFlashClientGoneError``, which ``_drive_producer`` swallowed with no
    terminator. A client that was merely SLOW (not gone) then saw an
    unexplained truncation. Now the producer converts that abort into a
    ``finish_reason="length"`` error notice + ``[DONE]`` so a resuming client
    always learns why generation stopped — even with ``timeout=0`` (no request
    deadline), where only this backpressure cap applies.
    """
    import asyncio
    import sys
    import types

    from vllm_mlx.api.models import ChatCompletionRequest, Message
    from vllm_mlx.speculative.dflash import server as srv

    # Tiny queue + tiny backpressure window so the test triggers the cap fast.
    monkeypatch.setattr(srv, "_STREAM_QUEUE_MAXSIZE", 2)
    monkeypatch.setattr(srv, "_STREAM_BACKPRESSURE_TIMEOUT_SECONDS", 0.05)

    class _ManyChunks:
        def __init__(self):
            self._n = 0

        def __iter__(self):
            return self

        def __next__(self):
            self._n += 1
            if self._n > 50:
                raise StopIteration
            return SimpleNamespace(
                text=f"tok{self._n}",
                generation_tokens=self._n,
                prompt_tokens=3,
                token=100 + self._n,
            )

        def close(self):
            pass

    fake_mlx_vlm = types.ModuleType("mlx_vlm")
    fake_mlx_vlm.stream_generate = lambda *a, **kw: _ManyChunks()
    monkeypatch.setitem(sys.modules, "mlx_vlm", fake_mlx_vlm)

    request = ChatCompletionRequest(
        model="qwen3.5-27b-8bit",
        messages=[Message(role="user", content="hi")],
        stream=True,
    )

    async def exercise() -> None:
        admission = srv._DFlashAdmission(max_concurrent_requests=1)
        reservation = admission.reserve()
        stream = srv._stream_completion(
            prompt="p",
            request=request,
            served_model_name="qwen3.5-27b-8bit",
            gen_kwargs={"max_tokens": 100},
            model=MagicMock(),
            processor=MagicMock(),
            timeout=0,  # NO request deadline — only the backpressure cap.
            admission_reservation=reservation,
        )
        # A SLOW-but-connected client: read the role marker, pause long enough
        # to trip the content-frame backpressure cap (0.05s) on the full
        # 2-slot queue, then keep draining. The producer aborts generation on
        # backpressure but must still deliver a terminal notice + [DONE]; the
        # continued draining gives those terminal frames queue room to land.
        assert b'"role": "assistant"' in await anext(stream)
        await asyncio.sleep(0.15)  # > backpressure window → content put aborts
        rest_frames: list[bytes] = []
        async for chunk in stream:
            rest_frames.append(chunk)
            await asyncio.sleep(0.005)  # keep draining, slowly
        rest = b"".join(rest_frames).decode()
        assert "server-side backpressure" in rest, (
            "codex #3 regression: backpressure abort truncated the stream "
            "without a terminal error notice"
        )
        assert '"finish_reason": "length"' in rest
        assert "data: [DONE]" in rest
        # codex round-5 #3: the [DONE] terminator MUST be last — after the
        # final finish_reason/error frame — even when both spill to the
        # direct-emit path. A regressed ordering would deliver [DONE] first.
        assert rest.index('"finish_reason": "length"') < rest.index("data: [DONE]"), (
            "codex #3 ordering regression: [DONE] delivered before final frame"
        )
        # [DONE] is the very last non-empty frame.
        nonempty = [f for f in rest.split("\n\n") if f.strip()]
        assert nonempty[-1].strip() == "data: [DONE]", (
            f"codex #3 ordering regression: last frame was {nonempty[-1]!r}"
        )
        # Lock + slot released after the abort.
        for _ in range(300):
            if not srv._dflash_lock.locked() and admission._reservations == 0:
                break
            await asyncio.sleep(0.01)
        assert not srv._dflash_lock.locked()
        assert admission._reservations == 0

    srv._dflash_lock = asyncio.Lock()
    try:
        asyncio.run(exercise())
    finally:
        srv._dflash_lock = asyncio.Lock()


def test_dflash_hung_generator_close_does_not_block_terminal_sse(monkeypatch) -> None:
    """codex round-6 #2: a hung ``generator.close()`` must NOT block the
    terminal SSE (final frame + [DONE]).

    ``__aexit__`` closes the generator on the serial worker. If that close
    hangs (a runaway mlx-vlm GPU-teardown finally block), awaiting it inline
    would stall the producer's terminal frames indefinitely. The lease now
    bounds the close wait and DETACHES a slow close, retaining the lock/slot
    until it finishes while letting the response terminate promptly. Here the
    generator errors mid-stream (so we hit the terminal path) and its
    ``close()`` blocks; the client must still get [DONE] quickly.
    """
    import asyncio
    import sys
    import threading
    import types

    from vllm_mlx.api.models import ChatCompletionRequest, Message
    from vllm_mlx.speculative.dflash import server as srv

    # Tiny close grace so the test doesn't wait the real 5s to detach.
    monkeypatch.setattr(srv, "_STREAM_GENERATOR_CLOSE_GRACE_SECONDS", 0.05)

    close_called = threading.Event()
    release_close = threading.Event()

    class _HangOnCloseGenerator:
        def __init__(self):
            self._n = 0

        def __iter__(self):
            return self

        def __next__(self):
            self._n += 1
            if self._n == 1:
                return SimpleNamespace(
                    text="tok1", generation_tokens=1, prompt_tokens=3, token=101
                )
            # Error mid-stream → terminal error path → __aexit__ closes gen.
            raise RuntimeError("boom")

        def close(self):
            close_called.set()
            # Block until the test explicitly releases — simulates a hung close.
            release_close.wait(timeout=5)

    fake_mlx_vlm = types.ModuleType("mlx_vlm")
    fake_mlx_vlm.stream_generate = lambda *a, **kw: _HangOnCloseGenerator()
    monkeypatch.setitem(sys.modules, "mlx_vlm", fake_mlx_vlm)

    request = ChatCompletionRequest(
        model="qwen3.5-27b-8bit",
        messages=[Message(role="user", content="hi")],
        stream=True,
    )

    async def exercise() -> None:
        admission = srv._DFlashAdmission(max_concurrent_requests=1)
        reservation = admission.reserve()
        stream = srv._stream_completion(
            prompt="p",
            request=request,
            served_model_name="qwen3.5-27b-8bit",
            gen_kwargs={"max_tokens": 8},
            model=MagicMock(),
            processor=MagicMock(),
            timeout=0,
            admission_reservation=reservation,
        )
        started = asyncio.get_running_loop().time()
        body = b"".join([chunk async for chunk in stream]).decode()
        elapsed = asyncio.get_running_loop().time() - started

        # The terminal [DONE] arrived even though close() is still blocked.
        assert "data: [DONE]" in body
        assert await asyncio.to_thread(close_called.wait, 1)
        # The stream terminated on the ~0.05s detach grace, NOT the 5s close.
        assert elapsed < 1.0, (
            f"codex #6 regression: terminal SSE blocked on hung close ({elapsed:.2f}s)"
        )
        # Lock/slot stay HELD until the detached close finishes.
        assert srv._dflash_lock.locked()
        assert admission._reservations == 1
        # Now let the close complete; lock + slot release from its callback.
        release_close.set()
        for _ in range(300):
            if not srv._dflash_lock.locked() and admission._reservations == 0:
                break
            await asyncio.sleep(0.01)
        assert not srv._dflash_lock.locked()
        assert admission._reservations == 0

    srv._dflash_lock = asyncio.Lock()
    try:
        asyncio.run(exercise())
    finally:
        release_close.set()
        srv._dflash_lock = asyncio.Lock()


def test_dflash_admission_reserved_before_body_parse(monkeypatch) -> None:
    """F1: the ASGI admission middleware rejects with 503 BEFORE FastAPI
    parses the request body.

    We fill the single admission slot, then post a body that is intentionally
    invalid JSON. If admission ran only inside the endpoint (post-parse),
    FastAPI would 400 on the bad JSON first; with the ASGI gate the 503 wins
    because it fires before the body is read.
    """
    from fastapi.testclient import TestClient

    from vllm_mlx.speculative.dflash.runtime import DFlashRuntime
    from vllm_mlx.speculative.dflash.server import _build_app

    app = _build_app(
        model=MagicMock(),
        processor=MagicMock(),
        runtime=DFlashRuntime(
            drafter=MagicMock(),
            kind="dflash",
            drafter_repo="z-lab/Qwen3.5-27B-DFlash",
        ),
        served_model_name="qwen3.5-27b-8bit",
        default_max_tokens=64,
        cors_origins=[],
        max_concurrent_requests=1,
    )
    # Occupy the only slot so the incoming request must be rejected.
    reservation = app.state.dflash_admission.reserve()
    try:
        response = TestClient(app).post(
            "/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            content=b'{"model": "x", "messages": [ THIS IS NOT VALID JSON',
        )
    finally:
        reservation.release()

    # 503 (admission) — NOT 400 (body parse) — proves the gate ran first.
    assert response.status_code == 503, response.text
    assert response.headers.get("Retry-After") == "1"
    assert response.json()["error"]["code"] == "at_capacity"


def test_dflash_slow_prompt_render_is_charged_against_deadline(monkeypatch) -> None:
    """codex round-4 #4: prompt rendering time is charged against the request
    ``timeout``, and rendering is offloaded off the event loop.

    The pre-fix endpoint rendered the prompt SYNCHRONOUSLY before either
    completion helper established its deadline, so an expensive chat template
    could block the event loop and blow past the configured ``timeout`` with
    no enforcement. Here we make ``_render_prompt`` take far longer than the
    request ``timeout`` and assert the request is rejected with 504 — proving
    the deadline now spans rendering. (It also could not have surfaced a 504
    at all if rendering still ran unbounded on the event loop.)
    """
    # A render that overruns the 0.05s request budget. Runs on the separate
    # render pool (offloaded), so the event loop stays free to enforce the
    # deadline via ``asyncio.wait_for``. Uses an event so the test controls
    # exactly when the (uncancellable, already-running) render finishes.
    import threading
    import time

    from fastapi.testclient import TestClient

    from vllm_mlx.speculative.dflash import server as srv
    from vllm_mlx.speculative.dflash.runtime import DFlashRuntime
    from vllm_mlx.speculative.dflash.server import _build_app

    render_may_finish = threading.Event()

    def _slow_render(*_args, **_kwargs) -> str:
        render_may_finish.wait(timeout=5)
        return "rendered"

    monkeypatch.setattr(srv, "_render_prompt", _slow_render)

    app = _build_app(
        model=MagicMock(),
        processor=MagicMock(),
        runtime=DFlashRuntime(
            drafter=MagicMock(),
            kind="dflash",
            drafter_repo="z-lab/Qwen3.5-27B-DFlash",
        ),
        served_model_name="qwen3.5-27b-8bit",
        default_max_tokens=64,
        cors_origins=[],
        max_concurrent_requests=1,
    )
    client = TestClient(app)
    admission = app.state.dflash_admission

    try:
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3.5-27b-8bit",
                "messages": [{"role": "user", "content": "hi"}],
                "timeout": 0.05,
            },
        )
        assert r.status_code == 504, r.text
        # codex round-7 #2: the render is STILL running (uncancellable), so the
        # admission slot is HELD — not freed — until it drains. This is what
        # stops a burst of cancelled/timed-out requests from piling up renders
        # on the single-worker pool while capacity is handed back out.
        assert admission._reservations == 1, (
            "codex #7 regression: slot freed while the render was still running"
        )
    finally:
        # Let the render finish; the deferred callback then releases the slot.
        render_may_finish.set()

    for _ in range(300):
        if admission._reservations == 0:
            break
        time.sleep(0.01)
    assert admission._reservations == 0, (
        "codex #4/#7 regression: slot not released after the render drained"
    )


def test_dflash_unauthenticated_request_does_not_reserve_slot() -> None:
    """codex round-3 #3: auth is enforced BEFORE the admission slot is
    reserved. A missing / wrong API key must return 401 and must NOT occupy
    a DFlash slot — otherwise an unauthenticated client could exhaust every
    slot (e.g. via slow body uploads) and deny service to authorized callers.
    """
    from fastapi.testclient import TestClient

    from vllm_mlx.speculative.dflash.runtime import DFlashRuntime
    from vllm_mlx.speculative.dflash.server import _build_app

    model = MagicMock()
    model.config = MagicMock()
    app = _build_app(
        model=model,
        processor=MagicMock(),
        runtime=DFlashRuntime(
            drafter=MagicMock(),
            kind="dflash",
            drafter_repo="z-lab/Qwen3.5-27B-DFlash",
        ),
        served_model_name="qwen3.5-27b-8bit",
        default_max_tokens=64,
        cors_origins=[],
        api_key="secret",
        max_concurrent_requests=1,
    )
    client = TestClient(app)
    admission = app.state.dflash_admission

    body = {
        "model": "qwen3.5-27b-8bit",
        "messages": [{"role": "user", "content": "hi"}],
    }

    # No key → 401 and no slot consumed.
    r = client.post("/v1/chat/completions", json=body)
    assert r.status_code == 401, r.text
    assert admission._reservations == 0, "codex #3 regression: 401 reserved a slot"

    # Wrong key → 401 and no slot consumed.
    r = client.post(
        "/v1/chat/completions", headers={"Authorization": "Bearer nope"}, json=body
    )
    assert r.status_code == 401, r.text
    assert admission._reservations == 0, (
        "codex #3 regression: wrong key reserved a slot"
    )

    # A slot held externally must NOT block the pre-admission 401 — the auth
    # rejection happens before the admission gate is even consulted.
    held = admission.reserve()
    try:
        r = client.post("/v1/chat/completions", json=body)
        assert r.status_code == 401, r.text
    finally:
        held.release()

    # codex round-5 #5: the 401 must carry the RFC 6750 bearer challenge.
    r = client.post("/v1/chat/completions", json=body)
    assert r.status_code == 401, r.text
    assert r.headers.get("WWW-Authenticate") == "Bearer", (
        "codex #5 regression: 401 dropped the WWW-Authenticate: Bearer "
        "challenge required for bearer-protected resources"
    )

    # codex round-6 #1: the /v1/models route (which uses a route dependency,
    # not the chat ASGI middleware) must ALSO emit the challenge on 401.
    r = client.get("/v1/models")
    assert r.status_code == 401, r.text
    assert r.headers.get("WWW-Authenticate") == "Bearer", (
        "codex #6 regression: /v1/models 401 dropped WWW-Authenticate: Bearer"
    )
    # And a correct key still passes.
    r = client.get("/v1/models", headers={"Authorization": "Bearer secret"})
    assert r.status_code == 200, r.text


def test_dflash_rate_limited_request_does_not_reserve_slot() -> None:
    """codex round-3 #3: rate-limit rejection also happens BEFORE reserving,
    and the limiter is consulted exactly once per request (no double-count
    from a leftover route dependency)."""
    from fastapi.testclient import TestClient

    from vllm_mlx.middleware.auth import configure_rate_limiter
    from vllm_mlx.speculative.dflash.runtime import DFlashRuntime
    from vllm_mlx.speculative.dflash.server import _build_app

    model = MagicMock()
    model.config = MagicMock()
    app = _build_app(
        model=model,
        processor=MagicMock(),
        runtime=DFlashRuntime(
            drafter=MagicMock(),
            kind="dflash",
            drafter_repo="z-lab/Qwen3.5-27B-DFlash",
        ),
        served_model_name="qwen3.5-27b-8bit",
        default_max_tokens=64,
        cors_origins=[],
        rate_limit=1,
        max_concurrent_requests=1,
    )
    try:
        client = TestClient(app)
        admission = app.state.dflash_admission

        # Empty messages → 400, but the request still counts as the single
        # allowed request (limiter consulted ONCE, not twice).
        empty = {"model": "qwen3.5-27b-8bit", "messages": []}
        r1 = client.post("/v1/chat/completions", json=empty)
        assert r1.status_code == 400, r1.text

        # codex round-4 #5: prove rate-limiting runs BEFORE admission by
        # occupying the *only* admission slot before the over-limit request.
        # A correct impl rejects with 429 (rate-limit checked first, no slot
        # touched). A regressed impl that reserves-first would instead find
        # the gate full and return 503 — or, worse, reserve + then release,
        # masking a pre-admission DoS. Asserting 429 (NOT 503) with the slot
        # held closes that hole; the old "reservations == 0 afterwards"
        # assertion alone could not tell reserve-then-release from
        # never-reserve.
        held = admission.reserve()
        try:
            assert admission._reservations == 1
            r2 = client.post("/v1/chat/completions", json=empty)
            assert r2.status_code == 429, (
                f"codex #5 regression: over-limit request returned "
                f"{r2.status_code} (expected 429 from pre-admission "
                f"rate-limit, not 503 from the full gate)"
            )
            assert r2.json()["error"]["code"] == "rate_limit_exceeded"
            # The externally held slot is the ONLY reservation; the rejected
            # request must not have minted (or transiently minted) another.
            assert admission._reservations == 1, (
                "codex #5 regression: 429 request touched the admission gate"
            )
        finally:
            held.release()
        assert admission._reservations == 0
    finally:
        # The rate limiter is a process-global singleton; leaving it enabled
        # would pollute every later test that exercises admission/rate paths
        # (e.g. tests/test_embeddings_timeout_admission.py). Reset it.
        configure_rate_limiter(0, enabled=False)


def test_dflash_stream_terminal_frame_delivered_on_timeout(monkeypatch) -> None:
    """codex round-3 #2: the deadline-bounded backpressure on content frames
    must NOT suppress the TERMINAL timeout/error + [DONE] frames. Even when
    the request deadline is already blown, the client must still receive the
    "timed out" notice and the stream terminator — otherwise it sees a
    truncated stream with no explanation.
    """
    import asyncio
    import sys
    import time
    import types

    from vllm_mlx.api.models import ChatCompletionRequest, Message
    from vllm_mlx.speculative.dflash import server as srv

    class _SlowGenerator:
        def __next__(self):
            time.sleep(0.05)
            raise StopIteration

        def close(self):
            pass

    fake_mlx_vlm = types.ModuleType("mlx_vlm")
    fake_mlx_vlm.stream_generate = lambda *_a, **_kw: _SlowGenerator()
    monkeypatch.setitem(sys.modules, "mlx_vlm", fake_mlx_vlm)

    request = ChatCompletionRequest(
        model="qwen3.5-27b-8bit",
        messages=[Message(role="user", content="hi")],
        stream=True,
    )

    async def exercise() -> None:
        stream = srv._stream_completion(
            prompt="p",
            request=request,
            served_model_name="qwen3.5-27b-8bit",
            gen_kwargs={"max_tokens": 8},
            model=MagicMock(),
            processor=MagicMock(),
            timeout=0.01,  # already blown by the time the terminal frames emit
        )
        body = b"".join([chunk async for chunk in stream]).decode()
        # Terminal frames survive the deadline bound.
        # codex round-6 #5: 0.01s renders as "0.010 seconds" (adaptive).
        assert "DFlash stream timed out after 0.010 seconds." in body
        assert "data: [DONE]" in body

    srv._dflash_lock = asyncio.Lock()
    try:
        asyncio.run(exercise())
    finally:
        srv._dflash_lock = asyncio.Lock()


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
