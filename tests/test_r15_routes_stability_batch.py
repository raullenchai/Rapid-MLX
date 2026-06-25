# SPDX-License-Identifier: Apache-2.0
"""R15 routes-stability batch — 5 small route-layer correctness fixes.

Bundle of 5 0.9-must-fix route bugs from the R15 dogfood triage,
shipped as one PR because they all sit on adjacent route surfaces
and share the same FastAPI TestClient scaffold.

Bug 1 (task #306, Sven B2): ``/healthz`` flips to 503 when the
server lifecycle reaches the drain window. Pre-fix the route 200'd
right up until process exit, so a load balancer / k8s readiness
probe couldn't distinguish "draining" from "healthy" and kept
sending new traffic into a tearing-down instance.

Bug 2 (task #307, Sven B1): ``rapid-mlx serve`` MUST exit non-zero
when the bind port is in use, so systemd / Docker / k8s auto-restart
loops detect the failure instead of treating it as a clean exit
("don't restart").

Bug 3 (task #309): ``/v1/responses`` MUST 400 on unknown roles in
``input[]`` (e.g. ``role="wizard"``) at the Pydantic layer with a
clean field-path message, instead of letting the unknown role
fall through ``responses_adapter._message_item_to_chat`` and 500
inside the Jinja chat template.

Bug 4 (task #310): ``/v1/completions`` MUST 400 when
``stream=true`` is combined with a list prompt. Pre-fix the
streaming branch only ever called the engine with ``prompts[0]``,
silently dropping data for prompts[1:N].

Bug 5 (task #311): ``/v1/chat/completions`` with
``logprobs=true, top_logprobs=None`` MUST return the sampled-token
logprob. Pre-fix the gate AND'd both fields, so the route silently
took the non-logprobs branch when the caller omitted ``top_logprobs``.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import textwrap
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Bug 1 (task #306): /healthz returns 503 during graceful drain
# ---------------------------------------------------------------------------


class TestHealthzDrainState:
    """``cfg.draining=True`` flips ``/healthz`` from 200 to 503."""

    def _make_app(self):
        from vllm_mlx.routes.health import probe_router

        app = FastAPI()
        app.include_router(probe_router)
        return app

    def _patch_config(self, **kwargs):
        from vllm_mlx.config import get_config

        cfg = get_config()
        originals = {}
        for k, v in kwargs.items():
            originals[k] = getattr(cfg, k)
            setattr(cfg, k, v)
        return originals

    def _restore_config(self, originals):
        from vllm_mlx.config import get_config

        cfg = get_config()
        for k, v in originals.items():
            setattr(cfg, k, v)

    def test_healthz_returns_200_when_not_draining(self):
        """Baseline: cfg.draining=False keeps /healthz on the healthy
        path (sanity check; no behaviour change for the common case)."""
        orig = self._patch_config(
            engine=None,
            mcp_manager=None,
            model_name="test-model",
            ready=True,
            draining=False,
        )
        try:
            app = self._make_app()
            client = TestClient(app)
            r = client.get("/healthz")
            assert r.status_code == 200, r.text
            assert r.json()["status"] == "healthy"
        finally:
            self._restore_config(orig)

    def test_healthz_returns_503_when_draining(self):
        """R15 Sven B2: cfg.draining=True flips /healthz to 503 so the
        load balancer / k8s readiness probe stops sending new traffic."""
        orig = self._patch_config(
            engine=None,
            mcp_manager=None,
            model_name="test-model",
            ready=True,
            draining=True,
        )
        try:
            app = self._make_app()
            client = TestClient(app)
            r = client.get("/healthz")
            assert r.status_code == 503, r.text
            # JSON body must carry a structured ``status: draining`` so
            # operators / dashboards parsing the body see the drain
            # state, not the bare FastAPI HTTPException envelope.
            body = r.json()
            assert body["status"] == "draining"
            assert body["ready"] is False
        finally:
            self._restore_config(orig)

    def test_healthz_drain_flip_does_not_break_livez(self):
        """``/livez`` is the k8s liveness probe ("is the process alive
        at all?"); a draining process is still alive, so /livez must
        stay 200 even when /healthz flips to 503. Without this split
        the orchestrator would SIGKILL a graceful-shutdown pod."""
        orig = self._patch_config(
            engine=None,
            mcp_manager=None,
            model_name="test-model",
            ready=True,
            draining=True,
        )
        try:
            app = self._make_app()
            client = TestClient(app)
            r = client.get("/livez")
            assert r.status_code == 200, r.text
            assert r.json()["status"] == "alive"
        finally:
            self._restore_config(orig)


class TestHealthzDrainStateFastPath:
    """The ASGI fast-path middleware MUST honor cfg.draining too —
    otherwise it diverges from the route handler under the perf-
    critical k8s-probe load (which is exactly when drain happens)."""

    def _patch_config(self, **kwargs):
        from vllm_mlx.config import get_config

        cfg = get_config()
        originals = {}
        for k, v in kwargs.items():
            originals[k] = getattr(cfg, k)
            setattr(cfg, k, v)
        return originals

    def _restore_config(self, originals):
        from vllm_mlx.config import get_config

        cfg = get_config()
        for k, v in originals.items():
            setattr(cfg, k, v)

    def test_fastpath_payload_builder_flips_to_503_when_draining(self):
        from vllm_mlx.middleware.probe_fastpath import _build_healthz_payload

        orig = self._patch_config(
            engine=None,
            mcp_manager=None,
            model_name="test-model",
            ready=True,
            draining=True,
        )
        try:
            status_code, body = _build_healthz_payload()
            assert status_code == 503
            assert b'"status":"draining"' in body
        finally:
            self._restore_config(orig)

    def test_fastpath_payload_builder_returns_200_when_not_draining(self):
        from vllm_mlx.middleware.probe_fastpath import _build_healthz_payload

        orig = self._patch_config(
            engine=None,
            mcp_manager=None,
            model_name="test-model",
            ready=True,
            draining=False,
        )
        try:
            status_code, body = _build_healthz_payload()
            assert status_code == 200
            assert b'"status":"healthy"' in body
        finally:
            self._restore_config(orig)


# ---------------------------------------------------------------------------
# Bug 2 (task #307): rapid-mlx serve exits non-zero on port collision
# ---------------------------------------------------------------------------


class TestServePortCollisionExitCode:
    """End-to-end subprocess test: spawn ``rapid-mlx serve`` against a
    port already bound by this test process and assert the child exits
    non-zero. The CLI's existing ``_port_preflight_or_die`` and
    ``_run_uvicorn`` paths both ``sys.exit(1)`` on EADDRINUSE — this
    test is the supervisor-side regression guard that pins the
    contract end-to-end (no monkeypatching uvicorn / no
    SystemExit interception)."""

    def test_serve_exits_nonzero_on_port_collision(self):
        # Bind a real loopback socket to claim the port. ``SO_REUSEADDR``
        # is intentionally NOT set on this holder: the whole point of
        # the test is to collide with a later bind that itself sets
        # SO_REUSEADDR. On macOS / Linux a second non-SO_REUSEPORT bind
        # on the same loopback (host, port) still fails with EADDRINUSE.
        holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        holder.bind(("127.0.0.1", 0))
        port = holder.getsockname()[1]
        holder.listen(1)
        try:
            # Drive the port preflight directly through python so the
            # test does not depend on a packaged ``rapid-mlx`` console
            # script, an editable install, or a model download.
            # ``_port_preflight_or_die`` is the EXACT helper
            # ``serve_command`` calls before any heavy boot work;
            # surfacing its exit code via a subprocess is the
            # supervisor-side contract.
            script = textwrap.dedent(
                f"""
                import sys
                from vllm_mlx.cli import _port_preflight_or_die
                _port_preflight_or_die("127.0.0.1", {port}, model="stub")
                # If preflight didn't exit, that's the bug — exit 0 here
                # so the test sees a passing subprocess and fails the
                # ``returncode != 0`` assertion.
                sys.exit(0)
                """
            )
            env = os.environ.copy()
            # Inherit PYTHONPATH so the spawned interpreter resolves
            # the in-worktree package, not whatever ``pip install``
            # cached system-wide.
            env["PYTHONPATH"] = (
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                + os.pathsep
                + env.get("PYTHONPATH", "")
            )
            proc = subprocess.run(
                [sys.executable, "-c", script],
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
            )
            assert proc.returncode != 0, (
                f"expected non-zero exit on port collision, got "
                f"returncode={proc.returncode}\nstdout={proc.stdout!r}\n"
                f"stderr={proc.stderr!r}"
            )
        finally:
            holder.close()


# ---------------------------------------------------------------------------
# Bug 3 (task #309): /v1/responses unknown role → 400
# ---------------------------------------------------------------------------


class TestResponsesUnknownRole:
    """Pydantic-layer rejection of unknown roles on ResponsesInputItem.

    The Pydantic validator raises ValidationError on construction, so
    the route layer (with the production ``RequestValidationError``
    handler) returns 400. Here we drive the validator directly so the
    test scope stays minimal (no route wiring, no engine, no auth).
    """

    def test_unknown_role_rejected_by_validator(self):
        from pydantic import ValidationError

        from vllm_mlx.api.responses_models import ResponsesInputItem

        with pytest.raises(ValidationError) as ei:
            ResponsesInputItem(
                type="message",
                role="wizard",
                content=[{"type": "input_text", "text": "hi"}],
            )
        msg = str(ei.value)
        # The error must name the allowed enum so the SDK consumer
        # can fix the request without grepping our source code.
        assert "wizard" in msg
        assert "role" in msg.lower()
        assert "user" in msg and "assistant" in msg

    @pytest.mark.parametrize(
        "role", ["user", "assistant", "system", "tool", "developer"]
    )
    def test_known_roles_accepted_by_validator(self, role):
        """Every documented Responses-API role passes validation."""
        from vllm_mlx.api.responses_models import ResponsesInputItem

        item = ResponsesInputItem(
            type="message",
            role=role,
            content=[{"type": "input_text", "text": "hi"}],
        )
        assert item.role == role

    def test_function_call_item_without_role_accepted(self):
        """``function_call`` / ``function_call_output`` / ``reasoning``
        items have no ``role`` on the wire — the validator must accept
        ``role=None`` so those item types still flow through."""
        from vllm_mlx.api.responses_models import ResponsesInputItem

        item = ResponsesInputItem(
            type="function_call",
            call_id="call_abc",
            name="my_tool",
            arguments="{}",
        )
        assert item.role is None

    def test_unknown_role_400_via_request_layer(self):
        """End-to-end at the ResponsesRequest layer: nested input[0]
        role validator must surface a Pydantic ValidationError with the
        ``input.0.role`` field path so the production route's
        validation handler renders a clean 400 with the bad field
        called out by name."""
        from pydantic import ValidationError

        from vllm_mlx.api.responses_models import ResponsesRequest

        with pytest.raises(ValidationError) as ei:
            ResponsesRequest(
                model="m",
                input=[
                    {
                        "type": "message",
                        "role": "wizard",
                        "content": [{"type": "input_text", "text": "hi"}],
                    }
                ],
            )
        # ``loc`` on the first error must point at the offending field.
        errs = ei.value.errors()
        assert any("role" in str(e.get("loc", ())) for e in errs), (
            f"expected error loc to reference 'role'; got {errs!r}"
        )


# ---------------------------------------------------------------------------
# Bug 4 (task #310): /v1/completions list-prompt + stream=true → 400
# ---------------------------------------------------------------------------


@pytest.fixture
def patched_config():
    """Patches the global cfg singleton for the test then restores."""
    from vllm_mlx.config import get_config

    cfg = get_config()
    saved: dict = {}

    def patch(**kwargs):
        for k, v in kwargs.items():
            saved.setdefault(k, getattr(cfg, k, None))
            setattr(cfg, k, v)

    yield patch

    for k, v in saved.items():
        setattr(cfg, k, v)


class _StubGenerationOutput:
    """Minimal ``GenerationOutput`` stand-in."""

    def __init__(self, text: str = " world", finish_reason: str = "stop"):
        self.text = text
        self.finish_reason = finish_reason
        self.completion_tokens = 4
        self.prompt_tokens = 1
        self.cached_tokens = 0


def _build_completions_app(patch_cfg, monkeypatch, *, engine_factory=None):
    """Wire a stub completions app with a MagicMock engine."""
    from vllm_mlx.routes import completions as comp_route

    app = FastAPI()
    app.include_router(comp_route.router)

    engine = engine_factory() if engine_factory else MagicMock()
    cap = getattr(engine, "supports_completion_logprobs", None)
    if not isinstance(cap, bool):
        engine.supports_completion_logprobs = (
            callable(getattr(engine, "stream_generate", None))
            and getattr(engine, "tokenizer", None) is not None
        )
    patch_cfg(
        engine=engine,
        model_name="stub-model",
        model_alias=None,
        model_path=None,
        model_registry=None,
        tool_call_parser=None,
        reasoning_parser=None,
        ready=True,
        api_key=None,
    )
    monkeypatch.setattr(comp_route, "get_engine", lambda *_a, **_kw: engine)
    monkeypatch.setattr(
        comp_route, "enforce_context_length_for_prompt", lambda *_a, **_kw: None
    )
    return TestClient(app, raise_server_exceptions=False), engine


class TestCompletionsListPromptStreaming:
    """``stream=true`` + list-prompt MUST 400 (no silent drop of
    prompts[1:N])."""

    def test_list_prompt_with_stream_true_rejected_with_400(
        self, patched_config, monkeypatch
    ):
        client, _ = _build_completions_app(patched_config, monkeypatch)
        r = client.post(
            "/v1/completions",
            json={
                "model": "stub-model",
                "prompt": ["a", "b", "c"],
                "stream": True,
            },
        )
        assert r.status_code == 400, r.text
        # ``detail`` here is the FastAPI HTTPException ``detail`` dict
        # (production servers run a RequestValidationError handler that
        # flattens this; the in-test app doesn't, so we walk both
        # shapes).
        body = r.json()
        raw_detail = body.get("detail")
        if isinstance(raw_detail, dict):
            detail_msg = (raw_detail.get("error") or {}).get("message", "")
        else:
            detail_msg = (body.get("error") or {}).get("message") or str(
                raw_detail or ""
            )
        assert "stream" in detail_msg.lower()
        assert "prompt" in detail_msg.lower() and "array" in detail_msg.lower()

    def test_list_prompt_with_stream_false_still_accepted(
        self, patched_config, monkeypatch
    ):
        """Non-streaming list-prompt is the existing (correct) shape;
        must not regress."""

        async def _fake_generate(*_a, **_kw):
            return _StubGenerationOutput()

        def _factory():
            e = MagicMock()
            e.generate = _fake_generate
            return e

        client, _ = _build_completions_app(
            patched_config, monkeypatch, engine_factory=_factory
        )
        r = client.post(
            "/v1/completions",
            json={
                "model": "stub-model",
                "prompt": ["a", "b"],
                "stream": False,
                "max_tokens": 4,
            },
        )
        assert r.status_code == 200, r.text
        # Both prompts must return a choice (the non-streaming branch
        # was always correct; pinning here so a future "fix" doesn't
        # accidentally drop prompts[1:N] from this path too).
        assert len(r.json()["choices"]) == 2

    def test_single_prompt_with_stream_true_still_accepted(
        self, patched_config, monkeypatch
    ):
        """Single-prompt streaming is the common case — must not
        regress under the new gate. We only validate the route admits
        the request (status 200) and returns SSE — full streaming
        behaviour is covered elsewhere."""

        async def _fake_stream(*_a, **_kw):
            # Empty generator — just need the SSE response shape.
            return
            yield  # pragma: no cover - structural

        def _factory():
            e = MagicMock()
            e.stream_generate = _fake_stream
            return e

        client, _ = _build_completions_app(
            patched_config, monkeypatch, engine_factory=_factory
        )
        with client.stream(
            "POST",
            "/v1/completions",
            json={
                "model": "stub-model",
                "prompt": "single",
                "stream": True,
                "max_tokens": 4,
            },
        ) as r:
            assert r.status_code == 200, r.read().decode()

    def test_single_element_list_prompt_with_stream_true_accepted(
        self, patched_config, monkeypatch
    ):
        """``prompt:["a"]`` (length-1 list) is NOT the broken multi-
        prompt case; streaming the single element is fine. Pin this
        so the rejection gate doesn't over-trigger on the trivial
        single-element list shape some SDKs always wrap."""

        async def _fake_stream(*_a, **_kw):
            return
            yield  # pragma: no cover - structural

        def _factory():
            e = MagicMock()
            e.stream_generate = _fake_stream
            return e

        client, _ = _build_completions_app(
            patched_config, monkeypatch, engine_factory=_factory
        )
        with client.stream(
            "POST",
            "/v1/completions",
            json={
                "model": "stub-model",
                "prompt": ["only"],
                "stream": True,
                "max_tokens": 4,
            },
        ) as r:
            assert r.status_code == 200, r.read().decode()


# ---------------------------------------------------------------------------
# Bug 5 (task #311): chat logprobs=true without top_logprobs
# ---------------------------------------------------------------------------


class _LogprobsCapableEngine:
    """Minimal mock engine for the chat logprobs path.

    Yields one streaming chunk carrying a fake per-token logprobs
    array so ``_extract_streaming_token_logprobs`` produces a non-
    empty list. The route-level gate is what we're testing; the
    extractor's correctness lives in its own tests.
    """

    preserve_native_tool_format = False
    is_mllm = False
    supports_guided_generation = False

    class _Tokenizer:
        def decode(self, ids):
            return "tok"

    tokenizer = _Tokenizer()

    def __init__(self):
        self.stream_calls: list[dict] = []

    def build_prompt(self, messages, tools=None, enable_thinking=None):
        return "PROMPT"

    async def chat(self, messages, **kwargs):
        """Non-streaming branch (logprobs=false). Single output, no
        per-step logprobs distribution."""
        from vllm_mlx.engine.base import GenerationOutput

        return GenerationOutput(
            text="hi",
            new_text="hi",
            tokens=[1],
            prompt_tokens=2,
            completion_tokens=1,
            finished=True,
            finish_reason="stop",
        )

    async def stream_chat(self, messages, **kwargs):
        import mlx.core as mx

        from vllm_mlx.engine.base import GenerationOutput

        # Fake per-step logprobs: a 1D mlx.array distribution over a
        # tiny vocab so ``argpartition`` is well-defined for both
        # top_k=1 and top_k>1. Pass the mlx.array directly so the
        # extractor's ``logprobs_array.astype(mx.float32)`` call works
        # — numpy arrays' ``.astype`` rejects an ``mlx.core`` dtype.
        fake_logprobs = mx.array([-1.5, -0.5, -2.0, -3.0, -4.0], dtype=mx.float32)
        self.stream_calls.append({"messages": messages, "kwargs": kwargs})
        yield GenerationOutput(
            text="hi",
            new_text="hi",
            tokens=[1],
            prompt_tokens=2,
            completion_tokens=1,
            finished=True,
            finish_reason="stop",
            logprobs=fake_logprobs,
        )


def _make_chat_client(engine) -> TestClient:
    from vllm_mlx.config import reset_config
    from vllm_mlx.routes.chat import router as chat_router

    cfg = reset_config()
    cfg.engine = engine
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = True
    cfg.reasoning_parser = None

    app = FastAPI()
    app.include_router(chat_router)
    return TestClient(app)


class TestChatLogprobsBaseSemantic:
    """``logprobs=true`` alone (no ``top_logprobs``) must return
    sampled-token logprob; ``top_logprobs=K`` adds K alternatives."""

    @pytest.fixture(autouse=True)
    def _reset(self):
        from vllm_mlx.config import reset_config

        yield
        reset_config()

    def test_logprobs_true_without_top_logprobs_returns_sampled_token_logprob(
        self,
    ):
        """R15 task #311 contract: ``logprobs=true, top_logprobs=None``
        — choice.logprobs.content is non-empty (route takes the
        logprobs-emitting branch even with top_logprobs absent), and
        ``top_logprobs`` on each entry is the empty list (no
        alternatives, OpenAI semantics for the bare logprobs case)."""
        engine = _LogprobsCapableEngine()
        client = _make_chat_client(engine)
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 4,
                "logprobs": True,
                # top_logprobs intentionally omitted (the bug)
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        lp = body["choices"][0].get("logprobs")
        assert lp is not None, (
            "logprobs=true MUST populate choice.logprobs (R15 task #311); "
            f"got {body['choices'][0]!r}"
        )
        content = lp.get("content")
        assert content, (
            f"choice.logprobs.content must be non-empty when logprobs=true; got {lp!r}"
        )
        first = content[0]
        # Sampled-token fields must be populated.
        assert "token" in first and "logprob" in first
        # ``top_logprobs`` MUST be the empty list — caller didn't
        # request alternatives.
        assert first.get("top_logprobs") == [], (
            "bare logprobs=true must NOT carry alternatives; "
            f"got top_logprobs={first.get('top_logprobs')!r}"
        )

    def test_logprobs_true_with_top_logprobs_one_still_returns_alternatives(
        self,
    ):
        """Regression: ``logprobs=true, top_logprobs=1`` must keep
        the alternatives field populated (single entry). The bug-5 fix
        must not break the existing top_logprobs path."""
        engine = _LogprobsCapableEngine()
        client = _make_chat_client(engine)
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 4,
                "logprobs": True,
                "top_logprobs": 1,
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        lp = body["choices"][0].get("logprobs")
        assert lp is not None
        content = lp.get("content")
        assert content
        first = content[0]
        # With top_logprobs=1 the alternatives field carries exactly
        # one entry.
        assert len(first.get("top_logprobs") or []) == 1, (
            "top_logprobs=1 must surface 1 alternative; "
            f"got {first.get('top_logprobs')!r}"
        )

    def test_logprobs_false_returns_no_logprobs(self):
        """Baseline: ``logprobs=false`` (default) returns
        ``choice.logprobs=None``. Pin so the new gate doesn't
        accidentally populate logprobs when the caller didn't ask."""
        engine = _LogprobsCapableEngine()
        client = _make_chat_client(engine)
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 4,
                "logprobs": False,
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["choices"][0].get("logprobs") is None
