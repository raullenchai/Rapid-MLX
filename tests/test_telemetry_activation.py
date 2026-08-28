# SPDX-License-Identifier: Apache-2.0
"""Activation / engagement semantics — the versioned caliber contract.

Pins docs/telemetry-activation.md (``ACTIVATION_SPEC_VERSION``) in code so a
silent drift in "what counts as engaged" fails CI. Covers, per the spec's
required list:

- the success predicate (2xx AND non-empty) for streaming + non-streaming,
- failed requests and empty generations NOT counting,
- health / models / server-startup NOT counting,
- once-per-install dedup (marker, across processes) and unsampled emission,
- ``surface`` resolution (cli when chat-spawned, else api).

Every ``vllm_mlx`` import is lazy (inside helpers/tests) so this collects and
runs on the no-mlx ``pr_validate`` gate, like the request-wiring mirrors.
"""

from __future__ import annotations

import asyncio
import importlib
from types import SimpleNamespace

import pytest

# --------------------------------------------------------------- spec pins


def test_spec_version_is_int_and_doc_exists():
    import re
    from pathlib import Path

    from vllm_mlx.telemetry import activation_spec as spec

    assert isinstance(spec.ACTIVATION_SPEC_VERSION, int)
    assert spec.ACTIVATION_SPEC_VERSION >= 1
    # The human half of the contract ships alongside the code half.
    doc = Path(__file__).resolve().parents[1] / "docs" / "telemetry-activation.md"
    assert doc.exists(), "docs/telemetry-activation.md must ship with the spec"
    text = doc.read_text()
    # Version LOCK, not a substring probe: parse the number the doc actually
    # states ("Spec version: `N`") and assert it equals the code constant, so
    # bumping one without the other fails CI. A bare ``"ACTIVATION_SPEC_VERSION"
    # in text`` check would still pass while the numbers silently disagree.
    m = re.search(r"Spec version:\s*`(\d+)`", text)
    assert m, "doc must state its version as 'Spec version: `N`'"
    assert int(m.group(1)) == spec.ACTIVATION_SPEC_VERSION, (
        f"doc says v{m.group(1)} but code is v{spec.ACTIVATION_SPEC_VERSION}"
    )


def test_kinds_and_surfaces_are_the_allowlist():
    from vllm_mlx.telemetry import activation_spec as spec

    assert {
        "first_inference",
        "model_pull",
        "agent_setup",
        "first_chat_reply",
        "first_vision_reply",
        "first_dictation",
        "first_image",
    } == spec.ACTIVATION_KINDS
    assert {
        "first_chat_reply",
        "first_vision_reply",
        "first_dictation",
        "first_image",
    } == spec.DESKTOP_ACTIVATION_KINDS
    assert {"cli", "api", "desktop"} == spec.ACTIVATION_SURFACES
    assert {
        ("first_inference", "cli"),
        ("first_inference", "api"),
        ("model_pull", "cli"),
        ("agent_setup", "cli"),
        ("first_chat_reply", "desktop"),
        ("first_vision_reply", "desktop"),
        ("first_dictation", "desktop"),
        ("first_image", "desktop"),
    } == spec.ACTIVATION_KIND_SURFACE_PAIRS


def test_health_and_models_are_not_inference_endpoints():
    """Liveness/metadata probes must never be classed as inference."""
    from vllm_mlx.telemetry import activation_spec as spec

    for probe in ("/health", "/healthz", "/v1/models", "/models"):
        assert probe not in spec.INFERENCE_ENDPOINTS


# ------------------------------------------------ success predicate (caliber)


@pytest.mark.parametrize(
    "status,tokens,expected",
    [
        (200, 8, True),  # non-streaming success
        (200, 1, True),  # a single real token still counts
        (201, 3, True),  # any 2xx
        (299, 3, True),
        (200, 0, False),  # EMPTY generation: "it ran" != "it worked"
        (204, 0, False),  # 204 no-content: zero tokens fails the non-empty check
        (400, 8, False),  # client error
        (404, 8, False),
        (500, 8, False),  # server error
        (300, 8, False),  # redirect is not success
        (200, -1, False),  # nonsense token count
    ],
)
def test_is_successful_inference(status, tokens, expected):
    from vllm_mlx.telemetry.activation_spec import is_successful_inference

    assert is_successful_inference(status, tokens) is expected


def test_is_successful_inference_tolerates_bad_types():
    from vllm_mlx.telemetry.activation_spec import is_successful_inference

    assert is_successful_inference("nope", 5) is False
    assert is_successful_inference(200, None) is False


# ------------------------------------------------------ marker (once/install)


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("RAPID_MLX_TELEMETRY", raising=False)
    monkeypatch.delenv("RAPID_MLX_CHAT_SPAWN", raising=False)

    import vllm_mlx.telemetry.state as state

    importlib.reload(state)
    import vllm_mlx.telemetry.emit as emit

    importlib.reload(emit)
    emit._reset_for_tests()
    return tmp_path


def test_claim_activation_marker_is_once_per_kind(fake_home):
    from vllm_mlx.telemetry import state

    assert state.claim_activation_marker("first_inference") is True
    # Second claim of the SAME kind loses — this is the once-per-install latch.
    assert state.claim_activation_marker("first_inference") is False
    # A different kind is independent.
    assert state.claim_activation_marker("model_pull") is True
    assert state.claim_activation_marker("model_pull") is False


def test_marker_path_is_per_kind_under_rapid_mlx(fake_home):
    from vllm_mlx.telemetry import state

    p = state.activation_marker_path("first_inference")
    assert p.name == "activation_seen_first_inference"
    assert p.parent.name == ".rapid-mlx"


def test_claim_activation_marker_is_atomic_across_processes(fake_home):
    """PRIMITIVE test: ``claim_activation_marker`` itself elects exactly one
    winner across REAL processes. ``fake_home`` set HOME in the environment;
    spawned children inherit it and resolve the SAME marker dir, so N
    interpreters race one O_CREAT|O_EXCL create and the kernel picks a single
    winner. This is the best-effort suppressor ``activation()`` is built on — it
    is NOT a claim that ``activation()`` *delivery* is exactly-once. Delivery is
    at-least-once by design (enqueue-before-claim); that ordering is pinned in
    ``test_activation_enqueues_before_claiming_marker``.

    ``claim_activation_marker`` is a top-level function in the importable
    ``vllm_mlx.telemetry.state`` package, so it pickles by reference and the
    spawned children import only that light module (no mlx), keeping the test
    fast and free of the tests-package import pitfalls of a custom worker.
    """
    import multiprocessing as mp

    from vllm_mlx.telemetry.state import claim_activation_marker

    n = 8
    ctx = mp.get_context("spawn")
    with ctx.Pool(n) as pool:
        results = pool.map(claim_activation_marker, ["first_inference"] * n)
    assert sum(1 for r in results if r) == 1, results


def test_activation_enqueues_before_claiming_marker(opted_in, monkeypatch):
    """Pins the enqueue-BEFORE-claim ordering (design A) PRECISELY: the payload
    must be durably enqueued before the once-ever marker is claimed. That
    ordering is the whole point — it is what makes retry-after-failure work: if
    the claim ran first, a failed enqueue would burn the marker and drop the
    install from the funnel forever (see docs/telemetry-activation.md and
    ``test_activation_retries_after_enqueue_failure``).

    A spy on both operations distinguishes the two orderings unambiguously —
    unlike a marker-delete-then-recall test, which enqueues twice under BOTH
    orderings and so cannot pin which came first. In design B (claim-first) the
    recorded order would be ``["claim", "enqueue"]``; design A must be the
    reverse."""
    from vllm_mlx.telemetry import emit, state

    order: list[str] = []

    class _SpyQueue:
        def enqueue(self, payload):
            order.append("enqueue")

    real_claim = state.claim_activation_marker

    def spy_claim(kind):
        order.append("claim")
        return real_claim(kind)

    # emit.activation does a call-time ``from ...state import claim_activation_marker``,
    # so patching the name on the state module is what it will resolve.
    monkeypatch.setattr(emit, "get_queue", lambda: _SpyQueue())
    monkeypatch.setattr(state, "claim_activation_marker", spy_claim)

    emit.activation(activation_kind="first_inference", surface="api")

    assert order == ["enqueue", "claim"]


# ----------------------------------------------------------- surface resolve


def test_server_surface_defaults_to_api(fake_home, monkeypatch):
    from vllm_mlx.telemetry import emit

    monkeypatch.delenv("RAPID_MLX_CHAT_SPAWN", raising=False)
    assert emit.server_surface() == "api"


def test_server_surface_is_cli_when_chat_spawned(fake_home, monkeypatch):
    from vllm_mlx.telemetry import emit

    monkeypatch.setenv("RAPID_MLX_CHAT_SPAWN", "1")
    assert emit.server_surface() == "cli"


@pytest.mark.parametrize("val", ["1", "true", "TRUE", "yes", "on", " 1 "])
def test_server_surface_truthy_values_are_cli(fake_home, monkeypatch, val):
    from vllm_mlx.telemetry import emit

    monkeypatch.setenv("RAPID_MLX_CHAT_SPAWN", val)
    assert emit.server_surface() == "cli"


@pytest.mark.parametrize("val", ["0", "false", "no", "off", "", "treu", "2", "cli"])
def test_server_surface_non_allowlisted_values_default_to_api(
    fake_home, monkeypatch, val
):
    """Only an explicit truthy allowlist is CLI; a typo like ``treu`` must
    degrade to ``api``, not silently corrupt attribution toward ``cli``."""
    from vllm_mlx.telemetry import emit

    monkeypatch.setenv("RAPID_MLX_CHAT_SPAWN", val)
    assert emit.server_surface() == "api"


# ------------------------------------------------------ emit.activation gate


@pytest.fixture
def opted_in(fake_home):
    from vllm_mlx.telemetry.state import record_consent

    record_consent(True, rapid_mlx_version="0.0.0+test")
    return fake_home


@pytest.fixture
def stub_queue(monkeypatch):
    from vllm_mlx.telemetry import emit

    captured: list[dict] = []

    class _StubQueue:
        def enqueue(self, payload):
            captured.append(payload)

    monkeypatch.setattr(emit, "get_queue", lambda: _StubQueue())
    return captured


def test_activation_no_op_when_disabled(fake_home, stub_queue):
    from vllm_mlx.telemetry import emit

    emit.activation(activation_kind="first_inference", surface="api")
    assert stub_queue == []
    # A disabled install must NOT burn its once-ever marker, so enabling
    # later still lets the first real inference emit.
    from vllm_mlx.telemetry import state

    assert not state.activation_marker_path("first_inference").exists()


def test_activation_emits_expected_envelope(opted_in, stub_queue):
    from vllm_mlx.telemetry import emit

    emit.activation(activation_kind="first_inference", surface="cli")
    assert len(stub_queue) == 1
    p = stub_queue[0]
    assert p["event"] == "activation"
    assert p["activation"] == {
        "activation_kind": "first_inference",
        "surface": "cli",
    }
    # Standard envelope present; no request/session/error payloads.
    assert p["schema_version"] == 1
    assert p["client_id"]
    assert "session_id" in p and "timestamp" in p and "platform" in p
    assert "request" not in p and "session" not in p and "error" not in p


def test_activation_is_once_per_install_within_process(opted_in, stub_queue):
    from vllm_mlx.telemetry import emit

    emit.activation(activation_kind="first_inference", surface="api")
    emit.activation(activation_kind="first_inference", surface="api")
    emit.activation(activation_kind="first_inference", surface="cli")
    assert len(stub_queue) == 1  # only the first ever


def test_activation_marker_dedup_survives_latch_reset(opted_in, stub_queue):
    """Dedup is keyed on the persistent on-disk marker, not the in-process
    latch: clearing the latch (as a fresh process would start) and re-calling
    must NOT re-emit, because ``claim_activation_marker`` loses on the existing
    marker. (The genuine cross-process *race* is covered by
    ``test_claim_activation_marker_is_atomic_across_processes``; this asserts the
    single-process restart path.)"""
    from vllm_mlx.telemetry import emit

    emit.activation(activation_kind="first_inference", surface="api")
    assert len(stub_queue) == 1
    # Simulate a restarted process: clear the in-process latch, keep the marker.
    emit._reset_for_tests()
    emit.activation(activation_kind="first_inference", surface="api")
    assert len(stub_queue) == 1  # marker on disk suppresses the second claim


def test_activation_retries_after_enqueue_failure(opted_in, monkeypatch):
    """A transient enqueue failure must NOT permanently suppress the milestone.
    Because emission enqueues BEFORE claiming the marker, a failing enqueue
    leaves the marker unclaimed, so the next successful inference retries and
    sends. (Claiming before enqueue would drop the install from the funnel
    forever on one queue hiccup — the failure mode this ordering avoids.)"""
    from vllm_mlx.telemetry import emit, state

    class _BoomThenOK:
        def __init__(self):
            self.sent = []
            self._armed = True

        def enqueue(self, payload):
            if self._armed:
                self._armed = False
                raise RuntimeError("queue down")
            self.sent.append(payload)

    q = _BoomThenOK()
    monkeypatch.setattr(emit, "get_queue", lambda: q)

    # First attempt: enqueue raises -> @_safe swallows; marker never claimed.
    emit.activation(activation_kind="first_inference", surface="api")
    assert q.sent == []
    assert not state.activation_marker_path("first_inference").exists()

    # A restarted process retries; the queue is healthy now -> it sends once.
    emit._reset_for_tests()
    emit.activation(activation_kind="first_inference", surface="api")
    assert len(q.sent) == 1
    assert state.activation_marker_path("first_inference").exists()


def test_activation_latches_even_when_marker_persist_fails(
    opted_in, stub_queue, monkeypatch
):
    """If the marker can't persist (e.g. ~/.rapid-mlx went read-only after
    consent), the process must still latch after its single enqueue so it does
    NOT re-emit on every later request. The cross-process duplicate this allows
    is folded downstream by the stable client_id (guaranteed present whenever
    activation can emit); the in-process latch bounds it to ONE per process."""
    from vllm_mlx.telemetry import emit, state

    # Simulate an unwritable state dir: claim always fails, marker never appears.
    monkeypatch.setattr(state, "claim_activation_marker", lambda kind: False)

    emit.activation(activation_kind="first_inference", surface="api")
    emit.activation(activation_kind="first_inference", surface="api")
    emit.activation(activation_kind="first_inference", surface="api")
    # Exactly one enqueue this process despite the marker never persisting.
    assert len(stub_queue) == 1


def test_activation_marker_rejects_path_traversal_kinds(fake_home, tmp_path):
    """`kind` is interpolated into a filename; an out-of-allowlist value with
    ``../`` must never let ``claim_activation_marker`` create files outside
    ~/.rapid-mlx. The path builder validates, so claim fail-safes to False and
    touches nothing on disk.

    The escape target is a UNIQUE path under this test's ``tmp_path`` (not a
    shared global like ``/tmp/rapid_pwn``), so the "nothing was created"
    assertion isolates THIS test's writes and can't be tripped by an unrelated
    process or a prior run."""
    import pytest as _pytest

    from vllm_mlx.telemetry import state

    # A kind that would escape ~/.rapid-mlx and land on a unique, guaranteed-
    # absent target if validation were missing.
    target = tmp_path / "rapid_pwn_target"
    assert not target.exists()
    evil = f"../../../../../../../..{target}"

    with _pytest.raises(ValueError):
        state.activation_marker_path(evil)
    # Public claim swallows the rejection into a safe False — no fs write.
    assert state.claim_activation_marker(evil) is False
    assert not target.exists()  # nothing was created at the escape target


def test_reset_state_clears_in_process_activation_latch(opted_in, stub_queue):
    """`telemetry reset` in the SAME process must let a re-enabled install
    re-earn milestones: reset_state wipes the markers AND the in-memory latch,
    so the next call is not silently suppressed by stale process state."""
    from vllm_mlx.telemetry import emit, state

    emit.activation(activation_kind="first_inference", surface="api")
    assert len(stub_queue) == 1
    assert "first_inference" in emit._activation_latched
    state.reset_state()
    assert "first_inference" not in emit._activation_latched
    # reset_state also wiped consent; re-enable to model "reset then opt in
    # again in the same process". Marker gone + latch cleared -> emits again.
    state.record_consent(True, rapid_mlx_version="0.0.0+test")
    emit.activation(activation_kind="first_inference", surface="api")
    assert len(stub_queue) == 2


def test_reset_state_clears_activation_markers(opted_in, stub_queue):
    """`telemetry reset` rotates the client_id; it must also drop the
    activation markers so the fresh identity can re-earn its milestones —
    otherwise dedup keyed on a stale marker permanently silences the funnel."""
    from vllm_mlx.telemetry import emit, state

    emit.activation(activation_kind="first_inference", surface="api")
    assert state.activation_marker_path("first_inference").exists()
    state.reset_state()
    assert not state.activation_marker_path("first_inference").exists()


def test_activation_kinds_are_independent(opted_in, stub_queue):
    from vllm_mlx.telemetry import emit

    emit.activation(activation_kind="first_inference", surface="api")
    emit.activation(activation_kind="model_pull", surface="cli")
    kinds = [p["activation"]["activation_kind"] for p in stub_queue]
    assert kinds == ["first_inference", "model_pull"]


def test_activation_drops_unknown_kind(opted_in, stub_queue):
    from vllm_mlx.telemetry import emit

    emit.activation(activation_kind="not_a_kind", surface="api")
    assert stub_queue == []


def test_activation_drops_unknown_surface(opted_in, stub_queue):
    """An off-allowlist surface is an instrumentation bug: drop the event (like
    an unknown kind) rather than silently mislabel it as ``api``."""
    from vllm_mlx.telemetry import emit, state

    emit.activation(activation_kind="first_inference", surface="carrier-pigeon")
    assert stub_queue == []
    # And it must not burn the once-ever marker — a later valid call still emits.
    assert not state.activation_marker_path("first_inference").exists()
    emit.activation(activation_kind="first_inference", surface="api")
    assert len(stub_queue) == 1
    assert stub_queue[0]["activation"]["surface"] == "api"


@pytest.mark.parametrize(
    "activation_kind,surface",
    [
        ("first_inference", "desktop"),
        ("model_pull", "api"),
        ("agent_setup", "desktop"),
        ("first_chat_reply", "cli"),
        ("first_vision_reply", "api"),
        ("first_dictation", "cli"),
        ("first_image", "api"),
    ],
)
def test_activation_drops_invalid_kind_surface_pair(
    opted_in, stub_queue, activation_kind, surface
):
    from vllm_mlx.telemetry import emit, state

    emit.activation(activation_kind=activation_kind, surface=surface)

    assert stub_queue == []
    assert not state.activation_marker_path(activation_kind).exists()


def test_activation_is_not_request_sampled(opted_in, stub_queue, monkeypatch):
    """Setting the request sample rate to 0 silences ``request`` events but
    MUST NOT silence activation — a first-touch milestone can't be sampled."""
    monkeypatch.setenv("RAPID_MLX_TELEMETRY_REQUEST_SAMPLE", "0")
    from vllm_mlx.telemetry import emit

    emit.activation(activation_kind="first_inference", surface="api")
    assert len(stub_queue) == 1


def test_session_start_does_not_emit_activation(opted_in, stub_queue):
    """Server startup / session_start is NOT engagement."""
    from vllm_mlx.telemetry import emit

    emit.session_start(subcommand="serve", first_session=True)
    assert len(stub_queue) == 1
    assert stub_queue[0]["event"] == "session_start"
    # And no activation marker was created by merely starting.
    from vllm_mlx.telemetry import state

    assert not state.activation_marker_path("first_inference").exists()


# ------------------------------------------ route caliber: non-streaming


class _RawRequest:
    def __init__(self, user_agent=None):
        self.headers = {} if user_agent is None else {"user-agent": user_agent}

    async def json(self):
        return {}

    async def is_disconnected(self):
        return False


class _FakeChatEngine:
    supports_guided_generation = False
    preserve_native_tool_format = False
    is_mllm = False
    model_name = "test-model"
    tokenizer = SimpleNamespace(encode=lambda _text: [1])
    _text = "hello there"
    _completion_tokens = 8

    async def chat(self, messages, **kwargs):
        from vllm_mlx.engine.base import GenerationOutput

        return GenerationOutput(
            text=self._text,
            finish_reason="stop",
            prompt_tokens=12,
            completion_tokens=self._completion_tokens,
        )


class _EmptyChatEngine(_FakeChatEngine):
    _text = ""
    _completion_tokens = 0


async def _await_direct(coro, *_a, **_k):
    return await coro


def _patch_route(monkeypatch, engine, activation_calls):
    from vllm_mlx.routes import chat
    from vllm_mlx.telemetry import emit

    monkeypatch.setattr(emit, "is_enabled", lambda *a, **k: True)
    monkeypatch.setattr(emit, "request", lambda **kw: None)  # silence the sampled event
    monkeypatch.setattr(emit, "activation", lambda **kw: activation_calls.append(kw))
    monkeypatch.setattr(chat, "_resolve_max_tokens", lambda *a, **k: 64)
    monkeypatch.setattr(chat, "get_engine", lambda *a, **k: engine)
    monkeypatch.setattr(chat, "_validate_model_name", lambda *a, **k: None)
    monkeypatch.setattr(chat, "_check_admission_or_503", lambda *a, **k: None)
    monkeypatch.setattr(
        chat, "_release_admission_unless_committed", lambda *a, **k: None
    )
    monkeypatch.setattr(chat, "_wait_with_disconnect", _await_direct)
    monkeypatch.setattr(
        chat, "validate_content_blocks_for_capabilities", lambda *a, **k: None
    )
    monkeypatch.setattr(chat, "enforce_context_length_for_messages", lambda *a, **k: 1)


def _request(model="test-model", stream=False):
    from vllm_mlx.api.models import ChatCompletionRequest

    return ChatCompletionRequest(
        model=model,
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=None,
        stream=stream,
    )


@pytest.mark.asyncio
async def test_nonstreaming_success_emits_first_inference(monkeypatch):
    from vllm_mlx.routes import chat

    calls: list[dict] = []
    _patch_route(monkeypatch, _FakeChatEngine(), calls)

    await chat._create_chat_completion_impl(
        _request(),
        _RawRequest(user_agent="claude-cli/1.4.2"),
        _FakeChatEngine(),
        _commit_state=[False],
        _admission_acquired=[False],
    )

    assert len(calls) == 1
    assert calls[0]["activation_kind"] == "first_inference"
    assert calls[0]["surface"] == "api"  # not chat-spawned


@pytest.mark.asyncio
async def test_nonstreaming_success_surface_cli_when_chat_spawned(monkeypatch):
    monkeypatch.setenv("RAPID_MLX_CHAT_SPAWN", "1")
    from vllm_mlx.routes import chat

    calls: list[dict] = []
    _patch_route(monkeypatch, _FakeChatEngine(), calls)

    await chat._create_chat_completion_impl(
        _request(),
        _RawRequest(),
        _FakeChatEngine(),
        _commit_state=[False],
        _admission_acquired=[False],
    )

    assert len(calls) == 1
    assert calls[0]["surface"] == "cli"


@pytest.mark.asyncio
async def test_nonstreaming_empty_generation_does_not_engage(monkeypatch):
    from vllm_mlx.routes import chat

    calls: list[dict] = []
    _patch_route(monkeypatch, _EmptyChatEngine(), calls)

    await chat._create_chat_completion_impl(
        _request(),
        _RawRequest(),
        _EmptyChatEngine(),
        _commit_state=[False],
        _admission_acquired=[False],
    )

    assert calls == []  # 200 + zero completion tokens is NOT engagement


# ------------------------------------------ route caliber: streaming


class _FakeStreamingOutput:
    def __init__(self, new_text, finished, completion_tokens):
        self.new_text = new_text
        self.text = new_text
        self.finished = finished
        self.finish_reason = "stop" if finished else None
        self.channel = None
        self.prompt_tokens = 11
        self.completion_tokens = completion_tokens
        self.cached_tokens = 0
        self.tokens = []
        self.logprobs = None
        self.tool_calls = None
        self.matched_stop = None
        self.raw_text = new_text


class _FakeStreamEngine:
    def __init__(self, deltas, completion_tokens):
        self._deltas = deltas
        self._completion_tokens = completion_tokens
        self.tokenizer = None
        self.is_mllm = False
        self.supports_tool_calls = False
        self.supports_guided_generation = False

    async def stream_chat(self, **kwargs):
        n = len(self._deltas)
        for i, d in enumerate(self._deltas):
            yield _FakeStreamingOutput(
                d, finished=(i == n - 1), completion_tokens=self._completion_tokens
            )

    def build_prompt(self, *args, **kwargs):
        return "prompt"


@pytest.fixture
def _stream_cfg(monkeypatch):
    from vllm_mlx.config import server_config

    cfg = server_config.get_config()
    monkeypatch.setattr(cfg, "tool_call_parser", None, raising=False)
    monkeypatch.setattr(cfg, "reasoning_parser_name", None, raising=False)
    monkeypatch.setattr(cfg, "reasoning_parser", None, raising=False)
    monkeypatch.setattr(cfg, "enable_auto_tool_choice", False, raising=False)
    monkeypatch.setattr(cfg, "gc_control", False, raising=False)
    yield


def _drive_stream(monkeypatch, engine, activation_calls):
    from vllm_mlx.routes import chat
    from vllm_mlx.telemetry import emit

    # monkeypatch (not direct assignment) so teardown restores emit's module
    # state — otherwise these fakes leak into later tests in the same process.
    monkeypatch.setattr(emit, "is_enabled", lambda *a, **k: True)
    monkeypatch.setattr(emit, "request", lambda **kw: None)
    monkeypatch.setattr(emit, "activation", lambda **kw: activation_calls.append(kw))

    async def _run():
        gen = chat.stream_chat_completion(
            engine, [{"role": "user", "content": "hi"}], _request(stream=True)
        )
        async for _chunk in gen:
            pass

    asyncio.run(_run())


def test_streaming_success_emits_first_inference(fake_home, _stream_cfg, monkeypatch):
    calls: list[dict] = []
    _drive_stream(
        monkeypatch, _FakeStreamEngine(["hello", " there"], completion_tokens=7), calls
    )
    assert len(calls) == 1
    assert calls[0]["activation_kind"] == "first_inference"


def test_streaming_empty_does_not_engage(fake_home, _stream_cfg, monkeypatch):
    calls: list[dict] = []
    _drive_stream(monkeypatch, _FakeStreamEngine([""], completion_tokens=0), calls)
    assert calls == []


class _RaisingStreamEngine(_FakeStreamEngine):
    async def stream_chat(self, **kwargs):
        # Deliver a real token, then fail mid-stream (engine crash / server error).
        yield _FakeStreamingOutput("par", finished=False, completion_tokens=3)
        raise RuntimeError("engine exploded mid-stream")


def test_streaming_failure_midway_does_not_engage(fake_home, _stream_cfg, monkeypatch):
    """A stream that yields a token then RAISES must not record engagement: the
    activation emit sits after a NORMAL drain of the generator, so a failed
    generator never reaches it. Under-counting a broken stream is the intended
    conservative behavior (never inflates the funnel)."""
    calls: list[dict] = []
    engine = _RaisingStreamEngine([], completion_tokens=3)
    # Require the SPECIFIC engine failure to reach us — otherwise an unrelated
    # setup/route error before the stream is consumed would also leave
    # ``calls == []`` and the test would pass without exercising the path.
    with pytest.raises(RuntimeError, match="engine exploded mid-stream"):
        _drive_stream(monkeypatch, engine, calls)
    assert calls == []


def test_streaming_client_cancel_does_not_engage(fake_home, _stream_cfg, monkeypatch):
    """Client disconnect mid-stream (break out of the async-for, then aclose ->
    GeneratorExit at the paused yield) must not record engagement — emission is
    reached only after a full normal drain."""
    from vllm_mlx.routes import chat
    from vllm_mlx.telemetry import emit

    calls: list[dict] = []
    monkeypatch.setattr(emit, "is_enabled", lambda *a, **k: True)
    monkeypatch.setattr(emit, "request", lambda **kw: None)
    monkeypatch.setattr(emit, "activation", lambda **kw: calls.append(kw))

    engine = _FakeStreamEngine(["a", "b", "c", "d"], completion_tokens=9)

    async def _run():
        gen = chat.stream_chat_completion(
            engine, [{"role": "user", "content": "hi"}], _request(stream=True)
        )
        async for _chunk in gen:
            break  # simulate client disconnect after the first chunk
        await gen.aclose()

    asyncio.run(_run())
    assert calls == []
