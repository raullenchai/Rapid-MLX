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


def test_session_id_is_stable_under_concurrent_first_callers(fake_home):
    """Round 6 codex review: the prior lazy ``_session_id`` init was
    unlocked. Two concurrent first-emitters could race past the
    ``is None`` check and generate different uuids → aggregation
    pipeline sees two sessions per real session. Pin that 32 threads
    racing ``session_id()`` agree on one value."""
    import threading

    from vllm_mlx.telemetry import emit

    emit._reset_for_tests()

    results: list[str] = []
    started = threading.Event()
    barrier = threading.Barrier(32)

    def racer():
        barrier.wait(timeout=5.0)
        results.append(emit.session_id())

    threads = [threading.Thread(target=racer) for _ in range(32)]
    for t in threads:
        t.start()
    started.set()
    for t in threads:
        t.join(timeout=5.0)

    assert len(results) == 32
    assert len(set(results)) == 1, (
        f"concurrent first callers generated {len(set(results))} distinct "
        f"session_ids: {set(results)}"
    )


def test_cli_kill_switch_overrides_opt_in(opted_in, stub_queue):
    """``--no-telemetry`` (threaded through ``set_cli_kill_switch``) must
    suppress every emit site, even when the user has previously opted in
    via the consent file. Before this was wired, ``rapid-mlx --no-telemetry
    models`` still POSTed two events to the collector."""
    from vllm_mlx.telemetry import emit
    from vllm_mlx.telemetry.state import set_cli_kill_switch

    set_cli_kill_switch(True)
    try:
        emit.session_start(subcommand="serve")
        emit.session_end(subcommand="serve", duration_seconds=42)
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
        emit.error(
            category="model_load_failure",
            exc=RuntimeError("x"),
            phase="startup",
        )
    finally:
        set_cli_kill_switch(False)
    assert stub_queue == []


# ---------------------------------------------------------- shape when on


def test_subcommand_normalized_to_allowlist(opted_in, stub_queue):
    """Round 11 codex review: ``subcommand`` was the last free-form
    ``str`` slot on ``session_start``/``session_end``, the same shape
    of escape hatch closed for ``endpoint`` / ``category`` / ``phase``.
    Pin known values pass through, unknown collapse to ``"other"``."""
    from vllm_mlx.telemetry import emit

    emit.session_start(subcommand="serve")
    assert stub_queue[-1]["session"]["subcommand"] == "serve"

    emit.session_end(subcommand="serve", duration_seconds=10)
    assert stub_queue[-1]["session"]["subcommand"] == "serve"

    # An internal / undocumented subcommand collapses.
    leak = "internal:dump?path=/Users/alice/secrets.txt"
    emit.session_start(subcommand=leak)
    assert stub_queue[-1]["session"]["subcommand"] == "other"
    assert "alice" not in repr(stub_queue[-1])


def test_runtime_payload_carries_every_schema_v1_field(opted_in, stub_queue):
    """Round 8 codex review: ``SCHEMA_VERSION == 1`` means the runtime
    payload must include EVERY key the dataclass
    ``SessionPayload`` documents — even default-valued ones — so v1
    consumers parsing the envelope can rely on the keys being present.
    The previous hand-built payload silently dropped ``duration_seconds``
    + ``engine`` from session_start and ``engine`` from session_end.
    Pin both runtime payloads expose the full v1 surface."""
    from dataclasses import fields as _fields

    from vllm_mlx.telemetry import emit
    from vllm_mlx.telemetry.schema import SessionPayload

    expected_keys = {f.name for f in _fields(SessionPayload)}

    emit.session_start(subcommand="serve", argv=["serve"])
    session = stub_queue[-1]["session"]
    missing = expected_keys - set(session)
    assert not missing, f"session_start dropped v1 keys: {missing}"

    emit.session_end(subcommand="serve", duration_seconds=42)
    session = stub_queue[-1]["session"]
    missing = expected_keys - set(session)
    assert not missing, f"session_end dropped v1 keys: {missing}"


def test_session_start_envelope_when_enabled(opted_in, stub_queue):
    from vllm_mlx.telemetry import emit

    emit.session_start(
        subcommand="serve",
        argv=["serve", "qwen3.5-9b", "--host", "0.0.0.0", "--port", "8000"],
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
    """Bucketed counts only — raw token counts and TTFT must not survive.

    Round 12 codex review caught that a whole-payload ``repr`` scan for
    raw numeric substrings was flaky: the envelope carries random
    UUIDs (``client_id``, ``session_id``) and a timestamp, so a uuid
    that happens to contain the substring ``"137"`` would CI-fail the
    test even with bucketing intact. Assert specific fields equal
    expected bucket labels instead.
    """
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
    # Bucket strings, not raw ints / floats — assert the exact labels
    # the bucketing primitives are documented to produce for these
    # inputs, so a future bucket-edge regression also trips.
    assert r["prompt_tokens_bucket"] == "0-256"  # 137 < 256
    assert r["completion_tokens_bucket"] == "1k-4k"  # 1024 <= 1729 < 4096
    assert r["ttft_ms_bucket"] == "100-500ms"  # 432.5 < 500
    assert r["tps_bucket"] == "50-100"  # 58.2 < 100
    # The raw int/float values must NOT be in any request field — scan
    # only the request sub-object so envelope UUIDs / timestamps cannot
    # cause a false positive.
    request_blob = repr(r)
    for raw in ("137", "1729", "432.5", "58.2"):
        assert raw not in request_blob, f"{raw!r} survived into request payload: {r}"


def test_error_category_and_phase_normalised_to_allowlist(opted_in, stub_queue):
    """Round 3 codex review: ``category`` + ``phase`` were stored
    verbatim. Same escape hatch as ``endpoint`` — a future caller
    threading exception text or user input would have leaked. Pin
    that off-allowlist values collapse to ``"other"`` and known
    values pass through."""
    from vllm_mlx.telemetry import emit

    # Known good values pass through.
    try:
        raise RuntimeError("synthetic")
    except RuntimeError as exc:
        emit.error(category="model_load_failure", exc=exc, phase="startup")
    e = stub_queue[-1]["error"]
    assert e["category"] == "model_load_failure"
    assert e["phase"] == "startup"

    # Free-form text — even text containing what looks like a prompt —
    # collapses to "other".
    leak = "user typed: please summarize Q3 numbers"
    try:
        raise RuntimeError("x")
    except RuntimeError as exc:
        emit.error(category=leak, exc=exc, phase=leak)
    e = stub_queue[-1]["error"]
    assert e["category"] == "other"
    assert e["phase"] == "other"
    blob = repr(stub_queue[-1])
    assert "summarize" not in blob
    assert "Q3" not in blob


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


# ---------------------------------------------------------- prompt-leak red line
#
# These tests do not exercise behaviour — they pin the API CONTRACT: the
# four public helpers must never expose a parameter that could carry user
# prompts, completions, generated text, file paths, secrets, or anything
# resembling them. A future maintainer adding ``prompt_text=...`` to
# ``request()`` will trip these and have to delete the test on the way
# in, which is the bright line we want.


def test_public_emit_signatures_have_no_prompt_or_completion_fields():
    """Lock the parameter names of the public emit helpers.

    If you are looking at this test because you want to ship a new field,
    that's fine — but the new field must NOT be raw text. Anything
    free-form must go through ``redact.py`` first and land here as a
    bucket / fingerprint / hash, never as the raw value.
    """
    import inspect

    from vllm_mlx.telemetry import emit

    forbidden = {
        "prompt",
        "prompt_text",
        "prompts",
        "messages",
        "user_message",
        "completion",
        "completion_text",
        "completions",
        "generated_text",
        "response_text",
        "input_text",
        "output_text",
        "content",
        "text",
        "system_prompt",
        "api_key",
        "auth_token",
        "bearer",
        "file_path",
        "filepath",
        "path",
        "url",
        "stream_url",
        "ip",
        "ip_address",
        "hostname",
        # ``engine`` was removed in round 4 because it was a free-form
        # ``str`` slot that the signature pin couldn't catch. If a
        # second engine ever lands, re-add it through a small enum
        # (the same shape as ``_ALLOWED_ENDPOINTS``).
        "engine",
    }
    for fn_name in ("session_start", "session_end", "request", "error"):
        fn = getattr(emit, fn_name)
        params = set(inspect.signature(fn).parameters.keys())
        leak = params & forbidden
        assert not leak, (
            f"emit.{fn_name} exposes prompt-like parameter(s) {leak!r}; "
            "free-form text must go through redact.py first"
        )


def test_request_endpoint_constrained_to_allowlist(opted_in, stub_queue):
    """Round 2 codex review: ``endpoint`` used to be stored verbatim,
    which was a free-form escape hatch — a caller threading a path
    with a query string (``/v1/chat?key=sk-xxx``) would have leaked
    the value. The helper now normalises to a tiny allowlist."""
    from vllm_mlx.telemetry import emit

    # Allowed endpoint round-trips verbatim (after strip).
    emit.request(
        endpoint="/v1/chat/completions",
        model_alias="qwen3.5-9b",
        stream=True,
        tool_call_used=False,
        prompt_tokens=10,
        completion_tokens=10,
        ttft_ms=100.0,
        tps=10.0,
        status=200,
    )
    assert stub_queue[-1]["request"]["endpoint"] == "/v1/chat/completions"

    # Query string + fragment stripped before allowlist match.
    emit.request(
        endpoint="/v1/chat/completions?api_key=sk-PROD-SECRET#anchor",
        model_alias="qwen3.5-9b",
        stream=True,
        tool_call_used=False,
        prompt_tokens=10,
        completion_tokens=10,
        ttft_ms=100.0,
        tps=10.0,
        status=200,
    )
    last = stub_queue[-1]
    assert last["request"]["endpoint"] == "/v1/chat/completions"
    blob = repr(last)
    assert "sk-PROD-SECRET" not in blob
    assert "anchor" not in blob

    # Anything off the allowlist collapses to "other" — no caller
    # string leaks into the payload.
    emit.request(
        endpoint="/internal/dump?path=/Users/alice/secrets.txt",
        model_alias="qwen3.5-9b",
        stream=True,
        tool_call_used=False,
        prompt_tokens=10,
        completion_tokens=10,
        ttft_ms=100.0,
        tps=10.0,
        status=200,
    )
    last = stub_queue[-1]
    assert last["request"]["endpoint"] == "other"
    assert "alice" not in repr(last)
    assert "secrets.txt" not in repr(last)


def test_request_endpoint_normalizes_full_url_to_path(opted_in, stub_queue):
    """Round 13 codex review: the previous string-split implementation
    of ``_normalize_endpoint`` left a full URL like
    ``https://host/v1/chat/completions`` unmatched and silently collapsed
    it to ``"other"`` — defeating the very purpose of the allowlist for
    any caller that builds an endpoint identifier from a full URL.
    ``urlsplit`` extracts ``/v1/chat/completions`` regardless of the
    surrounding scheme/netloc so the match succeeds and the field
    carries the allowlisted route."""
    from vllm_mlx.telemetry import emit

    emit.request(
        endpoint="https://api.example.com/v1/chat/completions",
        model_alias="qwen3.5-9b",
        stream=True,
        tool_call_used=False,
        prompt_tokens=10,
        completion_tokens=10,
        ttft_ms=100.0,
        tps=10.0,
        status=200,
    )
    last = stub_queue[-1]
    assert last["request"]["endpoint"] == "/v1/chat/completions"
    # And the surrounding URL bits must not have leaked into the payload.
    blob = repr(last)
    assert "api.example.com" not in blob
    assert "https://" not in blob

    # Combined: full URL + query + fragment still resolves correctly.
    emit.request(
        endpoint="https://host/v1/chat/completions?key=sk-PROD-LEAK#frag",
        model_alias="qwen3.5-9b",
        stream=True,
        tool_call_used=False,
        prompt_tokens=10,
        completion_tokens=10,
        ttft_ms=100.0,
        tps=10.0,
        status=200,
    )
    last = stub_queue[-1]
    assert last["request"]["endpoint"] == "/v1/chat/completions"
    blob = repr(last)
    assert "sk-PROD-LEAK" not in blob
    assert "frag" not in blob
    assert "host" not in blob


def test_session_models_loaded_does_not_materialize_full_input(opted_in, stub_queue):
    """Round 13 codex review: the helper sliced after building a tuple
    of the entire input, which contradicted the documented "slice
    before normalize" intent and wasted work for callers handing in
    large iterables. Pin that only the first 32 entries are even
    pulled from the source by passing a generator that records how
    many items it yields."""
    from vllm_mlx.telemetry import emit

    pulled: list[int] = []

    def big_gen():
        for i in range(10_000):
            pulled.append(i)
            yield f"mlx-community/model-{i}"

    emit.session_start(subcommand="serve", models_loaded=big_gen())
    last = stub_queue[-1]
    # Generator must have been pulled exactly 32 times, NOT 10000.
    assert len(pulled) == 32
    assert len(last["session"]["models_loaded"]) == 32

    # Same property for session_end.
    pulled.clear()
    emit.session_end(subcommand="serve", duration_seconds=1, models_loaded=big_gen())
    last = stub_queue[-1]
    assert len(pulled) == 32
    assert len(last["session"]["models_loaded"]) == 32


def test_safe_does_not_swallow_signature_mismatch(opted_in, stub_queue):
    """Round 14 codex review: the broad ``except Exception`` in
    ``_safe`` used to also swallow ``TypeError`` from a call site that
    drifted out of sync with the helper signature — a typo on a kwarg
    name or a missing positional argument silently turned into "no
    telemetry," and the integration tests couldn't see the wiring
    bug. ``inspect.signature(fn).bind(...)`` now runs BEFORE the broad
    catch so signature mismatches raise visibly."""
    from vllm_mlx.telemetry import emit

    # Bad kwarg name — must raise TypeError, NOT become a silent no-op.
    with pytest.raises(TypeError):
        emit.session_start(subkommand="serve")  # typo: subkommand

    # Missing required keyword.
    with pytest.raises(TypeError):
        emit.request(
            # endpoint missing
            model_alias="qwen3.5-9b",
            stream=True,
            tool_call_used=False,
            prompt_tokens=10,
            completion_tokens=10,
            ttft_ms=100.0,
            tps=10.0,
            status=200,
        )

    # And nothing leaked into the queue from the broken calls.
    assert stub_queue == []


def test_request_model_alias_local_path_redacted(opted_in, stub_queue):
    """``model_alias`` is the ONE free-form-ish field on ``request()``,
    but it must be funnelled through ``normalize_model_path`` so a local
    checkout path collapses to ``"<local>"`` instead of leaking the
    user's home-directory layout."""
    from vllm_mlx.telemetry import emit

    emit.request(
        endpoint="/v1/chat/completions",
        model_alias="/Users/alice/private-model-checkout",
        stream=True,
        tool_call_used=False,
        prompt_tokens=100,
        completion_tokens=400,
        ttft_ms=250.0,
        tps=42.0,
        status=200,
    )
    r = stub_queue[0]["request"]
    assert r["model_alias"] == "<local>"
    assert "alice" not in repr(stub_queue[0])


def test_argv_values_with_secrets_never_survive_redaction(opted_in, stub_queue):
    """argv carries the user's full shell command. The redactor must
    extract flag NAMES and nothing else — values are skipped before they
    ever land in a payload field."""
    from vllm_mlx.telemetry import emit

    secret = "sk-prod-XXXXXXXXXXXXXXXXXXXXXXXX"
    bearer = "Bearer eyJhbGciOiJIUzI1NiJ9.payload.sig"
    prompt = "summarize this confidential email about Q3 numbers"
    emit.session_start(
        subcommand="serve",
        argv=[
            "serve",
            "qwen3.5-9b",
            "--api-key",
            secret,
            "--auth-header",
            bearer,
            "--initial-prompt",
            prompt,
        ],
    )
    blob = repr(stub_queue[0])
    assert secret not in blob
    assert bearer not in blob
    assert prompt not in blob
    # Names should be preserved (the contract: we see WHICH flags people
    # use, never what they pass).
    flag_names = set(stub_queue[0]["session"]["flag_names"])
    assert {"api-key", "auth-header", "initial-prompt"} <= flag_names


def test_error_fingerprint_does_not_echo_exception_message(opted_in, stub_queue):
    """A user's prompt CAN end up in an exception message — e.g. a parser
    crash that prints the offending input. The fingerprint must not echo
    it."""
    from vllm_mlx.telemetry import emit

    prompt_in_exc = "summarize this confidential email about Q3 numbers"
    try:
        raise ValueError(f"parser failed on: {prompt_in_exc}")
    except ValueError as exc:
        emit.error(category="parser_failure", exc=exc, phase="chat")

    blob = repr(stub_queue[0])
    assert prompt_in_exc not in blob
    assert "parser failed" not in blob
