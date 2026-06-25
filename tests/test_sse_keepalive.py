# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the SSE keepalive heartbeat (F-070) + anti-buffering
response headers (F-073).

F-070: pre-fix, ``_disconnect_guard`` yielded only the upstream
generator's chunks, with no heartbeat. A long prefill (e.g. 64k-token
prompt → 60–80 s pure TCP silence before the first content delta)
silently killed EventSource clients (browser ~45 s idle), nginx
(``proxy_read_timeout 60``), Cloudflare (100 s), and most SaaS
gateways. The fix interleaves SSE comment lines (``: keepalive\n\n``)
into the yielded stream while the generator stalls — the comments are
ignored by every conforming SSE consumer per the WHATWG spec.

F-073: the same streaming responses were missing
``Cache-Control: no-cache, no-transform`` and ``X-Accel-Buffering: no``,
so any reverse proxy with default buffering settings was free to
collect the entire SSE response and ship it as one blob at end of
generation. The fix sets the headers on every SSE
``StreamingResponse`` in the codebase via the shared
``SSE_RESPONSE_HEADERS`` constant.
"""

from __future__ import annotations

import asyncio
import time

import pytest


@pytest.fixture(autouse=True)
def _isolate_config():
    """Each test starts from a clean ServerConfig singleton so a
    previous test that monkey-patched ``sse_keepalive_seconds`` does
    not leak its value into the next case."""
    from vllm_mlx.config.server_config import reset_config

    reset_config()
    yield
    reset_config()


def test_sse_response_headers_constant_matches_anti_buffering_contract():
    """The shared ``SSE_RESPONSE_HEADERS`` dict must carry the two
    documented anti-buffering keys. Without them, the F-073 proxy
    buffering symptom returns silently — clients see no streaming
    even though the server is emitting chunks correctly."""
    from vllm_mlx.service.helpers import SSE_RESPONSE_HEADERS

    assert SSE_RESPONSE_HEADERS["Cache-Control"] == "no-cache, no-transform"
    assert SSE_RESPONSE_HEADERS["X-Accel-Buffering"] == "no"


def test_disconnect_guard_passes_through_chunks_unchanged():
    """Positive control: with the upstream generator firing quickly
    (faster than the keepalive interval), the guard MUST NOT inject
    any ``: keepalive`` comments BETWEEN real chunks — only the real
    chunks reach the client (modulo the single post-first-chunk
    keepalive added in R15 #291, which lands AFTER the first chunk
    but BEFORE any subsequent ones).

    Otherwise a fast-streaming model would get its bandwidth inflated
    with no-op comments at every token boundary."""
    from vllm_mlx.service.helpers import _disconnect_guard

    async def _generator():
        for token in ["a", "b", "c"]:
            yield f"data: {token}\n\n"

    class _FakeRequest:
        async def is_disconnected(self) -> bool:
            return False

    async def _run():
        out = []
        # Long keepalive interval so the upstream's quick yields
        # always win the race in ``asyncio.wait``. ``poll_interval``
        # stays at its default (0.5 s) which the disconnect-watch task
        # uses; that task never fires here because the fake request
        # reports connected.
        async for chunk in _disconnect_guard(
            _generator(), _FakeRequest(), keepalive_seconds=60.0
        ):
            out.append(chunk)
        return out

    chunks = asyncio.run(_run())
    # R15 #291: a single keepalive comment lands immediately after the
    # role-equivalent first chunk. Subsequent fast-streaming chunks
    # MUST NOT carry interleaved comments.
    assert chunks == [
        "data: a\n\n",
        ": keepalive\n\n",
        "data: b\n\n",
        "data: c\n\n",
    ]
    keepalives = [c for c in chunks if c.startswith(": keepalive")]
    assert len(keepalives) == 1, (
        f"R15 #291 contract: at most one keepalive on fast streams "
        f"(the post-first-chunk emit). Got {len(keepalives)}: {chunks}"
    )


def test_disconnect_guard_emits_keepalive_when_generator_stalls():
    """F-070 fix: when the upstream generator stalls for longer than
    ``keepalive_seconds``, the guard MUST yield ``: keepalive\\n\\n``
    SSE comment lines until the generator produces something.

    Models the long-prefill scenario: pre-fix, a 64k prompt gave the
    client 60+ s of TCP silence; post-fix the client sees a comment
    line every ~20 s by default, which keeps EventSource + nginx +
    Cloudflare from tearing the connection down."""
    from vllm_mlx.service.helpers import _disconnect_guard

    async def _slow_generator():
        # Stall longer than the keepalive interval so the test
        # actually exercises the heartbeat path. We use 0.4 s of
        # stall and a 0.1 s keepalive: the guard should emit ~3
        # comments before the real chunk shows up.
        await asyncio.sleep(0.4)
        yield "data: first\n\n"
        yield "data: [DONE]\n\n"

    class _FakeRequest:
        async def is_disconnected(self) -> bool:
            return False

    async def _run():
        out = []
        async for chunk in _disconnect_guard(
            _slow_generator(), _FakeRequest(), keepalive_seconds=0.1
        ):
            out.append(chunk)
        return out

    chunks = asyncio.run(_run())
    keepalives = [c for c in chunks if c.startswith(": keepalive")]
    assert len(keepalives) >= 2, (
        f"expected >=2 keepalive comments before the real chunk; "
        f"observed chunks={chunks}"
    )
    # Comments MUST be in the canonical SSE shape (``:`` prefix +
    # blank line terminator) so spec-conforming consumers ignore them.
    for c in keepalives:
        assert c == ": keepalive\n\n", c
    # Real data eventually arrives too.
    assert "data: first\n\n" in chunks
    assert "data: [DONE]\n\n" in chunks


def test_disconnect_guard_keepalive_can_be_disabled():
    """Operator escape hatch: ``RAPID_MLX_SSE_KEEPALIVE_SECONDS=0``
    (mapped to ``keepalive_seconds=0``) must disable the heartbeat
    entirely. Otherwise an operator with generous upstream proxy
    timeouts can't opt out of the per-stream comment-line overhead."""
    from vllm_mlx.service.helpers import _disconnect_guard

    chunk_seen = asyncio.Event()

    async def _slow_generator():
        # Wait until the test has had time to observe the absence of
        # keepalives, then yield a single real chunk so the guard
        # exits cleanly.
        await chunk_seen.wait()
        yield "data: end\n\n"

    class _FakeRequest:
        async def is_disconnected(self) -> bool:
            return False

    async def _run():
        out = []

        # ``keepalive_seconds=0`` disables the heartbeat regardless
        # of how long the upstream stalls.
        async def _consume():
            async for c in _disconnect_guard(
                _slow_generator(), _FakeRequest(), keepalive_seconds=0.0
            ):
                out.append(c)

        task = asyncio.create_task(_consume())
        # Give the disconnect-guard plenty of time to emit a keepalive
        # if the disable knob is broken.
        await asyncio.sleep(0.3)
        chunk_seen.set()
        await task
        return out

    chunks = asyncio.run(_run())
    assert not any(c.startswith(": keepalive") for c in chunks), chunks
    assert chunks == ["data: end\n\n"]


def test_disconnect_guard_keepalive_reads_serverconfig_default():
    """When no ``keepalive_seconds`` is passed, ``_disconnect_guard``
    must consult the live ``ServerConfig`` singleton. This is what
    routes do — none of them thread the parameter through, and the
    env-var resolution path lives at server bootstrap. Pin the
    behaviour so a future refactor that moves config-resolution
    elsewhere can't silently regress to "always 20 s".
    """
    from vllm_mlx.config.server_config import get_config
    from vllm_mlx.service.helpers import _disconnect_guard

    # Pin a small but non-zero keepalive on the config singleton.
    get_config().sse_keepalive_seconds = 0.05

    async def _slow_generator():
        await asyncio.sleep(0.2)
        yield "data: end\n\n"

    class _FakeRequest:
        async def is_disconnected(self) -> bool:
            return False

    async def _run():
        out = []
        async for chunk in _disconnect_guard(_slow_generator(), _FakeRequest()):
            out.append(chunk)
        return out

    chunks = asyncio.run(_run())
    keepalives = [c for c in chunks if c.startswith(": keepalive")]
    assert len(keepalives) >= 2, chunks


def test_disconnect_guard_releases_engine_admission_with_keepalives():
    """The admission-release safety net must still fire on the
    keepalive path. A regression that forgot to thread ``finally``
    cleanup through the new ``timeout=``-branch would slowly leak
    engine admission slots until restart.

    We assert by passing an engine stub whose
    ``release_admission_reservation`` flips a flag, then driving a
    full keepalive-then-data cycle.
    """
    from vllm_mlx.service.helpers import _disconnect_guard

    released = {"value": False}

    class _Engine:
        def release_admission_reservation(self) -> None:
            released["value"] = True

    async def _slow_generator():
        await asyncio.sleep(0.15)
        yield "data: hi\n\n"

    class _FakeRequest:
        async def is_disconnected(self) -> bool:
            return False

    async def _run():
        async for _ in _disconnect_guard(
            _slow_generator(),
            _FakeRequest(),
            engine=_Engine(),
            keepalive_seconds=0.05,
        ):
            pass

    asyncio.run(_run())
    assert released["value"] is True


def test_disconnect_guard_does_not_eagerly_prefetch_after_yield():
    """Codex r3 BLOCKING on PR #732 — pre-fix, the guard scheduled
    the next ``__anext__()`` IMMEDIATELY after ``yield chunk`` (via
    ``asyncio.ensure_future(aiter.__anext__())``). That gave the
    upstream generator a head-start of one item: the next token's
    compute could run on the event loop while the response stream's
    consumer was still chewing through the previous yield. If the
    consumer was about to disconnect, the wasted token represented
    GPU work shipped after the client gave up.

    Post-fix the guard re-creates the in-flight future LAZILY at the
    TOP of the next iteration — only AFTER the consumer has pulled
    the previous chunk via ``__anext__`` on the wrapper. We pin the
    contract by instrumenting the upstream generator with a per-yield
    counter and asserting it never advances ahead of the consumer's
    pull count.
    """
    from vllm_mlx.service.helpers import _disconnect_guard

    upstream_pulls = {"value": 0}

    async def _instrumented_generator():
        for token in ["a", "b", "c"]:
            upstream_pulls["value"] += 1
            yield f"data: {token}\n\n"

    class _FakeRequest:
        async def is_disconnected(self) -> bool:
            return False

    real_chunk_pulls = 0
    upstream_at_consume: list[int] = []

    async def _run():
        nonlocal real_chunk_pulls
        agen = _disconnect_guard(
            _instrumented_generator(),
            _FakeRequest(),
            keepalive_seconds=60.0,
        )
        async for chunk in agen:
            # R15 #291: the wrapper also emits a one-shot keepalive
            # right after the first real chunk. That keepalive is
            # generated entirely inside the wrapper, with no upstream
            # ``__anext__`` involvement, so the prefetch invariant we
            # care about is sampled only on REAL data chunks.
            if chunk.startswith(": keepalive"):
                continue
            real_chunk_pulls += 1
            # Sample the upstream's pull counter at the moment the
            # consumer received a chunk. A regression that re-introduces
            # eager prefetch would have ``upstream_pulls`` jump ahead
            # of ``consumer_pulls`` by 1 (the queued-but-unconsumed
            # item).
            upstream_at_consume.append(upstream_pulls["value"])
            # Brief pause so any eager-prefetch regression has time
            # to actually advance the upstream on the event loop
            # before the consumer's next ``__anext__``.
            await asyncio.sleep(0.01)

    asyncio.run(_run())

    # Consumer pulled 3 real chunks (a/b/c). Upstream MUST not be
    # ahead — every snapshot equals the consumer count at that point.
    assert real_chunk_pulls == 3, real_chunk_pulls
    assert upstream_at_consume == [1, 2, 3], upstream_at_consume


def test_disconnect_guard_emits_keepalive_immediately_after_first_chunk():
    """R15 #291 / #308 (Vlad 8k-prompt boundary fix): the very first
    real upstream chunk on chat-completions SSE is a synthetic
    ``role`` delta that fires almost instantly (~30 ms). The next
    chunk is the first content token, which is gated on prefill
    completion. Pre-fix, on prompts whose prefill takes ~``keepalive_seconds``
    (the 8 k-token boundary case), ``asyncio.wait`` saw ``anext_task``
    complete in the ``done`` set and SKIPPED the keepalive branch:
    client got the role chunk, then ``keepalive_seconds`` of silence,
    then the first content delta. Any HTTP client with idle timeout <
    ``keepalive_seconds`` tore the connection down mid-prefill.

    Post-fix the guard emits a single SSE comment line IMMEDIATELY
    after the first real chunk. Comments are spec no-ops, so SDK
    parsers ignore them — but TCP-level proxies / EventSource pools
    see the bytes and reset their idle timer."""
    from vllm_mlx.service.helpers import _disconnect_guard

    role_chunk = 'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
    content_chunk = 'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'

    async def _role_then_long_prefill():
        # The role chunk fires fast (mirror real upstream's ~30 ms).
        yield role_chunk
        # Then the upstream stalls (= prefill in progress). We park
        # well under the ``keepalive_seconds`` boundary on purpose:
        # the post-role-chunk keepalive must fire from the FIRST-CHUNK
        # branch, NOT from the stall-timeout branch. If we relied only
        # on the stall-timeout branch, this assertion would never fire
        # because the second chunk arrives before the timer ticks.
        await asyncio.sleep(0.05)
        yield content_chunk

    class _FakeRequest:
        async def is_disconnected(self) -> bool:
            return False

    async def _run():
        out = []
        events_t = []
        t0 = time.monotonic()
        # ``keepalive_seconds`` is LONGER than the upstream's ~50 ms
        # stall — so the stall-timeout branch CANNOT contribute the
        # observed keepalive. Only the post-first-chunk emit can.
        async for chunk in _disconnect_guard(
            _role_then_long_prefill(),
            _FakeRequest(),
            keepalive_seconds=10.0,
        ):
            out.append(chunk)
            events_t.append(time.monotonic() - t0)
        return out, events_t

    chunks, event_times = asyncio.run(_run())

    # Sequence MUST be: role chunk, keepalive comment, content chunk.
    assert chunks[0] == role_chunk, chunks
    assert chunks[1] == ": keepalive\n\n", chunks
    assert chunks[2] == content_chunk, chunks

    # The keepalive must land within 1 s of the role chunk — pre-fix
    # it could be ``keepalive_seconds`` (20 s default) away.
    delay_role_to_keepalive = event_times[1] - event_times[0]
    assert delay_role_to_keepalive < 1.0, (
        f"keepalive lagged {delay_role_to_keepalive:.3f} s behind role "
        f"chunk; R15 #291 contract requires <1 s"
    )


def test_disconnect_guard_post_role_keepalive_respects_disable_knob():
    """Operator escape hatch contract: with
    ``RAPID_MLX_SSE_KEEPALIVE_SECONDS=0`` (``keepalive_seconds=0``),
    the post-role-chunk keepalive emit must ALSO be suppressed.
    Otherwise an operator who explicitly opted out of heartbeats
    still pays the per-stream extra frame, contradicting the F-070
    disable contract."""
    from vllm_mlx.service.helpers import _disconnect_guard

    async def _two_chunks():
        yield "data: a\n\n"
        await asyncio.sleep(0.05)
        yield "data: b\n\n"

    class _FakeRequest:
        async def is_disconnected(self) -> bool:
            return False

    async def _run():
        out = []
        async for chunk in _disconnect_guard(
            _two_chunks(), _FakeRequest(), keepalive_seconds=0.0
        ):
            out.append(chunk)
        return out

    chunks = asyncio.run(_run())
    assert chunks == ["data: a\n\n", "data: b\n\n"], chunks
    assert not any(c.startswith(": keepalive") for c in chunks)


def test_disconnect_guard_16k_long_stall_keepalive_cadence_regression():
    """R15 #291 regression guard: the longer-prefill case (16 k-prompt
    equivalent in this test — multiple stall windows past the
    keepalive interval) must STILL produce one keepalive per stall
    window. Pre-fix this worked already; the #291 fix must not
    inadvertently change the steady-state cadence into "one and done".

    Models the 16 k case: the upstream stalls for ~4×``keepalive_seconds``
    before producing the first content token. We expect:
      * 1 post-first-chunk keepalive (R15 #291 fix)
      * >=3 stall-timeout keepalives (steady-state F-070 cadence)
    """
    from vllm_mlx.service.helpers import _disconnect_guard

    async def _role_then_very_long_prefill():
        yield "data: role\n\n"
        # ~4 keepalive periods of stall — mirrors the 16 k case where
        # multiple cadence ticks fire before first content.
        await asyncio.sleep(0.4)
        yield "data: first_content\n\n"

    class _FakeRequest:
        async def is_disconnected(self) -> bool:
            return False

    async def _run():
        out = []
        async for chunk in _disconnect_guard(
            _role_then_very_long_prefill(),
            _FakeRequest(),
            keepalive_seconds=0.1,
        ):
            out.append(chunk)
        return out

    chunks = asyncio.run(_run())
    keepalives = [c for c in chunks if c.startswith(": keepalive")]
    # At least 4 keepalives: 1 from the post-first-chunk emit + ~3
    # from the stall-timeout branch (0.4 s / 0.1 s = 4 ticks, minus
    # scheduling jitter).
    assert len(keepalives) >= 4, (
        f"expected >=4 keepalives (1 post-role + >=3 cadence); "
        f"got {len(keepalives)}: {chunks}"
    )
    # Real frames still made it through.
    assert "data: role\n\n" in chunks
    assert "data: first_content\n\n" in chunks
    # First emission order is preserved: role then keepalive.
    assert chunks[0] == "data: role\n\n"
    assert chunks[1] == ": keepalive\n\n"


def test_disconnect_guard_keepalive_does_not_break_disconnect_detection():
    """Critical: the keepalive loop must NOT swallow client-
    disconnect events. The pre-existing ``_wait_disconnect`` task
    polls every ``poll_interval`` seconds, and the wait races both
    tasks alongside the keepalive timer — a regression in the wait
    ordering could leave a disconnected client emitting heartbeats
    into a closed socket forever.
    """
    from vllm_mlx.service.helpers import _disconnect_guard

    disconnect_after = 0.25

    class _StoppingRequest:
        def __init__(self):
            self._t0 = time.monotonic()

        async def is_disconnected(self) -> bool:
            return time.monotonic() - self._t0 > disconnect_after

    async def _slow_generator():
        # Forever-stalling generator — only the disconnect signal can
        # break us out.
        await asyncio.sleep(5.0)
        yield "data: never\n\n"

    async def _run():
        out = []
        # poll_interval lower than disconnect_after so the disconnect
        # task fires within the test's expected window.
        async for chunk in _disconnect_guard(
            _slow_generator(),
            _StoppingRequest(),
            poll_interval=0.05,
            keepalive_seconds=0.05,
        ):
            out.append(chunk)
        return out

    t0 = time.monotonic()
    chunks = asyncio.run(_run())
    elapsed = time.monotonic() - t0
    assert elapsed < 2.0, (
        f"guard did not exit promptly on client disconnect; elapsed={elapsed:.2f}s"
    )
    # We may have emitted a few keepalives before the disconnect, but
    # NOT the forever-stalled real chunk.
    assert "data: never\n\n" not in chunks
