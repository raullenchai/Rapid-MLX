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
    any ``: keepalive`` comments — only the real chunks reach the
    client. Otherwise a fast-streaming model would get its bandwidth
    inflated with no-op comments."""
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
    assert chunks == ["data: a\n\n", "data: b\n\n", "data: c\n\n"]
    # And critically, NO keepalive comments leaked in.
    assert not any(c.startswith(": keepalive") for c in chunks)


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
