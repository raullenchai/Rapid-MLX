# SPDX-License-Identifier: Apache-2.0
"""Cross-thread stream contract for the MTP spec-decode generator (#PR-fix).

Background
----------
The vendored MTP generator runs every backbone/MTP forward inside
``with mx.stream(generation_stream): ... mx.eval(...)`` blocks. mlx-lm's
module-level ``generation_stream`` is mutable process state: each resident
engine worker binds it to that worker's thread-local default stream.

Production paths route around this via
``engine_core._init_mlx_step_thread`` which, when the ``mlx-step``
executor worker spins up, re-assigns ``generation_stream`` to
``mx.default_stream(mx.default_device())``. **However** —
``mx.default_stream(device)`` returns the **current thread's** default
stream, not a process-wide stream. So when a pytest sweep test (e.g.
``test_batching_deterministic``) creates its own ``mlx-step`` worker
executor with ``_init_mlx_step_thread`` as initialiser, that worker
silently re-binds ``mlx_lm.generate.generation_stream`` to its OWN
default stream. After the worker shuts down, any subsequent test that
runs ``mtp_generate_step`` on the pytest main thread crashes at
``generator.py:420`` with::

    RuntimeError: There is no Stream(gpu, N) in current thread.

The production fix binds ``mx.default_stream(mx.default_device())`` when the
MTP generator begins execution on its scheduler-owned step thread, instead of
consuming that mutable global. Test fixtures still restore the global for
older direct mlx-lm paths exercised in the suite. This file pins both:

  1. **Static guard** — neither MTP test fixture body is allowed to
     call ``mx.new_stream(...)`` or ``mx.new_thread_local_stream(...)``.
     These factories are thread-bound and reintroduce the very class
     of cross-thread bug the fix addresses (the prior attempt at this
     fixture used ``mx.new_stream`` and was the immediate cause of the
     7-test crash cluster).

  2. **Dynamic production contract** — manually pollute
     ``mlx_lm.generate.generation_stream`` from a worker thread (the second
     resident load), then run ``mtp_generate_step`` directly without a
     fixture reset. It must ignore the foreign stream and complete cleanly.
"""

from __future__ import annotations

import ast
import inspect
import sys
import textwrap
import threading
from collections.abc import Iterable

import pytest

mx = pytest.importorskip("mlx.core")


# ---------------------------------------------------------------------------
# 1. Static guard — no thread-bound stream factories in the MTP fixtures
# ---------------------------------------------------------------------------

_FORBIDDEN_STREAM_FACTORIES = frozenset(
    {
        "new_stream",
        "new_thread_local_stream",
    }
)
_MX_ALIASES = frozenset({"mx", "mlx_core", "mlx"})


def _qualified_call_name(call: ast.Call) -> str | None:
    """Return the dotted callee name for a ``Call`` node, or ``None``
    if it's not a simple ``Name`` / ``Attribute`` chain.

    Examples:

        mx.default_stream(...)  -> "mx.default_stream"
        new_stream(...)         -> "new_stream"
    """
    parts: list[str] = []
    node: ast.AST = call.func
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _walk_calls(tree: ast.AST) -> Iterable[ast.Call]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            yield node


def _fixture_source(test_module_name: str) -> str:
    """Return the source of the ``_reset_mtp_module_state`` autouse fixture
    defined in the named test module."""
    mod = __import__(test_module_name, fromlist=["_reset_mtp_module_state"])
    fixture_obj = mod._reset_mtp_module_state
    # pytest wraps fixtures; the actual function is usually the wrapped
    # callable, but ``inspect.getsource`` walks through transparently for
    # both wrapped and bare functions.
    return textwrap.dedent(inspect.getsource(fixture_obj))


@pytest.mark.parametrize(
    "test_module_name",
    ["tests.test_mtp_spec_decode", "tests.test_mtp_lossless"],
)
def test_mtp_fixture_does_not_call_thread_bound_stream_factories(
    test_module_name: str,
):
    """The autouse ``_reset_mtp_module_state`` fixtures in both MTP test
    files MUST NOT call ``mx.new_stream`` or ``mx.new_thread_local_stream``.

    Both factories return a stream bound to the calling thread. The prior
    "fix" used ``mx.new_stream(mx.default_device())`` to allocate a stream
    in the pytest main thread and pinned it as the active default — but
    ``mtp_generate_step`` doesn't use the active default; it uses
    ``mlx_lm.generate.generation_stream`` via a ``with mx.stream(...)``
    block. The old fixture therefore left the bug in place AND introduced
    a new resource leak (one stream allocated per test).

    The canonical safe pattern is ``mx.default_stream(mx.default_device())``
    — returns the current thread's default stream, which can be ``mx.eval``'d
    from this thread by definition.
    """
    src = _fixture_source(test_module_name)
    tree = ast.parse(src)

    offending: list[str] = []
    for call in _walk_calls(tree):
        name = _qualified_call_name(call)
        if name is None:
            continue
        # Dotted form (e.g. ``mx.new_stream``).
        if "." in name:
            module, fn = name.rsplit(".", 1)
            if module in _MX_ALIASES and fn in _FORBIDDEN_STREAM_FACTORIES:
                offending.append(name)
        # Bare form (``from mlx.core import new_stream``).
        elif name in _FORBIDDEN_STREAM_FACTORIES:
            offending.append(name)

    assert not offending, (
        f"{test_module_name} fixture calls forbidden thread-bound stream "
        f"factory: {offending!r}. These allocate a stream bound to the "
        f"caller thread and reintroduce the cross-thread "
        f"`There is no Stream(gpu, N) in current thread` crash. Use "
        f"`mx.default_stream(mx.default_device())` to re-bind "
        f"`mlx_lm.generate.generation_stream` instead."
    )


# ---------------------------------------------------------------------------
# 2. Dynamic contract — drive the actual MTP autouse fixtures
# ---------------------------------------------------------------------------
#
# These tests do NOT inline-reset ``generation_stream``. They directly drive
# the fixture functions from ``test_mtp_spec_decode`` / ``test_mtp_lossless``
# as generators, pollute ``generation_stream`` BEFORE invoking ``next(gen)``,
# and then assert (a) the fixture's setup phase produced a stream this thread
# can ``mx.eval`` against, and (b) ``mtp_generate_step`` runs cleanly under
# the fixture-managed state. If the fixture's restoration logic is removed,
# ``next(gen)`` would leave the polluted stream in place and the subsequent
# assertions/``mtp_generate_step`` call would crash with the same
# ``RuntimeError: There is no Stream(gpu, N)`` the operator surfaced.


def _pollute_generation_stream_from_worker() -> None:
    """Replicate what a sweep test's ``mlx-step`` worker initialiser does:
    re-bind ``mlx_lm.generate.generation_stream`` from a worker thread."""
    # Lazily import to ensure ``mlx_lm.generate`` is in ``sys.modules``.
    import mlx_lm.generate  # noqa: F401

    def _worker() -> None:
        # Mirror engine_core._init_mlx_step_thread's reassignment.
        # ``mx.default_stream(device)`` is per-thread, so the assigned
        # stream is bound to THIS worker — exactly the leak the MTP
        # fixture must defend against.
        sys.modules["mlx_lm.generate"].generation_stream = mx.default_stream(
            mx.default_device()
        )

    t = threading.Thread(target=_worker, name="mlx-step-pollute")
    t.start()
    t.join()


def _unwrap_fixture_func(fixture_obj):
    """Return the bare generator function from a pytest fixture marker.

    ``@pytest.fixture(autouse=True)`` wraps the underlying function in a
    ``FixtureFunctionMarker``; the original generator function is
    available via the ``__wrapped__`` attribute (when supported by the
    pytest version) or via attribute walks. Falling through to the
    object itself is the safe default if it's already callable as a
    bare generator function.
    """
    candidate = fixture_obj
    for attr in ("__wrapped__", "func", "fn"):
        unwrapped = getattr(candidate, attr, None)
        if callable(unwrapped):
            candidate = unwrapped
            break
    return candidate


def _assert_main_thread_can_eval_under_current_generation_stream() -> None:
    """Asserts that ``mx.eval`` works on the main thread under the
    currently-set ``mlx_lm.generate.generation_stream``. Raises the
    underlying ``RuntimeError`` (``There is no Stream(gpu, N) in
    current thread``) if it doesn't — exactly the bug we're guarding
    against."""
    stream = sys.modules["mlx_lm.generate"].generation_stream
    with mx.stream(stream):
        out = mx.array([1.0]) + mx.array([2.0])
        mx.eval(out)
        assert out.item() == 3.0


@pytest.mark.parametrize(
    "test_module_name",
    ["tests.test_mtp_spec_decode", "tests.test_mtp_lossless"],
)
def test_mtp_fixture_setup_restores_generation_stream_after_worker_pollution(
    test_module_name: str,
):
    """End-to-end runtime contract: the actual autouse fixture's SETUP
    phase must restore ``mlx_lm.generate.generation_stream`` to a
    main-thread-safe stream, even when an earlier sweep test polluted
    it from a worker thread.

    We drive the fixture function directly (NOT via pytest's autouse
    machinery) so the assertion observes the FIXTURE behavior. No
    inline reset — if the fixture stops restoring ``generation_stream``,
    ``next(gen)`` leaves the worker stream in place and the
    ``mx.eval``-under-current-stream assertion raises.

    Reproduces ``test_batching_deterministic → _init_mlx_step_thread``
    triggers in the real pytest sweep.
    """
    mod = __import__(test_module_name, fromlist=["_reset_mtp_module_state"])
    fixture_func = _unwrap_fixture_func(mod._reset_mtp_module_state)

    # Pollute BEFORE the fixture's setup runs.
    _pollute_generation_stream_from_worker()
    polluted = sys.modules["mlx_lm.generate"].generation_stream

    # Confirm the polluted state is broken from this thread.
    with (
        pytest.raises(RuntimeError, match="Stream"),
        mx.stream(polluted),
    ):
        _ = (mx.array([1.0]) + mx.array([2.0])).item()

    # Drive the fixture's setup phase.
    gen = fixture_func()
    next(gen)
    try:
        # ASSERTION UNDER TEST: the fixture's setup MUST have rebound
        # ``generation_stream`` to a main-thread-safe stream. We do
        # NOT inline-reset here — if the fixture stopped restoring,
        # this assertion would raise the same RuntimeError the operator
        # bug report names.
        _assert_main_thread_can_eval_under_current_generation_stream()
    finally:
        # Run the fixture's teardown phase. ``next(gen)`` on a finished
        # generator raises StopIteration; that's expected.
        try:
            next(gen)
        except StopIteration:
            pass


def test_mtp_generate_step_owns_stream_after_second_resident_pollution():
    """A second resident's worker cannot poison the primary MTP generator.

    Loading another resident engine rebinds mlx-lm's process-global stream on
    that worker. The original MTP primary still executes on its own scheduler
    thread, so the generator must ignore the foreign global and bind that
    execution thread's default stream before any model/cache work.
    """
    _pollute_generation_stream_from_worker()

    from tests.test_mtp_spec_decode import _MockedQwen35Model
    from vllm_mlx.spec_decode.mtp.accept_counter import MTPAcceptCounter
    from vllm_mlx.spec_decode.mtp.generator import mtp_generate_step

    polluted = sys.modules["mlx_lm.generate"].generation_stream
    try:
        with pytest.raises(RuntimeError, match="Stream"), mx.stream(polluted):
            _ = (mx.array([1.0]) + mx.array([2.0])).item()

        backbone = [7, 11, 13]
        mtp_script = [11]
        model = _MockedQwen35Model(backbone, mtp_script)
        counter = MTPAcceptCounter()
        prompt = mx.array([1], dtype=mx.uint32)
        emitted = list(
            mtp_generate_step(
                prompt,
                model,
                max_tokens=3,
                accept_counter=counter,
            )
        )
        assert sys.modules["mlx_lm.generate"].generation_stream is polluted
        assert len(emitted) == 3, (
            "mtp_generate_step did not yield the expected 3 tokens — "
            "the generator did not bind the scheduler thread's stream. "
            f"Got: {emitted}"
        )
    finally:
        sys.modules["mlx_lm.generate"].generation_stream = mx.default_stream(
            mx.default_device()
        )


# ---------------------------------------------------------------------------
# 3. Production-code contract — stream ownership is established at execution
# ---------------------------------------------------------------------------


def test_mtp_generator_binds_execution_thread_default_stream():
    """The vendored generator must not consume mlx-lm's mutable global."""
    import vllm_mlx.spec_decode.mtp.generator as generator_mod

    assert not hasattr(generator_mod, "generation_stream"), (
        "the MTP module must not cache a process-global generation stream"
    )

    src = textwrap.dedent(inspect.getsource(generator_mod.mtp_generate_step))
    assert "from mlx_lm.generate import maybe_quantize_kv_cache" in src
    assert "from mlx_lm.generate import generation_stream" not in src
    assert "generation_stream = mx.default_stream(mx.default_device())" in src, (
        "mtp_generate_step must bind the scheduler execution thread's default "
        "stream before model/cache work"
    )
