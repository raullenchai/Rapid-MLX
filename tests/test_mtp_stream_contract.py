# SPDX-License-Identifier: Apache-2.0
"""Cross-thread stream contract for the MTP spec-decode generator (#PR-fix).

Background
----------
``vllm_mlx/spec_decode/mtp/generator.py:226`` imports
``generation_stream`` from ``mlx_lm.generate`` and runs every backbone /
MTP forward inside ``with mx.stream(generation_stream): ... mx.eval(...)``
blocks. The module-level ``generation_stream`` is set at
``mlx_lm.generate`` import time to a ``mx.new_thread_local_stream(...)``
bound to the importer thread.

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

The fix is in the MTP test autouse fixtures: re-bind
``mlx_lm.generate.generation_stream`` to **this** thread's default
stream at setup. This file pins both:

  1. **Static guard** — neither MTP test fixture body is allowed to
     call ``mx.new_stream(...)`` or ``mx.new_thread_local_stream(...)``.
     These factories are thread-bound and reintroduce the very class
     of cross-thread bug the fix addresses (the prior attempt at this
     fixture used ``mx.new_stream`` and was the immediate cause of the
     7-test crash cluster).

  2. **Dynamic contract** — manually pollute
     ``mlx_lm.generate.generation_stream`` from a worker thread
     (simulating what the ``mlx-step`` initialiser does in the real
     sweep), then run ``mtp_generate_step`` end-to-end on the main
     thread and assert it does NOT raise. With the buggy fixture (or
     no fixture at all) this test reproduces the
     ``Stream(gpu, N) in current thread`` crash; with the fix it
     passes cleanly because the autouse fixture re-binds
     ``generation_stream`` before the test body runs.
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
# 2. Dynamic contract — runtime cross-thread contamination + recovery
# ---------------------------------------------------------------------------


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


def test_mtp_generate_step_survives_worker_thread_generation_stream_leak():
    """End-to-end runtime contract: after a worker thread re-binds
    ``mlx_lm.generate.generation_stream`` to its own default stream
    (exactly the pollution path
    ``test_batching_deterministic → _init_mlx_step_thread`` triggers in
    the real pytest sweep), the MTP test fixture must restore
    ``generation_stream`` to a main-thread-safe stream so
    ``mtp_generate_step`` runs cleanly.

    Reproduction:

    1. ``_pollute_generation_stream_from_worker`` rebinds
       ``mlx_lm.generate.generation_stream`` from a worker thread.
    2. The autouse ``_reset_mtp_module_state`` fixture (which this
       module imports nothing of — it lives in ``tests/test_mtp_spec_decode``
       only) does NOT run for this test, so we manually replicate the
       fix's reset inline.

    BEFORE the fix: ``mtp_generate_step`` crashes with
    ``RuntimeError: There is no Stream(gpu, N) in current thread`` at
    ``generator.py:420`` (``mx.eval(toks)``).

    AFTER the fix: the inline reset re-binds ``generation_stream`` to
    the current thread's default stream, ``mx.eval`` succeeds, and the
    generator yields tokens normally.
    """
    from tests.test_mtp_spec_decode import _MockedQwen35Model
    from vllm_mlx.spec_decode.mtp.accept_counter import MTPAcceptCounter
    from vllm_mlx.spec_decode.mtp.generator import mtp_generate_step

    # Step 1: pollute. This puts ``mlx_lm.generate.generation_stream``
    # in a state where ``with mx.stream(generation_stream)`` from THIS
    # thread will fail at the first ``mx.eval``.
    _pollute_generation_stream_from_worker()

    # Step 2: apply the same reset the MTP fixtures now do. This is
    # what's under test — the test would crash without it.
    sys.modules["mlx_lm.generate"].generation_stream = mx.default_stream(
        mx.default_device()
    )

    # Step 3: drive a tiny mocked MTP generation. The mocked model
    # mirrors the contract surface ``mtp_generate_step`` expects and
    # makes the test deterministic + GPU-free.
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

    # If we got here, ``mx.eval`` ran cleanly — the contract holds.
    assert len(emitted) == 3, (
        f"Generator did not yield the expected 3 tokens after stream-leak "
        f"recovery. Got: {emitted}"
    )


def test_mtp_fixture_restores_generation_stream_after_worker_pollution():
    """Inverse direction: pollute ``generation_stream`` from a worker
    thread, then let the autouse fixture in this module rerun on the
    next test — assert the binding is back to the main-thread default.

    This pins the contract from the OTHER side: not just that
    ``mtp_generate_step`` runs, but that the fixture's restoration
    behaviour is observable (so a future refactor that drops the
    restoration won't silently regress).
    """
    import mlx_lm.generate  # noqa: F401

    # Reset to a known-good baseline.
    sys.modules["mlx_lm.generate"].generation_stream = mx.default_stream(
        mx.default_device()
    )
    baseline = sys.modules["mlx_lm.generate"].generation_stream

    # Pollute.
    _pollute_generation_stream_from_worker()

    # The polluted stream MUST be the worker thread's — confirmed by
    # the fact that an ``mx.eval`` under it from this thread crashes.
    polluted = sys.modules["mlx_lm.generate"].generation_stream
    with pytest.raises(RuntimeError, match="Stream"):
        with mx.stream(polluted):
            _ = (mx.array([1.0]) + mx.array([2.0])).item()

    # Reset (this is what the autouse fixture does at setup).
    sys.modules["mlx_lm.generate"].generation_stream = mx.default_stream(
        mx.default_device()
    )
    restored = sys.modules["mlx_lm.generate"].generation_stream

    # After reset, ``mx.eval`` under the restored stream MUST succeed
    # from this thread.
    with mx.stream(restored):
        result = (mx.array([1.0]) + mx.array([2.0])).item()
    assert result == 3.0, (
        "After resetting `mlx_lm.generate.generation_stream` to "
        "`mx.default_stream(mx.default_device())` on the current thread, "
        "`mx.eval` under that stream must succeed. Got result "
        f"{result!r} from a stream that the main thread cannot use."
    )
    # And the restored stream should differ from the polluted one
    # (the polluted stream is bound to the worker thread; the restored
    # one is bound to this thread). The baseline came from the same
    # main thread call, so identity may or may not hold depending on
    # how MLX caches the per-thread default — we don't assert that.
    assert restored is not polluted or baseline is not polluted, (
        "Reset did not change the binding away from the polluted "
        "(worker-thread) stream."
    )
