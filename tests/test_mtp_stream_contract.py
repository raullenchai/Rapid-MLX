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


@pytest.mark.parametrize(
    "test_module_name",
    ["tests.test_mtp_spec_decode", "tests.test_mtp_lossless"],
)
def test_mtp_generate_step_survives_worker_pollution_via_fixture(
    test_module_name: str,
):
    """End-to-end: pollute ``generation_stream`` from a worker thread,
    drive the MTP autouse fixture's setup, then run ``mtp_generate_step``
    and assert it yields tokens cleanly.

    This is the highest-confidence regression guard: it exercises the
    EXACT code path that crashes in the failing sweep (``mtp_generate_step``
    on the pytest main thread, after a worker re-bound
    ``generation_stream``) and routes recovery through the fixture
    function under test. No inline reset.

    Codex r2 BLOCKING defense — codex worried that importing
    ``mtp_generate_step`` BEFORE pollution would let ``generator.py``
    capture a pre-pollution stream and mask a broken fixture. The
    concern is empirically false on the current production code path
    because ``generator.py:226`` does
    ``from mlx_lm.generate import generation_stream`` INSIDE
    ``mtp_generate_step``'s body (function-scope, re-read on every
    call) — verified by
    ``test_mtp_generator_reads_generation_stream_at_call_time``
    below. But we still order the imports defensively (pollute FIRST,
    then import) so a future move of the import to module scope is
    caught by THIS test rather than silently masking the regression.
    """
    # Pollute BEFORE importing ``mtp_generate_step`` — defends against
    # any future regression that moves the ``from mlx_lm.generate
    # import generation_stream`` line out of the function body and
    # into module scope (see paired
    # ``test_mtp_generator_reads_generation_stream_at_call_time``).
    _pollute_generation_stream_from_worker()

    from tests.test_mtp_spec_decode import _MockedQwen35Model
    from vllm_mlx.spec_decode.mtp.accept_counter import MTPAcceptCounter
    from vllm_mlx.spec_decode.mtp.generator import mtp_generate_step

    mod = __import__(test_module_name, fromlist=["_reset_mtp_module_state"])
    fixture_func = _unwrap_fixture_func(mod._reset_mtp_module_state)

    gen = fixture_func()
    next(gen)
    try:
        # ``mtp_generate_step`` runs against ``mlx_lm.generate.generation_stream``
        # as set up by the fixture. If the fixture is broken, this crashes
        # at generator.py:420 (``mx.eval(toks)``).
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
        assert len(emitted) == 3, (
            "mtp_generate_step did not yield the expected 3 tokens — "
            "the autouse fixture's stream restoration logic is likely "
            f"broken. Got: {emitted}"
        )
    finally:
        try:
            next(gen)
        except StopIteration:
            pass


# ---------------------------------------------------------------------------
# 3. Production-code contract — the function-scope import is load-bearing
# ---------------------------------------------------------------------------


def test_mtp_generator_reads_generation_stream_at_call_time():
    """``mtp_generate_step`` MUST import ``generation_stream`` at
    function-call time (function-scope import), not at module-import
    time (module-scope ``from mlx_lm.generate import generation_stream``).

    The fixture's restoration works by rebinding
    ``sys.modules['mlx_lm.generate'].generation_stream``. That rebind
    only flows through to ``mtp_generate_step`` if the function looks
    up the attribute at every call. If a future refactor moves the
    import to module scope, the rebind no longer affects already-
    imported modules and the 7-test crash cluster comes back —
    silently, because the fixture would still LOOK like it's
    restoring the stream.

    Codex r2 BLOCKING #1 flagged this exact concern. Pin it here so
    the regression guard fires loudly if anyone hoists the import.

    Implementation:

    1. Module-scope: ``vllm_mlx.spec_decode.mtp.generator.generation_stream``
       attribute MUST NOT exist.
    2. Function source: ``mtp_generate_step``'s source MUST contain
       a ``from mlx_lm.generate import generation_stream`` line.

    Both checks are AST/source-text based — no runtime import order
    games.
    """
    import vllm_mlx.spec_decode.mtp.generator as generator_mod

    # 1. Module-scope check.
    assert not hasattr(generator_mod, "generation_stream"), (
        "`vllm_mlx.spec_decode.mtp.generator.generation_stream` exists "
        "as a module-level attribute. This means someone moved the "
        "`from mlx_lm.generate import generation_stream` import out of "
        "`mtp_generate_step`'s body and into module scope. That breaks "
        "the test fixture's restoration path because rebinding "
        "`sys.modules['mlx_lm.generate'].generation_stream` no longer "
        "affects already-imported modules. Move the import back inside "
        "`mtp_generate_step` (see generator.py:226 baseline)."
    )

    # 2. Function-source check.
    src = textwrap.dedent(inspect.getsource(generator_mod.mtp_generate_step))
    assert "from mlx_lm.generate import generation_stream" in src, (
        "`mtp_generate_step` no longer imports `generation_stream` at "
        "call time. The function-scope import is load-bearing — the "
        "test fixture's stream restoration relies on the function "
        "looking up `mlx_lm.generate.generation_stream` afresh on every "
        "call. See generator.py:226 baseline."
    )
