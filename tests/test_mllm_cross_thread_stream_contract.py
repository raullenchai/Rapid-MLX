"""Contract: the cross-thread ``Stream(gpu, N)`` shim stays even on the
post-bump floor (mlx ``0.31.2`` + mlx-lm ``0.31.3``).

Background
----------
The 0.7.41 hotfix (#720) swapped two ``mx.new_stream(...)`` allocations
to ``mx.default_stream(mx.default_device())`` in:

  * ``vllm_mlx/engine_core.py``   — ``_init_mlx_step_thread``
  * ``vllm_mlx/mllm_batch_generator.py`` — ``MLLMBatchGenerator.__init__``

…because mlx-lm 0.31.3 made ``generation_stream`` thread-local
(``mx.new_thread_local_stream`` in ``mlx_lm/generate.py``), and any
``mx.array`` produced under a thread-local / new stream on the
producer thread crashes with ``There is no Stream(gpu, N) in current
thread`` when it is materialised (``np.array(...)`` / lazy
``mx.eval``) on a different consumer thread (here, the asyncio route
handler reading per-step ``logprobs``).

The R15 floor bump to mlx ``0.31.2`` + mlx-lm ``0.31.3`` adds new
APIs — ``mx.new_thread_local_stream``, ``mx.ThreadLocalStream``,
``mx.clear_streams`` — that were *expected* to fix this, but
empirical repro on 0.31.2 confirms the crash persists across all
three of ``default_stream`` / ``new_stream`` / ``new_thread_local_stream``
producers when the consumer thread materialises via ``np.array(arr)``.

The only path that round-trips cleanly across threads remains the
process-wide default stream — exactly what the shim uses. The
assertions below pin that contract so a future refactor cannot
quietly revert it without the test suite calling it out by name.

The existing ``tests/test_mllm_logprobs_plumbing.py::
test_mllm_batch_generator_init_does_not_call_new_stream`` already
exercises the runtime behaviour by trapping ``mx.new_stream``; this
file is the static / lexical counterpart so the contract is also
documented in source against the bumped floor.

Codex r1 review (BLOCKING ×2) on this file's first draft surfaced two
substring-check loopholes that this revision closes:

  1. ``mx.default_stream(other_device)`` would have satisfied a naive
     ``"mx.default_stream" in src`` check while violating the contract
     (the shim REQUIRES ``mx.default_device()`` as the argument so the
     adopted stream is process-wide rather than per-device-name).
  2. ``from mlx.core import new_stream`` followed by a bare
     ``new_stream(...)`` call would have slipped past a naive
     ``"mx.new_stream" not in src`` check while reintroducing the
     crash path.

The current revision walks the function body with ``ast`` and:

  * REQUIRES at least one ``Call`` whose qualified callee is
    ``mx.default_stream`` AND whose first positional argument is the
    ``mx.default_device()`` call.
  * REJECTS any ``Call`` whose qualified callee resolves to
    ``mx.new_stream`` / ``mx.new_thread_local_stream``, including the
    ``from mlx.core import new_stream`` direct-import form.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from collections.abc import Iterable

import vllm_mlx.engine_core as engine_core
import vllm_mlx.mllm_batch_generator as mllm_batch_generator


# Symbols that resolve to the forbidden upstream stream-allocation APIs
# under any common spelling. ``mlx.core`` is the canonical import root;
# ``mx`` and ``mlx_core`` are the conventional aliases used across this
# repo. Add to this set if a new alias is introduced.
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
        f(x)(...)               -> None  (call-of-call; out of scope)
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


def _iter_calls(tree: ast.AST) -> Iterable[ast.Call]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            yield node


def _imported_names(module_ast: ast.AST) -> set[str]:
    """Collect names brought into module scope via ``from mlx.core import …``
    (with or without an alias). Used to detect the loophole where
    ``new_stream`` is called bare after a direct import.
    """
    out: set[str] = set()
    for node in ast.walk(module_ast):
        if isinstance(node, ast.ImportFrom) and node.module in {
            "mlx",
            "mlx.core",
        }:
            for alias in node.names:
                out.add(alias.asname or alias.name)
    return out


def _module_source(module: object) -> str:
    return textwrap.dedent(inspect.getsource(module))


def _assert_forbidden_factories_absent(
    func_src: str,
    module_src: str,
    where: str,
) -> None:
    func_tree = ast.parse(textwrap.dedent(func_src))
    module_tree = ast.parse(module_src)
    direct_imports = _imported_names(module_tree)

    for call in _iter_calls(func_tree):
        qname = _qualified_call_name(call)
        if qname is None:
            continue
        head, _, tail = qname.partition(".")
        # Form 1: mx.new_stream(...) / mlx_core.new_thread_local_stream(...)
        if head in _MX_ALIASES and tail in _FORBIDDEN_STREAM_FACTORIES:
            raise AssertionError(
                f"{where} contains a forbidden call ``{qname}(...)``. "
                "Under mlx-lm 0.31.3 the resulting stream is bound to "
                "the constructing thread; cross-thread np.array(...) "
                "on the produced mx.array crashes the consumer thread "
                "with `There is no Stream(gpu, N) in current thread.` "
                "(#720). Stay on ``mx.default_stream(mx.default_device())``."
            )
        # Form 2: bare `new_stream(...)` after `from mlx.core import new_stream`
        if (
            tail == ""  # qname is a bare name
            and head in _FORBIDDEN_STREAM_FACTORIES
            and head in direct_imports
        ):
            raise AssertionError(
                f"{where} contains a forbidden call ``{head}(...)`` "
                "imported directly from ``mlx.core``. This bypasses the "
                "``mx.<factory>`` namespace check but tags arrays with "
                "the same thread-bound stream that crashes the "
                "route-handler thread on cross-thread materialisation "
                "(#720). Stay on ``mx.default_stream(mx.default_device())``."
            )


def _assert_default_stream_with_default_device(
    func_src: str,
    where: str,
) -> None:
    """Require the function body to contain at least one call shaped
    exactly like ``mx.default_stream(mx.default_device())`` (or
    equivalently spelled via ``mlx.core`` / direct import). This is the
    only allocation that round-trips across threads.
    """
    func_tree = ast.parse(textwrap.dedent(func_src))

    for call in _iter_calls(func_tree):
        outer_name = _qualified_call_name(call)
        if outer_name is None:
            continue
        # Accept ``mx.default_stream`` / ``mlx_core.default_stream`` / bare
        # ``default_stream`` (after a from-import). The bare form is rare
        # but should not be artificially blocked here — what we're pinning
        # is the *shape* of the call.
        head, _, tail = outer_name.partition(".")
        is_default_stream = (
            (head in _MX_ALIASES and tail == "default_stream")
            or outer_name == "default_stream"
        )
        if not is_default_stream:
            continue
        if not call.args:
            continue
        first_arg = call.args[0]
        if not isinstance(first_arg, ast.Call):
            continue
        inner_name = _qualified_call_name(first_arg)
        if inner_name is None:
            continue
        ihead, _, itail = inner_name.partition(".")
        is_default_device = (
            (ihead in _MX_ALIASES and itail == "default_device")
            or inner_name == "default_device"
        )
        if is_default_device:
            return  # contract satisfied

    raise AssertionError(
        f"{where} must call ``mx.default_stream(mx.default_device())`` "
        "(the process-wide default stream that round-trips cleanly "
        "across threads). Any other argument — including a specific "
        "device-name or a thread-local stream — leaves logprobs "
        "mx.array crashing the route-handler thread on cross-thread "
        "np.array(...) materialisation. See #720."
    )


def test_engine_core_init_mlx_step_thread_uses_default_stream() -> None:
    """``_init_mlx_step_thread`` must call
    ``mx.default_stream(mx.default_device())`` and must NOT call
    ``mx.new_stream`` / ``mx.new_thread_local_stream`` (or their
    direct-import equivalents) even after the floor bump to mlx 0.31.2.
    """
    func_src = inspect.getsource(engine_core._init_mlx_step_thread)
    module_src = _module_source(engine_core)
    where = "engine_core._init_mlx_step_thread"
    _assert_forbidden_factories_absent(func_src, module_src, where)
    _assert_default_stream_with_default_device(func_src, where)


def test_mllm_batch_generator_init_uses_default_stream() -> None:
    """``MLLMBatchGenerator.__init__`` must call
    ``mx.default_stream(mx.default_device())`` for the same reason —
    the per-step logprob slice it produces is consumed on the
    route-handler thread.
    """
    func_src = inspect.getsource(mllm_batch_generator.MLLMBatchGenerator.__init__)
    module_src = _module_source(mllm_batch_generator)
    where = "MLLMBatchGenerator.__init__"
    _assert_forbidden_factories_absent(func_src, module_src, where)
    _assert_default_stream_with_default_device(func_src, where)
