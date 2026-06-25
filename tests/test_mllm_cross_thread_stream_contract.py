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
process-wide default stream — exactly what the shim uses. These
assertions pin that contract so a future refactor cannot quietly
revert it without the test suite calling it out by name.

The existing ``tests/test_mllm_logprobs_plumbing.py::
test_mllm_batch_generator_init_does_not_call_new_stream`` already
exercises the runtime behaviour by trapping ``mx.new_stream``; this
file is the static / lexical counterpart so the contract is also
documented in source against the bumped floor.
"""

from __future__ import annotations

import inspect
import io
import textwrap
import tokenize

import vllm_mlx.engine_core as engine_core
import vllm_mlx.mllm_batch_generator as mllm_batch_generator


def _strip_comments_and_strings(src: str) -> str:
    """Return ``src`` with all comments and string literals replaced by
    whitespace, so substring checks below cannot get false-positive hits
    from the documentation that *explains* why the legacy APIs are
    forbidden (that documentation references the exact substrings we
    want to forbid in code).
    """
    src = textwrap.dedent(src)
    tokens = tokenize.generate_tokens(io.StringIO(src).readline)
    out: list[str] = []
    for tok in tokens:
        if tok.type in (tokenize.COMMENT, tokenize.STRING):
            continue
        out.append(tok.string)
    # Join with no separator so attribute chains (``mx . default_stream``)
    # collapse back to the source form (``mx.default_stream``) the
    # assertions below look for.
    return "".join(out)


def test_engine_core_init_mlx_step_thread_uses_default_stream() -> None:
    """``_init_mlx_step_thread`` must allocate via ``mx.default_stream``
    (process-wide default) and NOT ``mx.new_stream`` / ``mx.new_thread_local_stream``
    even after the floor bump to mlx 0.31.2.
    """
    src = inspect.getsource(engine_core._init_mlx_step_thread)
    code = _strip_comments_and_strings(src)
    assert "mx.default_stream" in code, (
        "engine_core._init_mlx_step_thread must keep the "
        "mx.default_stream shim — cross-thread Stream(gpu, N) eval is "
        "still broken on the bumped floor (mlx 0.31.2 + mlx-lm 0.31.3)."
    )
    assert "mx.new_stream" not in code, (
        "_init_mlx_step_thread re-introduced mx.new_stream — under "
        "mlx-lm 0.31.3 the resulting stream is bound to the worker "
        "thread and any array it tags crashes the route-handler "
        "thread on np.array(...) materialisation. See #720."
    )
    assert "mx.new_thread_local_stream" not in code, (
        "_init_mlx_step_thread switched to mx.new_thread_local_stream — "
        "0.31.2 added that API but empirical repro confirms it still "
        "crashes on cross-thread np.array(arr). Stay on "
        "mx.default_stream(mx.default_device())."
    )


def test_mllm_batch_generator_init_uses_default_stream() -> None:
    """``MLLMBatchGenerator.__init__`` must allocate ``_stream`` via
    ``mx.default_stream`` for the same reason — the per-step logprob
    slice it produces is consumed on the route-handler thread.
    """
    src = inspect.getsource(mllm_batch_generator.MLLMBatchGenerator.__init__)
    code = _strip_comments_and_strings(src)
    assert "mx.default_stream" in code, (
        "MLLMBatchGenerator.__init__ must keep the mx.default_stream "
        "shim — under mlx-lm 0.31.3 the logprobs mx.array crashes the "
        "route-handler thread on cross-thread np.array(...) if "
        "_stream is bound to the constructing worker thread. See #720."
    )
    assert "mx.new_stream" not in code, (
        "MLLMBatchGenerator.__init__ re-introduced mx.new_stream — "
        "see #720 and the existing runtime trap test in "
        "tests/test_mllm_logprobs_plumbing.py."
    )
    assert "mx.new_thread_local_stream" not in code, (
        "MLLMBatchGenerator.__init__ switched to mx.new_thread_local_stream — "
        "0.31.2 added the API but cross-thread materialisation still "
        "crashes. Stay on mx.default_stream(mx.default_device())."
    )
