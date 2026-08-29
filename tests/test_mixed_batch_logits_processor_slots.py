# SPDX-License-Identifier: Apache-2.0
"""A mixed batch must not put ``None`` in a per-sequence processor slot.

#1525. Plain chat concurrent with a tool call — which is what every agent
we support does — could kill the engine loop:

    ERROR:rapid_mlx.engine_core:Engine loop error: 'NoneType' object is not iterable
      mlx_lm/generate.py:1809  gen_batch = self._prompt_batch.split(split).generate(...)
      mlx_lm/generate.py:1346  for processor in self.logits_processors[e]:
    TypeError: 'NoneType' object is not iterable
    INFO: "POST /v1/chat/completions HTTP/1.1" 503 Service Unavailable

``PromptProcessingBatch.extend`` normalizes "no processors" to ``None``
slots, guarded by ``any()`` — a whole-list question asked about a
per-sequence list. All-empty and all-present are both safe; only the MIXED
batch produces ``None``, and only after a split/merge reshuffles it, which
is why it reproduced in 3 of 5 consecutive stress runs rather than every
time.

These tests drive the real mlx-lm classes rather than a stand-in: the
defect is in upstream's bookkeeping, so a fake would only test the fake.
No model and no weights are involved — the batch objects are built with
``__new__`` and the lists that matter, exactly as the crash path leaves
them.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

pytestmark = pytest.mark.requires_mlx

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# BEFORE importing mlx_lm.generate, not after. That module captures
# ``mx.new_thread_local_stream(...)`` at import time, and ``install()`` is what
# makes that call safe on single-stream GPUs (#404) — the ordering invariant
# both ``_mlx_compat`` and ``scheduler`` document. Importing mlx-lm first here
# would pin the unusable stream for the rest of the pytest process, so a later
# scheduler import installs the shim too late and inference fails with
# "There is no Stream(gpu, 1) in current thread".
from vllm_mlx import _mlx_compat  # noqa: E402

_mlx_compat.install()

pytest.importorskip("mlx_lm.generate", reason="requires MLX")

from mlx_lm.generate import PromptProcessingBatch  # noqa: E402


def _noop_processor(token_context, logits):
    return logits


def _batch(uids, processors):
    """A PromptProcessingBatch carrying only the state ``extend`` touches."""
    b = PromptProcessingBatch.__new__(PromptProcessingBatch)
    b.uids = list(uids)
    b.samplers = [None] * len(uids)
    b.logits_processors = list(processors)
    b.prompt_cache = []
    b.tokens = [[] for _ in uids]
    b.max_tokens = [64] * len(uids)
    b.state_machines = [None] * len(uids)
    return b


@pytest.fixture(autouse=True)
def _guard_installed():
    _mlx_compat.install_batch_slot_guard()


def _step_would_crash(batch) -> bool:
    """Replay what ``GenerationBatch._step`` does with these slots."""
    if not any(batch.logits_processors):
        return False  # fast path: the loop is skipped entirely
    for e in range(len(batch.uids)):
        try:
            for _ in batch.logits_processors[e]:
                pass
        except TypeError:
            return True
    return False


def test_mixed_batch_leaves_no_none_slot():
    """Two plain requests joined by one that carries a processor."""
    plain = _batch([1, 2], [[], []])
    grammar = _batch([3], [[_noop_processor]])

    plain.extend(grammar)

    assert None not in plain.logits_processors, plain.logits_processors
    assert not _step_would_crash(plain)


def test_mixed_batch_the_other_way_round():
    """...and with the processor-carrying batch as the receiver."""
    grammar = _batch([1], [[_noop_processor]])
    plain = _batch([2, 3], [[], []])

    grammar.extend(plain)

    assert None not in grammar.logits_processors, grammar.logits_processors
    assert not _step_would_crash(grammar)


def test_the_processor_itself_survives_the_merge():
    """Normalizing empty slots must not drop a real processor."""
    plain = _batch([1], [[]])
    grammar = _batch([2], [[_noop_processor]])

    plain.extend(grammar)

    kept = [s for s in plain.logits_processors if s]
    assert kept == [[_noop_processor]], plain.logits_processors


def test_slots_stay_aligned_with_uids():
    plain = _batch([1, 2], [[], []])
    grammar = _batch([3], [[_noop_processor]])

    plain.extend(grammar)

    assert len(plain.logits_processors) == len(plain.uids)
    assert plain.logits_processors[2] == [_noop_processor]


def test_none_slots_do_not_survive_a_filter():
    """``filter`` keeps slots verbatim once ``any()`` is True."""
    plain = _batch([1, 2], [[], []])
    grammar = _batch([3], [[_noop_processor]])
    plain.extend(grammar)

    plain.prompt_cache = []
    plain.filter([0, 1, 2])

    assert None not in plain.logits_processors, plain.logits_processors
    assert not _step_would_crash(plain)


def test_all_empty_merge_keeps_the_fast_path():
    """Normalization must not make ``any()`` newly True.

    ``_step`` skips the entire processing loop when no sequence has a
    processor. Filling slots with anything truthy would run a per-token
    Python loop over every sequence for no reason.
    """
    a = _batch([1], [[]])
    b = _batch([2], [[]])

    a.extend(b)

    assert not any(a.logits_processors), a.logits_processors


def test_all_present_merge_is_unchanged():
    a = _batch([1], [[_noop_processor]])
    b = _batch([2], [[_noop_processor]])

    a.extend(b)

    assert a.logits_processors == [[_noop_processor], [_noop_processor]]


def test_guard_is_idempotent():
    """Installed twice must not wrap twice."""
    _mlx_compat.install_batch_slot_guard()
    once = PromptProcessingBatch.extend
    _mlx_compat.install_batch_slot_guard()
    assert PromptProcessingBatch.extend is once


def test_scheduler_import_installs_the_guard():
    """The guard has to be live for anyone who imports the scheduler.

    A shim nobody calls is a shim that does not exist; this is the wire
    between the fix and the crash path.

    In a SUBPROCESS, and deliberately outside the autouse fixture's reach.
    Asserting the flag in-process proves nothing: the fixture installs the
    guard before every test in this file, so the assertion would hold with
    the ``scheduler.py`` call site deleted (raised in review). The only
    honest question is whether importing the scheduler — and nothing else —
    arms it.
    """
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import importlib;"
            "m = importlib.import_module('mlx_lm.generate');"
            "assert not getattr(m.PromptProcessingBatch, '_rapid_mlx_slot_guard', False),"
            " 'guard armed before importing the scheduler — test is vacuous';"
            "importlib.import_module('vllm_mlx.scheduler');"
            "assert getattr(m.PromptProcessingBatch, '_rapid_mlx_slot_guard', False),"
            " 'importing vllm_mlx.scheduler did not install the guard';"
            "print('WIRED')",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=300,
    )
    # Both, not just the marker: a process can print WIRED and then die on
    # the way out, and a test that only greps stdout would call that a pass
    # (raised in review).
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "WIRED" in proc.stdout, proc.stderr + proc.stdout
