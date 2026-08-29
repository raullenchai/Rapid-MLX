# SPDX-License-Identifier: Apache-2.0
"""Tests for EngineCore._step_coalesced (#1861 conc investigation).

Deep batches run up to 4 scheduler steps per executor dispatch; the
helper must break early on finishes / no-work / pending admissions and
must deliver partial outputs when a later step raises (a step that
already advanced scheduler state has produced tokens the collectors
must still receive — codex r1 on #1878).
"""

from __future__ import annotations

import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx


from types import SimpleNamespace

from vllm_mlx.engine_core import EngineCore


def _out(finished=(), has_work=True):
    return SimpleNamespace(
        finished_request_ids=list(finished), has_work=has_work, outputs=[]
    )


class _ScriptedScheduler:
    """Yields scripted step results; an Exception instance raises.

    ``consume_waiting`` mirrors the real scheduler: step() ADMITS the
    waiting queue into the batch, so the count reads 0 afterwards.
    """

    def __init__(self, script, waiting=0, consume_waiting=False):
        self.script = list(script)
        self.calls = 0
        self._waiting = waiting
        self._consume = consume_waiting

    def step(self):
        self.calls += 1
        item = self.script.pop(0)
        if self._consume:
            self._waiting = 0
        if isinstance(item, Exception):
            raise item
        return item

    def get_num_waiting(self):
        return self._waiting


def _engine(scheduler):
    eng = EngineCore.__new__(EngineCore)
    eng.scheduler = scheduler
    return eng


def test_runs_up_to_max_steps():
    sched = _ScriptedScheduler([_out(), _out(), _out(), _out()])
    outs, err = _engine(sched)._step_coalesced(4)
    assert err is None
    assert len(outs) == 4
    assert sched.calls == 4


def test_breaks_on_finish():
    sched = _ScriptedScheduler([_out(), _out(finished=["r1"]), _out()])
    outs, err = _engine(sched)._step_coalesced(4)
    assert err is None
    assert len(outs) == 2  # stops the step after a finish
    assert sched.calls == 2


def test_breaks_on_no_work():
    sched = _ScriptedScheduler([_out(has_work=False), _out()])
    outs, err = _engine(sched)._step_coalesced(4)
    assert err is None
    assert len(outs) == 1


def test_breaks_on_pending_admissions():
    sched = _ScriptedScheduler([_out(), _out()], waiting=1)
    outs, err = _engine(sched)._step_coalesced(4)
    assert err is None
    assert len(outs) == 1  # work was already waiting at dispatch time


def test_breaks_when_step_consumes_waiting_queue():
    """codex r3 BLOCKING: the real scheduler ADMITS the waiting queue
    inside step(), so a post-step check reads 0 for exactly the request
    whose first output must not sit out the rest of a coalesced batch.
    The pre-step snapshot must stop coalescing after that step."""
    sched = _ScriptedScheduler(
        [_out(), _out(), _out()], waiting=1, consume_waiting=True
    )
    outs, err = _engine(sched)._step_coalesced(4)
    assert err is None
    assert len(outs) == 1
    assert sched.calls == 1


def test_coalesce_budget_caps_at_memory_boundary():
    """codex r4 BLOCKING: a dispatch must not cross the memory-check
    boundary, or a boundary crossed on the first step tolerates up to
    3 extra Metal allocations before the pressure check fires."""
    eng = EngineCore.__new__(EngineCore)
    eng._steps_executed = 0
    assert eng._coalesce_budget(8, 16) == 4  # fresh window, full depth
    eng._steps_executed = 14
    assert eng._coalesce_budget(8, 16) == 2  # 2 steps to boundary
    eng._steps_executed = 15
    assert eng._coalesce_budget(8, 16) == 1  # boundary next step
    eng._steps_executed = 16
    assert eng._coalesce_budget(8, 16) == 4  # window rolled over
    eng._steps_executed = 0
    assert eng._coalesce_budget(4, 16) == 2  # depth scales with active


def test_partial_outputs_preserved_on_error():
    """codex r1 BLOCKING: a later step raising must not discard outputs
    from earlier steps that already advanced scheduler state — their
    tokens are produced and their finish events must still fire."""
    boom = RuntimeError("step 3 exploded")
    sched = _ScriptedScheduler([_out(), _out(), boom])
    outs, err = _engine(sched)._step_coalesced(4)
    assert err is boom
    assert len(outs) == 2  # both successful steps' outputs survive
