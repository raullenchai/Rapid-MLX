"""Pure-Python coordination tests for continuous self-MTP delivery."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

MTP_DIR = Path(__file__).parents[1] / "vllm_mlx" / "spec_decode" / "mtp"
PACKAGE = "_continuous_mtp_driver_probe"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(MTP_DIR)]
sys.modules[PACKAGE] = package


def _load(name):
    spec = importlib.util.spec_from_file_location(
        f"{PACKAGE}.{name}", MTP_DIR / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


engine = _load("continuous_engine")
generation = _load("continuous_batch")
driver = _load("continuous_driver")


class _Compute:
    def __init__(self):
        self.calls = []
        self.queued_outputs = []

    def prepare(self, spec, forwards):
        del forwards
        self.calls.append(("prepare", spec.uid))
        token = 100 + spec.uid
        return engine.PreparedLaneData(
            cur=token,
            seed_hidden=f"hidden-{spec.uid}",
            token_prefix=(spec.uid,),
            caches=engine.SelfMTPCachePair(
                target=f"target-cache-{spec.uid}",
                draft=f"draft-cache-{spec.uid}",
            ),
            first_token=engine.MTPToken(token, f"lp-{token}", False),
        )

    def propose(self, lanes, caches, forwards):
        del caches, forwards
        self.calls.append(("propose", tuple(lane.uid for lane in lanes)))
        rows = self.queued_outputs.pop(0)
        accepted = tuple(
            sum(1 for token in row[:-1] if token.from_draft) for row in rows
        )
        return engine.CycleComputation(
            lane_uids=tuple(lane.uid for lane in lanes),
            draft_depths=accepted,
            accepted_lengths=accepted,
            target_drops=tuple(0 for _ in lanes),
            draft_drops=accepted,
            outputs=tuple(tuple(row) for row in rows),
        )

    def commit(self, lanes, computation, *, emitted_counts, terminal):
        self.calls.append(("commit", emitted_counts, terminal))
        for lane, outputs, count in zip(lanes, computation.outputs, emitted_counts):
            if count:
                lane.cur = outputs[count - 1].token

    def abort(self, lanes, caches, computation, cause):
        del lanes, caches, computation, cause

    def detach_lane(self, lane, caches):
        self.calls.append(("detach", lane.uid, caches.target, caches.draft))


class _Caches:
    def __init__(self):
        self.calls = []

    def attach(self, current, joining):
        self.calls.append(
            (
                "attach",
                None if current is None else tuple(current.target),
                tuple(pair.target for pair in joining),
            )
        )
        target = [] if current is None else list(current.target)
        draft = [] if current is None else list(current.draft)
        target.extend(pair.target for pair in joining)
        draft.extend(pair.draft for pair in joining)
        return engine.SelfMTPCachePair(target=target, draft=draft)

    def rollback(self, caches, *, target_drops, draft_drops, verify_width):
        del caches
        self.calls.append(
            ("rollback", tuple(target_drops), tuple(draft_drops), verify_width)
        )

    def detach(self, caches, indices, keep_indices):
        self.calls.append(("detach", tuple(indices), tuple(keep_indices)))
        detached = [
            engine.SelfMTPCachePair(
                target=caches.target[index], draft=caches.draft[index]
            )
            for index in indices
        ]
        remaining = engine.SelfMTPCachePair(
            target=[caches.target[index] for index in keep_indices],
            draft=[caches.draft[index] for index in keep_indices],
        )
        return remaining, detached


def _runtime(*, dynamic=True):
    compute = _Compute()
    caches = _Caches()
    runtime = engine.ContinuousSelfMTPRuntime(
        config=engine.ContinuousSelfMTPConfig(
            enabled=True,
            allow_dynamic_membership=dynamic,
            architecture="qwen3_5",
        ),
        capabilities=engine.ContinuousSelfMTPCapabilities(
            target_return_hidden=True,
            mtp_return_hidden=True,
            confirmed_target_forward=True,
            ragged_rollback=True,
            atomic_cache_commit=True,
            dynamic_membership=dynamic,
            flash_dynamic_membership_attested=dynamic,
        ),
        forwards=engine.RapidForwardSeams(
            lambda *args, **kwargs: None,
            lambda *args, **kwargs: None,
        ),
        compute=compute,
        caches=caches,
    )
    return runtime, compute, caches


def _spec(uid, *, max_tokens=8):
    return engine.SelfMTPLaneSpec(
        uid=uid,
        prompt=(uid,),
        max_tokens=max_tokens,
        num_draft=2,
    )


def _draft(token):
    return engine.MTPToken(token, f"lp-{token}", True)


def _target(token):
    return engine.MTPToken(token, f"lp-{token}", False)


def _triples(responses):
    return [
        (response.uid, response.token, response.finish_reason) for response in responses
    ]


def test_variable_burst_queues_and_drains_one_response_per_uid_before_next_step():
    runtime, compute, _caches = _runtime()
    batch_driver = driver.ContinuousMTPDriver.create([_spec(1), _spec(2)], runtime)

    assert _triples(batch_driver.next()) == [(1, 101, None), (2, 102, None)]
    compute.queued_outputs.extend(
        [
            [
                (_draft(111), _draft(112), _target(113)),
                (_draft(211), _target(212)),
            ],
            [(_target(114),), (_target(213),)],
        ]
    )

    first = batch_driver.next()
    second = batch_driver.next()
    third = batch_driver.next()

    assert _triples(first) == [(1, 111, None), (2, 211, None)]
    assert _triples(second) == [(1, 112, None), (2, 212, None)]
    assert _triples(third) == [(1, 113, None)]
    assert sum(call[0] == "propose" for call in compute.calls) == 1
    assert batch_driver.has_pending_responses is False

    assert _triples(batch_driver.next()) == [(1, 114, None), (2, 213, None)]
    assert sum(call[0] == "propose" for call in compute.calls) == 2


def test_join_waits_for_delivery_drain_then_emits_joined_initial_before_proposal():
    runtime, compute, caches = _runtime()
    batch_driver = driver.ContinuousMTPDriver.create([_spec(1)], runtime)
    batch_driver.next()
    compute.queued_outputs.append([(_draft(111), _target(112))])

    assert _triples(batch_driver.next()) == [(1, 111, None)]
    assert batch_driver.queue_lanes([_spec(2)]) == (2,)
    assert not any(call == ("prepare", 2) for call in compute.calls)

    # Lane 1 still has a committed response queued, so lane 2 cannot attach.
    assert _triples(batch_driver.next()) == [(1, 112, None)]
    assert batch_driver.last_attached_uids == ()
    assert batch_driver.pending_join_uids == (2,)

    joined_initial = batch_driver.next()
    assert _triples(joined_initial) == [(2, 102, None)]
    assert batch_driver.last_attached_uids == (2,)
    assert batch_driver.lane_uids == (1, 2)
    assert caches.calls[-1] == (
        "attach",
        ("target-cache-1",),
        ("target-cache-2",),
    )
    assert sum(call[0] == "propose" for call in compute.calls) == 1

    compute.queued_outputs.append([(_target(113),), (_target(213),)])
    assert [response.uid for response in batch_driver.next()] == [1, 2]
    assert ("propose", (1, 2)) in compute.calls


def test_terminal_detach_is_published_on_final_response_and_survivor_continues():
    runtime, compute, _caches = _runtime()
    batch_driver = driver.ContinuousMTPDriver.create(
        [_spec(1), _spec(2)], runtime, stop_tokens={1: {112}}
    )
    batch_driver.next()
    compute.queued_outputs.append([(_draft(111), _target(112)), (_target(211),)])

    first = batch_driver.next()
    assert _triples(first) == [(1, 111, None), (2, 211, None)]
    assert batch_driver.take_terminal_detaches() == ()
    assert batch_driver.lane_uids == (2,)

    terminal = batch_driver.next()
    assert _triples(terminal) == [(1, 112, "stop")]
    assert terminal[0].prompt_cache == "target-cache-1"
    assert terminal[0].all_tokens == [101, 111, 112]
    assert terminal[0].mtp_state == ("draft-cache-1", "hidden-1")
    packages = batch_driver.take_terminal_detaches()
    assert [package.uid for package in packages] == [1]
    assert batch_driver.take_resumable_detaches() == ()

    compute.queued_outputs.append([(_target(212),)])
    assert _triples(batch_driver.next()) == [(2, 212, None)]
    assert batch_driver.closed is False


def test_fixed_cohort_turnover_holds_survivor_until_its_burst_is_drained():
    runtime, compute, _caches = _runtime(dynamic=False)
    batch_driver = driver.ContinuousMTPDriver.create(
        [_spec(1, max_tokens=3), _spec(2)], runtime
    )
    batch_driver.next()
    compute.queued_outputs.append(
        [(_draft(111), _target(112)), (_draft(211), _target(212))]
    )

    first = batch_driver.next()
    assert _triples(first) == [(1, 111, None), (2, 211, None)]
    assert batch_driver.closed is True
    assert batch_driver.has_work is True
    assert batch_driver.take_resumable_detaches() == ()

    final = batch_driver.next()
    assert _triples(final) == [(1, 112, "length"), (2, 212, None)]
    terminal = batch_driver.take_terminal_detaches()
    assert [package.uid for package in terminal] == [1]
    assert batch_driver.resume_turnover() == (2,)
    assert batch_driver.closed is False
    assert batch_driver.take_resumable_detaches() == ()

    compute.queued_outputs.append([(_target(213),)])
    assert _triples(batch_driver.next()) == [(2, 213, None)]


def test_manual_shutdown_discards_queued_delivery_without_resumable_turnover():
    runtime, compute, _caches = _runtime()
    batch_driver = driver.ContinuousMTPDriver.create([_spec(1)], runtime)
    batch_driver.next()
    compute.queued_outputs.append([(_draft(111), _target(112))])
    batch_driver.next()

    packages = batch_driver.discard_all()
    assert [package.uid for package in packages] == [1]
    assert batch_driver.closed is True
    assert batch_driver.has_pending_responses is False
    assert batch_driver.next() == []
    assert batch_driver.take_resumable_detaches() == ()


def test_remove_uids_discards_cancelled_queue_and_keeps_dynamic_survivor():
    runtime, compute, _caches = _runtime()
    batch_driver = driver.ContinuousMTPDriver.create([_spec(1), _spec(2)], runtime)
    batch_driver.next()
    compute.queued_outputs.append(
        [(_draft(111), _target(112)), (_draft(211), _target(212))]
    )
    assert _triples(batch_driver.next()) == [(1, 111, None), (2, 211, None)]

    removed = batch_driver.remove_uids([1])
    assert [package.uid for package in removed] == [1]
    assert removed[0].terminal is False
    assert removed[0].target_cache == "target-cache-1"
    assert batch_driver.lane_uids == (2,)

    # Lane 1's remaining committed token is discarded; the survivor drains its
    # own queued token before another proposal can run.
    assert _triples(batch_driver.next()) == [(2, 212, None)]
    assert sum(call[0] == "propose" for call in compute.calls) == 1
    compute.queued_outputs.append([(_target(213),)])
    assert _triples(batch_driver.next()) == [(2, 213, None)]


def test_fixed_remove_turns_over_companion_without_transferring_its_ownership():
    runtime, compute, _caches = _runtime(dynamic=False)
    batch_driver = driver.ContinuousMTPDriver.create([_spec(1), _spec(2)], runtime)
    batch_driver.next()

    removed = batch_driver.remove_uids([1])
    assert [package.uid for package in removed] == [1]
    assert batch_driver.closed is True
    assert batch_driver.resume_turnover() == (2,)
    assert batch_driver.take_resumable_detaches() == ()

    compute.queued_outputs.append([(_target(211),)])
    assert _triples(batch_driver.next()) == [(2, 211, None)]
    assert not any(call == ("prepare", 2) for call in compute.calls[2:])


def test_driver_construction_resume_and_inspection_contracts():
    runtime, _compute, _caches = _runtime()
    with pytest.raises(TypeError, match="batch"):
        driver.ContinuousMTPDriver(object())
    batch = generation.ContinuousMTPGenerationBatch.create([_spec(1)], runtime)
    with pytest.raises(TypeError, match="response_factory"):
        driver.ContinuousMTPDriver(batch, response_factory=object())

    batch_driver = driver.ContinuousMTPDriver(batch)
    assert batch_driver.batch is batch
    assert batch_driver.last_burst is None
    assert batch_driver.has_work is True
    batch_driver.next()
    package = batch_driver.detach_all()[0]
    assert batch_driver.lane_uids == ()
    resumed = driver.ContinuousMTPDriver.resume([package])
    assert resumed.lane_uids == (1,)


def test_queue_validation_and_closed_driver_guards():
    runtime, _compute, _caches = _runtime()
    batch_driver = driver.ContinuousMTPDriver.create([_spec(1)], runtime)
    assert batch_driver.queue_lanes([]) == ()
    with pytest.raises(ValueError, match="unique"):
        batch_driver.queue_lanes([_spec(2), _spec(2)])
    batch_driver.queue_lanes([_spec(2)])
    with pytest.raises(ValueError, match="occupied"):
        batch_driver.queue_lanes([_spec(2)])
    with pytest.raises(ValueError, match="unknown queued"):
        batch_driver.queue_lanes([_spec(3)], stop_tokens={4: {400}})

    batch_driver.next()
    batch_driver.detach_all()
    with pytest.raises(driver.ContinuousMTPDriverError, match="detached"):
        batch_driver.queue_lanes([_spec(3)])

    fixed_runtime, _compute, _caches = _runtime(dynamic=False)
    fixed = driver.ContinuousMTPDriver.create([_spec(1)], fixed_runtime)
    with pytest.raises(engine.ContinuousSelfMTPUnsupportedError, match="dynamic"):
        fixed.queue_lanes([_spec(2)])


def test_driver_invariant_guards_for_delivery_and_detach():
    runtime, compute, _caches = _runtime()
    batch_driver = driver.ContinuousMTPDriver.create([_spec(1)], runtime)
    batch_driver.next()
    compute.queued_outputs.append([(_draft(111), _target(112))])
    batch_driver.next()

    with pytest.raises(driver.ContinuousMTPDriverError, match="before delivery drains"):
        batch_driver.detach_all()
    with pytest.raises(driver.ContinuousMTPDriverError, match="before delivery drains"):
        batch_driver._queue_burst(batch_driver.last_burst)

    empty = generation.ContinuousMTPGenerationBurst(
        emissions=(generation.ContinuousMTPLaneEmission(uid=9, tokens=()),),
        emitted_counts=(0,),
        initial=False,
    )
    other = driver.ContinuousMTPDriver.create([_spec(2)], runtime)
    other.next()
    other.detach_all()
    with pytest.raises(driver.ContinuousMTPDriverError, match="empty burst"):
        other._queue_burst(empty)

    orphan = driver.ContinuousMTPDriver.create([_spec(3)], runtime)
    orphan._delivery_order = (3,)
    orphan._delivery_queues[3] = driver.deque(
        [driver._QueuedToken(3, 300, None, False, "stop")]
    )
    with pytest.raises(driver.ContinuousMTPDriverError, match="no retained"):
        orphan._drain_one_per_uid()


def test_remove_covers_staged_held_and_empty_paths():
    runtime, compute, _caches = _runtime()
    batch_driver = driver.ContinuousMTPDriver.create([_spec(1)], runtime)
    assert batch_driver.remove_uids([]) == ()
    batch_driver.queue_lanes([_spec(2), _spec(3)], stop_tokens={2: {202}})
    batch_driver.remove_uids([2])
    assert batch_driver.pending_join_uids == (3,)
    batch_driver.remove_uids([3])

    batch_driver.next()
    compute.queued_outputs.append([(_draft(111), _target(112))])
    batch_driver.next()
    package = batch_driver._batch.detach_lanes([1])[0]
    batch_driver._held_detaches[1] = package
    removed = batch_driver.remove_uids([1])
    assert removed == (package,)


def test_turnover_validation_empty_and_deduplicated_recording():
    runtime, compute, _caches = _runtime(dynamic=False)
    batch_driver = driver.ContinuousMTPDriver.create([_spec(1), _spec(2)], runtime)
    with pytest.raises(driver.ContinuousMTPDriverError, match="detached cohort"):
        batch_driver.resume_turnover()

    batch_driver.next()
    compute.queued_outputs.append(
        [(_draft(111), _target(112)), (_draft(211), _target(212))]
    )
    batch_driver.next()
    batch_driver.remove_uids([1])
    with pytest.raises(driver.ContinuousMTPDriverError, match="before delivery drains"):
        batch_driver.resume_turnover()
    batch_driver.next()
    packages = batch_driver.take_resumable_detaches()
    assert [package.uid for package in packages] == [2]
    assert batch_driver.resume_turnover() == ()

    package = packages[0]
    batch_driver._record_detaches((package, package))
    assert batch_driver.take_resumable_detaches() == (package,)
