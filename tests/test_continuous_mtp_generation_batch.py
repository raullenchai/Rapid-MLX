"""Pure-Python contract tests for the continuous MTP generation wrapper."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

MTP_DIR = Path(__file__).parents[1] / "vllm_mlx" / "spec_decode" / "mtp"
PACKAGE = "_continuous_mtp_generation_batch_probe"
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
            payload=f"cycle-{len(self.calls)}",
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
            ("attach", None if current is None else "live", tuple(joining))
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


def _runtime(*, dynamic=False, flash=False, flash_attested=None):
    """Build a fake runtime.

    ``dynamic`` attests incremental membership (``allow_dynamic_membership`` and
    the ``dynamic_membership`` capability).  ``flash`` selects a Flash-family
    architecture, whose join additionally needs ``flash_attested`` (defaults to
    ``dynamic``) so the Flash-specific refusal can be exercised in isolation.
    """
    if flash_attested is None:
        flash_attested = dynamic
    compute = _Compute()
    caches = _Caches()
    capabilities = engine.ContinuousSelfMTPCapabilities(
        target_return_hidden=True,
        mtp_return_hidden=True,
        confirmed_target_forward=True,
        ragged_rollback=True,
        atomic_cache_commit=True,
        dynamic_membership=dynamic,
        flash_dynamic_membership_attested=flash_attested,
    )
    runtime = engine.ContinuousSelfMTPRuntime(
        config=engine.ContinuousSelfMTPConfig(
            enabled=True,
            allow_dynamic_membership=dynamic,
            architecture="qwen4_flash_next" if flash else "qwen3_5",
        ),
        capabilities=capabilities,
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


def test_initial_tokens_are_emitted_once_and_terminal_detaches_whole_cohort():
    runtime, compute, caches = _runtime()
    batch = generation.ContinuousMTPGenerationBatch.create(
        [_spec(1, max_tokens=1), _spec(2)], runtime
    )

    burst = batch.next_burst()

    assert burst.initial is True
    assert burst.emitted_counts == (1, 1)
    assert [emission.token_ids for emission in burst.emissions] == [(101,), (102,)]
    assert burst.emissions[0].finish_reason == "length"
    assert burst.emissions[1].finish_reason is None
    assert [package.uid for package in burst.terminal_detaches] == [1]
    assert [package.uid for package in burst.resumable_detaches] == [2]
    assert burst.resumable_detaches[0].token_ids == (102,)
    assert batch.closed is True
    assert not any(call[0] == "propose" for call in compute.calls)
    assert caches.calls[-1] == ("detach", (0, 1), ())


def test_max_tokens_one_closes_on_initial_without_forming_proposal() -> None:
    runtime, compute, _caches = _runtime()
    batch = generation.ContinuousMTPGenerationBatch.create(
        [_spec(1, max_tokens=1), _spec(2, max_tokens=1)], runtime
    )

    burst = batch.next_burst()

    assert burst.initial is True
    assert all(emission.finish_reason == "length" for emission in burst.emissions)
    assert batch.closed is True
    assert not any(call[0] == "propose" for call in compute.calls)


def test_max_tokens_two_commits_target_only_b1_and_detaches() -> None:
    runtime, compute, _caches = _runtime()
    compute.queued_outputs.append([(_target(111),)])
    batch = generation.ContinuousMTPGenerationBatch.create(
        [_spec(1, max_tokens=2)], runtime
    )
    batch.next_burst()

    burst = batch.next_burst()

    assert burst.emitted_counts == (1,)
    assert burst.emissions[0].token_ids == (111,)
    assert burst.emissions[0].finish_reason == "length"
    assert burst.terminal_detaches[0].token_ids == (101, 111)
    assert batch.closed is True


def test_one_proposal_burst_commits_exact_stop_prefix_then_extracts_caches():
    runtime, compute, _caches = _runtime()
    compute.queued_outputs.append([(_draft(111), _target(112)), (_target(212),)])
    batch = generation.ContinuousMTPGenerationBatch.create(
        [_spec(1), _spec(2)], runtime, stop_tokens={1: {111}}
    )
    batch.next_burst()  # prepared first-token cohort

    burst = batch.next_burst()

    assert burst.initial is False
    assert burst.emitted_counts == (1, 1)
    assert [emission.token_ids for emission in burst.emissions] == [(111,), (212,)]
    assert ("commit", (1, 1), (True, False)) in compute.calls
    terminal = burst.terminal_detaches[0]
    assert terminal.uid == 1
    assert terminal.finish_reason == "stop"
    assert terminal.token_ids == (101, 111)
    assert terminal.target_cache == "target-cache-1"
    assert terminal.draft_cache == "draft-cache-1"
    companion = burst.resumable_detaches[0]
    assert companion.uid == 2
    assert companion.token_ids == (102, 212)
    assert companion.terminal is False


def test_max_token_boundary_marks_the_full_final_proposal_as_length():
    runtime, compute, _caches = _runtime()
    compute.queued_outputs.append([(_draft(111), _target(112)), (_target(212),)])
    batch = generation.ContinuousMTPGenerationBatch.create(
        [_spec(1, max_tokens=3), _spec(2)], runtime
    )
    batch.next_burst()

    burst = batch.next_burst()

    assert burst.emitted_counts == (2, 1)
    assert burst.emissions[0].token_ids == (111, 112)
    assert burst.emissions[0].finish_reason == "length"
    assert ("commit", (2, 1), (True, False)) in compute.calls
    assert burst.terminal_detaches[0].token_ids == (101, 111, 112)


def test_one_proposal_per_call_and_manual_detach_is_idempotent():
    runtime, compute, _caches = _runtime()
    compute.queued_outputs.extend(
        [
            [(_target(111),), (_target(211),)],
            [(_target(112),), (_target(212),)],
        ]
    )
    batch = generation.ContinuousMTPGenerationBatch.create(
        [_spec(1), _spec(2)], runtime
    )

    initial = batch.next_burst()
    first = batch.next_burst()
    second = batch.next_burst()

    assert initial.initial is True
    assert first.initial is second.initial is False
    assert sum(call[0] == "propose" for call in compute.calls) == 2
    assert [state.emitted_tokens for state in batch.lane_states] == [3, 3]
    detached = batch.detach_all()
    assert [package.token_ids for package in detached] == [
        (101, 111, 112),
        (102, 211, 212),
    ]
    assert all(not package.terminal for package in detached)
    assert batch.detach_all() is detached
    with pytest.raises(
        generation.ContinuousMTPGenerationBatchError, match="already detached"
    ):
        batch.next_burst()


def test_failed_commit_does_not_publish_delivery_ledger():
    runtime, compute, _caches = _runtime()
    compute.queued_outputs.append([(_draft(111), _target(112))])
    batch = generation.ContinuousMTPGenerationBatch.create([_spec(1)], runtime)
    batch.next_burst()

    def fail_commit(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("commit failed")

    compute.commit = fail_commit
    with pytest.raises(RuntimeError, match="commit failed"):
        batch.next_burst()

    assert batch.lane_states[0].emitted_tokens == 1
    assert batch.closed is False


def test_fixed_cohort_wrapper_refuses_incremental_join():
    # Core capabilities alone (no dynamic attestation) never unlock a join.
    runtime, _compute, _caches = _runtime()
    batch = generation.ContinuousMTPGenerationBatch.create([_spec(1)], runtime)
    assert batch.dynamic_membership is False

    with pytest.raises(
        engine.ContinuousSelfMTPUnsupportedError,
        match="fixed-cohort.*including Flash",
    ):
        batch.attach_lanes([_spec(2)])


def test_flash_join_refused_without_flash_attestation():
    # Dynamic + core attested, but a Flash architecture still needs its own
    # attestation; the wrapper reports non-dynamic and the engine fails closed.
    runtime, _compute, _caches = _runtime(
        dynamic=True, flash=True, flash_attested=False
    )
    batch = generation.ContinuousMTPGenerationBatch.create([_spec(1)], runtime)
    assert batch.dynamic_membership is False

    with pytest.raises(engine.ContinuousSelfMTPUnsupportedError, match="fixed-cohort"):
        batch.attach_lanes([_spec(2)])


def test_dynamic_join_delivers_first_token_then_participates_in_a_proposal():
    runtime, compute, caches = _runtime(dynamic=True)
    batch = generation.ContinuousMTPGenerationBatch.create([_spec(1)], runtime)
    assert batch.dynamic_membership is True
    batch.next_burst()  # lane 1 delivers its prepared first token

    attached = batch.attach_lanes([_spec(2)])
    assert attached == (2,)
    assert batch.lane_uids == (1, 2)
    # The merge went into a live batch, not a fresh one.
    assert any(call[0] == "attach" and call[1] == "live" for call in caches.calls)

    # The next burst delivers only the joined lane's prepared first token.
    join_initial = batch.next_burst()
    assert join_initial.initial is True
    assert [emission.uid for emission in join_initial.emissions] == [2]
    assert join_initial.emissions[0].token_ids == (102,)
    assert not join_initial.detached
    assert not any(call[0] == "propose" for call in compute.calls)

    # Only now does a proposal run, over both lanes together.
    compute.queued_outputs.append([(_target(111),), (_target(211),)])
    proposal_burst = batch.next_burst()
    assert proposal_burst.initial is False
    assert ("propose", (1, 2)) in compute.calls
    assert {emission.uid for emission in proposal_burst.emissions} == {1, 2}
    assert batch.closed is False


def test_dynamic_terminal_lane_detaches_alone_while_companions_continue():
    runtime, compute, _caches = _runtime(dynamic=True)
    batch = generation.ContinuousMTPGenerationBatch.create(
        [_spec(1), _spec(2)], runtime, stop_tokens={1: {111}}
    )
    batch.next_burst()  # both deliver their prepared first tokens

    compute.queued_outputs.append([(_target(111),), (_target(211),)])
    burst = batch.next_burst()

    # Lane 1 hit its stop token and detaches by itself; lane 2 keeps decoding.
    assert [package.uid for package in burst.terminal_detaches] == [1]
    assert burst.resumable_detaches == ()
    assert burst.terminal_detaches[0].finish_reason == "stop"
    assert batch.closed is False
    assert batch.lane_uids == (2,)

    # The survivor continues to produce tokens on its own.
    compute.queued_outputs.append([(_target(212),)])
    survivor = batch.next_burst()
    assert [emission.uid for emission in survivor.emissions] == [2]
    assert survivor.emissions[0].token_ids == (212,)
    assert batch.closed is False


def test_dynamic_join_rejects_a_uid_already_live():
    runtime, _compute, _caches = _runtime(dynamic=True)
    batch = generation.ContinuousMTPGenerationBatch.create([_spec(1)], runtime)
    batch.next_burst()
    with pytest.raises(ValueError, match="reuse live uid"):
        batch.attach_lanes([_spec(1)])


def test_stop_token_configuration_fails_closed_on_unknown_lane_or_bad_token():
    runtime, _compute, _caches = _runtime()
    with pytest.raises(ValueError, match="unknown lane uid"):
        generation.ContinuousMTPGenerationBatch.create(
            [_spec(1)], runtime, stop_tokens={2: {200}}
        )
    with pytest.raises(ValueError, match="must be integers"):
        generation.ContinuousMTPGenerationBatch.create(
            [_spec(1)], runtime, stop_tokens={1: {True}}
        )


def test_wrapper_value_properties_and_constructor_guards():
    runtime, _compute, _caches = _runtime()
    detached, first = engine.prepare_self_mtp_lane(_spec(1), runtime)
    core = engine.attach_self_mtp_lanes(None, [detached], runtime=runtime)
    with pytest.raises(ValueError, match="one entry"):
        generation.ContinuousMTPGenerationBatch(
            batch=core,
            first_tokens=[],
            stop_tokens={1: frozenset()},
        )

    wrapper = generation.ContinuousMTPGenerationBatch(
        batch=core,
        first_tokens=[first],
        stop_tokens={1: frozenset()},
    )
    assert wrapper.initial_pending is True
    burst = wrapper.next_burst()
    emission = burst.emissions[0]
    assert emission.logprobs == ("lp-101",)
    assert burst.cohort_detached is False
    package = wrapper.detach_all()[0]
    assert package.uid == package.lane.uid == 1
    assert package.caches is package.detached.caches
    assert package.logprobs == ("lp-101",)


def test_create_and_resume_reject_invalid_cohorts():
    runtime, _compute, _caches = _runtime()
    with pytest.raises(ValueError, match="empty"):
        generation.ContinuousMTPGenerationBatch.create([], runtime)
    with pytest.raises(ValueError, match="unique"):
        generation.ContinuousMTPGenerationBatch.create([_spec(1), _spec(1)], runtime)
    with pytest.raises(ValueError, match="empty"):
        generation.ContinuousMTPGenerationBatch.resume([])

    wrapper = generation.ContinuousMTPGenerationBatch.create([_spec(1)], runtime)
    wrapper.next_burst()
    resumable = wrapper.detach_all()[0]
    terminal = generation.ContinuousMTPDetachPackage(
        detached=resumable.detached,
        tokens=resumable.tokens,
        stop_tokens=resumable.stop_tokens,
        terminal=True,
        finish_reason="stop",
    )
    with pytest.raises(ValueError, match="terminal"):
        generation.ContinuousMTPGenerationBatch.resume([terminal])
    with pytest.raises(ValueError, match="unique"):
        generation.ContinuousMTPGenerationBatch.resume([resumable, resumable])

    other_runtime, _compute, _caches = _runtime()
    other = generation.ContinuousMTPGenerationBatch.create([_spec(2)], other_runtime)
    other.next_burst()
    other_package = other.detach_all()[0]
    with pytest.raises(ValueError, match="different runtimes"):
        generation.ContinuousMTPGenerationBatch.resume([resumable, other_package])

    resumed = generation.ContinuousMTPGenerationBatch.resume([resumable])
    assert resumed.initial_pending is False
    assert resumed.lane_uids == (1,)


def test_fixed_membership_explicit_detach_turns_over_the_whole_cohort():
    runtime, _compute, _caches = _runtime(dynamic=False)
    batch = generation.ContinuousMTPGenerationBatch.create(
        [_spec(1), _spec(2)], runtime
    )
    batch.next_burst()

    detached = batch.detach_lanes([1])

    assert [package.uid for package in detached] == [1, 2]
    assert batch.closed is True


def test_dynamic_join_and_detach_validate_empty_duplicate_and_unknown_uids():
    runtime, _compute, _caches = _runtime(dynamic=True)
    wrapper = generation.ContinuousMTPGenerationBatch.create(
        [_spec(1), _spec(2)], runtime
    )
    wrapper.next_burst()
    assert wrapper.attach_lanes([]) == ()
    with pytest.raises(ValueError, match="unique"):
        wrapper.attach_lanes([_spec(3), _spec(3)])
    assert wrapper.detach_lanes([]) == ()
    with pytest.raises(ValueError, match="unique"):
        wrapper.detach_lanes([1, 1])
    with pytest.raises(KeyError, match="unknown"):
        wrapper.detach_lanes([9])
    detached = wrapper.detach_lanes([1])
    assert detached[0].uid == 1
    assert detached[0].terminal is False
    assert wrapper.lane_uids == (2,)


def test_initial_stop_and_corrupt_initial_state_paths():
    runtime, _compute, _caches = _runtime(dynamic=True)
    wrapper = generation.ContinuousMTPGenerationBatch.create(
        [_spec(1), _spec(2)],
        runtime,
        stop_tokens={1: {101}},
    )
    burst = wrapper.next_burst()
    assert burst.emissions[0].finish_reason == "stop"
    assert burst.cohort_detached is True

    wrapper = generation.ContinuousMTPGenerationBatch.create([_spec(3)], runtime)
    wrapper._states[3].pending_initial = None
    with pytest.raises(
        generation.ContinuousMTPGenerationBatchError, match="no pending"
    ):
        wrapper._deliver_initial([3])


def test_closed_cohort_and_exhausted_prefix_defensive_paths():
    runtime, _compute, _caches = _runtime()
    wrapper = generation.ContinuousMTPGenerationBatch.create([_spec(1)], runtime)
    wrapper.next_burst()
    detached = wrapper.detach_all()
    assert wrapper._detach_cohort() is detached

    state = generation._LaneDeliveryState(
        uid=1,
        max_tokens=1,
        stop_tokens=frozenset(),
        tokens=[_target(1)],
    )
    with pytest.raises(
        generation.ContinuousMTPGenerationBatchError, match="exhausting"
    ):
        generation._bounded_prefix(state, [_target(2)])
