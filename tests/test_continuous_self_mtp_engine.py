"""Mock-only tests for the fixed-membership continuous self-MTP engine."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).parents[1]
    / "vllm_mlx"
    / "spec_decode"
    / "mtp"
    / "continuous_engine.py"
)
SPEC = importlib.util.spec_from_file_location("continuous_engine_probe", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
engine = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = engine
SPEC.loader.exec_module(engine)


def _capabilities(**overrides):
    values = {
        "target_return_hidden": True,
        "mtp_return_hidden": True,
        "confirmed_target_forward": True,
        "ragged_rollback": True,
        "atomic_cache_commit": True,
    }
    values.update(overrides)
    return engine.ContinuousSelfMTPCapabilities(**values)


class _Compute:
    def __init__(self):
        self.calls = []

    def prepare(self, spec, forwards):
        target = forwards.target(spec.prompt, f"target-{spec.uid}", n_confirmed=0)
        draft = forwards.draft(target[1], spec.prompt, f"draft-{spec.uid}")
        self.calls.append(("prepare", spec.uid, target, draft))
        token = 100 + spec.uid
        return engine.PreparedLaneData(
            cur=token,
            seed_hidden=f"hidden-{spec.uid}",
            token_prefix=(spec.uid,),
            caches=engine.SelfMTPCachePair(
                target=[f"target-{spec.uid}"], draft=[f"draft-{spec.uid}"]
            ),
            first_token=engine.MTPToken(token, f"lp-{token}", False),
            backend_state={"uid": spec.uid},
        )

    def propose(self, lanes, caches, forwards):
        del caches, forwards
        self.calls.append(("propose", tuple(lane.uid for lane in lanes)))
        outputs = []
        accepted = []
        for row, lane in enumerate(lanes):
            n_accept = 1 if row == 0 else 0
            accepted.append(n_accept)
            row_outputs = [engine.MTPToken(lane.cur + 1, f"lp-draft-{lane.uid}", True)][
                :n_accept
            ]
            row_outputs.append(
                engine.MTPToken(lane.cur + 10, f"lp-target-{lane.uid}", False)
            )
            outputs.append(tuple(row_outputs))
        depths = tuple(2 for _ in lanes)
        accepted_tuple = tuple(accepted)
        return engine.CycleComputation(
            lane_uids=tuple(lane.uid for lane in lanes),
            draft_depths=depths,
            accepted_lengths=accepted_tuple,
            target_drops=tuple(2 - count for count in accepted_tuple),
            draft_drops=depths,
            outputs=tuple(outputs),
            payload="opaque-cycle",
        )

    def commit(self, lanes, computation, *, emitted_counts, terminal):
        self.calls.append(("commit", computation.payload, emitted_counts, terminal))
        for lane, outputs in zip(lanes, computation.outputs):
            if outputs:
                lane.cur = outputs[-1].token

    def abort(self, lanes, caches, computation, cause):
        del caches
        self.calls.append(
            (
                "abort",
                tuple(lane.uid for lane in lanes),
                None if computation is None else computation.payload,
                None if cause is None else str(cause),
            )
        )

    def detach_lane(self, lane, caches):
        self.calls.append(("detach", lane.uid, caches.target, caches.draft))


class _Caches:
    def __init__(self):
        self.calls = []

    def attach(self, current, joining):
        self.calls.append(("attach", current, tuple(joining)))
        target = [] if current is None else list(current.target)
        draft = [] if current is None else list(current.draft)
        for item in joining:
            target.extend(item.target)
            draft.extend(item.draft)
        return engine.SelfMTPCachePair(target, draft)

    def rollback(self, caches, *, target_drops, draft_drops, verify_width):
        self.calls.append(
            (
                "rollback",
                tuple(target_drops),
                tuple(draft_drops),
                verify_width,
            )
        )

    def detach(self, caches, indices, keep_indices):
        self.calls.append(("detach", tuple(indices), tuple(keep_indices)))
        detached = [
            engine.SelfMTPCachePair([caches.target[index]], [caches.draft[index]])
            for index in indices
        ]
        remaining = engine.SelfMTPCachePair(
            [caches.target[index] for index in keep_indices],
            [caches.draft[index] for index in keep_indices],
        )
        return remaining, detached


def _runtime(*, config=None, capabilities=None):
    calls = []

    def target(inputs, **kwargs):
        calls.append(("target", inputs, kwargs))
        return "target-logits", f"target-hidden-{inputs}"

    def draft(hidden, token_ids, cache, **kwargs):
        calls.append(("draft", hidden, token_ids, cache, kwargs))
        return "draft-logits", "draft-hidden"

    compute = _Compute()
    caches = _Caches()
    runtime = engine.ContinuousSelfMTPRuntime(
        config=config or engine.ContinuousSelfMTPConfig(enabled=True),
        capabilities=capabilities or _capabilities(),
        forwards=engine.RapidForwardSeams(target, draft),
        compute=compute,
        caches=caches,
    )
    return runtime, compute, caches, calls


def _prepare(runtime, uid, **spec_kwargs):
    num_draft = spec_kwargs.pop("num_draft", 2)
    spec = engine.SelfMTPLaneSpec(
        uid=uid,
        prompt=(uid, uid + 1),
        max_tokens=20,
        num_draft=num_draft,
        **spec_kwargs,
    )
    return engine.prepare_self_mtp_lane(spec, runtime)


def test_rapid_forward_seams_use_return_hidden_and_n_confirmed():
    runtime, _compute, _caches, calls = _runtime()
    detached, first = _prepare(runtime, 1)

    assert first.token == detached.lane.cur == 101
    assert calls[0] == (
        "target",
        (1, 2),
        {
            "cache": "target-1",
            "return_hidden": True,
            "n_confirmed": 0,
        },
    )
    assert calls[1][-1] == {"return_hidden": True}


def test_fixed_membership_prepare_attach_propose_commit_detach_lifecycle():
    runtime, compute, caches, _calls = _runtime()
    lane1, first1 = _prepare(runtime, 1)
    lane2, first2 = _prepare(runtime, 2)
    assert (first1.token, first2.token) == (101, 102)

    batch = engine.attach_self_mtp_lanes(None, [lane1, lane2])
    assert [lane.uid for lane in batch.lanes] == [1, 2]
    assert batch.membership_epoch == 1

    proposal = engine.propose_batched_self_mtp(batch)
    assert proposal.lane_uids == (1, 2)
    assert proposal.accepted_lengths == (1, 0)
    assert not [call for call in caches.calls if call[0] == "rollback"]
    with pytest.raises(engine.ContinuousSelfMTPError, match="proposal is open"):
        engine.detach_self_mtp_lanes(batch, [0, 1])

    engine.commit_batched_self_mtp(
        batch,
        proposal,
        emitted_counts=[2, 1],
        terminal=[False, False],
    )
    assert [lane.ntoks for lane in batch.lanes] == [3, 2]
    assert compute.calls[-1] == (
        "commit",
        "opaque-cycle",
        (2, 1),
        (False, False),
    )
    assert caches.calls[-1] == ("rollback", (1, 2), (2, 2), 3)

    previous_epoch = batch.membership_epoch
    batch, detached = engine.detach_self_mtp_lanes(batch, [0, 1])
    assert batch.lanes == []
    assert batch.membership_epoch == previous_epoch + 1
    assert [item.lane.uid for item in detached] == [1, 2]


def test_explicit_abort_restores_open_proposal_through_backend():
    runtime, compute, _caches, _calls = _runtime()
    lane, _first = _prepare(runtime, 1)
    batch = engine.attach_self_mtp_lanes(None, [lane])
    proposal = engine.propose_batched_self_mtp(batch)

    engine.abort_batched_self_mtp(batch, proposal)

    assert batch.proposal_open is False
    assert compute.calls[-1] == ("abort", (1,), "opaque-cycle", None)


def test_propose_exception_aborts_before_retry():
    runtime, compute, _caches, _calls = _runtime()
    lane, _first = _prepare(runtime, 1)
    batch = engine.attach_self_mtp_lanes(None, [lane])
    original = compute.propose

    def fail_once(*args, **kwargs):
        compute.propose = original
        raise RuntimeError("verify failed")

    compute.propose = fail_once
    with pytest.raises(RuntimeError, match="verify failed"):
        engine.propose_batched_self_mtp(batch)

    assert compute.calls[-1] == ("abort", (1,), None, "verify failed")
    assert engine.propose_batched_self_mtp(batch).lane_uids == (1,)


def test_commit_exception_aborts_and_closes_proposal():
    runtime, compute, _caches, _calls = _runtime()
    lane, _first = _prepare(runtime, 1)
    batch = engine.attach_self_mtp_lanes(None, [lane])
    proposal = engine.propose_batched_self_mtp(batch)

    def fail_commit(*args, **kwargs):
        raise RuntimeError("commit failed")

    compute.commit = fail_commit
    with pytest.raises(RuntimeError, match="commit failed"):
        engine.commit_batched_self_mtp(
            batch,
            proposal,
            emitted_counts=[2],
            terminal=[False],
        )

    assert batch.proposal_open is False
    assert compute.calls[-1] == ("abort", (1,), "opaque-cycle", "commit failed")


def test_abort_failure_poisons_batch_and_forbids_reuse():
    runtime, compute, _caches, _calls = _runtime()
    lane, _first = _prepare(runtime, 1)
    batch = engine.attach_self_mtp_lanes(None, [lane])

    def fail_proposal(*args, **kwargs):
        raise RuntimeError("forward failed")

    def fail_abort(*args, **kwargs):
        raise RuntimeError("rollback failed")

    compute.propose = fail_proposal
    compute.abort = fail_abort
    with pytest.raises(engine.ContinuousSelfMTPError, match="rollback failed"):
        engine.propose_batched_self_mtp(batch)

    assert batch.poisoned is True
    with pytest.raises(engine.ContinuousSelfMTPError, match="poisoned"):
        engine.detach_self_mtp_lanes(batch, [0])


def test_default_off_refuses_before_compute():
    runtime, compute, _caches, _calls = _runtime(
        config=engine.ContinuousSelfMTPConfig()
    )
    with pytest.raises(engine.ContinuousSelfMTPUnsupportedError, match="disabled"):
        _prepare(runtime, 1)
    assert compute.calls == []


def test_xtc_is_unconditionally_fail_closed():
    runtime, compute, _caches, _calls = _runtime(
        capabilities=_capabilities(
            transformed_sampling=True,
            logits_processors_exact=True,
            dynamic_membership=True,
            flash_dynamic_membership_attested=True,
        )
    )
    sampling = engine.SelfMTPSampling(temperature=0.8, uses_xtc=True)
    with pytest.raises(engine.ContinuousSelfMTPUnsupportedError, match="XTC"):
        _prepare(runtime, 1, sampling=sampling)
    assert compute.calls == []


def test_transformed_sampling_requires_per_lane_rng_attestation():
    runtime, compute, _caches, _calls = _runtime(
        capabilities=_capabilities(transformed_sampling=True)
    )
    sampling = engine.SelfMTPSampling(temperature=0.8)

    with pytest.raises(engine.ContinuousSelfMTPUnsupportedError, match="per-lane RNG"):
        _prepare(runtime, 1, sampling=sampling)

    assert compute.calls == []


def test_fixed_membership_refuses_incremental_attach_and_partial_detach():
    runtime, _compute, _caches, _calls = _runtime()
    lane1, _ = _prepare(runtime, 1)
    lane2, _ = _prepare(runtime, 2)
    batch = engine.attach_self_mtp_lanes(None, [lane1, lane2])
    lane3, _ = _prepare(runtime, 3)

    with pytest.raises(
        engine.ContinuousSelfMTPUnsupportedError, match="fixed-membership"
    ):
        engine.attach_self_mtp_lanes(batch, [lane3])
    with pytest.raises(
        engine.ContinuousSelfMTPUnsupportedError, match="fixed-membership"
    ):
        engine.detach_self_mtp_lanes(batch, [1])


def test_flash_dynamic_membership_requires_specific_attestation():
    config = engine.ContinuousSelfMTPConfig(
        enabled=True,
        allow_dynamic_membership=True,
        architecture="qwen4-flash-next",
    )
    runtime, _compute, _caches, _calls = _runtime(
        config=config,
        capabilities=_capabilities(dynamic_membership=True),
    )
    lane1, _ = _prepare(runtime, 1)
    batch = engine.attach_self_mtp_lanes(None, [lane1])
    lane2, _ = _prepare(runtime, 2)
    with pytest.raises(engine.ContinuousSelfMTPUnsupportedError, match="Flash dynamic"):
        engine.attach_self_mtp_lanes(batch, [lane2])


def test_attested_flash_runtime_can_use_explicit_dynamic_seam():
    config = engine.ContinuousSelfMTPConfig(
        enabled=True,
        allow_dynamic_membership=True,
        architecture="qwen4-flash-next",
    )
    runtime, _compute, _caches, _calls = _runtime(
        config=config,
        capabilities=_capabilities(
            dynamic_membership=True,
            flash_dynamic_membership_attested=True,
        ),
    )
    lane1, _ = _prepare(runtime, 1)
    lane2, _ = _prepare(runtime, 2)
    batch = engine.attach_self_mtp_lanes(None, [lane1])
    batch = engine.attach_self_mtp_lanes(batch, [lane2])
    assert [lane.uid for lane in batch.lanes] == [1, 2]
    batch, detached = engine.detach_self_mtp_lanes(batch, [1])
    assert [lane.uid for lane in batch.lanes] == [1]
    assert detached[0].lane.uid == 2


def test_commit_rejects_partial_nonterminal_delivery_without_closing_cycle():
    runtime, _compute, _caches, _calls = _runtime()
    lane1, _ = _prepare(runtime, 1)
    lane2, _ = _prepare(runtime, 2)
    batch = engine.attach_self_mtp_lanes(None, [lane1, lane2])
    proposal = engine.propose_batched_self_mtp(batch)

    with pytest.raises(ValueError, match="nonterminal lane"):
        engine.commit_batched_self_mtp(
            batch,
            proposal,
            emitted_counts=[1, 1],
            terminal=[False, False],
        )
    assert batch.proposal_open is True
    assert [lane.ntoks for lane in batch.lanes] == [1, 1]


@pytest.mark.parametrize(
    ("emitted", "terminal", "message"),
    [
        ([True, 1], [False, False], "emitted_counts values"),
        ([1.0, 1], [False, False], "emitted_counts values"),
        ([2, 1], [0, False], "terminal values"),
        ([2, 1], ["yes", False], "terminal values"),
    ],
)
def test_commit_rejects_coercible_vector_values(emitted, terminal, message):
    runtime, _compute, _caches, _calls = _runtime()
    lane1, _ = _prepare(runtime, 1)
    lane2, _ = _prepare(runtime, 2)
    batch = engine.attach_self_mtp_lanes(None, [lane1, lane2])
    proposal = engine.propose_batched_self_mtp(batch)

    with pytest.raises(ValueError, match=message):
        engine.commit_batched_self_mtp(
            batch,
            proposal,
            emitted_counts=emitted,
            terminal=terminal,
        )
    assert batch.proposal_open is True
    assert [lane.ntoks for lane in batch.lanes] == [1, 1]


@pytest.mark.parametrize(
    "factory",
    [
        lambda: engine.ContinuousSelfMTPConfig(enabled=1),
        lambda: engine.ContinuousSelfMTPConfig(allow_dynamic_membership="yes"),
        lambda: engine.ContinuousSelfMTPConfig(architecture=""),
        lambda: engine.ContinuousSelfMTPConfig(architecture=None),
        lambda: engine.RapidForwardSeams(None, lambda *args: None),
        lambda: engine.RapidForwardSeams(lambda *args: None, None),
        lambda: engine.SelfMTPSampling(temperature=True),
        lambda: engine.SelfMTPSampling(temperature="hot"),
        lambda: engine.SelfMTPSampling(temperature=-0.1),
        lambda: engine.SelfMTPSampling(has_logits_processors=1),
        lambda: engine.SelfMTPSampling(uses_xtc="yes"),
        lambda: engine.SelfMTPLaneSpec(True, (), 1),
        lambda: engine.SelfMTPLaneSpec("1", (), 1),
        lambda: engine.SelfMTPLaneSpec(1, (), 0),
        lambda: engine.SelfMTPLaneSpec(1, (), 1, num_draft=True),
        lambda: engine.MTPToken(True, None, False),
        lambda: engine.MTPToken("1", None, False),
        lambda: engine.MTPToken(1, None, 0),
    ],
)
def test_value_objects_reject_invalid_contract_values(factory):
    with pytest.raises((TypeError, ValueError)):
        factory()


@pytest.mark.parametrize("n_confirmed", [True, 1.5, "1", -1])
def test_forward_seam_rejects_invalid_confirmed_count(n_confirmed):
    seams = engine.RapidForwardSeams(lambda *args, **kwargs: None, lambda *args: None)
    with pytest.raises(ValueError):
        seams.target([], None, n_confirmed=n_confirmed)


def test_missing_core_and_sampling_capabilities_fail_closed():
    runtime, _compute, _caches, _calls = _runtime(
        capabilities=engine.ContinuousSelfMTPCapabilities()
    )
    with pytest.raises(engine.ContinuousSelfMTPUnsupportedError, match="missing"):
        _prepare(runtime, 1)

    runtime, _compute, _caches, _calls = _runtime(
        capabilities=_capabilities(per_lane_rng=True)
    )
    with pytest.raises(engine.ContinuousSelfMTPUnsupportedError, match="verification"):
        _prepare(
            runtime,
            1,
            sampling=engine.SelfMTPSampling(temperature=0.7),
        )

    runtime, _compute, _caches, _calls = _runtime()
    with pytest.raises(engine.ContinuousSelfMTPUnsupportedError, match="processors"):
        _prepare(
            runtime,
            1,
            sampling=engine.SelfMTPSampling(has_logits_processors=True),
        )


def test_prepare_rejects_foreign_or_inconsistent_backend_result():
    runtime, compute, _caches, _calls = _runtime()
    compute.prepare = lambda *args: object()
    with pytest.raises(TypeError, match="PreparedLaneData"):
        _prepare(runtime, 1)

    runtime, compute, _caches, _calls = _runtime()
    original = compute.prepare
    compute.prepare = lambda *args: replace(
        original(*args),
        first_token=engine.MTPToken(101, None, True),
    )
    with pytest.raises(engine.ContinuousSelfMTPError, match="cannot be a draft"):
        _prepare(runtime, 1)

    runtime, compute, _caches, _calls = _runtime()
    original = compute.prepare
    compute.prepare = lambda *args: replace(original(*args), cur=999)
    with pytest.raises(engine.ContinuousSelfMTPError, match="disagree"):
        _prepare(runtime, 1)


def test_attach_rejects_invalid_membership_boundaries():
    runtime, _compute, _caches, _calls = _runtime()
    with pytest.raises(ValueError, match="empty"):
        engine.attach_self_mtp_lanes(None, [], runtime=runtime)

    lane1, _ = _prepare(runtime, 1)
    batch = engine.attach_self_mtp_lanes(None, [lane1])
    assert engine.attach_self_mtp_lanes(batch, []) is batch
    batch.proposal_open = True
    with pytest.raises(engine.ContinuousSelfMTPError, match="proposal is open"):
        engine.attach_self_mtp_lanes(batch, [])
    batch.proposal_open = False

    other_runtime, _compute, _caches, _calls = _runtime()
    foreign, _ = _prepare(other_runtime, 2)
    with pytest.raises(engine.ContinuousSelfMTPError, match="one runtime"):
        engine.attach_self_mtp_lanes(None, [lane1, foreign], runtime=runtime)

    duplicate, _ = _prepare(runtime, 1)
    with pytest.raises(ValueError, match="unique"):
        engine.attach_self_mtp_lanes(None, [lane1, duplicate])

    different_depth, _ = _prepare(runtime, 3, num_draft=1)
    with pytest.raises(ValueError, match="draft depth"):
        engine.attach_self_mtp_lanes(None, [lane1, different_depth])


def test_dynamic_capability_and_missing_abort_fail_closed():
    default_runtime, _compute, _caches, _calls = _runtime()
    assert engine.supports_dynamic_membership(default_runtime) is False

    config = engine.ContinuousSelfMTPConfig(enabled=True, allow_dynamic_membership=True)
    runtime, compute, _caches, _calls = _runtime(
        config=config,
        capabilities=_capabilities(dynamic_membership=False),
    )
    first, _ = _prepare(runtime, 1)
    second, _ = _prepare(runtime, 2)
    batch = engine.attach_self_mtp_lanes(None, [first])
    assert engine.supports_dynamic_membership(runtime) is False
    with pytest.raises(engine.ContinuousSelfMTPUnsupportedError, match="capability"):
        engine.attach_self_mtp_lanes(batch, [second])

    flash_runtime, _compute, _caches, _calls = _runtime(
        config=engine.ContinuousSelfMTPConfig(
            enabled=True,
            allow_dynamic_membership=True,
            architecture="flash",
        ),
        capabilities=_capabilities(dynamic_membership=True),
    )
    assert engine.supports_dynamic_membership(flash_runtime) is False

    dense_runtime, _compute, _caches, _calls = _runtime(
        config=config,
        capabilities=_capabilities(dynamic_membership=True),
    )
    assert engine.supports_dynamic_membership(dense_runtime) is True

    compute.abort = None
    compute.propose = lambda *args: (_ for _ in ()).throw(RuntimeError("failed"))
    with pytest.raises(engine.ContinuousSelfMTPError, match="no abort"):
        engine.propose_batched_self_mtp(batch)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"lane_uids": (999,)}, "lane order"),
        ({"accepted_lengths": ()}, "lane count"),
        ({"draft_depths": (True,)}, "non-integer"),
        ({"accepted_lengths": (-1,)}, "invalid acceptance"),
        ({"accepted_lengths": (3,)}, "invalid acceptance"),
        (
            {
                "draft_depths": (20,),
                "accepted_lengths": (1,),
                "target_drops": (19,),
                "draft_drops": (20,),
            },
            "remaining draft budget",
        ),
        ({"target_drops": (0,)}, "rollback is inconsistent"),
        ({"outputs": ((),)}, "accepted drafts plus one"),
        ({"outputs": ((object(), object()),)}, "invalid token"),
        (
            {
                "outputs": (
                    (engine.MTPToken(1, None, False), engine.MTPToken(2, None, False)),
                )
            },
            "not marked as a draft",
        ),
        (
            {
                "outputs": (
                    (engine.MTPToken(1, None, True), engine.MTPToken(2, None, True)),
                )
            },
            "final target token",
        ),
    ],
)
def test_proposal_validation_aborts_malformed_computation(changes, message):
    runtime, compute, _caches, _calls = _runtime()
    lane, _ = _prepare(runtime, 1)
    batch = engine.attach_self_mtp_lanes(None, [lane])
    valid = compute.propose(batch.lanes, batch.caches, runtime.forwards)
    compute.propose = lambda *args: replace(valid, **changes)

    with pytest.raises(engine.ContinuousSelfMTPError, match=message):
        engine.propose_batched_self_mtp(batch)
    assert compute.calls[-1][0] == "abort"


def test_propose_rejects_open_empty_and_foreign_results():
    runtime, compute, _caches, _calls = _runtime()
    lane, _ = _prepare(runtime, 1)
    batch = engine.attach_self_mtp_lanes(None, [lane])
    proposal = engine.propose_batched_self_mtp(batch)
    with pytest.raises(engine.ContinuousSelfMTPError, match="already open"):
        engine.propose_batched_self_mtp(batch)
    engine.abort_batched_self_mtp(batch, proposal)

    batch.lanes = []
    with pytest.raises(ValueError, match="empty"):
        engine.propose_batched_self_mtp(batch)
    batch.lanes = [lane.lane]
    compute.propose = lambda *args: object()
    with pytest.raises(TypeError, match="CycleComputation"):
        engine.propose_batched_self_mtp(batch)


def test_commit_and_detach_reject_invalid_transaction_boundaries():
    runtime, _compute, caches, _calls = _runtime()
    lane, _ = _prepare(runtime, 1)
    batch = engine.attach_self_mtp_lanes(None, [lane])
    proposal = engine.propose_batched_self_mtp(batch)

    foreign = replace(proposal)
    with pytest.raises(engine.ContinuousSelfMTPError, match="currently open"):
        engine.commit_batched_self_mtp(
            batch, foreign, emitted_counts=[2], terminal=[False]
        )
    with pytest.raises(engine.ContinuousSelfMTPError, match="currently open"):
        engine.abort_batched_self_mtp(batch, foreign)

    proposal_epoch = proposal.membership_epoch
    object.__setattr__(proposal, "membership_epoch", proposal_epoch + 1)
    with pytest.raises(engine.ContinuousSelfMTPError, match="membership changed"):
        engine.commit_batched_self_mtp(
            batch, proposal, emitted_counts=[2], terminal=[False]
        )
    object.__setattr__(proposal, "membership_epoch", proposal_epoch)

    original_uids = proposal.lane_uids
    object.__setattr__(proposal, "lane_uids", (999,))
    with pytest.raises(engine.ContinuousSelfMTPError, match="lane order changed"):
        engine.commit_batched_self_mtp(
            batch, proposal, emitted_counts=[2], terminal=[False]
        )
    object.__setattr__(proposal, "lane_uids", original_uids)

    for emitted, terminal, message in [
        ([], [False], "one entry"),
        ([2], [], "one entry"),
        ([-1], [True], "out of range"),
        ([3], [True], "out of range"),
    ]:
        with pytest.raises(ValueError, match=message):
            engine.commit_batched_self_mtp(
                batch, proposal, emitted_counts=emitted, terminal=terminal
            )

    engine.abort_batched_self_mtp(batch, proposal)
    for indices, exception, message in [
        ([0, 0], ValueError, "unique"),
        ([-1], IndexError, "outside"),
        ([2], IndexError, "outside"),
    ]:
        with pytest.raises(exception, match=message):
            engine.detach_self_mtp_lanes(batch, indices)
    assert engine.detach_self_mtp_lanes(batch, []) == (batch, [])

    original_detach = caches.detach
    caches.detach = lambda *args: (batch.caches, [])
    with pytest.raises(engine.ContinuousSelfMTPError, match="wrong detached"):
        engine.detach_self_mtp_lanes(batch, [0])
    caches.detach = original_detach


def test_terminal_commit_rolls_back_undelivered_target_prefix():
    runtime, _compute, caches, _calls = _runtime()
    lane, _ = _prepare(runtime, 1)
    batch = engine.attach_self_mtp_lanes(None, [lane])
    proposal = engine.propose_batched_self_mtp(batch)

    engine.commit_batched_self_mtp(
        batch,
        proposal,
        emitted_counts=[1],
        terminal=[True],
    )

    assert caches.calls[-1] == ("rollback", (2,), (2,), 3)
