# SPDX-License-Identifier: Apache-2.0
"""Model-free contracts for the future continuous self-MTP coordinator."""

from __future__ import annotations

from dataclasses import replace

import pytest

from vllm_mlx.spec_decode.mtp.batched import (
    BatchedMTPCapabilities,
    BatchedMTPConfig,
    BatchedMTPRoute,
    LaneAdmission,
    SamplingContract,
    assess_lane,
    plan_admission,
)


def _capabilities(**overrides) -> BatchedMTPCapabilities:
    values = dict(
        target_batch_forward=True,
        mtp_batch_forward=True,
        ragged_target_rollback=True,
        ragged_mtp_rollback=True,
        atomic_cache_commit=True,
        per_lane_rng=True,
        transformed_distribution_verify=True,
        dynamic_membership=True,
    )
    values.update(overrides)
    return BatchedMTPCapabilities(**values)


def _config(**overrides) -> BatchedMTPConfig:
    values = dict(enabled=True, hard_reserve_bytes=100, max_draft_tokens=2)
    values.update(overrides)
    return BatchedMTPConfig(**values)


def test_feature_is_default_off_and_capabilities_fail_closed():
    lane = LaneAdmission("a")
    gate = assess_lane(
        lane,
        config=BatchedMTPConfig(),
        capabilities=BatchedMTPCapabilities(),
    )

    assert not gate.eligible
    assert "batched self-MTP is disabled" in gate.reasons
    assert "missing capability: target_batch_forward" in gate.reasons


def test_truthy_non_boolean_capability_does_not_open_gate():
    caps = replace(_capabilities(), target_batch_forward="yes")
    gate = assess_lane(LaneAdmission("a"), config=_config(), capabilities=caps)
    assert gate.reasons == ("missing capability: target_batch_forward",)


@pytest.mark.parametrize(
    "missing",
    [
        "target_batch_forward",
        "mtp_batch_forward",
        "ragged_target_rollback",
        "ragged_mtp_rollback",
        "atomic_cache_commit",
    ],
)
def test_each_core_capability_is_mandatory(missing):
    gate = assess_lane(
        LaneAdmission("a"),
        config=_config(),
        capabilities=replace(_capabilities(), **{missing: False}),
    )
    assert not gate.eligible
    assert gate.reasons == (f"missing capability: {missing}",)


def test_sampled_lanes_require_rng_and_transformed_verifier():
    lane = LaneAdmission("sampled", sampling=SamplingContract(greedy=False))
    gate = assess_lane(
        lane,
        config=_config(),
        capabilities=_capabilities(
            per_lane_rng=False, transformed_distribution_verify=False
        ),
    )
    assert gate.reasons == (
        "missing capability: per_lane_rng",
        "missing capability: transformed_distribution_verify",
    )


def test_xtc_and_logits_processors_fail_closed():
    xtc = LaneAdmission("xtc", sampling=SamplingContract(greedy=False, uses_xtc=True))
    processors = LaneAdmission(
        "processors", sampling=SamplingContract(has_logits_processors=True)
    )
    caps = _capabilities(xtc_exact_verify=False)

    assert assess_lane(xtc, config=_config(), capabilities=caps).reasons == (
        "XTC verification is not exact",
    )
    assert assess_lane(processors, config=_config(), capabilities=caps).reasons == (
        "logits processors are not supported",
    )


def test_dynamic_membership_requires_explicit_capability():
    gate = assess_lane(
        LaneAdmission("a"),
        config=_config(allow_dynamic_membership=True),
        capabilities=_capabilities(dynamic_membership=False),
    )
    assert gate.reasons == ("missing capability: dynamic_membership",)


def test_admission_lowers_depth_to_admit_more_lanes():
    lanes = [
        LaneAdmission(str(index), base_bytes=10, bytes_per_draft_token=20)
        for index in range(4)
    ]
    # usable=140: K=2 fits 2 lanes (50 each), K=1 fits 4 lanes (30 each).
    decision = plan_admission(
        lanes,
        config=_config(),
        capabilities=_capabilities(),
        free_bytes=240,
    )

    assert decision.route is BatchedMTPRoute.BATCHED_MTP
    assert decision.batched_lane_ids == ("0", "1", "2", "3")
    assert decision.draft_tokens == 1
    assert decision.estimated_bytes == 120


def test_admission_keeps_ineligible_lanes_plain_and_overflow_queued():
    lanes = [
        LaneAdmission("plain", cache_ready=False),
        LaneAdmission("a", base_bytes=60),
        LaneAdmission("b", base_bytes=60),
        LaneAdmission("c", base_bytes=60),
    ]
    decision = plan_admission(
        lanes,
        config=_config(max_lanes=3),
        capabilities=_capabilities(),
        free_bytes=230,
    )

    assert decision.batched_lane_ids == ("a", "b")
    assert decision.plain_lane_ids == ("plain",)
    assert decision.queued_lane_ids == ("c",)


def test_lanes_over_configured_limit_are_explicitly_queued():
    lanes = [LaneAdmission(str(index)) for index in range(5)]
    decision = plan_admission(
        lanes,
        config=_config(max_lanes=3),
        capabilities=_capabilities(),
        free_bytes=1000,
    )

    assert decision.batched_lane_ids == ("0", "1", "2")
    assert decision.queued_lane_ids == ("3", "4")


def test_memory_reserve_queues_an_otherwise_eligible_batch():
    lanes = [LaneAdmission("a", base_bytes=1), LaneAdmission("b", base_bytes=1)]
    decision = plan_admission(
        lanes,
        config=_config(),
        capabilities=_capabilities(),
        free_bytes=100,
    )

    assert decision.route is BatchedMTPRoute.QUEUE
    assert decision.queued_lane_ids == ("a", "b")
    assert decision.draft_tokens == 0


def test_single_eligible_lane_uses_plain_decode_instead_of_waiting():
    decision = plan_admission(
        [LaneAdmission("a")],
        config=_config(),
        capabilities=_capabilities(),
        free_bytes=1000,
    )
    assert decision.route is BatchedMTPRoute.PLAIN_DECODE
    assert decision.plain_lane_ids == ("a",)


def test_live_adapter_can_explicitly_admit_a_serial_continuous_cohort():
    decision = plan_admission(
        [LaneAdmission("a")],
        config=_config(min_batch_lanes=1),
        capabilities=_capabilities(),
        free_bytes=1000,
    )
    assert decision.route is BatchedMTPRoute.BATCHED_MTP
    assert decision.batched_lane_ids == ("a",)


def test_inputs_are_validated_without_model_or_array_dependencies():
    with pytest.raises(ValueError, match="unique"):
        plan_admission(
            [LaneAdmission("a"), LaneAdmission("a")],
            config=_config(),
            capabilities=_capabilities(),
            free_bytes=1000,
        )
    with pytest.raises(ValueError, match="policy flags"):
        BatchedMTPConfig(enabled=1)
    with pytest.raises(ValueError, match="free_bytes must be an integer"):
        plan_admission(
            [LaneAdmission("a"), LaneAdmission("b")],
            config=_config(),
            capabilities=_capabilities(),
            free_bytes=True,
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"max_lanes": True}, "positive integer"),
        ({"max_lanes": 0}, "must be positive"),
        ({"max_draft_tokens": True}, "positive integer"),
        ({"max_draft_tokens": 0}, "must be positive"),
        ({"min_batch_lanes": True}, "must be an integer"),
        ({"min_batch_lanes": 0}, "must be positive"),
        ({"max_lanes": 1, "min_batch_lanes": 2}, "cannot exceed"),
        ({"hard_reserve_bytes": -1}, "cannot be negative"),
    ],
)
def test_policy_numeric_contracts_fail_closed(changes, message):
    with pytest.raises(ValueError, match=message):
        BatchedMTPConfig(**changes)


@pytest.mark.parametrize("changes", [{"greedy": 1}, {"uses_xtc": "yes"}])
def test_sampling_contract_requires_real_booleans(changes):
    with pytest.raises(ValueError, match="flags must be booleans"):
        SamplingContract(**changes)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"lane_id": ""}, "non-empty string"),
        ({"base_bytes": True}, "estimates must be integers"),
        ({"bytes_per_draft_token": -1}, "cannot be negative"),
        ({"cache_ready": 1}, "flags must be booleans"),
    ],
)
def test_lane_metadata_contracts_fail_closed(changes, message):
    values = {"lane_id": "lane"}
    values.update(changes)
    with pytest.raises(ValueError, match=message):
        LaneAdmission(**values)


def test_terminal_empty_and_plain_on_pressure_routes_are_explicit():
    terminal = assess_lane(
        LaneAdmission("done", terminal=True),
        config=_config(),
        capabilities=_capabilities(),
    )
    assert terminal.reasons == ("lane is terminal",)

    empty = plan_admission(
        [], config=_config(), capabilities=_capabilities(), free_bytes=1000
    )
    assert empty.route is BatchedMTPRoute.PLAIN_DECODE
    assert empty.reasons == ("no candidate lanes",)

    lanes = [LaneAdmission("a", base_bytes=1), LaneAdmission("b", base_bytes=1)]
    plain = plan_admission(
        lanes,
        config=_config(queue_on_memory_pressure=False),
        capabilities=_capabilities(),
        free_bytes=100,
    )
    assert plain.route is BatchedMTPRoute.PLAIN_DECODE
    assert plain.plain_lane_ids == ("a", "b")

    overflow = plan_admission(
        lanes + [LaneAdmission("c", base_bytes=1)],
        config=_config(max_lanes=2, queue_on_memory_pressure=False),
        capabilities=_capabilities(),
        free_bytes=1000,
    )
    assert overflow.route is BatchedMTPRoute.BATCHED_MTP
    assert overflow.plain_lane_ids == ("c",)

    with pytest.raises(ValueError, match="cannot be negative"):
        plan_admission(
            lanes,
            config=_config(),
            capabilities=_capabilities(),
            free_bytes=-1,
        )
