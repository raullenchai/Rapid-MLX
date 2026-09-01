# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import replace

import pytest

from vllm_mlx.spec_decode.mtp.prepared_state import (
    PreparedMTPState,
    PreparedStateIdentity,
    PreparedStateMetadata,
    RestoreReason,
    evaluate_restore,
    fingerprint_config,
    fingerprint_tokens,
    prepare_mtp_state,
)


def _identity(**changes) -> PreparedStateIdentity:
    values = {
        "model_id": "Qwen/Qwen3.8-Flash-Next",
        "model_revision": "f5d08274",
        "speculative_config_fingerprint": fingerprint_config(
            {"method": "mtp", "num_speculative_tokens": 3}
        ),
        "target_cache_layout": "qwen4:12qsa+36arrays:bf16",
        "mtp_cache_layout": "qwen4-mtp:1qsa:bf16",
        "seed_hidden_layout": "bf16[1,1,2048]",
        "adapter_id": None,
        "tokenizer_fingerprint": "tokenizer-sha256",
    }
    values.update(changes)
    return PreparedStateIdentity(**values)


def _state(
    *,
    covered: int = 128,
    identity: PreparedStateIdentity | None = None,
    captured_at: float = 100.0,
) -> tuple[PreparedMTPState, tuple[int, ...]]:
    prefix = tuple(range(covered))
    return (
        prepare_mtp_state(
            identity=identity or _identity(),
            prefix_tokens=prefix,
            target_cache=object(),
            target_cache_tokens=covered,
            mtp_cache=object(),
            mtp_cache_pairs=covered - 1,
            seed_hidden=object(),
            captured_at=captured_at,
        ),
        prefix,
    )


def _evaluate(
    state: PreparedMTPState,
    prefix: tuple[int, ...],
    **changes,
):
    values = {
        "expected_identity": _identity(),
        "request_tokens": prefix + (999,),
        "target_cache_tokens": len(prefix),
        "mtp_cache_pairs": len(prefix) - 1,
        "now": 110.0,
        "max_age_seconds": 60.0,
        "min_useful_prefix_tokens": 64,
    }
    values.update(changes)
    return evaluate_restore(state, **values)


def test_exact_joint_boundary_is_restore_eligible() -> None:
    state, prefix = _state()

    decision = _evaluate(state, prefix)

    assert decision.eligible is True
    assert decision.reason is RestoreReason.ELIGIBLE
    assert decision.covered_tokens == 128
    assert decision.resume_at == 128
    assert decision.bypass_hit is False


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"target_cache_tokens": 127}, "exact covered prefix length"),
        ({"mtp_cache_pairs": 126}, "target_cache_tokens - 1"),
        ({"seed_hidden": None}, "seed_hidden"),
        ({"target_cache": None}, "target_cache"),
        ({"mtp_cache": None}, "mtp_cache"),
    ],
)
def test_capture_rejects_incomplete_or_unaligned_state(changes, message) -> None:
    kwargs = {
        "identity": _identity(),
        "prefix_tokens": tuple(range(128)),
        "target_cache": object(),
        "target_cache_tokens": 128,
        "mtp_cache": object(),
        "mtp_cache_pairs": 127,
        "seed_hidden": object(),
        "captured_at": 100.0,
    }
    kwargs.update(changes)

    with pytest.raises(ValueError, match=message):
        prepare_mtp_state(**kwargs)


def test_valid_trivial_hit_fails_open_to_normal_mtp() -> None:
    state, prefix = _state(covered=8)

    decision = _evaluate(state, prefix)

    assert decision.eligible is False
    assert decision.reason is RestoreReason.TRIVIAL_HIT
    assert decision.bypass_hit is True
    assert decision.resume_at is None


@pytest.mark.parametrize(
    "expected_identity",
    [
        _identity(model_id="other/model"),
        _identity(model_revision="different-revision"),
        _identity(adapter_id="adapter-a"),
        _identity(tokenizer_fingerprint="different-tokenizer"),
    ],
)
def test_model_identity_mismatch_refuses_restore(expected_identity) -> None:
    state, prefix = _state()

    decision = _evaluate(
        state,
        prefix,
        expected_identity=expected_identity,
    )

    assert decision.eligible is False
    assert decision.reason is RestoreReason.MODEL_MISMATCH
    assert decision.bypass_hit is False


@pytest.mark.parametrize(
    "expected_identity",
    [
        _identity(
            speculative_config_fingerprint=fingerprint_config(
                {"method": "mtp", "num_speculative_tokens": 2}
            )
        ),
        _identity(target_cache_layout="different-target-layout"),
        _identity(mtp_cache_layout="different-mtp-layout"),
        _identity(seed_hidden_layout="bf16[1,1,4096]"),
    ],
)
def test_config_or_layout_mismatch_refuses_restore(expected_identity) -> None:
    state, prefix = _state()

    decision = _evaluate(
        state,
        prefix,
        expected_identity=expected_identity,
    )

    assert decision.eligible is False
    assert decision.reason is RestoreReason.CONFIG_MISMATCH


def test_stale_state_refuses_restore() -> None:
    state, prefix = _state(captured_at=100.0)

    decision = _evaluate(
        state,
        prefix,
        now=161.0,
        max_age_seconds=60.0,
    )

    assert decision.eligible is False
    assert decision.reason is RestoreReason.STALE


@pytest.mark.parametrize(
    "changes",
    [
        {"target_cache_tokens": 127},
        {"mtp_cache_pairs": 126},
        {"request_tokens": tuple(range(127)) + (777, 999)},
        {"request_tokens": tuple(range(128))},
    ],
)
def test_live_or_token_boundary_mismatch_refuses_restore(changes) -> None:
    state, prefix = _state()

    decision = _evaluate(state, prefix, **changes)

    assert decision.eligible is False
    assert decision.reason is RestoreReason.BOUNDARY_MISMATCH


@pytest.mark.parametrize(
    "metadata_change",
    [
        {"mtp_covered_pairs": 3},
        {"schema_version": 999},
        {"captured_at": "not-a-timestamp"},
        {"boundary_fingerprint": "z" * 64},
    ],
)
def test_corrupt_persisted_metadata_refuses_without_raising(
    metadata_change,
) -> None:
    state, prefix = _state()
    corrupt = PreparedMTPState(
        metadata=replace(state.metadata, **metadata_change),
        target_cache=state.target_cache,
        mtp_cache=state.mtp_cache,
        seed_hidden=state.seed_hidden,
    )

    decision = _evaluate(corrupt, prefix)

    assert decision.eligible is False
    assert decision.reason is RestoreReason.MALFORMED


def test_config_fingerprint_is_order_independent_and_value_sensitive() -> None:
    left = fingerprint_config({"method": "mtp", "k": 3})
    reordered = fingerprint_config({"k": 3, "method": "mtp"})
    changed = fingerprint_config({"method": "mtp", "k": 2})

    assert left == reordered
    assert left != changed


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"model_id": ""}, "model_id"),
        ({"model_revision": 3}, "model_revision"),
        ({"adapter_id": " "}, "adapter_id"),
        ({"tokenizer_fingerprint": 3}, "tokenizer_fingerprint"),
    ],
)
def test_identity_rejects_empty_or_non_string_fields(changes, message) -> None:
    with pytest.raises(ValueError, match=message):
        _identity(**changes)


def test_identity_from_config_uses_canonical_fingerprint() -> None:
    identity = PreparedStateIdentity.from_config(
        model_id="model",
        model_revision="revision",
        speculative_config={"k": 3, "method": "mtp"},
        target_cache_layout="target",
        mtp_cache_layout="draft",
        seed_hidden_layout="hidden",
    )
    assert identity.speculative_config_fingerprint == fingerprint_config(
        {"method": "mtp", "k": 3}
    )


@pytest.mark.parametrize("config", [[], {"bad": object()}, {"nan": float("nan")}])
def test_config_fingerprint_rejects_noncanonical_input(config) -> None:
    error = TypeError if isinstance(config, list) else ValueError
    with pytest.raises(error):
        fingerprint_config(config)


@pytest.mark.parametrize("tokens", [[True], [-1], [1 << 64]])
def test_token_fingerprint_rejects_invalid_ids(tokens) -> None:
    with pytest.raises(ValueError, match="tokens|64-bit"):
        fingerprint_tokens(tokens)


@pytest.mark.parametrize(
    "changes",
    [
        {"prefix_tokens": (), "target_cache_tokens": 0, "mtp_cache_pairs": 0},
        {"target_cache_tokens": True},
        {"mtp_cache_pairs": -1},
        {"captured_at": float("inf")},
        {"captured_at": -1.0},
    ],
)
def test_capture_rejects_empty_counts_and_bad_timestamps(changes) -> None:
    kwargs = {
        "identity": _identity(),
        "prefix_tokens": tuple(range(2)),
        "target_cache": object(),
        "target_cache_tokens": 2,
        "mtp_cache": object(),
        "mtp_cache_pairs": 1,
        "seed_hidden": object(),
        "captured_at": 100.0,
    }
    kwargs.update(changes)
    with pytest.raises(ValueError):
        prepare_mtp_state(**kwargs)


@pytest.mark.parametrize("minimum", [True, 0, 1.5])
def test_restore_rejects_invalid_minimum(minimum) -> None:
    state, prefix = _state()
    with pytest.raises(ValueError, match="positive integer"):
        _evaluate(state, prefix, min_useful_prefix_tokens=minimum)


@pytest.mark.parametrize("max_age", [-1, float("inf")])
def test_restore_rejects_invalid_max_age(max_age) -> None:
    state, prefix = _state()
    with pytest.raises(ValueError, match="finite and non-negative"):
        _evaluate(state, prefix, max_age_seconds=max_age)


def test_restore_refuses_malformed_identity_time_and_live_values() -> None:
    state, prefix = _state()
    assert (
        _evaluate(state, prefix, expected_identity=object()).reason
        is RestoreReason.MALFORMED
    )
    assert _evaluate(state, prefix, now=object()).reason is RestoreReason.MALFORMED
    assert _evaluate(state, prefix, now=99.0).reason is RestoreReason.STALE
    assert (
        _evaluate(state, prefix, target_cache_tokens=True).reason
        is RestoreReason.MALFORMED
    )
    malformed_request = (True,) + prefix[1:] + (999,)
    assert (
        _evaluate(state, prefix, request_tokens=malformed_request).reason
        is RestoreReason.MALFORMED
    )


@pytest.mark.parametrize(
    "metadata",
    [
        None,
        PreparedStateMetadata(
            identity=_identity(),
            covered_tokens=-1,
            mtp_covered_pairs=0,
            boundary_fingerprint="0" * 64,
            captured_at=1.0,
        ),
        PreparedStateMetadata(
            identity=_identity(),
            covered_tokens=1,
            mtp_covered_pairs=True,
            boundary_fingerprint="0" * 64,
            captured_at=1.0,
        ),
    ],
)
def test_restore_refuses_non_metadata_and_invalid_metadata_counts(metadata) -> None:
    state, prefix = _state()
    corrupt = PreparedMTPState(
        metadata, state.target_cache, state.mtp_cache, state.seed_hidden
    )
    assert _evaluate(corrupt, prefix).reason is RestoreReason.MALFORMED
