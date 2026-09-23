# SPDX-License-Identifier: Apache-2.0
"""Hermetic contract tests for the Qwen auto-runtime planner."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace

import pytest

import rapid_mlx.qwen_runtime_plan as qwen_plan
from rapid_mlx.qwen_runtime_plan import (
    PlanReason,
    QwenDrafterIdentity,
    QwenFallbackTarget,
    QwenModeEvidence,
    QwenModeQualification,
    QwenQualificationRow,
    QwenRuntimePlan,
    QwenTargetIdentity,
    RecoveryAction,
    ResolvedQwenArtifact,
    SelectionSource,
    SpeculativeIntent,
    TargetLane,
    TextMode,
    complete_qwen_fallback_start,
    complete_qwen_reload,
    resolve_qwen_fallback_start_failure,
    resolve_qwen_install_failure,
    resolve_qwen_runtime_plan,
)

TARGET_REVISION = "1" * 40
DRAFTER_REVISION = "2" * 40
TARGET_VERIFICATION_ID = "hf-snapshot-sha256:" + "a" * 64
TARGET_VERIFICATION_AUTHORITY = "rapid_mlx.qwen_artifact:hub-snapshot-v1"
DRAFTER_VERIFICATION_ID = "hf_blob:" + "c" * 64
RUNTIME_VERSIONS = (
    ("mlx", "0.32.1"),
    ("rapid-mlx", "0.15.0"),
    ("vendored-mllm-abi", "3b"),
)


def _target() -> QwenTargetIdentity:
    return QwenTargetIdentity(
        target_repo="example/Qwen-Exact-4bit",
        target_revision=TARGET_REVISION,
        outer_model_type="qwen3_5_moe",
        language_model_type="qwen3_5_moe_text",
        quantization="group-size=64;bits=4",
        weight_layout="safetensors-sharded-v1",
        layer_layout=("linear", "linear", "full_attention"),
        cache_geometry=(("key_head_dim", "128"), ("value_heads", "32")),
    )


def _verified_target(
    target: QwenTargetIdentity | None = None,
    *,
    verification_id: str = TARGET_VERIFICATION_ID,
    authority: str = TARGET_VERIFICATION_AUTHORITY,
) -> qwen_plan.VerifiedQwenTarget:
    return qwen_plan._mint_verified_qwen_target(
        identity=target or _target(),
        verification_id=verification_id,
        verification_authority=authority,
    )


def _drafter(
    *,
    path: str = "mtp/model.safetensors",
    artifact_verification_id: str = DRAFTER_VERIFICATION_ID,
) -> QwenDrafterIdentity:
    return QwenDrafterIdentity(
        repo="example/Qwen-Exact-MTP-4bit",
        revision=DRAFTER_REVISION,
        artifact_path=path,
        artifact_verification_id=artifact_verification_id,
    )


def _native_qualification() -> QwenModeQualification:
    return QwenModeQualification(
        mode=TextMode.NATIVE_AR,
        required_probes=("native-cache-api-v1",),
        receipt_id="receipt/native-v1",
    )


def _mtp_qualification() -> QwenModeQualification:
    return QwenModeQualification(
        mode=TextMode.MTP,
        required_probes=("mtp-injector-v1", "mtp-install-v1"),
        receipt_id="receipt/mtp-v1",
        drafter_identity=_drafter(),
    )


def _row(
    *,
    qualification_id: str = "qwen-exact-auto-v1",
    public_alias: str = "qwen-exact-4bit",
    target: QwenTargetIdentity | None = None,
    expected_verification_id: str = TARGET_VERIFICATION_ID,
    expected_verification_authority: str = TARGET_VERIFICATION_AUTHORITY,
    modes: tuple[QwenModeQualification, ...] | None = None,
) -> QwenQualificationRow:
    return QwenQualificationRow(
        qualification_id=qualification_id,
        public_alias=public_alias,
        target_identity=target or _target(),
        expected_target_verification_id=expected_verification_id,
        expected_target_verification_authority=expected_verification_authority,
        mode_qualifications=(
            modes
            if modes is not None
            else (_native_qualification(), _mtp_qualification())
        ),
        preferred_text_mode=TextMode.MTP,
        fallback_chain=(TextMode.NATIVE_AR, TextMode.NONE),
        runtime_versions=RUNTIME_VERSIONS,
        hardware_classes=("m1-max-64gb", "m4-pro-48gb"),
    )


def _artifact(
    *,
    public_alias: str | None = "qwen-exact-4bit",
    target: QwenTargetIdentity | None = None,
    verified_target: qwen_plan.VerifiedQwenTarget | None = None,
    evidence: tuple[QwenModeEvidence, ...] | None = None,
    runtime_versions: tuple[tuple[str, str], ...] = RUNTIME_VERSIONS,
    hardware_class: str = "m4-pro-48gb",
) -> ResolvedQwenArtifact:
    return ResolvedQwenArtifact(
        verified_target=verified_target or _verified_target(target),
        public_alias=public_alias,
        runtime_versions=runtime_versions,
        hardware_class=hardware_class,
        mode_evidence=(
            evidence
            if evidence is not None
            else (
                QwenModeEvidence(TextMode.NATIVE_AR, ("native-cache-api-v1",)),
                QwenModeEvidence(
                    TextMode.MTP,
                    ("mtp-injector-v1", "mtp-install-v1"),
                    _drafter(),
                ),
            )
        ),
    )


def _legacy_plan() -> QwenRuntimePlan:
    return QwenRuntimePlan(
        target_lane=TargetLane.TEXT,
        text_mode=TextMode.MTP,
        selection_source=SelectionSource.ALIAS_DEFAULT,
        qualification_id=None,
        receipt_id=None,
        verified_target=None,
        reason=PlanReason.LEGACY_ALIAS_DEFAULT,
        media_enabled=False,
    )


def _resolve(
    *,
    legacy: QwenRuntimePlan | None = None,
    intent: SpeculativeIntent = SpeculativeIntent.ALIAS_DEFAULT,
    auto_enabled: bool = True,
    artifact: ResolvedQwenArtifact | None = None,
    rows: tuple[QwenQualificationRow, ...] | None = None,
    operator_lane: TargetLane | None = None,
) -> QwenRuntimePlan:
    return resolve_qwen_runtime_plan(
        legacy_plan=legacy or _legacy_plan(),
        speculative_intent=intent,
        auto_enabled=auto_enabled,
        artifact=artifact or _artifact(),
        qualification_rows=rows if rows is not None else (_row(),),
        operator_target_lane=operator_lane,
    )


def test_behavior_neutral_defaults_return_exact_legacy_object() -> None:
    legacy = _legacy_plan()

    result = resolve_qwen_runtime_plan(
        legacy_plan=legacy,
        speculative_intent=SpeculativeIntent.ALIAS_DEFAULT,
    )

    assert result is legacy


def test_legacy_default_reason_is_available_without_auto_metadata() -> None:
    legacy = QwenRuntimePlan(
        target_lane=TargetLane.VISION,
        text_mode=TextMode.NONE,
        selection_source=SelectionSource.FALLBACK,
        qualification_id=None,
        receipt_id=None,
        verified_target=None,
        reason=PlanReason.LEGACY_DEFAULT,
        media_enabled=True,
    )

    assert _resolve(legacy=legacy, auto_enabled=False) is legacy


def test_auto_disabled_ignores_a_complete_qualification() -> None:
    legacy = _legacy_plan()
    assert _resolve(legacy=legacy, auto_enabled=False) is legacy


@pytest.mark.parametrize("method", ["mtp", "dflash", "ddtree", "suffix"])
def test_every_explicit_enabled_method_preserves_legacy_text_semantics(
    method: str,
) -> None:
    del method  # Method-specific normalization stays outside this closed planner.
    legacy = replace(
        _legacy_plan(),
        selection_source=SelectionSource.OPERATOR,
        reason=PlanReason.LEGACY_OPERATOR,
    )

    result = _resolve(legacy=legacy, intent=SpeculativeIntent.EXPLICIT_ENABLED)

    assert result is legacy


def test_explicit_target_lane_preserves_legacy_plan() -> None:
    legacy = _legacy_plan()
    assert _resolve(legacy=legacy, operator_lane=TargetLane.TEXT) is legacy


def test_exact_mtp_mode_selects_its_own_receipt_and_native_fallback() -> None:
    result = _resolve()

    assert result.text_mode is TextMode.MTP
    assert result.selection_source is SelectionSource.QUALIFIED_AUTO
    assert result.receipt_id == "receipt/mtp-v1"
    assert result.fallback_chain == (
        QwenFallbackTarget(TextMode.NATIVE_AR, "receipt/native-v1"),
        QwenFallbackTarget(TextMode.NONE, None),
    )


def test_explicit_no_spec_selects_native_without_any_drafter() -> None:
    native_evidence = (QwenModeEvidence(TextMode.NATIVE_AR, ("native-cache-api-v1",)),)

    result = _resolve(
        intent=SpeculativeIntent.EXPLICIT_DISABLED,
        artifact=_artifact(evidence=native_evidence),
    )

    assert result.text_mode is TextMode.NATIVE_AR
    assert result.selection_source is SelectionSource.OPERATOR
    assert result.reason is PlanReason.QUALIFIED_AUTO_NO_SPEC
    assert result.receipt_id == "receipt/native-v1"
    assert result.fallback_chain == (QwenFallbackTarget(TextMode.NONE, None),)


def test_native_only_row_qualifies_without_any_drafter_contract() -> None:
    row = QwenQualificationRow(
        qualification_id="qwen-exact-native-v1",
        public_alias="qwen-exact-4bit",
        target_identity=_target(),
        expected_target_verification_id=TARGET_VERIFICATION_ID,
        expected_target_verification_authority=TARGET_VERIFICATION_AUTHORITY,
        mode_qualifications=(_native_qualification(),),
        preferred_text_mode=TextMode.NATIVE_AR,
        fallback_chain=(TextMode.NONE,),
        runtime_versions=RUNTIME_VERSIONS,
        hardware_classes=("m4-pro-48gb",),
    )
    artifact = _artifact(
        evidence=(QwenModeEvidence(TextMode.NATIVE_AR, ("native-cache-api-v1",)),)
    )

    result = _resolve(
        intent=SpeculativeIntent.NONE,
        artifact=artifact,
        rows=(row,),
    )

    assert result.text_mode is TextMode.NATIVE_AR
    assert result.receipt_id == "receipt/native-v1"


def test_mtp_cannot_borrow_native_probe_evidence() -> None:
    evidence = (
        QwenModeEvidence(
            TextMode.NATIVE_AR,
            ("mtp-injector-v1", "mtp-install-v1", "native-cache-api-v1"),
        ),
    )

    result = _resolve(artifact=_artifact(evidence=evidence))

    assert result.text_mode is TextMode.NATIVE_AR
    assert result.selection_source is SelectionSource.FALLBACK
    assert result.reason is PlanReason.MODE_EVIDENCE_MISSING
    assert result.receipt_id == "receipt/native-v1"


def test_mtp_requires_all_of_its_own_probes_before_native_fallback() -> None:
    evidence = (
        QwenModeEvidence(TextMode.NATIVE_AR, ("native-cache-api-v1",)),
        QwenModeEvidence(TextMode.MTP, ("mtp-injector-v1",), _drafter()),
    )

    result = _resolve(artifact=_artifact(evidence=evidence))

    assert result.text_mode is TextMode.NATIVE_AR
    assert result.reason is PlanReason.MODE_PROBE_FAILED


def test_mtp_drafter_path_mismatch_falls_back_to_independent_native() -> None:
    evidence = (
        QwenModeEvidence(TextMode.NATIVE_AR, ("native-cache-api-v1",)),
        QwenModeEvidence(
            TextMode.MTP,
            ("mtp-injector-v1", "mtp-install-v1"),
            _drafter(path="another-head/model.safetensors"),
        ),
    )

    result = _resolve(artifact=_artifact(evidence=evidence))

    assert result.text_mode is TextMode.NATIVE_AR
    assert result.reason is PlanReason.MODE_DRAFTER_IDENTITY_MISMATCH


def test_mtp_drafter_content_mismatch_falls_back_to_independent_native() -> None:
    evidence = (
        QwenModeEvidence(TextMode.NATIVE_AR, ("native-cache-api-v1",)),
        QwenModeEvidence(
            TextMode.MTP,
            ("mtp-injector-v1", "mtp-install-v1"),
            _drafter(artifact_verification_id="hf_blob:" + "d" * 64),
        ),
    )

    result = _resolve(artifact=_artifact(evidence=evidence))

    assert result.text_mode is TextMode.NATIVE_AR
    assert result.reason is PlanReason.MODE_DRAFTER_IDENTITY_MISMATCH


@pytest.mark.parametrize(
    "artifact_verification_id",
    ["", "main", "sha256:" + "c" * 64, "hf_blob:not-a-digest", "hf_blob:" + "c" * 39],
)
def test_drafter_requires_immutable_artifact_verification_id(
    artifact_verification_id: str,
) -> None:
    with pytest.raises(
        ValueError, match="artifact_verification_id must be an immutable"
    ):
        _drafter(artifact_verification_id=artifact_verification_id)


def test_failed_native_fallback_is_not_published_in_mtp_chain() -> None:
    evidence = (
        QwenModeEvidence(TextMode.NATIVE_AR, ("wrong-native-probe",)),
        QwenModeEvidence(
            TextMode.MTP,
            ("mtp-injector-v1", "mtp-install-v1"),
            _drafter(),
        ),
    )

    result = _resolve(artifact=_artifact(evidence=evidence))

    assert result.text_mode is TextMode.MTP
    assert result.fallback_chain == (QwenFallbackTarget(TextMode.NONE, None),)


def test_syntax_only_target_cannot_enter_resolved_artifact() -> None:
    with pytest.raises(ValueError, match="VerifiedQwenTarget"):
        replace(_artifact(), verified_target=_target())


def test_verified_binding_receipt_must_be_non_empty() -> None:
    with pytest.raises(ValueError, match="verification_id must be non-empty"):
        _verified_target(verification_id="")


def test_verified_target_cannot_be_constructed_outside_private_mint_seam() -> None:
    with pytest.raises(TypeError, match="_mint_token"):
        qwen_plan.VerifiedQwenTarget(
            identity=_target(),
            verification_id="forged",
            verification_authority="forged",
        )
    assert "VerifiedQwenTarget" not in qwen_plan.__all__


def test_alias_cannot_borrow_another_alias_exact_target() -> None:
    result = _resolve(artifact=_artifact(public_alias="other-alias"))
    assert result.reason is PlanReason.QUALIFICATION_ALIAS_MISMATCH


def test_raw_path_can_match_only_the_complete_verified_target_identity() -> None:
    result = _resolve(artifact=_artifact(public_alias=None))
    assert result.selection_source is SelectionSource.QUALIFIED_AUTO


def test_same_alias_with_different_target_identity_fails_closed() -> None:
    target = replace(_target(), target_revision="3" * 40)
    result = _resolve(artifact=_artifact(target=target))
    assert result.reason is PlanReason.QUALIFICATION_TARGET_IDENTITY_MISMATCH


def test_same_identity_different_verification_receipt_fails_closed() -> None:
    artifact = _artifact(
        verified_target=_verified_target(
            verification_id="hf-snapshot-sha256:" + "b" * 64
        )
    )

    result = _resolve(artifact=artifact)

    assert result.reason is PlanReason.QUALIFICATION_TARGET_RECEIPT_MISMATCH


def test_exact_receipt_disambiguates_rows_with_same_projected_identity() -> None:
    rows = (
        _row(qualification_id="receipt-a"),
        _row(
            qualification_id="receipt-b",
            expected_verification_id="hf-snapshot-sha256:" + "b" * 64,
        ),
    )

    result = _resolve(rows=rows)

    assert result.qualification_id == "receipt-a"
    assert result.selection_source is SelectionSource.QUALIFIED_AUTO


def test_same_verification_id_from_different_authority_fails_closed() -> None:
    artifact = _artifact(
        verified_target=_verified_target(authority="another-artifact-authority-v1")
    )

    result = _resolve(artifact=artifact)

    assert result.reason is PlanReason.QUALIFICATION_TARGET_RECEIPT_MISMATCH


@pytest.mark.parametrize(
    "verification_id",
    ["", "main", "hf-snapshot-sha256:not-a-digest", "a" * 40],
)
def test_qualification_row_requires_immutable_target_verification_receipt(
    verification_id: str,
) -> None:
    with pytest.raises(
        ValueError, match="expected_target_verification_id must be an immutable"
    ):
        _row(expected_verification_id=verification_id)


def test_qualification_row_requires_verification_authority() -> None:
    with pytest.raises(
        ValueError, match="expected_target_verification_authority must be non-empty"
    ):
        _row(expected_verification_authority="")


def test_multiple_exact_raw_rows_are_ambiguous() -> None:
    rows = (
        _row(qualification_id="row-a", public_alias="alias-a"),
        _row(qualification_id="row-b", public_alias="alias-b"),
    )
    result = _resolve(artifact=_artifact(public_alias=None), rows=rows)
    assert result.reason is PlanReason.QUALIFICATION_AMBIGUOUS


@pytest.mark.parametrize(
    ("artifact", "reason"),
    [
        (
            _artifact(
                runtime_versions=(
                    ("mlx", "0.32.2"),
                    ("rapid-mlx", "0.15.0"),
                    ("vendored-mllm-abi", "3b"),
                )
            ),
            PlanReason.QUALIFICATION_RUNTIME_MISMATCH,
        ),
        (
            _artifact(hardware_class="m3-ultra-256gb"),
            PlanReason.QUALIFICATION_HARDWARE_MISMATCH,
        ),
    ],
)
def test_runtime_and_hardware_gates_fail_closed(
    artifact: ResolvedQwenArtifact, reason: PlanReason
) -> None:
    assert _resolve(artifact=artifact).reason is reason


def test_mutated_install_failure_requires_reload_to_explicit_next_target() -> None:
    plan = _resolve()

    failed = resolve_qwen_install_failure(plan, target_was_mutated=True)

    assert failed.text_mode is TextMode.MTP
    assert failed.receipt_id == "receipt/mtp-v1"
    assert failed.recovery_action is RecoveryAction.RELOAD_REQUIRED
    assert failed.pending_fallback_target == QwenFallbackTarget(
        TextMode.NATIVE_AR, "receipt/native-v1"
    )
    assert failed.fallback_chain == (QwenFallbackTarget(TextMode.NONE, None),)


def test_repeated_install_failure_is_rejected_while_reload_is_pending() -> None:
    pending = resolve_qwen_install_failure(_resolve(), target_was_mutated=True)

    with pytest.raises(ValueError, match="recovery action is pending"):
        resolve_qwen_install_failure(pending, target_was_mutated=False)


def test_clean_reload_only_allows_fallback_start_without_publishing_it() -> None:
    pending = resolve_qwen_install_failure(_resolve(), target_was_mutated=True)

    ready = complete_qwen_reload(pending, reloaded_target=_verified_target())

    assert ready.text_mode is TextMode.MTP
    assert ready.receipt_id == "receipt/mtp-v1"
    assert ready.recovery_action is RecoveryAction.STARTUP_FALLBACK
    assert ready.pending_fallback_target == QwenFallbackTarget(
        TextMode.NATIVE_AR, "receipt/native-v1"
    )
    assert ready.reason is PlanReason.INSTALL_FAILED_STARTUP_FALLBACK

    active = complete_qwen_fallback_start(ready)

    assert active.text_mode is TextMode.NATIVE_AR
    assert active.receipt_id == "receipt/native-v1"
    assert active.recovery_action is RecoveryAction.NONE
    assert active.pending_fallback_target is None
    assert active.reason is PlanReason.INSTALL_FALLBACK_ACTIVE


def test_reload_recovery_rejects_absent_stale_or_mismatched_binding() -> None:
    pending = resolve_qwen_install_failure(_resolve(), target_was_mutated=True)

    with pytest.raises(TypeError, match="reloaded_target"):
        complete_qwen_reload(pending)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="fresh verification binding"):
        complete_qwen_reload(
            pending,
            reloaded_target=pending.verified_target,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="mismatched"):
        complete_qwen_reload(
            pending,
            reloaded_target=_verified_target(
                replace(_target(), target_revision="3" * 40)
            ),
        )
    with pytest.raises(ValueError, match="mismatched"):
        complete_qwen_reload(
            pending,
            reloaded_target=_verified_target(verification_id="another-binding"),
        )
    with pytest.raises(ValueError, match="mismatched"):
        complete_qwen_reload(
            pending,
            reloaded_target=_verified_target(authority="another-authority"),
        )


def test_unmutated_install_failure_requires_starting_next_fallback() -> None:
    pending = resolve_qwen_install_failure(_resolve(), target_was_mutated=False)

    assert pending.text_mode is TextMode.MTP
    assert pending.receipt_id == "receipt/mtp-v1"
    assert pending.recovery_action is RecoveryAction.STARTUP_FALLBACK
    assert pending.pending_fallback_target == QwenFallbackTarget(
        TextMode.NATIVE_AR, "receipt/native-v1"
    )
    with pytest.raises(ValueError, match="recovery action is pending"):
        resolve_qwen_install_failure(pending, target_was_mutated=False)


def test_unmutated_fallback_failure_advances_through_terminal_none() -> None:
    native_pending = resolve_qwen_install_failure(_resolve(), target_was_mutated=False)

    none_pending = resolve_qwen_fallback_start_failure(
        native_pending, target_was_mutated=False
    )

    assert none_pending.text_mode is TextMode.MTP
    assert none_pending.receipt_id == "receipt/mtp-v1"
    assert none_pending.pending_fallback_target == QwenFallbackTarget(
        TextMode.NONE, None
    )
    assert none_pending.fallback_chain == ()
    active = complete_qwen_fallback_start(none_pending)
    assert active.text_mode is TextMode.NONE
    assert active.receipt_id is None
    assert active.recovery_action is RecoveryAction.NONE


def test_post_reload_fallback_failure_requires_another_reload_to_none() -> None:
    initial = resolve_qwen_install_failure(_resolve(), target_was_mutated=True)
    native_ready = complete_qwen_reload(initial, reloaded_target=_verified_target())

    none_reload = resolve_qwen_fallback_start_failure(
        native_ready, target_was_mutated=True
    )

    assert none_reload.text_mode is TextMode.MTP
    assert none_reload.recovery_action is RecoveryAction.RELOAD_REQUIRED
    assert none_reload.pending_fallback_target == QwenFallbackTarget(
        TextMode.NONE, None
    )
    none_ready = complete_qwen_reload(none_reload, reloaded_target=_verified_target())
    assert none_ready.text_mode is TextMode.MTP
    assert none_ready.recovery_action is RecoveryAction.STARTUP_FALLBACK
    active = complete_qwen_fallback_start(none_ready)
    assert active.text_mode is TextMode.NONE
    assert active.receipt_id is None


def test_status_payload_is_json_safe_and_contains_selected_receipt() -> None:
    status = _resolve().to_status_dict()

    assert status["receipt_id"] == "receipt/mtp-v1"
    assert status["fallback_chain"] == [
        {"text_mode": "native_ar", "receipt_id": "receipt/native-v1"},
        {"text_mode": "none", "receipt_id": None},
    ]
    json.dumps(status)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: replace(_target(), cache_geometry=()),
        lambda: replace(_artifact(), runtime_versions=()),
        lambda: replace(_row(), runtime_versions=()),
        lambda: replace(_native_qualification(), required_probes=()),
        lambda: QwenModeEvidence(TextMode.NATIVE_AR, ()),
    ],
)
def test_empty_qualification_contracts_are_rejected(factory) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        factory()


def test_native_qualification_rejects_drafter_identity() -> None:
    with pytest.raises(ValueError, match="must not carry a drafter"):
        replace(_native_qualification(), drafter_identity=_drafter())


def test_mtp_qualification_requires_exact_drafter_identity() -> None:
    with pytest.raises(ValueError, match="exact drafter identity"):
        replace(_mtp_qualification(), drafter_identity=None)


@pytest.mark.parametrize("revision", ["", "main", "latest", "v1.0"])
def test_mutable_revisions_are_rejected(revision: str) -> None:
    with pytest.raises(ValueError, match="immutable commit or content digest"):
        replace(_target(), target_revision=revision)


def test_public_contracts_are_immutable() -> None:
    plan = _resolve()
    with pytest.raises(FrozenInstanceError):
        plan.text_mode = TextMode.NATIVE_AR  # type: ignore[misc]
