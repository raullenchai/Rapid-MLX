# SPDX-License-Identifier: Apache-2.0
"""Hermetic contract tests for the Qwen auto-runtime planner."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from rapid_mlx.qwen_runtime_plan import (
    PlanReason,
    QwenArtifactIdentity,
    QwenQualificationRow,
    QwenRuntimePlan,
    RecoveryAction,
    ResolvedQwenArtifact,
    SelectionSource,
    SpeculativeInput,
    TargetLane,
    TextMode,
    resolve_qwen_install_failure,
    resolve_qwen_runtime_plan,
)

TARGET_REVISION = "1" * 40
DRAFTER_REVISION = "2" * 40
RUNTIME_VERSIONS = (
    ("mlx", "0.32.1"),
    ("rapid-mlx", "0.15.0"),
    ("vendored-mllm-abi", "3b"),
)


def _identity(*, with_drafter: bool = True) -> QwenArtifactIdentity:
    return QwenArtifactIdentity(
        target_repo="example/Qwen-Exact-4bit",
        target_revision=TARGET_REVISION,
        target_subfolder=None,
        drafter_repo="example/Qwen-Exact-MTP-4bit" if with_drafter else None,
        drafter_revision=DRAFTER_REVISION if with_drafter else None,
        drafter_artifact_path=("mtp/model.safetensors" if with_drafter else None),
        outer_model_type="qwen3_5_moe",
        language_model_type="qwen3_5_moe_text",
        quantization="group-size=64;bits=4",
        weight_layout="safetensors-sharded-v1",
        layer_layout=("linear", "linear", "full_attention"),
        cache_geometry=(("key_head_dim", "128"), ("value_heads", "32")),
    )


def _artifact(
    *,
    public_alias: str | None = "qwen-exact-4bit",
    identity: QwenArtifactIdentity | None = None,
    runtime_versions: tuple[tuple[str, str], ...] = RUNTIME_VERSIONS,
    hardware_class: str = "m4-pro-48gb",
    passed_probes: tuple[str, ...] = ("cache-api-v1", "injector-v1"),
) -> ResolvedQwenArtifact:
    return ResolvedQwenArtifact(
        identity=identity or _identity(),
        public_alias=public_alias,
        runtime_versions=runtime_versions,
        hardware_class=hardware_class,
        passed_probes=passed_probes,
    )


def _row(
    *,
    qualification_id: str = "qwen-exact-mtp-v1",
    public_alias: str = "qwen-exact-4bit",
    identity: QwenArtifactIdentity | None = None,
    runtime_versions: tuple[tuple[str, str], ...] = RUNTIME_VERSIONS,
    hardware_classes: tuple[str, ...] = ("m1-max-64gb", "m4-pro-48gb"),
    required_probes: tuple[str, ...] = ("cache-api-v1", "injector-v1"),
) -> QwenQualificationRow:
    return QwenQualificationRow(
        qualification_id=qualification_id,
        public_alias=public_alias,
        identity=identity or _identity(),
        supported_text_modes=(TextMode.NATIVE_AR, TextMode.MTP),
        preferred_text_mode=TextMode.MTP,
        fallback_chain=(TextMode.NATIVE_AR, TextMode.NONE),
        runtime_versions=runtime_versions,
        hardware_classes=hardware_classes,
        required_probes=required_probes,
        supported_sampling=("greedy", "seeded"),
        supported_logits_processors=("grammar", "tools"),
        supported_kv_configurations=("bf16",),
        tools_supported=True,
        receipt_id="receipt/qwen-exact-mtp-v1",
    )


def _alias_mtp_legacy_plan() -> QwenRuntimePlan:
    return QwenRuntimePlan(
        target_lane=TargetLane.TEXT,
        text_mode=TextMode.MTP,
        selection_source=SelectionSource.ALIAS_DEFAULT,
        qualification_id=None,
        reason=PlanReason.LEGACY_ALIAS_DEFAULT,
        media_enabled=False,
    )


def _vision_legacy_plan() -> QwenRuntimePlan:
    return QwenRuntimePlan(
        target_lane=TargetLane.VISION,
        text_mode=TextMode.NONE,
        selection_source=SelectionSource.FALLBACK,
        qualification_id=None,
        reason=PlanReason.LEGACY_DEFAULT,
        media_enabled=True,
    )


def _resolve(
    *,
    legacy_plan: QwenRuntimePlan | None = None,
    speculative: SpeculativeInput | None = None,
    artifact: ResolvedQwenArtifact | None = None,
    rows: tuple[QwenQualificationRow, ...] | None = None,
    auto_enabled: bool = True,
    operator_target_lane: TargetLane | None = None,
) -> QwenRuntimePlan:
    return resolve_qwen_runtime_plan(
        legacy_plan=legacy_plan or _alias_mtp_legacy_plan(),
        speculative=speculative or SpeculativeInput(TextMode.MTP, explicit=False),
        auto_enabled=auto_enabled,
        artifact=artifact or _artifact(),
        qualification_rows=rows if rows is not None else (_row(),),
        operator_target_lane=operator_target_lane,
    )


def test_b0_defaults_return_the_exact_legacy_plan_object() -> None:
    legacy = _alias_mtp_legacy_plan()

    result = resolve_qwen_runtime_plan(
        legacy_plan=legacy,
        speculative=SpeculativeInput(TextMode.MTP, explicit=False),
    )

    assert result is legacy
    assert result.target_lane is TargetLane.TEXT
    assert result.text_mode is TextMode.MTP
    assert result.selection_source is SelectionSource.ALIAS_DEFAULT
    assert result.media_enabled is False


def test_auto_disabled_ignores_even_an_exact_production_shaped_row() -> None:
    legacy = _alias_mtp_legacy_plan()

    result = _resolve(legacy_plan=legacy, auto_enabled=False)

    assert result is legacy


def test_exact_qualified_alias_selects_dual_mtp_plan() -> None:
    result = _resolve()

    assert result == QwenRuntimePlan(
        target_lane=TargetLane.VISION,
        text_mode=TextMode.MTP,
        selection_source=SelectionSource.QUALIFIED_AUTO,
        qualification_id="qwen-exact-mtp-v1",
        reason=PlanReason.QUALIFIED_AUTO,
        media_enabled=True,
        fallback_chain=(TextMode.NATIVE_AR, TextMode.NONE),
    )


def test_raw_artifact_may_match_only_by_complete_exact_identity() -> None:
    result = _resolve(artifact=_artifact(public_alias=None))

    assert result.selection_source is SelectionSource.QUALIFIED_AUTO
    assert result.qualification_id == "qwen-exact-mtp-v1"


def test_explicit_no_spec_can_select_qualified_native_ar() -> None:
    result = _resolve(
        legacy_plan=_vision_legacy_plan(),
        speculative=SpeculativeInput(TextMode.NONE, explicit=True),
    )

    assert result.target_lane is TargetLane.VISION
    assert result.text_mode is TextMode.NATIVE_AR
    assert result.selection_source is SelectionSource.OPERATOR
    assert result.reason is PlanReason.QUALIFIED_AUTO_NO_SPEC
    assert result.qualification_id == "qwen-exact-mtp-v1"
    assert result.fallback_chain == (TextMode.NONE,)


def test_explicit_speculation_preserves_current_text_only_semantics() -> None:
    legacy = replace(
        _alias_mtp_legacy_plan(),
        selection_source=SelectionSource.OPERATOR,
        reason=PlanReason.LEGACY_OPERATOR,
    )

    result = _resolve(
        legacy_plan=legacy,
        speculative=SpeculativeInput(TextMode.MTP, explicit=True),
    )

    assert result is legacy


def test_explicit_target_lane_preserves_the_exact_legacy_plan() -> None:
    legacy = _vision_legacy_plan()

    result = _resolve(
        legacy_plan=legacy,
        operator_target_lane=TargetLane.VISION,
    )

    assert result is legacy


def test_empty_row_set_fails_closed_without_changing_runtime_behavior() -> None:
    legacy = _alias_mtp_legacy_plan()

    result = _resolve(legacy_plan=legacy, rows=())

    assert result.target_lane is legacy.target_lane
    assert result.text_mode is legacy.text_mode
    assert result.media_enabled is legacy.media_enabled
    assert result.selection_source is SelectionSource.FALLBACK
    assert result.reason is PlanReason.QUALIFICATION_NOT_FOUND
    assert result.recovery_action is RecoveryAction.NONE


def test_exact_identity_cannot_borrow_another_public_alias_row() -> None:
    legacy = _alias_mtp_legacy_plan()

    result = _resolve(
        legacy_plan=legacy,
        artifact=_artifact(public_alias="qwen-other-4bit"),
    )

    assert result.target_lane is legacy.target_lane
    assert result.text_mode is legacy.text_mode
    assert result.reason is PlanReason.QUALIFICATION_ALIAS_MISMATCH


def test_matching_alias_with_different_revision_fails_identity_gate() -> None:
    other = replace(_identity(), target_revision="3" * 40)

    result = _resolve(artifact=_artifact(identity=other))

    assert result.selection_source is SelectionSource.FALLBACK
    assert result.reason is PlanReason.QUALIFICATION_IDENTITY_MISMATCH


def test_same_repo_and_revision_cannot_match_the_wrong_nested_drafter() -> None:
    wrong_head = replace(
        _identity(), drafter_artifact_path="another-head/model.safetensors"
    )

    result = _resolve(artifact=_artifact(identity=wrong_head))

    assert result.selection_source is SelectionSource.FALLBACK
    assert result.reason is PlanReason.QUALIFICATION_IDENTITY_MISMATCH


def test_model_family_or_repository_name_never_confers_eligibility() -> None:
    qwen_named_but_unknown = replace(
        _identity(),
        target_repo="example/Qwen-Same-Family-4bit",
        target_revision="3" * 40,
    )

    result = _resolve(
        artifact=_artifact(public_alias=None, identity=qwen_named_but_unknown)
    )

    assert result.reason is PlanReason.QUALIFICATION_NOT_FOUND
    assert result.qualification_id is None


def test_multiple_exact_raw_identity_rows_are_ambiguous() -> None:
    rows = (
        _row(qualification_id="receipt-a", public_alias="alias-a"),
        _row(qualification_id="receipt-b", public_alias="alias-b"),
    )

    result = _resolve(artifact=_artifact(public_alias=None), rows=rows)

    assert result.reason is PlanReason.QUALIFICATION_AMBIGUOUS
    assert result.selection_source is SelectionSource.FALLBACK


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
        (
            _artifact(passed_probes=("cache-api-v1",)),
            PlanReason.QUALIFICATION_PROBE_FAILED,
        ),
    ],
)
def test_runtime_hardware_and_probe_gates_fail_closed(
    artifact: ResolvedQwenArtifact, reason: PlanReason
) -> None:
    result = _resolve(artifact=artifact)

    assert result.reason is reason
    assert result.selection_source is SelectionSource.FALLBACK


def test_mtp_row_requires_pinned_drafter_identity() -> None:
    with pytest.raises(ValueError, match="immutable drafter identity"):
        _row(identity=_identity(with_drafter=False))


def test_qualification_fallback_chain_must_end_at_vision_only() -> None:
    with pytest.raises(ValueError, match="terminate at vision-only NONE"):
        replace(_row(), fallback_chain=(TextMode.NATIVE_AR,))


def test_qualification_fallback_chain_cannot_upgrade_decoder_complexity() -> None:
    with pytest.raises(ValueError, match="simpler text modes"):
        replace(
            _row(),
            preferred_text_mode=TextMode.NATIVE_AR,
            fallback_chain=(TextMode.MTP, TextMode.NONE),
        )


@pytest.mark.parametrize("revision", ["", "main", "latest", "v1.0"])
def test_mutable_target_revision_is_rejected(revision: str) -> None:
    with pytest.raises(ValueError, match="immutable commit or content digest"):
        replace(_identity(), target_revision=revision)


def test_repo_only_drafter_identity_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be provided together"):
        replace(_identity(), drafter_revision=None)


@pytest.mark.parametrize(
    "path",
    [
        "",
        "/mtp/model.safetensors",
        "mtp/../model.safetensors",
        "mtp\\model.safetensors",
    ],
)
def test_drafter_artifact_path_must_be_canonical_and_relative(path: str) -> None:
    with pytest.raises(ValueError, match="canonical relative POSIX path"):
        replace(_identity(), drafter_artifact_path=path)


def test_install_failure_before_mutation_advances_startup_fallback() -> None:
    plan = _resolve()

    fallback = resolve_qwen_install_failure(plan, target_was_mutated=False)

    assert fallback.text_mode is TextMode.NATIVE_AR
    assert fallback.fallback_chain == (TextMode.NONE,)
    assert fallback.selection_source is SelectionSource.FALLBACK
    assert fallback.reason is PlanReason.INSTALL_FAILED_STARTUP_FALLBACK
    assert fallback.recovery_action is RecoveryAction.STARTUP_FALLBACK


def test_install_failure_transition_rejects_an_unqualified_legacy_plan() -> None:
    with pytest.raises(ValueError, match="qualified vision plan"):
        resolve_qwen_install_failure(_alias_mtp_legacy_plan(), target_was_mutated=False)


def test_install_failure_after_mutation_requires_reload_without_fallback_chain() -> (
    None
):
    plan = _resolve()

    failed = resolve_qwen_install_failure(plan, target_was_mutated=True)

    assert failed.text_mode is TextMode.MTP
    assert failed.fallback_chain == ()
    assert failed.selection_source is SelectionSource.FALLBACK
    assert failed.reason is PlanReason.INSTALL_FAILED_RELOAD_REQUIRED
    assert failed.recovery_action is RecoveryAction.RELOAD_REQUIRED


def test_reload_required_cannot_be_mislabelled_as_same_instance_fallback() -> None:
    with pytest.raises(ValueError, match="must require reload"):
        replace(
            _resolve(),
            reason=PlanReason.INSTALL_FAILED_RELOAD_REQUIRED,
            recovery_action=RecoveryAction.STARTUP_FALLBACK,
        )


def test_public_contracts_are_immutable() -> None:
    plan = _resolve()

    with pytest.raises(FrozenInstanceError):
        plan.text_mode = TextMode.NATIVE_AR  # type: ignore[misc]
