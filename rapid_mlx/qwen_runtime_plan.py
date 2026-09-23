# SPDX-License-Identifier: Apache-2.0
"""Pure, fail-closed planning primitives for the Qwen auto runtime.

This module has no model, MLX, registry, or filesystem access. Callers resolve
the current serving decision and separately supply binding-verified immutable
artifact facts. B0 registers no production rows and auto selection is disabled
by default, so its default call returns the exact legacy plan object.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import PurePosixPath


class SelectionSource(str, Enum):
    OPERATOR = "operator"
    ALIAS_DEFAULT = "alias_default"
    QUALIFIED_AUTO = "qualified_auto"
    FALLBACK = "fallback"


class TargetLane(str, Enum):
    VISION = "vision"
    TEXT = "text"


class TextMode(str, Enum):
    NONE = "none"
    NATIVE_AR = "native_ar"
    MTP = "mtp"


class SpeculativeIntent(str, Enum):
    """Speculative-input provenance captured before legacy normalization.

    Every explicit enabled method (MTP, DFlash, DDTree, suffix, and future
    methods) maps to ``EXPLICIT_ENABLED``. The planner therefore preserves its
    current text-only behavior without pretending the method is an output
    ``TextMode``. ``EXPLICIT_DISABLED`` represents an operator no-spec request.
    """

    NONE = "none"
    ALIAS_DEFAULT = "alias_default"
    EXPLICIT_DISABLED = "explicit_disabled"
    EXPLICIT_ENABLED = "explicit_enabled"


class PlanReason(str, Enum):
    LEGACY_OPERATOR = "legacy_operator"
    LEGACY_ALIAS_DEFAULT = "legacy_alias_default"
    LEGACY_DEFAULT = "legacy_default"
    QUALIFIED_AUTO = "qualified_auto"
    QUALIFIED_AUTO_NO_SPEC = "qualified_auto_no_spec"
    QUALIFICATION_NOT_FOUND = "qualification_not_found"
    QUALIFICATION_ALIAS_MISMATCH = "qualification_alias_mismatch"
    QUALIFICATION_TARGET_IDENTITY_MISMATCH = "qualification_target_identity_mismatch"
    QUALIFICATION_RUNTIME_MISMATCH = "qualification_runtime_mismatch"
    QUALIFICATION_HARDWARE_MISMATCH = "qualification_hardware_mismatch"
    QUALIFICATION_AMBIGUOUS = "qualification_ambiguous"
    MODE_NOT_QUALIFIED = "mode_not_qualified"
    MODE_EVIDENCE_MISSING = "mode_evidence_missing"
    MODE_DRAFTER_IDENTITY_MISMATCH = "mode_drafter_identity_mismatch"
    MODE_PROBE_FAILED = "mode_probe_failed"
    INSTALL_FAILED_STARTUP_FALLBACK = "install_failed_startup_fallback"
    INSTALL_FAILED_RELOAD_REQUIRED = "install_failed_reload_required"
    INSTALL_FALLBACK_ACTIVE = "install_fallback_active"


class RecoveryAction(str, Enum):
    NONE = "none"
    STARTUP_FALLBACK = "startup_fallback"
    RELOAD_REQUIRED = "reload_required"


def _require_non_empty(label: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty")


def _require_immutable_revision(label: str, value: object) -> None:
    if isinstance(value, str) and re.fullmatch(
        r"(?:[0-9a-f]{40}|[0-9a-f]{64}|sha256:[0-9a-f]{64})", value
    ):
        return
    raise ValueError(f"{label} must be an immutable commit or content digest")


def _require_relative_path(label: str, value: object) -> None:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a canonical relative POSIX path")
    path = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or path.is_absolute()
        or path.as_posix() != value
        or any(part in ("", ".", "..") for part in path.parts)
    ):
        raise ValueError(f"{label} must be a canonical relative POSIX path")


def _require_string_tuple(label: str, values: object) -> None:
    if (
        not isinstance(values, tuple)
        or not values
        or any(not isinstance(value, str) or not value.strip() for value in values)
    ):
        raise ValueError(f"{label} must be a non-empty tuple of non-empty strings")
    if len(set(values)) != len(values):
        raise ValueError(f"{label} values must be unique")
    if values != tuple(sorted(values)):
        raise ValueError(f"{label} must be sorted")


def _require_canonical_pairs(label: str, values: object) -> None:
    if (
        not isinstance(values, tuple)
        or not values
        or any(
            not isinstance(item, tuple)
            or len(item) != 2
            or not all(isinstance(part, str) and part.strip() for part in item)
            for item in values
        )
    ):
        raise ValueError(f"{label} must be a non-empty tuple of string pairs")
    keys = tuple(key for key, _value in values)
    if len(set(keys)) != len(keys):
        raise ValueError(f"{label} keys must be unique")
    if keys != tuple(sorted(keys)):
        raise ValueError(f"{label} must be sorted by key")


@dataclass(frozen=True, slots=True)
class QwenTargetIdentity:
    """Target-only immutable artifact and ordered loaded geometry.

    This is the narrow conversion seam for the truth layer. Drafter identity is
    intentionally absent: native AR qualifies this target without a sidecar;
    each speculative mode owns its exact drafter facts independently.
    """

    target_repo: str
    target_revision: str
    outer_model_type: str
    language_model_type: str
    quantization: str
    weight_layout: str
    layer_layout: tuple[str, ...]
    cache_geometry: tuple[tuple[str, str], ...]
    target_subfolder: str | None = None

    def __post_init__(self) -> None:
        for label in (
            "target_repo",
            "outer_model_type",
            "language_model_type",
            "quantization",
            "weight_layout",
        ):
            _require_non_empty(label, getattr(self, label))
        _require_immutable_revision("target_revision", self.target_revision)
        if self.target_subfolder is not None:
            _require_relative_path("target_subfolder", self.target_subfolder)
        if (
            not isinstance(self.layer_layout, tuple)
            or not self.layer_layout
            or any(
                not isinstance(layer, str) or not layer.strip()
                for layer in self.layer_layout
            )
        ):
            raise ValueError("layer_layout must contain non-empty layer kinds")
        _require_canonical_pairs("cache_geometry", self.cache_geometry)


_VERIFIED_TARGET_MINT_TOKEN = object()


@dataclass(frozen=True, slots=True, init=False)
class VerifiedQwenTarget:
    """Opaque truth-layer binding proving an identity was actually resolved.

    A syntactically valid 40-hex revision is insufficient. The truth layer must
    bind repository, revision, config, and weight-layout evidence first, then
    provide its opaque stable verification receipt here.
    """

    identity: QwenTargetIdentity
    verification_id: str
    verification_authority: str

    def __init__(
        self,
        *,
        identity: QwenTargetIdentity,
        verification_id: str,
        verification_authority: str,
        _mint_token: object,
    ) -> None:
        if _mint_token is not _VERIFIED_TARGET_MINT_TOKEN:
            raise TypeError("VerifiedQwenTarget must be minted by the truth resolver")
        if not isinstance(identity, QwenTargetIdentity):
            raise ValueError("identity must be a QwenTargetIdentity")
        _require_non_empty("verification_id", verification_id)
        _require_non_empty("verification_authority", verification_authority)
        object.__setattr__(self, "identity", identity)
        object.__setattr__(self, "verification_id", verification_id)
        object.__setattr__(self, "verification_authority", verification_authority)


def _mint_verified_qwen_target(
    *,
    identity: QwenTargetIdentity,
    verification_id: str,
    verification_authority: str,
) -> VerifiedQwenTarget:
    """Private composition seam used only by the artifact-truth resolver."""

    return VerifiedQwenTarget(
        identity=identity,
        verification_id=verification_id,
        verification_authority=verification_authority,
        _mint_token=_VERIFIED_TARGET_MINT_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class QwenDrafterIdentity:
    """Exact immutable file identity for one speculative sidecar."""

    repo: str
    revision: str
    artifact_path: str

    def __post_init__(self) -> None:
        _require_non_empty("repo", self.repo)
        _require_immutable_revision("revision", self.revision)
        _require_relative_path("artifact_path", self.artifact_path)


@dataclass(frozen=True, slots=True)
class QwenModeQualification:
    """Mode-specific probes, receipt, and optional exact drafter."""

    mode: TextMode
    required_probes: tuple[str, ...]
    receipt_id: str
    drafter_identity: QwenDrafterIdentity | None = None

    def __post_init__(self) -> None:
        if self.mode not in (TextMode.NATIVE_AR, TextMode.MTP):
            raise ValueError("mode qualification must be native_ar or mtp")
        _require_string_tuple("required_probes", self.required_probes)
        _require_non_empty("receipt_id", self.receipt_id)
        if self.mode is TextMode.NATIVE_AR and self.drafter_identity is not None:
            raise ValueError("native AR qualification must not carry a drafter")
        if self.mode is TextMode.MTP and not isinstance(
            self.drafter_identity, QwenDrafterIdentity
        ):
            raise ValueError("MTP qualification requires an exact drafter identity")


@dataclass(frozen=True, slots=True)
class QwenModeEvidence:
    """Mode-specific load-time probes and observed sidecar identity."""

    mode: TextMode
    passed_probes: tuple[str, ...]
    drafter_identity: QwenDrafterIdentity | None = None

    def __post_init__(self) -> None:
        if self.mode not in (TextMode.NATIVE_AR, TextMode.MTP):
            raise ValueError("mode evidence must be native_ar or mtp")
        _require_string_tuple("passed_probes", self.passed_probes)
        if self.mode is TextMode.NATIVE_AR and self.drafter_identity is not None:
            raise ValueError("native AR evidence must not carry a drafter")
        if self.mode is TextMode.MTP and not isinstance(
            self.drafter_identity, QwenDrafterIdentity
        ):
            raise ValueError("MTP evidence requires an exact drafter identity")


@dataclass(frozen=True, slots=True)
class ResolvedQwenArtifact:
    """Binding-verified target plus mode-specific observed facts."""

    verified_target: VerifiedQwenTarget
    public_alias: str | None
    runtime_versions: tuple[tuple[str, str], ...]
    hardware_class: str
    mode_evidence: tuple[QwenModeEvidence, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.verified_target, VerifiedQwenTarget):
            raise ValueError("verified_target must be a VerifiedQwenTarget")
        if self.public_alias is not None:
            _require_non_empty("public_alias", self.public_alias)
        _require_canonical_pairs("runtime_versions", self.runtime_versions)
        _require_non_empty("hardware_class", self.hardware_class)
        if not isinstance(self.mode_evidence, tuple) or any(
            not isinstance(item, QwenModeEvidence) for item in self.mode_evidence
        ):
            raise ValueError("mode_evidence must contain QwenModeEvidence values")
        modes = tuple(item.mode for item in self.mode_evidence)
        if len(set(modes)) != len(modes):
            raise ValueError("mode_evidence must contain at most one row per mode")


@dataclass(frozen=True, slots=True)
class QwenQualificationRow:
    """One target receipt set; every declared text mode qualifies separately."""

    qualification_id: str
    public_alias: str
    target_identity: QwenTargetIdentity
    mode_qualifications: tuple[QwenModeQualification, ...]
    preferred_text_mode: TextMode
    fallback_chain: tuple[TextMode, ...]
    runtime_versions: tuple[tuple[str, str], ...]
    hardware_classes: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_non_empty("qualification_id", self.qualification_id)
        _require_non_empty("public_alias", self.public_alias)
        if not isinstance(self.target_identity, QwenTargetIdentity):
            raise ValueError("target_identity must be a QwenTargetIdentity")
        if (
            not isinstance(self.mode_qualifications, tuple)
            or not self.mode_qualifications
        ):
            raise ValueError("mode_qualifications must be a non-empty tuple")
        if any(
            not isinstance(item, QwenModeQualification)
            for item in self.mode_qualifications
        ):
            raise ValueError(
                "mode_qualifications must contain QwenModeQualification values"
            )
        qualified_modes = tuple(item.mode for item in self.mode_qualifications)
        if len(set(qualified_modes)) != len(qualified_modes):
            raise ValueError("mode_qualifications must contain one row per mode")
        if self.preferred_text_mode not in qualified_modes:
            raise ValueError("preferred_text_mode must have its own qualification")
        if not isinstance(self.fallback_chain, tuple) or not self.fallback_chain:
            raise ValueError("fallback_chain must be a non-empty tuple")
        if any(not isinstance(mode, TextMode) for mode in self.fallback_chain):
            raise ValueError("fallback_chain must contain TextMode values")
        if len(set(self.fallback_chain)) != len(self.fallback_chain):
            raise ValueError("fallback_chain must not contain duplicates")
        if self.preferred_text_mode in self.fallback_chain:
            raise ValueError("fallback_chain must follow the preferred text mode")
        if self.fallback_chain[-1] is not TextMode.NONE:
            raise ValueError("fallback_chain must terminate at vision-only NONE")
        declared_modes = (self.preferred_text_mode, *self.fallback_chain[:-1])
        if set(declared_modes) != set(qualified_modes):
            raise ValueError("every declared text mode needs exactly one qualification")
        rank = {TextMode.MTP: 2, TextMode.NATIVE_AR: 1, TextMode.NONE: 0}
        ordered = (self.preferred_text_mode, *self.fallback_chain)
        if any(
            rank[current] <= rank[next_mode]
            for current, next_mode in zip(ordered, ordered[1:])
        ):
            raise ValueError("fallback_chain must move toward simpler text modes")
        _require_canonical_pairs("runtime_versions", self.runtime_versions)
        _require_string_tuple("hardware_classes", self.hardware_classes)


@dataclass(frozen=True, slots=True)
class QwenFallbackTarget:
    """One independently qualified fallback and its mode-specific receipt."""

    mode: TextMode
    receipt_id: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.mode, TextMode):
            raise ValueError("mode must be a TextMode")
        if self.mode is TextMode.NONE:
            if self.receipt_id is not None:
                raise ValueError("vision-only NONE must not carry a text receipt")
        else:
            _require_non_empty("receipt_id", self.receipt_id)

    def to_status_dict(self) -> dict[str, str | None]:
        return {"text_mode": self.mode.value, "receipt_id": self.receipt_id}


@dataclass(frozen=True, slots=True)
class QwenRuntimePlan:
    """Immutable, JSON-reportable runtime and monotonic recovery state."""

    target_lane: TargetLane
    text_mode: TextMode
    selection_source: SelectionSource
    qualification_id: str | None
    receipt_id: str | None
    verified_target: VerifiedQwenTarget | None
    reason: PlanReason
    media_enabled: bool
    fallback_chain: tuple[QwenFallbackTarget, ...] = ()
    recovery_action: RecoveryAction = RecoveryAction.NONE
    pending_fallback_target: QwenFallbackTarget | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.target_lane, TargetLane):
            raise ValueError("target_lane must be a TargetLane")
        if not isinstance(self.text_mode, TextMode):
            raise ValueError("text_mode must be a TextMode")
        if not isinstance(self.selection_source, SelectionSource):
            raise ValueError("selection_source must be a SelectionSource")
        if not isinstance(self.reason, PlanReason):
            raise ValueError("reason must be a PlanReason")
        if type(self.media_enabled) is not bool:
            raise ValueError("media_enabled must be a bool")
        if self.target_lane is TargetLane.TEXT and self.media_enabled:
            raise ValueError("the text target lane cannot enable media")
        if self.target_lane is TargetLane.TEXT and self.text_mode is TextMode.NONE:
            raise ValueError("the text target lane requires a text mode")
        if self.qualification_id is not None:
            _require_non_empty("qualification_id", self.qualification_id)
            if not isinstance(self.verified_target, VerifiedQwenTarget):
                raise ValueError("qualified plans require a verified_target")
        elif self.verified_target is not None:
            raise ValueError("legacy plans must not carry a verified_target")
        if self.text_mode is TextMode.NONE:
            if self.receipt_id is not None:
                raise ValueError("vision-only NONE must not carry a text receipt")
        elif self.qualification_id is not None:
            _require_non_empty("receipt_id", self.receipt_id)
        if (
            self.selection_source is SelectionSource.QUALIFIED_AUTO
            and self.qualification_id is None
        ):
            raise ValueError("qualified_auto plans require a qualification_id")
        if not isinstance(self.fallback_chain, tuple) or any(
            not isinstance(item, QwenFallbackTarget) for item in self.fallback_chain
        ):
            raise ValueError("fallback_chain must contain QwenFallbackTarget values")
        fallback_modes = tuple(item.mode for item in self.fallback_chain)
        if len(set(fallback_modes)) != len(fallback_modes):
            raise ValueError("fallback_chain must not contain duplicate modes")
        if self.text_mode in fallback_modes:
            raise ValueError("fallback_chain must not repeat the selected text mode")
        if not isinstance(self.recovery_action, RecoveryAction):
            raise ValueError("recovery_action must be a RecoveryAction")
        if self.recovery_action is RecoveryAction.NONE:
            if self.pending_fallback_target is not None:
                raise ValueError("pending fallback requires a recovery action")
        else:
            if not isinstance(self.pending_fallback_target, QwenFallbackTarget):
                raise ValueError("recovery requires an explicit pending fallback")
            if self.pending_fallback_target.mode is self.text_mode:
                raise ValueError("pending fallback must advance the failed mode")
            if self.pending_fallback_target.mode in fallback_modes:
                raise ValueError("pending fallback must be removed from fallback_chain")
        if self.recovery_action is RecoveryAction.RELOAD_REQUIRED and (
            self.reason is not PlanReason.INSTALL_FAILED_RELOAD_REQUIRED
        ):
            raise ValueError("reload_required needs the reload failure reason")
        if self.recovery_action is RecoveryAction.STARTUP_FALLBACK and (
            self.reason is not PlanReason.INSTALL_FAILED_STARTUP_FALLBACK
        ):
            raise ValueError("startup_fallback needs the startup failure reason")
        if self.recovery_action is RecoveryAction.NONE and self.reason in (
            PlanReason.INSTALL_FAILED_STARTUP_FALLBACK,
            PlanReason.INSTALL_FAILED_RELOAD_REQUIRED,
        ):
            raise ValueError("a pending install-failure reason needs recovery action")

    def to_status_dict(self) -> dict[str, object]:
        """Return stable JSON-safe values for boot logs and local status."""

        return {
            "target_lane": self.target_lane.value,
            "text_mode": self.text_mode.value,
            "selection_source": self.selection_source.value,
            "qualification_id": self.qualification_id,
            "receipt_id": self.receipt_id,
            "target_verification_id": (
                self.verified_target.verification_id
                if self.verified_target is not None
                else None
            ),
            "target_verification_authority": (
                self.verified_target.verification_authority
                if self.verified_target is not None
                else None
            ),
            "reason": self.reason.value,
            "media_enabled": self.media_enabled,
            "fallback_chain": [item.to_status_dict() for item in self.fallback_chain],
            "recovery_action": self.recovery_action.value,
            "pending_fallback_target": (
                self.pending_fallback_target.to_status_dict()
                if self.pending_fallback_target is not None
                else None
            ),
        }


def _legacy_fallback(
    legacy_plan: QwenRuntimePlan, reason: PlanReason
) -> QwenRuntimePlan:
    """Select the already-usable legacy plan; no recovery remains pending."""

    return replace(
        legacy_plan,
        selection_source=SelectionSource.FALLBACK,
        qualification_id=None,
        receipt_id=None,
        verified_target=None,
        reason=reason,
        recovery_action=RecoveryAction.NONE,
        pending_fallback_target=None,
    )


def _resolve_row(
    artifact: ResolvedQwenArtifact,
    rows: tuple[QwenQualificationRow, ...],
) -> tuple[QwenQualificationRow | None, PlanReason]:
    ids = tuple(row.qualification_id for row in rows)
    if len(set(ids)) != len(ids):
        return None, PlanReason.QUALIFICATION_AMBIGUOUS
    target = artifact.verified_target.identity

    if artifact.public_alias is not None:
        alias_rows = tuple(
            row for row in rows if row.public_alias == artifact.public_alias
        )
        if not alias_rows:
            if any(row.target_identity == target for row in rows):
                return None, PlanReason.QUALIFICATION_ALIAS_MISMATCH
            return None, PlanReason.QUALIFICATION_NOT_FOUND
        identity_rows = tuple(
            row for row in alias_rows if row.target_identity == target
        )
        if not identity_rows:
            return None, PlanReason.QUALIFICATION_TARGET_IDENTITY_MISMATCH
    else:
        identity_rows = tuple(row for row in rows if row.target_identity == target)
        if not identity_rows:
            return None, PlanReason.QUALIFICATION_NOT_FOUND

    if len(identity_rows) != 1:
        return None, PlanReason.QUALIFICATION_AMBIGUOUS
    row = identity_rows[0]
    if row.runtime_versions != artifact.runtime_versions:
        return None, PlanReason.QUALIFICATION_RUNTIME_MISMATCH
    if artifact.hardware_class not in row.hardware_classes:
        return None, PlanReason.QUALIFICATION_HARDWARE_MISMATCH
    return row, PlanReason.QUALIFIED_AUTO


def _mode_qualification(
    row: QwenQualificationRow, mode: TextMode
) -> QwenModeQualification | None:
    return next((item for item in row.mode_qualifications if item.mode is mode), None)


def _mode_evidence(
    artifact: ResolvedQwenArtifact, mode: TextMode
) -> QwenModeEvidence | None:
    return next((item for item in artifact.mode_evidence if item.mode is mode), None)


def _assess_mode(
    row: QwenQualificationRow,
    artifact: ResolvedQwenArtifact,
    mode: TextMode,
) -> tuple[QwenModeQualification | None, PlanReason | None]:
    qualification = _mode_qualification(row, mode)
    if qualification is None:
        return None, PlanReason.MODE_NOT_QUALIFIED
    evidence = _mode_evidence(artifact, mode)
    if evidence is None:
        return None, PlanReason.MODE_EVIDENCE_MISSING
    if qualification.drafter_identity != evidence.drafter_identity:
        return None, PlanReason.MODE_DRAFTER_IDENTITY_MISMATCH
    if not set(qualification.required_probes).issubset(evidence.passed_probes):
        return None, PlanReason.MODE_PROBE_FAILED
    return qualification, None


def _candidate_modes(
    row: QwenQualificationRow, intended: TextMode
) -> tuple[TextMode, ...]:
    ordered = (row.preferred_text_mode, *row.fallback_chain)
    if intended in ordered:
        return ordered[ordered.index(intended) :]
    return (intended, TextMode.NONE)


def _qualified_fallbacks(
    row: QwenQualificationRow,
    artifact: ResolvedQwenArtifact,
    modes: tuple[TextMode, ...],
) -> tuple[QwenFallbackTarget, ...]:
    fallbacks: list[QwenFallbackTarget] = []
    for mode in modes:
        if mode is TextMode.NONE:
            fallbacks.append(QwenFallbackTarget(TextMode.NONE, None))
            continue
        qualification, failure = _assess_mode(row, artifact, mode)
        if failure is None and qualification is not None:
            fallbacks.append(QwenFallbackTarget(mode, qualification.receipt_id))
    if not fallbacks or fallbacks[-1].mode is not TextMode.NONE:
        fallbacks.append(QwenFallbackTarget(TextMode.NONE, None))
    return tuple(fallbacks)


def resolve_qwen_runtime_plan(
    *,
    legacy_plan: QwenRuntimePlan,
    speculative_intent: SpeculativeIntent,
    auto_enabled: bool = False,
    artifact: ResolvedQwenArtifact | None = None,
    qualification_rows: tuple[QwenQualificationRow, ...] = (),
    operator_target_lane: TargetLane | None = None,
) -> QwenRuntimePlan:
    """Resolve legacy behavior or an explicitly enabled qualified auto plan."""

    if not isinstance(legacy_plan, QwenRuntimePlan):
        raise ValueError("legacy_plan must be a QwenRuntimePlan")
    if not isinstance(speculative_intent, SpeculativeIntent):
        raise ValueError("speculative_intent must be a SpeculativeIntent")
    if type(auto_enabled) is not bool:
        raise ValueError("auto_enabled must be a bool")
    if artifact is not None and not isinstance(artifact, ResolvedQwenArtifact):
        raise ValueError("artifact must be a ResolvedQwenArtifact")
    if not isinstance(qualification_rows, tuple) or any(
        not isinstance(row, QwenQualificationRow) for row in qualification_rows
    ):
        raise ValueError("qualification_rows must contain QwenQualificationRow values")
    if operator_target_lane is not None and not isinstance(
        operator_target_lane, TargetLane
    ):
        raise ValueError("operator_target_lane must be a TargetLane")

    if not auto_enabled:
        return legacy_plan
    if operator_target_lane is not None:
        return legacy_plan
    if speculative_intent is SpeculativeIntent.EXPLICIT_ENABLED:
        return legacy_plan
    if artifact is None or not qualification_rows:
        return _legacy_fallback(legacy_plan, PlanReason.QUALIFICATION_NOT_FOUND)

    row, row_failure = _resolve_row(artifact, qualification_rows)
    if row is None:
        return _legacy_fallback(legacy_plan, row_failure)

    explicit_no_spec = speculative_intent is SpeculativeIntent.EXPLICIT_DISABLED
    intended = TextMode.NATIVE_AR if explicit_no_spec else row.preferred_text_mode
    candidates = _candidate_modes(row, intended)
    intended_failure: PlanReason | None = None

    for index, mode in enumerate(candidates):
        if mode is TextMode.NONE:
            return QwenRuntimePlan(
                target_lane=TargetLane.VISION,
                text_mode=TextMode.NONE,
                selection_source=SelectionSource.FALLBACK,
                qualification_id=row.qualification_id,
                receipt_id=None,
                verified_target=artifact.verified_target,
                reason=intended_failure or PlanReason.MODE_NOT_QUALIFIED,
                media_enabled=True,
            )
        qualification, failure = _assess_mode(row, artifact, mode)
        if failure is not None or qualification is None:
            if intended_failure is None:
                intended_failure = failure or PlanReason.MODE_NOT_QUALIFIED
            continue

        is_intended = index == 0
        if is_intended:
            source = (
                SelectionSource.OPERATOR
                if explicit_no_spec
                else SelectionSource.QUALIFIED_AUTO
            )
            reason = (
                PlanReason.QUALIFIED_AUTO_NO_SPEC
                if explicit_no_spec
                else PlanReason.QUALIFIED_AUTO
            )
        else:
            source = SelectionSource.FALLBACK
            reason = intended_failure or PlanReason.MODE_NOT_QUALIFIED
        return QwenRuntimePlan(
            target_lane=TargetLane.VISION,
            text_mode=mode,
            selection_source=source,
            qualification_id=row.qualification_id,
            receipt_id=qualification.receipt_id,
            verified_target=artifact.verified_target,
            reason=reason,
            media_enabled=True,
            fallback_chain=_qualified_fallbacks(row, artifact, candidates[index + 1 :]),
        )

    raise AssertionError("candidate chain must terminate at NONE")


def resolve_qwen_install_failure(
    plan: QwenRuntimePlan, *, target_was_mutated: bool
) -> QwenRuntimePlan:
    """Advance one install fallback without allowing unsafe target reuse."""

    if not isinstance(plan, QwenRuntimePlan):
        raise ValueError("plan must be a QwenRuntimePlan")
    if type(target_was_mutated) is not bool:
        raise ValueError("target_was_mutated must be a bool")
    if plan.recovery_action is not RecoveryAction.NONE:
        raise ValueError("cannot transition while a recovery action is pending")
    if plan.target_lane is not TargetLane.VISION or plan.qualification_id is None:
        raise ValueError("install failure transition requires a qualified vision plan")
    if not plan.fallback_chain:
        raise ValueError("install failure transition requires a fallback target")

    next_target, *remaining = plan.fallback_chain
    action = (
        RecoveryAction.RELOAD_REQUIRED
        if target_was_mutated
        else RecoveryAction.STARTUP_FALLBACK
    )
    reason = (
        PlanReason.INSTALL_FAILED_RELOAD_REQUIRED
        if target_was_mutated
        else PlanReason.INSTALL_FAILED_STARTUP_FALLBACK
    )
    return replace(
        plan,
        selection_source=SelectionSource.FALLBACK,
        reason=reason,
        fallback_chain=tuple(remaining),
        recovery_action=action,
        pending_fallback_target=next_target,
    )


def complete_qwen_reload(
    plan: QwenRuntimePlan, *, reloaded_target: VerifiedQwenTarget
) -> QwenRuntimePlan:
    """Verify a fresh clean target reload, then permit fallback startup."""

    if not isinstance(plan, QwenRuntimePlan):
        raise ValueError("plan must be a QwenRuntimePlan")
    if plan.recovery_action is not RecoveryAction.RELOAD_REQUIRED:
        raise ValueError("no clean target reload is pending")
    if not isinstance(reloaded_target, VerifiedQwenTarget):
        raise ValueError("reloaded_target must be a freshly verified target")
    expected = plan.verified_target
    if expected is None:  # defended by QwenRuntimePlan, kept for type narrowing
        raise ValueError("reload recovery has no expected verified target")
    if reloaded_target is expected:
        raise ValueError("reloaded_target must be a fresh verification binding")
    if reloaded_target != expected:
        raise ValueError("reloaded target identity or verification binding mismatched")
    return replace(
        plan,
        reason=PlanReason.INSTALL_FAILED_STARTUP_FALLBACK,
        recovery_action=RecoveryAction.STARTUP_FALLBACK,
        verified_target=reloaded_target,
    )


def complete_qwen_fallback_start(plan: QwenRuntimePlan) -> QwenRuntimePlan:
    """Publish a fallback only after its startup completed successfully."""

    if not isinstance(plan, QwenRuntimePlan):
        raise ValueError("plan must be a QwenRuntimePlan")
    if plan.recovery_action is not RecoveryAction.STARTUP_FALLBACK:
        raise ValueError("no fallback startup is pending")
    target = plan.pending_fallback_target
    if target is None:  # defended by QwenRuntimePlan, kept for type narrowing
        raise ValueError("fallback startup has no target")
    return replace(
        plan,
        text_mode=target.mode,
        receipt_id=target.receipt_id,
        reason=PlanReason.INSTALL_FALLBACK_ACTIVE,
        recovery_action=RecoveryAction.NONE,
        pending_fallback_target=None,
    )


def resolve_qwen_fallback_start_failure(
    plan: QwenRuntimePlan, *, target_was_mutated: bool
) -> QwenRuntimePlan:
    """Advance after a pending fallback failed before becoming active."""

    if not isinstance(plan, QwenRuntimePlan):
        raise ValueError("plan must be a QwenRuntimePlan")
    if type(target_was_mutated) is not bool:
        raise ValueError("target_was_mutated must be a bool")
    if plan.recovery_action is not RecoveryAction.STARTUP_FALLBACK:
        raise ValueError("no fallback startup is pending")
    if not plan.fallback_chain:
        raise ValueError("failed terminal fallback has no next target")
    next_target, *remaining = plan.fallback_chain
    action = (
        RecoveryAction.RELOAD_REQUIRED
        if target_was_mutated
        else RecoveryAction.STARTUP_FALLBACK
    )
    reason = (
        PlanReason.INSTALL_FAILED_RELOAD_REQUIRED
        if target_was_mutated
        else PlanReason.INSTALL_FAILED_STARTUP_FALLBACK
    )
    return replace(
        plan,
        reason=reason,
        fallback_chain=tuple(remaining),
        recovery_action=action,
        pending_fallback_target=next_target,
    )


__all__ = [
    "PlanReason",
    "QwenDrafterIdentity",
    "QwenFallbackTarget",
    "QwenModeEvidence",
    "QwenModeQualification",
    "QwenQualificationRow",
    "QwenRuntimePlan",
    "QwenTargetIdentity",
    "RecoveryAction",
    "ResolvedQwenArtifact",
    "SelectionSource",
    "SpeculativeIntent",
    "TargetLane",
    "TextMode",
    "complete_qwen_fallback_start",
    "complete_qwen_reload",
    "resolve_qwen_fallback_start_failure",
    "resolve_qwen_install_failure",
    "resolve_qwen_runtime_plan",
]
