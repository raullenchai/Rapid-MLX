# SPDX-License-Identifier: Apache-2.0
"""Pure, fail-closed planning primitives for the Qwen auto runtime.

This module deliberately has no model, MLX, registry, or filesystem access.
Callers resolve the current (legacy) serving decision and immutable artifact
facts elsewhere, then pass them here.  Consequently a plan can be inspected at
boot and re-resolved after load-time probes without loading a model in tests.

B0 installs no production qualification rows.  With auto selection disabled
or an empty row set, :func:`resolve_qwen_runtime_plan` returns the exact legacy
plan object it was given.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import PurePosixPath
from typing import TypeVar


class SelectionSource(str, Enum):
    """Authority that selected a runtime plan."""

    OPERATOR = "operator"
    ALIAS_DEFAULT = "alias_default"
    QUALIFIED_AUTO = "qualified_auto"
    FALLBACK = "fallback"


class TargetLane(str, Enum):
    """Lane that owns the target model."""

    VISION = "vision"
    TEXT = "text"


class TextMode(str, Enum):
    """Companion text-decoder mode within the selected target lane."""

    NONE = "none"
    NATIVE_AR = "native_ar"
    MTP = "mtp"


class PlanReason(str, Enum):
    """Stable, closed explanations for runtime-plan selection."""

    LEGACY_OPERATOR = "legacy_operator"
    LEGACY_ALIAS_DEFAULT = "legacy_alias_default"
    LEGACY_DEFAULT = "legacy_default"
    QUALIFIED_AUTO = "qualified_auto"
    QUALIFIED_AUTO_NO_SPEC = "qualified_auto_no_spec"
    QUALIFICATION_NOT_FOUND = "qualification_not_found"
    QUALIFICATION_ALIAS_MISMATCH = "qualification_alias_mismatch"
    QUALIFICATION_IDENTITY_MISMATCH = "qualification_identity_mismatch"
    QUALIFICATION_RUNTIME_MISMATCH = "qualification_runtime_mismatch"
    QUALIFICATION_HARDWARE_MISMATCH = "qualification_hardware_mismatch"
    QUALIFICATION_PROBE_FAILED = "qualification_probe_failed"
    QUALIFICATION_AMBIGUOUS = "qualification_ambiguous"
    QUALIFICATION_TEXT_MODE_UNAVAILABLE = "qualification_text_mode_unavailable"
    INSTALL_FAILED_STARTUP_FALLBACK = "install_failed_startup_fallback"
    INSTALL_FAILED_RELOAD_REQUIRED = "install_failed_reload_required"


class RecoveryAction(str, Enum):
    """What the owner must do before the represented plan can proceed."""

    NONE = "none"
    STARTUP_FALLBACK = "startup_fallback"
    RELOAD_REQUIRED = "reload_required"


def _require_non_empty(label: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty")


def _require_immutable_revision(label: str, value: str) -> None:
    """Reject mutable names such as ``main`` or ``latest``.

    Hugging Face snapshots are keyed by a 40-character Git object id today.
    The 64-character forms leave room for Git SHA-256 and explicit content
    digests without allowing a mutable branch/tag spelling.
    """

    if isinstance(value, str) and re.fullmatch(
        r"(?:[0-9a-f]{40}|[0-9a-f]{64}|sha256:[0-9a-f]{64})", value
    ):
        return
    raise ValueError(f"{label} must be an immutable commit or content digest")


def _require_relative_artifact_path(label: str, value: str) -> None:
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


def _require_canonical_pairs(label: str, values: tuple[tuple[str, str], ...]) -> None:
    """Require one deterministic representation for mapping-like facts."""

    if not isinstance(values, tuple) or any(
        not isinstance(item, tuple)
        or len(item) != 2
        or not all(isinstance(part, str) for part in item)
        for item in values
    ):
        raise ValueError(f"{label} must be a tuple of string pairs")
    keys = tuple(key for key, _value in values)
    if any(not key.strip() for key in keys):
        raise ValueError(f"{label} keys must be non-empty")
    if any(not value.strip() for _key, value in values):
        raise ValueError(f"{label} values must be non-empty")
    if len(set(keys)) != len(keys):
        raise ValueError(f"{label} keys must be unique")
    if keys != tuple(sorted(keys)):
        raise ValueError(f"{label} must be sorted by key")


@dataclass(frozen=True, slots=True)
class QwenArtifactIdentity:
    """Immutable facts that identify one exact target/drafter artifact.

    All fields participate in equality.  In particular, the resolver never
    infers eligibility from a repository or model-family substring.  Geometry
    is intentionally represented as an ordered layer sequence plus canonical
    key/value facts: layer order is semantic while mapping order is not.
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
    drafter_repo: str | None = None
    drafter_revision: str | None = None
    drafter_artifact_path: str | None = None

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
        drafter_fields = (
            self.drafter_repo,
            self.drafter_revision,
            self.drafter_artifact_path,
        )
        if any(value is None for value in drafter_fields) and any(
            value is not None for value in drafter_fields
        ):
            raise ValueError(
                "drafter_repo, drafter_revision, and drafter_artifact_path "
                "must be provided together"
            )
        if self.target_subfolder is not None:
            _require_relative_artifact_path("target_subfolder", self.target_subfolder)
        if self.drafter_repo is not None:
            _require_non_empty("drafter_repo", self.drafter_repo)
            _require_immutable_revision("drafter_revision", self.drafter_revision or "")
            _require_relative_artifact_path(
                "drafter_artifact_path", self.drafter_artifact_path or ""
            )


@dataclass(frozen=True, slots=True)
class ResolvedQwenArtifact:
    """Exact artifact plus the environment/probes observed by the caller.

    ``public_alias`` is ``None`` for a raw repository or local path whose
    immutable identity was resolved independently.  A non-``None`` alias must
    match the qualification row in addition to the artifact identity.
    """

    identity: QwenArtifactIdentity
    public_alias: str | None
    runtime_versions: tuple[tuple[str, str], ...]
    hardware_class: str
    passed_probes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.identity, QwenArtifactIdentity):
            raise ValueError("identity must be a QwenArtifactIdentity")
        if self.public_alias is not None:
            _require_non_empty("public_alias", self.public_alias)
        _require_non_empty("hardware_class", self.hardware_class)
        _require_canonical_pairs("runtime_versions", self.runtime_versions)
        if not isinstance(self.passed_probes, tuple):
            raise ValueError("passed_probes must be a tuple")
        if len(set(self.passed_probes)) != len(self.passed_probes):
            raise ValueError("passed_probes must be unique")
        if self.passed_probes != tuple(sorted(self.passed_probes)):
            raise ValueError("passed_probes must be sorted")
        if any(not probe.strip() for probe in self.passed_probes):
            raise ValueError("passed_probes must be non-empty strings")


@dataclass(frozen=True, slots=True)
class SpeculativeInput:
    """Speculative decoder input with provenance preserved.

    ``explicit=True, mode=NONE`` represents an operator's no-spec request.
    ``explicit=False, mode=MTP`` represents an alias-injected default.  Those
    cases look identical after legacy CLI normalization unless this provenance
    is retained before normalization.
    """

    mode: TextMode
    explicit: bool

    def __post_init__(self) -> None:
        if not isinstance(self.mode, TextMode):
            raise ValueError("mode must be a TextMode")
        if type(self.explicit) is not bool:
            raise ValueError("explicit must be a bool")


@dataclass(frozen=True, slots=True)
class QwenRuntimePlan:
    """Immutable source of truth for one Qwen process runtime."""

    target_lane: TargetLane
    text_mode: TextMode
    selection_source: SelectionSource
    qualification_id: str | None
    reason: PlanReason
    media_enabled: bool
    fallback_chain: tuple[TextMode, ...] = ()
    recovery_action: RecoveryAction = RecoveryAction.NONE

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
        if not isinstance(self.fallback_chain, tuple) or any(
            not isinstance(mode, TextMode) for mode in self.fallback_chain
        ):
            raise ValueError("fallback_chain must contain TextMode values")
        if not isinstance(self.recovery_action, RecoveryAction):
            raise ValueError("recovery_action must be a RecoveryAction")
        if self.target_lane is TargetLane.TEXT and self.media_enabled:
            raise ValueError("the text target lane cannot enable media")
        if self.target_lane is TargetLane.TEXT and self.text_mode is TextMode.NONE:
            raise ValueError("the text target lane requires a text mode")
        if self.media_enabled and self.target_lane is not TargetLane.VISION:
            raise ValueError("media requires the vision target lane")
        if self.qualification_id is not None:
            _require_non_empty("qualification_id", self.qualification_id)
        if (
            self.selection_source is SelectionSource.QUALIFIED_AUTO
            and self.qualification_id is None
        ):
            raise ValueError("qualified_auto plans require a qualification_id")
        if len(set(self.fallback_chain)) != len(self.fallback_chain):
            raise ValueError("fallback_chain must not contain duplicate modes")
        if self.text_mode in self.fallback_chain:
            raise ValueError("fallback_chain must not repeat the selected text mode")
        if (
            self.reason is PlanReason.INSTALL_FAILED_RELOAD_REQUIRED
            and self.recovery_action is not RecoveryAction.RELOAD_REQUIRED
        ):
            raise ValueError("a mutated install failure must require reload")
        if (
            self.recovery_action is RecoveryAction.RELOAD_REQUIRED
            and self.reason is not PlanReason.INSTALL_FAILED_RELOAD_REQUIRED
        ):
            raise ValueError("reload_required is reserved for mutated install failure")
        if (
            self.reason is PlanReason.INSTALL_FAILED_STARTUP_FALLBACK
            and self.recovery_action is not RecoveryAction.STARTUP_FALLBACK
        ):
            raise ValueError("an unmutated install failure requires startup fallback")
        if (
            self.recovery_action is RecoveryAction.STARTUP_FALLBACK
            and self.reason is not PlanReason.INSTALL_FAILED_STARTUP_FALLBACK
        ):
            raise ValueError(
                "startup_fallback is reserved for an unmutated install failure"
            )


@dataclass(frozen=True, slots=True)
class QwenQualificationRow:
    """One reviewable, receipt-backed qualification contract.

    No rows are registered by this module.  Later slices may supply rows from a
    curated registry, but a row is usable only when its complete artifact,
    runtime, hardware, and probe contract matches.  ``receipt_id`` and the
    supported sampling/processor/tool/KV fields are immutable review metadata
    in B0; later request-policy wiring may consume them, while the B0 resolver
    gates only boot-wide facts.
    """

    qualification_id: str
    public_alias: str
    identity: QwenArtifactIdentity
    supported_text_modes: tuple[TextMode, ...]
    preferred_text_mode: TextMode
    fallback_chain: tuple[TextMode, ...]
    runtime_versions: tuple[tuple[str, str], ...]
    hardware_classes: tuple[str, ...]
    required_probes: tuple[str, ...]
    supported_sampling: tuple[str, ...]
    supported_logits_processors: tuple[str, ...]
    supported_kv_configurations: tuple[str, ...]
    tools_supported: bool
    receipt_id: str
    media_enabled: bool = True

    def __post_init__(self) -> None:
        for label in ("qualification_id", "public_alias", "receipt_id"):
            _require_non_empty(label, getattr(self, label))
        if not isinstance(self.identity, QwenArtifactIdentity):
            raise ValueError("identity must be a QwenArtifactIdentity")
        if not isinstance(self.preferred_text_mode, TextMode):
            raise ValueError("preferred_text_mode must be a TextMode")
        if type(self.media_enabled) is not bool:
            raise ValueError("media_enabled must be a bool")
        if not self.media_enabled:
            raise ValueError("auto-runtime qualification rows must enable media")
        if not isinstance(self.supported_text_modes, tuple) or any(
            not isinstance(mode, TextMode) for mode in self.supported_text_modes
        ):
            raise ValueError("supported_text_modes must contain TextMode values")
        if not self.supported_text_modes:
            raise ValueError("supported_text_modes must not be empty")
        if TextMode.NONE in self.supported_text_modes:
            raise ValueError("NONE is a fallback, not a qualified text mode")
        if len(set(self.supported_text_modes)) != len(self.supported_text_modes):
            raise ValueError("supported_text_modes must be unique")
        if self.preferred_text_mode not in self.supported_text_modes:
            raise ValueError("preferred_text_mode must be supported")
        if (
            TextMode.MTP in self.supported_text_modes
            and self.identity.drafter_revision is None
        ):
            raise ValueError("MTP qualification requires an immutable drafter identity")
        if not isinstance(self.fallback_chain, tuple) or any(
            not isinstance(mode, TextMode) for mode in self.fallback_chain
        ):
            raise ValueError("fallback_chain must contain TextMode values")
        if len(set(self.fallback_chain)) != len(self.fallback_chain):
            raise ValueError("fallback_chain must not contain duplicates")
        if self.preferred_text_mode in self.fallback_chain:
            raise ValueError("fallback_chain must follow the preferred text mode")
        if not self.fallback_chain or self.fallback_chain[-1] is not TextMode.NONE:
            raise ValueError("fallback_chain must terminate at vision-only NONE")
        if any(
            mode is not TextMode.NONE and mode not in self.supported_text_modes
            for mode in self.fallback_chain
        ):
            raise ValueError("fallback_chain contains an unsupported text mode")
        fallback_order = {
            TextMode.MTP: 2,
            TextMode.NATIVE_AR: 1,
            TextMode.NONE: 0,
        }
        ordered_modes = (self.preferred_text_mode, *self.fallback_chain)
        if any(
            fallback_order[current] <= fallback_order[next_mode]
            for current, next_mode in zip(ordered_modes, ordered_modes[1:])
        ):
            raise ValueError("fallback_chain must move toward simpler text modes")
        _require_canonical_pairs("runtime_versions", self.runtime_versions)
        if (
            not isinstance(self.hardware_classes, tuple)
            or not self.hardware_classes
            or any(
                not isinstance(item, str) or not item.strip()
                for item in self.hardware_classes
            )
        ):
            raise ValueError("hardware_classes must contain non-empty values")
        if len(set(self.hardware_classes)) != len(self.hardware_classes):
            raise ValueError("hardware_classes must be unique")
        if self.hardware_classes != tuple(sorted(self.hardware_classes)):
            raise ValueError("hardware_classes must be sorted")
        if not isinstance(self.required_probes, tuple):
            raise ValueError("required_probes must be a tuple")
        if len(set(self.required_probes)) != len(self.required_probes):
            raise ValueError("required_probes must be unique")
        if self.required_probes != tuple(sorted(self.required_probes)):
            raise ValueError("required_probes must be sorted")
        if any(
            not isinstance(probe, str) or not probe.strip()
            for probe in self.required_probes
        ):
            raise ValueError("required_probes must be non-empty strings")
        for label, values in (
            ("supported_sampling", self.supported_sampling),
            ("supported_logits_processors", self.supported_logits_processors),
            ("supported_kv_configurations", self.supported_kv_configurations),
        ):
            if (
                not isinstance(values, tuple)
                or not values
                or any(
                    not isinstance(value, str) or not value.strip() for value in values
                )
            ):
                raise ValueError(f"{label} must contain non-empty values")
            if len(set(values)) != len(values):
                raise ValueError(f"{label} must be unique")
            if values != tuple(sorted(values)):
                raise ValueError(f"{label} must be sorted")
        if type(self.tools_supported) is not bool:
            raise ValueError("tools_supported must be a bool")


_T = TypeVar("_T")


def _duplicates(values: tuple[_T, ...]) -> bool:
    return len(set(values)) != len(values)


def _fallback(
    legacy_plan: QwenRuntimePlan,
    reason: PlanReason,
) -> QwenRuntimePlan:
    """Keep behavior fields while recording that legacy was selected.

    The returned plan already *is* the usable legacy fallback, so it has no
    outstanding recovery action.  This differs from an install failure, where
    the engine still has to start another companion or reload the target.
    """

    return replace(
        legacy_plan,
        selection_source=SelectionSource.FALLBACK,
        qualification_id=None,
        reason=reason,
        recovery_action=RecoveryAction.NONE,
    )


def _resolve_row(
    artifact: ResolvedQwenArtifact,
    rows: tuple[QwenQualificationRow, ...],
) -> tuple[QwenQualificationRow | None, PlanReason]:
    """Return one exact row, or a closed fail-closed reason."""

    if _duplicates(tuple(row.qualification_id for row in rows)):
        return None, PlanReason.QUALIFICATION_AMBIGUOUS

    if artifact.public_alias is not None:
        alias_rows = tuple(
            row for row in rows if row.public_alias == artifact.public_alias
        )
        if not alias_rows:
            # An identity belonging to another public alias cannot be borrowed.
            if any(row.identity == artifact.identity for row in rows):
                return None, PlanReason.QUALIFICATION_ALIAS_MISMATCH
            return None, PlanReason.QUALIFICATION_NOT_FOUND
        identity_rows = tuple(
            row for row in alias_rows if row.identity == artifact.identity
        )
        if not identity_rows:
            return None, PlanReason.QUALIFICATION_IDENTITY_MISMATCH
    else:
        # Raw repo/local-path spelling is allowed only after the caller has
        # resolved the complete immutable identity.  Names are never matched.
        identity_rows = tuple(row for row in rows if row.identity == artifact.identity)
        if not identity_rows:
            return None, PlanReason.QUALIFICATION_NOT_FOUND

    if len(identity_rows) != 1:
        return None, PlanReason.QUALIFICATION_AMBIGUOUS
    row = identity_rows[0]
    if row.runtime_versions != artifact.runtime_versions:
        return None, PlanReason.QUALIFICATION_RUNTIME_MISMATCH
    if artifact.hardware_class not in row.hardware_classes:
        return None, PlanReason.QUALIFICATION_HARDWARE_MISMATCH
    if not set(row.required_probes).issubset(artifact.passed_probes):
        return None, PlanReason.QUALIFICATION_PROBE_FAILED
    return row, PlanReason.QUALIFIED_AUTO


def _fallback_after(
    row: QwenQualificationRow, selected_mode: TextMode
) -> tuple[TextMode, ...]:
    ordered = (row.preferred_text_mode, *row.fallback_chain)
    if selected_mode not in ordered:
        # A supported non-preferred mode need not be in the default chain.  Its
        # safe terminal fallback is the authoritative vision scheduler.
        return (TextMode.NONE,)
    return ordered[ordered.index(selected_mode) + 1 :]


def resolve_qwen_runtime_plan(
    *,
    legacy_plan: QwenRuntimePlan,
    speculative: SpeculativeInput,
    auto_enabled: bool = False,
    artifact: ResolvedQwenArtifact | None = None,
    qualification_rows: tuple[QwenQualificationRow, ...] = (),
    operator_target_lane: TargetLane | None = None,
) -> QwenRuntimePlan:
    """Resolve current legacy behavior or a future exact qualified auto plan.

    B0 is behavior-neutral: the defaults return ``legacy_plan`` unchanged.
    Explicit target-lane choice and explicit speculative decoding also preserve
    the current plan.  An explicit no-spec input may select a qualified native
    AR companion, matching the staged design, but only when auto is enabled and
    one exact row passes every gate.
    """

    if not isinstance(legacy_plan, QwenRuntimePlan):
        raise ValueError("legacy_plan must be a QwenRuntimePlan")
    if not isinstance(speculative, SpeculativeInput):
        raise ValueError("speculative must be a SpeculativeInput")
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
    if speculative.explicit and speculative.mode is not TextMode.NONE:
        return legacy_plan
    if artifact is None or not qualification_rows:
        return _fallback(legacy_plan, PlanReason.QUALIFICATION_NOT_FOUND)

    row, reason = _resolve_row(artifact, qualification_rows)
    if row is None:
        return _fallback(legacy_plan, reason)

    if speculative.explicit:
        selected_mode = TextMode.NATIVE_AR
        selection_source = SelectionSource.OPERATOR
        plan_reason = PlanReason.QUALIFIED_AUTO_NO_SPEC
    else:
        selected_mode = row.preferred_text_mode
        selection_source = SelectionSource.QUALIFIED_AUTO
        plan_reason = PlanReason.QUALIFIED_AUTO

    if selected_mode not in row.supported_text_modes:
        return _fallback(legacy_plan, PlanReason.QUALIFICATION_TEXT_MODE_UNAVAILABLE)

    return QwenRuntimePlan(
        target_lane=TargetLane.VISION,
        text_mode=selected_mode,
        selection_source=selection_source,
        qualification_id=row.qualification_id,
        reason=plan_reason,
        media_enabled=row.media_enabled,
        fallback_chain=_fallback_after(row, selected_mode),
    )


def resolve_qwen_install_failure(
    plan: QwenRuntimePlan,
    *,
    target_was_mutated: bool,
) -> QwenRuntimePlan:
    """Resolve a companion-lane install failure without unsafe reuse.

    Before target mutation, the next declared startup fallback is safe.  Once
    an injector has mutated the loaded target, B0 cannot prove rollback, so the
    returned state requires a clean reload and exposes no same-instance chain.
    """

    if not isinstance(plan, QwenRuntimePlan):
        raise ValueError("plan must be a QwenRuntimePlan")
    if type(target_was_mutated) is not bool:
        raise ValueError("target_was_mutated must be a bool")
    if plan.target_lane is not TargetLane.VISION or plan.qualification_id is None:
        raise ValueError("install failure transition requires a qualified vision plan")

    if target_was_mutated:
        return replace(
            plan,
            selection_source=SelectionSource.FALLBACK,
            reason=PlanReason.INSTALL_FAILED_RELOAD_REQUIRED,
            fallback_chain=(),
            recovery_action=RecoveryAction.RELOAD_REQUIRED,
        )

    next_mode = plan.fallback_chain[0] if plan.fallback_chain else TextMode.NONE
    remaining = plan.fallback_chain[1:] if plan.fallback_chain else ()
    return replace(
        plan,
        text_mode=next_mode,
        selection_source=SelectionSource.FALLBACK,
        reason=PlanReason.INSTALL_FAILED_STARTUP_FALLBACK,
        fallback_chain=remaining,
        recovery_action=RecoveryAction.STARTUP_FALLBACK,
    )


__all__ = [
    "PlanReason",
    "QwenArtifactIdentity",
    "QwenQualificationRow",
    "QwenRuntimePlan",
    "RecoveryAction",
    "ResolvedQwenArtifact",
    "SelectionSource",
    "SpeculativeInput",
    "TargetLane",
    "TextMode",
    "resolve_qwen_runtime_plan",
    "resolve_qwen_install_failure",
]
