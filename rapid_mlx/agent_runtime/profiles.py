# SPDX-License-Identifier: Apache-2.0
"""Model-specific budgets for the shared agent runtime."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .models import AgentProfile

MINICPM5_2B_PROFILE = AgentProfile(
    name="minicpm5-2b",
    # Preserve the original six-connector budget and reserve two additional
    # slots for Rapid's bounded host helpers. They are not substitutes for
    # tools the operator deliberately configured.
    max_visible_tools=8,
    # Physical Desktop dogfood showed a two-file + exact-calculation task
    # exhausting eight rounds before synthesis. Keep the six-connector
    # bound, repeat guard, and output ceiling, but allow the same twelve
    # bounded rounds as the default profile so ordinary organizing work can
    # finish instead of surfacing an internal budget code.
    max_tool_rounds=12,
    repeated_call_limit=2,
    max_output_tokens=900,
)

QWEN35_4B_PROFILE = AgentProfile(
    name="qwen3.5-4b",
    max_visible_tools=8,
    max_tool_rounds=8,
    repeated_call_limit=2,
    max_output_tokens=900,
)

QWEN35_9B_PROFILE = AgentProfile(
    name="qwen3.5-9b",
    max_visible_tools=8,
    max_tool_rounds=8,
    repeated_call_limit=2,
    max_output_tokens=900,
)

LFM25_1B_PROFILE = AgentProfile(
    name="lfm2.5-1b",
    # Three Desktop tools plus the two Rapid helpers reserved by the generic
    # selector. Intent routing still exposes at most one Desktop tool per turn.
    max_visible_tools=5,
    max_tool_rounds=4,
    repeated_call_limit=2,
    max_output_tokens=700,
)


def _top_model_profile(name: str, *, small: bool = False) -> AgentProfile:
    """Create a bounded candidate profile without granting product access."""

    return AgentProfile(
        name=name,
        max_visible_tools=6 if small else 8,
        max_tool_rounds=6 if small else 8,
        repeated_call_limit=2,
        max_output_tokens=700 if small else 900,
    )


QWEN36_27B_PROFILE = _top_model_profile("qwen3.6-27b")
QWEN36_35B_PROFILE = _top_model_profile("qwen3.6-35b")
QWEN38_27B_PROFILE = _top_model_profile("qwen3.8-27b")
BONSAI_27B_PROFILE = _top_model_profile("bonsai-27b")
QWEN3_CODER_30B_PROFILE = _top_model_profile("qwen3-coder-30b")
LING3_TINY_PROFILE = _top_model_profile("ling-3.0-tiny", small=True)
GPT_OSS_20B_PROFILE = _top_model_profile("gpt-oss-20b")

DEFAULT_PROFILE = AgentProfile(
    name="default",
    max_visible_tools=14,
    max_tool_rounds=12,
    repeated_call_limit=2,
)

CONSERVATIVE_LOCAL_PROFILE = AgentProfile(
    name="default-conservative",
    max_visible_tools=8,
    max_tool_rounds=8,
    repeated_call_limit=2,
    max_output_tokens=900,
)


@dataclass(frozen=True)
class PersonalIntelligenceQualification:
    """One evidence-backed model artifact and harness pairing.

    Public names and backing artifact names are separate because a Rapid alias
    is user-facing while the loaded repository is the runtime identity. Two
    quantizations may share a harness and parser without sharing qualification.
    """

    id: str
    public_identities: frozenset[str]
    backing_identities: frozenset[str]
    profile: AgentProfile
    tool_call_parser: str
    receipt: str
    evidence: str


PERSONAL_INTELLIGENCE_QUALIFICATIONS = (
    PersonalIntelligenceQualification(
        id="qwen3.5-4b-q4-v1",
        public_identities=frozenset(
            {"qwen3.5-4b-4bit", "mlx-community/Qwen3.5-4B-MLX-4bit"}
        ),
        backing_identities=frozenset({"mlx-community/Qwen3.5-4B-MLX-4bit"}),
        profile=QWEN35_4B_PROFILE,
        tool_call_parser="hermes",
        receipt="reports/benchmarks/personal-intelligence-qwen3.5-4b-4bit.json",
        evidence=(
            "docs/engineering/performance/"
            "2026-09-15-personal-intelligence-top-model-qualification.md"
        ),
    ),
    PersonalIntelligenceQualification(
        id="qwen3.5-4b-q8-v1",
        public_identities=frozenset({"mlx-community/Qwen3.5-4B-8bit"}),
        backing_identities=frozenset({"mlx-community/Qwen3.5-4B-8bit"}),
        profile=QWEN35_4B_PROFILE,
        tool_call_parser="hermes",
        receipt="reports/benchmarks/personal-intelligence-qwen3.5-4b-8bit.json",
        evidence=(
            "docs/engineering/performance/"
            "2026-09-16-personal-intelligence-remaining-builds-qualification.md"
        ),
    ),
    PersonalIntelligenceQualification(
        id="qwen3.5-9b-q4-v1",
        public_identities=frozenset(
            {"qwen3.5-9b-4bit", "mlx-community/Qwen3.5-9B-4bit"}
        ),
        backing_identities=frozenset({"mlx-community/Qwen3.5-9B-4bit"}),
        profile=QWEN35_9B_PROFILE,
        tool_call_parser="hermes",
        receipt="reports/benchmarks/personal-intelligence-qwen3.5-9b-4bit.json",
        evidence=(
            "docs/engineering/performance/"
            "2026-09-15-personal-intelligence-top-model-qualification.md"
        ),
    ),
    PersonalIntelligenceQualification(
        id="qwen3.5-9b-q8-v1",
        public_identities=frozenset({"mlx-community/Qwen3.5-9B-8bit"}),
        backing_identities=frozenset({"mlx-community/Qwen3.5-9B-8bit"}),
        profile=QWEN35_9B_PROFILE,
        tool_call_parser="hermes",
        receipt="reports/benchmarks/personal-intelligence-qwen3.5-9b-8bit.json",
        evidence=(
            "docs/engineering/performance/"
            "2026-09-16-personal-intelligence-remaining-builds-qualification.md"
        ),
    ),
    PersonalIntelligenceQualification(
        id="qwen3.6-35b-q8-v1",
        public_identities=frozenset(
            {"qwen3.6-35b-8bit", "mlx-community/Qwen3.6-35B-A3B-8bit"}
        ),
        backing_identities=frozenset({"mlx-community/Qwen3.6-35B-A3B-8bit"}),
        profile=QWEN36_35B_PROFILE,
        tool_call_parser="qwen3_coder_xml",
        receipt="reports/benchmarks/personal-intelligence-qwen3.6-35b-8bit.json",
        evidence=(
            "docs/engineering/performance/"
            "2026-09-15-personal-intelligence-top-model-qualification.md"
        ),
    ),
    PersonalIntelligenceQualification(
        id="qwen3.6-35b-q4-v1",
        public_identities=frozenset({"mlx-community/Qwen3.6-35B-A3B-4bit"}),
        backing_identities=frozenset({"mlx-community/Qwen3.6-35B-A3B-4bit"}),
        profile=QWEN36_35B_PROFILE,
        tool_call_parser="qwen3_coder_xml",
        receipt="reports/benchmarks/personal-intelligence-qwen3.6-35b-4bit.json",
        evidence=(
            "docs/engineering/performance/"
            "2026-09-16-personal-intelligence-remaining-builds-qualification.md"
        ),
    ),
    PersonalIntelligenceQualification(
        id="qwen3.6-27b-q4-v1",
        public_identities=frozenset({"mlx-community/Qwen3.6-27B-4bit"}),
        backing_identities=frozenset({"mlx-community/Qwen3.6-27B-4bit"}),
        profile=QWEN36_27B_PROFILE,
        tool_call_parser="qwen3_coder_xml",
        receipt="reports/benchmarks/personal-intelligence-qwen3.6-27b-4bit.json",
        evidence=(
            "docs/engineering/performance/"
            "2026-09-16-personal-intelligence-remaining-builds-qualification.md"
        ),
    ),
    PersonalIntelligenceQualification(
        id="qwen3.8-27b-upstream-q4-v1",
        public_identities=frozenset({"mlx-community/Qwen3.8-27B-4bit"}),
        backing_identities=frozenset({"mlx-community/Qwen3.8-27B-4bit"}),
        profile=QWEN38_27B_PROFILE,
        tool_call_parser="qwen3_coder_xml",
        receipt=(
            "reports/benchmarks/personal-intelligence-qwen3.8-27b-upstream-q4.json"
        ),
        evidence=(
            "docs/engineering/performance/"
            "2026-09-16-personal-intelligence-remaining-builds-qualification.md"
        ),
    ),
    PersonalIntelligenceQualification(
        id="qwen3.8-27b-rapid-mtp-q4-v1",
        public_identities=frozenset({"qwen3.8-27b-4bit"}),
        backing_identities=frozenset({"rapid-mlx/Qwen3.8-27B-4bit-MTP-MLX"}),
        profile=QWEN38_27B_PROFILE,
        tool_call_parser="qwen3_coder_xml",
        receipt="reports/benchmarks/personal-intelligence-qwen3.8-27b-4bit.json",
        evidence=(
            "docs/engineering/performance/"
            "2026-09-16-personal-intelligence-remaining-builds-qualification.md"
        ),
    ),
    PersonalIntelligenceQualification(
        id="qwen3.8-27b-mixed-3.5bpw-v1",
        public_identities=frozenset({"qwen3.8-27b-mixed-3.5bpw"}),
        backing_identities=frozenset({"rapid-mlx/Qwen3.8-27B-mixed-3.5bpw-MLX"}),
        profile=QWEN38_27B_PROFILE,
        tool_call_parser="qwen3_coder_xml",
        receipt=(
            "reports/benchmarks/personal-intelligence-qwen3.8-27b-mixed-3.5bpw.json"
        ),
        evidence=(
            "docs/engineering/performance/"
            "2026-09-16-personal-intelligence-remaining-builds-qualification.md"
        ),
    ),
    PersonalIntelligenceQualification(
        id="qwen3.8-27b-fp16-mtp-v1",
        public_identities=frozenset({"qwen3.8-27b-4bit-fp16"}),
        backing_identities=frozenset({"rapid-mlx/Qwen3.8-27B-4bit-MTP-fp16-MLX"}),
        profile=QWEN38_27B_PROFILE,
        tool_call_parser="qwen3_coder_xml",
        receipt=("reports/benchmarks/personal-intelligence-qwen3.8-27b-4bit-fp16.json"),
        evidence=(
            "docs/engineering/performance/"
            "2026-09-16-personal-intelligence-remaining-builds-qualification.md"
        ),
    ),
    PersonalIntelligenceQualification(
        id="bonsai-27b-2bit-v1",
        public_identities=frozenset({"bonsai-27b-2bit"}),
        backing_identities=frozenset({"prism-ml/Ternary-Bonsai-27B-mlx-2bit"}),
        profile=BONSAI_27B_PROFILE,
        tool_call_parser="hermes",
        receipt="reports/benchmarks/personal-intelligence-bonsai-27b-2bit.json",
        evidence=(
            "docs/engineering/performance/"
            "2026-09-16-personal-intelligence-remaining-builds-qualification.md"
        ),
    ),
    PersonalIntelligenceQualification(
        id="ling-3.0-tiny-4bit-v1",
        public_identities=frozenset({"ling-3.0-tiny-4bit"}),
        backing_identities=frozenset({"rapid-mlx/Ling-3.0-tiny-MLX-4bit"}),
        profile=LING3_TINY_PROFILE,
        tool_call_parser="glm47",
        receipt="reports/benchmarks/personal-intelligence-ling-3.0-tiny-4bit.json",
        evidence=(
            "docs/engineering/performance/"
            "2026-09-16-personal-intelligence-remaining-builds-qualification.md"
        ),
    ),
    PersonalIntelligenceQualification(
        id="lfm2.5-1.2b-q4-v1",
        public_identities=frozenset(
            {"lfm2.5-1b-4bit", "mlx-community/LFM2.5-1.2B-Instruct-4bit"}
        ),
        backing_identities=frozenset({"mlx-community/LFM2.5-1.2B-Instruct-4bit"}),
        profile=LFM25_1B_PROFILE,
        tool_call_parser="lfm",
        receipt="reports/benchmarks/personal-intelligence-lfm2.5-1b-4bit.json",
        evidence=(
            "docs/engineering/performance/"
            "2026-09-15-personal-intelligence-top-model-qualification.md"
        ),
    ),
    PersonalIntelligenceQualification(
        id="minicpm5-2b-q4-v1",
        public_identities=frozenset({"minicpm5-2b-4bit", "openbmb/minicpm5-2b-mlx"}),
        backing_identities=frozenset({"openbmb/minicpm5-2b-mlx"}),
        profile=MINICPM5_2B_PROFILE,
        tool_call_parser="minicpm",
        receipt="reports/benchmarks/personal-intelligence-minicpm5-2b-4bit.json",
        evidence=(
            "docs/engineering/performance/2026-09-13-minicpm5-small-agent-harness-ab.md"
        ),
    ),
)


def _normalize_identity(identity: str) -> str:
    return identity.casefold().replace("_", "-")


def _qualification_index() -> dict[str, PersonalIntelligenceQualification]:
    result: dict[str, PersonalIntelligenceQualification] = {}
    for qualification in PERSONAL_INTELLIGENCE_QUALIFICATIONS:
        for identity in qualification.public_identities:
            normalized = _normalize_identity(identity)
            if normalized in result:
                raise RuntimeError(  # pragma: no cover - import-time invariant
                    f"duplicate Personal Intelligence identity: {identity}"
                )
            result[normalized] = qualification
    return result


_PERSONAL_INTELLIGENCE_BY_PUBLIC_ID = _qualification_index()

_MINICPM5_2B_CATALOG_IDENTITIES = frozenset(
    {
        "minicpm5-2b-4bit",
        "openbmb/minicpm5-2b",
        "openbmb/minicpm5-2b-mlx",
        "mlx-community/minicpm5-2b-8bit",
    }
)
_MINICPM5_2B_TRUSTED_REPO = re.compile(
    r"^(?:openbmb|mlx-community)/minicpm5-2b(?:-(?:mlx|4bit|8bit|bf16))?$"
)
_EXACT_RUNTIME_PROFILES = {
    _normalize_identity(identity): profile
    for profile, identities in (
        (
            QWEN35_4B_PROFILE,
            {
                "qwen3.5-4b-4bit",
                "mlx-community/Qwen3.5-4B-MLX-4bit",
                "qwen3.5-4b-8bit",
                "mlx-community/Qwen3.5-4B-8bit",
            },
        ),
        (
            QWEN35_9B_PROFILE,
            {
                "qwen3.5-9b-4bit",
                "mlx-community/Qwen3.5-9B-4bit",
                "qwen3.5-9b-8bit",
                "mlx-community/Qwen3.5-9B-8bit",
            },
        ),
        (
            LFM25_1B_PROFILE,
            {
                "lfm2.5-1b-4bit",
                "mlx-community/LFM2.5-1.2B-Instruct-4bit",
            },
        ),
        (
            QWEN36_27B_PROFILE,
            {
                "qwen3.6-27b-4bit",
                "mlx-community/Qwen3.6-27B-4bit",
            },
        ),
        (
            QWEN36_35B_PROFILE,
            {
                "qwen3.6-35b-4bit",
                "mlx-community/Qwen3.6-35B-A3B-4bit",
                "qwen3.6-35b-8bit",
                "mlx-community/Qwen3.6-35B-A3B-8bit",
            },
        ),
        (
            QWEN38_27B_PROFILE,
            {
                "qwen3.8-27b-4bit",
                "rapid-mlx/Qwen3.8-27B-4bit-MTP-MLX",
                "mlx-community/Qwen3.8-27B-4bit",
                "qwen3.8-27b-mixed-3.5bpw",
                "rapid-mlx/Qwen3.8-27B-mixed-3.5bpw-MLX",
                "qwen3.8-27b-4bit-fp16",
                "rapid-mlx/Qwen3.8-27B-4bit-MTP-fp16-MLX",
            },
        ),
        (
            BONSAI_27B_PROFILE,
            {
                "bonsai-27b-2bit",
                "prism-ml/Ternary-Bonsai-27B-mlx-2bit",
            },
        ),
        (
            QWEN3_CODER_30B_PROFILE,
            {
                "qwen3-coder-30b-4bit",
                "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit",
            },
        ),
        (
            LING3_TINY_PROFILE,
            {
                "ling-3.0-tiny-4bit",
                "rapid-mlx/Ling-3.0-tiny-MLX-4bit",
            },
        ),
        (
            GPT_OSS_20B_PROFILE,
            {
                "gpt-oss-20b",
                "mlx-community/gpt-oss-20b-MXFP4-Q8",
            },
        ),
    )
    for identity in identities
}


def _is_minicpm5_2b_config(config: dict[str, Any] | None) -> bool:
    """Recognize the released dense 2B checkpoint from loaded metadata.

    MiniCPM5 deliberately uses the generic ``LlamaForCausalLM`` architecture,
    so a local directory has no family-bearing architecture name. Match the
    complete stable shape instead of trusting directory or served-alias text.
    """

    if config is None:
        return False
    return all(
        config.get(key) == value
        for key, value in {
            "model_type": "llama",
            "hidden_size": 2048,
            "intermediate_size": 6144,
            "num_hidden_layers": 42,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "vocab_size": 130560,
        }.items()
    )


def resolve_agent_profile(
    model: str,
    *,
    model_config: dict[str, Any] | None = None,
    tool_call_parser: str | None = None,
) -> AgentProfile:
    """Return budgets from catalog identity or exact loaded-model metadata."""

    normalized = _normalize_identity(model)
    catalog_match = (
        normalized in _MINICPM5_2B_CATALOG_IDENTITIES
        or _MINICPM5_2B_TRUSTED_REPO.fullmatch(normalized) is not None
    )
    if catalog_match or (
        tool_call_parser == "minicpm" and _is_minicpm5_2b_config(model_config)
    ):
        return MINICPM5_2B_PROFILE
    if exact_profile := _EXACT_RUNTIME_PROFILES.get(normalized):
        return exact_profile
    if "minicpm5-2b" in normalized:
        # Preserve the old conservative limits for local names without
        # granting them a verified MiniCPM identity.
        return CONSERVATIVE_LOCAL_PROFILE
    return DEFAULT_PROFILE


def resolve_personal_intelligence_profile(
    model: str,
    *,
    backing_model: str | None = None,
    model_config: dict[str, Any] | None = None,
    tool_call_parser: str | None = None,
) -> AgentProfile | None:
    """Return the qualified Personal Intelligence harness, if any.

    ``resolve_agent_profile`` deliberately retains a bounded generic fallback
    for API callers.  The product surface is stricter: tool-call capability or
    the generic fallback cannot opt a model into Personal Intelligence.
    """

    qualification = resolve_personal_intelligence_qualification(
        model,
        backing_model=backing_model,
        model_config=model_config,
        tool_call_parser=tool_call_parser,
    )
    return qualification.profile if qualification is not None else None


def resolve_personal_intelligence_qualification(
    model: str,
    *,
    backing_model: str | None = None,
    model_config: dict[str, Any] | None = None,
    tool_call_parser: str | None = None,
) -> PersonalIntelligenceQualification | None:
    """Return the exact admitted artifact/parser/harness record, if any."""

    normalized = _normalize_identity(model)
    qualification = _PERSONAL_INTELLIGENCE_BY_PUBLIC_ID.get(normalized)
    backing_identity = model if backing_model is None else backing_model
    backing_normalized = _normalize_identity(backing_identity)
    if qualification is None or backing_normalized not in map(
        _normalize_identity, qualification.backing_identities
    ):
        return None
    # A known checkpoint served with its parser explicitly disabled or
    # overridden is not the pairing that passed qualification.
    if tool_call_parser != qualification.tool_call_parser:
        return None
    profile = resolve_agent_profile(
        model,
        model_config=model_config,
        tool_call_parser=tool_call_parser,
    )
    if profile.name != qualification.profile.name:
        return None  # pragma: no cover - qualification/profile table invariant
    return qualification
