import Foundation
import Testing
@testable import Rapid

/// Pin contract for ``ToolUseCapability`` — the desktop-side capability
/// map that decides whether the Tools chip is enabled or disabled for
/// a given rapid-mlx alias.
///
/// Background: ``/v1/models`` reports a ``tool_call_parser`` per
/// alias, but that field describes parser wiring, not whether the
/// model's weights actually emit ``<tool_call>`` tokens. Fuzz cycles
/// F-11-5 (phi-4-mini-reasoning), cycle-4 F-1 (hermes3-8b), and
/// cycle-9 F9-001 (llama3-1b) caught aliases that silently fail or
/// schema-leak. The capability map is the desktop's own truth so
/// we can disable the Tools UI on those aliases without waiting for
/// a rapid-mlx fix.
///
/// These tests pin:
///
///   * Each ``.broken`` entry resolves to ``.broken``, with quant-
///     sibling prefix coverage.
///   * Each ``.known`` entry resolves to ``.known``, including the
///     specific aliases the bench loop covers.
///   * Unknown / new aliases resolve to ``.unknown`` (no regression).
///   * The disabled-chip helper agrees with ``.broken``.
///   * The disabled-tooltip is non-empty and English-only (per
///     repo-hygiene rule — see ``MEMORY.md``).
@Suite("ToolUseCapability — desktop-side tool-call capability map")
struct ToolUseCapabilityTests {

    // MARK: - .broken bucket

    @Test("phi-4-mini-reasoning-4bit is .broken (F-11-5)")
    func phi4MiniReasoningIsBroken() {
        #expect(
            ToolUseCapability.confidence(for: "phi-4-mini-reasoning-4bit") == .broken,
            "phi-4-mini-reasoning emits no <tool_call> tokens for tools-eligible prompts (cycle-11 F-11-5)."
        )
    }

    @Test("phi-4-mini-reasoning-8bit also .broken — quant-sibling prefix coverage")
    func phi4MiniReasoningQuantSibling() {
        #expect(ToolUseCapability.confidence(for: "phi-4-mini-reasoning-8bit") == .broken)
    }

    @Test("hermes3-8b-4bit is .broken (cycle-4 F-1)")
    func hermes3_8bIsBroken() {
        #expect(
            ToolUseCapability.confidence(for: "hermes3-8b-4bit") == .broken,
            "hermes3-8b silent-degradation: auto tool_choice never emits <tool_call> (cycle-4 F-1)."
        )
    }

    @Test("hermes3-8b without quant suffix also .broken — bare-alias coverage")
    func hermes3_8bBareAlias() {
        #expect(ToolUseCapability.confidence(for: "hermes3-8b") == .broken)
    }

    @Test("llama3-1b-4bit is .broken (cycle-9 F9-001)")
    func llama3_1bIsBroken() {
        #expect(
            ToolUseCapability.confidence(for: "llama3-1b-4bit") == .broken,
            "llama3-1b schema-leak: parser passes JSON-Schema wrapper into function.arguments verbatim (cycle-9 F9-001)."
        )
    }

    @Test("llama3-1b-8bit also .broken — quant-sibling prefix coverage")
    func llama3_1bQuantSibling() {
        #expect(ToolUseCapability.confidence(for: "llama3-1b-8bit") == .broken)
    }

    @Test("All brokenPrefixes round-trip to .broken")
    func everyBrokenPrefixResolvesToBroken() {
        for prefix in ToolUseCapability.brokenPrefixes {
            #expect(
                ToolUseCapability.confidence(for: prefix) == .broken,
                "Prefix '\(prefix)' is registered as broken but confidence(for:) doesn't return .broken — capability map is internally inconsistent."
            )
        }
    }

    // MARK: - .known bucket

    @Test("qwen3.5-4b is .known (bench backbone)")
    func qwen35_4bIsKnown() {
        #expect(ToolUseCapability.confidence(for: "qwen3.5-4b") == .known)
    }

    @Test("qwen3.5-4b-4bit is .known — quant-sibling prefix coverage")
    func qwen35_4bQuantSibling() {
        #expect(ToolUseCapability.confidence(for: "qwen3.5-4b-4bit") == .known)
    }

    @Test("qwen3.6-35b-a3b-4bit is .known — hybrid+MoE family covered")
    func qwen36_35bIsKnown() {
        #expect(ToolUseCapability.confidence(for: "qwen3.6-35b-a3b-4bit") == .known)
    }

    @Test("Ornith-1.5 official sizes are .known after Studio tool-call dogfood")
    func ornithOfficialSizesAreKnown() {
        #expect(ToolUseCapability.confidence(for: "ornith-1.5-9b-bf16") == .known)
        #expect(ToolUseCapability.confidence(for: "ornith-1.5-35b-a3b-bf16") == .known)
    }

    @Test("Unverified Ornith-1.5 size shapes remain .unknown")
    func ornithUnverifiedSizesAreUnknown() {
        #expect(ToolUseCapability.confidence(for: "ornith-1.5-19b") == .unknown)
        #expect(ToolUseCapability.confidence(for: "ornith-1.5-135b") == .unknown)
    }

    @Test("llama3-3b-4bit is .known — smallest empirically-good llama")
    func llama3_3bIsKnown() {
        #expect(ToolUseCapability.confidence(for: "llama3-3b-4bit") == .known)
    }

    @Test("llama-3.1-8b-4bit is .known (cycle-6 F-1/F-2 confirm tool_call path works)")
    func llama31_8bIsKnown() {
        #expect(ToolUseCapability.confidence(for: "llama-3.1-8b-4bit") == .known)
    }

    @Test("glm4.7-4bit is .known — glm47 parser, benched in eval suite")
    func glm47IsKnown() {
        #expect(ToolUseCapability.confidence(for: "glm4.7-4bit") == .known)
    }

    @Test("gpt-oss-20b-4bit is .known (cycle-7 clean on safety + tool-edge)")
    func gptOss20bIsKnown() {
        #expect(ToolUseCapability.confidence(for: "gpt-oss-20b-4bit") == .known)
    }

    @Test("gemma-4-26b-4bit is .known (cycle-6 cross-model walk)")
    func gemma4_26bIsKnown() {
        #expect(ToolUseCapability.confidence(for: "gemma-4-26b-4bit") == .known)
    }

    @Test("All knownPrefixes round-trip to .known")
    func everyKnownPrefixResolvesToKnown() {
        for prefix in ToolUseCapability.knownPrefixes {
            #expect(
                ToolUseCapability.confidence(for: prefix) == .known,
                "Prefix '\(prefix)' is registered as known but confidence(for:) doesn't return .known — capability map is internally inconsistent."
            )
        }
    }

    // MARK: - .unknown bucket (default for new aliases)

    @Test("Empty alias is .unknown (don't crash, don't lock out)")
    func emptyAliasIsUnknown() {
        #expect(ToolUseCapability.confidence(for: "") == .unknown)
    }

    @Test("Brand-new unbenched alias defaults to .unknown — no regression on future models")
    func unknownAliasDefaultsToUnknown() {
        // A hypothetical future alias the loop hasn't benched yet.
        // Default MUST be .unknown so we never accidentally hide
        // a working tool-capable model from the user.
        #expect(ToolUseCapability.confidence(for: "future-model-7b-mxfp4") == .unknown)
    }

    @Test("qwen3-0.6b-4bit is .unknown (sub-1B; bench loop never covers, picker filter handles separately)")
    func qwen06bIsUnknown() {
        // qwen3-0.6b-4bit is still a downloadable catalog alias (it is
        // no longer the first-run starter — that is bonsai-1.7b-2bit).
        // We deliberately don't mark qwen3-0.6b as .broken even
        // though it would fail at tools in practice — the
        // ModelPickerVisibility sub-1B filter handles surfacing
        // the alias in the picker. .broken is reserved for aliases
        // with first-hand fuzz evidence; leaving qwen3-0.6b as
        // .unknown keeps the capability map's "broken means
        // empirically observed" rule clean.
        #expect(ToolUseCapability.confidence(for: "qwen3-0.6b-4bit") == .unknown)
    }

    @Test("First-run starter bonsai-1.7b-2bit is .known (tool-capable — the whole point of the swap)")
    func starterBonsaiIsKnown() {
        // The starter must clear the ``.known`` gate so the empty-state
        // capability chip row renders and the first-run user sees a
        // genuinely tool-capable model (6/6 clean tool_calls, hermes
        // parser; rapid-mlx PR #1092). If this regresses to .unknown
        // the chip row would be hidden and the swap loses its point.
        #expect(ToolUseCapability.confidence(for: "bonsai-1.7b-2bit") == .known)
    }

    @Test("Bonsai tool evidence is checkpoint-specific")
    func bonsaiEvidenceDoesNotLeakAcrossSizes() {
        #expect(ToolUseCapability.confidence(for: "bonsai-8b-2bit") == .broken)
        #expect(ToolUseCapability.confidence(for: "bonsai-8b-4bit") == .broken)
        #expect(ToolUseCapability.confidence(for: "bonsai-4b-unpacked") == .unknown)
    }

    // MARK: - shouldDisableToolsChip helper

    @Test("shouldDisableToolsChip is true for every .broken alias")
    func disableHelperAgreesWithBroken() {
        for prefix in ToolUseCapability.brokenPrefixes {
            #expect(
                ToolUseCapability.shouldDisableToolsChip(alias: prefix),
                "shouldDisableToolsChip(alias:) must return true for broken prefix '\(prefix)'."
            )
        }
    }

    @Test("shouldDisableToolsChip is false for every .known alias")
    func disableHelperFalseForKnown() {
        for prefix in ToolUseCapability.knownPrefixes {
            #expect(
                !ToolUseCapability.shouldDisableToolsChip(alias: prefix),
                "shouldDisableToolsChip(alias:) must return false for known prefix '\(prefix)' — would regress on a working model."
            )
        }
    }

    @Test("shouldDisableToolsChip is false for .unknown (no regression on unbenched aliases)")
    func disableHelperFalseForUnknown() {
        #expect(!ToolUseCapability.shouldDisableToolsChip(alias: "future-model-7b-mxfp4"))
        #expect(!ToolUseCapability.shouldDisableToolsChip(alias: ""))
    }

    // MARK: - regression guard against stale annotations

    @Test("brokenPrefixes and knownPrefixes never overlap")
    func brokenAndKnownAreDisjoint() {
        // If a prefix is in both lists the priority-1 broken match
        // wins, but the overlap is a sign someone forgot to remove
        // an entry when promoting it. Catch it explicitly.
        let brokenSet = Set(ToolUseCapability.brokenPrefixes)
        let knownSet = Set(ToolUseCapability.knownPrefixes)
        let intersection = brokenSet.intersection(knownSet)
        #expect(
            intersection.isEmpty,
            "Capability map has prefixes registered as BOTH broken and known: \(intersection.sorted()). Pick one bucket."
        )
    }

    @Test("brokenPrefixes are non-empty and lowercased")
    func brokenPrefixesAreWellFormed() {
        for prefix in ToolUseCapability.brokenPrefixes {
            #expect(!prefix.isEmpty, "Empty broken prefix would match every alias.")
            #expect(prefix == prefix.localizedLowercase, "Broken prefix '\(prefix)' is not lowercased; matcher does case-insensitive compare on lowercased needle, so a mixed-case entry wastes a slot.")
        }
    }

    @Test("knownPrefixes are non-empty and lowercased")
    func knownPrefixesAreWellFormed() {
        for prefix in ToolUseCapability.knownPrefixes {
            #expect(!prefix.isEmpty, "Empty known prefix would match every alias.")
            #expect(prefix == prefix.localizedLowercase, "Known prefix '\(prefix)' is not lowercased; matcher does case-insensitive compare on lowercased needle, so a mixed-case entry wastes a slot.")
        }
    }

    @Test("Disabled-tools tooltip is non-empty English copy")
    func disabledTooltipShape() {
        let tip = ToolUseCapability.disabledToolsTooltip
        #expect(!tip.isEmpty, "Tooltip is the user-facing 'why is this off' signal; empty would leave the disabled chip mysterious.")
        // Repo-hygiene rule (MEMORY.md): all in-repo UI copy is
        // English-only. Cheapest faithful check is "no CJK
        // codepoints" — the ranges below cover Hiragana/Katakana/
        // CJK Unified Ideographs which would catch the most likely
        // accidental paste.
        let cjkRanges: [ClosedRange<Unicode.Scalar>] = [
            // Hiragana 3040..309F
            Unicode.Scalar(0x3040)!...Unicode.Scalar(0x309F)!,
            // Katakana 30A0..30FF
            Unicode.Scalar(0x30A0)!...Unicode.Scalar(0x30FF)!,
            // CJK Unified Ideographs 4E00..9FFF
            Unicode.Scalar(0x4E00)!...Unicode.Scalar(0x9FFF)!,
        ]
        for scalar in tip.unicodeScalars {
            for range in cjkRanges {
                #expect(
                    !range.contains(scalar),
                    "Tooltip contains CJK codepoint U+\(String(scalar.value, radix: 16, uppercase: true)) — repo rule: UI copy is English-only."
                )
            }
        }
    }

    // MARK: - case sensitivity

    @Test("Match is case-insensitive — defensive against upstream casing change")
    func matchIsCaseInsensitive() {
        // rapid-mlx alias keys are lowercase ASCII today; the
        // matcher lowercases before comparing so a hypothetical
        // mixed-case alias in some future picker render still
        // resolves correctly.
        #expect(ToolUseCapability.confidence(for: "Phi-4-Mini-Reasoning-4bit") == .broken)
        #expect(ToolUseCapability.confidence(for: "QWEN3.5-4B") == .known)
    }
}
