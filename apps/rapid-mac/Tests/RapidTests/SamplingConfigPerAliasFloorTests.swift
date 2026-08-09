import Foundation
import Testing
@testable import Rapid

/// FU-3 (post-v0.7.19) — per-alias override plumbing for the
/// reasoning_content max_tokens floors. Yesterday's PR #318 fixed
/// the *symptom* (auto-bump under reasoning_parser != nil); FU-3
/// hardens the *contract* so a future heavy-reasoning alias whose
/// median trace exceeds 8 KB (e.g. a 70B reasoning model) can lift
/// the floor without a desktop code change. The plumbing lives in
/// ``ServerModelProfile`` (optional ``reasoningChatFloor`` /
/// ``reasoningToolsFloor``) and ``SamplingConfig``
/// (``effectiveChatFloor`` / ``effectiveToolsFloor`` resolvers).
///
/// What these tests pin:
///  * Default behaviour unchanged when the server omits the new
///    vendor fields (rapid-mlx ≤ 0.7.19 reality).
///  * Override behaviour works when a profile carries explicit
///    floors — both chat and tools paths.
///  * JSON decode tolerates absent ``reasoning_chat_floor`` /
///    ``reasoning_tools_floor`` keys (every existing alias today).
///  * ``clearActiveReasoningParser`` / ``resetToDefaults`` drop the
///    per-alias floors alongside the parser signal so an alias swap
///    can't carry a previous alias's floor into a new conversation.
///  * The resolver clamps to ``maxTokensRange`` so a hostile or
///    typo'd server value (negative, 0, 1 billion) can't escape the
///    same bounds an honest user-dragged slider obeys.
@MainActor
@Suite("FU-3: SamplingConfig per-alias reasoning floor overrides")
final class SamplingConfigPerAliasFloorTests {
    nonisolated(unsafe) private var createdSuiteNames: [String] = []

    deinit { TestDefaultsScope.cleanup(suiteNames: createdSuiteNames) }

    private func freshDefaults() -> UserDefaults {
        let name = TestDefaultsScope.mintSuiteName(prefix: "rapid-fu3-floor-test-")
        createdSuiteNames.append(name)
        let d = UserDefaults(suiteName: name)!
        d.removePersistentDomain(forName: name)
        return d
    }

    // MARK: - Default behaviour unchanged

    /// Pin that a profile with nil floors (every alias rapid-mlx
    /// 0.7.x ships today) keeps the global default constants —
    /// zero observable behaviour change for any alias today.
    @Test("Profile with nil floor overrides → effective floors == defaults (no behaviour change today)")
    func defaultFloorsUnchangedWhenProfileSilent() {
        let s = SamplingConfig(defaults: freshDefaults())
        let profile = ServerModelProfile(
            id: "vibethinker-3b-8bit",
            recommendedSampling: nil,
            isHybrid: false,
            isMoe: false,
            toolCallParser: "hermes",
            reasoningParser: "vibethinker",
            modality: "text"
            // reasoningChatFloor / reasoningToolsFloor omitted → nil
        )
        _ = s.applyServerProfile(profile)
        #expect(s.activeReasoningChatFloor == nil)
        #expect(s.activeReasoningToolsFloor == nil)
        #expect(s.effectiveChatFloor == SamplingConfig.defaultReasoningChatFloor)
        #expect(s.effectiveToolsFloor == SamplingConfig.defaultReasoningToolsFloor)
        #expect(s.effectiveChatFloor == 2_048)
        #expect(s.effectiveToolsFloor == 16_384)
    }

    /// Pin the canonical default-alias names alias the legacy
    /// constants — a future maintainer who renames one MUST update
    /// the other (the assertion exists so the build fails before
    /// the call sites do).
    @Test("defaultReasoning…Floor aliases match the legacy reasoning…Floor constants")
    func defaultAliasesMatchLegacyConstants() {
        #expect(SamplingConfig.defaultReasoningChatFloor == SamplingConfig.reasoningChatFloor)
        #expect(SamplingConfig.defaultReasoningToolsFloor == SamplingConfig.reasoningToolsFloor)
    }

    // MARK: - Override behaviour

    /// The core FU-3 plumbing check: a profile that ships
    /// ``reasoning_chat_floor: 1_024`` (e.g. a tiny reasoner whose
    /// trace fits in 1 KB) must end up driving
    /// ``effectiveChatFloor`` AND the auto-scale write.
    @Test("Profile with reasoningChatFloor=1024 → effectiveChatFloor==1024 and auto-scale honours it")
    func chatFloorOverrideShadowsDefault() {
        let s = SamplingConfig(defaults: freshDefaults())
        let profile = ServerModelProfile(
            id: "tiny-reasoner-1b",
            recommendedSampling: nil,
            isHybrid: false,
            isMoe: false,
            toolCallParser: "hermes",
            reasoningParser: "tiny-reasoner",
            modality: "text",
            reasoningChatFloor: 1_024,
            reasoningToolsFloor: nil
        )
        _ = s.applyServerProfile(profile)
        #expect(s.activeReasoningChatFloor == 1_024)
        #expect(s.effectiveChatFloor == 1_024,
                "per-alias chat-floor override must shadow the default 2,048")
        // Tools-side still uses the default since the override is nil.
        #expect(s.effectiveToolsFloor == SamplingConfig.defaultReasoningToolsFloor)
        // Fresh install path uses max(maxTokens, chat floor); with
        // a 1,024 override and baseline 4,096, the slider stays at
        // 4,096 — the floor is the MINIMUM, never a cap.
        #expect(s.maxTokens == SamplingConfig.maxTokensDefault)
        // The auto-bumped value goes through max() so a lower
        // alias-side floor never downshifts an already-higher slider.
        let chatBudget = s.effectiveMaxTokens(toolsEnabled: false)
        #expect(chatBudget == max(s.maxTokens, 1_024))
        #expect(chatBudget == s.maxTokens, "1,024 < 4,096 → no lift on chat path")
    }

    /// Symmetric: tools-side override drives ``effectiveToolsFloor``
    /// AND the request-time bump. Pin a heavier value (8,192) so
    /// the lift becomes observable on a fresh install (baseline
    /// max_tokens = 4,096 → 4,096 < 8,192 → bumped to 8,192).
    @Test("Profile with reasoningToolsFloor=8192 → tools-path effectiveMaxTokens lifts to 8,192")
    func toolsFloorOverrideLiftsToolsPath() {
        let s = SamplingConfig(defaults: freshDefaults())
        let profile = ServerModelProfile(
            id: "heavy-reasoner-70b-4bit",
            recommendedSampling: nil,
            isHybrid: false,
            isMoe: false,
            toolCallParser: "hermes",
            reasoningParser: "heavy-reasoner",
            modality: "text",
            reasoningChatFloor: nil,
            reasoningToolsFloor: 8_192
        )
        _ = s.applyServerProfile(profile)
        #expect(s.activeReasoningToolsFloor == 8_192)
        #expect(s.effectiveToolsFloor == 8_192)
        // Chat-side stays at the default since override is nil.
        #expect(s.effectiveChatFloor == SamplingConfig.defaultReasoningChatFloor)
        // Fresh install max_tokens = 4,096; tools-on lifts to 8,192.
        let toolsBudget = s.effectiveMaxTokens(toolsEnabled: true)
        #expect(toolsBudget == 8_192, "tools-path lift must use per-alias override")
        // Chat-side still at baseline (4,096 > 2,048 default chat floor).
        let chatBudget = s.effectiveMaxTokens(toolsEnabled: false)
        #expect(chatBudget == SamplingConfig.maxTokensDefault)
    }

    /// A profile that ships BOTH overrides — the most realistic
    /// shape a future heavy-reasoning alias would use. Verify the
    /// chat-floor lift fires (4,096 → 6,000) AND the tools-floor
    /// (8,192) overrides on the same SamplingConfig instance.
    @Test("Profile with BOTH floor overrides — auto-scale uses chat, tools-path uses tools")
    func bothFloorOverridesWireThroughIndependently() {
        let s = SamplingConfig(defaults: freshDefaults())
        let profile = ServerModelProfile(
            id: "heavy-reasoner-70b-4bit",
            recommendedSampling: nil,
            isHybrid: false,
            isMoe: false,
            toolCallParser: "hermes",
            reasoningParser: "heavy-reasoner",
            modality: "text",
            reasoningChatFloor: 6_000,
            reasoningToolsFloor: 8_192
        )
        _ = s.applyServerProfile(profile)
        #expect(s.activeReasoningChatFloor == 6_000)
        #expect(s.activeReasoningToolsFloor == 8_192)
        #expect(s.effectiveChatFloor == 6_000)
        #expect(s.effectiveToolsFloor == 8_192)
        // Auto-scale path: fresh install max_tokens = 4,096, chat
        // floor = 6,000 → persisted slider lifts to 6,000.
        #expect(s.maxTokens == 6_000,
                "auto-scale must use the per-alias chat floor when it exceeds the baseline")
        // Tools-side lift on top of the auto-scaled value.
        #expect(s.effectiveMaxTokens(toolsEnabled: true) == 8_192)
        // Chat-side equals the (already-lifted) persisted slider.
        #expect(s.effectiveMaxTokens(toolsEnabled: false) == 6_000)
    }

    // MARK: - Clamping defends against hostile / typo'd server values

    /// A profile that ships ``reasoning_chat_floor: -1`` or ``0``
    /// (typo / malicious server) must NOT escape the slider's own
    /// ``maxTokensRange`` lower bound. Same contract a user-dragged
    /// slider obeys.
    @Test("Hostile reasoning_chat_floor (negative / above range) clamps to maxTokensRange")
    func hostileChatFloorIsClamped() {
        let s = SamplingConfig(defaults: freshDefaults())
        let negative = ServerModelProfile(
            id: "broken-alias",
            recommendedSampling: nil,
            isHybrid: false,
            isMoe: false,
            toolCallParser: "hermes",
            reasoningParser: "broken-alias",
            modality: "text",
            reasoningChatFloor: -1,
            reasoningToolsFloor: nil
        )
        _ = s.applyServerProfile(negative)
        #expect(s.effectiveChatFloor == SamplingConfig.maxTokensRange.lowerBound,
                "-1 must clamp to the same lower bound a user-dragged slider obeys")

        s.clearActiveReasoningParser()
        let huge = ServerModelProfile(
            id: "ridiculous-alias",
            recommendedSampling: nil,
            isHybrid: false,
            isMoe: false,
            toolCallParser: "hermes",
            reasoningParser: "ridiculous-alias",
            modality: "text",
            reasoningChatFloor: 1_000_000,
            reasoningToolsFloor: 1_000_000
        )
        _ = s.applyServerProfile(huge)
        #expect(s.effectiveChatFloor == SamplingConfig.maxTokensRange.upperBound)
        #expect(s.effectiveToolsFloor == SamplingConfig.maxTokensRange.upperBound)
    }

    // MARK: - Alias-swap hygiene

    /// Alias A ships ``reasoning_chat_floor: 6_000``. The user
    /// swaps to alias B (no override). The captured chat floor
    /// from A MUST be dropped — otherwise B silently inherits 6,000.
    @Test("clearActiveReasoningParser drops per-alias floor overrides — no carry into next conversation")
    func clearDropsPerAliasFloors() {
        let s = SamplingConfig(defaults: freshDefaults())
        let aliasA = ServerModelProfile(
            id: "heavy-reasoner-70b-4bit",
            recommendedSampling: nil,
            isHybrid: false,
            isMoe: false,
            toolCallParser: "hermes",
            reasoningParser: "heavy-reasoner",
            modality: "text",
            reasoningChatFloor: 6_000,
            reasoningToolsFloor: 8_192
        )
        _ = s.applyServerProfile(aliasA)
        #expect(s.activeReasoningChatFloor == 6_000)
        #expect(s.activeReasoningToolsFloor == 8_192)

        s.clearActiveReasoningParser()
        #expect(s.activeReasoningChatFloor == nil)
        #expect(s.activeReasoningToolsFloor == nil)
        #expect(s.effectiveChatFloor == SamplingConfig.defaultReasoningChatFloor)
        #expect(s.effectiveToolsFloor == SamplingConfig.defaultReasoningToolsFloor)
    }

    /// Codex r1 P1 (PR #352): heavier-than-baseline auto-scale
    /// from alias A (e.g. ``reasoningChatFloor: 6_000``) leaves
    /// ``maxTokens = 6_000`` persisted in ``UserDefaults``. On
    /// alias swap to a non-reasoning alias B, ``maxTokens`` MUST
    /// revert to the baseline — otherwise B silently sends 6,000
    /// max_tokens even though the user never chose it. The pre-FU-3
    /// codebase never hit this (max(4096, 2048) == 4096 was a no-op);
    /// per-alias overrides break that invariant, so the clear path
    /// has to undo our own footprint explicitly.
    @Test("clearActiveReasoningParser reverts auto-scaled maxTokens above baseline — no carry to next alias")
    func clearRevertsAutoScaledMaxTokensAboveBaseline() {
        let defaults = freshDefaults()
        let s = SamplingConfig(defaults: defaults)
        let heavy = ServerModelProfile(
            id: "heavy-reasoner-70b-4bit",
            recommendedSampling: nil,
            isHybrid: false,
            isMoe: false,
            toolCallParser: "hermes",
            reasoningParser: "heavy-reasoner",
            modality: "text",
            reasoningChatFloor: 6_000,
            reasoningToolsFloor: 8_192
        )
        _ = s.applyServerProfile(heavy)
        #expect(s.maxTokens == 6_000, "precondition: auto-scale lifted past baseline")

        s.clearActiveReasoningParser()
        // The slider — and the persisted value — must revert to
        // the baseline. Otherwise alias B silently inherits 6,000.
        #expect(s.maxTokens == SamplingConfig.maxTokensDefault,
                "auto-scaled maxTokens leaked across clear() — alias B would send \(s.maxTokens) instead of \(SamplingConfig.maxTokensDefault)")
        #expect(defaults.object(forKey: "rapid.sampling.v0.maxTokens") as? Int == SamplingConfig.maxTokensDefault,
                "persisted UserDefaults must also revert — otherwise relaunch reads the stale 6,000")
    }

    /// Counter-test: a user who manually dragged the slider to a
    /// distinct value (e.g. 1,500 — well below the auto-scale gate's
    /// ``maxTokens == maxTokensDefault`` landmark) and then triggers
    /// a profile fetch — the slider must NOT be reverted by clear.
    /// The clear path is OUR auto-scale rollback only; explicit user
    /// intent is sacred.
    @Test("clearActiveReasoningParser preserves user-set maxTokens — no spurious revert")
    func clearDoesNotTouchUserSetMaxTokens() {
        let s = SamplingConfig(defaults: freshDefaults())
        s.maxTokens = 1_500
        // Apply a reasoning profile WITHOUT an over-baseline floor;
        // the user override gate suppresses auto-scale entirely.
        let profile = ServerModelProfile(
            id: "vibethinker-3b-8bit",
            recommendedSampling: nil,
            isHybrid: false,
            isMoe: false,
            toolCallParser: "hermes",
            reasoningParser: "vibethinker",
            modality: "text"
        )
        _ = s.applyServerProfile(profile)
        #expect(s.maxTokens == 1_500, "precondition: user override suppressed auto-scale")

        s.clearActiveReasoningParser()
        #expect(s.maxTokens == 1_500,
                "clear() stomped a user-set maxTokens — explicit user intent must survive an alias swap")
    }

    /// Same hygiene rule for ``resetToDefaults`` — the user hitting
    /// the "Reset" button must get the global floors back, not the
    /// last alias's override.
    @Test("resetToDefaults drops per-alias floor overrides alongside other reasoning bookkeeping")
    func resetDropsPerAliasFloors() {
        let s = SamplingConfig(defaults: freshDefaults())
        let profile = ServerModelProfile(
            id: "heavy-reasoner-70b-4bit",
            recommendedSampling: nil,
            isHybrid: false,
            isMoe: false,
            toolCallParser: "hermes",
            reasoningParser: "heavy-reasoner",
            modality: "text",
            reasoningChatFloor: 6_000,
            reasoningToolsFloor: 8_192
        )
        _ = s.applyServerProfile(profile)
        #expect(s.activeReasoningChatFloor == 6_000)

        s.resetToDefaults()
        #expect(s.activeReasoningChatFloor == nil)
        #expect(s.activeReasoningToolsFloor == nil)
        #expect(s.effectiveChatFloor == SamplingConfig.defaultReasoningChatFloor)
        #expect(s.effectiveToolsFloor == SamplingConfig.defaultReasoningToolsFloor)
    }

    // MARK: - JSON decode tolerance

    /// Every alias rapid-mlx 0.7.x ships today omits the new
    /// vendor fields. The decoder must accept the missing keys
    /// as ``nil`` — no throw, no decode error. This is the wire
    /// contract the desktop relies on for backward compatibility
    /// with every previously-released rapid-mlx server.
    @Test("Decoder accepts ServerModelProfile JSON missing both reasoning_chat_floor + reasoning_tools_floor")
    func decoderToleratesAbsentFloorKeys() throws {
        let json = """
        {
          "id": "qwen3.6-35b-4bit",
          "object": "model",
          "owned_by": "rapid-mlx",
          "reasoning_parser": "qwen3",
          "tool_call_parser": "hermes",
          "modality": "text"
        }
        """
        let profile = try JSONDecoder().decode(
            ServerModelProfile.self,
            from: Data(json.utf8)
        )
        #expect(profile.reasoningChatFloor == nil)
        #expect(profile.reasoningToolsFloor == nil)
        #expect(profile.reasoningParser == "qwen3")
    }

    /// Forward-compatible decode: a rapid-mlx version that DOES
    /// ship the new fields must round-trip cleanly. Pin the
    /// snake_case wire shape (``reasoning_chat_floor`` /
    /// ``reasoning_tools_floor``) so a future rename server-side
    /// fails the test loudly instead of silently dropping to nil.
    @Test("Decoder reads reasoning_chat_floor + reasoning_tools_floor from snake_case wire JSON")
    func decoderReadsFloorOverridesWhenPresent() throws {
        let json = """
        {
          "id": "heavy-reasoner-70b-4bit",
          "object": "model",
          "owned_by": "rapid-mlx",
          "reasoning_parser": "heavy-reasoner",
          "tool_call_parser": "hermes",
          "modality": "text",
          "reasoning_chat_floor": 6000,
          "reasoning_tools_floor": 8192
        }
        """
        let profile = try JSONDecoder().decode(
            ServerModelProfile.self,
            from: Data(json.utf8)
        )
        #expect(profile.reasoningChatFloor == 6_000)
        #expect(profile.reasoningToolsFloor == 8_192)
    }

    /// Partial vendor extension — only one floor populated. Defends
    /// against a regression that would require both keys to land
    /// before either took effect (would silently downshift a server
    /// that only meant to override the tools floor).
    @Test("Decoder accepts partial floor block — only reasoning_chat_floor populated")
    func decoderReadsPartialFloorBlock() throws {
        let json = """
        {
          "id": "tiny-reasoner-1b",
          "object": "model",
          "owned_by": "rapid-mlx",
          "reasoning_parser": "tiny-reasoner",
          "reasoning_chat_floor": 1024
        }
        """
        let profile = try JSONDecoder().decode(
            ServerModelProfile.self,
            from: Data(json.utf8)
        )
        #expect(profile.reasoningChatFloor == 1_024)
        #expect(profile.reasoningToolsFloor == nil)
    }

    // MARK: - Today's aliases.json on disk still decodes cleanly

    /// Smoke check that JSONDecoder still accepts the canonical
    /// ``{"id", "object", "owned_by", recommended_sampling, ...}``
    /// shape every existing alias today emits — i.e. the FU-3
    /// fields didn't accidentally land as required.
    @Test("Existing OpenAI-baseline + recommended_sampling profile still decodes (every alias today)")
    func existingAliasShapeStillDecodes() throws {
        let json = """
        {
          "id": "qwen3.5-9b-4bit",
          "object": "model",
          "created": 1750000000,
          "owned_by": "rapid-mlx",
          "recommended_sampling": {
            "temperature": 0.3,
            "top_p": 0.9
          },
          "is_hybrid": true,
          "is_moe": false,
          "tool_call_parser": "hermes",
          "reasoning_parser": "qwen3",
          "modality": "text"
        }
        """
        let profile = try JSONDecoder().decode(
            ServerModelProfile.self,
            from: Data(json.utf8)
        )
        #expect(profile.id == "qwen3.5-9b-4bit")
        #expect(profile.recommendedSampling?["temperature"] == 0.3)
        #expect(profile.reasoningParser == "qwen3")
        // The FU-3 fields are absent — pin nil, not a throw.
        #expect(profile.reasoningChatFloor == nil)
        #expect(profile.reasoningToolsFloor == nil)
    }
}
