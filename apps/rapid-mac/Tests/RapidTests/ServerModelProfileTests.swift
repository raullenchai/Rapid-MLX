import Foundation
import Testing
@testable import Rapid

/// Pin the wire contract for ``ServerModelProfile`` decoding +
/// ``SamplingConfig.applyServerProfile`` gating. Both halves are
/// pure: the decoder operates on raw JSON, the apply method takes
/// a profile and mutates the SamplingConfig — neither needs a
/// network round-trip, so these tests run in milliseconds and pin
/// the exact server / desktop contract Rapid-MLX 0.7.4+ ships.
@MainActor
@Suite("ServerModelProfile decode + applyServerProfile gating")
final class ServerModelProfileTests {
    /// See ``TestDefaultsScope`` + issue #139 — RAII teardown for
    /// the ``UserDefaults(suiteName:)`` plists this suite mints.
    nonisolated(unsafe) private var createdSuiteNames: [String] = []

    deinit { TestDefaultsScope.cleanup(suiteNames: createdSuiteNames) }

    private func freshDefaults() -> UserDefaults {
        let name = TestDefaultsScope.mintSuiteName(prefix: "rapid-server-profile-test-")
        createdSuiteNames.append(name)
        let d = UserDefaults(suiteName: name)!
        d.removePersistentDomain(forName: name)
        return d
    }

    // MARK: - Decoding

    @Test("Full vendor-extension response decodes every field")
    func decodesFullResponse() throws {
        let json = """
        {
          "id": "qwen3.5-9b-4bit",
          "object": "model",
          "created": 1750000000,
          "owned_by": "rapid-mlx",
          "recommended_sampling": {
            "temperature": 0.3,
            "top_p": 0.9,
            "repetition_penalty": 1.1
          },
          "is_hybrid": true,
          "is_moe": false,
          "tool_call_parser": "hermes",
          "reasoning_parser": "qwen3",
          "capabilities": ["text", "vision", "tools"],
          "serving_lane": "vision",
          "serving_lane_reason": "vision_hybrid_runtime_supported",
          "modality": "text"
        }
        """
        let profile = try JSONDecoder().decode(
            ServerModelProfile.self,
            from: Data(json.utf8)
        )
        #expect(profile.id == "qwen3.5-9b-4bit")
        #expect(profile.recommendedSampling?["temperature"] == 0.3)
        #expect(profile.recommendedSampling?["top_p"] == 0.9)
        #expect(profile.recommendedSampling?["repetition_penalty"] == 1.1)
        #expect(profile.isHybrid == true)
        #expect(profile.isMoe == false)
        #expect(profile.toolCallParser == "hermes")
        #expect(profile.reasoningParser == "qwen3")
        #expect(profile.capabilities == ["text", "vision", "tools"])
        #expect(profile.servingLane == "vision")
        #expect(profile.servingLaneReason == "vision_hybrid_runtime_supported")
        #expect(profile.modality == "text")
    }

    @Test("Older Rapid-MLX (no vendor extensions) decodes with nil fields — backward compat")
    func decodesLegacyOpenAIShape() throws {
        // What Rapid-MLX <0.7.4 returns: pure OpenAI shape, no
        // vendor extensions. The decoder must NOT throw — every
        // extension field must round-trip to nil.
        let json = """
        {
          "id": "qwen3.5-9b-4bit",
          "object": "model",
          "created": 1750000000,
          "owned_by": "rapid-mlx"
        }
        """
        let profile = try JSONDecoder().decode(
            ServerModelProfile.self,
            from: Data(json.utf8)
        )
        #expect(profile.id == "qwen3.5-9b-4bit")
        #expect(profile.recommendedSampling == nil)
        #expect(profile.isHybrid == nil)
        #expect(profile.toolCallParser == nil)
        #expect(profile.capabilities == nil)
        #expect(profile.servingLane == nil)
        #expect(profile.servingLaneReason == nil)
    }

    @Test("Live text lane overrides a vision-capable catalog and explains why")
    func liveTextLaneBlocksPhotos() {
        let availability = ImageInputAvailability.resolve(
            fallbackSupportsImageInput: true,
            profile: ServerModelProfile(
                id: "model",
                capabilities: ["text", "tools"],
                servingLane: "text",
                servingLaneReason: "vision_weights_unavailable"
            )
        )
        #expect(!availability.isAvailable)
        #expect(availability.unavailableMessage?.contains("vision features") == true)
    }

    @Test("Live vision lane enables photos even when an old catalog fallback is false")
    func liveVisionLaneEnablesPhotos() {
        let availability = ImageInputAvailability.resolve(
            fallbackSupportsImageInput: false,
            profile: ServerModelProfile(
                id: "model",
                capabilities: ["text", "vision"],
                servingLane: "vision",
                servingLaneReason: "vision_supported"
            )
        )
        #expect(availability == ImageInputAvailability(
            isAvailable: true,
            unavailableMessage: nil
        ))
    }

    @Test("Legacy model profile preserves the existing launch capability fallback")
    func legacyProfileUsesFallback() {
        let legacy = ServerModelProfile(id: "model")
        #expect(ImageInputAvailability.resolve(
            fallbackSupportsImageInput: true,
            profile: legacy
        ).isAvailable)
        #expect(!ImageInputAvailability.resolve(
            fallbackSupportsImageInput: false,
            profile: legacy
        ).isAvailable)
    }

    // MARK: - Auto-fallback lane reasons
    //
    // Every reason below is emitted with `auto_text_fallback` by the engine:
    // the checkpoint IS vision-capable, but its vision lane was not admitted.
    // The generic fallback copy names a remedy the user has already applied
    // ("choose a vision-capable model"), so each needs its own case. The
    // shared assertion is therefore that the copy is NOT the generic string.

    /// The copy every auto-fallback reason must avoid.
    private static let genericLaneCopy =
        "This model is running text-only. Photos need a vision-capable model."

    /// Resolve a text lane for a vision-capable checkpoint downgraded by `reason`.
    private func downgradedVisionLane(reason: String) -> ImageInputAvailability {
        ImageInputAvailability.resolve(
            fallbackSupportsImageInput: true,
            profile: ServerModelProfile(
                id: "model",
                capabilities: ["text", "vision"],
                servingLane: "text",
                servingLaneReason: reason
            )
        )
    }

    @Test("Vision that does not fit in memory blames memory, not a missing capability")
    func visionMemoryInsufficientExplainsTheMemoryLimit() {
        let availability = downgradedVisionLane(reason: "vision_memory_insufficient")
        let message = availability.unavailableMessage ?? ""
        #expect(!availability.isAvailable)
        // The remedy must be memory-shaped. The generic copy's "choose a
        // vision-capable model" sends a user who is already running one after
        // the wrong fix — that is the bug this case exists to prevent.
        #expect(message.contains("memory"))
        // The gate is a per-alias physical-RAM floor (`vision_min_memory_gb`),
        // not model size: a smaller quant of the same model hits the identical
        // floor and freeing memory changes nothing. The only remedy the user
        // can apply is a different vision-capable model.
        #expect(message.contains("different vision-capable model"))
        #expect(message != Self.genericLaneCopy)
    }

    @Test("An unsupported vision runtime says so instead of falling through")
    func visionHybridRuntimeUnsupportedExplainsTheRuntime() {
        let availability = downgradedVisionLane(reason: "vision_hybrid_runtime_unsupported")
        let message = availability.unavailableMessage ?? ""
        #expect(!availability.isAvailable)
        #expect(message.contains("runtime"))
        #expect(message != Self.genericLaneCopy)
    }

    @Test("Speculative decoding downgrade names the decoder, not the model's capability")
    func speculativeDecodeDowngradeExplainsTheDecoder() {
        let availability = downgradedVisionLane(reason: "text_lane_speculative_decode")
        let message = availability.unavailableMessage ?? ""
        #expect(!availability.isAvailable)
        // This downgrade is a throughput tradeoff, not a capability limit, and
        // it is the one reason here the user can undo directly.
        #expect(message.contains("speculative decoding"))
        // Settings paths use the app-wide "Settings → X" arrow convention.
        #expect(message.contains("Settings → Performance"))
        #expect(message != Self.genericLaneCopy)
    }

    @Test("Registry-pinned text-only sends the user to the model picker")
    func textLaneForcedPointsAtTheModelPicker() {
        // The engine emits `text_lane_forced`; the app matched a string the
        // engine never sent, so this copy was unreachable until it was fixed.
        //
        // The CLI reaches this via --no-mllm, but the GUI has no such switch:
        // in-app it only comes from an alias pinned `is_text_only`, which the
        // user cannot change. Choosing another model is the real remedy.
        let availability = downgradedVisionLane(reason: "text_lane_forced")
        let message = availability.unavailableMessage ?? ""
        #expect(!availability.isAvailable)
        #expect(message.contains("text-only"))
        #expect(message.contains("vision-capable model"))
        #expect(message != Self.genericLaneCopy)
    }

    /// Reasons the user can undo from the app without changing model.
    ///
    /// Speculative decoding is a launch setting reachable in Settings, and the
    /// checkpoint underneath is fully vision-capable. Copy that recommends a
    /// vision-capable model would send the user shopping for something they
    /// are already running — the same wrong-remedy failure that made the
    /// memory case worth fixing.
    ///
    /// `text_lane_forced` is deliberately absent: it is operator-reversible on
    /// the CLI but not in this app, so for GUI users the model picker IS the
    /// remedy. See ``textLaneForcedPointsAtTheModelPicker``.
    @Test(
        "A downgrade the user can reverse must not recommend a different model",
        arguments: ["text_lane_speculative_decode"]
    )
    func reversibleDowngradesPointAtTheSetting(reason: String) {
        let message = downgradedVisionLane(reason: reason).unavailableMessage ?? ""
        #expect(!message.contains("vision-capable model"))
        // ...and must instead name the switch to flip. Matched loosely so the
        // copy can read naturally ("turn it off") without failing the intent.
        let lowered = message.lowercased()
        #expect(lowered.contains("turn") && lowered.contains("off"))
    }

    @Test("An unknown lane reason still degrades to the generic copy")
    func unknownLaneReasonKeepsTheGenericCopy() {
        // The default arm stays reachable for sidecars newer than the app.
        let availability = downgradedVisionLane(reason: "reason_from_a_future_sidecar")
        #expect(!availability.isAvailable)
        #expect(availability.unavailableMessage == Self.genericLaneCopy)
    }

    @Test("Every serving-lane reason maps to its stable photo-hint catalog key")
    func laneReasonsMapToCatalogKeys() {
        let cases: [(String?, ImageInputAvailability.PhotoHint)] = [
            ("text_lane_forced", .textLaneForced),
            ("text_lane_speculative_decode", .speculativeDecode),
            ("vision_memory_insufficient", .visionMemoryInsufficient),
            ("vision_hybrid_runtime_unsupported", .visionRuntimeUnsupported),
            ("vision_architecture_unavailable", .visionFeaturesUnavailable),
            ("vision_hybrid_cache_unsupported", .visionFeaturesUnavailable),
            ("vision_weights_unavailable", .visionFeaturesUnavailable),
            ("text_checkpoint", .textCheckpoint),
            ("reason_from_a_future_sidecar", .genericTextLane),
            (nil, .genericTextLane)
        ]

        for (reason, expected) in cases {
            #expect(
                ImageInputAvailability.photoHint(for: reason) == expected,
                "Unexpected catalog key for serving-lane reason \(reason ?? "nil")"
            )
        }
    }

    @Test("A replacement sidecar invalidates the selected-model profile task")
    func selectedProfileTracksServerSession() {
        let first = SelectedModelProfileKey(
            alias: "model", isResident: true, port: 8000, bearer: "session-a"
        )
        let replacement = SelectedModelProfileKey(
            alias: "model", isResident: true, port: 8000, bearer: "session-b"
        )
        #expect(first != replacement)
    }

    @Test("Selected live profile is retried on the existing residency cadence")
    func selectedProfileRetriesAfterInitialFailure() throws {
        let source = try String(
            contentsOf: URL(fileURLWithPath: #filePath)
                .deletingLastPathComponent()
                .deletingLastPathComponent()
                .deletingLastPathComponent()
                .appendingPathComponent("Sources/Rapid/UI/ContentView.swift"),
            encoding: .utf8
        )
        let residencyRefresh = try #require(source.range(of: "await server.refreshResidency()"))
        let retry = try #require(
            source.range(
                of: "await refreshSelectedModelProfile(for: alias)",
                range: residencyRefresh.lowerBound..<source.endIndex
            )
        )
        #expect(retry.lowerBound > residencyRefresh.lowerBound)
    }

    @Test("Partial sampling block — only some keys populated")
    func decodesPartialSampling() throws {
        // aliases.json may set only ``temperature`` for some
        // families. The decoder must accept this without surfacing
        // a parse error; missing keys stay absent in the dict.
        let json = """
        {
          "id": "qwen3.6-35b-4bit",
          "object": "model",
          "owned_by": "rapid-mlx",
          "recommended_sampling": { "temperature": 0.5 }
        }
        """
        let profile = try JSONDecoder().decode(
            ServerModelProfile.self,
            from: Data(json.utf8)
        )
        #expect(profile.recommendedSampling?["temperature"] == 0.5)
        #expect(profile.recommendedSampling?["top_p"] == nil)
        #expect(profile.recommendedSampling?.count == 1)
    }

    // MARK: - SamplingConfig.applyServerProfile gating

    @Test("Fresh SamplingConfig applies curated profile; values land + boolean reports true")
    func appliesWhenAtDefaults() {
        let s = SamplingConfig(defaults: freshDefaults())
        let profile = ServerModelProfile(
            id: "qwen3.5-9b-4bit",
            recommendedSampling: [
                "temperature": 0.3,
                "top_p": 0.9,
                "repetition_penalty": 1.15
            ],
            isHybrid: true,
            isMoe: false,
            toolCallParser: "hermes",
            reasoningParser: "qwen3",
            modality: "text"
        )
        let applied = s.applyServerProfile(profile)
        #expect(applied)
        #expect(s.temperature == 0.3)
        #expect(s.topP == 0.9)
        #expect(s.repetitionPenalty == 1.15)
        // isAtDefaults flips to false because we just wrote non-default values.
        #expect(!s.isAtDefaults)
    }

    @Test("User who has touched a slider keeps their override — server profile MUST NOT clobber")
    func skipsWhenUserOverrode() {
        let s = SamplingConfig(defaults: freshDefaults())
        // Simulate a user dragging the temperature slider AWAY from default.
        s.temperature = 0.42
        #expect(!s.isAtDefaults)
        let profile = ServerModelProfile(
            id: "qwen3.5-9b-4bit",
            recommendedSampling: ["temperature": 0.3],
            isHybrid: true,
            isMoe: nil,
            toolCallParser: nil,
            reasoningParser: nil,
            modality: nil
        )
        let applied = s.applyServerProfile(profile)
        #expect(!applied, "user's manual override must win — server profile skipped")
        #expect(s.temperature == 0.42, "user's value must survive the call")
    }

    @Test("Profile with no recommended_sampling block reports false + does not flip isAtDefaults")
    func skipsWhenNoSamplingBlock() {
        let s = SamplingConfig(defaults: freshDefaults())
        let profile = ServerModelProfile(
            id: "qwen3.5-4b-4bit",
            recommendedSampling: nil,
            isHybrid: true,
            isMoe: nil,
            toolCallParser: "hermes",
            reasoningParser: "qwen3",
            modality: "text"
        )
        let applied = s.applyServerProfile(profile)
        #expect(!applied)
        #expect(s.isAtDefaults, "config must stay at v0.4.12 defaults")
    }

    @Test("Out-of-range server value clamps to the slider range, not the fallback")
    func clampsOutOfRangeServerValue() {
        // A profile that sets temperature=5.0 (above the slider max of 2.0)
        // must clamp to 2.0, not silently fall back to the default 0.7.
        // Clamping matches the same defensive guard the load path uses.
        let s = SamplingConfig(defaults: freshDefaults())
        let profile = ServerModelProfile(
            id: "weird-alias",
            recommendedSampling: ["temperature": 5.0],
            isHybrid: nil,
            isMoe: nil,
            toolCallParser: nil,
            reasoningParser: nil,
            modality: nil
        )
        _ = s.applyServerProfile(profile)
        #expect(s.temperature == SamplingConfig.temperatureRange.upperBound)
    }

    @Test("Partial server profile only writes the keys it provides")
    func partialProfilePartialApply() {
        // A profile with only temperature must leave topP and
        // repetitionPenalty at their v0.4.12 defaults — those are
        // safe values and there's no signal in the absence to
        // override them.
        let s = SamplingConfig(defaults: freshDefaults())
        let profile = ServerModelProfile(
            id: "qwen3.6-35b-4bit",
            recommendedSampling: ["temperature": 0.4],
            isHybrid: true,
            isMoe: nil,
            toolCallParser: nil,
            reasoningParser: nil,
            modality: nil
        )
        let applied = s.applyServerProfile(profile)
        #expect(applied)
        #expect(s.temperature == 0.4)
        #expect(s.topP == SamplingConfig.topPDefault)
        #expect(s.repetitionPenalty == SamplingConfig.repetitionPenaltyDefault)
    }

    // MARK: - Cycle-3: reasoning-alias max_tokens floors

    /// Pins the structural contract that ``reasoning_parser != null``
    /// is the signal we use — the test reads the constant rather than
    /// hard-coding the parser name list so adding a new parser server-
    /// side ("granite4_reason", "phi5", …) needs zero desktop work.
    @Test("Cycle-3: reasoning floor constants exist + are non-decreasing (chat ≤ tools)")
    func reasoningFloorConstantsAreConsistent() {
        #expect(SamplingConfig.reasoningChatFloor == 2_048)
        #expect(SamplingConfig.reasoningToolsFloor == 16_384)
        #expect(SamplingConfig.reasoningChatFloor <= SamplingConfig.reasoningToolsFloor,
                "tools floor must be at least the chat floor — tools-heavy prompts emit more reasoning tokens before the first call")
        // The floor spent its whole life equal to maxTokensDefault, which
        // made `max(maxTokens, effectiveToolsFloor)` a no-op and lifted
        // nobody. A floor at or below the baseline is not a floor.
        #expect(SamplingConfig.reasoningToolsFloor > SamplingConfig.maxTokensDefault,
                "a tools floor that does not exceed the default budget can never raise it — see the 5-run repro in SamplingConfig")
        #expect(SamplingConfig.reasoningToolsFloor <= SamplingConfig.maxTokensRange.upperBound,
                "the floor must be reachable through the same range an explicit user choice obeys")
    }

    /// Non-reasoning alias path — the existing 4,096 default must NOT
    /// be perturbed when ``reasoning_parser == nil``. Defends against
    /// a regression that would auto-bump every alias indiscriminately.
    @Test("Cycle-3: non-reasoning alias keeps the v0.4.12 4,096 default + activeReasoningParser stays nil")
    func nonReasoningAliasUnchanged() {
        let s = SamplingConfig(defaults: freshDefaults())
        let profile = ServerModelProfile(
            id: "hermes3-8b-4bit",
            recommendedSampling: nil,
            isHybrid: false,
            isMoe: false,
            toolCallParser: "hermes",
            reasoningParser: nil,
            modality: "text"
        )
        _ = s.applyServerProfile(profile)
        #expect(s.maxTokens == SamplingConfig.maxTokensDefault)
        #expect(s.activeReasoningParser == nil)
        // Tools-on or tools-off, a non-reasoning alias never bumps.
        #expect(s.effectiveMaxTokens(toolsEnabled: false) == SamplingConfig.maxTokensDefault)
        #expect(s.effectiveMaxTokens(toolsEnabled: true) == SamplingConfig.maxTokensDefault)
    }

    /// Reasoning alias on a fresh install — the chat floor (2,048)
    /// today sits below the v0.4.12 baseline (4,096), so the persisted
    /// ``maxTokens`` stays at 4,096. The structural piece this test
    /// pins is ``activeReasoningParser`` being captured so that
    /// ``effectiveMaxTokens(toolsEnabled: true)`` returns ≥ 4,096.
    @Test("Cycle-3: reasoning alias on fresh install — activeReasoningParser captured, chat-side effective ≥ 2,048, tools-side effective ≥ 4,096")
    func reasoningAliasFreshInstall() {
        let s = SamplingConfig(defaults: freshDefaults())
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
        #expect(s.activeReasoningParser == "vibethinker")
        // The persisted slider value isn't downshifted — 4,096 > 2,048.
        #expect(s.maxTokens >= SamplingConfig.reasoningChatFloor)
        // Effective at request time honours BOTH floors.
        #expect(s.effectiveMaxTokens(toolsEnabled: false) >= SamplingConfig.reasoningChatFloor)
        #expect(s.effectiveMaxTokens(toolsEnabled: true) >= SamplingConfig.reasoningToolsFloor)
    }

    /// The cycle-2 fuzz-correctness P1 scenario — a user whose persisted
    /// slider sits at ``max_tokens = 512`` (cycle-2 repro literally
    /// returns ``content = null`` on vibethinker tools-on) but on a
    /// reasoning alias swap. Today the structural rule is "user override
    /// wins" so the auto-bump is suppressed. The test pins the literal
    /// behaviour so a future change that decides to *force* the floor
    /// regardless of override gets called out explicitly.
    @Test("Cycle-3: user who lowered max_tokens below the floor wins — auto-bump suppressed on reasoning alias")
    func userOverrideBelowFloorWins() {
        let s = SamplingConfig(defaults: freshDefaults())
        // User dragged the slider WAY below the chat floor.
        s.maxTokens = 512
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
        #expect(s.activeReasoningParser == "vibethinker",
                "the parser signal is still captured — only the auto-bump is suppressed")
        #expect(s.maxTokens == 512, "user's persisted choice survives the apply")
        #expect(s.effectiveMaxTokens(toolsEnabled: false) == 512,
                "request-time floor never overrides an explicit user choice")
        #expect(s.effectiveMaxTokens(toolsEnabled: true) == 512,
                "tools floor also yields to user override")
    }

    /// Future-proofing: if the v0.4.12 ``maxTokensDefault`` ever drops
    /// to a value LOWER than ``reasoningChatFloor`` (e.g. someone
    /// raises 4,096 → 8,192 for non-reasoning and lowers
    /// ``maxTokensDefault`` to 1,024 to compensate), the apply path
    /// must lift the persisted slider UP to the floor so the user-
    /// visible Settings number reflects what's actually being sent.
    /// We can't easily fake the constant, so we verify the lift logic
    /// directly via ``effectiveMaxTokens`` on a freshly-applied
    /// reasoning profile — the floor never appears below the constant.
    @Test("Cycle-3: effective tools-floor on a reasoning alias is never below reasoningToolsFloor")
    func effectiveToolsFloorIsContractualMinimum() {
        let s = SamplingConfig(defaults: freshDefaults())
        let profile = ServerModelProfile(
            id: "phi-4-mini-reasoning",
            recommendedSampling: nil,
            isHybrid: false,
            isMoe: false,
            toolCallParser: nil,
            reasoningParser: "deepseek_r1",
            modality: "text"
        )
        _ = s.applyServerProfile(profile)
        // Tools-on minimum is the contract; chat minimum is the looser
        // floor — both must hold regardless of subsequent slider drags
        // (as long as the slider stays at the auto-scaled landmarks).
        #expect(s.effectiveMaxTokens(toolsEnabled: true) >= SamplingConfig.reasoningToolsFloor)
        #expect(s.effectiveMaxTokens(toolsEnabled: false) >= SamplingConfig.reasoningChatFloor)
    }

    /// Empty-string ``reasoning_parser`` defensively treated as "no
    /// reasoning parser". rapid-mlx today emits either the string name
    /// or omits the key (→ nil after decode); we add this guard so a
    /// future server bug that serialises an empty string can't silently
    /// downshift the floor onto every alias.
    @Test("Cycle-3: empty-string reasoning_parser is treated as no-reasoning — guards against server-side serialisation bugs")
    func emptyReasoningParserStringIsIgnored() {
        let s = SamplingConfig(defaults: freshDefaults())
        let profile = ServerModelProfile(
            id: "weird-alias",
            recommendedSampling: nil,
            isHybrid: false,
            isMoe: false,
            toolCallParser: nil,
            reasoningParser: "",
            modality: "text"
        )
        _ = s.applyServerProfile(profile)
        #expect(s.effectiveMaxTokens(toolsEnabled: false) == s.maxTokens)
        #expect(s.effectiveMaxTokens(toolsEnabled: true) == s.maxTokens,
                "empty-string parser must not trip the tools-on auto-bump")
    }

    /// ``resetToDefaults`` must clear the in-memory ``activeReasoningParser``
    /// so a "Reset" press behaves like a clean v0.4.12 install. Without
    /// this clear, a user who resets while a reasoning alias is loaded
    /// would still see ``effectiveMaxTokens(toolsEnabled: true) ==
    /// reasoningToolsFloor``. That used to be harmless because the floor
    /// equalled the default; now that it is 16,384 the two genuinely
    /// diverge, so this assertion carries real weight.
    @Test("Cycle-3: resetToDefaults clears activeReasoningParser + the auto-scale bookkeeping")
    func resetClearsReasoningBookkeeping() {
        let s = SamplingConfig(defaults: freshDefaults())
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
        #expect(s.activeReasoningParser == "vibethinker")

        s.resetToDefaults()
        #expect(s.activeReasoningParser == nil)
        #expect(s.maxTokens == SamplingConfig.maxTokensDefault)
        // After reset, no reasoning floor applies regardless of tools.
        #expect(s.effectiveMaxTokens(toolsEnabled: true) == SamplingConfig.maxTokensDefault)
    }

    /// Mid-session alias swap from reasoning → non-reasoning. The
    /// captured ``activeReasoningParser`` must refresh to ``nil`` so the
    /// next tools-on send to the non-reasoning alias doesn't carry the
    /// stale floor. Mirrors the ``.task(id: server.servingAlias)``
    /// re-fire in ``RapidApp``.
    @Test("Cycle-3: alias swap from reasoning to non-reasoning clears the parser capture")
    func aliasSwapDownshiftsParser() {
        let s = SamplingConfig(defaults: freshDefaults())
        let reasoning = ServerModelProfile(
            id: "vibethinker-3b-8bit",
            recommendedSampling: nil,
            isHybrid: false,
            isMoe: false,
            toolCallParser: "hermes",
            reasoningParser: "vibethinker",
            modality: "text"
        )
        _ = s.applyServerProfile(reasoning)
        #expect(s.activeReasoningParser == "vibethinker")

        let nonReasoning = ServerModelProfile(
            id: "hermes3-8b-4bit",
            recommendedSampling: nil,
            isHybrid: false,
            isMoe: false,
            toolCallParser: "hermes",
            reasoningParser: nil,
            modality: "text"
        )
        _ = s.applyServerProfile(nonReasoning)
        #expect(s.activeReasoningParser == nil)
        #expect(s.effectiveMaxTokens(toolsEnabled: true) == s.maxTokens,
                "post-swap, no reasoning floor")
    }

    /// Power user dragged temperature only (still wants curated max_tokens
    /// behaviour). Cycle-3 gate is intentionally LESS strict than the
    /// curated-sampling gate: ``isAtDefaults`` blocks recommended_sampling
    /// because mixing a server's calibrated temperature with a user's
    /// dragged top_p produces a hybrid neither side validated. The
    /// max_tokens floor doesn't have that risk — it's a one-sided lift,
    /// independent of the other knobs. Pinning here so a refactor that
    /// "unifies" the gate doesn't silently regress the floor.
    /// Codex r1 MAJOR — pre-fix, ``effectiveMaxTokens`` matched
    /// ``maxTokens == reasoningChatFloor`` as an "auto-scaled
    /// landmark", which would silently bump a user who explicitly
    /// dragged the slider to 2,048 up to 4,096 on tools-on. With
    /// today's constants the auto-scale path never WRITES 2,048
    /// (``max(maxTokensDefault, reasoningChatFloor) ==
    /// maxTokensDefault`` for the current 4,096 baseline) so any
    /// 2,048 reaching the gate is user intent. The gate is now
    /// strictly ``maxTokensIsAutoScaled || maxTokens ==
    /// maxTokensDefault``; this test pins that contract.
    @Test("Cycle-3 codex r1: user-chosen max_tokens == reasoningChatFloor (2048) is NOT treated as auto-scaled — tools-on bump suppressed")
    func explicitChatFloorValueIsUserOverride() {
        let s = SamplingConfig(defaults: freshDefaults())
        // User explicitly drags slider to 2,048 — same numerical
        // value as the chat floor, but expressed BEFORE any
        // reasoning profile lands.
        s.maxTokens = 2_048
        #expect(s.maxTokens == 2_048)
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
        #expect(s.activeReasoningParser == "vibethinker")
        #expect(s.maxTokens == 2_048, "user's persisted choice survives the apply")
        // The critical assertion: tools-on must NOT bump to 4,096
        // because 2,048 was user intent, not an auto-scale write.
        #expect(s.effectiveMaxTokens(toolsEnabled: true) == 2_048,
                "tools-on must respect the explicit 2,048 choice, not lift to reasoningToolsFloor")
        #expect(s.effectiveMaxTokens(toolsEnabled: false) == 2_048,
                "chat-off must also respect the explicit 2,048 choice")
    }

    /// Codex r1 MAJOR follow-up — explicit ``maxTokensDefault``
    /// (4,096) value as a user-typed input is structurally
    /// indistinguishable from "untouched since first launch"; we
    /// document the chosen behaviour here so a future "track every
    /// keystroke" refactor doesn't accidentally split the cases.
    /// Either interpretation is safe today because the tools floor
    /// EQUALS the baseline (both 4,096) — the bump is a no-op
    /// regardless of which path activates. This test pins the
    /// invariant in case the tools floor is ever raised.
    @Test("Cycle-3: max_tokens == maxTokensDefault is treated as fresh-install — tools-on may lift (no-op today, future-proofs the contract)")
    func defaultValueAllowsAutoLift() {
        let s = SamplingConfig(defaults: freshDefaults())
        // No user mutation; value sits at baseline.
        #expect(s.maxTokens == SamplingConfig.maxTokensDefault)
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
        #expect(s.effectiveMaxTokens(toolsEnabled: true) >= SamplingConfig.reasoningToolsFloor)
        #expect(s.effectiveMaxTokens(toolsEnabled: false) >= SamplingConfig.reasoningChatFloor)
    }

    /// Codex r1 MAJOR (#2) — alias-swap race. A reasoning profile
    /// lands, then ``clearActiveReasoningParser`` fires (mirroring
    /// the ``RapidApp`` ``.task(id:)`` body's pre-fetch clear) BEFORE
    /// the next async fetch resolves. ``effectiveMaxTokens`` must
    /// return the slider value verbatim during that window — no
    /// stale parser-based bump leaks into a chat send mid-swap.
    @Test("Cycle-3 codex r1 #2: clearActiveReasoningParser closes the alias-swap race window")
    func clearParserClosesSwapRace() {
        let s = SamplingConfig(defaults: freshDefaults())
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
        #expect(s.activeReasoningParser == "vibethinker")
        // RapidApp fires this at the top of the .task(id:) body
        // BEFORE awaiting the next profile fetch.
        s.clearActiveReasoningParser()
        #expect(s.activeReasoningParser == nil)
        // During the in-flight window: no reasoning bump applied.
        #expect(s.effectiveMaxTokens(toolsEnabled: true) == s.maxTokens)
        #expect(s.effectiveMaxTokens(toolsEnabled: false) == s.maxTokens)
    }

    @Test("Cycle-3: user touched temperature only — max_tokens still auto-scales on reasoning alias")
    func temperatureOverrideDoesNotBlockMaxTokensFloor() {
        let s = SamplingConfig(defaults: freshDefaults())
        s.temperature = 0.42  // user dragged temp; max_tokens still at default
        #expect(s.maxTokens == SamplingConfig.maxTokensDefault)
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
        #expect(s.activeReasoningParser == "vibethinker")
        #expect(s.temperature == 0.42, "user's temperature override survives")
        // Effective max_tokens still honours the tools floor on reasoning.
        #expect(s.effectiveMaxTokens(toolsEnabled: true) >= SamplingConfig.reasoningToolsFloor)
    }

    // MARK: - Issue #363: context_window wire decode + applyServerProfile capture

    @Test("rapid-mlx ≥ 0.8.4 context_window decodes into ServerModelProfile.contextWindow")
    func decodesContextWindowField() throws {
        // Wire shape mirrors what the v0.8.4 fix landed:
        // ``ModelInfo.context_window`` populated from
        // ``service.helpers.get_model_max_context``.
        let json = """
        {
          "id": "qwen3.6-35b-4bit",
          "object": "model",
          "owned_by": "rapid-mlx",
          "context_window": 262144
        }
        """
        let profile = try JSONDecoder().decode(
            ServerModelProfile.self,
            from: Data(json.utf8)
        )
        #expect(profile.contextWindow == 262_144)
    }

    @Test("Older sidecar (no context_window field) decodes as nil — backward compat")
    func decodesAbsentContextWindowAsNil() throws {
        // rapid-mlx < 0.8.4 omits the field; the decoder must keep
        // round-tripping the legacy shape without throwing so the
        // desktop falls back to its per-family heuristic cleanly.
        let json = """
        {
          "id": "qwen3.5-4b-4bit",
          "object": "model",
          "owned_by": "rapid-mlx"
        }
        """
        let profile = try JSONDecoder().decode(
            ServerModelProfile.self,
            from: Data(json.utf8)
        )
        #expect(profile.contextWindow == nil)
    }

    @Test("Explicit JSON null for context_window decodes as nil (no decode error)")
    func decodesExplicitNullContextWindow() throws {
        // Server emits the field present-with-null when the engine
        // resolver couldn't probe a useful value (DoS-sentinel
        // suppression). The decoder must accept the shape and
        // surface nil so the fallback path engages.
        let json = """
        {
          "id": "qwen3.5-4b-4bit",
          "object": "model",
          "owned_by": "rapid-mlx",
          "context_window": null
        }
        """
        let profile = try JSONDecoder().decode(
            ServerModelProfile.self,
            from: Data(json.utf8)
        )
        #expect(profile.contextWindow == nil)
    }

    @Test("applyServerProfile captures context_window into activeContextWindow")
    func appliesContextWindowFromProfile() {
        let s = SamplingConfig(defaults: freshDefaults())
        #expect(s.activeContextWindow == nil)
        let profile = ServerModelProfile(
            id: "qwen3.6-35b-4bit",
            contextWindow: 262_144
        )
        _ = s.applyServerProfile(profile)
        #expect(s.activeContextWindow == 262_144)
    }

    @Test("applyServerProfile with nil context_window leaves activeContextWindow nil")
    func appliesNilContextWindow() {
        let s = SamplingConfig(defaults: freshDefaults())
        let profile = ServerModelProfile(
            id: "qwen3.5-4b-4bit",
            contextWindow: nil
        )
        _ = s.applyServerProfile(profile)
        #expect(s.activeContextWindow == nil)
    }

    @Test("applyServerProfile rejects 0 / negative context_window (server-side regression guard)")
    func appliesContextWindowDefensively() {
        let s = SamplingConfig(defaults: freshDefaults())
        // Inject a malformed wire value and assert we don't cache it.
        _ = s.applyServerProfile(ServerModelProfile(id: "x", contextWindow: 0))
        #expect(s.activeContextWindow == nil)
        _ = s.applyServerProfile(ServerModelProfile(id: "x", contextWindow: -1))
        #expect(s.activeContextWindow == nil)
    }

    @Test("clearActiveReasoningParser also clears activeContextWindow (alias swap clean slate)")
    func clearWipesContextWindow() {
        let s = SamplingConfig(defaults: freshDefaults())
        _ = s.applyServerProfile(
            ServerModelProfile(id: "qwen3.6-35b-4bit", contextWindow: 262_144)
        )
        #expect(s.activeContextWindow == 262_144)
        s.clearActiveReasoningParser()
        #expect(s.activeContextWindow == nil)
    }
}
