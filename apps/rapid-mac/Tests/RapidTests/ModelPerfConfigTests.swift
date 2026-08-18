import Foundation
import Testing
@testable import Rapid

/// Issue #1717: the engine's throughput knobs reach `serve` only as CLI flags,
/// so the desktop was the slow way to use our own engine. These tests pin the
/// three properties the issue's acceptance list turns on:
///
///   * an install that never opens the panel spawns byte-identical argv;
///   * an override supersedes the RAM-tier recommendation for the knob it
///     covers, and ONLY that knob;
///   * a KV choice never ships alongside a conflicting one, because the
///     engine silently ignores `--kv-cache-dtype` when TurboQuant is on and
///     "the UI said X, the engine did Y" is the failure mode the issue is
///     written to prevent.
/// ``ModelPerfConfigStore`` is `@MainActor` (it is `@Observable` state bound by
/// Settings), so the suite is too. The pure-value assertions below do not need
/// the isolation but cost nothing by inheriting it.
@MainActor
@Suite("Issue #1717 — per-model performance overrides")
struct ModelPerfConfigTests {

    // MARK: - Sparseness

    @Test("An untouched config contributes no flags")
    func untouchedConfigIsEmpty() {
        let config = ModelPerfConfig()
        #expect(config.isEmpty)
        #expect(config.launchFlags.isEmpty)
    }

    @Test("A store with no overrides contributes no flags for any alias")
    func untouchedStoreContributesNothing() {
        let store = makeStore()
        #expect(store.launchFlags(forAlias: "qwen3.6-35b-4bit").isEmpty)
        #expect(store.hasOverride(forAlias: "qwen3.6-35b-4bit") == false)
        #expect(store.configuredAliases.isEmpty)
    }

    @Test("MTP is off by default for both Qwen3.8 aliases")
    func mtpDefaultsOff() {
        let store = makeStore()
        #expect(store.launchFlags(forAlias: "qwen3.8-27b-4bit").isEmpty)
        #expect(store.launchFlags(forAlias: "qwen3.8-27b-mixed-3.5bpw").isEmpty)
    }

    @Test("Setting a knob then clearing it removes the row rather than pinning the default")
    func clearingAnOverrideRemovesTheRow() {
        let store = makeStore()
        store.setConfig(ModelPerfConfig(kvCacheMode: .int8), forAlias: "gemma-4-26b-4bit")
        #expect(store.hasOverride(forAlias: "gemma-4-26b-4bit"))

        store.setConfig(ModelPerfConfig(), forAlias: "gemma-4-26b-4bit")
        #expect(store.hasOverride(forAlias: "gemma-4-26b-4bit") == false)
        #expect(store.configuredAliases.isEmpty)
        // The point of the row removal: a later engine-side default change
        // reaches this user instead of being frozen by a snapshot.
        #expect(store.launchFlags(forAlias: "gemma-4-26b-4bit").isEmpty)
    }

    @Test("Reset restores the no-opinion state in one action")
    func resetClearsTheAlias() {
        let store = makeStore()
        store.setConfig(
            ModelPerfConfig(kvCacheMode: .turboquantK8V4, prefixCacheEnabled: false),
            forAlias: "bonsai-27b-2bit"
        )
        store.resetToDefaults(forAlias: "bonsai-27b-2bit")
        #expect(store.launchFlags(forAlias: "bonsai-27b-2bit").isEmpty)
    }

    // MARK: - Persistence

    @Test("Overrides survive a store reconstruction against the same defaults")
    func overridesPersist() {
        let defaults = makeDefaults()
        let first = ModelPerfConfigStore(defaults: defaults)
        first.setConfig(
            ModelPerfConfig(kvCacheMode: .int4, cacheMemoryMB: 4096),
            forAlias: "qwen3.6-35b-4bit"
        )

        let second = ModelPerfConfigStore(defaults: defaults)
        #expect(second.launchFlags(forAlias: "qwen3.6-35b-4bit")
            == ["--kv-cache-dtype", "int4", "--cache-memory-mb", "4096"])
    }

    @Test("Alias lookup ignores case and surrounding whitespace")
    func aliasLookupIsNormalized() {
        let store = makeStore()
        store.setConfig(ModelPerfConfig(kvCacheMode: .int8), forAlias: "Gemma-4-26B-4bit")
        // The picker, persisted settings and ``servingAlias`` do not agree on
        // case; without normalizing, the user's setting silently does nothing.
        #expect(store.launchFlags(forAlias: "  gemma-4-26b-4bit ")
            == ["--kv-cache-dtype", "int8"])
    }

    @Test("MTP opt-in persists and normalizes the alias")
    func mtpOptInPersists() {
        let defaults = makeDefaults()
        let first = ModelPerfConfigStore(defaults: defaults)
        first.setConfig(
            ModelPerfConfig(speculativePreset: .init(
                method: .mtp,
                model: "rapid-mlx/Qwen3.8-27B-4bit-MTP-MLX",
                tokens: 3
            )),
            forAlias: " Qwen3.8-27B-4bit "
        )

        let second = ModelPerfConfigStore(defaults: defaults)
        #expect(second.config(forAlias: "qwen3.8-27b-4bit").speculativePreset?.method == .mtp)
        #expect(second.launchFlags(forAlias: "qwen3.8-27b-4bit") == [
            "--speculative-config",
            #"{"method":"mtp","model":"rapid-mlx/Qwen3.8-27B-4bit-MTP-MLX","num_speculative_tokens":3}"#,
        ])
    }

    @Test("A corrupt persisted blob surfaces an error instead of reporting defaults")
    func corruptBlobSurfacesError() {
        let defaults = makeDefaults()
        defaults.set(Data("not json".utf8), forKey: ModelPerfConfigStore.storageKey)
        let store = ModelPerfConfigStore(defaults: defaults)
        #expect(store.loadError != nil)
        #expect(store.configuredAliases.isEmpty)
    }

    // MARK: - Flag rendering

    @Test("KV modes render the flag the engine actually reads", arguments: [
        (KVCacheMode.bf16, ["--kv-cache-dtype", "bf16"]),
        (KVCacheMode.int8, ["--kv-cache-dtype", "int8"]),
        (KVCacheMode.int4, ["--kv-cache-dtype", "int4"]),
        (KVCacheMode.turboquantV4, ["--kv-cache-turboquant", "v4"]),
        (KVCacheMode.turboquantK8V4, ["--kv-cache-turboquant", "k8v4"]),
    ])
    func kvModeFlags(mode: KVCacheMode, expected: [String]) {
        #expect(mode.launchFlags == expected)
    }

    @Test("Resolved launch flags round-trip into the residency config")
    func resolvedFlagsRoundTrip() {
        let config = ModelPerfConfig(launchFlags: [
            "--no-mllm", "--kv-cache-turboquant", "k8v4",
            "--disable-prefix-cache", "--cache-memory-mb", "4096",
        ])
        #expect(config == ModelPerfConfig(
            kvCacheMode: .turboquantK8V4,
            prefixCacheEnabled: false,
            cacheMemoryMB: 4096
        ))
    }

    @Test("MTP canonical config round-trips into residency state")
    func mtpFlagsRoundTrip() {
        let config = ModelPerfConfig(launchFlags: [
            "--speculative-config",
            #"{"method":"mtp","model":"rapid-mlx/Qwen3.8-27B-4bit-MTP-MLX","num_speculative_tokens":3}"#,
        ])
        #expect(config.speculativePreset?.method == .mtp)
    }

    @Test("Registry-selected methods render canonical configs")
    func speculativeMethodsRender() {
        let enabled = ModelPerfConfig(speculativePreset: .init(
            method: .mtp,
            model: "rapid-mlx/Qwen3.8-27B-mixed-3.5bpw-MLX",
            tokens: 3
        ))
        #expect(enabled.launchFlags(forAlias: "qwen3.8-27b-mixed-3.5bpw") == [
            "--speculative-config",
            #"{"method":"mtp","model":"rapid-mlx/Qwen3.8-27B-mixed-3.5bpw-MLX","num_speculative_tokens":3}"#,
        ])
        #expect(ModelPerfConfig(speculativePreset: .init(
            method: .suffix, model: nil, tokens: nil
        ))
            .launchFlags(forAlias: "llama3-3b-4bit") == [
                "--speculative-config", #"{"method":"suffix"}"#,
            ])
    }

    @Test("The deprecated --kv-cache-quantization spelling is never emitted")
    func legacyQuantizationFlagIsNeverEmitted() {
        // The engine documents it as "[deprecated alias of --kv-cache-dtype
        // int8]" AND hard-errors when it is combined with TurboQuant. Emitting
        // it could only reintroduce that conflict.
        for mode in KVCacheMode.allCases {
            #expect(mode.launchFlags.contains("--kv-cache-quantization") == false)
        }
    }

    @Test("Prefix caching renders as the engine's enable/disable pair")
    func prefixCacheFlags() {
        #expect(ModelPerfConfig(prefixCacheEnabled: true).launchFlags == ["--enable-prefix-cache"])
        #expect(ModelPerfConfig(prefixCacheEnabled: false).launchFlags == ["--disable-prefix-cache"])
    }

    @Test("Cache memory is clamped into the supported range")
    func cacheMemoryIsClamped() {
        #expect(ModelPerfConfig(cacheMemoryMB: 1).launchFlags
            == ["--cache-memory-mb", String(ModelPerfConfig.cacheMemoryMBRange.lowerBound)])
        #expect(ModelPerfConfig(cacheMemoryMB: 1_000_000).launchFlags
            == ["--cache-memory-mb", String(ModelPerfConfig.cacheMemoryMBRange.upperBound)])
    }

    @Test("Every KV mode carries a trade-off line")
    func everyModeStatesItsTradeOff() {
        // The issue requires the trade-off next to the control. Pinning it
        // here means a new case cannot ship without one.
        for mode in KVCacheMode.allCases {
            #expect(mode.tradeOff.isEmpty == false)
            #expect(mode.title.isEmpty == false)
        }
    }

    // MARK: - Merge against the RAM-tier recommendation

    /// The 24 GB tier's recommendation for this alias, verbatim from
    /// ``RAMBucketedDefault``. Hard-coded rather than fetched so the merge
    /// contract is pinned independently of a future recommendation change.
    private let gemmaRecommendation = [
        "--no-mllm", "--kv-cache-dtype", "bf16", "--cache-memory-mb", "512",
    ]

    @Test("No override leaves the recommendation untouched")
    func noOverrideKeepsRecommendation() {
        #expect(ServerManager.mergedPerformanceFlags(
            recommended: gemmaRecommendation, userOverrides: []
        ) == gemmaRecommendation)
    }

    @Test("An override replaces only its own knob")
    func overrideSupersedesOnlyItsOwnKnob() {
        let merged = ServerManager.mergedPerformanceFlags(
            recommended: gemmaRecommendation,
            userOverrides: ["--kv-cache-dtype", "int4"]
        )
        // --no-mllm and the recommendation's cache budget survive; only the
        // dtype is replaced.
        #expect(merged == ["--no-mllm", "--cache-memory-mb", "512", "--kv-cache-dtype", "int4"])
    }

    @Test("A TurboQuant choice drops the recommendation's conflicting dtype")
    func turboquantSupersedesRecommendedDtype() {
        let merged = ServerManager.mergedPerformanceFlags(
            recommended: gemmaRecommendation,
            userOverrides: ["--kv-cache-turboquant", "k8v4"]
        )
        // Leaving `--kv-cache-dtype bf16` on argv would be a lie: the engine
        // resolves the dtype only when TurboQuant is off.
        #expect(merged.contains("--kv-cache-dtype") == false)
        #expect(merged == ["--no-mllm", "--cache-memory-mb", "512", "--kv-cache-turboquant", "k8v4"])
    }

    @Test("A prefix-cache override does not disturb the KV recommendation")
    func prefixOverrideLeavesKVAlone() {
        let merged = ServerManager.mergedPerformanceFlags(
            recommended: gemmaRecommendation,
            userOverrides: ["--disable-prefix-cache"]
        )
        #expect(merged == gemmaRecommendation + ["--disable-prefix-cache"])
    }

    @Test("A cache-budget override replaces the recommended budget exactly once")
    func cacheBudgetOverrideReplacesValue() {
        let merged = ServerManager.mergedPerformanceFlags(
            recommended: gemmaRecommendation,
            userOverrides: ["--cache-memory-mb", "8192"]
        )
        #expect(merged == ["--no-mllm", "--kv-cache-dtype", "bf16", "--cache-memory-mb", "8192"])
        #expect(merged.filter { $0 == "--cache-memory-mb" }.count == 1)
    }

    @Test("Merging against an empty recommendation just yields the overrides")
    func emptyRecommendationYieldsOverrides() {
        let merged = ServerManager.mergedPerformanceFlags(
            recommended: [],
            userOverrides: ["--kv-cache-dtype", "int8", "--enable-prefix-cache"]
        )
        #expect(merged == ["--kv-cache-dtype", "int8", "--enable-prefix-cache"])
    }

    @Test("A bare value-carrying flag does not swallow the flag after it")
    func bareValueCarryingFlagDoesNotEatItsNeighbour() {
        // ``--kv-cache-turboquant`` is ``nargs="?"`` in the engine, so a
        // recommendation may carry it bare. Consuming the next token
        // unconditionally would silently drop ``--no-mllm``.
        let merged = ServerManager.mergedPerformanceFlags(
            recommended: ["--kv-cache-turboquant", "--no-mllm"],
            userOverrides: ["--kv-cache-dtype", "int8"]
        )
        #expect(merged == ["--no-mllm", "--kv-cache-dtype", "int8"])
    }

    // MARK: - Helpers

    private func makeDefaults() -> UserDefaults {
        // A private suite per test keeps these from colliding with each other
        // and with the real app domain.
        let suite = "rapid.tests.perf.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suite)!
        defaults.removePersistentDomain(forName: suite)
        return defaults
    }

    private func makeStore() -> ModelPerfConfigStore {
        ModelPerfConfigStore(defaults: makeDefaults())
    }
}
