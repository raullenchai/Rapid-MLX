import Foundation
import Testing
@testable import Rapid

@MainActor
@Suite("Quickstart hardware-aware starter policy")
struct QuickstartStarterPolicyTests {
    private func hardware(_ ramGB: Double) -> MacHardware {
        MacHardware(
            brandString: "Test Mac",
            family: .m3,
            tier: .base,
            physicalRAMBytes: UInt64(ramGB * 1_073_741_824),
            memoryBandwidthGBs: 100
        )
    }

    private func entry(
        _ alias: String,
        cached: Bool = false,
        kind: ModelKind = .chat
    ) -> ModelEntry {
        ModelEntry(
            alias: alias,
            hfRepo: "fixture/\(alias)",
            sizeOnDisk: nil,
            cached: cached,
            kind: kind,
            recommendationPolicyDigests: [RAMBucketedDefault.policyDigest]
        )
    }

    @Test("LFM 2.6B fits and is the default on 8, 16, 32 and 64 GB Macs")
    func ramMatrix() {
        let catalog = [entry("lfm2.5-1b-4bit"), entry("lfm2.5-2.6b-4bit"), entry("qwen3.5-4b-4bit")]
        for ram in [8.0, 15.99, 16, 32, 64] {
            let machine = hardware(ram)
            #expect(ModelSizing.classify(ModelSizing.estimate(alias: "lfm2.5-2.6b-4bit"), on: machine) != .tooBig)
            #expect(QuickstartCoordinator.defaultChoice(hardware: machine, catalog: catalog).alias == "lfm2.5-2.6b-4bit")
            #expect(QuickstartCoordinator.baselineChoice(hardware: machine).alias == "lfm2.5-2.6b-4bit")
            #expect(QuickstartCoordinator.baselineChoice(physicalRAMGB: ram).alias == "lfm2.5-2.6b-4bit")
        }
        #expect(ModelSizing.estimate(alias: "lfm2.5-2.6b-4bit").totalGB == 3.0)
        #expect(ModelSizing.classify(ModelSizing.estimate(alias: "lfm2.5-2.6b-4bit"), on: hardware(4)) == .tooBig)
        #expect(QuickstartCoordinator.defaultChoice(hardware: hardware(4), catalog: catalog).alias == "lfm2.5-1b-4bit")
    }

    @Test("The synchronous no-catalog baseline is safe before welcome Skip")
    func immediateWelcomeBaseline() {
        let coordinator = QuickstartCoordinator()
        coordinator.applyDefaultChoice(hardware: hardware(8), catalog: [])

        #expect(coordinator.selection.alias == "lfm2.5-2.6b-4bit")
    }

    @Test("The low-memory fallback keeps its lowest-memory spoken category")
    func lowMemoryStarterAccessibilityCategory() {
        let lowMemoryLabel = QuickstartRecommendedCard.accessibilityText(
            for: QuickstartCoordinator.lowMemoryChoice,
            sizeText: "720 MB"
        )
        let standardLabel = QuickstartRecommendedCard.accessibilityText(
            for: QuickstartCoordinator.defaultChoice,
            sizeText: "2.9 GB"
        )

        #expect(lowMemoryLabel.contains("Lowest memory"))
        #expect(lowMemoryLabel.contains("recommended starter"))
        #expect(!standardLabel.contains("Lowest memory"))
    }

    @Test("The 1.2B choice remains a fallback, not a starter, on a 16 GB Mac")
    func lowMemoryChoiceIsContextual() {
        let coordinator = QuickstartCoordinator()
        coordinator.applyDefaultChoice(
            hardware: hardware(16),
            catalog: [entry("qwen3.5-4b-4bit"), entry("lfm2.5-1b-4bit")]
        )
        coordinator.select(QuickstartCoordinator.lowMemoryChoice)

        #expect(coordinator.selection == QuickstartCoordinator.lowMemoryChoice)
        coordinator.applyDefaultChoice(hardware: hardware(16), catalog: [])
        #expect(coordinator.selection == QuickstartCoordinator.lowMemoryChoice,
                "a refresh must preserve the explicit low-memory choice")
    }

    @Test("An eligible cached chat model wins without a download")
    func cachedEligibleWins() {
        let pick = QuickstartCoordinator.defaultChoice(
            hardware: hardware(16),
            catalog: [
                entry("qwen3.5-4b-4bit"),
                entry("qwen3.5-9b-4bit", cached: true),
            ]
        )
        #expect(pick.alias == "qwen3.5-9b-4bit")
    }

    @Test("A cached experimental trade-up never becomes the automatic first model")
    func cachedExperimentalTradeUpIsNeverAutomatic() {
        let pick = QuickstartCoordinator.defaultChoice(
            hardware: hardware(24),
            catalog: [
                entry("qwen3.5-4b-4bit"),
                entry("neohorse-9b-4bit", cached: true),
            ]
        )

        #expect(pick.alias == "lfm2.5-2.6b-4bit")
        #expect(QuickstartCoordinator.optionalOnlyAliases.contains("neohorse-9b-4bit"))
    }

    @Test("A cached Qwen alternative stays visible when it wins below 16 GB")
    func cachedStandardStarterRemainsVisible() {
        let catalog = [
            entry("lfm2.5-1b-4bit"),
            entry("lfm2.5-2.6b-4bit"),
            entry("qwen3.5-4b-4bit", cached: true),
        ]
        let pick = QuickstartCoordinator.defaultChoice(
            hardware: hardware(15.99),
            catalog: catalog
        )
        let shortlist = QuickstartView.shortlist(
            catalog: catalog,
            selection: pick.alias,
            physicalRAMGB: 15.99
        )

        #expect(pick.alias == "qwen3.5-4b-4bit")
        #expect(shortlist.cached.map(\.alias).contains(pick.alias))
        #expect(shortlist.visibleAliases.contains(pick.alias))
    }

    @Test("An 8 GB Mac does not promote a cached model that fails its fit contract")
    func cachedStandardStarterMustFit() {
        let catalog = [
            entry("lfm2.5-1b-4bit"),
            entry("lfm2.5-2.6b-4bit"),
            entry("qwen3.5-4b-4bit", cached: true),
        ]
        let machine = hardware(8)

        #expect(ModelSizing.classify(
            ModelSizing.estimate(alias: "qwen3.5-4b-4bit"),
            on: machine
        ) == .tooBig)
        #expect(QuickstartCoordinator.defaultChoice(
            hardware: machine,
            catalog: catalog
        ).alias == "lfm2.5-2.6b-4bit")
    }

    @Test("A cached 2.6B model is the fitting 8 GB baseline")
    func cachedCompactUpgradeRemainsManual() {
        let catalog = [
            entry("lfm2.5-1b-4bit"),
            entry("lfm2.5-2.6b-4bit", cached: true),
            entry("qwen3.5-4b-4bit"),
        ]

        #expect(QuickstartCoordinator.defaultChoice(
            hardware: hardware(8), catalog: catalog
        ).alias == "lfm2.5-2.6b-4bit")
    }

    @Test("The shortlist starts with LFM 2.6B then Qwen 4B, with no 1.2B when 2.6B fits")
    func shortlistHasOneStarter() {
        let catalog = [entry("lfm2.5-1b-4bit"), entry("lfm2.5-2.6b-4bit"), entry("qwen3.5-4b-4bit")]
        for ram in [8.0, 16, 32, 64] {
            let list = QuickstartView.shortlist(catalog: catalog, selection: "lfm2.5-2.6b-4bit", physicalRAMGB: ram)
            #expect(list.starters.map(\.alias) == ["lfm2.5-2.6b-4bit"])
            #expect(list.recommended.first?.alias == "qwen3.5-4b-4bit")
            #expect(!list.visibleAliases.contains("lfm2.5-1b-4bit"))
            #expect(list.lowMemory.isEmpty)
        }
        let small = QuickstartView.shortlist(catalog: catalog, selection: "lfm2.5-1b-4bit", physicalRAMGB: 4)
        #expect(small.starters.map(\.alias) == ["lfm2.5-1b-4bit"])
        #expect(QuickstartCoordinator.defaultChoice.blurb == "Fast everyday chat and tools. Text only.")
        #expect(QuickstartCoordinator.visionChoice.blurb == "Smarter, and understands images. Twice the download.")
    }

    @Test("An automatically selected sibling alternate is surfaced as Your Pick")
    func selectedCachedAlternateRemainsVisible() {
        let catalog = [
            entry("qwen3.5-4b-4bit"),
            entry("llama3-8b-2bit", cached: true),
            entry("llama3-8b-4bit", cached: true),
        ]
        let pick = QuickstartCoordinator.defaultChoice(
            hardware: hardware(32),
            catalog: catalog
        )
        let shortlist = QuickstartView.shortlist(
            catalog: catalog,
            selection: pick.alias,
            physicalRAMGB: 32
        )

        #expect(pick.alias == "llama3-8b-2bit")
        #expect(shortlist.cached.map(\.alias) == ["llama3-8b-4bit"])
        #expect(shortlist.cachedAlternates.map(\.alias) == ["llama3-8b-2bit"])
        #expect(shortlist.yourPick?.alias == pick.alias)
        #expect(shortlist.visibleAliases.contains(pick.alias))
    }

    @Test("Cached media and the 1.2B escape never become automatic starters")
    func ineligibleCacheDoesNotWin() {
        let pick = QuickstartCoordinator.defaultChoice(
            hardware: hardware(16),
            catalog: [
                entry("qwen3.5-4b-4bit"),
                entry("lfm2.5-1b-4bit", cached: true),
                entry("flux-klein", cached: true, kind: .image),
            ]
        )
        #expect(pick.alias == "lfm2.5-2.6b-4bit")
        #expect(QuickstartCoordinator.lowMemoryChoice.alias == "lfm2.5-1b-4bit")
    }

    @Test("An older catalog cannot select 1.2B when the starter fits")
    func oldCatalogCompatibilityFallback() {
        let catalog = [entry("lfm2.5-1b-4bit")]

        #expect(QuickstartCoordinator.defaultChoice(
            hardware: hardware(8),
            catalog: catalog
        ).alias == "lfm2.5-2.6b-4bit")
        #expect(QuickstartCoordinator.defaultChoice(
            hardware: hardware(16),
            catalog: catalog
        ).alias == "lfm2.5-2.6b-4bit")
    }

    @Test("A later cache refresh never overrides an explicit user choice")
    func explicitSelectionWins() {
        let coordinator = QuickstartCoordinator()
        let explicit = QuickstartCoordinator.choice(forAlias: "qwen3.5-9b-4bit")
        coordinator.select(explicit)
        coordinator.applyDefaultChoice(
            hardware: hardware(8),
            catalog: [entry("lfm2.5-2.6b-4bit", cached: true)]
        )
        #expect(coordinator.selection.alias == explicit.alias)
    }

    @Test("Catalog refresh cannot retarget a model being browsed or reviewed")
    func navigationFreezesAutomaticSelection() {
        let newCache = [entry("qwen3.5-9b-4bit", cached: true)]

        let browsing = QuickstartCoordinator()
        browsing.applyDefaultChoice(hardware: hardware(16), catalog: [])
        let browsingAlias = browsing.selection.alias
        browsing.beginBrowsingCatalog()
        browsing.applyDefaultChoice(hardware: hardware(16), catalog: newCache)
        #expect(browsing.selection.alias == browsingAlias)

        let incompatibleMemory = QuickstartCoordinator()
        incompatibleMemory.applyDefaultChoice(hardware: hardware(16), catalog: [])
        let incompatibleMemoryAlias = incompatibleMemory.selection.alias
        incompatibleMemory.beginIncompatibleModelReview(origin: .shortlist)
        incompatibleMemory.applyDefaultChoice(hardware: hardware(16), catalog: newCache)
        #expect(incompatibleMemory.selection.alias == incompatibleMemoryAlias)
    }

    @Test("The first authoritative catalog settles the shortlist selection")
    func authoritativeCatalogSettlesSelection() {
        let coordinator = QuickstartCoordinator()
        coordinator.applyDefaultChoice(hardware: hardware(16), catalog: [])
        coordinator.settleDefaultChoice(
            hardware: hardware(16),
            catalog: [entry("qwen3.5-4b-4bit")]
        )
        let settled = coordinator.selection.alias

        coordinator.applyDefaultChoice(
            hardware: hardware(16),
            catalog: [entry("qwen3.5-9b-4bit", cached: true)]
        )
        #expect(coordinator.selection.alias == settled)
    }

    @Test("Entering Step 2 before catalog load preserves cached preference")
    func deferredAuthoritativeCatalogStillWins() {
        let coordinator = QuickstartCoordinator()
        coordinator.applyDefaultChoice(hardware: hardware(15.99), catalog: [])
        coordinator.advanceToChooseModel()
        #expect(coordinator.selection.alias == "lfm2.5-2.6b-4bit")

        coordinator.settleDefaultChoice(
            hardware: hardware(15.99),
            catalog: [
                entry("lfm2.5-1b-4bit"),
                entry("lfm2.5-2.6b-4bit"),
                entry("qwen3.5-4b-4bit", cached: true),
            ]
        )

        #expect(coordinator.selection.alias == "qwen3.5-4b-4bit")
    }
}
