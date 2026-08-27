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
            kind: kind
        )
    }

    @Test("RAM threshold chooses 1.2B below 16 GB and Qwen 4B at 16 GB or above")
    func ramMatrix() {
        let catalog = [
            entry("lfm2.5-1b-4bit"),
            entry("lfm2.5-2.6b-4bit"),
            entry("qwen3.5-4b-4bit"),
        ]

        #expect(QuickstartCoordinator.defaultChoice(
            hardware: hardware(8), catalog: catalog
        ).alias == "lfm2.5-1b-4bit")
        #expect(QuickstartCoordinator.defaultChoice(
            hardware: hardware(15.99), catalog: catalog
        ).alias == "lfm2.5-1b-4bit")
        #expect(QuickstartCoordinator.defaultChoice(
            hardware: hardware(16), catalog: catalog
        ).alias == "qwen3.5-4b-4bit")
        #expect(QuickstartCoordinator.defaultChoice(
            hardware: hardware(64), catalog: catalog
        ).alias == "qwen3.5-4b-4bit")
        #expect(QuickstartCoordinator.baselineChoice(
            hardware: hardware(8)
        ).alias == "lfm2.5-1b-4bit")
        #expect(QuickstartCoordinator.baselineChoice(
            hardware: hardware(16)
        ).alias == "qwen3.5-4b-4bit")
    }

    @Test("The synchronous no-catalog baseline is safe before welcome Skip")
    func immediateWelcomeBaseline() {
        let coordinator = QuickstartCoordinator()
        coordinator.applyDefaultChoice(hardware: hardware(8), catalog: [])

        #expect(coordinator.selection.alias == "lfm2.5-1b-4bit")
        #expect(coordinator.seedMessage.contains("a model picked so you can start"))
    }

    @Test("The automatic 8 GB choice keeps its lowest-memory spoken category")
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

    @Test("The 8 GB starter provenance survives a deferred-seed relaunch")
    func baselineStarterSurvivesRelaunch() {
        let suite = "QuickstartStarterPolicyTests.relaunch.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suite)!
        defer { defaults.removePersistentDomain(forName: suite) }

        let first = QuickstartCoordinator(defaults: defaults)
        first.applyDefaultChoice(hardware: hardware(8), catalog: [])
        defaults.set(true, forKey: QuickstartCoordinator.awaitingSeedKey)
        defaults.set(first.selection.alias, forKey: QuickstartCoordinator.awaitingSeedAliasKey)

        let relaunched = QuickstartCoordinator(defaults: defaults)
        #expect(relaunched.selection.alias == "lfm2.5-1b-4bit")
        #expect(relaunched.seedMessage.contains("a model picked so you can start"))
    }

    @Test("The 1.2B choice remains a fallback, not a starter, on a 16 GB Mac")
    func lowMemoryChoiceIsContextual() {
        let coordinator = QuickstartCoordinator()
        coordinator.applyDefaultChoice(
            hardware: hardware(16),
            catalog: [entry("qwen3.5-4b-4bit"), entry("lfm2.5-1b-4bit")]
        )
        coordinator.select(QuickstartCoordinator.lowMemoryChoice)

        #expect(!coordinator.seedMessage.contains("a model picked so you can start"))
        #expect(coordinator.seedMessage.contains("running entirely on your Mac"))
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

    @Test("A cached standard starter stays visible when it wins below 16 GB")
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
        ).alias == "lfm2.5-1b-4bit")
    }

    @Test("A cached 2.6B model cannot bypass the safe 8 GB automatic baseline")
    func cachedCompactUpgradeRemainsManual() {
        let catalog = [
            entry("lfm2.5-1b-4bit"),
            entry("lfm2.5-2.6b-4bit", cached: true),
            entry("qwen3.5-4b-4bit"),
        ]

        #expect(QuickstartCoordinator.defaultChoice(
            hardware: hardware(8), catalog: catalog
        ).alias == "lfm2.5-1b-4bit")
    }

    @Test("The chooser presents one hardware-fit starter, not two competing recommendations")
    func shortlistHasOneStarter() {
        let catalog = [
            entry("lfm2.5-1b-4bit"),
            entry("lfm2.5-2.6b-4bit"),
            entry("qwen3.5-4b-4bit"),
        ]

        let compact = QuickstartView.shortlist(
            catalog: catalog,
            selection: "lfm2.5-1b-4bit",
            physicalRAMGB: 8
        )
        #expect(compact.starters.map(\.alias) == ["lfm2.5-1b-4bit"])
        #expect(compact.lowMemory.isEmpty)
        #expect(compact.recommended.map(\.alias) == ["lfm2.5-2.6b-4bit"])
        #expect(QuickstartView.recommendedGroupLabel(physicalRAMGB: 8)
            == "OPTIONAL — MORE CAPABLE, USES MORE MEMORY")

        let standard = QuickstartView.shortlist(
            catalog: catalog,
            selection: "qwen3.5-4b-4bit",
            physicalRAMGB: 16
        )
        #expect(standard.starters.map(\.alias) == ["qwen3.5-4b-4bit"])
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
        #expect(pick.alias == "qwen3.5-4b-4bit")
        #expect(QuickstartCoordinator.lowMemoryChoice.alias == "lfm2.5-1b-4bit")
    }

    @Test("An older authoritative catalog falls back to its 1.2B ladder entry")
    func oldCatalogCompatibilityFallback() {
        let catalog = [entry("lfm2.5-1b-4bit")]

        #expect(QuickstartCoordinator.defaultChoice(
            hardware: hardware(8),
            catalog: catalog
        ).alias == "lfm2.5-1b-4bit")
        #expect(QuickstartCoordinator.defaultChoice(
            hardware: hardware(16),
            catalog: catalog
        ).alias == "lfm2.5-1b-4bit")
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

        let reviewing = QuickstartCoordinator()
        reviewing.applyDefaultChoice(hardware: hardware(16), catalog: [])
        let reviewingAlias = reviewing.selection.alias
        reviewing.beginReviewDownload(origin: .shortlist)
        reviewing.applyDefaultChoice(hardware: hardware(16), catalog: newCache)
        #expect(reviewing.selection.alias == reviewingAlias)
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
        #expect(coordinator.selection.alias == "lfm2.5-1b-4bit")

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
