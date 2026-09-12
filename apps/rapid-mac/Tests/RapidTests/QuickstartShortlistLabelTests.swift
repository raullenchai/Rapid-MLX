import Testing
@testable import Rapid

/// 0.14.1 dogfood: on a Mac that already held `qwen3.6-35b-4bit` — all 19.03
/// GiB of it — the Quickstart shortlist offered it as "download 20 GB · 87%",
/// three rows under an "ALREADY ON THIS MAC" heading, next to a rail reading
/// "Free space 6 GB". Following that row means a 20 GB download the user does
/// not need and, at 6 GB free, cannot finish.
///
/// The cause was lane-shaped, not data-shaped: the cached shortlist is bounded
/// to six rows, so a seventh cached model is rendered by one of the
/// download-shaped lanes — and only the trade-up lane passed `isCached`. These
/// tests hold all four lanes to the same contract.
@MainActor
@Suite("Quickstart shortlist labels the models already on disk")
struct QuickstartShortlistLabelTests {

    private func entry(_ alias: String, sizeOnDisk: String? = "19.03 GiB") -> ModelEntry {
        ModelEntry(
            alias: alias,
            hfRepo: "mlx-community/\(alias)",
            sizeOnDisk: sizeOnDisk,
            cached: true
        )
    }

    @Test("A cached model quotes its bytes on disk, never a download estimate")
    func cachedQuotesBytesOnDisk() {
        let choice = QuickstartCoordinator.lowMemoryChoice
        let download = QuickstartView.sizeText(for: choice)
        let cached = QuickstartView.shortlistSizeText(
            for: choice,
            cached: entry(choice.alias),
            recommendedForPhysicalRAMGB: nil
        )
        #expect(cached == "19.03 GiB")
        #expect(cached != download)
    }

    @Test("A cached recommendation keeps its capability score")
    func cachedKeepsCapabilityScore() throws {
        // Derived from the SSOT rather than hard-coded, so a policy refresh
        // does not silently retarget this test at a different model.
        let ram = 256.0
        let pick = try #require(RAMBucketedDefault.picks(forPhysicalRAMGB: ram).first)
        let choice = QuickstartCoordinator.choice(forAlias: pick.alias)

        let cached = QuickstartView.shortlistSizeText(
            for: choice,
            cached: entry(pick.alias),
            recommendedForPhysicalRAMGB: ram
        )
        // Being on disk changes what it costs, not how capable it is.
        #expect(cached == "19.03 GiB · \(pick.capabilityPct)%")
    }

    @Test("An uncached row is byte-identical to the lane it replaced")
    func uncachedLanesAreUnchanged() {
        // The fix must be invisible on a fresh Mac — any drift here is a
        // regression in first-run copy, which is the most-seen screen we ship.
        let ram = 64.0
        for choice in QuickstartCoordinator.onboardingChoices {
            #expect(
                QuickstartView.shortlistSizeText(
                    for: choice, cached: nil, recommendedForPhysicalRAMGB: nil
                ) == QuickstartView.sizeText(for: choice)
            )
            #expect(
                QuickstartView.shortlistSizeText(
                    for: choice, cached: nil, recommendedForPhysicalRAMGB: ram
                ) == QuickstartView.sizeText(forRecommended: choice, physicalRAMGB: ram)
            )
        }
    }

    @Test("A cached entry with no size falls back rather than showing nothing")
    func cachedWithoutSizeFallsBack() {
        // `rapid-mlx ls` can hand back a row with no size (an external model,
        // a partial snapshot). An empty size lane would read as "free".
        let choice = QuickstartCoordinator.lowMemoryChoice
        for blank in [nil, ""] as [String?] {
            #expect(
                QuickstartView.shortlistSizeText(
                    for: choice,
                    cached: entry(choice.alias, sizeOnDisk: blank),
                    recommendedForPhysicalRAMGB: nil
                ) == QuickstartView.sizeText(for: choice)
            )
        }
    }

    @Test("VoiceOver says \"on disk\", not \"download\", in every lane")
    func spokenLabelsFollowTheCachedState() {
        let choice = QuickstartCoordinator.lowMemoryChoice
        let recommendedCached = QuickstartRecommendedCard.accessibilityText(
            for: choice, sizeText: "19.03 GiB", isCached: true
        )
        #expect(recommendedCached.contains("on disk 19.03 GiB"))
        #expect(!recommendedCached.contains("download"))

        let recommendedFresh = QuickstartRecommendedCard.accessibilityText(
            for: choice, sizeText: "633 MB", isCached: false
        )
        #expect(recommendedFresh.contains("download 633 MB"))

        let lowMemoryCached = QuickstartLowMemoryCard.accessibilityText(
            for: choice, sizeText: "19.03 GiB", isCached: true
        )
        #expect(lowMemoryCached.contains("On disk 19.03 GiB"))
        #expect(!lowMemoryCached.lowercased().contains("download"))

        let lowMemoryFresh = QuickstartLowMemoryCard.accessibilityText(
            for: choice, sizeText: "633 MB", isCached: false
        )
        #expect(lowMemoryFresh.contains("Download 633 MB"))

        // A cached row with no size still says which side of the download it
        // is on, rather than dropping the fact entirely.
        #expect(QuickstartRecommendedCard.accessibilityText(
            for: choice, sizeText: "", isCached: true
        ).contains("on disk"))
    }
}
