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
    @Test("A cached model with no measurable size claims no size at all")
    func cachedWithoutSizeClaimsNothing() {
        // `rapid-mlx ls` can hand back a cached row with no size (an external
        // model, a partial snapshot). This test replaces an earlier one that
        // asserted the opposite — fall through to the download estimate, on
        // the theory that an empty size lane reads as "free". codex was right
        // that the fallback is worse: it shows, and VoiceOver speaks, an
        // estimated DOWNLOAD figure as an on-disk size for a model that is
        // not being downloaded at all. That is the same defect this whole
        // file exists for ("download 20 GB" on a model already holding 19.03
        // GiB), one lane along. A wrong number is worse than no number, and
        // the accessibility label already degrades to a bare "on disk" when
        // the size text is empty.
        let choice = QuickstartCoordinator.lowMemoryChoice
        // Both shapes of "we could not measure it": absent and empty.
        for unmeasured in [entry(choice.alias, sizeOnDisk: nil),
                           entry(choice.alias, sizeOnDisk: "")] {
            let text = QuickstartView.shortlistSizeText(
                for: choice,
                cached: unmeasured,
                recommendedForPhysicalRAMGB: 256
            )
            #expect(text.isEmpty,
                    "A cached row with no measured size must not borrow the download estimate.")
            // The spoken label degrades to a bare "on disk" — no number, and
            // in particular not a download figure described as one.
            let spoken = QuickstartRecommendedCard.accessibilityText(
                for: choice, sizeText: text, isCached: true
            )
            #expect(spoken.contains("on disk"))
            #expect(!spoken.contains("download"))
        }
        // Sanity check: the uncached lane still produces a size, so the
        // assertions above are about provenance and not about an empty helper.
        #expect(!QuickstartView.shortlistSizeText(
            for: choice, cached: nil, recommendedForPhysicalRAMGB: 256
        ).isEmpty)
    }

}
