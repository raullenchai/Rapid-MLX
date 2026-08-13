import Foundation
import Testing
@testable import Rapid

/// F-LWT-1 contract — the picker dropdown surfaces four sections in
/// a pinned order:
///
///   ┌── Quickstart ──────────────────────────────────────┐
///   │  lfm2.5-1b-4bit · Smallest model — fastest first   │ ← RAM-blind during first run
///   ├── Recommended for your <RAM> GB Mac ───────────────┤
///   │  Recommended <smart pick> · Faster <light pick>     │ ← RAM-tier measured pair
///   ├── All models (alphabetical) ───────────────────────┤
///   │  alpha … omega                                     │ ← dedup: Quickstart alias excluded
///   ├── Not fit for this Mac ────────────────────────────┤
///   │  oversized aliases remain downloadable            │
///   └────────────────────────────────────────────────────┘
///
/// The Quickstart section exists while first-run choice is still eligible,
/// so a user can one-click install the small coherent starter independently
/// of the RAM-aware smart/fast recommendations and the long-tail list.
///
/// The pinned section order matters because:
///   * Quickstart goes FIRST so a first-time browser sees the
///     smallest / fastest install as the top recommendation,
///     matching the Quickstart card's promise while first-run is eligible.
///   * Recommended goes SECOND because it's RAM-aware (different
///     advice on a 16 GB Mac vs a 64 GB Mac).
///   * All models contains runnable long-tail choices.
///   * Not fit for this Mac stays last: oversized choices remain
///     downloadable, but cannot masquerade as normal recommendations.
///
/// Dedup invariant: the Quickstart alias must NEVER appear in BOTH
/// the Quickstart section AND the All models section — otherwise
/// the same row renders twice and the user reads it as a UI bug.
/// The de-dup gate lives in
/// ``ModelPickerBar.dedupedAllEntries(filtered:quickstartRowRendered:)``.
@MainActor
@Suite("ModelPickerBar section order — F-LWT-1 Quickstart section")
struct ModelPickerBarSectionOrderTests {

    @Test("Oversized aliases are separated from runnable All models")
    func partitionsByConservativeFit() {
        let hardware = MacHardware(
            brandString: "Apple M3 Pro", family: .m3, tier: .pro,
            physicalRAMBytes: 18 * 1024 * 1024 * 1024,
            memoryBandwidthGBs: 150
        )
        let entries = [
            entry("qwen3.5-4b-4bit", hfRepo: "stub/qwen4", cached: false),
            entry("gemma-4-26b-4bit", hfRepo: "stub/gemma26", cached: false),
        ]
        let result = ModelPickerBar.partitionByFit(entries, hardware: hardware)
        #expect(result.fits.map(\.alias) == ["qwen3.5-4b-4bit"])
        #expect(result.notFit.map(\.alias) == ["gemma-4-26b-4bit"])
    }

    /// Synthetic catalog covering all three sections: one Quickstart
    /// alias, three Recommended-bucket aliases, four miscellaneous
    /// All-models aliases. Real catalog has 30+ entries; the
    /// truth-table only cares about the section-membership shape.
    private func makeCatalog() -> [ModelEntry] {
        return [
            entry(QuickstartCoordinator.defaultChoice.alias, hfRepo: "prism-ml/Ternary-Bonsai-1.7B-mlx-2bit", cached: false),
            entry("qwen3.5-9b-4bit", hfRepo: "mlx-community/Qwen3.5-9B-4bit", cached: false),
            entry("gemma3-1b-qat-4bit", hfRepo: "mlx-community/gemma-3-1b-it-qat-4bit", cached: false),
            entry("gemma-4-12b-4bit", hfRepo: "mlx-community/gemma-4-12b-4bit", cached: false),
            entry("alpha-test-3b-4bit", hfRepo: "stub/alpha", cached: false),
            entry("omega-test-7b-4bit", hfRepo: "stub/omega", cached: false),
        ]
    }

    @Test("Quickstart-eligible user keeps coherent LFM starter even when retired Bonsai is cached")
    func eligibleDefaultNeverResurrectsRetiredBonsai() {
        let catalog = [
            entry("bonsai-1.7b-2bit", hfRepo: "prism-ml/Ternary-Bonsai-1.7B-mlx-2bit", cached: true),
            entry(QuickstartCoordinator.defaultChoice.alias,
                  hfRepo: "mlx-community/LFM2.5-1.2B-Instruct-4bit",
                  cached: false),
            entry("bonsai-27b-2bit", hfRepo: "prism-ml/Ternary-Bonsai-27B-mlx-2bit", cached: false),
        ]

        let pick = ModelPickerBar.quickstartEligibleDefault(
            catalog: catalog,
            eligible: true
        )

        #expect(pick == "lfm2.5-1b-4bit")
        #expect(pick != "bonsai-1.7b-2bit")
    }

    @Test("Completed or ineligible user is left to normal cache/RAM default policy")
    func ineligibleDefaultDoesNotOverrideNormalPolicy() {
        let catalog = [
            entry("bonsai-1.7b-2bit", hfRepo: "prism-ml/Ternary-Bonsai-1.7B-mlx-2bit", cached: true),
            entry(QuickstartCoordinator.defaultChoice.alias,
                  hfRepo: "mlx-community/LFM2.5-1.2B-Instruct-4bit",
                  cached: false),
        ]

        #expect(ModelPickerBar.quickstartEligibleDefault(
            catalog: catalog,
            eligible: false
        ) == nil)
    }

    @Test("Auto-start off relaunch restores the last served non-retired model")
    func lastServedModelIsRestored() {
        let catalog = [
            entry("bonsai-1.7b-2bit", hfRepo: "prism-ml/Ternary-Bonsai-1.7B-mlx-2bit", cached: true),
            entry("qwen3-1.7b", hfRepo: "mlx-community/Qwen3-1.7B-4bit", cached: true),
        ]

        #expect(ModelPickerBar.lastServedDefault(
            catalog: catalog,
            lastServedAlias: "qwen3-1.7b"
        ) == "qwen3-1.7b")
    }

    @Test("Last served retired Bonsai is never restored automatically")
    func retiredLastServedModelIsRejected() {
        let catalog = [
            entry("bonsai-1.7b-2bit", hfRepo: "prism-ml/Ternary-Bonsai-1.7B-mlx-2bit", cached: true),
            entry(QuickstartCoordinator.defaultChoice.alias,
                  hfRepo: "mlx-community/LFM2.5-1.2B-Instruct-4bit",
                  cached: true),
        ]

        #expect(ModelPickerBar.lastServedDefault(
            catalog: catalog,
            lastServedAlias: "bonsai-1.7b-2bit"
        ) == nil)
    }

    @Test("Quickstart and automatic defaults agree on retired aliases")
    func retiredAliasPoliciesStayInSync() {
        #expect(QuickstartCoordinator.retiredStarters == CacheAwareDefault.retiredAutomaticAliases)
    }

    @Test("Deleted last-served model is not restored as a runnable default")
    func uncachedLastServedModelIsRejected() {
        let catalog = [
            entry("qwen3-1.7b", hfRepo: "mlx-community/Qwen3-1.7B-4bit", cached: false),
            entry(QuickstartCoordinator.defaultChoice.alias,
                  hfRepo: "mlx-community/LFM2.5-1.2B-Instruct-4bit",
                  cached: true),
        ]

        #expect(ModelPickerBar.lastServedDefault(
            catalog: catalog,
            lastServedAlias: "qwen3-1.7b"
        ) == nil)
    }

    @Test("Section order: Quickstart → Recommended → All (pinned by view-builder call order)")
    func sectionOrderPinned() throws {
        // The view-builder call order in ``ModelPickerBar.modelPicker``
        // pins the section order. Source-grep the file for the three
        // section calls inside the loaded-catalog `else` branch and
        // assert the offsets sort Quickstart < Recommended < All.
        let src = try modelPickerBarSource()
        // Anchor to the populated-catalog branch (skip the "catalog
        // unavailable" branch above).
        guard let anchor = src.range(of: "// v0.6.9: dropped the separate \"Cached\" section") else {
            Issue.record("Anchor docstring for populated-catalog branch missing in ModelPickerBar.swift")
            return
        }
        let scanRegion = src[anchor.lowerBound...]
        guard let qsIdx = scanRegion.range(of: "quickstartSection"),
              let recIdx = scanRegion.range(of: "recommendedSection"),
              let allIdx = scanRegion.range(of: "allAliasesSection") else {
            Issue.record("One of the three section view-builder calls is missing")
            return
        }
        #expect(qsIdx.lowerBound < recIdx.lowerBound,
                "Quickstart section must render BEFORE Recommended in the picker dropdown")
        #expect(recIdx.lowerBound < allIdx.lowerBound,
                "Recommended section must render BEFORE All models in the picker dropdown")
    }

    @Test("Quickstart section row title carries the alias + subtitle")
    func quickstartRowTitleShape() {
        let title = ModelPickerBar.quickstartRowTitle(alias: QuickstartCoordinator.defaultChoice.alias)
        #expect(title.contains(QuickstartCoordinator.defaultChoice.alias))
        #expect(title.contains(ModelPickerBar.quickstartSubtitle))
    }

    @Test("Quickstart subtitle stays under ~40 chars to match other section rows")
    func quickstartSubtitleLength() {
        // The Recommended section's "Default — qwen3.5-9b-4bit" row
        // sets the visual envelope. Keep the subtitle in the same
        // ballpark so the Quickstart row doesn't read as a different
        // surface from the rest of the dropdown.
        #expect(ModelPickerBar.quickstartSubtitle.count <= 40,
                "Quickstart subtitle is too long for the picker row: '\(ModelPickerBar.quickstartSubtitle)' (\(ModelPickerBar.quickstartSubtitle.count) chars)")
    }

    @Test("Quickstart row accessibility label includes alias + cached state + subtitle")
    func quickstartRowAccessibility() {
        let labelUncached = ModelPickerBar.quickstartRowAccessibilityLabel(
            alias: QuickstartCoordinator.defaultChoice.alias,
            cached: false
        )
        #expect(labelUncached.contains(QuickstartCoordinator.defaultChoice.alias))
        #expect(labelUncached.contains("not downloaded"))
        #expect(labelUncached.contains(ModelPickerBar.quickstartSubtitle))

        let labelCached = ModelPickerBar.quickstartRowAccessibilityLabel(
            alias: QuickstartCoordinator.defaultChoice.alias,
            cached: true
        )
        #expect(labelCached.contains("downloaded"))
        #expect(!labelCached.contains("not downloaded"))
    }

    // MARK: - Dedup invariant

    @Test("Dedup: when Quickstart row is rendered above, All models DROPS the Quickstart alias")
    func dedupStripsQuickstartFromAll() {
        let filtered = makeCatalog()
        let deduped = ModelPickerBar.dedupedAllEntries(
            filtered: filtered,
            quickstartRowRendered: true
        )
        let dedupedAliases = Set(deduped.map { $0.alias })
        #expect(!dedupedAliases.contains(QuickstartCoordinator.defaultChoice.alias),
                "Quickstart alias must not appear in All models when the Quickstart section is also rendering it")
        // The other entries survive.
        #expect(dedupedAliases.contains("qwen3.5-9b-4bit"))
        #expect(dedupedAliases.contains("alpha-test-3b-4bit"))
        #expect(dedupedAliases.contains("omega-test-7b-4bit"))
        #expect(deduped.count == filtered.count - 1)
    }

    @Test("Dedup: when Quickstart row is NOT rendered (catalog skew), All models KEEPS the Quickstart alias")
    func dedupKeepsQuickstartWhenSectionAbsent() {
        let filtered = makeCatalog()
        let deduped = ModelPickerBar.dedupedAllEntries(
            filtered: filtered,
            quickstartRowRendered: false
        )
        let dedupedAliases = Set(deduped.map { $0.alias })
        #expect(dedupedAliases.contains(QuickstartCoordinator.defaultChoice.alias),
                "Quickstart alias must remain in All models when the Quickstart section is not rendering (catalog skew / older rapid-mlx)")
        #expect(deduped.count == filtered.count)
    }

    @Test("Dedup invariant: any (filtered, true) input produces a set with NO duplicate Quickstart row")
    func dedupCannotRaceToTwoRows() {
        // Fuzz the dedup helper across a handful of filtered shapes
        // (empty, only-quickstart, double-quickstart, mixed) — the
        // invariant is "Quickstart alias appears at most once" when
        // the row is also rendered above.
        let qsAlias = QuickstartCoordinator.defaultChoice.alias
        let cases: [[ModelEntry]] = [
            [],
            [entry(qsAlias, hfRepo: "prism-ml/Ternary-Bonsai-1.7B-mlx-2bit", cached: true)],
            // Degenerate: two copies of the Quickstart alias in the
            // filtered list (shouldn't happen but the helper must
            // still strip BOTH).
            [
                entry(qsAlias, hfRepo: "prism-ml/Ternary-Bonsai-1.7B-mlx-2bit", cached: true),
                entry(qsAlias, hfRepo: "prism-ml/Ternary-Bonsai-1.7B-mlx-2bit", cached: false),
            ],
            makeCatalog(),
        ]
        for filtered in cases {
            let deduped = ModelPickerBar.dedupedAllEntries(
                filtered: filtered,
                quickstartRowRendered: true
            )
            let count = deduped.filter { $0.alias == qsAlias }.count
            #expect(count == 0,
                    "Quickstart alias appeared \(count) times in deduped output of \(filtered.count)-entry input")
        }
    }

    // MARK: - Quickstart in-flight gate (the Start CTA disabled-state)

    @Test("isQuickstartInFlight: nil coordinator → off (legacy / preview surface)")
    func inFlightNilOff() {
        #expect(!ModelPickerBar.isQuickstartInFlight(phase: nil))
    }

    @Test("isQuickstartInFlight: idle / dismissed / failed phases → off (CTA released)")
    func inFlightTerminalPhasesOff() {
        #expect(!ModelPickerBar.isQuickstartInFlight(phase: .idle))
        #expect(!ModelPickerBar.isQuickstartInFlight(phase: .dismissed))
        #expect(!ModelPickerBar.isQuickstartInFlight(
            phase: .failed(message: "boom", origin: .download)
        ))
    }

    @Test("isQuickstartInFlight: ready is in-flight — onboarding still owns the window")
    func inFlightReadyAwaitsConfirmation() {
        // Onboarding V3: ``.ready`` no longer means "handed off to chat".
        // The setup surface is still up, full-window, waiting for Start
        // chatting — so the picker's CTA stays gated.
        #expect(ModelPickerBar.isQuickstartInFlight(phase: .ready))
    }

    @Test("isQuickstartInFlight: lowDisk / downloading / starting → on (CTA disabled)")
    func inFlightActivePhasesOn() {
        #expect(ModelPickerBar.isQuickstartInFlight(
            phase: .lowDiskWarning(freeBytes: 1_000_000_000, requiredBytes: 3_000_000_000)
        ))
        #expect(ModelPickerBar.isQuickstartInFlight(phase: .downloading))
        #expect(ModelPickerBar.isQuickstartInFlight(phase: .starting))
    }

    // MARK: - Codex r1 MAJOR — quickstartPhaseGateKey transitions

    /// The codex r1 race fix added the catalog-availability bit to
    /// the ``.task(id:)`` gate key so the mirror observer re-fires
    /// when the catalog finally lands the Quickstart alias. Pin
    /// every transition of interest so a future refactor that drops
    /// the second axis (or collapses ``qs-absent`` and ``qs-present``
    /// into one bucket) regresses loudly. Codex r2 MAJOR.
    @Test("quickstartPhaseGateKey: off|qs-absent baseline (catalog empty, no in-flight)")
    func gateKeyOffAbsent() {
        let key = ModelPickerBar.quickstartPhaseGateKey(
            phase: nil,
            catalog: []
        )
        #expect(key == "off|qs-absent")
    }

    @Test("quickstartPhaseGateKey: off|qs-present (catalog landed, user hasn't clicked)")
    func gateKeyOffPresent() {
        let key = ModelPickerBar.quickstartPhaseGateKey(
            phase: .idle,
            catalog: makeCatalog()
        )
        #expect(key == "off|qs-present")
    }

    @Test("quickstartPhaseGateKey: in-flight|qs-absent (user clicked before catalog loaded)")
    func gateKeyInFlightAbsent() {
        let key = ModelPickerBar.quickstartPhaseGateKey(
            phase: .downloading,
            catalog: []
        )
        #expect(key == "in-flight|qs-absent")
    }

    @Test("quickstartPhaseGateKey: in-flight|qs-present (the codex r1 race target)")
    func gateKeyInFlightPresent() {
        let key = ModelPickerBar.quickstartPhaseGateKey(
            phase: .downloading,
            catalog: makeCatalog()
        )
        #expect(key == "in-flight|qs-present")
    }

    /// The race-fix transition: while Quickstart is downloading, the
    /// catalog gains the Quickstart row → mirror observer re-fires
    /// because the gate key flips. Without the catalog axis the
    /// observer would stay silent (both keys would be ``in-flight``).
    @Test("quickstartPhaseGateKey: in-flight|qs-absent → in-flight|qs-present re-fires (codex r1 race)")
    func gateKeyRaceTransition() {
        let before = ModelPickerBar.quickstartPhaseGateKey(
            phase: .downloading,
            catalog: []
        )
        let after = ModelPickerBar.quickstartPhaseGateKey(
            phase: .downloading,
            catalog: makeCatalog()
        )
        #expect(before != after,
                "gate key MUST change on catalog-landed-mid-flight transition; this is the codex r1 race fix")
    }

    @Test("quickstartPhaseGateKey: lowDisk and starting also map to in-flight half")
    func gateKeyInFlightAcrossActivePhases() {
        let cat = makeCatalog()
        let keys = [
            ModelPickerBar.quickstartPhaseGateKey(
                phase: .lowDiskWarning(freeBytes: 1_000, requiredBytes: 2_000),
                catalog: cat
            ),
            ModelPickerBar.quickstartPhaseGateKey(
                phase: .downloading,
                catalog: cat
            ),
            ModelPickerBar.quickstartPhaseGateKey(
                phase: .starting,
                catalog: cat
            ),
        ]
        for key in keys {
            #expect(key.hasPrefix("in-flight|"),
                    "active phase mapped to wrong gate half: \(key)")
        }
    }

    @Test("quickstartPhaseGateKey: dismissed / failed / idle all map to off half")
    func gateKeyOffAcrossInactivePhases() {
        let cat = makeCatalog()
        let keys = [
            ModelPickerBar.quickstartPhaseGateKey(phase: .idle, catalog: cat),
            ModelPickerBar.quickstartPhaseGateKey(phase: .dismissed, catalog: cat),
            ModelPickerBar.quickstartPhaseGateKey(
                phase: .failed(message: "boom", origin: .download),
                catalog: cat
            ),
            ModelPickerBar.quickstartPhaseGateKey(phase: nil, catalog: cat),
        ]
        for key in keys {
            #expect(key.hasPrefix("off|"),
                    "inactive phase mapped to wrong gate half: \(key)")
        }
    }
}

// MARK: - Convenience helpers for tests

/// Test factory — ``ModelEntry``'s synthesised init order is
/// ``(alias, hfRepo, sizeOnDisk, cached)``; the test cases want a
/// "size doesn't matter, just the section-membership shape"
/// shortcut.
private func entry(_ alias: String, hfRepo: String, cached: Bool) -> ModelEntry {
    return ModelEntry(
        alias: alias,
        hfRepo: hfRepo,
        sizeOnDisk: nil,
        cached: cached
    )
}

private func modelPickerBarSource() throws -> String {
    let url = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .appendingPathComponent("Sources/Rapid/UI/ModelPickerBar.swift")
    return try String(contentsOf: url, encoding: .utf8)
}
