import Foundation
import Testing

@testable import Rapid

/// Paper 05.2.D · `V3/Onb-2e-ReviewDownload-IncompatibleMemory` — a catalogue
/// row this Mac cannot run is openable, and what opens is an explanation.
///
/// ## What changed, and what did not
///
/// The row used to carry `.disabled(!available)`. That blocked the invalid
/// action (correct) by making the whole row take no click (too much): the user
/// could see WON'T FIT and the memory reason, and had no way to ask *why* or to
/// read the model's actual figures. Paper is explicit that this is the wrong
/// trade — "the user asked what this model is, and refusing to answer would be
/// worse than answering".
///
/// So the row is now a live control and the refusal moved to the primary of the
/// screen it opens. The invariant that mattered is unchanged and is stated more
/// strongly here than the `.disabled` ever stated it: no path from a WON'T FIT
/// row reaches ``DownloadManager``, the disk pre-flight, or
/// ``ServerManager/start(alias:hfPath:)``. Previously that was true because the
/// row swallowed clicks; now it is true because the one derivation every input
/// funnels through cannot produce a commit for it, in any context, cached or
/// not.
///
/// Source-level guards are used for the wiring — SwiftUI offers no seam to
/// observe "this row's click did nothing", and ViewInspector is not in this
/// target (#1492) — over the comment- and whitespace-stripped form shared with
/// the other Step 2 source-guard suites.
@MainActor
@Suite("Onboarding — WON'T FIT rows open a read-only Review")
struct OnboardingWontFitReviewTests {

    // MARK: - Fixtures

    /// A 32 GB M2 Max: 25.6 GB usable, so ``ModelSizing`` refuses anything
    /// needing more than 19.2 GB. Fixed rather than probed — the verdict under
    /// test must not depend on the machine running the suite.
    static let hardware32 = MacHardware(
        brandString: "Apple M2 Max",
        family: .m2,
        tier: .max,
        physicalRAMBytes: 32 * 1024 * 1024 * 1024,
        memoryBandwidthGBs: 400
    )

    /// Comfortably over the ceiling on ``hardware32``.
    static let tooBigAlias = "llama3.1-70b-4bit"
    /// Comfortably under it.
    static let fittingAlias = "qwen3.5-4b-4bit"

    private static func source(_ relativePath: String) throws -> String {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // RapidTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // rapid-mac
            .appendingPathComponent(relativePath)
        return try String(contentsOf: url, encoding: .utf8)
    }

    private static func stripped(_ relativePath: String) throws -> String {
        CapabilityChipRenderGateSourceGuardTests
            .stripCommentsAndWhitespace(try source(relativePath))
    }

    private static var quickstart: String {
        get throws { try stripped("Sources/Rapid/UI/QuickstartView.swift") }
    }
    private static var components: String {
        get throws { try stripped("Sources/Rapid/UI/OnboardingComponents.swift") }
    }
    private static var directionD: String {
        get throws { try stripped("Sources/Rapid/UI/OnboardingDirectionD.swift") }
    }

    private static func coordinator() -> QuickstartCoordinator {
        let coord = QuickstartCoordinator()
        coord._testingReset()
        return coord
    }

    private static func row(
        _ alias: String,
        cached: Bool = false,
        available: Bool = true
    ) -> OnboardingModelSelection.Row {
        .init(alias: alias, isCached: cached, isAvailable: available)
    }

    // MARK: - 1. The fixture is actually incompatible

    /// Everything below is meaningless if the alias under test happens to fit.
    /// Pinned against the real classification rather than a hand-set flag.
    @Test("The fixture model is genuinely too big for the fixture Mac")
    func fixtureIsIncompatible() {
        #expect(!OnboardingModelSelection.isAvailable(
            alias: Self.tooBigAlias, hardware: Self.hardware32
        ))
        #expect(OnboardingModelSelection.isAvailable(
            alias: Self.fittingAlias, hardware: Self.hardware32
        ))
        #expect(ModelSizing.classify(
            ModelSizing.estimate(alias: Self.tooBigAlias), on: Self.hardware32
        ) == .tooBig)
    }

    // MARK: - 2. The row is selectable

    @Test("The catalogue row is no longer disabled for an unrunnable model")
    func catalogRowIsNotDisabled() throws {
        let body = try Self.quickstart
        #expect(
            !body.contains(".disabled(!available)"),
            """
            A WON'T FIT row must stay a live control: Paper 05.2.D allows \
            opening its detail, and a disabled Button takes no click at all — \
            not even the one that selects. The refusal belongs on the primary \
            of the screen that opens, not on the row.
            """
        )
        // And the row still receives both a tap and the shared activation, so
        // "selectable" means the same thing it means for every other row.
        #expect(body.contains("onActivate:{activatePrimary(in:.catalogue)}"))
        #expect(body.contains("{coordinator.select(choice)coordinator.rememberCatalogAnchor(entry.alias)}"))
    }

    @Test("Selecting an unrunnable row records it, exactly like any other")
    func selectingAnUnrunnableRowSticks() {
        let coord = Self.coordinator()
        coord.advanceToChooseModel()
        coord.beginBrowsingCatalog()
        let entry = ModelEntry(
            alias: Self.tooBigAlias, hfRepo: "r", sizeOnDisk: nil, cached: false
        )
        coord.select(QuickstartView.choice(forCatalogEntry: entry))
        coord.rememberCatalogAnchor(entry.alias)
        #expect(coord.selection.alias == Self.tooBigAlias)
        #expect(coord.catalogScrollID == Self.tooBigAlias)
        coord._testingReset()
    }

    // MARK: - 3. Single click selects and does not navigate

    @Test("A single click on an unrunnable row selects and never navigates")
    func singleClickOnlySelects() throws {
        // Same guard the compatible rows carry: the tap closure selects and
        // records the anchor, and does nothing else. There is no separate
        // branch for unrunnable rows, which is the point — one row body.
        let body = try Self.quickstart
        let taps = body.components(
            separatedBy: "{coordinator.select(choice)coordinator.rememberCatalogAnchor(entry.alias)}"
        ).count - 1
        #expect(taps == 1, "one tap closure for every catalogue row, found \(taps)")
        // Navigation is only ever reached through the shared activation path.
        let components = try Self.components
        #expect(components.contains("simultaneousGesture(TapGesture(count:2)"),
                "double-click must remain a separate gesture from the row's select action")
    }

    // MARK: - 4. Double-click and Return mirror the visible primary

    @Test("Double-click and Return resolve to the informational Review")
    func doubleClickAndReturnMirrorThePrimary() throws {
        // Both inputs are the same call. The row's `onActivate` is
        // `activatePrimary(in: .catalogue)`, and Return is `.defaultAction` on
        // the footer whose action is the identical call — so neither can reach
        // an action the visible primary does not offer.
        let body = try Self.quickstart
        #expect(body.contains("onActivate:{activatePrimary(in:.catalogue)}"))
        #expect(body.contains("onPrimary:{activatePrimary(in:.catalogue)}"))
        let footer = try Self.directionD
        #expect(footer.contains(".disabled(!primaryEnabled).keyboardShortcut(.defaultAction)"))

        // And what that shared call resolves to, for this selection, is the
        // informational Review — the same thing the footer is showing.
        let primary = OnboardingModelSelection.primary(
            selection: Self.tooBigAlias,
            visibleRows: [Self.row(Self.tooBigAlias, available: false)],
            catalogState: .ready,
            context: .catalogue
        )
        #expect(primary.isEnabled)
        #expect(primary.action == .reviewIncompatible)
    }

    @Test("Activation routes both review actions to the same micro-stage")
    func bothReviewActionsOpenReview() throws {
        let body = try Self.quickstart
        #expect(
            body.contains("case.reviewDownload,.reviewIncompatible:coordinator.beginReviewDownload("),
            "an informational Review must be the same screen, not a second one"
        )
        // The commit cases stay a separate arm, and remain the only route into
        // the execution paths.
        #expect(body.contains("case.startExisting,.downloadAndStart:"))
    }

    // MARK: - 5. Review's primary is disabled

    @Test("Inside Review the primary is disabled and names what is withheld")
    func reviewPrimaryIsDisabled() {
        let uncached = OnboardingModelSelection.primary(
            selection: Self.tooBigAlias,
            visibleRows: [Self.row(Self.tooBigAlias, available: false)],
            catalogState: .ready,
            context: .review
        )
        #expect(!uncached.isEnabled)
        #expect(uncached.title == OnboardingModelSelection.Verb.downloadAndStart,
                "Paper draws the verb the model would have taken, greyed")

        let cached = OnboardingModelSelection.primary(
            selection: Self.tooBigAlias,
            visibleRows: [Self.row(Self.tooBigAlias, cached: true, available: false)],
            catalogState: .ready,
            context: .review
        )
        #expect(!cached.isEnabled)
        #expect(cached.title == OnboardingModelSelection.Verb.startExisting,
                "a model already here is not offered a download it does not need")
    }

    // MARK: - 6. No execution path is reachable

    @Test("No context produces a commit for an unrunnable model")
    func noContextEverCommits() {
        for cached in [false, true] {
            for context in [
                OnboardingModelSelection.ListContext.shortlist, .catalogue, .review,
            ] {
                let primary = OnboardingModelSelection.primary(
                    selection: Self.tooBigAlias,
                    visibleRows: [Self.row(Self.tooBigAlias, cached: cached, available: false)],
                    catalogState: .ready,
                    context: context
                )
                #expect(
                    !(primary.isEnabled && primary.action.isCommit),
                    "cached=\(cached) \(context): reached a commit"
                )
            }
        }
    }

    /// The structural half: activation refuses a disabled primary before it
    /// switches, and only the two commit cases call the execution route.
    @Test("Downloads, disk pre-flight and server start sit behind the commit arm")
    func executionPathsAreBehindTheGuard() throws {
        let body = try Self.quickstart
        #expect(
            body.contains("letprimary=primary(for:context)guardprimary.isEnabledelse{return}"),
            "activation must re-derive and refuse a disabled primary before acting"
        )
        // `startQuickstart` is the single door to the disk probe, the download
        // kickoff and the cached start. It must not be reachable from a review
        // arm, so the only call inside `activatePrimary` is under the commit
        // case checked above.
        let activation = body.components(separatedBy: "privatefuncactivatePrimary(")
        #expect(activation.count == 2, "one activation function")
        let afterGuard = activation[1].prefix(600)
        #expect(!afterGuard.contains("startQuickstart()downloadAndStart"))
        #expect(afterGuard.contains("case.reviewDownload,.reviewIncompatible:"))

        // And the three things a commit reaches are named in one place, so a
        // future case cannot pick one up silently.
        #expect(body.contains("privatefuncstartQuickstart(){"))
        #expect(body.contains("DiskSpaceProbe.decide("))
        #expect(body.contains("awaitserver.start("))
    }

    // MARK: - 7. Back returns to the exact origin

    @Test("Back from an informational Review returns to where it was opened")
    func backReturnsToOrigin() {
        for origin in [
            QuickstartCoordinator.ReviewOrigin.catalogue,
            .shortlist,
        ] {
            let coord = Self.coordinator()
            coord.advanceToChooseModel()
            if origin == .catalogue { coord.beginBrowsingCatalog() }
            let entry = ModelEntry(
                alias: Self.tooBigAlias, hfRepo: "r", sizeOnDisk: nil, cached: false
            )
            coord.select(QuickstartView.choice(forCatalogEntry: entry))
            coord.beginReviewDownload(origin: origin)
            #expect(coord.step2Stage == .reviewing)
            #expect(coord.reviewOrigin == origin)
            #expect(coord.step == .chooseModel, "Review is not a step of its own")

            coord.backFromReviewDownload()
            #expect(coord.step2Stage == (origin == .catalogue ? .browsing : .choosing),
                    "\(origin): Back must land on the list it was opened from")
            #expect(coord.selection.alias == Self.tooBigAlias,
                    "\(origin): the pick survives the round trip")
            coord._testingReset()
        }
    }

    // MARK: - 8. Browse All state survives the round trip

    @Test("Query, filter, sort, scroll anchor and selection all survive Back")
    func catalogueStateSurvivesAnInformationalReview() {
        let coord = Self.coordinator()
        coord.advanceToChooseModel()
        coord.beginBrowsingCatalog()
        coord.catalogQuery = "llama"
        coord.catalogFilter = .notCached
        coord.catalogSort = .sizeDescending
        let entry = ModelEntry(
            alias: Self.tooBigAlias, hfRepo: "r", sizeOnDisk: nil, cached: false
        )
        coord.select(QuickstartView.choice(forCatalogEntry: entry))
        coord.rememberCatalogAnchor(Self.tooBigAlias)

        coord.beginReviewDownload(origin: .catalogue)
        coord.backFromReviewDownload()

        #expect(coord.step2Stage == .browsing)
        #expect(coord.catalogQuery == "llama", "the query is restored verbatim")
        #expect(coord.catalogFilter == .notCached)
        #expect(coord.catalogSort == .sizeDescending)
        #expect(coord.catalogScrollID == Self.tooBigAlias,
                "the scroll anchor is an alias, not a pixel offset")
        #expect(coord.selection.alias == Self.tooBigAlias)
        coord._testingReset()
    }

    // MARK: - 9. Review content — truthful, derived, read-only

    @Test("The incompatible Review states the need, the Mac and the usable pool")
    func incompatibilityNoteIsDerived() {
        let note = QuickstartView.incompatibilityNote(
            alias: Self.tooBigAlias, hardware: Self.hardware32
        )
        let needed = ModelSizing.estimate(alias: Self.tooBigAlias).totalGB
        #expect(note.contains(QuickstartView.preciseGB(needed)),
                "the figure must be the same reading the fact table shows")
        #expect(note.contains("32 GB"), "this Mac's actual memory")
        #expect(note.contains(QuickstartView.preciseGB(Self.hardware32.usableRAMGB)))
        // No invented claim, no offer that does not exist.
        for forbidden in ["benchmark", "quality", "faster", "Free memory", "retry"] {
            #expect(!note.localizedCaseInsensitiveContains(forbidden),
                    "the explanation must not claim \(forbidden)")
        }
    }

    @Test("The footnote names the real ceiling, not just the usable pool")
    func footnoteNamesTheCeiling() throws {
        let footnote = try #require(
            QuickstartView.memoryHeadroomFootnote(hardware: Self.hardware32)
        )
        #expect(footnote.contains(QuickstartView.preciseGB(Self.hardware32.usableRAMGB)))
        #expect(footnote.contains(QuickstartView.preciseGB(
            ModelSizing.largestFittingGB(on: Self.hardware32)
        )), """
        Without the ceiling the screen is misleading at the margin: a 21 GB \
        model is refused on a Mac whose usable pool the same screen calls \
        25.6 GB. The limit is 75% of the pool and must be stated.
        """)
    }

    @Test("The ceiling is derived from the classification, not restated")
    func ceilingCannotDriftFromTheVerdict() {
        // A footprint just under the ceiling classifies as runnable; one just
        // over does not. If somebody changes the band in `classify` without
        // changing `largestFittingGB`, this fails.
        let ceiling = ModelSizing.largestFittingGB(on: Self.hardware32)
        #expect(ceiling > 0)
        let under = ModelSizing.Footprint(
            alias: "probe", paramsBillions: 1, bitsPerWeight: 4, weightsGB: ceiling * 0.98, baseOverheadGB: 0, kvReserveGB: 0
        )
        let over = ModelSizing.Footprint(
            alias: "probe", paramsBillions: 1, bitsPerWeight: 4, weightsGB: ceiling * 1.02, baseOverheadGB: 0, kvReserveGB: 0
        )
        #expect(ModelSizing.classify(under, on: Self.hardware32) != .tooBig)
        #expect(ModelSizing.classify(over, on: Self.hardware32) == .tooBig)
    }

    @Test("The subtitle states the verdict flatly, with no offer")
    func subtitleStatesTheVerdict() {
        #expect(QuickstartView.reviewSubtitle(cached: nil, runsHere: false)
            == "This model cannot run on this Mac.")
        // Cached-ness does not soften it: incompatibility outranks it.
        let cached = ModelEntry(
            alias: Self.tooBigAlias, hfRepo: "r", sizeOnDisk: "40 GB", cached: true
        )
        #expect(QuickstartView.reviewSubtitle(cached: cached, runsHere: false)
            == "This model cannot run on this Mac.")
        // And a runnable model is untouched.
        #expect(QuickstartView.reviewSubtitle(cached: nil, runsHere: true)
            == "This downloads once and then runs entirely on your Mac.")
        #expect(QuickstartView.reviewSubtitle(cached: cached, runsHere: true)
            == "Already on this Mac — nothing will be downloaded.")
    }

    @Test("The fact table gains the usable-memory row and flags the reason")
    func factsExplainTheVerdict() throws {
        let facts = QuickstartView.reviewFacts(
            alias: Self.tooBigAlias,
            cached: nil,
            cachedModels: [],
            hardware: Self.hardware32,
            freeBytes: 400 * Int64(1 << 30),
            runsHere: false
        )
        let ids = facts.compactMap(\.identifier)
        // Paper's order, and every row is one the screen already knew how to
        // show — the shape does not change between fitting and not.
        #expect(ids.contains("Quickstart.Review.Alias"))
        #expect(ids.contains("Quickstart.Review.Size"))
        #expect(ids.contains("Quickstart.Review.CachedStatus"))
        #expect(ids.contains("Quickstart.Review.Memory"))
        #expect(ids.contains("Quickstart.Review.UsableMemory"))
        #expect(ids.contains("Quickstart.Review.FreeSpace"))

        let memory = try #require(facts.first { $0.identifier == "Quickstart.Review.Memory" })
        #expect(memory.isAlert, "the offending number is the one marked")
        let usable = try #require(facts.first { $0.identifier == "Quickstart.Review.UsableMemory" })
        #expect(usable.value.contains("32 GB"))
        #expect(!usable.isAlert, "the pool is context, not the fault")

        // A model that fits does not carry the extra row, and nothing is
        // flagged.
        let fitting = QuickstartView.reviewFacts(
            alias: Self.fittingAlias,
            cached: nil,
            cachedModels: [],
            hardware: Self.hardware32,
            freeBytes: 400 * Int64(1 << 30)
        )
        #expect(!fitting.compactMap(\.identifier).contains("Quickstart.Review.UsableMemory"))
        #expect(fitting.allSatisfy { !$0.isAlert })
    }

    @Test("The informational Review offers no Free memory and retry")
    func noFreeMemoryAffordance() throws {
        // Paper 05.2.D is explicit: only the Step 4 pre-load guard has a live
        // MemoryProbe reading, so only it may offer this. Step 2's verdict is
        // a static estimate and cannot be retried into truth.
        let note = QuickstartView.incompatibilityNote(
            alias: Self.tooBigAlias, hardware: Self.hardware32
        )
        let footnote = QuickstartView.memoryHeadroomFootnote(hardware: Self.hardware32) ?? ""
        for text in [note, footnote] {
            #expect(!text.localizedCaseInsensitiveContains("free memory"))
            #expect(!text.localizedCaseInsensitiveContains("close other apps"))
        }
        // And Review's footer carries exactly one Back and one primary — no
        // third control was added for this state.
        let body = try Self.quickstart
        let reviewFooters = body.components(
            separatedBy: "onPrimary:{activatePrimary(in:.review)}"
        ).count - 1
        #expect(reviewFooters >= 1)
    }

    // MARK: - 10. Accessibility

    @Test("Every new surface is addressable and explains itself")
    func accessibilityIdentifiersAndLabels() throws {
        let body = try Self.quickstart
        #expect(body.contains(#".accessibilityIdentifier("Quickstart.Review.Incompatible")"#)
                || body.contains(#"identifier:"Quickstart.Review.Incompatible""#),
                "the explanation must be reachable by identifier")
        #expect(body.contains(#"identifier:"Quickstart.Review.UsableMemory""#))

        // The disabled primary carries a spoken reason. macOS says "dimmed"
        // and stops; the hint is the only place the why reaches VoiceOver.
        #expect(body.contains("primaryAccessibilityHint:runsHere?nil:Self.incompatiblePrimaryHint("))
        let hint = QuickstartView.incompatiblePrimaryHint(
            alias: Self.tooBigAlias, hardware: Self.hardware32
        )
        #expect(hint.hasPrefix("Unavailable."), "the state comes first")
        #expect(hint.contains("32 GB"), "and then the reason")

        let footer = try Self.directionD
        #expect(footer.contains(#".accessibilityIdentifier("Quickstart.Footer.Primary")"#))
        #expect(footer.contains(".accessibilityHint(primaryAccessibilityHint??\"\")"))
        #expect(footer.contains(#".accessibilityIdentifier("Quickstart.Footer.Back")"#))
        #expect(footer.contains(".accessibilityLabel(backAccessibilityLabel??backTitle)"))
    }

    @Test("The unrunnable row announces what it is and what opening it does")
    func rowAnnouncesItselfWithoutRelyingOnDimming() throws {
        let components = try Self.components
        // The label already carries alias, reason and badge; the hint carries
        // the consequence, which dimming alone cannot convey.
        #expect(components.contains(".accessibilityHint(accessibilityHint)"))
        #expect(
            components.contains(
                #""CannotrunonthisMac.Opensaread-onlyexplanation—nothingwillbedownloaded.""#
            ),
            "the hint must state the verdict AND that opening it costs nothing"
        )
        // A runnable row gets no hint: it behaves the way the rest of the list
        // does, and one sentence cannot honestly cover both a cached row
        // (which starts) and an uncached one (which opens Review).
        #expect(components.contains("isAvailable?\"\":"))

        // The reason itself is in the label, via the subtitle.
        let entry = ModelEntry(
            alias: Self.tooBigAlias, hfRepo: "mlx-community/x", sizeOnDisk: nil, cached: false
        )
        let subtitle = QuickstartView.catalogRowSubtitle(
            entry: entry, available: false, hardware: Self.hardware32
        )
        #expect(subtitle.contains("Needs"))
        #expect(subtitle.contains("32 GB"))
    }

    // MARK: - 11. Compatible models are untouched

    @Test("Runnable models keep the behaviour they had")
    func compatibleModelsAreUnchanged() {
        let uncached = OnboardingModelSelection.primary(
            selection: Self.fittingAlias,
            visibleRows: [Self.row(Self.fittingAlias)],
            catalogState: .ready, context: .catalogue
        )
        #expect(uncached == OnboardingModelSelection.Primary(
            title: OnboardingModelSelection.Verb.reviewDownload,
            action: .reviewDownload,
            isEnabled: true
        ))

        let inReview = OnboardingModelSelection.primary(
            selection: Self.fittingAlias,
            visibleRows: [Self.row(Self.fittingAlias)],
            catalogState: .ready, context: .review
        )
        #expect(inReview.action == .downloadAndStart)
        #expect(inReview.isEnabled)

        let cached = OnboardingModelSelection.primary(
            selection: Self.fittingAlias,
            visibleRows: [Self.row(Self.fittingAlias, cached: true)],
            catalogState: .ready, context: .catalogue
        )
        #expect(cached.action == .startExisting)
        #expect(cached.isEnabled)
    }

    @Test("A pick filtered out of view is still retained but not actionable")
    func filteringOutStillRemovesActionability() {
        // Unchanged by this PR, and re-pinned because the availability branch
        // now sits next to the visibility one: a pick that is not visible must
        // stay disabled whether or not it would have fitted.
        for available in [true, false] {
            let primary = OnboardingModelSelection.primary(
                selection: "hidden",
                visibleRows: [Self.row("other", available: available)],
                catalogState: .ready, context: .catalogue
            )
            #expect(primary == OnboardingModelSelection.disabledPrimary)
        }
    }

    @Test("A catalogue that has not spoken still disables everything")
    func loadingAndFailedOutrankAvailability() {
        for state in [
            OnboardingModelSelection.CatalogState.loading, .failed,
        ] {
            let primary = OnboardingModelSelection.primary(
                selection: Self.tooBigAlias,
                visibleRows: [Self.row(Self.tooBigAlias, available: false)],
                catalogState: state, context: .catalogue
            )
            #expect(primary == OnboardingModelSelection.disabledPrimary,
                    "\(state): nothing below the snapshot is knowable")
        }
    }

    // MARK: - 12. The visuals this PR must not change

    @Test("The WON'T FIT badge and the memory reason are exactly as they were")
    func badgeAndReasonVisualsAreUnchanged() {
        let entry = ModelEntry(
            alias: Self.tooBigAlias,
            hfRepo: "mlx-community/Llama-3.1-70B-Instruct-4bit",
            sizeOnDisk: nil,
            cached: true
        )
        // Still one badge, still replacing ON THIS MAC rather than stacking.
        let badges = QuickstartView.catalogRowBadges(entry: entry, available: false)
        #expect(badges.map { $0.text } == ["WON'T FIT"])
        #expect(badges.first?.tone == .error)

        // Still the reason in place of the repo.
        let subtitle = QuickstartView.catalogRowSubtitle(
            entry: entry, available: false, hardware: Self.hardware32
        )
        #expect(!subtitle.contains("mlx-community"))
        #expect(subtitle.contains("this Mac has 32 GB"))
    }

    @Test("The row keeps its muted treatment, driven by the same flag")
    func mutedTreatmentSurvives() throws {
        let components = try Self.components
        // `isAvailable` still drives every dimmed element. If somebody removes
        // the flag along with the `.disabled`, the row would read as an
        // ordinary candidate — which is the opposite of the intent.
        #expect(components.contains("tone:isAvailable?.neutral:.muted"))
        #expect(components.contains("foregroundStyle(isAvailable?RapidTheme.textPrimary:RapidTheme.textTertiary)"))
        #expect(components.contains("fill(isAvailable?RapidTheme.surfaceRaised:RapidTheme.surfaceCanvas)"))
        #expect(components.contains("OnboardingSelectionGlyph(isSelected:selected,isEnabled:isAvailable)"))
    }
}
