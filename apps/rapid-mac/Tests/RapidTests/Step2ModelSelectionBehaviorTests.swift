import Foundation
import Testing

@testable import Rapid

/// Paper 05.2 — Step 2 · Choose a model, as behaviour.
///
/// Step 2 grew from one screen to five micro-stages, and three of them can
/// offer the same model. That is exactly the shape where a UI drifts out of
/// agreement with itself: a button that says "Download & start" for a model
/// already on disk, a Return key that reaches something the user cannot see, a
/// Back that lands somewhere they never were. Every test here pins one of those
/// so it cannot come back quietly.
///
/// Two things are deliberately NOT tested by rendering: keyboard activation and
/// double-click. Neither has a seam SwiftUI exposes offscreen. They are pinned
/// instead at the two places where they are actually decided — the pure
/// derivation both of them run through, and the source wiring that routes them
/// there — which is where a regression would be introduced.
@MainActor
@Suite("Paper 05.2 — Step 2 model selection")
struct Step2ModelSelectionBehaviorTests {

    // MARK: - Fixtures

    private static func entry(
        _ alias: String,
        cached: Bool = false,
        kind: ModelKind = .chat,
        size: String? = nil
    ) -> ModelEntry {
        ModelEntry(
            alias: alias,
            hfRepo: "mlx-community/\(alias)",
            sizeOnDisk: cached ? (size ?? "2.9 GiB") : nil,
            cached: cached,
            kind: kind
        )
    }

    private static func row(
        _ alias: String,
        cached: Bool = false,
        available: Bool = true
    ) -> OnboardingModelSelection.Row {
        .init(alias: alias, isCached: cached, isAvailable: available)
    }

    private static func coordinator() -> QuickstartCoordinator {
        let coord = QuickstartCoordinator()
        coord._testingReset()
        return coord
    }

    private static var quickstartSource: String {
        get throws {
            let url = URL(fileURLWithPath: #filePath)
                .deletingLastPathComponent()
                .deletingLastPathComponent()
                .deletingLastPathComponent()
                .appendingPathComponent("Sources/Rapid/UI/QuickstartView.swift")
            return try String(contentsOf: url, encoding: .utf8)
        }
    }

    private static func componentsSource() throws -> String {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Sources/Rapid/UI/OnboardingComponents.swift")
        return try String(contentsOf: url, encoding: .utf8)
    }

    /// The Direction D design system, which owns the shared footer lane and
    /// therefore the Escape contract that used to live in the components file.
    private static func directionDSource() throws -> String {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Sources/Rapid/UI/OnboardingDirectionD.swift")
        return try String(contentsOf: url, encoding: .utf8)
    }

    /// Comment- and whitespace-stripped source, using the same helper the rest
    /// of the source-guard suites share. Comments must go, not just whitespace:
    /// otherwise a doc comment that *describes* a call counts as the call, and
    /// a wiring test passes on prose.
    private static func stripped(_ source: String) -> String {
        CapabilityChipRenderGateSourceGuardTests.stripCommentsAndWhitespace(source)
    }

    // MARK: - 1. Every micro-stage is still Step 2 of 4

    @Test("Every Step 2 micro-stage reports Step 2 of 4")
    func everyMicroStageIsStepTwo() {
        let coord = Self.coordinator()
        coord.advanceToChooseModel()
        for stage in QuickstartCoordinator.Step2Stage.allCases {
            switch stage {
            case .checkingHardware:
                coord.advanceToChooseModel()
            case .findingFit:
                coord.resolveRecommendationLoading(catalogLoaded: false)
            case .choosing:
                coord.resolveRecommendationLoading(catalogLoaded: true)
            case .browsing:
                coord.beginBrowsingCatalog()
            case .reviewing:
                coord.beginReviewDownload(origin: .shortlist)
            }
            #expect(coord.step2Stage == stage, "failed to reach \(stage)")
            #expect(
                coord.step == .chooseModel,
                "\(stage) reported \(coord.step) — a micro-stage is not a step"
            )
            #expect(coord.step.displayNumber == 2)
            #expect(QuickstartCoordinator.Step.total == 4)
            // Review is a one-way door in; get back out for the next iteration.
            if stage == .reviewing { coord.backFromReviewDownload() }
        }
        coord._testingReset()
    }

    /// The kicker is the only element that names the branch, so it is the one
    /// most likely to be "helpfully" sub-numbered into a fifth step.
    @Test("The micro-stage kicker names Step 2 of 4 and is never sub-numbered")
    func kickerFormatIsFixed() {
        let kicker = QuickstartView.microStageKicker("BROWSE ALL MODELS")
        #expect(kicker == "STEP 2 OF 4 · BROWSE ALL MODELS")
        #expect(!kicker.contains("2."), "no STEP 2.3 OF 4")
        #expect(!kicker.contains("OF 5"), "the catalogue must not add a step")
    }

    @Test("Every micro-stage renders the rail through one shared shell")
    func railIsRenderedOnce() throws {
        let body = Self.stripped(try Self.quickstartSource)
        // Direction D draws the rail once for the WHOLE surface, from
        // `coordinator.step`, rather than once per screen — which is how the
        // old three-step model ended up with two screens both claiming step 3.
        // The invariant is unchanged and is now enforced more strongly: no
        // Step 2 branch can render a rail at all, because none of them has one.
        let fullRail = body.components(separatedBy: "OnboardingSetupRail(").count - 1
        let compactRail = body.components(separatedBy: "OnboardingCompactRail(").count - 1
        #expect(fullRail == 1, "exactly one full-width rail call site, found \(fullRail)")
        #expect(compactRail == 1, "exactly one rotated-rail call site, found \(compactRail)")
        // Both live in the shell's rail planes, never inside a micro-stage.
        #expect(body.contains("privatevarrailPlane:someView{"))
        #expect(body.contains("privatevarcompactRailPlane:someView{"))
        // And every micro-stage still routes through the one scaffold.
        #expect(body.contains("privatefuncstep2Scaffold<Body:View,Footer:View>"))
    }

    @Test("The rail reports the macro step, and never a micro-stage")
    func railReportsTheMacroStepOnly() {
        // The rail reads `coordinator.step`, which is derived from (phase,
        // stage) and deliberately does not consult `step2Stage`. Browsing and
        // reviewing must therefore both still report Step 2.
        let coord = Self.coordinator()
        coord.advanceToChooseModel()
        #expect(coord.step == .chooseModel)
        coord.beginBrowsingCatalog()
        #expect(coord.step == .chooseModel, "the catalogue is not a step of its own")
        coord.beginReviewDownload(origin: .catalogue)
        #expect(coord.step == .chooseModel, "review is not a step of its own")
        #expect(QuickstartCoordinator.Step.total == 4)
    }

    // MARK: - 2. CTA derivation (Paper 05.2.G — canonical)

    @Test("A valid uncached selection derives Review download")
    func uncachedDerivesReview() {
        for context in [OnboardingModelSelection.ListContext.shortlist, .catalogue] {
            let primary = OnboardingModelSelection.primary(
                selection: "qwen3.5-9b-4bit",
                visibleRows: [Self.row("qwen3.5-9b-4bit"), Self.row("other", cached: true)],
                catalogState: .ready,
                context: context
            )
            #expect(primary.title == "Review download")
            #expect(primary.action == .reviewDownload)
            #expect(primary.isEnabled)
        }
    }

    @Test("A valid cached selection derives Start existing model")
    func cachedDerivesStartExisting() {
        for context in [
            OnboardingModelSelection.ListContext.shortlist, .catalogue, .review,
        ] {
            let primary = OnboardingModelSelection.primary(
                selection: "on-disk",
                visibleRows: [Self.row("on-disk", cached: true)],
                catalogState: .ready,
                context: context
            )
            #expect(primary.title == "Start existing model", "wrong verb in \(context)")
            #expect(primary.action == .startExisting)
            #expect(primary.isEnabled)
        }
    }

    /// The commit lives on one screen. If "Download & start" ever appears on a
    /// list, the review step has been bypassed.
    @Test("Download & start exists only on Review download")
    func commitVerbIsReviewOnly() {
        let rows = [Self.row("uncached")]
        #expect(
            OnboardingModelSelection.primary(
                selection: "uncached", visibleRows: rows,
                catalogState: .ready, context: .review
            ).action == .downloadAndStart
        )
        for context in [OnboardingModelSelection.ListContext.shortlist, .catalogue] {
            #expect(
                OnboardingModelSelection.primary(
                    selection: "uncached", visibleRows: rows,
                    catalogState: .ready, context: context
                ).action != .downloadAndStart,
                "\(context) must route through Review, not commit directly"
            )
        }
    }

    @Test("Loading, error, no-results and empty-cache all disable progression")
    func invalidContextsDisableProgression() {
        let cases: [(String, OnboardingModelSelection.Primary)] = [
            ("catalogue loading", OnboardingModelSelection.primary(
                selection: "a", visibleRows: [Self.row("a")],
                catalogState: .loading, context: .catalogue)),
            ("catalogue error", OnboardingModelSelection.primary(
                selection: "a", visibleRows: [],
                catalogState: .failed, context: .catalogue)),
            ("no results", OnboardingModelSelection.primary(
                selection: "a", visibleRows: [],
                catalogState: .ready, context: .catalogue)),
            ("empty cache under Cached", OnboardingModelSelection.primary(
                selection: "a", visibleRows: [],
                catalogState: .ready, context: .catalogue)),
            ("no selection", OnboardingModelSelection.primary(
                selection: nil, visibleRows: [Self.row("a")],
                catalogState: .ready, context: .shortlist)),
        ]
        for (name, primary) in cases {
            #expect(!primary.isEnabled, "\(name) must not allow progression")
            #expect(
                primary.title == "Review download",
                "\(name) must show the neutral verb, not a blank or a third label"
            )
        }
    }

    /// Superseded by 05.2.J · S7 — an empty result set used to leave the
    /// primary enabled, which let Return commit a model the list was not
    /// showing.
    @Test("An empty result set does not leave the primary enabled")
    func emptyResultsAreNotActionable() {
        let primary = OnboardingModelSelection.primary(
            selection: "mistral-7b", visibleRows: [],
            catalogState: .ready, context: .catalogue
        )
        #expect(!primary.isEnabled)
    }

    @Test("A selection hidden by search or filter is retained but not actionable")
    func hiddenSelectionIsRetainedButInert() {
        // Visible: the pick is actionable.
        let visible = OnboardingModelSelection.primary(
            selection: "qwen3.5-9b-4bit",
            visibleRows: [Self.row("qwen3.5-9b-4bit"), Self.row("gemma3-1b")],
            catalogState: .ready, context: .catalogue
        )
        #expect(visible.isEnabled)

        // Searched away: same alias, no longer in the visible set.
        let hidden = OnboardingModelSelection.primary(
            selection: "qwen3.5-9b-4bit",
            visibleRows: [Self.row("mistral-7b")],
            catalogState: .ready, context: .catalogue
        )
        #expect(!hidden.isEnabled, "a pick the user cannot see is not something they are choosing")
        #expect(hidden.title == "Review download")
    }

    @Test("Clearing the search restores the prior valid selection with no re-pick")
    func clearingSearchRestoresSelection() {
        let coord = Self.coordinator()
        let catalog = [
            Self.entry("qwen3.5-9b-4bit"),
            Self.entry("mistral-7b"),
            Self.entry("gemma3-1b", cached: true),
        ]
        coord.advanceToChooseModel()
        coord.beginBrowsingCatalog()
        coord.select(QuickstartView.choice(forCatalogEntry: catalog[0]))

        // Search for something else — the pick leaves the visible set.
        coord.catalogQuery = "mistral"
        var visible = QuickstartView.visibleCatalogEntries(
            catalog: catalog, query: coord.catalogQuery,
            filter: coord.catalogFilter, sort: coord.catalogSort
        )
        #expect(visible.map(\.alias) == ["mistral-7b"])
        #expect(coord.selection.alias == "qwen3.5-9b-4bit", "the alias is retained, not dropped")
        #expect(!OnboardingModelSelection.isActionable(
            selection: coord.selection.alias,
            visibleRows: visible.map { Self.row($0.alias, cached: $0.cached) },
            catalogState: .ready
        ))

        // Clear it — the row is re-admitted and the pick is live again, without
        // anything having re-selected it.
        coord.catalogQuery = ""
        visible = QuickstartView.visibleCatalogEntries(
            catalog: catalog, query: coord.catalogQuery,
            filter: coord.catalogFilter, sort: coord.catalogSort
        )
        #expect(coord.selection.alias == "qwen3.5-9b-4bit")
        #expect(OnboardingModelSelection.isActionable(
            selection: coord.selection.alias,
            visibleRows: visible.map { Self.row($0.alias, cached: $0.cached) },
            catalogState: .ready
        ))
        coord._testingReset()
    }

    /// Superseded in part by Paper 05.2.D: the row is now openable. What
    /// survives — and is the only part that ever mattered — is that nothing
    /// reachable from it commits.
    @Test("An unavailable model can be inspected but never committed")
    func unavailableModelIsInert() {
        // From a list: reachable, and reachable ONLY as an explanation.
        for context in [OnboardingModelSelection.ListContext.shortlist, .catalogue] {
            let primary = OnboardingModelSelection.primary(
                selection: "wont-fit",
                visibleRows: [Self.row("wont-fit", available: false)],
                catalogState: .ready, context: context
            )
            #expect(primary.isEnabled, "\(context): the detail must be reachable")
            #expect(primary.action == .reviewIncompatible, "\(context)")
            #expect(!primary.action.isCommit, "\(context): must not be a commit")
            // The catalogue's control never relabels itself between rows.
            #expect(primary.title == OnboardingModelSelection.Verb.reviewDownload)
        }
        // Inside Review: the verb it WOULD have taken, greyed.
        let review = OnboardingModelSelection.primary(
            selection: "wont-fit",
            visibleRows: [Self.row("wont-fit", available: false)],
            catalogState: .ready, context: .review
        )
        #expect(!review.isEnabled)
        #expect(review.title == OnboardingModelSelection.Verb.downloadAndStart)
    }

    /// Availability is the classification the model picker already disables on
    /// — not a new claim invented for onboarding.
    @Test("Availability comes from ModelSizing.classify, not from onboarding")
    func availabilityUsesTheExistingDecision() {
        let hardware = MacHardware.detect()
        for alias in ["qwen3.5-4b-4bit", "lfm2.5-1b-4bit", "qwen3-0.6b-4bit"] {
            let expected = ModelSizing.classify(
                ModelSizing.estimate(alias: alias), on: hardware
            ) != .tooBig
            #expect(OnboardingModelSelection.isAvailable(alias: alias, hardware: hardware) == expected)
        }
    }

    @Test("Cached-ness is read from the catalogue, never from copy or grouping")
    func cachedNessComesFromTheCatalogue() throws {
        let body = Self.stripped(try Self.quickstartSource)
        // The row's isCached must be fed by the catalogue snapshot.
        #expect(body.contains("isCached:Self.canStartWithoutDownload(alias:alias,cachedModels:cachedModels)"))
        // And the derivation must be structurally unable to branch on
        // presentation: it never imports SwiftUI and never sees the display
        // model, so there is no label, badge or card style in scope to read.
        let selection = try String(
            contentsOf: URL(fileURLWithPath: #filePath)
                .deletingLastPathComponent().deletingLastPathComponent()
                .deletingLastPathComponent()
                .appendingPathComponent("Sources/Rapid/UI/OnboardingModelSelection.swift"),
            encoding: .utf8
        )
        for forbidden in ["QuickstartModelChoice", "import SwiftUI", "displayName", "blurb"] {
            #expect(
                !selection.contains(forbidden),
                "the derivation must not see \(forbidden) — presentation is never evidence"
            )
        }
        // Its only inputs about a row are identity, cached-ness and availability.
        #expect(selection.contains("let alias: String"))
        #expect(selection.contains("let isCached: Bool"))
        #expect(selection.contains("let isAvailable: Bool"))
    }

    // MARK: - 3. Catalogue state truth

    @Test("An empty catalogue is the load-failure sentinel, not an empty cache")
    func emptyCatalogueIsAnError() {
        #expect(QuickstartView.catalogState(catalog: [], loaded: false) == .loading)
        #expect(QuickstartView.catalogState(catalog: [], loaded: true) == .failed)
        #expect(
            QuickstartView.catalogState(catalog: [Self.entry("a")], loaded: true) == .ready
        )
        // An empty CACHE still lists every downloadable alias — that is ready,
        // not failed, and it is what the Cached filter then finds nothing in.
        let emptyCache = [Self.entry("a"), Self.entry("b")]
        #expect(QuickstartView.catalogState(catalog: emptyCache, loaded: true) == .ready)
        #expect(
            QuickstartView.visibleCatalogEntries(
                catalog: emptyCache, query: "", filter: .cached, sort: .familyThenSize
            ).isEmpty
        )
    }

    /// Approved default D4, scoped to onboarding only.
    @Test("Browse all models offers chat models only")
    func catalogueIsChatOnly() {
        let mixed = [
            Self.entry("chat-a"),
            Self.entry("an-image-model", kind: .image),
            Self.entry("an-audio-model", kind: .audio),
            Self.entry("chat-b", cached: true),
        ]
        let visible = QuickstartView.visibleCatalogEntries(
            catalog: mixed, query: "", filter: .all, sort: .nameAscending
        )
        #expect(visible.map(\.alias) == ["chat-a", "chat-b"])
    }

    @Test("Search matches alias and Hugging Face repo through the shared primitive")
    func searchUsesTheSharedFilter() {
        let catalog = [Self.entry("qwen3.5-9b-4bit"), Self.entry("gemma3-1b")]
        #expect(
            QuickstartView.visibleCatalogEntries(
                catalog: catalog, query: "qwen", filter: .all, sort: .nameAscending
            ).map(\.alias) == ["qwen3.5-9b-4bit"]
        )
        // The repo half: "mlx-community" matches every row by repo, not alias.
        #expect(
            QuickstartView.visibleCatalogEntries(
                catalog: catalog, query: "mlx-community", filter: .all, sort: .nameAscending
            ).count == 2
        )
    }

    // MARK: - 4. Selection survives list transitions

    @Test("A catalogue pick returns to the shortlist as YOUR PICK, still selected")
    func cataloguePickComesBackAsYourPick() {
        let catalog = [
            Self.entry("exotic-13b"),
            Self.entry(QuickstartCoordinator.defaultChoice.alias),
        ]
        let list = QuickstartView.shortlist(catalog: catalog, selection: "exotic-13b")
        #expect(list.yourPick?.alias == "exotic-13b")
        #expect(list.visibleAliases.contains("exotic-13b"),
                "the shortlist must not disagree with the footer about what is selected")

        // And it disappears again once the selection moves back to a native row.
        let native = QuickstartView.shortlist(
            catalog: catalog, selection: QuickstartCoordinator.defaultChoice.alias
        )
        #expect(native.yourPick == nil)
    }

    @Test("A cached model past the shortlist's six-row bound is still reachable")
    func overflowCachedSelectionStaysVisible() {
        let catalog = (0..<9).map { Self.entry("cached-\($0)", cached: true) }
        let list = QuickstartView.shortlist(catalog: catalog, selection: "cached-8")
        #expect(list.cached.count == 6, "the presentation bound is unchanged")
        #expect(list.yourPick?.alias == "cached-8",
                "a pick outside the bound must not silently become unactionable")
        #expect(list.visibleAliases.contains("cached-8"))
    }

    @Test("Switching list context revalidates rather than assuming")
    func contextSwitchRevalidates() {
        let catalog = [Self.entry("exotic-13b"), Self.entry("gemma3-1b")]

        // This test is about context revalidation, not this runner's RAM.
        // Make availability explicit: `exotic-13b` correctly becomes too big
        // on smaller CI/Mini hardware, which made the old fixture machine-
        // dependent and turned a product truth into a false test failure.
        let catalogueRows = QuickstartView.visibleCatalogEntries(
            catalog: catalog, query: "", filter: .all, sort: .nameAscending
        ).map {
            OnboardingModelSelection.Row(alias: $0.alias, isCached: false, isAvailable: true)
        }
        #expect(OnboardingModelSelection.isActionable(
            selection: "exotic-13b", visibleRows: catalogueRows, catalogState: .ready
        ))

        // And carried onto the shortlist by YOUR PICK, so it stays actionable
        // across the transition rather than silently going dead.
        let list = QuickstartView.shortlist(catalog: catalog, selection: "exotic-13b")
        let shortlistRows = list.visibleAliases.map {
            OnboardingModelSelection.Row(alias: $0, isCached: false, isAvailable: true)
        }
        #expect(OnboardingModelSelection.isActionable(
            selection: "exotic-13b", visibleRows: shortlistRows, catalogState: .ready
        ))
    }

    // MARK: - 5. Activation: one action, three inputs

    /// Paper 05.2.G's truth table, exactly.
    @Test("Return and double-click resolve to the same action as the primary")
    func activationTruthTable() {
        let table: [(name: String, row: OnboardingModelSelection.Row,
                     action: OnboardingModelSelection.Action?, enabled: Bool)] = [
            ("uncached valid", Self.row("a"), .reviewDownload, true),
            ("cached valid", Self.row("a", cached: true), .startExisting, true),
            // Paper 05.2.D: from a LIST, an unrunnable pick opens its detail
            // and nothing else. The action is distinguishable from an ordinary
            // Review so that "this can never become a download" is a property
            // of the derivation rather than of every call site.
            ("unavailable", Self.row("a", available: false), .reviewIncompatible, true),
            ("unavailable cached", Self.row("a", cached: true, available: false),
             .reviewIncompatible, true),
        ]
        for entry in table {
            let primary = OnboardingModelSelection.primary(
                selection: "a", visibleRows: [entry.row],
                catalogState: .ready, context: .catalogue
            )
            #expect(primary.isEnabled == entry.enabled, "\(entry.name)")
            if let expected = entry.action {
                #expect(primary.action == expected, "\(entry.name)")
            }
        }
        // Nothing selected.
        let none = OnboardingModelSelection.primary(
            selection: nil, visibleRows: [Self.row("a")],
            catalogState: .ready, context: .catalogue
        )
        #expect(!none.isEnabled)
    }

    /// Superseded by 05.2.J · S6: double-click used to always open Review, even
    /// for a model already on disk, which is a detail screen about a download
    /// that will not happen.
    @Test("Double-click on a cached row starts it — it does not open Review")
    func doubleClickDoesNotAlwaysOpenReview() {
        let primary = OnboardingModelSelection.primary(
            selection: "on-disk", visibleRows: [Self.row("on-disk", cached: true)],
            catalogState: .ready, context: .catalogue
        )
        #expect(primary.action == .startExisting)
        #expect(primary.action != .reviewDownload)
    }

    @Test("Every activation input funnels through one shared path")
    func activationIsOneFunction() throws {
        let body = Self.stripped(try Self.quickstartSource)
        #expect(body.contains("privatefuncactivatePrimary(incontext:OnboardingModelSelection.ListContext){"))
        #expect(body.contains("letprimary=primary(for:context)guardprimary.isEnabledelse{return}"),
                "activation must re-derive and refuse a disabled primary before acting")

        // The footer primary, and the row double-click, on each list.
        for context in ["shortlist", "catalogue", "review"] {
            #expect(body.contains("onPrimary:{activatePrimary(in:.\(context))}"),
                    "the \(context) footer must activate through the shared path")
        }
        for context in ["shortlist", "catalogue"] {
            #expect(body.contains("onActivate:{activatePrimary(in:.\(context))}"),
                    "the \(context) rows' double-click must activate through the shared path")
        }
    }

    /// Return is AppKit's `.defaultAction` on the very control the derivation
    /// disables, which is what makes "Return cannot reach what the user cannot
    /// see" structural rather than re-implemented per screen.
    @Test("Return is the footer primary, and a disabled primary swallows it")
    func returnIsTheVisiblePrimary() throws {
        // Direction D moved the shared action lane out of the components file
        // and into the design system; the contract is unchanged, and it is
        // still one control rather than one per screen.
        let footer = Self.stripped(try Self.directionDSource())
        #expect(footer.contains(".disabled(!primaryEnabled).keyboardShortcut(.defaultAction)"),
                "the default action must sit on the control the derivation disables")
        #expect(footer.contains(#".accessibilityIdentifier("Quickstart.Footer.Primary")"#))
    }

    @Test("A row's single click selects and never navigates")
    func singleClickOnlySelects() throws {
        let body = Self.stripped(try Self.quickstartSource)
        // Catalogue rows: the tap closure selects and records the anchor. It
        // must not begin a review or a start.
        #expect(body.contains("{coordinator.select(choice)coordinator.rememberCatalogAnchor(entry.alias)}"))
        let components = Self.stripped(try Self.componentsSource())
        #expect(components.contains("simultaneousGesture(TapGesture(count:2)"),
                "double-click must be a separate gesture from the row's select action")
    }

    @Test("No per-row chevron and no hidden details route")
    func noDisclosureAffordance() throws {
        let body = try Self.quickstartSource
        #expect(!body.contains("chevron.right"), "Paper 05.2.G decided against a row chevron")
        #expect(!body.contains("chevron.forward"))
    }

    // MARK: - 6. Review download

    @Test("Review Back returns to the shortlist when that is where it came from")
    func reviewBackRestoresShortlist() {
        let coord = Self.coordinator()
        coord.advanceToChooseModel()
        coord.resolveRecommendationLoading(catalogLoaded: true)
        coord.beginReviewDownload(origin: .shortlist)
        #expect(coord.step2Stage == .reviewing)
        #expect(coord.reviewOrigin == .shortlist)

        coord.backFromReviewDownload()
        #expect(coord.step2Stage == .choosing)
        #expect(coord.stage == .chooseModel, "Back from Review must not land on Welcome (05.2.J · S2)")
        #expect(coord.step == .chooseModel)
        coord._testingReset()
    }

    @Test("Review Back returns to Browse all models when that is where it came from")
    func reviewBackRestoresCatalogue() {
        let coord = Self.coordinator()
        coord.advanceToChooseModel()
        coord.beginBrowsingCatalog()
        coord.beginReviewDownload(origin: .catalogue)

        coord.backFromReviewDownload()
        #expect(coord.step2Stage == .browsing)
        #expect(coord.step == .chooseModel)
        coord._testingReset()
    }

    @Test("Review Back restores query, filter, sort, scroll anchor and selection")
    func reviewBackRestoresListState() {
        let coord = Self.coordinator()
        let catalog = [
            Self.entry("qwen3.5-9b-4bit"),
            Self.entry("qwen3.5-4b-4bit"),
            Self.entry("gemma3-1b", cached: true),
        ]
        coord.advanceToChooseModel()
        coord.beginBrowsingCatalog()
        coord.catalogQuery = "qwen"
        coord.catalogFilter = .notCached
        coord.catalogSort = .nameAscending
        coord.select(QuickstartView.choice(forCatalogEntry: catalog[0]))
        coord.rememberCatalogAnchor("qwen3.5-9b-4bit")

        coord.beginReviewDownload(origin: .catalogue)
        coord.backFromReviewDownload()

        #expect(coord.catalogQuery == "qwen", "the search must come back verbatim")
        #expect(coord.catalogFilter == .notCached)
        #expect(coord.catalogSort == .nameAscending)
        #expect(coord.catalogScrollID == "qwen3.5-9b-4bit", "anchored by alias, not pixel offset")
        #expect(coord.selection.alias == "qwen3.5-9b-4bit")

        // And the restored list still contains the pick, so the primary comes
        // back enabled rather than pointing at something absent.
        let visible = QuickstartView.visibleCatalogEntries(
            catalog: catalog, query: coord.catalogQuery,
            filter: coord.catalogFilter, sort: coord.catalogSort
        )
        #expect(OnboardingModelSelection.isActionable(
            selection: coord.selection.alias,
            visibleRows: visible.map { Self.row($0.alias, cached: $0.cached) },
            catalogState: .ready
        ))
        coord._testingReset()
    }

    @Test("The catalogue records its visible alias anchor, not just the selection")
    func catalogueWiresRealScrollPosition() throws {
        let body = Self.stripped(try Self.quickstartSource)
        #expect(body.contains(".scrollTargetLayout()"))
        #expect(body.contains(".scrollPosition(id:catalogScrollPosition,anchor:.center)"))
        #expect(body.contains("ifletalias{coordinator.rememberCatalogAnchor(alias)}"))
        #expect(
            !body.contains("privatefuncreturnToRecommendedModels(){coordinator.rememberCatalogAnchor(coordinator.selection.alias)"),
            "Back must not replace the actual scroll anchor with the selected row"
        )
    }

    /// Order matters: rebuild, revalidate, then derive. If the alias is no
    /// longer in the restored context the primary must come back disabled
    /// rather than pointing at a row that is not there.
    @Test("Back re-derives the primary instead of assuming its prior state")
    func backRevalidatesBeforeEnabling() {
        let catalog = [Self.entry("qwen3.5-9b-4bit"), Self.entry("gemma3-1b", cached: true)]
        // The user changed the filter such that the pick is excluded.
        let visible = QuickstartView.visibleCatalogEntries(
            catalog: catalog, query: "", filter: .cached, sort: .nameAscending
        )
        #expect(visible.map(\.alias) == ["gemma3-1b"])
        let primary = OnboardingModelSelection.primary(
            selection: "qwen3.5-9b-4bit",
            visibleRows: OnboardingModelSelection.rows(for: visible, hardware: .detect()),
            catalogState: .ready, context: .catalogue
        )
        #expect(!primary.isEnabled)
    }

    @Test("Review is never the origin of another Review")
    func reviewDoesNotNest() {
        let coord = Self.coordinator()
        coord.advanceToChooseModel()
        coord.beginBrowsingCatalog()
        coord.beginReviewDownload(origin: .catalogue)
        // A second call, however it arrives, must not repoint Back at Review.
        coord.beginReviewDownload(origin: .shortlist)
        #expect(coord.reviewOrigin == .catalogue)
        coord.backFromReviewDownload()
        #expect(coord.step2Stage == .browsing)
        coord._testingReset()
    }

    @Test("Review shows only truthful, existing data for the selected model")
    func reviewFactsAreTruthful() {
        let cached = Self.entry("gemma3-1b", cached: true, size: "1.1 GiB")
        #expect(QuickstartView.reviewSizeText(alias: "gemma3-1b", cached: cached) == "1.1 GiB")
        // Uncached quotes the same estimate the rest of the app quotes.
        #expect(
            QuickstartView.reviewSizeText(alias: "qwen3.5-4b-4bit", cached: nil)
                == QuickstartView.sizeText(for: "qwen3.5-4b-4bit")
        )
        // An unknown alias says so rather than rendering a blank that reads as free.
        #expect(QuickstartView.reviewSizeText(alias: "", cached: nil) == "Unknown")
        // Free space is the real probe, and is omitted when there is no signal.
        #expect(QuickstartView.reviewFreeSpaceText(probe: { nil }) == nil)
        #expect(
            QuickstartView.reviewFreeSpaceText(probe: { 12_884_901_888 }) == "12.0 GB available"
        )
    }

    @Test("Review quotes no ETA and no benchmark claim")
    func reviewFabricatesNothing() throws {
        let body = try Self.quickstartSource
        guard let start = body.range(of: "private var reviewDownloadStep: some View"),
              let end = body.range(of: "// MARK: - Step 2 derivation (pure seams)")
        else {
            Issue.record("could not isolate the Review download section")
            return
        }
        let review = String(body[start.lowerBound..<end.lowerBound])
        for forbidden in ["ETA", "minutes remaining", "Accuracy", "benchmark"] {
            #expect(!review.contains(forbidden),
                    "Review must not claim \(forbidden) — it cannot compute it before the download")
        }
    }

    // MARK: - 7. Cached vs uncached routing

    @Test("A cached selection reaches Step 4 without a download or a fake Step 3")
    func cachedSelectionSkipsDownload() {
        let coord = Self.coordinator()
        coord.advanceToChooseModel()
        // startCachedModel's transition — no DownloadManager job is created.
        coord.enterStarting()
        #expect(coord.step == .start, "a cached start is Step 4")
        #expect(coord.step.displayNumber == 4)
        #expect(coord.phase != .downloading, "no fake download stage for a cached model")

        coord.enterReady()
        #expect(coord.phase == .ready)
        #expect(coord.step == .start)
        #expect(!coord.done, "PR #1917: readiness alone must still not complete onboarding")
        coord._testingReset()
    }

    @Test("An uncached selection proceeds through Step 3 Download")
    func uncachedSelectionRunsTheDownload() {
        let coord = Self.coordinator()
        coord.advanceToChooseModel()
        coord.enterDownloading()
        #expect(coord.step == .download)
        #expect(coord.step.displayNumber == 3)
        coord.enterStarting()
        #expect(coord.step == .start)
        coord._testingReset()
    }

    /// The cached branch must reuse the existing production route, not a second
    /// implementation of starting a model.
    @Test("Both enabled verbs run the one production start route")
    func oneStartRoute() throws {
        let body = Self.stripped(try Self.quickstartSource)
        #expect(body.contains("case.startExisting,.downloadAndStart:startQuickstart()"),
                "the cached and uncached commits must share startQuickstart()")
        #expect(body.contains("privatefuncstartCachedModel(_cached:ModelEntry){coordinator.enterStarting()"),
                "the cached route must still hand straight to ServerManager.start")
    }

    // MARK: - 8. Escape and Back priority

    @Test("Escape inside Browse all models or Review can only move one level in")
    func escapeNeverLeavesSetupFromASubStage() {
        let coord = Self.coordinator()
        coord.advanceToChooseModel()

        // Priority 3: from the catalogue, back to the shortlist.
        coord.beginBrowsingCatalog()
        #expect(coord.retreatWithinStep2())
        #expect(coord.step2Stage == .choosing)
        #expect(!coord.done)

        // Priority 2: from Review, back to its origin — and Review wins over
        // the catalogue rule when both could match.
        coord.beginBrowsingCatalog()
        coord.beginReviewDownload(origin: .catalogue)
        #expect(coord.retreatWithinStep2())
        #expect(coord.step2Stage == .browsing, "Review must yield to its origin, not to the shortlist")
        #expect(coord.retreatWithinStep2())
        #expect(coord.step2Stage == .choosing)

        // Priority 4: at the Step 2 root, onboarding's own meaning resumes and
        // the coordinator declines to handle the key.
        #expect(!coord.retreatWithinStep2(), "the Step 2 root must hand Escape back to onboarding")
        coord._testingReset()
    }

    @Test("The sheet's own dismissal is routed through the retreat first")
    func sheetDismissCannotSkipFromASubStage() throws {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Sources/Rapid/UI/ContentView.swift")
        let body = Self.stripped(try String(contentsOf: url, encoding: .utf8))
        #expect(
            body.contains("ifquickstart.retreatWithinStep2(){return}quickstartDismissedThisSession=true"),
            """
            A sheet dismissal must ask the coordinator to retreat before it is \
            treated as a skip, or Escape from two levels deep leaves setup.
            """
        )
    }

    @Test("A non-empty search field owns Escape; an empty one yields")
    func searchFieldOwnsEscapeOnlyWhenItHasText() throws {
        let body = Self.stripped(try Self.quickstartSource)
        #expect(
            body.contains(
                ".onKeyPress(.escape){guard!coordinator.catalogQuery.isEmptyelse{return.ignored}"
                    + "coordinator.catalogQuery=\"\"return.handled}"
            ),
            """
            Escape priority 1: a search field holding text clears itself and \
            stops the event; empty, it must decline so the footer's Back sees it.
            """
        )
    }

    @Test("Every Escape destination also has a visible control")
    func escapeMirrorsAVisibleControl() throws {
        let body = Self.stripped(try Self.quickstartSource)
        // Escape is `.cancelAction` on Back, and every Step 2 micro-stage
        // supplies a Back — so the key can only ever do what a visible control
        // already does.
        let directionD = Self.stripped(try Self.directionDSource())
        #expect(directionD.contains(".keyboardShortcut(.cancelAction).accessibilityIdentifier(\"Quickstart.Footer.Back\")"))
        #expect(body.contains(#"backTitle:"←Backtorecommendedmodels""#))
        #expect(body.contains(#""←Backtoallmodels""#))
        let footers = body.components(separatedBy: "OnboardingStepFooter(").count - 1
        #expect(footers >= 4, "each Step 2 micro-stage must carry its own footer, found \(footers)")
    }

    // MARK: - 9. PR #1917 is intact

    @Test("Readiness still parks and Start chatting is still the only completion")
    func priorContractSurvives() {
        let coord = Self.coordinator()
        coord.advanceToChooseModel()
        coord.beginBrowsingCatalog()
        coord.beginReviewDownload(origin: .catalogue)
        coord.enterDownloading()
        coord.enterStarting()
        coord.enterReady()

        #expect(coord.phase == .ready, "readiness parks; it does not dismiss")
        #expect(!coord.done, "readiness alone must not complete onboarding")
        #expect(coord.hasPendingReady)

        var seeded = 0
        #expect(coord.confirmStartChatting(seedWelcome: { seeded += 1; return true }))
        #expect(coord.done)
        #expect(seeded == 1)
        #expect(coord.phase == .dismissed)
        // Idempotent: a repeated activation seeds nothing more.
        #expect(!coord.confirmStartChatting(seedWelcome: { seeded += 1; return true }))
        #expect(seeded == 1)
        coord._testingReset()
    }

    @Test("Step 2 navigation never writes the completion flag")
    func navigationNeverCompletesOnboarding() {
        let coord = Self.coordinator()
        coord.advanceToChooseModel()
        for _ in 0..<3 {
            coord.beginBrowsingCatalog()
            coord.beginReviewDownload(origin: .catalogue)
            coord.backFromReviewDownload()
            coord.backToRecommendedModels()
            coord.beginReviewDownload(origin: .shortlist)
            coord.backFromReviewDownload()
        }
        #expect(!coord.done)
        #expect(coord.phase == .idle)
        #expect(coord.step == .chooseModel)
        coord._testingReset()
    }

    // MARK: - 10. Identifiers and keyboard semantics

    @Test("Existing Step 2 identifiers survive, and the new surfaces are addressable")
    func identifiersArePreserved() throws {
        let body = try Self.quickstartSource
        let required = [
            // Pre-existing — must not be renamed or dropped.
            #".accessibilityIdentifier("Quickstart.BrowseAll")"#,
            #".accessibilityIdentifier("Quickstart.Skip")"#,
            #".accessibilityIdentifier("Quickstart.GetStarted")"#,
            #".accessibilityIdentifier("Quickstart.Ready.StartChatting")"#,
            #".accessibilityIdentifier("Quickstart.CachedModel."#,
            // New in 05.2.
            #".accessibilityIdentifier("Quickstart.Step2.Kicker")"#,
            #".accessibilityIdentifier("Quickstart.BrowseAll.Search")"#,
            #".accessibilityIdentifier("Quickstart.BrowseAll.SortMenu")"#,
            #".accessibilityIdentifier("Quickstart.BrowseAll.Filter")"#,
            #".accessibilityIdentifier("Quickstart.BrowseAll.List")"#,
            // These reach `.accessibilityIdentifier` through the shared
            // `catalogNotice` / `reviewFact` builders, so the literal is what
            // the source carries.
            #"identifier: "Quickstart.BrowseAll.Loading""#,
            #"identifier: "Quickstart.BrowseAll.Error""#,
            #"identifier: "Quickstart.BrowseAll.NoResults""#,
            #"identifier: "Quickstart.BrowseAll.EmptyCache""#,
            #"identifier: "Quickstart.Review.Size""#,
            #"identifier: "Quickstart.Review.CachedStatus""#,
        ]
        for needle in required {
            #expect(body.contains(needle), "missing \(needle)")
        }
    }

    /// The container-identifier trap PR #1917 closed on the Ready screen: an
    /// identifier on a wrapper overwrites the child's, so the harness can see
    /// the box but never press the button inside it.
    @Test("No Step 2 container identifier shadows a control inside it")
    func noContainerShadowsAControl() throws {
        let body = try Self.quickstartSource
        #expect(!body.contains(#".accessibilityIdentifier("Quickstart.BrowseAll")"#
            + "\n                    .accessibilityIdentifier"))
        #expect(!body.contains(#".accessibilityIdentifier("Quickstart.Ready")"#),
                "the Ready container identifier must stay removed (#1917 review fix)")
    }
}
