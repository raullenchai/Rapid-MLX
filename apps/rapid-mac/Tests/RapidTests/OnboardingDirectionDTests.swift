import AppKit
import Foundation
import SwiftUI
import Testing
@testable import Rapid

/// Contract for the view logic the Direction D onboarding implementation
/// introduces (Paper 05.1 / 05.2).
///
/// This is a VISUAL slice, so most of it is composition and has no logic to
/// pin. What does have logic is every place the new surfaces had to *derive*
/// something in order to draw it — a lifecycle name, a badge, a byte line, a
/// fact table — and each of those is a place a redesign can quietly start
/// claiming something the app does not know.
///
/// The rule these all serve: the rail and the canvas may only state facts that
/// already exist. No synthesised percentage, no ETA before a measured rate, no
/// capability or compatibility claim beyond the ``ModelSizing`` classification
/// the primary is already gated on.
@MainActor
@Suite("Onboarding Direction D — derived display values")
struct OnboardingDirectionDTests {

    // MARK: - Full-window presentation

    private static func strippedSource(_ relativePath: String) throws -> String {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent(relativePath)
        return CapabilityChipRenderGateSourceGuardTests
            .stripCommentsAndWhitespace(try String(contentsOf: url, encoding: .utf8))
    }

    /// Setup owns the window; it is not a panel inside one.
    ///
    /// Manual verification caught this the hard way. Onboarding was presented
    /// with `.sheet`, which on macOS is a document-modal panel: AppKit sizes it
    /// to its content's ideal width, insets it, rounds its corners and leaves
    /// the parent visible around it — so Direction D rendered as a centred card
    /// over a dimmed chat surface, which is the one composition Paper 05.1.A
    /// rules out. It also settled at its 620pt minimum, below the 820pt
    /// breakpoint, so the full-height rail was unreachable on any display.
    @Test("Onboarding replaces the shell rather than floating over it")
    func onboardingIsNotPresentedAsASheet() throws {
        let content = try Self.strippedSource("Sources/Rapid/UI/ContentView.swift")

        #expect(
            !content.contains(".sheet(isPresented:quickstartSheetPresented)"),
            """
            Onboarding is being presented as a sheet again. A macOS sheet is \
            inset, rounded and leaves the parent window visible behind it — \
            Paper 05.1.A forbids exactly that ("no dimmed application, no modal \
            card floating over a live app").
            """
        )
        // The root branches between the two shells, so there is nothing behind
        // setup to show through.
        #expect(content.contains("ifquickstartVisible{onboardingShell}else{productionShell}"),
                "the root must swap the whole shell, not overlay one on the other")
        #expect(content.contains("privatevaronboardingShell:someView{"))
        #expect(content.contains("privatevarproductionShell:someView{"))
        // And the surface itself must not re-impose a panel-sized frame.
        #expect(!content.contains("quickstartSurface.frame(minWidth:620"),
                "the setup surface must take the window's size, not a panel's")
    }

    @Test("The setup surface fills whatever the window gives it")
    func setupSurfaceFillsTheWindow() throws {
        let view = try Self.strippedSource("Sources/Rapid/UI/QuickstartView.swift")
        #expect(view.contains(".frame(maxWidth:.infinity,maxHeight:.infinity)"),
                "the shell must expand to its container in both axes")
        // No maximum width anywhere in the shell: a cap would recreate the
        // centred-card look inside a full-width window.
        #expect(!view.contains(".frame(maxWidth:460)"),
                "the retired centred-card cap must not come back")
    }

    // MARK: - Two-column vertical centring

    /// Collects the laid-out frame of instrumented subviews.
    private struct FrameKey: PreferenceKey {
        static let defaultValue: [String: CGRect] = [:]
        static func reduce(value: inout [String: CGRect], nextValue: () -> [String: CGRect]) {
            value.merge(nextValue()) { _, new in new }
        }
    }

    private struct Probe: ViewModifier {
        let id: String
        func body(content: Content) -> some View {
            content.background(
                GeometryReader { proxy in
                    Color.clear.preference(
                        key: FrameKey.self,
                        value: [id: proxy.frame(in: .named("canvas"))]
                    )
                }
            )
        }
    }

    /// Render a Step 2 column layout and report where its right stack landed.
    ///
    /// The probe sits on the INNER stack, not on whatever wrapper the column
    /// uses. That distinction is the whole point: a greedy wrapper still
    /// reports a centred frame while the rows inside it stack from its top,
    /// which is exactly how the defect hid.
    @MainActor
    private static func measureRightStack<Wrapper: View>(
        canvasHeight: CGFloat = 824,
        footerHeight: CGFloat = 44,
        rows: [CGFloat] = [150, 70, 70, 70],
        @ViewBuilder wrap: @escaping (AnyView) -> Wrapper
    ) -> (stack: CGRect, usableMid: CGFloat) {
        var captured: [String: CGRect] = [:]
        let inner = AnyView(
            VStack(alignment: .leading, spacing: 10) {
                ForEach(Array(rows.enumerated()), id: \.offset) { _, height in
                    RoundedRectangle(cornerRadius: 12)
                        .fill(Color.gray.opacity(0.2))
                        .frame(height: height)
                }
            }
            .modifier(Probe(id: "stack"))
        )
        let columns = OnboardingStepColumns(
            kicker: "STEP 2 OF 4 · CHOOSE A MODEL",
            title: "Choose your\nfirst model",
            subtitle: "Start small — you can download bigger models anytime in Settings.",
            aside: { EmptyView() },
            content: { wrap(inner) }
        )
        let root = OnboardingCanvasLayout(principal: { columns }) {
            Color.clear.frame(height: footerHeight)
        }
        .coordinateSpace(name: "canvas")
        .onPreferenceChange(FrameKey.self) { captured = $0 }
        .environment(\.onboardingLayout, .wide)
        .frame(width: 1240, height: canvasHeight)

        let host = NSHostingView(rootView: root)
        host.frame = CGRect(x: 0, y: 0, width: 1240, height: canvasHeight)
        host.layoutSubtreeIfNeeded()
        _ = host.bitmapImageRepForCachingDisplay(in: host.bounds)
        RunLoop.main.run(until: Date().addingTimeInterval(0.35))
        host.layoutSubtreeIfNeeded()

        return (captured["stack"] ?? .zero, (canvasHeight - footerHeight) / 2)
    }

    @Test("The right model stack is centred, not pinned to the top")
    func rightColumnIsVerticallyCentred() {
        // Paper frame 04 · D1: the heading and the model list are one
        // principal group, centred together in the canvas above the footer.
        let measured = Self.measureRightStack { content in
            OnboardingIntrinsicColumn { content }
        }
        #expect(measured.stack.height > 0, "the stack did not lay out")
        #expect(
            abs(measured.stack.midY - measured.usableMid) <= 2,
            "the model stack midpoint must equal the usable canvas midpoint"
        )
        // And it kept its natural height rather than filling the canvas.
        #expect(measured.stack.height < measured.usableMid * 2 - 20,
                "the stack expanded to fill the canvas instead of staying intrinsic")
    }

    @Test("A greedy wrapper would top-pin the stack — the case this guards")
    func aGreedyWrapperTopPinsTheStack() {
        // The negative control. A bare ScrollView is vertically greedy, so the
        // enclosing centred HStack centres a child that is ALREADY full height
        // and the rows inside it start at its top. If this ever stops
        // differing from the intrinsic column above, the guard has gone blind.
        let greedy = Self.measureRightStack { content in
            ScrollView { content }
        }
        let intrinsic = Self.measureRightStack { content in
            OnboardingIntrinsicColumn { content }
        }
        #expect(greedy.stack.height > 0, "the greedy stack did not lay out")
        #expect(
            greedy.stack.midY < intrinsic.stack.midY - 20,
            "a bare ScrollView must pin the stack higher than an intrinsic column"
        )
    }

    @Test("A stack taller than the canvas still scrolls")
    func anOverflowingStackStillScrolls() {
        // The fallback must survive: when the list genuinely cannot fit,
        // filling the height and scrolling is the correct shape.
        let tall = Self.measureRightStack(
            rows: Array(repeating: 120, count: 12)
        ) { content in
            OnboardingIntrinsicColumn { content }
        }
        #expect(tall.stack.height > tall.usableMid * 2,
                "an overflowing stack must keep its full content height inside a scroller")
    }

    @Test("Both two-column stacks use the intrinsic column, not a bare scroller")
    func twoColumnStacksUseTheIntrinsicColumn() throws {
        let view = try Self.strippedSource("Sources/Rapid/UI/QuickstartView.swift")
        // The shortlist and the Review companion list are the two right-hand
        // columns; both must be intrinsic so the parent can centre them.
        let intrinsic = view.components(separatedBy: "OnboardingIntrinsicColumn{").count - 1
        #expect(intrinsic == 2,
                "expected the shortlist and Review columns to be intrinsic, found \(intrinsic)")
        // Browse all models is full-width and SHOULD fill, so exactly one bare
        // ScrollView remains — the catalogue list.
        let scrollers = view.components(separatedBy: "ScrollView{").count - 1
        #expect(scrollers == 1,
                "only the full-width catalogue may be a bare ScrollView, found \(scrollers)")
    }

    // MARK: - Responsive tiers

    @Test("The three layout tiers resolve at Paper's breakpoints")
    func layoutTiersResolveAtTheBreakpoints() {
        // 1440-class: full rail, heading beside the list.
        #expect(OnboardingLayout.resolve(width: 1440) == .wide)
        #expect(OnboardingLayout.resolve(width: OnboardingD.columnsMinWidth) == .wide)
        // 1000-class: rail narrows, canvas stacks (Paper 05.1.D).
        #expect(OnboardingLayout.resolve(width: OnboardingD.columnsMinWidth - 1) == .medium)
        #expect(OnboardingLayout.resolve(width: 1200) == .medium)
        #expect(OnboardingLayout.resolve(width: 1000) == .medium)
        #expect(OnboardingLayout.resolve(width: 820) == .medium)
        // Floor: the rail rotates (Paper 05.1.E).
        #expect(OnboardingLayout.resolve(width: 819) == .compact)
        #expect(OnboardingLayout.resolve(width: 720) == .compact)
    }

    /// A fresh install must MEET Step 2 in the two-column layout, not merely be
    /// able to reach it by dragging the window wider.
    ///
    /// The old default was 1200×820 — 90pt below the columns breakpoint — so
    /// every first run got the medium stacked tier and the composition Paper
    /// 05.1.A specifies was, in practice, unreachable without a manual resize.
    /// Pinned against the constant the scene actually uses rather than a
    /// literal, so the test cannot agree with a number the app has stopped
    /// using.
    @Test("The fresh-install window default opens in the wide layout")
    func defaultWindowSizeOpensWide() {
        #expect(MainWindowDefaults.width == 1440)
        #expect(MainWindowDefaults.height == 900)
        #expect(MainWindowDefaults.width > OnboardingD.columnsMinWidth,
                "the default must clear the columns breakpoint, not sit on it")
        #expect(OnboardingLayout.resolve(width: MainWindowDefaults.width) == .wide)
        #expect(OnboardingLayout.resolve(width: MainWindowDefaults.width).usesColumns)
    }

    /// The scene must consume the constant. Without this a future edit could
    /// hardcode a width back inline and leave the test above passing against a
    /// number nothing reads.
    @Test("The window scene takes its default size from the named constant")
    func defaultSizeIsWiredToTheConstant() throws {
        let app = try Self.strippedSource("Sources/Rapid/RapidApp.swift")
        #expect(app.contains(
            ".defaultSize(width:MainWindowDefaults.width,height:MainWindowDefaults.height)"
        ))
        // And the frame autosave still owns a returning user's size — the
        // default may never become a resize.
        #expect(app.contains(#"window.setFrameAutosaveName("Rapid.MainWindow")"#))
        #expect(app.contains("WindowFrameClamp.clamp(frame:window.frame,to:visibleFrame)"))
    }

    @Test("Every width from the floor up leaves the model rows usable")
    func everyTierLeavesUsableCanvas() {
        // Swept rather than sampled. Three fixed widths passed while 1200pt —
        // wide enough for the old breakpoint, too narrow for the columns —
        // squeezed the list to 332pt. A sweep finds that; three samples did
        // not.
        for width in stride(from: 720.0, through: 2000.0, by: 10.0) {
            let layout = OnboardingLayout.resolve(width: width)
            let railWidth = layout.isCompact ? 0 : layout.railWidth
            let gutters = layout.isCompact
                ? OnboardingD.compactGutter * 2
                : OnboardingD.canvasLeading + OnboardingD.canvasTrailing
            let content = width - railWidth - gutters
            #expect(content >= OnboardingD.rowMinWidth,
                    "at \(Int(width))pt the canvas leaves only \(Int(content))pt of content")
            if layout.usesColumns {
                let list = content - OnboardingD.headingColumnWidth - OnboardingD.columnGap
                #expect(list >= OnboardingD.rowMinWidth,
                        "at \(Int(width))pt the two-column list is only \(Int(list))pt wide")
            }
        }
    }

    @Test("The rail narrows before it rotates")
    func railNarrowsBeforeItRotates() {
        #expect(OnboardingLayout.wide.railWidth == OnboardingD.railWidth)
        #expect(OnboardingLayout.medium.railWidth == OnboardingD.railNarrowWidth)
        #expect(OnboardingD.railNarrowWidth < OnboardingD.railWidth)
        #expect(OnboardingLayout.wide.usesColumns)
        #expect(!OnboardingLayout.medium.usesColumns, "1000pt stacks, per Paper 05.1.D")
        #expect(!OnboardingLayout.compact.usesColumns)
        #expect(OnboardingLayout.compact.isCompact)
        #expect(!OnboardingLayout.medium.isCompact)
    }

    // MARK: - Selection narrative (Paper 05.1 states 04 / 05 / 06)

    private static let tradeUps = QuickstartCoordinator.onboardingChoices
        .filter { $0.tier == .tradeUp }

    private static func cachedEntry(_ alias: String) -> ModelEntry {
        ModelEntry(alias: alias, hfRepo: "r", sizeOnDisk: "1.0 GB", cached: true)
    }

    @Test("The starter selection tells the user to choose a first model")
    func starterSelectionUsesTheDefaultNarrative() {
        let narrative = QuickstartView.selectionNarrative(
            alias: QuickstartCoordinator.defaultChoice.alias,
            cachedModels: [],
            comparableTradeUps: Self.tradeUps
        )
        #expect(narrative == .chooseFirst)
        #expect(narrative.title == "Choose your\nfirst model")
    }

    @Test("The low-memory fallback keeps the default narrative")
    func lowMemorySelectionKeepsTheDefaultNarrative() {
        // Paper draws no separate frame for it, and inventing one would be a
        // claim about a choice the design deliberately leaves plain.
        let narrative = QuickstartView.selectionNarrative(
            alias: QuickstartCoordinator.lowMemoryChoice.alias,
            cachedModels: [],
            comparableTradeUps: Self.tradeUps
        )
        #expect(narrative == .chooseFirst)
    }

    @Test("A trade-up selection switches to the cost comparison")
    func tradeUpSelectionComparesCost() {
        for choice in Self.tradeUps {
            let narrative = QuickstartView.selectionNarrative(
                alias: choice.alias,
                cachedModels: [],
                comparableTradeUps: Self.tradeUps
            )
            #expect(narrative == .biggerAndCost, "\(choice.alias) should compare")
            #expect(narrative.title == "Bigger, and\nwhat it costs")
        }
    }

    @Test("A lone trade-up does not pretend to be a comparison")
    func loneTradeUpFallsBack() {
        // One column is a statement, not a comparison. If the shortlist ever
        // ships a single trade-up the heading must not claim otherwise.
        let single = [Self.tradeUps[0]]
        #expect(QuickstartView.selectionNarrative(
            alias: single[0].alias,
            cachedModels: [],
            comparableTradeUps: single
        ) == .chooseFirst)
    }

    @Test("A cached selection says it is already here, whatever its size")
    func cachedSelectionOutranksSize() {
        // Cached-ness wins over size: "you already have this" is the more
        // useful fact about a 9B on disk than what it would cost to fetch.
        let bigCached = Self.tradeUps.last!
        let narrative = QuickstartView.selectionNarrative(
            alias: bigCached.alias,
            cachedModels: [Self.cachedEntry(bigCached.alias)],
            comparableTradeUps: Self.tradeUps
        )
        #expect(narrative == .alreadyHere)
        #expect(narrative.title == "One is already\nhere.")
    }

    @Test("Every narrative carries its own subtitle")
    func everyNarrativeHasItsOwnSubtitle() {
        let hardware = Self.hardware32
        let subtitles = Set([
            QuickstartView.SelectionNarrative.chooseFirst,
            .biggerAndCost,
            .alreadyHere,
        ].map { QuickstartView.selectionSubtitle($0, hardware: hardware) })
        #expect(subtitles.count == 3, "each narrative must explain its own case")
        #expect(QuickstartView.selectionSubtitle(.alreadyHere, hardware: hardware)
            .contains("skips the download"))
        #expect(QuickstartView.selectionSubtitle(.biggerAndCost, hardware: hardware)
            .contains("32 GB"), "the comparison names this Mac's memory")
    }

    @Test("The comparison marks the pick and states only sourced figures")
    func comparisonColumnsAreSourced() {
        let picked = Self.tradeUps.last!
        let columns = QuickstartView.comparisonColumns(
            selection: picked.alias,
            tradeUps: Self.tradeUps,
            hardware: Self.hardware32
        )
        #expect(columns.count == Self.tradeUps.count)
        #expect(columns.filter(\.isPicked).count == 1, "exactly one column is the pick")
        #expect(columns.last?.isPicked == true)
        for column in columns {
            #expect(!column.download.isEmpty)
            #expect(column.memory.hasPrefix("≈") || column.memory == "Unknown")
            // Model estimates keep a decimal: 5.9 vs 6 is the comparison.
            #expect(column.memory == "Unknown" || column.memory.contains("."),
                    "a footprint estimate must not be rounded to a whole GB")
            #expect(["Comfortable", "Tight", "Won't fit"].contains(column.fit))
        }
    }

    @Test("Fit wording comes from the classification the primary is gated on")
    func fitWordingFollowsClassification() {
        #expect(QuickstartView.fitText(.recommended) == "Comfortable")
        #expect(QuickstartView.fitText(.borderline) == "Tight")
        #expect(QuickstartView.fitText(.tooBig) == "Won't fit")
    }

    @Test("Comparison headers name the parameter count being compared")
    func comparisonHeadersNameParameters() {
        for choice in Self.tradeUps {
            let title = QuickstartView.comparisonColumnTitle(for: choice)
            #expect(title.hasSuffix("B"), "\(choice.alias) header was \(title)")
        }
        #expect(Set(Self.tradeUps.map(QuickstartView.comparisonColumnTitle(for:))).count
                == Self.tradeUps.count, "columns must be distinguishable")
    }

    // MARK: - The click contract (unchanged by this visual pass)

    @Test("A single click selects and never navigates")
    func singleClickOnlySelects() {
        let coord = QuickstartCoordinator()
        coord._testingReset()
        defer { coord._testingReset() }
        coord.advanceToChooseModel()
        coord.resolveRecommendationLoading(catalogLoaded: true)
        #expect(coord.step2Stage == .choosing)

        for choice in QuickstartCoordinator.onboardingChoices {
            coord.select(choice)
            #expect(coord.selection.alias == choice.alias)
            #expect(coord.step2Stage == .choosing, "selecting must not navigate")
            #expect(coord.phase == .idle, "selecting must not start anything")
        }
    }

    @Test("Selecting re-resolves the narrative, so the heading follows the click")
    func selectionDrivesTheNarrative() {
        // The presentation mapping is a pure function of the selection, so a
        // click changes the heading by construction rather than by a separate
        // update path that could drift out of step.
        let starter = QuickstartCoordinator.defaultChoice.alias
        let tradeUp = Self.tradeUps.last!.alias
        #expect(QuickstartView.selectionNarrative(
            alias: starter, cachedModels: [], comparableTradeUps: Self.tradeUps
        ) != QuickstartView.selectionNarrative(
            alias: tradeUp, cachedModels: [], comparableTradeUps: Self.tradeUps
        ))
    }

    @Test("Double click activates the visible primary, and nothing else")
    func doubleClickMirrorsThePrimary() {
        // Paper 05.2.G "One action, three inputs" / approved default D3. The
        // row's double-click is a shortcut for whatever the footer currently
        // says — Review download on an uncached pick, Start existing model on
        // a cached one — never a separate route with its own rules.
        let rows = [
            OnboardingModelSelection.Row(alias: "uncached", isCached: false),
            OnboardingModelSelection.Row(alias: "cached", isCached: true),
        ]
        let uncached = OnboardingModelSelection.primary(
            selection: "uncached", visibleRows: rows, catalogState: .ready, context: .shortlist
        )
        #expect(uncached.action == .reviewDownload)
        #expect(uncached.title == OnboardingModelSelection.Verb.reviewDownload)

        let cached = OnboardingModelSelection.primary(
            selection: "cached", visibleRows: rows, catalogState: .ready, context: .shortlist
        )
        #expect(cached.action == .startExisting)

        // And a disabled primary makes the shortcut inert.
        let blocked = OnboardingModelSelection.primary(
            selection: "missing", visibleRows: rows, catalogState: .ready, context: .shortlist
        )
        #expect(!blocked.isEnabled)
    }

    @Test("Model details stay on the Review destination")
    func modelDetailsBelongToReview() {
        // "Model details" is the Review micro-stage, reached by the primary —
        // it must never replace the shortlist narrative in place.
        let coord = QuickstartCoordinator()
        coord._testingReset()
        defer { coord._testingReset() }
        coord.advanceToChooseModel()
        coord.resolveRecommendationLoading(catalogLoaded: true)
        coord.beginReviewDownload(origin: .shortlist)
        #expect(coord.step2Stage == .reviewing)
        #expect(coord.step == .chooseModel, "review is still Step 2")
        coord.backFromReviewDownload()
        #expect(coord.step2Stage == .choosing, "Back returns to the shortlist")
    }

    // MARK: - Transient states (Paper 05.1 states 02 / 03)

    @Test("Both pre-shortlist micro-stages are reachable and distinct")
    func transientStagesAreReachable() {
        let coord = QuickstartCoordinator()
        coord._testingReset()
        defer { coord._testingReset() }

        // Entering Step 2 lands on the hardware read, never straight on the
        // shortlist — the list cannot say what is cached yet.
        coord.advanceToChooseModel()
        #expect(coord.step2Stage == .checkingHardware)

        // An unresolved catalogue moves to the fit stage and stays there.
        coord.resolveRecommendationLoading(catalogLoaded: false)
        #expect(coord.step2Stage == .findingFit)
        coord.resolveRecommendationLoading(catalogLoaded: false)
        #expect(coord.step2Stage == .findingFit)

        // Only a landed snapshot opens the shortlist.
        coord.resolveRecommendationLoading(catalogLoaded: true)
        #expect(coord.step2Stage == .choosing)
    }

    @Test("A catalogue invalidated after the fact reports honestly")
    func staleCatalogueReturnsToFindingFit() {
        // A download completing bumps the cache generation. Reporting a stale
        // cached column would be worse than saying it is being re-read.
        let coord = QuickstartCoordinator()
        coord._testingReset()
        defer { coord._testingReset() }
        coord.advanceToChooseModel()
        coord.resolveRecommendationLoading(catalogLoaded: true)
        #expect(coord.step2Stage == .choosing)
        coord.resolveRecommendationLoading(catalogLoaded: false)
        #expect(coord.step2Stage == .findingFit)
    }

    @Test("Navigational micro-stages are never yanked by a catalogue reload")
    func reloadDoesNotDisturbNavigation() {
        let coord = QuickstartCoordinator()
        coord._testingReset()
        defer { coord._testingReset() }
        coord.advanceToChooseModel()
        coord.beginBrowsingCatalog()
        coord.resolveRecommendationLoading(catalogLoaded: false)
        #expect(coord.step2Stage == .browsing, "a reload must not eject the user")
        coord.beginReviewDownload(origin: .catalogue)
        coord.resolveRecommendationLoading(catalogLoaded: false)
        #expect(coord.step2Stage == .reviewing)
    }

    @Test("Both transient screens are wired, with no invented duration")
    func transientScreensAreWiredWithoutFakeDelay() throws {
        let view = try Self.strippedSource("Sources/Rapid/UI/QuickstartView.swift")
        #expect(view.contains("case.checkingHardware:checkingHardwareStep"))
        #expect(view.contains("case.findingFit:findingFitStep"))
        // Both screens name themselves through the shared transient scaffold,
        // which applies the identifier it is handed — so the literals appear
        // at the call sites and the modifier appears once.
        #expect(view.contains(#"identifier:"Quickstart.Step2.CheckingHardware""#))
        #expect(view.contains(#"identifier:"Quickstart.Step2.FindingFit""#))
        #expect(view.contains(".accessibilityIdentifier(identifier)"),
                "the transient scaffold must apply the identifier it is given")
        // Paper is explicit that neither may be dressed as a timed scan.
        for forbidden in ["Task.sleep", "DispatchQueue.main.asyncAfter", "repeatForever"] {
            #expect(!view.contains(forbidden.replacingOccurrences(of: " ", with: "")),
                    "the transient stages must not introduce an artificial delay (\(forbidden))")
        }
    }

    // MARK: - The rail

    @Test("The rail names all four steps, and only those")
    func railNamesTheFourSteps() {
        #expect(QuickstartCoordinator.Step.welcome.railTitle == "Welcome")
        #expect(QuickstartCoordinator.Step.chooseModel.railTitle == "Choose a model")
        #expect(QuickstartCoordinator.Step.download.railTitle == "Download")
        #expect(QuickstartCoordinator.Step.start.railTitle == "Start")
        #expect(QuickstartCoordinator.Step.allCases.count == 4)
    }

    @Test("The rail's spoken label keeps the exact golden-flow contract")
    func railLabelIsTheContractString() {
        // gui-golden-flows.sh matches this verbatim on Welcome, the chooser
        // and Ready. It is a contract, not styling — PR #1917 put it there to
        // prove the rail reports honest progress.
        for step in QuickstartCoordinator.Step.allCases {
            let label = OnboardingSetupRail.progressAccessibilityLabel(current: step)
            #expect(label == "Setup progress, step \(step.displayNumber) of 4")
        }
    }

    @Test("The rail's value carries the detail the label cannot")
    func railValueCarriesTheDetail() {
        let welcome = OnboardingSetupRail.progressAccessibilityValue(current: .welcome)
        #expect(welcome == "Welcome", "nothing is completed on step 1")

        let download = OnboardingSetupRail.progressAccessibilityValue(current: .download)
        #expect(download.hasPrefix("Download"))
        #expect(download.contains("Completed: Welcome, Choose a model"))
        #expect(!download.contains("Start"), "a step ahead of the user is not completed")
    }

    // MARK: - The D2 subject rail

    @Test("A pull with nothing observed yet is PREPARING, not DOWNLOADING")
    func lifecycleNameFollowsTheRealPhase() {
        // Paper draws these as separate states (09 and 14) and the difference
        // is real: one has bytes to report, the other has a request that
        // landed and no transfer yet.
        #expect(QuickstartView.downloadLifecycleName(phase: nil) == "PREPARING")
        #expect(QuickstartView.downloadLifecycleName(phase: .idle) == "PREPARING")
        #expect(QuickstartView.downloadLifecycleName(phase: .preparing) == "PREPARING")
        #expect(QuickstartView.downloadLifecycleName(phase: .warmingUp) == "PREPARING")
        #expect(QuickstartView.downloadLifecycleName(
            phase: .fetching(done: 0, total: 9, percent: 0)
        ) == "DOWNLOADING")
        #expect(QuickstartView.downloadLifecycleName(
            phase: .downloading(file: "f", done: "1", total: "2", percent: 50, speed: nil, eta: nil)
        ) == "DOWNLOADING")
    }

    @Test("The percentage floors, so it never reads 100% while files are landing")
    func percentFloors() {
        #expect(OnboardingSubjectRail.percentText(0) == "0%")
        #expect(OnboardingSubjectRail.percentText(0.436) == "43%")
        #expect(OnboardingSubjectRail.percentText(0.996) == "99%")
        #expect(OnboardingSubjectRail.percentText(1) == "100%")
        // Defensive clamping — a bad fraction must not print a nonsense figure.
        #expect(OnboardingSubjectRail.percentText(1.4) == "100%")
        #expect(OnboardingSubjectRail.percentText(-0.2) == "0%")
    }

    @Test("The byte line appears only once real disk growth is observed")
    func byteLineRequiresObservation() {
        let downloads = DownloadManager()
        let job = downloads._testingSeedJob(alias: "lfm2.5-1b-4bit", totalBytes: 633_000_000)
        // A seeded job has a total but no observation yet: a denominator alone
        // is not progress, and rendering "0 MB / 633 MB" would imply a
        // transfer that has not started.
        #expect(QuickstartView.subjectBytesLine(job: job) == nil)
        #expect(QuickstartView.subjectBytesLine(job: nil) == nil)
    }

    @Test("Measured bytes stay visible when an incorrect total is discarded")
    func byteLineKeepsTruthfulNumeratorAfterOverrun() {
        let downloads = DownloadManager()
        let job = downloads._testingSeedJob(alias: "lfm2.5-1b-4bit", totalBytes: 563 * 1024 * 1024)
        job.progress.seedDiskBaseline(bytes: 0)
        job.progress.applyDiskObservation(bytes: 633 * 1024 * 1024)

        #expect(job.progress.totalBytes == nil)
        #expect(QuickstartView.subjectBytesLine(job: job) == "633 MB downloaded")
        #expect(QuickstartView.downloadLifecycleName(progress: job.progress) == "DOWNLOADING")
    }

    @Test("The rate line appears only once a rate has been measured")
    func rateLineRequiresAMeasuredRate() {
        // Paper 05.1.A forbids an ETA before bytes move, so there is
        // deliberately no pre-download branch to test — only its absence.
        let downloads = DownloadManager()
        let job = downloads._testingSeedJob(alias: "lfm2.5-1b-4bit")
        #expect(QuickstartView.subjectRateLine(job: job) == nil)
        #expect(QuickstartView.subjectRateLine(job: nil) == nil)
    }

    // MARK: - Catalogue rows

    @Test("A runnable row shows its repo; an unrunnable one shows the reason")
    func catalogSubtitleSwapsForUnavailableRows() {
        let hardware = MacHardware(
            brandString: "Apple M2 Max",
            family: .m2,
            tier: .max,
            physicalRAMBytes: 32 * 1024 * 1024 * 1024,
            memoryBandwidthGBs: 400
        )
        let entry = ModelEntry(
            alias: "llama3.1-70b-4bit",
            hfRepo: "mlx-community/Llama-3.1-70B-Instruct-4bit",
            sizeOnDisk: nil,
            cached: false
        )

        let available = QuickstartView.catalogRowSubtitle(
            entry: entry, available: true, hardware: hardware
        )
        #expect(available == "mlx-community/Llama-3.1-70B-Instruct-4bit")

        let blocked = QuickstartView.catalogRowSubtitle(
            entry: entry, available: false, hardware: hardware
        )
        #expect(blocked.contains("Needs"))
        #expect(blocked.contains("32 GB"), "the reason must name what this Mac actually has")
        #expect(!blocked.contains("mlx-community"), "the reason replaces the repo, not joins it")
    }

    @Test("Row badges only ever restate facts the app already holds")
    func catalogBadgesAreClosedVocabulary() {
        let starter = ModelEntry(
            alias: QuickstartCoordinator.defaultChoice.alias,
            hfRepo: "r",
            sizeOnDisk: nil,
            cached: true
        )
        let starterBadges = QuickstartView.catalogRowBadges(entry: starter, available: true)
        #expect(starterBadges.map { $0.text } == ["RECOMMENDED", "ON THIS MAC"])

        let plain = ModelEntry(alias: "qwen3.5-4b-4bit", hfRepo: "r", sizeOnDisk: nil, cached: false)
        #expect(QuickstartView.catalogRowBadges(entry: plain, available: true).isEmpty,
                "an ordinary uncached row makes no claim at all")

        let cached = ModelEntry(alias: "qwen3.5-4b-4bit", hfRepo: "r", sizeOnDisk: nil, cached: true)
        #expect(QuickstartView.catalogRowBadges(entry: cached, available: true)
            .map { $0.text } == ["ON THIS MAC"])

        // WON'T FIT replaces the cached badge rather than stacking with it:
        // whether it is on disk stops being the useful fact once it cannot run.
        let tooBig = ModelEntry(alias: "llama3.1-70b-4bit", hfRepo: "r", sizeOnDisk: nil, cached: true)
        #expect(QuickstartView.catalogRowBadges(entry: tooBig, available: false)
            .map { $0.text } == ["WON'T FIT"])
    }

    // MARK: - Review download facts

    private var hardware32: MacHardware { Self.hardware32 }

    fileprivate static let hardware32 = MacHardware(
        brandString: "Apple M2 Max",
        family: .m2,
        tier: .max,
        physicalRAMBytes: 32 * 1024 * 1024 * 1024,
        memoryBandwidthGBs: 400
    )

    @Test("Review states the cost, and omits any row it cannot source")
    func reviewFactsAreSourcedOrAbsent() {
        let rows = QuickstartView.reviewFacts(
            alias: QuickstartCoordinator.defaultChoice.alias,
            cached: nil,
            cachedModels: [],
            hardware: hardware32,
            freeBytes: 400 * 1024 * 1024 * 1024
        )
        let labels = rows.map(\.label)
        #expect(labels.first == "Model", "the alias about to be pulled is named first")
        #expect(labels.contains("Download"))
        #expect(labels.contains("On this Mac"))
        #expect(labels.contains("Memory when loaded"))
        #expect(labels.contains("Free space"))
        // The golden-flow harness addresses this row by identifier.
        #expect(rows.contains { $0.identifier == "Quickstart.Review.Alias" })
        // One model, one number: Review must quote the same download figure
        // the card does, which for a pinned choice is its byte count and not
        // the parameter-derived estimate.
        let starter = QuickstartCoordinator.defaultChoice
        #expect(QuickstartView.reviewDownloadText(alias: starter.alias, cached: nil)
                == QuickstartView.sizeText(for: starter))
        #expect(rows.first { $0.label == "Download" }?.value
                == QuickstartView.sizeText(for: starter))
    }

    @Test("A probe with no signal removes the free-space row rather than guessing")
    func reviewOmitsFreeSpaceWithoutAProbe() {
        let rows = QuickstartView.reviewFacts(
            alias: QuickstartCoordinator.defaultChoice.alias,
            cached: nil,
            cachedModels: [],
            hardware: hardware32,
            freeBytes: nil
        )
        #expect(!rows.map(\.label).contains("Free space"))
    }

    @Test("A cached model reports what it occupies, not what it would cost")
    func reviewSwitchesToSizeOnDiskWhenCached() {
        let cached = ModelEntry(
            alias: "qwen3.5-4b-4bit",
            hfRepo: "r",
            sizeOnDisk: "2.2 GB",
            cached: true
        )
        let rows = QuickstartView.reviewFacts(
            alias: "qwen3.5-4b-4bit",
            cached: cached,
            cachedModels: [cached],
            hardware: hardware32,
            freeBytes: nil
        )
        #expect(rows.map(\.label).contains("Size on disk"))
        #expect(!rows.map(\.label).contains("Download"))
        #expect(rows.first { $0.label == "On this Mac" }?.value == "Already downloaded")
    }

    @Test("The Review footnote frames a cost only when there is one")
    func reviewFootnoteOnlyForUncached() {
        let uncached = QuickstartView.reviewFootnote(
            alias: QuickstartCoordinator.defaultChoice.alias,
            cached: nil
        )
        #expect(uncached != nil)
        #expect(uncached!.contains("once"))
        #expect(uncached!.lowercased().contains("no network"))

        let cached = ModelEntry(alias: "a", hfRepo: "r", sizeOnDisk: "1 GB", cached: true)
        #expect(QuickstartView.reviewFootnote(alias: "a", cached: cached) == nil,
                "nothing is being spent, so there is no cost to frame")
    }

    @Test("Memory figures round in exactly one place")
    func memoryFiguresShareOneFormatter() {
        // The catalogue row and the Review table both quote this Mac's memory.
        // If they rounded separately they could disagree by a gibibyte on the
        // same screen.
        #expect(QuickstartView.wholeGB(32.0) == "32 GB")
        #expect(QuickstartView.wholeGB(8.7) == "9 GB")
        // …and a model estimate does not.
        #expect(QuickstartView.preciseGB(8.7) == "8.7 GB")
        #expect(QuickstartView.preciseGB(5.94) == "5.9 GB")
        #expect(QuickstartView.wholeGB(0.4) == "0 GB")
    }

    // MARK: - Failure presentation

    @Test("A cancellation is drawn as a stop, not a fault")
    func cancellationGetsItsOwnGlyph() {
        #expect(QuickstartView.failureGlyph(for: .downloadCancelled) == "stop.circle")
        #expect(QuickstartView.failureGlyph(for: .downloadFailed) == "exclamationmark.triangle")
        #expect(QuickstartView.failureGlyph(for: .downloadSourceUnavailable) == "wifi.exclamationmark")
        #expect(QuickstartView.failureGlyph(for: .modelOutOfMemory) == "memorychip")
    }

    @Test("The failure kicker keeps the step the user was actually in")
    func failureKickerKeepsItsOriginStep() {
        // A failure never becomes a step of its own — a broken pull is still
        // Step 3, a load failure still Step 4.
        let download = QuickstartView.failureKicker(for: .downloadCancelled, origin: .download)
        #expect(download == "STEP 3 OF 4 · DOWNLOAD STOPPED")

        let start = QuickstartView.failureKicker(for: .modelLoadFailed, origin: .start)
        #expect(start == "STEP 4 OF 4 · COULDN'T LOAD")

        for kind in FailureDiagnosis.Kind.allCases {
            let kicker = QuickstartView.failureKicker(for: kind, origin: .download)
            #expect(kicker.hasPrefix("STEP 3 OF 4 · "))
            #expect(!kicker.contains("OF 5"), "a failure must never add a step")
        }
    }

    @Test("Cancellation and failure remain visibly different screens")
    func cancellationAndFailureLookDifferent() {
        // The same distinction the merged recovery slice made in the copy,
        // carried into the composition: a different glyph, a different tone,
        // a different kicker and a different heading.
        #expect(QuickstartView.failureGlyph(for: .downloadCancelled)
                != QuickstartView.failureGlyph(for: .downloadFailed))
        #expect(QuickstartView.failureKicker(for: .downloadCancelled, origin: .download)
                != QuickstartView.failureKicker(for: .downloadFailed, origin: .download))
        #expect(QuickstartView.failureTitle(for: .downloadCancelled)
                != QuickstartView.failureTitle(for: .downloadFailed))
        // Tone is chosen from severity, so the notice lane cannot go red.
        #expect(FailureDiagnosis.Kind.downloadCancelled.severity == .notice)
        #expect(FailureDiagnosis.Kind.downloadFailed.severity == .error)
    }
}
