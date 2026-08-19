import Foundation
import Testing

@testable import Rapid

/// "Browse all models →" must have a destination, and that destination must be
/// inside setup.
///
/// ## Two bugs, one root cause
///
/// #1653: the whole implementation was one line that set a session dismiss
/// flag, so clicking the link closed the wizard, discarded the model the user
/// had chosen, and dropped them onto the chat surface pinned to whatever the
/// alphabetical fallback picked (a 7.6 GB download nobody asked for).
///
/// The fix routed it to the Settings model catalogue instead: stage a tab, end
/// the sheet's modal session, wait out an AppKit race, open a second window,
/// then restore the wizard on the way back. That worked, and it is what this
/// suite used to pin. Paper 05.2.J · S1 supersedes it — the catalogue still
/// lived somewhere onboarding was not, and every one of those five steps was a
/// chance to lose the user or their pick.
///
/// Browse all models is now a micro-stage inside Step 2. The invariants this
/// suite has always protected — the control leads somewhere, the selection
/// survives, setup is not dismissed — are unchanged; only the destination
/// moved, and it moved somewhere with strictly fewer ways to fail.
///
/// Source-level for the wiring, because the failure mode is a missing call and
/// SwiftUI gives no seam to observe "this button's action did nothing". The
/// behavioural half runs against the real coordinator, and
/// `gui-golden-flows.sh browse-all-destination` presses the real control.
@MainActor
@Suite("Quickstart — Browse all models opens the in-window catalogue")
struct QuickstartBrowseAllDestinationTests {
    private static var quickstartSource: String {
        get throws {
            let url = URL(fileURLWithPath: #filePath)
                .deletingLastPathComponent()  // RapidTests
                .deletingLastPathComponent()  // Tests
                .deletingLastPathComponent()  // rapid-mac
                .appendingPathComponent("Sources/Rapid/UI/QuickstartView.swift")
            return try String(contentsOf: url, encoding: .utf8)
        }
    }

    /// Comment- and whitespace-stripped source, shared with the other
    /// source-guard suites. Comments must go, not just whitespace: otherwise a
    /// doc comment that *describes* a call counts as the call, and a wiring
    /// test passes on prose.
    private static func stripped(_ source: String) -> String {
        CapabilityChipRenderGateSourceGuardTests.stripCommentsAndWhitespace(source)
    }

    private static func coordinator() -> QuickstartCoordinator {
        let coord = QuickstartCoordinator()
        coord._testingReset()
        return coord
    }

    // MARK: - The link is wired

    @Test("The link's action calls browseAllModels(), not a dismiss flag")
    func browseAllIsWiredToTheCatalogue() throws {
        // Matched on comment- and whitespace-stripped source rather than raw
        // text: the invariant is which function the control calls, and pinning
        // the indentation as well made a pure re-layout fail a wiring test.
        let source = Self.stripped(try Self.quickstartSource)
        #expect(
            source.contains(#"Button("Browseallmodels→"){browseAllModels()}"#),
            """
            The "Browse all models →" button no longer calls browseAllModels(). \
            If it is back to a dismiss closure, that is #1653 returning: the \
            wizard vanishes and the user's selection is discarded.
            """
        )
    }

    /// The failure card's route back to choosing.
    ///
    /// It used to be this same "or browse all models →" link, which satisfied
    /// the invariant — the control leads somewhere, inside setup — but always
    /// landed the user in the catalogue even when they had never opened it.
    /// The onboarding-recovery slice replaced it with a return to the
    /// micro-stage they actually left. The invariant is unchanged and the
    /// destination is strictly better informed; what must never come back is a
    /// failure card with no way out, or one whose way out leaves setup.
    @Test("The failure card offers a wired route back to model selection")
    func failureCardLinkIsWiredToo() throws {
        let source = Self.stripped(try Self.quickstartSource)
        #expect(
            source.contains(
                "Button(Self.failureBackTitle(for:coordinator.step2Stage))"
                + "{returnToModelSelection()}"
            ),
            """
            The failure card's Back control must call returnToModelSelection() \
            and label itself from the origin micro-stage. It is offered \
            precisely when the user's chosen model failed, which is the moment \
            they most need to pick a different one.
            """
        )
    }

    @Test("Returning from a failure re-enters Step 2 without leaving setup")
    func failureReturnIsWiredToTheChooser() throws {
        let source = try Self.quickstartSource
        guard let start = source.range(of: "private func returnToModelSelection() {"),
              let end = source.range(of: "\n    }", range: start.upperBound..<source.endIndex)
        else {
            Issue.record("could not isolate returnToModelSelection()")
            return
        }
        let function = String(source[start.lowerBound..<end.upperBound])
        #expect(
            function.contains("coordinator.returnToChooser()"),
            """
            returnToModelSelection() must go through returnToChooser(), which \
            leaves step2Stage alone — that is what returns the user to the \
            shortlist, the catalogue or Review download rather than a default.
            """
        )
        // Same bans as the browse path: a failure is not a way out of setup.
        for needle in ["openWindow", "dismiss()", "settingsRouter", "onSkip", ".sheet"] {
            #expect(
                !function.contains(needle),
                """
                returnToModelSelection() references `\(needle)`. Recovering from \
                a failure must not open Settings, open a second window, or \
                dismiss setup.
                """
            )
        }
    }

    // MARK: - The destination is in-window

    @Test("browseAllModels() enters the Step 2 catalogue micro-stage")
    func browseAllModelsEntersTheInWindowCatalogue() throws {
        let source = try Self.quickstartSource
        let requiredInOrder = [
            "private func browseAllModels() {",
            "coordinator.returnToChooser()",
            "coordinator.beginBrowsingCatalog()",
        ]
        var cursor = source.startIndex
        for needle in requiredInOrder {
            guard let range = source.range(of: needle, range: cursor..<source.endIndex) else {
                Issue.record(
                    """
                    browseAllModels() is missing `\(needle)`. Browsing must move \
                    the user to the in-window catalogue, and must clear a failed \
                    phase first so the failure card's own link works too.
                    """
                )
                return
            }
            cursor = range.upperBound
        }
    }

    /// The heart of 05.2.J · S1. Each of these is a way to leave setup or to
    /// split it across surfaces, and none may appear on the browse path.
    ///
    /// Note what is NOT forbidden: `openWindow(id: "settings")` still exists in
    /// this file, for the failure card's "Open model management" diagnosis
    /// action. That is a deliberate deep-link out of a dead end, not a browse
    /// route, and it is out of this change's scope. The assertions below are
    /// therefore scoped to the browse function itself, plus a file-wide ban on
    /// the one call that only ever existed to serve browsing.
    @Test("Browsing opens no Settings window, no second window and no sheet")
    func browsingNeverLeavesTheSetupCanvas() throws {
        let source = try Self.quickstartSource

        #expect(
            !source.contains("settingsRouter.beginQuickstartCatalogRoundTrip()"),
            """
            The Quickstart→Settings catalogue round trip is back. Browse all \
            models is a micro-stage inside Step 2 (Paper 05.2.J · S1) — it \
            must not hand the catalogue to a second window.
            """
        )

        guard let start = source.range(of: "private func browseAllModels() {"),
              let end = source.range(of: "\n    }", range: start.upperBound..<source.endIndex)
        else {
            Issue.record("could not isolate browseAllModels()")
            return
        }
        let function = String(source[start.lowerBound..<end.upperBound])
        for needle in ["openWindow", "dismiss()", "settingsRouter", "onSkip", ".sheet"] {
            #expect(
                !function.contains(needle),
                """
                browseAllModels() references `\(needle)`. Browsing must not open \
                Settings, open a second window, end this sheet's modal session, \
                or dismiss setup.
                """
            )
        }
    }

    @Test("Browsing does not dismiss onboarding, and does not reset the step")
    func browsingKeepsSetupOnScreenAtStepTwo() {
        let coord = Self.coordinator()
        coord.advanceToChooseModel()
        coord.resolveRecommendationLoading(catalogLoaded: true)
        #expect(coord.step == .chooseModel)

        coord.beginBrowsingCatalog()

        #expect(coord.step2Stage == .browsing)
        #expect(coord.step == .chooseModel, "the catalogue is not a step of its own")
        #expect(coord.stage == .chooseModel, "browsing must not reset the public step")
        #expect(coord.phase == .idle, "browsing must not start or dismiss anything")
        #expect(!coord.done, "browsing must never complete onboarding")
        coord._testingReset()
    }

    @Test("Browsing carries the selection in, and Back carries it out")
    func browsingPreservesTheSelection() {
        let coord = Self.coordinator()
        let picked = QuickstartCoordinator.onboardingChoices[2]
        coord.advanceToChooseModel()
        coord.select(picked)

        coord.beginBrowsingCatalog()
        #expect(coord.selection.alias == picked.alias, "browsing must not discard the pick (#1653)")

        coord.backToRecommendedModels()
        #expect(coord.selection.alias == picked.alias)
        #expect(coord.step2Stage == .choosing)
        coord._testingReset()
    }

    // MARK: - Dismissal is still reachable from exactly one control

    @Test("Dismissing the wizard is reachable from exactly one control")
    func onlySkipDismissesTheWizard() throws {
        let source = try Self.quickstartSource
        let dismissCalls = source.components(separatedBy: "onSkip()").count - 1
        #expect(
            dismissCalls == 1,
            """
            onSkip() is the app's one genuine "leave onboarding" action and is \
            called \(dismissCalls) time(s). "Browse all models" shared it \
            until #1653 on the theory that both mean "let me look around \
            first" — they differ in exactly the thing that matters, whether \
            the user gets to choose.
            """
        )
        #expect(
            Self.stripped(source).contains(#"Button("Skipfornow"){onSkip()}"#),
            "The one onSkip() call must be Skip for now."
        )
    }

    /// Skip lives on the welcome hero only, so no amount of navigating inside
    /// Step 2 can reach it. Stated as a test because the alternative — a Skip
    /// rendered in the shared Step 2 footer — would look reasonable in review
    /// and would quietly hand Escape a way out of the catalogue.
    @Test("No Step 2 micro-stage offers a dismiss control")
    func stepTwoOffersNoWayOutOfSetup() throws {
        let source = try Self.quickstartSource
        guard let heroRange = source.range(of: "private var welcomeStep: some View"),
              let stepTwoRange = source.range(of: "// MARK: - Step 2 · Choose a model")
        else {
            Issue.record("could not locate the hero and Step 2 sections")
            return
        }
        #expect(!source[stepTwoRange.lowerBound...].contains("onSkip()"),
                "Step 2 must not carry a dismiss control — Escape inside it may only go back")
        #expect(source[heroRange.lowerBound...].contains("onSkip()"),
                "the hero must keep the one genuine dismiss control")
    }

    /// The routing contract Settings deep-links still depend on.
    ///
    /// `SettingsRouter.route(to:open:)` takes the open as a closure so the tab
    /// cannot be staged *after* the window opens — `SettingsView` reads the
    /// router from `.onAppear`, so the wrong order lands the user on their
    /// last-used tab. Browse all models no longer uses it, but the failure
    /// card's "Open model management" diagnosis action still does, and that is
    /// raised to a first-run user for exactly the same reason.
    @Test("The tab is staged before the window opens, never after")
    func routerStagesTheTabBeforeOpening() {
        let router = SettingsRouter()
        var categoryWhenOpened: SettingsView.Category??
        router.route(to: .modelManagement) {
            categoryWhenOpened = router.requestedCategory
        }
        #expect(categoryWhenOpened == .modelManagement)
    }
}
