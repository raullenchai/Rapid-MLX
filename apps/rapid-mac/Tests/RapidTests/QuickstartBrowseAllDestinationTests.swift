import Foundation
import Testing

@testable import Rapid

/// "Browse all models →" must have a destination (#1653).
///
/// The bug this pins was not a wrong destination — it was no destination. The
/// whole implementation was one line that set a session dismiss flag, so
/// clicking the link closed the wizard, discarded the model the user had
/// chosen, and dropped them onto the chat surface pinned to whatever the
/// alphabetical fallback picked (a 7.6 GB download nobody asked for).
///
/// It survived every check we had. The control was present, enabled, correctly
/// labelled and carried `Quickstart.BrowseAll`, so the AX structural baselines
/// recorded a perfectly healthy button. A tree dump cannot tell a working
/// button from a decorative one; only pressing it can. So this suite pins the
/// wiring in source, and `gui-golden-flows.sh browse-all-destination` presses
/// the real control and asserts where the user lands.
///
/// Source-level rather than a rendered-view assertion because the failure is a
/// missing call, and the value of catching it is highest at the point where
/// somebody edits that call site — SwiftUI gives no seam to observe "this
/// button's action did nothing".
@MainActor
@Suite("Quickstart — Browse all models opens the catalogue")
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

    @Test("The link's action calls browseAllModels(), not a dismiss flag")
    func browseAllIsWiredToTheCatalogue() throws {
        let source = try Self.quickstartSource
        #expect(
            source.contains(
                """
                                    Button {
                                        browseAllModels()
                                    } label: {
                                        Text("Browse all models →")
                """
            ),
            """
            The "Browse all models →" button no longer calls browseAllModels(). \
            If it is back to a dismiss closure, that is #1653 returning: the \
            wizard vanishes and the user's selection is discarded.
            """
        )
    }

    @Test("The failure card's 'or browse all models' goes to the same place")
    func failureCardLinkIsWiredToo() throws {
        let source = try Self.quickstartSource
        #expect(
            source.contains(
                """
                        Button {
                            browseAllModels()
                        } label: {
                            Text("or browse all models →")
                """
            ),
            """
            The failure card's browse link must reach the catalogue as well — \
            it is offered precisely when the user's chosen model failed, which \
            is the moment they most need to pick a different one.
            """
        )
    }

    @Test("browseAllModels() ends the sheet, then stages the models tab and opens Settings")
    func browseAllModelsRoutesToModelManagement() throws {
        let source = try Self.quickstartSource
        let requiredInOrder = [
            "settingsRouter.beginQuickstartCatalogRoundTrip()",
            "onBrowseAll()",
            "dismiss()",
            "settingsRouter.route(to: .modelManagement)",
            "openWindow(id: \"settings\")",
        ]
        var cursor = source.startIndex
        for needle in requiredInOrder {
            guard let range = source.range(of: needle, range: cursor..<source.endIndex) else {
                Issue.record(
                    """
                    browseAllModels() is missing `\(needle)`. The flow must lower \
                    Quickstart's modal sheet, preserve the selection through the \
                    one-shot router handoff, and open Model Management rather than \
                    the user's last-used Settings tab.
                    """
                )
                return
            }
            cursor = range.upperBound
        }
    }

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
            source.contains(
                """
                            Button("Skip for now") {
                                onSkip()
                            }
                """
            ),
            "The one onSkip() call must be Skip for now."
        )
    }

    /// The routing contract the button depends on, exercised rather than read.
    ///
    /// `SettingsRouter.route(to:open:)` takes the open as a closure so the tab
    /// cannot be staged *after* the window opens — `SettingsView` reads the
    /// router from `.onAppear`, so the wrong order lands the user on their
    /// last-used tab. Pinned here because "Browse all models" is a first-run
    /// surface: the person clicking it has never seen Settings before and has
    /// no last-used tab worth landing on.
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
