import AppKit
import Foundation

/// Presents the "Hide application icon from Dock?" ``NSAlert`` on the
/// first main-window close (rapid-desktop issue #260). Lives next to
/// ``DockVisibilityPromptStore`` so the prompt + persistence sit in
/// the same logical surface.
///
/// The alert mirrors Trend Micro Antivirus One's reference UX:
///
/// > **Would you like to hide the application icon from Dock?**
/// > The application will keep running in the background.
/// > [ ] Don't ask me again
/// > [ No ]   [ Yes ]
///
/// Accessibility identifiers (``DockHidePrompt.YesButton``,
/// ``DockHidePrompt.NoButton``, ``DockHidePrompt.DontAskCheckbox``)
/// match the convention from PR #259 so a future XCUITest / AX walker
/// can drive the dialog deterministically.
@MainActor
enum DockVisibilityPrompt {
    /// Result the close-handler consumes. ``hideDockNow`` tells the
    /// caller whether to flip ``NSApp`` to ``.accessory``; the store
    /// has already persisted the choice the user made (including
    /// "Don't ask again") by the time this returns.
    struct Outcome: Equatable {
        let hideDockNow: Bool
    }

    /// Build + present the modal alert and commit the user's answer
    /// back into ``store``. Returns the resolved ``Outcome``.
    /// ``runModal`` is synchronous so the caller blocks on the user's
    /// click — exactly what the ``windowShouldClose`` delegate needs
    /// (it returns ``false`` if hiding, ``true`` if quitting through
    /// SwiftUI's normal close).
    @discardableResult
    static func runModal(store: DockVisibilityPromptStore) -> Outcome {
        let alert = makeAlert()
        let response = alert.runModal()
        // ``NSAlert.runModal`` returns ``.alertFirstButtonReturn``
        // for the leading button. We make the leading button "Yes"
        // (the accept-the-suggestion default) so the keyboard
        // affordance (Return) commits to the hide path.
        let pickedYes = response == .alertFirstButtonReturn
        let dontAskAgain = isDontAskCheckboxOn(alert: alert)
        store.record(userPickedYes: pickedYes, dontAskAgain: dontAskAgain)
        return Outcome(hideDockNow: pickedYes)
    }

    /// Construct (but don't show) the configured ``NSAlert``. Split
    /// out so a future snapshot test can inspect the assembled
    /// dialog without ``runModal``-blocking the suite.
    static func makeAlert() -> NSAlert {
        let alert = NSAlert()
        alert.messageText = String(localized: "Hide application icon from Dock?")
        alert.informativeText = String(localized: "The application will keep running in the background. You can re-open the window from the menu-bar icon at any time.")
        alert.alertStyle = .informational

        // Yes is the leading (Return-defaulted) button so the
        // suggestion is the keyboard-default — matches the reference
        // UX's intent ("we expect most users to want this hidden").
        let yes = alert.addButton(withTitle: String(localized: "Yes"))
        yes.setAccessibilityIdentifier("DockHidePrompt.YesButton")
        let no = alert.addButton(withTitle: String(localized: "No"))
        no.keyEquivalent = "\u{1b}"  // Escape — preserves Cmd-period dismiss.
        no.setAccessibilityIdentifier("DockHidePrompt.NoButton")

        // "Don't ask me again" surfaces below the message via
        // ``accessoryView``. NSAlert's built-in ``showsSuppressionButton``
        // works too, but its copy is "Do not show this message again"
        // — the reference UX says "Don't ask me again", which is
        // less ambiguous about scope (we're not silencing all alerts
        // forever, just this one prompt). Rolling our own checkbox
        // also lets us pin the accessibility identifier explicitly.
        let checkbox = NSButton(checkboxWithTitle: "Don't ask me again", target: nil, action: nil)
        checkbox.state = .off
        checkbox.setAccessibilityIdentifier("DockHidePrompt.DontAskCheckbox")
        checkbox.sizeToFit()
        alert.accessoryView = checkbox

        return alert
    }

    /// Read the "Don't ask me again" checkbox state from an alert we
    /// previously built via ``makeAlert``. Returns ``false`` if the
    /// accessory view is not the expected ``NSButton`` (defensive
    /// against a future refactor that swaps the accessory shape).
    static func isDontAskCheckboxOn(alert: NSAlert) -> Bool {
        guard let checkbox = alert.accessoryView as? NSButton else {
            return false
        }
        return checkbox.state == .on
    }
}
