import Foundation
import Observation

/// Deep-link channel into the Settings window.
///
/// This app has **no SwiftUI `Settings` scene**. It declares a real
/// ``Window("Settings", id: "settings")`` (see ``RapidApp``) so the
/// menu-bar tray item can open it, and ⌘, is re-wired to the same
/// window. The consequence every call site has to respect:
/// ``@Environment(\.openSettings)`` — `OpenSettingsAction` — targets a
/// scene that does not exist here and is a **silent no-op**. The only
/// working path is ``@Environment(\.openWindow)`` with
/// `openWindow(id: "settings")`.
///
/// `openWindow` also takes no category, so it can only say "open
/// Settings," not "open Settings on the Tools panel." Users
/// (2026-06-10) called this out: clicking the bottom-bar status pill
/// landed them on the default tab and they had to hunt for the item
/// they came for.
///
/// ``SettingsRouter`` is a tiny ``@Observable`` shared via the SwiftUI
/// environment that closes that gap. A call site sets
/// ``requestedCategory`` to the desired tab and then calls
/// `openWindow(id: "settings")`; ``SettingsView`` observes the field
/// via ``.onAppear`` (covers the "first open this session" case) and
/// ``.onChange`` (covers the "already-open Settings gets re-focused"
/// case), applies the request, and clears it back to nil so a
/// subsequent open without a request lands on the user's last
/// selected tab.
///
/// **Order is load-bearing**: assign ``requestedCategory`` BEFORE
/// opening the window. `SettingsView` reads it from `.onAppear`, so an
/// assignment made afterwards races that read and drops the user on
/// the last-used tab.
///
/// Why an ``@Observable`` instead of a stored property on
/// ``SettingsView``: SettingsView's ``@State`` lifetime is tied to
/// the Settings scene, which can be created/destroyed independently
/// of the main window. A router living on ``RapidApp`` survives both
/// and lets ``ContentView`` (or any other surface) hand off a target
/// before the Settings scene even exists.
@MainActor
@Observable
final class SettingsRouter {
    /// Pending deep-link target. Set by call sites just before
    /// `openWindow(id: "settings")`; consumed and cleared by
    /// ``SettingsView`` on appear / on change. Nil means "no override —
    /// land on the user's last selected tab."
    var requestedCategory: SettingsView.Category?

    /// One-shot handoff used by Quickstart's Browse all round trip. The
    /// onboarding sheet must be lowered before opening a separate Settings
    /// window, otherwise AppKit's modal session traps that window.
    private(set) var quickstartReturnGeneration = 0
    private(set) var quickstartCatalogReturnPending = false

    func beginQuickstartCatalogRoundTrip() {
        quickstartCatalogReturnPending = true
    }

    func completeQuickstartCatalogRoundTrip() {
        guard quickstartCatalogReturnPending else { return }
        quickstartCatalogReturnPending = false
        quickstartReturnGeneration &+= 1
    }

    /// Which Settings tab a failure-recovery action deep-links to, or `nil`
    /// when the action is carried out in place (retry, restart, switch
    /// download source) and must NOT open Settings at all.
    ///
    /// Pure, static, and exhaustive so the routing DECISION — not merely the
    /// fact that some function ran — is pinned by the test suite. A SwiftUI
    /// body is not reachable from the suite, which is exactly how three of
    /// these buttons shipped wired to a no-op ``OpenSettingsAction``: nothing
    /// could observe where they were supposed to land.
    ///
    /// No `default` clause, deliberately: a newly added
    /// ``FailureDiagnosis/Action`` has to state its destination here (or state
    /// that it has none) rather than inheriting "nowhere" by omission.
    nonisolated static func settingsCategory(
        for action: FailureDiagnosis.Action
    ) -> SettingsView.Category? {
        switch action {
        case .openModelManagement:
            // Settings → Model Management is the cache inspector: what is on
            // disk, what to delete, what to download. That is the panel an
            // out-of-memory or failed-load message is telling the user to go
            // act in.
            return .modelManagement
        case .openWebSearchSettings:
            // Settings → Tools owns the web-search backend picker and its key.
            return .tools
        case .retry, .restart, .switchDownloadSource:
            // Handled by the view that owns the failed operation. Opening
            // Settings for these would be a non-sequitur.
            return nil
        }
    }

    /// Deep-link to the tab ``action`` belongs on, running ``open`` — the
    /// caller's `openWindow(id: "settings")` — once the target is staged.
    /// Returns whether the window was opened; `false` means the action is
    /// handled in place and Settings must stay shut.
    ///
    /// The closure is the point. A `prepare()`-then-`openWindow()` pair leaves
    /// the ordering rule in the caller's hands, where it can be written the
    /// wrong way round and produce a silent bug (the user lands on the
    /// last-used tab instead of the one the button named). Taking the open as a
    /// closure means the assignment cannot be sequenced after it.
    @discardableResult
    func route(_ action: FailureDiagnosis.Action, open: () -> Void) -> Bool {
        guard let category = Self.settingsCategory(for: action) else { return false }
        route(to: category, open: open)
        return true
    }

    /// The same ordering guarantee for call sites that name their tab directly
    /// rather than deriving it from a failure (the version pill → Settings →
    /// App). Pass `nil` for "just open Settings, leave the tab alone."
    func route(to category: SettingsView.Category?, open: () -> Void) {
        requestedCategory = category
        open()
    }
}
