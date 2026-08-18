import AppKit
import SwiftUI

/// The app's single menu-bar (tray) surface, built on AppKit's
/// ``NSStatusItem``.
///
/// Issue #502: SwiftUI's ``MenuBarExtra`` glyph does not render on
/// macOS 26 (Darwin 25.x / Tahoe) — the tray icon, and with it the
/// primary affordance for reopen / new chat / model status / settings /
/// quit, silently vanished for those users (confirmed on 26.5.2,
/// Mac16,12, app 0.8.20). ``NSStatusItem`` renders reliably on every
/// macOS version, so it is the single tray surface across all versions.
///
/// There is deliberately NO ``MenuBarExtra`` scene anywhere in the app.
/// Standing up two tray surfaces at once is exactly the double-icon bug
/// #475 fixed, so this controller is the one — and only — status item:
/// the AppKit tray and the (non-rendering) SwiftUI tray must never both
/// exist. ``MenuBarTests`` pins both halves of that invariant (exactly
/// one ``NSStatusItem`` creation, zero ``MenuBarExtra`` scenes).
///
/// Init wiring lives in ``AppDelegate.applicationDidFinishLaunching`` —
/// the controller reads its dependencies through ``AppDelegate.shared``,
/// which ``RapidApp.init`` populates before AppKit finishes launching,
/// and it must be created after the activation-policy + AX setup so the
/// status-bar slot inherits the correct appearance on the first frame.
@MainActor
final class MenuBarController: NSObject {

    private let statusItem: NSStatusItem
    private let menu = NSMenu()

    override init() {
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        super.init()
        configureButton()
        menu.delegate = self
        // Take full ownership of item enablement. With AppKit's default
        // auto-enabling, any item that has a target/action is forced
        // enabled regardless of ``isEnabled`` — which would silently
        // re-enable "Check for updates…" mid-check. We drive enablement
        // from the pure ``MenuBarStatus.menuItems`` model instead.
        menu.autoenablesItems = false
        statusItem.menu = menu
        // Pre-populate so the very first click before ``menuNeedsUpdate``
        // fires doesn't show an empty rectangle.
        rebuildMenu()
        // No glyph observer: the mark is a state-free template (see
        // ``trayGlyph``), so nothing about it changes while the app runs.
        // Update availability is reported by the menu's own
        // "Update available — vX.Y.Z" row, which ``menuNeedsUpdate``
        // rebuilds on every open.
    }

    // MARK: - Tray glyph

    private static func hasAvailableUpdate() -> Bool {
        AppDelegate.shared.updater?.availableUpdate != nil
    }

    private func configureButton() {
        guard let button = statusItem.button else { return }
        button.image = Self.trayGlyph()
        // Brand name, not "menu bar item" jargon — this is the hover
        // tooltip the user sees.
        button.toolTip = Self.accessibilityTitle
        // ``toolTip`` alone is not a reliable AX title for an
        // ``NSStatusBarButton`` whose label is a pure image, so name the
        // control explicitly. Both the image's description and the
        // button's title are set: VoiceOver reads the button, and the
        // image description is what an AX walker sees if it descends.
        button.setAccessibilityTitle(Self.accessibilityTitle)
        button.image?.accessibilityDescription = Self.accessibilityTitle
    }

    /// Spoken/hover name for the status item. One constant so the
    /// tooltip and the accessibility title can never drift apart.
    static let accessibilityTitle = "Rapid-MLX"

    /// The tray mark: the official Rapid-MLX `R`, as a **template**.
    ///
    /// Replaces the full-colour cheetah, which carried far too much
    /// detail for an 18pt slot — at menu-bar size the mascot resolved to
    /// a muddy amber smudge, and being non-template it ignored the bar's
    /// appearance entirely (a light-mode bar, a dark-mode bar and an
    /// open menu all got the same coloured bitmap).
    ///
    /// ``RapidRMark`` draws the brand's own vector geometry and tags the
    /// result ``isTemplate``, which hands colour back to macOS: black on
    /// a light bar, white on a dark one, white again while the menu is
    /// open or the bar sits on a tinted desktop. See ``RapidRMark`` for
    /// why the menu-bar variant is the `R` alone rather than the full
    /// lockup.
    ///
    /// Deliberately carries NO status: no amber "update waiting" dot, no
    /// Ready/Starting/Failed tint. A template image cannot express those
    /// (AppKit repaints every pixel), and lifecycle state belongs in the
    /// menu — where ``MenuBarStatus/menuItems`` already puts it, as a
    /// status line plus an "Update available — vX.Y.Z" row.
    ///
    /// The SF Symbol fallback covers a corrupted build whose geometry
    /// fails to parse; it is also a template, so the two paths behave
    /// identically as far as the bar is concerned.
    static func trayGlyph() -> NSImage {
        if let mark = RapidRMark.menuBarTemplateImage() {
            return mark
        }
        let fallback = NSImage(
            systemSymbolName: "bolt.fill",
            accessibilityDescription: accessibilityTitle
        ) ?? NSImage(
            size: NSSize(
                width: RapidRMark.menuBarGlyphHeight,
                height: RapidRMark.menuBarGlyphHeight
            )
        )
        fallback.isTemplate = true
        return fallback
    }

    // MARK: - Menu construction

    /// Rebuild the whole menu from the pure ``MenuBarStatus.menuItems``
    /// description. Cheap (a dozen ``NSMenuItem`` allocations) and keeps
    /// the dynamic rows — status line, update call-to-action, the
    /// "Check for updates…" disabled-while-checking state — fresh on
    /// every open without an AppKit-side observer.
    private func rebuildMenu() {
        menu.removeAllItems()
        for item in MenuBarStatus.menuItems(
            state: AppDelegate.shared.server?.state ?? .idle,
            hasUpdate: Self.hasAvailableUpdate(),
            updateVersion: AppDelegate.shared.updater?.availableUpdate?.version ?? "",
            checking: AppDelegate.shared.updater?.checking ?? false
        ) {
            switch item {
            case .separator:
                menu.addItem(.separator())

            case .status(let text):
                // A nil action renders the row as a disabled label.
                let line = NSMenuItem(title: text, action: nil, keyEquivalent: "")
                line.isEnabled = false
                menu.addItem(line)

            case .button(let action, let title, let enabled, let shortcut):
                let key = shortcut.map { String($0.key) } ?? ""
                let menuItem = NSMenuItem(
                    title: title,
                    action: #selector(handleMenuAction(_:)),
                    keyEquivalent: key
                )
                menuItem.target = self
                menuItem.isEnabled = enabled
                menuItem.tag = action.rawValue
                if let shortcut {
                    menuItem.keyEquivalentModifierMask = Self.modifierFlags(shortcut.modifiers)
                }
                menu.addItem(menuItem)
            }
        }
    }

    private static func modifierFlags(_ modifiers: [MenuBarStatus.MenuModifier]) -> NSEvent.ModifierFlags {
        var flags: NSEvent.ModifierFlags = []
        for modifier in modifiers {
            switch modifier {
            case .command:
                flags.insert(.command)
            case .option:
                flags.insert(.option)
            }
        }
        return flags
    }

    // MARK: - Actions

    /// Single dispatch point for every tappable row. The ``tag`` carries
    /// the ``MenuBarStatus.MenuBarAction`` raw value the row was built
    /// with, so there's one selector instead of ten.
    @objc private func handleMenuAction(_ sender: NSMenuItem) {
        guard let action = MenuBarStatus.MenuBarAction(rawValue: sender.tag) else { return }
        switch action {
        case .open:
            bringMainWindowForward()
        case .newChat:
            // Start a fresh conversation and surface the window.
            AppDelegate.shared.chat?.newConversation()
            bringMainWindowForward()
        case .update:
            if let sparkle = AppDelegate.shared.sparkleUpdater, sparkle.isEnabled {
                sparkle.checkForUpdates()
                return
            }
            // Unsigned build: Sparkle is off and there is no in-app installer
            // to fall back to any more. Settings → App names the running
            // version and, in this state, offers a link to the release page,
            // so send the user there rather than to a dead end.
            NSApp.activate(ignoringOtherApps: true)
            AppDelegate.openSettingsWindowAt?(.app)
        case .checkForUpdates:
            if let sparkle = AppDelegate.shared.sparkleUpdater, sparkle.isEnabled {
                sparkle.checkForUpdates()
                return
            }
            // Fire the check AND take the user somewhere that reports it.
            //
            // This used to be `Task { _ = await updater?.check() }` — result
            // discarded, no UI. When an update exists the menu grows an
            // "Update available" item on the next open, so that path looked
            // fine; when you are already current, clicking produced nothing
            // observable at all, which is indistinguishable from a broken
            // build or a click that missed (#1605).
            //
            // Settings → App already renders the outcome, including the
            // running version, so route there rather than inventing a second
            // surface that could drift from the first.
            Task { _ = await AppDelegate.shared.updater?.check() }
            NSApp.activate(ignoringOtherApps: true)
            AppDelegate.openSettingsWindowAt?(.app)
        case .about:
            if let server = AppDelegate.shared.server {
                AboutPanel.show(server: server)
            }
        case .settings:
            NSApp.activate(ignoringOtherApps: true)
            AppDelegate.openSettingsWindow?()
        case .quit:
            // Goes through ``applicationWillTerminate`` so the session
            // store flushes + the rapid-mlx subprocess is reaped, same
            // path as ⌘Q from the dock menu.
            NSApp.terminate(nil)
        }
    }

    // MARK: - Window restoration

    /// Bring the main chat window forward, restoring it if it was
    /// minimised or fully closed. Matches the SwiftUI ``Window(id: "main")``
    /// scene by its identifier — NOT by title / ``canBecomeMain`` — so we
    /// never grab the Update, Conversation pop-out, or Settings window
    /// when the chat window is closed (the same invariant
    /// ``RapidApp.applyWindowOnTop`` pins). Only when there is no such
    /// window on screen (⌘W tore the scene down) do we materialise it
    /// through SwiftUI's ``openWindow`` via the ``AppDelegate`` bridge,
    /// guarding the macOS 14.0–14.2 background-``openWindow`` race with a
    /// one-run-loop-tick yield.
    private func bringMainWindowForward() {
        NSApp.activate(ignoringOtherApps: true)
        if let target = NSApp.windows.first(where: { $0.identifier?.rawValue == "main" }) {
            if target.isMiniaturized {
                target.deminiaturize(nil)
            }
            target.makeKeyAndOrderFront(nil)
            return
        }
        Task { @MainActor in
            try? await Task.sleep(nanoseconds: 50_000_000)
            AppDelegate.openMainWindow?()
        }
    }

    // NOTE: a private `openSettings()` used to sit here, dispatching
    // ``showSettingsWindow:`` / ``showPreferencesWindow:`` through
    // ``NSApp.sendAction``. It was unreferenced — the ``.settings`` case above
    // has gone through ``AppDelegate.openSettingsWindow`` (the
    // ``openWindow(id: "settings")`` bridge) since the tray item was fixed —
    // and it could not have worked if it were called: both selectors are
    // installed by a SwiftUI ``Settings`` scene, which this app does not
    // declare. Deleted rather than left as a plausible-looking helper for the
    // next person to reach for. See ``SettingsRouter``.
}

// MARK: - NSMenuDelegate

extension MenuBarController: NSMenuDelegate {
    /// Rebuild the menu before every display so the dynamic rows are
    /// fresh. Cost is trivial and the alternative — an AppKit-side
    /// observer of the ``@Observable`` server / updater — buys nothing
    /// for content only visible during a click.
    func menuNeedsUpdate(_ menu: NSMenu) {
        rebuildMenu()
    }
}
