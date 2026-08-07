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
        configureButton(hasUpdate: Self.hasAvailableUpdate())
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
        // The status line + update row rebuild lazily on click, but the
        // glyph tint is visible WITHOUT opening the menu, so observe the
        // update state and repaint the glyph the moment it flips.
        startObservingUpdateState()
    }

    // MARK: - Tray glyph

    private static func hasAvailableUpdate() -> Bool {
        AppDelegate.shared.updater?.availableUpdate != nil
    }

    private func configureButton(hasUpdate: Bool) {
        guard let button = statusItem.button else { return }
        button.image = Self.trayGlyph(hasUpdate: hasUpdate)
        // Brand name, not "menu bar item" jargon — this is the accessible
        // label VoiceOver reads and the hover tooltip the user sees.
        button.toolTip = "Rapid-MLX"
    }

    /// Render the brand cheetah into a menu-bar image so the tray icon
    /// matches the app icon the user recognises — an Ollama-style
    /// coloured mascot in the tray, the single source of truth being the
    /// same vendored ``cheetah.png`` the in-app brand marks use.
    ///
    /// Issue #502 forced the tray onto ``NSStatusItem``; earlier
    /// iterations shipped a lightning ``bolt.fill`` because a naively
    /// shrunk cheetah read as a muddy blob and a code-drawn mark
    /// rasterised transparent. Two things fix that here:
    ///   * Render from the high-res master (``cheetah.png``, 440×390)
    ///     with ``.high`` interpolation via a resolution-independent
    ///     ``NSImage`` drawing handler, so AppKit redraws crisply at the
    ///     bar's native backing scale instead of upscaling a tiny crop.
    ///   * Keep it COLOURED (``isTemplate = false``). Template rendering
    ///     would flatten the mascot to an unrecognisable silhouette.
    ///
    /// A waiting update is signalled by a small amber dot in the corner
    /// rather than tinting the whole glyph (which would erase the
    /// mascot). If the asset can't be resolved (corrupted .app) a
    /// visible SF Symbol keeps the status item from collapsing to an
    /// invisible slot.
    static func trayGlyph(hasUpdate: Bool) -> NSImage {
        // Menu-bar content height in points; the drawing handler is
        // resolution-independent so AppKit fills it at 2x on Retina.
        let height: CGFloat = 18
        guard let source = CheetahLogo.load(forSize: 128) else {
            let fallback = NSImage(
                systemSymbolName: "hare.fill",
                accessibilityDescription: "Rapid-MLX"
            ) ?? NSImage(size: NSSize(width: height, height: height))
            // Corrupted .app (asset missing): still signal a waiting
            // update by tinting the fallback amber, mirroring the
            // coloured-cheetah path's dot so the cue is never lost.
            if hasUpdate {
                let tinted = NSImage(size: fallback.size)
                tinted.lockFocus()
                NSColor.systemOrange.set()
                let r = NSRect(origin: .zero, size: fallback.size)
                fallback.draw(in: r)
                r.fill(using: .sourceAtop)
                tinted.unlockFocus()
                tinted.isTemplate = false
                return tinted
            }
            fallback.isTemplate = true
            return fallback
        }
        let aspect = source.size.height > 0 ? source.size.width / source.size.height : 1
        let target = NSSize(width: (height * aspect).rounded(), height: height)
        let image = NSImage(size: target, flipped: false) { rect in
            NSGraphicsContext.current?.imageInterpolation = .high
            source.draw(
                in: rect, from: .zero, operation: .sourceOver, fraction: 1.0
            )
            if hasUpdate {
                let d = rect.height * 0.42
                let dot = NSRect(
                    x: rect.maxX - d, y: rect.maxY - d, width: d, height: d
                )
                NSColor.systemOrange.setFill()
                NSBezierPath(ovalIn: dot).fill()
            }
            return true
        }
        image.isTemplate = false
        return image
    }

    /// Re-render the glyph whenever an update becomes available (or is
    /// cleared). ``withObservationTracking``'s ``onChange`` fires exactly
    /// once per observed read, so it re-arms on the next main-actor tick,
    /// mirroring ``QuickAskWindowController.startTracking``.
    private func startObservingUpdateState() {
        withObservationTracking {
            _ = AppDelegate.shared.updater?.availableUpdate
        } onChange: { [weak self] in
            Task { @MainActor [weak self] in
                guard let self else { return }
                self.configureButton(hasUpdate: MenuBarController.hasAvailableUpdate())
                self.startObservingUpdateState()
            }
        }
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
            installerRunning: AppDelegate.shared.installer?.isRunning ?? false,
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
            openUpdateWindow()
        case .checkForUpdates:
            Task { _ = await AppDelegate.shared.updater?.check() }
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

    /// Open the dedicated update window through the ``AppDelegate`` bridge,
    /// guarding the same macOS 14.0–14.2 background-``openWindow`` race as
    /// ``bringMainWindowForward``.
    private func openUpdateWindow() {
        Task { @MainActor in
            NSApp.activate(ignoringOtherApps: true)
            try? await Task.sleep(nanoseconds: 50_000_000)
            AppDelegate.openUpdateWindow?()
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
