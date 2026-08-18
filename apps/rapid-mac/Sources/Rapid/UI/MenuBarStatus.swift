import Foundation

/// Pure, testable model backing the app's single menu-bar (tray)
/// surface.
///
/// Issue #502: SwiftUI's ``MenuBarExtra`` glyph does not render on
/// macOS 26 (Darwin 25.x / Tahoe), so the tray icon — the primary
/// affordance for reopen / new chat / model status / settings / quit —
/// silently vanished for those users. The tray is now an AppKit
/// ``NSStatusItem`` (see ``MenuBarController``), which renders reliably
/// on every macOS version.
///
/// The rendered ``NSMenu`` is not AX-introspectable, so the strongest
/// available regression guard is to keep the tray's dynamic content —
/// the status line, the ordered menu items, and their conditional
/// branches — in these pure helpers and pin them with unit tests.
/// ``MenuBarController`` renders this description into AppKit and owns
/// nothing the tests can't reach here.
enum MenuBarStatus {

    /// One-line "<model> · <state>" or just "<state>" when no model is
    /// resolved yet. Rendered as a disabled informational row in the
    /// tray menu; the user reads it, doesn't act on it.
    static func statusLine(state: ServerState) -> String {
        switch state {
        case .idle:
            return "Idle"
        case .stopped:
            return "Idle"
        case .missing:
            return "Setup needed"
        case .starting(let alias):
            return alias.isEmpty ? "Starting…" : "\(alias) · Starting…"
        case .ready(let alias):
            return alias.isEmpty ? "Ready" : "\(alias) · Ready"
        case .crashed(let alias, _):
            return alias.isEmpty ? "Crashed" : "\(alias) · Crashed"
        }
    }

    // NOTE: a ``glyphIsTemplate(hasUpdate:)`` helper used to live here,
    // encoding "the tray glyph goes non-template amber when an update is
    // waiting". The tray mark is now unconditionally a template (see
    // ``MenuBarController/trayGlyph``) so macOS owns its colour, and the
    // update signal is carried by the "Update available — vX.Y.Z" row in
    // ``menuItems`` below. The helper had no remaining branch to
    // describe, so it was removed rather than left returning a constant.

    // MARK: - Menu model

    /// Which action a tappable tray menu item performs. Stored as the
    /// ``NSMenuItem.tag`` so ``MenuBarController`` can dispatch every
    /// item through a single ``@objc`` handler without one selector per
    /// row. Raw ``Int`` values are an ABI detail of that tag bridge and
    /// carry no other meaning.
    enum MenuBarAction: Int, Equatable {
        case open
        case newChat
        case update
        case checkForUpdates
        case about
        case settings
        case quit
    }

    /// A keyboard chord attached to a tray menu item. AppKit-free so
    /// the menu model stays pure; ``MenuBarController`` maps ``modifiers``
    /// onto ``NSEvent.ModifierFlags`` at render time.
    enum MenuModifier: Equatable {
        case command
        case option
    }

    struct MenuShortcut: Equatable {
        let key: Character
        let modifiers: [MenuModifier]
    }

    /// One row of the tray menu, in display order.
    enum MenuBarItem: Equatable {
        /// A tappable command.
        case button(MenuBarAction, title: String, enabled: Bool, shortcut: MenuShortcut?)
        /// A non-interactive, disabled informational line.
        case status(String)
        /// A separator rule.
        case separator
    }

    /// The full, ordered tray menu given the live inputs. This is the
    /// single source of truth for the menu's structure and its dynamic
    /// branches (the update row only appears when one is available;
    /// "Check for updates…" is disabled mid-check).
    /// ``MenuBarController.rebuildMenu`` renders exactly this list, so a
    /// test against it pins the real menu.
    static func menuItems(
        state: ServerState,
        hasUpdate: Bool,
        updateVersion: String,
        checking: Bool
    ) -> [MenuBarItem] {
        var items: [MenuBarItem] = [
            .button(.open, title: "Open Rapid-MLX", enabled: true, shortcut: nil),
            .button(
                .newChat,
                title: "New Chat",
                enabled: true,
                shortcut: MenuShortcut(key: "n", modifiers: [.command])
            ),
            .separator,
            .status(statusLine(state: state)),
            .separator,
        ]

        if hasUpdate {
            // Newer version visible — surface it as the primary
            // call-to-action above the line. Selecting it hands off to
            // Sparkle's own update panel, which renders its own progress.
            items.append(
                .button(
                    .update,
                    title: "Update available — v\(updateVersion)",
                    enabled: true,
                    shortcut: nil
                )
            )
            items.append(.separator)
        }

        items.append(
            .button(.checkForUpdates, title: "Check for updates…", enabled: !checking, shortcut: nil)
        )
        items.append(.separator)
        items.append(.button(.about, title: "About Rapid-MLX…", enabled: true, shortcut: nil))
        items.append(.separator)
        items.append(
            .button(
                .settings,
                title: "Settings…",
                enabled: true,
                shortcut: MenuShortcut(key: ",", modifiers: [.command])
            )
        )
        items.append(
            .button(
                .quit,
                title: "Quit",
                enabled: true,
                shortcut: MenuShortcut(key: "q", modifiers: [.command])
            )
        )
        return items
    }
}
