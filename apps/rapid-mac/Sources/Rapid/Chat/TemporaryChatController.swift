import AppKit
import Foundation
import Observation

/// Owns the chat view's temporary mode. Turning it on swaps the visible chat
/// for a view model that persists nothing; turning it off deletes that chat,
/// after offering an export if the user has actually used it.
@MainActor
@Observable
final class TemporaryChatController {
    private let server: ServerManager?
    private let sampling: SamplingConfig?
    private let storageRoot: URL
    private let makeTools: (DocumentContentCache, BrowseContentCache) -> any ToolRegistry

    private(set) var session: TemporaryChatSession?
    var isConfirmingExit = false
    private var afterExit: (() -> Void)?

    init(
        server: ServerManager?,
        sampling: SamplingConfig?,
        storageRoot: URL = TemporaryChatStorage.root(),
        makeTools: @escaping (DocumentContentCache, BrowseContentCache) -> any ToolRegistry
    ) {
        self.server = server
        self.sampling = sampling
        self.storageRoot = storageRoot
        self.makeTools = makeTools
    }

    var isActive: Bool { session != nil }

    /// Whether leaving would lose anything.
    var hasInteraction: Bool { session.map { !$0.chat.messages.isEmpty } ?? false }

    func toggle() {
        if isActive {
            leave(then: {})
        } else {
            session = TemporaryChatSession(
                storageRoot: storageRoot,
                server: server,
                sampling: sampling,
                makeTools: makeTools
            )
        }
    }

    /// Run `action` once temporary mode is off. Warns first when there is a
    /// conversation to lose; the action is dropped if the user backs out.
    func leave(then action: @escaping () -> Void) {
        guard isActive else {
            action()
            return
        }
        guard hasInteraction else {
            discard()
            action()
            return
        }
        afterExit = action
        isConfirmingExit = true
    }

    func confirmDiscard() {
        isConfirmingExit = false
        let action = afterExit
        afterExit = nil
        discard()
        action?()
    }

    func cancelExit() {
        isConfirmingExit = false
        afterExit = nil
    }

    /// Leaves temporary mode only if the file was written; cancelling the save
    /// panel keeps the chat.
    func export(_ format: ConversationExport.Format) {
        isConfirmingExit = false
        let action = afterExit
        afterExit = nil
        export(format, then: { action?() }, orElse: {})
    }

    func export(
        _ format: ConversationExport.Format,
        then action: @escaping () -> Void,
        orElse onCancel: @escaping () -> Void
    ) {
        guard let snapshot = session?.exportSnapshot() else {
            onCancel()
            return
        }
        ConversationExportPanel.export(snapshot, format: format) { [weak self] saved in
            guard saved else {
                onCancel()
                return
            }
            self?.discard()
            action()
        }
    }

    enum QuitDecision {
        case discard
        case cancel
        case export(ConversationExport.Format)
    }

    /// Quitting skips the in-window warning, so it asks here instead.
    static func askBeforeQuitting() -> QuitDecision {
        let alert = NSAlert()
        alert.messageText = String(localized: "Quit and discard this temporary chat?")
        alert.informativeText = String(localized: "Temporary chats aren't saved. Quitting deletes the conversation. Export it first to keep a copy.")
        alert.addButton(withTitle: String(localized: "Export Markdown…"))
        alert.addButton(withTitle: String(localized: "Export JSON…"))
        alert.addButton(withTitle: String(localized: "Quit and Discard"))
        alert.addButton(withTitle: String(localized: "Cancel"))
        switch alert.runModal() {
        case .alertFirstButtonReturn: return .export(.markdown)
        case .alertSecondButtonReturn: return .export(.json)
        case .alertThirdButtonReturn: return .discard
        default: return .cancel
        }
    }

    /// Quit path: closes the open session, used or not, before sweeping so a
    /// late extraction cannot write the folder back after the sweep.
    func closeForTermination() {
        discard()
        TemporaryChatStorage.removeAll(in: storageRoot)
    }

    private func discard() {
        session?.close()
        session = nil
    }
}
