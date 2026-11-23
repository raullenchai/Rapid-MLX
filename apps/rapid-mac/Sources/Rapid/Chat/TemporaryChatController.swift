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
        guard let snapshot = session?.exportSnapshot() else { return }
        let action = afterExit
        afterExit = nil
        ConversationExportPanel.export(snapshot, format: format) { [weak self] saved in
            guard saved else { return }
            self?.discard()
            action?()
        }
    }

    private func discard() {
        session?.close()
        session = nil
    }
}
