import AppKit

/// The AppKit half of conversation export: put a save panel up, write what
/// ``ConversationExport`` rendered, and say so when it fails.
///
/// Structure mirrors ``DiagnosticsBundle/exportViaSavePanel(server:)`` —
/// success reveals the file in Finder, a write failure raises an alert rather
/// than failing silently, and cancelling is a clean no-op.
@MainActor
enum ConversationExportPanel {
    /// Export one conversation. `completion` reports whether the file was written.
    static func export(
        _ conversation: ChatConversation,
        format: ConversationExport.Format,
        now: Date = Date(),
        completion: ((Bool) -> Void)? = nil
    ) {
        let panel = NSSavePanel()
        panel.title = "Export Conversation"
        panel.nameFieldStringValue = ConversationExport.defaultFilename(
            for: conversation,
            format: format,
            at: now
        )
        panel.allowedContentTypes = [format.contentType]
        panel.isExtensionHidden = false
        panel.canCreateDirectories = true
        panel.begin { response in
            guard response == .OK, let url = panel.url else {
                completion?(false)
                return
            }
            do {
                let data: Data
                switch format {
                case .markdown:
                    data = Data(ConversationExport.markdown(conversation).utf8)
                case .json:
                    data = try ConversationExport.json(conversation)
                }
                try data.write(to: url, options: .atomic)
                NSWorkspace.shared.activateFileViewerSelecting([url])
                completion?(true)
            } catch {
                presentFailure(
                    "Couldn't export the conversation",
                    errorText: error.localizedDescription
                )
                completion?(false)
            }
        }
    }

    /// Export the whole library — transcripts plus the folder list — as one
    /// JSON file. This is the backup / escape hatch, so it is deliberately
    /// lossless and deliberately a single file rather than a directory of
    /// per-chat exports the user then has to keep together.
    static func exportAll(
        conversations: [ChatConversation],
        folders: [ChatFolder],
        now: Date = Date()
    ) {
        let panel = NSSavePanel()
        panel.title = "Export All Chats"
        panel.nameFieldStringValue = ConversationExport.defaultArchiveFilename(at: now)
        panel.allowedContentTypes = [.json]
        panel.isExtensionHidden = false
        panel.canCreateDirectories = true
        panel.begin { response in
            guard response == .OK, let url = panel.url else { return }
            do {
                let data = try ConversationExport.allChats(
                    conversations: conversations,
                    folders: folders,
                    exportedAt: now
                )
                try data.write(to: url, options: .atomic)
                NSWorkspace.shared.activateFileViewerSelecting([url])
            } catch {
                presentFailure(
                    "Couldn't export your chats",
                    errorText: error.localizedDescription
                )
            }
        }
    }

    private static func presentFailure(_ title: String, errorText: String) {
        let alert = NSAlert()
        alert.messageText = title
        alert.informativeText = errorText
        alert.alertStyle = .warning
        alert.runModal()
    }
}
