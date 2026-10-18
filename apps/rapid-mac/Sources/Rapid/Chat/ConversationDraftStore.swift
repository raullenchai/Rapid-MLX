import Foundation

/// Text only: pending attachments deliberately remain in their in-memory store.
struct ConversationDrafts: Codable, Equatable, Sendable {
    static let newChatKey = "new-chat"
    private var entries: [String: String] = [:]

    func text(for key: String) -> String { entries[key] ?? "" }

    mutating func set(_ text: String, for key: String) {
        entries[key] = text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? nil : text
    }

    mutating func remove(_ id: UUID) {
        entries.removeValue(forKey: id.uuidString)
    }

    func pruned(keeping ids: Set<UUID>) -> Self {
        var result = Self()
        for key in ids.map(\.uuidString) + [Self.newChatKey] {
            result.set(text(for: key), for: key)
        }
        return result
    }
}

/// Companion to ConversationStore, with the same durability and privacy contract.
enum ConversationDraftStore {
    static func fileURL(override: URL? = nil) -> URL {
        if let override { return override }
        // Reuse the transcript locator so both defaults always share a directory.
        return ConversationStore.fileURL().deletingLastPathComponent()
            .appendingPathComponent("drafts.json")
    }

    static func load(from override: URL? = nil) -> ConversationDrafts {
        let url = fileURL(override: override)
        let fm = FileManager.default
        guard fm.fileExists(atPath: url.path) else { return ConversationDrafts() }
        guard let data = try? Data(contentsOf: url),
              let decoded = try? JSONDecoder().decode(ConversationDrafts.self, from: data)
        else {
            let backup = url.deletingLastPathComponent().appendingPathComponent(
                "drafts.corrupt-\(UUID().uuidString).json"
            )
            try? fm.moveItem(at: url, to: backup)
            return ConversationDrafts()
        }
        return decoded
    }

    private static let writeQueue = DispatchQueue(
        label: "com.rapidmlx.rapid.conversation-draft-write",
        qos: .utility
    )

    static func save(_ drafts: ConversationDrafts, to override: URL? = nil) {
        writeQueue.async {
            guard let data = try? JSONEncoder().encode(drafts) else { return }
            let url = fileURL(override: override)
            try? FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            try? data.write(to: url, options: .atomic)
            try? FileManager.default.setAttributes(
                [.posixPermissions: 0o600], ofItemAtPath: url.path
            )
        }
    }

    static func flush() {
        writeQueue.sync {}
    }

    static func companionURL(forConversationStore url: URL?) -> URL? {
        url?.deletingLastPathComponent().appendingPathComponent("drafts.json")
    }
}
