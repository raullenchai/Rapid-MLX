import Darwin
import Foundation

/// Text only: pending attachments deliberately remain in their in-memory store.
struct ConversationDrafts: Codable, Equatable, Sendable {
    static let newChatKey = "new-chat"
    private var entries: [String: String] = [:]

    private enum CodingKeys: String, CodingKey { case entries }

    init() {}

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        entries = try container.decode([String: String].self, forKey: .entries)
            .filter { !$0.value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        // Preserve live keystrokes; blank drafts need no durable entry.
        try container.encode(entries.filter {
            !$0.value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        }, forKey: .entries)
    }

    func text(for key: String) -> String { entries[key] ?? "" }

    mutating func set(_ text: String, for key: String) {
        entries[key] = text.isEmpty ? nil : text
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
        let data: Data
        do {
            data = try Data(contentsOf: url)
        } catch {
            NSLog("Failed to load conversation drafts from %@: %@", url.path, String(describing: error))
            return ConversationDrafts()
        }
        guard let decoded = try? JSONDecoder().decode(ConversationDrafts.self, from: data) else {
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
            let url = fileURL(override: override)
            let fm = FileManager.default
            do {
                let data = try JSONEncoder().encode(drafts)
                let directory = url.deletingLastPathComponent()
                try fm.createDirectory(at: directory, withIntermediateDirectories: true)
                let temporary = directory.appendingPathComponent(".drafts-\(UUID().uuidString).tmp")
                defer { try? fm.removeItem(at: temporary) }
                // Set permissions at creation so no published or temporary bytes are exposed.
                guard fm.createFile(
                    atPath: temporary.path, contents: data, attributes: [.posixPermissions: 0o600]
                ) else { throw CocoaError(.fileWriteUnknown) }
                guard rename(temporary.path, url.path) == 0 else {
                    throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
                }
            } catch {
                NSLog("Failed to save conversation drafts to %@: %@", url.path, String(describing: error))
            }
        }
    }

    static func flush() {
        writeQueue.sync {}
    }

    static func companionURL(forConversationStore url: URL?) -> URL? {
        url?.deletingLastPathComponent().appendingPathComponent("drafts.json")
    }
}
