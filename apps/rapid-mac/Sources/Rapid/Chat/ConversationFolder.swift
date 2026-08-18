import Foundation

/// A user-created folder for filing conversations.
///
/// Single level, single membership: a conversation carries at most one
/// ``ChatConversation/folderID``. Nesting and multi-membership were both
/// considered and rejected — the rail is 200pt wide, so indentation runs out
/// almost immediately, and a row that appears under two folders makes the
/// selected-row highlight and the delete affordance ambiguous.
struct ChatFolder: Identifiable, Codable, Equatable {
    let id: UUID
    var name: String
    let createdAt: Date

    init(id: UUID = UUID(), name: String, createdAt: Date = Date()) {
        self.id = id
        self.name = name
        self.createdAt = createdAt
    }

    /// Hand-written for the same reason ``ChatConversation``'s is: a folder
    /// file written by a future build that adds a field must still decode
    /// here, because ``ConversationFolderStore/load(from:)`` turns one throw
    /// into "the whole folder list is corrupt".
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(UUID.self, forKey: .id)
        name = try c.decode(String.self, forKey: .name)
        createdAt = try c.decodeIfPresent(Date.self, forKey: .createdAt) ?? Date()
    }

    /// Folder ordering for the sidebar: case-insensitive by name.
    ///
    /// Creation order was the other candidate and is worse to live with — you
    /// look for a folder by its name, and after a rename a creation-ordered
    /// list puts it somewhere with no relationship to what it now says.
    static func displayOrder(_ folders: [ChatFolder]) -> [ChatFolder] {
        folders.sorted {
            let byName = $0.name.localizedCaseInsensitiveCompare($1.name)
            if byName != .orderedSame { return byName == .orderedAscending }
            // Total order: two folders may legitimately share a name.
            return $0.id.uuidString < $1.id.uuidString
        }
    }

    /// Stable suffix for this folder's accessibility identifiers.
    ///
    /// Derived from the NAME, not the id: ``gui-golden-flows.sh`` reaches
    /// controls by `AXIdentifier` alone, and a flow that creates "Work" can
    /// predict `Sidebar.Folder.Toggle.Work` but never a random UUID. Same
    /// reasoning as ``SidebarView/pinMenuItemIdentifier(for:)`` deriving its
    /// identifier from state the test already knows.
    var axSlug: String { ChatFolder.axSlug(for: name) }

    static func axSlug(for name: String) -> String {
        let mapped = name.unicodeScalars.map { scalar -> Character in
            CharacterSet.alphanumerics.contains(scalar) ? Character(scalar) : "-"
        }
        let collapsed = String(mapped)
            .split(separator: "-", omittingEmptySubsequences: true)
            .joined(separator: "-")
        return collapsed.isEmpty ? "Folder" : collapsed
    }

    /// Trim and reject blank folder names. Returns nil when the input can't
    /// make a usable name, so callers can simply not commit.
    static func normalizedName(_ raw: String) -> String? {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }
}

/// On-disk store for the folder list.
///
/// A SEPARATE `folders.json` beside `conversations.json` rather than a new key
/// inside it: the conversation file is a bare `[ChatConversation]` array, so
/// wrapping it in an envelope would make every already-shipped build read its
/// own history as corrupt and side it to `.corrupt-*.json`. A second file
/// costs one more read at launch and needs no migration at all.
///
/// Mirrors ``ConversationStore`` exactly — atomic writes, owner-only
/// permissions, a serial write queue, and corrupt-file side-loading — because
/// the two files are written in the same transactions and any divergence in
/// durability between them shows up as a folder list that disagrees with the
/// conversations filed into it.
enum ConversationFolderStore {
    static func fileURL(override: URL? = nil) -> URL {
        if let override { return override }
        let base = ApplicationSupportLocator.applicationSupportBase()
        let dir = base.appendingPathComponent(
            Bundle.main.bundleIdentifier ?? "Rapid",
            isDirectory: true
        )
        try? FileManager.default.createDirectory(
            at: dir, withIntermediateDirectories: true
        )
        return dir.appendingPathComponent("folders.json")
    }

    static func load(from override: URL? = nil) -> [ChatFolder] {
        let url = fileURL(override: override)
        let fm = FileManager.default
        // Missing file is the normal case for every install that predates
        // folders, and for anyone who has never made one → no folders.
        guard fm.fileExists(atPath: url.path) else { return [] }
        guard let data = try? Data(contentsOf: url),
              let decoded = try? JSONDecoder().decode([ChatFolder].self, from: data)
        else {
            let backup = url.deletingLastPathComponent().appendingPathComponent(
                "folders.corrupt-\(UUID().uuidString).json"
            )
            try? fm.moveItem(at: url, to: backup)
            return []
        }
        return decoded
    }

    private static let writeQueue = DispatchQueue(
        label: "com.rapidmlx.rapid.conversation-folder-write",
        qos: .utility
    )

    static func save(_ folders: [ChatFolder], to override: URL? = nil) {
        writeQueue.async {
            guard let data = try? JSONEncoder().encode(folders) else { return }
            let url = fileURL(override: override)
            try? FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            // 0600 for the same reason the transcripts are: folder names are
            // user content ("Divorce", "Job hunt") and leak intent on their own.
            try? data.write(to: url, options: .atomic)
            try? FileManager.default.setAttributes(
                [.posixPermissions: 0o600], ofItemAtPath: url.path
            )
        }
    }

    /// Block until queued writes commit. Called from app termination alongside
    /// ``ConversationStore/flush()``.
    static func flush() {
        writeQueue.sync {}
    }

    /// The folder file that belongs beside a given conversation store.
    ///
    /// Tests inject an isolated `conversations.json` under a temp directory;
    /// deriving the folder path from it keeps both files in that same
    /// directory without every test having to name a second URL.
    static func companionURL(forConversationStore url: URL?) -> URL? {
        url?.deletingLastPathComponent().appendingPathComponent("folders.json")
    }
}
