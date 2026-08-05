import Foundation

/// A saved conversation — the unit the sidebar "Older" list shows and the
/// on-disk history persists. ``ChatMessage`` is already ``Codable``, so a
/// conversation serialises as-is.
struct ChatConversation: Identifiable, Codable, Equatable {
    let id: UUID
    var title: String
    var messages: [ChatMessage]
    let createdAt: Date
    var updatedAt: Date
}

/// On-disk store for the conversation history. One JSON file under
/// Application Support, keyed by bundle identifier so a dogfood-isolated
/// instance (rewritten bundle id) keeps its own history separate from the
/// real app's. Writes are atomic; a missing / unreadable file reads as an
/// empty history (first run).
enum ConversationStore {
    static func fileURL() -> URL {
        let base = FileManager.default
            .urls(for: .applicationSupportDirectory, in: .userDomainMask)
            .first ?? FileManager.default.temporaryDirectory
        let dir = base.appendingPathComponent(
            Bundle.main.bundleIdentifier ?? "Rapid",
            isDirectory: true
        )
        try? FileManager.default.createDirectory(
            at: dir, withIntermediateDirectories: true
        )
        return dir.appendingPathComponent("conversations.json")
    }

    static func load() -> [ChatConversation] {
        let url = fileURL()
        let fm = FileManager.default
        // A genuinely MISSING file is a normal first run → empty history.
        guard fm.fileExists(atPath: url.path) else { return [] }
        // A file that EXISTS but can't be read (I/O / permissions) or can't
        // be decoded (schema change, corruption) must NOT be silently
        // treated as empty — the next save would atomically overwrite the
        // user's whole history. Side it to `.corrupt-<t>` so it's
        // recoverable AND so the next save writes to a fresh path instead of
        // clobbering the original, then start empty.
        guard let data = try? Data(contentsOf: url),
              let decoded = try? JSONDecoder().decode([ChatConversation].self, from: data)
        else {
            // Unique name (UUID) so the move can't fail on a pre-existing
            // backup → the live file is reliably cleared out of the way
            // before we return empty. A move that still fails here is a
            // same-directory permission/IO fault, in which case the next
            // ``save`` (same dir) also fails — so there's no clobber path.
            let backup = url.deletingLastPathComponent().appendingPathComponent(
                "conversations.corrupt-\(UUID().uuidString).json"
            )
            try? fm.moveItem(at: url, to: backup)
            return []
        }
        // Sort here rather than trusting the file's array order. ``save``
        // writes whatever order the in-memory array happens to be in, which
        // ``ChatViewModel.persistActive`` keeps newest-first only as a side
        // effect of its ``insert(at: 0)`` bubbling. Any path that doesn't go
        // through that bubble — a hand-edited file, a future bulk import, a
        // partially-applied migration — would surface out of order in the
        // sidebar with nothing to correct it. Guaranteeing the invariant at
        // the data boundary costs one sort and removes the whole class.
        return decoded.sorted { $0.updatedAt > $1.updatedAt }
    }

    /// Serial queue for history writes: every save re-encodes the whole
    /// history, so a CONCURRENT queue could let an older snapshot's write
    /// land after a newer one's — overwriting recent turns or resurrecting
    /// a deleted conversation. A serial queue guarantees writes commit in
    /// submission order.
    private static let writeQueue = DispatchQueue(
        label: "com.rapidmlx.rapid.conversation-history-write",
        qos: .utility
    )

    /// Persist off the main actor — the caller (``ChatViewModel``) is
    /// ``@MainActor`` and every save re-encodes the whole history, so
    /// encoding + disk I/O must not run on the main thread or history
    /// growth would stall the UI. The snapshot is passed by value (Codable
    /// value types), so the background write sees a stable copy, and the
    /// serial ``writeQueue`` keeps ordering.
    static func save(_ conversations: [ChatConversation]) {
        writeQueue.async {
            guard let data = try? JSONEncoder().encode(conversations) else { return }
            let url = fileURL()
            // Owner-only (0600): chat transcripts are private. The atomic
            // write would otherwise inherit the umask default (often 0644 =
            // world-readable), exposing history to other local users.
            try? data.write(to: url, options: .atomic)
            try? FileManager.default.setAttributes(
                [.posixPermissions: 0o600], ofItemAtPath: url.path
            )
        }
    }

    /// Block until every queued write has committed. Call from app
    /// termination so the last turn / edit / deletion isn't lost when the
    /// process exits before the async ``save`` lands.
    static func flush() {
        writeQueue.sync {}
    }

    /// Derive a one-line title from the transcript — the first user
    /// message, whitespace-collapsed and length-capped (Ollama shows the
    /// opening message as the row label). Falls back to "New chat".
    static func title(from messages: [ChatMessage]) -> String {
        guard let first = messages.first(where: { $0.role == .user }) else {
            return "New chat"
        }
        let collapsed = first.content
            .components(separatedBy: .whitespacesAndNewlines)
            .filter { !$0.isEmpty }
            .joined(separator: " ")
        if collapsed.isEmpty { return "New chat" }
        return collapsed.count > 42
            ? String(collapsed.prefix(42)) + "…"
            : collapsed
    }
}
