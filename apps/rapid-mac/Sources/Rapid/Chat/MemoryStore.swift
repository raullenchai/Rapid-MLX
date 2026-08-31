import Foundation
import Observation

/// A single durable fact the assistant has learned about the user across
/// conversations. Backed by the Open WebUI memory-review pattern: after a
/// conversation turn completes, a lightweight background pass decides
/// whether anything enduring should be saved.
struct MemoryEntry: Codable, Equatable, Hashable, Identifiable, Sendable {
    let id: UUID
    var content: String
    var evidenceCount: Int
    var sourceConversationIDs: [UUID]
    var createdAt: Date
    var updatedAt: Date

    init(
        id: UUID = UUID(),
        content: String,
        evidenceCount: Int = 1,
        sourceConversationIDs: [UUID] = [],
        createdAt: Date = Date(),
        updatedAt: Date = Date()
    ) {
        self.id = id
        self.content = content
        self.evidenceCount = evidenceCount
        self.sourceConversationIDs = sourceConversationIDs
        self.createdAt = createdAt
        self.updatedAt = updatedAt
    }
}

/// Schema version marker so future migrations can distinguish disk shapes.
struct MemoryLibrary: Codable, Sendable {
    var schemaVersion: Int = 1
    var entries: [MemoryEntry] = []
}

/// Stores and manages persistent memory entries. The store is the single
/// point of truth for what the assistant knows about the user; the
/// extractor proposes entries, the user can edit or delete them in
/// Settings, and the system prompt builder reads a formatted subset for
/// injection.
@MainActor
@Observable
final class MemoryStore {
    /// Maximum entries kept on disk. Older entries beyond this cap are
    /// dropped from the LOWEST ``evidenceCount`` first, then oldest
    /// ``updatedAt``, so frequently-corroborated facts survive pruning.
    static let maximumEntries = 80

    /// Maximum characters across all injected memories. Small models
    /// (4B–9B) have limited context windows; the memory block must stay
    /// well under the room the date context and instructions already use.
    static let maximumInjectedCharacters = 2_000

    /// File layout lives next to ``ConversationStore``'s JSON under
    /// Application Support/Rapid, keeping all user data in one directory.
    private static let storageKey = "memory-library-v1"

    private(set) var entries: [MemoryEntry] = []
    private let fileURL: URL
    private let defaults: UserDefaults

    /// Whether automatic memory extraction is enabled. Persisted so the
    /// user's choice survives relaunches. Defaults to off — memory is an
    /// opt-in feature the user enables in Settings, not something that
    /// silently collects data.
    var isEnabled: Bool {
        get { defaults.bool(forKey: "rapid.memory.enabled") }
        set { defaults.set(newValue, forKey: "rapid.memory.enabled") }
    }

    init(fileURL: URL? = nil, defaults: UserDefaults = .standard) {
        self.defaults = defaults
        if let fileURL {
            self.fileURL = fileURL
        } else {
            self.fileURL = ApplicationSupportLocator.applicationSupportRoot()
                .appendingPathComponent(Self.storageKey + ".json")
        }
        load()
    }

    // MARK: - CRUD

    /// Adds a new memory entry or increments the evidence count on a
    /// semantically identical one. Returns the affected entry.
    @discardableResult
    func upsert(content: String, conversationID: UUID) -> MemoryEntry {
        let trimmed = content.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            // Return a placeholder rather than force-unwrapping; callers
            // check the returned entry's content before using it.
            return MemoryEntry(content: "", createdAt: .distantPast, updatedAt: .distantPast)
        }

        // Case-insensitive substring match for dedup. A future NLP pass
        // can improve this, but substring catches the common case of the
        // same preference restated verbatim.
        if let existingIndex = entries.firstIndex(where: {
            $0.content.localizedCaseInsensitiveContains(trimmed)
                || trimmed.localizedCaseInsensitiveContains($0.content)
        }) {
            entries[existingIndex].evidenceCount += 1
            entries[existingIndex].updatedAt = Date()
            if !entries[existingIndex].sourceConversationIDs.contains(conversationID) {
                entries[existingIndex].sourceConversationIDs.append(conversationID)
            }
            persist()
            return entries[existingIndex]
        }

        let entry = MemoryEntry(
            content: trimmed,
            sourceConversationIDs: [conversationID]
        )
        entries.insert(entry, at: 0)
        prune()
        persist()
        return entry
    }

    func remove(id: UUID) {
        entries.removeAll { $0.id == id }
        persist()
    }

    func update(id: UUID, content: String) {
        guard let index = entries.firstIndex(where: { $0.id == id }) else { return }
        let trimmed = content.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        entries[index].content = trimmed
        entries[index].updatedAt = Date()
        persist()
    }

    func removeAll() {
        entries.removeAll()
        persist()
    }

    // MARK: - Prompt Injection

    /// Formatted memory block for system-prompt injection. Returns `nil`
    /// when there are no entries or the store is disabled, so the caller
    /// skips the block rather than emitting an empty tag.
    func formattedForPrompt() -> String? {
        guard isEnabled, !entries.isEmpty else { return nil }

        var lines: [String] = []
        var characterBudget = Self.maximumInjectedCharacters
        // Most-recently-updated first; small models benefit from seeing
        // the freshest context closest to the prompt tail.
        for entry in entries.sorted(by: { $0.updatedAt > $1.updatedAt }) {
            let line = "- \(entry.content)"
            guard line.count <= characterBudget else { break }
            lines.append(line)
            characterBudget -= line.count
        }
        guard !lines.isEmpty else { return nil }

        return """
        <memory_context>
        Durable facts and preferences learned from previous conversations:
        \(lines.joined(separator: "\n"))
        </memory_context>
        """
    }

    // MARK: - Persistence

    private func load() {
        guard FileManager.default.fileExists(atPath: fileURL.path) else { return }
        do {
            let data = try Data(contentsOf: fileURL)
            let library = try JSONDecoder().decode(MemoryLibrary.self, from: data)
            entries = library.entries
        } catch {
            // A corrupted memory file should never prevent the app from
            // launching; start fresh rather than crashing on read.
            entries = []
        }
    }

    private func persist() {
        let library = MemoryLibrary(entries: entries)
        do {
            let directory = fileURL.deletingLastPathComponent()
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            let data = try JSONEncoder().encode(library)
            try data.write(to: fileURL, options: .atomic)
        } catch {
            // Persistence failure is non-fatal; the in-memory state is
            // still usable for this session.
        }
    }

    /// Drops entries beyond ``maximumEntries``, evicting the lowest-
    /// evidence items first so corroboration acts as a retention signal.
    private func prune() {
        guard entries.count > Self.maximumEntries else { return }
        let sorted = entries.sorted { a, b in
            if a.evidenceCount != b.evidenceCount { return a.evidenceCount > b.evidenceCount }
            return a.updatedAt > b.updatedAt
        }
        entries = Array(sorted.prefix(Self.maximumEntries))
    }
}
