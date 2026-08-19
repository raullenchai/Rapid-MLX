import Foundation
import Observation

/// Recent dictations, plus the audio that produced them.
///
/// Keeping the audio is what makes the "Fix" flow trustworthy: when a user
/// corrects a word, the correction can be re-run against the original recording
/// to confirm the new vocabulary actually fixes it before it is saved. Adding a
/// term has been measured to regress a *different* term, so saving a hint
/// unverified is how a word list quietly rots.
///
/// It is also the only path to per-user model adaptation later — audio that was
/// never captured cannot be recovered retroactively.
@MainActor
@Observable
final class DictationHistory {
    struct Entry: Codable, Identifiable, Sendable {
        let id: UUID
        let date: Date
        var text: String
        let duration: TimeInterval
        let latency: TimeInterval
        let appName: String?
        /// File name inside the audio directory; nil when archiving is off.
        let audioFile: String?

        init(
            id: UUID = UUID(),
            date: Date = Date(),
            text: String,
            duration: TimeInterval,
            latency: TimeInterval,
            appName: String?,
            audioFile: String?
        ) {
            self.id = id
            self.date = date
            self.text = text
            self.duration = duration
            self.latency = latency
            self.appName = appName
            self.audioFile = audioFile
        }
    }

    /// Cap on retained entries. Audio dominates the footprint (16 kHz mono ≈
    /// 32 KB/s), so 200 entries of typical length stays well under ~100 MB.
    static let maxEntries = 200

    private(set) var entries: [Entry] = []

    private let directory: URL
    /// Serialize index/audio mutations so a later clear or edit can never be
    /// overtaken by an older detached write. The UI keeps a pending copy of a
    /// just-recorded clip so "Fix" works even before disk I/O completes.
    private let persistenceQueue = DispatchQueue(label: "ai.rapidmlx.dictation-history")
    private var pendingAudio: [UUID: Data] = [:]
    private var indexURL: URL { directory.appendingPathComponent("history.json") }
    private var audioDirectory: URL { directory.appendingPathComponent("audio", isDirectory: true) }

    init(directory: URL? = nil) {
        self.directory = directory ?? ApplicationSupportLocator.applicationSupportRoot()
            .appendingPathComponent("Dictation", isDirectory: true)
        load()
    }

    // MARK: - Recording

    @discardableResult
    func record(
        text: String,
        audio: Data?,
        duration: TimeInterval,
        latency: TimeInterval,
        appName: String?,
        archiveAudio: Bool
    ) -> Entry {
        let id = UUID()
        var audioFile: String?

        if archiveAudio, let audio {
            let name = "\(id.uuidString).wav"
            let url = audioDirectory.appendingPathComponent(name)
            let directory = audioDirectory
            pendingAudio[id] = audio
            persistenceQueue.async { [weak self] in
                try? FileManager.default.createDirectory(
                    at: directory,
                    withIntermediateDirectories: true
                )
                try? audio.write(to: url, options: .atomic)
                Task { @MainActor [weak self] in
                    self?.pendingAudio[id] = nil
                }
            }
            audioFile = name
        }

        let entry = Entry(
            id: id,
            text: text,
            duration: duration,
            latency: latency,
            appName: appName,
            audioFile: audioFile
        )
        entries.insert(entry, at: 0)
        trimIfNeeded()
        save()
        return entry
    }

    func updateText(_ text: String, for id: UUID) {
        guard let index = entries.firstIndex(where: { $0.id == id }) else { return }
        entries[index].text = text
        save()
    }

    func audioURL(for entry: Entry) -> URL? {
        entry.audioFile.map { audioDirectory.appendingPathComponent($0) }
    }

    func audioData(for entry: Entry) -> Data? {
        if let pending = pendingAudio[entry.id] { return pending }
        guard let url = audioURL(for: entry) else { return nil }
        return try? Data(contentsOf: url)
    }

    func remove(_ id: UUID) {
        guard let index = entries.firstIndex(where: { $0.id == id }) else { return }
        let entry = entries.remove(at: index)
        deleteAudio(for: entry)
        save()
    }

    func clear() {
        let toDelete = entries
        entries.removeAll()
        for entry in toDelete { deleteAudio(for: entry) }
        save()
    }

    /// Wait until all mutations queued before this call have reached disk.
    /// Used by deterministic tests and by any future shutdown flush path.
    func waitForPersistence() async {
        await withCheckedContinuation { continuation in
            persistenceQueue.async { continuation.resume() }
        }
    }

    // MARK: - Persistence

    private func trimIfNeeded() {
        guard entries.count > Self.maxEntries else { return }
        let dropped = entries.suffix(from: Self.maxEntries)
        for entry in dropped { deleteAudio(for: entry) }
        entries = Array(entries.prefix(Self.maxEntries))
    }

    private func deleteAudio(for entry: Entry) {
        pendingAudio[entry.id] = nil
        guard let url = audioURL(for: entry) else { return }
        persistenceQueue.async {
            try? FileManager.default.removeItem(at: url)
        }
    }

    private func load() {
        let decoder = JSONDecoder()
        // Must mirror `save()`'s `.iso8601` encoding, or every launch decodes to
        // nothing and silently starts with an empty history.
        decoder.dateDecodingStrategy = .iso8601
        guard let data = try? Data(contentsOf: indexURL),
              let decoded = try? decoder.decode([Entry].self, from: data) else { return }
        entries = decoded
    }

    private func save() {
        let url = indexURL
        let snapshot = entries
        persistenceQueue.async {
            try? FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            encoder.dateEncodingStrategy = .iso8601
            guard let data = try? encoder.encode(snapshot) else { return }
            try? data.write(to: url, options: .atomic)
        }
    }
}
