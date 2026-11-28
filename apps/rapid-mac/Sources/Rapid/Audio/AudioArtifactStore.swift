import Foundation

/// Speech the user has synthesized, kept so the Artifacts library can list it
/// and play it back. Same disk discipline as ``ImageArtifactStore``: serial
/// transactions, files published before the index, retired files removed
/// after the new index is written.
final class AudioArtifactStore: Sendable {
    static let defaultLimit = 50
    private static let queue = DispatchQueue(
        label: "com.rapidmlx.rapid.audio-artifacts", qos: .utility
    )

    struct Entry: Codable, Equatable, Identifiable, Sendable {
        let id: UUID
        let text: String
        let voice: String
        let modelAlias: String
        let fileExtension: String
        let byteCount: Int
        let createdAt: Date
    }

    let directory: URL
    let limit: Int

    init(
        directory: URL = ApplicationSupportLocator.audioArtifactsDirectory(),
        limit: Int = AudioArtifactStore.defaultLimit
    ) {
        self.directory = directory
        self.limit = max(0, limit)
    }

    private var indexURL: URL { directory.appendingPathComponent("index.json") }

    func fileURL(for entry: Entry) -> URL {
        directory.appendingPathComponent("\(entry.id.uuidString).\(entry.fileExtension)")
    }

    private func readIndex() throws -> [Entry] {
        let data: Data
        do { data = try Data(contentsOf: indexURL) }
        catch CocoaError.fileReadNoSuchFile { return [] }
        return try JSONDecoder().decode([Entry].self, from: data)
            .sorted { $0.createdAt > $1.createdAt }
    }

    private func prepareDirectory() throws {
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true,
            attributes: [.posixPermissions: 0o700]
        )
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o700], ofItemAtPath: directory.path
        )
    }

    private func commit(_ entries: [Entry], retiring: [Entry]) throws {
        try OwnerOnlyFile.writeAtomically(JSONEncoder().encode(entries), to: indexURL)
        for entry in retiring {
            let url = fileURL(for: entry)
            if FileManager.default.fileExists(atPath: url.path) {
                try FileManager.default.removeItem(at: url)
            }
        }
    }

    /// Newest first. Never writes, so listing cannot damage an index it
    /// failed to read.
    func entries() async throws -> [Entry] {
        try await withCheckedThrowingContinuation { continuation in
            Self.queue.async {
                continuation.resume(with: Result { try self.readIndex() })
            }
        }
    }

    /// Enqueue synchronously so termination's flush includes every submitted write.
    func insert(
        _ audio: SynthesizedAudio, text: String, voice: String, modelAlias: String,
        completion: @escaping @Sendable (String?) -> Void = { _ in }
    ) {
        let entry = Entry(
            id: UUID(), text: text, voice: voice, modelAlias: modelAlias,
            fileExtension: audio.fileExtension, byteCount: audio.data.count,
            createdAt: Date()
        )
        Self.queue.async {
            do {
                let previous = try self.readIndex()
                try self.prepareDirectory()
                try OwnerOnlyFile.writeAtomically(audio.data, to: self.fileURL(for: entry))
                let all = [entry] + previous
                try self.commit(Array(all.prefix(self.limit)), retiring: Array(all.dropFirst(self.limit)))
                completion(nil)
            } catch { completion(error.localizedDescription) }
        }
    }

    static func flush() { queue.sync {} }
}
