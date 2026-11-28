import Foundation

/// Serial disk transactions: publish PNGs before the index, and remove retired
/// files only after publishing the new index. Never sweep unindexed files: they
/// may be recoverable artifacts from an interrupted write or damaged index.
final class ImageArtifactStore: Sendable {
    static let defaultLimit = 30
    private static let queue = DispatchQueue(label: "com.rapidmlx.rapid.image-artifacts", qos: .utility)
    let directory: URL
    let limit: Int

    private struct Entry: Codable {
        let id: UUID
        let prompt: String
        let modelAlias: String
        let size: String
        let seed: Int?
        let createdAt: Date
        let isEdit: Bool

        init(_ image: GeneratedImage) {
            id = image.id
            prompt = image.prompt
            modelAlias = image.modelAlias
            size = image.size
            seed = image.seed
            createdAt = image.createdAt
            isEdit = image.isEdit
        }

        func image(_ data: Data) -> GeneratedImage {
            GeneratedImage(id: id, pngData: data, prompt: prompt, isEdit: isEdit,
                           modelAlias: modelAlias, size: size, seed: seed, createdAt: createdAt)
        }
    }

    init(directory: URL = ApplicationSupportLocator.imageArtifactsDirectory(), limit: Int = ImageArtifactStore.defaultLimit) {
        self.directory = directory
        self.limit = max(0, limit)
    }

    private var indexURL: URL { directory.appendingPathComponent("index.json") }
    func imageURL(_ id: UUID) -> URL { directory.appendingPathComponent("\(id.uuidString).png") }

    /// Missing is a first run. All other read/decode errors propagate, preventing
    /// subsequent mutations from replacing an unreadable index with an empty one.
    private func readIndex() throws -> [Entry] {
        let data: Data
        do { data = try Data(contentsOf: indexURL) }
        catch CocoaError.fileReadNoSuchFile { return [] }
        return try JSONDecoder().decode([Entry].self, from: data)
            .sorted { $0.createdAt > $1.createdAt }
    }

    private func commit(_ entries: [Entry], retiring: [Entry]) throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true,
                                                attributes: [.posixPermissions: 0o700])
        try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: directory.path)
        try OwnerOnlyFile.writeAtomically(JSONEncoder().encode(entries), to: indexURL)
        for entry in retiring {
            let url = imageURL(entry.id)
            if FileManager.default.fileExists(atPath: url.path) {
                try FileManager.default.removeItem(at: url)
            }
        }
    }

    struct Metadata: Equatable, Identifiable, Sendable {
        let id: UUID
        let prompt: String
        let modelAlias: String
        let size: String
        let createdAt: Date
        let isEdit: Bool
        let byteCount: Int
    }

    /// The gallery's index without the pixels, for listings. Never writes.
    func metadata() async throws -> [Metadata] {
        try await withCheckedThrowingContinuation { continuation in
            Self.queue.async {
                continuation.resume(with: Result {
                    try self.readIndex().prefix(self.limit).map { entry in
                        let bytes = (try? FileManager.default.attributesOfItem(
                            atPath: self.imageURL(entry.id).path
                        )[.size] as? Int) ?? 0
                        return Metadata(
                            id: entry.id, prompt: entry.prompt, modelAlias: entry.modelAlias,
                            size: entry.size, createdAt: entry.createdAt,
                            isEdit: entry.isEdit, byteCount: bytes
                        )
                    }
                })
            }
        }
    }

    func load() async throws -> [GeneratedImage] {
        try await withCheckedThrowingContinuation { continuation in
            Self.queue.async {
                do {
                    let entries = try self.readIndex()
                    let retained = Array(entries.prefix(self.limit))
                    // Read before evicting: unreadable artifacts must not cause a
                    // partial gallery to be persisted over the original index.
                    let images = try retained.map { try $0.image(Data(contentsOf: self.imageURL($0.id))) }
                    if entries.count > self.limit {
                        try self.commit(retained, retiring: Array(entries.dropFirst(self.limit)))
                    }
                    continuation.resume(returning: images)
                } catch { continuation.resume(throwing: error) }
            }
        }
    }

    /// Enqueue synchronously so termination's flush includes every submitted mutation.
    func insert(_ images: [GeneratedImage], completion: @escaping @Sendable (String?) -> Void = { _ in }) {
        Self.queue.async {
            do {
                let previous = try self.readIndex()
                try FileManager.default.createDirectory(at: self.directory, withIntermediateDirectories: true,
                                                        attributes: [.posixPermissions: 0o700])
                try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: self.directory.path)
                for image in images {
                    try OwnerOnlyFile.writeAtomically(image.pngData, to: self.imageURL(image.id))
                }
                let ids = Set(images.map(\.id))
                let all = (images.map(Entry.init) + previous.filter { !ids.contains($0.id) })
                    .sorted { $0.createdAt > $1.createdAt }
                try self.commit(Array(all.prefix(self.limit)), retiring: Array(all.dropFirst(self.limit)))
                completion(nil)
            } catch { completion(error.localizedDescription) }
        }
    }

    func delete(_ id: UUID, completion: @escaping @Sendable (String?) -> Void = { _ in }) {
        Self.queue.async {
            do {
                let entries = try self.readIndex()
                // Imported edit sources have no durable gallery entry.
                guard entries.contains(where: { $0.id == id }) else { completion(nil); return }
                try self.commit(entries.filter { $0.id != id }, retiring: entries.filter { $0.id == id })
                completion(nil)
            } catch { completion(error.localizedDescription) }
        }
    }

    static func flush() { queue.sync {} }
}
