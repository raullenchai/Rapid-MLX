import Foundation
import Testing
@testable import Rapid

@Suite("Durable image gallery")
struct ImageArtifactStoreTests {
    private func directory() throws -> URL {
        let url = URL(fileURLWithPath: "/private/tmp/harbor-gallery-persist-tests", isDirectory: true)
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func image(_ prompt: String, time: TimeInterval, edit: Bool = false) -> GeneratedImage {
        GeneratedImage(pngData: Data(base64Encoded:
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+j4ZkAAAAASUVORK5CYII=")!,
            prompt: prompt, isEdit: edit, modelAlias: "test-image-model", size: "1x1",
            seed: edit ? nil : 42, createdAt: Date(timeIntervalSince1970: time))
    }

    private func insert(_ images: [GeneratedImage], into store: ImageArtifactStore) async -> String? {
        await withCheckedContinuation { continuation in
            store.insert(images) { continuation.resume(returning: $0) }
        }
    }

    private func delete(_ id: UUID, from store: ImageArtifactStore) async -> String? {
        await withCheckedContinuation { continuation in
            store.delete(id) { continuation.resume(returning: $0) }
        }
    }

    @Test func roundTripRetainsBytesMetadataAndNewestFirstOrder() async throws {
        let root = try directory()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = ImageArtifactStore(directory: root)
        #expect(try await store.load().isEmpty)
        let old = image("original", time: 1)
        let new = image("edit prompt", time: 2, edit: true)
        #expect(await insert([old, new], into: store) == nil)
        let restored = try await ImageArtifactStore(directory: root).load()
        #expect(restored == [new, old])
        for url in [root] + [root.appendingPathComponent("index.json"), store.imageURL(old.id), store.imageURL(new.id)] {
            let mode = try FileManager.default.attributesOfItem(atPath: url.path)[.posixPermissions] as? Int
            #expect(mode == (url == root ? 0o700 : 0o600))
        }
        let index = try String(contentsOf: root.appendingPathComponent("index.json"), encoding: .utf8)
        #expect(!index.contains(old.pngData.base64EncodedString()))
    }

    @Test func capEvictsOldestFilesOnInsertAndLoad() async throws {
        let root = try directory()
        defer { try? FileManager.default.removeItem(at: root) }
        let images = (0..<4).map { image("image \($0)", time: Double($0)) }
        let store = ImageArtifactStore(directory: root, limit: 3)
        #expect(await insert(Array(images.prefix(3)), into: store) == nil)
        #expect(await insert([images[3]], into: store) == nil)
        #expect(!FileManager.default.fileExists(atPath: store.imageURL(images[0].id).path))
        #expect(try await store.load() == Array(images.suffix(3).reversed()))
        let smaller = ImageArtifactStore(directory: root, limit: 2)
        #expect(try await smaller.load() == Array(images.suffix(2).reversed()))
        #expect(!FileManager.default.fileExists(atPath: store.imageURL(images[1].id).path))
        #expect(try await ImageArtifactStore(directory: root).load() == Array(images.suffix(2).reversed()))
    }

    @Test func deleteRemovesOnlyOwnedArtifactAndSurvivesRelaunch() async throws {
        let root = try directory()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = ImageArtifactStore(directory: root)
        let first = image("first", time: 1)
        let second = image("second", time: 2)
        #expect(await insert([first, second], into: store) == nil)
        let exported = root.appendingPathComponent("export.png")
        try first.pngData.write(to: exported)
        #expect(await delete(first.id, from: store) == nil)
        #expect(await delete(UUID(), from: store) == nil)
        #expect(!FileManager.default.fileExists(atPath: store.imageURL(first.id).path))
        #expect(try Data(contentsOf: exported) == first.pngData)
        #expect(try await ImageArtifactStore(directory: root).load() == [second])
    }

    @Test func corruptIndexCannotBeOverwrittenOrDeleteArtifacts() async throws {
        let root = try directory()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = ImageArtifactStore(directory: root, limit: 1)
        let original = image("original", time: 1)
        #expect(await insert([original], into: store) == nil)
        let index = root.appendingPathComponent("index.json")
        let corrupt = Data("{broken index".utf8)
        try corrupt.write(to: index)
        await #expect(throws: (any Error).self) { try await store.load() }
        #expect(await insert([image("new", time: 2)], into: store) != nil)
        #expect(await delete(original.id, from: store) != nil)
        #expect(try Data(contentsOf: index) == corrupt)
        #expect(try Data(contentsOf: store.imageURL(original.id)) == original.pngData)
    }

    @Test func unreadableIndexIsNotFirstRun() async throws {
        let root = try directory()
        defer { try? FileManager.default.removeItem(at: root) }
        // A directory at the index path is unreadable as JSON even for root.
        let index = root.appendingPathComponent("index.json", isDirectory: true)
        try FileManager.default.createDirectory(at: index, withIntermediateDirectories: true)
        let store = ImageArtifactStore(directory: root)
        await #expect(throws: (any Error).self) { try await store.load() }
        #expect(await insert([image("new", time: 1)], into: store) != nil)
        #expect((try index.resourceValues(forKeys: [.isDirectoryKey])).isDirectory == true)
    }

    @Test @MainActor func viewModelRestoresAndDeletesDurableGallery() async throws {
        let root = try directory()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = ImageArtifactStore(directory: root)
        let result = image("restored", time: 1, edit: true)
        #expect(await insert([result], into: store) == nil)
        let model = ImageGenViewModel(server: ServerManager(), artifactStore: store)
        await model.galleryLoadTask?.value
        #expect(model.results == [result])
        #expect(model.activeImage == result)
        model.delete(result)
        #expect(model.results.isEmpty)
        // The load is queued after the synchronous delete enqueue.
        #expect(try await ImageArtifactStore(directory: root).load().isEmpty)
        #expect(!FileManager.default.fileExists(atPath: store.imageURL(result.id).path))
    }

    @Test func editSizeIsReadFromPNG() {
        let original = image("source", time: 1)
        let edited = GeneratedImage(pngData: original.pngData, prompt: "edit", isEdit: true, modelAlias: "edit-model")
        #expect(edited.size == "1x1")
        #expect(edited.seed == nil)
    }

    @Test func homeOverrideIsolatesArtifacts() async throws {
        let root = try directory()
        defer { try? FileManager.default.removeItem(at: root) }
        let artifacts = ApplicationSupportLocator.imageArtifactsDirectory(environment: ["HOME": root.path])
        #expect(artifacts.path == root.path + "/Library/Application Support/Rapid/ImageArtifacts")
        let store = ImageArtifactStore(directory: artifacts)
        let result = image("isolated", time: 1)
        #expect(await insert([result], into: store) == nil)
        #expect(try await ImageArtifactStore(directory: artifacts).load() == [result])
    }
}
