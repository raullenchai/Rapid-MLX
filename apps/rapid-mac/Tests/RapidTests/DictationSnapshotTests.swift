import Foundation
import Testing
@testable import Rapid

struct DictationSnapshotTests {
    @Test("Main revision deletion is not hidden by another cached revision")
    func mainRevision() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("dictation-snapshot-\(UUID())")
        defer { try? FileManager.default.removeItem(at: root) }
        let repo = root.appendingPathComponent("models--test--audio")
        let main = repo.appendingPathComponent("snapshots/current", isDirectory: true)
        let other = repo.appendingPathComponent("snapshots/older")
        for directory in [main, other, repo.appendingPathComponent("refs")] {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        }
        try populate(main)
        try populate(other)
        try Data("current\n".utf8).write(to: repo.appendingPathComponent("refs/main"))
        let entry = ModelEntry(alias: "audio", hfRepo: "test/audio", sizeOnDisk: nil, cached: true)
        let paths = DictationSnapshot.cachedSnapshots(for: entry, cacheRoot: root)
        #expect(paths.map(\.directory) == [main])
        #expect(paths.contains { $0.isOnDisk })
        try FileManager.default.removeItem(at: main)
        #expect(!paths.contains { $0.isOnDisk })
        #expect(DictationSnapshot.cachedSnapshots(for: entry, cacheRoot: root).isEmpty)
    }

    @Test("Offline revisions and internal cache symlinks are resolved only at refresh")
    func offlineSnapshot() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("dictation-snapshot-\(UUID())")
        defer { try? FileManager.default.removeItem(at: root) }
        let target = root.appendingPathComponent("archive/snapshots/pinned")
        try FileManager.default.createDirectory(at: target, withIntermediateDirectories: true)
        try populate(target)
        try FileManager.default.createSymbolicLink(at: root.appendingPathComponent("models--test--audio"),
                                                 withDestinationURL: root.appendingPathComponent("archive"))
        let entry = ModelEntry(alias: "audio", hfRepo: "test/audio", sizeOnDisk: nil, cached: true)
        let paths = DictationSnapshot.cachedSnapshots(for: entry, cacheRoot: root)
        #expect(paths.count == 1)
        #expect(paths.contains { $0.isOnDisk })
        try FileManager.default.removeItem(at: target)
        #expect(!paths.contains { $0.isOnDisk })
        let unknown = ModelEntry(alias: "audio", hfRepo: nil, sizeOnDisk: nil, cached: true)
        #expect(DictationSnapshot.cachedSnapshots(for: unknown, cacheRoot: root).isEmpty)
        #expect(DictationSnapshot.cachedSnapshots(for: entry, cacheRoot: nil).isEmpty)
        let missing = ModelEntry(alias: "audio", hfRepo: "test/audio", sizeOnDisk: nil, cached: false)
        #expect(DictationSnapshot.cachedSnapshots(for: missing, cacheRoot: root).isEmpty)
    }
    @Test("Missing or truncated artifacts and deleted blob targets refuse capture", arguments: ["weight", "config", "blob", "truncate"])
    func partialDeletion(kind: String) throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("dictation-artifacts-\(UUID())")
        defer { try? FileManager.default.removeItem(at: root) }
        let snapshot = root.appendingPathComponent("models--test--audio/snapshots/pinned")
        try FileManager.default.createDirectory(at: snapshot, withIntermediateDirectories: true)
        let entry = ModelEntry(alias: "audio", hfRepo: "test/audio", sizeOnDisk: nil, cached: true)
        #expect(DictationSnapshot.cachedSnapshots(for: entry, cacheRoot: root).isEmpty, "Empty snapshots are not cached models")
        try populate(snapshot)
        let weight = snapshot.appendingPathComponent("model.safetensors")
        let blob = root.appendingPathComponent("blob")
        if kind == "blob" {
            try FileManager.default.moveItem(at: weight, to: blob)
            try FileManager.default.createSymbolicLink(at: weight, withDestinationURL: blob)
        }
        let paths = DictationSnapshot.cachedSnapshots(for: entry, cacheRoot: root)
        #expect(paths.count == 1)
        #expect(paths.allSatisfy { $0.isOnDisk })
        switch kind {
        case "config": try FileManager.default.removeItem(at: snapshot.appendingPathComponent("config.json"))
        case "blob": try FileManager.default.removeItem(at: blob)
        case "truncate": try Data().write(to: weight)
        default: try FileManager.default.removeItem(at: weight)
        }
        #expect(DictationSnapshot.isDirectory(snapshot))
        #expect(!paths.contains { $0.isOnDisk })
        #expect(DictationSnapshot.cachedSnapshots(for: entry, cacheRoot: root).isEmpty)
    }

    private func populate(_ snapshot: URL) throws {
        try Data("{}".utf8).write(to: snapshot.appendingPathComponent("config.json"))
        try Data([1, 2, 3]).write(to: snapshot.appendingPathComponent("model.safetensors"))
    }

}
