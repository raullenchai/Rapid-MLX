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
        try Data("current\n".utf8).write(to: repo.appendingPathComponent("refs/main"))
        let entry = ModelEntry(alias: "audio", hfRepo: "test/audio", sizeOnDisk: nil, cached: true)
        let paths = DictationSnapshot.cachedSnapshots(for: entry, cacheRoot: root)
        #expect(paths == [main])
        #expect(paths.contains(where: DictationSnapshot.isDirectory))
        try FileManager.default.removeItem(at: main)
        #expect(!paths.contains(where: DictationSnapshot.isDirectory))
        #expect(DictationSnapshot.cachedSnapshots(for: entry, cacheRoot: root) == [main])
    }

    @Test("Offline revisions and internal cache symlinks are resolved only at refresh")
    func offlineSnapshot() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("dictation-snapshot-\(UUID())")
        defer { try? FileManager.default.removeItem(at: root) }
        let target = root.appendingPathComponent("archive/snapshots/pinned")
        try FileManager.default.createDirectory(at: target, withIntermediateDirectories: true)
        try FileManager.default.createSymbolicLink(at: root.appendingPathComponent("models--test--audio"),
                                                 withDestinationURL: root.appendingPathComponent("archive"))
        let entry = ModelEntry(alias: "audio", hfRepo: "test/audio", sizeOnDisk: nil, cached: true)
        let paths = DictationSnapshot.cachedSnapshots(for: entry, cacheRoot: root)
        #expect(paths.count == 1)
        #expect(paths.contains(where: DictationSnapshot.isDirectory))
        try FileManager.default.removeItem(at: target)
        #expect(!paths.contains(where: DictationSnapshot.isDirectory))
        let unknown = ModelEntry(alias: "audio", hfRepo: nil, sizeOnDisk: nil, cached: true)
        #expect(DictationSnapshot.cachedSnapshots(for: unknown, cacheRoot: root).isEmpty)
        #expect(DictationSnapshot.cachedSnapshots(for: entry, cacheRoot: nil).isEmpty)
        let missing = ModelEntry(alias: "audio", hfRepo: "test/audio", sizeOnDisk: nil, cached: false)
        #expect(DictationSnapshot.cachedSnapshots(for: missing, cacheRoot: root).isEmpty)
    }
}
