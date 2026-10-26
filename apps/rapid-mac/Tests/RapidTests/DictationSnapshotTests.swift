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
    @Test("Missing or truncated artifacts and deleted blob targets refuse capture", arguments: ["weight", "config", "blob", "truncate", "replace", "rewrite"])
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
        case "replace": try Data([4, 5, 6]).write(to: weight, options: .atomic)
        case "rewrite": try Data([4, 5, 6]).write(to: weight)
        default: try FileManager.default.removeItem(at: weight)
        }
        #expect(DictationSnapshot.isDirectory(snapshot))
        #expect(!paths.contains { $0.isOnDisk })
        if kind != "replace" && kind != "rewrite" {
            #expect(DictationSnapshot.cachedSnapshots(for: entry, cacheRoot: root).isEmpty)
        }
    }

    @Test("Repointing main cannot authorize the old snapshot", arguments: [false, true])
    func changedReference(atomic: Bool) throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("dictation-reference-\(UUID())")
        defer { try? FileManager.default.removeItem(at: root) }
        let repo = root.appendingPathComponent("models--test--audio")
        let snapshot = repo.appendingPathComponent("snapshots/current")
        try FileManager.default.createDirectory(at: snapshot, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: repo.appendingPathComponent("refs"), withIntermediateDirectories: true)
        try populate(snapshot)
        let ref = repo.appendingPathComponent("refs/main")
        try Data("current".utf8).write(to: ref)
        let entry = ModelEntry(alias: "audio", hfRepo: "test/audio", sizeOnDisk: nil, cached: true)
        let paths = DictationSnapshot.cachedSnapshots(for: entry, cacheRoot: root)
        #expect(paths.count == 1)
        #expect(paths.allSatisfy { $0.isOnDisk })
        // Same byte count: size alone must not authorize a different revision.
        try Data("missing".utf8).write(to: ref, options: atomic ? .atomic : [])
        #expect(!paths.contains { $0.isOnDisk })
        #expect(DictationSnapshot.isDirectory(snapshot))
        #expect(DictationSnapshot.cachedSnapshots(for: entry, cacheRoot: root).isEmpty)
        try FileManager.default.removeItem(at: ref)
        let offline = DictationSnapshot.cachedSnapshots(for: entry, cacheRoot: root)
        #expect(offline.count == 1)
        #expect(offline.allSatisfy { $0.isOnDisk })
        try Data("missing".utf8).write(to: ref)
        #expect(!offline.contains { $0.isOnDisk })
    }

    @Test("Files added to the snapshot or nested directories invalidate its inventory", arguments: ["", "tokenizer", ".metadata"])
    func addedArtifacts(subdirectory: String) throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("dictation-additions-\(UUID())")
        defer { try? FileManager.default.removeItem(at: root) }
        let snapshot = root.appendingPathComponent("models--test--audio/snapshots/pinned")
        let destination = snapshot.appendingPathComponent(subdirectory)
        try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
        try populate(snapshot)
        let entry = ModelEntry(alias: "audio", hfRepo: "test/audio", sizeOnDisk: nil, cached: true)
        let paths = DictationSnapshot.cachedSnapshots(for: entry, cacheRoot: root)
        #expect(paths.count == 1)
        #expect(paths.allSatisfy { $0.isOnDisk })
        try Data("new metadata".utf8).write(to: destination.appendingPathComponent("added.json"))
        #expect(!paths.contains { $0.isOnDisk })
    }

    @Test("Nested configs and auxiliary binaries cannot stand in for root STT artifacts", arguments: ["config", "weights"])
    func unrelatedArtifacts(missing: String) throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("dictation-layout-\(UUID())")
        defer { try? FileManager.default.removeItem(at: root) }
        let snapshot = root.appendingPathComponent("models--test--audio/snapshots/pinned")
        let nested = snapshot.appendingPathComponent("auxiliary")
        try FileManager.default.createDirectory(at: nested, withIntermediateDirectories: true)
        try populate(snapshot)
        try populate(nested)
        try Data([1]).write(to: snapshot.appendingPathComponent("tokenizer.bin"))
        try FileManager.default.removeItem(at: snapshot.appendingPathComponent(missing == "config" ? "config.json" : "model.safetensors"))
        let entry = ModelEntry(alias: "audio", hfRepo: "test/audio", sizeOnDisk: nil, cached: true)
        #expect(DictationSnapshot.cachedSnapshots(for: entry, cacheRoot: root).isEmpty)
    }

    @Test("Whisper accepts only its named root weights", arguments: ["weights.npz", "weights.safetensors"])
    func whisperLayout(weight: String) throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("dictation-whisper-\(UUID())")
        defer { try? FileManager.default.removeItem(at: root) }
        let snapshot = root.appendingPathComponent("models--test--audio/snapshots/pinned")
        try FileManager.default.createDirectory(at: snapshot, withIntermediateDirectories: true)
        try populate(snapshot)
        let entry = ModelEntry(alias: "audio", hfRepo: "test/audio", sizeOnDisk: nil, cached: true, audioFamily: "whisper")
        #expect(DictationSnapshot.cachedSnapshots(for: entry, cacheRoot: root).isEmpty)
        try Data([1, 2, 3]).write(to: snapshot.appendingPathComponent(weight))
        #expect(DictationSnapshot.cachedSnapshots(for: entry, cacheRoot: root).count == 1)
    }

    @Test("A shard or index cannot declare missing model weights complete")
    func incompleteShards() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("dictation-shards-\(UUID())")
        defer { try? FileManager.default.removeItem(at: root) }
        let snapshot = root.appendingPathComponent("models--test--audio/snapshots/pinned")
        try FileManager.default.createDirectory(at: snapshot, withIntermediateDirectories: true)
        try populate(snapshot)
        let entry = ModelEntry(alias: "audio", hfRepo: "test/audio", sizeOnDisk: nil, cached: true)
        try Data([1]).write(to: snapshot.appendingPathComponent("model-00001-of-00002.safetensors"))
        #expect(DictationSnapshot.cachedSnapshots(for: entry, cacheRoot: root).isEmpty)
        try Data([2]).write(to: snapshot.appendingPathComponent("model-00002-of-00002.safetensors"))
        #expect(DictationSnapshot.cachedSnapshots(for: entry, cacheRoot: root).count == 1)
        let index = snapshot.appendingPathComponent("model.safetensors.index.json")
        try Data(#"{"weight_map":{"weight":"absent.safetensors"}}"#.utf8).write(to: index)
        #expect(DictationSnapshot.cachedSnapshots(for: entry, cacheRoot: root).isEmpty)
        try Data(#"{"weight_map":{"weight":"model-00001-of-00002.safetensors"}}"#.utf8).write(to: index)
        #expect(DictationSnapshot.cachedSnapshots(for: entry, cacheRoot: root).count == 1)
    }

    @Test("A dangling main reference never authorizes offline fallback")
    func danglingReference() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("dictation-dangling-\(UUID())")
        defer { try? FileManager.default.removeItem(at: root) }
        let repo = root.appendingPathComponent("models--test--audio")
        let snapshot = repo.appendingPathComponent("snapshots/pinned")
        try FileManager.default.createDirectory(at: snapshot, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: repo.appendingPathComponent("refs"), withIntermediateDirectories: true)
        try populate(snapshot)
        let reference = repo.appendingPathComponent("refs/main")
        let target = repo.appendingPathComponent("refs/target")
        let entry = ModelEntry(alias: "audio", hfRepo: "test/audio", sizeOnDisk: nil, cached: true)
        try FileManager.default.createSymbolicLink(at: reference, withDestinationURL: target)
        #expect(DictationSnapshot.cachedSnapshots(for: entry, cacheRoot: root).isEmpty)
        try FileManager.default.removeItem(at: reference)
        let offline = DictationSnapshot.cachedSnapshots(for: entry, cacheRoot: root)
        #expect(offline.count == 1)
        #expect(offline.allSatisfy { $0.isOnDisk })
        try FileManager.default.createSymbolicLink(at: reference, withDestinationURL: target)
        #expect(!offline.contains { $0.isOnDisk })
        try Data("pinned".utf8).write(to: target)
        let repaired = DictationSnapshot.cachedSnapshots(for: entry, cacheRoot: root)
        #expect(repaired.count == 1)
        #expect(repaired.allSatisfy { $0.isOnDisk })
        try FileManager.default.removeItem(at: target)
        #expect(!repaired.contains { $0.isOnDisk })
    }

    private func populate(_ snapshot: URL) throws {
        try Data("{}".utf8).write(to: snapshot.appendingPathComponent("config.json"))
        try Data([1, 2, 3]).write(to: snapshot.appendingPathComponent("model.safetensors"))
    }

}
