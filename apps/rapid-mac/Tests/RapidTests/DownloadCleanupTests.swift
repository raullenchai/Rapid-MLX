import Foundation
import Testing
@testable import Rapid

@Suite("Downloads cleanup", .serialized)
struct DownloadCleanupTests {
    @Test("Scan returns only old ordinary top-level files")
    func scanIsNarrow() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let now = Date(timeIntervalSince1970: 2_000_000_000)
        let old = now.addingTimeInterval(-100 * 86_400)
        let recent = now.addingTimeInterval(-2 * 86_400)

        let large = try write("large.zip", bytes: 128, in: root, modifiedAt: old)
        _ = try write("small.txt", bytes: 8, in: root, modifiedAt: old)
        _ = try write("recent.txt", bytes: 256, in: root, modifiedAt: recent)
        _ = try write(".hidden.txt", bytes: 256, in: root, modifiedAt: old)
        let folder = root.appendingPathComponent("old-folder", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: false)
        _ = try write("nested.txt", bytes: 256, in: folder, modifiedAt: old)
        let link = root.appendingPathComponent("linked.zip")
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: large)

        let result = try DownloadCleanup.scan(downloadsURL: root, now: now)

        #expect(result.map(\.name) == ["large.zip", "small.txt"])
    }

    @Test("Move revalidates scope and file identity before trash")
    func moveRevalidates() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let old = Date(timeIntervalSince1970: 1_000_000_000)
        _ = try write("candidate.txt", bytes: 16, in: root, modifiedAt: old)
        let candidate = try #require(
            DownloadCleanup.scan(
                downloadsURL: root,
                now: old.addingTimeInterval(100 * 86_400)
            ).first
        )
        var trashed: URL?

        try DownloadCleanup.moveToTrash(
            candidate,
            downloadsURL: root,
            trash: { trashed = $0 }
        )
        #expect(trashed?.lastPathComponent == "candidate.txt")
        #expect(trashed?.deletingLastPathComponent() != root.standardizedFileURL)

        let changed = try write("changed.txt", bytes: 16, in: root, modifiedAt: old)
        let changedCandidate = try #require(
            DownloadCleanup.scan(
                downloadsURL: root,
                now: old.addingTimeInterval(100 * 86_400)
            ).first { $0.url == changed }
        )
        try FileManager.default.setAttributes(
            [.modificationDate: old.addingTimeInterval(1)],
            ofItemAtPath: changed.path
        )
        #expect(throws: DownloadCleanup.Failure.targetChanged) {
            try DownloadCleanup.moveToTrash(
                changedCandidate,
                downloadsURL: root,
                trash: { _ in Issue.record("A changed file must not reach Trash") }
            )
        }
        #expect(FileManager.default.fileExists(atPath: changed.path))
    }

    @Test("Move rejects a candidate outside Downloads")
    func moveRejectsEscape() throws {
        let root = try temporaryDirectory()
        let outside = try temporaryDirectory()
        defer {
            try? FileManager.default.removeItem(at: root)
            try? FileManager.default.removeItem(at: outside)
        }
        let old = Date(timeIntervalSince1970: 1_000_000_000)
        _ = try write("outside.txt", bytes: 4, in: outside, modifiedAt: old)
        let candidate = try #require(
            DownloadCleanup.scan(
                downloadsURL: outside,
                now: old.addingTimeInterval(100 * 86_400)
            ).first
        )

        #expect(throws: DownloadCleanup.Failure.targetEscapedDownloads) {
            try DownloadCleanup.moveToTrash(
                candidate,
                downloadsURL: root,
                trash: { _ in Issue.record("An out-of-scope file must not reach Trash") }
            )
        }
    }

    @Test("A replacement symlink cannot move its target")
    func replacementSymlinkIsRejected() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let old = Date(timeIntervalSince1970: 1_000_000_000)
        let original = try write("selected.txt", bytes: 4, in: root, modifiedAt: old)
        let target = try write("target.txt", bytes: 4, in: root, modifiedAt: old)
        let candidate = try #require(DownloadCleanup.scan(
            downloadsURL: root, now: old.addingTimeInterval(100 * 86_400)
        ).first { $0.url == original })
        try FileManager.default.removeItem(at: original)
        try FileManager.default.createSymbolicLink(at: original, withDestinationURL: target)

        #expect(throws: DownloadCleanup.Failure.targetChanged) {
            try DownloadCleanup.moveToTrash(
                candidate,
                downloadsURL: root,
                trash: { _ in Issue.record("A replacement symlink must not reach Trash") }
            )
        }
    }

    @Test("A path replacement during the atomic claim is restored, not trashed")
    func claimRaceIsRejected() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let old = Date(timeIntervalSince1970: 1_000_000_000)
        let original = try write("selected.txt", bytes: 4, in: root, modifiedAt: old)
        let candidate = try #require(DownloadCleanup.scan(
            downloadsURL: root,
            now: old.addingTimeInterval(100 * 86_400)
        ).first)
        let approvedBackup = root.appendingPathComponent("approved-backup.txt")
        var trashCalled = false

        #expect(throws: DownloadCleanup.Failure.targetChanged) {
            try DownloadCleanup.moveToTrash(
                candidate,
                downloadsURL: root,
                claim: { source, destination in
                    try FileManager.default.moveItem(at: source, to: approvedBackup)
                    _ = try write("selected.txt", bytes: 4, in: root, modifiedAt: old)
                    try FileManager.default.moveItem(at: source, to: destination)
                    // The replacement is now atomically claimed at `destination`;
                    // the original path is vacant, so rejection can restore it.
                    #expect(!FileManager.default.fileExists(atPath: source.path))
                },
                trash: { _ in trashCalled = true }
            )
        }

        #expect(!trashCalled)
        #expect(FileManager.default.fileExists(atPath: original.path))
        #expect(FileManager.default.fileExists(atPath: approvedBackup.path))
    }

    @Test("Batch result reports files moved before a failure")
    func batchReportsPartialSuccess() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let old = Date(timeIntervalSince1970: 1_000_000_000)
        _ = try write("a.txt", bytes: 4, in: root, modifiedAt: old)
        _ = try write("b.txt", bytes: 4, in: root, modifiedAt: old)
        let candidates = try DownloadCleanup.scan(
            downloadsURL: root, now: old.addingTimeInterval(100 * 86_400)
        )
        var calls = 0
        let result = DownloadCleanup.moveToTrash(candidates, downloadsURL: root) { _ in
            calls += 1
            if calls == 2 { throw CocoaError(.fileWriteUnknown) }
        }

        #expect(result.movedCount == 1)
        #expect(result.failureDescription != nil)
        #expect(!result.outcomeUncertain)
    }

    @Test("A Trash failure restores the claimed file")
    func trashFailureRestoresClaim() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let old = Date(timeIntervalSince1970: 1_000_000_000)
        let original = try write("selected.txt", bytes: 4, in: root, modifiedAt: old)
        let candidate = try #require(DownloadCleanup.scan(
            downloadsURL: root,
            now: old.addingTimeInterval(100 * 86_400)
        ).first)

        #expect(throws: CocoaError.self) {
            try DownloadCleanup.moveToTrash(
                candidate,
                downloadsURL: root,
                trash: { _ in throw CocoaError(.fileWriteUnknown) }
            )
        }
        #expect(FileManager.default.fileExists(atPath: original.path))
        #expect(try Data(contentsOf: original) == Data(repeating: 0x41, count: 4))
    }

    @Test("A later scan recovers a claim interrupted by app termination")
    func scanRecoversInterruptedClaim() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let old = Date(timeIntervalSince1970: 1_000_000_000)
        let original = try write("selected.txt", bytes: 4, in: root, modifiedAt: old)
        let staging = root.appendingPathComponent(".rapid-cleanup-interrupted", isDirectory: true)
        try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: false)
        try Data(#"{"originalName":"selected.txt"}"#.utf8).write(
            to: staging.appendingPathComponent("claim.json")
        )
        try FileManager.default.moveItem(
            at: original,
            to: staging.appendingPathComponent("selected.txt")
        )

        let candidates = try DownloadCleanup.scan(
            downloadsURL: root,
            now: old.addingTimeInterval(100 * 86_400)
        )

        #expect(candidates.map(\.name) == ["selected.txt"])
        #expect(FileManager.default.fileExists(atPath: original.path))
        #expect(!FileManager.default.fileExists(atPath: staging.path))
    }

    @Test("An indeterminate Trash result is not reported as a safe stop")
    func batchReportsUnknownTrashOutcome() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let old = Date(timeIntervalSince1970: 1_000_000_000)
        _ = try write("selected.txt", bytes: 4, in: root, modifiedAt: old)
        let candidate = try #require(DownloadCleanup.scan(
            downloadsURL: root,
            now: old.addingTimeInterval(100 * 86_400)
        ).first)

        let result = DownloadCleanup.moveToTrash([candidate], downloadsURL: root) { staged in
            try FileManager.default.removeItem(at: staged)
            throw CocoaError(.fileWriteUnknown)
        }

        #expect(result.movedCount == 0)
        #expect(result.outcomeUncertain)
        #expect(result.failureDescription?.contains("Check Trash") == true)
    }

    private func temporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("rapid-download-cleanup-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: false)
        return url
    }

    @discardableResult
    private func write(
        _ name: String,
        bytes: Int,
        in directory: URL,
        modifiedAt: Date
    ) throws -> URL {
        let url = directory.appendingPathComponent(name)
        try Data(repeating: 0x41, count: bytes).write(to: url)
        try FileManager.default.setAttributes(
            [.modificationDate: modifiedAt],
            ofItemAtPath: url.path
        )
        return url
    }
}
