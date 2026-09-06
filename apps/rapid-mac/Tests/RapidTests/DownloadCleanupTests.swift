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
        let target = try write("candidate.txt", bytes: 16, in: root, modifiedAt: old)
        let candidate = try #require(
            DownloadCleanup.scan(
                downloadsURL: root,
                now: old.addingTimeInterval(100 * 86_400)
            ).first
        )
        var trashed: URL?

        try DownloadCleanup.moveToTrash(candidate, downloadsURL: root) { trashed = $0 }
        #expect(trashed == target.standardizedFileURL)

        try FileManager.default.setAttributes(
            [.modificationDate: old.addingTimeInterval(1)],
            ofItemAtPath: target.path
        )
        #expect(throws: DownloadCleanup.Failure.targetChanged) {
            try DownloadCleanup.moveToTrash(candidate, downloadsURL: root) { _ in
                Issue.record("A changed file must not reach Trash")
            }
        }
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
        let url = try write("outside.txt", bytes: 4, in: outside, modifiedAt: old)
        let candidate = DownloadCleanupCandidate(url: url, byteCount: 4, modifiedAt: old)

        #expect(throws: DownloadCleanup.Failure.targetEscapedDownloads) {
            try DownloadCleanup.moveToTrash(candidate, downloadsURL: root) { _ in
                Issue.record("An out-of-scope file must not reach Trash")
            }
        }
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
