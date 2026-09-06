import Foundation

struct DownloadCleanupCandidate: Identifiable, Equatable, Sendable {
    let url: URL
    let byteCount: Int64
    let modifiedAt: Date
    let resourceIdentifier: Data

    var id: URL { url }
    var name: String { url.lastPathComponent }
}

/// A deliberately narrow first cleanup surface.
///
/// The scanner considers only ordinary, top-level files in Downloads that
/// have not been modified within the chosen age. It never descends into
/// folders, follows symlinks, touches dotfiles, or inspects arbitrary app
/// caches. This is less magical than a whole-disk cleaner and materially safer:
/// every returned URL is understandable, independently selectable, and moved
/// through macOS Trash rather than unlinked.
struct DownloadCleanup {
    struct BatchResult: Equatable, Sendable {
        let movedCount: Int
        let failureDescription: String?
    }
    enum Failure: LocalizedError, Equatable {
        case targetEscapedDownloads
        case targetChanged

        var errorDescription: String? {
            switch self {
            case .targetEscapedDownloads:
                return "The file is no longer directly inside Downloads."
            case .targetChanged:
                return "The file changed after the preview. Review the refreshed list before trying again."
            }
        }
    }

    static let defaultMinimumAgeDays = 90

    static func scan(
        downloadsURL: URL,
        now: Date = Date(),
        minimumAgeDays: Int = defaultMinimumAgeDays,
        fileManager: FileManager = .default
    ) throws -> [DownloadCleanupCandidate] {
        let root = downloadsURL.standardizedFileURL.resolvingSymlinksInPath()
        let cutoff = now.addingTimeInterval(-Double(minimumAgeDays) * 86_400)
        let keys: Set<URLResourceKey> = [
            .isRegularFileKey, .isSymbolicLinkKey, .isHiddenKey,
            .contentModificationDateKey, .fileAllocatedSizeKey, .fileSizeKey,
            .fileResourceIdentifierKey
        ]
        let children = try fileManager.contentsOfDirectory(
            at: root,
            includingPropertiesForKeys: Array(keys),
            options: [.skipsSubdirectoryDescendants]
        )

        return try children.compactMap { url in
            let values = try url.resourceValues(forKeys: keys)
            guard values.isRegularFile == true,
                  values.isSymbolicLink != true,
                  values.isHidden != true,
                  let modifiedAt = values.contentModificationDate,
                  let identifier = values.fileResourceIdentifier as? NSData,
                  modifiedAt <= cutoff else { return nil }
            let bytes = Int64(values.fileAllocatedSize ?? values.fileSize ?? 0)
            return DownloadCleanupCandidate(
                url: url.standardizedFileURL,
                byteCount: max(0, bytes),
                modifiedAt: modifiedAt,
                resourceIdentifier: identifier as Data
            )
        }
        .sorted {
            if $0.byteCount != $1.byteCount { return $0.byteCount > $1.byteCount }
            return $0.name.localizedStandardCompare($1.name) == .orderedAscending
        }
    }

    static func moveToTrash(
        _ candidate: DownloadCleanupCandidate,
        downloadsURL: URL,
        trash: (URL) throws -> Void = { url in
            _ = try FileManager.default.trashItem(at: url, resultingItemURL: nil)
        }
    ) throws {
        let root = downloadsURL.standardizedFileURL.resolvingSymlinksInPath()
        let current = candidate.url.standardizedFileURL
        guard current.deletingLastPathComponent().resolvingSymlinksInPath() == root else {
            throw Failure.targetEscapedDownloads
        }
        let values = try current.resourceValues(forKeys: [
            .isRegularFileKey, .isSymbolicLinkKey, .contentModificationDateKey,
            .fileAllocatedSizeKey, .fileSizeKey, .fileResourceIdentifierKey
        ])
        let currentBytes = Int64(values.fileAllocatedSize ?? values.fileSize ?? 0)
        let currentIdentifier = (values.fileResourceIdentifier as? NSData).map { $0 as Data }
        guard values.isRegularFile == true,
              values.isSymbolicLink != true,
              values.contentModificationDate == candidate.modifiedAt,
              currentBytes == candidate.byteCount,
              currentIdentifier == candidate.resourceIdentifier else {
            throw Failure.targetChanged
        }
        try trash(current)
    }

    static func moveToTrash(
        _ candidates: [DownloadCleanupCandidate],
        downloadsURL: URL,
        trash: (URL) throws -> Void = { url in
            _ = try FileManager.default.trashItem(at: url, resultingItemURL: nil)
        }
    ) -> BatchResult {
        var moved = 0
        for candidate in candidates {
            do {
                try moveToTrash(candidate, downloadsURL: downloadsURL, trash: trash)
                moved += 1
            } catch {
                return BatchResult(movedCount: moved, failureDescription: error.localizedDescription)
            }
        }
        return BatchResult(movedCount: moved, failureDescription: nil)
    }
}
