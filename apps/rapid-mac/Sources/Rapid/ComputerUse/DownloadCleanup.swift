import Foundation

struct DownloadCleanupCandidate: Identifiable, Equatable, Sendable {
    let url: URL
    let byteCount: Int64
    let modifiedAt: Date

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
            .contentModificationDateKey, .fileAllocatedSizeKey, .fileSizeKey
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
                  modifiedAt <= cutoff else { return nil }
            let bytes = Int64(values.fileAllocatedSize ?? values.fileSize ?? 0)
            return DownloadCleanupCandidate(
                url: url.standardizedFileURL,
                byteCount: max(0, bytes),
                modifiedAt: modifiedAt
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
        let current = candidate.url.standardizedFileURL.resolvingSymlinksInPath()
        guard current.deletingLastPathComponent() == root else {
            throw Failure.targetEscapedDownloads
        }
        let values = try current.resourceValues(forKeys: [
            .isRegularFileKey, .isSymbolicLinkKey, .contentModificationDateKey
        ])
        guard values.isRegularFile == true,
              values.isSymbolicLink != true,
              values.contentModificationDate == candidate.modifiedAt else {
            throw Failure.targetChanged
        }
        try trash(current)
    }
}
