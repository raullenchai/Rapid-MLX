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
        let outcomeUncertain: Bool
    }
    enum Failure: LocalizedError, Equatable {
        case targetEscapedDownloads
        case targetChanged
        case recoveryRequired(String)
        case trashOutcomeUnknown(String)

        var errorDescription: String? {
            switch self {
            case .targetEscapedDownloads:
                return "The file is no longer directly inside Downloads."
            case .targetChanged:
                return "The file changed after the preview. Review the refreshed list before trying again."
            case .recoveryRequired(let path):
                return "Cleanup stopped, but Rapid could not restore the file automatically. Recover it from \(path)."
            case .trashOutcomeUnknown(let name):
                return "Rapid could not confirm whether \(name) moved to Trash. Check Trash before trying again."
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

        return children.compactMap { url in
            guard let values = try? url.resourceValues(forKeys: keys) else { return nil }
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
        fileManager: FileManager = .default,
        claim: ((URL, URL) throws -> Void)? = nil,
        trash: (URL) throws -> Void = { url in
            _ = try FileManager.default.trashItem(at: url, resultingItemURL: nil)
        }
    ) throws {
        let root = downloadsURL.standardizedFileURL.resolvingSymlinksInPath()
        let current = candidate.url.standardizedFileURL
        guard current.deletingLastPathComponent().resolvingSymlinksInPath() == root else {
            throw Failure.targetEscapedDownloads
        }
        let stagingDirectory = root.appendingPathComponent(
            ".rapid-cleanup-\(UUID().uuidString)",
            isDirectory: true
        )
        try fileManager.createDirectory(
            at: stagingDirectory,
            withIntermediateDirectories: false,
            attributes: [.posixPermissions: 0o700]
        )
        var removeStagingDirectory = true
        defer {
            if removeStagingDirectory {
                try? fileManager.removeItem(at: stagingDirectory)
            }
        }
        let staged = stagingDirectory.appendingPathComponent(candidate.name)
        try (claim ?? { source, destination in
            try fileManager.moveItem(at: source, to: destination)
        })(current, staged)

        let restoreClaimedFile: () throws -> Void = {
            guard !fileManager.fileExists(atPath: current.path) else {
                removeStagingDirectory = false
                throw Failure.recoveryRequired(staged.path)
            }
            do {
                try fileManager.moveItem(at: staged, to: current)
            } catch {
                removeStagingDirectory = false
                throw Failure.recoveryRequired(staged.path)
            }
        }

        let values: URLResourceValues
        do {
            values = try staged.resourceValues(forKeys: [
                .isRegularFileKey, .isSymbolicLinkKey, .contentModificationDateKey,
                .fileAllocatedSizeKey, .fileSizeKey, .fileResourceIdentifierKey
            ])
        } catch {
            try restoreClaimedFile()
            throw error
        }
        let currentBytes = Int64(values.fileAllocatedSize ?? values.fileSize ?? 0)
        let currentIdentifier = (values.fileResourceIdentifier as? NSData).map { $0 as Data }
        guard values.isRegularFile == true,
              values.isSymbolicLink != true,
              values.contentModificationDate == candidate.modifiedAt,
              currentBytes == candidate.byteCount,
              currentIdentifier == candidate.resourceIdentifier else {
            try restoreClaimedFile()
            throw Failure.targetChanged
        }
        do {
            try trash(staged)
        } catch {
            guard fileManager.fileExists(atPath: staged.path) else {
                throw Failure.trashOutcomeUnknown(candidate.name)
            }
            try restoreClaimedFile()
            throw error
        }
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
                var outcomeUncertain = false
                if let failure = error as? Failure,
                   case .trashOutcomeUnknown = failure {
                    outcomeUncertain = true
                }
                return BatchResult(
                    movedCount: moved,
                    failureDescription: error.localizedDescription,
                    outcomeUncertain: outcomeUncertain
                )
            }
        }
        return BatchResult(movedCount: moved, failureDescription: nil, outcomeUncertain: false)
    }
}
