import Darwin
import Foundation

struct FreeUpSpaceFileIdentity: Equatable, Hashable, Sendable {
    let device: UInt64
    let inode: UInt64
    let size: Int64
    let modifiedSeconds: Int64
    let modifiedNanoseconds: Int64
}

struct FreeUpSpaceCandidate: Identifiable, Equatable, Hashable, Sendable {
    let url: URL
    let identity: FreeUpSpaceFileIdentity
    let modifiedAt: Date

    var id: String {
        "\(identity.device):\(identity.inode):\(url.lastPathComponent)"
    }

    var name: String { url.lastPathComponent }
    var byteCount: Int64 { identity.size }

    func ageInDays(relativeTo now: Date) -> Int {
        max(0, Calendar.current.dateComponents(
            [.day],
            from: modifiedAt,
            to: now
        ).day ?? 0)
    }
}

enum FreeUpSpaceScanError: Error, Equatable, Sendable {
    case downloadsUnavailable
    case permissionDenied
    case enumerationFailed
    case cancelled
}

enum FreeUpSpaceMoveFailureReason: Equatable, Sendable {
    case changedSinceReview
    case permissionDenied
    case moveRejected
    case verificationFailed
}

struct FreeUpSpaceMoveFailure: Equatable, Sendable {
    let candidate: FreeUpSpaceCandidate
    let reason: FreeUpSpaceMoveFailureReason
}

struct FreeUpSpaceMoveOutcome: Equatable, Sendable {
    var moved: [FreeUpSpaceCandidate] = []
    var failures: [FreeUpSpaceMoveFailure] = []
    var wasCancelled = false

    var movedByteCount: Int64 {
        moved.reduce(0) { $0 + $1.byteCount }
    }
}

protocol FreeUpSpaceServicing: Sendable {
    func scan(now: Date) async throws -> [FreeUpSpaceCandidate]
    func moveToTrash(_ candidates: [FreeUpSpaceCandidate]) async -> FreeUpSpaceMoveOutcome
}

/// A deliberately narrow filesystem adapter for the first cleanup flow.
///
/// The service only inventories direct, old, regular files in one canonical
/// Downloads root. Moving is a second operation and repeats every safety check
/// against the reviewed inode immediately before asking macOS to move it to
/// Trash. Folders, hidden entries, symlinks, recently changed files, and any
/// path outside the root never become candidates.
actor MacOSDownloadsCleanupService: FreeUpSpaceServicing {
    static let defaultMinimumAge: TimeInterval = 30 * 24 * 60 * 60
    static let defaultMaximumCandidates = 200

    typealias TrashOperation = @Sendable (URL) throws -> Void

    private let root: URL
    private let minimumAge: TimeInterval
    private let maximumCandidates: Int
    private let trashOperation: TrashOperation

    init?(
        root: URL? = FileManager.default.urls(
            for: .downloadsDirectory,
            in: .userDomainMask
        ).first,
        minimumAge: TimeInterval = defaultMinimumAge,
        maximumCandidates: Int = defaultMaximumCandidates,
        trashOperation: @escaping TrashOperation = { url in
            var resultingURL: NSURL?
            try FileManager.default.trashItem(
                at: url,
                resultingItemURL: &resultingURL
            )
        }
    ) {
        guard let root, minimumAge >= 0, maximumCandidates > 0 else { return nil }
        self.root = root.resolvingSymlinksInPath().standardizedFileURL
        self.minimumAge = minimumAge
        self.maximumCandidates = maximumCandidates
        self.trashOperation = trashOperation
    }

    func scan(now: Date) async throws -> [FreeUpSpaceCandidate] {
        guard Self.isDirectory(root) else {
            throw FreeUpSpaceScanError.downloadsUnavailable
        }
        let entries: [URL]
        do {
            entries = try FileManager.default.contentsOfDirectory(
                at: root,
                includingPropertiesForKeys: nil,
                options: [.skipsHiddenFiles, .skipsSubdirectoryDescendants]
            )
        } catch {
            throw Self.scanError(for: error)
        }

        let cutoff = now.addingTimeInterval(-minimumAge)
        var candidates: [FreeUpSpaceCandidate] = []
        candidates.reserveCapacity(min(entries.count, maximumCandidates))
        for entry in entries {
            guard !Task.isCancelled else { throw FreeUpSpaceScanError.cancelled }
            guard let candidate = Self.candidate(
                at: entry,
                root: root,
                modifiedBefore: cutoff
            ) else { continue }
            candidates.append(candidate)
        }
        candidates.sort {
            if $0.byteCount != $1.byteCount { return $0.byteCount > $1.byteCount }
            if $0.modifiedAt != $1.modifiedAt { return $0.modifiedAt < $1.modifiedAt }
            return $0.name.localizedStandardCompare($1.name) == .orderedAscending
        }
        return Array(candidates.prefix(maximumCandidates))
    }

    func moveToTrash(
        _ candidates: [FreeUpSpaceCandidate]
    ) async -> FreeUpSpaceMoveOutcome {
        var outcome = FreeUpSpaceMoveOutcome()
        for candidate in candidates {
            if Task.isCancelled {
                outcome.wasCancelled = true
                break
            }
            guard let current = Self.candidate(
                at: candidate.url,
                root: root,
                modifiedBefore: .distantFuture
            ), current.identity == candidate.identity else {
                outcome.failures.append(FreeUpSpaceMoveFailure(
                    candidate: candidate,
                    reason: .changedSinceReview
                ))
                continue
            }
            do {
                try trashOperation(candidate.url)
            } catch {
                outcome.failures.append(FreeUpSpaceMoveFailure(
                    candidate: candidate,
                    reason: Self.isPermissionError(error)
                        ? .permissionDenied
                        : .moveRejected
                ))
                continue
            }
            if Self.pathIsAbsent(candidate.url) {
                outcome.moved.append(candidate)
            } else {
                outcome.failures.append(FreeUpSpaceMoveFailure(
                    candidate: candidate,
                    reason: .verificationFailed
                ))
            }
        }
        return outcome
    }

    private static func candidate(
        at input: URL,
        root: URL,
        modifiedBefore cutoff: Date
    ) -> FreeUpSpaceCandidate? {
        let url = input.standardizedFileURL
        guard url.deletingLastPathComponent() == root,
              !url.lastPathComponent.hasPrefix("."),
              let identity = fileIdentity(at: url),
              !isTransientDownload(url.lastPathComponent)
        else { return nil }

        let resolved = url.resolvingSymlinksInPath().standardizedFileURL
        guard resolved == url,
              resolved.deletingLastPathComponent() == root
        else { return nil }

        let modifiedAt = Date(
            timeIntervalSince1970: TimeInterval(identity.modifiedSeconds)
                + TimeInterval(identity.modifiedNanoseconds) / 1_000_000_000
        )
        guard modifiedAt <= cutoff else { return nil }
        return FreeUpSpaceCandidate(
            url: url,
            identity: identity,
            modifiedAt: modifiedAt
        )
    }

    private static func fileIdentity(at url: URL) -> FreeUpSpaceFileIdentity? {
        var value = stat()
        guard lstat(url.path, &value) == 0,
              (value.st_mode & S_IFMT) == S_IFREG
        else { return nil }
        return FreeUpSpaceFileIdentity(
            device: UInt64(value.st_dev),
            inode: UInt64(value.st_ino),
            size: max(0, Int64(value.st_size)),
            modifiedSeconds: Int64(value.st_mtimespec.tv_sec),
            modifiedNanoseconds: Int64(value.st_mtimespec.tv_nsec)
        )
    }

    private static func isDirectory(_ url: URL) -> Bool {
        var value = stat()
        guard lstat(url.path, &value) == 0 else { return false }
        return (value.st_mode & S_IFMT) == S_IFDIR
    }

    private static func pathIsAbsent(_ url: URL) -> Bool {
        var value = stat()
        guard lstat(url.path, &value) != 0 else { return false }
        return errno == ENOENT || errno == ENOTDIR
    }

    private static func isTransientDownload(_ name: String) -> Bool {
        let lowered = name.lowercased()
        return lowered.hasSuffix(".crdownload")
            || lowered.hasSuffix(".download")
            || lowered.hasSuffix(".part")
    }

    static func scanError(for error: Error) -> FreeUpSpaceScanError {
        isPermissionError(error) ? .permissionDenied : .enumerationFailed
    }

    private static func isPermissionError(
        _ error: Error,
        remainingUnderlyingErrors: Int = 4
    ) -> Bool {
        let nsError = error as NSError
        if nsError.domain == NSCocoaErrorDomain,
           (nsError.code == CocoaError.fileReadNoPermission.rawValue
               || nsError.code == CocoaError.fileWriteNoPermission.rawValue) {
            return true
        }
        if nsError.domain == NSPOSIXErrorDomain,
           (nsError.code == Int(EACCES) || nsError.code == Int(EPERM)) {
            return true
        }
        if remainingUnderlyingErrors > 0,
           let underlying = nsError.userInfo[NSUnderlyingErrorKey] as? Error {
            return isPermissionError(
                underlying,
                remainingUnderlyingErrors: remainingUnderlyingErrors - 1
            )
        }
        return false
    }
}
