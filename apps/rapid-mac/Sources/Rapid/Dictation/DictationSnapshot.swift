import Darwin
import Foundation

/// Resolve cache paths and inventory artifacts during catalog refresh, never
/// during a keypress. Capture the files behind HF symlinks too: deleting a
/// blob can leave its snapshot directory and symlink names intact.
enum DictationSnapshot {
    struct FileIdentity: Equatable, Sendable {
        let device: Int32
        let inode: UInt64
        let size: Int64
        let modifiedSeconds: Int
        let modifiedNanoseconds: Int
        let changedSeconds: Int
        let changedNanoseconds: Int
    }

    struct Snapshot: Sendable {
        let directory: URL
        let artifacts: [(path: String, identity: FileIdentity)]
        let referencePath: String
        let referenceIdentity: FileIdentity?

        var isOnDisk: Bool {
            DictationSnapshot.fileIdentity(referencePath) == referenceIdentity
                && DictationSnapshot.isDirectory(directory) && artifacts.allSatisfy {
                DictationSnapshot.fileIdentity($0.path) == $0.identity
            }
        }
    }

    static func cachedSnapshots(for entry: ModelEntry, cacheRoot: URL?) -> [Snapshot] {
        guard entry.cached, let repo = entry.hfRepo, let cacheRoot,
              let directory = HFCacheByteMonitor.cacheDirectoryURL(hubCacheRoot: cacheRoot, hfPath: repo)
        else { return [] }
        let snapshots = directory.appendingPathComponent("snapshots", isDirectory: true)
        let reference = directory.appendingPathComponent("refs/main")
        let identity = fileIdentity(reference.path)
        if FileManager.default.fileExists(atPath: reference.path) {
            guard identity != nil,
                  let revision = try? String(contentsOf: reference, encoding: .utf8)
                    .trimmingCharacters(in: .whitespacesAndNewlines),
                  !revision.isEmpty, revision != ".", revision != "..", !revision.contains("/")
            else { return [] }
            // Another revision must not hide deletion of the selected main
            // snapshot, or of any artifacts inside it.
            return inventory(snapshots.appendingPathComponent(revision, isDirectory: true),
                             referencePath: reference.path, referenceIdentity: identity).map { [$0] } ?? []
        }
        // Pinned/offline caches may have snapshots without a main ref. Only
        // remember revisions present at catalog refresh, never discover new
        // ones (or trigger downloads) at record time.
        return ((try? FileManager.default.contentsOfDirectory(
            at: snapshots, includingPropertiesForKeys: nil
        )) ?? []).compactMap {
            inventory($0, referencePath: reference.path, referenceIdentity: identity)
        }
    }

    private static func inventory(
        _ directory: URL, referencePath: String, referenceIdentity: FileIdentity?
    ) -> Snapshot? {
        guard isDirectory(directory) else { return nil }
        var failed = false
        guard let entries = FileManager.default.enumerator(
            at: directory, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles],
            errorHandler: { _, _ in failed = true; return false }
        ) else { return nil }
        var artifacts: [(path: String, identity: FileIdentity)] = []
        var hasConfig = false
        var hasWeights = false
        for case let file as URL in entries {
            if isDirectory(file) { continue }
            guard let identity = fileIdentity(file.path) else { return nil }
            artifacts.append((file.path, identity))
            let size = identity.size
            hasConfig = hasConfig || (file.lastPathComponent == "config.json" && size > 0)
            hasWeights = hasWeights || (["safetensors", "npz", "bin", "pt"].contains(file.pathExtension) && size > 0)
        }
        guard !failed, hasConfig, hasWeights else { return nil }
        return Snapshot(directory: directory, artifacts: artifacts,
                        referencePath: referencePath, referenceIdentity: referenceIdentity)
    }

    private static func fileIdentity(_ path: String) -> FileIdentity? {
        // stat follows HF blob symlinks without cached NSURL resource values.
        // Detect replacement, truncation and same-size rewrites without file
        // reads or hashing on the hot path. Apply this to refs and artifacts.
        var info = stat()
        guard stat(path, &info) == 0, info.st_mode & S_IFMT == S_IFREG else { return nil }
        return FileIdentity(device: info.st_dev, inode: info.st_ino, size: info.st_size,
                                 modifiedSeconds: info.st_mtimespec.tv_sec,
                                 modifiedNanoseconds: info.st_mtimespec.tv_nsec,
                                 changedSeconds: info.st_ctimespec.tv_sec,
                                 changedNanoseconds: info.st_ctimespec.tv_nsec)
    }

    static func isDirectory(_ url: URL) -> Bool {
        var directory: ObjCBool = false
        return FileManager.default.fileExists(atPath: url.path, isDirectory: &directory) && directory.boolValue
    }
}
