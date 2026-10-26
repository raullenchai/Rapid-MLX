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

    enum ReferenceIdentity: Equatable, Sendable {
        case missing
        case present(FileIdentity)
        case invalid
    }

    struct Snapshot: Sendable {
        let directory: URL
        let artifacts: [(path: String, identity: FileIdentity)]
        let directories: [(path: String, identity: FileIdentity)]
        let referencePath: String
        let referenceIdentity: ReferenceIdentity

        var isOnDisk: Bool {
            DictationSnapshot.referenceIdentity(at: referencePath) == referenceIdentity
                && directories.allSatisfy {
                    DictationSnapshot.fileIdentity($0.path, kind: S_IFDIR) == $0.identity
                } && artifacts.allSatisfy {
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
        let identity = referenceIdentity(at: reference.path)
        guard identity != .invalid else { return [] }
        if case .present = identity {
            guard let revision = try? String(contentsOf: reference, encoding: .utf8)
                    .trimmingCharacters(in: .whitespacesAndNewlines),
                  !revision.isEmpty, revision != ".", revision != "..", !revision.contains("/")
            else { return [] }
            // Another revision must not hide deletion of the selected main
            // snapshot, or of any artifacts inside it.
            return inventory(snapshots.appendingPathComponent(revision, isDirectory: true),
                             referencePath: reference.path, referenceIdentity: identity, family: entry.audioFamily).map { [$0] } ?? []
        }
        // Pinned/offline caches may have snapshots without a main ref. Only
        // remember revisions present at catalog refresh, never discover new
        // ones (or trigger downloads) at record time.
        return ((try? FileManager.default.contentsOfDirectory(
            at: snapshots, includingPropertiesForKeys: nil
        )) ?? []).compactMap {
            inventory($0, referencePath: reference.path, referenceIdentity: identity, family: entry.audioFamily)
        }
    }

    private static func inventory(
        _ directory: URL, referencePath: String, referenceIdentity: ReferenceIdentity, family: String?
    ) -> Snapshot? {
        guard let rootIdentity = fileIdentity(directory.path, kind: S_IFDIR) else { return nil }
        var failed = false
        guard let entries = FileManager.default.enumerator(
            at: directory, includingPropertiesForKeys: nil, options: [],
            errorHandler: { _, _ in failed = true; return false }
        ) else { return nil }
        var artifacts: [(path: String, identity: FileIdentity)] = []
        var directories = [(path: directory.path, identity: rootIdentity)]
        var hasConfig = false
        var rootWeights: [String] = []
        for case let file as URL in entries {
            if isDirectory(file) {
                guard let identity = fileIdentity(file.path, kind: S_IFDIR) else { return nil }
                directories.append((file.path, identity))
                continue
            }
            guard let identity = fileIdentity(file.path) else { return nil }
            artifacts.append((file.path, identity))
            let size = identity.size
            guard file.deletingLastPathComponent().standardizedFileURL.path == directory.standardizedFileURL.path else { continue }
            let name = file.lastPathComponent
            hasConfig = hasConfig || (name == "config.json" && size > 0)
            // Match the STT layouts used by the engine's download gate.
            // A tokenizer/auxiliary binary or nested config is not a model.
            let weight = family == "whisper"
                ? ["weights.npz", "weights.safetensors"].contains(name)
                : name.hasPrefix("model") && name.hasSuffix(".safetensors")
            if weight {
                guard size > 0 else { return nil }
                rootWeights.append(name)
            }
        }
        guard !failed, hasConfig, !rootWeights.isEmpty,
              family == "whisper" || completeShards(rootWeights, in: directory) else { return nil }
        return Snapshot(directory: directory, artifacts: artifacts, directories: directories,
                        referencePath: referencePath, referenceIdentity: referenceIdentity)
    }

    /// Completeness is checked only during refresh. Later capture stats every
    /// recorded artifact, including the index and each referenced shard.
    private static func completeShards(_ names: [String], in directory: URL) -> Bool {
        var numbered: Set<Int> = []
        var total: Int?
        for name in names {
            let parts = name.replacingOccurrences(of: ".safetensors", with: "").components(separatedBy: "-")
            if parts.count == 4, parts[0] == "model", parts[2] == "of",
               let ordinal = Int(parts[1]), let count = Int(parts[3]) {
                guard ordinal > 0, ordinal <= count, total == nil || total == count else { return false }
                total = count
                numbered.insert(ordinal)
            }
        }
        if let total, numbered.count != total { return false }
        let index = directory.appendingPathComponent("model.safetensors.index.json")
        guard FileManager.default.fileExists(atPath: index.path) else { return true }
        guard let data = try? Data(contentsOf: index),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let weights = object["weight_map"] as? [String: String], !weights.isEmpty else { return false }
        return Set(weights.values).allSatisfy { name in
            let parts = name.components(separatedBy: "/")
            guard parts.allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." }),
                  let file = fileIdentity(directory.appendingPathComponent(name).path) else { return false }
            return file.size > 0
        }
    }

    private static func referenceIdentity(at path: String) -> ReferenceIdentity {
        // A dangling link is a broken selected reference, not an offline
        // cache with no main reference. Other stat errors also fail closed.
        var entry = stat()
        guard lstat(path, &entry) == 0 else { return errno == ENOENT ? .missing : .invalid }
        guard let identity = fileIdentity(path) else { return .invalid }
        return .present(identity)
    }

    private static func fileIdentity(_ path: String, kind: mode_t = S_IFREG) -> FileIdentity? {
        // stat follows HF blob symlinks without cached NSURL resource values.
        // Detect replacement, truncation and same-size rewrites without file
        // reads or hashing on the hot path. Apply this to refs and artifacts.
        var info = stat()
        guard stat(path, &info) == 0, info.st_mode & S_IFMT == kind else { return nil }
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
