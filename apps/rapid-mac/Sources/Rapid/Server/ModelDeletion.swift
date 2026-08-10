import Foundation
import Darwin

/// Deletes one cached Hugging Face model directory from inside the desktop app so users
/// can free disk without dropping to a terminal. Matches the pattern
/// LM Studio and the Ollama desktop app use ("hover row → red x →
/// confirm → delete"). The desktop owns the dangerous filesystem step:
/// resolve alias → HF repo via `rapid-mlx ls`, construct the expected
/// `~/.cache/huggingface/hub/models--...` directory, verify the realpath
/// stays under the canonical HF cache root, then remove only that directory.
///
/// Why a standalone helper:
///   * Pure function, ``nonisolated``: trivial to unit-test without
///     spinning up a real ``rapid-mlx`` binary (inject a stub URL or
///     skip the spawn entirely in parser tests).
///   * Keeps ``ServerManager`` focused on the live child process; a
///     deletion is a one-shot subprocess with no lifecycle.
///   * Subprocess-spawn shape is kept narrow + isolated so the patterns
///     stay consistent with the rest of the binary-driver surface.
enum ModelDeletion {
    /// Result of one cache-delete invocation. ``freed`` carries
    /// the bytes-freed estimate from the verified cache directory so
    /// the UI can show a "Freed 3.1 GB" toast without having to
    /// double-check by re-walking the cache after deletion.
    enum Outcome: Equatable {
        case freed(bytes: Int64?, raw: String)
        case failed(message: String)
    }

    private struct DeletionTarget {
        let repo: String
        let url: URL
        let bytes: Int64
    }

    /// Resolve and remove the verified HF cache directory for ``alias``.
    /// The caller MUST gate this behind its own confirmation alert.
    nonisolated static func deleteCachedModel(
        binaryPath: URL?,
        alias: String,
        knownRepo: String? = nil,
        hubCacheRoot: URL? = nil
    ) async -> Outcome {
        let trimmedAlias = alias.trimmingCharacters(in: .whitespacesAndNewlines)
        guard ModelCatalog.isSafeAlias(trimmedAlias),
              !trimmedAlias.contains("..") else {
            return .failed(message: "That model name isn't valid.")
        }
        guard let tool = binaryPath,
              FileManager.default.isExecutableFile(atPath: tool.path) else {
            return .failed(message: "Rapid isn't fully set up. Please restart Rapid-MLX.")
        }

        let repo: String?
        if let known = ModelCatalog.sanitizedHuggingFaceRepo(knownRepo) {
            repo = known
        } else {
            repo = await cachedRepo(for: trimmedAlias, binary: tool)
        }
        guard let repo else {
            return .failed(message: "Cached model path could not be verified.")
        }
        // Issue #503: when the caller didn't pin an explicit root, prefer
        // the user's chosen models folder (when set AND currently a
        // reachable directory) so deletion targets the SAME directory the
        // engine loads from and the catalog scans. Falls through to the
        // default location when no folder is set or the drive is
        // unplugged. Tests pin behaviour by passing ``hubCacheRoot``
        // explicitly, which bypasses the preference read entirely.
        guard let root = hubCacheRoot
            ?? ModelsFolderPreference.validatedOverrideURL()
            ?? defaultHubCacheRoot() else {
            return .failed(message: "Model storage folder could not be verified.")
        }
        guard let target = deletionTarget(forRepo: repo, hubCacheRoot: root) else {
            return .failed(message: "Cached model path escaped the Hugging Face cache.")
        }

        do {
            try FileManager.default.removeItem(at: target.url)
        } catch {
            return .failed(message: "Delete failed: \(error.localizedDescription)")
        }

        return .freed(
            bytes: target.bytes,
            raw: "Removed \(target.repo) from \(target.url.path)"
        )
    }

    /// Pull a bytes-freed estimate out of the CLI's
    /// ``Removing <repo> (3.1G) ...`` stdout line. Returns ``nil`` if
    /// the line is absent or the size token doesn't parse; the UI
    /// falls back to a plain "Deleted" toast in that case.
    ///
    /// Lifted to its own helper so unit tests can pin the format
    /// drift on the CLI side (size suffix changes, locale-aware
    /// numbers, etc.) without standing up a fake subprocess.
    nonisolated static func parseFreedBytes(stdout: String) -> Int64? {
        let pattern = #"\(([\d.]+)\s*([KMGT]?)i?B?\)"#
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return nil }
        let ns = stdout as NSString
        let range = NSRange(location: 0, length: ns.length)
        guard let match = regex.firstMatch(in: stdout, range: range),
              match.numberOfRanges == 3 else { return nil }
        let numStr = ns.substring(with: match.range(at: 1))
        let unit = ns.substring(with: match.range(at: 2)).uppercased()
        guard let value = Double(numStr) else { return nil }
        let multiplier: Double
        switch unit {
        case "K": multiplier = 1024
        case "M": multiplier = 1024 * 1024
        case "G": multiplier = 1024 * 1024 * 1024
        case "T": multiplier = 1024 * 1024 * 1024 * 1024
        case "":  multiplier = 1
        default:  return nil
        }
        return Int64(value * multiplier)
    }

    private static func cachedRepo(for alias: String, binary: URL) async -> String? {
        // Deliberately NOT routed through ``ModelCatalogCache``: this resolves
        // which directory is about to be deleted. A snapshot that is even
        // slightly stale could name the wrong repo, and the cost of being
        // wrong here is destroying the wrong download. Two subprocesses is a
        // fine price for reading the world as it is at deletion time.
        let entries = await ModelCatalog.load(binary: binary)
        return entries.first { $0.alias == alias && $0.cached }?.hfRepo
    }

    private static func deletionTarget(forRepo repo: String, hubCacheRoot: URL) -> DeletionTarget? {
        guard let dirName = cacheDirectoryName(forRepo: repo) else { return nil }
        let candidate = hubCacheRoot.appendingPathComponent(dirName, isDirectory: true)
        guard let verified = validatedDeletionURL(candidate, hubCacheRoot: hubCacheRoot) else {
            return nil
        }
        return DeletionTarget(
            repo: repo,
            url: verified,
            bytes: directoryByteCount(at: verified)
        )
    }

    private static func cacheDirectoryName(forRepo repo: String) -> String? {
        guard let safe = ModelCatalog.sanitizedHuggingFaceRepo(repo),
              !safe.contains("..") else {
            return nil
        }
        return "models--" + safe.replacingOccurrences(of: "/", with: "--")
    }

    private static func defaultHubCacheRoot(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> URL? {
        if let raw = environment["HF_HUB_CACHE"], !raw.isEmpty {
            return absoluteDirectoryURL(raw)
        }
        if let raw = environment["HF_HOME"], !raw.isEmpty {
            guard let url = absoluteDirectoryURL(raw) else { return nil }
            return url.appendingPathComponent("hub", isDirectory: true)
        }
        guard let home = environment["HOME"],
              let homeURL = absoluteDirectoryURL(home) else {
            return nil
        }
        return homeURL
            .appendingPathComponent(".cache", isDirectory: true)
            .appendingPathComponent("huggingface", isDirectory: true)
            .appendingPathComponent("hub", isDirectory: true)
    }

    private static func absoluteDirectoryURL(_ raw: String) -> URL? {
        let expanded = (raw as NSString).expandingTildeInPath
        guard expanded.hasPrefix("/"),
              !containsTraversalComponent(expanded) else {
            return nil
        }
        return URL(fileURLWithPath: expanded, isDirectory: true)
    }

    private static func validatedDeletionURL(_ candidate: URL, hubCacheRoot: URL) -> URL? {
        guard !containsTraversalComponent(candidate.path),
              !containsTraversalComponent(hubCacheRoot.path) else {
            return nil
        }

        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: candidate.path, isDirectory: &isDir),
              isDir.boolValue,
              ((try? candidate.resourceValues(forKeys: [.isSymbolicLinkKey]).isSymbolicLink) ?? false) == false,
              let rootCanonical = realpathString(hubCacheRoot),
              let candidateCanonical = realpathString(candidate),
              candidateCanonical != rootCanonical else {
            return nil
        }

        let rootPrefix = rootCanonical.hasSuffix("/") ? rootCanonical : rootCanonical + "/"
        guard candidateCanonical.hasPrefix(rootPrefix) else { return nil }
        return candidate
    }

    private static func containsTraversalComponent(_ path: String) -> Bool {
        path.split(separator: "/", omittingEmptySubsequences: false).contains("..")
    }

    private static func realpathString(_ url: URL) -> String? {
        var buffer = [CChar](repeating: 0, count: Int(PATH_MAX))
        return url.withUnsafeFileSystemRepresentation { ptr in
            guard let ptr,
                  realpath(ptr, &buffer) != nil else {
                return nil
            }
            let end = buffer.firstIndex(of: 0) ?? buffer.endIndex
            let bytes = buffer[..<end].map { UInt8(bitPattern: $0) }
            return String(decoding: bytes, as: UTF8.self)
        }
    }

    private static func directoryByteCount(at url: URL) -> Int64 {
        let keys: Set<URLResourceKey> = [
            .isRegularFileKey,
            .isSymbolicLinkKey,
            .fileSizeKey,
            .fileAllocatedSizeKey,
            .totalFileAllocatedSizeKey,
        ]
        var total: Int64 = 0
        let enumerator = FileManager.default.enumerator(
            at: url,
            includingPropertiesForKeys: Array(keys),
            options: [],
            errorHandler: { _, _ in true }
        )
        while let file = enumerator?.nextObject() as? URL {
            guard let values = try? file.resourceValues(forKeys: keys),
                  values.isSymbolicLink != true,
                  values.isRegularFile == true else {
                continue
            }
            let size = values.totalFileAllocatedSize
                ?? values.fileAllocatedSize
                ?? values.fileSize
                ?? 0
            total += Int64(size)
        }
        return total
    }

    static func _testingCacheDirectoryName(forRepo repo: String) -> String? {
        cacheDirectoryName(forRepo: repo)
    }

    static func _testingDefaultHubCacheRoot(environment: [String: String]) -> URL? {
        defaultHubCacheRoot(environment: environment)
    }

    static func _testingValidatedDeletionURL(_ candidate: URL, hubCacheRoot: URL) -> URL? {
        validatedDeletionURL(candidate, hubCacheRoot: hubCacheRoot)
    }
}
