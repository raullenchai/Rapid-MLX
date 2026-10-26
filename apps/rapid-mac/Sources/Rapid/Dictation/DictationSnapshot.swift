import Foundation

/// Resolve cache paths during catalog refresh, never during a keypress.
enum DictationSnapshot {
    static func cachedSnapshots(for entry: ModelEntry, cacheRoot: URL?) -> [URL] {
        guard entry.cached, let repo = entry.hfRepo, let cacheRoot,
              let directory = HFCacheByteMonitor.cacheDirectoryURL(hubCacheRoot: cacheRoot, hfPath: repo)
        else { return [] }
        let snapshots = directory.appendingPathComponent("snapshots", isDirectory: true)
        if let revision = try? String(contentsOf: directory.appendingPathComponent("refs/main"), encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines),
           !revision.isEmpty, revision != ".", revision != "..", !revision.contains("/") {
            // Keep the path even if already missing: another revision must
            // not hide deletion of the selected main snapshot.
            return [snapshots.appendingPathComponent(revision, isDirectory: true)]
        }
        // Pinned/offline caches may have snapshots without a main ref. Only
        // remember revisions present at catalog refresh, never discover new
        // ones (or trigger downloads) at record time.
        return ((try? FileManager.default.contentsOfDirectory(
            at: snapshots, includingPropertiesForKeys: nil
        )) ?? []).filter(isDirectory)
    }

    static func isDirectory(_ url: URL) -> Bool {
        var directory: ObjCBool = false
        return FileManager.default.fileExists(atPath: url.path, isDirectory: &directory) && directory.boolValue
    }
}
