import Foundation
import CryptoKit

/// Store of fetched-and-rendered page bodies so the ``browse`` tool can page
/// through a long document (`offset` param) WITHOUT re-fetching — and so a model
/// that asks for page 2 gets the same bytes it saw on page 1.
///
/// Two tiers:
///   * **Memory (hot)** — bounded on entry count and total bytes (LRU eviction)
///     so a session that browses many large pages can't grow the app's memory
///     without limit.
///   * **Disk (persistent)** — every ``put`` also writes the rendered page to
///     ``~/Library/Application Support/Rapid/browse-cache/<sha256(key)>.json``,
///     and a memory miss falls back to disk. This makes a re-read of a URL from
///     a PREVIOUS launch a zero-network cache hit while the entry is fresh — the
///     "runs on your Mac" spine the built-in-tools direction asks for. Disk is
///     independently bounded (entry count + total bytes) by an LRU sweep keyed
///     on file modification time, so the folder can't grow unbounded either.
///
/// Keyed by the exact URL string the model passed (that is what it re-sends for
/// paging), lightly normalised. Thread-safe via a lock — the tool runs on the
/// main actor today, but the cache makes no actor assumptions so it stays
/// correct if a future fetch path moves off it.
final class BrowseContentCache: @unchecked Sendable {
    struct Entry: Codable, Sendable {
        /// Sparse character-offset checkpoints used by pagination. Building
        /// these once avoids walking from `startIndex` again for every page.
        /// They are derived from `markdown`, so they are never persisted.
        private let characterCheckpoints: [String.Index]
        private let characterCount: Int
        private static let checkpointStride = 4_096

        let title: String?
        /// Full rendered Markdown; `browse` returns `offset..<offset+budget`
        /// slices of this.
        let markdown: String
        /// Final URL after redirects (informational; the key is the requested URL).
        let finalURL: String
        /// Wall-clock time when the network response was fetched. Persisting
        /// this separately from file mtime lets disk LRU touches preserve the
        /// original freshness deadline.
        let fetchedAt: Date

        init(title: String?, markdown: String, finalURL: String, fetchedAt: Date = Date()) {
            self.title = title
            self.markdown = markdown
            self.finalURL = finalURL
            self.fetchedAt = fetchedAt
            (characterCheckpoints, characterCount) = Self.makeCharacterCheckpoints(markdown)
        }

        private enum CodingKeys: String, CodingKey {
            case title, markdown, finalURL, fetchedAt
        }

        init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            title = try container.decodeIfPresent(String.self, forKey: .title)
            markdown = try container.decode(String.self, forKey: .markdown)
            finalURL = try container.decode(String.self, forKey: .finalURL)
            // Entries written before TTL support have no trustworthy fetch
            // time. Treat them as expired instead of silently serving an
            // arbitrarily old page after upgrade.
            fetchedAt = try container.decodeIfPresent(Date.self, forKey: .fetchedAt) ?? .distantPast
            (characterCheckpoints, characterCount) = Self.makeCharacterCheckpoints(markdown)
        }

        func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encodeIfPresent(title, forKey: .title)
            try container.encode(markdown, forKey: .markdown)
            try container.encode(finalURL, forKey: .finalURL)
            try container.encode(fetchedAt, forKey: .fetchedAt)
        }

        var count: Int { characterCount }

        func index(atCharacterOffset rawOffset: Int) -> String.Index {
            let offset = min(max(0, rawOffset), characterCount)
            let checkpointNumber = offset / Self.checkpointStride
            let checkpointOffset = checkpointNumber * Self.checkpointStride
            return markdown.index(
                characterCheckpoints[checkpointNumber],
                offsetBy: offset - checkpointOffset
            )
        }

        private static func makeCharacterCheckpoints(
            _ text: String
        ) -> ([String.Index], Int) {
            var checkpoints = [text.startIndex]
            var index = text.startIndex
            var count = 0
            while index < text.endIndex {
                index = text.index(after: index)
                count += 1
                if count.isMultiple(of: checkpointStride) {
                    checkpoints.append(index)
                }
            }
            return (checkpoints, count)
        }
    }

    static let defaultTTL: TimeInterval = 15 * 60
    static let shared = BrowseContentCache()

    private let memoryLock = NSLock()
    private let diskLock = NSLock()
    private var store: [String: Entry] = [:]
    private var order: [String] = []          // LRU: front = oldest
    private var totalBytes = 0
    private var nextDiskWriteID: UInt64 = 0
    private var pendingDiskWrites: [String: UInt64] = [:]

    private let maxEntries: Int
    private let maxBytes: Int
    private let ttl: TimeInterval

    /// Directory backing the persistent tier, or nil to run memory-only (the
    /// legacy behaviour — used by unit tests that don't want to touch disk).
    private let diskDirectory: URL?
    /// Disk-tier caps. Larger than the memory caps: disk is cheap and the whole
    /// point is surviving across launches, so we keep more history on disk than
    /// we hold hot in memory.
    private let maxDiskEntries: Int
    private let maxDiskBytes: Int

    /// Production initialiser — persists fresh entries for 15 minutes to
    /// ``Application Support/Rapid/browse-cache``. The default ``shared``
    /// instance uses this shape so real browsing survives a relaunch.
    init(
        maxEntries: Int = 16,
        maxBytes: Int = 8 * 1024 * 1024,
        ttl: TimeInterval = BrowseContentCache.defaultTTL
    ) {
        self.maxEntries = maxEntries
        self.maxBytes = maxBytes
        self.ttl = ttl
        self.diskDirectory = Self.defaultDiskDirectory()
        self.maxDiskEntries = 128
        self.maxDiskBytes = 64 * 1024 * 1024
        sweepDiskOnInitialization()
    }

    /// Test / custom initialiser. Pass ``diskDirectory: nil`` for a memory-only
    /// cache (the pre-persistence behaviour), or a per-test temp directory to
    /// exercise the disk tier without touching the user's real Application
    /// Support tree.
    init(
        maxEntries: Int = 16,
        maxBytes: Int = 8 * 1024 * 1024,
        diskDirectory: URL?,
        maxDiskEntries: Int = 128,
        maxDiskBytes: Int = 64 * 1024 * 1024,
        ttl: TimeInterval = BrowseContentCache.defaultTTL
    ) {
        self.maxEntries = maxEntries
        self.maxBytes = maxBytes
        self.ttl = ttl
        self.diskDirectory = diskDirectory
        self.maxDiskEntries = maxDiskEntries
        self.maxDiskBytes = maxDiskBytes
        sweepDiskOnInitialization()
    }

    /// ``Application Support/Rapid/browse-cache`` — honours the ``$HOME``
    /// override the same way every other on-disk store in the app does
    /// (#419/#420), so a dogfood instance with an overridden HOME reads/writes
    /// its own cache rather than the real user's.
    private static func defaultDiskDirectory() -> URL {
        ApplicationSupportLocator.applicationSupportRoot()
            .appendingPathComponent("browse-cache", isDirectory: true)
    }

    /// File name a cache key maps to on disk: ``<sha256(key)>.json``. The URL
    /// key can contain slashes, query strings, and arbitrary bytes, so it can't
    /// be a filename directly — hashing gives a fixed-length, filesystem-safe,
    /// collision-resistant name.
    private static func diskFileName(for key: String) -> String {
        let digest = SHA256.hash(data: Data(key.utf8))
        let hex = digest.map { String(format: "%02x", $0) }.joined()
        return "\(hex).json"
    }

    /// Normalise a model-supplied URL to a stable cache key: trim, and lowercase
    /// only the scheme + host (never the path/query — those are case-sensitive).
    static func key(for rawURL: String) -> String {
        let trimmed = rawURL.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let comps = URLComponents(string: trimmed),
              let scheme = comps.scheme, let host = comps.host else {
            return trimmed
        }
        var c = comps
        c.scheme = scheme.lowercased()
        c.host = host.lowercased()
        return c.string ?? trimmed
    }

    func get(_ rawURL: String) -> Entry? {
        let k = Self.key(for: rawURL)
        memoryLock.lock()
        if let e = store[k] {
            if isFresh(e) {
                touch(k)
                memoryLock.unlock()
                return e
            }
            removeLocked(k)
        }
        memoryLock.unlock()

        // Memory miss: fall back to the persistent tier. A hit here is a page
        // fetched on a PREVIOUS launch (or evicted from the hot tier) — reading
        // it costs zero network. Promote it back into memory so subsequent
        // paging calls stay hot. Disk I/O deliberately happens without the
        // memory lock so a slow miss cannot block unrelated hot entries.
        guard let e = loadFromDisk(k) else { return nil }

        memoryLock.lock(); defer { memoryLock.unlock() }
        // A concurrent put may have installed a newer value while the disk read
        // was in flight. Keep and return that value rather than replacing it
        // with the older persisted copy.
        if let current = store[k] {
            touch(k)
            return current
        }
        insertLocked(k, entry: e)
        return e
    }

    func put(_ rawURL: String, entry: Entry) {
        let k = Self.key(for: rawURL)
        memoryLock.lock()
        insertLocked(k, entry: entry)
        let writeID: UInt64?
        if diskDirectory != nil {
            nextDiskWriteID &+= 1
            writeID = nextDiskWriteID
            pendingDiskWrites[k] = writeID
        } else {
            writeID = nil
        }
        memoryLock.unlock()
        // Disk I/O happens OUTSIDE the memory lock: writing a page + the LRU sweep
        // touch the filesystem, which we don't want to serialise the (fast)
        // in-memory paging path behind.
        if let writeID {
            writeToDisk(k, entry: entry, writeID: writeID)
        }
    }

    // MARK: - Memory tier (lock held by callers)

    private func insertLocked(_ k: String, entry: Entry) {
        let cost = entry.markdown.utf8.count
        if let old = store[k] {
            totalBytes -= old.markdown.utf8.count
            order.removeAll { $0 == k }
        }
        store[k] = entry
        order.append(k)
        totalBytes += cost
        evictIfNeeded()
    }

    private func touch(_ k: String) {
        order.removeAll { $0 == k }
        order.append(k)
    }

    private func removeLocked(_ k: String) {
        order.removeAll { $0 == k }
        if let entry = store.removeValue(forKey: k) {
            totalBytes -= entry.markdown.utf8.count
        }
    }

    func expirationDate(for entry: Entry) -> Date {
        entry.fetchedAt.addingTimeInterval(ttl)
    }

    private func isFresh(_ entry: Entry, now: Date = Date()) -> Bool {
        guard ttl > 0 else { return false }
        let age = now.timeIntervalSince(entry.fetchedAt)
        // A small negative age tolerates a routine clock correction. A cache
        // timestamp far in the future is corrupt and must not become immortal.
        return age >= -300 && age < ttl
    }

    private func evictIfNeeded() {
        while (order.count > maxEntries || totalBytes > maxBytes), let oldest = order.first {
            order.removeFirst()
            if let e = store.removeValue(forKey: oldest) {
                totalBytes -= e.markdown.utf8.count
            }
        }
    }

    // MARK: - Disk tier

    private func sweepDiskOnInitialization() {
        guard let dir = diskDirectory else { return }
        diskLock.lock(); defer { diskLock.unlock() }
        if FileManager.default.fileExists(atPath: dir.path) {
            try? FileManager.default.setAttributes(
                [.posixPermissions: 0o700],
                ofItemAtPath: dir.path
            )
        }
        sweepDiskLocked(dir)
    }

    private func loadFromDisk(_ key: String) -> Entry? {
        guard let dir = diskDirectory else { return nil }
        diskLock.lock(); defer { diskLock.unlock() }
        let url = dir.appendingPathComponent(Self.diskFileName(for: key), isDirectory: false)
        // Validate the file size BEFORE reading it into memory. A corrupted or
        // locally-modified entry can be arbitrarily large — the disk LRU sweep
        // only bounds the directory total and never runs on the read path — so
        // without this guard `Data(contentsOf:)` could allocate an unbounded
        // buffer (and hang the main actor). A loaded entry is promoted into the
        // memory tier, so we bound it by the disk-tier byte cap and treat
        // anything larger as a miss, deleting it so it isn't re-checked forever.
        let fileSize = (try? url.resourceValues(forKeys: [.fileSizeKey]))?.fileSize
        guard let fileSize, fileSize <= maxDiskBytes else {
            try? FileManager.default.removeItem(at: url)
            return nil
        }
        guard let data = try? Data(contentsOf: url),
              let entry = try? JSONDecoder().decode(Entry.self, from: data) else {
            // Corrupt / unreadable / schema-drifted file: drop it so every
            // future lookup for this URL doesn't reread and redecode the same
            // dead bytes forever.
            try? FileManager.default.removeItem(at: url)
            return nil
        }
        guard isFresh(entry) else {
            try? FileManager.default.removeItem(at: url)
            return nil
        }
        // A successful disk hit is an access for disk-tier LRU purposes.
        try? FileManager.default.setAttributes(
            [.modificationDate: Date(), .posixPermissions: 0o600],
            ofItemAtPath: url.path
        )
        return entry
    }

    private func writeToDisk(_ key: String, entry: Entry, writeID: UInt64) {
        guard let dir = diskDirectory else { return }
        diskLock.lock(); defer { diskLock.unlock() }

        // Memory updates intentionally do not wait on filesystem work. If a
        // newer put for this key arrived while this call was waiting for the
        // disk lock, drop this stale write so it cannot overwrite newer data.
        guard isCurrentDiskWrite(key: key, writeID: writeID) else { return }
        defer { completeDiskWrite(key: key, writeID: writeID) }

        let fm = FileManager.default
        // Best-effort persistence: a failure to create the directory or write
        // the file just means this page won't survive a relaunch — the memory
        // tier already served the current session, so we never surface the
        // error to the tool caller.
        guard ensureDiskDirectory(dir, fileManager: fm) else { return }
        guard let data = try? JSONEncoder().encode(entry) else { return }
        let url = dir.appendingPathComponent(Self.diskFileName(for: key), isDirectory: false)
        // ``.atomic`` so a torn write never surfaces as a half-decoded page on
        // the next read (loadFromDisk would just miss and re-fetch).
        do {
            try data.write(to: url, options: [.atomic])
        } catch {
            return
        }
        do {
            try fm.setAttributes([.posixPermissions: 0o600], ofItemAtPath: url.path)
        } catch {
            // The write landed but could not be restricted. Discard it rather
            // than leave browsed content world-readable.
            try? fm.removeItem(at: url)
            return
        }
        sweepDiskLocked(dir)
    }

    private func ensureDiskDirectory(_ dir: URL, fileManager fm: FileManager) -> Bool {
        do {
            try fm.createDirectory(
                at: dir,
                withIntermediateDirectories: true,
                attributes: [.posixPermissions: 0o700]
            )
            try fm.setAttributes([.posixPermissions: 0o700], ofItemAtPath: dir.path)
            return true
        } catch {
            return false
        }
    }

    private func isCurrentDiskWrite(key: String, writeID: UInt64) -> Bool {
        memoryLock.lock(); defer { memoryLock.unlock() }
        return pendingDiskWrites[key] == writeID
    }

    private func completeDiskWrite(key: String, writeID: UInt64) {
        memoryLock.lock(); defer { memoryLock.unlock() }
        if pendingDiskWrites[key] == writeID {
            pendingDiskWrites.removeValue(forKey: key)
        }
    }

    /// LRU sweep of the persistent tier keyed on file modification time: delete
    /// the oldest ``.json`` files until both the entry count and total byte caps
    /// are satisfied. Conservative — a directory-listing failure is a no-op, and
    /// only our own ``<64-hex>.json`` files are ever considered for deletion.
    private func sweepDiskLocked(_ dir: URL) {
        let fm = FileManager.default
        guard let entries = try? fm.contentsOfDirectory(
            at: dir,
            includingPropertiesForKeys: [.contentModificationDateKey, .fileSizeKey],
            options: [.skipsHiddenFiles]
        ) else {
            return
        }
        struct DiskFile {
            let url: URL
            let modified: Date
            let size: Int
        }
        var files: [DiskFile] = []
        var totalDiskBytes = 0
        for entry in entries {
            guard Self.isDiskCacheFileName(entry.lastPathComponent) else { continue }
            let values = try? entry.resourceValues(forKeys: [.contentModificationDateKey, .fileSizeKey])
            let modified = values?.contentModificationDate ?? .distantPast
            let size = values?.fileSize ?? 0
            files.append(DiskFile(url: entry, modified: modified, size: size))
            totalDiskBytes += size
        }
        guard files.count > maxDiskEntries || totalDiskBytes > maxDiskBytes else { return }
        // Oldest first — those are the eviction candidates.
        files.sort { $0.modified < $1.modified }
        var count = files.count
        var bytes = totalDiskBytes
        for file in files {
            guard count > maxDiskEntries || bytes > maxDiskBytes else { break }
            if (try? fm.removeItem(at: file.url)) != nil {
                count -= 1
                bytes -= file.size
            }
        }
    }

    /// True iff ``name`` is a ``<64-hex>.json`` file — the exact shape
    /// ``diskFileName(for:)`` produces. The disk sweep refuses to delete
    /// anything else in the directory.
    static func isDiskCacheFileName(_ name: String) -> Bool {
        guard name.hasSuffix(".json") else { return false }
        let hex = name.dropLast(".json".count)
        guard hex.count == 64 else { return false }
        return hex.unicodeScalars.allSatisfy { c in
            (c >= "0" && c <= "9") || (c >= "a" && c <= "f")
        }
    }
}
