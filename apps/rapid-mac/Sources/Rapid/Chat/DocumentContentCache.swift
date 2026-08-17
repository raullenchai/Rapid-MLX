import Foundation

/// Store of fully extracted document text so the ``read_document`` tool can page
/// through a long attachment WITHOUT the whole extract ever entering the prompt.
///
/// This is the counterpart to ``BrowseContentCache``: same two-tier shape, same
/// character-checkpoint pagination, but keyed by the attachment's UUID and with
/// no TTL. A browsed page goes stale because the web changes underneath it; the
/// text extracted from a file the user attached does not, and expiring it would
/// break follow-up questions about a conversation the user reopens next week —
/// exactly the persistence promise ``ChatFileAttachment`` makes.
///
/// ## Why the full text lives here and not on ``ChatFileAttachment``
///
/// ``ChatMessage`` encodes its `fileAttachments` into the conversation history
/// file, and the sidebar loads every conversation at launch. Persisting whole
/// documents inline would make startup cost scale with the total size of every
/// document ever attached. Keeping only a preview on the message and the full
/// text in this separately-swept, LRU-bounded store keeps history files small
/// and lets old extracts age out without touching the transcript.
///
/// Thread-safe via locks — the tool runs on the main actor today, but the cache
/// makes no actor assumptions, matching ``BrowseContentCache``.
final class DocumentContentCache: @unchecked Sendable {
    /// One entry in a document's structural map: a heading, how deeply it
    /// nests, and where it starts.
    ///
    /// Sourced from the PDF's own bookmarks when it has them — a real book
    /// carries an accurate, hand-authored tree (289 entries in the sample,
    /// readable in 0.03s without touching page text), which beats any heading
    /// heuristic run over extracted prose.
    struct OutlineNode: Codable, Sendable, Equatable {
        let title: String
        /// Nesting level, 0 for a top-level heading.
        let depth: Int
        /// 1-based page, when the source knows one.
        let page: Int?
        /// Character offset into the entry's text, when the source knows one.
        /// Lets the model jump from a heading straight to a sequential read.
        let offset: Int?

        init(title: String, depth: Int, page: Int? = nil, offset: Int? = nil) {
            self.title = title
            self.depth = depth
            self.page = page
            self.offset = offset
        }
    }

    struct Entry: Codable, Sendable {
        /// Sparse character-offset checkpoints used by pagination. Building
        /// these once avoids walking from `startIndex` again for every page.
        /// They are derived from `text`, so they are never persisted.
        private let characterCheckpoints: [String.Index]
        private let characterCount: Int
        private static let checkpointStride = 4_096

        let filename: String
        /// Full extracted text; `read_document` returns slices of this.
        let text: String
        /// Page count for PDFs, nil otherwise. Informational — lets the tool
        /// tell the model how much document is behind the character count.
        let pageCount: Int?
        /// The document's structural map, when it has one. Empty for formats
        /// and files that carry no headings.
        let outline: [OutlineNode]

        init(
            filename: String,
            text: String,
            pageCount: Int? = nil,
            outline: [OutlineNode] = []
        ) {
            self.filename = filename
            self.text = text
            self.pageCount = pageCount
            self.outline = outline
            (characterCheckpoints, characterCount) = Self.makeCharacterCheckpoints(text)
        }

        private enum CodingKeys: String, CodingKey {
            case filename, text, pageCount, outline
        }

        init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            filename = try container.decode(String.self, forKey: .filename)
            text = try container.decode(String.self, forKey: .text)
            pageCount = try container.decodeIfPresent(Int.self, forKey: .pageCount)
            // Absent in entries written before outline support; an empty map
            // degrades to "this document has no outline", which is correct.
            outline = try container.decodeIfPresent([OutlineNode].self, forKey: .outline) ?? []
            (characterCheckpoints, characterCount) = Self.makeCharacterCheckpoints(text)
        }

        func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode(filename, forKey: .filename)
            try container.encode(text, forKey: .text)
            try container.encodeIfPresent(pageCount, forKey: .pageCount)
            if !outline.isEmpty { try container.encode(outline, forKey: .outline) }
        }

        var count: Int { characterCount }

        func index(atCharacterOffset rawOffset: Int) -> String.Index {
            let offset = min(max(0, rawOffset), characterCount)
            let checkpointNumber = offset / Self.checkpointStride
            let checkpointOffset = checkpointNumber * Self.checkpointStride
            return text.index(
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

    static let shared = DocumentContentCache()

    private let memoryLock = NSLock()
    private let diskLock = NSLock()
    private var store: [String: Entry] = [:]
    private var order: [String] = []          // LRU: front = oldest
    private var totalBytes = 0

    private let maxEntries: Int
    private let maxBytes: Int

    /// Directory backing the persistent tier, or nil to run memory-only (used by
    /// unit tests that don't want to touch disk).
    private let diskDirectory: URL?
    private let maxDiskEntries: Int
    private let maxDiskBytes: Int

    /// Production initialiser — persists to
    /// ``Application Support/Rapid/document-cache``.
    ///
    /// The caps are larger than ``BrowseContentCache``'s because a document is
    /// bigger than a web page by construction and the whole feature exists to
    /// handle files that don't fit in a prompt.
    init() {
        self.maxEntries = 16
        self.maxBytes = 64 * 1024 * 1024
        self.diskDirectory = Self.defaultDiskDirectory()
        self.maxDiskEntries = 64
        self.maxDiskBytes = 512 * 1024 * 1024
        sweepDiskOnInitialization()
    }

    /// Test / custom initialiser. Pass ``diskDirectory: nil`` for a memory-only
    /// cache, or a per-test temp directory to exercise the disk tier without
    /// touching the user's real Application Support tree.
    init(
        maxEntries: Int = 16,
        maxBytes: Int = 64 * 1024 * 1024,
        diskDirectory: URL?,
        maxDiskEntries: Int = 64,
        maxDiskBytes: Int = 512 * 1024 * 1024
    ) {
        self.maxEntries = maxEntries
        self.maxBytes = maxBytes
        self.diskDirectory = diskDirectory
        self.maxDiskEntries = maxDiskEntries
        self.maxDiskBytes = maxDiskBytes
        sweepDiskOnInitialization()
    }

    /// ``Application Support/Rapid/document-cache`` — honours the ``$HOME``
    /// override the same way every other on-disk store in the app does
    /// (#419/#420).
    private static func defaultDiskDirectory() -> URL {
        ApplicationSupportLocator.applicationSupportRoot()
            .appendingPathComponent("document-cache", isDirectory: true)
    }

    /// The key is an attachment UUID string, which is already fixed-length and
    /// filesystem-safe, so unlike ``BrowseContentCache`` no hashing is needed.
    /// It is still validated on the sweep path (``isDiskCacheFileName``) so the
    /// sweep can never delete a file this cache did not write.
    private static func diskFileName(for key: String) -> String {
        "\(key).json"
    }

    static func key(for id: UUID) -> String { id.uuidString }

    // MARK: - Pending extraction
    //
    // Attaching a large PDF extracts only the pages the preview needs; the
    // rest is finished on a background task (see ``ChatFileAttachment``). That
    // leaves a window where ``read_document`` can be called for a document
    // whose full text has not landed yet. Returning "not found" there would be
    // a lie — the document IS attached — so a caller can instead wait for the
    // in-flight work to complete.

    /// Documents whose full extraction is still running.
    private var pending: Set<String> = []
    /// Signalled whenever a pending extraction finishes or fails.
    private let pendingSignal = NSCondition()

    /// Mark `id` as having a full extraction in flight. Balanced by
    /// ``finishPending(_:)``, which MUST be called on every path — including
    /// failure — or a waiter would block until its timeout.
    func beginPending(_ id: UUID) {
        let k = Self.key(for: id)
        pendingSignal.lock()
        pending.insert(k)
        pendingSignal.unlock()
    }

    /// Clear the in-flight mark and wake any waiter.
    func finishPending(_ id: UUID) {
        let k = Self.key(for: id)
        pendingSignal.lock()
        pending.remove(k)
        pendingSignal.broadcast()
        pendingSignal.unlock()
    }

    private func isPending(_ id: UUID) -> Bool {
        let k = Self.key(for: id)
        pendingSignal.lock(); defer { pendingSignal.unlock() }
        return pending.contains(k)
    }

    /// Wait for an in-flight extraction of `id`, giving up only once it stops
    /// making progress for `stallTimeout` seconds.
    ///
    /// A fixed total timeout cannot work here: text extraction finishes in
    /// milliseconds while recognizing a 529-page scan takes ~9 minutes, and
    /// any single number is either too short for the scan or too long to wait
    /// on a task that died. Progress is the honest signal — a run that is
    /// still publishing pages deserves more time, one that has gone quiet does
    /// not.
    private func waitForPending(_ id: UUID, stallTimeout: TimeInterval) {
        let k = Self.key(for: id)
        pendingSignal.lock(); defer { pendingSignal.unlock() }
        while pending.contains(k) {
            let generationBefore = progressGeneration
            let deadline = Date().addingTimeInterval(stallTimeout)
            pendingSignal.wait(until: deadline)
            guard pending.contains(k) else { return }
            // Woken by a timeout with no progress recorded: the task is stuck
            // or gone. Return what is cached rather than blocking forever.
            if progressGeneration == generationBefore, Date() >= deadline { return }
        }
    }

    /// Bumped by ``reportProgress(_:)`` so a waiter can distinguish "still
    /// working" from "stalled" without knowing anything about the work.
    private var progressGeneration: UInt64 = 0

    /// Signal that a long extraction is still advancing. Cheap enough to call
    /// per page.
    func reportProgress(_ id: UUID) {
        pendingSignal.lock()
        progressGeneration &+= 1
        pendingSignal.broadcast()
        pendingSignal.unlock()
    }

    /// Like ``get(_:)`` but waits for an in-flight full extraction first, so a
    /// tool call that arrives while the background pass is still running sees
    /// the complete document instead of a partial one.
    ///
    /// On timeout the caller still gets whatever has been published, which is
    /// partial but real.
    func getAwaitingCompletion(_ id: UUID, stallTimeout: TimeInterval = 30) -> Entry? {
        if isPending(id) { waitForPending(id, stallTimeout: stallTimeout) }
        return get(id)
    }

    func get(_ id: UUID) -> Entry? {
        let k = Self.key(for: id)
        memoryLock.lock()
        if let e = store[k] {
            touch(k)
            memoryLock.unlock()
            return e
        }
        memoryLock.unlock()

        // Memory miss: fall back to the persistent tier. A hit here is a
        // document attached in a PREVIOUS launch (or evicted from the hot
        // tier). Disk I/O deliberately happens without the memory lock so a
        // slow miss cannot block unrelated hot entries.
        guard let e = loadFromDisk(k) else { return nil }

        memoryLock.lock(); defer { memoryLock.unlock() }
        // A concurrent put may have installed a value while the disk read was
        // in flight. Keep that rather than replacing it with the persisted copy.
        if let current = store[k] {
            touch(k)
            return current
        }
        insertLocked(k, entry: e)
        return e
    }

    func put(_ id: UUID, entry: Entry) {
        let k = Self.key(for: id)
        memoryLock.lock()
        insertLocked(k, entry: entry)
        let shouldPersist = diskDirectory != nil
        memoryLock.unlock()
        // Disk I/O happens OUTSIDE the memory lock: writing the document + the
        // LRU sweep touch the filesystem, which we don't want to serialise the
        // (fast) in-memory paging path behind.
        //
        // Unlike ``BrowseContentCache`` there is no write-ordering token here:
        // a document is immutable once extracted, so two puts for the same
        // UUID carry identical bytes and a "stale" write cannot lose data.
        if shouldPersist {
            writeToDisk(k, entry: entry)
        }
    }

    // MARK: - Memory tier (lock held by callers)

    private func insertLocked(_ k: String, entry: Entry) {
        let cost = entry.text.utf8.count
        if let old = store[k] {
            totalBytes -= old.text.utf8.count
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

    private func evictIfNeeded() {
        while (order.count > maxEntries || totalBytes > maxBytes), let oldest = order.first {
            order.removeFirst()
            if let e = store.removeValue(forKey: oldest) {
                totalBytes -= e.text.utf8.count
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
        // Validate the file size BEFORE reading it into memory: a corrupted or
        // locally-modified entry can be arbitrarily large, and a loaded entry is
        // promoted into the memory tier. Anything over the disk cap is treated
        // as a miss and deleted so it isn't re-checked forever.
        let fileSize = (try? url.resourceValues(forKeys: [.fileSizeKey]))?.fileSize
        guard let fileSize, fileSize <= maxDiskBytes else {
            try? FileManager.default.removeItem(at: url)
            return nil
        }
        guard let data = try? Data(contentsOf: url),
              let entry = try? JSONDecoder().decode(Entry.self, from: data) else {
            // Corrupt / unreadable / schema-drifted file: drop it so every
            // future lookup doesn't reread and redecode the same dead bytes.
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

    private func writeToDisk(_ key: String, entry: Entry) {
        guard let dir = diskDirectory else { return }
        diskLock.lock(); defer { diskLock.unlock() }

        let fm = FileManager.default
        // Best-effort persistence: a failure just means this document won't
        // survive a relaunch — the memory tier already served the current
        // session — so we never surface the error to the tool caller.
        guard ensureDiskDirectory(dir, fileManager: fm) else { return }
        guard let data = try? JSONEncoder().encode(entry) else { return }
        let url = dir.appendingPathComponent(Self.diskFileName(for: key), isDirectory: false)
        // ``.atomic`` so a torn write never surfaces as a half-decoded document
        // on the next read (loadFromDisk would just miss).
        do {
            try data.write(to: url, options: [.atomic])
        } catch {
            return
        }
        do {
            try fm.setAttributes([.posixPermissions: 0o600], ofItemAtPath: url.path)
        } catch {
            // The write landed but could not be restricted. Discard it rather
            // than leave the user's document world-readable.
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

    /// LRU sweep of the persistent tier keyed on file modification time: delete
    /// the oldest ``.json`` files until both caps are satisfied. Conservative —
    /// a directory-listing failure is a no-op, and only our own
    /// ``<uuid>.json`` files are ever considered for deletion.
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

    /// True iff ``name`` is a ``<uuid>.json`` file — the exact shape
    /// ``diskFileName(for:)`` produces. The disk sweep refuses to delete
    /// anything else in the directory.
    static func isDiskCacheFileName(_ name: String) -> Bool {
        guard name.hasSuffix(".json") else { return false }
        let stem = String(name.dropLast(".json".count))
        return UUID(uuidString: stem) != nil
    }
}
