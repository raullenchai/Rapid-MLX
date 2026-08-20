import Foundation

/// Store of fully extracted document text so the ``read_document`` tool can page
/// through a long attachment WITHOUT the whole extract ever entering the prompt.
///
/// This is the counterpart to ``BrowseContentCache``: same two-tier shape, same
/// character-checkpoint pagination, but keyed by the attachment's UUID and with
/// a far longer life. A browsed page goes stale because the web changes
/// underneath it; the text extracted from a file the user attached does not,
/// and expiring it quickly would break follow-up questions about a conversation
/// the user reopens next week — exactly the persistence promise
/// ``ChatFileAttachment`` makes.
///
/// ## This store holds plaintext, so deletion is part of its contract
///
/// An entry is the COMPLETE text of a document the user chose to hand over: a
/// contract, a medical letter, a payslip. Two things follow, and neither is
/// optional.
///
/// Deleting the user-visible thing must delete the extract. When a user removes
/// an attachment or deletes a conversation, they have deleted the document as
/// far as they can tell; leaving `<uuid>.json` in Application Support until
/// unrelated LRU pressure happens to evict it is a retention the user never
/// agreed to. ``remove(_:)`` is wired to both paths.
///
/// And an extract nobody deletes still expires (``diskTTL``). The size caps are
/// a size policy, not a retention one — a user who attaches four documents a
/// year would keep all of them forever, because nothing ever pushes past the
/// caps.
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

        /// Inverse of ``index(atCharacterOffset:)``: the Character offset of an
        /// index the caller already holds.
        ///
        /// The obvious spelling — ``text.distance(from: text.startIndex, to:)``
        /// — walks the whole prefix, so reporting ten `grep` hits near the end
        /// of a 20,000,000-character extract would walk 200,000,000 Characters
        /// to produce ten integers. Binary-searching the same checkpoints the
        /// forward lookup uses bounds it to a stride's worth of stepping.
        func characterOffset(of index: String.Index) -> Int {
            // Last checkpoint at or before `index`.
            var low = 0
            var high = characterCheckpoints.count - 1
            while low < high {
                let mid = (low + high + 1) / 2
                if characterCheckpoints[mid] <= index { low = mid } else { high = mid - 1 }
            }
            return low * Self.checkpointStride
                + text.distance(from: characterCheckpoints[low], to: index)
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
    /// How long an untouched extract may sit on disk before the sweep deletes
    /// it, regardless of how much room the caps still allow.
    ///
    /// The caps alone are a SIZE policy, not a retention one: a user who
    /// attaches four documents a year keeps the plaintext of all of them
    /// forever, because nothing ever pushes past 64 entries or 512 MB. That is
    /// the wrong default for a store holding the complete text of whatever the
    /// user dropped into a chat — a contract, a medical letter, a payslip.
    ///
    /// Ninety days is long enough that reopening last quarter's conversation
    /// still works and short enough that a document is not retained
    /// indefinitely by accident. Expiry is not data loss: the attachment's
    /// preview is still in the transcript, and ``read_document`` tells the
    /// user to attach the file again.
    private let diskTTL: TimeInterval

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
        self.diskTTL = 90 * 24 * 60 * 60
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
        maxDiskBytes: Int = 512 * 1024 * 1024,
        diskTTL: TimeInterval = 90 * 24 * 60 * 60
    ) {
        self.maxEntries = maxEntries
        self.maxBytes = maxBytes
        self.diskDirectory = diskDirectory
        self.maxDiskEntries = maxDiskEntries
        self.maxDiskBytes = maxDiskBytes
        self.diskTTL = diskTTL
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
        // The handle is dead weight once the task has returned; dropping it
        // here is what keeps this map from growing for the life of the process.
        taskLock.lock()
        extractionTasks[k] = nil
        taskLock.unlock()
    }

    // MARK: - Cancelling an extraction

    /// Handles on the background extraction of each pending document.
    ///
    /// Recognizing a 529-page scan is a ~6-minute job. Without a handle the
    /// only way to stop one was to quit the app: an unstructured
    /// ``Task.detached`` whose result is discarded has no parent to cancel it,
    /// so removing the attachment left Vision and PDFKit chewing through pages
    /// nobody would ever read. ``PDFTextRecognizer`` already checks
    /// ``Task.isCancelled`` between pages — this is what makes that check
    /// reachable.
    private var extractionTasks: [String: Task<Void, Never>] = [:]
    private let taskLock = NSLock()

    /// Hand the cache the task extracting `id` so it can be cancelled later.
    ///
    /// Registration races the task itself: a short document can finish (and
    /// call ``finishPending``) before the caller gets here, and storing the
    /// handle then would leave a completed task in the map forever. Checking
    /// pending under the lock keeps the map to genuinely live work.
    func registerExtraction(_ id: UUID, task: Task<Void, Never>) {
        let k = Self.key(for: id)
        pendingSignal.lock()
        let stillRunning = pending.contains(k)
        pendingSignal.unlock()
        guard stillRunning else { return }
        taskLock.lock()
        extractionTasks[k] = task
        taskLock.unlock()
    }

    /// Stop the background extraction of `id`, if one is running.
    ///
    /// Returns once the task has been ASKED to stop, not once it has stopped:
    /// recognition checks cancellation between pages, so the last page in
    /// flight still finishes. Waiting for that here would block the caller —
    /// the main actor, on an attachment-removal click — for up to a second.
    ///
    /// ``finishPending`` still runs on the cancelled task's own `defer`, so
    /// any ``read_document`` call waiting on this document is released rather
    /// than left to time out.
    func cancelExtraction(_ id: UUID) {
        let k = Self.key(for: id)
        taskLock.lock()
        let task = extractionTasks.removeValue(forKey: k)
        taskLock.unlock()
        task?.cancel()
    }

    /// True while a background extraction handle is registered for `id`.
    /// Exists for tests asserting the lifecycle; production code cancels
    /// unconditionally rather than asking first.
    func hasRegisteredExtraction(_ id: UUID) -> Bool {
        taskLock.lock(); defer { taskLock.unlock() }
        return extractionTasks[Self.key(for: id)] != nil
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

    /// Forget a document completely: the hot copy, the persisted plaintext, and
    /// any in-flight extraction still producing more of it.
    ///
    /// This is the deletion half of the store's contract. Removing an
    /// attachment or deleting a conversation removes the document as far as the
    /// user can tell, so the extract must go with it rather than lingering in
    /// Application Support until unrelated LRU pressure evicts it.
    ///
    /// Cancelling first matters: a background OCR pass that is still running
    /// calls ``put`` when it finishes, which would re-create the file we just
    /// deleted. Ordering the cancel ahead of the delete closes that window.
    ///
    /// Safe to call for an id that was never cached — the common case for a
    /// conversation restored from history, whose extracts aged out long ago.
    func remove(_ id: UUID) {
        cancelExtraction(id)
        let k = Self.key(for: id)
        memoryLock.lock()
        if let old = store.removeValue(forKey: k) {
            totalBytes -= old.text.utf8.count
            order.removeAll { $0 == k }
        }
        memoryLock.unlock()

        guard let dir = diskDirectory else { return }
        diskLock.lock(); defer { diskLock.unlock() }
        try? FileManager.default.removeItem(
            at: dir.appendingPathComponent(Self.diskFileName(for: k), isDirectory: false)
        )
    }

    /// Forget several documents — the shape conversation deletion needs, where
    /// every attachment on every message goes at once.
    func remove<S: Sequence>(contentsOf ids: S) where S.Element == UUID {
        for id in ids { remove(id) }
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

    /// Sweep the persistent tier: delete anything past ``diskTTL``, then evict
    /// oldest-first until both caps are satisfied.
    ///
    /// TTL runs unconditionally, before and independently of the caps. The caps
    /// only fire under pressure, so on their own they let a handful of
    /// documents — the ordinary usage pattern — keep their plaintext on disk
    /// for the life of the install.
    ///
    /// Conservative in the same two ways as before: a directory-listing failure
    /// is a no-op, and only our own ``<uuid>.json`` files are ever considered
    /// for deletion.
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
        let expiry = Date().addingTimeInterval(-diskTTL)
        for entry in entries {
            guard Self.isDiskCacheFileName(entry.lastPathComponent) else { continue }
            let values = try? entry.resourceValues(forKeys: [.contentModificationDateKey, .fileSizeKey])
            let modified = values?.contentModificationDate ?? .distantPast
            let size = values?.fileSize ?? 0
            // Past the TTL: delete now rather than counting it toward caps it
            // would only be evicted under. A file with no readable date has an
            // unknowable age, and `.distantPast` deliberately expires it — an
            // extract we cannot reason about is not one to keep indefinitely.
            if modified < expiry {
                try? fm.removeItem(at: entry)
                continue
            }
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
