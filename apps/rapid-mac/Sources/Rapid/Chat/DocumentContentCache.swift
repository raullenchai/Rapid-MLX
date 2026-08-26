import Foundation

/// Thread-safe memory/disk cache for document text paged by `read_document`.
/// Explicit removal and TTL expiry are part of its plaintext-retention contract.
final class DocumentContentCache: @unchecked Sendable {
    struct OutlineNode: Codable, Sendable, Equatable {
        let title: String
        let depth: Int
        let page: Int?
        let offset: Int?

        init(title: String, depth: Int, page: Int? = nil, offset: Int? = nil) {
            self.title = title
            self.depth = depth
            self.page = page
            self.offset = offset
        }
    }

    struct Entry: Codable, Sendable {
        /// Derived checkpoints avoid rescanning long string prefixes.
        private let characterCheckpoints: [String.Index]
        private let characterCount: Int
        private static let checkpointStride = 4_096

        let filename: String
        let text: String
        let pageCount: Int?
        let outline: [OutlineNode]
        /// Persisted because in-memory pending state disappears on relaunch.
        let isComplete: Bool
        /// Distinguishes deterministic truncation from interrupted extraction.
        let hitSizeCeiling: Bool

        init(
            filename: String,
            text: String,
            pageCount: Int? = nil,
            outline: [OutlineNode] = [],
            isComplete: Bool = true,
            hitSizeCeiling: Bool = false
        ) {
            self.filename = filename
            self.text = text
            self.pageCount = pageCount
            self.outline = outline
            self.isComplete = isComplete
            self.hitSizeCeiling = hitSizeCeiling
            (characterCheckpoints, characterCount) = Self.makeCharacterCheckpoints(text)
        }

        private enum CodingKeys: String, CodingKey {
            case filename, text, pageCount, outline, isComplete, hitSizeCeiling
        }

        init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            filename = try container.decode(String.self, forKey: .filename)
            text = try container.decode(String.self, forKey: .text)
            pageCount = try container.decodeIfPresent(Int.self, forKey: .pageCount)
            outline = try container.decodeIfPresent([OutlineNode].self, forKey: .outline) ?? []
            isComplete = try container.decodeIfPresent(Bool.self, forKey: .isComplete) ?? true
            hitSizeCeiling = try container.decodeIfPresent(Bool.self, forKey: .hitSizeCeiling) ?? false
            (characterCheckpoints, characterCount) = Self.makeCharacterCheckpoints(text)
        }

        func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode(filename, forKey: .filename)
            try container.encode(text, forKey: .text)
            try container.encodeIfPresent(pageCount, forKey: .pageCount)
            if !outline.isEmpty { try container.encode(outline, forKey: .outline) }
            if !isComplete { try container.encode(false, forKey: .isComplete) }
            if hitSizeCeiling { try container.encode(true, forKey: .hitSizeCeiling) }
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

        /// Inverse of `index(atCharacterOffset:)`, bounded by one checkpoint stride.
        func characterOffset(of index: String.Index) -> Int {
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

    struct AwaitedEntry: Sendable {
        let entry: Entry
        let extractionPending: Bool
    }

    static let shared = DocumentContentCache()

    private let memoryLock = NSLock()
    private let diskLock = NSLock()
    private var store: [String: Entry] = [:]
    private var order: [String] = []          // LRU: front = oldest
    private var totalBytes = 0
    private var lastAccessedAt: [String: Date] = [:]
    /// Survives entry deletion so in-flight publications can be invalidated.
    private var removalGeneration: [String: UInt64] = [:]

    private let maxEntries: Int
    private let maxBytes: Int

    private let diskDirectory: URL?
    private let maxDiskEntries: Int
    private let maxDiskBytes: Int
    private let now: () -> Date
    /// User-visible retention period referenced by `read_document` errors.
    static let retentionDays = 90
    private let diskTTL: TimeInterval

    init() {
        self.maxEntries = 16
        self.maxBytes = 64 * 1024 * 1024
        self.diskDirectory = Self.defaultDiskDirectory()
        self.maxDiskEntries = 64
        self.maxDiskBytes = 512 * 1024 * 1024
        self.diskTTL = TimeInterval(Self.retentionDays) * 24 * 60 * 60
        self.now = { Date() }
        sweepDiskOnInitialization()
    }

    /// Pass nil for a memory-only test cache.
    init(
        maxEntries: Int = 16,
        maxBytes: Int = 64 * 1024 * 1024,
        diskDirectory: URL?,
        maxDiskEntries: Int = 64,
        maxDiskBytes: Int = 512 * 1024 * 1024,
        diskTTL: TimeInterval = TimeInterval(DocumentContentCache.retentionDays) * 24 * 60 * 60,
        now: @escaping () -> Date = { Date() }
    ) {
        self.maxEntries = maxEntries
        self.maxBytes = maxBytes
        self.diskDirectory = diskDirectory
        self.maxDiskEntries = maxDiskEntries
        self.maxDiskBytes = maxDiskBytes
        self.diskTTL = diskTTL
        self.now = now
        sweepDiskOnInitialization()
    }

    private static func defaultDiskDirectory() -> URL {
        ApplicationSupportLocator.applicationSupportRoot()
            .appendingPathComponent("document-cache", isDirectory: true)
    }

    private static func diskFileName(for key: String) -> String {
        "\(key).json"
    }

    static func key(for id: UUID) -> String { id.uuidString }

    // MARK: - Pending extraction

    private var pending: Set<String> = []
    private let pendingSignal = NSCondition()

    /// Must be balanced by `finishPending` on every completion path.
    func beginPending(_ id: UUID) {
        let k = Self.key(for: id)
        pendingSignal.lock()
        pending.insert(k)
        progressGenerations[k] = 0
        pendingSignal.unlock()
    }

    func finishPending(_ id: UUID) {
        let k = Self.key(for: id)
        pendingSignal.lock()
        pending.remove(k)
        progressGenerations.removeValue(forKey: k)
        extractionTasks.removeValue(forKey: k)
        pendingSignal.broadcast()
        pendingSignal.unlock()
    }

    // MARK: - Cancelling an extraction

    private var extractionTasks: [String: Task<Void, Never>] = [:]

    /// Registers only tasks still pending under the condition lock.
    @discardableResult
    func registerExtraction(_ id: UUID, task: Task<Void, Never>) -> Bool {
        let k = Self.key(for: id)
        pendingSignal.lock()
        let registered = pending.contains(k)
        if registered {
            extractionTasks[k] = task
        }
        pendingSignal.unlock()
        return registered
    }

    func cancelExtraction(_ id: UUID) {
        let k = Self.key(for: id)
        pendingSignal.lock()
        let task = extractionTasks.removeValue(forKey: k)
        pendingSignal.unlock()
        task?.cancel()
    }

    func hasRegisteredExtraction(_ id: UUID) -> Bool {
        pendingSignal.lock(); defer { pendingSignal.unlock() }
        return extractionTasks[Self.key(for: id)] != nil
    }

    private func isPending(_ id: UUID) -> Bool {
        let k = Self.key(for: id)
        pendingSignal.lock(); defer { pendingSignal.unlock() }
        return pending.contains(k)
    }

    /// Waits until completion or until progress stalls for `stallTimeout`.
    private func waitForPending(_ id: UUID, stallTimeout: TimeInterval) {
        let k = Self.key(for: id)
        pendingSignal.lock(); defer { pendingSignal.unlock() }
        var observedGeneration = progressGenerations[k] ?? 0
        var deadline = Date().addingTimeInterval(stallTimeout)
        while pending.contains(k) {
            pendingSignal.wait(until: deadline)
            guard pending.contains(k) else { return }
            let currentGeneration = progressGenerations[k] ?? 0
            if currentGeneration != observedGeneration {
                observedGeneration = currentGeneration
                deadline = Date().addingTimeInterval(stallTimeout)
            } else if Date() >= deadline {
                return
            }
        }
    }

    private var progressGenerations: [String: UInt64] = [:]

    func reportProgress(_ id: UUID) {
        let k = Self.key(for: id)
        pendingSignal.lock()
        guard pending.contains(k) else {
            pendingSignal.unlock()
            return
        }
        progressGenerations[k, default: 0] &+= 1
        pendingSignal.broadcast()
        pendingSignal.unlock()
    }

    func getAwaitingCompletion(_ id: UUID, stallTimeout: TimeInterval = 30) -> Entry? {
        getAwaitingCompletionStatus(id, stallTimeout: stallTimeout)?.entry
    }

    func getAwaitingCompletionStatus(
        _ id: UUID,
        stallTimeout: TimeInterval = 30
    ) -> AwaitedEntry? {
        if isPending(id) { waitForPending(id, stallTimeout: stallTimeout) }
        let extractionPending = isPending(id)
        guard let entry = get(id) else { return nil }
        return AwaitedEntry(entry: entry, extractionPending: extractionPending)
    }

    func get(_ id: UUID) -> Entry? {
        let k = Self.key(for: id)
        let accessDate = now()
        let expiry = accessDate.addingTimeInterval(-diskTTL)
        memoryLock.lock()
        if let e = store[k] {
            guard (lastAccessedAt[k] ?? .distantPast) >= expiry else {
                memoryLock.unlock()
                remove(id)
                return nil
            }
            lastAccessedAt[k] = accessDate
            touch(k)
            memoryLock.unlock()
            touchDiskEntry(k, at: accessDate)
            return e
        }
        let generationBefore = removalGeneration[k] ?? 0
        memoryLock.unlock()

        guard let e = loadFromDisk(k, at: accessDate) else { return nil }

        memoryLock.lock(); defer { memoryLock.unlock() }
        if let current = store[k] {
            touch(k)
            return current
        }
        // Do not resurrect an entry removed during disk I/O.
        guard (removalGeneration[k] ?? 0) == generationBefore else { return nil }
        insertLocked(k, entry: e, accessedAt: accessDate)
        return e
    }

    func put(_ id: UUID, entry: Entry) {
        publish(id, entry: entry, ifGenerationIs: generation(for: id))
    }

    /// Capture before slow work; removal invalidates a later publication.
    func generation(for id: UUID) -> UInt64 {
        memoryLock.lock(); defer { memoryLock.unlock() }
        return removalGeneration[Self.key(for: id)] ?? 0
    }

    @discardableResult
    func publish(_ id: UUID, entry: Entry, ifGenerationIs expected: UInt64) -> Bool {
        let k = Self.key(for: id)
        let accessDate = now()
        memoryLock.lock()
        guard (removalGeneration[k] ?? 0) == expected else {
            memoryLock.unlock()
            return false
        }
        insertLocked(k, entry: entry, accessedAt: accessDate)
        let shouldPersist = diskDirectory != nil
        memoryLock.unlock()
        guard shouldPersist else { return true }

        // Recheck while holding diskLock so remove cannot race the disk write.
        diskLock.lock(); defer { diskLock.unlock() }
        memoryLock.lock()
        let stillValid = (removalGeneration[k] ?? 0) == expected
        memoryLock.unlock()
        guard stillValid else { return false }
        writeToDiskLocked(k, entry: entry, accessedAt: accessDate)
        return true
    }

    /// Removes memory, disk and in-flight work without waiting for cancellation.
    func remove(_ id: UUID) {
        let k = Self.key(for: id)
        // Invalidate before deleting either tier.
        memoryLock.lock()
        removalGeneration[k] = (removalGeneration[k] ?? 0) &+ 1
        if let old = store.removeValue(forKey: k) {
            totalBytes -= old.text.utf8.count
            order.removeAll { $0 == k }
        }
        lastAccessedAt.removeValue(forKey: k)
        memoryLock.unlock()

        cancelExtraction(id)

        guard let dir = diskDirectory else { return }
        diskLock.lock(); defer { diskLock.unlock() }
        try? FileManager.default.removeItem(
            at: dir.appendingPathComponent(Self.diskFileName(for: k), isDirectory: false)
        )
    }

    func remove<S: Sequence>(contentsOf ids: S) where S.Element == UUID {
        for id in ids { remove(id) }
    }

    // MARK: - Memory tier (lock held by callers)
    private func insertLocked(_ k: String, entry: Entry, accessedAt: Date) {
        let cost = entry.text.utf8.count
        if let old = store[k] {
            totalBytes -= old.text.utf8.count
            order.removeAll { $0 == k }
        }
        store[k] = entry
        lastAccessedAt[k] = accessedAt
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
            lastAccessedAt.removeValue(forKey: oldest)
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

    private func loadFromDisk(_ key: String, at accessDate: Date) -> Entry? {
        guard let dir = diskDirectory else { return nil }
        diskLock.lock(); defer { diskLock.unlock() }
        let url = dir.appendingPathComponent(Self.diskFileName(for: key), isDirectory: false)
        // Reject oversized or expired files before loading plaintext.
        let values = try? url.resourceValues(forKeys: [.contentModificationDateKey, .fileSizeKey])
        let expiry = accessDate.addingTimeInterval(-diskTTL)
        guard let modified = values?.contentModificationDate,
              modified >= expiry,
              let fileSize = values?.fileSize,
              fileSize <= maxDiskBytes else {
            try? FileManager.default.removeItem(at: url)
            return nil
        }
        guard let data = try? Data(contentsOf: url),
              let entry = try? JSONDecoder().decode(Entry.self, from: data) else {
            try? FileManager.default.removeItem(at: url)
            return nil
        }
        try? FileManager.default.setAttributes(
            [.modificationDate: accessDate, .posixPermissions: 0o600],
            ofItemAtPath: url.path
        )
        return entry
    }

    private func touchDiskEntry(_ key: String, at accessDate: Date) {
        guard let dir = diskDirectory else { return }
        diskLock.lock(); defer { diskLock.unlock() }
        let url = dir.appendingPathComponent(Self.diskFileName(for: key), isDirectory: false)
        try? FileManager.default.setAttributes(
            [.modificationDate: accessDate, .posixPermissions: 0o600],
            ofItemAtPath: url.path
        )
    }

    /// Requires `diskLock`; failures leave the in-memory entry usable.
    private func writeToDiskLocked(_ key: String, entry: Entry, accessedAt: Date) {
        guard let dir = diskDirectory else { return }

        let fm = FileManager.default
        guard ensureDiskDirectory(dir, fileManager: fm) else { return }
        guard let data = try? JSONEncoder().encode(entry) else { return }
        let url = dir.appendingPathComponent(Self.diskFileName(for: key), isDirectory: false)
        do {
            try data.write(to: url, options: [.atomic])
        } catch {
            return
        }
        do {
            try fm.setAttributes(
                [.modificationDate: accessedAt, .posixPermissions: 0o600],
                ofItemAtPath: url.path
            )
        } catch {
            // Never retain document text with permissive file access.
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

    /// Expires cache-owned files, then enforces count and byte caps oldest-first.
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
        let expiry = now().addingTimeInterval(-diskTTL)
        for entry in entries {
            guard Self.isDiskCacheFileName(entry.lastPathComponent) else { continue }
            let values = try? entry.resourceValues(forKeys: [.contentModificationDateKey, .fileSizeKey])
            let modified = values?.contentModificationDate ?? .distantPast
            let size = values?.fileSize ?? 0
            if modified < expiry {
                try? fm.removeItem(at: entry)
                continue
            }
            files.append(DiskFile(url: entry, modified: modified, size: size))
            totalDiskBytes += size
        }
        guard files.count > maxDiskEntries || totalDiskBytes > maxDiskBytes else { return }
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

    /// Keeps sweeping constrained to `<uuid>.json` files created by this cache.
    static func isDiskCacheFileName(_ name: String) -> Bool {
        guard name.hasSuffix(".json") else { return false }
        let stem = String(name.dropLast(".json".count))
        return UUID(uuidString: stem) != nil
    }
}
