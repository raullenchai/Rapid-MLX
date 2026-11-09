import Foundation

/// Allocator counters from /v1/status, in decimal GB (the engine divides by 1e9).
/// These overlap process memory and are not a weights/prefix-cache breakdown.
struct MetalMemorySnapshot: Decodable, Sendable, Equatable {
    let active: Double?
    let cache: Double?
    let peak: Double?
    var prefixCache: PrefixCacheSnapshot? = nil
    var numRunning: Int? = nil
    var numWaiting: Int? = nil

    enum CodingKeys: String, CodingKey {
        case active = "active_memory_gb"
        case cache = "cache_memory_gb"
        case peak = "peak_memory_gb"
    }

    static func label(_ value: Double?) -> String {
        guard let value, value.isFinite, value >= 0 else { return "—" }
        return String(format: "%.2f GB", locale: Locale(identifier: "en_US_POSIX"), value)
    }
}

/// Reusable prompt cache budget, independent of Metal allocator counters.
struct PrefixCacheSnapshot: Decodable, Sendable, Equatable {
    let currentBytes: Double?
    let maxBytes: Double?
    let enabled: Bool?

    enum CodingKeys: String, CodingKey {
        case currentBytes = "current_memory_bytes"
        case maxBytes = "max_memory_bytes"
        case enabled
    }

    var utilization: Double? {
        guard enabled != false, let current = MemoryOverviewFormat.valid(currentBytes),
              let cap = MemoryOverviewFormat.valid(maxBytes), cap > 0 else { return nil }
        return MemoryOverviewFormat.valid(current / cap)
    }

    static func label(_ snapshot: Self?) -> String {
        guard let snapshot, snapshot.utilization != nil else { return "—" }
        return "\(MetalMemorySnapshot.label(snapshot.currentBytes.map { $0 / 1e9 })) / \(MetalMemorySnapshot.label(snapshot.maxBytes.map { $0 / 1e9 }))"
    }
}

struct MetalMemoryClient {
    private struct Status: Decodable {
        let metal: MetalMemorySnapshot?
        let cache: PrefixCacheSnapshot?
        let numRunning: Int?
        let numWaiting: Int?

        enum CodingKeys: String, CodingKey {
            case metal, cache
            case numRunning = "num_running"
            case numWaiting = "num_waiting"
        }
    }

    /// Decode all operator signals from the same authenticated status sample.
    static func decodeStatus(_ data: Data) throws -> MetalMemorySnapshot? {
        let status = try JSONDecoder().decode(Status.self, from: data)
        guard status.metal != nil || status.cache != nil || status.numRunning != nil || status.numWaiting != nil
        else { return nil }
        var snapshot = status.metal ?? MetalMemorySnapshot(active: nil, cache: nil, peak: nil)
        snapshot.prefixCache = status.cache
        snapshot.numRunning = status.numRunning
        snapshot.numWaiting = status.numWaiting
        return snapshot
    }

    var session: URLSession = {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = 3
        configuration.timeoutIntervalForResource = 3
        return URLSession(configuration: configuration)
    }()

    static func request(port: Int, bearer: String) -> URLRequest? {
        guard (1...65535).contains(port), !bearer.isEmpty else { return nil }
        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/v1/status")!)
        request.setValue("Bearer \(bearer)", forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        request.applyRapidClientHeader()
        return request
    }

    func fetch(port: Int, bearer: String) async -> MetalMemorySnapshot? {
        guard let request = Self.request(port: port, bearer: bearer),
              let (data, response) = try? await session.data(for: request),
              let http = response as? HTTPURLResponse,
              (200...299).contains(http.statusCode)
        else { return nil }
        return try? Self.decodeStatus(data)
    }
}

/// The same isolation boundary used by the view's authenticated polling task.
/// Keep this in memory only: it contains the credential, never presentation data.
struct MemoryOverviewSession: Equatable, Sendable {
    let sessionID: UUID?
    let port: Int
    let bearer: String?
    let model: String?
}

/// Fixed storage and a time bound: at three seconds, 101 samples cover five
/// minutes including both endpoints. Missing readings remain gaps, not zeros.
struct MemoryOverviewHistory: Sendable {
    struct Sample: Sendable {
        let date: Date
        let systemUsedGB: Double?
        let metalActiveGB: Double?
    }

    static let capacity = 101
    static let window: TimeInterval = 300
    private var storage = [Sample?](repeating: nil, count: capacity)
    private var head = 0
    private var count = 0
    private(set) var session: MemoryOverviewSession?

    var samples: [Sample] {
        (0..<count).compactMap { storage[(head + $0) % Self.capacity] }
    }

    mutating func reset(for session: MemoryOverviewSession) {
        storage = [Sample?](repeating: nil, count: Self.capacity)
        head = 0
        count = 0
        self.session = session
    }

    mutating func append(date: Date, system: MemoryProbe.Snapshot?, metal: MetalMemorySnapshot?,
                         session: MemoryOverviewSession) {
        if self.session != session { reset(for: session) }
        // A wall-clock correction must not leave future samples in the window.
        if count > 0, let last = storage[(head + count - 1) % Self.capacity], date < last.date {
            reset(for: session)
        }
        while count > 0, let first = storage[head], date.timeIntervalSince(first.date) > Self.window {
            storage[head] = nil
            head = (head + 1) % Self.capacity
            count -= 1
        }
        let sample = Sample(date: date, systemUsedGB: MemoryOverviewFormat.gigabytes(system?.usedBytes),
                            metalActiveGB: MemoryOverviewFormat.valid(metal?.active))
        if count == Self.capacity {
            storage[head] = sample
            head = (head + 1) % Self.capacity
        } else {
            storage[(head + count) % Self.capacity] = sample
            count += 1
        }
    }
}

enum MemoryOverviewFormat {
    static func requestCount(_ count: Int?) -> String {
        guard let count, count >= 0 else { return "—" }
        return count.formatted()
    }

    static func valid(_ value: Double?) -> Double? {
        guard let value, value.isFinite, value >= 0 else { return nil }
        return value
    }

    static func gigabytes(_ bytes: UInt64?) -> Double? {
        bytes.map { Double($0) / 1e9 }
    }

    static func bytes(_ bytes: UInt64?) -> String {
        MetalMemorySnapshot.label(gigabytes(bytes))
    }

    /// Active allocations only; cache and peak are not added to this share.
    static func engineShare(activeGB: Double?, totalBytes: UInt64?) -> Double? {
        guard let active = valid(activeGB), let totalBytes, totalBytes > 0 else { return nil }
        return valid(active / (Double(totalBytes) / 1e9))
    }

    static func percent(_ ratio: Double?) -> String {
        guard let ratio = valid(ratio), (ratio * 100).isFinite else { return "—" }
        return String(format: "%.1f%%", locale: Locale.current, ratio * 100)
    }
}
