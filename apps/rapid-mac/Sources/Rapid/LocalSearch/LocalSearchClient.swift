import Foundation

struct LocalSearchFolder: Decodable, Identifiable, Sendable, Equatable {
    enum State: String, Decodable, Sendable {
        case idle, queued, indexing, waitingForModel = "waiting_for_model", missing, error, unknown
        init(from decoder: Decoder) throws {
            self = Self(rawValue: try decoder.singleValueContainer().decode(String.self)) ?? .unknown
        }
        var title: String {
            switch self {
            case .idle: String(localized: "Ready")
            case .queued: String(localized: "Queued")
            case .indexing: String(localized: "Indexing")
            case .waitingForModel: String(localized: "Waiting for search model")
            case .missing: String(localized: "Folder unavailable. Reconnect the drive or add the folder again.")
            case .error: String(localized: "Could not index folder")
            case .unknown: String(localized: "Unknown folder status. Refresh or update the engine.")
            }
        }
    }
    let id: String
    let path: String
    let state: State
    let filesIndexed: Int
    let filesSkipped: Int
    let chunks: Int
    let lastIndexedAt: Double?
    let error: String?
    var filesByKind: LocalSearchKindCounts? = nil
    var skipReasons: [String: Int]? = nil
}

struct LocalSearchStatus: Decodable, Sendable, Equatable {
    struct Model: Decodable, Sendable, Equatable {
        let id: String
        let downloaded: Bool
        let loaded: Bool
        var vision: Bool? = nil
    }
    struct Indexing: Decodable, Sendable, Equatable {
        let active: Bool
        let folderId: String?
        let currentPath: String?
        let filesDone: Int
        let filesTotal: Int
        var pausedReason: String? = nil
    }
    struct Totals: Decodable, Sendable, Equatable {
        let files: Int
        let chunks: Int
        let indexBytes: Int
        var filesByKind: LocalSearchKindCounts? = nil
        var semanticIndexFull: Bool? = nil
        var embeddedChunks: Int? = nil
        var semanticChunkBudget: Int? = nil
        var keywordOnlyChunks: Int? = nil
        var indexedSummary: String {
            filesByKind?.label ?? String(localized: "\(files) files indexed")
        }
    }
    let enabled: Bool
    let model: Model
    let folders: [LocalSearchFolder]
    let indexing: Indexing
    let totals: Totals
    var semanticUnavailableReason: String? = nil
    var needsPolling: Bool {
        indexing.active || folders.contains { $0.state == .queued || $0.state == .indexing }
    }
}

struct LocalSearchResult: Decodable, Identifiable, Sendable, Equatable {
    var id: String {
        let documentID = "\(folderId):\(path):\(page ?? -1):\(line ?? -1):\(snippet)"
        let baseID = kind == .text ? documentID : "\(documentID):\(kind.rawValue):\(timestamp ?? -1)"
        return duplicateOrdinal == 0 ? baseID : "\(baseID):duplicate:\(duplicateOrdinal)"
    }
    var duplicateOrdinal: Int = 0
    let path: String
    let folderId: String
    let title: String
    let snippet: String
    let highlights: [[Int]]
    let score: Double
    let matched: [String]
    let page: Int?
    let line: Int?
    let modifiedAt: Double
    let size: Int
    var kind: LocalSearchKind = .text
    var timestamp: Double? = nil
    var contentIndexed: Bool = true
    var skipReason: String? = nil

}

extension LocalSearchResult {
    enum CodingKeys: String, CodingKey {
        case path, folderId, title, snippet, highlights, score, matched, page, line, modifiedAt, size, kind, timestamp, contentIndexed, skipReason
    }
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        path = try c.decode(String.self, forKey: .path)
        folderId = try c.decode(String.self, forKey: .folderId)
        title = try c.decode(String.self, forKey: .title)
        snippet = try c.decode(String.self, forKey: .snippet)
        highlights = try c.decode([[Int]].self, forKey: .highlights)
        score = try c.decode(Double.self, forKey: .score)
        matched = try c.decode([String].self, forKey: .matched)
        page = try c.decodeIfPresent(Int.self, forKey: .page)
        line = try c.decodeIfPresent(Int.self, forKey: .line)
        modifiedAt = try c.decode(Double.self, forKey: .modifiedAt)
        size = try c.decode(Int.self, forKey: .size)
        kind = try c.decodeIfPresent(LocalSearchKind.self, forKey: .kind) ?? .text
        timestamp = try c.decodeIfPresent(Double.self, forKey: .timestamp)
        contentIndexed = try c.decodeIfPresent(Bool.self, forKey: .contentIndexed) ?? true
        skipReason = try c.decodeIfPresent(String.self, forKey: .skipReason)
    }
}

struct LocalSearchGroup: Decodable, Sendable, Equatable {
    let kind: LocalSearchKind
    let results: [LocalSearchResult]
    let total: Int
}

struct LocalSearchResponse: Decodable, Sendable, Equatable {
    let results: [LocalSearchResult]
    let tookMs: Double
    let mode: String
    var groups: [LocalSearchGroup]? = nil
}

/// Chunk IDs are internal to the engine. Preserve existing identities and
/// disambiguate repeated rows within each decoded result list.
private struct LocalSearchResultList: Decodable {
    let results: [LocalSearchResult]

    init(from decoder: Decoder) throws {
        let decoded = try decoder.singleValueContainer().decode([LocalSearchResult].self)
        var reserved = Set(decoded.map(\.id))
        var occurrences: [String: Int] = [:]
        results = decoded.map { result in
            var result = result
            let baseID = result.id
            var ordinal = occurrences[baseID, default: 0]
            if ordinal > 0 {
                repeat {
                    result.duplicateOrdinal = ordinal
                    ordinal += 1
                } while reserved.contains(result.id)
                reserved.insert(result.id)
            }
            occurrences[baseID] = max(1, ordinal)
            return result
        }
    }
}

extension LocalSearchGroup {
    enum CodingKeys: String, CodingKey { case kind, results, total }
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        kind = try c.decode(LocalSearchKind.self, forKey: .kind)
        results = try c.decode(LocalSearchResultList.self, forKey: .results).results
        total = try c.decode(Int.self, forKey: .total)
    }
}

extension LocalSearchResponse {
    enum CodingKeys: String, CodingKey { case results, tookMs, mode, groups }
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        results = try c.decode(LocalSearchResultList.self, forKey: .results).results
        tookMs = try c.decode(Double.self, forKey: .tookMs)
        mode = try c.decode(String.self, forKey: .mode)
        groups = try c.decodeIfPresent([LocalSearchGroup].self, forKey: .groups)
    }
}

/// Same ephemeral loopback session, per-call port/bearer, and Desktop header
/// conventions as the media clients. No credentials or responses are persisted.
struct LocalSearchClient: Sendable {
    var session: URLSession = ImageClient.sharedSession
    struct Endpoint: Equatable, Sendable {
        let port: Int
        let bearer: String?
    }
    enum Failure: Error, LocalizedError {
        case unsupported
        case http(Int, String?)
        var errorDescription: String? {
            switch self {
            case .unsupported: String(localized: "This engine does not support local search. Update the engine and restart a model.")
            case let .http(code, message): message ?? String(localized: "Search request failed (HTTP \(code)).")
            }
        }
    }
    static func decode<T: Decodable>(_ type: T.Type, from data: Data) throws -> T {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(type, from: data)
    }
    func request(_ route: String, method: String = "GET", body: Data? = nil, endpoint: Endpoint) async throws -> Data {
        var request = URLRequest(url: ImageClient.loopbackURL(port: endpoint.port)
            .appendingPathComponent("v1/local-search").appendingPathComponent(route))
        request.httpMethod = method
        request.httpBody = body
        request.timeoutInterval = 30
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        if body != nil { request.setValue("application/json", forHTTPHeaderField: "Content-Type") }
        request.applyRapidClientHeader()
        if let bearer = endpoint.bearer, !bearer.isEmpty {
            request.setValue("Bearer \(bearer)", forHTTPHeaderField: "Authorization")
        }
        let (data, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse else { throw Failure.http(0, nil) }
        guard (200..<300).contains(http.statusCode) else {
            if http.statusCode == 404, route == "status" { throw Failure.unsupported }
            struct Envelope: Decodable {
                struct Detail: Decodable { let message: String? }
                let error: Detail?
            }
            throw Failure.http(http.statusCode, (try? JSONDecoder().decode(Envelope.self, from: data))?.error?.message)
        }
        return data
    }
    func status(_ endpoint: Endpoint) async throws -> LocalSearchStatus {
        try Self.decode(LocalSearchStatus.self, from: await request("status", endpoint: endpoint))
    }
    func search(_ query: String, kinds: [LocalSearchKind]? = nil, endpoint: Endpoint) async throws -> LocalSearchResponse {
        struct Body: Encodable {
            let query: String
            let limit = 50
            let folder_ids: [String]? = nil
            let kinds: [LocalSearchKind]?
            enum CodingKeys: String, CodingKey { case query, limit, folder_ids, kinds, group_limit }
            func encode(to encoder: Encoder) throws {
                var container = encoder.container(keyedBy: CodingKeys.self)
                try container.encode(query, forKey: .query)
                try container.encode(limit, forKey: .limit)
                try container.encodeNil(forKey: .folder_ids)
                try container.encodeIfPresent(kinds, forKey: .kinds)
                if kinds == nil { try container.encode(8, forKey: .group_limit) }
            }
        }
        return try Self.decode(LocalSearchResponse.self, from: await request("search", method: "POST", body: JSONEncoder().encode(Body(query: query, kinds: kinds)), endpoint: endpoint))
    }
    func add(path: String, endpoint: Endpoint) async throws {
        _ = try await request("folders", method: "POST", body: JSONEncoder().encode(["path": path]), endpoint: endpoint)
    }
    func remove(id: String, endpoint: Endpoint) async throws {
        // IDs are a single URL path component, never a route supplied by the server.
        guard !id.isEmpty, !id.contains("/"), id != ".", id != ".." else { throw Failure.http(0, String(localized: "Invalid folder identifier.")) }
        _ = try await request("folders/\(id)", method: "DELETE", endpoint: endpoint)
    }
    func reindex(id: String?, endpoint: Endpoint) async throws {
        let body = try JSONSerialization.data(withJSONObject: ["folder_id": id.map { $0 as Any } ?? NSNull()])
        _ = try await request("reindex", method: "POST", body: body, endpoint: endpoint)
    }
}

/// Character offsets in the HTTP contract count Unicode scalars, rather than
/// UTF-16 code units. Clip malformed ranges and merge overlaps before rendering.
enum LocalSearchHighlights {
    struct Segment: Equatable { let text: String; let bold: Bool }
    static func segments(_ text: String, ranges: [[Int]]) -> [Segment] {
        let scalars = Array(text.unicodeScalars)
        var marked = Array(repeating: false, count: scalars.count)
        for range in ranges where range.count == 2 {
            let start = max(0, min(scalars.count, range[0]))
            let end = max(0, min(scalars.count, range[1]))
            guard start < end else { continue }
            for offset in start..<end { marked[offset] = true }
        }
        var segments: [Segment] = []
        var start = 0
        while start < scalars.count {
            var end = start + 1
            while end < scalars.count, marked[end] == marked[start] { end += 1 }
            segments.append(Segment(text: String(String.UnicodeScalarView(scalars[start..<end])), bold: marked[start]))
            start = end
        }
        return segments
    }
}
