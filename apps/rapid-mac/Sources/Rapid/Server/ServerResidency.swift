import Foundation

struct ResidentModelStatus: Codable, Sendable, Equatable, Identifiable {
    let id: String
    let modelPath: String
    let aliases: [String]
    let modality: String
    let state: String
    let pinned: Bool
    let primary: Bool
    let activeRequests: Int
    let estimatedBytes: UInt64
    let measuredBytes: UInt64?
    let idleSeconds: Double

    enum CodingKeys: String, CodingKey {
        case id
        case modelPath = "model_path"
        case aliases
        case modality
        case state
        case pinned
        case primary
        case activeRequests = "active_requests"
        case estimatedBytes = "estimated_bytes"
        case measuredBytes = "measured_bytes"
        case idleSeconds = "idle_seconds"
    }

    func matches(_ alias: String) -> Bool {
        id == alias || modelPath == alias || aliases.contains(alias)
    }

    func displayName(preferredAlias: String? = nil) -> String {
        if let preferredAlias, matches(preferredAlias) {
            return preferredAlias
        }
        // Startup entries use the resolved HF repo as their canonical id and
        // retain the catalog alias separately. Prefer that short, recognizable
        // name in the sidebar so it matches the chat/image picker.
        return aliases.min { lhs, rhs in
            if lhs.count != rhs.count { return lhs.count < rhs.count }
            return lhs.localizedCaseInsensitiveCompare(rhs) == .orderedAscending
        } ?? id
    }

    /// A lazy engine's load-time process delta may cover only metadata while
    /// its first request materializes the weights. Never present that partial
    /// delta as smaller than the admission reservation.
    var displayBytes: UInt64 { max(estimatedBytes, measuredBytes ?? 0) }
}

struct ModelResidencySnapshot: Codable, Sendable, Equatable {
    let memoryLimitBytes: UInt64
    let memoryUsedBytes: UInt64
    let memoryAvailableBytes: UInt64?
    let idleTTLSeconds: Double
    let loadsTotal: Int
    let evictionsTotal: Int
    let models: [ResidentModelStatus]

    enum CodingKeys: String, CodingKey {
        case memoryLimitBytes = "memory_limit_bytes"
        case memoryUsedBytes = "memory_used_bytes"
        case memoryAvailableBytes = "memory_available_bytes"
        case idleTTLSeconds = "idle_ttl_seconds"
        case loadsTotal = "loads_total"
        case evictionsTotal = "evictions_total"
        case models
    }

    static let empty = ModelResidencySnapshot(
        memoryLimitBytes: 0,
        memoryUsedBytes: 0,
        memoryAvailableBytes: nil,
        idleTTLSeconds: 0,
        loadsTotal: 0,
        evictionsTotal: 0,
        models: []
    )

    func contains(_ alias: String) -> Bool {
        models.contains { $0.matches(alias) && $0.state != "evicting" }
    }
}

enum ResidentModelLoadResult: Sendable, Equatable {
    case loaded(ResidentModelStatus)
    case unsupported
    case rejected(String)
}

enum ResidentModelReplacementGroup: String, Sendable {
    case assistant
}

struct ServerResidencyClient {
    private struct LoadBody: Encodable {
        let model: String
        let model_path: String?
        let estimated_size_gb: Double
        let pin: Bool
        let replace_group: String?
    }

    private struct ErrorEnvelope: Decodable {
        let detail: String?
    }

    var session: URLSession = {
        let config = URLSessionConfiguration.ephemeral
        config.timeoutIntervalForRequest = 30 * 60
        config.timeoutIntervalForResource = 30 * 60
        return URLSession(configuration: config)
    }()

    private func request(path: String, port: Int, bearer: String?) -> URLRequest {
        var request = URLRequest(
            url: URL(string: "http://127.0.0.1:\(port)\(path)")!
        )
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        if let bearer, !bearer.isEmpty {
            request.setValue("Bearer \(bearer)", forHTTPHeaderField: "Authorization")
        }
        return request
    }

    func fetch(port: Int, bearer: String?) async -> ModelResidencySnapshot? {
        let request = request(path: "/v1/models/residency", port: port, bearer: bearer)
        guard let (data, response) = try? await session.data(for: request),
              let http = response as? HTTPURLResponse,
              (200...299).contains(http.statusCode)
        else { return nil }
        return try? JSONDecoder().decode(ModelResidencySnapshot.self, from: data)
    }

    func load(
        alias: String,
        hfPath: String?,
        estimatedSizeGB: Double,
        replaceGroup: ResidentModelReplacementGroup? = nil,
        port: Int,
        bearer: String?
    ) async -> ResidentModelLoadResult {
        var request = request(path: "/v1/models/load", port: port, bearer: bearer)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try? JSONEncoder().encode(
            LoadBody(
                model: alias,
                model_path: hfPath,
                estimated_size_gb: estimatedSizeGB,
                pin: false,
                replace_group: replaceGroup?.rawValue
            )
        )
        do {
            let (data, response) = try await session.data(for: request)
            guard let http = response as? HTTPURLResponse else {
                return .rejected("The model server returned an invalid response.")
            }
            if (200...299).contains(http.statusCode) {
                guard let status = try? JSONDecoder().decode(ResidentModelStatus.self, from: data) else {
                    return .rejected("The model server returned invalid residency data.")
                }
                return .loaded(status)
            }
            if http.statusCode == 404 || http.statusCode == 405 {
                return .unsupported
            }
            let detail = (try? JSONDecoder().decode(ErrorEnvelope.self, from: data))?.detail
            return .rejected(detail ?? "The model could not be kept resident (HTTP \(http.statusCode)).")
        } catch {
            return .rejected("The model server could not load another resident model.")
        }
    }
}
