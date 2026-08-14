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
    var performance: ResidentPerformanceStatus? = nil

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
        case performance
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

struct ResidentPerformanceStatus: Codable, Sendable, Equatable {
    let kvCacheDtype: String?
    let kvCacheTurboquant: String?
    let prefixCacheEnabled: Bool?
    let cacheMemoryMB: Int?

    enum CodingKeys: String, CodingKey {
        case kvCacheDtype = "kv_cache_dtype"
        case kvCacheTurboquant = "kv_cache_turboquant"
        case prefixCacheEnabled = "prefix_cache_enabled"
        case cacheMemoryMB = "cache_memory_mb"
    }

    init(config: ModelPerfConfig) {
        switch config.kvCacheMode {
        case .bf16, .int8, .int4:
            kvCacheDtype = config.kvCacheMode?.rawValue
            kvCacheTurboquant = nil
        case .turboquantV4:
            kvCacheDtype = nil
            kvCacheTurboquant = "v4"
        case .turboquantK8V4:
            kvCacheDtype = nil
            kvCacheTurboquant = "k8v4"
        case nil:
            kvCacheDtype = nil
            kvCacheTurboquant = nil
        }
        prefixCacheEnabled = config.prefixCacheEnabled
        cacheMemoryMB = config.cacheMemoryMB
    }

    func matches(_ config: ModelPerfConfig) -> Bool {
        self == ResidentPerformanceStatus(config: config)
    }
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

    /// Pick the resident text model that can host chat-only subsystems such
    /// as MCP. The process-owning alias may be an audio model after a user
    /// visits Speech, even while a text engine is resident in that process.
    func preferredTextAlias(fallback: String?) -> String? {
        let textModels = models.filter {
            $0.modality == "text" && $0.state != "evicting"
        }
        guard let model = textModels.first(where: { $0.primary }) ?? textModels.first else {
            return fallback
        }
        return model.displayName(preferredAlias: fallback)
    }
}

enum ResidentModelLoadResult: Sendable, Equatable {
    case loaded(ResidentModelStatus)
    case unsupported
    case rejected(String)
}

/// A resident-model load that the engine rejected, kept long enough for the
/// surface that initiated the load to read and present the reason verbatim
/// instead of only writing it to the log pane (#1838). The engine's own
/// `detail` string (e.g. `image generation requires the 'rapid-mlx[image]'
/// Python extra (pip install 'rapid-mlx[image]')`) is specific and actionable,
/// so it is preserved here rather than flattened to a generic "couldn't load".
struct ResidentLoadFailure: Sendable, Equatable {
    let alias: String
    let message: String
}

enum ResidentModelReplacementGroup: String, Sendable {
    case assistant
}

enum ResidentImageMode: String, Sendable, Encodable {
    case generation
    case editing
}

struct ServerResidencyClient {
    private struct LoadBody: Encodable {
        let model: String
        let model_path: String?
        let estimated_size_gb: Double
        let pin: Bool
        let replace_group: String?
        let image_mode: ResidentImageMode?
        let performance: ResidentPerformanceStatus?
        let reload_if_changed: Bool
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
        imageMode: ResidentImageMode? = nil,
        performance: ModelPerfConfig? = nil,
        reloadIfChanged: Bool = false,
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
                replace_group: replaceGroup?.rawValue,
                image_mode: imageMode,
                performance: performance.map(ResidentPerformanceStatus.init),
                reload_if_changed: reloadIfChanged
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
