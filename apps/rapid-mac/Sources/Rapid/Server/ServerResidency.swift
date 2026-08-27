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
    var replacementProjection: ResidentReplacementProjection? = nil

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
        case replacementProjection = "replacement_projection"
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

/// Engine-authored admission plan for an assistant replacement. Desktop sends
/// the policy but never recreates the engine's role-capacity decision: this
/// response is the authoritative account of what the load kept or released.
struct ResidentReplacementProjection: Codable, Sendable, Equatable {
    struct ModelToFree: Codable, Sendable, Equatable {
        let id: String
        let estimatedBytes: UInt64

        enum CodingKeys: String, CodingKey {
            case id
            case estimatedBytes = "estimated_bytes"
        }
    }

    let strategy: String
    let modelsToFree: [ModelToFree]
    let currentBytes: UInt64
    let requestedBytes: UInt64
    let projectedBytes: UInt64
    let limitBytes: UInt64
    let reason: String

    enum CodingKeys: String, CodingKey {
        case strategy
        case modelsToFree = "models_to_free"
        case currentBytes = "current_bytes"
        case requestedBytes = "requested_bytes"
        case projectedBytes = "projected_bytes"
        case limitBytes = "limit_bytes"
        case reason
    }

    func rejectionMessage(alias: String) -> String? {
        guard reason == "role_capacity_insufficient_after_eviction" else { return nil }
        let gib = Double(UInt64(1) << 30)
        let releasedGB = modelsToFree.reduce(0.0) {
            $0 + Double($1.estimatedBytes)
        } / gib
        let projectedGB = Double(projectedBytes) / gib
        let limitGB = Double(limitBytes) / gib
        let release = releasedGB > 0
            ? "Rapid can release about \(max(1, Int(releasedGB.rounded()))) GB from the current model, but "
            : ""
        return release
            + "\(alias) would still need about \(Int(projectedGB.rounded())) GB "
            + "of the \(Int(limitGB.rounded())) GB model-memory budget."
    }
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

/// One lazily loaded speech engine mounted beside the primary chat model.
/// The server reports the authoritative model path and lifecycle state; the
/// Desktop uses those fields instead of assuming that an audio-capable route
/// means the selected speech weights are still resident.
struct ResidentAudioLaneStatus: Codable, Sendable, Equatable {
    let lane: String
    let model: String?
    let state: String

    func matches(modelPath: String) -> Bool {
        model == modelPath && state == "resident"
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
    let audioLanes: [ResidentAudioLaneStatus]

    enum CodingKeys: String, CodingKey {
        case memoryLimitBytes = "memory_limit_bytes"
        case memoryUsedBytes = "memory_used_bytes"
        case memoryAvailableBytes = "memory_available_bytes"
        case idleTTLSeconds = "idle_ttl_seconds"
        case loadsTotal = "loads_total"
        case evictionsTotal = "evictions_total"
        case models
        case audioLanes = "audio_lanes"
    }

    init(
        memoryLimitBytes: UInt64,
        memoryUsedBytes: UInt64,
        memoryAvailableBytes: UInt64?,
        idleTTLSeconds: Double,
        loadsTotal: Int,
        evictionsTotal: Int,
        models: [ResidentModelStatus],
        audioLanes: [ResidentAudioLaneStatus] = []
    ) {
        self.memoryLimitBytes = memoryLimitBytes
        self.memoryUsedBytes = memoryUsedBytes
        self.memoryAvailableBytes = memoryAvailableBytes
        self.idleTTLSeconds = idleTTLSeconds
        self.loadsTotal = loadsTotal
        self.evictionsTotal = evictionsTotal
        self.models = models
        self.audioLanes = audioLanes
    }

    init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        memoryLimitBytes = try values.decode(UInt64.self, forKey: .memoryLimitBytes)
        memoryUsedBytes = try values.decode(UInt64.self, forKey: .memoryUsedBytes)
        memoryAvailableBytes = try values.decodeIfPresent(UInt64.self, forKey: .memoryAvailableBytes)
        idleTTLSeconds = try values.decode(Double.self, forKey: .idleTTLSeconds)
        loadsTotal = try values.decode(Int.self, forKey: .loadsTotal)
        evictionsTotal = try values.decode(Int.self, forKey: .evictionsTotal)
        models = try values.decode([ResidentModelStatus].self, forKey: .models)
        audioLanes = try values.decodeIfPresent(
            [ResidentAudioLaneStatus].self,
            forKey: .audioLanes
        ) ?? []
    }

    static let empty = ModelResidencySnapshot(
        memoryLimitBytes: 0,
        memoryUsedBytes: 0,
        memoryAvailableBytes: nil,
        idleTTLSeconds: 0,
        loadsTotal: 0,
        evictionsTotal: 0,
        models: [],
        audioLanes: []
    )

    func contains(_ alias: String) -> Bool {
        models.contains { $0.matches(alias) && $0.state != "evicting" }
    }

    func containsResidentAudioLane(modelPath: String) -> Bool {
        audioLanes.contains { $0.matches(modelPath: modelPath) }
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

/// User-visible risk carried by an alias-replacing model activation.
/// Missing residency data deliberately means zero active requests so an older
/// sidecar or failed refresh cannot invent a busy state.
struct ModelSwitchRisk: Equatable, Sendable {
    let currentAlias: String
    let targetAlias: String
    let activeRequests: Int

    static func evaluate(
        currentAlias: String,
        targetAlias: String,
        residency: ModelResidencySnapshot?
    ) -> ModelSwitchRisk? {
        guard currentAlias != targetAlias else { return nil }
        let activeRequests = residency?.models.first {
            $0.matches(currentAlias) && $0.state != "evicting"
        }?.activeRequests ?? 0
        guard activeRequests > 0 else { return nil }
        return ModelSwitchRisk(
            currentAlias: currentAlias,
            targetAlias: targetAlias,
            activeRequests: activeRequests
        )
    }

    var title: String {
        let noun = activeRequests == 1 ? "request" : "requests"
        return "Model \(currentAlias) is serving \(activeRequests) active \(noun). Switch anyway?"
    }
}

/// Result of the Desktop's advisory active-request guard. An explicit user
/// approval must use the existing stop/start path because the resident loader
/// correctly refuses to evict a model while it is serving a request.
enum ModelSwitchDecision: Equatable, Sendable {
    case notNeeded
    case approved
    case cancelled

    var requiresProcessRestart: Bool { self == .approved }

    /// A residency decision applies only to the model it inspected.
    /// `ensureServing` is actor-reentrant while the refresh or dialog is
    /// awaiting, so a different live alias needs its own fresh decision.
    static func requiresRevalidation(
        validatedAlias: String?,
        liveAlias: String
    ) -> Bool {
        validatedAlias != liveAlias
    }

    /// A concurrent switch may finish while this request is waiting on its
    /// prompt. The newly live target is success, never a child to tear down.
    static func requiresStop(liveAlias: String, targetAlias: String) -> Bool {
        liveAlias != targetAlias
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

enum ResidentMemoryPolicy: String, Sendable, Encodable {
    case keepThenCommit = "keep_then_commit"
    case evictFirstIfNeeded = "evict_first_if_needed"
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
        let memory_policy: ResidentMemoryPolicy?
        let image_mode: ResidentImageMode?
        let performance: ResidentPerformanceStatus?
        let reload_if_changed: Bool
    }

    private struct ErrorEnvelope: Decodable {
        struct StructuredDetail: Decodable {
            struct Error: Decodable {
                let message: String?
            }

            let error: Error?
            let replacementProjection: ResidentReplacementProjection?

            enum CodingKeys: String, CodingKey {
                case error
                case replacementProjection = "replacement_projection"
            }
        }

        enum Detail: Decodable {
            case message(String)
            case structured(StructuredDetail)

            init(from decoder: Decoder) throws {
                let container = try decoder.singleValueContainer()
                if let message = try? container.decode(String.self) {
                    self = .message(message)
                } else {
                    self = .structured(try container.decode(StructuredDetail.self))
                }
            }
        }

        let detail: Detail?
        let error: StructuredDetail.Error?
        let replacementProjection: ResidentReplacementProjection?

        enum CodingKeys: String, CodingKey {
            case detail
            case error
            case replacementProjection = "replacement_projection"
        }
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
        memoryPolicy: ResidentMemoryPolicy? = nil,
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
                memory_policy: memoryPolicy,
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
            let envelope = try? JSONDecoder().decode(ErrorEnvelope.self, from: data)
            let nestedDetail: String? = switch envelope?.detail {
            case .message(let message): message
            case .structured(let structured):
                structured.replacementProjection?.rejectionMessage(alias: alias)
                    ?? structured.error?.message
            case nil: nil
            }
            let detail = envelope?.replacementProjection?.rejectionMessage(alias: alias)
                ?? envelope?.error?.message
                ?? nestedDetail
            return .rejected(detail ?? "The model could not be kept resident (HTTP \(http.statusCode)).")
        } catch {
            return .rejected("The model server could not load another resident model.")
        }
    }
}
