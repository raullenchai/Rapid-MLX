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

/// One resident auxiliary role drawn from the typed 507 capacity envelope
/// (#2305). Desktop consumes the server-declared fields as-is; it never derives
/// a role from a model alias or re-estimates memory. Decoding is defensive so a
/// brand-new/unknown field cannot crash the surface reading the conflict.
struct ResidentRoleStatus: Codable, Sendable, Equatable, Identifiable {
    let role: String
    let modelID: String?
    let reservedBytes: UInt64?
    let state: String?

    enum CodingKeys: String, CodingKey {
        case role
        case modelID = "model_id"
        case reservedBytes = "reserved_bytes"
        case state
    }

    var id: String { "\(role):\(modelID ?? "")" }

    /// User-readable reservation size (GiB), or nil when the server reported
    /// none for this role. Desktop shows only what the engine declares.
    var reservedGib: Double? {
        guard let reservedBytes else { return nil }
        return Double(reservedBytes) / Double(UInt64(1) << 30)
    }
}

/// The typed `insufficient_capacity_error` role-capacity conflict (#2305).
///
/// Presents the conflicting runtime roles, the requested role, the server's
/// memory quantities, and ONLY the `recovery_actions` the server declared valid.
/// Unknown/new envelope fields and unrecognized recovery-action strings are
/// ignored (fail safe) rather than surfaced as buttons or crashing the decode.
struct ResidentRoleConflict: Sendable, Equatable {
    let message: String
    let requestedRole: String?
    let requestedBytes: UInt64?
    let limitBytes: UInt64?
    let usedBytes: UInt64?
    let reason: String?
    let residentRoles: [ResidentRoleStatus]
    let recoveryActions: [String]

    /// User-readable sizes in GiB (nil when the server reported none).
    var requestedGib: Double? {
        guard let requestedBytes else { return nil }
        return Double(requestedBytes) / Double(UInt64(1) << 30)
    }
    var limitGib: Double? {
        guard let limitBytes else { return nil }
        return Double(limitBytes) / Double(UInt64(1) << 30)
    }
    var usedGib: Double? {
        guard let usedBytes else { return nil }
        return Double(usedBytes) / Double(UInt64(1) << 30)
    }

    /// Recovery actions the server declared valid for this conflict. Desktop
    /// only ever offers these; an unknown server value is ignored (never a
    /// button). Order follows the server contract's declaration order.
    var scopedRecoveryActions: [ResidentRecoveryAction] {
        recoveryActions.compactMap(ResidentRecoveryAction.init(rawValue:))
    }
}

/// The closed set of server-declared recovery actions Desktop knows how to
/// perform. An action string outside this set is never presented as a button.
enum ResidentRecoveryAction: String, Sendable, Hashable, CaseIterable {
    case selectSmallerSpeechInput = "select_smaller_speech_input"
    case stopSpeechOutput = "stop_speech_output"
    case unloadAssistant = "unload_assistant"

    var title: String {
        switch self {
        case .selectSmallerSpeechInput: return "Choose a Smaller Speech Input Model"
        case .stopSpeechOutput: return "Stop Speech Output"
        case .unloadAssistant: return "Unload the Conversation Model"
        }
    }

    var hint: String {
        switch self {
        case .selectSmallerSpeechInput:
            return "Pick a smaller speech-to-text model that fits the current memory budget."
        case .stopSpeechOutput:
            return "Release the spoken-response engine so the requested model can fit."
        case .unloadAssistant:
            return "Free the conversation model so this voice workflow can proceed. You can reload it later."
        }
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

/// A snapshot state that means the model is genuinely serving requests.
///
/// The shared residency lifecycle reports the fetch-completion states
/// ``"resident"`` (serving) and the non-serving lifecycle states ``"evicting"``
/// (being freed), ``"retiring"`` (being replaced/rolled back), and ``"failed"``
/// (admission or runtime failure). Only ``"resident"`` is serving; everything
/// else — including any unknown/future state — fails closed to "not serving" so
/// a brand-new lifecycle state can never be mislabelled as available.
func isServingState(_ state: String) -> Bool {
    state == "resident"
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
        models.contains { $0.matches(alias) && isServingState($0.state) }
    }

    func containsResidentAudioLane(modelPath: String) -> Bool {
        audioLanes.contains { $0.matches(modelPath: modelPath) }
    }

    /// Pick the resident text model that can host chat-only subsystems such
    /// as MCP. The process-owning alias may be an audio model after a user
    /// visits Speech, even while a text engine is resident in that process.
    func preferredTextAlias(fallback: String?) -> String? {
        let textModels = models.filter {
            $0.modality == "text" && isServingState($0.state)
        }
        guard let model = textModels.first(where: { $0.primary }) ?? textModels.first else {
            return fallback
        }
        return model.displayName(preferredAlias: fallback)
    }

    /// The engine's own modality for whatever is serving `alias`, or nil when
    /// this snapshot has never heard of it — an old sidecar that reports no
    /// residency, or the window before the first poll lands.
    ///
    /// This is the authoritative lane signal. ``ModelBrandStyle/modelType(forAlias:)``
    /// is a second, name-based guess that already decides whether image parts
    /// go on the wire, but it classifies whole families (`qwen3.5-`, `gemma3-`)
    /// as vision — accurate enough for "might accept an image", far too broad
    /// for "is this the serialised one-at-a-time lane".
    func modality(for alias: String) -> String? {
        models.first { $0.matches(alias) && isServingState($0.state) }?.modality
    }

    /// Requests the engine currently has in flight on `alias`. On the
    /// serialised `--mllm` lane a non-zero value means anything we send now
    /// queues behind real work.
    func activeRequests(for alias: String) -> Int? {
        models.first { $0.matches(alias) && isServingState($0.state) }?.activeRequests
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
            $0.matches(currentAlias) && isServingState($0.state)
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

/// User preference for the advisory active-request switch guard. The missing
/// key means enabled so upgrades keep the safe interactive behavior, while
/// unattended automation can explicitly opt out in Settings.
enum ModelSwitchConfirmationPreference {
    static let storageKey = "rapid.server.confirm_active_request_switch.v1"
    static let defaultValue = true

    static func isEnabled(in defaults: UserDefaults = .standard) -> Bool {
        defaults.object(forKey: storageKey) as? Bool ?? defaultValue
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
    /// A typed role-capacity conflict (#2305/#2306): the model was not admitted
    /// because the requested role cannot coexist with the resident roles.
    /// ``ResidentRoleConflict`` carries the message, structured resident roles,
    /// and server-declared recovery actions the UI presents in the conflict
    /// sheet.
    case conflicted(ResidentRoleConflict)
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
    /// The typed role-capacity conflict (#2305/#2306), when this rejection was
    /// an ``insufficient_capacity_error`` carrying the structured role fields.
    /// Nil for a plain rejection (image-extra missing, HTTP failure, etc.).
    var roleConflict: ResidentRoleConflict? = nil
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
            /// The typed `insufficient_capacity_error` body (#2305). All role
            /// capacity fields are siblings of ``message`` at the error level.
            /// Defensive: each field is optional so a missing/unknown shape is
            /// still decodable (Swift already ignores unknown keys).
            struct Error: Decodable {
                let message: String?
                let requestedRole: String?
                let requestedBytes: UInt64?
                let limitBytes: UInt64?
                let usedBytes: UInt64?
                let reason: String?
                let residentRoles: [ResidentRoleStatus]?
                let recoveryActions: [String]?

                enum CodingKeys: String, CodingKey {
                    case message
                    case requestedRole = "requested_role"
                    case requestedBytes = "requested_bytes"
                    case limitBytes = "limit_bytes"
                    case usedBytes = "used_bytes"
                    case reason
                    case residentRoles = "resident_roles"
                    case recoveryActions = "recovery_actions"
                }
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

            /// The structured error nested under ``detail``, when present.
            var structuredError: StructuredDetail.Error? {
                if case .structured(let structured) = self {
                    return structured.error
                }
                return nil
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

            // A typed role-capacity conflict (#2305/#2306) surfaces the
            // structured `resident_roles` + server-declared `recovery_actions`
            // that the conflict sheet needs. Decode it additively: if the
            // envelope lacks a typed conflict (a plain rejection), fall through
            // to the existing `.rejected` path unchanged.
            if let roleConflict = Self.roleConflict(
                in: envelope,
                fallbackMessage: detail ?? ""
            ) {
                return .conflicted(roleConflict)
            }
            return .rejected(detail ?? "The model could not be kept resident (HTTP \(http.statusCode)).")
        } catch {
            return .rejected("The model server could not load another resident model.")
        }
    }

    /// Build a ``ResidentRoleConflict`` from a decoded 507 ``ErrorEnvelope``,
    /// or return nil when the envelope lacks the typed role-capacity fields
    /// (`requested_role` / `resident_roles` / `recovery_actions`). Nil keeps a
    /// plain rejection on the legacy `.rejected` path. Additive-safe: a missing
    /// or unknown role field never prevents the conflict from being surfaced.
    private static func roleConflict(
        in envelope: ErrorEnvelope?,
        fallbackMessage: String
    ) -> ResidentRoleConflict? {
        // Role fields live on `detail.error` (the typed envelope from
        // `admit_role` / `residency.admit_role`) AND on the top-level
        // `error` for the projection-shaped envelope. Check both, preferring
        // whichever actually carries `resident_roles`.
        let candidates = [envelope?.detail?.structuredError, envelope?.error]
            .compactMap { $0 }
        guard let error = candidates.first(where: { $0.requestedRole != nil || $0.reason != nil }) else {
            return nil
        }
        let residentRoles = (error.residentRoles ?? [])
            .filter { !$0.role.isEmpty }
        let recoveryActions = error.recoveryActions ?? []
        // Only surface a conflict when the envelope actually names a role; a
        // legacy projection-only 507 stays a plain rejection.
        guard error.requestedRole != nil || error.reason?.hasPrefix("role_capacity_") == true else {
            return nil
        }
        return ResidentRoleConflict(
            message: error.message ?? fallbackMessage,
            requestedRole: error.requestedRole,
            requestedBytes: error.requestedBytes,
            limitBytes: error.limitBytes,
            usedBytes: error.usedBytes,
            reason: error.reason,
            residentRoles: residentRoles,
            recoveryActions: recoveryActions
        )
    }
}
