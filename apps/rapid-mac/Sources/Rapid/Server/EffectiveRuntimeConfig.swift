import Foundation

enum EffectiveRuntimeValue: Codable, Equatable, Sendable {
    case bool(Bool)
    case int(Int)
    case double(Double)
    case string(String)
    case null

    init(from decoder: Decoder) throws {
        let value = try decoder.singleValueContainer()
        if value.decodeNil() { self = .null }
        else if let decoded = try? value.decode(Bool.self) { self = .bool(decoded) }
        else if let decoded = try? value.decode(Int.self) { self = .int(decoded) }
        else if let decoded = try? value.decode(Double.self) { self = .double(decoded) }
        else { self = .string(try value.decode(String.self)) }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .bool(let value): try container.encode(value)
        case .int(let value): try container.encode(value)
        case .double(let value): try container.encode(value)
        case .string(let value): try container.encode(value)
        case .null: try container.encodeNil()
        }
    }

    var displayText: String {
        switch self {
        case .bool(let value): value ? String(localized: "On") : String(localized: "Off")
        case .int(let value): String(value)
        case .double(let value): String(format: "%.2f", value)
        case .string(let value): value
        case .null: String(localized: "Automatic")
        }
    }
}

struct EffectiveRuntimeTraceEntry: Codable, Equatable, Sendable {
    let value: EffectiveRuntimeValue
    let source: String
    let sourceID: String
    let reasonCode: String
    let action: String

    enum CodingKeys: String, CodingKey {
        case value, source, action
        case sourceID = "source_id"
        case reasonCode = "reason_code"
    }
}

struct EffectiveRuntimeField: Codable, Equatable, Sendable, Identifiable {
    let field: String
    let value: EffectiveRuntimeValue
    let source: String
    let sourceID: String
    let reasonCode: String
    let trace: [EffectiveRuntimeTraceEntry]

    var id: String { field }

    enum CodingKeys: String, CodingKey {
        case field, value, source, trace
        case sourceID = "source_id"
        case reasonCode = "reason_code"
    }

    var displayName: String {
        switch field {
        case "prefill_step_size": String(localized: "Prompt processing batch size")
        case "max_num_seqs": String(localized: "Maximum concurrent requests")
        case "gpu_memory_utilization": String(localized: "GPU memory budget")
        case "enable_prefix_cache": String(localized: "Reuse prompt prefixes")
        case "kv_cache_dtype": String(localized: "Attention cache precision")
        default: String(localized: "Additional engine setting")
        }
    }

    var helpText: String {
        switch field {
        case "prefill_step_size": String(localized: "Tokens processed together while reading a prompt. Larger batches can use more memory.")
        case "max_num_seqs": String(localized: "The most requests the model can process at once.")
        case "gpu_memory_utilization": String(localized: "The share of GPU memory available to the engine.")
        case "enable_prefix_cache": String(localized: "Reuses earlier work when requests begin with the same text.")
        case "kv_cache_dtype": String(localized: "The precision used to store attention history. Lower precision saves memory.")
        default: String(localized: "An engine setting reported by this model.")
        }
    }

    var provenanceText: String {
        switch source {
        case "global_default": String(localized: "Default")
        case "performance_profile": String(localized: "Model profile")
        case "user_override": String(localized: "Your setting")
        case "compatibility": String(localized: "Compatibility fallback")
        case "safety": String(localized: "Safety fallback")
        default: sourceID
        }
    }
}

struct EffectiveRuntimeConfigSnapshot: Codable, Equatable, Sendable {
    let model: String
    let schemaVersion: Int
    let fields: [EffectiveRuntimeField]

    enum CodingKeys: String, CodingKey {
        case model
        case schemaVersion = "schema_version"
        case fields
    }

    func belongs(to alias: String) -> Bool {
        model.caseInsensitiveCompare(alias) == .orderedSame
    }

    var hasSupportedSchema: Bool { schemaVersion == 1 }
}

struct EffectiveRuntimeConfigClient: Sendable {
    static let requestTimeout: TimeInterval = 5

    func fetch(
        port: Int,
        bearer: String?,
        session: URLSession = .shared
    ) async -> EffectiveRuntimeConfigSnapshot? {
        guard let url = URL(string: "http://127.0.0.1:\(port)/v1/runtime/config")
        else { return nil }
        var request = URLRequest(url: url)
        request.timeoutInterval = Self.requestTimeout
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        request.applyRapidClientHeader()
        if let bearer, !bearer.isEmpty {
            request.setValue("Bearer \(bearer)", forHTTPHeaderField: "Authorization")
        }
        guard let (data, response) = try? await session.data(for: request),
              let http = response as? HTTPURLResponse,
              (200...299).contains(http.statusCode)
        else { return nil }
        guard let snapshot = try? JSONDecoder().decode(
            EffectiveRuntimeConfigSnapshot.self,
            from: data
        ), snapshot.hasSupportedSchema else { return nil }
        return snapshot
    }
}
