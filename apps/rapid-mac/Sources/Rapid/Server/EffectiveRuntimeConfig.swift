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
        case .bool(let value): value ? "On" : "Off"
        case .int(let value): String(value)
        case .double(let value): String(format: "%.2f", value)
        case .string(let value): value
        case .null: "Automatic"
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
        field.split(separator: "_").map { word in
            word.prefix(1).uppercased() + word.dropFirst()
        }.joined(separator: " ")
    }

    var provenanceText: String {
        switch source {
        case "global_default": "Default"
        case "performance_profile": "Model profile"
        case "user_override": "Your setting"
        case "compatibility": "Compatibility fallback"
        case "safety": "Safety fallback"
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
