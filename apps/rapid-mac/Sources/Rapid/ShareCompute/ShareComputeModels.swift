import Foundation

struct ShareComputeModel: Identifiable, Hashable, Sendable {
    let catalogID: String
    let alias: String
    let title: String
    let detail: String

    var id: String { catalogID }

    static let supported: [ShareComputeModel] = [
        .init(
            catalogID: "qwen3.8-27b",
            alias: "qwen3.8-27b-4bit",
            title: "Qwen3.8 27B · 4-bit",
            detail: "Balanced capacity and memory use"
        ),
        .init(
            catalogID: "qwen3.6-35b",
            alias: "qwen3.6-35b",
            title: "Qwen3.6 35B",
            detail: "Higher-capacity pool model"
        ),
        .init(
            catalogID: "nemotron-3.5-lightning",
            alias: "nemotron-3.5-lightning-30b-4bit",
            title: "Nemotron 3.5 Lightning 30B · 4-bit",
            detail: "Fast reasoning-focused model"
        ),
    ]

    static func sanitizedWorker(_ value: String) -> String {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "._-"))
        let scalars = trimmed.unicodeScalars.filter { allowed.contains($0) }.prefix(64)
        let cleaned = String(String.UnicodeScalarView(scalars))
        return cleaned.isEmpty ? "node" : cleaned
    }
}

struct ShareComputeStatusSnapshot: Decodable, Equatable, Sendable {
    let schemaVersion: Int
    let session: String
    let phase: String
    let catalogID: String?
    let alias: String?
    let worker: String?
    let nodeID: String?
    let payoutAccount: String?
    let heartbeatInterval: Double?
    let inflight: Int?
    let connectedAt: Double?
    let message: String?
    let updatedAt: Double

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case session, phase
        case catalogID = "catalog_id"
        case alias, worker
        case nodeID = "node_id"
        case payoutAccount = "payout_account"
        case heartbeatInterval = "heartbeat_interval_s"
        case inflight
        case connectedAt = "connected_at"
        case message
        case updatedAt = "updated_at"
    }
}

struct ShareComputeRegistrationSnapshot: Decodable, Equatable, Sendable {
    let schemaVersion: Int
    let model: String
    let alias: String
    let worker: String

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case model, alias, worker
    }
}
