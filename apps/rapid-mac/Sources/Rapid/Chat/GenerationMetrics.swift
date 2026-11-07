import Foundation

/// Facts from one successful inference response, never the picker or turn totals.
struct GenerationMetrics: Codable, Equatable, Hashable, Sendable {
    struct Reference: Codable, Equatable, Hashable, Sendable {
        let version: Int
        let status: String
        let repoID: String?
        let method: String
        let assistantSource: String?
        let backend: String
        let maxDraftTokens: Int?
        let observedMaxDraftTokens: Int?
        let configKey: String?
        let completionTokens: Int
        enum CodingKeys: String, CodingKey, CaseIterable {
            case version, status, method, backend
            case repoID = "repo_id", assistantSource = "assistant_source"
            case maxDraftTokens = "max_draft_tokens", observedMaxDraftTokens = "observed_max_draft_tokens"
            case configKey = "config_key", completionTokens = "completion_tokens"
        }
        init(from decoder: Decoder) throws {
            try generationKeys(decoder, allowed: Set(CodingKeys.allCases.map(\.rawValue)))
            let c = try decoder.container(keyedBy: CodingKeys.self)
            version = try c.decodeIfPresent(Int.self, forKey: .version) ?? 1
            status = try c.decode(String.self, forKey: .status)
            repoID = try c.decodeIfPresent(String.self, forKey: .repoID)
            method = try c.decode(String.self, forKey: .method)
            assistantSource = try c.decodeIfPresent(String.self, forKey: .assistantSource)
            backend = try c.decode(String.self, forKey: .backend)
            maxDraftTokens = try c.decodeIfPresent(Int.self, forKey: .maxDraftTokens)
            observedMaxDraftTokens = try c.decodeIfPresent(Int.self, forKey: .observedMaxDraftTokens)
            configKey = try c.decodeIfPresent(String.self, forKey: .configKey)
            completionTokens = try c.decode(Int.self, forKey: .completionTokens)
        }
    }
    struct Host: Codable, Equatable, Hashable, Sendable {
        let chip: String
        let memoryBytes: UInt64
        enum CodingKeys: String, CodingKey { case chip; case memoryBytes = "memory_bytes" }
        init(from decoder: Decoder) throws {
            try generationKeys(decoder, allowed: ["chip", "memory_bytes"])
            let c = try decoder.container(keyedBy: CodingKeys.self)
            chip = try c.decode(String.self, forKey: .chip)
            memoryBytes = try c.decode(UInt64.self, forKey: .memoryBytes)
        }
    }
    let executionReference: Reference
    let executionHost: Host?
    let timeToFirstTokenMS: Double?
    let meanITLMS: Double?
    enum CodingKeys: String, CodingKey {
        case executionReference = "execution_reference", executionHost = "execution_host"
        case timeToFirstTokenMS = "time_to_first_token_ms", meanITLMS = "mean_itl_ms"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        executionReference = try c.decode(Reference.self, forKey: .executionReference)
        executionHost = try c.decodeIfPresent(Host.self, forKey: .executionHost)
        timeToFirstTokenMS = try c.decodeIfPresent(Double.self, forKey: .timeToFirstTokenMS)
        meanITLMS = try c.decodeIfPresent(Double.self, forKey: .meanITLMS)
        let r = executionReference
        let knownRepos = Set(DecodeExpectations.snapshot?.models.values.map(\.repoID) ?? [])
        let expectedKey: String? = r.method == "none" && r.backend == "ordinary" ? "none"
            : r.method == "mtp" && r.backend == "ordinary" && r.assistantSource == "embedded_head" && r.maxDraftTokens != nil
                ? "mtp:embedded_head:k=\(r.maxDraftTokens!):ordinary" : nil
        guard r.version == 1, ["known", "unknown", "mixed"].contains(r.status),
              ["none", "mtp", "unknown", "mixed"].contains(r.method),
              ["ordinary", "unknown", "mixed"].contains(r.backend),
              r.assistantSource == nil || ["embedded_head", "external_model", "unknown"].contains(r.assistantSource!),
              r.repoID == nil || knownRepos.contains(r.repoID!),
              r.completionTokens >= 0,
              [r.maxDraftTokens, r.observedMaxDraftTokens].allSatisfy({ $0 == nil || (1...128).contains($0!) }),
              r.configKey == nil || r.configKey == expectedKey,
              r.method == "mtp" || (r.assistantSource == nil && r.maxDraftTokens == nil && r.observedMaxDraftTokens == nil),
              (r.status == "mixed") == (r.method == "mixed"),
              r.status != "known" || (r.repoID != nil && expectedKey != nil && r.configKey == expectedKey),
              timeToFirstTokenMS == nil || (timeToFirstTokenMS!.isFinite && timeToFirstTokenMS! >= 0),
              meanITLMS == nil || (meanITLMS!.isFinite && meanITLMS! > 0),
              executionHost == nil || (executionHost!.chip.range(of: #"^M[1-9][0-9]*(?: Pro| Max| Ultra)?$"#, options: .regularExpression) != nil && executionHost!.memoryBytes > 0 && executionHost!.memoryBytes <= 1 << 44)
        else { throw DecodingError.dataCorrupted(.init(codingPath: decoder.codingPath, debugDescription: "Invalid generation metrics")) }
    }

    var decodeTokensPerSecond: Double? {
        guard executionReference.completionTokens >= 2, let interval = meanITLMS,
              interval > 0, interval.isFinite else { return nil }
        let value = 1000 / interval
        return value.isFinite ? value : nil
    }

    /// Hardware is the immutable executing host, including when reached by tunnel.
    func cohort(snapshot: DecodeExpectations.Snapshot? = DecodeExpectations.snapshot) -> DecodeExpectations.Cohort? {
        let r = executionReference
        guard decodeTokensPerSecond != nil, timeToFirstTokenMS != nil,
              r.status == "known", let repo = r.repoID, let key = r.configKey,
              let host = executionHost, let snapshot else { return nil }
        let ram = Double(host.memoryBytes) / Double(1 << 30)
        return snapshot.models.sorted(by: { $0.key < $1.key }).lazy
            .filter { $0.value.repoID == repo }.flatMap { $0.value.cohorts }
            .first { $0.chip == host.chip && $0.method == r.method && $0.configKey == key && $0.n >= 3 && $0.ramMinGiB <= ram && ram <= $0.ramMaxGiB }
    }

    var detailText: String {
        var lines: [String] = []
        if let rate = decodeTokensPerSecond {
            lines.append(String(format: String(localized: "Server decode: %.1f tok/s"), rate))
        }
        if let ttft = timeToFirstTokenMS {
            lines.append(String(format: String(localized: "Server time to first token: %.0f ms"), ttft))
        }
        if let host = executionHost {
            lines.append(String(format: String(localized: "Execution host: %@ · %g GB"), host.chip, Double(host.memoryBytes) / Double(1 << 30)))
        }
        if let row = cohort() {
            lines.append(String(format: String(localized: "Measured reference: %.0f–%.0f tok/s · n=%d"), row.lower, row.upper, row.n))
            if row.ramMinGiB == row.ramMaxGiB {
                lines.append(String(format: String(localized: "Sampled RAM: %g GB"), row.ramMinGiB))
            } else {
                lines.append(String(format: String(localized: "Sampled RAM: %g–%g GB"), row.ramMinGiB, row.ramMaxGiB))
            }
            lines.append(String(localized: "Reference shows the middle 50% of submitted measurements, not a performance guarantee."))
        } else { lines.append(String(localized: "No matching measured reference for this execution.")) }
        if let repo = executionReference.repoID { lines.append(String(format: String(localized: "Executed model: %@"), repo)) }
        let configuration: String
        if executionReference.status == "mixed" { configuration = String(localized: "Multiple execution methods") }
        else if executionReference.status == "known" && executionReference.method == "none" { configuration = String(localized: "MTP off") }
        else if executionReference.status == "known", executionReference.method == "mtp", let depth = executionReference.maxDraftTokens {
            configuration = String(format: String(localized: "MTP on (K=%d)"), depth)
        } else { configuration = String(localized: "configuration unavailable") }
        lines.append(String(format: String(localized: "Execution configuration: %@"), configuration))
        return lines.joined(separator: "\n")
    }
}

struct SavedGenerationMetrics: Decodable, Equatable, Sendable {
    let turnID: String
    let messageID: String
    let metrics: GenerationMetrics
    enum CodingKeys: String, CodingKey { case turnID = "turn_id", messageID = "message_id", metrics }
}

private struct GenerationKey: CodingKey {
    let stringValue: String
    let intValue: Int? = nil
    init?(stringValue: String) { self.stringValue = stringValue }
    init?(intValue: Int) { return nil }
}
private func generationKeys(_ decoder: Decoder, allowed: Set<String>) throws {
    let c = try decoder.container(keyedBy: GenerationKey.self)
    guard c.allKeys.allSatisfy({ allowed.contains($0.stringValue) }) else {
        throw DecodingError.dataCorrupted(.init(codingPath: decoder.codingPath, debugDescription: "Unsupported generation facts"))
    }
}
