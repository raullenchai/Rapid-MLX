import Foundation

/// Offline benchmark references. Never treat an MTP-off cohort as the default
/// MTP-on expectation. Sparse data stays unavailable.
enum DecodeExpectations {
    struct Cohort: Decodable, Sendable {
        let chip: String
        let method: String
        let n: Int
        let ramMinGiB: Double
        let ramMaxGiB: Double
        let lower: Double
        let median: Double
        let upper: Double
        enum CodingKeys: String, CodingKey {
            case chip, method, n, lower, median, upper
            case ramMinGiB = "ram_min_gib", ramMaxGiB = "ram_max_gib"
        }
    }
    struct Model: Decodable, Sendable {
        let repoID: String
        let defaultMethod: String
        let minRAMGiB: Double
        let cohorts: [Cohort]
        enum CodingKeys: String, CodingKey {
            case defaultMethod = "default_method", minRAMGiB = "min_ram_gib"
            case repoID = "repo_id", cohorts
        }
    }
    struct Snapshot: Decodable, Sendable {
        let schemaVersion: Int
        let refreshedAt: String
        let models: [String: Model]
        enum CodingKeys: String, CodingKey {
            case schemaVersion = "schema_version", refreshedAt = "refreshed_at"
            case models
        }
    }

    private static func localized(_ key: String, _ arguments: CVarArg...) -> String {
        String(format: NSLocalizedString(key, comment: "Offline decode expectation"), locale: Locale(identifier: "en_US_POSIX"), arguments: arguments)
    }

    static let snapshot: Snapshot? = {
        let source = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("rapid_mlx/decode_expectations.json")
        guard let url = Bundle.main.url(forResource: "decode_expectations", withExtension: "json") ?? (FileManager.default.fileExists(atPath: source.path) ? source : nil),
              let data = try? Data(contentsOf: url),
              let snapshot = try? JSONDecoder().decode(Snapshot.self, from: data),
              snapshot.schemaVersion == 1 else { return nil }
        return snapshot
    }()

    static func summary(alias: String, repoID: String? = nil, hardware: MacHardware = .detect(), compact: Bool = false) -> String {
        summary(alias: alias, chip: hardware.brandString.replacingOccurrences(of: "Apple ", with: ""), ramGiB: hardware.physicalRAMGB, repoID: repoID, snapshot: snapshot, compact: compact)
    }

    static func summary(alias: String, chip: String, ramGiB: Double, repoID: String?, snapshot: Snapshot?, compact: Bool = false) -> String {
        let unavailable = localized("Expected decode range unavailable")
        guard let snapshot, let model = snapshot.models[alias], model.repoID == repoID, ramGiB.isFinite, ramGiB > 0 else { return unavailable }
        if ramGiB < model.minRAMGiB { return localized("Expected decode unavailable: below catalog RAM requirement") }
        let method = localized(model.defaultMethod == "mtp" ? "MTP on" : model.defaultMethod == "none" ? "MTP off" : "configuration unavailable")
        let cohorts = model.cohorts.filter { $0.chip == chip && $0.n >= 3 && $0.ramMinGiB <= ramGiB && ramGiB <= $0.ramMaxGiB }
        func range(_ row: Cohort) -> String { String(format: "%.0f–%.0f", row.lower, row.upper) }
        func scope(_ row: Cohort) -> String { localized("n=%d, %g–%g GB sampled", row.n, row.ramMinGiB, row.ramMaxGiB) }
        if let row = cohorts.first(where: { $0.method == model.defaultMethod }) {
            let feel = localized(row.median >= 30 ? "smooth" : row.median >= 10 ? "usable" : "tight")
            if compact { return localized("%@ tok/s · %@ · %@", range(row), method, feel) }
            return localized("%@ tok/s measured · %@ · %@ (%@)", range(row), method, feel, scope(row))
        }
        var text = localized("%@ · %@", unavailable, method)
        if !compact, let baseline = cohorts.first(where: { $0.method == "none" }) {
            text += localized("; MTP-off reference %@ tok/s (%@)", range(baseline), scope(baseline))
        }
        return text
    }

}
