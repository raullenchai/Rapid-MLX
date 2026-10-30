import Foundation

/// One formatter for every model size label, so the same model reads the same
/// in the picker, Settings, and download progress.
enum ModelSizeText {
    /// Accept localized separators because size strings can arrive from other producers.
    nonisolated static func bytes(_ raw: String?) -> Int64? {
        guard let raw = raw?.trimmingCharacters(in: .whitespacesAndNewlines), !raw.isEmpty else { return nil }
        let pattern = #"^([0-9]+(?:[.,' \u00A0\u202F][0-9]+)*)\s*(B|KB|MB|GB|TB|KiB|MiB|GiB|TiB)$"#
        guard let regex = try? NSRegularExpression(pattern: pattern, options: .caseInsensitive),
              let match = regex.firstMatch(in: raw, range: NSRange(raw.startIndex..., in: raw)),
              let numberRange = Range(match.range(at: 1), in: raw),
              let unitRange = Range(match.range(at: 2), in: raw) else { return nil }
        let separators = CharacterSet(charactersIn: ".,' \u{00A0}\u{202F}")
        let parts = String(raw[numberRange]).components(separatedBy: separators)
        let hasDecimal = parts.count > 1 && (1...2).contains(parts.last!.count)
        let groups = hasDecimal ? parts.dropLast() : parts[...]
        guard groups.dropFirst().allSatisfy({ $0.count == 3 }) else { return nil }
        let normalized = groups.joined() + (hasDecimal ? "." + parts.last! : "")
        guard let number = Double(normalized) else { return nil }
        let unit = raw[unitRange].uppercased().replacingOccurrences(of: "I", with: "")
        let exponent: Int
        switch unit {
        case "B": exponent = 0
        case "KB": exponent = 1
        case "MB": exponent = 2
        case "GB": exponent = 3
        case "TB": exponent = 4
        default: return nil
        }
        let value = number * pow(1024, Double(exponent))
        guard value.isFinite, value >= 0, value < Double(Int64.max) else { return nil }
        return Int64(value.rounded())
    }

    nonisolated static func display(_ raw: String?) -> String? {
        guard let trimmed = raw?.trimmingCharacters(in: .whitespacesAndNewlines), !trimmed.isEmpty else { return nil }
        guard let count = bytes(trimmed) else { return trimmed }
        return display(bytes: count)
    }

    nonisolated static func display(bytes: Int64) -> String {
        DownloadProgress.formatBytes(bytes)
    }
}
