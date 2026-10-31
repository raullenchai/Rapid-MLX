import Foundation

/// Byte counts use the shared file formatter; parseable legacy units are
/// converted to bytes, while unknown text remains verbatim.
enum ModelSize: Hashable, Sendable, ExpressibleByStringLiteral {
    case bytes(Int64)
    case legacyText(String)

    init(stringLiteral value: String) { self = Self.legacy(value) ?? .legacyText(value) }

    static func legacy(_ raw: String?) -> ModelSize? {
        guard let text = raw?.trimmingCharacters(in: .whitespacesAndNewlines), !text.isEmpty else { return nil }
        return parseBytes(text).map(Self.bytes) ?? .legacyText(text)
    }

    static func parseBytes(_ raw: String?, locale: Locale = .current) -> Int64? {
        guard let raw = raw?.trimmingCharacters(in: .whitespacesAndNewlines), !raw.isEmpty else {
            return nil
        }
        // Strip the trailing " GB" / " GiB" / " MB" / etc. Accept
        // both decimal (GB) and binary (GiB) units — rapid-mlx
        // currently emits decimal but a future format change to
        // binary shouldn't silently zero the footer.
        let lower = raw.lowercased()
        let units: [(String, Double)] = [
            ("tib", 1024 * 1024 * 1024 * 1024),
            ("gib", 1024 * 1024 * 1024),
            ("mib", 1024 * 1024),
            ("kib", 1024),
            ("tb", 1_000_000_000_000),
            ("gb", 1_000_000_000),
            ("mb", 1_000_000),
            ("kb", 1_000),
            ("t", 1_000_000_000_000),
            ("g", 1_000_000_000),
            ("m", 1_000_000),
            ("k", 1_000),
            ("b", 1),
        ]
        for (suffix, multiplier) in units {
            if lower.hasSuffix(suffix) {
                let numPart = lower.dropLast(suffix.count).trimmingCharacters(in: .whitespaces)
                let normalized: String?
                if let decimal = locale.decimalSeparator, decimal != ".", numPart.contains(decimal) {
                    var text = numPart
                    if let grouping = locale.groupingSeparator, !grouping.isEmpty {
                        text = text.replacingOccurrences(of: grouping, with: "")
                    }
                    normalized = text.replacingOccurrences(of: decimal, with: ".")
                } else {
                    normalized = nil
                }
                if let value = Double(numPart) ?? normalized.flatMap(Double.init) {
                    let bytes = value * multiplier
                    guard bytes.isFinite, bytes >= 0, bytes < Double(Int64.max) else { return nil }
                    return Int64(bytes)
                }
                return nil
            }
        }
        // Bare number with no unit — treat as bytes.
        if let bytes = Int64(lower), bytes >= 0 { return bytes }
        return nil
    }


    var byteCount: Int64? {
        if case .bytes(let bytes) = self { return bytes }
        return nil
    }

    var displayText: String {
        switch self {
        case .bytes(let bytes): return DownloadProgress.formatBytes(bytes)
        case .legacyText(let text): return text
        }
    }
}
