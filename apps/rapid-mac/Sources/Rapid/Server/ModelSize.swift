import Foundation

/// DownloadProgress.formatBytes is the single byte formatter. Legacy strings
/// are displayed verbatim because reinterpreting formatted text is ambiguous.
enum ModelSize: Hashable, Sendable, ExpressibleByStringLiteral {
    case bytes(Int64)
    case legacyText(String)

    init(stringLiteral value: String) { self = .legacyText(value) }

    static func legacy(_ raw: String?) -> ModelSize? {
        guard let text = raw?.trimmingCharacters(in: .whitespacesAndNewlines), !text.isEmpty else { return nil }
        return .legacyText(text)
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
