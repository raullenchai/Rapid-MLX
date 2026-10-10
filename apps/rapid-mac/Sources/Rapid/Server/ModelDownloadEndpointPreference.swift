import Foundation

/// Desktop override for the Hugging Face Hub endpoint (issue #4336).
/// Apply before spawning Python: huggingface_hub reads HF_ENDPOINT at import time.
enum ModelDownloadEndpointPreference {
    static let storageKey = "rapid.models.downloadEndpoint"

    /// Empty means inherit the launch environment / normal download defaults.
    /// Reject credentials, queries and fragments: this is a service base URL.
    static func normalized(_ raw: String) -> String? {
        let value = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        if value.isEmpty { return "" }
        guard !value.contains(where: { $0.isWhitespace || $0.isNewline }),
              let url = URLComponents(string: value),
              ["https", "http"].contains(url.scheme?.lowercased() ?? ""),
              let host = url.host, !host.isEmpty,
              url.user == nil, url.password == nil,
              url.query == nil, url.fragment == nil,
              url.port.map({ (1...65535).contains($0) }) ?? true else { return nil }
        var result = value
        while result.hasSuffix("/") { result.removeLast() }
        return result
    }

    static func storedEndpoint(defaults: UserDefaults = .standard) -> String? {
        guard let raw = defaults.string(forKey: storageKey),
              let value = normalized(raw), !value.isEmpty else { return nil }
        return value
    }

    @discardableResult
    static func save(_ raw: String, defaults: UserDefaults = .standard) -> Bool {
        guard let value = normalized(raw) else { return false }
        if value.isEmpty { defaults.removeObject(forKey: storageKey) }
        else { defaults.set(value, forKey: storageKey) }
        return true
    }

    /// Explicit Settings choices win over ambient configuration. HF mirrors
    /// implement the Hub API, unlike the flat-file Rapid model CDN.
    static func apply(_ endpoint: String?, env: inout [String: String]) {
        guard let endpoint, let value = normalized(endpoint), !value.isEmpty else { return }
        env["HF_ENDPOINT"] = value
        env["RAPID_MLX_MODEL_MIRROR"] = ""
    }
}
