import Foundation

/// ``X-Rapid-Client`` — how a Rapid-owned client identifies itself to a
/// Rapid server.
///
/// The sidecar buckets the caller from this header first and falls back to
/// the User-Agent allow-list only for third-party callers. Without it the
/// desktop is indistinguishable from whatever HTTP stack URLSession
/// advertises this OS release, and lands in ``other``.
///
/// The value is a closed label set shared with the engine
/// (``rapid_mlx/client_header.py``); it carries no version and no free text.
enum RapidClientHeader {
    static let field = "X-Rapid-Client"
    static let desktop = "rapid-desktop"
}

extension URLRequest {
    /// Stamp this request as coming from the desktop app. Every request to
    /// the local sidecar goes through here.
    mutating func applyRapidClientHeader() {
        setValue(RapidClientHeader.desktop, forHTTPHeaderField: RapidClientHeader.field)
    }
}
