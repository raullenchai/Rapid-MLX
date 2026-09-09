import Foundation

/// Explicit opt-in for lending this Mac to a third-party compute pool.
/// The preference controls discoverability only; enabling it never downloads
/// weights, starts a model, or contacts QuickSilver.
enum ShareComputeFeatureConfig {
    static let enabledKey = "Rapid.experimental.shareComputeEnabled"
    static let defaultEnabled = false

    static func isEnabled(in defaults: UserDefaults = .standard) -> Bool {
        defaults.object(forKey: enabledKey) as? Bool ?? defaultEnabled
    }
}
