import Foundation

/// Experimental opt-in, resolved again at each engine spawn. Disabling keeps
/// the on-disk index and restarts an enabled child so indexing stops.
enum LocalSearchFeatureConfig {
    static let enabledKey = "Rapid.experimental.localSearchEnabled"
    static let defaultEnabled = false

    static func restartAlias(
        enabled: Bool,
        enabledAtSpawn: Bool,
        state: ServerState
    ) -> String? {
        guard enabled != enabledAtSpawn else { return nil }
        switch state {
        case .ready(let alias), .starting(let alias): return alias
        default: return nil
        }
    }

    static func isEnabled(in defaults: UserDefaults = .standard) -> Bool {
        defaults.object(forKey: enabledKey) as? Bool ?? defaultEnabled
    }
}
