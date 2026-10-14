import Foundation

/// Experimental opt-in, resolved again at each engine spawn. Disabling keeps
/// the on-disk index; the running engine retains its flags until restarted.
enum LocalSearchFeatureConfig {
    static let enabledKey = "Rapid.experimental.localSearchEnabled"
    static let defaultEnabled = false

    static func restartAlias(enabling: Bool, state: ServerState) -> String? {
        guard enabling else { return nil }
        switch state {
        case .ready(let alias), .starting(let alias): return alias
        default: return nil
        }
    }

    static func isEnabled(in defaults: UserDefaults = .standard) -> Bool {
        defaults.object(forKey: enabledKey) as? Bool ?? defaultEnabled
    }
}
