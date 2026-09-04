import Foundation

/// User-controlled opt-in for the Community Benchmark workspace.
///
/// The benchmark runner remains installed and the CLI remains available when
/// this is off. This preference controls Desktop discoverability only: the
/// sidebar destination is absent until the user enables it in Settings.
enum CommunityBenchmarkFeatureConfig {
    static let enabledKey = "Rapid.experimental.communityBenchmarkEnabled"
    static let defaultEnabled = false

    static func isEnabled(in defaults: UserDefaults = .standard) -> Bool {
        defaults.object(forKey: enabledKey) as? Bool ?? defaultEnabled
    }
}
