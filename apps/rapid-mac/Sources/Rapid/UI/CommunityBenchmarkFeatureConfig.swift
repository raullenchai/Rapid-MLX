import Foundation

/// Explicit opt-in for the Benchmark workspace (the local community-benchmark
/// runner).
///
/// Like the Video and Computer Use previews, the sidebar tab stays hidden until
/// the user enables it in Settings → Experimental. Reading or writing this
/// preference only controls discoverability: it never starts the server,
/// downloads a benchmark model, or runs a measurement. Those remain explicit
/// actions inside the Benchmark surface itself.
enum CommunityBenchmarkFeatureConfig {
    static let enabledKey = "Rapid.experimental.communityBenchmarkEnabled"
    static let defaultEnabled = false

    static func isEnabled(in defaults: UserDefaults = .standard) -> Bool {
        defaults.object(forKey: enabledKey) as? Bool ?? defaultEnabled
    }
}
