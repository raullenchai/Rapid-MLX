import Foundation

/// User-controlled opt-in for the resource-intensive Video workspace.
///
/// The feature is deliberately absent from the sidebar until the user enables
/// it in Settings. Reading this key never starts a server or downloads a model;
/// it controls discoverability only. The actual model lifecycle remains an
/// explicit action inside ``VideoView``.
enum VideoFeatureConfig {
    static let enabledKey = "Rapid.experimental.videoGenerationEnabled"
    static let defaultEnabled = false

    static func isEnabled(in defaults: UserDefaults = .standard) -> Bool {
        defaults.object(forKey: enabledKey) as? Bool ?? defaultEnabled
    }
}
