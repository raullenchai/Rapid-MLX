import Foundation

/// Explicit opt-in for the early Computer Use workspace.
///
/// Reading or writing this preference only controls discoverability. It must
/// never download a model, request Screen Recording / Accessibility access, or
/// begin observing the desktop. Those prompts belong to the action that needs
/// them, not a Settings toggle.
enum ComputerUseFeatureConfig {
    static let enabledKey = "Rapid.experimental.computerUseEnabled"
    static let defaultEnabled = false

    static func isEnabled(in defaults: UserDefaults = .standard) -> Bool {
        defaults.object(forKey: enabledKey) as? Bool ?? defaultEnabled
    }
}
