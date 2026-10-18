import Foundation

/// Remove retired preferences once at startup; repeated launches are harmless.
/// Only UserDefaults keys are removed; user files are never touched.
enum LegacyPreferenceCleanup {
    static func run(defaults: UserDefaults = .standard) {
        for key in [
            "Rapid.personalIntelligence.introductionCompleted",
            "Rapid.personalIntelligence.preferredEnabled",
            "Rapid.personalIntelligence.conversationStates",
            "rapid.tools.enabled.local_search",
            "rapid.tools.enabled.local_read",
            "rapid.tools.enabled.local_write",
            "rapid.tools.enabled.local_trash",
            "rapid.tools.enabled.local_run",
        ] {
            defaults.removeObject(forKey: key)
        }
    }
}
