import Foundation

/// Remove retired preferences once at startup; repeated launches are harmless.
enum LegacyPreferenceCleanup {
    static func run(defaults: UserDefaults = .standard) {
        for key in [
            "Rapid.personalIntelligence.introductionCompleted",
            "Rapid.personalIntelligence.preferredEnabled",
            "Rapid.personalIntelligence.conversationStates",
        ] {
            defaults.removeObject(forKey: key)
        }
    }
}
