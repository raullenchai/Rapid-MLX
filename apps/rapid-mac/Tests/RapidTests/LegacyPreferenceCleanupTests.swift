import Foundation
import Testing
@testable import Rapid

@Suite("Legacy preference cleanup")
struct LegacyPreferenceCleanupTests {
    @Test("Retired keys are removed idempotently without changing other preferences")
    func removesOnlyRetiredKeys() throws {
        let suite = "rapid-legacy-cleanup-test-\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        defaults.set(true, forKey: "Rapid.personalIntelligence.introductionCompleted")
        defaults.set(false, forKey: "Rapid.personalIntelligence.preferredEnabled")
        defaults.set("{legacy}", forKey: "Rapid.personalIntelligence.conversationStates")
        for key in [
            "rapid.tools.enabled.local_search",
            "rapid.tools.enabled.local_read",
            "rapid.tools.enabled.local_write",
            "rapid.tools.enabled.local_trash",
            "rapid.tools.enabled.local_run",
        ] {
            defaults.set(true, forKey: key)
        }
        let retained: [String: Any] = [
            "rapid.tools.enabled.web_search": false,
            "conversation": "keep this",
            "Rapid.unrelated": false,
        ]
        for (key, value) in retained { defaults.set(value, forKey: key) }

        LegacyPreferenceCleanup.run(defaults: defaults)
        LegacyPreferenceCleanup.run(defaults: defaults)

        #expect(defaults.persistentDomain(forName: suite) as NSDictionary? == retained as NSDictionary)
    }
}
