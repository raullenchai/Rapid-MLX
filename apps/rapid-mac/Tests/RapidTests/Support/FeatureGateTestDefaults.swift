import Foundation

/// A new suite still inherits NSGlobalDomain. Keep default-off gate tests
/// independent of the developer's global opt-ins without changing their prefs.
final class FeatureGateTestDefaults: UserDefaults, @unchecked Sendable {
    private let domain: String

    init(domain: String) {
        self.domain = domain
        super.init(suiteName: domain)!
    }

    override func object(forKey defaultName: String) -> Any? {
        persistentDomain(forName: domain)?[defaultName]
    }
}
