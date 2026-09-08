import Foundation
import Testing
@testable import Rapid

@Suite("Experimental Benchmark gate", .serialized)
struct CommunityBenchmarkFeatureGateTests {
    @Test("Benchmark is opt-in when no preference exists")
    func defaultsOff() throws {
        let suite = "rapid.benchmark-gate-tests.\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }

        #expect(!CommunityBenchmarkFeatureConfig.isEnabled(in: defaults))
        defaults.set(true, forKey: CommunityBenchmarkFeatureConfig.enabledKey)
        #expect(CommunityBenchmarkFeatureConfig.isEnabled(in: defaults))
    }

    @MainActor
    @Test("Disabling while Benchmark is active returns to Chat")
    func disablingRecoversNavigation() {
        #expect(ContentView.sectionAfterBenchmarkGateChange(current: .benchmark, enabled: false) == .chat)
        #expect(ContentView.sectionAfterBenchmarkGateChange(current: .benchmark, enabled: true) == .benchmark)
        #expect(ContentView.sectionAfterBenchmarkGateChange(current: .images, enabled: false) == .images)
    }
}
