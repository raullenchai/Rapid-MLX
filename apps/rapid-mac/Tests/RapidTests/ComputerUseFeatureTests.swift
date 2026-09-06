import Foundation
import Testing
@testable import Rapid

@Suite("Experimental Computer Use")
struct ComputerUseFeatureTests {
    @Test("Computer Use is opt-in")
    func defaultsOff() throws {
        let suite = "rapid.computer-use-gate-tests.\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }

        #expect(!ComputerUseFeatureConfig.isEnabled(in: defaults))
        defaults.set(true, forKey: ComputerUseFeatureConfig.enabledKey)
        #expect(ComputerUseFeatureConfig.isEnabled(in: defaults))
    }

    @MainActor
    @Test("Disabling while Computer Use is active returns to Chat")
    func disablingRecoversNavigation() {
        #expect(ContentView.sectionAfterComputerUseGateChange(
            current: .computerUse,
            enabled: false
        ) == .chat)
        #expect(ContentView.sectionAfterComputerUseGateChange(
            current: .computerUse,
            enabled: true
        ) == .computerUse)
        #expect(ContentView.sectionAfterComputerUseGateChange(
            current: .images,
            enabled: false
        ) == .images)
    }

    @Test("Starter catalog is explicit about preview availability")
    func starterCatalog() {
        #expect(ComputerUseStarter.catalog.map(\.kind) == [
            .freeUpSpace, .tidyInbox, .draftAndPost, .orderLunch
        ])
        #expect(ComputerUseStarter.catalog.first?.availability == .available)
        #expect(ComputerUseStarter.catalog.dropFirst().allSatisfy {
            $0.availability == .comingSoon
        })
        #expect(ComputerUseStarter.catalog.allSatisfy {
            !$0.approvalNote.isEmpty
        })
    }
}
