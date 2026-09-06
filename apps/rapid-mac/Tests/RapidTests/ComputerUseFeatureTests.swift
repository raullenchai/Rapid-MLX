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

    /// ViewInspector is not available in this target, so the behavioral
    /// transition above is paired with the repository's established wiring
    /// guard pattern. This fails if ContentView stops observing the stored
    /// gate or stops applying the tested transition to its live selection.
    @Test("The stored Computer Use gate drives live navigation recovery")
    func gateChangeIsWiredToNavigation() throws {
        let contentView = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Sources/Rapid/UI/ContentView.swift")
        let source = try String(contentsOf: contentView, encoding: .utf8)
        let canonical = SourceGuardSupport.canonicalSource(source, literals: .preserve)

        #expect(canonical.contains("@AppStorage(ComputerUseFeatureConfig.enabledKey)privatevarcomputerUseEnabled"))
        #expect(canonical.contains(".onChange(of:experimentalDestinationState){_,statein"))
        #expect(canonical.contains("section=Self.sectionAfterComputerUseGateChange(current:section,enabled:state.computerUseEnabled)"))
    }

    @Test("Starter catalog is explicit about preview availability")
    func starterCatalog() {
        #expect(ComputerUseStarter.catalog.map(\.kind) == [
            .freeUpSpace,
            .tidyInbox,
            .draftAndPost,
            .prospectCustomers,
            .createDemoVideo,
            .reserved,
        ])
        #expect(ComputerUseStarter.catalog.dropLast().allSatisfy {
            $0.availability == .comingSoon
        })
        #expect(ComputerUseStarter.catalog.last?.availability == .reserved)
        #expect(ComputerUseStarter.catalog.allSatisfy {
            !$0.approvalNote.isEmpty
        })
    }
}
