import Foundation
import Testing
@testable import Rapid

@Suite("Experimental Community Benchmark gate", .serialized)
struct CommunityBenchmarkFeatureGateTests {
    private static var sourceRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // RapidTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // package root
    }

    @Test("Community Benchmark is opt-in when no preference exists")
    func defaultsOff() throws {
        let suite = "rapid.community-benchmark-gate-tests.\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }

        #expect(!CommunityBenchmarkFeatureConfig.isEnabled(in: defaults))
        defaults.set(true, forKey: CommunityBenchmarkFeatureConfig.enabledKey)
        #expect(CommunityBenchmarkFeatureConfig.isEnabled(in: defaults))
    }

    @MainActor
    @Test("Disabling while Community Benchmark is active returns to Chat")
    func disablingRecoversNavigation() {
        #expect(
            ContentView.sectionAfterCommunityBenchmarkGateChange(
                current: .benchmark,
                enabled: false
            ) == .chat
        )
        #expect(
            ContentView.sectionAfterCommunityBenchmarkGateChange(
                current: .benchmark,
                enabled: true
            ) == .benchmark
        )
        #expect(
            ContentView.sectionAfterCommunityBenchmarkGateChange(
                current: .images,
                enabled: false
            ) == .images
        )
    }

    @Test("Desktop wires one preference through Settings, sidebar, and detail routing")
    func desktopWiring() throws {
        let settings = try source("Sources/Rapid/UI/SettingsView.swift")
        let sidebar = try source("Sources/Rapid/UI/SidebarView.swift")
        let content = try source("Sources/Rapid/UI/ContentView.swift")

        #expect(settings.contains("@AppStorage(CommunityBenchmarkFeatureConfig.enabledKey)"))
        #expect(settings.contains("Settings.Experimental.CommunityBenchmarkToggle"))
        #expect(sidebar.contains("if communityBenchmarkEnabled"))
        #expect(sidebar.contains("Sidebar.CommunityBenchmark"))
        #expect(content.contains("communityBenchmarkEnabled: communityBenchmarkEnabled"))
        #expect(content.contains("if communityBenchmarkEnabled"))
    }

    private func source(_ relativePath: String) throws -> String {
        try String(
            contentsOf: Self.sourceRoot.appendingPathComponent(relativePath),
            encoding: .utf8
        )
    }
}
