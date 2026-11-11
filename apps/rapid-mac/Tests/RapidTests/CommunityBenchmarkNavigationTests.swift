import Foundation
import Testing
@testable import Rapid

@Suite("Community Benchmark navigation")
struct CommunityBenchmarkNavigationTests {
    @MainActor
    @Test("Other experimental gates never dismiss Benchmark")
    func benchmarkSurvivesGateChanges() {
        for enabled in [false, true] {
            #expect(ContentView.sectionAfterLocalSearchGateChange(
                current: .benchmark, enabled: enabled
            ) == .benchmark)
            #expect(ContentView.sectionAfterVideoGateChange(
                current: .benchmark, enabled: enabled
            ) == .benchmark)
            #expect(ContentView.sectionAfterComputerUseGateChange(
                current: .benchmark, enabled: enabled
            ) == .benchmark)
            #expect(ContentView.sectionAfterShareComputeGateChange(
                current: .benchmark, enabled: enabled
            ) == .benchmark)
        }
    }

    // Follow the existing destination wiring guards: ViewInspector is not
    // available in this target, so inspect the production view declarations.
    @Test("Both sidebar layouts always place Benchmark directly after Agent")
    func mainGroupPlacement() throws {
        let sidebar = try canonicalSource("SidebarView.swift")
        for renderer in ["row", "compactItem"] {
            let expected = """
            .accessibilityIdentifier("Sidebar.Launch")
            \(renderer)(
                title: String(localized: "Benchmark"),
                systemImage: "gauge.with.dots.needle.50percent",
                isSelected: selection == .benchmark,
                action: { selection = .benchmark }
            )
            .accessibilityIdentifier("Sidebar.CommunityBenchmark")
            """
            #expect(sidebar.contains(SourceGuardSupport.canonicalSource(expected, literals: .preserve)))
        }
        let launch = try #require(sidebar.range(of: #".accessibilityIdentifier("Sidebar.Launch")"#))
        let benchmark = try #require(sidebar.range(of: #".accessibilityIdentifier("Sidebar.CommunityBenchmark")"#))
        let computerUse = try #require(sidebar.range(of: "ifcomputerUseEnabled{", range: launch.upperBound..<sidebar.endIndex))
        let experimental = try #require(sidebar.range(of: #"SectionHeader(String(localized:"Experimental"))"#))
        #expect(launch.lowerBound < benchmark.lowerBound)
        #expect(benchmark.lowerBound < computerUse.lowerBound)
        #expect(computerUse.lowerBound < experimental.lowerBound)
        #expect(sidebar.components(separatedBy: #".accessibilityIdentifier("Sidebar.CommunityBenchmark")"#).count == 3)
        #expect(sidebar.contains("iflocalSearchEnabled||videoGenerationEnabled||shareComputeEnabled{"))
    }

    @Test("Benchmark routes directly to its view and ignores the legacy preference")
    func destinationIsAlwaysAvailable() throws {
        let content = try canonicalSource("ContentView.swift")
        #expect(content.contains("case.benchmark:CommunityBenchmarkView("))
        for file in ["ContentView.swift", "SidebarView.swift", "SettingsView.swift"] {
            let source = try canonicalSource(file)
            #expect(!source.contains("Rapid.experimental.communityBenchmarkEnabled"))
            #expect(!source.contains("CommunityBenchmarkFeatureConfig"))
        }
        let settings = try canonicalSource("SettingsView.swift")
        #expect(!settings.contains("Settings.Experimental.BenchmarkToggle"))
    }

    private func canonicalSource(_ name: String) throws -> String {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Sources/Rapid/UI/\(name)")
        return SourceGuardSupport.canonicalSource(
            try String(contentsOf: url, encoding: .utf8), literals: .preserve
        )
    }
}
