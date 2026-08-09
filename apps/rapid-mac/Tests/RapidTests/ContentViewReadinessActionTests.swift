import Foundation
import Testing
@testable import Rapid

@Suite("ContentView readiness action wiring")
struct ContentViewReadinessActionTests {
    private static var packageRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // RapidTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // package root
    }

    @Test("Chat readiness replaces a different resident model")
    func readinessUsesEnsureServing() throws {
        let url = Self.packageRoot
            .appendingPathComponent("Sources/Rapid/UI/ContentView.swift")
        let body = try String(contentsOf: url, encoding: .utf8)
        let source = CapabilityChipRenderGateSourceGuardTests
            .stripCommentsAndWhitespace(body)

        guard let functionStart = source.range(
            of: "privatefuncstartModel(_target:String){"
        )?.lowerBound else {
            Issue.record("ContentView.startModel could not be found")
            return
        }
        let suffix = source[functionStart...]
        guard let functionEnd = suffix.firstIndex(of: "}") else {
            Issue.record("ContentView.startModel has no closing brace")
            return
        }
        let function = String(suffix[...functionEnd])

        #expect(
            function.contains("server.ensureServing(alias:target,hfPath:hfPath)"),
            "The Chat readiness button must switch away from a resident Images model."
        )
        #expect(
            !function.contains("server.start(alias:target,hfPath:hfPath)"),
            "ServerManager.start is cold-start only and silently no-ops while Images is resident."
        )
    }
}
