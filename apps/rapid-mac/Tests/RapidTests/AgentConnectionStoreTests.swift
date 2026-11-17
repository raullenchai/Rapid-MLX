import Foundation
import Testing
@testable import Rapid

@Suite("Agent connection HOME resolution")
struct AgentConnectionStoreTests {
    @Test("Absolute HOME isolates the shared Desktop/CLI connection directory")
    func absoluteHome() {
        #expect(AgentConnectionStore.directory(environment: ["HOME": "/tmp/dogfood/home"]).path
            == "/tmp/dogfood/home/.rapid-mlx/connect")
    }

    @Test("Missing, empty and relative HOME use the FileManager fallback",
          arguments: [[:], ["HOME": ""], ["HOME": "relative/home"]])
    func invalidHome(environment: [String: String]) {
        #expect(AgentConnectionStore.directory(environment: environment)
            == FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent(".rapid-mlx/connect", isDirectory: true))
    }

    @Test("Production accessor uses the process environment")
    func productionAccessor() {
        #expect(AgentConnectionStore.directory
            == AgentConnectionStore.directory(environment: ProcessInfo.processInfo.environment))
    }
}
