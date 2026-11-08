import Foundation
import Testing
@testable import Rapid

@MainActor
@Suite("Quickstart memory admission")
struct QuickstartLoadMemoryTests {
    @Test("Onboarding bypass reaches a real startup failure while a manual load still warns")
    func bypassDoesNotHideStartupFailure() async throws {
        let server = ServerManager(testingState: .idle, binaryPath: URL(fileURLWithPath: "/usr/bin/true"))
        defer { server.shutdownSync() }
        server.memorySnapshotProvider = {
            .init(totalBytes: 18 * 1_073_741_824, usedBytes: 17 * 1_073_741_824)
        }
        let alias = "qwen3.5-4b-4bit"
        await server.start(alias: alias)
        #expect(server.pendingMemoryWarning?.alias == alias)
        server.cancelPendingMemoryLoad(try #require(server.pendingMemoryWarning))
        await server.start(alias: alias, bypassMemoryGuard: true)
        #expect(server.pendingMemoryWarning == nil)
        // /usr/bin/true is deliberately not an engine: proceeding beyond the
        // admission guard must expose its failure via normal startup recovery.
        guard case .crashed(let failedAlias, let message) = server.state else {
            Issue.record("The non-engine executable should surface a startup error: \(server.state)")
            return
        }
        #expect(failedAlias == alias)
        #expect(!message.isEmpty)
    }
}
