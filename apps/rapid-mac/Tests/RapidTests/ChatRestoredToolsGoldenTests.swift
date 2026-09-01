import Foundation
import SwiftUI
import Testing

@testable import Rapid

/// The `restored-tools` golden journey, sunk from the AX bash harness: a
/// conversation whose answer was produced through native web research is
/// closed, "relaunched", reopened from the real sidebar row, and continued
/// — and the follow-up turn still advertises tools and carries the web
/// evidence to synthesis. The in-process relaunch mounts a second surface
/// over the same conversation store with a fresh fake and a fresh tool
/// registry, mirroring `relaunch_persona`'s fresh app process + fresh
/// sidecar over a kept persona home.
@MainActor
@Suite("Golden journey: restored-tools", .serialized)
struct ChatRestoredToolsGoldenTests {

    /// The deterministic search provider — the injected-runner analog of
    /// `RAPID_GUI_WEB_SEARCH_FIXTURE=1`, producing the exact fixture result
    /// `WebSearchTool` ships for that switch, plus an execution count so
    /// the app side of every search is measured too.
    final class FixtureSearchRunner: @unchecked Sendable {
        private let lock = NSLock()
        private var count = 0

        func executions() -> Int {
            lock.lock()
            defer { lock.unlock() }
            return count
        }

        func run() -> ToolCallResult {
            lock.lock()
            count += 1
            lock.unlock()
            return WebSearchTool.formatOutput(
                query: "golden restored tools",
                provider: .duckduckgo,
                results: [
                    WebSearchTool.Result(
                        title: "Golden technology story",
                        url: "https://example.com/golden-tech",
                        snippet: "A concrete dated technology result used by the restored-thread GUI integration test."
                    )
                ]
            )
        }
    }

    static func makeSurface(over storeURL: URL) -> (GoldenChatSurface, FixtureSearchRunner) {
        let fake = GoldenChatFake()
        fake.nativeWebSearchFixture = true
        let runner = FixtureSearchRunner()
        let registry = BuiltinToolRegistry(webSearchRunner: { _, _, _ in runner.run() })
        let surface = GoldenChatSurface.mountWithSidebar(
            fake: fake,
            tools: registry,
            conversationStoreURL: storeURL
        )
        return (surface, runner)
    }

    /// The bash flow's closing event check, on the recorded bodies: the
    /// synthesis request for a researched turn must still carry the web
    /// evidence — its trailing message is a tool result holding the fixture
    /// story and URL, not just any tool role — AND the tools.
    static func synthesisRequestsCarryingWebEvidence(in fake: GoldenChatFake) -> Int {
        fake.recordedBodies().filter { body in
            guard
                let object = try? JSONSerialization.jsonObject(with: body) as? [String: Any],
                let messages = object["messages"] as? [[String: Any]],
                let trailing = messages.last,
                (trailing["role"] as? String) == "tool",
                let evidence = trailing["content"] as? String,
                evidence.contains("Golden technology story"),
                evidence.contains("https://example.com/golden-tech")
            else { return false }
            return ((object["tools"] as? [[String: Any]]) ?? []).contains {
                (($0["function"] as? [String: Any])?["name"] as? String) == "web_search"
            }
        }.count
    }

    @Test("A restored conversation keeps deterministic web research")
    func restoredConversationKeepsWebResearch() async throws {
        let storeURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("golden-restored-tools-\(UUID().uuidString)")
            .appendingPathComponent("conversations.json")
        try FileManager.default.createDirectory(
            at: storeURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        defer { try? FileManager.default.removeItem(at: storeURL.deletingLastPathComponent()) }

        // Session 1: research a fresh question through the native tool
        // path. Scoped in a nested function so every strong reference to
        // the first surface dies before the "relaunch" below — a surface
        // retained past this point would leave two live owners of the same
        // conversation store and make the restore assertion dishonest.
        func runFirstSession() async throws {
            let (first, firstRunner) = Self.makeSurface(over: storeURL)
            try await first.sendPrompt("What's a major news story from the last week?")
            try await first.stage.waitForText("Tool call web_search")
            try await first.stage.waitForText("Golden technology story")
            try await first.waitForSendIdle()
            #expect(firstRunner.executions() == 1)
            #expect(
                Self.synthesisRequestsCarryingWebEvidence(in: first.fake) == 1,
                "the fresh synthesis request did not carry web evidence and tools"
            )
        }
        try await runFirstSession()

        // "Relaunch": a fresh surface, fake, and registry over the same
        // conversation store.
        let (restored, restoredRunner) = Self.makeSurface(over: storeURL)
        let stage = restored.stage

        // Match the sidebar ROW exactly. `Sidebar.Conversation.` is a
        // namespace: it also contains `…Pin.<uuid>`, `…Menu.<uuid>` and
        // `…Action.*`, and a prefix match could press one of those instead.
        try await stage.wait(for: "the restored conversation row in the sidebar") {
            stage.identifiers().contains { Self.isConversationRow($0) }
        }
        let rowID = try #require(stage.identifiers().first { Self.isConversationRow($0) })
        try stage.press(rowID)
        try await stage.waitForText("Golden technology story")
        try await restored.waitForSendIdle()

        // The follow-up turn must research again: tools re-advertised, a
        // second native call chosen, a second app-side execution.
        try await restored.sendPrompt(
            "What about technology? Find one concrete story and summarize it."
        )
        try await stage.wait(for: "the follow-up turn's web evidence") {
            restoredRunner.executions() >= 1
        }
        try await restored.waitForSendIdle()
        #expect(restoredRunner.executions() == 1)
        #expect(
            Self.synthesisRequestsCarryingWebEvidence(in: restored.fake) == 1,
            "the restored synthesis request did not carry web evidence and tools"
        )
        let nativeCalls = restored.fake.events().filter {
            if case .nativeWebSearchCall = $0 { return true }
            return false
        }
        #expect(
            nativeCalls.count == 1,
            "the fake model did not natively choose web_search exactly once after restore"
        )
    }

    private static func isConversationRow(_ identifier: String) -> Bool {
        identifier.range(
            of: #"^Sidebar\.Conversation\.[0-9A-Fa-f-]{36}$"#,
            options: .regularExpression
        ) != nil
    }
}
