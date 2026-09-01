import Foundation
import SwiftUI
import Testing

@testable import Rapid

/// The `tool-loop-budget` golden journey, sunk from the AX bash harness:
/// against a model that answers every tools-carrying request with yet
/// another tool call, the app must execute only its bounded budget and then
/// force a final synthesis by withdrawing the tools. The real
/// ``BuiltinToolRegistry`` runs with an injected search runner, so the
/// count of actual tool EXECUTIONS is measured on the app side too — not
/// just inferred from the fake's request log.
@MainActor
@Suite("Golden journey: tool-loop-budget", .serialized)
struct ChatToolLoopBudgetGoldenTests {

    /// Deterministic in-process `web_search` provider: no network, no
    /// fixture env var, and an execution counter the assertions read.
    final class CountingSearchRunner: @unchecked Sendable {
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
            let ordinal = count
            lock.unlock()
            return ToolCallResult(
                toolCallID: "",
                content: #"{"results": [{"title": "Golden loop evidence \#(ordinal)", "url": "https://example.com/golden-\#(ordinal)", "snippet": "Deterministic in-process search result."}]}"#
            )
        }
    }

    @Test("Runaway tool use ends with a bounded synthesis answer")
    func boundedToolLoop() async throws {
        let fake = GoldenChatFake()
        let runner = CountingSearchRunner()
        let registry = BuiltinToolRegistry(
            webSearchRunner: { _, _, _ in runner.run() }
        )
        let surface = GoldenChatSurface.mount(fake: fake, tools: registry)
        let stage = surface.stage

        try await surface.sendPrompt("shape:tool-loop research this topic thoroughly")
        try await stage.waitForText(GoldenChatFake.toolLoopSynthesisText)
        try await surface.waitForSendIdle()

        // The app executed exactly its budget of tools — measured at the
        // registry seam, the same place production executions run.
        #expect(
            runner.executions() == 3,
            "the app did not stop after exactly three tool executions"
        )

        // And the fake's request-lifecycle log agrees: three tool calls
        // served, then one synthesis request carrying all three results.
        let calls = fake.events().filter {
            if case .toolLoopCall = $0 { return true }
            return false
        }
        #expect(calls.count == 3, "the fake model was not asked for exactly three tool calls")
        let syntheses = fake.events().filter {
            if case .toolLoopSynthesis = $0 { return true }
            return false
        }
        #expect(
            syntheses == [.toolLoopSynthesis(toolResults: 3)],
            "the capped loop did not finish with one synthesis request carrying three tool results"
        )

        // The final request must have withdrawn the tools — that is the
        // mechanism that forces the synthesis instead of a fourth call.
        let lastBody = try #require(fake.recordedBodies().last)
        let object = try #require(
            try JSONSerialization.jsonObject(with: lastBody) as? [String: Any]
        )
        let advertisedTools = (object["tools"] as? [Any]) ?? []
        #expect(
            advertisedTools.isEmpty,
            "the final synthesis request still advertised tools"
        )
    }
}
