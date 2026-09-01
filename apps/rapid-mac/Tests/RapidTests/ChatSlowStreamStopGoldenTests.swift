import Foundation
import SwiftUI
import Testing

@testable import Rapid

/// The `slow-stream-stop` golden journey, sunk from the AX bash harness:
/// a user can stop a response that is actually being produced, the server
/// observes the cancellation (the app looking stopped is not enough), and
/// the conversation stays usable — including after the nastier zero-content
/// stop. The paced ``GoldenChatFake`` is the in-process
/// `FAKE_INTER_TOKEN_SLEEP_S`/`FAKE_CONTENT_REPEAT` fixture, and its event
/// log is the in-process `fake-events.jsonl`.
@MainActor
@Suite("Golden journey: slow-stream-stop", .serialized)
struct ChatSlowStreamStopGoldenTests {

    static func waitForStopButton(on stage: GoldenStage) async throws {
        try await stage.wait(for: "the send button to become Stop generating") {
            stage.tree().contains {
                $0.id == "ChatView.SendOrStopButton" && $0.text == "Stop generating"
            }
        }
    }

    @Test("Stop mid-content cancels the stream on the server side")
    func stopWhileContentIsStreaming() async throws {
        let fake = GoldenChatFake()
        // Slow enough that "streaming" is an observable state, endless
        // enough that the stream can never outrun the Stop press — the
        // in-process analog of the bash flow's
        // FAKE_INTER_TOKEN_SLEEP_S=0.01 FAKE_CONTENT_REPEAT=20000.
        fake.interChunkDelay = 0.02
        fake.contentRepeat = 20000
        let surface = GoldenChatSurface.mount(fake: fake)
        let stage = surface.stage

        try await surface.sendPrompt("golden stop marker")
        try await Self.waitForStopButton(on: stage)

        // Stop a stream that is actually streaming CONTENT. The button
        // flips to "Stop generating" on the first delta, and that delta is
        // a REASONING token — the answer itself has not started. Waiting
        // for the first content token sharpens what this test claims:
        // cancelling a response that is being produced, not one that has
        // yet to start.
        try await stage.waitForText("Hello")
        try stage.press("ChatView.SendOrStopButton")
        try await surface.waitForSendIdle()

        // The app looking stopped is its own story about itself; the
        // server observing the cancellation is the independent witness.
        try await stage.wait(for: "the server to observe the cancellation") {
            fake.events().contains {
                if case .chatCancelled = $0 { return true }
                return false
            }
        }
        #expect(
            !fake.events().contains {
                if case .chatFinished = $0 { return true }
                return false
            },
            "slow response finished instead of being stopped early"
        )
    }

    @Test("A send immediately after a zero-content Stop answers the new prompt")
    func zeroContentStopThenNewPrompt() async throws {
        let fake = GoldenChatFake()
        // A long reasoning runway (7 chunks x 80ms) so the Stop press
        // reliably lands before the first content token — the
        // release-dogfood edge where a cancelled prompt left an unanswered
        // user turn in wire history and the NEXT request answered that
        // cancelled prompt instead of its own.
        fake.interChunkDelay = 0.08
        fake.contentRepeat = 20000
        let surface = GoldenChatSurface.mount(fake: fake)
        let stage = surface.stage

        try await surface.sendPrompt("cancel this before content")
        try await Self.waitForStopButton(on: stage)
        try stage.press("ChatView.SendOrStopButton")
        try await surface.waitForSendIdle()
        try await stage.wait(for: "the server to observe the zero-content cancellation") {
            fake.events().contains {
                if case .chatCancelled = $0 { return true }
                return false
            }
        }

        // The immediately-following turn must be routed from its own
        // prompt. The fake shapes its answer by the LAST user message, so
        // a request still keyed to the cancelled prompt cannot produce the
        // list fixture.
        try await surface.sendPrompt("shape:list answer the new request")
        try await stage.waitForText("Three things, in order:")
        try await surface.waitForSendIdle()
        let lastPrompt = try #require(fake.recordedPrompts().last)
        #expect(lastPrompt.contains("shape:list answer the new request"))
        #expect(
            fake.events().contains {
                if case .chatFinished = $0 { return true }
                return false
            },
            "the post-stop turn never finished cleanly"
        )
    }

    @Test("A forming code fence never becomes a text row while streaming")
    func streamingFenceNeverFlickersIntoTheTree() async throws {
        let fake = GoldenChatFake()
        // Paced so the deliberately split fence fixture ("```", "python",
        // "\n", ...) is on screen mid-assembly across many samples.
        fake.interChunkDelay = 0.02
        let surface = GoldenChatSurface.mount(fake: fake)
        let stage = surface.stage

        try await surface.sendPrompt("shape:code stream a code block")

        // Sample the live tree while the stream assembles: no raw fence
        // marker may surface as its own accessibility row at any
        // intermediate revision.
        let deadline = Date(timeIntervalSinceNow: GoldenStage.defaultTimeout)
        var sawIdle = false
        while Date() < deadline {
            let tree = stage.tree()
            for node in tree {
                let stripped = node.text.filter { !$0.isWhitespace }
                // Any all-backtick row is a leaked fence — including the
                // full "```" the fixture emits as one complete chunk.
                if !stripped.isEmpty, stripped.allSatisfy({ $0 == "`" }) {
                    Issue.record("a forming code fence flickered into the streaming AX tree")
                    return
                }
            }
            if tree.contains(where: {
                $0.id == "ChatView.SendOrStopButton" && $0.text == "Send message"
            }) {
                sawIdle = true
                break
            }
            try await Task.sleep(nanoseconds: 10_000_000)
        }
        #expect(sawIdle, "the streamed code answer never settled")

        try await stage.waitForText("It runs in O(n) time and constant space.")
        GoldenMarkdownAssertions.assertCodeBlockIsItsOwnView(
            prose: "Here is the function you asked for",
            code: "def fib(n):",
            in: stage.tree()
        )
    }
}
