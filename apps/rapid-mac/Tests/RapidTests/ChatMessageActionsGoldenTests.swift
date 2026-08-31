import AppKit
import Foundation
import SwiftUI
import Testing

@testable import Rapid

/// The `message-actions` golden journey, sunk from the AX bash harness
/// (`gui-golden-flows.sh`) into `swift test`: every inline message action
/// must produce its advertised result. The real ``ChatView`` is mounted on
/// a ``GoldenStage`` and driven through the same accessibility identifiers
/// the out-of-process flow used; the streaming backend is a recording SSE
/// fake standing in for the fake sidecar, so the "a request actually left
/// the process" assertions keep an independent witness.
///
/// Journey inventory: `Tests/GUIGoldenFlows/journeys.yaml` lists this as
/// `driver: swift`; the GUI coverage contract maps that driver to this
/// suite's presence instead of a bash dispatcher case.
@MainActor
@Suite("Golden journey: message-actions", .serialized)
struct ChatMessageActionsGoldenTests {

    // MARK: - Recording SSE fake (in-process fake sidecar)

    /// Streams a deterministic multi-chunk assistant reply for every chat
    /// completion request and records each request body, mirroring the fake
    /// sidecar's `chat_request` event log that the bash flow counted.
    final class RecordingSSEProtocol: URLProtocol, @unchecked Sendable {
        private static let lock = NSLock()
        nonisolated(unsafe) private static var bodies: [Data] = []

        /// Chunked mid-sentence like the fake sidecar's CONTENT_CHUNKS so
        /// the transcript assertions ("deterministic content") stay
        /// word-for-word compatible with the bash journey.
        static let replyChunks = [
            "Hello", " from", " the", " fake", " rapid-mlx", " mock.",
            " I", " return", " deterministic", " content", " so", " the",
            " golden", " journey", " has", " something", " to", " assert", " on.",
        ]

        /// The fake sidecar shapes its answer by prompt keywords; mirror the
        /// one shape this suite needs: `shape:long` yields an answer taller
        /// than the stage viewport so scroll journeys have somewhere to go.
        static let longReplyChunks: [String] =
            (1...48).map { paragraph in
                "Paragraph \(paragraph) of the long settled answer that "
                    + "overflows the stage viewport.\n\n"
            } + ["END-OF-LONG-ANSWER"]

        private static func chunks(forPromptIn body: Data) -> [String] {
            guard
                let object = try? JSONSerialization.jsonObject(with: body) as? [String: Any],
                let messages = object["messages"] as? [[String: Any]],
                let prompt = messages.last(where: { ($0["role"] as? String) == "user" })?["content"] as? String
            else { return replyChunks }
            return prompt.contains("shape:long") ? longReplyChunks : replyChunks
        }

        static func reset() {
            lock.lock()
            bodies = []
            lock.unlock()
        }

        static func recordedBodies() -> [Data] {
            lock.lock()
            defer { lock.unlock() }
            return bodies
        }

        static func recordedPrompts() -> [String] {
            recordedBodies().compactMap { body in
                guard
                    let object = try? JSONSerialization.jsonObject(with: body) as? [String: Any],
                    let messages = object["messages"] as? [[String: Any]]
                else { return nil }
                return messages.last { ($0["role"] as? String) == "user" }
                    .flatMap { $0["content"] as? String }
            }
        }

        static func session() -> URLSession {
            let configuration = URLSessionConfiguration.ephemeral
            configuration.protocolClasses = [RecordingSSEProtocol.self]
            return URLSession(configuration: configuration)
        }

        override class func canInit(with request: URLRequest) -> Bool { true }
        override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

        /// URLSession surfaces an upload body as either `httpBody` or a
        /// stream depending on how the request was built; missing one form
        /// would silently drop the "a request actually left the process"
        /// witness for those requests.
        private func requestBody() -> Data? {
            if let body = request.httpBody { return body }
            guard let stream = request.httpBodyStream else { return nil }
            stream.open()
            defer { stream.close() }
            var body = Data()
            let bufferSize = 64 * 1024
            var buffer = [UInt8](repeating: 0, count: bufferSize)
            while stream.hasBytesAvailable {
                let read = stream.read(&buffer, maxLength: bufferSize)
                guard read > 0 else { break }
                body.append(buffer, count: read)
            }
            return body
        }

        override func startLoading() {
            var chunks = Self.replyChunks
            if let body = requestBody() {
                Self.lock.lock()
                Self.bodies.append(body)
                Self.lock.unlock()
                chunks = Self.chunks(forPromptIn: body)
            }

            let response = HTTPURLResponse(
                url: request.url!,
                statusCode: 200,
                httpVersion: "HTTP/1.1",
                headerFields: ["Content-Type": "text/event-stream"]
            )!
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            for chunk in chunks {
                let payload: [String: Any] = [
                    "choices": [["delta": ["content": chunk], "finish_reason": NSNull()]]
                ]
                let json = try! JSONSerialization.data(withJSONObject: payload)
                client?.urlProtocol(self, didLoad: Data("data: ".utf8) + json + Data("\n\n".utf8))
            }
            let finish: [String: Any] = [
                "choices": [["delta": [String: Any](), "finish_reason": "stop"]]
            ]
            let finishJSON = try! JSONSerialization.data(withJSONObject: finish)
            client?.urlProtocol(self, didLoad: Data("data: ".utf8) + finishJSON + Data("\n\n".utf8))
            client?.urlProtocol(self, didLoad: Data("data: [DONE]\n\n".utf8))
            client?.urlProtocolDidFinishLoading(self)
        }

        override func stopLoading() {}
    }

    // MARK: - Stage assembly

    static let alias = "fake-alias"

    @MainActor
    struct Mounted {
        let stage: GoldenStage
        let chat: ChatViewModel
        let server: ServerManager
    }

    /// Mount the real chat surface the way ``ContentView`` hosts it, with
    /// the smallest honest dependency set: a ready fake server, an SSE
    /// recording client, and throwaway stores. No conversation persists.
    static func mountChatSurface() -> Mounted {
        RecordingSSEProtocol.reset()
        let server = ServerManager(testingState: .ready(alias: alias))
        let chat = ChatViewModel(
            client: ChatStreamClient(
                baseURL: URL(string: "fake://golden-message-actions")!,
                session: RecordingSSEProtocol.session()
            ),
            server: server,
            persistsConversations: false
        )
        let downloads = DownloadManager()
        let quickstart = QuickstartCoordinator()

        let view = ChatView(
            viewModel: chat,
            server: server,
            alias: .constant(alias),
            readiness: .ready(alias: alias)
        )
        .environment(downloads)
        .environment(quickstart)

        let stage = GoldenStage(view)
        return Mounted(stage: stage, chat: chat, server: server)
    }

    /// `send_prompt` from the bash harness: type into the composer via AX
    /// set-value, press send, then require BOTH the drained composer and
    /// the recorded request — the composer clearing is the app's story
    /// about itself; the recorded body is the independent witness that a
    /// request actually left the process.
    static func sendPrompt(_ prompt: String, on stage: GoldenStage) async throws {
        let requestsBefore = RecordingSSEProtocol.recordedBodies().count
        try stage.setValue(prompt, for: "rapid.chat.compose")
        try stage.press("ChatView.SendOrStopButton")
        try await stage.wait(for: "composer to drain and the request to be recorded") {
            stage.value(of: "rapid.chat.compose") == ""
                && RecordingSSEProtocol.recordedBodies().count > requestsBefore
        }
    }

    /// `wait_send_idle` from the bash harness: the drained text can land a
    /// beat before the stream formally completes, and several message
    /// actions (Edit, Retry) are disabled while streaming — a press on a
    /// disabled control no-ops silently. The send button relabelling back
    /// from "Stop generating" is the AX-visible idle signal.
    static func waitForSendIdle(on stage: GoldenStage) async throws {
        try await stage.wait(for: "composer to settle into a ready, non-streaming state") {
            stage.tree().contains {
                $0.id == "ChatView.SendOrStopButton" && $0.text == "Send message"
            }
        }
    }

    static func waitForSettledReply(on stage: GoldenStage) async throws {
        try await stage.waitForText("deterministic content")
        // The reply streams in many chunks; settle on the final words so
        // later assertions see the finished turn, not a partial one.
        try await stage.waitForText("to assert on.")
        try await Self.waitForSendIdle(on: stage)
    }

    // MARK: - The journey

    @Test("Every inline message action produces its advertised result")
    func messageActionsJourney() async throws {
        let mounted = Self.mountChatSurface()
        let stage = mounted.stage

        try await Self.sendPrompt("original message action prompt", on: stage)
        try await Self.waitForSettledReply(on: stage)

        // Discover the per-message action identifiers from the live tree,
        // exactly as the bash flow did — no test-only entry points.
        guard let copyID = stage.identifier(withPrefix: "ChatView.Message.Copy.", last: true) else {
            Issue.record("completed turn exposes no assistant Copy action: \(stage.identifiers())")
            return
        }
        let selectID = copyID.replacingOccurrences(of: "Message.Copy.", with: "Message.SelectText.")
        let retryID = copyID.replacingOccurrences(of: "Message.Copy.", with: "Message.Retry.")
        guard let editID = stage.identifier(withPrefix: "ChatView.Message.Edit.") else {
            Issue.record("completed turn exposes no user Edit action: \(stage.identifiers())")
            return
        }

        // Copy response fills the pasteboard. The general pasteboard belongs
        // to whoever runs this suite, so snapshot it and put their contents
        // back afterwards.
        let savedPasteboardItems = (NSPasteboard.general.pasteboardItems ?? []).map {
            item -> NSPasteboardItem in
            let copy = NSPasteboardItem()
            for type in item.types {
                if let data = item.data(forType: type) {
                    copy.setData(data, forType: type)
                }
            }
            return copy
        }
        defer {
            NSPasteboard.general.clearContents()
            if !savedPasteboardItems.isEmpty {
                NSPasteboard.general.writeObjects(savedPasteboardItems)
            }
        }
        NSPasteboard.general.clearContents()
        try stage.press(copyID)
        try await stage.wait(for: "pasteboard content after Copy") {
            (NSPasteboard.general.string(forType: .string) ?? "").isEmpty == false
        }
        #expect(
            NSPasteboard.general.string(forType: .string)?.contains("deterministic content") == true,
            "Copy response did not place the assistant reply on the pasteboard"
        )

        // Select text opens its sheet; Done dismisses it.
        try stage.press(selectID)
        try await stage.waitForIdentifier("SelectText.Done")
        #expect(stage.treeText().contains("Selection here crosses paragraphs"))
        try stage.press("SelectText.Done")
        try await stage.waitForIdentifierGone("SelectText.Done")

        // Edit, then cancel: the draft must never be sent.
        let editSuffix = String(editID.dropFirst("ChatView.Message.Edit.".count))
        try stage.press(editID)
        try await stage.waitForIdentifier("ChatView.Message.EditField.\(editSuffix)")
        try stage.setValue(
            "cancelled edit must not send",
            for: "ChatView.Message.EditField.\(editSuffix)"
        )
        try stage.press("ChatView.Message.CancelEdit.\(editSuffix)")
        try await stage.waitForIdentifier(editID)
        // A wrongly-scheduled asynchronous send would leave the process a
        // beat after Cancel; give it that beat so the assertion can catch it.
        try await Task.sleep(nanoseconds: 300_000_000)
        #expect(
            !RecordingSSEProtocol.recordedPrompts().contains { $0.contains("cancelled edit must not send") },
            "cancelling a message edit sent the draft"
        )

        // Retry sends a replacement request.
        let requestsBeforeRetry = RecordingSSEProtocol.recordedBodies().count
        try stage.press(retryID)
        try await stage.wait(for: "replacement request after Retry") {
            RecordingSSEProtocol.recordedBodies().count > requestsBeforeRetry
        }
        try await Self.waitForSettledReply(on: stage)

        // Edit, then save: the edited prompt replaces the turn and sends.
        guard let editAfterRetry = stage.identifier(withPrefix: "ChatView.Message.Edit.") else {
            Issue.record("retried turn exposes no Edit action: \(stage.identifiers())")
            return
        }
        let saveSuffix = String(editAfterRetry.dropFirst("ChatView.Message.Edit.".count))
        try stage.press(editAfterRetry)
        try await stage.waitForIdentifier("ChatView.Message.EditField.\(saveSuffix)")
        try stage.setValue(
            "saved edited message prompt",
            for: "ChatView.Message.EditField.\(saveSuffix)"
        )
        try stage.press("ChatView.Message.SaveEdit.\(saveSuffix)")
        try await stage.wait(for: "the edited prompt to be sent") {
            RecordingSSEProtocol.recordedPrompts().contains("saved edited message prompt")
        }
        try await Self.waitForSettledReply(on: stage)
        #expect(stage.treeText().contains("saved edited message prompt"))
    }

    @Test("Jump to latest returns a settled transcript to its tail")
    func jumpToLatestOnSettledTranscript() async throws {
        let mounted = Self.mountChatSurface()
        let stage = mounted.stage

        // #1904 regression shape: a long, fully settled answer, moved away
        // from its tail. Jump to latest used to re-pin the follow flag but
        // leave the transcript physically scrolled up, because nothing was
        // streaming and no document-frame change would ever fire again.
        try await Self.sendPrompt("shape:long finished answer for jump-to-bottom", on: stage)
        try await stage.waitForText("END-OF-LONG-ANSWER")
        try await Self.waitForSendIdle(on: stage)
        try await stage.wait(for: "the settled transcript to rest at its tail") {
            (stage.scrollFraction() ?? 0) > 0.97
        }
        let bottom = try #require(stage.scrollFraction())

        try stage.setScrollFraction(0)
        try await stage.waitForIdentifier("Transcript.JumpToBottom")
        let scrolled = try #require(stage.scrollFraction())
        #expect(
            scrolled < bottom - 0.02,
            "scroll fixture did not move away from the settled transcript tail"
        )

        try stage.press("Transcript.JumpToBottom")
        try await stage.wait(for: "the transcript to return to its tail") {
            (stage.scrollFraction() ?? 0) > scrolled + 0.02
        }
        try await stage.waitForIdentifierGone("Transcript.JumpToBottom")
    }

    @Test("Model info opens from its anchor, dismisses, and can reopen")
    func modelInfoPopoverRoundTrip() async throws {
        let mounted = Self.mountChatSurface()
        let stage = mounted.stage

        try stage.press("ModelPickerBar.ModelInfo")
        try await stage.waitForText("Parameters")
        #expect(stage.treeText().contains(Self.alias))
        // Esc is the user's dismissal gesture for a transient popover.
        // Dismiss-then-reopen proves it is not a one-way overlay that traps
        // the rest of the composer, and that the anchor stays live — the
        // bash journey only ever asserted the anchor's second press
        // succeeded, so this is a strictly stronger contract.
        stage.pressEscape()
        try await stage.wait(for: "model info popover to close") {
            !stage.treeText().contains("Parameters")
        }
        try stage.press("ModelPickerBar.ModelInfo")
        try await stage.waitForText("Parameters")
        stage.pressEscape()
        try await stage.wait(for: "model info popover to close again") {
            !stage.treeText().contains("Parameters")
        }
    }
}
