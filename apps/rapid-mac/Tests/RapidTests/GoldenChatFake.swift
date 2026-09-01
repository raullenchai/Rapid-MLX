import AppKit
import Foundation
import SwiftUI
import Testing

@testable import Rapid

/// Fake ports count upward from far above any real `PortSweep` allocation,
/// so an in-process fake can never be confused with a genuinely running
/// server, and each fake's traffic is distinguishable in the shared
/// protocol registry.
private let portAllocationLock = NSLock()
private nonisolated(unsafe) var nextFakePort = 52_100

/// In-process stand-in for `scripts/fake-rapid-mlx.sh`'s chat endpoint,
/// shared by the `driver: swift` golden journeys. It mirrors the pieces the
/// bash fake made load-bearing:
///
/// * **Response shapes** — the `shape:*` chunk fixtures, ported verbatim,
///   selected by a marker in the LAST user message so multi-turn journeys
///   get a different shape per turn. Chunks split mid-token on purpose: a
///   renderer that only works when a fence or table row arrives whole is a
///   renderer that breaks on a real stream.
/// * **Reasoning first** — `reasoning_content` deltas stream before the
///   answer, exactly like the bash fake, so "the button flipped to Stop but
///   no content exists yet" is a reachable app state.
/// * **Pacing** — `interChunkDelay`/`contentRepeat` are the in-process
///   `FAKE_INTER_TOKEN_SLEEP_S`/`FAKE_CONTENT_REPEAT`, giving the Stop
///   journeys a stream that is observably in flight.
/// * **Lifecycle events** — the analog of `FAKE_EVENT_LOG`: an app that
///   looks stopped must be backed by a server that observed the
///   cancellation, not one that quietly finished.
/// * **The runaway tool loop** — `shape:tool-loop` answers every
///   tools-carrying request with one more `web_search` call and answers the
///   final tool-less request with a synthesis, so the app's bounded budget
///   is what ends the loop.
///
/// Each instance owns a unique fake port, and the `URLProtocol` routes by
/// port, so concurrently running suites never observe each other's traffic.
final class GoldenChatFake: @unchecked Sendable {

    enum Event: Equatable, Sendable {
        case chatFinished(chunks: Int)
        case chatCancelled(chunks: Int)
        case toolLoopCall(id: String)
        case toolLoopSynthesis(toolResults: Int)
        case nativeWebSearchCall(id: String)
    }

    // MARK: - Fixtures (ported from fake-rapid-mlx.sh)

    /// `CONTENT_CHUNKS`, with the smoke-runner wording adjusted to name
    /// this harness ("golden journey" instead of "smoke test").
    static let contentChunks = [
        "Hello", " from", " the", " fake", " rapid-mlx", " mock.",
        " I", " return", " deterministic", " content", " so", " the",
        " golden", " journey", " has", " something", " to", " assert", " on.",
    ]

    /// `REASONING_CHUNKS`.
    static let reasoningChunks = ["Let", " me", " think", " about", " the", " prompt", "."]

    static let toolLoopSynthesisText = "Golden tool-loop synthesis from existing evidence."

    /// `RESPONSE_SHAPES`, in the bash fake's marker-matching order. The one
    /// divergence is `shape:long`: the bash fake repeats `CONTENT_CHUNKS`,
    /// while the in-process scroll journeys need a unique tail sentinel to
    /// wait on, so the long answer is numbered paragraphs ending in
    /// `END-OF-LONG-ANSWER`.
    static let responseShapes: [(marker: String, chunks: [String])] = [
        (
            "shape:code",
            [
                "Here is the function you asked for:\n\n",
                "```", "python", "\n",
                "def fib(n):\n", "    a, b = 0, 1\n",
                "    for _ in range(n):\n", "        a, b = b, a + b\n",
                "    return a\n",
                "```", "\n\n",
                "The same renderer also handles punctuation-bearing configured tokens:\n\n",
                "```", "css", "\n",
                ".card { background-", "color: red; }\n",
                "@font-", "face { font-family: Demo; }\n",
                "```", "\n\n",
                "```", "makefile", "\n",
                ".PH", "ONY: all\n",
                "FILES := $(filter-", "out %.tmp,$(ALL_FILES))\n",
                "```", "\n\n",
                "It runs in O(n) time and constant space.",
            ]
        ),
        (
            "shape:table",
            [
                "| model | size | speed |\n",
                "| --- | --- | ---", " |\n",
                "| qwen3.5-9b | 5.2 GB | 74 tok/s |\n",
                "| llama-3.1-8b | 4.5 GB | 68 tok/s |\n",
                "\nBoth fit comfortably in 16 GB.",
            ]
        ),
        (
            "shape:math",
            [
                "The Gaussian integral is\n\n",
                "$$\\int_{-\\infty}^{\\infty} e^{-x^2}\\,dx = \\sqrt{\\pi}$$",
                "\n\nand inline it reads $e^{i\\pi} + 1 = 0$.",
                "\n\nA bridged congruence is $$a^{p-1} \\equiv 1 \\mod p$$.",
                "\n\nA bridged alignment is $$\\begin{align}x &= 1 \\\\ y &= \\boxed{2}\\end{align}$$.",
            ]
        ),
        (
            "shape:list",
            [
                "Three things, in order:\n\n",
                "1. First, ", "read the prompt.\n",
                "2. Second, ", "plan the answer.\n",
                "   - a nested point\n", "   - another one\n",
                "3. Third, ", "write it down.\n",
            ]
        ),
        (
            "shape:long",
            (1...48).map { paragraph in
                "Paragraph \(paragraph) of the long settled answer that "
                    + "overflows the stage viewport.\n\n"
            } + ["END-OF-LONG-ANSWER"]
        ),
    ]

    // MARK: - Per-instance configuration

    /// Seconds between deltas (`FAKE_INTER_TOKEN_SLEEP_S`). Configure
    /// before mounting; the emitter reads it once per request.
    var interChunkDelay: TimeInterval = 0

    /// How many times the selected content chunk list repeats
    /// (`FAKE_CONTENT_REPEAT`). Stop journeys set this high enough that a
    /// stream can never outrun the press racing to cancel it.
    var contentRepeat: Int = 1

    /// The in-process `RAPID_GUI_WEB_SEARCH_FIXTURE`: when enabled and the
    /// request advertises `web_search`, the fake "model" natively chooses
    /// that tool once per user turn, then synthesizes normally after the app
    /// appends the call's result. Deliberately not keyed to prompt keywords:
    /// product routing belongs to the model's tool choice, never to
    /// app-side regexes.
    var nativeWebSearchFixture = false

    // MARK: - Recorded evidence

    private let lock = NSLock()
    private var bodies: [Data] = []
    private var eventLog: [Event] = []

    /// Every recorded request body — the independent "a request actually
    /// left the process" witness, mirroring the bash fake's `chat_request`
    /// event log.
    func recordedBodies() -> [Data] {
        lock.lock()
        defer { lock.unlock() }
        return bodies
    }

    /// The last user-message text of each recorded request.
    func recordedPrompts() -> [String] {
        recordedBodies().compactMap { Self.lastUserText(in: $0) }
    }

    func events() -> [Event] {
        lock.lock()
        defer { lock.unlock() }
        return eventLog
    }

    fileprivate func record(body: Data) {
        lock.lock()
        bodies.append(body)
        lock.unlock()
    }

    fileprivate func record(event: Event) {
        lock.lock()
        eventLog.append(event)
        lock.unlock()
    }

    // MARK: - Session plumbing

    /// Unique per instance so the protocol registry can route each request
    /// to the fake behind it. A PORT, not a host, because
    /// ``ChatViewModel`` re-targets its client onto
    /// `127.0.0.1:server.activePort` before every send — any base URL the
    /// test hands the client is overwritten, so the port is the one part
    /// of the address a test controls end to end (by giving its
    /// ``ServerManager`` the same value). Nothing ever binds the port; the
    /// protocol intercepts these requests before they reach a socket.
    let port: Int = {
        portAllocationLock.lock()
        defer { portAllocationLock.unlock() }
        nextFakePort += 1
        return nextFakePort
    }()

    var baseURL: URL { ChatStreamClient.loopbackURL(port: port) }

    func session() -> URLSession {
        SSEProtocol.register(self, port: port)
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [SSEProtocol.self]
        return URLSession(configuration: configuration)
    }

    // MARK: - Request parsing

    fileprivate static func lastUserText(in body: Data) -> String? {
        guard
            let object = try? JSONSerialization.jsonObject(with: body) as? [String: Any],
            let messages = object["messages"] as? [[String: Any]]
        else { return nil }
        for message in messages.reversed() where (message["role"] as? String) == "user" {
            if let text = message["content"] as? String { return text }
            if let parts = message["content"] as? [[String: Any]] {
                // OpenAI content-parts form.
                return parts.compactMap { $0["text"] as? String }.joined(separator: " ")
            }
            return nil
        }
        return nil
    }

    /// The chunk list one request should stream. Mirrors the bash fake's
    /// `_shape_for`: a matched shape streams exactly once — `contentRepeat`
    /// (like `FAKE_CONTENT_REPEAT`) applies only to the default reply, so a
    /// Stop journey's endless default stream coexists with same-session
    /// shaped turns that must settle.
    fileprivate func chunksForRequest(withLastUserText text: String) -> [String] {
        for (marker, chunks) in Self.responseShapes where text.contains(marker) {
            return chunks
        }
        return Array(
            repeating: Self.contentChunks,
            count: max(1, contentRepeat)
        ).flatMap { $0 }
    }

    // MARK: - SSE encoding

    fileprivate static func sse(_ payload: [String: Any]) -> Data {
        let json = try! JSONSerialization.data(withJSONObject: payload)
        return Data("data: ".utf8) + json + Data("\n\n".utf8)
    }

    fileprivate static func delta(
        content: String? = nil,
        reasoning: String? = nil,
        finish: String? = nil
    ) -> [String: Any] {
        var delta: [String: Any] = [:]
        if let content { delta["content"] = content }
        if let reasoning { delta["reasoning_content"] = reasoning }
        let finishValue: Any = finish ?? NSNull()
        return ["choices": [["delta": delta, "finish_reason": finishValue]]]
    }

    fileprivate static func toolCallDelta(id: String) -> [String: Any] {
        let arguments = try! JSONSerialization.data(
            withJSONObject: ["query": "golden tool loop evidence"]
        )
        return [
            "choices": [
                [
                    "delta": [
                        "tool_calls": [
                            [
                                "index": 0,
                                "id": id,
                                "type": "function",
                                "function": [
                                    "name": "web_search",
                                    "arguments": String(decoding: arguments, as: UTF8.self),
                                ],
                            ]
                        ]
                    ],
                    "finish_reason": "tool_calls",
                ]
            ]
        ]
    }

    // MARK: - The URLProtocol

    final class SSEProtocol: URLProtocol, @unchecked Sendable {
        /// Registry entries hold the fake weakly: registration happens once
        /// per `session()` and is never explicitly undone, so a strong entry
        /// would keep every fake alive for the life of the test process.
        private final class WeakFake {
            weak var fake: GoldenChatFake?
            init(_ fake: GoldenChatFake) { self.fake = fake }
        }

        private static let registryLock = NSLock()
        nonisolated(unsafe) private static var registry: [Int: WeakFake] = [:]

        static func register(_ fake: GoldenChatFake, port: Int) {
            registryLock.lock()
            registry = registry.filter { $0.value.fake != nil }
            registry[port] = WeakFake(fake)
            registryLock.unlock()
        }

        private static func fake(forPort port: Int?) -> GoldenChatFake? {
            registryLock.lock()
            defer { registryLock.unlock() }
            guard let port else { return nil }
            return registry[port]?.fake
        }

        /// Guards `stopped` AND every client callback: `stopLoading` flips
        /// the flag under the same lock the emitter holds while delivering,
        /// so no callback can land after the loading system said stop.
        /// Recursive because a client callback may synchronously re-enter
        /// `stopLoading` on the delivering thread (e.g. a cancel triggered
        /// by the bytes it just received); a plain lock would deadlock there.
        private let stateLock = NSRecursiveLock()
        private var stopped = false

        override class func canInit(with request: URLRequest) -> Bool { true }
        override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

        /// URLSession surfaces an upload body as either `httpBody` or a
        /// stream depending on how the request was built; missing one form
        /// would silently drop the request witness for those requests.
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
            guard let fake = Self.fake(forPort: request.url?.port) else {
                client?.urlProtocol(
                    self,
                    didFailWithError: URLError(.cannotFindHost)
                )
                return
            }
            let body = requestBody() ?? Data()
            fake.record(body: body)

            let response = HTTPURLResponse(
                url: request.url!,
                statusCode: 200,
                httpVersion: "HTTP/1.1",
                headerFields: ["Content-Type": "text/event-stream"]
            )!
            deliver { $0.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed) }

            // Emit off the loading thread so `stopLoading` — dispatched to
            // this same work queue — can interleave with a paced stream
            // instead of queuing behind the whole emission.
            DispatchQueue.global(qos: .userInitiated).async { [self] in
                emit(fake: fake, body: body)
            }
        }

        override func stopLoading() {
            stateLock.lock()
            stopped = true
            stateLock.unlock()
        }

        /// Runs one client callback unless loading has been stopped.
        /// Returns false once stopped so emission loops can bail.
        @discardableResult
        private func deliver(_ callback: (URLProtocolClient) -> Void) -> Bool {
            stateLock.lock()
            defer { stateLock.unlock() }
            guard !stopped, let client else { return false }
            callback(client)
            return true
        }

        private func emit(fake: GoldenChatFake, body: Data) {
            let lastUser = GoldenChatFake.lastUserText(in: body) ?? ""
            let delay = fake.interChunkDelay

            // Deterministic runaway-model fixture: the app must execute only
            // its bounded budget, then issue one final request with no tools;
            // that request gets a synthesis rather than another tool call.
            if lastUser.contains("shape:tool-loop") {
                let object = try? JSONSerialization.jsonObject(with: body) as? [String: Any]
                let messages = (object?["messages"] as? [[String: Any]]) ?? []
                let toolResults = messages.filter { ($0["role"] as? String) == "tool" }.count
                let hasTools = ((object?["tools"] as? [Any]) ?? []).isEmpty == false
                if hasTools {
                    let callID = "golden_loop_\(toolResults + 1)"
                    deliver {
                        // Recorded before `[DONE]` becomes observable so a
                        // consumer that stops reading at `[DONE]` never
                        // races the event log.
                        fake.record(event: .toolLoopCall(id: callID))
                        $0.urlProtocol(
                            self,
                            didLoad: GoldenChatFake.sse(GoldenChatFake.toolCallDelta(id: callID))
                        )
                        $0.urlProtocol(self, didLoad: Data("data: [DONE]\n\n".utf8))
                        $0.urlProtocolDidFinishLoading(self)
                    }
                } else {
                    deliver {
                        fake.record(event: .toolLoopSynthesis(toolResults: toolResults))
                        $0.urlProtocol(
                            self,
                            didLoad: GoldenChatFake.sse(
                                GoldenChatFake.delta(
                                    content: GoldenChatFake.toolLoopSynthesisText,
                                    finish: "stop"
                                )
                            )
                        )
                        $0.urlProtocol(self, didLoad: Data("data: [DONE]\n\n".utf8))
                        $0.urlProtocolDidFinishLoading(self)
                    }
                }
                return
            }

            // Native web-tool fixture (`RAPID_GUI_WEB_SEARCH_FIXTURE`): on
            // each new user turn the fake model chooses the advertised
            // `web_search` tool, then synthesizes normally once the app has
            // appended that call's result.
            if fake.nativeWebSearchFixture {
                let object = try? JSONSerialization.jsonObject(with: body) as? [String: Any]
                let messages = (object?["messages"] as? [[String: Any]]) ?? []
                let advertisesWebSearch = ((object?["tools"] as? [[String: Any]]) ?? [])
                    .contains {
                        (($0["function"] as? [String: Any])?["name"] as? String) == "web_search"
                    }
                let lastUserIndex = messages.lastIndex { ($0["role"] as? String) == "user" }
                if advertisesWebSearch, let lastUserIndex {
                    let hasResultForTurn = messages[(lastUserIndex + 1)...]
                        .contains { ($0["role"] as? String) == "tool" }
                    if !hasResultForTurn {
                        let callID = "golden_search_\(lastUserIndex)"
                        deliver {
                            fake.record(event: .nativeWebSearchCall(id: callID))
                            $0.urlProtocol(
                                self,
                                didLoad: GoldenChatFake.sse(
                                    GoldenChatFake.toolCallDelta(id: callID)
                                )
                            )
                            $0.urlProtocol(self, didLoad: Data("data: [DONE]\n\n".utf8))
                            $0.urlProtocolDidFinishLoading(self)
                        }
                        return
                    }
                }
            }

            var contentEmitted = 0
            for reasoning in GoldenChatFake.reasoningChunks {
                guard deliver({
                    $0.urlProtocol(
                        self,
                        didLoad: GoldenChatFake.sse(GoldenChatFake.delta(reasoning: reasoning))
                    )
                }) else {
                    fake.record(event: .chatCancelled(chunks: contentEmitted))
                    return
                }
                if delay > 0 { Thread.sleep(forTimeInterval: delay) }
            }
            for chunk in fake.chunksForRequest(withLastUserText: lastUser) {
                guard deliver({
                    $0.urlProtocol(
                        self,
                        didLoad: GoldenChatFake.sse(GoldenChatFake.delta(content: chunk))
                    )
                }) else {
                    fake.record(event: .chatCancelled(chunks: contentEmitted))
                    return
                }
                contentEmitted += 1
                if delay > 0 { Thread.sleep(forTimeInterval: delay) }
            }
            guard deliver({
                // Recorded before `[DONE]` becomes observable so a consumer
                // that stops reading at `[DONE]` never races the event log.
                fake.record(event: .chatFinished(chunks: contentEmitted))
                $0.urlProtocol(
                    self,
                    didLoad: GoldenChatFake.sse(GoldenChatFake.delta(finish: "stop"))
                )
                $0.urlProtocol(self, didLoad: Data("data: [DONE]\n\n".utf8))
                $0.urlProtocolDidFinishLoading(self)
            }) else {
                fake.record(event: .chatCancelled(chunks: contentEmitted))
                return
            }
        }
    }
}

// MARK: - Shared chat-surface mount

/// The chat surface the `driver: swift` golden journeys share: the real
/// ``ChatView`` mounted on a ``GoldenStage`` the way ``ContentView`` hosts
/// it, with the smallest honest dependency set — a ready fake server, a
/// ``GoldenChatFake``-backed SSE client, and throwaway stores. No
/// conversation persists.
@MainActor
struct GoldenChatSurface {
    static let alias = "fake-alias"

    let stage: GoldenStage
    let chat: ChatViewModel
    let server: ServerManager
    let fake: GoldenChatFake

    static func mount(
        fake: GoldenChatFake = GoldenChatFake(),
        tools: (any ToolRegistry)? = nil
    ) -> GoldenChatSurface {
        let (server, chat) = assemble(fake: fake, tools: tools, conversationStoreURL: nil)
        let view = ChatView(
            viewModel: chat,
            server: server,
            alias: .constant(alias),
            readiness: .ready(alias: alias)
        )
        .environment(DownloadManager())
        .environment(QuickstartCoordinator())

        let stage = GoldenStage(view)
        return GoldenChatSurface(stage: stage, chat: chat, server: server, fake: fake)
    }

    /// The chat surface plus the real ``SidebarView``, composed the way
    /// ``ContentView`` composes them (row press → `chat.selectConversation`)
    /// — for restore journeys that must walk back into a persisted
    /// conversation through the same rows a user presses. Passing a
    /// `conversationStoreURL` turns persistence on against that store;
    /// mounting a second surface over the same URL is the in-process
    /// analog of the bash harness's `relaunch_persona`.
    static func mountWithSidebar(
        fake: GoldenChatFake = GoldenChatFake(),
        tools: (any ToolRegistry)? = nil,
        conversationStoreURL: URL
    ) -> GoldenChatSurface {
        let (server, chat) = assemble(
            fake: fake,
            tools: tools,
            conversationStoreURL: conversationStoreURL
        )
        let view = SidebarChatHarnessView(chat: chat, server: server)
            .environment(DownloadManager())
            .environment(QuickstartCoordinator())

        let stage = GoldenStage(view)
        return GoldenChatSurface(stage: stage, chat: chat, server: server, fake: fake)
    }

    private static func assemble(
        fake: GoldenChatFake,
        tools: (any ToolRegistry)?,
        conversationStoreURL: URL?
    ) -> (ServerManager, ChatViewModel) {
        // The server publishes the fake's port so ChatViewModel's
        // before-send re-target (`client.baseURL = loopback(activePort)`)
        // lands back on this fake instead of a real engine port.
        let server = ServerManager(
            testingState: .ready(alias: alias),
            activePort: fake.port
        )
        let chat = ChatViewModel(
            client: ChatStreamClient(baseURL: fake.baseURL, session: fake.session()),
            tools: tools ?? EmptyToolRegistry(),
            // A fresh, never-written suite: tools stay at their built-in
            // enabled default regardless of this machine's real settings.
            toolDefaults: UserDefaults(suiteName: "golden-surface-\(UUID().uuidString)")!,
            server: server,
            persistsConversations: conversationStoreURL != nil,
            conversationStoreURL: conversationStoreURL
        )
        return (server, chat)
    }

    /// `send_prompt` from the bash harness: type into the composer via AX
    /// set-value, press send, then require BOTH the drained composer and
    /// the recorded request — the composer clearing is the app's story
    /// about itself; the recorded body is the independent witness that a
    /// request actually left the process.
    func sendPrompt(_ prompt: String) async throws {
        let requestsBefore = fake.recordedBodies().count
        try stage.setValue(prompt, for: "rapid.chat.compose")
        try stage.press("ChatView.SendOrStopButton")
        try await stage.wait(for: "composer to drain and the request to be recorded") {
            stage.value(of: "rapid.chat.compose") == ""
                && fake.recordedBodies().count > requestsBefore
        }
    }

    /// `wait_send_idle` from the bash harness: the drained text can land a
    /// beat before the stream formally completes, and several message
    /// actions are disabled while streaming — a press on a disabled control
    /// no-ops silently. The send button relabelling back from
    /// "Stop generating" is the AX-visible idle signal.
    func waitForSendIdle(timeout: TimeInterval = GoldenStage.defaultTimeout) async throws {
        try await stage.wait(
            for: "composer to settle into a ready, non-streaming state",
            timeout: timeout
        ) {
            stage.tree().contains {
                $0.id == "ChatView.SendOrStopButton" && $0.text == "Send message"
            }
        }
    }
}

/// Sidebar + chat detail in the same composition ``ContentView`` ships:
/// row selection routes through `chat.selectConversation`, new-chat through
/// `chat.newConversation`. The window-level affordances a borderless stage
/// cannot host (toolbar search, native panels) stay with the bash/XCUI
/// journeys that own them.
private struct SidebarChatHarnessView: View {
    @State private var section: SidebarSection = .chat
    let chat: ChatViewModel
    let server: ServerManager

    var body: some View {
        NavigationSplitView {
            SidebarView(
                selection: $section,
                chat: chat,
                onNewChat: {
                    chat.newConversation()
                    section = .chat
                },
                onSelectConversation: { id in
                    chat.selectConversation(id)
                    section = .chat
                },
                server: server
            )
        } detail: {
            ChatView(
                viewModel: chat,
                server: server,
                alias: .constant(GoldenChatSurface.alias),
                readiness: .ready(alias: GoldenChatSurface.alias)
            )
        }
    }
}

// MARK: - Shared markdown structure assertions

/// The bash harness's markdown structural checks
/// (`assert_code_block_is_its_own_view`,
/// `assert_rendered_as_separate_nodes`), against a ``GoldenStage`` tree.
@MainActor
enum GoldenMarkdownAssertions {

    /// A fenced block is its own view, not a paragraph that happens to
    /// contain code. If a refactor flattens that, the code still
    /// "appears" — as a wrapped, unindented, uncopyable smear.
    static func assertCodeBlockIsItsOwnView(
        prose: String,
        code: String,
        in tree: [GoldenStage.Node]
    ) {
        guard tree.contains(where: { $0.text.contains(prose) }) else {
            Issue.record("prose not found: \(prose)")
            return
        }
        guard let codeNode = tree.first(where: { $0.text.contains(code) }) else {
            Issue.record("code not found: \(code)")
            return
        }
        #expect(
            !codeNode.text.contains(prose),
            "code block was flattened into the prose accessibility node"
        )
        #expect(codeNode.text.contains("\n"), "code block lost its line breaks")
    }

    /// `assert_markdown_code_and_table`: the settled code turn keeps its
    /// own indented block, no markdown syntax reaches the screen verbatim,
    /// and the settled table turn keeps its navigable representation.
    ///
    /// The bash flow asserted per-cell AX values (`AXCell` text) through
    /// the real AX server; in-process, a SwiftUI `Table` resolves cell
    /// values only via the server's parameterized queries, so this asserts
    /// the structure the process CAN see — the labelled `AXOutline`
    /// representation with the fixture's exact row/column shape — while
    /// cell text extraction stays pinned by `MarkdownTableAccessibilityTests`.
    static func assertMarkdownCodeAndTable(on stage: GoldenStage) {
        let tree = stage.tree()
        assertCodeBlockIsItsOwnView(
            prose: "Here is the function you asked for",
            code: "def fib(n)",
            in: tree
        )
        #expect(
            tree.contains { $0.text.contains("    return a") },
            "code block lost its indentation"
        )

        // `assert_markdown_rendered`: the loudest regression is the renderer
        // falling back to plain text — every "text appears" assertion passes
        // on that, wearing its syntax.
        #expect(
            !tree.contains { $0.text.contains("```") },
            "a code fence reached the screen verbatim — markdown was printed, not rendered"
        )
        #expect(
            !tree.contains {
                $0.text.range(of: #"\| *-{2,} *\|"#, options: .regularExpression) != nil
            },
            "a table separator row reached the screen verbatim — the table was not rendered"
        )

        #expect(
            tree.contains { $0.role == "AXOutline" && $0.text == "Markdown table" },
            "markdown table lost its AXOutline container"
        )
        let structure = stage.tableStructure()
        #expect(
            structure?.rows == 2 && structure?.columns == 3,
            "markdown table lost its navigable row/column structure: \(String(describing: structure))"
        )
    }
}
