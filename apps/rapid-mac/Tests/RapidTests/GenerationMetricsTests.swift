import Foundation
import Testing
import SwiftUI
@testable import Rapid

@Suite("Frozen generation metrics", .serialized) @MainActor
struct GenerationMetricsTests {
    private func fixtures() throws -> [[String: Any]] {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        return try #require(JSONSerialization.jsonObject(with: Data(contentsOf: root.appendingPathComponent("tests/fixtures/decode_execution_golden.json"))) as? [[String: Any]])
    }
    private func metricsData(_ index: Int = 0) throws -> Data {
        try JSONSerialization.data(withJSONObject: fixtures()[index]["metrics"]!)
    }

    @Test func sharedVectorsRequireExecutedConfigAndHost() throws {
        for row in try fixtures() {
            let bytes = try JSONSerialization.data(withJSONObject: row["metrics"]!)
            let metrics = try? JSONDecoder().decode(GenerationMetrics.self, from: bytes)
            #expect((metrics != nil) == (row["valid"] as? Bool), "\(row["name"]!)")
            guard let metrics else { continue }
            #expect((metrics.cohort() != nil) == (row["cohort"] as? Bool))
            #expect(metrics.decodeTokensPerSecond == (row["rate"] as? Double))
            let saved = String(decoding: try JSONEncoder().encode(metrics), as: UTF8.self)
            #expect(!saved.contains("private_path") && !saved.contains("hostname"))
        }
    }

    @Test func historyKeepsFrozenFactsAndOldTranscriptsStayCompatible() throws {
        let metrics = try JSONDecoder().decode(GenerationMetrics.self, from: metricsData())
        var message = ChatMessage(role: .assistant, content: "answer", status: .complete)
        message.generationMetrics = metrics
        let bytes = try JSONEncoder().encode(message)
        #expect(try JSONDecoder().decode(ChatMessage.self, from: bytes).generationMetrics == metrics)
        var old = try #require(JSONSerialization.jsonObject(with: bytes) as? [String: Any])
        old.removeValue(forKey: "generationMetrics")
        #expect(try JSONDecoder().decode(ChatMessage.self, from: JSONSerialization.data(withJSONObject: old)).generationMetrics == nil)
        old["generationMetrics"] = ["private": "unsupported"]
        #expect(try JSONDecoder().decode(ChatMessage.self, from: JSONSerialization.data(withJSONObject: old)).content == "answer")
        #expect(metrics.detailText.contains("M2 Pro") && metrics.detailText.contains("n=3"))
        #expect(!metrics.detailText.lowercased().contains("below"))
    }

    @Test func malformedOptionalMetricsCannotDropFinishOrContent() throws {
        let data = Data(#"{"choices":[{"delta":{"content":"answer"},"finish_reason":"stop"}],"metrics":{"private":"bad"}}"#.utf8)
        let chunk = try JSONDecoder().decode(Wire.StreamChunk.self, from: data)
        #expect(chunk.choices.first?.delta?.content == "answer")
        #expect(chunk.choices.first?.finish_reason == "stop")
        #expect(chunk.metrics == nil)
    }

    @Test func actualStreamKeepsFinishMetricsBeforeSeparateUsageAndToolFinish() async throws {
        let metricJSON = String(decoding: try metricsData(), as: UTF8.self)
        for finish in ["stop", "tool_calls"] {
            let body = "data: {\"choices\":[{\"delta\":{\"content\":\"answer\"},\"finish_reason\":null}]}\n\n"
                + "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"\(finish)\"}],\"metrics\":\(metricJSON)}\n\n"
                + "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":4}}\n\ndata: [DONE]\n\n"
            GenerationMetricsProtocol.install(body: Data(body.utf8), stream: true)
            let client = ChatStreamClient(baseURL: URL(string: "http://localhost:18000")!, session: GenerationMetricsProtocol.session())
            var order: [String] = []
            var received: GenerationMetrics?
            try await client.send(.init(alias: "alias", messages: [ChatMessage(role: .user, content: "question")])) { event in
                switch event {
                case .executionMetrics(let value): received = value; order.append("metrics")
                case .usage: order.append("usage")
                case .finished: order.append("finish")
                default: break
                }
            }
            #expect(received?.executionHost?.chip == "M2 Pro")
            #expect(order == ["metrics", "usage", "finish"])
        }
    }

    @Test func actualDirectModelBindsGenerationAndRejectsMismatchedUsage() async throws {
        let metricJSON = String(decoding: try metricsData(), as: UTF8.self)
        for count in [4, 40] {
            let body = "data: {\"choices\":[{\"delta\":{\"content\":\"answer\"},\"finish_reason\":null}]}\n\n"
                + "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"metrics\":\(metricJSON)}\n\n"
                + "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":\(count)}}\n\ndata: [DONE]\n\n"
            GenerationMetricsProtocol.install(body: Data(body.utf8), stream: true)
            let model = ChatViewModel(client: ChatStreamClient(baseURL: URL(string: "http://localhost:18000")!, session: GenerationMetricsProtocol.session()), persistsConversations: false)
            model.send("question", alias: "alias")
            for _ in 0..<200 where model.isStreaming { try await Task.sleep(for: .milliseconds(10)) }
            #expect(!model.isStreaming)
            let answer = try #require(model.messages.last)
            #expect(answer.content == "answer")
            #expect((answer.generationMetrics != nil) == (count == 4))
            #expect(answer.stats?.completionTokens == count)
        }
    }

    @Test(arguments: [520.0, 720.0]) func nativeDetailsUseFrozenFixtureHost(width: Double) async throws {
        let metricJSON = String(decoding: try metricsData(), as: UTF8.self)
        let body = "data: {\"choices\":[{\"delta\":{\"content\":\"A saved answer.\"},\"finish_reason\":\"stop\"}],\"metrics\":\(metricJSON)}\n\n"
            + "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":4}}\n\ndata: [DONE]\n\n"
        GenerationMetricsProtocol.install(body: Data(body.utf8), stream: true)
        let model = ChatViewModel(client: ChatStreamClient(baseURL: URL(string: "http://localhost:18000")!, session: GenerationMetricsProtocol.session()), persistsConversations: false)
        model.send("question", alias: "alias-unrelated-to-executed-repo")
        for _ in 0..<200 where model.isStreaming { try await Task.sleep(for: .milliseconds(10)) }
        #expect(model.messages.last?.generationMetrics?.executionHost?.chip == "M2 Pro")
        let stage = GoldenStage(
            ChatView(viewModel: model, server: ServerManager(testingState: .idle), alias: .constant("changed-picker-alias"), readiness: .ready(alias: "changed-picker-alias"))
                .environment(DownloadManager()).environment(QuickstartCoordinator()),
            size: CGSize(width: width, height: 620)
        )
        try await stage.waitForIdentifier("Chat.GenerationDetails")
        try stage.press("Chat.GenerationDetails")
        try await stage.waitForText("Execution host: M2 Pro")
        // AX exposes disclosure text before the native expand animation ends.
        try await Task.sleep(for: .milliseconds(350))
        #expect(stage.treeText().contains("n=3"))
        if let directory = ProcessInfo.processInfo.environment["RAPID_GENERATION_REVIEW_DIR"] {
            try stage.writeScreenshot(to: URL(fileURLWithPath: directory).appendingPathComponent("generation-fixture-\(Int(width)).png"))
        }
    }

    @Test func ownerSessionBindsSavedFinalResponseWithoutChangingStrictEvents() async throws {
        let fixture = OwnerChatSessionTests()
        let metrics = try JSONDecoder().decode(GenerationMetrics.self, from: metricsData())
        for outcome in ["completed", "cancelled", "error"] {
            for messageID in ["final", "different"] {
                let client = try await fixture.fake([
                    fixture.frame(#"{"type":"text.delta","part_id":"p","text":"answer"}"#, 1),
                    fixture.frame("{\"type\":\"turn.terminal\",\"outcome\":\"\(outcome)\",\"final_message_id\":\"final\",\"usage\":{\"input_tokens\":100,\"output_tokens\":400,\"requests\":8}}", 2),
                ])
                await client.setGeneration(metrics, messageID: messageID)
                let session = try await fixture.start(client)
                for _ in 0..<100 {
                    if await session.snapshot().state == .finished { break }
                    try await Task.sleep(for: .milliseconds(10))
                }
                let snapshot = await session.snapshot()
                let projection = try #require(snapshot.projection)
                let rows = ownerProjectionMessages(projection, assistantID: UUID(), createdAt: Date(), generation: snapshot.generation)
                let saved = rows.first?.generationMetrics
                #expect((saved != nil) == (outcome == "completed" && messageID == "final"))
                #expect(await client.generationReads == (outcome == "completed" ? 1 : 0))
                if let saved { #expect(saved.executionReference.completionTokens == 4) }
                await session.close()
            }
        }
    }

    @Test func staleMetricsReadCannotFinishANewerTurnAfterClose() async throws {
        actor Gate {
            var first = true
            var continuation: CheckedContinuation<Void, Never>?
            func wait() async {
                guard first else { return }
                first = false
                await withCheckedContinuation { continuation = $0 }
            }
            func release() { continuation?.resume(); continuation = nil }
        }
        let gate = Gate()
        let fixture = OwnerChatSessionTests()
        let client = try await fixture.fake([
            fixture.frame(#"{"type":"text.delta","part_id":"p","text":"old answer"}"#, 1),
            fixture.frame(#"{"type":"turn.terminal","outcome":"completed","final_message_id":"final"}"#, 2),
        ])
        await client.setGenerationGate { await gate.wait() }
        let session = try await fixture.start(client)
        for _ in 0..<100 {
            if await client.generationReads == 1 { break }
            try await Task.sleep(for: .milliseconds(10))
        }
        #expect(await client.generationReads == 1)
        await session.close()
        await client.setFrames([fixture.frame(#"{"type":"text.delta","part_id":"p","text":"new partial answer"}"#, 1)])
        try await session.submit(text: "New question", target: .init(url: URL(string: "http://localhost:8000")!, bearer: nil, model: "m"))
        for _ in 0..<100 {
            if await session.snapshot().projection?.parts.isEmpty == false { break }
            try await Task.sleep(for: .milliseconds(10))
        }
        await gate.release()
        try await Task.sleep(for: .milliseconds(30))
        let snapshot = await session.snapshot()
        #expect(snapshot.projection?.terminal == nil)
        #expect(snapshot.state != .finished)
        #expect(snapshot.generation == nil)
        await session.close()
    }

    @Test func oldOwnerNeverReadsUnsupportedEndpointAndNewOwnerBindsTurn() async throws {
        let client = try URLSessionChatTurnClient(endpoint: URL(string: "http://localhost:18001")!, bearer: "test-only", session: GenerationMetricsProtocol.session())
        GenerationMetricsProtocol.install(body: Data(#"{"capabilities":{}}"#.utf8))
        #expect(try await client.generationMetrics(conversationID: "c", turnID: "t") == nil)
        #expect(GenerationMetricsProtocol.paths() == ["/v1/rapid/chat/capabilities"])
        let metrics = String(decoding: try metricsData(), as: UTF8.self)
        let body = Data("{\"generation\":{\"turn_id\":\"t\",\"message_id\":\"m\",\"metrics\":\(metrics)}}".utf8)
        GenerationMetricsProtocol.install(body: body, capability: true)
        #expect(try await client.generationMetrics(conversationID: "c", turnID: "t")?.messageID == "m")
        #expect(GenerationMetricsProtocol.paths().last == "/v1/rapid/chat/conversations/c/turns/t/execution-metrics")
        await #expect(throws: ChatTurnClientError.invalidResponse) {
            try await client.generationMetrics(conversationID: "c", turnID: "wrong")
        }
    }
}

private final class GenerationMetricsProtocol: URLProtocol, @unchecked Sendable {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var body = Data()
    nonisolated(unsafe) private static var stream = false
    nonisolated(unsafe) private static var capability = false
    nonisolated(unsafe) private static var recorded: [String] = []
    static func install(body: Data, stream: Bool = false, capability: Bool = false) {
        lock.lock(); defer { lock.unlock() }
        self.body = body; self.stream = stream; self.capability = capability; recorded = []
    }
    static func paths() -> [String] { lock.lock(); defer { lock.unlock() }; return recorded }
    static func session() -> URLSession {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [GenerationMetricsProtocol.self]
        return URLSession(configuration: config)
    }
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        Self.lock.lock()
        Self.recorded.append(request.url!.path)
        let isStream = Self.stream
        let bytes = Self.capability && request.url!.path.hasSuffix("capabilities")
            ? Data(#"{"capabilities":{"generation_metrics_v1":true}}"#.utf8) : Self.body
        Self.lock.unlock()
        let response = HTTPURLResponse(url: request.url!, statusCode: 200, httpVersion: "HTTP/1.1", headerFields: ["Content-Type": isStream ? "text/event-stream" : "application/json"])!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: bytes)
        client?.urlProtocolDidFinishLoading(self)
    }
    override func stopLoading() {}
}

@Suite("Generation metrics real local user path") @MainActor
struct GenerationMetricsRealUserTests {
    @Test(.enabled(if: ProcessInfo.processInfo.environment["RAPID_FOLDER_REAL_OWNER_URL"] != nil))
    func directAndOwnerFirstChatKeepActualExecution() async throws {
        let env = ProcessInfo.processInfo.environment
        let ownerString = try #require(env["RAPID_FOLDER_REAL_OWNER_URL"])
        let ownerURL = try #require(URL(string: ownerString))
        let portString = try #require(env["RAPID_FOLDER_REAL_INFERENCE_PORT"])
        let port = try #require(Int(portString))
        let directory = try #require(env["RAPID_FOLDER_REVIEW_DIR"])
        var receipts: [[String: Any]] = []
        for owner in [false, true] {
            let defaults = UserDefaults(suiteName: UUID().uuidString)!
            let config = ChatBackendConfig(defaults: defaults)
            config.harnessEnabled = owner
            let ownerClient = try URLSessionChatTurnClient(endpoint: ownerURL, bearer: "owned-folder-test")
            let server = ServerManager(testingState: .ready(alias: "default"), activePort: port, activeBearer: "owned-inference-test", sessionDefaults: defaults)
            let model = ChatViewModel(client: ChatStreamClient(baseURL: URL(string: "http://127.0.0.1:\(port)")!), chatBackendConfig: config, ownerClientFactory: { ownerClient }, toolDefaults: defaults, server: server, persistsConversations: false)
            let stage = GoldenStage(ChatView(viewModel: model, server: server, alias: .constant("default"), readiness: .ready(alias: "default"))
                .environment(DownloadManager()).environment(QuickstartCoordinator()), size: CGSize(width: 720, height: 680))
            let question = "What do trees need to grow? Answer in one concise sentence."
            try await stage.waitForIdentifier(AutosizingTextView.composeAccessibilityIdentifier)
            try stage.setValue(question, for: AutosizingTextView.composeAccessibilityIdentifier)
            try stage.submitComposer()
            try await stage.wait(for: "real question admitted") { model.messages.contains { $0.role == .user && $0.content == question } }
            await model._testingWaitForCurrentTurn()
            #expect(model.lastError == nil)
            let answer = try #require(model.messages.last(where: { $0.role == .assistant && !$0.content.isEmpty }))
            let metrics = try #require(answer.generationMetrics)
            #expect(metrics.executionReference.repoID == "mlx-community/Qwen3.5-4B-MLX-4bit")
            #expect(metrics.executionReference.status == "known")
            #expect(metrics.executionHost != nil)
            #expect(metrics.decodeTokensPerSecond != nil)
            try await stage.waitForIdentifier("Chat.GenerationDetails")
            try stage.press("Chat.GenerationDetails")
            try await stage.waitForText("Execution host:")
            try await Task.sleep(for: .milliseconds(350))
            try stage.writeScreenshot(to: URL(fileURLWithPath: directory).appendingPathComponent("real-generation-\(owner ? "owner" : "direct").png"))
            let serialized = try JSONSerialization.jsonObject(with: JSONEncoder().encode(metrics))
            receipts.append(["lane": owner ? "owner" : "direct", "answer": answer.content, "metrics": serialized])
        }
        try JSONSerialization.data(withJSONObject: receipts, options: [.prettyPrinted, .sortedKeys]).write(to: URL(fileURLWithPath: directory).appendingPathComponent("real-generation-receipts.json"))
    }
}
