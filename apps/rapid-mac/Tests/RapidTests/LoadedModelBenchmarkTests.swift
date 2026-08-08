import Foundation
import Testing
@testable import Rapid

@Suite("Loaded-model speed test")
struct LoadedModelBenchmarkTests {
    /// The speed test and chat must reach the SAME endpoint.
    ///
    /// They are two callers of one local server, each of which used to build
    /// the path itself from a base URL that carries none. Chat appended
    /// `v1/chat/completions`; the benchmark appended `chat/completions`. Both
    /// "looked right" in isolation, and only chat was ever exercised against a
    /// real server. Pinning the equality — rather than either literal — is
    /// what makes the next divergence a test failure.
    ///
    /// The recorder below holds its captured URL in static state, which is safe
    /// only because this is its one and only writer. Give it a second test and
    /// it needs its own instance state, not a `.serialized` trait — that trait
    /// orders the cases of a parameterized test and does nothing here.
    @Test("The speed test and chat resolve to the same endpoint")
    @MainActor
    func benchmarkAndChatAgreeOnTheEndpoint() async throws {
        let base = ChatStreamClient.loopbackURL(port: 8123)

        let benchmark = try BenchmarkRunner.loadedBenchmarkRequest(
            baseURL: base, bearer: "", alias: "a", maxTokens: 8, prompt: "p")

        // Chat's URL is captured from a real ``send`` through the URLProtocol
        // seam, NOT recomputed from the same helper the benchmark calls.
        // Comparing two callers of one helper only proves the helper is
        // deterministic; it would stay green if ``send`` went back to building
        // its own path, which is precisely how the two drifted in the first
        // place.
        BenchmarkURLRecorderProtocol.reset()
        let cfg = URLSessionConfiguration.ephemeral
        cfg.protocolClasses = [BenchmarkURLRecorderProtocol.self] + (cfg.protocolClasses ?? [])
        let client = ChatStreamClient(baseURL: base, session: URLSession(configuration: cfg))
        _ = try? await client.send(
            ChatStreamClient.Request(
                alias: "a", messages: [ChatMessage(role: .user, content: "hi")])
        ) { _ in }
        let chatURL = BenchmarkURLRecorderProtocol.lastURL

        #expect(chatURL != nil, "the recorder saw no chat request at all")
        #expect(benchmark.url == chatURL, "the speed test and chat must hit one endpoint")
        #expect(
            benchmark.url?.absoluteString == "http://127.0.0.1:8123/v1/chat/completions",
            "the engine serves /v1/chat/completions; anything else 404s")
    }

    @Test("Speed test targets the current authenticated server without a model-loader command")
    func currentServerRequest() throws {
        // The base URL PRODUCTION passes — `ChatStreamClient.loopbackURL`,
        // which is host:port with NO path. This test used to hand in
        // ".../v1" instead, a value no call site ever produces, so it went
        // green while the shipped build POSTed to /chat/completions and got
        // a 404 on every run (#1668). A test that invents its own input
        // cannot fail when the caller is wrong.
        let request = try BenchmarkRunner.loadedBenchmarkRequest(
            baseURL: ChatStreamClient.loopbackURL(port: 8123),
            bearer: "test-secret",
            alias: "lfm2.5-8b-a1b-4bit",
            maxTokens: 128,
            prompt: "measure me"
        )

        #expect(request.url?.absoluteString == "http://127.0.0.1:8123/v1/chat/completions")
        #expect(request.httpMethod == "POST")
        #expect(request.value(forHTTPHeaderField: "Authorization") == "Bearer test-secret")
        let bodyData = try #require(request.httpBody)
        let body = try #require(
            JSONSerialization.jsonObject(with: bodyData) as? [String: Any])
        #expect(body["model"] as? String == "lfm2.5-8b-a1b-4bit")
        #expect(body["max_tokens"] as? Int == 128)
        #expect(body["stream"] as? Bool == false)
    }

    @Test("Speed test adds the OpenAI version path to the desktop server root")
    func desktopServerRootRequest() throws {
        let request = try BenchmarkRunner.loadedBenchmarkRequest(
            baseURL: URL(string: "http://127.0.0.1:8123")!,
            bearer: "",
            alias: "fake-alias",
            maxTokens: 8,
            prompt: "warm up"
        )

        #expect(request.url?.absoluteString == "http://127.0.0.1:8123/v1/chat/completions")
    }

    @Test("Displayed speed uses completion tokens over measured wall time")
    func completionSpeed() {
        let measurement = BenchmarkRunner.LoadedMeasurement(
            completionTokens: 120, elapsedSeconds: 4)
        #expect(measurement.tokensPerSecond == 30)
    }

    @Test("OpenAI usage supplies the measured completion count")
    func usageParsing() throws {
        let data = Data(#"{"usage":{"prompt_tokens":12,"completion_tokens":96,"total_tokens":108}}"#.utf8)
        #expect(try BenchmarkRunner.loadedCompletionTokens(from: data) == 96)
    }

    @Test("Missing completion usage is rejected instead of showing zero")
    func missingUsageRejected() {
        let data = Data(#"{"choices":[]}"#.utf8)
        #expect(throws: Error.self) {
            _ = try BenchmarkRunner.loadedCompletionTokens(from: data)
        }
    }
}


/// Captures the URL a real ``ChatStreamClient.send`` puts on the wire, and
/// answers with a terminating SSE frame so the client returns instead of
/// hanging on a bodyless 200.
final class BenchmarkURLRecorderProtocol: URLProtocol, @unchecked Sendable {
    nonisolated(unsafe) static var lastURL: URL?

    static func reset() { lastURL = nil }

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        Self.lastURL = request.url
        let response = HTTPURLResponse(
            url: request.url!, statusCode: 200, httpVersion: "HTTP/1.1",
            headerFields: ["Content-Type": "text/event-stream"])!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: Data("data: [DONE]\n\n".utf8))
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}
}
