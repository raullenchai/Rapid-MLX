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
    @Test("The speed test and chat resolve to the same endpoint")
    func benchmarkAndChatAgreeOnTheEndpoint() throws {
        let base = ChatStreamClient.loopbackURL(port: 8123)
        let benchmark = try BenchmarkRunner.loadedBenchmarkRequest(
            baseURL: base, bearer: "", alias: "a", maxTokens: 8, prompt: "p")
        #expect(benchmark.url == ChatStreamClient.chatCompletionsURL(base: base))
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
