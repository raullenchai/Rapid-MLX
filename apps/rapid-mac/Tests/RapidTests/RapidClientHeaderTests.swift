import Foundation
import Testing
@testable import Rapid

/// Every desktop HTTP client that talks to the bundled rapid-mlx sidecar
/// must identify itself with ``X-Rapid-Client: rapid-desktop``. Without the
/// header the server falls back to the User-Agent allow-list, where the
/// desktop is indistinguishable from whatever HTTP stack URLSession
/// advertises this OS release and lands in the ``other`` bucket.
///
/// Each case drives the real client through a ``URLProtocol`` stub and
/// asserts on the header the live request shape produces, so dropping the
/// ``applyRapidClientHeader()`` call from any one client fails here.
/// ``.serialized`` because the stub's recorded requests are process-wide
/// static state.
@Suite("X-Rapid-Client on every sidecar client", .serialized)
struct RapidClientHeaderTests {
    private func makeSession() -> URLSession {
        RapidClientHeaderStubProtocol.reset()
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [RapidClientHeaderStubProtocol.self]
        return URLSession(configuration: configuration)
    }

    /// The single assertion every case below shares.
    private func expectDesktopHeader(
        at index: Int = 0,
        _ label: Comment,
        sourceLocation: SourceLocation = #_sourceLocation
    ) throws {
        let requests = RapidClientHeaderStubProtocol.requests
        let request = try #require(
            requests.indices.contains(index) ? requests[index] : nil,
            label,
            sourceLocation: sourceLocation
        )
        #expect(
            request.value(forHTTPHeaderField: "X-Rapid-Client") == "rapid-desktop",
            label,
            sourceLocation: sourceLocation
        )
    }

    @Test("The label matches the closed set shared with the engine")
    func headerConstants() {
        #expect(RapidClientHeader.field == "X-Rapid-Client")
        #expect(RapidClientHeader.desktop == "rapid-desktop")
        var request = URLRequest(url: URL(string: "http://127.0.0.1:8123/healthz")!)
        request.applyRapidClientHeader()
        #expect(request.value(forHTTPHeaderField: "X-Rapid-Client") == "rapid-desktop")
    }

    @Test("ImageClient stamps the header")
    func imageClient() async throws {
        let client = ImageClient(session: makeSession())
        let png = Data("png".utf8)
        RapidClientHeaderStubProtocol.response = (
            200,
            Data(#"{"data":[{"b64_json":"\#(png.base64EncodedString())"}]}"#.utf8)
        )

        _ = try await client.generate(
            prompt: "a kite",
            model: "flux2-klein-4b",
            size: "512x512",
            count: 1,
            seed: nil,
            port: 8123,
            bearer: "secret"
        )

        try expectDesktopHeader("POST /v1/images/generations")
    }

    @Test("AudioClient stamps the header")
    func audioClient() async throws {
        let client = AudioClient(session: makeSession())
        RapidClientHeaderStubProtocol.response = (200, Data(#"{"voices":["af"]}"#.utf8))

        _ = try await client.voices(model: "kokoro", port: 8123, bearer: "secret")

        try expectDesktopHeader("GET /v1/audio/voices")
    }

    @Test("VideoClient stamps the header on both the shared builder and list")
    func videoClient() async throws {
        let client = VideoClient(session: makeSession())
        RapidClientHeaderStubProtocol.response = (200, Data(#"{"data":[]}"#.utf8))

        _ = try await client.list(port: 8123, bearer: "secret")
        // `list` builds its own request; everything else goes through the
        // shared `request(path:port:bearer:)` builder. Cover both.
        _ = try? await client.capabilities(port: 8123, bearer: "secret")

        try expectDesktopHeader(at: 0, "GET /v1/videos (ad-hoc request)")
        try expectDesktopHeader(at: 1, "GET /v1/videos/capabilities (shared builder)")
    }

    @Test("AgentRuntimeClient stamps the header")
    func agentRuntimeClient() async throws {
        let client = AgentRuntimeClient(
            baseURL: URL(string: "http://127.0.0.1:8123")!,
            session: makeSession()
        )
        let runID = "3f1a2b4c-5d6e-4f80-9a1b-2c3d4e5f6071"
        RapidClientHeaderStubProtocol.response = (
            200,
            Data(#"{"run_id":"\#(runID)","status":"ready","events":[],"next_after":0}"#.utf8)
        )

        _ = try await client.events(runID: runID, after: 0, bearerToken: "secret")

        try expectDesktopHeader("GET /v1/agent/runs/<id>/events")
    }

    @Test("ServerProfileFetcher stamps the header")
    func serverModelProfile() async throws {
        let session = makeSession()
        RapidClientHeaderStubProtocol.response = (404, Data())

        _ = await ServerProfileFetcher.fetch(
            baseURL: URL(string: "http://127.0.0.1:8123")!,
            alias: "qwen3.5-4b",
            bearer: "secret",
            session: session
        )

        try expectDesktopHeader("GET /v1/models/<alias>")
    }

    @Test("ServerResidencyClient stamps the header")
    func serverResidency() async throws {
        var client = ServerResidencyClient()
        client.session = makeSession()
        RapidClientHeaderStubProtocol.response = (404, Data())

        _ = await client.fetch(port: 8123, bearer: "secret")

        try expectDesktopHeader("GET /v1/models/residency")
    }

    @Test("ChatStreamClient stamps the header")
    @MainActor
    func chatStreamClient() async throws {
        let client = ChatStreamClient(
            baseURL: URL(string: "http://127.0.0.1:8123")!,
            session: makeSession()
        )
        RapidClientHeaderStubProtocol.response = (200, Data("data: [DONE]\n\n".utf8))

        // The stub replies with a bare [DONE]; the parser may surface that as
        // no content. Either way the header was captured pre-network.
        try? await client.send(
            ChatStreamClient.Request(
                alias: "qwen3.5-4b",
                messages: [ChatMessage(role: .user, content: "hi")]
            ),
            bearerToken: "secret"
        ) { _ in }

        try expectDesktopHeader("POST /v1/chat/completions")
    }
}

/// Records every outgoing request and replies with a canned status + body.
private final class RapidClientHeaderStubProtocol: URLProtocol, @unchecked Sendable {
    nonisolated(unsafe) static var requests: [URLRequest] = []
    nonisolated(unsafe) static var response: (Int, Data) = (200, Data())

    static func reset() {
        requests = []
        response = (200, Data())
    }

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        Self.requests.append(request)
        let response = HTTPURLResponse(
            url: request.url!,
            statusCode: Self.response.0,
            httpVersion: "HTTP/1.1",
            headerFields: ["Content-Type": "application/json"]
        )!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: Self.response.1)
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}
}
