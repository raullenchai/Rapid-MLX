import Foundation
import Testing
@testable import Rapid

@MainActor
@Suite("Request-time image capability", .serialized)
struct ChatWireImageCapabilityTests {
    @Test("Live lane overrides the submitted snapshot in both request builders",
          arguments: [false, true])
    func liveLaneControlsEncodedRequest(withSampling: Bool) async throws {
        let alias = "wire-image-test"
        let server = ServerManager(testingState: .ready(alias: alias))
        let model = ChatViewModel(
            client: ChatStreamClient(session: WireImageProtocol.session()),
            sampling: withSampling ? SamplingConfig() : nil,
            server: server,
            persistsConversations: false
        )
        let image = try ChatImageAttachment(
            filename: "photo.png", mimeType: "image/png", data: Data("test-photo".utf8)
        )
        // Admission accepts the image, then startup reports a degraded text lane.
        server.applyActiveModelProfile(ServerModelProfile(
            id: alias, capabilities: ["text"], servingLane: "text"
        ), forAlias: alias)
        WireImageProtocol.reset()
        model.send("Describe this", alias: alias, supportsImageInput: true,
                   imageAttachments: [image])
        try await waitForStream(model)
        let textBody = try #require(WireImageProtocol.lastBody())
        #expect(!String(decoding: textBody, as: UTF8.self).contains("image_url"))

        // A later request must read the new lane again even when the submitted
        // capability is false.
        server.applyActiveModelProfile(ServerModelProfile(
            id: alias, capabilities: ["text", "vision"], servingLane: "vision"
        ), forAlias: alias)
        WireImageProtocol.reset()
        model.send("Now inspect the photo", alias: alias, supportsImageInput: false,
                   imageAttachments: [image])
        try await waitForStream(model)
        let visionBody = try #require(WireImageProtocol.lastBody())
        #expect(String(decoding: visionBody, as: UTF8.self).contains(image.data.base64EncodedString()))
    }

    @Test("Clients without an embedded server retain the submitted capability")
    func isolatedClientFallback() {
        let model = ChatViewModel(persistsConversations: false)
        #expect(model.imageInputCapabilityForWire(alias: "vision", submittedCapability: true))
        #expect(!model.imageInputCapabilityForWire(alias: "vision", submittedCapability: false))
    }

    @Test("Missing and legacy profiles preserve catalog and launch fallback")
    func missingProfileFallback() {
        for profile in [nil, ServerModelProfile(id: "vision")] {
            #expect(ImageInputAvailability.resolve(
                fallbackSupportsImageInput: true, profile: profile
            ).isAvailable)
            #expect(!ImageInputAvailability.resolve(
                fallbackSupportsImageInput: false, profile: profile
            ).isAvailable)
        }
        let server = ServerManager(testingState: .ready(alias: "unknown"), binaryPath: nil)
        let model = ChatViewModel(server: server, persistsConversations: false)
        // No trusted catalog or live lane must not inherit a stale UI true.
        #expect(!model.imageInputCapabilityForWire(alias: "unknown", submittedCapability: true))
        #expect(server.supportsImageInput(forAlias: "unknown", catalogSupportsImageInput: true))
    }

    private func waitForStream(_ model: ChatViewModel) async throws {
        for _ in 0..<500 {
            if !model.isStreaming { return }
            try await Task.sleep(for: .milliseconds(10))
        }
        model.stop()
        Issue.record("Image request did not complete")
    }
}

private final class WireImageProtocol: URLProtocol, @unchecked Sendable {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var body: Data?

    static func reset() {
        lock.lock()
        defer { lock.unlock() }
        body = nil
    }

    static func lastBody() -> Data? {
        lock.lock()
        defer { lock.unlock() }
        return body
    }

    static func session() -> URLSession {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [WireImageProtocol.self]
        return URLSession(configuration: configuration)
    }

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        var data = request.httpBody ?? Data()
        if let stream = request.httpBodyStream {
            stream.open()
            defer { stream.close() }
            var buffer = [UInt8](repeating: 0, count: 4096)
            while true {
                let count = stream.read(&buffer, maxLength: buffer.count)
                if count <= 0 { break }
                data.append(buffer, count: count)
            }
        }
        Self.lock.lock()
        Self.body = data
        Self.lock.unlock()
        let response = HTTPURLResponse(
            url: request.url!, statusCode: 200, httpVersion: "HTTP/1.1",
            headerFields: ["Content-Type": "text/event-stream"]
        )!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: Data("""
            data: {"choices":[{"delta":{"content":"Done"},"finish_reason":"stop"}]}

            data: [DONE]

            """.utf8))
        client?.urlProtocolDidFinishLoading(self)
    }
    override func stopLoading() {}
}
