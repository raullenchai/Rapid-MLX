import Foundation
import AppKit
import Testing
@testable import Rapid

@Suite("ImageClient wire contract", .serialized)
struct ImageClientTests {
    private func makeClient() -> ImageClient {
        ImageStubProtocol.reset()
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [ImageStubProtocol.self]
        return ImageClient(session: URLSession(configuration: config))
    }

    @Test("Request timeout leaves headroom for cold image model loads")
    func editTimeout() {
        #expect(ImageClient.requestTimeout >= 30 * 60)
    }

    @Test("Edit uploads the source and omits the unsupported size field")
    func editRequest() async throws {
        let client = makeClient()
        let resultPNG = Data("result-png".utf8)
        ImageStubProtocol.response = (
            200,
            Data(#"{"data":[{"b64_json":"\#(resultPNG.base64EncodedString())"}]}"#.utf8)
        )

        let results = try await client.edit(
            imagePNG: Data("source-png".utf8),
            prompt: "replace the sky",
            model: "flux2-klein-4b",
            count: 1,
            seed: 42,
            port: 8123,
            bearer: "secret"
        )

        #expect(results.map(\.pngData) == [resultPNG])
        #expect(results.first?.isEdit == true)
        let request = try #require(ImageStubProtocol.requests.first)
        #expect(request.url?.path == "/v1/images/edits")
        #expect(request.value(forHTTPHeaderField: "Authorization") == "Bearer secret")
        let body = String(decoding: try #require(ImageStubProtocol.bodies.first), as: UTF8.self)
        #expect(body.contains("name=\"prompt\"\r\n\r\nreplace the sky"))
        #expect(body.contains("name=\"model\"\r\n\r\nflux2-klein-4b"))
        #expect(body.contains("name=\"seed\"\r\n\r\n42"))
        #expect(body.contains("name=\"image\"; filename=\"input.png\""))
        #expect(body.contains("source-png"))
        #expect(!body.contains("name=\"size\""))
    }

    @Test("Nested FastAPI error detail reaches the Images UI")
    func nestedError() async throws {
        let client = makeClient()
        ImageStubProtocol.response = (
            409,
            Data(#"{"detail":{"error":{"message":"start an image-edit model"}}}"#.utf8)
        )

        do {
            _ = try await client.edit(
                imagePNG: Data([1]), prompt: "change it", model: "wrong-model",
                count: 1, seed: nil, port: 8123, bearer: nil
            )
            Issue.record("Expected the HTTP error")
        } catch let error as ImageClientError {
            #expect(error.errorDescription == "start an image-edit model")
        }
    }

    @Test("Imported edit images reject excessive pixel dimensions before decoding")
    func importPixelLimit() throws {
        let rep = NSBitmapImageRep(
            bitmapDataPlanes: nil,
            pixelsWide: EditImageImporter.maxDimension + 1,
            pixelsHigh: 1,
            bitsPerSample: 8,
            samplesPerPixel: 3,
            hasAlpha: false,
            isPlanar: false,
            colorSpaceName: .deviceRGB,
            bytesPerRow: 0,
            bitsPerPixel: 0
        )
        let data = try #require(rep?.representation(using: .png, properties: [:]))

        do {
            _ = try EditImageImporter.pngData(from: data)
            Issue.record("Expected the imported image dimensions to be rejected")
        } catch ImportedEditImageError.tooManyPixels {
            // Expected: metadata validation runs before full image decoding.
        }
    }
}

private final class ImageStubProtocol: URLProtocol, @unchecked Sendable {
    nonisolated(unsafe) static var requests: [URLRequest] = []
    nonisolated(unsafe) static var bodies: [Data] = []
    nonisolated(unsafe) static var response: (Int, Data) = (200, Data())

    static func reset() {
        requests = []
        bodies = []
        response = (200, Data())
    }

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        Self.requests.append(request)
        Self.bodies.append(Self.readBody(from: request))
        let reply = HTTPURLResponse(
            url: request.url!, statusCode: Self.response.0,
            httpVersion: "HTTP/1.1", headerFields: ["Content-Type": "application/json"]
        )!
        client?.urlProtocol(self, didReceive: reply, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: Self.response.1)
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}

    private static func readBody(from request: URLRequest) -> Data {
        if let body = request.httpBody { return body }
        guard let stream = request.httpBodyStream else { return Data() }
        stream.open()
        defer { stream.close() }
        var data = Data()
        var buffer = [UInt8](repeating: 0, count: 4096)
        while true {
            let count = stream.read(&buffer, maxLength: buffer.count)
            if count <= 0 { break }
            data.append(buffer, count: count)
        }
        return data
    }
}
