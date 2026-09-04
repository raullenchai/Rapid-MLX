import AppKit
import Foundation
import Testing
@testable import Rapid

@Suite("VideoClient wire contract", .serialized)
struct VideoClientTests {
    private func makeClient() -> VideoClient {
        VideoStubProtocol.reset()
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [VideoStubProtocol.self]
        return VideoClient(session: URLSession(configuration: configuration))
    }

    @Test("Capabilities authenticate and produce conservative controls")
    func capabilities() async throws {
        let client = makeClient()
        VideoStubProtocol.response = (200, Data(Self.capabilitiesJSON.utf8))

        let value = try await client.capabilities(port: 8123, bearer: "secret")

        #expect(value.modes == [.textToVideo, .imageToVideo])
        #expect(value.sizePresets == [
            "512x512", "768x512", "512x768", "1280x720", "720x1280",
        ])
        #expect(value.durationPresets(for: "512x512") == [1, 2, 4])
        #expect(value.durationPresets(for: "1280x720") == [1])
        #expect(value.referenceMaximumBytes == 20 * 1024 * 1024)
        let request = try #require(VideoStubProtocol.requests.first)
        #expect(request.url?.path == "/v1/videos/capabilities")
        #expect(request.value(forHTTPHeaderField: "Authorization") == "Bearer secret")
    }

    @Test("Duration presets include the shortest server-supported duration")
    func durationPresetsIncludeServerMinimum() throws {
        let json = Self.capabilitiesJSON.replacingOccurrences(
            of: #""minimum":1,"maximum":20,"default":4"#,
            with: #""minimum":3,"maximum":20,"default":4"#
        )
        let value = try JSONDecoder().decode(VideoCapabilities.self, from: Data(json.utf8))

        #expect(value.durationPresets(for: "512x512") == [3, 4])
    }

    @Test("Malformed capability ranges fail closed")
    func malformedCapabilitiesFailClosed() async {
        let client = makeClient()
        let json = Self.capabilitiesJSON.replacingOccurrences(
            of: #""minimum":1,"maximum":20,"default":4"#,
            with: #""minimum":5,"maximum":3,"default":4"#
        )
        VideoStubProtocol.response = (200, Data(json.utf8))

        await #expect(throws: VideoClientError.invalidResponse) {
            _ = try await client.capabilities(port: 8123, bearer: nil)
        }
    }

    @Test("Unrecognized dimension_rounding fails closed")
    func unsupportedRoundingFailsClosed() async {
        // A future alignment can be larger than the values this client knows.
        // Guessing 64 would under-count workload for e.g. ceil_to_128, so an
        // unknown value must reject the capabilities payload.
        let client = makeClient()
        let json = Self.capabilitiesJSON.replacingOccurrences(
            of: #""dimension_rounding":"ceil_to_64""#,
            with: #""dimension_rounding":"ceil_to_128""#
        )
        VideoStubProtocol.response = (200, Data(json.utf8))

        await #expect(throws: VideoClientError.invalidResponse) {
            _ = try await client.capabilities(port: 8123, bearer: nil)
        }
    }

    @Test("Malformed (blank) dimension_rounding fails closed")
    func blankRoundingFailsClosed() async {
        // An empty/blank rounding label is malformed and rejects too.
        let client = makeClient()
        let json = Self.capabilitiesJSON.replacingOccurrences(
            of: #""dimension_rounding":"ceil_to_64""#,
            with: #""dimension_rounding":""#
        )
        VideoStubProtocol.response = (200, Data(json.utf8))

        await #expect(throws: VideoClientError.invalidResponse) {
            _ = try await client.capabilities(port: 8123, bearer: nil)
        }
    }

    @Test("Workload uses 64-pixel rounding independently of size alignment")
    func workloadRoundingIsIndependent() throws {
        let json = Self.capabilitiesJSON
            .replacingOccurrences(of: #""multiple_of":64"#, with: #""multiple_of":16"#)
            .replacingOccurrences(
                of: #"["1280x720","720x1280"]"#,
                with: #"["592x592"]"#
            )
        let value = try JSONDecoder().decode(VideoCapabilities.self, from: Data(json.utf8))

        #expect(value.durationPresets(for: "592x592") == [1, 2])
    }

    @Test("Image input is enabled by input_reference presence, not an accepted boolean")
    func imageInputUsesCapabilityContract() throws {
        // The input_reference object has no `accepted` field in the current
        // contract; its presence with usable limits enables image-to-video.
        let value = try JSONDecoder().decode(
            VideoCapabilities.self, from: Data(Self.capabilitiesJSON.utf8)
        )
        #expect(value.supportsImageInput)
        #expect(value.acceptedReferenceMIMETypes == ["image/jpeg", "image/png", "image/webp"])
        #expect(value.referenceMaximumBytes == 20 * 1024 * 1024)

        // Limited formats narrow the accepted reference MIME types.
        let jpegOnly = Self.capabilitiesJSON.replacingOccurrences(
            of: #"["jpeg","png","webp"]"#,
            with: #"["jpeg"]"#
        )
        let jpegValue = try JSONDecoder().decode(
            VideoCapabilities.self, from: Data(jpegOnly.utf8)
        )
        #expect(jpegValue.supportsImageInput)
        #expect(jpegValue.acceptedReferenceMIMETypes == ["image/jpeg"])
    }

    @Test("Image input is disabled when input_reference is absent")
    func imageInputDisabledWithoutReference() throws {
        // Omit the input_reference object entirely (a video model that only
        // supports text-to-video, like CogVideoX-Fun, never sends it).
        // Replace the object with JSON null so the container stays well-formed;
        // the optional decodes to nil, identical to the key being absent.
        let json = Self.capabilitiesJSON.replacingOccurrences(
            of: #""input_reference":{"maximum_bytes":20971520,"maximum_pixels":16777216,"formats":["jpeg","png","webp"]}"#,
            with: #""input_reference":null"#
        )
        let value = try JSONDecoder().decode(VideoCapabilities.self, from: Data(json.utf8))
        #expect(value.limits.inputReference == nil)
        #expect(!value.supportsImageInput)
        #expect(value.referenceMaximumBytes == 0)
        #expect(value.acceptedReferenceMIMETypes.isEmpty)
    }

    @Test("Legacy accepted:false keeps image input disabled without rejecting the payload")
    func legacyAcceptedFalseDisablesImageInput() async throws {
        // A transitional server may still send the retired `accepted` boolean.
        // `accepted: false` must disable image input (not enable it by being
        // ignored) yet must not reject the whole payload during skew.
        let client = makeClient()
        let json = Self.capabilitiesJSON.replacingOccurrences(
            of: #""input_reference":{"maximum_bytes":20971520,"maximum_pixels":16777216,"formats":["jpeg","png","webp"]}"#,
            with: #""input_reference":{"accepted":false,"maximum_bytes":20971520,"formats":["jpeg","png","webp"]}"#
        )
        let decoded = try? JSONDecoder().decode(
            VideoCapabilities.self, from: Data(json.utf8)
        )
        #expect(decoded.map { !$0.supportsImageInput } == true)

        VideoStubProtocol.response = (200, Data(json.utf8))
        let value = try await client.capabilities(port: 8123, bearer: nil)
        #expect(!value.supportsImageInput)
        #expect(value.referenceMaximumBytes == 0)
        #expect(value.acceptedReferenceMIMETypes.isEmpty)
        #expect(!value.sizePresets.isEmpty)
    }

    @Test("Image input is disabled when reference limits are unusable")
    func imageInputDisabledWhenReferenceUnusable() async {
        // A zero byte budget leaves no usable reference limit, so image input is
        // disabled (and no MIME types are advertised), and the payload itself
        // fails closed on validation.
        let client = makeClient()
        let zeroBytes = Self.capabilitiesJSON.replacingOccurrences(
            of: #""maximum_bytes":20971520"#,
            with: #""maximum_bytes":0"#
        )
        let zeroValue = try! JSONDecoder().decode(
            VideoCapabilities.self, from: Data(zeroBytes.utf8)
        )
        #expect(!zeroValue.supportsImageInput)
        #expect(zeroValue.acceptedReferenceMIMETypes.isEmpty)
        VideoStubProtocol.response = (200, Data(zeroBytes.utf8))
        await #expect(throws: VideoClientError.invalidResponse) {
            _ = try await client.capabilities(port: 8123, bearer: nil)
        }

        // A non-positive pixel ceiling is likewise unusable and fails closed.
        let zeroPixels = Self.capabilitiesJSON.replacingOccurrences(
            of: #""maximum_pixels":16777216"#,
            with: #""maximum_pixels":0"#
        )
        let pixelsValue = try! JSONDecoder().decode(
            VideoCapabilities.self, from: Data(zeroPixels.utf8)
        )
        #expect(!pixelsValue.supportsImageInput)
        #expect(pixelsValue.acceptedReferenceMIMETypes.isEmpty)
        VideoStubProtocol.response = (200, Data(zeroPixels.utf8))
        await #expect(throws: VideoClientError.invalidResponse) {
            _ = try await client.capabilities(port: 8123, bearer: nil)
        }
    }

    @Test("Current contract with ceil_to_64 rounding decodes to working video")
    func currentContractDecodesToWorkingVideo() async throws {
        let client = makeClient()
        VideoStubProtocol.response = (200, Data(Self.capabilitiesJSON.utf8))

        let value = try await client.capabilities(port: 8123, bearer: "secret")

        #expect(value.modes == [.textToVideo, .imageToVideo])
        #expect(!value.sizePresets.isEmpty)
        #expect(!value.durationPresets(for: "512x512").isEmpty)
        #expect(value.supportsImageInput)
    }

    @Test("Reference MIME type is detected from bytes, not the filename")
    func referenceMIMETypeUsesBytes() throws {
        let png = try #require(Data(base64Encoded:
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
        ))

        #expect(VideoReferenceLoader.mimeType(for: png) == "image/png")
    }

    @Test("Create uses the documented multipart fields and reference name")
    func createMultipart() async throws {
        let client = makeClient()
        VideoStubProtocol.response = (200, Data(Self.jobJSON.utf8))

        let job = try await client.create(
            VideoCreateRequest(
                prompt: "A fox runs through snow",
                model: "ltx-2.3-mlx-q4",
                seconds: 2,
                size: "768x512",
                seed: 42,
                reference: Data("reference-bytes".utf8),
                referenceFileName: "fox.png",
                referenceMIMEType: "image/png"
            ),
            port: 8123,
            bearer: "secret"
        )

        #expect(job.status == .queued)
        let body = String(decoding: try #require(VideoStubProtocol.bodies.first), as: UTF8.self)
        #expect(body.contains("name=\"prompt\"\r\n\r\nA fox runs through snow"))
        #expect(body.contains("name=\"model\"\r\n\r\nltx-2.3-mlx-q4"))
        #expect(body.contains("name=\"seconds\"\r\n\r\n2"))
        #expect(body.contains("name=\"size\"\r\n\r\n768x512"))
        #expect(body.contains("name=\"seed\"\r\n\r\n42"))
        #expect(body.contains("name=\"input_reference\"; filename=\"fox.png\""))
        #expect(body.contains("reference-bytes"))
    }

    @Test("Nested server errors remain actionable")
    func nestedError() async throws {
        let client = makeClient()
        VideoStubProtocol.response = (
            409,
            Data(#"{"detail":{"error":{"message":"start a video model"}}}"#.utf8)
        )

        do {
            _ = try await client.capabilities(port: 8123, bearer: nil)
            Issue.record("Expected an HTTP error")
        } catch let error as VideoClientError {
            #expect(error.errorDescription == "start a video model")
        }
    }

    @Test("Multipart filenames cannot inject headers")
    func multipartFilenameEscaping() {
        let body = VideoClient.multipartBody(
            boundary: "test-boundary",
            fields: [],
            file: (
                field: "input_reference",
                name: "fox\"\\\r\nX-Injected: yes.png",
                mime: "image/png",
                data: Data()
            )
        )
        let text = String(decoding: body, as: UTF8.self)

        #expect(text.contains(#"filename="fox\"\\_X-Injected: yes.png""#))
        #expect(!text.contains("\r\nX-Injected:"))
    }

    @Test("Preview cache keys preserve complete validated job identity")
    func previewCacheIdentity() throws {
        let dashed = try VideoClient.cacheFileName(for: "a-b")
        let plain = try VideoClient.cacheFileName(for: "ab")

        #expect(dashed != plain)
        #expect(dashed.hasSuffix(".mp4"))
        #expect(throws: VideoClientError.invalidJobID) {
            _ = try VideoClient.cacheFileName(for: "a/b")
        }
        #expect(throws: VideoClientError.invalidJobID) {
            _ = try VideoClient.cacheFileName(for: "")
        }
    }

    @Test("Invalid job IDs fail before a cache or network lookup")
    func invalidJobIDStopsContentLookup() async {
        let client = makeClient()

        await #expect(throws: VideoClientError.invalidJobID) {
            _ = try await client.content(id: "../another-job", port: 8123, bearer: "secret")
        }
        #expect(VideoStubProtocol.requests.isEmpty)
    }

    @Test("Cache cleanup failure is retryable after server deletion")
    func cacheCleanupFailureIsRetryable() async throws {
        VideoStubProtocol.reset()
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [VideoStubProtocol.self]
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            "rapid-video-delete-tests-\(UUID().uuidString)", isDirectory: true
        )
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let id = "video_0123456789abcdef0123456789abcdef"
        let cached = directory.appendingPathComponent(try VideoClient.cacheFileName(for: id))
        try Data("video".utf8).write(to: cached)
        let client = VideoClient(
            session: URLSession(configuration: configuration),
            cacheDirectory: directory,
            removeCachedItem: { _ in throw CocoaError(.fileWriteNoPermission) }
        )

        await #expect(throws: VideoClientError.cacheRemoval) {
            try await client.delete(id: id, port: 8123, bearer: nil)
        }
        #expect(FileManager.default.fileExists(atPath: cached.path))
        #expect(VideoStubProtocol.requests.count == 1)
        #expect(VideoStubProtocol.requests.first?.httpMethod == "DELETE")

        VideoStubProtocol.response = (404, Data())
        let retryClient = VideoClient(
            session: URLSession(configuration: configuration),
            cacheDirectory: directory
        )
        try await retryClient.delete(id: id, port: 8123, bearer: nil)
        #expect(!FileManager.default.fileExists(atPath: cached.path))
    }

    @Test("Server deletion failure preserves an available cached preview")
    func serverDeleteFailurePreservesCache() async throws {
        VideoStubProtocol.reset()
        VideoStubProtocol.response = (500, Data())
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [VideoStubProtocol.self]
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            "rapid-video-delete-tests-\(UUID().uuidString)", isDirectory: true
        )
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let id = "video_0123456789abcdef0123456789abcdef"
        let cached = directory.appendingPathComponent(try VideoClient.cacheFileName(for: id))
        try Data("video".utf8).write(to: cached)
        let client = VideoClient(
            session: URLSession(configuration: configuration), cacheDirectory: directory
        )

        await #expect(throws: VideoClientError.self) {
            try await client.delete(id: id, port: 8123, bearer: nil)
        }
        #expect(FileManager.default.fileExists(atPath: cached.path))
    }

    @Test("Reference loader rejects an oversized file before allocating it")
    func oversizedReferenceIsRejected() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            "rapid-video-reference-tests-\(UUID().uuidString)", isDirectory: true
        )
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let source = directory.appendingPathComponent("oversized.png")
        _ = FileManager.default.createFile(atPath: source.path, contents: nil)
        let handle = try FileHandle(forWritingTo: source)
        try handle.truncate(atOffset: UInt64(VideoClient.maxReferenceBytes + 1))
        try handle.close()

        #expect(throws: VideoReferenceLoaderError.tooLarge) {
            _ = try VideoReferenceLoader.load(from: source)
        }
    }

    @Test("Reference loader enforces the advertised decoded-pixel ceiling")
    func referencePixelCeilingIsEnforced() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            "rapid-video-reference-pixel-tests-\(UUID().uuidString)", isDirectory: true
        )
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let source = directory.appendingPathComponent("two-pixel.png")
        let bitmap = try #require(NSBitmapImageRep(
            bitmapDataPlanes: nil,
            pixelsWide: 2,
            pixelsHigh: 1,
            bitsPerSample: 8,
            samplesPerPixel: 4,
            hasAlpha: true,
            isPlanar: false,
            colorSpaceName: .deviceRGB,
            bytesPerRow: 0,
            bitsPerPixel: 0
        ))
        let png = try #require(bitmap.representation(using: .png, properties: [:]))
        try png.write(to: source)

        #expect(try VideoReferenceLoader.load(from: source, maximumPixels: 2) == png)
        #expect(throws: VideoReferenceLoaderError.tooLarge) {
            _ = try VideoReferenceLoader.load(from: source, maximumPixels: 1)
        }
    }

    @Test("Failed save leaves an existing destination untouched")
    func failedSavePreservesDestination() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            "rapid-video-save-tests-\(UUID().uuidString)", isDirectory: true
        )
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let destination = directory.appendingPathComponent("kept.mp4")
        try Data("original".utf8).write(to: destination)

        do {
            try VideoPreviewSaver.save(
                source: directory.appendingPathComponent("missing.mp4"),
                destination: destination
            )
            Issue.record("Expected a missing-source failure")
        } catch {}

        #expect(try Data(contentsOf: destination) == Data("original".utf8))
    }

    @Test("Successful save atomically replaces an existing destination")
    func successfulSaveReplacesDestination() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            "rapid-video-save-tests-\(UUID().uuidString)", isDirectory: true
        )
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let source = directory.appendingPathComponent("preview.mp4")
        let destination = directory.appendingPathComponent("saved.mp4")
        try Data("new video".utf8).write(to: source)
        try Data("old video".utf8).write(to: destination)

        try VideoPreviewSaver.save(source: source, destination: destination)

        #expect(try Data(contentsOf: destination) == Data("new video".utf8))
    }

    private static let capabilitiesJSON = #"""
    {
      "object":"video.capabilities","model":"ltx","modality":"video-gen","family":"ltx-2.3",
      "modes":["text-to-video","image-to-video"],
      "limits":{
        "size":{"type":"range","width":{"minimum":256,"maximum":1920,"multiple_of":64},"height":{"minimum":256,"maximum":1920,"multiple_of":64},"also_supported":["1280x720","720x1280"]},
        "seconds":{"minimum":1,"maximum":20,"default":4},
        "fps":{"minimum":1,"maximum":60,"default":24,"fixed":false},
        "frames":{"minimum":9,"maximum":1201,"step":8,"offset":1},
        "workload":{"metric":"pixel_frames","maximum":38141952,"dimension_rounding":"ceil_to_64"},
        "input_reference":{"maximum_bytes":20971520,"maximum_pixels":16777216,"formats":["jpeg","png","webp"]}
      },
      "controls":{}
    }
    """#

    private static let jobJSON = #"""
    {"id":"video_0123456789abcdef0123456789abcdef","model":"ltx-2.3-mlx-q4","prompt":"A fox runs through snow","seconds":"2","size":"768x512","status":"queued","progress":0,"created_at":123,"completed_at":null,"error":null,"object":"video"}
    """#
}

private final class VideoStubProtocol: URLProtocol, @unchecked Sendable {
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
