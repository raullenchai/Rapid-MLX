import Foundation
import Testing
@testable import Rapid

@Suite("H3 Desktop video")
struct VideoH3Tests {
    @Test("The packaged video allowlist includes both H3 aliases and no separate Turbo alias")
    func packagedAliases() {
        #expect(ModelCatalog.packagedVideoAliases.contains("minimax-h3"))
        #expect(ModelCatalog.packagedVideoAliases.contains("pika-h3-flex"))
        #expect(!ModelCatalog.packagedVideoAliases.contains("minimax-h3-turbo"))
    }

    @Test("Catalog admits the packaged H3 rows and preserves their server memory floors")
    func catalogEntries() async throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("rapid-h3-catalog-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let binary = directory.appendingPathComponent("rapid-mlx")
        let script = #"""
        #!/bin/sh
        if [ "$1" = "models" ]; then
          printf '%s' '{"text":[],"audio":[],"image":[],"video":[{"alias":"minimax-h3","hf_path":"org/h3","video_modes":["text-to-video","image-to-video"],"min_memory_gb":160},{"alias":"pika-h3-flex","hf_path":"org/flex","video_modes":["text-to-video","image-to-video","reference-to-video"],"min_memory_gb":192},{"alias":"future-unpackaged-video","hf_path":"org/future","video_modes":["text-to-video"],"min_memory_gb":24}]}'
        else
          printf '%s' 'No cached models'
        fi
        """#
        try script.write(to: binary, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: binary.path)
        let entries = await ModelCatalog.videoEntries(binary: binary, hubCacheOverride: nil)
        #expect(entries.count == 2)
        #expect(entries.first { $0.alias == "minimax-h3" }?.minimumMemoryGB == 160)
        let flex = try #require(entries.first { $0.alias == "pika-h3-flex" })
        #expect(flex.minimumMemoryGB == 192)
        #expect(flex.videoCapabilities.contains(.referenceToVideo))
        #expect(!entries.contains { $0.alias == "future-unpackaged-video" })
    }

    @MainActor
    @Test("H3 memory floors keep undersized Macs unavailable with an explanation", arguments: [
        ("minimax-h3", 160.0), ("pika-h3-flex", 192.0),
    ])
    func memoryGating(_ alias: String, _ minimum: Double) async {
        let entry = Self.entry(alias: alias, minimum: minimum)
        let server = ServerManager(
            testingState: .idle,
            binaryPath: URL(fileURLWithPath: "/usr/bin/true")
        )
        let model = VideoGenViewModel(
            server: server, client: H3VideoClient(), physicalRAMGB: minimum - 1,
            catalogLoader: { _ in [entry] }
        )
        await model.refreshCatalog()
        #expect(model.videoModels == [entry])
        #expect(!model.isSelectedModelEligible)
        #expect(!model.isModelEligible(entry))
        #expect(model.memoryRequirementText?.contains(String(Int(minimum))) == true)
        #expect(model.memoryRequirementText?.contains(String(Int(minimum - 1))) == true)
        await model.prepareSelectedModel()
        #expect(server.servingAlias == nil)
        #expect(!model.isPreparing)
        #expect(!model.canSubmit)
        let gib = UInt64(1) << 30
        #expect(!ServerManager.videoMemoryFloorSatisfied(
            minimumMemoryGB: minimum,
            snapshot: .init(totalBytes: UInt64(minimum - 1) * gib, usedBytes: gib)
        ))
        #expect(ServerManager.videoMemoryFloorSatisfied(
            minimumMemoryGB: minimum,
            snapshot: .init(totalBytes: UInt64(minimum) * gib, usedBytes: gib)
        ))
    }

    @Test("Old capabilities omit H3 additions; advertised limits decode without legacy flags")
    func optionalCapabilityDecoding() throws {
        let legacy = try Self.capabilities(lastFrame: false, references: false)
        #expect(legacy.limits.lastFrame == nil)
        #expect(legacy.limits.references == nil)
        let h3 = try Self.capabilities()
        #expect(h3.limits.lastFrame?.maximumBytes == 1024)
        #expect(h3.limits.lastFrame?.formats == ["jpeg", "png", "webp"])
        #expect(h3.limits.references?.maximumTotal == 12)
        #expect(h3.limits.references?.image?.maximum == 9)
        #expect(h3.limits.references?.video?.maximum == 3)
        #expect(h3.limits.references?.audio?.maximum == 3)
        #expect(h3.limits.references?.requiresVisual == true)
        #expect(h3.limits.references?.exclusiveWith == ["input_reference", "last_frame"])
        #expect(h3.modes.contains(.referenceToVideo))
        #expect(try h3.validated() == h3)
        #expect(h3.durationPresets(for: "512x512").allSatisfy { (5...15).contains($0) })
        #expect(h3.durationPresets(for: "512x512") == [5, 10, 15])
    }

    @MainActor
    @Test("Last frame and references controls follow live capabilities")
    func controlVisibility() async throws {
        let model = await Self.viewModel()
        #expect(!model.supportsLastFrame)
        #expect(!model.supportedModes.contains(.references))
        model.capabilities = try Self.capabilities(lastFrame: false, references: false)
        #expect(!model.supportsLastFrame)
        #expect(!model.supportedModes.contains(.references))
        model.capabilities = try Self.capabilities()
        #expect(model.supportsLastFrame)
        #expect(model.supportedModes.contains(.references))
        model.capabilities = try Self.capabilities(lastFrame: true, references: false)
        #expect(model.supportsLastFrame)
        #expect(!model.supportedModes.contains(.references))
    }

    @MainActor
    @Test("References and first/last frames are mutually exclusive")
    func mutuallyExclusiveInputs() async throws {
        let model = await Self.viewModel()
        model.capabilities = try Self.capabilities()
        model.selectMode(.image)
        model.setReference(Self.image("first.png"))
        model.setLastFrame(Self.image("last.png"))
        #expect(model.referenceImage != nil)
        #expect(model.lastFrame != nil)
        model.selectMode(.references)
        #expect(model.referenceImage == nil)
        #expect(model.lastFrame == nil)
        #expect(model.addReferences([Self.image("one.png"), Self.image("two.png")]))
        #expect(model.references.count == 2)
        model.selectMode(.image)
        model.setReference(Self.image("first.png"))
        #expect(model.references.isEmpty)
        model.selectMode(.references)
        #expect(model.addReferences([Self.image("one.png")]))
        model.setLastFrame(Self.image("last.png"))
        #expect(model.references.isEmpty)
    }

    @MainActor
    @Test("References enforce image count, total count, bytes, formats and visual requirements")
    func referenceLimits() async throws {
        let model = await Self.viewModel()
        model.capabilities = try Self.capabilities(kinds: ["image"])
        model.selectMode(.references)
        #expect(model.capabilities?.referenceValidationMessage([]) != nil)
        #expect(model.addReferences((0..<9).map { Self.image("\($0).png") }))
        #expect(!model.addReferences([Self.image("extra.png")]))
        #expect(model.references.count == 9)
        #expect(model.errorMessage != nil)

        model.references = []
        #expect(!model.addReferences([.init(data: Data(repeating: 0, count: 1025), fileName: "large.png", mimeType: "image/png")]))
        #expect(model.references.isEmpty)
        #expect(!model.addReferences([.init(data: Data([1]), fileName: "bad.gif", mimeType: "image/gif")]))
        #expect(model.references.isEmpty)
        #expect(!model.addReferences([.init(data: Data([1]), fileName: "sound.wav", mimeType: "audio/wav")]))
        #expect(model.references.isEmpty)

        model.capabilities = try Self.capabilities(maximumTotal: 2)
        #expect(model.addReferences([Self.image("a.png"), Self.image("b.png")]))
        #expect(!model.addReferences([Self.image("c.png")]))
        #expect(model.references.count == 2)
    }

    @MainActor
    @Test("Submission revalidates stale or directly assigned attachments against current limits")
    func submissionRevalidatesAttachments() async throws {
        let model = await Self.viewModel()
        model.capabilities = try Self.capabilities()
        model.prompt = "A scene"
        model.size = "512x512"
        model.seconds = 5
        model.selectMode(.image)
        model.setReference(Self.image("first.png"))
        model.setLastFrame(Self.image("last.png"))
        #expect(model.canSubmit)
        model.lastFrame = .init(data: Data(repeating: 0, count: 1025), fileName: "large.png", mimeType: "image/png")
        #expect(!model.canSubmit)
        model.lastFrame = Self.image("last.png")
        model.capabilities = try Self.capabilities(lastFrame: false)
        #expect(!model.canSubmit)
        model.lastFrame = nil
        #expect(model.canSubmit)
        model.referenceImage = .init(data: Data([1]), fileName: "bad.gif", mimeType: "image/gif")
        #expect(!model.canSubmit)
        model.selectMode(.references)
        #expect(!model.canSubmit)
        model.references = [Self.image("one.png")]
        #expect(model.canSubmit)
        model.references = (0..<10).map { Self.image("\($0).png") }
        #expect(!model.canSubmit)
        model.references = [.init(data: Data([1]), fileName: "voice.wav", mimeType: "audio/wav")]
        #expect(!model.canSubmit)
    }

    @Test("Malformed optional H3 limits disable their controls without breaking older controls")
    func malformedOptionalLimits() throws {
        let fixtures: [(String, String)] = [
            (#""last_frame":{"maximum_bytes":1024,"maximum_pixels":16777216,"formats":["jpeg","png","webp"]}"#, #""last_frame":{}"#),
            (#""last_frame":{"maximum_bytes":1024,"maximum_pixels":16777216,"formats":["jpeg","png","webp"]}"#, #""last_frame":{"maximum_bytes":1024}"#),
        ]
        for (old, replacement) in fixtures {
            let json = Self.capabilitiesJSON.replacingOccurrences(of: old, with: replacement)
            let caps = try JSONDecoder().decode(VideoCapabilities.self, from: Data(json.utf8))
            #expect(caps.supportsImageInput)
            #expect(!caps.supportsLastFrame)
        }
        var json = try #require(JSONSerialization.jsonObject(with: Data(Self.capabilitiesJSON.utf8)) as? [String: Any])
        var limits = try #require(json["limits"] as? [String: Any])
        let malformedLimits: [[String: Any]] = [[:], ["field": "references", "maximum_total": 12]]
        for malformed in malformedLimits {
            limits["references"] = malformed
            json["limits"] = limits
            let caps = try JSONDecoder().decode(VideoCapabilities.self, from: JSONSerialization.data(withJSONObject: json))
            #expect(caps.supportsImageInput)
            #expect(caps.supportsLastFrame)
            #expect(!caps.supportsReferences)
        }
    }

    @Test("Importer types and missing-kind errors follow advertised limits")
    func advertisedReferenceKinds() throws {
        let images = try Self.capabilities(kinds: ["image"])
        #expect(images.referenceMIMETypes == ["image/jpeg", "image/png", "image/webp"])
        #expect(images.referenceContentTypes.count == 3)
        #expect(images.referenceDurationHints.isEmpty)
        for (kind, mime) in [("video", "video/mp4"), ("audio", "audio/wav")] {
            let file = VideoAttachment(data: Data([1]), fileName: "reference", mimeType: mime)
            #expect(images.referenceValidationMessage([file], requireVisual: false)
                == VideoCapabilities.unavailableReferenceMessage(for: kind))
        }
        let timed = try Self.capabilities()
        #expect(timed.referenceMIMETypes.count == 13)
        #expect(timed.referenceContentTypes.count == 13)
        #expect(timed.referenceDurationHints.count == 2)
        #expect(timed.referenceDurationHints.allSatisfy { $0.contains("2") && $0.contains("15") })
        for format in ["webm", "mkv", "aac", "ogg"] {
            let mime = try #require(VideoAttachment.mimeType(for: format))
            #expect(timed.referenceMIMETypes.contains(mime))
            #expect(VideoAttachment.contentType(for: mime) != nil)
        }
        let noImages = try Self.capabilities(kinds: ["video", "audio"])
        #expect(noImages.supportsReferences)
        #expect(noImages.referenceValidationMessage([Self.image("one.png")])
            == VideoCapabilities.unavailableReferenceMessage(for: "image"))
    }

    @MainActor
    @Test("Mixed references enforce each kind's counts, byte budgets, total and visual requirement")
    func mixedReferenceLimits() async throws {
        let model = await Self.viewModel()
        model.capabilities = try Self.capabilities(perKindMaximum: 1)
        model.selectMode(.references)
        let video = VideoAttachment(data: Data([1]), fileName: "clip.webm", mimeType: "video/webm")
        let audio = VideoAttachment(data: Data([1]), fileName: "voice.aac", mimeType: "audio/aac")
        #expect(model.addReferences([Self.image("one.png"), video, audio]))
        #expect(model.referencesValidationMessage == nil)
        for file in [Self.image("two.png"), video, audio] {
            #expect(!model.addReferences([file]))
            #expect(model.references.count == 3)
        }
        let caps = try Self.capabilities()
        for (mime, maximum) in [("image/png", 1024), ("video/mp4", 2048), ("audio/ogg", 1024)] {
            let file = VideoAttachment(data: Data(repeating: 1, count: maximum), fileName: "file", mimeType: mime)
            #expect(caps.referenceValidationMessage([file], requireVisual: false) == nil)
            let large = VideoAttachment(data: file.data + Data([1]), fileName: "file", mimeType: mime)
            #expect(caps.referenceValidationMessage([large], requireVisual: false) != nil)
        }
        #expect(caps.referenceValidationMessage([audio]) != nil)
        #expect(caps.referenceValidationMessage([video]) == nil)
        #expect(try Self.capabilities(requiresVisual: false).referenceValidationMessage([audio]) == nil)
        let total = try Self.capabilities(maximumTotal: 2)
        #expect(total.referenceValidationMessage([Self.image("one.png"), video, audio]) != nil)
        #expect(caps.referenceValidationMessage([
            .init(data: Data([1]), fileName: "bad.avi", mimeType: "video/x-msvideo")
        ]) != nil)
    }

    @Test("Reference importer loads advertised timed formats and rejects absent kinds before reading")
    func referenceFileLoading() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("h3-references-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let caps = try Self.capabilities()
        for ext in ["mp4", "mov", "webm", "mkv", "wav", "mp3", "m4a", "aac", "flac", "ogg"] {
            let url = directory.appendingPathComponent("ref.\(ext)")
            try Data([1]).write(to: url)
            let file = try VideoReferenceLoader.loadReference(from: url, capabilities: caps)
            #expect(file.mimeType == VideoAttachment.mimeType(for: ext))
            #expect(throws: VideoReferenceLoaderError.unavailableKind(file.kind!)) {
                try VideoReferenceLoader.loadReference(from: url, capabilities: Self.capabilities(kinds: ["image"]))
            }
        }
    }

    @Test("Multipart supports a last frame and repeated named reference parts")
    func multipartEncoding() throws {
        let files: [(field: String, name: String, mime: String, data: Data)] = [
            ("last_frame", "end.png", "image/png", Data("last-frame".utf8)),
            ("references", "subject.jpg", "image/jpeg", Data("subject".utf8)),
            ("references", "motion.mp4", "video/mp4", Data("motion".utf8)),
            ("references", "voice.wav", "audio/wav", Data("voice".utf8)),
        ]
        let body = VideoClient.multipartBody(
            boundary: "h3-test", fields: [("prompt", "A scene")], file: nil, files: files
        )
        let string = try #require(String(data: body, encoding: .utf8))
        #expect(string.components(separatedBy: "name=\"references\"").count - 1 == 3)
        #expect(string.contains("name=\"last_frame\"; filename=\"end.png\"\r\nContent-Type: image/png\r\n\r\nlast-frame"))
        #expect(string.contains("name=\"references\"; filename=\"subject.jpg\"\r\nContent-Type: image/jpeg"))
        #expect(string.contains("name=\"references\"; filename=\"motion.mp4\"\r\nContent-Type: video/mp4"))
        #expect(string.contains("name=\"references\"; filename=\"voice.wav\"\r\nContent-Type: audio/wav"))
        #expect(string.hasSuffix("--h3-test--\r\n"))
    }

    private static func image(_ name: String) -> VideoAttachment {
        .init(data: Data([1]), fileName: name, mimeType: "image/png")
    }

    private static func entry(alias: String = "pika-h3-flex", minimum: Double = 192) -> ModelEntry {
        ModelEntry(
            alias: alias, hfRepo: "org/h3", sizeOnDisk: "1 GB", cached: true,
            kind: .video,
            videoCapabilities: [.textToVideo, .imageToVideo, .referenceToVideo],
            minimumMemoryGB: minimum
        )
    }

    @MainActor
    private static func viewModel() async -> VideoGenViewModel {
        let entry = entry()
        let server = ServerManager(
            testingState: .ready(alias: entry.alias),
            binaryPath: URL(fileURLWithPath: "/usr/bin/true"), activeBearer: "test-bearer"
        )
        let model = VideoGenViewModel(
            server: server, client: H3VideoClient(), physicalRAMGB: 256,
            catalogLoader: { _ in [entry] }
        )
        await model.refreshCatalog()
        return model
    }

    static func capabilities(
        lastFrame: Bool = true, references: Bool = true, maximumTotal: Int = 12,
        kinds: [String] = ["image", "video", "audio"], requiresVisual: Bool = true,
        perKindMaximum: Int? = nil
    ) throws -> VideoCapabilities {
        var json = try #require(JSONSerialization.jsonObject(with: Data(capabilitiesJSON.utf8)) as? [String: Any])
        var limits = try #require(json["limits"] as? [String: Any])
        if !lastFrame { limits.removeValue(forKey: "last_frame") }
        if references {
            var referenceLimits = try #require(limits["references"] as? [String: Any])
            referenceLimits["maximum_total"] = maximumTotal
            referenceLimits["requires_visual"] = requiresVisual
            for kind in ["image", "video", "audio"] {
                if !kinds.contains(kind) { referenceLimits.removeValue(forKey: kind) }
                if let perKindMaximum, var limit = referenceLimits[kind] as? [String: Any] {
                    limit["maximum"] = perKindMaximum
                    referenceLimits[kind] = limit
                }
            }
            limits["references"] = referenceLimits
        } else {
            limits.removeValue(forKey: "references")
            json["modes"] = ["text-to-video", "image-to-video"]
        }
        json["limits"] = limits
        return try JSONDecoder().decode(VideoCapabilities.self, from: JSONSerialization.data(withJSONObject: json))
    }

    private static let capabilitiesJSON = #"""
    {
      "model":"org/h3","family":"minimax-h3",
      "modes":["text-to-video","image-to-video","reference-to-video"],
      "limits":{
        "size":{"type":"range","width":{"minimum":256,"maximum":1920,"multiple_of":32},"height":{"minimum":256,"maximum":1920,"multiple_of":32},"also_supported":["1280x720","720x1280"]},
        "seconds":{"minimum":5,"maximum":15,"default":5},
        "fps":{"minimum":24,"maximum":24,"default":24,"fixed":true},
        "frames":{"minimum":124,"maximum":345,"step":17,"offset":5},
        "workload":{"metric":"pixel_frames","maximum":300000000,"dimension_rounding":"ceil_to_32"},
        "input_reference":{"maximum_bytes":1024,"maximum_pixels":16777216,"formats":["jpeg","png","webp"]},
        "last_frame":{"maximum_bytes":1024,"maximum_pixels":16777216,"formats":["jpeg","png","webp"]},
        "references":{
          "field":"references","maximum_total":12,
          "image":{"maximum":9,"maximum_bytes":1024,"formats":["jpeg","png","webp"]},
          "video":{"maximum":3,"maximum_bytes":2048,"formats":["mp4","mov","webm","mkv"],"duration_seconds":{"minimum":2,"maximum":15}},
          "audio":{"maximum":3,"maximum_bytes":1024,"formats":["wav","mp3","m4a","aac","flac","ogg"],"duration_seconds":{"minimum":2,"maximum":15}},
          "total_duration_seconds_per_kind":15,"requires_visual":true,"exclusive_with":["input_reference","last_frame"]
        }
      }
    }
    """#
}

private actor H3VideoClient: VideoClientProtocol {
    func capabilities(port: Int, bearer: String?) async throws -> VideoCapabilities {
        try VideoH3Tests.capabilities()
    }
    func create(_ request: VideoCreateRequest, port: Int, bearer: String?) async throws -> VideoJob {
        VideoJob(
            id: "video_0123456789abcdef0123456789abcdef", model: request.model,
            prompt: request.prompt, seconds: String(request.seconds), size: request.size,
            status: .queued, progress: 0, createdAt: 1, completedAt: nil, error: nil
        )
    }
    func list(port: Int, bearer: String?, limit: Int) async throws -> [VideoJob] { [] }
    func cancel(id: String, port: Int, bearer: String?) async throws {}
    func delete(id: String, port: Int, bearer: String?) async throws {}
    func content(id: String, port: Int, bearer: String?) async throws -> URL {
        URL(fileURLWithPath: "/nonexistent/video.mp4")
    }
}
