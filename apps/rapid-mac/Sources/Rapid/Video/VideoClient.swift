import CryptoKit
import Foundation
import UniformTypeIdentifiers

enum VideoJobStatus: String, Codable, Sendable, Hashable {
    case queued
    case inProgress = "in_progress"
    case completed
    case failed
}

struct VideoJobError: Codable, Sendable, Hashable {
    let code: String?
    let message: String?
}

struct VideoJob: Identifiable, Codable, Sendable, Hashable {
    let id: String
    let model: String
    let prompt: String
    let seconds: String
    let size: String
    let status: VideoJobStatus
    let progress: Int
    let createdAt: Int
    let completedAt: Int?
    let error: VideoJobError?

    var cancelRequested: Bool = false
    var phase: String? = nil

    var isCancelled: Bool {
        status == .failed && error?.code == "video_generation_cancelled"
    }

    enum CodingKeys: String, CodingKey {
        case cancelRequested = "cancel_requested"
        case phase
        case id, model, prompt, seconds, size, status, progress, error
        case createdAt = "created_at"
        case completedAt = "completed_at"
    }
}

extension VideoJob {
    init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        id = try values.decode(String.self, forKey: .id)
        model = try values.decode(String.self, forKey: .model)
        prompt = try values.decode(String.self, forKey: .prompt)
        seconds = try values.decode(String.self, forKey: .seconds)
        size = try values.decode(String.self, forKey: .size)
        status = try values.decode(VideoJobStatus.self, forKey: .status)
        progress = try values.decode(Int.self, forKey: .progress)
        createdAt = try values.decode(Int.self, forKey: .createdAt)
        completedAt = try values.decodeIfPresent(Int.self, forKey: .completedAt)
        error = try values.decodeIfPresent(VideoJobError.self, forKey: .error)
        cancelRequested = try values.decodeIfPresent(Bool.self, forKey: .cancelRequested) ?? false
        phase = try values.decodeIfPresent(String.self, forKey: .phase)
    }
}

struct VideoCapabilities: Decodable, Sendable, Hashable {
    struct Limits: Decodable, Sendable, Hashable {
        struct Dimension: Decodable, Sendable, Hashable {
            let minimum: Int
            let maximum: Int
            let multipleOf: Int

            enum CodingKeys: String, CodingKey {
                case minimum, maximum
                case multipleOf = "multiple_of"
            }
        }

        struct SizeLimit: Decodable, Sendable, Hashable {
            let type: String
            let values: [String]?
            let width: Dimension?
            let height: Dimension?
            let maximumArea: Int?
            let alsoSupported: [String]?

            enum CodingKeys: String, CodingKey {
                case type, values, width, height
                case maximumArea = "maximum_area"
                case alsoSupported = "also_supported"
            }
        }

        struct SecondsLimit: Decodable, Sendable, Hashable {
            let minimum: Int
            let maximum: Int
            let `default`: Int
        }

        struct FPSLimit: Decodable, Sendable, Hashable {
            let minimum: Int
            let maximum: Int
            let `default`: Int
            let fixed: Bool
        }

        struct FramesLimit: Decodable, Sendable, Hashable {
            let minimum: Int
            let maximum: Int
            let step: Int
            let offset: Int
        }

        struct WorkloadLimit: Decodable, Sendable, Hashable {
            let metric: String
            let maximum: Int
            let dimensionRounding: String

            enum CodingKeys: String, CodingKey {
                case metric, maximum
                case dimensionRounding = "dimension_rounding"
            }

            /// Alignment step applied to generation dimensions before the
            /// pixel-workload budget is checked. Mirrors the server's
            /// `dimension_rounding` vocabulary (`none` -> raw dimensions,
            /// `ceil_to_32`, `ceil_to_64`).
            ///
            /// Unknown values have no safe fallback: treating a future larger
            /// alignment as 64 could under-count workload and expose a preset the
            /// server must reject. Callers therefore fail closed on `nil`.
            var alignmentStep: Int? {
                switch dimensionRounding {
                case "none": 1
                case "ceil_to_32": 32
                case "ceil_to_64": 64
                default: nil
                }
            }
        }

        struct InputReferenceLimit: Decodable, Sendable, Hashable {
            let maximumBytes: Int
            let maximumPixels: Int?
            let formats: [String]
            /// Legacy field from the retired contract. Absent for the current
            /// server (which signals image-to-video support by the reference's
            /// presence); honored here so a transitional server that still sends
            /// `accepted: false` keeps image input disabled rather than enabling
            /// it by ignoring the flag.
            let accepted: Bool?

            enum CodingKeys: String, CodingKey {
                case formats
                case maximumBytes = "maximum_bytes"
                case maximumPixels = "maximum_pixels"
                case accepted
            }
        }

        struct ReferencesLimit: Decodable, Sendable, Hashable {
            struct MediaLimit: Decodable, Sendable, Hashable {
                struct Duration: Decodable, Sendable, Hashable {
                    let minimum: Double
                    let maximum: Double
                }
                let maximum: Int
                let maximumBytes: Int
                let formats: [String]
                let durationSeconds: Duration?
                enum CodingKeys: String, CodingKey {
                    case maximum, formats
                    case maximumBytes = "maximum_bytes"
                    case durationSeconds = "duration_seconds"
                }
            }
            let field: String
            let maximumTotal: Int
            let image: MediaLimit?
            let video: MediaLimit?
            let audio: MediaLimit?
            let requiresVisual: Bool
            let exclusiveWith: [String]
            let totalDurationSecondsPerKind: Double?
            enum CodingKeys: String, CodingKey {
                case field, image, video, audio
                case maximumTotal = "maximum_total"
                case requiresVisual = "requires_visual"
                case exclusiveWith = "exclusive_with"
                case totalDurationSecondsPerKind = "total_duration_seconds_per_kind"
            }
        }

        let size: SizeLimit
        let seconds: SecondsLimit
        let fps: FPSLimit
        let frames: FramesLimit
        let workload: WorkloadLimit
        let inputReference: InputReferenceLimit?
        let lastFrame: InputReferenceLimit?
        let references: ReferencesLimit?

        init(
            size: SizeLimit,
            seconds: SecondsLimit,
            fps: FPSLimit,
            frames: FramesLimit,
            workload: WorkloadLimit,
            inputReference: InputReferenceLimit? = nil,
            lastFrame: InputReferenceLimit? = nil,
            references: ReferencesLimit? = nil
        ) {
            self.size = size
            self.seconds = seconds
            self.fps = fps
            self.frames = frames
            self.workload = workload
            self.inputReference = inputReference
            self.lastFrame = lastFrame
            self.references = references
        }

        enum CodingKeys: String, CodingKey {
            case size, seconds, fps, frames, workload
            case inputReference = "input_reference"
            case lastFrame = "last_frame"
            case references
        }

        init(from decoder: Decoder) throws {
            let values = try decoder.container(keyedBy: CodingKeys.self)
            size = try values.decode(SizeLimit.self, forKey: .size)
            seconds = try values.decode(SecondsLimit.self, forKey: .seconds)
            fps = try values.decode(FPSLimit.self, forKey: .fps)
            frames = try values.decode(FramesLimit.self, forKey: .frames)
            workload = try values.decode(WorkloadLimit.self, forKey: .workload)
            inputReference = try values.decodeIfPresent(InputReferenceLimit.self, forKey: .inputReference)
            lastFrame = try? values.decodeIfPresent(InputReferenceLimit.self, forKey: .lastFrame)
            references = try? values.decodeIfPresent(ReferencesLimit.self, forKey: .references)
        }
    }

    struct Cancellation: Decodable, Sendable, Hashable {
        let queued: Bool?
        let inProgress: Bool?

        enum CodingKeys: String, CodingKey {
            case queued
            case inProgress = "in_progress"
        }
    }

    let model: String
    let family: String
    let modes: [VideoModelCapability]
    var cancellation: Cancellation? = nil
    var supportsInProgressCancel: Bool { cancellation?.inProgress == true }

    let limits: Limits

    /// Image MIME types advertised by the reference, independent of whether the
    /// reference is currently usable. Kept separate from the gated
    /// ``acceptedReferenceMIMETypes`` so the usability checks (which read this)
    /// do not depend on a gated value.
    private var inputReferenceFormats: Set<String> {
        guard let input = limits.inputReference else { return [] }
        return Set(input.formats.compactMap { format in
            switch format.lowercased() {
            case "jpeg", "jpg", "image/jpeg": "image/jpeg"
            case "png", "image/png": "image/png"
            case "webp", "image/webp": "image/webp"
            default: nil
            }
        })
    }

    /// The `input_reference` limit is usable when the server advertises a
    /// positive byte budget, at least one image MIME type this client can
    /// produce, and (when declared) a positive pixel ceiling. A legacy
    /// `accepted: false` (retired current-server field) explicitly disables it.
    private var usableReferenceLimits: Bool {
        guard let input = limits.inputReference,
              input.accepted != false,
              input.maximumBytes > 0,
              !inputReferenceFormats.isEmpty else { return false }
        return input.maximumPixels.map { $0 > 0 } ?? true
    }

    /// The image MIME types this client accepts for a reference image.
    /// Non-empty only while the reference is present AND usable, so an
    /// explicitly disabled (legacy `accepted: false`) or malformed reference
    /// never advertises supported formats.
    var acceptedReferenceMIMETypes: Set<String> {
        usableReferenceLimits ? inputReferenceFormats : []
    }

    /// Image-to-video is enabled by the presence of a usable `input_reference`
    /// limit object, not by a retired `accepted` boolean.
    var supportsImageInput: Bool {
        modes.contains(.imageToVideo) && usableReferenceLimits
    }

    var supportsLastFrame: Bool {
        guard let limit = limits.lastFrame else { return false }
        return limit.accepted != false && limit.maximumBytes > 0
            && (limit.maximumPixels.map { $0 > 0 } ?? true)
            && limit.formats.compactMap(VideoAttachment.mimeType(for:)).contains { $0.hasPrefix("image/") }
    }

    func referenceLimit(for kind: String) -> Limits.ReferencesLimit.MediaLimit? {
        guard let references = limits.references else { return nil }
        switch kind {
        case "image": return references.image
        case "video": return references.video
        case "audio": return references.audio
        default: return nil
        }
    }

    var referenceMIMETypes: Set<String> {
        Set(["image", "video", "audio"].flatMap { kind in
            guard let limit = referenceLimit(for: kind), limit.maximum > 0,
                  limit.maximumBytes > 0 else { return [String]() }
            return limit.formats.compactMap(VideoAttachment.mimeType(for:))
                .filter { $0.hasPrefix(kind + "/") }
        })
    }

    var referenceContentTypes: [UTType] {
        referenceMIMETypes.sorted().compactMap(VideoAttachment.contentType(for:))
    }

    var supportsReferences: Bool {
        guard modes.contains(.referenceToVideo), let limit = limits.references,
              limit.field == "references", limit.maximumTotal > 0 else { return false }
        return !referenceMIMETypes.isEmpty
    }

    static func unavailableReferenceMessage(for kind: String) -> String {
        switch kind {
        case "video": String(localized: "This model's server does not accept video references.")
        case "audio": String(localized: "This model's server does not accept audio references.")
        default: String(localized: "This model's server does not accept image references.")
        }
    }

    var referenceDurationHints: [String] {
        ["video", "audio"].compactMap { kind in
            guard let duration = referenceLimit(for: kind)?.durationSeconds else { return nil }
            let minimum = duration.minimum.formatted()
            let maximum = duration.maximum.formatted()
            return kind == "video"
                ? String(localized: "Video references: \(minimum)–\(maximum) seconds.")
                : String(localized: "Audio references: \(minimum)–\(maximum) seconds.")
        }
    }

    func referenceValidationMessage(_ files: [VideoAttachment], requireVisual: Bool = true) -> String? {
        guard supportsReferences, let limits = limits.references else {
            return String(localized: "Reference-to-video is unavailable for this model.")
        }
        guard files.count <= limits.maximumTotal else {
            return String(localized: "Too many reference files for this model.")
        }
        for kind in ["image", "video", "audio"] {
            let selected = files.filter { $0.kind == kind }
            guard !selected.isEmpty else { continue }
            guard let limit = referenceLimit(for: kind) else {
                return Self.unavailableReferenceMessage(for: kind)
            }
            guard selected.count <= limit.maximum else {
                return String(localized: "Too many references of this type for this model.")
            }
            let formats = Set(limit.formats.compactMap(VideoAttachment.mimeType(for:)))
            guard selected.allSatisfy({ formats.contains($0.mimeType) }) else {
                return String(localized: "That reference format is not supported by this model.")
            }
            guard selected.allSatisfy({ !$0.data.isEmpty && $0.data.count <= limit.maximumBytes }) else {
                return String(localized: "That reference file exceeds this model's size limit.")
            }
        }
        guard files.allSatisfy({ $0.kind != nil }) else {
            return String(localized: "That reference format is not supported by this model.")
        }
        if requireVisual && (files.isEmpty || (limits.requiresVisual && !files.contains { $0.kind == "image" || $0.kind == "video" })) {
            return String(localized: "Add at least one image or video reference.")
        }
        return nil
    }

    /// Conservative, familiar output shapes. The API remains authoritative:
    /// candidates outside its dimensions, alignment, or area are removed.
    var sizePresets: [String] {
        if limits.size.type == "fixed" {
            return limits.size.values ?? []
        }
        let explicitlySupported = Set(limits.size.alsoSupported ?? [])
        let candidates = ["512x512", "768x512", "512x768"]
            + (limits.size.alsoSupported ?? [])
        var seen = Set<String>()
        var accepted: [(area: Int, order: Int, value: String)] = []
        for (index, value) in candidates.enumerated() {
            guard seen.insert(value).inserted,
                  let dimensions = Self.parseSize(value),
                  let area = Self.product(dimensions.width, dimensions.height) else { continue }
            if explicitlySupported.contains(value) {
                accepted.append((area, index, value))
                continue
            }
            guard let width = limits.size.width,
                  let height = limits.size.height,
                  Self.contains(dimensions.width, minimum: width.minimum, maximum: width.maximum),
                  Self.contains(dimensions.height, minimum: height.minimum, maximum: height.maximum),
                  dimensions.width.isMultiple(of: width.multipleOf),
                  dimensions.height.isMultiple(of: height.multipleOf) else {
                continue
            }
            if let maximumArea = limits.size.maximumArea {
                guard let roundedWidth = Self.roundUp(dimensions.width, to: width.multipleOf),
                      let roundedHeight = Self.roundUp(dimensions.height, to: height.multipleOf),
                      let roundedArea = Self.product(roundedWidth, roundedHeight),
                      roundedArea <= maximumArea else { continue }
            }
            accepted.append((area, index, value))
        }
        return accepted.sorted { lhs, rhs in
            lhs.0 == rhs.0 ? lhs.1 < rhs.1 : lhs.0 < rhs.0
        }.map { $0.value }
    }

    func durationPresets(for size: String) -> [Int] {
        let candidates = Set(limits.seconds.minimum >= 5
            ? [limits.seconds.minimum, limits.seconds.default, 10, limits.seconds.maximum]
            : [limits.seconds.minimum, 1, 2, 4]).sorted()
        let values = candidates.filter {
            Self.contains($0, minimum: limits.seconds.minimum, maximum: limits.seconds.maximum)
                && workloadAllows(seconds: $0, size: size)
        }
        if !values.isEmpty { return values }
        return Self.contains(
            limits.seconds.default,
            minimum: limits.seconds.minimum,
            maximum: limits.seconds.maximum
        )
            && workloadAllows(seconds: limits.seconds.default, size: size)
            ? [limits.seconds.default] : []
    }

    /// The usable byte budget for an input reference image. Returns 0 unless the
    /// reference limit is fully usable (positive byte budget, at least one image
    /// MIME type, and a positive pixel ceiling when declared) so a disabled or
    /// malformed reference never exposes a positive budget to callers.
    var referenceMaximumBytes: Int {
        guard let input = limits.inputReference, usableReferenceLimits else { return 0 }
        return min(input.maximumBytes, VideoClient.maxReferenceBytes)
    }

    /// Optional decoded-pixel ceiling advertised by the active server. Older
    /// compatible servers omit it, in which case the byte limit remains the
    /// only client-side bound and the server stays authoritative.
    var referenceMaximumPixels: Int? {
        guard let input = limits.inputReference, usableReferenceLimits else { return nil }
        return input.maximumPixels
    }

    func validated() throws -> Self {
        let size = limits.size
        let validDimension: (Limits.Dimension) -> Bool = {
            $0.minimum > 0 && $0.maximum >= $0.minimum && $0.multipleOf > 0
        }
        let validSize: Bool
        switch size.type {
        case "fixed":
            validSize = !(size.values ?? []).isEmpty
                && (size.values ?? []).allSatisfy { Self.parseSize($0) != nil }
        case "range":
            validSize = size.width.map(validDimension) == true
                && size.height.map(validDimension) == true
                && (size.maximumArea.map { $0 > 0 } ?? true)
        default:
            validSize = false
        }
        let seconds = limits.seconds
        let fps = limits.fps
        let frames = limits.frames
        let workload = limits.workload
        // A present input_reference must be either usable, or explicitly
        // disabled by a legacy `accepted: false` (a transitional server that
        // still sends the retired flag). Any other present-but-unusable
        // reference misses the byte/formats/pixel limits and fails closed;
        // absent is fine — it simply means text-to-video.
        let validInput = limits.inputReference.map { reference in
            reference.accepted == false
                || (
                    reference.maximumBytes > 0
                        && !reference.formats.isEmpty
                        && !acceptedReferenceMIMETypes.isEmpty
                        && (reference.maximumPixels.map { $0 > 0 } ?? true)
                )
        } ?? true
        guard !model.isEmpty,
              !family.isEmpty,
              !modes.isEmpty,
              validSize,
              seconds.minimum > 0,
              seconds.maximum >= seconds.minimum,
              Self.contains(seconds.default, minimum: seconds.minimum, maximum: seconds.maximum),
              fps.minimum > 0,
              fps.maximum >= fps.minimum,
              Self.contains(fps.default, minimum: fps.minimum, maximum: fps.maximum),
              frames.minimum > 0,
              frames.maximum >= frames.minimum,
              frames.step > 0,
              frames.offset >= 0,
              frames.offset <= frames.maximum,
              workload.metric == "pixel_frames",
              workload.maximum > 0,
              workload.alignmentStep != nil,
              validInput else {
            throw VideoClientError.invalidResponse
        }
        return self
    }

    private func workloadAllows(seconds: Int, size: String) -> Bool {
        guard limits.workload.metric == "pixel_frames",
              let dimensions = Self.parseSize(size),
              seconds > 0,
              limits.fps.default > 0 else { return false }
        // Workload normalization is a separate server contract from the
        // request-size alignment. The rounding step is taken from the
        // server-advertised dimension_rounding vocabulary (see WorkloadLimit).
        guard let step = limits.workload.alignmentStep else { return false }
        guard let width = Self.roundUp(dimensions.width, to: step),
              let height = Self.roundUp(dimensions.height, to: step) else { return false }
        let frameStep = max(1, limits.frames.step)
        let frameOffset = limits.frames.offset
        guard frameOffset >= 0 else { return false }
        let (requestedFrames, requestOverflow) = seconds.multipliedReportingOverflow(
            by: limits.fps.default
        )
        guard !requestOverflow else { return false }
        let frameDelta = requestedFrames > frameOffset
            ? requestedFrames - frameOffset : 0
        let (roundedDelta, roundingOverflow) = frameDelta.addingReportingOverflow(frameStep - 1)
        guard !roundingOverflow else { return false }
        let steps = roundedDelta / frameStep
        let (steppedFrames, stepOverflow) = steps.multipliedReportingOverflow(by: frameStep)
        let (normalizedFrames, offsetOverflow) = steppedFrames.addingReportingOverflow(frameOffset)
        guard !stepOverflow, !offsetOverflow else { return false }
        // H3 bounds its seconds-to-frame normalization at the advertised
        // ceiling, so the longest duration remains selectable on its grid.
        let frames = family == "minimax-h3"
            ? min(limits.frames.maximum, max(limits.frames.minimum, normalizedFrames))
            : max(limits.frames.minimum, normalizedFrames)
        guard frames <= limits.frames.maximum else { return false }
        let (pixelCount, pixelOverflow) = width.multipliedReportingOverflow(by: height)
        let (workload, workloadOverflow) = pixelCount.multipliedReportingOverflow(by: frames)
        return !pixelOverflow && !workloadOverflow && workload <= limits.workload.maximum
    }

    private static func parseSize(_ value: String) -> (width: Int, height: Int)? {
        let parts = value.lowercased().split(separator: "x", maxSplits: 1)
        guard parts.count == 2,
              let width = Int(parts[0]), let height = Int(parts[1]),
              width > 0, height > 0 else { return nil }
        return (width, height)
    }

    private static func contains(_ value: Int, minimum: Int, maximum: Int) -> Bool {
        minimum <= maximum && value >= minimum && value <= maximum
    }

    private static func product(_ lhs: Int, _ rhs: Int) -> Int? {
        let (value, overflow) = lhs.multipliedReportingOverflow(by: rhs)
        return overflow ? nil : value
    }

    private static func roundUp(_ value: Int, to multiple: Int) -> Int? {
        guard value >= 0, multiple > 0 else { return nil }
        let (offsetValue, additionOverflow) = value.addingReportingOverflow(multiple - 1)
        guard !additionOverflow else { return nil }
        let quotient = offsetValue / multiple
        let (result, multiplicationOverflow) = quotient.multipliedReportingOverflow(by: multiple)
        return multiplicationOverflow ? nil : result
    }
}

struct VideoAttachment: Sendable, Equatable {
    let data: Data
    let fileName: String
    let mimeType: String

    var kind: String? {
        ["image", "video", "audio"].first { mimeType.hasPrefix($0 + "/") }
    }

    static func mimeType(for format: String) -> String? {
        switch format.lowercased() {
        case "jpg", "jpeg", "image/jpeg": "image/jpeg"
        case "png", "image/png": "image/png"
        case "webp", "image/webp": "image/webp"
        case "mp4", "video/mp4": "video/mp4"
        case "mov", "video/quicktime": "video/quicktime"
        case "webm", "video/webm": "video/webm"
        case "mkv", "video/x-matroska": "video/x-matroska"
        case "wav", "audio/wav": "audio/wav"
        case "mp3", "audio/mpeg": "audio/mpeg"
        case "m4a", "audio/mp4": "audio/mp4"
        case "flac", "audio/flac": "audio/flac"
        case "aac", "audio/aac": "audio/aac"
        case "ogg", "audio/ogg": "audio/ogg"
        default: nil
        }
    }

    static func contentType(for mime: String) -> UTType? {
        switch mime {
        case "video/webm": UTType(filenameExtension: "webm") ?? UTType(importedAs: "org.webmproject.webm", conformingTo: .movie)
        case "video/x-matroska": UTType(filenameExtension: "mkv") ?? UTType(importedAs: "org.matroska.mkv", conformingTo: .movie)
        case "audio/ogg": UTType(filenameExtension: "ogg") ?? UTType(importedAs: "org.xiph.ogg-audio", conformingTo: .audio)
        default: UTType(mimeType: mime)
        }
    }

}

struct VideoCreateRequest: Sendable, Equatable {
    let prompt: String
    let model: String
    let seconds: Int
    let size: String
    let seed: Int
    let reference: Data?
    let referenceFileName: String?
    let referenceMIMEType: String?
    var lastFrame: VideoAttachment? = nil
    var references: [VideoAttachment] = []
}

enum VideoClientError: Error, LocalizedError, Equatable {
    case notReady
    case http(status: Int, message: String?)
    case cacheRemoval
    case invalidJobID
    case invalidResponse
    case transport(String)

    var errorDescription: String? {
        switch self {
        case .notReady:
            return "The video model isn't running yet."
        case let .http(status, message):
            return message ?? "Video request failed (HTTP \(status))."
        case .cacheRemoval:
            return "Rapid couldn't remove the cached video. Check file access and try again."
        case .invalidJobID:
            return "The video server returned an invalid job identifier."
        case .invalidResponse:
            return "The video server returned an invalid response."
        case let .transport(message):
            return message
        }
    }
}

protocol VideoClientProtocol: Sendable {
    func capabilities(port: Int, bearer: String?) async throws -> VideoCapabilities
    func create(_ request: VideoCreateRequest, port: Int, bearer: String?) async throws -> VideoJob
    func list(port: Int, bearer: String?, limit: Int) async throws -> [VideoJob]
    func cancel(id: String, port: Int, bearer: String?) async throws
    func delete(id: String, port: Int, bearer: String?) async throws
    func content(id: String, port: Int, bearer: String?) async throws -> URL
}

struct VideoClient: VideoClientProtocol, @unchecked Sendable {
    static let maxReferenceBytes = 20 * 1024 * 1024
    static let requestTimeout: TimeInterval = 60

    static let sharedSession: URLSession = {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = requestTimeout
        configuration.timeoutIntervalForResource = 30 * 60
        return URLSession(configuration: configuration)
    }()

    var session: URLSession = sharedSession
    var cacheDirectory: URL = FileManager.default.urls(
        for: .cachesDirectory, in: .userDomainMask
    )[0].appendingPathComponent("Rapid/VideoPreviews", isDirectory: true)
    var removeCachedItem: @Sendable (URL) throws -> Void = {
        try FileManager.default.removeItem(at: $0)
    }

    func capabilities(port: Int, bearer: String?) async throws -> VideoCapabilities {
        let value: VideoCapabilities = try await decode(
            request(path: "v1/videos/capabilities", port: port, bearer: bearer)
        )
        return try value.validated()
    }

    func create(
        _ value: VideoCreateRequest,
        port: Int,
        bearer: String?
    ) async throws -> VideoJob {
        var request = request(path: "v1/videos", port: port, bearer: bearer)
        request.httpMethod = "POST"
        request.timeoutInterval = Self.requestTimeout
        let boundary = "rapid-video-\(UUID().uuidString)"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        request.httpBody = Self.multipartBody(
            boundary: boundary,
            fields: [
                ("prompt", value.prompt),
                ("model", value.model),
                ("seconds", String(value.seconds)),
                ("size", value.size),
                ("seed", String(value.seed)),
            ],
            file: value.reference.map {
                (
                    field: "input_reference",
                    name: value.referenceFileName ?? "reference.png",
                    mime: value.referenceMIMEType ?? "image/png",
                    data: $0
                )
            },
            files: (value.lastFrame.map { [(field: "last_frame", name: $0.fileName, mime: $0.mimeType, data: $0.data)] } ?? [])
                + value.references.map { (field: "references", name: $0.fileName, mime: $0.mimeType, data: $0.data) }
        )
        let job: VideoJob = try await decode(request)
        guard Self.isValidJobID(job.id) else { throw VideoClientError.invalidJobID }
        return job
    }

    func list(port: Int, bearer: String?, limit: Int = 30) async throws -> [VideoJob] {
        var components = URLComponents(
            url: Self.loopbackURL(port: port).appendingPathComponent("v1/videos"),
            resolvingAgainstBaseURL: false
        )!
        components.queryItems = [URLQueryItem(name: "limit", value: String(limit))]
        var request = URLRequest(url: components.url!)
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        applyBearer(&request, bearer)
        let envelope: ListEnvelope = try await decode(request)
        guard envelope.data.allSatisfy({ Self.isValidJobID($0.id) }) else {
            throw VideoClientError.invalidJobID
        }
        return envelope.data
    }

    func cancel(id: String, port: Int, bearer: String?) async throws {
        guard Self.isValidJobID(id) else { throw VideoClientError.invalidJobID }
        var request = request(path: "v1/videos/\(id)", port: port, bearer: bearer)
        request.httpMethod = "DELETE"
        do {
            _ = try await send(request)
        } catch let VideoClientError.http(status, _) where status == 404 {
            // The job may already have been removed by another request.
        }
    }

    func delete(id: String, port: Int, bearer: String?) async throws {
        let cached = try cacheURL(for: id)
        var request = request(path: "v1/videos/\(id)", port: port, bearer: bearer)
        request.httpMethod = "DELETE"
        do {
            _ = try await send(request)
        } catch let VideoClientError.http(status, _) where status == 404 {
            // DELETE is idempotent: a prior attempt may have removed the
            // server job before local cache cleanup failed.
        }
        if FileManager.default.fileExists(atPath: cached.path) {
            do {
                try removeCachedItem(cached)
            } catch {
                throw VideoClientError.cacheRemoval
            }
        }
    }

    func content(id: String, port: Int, bearer: String?) async throws -> URL {
        let destination = try cacheURL(for: id)
        if FileManager.default.fileExists(atPath: destination.path) { return destination }
        try FileManager.default.createDirectory(
            at: cacheDirectory,
            withIntermediateDirectories: true,
            attributes: [.posixPermissions: 0o700]
        )
        let request = request(path: "v1/videos/\(id)/content", port: port, bearer: bearer)
        do {
            let (temporary, response) = try await session.download(for: request)
            try Self.validate(response: response, data: nil)
            let staging = cacheDirectory.appendingPathComponent(".\(UUID().uuidString).mp4")
            try FileManager.default.moveItem(at: temporary, to: staging)
            do {
                try FileManager.default.moveItem(at: staging, to: destination)
            } catch {
                try? FileManager.default.removeItem(at: staging)
                guard FileManager.default.fileExists(atPath: destination.path) else {
                    throw error
                }
            }
            return destination
        } catch let error as VideoClientError {
            throw error
        } catch {
            throw VideoClientError.transport(error.localizedDescription)
        }
    }

    static func loopbackURL(port: Int) -> URL {
        URL(string: "http://127.0.0.1:\(port)")!
    }

    static func multipartBody(
        boundary: String,
        fields: [(String, String)],
        file: (field: String, name: String, mime: String, data: Data)?,
        files: [(field: String, name: String, mime: String, data: Data)] = []
    ) -> Data {
        var body = Data()
        func append(_ string: String) { body.append(Data(string.utf8)) }
        for (name, value) in fields {
            append("--\(boundary)\r\n")
            append("Content-Disposition: form-data; name=\"\(name)\"\r\n\r\n")
            append(value)
            append("\r\n")
        }
        for file in (file.map { [$0] } ?? []) + files {
            let fileName = escapedMultipartFilename(file.name)
            append("--\(boundary)\r\n")
            append("Content-Disposition: form-data; name=\"\(file.field)\"; filename=\"\(fileName)\"\r\n")
            append("Content-Type: \(file.mime)\r\n\r\n")
            body.append(file.data)
            append("\r\n")
        }
        append("--\(boundary)--\r\n")
        return body
    }

    /// Multipart quoted strings must not let a local filename terminate the
    /// header or inject another one. Preserve the user-visible name while
    /// escaping quoted-string delimiters and replacing control bytes.
    static func escapedMultipartFilename(_ value: String) -> String {
        let source = value.isEmpty ? "reference.png" : value
        var escaped = ""
        for character in source {
            if character.unicodeScalars.contains(where: {
                $0.value < 0x20 || $0.value == 0x7F
            }) {
                escaped.append("_")
            } else {
                if character == "\"" || character == "\\" {
                    escaped.append("\\")
                }
                escaped.append(character)
            }
        }
        return escaped
    }

    static func cacheFileName(for id: String) throws -> String {
        guard isValidJobID(id) else {
            throw VideoClientError.invalidJobID
        }
        let digest = SHA256.hash(data: Data(id.utf8))
            .map { String(format: "%02x", $0) }
            .joined()
        return "\(digest).mp4"
    }

    private static func isValidJobID(_ id: String) -> Bool {
        !id.isEmpty && id.utf8.count <= 128
            && id.utf8.allSatisfy {
                ($0 >= 48 && $0 <= 57)
                    || ($0 >= 65 && $0 <= 90)
                    || ($0 >= 97 && $0 <= 122)
                    || $0 == 45 || $0 == 95
            }
    }

    private struct ListEnvelope: Decodable { let data: [VideoJob] }

    private struct ErrorEnvelope: Decodable {
        struct Inner: Decodable { let message: String? }
        struct Detail: Decodable { let error: Inner? }
        let error: Inner?
        let detailText: String?
        let detailObject: Detail?

        enum CodingKeys: String, CodingKey { case error, detail }

        init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            error = try container.decodeIfPresent(Inner.self, forKey: .error)
            detailText = try? container.decode(String.self, forKey: .detail)
            detailObject = try? container.decode(Detail.self, forKey: .detail)
        }

        var message: String? {
            error?.message ?? detailObject?.error?.message ?? detailText
        }
    }

    private func request(path: String, port: Int, bearer: String?) -> URLRequest {
        var request = URLRequest(url: Self.loopbackURL(port: port).appendingPathComponent(path))
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        applyBearer(&request, bearer)
        return request
    }

    /// Both the shared ``request(path:port:bearer:)`` builder and the ad-hoc
    /// ``list`` request route through here, so the desktop client header rides
    /// along with the per-launch bearer.
    private func applyBearer(_ request: inout URLRequest, _ bearer: String?) {
        request.applyRapidClientHeader()
        if let bearer, !bearer.isEmpty {
            request.setValue("Bearer \(bearer)", forHTTPHeaderField: "Authorization")
        }
    }

    private func decode<T: Decodable>(_ request: URLRequest) async throws -> T {
        let data = try await send(request)
        do {
            return try JSONDecoder().decode(T.self, from: data)
        } catch {
            throw VideoClientError.invalidResponse
        }
    }

    private func send(_ request: URLRequest) async throws -> Data {
        do {
            let (data, response) = try await session.data(for: request)
            try Self.validate(response: response, data: data)
            return data
        } catch let error as VideoClientError {
            throw error
        } catch {
            throw VideoClientError.transport(error.localizedDescription)
        }
    }

    private static func validate(response: URLResponse, data: Data?) throws {
        guard let http = response as? HTTPURLResponse else {
            throw VideoClientError.invalidResponse
        }
        guard (200...299).contains(http.statusCode) else {
            let message = data.flatMap {
                try? JSONDecoder().decode(ErrorEnvelope.self, from: $0).message
            }
            throw VideoClientError.http(status: http.statusCode, message: message)
        }
    }

    private func cacheURL(for id: String) throws -> URL {
        cacheDirectory.appendingPathComponent(try Self.cacheFileName(for: id))
    }
}
