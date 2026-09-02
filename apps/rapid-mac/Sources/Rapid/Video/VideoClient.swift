import CryptoKit
import Foundation

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

    enum CodingKeys: String, CodingKey {
        case id, model, prompt, seconds, size, status, progress, error
        case createdAt = "created_at"
        case completedAt = "completed_at"
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
        }

        struct InputReferenceLimit: Decodable, Sendable, Hashable {
            let accepted: Bool
            let maximumBytes: Int
            let formats: [String]

            enum CodingKeys: String, CodingKey {
                case accepted, formats
                case maximumBytes = "maximum_bytes"
            }
        }

        let size: SizeLimit
        let seconds: SecondsLimit
        let fps: FPSLimit
        let frames: FramesLimit
        let workload: WorkloadLimit
        let inputReference: InputReferenceLimit?

        init(
            size: SizeLimit,
            seconds: SecondsLimit,
            fps: FPSLimit,
            frames: FramesLimit,
            workload: WorkloadLimit,
            inputReference: InputReferenceLimit? = nil
        ) {
            self.size = size
            self.seconds = seconds
            self.fps = fps
            self.frames = frames
            self.workload = workload
            self.inputReference = inputReference
        }

        enum CodingKeys: String, CodingKey {
            case size, seconds, fps, frames, workload
            case inputReference = "input_reference"
        }
    }

    let model: String
    let family: String
    let modes: [VideoModelCapability]
    let limits: Limits

    var acceptedReferenceMIMETypes: Set<String> {
        guard let input = limits.inputReference, input.accepted else { return [] }
        return Set(input.formats.compactMap { format in
            switch format.lowercased() {
            case "jpeg", "jpg", "image/jpeg": "image/jpeg"
            case "png", "image/png": "image/png"
            case "webp", "image/webp": "image/webp"
            default: nil
            }
        })
    }

    var supportsImageInput: Bool {
        modes.contains(.imageToVideo)
            && referenceMaximumBytes > 0
            && !acceptedReferenceMIMETypes.isEmpty
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
        let candidates = Set([limits.seconds.minimum, 1, 2, 4]).sorted()
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

    var referenceMaximumBytes: Int {
        guard let inputReference = limits.inputReference,
              inputReference.accepted,
              inputReference.maximumBytes > 0 else { return 0 }
        return min(inputReference.maximumBytes, VideoClient.maxReferenceBytes)
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
        let validInput = limits.inputReference.map {
            $0.maximumBytes >= 0
                && (!$0.accepted || ($0.maximumBytes > 0 && !acceptedReferenceMIMETypes.isEmpty))
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
              workload.dimensionRounding == "multiple_of_64",
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
        // request-size alignment. This client validates only multiple_of_64.
        guard let width = Self.roundUp(dimensions.width, to: 64),
              let height = Self.roundUp(dimensions.height, to: 64) else { return false }
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
        let frames = max(limits.frames.minimum, normalizedFrames)
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

struct VideoCreateRequest: Sendable, Equatable {
    let prompt: String
    let model: String
    let seconds: Int
    let size: String
    let seed: Int
    let reference: Data?
    let referenceFileName: String?
    let referenceMIMEType: String?
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
            }
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
        file: (field: String, name: String, mime: String, data: Data)?
    ) -> Data {
        var body = Data()
        func append(_ string: String) { body.append(Data(string.utf8)) }
        for (name, value) in fields {
            append("--\(boundary)\r\n")
            append("Content-Disposition: form-data; name=\"\(name)\"\r\n\r\n")
            append(value)
            append("\r\n")
        }
        if let file {
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

    private func applyBearer(_ request: inout URLRequest, _ bearer: String?) {
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
