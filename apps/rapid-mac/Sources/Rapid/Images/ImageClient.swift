import Foundation

/// One generated (or edited) image plus the seed that produced it, so the
/// gallery can label results and an edit can re-target a specific image.
struct GeneratedImage: Identifiable, Hashable, Sendable {
    let id = UUID()
    /// Decoded PNG bytes (from the API's ``b64_json``).
    let pngData: Data
    /// The prompt that produced this image — shown as the gallery caption.
    let prompt: String
    /// True when this came from ``/v1/images/edits`` (vs. generations).
    let isEdit: Bool
}

/// Errors surfaced to the Images tab. Mirrors ``ChatStreamError`` in shape:
/// a small, user-actionable enum rather than raw ``URLError``/decode noise.
enum ImageClientError: Error, LocalizedError {
    case notReady
    case http(status: Int, message: String?)
    case emptyResponse
    case transport(String)

    var errorDescription: String? {
        switch self {
        case .notReady:
            return "The image model isn't running yet."
        case let .http(status, message):
            return message ?? "Image request failed (HTTP \(status))."
        case .emptyResponse:
            return "The server returned no image."
        case let .transport(detail):
            return detail
        }
    }
}

/// HTTP client for the OpenAI-compatible image endpoints. Non-streaming
/// (unlike ``ChatStreamClient``): a single request/response per image batch,
/// so it uses ``session.data(for:)`` like ``ServerProfileFetcher`` rather
/// than the SSE byte loop.
///
/// Port and bearer are passed per call — the caller reads
/// ``ServerManager.activePort`` / ``activeBearer`` at request time (they can
/// change across a stop/start reload), never caching them.
struct ImageClient {
    static let maxEditImageBytes = 25 * 1024 * 1024
    /// Keep enough headroom for cold diffusion model loads and slower hardware;
    /// progress polling and Cancel remain separate short requests.
    static let requestTimeout: TimeInterval = 30 * 60

    static let sharedSession: URLSession = {
        let config = URLSessionConfiguration.ephemeral
        config.timeoutIntervalForRequest = requestTimeout
        config.timeoutIntervalForResource = requestTimeout
        return URLSession(configuration: config)
    }()

    var session: URLSession = ImageClient.sharedSession

    static func loopbackURL(port: Int) -> URL {
        URL(string: "http://127.0.0.1:\(port)")!
    }

    // MARK: - Wire types

    private struct GenerationBody: Encodable {
        let model: String
        let prompt: String
        let n: Int
        let size: String
        let response_format = "b64_json"
        let seed: Int?
    }

    private struct ImageResponse: Decodable {
        struct Item: Decodable { let b64_json: String? }
        let data: [Item]
        let cancelled: Bool?
    }

    private struct ErrorEnvelope: Decodable {
        struct Inner: Decodable { let message: String? }
        struct Detail: Decodable { let error: Inner? }
        let error: Inner?
        let detail: Detail?

        var message: String? { error?.message ?? detail?.error?.message }
    }

    // MARK: - Generations

    /// ``POST /v1/images/generations`` — text→image.
    func generate(
        prompt: String,
        model: String,
        size: String,
        count: Int,
        seed: Int?,
        port: Int,
        bearer: String?
    ) async throws -> [GeneratedImage] {
        let url = Self.loopbackURL(port: port).appendingPathComponent("v1/images/generations")
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.timeoutInterval = Self.requestTimeout
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.setValue("application/json", forHTTPHeaderField: "Accept")
        applyBearer(&req, bearer)
        req.httpBody = try JSONEncoder().encode(
            GenerationBody(model: model, prompt: prompt, n: count, size: size, seed: seed)
        )
        let images = try await sendAndDecode(req)
        return images.map { GeneratedImage(pngData: $0, prompt: prompt, isEdit: false) }
    }

    // MARK: - Edits

    /// ``POST /v1/images/edits`` — image + instruction → image. Multipart,
    /// built by hand (there is no shared multipart helper in the app).
    func edit(
        imagePNG: Data,
        prompt: String,
        model: String,
        count: Int,
        seed: Int?,
        port: Int,
        bearer: String?
    ) async throws -> [GeneratedImage] {
        let url = Self.loopbackURL(port: port).appendingPathComponent("v1/images/edits")
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.timeoutInterval = Self.requestTimeout
        req.setValue("application/json", forHTTPHeaderField: "Accept")
        applyBearer(&req, bearer)

        let boundary = "rapid-\(UUID().uuidString)"
        req.setValue(
            "multipart/form-data; boundary=\(boundary)",
            forHTTPHeaderField: "Content-Type"
        )
        var fields: [(String, String)] = [
            ("prompt", prompt),
            ("model", model),
            ("n", String(count)),
            ("response_format", "b64_json"),
        ]
        if let seed { fields.append(("seed", String(seed))) }
        req.httpBody = Self.multipartBody(
            boundary: boundary, fields: fields,
            fileField: "image", fileName: "input.png",
            fileMime: "image/png", fileData: imagePNG
        )
        let images = try await sendAndDecode(req)
        return images.map { GeneratedImage(pngData: $0, prompt: prompt, isEdit: true) }
    }

    // MARK: - Shared

    private func applyBearer(_ req: inout URLRequest, _ bearer: String?) {
        if let bearer, !bearer.isEmpty {
            req.setValue("Bearer \(bearer)", forHTTPHeaderField: "Authorization")
        }
    }

    /// Send, validate status, decode the ``{data:[{b64_json}]}`` envelope
    /// into raw PNG byte blobs.
    private func sendAndDecode(_ req: URLRequest) async throws -> [Data] {
        let data: Data
        let response: URLResponse
        do {
            (data, response) = try await session.data(for: req)
        } catch {
            throw ImageClientError.transport(error.localizedDescription)
        }
        guard let http = response as? HTTPURLResponse else {
            throw ImageClientError.transport("Malformed server response.")
        }
        guard (200...299).contains(http.statusCode) else {
            let message = (try? JSONDecoder().decode(ErrorEnvelope.self, from: data))?.message
            throw ImageClientError.http(status: http.statusCode, message: message)
        }
        guard let decoded = try? JSONDecoder().decode(ImageResponse.self, from: data) else {
            throw ImageClientError.emptyResponse
        }
        // A cancel that lands before the first image finishes returns an empty,
        // non-error batch — surface it as "no images" rather than a failure.
        if decoded.data.isEmpty {
            if decoded.cancelled == true { return [] }
            throw ImageClientError.emptyResponse
        }
        // Fail on any malformed item rather than silently dropping it — a
        // "successful" batch that quietly returns fewer images than requested
        // would hide server corruption behind a partial gallery.
        var blobs: [Data] = []
        for item in decoded.data {
            guard let b64 = item.b64_json, let data = Data(base64Encoded: b64) else {
                throw ImageClientError.emptyResponse
            }
            blobs.append(data)
        }
        return blobs
    }

    // MARK: - Progress & cancel

    /// A live snapshot of the single in-flight render. `running` is false
    /// during the cold model-load phase (before the denoise loop starts) and
    /// after it finishes; `step`/`total` drive a determinate progress bar.
    struct ImageProgress: Decodable, Sendable {
        let running: Bool
        let step: Int
        let total: Int
        let elapsedMs: Int
        enum CodingKeys: String, CodingKey {
            case running, step, total, elapsedMs = "elapsed_ms"
        }
        /// Fraction complete in [0, 1], or nil when the step total is unknown.
        var fraction: Double? {
            total > 0 ? min(1, Double(step) / Double(total)) : nil
        }
    }

    /// ``GET /v1/images/progress`` — polled during a render. Returns nil on any
    /// transport hiccup so the caller simply keeps its last known state.
    func fetchProgress(model: String, port: Int, bearer: String?) async -> ImageProgress? {
        var components = URLComponents(
            url: Self.loopbackURL(port: port).appendingPathComponent("v1/images/progress"),
            resolvingAgainstBaseURL: false
        )!
        components.queryItems = [URLQueryItem(name: "model", value: model)]
        let url = components.url!
        var req = URLRequest(url: url)
        req.timeoutInterval = 5
        req.setValue("application/json", forHTTPHeaderField: "Accept")
        applyBearer(&req, bearer)
        guard let (data, response) = try? await session.data(for: req),
              let http = response as? HTTPURLResponse,
              (200...299).contains(http.statusCode),
              let progress = try? JSONDecoder().decode(ImageProgress.self, from: data)
        else { return nil }
        return progress
    }

    /// ``POST /v1/images/cancel`` — best-effort; the render stops at its next
    /// denoise step and its in-flight ``generate`` returns the finished images.
    func cancel(model: String, port: Int, bearer: String?) async {
        var components = URLComponents(
            url: Self.loopbackURL(port: port).appendingPathComponent("v1/images/cancel"),
            resolvingAgainstBaseURL: false
        )!
        components.queryItems = [URLQueryItem(name: "model", value: model)]
        let url = components.url!
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.timeoutInterval = 5
        applyBearer(&req, bearer)
        _ = try? await session.data(for: req)
    }

    /// Assemble a multipart/form-data body from text fields + one file part.
    static func multipartBody(
        boundary: String,
        fields: [(String, String)],
        fileField: String,
        fileName: String,
        fileMime: String,
        fileData: Data
    ) -> Data {
        var body = Data()
        let dashes = "--\(boundary)\r\n"
        for (name, value) in fields {
            body.append(Data(dashes.utf8))
            body.append(Data("Content-Disposition: form-data; name=\"\(name)\"\r\n\r\n".utf8))
            body.append(Data("\(value)\r\n".utf8))
        }
        body.append(Data(dashes.utf8))
        body.append(Data(
            "Content-Disposition: form-data; name=\"\(fileField)\"; filename=\"\(fileName)\"\r\n".utf8
        ))
        body.append(Data("Content-Type: \(fileMime)\r\n\r\n".utf8))
        body.append(fileData)
        body.append(Data("\r\n".utf8))
        body.append(Data("--\(boundary)--\r\n".utf8))
        return body
    }
}
