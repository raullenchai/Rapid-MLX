import Foundation
import AVFoundation

struct AudioTranscriptionResult: Equatable, Sendable {
    let text: String
    let language: String?
    let duration: Double?
}

struct SynthesizedAudio: Equatable, Sendable {
    let data: Data
    let contentType: String

    var fileExtension: String {
        switch contentType.lowercased() {
        case let type where type.contains("mpeg"): return "mp3"
        case let type where type.contains("flac"): return "flac"
        case let type where type.contains("ogg"): return "ogg"
        default: return "wav"
        }
    }
}

enum AudioClientError: Error, LocalizedError, Equatable {
    case fileTooLarge(maxBytes: Int)
    case unreadableFile(String)
    case http(status: Int, message: String?)
    case invalidResponse
    case emptyAudio
    case transport(String)

    var errorDescription: String? {
        switch self {
        case .fileTooLarge(let maxBytes):
            return "That file is larger than \(maxBytes / 1024 / 1024) MB. Choose a shorter recording."
        case .unreadableFile(let detail):
            return "The audio file couldn't be read: \(detail)"
        case .http(let status, let message):
            return message ?? "Audio request failed (HTTP \(status))."
        case .invalidResponse:
            return "The audio server returned an unreadable response."
        case .emptyAudio:
            return "The audio server returned no sound."
        case .transport(let detail):
            return detail
        }
    }
}

/// Native loopback client for the OpenAI-compatible audio routes. Port and
/// bearer are supplied per request because both rotate when ServerManager
/// replaces the single resident model.
struct AudioClient {
    static let maxUploadBytes = 25 * 1024 * 1024
    static let requestTimeout: TimeInterval = 30 * 60

    static let sharedSession: URLSession = {
        let config = URLSessionConfiguration.ephemeral
        config.timeoutIntervalForRequest = requestTimeout
        config.timeoutIntervalForResource = requestTimeout
        return URLSession(configuration: config)
    }()

    var session: URLSession = AudioClient.sharedSession

    private struct TranscriptionWire: Decodable {
        let text: String
        let language: String?
        let duration: Double?

        private enum CodingKeys: String, CodingKey {
            case text, language, duration
        }

        init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            text = try container.decode(String.self, forKey: .text)
            duration = try? container.decodeIfPresent(Double.self, forKey: .duration)

            // `language` is not one shape across backends: Whisper reports a
            // single code ("zh"), Qwen3-ASR reports a list (["Chinese"]) because
            // it detects per segment. Decoding it as a plain String made the
            // WHOLE response fail to decode against a Qwen3-ASR reply, which the
            // UI then reported as "the audio server returned an unreadable
            // response" — pointing at the server for a client-side type error.
            if let single = try? container.decodeIfPresent(String.self, forKey: .language) {
                language = single
            } else if let many = try? container.decodeIfPresent([String].self, forKey: .language) {
                language = many.first
            } else {
                language = nil
            }
        }
    }

    private struct VoicesWire: Decodable { let voices: [String] }

    private struct SpeechBody: Encodable {
        let model: String
        let input: String
        let voice: String
        let speed: Double
        let response_format = "wav"
    }

    func transcribe(
        fileURL: URL,
        model: String,
        port: Int,
        bearer: String?
    ) async throws -> AudioTranscriptionResult {
        let upload: AudioUpload
        do {
            upload = try await Task.detached(priority: .userInitiated) {
                let values = try fileURL.resourceValues(forKeys: [.fileSizeKey])
                if let size = values.fileSize, size > Self.maxUploadBytes {
                    throw AudioClientError.fileTooLarge(maxBytes: Self.maxUploadBytes)
                }
                let data = try Data(contentsOf: fileURL, options: [.mappedIfSafe])
                guard data.count <= Self.maxUploadBytes else {
                    throw AudioClientError.fileTooLarge(maxBytes: Self.maxUploadBytes)
                }
                let ext = Self.safeExtension(fileURL.pathExtension)
                if AudioUploadTranscoder.requiresWAV(ext) {
                    let wav = try AudioUploadTranscoder.wavData(
                        from: fileURL,
                        maxBytes: Self.maxUploadBytes
                    )
                    return AudioUpload(data: wav, fileExtension: "wav", mimeType: "audio/wav")
                }
                return AudioUpload(
                    data: data,
                    fileExtension: ext,
                    mimeType: Self.mimeType(forExtension: ext)
                )
            }.value
        } catch let error as AudioClientError {
            throw error
        } catch {
            throw AudioClientError.unreadableFile(error.localizedDescription)
        }

        let boundary = "rapid-audio-\(UUID().uuidString)"
        var request = URLRequest(
            url: Self.loopbackURL(port: port)
                .appendingPathComponent("v1/audio/transcriptions")
        )
        request.httpMethod = "POST"
        request.timeoutInterval = Self.requestTimeout
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        request.setValue(
            "multipart/form-data; boundary=\(boundary)",
            forHTTPHeaderField: "Content-Type"
        )
        applyBearer(&request, bearer)
        request.httpBody = ImageClient.multipartBody(
            boundary: boundary,
            fields: [("model", model), ("response_format", "json")],
            fileField: "file",
            fileName: "input.\(upload.fileExtension)",
            fileMime: upload.mimeType,
            fileData: upload.data
        )

        let (data, response) = try await send(request)
        try validate(response: response, data: data)
        guard let decoded = try? JSONDecoder().decode(TranscriptionWire.self, from: data) else {
            throw AudioClientError.invalidResponse
        }
        return AudioTranscriptionResult(
            text: decoded.text,
            language: decoded.language,
            duration: decoded.duration
        )
    }

    /// Transcribe audio already held in memory.
    ///
    /// Dictation captures straight into a 16 kHz mono WAV buffer, so routing it
    /// through the file-based path above would mean a pointless write/read of a
    /// temporary file on the latency-critical path.
    ///
    /// `context` carries the user's proper-noun list. The server maps it onto
    /// whichever decoding-hint parameter the loaded backend exposes
    /// (`initial_prompt` for whisper, `system_prompt` for Qwen3-ASR); it is
    /// omitted entirely when empty, because an empty hint still consumes
    /// decoder attention.
    func transcribe(
        audioData: Data,
        model: String,
        context: String?,
        port: Int,
        bearer: String?
    ) async throws -> AudioTranscriptionResult {
        guard audioData.count <= Self.maxUploadBytes else {
            throw AudioClientError.fileTooLarge(maxBytes: Self.maxUploadBytes)
        }
        guard !audioData.isEmpty else { throw AudioClientError.emptyAudio }

        let boundary = "rapid-dictation-\(UUID().uuidString)"
        var request = URLRequest(
            url: Self.loopbackURL(port: port)
                .appendingPathComponent("v1/audio/transcriptions")
        )
        request.httpMethod = "POST"
        request.timeoutInterval = Self.requestTimeout
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        request.setValue(
            "multipart/form-data; boundary=\(boundary)",
            forHTTPHeaderField: "Content-Type"
        )
        applyBearer(&request, bearer)

        var fields = [("model", model), ("response_format", "json")]
        if let context, !context.isEmpty {
            fields.append(("context", context))
        }

        request.httpBody = ImageClient.multipartBody(
            boundary: boundary,
            fields: fields,
            fileField: "file",
            fileName: "dictation.wav",
            fileMime: "audio/wav",
            fileData: audioData
        )

        let (data, response) = try await send(request)
        try validate(response: response, data: data)
        guard let decoded = try? JSONDecoder().decode(TranscriptionWire.self, from: data) else {
            throw AudioClientError.invalidResponse
        }
        return AudioTranscriptionResult(
            text: decoded.text,
            language: decoded.language,
            duration: decoded.duration
        )
    }

    func voices(
        model: String,
        port: Int,
        bearer: String?
    ) async throws -> [String] {
        var components = URLComponents(
            url: Self.loopbackURL(port: port).appendingPathComponent("v1/audio/voices"),
            resolvingAgainstBaseURL: false
        )!
        components.queryItems = [URLQueryItem(name: "model", value: model)]
        var request = URLRequest(url: components.url!)
        request.timeoutInterval = Self.requestTimeout
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        applyBearer(&request, bearer)
        let (data, response) = try await send(request)
        try validate(response: response, data: data)
        guard let decoded = try? JSONDecoder().decode(VoicesWire.self, from: data) else {
            throw AudioClientError.invalidResponse
        }
        return decoded.voices
    }

    func synthesize(
        text: String,
        model: String,
        voice: String,
        speed: Double,
        port: Int,
        bearer: String?
    ) async throws -> SynthesizedAudio {
        var request = URLRequest(
            url: Self.loopbackURL(port: port).appendingPathComponent("v1/audio/speech")
        )
        request.httpMethod = "POST"
        request.timeoutInterval = Self.requestTimeout
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("audio/wav", forHTTPHeaderField: "Accept")
        applyBearer(&request, bearer)
        request.httpBody = try JSONEncoder().encode(
            SpeechBody(model: model, input: text, voice: voice, speed: speed)
        )
        let (data, response) = try await send(request)
        try validate(response: response, data: data)
        guard !data.isEmpty else { throw AudioClientError.emptyAudio }
        let contentType = (response as? HTTPURLResponse)?
            .value(forHTTPHeaderField: "Content-Type") ?? "audio/wav"
        return SynthesizedAudio(data: data, contentType: contentType)
    }

    private static func loopbackURL(port: Int) -> URL {
        URL(string: "http://127.0.0.1:\(port)")!
    }

    private func applyBearer(_ request: inout URLRequest, _ bearer: String?) {
        if let bearer, !bearer.isEmpty {
            request.setValue("Bearer \(bearer)", forHTTPHeaderField: "Authorization")
        }
    }

    private func send(_ request: URLRequest) async throws -> (Data, URLResponse) {
        do {
            return try await session.data(for: request)
        } catch let error as AudioClientError {
            throw error
        } catch {
            throw AudioClientError.transport(error.localizedDescription)
        }
    }

    private func validate(response: URLResponse, data: Data) throws {
        guard let http = response as? HTTPURLResponse else {
            throw AudioClientError.invalidResponse
        }
        guard (200...299).contains(http.statusCode) else {
            throw AudioClientError.http(
                status: http.statusCode,
                message: Self.errorMessage(from: data)
            )
        }
    }

    static func errorMessage(from data: Data) -> String? {
        guard let root = try? JSONSerialization.jsonObject(with: data) else { return nil }
        return findMessage(root)
    }

    private static func findMessage(_ value: Any) -> String? {
        if let string = value as? String, !string.isEmpty { return string }
        if let dict = value as? [String: Any] {
            if let message = dict["message"] as? String, !message.isEmpty { return message }
            for key in ["error", "detail"] {
                if let nested = dict[key], let message = findMessage(nested) { return message }
            }
        }
        return nil
    }

    private static func safeExtension(_ raw: String) -> String {
        let lower = raw.lowercased()
        let allowed = Set([
            "wav", "mp3", "m4a", "aac", "flac", "ogg", "opus", "webm", "mp4",
            "aif", "aiff", "aifc", "caf",
        ])
        return allowed.contains(lower) ? lower : "audio"
    }

    private static func mimeType(forExtension ext: String) -> String {
        switch ext {
        case "wav": return "audio/wav"
        case "mp3": return "audio/mpeg"
        case "m4a", "mp4": return "audio/mp4"
        case "aac": return "audio/aac"
        case "flac": return "audio/flac"
        case "ogg", "opus": return "audio/ogg"
        case "webm": return "audio/webm"
        default: return "application/octet-stream"
        }
    }
}

private struct AudioUpload: Sendable {
    let data: Data
    let fileExtension: String
    let mimeType: String
}

enum AudioUploadTranscoder {
    static let wavSampleRate = 16_000.0
    private static let wavChannels: AVAudioChannelCount = 1
    private static let bytesPerFrame = 2

    static func requiresWAV(_ fileExtension: String) -> Bool {
        ["m4a", "mp4", "aac", "aif", "aiff", "aifc", "caf"]
            .contains(fileExtension.lowercased())
    }

    /// Decode Apple-native compressed containers and normalize them to the
    /// format Whisper expects. AVAudioFile exposes decoded PCM through
    /// ``processingFormat``, so the converter never has to understand AAC.
    static func wavData(from sourceURL: URL, maxBytes: Int) throws -> Data {
        let source = try AVAudioFile(forReading: sourceURL)
        let sourceFormat = source.processingFormat
        guard source.length > 0,
              source.length <= AVAudioFramePosition(UInt32.max),
              let input = AVAudioPCMBuffer(
                  pcmFormat: sourceFormat,
                  frameCapacity: AVAudioFrameCount(source.length)
              ) else {
            throw AudioClientError.unreadableFile("The recording contains no decodable audio frames.")
        }
        try source.read(into: input)
        guard input.frameLength > 0 else {
            throw AudioClientError.unreadableFile("The recording contains no decodable audio frames.")
        }

        guard let outputFormat = AVAudioFormat(
            commonFormat: .pcmFormatInt16,
            sampleRate: wavSampleRate,
            channels: wavChannels,
            interleaved: false
        ), let converter = AVAudioConverter(from: sourceFormat, to: outputFormat) else {
            throw AudioClientError.unreadableFile("macOS couldn't create an audio converter for this recording.")
        }

        let ratio = wavSampleRate / sourceFormat.sampleRate
        let outputFrames = Int(ceil(Double(input.frameLength) * ratio)) + 32
        let estimatedBytes = outputFrames * bytesPerFrame + 44
        guard estimatedBytes <= maxBytes else {
            throw AudioClientError.fileTooLarge(maxBytes: maxBytes)
        }
        guard outputFrames <= Int(UInt32.max),
              let output = AVAudioPCMBuffer(
                  pcmFormat: outputFormat,
                  frameCapacity: AVAudioFrameCount(outputFrames)
              ) else {
            throw AudioClientError.unreadableFile("The decoded recording is too long.")
        }

        let inputState = AudioConverterInput(buffer: input)
        var conversionError: NSError?
        let status = converter.convert(to: output, error: &conversionError) { _, inputStatus in
            inputState.next(status: inputStatus)
        }
        guard status != .error, conversionError == nil,
              output.frameLength > 0,
              let samples = output.int16ChannelData?[0] else {
            throw AudioClientError.unreadableFile(
                conversionError?.localizedDescription ?? "macOS couldn't decode this audio file."
            )
        }

        let pcmByteCount = Int(output.frameLength) * bytesPerFrame
        guard pcmByteCount + 44 <= maxBytes else {
            throw AudioClientError.fileTooLarge(maxBytes: maxBytes)
        }
        return makeWAV(samples: samples, byteCount: pcmByteCount)
    }

    private static func makeWAV(samples: UnsafePointer<Int16>, byteCount: Int) -> Data {
        var data = Data()
        data.reserveCapacity(44 + byteCount)
        data.append(contentsOf: "RIFF".utf8)
        appendLittleEndian(UInt32(36 + byteCount), to: &data)
        data.append(contentsOf: "WAVE".utf8)
        data.append(contentsOf: "fmt ".utf8)
        appendLittleEndian(UInt32(16), to: &data)
        appendLittleEndian(UInt16(1), to: &data)
        appendLittleEndian(UInt16(wavChannels), to: &data)
        appendLittleEndian(UInt32(wavSampleRate), to: &data)
        appendLittleEndian(UInt32(Int(wavSampleRate) * bytesPerFrame), to: &data)
        appendLittleEndian(UInt16(bytesPerFrame), to: &data)
        appendLittleEndian(UInt16(16), to: &data)
        data.append(contentsOf: "data".utf8)
        appendLittleEndian(UInt32(byteCount), to: &data)
        data.append(UnsafeRawPointer(samples).assumingMemoryBound(to: UInt8.self), count: byteCount)
        return data
    }

    private static func appendLittleEndian<T: FixedWidthInteger>(_ value: T, to data: inout Data) {
        var littleEndian = value.littleEndian
        Swift.withUnsafeBytes(of: &littleEndian) { data.append(contentsOf: $0) }
    }
}

private final class AudioConverterInput: @unchecked Sendable {
    private let buffer: AVAudioPCMBuffer
    private let lock = NSLock()
    private var supplied = false

    init(buffer: AVAudioPCMBuffer) {
        self.buffer = buffer
    }

    func next(
        status: UnsafeMutablePointer<AVAudioConverterInputStatus>
    ) -> AVAudioBuffer? {
        lock.lock()
        defer { lock.unlock() }
        guard !supplied else {
            status.pointee = .endOfStream
            return nil
        }
        supplied = true
        status.pointee = .haveData
        return buffer
    }
}
