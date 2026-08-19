import Foundation
import AVFoundation
import Testing
@testable import Rapid

@Suite("AudioClient wire contract", .serialized)
struct AudioClientTests {
    private func makeClient() -> AudioClient {
        AudioStubProtocol.reset()
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [AudioStubProtocol.self]
        return AudioClient(session: URLSession(configuration: config))
    }

    @Test("Transcription uploads multipart audio with model and bearer")
    @MainActor
    func transcriptionRequest() async throws {
        let client = makeClient()
        AudioStubProtocol.response = (
            200,
            ["Content-Type": "application/json"],
            Data(#"{"text":"hello world","language":"en","duration":1.25}"#.utf8)
        )
        let file = temporaryFile(name: "sample.wav", data: Data("RIFFrapid-audio".utf8))
        defer { try? FileManager.default.removeItem(at: file.deletingLastPathComponent()) }

        let result = try await client.transcribe(
            fileURL: file,
            model: "whisper-tiny",
            port: 8123,
            bearer: "secret"
        )

        #expect(result == AudioTranscriptionResult(text: "hello world", language: "en", duration: 1.25))
        let request = try #require(AudioStubProtocol.requests.first)
        #expect(request.url?.absoluteString == "http://127.0.0.1:8123/v1/audio/transcriptions")
        #expect(request.httpMethod == "POST")
        #expect(request.value(forHTTPHeaderField: "Authorization") == "Bearer secret")
        #expect(request.value(forHTTPHeaderField: "Content-Type")?.hasPrefix("multipart/form-data; boundary=") == true)

        let body = String(decoding: try #require(AudioStubProtocol.bodies.first), as: UTF8.self)
        #expect(body.contains("name=\"model\"\r\n\r\nwhisper-tiny"))
        #expect(body.contains("name=\"response_format\"\r\n\r\njson"))
        #expect(body.contains("name=\"file\"; filename=\"input.wav\""))
        #expect(body.contains("Content-Type: audio/wav"))
        #expect(body.contains("RIFFrapid-audio"))
    }

    @Test("M4A transcription is normalized to a 16 kHz WAV upload")
    @MainActor
    func m4aTranscriptionRequest() async throws {
        let client = makeClient()
        AudioStubProtocol.response = (
            200,
            ["Content-Type": "application/json"],
            Data(#"{"text":"local m4a","language":"en","duration":0.1}"#.utf8)
        )
        let file = try temporaryM4A()
        defer { try? FileManager.default.removeItem(at: file.deletingLastPathComponent()) }

        _ = try await client.transcribe(
            fileURL: file,
            model: "whisper-medium",
            port: 8124,
            bearer: nil
        )

        let bodyData = try #require(AudioStubProtocol.bodies.first)
        let body = String(decoding: bodyData, as: UTF8.self)
        #expect(body.contains("name=\"file\"; filename=\"input.wav\""))
        #expect(body.contains("Content-Type: audio/wav"))
        #expect(bodyData.range(of: Data("RIFF".utf8)) != nil)
        #expect(bodyData.range(of: Data("WAVEfmt ".utf8)) != nil)
        let format = try wavFormat(in: bodyData)
        #expect(format.sampleRate == 16_000)
        #expect(format.channels == 1)
    }

    @Test("AIFF transcription is normalized to a 16 kHz WAV upload")
    @MainActor
    func aiffTranscriptionRequest() async throws {
        let client = makeClient()
        AudioStubProtocol.response = (
            200,
            ["Content-Type": "application/json"],
            Data(#"{"text":"local aiff","language":"en","duration":0.1}"#.utf8)
        )
        let file = try temporaryAIFF()
        defer { try? FileManager.default.removeItem(at: file.deletingLastPathComponent()) }

        _ = try await client.transcribe(
            fileURL: file,
            model: "whisper-medium",
            port: 8125,
            bearer: nil
        )

        let bodyData = try #require(AudioStubProtocol.bodies.first)
        let body = String(decoding: bodyData, as: UTF8.self)
        #expect(body.contains("name=\"file\"; filename=\"input.wav\""))
        #expect(body.contains("Content-Type: audio/wav"))
        #expect(bodyData.range(of: Data("RIFF".utf8)) != nil)
        #expect(bodyData.range(of: Data("WAVEfmt ".utf8)) != nil)
        let format = try wavFormat(in: bodyData)
        #expect(format.sampleRate == 16_000)
        #expect(format.channels == 1)
    }

    @Test("Voices sends model query and omits an empty bearer")
    @MainActor
    func voicesRequest() async throws {
        let client = makeClient()
        AudioStubProtocol.response = (
            200,
            ["Content-Type": "application/json"],
            Data(#"{"voices":["af_heart","bf_emma"]}"#.utf8)
        )

        let voices = try await client.voices(model: "kokoro", port: 8222, bearer: "")

        #expect(voices == ["af_heart", "bf_emma"])
        let request = try #require(AudioStubProtocol.requests.first)
        #expect(request.url?.path == "/v1/audio/voices")
        #expect(URLComponents(url: try #require(request.url), resolvingAgainstBaseURL: false)?
            .queryItems == [URLQueryItem(name: "model", value: "kokoro")])
        #expect(request.value(forHTTPHeaderField: "Authorization") == nil)
    }

    @Test("Speech sends the OpenAI JSON shape and preserves WAV bytes")
    @MainActor
    func speechRequest() async throws {
        let client = makeClient()
        let wav = Data([0x52, 0x49, 0x46, 0x46, 0x01, 0x02])
        AudioStubProtocol.response = (200, ["Content-Type": "audio/wav"], wav)

        let result = try await client.synthesize(
            text: "Hello locally",
            model: "kokoro-4bit",
            voice: "af_heart",
            speed: 1.15,
            port: 8333,
            bearer: "token"
        )

        #expect(result == SynthesizedAudio(data: wav, contentType: "audio/wav"))
        let request = try #require(AudioStubProtocol.requests.first)
        #expect(request.url?.path == "/v1/audio/speech")
        #expect(request.value(forHTTPHeaderField: "Authorization") == "Bearer token")
        #expect(request.value(forHTTPHeaderField: "Accept") == "audio/wav")
        let body = try JSONSerialization.jsonObject(
            with: try #require(AudioStubProtocol.bodies.first)
        ) as? [String: Any]
        #expect(body?["model"] as? String == "kokoro-4bit")
        #expect(body?["input"] as? String == "Hello locally")
        #expect(body?["voice"] as? String == "af_heart")
        #expect(body?["speed"] as? Double == 1.15)
        #expect(body?["response_format"] as? String == "wav")
    }

    @Test("Voice preview uses the requested voice without changing the selection")
    @MainActor
    func voicePreview() async throws {
        let client = makeClient()
        let wav = Data([0x52, 0x49, 0x46, 0x46, 0x03, 0x04])
        AudioStubProtocol.response = (200, ["Content-Type": "audio/wav"], wav)
        let server = ServerManager(testingState: .ready(alias: "qwen3-tts-4bit"))
        let viewModel = AudioViewModel(server: server, client: client)
        viewModel.audioModels = [
            ModelEntry(
                alias: "qwen3-tts-4bit",
                hfRepo: "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit",
                sizeOnDisk: "1.1 GiB",
                cached: true,
                kind: .audio,
                audioCapability: .speech,
                audioFamily: "qwen3_tts"
            )
        ]
        viewModel.selectedSpeechAlias = "qwen3-tts-4bit"
        viewModel.voices = ["Vivian", "Serena"]
        viewModel.selectedVoice = "Vivian"

        let result = await viewModel.previewVoice("Serena")

        #expect(result == SynthesizedAudio(data: wav, contentType: "audio/wav"))
        #expect(viewModel.selectedVoice == "Vivian")
        #expect(viewModel.previewingVoice == nil)
        let body = try JSONSerialization.jsonObject(
            with: try #require(AudioStubProtocol.bodies.first)
        ) as? [String: Any]
        #expect(body?["model"] as? String == "qwen3-tts-4bit")
        #expect(body?["input"] as? String == "你好，这是我的声音，很高兴认识你。")
        #expect(body?["voice"] as? String == "Serena")
    }

    @Test("Nested server detail is exposed as the user-facing failure")
    @MainActor
    func nestedServerError() async throws {
        let client = makeClient()
        AudioStubProtocol.response = (
            500,
            ["Content-Type": "application/json"],
            Data(#"{"detail":{"error":{"message":"audio runtime is unavailable"}}}"#.utf8)
        )

        do {
            _ = try await client.voices(model: "kokoro", port: 8444, bearer: nil)
            Issue.record("Expected the HTTP error")
        } catch let error as AudioClientError {
            #expect(error == .http(status: 500, message: "audio runtime is unavailable"))
            #expect(error.errorDescription == "audio runtime is unavailable")
        }
    }

    @Test("Files larger than 25 MB are rejected before transport")
    @MainActor
    func oversizedFile() async throws {
        let client = makeClient()
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("rapid-audio-test-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let file = directory.appendingPathComponent("large.wav")
        FileManager.default.createFile(atPath: file.path, contents: nil)
        let handle = try FileHandle(forWritingTo: file)
        try handle.truncate(atOffset: UInt64(AudioClient.maxUploadBytes + 1))
        try handle.close()

        do {
            _ = try await client.transcribe(
                fileURL: file,
                model: "whisper-tiny",
                port: 8555,
                bearer: nil
            )
            Issue.record("Expected the file-size error")
        } catch let error as AudioClientError {
            #expect(error == .fileTooLarge(maxBytes: AudioClient.maxUploadBytes))
        }
        #expect(AudioStubProtocol.requests.isEmpty)
    }

    private func temporaryFile(name: String, data: Data) -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("rapid-audio-test-\(UUID().uuidString)", isDirectory: true)
        try! FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let file = directory.appendingPathComponent(name)
        try! data.write(to: file)
        return file
    }

    private func temporaryM4A() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("rapid-audio-test-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let file = directory.appendingPathComponent("sample.m4a")
        let sampleRate = 44_100.0
        let format = try #require(AVAudioFormat(
            standardFormatWithSampleRate: sampleRate,
            channels: 1
        ))
        let frames: AVAudioFrameCount = 4_410
        let buffer = try #require(AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frames))
        buffer.frameLength = frames
        let samples = try #require(buffer.floatChannelData?[0])
        for index in 0..<Int(frames) {
            samples[index] = Float(sin(2.0 * .pi * 440.0 * Double(index) / sampleRate) * 0.2)
        }
        let output = try AVAudioFile(
            forWriting: file,
            settings: [
                AVFormatIDKey: kAudioFormatMPEG4AAC,
                AVSampleRateKey: sampleRate,
                AVNumberOfChannelsKey: 1,
                AVEncoderBitRateKey: 64_000,
            ]
        )
        try output.write(from: buffer)
        return file
    }

    private func temporaryAIFF() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("rapid-audio-test-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let file = directory.appendingPathComponent("sample.aiff")
        let sampleRate = 22_050.0
        let format = try #require(AVAudioFormat(
            standardFormatWithSampleRate: sampleRate,
            channels: 1
        ))
        let frames: AVAudioFrameCount = 2_205
        let buffer = try #require(AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frames))
        buffer.frameLength = frames
        let samples = try #require(buffer.floatChannelData?[0])
        for index in 0..<Int(frames) {
            samples[index] = Float(sin(2.0 * .pi * 440.0 * Double(index) / sampleRate) * 0.2)
        }
        let output = try AVAudioFile(
            forWriting: file,
            settings: [
                AVFormatIDKey: kAudioFormatLinearPCM,
                AVSampleRateKey: sampleRate,
                AVNumberOfChannelsKey: 1,
                AVLinearPCMBitDepthKey: 16,
                AVLinearPCMIsFloatKey: false,
                AVLinearPCMIsBigEndianKey: true,
            ]
        )
        try output.write(from: buffer)
        return file
    }

    private func wavFormat(in multipartBody: Data) throws -> (sampleRate: UInt32, channels: UInt16) {
        let riff = try #require(multipartBody.range(of: Data("RIFF".utf8))?.lowerBound)
        let wave = riff + 8
        #expect(multipartBody[wave..<(wave + 4)] == Data("WAVE".utf8))

        var chunk = wave + 4
        while chunk + 8 <= multipartBody.endIndex {
            let chunkID = multipartBody[chunk..<(chunk + 4)]
            let chunkSize = Int(littleEndianUInt32(in: multipartBody, at: chunk + 4))
            let payload = chunk + 8
            guard payload + chunkSize <= multipartBody.endIndex else { break }
            if chunkID == Data("fmt ".utf8) {
                #expect(chunkSize >= 16)
                return (
                    littleEndianUInt32(in: multipartBody, at: payload + 4),
                    littleEndianUInt16(in: multipartBody, at: payload + 2)
                )
            }
            chunk = payload + chunkSize + (chunkSize % 2)
        }
        Issue.record("Uploaded WAV is missing its fmt chunk")
        throw CocoaError(.fileReadCorruptFile)
    }

    private func littleEndianUInt16(in data: Data, at offset: Int) -> UInt16 {
        UInt16(data[offset]) | (UInt16(data[offset + 1]) << 8)
    }

    private func littleEndianUInt32(in data: Data, at offset: Int) -> UInt32 {
        UInt32(data[offset])
            | (UInt32(data[offset + 1]) << 8)
            | (UInt32(data[offset + 2]) << 16)
            | (UInt32(data[offset + 3]) << 24)
    }
}

private final class AudioStubProtocol: URLProtocol, @unchecked Sendable {
    nonisolated(unsafe) static var requests: [URLRequest] = []
    nonisolated(unsafe) static var bodies: [Data] = []
    nonisolated(unsafe) static var response: (Int, [String: String], Data) = (200, [:], Data())

    static func reset() {
        requests = []
        bodies = []
        response = (200, [:], Data())
    }

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        Self.requests.append(request)
        Self.bodies.append(Self.readBody(from: request))
        let (status, headers, data) = Self.response
        let response = HTTPURLResponse(
            url: request.url!,
            statusCode: status,
            httpVersion: "HTTP/1.1",
            headerFields: headers
        )!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: data)
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
