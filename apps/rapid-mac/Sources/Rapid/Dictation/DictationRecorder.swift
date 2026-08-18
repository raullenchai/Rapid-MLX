import AVFoundation
import Foundation

/// Microphone capture for dictation.
///
/// The engine is kept running for a short while *after* a dictation ends rather
/// than being torn down immediately. Opening a CoreAudio input device costs
/// ~0.3 s, which is long enough to swallow the first word or two — the whole
/// reason a "get ready" indicator would otherwise be necessary. Staying warm
/// makes every dictation after the first one start instantly.
///
/// It is deliberately *not* kept warm forever: a running input node holds the
/// system microphone indicator on, and an always-listening indicator is not a
/// trade a dictation feature gets to make on the user's behalf. After
/// ``keepWarmInterval`` of inactivity the engine is torn down and the indicator
/// goes out.
final class DictationRecorder: @unchecked Sendable {
    enum RecorderError: Error, LocalizedError {
        case microphoneDenied
        case engineFailed(String)
        case noAudio

        var errorDescription: String? {
            switch self {
            case .microphoneDenied:
                return "Rapid needs microphone access to dictate. Grant it in System Settings → Privacy & Security → Microphone."
            case .engineFailed(let detail):
                return "The microphone couldn't start: \(detail)"
            case .noAudio:
                return "No audio was captured."
            }
        }
    }

    /// Whisper-family and Qwen3-ASR both consume 16 kHz mono; converting here
    /// keeps the upload small and skips a resample on the server.
    static let sampleRate: Double = 16_000

    /// Hard ceiling so a forgotten session cannot grow without bound.
    /// 10 minutes at 16 kHz mono 16-bit ≈ 19 MB, comfortably under the
    /// 25 MB upload limit in ``AudioClient``.
    static let maxDuration: TimeInterval = 10 * 60

    private let engine = AVAudioEngine()
    private let lock = NSLock()

    private var samples: [Int16] = []
    private var isCapturing = false
    private var tapInstalled = false
    private var converter: AVAudioConverter?
    private var converterSourceFormat: AVAudioFormat?
    private var teardownTask: Task<Void, Never>?

    /// How long the engine stays warm after a dictation ends.
    var keepWarmInterval: TimeInterval = 90

    /// Peak level (0…1) for the recording indicator, delivered on the audio
    /// thread — hop to the main actor before touching UI.
    var onLevel: (@Sendable (Float) -> Void)?

    /// Fires once the first converted samples land, i.e. the microphone is
    /// genuinely capturing. The indicator flips to "recording" here rather than
    /// on key-down so it never claims to be listening before it is.
    var onFirstSample: (@Sendable () -> Void)?

    var isWarm: Bool {
        lock.lock(); defer { lock.unlock() }
        return tapInstalled && engine.isRunning
    }

    // MARK: - Permission

    static var microphoneAuthorization: AVAuthorizationStatus {
        AVCaptureDevice.authorizationStatus(for: .audio)
    }

    static func requestMicrophoneAccess() async -> Bool {
        switch microphoneAuthorization {
        case .authorized: return true
        case .notDetermined: return await AVCaptureDevice.requestAccess(for: .audio)
        default: return false
        }
    }

    // MARK: - Capture

    /// Spins the engine up without capturing, so the first real dictation does
    /// not pay the device-open cost. Safe to call repeatedly.
    func warmUp() throws {
        try ensureEngineRunning()
    }

    func startCapture() throws {
        guard Self.microphoneAuthorization == .authorized else {
            throw RecorderError.microphoneDenied
        }
        teardownTask?.cancel()
        teardownTask = nil

        try ensureEngineRunning()

        lock.lock()
        samples.removeAll(keepingCapacity: true)
        isCapturing = true
        lock.unlock()
    }

    /// Stops accumulating and returns the session as a 16 kHz mono WAV.
    /// The engine stays warm; see ``keepWarmInterval``.
    func stopCapture() -> Data? {
        lock.lock()
        isCapturing = false
        let captured = samples
        samples.removeAll(keepingCapacity: false)
        lock.unlock()

        scheduleTeardown()

        guard captured.count > Int(Self.sampleRate * 0.1) else { return nil }
        return Self.wavData(from: captured, sampleRate: Int(Self.sampleRate))
    }

    /// Abandons the session without producing audio.
    func cancelCapture() {
        lock.lock()
        isCapturing = false
        samples.removeAll(keepingCapacity: false)
        lock.unlock()
        scheduleTeardown()
    }

    /// Releases the input device immediately, extinguishing the microphone
    /// indicator. Called when dictation is switched off entirely.
    func shutdown() {
        teardownTask?.cancel()
        teardownTask = nil
        tearDownEngine()
    }

    // MARK: - Engine

    private func ensureEngineRunning() throws {
        lock.lock()
        let alreadyTapped = tapInstalled
        lock.unlock()

        let input = engine.inputNode
        let hardwareFormat = input.inputFormat(forBus: 0)
        guard hardwareFormat.sampleRate > 0 else {
            throw RecorderError.engineFailed("no input device is available")
        }

        if !alreadyTapped {
            guard let target = AVAudioFormat(
                commonFormat: .pcmFormatFloat32,
                sampleRate: Self.sampleRate,
                channels: 1,
                interleaved: false
            ) else {
                throw RecorderError.engineFailed("unsupported capture format")
            }
            converter = AVAudioConverter(from: hardwareFormat, to: target)
            converterSourceFormat = hardwareFormat

            input.installTap(onBus: 0, bufferSize: 2048, format: hardwareFormat) {
                [weak self] buffer, _ in
                self?.consume(buffer)
            }
            lock.lock(); tapInstalled = true; lock.unlock()
        }

        guard !engine.isRunning else { return }
        engine.prepare()
        do {
            try engine.start()
        } catch {
            throw RecorderError.engineFailed(error.localizedDescription)
        }
    }

    private func scheduleTeardown() {
        guard keepWarmInterval > 0 else { tearDownEngine(); return }
        teardownTask?.cancel()
        let interval = keepWarmInterval
        teardownTask = Task { [weak self] in
            try? await Task.sleep(for: .seconds(interval))
            guard !Task.isCancelled, let self else { return }
            // Locking must happen in a synchronous frame: `NSLock` is
            // unavailable from async contexts because a suspension while held
            // would deadlock the next waiter.
            self.tearDownIfIdle()
        }
    }

    private func tearDownIfIdle() {
        lock.lock()
        let busy = isCapturing
        lock.unlock()
        guard !busy else { return }
        tearDownEngine()
    }

    private func tearDownEngine() {
        lock.lock()
        let hadTap = tapInstalled
        tapInstalled = false
        isCapturing = false
        lock.unlock()

        if hadTap { engine.inputNode.removeTap(onBus: 0) }
        if engine.isRunning { engine.stop() }
        converter = nil
        converterSourceFormat = nil
    }

    /// Audio-thread callback. Converts to 16 kHz mono and appends while a
    /// session is open; always reports level so the indicator can show that the
    /// right input device is picking the user up.
    private func consume(_ buffer: AVAudioPCMBuffer) {
        lock.lock()
        let capturing = isCapturing
        let converter = self.converter
        let sourceFormat = self.converterSourceFormat
        lock.unlock()

        guard let converter,
              let sourceFormat,
              sourceFormat.sampleRate == buffer.format.sampleRate else { return }

        let ratio = Self.sampleRate / buffer.format.sampleRate
        let capacity = AVAudioFrameCount(Double(buffer.frameLength) * ratio) + 64
        guard let output = AVAudioPCMBuffer(
            pcmFormat: converter.outputFormat,
            frameCapacity: capacity
        ) else { return }

        var consumed = false
        var conversionError: NSError?
        converter.convert(to: output, error: &conversionError) { _, status in
            if consumed {
                status.pointee = .noDataNow
                return nil
            }
            consumed = true
            status.pointee = .haveData
            return buffer
        }
        guard conversionError == nil,
              output.frameLength > 0,
              let channel = output.floatChannelData?[0] else { return }

        var peak: Float = 0
        var converted = [Int16]()
        converted.reserveCapacity(Int(output.frameLength))
        for index in 0..<Int(output.frameLength) {
            let sample = channel[index]
            peak = max(peak, abs(sample))
            let clamped = max(-1, min(1, sample))
            converted.append(Int16(clamped * Float(Int16.max)))
        }

        onLevel?(peak)

        guard capturing else { return }

        lock.lock()
        let wasEmpty = samples.isEmpty
        if samples.count < Int(Self.sampleRate * Self.maxDuration) {
            samples.append(contentsOf: converted)
        }
        lock.unlock()

        if wasEmpty { onFirstSample?() }
    }

    // MARK: - WAV

    /// Minimal 16-bit PCM WAV container. `AVAudioFile` would need a real file
    /// on disk; dictation keeps the audio in memory and only writes it out when
    /// history archiving is enabled.
    static func wavData(from samples: [Int16], sampleRate: Int) -> Data {
        let channels = 1
        let bitsPerSample = 16
        let byteRate = sampleRate * channels * bitsPerSample / 8
        let blockAlign = channels * bitsPerSample / 8
        let payloadBytes = samples.count * 2

        var data = Data(capacity: 44 + payloadBytes)
        func append(_ string: String) { data.append(contentsOf: string.utf8) }
        func append32(_ value: Int) { withUnsafeBytes(of: UInt32(value).littleEndian) { data.append(contentsOf: $0) } }
        func append16(_ value: Int) { withUnsafeBytes(of: UInt16(value).littleEndian) { data.append(contentsOf: $0) } }

        append("RIFF")
        append32(36 + payloadBytes)
        append("WAVE")
        append("fmt ")
        append32(16)
        append16(1)                     // PCM
        append16(channels)
        append32(sampleRate)
        append32(byteRate)
        append16(blockAlign)
        append16(bitsPerSample)
        append("data")
        append32(payloadBytes)
        samples.withUnsafeBufferPointer { buffer in
            buffer.baseAddress.map {
                data.append(UnsafeBufferPointer(start: $0, count: buffer.count))
            }
        }
        return data
    }
}
