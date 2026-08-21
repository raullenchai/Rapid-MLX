import AppKit
import AVFoundation
import Foundation
import Observation

/// Drives the whole dictation loop: hotkey → capture → transcribe → inject.
///
/// Lives for the lifetime of the app rather than the Audio tab, because the
/// point of the feature is that it works while Rapid's own window is closed.
@MainActor
@Observable
final class DictationController {
    enum Phase: Equatable {
        case off
        case idle
        case starting
        case recording
        case transcribing
    }

    /// Everything that has to be true before the hotkey is armed. Surfaced
    /// individually so setup can show which single item is missing instead of a
    /// blanket "not ready".
    struct Readiness: Equatable {
        var microphone: Bool
        var accessibility: Bool
        var modelSelected: Bool

        var isReady: Bool { microphone && accessibility && modelSelected }
    }

    private(set) var phase: Phase = .off
    private(set) var lastError: String?
    private(set) var lastLatency: TimeInterval?
    /// Phase split of ``lastLatency`` ("model 1.2 s · asr 0.3 s"), present
    /// only when model bring-up took noticeable time. Answers "why was that
    /// one slow" without a log dive.
    private(set) var lastLatencyDetail: String?
    private(set) var elapsed: TimeInterval = 0
    /// Set when the TCC row says Accessibility is granted but this process
    /// still cannot install an event tap — i.e. the grant landed after launch.
    private(set) var accessibilityNeedsRelaunch = false

    let vocabulary: DictationVocabulary
    let history: DictationHistory

    /// Persisted preferences.
    var isEnabled: Bool {
        didSet {
            guard isEnabled != oldValue else { return }
            UserDefaults.standard.set(isEnabled, forKey: Keys.enabled)
            Task { isEnabled ? await enable() : disable() }
        }
    }

    var trigger: DictationHotkey.Trigger {
        didSet {
            guard trigger != oldValue else { return }
            UserDefaults.standard.set(trigger.rawValue, forKey: Keys.trigger)
            hotkey.trigger = trigger
        }
    }

    var modelAlias: String {
        didSet {
            guard modelAlias != oldValue else { return }
            UserDefaults.standard.set(modelAlias, forKey: Keys.model)
            // `modelSelected` is part of readiness, so the snapshot the UI
            // renders from has to move with it — otherwise picking a model
            // leaves the Enable switch stuck until something else refreshes.
            refreshReadiness()
            // Replacing, not joining: a prewarm still in flight here is
            // warming the PREVIOUS model, and joining it would return with
            // the new selection never warmed.
            Task { await prewarmModel(replacingCurrent: true) }
        }
    }

    var archiveAudio: Bool {
        didSet {
            guard archiveAudio != oldValue else { return }
            UserDefaults.standard.set(archiveAudio, forKey: Keys.archiveAudio)
        }
    }

    private enum Keys {
        static let enabled = "dictation.enabled"
        static let trigger = "dictation.trigger"
        static let model = "dictation.model"
        static let archiveAudio = "dictation.archiveAudio"
    }

    private let server: ServerManager
    private let client: AudioClient
    private let hotkey = DictationHotkey()
    private let recorder = DictationRecorder()
    private let hud = DictationHUD()

    private var tickTimer: Timer?
    private var recordingStart: Date?
    private var capturingApp: String?
    private var level: Float = 0
    private var transcribeTask: Task<Void, Never>?
    /// The in-flight prewarm, retained so ``disable()`` and a hotkey press
    /// can cancel it and so concurrent triggers join it (single-flight)
    /// instead of stacking probes in the engine's serial STT lane.
    private var prewarmTask: Task<Void, Never>?
    /// alias → HuggingFace repo. `ensureServing` needs the repo to fetch a
    /// model that is not on disk yet; passing nil silently limits it to models
    /// already cached.
    private var repoByAlias: [String: String] = [:]

    init(
        server: ServerManager,
        client: AudioClient = AudioClient(),
        vocabulary: DictationVocabulary? = nil,
        history: DictationHistory? = nil
    ) {
        self.server = server
        self.client = client
        self.vocabulary = vocabulary ?? DictationVocabulary()
        self.history = history ?? DictationHistory()

        let defaults = UserDefaults.standard
        self.isEnabled = defaults.bool(forKey: Keys.enabled)
        self.trigger = DictationHotkey.Trigger(
            rawValue: defaults.string(forKey: Keys.trigger) ?? ""
        ) ?? .rightCommand
        self.modelAlias = defaults.string(forKey: Keys.model) ?? ""
        // Raw microphone recordings are more sensitive than the transcript.
        // Keep them only after the user explicitly opts in from the Recent
        // section; existing explicit preferences continue to be respected.
        self.archiveAudio = defaults.object(forKey: Keys.archiveAudio) as? Bool ?? false

        hotkey.trigger = trigger
        hotkey.onTap = { [weak self] in self?.handleHotkey() }

        recorder.onLevel = { [weak self] value in
            Task { @MainActor in self?.level = value }
        }
        recorder.onFirstSample = { [weak self] in
            Task { @MainActor in self?.markRecordingStarted() }
        }
    }

    // MARK: - Readiness

    var readiness: Readiness {
        Readiness(
            microphone: DictationRecorder.microphoneAuthorization == .authorized,
            accessibility: DictationHotkey.hasAccessibilityPermission,
            modelSelected: !modelAlias.isEmpty
        )
    }

    /// Kept as a stored mirror so SwiftUI re-renders after an out-of-process
    /// permission grant; macOS gives no notification when TCC state flips.
    private(set) var readinessSnapshot = Readiness(
        microphone: false,
        accessibility: false,
        modelSelected: false
    )

    func refreshReadiness() {
        readinessSnapshot = readiness
    }

    func requestMicrophone() async {
        _ = await DictationRecorder.requestMicrophoneAccess()
        refreshReadiness()
    }

    func requestAccessibility() {
        DictationHotkey.requestAccessibilityPermission()
        // The prompt only appears once per app version; send returning users
        // straight to the pane so they are never stuck with a dead button.
        DictationHotkey.openAccessibilitySettings()
    }

    // MARK: - Lifecycle

    /// Apply the persisted switch at launch.
    ///
    /// Swift does not run `didSet` for assignments made inside `init`, so
    /// restoring `isEnabled = true` from defaults silently skipped the work
    /// that normally follows flipping the switch: the event tap was never
    /// installed, the banner still read "Ready", and the hotkey did nothing
    /// until the user toggled it off and on again.
    func bootstrap() async {
        guard isEnabled, phase == .off else { return }
        await enable()
    }

    func enable() async {
        guard isEnabled else { return }
        refreshReadiness()
        guard readinessSnapshot.microphone else {
            lastError = "Dictation needs Microphone access before it can be enabled."
            phase = .off
            isEnabled = false
            return
        }
        guard readinessSnapshot.modelSelected else {
            lastError = "Choose a transcription model before enabling dictation."
            phase = .off
            isEnabled = false
            return
        }
        guard readinessSnapshot.accessibility else {
            lastError = "Dictation needs Accessibility access before the hotkey can be used."
            // The switch records the user's intent and is left alone. Writing
            // it back to off means a permission that lapses once disables
            // dictation permanently: the grant is fixed later, but the stored
            // flag stays false and nothing re-arms. The UI shows an amber dot
            // and an Arm control for this state, and turning it off by hand is
            // never blocked, so nobody is stranded by keeping it on.
            phase = .off
            return
        }
        guard hotkey.start() else {
            // macOS does not apply an Accessibility grant to an already-running
            // process, so this is the common shape right after the user flips
            // the switch in System Settings: the TCC row says yes, this process
            // still sees no. A relaunch is the fix, not another grant.
            accessibilityNeedsRelaunch = DictationHotkey.hasAccessibilityPermission
            lastError = accessibilityNeedsRelaunch
                ? "Accessibility is granted, but this running copy hasn't picked it up. Relaunch Rapid to finish."
                : "The dictation hotkey couldn't be registered."
            phase = .off
            return
        }
        accessibilityNeedsRelaunch = false
        lastError = nil
        phase = .idle
        // Warm the whole lane now — catalog cache, sidecar, STT weights — so
        // the first hotkey press of the session starts from a hot path. Fire
        // and forget: enabling must not block on a model coming up.
        Task { [weak self] in await self?.prewarmModel() }
    }

    func disable() {
        // Stop accepting global input before tearing down anything the input
        // depends on. During app termination the server can spend several
        // seconds in its graceful-shutdown window; leaving the event tap live
        // for that window makes the hotkey appear to work even though no new
        // transcription can possibly complete.
        hotkey.stop()
        transcribeTask?.cancel()
        transcribeTask = nil
        prewarmTask?.cancel()
        prewarmTask = nil
        stopTicking()
        recorder.shutdown()
        hud.hide()
        phase = .off
    }

    /// Tear down the process-wide dictation service without changing the
    /// user's persisted Enabled preference. A normal relaunch should re-arm
    /// dictation, but no global hotkey may survive into the app's synchronous
    /// server/download shutdown window.
    func shutdownForTermination() {
        disable()
    }

    /// The system silently disables an event tap that misbehaves; re-arm when
    /// the app comes forward rather than leaving the user with a dead hotkey.
    func revalidate() {
        guard isEnabled else { return }
        refreshReadiness()
        guard readinessSnapshot.isReady else {
            lastError = "Dictation is no longer ready. Check its model and permissions."
            isEnabled = false
            return
        }
        // Returning to the app is also when a permission granted elsewhere
        // becomes usable, so a session that failed to arm gets another try
        // rather than staying dead until the switch is cycled by hand.
        guard phase != .off else {
            Task { await enable() }
            return
        }
        hotkey.reEnableIfDisabled()
    }

    // MARK: - Model

    /// Brings the transcription model up.
    ///
    /// Voice co-loading: when the app is already serving a chat LLM/VLM, the
    /// voice lane mounts in that same process (``--enable-audio``), so dictation
    /// reuses the primary server instead of swapping it away — LLM/VLM + speech
    /// run side by side. Only when no primary model is up do we serve the
    /// transcription model as its own audio process. See
    /// ``ServerManager.ensureVoiceLane``.
    @discardableResult
    private func ensureModelServing() async -> Bool {
        guard !modelAlias.isEmpty else { return false }
        let repo = await resolveRepo(for: modelAlias)
        return await server.ensureVoiceLane(alias: modelAlias, hfPath: repo)
    }

    private func resolveRepo(for alias: String) async -> String? {
        if let cached = repoByAlias[alias] { return cached }
        guard let binary = server.binaryPath else { return nil }
        let entries = await ModelCatalog.audioEntries(binary: binary)
        for entry in entries { repoByAlias[entry.alias] = entry.hfRepo }
        return repoByAlias[alias]
    }

    /// Loads the model ahead of the first hotkey press. Without this the first
    /// dictation of a session pays for loading the STT engine *and* a possible
    /// weight download while the user is already talking.
    ///
    /// The costs move off the hotkey path here, in order:
    /// 1. The alias→repo catalog lookup. On a cache miss ``resolveRepo``
    ///    spawns `rapid-mlx` CLI subprocesses — one to three SECONDS of cold
    ///    interpreter — and the old early-return below skipped it exactly when
    ///    the sidecar was already serving this model, so the most common warm
    ///    session still paid it inside the first transcription.
    /// 2. The STT engine co-load: when a primary chat model is up, dictation
    ///    reuses its server (voice co-loading) and the engine lazy-loads on the
    ///    first request; when nothing is up, a fresh audio server must start.
    /// 3. The STT weights: the engine loads them lazily on the first
    ///    transcription of each process lifetime (measured ~1.2 s for
    ///    parakeet), so ``warmUpEngine()`` sends a beat of silence to make
    ///    the server pay that now instead of inside the user's first real
    ///    dictation.
    /// - Parameter replacingCurrent: pass `true` when the model CHANGED —
    ///   an in-flight prewarm is then warming the wrong model and must be
    ///   superseded, not joined. The default joins it: for a same-model
    ///   trigger (enable + tab appear firing close together) the running
    ///   flight already covers this call.
    func prewarmModel(replacingCurrent: Bool = false) async {
        if replacingCurrent {
            prewarmTask?.cancel()
            prewarmTask = nil
        }
        // Single-flight. The check-and-assign below is MainActor-synchronous
        // (no await between them), so concurrent triggers cannot both slip
        // past: the second joins the task the first created instead of
        // racing it through the engine's serial STT lane.
        if let running = prewarmTask {
            await running.value
            return
        }
        var created: Task<Void, Never>!
        created = Task { [weak self] in
            await self?.performPrewarm()
            // Only the flight that still OWNS the slot may clear it. A
            // cancelled predecessor finishing late must not null out the
            // task a later enable started, or single-flight breaks.
            if let self, self.prewarmTask == created {
                self.prewarmTask = nil
            }
        }
        prewarmTask = created
        await created.value
    }

    private func performPrewarm() async {
        guard isEnabled, !modelAlias.isEmpty else { return }
        let alias = modelAlias
        _ = await resolveRepo(for: alias)
        // Actor reentrancy: every await above and below is a window for
        // disable() or a model change to land. Re-check before each step
        // that mutates the sidecar or touches the wire.
        guard !Task.isCancelled, isEnabled, modelAlias == alias else { return }
        if server.servingAlias != alias {
            guard await ensureModelServing() else { return }
            guard !Task.isCancelled, isEnabled, modelAlias == alias else { return }
        }
        await warmUpEngine()
    }

    /// Forces the sidecar to load the STT weights by transcribing a beat of
    /// silence. Skipped whenever a real dictation is underway — the engine
    /// serialises transcriptions, so a probe would queue in front of it.
    private func warmUpEngine() async {
        guard phase == .idle else { return }
        // Voice co-loading: with a primary model already up, the dictation STT
        // engine lazy-loads onto that same server's lane (--enable-audio), so
        // the probe warms it there. Otherwise we only warm when the sidecar is
        // itself serving this exact model.
        guard server.servingAlias == modelAlias || server.voiceCoLoadsOnPrimary else { return }
        do {
            _ = try await client.transcribe(
                audioData: Self.silentProbeWAV,
                model: modelAlias,
                context: nil,
                port: server.activePort,
                bearer: server.activeBearer
            )
        } catch is CancellationError {
            // Expected: a hotkey press or disable() superseded the probe.
        } catch {
            // Not user-facing — the cost of a failed probe is only that the
            // first real dictation pays the weight load again — but leave a
            // trace so a recurring failure is diagnosable.
            NSLog("Dictation prewarm probe failed for %@: %@", modelAlias, String(describing: error))
        }
    }

    /// 0.2 s of 16 kHz mono PCM silence — the smallest useful body for
    /// ``warmUpEngine()``. Same WAV layout ``DictationRecorder`` produces.
    static let silentProbeWAV: Data = {
        let sampleCount = 3_200
        let dataBytes = sampleCount * 2
        var wav = Data()
        wav.append(contentsOf: Array("RIFF".utf8))
        wav.append(UInt32(36 + dataBytes).littleEndianBytes)
        wav.append(contentsOf: Array("WAVEfmt ".utf8))
        wav.append(UInt32(16).littleEndianBytes)
        wav.append(UInt16(1).littleEndianBytes)
        wav.append(UInt16(1).littleEndianBytes)
        wav.append(UInt32(16_000).littleEndianBytes)
        wav.append(UInt32(16_000 * 2).littleEndianBytes)
        wav.append(UInt16(2).littleEndianBytes)
        wav.append(UInt16(16).littleEndianBytes)
        wav.append(contentsOf: Array("data".utf8))
        wav.append(UInt32(dataBytes).littleEndianBytes)
        wav.append(Data(count: dataBytes))
        return wav
    }()

    // MARK: - Hotkey

    private func handleHotkey() {
        switch phase {
        case .idle: beginRecording()
        case .starting, .recording: finishRecording()
        case .transcribing, .off: break
        }
    }

    /// Exposed so the UI (and tests) can drive a session without a real keypress.
    func toggleFromUI() { handleHotkey() }

    private func beginRecording() {
        guard !modelAlias.isEmpty else {
            lastError = "Choose a transcription model first."
            return
        }
        // A probe still in flight would sit in front of this dictation in the
        // engine's serial STT lane. Abandon it — if it already reached the
        // sidecar, the weight load it triggered continues server-side and
        // benefits the transcription that follows either way.
        prewarmTask?.cancel()
        prewarmTask = nil
        do {
            try recorder.startCapture()
        } catch {
            lastError = (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
            refreshReadiness()
            return
        }
        lastError = nil
        capturingApp = NSWorkspace.shared.frontmostApplication?.localizedName
        recordingStart = nil
        elapsed = 0
        level = 0
        phase = .starting
        hud.show(.starting)
        startTicking()
    }

    /// Fired from the audio thread the moment real samples arrive. Until this
    /// point the microphone is still opening and anything spoken is lost, so the
    /// indicator must not claim to be recording yet.
    private func markRecordingStarted() {
        guard phase == .starting else { return }
        recordingStart = Date()
        phase = .recording
        hud.update(.recording(seconds: 0, level: level))
        NSSound(named: "Tink")?.play()
    }

    private func finishRecording() {
        stopTicking()
        let audio = recorder.stopCapture()
        let duration = recordingStart.map { Date().timeIntervalSince($0) } ?? 0
        recordingStart = nil

        guard let audio else {
            hud.hide()
            phase = .idle
            lastError = "No audio was captured."
            return
        }

        phase = .transcribing
        hud.update(.transcribing)

        let app = capturingApp
        transcribeTask = Task { [weak self] in
            await self?.transcribe(audio: audio, duration: duration, appName: app)
        }
    }

    private func transcribe(audio: Data, duration: TimeInterval, appName: String?) async {
        defer {
            hud.hide()
            phase = .idle
            transcribeTask = nil
        }

        let started = Date()
        // Phase timing: "how long was that" is unanswerable from one opaque
        // number when the cost can hide in catalog resolution, a cold
        // co-load of the STT engine, or inference. The split is surfaced in
        // the Dictation tab.
        let ensureStarted = Date()
        guard await ensureModelServing() else {
            lastError = repoByAlias[modelAlias] == nil
                ? "\(modelAlias) isn't in the audio model catalog. Pick another model."
                : "\(modelAlias) couldn't start. It may still be downloading, or there may not be enough memory to load it."
            return
        }
        guard !Task.isCancelled else { return }
        let ensureSeconds = Date().timeIntervalSince(ensureStarted)

        do {
            let context = vocabulary.contextPrompt
            let requestStarted = Date()
            let result = try await client.transcribe(
                audioData: audio,
                model: modelAlias,
                context: context.isEmpty ? nil : context,
                port: server.activePort,
                bearer: server.activeBearer
            )
            let requestSeconds = Date().timeIntervalSince(requestStarted)
            guard !Task.isCancelled else { return }

            let text = Self.tidy(result.text)
            guard !text.isEmpty else {
                lastError = "Nothing was recognised in that recording."
                return
            }

            let latency = Date().timeIntervalSince(started)
            lastLatency = latency
            // Only worth spelling out when something besides inference took
            // real time — a warm run reads better as one number.
            lastLatencyDetail = ensureSeconds >= 0.1
                ? String(format: "model %.1f s · asr %.1f s", ensureSeconds, requestSeconds)
                : nil
            lastError = nil

            // Suspend the tap across injection: synthesising ⌘V puts a Command
            // flag change on the same event stream the hotkey listens to.
            hotkey.isSuspended = true
            // Say so when the text could only be copied. Silently landing it on
            // the clipboard looks identical to the feature being broken.
            let pasted = DictationInjector.canPaste
            DictationInjector.deliver(text, paste: pasted)
            if !pasted {
                lastError = "Copied to the clipboard — Accessibility access is needed to type it into the app."
            }
            Task { @MainActor [weak self] in
                try? await Task.sleep(for: .milliseconds(300))
                self?.hotkey.isSuspended = false
            }

            history.record(
                text: text,
                audio: audio,
                duration: duration,
                latency: latency,
                appName: appName,
                archiveAudio: archiveAudio
            )
        } catch {
            guard !Task.isCancelled else { return }
            lastError = (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
        }
    }

    // MARK: - Re-run (used by the Fix flow)

    /// Re-transcribes archived audio with the current vocabulary so a proposed
    /// correction can be verified before it is saved. Adding a term has been
    /// observed to regress an unrelated one, which makes "trust the fix" an
    /// unsafe default.
    func retranscribe(_ entry: DictationHistory.Entry) async -> String? {
        guard let audio = history.audioData(for: entry), !modelAlias.isEmpty else { return nil }
        guard await ensureModelServing() else { return nil }
        let context = vocabulary.contextPrompt
        guard let result = try? await client.transcribe(
            audioData: audio,
            model: modelAlias,
            context: context.isEmpty ? nil : context,
            port: server.activePort,
            bearer: server.activeBearer
        ) else { return nil }
        return Self.tidy(result.text)
    }

    // MARK: - Ticking

    private func startTicking() {
        stopTicking()
        let timer = Timer(timeInterval: 0.1, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.tick() }
        }
        RunLoop.main.add(timer, forMode: .common)
        tickTimer = timer
    }

    private func tick() {
        guard phase == .recording, let recordingStart else { return }
        elapsed = Date().timeIntervalSince(recordingStart)
        hud.update(.recording(seconds: elapsed, level: level))

        if elapsed >= DictationRecorder.maxDuration { finishRecording() }
    }

    private func stopTicking() {
        tickTimer?.invalidate()
        tickTimer = nil
    }

    // MARK: - Text

    /// Strips the trailing sentence period models add by default. A dictated
    /// fragment is usually pasted mid-sentence, where a stray period is noise.
    ///
    /// Note the two separate replacements: `。` is three UTF-8 bytes, and folding
    /// both into one character class would let a byte-wise match strip only the
    /// final byte and leave mojibake behind.
    nonisolated static func tidy(_ raw: String) -> String {
        var text = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        if text.hasSuffix("。") { text.removeLast() }
        else if text.hasSuffix(".") && !text.hasSuffix("...") { text.removeLast() }
        return text.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

private extension FixedWidthInteger {
    /// The value's little-endian bytes, for hand-assembling the WAV header of
    /// ``DictationController/silentProbeWAV``.
    var littleEndianBytes: Data {
        withUnsafeBytes(of: littleEndian) { Data($0) }
    }
}
