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
        self.archiveAudio = defaults.object(forKey: Keys.archiveAudio) as? Bool ?? true

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

    /// Relaunch so a post-launch Accessibility grant takes effect.
    func relaunch() {
        let url = Bundle.main.bundleURL
        let configuration = NSWorkspace.OpenConfiguration()
        configuration.createsNewApplicationInstance = true
        NSWorkspace.shared.openApplication(at: url, configuration: configuration) { _, _ in
            Task { @MainActor in NSApp.terminate(nil) }
        }
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
        guard readinessSnapshot.accessibility else {
            lastError = "Dictation needs Accessibility access before the hotkey can be used."
            phase = .off
            // Roll the switch back. Leaving it on while nothing works strands
            // the user: the control reads "enabled", the hotkey is dead, and a
            // readiness-gated `.disabled` would stop them turning it off again.
            isEnabled = false
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
            isEnabled = false
            return
        }
        accessibilityNeedsRelaunch = false
        lastError = nil
        phase = .idle
    }

    func disable() {
        transcribeTask?.cancel()
        transcribeTask = nil
        stopTicking()
        hotkey.stop()
        recorder.shutdown()
        hud.hide()
        phase = .off
    }

    /// The system silently disables an event tap that misbehaves; re-arm when
    /// the app comes forward rather than leaving the user with a dead hotkey.
    func revalidate() {
        guard isEnabled else { return }
        // Returning to the app is also when a permission granted elsewhere
        // becomes usable, so a session that failed to arm gets another try
        // rather than staying dead until the switch is cycled by hand.
        guard phase != .off else {
            Task { await enable() }
            return
        }
        hotkey.reEnableIfDisabled()
    }

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
        guard await server.ensureServing(alias: modelAlias, hfPath: nil) else {
            lastError = "The transcription model couldn't start."
            return
        }
        guard !Task.isCancelled else { return }

        do {
            let context = vocabulary.contextPrompt
            let result = try await client.transcribe(
                audioData: audio,
                model: modelAlias,
                context: context.isEmpty ? nil : context,
                port: server.activePort,
                bearer: server.activeBearer
            )
            guard !Task.isCancelled else { return }

            let text = Self.tidy(result.text)
            guard !text.isEmpty else {
                lastError = "Nothing was recognised in that recording."
                return
            }

            let latency = Date().timeIntervalSince(started)
            lastLatency = latency
            lastError = nil

            // Suspend the tap across injection: synthesising ⌘V puts a Command
            // flag change on the same event stream the hotkey listens to.
            hotkey.isSuspended = true
            DictationInjector.deliver(text, paste: DictationInjector.canPaste)
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
        guard await server.ensureServing(alias: modelAlias, hfPath: nil) else { return nil }
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
