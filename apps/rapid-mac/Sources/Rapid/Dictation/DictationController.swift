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
        case preparingModel
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
        /// The selected model's weights are actually in the cache. Without
        /// this bit the green "Ready" light could be lit while the first
        /// hotkey press still owed a multi-hundred-MB download.
        var modelOnDisk: Bool

        var isReady: Bool { microphone && accessibility && modelSelected && modelOnDisk }

        /// What ``revalidate()`` is allowed to police. Cache state arrives
        /// asynchronously (a catalog subprocess), so a synchronous check that
        /// treated "not fetched yet" as "not on disk" would disable dictation
        /// on every app activation.
        var permissionsAndSelection: Bool { microphone && accessibility && modelSelected }
    }

    private(set) var phase: Phase = .off
    private(set) var lastError: String?
    private(set) var lastLatency: TimeInterval?
    /// Phase split of ``lastLatency`` ("model 1.2 s · asr 0.3 s"), present
    /// only when model bring-up took noticeable time. Answers "why was that
    /// one slow" without a log dive.
    private(set) var lastLatencyDetail: String?
    /// Non-fatal preparation diagnostic. The sidecar is usable, but the
    /// optional lazy-weight probe failed and the first dictation may be cold.
    private(set) var lastWarmupWarning: String?
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
            if isEnabled {
                cancelActiveSessionForModelChange()
                phase = .preparingModel
            }
            Task {
                await refreshModelCacheState()
                if isEnabled {
                    await enable(replacingCurrentPrewarm: true)
                }
            }
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
    /// Constructing `DictationRecorder` initializes AVAudioEngine/CoreAudio.
    /// Keep that work behind the first real capture so state-only controllers
    /// (including launch restore and tests) never claim audio resources.
    private final class CaptureLifecycle: @unchecked Sendable {
        var recorder: DictationRecorder?
        var ticker: Timer?

        deinit {
            ticker?.invalidate()
            recorder?.shutdown()
        }
    }

    private let captureLifecycle = CaptureLifecycle()
    private var recorderStorage: DictationRecorder? {
        get { captureLifecycle.recorder }
        set { captureLifecycle.recorder = newValue }
    }
    private var recorder: DictationRecorder {
        if let recorderStorage { return recorderStorage }
        let recorder = DictationRecorder()
        recorder.onLevel = { [weak self] value in
            Task { @MainActor in self?.level = value }
        }
        recorder.onFirstSample = { [weak self] in
            Task { @MainActor in self?.markRecordingStarted() }
        }
        recorderStorage = recorder
        return recorder
    }
    private let hud = DictationHUD()
    /// `nil` means the catalog probe itself failed; an array is authoritative.
    /// Keeping those states distinct prevents a transient CLI failure from
    /// masquerading as the user deleting every cached audio model.
    private let audioCatalogLoader: @MainActor (URL) async -> [ModelEntry]?
    private let testingReadiness: Readiness?
    private let testingPrewarm: (@MainActor () async -> Bool)?
    private let testingWarmup: (@MainActor () async -> Bool)?
    private let testingHotkeyStart: (@MainActor () -> Bool)?
    private let testingHotkeyStop: (@MainActor () -> Void)?
    private let testingRecorderStart: (@MainActor () throws -> Void)?
    private let testingRecorderCancel: (@MainActor () -> Void)?
    private let testingTranscribeCancel: (@MainActor () -> Void)?

    private var tickTimer: Timer? {
        get { captureLifecycle.ticker }
        set { captureLifecycle.ticker = newValue }
    }
    /// The event tap belongs to the user's Enabled intent, not to whichever
    /// model process happens to be serving. Model transitions temporarily gate
    /// capture through ``phase`` while keeping this registration alive.
    private(set) var isHotkeyArmed = false
    private var recordingStart: Date?
    private var capturingApp: String?
    private var level: Float = 0
    private var transcribeTask: Task<Void, Never>?
    private var transcribeRequestID: UUID?
    /// The on-disk revalidation between a hotkey tap and capture. Keeping the
    /// task cancellable means turning dictation off while the catalog CLI is
    /// running cannot let its stale continuation start the microphone later.
    private var beginRecordingTask: Task<Void, Never>?
    private var beginRecordingRequestID: UUID?
    /// The in-flight prewarm, retained so ``disable()`` and a hotkey press
    /// can cancel it and so concurrent triggers join it (single-flight)
    /// instead of stacking probes in the engine's serial STT lane.
    private var prewarmTask: Task<Bool, Never>?
    private var prewarmRequestID: UUID?
    /// Invalidates stale `enable()` continuations when the model changes, the
    /// feature is disabled, or another enable attempt supersedes them.
    private var enableRequestID: UUID?
    /// Launch restore arms the global shortcut before the primary model has
    /// finished its potentially long health-check window, but must not let an
    /// audio-only fallback race that primary launch. Cleared as soon as the
    /// chat restore settles and normal voice-lane preparation begins.
    private var modelPreparationDeferred = false
    /// alias → catalog facts. ``ensureServing`` needs the repo; readiness
    /// needs ``cached``. One `audioEntries` fetch fills both. The repo is
    /// only ever passed to the server for models already on disk — passing
    /// nil limits `ensureServing` to cached models, which is exactly the
    /// contract here: downloading is the Download button's job, with
    /// progress the user can see.
    private struct CatalogFacts {
        let repo: String?
        let cached: Bool
    }
    private var catalogByAlias: [String: CatalogFacts] = [:]
    private let onProductValueDelivered: @MainActor (ProductValueKind) -> Void

    init(
        server: ServerManager,
        client: AudioClient = AudioClient(),
        vocabulary: DictationVocabulary? = nil,
        history: DictationHistory? = nil,
        testingEnabled: Bool? = nil,
        testingModelAlias: String? = nil,
        testingPhase: Phase? = nil,
        testingReadiness: Readiness? = nil,
        testingPrewarm: (@MainActor () async -> Bool)? = nil,
        testingWarmup: (@MainActor () async -> Bool)? = nil,
        testingHotkeyStart: (@MainActor () -> Bool)? = nil,
        testingHotkeyStop: (@MainActor () -> Void)? = nil,
        testingRecorderStart: (@MainActor () throws -> Void)? = nil,
        testingRecorderCancel: (@MainActor () -> Void)? = nil,
        testingTranscribeCancel: (@MainActor () -> Void)? = nil,
        testingInitialModelPreparationDeferred: Bool? = nil,
        onProductValueDelivered: @escaping @MainActor (ProductValueKind) -> Void = { _ in },
        audioCatalogLoader: @escaping @MainActor (URL) async -> [ModelEntry]? = {
            await ModelCatalog.audioEntriesIfAvailable(binary: $0)
        }
    ) {
        self.server = server
        self.client = client
        self.vocabulary = vocabulary ?? DictationVocabulary()
        self.history = history ?? DictationHistory()
        self.audioCatalogLoader = audioCatalogLoader
        self.testingReadiness = testingReadiness
        self.testingPrewarm = testingPrewarm
        self.testingWarmup = testingWarmup
        self.testingHotkeyStart = testingHotkeyStart
        self.testingHotkeyStop = testingHotkeyStop
        self.testingRecorderStart = testingRecorderStart
        self.testingRecorderCancel = testingRecorderCancel
        self.testingTranscribeCancel = testingTranscribeCancel
        self.onProductValueDelivered = onProductValueDelivered

        let defaults = UserDefaults.standard
        let restoredIsEnabled = testingEnabled ?? defaults.bool(forKey: Keys.enabled)
        self.isEnabled = restoredIsEnabled
        // Persisted enabled intent is a launch-time fact. Establish the same
        // barrier synchronously with the controller so a mounted Audio view's
        // revalidation cannot outrun ContentView's async session restore.
        self.modelPreparationDeferred = testingInitialModelPreparationDeferred
            ?? (testingEnabled == nil && restoredIsEnabled)
        self.trigger = DictationHotkey.Trigger(
            rawValue: defaults.string(forKey: Keys.trigger) ?? ""
        ) ?? .rightCommand
        self.modelAlias = testingModelAlias ?? defaults.string(forKey: Keys.model) ?? ""
        self.phase = testingPhase ?? .off
        // Raw microphone recordings are more sensitive than the transcript.
        // Keep them only after the user explicitly opts in from the Recent
        // section; existing explicit preferences continue to be respected.
        self.archiveAudio = defaults.object(forKey: Keys.archiveAudio) as? Bool ?? false

        hotkey.trigger = trigger
        hotkey.onTap = { [weak self] in self?.handleHotkey() }

    }

    // MARK: - Readiness

    var readiness: Readiness {
        if let testingReadiness { return testingReadiness }
        return Readiness(
            microphone: DictationRecorder.microphoneAuthorization == .authorized,
            accessibility: DictationHotkey.hasAccessibilityPermission,
            modelSelected: !modelAlias.isEmpty,
            modelOnDisk: catalogByAlias[modelAlias]?.cached == true
        )
    }

    /// Kept as a stored mirror so SwiftUI re-renders after an out-of-process
    /// permission grant; macOS gives no notification when TCC state flips.
    private(set) var readinessSnapshot = Readiness(
        microphone: false,
        accessibility: false,
        modelSelected: false,
        modelOnDisk: false
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
    func bootstrap(deferModelPreparation: Bool = false) async {
        guard isEnabled, phase == .off else { return }
        await enable(deferModelPreparation: deferModelPreparation)
    }

    /// Finish the audio half of launch restore after the chat launch has
    /// settled. The shortcut was already armed by ``bootstrap`` so a slow
    /// primary launch never leaves the user's persisted global shortcut
    /// silently unregistered.
    func finishDeferredBootstrap() async {
        guard isEnabled, modelPreparationDeferred else { return }
        modelPreparationDeferred = false
        // Cancellation still owns cleanup: leave the already-registered
        // shortcut able to prepare on its next use, but do not start an audio
        // model from a superseded launch task.
        guard !Task.isCancelled else { return }
        await enable()
    }

    /// Keep the enabled speech lane attached to whichever chat process owns
    /// the current session. A process-replacing model switch discards every
    /// lazy audio engine, while an in-process assistant replacement preserves
    /// it; both transitions arrive through the same ``ServerState`` boundary.
    /// Re-running the existing preparation flight is therefore idempotent for
    /// the latter and restores the former without inventing another lifecycle.
    func serverStateDidChange(_ newState: ServerState) {
        guard isEnabled, !modelPreparationDeferred else { return }
        switch newState {
        case .starting(let alias):
            // `prewarmModel` owns an audio-only fallback while its flight is
            // present. A same-alias transition with no such flight is an
            // external restart (for example ServerManager auto-respawn) and
            // must reconcile just like a chat-process replacement.
            guard alias != modelAlias || prewarmTask == nil else { break }
            cancelActiveSessionForModelChange()
            cancelModelPreparation()
            phase = .preparingModel
        case .ready(let alias):
            guard alias != modelAlias || prewarmTask == nil else { break }
            cancelActiveSessionForModelChange()
            phase = .preparingModel
            Task { [weak self] in
                await self?.enable(replacingCurrentPrewarm: true)
            }
        case .crashed, .stopped, .idle, .missing:
            cancelActiveSessionForModelChange()
            cancelModelPreparation()
            // Preserve the user's Enabled intent, but publish a terminal
            // non-ready phase. Keep the feature-owned event tap registered:
            // foreground revalidation or a later server transition can retry
            // the model without asking the user to arm dictation by hand.
            phase = .off
        }
    }

    func enable(
        replacingCurrentPrewarm: Bool = false,
        deferModelPreparation: Bool = false
    ) async {
        guard isEnabled else { return }
        // Establish the launch barrier before the first suspension. Catalog
        // refresh is re-entrant: a model change or download completion can
        // start another enable while this call awaits the subprocess.
        if deferModelPreparation {
            modelPreparationDeferred = true
        }
        let requestID = UUID()
        enableRequestID = requestID
        // The on-disk bit comes from a catalog subprocess; fetch it before
        // judging readiness so a fresh launch doesn't refuse to arm a model
        // that is sitting right there in the cache.
        await refreshModelCacheState()
        // The user can turn the switch off while the catalog subprocess is
        // running. Never let that stale enable continuation install a hotkey
        // after ``disable()`` has already torn the session down.
        guard isEnabled, enableRequestID == requestID else { return }
        refreshReadiness()
        let decision = DictationEnablePolicy.evaluate(.init(
            microphone: readinessSnapshot.microphone,
            accessibility: readinessSnapshot.accessibility,
            modelSelected: readinessSnapshot.modelSelected,
            modelOnDisk: readinessSnapshot.modelOnDisk,
            modelAlias: modelAlias
        ))
        if case .reject(let message, let disableIntent) = decision {
            lastError = message
            phase = .off
            stopHotkey()
            // Missing local model/recording prerequisites make the persisted
            // intent invalid. Accessibility is different: the user already
            // expressed intent, and granting TCC later should allow re-arm.
            if disableIntent { isEnabled = false }
            return
        }
        // Once launch restore establishes this barrier, every re-entrant
        // enable path (model selection, download completion, activation) must
        // inherit it. Only `finishDeferredBootstrap` clears it after chat has
        // settled; otherwise a model change can start an audio-only fallback
        // and tear down the still-starting primary child.
        if deferModelPreparation || modelPreparationDeferred {
            modelPreparationDeferred = true
            guard registerHotkey() else { return }
            enableRequestID = nil
            return
        }
        modelPreparationDeferred = false
        phase = .preparingModel
        let preparingAlias = modelAlias
        let prewarmSucceeded = await prewarmModel(
            replacingCurrent: replacingCurrentPrewarm
        )
        // A test prewarm seam represents the whole preparation contract. In
        // production, readiness comes only from the server's exact catalog
        // model path in the latest audio-lane residency snapshot.
        let voiceLaneReady = testingPrewarm != nil
            ? prewarmSucceeded
            : server.isVoiceLaneResident(
                for: preparingAlias,
                modelPath: catalogByAlias[preparingAlias]?.repo
            )
        guard DictationEnablePolicy.mayRegisterHotkey(after: .init(
            prewarmSucceeded: prewarmSucceeded,
            isEnabled: isEnabled,
            requestIsCurrent: enableRequestID == requestID,
            selectedAlias: modelAlias,
            preparingAlias: preparingAlias,
            isPreparing: phase == .preparingModel,
            voiceLaneReady: voiceLaneReady
        )) else {
            if isEnabled, enableRequestID == requestID, modelAlias == preparingAlias {
                lastError = "\(preparingAlias) couldn't load. There may not be enough memory to start dictation."
                phase = .off
            }
            return
        }
        guard registerHotkey() else { return }
        enableRequestID = nil
    }

    @discardableResult
    private func registerHotkey() -> Bool {
        guard !isHotkeyArmed else {
            lastError = nil
            phase = .idle
            return true
        }
        guard testingHotkeyStart?() ?? hotkey.start() else {
            // macOS does not apply an Accessibility grant to an already-running
            // process, so this is the common shape right after the user flips
            // the switch in System Settings: the TCC row says yes, this process
            // still sees no. A relaunch is the fix, not another grant.
            accessibilityNeedsRelaunch = DictationHotkey.hasAccessibilityPermission
            lastError = accessibilityNeedsRelaunch
                ? "Accessibility is granted, but this running copy hasn't picked it up. Relaunch Rapid to finish."
                : "The dictation hotkey couldn't be registered."
            phase = .off
            return false
        }
        accessibilityNeedsRelaunch = false
        isHotkeyArmed = true
        lastError = nil
        phase = .idle
        return true
    }

    private func stopHotkey() {
        guard isHotkeyArmed else { return }
        if let testingHotkeyStop {
            testingHotkeyStop()
        } else {
            hotkey.stop()
        }
        isHotkeyArmed = false
    }

    func disable() {
        // Stop accepting global input before tearing down anything the input
        // depends on. During app termination the server can spend several
        // seconds in its graceful-shutdown window; leaving the event tap live
        // for that window makes the hotkey appear to work even though no new
        // transcription can possibly complete.
        stopHotkey()
        transcribeTask?.cancel()
        transcribeTask = nil
        transcribeRequestID = nil
        beginRecordingTask?.cancel()
        beginRecordingTask = nil
        beginRecordingRequestID = nil
        prewarmTask?.cancel()
        prewarmTask = nil
        prewarmRequestID = nil
        enableRequestID = nil
        modelPreparationDeferred = false
        stopTicking()
        recorderStorage?.shutdown()
        hud.hide()
        phase = .off
    }

    /// A model change cannot safely preserve an in-flight utterance: after
    /// the swap it would be submitted to a different model. Tear the active
    /// session down before entering preparation so the microphone cannot be
    /// stranded behind a phase that ignores the stop hotkey.
    private func cancelActiveSessionForModelChange() {
        beginRecordingTask?.cancel()
        beginRecordingTask = nil
        beginRecordingRequestID = nil
        if phase == .starting || phase == .recording {
            if let testingRecorderCancel {
                testingRecorderCancel()
            } else {
                recorderStorage?.cancelCapture()
            }
        } else if phase == .transcribing {
            testingTranscribeCancel?()
            transcribeTask?.cancel()
            transcribeTask = nil
            transcribeRequestID = nil
        }
        stopTicking()
        recordingStart = nil
        capturingApp = nil
        hud.hide()
    }

    private func cancelModelPreparation() {
        prewarmTask?.cancel()
        prewarmTask = nil
        prewarmRequestID = nil
        enableRequestID = nil
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
        // Deliberately NOT the full `isReady`: `modelOnDisk` may simply not
        // have been fetched yet in this synchronous path, and a model deleted
        // mid-session is caught at the next hotkey press instead.
        guard readinessSnapshot.permissionsAndSelection else {
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
        // A foreground activation must never sneak the event tap back in
        // while `enable()` is still loading the model.
        guard phase != .preparingModel else { return }
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
    private func ensureModelServing(alias requestedAlias: String? = nil) async -> Bool {
        let alias = requestedAlias ?? modelAlias
        guard !alias.isEmpty else { return false }
        // Only models already on disk get through. This is the choke point
        // that kills every silent-download path dictation used to have: the
        // server is never handed a repo to fetch, so the worst it can do is
        // fail fast on a missing model.
        guard let facts = await catalogFacts(for: alias), facts.cached else {
            return false
        }
        return await server.ensureVoiceLane(alias: alias, hfPath: facts.repo)
    }

    private func catalogFacts(for alias: String) async -> CatalogFacts? {
        if let facts = catalogByAlias[alias] { return facts }
        await refreshModelCacheState()
        return catalogByAlias[alias]
    }

    /// Re-reads the audio catalog so ``Readiness/modelOnDisk`` reflects what
    /// is actually in the cache. Called when the pane opens, when the alias
    /// changes, and when a download finishes.
    func refreshModelCacheState() async {
        guard let binary = server.binaryPath else { return }
        guard let entries = await audioCatalogLoader(binary) else { return }
        var next: [String: CatalogFacts] = [:]
        for entry in entries {
            next[entry.alias] = CatalogFacts(repo: entry.hfRepo, cached: entry.cached)
        }
        catalogByAlias = next
        refreshReadiness()
    }

    /// Loads the model ahead of the first hotkey press. Without this the first
    /// dictation of a session pays for loading the STT engine while the user
    /// is already talking. Never a download: ``ensureModelServing`` refuses
    /// models that aren't on disk.
    ///
    /// The costs move off the hotkey path here, in order:
    /// 1. The alias→repo catalog lookup. On a cache miss ``catalogFacts``
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
    @discardableResult
    private func prewarmModel(replacingCurrent: Bool = false) async -> Bool {
        // Same-alias requests share one flight. A replacement creates a new
        // flight immediately, but chains it behind the cancelled predecessor:
        // ServerManager's stop/start sequence is not cancellation-aware, so
        // waiting for it is what prevents model A from resuming after B and
        // stealing the shared sidecar back.
        if !replacingCurrent, let running = prewarmTask {
            return await running.value
        }
        let predecessor = prewarmTask
        if replacingCurrent { predecessor?.cancel() }
        let requestID = UUID()
        prewarmRequestID = requestID
        let created = Task { [weak self] in
            if let predecessor { _ = await predecessor.value }
            guard !Task.isCancelled else { return false }
            let succeeded = await self?.performPrewarm() ?? false
            // Only the flight that still OWNS the slot may clear it. A
            // cancelled predecessor finishing late must not null out the
            // task a later enable started, or single-flight breaks.
            if let self, self.prewarmRequestID == requestID {
                self.prewarmTask = nil
                self.prewarmRequestID = nil
            }
            return succeeded
        }
        prewarmTask = created
        return await created.value
    }

    private func performPrewarm() async -> Bool {
        guard isEnabled, !modelAlias.isEmpty else { return false }
        if let testingPrewarm { return await testingPrewarm() }
        let alias = modelAlias
        let facts = await catalogFacts(for: alias)
        // Actor reentrancy: every await above and below is a window for
        // disable() or a model change to land. Re-check before each step
        // that mutates the sidecar or touches the wire.
        guard !Task.isCancelled, isEnabled, modelAlias == alias else { return false }
        if !server.isVoiceLaneReady(for: alias) {
            guard await ensureModelServing(alias: alias) else { return false }
            guard !Task.isCancelled, isEnabled, modelAlias == alias else { return false }
        }
        let warmed = await warmUpEngine()
        // The silence request materializes the lazy STT engine. Refresh the
        // authoritative snapshot before arming the global hotkey: a mounted
        // audio route alone is not proof that a process-replacing model switch
        // restored the selected speech weights.
        let voiceLaneResident = await server.refreshVoiceLaneResidency(
            for: alias,
            modelPath: facts?.repo
        )
        guard !Task.isCancelled,
              isEnabled,
              modelAlias == alias,
              voiceLaneResident else { return false }
        if warmed {
            lastWarmupWarning = nil
        } else {
            // Serving readiness is the hard boundary. The silence probe only
            // moves lazy weight cost earlier; a transient HTTP failure must
            // not make an otherwise healthy local model unusable.
            lastWarmupWarning = "Model is ready, but its warmup probe failed. The first dictation may be slower."
        }
        return true
    }

    /// Forces the sidecar to load the STT weights by transcribing a beat of
    /// silence. Skipped whenever a real dictation is underway — the engine
    /// serialises transcriptions, so a probe would queue in front of it.
    private func warmUpEngine() async -> Bool {
        guard phase == .idle || phase == .preparingModel else { return false }
        // A live conversation server owns the lazy audio lane; otherwise the
        // audio-only fallback must itself be serving this transcription model.
        guard server.isVoiceLaneReady(for: modelAlias) else {
            return false
        }
        if let testingWarmup { return await testingWarmup() }
        do {
            _ = try await client.transcribe(
                audioData: Self.silentProbeWAV,
                model: modelAlias,
                context: nil,
                port: server.activePort,
                bearer: server.activeBearer
            )
            return true
        } catch is CancellationError {
            // Expected: a hotkey press or disable() superseded the probe.
            return false
        } catch {
            // Not user-facing — the cost of a failed probe is only that the
            // first real dictation pays the weight load again — but leave a
            // trace so a recurring failure is diagnosable.
            NSLog("Dictation prewarm probe failed for %@: %@", modelAlias, String(describing: error))
            return false
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

    // MARK: - Model download

    /// Downloads themselves live in the app-wide ``DownloadManager`` — the
    /// Dictation pane drives them through the shared ``ReadinessBanner``,
    /// exactly like the sibling Audio tabs. The controller only needs to
    /// hear that a pull landed.
    ///
    /// Called by the view when the pull reaches `.completed`.
    func modelDownloadDidFinish() async {
        await refreshModelCacheState()
        if isEnabled {
            // DownloadManager retains its completed job. A recreated view can
            // replay that completion, but a hot same-alias session must keep
            // its working hotkey instead of needlessly disarming and probing.
            guard phase != .idle || server.servingAlias != modelAlias else { return }
            await enable(replacingCurrentPrewarm: true)
        }
    }

    // MARK: - Hotkey

    private func handleHotkey() {
        switch phase {
        case .idle:
            if beginRecordingTask != nil {
                beginRecordingTask?.cancel()
                beginRecordingTask = nil
                beginRecordingRequestID = nil
                hud.hide()
                return
            }
            let requestID = UUID()
            beginRecordingRequestID = requestID
            beginRecordingTask = Task { [weak self] in
                await self?.beginRecording(requestID: requestID)
                guard let self, self.beginRecordingRequestID == requestID else { return }
                self.beginRecordingTask = nil
                self.beginRecordingRequestID = nil
            }
        case .starting, .recording: finishRecording()
        case .preparingModel, .transcribing, .off: break
        }
    }

    /// Exposed so the UI (and tests) can drive a session without a real keypress.
    func toggleFromUI() { handleHotkey() }

    private func beginRecording(requestID: UUID) async {
        guard !modelAlias.isEmpty else {
            lastError = "Choose a transcription model first."
            return
        }
        let requestedAlias = modelAlias
        guard !modelPreparationDeferred else {
            lastError = "Dictation will be ready after your chat model finishes starting."
            return
        }
        guard server.isVoiceLaneReady(for: requestedAlias) else {
            phase = .preparingModel
            beginRecordingTask = nil
            beginRecordingRequestID = nil
            Task { await enable(replacingCurrentPrewarm: true) }
            return
        }
        // Model Management changes the cache out of process from this
        // controller. Re-read disk facts on every hotkey press; trusting the
        // enable-time snapshot here could pass a stale repo to ``serve`` and
        // silently redownload a model the user just deleted.
        let modelOnDisk = await modelIsOnDiskAfterRefresh(requestedAlias)
        guard !Task.isCancelled,
              beginRecordingRequestID == requestID,
              isEnabled,
              phase == .idle,
              modelAlias == requestedAlias
        else { return }
        guard modelOnDisk else {
            lastError = "\(requestedAlias) isn't on disk anymore. Open Rapid → Audio to download it."
            hud.update(.failed(message: "Model not downloaded"))
            Task {
                try? await Task.sleep(nanoseconds: 1_600_000_000)
                if phase == .idle { hud.hide() }
            }
            return
        }
        do {
            if let testingRecorderStart {
                try testingRecorderStart()
            } else {
                try recorder.startCapture()
            }
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

    /// Fresh cache truth used at the recording boundary. Internal so the
    /// deletion regression can exercise the real catalog refresh without
    /// requiring microphone or Accessibility grants in the test process.
    func modelIsOnDiskAfterRefresh(_ alias: String) async -> Bool {
        await refreshModelCacheState()
        return catalogByAlias[alias]?.cached == true
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
        let audio = recorderStorage?.stopCapture()
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
        let alias = modelAlias
        let requestID = UUID()
        transcribeRequestID = requestID
        transcribeTask = Task { [weak self] in
            await self?.transcribe(
                audio: audio,
                duration: duration,
                appName: app,
                alias: alias,
                requestID: requestID
            )
        }
    }

    private func transcribe(
        audio: Data,
        duration: TimeInterval,
        appName: String?,
        alias: String,
        requestID: UUID
    ) async {
        defer {
            if transcribeRequestID == requestID {
                hud.hide()
                phase = .idle
                transcribeTask = nil
                transcribeRequestID = nil
            }
        }

        let started = Date()
        // Phase timing: "how long was that" is unanswerable from one opaque
        // number when the cost can hide in catalog resolution, a cold
        // co-load of the STT engine, or inference. The split is surfaced in
        // the Dictation tab.
        let ensureStarted = Date()
        guard await ensureModelServing(alias: alias) else {
            guard transcribeRequestID == requestID, modelAlias == alias else { return }
            let facts = catalogByAlias[alias]
            if facts == nil {
                lastError = "\(alias) isn't in the audio model catalog. Pick another model."
            } else if facts?.cached != true {
                lastError = "\(alias) isn't downloaded. Open Rapid → Audio to download it."
            } else {
                lastError = "\(alias) couldn't start. There may not be enough memory to load it alongside the conversation model."
            }
            // The user is mid-flow in another app with only the HUD visible.
            // Leaving "Transcribing…" up while silently failing reads as a
            // hang; name the problem there before the capsule goes away.
            hud.update(.failed(message: facts?.cached != true ? "Model not downloaded" : "Couldn't start the model"))
            try? await Task.sleep(nanoseconds: 1_600_000_000)
            return
        }
        guard !Task.isCancelled,
              transcribeRequestID == requestID,
              modelAlias == alias else { return }
        let ensureSeconds = Date().timeIntervalSince(ensureStarted)

        do {
            let context = vocabulary.contextPrompt
            let requestStarted = Date()
            let result = try await client.transcribe(
                audioData: audio,
                model: alias,
                context: context.isEmpty ? nil : context,
                port: server.activePort,
                bearer: server.activeBearer
            )
            let requestSeconds = Date().timeIntervalSince(requestStarted)
            guard !Task.isCancelled,
                  transcribeRequestID == requestID,
                  modelAlias == alias else { return }

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
            onProductValueDelivered(.dictationTranscript)
        } catch {
            guard !Task.isCancelled,
                  transcribeRequestID == requestID,
                  modelAlias == alias else { return }
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
        let alias = modelAlias
        guard await ensureModelServing(alias: alias), modelAlias == alias else { return nil }
        let context = vocabulary.contextPrompt
        guard let result = try? await client.transcribe(
            audioData: audio,
            model: alias,
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

    /// Lifecycle assertions for state-only tests. These intentionally expose
    /// no way to drive capture; they only prove tests leave no run-loop or
    /// CoreAudio work behind for the next MainActor test.
    internal var testingHasActiveTicker: Bool { tickTimer?.isValid == true }
    internal var testingHasRecorder: Bool { recorderStorage != nil }

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
