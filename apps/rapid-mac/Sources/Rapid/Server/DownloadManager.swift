import Darwin
import Foundation
import Observation

/// Owns ``rapid-mlx pull <alias>`` subprocesses that fetch HuggingFace
/// weights in the background **independently** of the live serve child.
///
/// v0.5.6 wired a single embedded ``rapid-mlx serve <alias>`` child via
/// ``ServerManager``. That child handles its own in-band download when
/// the alias isn't yet cached — but it does so as part of the SAME
/// process that wants to actually serve a model. While a 4-30 GB cold
/// download is in flight the user can't chat: the picker is bound to
/// the serving alias, clicking a different alias tears the serve down
/// and starts the new alias's download from scratch, and the chat
/// surface has no live backend in either direction.
///
/// v0.5.7 closes that hole by splitting download from serve:
///
///   * Clicking an UN-cached row in ``ModelPickerBar`` no longer
///     blocks the active serve — it spawns a side-car
///     ``rapid-mlx pull <alias>`` and registers the job with this
///     manager. The picker shows a spinner + tqdm-derived progress
///     next to the row until the job completes.
///   * The user keeps chatting with whatever ``ServerManager`` is
///     already serving. When the side-car completes, the row icon
///     flips to ``checkmark.circle.fill`` (because ``ModelCatalog``
///     re-walks and now sees the alias cached) and a single click
///     switches the active serve in the usual flow.
///   * Cancelling mid-download is SIGTERM → SIGKILL with a 2 s grace,
///     mirroring ``ServerManager.terminateChild``. HuggingFace's
///     resumable downloader leaves a partial snapshot under
///     ``~/.cache/huggingface`` so a later retry picks up where it
///     left off — the manager does not delete the partial cache.
///
/// Why a dedicated component (instead of folding into ServerManager):
///
///   1. ``ServerManager`` is single-tenant by design — it owns one
///      ``Process`` and one ``state`` enum. Adding a map of background
///      processes there would force every read of ``state`` to also
///      decide which child is being asked about.
///   2. ``DownloadManager`` is many-tenant by design — multiple
///      concurrent pulls are explicitly allowed (HF resolves their
///      file lists independently and writes to distinct cache dirs).
///      Co-locating it with the serve lifecycle would conflate the
///      two cardinalities.
///   3. Tests don't need a live binary — the manager exposes a
///      ``_testingInject`` hook so unit suites can drive every state
///      transition (started → progress → completed / failed /
///      cancelled) without spawning a real subprocess.
@MainActor
@Observable
final class DownloadManager {
    enum DownloadSource: String, Equatable, Sendable {
        case mirror
        case huggingFace
    }

    /// Per-alias job snapshot. ``progress`` is the same parser
    /// ``ServerManager`` already uses for in-band downloads, so the
    /// picker can re-use ``progressSubtitle`` /
    /// ``progressFraction`` from the existing badge plumbing.
    ///
    /// ``@Observable`` is load-bearing — without it, the terminal
    /// ``status`` flip from ``handleExit`` (``.running → .completed``)
    /// doesn't notify SwiftUI, so any view that gates on
    /// ``job.status`` via ``.task(id:)`` keeps showing the running /
    /// in-flight copy forever after the pull subprocess exits cleanly.
    ///
    /// Concretely: ``QuickstartView`` runs
    /// ``.task(id: downloadJobStatusKey)`` whose key reads
    /// ``downloads.job(for: alias)?.status``. Before this annotation,
    /// the only thing that re-rendered the body during a pull was
    /// progress ticks on ``Job.progress`` (which IS ``@Observable``).
    /// ``handleExit`` stops the byte monitor BEFORE flipping ``status``
    /// — so no further body re-render fires after the terminal flip,
    /// the ``.task(id:)`` observer never sees ``.completed``,
    /// ``server.start(alias:)`` is never invoked, and the Quickstart
    /// card sits at ``99% · <1 min left`` forever (rapid-desktop #440).
    /// Marking ``Job`` ``@Observable`` makes the ``status`` mutation
    /// itself a tracked write so the body re-evaluates and the
    /// existing handler fires.
    @MainActor
    @Observable
    final class Job: Identifiable, Equatable {
        let id: String  // == alias
        let alias: String
        let progress: DownloadProgress
        fileprivate(set) var status: Status
        let hfPath: String?
        let totalBytes: Int64?
        let source: DownloadSource
        fileprivate(set) var failureKind: FailureDiagnosis.Kind?
        /// Live handle to the HF cache-directory byte monitor, when
        /// one is running for this job. ``nil`` when the alias's HF
        /// path was unknown (so no cache dir to watch) or when the
        /// job has already exited. Retained here so the manager can
        /// cancel the monitor on exit alongside the subprocess.
        fileprivate(set) var byteMonitor: HFCacheByteMonitor.Handle?
        fileprivate var lastProgressAt: Date
        fileprivate var lastObservedBytes: Int64?

        enum Status: Equatable {
            /// Subprocess is running. Most lines we see are tqdm ticks
            /// that ``progress`` ingests.
            case running
            /// Exited with code 0 — the HuggingFace cache now contains
            /// the alias. ``ModelCatalog`` will surface it as cached
            /// on its next refresh.
            case completed
            /// Exited with a non-zero code OR was SIGKILL'd. ``message``
            /// is a short, single-line human string drawn from the
            /// last few stderr lines (mirrors ``ModelDeletion`` shape).
            case failed(message: String)
            /// User clicked the cancel affordance. Distinct from
            /// ``.failed`` so the picker can show a quieter "cancelled"
            /// chip instead of a red error.
            case cancelled
        }

        init(
            alias: String,
            hfPath: String? = nil,
            totalBytes: Int64? = nil,
            source: DownloadSource = .mirror
        ) {
            self.id = alias
            self.alias = alias
            self.progress = DownloadProgress()
            self.status = .running
            self.hfPath = hfPath
            self.totalBytes = totalBytes
            self.source = source
            self.failureKind = nil
            self.byteMonitor = nil
            self.lastProgressAt = Date()
            self.lastObservedBytes = nil
        }

        nonisolated static func == (lhs: Job, rhs: Job) -> Bool { lhs.id == rhs.id }
    }

    // MARK: - Public state

    /// All known jobs keyed by alias. SwiftUI reads this via
    /// ``job(for:)`` so a row only re-renders when its own job's
    /// ``progress`` / ``status`` changes.
    private(set) var jobs: [String: Job] = [:]

    /// Bumped whenever the set of models on disk changes — a pull
    /// completes, or a model is deleted from any surface.
    ///
    /// Four independent snapshots of "what's on disk" exist in the app
    /// (ModelPickerBar, the two Settings panels, and
    /// UpgradeBannerCoordinator), each built from `rapid-mlx ls` and
    /// each refreshed on its own trigger. Before this counter there was
    /// no channel between them, so deleting a model in Settings left
    /// the picker's dropdown still showing it as downloaded for the
    /// rest of the session — the panel refreshed its own copy and
    /// nobody told the others.
    ///
    /// This lives on ``DownloadManager`` rather than in a new injected
    /// observable purely because every one of those surfaces already
    /// has this object in scope. Views key a `.task(id:)` on it to
    /// re-fetch.
    private(set) var cacheGeneration: UInt = 0

    /// Announce that the on-disk model set changed. Safe to call more
    /// often than strictly needed — the cost is one `rapid-mlx ls` per
    /// observing surface, and a stale "downloaded" badge is far worse
    /// than a redundant refresh.
    func markCacheChanged() {
        cacheGeneration &+= 1
    }

    // MARK: - Private bookkeeping

    private var binaryPath: URL?
    private let resolvesBinaryAtStart: Bool
    private var shutdownSignalledAt: Date?

    private var processes: [String: Process] = [:]
    private var cancellingProcesses: [String: Process] = [:]
    private var stalledProcesses: [String: Process] = [:]
    private var stallWatchdogs: [String: Task<Void, Never>] = [:]
    private let cancellationTracker = DownloadCancellationTracker()
    private var stdoutPipes: [String: Pipe] = [:]
    private var stderrPipes: [String: Pipe] = [:]
    /// Most recent stderr lines per alias. Bounded so a chatty pull
    /// can't blow memory on cancel. Used to compose ``.failed``'s
    /// human message.
    private var stderrTails: [String: [String]] = [:]
    private let stderrTailCapacity: Int = 12
    nonisolated private static let maxOutputChunkBytes = 64 * 1024
    nonisolated private static let maxOutputLineBytes = 8 * 1024
    /// Two minutes without a progress tick or byte growth is long enough to
    /// distinguish a dead network path from an ordinary slow shard.
    nonisolated static let downloadStallWindow: TimeInterval = 120

    nonisolated static func isStalled(
        lastProgressAt: Date,
        now: Date,
        window: TimeInterval = downloadStallWindow
    ) -> Bool {
        now.timeIntervalSince(lastProgressAt) >= window
    }

    // MARK: - Construction

    init(binaryPath: URL?) {
        self.binaryPath = binaryPath
        self.resolvesBinaryAtStart = true
    }

    /// Internal test seam. Lets ``RapidTests`` drive the state
    /// machine without spawning a real subprocess: callers seed a
    /// job, ingest synthetic tqdm lines via ``_testingIngest``, and
    /// finalize via ``_testingFinish``.
    internal init() {
        self.binaryPath = nil
        self.resolvesBinaryAtStart = false
    }

    // MARK: - Public API

    /// Returns the in-flight or most-recent job for ``alias`` (if any).
    /// ``nil`` means the alias has never been downloaded by this
    /// manager during the current app session.
    func job(for alias: String) -> Job? {
        jobs[alias]
    }

    /// True iff the alias has a job whose ``status == .running``.
    /// Drives picker spinner visibility.
    func isDownloading(_ alias: String) -> Bool {
        guard let job = jobs[alias] else { return false }
        if case .running = job.status { return true }
        return false
    }

    /// Suspend until the in-flight pull for ``alias`` reaches a terminal
    /// state (``completed`` / ``failed`` / ``cancelled``). Returns
    /// immediately if no running job exists for the alias.
    ///
    /// ``ServerManager.start`` calls this before spawning ``rapid-mlx
    /// serve`` so the serve subprocess doesn't race the in-flight
    /// background pull onto the HF cache. Without the stagger, both
    /// subprocesses write the same shards (pull's R2 path + serve's
    /// HF-fallback path because pull is already rewriting blobs), which
    /// doubles disk + bandwidth and leaves an orphan blob behind — see
    /// rapid-desktop issue #253.
    ///
    /// Polls every 250 ms. The user can release the wait by hitting the
    /// picker's Cancel affordance on the pull job itself — that flips
    /// the job to ``.cancelled`` and ``isDownloading`` returns ``false``
    /// on the next poll. ``Task`` cancellation also unblocks: if the
    /// enclosing start ``Task`` is cancelled (e.g. the view tears down
    /// during the wait), ``Task.sleep`` throws ``CancellationError`` and
    /// we exit by returning — falling through to a ``try?`` would turn
    /// the loop into a tight MainActor busy-poll the instant
    /// cancellation lands, freezing the UI until the pull settled.
    func awaitDownloadSettlement(alias: String) async {
        let trimmed = alias.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        while isDownloading(trimmed) {
            do {
                try await Task.sleep(nanoseconds: 250_000_000)
            } catch {
                // Task cancellation — bail instead of busy-looping on a
                // sleep that throws immediately every iteration.
                return
            }
        }
    }

    /// Spawn ``rapid-mlx pull <alias>`` in the background. No-op if a
    /// running job already exists for this alias (clicking the
    /// download icon a second time mid-pull should be idle, not
    /// double-fire). Returns ``true`` if a new process was spawned.
    ///
    /// ``hfPath`` (optional) and ``totalBytes`` (optional) are routed
    /// through to the HF cache-directory byte monitor so the picker
    /// can render real bytes-on-disk progress alongside HF's
    /// file-count tqdm. Callers that hold a ``ModelEntry`` should pass
    /// ``entry.hfRepo`` here; ``totalBytes`` defaults to the
    /// ``ModelSizing`` weight estimate when ``nil``. Both args fall
    /// back to "unknown" without affecting the spawn — the existing
    /// tqdm parser keeps driving the UI just like before.
    @discardableResult
    func startDownload(
        alias: String,
        hfPath: String? = nil,
        totalBytes: Int64? = nil,
        source: DownloadSource = .mirror
    ) -> Bool {
        let trimmed = alias.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return false }
        guard ModelCatalog.isSafeAlias(trimmed) else {
            if trimmed.utf8.count <= ModelCatalog.maxAliasBytes {
                let job = Job(alias: trimmed, hfPath: hfPath, totalBytes: totalBytes, source: source)
                job.failureKind = .downloadFailed
                job.status = .failed(message: "That model name isn't valid.")
                jobs[trimmed] = job
            }
            return false
        }
        if isDownloading(trimmed) { return false }
        let binaryResolution = Self.resolveBinaryForStart(
            cached: binaryPath,
            shouldRelocate: resolvesBinaryAtStart
        )
        guard case .resolved(let binary, let changed) = binaryResolution else {
            // Record a synthetic failed job so the picker UI can
            // surface "binary not found" inline instead of silently
            // doing nothing.
            let job = Job(alias: trimmed, hfPath: hfPath, totalBytes: totalBytes, source: source)
            job.failureKind = .downloadFailed
            job.status = .failed(message: binaryResolution.failureMessage)
            jobs[trimmed] = job
            return false
        }
        if changed {
            print("DownloadManager: rapid-mlx path changed from \(binaryPath?.path ?? "(nil)") to \(binary.path); using refreshed path.")
            binaryPath = binary
        }

        let job = Job(alias: trimmed, hfPath: hfPath, totalBytes: totalBytes, source: source)
        jobs[trimmed] = job
        stderrTails[trimmed] = []

        let process = Process()
        process.executableURL = binary
        process.arguments = ["pull", trimmed]
        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe
        process.standardInput = FileHandle.nullDevice
        process.environment = augmentedEnv(for: binary, source: source)

        // tqdm refreshes via \r when stderr is not a TTY (same shape
        // ServerManager parses). Split on both separators.
        // Crash-safe + non-blocking drain. `availableData` raises an
        // uncatchable NSException on a bad descriptor (SIGABRTs the
        // process) when the child's pipe FD races teardown, and
        // `read(upToCount:)` blocks until 64 KiB fills — which would
        // freeze the tqdm download-progress tail until that much accrues.
        // A per-pipe ``PipeDrainer`` OWNS its read handle (keeping the FD
        // valid for any late handler firing) and drains non-blocking; each
        // is constructed here while the handle is live. An empty result
        // (bad FD, EOF, or nothing pending) is treated as "no lines".
        let stderrDrainer = PipeDrainer(stderrPipe.fileHandleForReading)
        let stdoutDrainer = PipeDrainer(stdoutPipe.fileHandleForReading)
        let onProgressChunk: @Sendable (FileHandle) -> Void = { [weak self] _ in
            let data = stderrDrainer.drain().data
            guard !data.isEmpty else { return }
            let lines = Self.lines(from: data)
            guard !lines.isEmpty else { return }
            Task { @MainActor [weak self] in
                self?.ingestLines(alias: trimmed, lines: lines, isStderr: true)
            }
        }
        let onStdoutChunk: @Sendable (FileHandle) -> Void = { [weak self] _ in
            let data = stdoutDrainer.drain().data
            guard !data.isEmpty else { return }
            let lines = Self.lines(from: data)
            guard !lines.isEmpty else { return }
            Task { @MainActor [weak self] in
                self?.ingestLines(alias: trimmed, lines: lines, isStderr: false)
            }
        }
        stderrPipe.fileHandleForReading.readabilityHandler = onProgressChunk
        stdoutPipe.fileHandleForReading.readabilityHandler = onStdoutChunk

        let cancellationTracker = self.cancellationTracker
        process.terminationHandler = { [weak self, cancellationTracker] proc in
            let wasCancelling = cancellationTracker.isCancelling(proc)
            let status = proc.terminationStatus
            let reason = proc.terminationReason
            Task { @MainActor [weak self] in
                self?.handleExit(
                    alias: trimmed,
                    process: proc,
                    status: status,
                    reason: reason,
                    wasCancelling: wasCancelling
                )
            }
        }

        do {
            try process.run()
        } catch {
            stdoutPipe.fileHandleForReading.readabilityHandler = nil
            stderrPipe.fileHandleForReading.readabilityHandler = nil
            print("[download] couldn't start \(trimmed): \(error.localizedDescription)")
            job.failureKind = .downloadFailed
            job.status = .failed(
                message: FailureDiagnoser.diagnosis(for: .downloadFailed).message
            )
            return false
        }
        // Register pipes + process AFTER spawn succeeds so a throw
        // above leaves the manager's bookkeeping untouched.
        processes[trimmed] = process
        stdoutPipes[trimmed] = stdoutPipe
        stderrPipes[trimmed] = stderrPipe

        // Wire byte-monitor: the HF "Fetching N files" tqdm bar counts
        // FILES, not BYTES. On a 6.8 GB / 11-shard cold download the
        // outer bar reads "0/9 files (0%)" for many minutes while the
        // first shard streams silently. Sample the on-disk cache dir
        // so the UI can render real bytes independent of tqdm cadence.
        // Failure to resolve / start the monitor is silently ignored —
        // the existing tqdm-derived copy stays in charge.
        installByteMonitor(
            job: job,
            hfPath: hfPath,
            totalBytes: totalBytes
        )
        installStallWatchdog(alias: trimmed, process: process, job: job)
        return true
    }

    private func installStallWatchdog(alias: String, process: Process, job: Job) {
        stallWatchdogs[alias]?.cancel()
        stallWatchdogs[alias] = Task { @MainActor [weak self, weak job] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(5))
                guard let self, let job, process.isRunning else { return }
                // The network transfer is over. Model loading / shader
                // compilation can legitimately be quiet for minutes and is
                // covered by ServerManager's separate startup watchdog.
                if case .warmingUp = job.progress.phase { return }
                if let bytes = job.progress.bytesDownloaded,
                   bytes != job.lastObservedBytes {
                    job.lastObservedBytes = bytes
                    job.lastProgressAt = Date()
                }
                guard Self.isStalled(
                    lastProgressAt: job.lastProgressAt,
                    now: Date()
                ) else {
                    continue
                }
                self.stalledProcesses[alias] = process
                job.failureKind = .downloadFailed
                job.status = .failed(
                    message: "The download stopped making progress. Check your network and try again — saved partial data will be reused."
                )
                process.terminate()
                let deadline = Date().addingTimeInterval(2)
                while process.isRunning && Date() < deadline {
                    try? await Task.sleep(for: .milliseconds(100))
                }
                if process.isRunning {
                    kill(process.processIdentifier, SIGKILL)
                }
                return
            }
        }
    }

    /// Retry a terminal job while preserving the cache-monitor metadata.
    /// Passing a source overrides only the new child process.
    @discardableResult
    func retryDownload(
        alias: String,
        source: DownloadSource? = nil
    ) -> Bool {
        let trimmed = alias.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let previous = jobs[trimmed] else { return false }
        guard previous.status != .running else { return false }
        // A watchdog failure is surfaced before SIGTERM/SIGKILL reaps the old
        // child. Do not overlap two writers against the same cache sidecars.
        guard processes[trimmed] == nil else { return false }
        let hfPath = previous.hfPath
        let totalBytes = previous.totalBytes
        let nextSource = source ?? previous.source
        dismissJob(alias: trimmed)
        return startDownload(
            alias: trimmed,
            hfPath: hfPath,
            totalBytes: totalBytes,
            source: nextSource
        )
    }

    /// Resolve the per-alias HF cache directory and kick off a
    /// ``HFCacheByteMonitor`` polling task. Wires
    /// ``progress.setTotalBytes`` from the catalog estimate (or the
    /// explicit override) up front so the UI can render
    /// "X / Y GB · Z%" the moment the first observation lands.
    ///
    /// All failure modes (missing HF env, sanitisation reject, dir
    /// not yet created) leave the byte channel at ``nil`` — the UI
    /// falls through to existing tqdm-derived copy without breaking.
    private func installByteMonitor(
        job: Job,
        hfPath: String?,
        totalBytes: Int64?
    ) {
        let resolvedTotal = totalBytes ?? Self.estimateTotalBytes(for: job.alias)
        job.progress.setTotalBytes(resolvedTotal)
        guard let hfPath, !hfPath.isEmpty else { return }
        guard let hubCacheRoot = BundledModel.userHFCacheURL(
            environment: ProcessInfo.processInfo.environment,
            // Issue #503: watch the custom models folder (when set) so
            // the bytes-on-disk overlay tracks the real download dir.
            preferredOverride: ModelsFolderPreference.validatedOverrideURL()
        ) else {
            return
        }
        guard let cacheDir = HFCacheByteMonitor.cacheDirectoryURL(
            hubCacheRoot: hubCacheRoot,
            hfPath: hfPath
        ) else {
            return
        }
        let progress = job.progress
        let handle = HFCacheByteMonitor.start(
            cacheDir: cacheDir,
            progress: progress
        )
        job.byteMonitor = handle
    }

    /// Catalog-free estimate of the alias's weight footprint in bytes.
    /// Used when the caller didn't supply ``totalBytes`` so the byte
    /// monitor still has a denominator to render "X / Y GB · Z%".
    /// Returns ``nil`` for unfamiliar aliases (params unknown →
    /// ``ModelSizing.weightsGB == 0``); the UI then renders
    /// "X MB downloaded" without a fraction.
    nonisolated static func estimateTotalBytes(for alias: String) -> Int64? {
        let footprint = ModelSizing.estimate(alias: alias)
        guard footprint.weightsGB > 0 else { return nil }
        let bytes = footprint.weightsGB * 1024.0 * 1024.0 * 1024.0
        guard bytes.isFinite, bytes > 0 else { return nil }
        return Int64(bytes)
    }

    /// SIGTERM → 2 s grace → SIGKILL the in-flight pull for ``alias``.
    /// Idempotent; safe to call when no job exists. After cancel the
    /// job's status flips to ``.cancelled`` via ``handleExit``.
    func cancelDownload(alias: String) {
        let trimmed = alias.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let process = processes[trimmed] else { return }
        guard process.isRunning else { return }
        // #19 cancel-race: flag the exact Process object BEFORE
        // signalling. The termination handler captures this identity
        // at exit time, so a late cancel click cannot rewrite a normal
        // completion from a process whose handler had already
        // observed status 0.
        cancellingProcesses[trimmed] = process
        cancellationTracker.markCancelling(process)
        // Audit batch 9: also flip the job optimistically so the UI
        // doesn't show a stale ``.running`` row between SIGTERM and
        // the terminationHandler firing. ``handleExit`` re-asserts
        // ``.cancelled`` when ``wasCancelling == true`` so this is
        // idempotent with the wasCancelling branch.
        if let job = jobs[trimmed], case .running = job.status {
            markCancelled(job)
        }
        process.terminate()
        // Async hard-kill check; matches ServerManager.terminateChild.
        Task { @MainActor [weak self] in
            let deadline = Date().addingTimeInterval(2.0)
            while Date() < deadline && process.isRunning {
                try? await Task.sleep(nanoseconds: 100_000_000)
            }
            if process.isRunning {
                kill(process.processIdentifier, SIGKILL)
            }
            // Cleanup runs in handleExit once terminationHandler fires.
            _ = self
        }
    }

    /// Forget a finished (completed / failed / cancelled) job so the
    /// picker can re-trigger a fresh download. Idempotent. Calling
    /// this on a ``.running`` job is a no-op — the caller should
    /// ``cancelDownload`` first.
    func dismissJob(alias: String) {
        let trimmed = alias.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let job = jobs[trimmed] else { return }
        if case .running = job.status { return }
        jobs.removeValue(forKey: trimmed)
        stderrTails.removeValue(forKey: trimmed)
    }

    /// Synchronous teardown for app shutdown. Mirrors
    /// ``ServerManager.shutdownSync``. We can't await Process.exit
    /// from ``applicationWillTerminate``, so SIGTERM + short blocking
    /// poll + SIGKILL.
    ///
    /// Split into ``beginShutdown`` (signal, non-blocking) and
    /// ``finishShutdown`` (reap, blocking) so the caller can overlap
    /// this grace window with ``ServerManager``'s. Previously both
    /// teardowns ran start-to-finish back-to-back on the main thread,
    /// so the app quit path serialised the server's 5.5 s budget and
    /// this 2 s one into 7.5 s of blocking — see
    /// ``AppDelegate.runTerminationSequence``. Signalling here first
    /// lets these children die WHILE the server grace is running, and
    /// by the time ``finishShutdown`` polls they are almost always
    /// already gone.
    func beginShutdown() {
        if shutdownSignalledAt == nil { shutdownSignalledAt = Date() }
        for (_, process) in processes where process.isRunning {
            process.terminate()
        }
    }

    /// When the children were SIGTERM'd, so ``finishShutdown`` can measure
    /// its grace from the SIGNAL rather than from the reap.
    private static let downloadShutdownGrace: TimeInterval = 2.0

    /// Second half of the split teardown: wait briefly for the
    /// already-SIGTERM'd children, SIGKILL any survivor, then run the
    /// normal per-alias bookkeeping so nothing is left half-torn-down.
    func finishShutdown() {
        let aliases = Array(processes.keys)
        // Measured from when the children were SIGNALLED, not from when we
        // got round to reaping them. The server's grace runs first, so a
        // fresh 2 s here would still SUM with it whenever a child ignores
        // SIGTERM — which is exactly the serialisation this split exists to
        // remove. Children that already had their whole window are reaped
        // immediately; only genuinely-new time is ever spent.
        let deadline = (shutdownSignalledAt ?? Date())
            .addingTimeInterval(Self.downloadShutdownGrace)
        while Date() < deadline && processes.values.contains(where: { $0.isRunning }) {
            Thread.sleep(forTimeInterval: 0.1)
        }
        for (_, process) in processes where process.isRunning {
            kill(process.processIdentifier, SIGKILL)
        }
        for alias in aliases {
            if let job = jobs[alias], case .running = job.status {
                markCancelled(job)
            }
            if let process = processes[alias] {
                cleanupProcessBookkeeping(alias: alias, process: process)
            }
        }
    }

    // MARK: - Internals

    /// Ingest a batch of lines from a child's stdout or stderr. tqdm
    /// progress ticks (matched by ``DownloadProgress``) update the
    /// job's progress; non-tqdm stderr is retained in a bounded tail
    /// for the eventual failure message.
    private func ingestLines(alias: String, lines: [String], isStderr: Bool) {
        guard let job = jobs[alias] else { return }
        for line in lines {
            let consumed = job.progress.ingest(line)
            if consumed { job.lastProgressAt = Date() }
            if isStderr && !consumed {
                var tail = stderrTails[alias] ?? []
                // Scrub before storage — the stderr tail feeds the
                // failure message surfaced in the UI; a leaked HF
                // token in there ends up in a copy-pasted support
                // ticket. Audit P1 (ServerManager stderr → log).
                let scrubbed = LogScrubber.scrub(line.trimmingCharacters(in: .whitespacesAndNewlines))
                tail.append(scrubbed)
                if tail.count > stderrTailCapacity {
                    tail.removeFirst(tail.count - stderrTailCapacity)
                }
                stderrTails[alias] = tail
            }
        }
    }

    private func handleExit(
        alias: String,
        process: Process,
        status: Int32,
        reason: Process.TerminationReason,
        wasCancelling: Bool
    ) {
        guard let job = jobs[alias] else { return }
        if let current = processes[alias], current !== process {
            return
        }
        // Drain readability handlers so ARC can free the pipes.
        cleanupProcessBookkeeping(alias: alias, process: process)

        if stalledProcesses.removeValue(forKey: alias) === process {
            // The watchdog already installed the actionable retry state.
            return
        }

        if wasCancelling {
            markCancelled(job)
            return
        }

        switch reason {
        case .exit where status == 0:
            job.failureKind = nil
            job.status = .completed
            // Weights just landed: every catalog snapshot in the app is
            // now stale.
            markCacheChanged()
        case .exit:
            recordFailure(job: job, alias: alias, signal: false)
        case .uncaughtSignal:
            recordFailure(job: job, alias: alias, signal: true)
        @unknown default:
            job.failureKind = .downloadFailed
            job.status = .failed(
                message: FailureDiagnoser.diagnosis(for: .downloadFailed).message
            )
        }
    }

    /// Flip a job to ``Status/cancelled`` and say so in ``Job/failureKind``.
    ///
    /// The kind is the load-bearing half. Every surface that explains a
    /// terminal job reads ``failureKind`` first and only falls back to
    /// classifying a raw string — and the string classifier has no signal for
    /// "the user stopped it", so a cancelled job carrying a `nil` kind was
    /// read as ``FailureDiagnosis/Kind/downloadFailed`` and told the user to
    /// check a connection that was never at fault (Paper 05.1 state 10,
    /// flagged). Recording the kind at every cancellation site is what stops
    /// that inference happening at all.
    private func markCancelled(_ job: Job) {
        job.failureKind = .downloadCancelled
        job.status = .cancelled
    }

    private func recordFailure(job: Job, alias: String, signal: Bool) {
        // The child's stderr tail can carry engine internals (the
        // "rapid-mlx" name, HTTP codes, Python stack frames). Log it
        // for support, but NEVER surface it to the user — they get a
        // plain, actionable message instead.
        let tail = (stderrTails[alias] ?? [])
            .filter { !$0.isEmpty }
            .suffix(3)
            .joined(separator: " — ")
        if !tail.isEmpty {
            print("[download] \(alias) failed: \(tail)")
        }
        let kind = signal
            ? FailureDiagnosis.Kind.downloadFailed
            : FailureDiagnoser.downloadFailureKind(
                raw: tail,
                usingMirror: job.source == .mirror
            )
        job.failureKind = kind
        if signal {
            job.status = .failed(message: "The model download was interrupted. Try again.")
        } else {
            job.status = .failed(message: FailureDiagnoser.diagnosis(for: kind).message)
        }
    }

    private func augmentedEnv(for binary: URL, source: DownloadSource) -> [String: String] {
        var env = ProcessInfo.processInfo.environment
        let existingPath = env["PATH"] ?? ""
        let augmented = [
            binary.deletingLastPathComponent().path,
            "/opt/homebrew/bin",
            "/usr/local/bin",
            "/usr/bin",
            "/bin",
            existingPath,
        ].filter { !$0.isEmpty }.joined(separator: ":")
        env["PATH"] = augmented
        DownloadManager.applyXetConcurrencyCaps(env: &env)
        DownloadManager.applyDownloadSource(source, env: &env)
        DownloadManager.applyModelsFolderOverride(env: &env)
        return env
    }

    /// Issue #503: point a background ``rapid-mlx pull`` at the user's
    /// chosen models folder so the download lands in the SAME directory
    /// the engine loads from and the app scans — otherwise the Download
    /// button in Settings → Model Management would silently fill the
    /// internal disk while the engine reads the external folder.
    ///
    /// Direct write (the desktop preference is authoritative) but only
    /// when the folder is set AND currently a reachable directory. An
    /// unplugged drive resolves to ``nil`` and we leave the ambient env
    /// untouched so the pull falls back to the default location instead
    /// of failing — matching the serve-side and app-side fallback.
    nonisolated static func applyModelsFolderOverride(env: inout [String: String]) {
        if let override = ModelsFolderPreference.validatedOverrideURL() {
            env["HF_HUB_CACHE"] = override.path
        }
    }

    /// Bound HuggingFace xet client concurrency so a single `rapid-mlx
    /// pull` can't peg the user's home network.
    ///
    /// Stock `huggingface_hub.snapshot_download` fans out to 8 parallel
    /// files, and each xet-backed file ramps its own range-stream
    /// concurrency from 1 → 64 — worst case ~512 simultaneous TCP
    /// ranges with zero global cap. Field reports
    /// ("computer's network freezes during downloads") match the
    /// expected symptom of bufferbloat + flow-table exhaustion on
    /// consumer routers.
    ///
    /// 2 files × 8 streams = 16 ranges max, which keeps the pipe
    /// near-full on a single 1 Gbps link (xet chunking already
    /// saturates one stream) while staying below the per-host limits
    /// of typical home gateways.
    ///
    /// v0.6.11: also pin
    /// ``HF_XET_FIXED_DOWNLOAD_CONCURRENCY=8`` so the per-file stream
    /// count starts at 8 instead of ramping 1 → 8 over the first
    /// 30–60 seconds of the download. Our target user runs a laptop
    /// on home WiFi — Ollama's "16 fixed parts from t=0" pattern is
    /// what reads as a smooth first-run download, not xet's adaptive
    /// ramp ("why is it slow for 30 seconds then suddenly fast?").
    /// The Hugging Face docs flag this override as a power-user knob
    /// that bypasses the adaptive controller entirely; the symptom
    /// it was designed to fix (multi-gigabit backbones with spare
    /// headroom) doesn't apply to our user base.
    ///
    /// Power users can override any var from the parent shell —
    /// ``ProcessInfo`` carries the inherited environment, so an
    /// explicit `export HF_XET_… = …` wins.
    ///
    /// Numbers tuned to defaults from the 2026-06-16 research report
    /// (`docs/research/download-ux-2026-06-16.md`). Mirror this in
    /// `vllm_mlx/cli.py` so terminal users get the same caps.
    nonisolated static func applyXetConcurrencyCaps(env: inout [String: String]) {
        // FIXED bypasses the adaptive controller entirely (it
        // aliases initial/min/max). A user who explicitly set any
        // adaptive-controller knob clearly WANTS the adaptive
        // controller — injecting FIXED would silently defeat their
        // `MAX=4` (or `MIN=2`, etc.) by pinning the stream count to
        // our 8 instead. Snapshot the pre-state of those knobs
        // BEFORE we apply our own defaults so the MAX=8 default we
        // set below doesn't look like a user override later.
        let userTouchedAdaptive =
            env["HF_XET_CLIENT_AC_MAX_DOWNLOAD_CONCURRENCY"] != nil
            || env["HF_XET_CLIENT_AC_MIN_DOWNLOAD_CONCURRENCY"] != nil
            || env["HF_XET_CLIENT_AC_INITIAL_DOWNLOAD_CONCURRENCY"] != nil

        if env["HF_XET_DATA_MAX_CONCURRENT_FILE_DOWNLOADS"] == nil {
            env["HF_XET_DATA_MAX_CONCURRENT_FILE_DOWNLOADS"] = "2"
        }
        if env["HF_XET_CLIENT_AC_MAX_DOWNLOAD_CONCURRENCY"] == nil {
            env["HF_XET_CLIENT_AC_MAX_DOWNLOAD_CONCURRENCY"] = "8"
        }
        if env["HF_XET_FIXED_DOWNLOAD_CONCURRENCY"] == nil && !userTouchedAdaptive {
            env["HF_XET_FIXED_DOWNLOAD_CONCURRENCY"] = "8"
        }
    }

    /// Point ``rapid-mlx pull`` at the rapid-mlx R2-hosted weight
    /// mirror so cold downloads come off our CDN by default instead
    /// of HuggingFace Hub.
    ///
    /// rapid-mlx (the CLI) honours ``RAPID_MLX_MODEL_MIRROR`` as a
    /// prefetch base URL: when set, it tries
    /// ``$RAPID_MLX_MODEL_MIRROR/<hf_owner>/<hf_repo>/<filename>``
    /// for each weight blob first and falls through to the HuggingFace
    /// Hub on a miss. ``models.rapidmlx.com`` is the project's public
    /// R2-backed mirror (rate-limited Cloudflare Worker → R2 bucket),
    /// so a default-on value gives every first-run user a faster,
    /// less rate-limited cold download with zero config.
    ///
    /// Override semantics match ``applyXetConcurrencyCaps`` — a power
    /// user can `export RAPID_MLX_MODEL_MIRROR=…` (including the
    /// empty string to disable the mirror entirely) from their parent
    /// shell, and ``augmentedEnv`` carries that value via
    /// ``ProcessInfo.processInfo.environment`` before we ever touch
    /// the dict. We only fill the variable when it's truly absent.
    nonisolated static func applyModelMirror(env: inout [String: String]) {
        // Use ``setdefault`` semantics — only fill when the key is
        // entirely absent so a user who set it (even to "") in their
        // parent shell keeps full control. An empty string disables
        // the mirror in rapid-mlx itself; we preserve that intent.
        if env["RAPID_MLX_MODEL_MIRROR"] == nil {
            env["RAPID_MLX_MODEL_MIRROR"] = "https://models.rapidmlx.com"
        }
    }

    /// Apply a source choice to one child environment. The direct-Hugging
    /// Face path intentionally overrides even a parent-shell mirror for this
    /// retry, but never mutates the parent process or stored preferences.
    nonisolated static func applyDownloadSource(
        _ source: DownloadSource,
        env: inout [String: String]
    ) {
        switch source {
        case .mirror:
            applyModelMirror(env: &env)
        case .huggingFace:
            env["RAPID_MLX_MODEL_MIRROR"] = ""
        }
    }

    private func cleanupProcessBookkeeping(alias: String, process: Process) {
        stallWatchdogs.removeValue(forKey: alias)?.cancel()
        stdoutPipes[alias]?.fileHandleForReading.readabilityHandler = nil
        stderrPipes[alias]?.fileHandleForReading.readabilityHandler = nil
        stdoutPipes.removeValue(forKey: alias)
        stderrPipes.removeValue(forKey: alias)
        processes.removeValue(forKey: alias)
        // Stop the cache-dir byte poller so it doesn't keep walking
        // ``models--<owner>--<repo>/`` forever after the pull exits.
        // ``Handle.stop`` is idempotent — safe even when the monitor
        // never started (alias had no resolvable hf_path).
        if let job = jobs[alias] {
            job.byteMonitor?.stop()
            job.byteMonitor = nil
        }
        // #19 cancel-race: clear per-process cancellation flags so a
        // stale Process pointer can never poison the next start of
        // the same alias.
        if cancellingProcesses[alias] === process {
            cancellingProcesses.removeValue(forKey: alias)
        }
        cancellationTracker.clear(process)
    }

    nonisolated private static func lines(from data: Data) -> [String] {
        let bounded = data.prefix(maxOutputChunkBytes)
        guard let text = String(data: bounded, encoding: .utf8) else { return [] }
        return text.split(whereSeparator: { $0 == "\r" || $0 == "\n" })
            .compactMap { raw -> String? in
                guard !raw.isEmpty, raw.utf8.count <= maxOutputLineBytes else { return nil }
                return String(raw)
            }
    }

    // #19 binary-path freshness: re-resolve the rapid-mlx binary at
    // the start of each pull when the manager was told to relocate
    // (i.e. when ServerLocator may have a fresher result than the
    // cached URL). If the cached URL has been deleted under us by a
    // brew/pip upgrade, surface the relaunch message instead of
    // spawning a stale binary.
    internal enum BinaryResolution {
        case resolved(URL, changed: Bool)
        case missing(message: String)

        var failureMessage: String {
            switch self {
            case .resolved:
                return ""
            case .missing(let message):
                return message
            }
        }
    }

    internal static func resolveBinaryForStart(
        cached: URL?,
        shouldRelocate: Bool,
        locate: () -> URL? = ServerLocator.find
    ) -> BinaryResolution {
        let fm = FileManager.default
        let cachedIsExecutable = cached.map { fm.isExecutableFile(atPath: $0.path) } ?? false
        guard shouldRelocate else {
            if let cached, cachedIsExecutable {
                return .resolved(cached, changed: false)
            }
            return .missing(message: "Rapid isn't fully set up. Please restart Rapid-MLX.")
        }

        if let resolved = locate() {
            return .resolved(resolved, changed: resolved != cached)
        }

        if cached != nil && !cachedIsExecutable {
            return .missing(message: "Rapid was updated. Please relaunch Rapid-MLX to finish.")
        }
        return .missing(message: "Rapid isn't fully set up. Please restart Rapid-MLX.")
    }

    // MARK: - Test seam

    /// Test-only: seed a fresh ``.running`` job without spawning a
    /// subprocess. Returns the job so the test can also drive its
    /// ``progress.ingest`` directly.
    internal func _testingSeedJob(
        alias: String,
        hfPath: String? = nil,
        totalBytes: Int64? = nil,
        source: DownloadSource = .mirror
    ) -> Job {
        let job = Job(
            alias: alias,
            hfPath: hfPath,
            totalBytes: totalBytes,
            source: source
        )
        jobs[alias] = job
        stderrTails[alias] = []
        return job
    }

    /// Test-only: feed a stderr line to the manager exactly the way
    /// the readability handler would, but synchronously and without
    /// a backing process.
    internal func _testingIngestStderr(alias: String, line: String) {
        ingestLines(alias: alias, lines: [line], isStderr: true)
    }

    /// Test-only: drive the exit branch to one of the terminal
    /// statuses without a real ``Process``. Mirrors what
    /// ``handleExit`` would do given the same status / reason pair.
    internal func _testingFinish(
        alias: String,
        status: Int32,
        reason: Process.TerminationReason,
        wasCancelling: Bool = false
    ) {
        let process = Process()
        processes[alias] = process
        handleExit(
            alias: alias,
            process: process,
            status: status,
            reason: reason,
            wasCancelling: wasCancelling
        )
    }

    internal func _testingDecodedLines(from data: Data) -> [String] {
        Self.lines(from: data)
    }
}

private final class DownloadCancellationTracker: @unchecked Sendable {
    private let lock = NSLock()
    private var cancelling: Set<ObjectIdentifier> = []

    func markCancelling(_ process: Process) {
        lock.lock()
        cancelling.insert(ObjectIdentifier(process))
        lock.unlock()
    }

    func isCancelling(_ process: Process) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return cancelling.contains(ObjectIdentifier(process))
    }

    func clear(_ process: Process) {
        lock.lock()
        cancelling.remove(ObjectIdentifier(process))
        lock.unlock()
    }
}
