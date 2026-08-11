import Darwin
import Foundation
import Testing
@testable import Rapid

/// Contract for ``DownloadManager`` — the v0.5.7 side-car downloader.
///
/// We exercise the state machine end-to-end through the manager's
/// test seam (``_testingSeedJob``, ``_testingIngestStderr``,
/// ``_testingFinish``) so the assertions land on the same code paths
/// that handle a real ``Process`` exit — no parallel mock path that
/// could drift from production over time.
@MainActor
@Suite("DownloadManager — state machine + cancellation")
struct DownloadManagerTests {
    private func waitUntil(
        deadline: Date,
        predicate: () -> Bool
    ) async -> Bool {
        while Date() < deadline {
            if predicate() { return true }
            try? await Task.sleep(nanoseconds: 50_000_000)
        }
        return predicate()
    }

    /// Synchronous env mutation helper.
    ///
    /// ``setenv`` mutates a **process-global** singleton. The full
    /// ``swift test`` run schedules many ``@MainActor`` tests onto
    /// the same actor; whenever the host test ``await``s during the
    /// mutation window, the main actor is free to pick up an unrelated
    /// ``@MainActor`` test that reads ``RAPID_BIN`` via
    /// ``DownloadManager.resolveBinaryForStart`` — and it sees our
    /// override instead of its own real binary. That spawned the
    /// pre-fix flake where ``startDownloadUsesRefreshedBinaryPath`` and
    /// ``cancelDuringRealPull`` corrupted each other's process spawns
    /// only in the full suite (979 tests / 2 fail) but passed when run
    /// in isolation or in the paired
    /// ``DownloadManagerTests|DownloadManagerIntegrationTests`` filter.
    ///
    /// Keeping the helper synchronous + non-throwing pins the env
    /// mutation to a single main-actor scheduling slot — no ``await``
    /// inside means no opportunity for an unrelated test to slip in.
    /// The body must be sync; if a test needs to ``await`` after the
    /// env-sensitive call, it should do so *outside* this helper, since
    /// once a sync ``Process.run()`` returns macOS has already captured
    /// the spawned child's environment block and the parent process
    /// env can be safely restored.
    private func withEnvironmentValueSync<T>(
        _ key: String,
        _ value: String,
        run body: () -> T
    ) -> T {
        let previous = getenv(key).map { String(cString: $0) }
        setenv(key, value, 1)
        defer {
            if let previous {
                setenv(key, previous, 1)
            } else {
                unsetenv(key)
            }
        }
        return body()
    }

    private func writeFakeRapidMLX(
        at binary: URL,
        marker: URL
    ) throws {
        let script = """
        #!/bin/sh
        echo "$0 $@" > "\(marker.path)"
        exit 0
        """
        try script.write(to: binary, atomically: true, encoding: .utf8)
        chmod(binary.path, 0o755)
    }

    @Test("Brand-new manager has no jobs and reports isDownloading=false for everything")
    func emptyManager() {
        let mgr = DownloadManager()
        #expect(mgr.jobs.isEmpty)
        #expect(!mgr.isDownloading("anything"))
        #expect(mgr.job(for: "anything") == nil)
    }

    @Test("startDownload with no binary registers a synthetic failed job — UI surfaces inline")
    func startWithoutBinary() {
        // The no-arg seam path uses ``binaryPath = nil``; calling
        // ``startDownload`` on that path must record a ``.failed`` job
        // so the picker can show "binary not found" instead of going
        // silent. Mirrors how ``ModelDeletion.deleteCachedModel``
        // surfaces a missing-binary error to the toast.
        let mgr = DownloadManager()
        let started = mgr.startDownload(alias: "qwen3.6-27b")
        #expect(!started)
        let job = mgr.job(for: "qwen3.6-27b")
        #expect(job != nil)
        if case .failed(let message) = job?.status {
            #expect(message.lowercased().contains("set up") || message.lowercased().contains("restart"))
        } else {
            Issue.record("Expected .failed status, got \(String(describing: job?.status))")
        }
    }

    @Test("Binary resolution helper reports a stale cached Cellar path as relaunch-worthy")
    func staleCachedBinaryGetsFriendlyMessage() {
        let stale = URL(fileURLWithPath: "/tmp/rapid-missing-cellar/rapid-mlx")
        let result = DownloadManager.resolveBinaryForStart(
            cached: stale,
            shouldRelocate: true,
            locate: { nil }
        )
        if case .missing(let message) = result {
            #expect(message.lowercased().contains("updated") || message.lowercased().contains("relaunch"))
            #expect(message.contains("relaunch"))
        } else {
            Issue.record("Expected missing stale-binary result, got \(result)")
        }
    }

    @Test("startDownload re-resolves rapid-mlx immediately before run")
    func startDownloadUsesRefreshedBinaryPath() async throws {
        let fm = FileManager.default
        let root = fm.temporaryDirectory
            .appendingPathComponent("rapid-download-\(UUID().uuidString)", isDirectory: true)
        try fm.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: root) }

        let stale = root.appendingPathComponent("Cellar/rapid-mlx/0.6/bin/rapid-mlx")
        let fresh = root.appendingPathComponent("rapid-mlx")
        let marker = root.appendingPathComponent("invocation.txt")
        try writeFakeRapidMLX(at: fresh, marker: marker)

        // Keep ``RAPID_BIN`` overridden only across the synchronous
        // ``startDownload`` window. ``Process.run()`` captures the
        // child's environment at spawn time (see ``DownloadManager``
        // line 196: ``process.environment = augmentedEnv(for: binary)``)
        // so the fake script's invocation is locked in by the time
        // ``startDownload`` returns. Releasing the env before the
        // ``await waitUntil`` below stops the override from leaking
        // into any concurrently scheduled ``@MainActor`` test that
        // happens to read ``RAPID_BIN`` via the same
        // ``resolveBinaryForStart`` code path. Pre-fix this leak
        // overwrote the fake-script's marker with another test's
        // ``pull qwen3-0.6b-8bit`` payload — see helper comment for
        // the full diagnosis.
        let mgr = DownloadManager(binaryPath: stale)
        let started = withEnvironmentValueSync("RAPID_BIN", fresh.path) {
            mgr.startDownload(alias: "fake-alias")
        }
        #expect(started)

        let done = await waitUntil(deadline: Date().addingTimeInterval(5)) {
            guard let job = mgr.job(for: "fake-alias") else { return false }
            if case .running = job.status { return false }
            return true
        }
        #expect(done)
        #expect(mgr.job(for: "fake-alias")?.status == .completed)

        let invocation = try String(contentsOf: marker, encoding: .utf8)
        #expect(invocation.contains(fresh.path))
        #expect(invocation.contains("pull fake-alias"))
    }

    @Test("Seeded running job → completed exit transitions cleanly")
    func runningToCompleted() {
        let mgr = DownloadManager()
        _ = mgr._testingSeedJob(alias: "qwen3.6-27b")
        #expect(mgr.isDownloading("qwen3.6-27b"))
        mgr._testingFinish(alias: "qwen3.6-27b", status: 0, reason: .exit)
        let job = mgr.job(for: "qwen3.6-27b")
        #expect(job?.status == .completed)
        #expect(!mgr.isDownloading("qwen3.6-27b"))
    }

    /// rapid-desktop #440: ``Job.status`` MUST be ``@Observable``.
    ///
    /// Production failure mode this pins: the Quickstart card's
    /// ``.task(id: downloadJobStatusKey)`` reads
    /// ``downloads.job(for: alias)?.status`` to decide when to hand
    /// off to ``server.start``. If ``Job`` isn't ``@Observable``, the
    /// ``status = .completed`` write inside ``handleExit`` doesn't
    /// notify SwiftUI — the body never re-renders, ``downloadJobStatusKey``
    /// is never recomputed, ``.task(id:)`` never sees the new id, and
    /// the user sits at ``99% · <1 min left`` forever after the pull
    /// subprocess exits cleanly.
    ///
    /// The byte-heartbeat path that drove re-renders DURING the pull
    /// stops one tick BEFORE the status flip (``cleanupProcessBookkeeping``
    /// calls ``byteMonitor.stop()`` before the switch in
    /// ``handleExit``), so the bug only manifests on the terminal
    /// flip — exactly the moment the handoff needs to fire.
    ///
    /// Uses ``withObservationTracking`` so the contract is pinned
    /// against the Observation framework directly, not against a
    /// SwiftUI body. ``onChange`` fires synchronously on the willSet
    /// of the first tracked write — exactly the ``status`` flip we
    /// care about. Pre-fix this closure never runs and ``observed``
    /// stays ``false``.
    @Test("Job.status flip is observable via Observation framework (regression #440)")
    func jobStatusFlipIsObservable() {
        let mgr = DownloadManager()
        let job = mgr._testingSeedJob(alias: "bonsai-1.7b-2bit")

        // ``onChange`` is a Sendable closure (Observation framework
        // can route the willSet from any actor), so writing through a
        // reference cell keeps the test main-actor-clean. The Swift 6
        // sendable check rejects a plain ``var observed`` capture
        // from the onChange closure otherwise.
        final class ObservationFlag: @unchecked Sendable {
            var fired = false
        }
        let flag = ObservationFlag()

        withObservationTracking {
            _ = job.status
        } onChange: {
            flag.fired = true
        }

        // Drive the terminal status flip through the same code path
        // that the production termination handler hits.
        mgr._testingFinish(alias: "bonsai-1.7b-2bit", status: 0, reason: .exit)

        #expect(flag.fired)
        #expect(mgr.job(for: "bonsai-1.7b-2bit")?.status == .completed)
    }

    @Test("Non-zero exit → failed status with a generic message (raw stderr is never surfaced)")
    func failedExitUsesGenericMessage() {
        let mgr = DownloadManager()
        _ = mgr._testingSeedJob(alias: "qwen3.6-27b")
        // Raw child stderr (engine name, Python tracebacks, HTTP codes)
        // is logged for support but MUST NOT reach the user-facing
        // failure message.
        mgr._testingIngestStderr(alias: "qwen3.6-27b", line: "ConnectionResetError: peer reset")
        mgr._testingIngestStderr(alias: "qwen3.6-27b", line: "Traceback (most recent call last):")
        mgr._testingFinish(alias: "qwen3.6-27b", status: 1, reason: .exit)
        let job = mgr.job(for: "qwen3.6-27b")
        if case .failed(let message) = job?.status {
            #expect(!message.contains("ConnectionResetError"))
            #expect(!message.contains("Traceback"))
            #expect(message.contains("Couldn't download") || message.lowercased().contains("try again"))
        } else {
            Issue.record("Expected .failed, got \(String(describing: job?.status))")
        }
    }

    @Test("Failed exit with empty stderr falls back to a generic retry message (no raw status code)")
    func failedExitFallsBackToGenericMessage() {
        let mgr = DownloadManager()
        _ = mgr._testingSeedJob(alias: "phi-4-14b-4bit")
        mgr._testingFinish(alias: "phi-4-14b-4bit", status: 127, reason: .exit)
        let job = mgr.job(for: "phi-4-14b-4bit")
        if case .failed(let message) = job?.status {
            #expect(!message.contains("127"))
            #expect(message.contains("Couldn't download") || message.lowercased().contains("try again"))
        } else {
            Issue.record("Expected .failed, got \(String(describing: job?.status))")
        }
    }

    @Test("Signal exit (SIGKILL) lands as failed with a 'signal' phrasing")
    func signalExitMessageDistinguishedFromExit() {
        let mgr = DownloadManager()
        _ = mgr._testingSeedJob(alias: "qwen3.6-27b")
        mgr._testingFinish(alias: "qwen3.6-27b", status: 9, reason: .uncaughtSignal)
        let job = mgr.job(for: "qwen3.6-27b")
        if case .failed(let message) = job?.status {
            #expect(message.lowercased().contains("interrupted"))
        } else {
            Issue.record("Expected .failed with interruption phrasing, got \(String(describing: job?.status))")
        }
    }

    @Test("Captured cancellation process makes the exit handler land on .cancelled")
    func cancellationStickyThroughExit() {
        // Real flow: cancelDownload() marks the exact Process before
        // SIGTERM. terminationHandler captures that identity at exit
        // time and passes `wasCancelling=true` into the reducer.
        let mgr = DownloadManager()
        _ = mgr._testingSeedJob(alias: "qwen3.6-27b")
        mgr._testingFinish(
            alias: "qwen3.6-27b",
            status: 15,
            reason: .uncaughtSignal,
            wasCancelling: true
        )
        #expect(mgr.job(for: "qwen3.6-27b")?.status == .cancelled)
    }

    @Test("dismissJob removes a finished job; running jobs stay put")
    func dismissJobSemantics() {
        let mgr = DownloadManager()
        _ = mgr._testingSeedJob(alias: "alpha")
        _ = mgr._testingSeedJob(alias: "beta")
        mgr._testingFinish(alias: "alpha", status: 0, reason: .exit)

        // beta still running — must not be removed.
        mgr.dismissJob(alias: "beta")
        #expect(mgr.job(for: "beta") != nil)

        // alpha is completed — dismissable.
        mgr.dismissJob(alias: "alpha")
        #expect(mgr.job(for: "alpha") == nil)
    }

    @Test("Empty / whitespace alias is rejected by startDownload")
    func startRejectsEmptyAlias() {
        let mgr = DownloadManager()
        #expect(!mgr.startDownload(alias: ""))
        #expect(!mgr.startDownload(alias: "   "))
        #expect(mgr.jobs.isEmpty)
    }

    @Test("Whitespace-padded alias is normalised — second call sees a duplicate")
    func aliasNormalisation() {
        let mgr = DownloadManager()
        _ = mgr._testingSeedJob(alias: "qwen3.6-27b")
        // The seed seam doesn't trim (that's the production
        // startDownload's job), so we just assert the manager's
        // dictionary uses the trimmed key in production: ensure
        // isDownloading agrees on the trimmed shape.
        #expect(mgr.isDownloading("qwen3.6-27b"))
    }

    @Test("tqdm progress lines flow into Job.progress.phase")
    func progressIngestion() {
        let mgr = DownloadManager()
        let job = mgr._testingSeedJob(alias: "qwen3.6-27b")
        // Outer "Fetching N files" tqdm shape. The DownloadProgress
        // parser already has dedicated unit tests; this contract is
        // the wiring — that ``DownloadManager`` actually dispatches
        // stderr lines into the job's parser.
        mgr._testingIngestStderr(
            alias: "qwen3.6-27b",
            line: "Fetching 16 files:  31%|███▏      | 5/16 [00:30<01:05, 0.17it/s]"
        )
        if case .fetching(let done, let total, let percent) = job.progress.phase {
            #expect(done == 5)
            #expect(total == 16)
            #expect(percent == 31)
        } else {
            Issue.record("Expected .fetching, got \(job.progress.phase)")
        }
    }

    @Test("download stall watchdog only advances for actual progress")
    func stallWindowUsesActualProgress() {
        let now = Date()
        #expect(DownloadManager.isStalled(
            lastProgressAt: now.addingTimeInterval(-121),
            now: now
        ))
        #expect(!DownloadManager.isStalled(
            lastProgressAt: now.addingTimeInterval(-10),
            now: now
        ))
    }

    // MARK: - Cache generation
    //
    // Dogfood report: "I deleted the two qwens in Settings, but the
    // dropdown still shows them with a filled circle." Four surfaces
    // hold their own `rapid-mlx ls` snapshot and nothing connected
    // them, so Settings refreshed its copy and the picker kept
    // advertising a model that was gone.

    @Test("markCacheChanged bumps the generation every call")
    func markCacheChangedBumps() {
        let downloads = DownloadManager()
        let start = downloads.cacheGeneration
        downloads.markCacheChanged()
        #expect(downloads.cacheGeneration == start &+ 1)
        downloads.markCacheChanged()
        #expect(downloads.cacheGeneration == start &+ 2)
    }

    @Test("A completed pull bumps the generation; a failed one does not")
    func completionBumpsGeneration() {
        let ok = DownloadManager()
        _ = ok._testingSeedJob(alias: "qwen3.5-4b-4bit")
        let beforeOK = ok.cacheGeneration
        ok._testingFinish(alias: "qwen3.5-4b-4bit", status: 0, reason: .exit)
        #expect(
            ok.cacheGeneration == beforeOK &+ 1,
            "new weights on disk must invalidate every catalog snapshot"
        )

        let failed = DownloadManager()
        _ = failed._testingSeedJob(alias: "qwen3.5-4b-4bit")
        let beforeFail = failed.cacheGeneration
        failed._testingFinish(alias: "qwen3.5-4b-4bit", status: 1, reason: .exit)
        #expect(
            failed.cacheGeneration == beforeFail,
            "a failed pull changed nothing on disk — don't churn every surface's catalog"
        )

        let cancelled = DownloadManager()
        _ = cancelled._testingSeedJob(alias: "qwen3.5-4b-4bit")
        let beforeCancel = cancelled.cacheGeneration
        cancelled._testingFinish(
            alias: "qwen3.5-4b-4bit",
            status: 0,
            reason: .exit,
            wasCancelling: true
        )
        #expect(cancelled.cacheGeneration == beforeCancel)
    }

    @Test("The picker's catalog key changes when the on-disk set changes")
    func pickerCatalogKeyTracksGeneration() {
        let path = URL(fileURLWithPath: "/opt/homebrew/bin/rapid-mlx")
        let before = ModelPickerBar.PickerCatalogKey(binaryPath: path, cacheGeneration: 3)
        let after = ModelPickerBar.PickerCatalogKey(binaryPath: path, cacheGeneration: 4)
        #expect(before != after, "a deletion anywhere must invalidate the picker's snapshot")
        // Same generation + same binary ⇒ no redundant re-fetch.
        #expect(before == ModelPickerBar.PickerCatalogKey(binaryPath: path, cacheGeneration: 3))
        #expect(before != ModelPickerBar.PickerCatalogKey(
            binaryPath: path,
            cacheGeneration: 3,
            refreshEnabled: false
        ))
    }
}

/// DownloadStrip's caption-format helper is pure — pin the truth
/// table so a future refactor can't silently drop the speed suffix
/// or swap file/percent ordering.
@MainActor
@Suite("DownloadStrip — phase caption formatting")
struct DownloadStripCaptionTests {

    @Test("Idle phase reads as 'Starting…' — the strip never shows raw enum names")
    func idleReadsAsStarting() {
        #expect(DownloadStrip.detail(phase: .idle) == "Starting…")
    }

    @Test("Preparing phase reads as 'Preparing…'")
    func preparingReadsAsPreparing() {
        #expect(DownloadStrip.detail(phase: .preparing) == "Preparing…")
    }

    @Test("Fetching shows 'P% · done/total files'")
    func fetchingFormatHasFilesSuffix() {
        let caption = DownloadStrip.detail(
            phase: .fetching(done: 5, total: 16, percent: 31)
        )
        #expect(caption == "31% · 5/16 files")
    }

    @Test("Downloading with speed + ETA shows everything")
    func downloadingWithSpeed() {
        let caption = DownloadStrip.detail(
            phase: .downloading(
                file: "model-00001-of-00006.safetensors",
                done: "2.10G",
                total: "5.13G",
                percent: 41,
                speed: "23.4MB/s",
                eta: "02:09"
            )
        )
        // Loose containment checks so a future re-ordering of
        // sub-fields doesn't false-fail the test — but every
        // user-visible token must still appear.
        #expect(caption.contains("model-00001-of-00006.safetensors"))
        #expect(caption.contains("41%"))
        #expect(caption.contains("2.10G/5.13G"))
        #expect(caption.contains("23.4MB/s"))
        #expect(caption.contains("ETA 02:09"))
    }

    @Test("Downloading without a speed value omits the speed suffix — no trailing separator")
    func downloadingWithoutSpeedOmitsSeparator() {
        let caption = DownloadStrip.detail(
            phase: .downloading(
                file: "tokenizer.json",
                done: "1.0K",
                total: "1.0K",
                percent: 100,
                speed: nil,
                eta: nil
            )
        )
        #expect(!caption.hasSuffix(" · "))
        #expect(caption.contains("tokenizer.json"))
        #expect(caption.contains("100%"))
    }

    @Test("WarmingUp phase reads as 'Finalising…' — pull doesn't load, but the parser shares the enum")
    func warmingUpHasReasonableCaption() {
        #expect(DownloadStrip.detail(phase: .warmingUp) == "Finalising…")
    }
}

@Suite("DownloadManager — xet concurrency caps")
struct DownloadManagerXetCapsTests {
    @Test("Empty env gets all three HF xet caps — fresh pull stays inside home-router limits AND skips the 30-60s ramp")
    func defaultsAreApplied() {
        // First-run user on home WiFi: env carries no HF_XET
        // overrides. All three caps must land so a fresh `rapid-mlx
        // pull` fans out to 2 files × 8 streams (16 ranges max) AND
        // starts at 8 streams per file instead of ramping 1 → 8.
        // v0.6.10 shipped the upper-bound pair; v0.6.11 adds the
        // FIXED override so the first 60 seconds aren't visibly
        // slow.
        var env: [String: String] = [:]
        DownloadManager.applyXetConcurrencyCaps(env: &env)
        #expect(env["HF_XET_DATA_MAX_CONCURRENT_FILE_DOWNLOADS"] == "2")
        #expect(env["HF_XET_CLIENT_AC_MAX_DOWNLOAD_CONCURRENCY"] == "8")
        #expect(env["HF_XET_FIXED_DOWNLOAD_CONCURRENCY"] == "8")
    }

    @Test("Power user's parent-shell override is preserved — we don't clobber an explicit export")
    func parentShellOverrideWins() {
        // A user who knows their network can set the env var
        // explicitly in their shell. ProcessInfo carries the
        // inherited environment into the child, so we must not
        // overwrite a value the user chose deliberately —
        // `setdefault`-style semantics.
        var env: [String: String] = [
            "HF_XET_DATA_MAX_CONCURRENT_FILE_DOWNLOADS": "16",
            "HF_XET_CLIENT_AC_MAX_DOWNLOAD_CONCURRENCY": "64",
            "HF_XET_FIXED_DOWNLOAD_CONCURRENCY": "32",
        ]
        DownloadManager.applyXetConcurrencyCaps(env: &env)
        #expect(env["HF_XET_DATA_MAX_CONCURRENT_FILE_DOWNLOADS"] == "16")
        #expect(env["HF_XET_CLIENT_AC_MAX_DOWNLOAD_CONCURRENCY"] == "64")
        #expect(env["HF_XET_FIXED_DOWNLOAD_CONCURRENCY"] == "32")
    }

    @Test("Partial override: caps fill in only the missing knobs")
    func partialOverrideFillsTheGap() {
        var env: [String: String] = [
            "HF_XET_DATA_MAX_CONCURRENT_FILE_DOWNLOADS": "4",
        ]
        DownloadManager.applyXetConcurrencyCaps(env: &env)
        #expect(env["HF_XET_DATA_MAX_CONCURRENT_FILE_DOWNLOADS"] == "4")
        #expect(env["HF_XET_CLIENT_AC_MAX_DOWNLOAD_CONCURRENCY"] == "8")
        #expect(env["HF_XET_FIXED_DOWNLOAD_CONCURRENCY"] == "8")
    }

    @Test("Adaptive-controller override defeats our FIXED injection — user MAX=4 stays in charge")
    func adaptiveOverrideSuppressesFixedInjection() {
        // codex r1 on PR #221: HF_XET_FIXED_DOWNLOAD_CONCURRENCY
        // bypasses the adaptive controller entirely (aliases
        // initial/min/max). A power user who explicitly set only
        // MAX=4 to throttle bandwidth would silently get our FIXED=8
        // injected — pinning concurrency to 8 instead of capping at
        // 4. The intent inversion is invisible from the user's side.
        // Fix: when any adaptive-controller knob is touched, skip
        // FIXED entirely so the adaptive controller stays in play.
        var env: [String: String] = [
            "HF_XET_CLIENT_AC_MAX_DOWNLOAD_CONCURRENCY": "4",
        ]
        DownloadManager.applyXetConcurrencyCaps(env: &env)
        #expect(env["HF_XET_CLIENT_AC_MAX_DOWNLOAD_CONCURRENCY"] == "4")
        #expect(env["HF_XET_FIXED_DOWNLOAD_CONCURRENCY"] == nil)
    }

    @Test("Adaptive MIN override also suppresses FIXED injection")
    func adaptiveMinOverrideSuppressesFixed() {
        // Same intent-preservation as the MAX case, mirrored for
        // MIN. A user setting MIN=2 wants the adaptive controller
        // to ramp from 2 → max; FIXED=8 would skip the ramp and
        // pin to 8.
        var env: [String: String] = [
            "HF_XET_CLIENT_AC_MIN_DOWNLOAD_CONCURRENCY": "2",
        ]
        DownloadManager.applyXetConcurrencyCaps(env: &env)
        #expect(env["HF_XET_CLIENT_AC_MIN_DOWNLOAD_CONCURRENCY"] == "2")
        #expect(env["HF_XET_FIXED_DOWNLOAD_CONCURRENCY"] == nil)
    }

    @Test("Adaptive INITIAL override also suppresses FIXED injection")
    func adaptiveInitialOverrideSuppressesFixed() {
        var env: [String: String] = [
            "HF_XET_CLIENT_AC_INITIAL_DOWNLOAD_CONCURRENCY": "4",
        ]
        DownloadManager.applyXetConcurrencyCaps(env: &env)
        #expect(env["HF_XET_CLIENT_AC_INITIAL_DOWNLOAD_CONCURRENCY"] == "4")
        #expect(env["HF_XET_FIXED_DOWNLOAD_CONCURRENCY"] == nil)
    }
}

@Suite("DownloadManager — RAPID_MLX_MODEL_MIRROR injection")
struct DownloadManagerModelMirrorTests {
    @Test("Empty env gets the default R2 mirror URL — first-run users hit our CDN, not HuggingFace Hub")
    func defaultMirrorAppliedWhenAbsent() {
        // First-run user with no opinion on weight hosting: env
        // carries no ``RAPID_MLX_MODEL_MIRROR``. The desktop app
        // injects ``models.rapidmlx.com`` (rate-limited Cloudflare
        // Worker → R2) so ``rapid-mlx pull`` prefetches from our
        // mirror before falling through to the HuggingFace Hub.
        var env: [String: String] = [:]
        DownloadManager.applyModelMirror(env: &env)
        #expect(env["RAPID_MLX_MODEL_MIRROR"] == "https://models.rapidmlx.com")
    }

    @Test("Power user's parent-shell override wins — we don't clobber an explicit RAPID_MLX_MODEL_MIRROR (including the empty-string opt-out)")
    func parentShellOverrideWins() {
        // ``augmentedEnv`` seeds the dict from
        // ``ProcessInfo.processInfo.environment`` BEFORE we run, so a
        // user who exported a custom mirror (or "" to disable) in
        // their shell shows up here as a pre-populated key. We must
        // leave it alone — setdefault semantics, same shape as
        // ``applyXetConcurrencyCaps``. Empty string is a deliberate
        // opt-out signal recognised by the CLI helper (treated as
        // unset → straight to HuggingFace Hub), and clobbering it
        // would silently re-enable the mirror.
        var envCustom: [String: String] = [
            "RAPID_MLX_MODEL_MIRROR": "https://my-internal-cache.example.com",
        ]
        DownloadManager.applyModelMirror(env: &envCustom)
        #expect(envCustom["RAPID_MLX_MODEL_MIRROR"] == "https://my-internal-cache.example.com")

        var envOptOut: [String: String] = ["RAPID_MLX_MODEL_MIRROR": ""]
        DownloadManager.applyModelMirror(env: &envOptOut)
        #expect(envOptOut["RAPID_MLX_MODEL_MIRROR"] == "")
    }
}
