import Foundation
import Testing
@testable import Rapid

/// v0.5.7 integration smoke. Drives a REAL ``rapid-mlx pull``
/// subprocess (no ``_testingSeedJob`` / ``_testingFinish`` seam) to
/// catch wiring bugs the unit suite can't see by construction:
///
///   * The readability handler hop-back from the IO thread to the
///     ``@MainActor`` actually delivers stderr lines.
///   * ``terminationHandler`` actually fires on real-process exit and
///     transitions the job out of ``.running``.
///   * Pipe / env / stdin-nulldev plumbing matches what the CLI needs.
///   * The subprocess actually finds itself on PATH the way the
///     production code expects.
///
/// Picks cached models (HF resolves snapshot, sees all files present,
/// exits 0) so each test takes ~2 s instead of 10 minutes; we are
/// validating the manager's plumbing, not the CLI's download speed.
///
/// All tests skip cleanly if ``rapid-mlx`` isn't on the host PATH.
/// Doesn't depend on network — tested models are already cached on
/// the dev machine (and on any first-touch user's machine after
/// they've used rapid-desktop once).
@MainActor
@Suite("DownloadManager — real ``rapid-mlx pull`` subprocess wiring")
struct DownloadManagerIntegrationTests {

    /// Resolved ``rapid-mlx`` URL or ``nil`` if not on PATH.
    /// Tests use this to decide whether to skip vs proceed.
    private static func resolveBinary() -> URL? {
        let candidates = [
            "/opt/homebrew/bin/rapid-mlx",
            "/usr/local/bin/rapid-mlx",
        ]
        for path in candidates where FileManager.default.isExecutableFile(atPath: path) {
            return URL(fileURLWithPath: path)
        }
        return nil
    }

    /// Scope ``RAPID_BIN`` to the SYNCHRONOUS duration of ``body``,
    /// then restore the prior value (or unset entirely if the
    /// variable wasn't set when we entered).
    ///
    /// v0.8.10 cutover: ``ServerLocator.find()`` no longer walks
    /// ``/opt/homebrew/bin`` etc., so ``DownloadManager``'s relocate
    /// branch (``shouldRelocate=true`` in the production
    /// ``init(binaryPath:)``) would return nil and the job would
    /// synthesize ``"rapid-mlx binary not found."`` even though we
    /// passed the host's brew binary to the constructor. Expose the
    /// host install via ``RAPID_BIN`` (slot 1 of the new 3-slot
    /// chain) so the relocate call resolves through it.
    ///
    /// SYNC-only by design — the helper must wrap only the
    /// ``startDownload`` invocation, NOT any ``await``ed polling
    /// window that follows. ``setenv`` mutates a process-global
    /// singleton; if the wrapper spans an ``await``, the Swift
    /// Testing scheduler can pick up an unrelated parallel suite that
    /// reads the process environment through ``ServerLocator.find()``
    /// or ``classify(resolved:)`` and sees our override leak, flaking
    /// that suite. ``Process.run()`` captures
    /// ``process.environment`` synchronously inside ``startDownload``,
    /// so by the time ``startDownload`` returns the child has its
    /// own env copy and the parent can restore safely. Codex r1 +
    /// r2 reviews both flagged this; the existing
    /// ``DownloadManagerTests.withEnvironmentValueSync`` uses the
    /// same sync-scope pattern.
    @discardableResult
    private static func withRapidBin<R>(
        _ binary: URL,
        run body: () -> R
    ) -> R {
        let prior = getenv("RAPID_BIN").map { String(cString: $0) }
        setenv("RAPID_BIN", binary.path, 1)
        defer {
            if let prior {
                setenv("RAPID_BIN", prior, 1)
            } else {
                unsetenv("RAPID_BIN")
            }
        }
        return body()
    }

    /// Spin until ``predicate`` is true or ``deadline`` expires.
    /// Yields to the run loop on each poll so ``terminationHandler``'s
    /// MainActor hop can land. Returns true if the predicate held by
    /// the deadline; false on timeout.
    private func waitUntil(
        deadline: Date,
        predicate: () -> Bool
    ) async -> Bool {
        while Date() < deadline {
            if predicate() { return true }
            try? await Task.sleep(nanoseconds: 100_000_000)
        }
        return predicate()
    }

    @Test("Cached alias: real pull exits 0 → job lands in .completed via terminationHandler")
    func cachedAliasCompletes() async throws {
        guard let binary = Self.resolveBinary() else {
            // No rapid-mlx on PATH — skip cleanly. Real CI would
            // install it; this guard keeps the suite green on
            // bare developer checkouts.
            return
        }
        // ``qwen3-0.6b-8bit`` is cached on the dev machine and 1.2 GiB
        // on disk (per ``rapid-mlx ls``). HF resolves the snapshot,
        // sees every file present, exits 0 in ~1-2 s. Same shape
        // every first-touch user hits after their second model swap.
        // The test is about plumbing, not model choice — using a
        // sub-1B model here is fine because we're not running
        // inference (which is what the no-tiny-models rule guards).
        let mgr = DownloadManager(binaryPath: binary, cacheAccessCheck: { nil })
        let alias = "qwen3-0.6b-8bit"
        let started = Self.withRapidBin(binary) {
            mgr.startDownload(alias: alias)
        }
        #expect(started, "startDownload should succeed against a real binary")
        #expect(mgr.isDownloading(alias))

        // Cached pulls land in 1-3 s; give 15 s of headroom for
        // a slow CI host. If we time out here it's a real bug,
        // not test flake.
        let done = await waitUntil(deadline: Date().addingTimeInterval(15)) {
            guard let job = mgr.job(for: alias) else { return false }
            if case .running = job.status { return false }
            return true
        }
        #expect(done, "Pull subprocess did not exit within 15 s — terminationHandler bug?")

        // The transition we care about most: real Process.exit code
        // 0 flows through to .completed (NOT .failed).
        let job = mgr.job(for: alias)
        #expect(job?.status == .completed, "Expected .completed, got \(String(describing: job?.status))")
        #expect(!mgr.isDownloading(alias))
    }

    @Test("Bad alias: real pull exits non-zero → job lands in .failed with a non-empty message")
    func badAliasFails() async throws {
        guard let binary = Self.resolveBinary() else { return }
        let mgr = DownloadManager(binaryPath: binary, cacheAccessCheck: { nil })
        // Deliberately bogus alias — rapid-mlx prints an error and
        // exits non-zero. The pattern catches a bug where bad-alias
        // failures silently land as .completed (e.g. forgetting to
        // check terminationStatus).
        let alias = "this-alias-does-not-exist-zzz-xyz"
        let started = Self.withRapidBin(binary) {
            mgr.startDownload(alias: alias)
        }
        #expect(started)

        let done = await waitUntil(deadline: Date().addingTimeInterval(15)) {
            guard let job = mgr.job(for: alias) else { return false }
            if case .running = job.status { return false }
            return true
        }
        #expect(done)

        let job = mgr.job(for: alias)
        if case .failed(let message) = job?.status {
            #expect(!message.isEmpty, "Failure message should not be empty")
        } else {
            Issue.record("Expected .failed for bogus alias, got \(String(describing: job?.status))")
        }
    }

    @Test("Cancel during real pull: SIGTERM lands as .cancelled, not .failed")
    func cancelDuringRealPull() async throws {
        guard let binary = Self.resolveBinary() else { return }
        let mgr = DownloadManager(binaryPath: binary, cacheAccessCheck: { nil })
        let alias = "qwen3-0.6b-8bit"
        let started = Self.withRapidBin(binary) {
            mgr.startDownload(alias: alias)
        }
        #expect(started)

        // Cancel within a few hundred ms — even cached pulls have a
        // brief Python-startup window we can interrupt. The cancel
        // path is what users hit when they realise they picked the
        // wrong alias; the bug we're guarding against is the SIGTERM
        // exit being misclassified as ``.failed`` (it's a non-zero
        // exit code) instead of ``.cancelled``.
        try? await Task.sleep(nanoseconds: 300_000_000)
        mgr.cancelDownload(alias: alias)

        // Give the kill window + handleExit hop a generous budget.
        let done = await waitUntil(deadline: Date().addingTimeInterval(15)) {
            guard let job = mgr.job(for: alias) else { return false }
            if case .running = job.status { return false }
            return true
        }
        #expect(done)

        let job = mgr.job(for: alias)
        // Edge case: if cancel raced the natural completion (cached
        // alias completed before the 300 ms window), the job may
        // legitimately be ``.completed``. That's still correct
        // behaviour — the manager respected whichever signal won.
        // The bug we're catching is ``.failed`` (SIGTERM
        // misclassified as crash).
        switch job?.status {
        case .cancelled, .completed:
            break  // acceptable
        default:
            let status = String(describing: job?.status)
            Issue.record(Comment(rawValue: "Cancel during pull landed unexpectedly as \(status) — expected .cancelled (raced) or .completed (cached win)."))
        }
    }

    @Test("dismissJob after a real completed pull frees the slot for a fresh start")
    func dismissAndRestart() async throws {
        guard let binary = Self.resolveBinary() else { return }
        let mgr = DownloadManager(binaryPath: binary, cacheAccessCheck: { nil })
        let alias = "qwen3-0.6b-8bit"

        // Round 1.
        Self.withRapidBin(binary) {
            _ = mgr.startDownload(alias: alias)
        }
        _ = await waitUntil(deadline: Date().addingTimeInterval(15)) {
            guard let job = mgr.job(for: alias) else { return false }
            if case .running = job.status { return false }
            return true
        }
        #expect(mgr.job(for: alias)?.status == .completed)

        // Dismiss → slot must be empty so startDownload can register
        // a fresh job (the production code rejects double-start while
        // a job exists).
        mgr.dismissJob(alias: alias)
        #expect(mgr.job(for: alias) == nil)

        // Round 2 — fresh start must succeed against the same alias.
        // Codex r2 MAJOR: each ``startDownload`` (BOTH rounds) needs
        // its own sync ``withRapidBin`` window — if we only set it
        // around round 1, round 2's relocate call reads the
        // (already-restored) un-set ``RAPID_BIN`` and falls back to
        // the missing branch.
        let started = Self.withRapidBin(binary) {
            mgr.startDownload(alias: alias)
        }
        #expect(started)
        #expect(mgr.isDownloading(alias))

        // Let it finish so we don't leak the subprocess past the test.
        _ = await waitUntil(deadline: Date().addingTimeInterval(15)) {
            guard let job = mgr.job(for: alias) else { return false }
            if case .running = job.status { return false }
            return true
        }
    }
}
