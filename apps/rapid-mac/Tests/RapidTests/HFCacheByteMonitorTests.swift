import Foundation
import Testing
@testable import Rapid

/// Contract for ``HFCacheByteMonitor`` + the byte-progress fields it
/// drives on ``DownloadProgress``.
///
/// The HuggingFace "Fetching N files" tqdm bar that ``DownloadProgress``
/// parses today counts FILES, not BYTES. On a 6.8 GB / 11-shard cold
/// download the outer bar reads "0/11 files (0%)" for many minutes
/// while the first shard streams silently. The byte monitor closes
/// that gap by sampling the cache dir directly so the UI can render
/// real bytes-on-disk progress.
@MainActor
@Suite("HFCacheByteMonitor — bytes-on-disk progress")
struct HFCacheByteMonitorTests {

    /// Build a fresh HF-cache-shaped fixture directory and return its
    /// per-alias subdir URL. Caller is responsible for cleanup via
    /// ``fm.removeItem``.
    private func makeFixtureCacheDir(
        hubRoot: URL,
        owner: String,
        repo: String
    ) throws -> URL {
        let fm = FileManager.default
        let dirName = "models--\(owner)--\(repo)"
        let modelDir = hubRoot.appendingPathComponent(dirName, isDirectory: true)
        let blobsDir = modelDir.appendingPathComponent("blobs", isDirectory: true)
        let snapshotsDir = modelDir
            .appendingPathComponent("snapshots", isDirectory: true)
            .appendingPathComponent("main", isDirectory: true)
        try fm.createDirectory(at: blobsDir, withIntermediateDirectories: true)
        try fm.createDirectory(at: snapshotsDir, withIntermediateDirectories: true)
        return modelDir
    }

    // MARK: - cacheDirectoryURL

    @Test("cacheDirectoryURL: owner/repo → models--<owner>--<repo>")
    func cacheDirectoryURLWellFormed() {
        let root = URL(fileURLWithPath: "/tmp/hub", isDirectory: true)
        let url = HFCacheByteMonitor.cacheDirectoryURL(
            hubCacheRoot: root,
            hfPath: "mlx-community/gemma-4-12B-it-4bit"
        )
        #expect(url?.path == "/tmp/hub/models--mlx-community--gemma-4-12B-it-4bit")
    }

    @Test("cacheDirectoryURL: bare repo (no slash) still resolves")
    func cacheDirectoryURLBareRepo() {
        let root = URL(fileURLWithPath: "/tmp/hub", isDirectory: true)
        let url = HFCacheByteMonitor.cacheDirectoryURL(
            hubCacheRoot: root,
            hfPath: "gpt2"
        )
        #expect(url?.path == "/tmp/hub/models--gpt2")
    }

    @Test("cacheDirectoryURL: rejects path traversal in hfPath")
    func cacheDirectoryURLRejectsTraversal() {
        let root = URL(fileURLWithPath: "/tmp/hub", isDirectory: true)
        let url = HFCacheByteMonitor.cacheDirectoryURL(
            hubCacheRoot: root,
            hfPath: "../escape/repo"
        )
        #expect(url == nil)
    }

    @Test("cacheDirectoryURL: rejects empty / whitespace hfPath")
    func cacheDirectoryURLRejectsEmpty() {
        let root = URL(fileURLWithPath: "/tmp/hub", isDirectory: true)
        #expect(HFCacheByteMonitor.cacheDirectoryURL(hubCacheRoot: root, hfPath: "") == nil)
        #expect(HFCacheByteMonitor.cacheDirectoryURL(hubCacheRoot: root, hfPath: "   ") == nil)
    }

    // MARK: - directoryByteCount

    @Test("directoryByteCount: missing dir returns 0 — no observation forwarded")
    func directoryByteCountMissing() {
        let url = URL(fileURLWithPath: "/tmp/does-not-exist-\(UUID().uuidString)", isDirectory: true)
        #expect(HFCacheByteMonitor.directoryByteCount(at: url) == 0)
    }

    @Test("directoryByteCount: sums regular files across blobs/ + snapshots/")
    func directoryByteCountSums() throws {
        let fm = FileManager.default
        let hubRoot = fm.temporaryDirectory
            .appendingPathComponent("rapid-hf-test-\(UUID().uuidString)", isDirectory: true)
        try fm.createDirectory(at: hubRoot, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: hubRoot) }

        let modelDir = try makeFixtureCacheDir(
            hubRoot: hubRoot,
            owner: "owner",
            repo: "repo"
        )
        let blob1 = modelDir.appendingPathComponent("blobs/aaaa")
        let blob2 = modelDir.appendingPathComponent("blobs/bbbb")
        let payload1 = Data(repeating: 0x41, count: 1024)
        let payload2 = Data(repeating: 0x42, count: 2048)
        try payload1.write(to: blob1)
        try payload2.write(to: blob2)

        let total = HFCacheByteMonitor.directoryByteCount(at: modelDir)
        #expect(total >= 1024 + 2048)
    }

    @Test("directoryByteCount: hardlinks counted once (snapshot ↔ blob dedup)")
    func directoryByteCountDedupesHardlinks() throws {
        let fm = FileManager.default
        let hubRoot = fm.temporaryDirectory
            .appendingPathComponent("rapid-hf-test-\(UUID().uuidString)", isDirectory: true)
        try fm.createDirectory(at: hubRoot, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: hubRoot) }

        let modelDir = try makeFixtureCacheDir(
            hubRoot: hubRoot,
            owner: "owner",
            repo: "repo"
        )
        let blob = modelDir.appendingPathComponent("blobs/abc123")
        let snapshot = modelDir.appendingPathComponent("snapshots/main/model.bin")
        let payload = Data(repeating: 0x55, count: 4096)
        try payload.write(to: blob)
        // HF stores blobs once under blobs/ and hardlinks them into
        // snapshots/. Counting both naively would double the bytes for
        // a model with a full snapshot tree. Replicate the layout:
        try fm.linkItem(at: blob, to: snapshot)

        let total = HFCacheByteMonitor.directoryByteCount(at: modelDir)
        // 4096 once, not 8192 — even though both blob and snapshot
        // paths show up in the enumerator.
        #expect(total == 4096)
    }

    // MARK: - DownloadProgress integration

    @Test("DownloadProgress.applyDiskObservation: ignores zero (no flap on transient read failure)")
    func applyDiskObservationRejectsZero() {
        let progress = DownloadProgress()
        progress.applyDiskObservation(bytes: 0)
        #expect(progress.bytesDownloaded == nil)
        #expect(progress.hasDiskObservation == false)
    }

    @Test("DownloadProgress.applyDiskObservation: positive bytes set the channel + tick the lastTickAt clock")
    func applyDiskObservationSetsBytes() {
        let progress = DownloadProgress()
        let before = progress.lastTickAt
        progress.applyDiskObservation(bytes: 1_500_000_000)
        #expect(progress.bytesDownloaded == 1_500_000_000)
        #expect(progress.hasDiskObservation == true)
        #expect(progress.lastTickAt > before)
    }

    @Test("DownloadProgress.setTotalBytes: rejects zero / negative — leaves UI in 'X MB downloaded' mode")
    func setTotalBytesRejectsNonPositive() {
        let progress = DownloadProgress()
        progress.setTotalBytes(0)
        #expect(progress.totalBytes == nil)
        progress.setTotalBytes(-5)
        #expect(progress.totalBytes == nil)
        progress.setTotalBytes(nil)
        #expect(progress.totalBytes == nil)
    }

    @Test("DownloadProgress.reset clears bytes + total + disk-observation flag")
    func resetClearsBytesAndTotal() {
        let progress = DownloadProgress()
        progress.setTotalBytes(6_800_000_000)
        progress.applyDiskObservation(bytes: 1_200_000_000)
        progress.reset()
        #expect(progress.bytesDownloaded == nil)
        #expect(progress.totalBytes == nil)
        #expect(progress.hasDiskObservation == false)
    }

    // MARK: - progressFraction priority order

    @Test("progressFraction prefers bytes/total when disk observation present (the desired UX)")
    func progressFractionPrefersBytes() {
        let progress = DownloadProgress()
        progress.setTotalBytes(10 * 1024 * 1024 * 1024)        // 10 GiB total
        progress.applyDiskObservation(bytes: 2 * 1024 * 1024 * 1024)  // 2 GiB on disk
        // The tqdm phase says fetching at 0% (the bug): bytes-on-disk
        // should win and report ~20%, not 0%.
        progress.ingest("Fetching 9 files:  0%| | 0/9 [00:00<?, ?it/s]")
        let fraction = progress.progressFraction ?? -1
        #expect(abs(fraction - 0.2) < 0.001)
    }

    @Test("progressFraction falls back to file-count percent when bytes unknown")
    func progressFractionFallsBackToTqdm() {
        let progress = DownloadProgress()
        // No disk observation, no total — fetching tqdm percent must win.
        progress.ingest("Fetching 16 files:  31%|███▏      | 5/16 [00:42<01:32, 0.12it/s]")
        #expect(progress.progressFraction == 0.31)
    }

    @Test("progressFraction discards a stale total that observed bytes exceed (#1550)")
    func progressFractionDiscardsStaleTotal() {
        let progress = DownloadProgress()
        progress.setTotalBytes(1_000_000)
        // The user re-downloaded after expanding the model on disk:
        // catalog total is stale and bytes-on-disk overshoots. The bar
        // must stop presenting the disproven denominator as 100% complete.
        progress.applyDiskObservation(bytes: 3_000_000)
        #expect(progress.totalBytes == nil)
        #expect(progress.progressFraction == nil)
        #expect(progress.progressSubtitle == "2.9 MB downloaded")
    }

    @Test("progressFraction nil for idle / preparing / warmingUp without disk observation")
    func progressFractionNilIndeterminate() {
        let progress = DownloadProgress()
        #expect(progress.progressFraction == nil)
        progress.ingest("INFO:vllm_mlx.server:Loading model with BatchedEngine: x/y")
        #expect(progress.progressFraction == nil)
        progress.ingest("INFO:vllm_mlx.server:Warming up (compiling Metal shaders)")
        #expect(progress.progressFraction == nil)
    }

    // MARK: - progressSubtitle priority order

    @Test("progressSubtitle: bytes + total → 'X.X / Y.Y GB · Z%' (the desired UX)")
    func progressSubtitleBytesAndTotal() {
        let progress = DownloadProgress()
        progress.setTotalBytes(Int64(6.8 * 1024 * 1024 * 1024))
        progress.applyDiskObservation(bytes: Int64(1.2 * 1024 * 1024 * 1024))
        let subtitle = progress.progressSubtitle ?? ""
        #expect(subtitle.contains("1.2 GB"))
        #expect(subtitle.contains("6.8 GB"))
        #expect(subtitle.contains("·"))
        // 1.2 / 6.8 ≈ 17.6% → either "17%" or "18%" depending on rounding.
        #expect(subtitle.contains("17%") || subtitle.contains("18%"))
    }

    @Test("progressSubtitle: bytes only → 'X.X GB downloaded'")
    func progressSubtitleBytesOnly() {
        let progress = DownloadProgress()
        progress.applyDiskObservation(bytes: Int64(1.2 * 1024 * 1024 * 1024))
        let subtitle = progress.progressSubtitle ?? ""
        #expect(subtitle.contains("1.2 GB"))
        #expect(subtitle.contains("downloaded"))
    }

    @Test("progressSubtitle: no bytes — falls back to tqdm file-count copy")
    func progressSubtitleFallbackToFiles() {
        let progress = DownloadProgress()
        progress.ingest("Fetching 9 files:  0%| | 0/9 [00:00<?, ?it/s]")
        let subtitle = progress.progressSubtitle ?? ""
        #expect(subtitle.contains("0 of 9 files"))
    }

    @Test("progressSubtitle: nil for idle / preparing / warmingUp without disk observation")
    func progressSubtitleNilIndeterminate() {
        let progress = DownloadProgress()
        #expect(progress.progressSubtitle == nil)
        progress.ingest("INFO:vllm_mlx.server:Loading model with BatchedEngine: x/y")
        #expect(progress.progressSubtitle == nil)
        progress.ingest("INFO:vllm_mlx.server:Warming up (compiling Metal shaders)")
        #expect(progress.progressSubtitle == nil)
    }

    // MARK: - formatBytes contract

    @Test("formatBytes: sub-100 of any unit gets 1 decimal place")
    func formatBytesOneDecimal() {
        let oneGiB = Int64(1.2 * 1024 * 1024 * 1024)
        #expect(DownloadProgress.formatBytes(oneGiB) == "1.2 GB")
        let fourMB = Int64(4.7 * 1024 * 1024)
        #expect(DownloadProgress.formatBytes(fourMB) == "4.7 MB")
    }

    @Test("formatBytes: ≥100 of any unit drops the decimal — '250 MB' not '250.4 MB'")
    func formatBytesNoDecimalForLarge() {
        let twoFiftyMB = Int64(250.4 * 1024 * 1024)
        #expect(DownloadProgress.formatBytes(twoFiftyMB) == "250 MB")
    }

    @Test("formatBytes: sub-KB stays in B")
    func formatBytesBytes() {
        #expect(DownloadProgress.formatBytes(123) == "123 B")
    }

    @Test("formatBytes: negative / zero → '0 B', not garbage")
    func formatBytesDefensive() {
        #expect(DownloadProgress.formatBytes(0) == "0 B")
        #expect(DownloadProgress.formatBytes(-1) == "0 B")
    }

    // MARK: - DownloadManager.estimateTotalBytes

    @Test("estimateTotalBytes: known alias → positive, roughly matches ModelSizing weight estimate")
    func estimateTotalBytesKnown() {
        // gemma-4-12b is in ModelSizing's family table — params=12,
        // 4-bit → weightsGB ≈ 12 × 0.55 = 6.6 GB → ~7 billion bytes.
        let bytes = DownloadManager.estimateTotalBytes(for: "gemma-4-12b-4bit") ?? 0
        // Allow a wide band — the contract is "positive, GB-scale";
        // exact magic numbers belong in ModelSizing's own tests.
        #expect(bytes >= 1_000_000_000)
        #expect(bytes <= 50_000_000_000)
    }

    @Test("estimateTotalBytes: alias with no parseable params → nil (UI falls back to 'X MB downloaded')")
    func estimateTotalBytesUnknown() {
        // Bare alias with no '<n>b' shape — ModelSizing returns
        // params=nil → weightsGB=0 → estimator returns nil.
        let bytes = DownloadManager.estimateTotalBytes(for: "totally-fake-alias")
        #expect(bytes == nil)
    }

    // MARK: - End-to-end: monitor → progress → UI subtitle

    @Test("End-to-end: HFCacheByteMonitor.start polls a fixture dir and updates DownloadProgress")
    func endToEndPollsFixture() async throws {
        let fm = FileManager.default
        let hubRoot = fm.temporaryDirectory
            .appendingPathComponent("rapid-hf-e2e-\(UUID().uuidString)", isDirectory: true)
        try fm.createDirectory(at: hubRoot, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: hubRoot) }

        let modelDir = try makeFixtureCacheDir(
            hubRoot: hubRoot,
            owner: "test-org",
            repo: "test-repo"
        )
        let blob = modelDir.appendingPathComponent("blobs/seed")
        try Data(repeating: 0x7F, count: 8192).write(to: blob)

        let progress = DownloadProgress()
        progress.setTotalBytes(64 * 1024)  // 64 KiB total — % well-defined
        let handle = HFCacheByteMonitor.start(
            cacheDir: modelDir,
            progress: progress,
            pollInterval: 0.2
        )
        defer { handle.stop() }
        #expect(await handle.waitForFirstPoll())

        #expect(progress.hasDiskObservation == true)
        #expect((progress.bytesDownloaded ?? 0) >= 8192)
        let subtitle = progress.progressSubtitle ?? ""
        // 8 KiB / 64 KiB = 12.5% → either 12 or 13 after rounding.
        #expect(subtitle.contains("KB"))
        await handle.stopAndWait()
    }

    @Test("End-to-end: missing cache dir leaves DownloadProgress untouched — UI falls back cleanly")
    func endToEndMissingDir() async throws {
        let absent = URL(fileURLWithPath: "/tmp/rapid-hf-absent-\(UUID().uuidString)", isDirectory: true)
        let progress = DownloadProgress()
        let handle = HFCacheByteMonitor.start(
            cacheDir: absent,
            progress: progress,
            pollInterval: 0.2
        )
        defer { handle.stop() }
        #expect(await handle.waitForFirstPoll() == false)
        #expect(progress.hasDiskObservation == false)
        #expect(progress.bytesDownloaded == nil)
        await handle.stopAndWait()
    }

    @Test("End-to-end: Handle.stop() cancels the poll task — no further updates after cancel")
    func endToEndStopHaltsPolling() async throws {
        let fm = FileManager.default
        let hubRoot = fm.temporaryDirectory
            .appendingPathComponent("rapid-hf-stop-\(UUID().uuidString)", isDirectory: true)
        try fm.createDirectory(at: hubRoot, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: hubRoot) }

        let modelDir = try makeFixtureCacheDir(
            hubRoot: hubRoot,
            owner: "stop",
            repo: "test"
        )
        let blob = modelDir.appendingPathComponent("blobs/a")
        try Data(repeating: 0x11, count: 1024).write(to: blob)

        let progress = DownloadProgress()
        let handle = HFCacheByteMonitor.start(
            cacheDir: modelDir,
            progress: progress,
            pollInterval: 0.1
        )
        defer { handle.stop() }
        #expect(await handle.waitForFirstPoll())
        #expect(progress.hasDiskObservation == true)
        let firstBytes = progress.bytesDownloaded
        await handle.stopAndWait()
        // Grow the dir AFTER stop — the monitor must NOT re-observe.
        let blob2 = modelDir.appendingPathComponent("blobs/b")
        try Data(repeating: 0x22, count: 1024 * 1024).write(to: blob2)
        // The byte count should not have advanced from the post-stop write.
        #expect(progress.bytesDownloaded == firstBytes)
    }

    @Test("waitForFirstPoll shares one completed result with concurrent and later callers")
    func firstPollResultIsShared() async throws {
        let fm = FileManager.default
        let hubRoot = fm.temporaryDirectory
            .appendingPathComponent("rapid-hf-shared-poll-\(UUID().uuidString)", isDirectory: true)
        try fm.createDirectory(at: hubRoot, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: hubRoot) }

        let modelDir = try makeFixtureCacheDir(
            hubRoot: hubRoot,
            owner: "shared",
            repo: "poll"
        )
        try Data(repeating: 0x33, count: 4096)
            .write(to: modelDir.appendingPathComponent("blobs/seed"))

        let handle = HFCacheByteMonitor.start(
            cacheDir: modelDir,
            progress: DownloadProgress(),
            pollInterval: 0.2
        )
        defer { handle.stop() }

        async let first = handle.waitForFirstPoll()
        async let second = handle.waitForFirstPoll()
        let (firstResult, secondResult) = await (first, second)
        #expect(firstResult)
        #expect(secondResult)
        #expect(await handle.waitForFirstPoll())
        await handle.stopAndWait()
    }

    @Test("stop before the first poll completes waiters without publishing")
    func stopBeforeFirstPollCompletesWaiters() async {
        let progress = DownloadProgress()
        let handle = HFCacheByteMonitor.start(
            cacheDir: URL(
                fileURLWithPath: "/tmp/rapid-hf-prepoll-stop-\(UUID().uuidString)",
                isDirectory: true
            ),
            progress: progress,
            pollInterval: 0.2,
            isCancelled: { true }
        )

        handle.stop()
        #expect(await handle.waitForFirstPoll() == false)
        await handle.stopAndWait()
        #expect(progress.hasDiskObservation == false)
        #expect(progress.bytesDownloaded == nil)
    }
}

/// Caption-format tests for the byte-aware ``DownloadStrip.detail``
/// helper. The base phase-only contract still lives in
/// ``DownloadStripCaptionTests``; the cases here pin the byte-takeover
/// rules.
@MainActor
@Suite("DownloadStrip — byte-aware caption formatting")
struct DownloadStripByteCaptionTests {

    @Test("Bytes subtitle wins over .fetching files-only copy")
    func bytesWinOverFetching() {
        let caption = DownloadStrip.detail(
            phase: .fetching(done: 0, total: 9, percent: 0),
            bytesSubtitle: "1.2 / 6.8 GB · 18%"
        )
        #expect(caption == "1.2 / 6.8 GB · 18%")
    }

    @Test("Bytes subtitle augments .downloading per-file copy (file name still surfaces)")
    func bytesAugmentDownloading() {
        let caption = DownloadStrip.detail(
            phase: .downloading(
                file: "model-00001-of-00006.safetensors",
                done: "2.10G",
                total: "5.13G",
                percent: 41,
                speed: "23.4MB/s",
                eta: "02:09"
            ),
            bytesSubtitle: "3.2 / 6.8 GB · 47%"
        )
        // Bytes lead, the existing per-file detail stays so the user
        // still sees which shard is mid-transfer.
        #expect(caption.hasPrefix("3.2 / 6.8 GB · 47%"))
        #expect(caption.contains("model-00001-of-00006.safetensors"))
    }

    @Test("Empty bytes subtitle is treated as missing — phase-only copy stays in charge")
    func emptyBytesSubtitleFallsThrough() {
        let caption = DownloadStrip.detail(
            phase: .fetching(done: 0, total: 9, percent: 0),
            bytesSubtitle: ""
        )
        #expect(caption == "0% · 0/9 files")
    }
}

/// Growth-baseline truth table (2026-07 progress redesign): the word
/// "Downloading" is only allowed on screen when bytes provably move —
/// disk count rising above the pre-spawn baseline, or a per-file tqdm
/// line. The pill's byte read-out is gone (it is a summary now; the
/// chat startup banner owns the detail), so these tests pin the
/// DownloadProgress signal the surfaces key off instead.
@MainActor
@Suite("DownloadProgress.startupActivity — growth baseline")
struct GrowthDetectionTests {

    @Test("cached model: full weights at baseline, equal observation → loading, never downloading")
    func cachedModelReadsLoading() {
        let progress = DownloadProgress()
        let full: Int64 = 5_600_000_000
        progress.seedDiskBaseline(bytes: full)
        progress.applyDiskObservation(bytes: full, at: Date())
        #expect(progress.hasObservedGrowth == false)
        #expect(progress.startupActivity == .loading)
    }

    @Test("fresh pull: baseline 0, first positive observation → downloading")
    func freshPullReadsDownloading() {
        let progress = DownloadProgress()
        progress.seedDiskBaseline(bytes: 0)
        progress.applyDiskObservation(bytes: 120_000_000, at: Date())
        #expect(progress.hasObservedGrowth == true)
        #expect(progress.startupActivity == .downloading)
    }

    @Test("resumed partial: pre-existing bytes read as loading until growth, then flip")
    func resumedPartialFlipsOnGrowth() {
        let progress = DownloadProgress()
        let partial: Int64 = 2_000_000_000
        progress.seedDiskBaseline(bytes: partial)
        progress.applyDiskObservation(bytes: partial, at: Date())
        #expect(progress.startupActivity == .loading)
        progress.applyDiskObservation(
            bytes: partial + DownloadProgress.growthEpsilonBytes + 1_000_000,
            at: Date()
        )
        #expect(progress.hasObservedGrowth == true)
        #expect(progress.startupActivity == .downloading)
    }

    @Test("epsilon boundary: metadata churn below 4 MiB does not count as growth")
    func epsilonAbsorbsChurn() {
        let progress = DownloadProgress()
        progress.seedDiskBaseline(bytes: 1_000_000_000)
        progress.applyDiskObservation(
            bytes: 1_000_000_000 + DownloadProgress.growthEpsilonBytes,
            at: Date()
        )
        #expect(progress.hasObservedGrowth == false)
        #expect(progress.startupActivity == .loading)
    }

    @Test("growth is a latch: an inter-file dip does not un-download")
    func growthLatches() {
        let progress = DownloadProgress()
        progress.seedDiskBaseline(bytes: 0)
        progress.applyDiskObservation(bytes: 500_000_000, at: Date())
        #expect(progress.hasObservedGrowth == true)
        // Later observation static — the latch must hold.
        progress.applyDiskObservation(bytes: 500_000_000, at: Date())
        #expect(progress.hasObservedGrowth == true)
        #expect(progress.startupActivity == .downloading)
    }

    @Test("no explicit seed: first observation becomes the baseline, growth latches from the second")
    func lazyBaselineFromFirstObservation() {
        let progress = DownloadProgress()
        let preexisting: Int64 = 3_000_000_000
        progress.applyDiskObservation(bytes: preexisting, at: Date())
        #expect(progress.hasObservedGrowth == false, "first tick can never prove movement")
        progress.applyDiskObservation(
            bytes: preexisting + 50_000_000,
            at: Date()
        )
        #expect(progress.hasObservedGrowth == true)
    }

    @Test("per-file tqdm line latches growth even before a disk poll")
    func perFileTqdmLatches() {
        let progress = DownloadProgress()
        progress.seedDiskBaseline(bytes: 0)
        _ = progress.ingest("model-00001-of-00006.safetensors:  41%|████▏     | 2.10G/5.13G [01:28<02:09, 23.4MB/s]")
        #expect(progress.hasObservedGrowth == true)
        #expect(progress.startupActivity == .downloading)
    }

    @Test("warm-up phase outranks everything; reset clears baseline and latch")
    func warmupAndReset() {
        let progress = DownloadProgress()
        progress.seedDiskBaseline(bytes: 0)
        progress.applyDiskObservation(bytes: 500_000_000, at: Date())
        _ = progress.ingest("Warming up (compiling Metal shaders)")
        #expect(progress.startupActivity == .warmingUp)
        progress.reset()
        #expect(progress.baselineDiskBytes == nil)
        #expect(progress.hasObservedGrowth == false)
        #expect(progress.startupActivity == .starting)
    }
}
