import Foundation
import Testing
@testable import Rapid

/// Regression contract for the composer's model-load status: what each
/// lifecycle moment shows, which model it is allowed to describe, what
/// VoiceOver hears, and that none of it touches the user's draft.
@Suite("Composer load status")
@MainActor
struct ComposerLoadStatusTests {
    private let alias = "qwen3.5-4b-4bit"
    private let other = "gemma-4-26b-4bit"
    private let gib: Int64 = 1 << 30

    private func resolve(
        _ serverState: ServerState,
        selected: String? = nil,
        cache: ModelReadiness.CacheState = .onDisk,
        progress: ModelReadiness.ProgressSnapshot? = nil,
        downloadInFlight: Bool = false
    ) -> (ModelReadiness, ComposerLoadStatus?) {
        let readiness = ModelReadiness.resolve(
            serverState: serverState,
            alias: selected ?? alias,
            cacheState: cache,
            progress: progress,
            downloadInFlight: downloadInFlight
        )
        return (readiness, ComposerLoadStatus.resolve(readiness: readiness, progress: progress))
    }

    // MARK: - Downloading

    @Test("A download shows the real percent, bytes of total, and the stable ETA")
    func determinateDownload() throws {
        let progress = ModelReadiness.ProgressSnapshot(
            activity: .downloading, fraction: 0.594,
            bytesDownloaded: gib * 17 / 10, totalBytes: gib * 29 / 10,
            etaText: "< 1 min left"
        )
        let (_, status) = resolve(.starting(alias: alias), progress: progress)
        let s = try #require(status)
        #expect(s.phase == .downloading)
        #expect(s.progress == .determinate(0.594))
        #expect(s.percentText == "59%")
        #expect(s.detail == "\(ModelDisplayName.name(for: alias)) · \(DownloadProgress.formatBytes(gib * 17 / 10)) of \(DownloadProgress.formatBytes(gib * 29 / 10))")
        #expect(s.etaText == "< 1 min left")
    }

    @Test("No ETA is shown until the monitor offers a stable one")
    func etaWithheldUntilStable() throws {
        let progress = ModelReadiness.ProgressSnapshot(
            activity: .downloading, fraction: 0.05,
            bytesDownloaded: gib / 10, totalBytes: gib * 2, etaText: nil
        )
        let s = try #require(resolve(.starting(alias: alias), progress: progress).1)
        #expect(s.etaText == nil)
        #expect(s.percentText == "5%")
    }

    @Test("Bytes without a total say what arrived and claim no percentage")
    func unknownTotalIsIndeterminate() throws {
        let progress = ModelReadiness.ProgressSnapshot(
            activity: .downloading, bytesDownloaded: gib * 29 / 10
        )
        let s = try #require(resolve(.starting(alias: alias), progress: progress).1)
        #expect(s.progress == .indeterminate)
        #expect(s.percentText == nil)
        #expect(s.detail == "\(ModelDisplayName.name(for: alias)) · \(DownloadProgress.formatBytes(gib * 29 / 10)) downloaded")
    }

    @Test("Before the disk is observed the slot names the model, never 0 B")
    func noBytesYet() throws {
        let progress = ModelReadiness.ProgressSnapshot(activity: .downloading, bytesDownloaded: 0)
        let s = try #require(resolve(.starting(alias: alias), progress: progress).1)
        #expect(s.detail == ModelDisplayName.name(for: alias))
    }

    @Test("A byte count past an under-estimated total falls back to 'downloaded'")
    func overtakenEstimate() {
        let text = ComposerLoadStatus.bytesText(.init(
            activity: .downloading, bytesDownloaded: gib * 3, totalBytes: gib * 2
        ))
        #expect(text == "\(DownloadProgress.formatBytes(gib * 3)) downloaded")
    }

    @Test("Percent clamps and rounds")
    func percentClamps() {
        #expect(ComposerLoadStatus.percentText(for: nil) == nil)
        #expect(ComposerLoadStatus.percentText(for: 0) == "0%")
        #expect(ComposerLoadStatus.percentText(for: 1.04) == "100%")
        #expect(ComposerLoadStatus.percentText(for: -0.2) == "0%")
    }

    @Test("Non-finite download progress stays indeterminate", arguments: [Double.nan, .infinity, -.infinity])
    func nonFiniteProgress(fraction: Double) throws {
        #expect(ComposerLoadStatus.percentText(for: fraction) == nil)
        let status = try #require(ComposerLoadStatus.resolve(
            readiness: .downloading(alias: alias, detail: nil, fraction: fraction), progress: nil))
        #expect(status.progress == .indeterminate)
        #expect(status.percentText == nil)
    }

    @Test("A download-only job shows progress while the server stays idle")
    func downloadOnlyJob() throws {
        let progress = ModelReadiness.ProgressSnapshot(
            activity: .downloading, fraction: 0.3,
            bytesDownloaded: gib, totalBytes: gib * 3
        )
        let s = try #require(
            resolve(.idle, cache: .notOnDisk, progress: progress, downloadInFlight: true).1
        )
        #expect(s.phase == .downloading)
        #expect(s.percentText == "30%")
    }

    // MARK: - Loading

    /// The cached-start case: the byte monitor sees the full weights on its
    /// first poll and reports ~1.0. That must never surface as "100%" while
    /// the model is actually loading into memory.
    @Test("Loading never reuses the download percentage")
    func loadingIsIndeterminate() throws {
        let progress = ModelReadiness.ProgressSnapshot(
            activity: .loading, subtitle: "2.9 GB / 2.9 GB · 100%", fraction: 1.0,
            bytesDownloaded: gib * 29 / 10, totalBytes: gib * 29 / 10, etaText: "< 1 min left"
        )
        let s = try #require(resolve(.starting(alias: alias), progress: progress).1)
        #expect(s.phase == .loading)
        #expect(s.progress == .indeterminate)
        #expect(s.percentText == nil)
        #expect(s.etaText == nil)
        #expect(s.detail == ModelDisplayName.name(for: alias))
        #expect(!s.accessibilityLabel.contains("%"))
    }

    @Test("Each loading step has its own plain title")
    func loadingTitles() throws {
        let titles = try [DownloadProgress.StartupActivity.starting, .loading, .warmingUp].map {
            try #require(resolve(.starting(alias: alias), progress: .init(activity: $0)).1).title
        }
        #expect(titles == ["Starting the model…", "Loading the model into memory…", "Warming up…"])
    }

    @Test("Download hands over to loading in place")
    func downloadThenLoad() throws {
        let downloading = try #require(resolve(
            .starting(alias: alias),
            progress: .init(activity: .downloading, fraction: 0.99,
                            bytesDownloaded: gib * 2, totalBytes: gib * 2)
        ).1)
        let loading = try #require(resolve(
            .starting(alias: alias),
            progress: .init(activity: .warmingUp, fraction: 1.0)
        ).1)
        #expect(downloading.phase == .downloading)
        #expect(loading.phase == .loading)
        #expect(loading.percentText == nil)
    }

    // MARK: - Model binding

    /// Switching from A to B while A is still starting: A's progress must
    /// not describe B. B shows its own decision state instead, which is not
    /// a composer-slot state at all.
    @Test("An old model's start cannot drive the new selection's slot")
    func oldModelProgressDoesNotLeak() {
        let progress = ModelReadiness.ProgressSnapshot(
            activity: .downloading, fraction: 0.8, bytesDownloaded: gib, totalBytes: gib
        )
        let (readiness, status) = resolve(.starting(alias: other), progress: progress)
        #expect(readiness == .needsStart(alias: alias))
        #expect(status == nil)
    }

    @Test("A model already serving does not mark a new selection ready")
    func readyBelongsToTheServingModel() {
        let (readiness, status) = resolve(.ready(alias: other), cache: .notOnDisk)
        #expect(!readiness.sendAllowed)
        #expect(status == nil)
    }

    @Test("A crash belongs to the model that crashed")
    func crashDoesNotFollowTheSelection() {
        let (_, status) = resolve(.crashed(alias: other, message: "boom"))
        #expect(status == nil)
    }

    // MARK: - Failure and recovery

    @Test("A failed start shows its message and keeps the recovery action")
    func failedStart() throws {
        let (readiness, status) = resolve(.crashed(alias: alias, message: "Metal OOM"))
        let s = try #require(status)
        #expect(s.isFailure)
        #expect(s.progress == .none)
        #expect(s.title == readiness.headline)
        #expect(s.detail == readiness.detail)
        #expect(s.action == .retry(alias: alias))
    }

    @Test("Retry leaves the failure for loading — never an endless spinner over an error")
    func retryReentersLoading() throws {
        let failed = try #require(resolve(.crashed(alias: alias, message: "boom")).1)
        let retrying = try #require(resolve(.starting(alias: alias), progress: .init(activity: .starting)).1)
        let (ready, readyStatus) = resolve(.ready(alias: alias))
        #expect(failed.phase == .failed)
        #expect(retrying.phase == .loading)
        #expect(ready.sendAllowed)
        #expect(readyStatus == nil)
    }

    // MARK: - VoiceOver

    @Test("Percentage and byte ticks are silent")
    func ticksAreSilent() {
        let a = ModelReadiness.downloading(alias: alias, detail: "1 GB", fraction: 0.10)
        let b = ModelReadiness.downloading(alias: alias, detail: "2 GB", fraction: 0.20)
        #expect(ComposerLoadStatus.announcement(from: a, to: b) == nil)
        let c = ModelReadiness.starting(alias: alias, detail: "Loading the model into memory…")
        let d = ModelReadiness.starting(alias: alias, detail: "Warming up…")
        #expect(ComposerLoadStatus.announcement(from: c, to: d) == nil)
    }

    @Test("Phase changes are announced once each")
    func phaseChangesAnnounce() {
        let needsStart = ModelReadiness.needsStart(alias: alias)
        let downloading = ModelReadiness.downloading(alias: alias, detail: nil, fraction: 0.1)
        let starting = ModelReadiness.starting(alias: alias, detail: nil)
        let ready = ModelReadiness.ready(alias: alias)
        #expect(ComposerLoadStatus.announcement(from: needsStart, to: downloading) == downloading.headline)
        #expect(ComposerLoadStatus.announcement(from: downloading, to: starting) == starting.headline)
        #expect(ComposerLoadStatus.announcement(from: starting, to: ready) == ready.headline)
    }

    @Test("Stop is announced once even when cancellation status arrives after readiness")
    func stopAnnounced() {
        let running = ComposerLoadStatus.AnnouncementState(
            readiness: .downloading(alias: alias, detail: nil, fraction: 0.5), downloadStatus: .running)
        let decision = ComposerLoadStatus.AnnouncementState(
            readiness: .needsDownload(alias: alias, sizeText: nil), downloadStatus: .running)
        let stopped = ComposerLoadStatus.AnnouncementState(
            readiness: decision.readiness, downloadStatus: .cancelled)
        let cancelledFirst = ComposerLoadStatus.AnnouncementState(
            readiness: running.readiness, downloadStatus: .cancelled)
        #expect(ComposerLoadStatus.announcement(from: running, to: cancelledFirst) == "Download stopped")
        #expect(ComposerLoadStatus.announcement(from: cancelledFirst, to: stopped) == nil)
        #expect(ComposerLoadStatus.announcement(from: running, to: stopped) == "Download stopped")
        #expect(ComposerLoadStatus.announcement(from: running, to: decision) == nil)
        #expect(ComposerLoadStatus.announcement(from: decision, to: stopped) == "Download stopped")
        #expect(ComposerLoadStatus.announcement(from: stopped, to: stopped) == nil)
        #expect(ComposerLoadStatus.announcement(from: stopped, to: running) == running.readiness.headline)
    }

    @Test("Selecting a model that is already serving is not announced as ready")
    func readyWithoutWaitIsSilent() {
        #expect(ComposerLoadStatus.announcement(
            from: .needsStart(alias: other), to: .ready(alias: alias)
        ) == nil)
    }

    @Test("A failure is announced with its message, and only once")
    func failureAnnounced() {
        let starting = ModelReadiness.starting(alias: alias, detail: nil)
        let failed = ModelReadiness.failed(alias: alias, message: "Out of memory", action: .retry(alias: alias))
        #expect(ComposerLoadStatus.announcement(from: starting, to: failed) == failed.accessibilityLabel)
        #expect(ComposerLoadStatus.announcement(from: failed, to: failed) == nil)
    }

    @Test("Switching to a different model's load is announced")
    func switchingModelsAnnounces() {
        let a = ModelReadiness.starting(alias: other, detail: nil)
        let b = ModelReadiness.starting(alias: alias, detail: nil)
        #expect(ComposerLoadStatus.announcement(from: a, to: b) == b.headline)
    }

    // MARK: - Draft and copy

    @Test("Leaving a working phase for a decision state stays silent")
    func decisionAfterWaitIsSilent() {
        #expect(ComposerLoadStatus.announcement(
            from: .starting(alias: alias, detail: nil),
            to: .needsStart(alias: alias)
        ) == nil)
        #expect(ComposerLoadStatus.announcement(
            from: .downloading(alias: alias, detail: nil, fraction: nil),
            to: .needsDownload(alias: alias, sizeText: nil)
        ) == nil)
    }

    /// Readiness never writes the draft. A gated submit during a load keeps
    /// every character, and the same draft goes out once the model is ready.
    @Test("The draft survives loading and is sent unchanged when ready")
    func draftSurvivesLoading() {
        let chat = ChatViewModel(persistsConversations: false)
        chat.composerDraft = "Summarise the attached notes"
        var sent: [String] = []
        for readiness in [
            ModelReadiness.downloading(alias: alias, detail: nil, fraction: 0.4),
            .starting(alias: alias, detail: nil),
            .failed(alias: alias, message: "boom", action: .retry(alias: alias)),
            .starting(alias: alias, detail: nil),
        ] {
            let accepted = chat.submitComposerDraft(
                hasAttachments: false, canSubmit: { readiness.sendAllowed }
            ) { sent.append($0) }
            #expect(!accepted)
            #expect(chat.composerDraft == "Summarise the attached notes")
        }
        let ready = ModelReadiness.ready(alias: alias)
        #expect(chat.submitComposerDraft(hasAttachments: false, canSubmit: { ready.sendAllowed }) {
            sent.append($0)
        })
        #expect(sent == ["Summarise the attached notes"])
    }

    @Test("While loading, the placeholder invites typing instead of repeating the status")
    func placeholderDuringLoad() {
        for state in [
            ModelReadiness.downloading(alias: alias, detail: nil, fraction: 0.5),
            .starting(alias: alias, detail: nil),
        ] {
            #expect(!state.composerPlaceholder.contains(alias))
            #expect(!state.sendAllowed)
        }
    }

    // MARK: - Hero

    /// While the composer carries the download, the load or the failure,
    /// the hero above it says nothing about them: no progress, no bytes,
    /// no error text — one neutral line under "Ask anything".
    @Test("The hero steps back while the composer slot speaks")
    func heroIsNeutralDuringComposerStates() {
        let states: [ModelReadiness] = [
            .downloading(alias: alias, detail: "1.7 GB / 2.9 GB · 59% · 8 MB/s · < 1 min left", fraction: 0.59),
            .starting(alias: alias, detail: "Loading the model into memory…"),
            .failed(alias: alias, message: "Out of memory while loading.", action: .retry(alias: alias)),
        ]
        let neutral = ChatView.heroCopy(for: states[0])
        for state in states {
            let copy = ChatView.heroCopy(for: state)
            #expect(copy.subtitle == neutral.subtitle, "\(state)")
            #expect(copy.hint == nil, "\(state)")
            #expect(!copy.subtitle.contains(alias))
            #expect(!copy.subtitle.contains("GB") && !copy.subtitle.contains("%"))
            #expect(!copy.subtitle.contains("memory"))
        }
    }

    @Test("Decision and ready states keep their hero copy")
    func heroKeepsDecisionCopy() {
        let kept: [ModelReadiness] = [
            .noModel,
            .needsDownload(alias: alias, sizeText: "~2.9 GB"),
            .needsStart(alias: alias),
            .unknownModel(alias: alias),
            .ready(alias: alias),
        ]
        for state in kept {
            let copy = ChatView.heroCopy(for: state)
            #expect(copy.subtitle == state.emptyStateSubtitle, "\(state)")
            #expect(copy.hint == state.emptyStateHint, "\(state)")
        }
        #expect(ChatView.heroCopy(for: .needsDownload(alias: alias, sizeText: "~2.9 GB")).hint != nil)
    }

    // MARK: - Rail motion

    @Test("The indeterminate segment enters before the track and leaves after it")
    func segmentSweep() {
        let width: CGFloat = 700, segment: CGFloat = 210
        let period = ComposerProgressRail.sweepPeriod
        #expect(ComposerProgressRail.segmentOffset(at: 0, trackWidth: width, segmentWidth: segment) == -segment)
        let mid = ComposerProgressRail.segmentOffset(at: period / 2, trackWidth: width, segmentWidth: segment)
        #expect(abs(mid - (width - segment) / 2) < 0.001)
        for step in 0..<20 {
            let x = ComposerProgressRail.segmentOffset(
                at: period * Double(step) / 20, trackWidth: width, segmentWidth: segment
            )
            #expect(x >= -segment && x < width)
        }
    }
}
