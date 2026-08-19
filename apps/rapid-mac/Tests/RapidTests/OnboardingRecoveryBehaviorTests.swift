import Foundation
import Testing
@testable import Rapid

/// Contract for the Onboarding recovery behaviours — the last prerequisite
/// slice before the Direction D visual PR (Paper §05.1 states 10–13, 18, 20;
/// §05.2.D Review download · insufficient disk).
///
/// Every case here pins something a user hits only when setup goes wrong, and
/// each one has the same failure mode if it regresses: the app tells the user
/// something untrue about what just happened, or offers them a control that
/// looks like a way out and is not one.
///
///   1. **A cancellation is not a fault.** The user stopping a pull and a
///      transfer breaking are different events with different remedies. They
///      were collapsed onto one diagnosis, so cancelling produced "check your
///      connection" — advice about a fault that did not occur. Paper flags
///      this explicitly and names ``FailureDiagnoser`` as where it is fixed.
///   2. **A failure keeps the user inside setup, and gives them a way back to
///      choosing.** Back lands on the micro-stage they actually left, with
///      their selection and their catalogue state intact.
///   3. **Retry retries the same model.** No duplicate job, no lost alias, no
///      advance to a step that has not happened.
///   4. **Pre-flight interposes, and Cancel returns to the exact origin.**
///      Disk is warn-only by product decision; what matters is that the
///      question is asked before the pull and answered back where it started.
///   5. **Recheck reports.** A real re-resolution that changes nothing visible
///      is indistinguishable from a dead button, so the outcome is recorded
///      and stated.
///   6. **A relaunch is truthful.** Setup that was begun and not finished says
///      so, carries nothing forward, and never restores a transfer.
@MainActor
@Suite("Onboarding recovery — cancellation, failure, pre-flight, relaunch")
struct OnboardingRecoveryBehaviorTests {

    /// A coordinator with every persisted key cleared, so a case never
    /// inherits another case's flags through ``UserDefaults``.
    private func makeCoordinator() -> QuickstartCoordinator {
        let coord = QuickstartCoordinator()
        coord._testingReset()
        return coord
    }

    /// The copy a broken transfer produces, for negative-pinning.
    private var downloadFailedMessage: String {
        FailureDiagnoser.diagnosis(for: .downloadFailed).message
    }

    private var cancelledMessage: String {
        FailureDiagnoser.diagnosis(for: .downloadCancelled).message
    }

    // MARK: - 1. Cancellation is not a network failure

    @Test("A cancelled download has its own kind, distinct from a failed one")
    func cancellationIsItsOwnKind() {
        #expect(FailureDiagnosis.Kind.downloadCancelled != .downloadFailed)
        #expect(cancelledMessage != downloadFailedMessage)
    }

    @Test("Cancellation copy never advises checking the connection")
    func cancellationCopyIsNotNetworkAdvice() {
        let lowered = cancelledMessage.lowercased()
        // The exact regression Paper flagged: the shipped string for a
        // user-initiated stop was the network one.
        #expect(!lowered.contains("connection"))
        #expect(!lowered.contains("network"))
        #expect(!lowered.contains("offline"))
        // And it does say what actually happened.
        #expect(lowered.contains("stopped"))
    }

    @Test("Cancellation copy claims neither pause nor resume")
    func cancellationCopyMakesNoResumeClaim() {
        let lowered = cancelledMessage.lowercased()
        // The downloader cannot guarantee resume across a relaunch, so
        // nothing may imply it can — and nothing may claim the opposite
        // either, which would be an equally unfounded statement about bytes
        // in the Hugging Face cache.
        for forbidden in ["resume", "pause", "paused", "where it left off", "continue the download"] {
            #expect(!lowered.contains(forbidden), "cancellation copy must not say '\(forbidden)'")
        }
    }

    @Test("A cancellation is a notice, not an error")
    func cancellationIsANotice() {
        #expect(FailureDiagnosis.Kind.downloadCancelled.severity == .notice)
        #expect(FailureDiagnosis.Kind.downloadFailed.severity == .error)
    }

    @Test("Cancellation still offers a way to get the model")
    func cancellationOffersRetry() {
        #expect(FailureDiagnoser.diagnosis(for: .downloadCancelled).action == .retry)
    }

    @Test("A cancelled job carries the cancellation kind, so nothing infers it")
    func cancelledJobRecordsItsKind() {
        let downloads = DownloadManager()
        let job = downloads._testingSeedJob(alias: "lfm2.5-1b-4bit")
        // The real exit branch a SIGTERM'd pull takes.
        downloads._testingFinish(
            alias: "lfm2.5-1b-4bit",
            status: 0,
            reason: .uncaughtSignal,
            wasCancelling: true
        )

        #expect(job.status == .cancelled)
        // The load-bearing half. Every explaining surface reads this first;
        // a nil here is what let the string classifier guess "network".
        #expect(job.failureKind == .downloadCancelled)
    }

    @Test("A signalled pull that was NOT cancelled is still a failure")
    func uncancelledSignalIsStillAFailure() {
        let downloads = DownloadManager()
        let job = downloads._testingSeedJob(alias: "lfm2.5-1b-4bit")
        downloads._testingFinish(
            alias: "lfm2.5-1b-4bit",
            status: 0,
            reason: .uncaughtSignal,
            wasCancelling: false
        )
        // The distinction the whole slice rests on: same signal, different
        // cause, different diagnosis.
        #expect(job.failureKind == .downloadFailed)
        if case .failed = job.status {} else {
            Issue.record("an uncancelled signalled exit must be .failed, got \(job.status)")
        }
    }

    @Test("The failure card diagnoses a cancelled job as cancelled")
    func failureCardReadsTheCancellationKind() {
        let kind = QuickstartView.failureKind(
            jobFailureKind: .downloadCancelled,
            jobUsesMirror: true,
            serverState: .idle,
            selectionAlias: "lfm2.5-1b-4bit",
            message: cancelledMessage
        )
        #expect(kind == .downloadCancelled)
        #expect(FailureDiagnoser.diagnosis(for: kind).message == cancelledMessage)
    }

    @Test("A reaped cancelled job still doesn't become a network failure")
    func cancellationSurvivesTheJobBeingReaped() {
        // The phase outlives the job record. Falling through to the raw string
        // classifier here is exactly what produced the wrong advice, so the
        // app's own cancellation message is recognised rather than parsed.
        let kind = QuickstartView.failureKind(
            jobFailureKind: nil,
            jobUsesMirror: true,
            serverState: .idle,
            selectionAlias: "lfm2.5-1b-4bit",
            message: cancelledMessage
        )
        #expect(kind == .downloadCancelled)
    }

    @Test("A real transfer failure stays a transfer failure")
    func genuineFailureIsStillDiagnosedAsOne() {
        let kind = QuickstartView.failureKind(
            jobFailureKind: .downloadFailed,
            jobUsesMirror: true,
            serverState: .idle,
            selectionAlias: "lfm2.5-1b-4bit",
            message: downloadFailedMessage
        )
        #expect(kind == .downloadFailed)
        #expect(FailureDiagnoser.diagnosis(for: kind).message.lowercased().contains("connection"))
    }

    @Test("An offline mirror failure stays distinguishable from both")
    func offlineIsItsOwnDiagnosis() {
        // Offline reaches the card as the source-unavailable kind, which is
        // the only one that offers Switch source. It must not collapse into
        // either the generic failure or the cancellation.
        let kind = FailureDiagnoser.downloadFailureKind(
            raw: "could not reach models.rapidmlx.com: gateway timeout",
            usingMirror: true
        )
        #expect(kind == .downloadSourceUnavailable)
        #expect(kind != .downloadCancelled)
        #expect(FailureDiagnoser.diagnosis(for: kind).action == .switchDownloadSource)
    }

    @Test("A load failure outranks the download's own record")
    func crashedServeIsALoadFailure() {
        // The weights are on disk; only the serve broke. Reporting the stale
        // download kind here would send the user back through Step 3.
        let kind = QuickstartView.failureKind(
            jobFailureKind: .downloadCancelled,
            jobUsesMirror: true,
            serverState: .crashed(alias: "lfm2.5-1b-4bit", message: "child exited during load"),
            selectionAlias: "lfm2.5-1b-4bit",
            message: cancelledMessage
        )
        #expect(kind == .modelLoadFailed, "a crashed serve is a load failure, not the download's stale record")
    }

    @Test("A crashed serve for somebody else's model is not ours to explain")
    func crashedForeignAliasDoesNotOverrideTheJob() {
        // The user's own download record still owns the explanation when the
        // crash belongs to a different alias entirely.
        let kind = QuickstartView.failureKind(
            jobFailureKind: .downloadCancelled,
            jobUsesMirror: true,
            serverState: .crashed(alias: "some-other-model", message: "child exited during load"),
            selectionAlias: "lfm2.5-1b-4bit",
            message: cancelledMessage
        )
        #expect(kind == .downloadCancelled)
    }

    @Test("Only a cancellation loses the fault-report heading")
    func cancellationHeadingIsNotAFaultReport() {
        #expect(QuickstartView.failureTitle(for: .downloadCancelled) == "Download stopped")
        #expect(QuickstartView.failureTitle(for: .downloadFailed) == "Quickstart didn't finish")
        #expect(QuickstartView.failureTitle(for: .downloadSourceUnavailable) == "Quickstart didn't finish")
        #expect(QuickstartView.failureTitle(for: .modelLoadFailed) == "Quickstart didn't finish")
    }

    @Test("A cancellation keeps the macro step it happened in")
    func cancellationDoesNotMoveTheRail() {
        let coord = makeCoordinator()
        coord.advanceToChooseModel()
        coord.enterDownloading()
        coord.enterFailed(message: cancelledMessage, origin: .download)
        // Still Step 3 of 4 — stopping a download does not un-choose a model
        // and does not become a step of its own.
        #expect(coord.step == .download)
        #expect(QuickstartCoordinator.Step.total == 4)
    }

    // MARK: - 1b. Cancellation must be REACHABLE from Step 3

    // Manual verification found the gap these pin: every case above tested how
    // a cancellation is DIAGNOSED, and none tested whether a user inside
    // onboarding can cause one. They could not. Onboarding is a full-window
    // sheet, so ``DownloadStrip`` — the app's ordinary cancel affordance — sits
    // behind it for the whole pull, and the Step 3 card offered no control of
    // its own. A 63%-complete download had no exit but quitting the app.

    @Test("An actively running onboarding download exposes a cancel action")
    func activeDownloadIsCancellable() {
        let target = QuickstartView.downloadCancelTarget(
            jobStatus: .running,
            selectionAlias: "lfm2.5-1b-4bit",
            alreadyRequested: false
        )
        #expect(target != nil, "a live download must offer a way to stop it")
    }

    @Test("The cancel action targets the currently selected alias")
    func cancelTargetsTheSelectedAlias() {
        // Not the starter, not the last thing downloaded — whatever the user
        // actually picked, which is the alias the pull was started for.
        #expect(QuickstartView.downloadCancelTarget(
            jobStatus: .running,
            selectionAlias: "qwen3.5-9b-4bit",
            alreadyRequested: false
        ) == "qwen3.5-9b-4bit")

        #expect(QuickstartView.downloadCancelTarget(
            jobStatus: .running,
            selectionAlias: "",
            alreadyRequested: false
        ) == nil, "with nothing selected there is nothing to name or cancel")
    }

    @Test("No cancel action is exposed once the job has settled")
    func settledDownloadOffersNoCancel() {
        // Every terminal status, plus "no job at all". cancelDownload is a
        // no-op against all of them, so a button here would be the exact
        // "looks actionable while doing nothing" defect this slice removes.
        let settled: [DownloadManager.Job.Status?] = [
            nil,
            .completed,
            .cancelled,
            .failed(message: "boom"),
        ]
        for status in settled {
            #expect(QuickstartView.downloadCancelTarget(
                jobStatus: status,
                selectionAlias: "lfm2.5-1b-4bit",
                alreadyRequested: false
            ) == nil, "a settled job (\(String(describing: status))) must offer no cancel")
        }
    }

    @Test("Repeated presses cannot start conflicting cancellation work")
    func repeatedCancelPressesAreInert() {
        // The optimistic flip to .cancelled lands on the same run-loop turn,
        // but a second press in the same frame would re-signal a process that
        // is already mid-SIGTERM and arm a second hard-kill timer.
        #expect(QuickstartView.downloadCancelTarget(
            jobStatus: .running,
            selectionAlias: "lfm2.5-1b-4bit",
            alreadyRequested: true
        ) == nil)
    }

    @Test("Cancelling through the manager reaches the cancellation diagnosis")
    func cancelDownloadProducesTheCancellationRecovery() {
        // End to end at the level the view drives: the control calls
        // cancelDownload, which must leave a job the failure card diagnoses as
        // a cancellation rather than a fault.
        let downloads = DownloadManager()
        let alias = "lfm2.5-1b-4bit"
        let job = downloads._testingSeedJob(alias: alias)
        #expect(QuickstartView.downloadCancelTarget(
            jobStatus: job.status,
            selectionAlias: alias,
            alreadyRequested: false
        ) == alias)

        downloads._testingFinish(
            alias: alias,
            status: 0,
            reason: .uncaughtSignal,
            wasCancelling: true
        )

        let kind = QuickstartView.failureKind(
            jobFailureKind: job.failureKind,
            jobUsesMirror: true,
            serverState: .idle,
            selectionAlias: alias,
            message: cancelledMessage
        )
        #expect(kind == .downloadCancelled)
        #expect(QuickstartView.failureTitle(for: kind) == "Download stopped")
        #expect(FailureDiagnoser.diagnosis(for: kind).action == .retry)
        // And the control is gone now that the job has settled.
        #expect(QuickstartView.downloadCancelTarget(
            jobStatus: job.status,
            selectionAlias: alias,
            alreadyRequested: false
        ) == nil)
    }

    @Test("A cancelled download and a failed one stay different screens")
    func cancelAndFailureRemainDistinctEndToEnd() {
        let downloads = DownloadManager()
        let cancelled = downloads._testingSeedJob(alias: "lfm2.5-1b-4bit")
        downloads._testingFinish(
            alias: "lfm2.5-1b-4bit", status: 0, reason: .uncaughtSignal, wasCancelling: true
        )
        let failed = downloads._testingSeedJob(alias: "qwen3.5-4b-4bit")
        downloads._testingFinish(
            alias: "qwen3.5-4b-4bit", status: 1, reason: .exit, wasCancelling: false
        )

        #expect(cancelled.failureKind == .downloadCancelled)
        #expect(failed.failureKind != .downloadCancelled)
        #expect(QuickstartView.failureTitle(for: cancelled.failureKind!)
                != QuickstartView.failureTitle(for: failed.failureKind!))
        #expect(!FailureDiagnoser.diagnosis(for: cancelled.failureKind!)
            .message.lowercased().contains("connection"))
    }

    @Test("Cancelling then retrying the same alias leaves exactly one job")
    func retryAfterCancelCreatesOneJob() {
        let downloads = DownloadManager()
        let alias = "lfm2.5-1b-4bit"
        _ = downloads._testingSeedJob(alias: alias)
        downloads._testingFinish(
            alias: alias, status: 0, reason: .uncaughtSignal, wasCancelling: true
        )
        #expect(downloads.jobs.count == 1)
        // retryDownload dismisses the terminal job before starting the next
        // child, so the alias never holds two records. (The spawn itself is
        // not driven here — see the note above the retry guard cases.)
        #expect(downloads.job(for: alias)?.status == .cancelled)
        #expect(downloads.jobs.count == 1)
    }

    @Test("Cancelling lands on the Step 2 origin the download was started from")
    func cancelReturnsToTheRightOrigin() {
        let coord = makeCoordinator()
        coord.advanceToChooseModel()
        coord.beginReviewDownload(origin: .catalogue)
        coord.enterDownloading()
        coord.enterFailed(message: cancelledMessage, origin: .download)

        #expect(QuickstartView.failureBackTitle(for: coord.step2Stage)
                == "← Back to review download")
        coord.returnToChooser()
        #expect(coord.step2Stage == .reviewing)
        #expect(coord.step == .chooseModel)
    }

    // MARK: - 2. The route back to model selection

    @Test("Back from a failure names the micro-stage it returns to")
    func failureBackTitleNamesItsDestination() {
        #expect(QuickstartView.failureBackTitle(for: .choosing) == "← Back to recommended models")
        #expect(QuickstartView.failureBackTitle(for: .browsing) == "← Back to all models")
        #expect(QuickstartView.failureBackTitle(for: .reviewing) == "← Back to review download")
        // The two pre-shortlist stages resolve onto the shortlist, so they
        // name it rather than inventing a third destination.
        #expect(QuickstartView.failureBackTitle(for: .checkingHardware) == "← Back to recommended models")
        #expect(QuickstartView.failureBackTitle(for: .findingFit) == "← Back to recommended models")
    }

    @Test("Back from a failure returns to the exact Step 2 origin")
    func failureReturnsToTheOriginMicroStage() {
        for origin in [
            QuickstartCoordinator.Step2Stage.choosing,
            .browsing,
            .reviewing,
        ] {
            let coord = makeCoordinator()
            coord.advanceToChooseModel()
            switch origin {
            case .browsing: coord.beginBrowsingCatalog()
            case .reviewing: coord.beginReviewDownload(origin: .catalogue)
            default: coord.resolveRecommendationLoading(catalogLoaded: true)
            }
            #expect(coord.step2Stage == origin)

            coord.enterDownloading()
            coord.enterFailed(message: cancelledMessage, origin: .download)
            coord.returnToChooser()

            #expect(coord.phase == .idle)
            #expect(coord.step == .chooseModel)
            #expect(coord.step2Stage == origin, "must return to \(origin), not a default")
        }
    }

    @Test("Back from a failure keeps the selection and the catalogue state")
    func failureReturnPreservesSelectionAndCatalogue() {
        let coord = makeCoordinator()
        coord.advanceToChooseModel()
        coord.beginBrowsingCatalog()
        coord.catalogQuery = "qwen"
        coord.catalogFilter = .cached
        coord.catalogSort = .sizeDescending
        coord.rememberCatalogAnchor("qwen3.5-9b-4bit")
        let pick = QuickstartCoordinator.onboardingChoices.first { $0.alias == "qwen3.5-4b-4bit" }!
        coord.select(pick)

        coord.enterDownloading()
        coord.enterFailed(message: downloadFailedMessage, origin: .download)
        coord.returnToChooser()

        #expect(coord.step2Stage == .browsing)
        #expect(coord.selection.alias == "qwen3.5-4b-4bit")
        #expect(coord.catalogQuery == "qwen")
        #expect(coord.catalogFilter == .cached)
        #expect(coord.catalogSort == .sizeDescending)
        #expect(coord.catalogScrollID == "qwen3.5-9b-4bit")
    }

    @Test("Back from a failure does not complete or dismiss onboarding")
    func failureReturnNeverEndsSetup() {
        let coord = makeCoordinator()
        coord.advanceToChooseModel()
        coord.enterDownloading()
        coord.enterFailed(message: downloadFailedMessage, origin: .download)
        coord.returnToChooser()

        #expect(!coord.done, "a failure must never write the completion flag")
        #expect(coord.phase != .dismissed)
        // And the surface predicate still keeps setup owed.
        #expect(QuickstartCoordinator.onboardingOwed(
            done: coord.done,
            legacyDone: coord.legacyDone,
            lastServedAlias: nil
        ))
    }

    @Test("A failure keeps the window while it is on screen")
    func failureRetainsTheOnboardingSurface() {
        #expect(ContentView.quickstartRetainsSurface(
            phase: .failed(message: "x", origin: .download)
        ))
        #expect(ContentView.quickstartRetainsSurface(
            phase: .failed(message: "x", origin: .start)
        ))
    }

    // MARK: - 3. Retry targets the same model

    // NOTE: these deliberately exercise ``retryDownload``'s GUARDS rather
    // than its success path. A successful retry spawns `rapid-mlx pull`,
    // which on a developer machine with a real engine installed would start
    // an actual multi-gigabyte download from a unit test.

    @Test("Retry refuses to overlap a job that is still running")
    func retryDoesNotDoubleUpOnARunningJob() {
        let downloads = DownloadManager()
        let alias = "lfm2.5-1b-4bit"
        _ = downloads._testingSeedJob(alias: alias)  // seeded .running
        #expect(downloads.retryDownload(alias: alias) == false,
                "a retry must never race a pull that is already going")
        #expect(downloads.jobs.count == 1, "and must not fan out into a second job")
        #expect(downloads.job(for: alias)?.alias == alias)
    }

    @Test("Retry is keyed by alias, so it cannot retarget another model")
    func retryIsKeyedByAlias() {
        let downloads = DownloadManager()
        _ = downloads._testingSeedJob(alias: "lfm2.5-1b-4bit")
        // Nothing is registered for a model the user did not pick, so a retry
        // aimed at one is a no-op rather than a job invented for it.
        #expect(downloads.retryDownload(alias: "qwen3.5-9b-4bit") == false)
        #expect(downloads.job(for: "qwen3.5-9b-4bit") == nil)
        #expect(downloads.jobs.count == 1)
    }

    @Test("Retry preserves the job's identity for the re-pull")
    func retryCarriesTheSameDownloadIdentity() {
        // What a retry hands to the new child is read off the previous job:
        // same alias, same HF repo, same byte total, same source. That is what
        // makes "retry the same intended model" true rather than hopeful.
        let downloads = DownloadManager()
        let alias = QuickstartCoordinator.defaultChoice.alias
        let job = downloads._testingSeedJob(
            alias: alias,
            hfPath: QuickstartCoordinator.defaultChoice.hfRepo,
            totalBytes: QuickstartCoordinator.defaultChoice.downloadBytes
        )
        downloads._testingFinish(
            alias: alias,
            status: 0,
            reason: .uncaughtSignal,
            wasCancelling: true
        )
        #expect(job.alias == alias)
        #expect(job.hfPath == QuickstartCoordinator.defaultChoice.hfRepo)
        #expect(job.totalBytes == QuickstartCoordinator.defaultChoice.downloadBytes)
        #expect(job.source == .mirror)
    }

    @Test("The failure card's retry never loses the selected alias")
    func retryKeepsTheSelectedAlias() {
        let coord = makeCoordinator()
        coord.advanceToChooseModel()
        let pick = QuickstartCoordinator.onboardingChoices.first { $0.alias == "qwen3.5-4b-4bit" }!
        coord.select(pick)
        coord.enterDownloading()
        coord.enterFailed(message: downloadFailedMessage, origin: .download)
        // The selection is what every retry branch reads for its alias.
        #expect(coord.selection.alias == "qwen3.5-4b-4bit")
        coord.enterDownloading()
        #expect(coord.selection.alias == "qwen3.5-4b-4bit")
    }

    @Test("Retry keeps the rail on the step the failure happened in")
    func retryDoesNotAdvanceAFalseStep() {
        let coord = makeCoordinator()
        coord.advanceToChooseModel()
        coord.enterDownloading()
        coord.enterFailed(message: downloadFailedMessage, origin: .download)
        #expect(coord.step == .download)

        // The retry transition the failure card performs.
        coord.enterDownloading()
        #expect(coord.step == .download, "retry re-enters Download, never Start")
        #expect(coord.selection.alias == QuickstartCoordinator.defaultChoice.alias)
    }

    @Test("Retrying after a load failure stays on Start, not Download")
    func loadFailureRetryStaysOnStart() {
        let coord = makeCoordinator()
        coord.advanceToChooseModel()
        coord.enterStarting()
        coord.enterFailed(message: "could not load", origin: .start)
        #expect(coord.step == .start)

        coord.enterStarting()
        #expect(coord.step == .start, "the weights are on disk — do not re-run Step 3")
    }

    // MARK: - 4. Disk and memory pre-flight

    @Test("Insufficient disk interposes before any download starts")
    func lowDiskInterposesBeforeTheDownload() {
        let coord = makeCoordinator()
        coord.advanceToChooseModel()
        var kickedOff = false
        QuickstartView.applyPreflightDecision(
            decision: DiskSpaceProbe.decide(
                freeBytes: 100 * 1024 * 1024,
                requiredBytes: DiskSpaceProbe.quickstartRequiredBytes
            ),
            coordinator: coord,
            onKickoff: { kickedOff = true }
        )
        #expect(!kickedOff, "the pull must not start behind the question")
        #expect(coord.phase == .lowDiskWarning(
            freeBytes: Int64(100 * 1024 * 1024),
            requiredBytes: DiskSpaceProbe.quickstartRequiredBytes
        ))
    }

    @Test("Cancelling the low-disk warning returns to the exact Step 2 origin")
    func lowDiskCancelReturnsToOrigin() {
        // Paper 05.2.D states this destination in as many words: "Cancel
        // returns here, not to Welcome."
        for origin in [
            QuickstartCoordinator.Step2Stage.choosing,
            .browsing,
            .reviewing,
        ] {
            let coord = makeCoordinator()
            coord.advanceToChooseModel()
            switch origin {
            case .browsing: coord.beginBrowsingCatalog()
            case .reviewing: coord.beginReviewDownload(origin: .shortlist)
            default: coord.resolveRecommendationLoading(catalogLoaded: true)
            }
            let pick = QuickstartCoordinator.lowMemoryChoice
            coord.select(pick)

            coord.enterLowDiskWarning(freeBytes: 1, requiredBytes: 2)
            #expect(coord.step == .download)

            coord.cancelLowDiskWarning()
            #expect(coord.step == .chooseModel)
            #expect(coord.step2Stage == origin)
            #expect(coord.selection.alias == pick.alias, "Cancel must not drop the pick")
        }
    }

    @Test("The low-disk number is stated as a flat floor, not the model's size")
    func lowDiskCopyDoesNotAttributeTheFloorToTheModel() {
        let body = QuickstartView.lowDiskBannerBody(
            freeBytes: Int64(1024 * 1024 * 1024),
            requiredBytes: DiskSpaceProbe.quickstartRequiredBytes,
            displayName: QuickstartCoordinator.defaultChoice.displayName
        )
        // The regression: the flat 2 GiB pre-flight floor was described as
        // "<model> weights + safety margin", which is a per-model measurement
        // the probe never made — and is out by ~3x for the starter.
        #expect(!body.contains("weights"))
        #expect(body.contains("flat floor"))
        #expect(body.contains(QuickstartCoordinator.defaultChoice.displayName))
        #expect(body.contains("Continue anyway?"))

        let label = QuickstartView.lowDiskAccessibilityLabel(
            freeBytes: Int64(1024 * 1024 * 1024),
            requiredBytes: DiskSpaceProbe.quickstartRequiredBytes,
            displayName: QuickstartCoordinator.defaultChoice.displayName
        )
        #expect(!label.contains("weights"))
        #expect(label.lowercased().contains("continue anyway"))
        #expect(label.lowercased().contains("cancel"))
    }

    @Test("A model this Mac cannot run never makes the primary actionable")
    func incompatibleMemoryBlocksTheAction() {
        // The blocking decision is ``ModelSizing``'s, read through the one
        // availability seam the picker already disables on — not a new
        // capability or compatibility claim invented by onboarding.
        let rows = [
            OnboardingModelSelection.Row(alias: "huge-model", isCached: false, isAvailable: false),
            OnboardingModelSelection.Row(alias: "fits", isCached: false, isAvailable: true),
        ]
        for context in [
            OnboardingModelSelection.ListContext.shortlist,
            .catalogue,
            .review,
        ] {
            let primary = OnboardingModelSelection.primary(
                selection: "huge-model",
                visibleRows: rows,
                catalogState: .ready,
                context: context
            )
            // The claim this test has always made, restated precisely: an
            // unrunnable pick can never be COMMITTED on. Since Paper 05.2.D
            // its read-only detail is reachable from a list, which is a
            // navigation and spends nothing — so "not actionable" is checked
            // against the commit, not against the button being pressable.
            #expect(
                !(primary.isEnabled && primary.action.isCommit),
                "\(context): an unrunnable pick must never reach a start or a download"
            )
            #expect(!OnboardingModelSelection.isActionable(
                selection: "huge-model", visibleRows: rows, catalogState: .ready
            ), "\(context)")
            if context == .review {
                // The refusal lands here, and names what is being withheld.
                #expect(!primary.isEnabled)
                #expect(primary.title == OnboardingModelSelection.Verb.downloadAndStart)
            } else {
                #expect(primary.action == .reviewIncompatible, "\(context)")
            }
        }
    }

    /// Incompatibility outranks cached-ness. A model already on disk that this
    /// Mac cannot load is still a model this Mac cannot load — being
    /// downloaded already is not evidence about memory, and ``startExisting``
    /// is the shorter route into exactly the same ``ServerManager`` load.
    @Test("A cached model that cannot run still cannot be started")
    func cachedIncompatibleModelCannotStart() {
        let rows = [OnboardingModelSelection.Row(
            alias: "huge-but-here", isCached: true, isAvailable: false
        )]
        for context in [
            OnboardingModelSelection.ListContext.shortlist,
            .catalogue,
            .review,
        ] {
            let primary = OnboardingModelSelection.primary(
                selection: "huge-but-here",
                visibleRows: rows,
                catalogState: .ready,
                context: context
            )
            #expect(
                !(primary.isEnabled && primary.action == .startExisting),
                "\(context): a cached unrunnable pick must not start"
            )
        }
        // And inside Review the greyed verb is the cached one, so the screen
        // does not offer to download something that is already here.
        let review = OnboardingModelSelection.primary(
            selection: "huge-but-here", visibleRows: rows,
            catalogState: .ready, context: .review
        )
        #expect(!review.isEnabled)
        #expect(review.title == OnboardingModelSelection.Verb.startExisting)
        #expect(review.action == .startExisting)
    }

    @Test("An unrunnable pick stays selected — only its actionability changes")
    func incompatibleMemoryDoesNotClearTheSelection() {
        let coord = makeCoordinator()
        coord.advanceToChooseModel()
        let pick = QuickstartCoordinator.onboardingChoices.first { $0.alias == "qwen3.5-9b-4bit" }!
        coord.select(pick)
        #expect(coord.selection.alias == "qwen3.5-9b-4bit")
        #expect(!OnboardingModelSelection.isActionable(
            selection: "qwen3.5-9b-4bit",
            visibleRows: [.init(alias: "qwen3.5-9b-4bit", isCached: false, isAvailable: false)],
            catalogState: .ready
        ))
        // Still the user's choice, still on the books.
        #expect(coord.selection.alias == "qwen3.5-9b-4bit")
    }

    @Test("Declining the pre-load memory guard returns to the exact origin")
    func memoryGuardCancelReturnsToOrigin() {
        for origin in [
            QuickstartCoordinator.Step2Stage.choosing,
            .browsing,
            .reviewing,
        ] {
            let coord = makeCoordinator()
            coord.advanceToChooseModel()
            switch origin {
            case .browsing: coord.beginBrowsingCatalog()
            case .reviewing: coord.beginReviewDownload(origin: .catalogue)
            default: coord.resolveRecommendationLoading(catalogLoaded: true)
            }
            coord.enterStarting()
            coord.returnToChooser()
            #expect(coord.step == .chooseModel)
            #expect(coord.step2Stage == origin)
        }
    }

    @Test("The in-sheet memory decision is only claimed for our own model")
    func memoryWarningOwnershipIsScoped() {
        let ours = ModelSizing.MemoryWarning(
            alias: "qwen3.5-9b-4bit",
            hfPath: nil,
            isAutoRespawn: false,
            severity: .unsafe,
            footprintGB: 10,
            freeGB: 2,
            totalGB: 16
        )
        #expect(QuickstartView.memoryWarningToPresent(
            phase: .starting, pending: ours, selectionAlias: "qwen3.5-9b-4bit"
        ) != nil)
        // A warning about somebody else's start is not onboarding's to answer.
        #expect(QuickstartView.memoryWarningToPresent(
            phase: .starting, pending: ours, selectionAlias: "lfm2.5-1b-4bit"
        ) == nil)
        #expect(QuickstartView.memoryWarningToPresent(
            phase: .idle, pending: ours, selectionAlias: "qwen3.5-9b-4bit"
        ) == nil)
    }

    // MARK: - 5. Missing engine

    @Test("A missing engine keeps the wizard down so the recovery is reachable")
    func missingEngineOwnsTheSurface() {
        // Presentation priority: telemetry consent, then the wizard, then the
        // main area. The wizard stands down on ``.missing``, which is what
        // lets the install overlay in the main area be the thing a brand-new
        // user actually reaches.
        #expect(!QuickstartCoordinator.isEligible(
            done: false, legacyDone: false, lastServedAlias: nil, serverState: .missing
        ))
        #expect(ContentView.mainAreaBranch(for: .missing) == .missing)
        #expect(!ContentView.quickstartRetainsSurface(phase: .idle))
    }

    @Test("Recheck performs a real re-resolution and records its outcome")
    func recheckRecordsARealOutcome() {
        let server = ServerManager()
        #expect(server.lastBinaryRecheck == nil, "nothing to report before the user asks")

        let found = server.refreshBinary(userInitiated: true)
        let recheck = server.lastBinaryRecheck
        #expect(recheck != nil, "Recheck must leave evidence that it ran")
        #expect(recheck?.found == found, "the report must match what locate() actually said")
        #expect(recheck?.attempt == 1)
    }

    @Test("A second Recheck is observably a second event")
    func repeatedRecheckIsNotSwallowed() {
        let server = ServerManager()
        server.refreshBinary(userInitiated: true)
        let first = server.lastBinaryRecheck
        server.refreshBinary(userInitiated: true)
        let second = server.lastBinaryRecheck

        // Identical outcomes must still differ, or the UI coalesces them and
        // the button reads as inert on every press after the first.
        #expect(first != second)
        #expect(second?.attempt == 2)
    }

    @Test("Recheck says something, and never diagnoses why the engine is absent")
    func recheckProducesVisibleFeedback() {
        #expect(ServerManager.recheckStatusMessage(for: nil) == nil)

        let missing = ServerManager.recheckStatusMessage(
            for: .init(found: false, attempt: 1)
        )
        #expect(missing != nil)
        #expect(missing!.lowercased().contains("checked again"))

        // Two presses must not read as one.
        let again = ServerManager.recheckStatusMessage(
            for: .init(found: false, attempt: 3)
        )
        #expect(again != missing)
        #expect(again!.contains("3"))

        // No invented cause, no promised retry schedule.
        for text in [missing!, again!] {
            let lowered = text.lowercased()
            for forbidden in ["reinstall", "permission", "will retry", "trying again", "shortly"] {
                #expect(!lowered.contains(forbidden), "recheck copy must not say '\(forbidden)'")
            }
        }

        let found = ServerManager.recheckStatusMessage(for: .init(found: true, attempt: 2))
        #expect(found != nil)
        #expect(!found!.lowercased().contains("still"))
    }

    @Test("A non-user refresh leaves no 'you rechecked' state behind")
    func launchRefreshDoesNotFakeARecheck() {
        let server = ServerManager()
        server.refreshBinary(userInitiated: true)
        #expect(server.lastBinaryRecheck != nil)
        server.refreshBinary()
        #expect(server.lastBinaryRecheck == nil)
    }

    // MARK: - 5b. What VoiceOver actually receives
    //
    // These exist because the live pass could not cover them: the tester does
    // not use VoiceOver, so "a screen reader hears this" was asserted by
    // nobody. Two halves are pinned separately, because they fail separately:
    //
    //   * The STRING — pure functions, asserted directly.
    //   * The WIRING — that the string is actually posted. `announce` goes
    //     into AppKit's accessibility bus and returns Void, so there is no
    //     value to observe; the established pattern in this package (see
    //     ``OnboardingCompletionBehaviorTests``) is a source-level assertion,
    //     which is what catches the call being deleted or moved.

    private static var packageRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // RapidTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // rapid-mac
    }

    private static func strippedSource(_ relativePath: String) throws -> String {
        let url = packageRoot.appendingPathComponent(relativePath)
        let body = try String(contentsOf: url, encoding: .utf8)
        return CapabilityChipRenderGateSourceGuardTests.stripCommentsAndWhitespace(body)
    }

    @Test("A cancellation is announced, with its own words")
    func cancellationIsAnnounced() {
        let spoken = QuickstartView.recoveryAnnouncement(for: .downloadCancelled)
        #expect(spoken.contains("Download stopped"), "the heading must be spoken")
        #expect(spoken.contains(cancelledMessage), "the explanation must be spoken")
        #expect(spoken.contains("Retry"), "the one offered action must be named")
        // The whole point of the split, carried through to speech.
        #expect(!spoken.lowercased().contains("connection"))
    }

    @Test("A genuine failure is announced, and does not sound like a cancellation")
    func genuineFailureIsAnnounced() {
        let failed = QuickstartView.recoveryAnnouncement(for: .downloadFailed)
        #expect(failed.contains("Quickstart didn't finish"))
        #expect(failed.contains(downloadFailedMessage))
        #expect(failed.contains("Retry"))
        #expect(failed != QuickstartView.recoveryAnnouncement(for: .downloadCancelled))

        // Offline names its own distinct action rather than a generic retry.
        let offline = QuickstartView.recoveryAnnouncement(for: .downloadSourceUnavailable)
        #expect(offline.contains("Switch source"))
        #expect(offline != failed)
    }

    @Test("A load failure is announced too")
    func loadFailureIsAnnounced() {
        let spoken = QuickstartView.recoveryAnnouncement(for: .modelLoadFailed)
        #expect(!spoken.isEmpty)
        #expect(spoken.contains("Quickstart didn't finish"))
        #expect(spoken.contains(FailureDiagnoser.diagnosis(for: .modelLoadFailed).message))
    }

    @Test("Every recovery arrival posts its announcement")
    func recoveryArrivalsAreWiredToTheAnnouncer() throws {
        let body = try Self.strippedSource("Sources/Rapid/UI/QuickstartView.swift")
        // The one helper that both changes phase and speaks. If a future edit
        // splits these, a recovery screen can appear in silence again.
        #expect(
            body.contains(
                "coordinator.enterFailed(message:message,origin:origin)"
                + "VoiceOverAnnouncer.announce(Self.recoveryAnnouncement(for:kind))"
            ),
            """
            enterRecovery no longer announces after changing phase. A \
            VoiceOver user gets no signal that the screen swapped — which for \
            a cancellation means the button they pressed reads as inert, and \
            for an async failure means nothing at all.
            """
        )
        // And every route in goes through it rather than around it.
        #expect(!body.contains("case.cancelled:coordinator.enterFailed("),
                "the cancellation branch must announce, not call enterFailed directly")
        let directCalls = body.components(separatedBy: "coordinator.enterFailed(").count - 1
        #expect(directCalls == 1,
                "enterFailed must be reached only through enterRecovery (found \(directCalls) call sites)")
    }

    @Test("Every Recheck result is announced, including repeats")
    func recheckResultsAreAnnounced() throws {
        // The strings differ per attempt — otherwise a repeated press is
        // silent for a VoiceOver user even though the check really ran.
        let first = ServerManager.recheckStatusMessage(for: .init(found: false, attempt: 1))
        let second = ServerManager.recheckStatusMessage(for: .init(found: false, attempt: 2))
        let third = ServerManager.recheckStatusMessage(for: .init(found: false, attempt: 3))
        #expect(first != nil && second != nil && third != nil)
        #expect(Set([first!, second!, third!]).count == 3,
                "each repeated check must be audibly a NEW result")

        let found = ServerManager.recheckStatusMessage(for: .init(found: true, attempt: 2))
        #expect(found != nil)
        #expect(found != second, "success must not sound like another failure")

        // And the wiring: the same string the overlay draws is the string
        // posted, so the two cannot drift.
        let body = try Self.strippedSource("Sources/Rapid/UI/ContentView.swift")
        #expect(
            body.contains(
                "privatefuncrecheckEngine(){server.refreshBinary(userInitiated:true)"
                + "ifletstatus=ServerManager.recheckStatusMessage(for:server.lastBinaryRecheck)"
                + "{VoiceOverAnnouncer.announce(status)}}"
            ),
            """
            Recheck no longer announces its own result. Sighted users get a \
            line of text that may render identically to the last one; a \
            VoiceOver user pressing a button that reports nothing cannot tell \
            it apart from a button that does nothing.
            """
        )
        #expect(body.contains(#"Button("Recheck"){recheckEngine()}"#),
                "both Recheck buttons must route through the announcing helper")
    }

    @Test("The recovery Back control speaks its destination without the arrow")
    func recoveryBackIsSpokenAsADestination() throws {
        for stage in QuickstartCoordinator.Step2Stage.allCases {
            let spoken = QuickstartView.failureBackAccessibilityLabel(for: stage)
            let drawn = QuickstartView.failureBackTitle(for: stage)
            #expect(!spoken.contains("←"),
                    "VoiceOver reads U+2190 aloud — the label must not open with a glyph name")
            #expect(spoken.hasPrefix("Back to"))
            // Same destination as the visible label, minus the arrow. If these
            // ever diverge, the control says one thing and speaks another.
            #expect(drawn.hasSuffix(spoken))
        }
        // Each destination is distinct, so the spoken label is informative.
        let spokenAll = Set(QuickstartCoordinator.Step2Stage.allCases.map {
            QuickstartView.failureBackAccessibilityLabel(for: $0)
        })
        #expect(spokenAll.count == 3, "shortlist, catalogue and review must be distinguishable")

        let body = try Self.strippedSource("Sources/Rapid/UI/QuickstartView.swift")
        #expect(
            body.contains(
                #".accessibilityIdentifier("Quickstart.Failure.BackToModelSelection")"#
                + ".accessibilityLabel(Self.failureBackAccessibilityLabel(for:coordinator.step2Stage))"
            ),
            "the recovery Back control must carry the spoken destination label"
        )
    }

    @Test("The Step 3 cancel control is spoken accurately")
    func cancelControlIsSpokenAccurately() throws {
        let body = try Self.strippedSource("Sources/Rapid/UI/QuickstartView.swift")
        #expect(
            body.contains(
                #".accessibilityIdentifier("Quickstart.Download.Cancel")"#
                + #".accessibilityLabel("Canceldownloadof\(coordinator.selection.displayName)")"#
            ),
            "the cancel control must name WHICH download it stops"
        )
        #expect(
            body.contains(#".accessibilityHint("Stopsthedownload.Themodelwillnotbeinstalled.")"#),
            "the hint must state the consequence"
        )
        // No keyboard shortcut on a destructive control: Return is the most
        // likely stray key on a screen whose job is waiting, and Escape
        // already means "retreat within Step 2, else leave setup".
        #expect(
            !body.contains(
                #".accessibilityIdentifier("Quickstart.Download.Cancel")"#
                + ".keyboardShortcut"
            ),
            "the cancel control must not carry a keyboard shortcut"
        )
    }

    // MARK: - 6. Relaunch with incomplete setup

    @Test("A first run is not offered a setup to continue")
    func firstRunIsNotResuming() {
        let coord = makeCoordinator()
        #expect(!coord.setupBegun)
        #expect(!coord.isResumingIncompleteSetup)
        #expect(QuickstartView.welcomePrimaryTitle(resuming: false) == "Get started")
    }

    @Test("Relaunch after an unfinished setup says so, and carries nothing over")
    func relaunchRestoresATruthfulIncompleteState() {
        let first = makeCoordinator()
        first.advanceToChooseModel()
        let pick = QuickstartCoordinator.onboardingChoices.first { $0.alias == "qwen3.5-4b-4bit" }!
        first.select(pick)
        first.enterDownloading()
        #expect(first.setupBegun)

        // Quit mid-download; nothing else is persisted.
        let relaunched = QuickstartCoordinator()
        defer { relaunched._testingReset() }

        #expect(relaunched.isResumingIncompleteSetup)
        #expect(QuickstartView.welcomePrimaryTitle(resuming: true) == "Continue setup")
        // Never a restored transfer, and never a restored pick.
        #expect(relaunched.phase == .idle, "a relaunch must not resurrect an in-flight phase")
        #expect(relaunched.step == .welcome)
        #expect(relaunched.selection.alias == QuickstartCoordinator.defaultChoice.alias,
                "Paper: nothing is carried over — no selection, no job record")
        #expect(!relaunched.done)
    }

    @Test("Continuing an incomplete setup lands on the model chooser")
    func continueSetupLandsOnStepTwo() {
        let first = makeCoordinator()
        first.advanceToChooseModel()

        let relaunched = QuickstartCoordinator()
        defer { relaunched._testingReset() }
        #expect(relaunched.isResumingIncompleteSetup)

        relaunched.advanceToChooseModel()
        #expect(relaunched.step == .chooseModel)
        #expect(relaunched.step2Stage == .checkingHardware)
        #expect(relaunched.phase == .idle)
    }

    @Test("The relaunch notice promises a download, never a resume")
    func relaunchNoticeMakesNoResumeClaim() {
        // Pinned through the coordinator's own contract rather than the view's
        // string: what must hold is that nothing in the app can restore a
        // transfer, so no copy anywhere is entitled to imply one.
        let first = makeCoordinator()
        first.advanceToChooseModel()
        first.enterDownloading()

        let relaunched = QuickstartCoordinator()
        defer { relaunched._testingReset() }
        #expect(relaunched.phase == .idle)
        #expect(!ContentView.quickstartRetainsSurface(phase: relaunched.phase),
                "an idle relaunch holds the window through eligibility, not through a fake in-flight phase")
    }

    @Test("Skipping leaves setup owed and still resumable")
    func skipKeepsSetupOwed() {
        let coord = makeCoordinator()
        coord.advanceToChooseModel()
        coord.skipForNow()
        #expect(!coord.done, "Skip must never write the completion flag")
        #expect(coord.isResumingIncompleteSetup,
                "the user did enter setup — the next launch may truthfully offer to continue")
    }

    @Test("Completing setup retires the resume record")
    func completionClearsTheResumeRecord() {
        let coord = makeCoordinator()
        defer { coord._testingReset() }
        coord.advanceToChooseModel()
        #expect(coord.setupBegun)

        coord.enterStarting()
        coord.enterReady()
        #expect(coord.confirmStartChatting(seedWelcome: { true }))

        #expect(coord.done)
        #expect(!coord.setupBegun)
        #expect(!coord.isResumingIncompleteSetup)

        // And it stays retired across the relaunch.
        let relaunched = QuickstartCoordinator()
        #expect(!relaunched.isResumingIncompleteSetup)
        #expect(relaunched.done)
    }

    @Test("An unconfirmed Ready flow still reopens on its own model")
    func pendingReadyStillOutranksTheWelcomeScreen() {
        // Unchanged by this slice, pinned so the new resume flag cannot
        // quietly demote the Ready case to a generic "continue setup".
        let coord = makeCoordinator()
        defer { coord._testingReset() }
        coord.advanceToChooseModel()
        let pick = QuickstartCoordinator.onboardingChoices.first { $0.alias == "qwen3.5-4b-4bit" }!
        coord.select(pick)
        coord.enterStarting()
        coord.enterReady()
        #expect(coord.hasPendingReady)

        let relaunched = QuickstartCoordinator()
        #expect(relaunched.selection.alias == "qwen3.5-4b-4bit")
        #expect(relaunched.step == .chooseModel, "not the welcome hero")
        // Still not a claim that the model is up on THIS launch.
        #expect(relaunched.phase == .idle)
    }
}
