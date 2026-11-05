import Foundation
import Testing
@testable import Rapid

@Suite("Anonymous Desktop first-run funnel", .serialized)
struct DesktopFunnelReporterTests {
    private func temporaryDirectory(_ label: String) -> URL {
        URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
            .appendingPathComponent("rapid-desktop-funnel-\(label)-\(UUID().uuidString)")
    }

    private func marker(_ name: String, in directory: URL) -> URL {
        directory.appendingPathComponent(name, isDirectory: false)
    }

    private func enroll(_ directory: URL) throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Data().write(to: marker(DesktopFunnelReporter.cohortMarkerName, in: directory))
    }

    @Test("Milestone raw values exactly match the shared contract")
    func milestoneRawValues() {
        #expect(DesktopFunnelReporter.Milestone.allCases.map(\.rawValue) == [
            "onboarding_shown",
            "model_download_started",
            "model_download_completed",
            "model_download_failed",
            "engine_ready",
            "engine_start_failed",
            "first_chat_reply",
            "setup_resumed",
            "model_choice_shown",
            "catalog_load_failed",
            "download_review_shown",
            "onboarding_skipped",
        ])
    }

    @Test("An enrolled unfinished setup resumes after relaunch without repeating milestones")
    @MainActor
    func resumedSetupSendsLaterMilestones() async throws {
        let directory = temporaryDirectory("resume")
        let suite = TestDefaultsScope.mintSuiteName(prefix: "desktop-funnel-resume")
        defer {
            DesktopFunnelReporter.resetProcessStateForTesting()
            TestDefaultsScope.cleanup(suiteNames: [suite])
            try? FileManager.default.removeItem(at: directory)
        }
        let defaults = try #require(UserDefaults(suiteName: suite))
        let probe = FunnelSendProbe()
        func reporter() -> DesktopFunnelReporter {
            DesktopFunnelReporter(isEnabled: { true }, send: { await probe.send($0) },
                                  markerDirectory: directory, version: "1.0.0")
        }
        let first = QuickstartCoordinator(defaults: defaults)
        let original = try #require(first.enrollDesktopFunnelFlowIfNeeded(markerDirectory: directory))
        let initialReporter = reporter()
        await initialReporter.report(.onboardingShown, flowToken: original)
        first.advanceToChooseModel()
        await initialReporter.report(.modelDownloadStarted, flowToken: original)
        let cohort = marker(DesktopFunnelReporter.cohortMarkerName, in: directory)
        let date = DesktopFunnelReporter.enrollmentDate(at: cohort)

        for _ in 0..<2 {
            DesktopFunnelReporter.resetProcessStateForTesting()
            let later = QuickstartCoordinator(defaults: defaults, hadPreviousLaunch: true)
            #expect(later.hasPriorUse)
            let resumed = try #require(later.enrollDesktopFunnelFlowIfNeeded(markerDirectory: directory))
            #expect(resumed != original)
            #expect(later.enrollDesktopFunnelFlowIfNeeded(markerDirectory: directory) == resumed)
            let laterReporter = reporter()
            let countBeforeStaleToken = await probe.count
            await laterReporter.report(.modelChoiceShown, flowToken: original)
            #expect(await probe.count == countBeforeStaleToken)
            for milestone in DesktopFunnelReporter.Milestone.allCases {
                await laterReporter.report(milestone, flowToken: resumed)
            }
            #expect(DesktopFunnelReporter.enrollmentDate(at: cohort) == date)
        }
        #expect(await probe.count == DesktopFunnelReporter.Milestone.allCases.count)
    }

    @Test("Resume cannot enroll an upgrader, a completed setup, or a reset rehearsal")
    @MainActor
    func resumeEligibility() async throws {
        let directory = temporaryDirectory("resume-eligibility")
        let suite = TestDefaultsScope.mintSuiteName(prefix: "desktop-funnel-resume-eligibility")
        defer {
            DesktopFunnelReporter.resetProcessStateForTesting()
            TestDefaultsScope.cleanup(suiteNames: [suite])
            try? FileManager.default.removeItem(at: directory)
        }
        let defaults = try #require(UserDefaults(suiteName: suite))
        let upgrader = QuickstartCoordinator(defaults: defaults, hadPreviousLaunch: true)
        #expect(upgrader.enrollDesktopFunnelFlowIfNeeded(markerDirectory: directory) == nil)
        #expect(!FileManager.default.fileExists(atPath: directory.path))
        #expect(DesktopFunnelReporter.resumeFirstRunFlow(
            isSetupUnfinished: true, markerDirectory: directory
        ) == nil)
        try enroll(directory)
        let unfinished = QuickstartCoordinator(defaults: defaults, hadPreviousLaunch: true)
        #expect(unfinished.enrollDesktopFunnelFlowIfNeeded(markerDirectory: directory) != nil)
        unfinished.markDone()
        #expect(unfinished.enrollDesktopFunnelFlowIfNeeded(markerDirectory: directory) == nil)
        unfinished.resetForReonboarding()
        #expect(unfinished.enrollDesktopFunnelFlowIfNeeded(markerDirectory: directory) == nil)
        DesktopFunnelReporter.resetProcessStateForTesting()
        let resetRelaunch = QuickstartCoordinator(defaults: defaults, hadPreviousLaunch: true)
        #expect(!resetRelaunch.done)
        #expect(resetRelaunch.enrollDesktopFunnelFlowIfNeeded(markerDirectory: directory) == nil)
        #expect(DesktopFunnelReporter.resumeFirstRunFlow(
            isSetupUnfinished: false, markerDirectory: directory
        ) == nil)
    }

    @Test("Every new milestone has the same minimal bounded wire body")
    func stepMilestoneWireBodies() throws {
        for milestone in [DesktopFunnelReporter.Milestone.setupResumed, .modelChoiceShown,
                          .catalogLoadFailed, .downloadReviewShown, .onboardingSkipped] {
            let body = try #require(DesktopFunnelReporter.request(
                version: "1.0.0", milestone: milestone,
                detail: .download(.network, source: .mirror), firstReplyBucket: .gt24h
            )?.httpBody)
            #expect(String(data: body, encoding: .utf8) == "{\"m\":\"\(milestone.rawValue)\",\"v\":\"1.0.0\"}")
            #expect(body.count <= DesktopFunnelReporter.maxBodyBytes)
        }
    }

    @Test("First reply bucket boundaries include exact edges")
    func firstReplyBucketBoundaries() {
        #expect(DesktopFunnelReporter.FirstReplyBucket.allCases.map(\.rawValue) == [
            "lt_3m", "3_10m", "10_60m", "1_24h", "gt_24h",
        ])
        let date = Date(timeIntervalSince1970: 1_700_000_000)
        let cases: [(TimeInterval, DesktopFunnelReporter.FirstReplyBucket)] = [
            (-1, .lt3m), (0, .lt3m), (179.999, .lt3m),
            (180, .threeToTenMinutes), (599.999, .threeToTenMinutes),
            (600, .tenToSixtyMinutes), (3599.999, .tenToSixtyMinutes),
            (3600, .oneToTwentyFourHours), (86399.999, .oneToTwentyFourHours),
            (86400, .gt24h), (86401, .gt24h),
        ]
        for (elapsed, bucket) in cases {
            #expect(DesktopFunnelReporter.firstReplyBucket(
                enrolledAt: date, now: date.addingTimeInterval(elapsed)
            ) == bucket)
        }
        #expect(DesktopFunnelReporter.firstReplyBucket(enrolledAt: nil, now: date) == nil)
    }

    @Test("First reply sends only the bucket, without a flow token, and omits unavailable dates")
    func firstReplyBucketTransport() async throws {
        let date = Date(timeIntervalSince1970: 1_700_000_000)
        let cases: [(Date?, String?)] = [(date, "3_10m"), (nil, nil)]
        for (enrolledAt, bucket) in cases {
            let directory = temporaryDirectory("bucket")
            defer { try? FileManager.default.removeItem(at: directory) }
            try enroll(directory)
            let probe = FunnelSendProbe()
            let reporter = DesktopFunnelReporter(
                isEnabled: { true }, send: { await probe.send($0) },
                markerDirectory: directory, version: "1.0.0",
                markerDate: { _ in enrolledAt }, now: { date.addingTimeInterval(180) }
            )
            await reporter.report(.firstChatReply)
            let body = try #require(await probe.firstBody)
            let object = try #require(JSONSerialization.jsonObject(with: body) as? [String: String])
            var expected = ["v": "1.0.0", "m": "first_chat_reply"]
            expected["k"] = bucket
            #expect(object == expected)
            #expect(body.count <= DesktopFunnelReporter.maxBodyBytes)
        }
        for bucket in DesktopFunnelReporter.FirstReplyBucket.allCases {
            let body = try #require(DesktopFunnelReporter.request(
                version: "1.0.0", milestone: .firstChatReply, firstReplyBucket: bucket
            )?.httpBody)
            #expect(body.count <= DesktopFunnelReporter.maxBodyBytes)
            let object = try #require(JSONSerialization.jsonObject(with: body) as? [String: String])
            #expect(object == ["v": "1.0.0", "m": "first_chat_reply", "k": bucket.rawValue])
        }
    }

    @Test("Production date reader uses the existing cohort's filesystem date")
    func enrollmentDateReader() throws {
        let directory = temporaryDirectory("marker-date")
        defer { try? FileManager.default.removeItem(at: directory) }
        let cohort = marker(DesktopFunnelReporter.cohortMarkerName, in: directory)
        #expect(DesktopFunnelReporter.enrollmentDate(at: cohort) == nil)
        try enroll(directory)
        let attributes = try FileManager.default.attributesOfItem(atPath: cohort.path)
        #expect(DesktopFunnelReporter.enrollmentDate(at: cohort) ==
                ((attributes[.creationDate] as? Date) ?? (attributes[.modificationDate] as? Date)))
    }

    @Test("Payload contains exactly version and milestone")
    func payloadShape() throws {
        let request = try #require(DesktopFunnelReporter.request(
            version: "0.15.3",
            milestone: .engineReady
        ))
        #expect(request.url == DesktopFunnelReporter.endpoint)
        #expect(request.url?.query == nil)
        #expect(request.httpMethod == "POST")
        #expect(request.timeoutInterval == 3)
        #expect(request.allHTTPHeaderFields == [
            "Content-Type": "application/json",
            "User-Agent": "Rapid-Desktop/0.15.3",
        ])

        let body = try #require(request.httpBody)
        #expect(body.count <= DesktopFunnelReporter.maxBodyBytes)
        let object = try #require(
            try JSONSerialization.jsonObject(with: body) as? [String: String]
        )
        #expect(object == ["v": "0.15.3", "m": "engine_ready"])
        #expect(Set(object.keys) == ["v", "m"])
    }

    @Test("Failure wire contract")
    func failureWireContract() throws {
        typealias Reporter = DesktopFunnelReporter
        let downloads = ["network", "disk_full", "source_unavailable", "not_found", "integrity", "other"]
        let sources = ["huggingface", "mirror"]
        let engines = ["out_of_memory", "model_load_failed", "engine_not_running", "other"]
        #expect(Reporter.DownloadFailureCause.allCases.map(\.rawValue) == downloads)
        #expect(Reporter.DownloadSourceTag.allCases.map(\.rawValue) == sources)
        #expect(Reporter.EngineStartFailureCause.allCases.map(\.rawValue) == engines)
        #expect(Reporter.DownloadSourceTag(.mirror) == .mirror)
        #expect(Reporter.DownloadSourceTag(.huggingFace) == .huggingface)
        for (cause, wire) in zip(Reporter.DownloadFailureCause.allCases, downloads) {
            let tags: [(Reporter.DownloadSourceTag?, String?)] = [(.huggingface, "huggingface"), (.mirror, "mirror"), (nil, nil)]
            for (tag, source) in tags {
                let body = try #require(Reporter.request(version: "1.0.0", milestone: .modelDownloadFailed, detail: .download(cause, source: tag))?.httpBody)
                let sourcePart = source.map { ",\"s\":\"\($0)\"" } ?? ""
                #expect(String(data: body, encoding: .utf8) == "{\"k\":\"\(wire)\",\"m\":\"model_download_failed\"\(sourcePart),\"v\":\"1.0.0\"}")
                #expect(body.count <= Reporter.maxBodyBytes)
            }
        }
        for (cause, wire) in zip(Reporter.EngineStartFailureCause.allCases, engines) {
            let body = try #require(Reporter.request(version: "1.0.0", milestone: .engineStartFailed, detail: .engine(cause))?.httpBody)
            #expect(String(data: body, encoding: .utf8) == "{\"k\":\"\(wire)\",\"m\":\"engine_start_failed\",\"v\":\"1.0.0\"}")
            #expect(body.count <= Reporter.maxBodyBytes)
        }
        let cases: [(Reporter.Milestone, Reporter.FailureDetail?, String)] = [
            (.modelDownloadFailed, nil, "model_download_failed"),
            (.modelDownloadFailed, .engine(.outOfMemory), "model_download_failed"),
            (.engineStartFailed, nil, "engine_start_failed"),
            (.engineStartFailed, .download(.network, source: .mirror), "engine_start_failed"),
        ]
        for (milestone, detail, wire) in cases {
            let body = try #require(Reporter.request(version: "1.0.0", milestone: milestone, detail: detail)?.httpBody)
            #expect(String(data: body, encoding: .utf8) == "{\"k\":\"other\",\"m\":\"\(wire)\",\"v\":\"1.0.0\"}")
            #expect(body.count <= Reporter.maxBodyBytes)
        }
        for milestone in Reporter.Milestone.allCases where milestone != .modelDownloadFailed && milestone != .engineStartFailed {
            for detail: Reporter.FailureDetail in [.engine(.outOfMemory), .download(.network, source: .mirror)] {
                let body = try #require(Reporter.request(version: "1.0.0", milestone: milestone, detail: detail)?.httpBody)
                let object = try #require(JSONSerialization.jsonObject(with: body) as? [String: String])
                #expect(Set(object.keys) == ["v", "m"])
                #expect(body.count <= Reporter.maxBodyBytes)
            }
        }
    }

    @Test("Failure detail reaches transport")
    func failureDetailReachesTransport() async throws {
        let directory = temporaryDirectory("failure-detail")
        defer { try? FileManager.default.removeItem(at: directory) }
        let probe = FunnelSendProbe()
        let reporter = DesktopFunnelReporter(isEnabled: { true }, send: { await probe.send($0) }, markerDirectory: directory, version: "1.0.0")
        let token = try #require(await reporter.beginFirstRunFlow(isFirstRun: true))
        await reporter.report(.modelDownloadFailed, detail: .download(.network, source: .huggingface), flowToken: token)
        let body = try #require(await probe.firstBody)
        #expect(String(data: body, encoding: .utf8) == #"{"k":"network","m":"model_download_failed","s":"huggingface","v":"1.0.0"}"#)
    }

    @Test("Oversize payload is never constructed")
    func oversizePayloadIsRejectedLocally() {
        #expect(DesktopFunnelReporter.request(
            version: String(repeating: "1", count: 256),
            milestone: .onboardingShown
        ) == nil)
    }

    @Test("Milestone is durably claimed before transport and never repeats")
    func claimedBeforeTransport() async throws {
        let directory = temporaryDirectory("accepted")
        defer { try? FileManager.default.removeItem(at: directory) }
        let probe = FunnelSendProbe()

        func makeReporter() -> DesktopFunnelReporter {
            DesktopFunnelReporter(
                isEnabled: { true },
                send: { request in
                    #expect(FileManager.default.fileExists(
                        atPath: directory.appendingPathComponent(
                            "desktop_funnel_onboarding_shown"
                        ).path
                    ))
                    await probe.send(request)
                },
                markerDirectory: directory,
                version: "0.15.3"
            )
        }

        let first = makeReporter()
        let token = try #require(await first.beginFirstRunFlow(isFirstRun: true))
        await first.report(.onboardingShown, flowToken: token)
        await first.report(.onboardingShown, flowToken: token)
        let second = makeReporter()
        await second.report(.onboardingShown, flowToken: token)

        #expect(await probe.count == 1)
        #expect(FileManager.default.fileExists(
            atPath: directory.appendingPathComponent(
                "desktop_funnel_onboarding_shown"
            ).path
        ))
    }

    @Test("A milestone is not sent when its durable claim fails")
    func markerFailureBlocksTransport() async throws {
        let directory = temporaryDirectory("marker-failure")
        defer { try? FileManager.default.removeItem(at: directory) }
        let probe = FunnelSendProbe()
        let reporter = DesktopFunnelReporter(
            isEnabled: { true },
            send: { request in await probe.send(request) },
            markerDirectory: directory,
            version: "0.15.3",
            claimMarker: { url in
                guard url.lastPathComponent == DesktopFunnelReporter.cohortMarkerName else {
                    return false
                }
                try? FileManager.default.createDirectory(
                    at: directory,
                    withIntermediateDirectories: true
                )
                return FileManager.default.createFile(atPath: url.path, contents: Data())
            }
        )
        let token = try #require(await reporter.beginFirstRunFlow(isFirstRun: true))
        await reporter.report(.modelDownloadStarted, flowToken: token)

        #expect(await probe.count == 0)
        #expect(!FileManager.default.fileExists(
            atPath: marker("desktop_funnel_model_download_started", in: directory).path
        ))
    }

    @Test("Existing installs without a cohort send none of the downstream milestones")
    func existingInstallIsNotBackfilled() async {
        let directory = temporaryDirectory("existing")
        defer { try? FileManager.default.removeItem(at: directory) }
        let probe = FunnelSendProbe()
        let reporter = DesktopFunnelReporter(
            isEnabled: { true },
            send: { request in await probe.send(request) },
            markerDirectory: directory,
            version: "0.15.3"
        )

        for milestone in DesktopFunnelReporter.Milestone.allCases
            where milestone != .onboardingShown {
            await reporter.report(milestone)
        }

        #expect(await probe.count == 0)
        #expect(!FileManager.default.fileExists(atPath: directory.path))
    }

    @Test("A first-run cohort sends all downstream milestones")
    func cohortSendsEveryDownstreamMilestone() async {
        let directory = temporaryDirectory("cohort")
        defer { try? FileManager.default.removeItem(at: directory) }
        let probe = FunnelSendProbe()
        let reporter = DesktopFunnelReporter(
            isEnabled: { true },
            send: { request in await probe.send(request) },
            markerDirectory: directory,
            version: "0.15.3"
        )

        let token = await reporter.beginFirstRunFlow(isFirstRun: true)
        #expect(token != nil)
        guard let token else { return }
        await reporter.report(.onboardingShown, flowToken: token)
        #expect(FileManager.default.fileExists(
            atPath: marker(DesktopFunnelReporter.cohortMarkerName, in: directory).path
        ))
        for milestone in DesktopFunnelReporter.Milestone.allCases
            where milestone != .onboardingShown {
            await reporter.report(milestone, flowToken: token)
        }

        #expect(await probe.count == DesktopFunnelReporter.Milestone.allCases.count)
        for milestone in DesktopFunnelReporter.Milestone.allCases {
            #expect(FileManager.default.fileExists(
                atPath: marker("desktop_funnel_\(milestone.rawValue)", in: directory).path
            ))
        }
    }

    @Test("Re-onboarding an install with chat history does not create a cohort")
    @MainActor
    func reonboardingIsNotANewInstall() async {
        let directory = temporaryDirectory("reonboarding")
        defer { try? FileManager.default.removeItem(at: directory) }
        let suite = TestDefaultsScope.mintSuiteName(prefix: "desktop-funnel-reonboarding")
        defer { TestDefaultsScope.cleanup(suiteNames: [suite]) }
        let defaults = UserDefaults(suiteName: suite)!
        let coordinator = QuickstartCoordinator(defaults: defaults, hasChatHistory: true)
        let probe = FunnelSendProbe()
        let reporter = DesktopFunnelReporter(
            isEnabled: { true },
            send: { request in await probe.send(request) },
            markerDirectory: directory,
            version: "0.15.3"
        )

        let token = await reporter.beginFirstRunFlow(isFirstRun: !coordinator.hasPriorUse)
        #expect(token == nil)

        #expect(await probe.count == 0)
        #expect(!FileManager.default.fileExists(
            atPath: marker(DesktopFunnelReporter.cohortMarkerName, in: directory).path
        ))
    }

    @Test("A cohort install re-running guided setup cannot send download or engine milestones")
    func reonboardingCohortHasNoFlowToken() async throws {
        let directory = temporaryDirectory("cohort-reonboarding")
        defer {
            DesktopFunnelReporter.resetProcessStateForTesting()
            try? FileManager.default.removeItem(at: directory)
        }
        let probe = FunnelSendProbe()
        let reporter = DesktopFunnelReporter(
            isEnabled: { true },
            send: { request in await probe.send(request) },
            markerDirectory: directory,
            version: "0.15.3"
        )

        let firstRunToken = try #require(
            await reporter.beginFirstRunFlow(isFirstRun: true)
        )
        await reporter.report(.onboardingShown, flowToken: firstRunToken)
        #expect(await probe.count == 1)

        // Enrollment alone cannot reconstruct a token. The separate resume
        // path additionally requires the coordinator's unfinished-setup checks.
        DesktopFunnelReporter.resetProcessStateForTesting()
        let reonboardingToken = await reporter.beginFirstRunFlow(isFirstRun: false)
        #expect(reonboardingToken == nil)
        for milestone in DesktopFunnelReporter.Milestone.allCases where milestone != .firstChatReply {
            await reporter.report(milestone, flowToken: reonboardingToken)
        }

        #expect(await probe.count == 1)
    }

    @Test("Upgrade with untouched onboarding is durably excluded before InstallTracker rollover")
    @MainActor
    func untouchedUpgradeIsExcluded() {
        let suite = TestDefaultsScope.mintSuiteName(prefix: "desktop-funnel-upgrade-untouched")
        defer { TestDefaultsScope.cleanup(suiteNames: [suite]) }
        let defaults = UserDefaults(suiteName: suite)!
        defaults.set("0.14.0", forKey: InstallTracker.lastSeenVersionKey)

        let hadPreviousLaunch = defaults.string(
            forKey: InstallTracker.lastSeenVersionKey
        ) != nil
        _ = InstallTracker(
            currentVersion: "0.15.3",
            currentInfoPlistMtime: Date(),
            currentBundleURL: URL(fileURLWithPath: "/Applications/Rapid-MLX Desktop.app"),
            defaults: defaults
        )
        let coordinator = QuickstartCoordinator(
            defaults: defaults,
            hadPreviousLaunch: hadPreviousLaunch
        )

        #expect(defaults.string(forKey: InstallTracker.lastSeenVersionKey) == "0.15.3")
        #expect(coordinator.hasPriorUse)
        #expect(defaults.bool(forKey: QuickstartCoordinator.priorUseStorageKey))
        #expect(QuickstartCoordinator(defaults: defaults).hasPriorUse)
    }

    @Test("Upgrade with incomplete onboarding is durably excluded")
    @MainActor
    func incompleteOnboardingUpgradeIsExcluded() {
        let suite = TestDefaultsScope.mintSuiteName(prefix: "desktop-funnel-upgrade-incomplete")
        defer { TestDefaultsScope.cleanup(suiteNames: [suite]) }
        let defaults = UserDefaults(suiteName: suite)!
        defaults.set(true, forKey: QuickstartCoordinator.setupBegunKey)

        let coordinator = QuickstartCoordinator(
            defaults: defaults,
            hadPreviousLaunch: false
        )

        #expect(coordinator.setupBegun)
        #expect(coordinator.hasPriorUse)
        #expect(defaults.bool(forKey: QuickstartCoordinator.priorUseStorageKey))
    }

    @Test("A failed request is at most once and is not retried by a later reporter")
    func failedRequestIsNotRetried() async throws {
        let directory = temporaryDirectory("at-most-once")
        defer {
            DesktopFunnelReporter.resetProcessStateForTesting()
            try? FileManager.default.removeItem(at: directory)
        }
        try enroll(directory)
        let probe = FunnelSendProbe()

        func makeReporter() -> DesktopFunnelReporter {
            DesktopFunnelReporter(
                isEnabled: { true },
                send: { request in await probe.send(request) },
                markerDirectory: directory,
                version: "0.15.3"
            )
        }

        let firstLaunch = makeReporter()
        await firstLaunch.report(.firstChatReply)
        #expect(await probe.count == 1)
        #expect(FileManager.default.fileExists(
            atPath: directory.appendingPathComponent(
                "desktop_funnel_first_chat_reply"
            ).path
        ))

        DesktopFunnelReporter.resetProcessStateForTesting()
        let laterLaunch = makeReporter()
        await laterLaunch.report(.firstChatReply)
        #expect(await probe.count == 1)
        #expect(FileManager.default.fileExists(
            atPath: directory.appendingPathComponent(
                "desktop_funnel_first_chat_reply"
            ).path
        ))
    }

    @Test("Every milestone consumes its first opportunity while sending is gated off")
    func gatedFirstOpportunityIsNeverReplacedByALaterEvent() async throws {
        let directory = temporaryDirectory("first-opportunity")
        defer {
            DesktopFunnelReporter.resetProcessStateForTesting()
            try? FileManager.default.removeItem(at: directory)
        }
        let probe = FunnelSendProbe()
        let blocked = DesktopFunnelReporter(
            isEnabled: { false },
            send: { request in await probe.send(request) },
            markerDirectory: directory,
            version: "0.15.3"
        )
        let token = try #require(await blocked.beginFirstRunFlow(isFirstRun: true))

        for milestone in DesktopFunnelReporter.Milestone.allCases {
            await blocked.report(milestone, flowToken: token)
        }

        let optedInLater = DesktopFunnelReporter(
            isEnabled: { true },
            send: { request in await probe.send(request) },
            markerDirectory: directory,
            version: "0.15.3"
        )
        for milestone in DesktopFunnelReporter.Milestone.allCases {
            await optedInLater.report(milestone, flowToken: token)
        }

        #expect(await probe.count == 0)
        for milestone in DesktopFunnelReporter.Milestone.allCases {
            #expect(FileManager.default.fileExists(
                atPath: marker("desktop_funnel_\(milestone.rawValue)", in: directory).path
            ))
        }
    }

    @Test("Cohort enrollment and milestone claim survive a failed onboarding delivery")
    func cohortMarkerIsIndependentOfNetwork() async throws {
        let directory = temporaryDirectory("offline-enrollment")
        defer { try? FileManager.default.removeItem(at: directory) }
        let probe = FunnelSendProbe()
        let reporter = DesktopFunnelReporter(
            isEnabled: { true },
            send: { request in await probe.send(request) },
            markerDirectory: directory,
            version: "0.15.3"
        )

        let token = try #require(await reporter.beginFirstRunFlow(isFirstRun: true))
        await reporter.report(.onboardingShown, flowToken: token)

        #expect(await probe.count == 1)
        #expect(FileManager.default.fileExists(
            atPath: marker(DesktopFunnelReporter.cohortMarkerName, in: directory).path
        ))
        #expect(FileManager.default.fileExists(
            atPath: marker("desktop_funnel_onboarding_shown", in: directory).path
        ))
    }

    @Test("Declined telemetry gate blocks sending")
    func declinedTelemetryBlocks() async {
        await expectBlocked(telemetryDeclined: true)
    }

    @Test("Disabled update checks gate blocks sending")
    func disabledUpdateChecksBlock() async {
        await expectBlocked(updateChecksEnabled: false)
    }

    @Test("DO_NOT_TRACK gate blocks sending")
    func doNotTrackBlocks() async {
        await expectBlocked(environment: ["DO_NOT_TRACK": "1"])
    }

    @Test("Telemetry environment kill switch blocks sending")
    func telemetryKillSwitchBlocks() async {
        await expectBlocked(environment: ["RAPID_MLX_TELEMETRY": "0"])
    }

    @Test("XCTest and UI harness gates block sending", arguments: [
        ["XCTestConfigurationFilePath": "/tmp/test.xctestconfiguration"],
        ["RAPID_GUI_GOLDEN_MODE": "1"],
        ["RAPID_XCUI_DRAG_FILE": "/tmp/drag"],
    ])
    func testHarnessBlocks(environment: [String: String]) async {
        await expectBlocked(environment: environment)
    }

    @Test("Dogfood and development bundles block sending", arguments: [
        nil,
        "com.rapidmlx.rapid.dogfood-deadbeef",
        "com.example.RapidDevelopment",
    ] as [String?])
    func nonProductionBundleBlocks(bundleIdentifier: String?) async {
        await expectBlocked(bundleIdentifier: bundleIdentifier)
    }

    @Test("Normal development packaging omits the release-only marker")
    func developmentPackagingIsIneligible() async throws {
        let appRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let directory = temporaryDirectory("packaging")
        defer { try? FileManager.default.removeItem(at: directory) }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let plist = directory.appendingPathComponent("Info.plist")
        try FileManager.default.copyItem(
            at: appRoot.appendingPathComponent("Resources/Info.plist"),
            to: plist
        )

        let process = try await TestSubprocess.run(
            executableURL: URL(fileURLWithPath: "/bin/bash"),
            arguments: [
                appRoot.appendingPathComponent(
                    "scripts/configure-desktop-funnel-build.sh"
                ).path,
                plist.path,
                "0",
            ]
        )

        #expect(process.terminationStatus == 0)
        let packaged = try PropertyListSerialization.propertyList(
            from: Data(contentsOf: plist),
            format: nil
        ) as? [String: Any]
        #expect(packaged?[DesktopFunnelReporter.releaseBuildInfoKey] == nil)
        let buildScript = try String(
            contentsOf: appRoot.appendingPathComponent("scripts/build.sh"),
            encoding: .utf8
        )
        #expect(buildScript.contains(#""${RAPID_MLX_OFFICIAL_RELEASE:-0}""#),
                "the actual build.sh default must execute the tested development path")
        #expect(!DesktopFunnelReporter.gatesAllowSending(
            telemetryDeclined: false,
            updateChecksEnabled: true,
            environment: [:],
            bundleIdentifier: DesktopFunnelReporter.productionBundleIdentifier,
            releaseBuild: false
        ))
    }

    @Test("Canonical release packaging injects the release-only marker")
    func releasePackagingIsEligible() async throws {
        let appRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let directory = temporaryDirectory("release-packaging")
        defer { try? FileManager.default.removeItem(at: directory) }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let plist = directory.appendingPathComponent("Info.plist")
        try FileManager.default.copyItem(
            at: appRoot.appendingPathComponent("Resources/Info.plist"),
            to: plist
        )

        let process = try await TestSubprocess.run(
            executableURL: URL(fileURLWithPath: "/bin/bash"),
            arguments: [
                appRoot.appendingPathComponent(
                    "scripts/configure-desktop-funnel-build.sh"
                ).path,
                plist.path,
                "1",
                "developer-id-fixture",
                "TEAMFIXTURE",
            ]
        )

        #expect(process.terminationStatus == 0)
        let packaged = try PropertyListSerialization.propertyList(
            from: Data(contentsOf: plist),
            format: nil
        ) as? [String: Any]
        #expect(packaged?[DesktopFunnelReporter.releaseBuildInfoKey] as? Bool == true)
    }

    @Test("Ad-hoc packaging cannot inject the release-only marker")
    func adHocPackagingCannotBecomeEligible() async throws {
        let appRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let directory = temporaryDirectory("adhoc-packaging")
        defer { try? FileManager.default.removeItem(at: directory) }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let plist = directory.appendingPathComponent("Info.plist")
        try FileManager.default.copyItem(
            at: appRoot.appendingPathComponent("Resources/Info.plist"),
            to: plist
        )

        let process = try await TestSubprocess.run(
            executableURL: URL(fileURLWithPath: "/bin/bash"),
            arguments: [
                appRoot.appendingPathComponent(
                    "scripts/configure-desktop-funnel-build.sh"
                ).path,
                plist.path,
                "1",
                "-",
                "",
            ]
        )

        #expect(process.terminationStatus != 0)
        let packaged = try PropertyListSerialization.propertyList(
            from: Data(contentsOf: plist),
            format: nil
        ) as? [String: Any]
        #expect(packaged?[DesktopFunnelReporter.releaseBuildInfoKey] == nil)
    }

    @Test("Settings opt-out latch blocks both report gate evaluations")
    func immediateOptOutLatchBlocksBeforeAndImmediatelyBeforeTransport() async throws {
        let firstDirectory = temporaryDirectory("latched-before-report")
        let secondDirectory = temporaryDirectory("latched-between-gates")
        defer {
            DesktopFunnelReporter.resetProcessStateForTesting()
            try? FileManager.default.removeItem(at: firstDirectory)
            try? FileManager.default.removeItem(at: secondDirectory)
        }
        try enroll(firstDirectory)
        try enroll(secondDirectory)
        let probe = FunnelSendProbe()

        DesktopFunnelReporter.latchProcessOptOut()
        let blockedAtFirstGate = DesktopFunnelReporter(
            isEnabled: { true },
            send: { request in await probe.send(request) },
            markerDirectory: firstDirectory,
            version: "0.15.3"
        )
        await blockedAtFirstGate.report(.firstChatReply)

        DesktopFunnelReporter.resetProcessStateForTesting()
        let blockedAtSecondGate = DesktopFunnelReporter(
            isEnabled: {
                DesktopFunnelReporter.latchProcessOptOut()
                return true
            },
            send: { request in await probe.send(request) },
            markerDirectory: secondDirectory,
            version: "0.15.3"
        )
        await blockedAtSecondGate.report(.firstChatReply)

        #expect(await probe.count == 0)
        #expect(FileManager.default.fileExists(
            atPath: marker("desktop_funnel_first_chat_reply", in: firstDirectory).path
        ))
        #expect(FileManager.default.fileExists(
            atPath: marker("desktop_funnel_first_chat_reply", in: secondDirectory).path
        ))
    }

    @Test("Settings opt-out latch flipping during claim blocks transport")
    func optOutLatchDuringClaimBlocksTransport() async throws {
        let directory = temporaryDirectory("latched-during-claim")
        defer {
            DesktopFunnelReporter.resetProcessStateForTesting()
            try? FileManager.default.removeItem(at: directory)
        }
        try enroll(directory)
        let probe = FunnelSendProbe()
        let reporter = DesktopFunnelReporter(
            isEnabled: { true },
            send: { request in await probe.send(request) },
            markerDirectory: directory,
            version: "0.15.3",
            claimMarker: { url in
                DesktopFunnelReporter.latchProcessOptOut()
                return FileManager.default.createFile(atPath: url.path, contents: Data())
            }
        )

        await reporter.report(.firstChatReply)

        #expect(await probe.count == 0)
        #expect(FileManager.default.fileExists(
            atPath: marker("desktop_funnel_first_chat_reply", in: directory).path
        ))
    }

    @Test("Only the armed onboarding engine attempt emits a terminal outcome")
    func engineAttemptCausalityExcludesLaterManualRestartAndModelSwitch() {
        let gate = DesktopFunnelEngineAttemptGate()
        let flowToken = DesktopFunnelReporter.FlowToken(id: UUID())
        let first = gate.arm(alias: "starter", flowToken: flowToken)

        #expect(gate.consume(milestone: .engineReady, alias: "starter") == flowToken)
        #expect(gate.consume(milestone: .engineReady, alias: "starter") == nil,
                "later manual start must be silent")
        #expect(gate.consume(milestone: .engineStartFailed, alias: "starter") == nil,
                "later restart must be silent")

        let switched = gate.arm(alias: "starter", flowToken: flowToken)
        #expect(gate.consume(milestone: .engineStartFailed, alias: "larger-model") == nil,
                "a model-switch failure is not the armed onboarding start")
        gate.disarm(token: switched)
        #expect(gate.consume(milestone: .engineStartFailed, alias: "starter") == nil)
        gate.disarm(token: first)
    }

    @Test("Undecided or enabled production install passes all gates")
    func eligibleProductionInstall() {
        #expect(DesktopFunnelReporter.gatesAllowSending(
            telemetryDeclined: false,
            updateChecksEnabled: true,
            environment: [:],
            bundleIdentifier: DesktopFunnelReporter.productionBundleIdentifier,
            releaseBuild: true
        ))
    }

    @Test("Shared consent distinguishes undecided, enabled, declined, and unreadable")
    func sharedConsentGate() throws {
        let directory = temporaryDirectory("consent")
        defer { try? FileManager.default.removeItem(at: directory) }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let url = directory.appendingPathComponent("telemetry-consent.yaml")

        #expect(!TelemetryConsent.hasExplicitlyDeclined(telemetryDirectory: directory))
        try Data(#"{"consent":true,"desktop_consent":true}"#.utf8).write(to: url)
        #expect(!TelemetryConsent.hasExplicitlyDeclined(telemetryDirectory: directory))
        try Data(#"{"consent":true,"desktop_consent":false}"#.utf8).write(to: url)
        #expect(TelemetryConsent.hasExplicitlyDeclined(telemetryDirectory: directory))
        try Data("not: [valid".utf8).write(to: url)
        #expect(TelemetryConsent.hasExplicitlyDeclined(telemetryDirectory: directory))
    }

    private func expectBlocked(
        telemetryDeclined: Bool = false,
        updateChecksEnabled: Bool = true,
        environment: [String: String] = [:],
        bundleIdentifier: String? = DesktopFunnelReporter.productionBundleIdentifier,
        releaseBuild: Bool = true
    ) async {
        let directory = temporaryDirectory("blocked")
        defer { try? FileManager.default.removeItem(at: directory) }
        try? enroll(directory)
        let probe = FunnelSendProbe()
        let reporter = DesktopFunnelReporter(
            isEnabled: {
                DesktopFunnelReporter.gatesAllowSending(
                    telemetryDeclined: telemetryDeclined,
                    updateChecksEnabled: updateChecksEnabled,
                    environment: environment,
                    bundleIdentifier: bundleIdentifier,
                    releaseBuild: releaseBuild
                )
            },
            send: { request in await probe.send(request) },
            markerDirectory: directory,
            version: "0.15.3"
        )

        await reporter.report(.firstChatReply)

        #expect(await probe.count == 0)
        #expect(FileManager.default.fileExists(
            atPath: marker("desktop_funnel_first_chat_reply", in: directory).path
        ))
    }
}

private actor FunnelSendProbe {
    private var requests: [URLRequest] = []

    var count: Int { requests.count }
    var firstBody: Data? { requests.first?.httpBody }

    func send(_ request: URLRequest) {
        requests.append(request)
    }
}
