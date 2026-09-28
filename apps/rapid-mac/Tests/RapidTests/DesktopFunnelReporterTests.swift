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
        ])
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

    @Test("Existing installs without a cohort send none of the six downstream milestones")
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

    @Test("A first-run cohort sends all six downstream milestones")
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

        // A new process can see the durable cohort but cannot reconstruct the
        // active first-run flow capability from it.
        DesktopFunnelReporter.resetProcessStateForTesting()
        let reonboardingToken = await reporter.beginFirstRunFlow(isFirstRun: false)
        #expect(reonboardingToken == nil)
        for milestone in [
            DesktopFunnelReporter.Milestone.modelDownloadStarted,
            .modelDownloadCompleted,
            .modelDownloadFailed,
            .engineReady,
            .engineStartFailed,
        ] {
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
    func developmentPackagingIsIneligible() throws {
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

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/bash")
        process.arguments = [
            appRoot.appendingPathComponent(
                "scripts/configure-desktop-funnel-build.sh"
            ).path,
            plist.path,
            "0",
        ]
        try process.run()
        process.waitUntilExit()

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
    func releasePackagingIsEligible() throws {
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

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/bash")
        process.arguments = [
            appRoot.appendingPathComponent(
                "scripts/configure-desktop-funnel-build.sh"
            ).path,
            plist.path,
            "1",
            "developer-id-fixture",
            "TEAMFIXTURE",
        ]
        try process.run()
        process.waitUntilExit()

        #expect(process.terminationStatus == 0)
        let packaged = try PropertyListSerialization.propertyList(
            from: Data(contentsOf: plist),
            format: nil
        ) as? [String: Any]
        #expect(packaged?[DesktopFunnelReporter.releaseBuildInfoKey] as? Bool == true)
    }

    @Test("Ad-hoc packaging cannot inject the release-only marker")
    func adHocPackagingCannotBecomeEligible() throws {
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

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/bash")
        process.arguments = [
            appRoot.appendingPathComponent(
                "scripts/configure-desktop-funnel-build.sh"
            ).path,
            plist.path,
            "1",
            "-",
            "",
        ]
        process.standardError = Pipe()
        try process.run()
        process.waitUntilExit()

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
        #expect(!FileManager.default.fileExists(
            atPath: marker("desktop_funnel_first_chat_reply", in: directory).path
        ))
    }
}

private actor FunnelSendProbe {
    private var requests: [URLRequest] = []

    var count: Int { requests.count }

    func send(_ request: URLRequest) {
        requests.append(request)
    }
}
