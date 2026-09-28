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

    @Test("Accepted milestone is marked and never repeats across reporters")
    func acceptedThenMarked() async throws {
        let directory = temporaryDirectory("accepted")
        defer { try? FileManager.default.removeItem(at: directory) }
        let probe = FunnelSendProbe(results: [204])

        func makeReporter() -> DesktopFunnelReporter {
            DesktopFunnelReporter(
                isEnabled: { true },
                send: { request in await probe.send(request) },
                markerDirectory: directory,
                version: "0.15.3"
            )
        }

        let first = makeReporter()
        await first.reportOnboardingShown(isFirstRun: true)
        await first.reportOnboardingShown(isFirstRun: true)
        let second = makeReporter()
        await second.reportOnboardingShown(isFirstRun: true)

        #expect(await probe.count == 1)
        #expect(FileManager.default.fileExists(
            atPath: directory.appendingPathComponent(
                "desktop_funnel_onboarding_shown"
            ).path
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

        await reporter.reportOnboardingShown(isFirstRun: true)
        #expect(FileManager.default.fileExists(
            atPath: marker(DesktopFunnelReporter.cohortMarkerName, in: directory).path
        ))
        for milestone in DesktopFunnelReporter.Milestone.allCases
            where milestone != .onboardingShown {
            await reporter.report(milestone)
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

        await reporter.reportOnboardingShown(isFirstRun: !coordinator.hasPriorUse)

        #expect(await probe.count == 0)
        #expect(!FileManager.default.fileExists(
            atPath: marker(DesktopFunnelReporter.cohortMarkerName, in: directory).path
        ))
    }

    @Test("Transport and transient HTTP failures retry on a later launch", arguments: [
        nil,
        408,
        429,
        500,
    ] as [Int?])
    func transientRequestRetriesOnlyInNewReporter(firstStatus: Int?) async throws {
        let directory = temporaryDirectory("retry")
        defer { try? FileManager.default.removeItem(at: directory) }
        try enroll(directory)
        let probe = FunnelSendProbe(results: [firstStatus, 204])

        func makeReporter() -> DesktopFunnelReporter {
            DesktopFunnelReporter(
                isEnabled: { true },
                send: { request in await probe.send(request) },
                markerDirectory: directory,
                version: "0.15.3"
            )
        }

        let firstLaunch = makeReporter()
        await firstLaunch.report(.modelDownloadFailed)
        await firstLaunch.report(.modelDownloadFailed)
        #expect(await probe.count == 1)
        #expect(!FileManager.default.fileExists(
            atPath: directory.appendingPathComponent(
                "desktop_funnel_model_download_failed"
            ).path
        ))

        let laterLaunch = makeReporter()
        await laterLaunch.report(.modelDownloadFailed)
        await laterLaunch.report(.modelDownloadFailed)
        #expect(await probe.count == 2)
        #expect(FileManager.default.fileExists(
            atPath: directory.appendingPathComponent(
                "desktop_funnel_model_download_failed"
            ).path
        ))
    }

    @Test("Permanent HTTP rejections are marked and never retried", arguments: [
        400,
        404,
        405,
        410,
    ])
    func permanentRejectionIsDiscarded(status: Int) async throws {
        let directory = temporaryDirectory("discard")
        defer { try? FileManager.default.removeItem(at: directory) }
        try enroll(directory)
        let probe = FunnelSendProbe(results: [status, 204])

        func makeReporter() -> DesktopFunnelReporter {
            DesktopFunnelReporter(
                isEnabled: { true },
                send: { request in await probe.send(request) },
                markerDirectory: directory,
                version: "0.15.3"
            )
        }

        await makeReporter().report(.engineStartFailed)
        await makeReporter().report(.engineStartFailed)

        #expect(await probe.count == 1)
        #expect(FileManager.default.fileExists(
            atPath: marker("desktop_funnel_engine_start_failed", in: directory).path
        ))
    }

    @Test("Cohort enrollment survives a failed onboarding delivery")
    func cohortMarkerIsIndependentOfNetwork() async {
        let directory = temporaryDirectory("offline-enrollment")
        defer { try? FileManager.default.removeItem(at: directory) }
        let probe = FunnelSendProbe(results: [nil])
        let reporter = DesktopFunnelReporter(
            isEnabled: { true },
            send: { request in await probe.send(request) },
            markerDirectory: directory,
            version: "0.15.3"
        )

        await reporter.reportOnboardingShown(isFirstRun: true)

        #expect(await probe.count == 1)
        #expect(FileManager.default.fileExists(
            atPath: marker(DesktopFunnelReporter.cohortMarkerName, in: directory).path
        ))
        #expect(!FileManager.default.fileExists(
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

    @Test("Undecided or enabled production install passes all gates")
    func eligibleProductionInstall() {
        #expect(DesktopFunnelReporter.gatesAllowSending(
            telemetryDeclined: false,
            updateChecksEnabled: true,
            environment: [:],
            bundleIdentifier: DesktopFunnelReporter.productionBundleIdentifier
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
        bundleIdentifier: String? = DesktopFunnelReporter.productionBundleIdentifier
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
                    bundleIdentifier: bundleIdentifier
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
    private var results: [Int?]

    init(results: [Int?] = []) {
        self.results = results
    }

    var count: Int { requests.count }

    func send(_ request: URLRequest) -> Int? {
        requests.append(request)
        return results.isEmpty ? 204 : results.removeFirst()
    }
}
