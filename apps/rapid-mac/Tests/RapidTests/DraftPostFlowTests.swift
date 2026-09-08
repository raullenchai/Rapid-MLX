import AppKit
import ApplicationServices
import CoreGraphics
import Foundation
import Testing
@testable import Rapid

@Suite("Computer Use draft-to-post flow")
struct DraftPostFlowTests {
    /// Operator-only dogfood. It intentionally requires already-open local
    /// fixtures and never runs in ordinary CI:
    ///
    /// - TextEdit window title contains `rapid-cua-draft.txt`.
    /// - Browser window title contains `Local Post Composer` and has
    ///   one empty text area labelled `Post text`.
    ///
    /// Run with `RAPID_LIVE_CUA_DOGFOOD=1 swift test --no-parallel --filter
    /// DraftPostFlowTests/liveFixture` from a trusted GUI session.
    @Test(.enabled(if: ProcessInfo.processInfo.environment["RAPID_LIVE_CUA_DOGFOOD"] == "1"))
    func liveFixture() async throws {
        let (source, destination) = try await Self.liveOptions()
        let actuator = RecordingDraftPostComposerActuator(
            base: AXDraftPostComposerActuator()
        )
        var successes = 0
        for _ in 0 ..< 30 {
            let outcome = await DraftPostFlowCoordinator(
                driver: MacOSDraftPostFlowDriver(actuator: actuator)
            ).run(source: source, destination: destination)
            guard case .readyForReview(let metrics) = outcome else {
                Issue.record("Live fixture did not reach review: \(outcome)")
                continue
            }
            successes += 1
            #expect(metrics.attempts <= 3)
            #expect(metrics.completedSteps == 3)
            try actuator.clearLastDraftIfUnchanged()
        }
        #expect(successes == 30)
        let actions = actuator.actions
        #expect(actions.count == 60)
        for index in stride(from: 0, to: actions.count, by: 2) {
            #expect(actions[index] == .focusComposer)
            guard case .setDraft = actions[index + 1] else {
                Issue.record("Unexpected composer action sequence")
                continue
            }
        }
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["RAPID_LIVE_CUA_DOGFOOD"] == "1"))
    func liveFocusStealRecovery() async throws {
        let (source, destination) = try await Self.liveOptions()
        let actuator = RecordingDraftPostComposerActuator(
            base: AXDraftPostComposerActuator()
        )
        let driver = FocusStealingDraftPostDriver(
            base: MacOSDraftPostFlowDriver(actuator: actuator)
        )
        let outcome = await DraftPostFlowCoordinator(driver: driver).run(
            source: source,
            destination: destination
        )
        guard case .readyForReview(let metrics) = outcome else {
            Issue.record("Focus-steal fixture did not recover: \(outcome)")
            return
        }
        #expect(metrics.attempts == 2)
        #expect(metrics.automaticRecoveries == 1)
        try actuator.clearLastDraftIfUnchanged()
    }

    /// Operator-only end-to-end visual recovery. The browser fixture exposes
    /// an empty but deliberately unlabelled composer, so deterministic AX
    /// lookup must fail before one local model grounding focuses it.
    @Test(.enabled(if: ProcessInfo.processInfo.environment["RAPID_LIVE_CUA_VISUAL"] == "1"))
    func liveVisualFallbackFixture() async throws {
        let environment = ProcessInfo.processInfo.environment
        let (source, destination) = try await Self.liveOptions()
        let baseURL = try #require(URL(string: environment["RAPID_LIVE_CUA_BASE_URL"]
            ?? "http://127.0.0.1:8377/v1"))
        let model = environment["RAPID_LIVE_CUA_MODEL"] ?? "EvoCUA_8B_4bit"
        let configuration = try LocalComputerUseVisualGrounder.Configuration(
            baseURL: baseURL,
            model: model,
            bearerToken: environment["RAPID_LIVE_CUA_BEARER"],
            wireContract: environment["RAPID_LIVE_CUA_WIRE_CONTRACT"] == "ui_tars"
                ? .uiTars
                : .genericFunction
        )
        let actuator = RecordingDraftPostComposerActuator(
            base: AXDraftPostComposerActuator()
        )
        let transport = RecordingDraftPostGroundingTransport()
        let recovery = MacOSDraftPostVisualRecovery(
            configuration: configuration,
            sessionValidator: { true },
            transport: transport,
            actuator: actuator
        )
        let outcome = await DraftPostFlowCoordinator(
            driver: MacOSDraftPostFlowDriver(
                actuator: actuator,
                visualRecovery: recovery
            )
        ).run(source: source, destination: destination)
        guard case .readyForReview(let metrics) = outcome else {
            let responses = await transport.responseBodies
            Issue.record("Visual fallback did not reach review: \(outcome); responses=\(responses)")
            return
        }
        #expect(metrics.completedSteps == 3)
        let actions = actuator.actions
        #expect(actions.count == 3)
        #expect(actions[0] == .focusComposer)
        #expect(actions[1] == .focusComposer)
        guard case .setDraft(let writtenDraft) = actions[2] else {
            Issue.record("Visual fallback escaped the focus-and-write capability boundary")
            return
        }
        #expect(writtenDraft == "Local AI can turn repetitive Mac work into a private, reviewable workflow.\n")
        try actuator.clearLastDraftIfUnchanged()
    }

    @Test("Draft and post remains runnable with its publish boundary")
    func catalogAvailability() throws {
        let starter = try #require(ComputerUseStarter.catalog.first {
            $0.kind == .draftAndPost
        })
        #expect(starter.availability == .available)
        #expect(starter.approvalNote == "Rapid will stop before publishing.")
    }

    @Test("A verified transfer finishes without recovery")
    func happyPath() async {
        let driver = ScriptedDraftPostDriver(results: [.success(())])
        let outcome = await DraftPostFlowCoordinator(driver: driver).run(
            source: Self.source,
            destination: Self.destination
        )
        #expect(outcome == .readyForReview(DraftPostFlowMetrics(
            attempts: 1,
            automaticRecoveries: 0,
            completedSteps: 3
        )))
        #expect(await driver.callCount == 1)
    }

    @Test("Recoverable focus drift is retried within the fixed budget")
    func boundedRecovery() async {
        let driver = ScriptedDraftPostDriver(results: [
            .failure(.focusChanged),
            .failure(.focusChanged),
            .success(()),
        ])
        let outcome = await DraftPostFlowCoordinator(driver: driver).run(
            source: Self.source,
            destination: Self.destination
        )
        #expect(outcome == .readyForReview(DraftPostFlowMetrics(
            attempts: 3,
            automaticRecoveries: 2,
            completedSteps: 3
        )))
        #expect(await driver.callCount == 3)
    }

    @Test("Recovery exhaustion pauses instead of looping")
    func exhaustedRecovery() async {
        let driver = ScriptedDraftPostDriver(results: [
            .failure(.focusChanged),
            .failure(.focusChanged),
            .failure(.focusChanged),
            .success(()),
        ])
        let outcome = await DraftPostFlowCoordinator(driver: driver).run(
            source: Self.source,
            destination: Self.destination
        )
        #expect(outcome == .failed(
            DraftPostFlowFailure.focusChanged,
            DraftPostFlowMetrics(
                attempts: 3,
                automaticRecoveries: 2,
                completedSteps: 0
            )
        ))
        #expect(await driver.callCount == 3)
    }

    @Test("Unsafe or ambiguous state never retries")
    func unsafeStateStops() async {
        for failure in [
            DraftPostFlowFailure.composerAmbiguous,
            .composerNotEmpty,
            .permissionMissing,
            .writeRejected,
            .verificationFailed,
        ] {
            let driver = ScriptedDraftPostDriver(results: [.failure(failure), .success(())])
            let outcome = await DraftPostFlowCoordinator(driver: driver).run(
                source: Self.source,
                destination: Self.destination
            )
            #expect(outcome == .failed(
                failure,
                DraftPostFlowMetrics(
                    attempts: 1,
                    automaticRecoveries: 0,
                    completedSteps: 0
                )
            ))
            #expect(await driver.callCount == 1)
        }
    }

    @Test("Unexpected adapter errors fail closed without retry")
    func unexpectedErrorStops() async {
        let driver = UnexpectedDraftPostDriver()
        let outcome = await DraftPostFlowCoordinator(driver: driver).run(
            source: Self.source,
            destination: Self.destination
        )
        #expect(outcome == .failed(
            .dependencyFailure,
            DraftPostFlowMetrics(
                attempts: 1,
                automaticRecoveries: 0,
                completedSteps: 0
            )
        ))
        #expect(await driver.callCount == 1)
    }

    @Test("Only exact composer labels are accepted")
    func exactComposerLabels() {
        #expect(MacOSDraftPostFlowDriver.isExplicitComposerLabel("Post text"))
        #expect(MacOSDraftPostFlowDriver.isExplicitComposerLabel("What’s happening?"))
        #expect(!MacOSDraftPostFlowDriver.isExplicitComposerLabel("Search posts"))
        #expect(!MacOSDraftPostFlowDriver.isExplicitComposerLabel("Update profile"))
        #expect(!MacOSDraftPostFlowDriver.isExplicitComposerLabel("Post"))
    }

    @Test("Safari and Chrome address fields use explicit browser contracts")
    func browserAddressFields() {
        #expect(MacOSDraftPostFlowDriver.matchesBrowserAddressField(
            browserBundleIdentifier: "com.apple.Safari",
            role: kAXTextFieldRole as String,
            identifier: "WEB_BROWSER_ADDRESS_AND_SEARCH_FIELD",
            description: "smart search field"
        ))
        #expect(MacOSDraftPostFlowDriver.matchesBrowserAddressField(
            browserBundleIdentifier: "com.google.Chrome",
            role: kAXTextFieldRole as String,
            identifier: nil,
            description: "Address and search bar"
        ))
        #expect(!MacOSDraftPostFlowDriver.matchesBrowserAddressField(
            browserBundleIdentifier: "com.google.Chrome",
            role: kAXTextFieldRole as String,
            identifier: nil,
            description: "Search posts"
        ))
        #expect(!MacOSDraftPostFlowDriver.matchesBrowserAddressField(
            browserBundleIdentifier: "com.google.Chrome",
            role: kAXButtonRole as String,
            identifier: nil,
            description: "Address and search bar"
        ))
        #expect(!MacOSDraftPostFlowDriver.matchesBrowserAddressField(
            browserBundleIdentifier: "com.example.browser",
            role: kAXTextFieldRole as String,
            identifier: "WEB_BROWSER_ADDRESS_AND_SEARCH_FIELD",
            description: "Address and search bar"
        ))
    }

    @Test("Visual recovery binds only to the exact selected UI-TARS profile")
    func visualRuntimeEligibility() throws {
        let profile = ServerModelProfile(
            id: "ui-tars-1.5-7b-4bit",
            toolCallParser: "ui_tars"
        )
        let currentSession: DraftPostVisualRuntime.SessionValidator = { true }
        let runtime = try #require(DraftPostVisualRuntime(
            profile: profile,
            selectedAlias: "UI-TARS-1.5-7B-4BIT",
            host: "127.0.0.1",
            port: 7659,
            bearerToken: "secret",
            sessionValidator: currentSession
        ))
        #expect(runtime.model == profile.id)
        #expect(runtime.baseURL.absoluteString == "http://127.0.0.1:7659/v1")

        #expect(DraftPostVisualRuntime(
            profile: profile,
            selectedAlias: "another-model",
            host: "127.0.0.1",
            port: 7659,
            bearerToken: "secret",
            sessionValidator: currentSession
        ) == nil)
        #expect(DraftPostVisualRuntime(
            profile: ServerModelProfile(id: profile.id, toolCallParser: "hermes"),
            selectedAlias: profile.id,
            host: "127.0.0.1",
            port: 7659,
            bearerToken: "secret",
            sessionValidator: currentSession
        ) == nil)
        #expect(DraftPostVisualRuntime(
            profile: profile,
            selectedAlias: profile.id,
            host: "localhost",
            port: 7659,
            bearerToken: "secret",
            sessionValidator: currentSession
        ) == nil)
        #expect(DraftPostVisualRuntime(
            profile: profile,
            selectedAlias: profile.id,
            host: "127.0.0.1",
            port: 7659,
            bearerToken: nil,
            sessionValidator: currentSession
        ) == nil)
    }

    @Test("Visual inference rejects a server-session change before acceptance")
    func visualInferenceRequiresOneSession() async {
        let session = VisualSessionProbe(isCurrent: true)
        let validator: DraftPostVisualRuntime.SessionValidator = {
            session.current
        }
        let base = RotatingDraftPostGroundingTransport(session: session)
        let transport = SessionValidatedComputerUseGroundingTransport(
            base: base,
            sessionValidator: validator
        )
        await #expect(throws: DraftPostFlowFailure.dependencyFailure) {
            _ = try await transport.send(
                URLRequest(url: URL(string: "http://127.0.0.1:7659/v1")!),
                maximumResponseBytes: 128
            )
        }
        #expect(await base.callCount == 1)

        let blockedBase = RotatingDraftPostGroundingTransport(session: session)
        let blockedTransport = SessionValidatedComputerUseGroundingTransport(
            base: blockedBase,
            sessionValidator: validator
        )
        await #expect(throws: DraftPostFlowFailure.dependencyFailure) {
            _ = try await blockedTransport.send(
                URLRequest(url: URL(string: "http://127.0.0.1:7659/v1")!),
                maximumResponseBytes: 128
            )
        }
        #expect(await blockedBase.callCount == 0)
    }

    @MainActor
    @Test("A retained visual runtime rejects a restarted server with reused connection values")
    func visualRuntimeRejectsRestartedServerSession() async throws {
        let alias = "ui-tars-1.5-7b-4bit"
        let bearer = "persisted-secret"
        let profile = ServerModelProfile(id: alias, toolCallParser: "ui_tars")
        let server = ServerManager(
            testingState: .ready(alias: alias),
            activePort: 7659,
            activeBearer: bearer
        )
        server.applyActiveModelProfile(profile, forAlias: alias)
        let runtime = try #require(DraftPostVisualRuntime(
            profile: profile,
            selectedAlias: alias,
            host: server.host,
            port: server.activePort,
            bearerToken: bearer,
            liveServer: server
        ))
        let originalSessionID = try #require(runtime.sessionID)
        #expect(await runtime.isCurrentSession())

        server._testReplaceActiveServerSession(bearer: bearer)
        #expect(server.activeServerSessionID != originalSessionID)
        server.applyActiveModelProfile(profile, forAlias: alias)

        #expect(!(await runtime.isCurrentSession()))
    }

    @MainActor
    @Test("A long-lived embedded credential cannot authorize visual recovery")
    func persistentCredentialIsIneligibleForVisualRecovery() throws {
        let alias = "ui-tars-1.5-7b-4bit"
        let profile = ServerModelProfile(id: alias, toolCallParser: "ui_tars")
        let server = ServerManager(
            testingState: .ready(alias: alias),
            activePort: 7659,
            activeBearer: "persisted-secret"
        )
        server.applyActiveModelProfile(profile, forAlias: alias)
        server.setEmbeddedBearerLifetime(.daily)

        #expect(DraftPostVisualRuntime(
            profile: profile,
            selectedAlias: alias,
            host: server.host,
            port: server.activePort,
            bearerToken: server.activeBearer,
            liveServer: server
        ) == nil)
    }

    @Test("Chrome accessibility activation balances cancellation")
    func browserAccessibilityActivationBalancesCancellation() async {
        let probe = BrowserAccessibilityLeaseProbe()
        await #expect(throws: CancellationError.self) {
            _ = try await MacOSDraftPostFlowDriver.establishBrowserAccessibilityLease(
                previousValue: true,
                activate: { probe.activate() },
                settle: { throw CancellationError() },
                restore: { probe.restore(to: $0) }
            )
        }
        #expect(probe.activations == 1)
        #expect(probe.releases == 1)
        #expect(probe.restoredValue == true)
    }

    @Test("Post-visual focus drift cannot start another visual budget")
    func postVisualDriftIsTerminal() async {
        await #expect(throws: DraftPostFlowFailure.verificationFailed) {
            try await MacOSDraftPostFlowDriver.verifyAfterVisualRecovery {
                throw DraftPostFlowFailure.focusChanged
            }
        }
    }

    @Test("Visual coordinates can authorize only the expected editable value")
    func safeVisualComposerContract() {
        let accepted = MacOSDraftPostFlowDriver.matchesSafeVisualComposer(
            role: kAXTextAreaRole as String,
            subrole: nil,
            isBrowserAddressField: false,
            isEnabled: true,
            value: "",
            requiredValue: "",
            isValueSettable: true
        )
        #expect(accepted)

        for rejected in [
            MacOSDraftPostFlowDriver.matchesSafeVisualComposer(
                role: kAXButtonRole as String,
                subrole: nil,
                isBrowserAddressField: false,
                isEnabled: true,
                value: "",
                requiredValue: "",
                isValueSettable: true
            ),
            MacOSDraftPostFlowDriver.matchesSafeVisualComposer(
                role: kAXTextFieldRole as String,
                subrole: nil,
                isBrowserAddressField: true,
                isEnabled: true,
                value: "",
                requiredValue: "",
                isValueSettable: true
            ),
            MacOSDraftPostFlowDriver.matchesSafeVisualComposer(
                role: kAXTextAreaRole as String,
                subrole: kAXSecureTextFieldSubrole as String,
                isBrowserAddressField: false,
                isEnabled: true,
                value: "",
                requiredValue: "",
                isValueSettable: true
            ),
            MacOSDraftPostFlowDriver.matchesSafeVisualComposer(
                role: kAXTextAreaRole as String,
                subrole: nil,
                isBrowserAddressField: false,
                isEnabled: true,
                value: "already present",
                requiredValue: "",
                isValueSettable: true
            ),
        ] {
            #expect(!rejected)
        }
    }

    @Test("Visual grounding tolerance is local and window bounded")
    func visualGroundingTolerance() {
        let window = CGRect(x: 100, y: 100, width: 1_000, height: 800)
        let composer = CGRect(x: 300, y: 350, width: 600, height: 260)
        #expect(MacOSDraftPostFlowDriver.isWithinGroundingTolerance(
            point: CGPoint(x: 600, y: 320),
            elementFrame: composer,
            windowFrame: window
        ))
        #expect(!MacOSDraftPostFlowDriver.isWithinGroundingTolerance(
            point: CGPoint(x: 600, y: 280),
            elementFrame: composer,
            windowFrame: window
        ))
        #expect(!MacOSDraftPostFlowDriver.isWithinGroundingTolerance(
            point: CGPoint(x: 99, y: 400),
            elementFrame: composer,
            windowFrame: window
        ))
        #expect(MacOSDraftPostFlowDriver.authorizesUniqueVisualEditor(
            candidateCount: 1
        ))
        #expect(!MacOSDraftPostFlowDriver.authorizesUniqueVisualEditor(
            candidateCount: 0
        ))
        #expect(!MacOSDraftPostFlowDriver.authorizesUniqueVisualEditor(
            candidateCount: 2
        ))
    }

    @Test("Visual recovery retries only the bounded pre-mutation attempt")
    func visualRecoveryIsBounded() async throws {
        let script = ScriptedVisualRecoveryAttempt(failuresBeforeSuccess: 2)
        let recovery = MacOSDraftPostVisualRecovery { destination, identity in
            #expect(destination.id == Self.destination.id)
            #expect(identity == "bound-document")
            try await script.run()
        }

        try await recovery.focusComposer(
            in: Self.destination,
            documentIdentity: "bound-document"
        )

        #expect(await script.callCount == 3)
    }

    @Test("Visual recovery exhaustion fails closed after three attempts")
    func visualRecoveryExhaustion() async {
        let script = ScriptedVisualRecoveryAttempt(failuresBeforeSuccess: 4)
        let recovery = MacOSDraftPostVisualRecovery { _, _ in
            try await script.run()
        }

        await #expect(throws: DraftPostFlowFailure.composerMissing) {
            try await recovery.focusComposer(
                in: Self.destination,
                documentIdentity: "bound-document"
            )
        }
        #expect(await script.callCount == 3)
    }

    @Test("Cancellation is never converted into a visual retry")
    func visualRecoveryCancellation() async {
        let script = ScriptedVisualRecoveryAttempt(
            failuresBeforeSuccess: 0,
            cancels: true
        )
        let recovery = MacOSDraftPostVisualRecovery { _, _ in
            try await script.run()
        }

        await #expect(throws: CancellationError.self) {
            try await recovery.focusComposer(
                in: Self.destination,
                documentIdentity: "bound-document"
            )
        }
        #expect(await script.callCount == 1)
    }

    @Test("Post-mutation drift is always terminal")
    func postMutationDriftStops() throws {
        try MacOSDraftPostFlowDriver.verifyAfterMutation { true }
        #expect(throws: DraftPostFlowFailure.verificationFailed) {
            try MacOSDraftPostFlowDriver.verifyAfterMutation {
                throw DraftPostFlowFailure.focusChanged
            }
        }
    }

    @Test("Verification compares exact UTF-8 bytes")
    func exactUTF8Verification() {
        #expect(MacOSDraftPostFlowDriver.utf8Matches("draft", "draft"))
        #expect(!MacOSDraftPostFlowDriver.utf8Matches("é", "e\u{301}"))
    }

    @Test("Browser tab identity stays bound to the selected title")
    func browserTabIdentity() {
        #expect(MacOSDraftPostFlowDriver.browserDocumentMatches(
            currentTitle: "Compose — Account A",
            selectedTitle: "Compose — Account A"
        ))
        #expect(!MacOSDraftPostFlowDriver.browserDocumentMatches(
            currentTitle: "Compose — Account B",
            selectedTitle: "Compose — Account A"
        ))
        #expect(!MacOSDraftPostFlowDriver.browserDocumentMatches(
            currentTitle: "Compose",
            selectedTitle: ""
        ))
        #expect(MacOSDraftPostFlowDriver.browserWindowTitleMatches(
            browserBundleIdentifier: "com.google.Chrome",
            currentTitle: "Compose — Account A - Google Chrome",
            selectedTitle: "Compose — Account A"
        ))
        #expect(!MacOSDraftPostFlowDriver.browserWindowTitleMatches(
            browserBundleIdentifier: "com.apple.Safari",
            currentTitle: "Compose — Account A - Google Chrome",
            selectedTitle: "Compose — Account A"
        ))
        #expect(!MacOSDraftPostFlowDriver.browserWindowTitleMatches(
            browserBundleIdentifier: "com.google.Chrome",
            currentTitle: "Other - Google Chrome",
            selectedTitle: "Compose — Account A"
        ))
    }

    @Test("TextEdit source must expose one document editor")
    func uniqueDraft() throws {
        #expect(try MacOSDraftPostFlowDriver.uniqueDraft(in: ["Draft"]) == "Draft")
        #expect(throws: DraftPostFlowFailure.draftMissing) {
            try MacOSDraftPostFlowDriver.uniqueDraft(in: [])
        }
        #expect(throws: DraftPostFlowFailure.draftMissing) {
            try MacOSDraftPostFlowDriver.uniqueDraft(in: [nil])
        }
        #expect(throws: DraftPostFlowFailure.draftAmbiguous) {
            try MacOSDraftPostFlowDriver.uniqueDraft(in: ["Draft", ""])
        }
    }

    @MainActor
    @Test("Catalog refresh clears stale selections")
    func refreshClearsSelections() async {
        let catalog = StaticWindowCatalog(windows: [Self.source, Self.destination])
        let viewModel = DraftPostFlowViewModel(
            catalog: catalog,
            driver: ScriptedDraftPostDriver(results: [.success(())])
        )
        await viewModel.load()
        #expect(viewModel.sourceID == nil)
        viewModel.sourceID = Self.source.id
        viewModel.destinationID = Self.destination.id
        #expect(viewModel.canRun)

        await catalog.replace(with: [Self.source])
        await viewModel.load()
        #expect(viewModel.destinationID == nil)
        #expect(!viewModel.canRun)
    }

    @MainActor
    @Test("Untitled windows are not offered to the flow")
    func untitledWindowsAreExcluded() async {
        let untitledBrowser = ComputerUseWindowOption(
            id: "2:21",
            applicationName: "Safari",
            windowTitle: "   ",
            selection: ComputerUseWindowSelection(
                bundleIdentifier: "com.apple.Safari",
                processIdentifier: 2,
                processLaunchDate: Date(timeIntervalSince1970: 1),
                windowID: 21
            )
        )
        let untitledSource = ComputerUseWindowOption(
            id: "1:11",
            applicationName: "TextEdit",
            windowTitle: "",
            selection: ComputerUseWindowSelection(
                bundleIdentifier: "com.apple.TextEdit",
                processIdentifier: 1,
                processLaunchDate: Date(timeIntervalSince1970: 1),
                windowID: 11
            )
        )
        let viewModel = DraftPostFlowViewModel(
            catalog: StaticWindowCatalog(windows: [
                Self.source, untitledSource, Self.destination, untitledBrowser,
            ]),
            driver: ScriptedDraftPostDriver(results: [.success(())])
        )

        await viewModel.load()

        #expect(viewModel.sourceOptions.map(\.id) == [Self.source.id])
        #expect(viewModel.destinationOptions.map(\.id) == [Self.destination.id])
    }

    @MainActor
    @Test("An older window refresh cannot overwrite a newer result")
    func staleRefreshIsIgnored() async {
        let catalog = OutOfOrderWindowCatalog(
            older: [Self.source],
            newer: [Self.destination]
        )
        let viewModel = DraftPostFlowViewModel(
            catalog: catalog,
            driver: ScriptedDraftPostDriver(results: [.success(())])
        )

        let olderLoad = Task { await viewModel.load() }
        while await catalog.callCount == 0 {
            await Task.yield()
        }
        let newerLoad = Task { await viewModel.load() }
        await newerLoad.value
        await olderLoad.value

        #expect(viewModel.windows == [Self.destination])
        #expect(viewModel.phase == .ready)
    }

    @MainActor
    @Test("A late cancellation reports the definitive driver outcome")
    func lateCancellationIsHonest() async {
        let driver = CancellationIgnoringDraftPostDriver()
        let viewModel = DraftPostFlowViewModel(
            catalog: StaticWindowCatalog(windows: [Self.source, Self.destination]),
            driver: driver
        )
        await viewModel.load()
        viewModel.sourceID = Self.source.id
        viewModel.destinationID = Self.destination.id
        viewModel.run()
        while !(await driver.didStart) {
            await Task.yield()
        }
        viewModel.stop()
        #expect(viewModel.phase == .stopping)
        await driver.complete()
        for _ in 0 ..< 100 where viewModel.phase == .stopping {
            await Task.yield()
        }
        guard case .readyForReview = viewModel.phase else {
            Issue.record("A completed mutation was incorrectly reported as cancelled")
            return
        }
    }

    @Test("The coordinator can request only a draft transfer")
    func coordinatorCapabilityIsLimited() async {
        let driver = ScriptedDraftPostDriver(results: [.success(())])
        let coordinator = DraftPostFlowCoordinator(driver: driver)

        let outcome = await coordinator.run(
            source: Self.source,
            destination: Self.destination
        )

        #expect(outcome == .readyForReview(DraftPostFlowMetrics(
            attempts: 1,
            automaticRecoveries: 0,
            completedSteps: 3
        )))
        #expect(await driver.actions == [
            .transferDraft(sourceID: Self.source.id, destinationID: Self.destination.id)
        ])
    }

    @MainActor
    @Test("The injected actuator records only focus and draft writes")
    func actuatorCapabilityIsLimited() throws {
        let actuator = RecordingDraftPostComposerActuator(
            base: NoopDraftPostComposerActuator()
        )
        let element = AXUIElementCreateSystemWide()
        try actuator.focusComposer(element)
        try actuator.setDraft("Fixture", on: element)
        #expect(actuator.actions == [.focusComposer, .setDraft("Fixture")])
    }

    @MainActor
    @Test("Run accepts only one active transfer")
    func singleActiveTransfer() async {
        let driver = CancellationIgnoringDraftPostDriver()
        let viewModel = DraftPostFlowViewModel(
            catalog: StaticWindowCatalog(windows: [Self.source, Self.destination]),
            driver: driver
        )
        await viewModel.load()
        viewModel.sourceID = Self.source.id
        viewModel.destinationID = Self.destination.id
        viewModel.run()
        viewModel.run()
        while !(await driver.didStart) {
            await Task.yield()
        }
        #expect(await driver.callCount == 1)
        await driver.complete()
    }

    private static let source = option(
        id: "1:10",
        application: "TextEdit",
        bundle: "com.apple.TextEdit",
        pid: 1,
        window: 10
    )
    private static let destination = option(
        id: "2:20",
        application: "Safari",
        bundle: "com.apple.Safari",
        pid: 2,
        window: 20
    )

    private static func option(
        id: String,
        application: String,
        bundle: String,
        pid: pid_t,
        window: CGWindowID
    ) -> ComputerUseWindowOption {
        ComputerUseWindowOption(
            id: id,
            applicationName: application,
            windowTitle: "Fixture",
            selection: ComputerUseWindowSelection(
                bundleIdentifier: bundle,
                processIdentifier: pid,
                processLaunchDate: Date(timeIntervalSince1970: 1),
                windowID: window
            )
        )
    }

    private static func liveOptions() async throws -> (
        ComputerUseWindowOption,
        ComputerUseWindowOption
    ) {
        let windows = try await MacOSComputerUseWindowCatalog().windows()
        let source = try #require(windows.first {
            $0.selection.bundleIdentifier == "com.apple.TextEdit"
                && $0.windowTitle.contains("rapid-cua-draft.txt")
        })
        let destination = try #require(windows.first {
            $0.selection.bundleIdentifier == liveBrowserBundle
                && $0.windowTitle.contains(liveWindowTitle)
        })
        return (source, destination)
    }

    private static var liveBrowserBundle: String {
        ProcessInfo.processInfo.environment["RAPID_LIVE_CUA_BROWSER_BUNDLE"]
            ?? "com.apple.Safari"
    }

    private static var liveWindowTitle: String {
        ProcessInfo.processInfo.environment["RAPID_LIVE_CUA_WINDOW_TITLE"]
            ?? "Local Post Composer"
    }

}

private actor RecordingDraftPostGroundingTransport:
    LocalComputerUseGroundingTransport
{
    private let base = URLSessionComputerUseGroundingTransport()
    private(set) var responseBodies: [String] = []

    func send(
        _ request: URLRequest,
        maximumResponseBytes: Int
    ) async throws -> LocalComputerUseGroundingHTTPResponse {
        let response = try await base.send(
            request,
            maximumResponseBytes: maximumResponseBytes
        )
        responseBodies.append(String(decoding: response.body, as: UTF8.self))
        return response
    }
}

private actor ScriptedVisualRecoveryAttempt {
    private let failuresBeforeSuccess: Int
    private let cancels: Bool
    private(set) var callCount = 0

    init(failuresBeforeSuccess: Int, cancels: Bool = false) {
        self.failuresBeforeSuccess = failuresBeforeSuccess
        self.cancels = cancels
    }

    func run() throws {
        callCount += 1
        if cancels { throw CancellationError() }
        if callCount <= failuresBeforeSuccess {
            throw DraftPostFlowFailure.targetUnavailable
        }
    }
}

private final class VisualSessionProbe: @unchecked Sendable {
    private let lock = NSLock()
    private var value: Bool

    init(isCurrent: Bool) {
        self.value = isCurrent
    }

    var current: Bool {
        get { lock.withLock { value } }
        set { lock.withLock { value = newValue } }
    }
}

private final class BrowserAccessibilityLeaseProbe: @unchecked Sendable {
    private let lock = NSLock()
    private var activationCount = 0
    private var releaseCount = 0
    private var restored: Bool?

    var activations: Int { lock.withLock { activationCount } }
    var releases: Int { lock.withLock { releaseCount } }
    var restoredValue: Bool? { lock.withLock { restored } }

    func activate() { lock.withLock { activationCount += 1 } }
    func restore(to value: Bool) {
        lock.withLock {
            releaseCount += 1
            restored = value
        }
    }
}

private actor RotatingDraftPostGroundingTransport:
    LocalComputerUseGroundingTransport
{
    private let session: VisualSessionProbe
    private(set) var callCount = 0

    init(session: VisualSessionProbe) {
        self.session = session
    }

    func send(
        _: URLRequest,
        maximumResponseBytes _: Int
    ) async throws -> LocalComputerUseGroundingHTTPResponse {
        callCount += 1
        session.current = false
        return LocalComputerUseGroundingHTTPResponse(
            statusCode: 200,
            contentType: "application/json",
            body: Data()
        )
    }
}

private enum RecordedDraftPostComposerAction: Equatable {
    case focusComposer
    case setDraft(String)
}

private final class RecordingDraftPostComposerActuator: DraftPostComposerActuating,
    @unchecked Sendable
{
    private let base: any DraftPostComposerActuating
    private let lock = NSLock()
    private var recordedActions: [RecordedDraftPostComposerAction] = []
    private var lastWrite: (element: AXUIElement, draft: String)?

    var actions: [RecordedDraftPostComposerAction] {
        lock.withLock { recordedActions }
    }

    init(base: any DraftPostComposerActuating) {
        self.base = base
    }

    func focusComposer(_ composer: AXUIElement) throws {
        lock.withLock { recordedActions.append(.focusComposer) }
        try base.focusComposer(composer)
    }

    func setDraft(_ draft: String, on composer: AXUIElement) throws {
        lock.withLock { recordedActions.append(.setDraft(draft)) }
        try base.setDraft(draft, on: composer)
        lock.withLock { lastWrite = (composer, draft) }
    }

    /// The live test resets only the exact element that this actuator wrote,
    /// and only while its bytes still match that write. It never searches or
    /// clears a process's currently focused control.
    func clearLastDraftIfUnchanged() throws {
        let write = lock.withLock { lastWrite }
        guard let write else {
            throw DraftPostFlowFailure.verificationFailed
        }
        var value: CFTypeRef?
        guard AXUIElementCopyAttributeValue(
            write.element,
            kAXValueAttribute as CFString,
            &value
        ) == .success,
            let current = value as? String,
            MacOSDraftPostFlowDriver.utf8Matches(current, write.draft),
            AXUIElementSetAttributeValue(
                write.element,
                kAXValueAttribute as CFString,
                "" as CFString
            ) == .success
        else {
            throw DraftPostFlowFailure.verificationFailed
        }
        // Chrome may retire and replace a renderer AX node immediately after
        // clearing it. The next full flow iteration re-resolves the selected
        // window and proves that its composer is empty; retaining the stale
        // node here would turn successful cleanup into a false failure.
        lock.withLock { lastWrite = nil }
    }
}

private struct NoopDraftPostComposerActuator: DraftPostComposerActuating {
    func focusComposer(_: AXUIElement) throws {}
    func setDraft(_: String, on _: AXUIElement) throws {}
}

private enum ScriptedDraftPostAction: Equatable, Sendable {
    case transferDraft(sourceID: String, destinationID: String)
}

private actor ScriptedDraftPostDriver: DraftPostFlowDriving {
    private var results: [Result<Void, DraftPostFlowFailure>]
    private(set) var callCount = 0
    private(set) var actions: [ScriptedDraftPostAction] = []

    init(results: [Result<Void, DraftPostFlowFailure>]) {
        self.results = results
    }

    func transferDraft(
        from source: ComputerUseWindowOption,
        to destination: ComputerUseWindowOption
    ) async throws {
        callCount += 1
        actions.append(.transferDraft(
            sourceID: source.id,
            destinationID: destination.id
        ))
        guard !results.isEmpty else {
            throw DraftPostFlowFailure.targetUnavailable
        }
        try results.removeFirst().get()
    }
}

private actor FocusStealingDraftPostDriver: DraftPostFlowDriving {
    private let base: any DraftPostFlowDriving
    private var shouldSteal = true

    init(base: any DraftPostFlowDriving) {
        self.base = base
    }

    func transferDraft(
        from source: ComputerUseWindowOption,
        to destination: ComputerUseWindowOption
    ) async throws {
        if shouldSteal {
            shouldSteal = false
            Task { @MainActor in
                try? await Task.sleep(for: .milliseconds(90))
                NSRunningApplication.runningApplications(
                    withBundleIdentifier: "com.apple.finder"
                ).first?.activate()
            }
        }
        try await base.transferDraft(from: source, to: destination)
    }
}

private actor UnexpectedDraftPostDriver: DraftPostFlowDriving {
    private(set) var callCount = 0

    func transferDraft(
        from _: ComputerUseWindowOption,
        to _: ComputerUseWindowOption
    ) async throws {
        callCount += 1
        throw CocoaError(.fileReadUnknown)
    }
}

private actor CancellationIgnoringDraftPostDriver: DraftPostFlowDriving {
    private(set) var didStart = false
    private(set) var callCount = 0
    private var mayComplete = false

    func transferDraft(
        from _: ComputerUseWindowOption,
        to _: ComputerUseWindowOption
    ) async throws {
        callCount += 1
        didStart = true
        while !mayComplete {
            await Task.yield()
        }
        // Models a cancellation that arrives after the mutation boundary: the
        // operation has completed and its success must remain observable.
    }

    func complete() {
        mayComplete = true
    }
}

private actor StaticWindowCatalog: ComputerUseWindowListing {
    private var storedWindows: [ComputerUseWindowOption]

    init(windows: [ComputerUseWindowOption]) {
        storedWindows = windows
    }

    func windows() async throws -> [ComputerUseWindowOption] {
        storedWindows
    }

    func replace(with windows: [ComputerUseWindowOption]) {
        storedWindows = windows
    }
}

private actor OutOfOrderWindowCatalog: ComputerUseWindowListing {
    private let older: [ComputerUseWindowOption]
    private let newer: [ComputerUseWindowOption]
    private(set) var callCount = 0

    init(older: [ComputerUseWindowOption], newer: [ComputerUseWindowOption]) {
        self.older = older
        self.newer = newer
    }

    func windows() async throws -> [ComputerUseWindowOption] {
        callCount += 1
        if callCount == 1 {
            try await Task.sleep(for: .milliseconds(100))
            return older
        }
        return newer
    }
}
