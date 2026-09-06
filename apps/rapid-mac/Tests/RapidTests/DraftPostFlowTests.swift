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
    /// - Browser window title contains `Rapid Computer Use Fixture` and has
    ///   one empty text area labelled `Post text`.
    ///
    /// Run with `RAPID_LIVE_CUA_DOGFOOD=1 swift test --no-parallel --filter
    /// DraftPostFlowTests/liveFixture` from a trusted GUI session.
    @Test(.enabled(if: ProcessInfo.processInfo.environment["RAPID_LIVE_CUA_DOGFOOD"] == "1"))
    func liveFixture() async throws {
        let (source, destination) = try await Self.liveOptions()
        try await Self.clearLiveComposer(
            processIdentifier: destination.selection.processIdentifier
        )
        var successes = 0
        for _ in 0 ..< 30 {
            let outcome = await DraftPostFlowCoordinator(
                driver: MacOSDraftPostFlowDriver()
            ).run(source: source, destination: destination)
            guard case .readyForReview(let metrics) = outcome else {
                Issue.record("Live fixture did not reach review: \(outcome)")
                continue
            }
            successes += 1
            #expect(metrics.attempts <= 3)
            #expect(metrics.completedSteps == 3)
            try await Self.clearLiveComposer(processIdentifier: destination.selection.processIdentifier)
        }
        #expect(successes == 30)
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["RAPID_LIVE_CUA_DOGFOOD"] == "1"))
    func liveFocusStealRecovery() async throws {
        let (source, destination) = try await Self.liveOptions()
        let driver = FocusStealingDraftPostDriver(base: MacOSDraftPostFlowDriver())
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
    }

    @Test("The first runnable starter is draft and post")
    func catalogAvailability() throws {
        let available = ComputerUseStarter.catalog.filter {
            $0.availability == .available
        }
        #expect(available.map(\.kind) == [.draftAndPost])
        #expect(try #require(available.first).approvalNote == "Rapid will stop before publishing.")
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
            .failure(.targetUnavailable),
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

    @Test("The driver contract exposes no publish action")
    func noPublishSurface() throws {
        let source = try String(
            contentsOf: Self.sourceFile("DraftPostFlow.swift"),
            encoding: .utf8
        )
        let protocolSlice = try #require(source.range(
            of: "protocol DraftPostFlowDriving: Sendable"
        )).lowerBound ..< #require(source.range(
            of: "/// Runs one idempotent local transfer"
        )).lowerBound
        let contract = String(source[protocolSlice])
        #expect(contract.contains("transferDraft"))
        #expect(!contract.lowercased().contains("publish"))
        #expect(!contract.lowercased().contains("post("))
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

    private static func sourceFile(_ name: String) -> URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Sources/Rapid/ComputerUse")
            .appendingPathComponent(name)
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
            MacOSDraftPostFlowDriver.browserBundles.contains(
                $0.selection.bundleIdentifier
            ) && $0.windowTitle.contains("Rapid Computer Use Fixture")
        })
        return (source, destination)
    }

    @MainActor
    private static func clearLiveComposer(processIdentifier: pid_t) throws {
        let application = AXUIElementCreateApplication(processIdentifier)
        var focusedValue: CFTypeRef?
        guard AXUIElementCopyAttributeValue(
            application,
            kAXFocusedWindowAttribute as CFString,
            &focusedValue
        ) == .success,
            let window = focusedValue,
            CFGetTypeID(window) == AXUIElementGetTypeID()
        else {
            throw DraftPostFlowFailure.verificationFailed
        }
        var queue = [unsafeDowncast(window, to: AXUIElement.self)]
        var cursor = 0
        while cursor < queue.count, cursor < 2_048 {
            let element = queue[cursor]
            cursor += 1
            let labels = [
                liveStringAttribute(kAXTitleAttribute as CFString, from: element),
                liveStringAttribute(kAXDescriptionAttribute as CFString, from: element),
                liveStringAttribute(kAXHelpAttribute as CFString, from: element),
                liveStringAttribute("AXPlaceholderValue" as CFString, from: element),
            ].compactMap { $0 }
            if liveStringAttribute(kAXRoleAttribute as CFString, from: element)
                == kAXTextAreaRole as String,
                labels.contains(where: MacOSDraftPostFlowDriver.isExplicitComposerLabel)
            {
                guard AXUIElementSetAttributeValue(
                    element,
                    kAXValueAttribute as CFString,
                    "" as CFString
                ) == .success,
                    liveStringAttribute(kAXValueAttribute as CFString, from: element) == ""
                else {
                    throw DraftPostFlowFailure.verificationFailed
                }
                return
            }
            var childrenValue: CFTypeRef?
            if AXUIElementCopyAttributeValue(
                element,
                kAXChildrenAttribute as CFString,
                &childrenValue
            ) == .success,
                let children = childrenValue as? [AXUIElement]
            {
                queue.append(contentsOf: children)
            }
        }
        throw DraftPostFlowFailure.composerMissing
    }

    @MainActor
    private static func liveStringAttribute(
        _ attribute: CFString,
        from element: AXUIElement
    ) -> String? {
        var value: CFTypeRef?
        guard AXUIElementCopyAttributeValue(element, attribute, &value) == .success else {
            return nil
        }
        return value as? String
    }
}

private actor ScriptedDraftPostDriver: DraftPostFlowDriving {
    private var results: [Result<Void, DraftPostFlowFailure>]
    private(set) var callCount = 0

    init(results: [Result<Void, DraftPostFlowFailure>]) {
        self.results = results
    }

    func transferDraft(
        from _: ComputerUseWindowOption,
        to _: ComputerUseWindowOption
    ) async throws {
        callCount += 1
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
    private var mayComplete = false

    func transferDraft(
        from _: ComputerUseWindowOption,
        to _: ComputerUseWindowOption
    ) async throws {
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
