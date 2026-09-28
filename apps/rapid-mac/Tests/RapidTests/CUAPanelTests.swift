import Foundation
import Testing
@testable import Rapid

// MARK: - Fakes

private final class MockAgentAPI: CUAAPI, @unchecked Sendable {
    enum Failure: Error { case requested }

    var plannersResult: [CUAPlannerOption] = []
    var appsResult = [CUAAppOption(name: "Google Chrome", bundleID: "com.google.Chrome", pid: 42)]
    var windowsResult = [
        CUAWindowOption(
            windowID: "cg:123", index: 0, title: "Research — Apple Silicon",
            x: 0, y: 0, width: 1200, height: 800
        ),
    ]
    var discoveryShouldFail = false
    var discoveryError: Error?
    var createError: Error?
    var queuedCreateErrors: [Error] = []
    var createAttempts = 0
    var capabilitiesSupported = true
    var capabilitiesDelayNanos: UInt64 = 0
    var lookupResult: CUARunCreated?
    var lookupHandler: ((String) -> CUARunCreated)?
    var lookupError: Error?
    var createDelayNanos: UInt64 = 0
    var approveDelayNanos: UInt64 = 0
    var cancelDelayNanos: UInt64 = 0
    var eventsDelayNanos: UInt64 = 0
    var createdRequests: [CUARunRequest] = []
    var attemptedRequests: [CUARunRequest] = []
    var scriptedEvents: [CUAEvent] = []
    var finalSummary = "opened the article"
    var approveCalls = 0
    var approvedGateIDs: [String?] = []
    var cancelCalls = 0
    var runStatus = "running"
    var approveShouldFail = false
    var cancelShouldFail = false
    var eventsShouldFail = false
    var eventsCalls = 0
    var permissionsResult = CUAPermissionStatus(accessibility: true, screenRecording: true)
    var pendingGateResult: CUAPendingGate?

    var addedPlanners: [CUAPlannerCreateRequest] = []
    var deletedPlannerNames: [String] = []
    var addShouldFail = false

    func capabilities() async throws -> CUACapabilities {
        if capabilitiesDelayNanos > 0 {
            try? await Task.sleep(nanoseconds: capabilitiesDelayNanos)
        }
        return CUACapabilities(
            features: .init(idempotentRunCreate: capabilitiesSupported)
        )
    }

    func planners() async throws -> [CUAPlannerOption] {
        plannersResult
    }

    func apps() async throws -> [CUAAppOption] {
        if let discoveryError { throw discoveryError }
        if discoveryShouldFail { throw Failure.requested }
        return appsResult
    }

    func windows(app: String) async throws -> [CUAWindowOption] {
        if let discoveryError { throw discoveryError }
        if discoveryShouldFail { throw Failure.requested }
        return windowsResult
    }

    func addPlanner(_ request: CUAPlannerCreateRequest) async throws {
        if addShouldFail { throw Failure.requested }
        addedPlanners.append(request)
    }

    func deletePlanner(name: String) async throws {
        deletedPlannerNames.append(name)
    }

    func create(_ request: CUARunRequest) async throws -> String {
        createAttempts += 1
        attemptedRequests.append(request)
        if createDelayNanos > 0 { try? await Task.sleep(nanoseconds: createDelayNanos) }
        if !queuedCreateErrors.isEmpty { throw queuedCreateErrors.removeFirst() }
        if let createError { throw createError }
        createdRequests.append(request)
        return "run123"
    }

    func run(clientRequestID: String) async throws -> CUARunCreated {
        if let lookupError { throw lookupError }
        if let lookupHandler { return lookupHandler(clientRequestID) }
        if let lookupResult { return lookupResult }
        throw CUAClientError.typedHTTP(
            404, code: "request_identity_not_found", message: "not found", recovery: []
        )
    }

    func permissions() async throws -> CUAPermissionStatus {
        permissionsResult
    }

    func events(runID: String, after: Int) async throws -> CUARunView {
        eventsCalls += 1
        if eventsDelayNanos > 0 { try? await Task.sleep(nanoseconds: eventsDelayNanos) }
        if eventsShouldFail { throw Failure.requested }
        var events: [CUAEvent] = []
        if after == 0 {
            events.append(CUAEvent(seq: 1, kind: "started", step: nil, action: nil, stepInstruction: nil, outcome: nil, targetLabel: nil, status: nil, finalSummary: nil, reason: nil))
            events.append(contentsOf: scriptedEvents.enumerated().map { index, event in
                CUAEvent(
                    seq: index + 2, kind: event.kind, step: event.step,
                    action: event.action, stepInstruction: event.stepInstruction,
                    outcome: event.outcome, targetLabel: event.targetLabel,
                    status: event.status, finalSummary: event.finalSummary,
                    reason: event.reason, error: event.error, app: event.app,
                    gateID: event.gateID, target: event.target
                )
            })
        }
        return CUARunView(
            runID: runID,
            app: "Google Chrome",
            goal: "g",
            status: runStatus,
            finalSummary: finalSummary,
            error: "",
            planner: "local-9b [local]",
            eventsAfterSeq: after,
            events: events,
            pendingGate: pendingGateResult
        )
    }

    func approve(runID: String, gateID: String) async throws {
        approveCalls += 1
        approvedGateIDs.append(gateID)
        if approveDelayNanos > 0 { try? await Task.sleep(nanoseconds: approveDelayNanos) }
        if approveShouldFail { throw Failure.requested }
    }

    func cancel(runID: String) async throws {
        cancelCalls += 1
        if cancelDelayNanos > 0 { try? await Task.sleep(nanoseconds: cancelDelayNanos) }
        if cancelShouldFail { throw Failure.requested }
    }
}

private actor WindowDiscoveryRaceAPI: CUAAPI {
    private var windowCall = 0

    func capabilities() async throws -> CUACapabilities {
        CUACapabilities(features: .init(idempotentRunCreate: true))
    }

    func apps() async throws -> [CUAAppOption] { [] }

    func windows(app: String) async throws -> [CUAWindowOption] {
        windowCall += 1
        let call = windowCall
        try await Task.sleep(nanoseconds: call == 1 ? 50_000_000 : 1_000_000)
        return [
            CUAWindowOption(
                windowID: call == 1 ? "cg:old" : "cg:new", index: 0,
                title: call == 1 ? "Old" : "New", x: 0, y: 0, width: 900, height: 700
            ),
        ]
    }

    func planners() async throws -> [CUAPlannerOption] { [] }
    func addPlanner(_ request: CUAPlannerCreateRequest) async throws {}
    func deletePlanner(name: String) async throws {}
    func create(_ request: CUARunRequest) async throws -> String { "run" }
    func run(clientRequestID: String) async throws -> CUARunCreated {
        throw CUAClientError.http(404, "not found")
    }
    func permissions() async throws -> CUAPermissionStatus {
        CUAPermissionStatus(accessibility: true, screenRecording: true)
    }
    func events(runID: String, after: Int) async throws -> CUARunView {
        CUARunView(
            runID: runID, app: "", goal: "", status: "running", finalSummary: "",
            error: "", planner: "", eventsAfterSeq: after, events: []
        )
    }
    func approve(runID: String, gateID: String) async throws {}
    func cancel(runID: String) async throws {}
}

private func drain() async {
    await Task.yield()
    try? await Task.sleep(nanoseconds: 50_000_000)
    await Task.yield()
}

@MainActor
private func selectTarget(_ viewModel: CUAViewModel) {
    viewModel.appOptions = [
        CUAAppOption(name: "Google Chrome", bundleID: "com.google.Chrome", pid: 42),
    ]
    viewModel.windowOptions = [
        CUAWindowOption(
            windowID: "cg:123", index: 0, title: "Research — Apple Silicon",
            x: nil, y: nil, width: nil, height: nil
        ),
    ]
    viewModel.selectedPID = 42
    viewModel.selectedWindowID = "cg:123"
    viewModel.appName = "Google Chrome"
}

// MARK: - View model

@Suite("Agent Task Panel")
@MainActor
struct CUAViewModelTests {
    private func makeEvent(
        seq: Int, kind: String, step: Int? = nil, action: String? = nil,
        instruction: String? = nil, outcome: String? = nil,
        target: String? = nil, status: String? = nil, summary: String? = nil,
        reason: String? = nil, error: String? = nil, app: String? = nil,
        gateID: String? = nil, gateTarget: String? = nil
    ) -> CUAEvent {
        CUAEvent(
            seq: seq, kind: kind, step: step, action: action,
            stepInstruction: instruction, outcome: outcome,
            targetLabel: target, status: status, finalSummary: summary, reason: reason,
            error: error, app: app, gateID: gateID, target: gateTarget
        )
    }

    @Test("Invalidated sidecar session stops old polling before a replacement starts")
    func invalidatedSessionStopsPolling() async {
        let oldAPI = MockAgentAPI()
        let oldViewModel = CUAViewModel(api: oldAPI, pollIntervalNanos: 5_000_000)
        oldViewModel.goal = "format the document"
        selectTarget(oldViewModel)
        await oldViewModel.start()
        await drain()
        #expect(oldAPI.eventsCalls > 0)

        oldViewModel.invalidateSession()
        // A request already dispatched at invalidation may finish once. Let
        // that cancellation boundary settle before proving no retry occurs.
        await drain()
        let callsAfterInvalidation = oldAPI.eventsCalls

        let newAPI = MockAgentAPI()
        let newViewModel = CUAViewModel(api: newAPI, pollIntervalNanos: 5_000_000)
        newViewModel.goal = "new session task"
        selectTarget(newViewModel)
        await newViewModel.start()
        await drain()

        #expect(oldAPI.eventsCalls == callsAfterInvalidation)
        #expect(newAPI.eventsCalls > 0)
        #expect(oldViewModel.phase == .running)
        newViewModel.invalidateSession()
    }

    @Test("Start creates a run and finishes with the server summary")
    func startToFinished() async throws {
        let api = MockAgentAPI()
        api.scriptedEvents = [
            makeEvent(seq: 2, kind: "plan", step: 1, instruction: "click the search box"),
            makeEvent(seq: 3, kind: "executed", step: 1, outcome: "success"),
            makeEvent(seq: 4, kind: "terminal", status: "completed", summary: "opened Apple Silicon"),
        ]
        let viewModel = CUAViewModel(api: api, pollIntervalNanos: 5_000_000)
        viewModel.goal = "open Apple Silicon"
        selectTarget(viewModel)
        await viewModel.start()
        await drain()
        #expect(viewModel.phase == .finished(summary: "opened Apple Silicon"))
        #expect(api.createdRequests.count == 1)
        #expect(api.createdRequests[0].goal == "open Apple Silicon")
        #expect(api.createdRequests[0].humanLogin)
        #expect(api.createdRequests[0].app == "pid:42")
        #expect(api.createdRequests[0].windowID == "cg:123")
        #expect(viewModel.events.count == 4)
    }

    @Test("Gate event flips to awaitingApproval and approve() hits the API once")
    func gateApproval() async throws {
        let api = MockAgentAPI()
        api.scriptedEvents = [
            makeEvent(seq: 2, kind: "plan", step: 2, instruction: "fill email"),
        ]
        let viewModel = CUAViewModel(api: api, pollIntervalNanos: 5_000_000)
        viewModel.goal = "check flights"
        selectTarget(viewModel)
        await viewModel.start()
        api.scriptedEvents = [
            makeEvent(
                seq: 3, kind: "gate", action: "sign_in", target: "Account",
                reason: "sign-in", app: "Safari", gateID: "gate-7", gateTarget: "Account"
            ),
        ]
        await drain()
        #expect(viewModel.phase == .awaitingApproval)
        #expect(
            viewModel.pendingApproval == CUAPendingApproval(
                gateID: "gate-7", app: "Safari", action: "sign_in",
                target: "Account", reason: "sign-in"
            )
        )
        await viewModel.approve()
        #expect(api.approveCalls == 1)
        #expect(api.approvedGateIDs.first == "gate-7")
        #expect(viewModel.pendingApproval == nil)
    }

    @Test("Legacy gate remains usable without optional structured fields")
    func legacyGateFallback() async {
        let api = MockAgentAPI()
        api.scriptedEvents = [makeEvent(seq: 2, kind: "gate", reason: "sign-in")]
        let viewModel = CUAViewModel(api: api, pollIntervalNanos: 5_000_000)
        viewModel.goal = "check flights"
        viewModel.appName = "Google Chrome"
        selectTarget(viewModel)
        await viewModel.start()
        await drain()

        #expect(viewModel.phase == .awaitingApproval)
        #expect(viewModel.pendingApproval?.app == "Google Chrome")
        #expect(viewModel.pendingApproval?.action == nil)
        #expect(viewModel.pendingApproval?.target == nil)
        #expect(viewModel.pendingApproval?.reason == "sign-in")
        #expect(!viewModel.canApprove)
        #expect(viewModel.approvalUnavailableMessage?.contains("missing its gate identity") == true)
        await viewModel.approve()
        #expect(api.approveCalls == 0)
    }

    @Test("Approval card explains that Stop is in progress")
    func approvalCardExplainsStoppingState() async {
        let api = MockAgentAPI()
        api.pendingGateResult = CUAPendingGate(
            gateID: "gate-stop", reason: "confirm", action: "click", target: "Submit"
        )
        api.cancelDelayNanos = 50_000_000
        let viewModel = CUAViewModel(api: api, pollIntervalNanos: 5_000_000)
        viewModel.goal = "g"
        selectTarget(viewModel)
        await viewModel.start()
        await drain()
        #expect(viewModel.phase == .awaitingApproval)

        let stop = Task { await viewModel.cancel() }
        try? await Task.sleep(nanoseconds: 5_000_000)

        #expect(viewModel.isStopping)
        #expect(viewModel.approvalUnavailableMessage?.contains("stopping this task") == true)
        await stop.value
        #expect(viewModel.approvalUnavailableMessage == nil)
    }

    @Test("Canonical pending gate survives a missed gate event")
    func canonicalPendingGateAfterCursor() async {
        let api = MockAgentAPI()
        api.pendingGateResult = CUAPendingGate(
            gateID: "gate-reconnect", reason: "Confirm submission",
            action: "submit", target: "Expense report"
        )
        api.scriptedEvents = [makeEvent(seq: 2, kind: "gate_resolved")]
        let viewModel = CUAViewModel(api: api, pollIntervalNanos: 5_000_000)
        viewModel.goal = "submit my report"
        viewModel.appName = "Safari"
        selectTarget(viewModel)
        await viewModel.start()
        await drain()

        #expect(viewModel.phase == .awaitingApproval)
        #expect(
            viewModel.pendingApproval == CUAPendingApproval(
                gateID: "gate-reconnect", app: "Google Chrome",
                action: "submit", target: "Expense report", reason: "Confirm submission"
            )
        )
        await viewModel.approve()
        #expect(api.approvedGateIDs.first == "gate-reconnect")
    }

    @Test("Legacy resolution cannot clear a different identified gate")
    func legacyResolutionMatchesGateIdentity() async {
        let api = MockAgentAPI()
        api.scriptedEvents = [
            makeEvent(seq: 2, kind: "gate", gateID: "gate-current"),
            makeEvent(seq: 3, kind: "gate_resolved", gateID: "gate-previous"),
        ]
        let viewModel = CUAViewModel(api: api, pollIntervalNanos: 5_000_000)
        viewModel.goal = "continue safely"
        selectTarget(viewModel)
        await viewModel.start()
        await drain()

        #expect(viewModel.phase == .awaitingApproval)
        #expect(viewModel.pendingApproval?.gateID == "gate-current")
    }

    @Test("Matching gate resolution removes the stale approval card")
    func matchingResolutionResumesRun() async {
        let api = MockAgentAPI()
        api.scriptedEvents = [
            makeEvent(seq: 2, kind: "gate", gateID: "gate-current"),
            makeEvent(seq: 3, kind: "gate_resolved", gateID: "gate-current"),
        ]
        let viewModel = CUAViewModel(api: api, pollIntervalNanos: 5_000_000)
        viewModel.goal = "continue safely"
        selectTarget(viewModel)
        await viewModel.start()
        await drain()

        #expect(viewModel.phase == .running)
        #expect(viewModel.pendingApproval == nil)
        #expect(!viewModel.canApprove)
        await viewModel.approve()
        #expect(api.approveCalls == 0)
    }

    @Test("Active progress reports structured step and honest verifier outcome")
    func activeProgressPresentation() {
        let viewModel = CUAViewModel(api: MockAgentAPI())
        viewModel.maxSteps = 8
        viewModel.events = [
            makeEvent(
                seq: 1, kind: "plan", step: 3, action: "click",
                instruction: "Open the result", target: "Apple Silicon"
            ),
            makeEvent(seq: 2, kind: "executed", step: 3, outcome: "uncertain"),
        ]

        #expect(viewModel.activeProgress?.step == 3)
        #expect(viewModel.activeProgress?.maxSteps == 8)
        #expect(viewModel.activeProgress?.action == "click")
        #expect(viewModel.activeProgress?.target == "Apple Silicon")
        #expect(viewModel.activeProgress?.outcomeLabel == "Could not verify")
        viewModel.events.append(makeEvent(seq: 3, kind: "executed", step: 3, outcome: "success"))
        #expect(viewModel.activeProgress?.outcomeLabel == "Observed expected change")
    }

    @Test("Approval failure keeps the active run controllable")
    func gateApprovalFailure() async throws {
        let api = MockAgentAPI()
        api.scriptedEvents = [makeEvent(seq: 2, kind: "gate", gateID: "gate-7")]
        let viewModel = CUAViewModel(api: api, pollIntervalNanos: 5_000_000)
        viewModel.goal = "check flights"
        selectTarget(viewModel)
        await viewModel.start()
        await drain()
        #expect(viewModel.phase == .awaitingApproval)

        api.approveShouldFail = true
        await viewModel.approve()

        #expect(viewModel.phase == .awaitingApproval)
        #expect(viewModel.actionError?.hasPrefix("Approval failed:") == true)
        #expect(viewModel.phase.isBusy)
    }

    @Test("Cancel hits the API and is a no-op when idle")
    func cancel() async {
        let api = MockAgentAPI()
        let viewModel = CUAViewModel(api: api)
        await viewModel.cancel()
        #expect(api.cancelCalls == 0)
        viewModel.goal = "g"
        selectTarget(viewModel)
        await viewModel.start()
        await viewModel.cancel()
        #expect(api.cancelCalls == 1)
    }

    @Test("Stop failure keeps the Stop action available")
    func cancelFailure() async {
        let api = MockAgentAPI()
        api.cancelShouldFail = true
        let viewModel = CUAViewModel(api: api)
        viewModel.goal = "g"
        selectTarget(viewModel)
        await viewModel.start()
        await viewModel.cancel()

        #expect(api.cancelCalls == 1)
        #expect(viewModel.phase.isBusy)
        #expect(viewModel.actionError?.hasPrefix("Stop failed:") == true)
    }

    @Test("Repeated Stop while cancellation is in flight sends one request")
    func repeatedStopIsCoalesced() async {
        let api = MockAgentAPI()
        api.cancelDelayNanos = 50_000_000
        let viewModel = CUAViewModel(api: api, pollIntervalNanos: 5_000_000)
        viewModel.goal = "g"
        selectTarget(viewModel)
        await viewModel.start()

        let firstStop = Task { await viewModel.cancel() }
        try? await Task.sleep(nanoseconds: 5_000_000)
        #expect(viewModel.isStopping)
        await viewModel.cancel()
        await firstStop.value

        #expect(api.cancelCalls == 1)
        #expect(!viewModel.isStopping)
        #expect(viewModel.phase == .idle)
    }

    @Test("Stop during create cancels the server run before polling starts")
    func cancelWhileStarting() async {
        let api = MockAgentAPI()
        api.createDelayNanos = 50_000_000
        let viewModel = CUAViewModel(api: api, pollIntervalNanos: 5_000_000)
        viewModel.goal = "g"
        selectTarget(viewModel)

        let start = Task { await viewModel.start() }
        try? await Task.sleep(nanoseconds: 5_000_000)
        await viewModel.cancel()

        #expect(viewModel.phase == .starting)
        #expect(viewModel.isStopping)
        #expect(viewModel.actionError?.contains("Stop requested") == true)
        await start.value

        #expect(api.cancelCalls == 1)
        #expect(viewModel.phase == .idle)
        #expect(!viewModel.isStopping)
        #expect(viewModel.events.isEmpty)
        #expect(viewModel.actionError == nil)
    }

    @Test("Failed cleanup after stopping create preserves a retryable Stop")
    func cancelWhileStartingFailure() async {
        let api = MockAgentAPI()
        api.createDelayNanos = 30_000_000
        api.cancelShouldFail = true
        let viewModel = CUAViewModel(api: api, pollIntervalNanos: 5_000_000)
        viewModel.goal = "g"
        selectTarget(viewModel)

        let start = Task { await viewModel.start() }
        try? await Task.sleep(nanoseconds: 5_000_000)
        await viewModel.cancel()
        await start.value

        #expect(api.cancelCalls == 1)
        #expect(viewModel.phase == .running)
        #expect(!viewModel.isStopping)
        #expect(viewModel.phase.isBusy)
        #expect(viewModel.actionError?.contains("Try Stop again") == true)
    }

    @Test("Late approval cannot revive the UI after Stop")
    func approvalDoesNotReviveCancelledRun() async {
        let api = MockAgentAPI()
        api.pendingGateResult = CUAPendingGate(
            gateID: "gate-7", reason: "confirm", action: "click", target: "Submit"
        )
        api.approveDelayNanos = 50_000_000
        let viewModel = CUAViewModel(api: api, pollIntervalNanos: 5_000_000)
        viewModel.goal = "g"
        selectTarget(viewModel)
        await viewModel.start()
        await drain()
        #expect(viewModel.phase == .awaitingApproval)

        let approval = Task { await viewModel.approve() }
        try? await Task.sleep(nanoseconds: 5_000_000)
        await viewModel.cancel()
        await approval.value

        #expect(api.approveCalls == 1)
        #expect(api.cancelCalls == 1)
        #expect(viewModel.phase == .idle)
        #expect(viewModel.pendingApproval == nil)
    }

    @Test("Late poll response cannot revive the UI after Stop")
    func pollDoesNotReviveCancelledRun() async {
        let api = MockAgentAPI()
        api.eventsDelayNanos = 50_000_000
        api.runStatus = "completed"
        let viewModel = CUAViewModel(api: api, pollIntervalNanos: 5_000_000)
        viewModel.goal = "g"
        selectTarget(viewModel)
        await viewModel.start()
        try? await Task.sleep(nanoseconds: 5_000_000)

        await viewModel.cancel()
        try? await Task.sleep(nanoseconds: 70_000_000)

        #expect(api.cancelCalls == 1)
        #expect(viewModel.phase == .idle)
        #expect(viewModel.events.isEmpty)
        #expect(viewModel.pendingApproval == nil)
    }

    @Test("Polling failure keeps the active run controllable and retries")
    func pollingFailure() async {
        let api = MockAgentAPI()
        api.eventsShouldFail = true
        let viewModel = CUAViewModel(api: api, pollIntervalNanos: 5_000_000)
        viewModel.goal = "g"
        selectTarget(viewModel)
        await viewModel.start()
        await drain()

        #expect(viewModel.phase == .running)
        #expect(viewModel.phase.isBusy)
        #expect(viewModel.actionError?.hasPrefix("Connection interrupted:") == true)
        await viewModel.cancel()
        #expect(api.cancelCalls == 1)
        #expect(viewModel.phase == .idle)
    }

    @Test("Empty goal cannot start")
    func cannotStartEmptyGoal() async {
        let api = MockAgentAPI()
        let viewModel = CUAViewModel(api: api)
        #expect(!viewModel.canStart)
        viewModel.goal = "   "
        #expect(!viewModel.canStart)
        selectTarget(viewModel)
        await viewModel.start()
        #expect(api.createdRequests.isEmpty)
    }

    @Test("Start fails closed before POST when safe create recovery is unavailable")
    func requiresIdempotentCreateCapability() async {
        let api = MockAgentAPI()
        api.capabilitiesSupported = false
        let viewModel = CUAViewModel(api: api)
        viewModel.goal = "g"
        selectTarget(viewModel)

        await viewModel.start()

        #expect(api.createAttempts == 0)
        #expect(!viewModel.phase.isBusy)
        guard case let .failed(message) = viewModel.phase else {
            Issue.record("expected capability failure")
            return
        }
        #expect(message.contains("cannot safely recover"))
    }

    @Test("Typed create rejection does not retry or enter recovery")
    func typedCreateFailureIsDefinitive() async {
        let api = MockAgentAPI()
        api.createError = CUAClientError.typedHTTP(
            409, code: "request_identity_conflict", message: "conflict", recovery: []
        )
        let viewModel = CUAViewModel(api: api)
        viewModel.goal = "g"
        selectTarget(viewModel)

        await viewModel.start()

        #expect(api.createAttempts == 1)
        #expect(!viewModel.isRecoveringCreate)
        #expect(!viewModel.phase.isBusy)
    }

    @Test("Ambiguous create retries the same identity and stops the recovered run")
    func ambiguousCreateRetryStopsRecoveredRun() async {
        let api = MockAgentAPI()
        api.queuedCreateErrors = [URLError(.networkConnectionLost)]
        let viewModel = CUAViewModel(api: api)
        viewModel.goal = "g"
        selectTarget(viewModel)

        await viewModel.start()

        #expect(api.createAttempts == 2)
        #expect(Set(api.attemptedRequests.map(\.clientRequestID)).count == 1)
        #expect(api.createdRequests.count == 1)
        #expect(api.cancelCalls == 1)
        #expect(viewModel.phase == .idle)
        #expect(!viewModel.isRecoveringCreate)
    }

    @Test("Starting owns preflight so a second Start is ignored and Stop prevents POST")
    func capabilityPreflightOwnsLifecycle() async {
        let api = MockAgentAPI()
        api.capabilitiesDelayNanos = 40_000_000
        let viewModel = CUAViewModel(api: api)
        viewModel.goal = "g"
        selectTarget(viewModel)

        let first = Task { await viewModel.start() }
        try? await Task.sleep(nanoseconds: 5_000_000)
        #expect(viewModel.phase == .starting)
        await viewModel.start()
        await viewModel.cancel()
        await first.value

        #expect(api.createAttempts == 0)
        #expect(viewModel.phase == .idle)
        #expect(!viewModel.isStopping)
    }

    @Test("Unreachable recovery locks Start until lookup can find and stop the run")
    func ambiguousCreateRecoveryRemainsLocked() async {
        let api = MockAgentAPI()
        api.createError = URLError(.networkConnectionLost)
        api.lookupError = URLError(.cannotConnectToHost)
        let viewModel = CUAViewModel(api: api)
        viewModel.goal = "g"
        selectTarget(viewModel)

        await viewModel.start()

        #expect(api.createAttempts == 2)
        #expect(viewModel.phase == .starting)
        #expect(viewModel.isRecoveringCreate)
        #expect(!viewModel.canStart)
        #expect(viewModel.actionError?.contains("may still be executing") == true)

        let requestID = try? #require(api.attemptedRequests.first?.clientRequestID)
        api.lookupError = nil
        api.lookupResult = CUARunCreated(
            runID: "recovered", status: "running", windowID: "cg:123",
            clientRequestID: requestID
        )
        await viewModel.cancel()

        #expect(api.cancelCalls == 1)
        #expect(viewModel.phase == .idle)
        #expect(!viewModel.isRecoveringCreate)
    }

    @Test("Recovered run stays quarantined when cancellation fails")
    func recoveredRunCancellationFailureIsStoppable() async {
        let api = MockAgentAPI()
        api.queuedCreateErrors = [URLError(.networkConnectionLost)]
        api.cancelShouldFail = true
        let viewModel = CUAViewModel(api: api, pollIntervalNanos: 1_000_000_000)
        viewModel.goal = "g"
        selectTarget(viewModel)

        await viewModel.start()

        #expect(api.createAttempts == 2)
        #expect(api.cancelCalls == 1)
        #expect(viewModel.phase == .running)
        #expect(viewModel.phase.isBusy)
        #expect(!viewModel.canApprove)
        #expect(viewModel.actionError?.contains("may still be executing") == true)

        api.cancelShouldFail = false
        await viewModel.cancel()
        #expect(api.cancelCalls == 2)
        #expect(viewModel.phase == .idle)
    }

    @Test("Mismatched lookup metadata still stops the concrete recovered run")
    func mismatchedLookupIsCancelled() async {
        let api = MockAgentAPI()
        api.createError = URLError(.networkConnectionLost)
        api.lookupResult = CUARunCreated(
            runID: "mismatched", status: "running", windowID: "cg:123",
            clientRequestID: "wrong-request"
        )
        let viewModel = CUAViewModel(api: api)
        viewModel.goal = "g"
        selectTarget(viewModel)

        await viewModel.start()

        #expect(api.cancelCalls == 1)
        #expect(viewModel.phase == .idle)
        #expect(!viewModel.isRecoveringCreate)
    }

    @Test("Mismatched lookup stays stoppable when cancellation fails")
    func mismatchedLookupCancelFailureIsQuarantined() async {
        let api = MockAgentAPI()
        api.createError = URLError(.networkConnectionLost)
        api.lookupHandler = { requestID in
            CUARunCreated(
                runID: "mismatched", status: "running", windowID: "cg:other",
                clientRequestID: requestID
            )
        }
        api.cancelShouldFail = true
        let viewModel = CUAViewModel(api: api, pollIntervalNanos: 1_000_000_000)
        viewModel.goal = "g"
        selectTarget(viewModel)

        await viewModel.start()

        #expect(api.cancelCalls == 1)
        #expect(viewModel.phase == .running)
        #expect(viewModel.phase.isBusy)
        #expect(!viewModel.canApprove)
        #expect(viewModel.actionError?.contains("may still be executing") == true)

        api.cancelShouldFail = false
        await viewModel.cancel()
        #expect(api.cancelCalls == 2)
        #expect(viewModel.phase == .idle)
    }

    @Test("Start requires an explicit live process and window selection")
    func requiresExplicitWindowSelection() async {
        let api = MockAgentAPI()
        let viewModel = CUAViewModel(api: api)
        viewModel.goal = "open the selected document"
        #expect(!viewModel.canStart)

        await viewModel.loadTargets()
        #expect(viewModel.appOptions.first?.displayName == "Google Chrome — PID 42")
        #expect(viewModel.selectedPID == nil)
        #expect(!viewModel.canStart)

        await viewModel.selectApp(pid: 42)
        #expect(viewModel.windowOptions.first?.displayTitle == "Research — Apple Silicon")
        #expect(
            viewModel.windowOptions.first?.displayName
                == "Research — Apple Silicon — Window 1 · 1200×800"
        )
        #expect(viewModel.selectedWindowID == nil)
        viewModel.selectedWindowID = "cg:123"
        #expect(viewModel.canStart)
    }

    @Test("Refresh clears a window that moved or disappeared")
    func refreshRevalidatesSelection() async {
        let api = MockAgentAPI()
        let viewModel = CUAViewModel(api: api)
        viewModel.goal = "continue"
        selectTarget(viewModel)
        api.windowsResult = [
            CUAWindowOption(
                windowID: "cg:999", index: 0, title: "Replacement",
                x: nil, y: nil, width: nil, height: nil
            ),
        ]

        await viewModel.loadTargets()

        #expect(viewModel.selectedPID == 42)
        #expect(viewModel.selectedWindowID == nil)
        #expect(!viewModel.canStart)
        #expect(viewModel.targetError?.contains("no longer available") == true)
    }

    @Test("Late discovery response cannot replace a newer window list")
    func discoveryGenerationRejectsLateResponse() async {
        let viewModel = CUAViewModel(api: WindowDiscoveryRaceAPI())
        viewModel.appOptions = [CUAAppOption(name: "Finder", bundleID: nil, pid: 42)]
        let first = Task { await viewModel.selectApp(pid: 42) }
        try? await Task.sleep(nanoseconds: 5_000_000)
        await viewModel.refreshWindows()
        await first.value

        #expect(viewModel.windowOptions.map(\.windowID) == ["cg:new"])
        #expect(viewModel.selectedWindowID == nil)
        #expect(!viewModel.isLoadingWindows)
    }

    @Test("Discovery failure remains visible after clearing a previous selection")
    func discoveryFailureIsNotSwallowed() async {
        let api = MockAgentAPI()
        api.discoveryShouldFail = true
        let viewModel = CUAViewModel(api: api)
        selectTarget(viewModel)

        await viewModel.loadTargets()

        #expect(viewModel.selectedPID == nil)
        #expect(viewModel.selectedWindowID == nil)
        #expect(viewModel.targetError?.contains("Could not load apps and windows") == true)
    }

    @Test("Typed discovery 404 distinguishes a closed process from an old server")
    func typedDiscoveryNotFoundIsActionable() async {
        let api = MockAgentAPI()
        api.discoveryError = CUAClientError.typedHTTP(
            404,
            code: "app_not_found",
            message: "no running app for pid:42",
            recovery: ["Refresh the process list."]
        )
        let viewModel = CUAViewModel(api: api)
        selectTarget(viewModel)

        await viewModel.refreshWindows()

        #expect(viewModel.selectedWindowID == nil)
        #expect(viewModel.targetError?.contains("no longer running") == true)
        #expect(viewModel.targetError?.contains("does not support") == false)
    }

    @Test("Server rejection clears stale target and offers recovery")
    func staleTargetCreateFailure() async {
        let api = MockAgentAPI()
        api.createError = CUAClientError.http(409, "selected window was replaced")
        let viewModel = CUAViewModel(api: api)
        viewModel.goal = "continue"
        selectTarget(viewModel)

        await viewModel.start()

        #expect(viewModel.selectedWindowID == nil)
        #expect(!viewModel.canStart)
        guard case let .failed(message) = viewModel.phase else {
            Issue.record("expected failed phase")
            return
        }
        #expect(message.contains("Refresh the window list"))
    }

    @Test("Failed binding cleanup keeps only Stop available for the created run")
    func failedBindingCleanupIsRetryable() async {
        let api = MockAgentAPI()
        api.createError = CUAClientError.windowBinding(
            expected: "cg:123", actual: nil, runID: "unsafe", cancellationFailed: true
        )
        api.eventsShouldFail = true
        let viewModel = CUAViewModel(api: api, pollIntervalNanos: 5_000_000)
        viewModel.goal = "continue"
        selectTarget(viewModel)

        await viewModel.start()

        #expect(viewModel.phase == .running)
        #expect(viewModel.phase.isBusy)
        #expect(viewModel.selectedWindowID == nil)
        #expect(!viewModel.canApprove)
        #expect(viewModel.actionError?.contains("may still be executing") == true)

        try? await Task.sleep(nanoseconds: 10_000_000)
        api.eventsShouldFail = false
        try? await Task.sleep(nanoseconds: 15_000_000)
        #expect(viewModel.actionError?.contains("may still be executing") == true)

        api.cancelShouldFail = true
        await viewModel.cancel()
        #expect(api.cancelCalls == 1)
        #expect(viewModel.phase.isBusy)
        #expect(viewModel.actionError?.contains("may still be executing") == true)
        #expect(viewModel.actionError?.contains("Try Stop again") == true)

        api.cancelShouldFail = false
        await viewModel.cancel()
        #expect(api.cancelCalls == 2)
        #expect(viewModel.phase == .idle)
    }

    @Test("Permission readiness comes from the executor API")
    func executorPermissionReadiness() async {
        let api = MockAgentAPI()
        api.permissionsResult = CUAPermissionStatus(
            accessibility: true, screenRecording: false
        )
        let viewModel = CUAViewModel(api: api)
        await viewModel.loadPermissions()
        #expect(viewModel.executorPermissions?.accessibility == true)
        #expect(viewModel.executorPermissions?.screenRecording == false)
        #expect(viewModel.executorPermissions?.isReady == false)
    }

    @Test("Terminal failure surfaces the reason")
    func stalledRunFails() async throws {
        let api = MockAgentAPI()
        api.finalSummary = ""
        api.scriptedEvents = [
            makeEvent(seq: 2, kind: "terminal", status: "stalled", summary: ""),
        ]
        let viewModel = CUAViewModel(api: api, pollIntervalNanos: 5_000_000)
        viewModel.goal = "long task"
        selectTarget(viewModel)
        await viewModel.start()
        await drain()
        guard case .failed = viewModel.phase else {
            Issue.record("expected failed phase, got \(viewModel.phase)")
            return
        }
        #expect(!viewModel.phase.isBusy)
    }

    @Test("Terminal server errors remain visible")
    func terminalErrorIsVisible() async throws {
        let api = MockAgentAPI()
        api.scriptedEvents = [
            makeEvent(seq: 2, kind: "terminal", status: "failed", error: "planner unavailable"),
        ]
        let viewModel = CUAViewModel(api: api, pollIntervalNanos: 5_000_000)
        viewModel.goal = "open an article"
        selectTarget(viewModel)
        await viewModel.start()
        await drain()

        #expect(viewModel.phase == .failed(message: "planner unavailable"))
    }

    @Test("Typed stale terminal clears the frozen window before retry")
    func terminalStaleWindowIsRecoverable() async {
        let api = MockAgentAPI()
        api.scriptedEvents = [
            makeEvent(
                seq: 2, kind: "terminal", status: "stopped",
                reason: "selected window unavailable: it moved", error: "window_stale"
            ),
        ]
        let viewModel = CUAViewModel(api: api, pollIntervalNanos: 5_000_000)
        viewModel.goal = "open the report"
        selectTarget(viewModel)
        await viewModel.start()
        await drain()

        #expect(viewModel.selectedPID == 42)
        #expect(viewModel.selectedWindowID == nil)
        #expect(!viewModel.canStart)
        guard case let .failed(message) = viewModel.phase else {
            Issue.record("expected failed phase")
            return
        }
        #expect(message.contains("choose it again"))
    }
}

// MARK: - Client

struct RecordingStub: @unchecked Sendable {
    var request: URLRequest?
    var body: Data = Data("{}".utf8)
    var status = 200
}

@Suite("Agent Task Client", .serialized)
struct CUAClientTests {
    private func makeClient() -> CUAClient {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [RecordingURLProtocol.self]
        let session = URLSession(configuration: config)
        guard let client = CUAClient(
            host: "127.0.0.1", port: 8899, bearerToken: "tok", session: session
        ) else {
            Issue.record("client init failed")
            fatalError("unreachable")
        }
        return client
    }

    @Test("Rejects non-loopback hosts and empty tokens")
    func loopbackGuard() {
        #expect(CUAClient(host: "10.0.0.5", port: 8000, bearerToken: "t") == nil)
        #expect(CUAClient(host: "127.0.0.1", port: 8000, bearerToken: "") == nil)
        #expect(CUAClient(host: "localhost", port: 8000, bearerToken: "t") == nil)
    }

    @Test("Create posts JSON with bearer auth and decodes run_id")
    func createDecodes() async throws {
        RecordingURLProtocol.stubResponse(
            path: "/v1/cua/runs",
            body: Data(#"{"run_id":"abc","status":"running","window_id":"opaque:abc","client_request_id":"request-1"}"#.utf8)
        )
        let client = makeClient()
        let runID = try await client.create(
            CUARunRequest(
                app: "pid:42", goal: "g", planner: "local-9b",
                openURL: "", allowedDomain: "wikipedia.org", maxSteps: 8,
                humanLogin: true, windowID: "opaque:abc", clientRequestID: "request-1"
            )
        )
        #expect(runID == "abc")
        let captured = try #require(RecordingURLProtocol.captured["/v1/cua/runs"])
        #expect(captured.request.value(forHTTPHeaderField: "Authorization") == "Bearer tok")
        let sent = captured.body
        let json = try #require(JSONSerialization.jsonObject(with: sent) as? [String: Any])
        #expect(json["allowed_domain"] as? String == "wikipedia.org")
        #expect(json["max_steps"] as? Int == 8)
        #expect(json["app"] as? String == "pid:42")
        #expect(json["window_id"] as? String == "opaque:abc")
        #expect(json["client_request_id"] as? String == "request-1")
    }

    @Test("Create rejects missing or changed window binding and cancels the run")
    func createRequiresMatchingWindowConfirmation() async throws {
        let request = CUARunRequest(
            app: "pid:42", goal: "g", planner: "local-9b", openURL: "",
            allowedDomain: "", maxSteps: 8, humanLogin: true, windowID: "cg:123"
        )
        for actual in [nil, "cg:999"] as [String?] {
            RecordingURLProtocol.reset()
            var response: [String: Any] = ["run_id": "unsafe", "status": "running"]
            if let actual {
                response["window_id"] = actual
            }
            RecordingURLProtocol.stubResponse(
                path: "/v1/cua/runs",
                body: try JSONSerialization.data(withJSONObject: response)
            )
            RecordingURLProtocol.stubResponse(path: "/v1/cua/runs/unsafe/cancel")

            do {
                _ = try await makeClient().create(request)
                Issue.record("expected binding rejection")
            } catch let error as CUAClientError {
                #expect(
                    error == .windowBinding(
                        expected: "cg:123", actual: actual, runID: "unsafe",
                        cancellationFailed: false
                    )
                )
            }
            #expect(RecordingURLProtocol.captured["/v1/cua/runs/unsafe/cancel"] != nil)
        }
    }

    @Test("Create preserves the run identity when binding cleanup fails")
    func failedBindingCancellationPreservesRunID() async throws {
        RecordingURLProtocol.stubResponse(
            path: "/v1/cua/runs",
            body: Data(#"{"run_id":"unsafe","status":"running"}"#.utf8)
        )
        RecordingURLProtocol.stubResponse(
            path: "/v1/cua/runs/unsafe/cancel", status: 503
        )
        let request = CUARunRequest(
            app: "pid:42", goal: "g", planner: "local-9b", openURL: "",
            allowedDomain: "", maxSteps: 8, humanLogin: true, windowID: "cg:123"
        )

        do {
            _ = try await makeClient().create(request)
            Issue.record("expected binding rejection")
        } catch let error as CUAClientError {
            #expect(
                error == .windowBinding(
                    expected: "cg:123", actual: nil, runID: "unsafe",
                    cancellationFailed: true
                )
            )
        }
        #expect(RecordingURLProtocol.captured["/v1/cua/runs/unsafe/cancel"] != nil)
    }

    @Test("Create rejects a missing request identity and stops the run")
    func createRequiresMatchingRequestIdentity() async throws {
        RecordingURLProtocol.stubResponse(
            path: "/v1/cua/runs",
            body: Data(#"{"run_id":"unsafe","status":"running","window_id":"cg:123"}"#.utf8)
        )
        RecordingURLProtocol.stubResponse(path: "/v1/cua/runs/unsafe/cancel")
        let request = CUARunRequest(
            app: "pid:42", goal: "g", planner: "local-9b", openURL: "",
            allowedDomain: "", maxSteps: 8, humanLogin: true, windowID: "cg:123",
            clientRequestID: "request-1"
        )

        do {
            _ = try await makeClient().create(request)
            Issue.record("expected request identity rejection")
        } catch let error as CUAClientError {
            #expect(
                error == .requestBinding(
                    expected: "request-1", actual: nil, runID: "unsafe",
                    cancellationFailed: false
                )
            )
        }
        #expect(RecordingURLProtocol.captured["/v1/cua/runs/unsafe/cancel"] != nil)
    }

    @Test("Capabilities and request lookup decode the idempotent create contract")
    func idempotentCreateContract() async throws {
        RecordingURLProtocol.stubResponse(
            path: "/v1/cua/capabilities",
            body: Data(#"{"features":{"idempotent_run_create":true}}"#.utf8)
        )
        RecordingURLProtocol.stubResponse(
            path: "/v1/cua/runs/by-request/request-1",
            body: Data(#"{"run_id":"abc","status":"running","window_id":"cg:123","client_request_id":"request-1"}"#.utf8)
        )
        let client = makeClient()

        #expect(try await client.capabilities().features.idempotentRunCreate)
        let found = try await client.run(clientRequestID: "request-1")
        #expect(found.runID == "abc")
        #expect(found.windowID == "cg:123")
        #expect(found.clientRequestID == "request-1")
    }

    @Test("Discovery decodes bundle_id and URL-encodes the PID selector")
    func discoveryContract() async throws {
        RecordingURLProtocol.stubResponse(
            path: "/v1/cua/apps",
            body: Data(#"[{"name":"Finder","bundle_id":"com.apple.finder","pid":42}]"#.utf8)
        )
        RecordingURLProtocol.stubResponse(
            path: "/v1/cua/apps/pid:42/windows",
            body: Data(#"[{"window_id":"opaque:abc","index":0,"title":"Downloads"}]"#.utf8)
        )
        let client = makeClient()
        let apps = try await client.apps()
        let windows = try await client.windows(app: "pid:42")

        #expect(apps.first?.bundleID == "com.apple.finder")
        #expect(apps.first?.displayName == "Finder — PID 42")
        #expect(windows.first?.windowID == "opaque:abc")
        let captured = try #require(
            RecordingURLProtocol.captured["/v1/cua/apps/pid:42/windows"]
        )
        #expect(captured.request.url?.absoluteString.contains("pid%3A42") == true)
    }

    @Test("Events decode snake_case payload")
    func eventsDecode() async throws {
        let payload = """
        {"run_id":"abc","app":"Google Chrome","goal":"g","status":"running",
         "final_summary":"","error":"","planner":"p","events_after_seq":1,
         "pending_gate":{"gate_id":"gate-7","reason":"sign-in","action":"sign_in","target":"Account"},
         "events":[{"seq":2,"kind":"gate","step":1,"step_instruction":"click it",
                    "gate_id":"gate-7","app":"Safari","action":"sign_in","target":"Account",
                    "latency_s":0.4}]}
        """
        RecordingURLProtocol.stubResponse(path: "/v1/cua/runs/abc/events", body: Data(payload.utf8))
        let client = makeClient()
        let view = try await client.events(runID: "abc", after: 1)
        #expect(view.events.count == 1)
        #expect(view.events[0].stepInstruction == "click it")
        #expect(view.events[0].app == "Safari")
        #expect(view.events[0].gateID == "gate-7")
        #expect(view.events[0].action == "sign_in")
        #expect(view.events[0].target == "Account")
        #expect(view.pendingGate?.gateID == "gate-7")
        #expect(view.pendingGate?.target == "Account")
        #expect(view.windowID == nil)
    }

    @Test("Approval binds the decision to the pending gate")
    func approvalRequestBody() async throws {
        RecordingURLProtocol.stubResponse(path: "/v1/cua/runs/abc/approval")
        let client = makeClient()
        try await client.approve(runID: "abc", gateID: "gate-7")

        let captured = try #require(
            RecordingURLProtocol.captured["/v1/cua/runs/abc/approval"]
        )
        let json = try #require(
            JSONSerialization.jsonObject(with: captured.body) as? [String: Any]
        )
        #expect(json["gate_id"] as? String == "gate-7")
        #expect(json["approved"] as? Bool == true)
    }

    @Test("Permissions decode the executor process status")
    func permissionsDecode() async throws {
        RecordingURLProtocol.stubResponse(
            path: "/v1/cua/permissions",
            body: Data(#"{"accessibility":true,"screen_recording":false}"#.utf8)
        )
        let status = try await makeClient().permissions()
        #expect(status.accessibility)
        #expect(status.screenRecording == false)
        #expect(!status.isReady)
    }

    @Test("HTTP errors surface as typed failures")
    func httpError() async {
        RecordingURLProtocol.stubResponse(
            path: "/v1/cua/runs/abc/approval",
            body: Data(#"{"detail":"run is not awaiting approval"}"#.utf8),
            status: 409
        )
        let client = makeClient()
        do {
            try await client.approve(runID: "abc", gateID: "gate-7")
            Issue.record("expected throw")
        } catch let error as CUAClientError {
            #expect(error == .http(409, "run is not awaiting approval"))
        } catch {
            Issue.record("unexpected error type: \(error)")
        }
    }

    @Test("Structured discovery errors preserve code, message, and recovery")
    func typedHTTPError() async {
        RecordingURLProtocol.stubResponse(
            path: "/v1/cua/apps/pid:42/windows",
            body: Data(
                #"{"detail":{"code":"window_not_found","message":"no windows for pid:42","recovery":["Open a window."]}}"#.utf8
            ),
            status: 404
        )
        do {
            _ = try await makeClient().windows(app: "pid:42")
            Issue.record("expected throw")
        } catch let error as CUAClientError {
            #expect(
                error == .typedHTTP(
                    404,
                    code: "window_not_found",
                    message: "no windows for pid:42",
                    recovery: ["Open a window."]
                )
            )
        } catch {
            Issue.record("unexpected error type: \(error)")
        }
    }
}

/// Minimal URLProtocol double: stubs and captures are keyed by request path so
/// parallel tests never observe each other's responses.
final class RecordingURLProtocol: URLProtocol {
    struct Captured {
        let request: URLRequest
        let body: Data
    }

    nonisolated(unsafe) static var stubs: [String: RecordingStub] = [:]
    nonisolated(unsafe) static var captured: [String: Captured] = [:]

    static func stubResponse(path: String, body: Data = Data("{}".utf8), status: Int = 200) {
        stubs[path] = RecordingStub(body: body, status: status)
    }

    static func reset() {
        stubs = [:]
        captured = [:]
    }

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        guard let url = request.url else {
            client?.urlProtocol(self, didFailWithError: URLError(.badURL))
            return
        }
        Self.captured[url.path] = Captured(request: request, body: requestBody(request))
        let stub = Self.stubs[url.path] ?? RecordingStub(body: Data(), status: 404)
        let response = HTTPURLResponse(
            url: url, statusCode: stub.status, httpVersion: "HTTP/1.1", headerFields: nil
        )!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: stub.body)
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}

    private func requestBody(_ request: URLRequest) -> Data {
        if let body = request.httpBody { return body }
        guard let stream = request.httpBodyStream else { return Data() }
        stream.open()
        defer { stream.close() }
        let buffer = UnsafeMutablePointer<UInt8>.allocate(capacity: 65_536)
        defer { buffer.deallocate() }
        var data = Data()
        while stream.hasBytesAvailable {
            let read = stream.read(buffer, maxLength: 65_536)
            if read <= 0 { break }
            data.append(buffer, count: read)
        }
        return data
    }
}

// MARK: - Add-brain settings

@Suite(.serialized)
struct CUAAddBrainTests {
    @MainActor
    @Test func saveBrainPostsRequestAndReloads() async throws {
        let api = MockAgentAPI()
        let vm = CUAViewModel(api: api)
        vm.newBrainName = "deepseek"
        vm.newBrainURL = "https://api.example.com/v1/chat/completions"
        vm.newBrainModel = "deepseek-reasoner"
        vm.newBrainAPIKey = "sk-test"
        vm.newBrainAllowRemote = true
        await vm.saveBrain()
        #expect(api.addedPlanners.count == 1)
        #expect(api.addedPlanners[0].name == "deepseek")
        #expect(api.addedPlanners[0].apiKey == "sk-test")
        #expect(api.addedPlanners[0].allowRemote == true)
        #expect(vm.showAddBrain == false)
        #expect(vm.brainError == nil)
        #expect(vm.newBrainName.isEmpty)
    }

    @MainActor
    @Test func saveBrainSurfacesError() async throws {
        let api = MockAgentAPI()
        api.addShouldFail = true
        let vm = CUAViewModel(api: api)
        vm.showAddBrain = true
        vm.newBrainName = "bad"
        vm.newBrainURL = "https://api.example.com/v1"
        vm.newBrainModel = "m"
        await vm.saveBrain()
        #expect(vm.brainError != nil)
        #expect(vm.showAddBrain == true)
    }

    @MainActor
    @Test func addBrainIsValidRequiresAllFields() {
        let api = MockAgentAPI()
        let vm = CUAViewModel(api: api)
        #expect(vm.addBrainIsValid == false)
        vm.newBrainName = "x"
        vm.newBrainURL = "https://api.example.com/v1"
        #expect(vm.addBrainIsValid == false)
        vm.newBrainModel = "m"
        #expect(vm.addBrainIsValid == true)
    }

    @MainActor
    @Test func plannerDisclosureSeparatesMacExecutionFromExternalBrainData() {
        let vm = CUAViewModel(api: MockAgentAPI())
        vm.plannerName = "external"
        vm.plannerOptions = [
            CUAPlannerOption(
                name: "external", model: "m", url: "https://planner.example/v1",
                textOnly: false, allowRemote: true
            )
        ]
        #expect(vm.plannerDisclosure.contains("Actions run on this Mac"))
        #expect(vm.plannerDisclosure.contains("external brain"))
        #expect(vm.plannerDisclosure.contains("screenshot"))

        vm.plannerOptions[0].textOnly = true
        #expect(!vm.plannerDisclosure.contains("screenshot"))
        #expect(vm.plannerDisclosure.contains("Accessibility snapshot"))

        vm.plannerName = "loopback"
        vm.plannerOptions = [
            CUAPlannerOption(
                name: "loopback", model: "m", url: "http://127.0.0.1:8080/v1",
                textOnly: true
            ),
        ]
        #expect(vm.plannerDisclosure.contains("Actions run on this Mac"))
        #expect(vm.plannerDisclosure.contains("loopback endpoint"))
        #expect(vm.plannerDisclosure.contains("may forward it"))
        #expect(!vm.plannerDisclosure.contains("brain run on this Mac"))
    }

    @MainActor
    @Test func loopbackEndpointClassificationMatchesServerRules() {
        #expect(CUAViewModel.isLoopbackEndpoint("http://localhost:1234/v1"))
        #expect(CUAViewModel.isLoopbackEndpoint("http://localhost.:1234/v1"))
        #expect(CUAViewModel.isLoopbackEndpoint("http://127.0.0.1:1234/v1"))
        #expect(CUAViewModel.isLoopbackEndpoint("http://127.0.0.2:1234/v1"))
        #expect(CUAViewModel.isLoopbackEndpoint("http://[::1]:1234/v1"))
        #expect(!CUAViewModel.isLoopbackEndpoint("https://planner.example/v1"))
        #expect(!CUAViewModel.isLoopbackEndpoint("not a url"))
    }
}

@Suite(.serialized)
struct CUAPlannerDecodeTests {
    @Test func decodesLegacyServerJSONWithoutNewFields() throws {
        // Older sidecars predate has_api_key/user_created; the client must
        // still decode their planner list instead of failing the picker.
        let legacy = #"{"name":"local-9b","model":"m9","url":"http://127.0.0.1:1/v1","text_only":true}"#
        let data = Data(legacy.utf8)
        let option = try JSONDecoder().decode(CUAPlannerOption.self, from: data)
        #expect(option.name == "local-9b")
        #expect(option.hasApiKey == false)
        #expect(option.userCreated == false)
    }

    @Test func decodesSnakeCaseFields() throws {
        let full = #"{"name":"my-cloud","model":"m","url":"https://x/v1","text_only":false,"note":"","has_api_key":true,"user_created":true,"allow_remote":true}"#
        let option = try JSONDecoder().decode(CUAPlannerOption.self, from: Data(full.utf8))
        #expect(option.hasApiKey == true)
        #expect(option.userCreated == true)
        #expect(option.allowRemote == true)
    }
}

@Suite("Agent Task Target UI")
struct CUATargetUISourceTests {
    @Test("Completed result separates recognized evidence without losing the answer")
    func completedResultPresentation() {
        let result = CUAResultPresentation(
            summary: "Created the folder and moved 3 files.\nSupporting evidence: Finder shows all 3 files."
        )

        #expect(result.answer == "Created the folder and moved 3 files.")
        #expect(result.evidence == "Finder shows all 3 files.")
    }

    @Test("Free-form completed result remains intact")
    func freeFormCompletedResultPresentation() {
        let summary = "Compared both documents. The second contains the newer pricing and evidence gathered from the page."
        let result = CUAResultPresentation(summary: summary)

        #expect(result.answer == summary)
        #expect(result.evidence == nil)
    }

    @Test("Completed task without a summary retains a visible result")
    func emptyCompletedResultPresentation() {
        let result = CUAResultPresentation(summary: "  \n")

        #expect(result.answer.isEmpty)
        #expect(result.displayedAnswer == "The task completed without a result summary.")
        #expect(result.evidence == nil)
    }

    @Test("Window controls are addressable and privacy copy distinguishes execution")
    func targetControlsAndPrivacyCopy() throws {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let section = try String(
            contentsOf: root.appendingPathComponent("Sources/Rapid/UI/CUASection.swift"),
            encoding: .utf8
        )
        let page = try String(
            contentsOf: root.appendingPathComponent("Sources/Rapid/UI/ComputerUseView.swift"),
            encoding: .utf8
        )
        let content = try String(
            contentsOf: root.appendingPathComponent("Sources/Rapid/UI/ContentView.swift"),
            encoding: .utf8
        )

        #expect(section.contains("ComputerUse.Agent.Target.Process"))
        #expect(section.contains("ComputerUse.Agent.Target.Window"))
        #expect(section.contains("ComputerUse.Agent.Target.Refresh"))
        #expect(section.contains("ComputerUse.Agent.Target.Error"))
        #expect(section.contains("Retry Recovery"))
        #expect(section.contains("Describe what you want Rapid to do"))
        #expect(section.contains("accessibilityLabel(\"Task goal\")"))
        #expect(section.contains("ComputerUse.Agent.ActiveProgress"))
        #expect(section.contains("ComputerUse.Agent.Summary"))
        #expect(section.contains("ComputerUse.Agent.Summary.Answer"))
        #expect(section.contains("ComputerUse.Agent.Evidence"))
        #expect(section.contains("ComputerUse.Agent.History"))
        #expect(section.contains("DisclosureGroup"))
        #expect(section.contains("Task complete"))
        #expect(section.contains("if case let .finished(summary) = viewModel.phase"))
        #expect(section.contains("COMPLETED"))
        #expect(section.contains("RUNNING"))
        #expect(page.contains("ComputerUse.Server.Starting"))
        #expect(page.contains("ComputerUse.Server.Error"))
        #expect(page.contains("ComputerUse.Server.Retry"))
        #expect(page.contains("Actions run on this Mac"))
        #expect(page.contains("choose the brain endpoint"))
        #expect(!page.contains("Everything runs locally"))
        #expect(page.contains("CUASection(viewModel: cuaViewModel)"))
        #expect(!page.contains("Start with a flow"))
        #expect(!page.contains("CREATE YOUR OWN"))
        #expect(!page.contains("ComputerUseStarter"))
        #expect(!page.contains("DraftPostFlowSheet"))
        #expect(!page.contains("FreeUpSpaceFlowSheet"))
        #expect(!content.contains("languageRuntime: DraftPostLanguageRuntime"))
        #expect(!content.contains("visualRuntime: DraftPostVisualRuntime"))
    }
}
