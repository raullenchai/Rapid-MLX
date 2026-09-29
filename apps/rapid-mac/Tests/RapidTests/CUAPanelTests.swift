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
    var windowsByApp: [String: [CUAWindowOption]] = [:]
    var discoveryShouldFail = false
    var discoveryError: Error?
    var appsCalls = 0
    var createError: Error?
    var queuedCreateErrors: [Error] = []
    var createAttempts = 0
    var capabilitiesSupported = true
    var multiTargetSupported = true
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
    var permissionRequestSupported = true
    var permissionRequestError: Error?
    var requestedPermissions: [MacAutomationPermission] = []
    var permissionsCalls = 0
    var suspendPermissionRequest = false
    var permissionRequestContinuation: CheckedContinuation<Void, Never>?
    var suspendPermissions = false
    var permissionsContinuation: CheckedContinuation<Void, Never>?
    var pendingGateResult: CUAPendingGate?
    var activeTargetIDResult: String?
  var targetResolutionResult = CUATargetResolution(
    status: "unresolved", targets: [], initialTargetID: nil,
    reason: "No matching app", approval: nil
  )
  var queuedTargetResolutionResults: [CUATargetResolution] = []
  var targetResolutionRequests: [
    (goal: String, planner: String, allowRemoteAppDiscovery: Bool)
  ] = []

    var addedPlanners: [CUAPlannerCreateRequest] = []
    var deletedPlannerNames: [String] = []
    var addShouldFail = false

    func capabilities() async throws -> CUACapabilities {
        if capabilitiesDelayNanos > 0 {
            try? await Task.sleep(nanoseconds: capabilitiesDelayNanos)
        }
        return CUACapabilities(
            features: .init(
                idempotentRunCreate: capabilitiesSupported,
                multiTargetRuns: multiTargetSupported,
                switchTarget: multiTargetSupported,
                permissionRequest: permissionRequestSupported
            ),
            maxRunTargets: 3
        )
    }

    func planners() async throws -> [CUAPlannerOption] {
        plannersResult
    }

  func resolveTargets(
    goal: String, planner: String, allowRemoteAppDiscovery: Bool
  ) async throws -> CUATargetResolution {
    targetResolutionRequests.append((goal, planner, allowRemoteAppDiscovery))
    if !queuedTargetResolutionResults.isEmpty {
      return queuedTargetResolutionResults.removeFirst()
    }
    return targetResolutionResult
  }

    func apps() async throws -> [CUAAppOption] {
        appsCalls += 1
        if let discoveryError { throw discoveryError }
        if discoveryShouldFail { throw Failure.requested }
        return appsResult
    }

    func windows(app: String) async throws -> [CUAWindowOption] {
        if let discoveryError { throw discoveryError }
        if discoveryShouldFail { throw Failure.requested }
        return windowsByApp[app] ?? windowsResult
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
        permissionsCalls += 1
        if suspendPermissions {
            await withCheckedContinuation { permissionsContinuation = $0 }
        }
        return permissionsResult
    }

    func requestPermission(
        _ permission: MacAutomationPermission
    ) async throws -> CUAPermissionRequestResult {
        requestedPermissions.append(permission)
        if suspendPermissionRequest {
            await withCheckedContinuation { permissionRequestContinuation = $0 }
        }
        if let permissionRequestError { throw permissionRequestError }
        return CUAPermissionRequestResult(
            permission: permission == .accessibility
                ? "accessibility" : "screen_recording",
            granted: permissionsResult.isGranted(permission),
            permissions: permissionsResult
        )
    }

    func resumePermissionRequest() {
        permissionRequestContinuation?.resume()
        permissionRequestContinuation = nil
    }

    func resumePermissions() {
        permissionsContinuation?.resume()
        permissionsContinuation = nil
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
                    gateID: event.gateID, target: event.target,
                    targetID: event.targetID, fromTargetID: event.fromTargetID
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
            pendingGate: pendingGateResult,
            activeTargetID: activeTargetIDResult
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
      )
        ]
    }

    func planners() async throws -> [CUAPlannerOption] { [] }
  func resolveTargets(
    goal: String, planner: String, allowRemoteAppDiscovery: Bool
  ) async throws -> CUATargetResolution {
    CUATargetResolution(
      status: "unresolved", targets: [], initialTargetID: nil,
      reason: "Unavailable", approval: nil
    )
  }
    func addPlanner(_ request: CUAPlannerCreateRequest) async throws {}
    func deletePlanner(name: String) async throws {}
    func create(_ request: CUARunRequest) async throws -> String { "run" }
    func run(clientRequestID: String) async throws -> CUARunCreated {
        throw CUAClientError.http(404, "not found")
    }
    func permissions() async throws -> CUAPermissionStatus {
        CUAPermissionStatus(accessibility: true, screenRecording: true)
    }
    func requestPermission(
        _ permission: MacAutomationPermission
    ) async throws -> CUAPermissionRequestResult {
        CUAPermissionRequestResult(
            permission: "accessibility", granted: true,
            permissions: CUAPermissionStatus(
                accessibility: true, screenRecording: true
            )
        )
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
    viewModel.allowedDomain = "example.com"
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
        gateID: String? = nil, gateTarget: String? = nil, targetID: String? = nil
    ) -> CUAEvent {
        CUAEvent(
            seq: seq, kind: kind, step: step, action: action,
            stepInstruction: instruction, outcome: outcome,
            targetLabel: target, status: status, finalSummary: summary, reason: reason,
            error: error, app: app, gateID: gateID, target: gateTarget,
            targetID: targetID
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
        #expect(viewModel.runContext?.goal == "open Apple Silicon")
        #expect(viewModel.runContext?.targetDisplayName.contains("PID 42") == true)
        #expect(!viewModel.canStart)

        viewModel.newTask()
        #expect(viewModel.phase == .idle)
        #expect(viewModel.runContext == nil)
        #expect(viewModel.goal.isEmpty)
        #expect(viewModel.events.isEmpty)
        #expect(viewModel.plannerName == "local-27b")
        #expect(viewModel.selectedWindowID == "cg:123")
    }

    @Test("Start freezes an ordered authorized target set with per-target domains")
    func startFreezesMultiTargetSet() async {
        let api = MockAgentAPI()
        let viewModel = CUAViewModel(api: api, pollIntervalNanos: 5_000_000)
        let safari = CUAAppOption(name: "Safari", bundleID: "com.apple.Safari", pid: 42)
        let textEdit = CUAAppOption(name: "TextEdit", bundleID: "com.apple.TextEdit", pid: 43)
        viewModel.appOptions = [safari, textEdit]
        viewModel.windowOptions = [
            CUAWindowOption(
                windowID: "cg:1", index: 0, title: "Research", x: nil, y: nil,
                width: 900, height: 700
            ),
        ]
        viewModel.selectedPID = 42
        viewModel.selectedWindowID = "cg:1"
        viewModel.allowedDomain = "example.com"
        viewModel.addSelectedTarget()
        #expect(!viewModel.canStart)

        viewModel.windowOptions = [
            CUAWindowOption(
                windowID: "cg:2", index: 0, title: "Notes", x: nil, y: nil,
                width: 700, height: 600
            ),
        ]
        viewModel.selectedPID = 43
        viewModel.selectedWindowID = "cg:2"
        viewModel.addSelectedTarget()
        let secondTargetID = try! #require(viewModel.selectedTargets.last?.targetID)
        viewModel.setInitialTarget(id: secondTargetID)
        viewModel.goal = "read the page and update notes"
        #expect(viewModel.canStart)

        await viewModel.start()

        let request = api.attemptedRequests.first
        #expect(request?.targets?.map(\.windowID) == ["cg:1", "cg:2"])
        #expect(request?.targets?.map(\.allowedDomain) == ["example.com", ""])
        #expect(request?.initialTargetID == secondTargetID)
        #expect(request?.app == "pid:43")
        #expect(request?.windowID == "cg:2")
        #expect(viewModel.runContext?.targets.map(\.displayName).count == 2)
        #expect(viewModel.runContext?.allowedDomain.isEmpty == true)
        #expect(viewModel.activeTargetID == request?.initialTargetID)
        viewModel.invalidateSession()
    }

    @Test("One explicitly authorized browser window can start")
    func oneAuthorizedBrowserWindowCanStart() async {
        let api = MockAgentAPI()
        api.multiTargetSupported = false
        let viewModel = CUAViewModel(api: api, pollIntervalNanos: 5_000_000)
        viewModel.appOptions = [
            CUAAppOption(name: "Google Chrome", bundleID: "com.google.Chrome", pid: 42),
        ]
        viewModel.windowOptions = [
            CUAWindowOption(
                windowID: "cg:chrome", index: 0, title: "Example", x: nil, y: nil,
                width: 900, height: 700
            ),
        ]
        viewModel.selectedPID = 42
        viewModel.selectedWindowID = "cg:chrome"
        viewModel.allowedDomain = "example.com"
        viewModel.addSelectedTarget()
        viewModel.goal = "Summarize this page"

        #expect(viewModel.selectedTargets.count == 1)
        #expect(viewModel.targetError == nil)
        #expect(viewModel.canStart)

        await viewModel.start()

        let request = api.attemptedRequests.first
        #expect(request?.app == "pid:42")
        #expect(request?.windowID == "cg:chrome")
        #expect(request?.allowedDomain == "example.com")
        #expect(request?.targets == nil)
        #expect(request?.initialTargetID == nil)
        #expect(viewModel.runContext?.targets.isEmpty == true)
        #expect(viewModel.runContext?.initialTargetID == nil)
        #expect(viewModel.activeTargetID == nil)
        viewModel.invalidateSession()
    }

    @Test("Removing the starting target safely falls back to the first remaining target")
    func removingInitialTargetChoosesSafeFallback() {
        let viewModel = CUAViewModel(api: MockAgentAPI())
        viewModel.appOptions = [
            CUAAppOption(name: "TextEdit", bundleID: "com.apple.TextEdit", pid: 42),
            CUAAppOption(name: "Notes", bundleID: "com.apple.Notes", pid: 43),
        ]
        for (pid, id, title) in [(42, "cg:1", "Draft"), (43, "cg:2", "Notes")] {
            viewModel.windowOptions = [
                CUAWindowOption(
                    windowID: id, index: 0, title: title, x: nil, y: nil,
                    width: nil, height: nil
                ),
            ]
            viewModel.selectedPID = pid
            viewModel.selectedWindowID = id
            viewModel.addSelectedTarget()
        }
        let second = viewModel.selectedTargets[1].targetID
        viewModel.setInitialTarget(id: second)

        viewModel.removeSelectedTarget(id: second)

        #expect(viewModel.selectedInitialTargetID == viewModel.selectedTargets.first?.targetID)
    }

    @Test("Multi-window approval requires the active target identity")
    func multiTargetApprovalIsTargetBound() async {
        let api = MockAgentAPI()
        api.pendingGateResult = CUAPendingGate(
            gateID: "gate-1", reason: "confirm", action: "click", target: "Send",
            targetID: "target_2"
        )
        api.activeTargetIDResult = "target_1"
        let viewModel = CUAViewModel(api: api, pollIntervalNanos: 5_000_000)
        viewModel.appOptions = [
            CUAAppOption(name: "TextEdit", bundleID: "com.apple.TextEdit", pid: 42),
            CUAAppOption(name: "Notes", bundleID: "com.apple.Notes", pid: 43),
        ]
        viewModel.windowOptions = [
            CUAWindowOption(
                windowID: "cg:1", index: 0, title: "First", x: nil, y: nil,
                width: nil, height: nil
            ),
        ]
        viewModel.selectedPID = 42
        viewModel.selectedWindowID = "cg:1"
        viewModel.addSelectedTarget()
        viewModel.windowOptions = [
            CUAWindowOption(
                windowID: "cg:2", index: 0, title: "Second", x: nil, y: nil,
                width: nil, height: nil
            ),
        ]
        viewModel.selectedPID = 43
        viewModel.selectedWindowID = "cg:2"
        viewModel.addSelectedTarget()
        viewModel.goal = "work across both windows"

        await viewModel.start()
        await drain()

        #expect(viewModel.phase == .awaitingApproval)
        #expect(!viewModel.canApprove)
        #expect(viewModel.approvalUnavailableMessage?.contains("active app") == true)
        await viewModel.approve()
        #expect(api.approveCalls == 0)
        viewModel.invalidateSession()
    }

    @Test("Browser target requires an explicit domain before authorization")
    func browserTargetRequiresDomain() {
        let viewModel = CUAViewModel(api: MockAgentAPI())
        viewModel.appOptions = [
            CUAAppOption(name: "Safari", bundleID: "com.apple.Safari", pid: 42),
        ]
        viewModel.windowOptions = [
            CUAWindowOption(
                windowID: "cg:1", index: 0, title: "Research", x: nil, y: nil,
                width: nil, height: nil
            ),
        ]
        viewModel.selectedPID = 42
        viewModel.selectedWindowID = "cg:1"

        #expect(!viewModel.canAddSelectedTarget)
        viewModel.allowedDomain = "example.com"
        #expect(viewModel.canAddSelectedTarget)
    }

    @Test("Start freezes request fields before asynchronous preflight")
    func startFreezesRunContext() async {
        let api = MockAgentAPI()
        api.capabilitiesDelayNanos = 100_000_000
        let viewModel = CUAViewModel(api: api, pollIntervalNanos: 5_000_000)
        viewModel.goal = "original goal"
        viewModel.plannerName = "brain-a"
        viewModel.plannerOptions = [
            CUAPlannerOption(
                name: "brain-a", model: "model-a", url: "http://127.0.0.1:8080/v1",
                textOnly: true
            ),
        ]
        viewModel.maxSteps = 7
        selectTarget(viewModel)

        let start = Task { await viewModel.start() }
        try? await Task.sleep(nanoseconds: 10_000_000)
        #expect(viewModel.phase == .starting)
        #expect(viewModel.runContext?.goal == "original goal")
        #expect(viewModel.runContext?.plannerDisplayName.contains("model-a") == true)

        viewModel.goal = "changed while starting"
        viewModel.plannerName = "brain-b"
        viewModel.maxSteps = 39
        viewModel.selectedPID = nil
        viewModel.selectedWindowID = nil
        viewModel.allowedDomain = "changed.example"
        await start.value

        #expect(api.attemptedRequests.first?.goal == "original goal")
        #expect(api.attemptedRequests.first?.planner == "brain-a")
        #expect(api.attemptedRequests.first?.maxSteps == 7)
        #expect(api.attemptedRequests.first?.app == "pid:42")
        #expect(api.attemptedRequests.first?.windowID == "cg:123")
        #expect(api.attemptedRequests.first?.allowedDomain == "example.com")
        #expect(viewModel.runContext?.maxSteps == 7)
        #expect(viewModel.runContext?.allowedDomain == "example.com")
        let frozenContext = viewModel.runContext
        viewModel.newTask()
        #expect(viewModel.runContext == frozenContext)
        viewModel.invalidateSession()
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
        #expect(!viewModel.canStart)
        #expect(viewModel.selectedBrowserDomainError != nil)
        viewModel.allowedDomain = " Flights.Example.COM. "
        #expect(viewModel.canStart)
        #expect(viewModel.selectedBrowserDomainError == nil)
    }

    @Test("Single-browser scope rejects blank and ambiguous host input")
    func singleBrowserDomainValidation() {
        let viewModel = CUAViewModel(api: MockAgentAPI())
        viewModel.goal = "compare flights"
        selectTarget(viewModel)

        for invalid in ["", ".example.com", "example.com/path", "example.com:443", "exa_mple.com"] {
            viewModel.allowedDomain = invalid
            #expect(!viewModel.canStart)
            #expect(viewModel.selectedBrowserDomainError != nil)
        }

        for valid in ["example.com", "flights.example.com", "EXAMPLE.COM."] {
            viewModel.allowedDomain = valid
            #expect(viewModel.canStart)
            #expect(viewModel.selectedBrowserDomainError == nil)
        }
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

    @Test("Refresh clears the whole authorized set when one frozen window disappears")
    func refreshRevalidatesAuthorizedSet() async {
        let api = MockAgentAPI()
        let safari = CUAAppOption(name: "Safari", bundleID: "com.apple.Safari", pid: 42)
        let textEdit = CUAAppOption(name: "TextEdit", bundleID: "com.apple.TextEdit", pid: 43)
        let web = CUAWindowOption(
            windowID: "cg:1", index: 0, title: "Web", x: nil, y: nil,
            width: nil, height: nil
        )
        let notes = CUAWindowOption(
            windowID: "cg:2", index: 0, title: "Notes", x: nil, y: nil,
            width: nil, height: nil
        )
        api.appsResult = [safari, textEdit]
        api.windowsByApp = ["pid:42": [web], "pid:43": [notes]]
        let viewModel = CUAViewModel(api: api)
        viewModel.appOptions = api.appsResult
        viewModel.windowOptions = [web]
        viewModel.selectedPID = 42
        viewModel.selectedWindowID = "cg:1"
        viewModel.allowedDomain = "example.com"
        viewModel.addSelectedTarget()
        viewModel.windowOptions = [notes]
        viewModel.selectedPID = 43
        viewModel.selectedWindowID = "cg:2"
        viewModel.addSelectedTarget()
        api.windowsByApp["pid:43"] = []

        await viewModel.loadTargets()

        #expect(viewModel.selectedTargets.isEmpty)
        #expect(viewModel.targetError?.contains("Start setup again") == true)
    }

    @Test("Discovery failure clears every authorized target")
    func discoveryFailureClearsAuthorizedSet() async {
        let api = MockAgentAPI()
        let first = CUAWindowOption(
            windowID: "cg:1", index: 0, title: "First", x: nil, y: nil,
            width: nil, height: nil
        )
        let second = CUAWindowOption(
            windowID: "cg:2", index: 0, title: "Second", x: nil, y: nil,
            width: nil, height: nil
        )
        let viewModel = CUAViewModel(api: api)
        viewModel.appOptions = [
            CUAAppOption(name: "TextEdit", bundleID: "com.apple.TextEdit", pid: 42),
            CUAAppOption(name: "Notes", bundleID: "com.apple.Notes", pid: 43),
        ]
        viewModel.windowOptions = [first]
        viewModel.selectedPID = 42
        viewModel.selectedWindowID = first.windowID
        viewModel.addSelectedTarget()
        viewModel.windowOptions = [second]
        viewModel.selectedPID = 43
        viewModel.selectedWindowID = second.windowID
        viewModel.addSelectedTarget()
        api.discoveryShouldFail = true

        await viewModel.loadTargets()

        #expect(viewModel.selectedTargets.isEmpty)
        #expect(!viewModel.canStart)
        #expect(viewModel.targetError != nil)
    }

    @Test("Replacement session discovery and refresh use only the new API")
    func replacementSessionUsesCurrentDiscoveryAPI() async {
        let oldAPI = MockAgentAPI()
        oldAPI.appsResult = []
        let oldViewModel = CUAViewModel(api: oldAPI)
        await oldViewModel.loadTargets()

        let replacementAPI = MockAgentAPI()
        replacementAPI.appsResult = [
            CUAAppOption(name: "Safari", bundleID: "com.apple.Safari", pid: 23_429),
        ]
        let replacement = CUAViewModel(api: replacementAPI)
        await replacement.loadTargets()
        await replacement.loadTargets()

        #expect(oldAPI.appsCalls == 1)
        #expect(oldViewModel.appOptions.isEmpty)
        #expect(replacementAPI.appsCalls == 2)
        #expect(replacement.appOptions.map(\.pid) == [23_429])
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
        #expect(viewModel.targetError?.contains("could not find the apps") == true)
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
        #expect(viewModel.targetError?.contains("no longer open") == true)
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
        #expect(message.contains("Start setup again"))
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
        let discoveryCalls = api.appsCalls
        viewModel.phase = .failed(message: "binding could not be verified")
        #expect(!viewModel.canRetrySetup)
        await viewModel.retrySetup()
        #expect(viewModel.phase == .failed(message: "binding could not be verified"))
        #expect(api.appsCalls == discoveryCalls)
        #expect(viewModel.actionError?.contains("may still be executing") == true)
        viewModel.phase = .running

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
        #expect(viewModel.supportsPermissionRequest)
    }

    @Test("Accessibility request is explicit, capability gated, and refreshed")
    func accessibilityPermissionRequestUsesHelperAndRefreshes() async {
        let api = MockAgentAPI()
        api.permissionsResult = CUAPermissionStatus(
            accessibility: false, screenRecording: true
        )
        let viewModel = CUAViewModel(api: api)

        await viewModel.loadPermissions()
        #expect(api.requestedPermissions.isEmpty)
        let readsBeforeClick = api.permissionsCalls

        await viewModel.requestPermission(.accessibility)
        #expect(api.requestedPermissions == [.accessibility])
        #expect(api.permissionsCalls == readsBeforeClick + 1)
        #expect(viewModel.executorPermissions?.accessibility == false)
        #expect(viewModel.permissionRequestMessage?.contains("still shows") == true)
    }

    @Test("Screen Recording request comes from the main app and refreshes helper status")
    func screenRecordingPermissionRequestUsesMainAppAndRefreshes() async {
        let api = MockAgentAPI()
        api.permissionsResult = CUAPermissionStatus(
            accessibility: true, screenRecording: false
        )
        var mainAppRequests = 0
        let viewModel = CUAViewModel(
            api: api,
            mainAppScreenRecordingRequest: {
                mainAppRequests += 1
                return false
            }
        )

        await viewModel.loadPermissions()
        let readsBeforeClick = api.permissionsCalls
        await viewModel.requestPermission(.screenRecording)

        #expect(mainAppRequests == 1)
        #expect(api.requestedPermissions.isEmpty)
        #expect(api.permissionsCalls == readsBeforeClick + 1)
        #expect(viewModel.executorPermissions?.screenRecording == false)
        #expect(viewModel.permissionRequestMessage?.contains("Rapid-MLX Desktop") == true)
    }

    @Test("Unsupported servers block helper requests but not the main app Screen Recording request")
    func permissionRequestRequiresCapability() async {
        let api = MockAgentAPI()
        api.permissionRequestSupported = false
        var mainAppRequests = 0
        let viewModel = CUAViewModel(
            api: api,
            mainAppScreenRecordingRequest: {
                mainAppRequests += 1
                return false
            }
        )

        await viewModel.loadPermissions()
        await viewModel.requestPermission(.accessibility)
        await viewModel.requestPermission(.screenRecording)

        #expect(!viewModel.supportsPermissionRequest)
        #expect(api.requestedPermissions.isEmpty)
        #expect(mainAppRequests == 1)
    }

    @Test("Permission request errors still refresh helper status")
    func permissionRequestFailureRefreshes() async {
        let api = MockAgentAPI()
        api.permissionsResult = CUAPermissionStatus(
            accessibility: true, screenRecording: false
        )
        api.permissionRequestError = MockAgentAPI.Failure.requested
        let viewModel = CUAViewModel(api: api)

        await viewModel.loadPermissions()
        let readsBeforeClick = api.permissionsCalls
        await viewModel.requestPermission(.accessibility)

        #expect(api.requestedPermissions == [.accessibility])
        #expect(api.permissionsCalls == readsBeforeClick + 1)
        #expect(viewModel.executorPermissions?.screenRecording == false)
        #expect(viewModel.permissionRequestMessage != nil)
    }

    @Test("Detached session ignores a late permission load")
    func detachedSessionIgnoresLatePermissionLoad() async {
        let api = MockAgentAPI()
        api.permissionsResult = CUAPermissionStatus(
            accessibility: true, screenRecording: true
        )
        api.suspendPermissions = true
        let viewModel = CUAViewModel(api: api)

        let load = Task { await viewModel.loadPermissions() }
        while api.permissionsContinuation == nil { await Task.yield() }
        viewModel.detachFromSession()
        api.resumePermissions()
        await load.value

        #expect(viewModel.executorPermissions == nil)
        #expect(!viewModel.supportsPermissionRequest)
        #expect(viewModel.permissionRequestMessage == nil)
        #expect(viewModel.permissionRequestInFlight == nil)
    }

    @Test("Detached session stops a delayed permission request before refresh")
    func detachedSessionStopsLatePermissionRequest() async {
        let api = MockAgentAPI()
        let viewModel = CUAViewModel(api: api)
        await viewModel.loadPermissions()
        let readsBeforeRequest = api.permissionsCalls
        api.suspendPermissionRequest = true

        let request = Task { await viewModel.requestPermission(.accessibility) }
        while api.permissionRequestContinuation == nil { await Task.yield() }
        viewModel.detachFromSession()
        api.resumePermissionRequest()
        await request.value

        #expect(api.permissionsCalls == readsBeforeRequest)
        #expect(viewModel.executorPermissions == nil)
        #expect(!viewModel.supportsPermissionRequest)
        #expect(viewModel.permissionRequestMessage == nil)
        #expect(viewModel.permissionRequestInFlight == nil)
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
        #expect(!viewModel.browserAutomationRecoveryRequired)
    }

    @Test("Typed browser Automation failure offers persistent recovery")
    func automationPermissionFailureIsRecoverable() async {
        let api = MockAgentAPI()
        api.scriptedEvents = [
            makeEvent(
                seq: 2, kind: "terminal", status: "failed",
                reason: "browser URL access is not authorized",
                error: "automation_permission_required"
            ),
        ]
        let viewModel = CUAViewModel(api: api, pollIntervalNanos: 5_000_000)
        viewModel.goal = "check the selected page"
        viewModel.plannerName = "reviewed-brain"
        viewModel.maxSteps = 17
        selectTarget(viewModel)

        await viewModel.start()
        await drain()

        #expect(viewModel.phase == .failed(message: "browser URL access is not authorized"))
        #expect(viewModel.browserAutomationRecoveryRequired)
        #expect(viewModel.runContext?.allowedDomain == "example.com")

        let createCount = api.createdRequests.count
        await viewModel.retrySetup()

        #expect(!viewModel.browserAutomationRecoveryRequired)
        #expect(viewModel.phase == .idle)
        #expect(viewModel.goal == "check the selected page")
        #expect(viewModel.plannerName == "reviewed-brain")
        #expect(viewModel.maxSteps == 17)
        #expect(viewModel.runContext == nil)
        #expect(viewModel.events.isEmpty)
        #expect(viewModel.selectedPID == nil)
        #expect(viewModel.selectedWindowID == nil)
        #expect(viewModel.selectedTargets.isEmpty)
        #expect(viewModel.allowedDomain.isEmpty)
        #expect(!viewModel.canStart)
        #expect(api.createdRequests.count == createCount)
        #expect(api.appsCalls == 1)
        #expect(viewModel.targetError?.contains("choose the apps") == true)
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
        #expect(message.contains("Start setup again"))
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

    @Test("Multi-target create omits legacy binding and verifies the full echoed set")
    func multiTargetCreateContract() async throws {
        let targets = [
            CUARunTarget(
                targetID: "target_1", app: "pid:42", pid: 42,
                windowID: "cg:1", allowedDomain: "example.com"
            ),
            CUARunTarget(
                targetID: "target_2", app: "pid:43", pid: 43,
                windowID: "cg:2", allowedDomain: ""
            ),
        ]
        let response = CUARunCreated(
            runID: "multi", status: "running", windowID: "cg:1",
            clientRequestID: "request-multi", targets: targets,
            activeTargetID: "target_1"
        )
        RecordingURLProtocol.stubResponse(
            path: "/v1/cua/runs", body: try JSONEncoder().encode(response)
        )
        let request = CUARunRequest(
            app: "pid:42", goal: "read then edit", planner: "brain", openURL: "",
            allowedDomain: "", maxSteps: 12, humanLogin: true, windowID: "cg:1",
            clientRequestID: "request-multi", targets: targets,
            initialTargetID: "target_1"
        )

        #expect(try await makeClient().create(request) == "multi")
        let sent = try #require(RecordingURLProtocol.captured["/v1/cua/runs"]?.body)
        let json = try #require(JSONSerialization.jsonObject(with: sent) as? [String: Any])
        #expect(json["targets"] != nil)
        #expect(json["initial_target_id"] as? String == "target_1")
        #expect(json["window_id"] == nil)
        #expect(json["allowed_domain"] == nil)
    }

    @Test("Multi-target create mismatch cancels the concrete run")
    func multiTargetCreateMismatchFailsClosed() async throws {
        let requested = [
            CUARunTarget(
                targetID: "target_1", app: "pid:42", pid: 42,
                windowID: "cg:1", allowedDomain: "example.com"
            ),
            CUARunTarget(
                targetID: "target_2", app: "pid:43", pid: 43,
                windowID: "cg:2", allowedDomain: ""
            ),
        ]
        RecordingURLProtocol.stubResponse(
            path: "/v1/cua/runs",
            body: try JSONEncoder().encode(
                CUARunCreated(
                    runID: "unsafe", status: "running", windowID: "cg:1",
                    clientRequestID: "request-multi", targets: Array(requested.reversed()),
                    activeTargetID: "target_1"
                )
            )
        )
        RecordingURLProtocol.stubResponse(path: "/v1/cua/runs/unsafe/cancel")
        let request = CUARunRequest(
            app: "pid:42", goal: "g", planner: "brain", openURL: "",
            allowedDomain: "", maxSteps: 12, humanLogin: true, windowID: "cg:1",
            clientRequestID: "request-multi", targets: requested,
            initialTargetID: "target_1"
        )

        do {
            _ = try await makeClient().create(request)
            Issue.record("expected target binding rejection")
        } catch let error as CUAClientError {
            #expect(error == .targetBinding(runID: "unsafe", cancellationFailed: false))
        }
        #expect(RecordingURLProtocol.captured["/v1/cua/runs/unsafe/cancel"] != nil)
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

    @Test("Permission request sends the authenticated helper contract")
    func permissionRequestContract() async throws {
        RecordingURLProtocol.stubResponse(
            path: "/v1/cua/permissions/request",
            body: Data(
                #"{"permission":"screen_recording","granted":false,"permissions":{"accessibility":true,"screen_recording":false}}"#.utf8
            )
        )

        let result = try await makeClient().requestPermission(.screenRecording)
        let captured = try #require(
            RecordingURLProtocol.captured["/v1/cua/permissions/request"]
        )
        let json = try #require(
            JSONSerialization.jsonObject(with: captured.body) as? [String: Any]
        )

        #expect(captured.request.httpMethod == "POST")
        #expect(captured.request.value(forHTTPHeaderField: "Authorization") == "Bearer tok")
        #expect(json["permission"] as? String == "screen_recording")
        #expect(json.count == 1)
        #expect(!result.granted)
        #expect(result.permissions.accessibility)
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
struct CUATaskFirstResolutionTests {
  private actor AutomationGate {
    private var continuation: CheckedContinuation<Void, Never>?
    private(set) var requestedBundleID: String?

    func wait(bundleID: String) async -> BrowserAutomationAuthorization {
      requestedBundleID = bundleID
      await withCheckedContinuation { continuation = $0 }
      return .authorized
    }

    func release() {
      continuation?.resume()
      continuation = nil
    }
  }

  @MainActor
  private static func viewModel(
    api: MockAgentAPI,
    browserAutomationRequest: @escaping @Sendable (String) async -> BrowserAutomationAuthorization = { _ in .authorized }
  ) -> CUAViewModel {
    let vm = CUAViewModel(api: api, browserAutomationRequest: browserAutomationRequest)
    vm.plannerOptions = [
      CUAPlannerOption(
        name: "local-27b", model: "local", url: "http://127.0.0.1:8080/v1",
        textOnly: true
      )
    ]
    return vm
  }

  private static func target(
    id: String = "target_1", pid: Int = 42, domain: String = ""
  ) -> CUATargetProposal {
    CUATargetProposal(
      targetID: id, app: "pid:\(pid)", pid: pid,
      windowID: "cg:\(pid)", allowedDomain: domain,
      displayName: domain.isEmpty ? "Notes — Project" : "Browser — \(domain)",
      bundleID: domain.isEmpty ? "com.apple.Notes" : "com.apple.Safari",
      processStartTime: 1234.5
    )
  }

  @MainActor
  @Test func resolvedSingleTargetStartsWithLegacyWireShape() async {
    let api = MockAgentAPI()
    api.targetResolutionResult = CUATargetResolution(
      status: "resolved", targets: [Self.target()], initialTargetID: "target_1",
      reason: "", approval: nil
    )
    let vm = Self.viewModel(api: api)
    vm.goal = "Summarize the project note"

    await vm.resolveAndStart()

    #expect(api.targetResolutionRequests.count == 1)
    let request = try? #require(api.attemptedRequests.first)
    #expect(request?.app == "pid:42")
    #expect(request?.windowID == "cg:42")
    #expect(request?.targets == nil)
    #expect(request?.initialTargetID == nil)
    #expect(request?.bundleID == "com.apple.Notes")
    #expect(request?.processStartTime == 1234.5)
    #expect(vm.runContext?.targetDisplayName == "Notes — Project")
  }

  @MainActor
  @Test func websiteScopeWaitsForPlainLanguageApproval() async {
    let api = MockAgentAPI()
    let target = Self.target(domain: "example.test")
    api.targetResolutionResult = CUATargetResolution(
      status: "needs_approval", targets: [target], initialTargetID: target.targetID,
      reason: "",
      approval: CUATargetApproval(
        kind: "website_scope",
        prompt:
          "Rapid plans to work in Browser. Website access is limited to example.test. Continue?",
        options: [
          CUATargetApprovalOption(
            optionID: "use_proposed", label: "Use this website", targetIDs: [target.targetID]
          )
        ]
      )
    )
    let vm = Self.viewModel(api: api)
    vm.goal = "Summarize the open article"

    await vm.resolveAndStart()
    #expect(api.createAttempts == 0)
    #expect(vm.targetResolutionApproval?.approval?.kind == "website_scope")

    await vm.approveResolvedTargets(optionID: "use_proposed")
    #expect(api.createAttempts == 1)
    #expect(api.attemptedRequests.first?.allowedDomain == "example.test")
  }

  @MainActor
  @Test func firstUseBrowserAutomationRerunsResolutionBeforeWebsiteApproval() async {
    let api = MockAgentAPI()
    let target = Self.target(domain: "example.test")
    api.queuedTargetResolutionResults = [
      CUATargetResolution(
        status: "needs_automation", targets: [], initialTargetID: nil,
        reason: "Browser access is needed.", approval: nil,
        automation: CUATargetAutomationRequest(
          bundleID: "com.apple.Safari", displayName: "Safari"
        )
      ),
      CUATargetResolution(
        status: "needs_approval", targets: [target], initialTargetID: target.targetID,
        reason: "", approval: CUATargetApproval(
          kind: "website_scope", prompt: "Use example.test?",
          options: [
            CUATargetApprovalOption(
              optionID: "site", label: "Use this website", targetIDs: [target.targetID]
            )
          ]
        )
      ),
    ]
    let vm = Self.viewModel(api: api) { bundleID in
      #expect(bundleID == "com.apple.Safari")
      return .authorized
    }
    vm.goal = "Read the open article"

    await vm.resolveAndStart()

    #expect(api.targetResolutionRequests.count == 2)
    #expect(api.createAttempts == 0)
    #expect(vm.targetResolutionApproval?.approval?.kind == "website_scope")

    await vm.approveResolvedTargets(optionID: "site")
    #expect(api.createAttempts == 1)
    #expect(api.attemptedRequests.first?.allowedDomain == "example.test")
  }

  @MainActor
  @Test func deniedBootstrapAutomationCreatesNoRun() async {
    let api = MockAgentAPI()
    api.targetResolutionResult = CUATargetResolution(
      status: "needs_automation", targets: [], initialTargetID: nil,
      reason: "Browser access is needed.", approval: nil,
      automation: CUATargetAutomationRequest(
        bundleID: "com.apple.Safari", displayName: "Safari"
      )
    )
    let vm = Self.viewModel(api: api) { _ in .denied }
    vm.goal = "Read the open article"

    await vm.resolveAndStart()

    #expect(api.createAttempts == 0)
    #expect(api.targetResolutionRequests.count == 1)
    #expect(vm.browserAutomationRecoveryRequired)
  }

  @MainActor
  @Test func changingTaskInvalidatesPendingScope() async {
    let api = MockAgentAPI()
    let target = Self.target()
    api.targetResolutionResult = CUATargetResolution(
      status: "needs_approval", targets: [target], initialTargetID: target.targetID,
      reason: "",
      approval: CUATargetApproval(
        kind: "ambiguity", prompt: "Use Notes?",
        options: [
          CUATargetApprovalOption(
            optionID: "notes", label: "Use Notes", targetIDs: [target.targetID]
          )
        ]
      )
    )
    let vm = Self.viewModel(api: api)
    vm.goal = "Read the note"
    await vm.resolveAndStart()

    vm.goal = "Delete the note"
    await vm.approveResolvedTargets(optionID: "notes")

    #expect(api.createAttempts == 0)
  }

  @MainActor
  @Test func deniedBrowserAutomationBlocksCreateAndOffersRecovery() async {
    let api = MockAgentAPI()
    let target = Self.target(domain: "example.test")
    api.targetResolutionResult = CUATargetResolution(
      status: "needs_approval", targets: [target], initialTargetID: target.targetID,
      reason: "", approval: CUATargetApproval(
        kind: "website_scope", prompt: "Use example.test?",
        options: [
          CUATargetApprovalOption(
            optionID: "site", label: "Use this website", targetIDs: [target.targetID]
          )
        ]
      )
    )
    let vm = Self.viewModel(api: api) { bundleID in
      #expect(bundleID == "com.apple.Safari")
      return .denied
    }
    vm.goal = "Read the page"

    await vm.resolveAndStart()
    await vm.approveResolvedTargets(optionID: "site")

    #expect(api.createAttempts == 0)
    #expect(vm.browserAutomationRecoveryRequired)
    #expect(vm.selectedTargets.isEmpty)
    #expect(vm.targetError?.contains("Browser control is not allowed") == true)
  }

  @MainActor
  @Test func remoteModelRequiresPerTaskAppDiscoveryConsent() async {
    let api = MockAgentAPI()
    let vm = CUAViewModel(api: api)
    vm.plannerName = "cloud"
    vm.plannerOptions = [
      CUAPlannerOption(
        name: "cloud", model: "custom", url: "https://models.example/v1",
        textOnly: true, allowRemote: true
      )
    ]
    vm.goal = "Read the note"

    #expect(!vm.canResolveTask)
    vm.allowRemoteAppDiscovery = true
    #expect(vm.canResolveTask)
    await vm.resolveAndStart()

    #expect(api.targetResolutionRequests.first?.allowRemoteAppDiscovery == true)
  }

  @MainActor
  @Test func automationPreflightKeepsResolvedScopeFrozen() async {
    let api = MockAgentAPI()
    let safari = Self.target(domain: "example.test")
    api.targetResolutionResult = CUATargetResolution(
      status: "needs_approval", targets: [safari], initialTargetID: safari.targetID,
      reason: "", approval: CUATargetApproval(
        kind: "website_scope", prompt: "Use example.test?",
        options: [
          CUATargetApprovalOption(
            optionID: "site", label: "Use this website", targetIDs: [safari.targetID]
          )
        ]
      )
    )
    let gate = AutomationGate()
    let vm = Self.viewModel(api: api) { await gate.wait(bundleID: $0) }
    vm.goal = "Read the page"
    await vm.resolveAndStart()

    let approval = Task { await vm.approveResolvedTargets(optionID: "site") }
    while await gate.requestedBundleID == nil { await Task.yield() }
    #expect(!vm.canResolveTask)

    let chrome = CUATargetProposal(
      targetID: "target_2", app: "pid:99", pid: 99, windowID: "cg:99",
      allowedDomain: "other.test", displayName: "Browser — other.test",
      bundleID: "com.google.Chrome", processStartTime: 2345.6
    )
    api.targetResolutionResult = CUATargetResolution(
      status: "resolved", targets: [chrome], initialTargetID: chrome.targetID,
      reason: "", approval: nil
    )
    await vm.resolveAndStart()
    #expect(api.targetResolutionRequests.count == 1)
    #expect(api.createAttempts == 0)

    await gate.release()
    await approval.value
    #expect(api.createAttempts == 1)
    #expect(api.attemptedRequests.first?.bundleID == "com.apple.Safari")
  }
}

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
  @Test func plannerLoadSelectsAnAvailableModelAndDeleteIsUserCreatedOnly() async {
    let api = MockAgentAPI()
    api.plannersResult = [
      CUAPlannerOption(
        name: "built-in", model: "local", url: "http://127.0.0.1:8080/v1",
        textOnly: true
      ),
      CUAPlannerOption(
        name: "mine", model: "custom", url: "https://models.example/v1",
        textOnly: true, userCreated: true, allowRemote: true
      ),
    ]
    let vm = CUAViewModel(api: api)
    vm.plannerName = "missing"

    await vm.loadPlanners()
    #expect(vm.plannerName == "built-in")

    await vm.deleteModel(named: "built-in")
    #expect(api.deletedPlannerNames.isEmpty)

    await vm.deleteModel(named: "mine")
    #expect(api.deletedPlannerNames == ["mine"])
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
    #expect(vm.plannerDisclosure.contains("external model"))
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
    @Test("Sidebar exposes only active and approval Computer Use states")
    func sidebarStatusPresentation() {
        #expect(CUASidebarStatus(phase: .starting) == .running)
        #expect(CUASidebarStatus(phase: .running) == .running)
        #expect(CUASidebarStatus(phase: .awaitingApproval) == .awaitingApproval)
        #expect(CUASidebarStatus(phase: .idle) == nil)
        #expect(CUASidebarStatus(phase: .finished(summary: "done")) == nil)
        #expect(CUASidebarStatus(phase: .failed(message: "stopped")) == nil)
        #expect(CUASidebarStatus.awaitingApproval.accessibilityValue == "Approval needed")
    }

    @Test("Completed result separates recognized evidence without losing the answer")
    func completedResultPresentation() {
        let result = CUAResultPresentation(
            summary: "Created the folder and moved 3 files.\nSupporting evidence: Finder shows all 3 files."
        )

        #expect(result.answer == "Created the folder and moved 3 files.")
        #expect(result.evidence == "Finder shows all 3 files.")
        #expect(result.detailLabel == "Supporting evidence")
        #expect(result.completionNote == nil)
    }

    @Test("Free-form completed result remains intact")
    func freeFormCompletedResultPresentation() {
        let summary = "Compared both documents. The second contains the newer pricing and evidence gathered from the page."
        let result = CUAResultPresentation(summary: summary)

        #expect(result.answer == summary)
        #expect(result.evidence == nil)
        #expect(result.completionNote == nil)
    }

    @Test("Long free-form result is concise without losing the original")
    func longFreeFormCompletedResultPresentation() {
        let summary = "The selected folder is tmp. "
            + "The accessibility snapshot reports the title, selection count, sidebar locations, storage volumes, tags, and status details. "
            + String(repeating: "Additional raw accessibility context remains available. ", count: 4)
        let result = CUAResultPresentation(summary: summary)

        #expect(result.answer.count <= 220)
        #expect(result.answer.hasPrefix("The selected folder is tmp."))
        #expect(result.answer != summary.trimmingCharacters(in: .whitespacesAndNewlines))
        #expect(result.evidence == summary.trimmingCharacters(in: .whitespacesAndNewlines))
        #expect(result.detailLabel == "Full result")
        #expect(result.completionNote == nil)
    }

    @Test("Long structured result keeps its complete text in details")
    func longStructuredCompletedResultPresentation() {
        let answer = String(repeating: "Confirmed the requested state with the selected window. ", count: 6)
        let summary = answer + "Supporting evidence: AX title and status values matched."
        let result = CUAResultPresentation(summary: summary)

        #expect(result.answer.count <= 220)
        #expect(result.evidence == summary)
        #expect(result.detailLabel == "Full result")
        #expect(result.completionNote == "Supporting evidence: AX title and status values matched.")
    }

    @Test("Long unpunctuated result reserves room for its ellipsis")
    func longUnpunctuatedCompletedResultPresentation() {
        let summary = String(repeating: "状态已确认", count: 60)
        let result = CUAResultPresentation(summary: summary)

        #expect(result.answer.count <= 220)
        #expect(result.answer.hasSuffix("…"))
        #expect(result.evidence == summary)
        #expect(result.detailLabel == "Full result")
        #expect(result.completionNote == nil)
    }

    @Test("Long final sentence is bounded from its conclusion")
    func longFinalSentenceIsBounded() {
        let summary = String(repeating: "Verified context. ", count: 14)
            + String(repeating: "persistence remains unconfirmed ", count: 12)
        let result = CUAResultPresentation(summary: summary)

        #expect(result.completionNote?.count == 180)
        #expect(result.completionNote?.hasPrefix("…") == true)
        #expect(result.completionNote?.hasSuffix("unconfirmed") == true)
        #expect(result.evidence == summary.trimmingCharacters(in: .whitespacesAndNewlines))
    }

    @Test("Completed task without a summary retains a visible result")
    func emptyCompletedResultPresentation() {
        let result = CUAResultPresentation(summary: "  \n")

        #expect(result.answer.isEmpty)
        #expect(result.displayedAnswer == "The task completed without a result summary.")
        #expect(result.evidence == nil)
        #expect(result.detailLabel == "Supporting evidence")
        #expect(result.completionNote == nil)
    }

    @Test("Long completed result keeps its final completion condition visible")
    func longCompletedResultShowsCompletionNote() {
        let summary = "The text area [4] now contains both lines: 'CUA SOTA TextEdit test.' followed by 'Rapid-MLX CUA SOTA verified.' (first line unchanged). However, the 'document actions' menu button [3] was clicked three times without revealing a Save option in the accessibility snapshot, so the save could not be confirmed via the UI. The document title [6] shows 'harbor-desk-cua-textedit-sota-test.txt'. Honest blocker: Save action could not be verified; the appended text is present but persistence is unconfirmed."
        let result = CUAResultPresentation(summary: summary)

        #expect(result.answer.hasPrefix("The text area [4] now contains both lines"))
        #expect(
            result.completionNote
                == "Honest blocker: Save action could not be verified; the appended text is present but persistence is unconfirmed."
        )
        #expect(result.evidence == summary)
    }

    @Test("Completion note does not repeat an already visible answer")
    func completionNoteDoesNotRepeatAnswer() {
        let repeated = "The requested result remains unconfirmed."
        let summary = String(repeating: repeated + " ", count: 8)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let result = CUAResultPresentation(summary: summary)

        #expect(result.answer.contains(repeated))
        #expect(result.completionNote == nil)
        #expect(result.evidence == summary)
    }

    @Test("Planner payload is folded behind a readable failure summary")
    func plannerPayloadFailurePresentation() {
        let message = #"planner produced invalid plans: model returned no JSON object: {"action":"done","final_summary":"raw"}"#
        let failure = CUAFailurePresentation(message: message)

        #expect(failure.summary == "planner produced invalid plans: model returned no JSON object")
        #expect(failure.technicalDetails == message)
    }

    @Test("Unknown long failure remains complete in technical details")
    func unknownLongFailurePresentation() {
        let message = String(repeating: "unexpected transport failure ", count: 20)
        let failure = CUAFailurePresentation(message: message)

        #expect(failure.summary.count <= 240)
        #expect(failure.summary.hasSuffix("…"))
        #expect(failure.technicalDetails == message.trimmingCharacters(in: .whitespacesAndNewlines))
    }

    @Test("Long text before a technical payload is also bounded")
    func longPayloadFailurePresentation() {
        let message = String(repeating: "planner explanation ", count: 20) + #": {"raw":true}"#
        let failure = CUAFailurePresentation(message: message)

        #expect(failure.summary.count <= 240)
        #expect(failure.technicalDetails == message)
    }

    @Test("Missing failure text still uses explicit failure semantics")
    func emptyFailurePresentation() {
        let failure = CUAFailurePresentation(message: " \n ")

        #expect(failure.summary == "The task could not be completed.")
        #expect(failure.technicalDetails == "No failure details were provided.")
    }

    @Test("Executed actions add a truthful partial-change warning")
    func executedActionFailureWarning() {
        func event(_ action: String) -> CUAEvent {
            CUAEvent(
                seq: 1, kind: "executed", step: 1, action: action,
                stepInstruction: nil, outcome: nil, targetLabel: nil,
                status: nil, finalSummary: nil, reason: nil
            )
        }
        let withAction = CUAFailurePresentation(
            message: "The task stopped.", hasExecutedActions: true
        )
        let withoutAction = CUAFailurePresentation(message: "The task stopped.")

        #expect(withAction.changeWarning == "Some changes may have been made; check the target app.")
        #expect(withoutAction.changeWarning == nil)
        #expect(CUAFailurePresentation.hasPotentialSideEffects(in: [
            event("click"),
        ]))
        #expect(CUAFailurePresentation.hasPotentialSideEffects(in: [
            event("save"),
        ]))
        #expect(!CUAFailurePresentation.hasPotentialSideEffects(in: [
            event("observe"), event("wait"),
        ]))
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
        let sidebar = try String(
            contentsOf: root.appendingPathComponent("Sources/Rapid/UI/SidebarView.swift"),
            encoding: .utf8
        )

    #expect(!section.contains("ComputerUse.Agent.Target.Process"))
    #expect(!section.contains("ComputerUse.Agent.Target.Window"))
    #expect(!section.contains("ComputerUse.Agent.Target.Domain"))
    #expect(section.contains("ComputerUse.Agent.ScopeApproval"))
    #expect(section.contains("ComputerUse.Agent.ScopeOption"))
    #expect(section.contains("ComputerUse.Agent.Model"))
    #expect(section.contains("ComputerUse.Agent.AddModel"))
    #expect(section.contains("ComputerUse.Agent.ManageModels"))
    #expect(section.contains("ComputerUse.Agent.ScopeCancel"))
    #expect(section.contains("ComputerUse.Agent.ModelRemove"))
    #expect(section.contains("ComputerUse.Agent.ModelsAdd"))
    #expect(section.contains("ComputerUse.Agent.ModelsDone"))
    #expect(section.contains("ComputerUse.Agent.ModelRemoveConfirm"))
    #expect(section.contains("ComputerUse.Agent.ModelRemoveCancel"))
        #expect(section.contains("ComputerUse.Agent.RunContext.Targets"))
        #expect(section.contains("ComputerUse.Agent.RunContext.Domain"))
        #expect(section.contains("ComputerUse.Agent.Approval.TargetWindow"))
        #expect(section.contains("ComputerUse.Agent.Approval.Domain"))
        #expect(section.contains("Retry Recovery"))
        #expect(section.contains("Describe what you want Rapid to do"))
        #expect(section.contains("accessibilityLabel(\"Task goal\")"))
        #expect(section.contains("ComputerUse.Agent.ActiveProgress"))
        #expect(section.contains("ComputerUse.Agent.Summary"))
        #expect(section.contains("ComputerUse.Agent.Summary.Answer"))
        #expect(section.contains("ComputerUse.Agent.Failure.Summary"))
        #expect(section.contains("ComputerUse.Agent.Failure.Details"))
        #expect(section.contains("ComputerUse.Agent.Failure.ChangeWarning"))
        #expect(section.contains("ComputerUse.Agent.Failure.AutomationRecovery"))
        #expect(section.contains("ComputerUse.Agent.Failure.OpenAutomationSettings"))
        #expect(section.contains("ComputerUse.Agent.Failure.RetrySetup"))
        #expect(section.contains("ComputerUse.Agent.Failure.RetrySetupHelp"))
        #expect(section.contains("ComputerUse.Agent.RunContext"))
        #expect(section.contains("ComputerUse.Agent.RunContext.Goal"))
        #expect(section.contains("ComputerUse.Agent.NewTask"))
        #expect(section.contains("ComputerUse.Agent.Evidence"))
        #expect(section.contains("ComputerUse.Agent.History"))
        #expect(section.contains("DisclosureGroup"))
        #expect(section.contains("Task complete"))
        #expect(section.contains("Task failed"))
        #expect(section.contains("FAILED"))
        #expect(section.contains("if case let .finished(summary) = viewModel.phase"))
        #expect(section.contains("COMPLETED"))
        #expect(section.contains("RUNNING"))
        #expect(page.contains("ComputerUse.Server.Starting"))
        #expect(page.contains("ComputerUse.Server.Error"))
        #expect(page.contains("ComputerUse.Server.Retry"))
        #expect(page.contains("Actions run on this Mac"))
        #expect(page.contains("local or cloud model you choose"))
        #expect(!page.contains("EXPERIMENTAL"))
        let computerUsePosition = try #require(sidebar.range(of: "if computerUseEnabled"))
        let experimentalPosition = try #require(sidebar.range(of: "SectionHeader(\"Experimental\")"))
        #expect(computerUsePosition.lowerBound < experimentalPosition.lowerBound)
        #expect(!page.contains("Everything runs locally"))
        #expect(page.contains("CUASection(viewModel: cuaViewModel)"))
        #expect(!page.contains("Start with a flow"))
        #expect(!page.contains("CREATE YOUR OWN"))
        #expect(!page.contains("ComputerUseStarter"))
        #expect(!page.contains("DraftPostFlowSheet"))
        #expect(!page.contains("FreeUpSpaceFlowSheet"))
        #expect(!content.contains("languageRuntime: DraftPostLanguageRuntime"))
        #expect(!content.contains("visualRuntime: DraftPostVisualRuntime"))
        #expect(content.contains("cuaViewModel: cuaServer.viewModel"))
        #expect(content.contains(".id(cuaServer.sessionID)"))
        #expect(sidebar.contains("@ObservedObject var viewModel: CUAViewModel"))
        #expect(sidebar.contains("accessibilityValue"))
        #expect(sidebar.contains("Approval needed"))
        #expect(sidebar.contains("Sidebar.ComputerUse"))
    }
}
