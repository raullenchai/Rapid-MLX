import Foundation

/// Transport contract for the agent-task panel, so the view model can be
/// unit-tested without a live server.
protocol CUAAPI: Sendable {
    func capabilities() async throws -> CUACapabilities
    func apps() async throws -> [CUAAppOption]
    func windows(app: String) async throws -> [CUAWindowOption]
    func planners() async throws -> [CUAPlannerOption]
    func addPlanner(_ request: CUAPlannerCreateRequest) async throws
    func deletePlanner(name: String) async throws
    func create(_ request: CUARunRequest) async throws -> String
    func run(clientRequestID: String) async throws -> CUARunCreated
    func permissions() async throws -> CUAPermissionStatus
    func events(runID: String, after: Int) async throws -> CUARunView
    func approve(runID: String, gateID: String) async throws
    func cancel(runID: String) async throws
}

struct CUACapabilities: Codable, Equatable, Sendable {
    struct Features: Codable, Equatable, Sendable {
        var idempotentRunCreate: Bool

        enum CodingKeys: String, CodingKey {
            case idempotentRunCreate = "idempotent_run_create"
        }
    }

    var features: Features
}

/// Phase of the agent-task panel.
enum CUAPhase: Equatable, Sendable {
    case idle
    case starting
    case running
    case awaitingApproval
    case finished(summary: String)
    case failed(message: String)

    var isBusy: Bool {
        switch self {
        case .starting, .running, .awaitingApproval:
            return true
        case .idle, .finished, .failed:
            return false
        }
    }
}

struct CUAPendingApproval: Equatable, Sendable {
    var gateID: String? = nil
    var app: String
    var action: String?
    var target: String?
    var reason: String
}

struct CUAPermissionStatus: Codable, Equatable, Sendable {
    var accessibility: Bool
    var screenRecording: Bool?

    enum CodingKeys: String, CodingKey {
        case accessibility
        case screenRecording = "screen_recording"
    }

    var isReady: Bool { accessibility && screenRecording == true }
}

struct CUAProgressPresentation: Equatable, Sendable {
    var step: Int
    var maxSteps: Int
    var instruction: String
    var action: String?
    var target: String?
    var outcome: String?

    var outcomeLabel: String? {
        switch outcome {
        case "success": "Observed expected change"
        case "no_effect": "No effect observed"
        case "wrong_effect": "Unexpected result observed"
        case "uncertain": "Could not verify"
        case "unavailable": "Verification unavailable"
        case .some: "Outcome not recognized"
        case nil: nil
        }
    }
}

/// Drives one agent task from the GUI: create on the app-owned server, poll
/// numbered events, surface gate approvals, and finish with the summary.
@MainActor
final class CUAViewModel: ObservableObject {
    @Published var goal = ""
    @Published var appName = "Google Chrome"
    @Published var plannerName = "local-27b"
    @Published var openURL = ""
    @Published var allowedDomain = ""
    @Published var maxSteps = 12
    @Published var phase: CUAPhase = .idle
    @Published var events: [CUAEvent] = []
    @Published var plannerOptions: [CUAPlannerOption] = []
    @Published var pendingGateReason: String?
    @Published var pendingApproval: CUAPendingApproval?
    @Published var executorPermissions: CUAPermissionStatus?
    @Published var actionError: String?
    @Published var appOptions: [CUAAppOption] = []
    @Published var windowOptions: [CUAWindowOption] = []
    @Published var selectedPID: Int?
    @Published var selectedWindowID: String?
    @Published var targetError: String?
    @Published var isLoadingApps = false
    @Published var isLoadingWindows = false
    @Published private(set) var isStopping = false

    private let api: CUAAPI
    private var runID: String?
    private var pollTask: Task<Void, Never>?
    private var showingPollError = false
    private let pollIntervalNanos: UInt64
    private var targetDiscoveryGeneration = 0
    private var requiresBindingCleanup = false

    private static let bindingCleanupWarning =
        "Warning: this unverified task may still be executing. Stop it immediately before retrying."
    private static let createRecoveryWarning =
        "Rapid could not confirm whether the task started. It may still be executing. Restore the local server connection, then use Stop again. Start remains locked."
    private var lifecycleGeneration = 0
    private var stoppingStartGeneration: Int?
    private var pendingCreateRecovery: CUARunRequest?

    init(api: CUAAPI?, pollIntervalNanos: UInt64 = 700_000_000) {
        self.api = api ?? NullCUAAPI()
        self.pollIntervalNanos = pollIntervalNanos
    }

    var canStart: Bool {
        !goal.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && !phase.isBusy
            && selectedApp != nil
            && selectedWindow != nil
            && openURL.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    var selectedApp: CUAAppOption? {
        guard let selectedPID else { return nil }
        return appOptions.first { $0.pid == selectedPID }
    }

    var selectedWindow: CUAWindowOption? {
        guard let selectedWindowID else { return nil }
        return windowOptions.first { $0.windowID == selectedWindowID }
    }

    var targetSummary: String? {
        guard let app = selectedApp, let window = selectedWindow else { return nil }
        return "\(window.displayName) in \(app.displayName)"
    }

    var canApprove: Bool {
        phase == .awaitingApproval && pendingApproval?.gateID != nil
            && !requiresBindingCleanup && !isStopping
    }

    var isRecoveringCreate: Bool { pendingCreateRecovery != nil && runID == nil }

    var approvalUnavailableMessage: String? {
        guard phase == .awaitingApproval, !canApprove else { return nil }
        if isStopping {
            return "Rapid is stopping this task. Approval is unavailable while cancellation finishes."
        }
        if requiresBindingCleanup {
            return "Rapid could not verify this task start. Approval is disabled; stop the task immediately."
        }
        return "This approval is missing its gate identity. Stop the task and retry."
    }

    var activeProgress: CUAProgressPresentation? {
        guard let plan = events.last(where: { $0.kind == "plan" }), let step = plan.step else {
            return nil
        }
        let outcome = events.last(where: {
            $0.kind == "executed" && $0.step == step
        })?.outcome
        return CUAProgressPresentation(
            step: step,
            maxSteps: maxSteps,
            instruction: plan.stepInstruction ?? "Planning the next action",
            action: plan.action,
            target: plan.targetLabel,
            outcome: outcome
        )
    }

    func loadPlanners() async {
        plannerOptions = (try? await api.planners()) ?? []
    }

    func loadPermissions() async {
        executorPermissions = try? await api.permissions()
    }

    func loadTargets() async {
        targetDiscoveryGeneration += 1
        let generation = targetDiscoveryGeneration
        isLoadingApps = true
        isLoadingWindows = false
        targetError = nil
        defer {
            if generation == targetDiscoveryGeneration { isLoadingApps = false }
        }
        do {
            let discovered = try await api.apps().sorted {
                ($0.name ?? $0.bundleID ?? "").localizedCaseInsensitiveCompare(
                    $1.name ?? $1.bundleID ?? ""
                ) == .orderedAscending
            }
            guard generation == targetDiscoveryGeneration else { return }
            appOptions = discovered
            guard let selectedPID, discovered.contains(where: { $0.pid == selectedPID }) else {
                self.selectedPID = nil
                selectedWindowID = nil
                windowOptions = []
                return
            }
            await loadWindows(
                pid: selectedPID, preserveSelection: true, generation: generation
            )
        } catch {
            guard generation == targetDiscoveryGeneration else { return }
            appOptions = []
            windowOptions = []
            selectedPID = nil
            selectedWindowID = nil
            targetError = targetDiscoveryMessage(error)
        }
    }

    func selectApp(pid: Int?) async {
        targetDiscoveryGeneration += 1
        let generation = targetDiscoveryGeneration
        isLoadingApps = false
        selectedPID = pid
        selectedWindowID = nil
        windowOptions = []
        targetError = nil
        guard let pid else { return }
        appName = appOptions.first(where: { $0.pid == pid })?.name ?? "PID \(pid)"
        await loadWindows(pid: pid, preserveSelection: false, generation: generation)
    }

    func refreshWindows() async {
        guard let selectedPID else { return }
        targetDiscoveryGeneration += 1
        let generation = targetDiscoveryGeneration
        isLoadingApps = false
        await loadWindows(
            pid: selectedPID, preserveSelection: true, generation: generation
        )
    }

    private func loadWindows(pid: Int, preserveSelection: Bool, generation: Int) async {
        isLoadingWindows = true
        targetError = nil
        defer {
            if generation == targetDiscoveryGeneration { isLoadingWindows = false }
        }
        let oldSelection = preserveSelection ? selectedWindowID : nil
        do {
            let discovered = try await api.windows(app: "pid:\(pid)")
            guard generation == targetDiscoveryGeneration, selectedPID == pid else { return }
            windowOptions = discovered
            selectedWindowID = discovered.contains(where: { $0.windowID == oldSelection })
                ? oldSelection : nil
            if discovered.isEmpty {
                targetError = "No usable windows were found for this process. Open a window, then refresh."
            } else if oldSelection != nil, selectedWindowID == nil {
                targetError = "The selected window is no longer available. Choose a window again."
            }
        } catch {
            guard generation == targetDiscoveryGeneration, selectedPID == pid else { return }
            windowOptions = []
            selectedWindowID = nil
            targetError = targetDiscoveryMessage(error)
        }
    }

    private func targetDiscoveryMessage(_ error: Error) -> String {
        if case let CUAClientError.typedHTTP(_, code, message, recovery) = error {
            let hint = recovery.first.map { " \($0)" } ?? ""
            switch code {
            case "app_not_found":
                return "The selected app process is no longer running. Refresh the app list.\(hint)"
            case "window_not_found":
                return "No usable windows were found for this process. Open a window, then refresh.\(hint)"
            default:
                return "Could not load apps and windows: \(message)\(hint)"
            }
        }
        if case let CUAClientError.http(code, _) = error, code == 404 {
            return "This local server does not support window selection. Update or restart Rapid, then refresh."
        }
        return "Could not load apps and windows: \(Self.describe(error)) Refresh to try again."
    }

    // MARK: Add-brain settings

    @Published var showAddBrain = false
    @Published var newBrainName = ""
    @Published var newBrainURL = ""
    @Published var newBrainModel = ""
    @Published var newBrainAPIKey = ""
    @Published var newBrainTextOnly = false
    @Published var newBrainAllowRemote = false
    @Published var brainError: String?

    var addBrainIsValid: Bool {
        !newBrainName.trimmingCharacters(in: .whitespaces).isEmpty
            && !newBrainURL.trimmingCharacters(in: .whitespaces).isEmpty
            && !newBrainModel.trimmingCharacters(in: .whitespaces).isEmpty
    }

    func saveBrain() async {
        let api = self.api
        brainError = nil
        do {
            try await api.addPlanner(
                CUAPlannerCreateRequest(
                    name: newBrainName,
                    url: newBrainURL.trimmingCharacters(in: .whitespaces),
                    model: newBrainModel.trimmingCharacters(in: .whitespaces),
                    apiKey: newBrainAPIKey.isEmpty ? nil : newBrainAPIKey,
                    reasoningEffort: nil,
                    textOnly: newBrainTextOnly,
                    allowRemote: newBrainAllowRemote
                )
            )
            newBrainName = ""
            newBrainURL = ""
            newBrainModel = ""
            newBrainAPIKey = ""
            newBrainTextOnly = false
            newBrainAllowRemote = false
            showAddBrain = false
            await loadPlanners()
        } catch {
            brainError = String(describing: error)
        }
    }

    var selectedPlanner: CUAPlannerOption? {
        plannerOptions.first { $0.name == plannerName }
    }

    var plannerDisclosure: String {
        guard let planner = selectedPlanner else {
            return "Actions run on this Mac. Select a brain to review what it receives."
        }
        if Self.isLoopbackEndpoint(planner.url) {
            return planner.textOnly
                ? "Actions run on this Mac. The planner request goes to the configured loopback endpoint, which may forward it. It receives the goal and Accessibility snapshot."
                : "Actions run on this Mac. The planner request goes to the configured loopback endpoint, which may forward it. It receives the goal, Accessibility snapshot, and screenshot."
        }
        return planner.textOnly
            ? "Actions run on this Mac. This external brain receives the goal and Accessibility snapshot over HTTPS."
            : "Actions run on this Mac. This external brain receives the goal, Accessibility snapshot, and screenshot over HTTPS."
    }

    static func isLoopbackEndpoint(_ value: String) -> Bool {
        guard
            let url = URL(string: value),
            url.scheme == "http" || url.scheme == "https",
            var host = url.host?.lowercased()
        else { return false }
        while host.hasSuffix(".") { host.removeLast() }
        if host == "localhost" || host == "::1" || host == "[::1]" { return true }
        let octets = host.split(separator: ".", omittingEmptySubsequences: false)
        guard octets.count == 4 else { return false }
        let numbers = octets.compactMap { Int($0) }
        return numbers.count == 4 && numbers[0] == 127
            && numbers.allSatisfy { (0 ... 255).contains($0) }
    }

    func start() async {
        guard canStart else { return }
        guard let app = selectedApp, let window = selectedWindow else { return }
        lifecycleGeneration += 1
        let generation = lifecycleGeneration
        stoppingStartGeneration = nil
        isStopping = false
        phase = .starting
        events = []
        pendingGateReason = nil
        pendingApproval = nil
        actionError = nil
        showingPollError = false
        requiresBindingCleanup = false
        pendingCreateRecovery = nil
        stopPolling()
        runID = nil
        do {
            let capabilities = try await api.capabilities()
            guard generation == lifecycleGeneration else { return }
            if stoppingStartGeneration == generation {
                stoppingStartGeneration = nil
                isStopping = false
                phase = .idle
                return
            }
            guard capabilities.features.idempotentRunCreate else {
                phase = .failed(
                    message: "This local server cannot safely recover an interrupted task start. Update or restart Rapid before starting Computer Use."
                )
                return
            }
        } catch {
            guard generation == lifecycleGeneration else { return }
            if stoppingStartGeneration == generation {
                stoppingStartGeneration = nil
                isStopping = false
                phase = .idle
            } else {
                phase = .failed(
                    message: "Rapid could not verify safe task-start recovery: \(Self.describe(error))"
                )
            }
            return
        }
        let request = CUARunRequest(
            app: "pid:\(app.pid)",
            goal: goal.trimmingCharacters(in: .whitespacesAndNewlines),
            planner: plannerName,
            openURL: openURL.trimmingCharacters(in: .whitespacesAndNewlines),
            allowedDomain: allowedDomain.trimmingCharacters(in: .whitespacesAndNewlines),
            maxSteps: maxSteps,
            humanLogin: true,
            windowID: window.windowID,
            clientRequestID: UUID().uuidString.lowercased()
        )
        do {
            let createdRunID = try await api.create(request)
            guard generation == lifecycleGeneration else {
                // Defensive cleanup for a programmatic lifecycle replacement.
                // The UI cannot reach this branch: `.starting` disables Start,
                // and Stop records `stoppingStartGeneration` without advancing
                // the generation so the result is handled below. A newer owner
                // must not be overwritten with an error from this old request.
                try? await api.cancel(runID: createdRunID)
                return
            }
            if stoppingStartGeneration == generation {
                do {
                    try await api.cancel(runID: createdRunID)
                    guard generation == lifecycleGeneration else { return }
                    stoppingStartGeneration = nil
                    isStopping = false
                    actionError = nil
                    phase = .idle
                } catch {
                    guard generation == lifecycleGeneration else { return }
                    stoppingStartGeneration = nil
                    isStopping = false
                    runID = createdRunID
                    actionError = "Stop failed: \(Self.describe(error)) Try Stop again."
                    phase = .running
                    beginPolling(runID: createdRunID, generation: generation)
                }
                return
            }
            runID = createdRunID
        } catch {
            guard generation == lifecycleGeneration else { return }
            let stopWasRequested = stoppingStartGeneration == generation
            if case let CUAClientError.windowBinding(
                _, _, createdRunID, cancellationFailed
            ) = error {
                stoppingStartGeneration = nil
                isStopping = false
                selectedWindowID = nil
                if cancellationFailed {
                    runID = createdRunID
                    requiresBindingCleanup = true
                    actionError = Self.bindingCleanupWarning
                    phase = .running
                    beginPolling(runID: createdRunID, generation: generation)
                } else if stopWasRequested {
                    actionError = nil
                    phase = .idle
                } else {
                    phase = .failed(message: Self.describe(error))
                }
                return
            }
            if case let CUAClientError.requestBinding(
                _, _, createdRunID, cancellationFailed
            ) = error {
                stoppingStartGeneration = nil
                isStopping = false
                if cancellationFailed {
                    quarantine(runID: createdRunID, warning: Self.createRecoveryWarning)
                } else if stopWasRequested {
                    actionError = nil
                    phase = .idle
                } else {
                    phase = .failed(message: Self.describe(error))
                }
                return
            }
            if Self.isAmbiguousCreateError(error) {
                stoppingStartGeneration = nil
                await recoverAmbiguousCreate(request, generation: generation)
                return
            }
            if stoppingStartGeneration == generation {
                stoppingStartGeneration = nil
                isStopping = false
                actionError = nil
                phase = .idle
                return
            }
            let typedTargetFailure: Bool
            if case let CUAClientError.typedHTTP(_, code, message, _) = error {
                typedTargetFailure = Self.invalidatesSelectedTarget(
                    code: code, message: message
                )
            } else if case let CUAClientError.http(_, detail) = error {
                typedTargetFailure = Self.invalidatesSelectedTarget(
                    code: nil, message: detail
                )
            } else {
                typedTargetFailure = false
            }
            if typedTargetFailure {
                selectedWindowID = nil
                var message = Self.describe(error)
                if !message.localizedCaseInsensitiveContains("refresh") {
                    message += " Refresh the window list and choose it again."
                }
                phase = .failed(message: message)
            } else {
                phase = .failed(message: Self.describe(error))
            }
            return
        }
        phase = .running
        guard let runID else { return }
        beginPolling(runID: runID, generation: generation)
    }

    private func beginPolling(runID: String, generation: Int) {
        pollTask = Task { [weak self] in
            await self?.pollUntilTerminal(runID: runID, generation: generation)
        }
        _ = pollTask
    }

    func approve() async {
        guard let runID, canApprove, let gateID = pendingApproval?.gateID else { return }
        let generation = lifecycleGeneration
        do {
            try await api.approve(runID: runID, gateID: gateID)
            guard generation == lifecycleGeneration, self.runID == runID else { return }
            pendingGateReason = nil
            pendingApproval = nil
            actionError = nil
            showingPollError = false
            phase = .running
        } catch {
            guard generation == lifecycleGeneration, self.runID == runID else { return }
            // Keep the run controllable so the user can retry approval or
            // stop it. Treating a transport error as a terminal phase leaves
            // the server run active with no Stop button.
            actionError = "Approval failed: \(Self.describe(error))"
            showingPollError = false
        }
    }

    func cancel() async {
        guard phase.isBusy, !isStopping else { return }
        if let request = pendingCreateRecovery, runID == nil {
            isStopping = true
            await recoverAmbiguousCreate(request, generation: lifecycleGeneration)
            return
        }
        if phase == .starting, runID == nil {
            stoppingStartGeneration = lifecycleGeneration
            isStopping = true
            actionError = "Stop requested. Waiting for the server to finish creating the task."
            return
        }
        guard let runID else { return }
        let previousPhase = phase
        lifecycleGeneration += 1
        let generation = lifecycleGeneration
        isStopping = true
        stopPolling()
        do {
            try await api.cancel(runID: runID)
            guard generation == lifecycleGeneration, self.runID == runID else { return }
            // The server has accepted cancellation; return to idle while its
            // trace records the terminal event.
            stopPolling()
            self.runID = nil
            pendingGateReason = nil
            pendingApproval = nil
            actionError = nil
            showingPollError = false
            requiresBindingCleanup = false
            pendingCreateRecovery = nil
            isStopping = false
            phase = .idle
        } catch {
            guard generation == lifecycleGeneration, self.runID == runID else { return }
            // Preserve the Stop control. A failed request must not make an
            // active run appear cancelled locally.
            actionError = requiresBindingCleanup
                ? "\(Self.bindingCleanupWarning) Stop failed: \(Self.describe(error)) Try Stop again."
                : "Stop failed: \(Self.describe(error))"
            showingPollError = false
            isStopping = false
            phase = previousPhase
            beginPolling(runID: runID, generation: generation)
        }
    }

    private func recoverAmbiguousCreate(_ request: CUARunRequest, generation: Int) async {
        pendingCreateRecovery = request
        phase = .starting
        isStopping = true
        actionError = "Recovering the interrupted task start and stopping any accepted run…"

        var recovered: CUARunCreated?
        do {
            let recoveredRunID = try await api.create(request)
            recovered = CUARunCreated(
                runID: recoveredRunID,
                status: "running",
                windowID: request.windowID,
                clientRequestID: request.clientRequestID
            )
        } catch {
            if let cleanup = Self.bindingCleanup(from: error) {
                guard generation == lifecycleGeneration else { return }
                pendingCreateRecovery = nil
                if cleanup.cancellationFailed {
                    quarantine(runID: cleanup.runID, warning: Self.createRecoveryWarning)
                } else {
                    isStopping = false
                    actionError = nil
                    phase = .idle
                }
                return
            }
            do {
                recovered = try await api.run(clientRequestID: request.clientRequestID)
            } catch {
                guard generation == lifecycleGeneration else { return }
                isStopping = false
                actionError = Self.createRecoveryWarning
                return
            }
        }

        guard generation == lifecycleGeneration, let recovered else { return }
        // Lookup is keyed by the client request identity. If a broken or
        // mismatched server echoes different metadata, the returned run is
        // still the only concrete automation authority we can stop. Treat it
        // as untrusted and cancel it rather than discarding its run ID.
        do {
            try await api.cancel(runID: recovered.runID)
            guard generation == lifecycleGeneration else { return }
            pendingCreateRecovery = nil
            runID = nil
            isStopping = false
            requiresBindingCleanup = false
            actionError = nil
            phase = .idle
        } catch {
            guard generation == lifecycleGeneration else { return }
            pendingCreateRecovery = nil
            quarantine(runID: recovered.runID, warning: Self.createRecoveryWarning)
        }
    }

    private func quarantine(runID: String, warning: String) {
        self.runID = runID
        requiresBindingCleanup = true
        isStopping = false
        actionError = warning
        phase = .running
        beginPolling(runID: runID, generation: lifecycleGeneration)
    }

    private static func isAmbiguousCreateError(_ error: Error) -> Bool {
        switch error {
        case CUAClientError.windowBinding, CUAClientError.requestBinding:
            return false
        case let CUAClientError.typedHTTP(status, _, _, _),
             let CUAClientError.http(status, _):
            return status >= 500
        default:
            return true
        }
    }

    private static func bindingCleanup(
        from error: Error
    ) -> (runID: String, cancellationFailed: Bool)? {
        switch error {
        case let CUAClientError.windowBinding(_, _, runID, cancellationFailed),
             let CUAClientError.requestBinding(_, _, runID, cancellationFailed):
            return (runID, cancellationFailed)
        default:
            return nil
        }
    }

    func stopPolling() {
        pollTask?.cancel()
        pollTask = nil
    }

    /// Permanently detaches this model from its authenticated sidecar session.
    /// Incrementing the generation makes every in-flight create, approval,
    /// cancellation, discovery, and poll continuation ignore its late result.
    func invalidateSession() {
        lifecycleGeneration += 1
        targetDiscoveryGeneration += 1
        stopPolling()
        runID = nil
        stoppingStartGeneration = nil
        pendingCreateRecovery = nil
        isStopping = false
    }

    private func pollUntilTerminal(runID: String, generation: Int) async {
        var lastSeq = events.map(\.seq).max() ?? 0
        while !Task.isCancelled {
            do {
                let view = try await api.events(runID: runID, after: lastSeq)
                guard generation == lifecycleGeneration, self.runID == runID else { return }
                if showingPollError {
                    actionError = requiresBindingCleanup ? Self.bindingCleanupWarning : nil
                    showingPollError = false
                }
                if !view.events.isEmpty {
                    events.append(contentsOf: view.events)
                    lastSeq = view.events.map(\.seq).max() ?? lastSeq
                }
                if let gate = view.pendingGate {
                    pendingGateReason = gate.reason ?? "approval required"
                    pendingApproval = CUAPendingApproval(
                        gateID: gate.gateID,
                        app: view.app,
                        action: gate.action,
                        target: gate.target,
                        reason: gate.reason ?? "Approval is required before Rapid continues."
                    )
                    phase = .awaitingApproval
                } else {
                    // Older servers expose approval state only as events.
                    for event in view.events where event.kind == "gate" {
                        pendingGateReason = event.reason ?? "sign-in"
                        pendingApproval = CUAPendingApproval(
                            gateID: event.gateID,
                            app: event.app ?? view.app,
                            action: event.action,
                            target: event.target ?? event.targetLabel,
                            reason: event.reason
                                ?? "Approval is required before Rapid continues."
                        )
                        phase = .awaitingApproval
                    }
                    for event in view.events where
                        event.kind == "gate_detail" && phase == .awaitingApproval
                    {
                        pendingGateReason = event.reason ?? pendingGateReason
                        pendingApproval = CUAPendingApproval(
                            gateID: event.gateID ?? pendingApproval?.gateID,
                            app: event.app ?? pendingApproval?.app ?? view.app,
                            action: event.action ?? pendingApproval?.action,
                            target: event.target ?? event.targetLabel
                                ?? pendingApproval?.target,
                            reason: event.reason ?? pendingApproval?.reason
                                ?? "Approval is required before Rapid continues."
                        )
                    }
                }
                if view.pendingGate == nil {
                    for event in view.events where event.kind == "gate_resolved" {
                        if let resolvedID = event.gateID,
                           let currentID = pendingApproval?.gateID,
                           resolvedID != currentID
                        {
                            continue
                        }
                        pendingGateReason = nil
                        pendingApproval = nil
                        phase = .running
                    }
                }
                for event in view.events where event.isTerminal {
                    switch event.status {
                    case "completed":
                        phase = .finished(summary: event.finalSummary ?? "")
                    case let terminal where terminal != nil:
                        let message = Self.firstNonEmpty(
                                event.reason,
                                event.error,
                                event.finalSummary,
                                view.error,
                                fallback: "run ended: \(terminal ?? "unknown")"
                            )
                        if Self.invalidatesSelectedTarget(code: event.error, message: message) {
                            selectedWindowID = nil
                            phase = .failed(
                                message: "\(message) Refresh the window list and choose it again."
                            )
                        } else {
                            phase = .failed(message: message)
                        }
                    default:
                        break
                    }
                    self.runID = nil
                    return
                }
                if view.status == "completed" {
                    phase = .finished(summary: view.finalSummary)
                    self.runID = nil
                    return
                }
                if ["stopped", "stalled", "failed"].contains(view.status) {
                    let message = Self.firstNonEmpty(
                            view.error,
                            view.finalSummary,
                            fallback: "run ended: \(view.status)"
                        )
                    if Self.invalidatesSelectedTarget(code: view.error, message: message) {
                        selectedWindowID = nil
                        phase = .failed(
                            message: "\(message) Refresh the window list and choose it again."
                        )
                    } else {
                        phase = .failed(message: message)
                    }
                    self.runID = nil
                    return
                }
            } catch {
                if Task.isCancelled { return }
                guard generation == lifecycleGeneration, self.runID == runID else { return }
                if case let CUAClientError.http(code, detail) = error, code == 404 {
                    phase = .failed(
                        message: Self.firstNonEmpty(
                            detail, fallback: "The server no longer has this run."
                        )
                    )
                    self.runID = nil
                    return
                }
                // A transient polling failure does not stop the server run.
                // Keep Stop available and retry instead of presenting a
                // terminal local state while automation continues unseen.
                actionError = requiresBindingCleanup
                    ? Self.bindingCleanupWarning
                    : "Connection interrupted: \(Self.describe(error))"
                showingPollError = true
            }
            try? await Task.sleep(nanoseconds: pollIntervalNanos)
        }
    }

    private static func describe(_ error: Error) -> String {
        (error as? LocalizedError)?.errorDescription ?? String(describing: error)
    }

    private static func invalidatesSelectedTarget(code: String?, message: String) -> Bool {
        let targetCodes = [
            "window_not_found", "window_stale", "target_drift", "target_stale",
            "stale_observation",
        ]
        if let code, targetCodes.contains(code.lowercased()) { return true }
        let lower = message.lowercased()
        return targetCodes.contains(where: { lower.contains($0) })
            || lower.contains("selected window unavailable")
            || lower.contains("window was replaced")
            || lower.contains("window is no longer available")
    }

    private static func firstNonEmpty(
        _ values: String?..., fallback: String
    ) -> String {
        for value in values {
            if let value,
               !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            {
                return value
            }
        }
        return fallback
    }
}

/// Placeholder used before the server connection is available; starting with
/// it fails loudly instead of silently doing nothing.
private struct NullCUAAPI: CUAAPI {
    struct Unavailable: LocalizedError {
        var errorDescription: String {
            "The local server is not connected yet."
        }
    }

    func capabilities() async throws -> CUACapabilities { throw Unavailable() }

    func planners() async throws -> [CUAPlannerOption] {
        throw Unavailable()
    }

    func apps() async throws -> [CUAAppOption] { throw Unavailable() }

    func windows(app: String) async throws -> [CUAWindowOption] { throw Unavailable() }

    func addPlanner(_ request: CUAPlannerCreateRequest) async throws {
        throw Unavailable()
    }

    func deletePlanner(name: String) async throws {
        throw Unavailable()
    }

    func create(_ request: CUARunRequest) async throws -> String {
        throw Unavailable()
    }

    func run(clientRequestID: String) async throws -> CUARunCreated {
        throw Unavailable()
    }

    func permissions() async throws -> CUAPermissionStatus {
        throw Unavailable()
    }

    func events(runID: String, after: Int) async throws -> CUARunView {
        throw Unavailable()
    }

    func approve(runID: String, gateID: String) async throws {
        throw Unavailable()
    }

    func cancel(runID: String) async throws {
        throw Unavailable()
    }
}
