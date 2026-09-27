import Foundation

/// Transport contract for the agent-task panel, so the view model can be
/// unit-tested without a live server.
protocol CUAAPI: Sendable {
    func planners() async throws -> [CUAPlannerOption]
    func create(_ request: CUARunRequest) async throws -> String
    func events(runID: String, after: Int) async throws -> CUARunView
    func approve(runID: String) async throws
    func cancel(runID: String) async throws
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

/// Drives one agent task from the GUI: create on the app-owned server, poll
/// numbered events, surface gate approvals, and finish with the summary.
@MainActor
final class CUAViewModel: ObservableObject {
    @Published var goal = ""
    @Published var appName = "Google Chrome"
    @Published var plannerName = "cloud-glm"
    @Published var openURL = ""
    @Published var allowedDomain = ""
    @Published var maxSteps = 12
    @Published var phase: CUAPhase = .idle
    @Published var events: [CUAEvent] = []
    @Published var plannerOptions: [CUAPlannerOption] = []
    @Published var pendingGateReason: String?
    @Published var actionError: String?

    private let api: CUAAPI
    private var runID: String?
    private var pollTask: Task<Void, Never>?
    private var showingPollError = false
    private let pollIntervalNanos: UInt64

    init(api: CUAAPI?, pollIntervalNanos: UInt64 = 700_000_000) {
        self.api = api ?? NullCUAAPI()
        self.pollIntervalNanos = pollIntervalNanos
    }

    var canStart: Bool {
        !goal.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && !phase.isBusy
            && !appName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    func loadPlanners() async {
        plannerOptions = (try? await api.planners()) ?? []
        if plannerOptions.isEmpty {
            plannerOptions = []
        }
    }

    func start() async {
        guard canStart else { return }
        phase = .starting
        events = []
        pendingGateReason = nil
        actionError = nil
        showingPollError = false
        stopPolling()
        runID = nil
        let request = CUARunRequest(
            app: appName.trimmingCharacters(in: .whitespacesAndNewlines),
            goal: goal.trimmingCharacters(in: .whitespacesAndNewlines),
            planner: plannerName,
            openURL: openURL.trimmingCharacters(in: .whitespacesAndNewlines),
            allowedDomain: allowedDomain.trimmingCharacters(in: .whitespacesAndNewlines),
            maxSteps: maxSteps,
            humanLogin: true
        )
        do {
            runID = try await api.create(request)
        } catch {
            phase = .failed(message: Self.describe(error))
            return
        }
        phase = .running
        pollTask = Task { [weak self] in
            await self?.pollUntilTerminal()
        }
        _ = pollTask
    }

    func approve() async {
        guard let runID, phase == .awaitingApproval else { return }
        do {
            try await api.approve(runID: runID)
            pendingGateReason = nil
            actionError = nil
            showingPollError = false
            phase = .running
        } catch {
            // Keep the run controllable so the user can retry approval or
            // stop it. Treating a transport error as a terminal phase leaves
            // the server run active with no Stop button.
            actionError = "Approval failed: \(Self.describe(error))"
            showingPollError = false
        }
    }

    func cancel() async {
        guard let runID, phase.isBusy else { return }
        do {
            try await api.cancel(runID: runID)
            // The server has accepted cancellation; return to idle while its
            // trace records the terminal event.
            stopPolling()
            self.runID = nil
            pendingGateReason = nil
            actionError = nil
            showingPollError = false
            phase = .idle
        } catch {
            // Preserve the Stop control. A failed request must not make an
            // active run appear cancelled locally.
            actionError = "Stop failed: \(Self.describe(error))"
            showingPollError = false
        }
    }

    func stopPolling() {
        pollTask?.cancel()
        pollTask = nil
    }

    private func pollUntilTerminal() async {
        guard let runID else { return }
        var lastSeq = 0
        while !Task.isCancelled {
            do {
                let view = try await api.events(runID: runID, after: lastSeq)
                if showingPollError {
                    actionError = nil
                    showingPollError = false
                }
                if !view.events.isEmpty {
                    events.append(contentsOf: view.events)
                    lastSeq = view.events.map(\.seq).max() ?? lastSeq
                }
                for event in view.events where event.kind == "gate" {
                    pendingGateReason = event.reason ?? "sign-in"
                    phase = .awaitingApproval
                }
                for event in view.events where event.kind == "gate_resolved" {
                    pendingGateReason = nil
                }
                for event in view.events where event.isTerminal {
                    switch event.status {
                    case "completed":
                        phase = .finished(summary: event.finalSummary ?? "")
                    case let terminal where terminal != nil:
                        phase = .failed(
                            message: Self.firstNonEmpty(
                                event.reason,
                                event.error,
                                event.finalSummary,
                                view.error,
                                fallback: "run ended: \(terminal ?? "unknown")"
                            )
                        )
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
                    phase = .failed(
                        message: Self.firstNonEmpty(
                            view.error,
                            view.finalSummary,
                            fallback: "run ended: \(view.status)"
                        )
                    )
                    self.runID = nil
                    return
                }
            } catch {
                if Task.isCancelled { return }
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
                actionError = "Connection interrupted: \(Self.describe(error))"
                showingPollError = true
            }
            try? await Task.sleep(nanoseconds: pollIntervalNanos)
        }
    }

    private static func describe(_ error: Error) -> String {
        (error as? LocalizedError)?.errorDescription ?? String(describing: error)
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

    func planners() async throws -> [CUAPlannerOption] {
        throw Unavailable()
    }

    func create(_ request: CUARunRequest) async throws -> String {
        throw Unavailable()
    }

    func events(runID: String, after: Int) async throws -> CUARunView {
        throw Unavailable()
    }

    func approve(runID: String) async throws {
        throw Unavailable()
    }

    func cancel(runID: String) async throws {
        throw Unavailable()
    }
}
