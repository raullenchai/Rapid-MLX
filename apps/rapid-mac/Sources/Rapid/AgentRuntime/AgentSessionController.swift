import Foundation
import Observation

struct AgentClientToolResult: Equatable, Sendable {
    let content: String
    let isError: Bool
    let executed: Bool
    /// True when the person pressed Don't allow in the approval sheet. The
    /// server drops client content for a tool that never ran, so this flag is
    /// how the model learns the step was refused rather than broken.
    let declined: Bool

    init(content: String, isError: Bool, executed: Bool, declined: Bool = false) {
        self.content = content
        self.isError = isError
        self.executed = executed
        self.declined = declined
    }
}

/// Exact UI context that owns a Personal Intelligence run.
///
/// Keeping the conversation alongside the live model identity gives SwiftUI
/// one transition to observe. A run is invalid as soon as either the owning
/// conversation or any component of the qualified model binding changes.
struct PersonalIntelligenceRunBinding: Equatable, Sendable {
    let conversationID: UUID
    let selectedAlias: String
    let serverModelID: String?
    let parser: String?
    let profile: String?
    let qualification: String?

    func invalidatesRun(boundTo previous: Self) -> Bool {
        self != previous
    }
}

/// UI-ready ownership for one live server-owned Agent run.
///
/// The controller observes and presents the Python runtime's state machine. The
/// server still owns planning, budgets, call identity, and approvals; Desktop
/// may execute an explicitly selected built-in read-only tool and return its
/// result for that exact server-issued call ID.
@MainActor
@Observable
final class AgentSessionController {
    enum Phase: Equatable {
        case idle
        case starting
        case running
        case awaitingApproval
        case completed
        case failed
        case cancelled
    }

    typealias TransportFactory = @MainActor (URL) -> any AgentRuntimeTransport
    typealias PollDelay = @Sendable () async throws -> Void
    typealias ClientToolExecutor = @MainActor @Sendable (
        AgentPendingAction
    ) async -> AgentClientToolResult

    private(set) var phase: Phase = .idle
    private(set) var run: AgentRunView?
    private(set) var events: [AgentEvent] = []
    private(set) var eventCursor = 0
    private(set) var errorMessage: String?

    var pendingApproval: AgentPendingAction? {
        guard phase == .awaitingApproval else { return nil }
        return run?.pendingAction
    }

    var isActive: Bool {
        switch phase {
        case .starting, .running, .awaitingApproval: true
        case .idle, .completed, .failed, .cancelled: false
        }
    }

    var progressText: String? {
        switch phase {
        case .idle: nil
        case .starting: "Starting agent…"
        case .running: "Working…"
        case .awaitingApproval: "Waiting for your approval"
        case .completed: nil
        case .failed: errorMessage ?? "The agent run failed."
        case .cancelled: "Stopped."
        }
    }

    private let transportFactory: TransportFactory
    private let pollDelay: PollDelay
    private var transport: (any AgentRuntimeTransport)?
    private var bearerToken: String?
    private var driverTask: Task<Void, Never>?
    private var recentlyCancelledDriverTask: Task<Void, Never>?
    private var activeRunID: String?
    private var activeClientToolExecutor: ClientToolExecutor?
    private var generation = 0
    private let lifetimeCleanup = LifetimeCleanup()

    private final class WeakOwner: @unchecked Sendable {
        weak var value: AgentSessionController?

        init(_ value: AgentSessionController) {
            self.value = value
        }
    }

    private struct DriverInputs {
        let eventCursor: Int
        let pollDelay: PollDelay
    }

    private final class LifetimeCleanup: @unchecked Sendable {
        private let lock = NSLock()
        private var task: Task<Void, Never>?
        private var remoteCancellation: (@Sendable () -> Void)?

        func setTask(_ task: Task<Void, Never>?) {
            lock.withLock { self.task = task }
        }

        func setRemoteCancellation(_ cancellation: @escaping @Sendable () -> Void) {
            lock.withLock { remoteCancellation = cancellation }
        }

        func clear() {
            lock.withLock {
                task = nil
                remoteCancellation = nil
            }
        }

        deinit {
            let cleanup = lock.withLock { (task, remoteCancellation) }
            cleanup.0?.cancel()
            cleanup.1?()
        }
    }

    init(
        transportFactory: @escaping TransportFactory = { AgentRuntimeClient(baseURL: $0) },
        pollDelay: @escaping PollDelay = {
            try await Task.sleep(nanoseconds: 250_000_000)
        }
    ) {
        self.transportFactory = transportFactory
        self.pollDelay = pollDelay
    }

    func start(
        goal: String,
        model: String?,
        expectedProfile: String? = nil,
        expectedQualification: String? = nil,
        toolNames: [String]? = nil,
        trustedInstructions: String? = nil,
        localContext: String? = nil,
        recentUserMessages: [String]? = nil,
        clientToolExecutor: ClientToolExecutor? = nil,
        baseURL: URL,
        bearerToken: String?
    ) {
        guard !isActive else { return }
        let trimmed = goal.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }

        generation &+= 1
        let expectedGeneration = generation
        let nextTransport = transportFactory(baseURL)
        transport = nextTransport
        self.bearerToken = bearerToken
        phase = .starting
        run = nil
        events = []
        eventCursor = 0
        errorMessage = nil
        activeRunID = nil
        activeClientToolExecutor = clientToolExecutor
        recentlyCancelledDriverTask = nil

        let owner = WeakOwner(self)
        let task = Task {
            do {
                // `create` is a commit-style request: URLSession cancellation
                // can race a server commit and discard the response containing
                // the only run ID we can use for cleanup. Keep that request in
                // an independent task, then reconcile its response against the
                // owning generation before observing the run.
                let createTask = Task { @MainActor in
                    try await nextTransport.create(
                        goal: trimmed,
                        model: model,
                        toolNames: toolNames,
                        trustedInstructions: trustedInstructions,
                        localContext: localContext,
                        recentUserMessages: recentUserMessages,
                        execution: clientToolExecutor == nil ? .server : .client,
                        bearerToken: bearerToken
                    )
                }
                let created = try await createTask.value
                if let model, created.model != model {
                    Self.requestRemoteCancellation(
                        transport: nextTransport,
                        runID: created.id,
                        bearerToken: bearerToken
                    )
                    throw AgentRuntimeClientError.modelBindingMismatch(
                        expected: model,
                        received: created.model
                    )
                }
                if let expectedProfile, created.profile != expectedProfile {
                    Self.requestRemoteCancellation(
                        transport: nextTransport,
                        runID: created.id,
                        bearerToken: bearerToken
                    )
                    throw AgentRuntimeClientError.harnessProfileMismatch(
                        expected: expectedProfile,
                        received: created.profile
                    )
                }
                if let expectedQualification,
                   created.personalIntelligenceQualification != expectedQualification {
                    Self.requestRemoteCancellation(
                        transport: nextTransport,
                        runID: created.id,
                        bearerToken: bearerToken
                    )
                    throw AgentRuntimeClientError.qualificationMismatch(
                        expected: expectedQualification,
                        received: created.personalIntelligenceQualification
                    )
                }
                guard owner.value?.generation == expectedGeneration else {
                    // The server may have committed the run even when the
                    // observing task was cancelled. Cancel from a fresh task:
                    // the current task's cancelled bit would otherwise abort
                    // URLSession before the cleanup request left the Mac.
                    Self.requestRemoteCancellation(
                        transport: nextTransport,
                        runID: created.id,
                        bearerToken: bearerToken
                    )
                    return
                }
                owner.value?.activeRunID = created.id
                owner.value?.lifetimeCleanup.setRemoteCancellation {
                    Task { @MainActor in
                        _ = try? await nextTransport.cancel(
                            runID: created.id,
                            bearerToken: bearerToken
                        )
                    }
                }
                await Self.drive(
                    created,
                    owner: owner,
                    transport: nextTransport,
                    bearerToken: bearerToken,
                    clientToolExecutor: clientToolExecutor,
                    generation: expectedGeneration
                )
            } catch is CancellationError {
                owner.value?.publishCancellationFailureIfCurrent(
                    generation: expectedGeneration
                )
                return
            } catch {
                owner.value?.publishFailure(error, generation: expectedGeneration)
            }
        }
        driverTask = task
        lifetimeCleanup.setTask(task)
    }

    /// Invalidate a run whose selected model or owning conversation changed.
    /// This is intentionally the same remote cancellation contract as Stop;
    /// the semantic entry point makes context-transition behavior directly
    /// testable without relying on SwiftUI source inspection.
    func bindingDidChange() {
        cancel()
    }

    func resolvePendingApproval(approved: Bool) {
        guard let transport,
              let run,
              let pending = pendingApproval,
              driverTask == nil else { return }
        let expectedGeneration = generation
        let approvalBearer = bearerToken
        let owner = WeakOwner(self)
        phase = .running
        let task = Task {
            do {
                let resolved = try await transport.resolveApproval(
                    runID: run.id,
                    callID: pending.callID,
                    approved: approved,
                    bearerToken: approvalBearer
                )
                guard owner.value?.generation == expectedGeneration else {
                    Self.requestRemoteCancellation(
                        transport: transport,
                        runID: run.id,
                        bearerToken: approvalBearer
                    )
                    return
                }
                await Self.drive(
                    resolved,
                    owner: owner,
                    transport: transport,
                    bearerToken: approvalBearer,
                    clientToolExecutor: activeClientToolExecutor,
                    generation: expectedGeneration
                )
            } catch is CancellationError {
                if owner.value?.generation == expectedGeneration {
                    Self.requestRemoteCancellation(
                        transport: transport,
                        runID: run.id,
                        bearerToken: approvalBearer
                    )
                    owner.value?.publishCancellationFailureIfCurrent(
                        generation: expectedGeneration
                    )
                }
                return
            } catch {
                Self.requestRemoteCancellation(
                    transport: transport,
                    runID: run.id,
                    bearerToken: approvalBearer
                )
                owner.value?.publishFailure(error, generation: expectedGeneration)
            }
        }
        driverTask = task
        lifetimeCleanup.setTask(task)
    }

    func cancel() {
        guard isActive else { return }
        let cancelledTransport = transport
        let cancelledRunID = activeRunID ?? run?.id
        let cancelledBearer = bearerToken
        generation &+= 1
        let cancelledDriverTask = driverTask
        cancelledDriverTask?.cancel()
        recentlyCancelledDriverTask = cancelledDriverTask
        driverTask = nil
        lifetimeCleanup.clear()
        phase = .cancelled
        errorMessage = nil
        activeRunID = nil
        activeClientToolExecutor = nil
        if let cancelledTransport, let cancelledRunID {
            Self.requestRemoteCancellation(
                transport: cancelledTransport,
                runID: cancelledRunID,
                bearerToken: cancelledBearer
            )
        }
    }

    func reset() {
        guard !isActive else { return }
        phase = .idle
        run = nil
        events = []
        eventCursor = 0
        errorMessage = nil
        transport = nil
        bearerToken = nil
        activeRunID = nil
        activeClientToolExecutor = nil
        recentlyCancelledDriverTask = nil
        lifetimeCleanup.clear()
    }

    func _testingWaitForDriver() async {
        await (driverTask ?? recentlyCancelledDriverTask)?.value
    }

    private static func drive(
        _ initialRun: AgentRunView,
        owner: WeakOwner,
        transport: any AgentRuntimeTransport,
        bearerToken: String?,
        clientToolExecutor: ClientToolExecutor?,
        generation expectedGeneration: Int
    ) async {
        var latest = initialRun
        while !Task.isCancelled {
            guard let inputs = owner.value?.driverInputs(generation: expectedGeneration) else {
                return
            }
            switch latest.status {
            case .completed, .failed, .cancelled, .awaitingApproval:
                // `get` and `events` are separate requests. Drain once more
                // after observing a boundary state so events committed between
                // the preceding event page and this terminal/approval view are
                // not silently lost from Desktop's progress history.
                do {
                    let finalPage = try await transport.events(
                        runID: latest.id,
                        after: inputs.eventCursor,
                        bearerToken: bearerToken
                    )
                    guard owner.value?.accept(
                        finalPage,
                        generation: expectedGeneration
                    ) == true else { return }
                } catch {
                    // The authoritative run view already reached a useful
                    // boundary. Event history is diagnostic presentation, so
                    // a failed final drain must not erase a completed answer
                    // or prevent the user from seeing an approval.
                }
                owner.value?.completeBoundary(
                    latest,
                    generation: expectedGeneration
                )
                return
            case .awaitingToolResult:
                owner.value?.publishRunning(latest, generation: expectedGeneration)
                guard let pending = latest.pendingAction,
                      let clientToolExecutor else {
                    requestRemoteCancellation(
                        transport: transport,
                        runID: latest.id,
                        bearerToken: bearerToken
                    )
                    owner.value?.publishFailure(
                        AgentRuntimeClientError.invalidResponse,
                        generation: expectedGeneration
                    )
                    return
                }
                let result = await clientToolExecutor(pending)
                guard !Task.isCancelled,
                      owner.value?.generation == expectedGeneration else {
                    requestRemoteCancellation(
                        transport: transport,
                        runID: latest.id,
                        bearerToken: bearerToken
                    )
                    return
                }
                do {
                    latest = try await transport.submitToolResult(
                        runID: latest.id,
                        callID: pending.callID,
                        content: result.content,
                        isError: result.isError,
                        executed: result.executed,
                        declined: result.declined,
                        bearerToken: bearerToken
                    )
                } catch is CancellationError {
                    if owner.value?.generation == expectedGeneration {
                        requestRemoteCancellation(
                            transport: transport,
                            runID: latest.id,
                            bearerToken: bearerToken
                        )
                        owner.value?.publishCancellationFailureIfCurrent(
                            generation: expectedGeneration
                        )
                    }
                    return
                } catch {
                    requestRemoteCancellation(
                        transport: transport,
                        runID: latest.id,
                        bearerToken: bearerToken
                    )
                    owner.value?.publishFailure(error, generation: expectedGeneration)
                    return
                }
                continue
            case .ready, .awaitingModel:
                owner.value?.publishRunning(latest, generation: expectedGeneration)
            }

            do {
                try await inputs.pollDelay()
                guard let cursor = owner.value?.cursor(generation: expectedGeneration) else {
                    return
                }
                let page = try await transport.events(
                    runID: latest.id,
                    after: cursor,
                    bearerToken: bearerToken
                )
                guard owner.value?.accept(page, generation: expectedGeneration) == true else {
                    return
                }
                latest = try await transport.get(
                    runID: latest.id,
                    bearerToken: bearerToken
                )
            } catch is CancellationError {
                if owner.value?.generation == expectedGeneration {
                    requestRemoteCancellation(
                        transport: transport,
                        runID: latest.id,
                        bearerToken: bearerToken
                    )
                    owner.value?.publishCancellationFailureIfCurrent(
                        generation: expectedGeneration
                    )
                }
                return
            } catch {
                requestRemoteCancellation(
                    transport: transport,
                    runID: latest.id,
                    bearerToken: bearerToken
                )
                owner.value?.publishFailure(error, generation: expectedGeneration)
                return
            }
        }
    }

    private func driverInputs(generation expectedGeneration: Int) -> DriverInputs? {
        guard generation == expectedGeneration else { return nil }
        return DriverInputs(eventCursor: eventCursor, pollDelay: pollDelay)
    }

    private func cursor(generation expectedGeneration: Int) -> Int? {
        generation == expectedGeneration ? eventCursor : nil
    }

    private func accept(_ page: AgentEventsView, generation expectedGeneration: Int) -> Bool {
        guard generation == expectedGeneration else { return false }
        append(page)
        return true
    }

    private func publishRunning(_ nextRun: AgentRunView, generation expectedGeneration: Int) {
        guard generation == expectedGeneration else { return }
        publish(nextRun)
    }

    private func completeBoundary(
        _ nextRun: AgentRunView,
        generation expectedGeneration: Int
    ) {
        guard generation == expectedGeneration else { return }
        driverTask = nil
        publish(nextRun)
        if nextRun.status == .awaitingApproval {
            lifetimeCleanup.setTask(nil)
        } else {
            activeRunID = nil
            activeClientToolExecutor = nil
            lifetimeCleanup.clear()
        }
    }

    private static func requestRemoteCancellation(
        transport: any AgentRuntimeTransport,
        runID: String,
        bearerToken: String?
    ) {
        Task { @MainActor in
            _ = try? await transport.cancel(runID: runID, bearerToken: bearerToken)
        }
    }

    private func publish(_ nextRun: AgentRunView) {
        run = nextRun
        switch nextRun.status {
        case .completed: phase = .completed
        case .failed:
            phase = .failed
            errorMessage = AgentFailurePresentation.message(for: nextRun.failureCode)
        case .cancelled: phase = .cancelled
        case .awaitingApproval: phase = .awaitingApproval
        case .ready, .awaitingModel, .awaitingToolResult: phase = .running
        }
    }

    private func append(_ page: AgentEventsView) {
        let unseen = page.events.filter { $0.sequence > eventCursor }
        events.append(contentsOf: unseen)
        if events.count > 128 {
            events.removeFirst(events.count - 128)
        }
        let highestEvent = unseen.map(\.sequence).max() ?? eventCursor
        eventCursor = max(eventCursor, page.nextAfter, highestEvent)
    }

    private func publishFailure(_ error: Error, generation expectedGeneration: Int) {
        guard generation == expectedGeneration else { return }
        phase = .failed
        errorMessage = (error as? LocalizedError)?.errorDescription
            ?? "The Agent Runtime request failed."
        driverTask = nil
        activeRunID = nil
        activeClientToolExecutor = nil
        lifetimeCleanup.clear()
    }

    private func publishCancellationFailureIfCurrent(generation expectedGeneration: Int) {
        guard generation == expectedGeneration else { return }
        publishFailure(CancellationError(), generation: expectedGeneration)
    }
}
