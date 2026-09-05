import Foundation
import Testing
@testable import Rapid

@Suite("Local workflow execution kernel")
struct LocalWorkflowExecutorTests {
    @Test("verified steps advance deterministically without asking for harmless actions")
    func verifiedStepsComplete() async throws {
        let stepA = step(id: "open", risk: .readOnly)
        let stepB = step(id: "select")
        let workflow = LocalWorkflow(title: "Lunch", steps: [stepA, stepB])
        let observer = ScriptedWorkflowObserver([
            observation(revision: "a"), observation(revision: "a"), observation(revision: "b"),
            observation(revision: "b"), observation(revision: "b"), observation(revision: "c"),
        ])
        let dependencies = Dependencies(observer: observer, verifications: [.satisfied, .satisfied])

        let run = await dependencies.executor.execute(workflow)

        #expect(run.status == .completed)
        #expect(run.nextStepIndex == 2)
        #expect(await dependencies.actuator.count == 2)
        #expect(await dependencies.approver.count == 0)
    }

    @Test("a changed window snapshot rejects the stale action and re-observes")
    func staleObservationRetriesWithoutActing() async throws {
        let workflow = LocalWorkflow(title: "Lunch", steps: [step(maxAttempts: 2)])
        let observer = ScriptedWorkflowObserver([
            observation(revision: "old"),
            observation(revision: "changed"),
            observation(revision: "changed"),
            observation(revision: "changed"),
            observation(revision: "done"),
        ])
        let dependencies = Dependencies(observer: observer, verifications: [.satisfied])

        let run = await dependencies.executor.execute(workflow)

        #expect(run.status == .completed)
        #expect(await dependencies.grounder.count == 2)
        #expect(await dependencies.actuator.count == 1)
        let events = await dependencies.ledger.events
        #expect(events.contains { $0.kind == .staleActionRejected && $0.code == .interactionStateChanged })
    }

    @Test("failed visual retries use one semantic fallback before pausing")
    func semanticFallbackRecovers() async throws {
        let workflow = LocalWorkflow(
            title: "Lunch",
            steps: [step(maxAttempts: 2, fallback: "student-profile")]
        )
        let observer = ScriptedWorkflowObserver([
            observation(revision: "a"), observation(revision: "a"), observation(revision: "b"),
            observation(revision: "b"), observation(revision: "b"), observation(revision: "c"),
            observation(revision: "c"), observation(revision: "c"), observation(revision: "done"),
        ])
        let dependencies = Dependencies(
            observer: observer,
            verifications: [
                .notSatisfied(code: .targetUnchanged),
                .notSatisfied(code: .targetUnchanged),
                .satisfied,
            ]
        )

        let run = await dependencies.executor.execute(workflow)

        #expect(run.status == .completed)
        #expect(await dependencies.actuator.count == 3)
        #expect(await dependencies.fallback.count == 1)
        let events = await dependencies.ledger.events
        #expect(events.contains { $0.kind == .semanticFallbackUsed })
        #expect(events.last?.kind == .runCompleted)
    }

    @Test("exhausted idempotent retries preserve that an action occurred")
    func exhaustedRetriesPreserveActionUncertainty() async throws {
        let workflow = LocalWorkflow(title: "Lunch", steps: [step(maxAttempts: 2)])
        let observer = ScriptedWorkflowObserver([
            observation(revision: "a"), observation(revision: "a"), observation(revision: "b"),
            observation(revision: "b"), observation(revision: "b"), observation(revision: "c"),
        ])
        let dependencies = Dependencies(
            observer: observer,
            verifications: [
                .notSatisfied(code: .targetUnchanged),
                .notSatisfied(code: .targetUnchanged),
            ]
        )

        let run = await dependencies.executor.execute(workflow)

        #expect(run.status == .paused(
            stepID: "choose",
            reason: .recoveryExhausted,
            actionMayHaveOccurred: true
        ))
    }

    @Test("approval is action-scoped and changed UI after approval forces a fresh proposal")
    func approvalRebindsBeforeActing() async throws {
        let workflow = LocalWorkflow(
            title: "Publish",
            steps: [step(risk: .externalCommunication, maxAttempts: 2)]
        )
        let observer = ScriptedWorkflowObserver([
            observation(revision: "draft"),
            observation(revision: "draft"),
            observation(revision: "edited-during-approval"),
            observation(revision: "edited-during-approval"),
            observation(revision: "edited-during-approval"),
            observation(revision: "edited-during-approval"),
            observation(revision: "published"),
        ])
        let dependencies = Dependencies(
            observer: observer,
            verifications: [.satisfied],
            approvals: [.approved, .approved]
        )

        let run = await dependencies.executor.execute(workflow)

        #expect(run.status == .completed)
        #expect(await dependencies.approver.count == 2)
        #expect(await dependencies.actuator.count == 1)
        let events = await dependencies.ledger.events
        #expect(events.contains { $0.code == .stateChangedDuringApproval })
    }

    @Test("a protected action denied by the user never reaches the actuator")
    func approvalDenialFailsClosed() async throws {
        let workflow = LocalWorkflow(
            title: "Checkout",
            steps: [step(risk: .financial)]
        )
        let observer = ScriptedWorkflowObserver([
            observation(revision: "review"), observation(revision: "review"),
        ])
        let dependencies = Dependencies(
            observer: observer,
            verifications: [],
            approvals: [.denied]
        )

        let run = await dependencies.executor.execute(workflow)

        #expect(run.status == .paused(
            stepID: "choose",
            reason: .approvalDenied,
            actionMayHaveOccurred: false
        ))
        #expect(run.nextStepIndex == 0)
        #expect(await dependencies.actuator.count == 0)
    }

    @Test("approval text flattens whitespace, exposes invisible controls, and is bounded")
    func approvalSummaryIsDisplaySafe() async throws {
        let unsafeTitle = "Publish \u{202E}now"
        let workflow = LocalWorkflow(
            title: "Publish",
            steps: [
                LocalWorkflowStep(
                    id: "choose",
                    title: unsafeTitle,
                    instruction: "Publish the draft",
                    successCriteria: "The draft is published",
                    risk: .externalCommunication
                ),
            ]
        )
        let observer = ScriptedWorkflowObserver([
            observation(revision: "draft"), observation(revision: "draft"),
        ])
        let rawSummary = "Publish\nnow \u{202E}e" + String(repeating: "\u{0301}", count: 300)
        let dependencies = Dependencies(
            observer: observer,
            verifications: [],
            approvals: [.denied],
            actionSummary: rawSummary
        )

        _ = await dependencies.executor.execute(workflow)
        let request = try #require(await dependencies.approver.requests.first)

        #expect(!request.actionSummary.contains("\n"))
        #expect(request.actionSummary.contains("\\u{202E}"))
        #expect(request.actionSummary.utf8.count <= 240)
        #expect(request.stepTitle.contains("\\u{202E}"))
    }

    @Test("an unsafe verifier result pauses without trying a different click")
    func unsafeVerificationStopsRecovery() async throws {
        let workflow = LocalWorkflow(title: "Lunch", steps: [step(maxAttempts: 3)])
        let observer = ScriptedWorkflowObserver([
            observation(revision: "a"), observation(revision: "a"), observation(revision: "unexpected"),
        ])
        let dependencies = Dependencies(
            observer: observer,
            verifications: [.unsafe(code: .unexpectedAccount)]
        )

        let run = await dependencies.executor.execute(workflow)

        #expect(run.status == .paused(
            stepID: "choose",
            reason: .unsafeState,
            actionMayHaveOccurred: true
        ))
        #expect(await dependencies.grounder.count == 1)
        #expect(await dependencies.actuator.count == 1)
    }

    @Test("a non-idempotent step never repeats an action that could have taken effect")
    func nonIdempotentActionDoesNotRetry() async throws {
        let workflow = LocalWorkflow(
            title: "Submit",
            steps: [step(risk: .externalCommunication, maxAttempts: 1)]
        )
        let observer = ScriptedWorkflowObserver([
            observation(revision: "draft"),
            observation(revision: "draft"),
            observation(revision: "draft"),
            observation(revision: "unknown"),
        ])
        let dependencies = Dependencies(
            observer: observer,
            verifications: [.notSatisfied(code: .targetUnchanged)],
            approvals: [.approved]
        )

        let run = await dependencies.executor.execute(workflow)

        #expect(run.status == .paused(
            stepID: "choose",
            reason: .recoveryExhausted,
            actionMayHaveOccurred: true
        ))
        #expect(await dependencies.grounder.count == 1)
        #expect(await dependencies.actuator.count == 1)

        let resumed = await dependencies.executor.execute(workflow, resuming: run)
        #expect(resumed.status == .paused(
            stepID: "invalid-run",
            reason: .unsafeState,
            actionMayHaveOccurred: true
        ))
        #expect(await dependencies.actuator.count == 1)
    }

    @Test("invalid normalized coordinates fail closed before input injection")
    func invalidActionFailsClosed() async throws {
        let workflow = LocalWorkflow(title: "Lunch", steps: [step()])
        let observer = ScriptedWorkflowObserver([observation(revision: "menu")])
        let dependencies = Dependencies(
            observer: observer,
            verifications: [],
            payload: .click(normalizedX: 1.5, normalizedY: 0.5)
        )

        let run = await dependencies.executor.execute(workflow)

        #expect(run.status == .paused(
            stepID: "choose",
            reason: .unsafeState,
            actionMayHaveOccurred: false
        ))
        #expect(await dependencies.actuator.count == 0)
        #expect((await dependencies.ledger.events).contains { $0.code == .invalidAction })
    }

    @Test("an invalid observation fails closed before grounding")
    func invalidObservationFailsClosed() async throws {
        let workflow = LocalWorkflow(title: "Lunch", steps: [step()])
        let invalid = WorkflowObservation(
            target: WorkflowInteractionTarget(
                bundleIdentifier: "com.example.portal",
                processIdentifier: 42,
                windowIdentifier: "main",
                windowFrame: WorkflowWindowFrame(x: 0, y: 0, width: 0, height: 800)
            ),
            contentRevision: "menu"
        )
        let dependencies = Dependencies(
            observer: ScriptedWorkflowObserver([invalid]),
            verifications: []
        )

        let run = await dependencies.executor.execute(workflow)

        #expect(run.status == .paused(
            stepID: "choose",
            reason: .unsafeState,
            actionMayHaveOccurred: false
        ))
        #expect(await dependencies.grounder.count == 0)
        #expect((await dependencies.ledger.events).contains { $0.code == .invalidObservation })
    }

    @Test("an interrupted in-flight run cannot replay a possibly completed action")
    func runningResumeFailsClosed() async throws {
        let workflow = LocalWorkflow(title: "Lunch", steps: [step()])
        let interrupted = LocalWorkflowRun(
            workflowID: workflow.id,
            status: .running
        )
        let dependencies = Dependencies(
            observer: ScriptedWorkflowObserver([]),
            verifications: []
        )

        let run = await dependencies.executor.execute(workflow, resuming: interrupted)

        #expect(run.status == .paused(
            stepID: "invalid-run",
            reason: .unsafeState,
            actionMayHaveOccurred: true
        ))
        #expect(await dependencies.actuator.count == 0)
    }

    @Test("a forged ready checkpoint cannot skip workflow steps")
    func advancedReadyResumeFailsClosed() async throws {
        let workflow = LocalWorkflow(
            title: "Lunch",
            steps: [step(id: "choose"), step(id: "checkout", risk: .financial)]
        )
        let dependencies = Dependencies(
            observer: ScriptedWorkflowObserver([]),
            verifications: []
        )

        for nextStepIndex in [1, workflow.steps.count] {
            let forged = LocalWorkflowRun(
                workflowID: workflow.id,
                nextStepIndex: nextStepIndex,
                status: .ready
            )
            let run = await dependencies.executor.execute(workflow, resuming: forged)

            #expect(run.status == .paused(
                stepID: "invalid-run",
                reason: .unsafeState,
                actionMayHaveOccurred: false
            ))
        }
        #expect(await dependencies.actuator.count == 0)
        #expect(await dependencies.approver.count == 0)
    }

    @Test("the executor rejects an overlapping claim before duplicate actuation")
    func overlappingRunIsRejected() async throws {
        let workflow = LocalWorkflow(title: "Lunch", steps: [step()])
        let initial = LocalWorkflowRun(workflowID: workflow.id)
        let observer = BlockingFirstWorkflowObserver([
            observation(revision: "menu"),
            observation(revision: "menu"),
            observation(revision: "selected"),
        ])
        let dependencies = Dependencies(observer: observer, verifications: [.satisfied])

        let first = Task {
            await dependencies.executor.execute(workflow, resuming: initial)
        }
        await observer.waitUntilBlocked()
        let overlapping = await dependencies.executor.execute(workflow, resuming: initial)
        await observer.release()
        let completed = await first.value

        #expect(overlapping.status == .paused(
            stepID: "executor-busy",
            reason: .unsafeState,
            actionMayHaveOccurred: true
        ))
        #expect(completed.status == .completed)
        #expect(await dependencies.actuator.count == 1)
    }

    @Test("cancellation becomes durable run state")
    func cancellationIsRecorded() async throws {
        let workflow = LocalWorkflow(title: "Lunch", steps: [step()])
        let observer = CancellingWorkflowObserver()
        let dependencies = Dependencies(observer: observer, verifications: [])

        let run = await dependencies.executor.execute(workflow)

        #expect(run.status == .cancelled(stepID: "choose", actionMayHaveOccurred: false))
        #expect(run.nextStepIndex == 0)
        #expect((await dependencies.ledger.events).last?.kind == .runCancelled)
    }

    @Test("cancellation while input is suspended cannot become completed")
    func cancellationDuringActuationIsConservative() async throws {
        let workflow = LocalWorkflow(title: "Lunch", steps: [step()])
        let observer = ScriptedWorkflowObserver([
            observation(revision: "menu"),
            observation(revision: "menu"),
        ])
        let grounder = ScriptedWorkflowGrounder(
            payload: .click(normalizedX: 0.2, normalizedY: 0.3),
            actionSummary: "Choose meal"
        )
        let actuator = BlockingWorkflowActuator()
        let ledger = RecordingWorkflowLedger()
        let executor = LocalWorkflowExecutor(
            observer: observer,
            grounder: grounder,
            actuator: actuator,
            verifier: ScriptedWorkflowVerifier([.satisfied]),
            fallbackResolver: ScriptedWorkflowFallback(),
            approver: ScriptedWorkflowApprover([]),
            ledger: ledger
        )

        let task = Task { await executor.execute(workflow) }
        await actuator.waitUntilBlocked()
        task.cancel()
        await actuator.release()
        let run = await task.value

        #expect(run.status == .cancelled(
            stepID: "choose",
            actionMayHaveOccurred: true
        ))
        #expect((await ledger.events).last?.kind == .runCancelled)
    }

    @Test("audit events have no field capable of persisting action text or workflow instructions")
    func auditLedgerRedactsPayloads() async throws {
        let secret = "4111-1111-1111-1111"
        let workflow = LocalWorkflow(
            title: "Private task",
            steps: [
                LocalWorkflowStep(
                    id: "choose",
                    title: "Enter saved value",
                    instruction: "Type \(secret)",
                    successCriteria: "The secret appears",
                    risk: .localChange
                ),
            ]
        )
        let observer = ScriptedWorkflowObserver([
            observation(revision: "a"), observation(revision: "a"), observation(revision: "b"),
        ])
        let dependencies = Dependencies(
            observer: observer,
            verifications: [.satisfied],
            payload: .typeText(secret)
        )

        _ = await dependencies.executor.execute(workflow)
        let encoded = try JSONEncoder().encode(await dependencies.ledger.events)
        let json = String(decoding: encoded, as: UTF8.self)

        #expect(!json.contains(secret))
        #expect(!json.contains("Type"))
        #expect(!json.contains("secret appears"))
    }

    private func step(
        id: String = "choose",
        risk: WorkflowActionRisk = .localChange,
        maxAttempts: Int = 2,
        fallback: String? = nil
    ) -> LocalWorkflowStep {
        LocalWorkflowStep(
            id: id,
            title: "Choose meal",
            instruction: "Choose the preferred meal",
            successCriteria: "The meal is selected",
            risk: risk,
            isIdempotent: maxAttempts > 1 || fallback != nil,
            maxGroundingAttempts: maxAttempts,
            semanticFallbackIdentifier: fallback
        )
    }

    private func observation(revision: String) -> WorkflowObservation {
        WorkflowObservation(
            target: WorkflowInteractionTarget(
                bundleIdentifier: "com.example.portal",
                processIdentifier: 42,
                windowIdentifier: "main",
                windowFrame: WorkflowWindowFrame(x: 0, y: 0, width: 1200, height: 800)
            ),
            contentRevision: revision
        )
    }
}

private actor ScriptedWorkflowObserver: LocalWorkflowObserving {
    private var observations: [WorkflowObservation]

    init(_ observations: [WorkflowObservation]) { self.observations = observations }

    func observe(for _: LocalWorkflowStep) async throws -> WorkflowObservation {
        guard !observations.isEmpty else { throw TestDependencyError.exhausted }
        return observations.removeFirst()
    }
}

private actor OneShotWorkflowGate {
    private var isOpen = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    func wait() async {
        if isOpen { return }
        await withCheckedContinuation { continuation in
            waiters.append(continuation)
        }
    }

    func open() {
        guard !isOpen else { return }
        isOpen = true
        let current = waiters
        waiters.removeAll()
        current.forEach { $0.resume() }
    }
}

private actor BlockingFirstWorkflowObserver: LocalWorkflowObserving {
    private var observations: [WorkflowObservation]
    private var callCount = 0
    private let blocked = OneShotWorkflowGate()
    private let released = OneShotWorkflowGate()

    init(_ observations: [WorkflowObservation]) { self.observations = observations }

    func observe(for _: LocalWorkflowStep) async throws -> WorkflowObservation {
        callCount += 1
        if callCount == 1 {
            await blocked.open()
            await released.wait()
        }
        guard !observations.isEmpty else { throw TestDependencyError.exhausted }
        return observations.removeFirst()
    }

    func waitUntilBlocked() async { await blocked.wait() }
    func release() async { await released.open() }
}

private actor CancellingWorkflowObserver: LocalWorkflowObserving {
    func observe(for _: LocalWorkflowStep) async throws -> WorkflowObservation {
        throw CancellationError()
    }
}

private actor ScriptedWorkflowGrounder: LocalWorkflowGrounding {
    private(set) var count = 0
    private let payload: WorkflowActionPayload
    private let actionSummary: String

    init(payload: WorkflowActionPayload, actionSummary: String) {
        self.payload = payload
        self.actionSummary = actionSummary
    }

    func ground(
        step _: LocalWorkflowStep,
        observation: WorkflowObservation
    ) async throws -> GroundedWorkflowAction {
        count += 1
        return GroundedWorkflowAction(
            observationID: observation.id,
            payload: payload,
            source: .visualGrounding,
            safeSummary: actionSummary,
            risk: .readOnly
        )
    }
}

private actor RecordingWorkflowActuator: LocalWorkflowActuating {
    private(set) var actions: [GroundedWorkflowAction] = []
    var count: Int { actions.count }

    func perform(
        _ action: GroundedWorkflowAction,
        against _: WorkflowObservation
    ) async throws {
        actions.append(action)
    }
}

private actor BlockingWorkflowActuator: LocalWorkflowActuating {
    private let blocked = OneShotWorkflowGate()
    private let released = OneShotWorkflowGate()

    func perform(
        _: GroundedWorkflowAction,
        against _: WorkflowObservation
    ) async throws {
        await blocked.open()
        await released.wait()
    }

    func waitUntilBlocked() async { await blocked.wait() }
    func release() async { await released.open() }
}

private actor ScriptedWorkflowVerifier: LocalWorkflowVerifying {
    private var results: [WorkflowVerification]

    init(_ results: [WorkflowVerification]) { self.results = results }

    func verify(
        step _: LocalWorkflowStep,
        before _: WorkflowObservation,
        after _: WorkflowObservation
    ) async throws -> WorkflowVerification {
        guard !results.isEmpty else { throw TestDependencyError.exhausted }
        return results.removeFirst()
    }
}

private actor ScriptedWorkflowFallback: LocalWorkflowFallbackResolving {
    private(set) var count = 0

    func fallbackAction(
        identifier _: String,
        step _: LocalWorkflowStep,
        observation: WorkflowObservation
    ) async throws -> GroundedWorkflowAction? {
        count += 1
        return GroundedWorkflowAction(
            observationID: observation.id,
            payload: .click(normalizedX: 0.5, normalizedY: 0.5),
            source: .semanticFallback,
            safeSummary: "Activate the named control",
            risk: .readOnly
        )
    }
}

private actor ScriptedWorkflowApprover: LocalWorkflowApproving {
    private var decisions: [WorkflowApprovalDecision]
    private(set) var requests: [WorkflowApprovalRequest] = []
    var count: Int { requests.count }

    init(_ decisions: [WorkflowApprovalDecision]) { self.decisions = decisions }

    func requestApproval(_ request: WorkflowApprovalRequest) async -> WorkflowApprovalDecision {
        requests.append(request)
        guard !decisions.isEmpty else { return .unavailable }
        return decisions.removeFirst()
    }
}

private actor RecordingWorkflowLedger: LocalWorkflowLedgerWriting {
    private(set) var events: [WorkflowLedgerEvent] = []

    func append(_ event: WorkflowLedgerEvent) async { events.append(event) }
}

private struct Dependencies {
    let grounder: ScriptedWorkflowGrounder
    let actuator: RecordingWorkflowActuator
    let fallback: ScriptedWorkflowFallback
    let approver: ScriptedWorkflowApprover
    let ledger: RecordingWorkflowLedger
    let executor: LocalWorkflowExecutor

    init(
        observer: any LocalWorkflowObserving,
        verifications: [WorkflowVerification],
        approvals: [WorkflowApprovalDecision] = [],
        payload: WorkflowActionPayload = .click(normalizedX: 0.2, normalizedY: 0.3),
        actionSummary: String = "Activate the selected control"
    ) {
        let grounder = ScriptedWorkflowGrounder(payload: payload, actionSummary: actionSummary)
        let actuator = RecordingWorkflowActuator()
        let fallback = ScriptedWorkflowFallback()
        let approver = ScriptedWorkflowApprover(approvals)
        let ledger = RecordingWorkflowLedger()
        self.grounder = grounder
        self.actuator = actuator
        self.fallback = fallback
        self.approver = approver
        self.ledger = ledger
        self.executor = LocalWorkflowExecutor(
            observer: observer,
            grounder: grounder,
            actuator: actuator,
            verifier: ScriptedWorkflowVerifier(verifications),
            fallbackResolver: fallback,
            approver: approver,
            ledger: ledger
        )
    }
}

private enum TestDependencyError: Error {
    case exhausted
}
