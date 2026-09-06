import Foundation

protocol LocalWorkflowObserving: Sendable {
    func observe(for step: LocalWorkflowStep) async throws -> WorkflowObservation
}

protocol LocalWorkflowGrounding: Sendable {
    func ground(
        step: LocalWorkflowStep,
        observation: WorkflowObservation
    ) async throws -> GroundedWorkflowAction
}

protocol LocalWorkflowActuating: Sendable {
    func perform(
        _ action: GroundedWorkflowAction,
        against currentObservation: WorkflowObservation
    ) async throws
}

protocol LocalWorkflowVerifying: Sendable {
    func verify(
        step: LocalWorkflowStep,
        before: WorkflowObservation,
        after: WorkflowObservation
    ) async throws -> WorkflowVerification
}

protocol LocalWorkflowFallbackResolving: Sendable {
    func fallbackAction(
        identifier: String,
        step: LocalWorkflowStep,
        observation: WorkflowObservation
    ) async throws -> GroundedWorkflowAction?
}

protocol LocalWorkflowApproving: Sendable {
    /// The adapter owns its UI continuation and must enforce this deadline
    /// without leaving an orphan waiter. It must also return promptly when its
    /// task is cancelled.
    func requestApproval(
        _ request: WorkflowApprovalRequest,
        timeoutNanoseconds: UInt64
    ) async -> WorkflowApprovalDecision
}

protocol LocalWorkflowLedgerWriting: Sendable {
    func append(_ event: WorkflowLedgerEvent) async throws
}

/// Deterministic orchestration for a learned local workflow.
///
/// The grounder proposes one action; it never owns progress, retries, approval,
/// or completion. Every action is rebound to a fresh observation immediately
/// before execution, and protected actions are rebound once more after the user
/// answers so a slow approval cannot authorize a click against changed UI.
actor LocalWorkflowExecutor {
    private let observer: any LocalWorkflowObserving
    private let grounder: any LocalWorkflowGrounding
    private let actuator: any LocalWorkflowActuating
    private let verifier: any LocalWorkflowVerifying
    private let fallbackResolver: any LocalWorkflowFallbackResolving
    private let approver: any LocalWorkflowApproving
    private let ledger: any LocalWorkflowLedgerWriting
    private let approvalTimeoutNanoseconds: UInt64
    private var activeRunID: UUID?

    init(
        observer: any LocalWorkflowObserving,
        grounder: any LocalWorkflowGrounding,
        actuator: any LocalWorkflowActuating,
        verifier: any LocalWorkflowVerifying,
        fallbackResolver: any LocalWorkflowFallbackResolving,
        approver: any LocalWorkflowApproving,
        ledger: any LocalWorkflowLedgerWriting,
        approvalTimeoutNanoseconds: UInt64 = 300_000_000_000
    ) {
        self.observer = observer
        self.grounder = grounder
        self.actuator = actuator
        self.verifier = verifier
        self.fallbackResolver = fallbackResolver
        self.approver = approver
        self.ledger = ledger
        self.approvalTimeoutNanoseconds = approvalTimeoutNanoseconds
    }

    func execute(_ workflow: LocalWorkflow, resuming run: LocalWorkflowRun? = nil) async -> LocalWorkflowRun {
        var run = run ?? LocalWorkflowRun(workflowID: workflow.id)
        guard activeRunID == nil else {
            return pausedWithoutRecording(
                run,
                stepID: "executor-busy",
                reason: .unsafeState,
                actionMayHaveOccurred: true
            )
        }
        activeRunID = run.id
        defer { activeRunID = nil }

        guard run.workflowID == workflow.id,
              run.nextStepIndex >= 0,
              run.nextStepIndex <= workflow.steps.count,
              run.status != .ready || run.nextStepIndex == 0,
              Set(workflow.steps.map(\.id)).count == workflow.steps.count,
              workflow.steps.allSatisfy({ !$0.id.isEmpty })
        else {
            return await pause(
                run,
                stepID: "invalid-run",
                reason: .unsafeState,
                code: .invalidResumeState,
                actionMayHaveOccurred: run.status.actionMayHaveOccurred
            )
        }

        if run.status == .completed, run.nextStepIndex == workflow.steps.count {
            return run
        }

        if run.nextStepIndex == workflow.steps.count {
            guard run.status.permitsExecution else {
                return await pause(
                    run,
                    stepID: "invalid-run",
                    reason: .unsafeState,
                    code: .invalidResumeState,
                    actionMayHaveOccurred: run.status.actionMayHaveOccurred
                )
            }
            run.status = .completed
            do {
                try await record(run, stepID: nil, kind: .runCompleted)
            } catch {
                return pausedWithoutRecording(
                    run,
                    stepID: "ledger",
                    reason: .dependencyFailure,
                    actionMayHaveOccurred: false
                )
            }
            return run
        }

        guard run.status.permitsExecution else {
            return await pause(
                run,
                stepID: "invalid-run",
                reason: .unsafeState,
                code: .invalidResumeState,
                actionMayHaveOccurred: run.status.actionMayHaveOccurred
            )
        }

        run.status = .running
        do {
            try await record(run, stepID: nil, kind: .runStarted)
        } catch {
            return pausedWithoutRecording(
                run,
                stepID: "ledger",
                reason: .dependencyFailure,
                actionMayHaveOccurred: false
            )
        }

        while run.nextStepIndex < workflow.steps.count {
            if Task.isCancelled {
                return await cancel(
                    run,
                    stepID: nil,
                    actionMayHaveOccurred: run.nextStepIndex > 0
                )
            }
            let step = workflow.steps[run.nextStepIndex]
            let result = await executeStep(step, workflow: workflow, run: run)
            switch result {
            case .completed:
                run.nextStepIndex += 1
                if Task.isCancelled {
                    return await cancel(
                        run,
                        stepID: step.id,
                        actionMayHaveOccurred: true
                    )
                }
                run.status = .running
            case .paused(let reason, let code, let actionMayHaveOccurred):
                return await pause(
                    run,
                    stepID: step.id,
                    reason: reason,
                    code: code,
                    actionMayHaveOccurred: actionMayHaveOccurred
                )
            case .cancelled(let actionMayHaveOccurred):
                return await cancel(
                    run,
                    stepID: step.id,
                    actionMayHaveOccurred: actionMayHaveOccurred
                )
            }
        }

        if Task.isCancelled {
            return await cancel(
                run,
                stepID: nil,
                actionMayHaveOccurred: run.nextStepIndex > 0
            )
        }
        run.status = .completed
        do {
            try await record(
                run,
                stepID: nil,
                kind: .runCompleted,
                actionMayHaveOccurredOnFailure: run.nextStepIndex > 0
            )
        } catch {
            return pausedWithoutRecording(
                run,
                stepID: "ledger",
                reason: .dependencyFailure,
                actionMayHaveOccurred: run.nextStepIndex > 0
            )
        }
        return run
    }

    private enum StepResult {
        case completed
        case paused(WorkflowPauseReason, code: WorkflowLedgerCode, actionMayHaveOccurred: Bool)
        case cancelled(actionMayHaveOccurred: Bool)
    }

    private func executeStep(
        _ step: LocalWorkflowStep,
        workflow: LocalWorkflow,
        run: LocalWorkflowRun
    ) async -> StepResult {
        var priorActionMayHaveOccurred = run.nextStepIndex > 0
        do {
            // Decode is allowed to bypass ``LocalWorkflowStep``'s initializer,
            // so clamp again at the trust boundary before constructing a range.
            let attemptLimit = min(max(1, step.maxGroundingAttempts), 3)
            for attempt in 1 ... attemptLimit {
                if Task.isCancelled {
                    return .cancelled(actionMayHaveOccurred: priorActionMayHaveOccurred)
                }
                let observed = try await validObservation(for: step)
                let proposed = try await grounder.ground(step: step, observation: observed)
                let action = GroundedWorkflowAction(
                    observationID: proposed.observationID,
                    payload: proposed.payload,
                    source: .visualGrounding,
                    safeSummary: proposed.safeSummary,
                    risk: proposed.risk
                )
                try await record(
                    run,
                    stepID: step.id,
                    kind: .actionGrounded,
                    attempt: attempt,
                    source: action.source
                )
                let outcome = try await performIfCurrent(
                    action,
                    step: step,
                    workflow: workflow,
                    run: run,
                    observed: observed,
                    attempt: attempt
                )
                switch outcome {
                case .verified:
                    return .completed
                case .retry(let actionWasPerformed):
                    priorActionMayHaveOccurred = priorActionMayHaveOccurred || actionWasPerformed
                    if actionWasPerformed && !step.isIdempotent {
                        return .paused(
                            .recoveryExhausted,
                            code: .visualRetriesExhausted,
                            actionMayHaveOccurred: true
                        )
                    }
                    continue
                case .paused(let reason, let code, let actionMayHaveOccurred):
                    return .paused(
                        reason,
                        code: code,
                        actionMayHaveOccurred: priorActionMayHaveOccurred || actionMayHaveOccurred
                    )
                }
            }

            guard let fallbackIdentifier = step.semanticFallbackIdentifier else {
                return .paused(
                    .recoveryExhausted,
                    code: .visualRetriesExhausted,
                    actionMayHaveOccurred: priorActionMayHaveOccurred
                )
            }
            guard step.isIdempotent else {
                return .paused(
                    .recoveryExhausted,
                    code: .visualRetriesExhausted,
                    actionMayHaveOccurred: priorActionMayHaveOccurred
                )
            }
            let observed = try await validObservation(for: step)
            guard var fallback = try await fallbackResolver.fallbackAction(
                identifier: fallbackIdentifier,
                step: step,
                observation: observed
            ) else {
                return .paused(
                    .recoveryExhausted,
                    code: .fallbackUnavailable,
                    actionMayHaveOccurred: priorActionMayHaveOccurred
                )
            }
            fallback = GroundedWorkflowAction(
                observationID: fallback.observationID,
                payload: fallback.payload,
                source: .semanticFallback,
                safeSummary: fallback.safeSummary,
                risk: fallback.risk
            )
            try await record(
                run,
                stepID: step.id,
                kind: .semanticFallbackUsed,
                attempt: attemptLimit + 1,
                source: .semanticFallback
            )
            let outcome = try await performIfCurrent(
                fallback,
                step: step,
                workflow: workflow,
                run: run,
                observed: observed,
                attempt: attemptLimit + 1
            )
            switch outcome {
            case .verified:
                return .completed
            case .retry(let actionWasPerformed):
                return .paused(
                    .recoveryExhausted,
                    code: .fallbackNotVerified,
                    actionMayHaveOccurred: priorActionMayHaveOccurred || actionWasPerformed
                )
            case .paused(let reason, let code, let actionMayHaveOccurred):
                return .paused(
                    reason,
                    code: code,
                    actionMayHaveOccurred: priorActionMayHaveOccurred || actionMayHaveOccurred
                )
            }
        } catch WorkflowKernelError.cancelled(let actionMayHaveOccurred) {
            return .cancelled(
                actionMayHaveOccurred: priorActionMayHaveOccurred || actionMayHaveOccurred
            )
        } catch is CancellationError {
            return .cancelled(actionMayHaveOccurred: priorActionMayHaveOccurred)
        } catch WorkflowKernelError.invalidObservation(let actionMayHaveOccurred) {
            return .paused(
                .unsafeState,
                code: .invalidObservation,
                actionMayHaveOccurred: priorActionMayHaveOccurred || actionMayHaveOccurred
            )
        } catch WorkflowKernelError.dependencyFailure(let actionMayHaveOccurred) {
            return .paused(
                .dependencyFailure,
                code: .dependencyFailure,
                actionMayHaveOccurred: priorActionMayHaveOccurred || actionMayHaveOccurred
            )
        } catch {
            return .paused(
                .dependencyFailure,
                code: .dependencyFailure,
                actionMayHaveOccurred: priorActionMayHaveOccurred
            )
        }
    }

    private enum AttemptResult {
        case verified
        case retry(actionWasPerformed: Bool)
        case paused(WorkflowPauseReason, code: WorkflowLedgerCode, actionMayHaveOccurred: Bool)
    }

    private func performIfCurrent(
        _ action: GroundedWorkflowAction,
        step: LocalWorkflowStep,
        workflow: LocalWorkflow,
        run: LocalWorkflowRun,
        observed: WorkflowObservation,
        attempt: Int
    ) async throws -> AttemptResult {
        guard action.payload.isStructurallyValid else {
            try await record(
                run,
                stepID: step.id,
                kind: .staleActionRejected,
                attempt: attempt,
                source: action.source,
                code: .invalidAction
            )
            return .paused(
                .unsafeState,
                code: .invalidAction,
                actionMayHaveOccurred: false
            )
        }
        guard action.observationID == observed.id else {
            try await record(
                run,
                stepID: step.id,
                kind: .staleActionRejected,
                attempt: attempt,
                source: action.source,
                code: .observationIDMismatch
            )
            return .retry(actionWasPerformed: false)
        }

        var current = try await validObservation(for: step)
        guard observed.representsSameInteractionState(as: current) else {
            try await record(
                run,
                stepID: step.id,
                kind: .staleActionRejected,
                attempt: attempt,
                source: action.source,
                code: .interactionStateChanged
            )
            return .retry(actionWasPerformed: false)
        }

        let effectiveRisk = max(step.risk, action.risk)
        if effectiveRisk.requiresApproval {
            try await record(run, stepID: step.id, kind: .approvalRequested, attempt: attempt, source: action.source)
            let decision = await requestApproval(
                WorkflowApprovalRequest(
                    workflowID: workflow.id,
                    runID: run.id,
                    stepID: step.id,
                    stepTitle: Self.displaySafeSummary(step.title),
                    actionSummary: Self.displaySafeSummary(action.safeSummary),
                    risk: effectiveRisk
                )
            )
            if Task.isCancelled { throw CancellationError() }
            switch decision {
            case .denied:
                try await record(run, stepID: step.id, kind: .approvalDenied, attempt: attempt, source: action.source)
                return .paused(
                    .approvalDenied,
                    code: .userDenied,
                    actionMayHaveOccurred: false
                )
            case .unavailable:
                return .paused(
                    .approvalUnavailable,
                    code: .approvalUnavailable,
                    actionMayHaveOccurred: false
                )
            case .approved:
                try await record(run, stepID: step.id, kind: .approvalGranted, attempt: attempt, source: action.source)
            }

            if Task.isCancelled { throw CancellationError() }
            let afterApproval = try await validObservation(for: step)
            guard current.representsSameInteractionState(as: afterApproval) else {
                try await record(
                    run,
                    stepID: step.id,
                    kind: .staleActionRejected,
                    attempt: attempt,
                    source: action.source,
                    code: .stateChangedDuringApproval
                )
                return .retry(actionWasPerformed: false)
            }
            current = afterApproval
        }

        if Task.isCancelled { throw CancellationError() }
        do {
            try await actuator.perform(action, against: current)
        } catch is CancellationError {
            throw WorkflowKernelError.cancelled(actionMayHaveOccurred: true)
        } catch {
            throw WorkflowKernelError.dependencyFailure(actionMayHaveOccurred: true)
        }
        if Task.isCancelled {
            throw WorkflowKernelError.cancelled(actionMayHaveOccurred: true)
        }
        try await record(
            run,
            stepID: step.id,
            kind: .actionPerformed,
            attempt: attempt,
            source: action.source,
            actionMayHaveOccurredOnFailure: true
        )
        if Task.isCancelled {
            throw WorkflowKernelError.cancelled(actionMayHaveOccurred: true)
        }
        let after = try await validObservation(for: step, actionMayHaveOccurred: true)
        let verification: WorkflowVerification
        do {
            verification = try await verifier.verify(step: step, before: current, after: after)
        } catch is CancellationError {
            throw WorkflowKernelError.cancelled(actionMayHaveOccurred: true)
        } catch {
            throw WorkflowKernelError.dependencyFailure(actionMayHaveOccurred: true)
        }
        if Task.isCancelled {
            throw WorkflowKernelError.cancelled(actionMayHaveOccurred: true)
        }
        switch verification {
        case .satisfied:
            try await record(
                run,
                stepID: step.id,
                kind: .verificationPassed,
                attempt: attempt,
                source: action.source,
                actionMayHaveOccurredOnFailure: true
            )
            return .verified
        case .notSatisfied(let code):
            try await record(
                run,
                stepID: step.id,
                kind: .verificationFailed,
                attempt: attempt,
                source: action.source,
                code: WorkflowLedgerCode(code),
                actionMayHaveOccurredOnFailure: true
            )
            return .retry(actionWasPerformed: true)
        case .unsafe(let code):
            try await record(
                run,
                stepID: step.id,
                kind: .verificationFailed,
                attempt: attempt,
                source: action.source,
                code: WorkflowLedgerCode(code),
                actionMayHaveOccurredOnFailure: true
            )
            return .paused(
                .unsafeState,
                code: WorkflowLedgerCode(code),
                actionMayHaveOccurred: true
            )
        }
    }

    private func pause(
        _ original: LocalWorkflowRun,
        stepID: String,
        reason: WorkflowPauseReason,
        code: WorkflowLedgerCode,
        actionMayHaveOccurred: Bool
    ) async -> LocalWorkflowRun {
        var run = original
        run.status = .paused(
            stepID: stepID,
            reason: reason,
            actionMayHaveOccurred: actionMayHaveOccurred
        )
        do {
            try await record(
                run,
                stepID: stepID,
                kind: .runPaused,
                code: code,
                actionMayHaveOccurredOnFailure: actionMayHaveOccurred
            )
        } catch {
            return pausedWithoutRecording(
                run,
                stepID: stepID,
                reason: .dependencyFailure,
                actionMayHaveOccurred: actionMayHaveOccurred
            )
        }
        return run
    }

    private func pausedWithoutRecording(
        _ original: LocalWorkflowRun,
        stepID: String,
        reason: WorkflowPauseReason,
        actionMayHaveOccurred: Bool
    ) -> LocalWorkflowRun {
        var run = original
        run.status = .paused(
            stepID: stepID,
            reason: reason,
            actionMayHaveOccurred: actionMayHaveOccurred
        )
        return run
    }

    private enum WorkflowKernelError: Error {
        case invalidObservation(actionMayHaveOccurred: Bool)
        case dependencyFailure(actionMayHaveOccurred: Bool)
        case cancelled(actionMayHaveOccurred: Bool)
    }

    private func validObservation(
        for step: LocalWorkflowStep,
        actionMayHaveOccurred: Bool = false
    ) async throws -> WorkflowObservation {
        let observation: WorkflowObservation
        do {
            observation = try await observer.observe(for: step)
        } catch is CancellationError {
            throw WorkflowKernelError.cancelled(actionMayHaveOccurred: actionMayHaveOccurred)
        } catch {
            throw WorkflowKernelError.dependencyFailure(actionMayHaveOccurred: actionMayHaveOccurred)
        }
        guard observation.isStructurallyValid else {
            throw WorkflowKernelError.invalidObservation(
                actionMayHaveOccurred: actionMayHaveOccurred
            )
        }
        return observation
    }

    private func requestApproval(_ request: WorkflowApprovalRequest) async -> WorkflowApprovalDecision {
        await approver.requestApproval(
            request,
            timeoutNanoseconds: approvalTimeoutNanoseconds
        )
    }

    /// Model text shown in an approval prompt is untrusted display data. Make
    /// controls and Unicode formatting characters visible, flatten whitespace,
    /// and cap it so a misleading suffix cannot be pushed off-screen.
    private static func displaySafeSummary(_ raw: String, capBytes: Int = 240) -> String {
        let ellipsis = "…"
        let contentLimit = max(0, capBytes - ellipsis.utf8.count)
        var result = ""
        result.reserveCapacity(min(raw.utf8.count, contentLimit))
        var byteCount = 0
        var previousWasWhitespace = false
        var truncated = false
        for scalar in raw.unicodeScalars {
            let token: String
            if scalar.properties.isWhitespace {
                if result.isEmpty || previousWasWhitespace {
                    previousWasWhitespace = true
                    continue
                }
                token = " "
                previousWasWhitespace = true
            } else {
                previousWasWhitespace = false
                switch scalar.properties.generalCategory {
                case .control, .format, .lineSeparator, .paragraphSeparator,
                     .privateUse, .surrogate, .unassigned:
                    token = String(format: "\\u{%04X}", scalar.value)
                default:
                    token = String(scalar)
                }
            }
            let tokenBytes = token.utf8.count
            guard byteCount + tokenBytes <= contentLimit else {
                truncated = true
                break
            }
            result += token
            byteCount += tokenBytes
        }
        while result.last == " " { result.removeLast() }
        if result.isEmpty && !raw.isEmpty { return "Protected action" }
        return truncated ? result + ellipsis : result
    }

    private func cancel(
        _ original: LocalWorkflowRun,
        stepID: String?,
        actionMayHaveOccurred: Bool
    ) async -> LocalWorkflowRun {
        var run = original
        run.status = .cancelled(
            stepID: stepID,
            actionMayHaveOccurred: actionMayHaveOccurred
        )
        do {
            try await record(
                run,
                stepID: nil,
                kind: .runCancelled,
                actionMayHaveOccurredOnFailure: actionMayHaveOccurred
            )
        } catch {
            return pausedWithoutRecording(
                run,
                stepID: stepID ?? "ledger",
                reason: .dependencyFailure,
                actionMayHaveOccurred: actionMayHaveOccurred
            )
        }
        return run
    }

    private func record(
        _ run: LocalWorkflowRun,
        stepID: String?,
        kind: WorkflowLedgerEventKind,
        attempt: Int? = nil,
        source: WorkflowActionSource? = nil,
        code: WorkflowLedgerCode? = nil,
        actionMayHaveOccurredOnFailure: Bool = false
    ) async throws {
        do {
            try await ledger.append(
                WorkflowLedgerEvent(
                    runID: run.id,
                    workflowID: run.workflowID,
                    stepID: stepID,
                    kind: kind,
                    attempt: attempt,
                    actionSource: source,
                    code: code
                )
            )
        } catch {
            throw WorkflowKernelError.dependencyFailure(
                actionMayHaveOccurred: actionMayHaveOccurredOnFailure
            )
        }
    }
}
