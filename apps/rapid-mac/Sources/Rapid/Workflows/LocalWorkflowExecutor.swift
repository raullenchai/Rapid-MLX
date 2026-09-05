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
    func requestApproval(_ request: WorkflowApprovalRequest) async -> WorkflowApprovalDecision
}

protocol LocalWorkflowLedgerWriting: Sendable {
    func append(_ event: WorkflowLedgerEvent) async
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

    init(
        observer: any LocalWorkflowObserving,
        grounder: any LocalWorkflowGrounding,
        actuator: any LocalWorkflowActuating,
        verifier: any LocalWorkflowVerifying,
        fallbackResolver: any LocalWorkflowFallbackResolving,
        approver: any LocalWorkflowApproving,
        ledger: any LocalWorkflowLedgerWriting
    ) {
        self.observer = observer
        self.grounder = grounder
        self.actuator = actuator
        self.verifier = verifier
        self.fallbackResolver = fallbackResolver
        self.approver = approver
        self.ledger = ledger
    }

    func execute(_ workflow: LocalWorkflow, resuming run: LocalWorkflowRun? = nil) async -> LocalWorkflowRun {
        var run = run ?? LocalWorkflowRun(workflowID: workflow.id)
        guard run.workflowID == workflow.id,
              run.nextStepIndex >= 0,
              run.nextStepIndex <= workflow.steps.count,
              Set(workflow.steps.map(\.id)).count == workflow.steps.count,
              workflow.steps.allSatisfy({ !$0.id.isEmpty })
        else {
            return await pause(run, stepID: "invalid-run", reason: .unsafeState, code: .invalidResumeState)
        }

        if run.nextStepIndex == workflow.steps.count {
            guard run.status.permitsExecution || run.status == .completed else {
                return await pause(run, stepID: "invalid-run", reason: .unsafeState, code: .invalidResumeState)
            }
            run.status = .completed
            await record(run, stepID: nil, kind: .runCompleted)
            return run
        }

        guard run.status.permitsExecution else {
            return await pause(run, stepID: "invalid-run", reason: .unsafeState, code: .invalidResumeState)
        }

        run.status = .running
        await record(run, stepID: nil, kind: .runStarted)

        while run.nextStepIndex < workflow.steps.count {
            if Task.isCancelled { return await cancel(run) }
            let step = workflow.steps[run.nextStepIndex]
            let result = await executeStep(step, workflow: workflow, run: run)
            switch result {
            case .completed:
                run.nextStepIndex += 1
                run.status = .running
            case .paused(let reason, let code):
                return await pause(run, stepID: step.id, reason: reason, code: code)
            case .cancelled:
                return await cancel(run)
            }
        }

        run.status = .completed
        await record(run, stepID: nil, kind: .runCompleted)
        return run
    }

    private enum StepResult {
        case completed
        case paused(WorkflowPauseReason, code: WorkflowLedgerCode)
        case cancelled
    }

    private func executeStep(
        _ step: LocalWorkflowStep,
        workflow: LocalWorkflow,
        run: LocalWorkflowRun
    ) async -> StepResult {
        do {
            // Decode is allowed to bypass ``LocalWorkflowStep``'s initializer,
            // so clamp again at the trust boundary before constructing a range.
            let attemptLimit = min(max(1, step.maxGroundingAttempts), 3)
            for attempt in 1 ... attemptLimit {
                if Task.isCancelled { return .cancelled }
                let observed = try await validObservation(for: step)
                let proposed = try await grounder.ground(step: step, observation: observed)
                let action = GroundedWorkflowAction(
                    observationID: proposed.observationID,
                    payload: proposed.payload,
                    source: .visualGrounding,
                    safeSummary: proposed.safeSummary,
                    risk: proposed.risk
                )
                await record(
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
                    if actionWasPerformed && !step.isIdempotent {
                        return .paused(.recoveryExhausted, code: .visualRetriesExhausted)
                    }
                    continue
                case .paused(let reason, let code):
                    return .paused(reason, code: code)
                }
            }

            guard let fallbackIdentifier = step.semanticFallbackIdentifier else {
                return .paused(.recoveryExhausted, code: .visualRetriesExhausted)
            }
            guard step.isIdempotent else {
                return .paused(.recoveryExhausted, code: .visualRetriesExhausted)
            }
            let observed = try await validObservation(for: step)
            guard var fallback = try await fallbackResolver.fallbackAction(
                identifier: fallbackIdentifier,
                step: step,
                observation: observed
            ) else {
                return .paused(.recoveryExhausted, code: .fallbackUnavailable)
            }
            fallback = GroundedWorkflowAction(
                observationID: fallback.observationID,
                payload: fallback.payload,
                source: .semanticFallback,
                safeSummary: fallback.safeSummary,
                risk: fallback.risk
            )
            await record(
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
            case .retry:
                return .paused(.recoveryExhausted, code: .fallbackNotVerified)
            case .paused(let reason, let code):
                return .paused(reason, code: code)
            }
        } catch is CancellationError {
            return .cancelled
        } catch WorkflowKernelError.invalidObservation {
            return .paused(.unsafeState, code: .invalidObservation)
        } catch {
            return .paused(.dependencyFailure, code: .dependencyFailure)
        }
    }

    private enum AttemptResult {
        case verified
        case retry(actionWasPerformed: Bool)
        case paused(WorkflowPauseReason, code: WorkflowLedgerCode)
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
            await record(
                run,
                stepID: step.id,
                kind: .staleActionRejected,
                attempt: attempt,
                source: action.source,
                code: .invalidAction
            )
            return .paused(.unsafeState, code: .invalidAction)
        }
        guard action.observationID == observed.id else {
            await record(
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
            await record(
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
            await record(run, stepID: step.id, kind: .approvalRequested, attempt: attempt, source: action.source)
            let decision = await approver.requestApproval(
                WorkflowApprovalRequest(
                    workflowID: workflow.id,
                    runID: run.id,
                    stepID: step.id,
                    stepTitle: step.title,
                    actionSummary: Self.displaySafeSummary(action.safeSummary),
                    risk: effectiveRisk
                )
            )
            switch decision {
            case .denied:
                await record(run, stepID: step.id, kind: .approvalDenied, attempt: attempt, source: action.source)
                return .paused(.approvalDenied, code: .userDenied)
            case .unavailable:
                return .paused(.approvalUnavailable, code: .approvalUnavailable)
            case .approved:
                await record(run, stepID: step.id, kind: .approvalGranted, attempt: attempt, source: action.source)
            }

            if Task.isCancelled { throw CancellationError() }
            let afterApproval = try await validObservation(for: step)
            guard current.representsSameInteractionState(as: afterApproval) else {
                await record(
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
        try await actuator.perform(action, against: current)
        await record(run, stepID: step.id, kind: .actionPerformed, attempt: attempt, source: action.source)
        let after = try await validObservation(for: step)
        switch try await verifier.verify(step: step, before: current, after: after) {
        case .satisfied:
            await record(run, stepID: step.id, kind: .verificationPassed, attempt: attempt, source: action.source)
            return .verified
        case .notSatisfied(let code):
            await record(
                run,
                stepID: step.id,
                kind: .verificationFailed,
                attempt: attempt,
                source: action.source,
                code: WorkflowLedgerCode(code)
            )
            return .retry(actionWasPerformed: true)
        case .unsafe(let code):
            await record(
                run,
                stepID: step.id,
                kind: .verificationFailed,
                attempt: attempt,
                source: action.source,
                code: WorkflowLedgerCode(code)
            )
            return .paused(.unsafeState, code: WorkflowLedgerCode(code))
        }
    }

    private func pause(
        _ original: LocalWorkflowRun,
        stepID: String,
        reason: WorkflowPauseReason,
        code: WorkflowLedgerCode
    ) async -> LocalWorkflowRun {
        var run = original
        run.status = .paused(stepID: stepID, reason: reason)
        await record(run, stepID: stepID, kind: .runPaused, code: code)
        return run
    }

    private enum WorkflowKernelError: Error {
        case invalidObservation
    }

    private func validObservation(for step: LocalWorkflowStep) async throws -> WorkflowObservation {
        let observation = try await observer.observe(for: step)
        guard observation.isStructurallyValid else {
            throw WorkflowKernelError.invalidObservation
        }
        return observation
    }

    /// Model text shown in an approval prompt is untrusted display data. Make
    /// controls and Unicode formatting characters visible, flatten whitespace,
    /// and cap it so a misleading suffix cannot be pushed off-screen.
    private static func displaySafeSummary(_ raw: String, cap: Int = 240) -> String {
        var result = ""
        result.reserveCapacity(min(raw.count, cap))
        var previousWasWhitespace = false
        for scalar in raw.unicodeScalars {
            if result.count >= cap { break }
            if scalar.properties.isWhitespace {
                if !result.isEmpty && !previousWasWhitespace { result.append(" ") }
                previousWasWhitespace = true
                continue
            }
            previousWasWhitespace = false
            switch scalar.properties.generalCategory {
            case .control, .format, .lineSeparator, .paragraphSeparator,
                 .privateUse, .surrogate, .unassigned:
                result += String(format: "\\u{%04X}", scalar.value)
            default:
                result.unicodeScalars.append(scalar)
            }
        }
        while result.last == " " { result.removeLast() }
        return raw.count > cap ? result + "…" : result
    }

    private func cancel(_ original: LocalWorkflowRun) async -> LocalWorkflowRun {
        var run = original
        run.status = .cancelled
        await record(run, stepID: nil, kind: .runCancelled)
        return run
    }

    private func record(
        _ run: LocalWorkflowRun,
        stepID: String?,
        kind: WorkflowLedgerEventKind,
        attempt: Int? = nil,
        source: WorkflowActionSource? = nil,
        code: WorkflowLedgerCode? = nil
    ) async {
        await ledger.append(
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
    }
}
