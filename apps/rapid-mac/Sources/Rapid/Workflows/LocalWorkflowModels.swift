import Foundation

/// A learned desktop procedure expressed as semantic, verifiable steps.
///
/// The workflow intentionally stores intent rather than screen coordinates.
/// Coordinates belong to a short-lived grounded action and are valid only for
/// the exact observation from which they were produced.
struct LocalWorkflow: Codable, Equatable, Identifiable, Sendable {
    let id: UUID
    let title: String
    let steps: [LocalWorkflowStep]

    init(id: UUID = UUID(), title: String, steps: [LocalWorkflowStep]) {
        self.id = id
        self.title = title
        self.steps = steps
    }
}

struct LocalWorkflowStep: Codable, Equatable, Identifiable, Sendable {
    let id: String
    let title: String
    let instruction: String
    let successCriteria: String
    let risk: WorkflowActionRisk
    let isIdempotent: Bool
    let maxGroundingAttempts: Int
    let semanticFallbackIdentifier: String?

    init(
        id: String,
        title: String,
        instruction: String,
        successCriteria: String,
        risk: WorkflowActionRisk = .localChange,
        isIdempotent: Bool = false,
        maxGroundingAttempts: Int = 1,
        semanticFallbackIdentifier: String? = nil
    ) {
        self.id = id
        self.title = title
        self.instruction = instruction
        self.successCriteria = successCriteria
        self.risk = risk
        self.isIdempotent = isIdempotent
        self.maxGroundingAttempts = min(max(1, maxGroundingAttempts), 3)
        self.semanticFallbackIdentifier = semanticFallbackIdentifier
    }
}

/// The consequence class is authored by the compiled workflow, not trusted to
/// the visual model. The executor uses the stricter of the step and action.
enum WorkflowActionRisk: Int, Codable, Comparable, Sendable {
    case readOnly = 0
    case localChange = 1
    case externalCommunication = 2
    case financial = 3
    case destructive = 4

    static func < (lhs: Self, rhs: Self) -> Bool { lhs.rawValue < rhs.rawValue }

    var requiresApproval: Bool { self >= .externalCommunication }
}

struct WorkflowWindowFrame: Codable, Equatable, Sendable {
    let x: Double
    let y: Double
    let width: Double
    let height: Double

    var isStructurallyValid: Bool {
        x.isFinite && y.isFinite && width.isFinite && height.isFinite
            && width > 0 && height > 0
    }
}

struct WorkflowInteractionTarget: Codable, Equatable, Sendable {
    let bundleIdentifier: String
    let processIdentifier: Int32
    let windowIdentifier: String
    let windowFrame: WorkflowWindowFrame
}

/// Metadata for one screen observation. Pixel data and accessibility contents
/// remain ephemeral in the observer implementation and are never part of the
/// persistent run state or audit ledger.
struct WorkflowObservation: Equatable, Sendable {
    let id: UUID
    let target: WorkflowInteractionTarget
    let contentRevision: String

    init(
        id: UUID = UUID(),
        target: WorkflowInteractionTarget,
        contentRevision: String
    ) {
        self.id = id
        self.target = target
        self.contentRevision = contentRevision
    }

    func representsSameInteractionState(as other: Self) -> Bool {
        target == other.target && contentRevision == other.contentRevision
    }

    var isStructurallyValid: Bool {
        !target.bundleIdentifier.isEmpty
            && target.processIdentifier > 0
            && !target.windowIdentifier.isEmpty
            && target.windowFrame.isStructurallyValid
            && !contentRevision.isEmpty
    }
}

enum WorkflowActionPayload: Equatable, Sendable {
    /// Keeps a compromised or confused grounder from handing the input layer
    /// an unbounded allocation. This is bytes, not graphemes, so the bound is
    /// stable for every Unicode payload.
    static let maximumTypedTextBytes = 65_536

    case click(normalizedX: Double, normalizedY: Double)
    case typeText(String)
    case keyPress(key: String, modifiers: [String])

    var isStructurallyValid: Bool {
        switch self {
        case .click(let x, let y):
            x.isFinite && y.isFinite && (0 ... 1).contains(x) && (0 ... 1).contains(y)
        case .typeText(let text):
            !text.isEmpty && text.utf8.count <= Self.maximumTypedTextBytes
        case .keyPress(let key, let modifiers):
            !key.isEmpty && key.count <= 40 && modifiers.count <= 4
                && modifiers.allSatisfy { !$0.isEmpty && $0.count <= 20 }
        }
    }
}

enum WorkflowActionSource: String, Codable, Equatable, Sendable {
    case visualGrounding
    case semanticFallback
}

/// A runtime-only action. Payloads may contain typed text, so this type is
/// deliberately not Codable and must never be copied into the audit ledger.
struct GroundedWorkflowAction: Equatable, Sendable {
    let observationID: UUID
    let payload: WorkflowActionPayload
    let source: WorkflowActionSource
    let safeSummary: String
    let risk: WorkflowActionRisk
}

enum WorkflowVerificationCode: String, Codable, Equatable, Sendable {
    case targetUnchanged
    case unexpectedAccount
    case unexpectedDestination
    case focusChanged
    case windowChanged
    case contentChanged
    case unknown
}

enum WorkflowVerification: Equatable, Sendable {
    case satisfied
    case notSatisfied(code: WorkflowVerificationCode)
    case unsafe(code: WorkflowVerificationCode)
}

struct WorkflowApprovalRequest: Equatable, Sendable {
    let workflowID: UUID
    let runID: UUID
    let stepID: String
    let stepTitle: String
    let actionSummary: String
    let risk: WorkflowActionRisk
}

enum WorkflowApprovalDecision: Equatable, Sendable {
    case approved
    case denied
    case unavailable
}

enum WorkflowPauseReason: String, Codable, Equatable, Sendable {
    case approvalDenied
    case approvalUnavailable
    case recoveryExhausted
    case unsafeState
    case dependencyFailure
}

enum LocalWorkflowRunStatus: Codable, Equatable, Sendable {
    case ready
    case running
    case paused(stepID: String, reason: WorkflowPauseReason, actionMayHaveOccurred: Bool)
    case completed
    case cancelled(stepID: String?, actionMayHaveOccurred: Bool)

    var permitsExecution: Bool {
        switch self {
        case .ready:
            true
        case .running, .paused, .completed, .cancelled:
            false
        }
    }

    /// Conservative persisted uncertainty used when an invalid resume is
    /// rejected. A run interrupted while executing may already have crossed
    /// the external side-effect boundary even if no result was recorded.
    var actionMayHaveOccurred: Bool {
        switch self {
        case .running, .completed:
            true
        case .paused(_, _, let value), .cancelled(_, let value):
            value
        case .ready:
            false
        }
    }
}

struct LocalWorkflowRun: Codable, Equatable, Identifiable, Sendable {
    let id: UUID
    let workflowID: UUID
    var nextStepIndex: Int
    var status: LocalWorkflowRunStatus

    init(
        id: UUID = UUID(),
        workflowID: UUID,
        nextStepIndex: Int = 0,
        status: LocalWorkflowRunStatus = .ready
    ) {
        self.id = id
        self.workflowID = workflowID
        self.nextStepIndex = nextStepIndex
        self.status = status
    }
}

enum WorkflowLedgerEventKind: String, Codable, Equatable, Sendable {
    case runStarted
    case actionGrounded
    case staleActionRejected
    case approvalRequested
    case approvalGranted
    case approvalDenied
    case actionPerformed
    case verificationPassed
    case verificationFailed
    case semanticFallbackUsed
    case runPaused
    case runCompleted
    case runCancelled
}

/// Deliberately metadata-only. Instructions, screenshots, typed values, model
/// reasoning, errors, and clipboard contents do not have fields to leak into.
struct WorkflowLedgerEvent: Codable, Equatable, Sendable {
    let runID: UUID
    let workflowID: UUID
    let stepID: String?
    let kind: WorkflowLedgerEventKind
    let attempt: Int?
    let actionSource: WorkflowActionSource?
    let code: WorkflowLedgerCode?
}

enum WorkflowLedgerCode: String, Codable, Equatable, Sendable {
    case invalidResumeState
    case executorBusy
    case observationIDMismatch
    case interactionStateChanged
    case invalidAction
    case invalidObservation
    case stateChangedDuringApproval
    case userDenied
    case approvalUnavailable
    case visualRetriesExhausted
    case fallbackUnavailable
    case fallbackNotVerified
    case dependencyFailure
    case targetUnchanged
    case unexpectedAccount
    case unexpectedDestination
    case focusChanged
    case windowChanged
    case contentChanged
    case unknown

    init(_ verification: WorkflowVerificationCode) {
        self = WorkflowLedgerCode(rawValue: verification.rawValue) ?? .unknown
    }
}
