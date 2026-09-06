import AppKit
import ApplicationServices
import Foundation

enum MacOSComputerUseActuationError: Error, Equatable {
    case permissionMissing([MacAutomationPermission])
    case invalidAction
    case staleObservation
    case targetUnavailable
    case targetNotFrontmost
    case targetChanged
    case targetOccluded
    case eventCreationFailed
}

protocol ComputerUseTargetProbing: Sendable {
    func currentTarget(for expected: WorkflowInteractionTarget) async throws
        -> WorkflowInteractionTarget
}

protocol ComputerUseContentProbing: Sendable {
    func currentObservation(for expected: WorkflowInteractionTarget) async throws
        -> WorkflowObservation
}

protocol ComputerUseInputEmitting: Sendable {
    func emit(
        _ payload: WorkflowActionPayload,
        in target: WorkflowInteractionTarget
    ) async throws
}

/// Production action boundary for local Computer Use.
///
/// The executor already re-observes immediately before calling this adapter.
/// The adapter intentionally repeats the safety checks at the final input
/// boundary: permission, observation identity, foreground process, exact
/// window identity, and window geometry must all still match. It never
/// activates an app or searches for a similar window on the user's behalf.
actor MacOSComputerUseActuator: LocalWorkflowActuating {
    typealias PermissionReader = @Sendable () -> MacAutomationPermissionSnapshot

    private let permissionReader: PermissionReader
    private let targetProbe: any ComputerUseTargetProbing
    private let contentProbe: any ComputerUseContentProbing
    private let inputEmitter: any ComputerUseInputEmitting

    init(
        targetProbe: any ComputerUseTargetProbing = CGWindowComputerUseTargetProbe(),
        contentProbe: any ComputerUseContentProbing =
            ScreenCaptureKitComputerUseContentProbe(),
        inputEmitter: any ComputerUseInputEmitting = CGEventComputerUseInputEmitter(),
        permissionReader: @escaping PermissionReader = MacAutomationPermissions.snapshot
    ) {
        self.targetProbe = targetProbe
        self.contentProbe = contentProbe
        self.inputEmitter = inputEmitter
        self.permissionReader = permissionReader
    }

    func perform(
        _ action: GroundedWorkflowAction,
        groundedAgainst groundingObservation: WorkflowObservation,
        currentObservation: WorkflowObservation
    ) async throws {
        try Task.checkCancellation()
        // The kernel deliberately re-observes before actuation, so the two
        // observation UUIDs differ. Carry both states across this boundary so
        // the adapter can independently verify the action's original binding
        // and the fresh observation's target/content equivalence.
        guard groundingObservation.isStructurallyValid,
              currentObservation.isStructurallyValid,
              action.observationID == groundingObservation.id,
              groundingObservation.representsSameInteractionState(
                as: currentObservation
              )
        else {
            throw MacOSComputerUseActuationError.staleObservation
        }
        guard Self.isSafeForWindowActuation(action.payload) else {
            throw MacOSComputerUseActuationError.invalidAction
        }

        let permissions = permissionReader()
        guard permissions.isReadyForComputerUse else {
            throw MacOSComputerUseActuationError.permissionMissing(
                permissions.missingForComputerUse
            )
        }

        let liveTarget = try await targetProbe.currentTarget(
            for: currentObservation.target
        )
        try Task.checkCancellation()
        guard MacOSComputerUseWindowIdentity.targetsMatch(
            liveTarget,
            currentObservation.target
        ) else {
            throw MacOSComputerUseActuationError.targetChanged
        }
        // Capture once more after the target preflight. A matching frame alone
        // is insufficient: controls can move while the window identity and
        // geometry remain unchanged. Do not emit coordinates grounded against
        // pixels that are no longer current.
        let finalObservation = try await contentProbe.currentObservation(
            for: currentObservation.target
        )
        try Task.checkCancellation()
        guard currentObservation.representsSameInteractionState(
            as: finalObservation
        ) else {
            throw MacOSComputerUseActuationError.staleObservation
        }
        // The production emitter repeats this probe synchronously in the same
        // MainActor turn as each CGEvent post. The preflight here keeps all
        // emitters testable and rejects drift before entering the input layer.
        try await inputEmitter.emit(action.payload, in: currentObservation.target)
    }

    private static func isSafeForWindowActuation(
        _ payload: WorkflowActionPayload
    ) -> Bool {
        guard payload.isStructurallyValid else { return false }
        if case .click(let x, let y) = payload {
            // 0 and 1 are window edges, not interior points. A CGEvent at
            // maxX/maxY can belong to the adjacent or underlying window.
            return x > 0 && x < 1 && y > 0 && y < 1
        }
        // Public macOS event APIs can target a process, but not one exact
        // window inside that process. Until a semantic Accessibility adapter
        // can bind text/key input to a verified element in this window, these
        // payloads must fail closed rather than rely on focus timing.
        return false
    }
}

/// Performs the final content-revision check without retaining another image.
struct ScreenCaptureKitComputerUseContentProbe: ComputerUseContentProbing {
    private let captureSource: any ComputerUseWindowCapturing

    init(
        captureSource: any ComputerUseWindowCapturing =
            ScreenCaptureKitComputerUseCapture()
    ) {
        self.captureSource = captureSource
    }

    func currentObservation(
        for expected: WorkflowInteractionTarget
    ) async throws -> WorkflowObservation {
        guard let windowID = CGWindowID(expected.windowIdentifier), windowID != 0 else {
            throw MacOSComputerUseActuationError.targetUnavailable
        }
        let captured = try await captureSource.capture(
            ComputerUseWindowSelection(
                bundleIdentifier: expected.bundleIdentifier,
                processIdentifier: expected.processIdentifier,
                windowID: windowID
            )
        )
        try Task.checkCancellation()
        guard captured.isStructurallyValid,
              MacOSComputerUseWindowIdentity.targetsMatch(
                captured.target,
                expected
              )
        else {
            throw MacOSComputerUseActuationError.targetChanged
        }
        return WorkflowObservation(
            target: captured.target,
            contentRevision: MacOSComputerUseObserver.contentRevision(
                target: captured.target,
                pngData: captured.artifact.pngData
            )
        )
    }
}

/// Reads one exact CGWindow entry and requires its owner to remain frontmost.
/// This is a probe only; it has no activation or focus-changing side effects.
struct CGWindowComputerUseTargetProbe: ComputerUseTargetProbing {
    func currentTarget(for expected: WorkflowInteractionTarget) async throws
        -> WorkflowInteractionTarget
    {
        try Task.checkCancellation()
        guard let windowID = CGWindowID(expected.windowIdentifier), windowID != 0 else {
            throw MacOSComputerUseActuationError.targetUnavailable
        }

        return try await MainActor.run {
            try Self.currentTargetSynchronously(for: expected)
        }
    }

    @MainActor
    static func currentTargetSynchronously(
        for expected: WorkflowInteractionTarget
    ) throws -> WorkflowInteractionTarget {
        guard let windowID = CGWindowID(expected.windowIdentifier), windowID != 0 else {
            throw MacOSComputerUseActuationError.targetUnavailable
        }
        guard NSWorkspace.shared.frontmostApplication?.processIdentifier
                == expected.processIdentifier
        else {
            throw MacOSComputerUseActuationError.targetNotFrontmost
        }
        guard let focusedFrame = MacOSComputerUseWindowIdentity.focusedWindowFrame(
            processIdentifier: expected.processIdentifier
        ) else {
            throw MacOSComputerUseActuationError.targetNotFrontmost
        }
        guard let records = CGWindowListCopyWindowInfo(
            [.optionOnScreenOnly, .excludeDesktopElements],
            kCGNullWindowID
        ) as? [[CFString: Any]],
            let record = records.first(where: {
                ($0[kCGWindowNumber] as? NSNumber)?.uint32Value == windowID
            }),
            let ownerPID = record[kCGWindowOwnerPID] as? NSNumber,
            ownerPID.int32Value == expected.processIdentifier,
            let bounds = record[kCGWindowBounds] as? [String: NSNumber],
            let frame = CGRect(dictionaryRepresentation: bounds as CFDictionary),
            frame.width > 0,
            frame.height > 0,
            let application = NSRunningApplication(
                processIdentifier: expected.processIdentifier
            ),
            let bundleIdentifier = application.bundleIdentifier
        else {
            throw MacOSComputerUseActuationError.targetUnavailable
        }

        let focusedCandidates = records.filter { record in
            guard let ownerPID = record[kCGWindowOwnerPID] as? NSNumber,
                  ownerPID.int32Value == expected.processIdentifier,
                  let bounds = record[kCGWindowBounds] as? [String: NSNumber],
                  let frame = CGRect(
                    dictionaryRepresentation: bounds as CFDictionary
                  )
            else { return false }
            return MacOSComputerUseWindowIdentity.framesMatch(frame, focusedFrame)
        }
        guard focusedCandidates.count == 1,
              (focusedCandidates[0][kCGWindowNumber] as? NSNumber)?.uint32Value
                == windowID
        else {
            throw MacOSComputerUseActuationError.targetNotFrontmost
        }

        return WorkflowInteractionTarget(
            bundleIdentifier: bundleIdentifier,
            processIdentifier: expected.processIdentifier,
            windowIdentifier: String(windowID),
            windowFrame: WorkflowWindowFrame(
                x: frame.origin.x,
                y: frame.origin.y,
                width: frame.width,
                height: frame.height
            )
        )
    }
}

/// Narrow CGEvent adapter for process-bound clicks in one verified window.
struct CGEventComputerUseInputEmitter: ComputerUseInputEmitting {
    typealias TargetReader = @MainActor @Sendable (
        WorkflowInteractionTarget
    ) throws -> WorkflowInteractionTarget
    typealias CancellationCheck = @MainActor @Sendable () throws -> Void
    typealias WindowAtPointReader = @MainActor @Sendable (CGPoint) -> String?
    typealias EventPoster = @MainActor @Sendable (CGEvent, pid_t) -> Void

    private let targetReader: TargetReader
    private let cancellationCheck: CancellationCheck
    private let windowAtPointReader: WindowAtPointReader
    private let eventPoster: EventPoster

    init(
        targetReader: @escaping TargetReader = {
            try CGWindowComputerUseTargetProbe.currentTargetSynchronously(for: $0)
        },
        cancellationCheck: @escaping CancellationCheck = { try Task.checkCancellation() },
        windowAtPointReader: @escaping WindowAtPointReader = {
            MacOSComputerUseWindowIdentity.topmostWindowIdentifier(at: $0)
        },
        eventPoster: @escaping EventPoster = { event, processIdentifier in
            event.postToPid(processIdentifier)
        }
    ) {
        self.targetReader = targetReader
        self.cancellationCheck = cancellationCheck
        self.windowAtPointReader = windowAtPointReader
        self.eventPoster = eventPoster
    }

    func emit(
        _ payload: WorkflowActionPayload,
        in target: WorkflowInteractionTarget
    ) async throws {
        try Task.checkCancellation()

        try await MainActor.run {
            guard let source = CGEventSource(stateID: .combinedSessionState) else {
                throw MacOSComputerUseActuationError.eventCreationFailed
            }

            switch payload {
            case .click(let normalizedX, let normalizedY):
                let initial = try Self.requireCurrent(
                    target,
                    using: targetReader,
                    cancellationCheck: cancellationCheck
                )
                let point = try Self.clickPoint(
                    normalizedX: normalizedX,
                    normalizedY: normalizedY,
                    in: initial.windowFrame
                )
                guard let down = CGEvent(
                    mouseEventSource: source,
                    mouseType: .leftMouseDown,
                    mouseCursorPosition: point,
                    mouseButton: .left
                ),
                    let up = CGEvent(
                        mouseEventSource: source,
                        mouseType: .leftMouseUp,
                        mouseCursorPosition: point,
                        mouseButton: .left
                    )
                else {
                    throw MacOSComputerUseActuationError.eventCreationFailed
                }
                try Self.post(
                    down,
                    at: point,
                    expected: target,
                    targetReader: targetReader,
                    cancellationCheck: cancellationCheck,
                    windowAtPointReader: windowAtPointReader,
                    eventPoster: eventPoster
                )
                do {
                    try Self.post(
                        up,
                        at: point,
                        expected: target,
                        targetReader: targetReader,
                        cancellationCheck: cancellationCheck,
                        windowAtPointReader: windowAtPointReader,
                        eventPoster: eventPoster
                    )
                } catch {
                    // Once mouse-down has reached the selected process, always
                    // balance its button state. Leaving it held can turn later
                    // user movement into an unintended drag. This cleanup is
                    // still process-bound; the original validation error is
                    // preserved so the workflow cannot advance.
                    eventPoster(up, target.processIdentifier)
                    throw error
                }

            case .typeText, .keyPress:
                throw MacOSComputerUseActuationError.invalidAction
            }
        }
    }

    @MainActor
    private static func post(
        _ event: CGEvent,
        at point: CGPoint,
        expected: WorkflowInteractionTarget,
        targetReader: TargetReader,
        cancellationCheck: CancellationCheck,
        windowAtPointReader: WindowAtPointReader,
        eventPoster: EventPoster
    ) throws {
        _ = try requireCurrent(
            expected,
            using: targetReader,
            cancellationCheck: cancellationCheck
        )
        guard windowAtPointReader(point) == expected.windowIdentifier else {
            throw MacOSComputerUseActuationError.targetOccluded
        }
        try cancellationCheck()
        eventPoster(event, expected.processIdentifier)
    }

    static func clickPoint(
        normalizedX: Double,
        normalizedY: Double,
        in frame: WorkflowWindowFrame
    ) throws -> CGPoint {
        let point = CGPoint(
            x: frame.x + normalizedX * frame.width,
            y: frame.y + normalizedY * frame.height
        )
        // A mathematically interior normalized value can round onto a pixel at
        // the frame boundary. Never let that event escape the live window.
        guard point.x > frame.x,
              point.x < frame.x + frame.width,
              point.y > frame.y,
              point.y < frame.y + frame.height
        else {
            throw MacOSComputerUseActuationError.invalidAction
        }
        return point
    }

    @MainActor
    private static func requireCurrent(
        _ expected: WorkflowInteractionTarget,
        using targetReader: TargetReader,
        cancellationCheck: CancellationCheck
    ) throws -> WorkflowInteractionTarget {
        let current = try targetReader(expected)
        guard MacOSComputerUseWindowIdentity.targetsMatch(current, expected) else {
            throw MacOSComputerUseActuationError.targetChanged
        }
        try cancellationCheck()
        return current
    }
}
