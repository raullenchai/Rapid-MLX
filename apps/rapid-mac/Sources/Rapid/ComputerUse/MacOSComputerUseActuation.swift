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
    case unsupportedKey
    case eventCreationFailed
}

protocol ComputerUseTargetProbing: Sendable {
    func currentTarget(for expected: WorkflowInteractionTarget) async throws
        -> WorkflowInteractionTarget
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
    private let inputEmitter: any ComputerUseInputEmitting

    init(
        targetProbe: any ComputerUseTargetProbing = CGWindowComputerUseTargetProbe(),
        inputEmitter: any ComputerUseInputEmitting = CGEventComputerUseInputEmitter(),
        permissionReader: @escaping PermissionReader = MacAutomationPermissions.snapshot
    ) {
        self.targetProbe = targetProbe
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
        return true
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

/// Narrow CGEvent adapter. It supports only the three payloads authored by the
/// workflow kernel and an explicit key/modifier allowlist.
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

    private static let keyCodes: [String: CGKeyCode] = [
        "return": 36,
        "tab": 48,
        "space": 49,
        "delete": 51,
        "escape": 53,
        "left": 123,
        "right": 124,
        "down": 125,
        "up": 126,
    ]
    private static let modifierFlags: [String: CGEventFlags] = [
        "command": .maskCommand,
        "shift": .maskShift,
        "option": .maskAlternate,
        "control": .maskControl,
    ]
    private static let allowedKeyChords: Set<String> = [
        "return", "tab", "space", "delete", "escape",
        "left", "right", "down", "up",
        "shift+tab",
        "shift+left", "shift+right", "shift+down", "shift+up",
        "option+left", "option+right",
        "command+left", "command+right", "command+down", "command+up",
    ]

    func emit(
        _ payload: WorkflowActionPayload,
        in target: WorkflowInteractionTarget
    ) async throws {
        try Task.checkCancellation()
        let textChunks: [[UniChar]]
        if case .typeText(let text) = payload {
            textChunks = Self.unicodeChunks(for: text)
        } else {
            textChunks = []
        }

        try await MainActor.run {
            guard let source = CGEventSource(stateID: .combinedSessionState) else {
                throw MacOSComputerUseActuationError.eventCreationFailed
            }

            switch payload {
            case .click(let normalizedX, let normalizedY):
                let current = try Self.requireCurrent(
                    target,
                    using: targetReader,
                    cancellationCheck: cancellationCheck
                )
                let point = try Self.clickPoint(
                    normalizedX: normalizedX,
                    normalizedY: normalizedY,
                    in: current.windowFrame
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
                guard windowAtPointReader(point) == target.windowIdentifier else {
                    throw MacOSComputerUseActuationError.targetOccluded
                }
                try cancellationCheck()
                eventPoster(down, target.processIdentifier)
                eventPoster(up, target.processIdentifier)

            case .typeText:
                // CGEvent accepts UTF-16. Scalar-safe chunks keep text off the
                // clipboard without splitting surrogate pairs between events.
                for chunk in textChunks {
                    try Task.checkCancellation()
                    _ = try Self.requireCurrent(
                        target,
                        using: targetReader,
                        cancellationCheck: cancellationCheck
                    )
                    guard let down = CGEvent(
                        keyboardEventSource: source,
                        virtualKey: 0,
                        keyDown: true
                    ),
                        let up = CGEvent(
                            keyboardEventSource: source,
                            virtualKey: 0,
                            keyDown: false
                        )
                    else {
                        throw MacOSComputerUseActuationError.eventCreationFailed
                    }
                    down.keyboardSetUnicodeString(
                        stringLength: chunk.count,
                        unicodeString: chunk
                    )
                    up.keyboardSetUnicodeString(
                        stringLength: chunk.count,
                        unicodeString: chunk
                    )
                    eventPoster(down, target.processIdentifier)
                    eventPoster(up, target.processIdentifier)
                }

            case .keyPress(let key, let modifiers):
                let normalizedKey = key.lowercased()
                let normalizedModifiers = modifiers.map { $0.lowercased() }
                guard Set(normalizedModifiers).count == normalizedModifiers.count,
                      let keyCode = Self.keyCodes[normalizedKey],
                      normalizedModifiers.allSatisfy({ Self.modifierFlags[$0] != nil }),
                      Self.allowedKeyChords.contains(
                        (normalizedModifiers.sorted() + [normalizedKey]).joined(separator: "+")
                      )
                else {
                    throw MacOSComputerUseActuationError.unsupportedKey
                }
                var flags: CGEventFlags = []
                for modifier in normalizedModifiers {
                    guard let flag = Self.modifierFlags[modifier] else {
                        throw MacOSComputerUseActuationError.unsupportedKey
                    }
                    flags.insert(flag)
                }
                guard let down = CGEvent(
                    keyboardEventSource: source,
                    virtualKey: keyCode,
                    keyDown: true
                ),
                    let up = CGEvent(
                        keyboardEventSource: source,
                        virtualKey: keyCode,
                        keyDown: false
                    )
                else {
                    throw MacOSComputerUseActuationError.eventCreationFailed
                }
                down.flags = flags
                up.flags = flags
                try Task.checkCancellation()
                _ = try Self.requireCurrent(
                    target,
                    using: targetReader,
                    cancellationCheck: cancellationCheck
                )
                eventPoster(down, target.processIdentifier)
                eventPoster(up, target.processIdentifier)
            }
        }
    }

    static func unicodeChunks(
        for text: String,
        maximumUTF16Units: Int = 1_024
    ) -> [[UniChar]] {
        precondition(maximumUTF16Units >= 2)
        var chunks: [[UniChar]] = []
        var current: [UniChar] = []
        current.reserveCapacity(maximumUTF16Units)

        for scalar in text.unicodeScalars {
            let units = Array(String(scalar).utf16)
            if current.count + units.count > maximumUTF16Units {
                chunks.append(current)
                current = []
                current.reserveCapacity(maximumUTF16Units)
            }
            current.append(contentsOf: units)
        }
        if !current.isEmpty { chunks.append(current) }
        return chunks
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
