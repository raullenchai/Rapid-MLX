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
    case elementUnavailable
    case elementChanged
    case elementActionUnsupported
    case elementActionFailed
}

protocol ComputerUseTargetProbing: Sendable {
    func currentTarget(for expected: WorkflowInteractionTarget) async throws
        -> WorkflowInteractionTarget
}

protocol ComputerUseInputEmitting: Sendable {
    func emit(
        _ payload: WorkflowActionPayload,
        verifiedAgainst observation: WorkflowObservation
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
        inputEmitter: any ComputerUseInputEmitting =
            AXComputerUseInputEmitter(),
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
        // The production emitter binds the click to one Accessibility element,
        // re-captures the exact window, then re-resolves and presses that same
        // semantic element. The preflight here keeps alternate emitters
        // testable and rejects drift before entering the input layer.
        try await inputEmitter.emit(
            action.payload,
            verifiedAgainst: currentObservation
        )
    }

    private static func isSafeForWindowActuation(
        _ payload: WorkflowActionPayload
    ) -> Bool {
        guard payload.isStructurallyValid else { return false }
        if case .click(let x, let y) = payload {
            // 0 and 1 are window edges, not interior points. A coordinate at
            // maxX/maxY can resolve to an adjacent or underlying element.
            return x > 0 && x < 1 && y > 0 && y < 1
        }
        // Public keyboard event APIs can target a process, but not one exact
        // window inside that process. Until semantic Accessibility input can
        // bind text/key actions to a verified element in this window, these
        // payloads must fail closed rather than rely on focus timing.
        return false
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
            let bundleIdentifier = application.bundleIdentifier,
            let launchDate = application.launchDate,
            launchDate == expected.processLaunchDate
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
            processLaunchDate: launchDate,
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

struct ComputerUseElementFingerprint: Equatable, Sendable {
    let role: String
    let subrole: String?
    let identifier: String?
    let title: String?
    let frame: WorkflowWindowFrame
}

/// Main-actor token that retains the exact Accessibility object resolved
/// before the final window capture. Metadata is kept as a second, defensive
/// check, but is not sufficient by itself: a replacement control can expose
/// identical role, title, identifier, and geometry.
@MainActor
final class ComputerUseElementBinding {
    let fingerprint: ComputerUseElementFingerprint
    fileprivate let element: AXUIElement?

    init(
        fingerprint: ComputerUseElementFingerprint,
        element: AXUIElement? = nil
    ) {
        self.fingerprint = fingerprint
        self.element = element
    }

    func representsSameElement(as other: ComputerUseElementBinding) -> Bool {
        guard fingerprint == other.fingerprint else { return false }
        switch (element, other.element) {
        case let (original?, current?):
            return CFEqual(original, current)
        case (nil, nil):
            // Synthetic bindings exist only behind the injected unit-test
            // boundary. Preserve identity semantics there as well.
            return self === other
        default:
            return false
        }
    }
}

/// Element-bound click adapter. A coordinate is used only to resolve the
/// current Accessibility element; input is delivered with AXPress to that
/// element rather than through the global pointer event stream.
struct AXComputerUseInputEmitter: ComputerUseInputEmitting {
    typealias ElementResolver = @MainActor @Sendable (
        WorkflowActionPayload,
        WorkflowInteractionTarget
    ) throws -> ComputerUseElementBinding
    typealias ElementPerformer = @MainActor @Sendable (
        ComputerUseElementBinding
    ) throws -> Void

    private let captureSource: any ComputerUseWindowCapturing
    private let elementResolver: ElementResolver
    private let elementPerformer: ElementPerformer

    init(
        captureSource: any ComputerUseWindowCapturing =
            ScreenCaptureKitComputerUseCapture(),
        elementResolver: @escaping ElementResolver = Self.resolve,
        elementPerformer: @escaping ElementPerformer = Self.performPress
    ) {
        self.captureSource = captureSource
        self.elementResolver = elementResolver
        self.elementPerformer = elementPerformer
    }

    func emit(
        _ payload: WorkflowActionPayload,
        verifiedAgainst observation: WorkflowObservation
    ) async throws {
        try Task.checkCancellation()
        let target = observation.target
        let binding = try await MainActor.run {
            try elementResolver(payload, target)
        }

        guard let windowID = CGWindowID(target.windowIdentifier), windowID != 0,
              let processLaunchDate = target.processLaunchDate
        else {
            throw MacOSComputerUseActuationError.targetUnavailable
        }
        let captured = try await captureSource.capture(
            ComputerUseWindowSelection(
                bundleIdentifier: target.bundleIdentifier,
                processIdentifier: target.processIdentifier,
                processLaunchDate: processLaunchDate,
                windowID: windowID
            )
        )
        try Task.checkCancellation()
        guard captured.isStructurallyValid,
              MacOSComputerUseWindowIdentity.targetsMatch(captured.target, target)
        else {
            throw MacOSComputerUseActuationError.targetChanged
        }
        let finalRevision = MacOSComputerUseObserver.contentRevision(
            target: captured.target,
            pngData: captured.artifact.pngData
        )
        guard finalRevision == observation.contentRevision else {
            throw MacOSComputerUseActuationError.staleObservation
        }

        try await MainActor.run {
            try Task.checkCancellation()
            let currentBinding = try elementResolver(payload, target)
            guard binding.representsSameElement(as: currentBinding) else {
                throw MacOSComputerUseActuationError.elementChanged
            }
            try Task.checkCancellation()
            try elementPerformer(currentBinding)
        }
    }

    @MainActor
    private static func resolve(
        _ payload: WorkflowActionPayload,
        _ expected: WorkflowInteractionTarget
    ) throws -> ComputerUseElementBinding {
        guard case .click(let normalizedX, let normalizedY) = payload else {
            throw MacOSComputerUseActuationError.invalidAction
        }
        let current = try CGWindowComputerUseTargetProbe
            .currentTargetSynchronously(for: expected)
        guard MacOSComputerUseWindowIdentity.targetsMatch(current, expected) else {
            throw MacOSComputerUseActuationError.targetChanged
        }
        let point = try clickPoint(
            normalizedX: normalizedX,
            normalizedY: normalizedY,
            in: current.windowFrame
        )
        guard MacOSComputerUseWindowIdentity.topmostWindowIdentifier(at: point)
                == expected.windowIdentifier
        else {
            throw MacOSComputerUseActuationError.targetOccluded
        }
        try Task.checkCancellation()

        let application = AXUIElementCreateApplication(expected.processIdentifier)
        var focusedValue: CFTypeRef?
        guard AXUIElementCopyAttributeValue(
            application,
            kAXFocusedWindowAttribute as CFString,
            &focusedValue
        ) == .success,
            let focusedValue,
            CFGetTypeID(focusedValue) == AXUIElementGetTypeID()
        else {
            throw MacOSComputerUseActuationError.targetNotFrontmost
        }
        let focusedWindow = unsafeDowncast(focusedValue, to: AXUIElement.self)

        var element: AXUIElement?
        guard AXUIElementCopyElementAtPosition(
            application,
            Float(point.x),
            Float(point.y),
            &element
        ) == .success,
            let element,
            elementBelongsToWindow(element, focusedWindow: focusedWindow)
        else {
            throw MacOSComputerUseActuationError.elementUnavailable
        }
        guard elementSupportsPress(element) else {
            throw MacOSComputerUseActuationError.elementActionUnsupported
        }
        var elementPID: pid_t = 0
        guard AXUIElementGetPid(element, &elementPID) == .success,
              elementPID == expected.processIdentifier
        else {
            throw MacOSComputerUseActuationError.elementUnavailable
        }
        let fingerprint = try elementFingerprint(element)
        return ComputerUseElementBinding(
            fingerprint: fingerprint,
            element: element
        )
    }

    @MainActor
    private static func performPress(
        _ binding: ComputerUseElementBinding
    ) throws {
        guard let element = binding.element else {
            throw MacOSComputerUseActuationError.elementUnavailable
        }
        guard AXUIElementPerformAction(
            element,
            kAXPressAction as CFString
        ) == .success
        else {
            throw MacOSComputerUseActuationError.elementActionFailed
        }
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
    private static func elementBelongsToWindow(
        _ element: AXUIElement,
        focusedWindow: AXUIElement
    ) -> Bool {
        var current = element
        for _ in 0 ..< 64 {
            if CFEqual(current, focusedWindow) { return true }
            var parentValue: CFTypeRef?
            guard AXUIElementCopyAttributeValue(
                current,
                kAXParentAttribute as CFString,
                &parentValue
            ) == .success,
                let parentValue,
                CFGetTypeID(parentValue) == AXUIElementGetTypeID()
            else { return false }
            current = unsafeDowncast(parentValue, to: AXUIElement.self)
        }
        return false
    }

    @MainActor
    private static func elementSupportsPress(_ element: AXUIElement) -> Bool {
        var names: CFArray?
        guard AXUIElementCopyActionNames(element, &names) == .success,
              let actions = names as? [String]
        else { return false }
        return actions.contains(kAXPressAction as String)
    }

    @MainActor
    private static func elementFingerprint(
        _ element: AXUIElement
    ) throws -> ComputerUseElementFingerprint {
        guard let role = stringAttribute(kAXRoleAttribute as CFString, from: element),
              let frame = elementFrame(element)
        else {
            throw MacOSComputerUseActuationError.elementUnavailable
        }
        return ComputerUseElementFingerprint(
            role: role,
            subrole: stringAttribute(kAXSubroleAttribute as CFString, from: element),
            identifier: stringAttribute(kAXIdentifierAttribute as CFString, from: element),
            title: stringAttribute(kAXTitleAttribute as CFString, from: element),
            frame: frame
        )
    }

    @MainActor
    private static func stringAttribute(
        _ attribute: CFString,
        from element: AXUIElement
    ) -> String? {
        var value: CFTypeRef?
        guard AXUIElementCopyAttributeValue(element, attribute, &value) == .success else {
            return nil
        }
        return value as? String
    }

    @MainActor
    private static func elementFrame(
        _ element: AXUIElement
    ) -> WorkflowWindowFrame? {
        var positionValue: CFTypeRef?
        var sizeValue: CFTypeRef?
        guard AXUIElementCopyAttributeValue(
            element,
            kAXPositionAttribute as CFString,
            &positionValue
        ) == .success,
            AXUIElementCopyAttributeValue(
                element,
                kAXSizeAttribute as CFString,
                &sizeValue
            ) == .success,
            let positionValue,
            let sizeValue,
            CFGetTypeID(positionValue) == AXValueGetTypeID(),
            CFGetTypeID(sizeValue) == AXValueGetTypeID()
        else { return nil }
        var origin = CGPoint.zero
        var size = CGSize.zero
        guard AXValueGetValue(
            unsafeDowncast(positionValue, to: AXValue.self),
            .cgPoint,
            &origin
        ),
            AXValueGetValue(
                unsafeDowncast(sizeValue, to: AXValue.self),
                .cgSize,
                &size
            )
        else { return nil }
        let frame = WorkflowWindowFrame(
            x: origin.x,
            y: origin.y,
            width: size.width,
            height: size.height
        )
        return frame.isStructurallyValid ? frame : nil
    }
}
