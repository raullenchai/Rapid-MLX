import CoreGraphics
import Foundation
import Testing
@testable import Rapid

@Suite("macOS Computer Use actuation")
struct MacOSComputerUseActuationTests {
    @Test("A fresh equivalent observation may emit the grounded action")
    func freshEquivalentObservationEmits() async throws {
        let groundedObservation = Self.observation()
        let currentObservation = WorkflowObservation(
            target: groundedObservation.target,
            contentRevision: groundedObservation.contentRevision
        )
        #expect(groundedObservation.id != currentObservation.id)
        let probe = TargetProbe(result: .success(currentObservation.target))
        let emitter = InputEmitter()
        let actuator = MacOSComputerUseActuator(
            targetProbe: probe,
            inputEmitter: emitter,
            permissionReader: { .init(screenRecording: true, accessibility: true) }
        )

        try await actuator.perform(
            Self.action(for: groundedObservation),
            groundedAgainst: groundedObservation,
            currentObservation: currentObservation
        )

        let emissions = await emitter.emissions
        #expect(emissions == [.click(normalizedX: 0.25, normalizedY: 0.75)])
    }

    @Test("A forged action binding is rejected before desktop access")
    func mismatchedGroundingIDIsRejected() async {
        let groundingObservation = Self.observation()
        let probe = TargetProbe(result: .success(groundingObservation.target))
        let emitter = InputEmitter()
        let actuator = MacOSComputerUseActuator(
            targetProbe: probe,
            inputEmitter: emitter,
            permissionReader: { .init(screenRecording: true, accessibility: true) }
        )
        let forged = GroundedWorkflowAction(
            observationID: UUID(),
            payload: .click(normalizedX: 0.25, normalizedY: 0.75),
            source: .visualGrounding,
            safeSummary: "Forged",
            risk: .readOnly
        )

        await #expect(throws: MacOSComputerUseActuationError.staleObservation) {
            try await actuator.perform(
                forged,
                groundedAgainst: groundingObservation,
                currentObservation: groundingObservation
            )
        }
        #expect(await probe.callCount == 0)
        #expect(await emitter.emissions.isEmpty)
    }

    @Test("Changed content cannot be substituted at the actuation boundary")
    func changedGroundingStateIsRejected() async {
        let groundingObservation = Self.observation()
        let currentObservation = WorkflowObservation(
            target: groundingObservation.target,
            contentRevision: "different-revision"
        )
        let probe = TargetProbe(result: .success(groundingObservation.target))
        let emitter = InputEmitter()
        let actuator = MacOSComputerUseActuator(
            targetProbe: probe,
            inputEmitter: emitter,
            permissionReader: { .init(screenRecording: true, accessibility: true) }
        )

        await #expect(throws: MacOSComputerUseActuationError.staleObservation) {
            try await actuator.perform(
                Self.action(for: groundingObservation),
                groundedAgainst: groundingObservation,
                currentObservation: currentObservation
            )
        }
        #expect(await probe.callCount == 0)
        #expect(await emitter.emissions.isEmpty)
    }

    @Test("An invalid current observation is rejected before probing the desktop")
    func invalidCurrentObservationIsRejected() async {
        let groundedObservation = Self.observation()
        let invalidCurrent = WorkflowObservation(
            target: WorkflowInteractionTarget(
                bundleIdentifier: groundedObservation.target.bundleIdentifier,
                processIdentifier: groundedObservation.target.processIdentifier,
                processLaunchDate: groundedObservation.target.processLaunchDate,
                windowIdentifier: groundedObservation.target.windowIdentifier,
                windowFrame: .init(x: 0, y: 0, width: 0, height: 600)
            ),
            contentRevision: groundedObservation.contentRevision
        )
        let probe = TargetProbe(result: .success(groundedObservation.target))
        let emitter = InputEmitter()
        let actuator = MacOSComputerUseActuator(
            targetProbe: probe,
            inputEmitter: emitter,
            permissionReader: { .init(screenRecording: true, accessibility: true) }
        )
        await #expect(throws: MacOSComputerUseActuationError.staleObservation) {
            try await actuator.perform(
                Self.action(for: groundedObservation),
                groundedAgainst: groundedObservation,
                currentObservation: invalidCurrent
            )
        }
        #expect(await probe.callCount == 0)
        #expect(await emitter.emissions.isEmpty)
    }

    @Test("Revoked permissions stop the action before desktop access")
    func revokedPermissionStopsAction() async {
        let observation = Self.observation()
        let probe = TargetProbe(result: .success(observation.target))
        let emitter = InputEmitter()
        let actuator = MacOSComputerUseActuator(
            targetProbe: probe,
            inputEmitter: emitter,
            permissionReader: { .init(screenRecording: true, accessibility: false) }
        )

        await #expect(
            throws: MacOSComputerUseActuationError.permissionMissing([.accessibility])
        ) {
            try await actuator.perform(
                Self.action(for: observation),
                groundedAgainst: observation,
                currentObservation: observation
            )
        }
        #expect(await probe.callCount == 0)
        #expect(await emitter.emissions.isEmpty)
    }

    @Test("Window movement invalidates coordinates instead of clicking")
    func movedWindowStopsAction() async {
        let observation = Self.observation()
        let moved = WorkflowInteractionTarget(
            bundleIdentifier: observation.target.bundleIdentifier,
            processIdentifier: observation.target.processIdentifier,
            processLaunchDate: observation.target.processLaunchDate,
            windowIdentifier: observation.target.windowIdentifier,
            windowFrame: .init(x: 101, y: 200, width: 800, height: 600)
        )
        let probe = TargetProbe(result: .success(moved))
        let emitter = InputEmitter()
        let actuator = MacOSComputerUseActuator(
            targetProbe: probe,
            inputEmitter: emitter,
            permissionReader: { .init(screenRecording: true, accessibility: true) }
        )

        await #expect(throws: MacOSComputerUseActuationError.targetChanged) {
            try await actuator.perform(
                Self.action(for: observation),
                groundedAgainst: observation,
                currentObservation: observation
            )
        }
        #expect(await emitter.emissions.isEmpty)
    }

    @Test("Invalid payloads fail closed before reading permissions")
    func invalidPayloadStopsAction() async {
        let observation = Self.observation()
        let probe = TargetProbe(result: .success(observation.target))
        let emitter = InputEmitter()
        let actuator = MacOSComputerUseActuator(
            targetProbe: probe,
            inputEmitter: emitter,
            permissionReader: { .init(screenRecording: true, accessibility: true) }
        )
        let invalid = GroundedWorkflowAction(
            observationID: observation.id,
            payload: .click(normalizedX: .nan, normalizedY: 0.5),
            source: .visualGrounding,
            safeSummary: "Invalid",
            risk: .readOnly
        )

        await #expect(throws: MacOSComputerUseActuationError.invalidAction) {
            try await actuator.perform(
                invalid,
                groundedAgainst: observation,
                currentObservation: observation
            )
        }
        #expect(await probe.callCount == 0)
        #expect(await emitter.emissions.isEmpty)
    }

    @Test("Window-edge coordinates are rejected before target probing", arguments: [
        WorkflowActionPayload.click(normalizedX: 0, normalizedY: 0.5),
        WorkflowActionPayload.click(normalizedX: 1, normalizedY: 0.5),
        WorkflowActionPayload.click(normalizedX: 0.5, normalizedY: 0),
        WorkflowActionPayload.click(normalizedX: 0.5, normalizedY: 1),
    ])
    func edgeCoordinatesStopAction(payload: WorkflowActionPayload) async {
        let observation = Self.observation()
        let probe = TargetProbe(result: .success(observation.target))
        let emitter = InputEmitter()
        let actuator = MacOSComputerUseActuator(
            targetProbe: probe,
            inputEmitter: emitter,
            permissionReader: { .init(screenRecording: true, accessibility: true) }
        )
        let action = GroundedWorkflowAction(
            observationID: observation.id,
            payload: payload,
            source: .visualGrounding,
            safeSummary: "Edge click",
            risk: .readOnly
        )

        await #expect(throws: MacOSComputerUseActuationError.invalidAction) {
            try await actuator.perform(
                action,
                groundedAgainst: observation,
                currentObservation: observation
            )
        }
        #expect(await probe.callCount == 0)
        #expect(await emitter.emissions.isEmpty)
    }

    @Test("An unchanged capture presses the same resolved AX element")
    @MainActor
    func unchangedCapturePressesBoundElement() async throws {
        let fixture = Self.captureFixture()
        let recorder = ElementBoundaryRecorder()
        let emitter = AXComputerUseInputEmitter(
            captureSource: ActuationCaptureStub(result: fixture.capture),
            elementResolver: { payload, target in
                recorder.calls.append(.init(payload: payload, target: target))
                return recorder.binding
            },
            elementPerformer: { binding in
                recorder.performed.append(binding)
            },
            permissionReader: Self.granted
        )

        try await emitter.emit(
            .click(normalizedX: 0.25, normalizedY: 0.75),
            verifiedAgainst: fixture.observation
        )

        #expect(recorder.calls.count == 2)
        #expect(recorder.calls.allSatisfy { $0.target == fixture.observation.target })
        #expect(recorder.performed.count == 1)
        #expect(recorder.performed[0] === recorder.binding)
    }

    @Test("Changed pixels prevent the final AX element boundary")
    @MainActor
    func changedPixelsPreventPress() async {
        let fixture = Self.captureFixture()
        let changedCapture = ComputerUseCapturedWindow(
            target: fixture.capture.target,
            artifact: .init(
                pngData: Data([9, 9, 9]),
                pixelWidth: fixture.capture.artifact.pixelWidth,
                pixelHeight: fixture.capture.artifact.pixelHeight
            )
        )
        let recorder = ElementBoundaryRecorder()
        let emitter = AXComputerUseInputEmitter(
            captureSource: ActuationCaptureStub(result: changedCapture),
            elementResolver: { payload, target in
                recorder.calls.append(.init(payload: payload, target: target))
                return recorder.binding
            },
            elementPerformer: { binding in
                recorder.performed.append(binding)
            },
            permissionReader: Self.granted
        )

        await #expect(throws: MacOSComputerUseActuationError.staleObservation) {
            try await emitter.emit(
                .click(normalizedX: 0.25, normalizedY: 0.75),
                verifiedAgainst: fixture.observation
            )
        }
        #expect(recorder.calls.count == 1)
        #expect(recorder.performed.isEmpty)
    }

    @Test("A changed AX element is rejected at the press boundary")
    @MainActor
    func changedElementPreventsPress() async {
        let fixture = Self.captureFixture()
        let recorder = ElementBoundaryRecorder()
        let replacement = ComputerUseElementBinding(
            fingerprint: recorder.binding.fingerprint
        )
        let emitter = AXComputerUseInputEmitter(
            captureSource: ActuationCaptureStub(result: fixture.capture),
            elementResolver: { payload, target in
                recorder.calls.append(.init(payload: payload, target: target))
                return recorder.calls.count == 1 ? recorder.binding : replacement
            },
            elementPerformer: { binding in
                recorder.performed.append(binding)
            },
            permissionReader: Self.granted
        )

        await #expect(throws: MacOSComputerUseActuationError.elementChanged) {
            try await emitter.emit(
                .click(normalizedX: 0.25, normalizedY: 0.75),
                verifiedAgainst: fixture.observation
            )
        }
        #expect(recorder.calls.count == 2)
        #expect(recorder.performed.isEmpty)
    }

    @Test("A focus change at the final resolver prevents AXPress")
    @MainActor
    func finalFocusChangePreventsPress() async {
        let fixture = Self.captureFixture()
        let recorder = ElementBoundaryRecorder()
        let emitter = AXComputerUseInputEmitter(
            captureSource: ActuationCaptureStub(result: fixture.capture),
            elementResolver: { payload, target in
                recorder.calls.append(.init(payload: payload, target: target))
                if recorder.calls.count == 2 {
                    throw MacOSComputerUseActuationError.targetNotFrontmost
                }
                return recorder.binding
            },
            elementPerformer: { binding in
                recorder.performed.append(binding)
            },
            permissionReader: Self.granted
        )

        await #expect(throws: MacOSComputerUseActuationError.targetNotFrontmost) {
            try await emitter.emit(
                .click(normalizedX: 0.25, normalizedY: 0.75),
                verifiedAgainst: fixture.observation
            )
        }
        #expect(recorder.calls.count == 2)
        #expect(recorder.performed.isEmpty)
    }

    @Test("Permission revoked at the final boundary prevents AXPress")
    @MainActor
    func finalPermissionRevocationPreventsPress() async {
        let fixture = Self.captureFixture()
        let recorder = ElementBoundaryRecorder()
        let emitter = AXComputerUseInputEmitter(
            captureSource: ActuationCaptureStub(result: fixture.capture),
            elementResolver: { payload, target in
                recorder.calls.append(.init(payload: payload, target: target))
                return recorder.binding
            },
            elementPerformer: { binding in
                recorder.performed.append(binding)
            },
            permissionReader: {
                .init(screenRecording: true, accessibility: false)
            }
        )

        await #expect(
            throws: MacOSComputerUseActuationError.permissionMissing([.accessibility])
        ) {
            try await emitter.emit(
                .click(normalizedX: 0.25, normalizedY: 0.75),
                verifiedAgainst: fixture.observation
            )
        }
        #expect(recorder.calls.count == 2)
        #expect(recorder.performed.isEmpty)
    }

    @Test("Window-unbound keyboard payloads fail before desktop access", arguments: [
        WorkflowActionPayload.typeText("draft"),
        WorkflowActionPayload.keyPress(key: "tab", modifiers: []),
        WorkflowActionPayload.keyPress(
            key: "delete",
            modifiers: ["command", "option"]
        ),
    ])
    func keyboardPayloadsFailClosed(payload: WorkflowActionPayload) async {
        let observation = Self.observation()
        let probe = TargetProbe(result: .success(observation.target))
        let emitter = InputEmitter()
        let actuator = MacOSComputerUseActuator(
            targetProbe: probe,
            inputEmitter: emitter,
            permissionReader: { .init(screenRecording: true, accessibility: true) }
        )
        let action = GroundedWorkflowAction(
            observationID: observation.id,
            payload: payload,
            source: .visualGrounding,
            safeSummary: "Keyboard input",
            risk: .localChange
        )

        await #expect(throws: MacOSComputerUseActuationError.invalidAction) {
            try await actuator.perform(
                action,
                groundedAgainst: observation,
                currentObservation: observation
            )
        }
        #expect(await probe.callCount == 0)
        #expect(await emitter.emissions.isEmpty)
    }

    @Test("Click coordinates use the tolerated live frame")
    func clickUsesLiveFrame() throws {
        let liveFrame = WorkflowWindowFrame(
            x: 100.5,
            y: 200.5,
            width: 799.5,
            height: 599.5
        )

        let point = try AXComputerUseInputEmitter.clickPoint(
            normalizedX: 0.25,
            normalizedY: 0.75,
            in: liveFrame
        )

        #expect(point.x == 300.375)
        #expect(point.y == 650.125)
    }

    @Test("A normalized coordinate that rounds onto the live edge is rejected")
    func roundedEdgeIsRejected() {
        let frame = WorkflowWindowFrame(x: 100, y: 200, width: 800, height: 600)

        #expect(throws: MacOSComputerUseActuationError.invalidAction) {
            try AXComputerUseInputEmitter.clickPoint(
                normalizedX: Double.leastNonzeroMagnitude,
                normalizedY: 0.5,
                in: frame
            )
        }
    }

    private static func observation() -> WorkflowObservation {
        WorkflowObservation(
            target: WorkflowInteractionTarget(
                bundleIdentifier: "com.example.Editor",
                processIdentifier: 42,
                processLaunchDate: Date(timeIntervalSinceReferenceDate: 1_000),
                windowIdentifier: "7",
                windowFrame: .init(x: 100, y: 200, width: 800, height: 600)
            ),
            contentRevision: "revision"
        )
    }

    private static func action(for observation: WorkflowObservation) -> GroundedWorkflowAction {
        GroundedWorkflowAction(
            observationID: observation.id,
            payload: .click(normalizedX: 0.25, normalizedY: 0.75),
            source: .visualGrounding,
            safeSummary: "Click draft",
            risk: .localChange
        )
    }

    private static func granted() -> MacAutomationPermissionSnapshot {
        .init(screenRecording: true, accessibility: true)
    }

    private static func captureFixture() -> (
        observation: WorkflowObservation,
        capture: ComputerUseCapturedWindow
    ) {
        let target = observation().target
        let artifact = ComputerUseObservationArtifact(
            pngData: Data([1, 2, 3]),
            pixelWidth: 800,
            pixelHeight: 600
        )
        let capture = ComputerUseCapturedWindow(target: target, artifact: artifact)
        return (
            WorkflowObservation(
                target: target,
                contentRevision: MacOSComputerUseObserver.contentRevision(
                    target: target,
                    pngData: artifact.pngData
                )
            ),
            capture
        )
    }
}

@MainActor
private struct ElementBoundaryCall {
    let payload: WorkflowActionPayload
    let target: WorkflowInteractionTarget
}

@MainActor
private final class ElementBoundaryRecorder {
    let binding = ComputerUseElementBinding(
        fingerprint: ComputerUseElementFingerprint(
            role: "AXButton",
            subrole: nil,
            identifier: "submit",
            title: "Submit",
            frame: .init(x: 200, y: 300, width: 80, height: 30)
        )
    )
    var calls: [ElementBoundaryCall] = []
    var performed: [ComputerUseElementBinding] = []
}

private actor TargetProbe: ComputerUseTargetProbing {
    private let result: Result<WorkflowInteractionTarget, Error>
    private(set) var callCount = 0

    init(result: Result<WorkflowInteractionTarget, Error>) {
        self.result = result
    }

    func currentTarget(for _: WorkflowInteractionTarget) async throws
        -> WorkflowInteractionTarget
    {
        callCount += 1
        return try result.get()
    }
}

private actor InputEmitter: ComputerUseInputEmitting {
    private(set) var emissions: [WorkflowActionPayload] = []

    func emit(
        _ payload: WorkflowActionPayload,
        verifiedAgainst _: WorkflowObservation
    ) async throws {
        emissions.append(payload)
    }
}

private actor ActuationCaptureStub: ComputerUseWindowCapturing {
    private let result: ComputerUseCapturedWindow

    init(result: ComputerUseCapturedWindow) {
        self.result = result
    }

    func capture(_: ComputerUseWindowSelection) async throws
        -> ComputerUseCapturedWindow
    {
        result
    }
}
