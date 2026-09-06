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

    @Test("The production emitter rechecks target drift at its event boundary")
    @MainActor
    func emitterRejectsLastMomentDrift() async {
        let expected = Self.observation().target
        let moved = WorkflowInteractionTarget(
            bundleIdentifier: expected.bundleIdentifier,
            processIdentifier: expected.processIdentifier,
            windowIdentifier: expected.windowIdentifier,
            windowFrame: .init(x: 104, y: 200, width: 800, height: 600)
        )
        let emitter = CGEventComputerUseInputEmitter(targetReader: { _ in moved })

        await #expect(throws: MacOSComputerUseActuationError.targetChanged) {
            try await emitter.emit(
                .click(normalizedX: 0.25, normalizedY: 0.75),
                in: expected
            )
        }
    }

    @Test("Cancellation during the final target probe prevents the event")
    @MainActor
    func emitterRechecksCancellationAfterProbe() async {
        let expected = Self.observation().target
        let emitter = CGEventComputerUseInputEmitter(
            targetReader: { $0 },
            cancellationCheck: { throw CancellationError() },
            windowAtPointReader: { _ in expected.windowIdentifier }
        )

        await #expect(throws: CancellationError.self) {
            try await emitter.emit(
                .click(normalizedX: 0.25, normalizedY: 0.75),
                in: expected
            )
        }
    }

    @Test("An overlay at the click point prevents global input")
    @MainActor
    func occludingWindowStopsClick() async {
        let expected = Self.observation().target
        let emitter = CGEventComputerUseInputEmitter(
            targetReader: { $0 },
            windowAtPointReader: { _ in "999" }
        )

        await #expect(throws: MacOSComputerUseActuationError.targetOccluded) {
            try await emitter.emit(
                .click(normalizedX: 0.25, normalizedY: 0.75),
                in: expected
            )
        }
    }

    @Test("Events are delivered only to the selected process")
    @MainActor
    func inputDeliveryIsProcessBound() async throws {
        let expected = Self.observation().target
        let recorder = PostedEventRecorder()
        let emitter = CGEventComputerUseInputEmitter(
            targetReader: { $0 },
            windowAtPointReader: { _ in expected.windowIdentifier },
            eventPoster: { _, processIdentifier in
                recorder.processIdentifiers.append(processIdentifier)
            }
        )

        try await emitter.emit(
            .click(normalizedX: 0.25, normalizedY: 0.75),
            in: expected
        )

        #expect(recorder.processIdentifiers == [42, 42])
    }

    @Test("Destructive modifier combinations are rejected before input")
    @MainActor
    func destructiveKeyChordIsRejected() async {
        let expected = Self.observation().target
        let emitter = CGEventComputerUseInputEmitter(targetReader: { $0 })

        await #expect(throws: MacOSComputerUseActuationError.unsupportedKey) {
            try await emitter.emit(
                .keyPress(key: "delete", modifiers: ["command", "option"]),
                in: expected
            )
        }
    }

    @Test("Duplicate modifiers cannot bypass the complete-chord allowlist")
    @MainActor
    func duplicateModifiersAreRejected() async {
        let expected = Self.observation().target
        let emitter = CGEventComputerUseInputEmitter(targetReader: { $0 })

        await #expect(throws: MacOSComputerUseActuationError.unsupportedKey) {
            try await emitter.emit(
                .keyPress(key: "tab", modifiers: ["shift", "shift"]),
                in: expected
            )
        }
    }

    @Test("Click coordinates use the tolerated live frame")
    func clickUsesLiveFrame() throws {
        let liveFrame = WorkflowWindowFrame(
            x: 100.5,
            y: 200.5,
            width: 799.5,
            height: 599.5
        )

        let point = try CGEventComputerUseInputEmitter.clickPoint(
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
            try CGEventComputerUseInputEmitter.clickPoint(
                normalizedX: Double.leastNonzeroMagnitude,
                normalizedY: 0.5,
                in: frame
            )
        }
    }

    @Test("Unicode chunks never split a surrogate pair")
    func unicodeChunksPreserveScalars() {
        let text = String(repeating: "a", count: 1_023) + "😀" + "b"
        let chunks = CGEventComputerUseInputEmitter.unicodeChunks(for: text)

        #expect(chunks.map(\.count) == [1_023, 3])
        #expect(chunks.allSatisfy { $0.count <= 1_024 })
        #expect(String(decoding: chunks.flatMap { $0 }, as: UTF16.self) == text)
    }

    private static func observation() -> WorkflowObservation {
        WorkflowObservation(
            target: WorkflowInteractionTarget(
                bundleIdentifier: "com.example.Editor",
                processIdentifier: 42,
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
}

@MainActor
private final class PostedEventRecorder {
    var processIdentifiers: [pid_t] = []
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
        in _: WorkflowInteractionTarget
    ) async throws {
        emissions.append(payload)
    }
}
