import Foundation
import Testing
@testable import Rapid

@Suite("macOS Computer Use actuation")
struct MacOSComputerUseActuationTests {
    @Test("The exact current observation may emit one bounded action")
    func exactObservationEmits() async throws {
        let observation = Self.observation()
        let probe = TargetProbe(result: .success(observation.target))
        let emitter = InputEmitter()
        let actuator = MacOSComputerUseActuator(
            targetProbe: probe,
            inputEmitter: emitter,
            permissionReader: { .init(screenRecording: true, accessibility: true) }
        )

        try await actuator.perform(Self.action(for: observation), against: observation)

        let emissions = await emitter.emissions
        #expect(emissions == [.click(normalizedX: 0.25, normalizedY: 0.75)])
    }

    @Test("Stale model output is rejected before probing the live desktop")
    func staleObservationIsRejected() async {
        let observation = Self.observation()
        let probe = TargetProbe(result: .success(observation.target))
        let emitter = InputEmitter()
        let actuator = MacOSComputerUseActuator(
            targetProbe: probe,
            inputEmitter: emitter,
            permissionReader: { .init(screenRecording: true, accessibility: true) }
        )
        let stale = GroundedWorkflowAction(
            observationID: UUID(),
            payload: .click(normalizedX: 0.25, normalizedY: 0.75),
            source: .visualGrounding,
            safeSummary: "Click draft",
            risk: .localChange
        )

        await #expect(throws: MacOSComputerUseActuationError.staleObservation) {
            try await actuator.perform(stale, against: observation)
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
            try await actuator.perform(Self.action(for: observation), against: observation)
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
            try await actuator.perform(Self.action(for: observation), against: observation)
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
            try await actuator.perform(invalid, against: observation)
        }
        #expect(await probe.callCount == 0)
        #expect(await emitter.emissions.isEmpty)
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
