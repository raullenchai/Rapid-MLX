import Foundation
import Testing
@testable import Rapid

@Suite("macOS Computer Use observation")
struct MacOSComputerUseObservationTests {
    private let selection = ComputerUseWindowSelection(
        bundleIdentifier: "com.example.Editor",
        processIdentifier: 123,
        windowID: 42
    )

    @Test("Observation requires both contextual grants before capture")
    func permissionGate() async {
        let capture = CaptureStub(result: Self.captureResult())
        let vault = ComputerUseObservationVault()
        let observer = MacOSComputerUseObserver(
            selections: ["draft": selection],
            vault: vault,
            captureSource: capture,
            permissionReader: {
                MacAutomationPermissionSnapshot(
                    screenRecording: true,
                    accessibility: false
                )
            }
        )

        do {
            _ = try await observer.observe(for: Self.step())
            Issue.record("Expected permission failure")
        } catch let error as MacOSComputerUseObservationError {
            #expect(error == .permissionMissing([.accessibility]))
        } catch {
            Issue.record("Unexpected error: \(error)")
        }
        #expect(await capture.callCount == 0)
    }

    @Test("An exact selected-window capture is metadata-only and retrievable ephemerally")
    func exactWindowCapture() async throws {
        let result = Self.captureResult()
        let capture = CaptureStub(result: result)
        let vault = ComputerUseObservationVault()
        let observer = MacOSComputerUseObserver(
            selections: ["draft": selection],
            vault: vault,
            captureSource: capture,
            permissionReader: Self.granted
        )

        let first = try await observer.observe(for: Self.step())
        let second = try await observer.observe(for: Self.step())

        #expect(first.id != second.id)
        #expect(first.target == result.target)
        #expect(first.contentRevision == second.contentRevision)
        #expect(first.contentRevision.count == 64)
        #expect(await vault.artifact(for: first.id) == result.artifact)
        #expect(await capture.callCount == 2)
    }

    @Test("The final content probe hashes one exact selected-window capture")
    func finalContentProbe() async throws {
        let result = Self.captureResult()
        let capture = CaptureStub(result: result)
        let probe = ScreenCaptureKitComputerUseContentProbe(captureSource: capture)

        let observation = try await probe.currentObservation(for: result.target)

        #expect(observation.target == result.target)
        #expect(
            observation.contentRevision
                == MacOSComputerUseObserver.contentRevision(
                    target: result.target,
                    pngData: result.artifact.pngData
                )
        )
        #expect(await capture.callCount == 1)
    }

    @Test("A recycled window ID from a different process is rejected")
    func recycledWindowIDFailsClosed() async {
        let original = Self.captureResult()
        let recycled = ComputerUseCapturedWindow(
            target: WorkflowInteractionTarget(
                bundleIdentifier: original.target.bundleIdentifier,
                processIdentifier: 124,
                windowIdentifier: original.target.windowIdentifier,
                windowFrame: original.target.windowFrame
            ),
            artifact: original.artifact
        )
        let capture = CaptureStub(result: recycled)
        let observer = MacOSComputerUseObserver(
            selections: ["draft": selection],
            vault: ComputerUseObservationVault(),
            captureSource: capture,
            permissionReader: Self.granted
        )

        await #expect(throws: MacOSComputerUseObservationError.invalidCapture) {
            _ = try await observer.observe(for: Self.step())
        }
    }

    @Test("Unknown steps cannot widen observation to another window")
    func unknownStepFailsClosed() async {
        let capture = CaptureStub(result: Self.captureResult())
        let observer = MacOSComputerUseObserver(
            selections: [:],
            vault: ComputerUseObservationVault(),
            captureSource: capture,
            permissionReader: Self.granted
        )
        do {
            _ = try await observer.observe(for: Self.step())
            Issue.record("Expected target configuration failure")
        } catch let error as MacOSComputerUseObservationError {
            #expect(error == .targetNotConfigured)
        } catch {
            Issue.record("Unexpected error: \(error)")
        }
        #expect(await capture.callCount == 0)
    }

    @Test("The screenshot vault evicts old pixels and clears all run data")
    func vaultIsBounded() async {
        let vault = ComputerUseObservationVault(maximumArtifacts: 2)
        let one = UUID()
        let two = UUID()
        let three = UUID()
        let artifact = Self.captureResult().artifact

        await vault.store(artifact, for: one)
        await vault.store(artifact, for: two)
        await vault.store(artifact, for: three)
        #expect(await vault.artifact(for: one) == nil)
        #expect(await vault.artifact(for: two) != nil)
        #expect(await vault.artifact(for: three) != nil)

        await vault.removeAll()
        #expect(await vault.artifact(for: two) == nil)
        #expect(await vault.artifact(for: three) == nil)
    }

    private static func granted() -> MacAutomationPermissionSnapshot {
        MacAutomationPermissionSnapshot(
            screenRecording: true,
            accessibility: true
        )
    }

    private static func step() -> LocalWorkflowStep {
        LocalWorkflowStep(
            id: "draft",
            title: "Draft locally",
            instruction: "Focus the document",
            successCriteria: "The document is focused"
        )
    }

    private static func captureResult() -> ComputerUseCapturedWindow {
        ComputerUseCapturedWindow(
            target: WorkflowInteractionTarget(
                bundleIdentifier: "com.example.Editor",
                processIdentifier: 123,
                windowIdentifier: "42",
                windowFrame: WorkflowWindowFrame(
                    x: 10,
                    y: 20,
                    width: 800,
                    height: 600
                )
            ),
            artifact: ComputerUseObservationArtifact(
                pngData: Data([1, 2, 3, 4]),
                pixelWidth: 800,
                pixelHeight: 600
            )
        )
    }
}

private actor CaptureStub: ComputerUseWindowCapturing {
    private let result: ComputerUseCapturedWindow
    private(set) var callCount = 0

    init(result: ComputerUseCapturedWindow) {
        self.result = result
    }

    func capture(_ selection: ComputerUseWindowSelection) async throws
        -> ComputerUseCapturedWindow
    {
        callCount += 1
        return result
    }
}
