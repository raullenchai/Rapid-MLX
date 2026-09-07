import Foundation
import Testing
@testable import Rapid

@Suite("macOS Computer Use observation")
struct MacOSComputerUseObservationTests {
    private let selection = ComputerUseWindowSelection(
        bundleIdentifier: "com.example.Editor",
        processIdentifier: 123,
        processLaunchDate: Date(timeIntervalSinceReferenceDate: 1_000),
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

    @Test("ScreenCaptureKit boundary resolves the exact selection before and after pixels")
    func screenCaptureBoundaryUsesExactSelection() async throws {
        let target = Self.captureResult().target
        let resolver = WindowResolverStub(
            results: [
                .success(.init(window: nil, target: target)),
                .success(.init(window: nil, target: target)),
            ]
        )
        let image = ImageCapturerStub(artifact: Self.captureResult().artifact)
        let capture = ScreenCaptureKitComputerUseCapture(
            windowResolver: { selection in try await resolver.resolve(selection) },
            imageCapturer: { resolved, size in
                await image.capture(resolved, size: size)
            }
        )

        let result = try await capture.capture(selection)

        #expect(result == Self.captureResult())
        #expect(await resolver.selections == [selection, selection])
        #expect(await image.callCount == 1)
        #expect(await image.requestedTarget == target)
        #expect(await image.requestedSize == CGSize(width: 800, height: 600))
    }

    @Test("A resolver cannot substitute another bundle, process, or window")
    func screenCaptureBoundaryRejectsSubstitution() async {
        let original = Self.captureResult().target
        let substitutions = [
            WorkflowInteractionTarget(
                bundleIdentifier: "com.example.Other",
                processIdentifier: original.processIdentifier,
                processLaunchDate: original.processLaunchDate,
                windowIdentifier: original.windowIdentifier,
                windowFrame: original.windowFrame
            ),
            WorkflowInteractionTarget(
                bundleIdentifier: original.bundleIdentifier,
                processIdentifier: 999,
                processLaunchDate: original.processLaunchDate,
                windowIdentifier: original.windowIdentifier,
                windowFrame: original.windowFrame
            ),
            WorkflowInteractionTarget(
                bundleIdentifier: original.bundleIdentifier,
                processIdentifier: original.processIdentifier,
                processLaunchDate: Date(timeIntervalSinceReferenceDate: 2_000),
                windowIdentifier: original.windowIdentifier,
                windowFrame: original.windowFrame
            ),
            WorkflowInteractionTarget(
                bundleIdentifier: original.bundleIdentifier,
                processIdentifier: original.processIdentifier,
                processLaunchDate: original.processLaunchDate,
                windowIdentifier: "99",
                windowFrame: original.windowFrame
            ),
        ]

        for substitution in substitutions {
            let resolver = WindowResolverStub(
                results: [.success(.init(window: nil, target: substitution))]
            )
            let image = ImageCapturerStub(artifact: Self.captureResult().artifact)
            let capture = ScreenCaptureKitComputerUseCapture(
                windowResolver: { selection in try await resolver.resolve(selection) },
                imageCapturer: { resolved, size in
                    await image.capture(resolved, size: size)
                }
            )

            await #expect(throws: MacOSComputerUseObservationError.invalidCapture) {
                _ = try await capture.capture(selection)
            }
            #expect(await image.callCount == 0)
        }
    }

    @Test("Window filtering rejects bundle, process, window, visibility, and focus mismatches")
    func windowFilteringFailsClosed() {
        let frame = CGRect(x: 10, y: 20, width: 800, height: 600)
        let foreground = ScreenCaptureKitComputerUseCapture.ForegroundRecord(
            bundleIdentifier: selection.bundleIdentifier,
            processIdentifier: selection.processIdentifier,
            processLaunchDate: selection.processLaunchDate,
            focusedFrame: frame
        )
        let valid = ScreenCaptureKitComputerUseCapture.WindowRecord(
            windowID: selection.windowID,
            bundleIdentifier: selection.bundleIdentifier,
            processIdentifier: selection.processIdentifier,
            frame: frame,
            isOnScreen: true
        )
        let invalidRecords = [
            ScreenCaptureKitComputerUseCapture.WindowRecord(
                windowID: 99,
                bundleIdentifier: valid.bundleIdentifier,
                processIdentifier: valid.processIdentifier,
                frame: frame,
                isOnScreen: true
            ),
            ScreenCaptureKitComputerUseCapture.WindowRecord(
                windowID: valid.windowID,
                bundleIdentifier: "com.example.Other",
                processIdentifier: valid.processIdentifier,
                frame: frame,
                isOnScreen: true
            ),
            ScreenCaptureKitComputerUseCapture.WindowRecord(
                windowID: valid.windowID,
                bundleIdentifier: valid.bundleIdentifier,
                processIdentifier: 999,
                frame: frame,
                isOnScreen: true
            ),
            ScreenCaptureKitComputerUseCapture.WindowRecord(
                windowID: valid.windowID,
                bundleIdentifier: valid.bundleIdentifier,
                processIdentifier: valid.processIdentifier,
                frame: frame,
                isOnScreen: false
            ),
        ]

        for record in invalidRecords {
            #expect(throws: MacOSComputerUseObservationError.targetUnavailable) {
                _ = try ScreenCaptureKitComputerUseCapture.validatedTarget(
                    selection,
                    foreground: foreground,
                    windows: [record]
                )
            }
        }
        #expect(throws: MacOSComputerUseObservationError.targetNotFrontmost) {
            _ = try ScreenCaptureKitComputerUseCapture.validatedTarget(
                selection,
                foreground: .init(
                    bundleIdentifier: selection.bundleIdentifier,
                    processIdentifier: selection.processIdentifier,
                    processLaunchDate: Date(timeIntervalSinceReferenceDate: 2_000),
                    focusedFrame: frame
                ),
                windows: [valid]
            )
        }
        #expect(throws: MacOSComputerUseObservationError.targetNotFrontmost) {
            _ = try ScreenCaptureKitComputerUseCapture.validatedTarget(
                selection,
                foreground: .init(
                    bundleIdentifier: selection.bundleIdentifier,
                    processIdentifier: selection.processIdentifier,
                    processLaunchDate: selection.processLaunchDate,
                    focusedFrame: CGRect(x: 0, y: 0, width: 10, height: 10)
                ),
                windows: [valid]
            )
        }
    }

    @Test("A recycled window ID from a different process is rejected")
    func recycledWindowIDFailsClosed() async {
        let original = Self.captureResult()
        let recycled = ComputerUseCapturedWindow(
            target: WorkflowInteractionTarget(
                bundleIdentifier: original.target.bundleIdentifier,
                processIdentifier: 124,
                processLaunchDate: original.target.processLaunchDate,
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

    @Test("A recycled PID and window ID from another app launch is rejected")
    func recycledProcessLaunchFailsClosed() async {
        let original = Self.captureResult()
        let recycled = ComputerUseCapturedWindow(
            target: WorkflowInteractionTarget(
                bundleIdentifier: original.target.bundleIdentifier,
                processIdentifier: original.target.processIdentifier,
                processLaunchDate: Date(timeIntervalSinceReferenceDate: 2_000),
                windowIdentifier: original.target.windowIdentifier,
                windowFrame: original.target.windowFrame
            ),
            artifact: original.artifact
        )
        let observer = MacOSComputerUseObserver(
            selections: ["draft": selection],
            vault: ComputerUseObservationVault(),
            captureSource: CaptureStub(result: recycled),
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
                processLaunchDate: Date(timeIntervalSinceReferenceDate: 1_000),
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

private actor WindowResolverStub {
    private var results: [Result<ScreenCaptureKitComputerUseCapture.ResolvedWindow, Error>]
    private(set) var selections: [ComputerUseWindowSelection] = []

    init(results: [Result<ScreenCaptureKitComputerUseCapture.ResolvedWindow, Error>]) {
        self.results = results
    }

    func resolve(_ selection: ComputerUseWindowSelection) throws
        -> ScreenCaptureKitComputerUseCapture.ResolvedWindow
    {
        selections.append(selection)
        guard !results.isEmpty else {
            throw MacOSComputerUseObservationError.targetUnavailable
        }
        return try results.removeFirst().get()
    }
}

private actor ImageCapturerStub {
    private let artifact: ComputerUseObservationArtifact
    private(set) var callCount = 0
    private(set) var requestedTarget: WorkflowInteractionTarget?
    private(set) var requestedSize: CGSize?

    init(artifact: ComputerUseObservationArtifact) {
        self.artifact = artifact
    }

    func capture(
        _ resolved: ScreenCaptureKitComputerUseCapture.ResolvedWindow,
        size: CGSize
    ) -> ComputerUseObservationArtifact {
        callCount += 1
        requestedTarget = resolved.target
        requestedSize = size
        return artifact
    }
}
