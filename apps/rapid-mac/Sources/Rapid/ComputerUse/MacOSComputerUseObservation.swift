import AppKit
import CryptoKit
import Foundation
import ScreenCaptureKit

/// A window the user explicitly allowed one Computer Use step to observe.
/// Window IDs are session-scoped and are never treated as durable selectors.
struct ComputerUseWindowSelection: Equatable, Sendable {
    let bundleIdentifier: String
    let windowID: CGWindowID

    var isStructurallyValid: Bool {
        !bundleIdentifier.isEmpty && windowID != 0
    }
}

/// Ephemeral pixels for a single ``WorkflowObservation``. This deliberately is
/// not Codable: screenshots may be given to the local grounder during a run,
/// but they must not enter workflow persistence or the audit ledger.
struct ComputerUseObservationArtifact: Equatable, Sendable {
    let pngData: Data
    let pixelWidth: Int
    let pixelHeight: Int

    var isStructurallyValid: Bool {
        !pngData.isEmpty && pixelWidth > 0 && pixelHeight > 0
    }
}

struct ComputerUseCapturedWindow: Equatable, Sendable {
    let target: WorkflowInteractionTarget
    let artifact: ComputerUseObservationArtifact

    var isStructurallyValid: Bool {
        target.windowFrame.isStructurallyValid && artifact.isStructurallyValid
    }
}

enum MacOSComputerUseObservationError: Error, Equatable {
    case permissionMissing([MacAutomationPermission])
    case targetNotConfigured
    case targetUnavailable
    case targetNotFrontmost
    case invalidCapture
}

protocol ComputerUseWindowCapturing: Sendable {
    func capture(_ selection: ComputerUseWindowSelection) async throws
        -> ComputerUseCapturedWindow
}

/// Memory-only bridge between the observer and a later local visual grounder.
/// A small FIFO cap prevents a long or failed run from retaining desktop
/// screenshots indefinitely.
actor ComputerUseObservationVault {
    private let maximumArtifacts: Int
    private var artifacts: [UUID: ComputerUseObservationArtifact] = [:]
    private var insertionOrder: [UUID] = []

    init(maximumArtifacts: Int = 6) {
        self.maximumArtifacts = max(1, maximumArtifacts)
    }

    func store(_ artifact: ComputerUseObservationArtifact, for id: UUID) {
        if artifacts[id] == nil { insertionOrder.append(id) }
        artifacts[id] = artifact
        while insertionOrder.count > maximumArtifacts {
            let evicted = insertionOrder.removeFirst()
            artifacts.removeValue(forKey: evicted)
        }
    }

    func artifact(for id: UUID) -> ComputerUseObservationArtifact? {
        artifacts[id]
    }

    func removeAll() {
        artifacts.removeAll(keepingCapacity: false)
        insertionOrder.removeAll(keepingCapacity: false)
    }
}

/// Production ``LocalWorkflowObserving`` adapter. It rechecks both TCC grants
/// for every observation and resolves the target from the compiled step ID.
/// The caller must populate this map from an explicit per-run window choice;
/// the observer never falls back to the whole desktop or a similarly titled
/// window.
actor MacOSComputerUseObserver: LocalWorkflowObserving {
    typealias PermissionReader = @Sendable () -> MacAutomationPermissionSnapshot

    private let selections: [String: ComputerUseWindowSelection]
    private let permissionReader: PermissionReader
    private let captureSource: any ComputerUseWindowCapturing
    private let vault: ComputerUseObservationVault

    init(
        selections: [String: ComputerUseWindowSelection],
        vault: ComputerUseObservationVault,
        captureSource: any ComputerUseWindowCapturing = ScreenCaptureKitComputerUseCapture(),
        permissionReader: @escaping PermissionReader = MacAutomationPermissions.snapshot
    ) {
        self.selections = selections
        self.vault = vault
        self.captureSource = captureSource
        self.permissionReader = permissionReader
    }

    func observe(for step: LocalWorkflowStep) async throws -> WorkflowObservation {
        try Task.checkCancellation()
        let permissions = permissionReader()
        guard permissions.isReadyForComputerUse else {
            throw MacOSComputerUseObservationError.permissionMissing(
                permissions.missingForComputerUse
            )
        }
        guard let selection = selections[step.id], selection.isStructurallyValid else {
            throw MacOSComputerUseObservationError.targetNotConfigured
        }

        let captured = try await captureSource.capture(selection)
        try Task.checkCancellation()
        guard captured.isStructurallyValid,
              captured.target.bundleIdentifier == selection.bundleIdentifier,
              captured.target.windowIdentifier == String(selection.windowID)
        else {
            throw MacOSComputerUseObservationError.invalidCapture
        }

        let id = UUID()
        let observation = WorkflowObservation(
            id: id,
            target: captured.target,
            contentRevision: Self.contentRevision(
                target: captured.target,
                pngData: captured.artifact.pngData
            )
        )
        await vault.store(captured.artifact, for: id)
        return observation
    }

    private static func contentRevision(
        target: WorkflowInteractionTarget,
        pngData: Data
    ) -> String {
        var bytes = Data(
            "\(target.bundleIdentifier)|\(target.processIdentifier)|"
                .utf8
        )
        bytes.append(Data("\(target.windowIdentifier)|".utf8))
        let frame = target.windowFrame
        bytes.append(Data("\(frame.x)|\(frame.y)|\(frame.width)|\(frame.height)|".utf8))
        bytes.append(pngData)
        return SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
    }
}

/// ScreenCaptureKit implementation restricted to one exact, on-screen window.
/// It rejects background/focus drift rather than bringing an app forward as a
/// hidden side effect of observation.
struct ScreenCaptureKitComputerUseCapture: ComputerUseWindowCapturing {
    func capture(_ selection: ComputerUseWindowSelection) async throws
        -> ComputerUseCapturedWindow
    {
        guard selection.isStructurallyValid else {
            throw MacOSComputerUseObservationError.targetNotConfigured
        }
        try Task.checkCancellation()

        let frontmost = await MainActor.run {
            let app = NSWorkspace.shared.frontmostApplication
            return (app?.bundleIdentifier, app?.processIdentifier)
        }
        guard frontmost.0 == selection.bundleIdentifier else {
            throw MacOSComputerUseObservationError.targetNotFrontmost
        }

        let content = try await SCShareableContent.excludingDesktopWindows(
            true,
            onScreenWindowsOnly: true
        )
        try Task.checkCancellation()
        guard let window = content.windows.first(where: {
            $0.windowID == selection.windowID
                && $0.owningApplication?.bundleIdentifier == selection.bundleIdentifier
        }),
            let application = window.owningApplication,
            application.processID == frontmost.1,
            window.isOnScreen
        else {
            throw MacOSComputerUseObservationError.targetUnavailable
        }

        let frame = window.frame
        guard frame.origin.x.isFinite, frame.origin.y.isFinite,
              frame.width.isFinite, frame.height.isFinite,
              frame.width > 0, frame.height > 0
        else {
            throw MacOSComputerUseObservationError.invalidCapture
        }

        let output = Self.outputSize(for: frame.size)
        let configuration = SCStreamConfiguration()
        configuration.width = output.width
        configuration.height = output.height
        configuration.showsCursor = false
        configuration.ignoreShadowsSingleWindow = true

        let filter = SCContentFilter(desktopIndependentWindow: window)
        let image = try await SCScreenshotManager.captureImage(
            contentFilter: filter,
            configuration: configuration
        )
        try Task.checkCancellation()
        guard let png = NSBitmapImageRep(cgImage: image).representation(
            using: .png,
            properties: [:]
        ) else {
            throw MacOSComputerUseObservationError.invalidCapture
        }

        return ComputerUseCapturedWindow(
            target: WorkflowInteractionTarget(
                bundleIdentifier: selection.bundleIdentifier,
                processIdentifier: application.processID,
                windowIdentifier: String(selection.windowID),
                windowFrame: WorkflowWindowFrame(
                    x: frame.origin.x,
                    y: frame.origin.y,
                    width: frame.width,
                    height: frame.height
                )
            ),
            artifact: ComputerUseObservationArtifact(
                pngData: png,
                pixelWidth: image.width,
                pixelHeight: image.height
            )
        )
    }

    private static func outputSize(for source: CGSize) -> (width: Int, height: Int) {
        let scale = min(1, min(1280 / source.width, 720 / source.height))
        return (
            max(1, Int((source.width * scale).rounded())),
            max(1, Int((source.height * scale).rounded()))
        )
    }
}
