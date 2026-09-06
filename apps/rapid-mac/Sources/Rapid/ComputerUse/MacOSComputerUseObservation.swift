import AppKit
import CryptoKit
import Foundation
import ScreenCaptureKit

/// A window the user explicitly allowed one Computer Use step to observe.
/// Window IDs are session-scoped and are never treated as durable selectors.
struct ComputerUseWindowSelection: Equatable, Sendable {
    let bundleIdentifier: String
    let processIdentifier: pid_t
    let processLaunchDate: Date
    let windowID: CGWindowID

    var isStructurallyValid: Bool {
        !bundleIdentifier.isEmpty && processIdentifier > 0 && windowID != 0
            && processLaunchDate.timeIntervalSinceReferenceDate.isFinite
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
              captured.target.processIdentifier == selection.processIdentifier,
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

    static func contentRevision(
        target: WorkflowInteractionTarget,
        pngData: Data
    ) -> String {
        var bytes = Data(
            "\(target.bundleIdentifier)|\(target.processIdentifier)|"
                .utf8
        )
        bytes.append(
            Data("\(target.processLaunchDate?.timeIntervalSinceReferenceDate.bitPattern ?? 0)|".utf8)
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
    struct WindowRecord: Equatable, Sendable {
        let windowID: CGWindowID
        let bundleIdentifier: String?
        let processIdentifier: pid_t?
        let frame: CGRect
        let isOnScreen: Bool
    }

    struct ForegroundRecord: Equatable, Sendable {
        let bundleIdentifier: String?
        let processIdentifier: pid_t?
        let processLaunchDate: Date?
        let focusedFrame: CGRect?
    }

    // SCWindow is an immutable system descriptor but is not annotated Sendable.
    // The wrapper never mutates it and passes it only to ScreenCaptureKit's
    // async screenshot API; all security decisions use the Sendable target.
    struct ResolvedWindow: @unchecked Sendable {
        let window: SCWindow?
        let target: WorkflowInteractionTarget
    }

    typealias WindowResolver = @Sendable (
        ComputerUseWindowSelection
    ) async throws -> ResolvedWindow
    typealias ImageCapturer = @Sendable (
        ResolvedWindow,
        CGSize
    ) async throws -> ComputerUseObservationArtifact

    private let windowResolver: WindowResolver
    private let imageCapturer: ImageCapturer

    init(
        windowResolver: @escaping WindowResolver = Self.resolve,
        imageCapturer: @escaping ImageCapturer = Self.captureImage
    ) {
        self.windowResolver = windowResolver
        self.imageCapturer = imageCapturer
    }

    func capture(_ selection: ComputerUseWindowSelection) async throws
        -> ComputerUseCapturedWindow
    {
        guard selection.isStructurallyValid else {
            throw MacOSComputerUseObservationError.targetNotConfigured
        }
        try Task.checkCancellation()
        let before = try await windowResolver(selection)
        guard Self.target(before.target, matches: selection) else {
            throw MacOSComputerUseObservationError.invalidCapture
        }
        let targetFrame = before.target.windowFrame
        let frame = CGRect(
            x: targetFrame.x,
            y: targetFrame.y,
            width: targetFrame.width,
            height: targetFrame.height
        )
        guard frame.origin.x.isFinite, frame.origin.y.isFinite,
              frame.width.isFinite, frame.height.isFinite,
              frame.width > 0, frame.height > 0
        else {
            throw MacOSComputerUseObservationError.invalidCapture
        }

        let output = Self.outputSize(for: frame.size)
        let artifact = try await imageCapturer(
            before,
            CGSize(width: output.width, height: output.height)
        )
        try Task.checkCancellation()
        let after = try await windowResolver(selection)
        guard Self.target(after.target, matches: selection),
              MacOSComputerUseWindowIdentity.targetsMatch(
            before.target,
            after.target
        ) else {
            throw MacOSComputerUseObservationError.invalidCapture
        }
        guard artifact.isStructurallyValid else {
            throw MacOSComputerUseObservationError.invalidCapture
        }

        return ComputerUseCapturedWindow(
            target: after.target,
            artifact: artifact
        )
    }

    private static func resolve(
        _ selection: ComputerUseWindowSelection
    ) async throws -> ResolvedWindow {
        let content = try await SCShareableContent.excludingDesktopWindows(
            true,
            onScreenWindowsOnly: true
        )
        try Task.checkCancellation()
        let foreground = await MainActor.run {
            let application = NSWorkspace.shared.frontmostApplication
            let processIdentifier = application?.processIdentifier
            return (
                application?.bundleIdentifier,
                processIdentifier,
                application?.launchDate,
                processIdentifier.flatMap {
                    MacOSComputerUseWindowIdentity.focusedWindowFrame(
                        processIdentifier: $0
                    )
                }
            )
        }
        let records = content.windows.map {
            WindowRecord(
                windowID: $0.windowID,
                bundleIdentifier: $0.owningApplication?.bundleIdentifier,
                processIdentifier: $0.owningApplication?.processID,
                frame: $0.frame,
                isOnScreen: $0.isOnScreen
            )
        }
        let target = try validatedTarget(
            selection,
            foreground: ForegroundRecord(
                bundleIdentifier: foreground.0,
                processIdentifier: foreground.1,
                processLaunchDate: foreground.2,
                focusedFrame: foreground.3
            ),
            windows: records
        )
        guard let window = content.windows.first(where: {
            $0.windowID == selection.windowID
                && $0.owningApplication?.bundleIdentifier == selection.bundleIdentifier
                && $0.owningApplication?.processID == selection.processIdentifier
                && $0.isOnScreen
        }) else {
            throw MacOSComputerUseObservationError.targetUnavailable
        }
        return ResolvedWindow(
            window: window,
            target: target
        )
    }

    static func validatedTarget(
        _ selection: ComputerUseWindowSelection,
        foreground: ForegroundRecord,
        windows: [WindowRecord]
    ) throws -> WorkflowInteractionTarget {
        guard foreground.bundleIdentifier == selection.bundleIdentifier,
              foreground.processIdentifier == selection.processIdentifier,
              foreground.processLaunchDate == selection.processLaunchDate,
              let focusedFrame = foreground.focusedFrame
        else {
            throw MacOSComputerUseObservationError.targetNotFrontmost
        }
        guard let window = windows.first(where: {
            $0.windowID == selection.windowID
                && $0.bundleIdentifier == selection.bundleIdentifier
                && $0.processIdentifier == selection.processIdentifier
                && $0.isOnScreen
        }) else {
            throw MacOSComputerUseObservationError.targetUnavailable
        }
        let focusedCandidates = windows.filter {
            $0.isOnScreen
                && $0.processIdentifier == selection.processIdentifier
                && MacOSComputerUseWindowIdentity.framesMatch($0.frame, focusedFrame)
        }
        guard focusedCandidates.count == 1,
              focusedCandidates[0].windowID == selection.windowID
        else {
            throw MacOSComputerUseObservationError.targetNotFrontmost
        }
        return WorkflowInteractionTarget(
            bundleIdentifier: selection.bundleIdentifier,
            processIdentifier: selection.processIdentifier,
            processLaunchDate: selection.processLaunchDate,
            windowIdentifier: String(selection.windowID),
            windowFrame: WorkflowWindowFrame(
                x: window.frame.origin.x,
                y: window.frame.origin.y,
                width: window.frame.width,
                height: window.frame.height
            )
        )
    }

    private static func target(
        _ target: WorkflowInteractionTarget,
        matches selection: ComputerUseWindowSelection
    ) -> Bool {
        target.bundleIdentifier == selection.bundleIdentifier
            && target.processIdentifier == selection.processIdentifier
            && target.processLaunchDate == selection.processLaunchDate
            && target.windowIdentifier == String(selection.windowID)
    }

    private static func captureImage(
        _ resolved: ResolvedWindow,
        _ outputSize: CGSize
    ) async throws -> ComputerUseObservationArtifact {
        guard let window = resolved.window else {
            throw MacOSComputerUseObservationError.targetUnavailable
        }
        let configuration = SCStreamConfiguration()
        configuration.width = Int(outputSize.width)
        configuration.height = Int(outputSize.height)
        configuration.showsCursor = false
        configuration.ignoreShadowsSingleWindow = true
        let filter = SCContentFilter(desktopIndependentWindow: window)
        let image = try await SCScreenshotManager.captureImage(
            contentFilter: filter,
            configuration: configuration
        )
        guard let png = NSBitmapImageRep(cgImage: image).representation(
            using: .png,
            properties: [:]
        ) else {
            throw MacOSComputerUseObservationError.invalidCapture
        }
        return ComputerUseObservationArtifact(
            pngData: png,
            pixelWidth: image.width,
            pixelHeight: image.height
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
