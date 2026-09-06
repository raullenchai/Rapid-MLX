import AppKit
import ApplicationServices
import CoreGraphics
import Foundation

/// macOS privacy grants used when Rapid observes or controls another app.
///
/// Checking a grant is side-effect free. Requesting one is deliberately kept
/// separate so merely enabling an experimental feature never opens a system
/// prompt or settings pane.
enum MacAutomationPermission: String, CaseIterable, Equatable, Sendable {
    case screenRecording
    case accessibility

    var title: String {
        switch self {
        case .screenRecording: "Screen Recording"
        case .accessibility: "Accessibility"
        }
    }
}

struct MacAutomationPermissionSnapshot: Equatable, Sendable {
    let screenRecording: Bool
    let accessibility: Bool

    func isGranted(_ permission: MacAutomationPermission) -> Bool {
        switch permission {
        case .screenRecording: screenRecording
        case .accessibility: accessibility
        }
    }

    var missingForComputerUse: [MacAutomationPermission] {
        // Observation comes before action in every Computer Use run, so keep
        // Screen Recording first in both setup copy and VoiceOver order.
        MacAutomationPermission.allCases.filter { !isGranted($0) }
    }

    var isReadyForComputerUse: Bool { missingForComputerUse.isEmpty }
}

/// Thin system boundary shared by Dictation and Computer Use.
///
/// The TCC APIs do not distinguish "not asked" from "denied" here. Product
/// UI therefore says "Not allowed" and offers the exact Settings pane instead
/// of claiming a state macOS has not exposed to us.
enum MacAutomationPermissions {
    static func snapshot() -> MacAutomationPermissionSnapshot {
        MacAutomationPermissionSnapshot(
            screenRecording: CGPreflightScreenCaptureAccess(),
            accessibility: AXIsProcessTrusted()
        )
    }

    static func isGranted(_ permission: MacAutomationPermission) -> Bool {
        snapshot().isGranted(permission)
    }

    /// Requests exactly one capability in direct response to a user action.
    /// The returned value is only the state at call completion; callers must
    /// refresh after the user changes System Settings and after app relaunch.
    @discardableResult
    static func request(_ permission: MacAutomationPermission) -> Bool {
        switch permission {
        case .screenRecording:
            return CGRequestScreenCaptureAccess()
        case .accessibility:
            // Swift 6 treats the SDK's exported CFString as shared mutable
            // state. Its documented literal value is stable.
            let key = "AXTrustedCheckOptionPrompt" as CFString
            return AXIsProcessTrustedWithOptions([key: true] as CFDictionary)
        }
    }

    @MainActor
    static func openSettings(for permission: MacAutomationPermission) {
        let anchor: String
        switch permission {
        case .screenRecording:
            anchor = "Privacy_ScreenCapture"
        case .accessibility:
            anchor = "Privacy_Accessibility"
        }
        guard let url = URL(
            string: "x-apple.systempreferences:com.apple.preference.security?\(anchor)"
        ) else { return }
        NSWorkspace.shared.open(url)
    }
}
