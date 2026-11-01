import Foundation

/// Pure enable/arming decisions for Dictation.
///
/// The controller owns async catalog, model and event-tap side effects. This
/// value owns the policy between them so every prerequisite and stale-request
/// boundary can be exhaustively tested without constructing AppKit/AV objects.
struct DictationEnablePolicy {
    struct Prerequisites: Equatable {
        var microphone: Bool
        var accessibility: Bool
        var modelSelected: Bool
        var modelOnDisk: Bool
        var modelAlias: String
    }

    enum Decision: Equatable {
        case prepareModel
        case reject(message: String, disableIntent: Bool)
    }

    static func modelNotDownloadedMessage(alias: String) -> String {
        "\(ModelDisplayName.name(for: alias)) isn't downloaded yet. Download it in the Model row, then turn dictation on."
    }

    static func modelNoLongerOnDiskMessage(alias: String) -> String {
        "\(ModelDisplayName.name(for: alias)) isn't on disk anymore. Open Rapid → Audio to download it."
    }

    static func showsErrorBanner(lastError: String?, modelAlias: String, modelDownloadRowVisible: Bool) -> Bool {
        guard let lastError, !lastError.isEmpty else { return false }
        return !modelDownloadRowVisible || (lastError != modelNotDownloadedMessage(alias: modelAlias)
            && lastError != modelNoLongerOnDiskMessage(alias: modelAlias))
    }

    static func evaluate(_ input: Prerequisites) -> Decision {
        guard input.microphone else {
            return .reject(
                message: "Dictation needs Microphone access before it can be enabled.",
                disableIntent: true
            )
        }
        guard input.modelSelected else {
            return .reject(
                message: "Choose a transcription model before enabling dictation.",
                disableIntent: true
            )
        }
        guard input.modelOnDisk else {
            return .reject(
                message: modelNotDownloadedMessage(alias: input.modelAlias),
                disableIntent: true
            )
        }
        guard input.accessibility else {
            return .reject(
                message: "Dictation needs Accessibility access before the hotkey can be used.",
                disableIntent: false
            )
        }
        return .prepareModel
    }

    struct PreparedState: Equatable {
        var prewarmSucceeded: Bool
        var isEnabled: Bool
        var requestIsCurrent: Bool
        var selectedAlias: String
        var preparingAlias: String
        var isPreparing: Bool
        var voiceLaneReady: Bool
    }

    /// The sole policy gate before installing the process-global hotkey.
    /// Readiness is capability-based because a conversation process can own a
    /// separate, lazily loaded voice lane without changing its serving alias.
    /// Every async identity must still match after model preparation; a stale
    /// completion is never allowed to arm input for a different user state.
    static func mayRegisterHotkey(after state: PreparedState) -> Bool {
        state.prewarmSucceeded
            && state.isEnabled
            && state.requestIsCurrent
            && state.selectedAlias == state.preparingAlias
            && state.isPreparing
            && state.voiceLaneReady
    }
}
