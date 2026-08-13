import Foundation
import Observation
import Sparkle

/// Owns Sparkle's updater lifecycle for signed production builds.
///
/// The source Info.plist deliberately has no `SUPublicEDKey`; build.sh injects
/// it for release builds. When the key is absent (normal local development),
/// this controller stays disabled and callers use the legacy manifest/DMG
/// updater. This also gives already-installed clients a migration bridge
/// instead of requiring both update systems to change in one release.
@MainActor
@Observable
final class SparkleUpdateController {
    private var standardController: SPUStandardUpdaterController?
    private(set) var isStarted = false
    private(set) var automaticallyDownloadsUpdates: Bool

    let isEnabled: Bool

    init(
        infoDictionary: [String: Any]? = Bundle.main.infoDictionary,
        checksEnabled: Bool = UpdateChecker.updateChecksEnabled()
    ) {
        isEnabled = checksEnabled && Self.hasValidConfiguration(infoDictionary)
        automaticallyDownloadsUpdates = isEnabled
            ? (UserDefaults.standard.object(forKey: "SUAutomaticallyUpdate") as? Bool ?? true)
            : false
    }

    /// Start after AppKit has finished constructing the application. Sparkle
    /// owns its six-hour schedule and automatically downloads updates; a
    /// downloaded update is installed when Rapid next quits normally.
    func start() {
        guard isEnabled, !isStarted else { return }
        let controller = SPUStandardUpdaterController(
            startingUpdater: false,
            updaterDelegate: nil,
            userDriverDelegate: nil
        )
        standardController = controller
        controller.startUpdater()
        automaticallyDownloadsUpdates = controller.updater.automaticallyDownloadsUpdates
        isStarted = true
    }

    /// Foreground check used by menu and Settings commands. The standard
    /// Sparkle UI reports current/update/error states and, when an update was
    /// already downloaded in the background, offers the install action.
    func checkForUpdates() {
        guard isEnabled else { return }
        if !isStarted { start() }
        standardController?.checkForUpdates(nil)
    }

    var canCheckForUpdates: Bool {
        isEnabled && (standardController?.updater.canCheckForUpdates ?? true)
    }

    func setAutomaticallyDownloadsUpdates(_ enabled: Bool) {
        guard isEnabled else { return }
        if !isStarted { start() }
        standardController?.updater.automaticallyDownloadsUpdates = enabled
        automaticallyDownloadsUpdates = standardController?.updater.automaticallyDownloadsUpdates ?? enabled
    }

    nonisolated static func hasValidConfiguration(_ info: [String: Any]?) -> Bool {
        guard let info,
              let feed = info["SUFeedURL"] as? String,
              let url = URL(string: feed),
              url.scheme?.lowercased() == "https",
              url.user == nil,
              url.password == nil,
              url.host?.isEmpty == false,
              let publicKey = info["SUPublicEDKey"] as? String,
              let keyData = Data(base64Encoded: publicKey),
              keyData.count == 32 else {
            return false
        }
        return true
    }
}
