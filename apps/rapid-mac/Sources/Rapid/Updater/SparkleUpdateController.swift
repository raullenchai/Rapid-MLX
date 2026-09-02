import Foundation
import Observation
import Sparkle

/// Owns Sparkle's updater lifecycle for signed production builds.
///
/// The source Info.plist deliberately has no `SUPublicEDKey`; build.sh injects
/// it for release builds. When the key is absent (normal local development)
/// this controller stays disabled and every method here is a silent no-op —
/// there is no in-app fallback installer to hand off to any more. Callers must
/// therefore offer something else in the disabled state (Settings → App falls
/// back to a link to the release page) rather than a control that does
/// nothing.
@MainActor
@Observable
final class SparkleUpdateController {
    enum FixtureState {
        case busy
    }

    private var standardController: SPUStandardUpdaterController?
    private var canCheckObservation: NSKeyValueObservation?
    private(set) var isStarted = false
    private(set) var automaticallyDownloadsUpdates: Bool
    /// Mirrors Sparkle's KVO-compliant state so SwiftUI never presents an
    /// enabled control whose action Sparkle is required to ignore. In
    /// particular, `canCheckForUpdates` is false while an automatic appcast or
    /// update download is running in the background.
    private(set) var canCheckForUpdates: Bool
    private let fixtureState: FixtureState?

    let isEnabled: Bool

    init(
        infoDictionary: [String: Any]? = Bundle.main.infoDictionary,
        checksEnabled: Bool = UpdateChecker.updateChecksEnabled(),
        fixtureState: FixtureState? = nil
    ) {
        self.fixtureState = fixtureState
        isEnabled = fixtureState != nil || (checksEnabled && Self.hasValidConfiguration(infoDictionary))
        // Preserve the pre-start contract: an enabled production controller
        // is actionable until Sparkle starts and publishes its real KVO
        // state. The deterministic busy fixture is the sole exception.
        if case .busy = fixtureState {
            canCheckForUpdates = false
        } else {
            canCheckForUpdates = isEnabled
        }
        automaticallyDownloadsUpdates = isEnabled
            ? (UserDefaults.standard.object(forKey: "SUAutomaticallyUpdate") as? Bool ?? true)
            : false
    }

    /// Start after AppKit has finished constructing the application. Sparkle
    /// owns its six-hour schedule and automatically downloads updates; a
    /// downloaded update is installed when Rapid next quits normally.
    func start() {
        guard isEnabled, !isStarted else { return }
        if fixtureState != nil {
            isStarted = true
            return
        }
        let controller = SPUStandardUpdaterController(
            startingUpdater: false,
            updaterDelegate: nil,
            userDriverDelegate: nil
        )
        standardController = controller
        controller.startUpdater()
        let updater = controller.updater
        automaticallyDownloadsUpdates = updater.automaticallyDownloadsUpdates
        canCheckForUpdates = updater.canCheckForUpdates
        canCheckObservation = updater.observe(\.canCheckForUpdates, options: [.initial, .new]) {
            [weak self] _, change in
            guard let value = change.newValue else { return }
            Task { @MainActor [weak self] in
                self?.canCheckForUpdates = value
            }
        }
        isStarted = true
    }

    /// Foreground check used by menu and Settings commands. The standard
    /// Sparkle UI reports current/update/error states and, when an update was
    /// already downloaded in the background, offers the install action.
    ///
    /// The observable mirror is presentation state and may trail Sparkle's
    /// KVO value by one main-actor hop. Re-read the updater synchronously at
    /// dispatch time so a stale enabled button cannot hide the discovery card
    /// after Sparkle has already become unable to accept the action.
    @discardableResult
    func checkForUpdates() -> Bool {
        guard isEnabled else { return false }
        if !isStarted { start() }
        guard let updater = standardController?.updater else { return false }
        return Self.dispatchForegroundCheck(
            authoritativeCanCheck: { updater.canCheckForUpdates },
            synchronizeMirror: { canCheckForUpdates = $0 },
            perform: { updater.checkForUpdates() }
        )
    }

    /// Keep the authoritative read and foreground dispatch in one synchronous
    /// main-actor turn. Sparkle requires both operations on the main thread, so
    /// its state cannot change through a queued KVO delivery between them.
    @discardableResult
    static func dispatchForegroundCheck(
        authoritativeCanCheck: () -> Bool,
        synchronizeMirror: (Bool) -> Void,
        perform: () -> Void
    ) -> Bool {
        let canCheck = authoritativeCanCheck()
        synchronizeMirror(canCheck)
        guard canCheck else { return false }
        perform()
        return true
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
