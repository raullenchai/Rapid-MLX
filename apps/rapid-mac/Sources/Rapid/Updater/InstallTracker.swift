import Foundation
import Observation

/// Detects the "Finder Replace into /Applications silently failed because
/// Rapid-MLX Desktop was still running" footgun (rapid-desktop issue #251).
///
/// macOS Finder lets the user drop ``Rapid-MLX Desktop.app`` onto
/// ``/Applications/`` while an instance of the app is already running and
/// click **Replace** in the standard confirmation prompt. The Replace
/// appears to succeed (no error dialog) but Finder cannot actually
/// overwrite files inside the live ``.app`` bundle, so the on-disk
/// bundle stays at the old version. On the next launch the user is
/// silently running the old build and reports "feature X doesn't work"
/// while the actual cause is a stale bundle.
///
/// We can't reliably PREVENT the drag-replace (the user may use any
/// downloader — website link, GitHub Releases, the bundled DMG); the
/// in-app updater takes the safe path already, because Sparkle installs
/// on quit rather than over a running bundle. What we can do is DETECT
/// the failure mode on
/// the next launch and surface a sticky banner explaining what happened,
/// so the user can quit + re-run the installer instead of silently
/// staying on the old build for weeks.
///
/// Signal: the running bundle's ``Contents/Info.plist`` mtime is newer
/// than the previous launch's recorded mtime, yet
/// ``CFBundleShortVersionString`` is unchanged, and the running bundle
/// lives under ``/Applications``. Finder's failed Replace DOES touch the
/// bundle's mtime (it goes through `copyfile` even when the writes don't
/// take), but obviously can't change the version a stale file already
/// advertises. The location gate excludes dev builds and mounted DMGs;
/// a legitimate upgrade bumps the version, and a routine launch without
/// an update attempt leaves the mtime alone.
///
/// Persistence lives in UserDefaults so the comparison survives the
/// app exit + relaunch cycle that the Finder Replace forces.
@MainActor
@Observable
final class InstallTracker {
    /// True iff the previous-launch bookkeeping suggests the user
    /// attempted an upgrade (bundle mtime advanced) but the version
    /// didn't move. Banner-driving flag for the chat surface.
    private(set) var failedReplaceDetected: Bool = false

    /// The version string the app booted with. Cached so the banner
    /// copy can echo what the user is currently running (matches what
    /// the About panel + status bar already show).
    let currentVersion: String

    private let defaults: UserDefaults

    /// UserDefaults keys. Namespaced so they don't collide with the
    /// other ``rapid.*`` keys in the same suite.
    static let lastSeenMtimeKey = "rapid.install.lastSeenInfoPlistMtime"
    static let lastSeenVersionKey = "rapid.install.lastSeenVersion"

    /// Production init — reads the running bundle's Info.plist mtime
    /// and ``CFBundleShortVersionString``, compares against the
    /// previous launch's persisted values, and updates persistence so
    /// the NEXT launch can do the same comparison.
    convenience init(
        bundleURL: URL = Bundle.main.bundleURL,
        defaults: UserDefaults = .standard
    ) {
        let infoPlist = bundleURL.appendingPathComponent("Contents/Info.plist")
        let mtime = (try? FileManager.default.attributesOfItem(atPath: infoPlist.path))
            .flatMap { $0[.modificationDate] as? Date }
        let version = UpdateChecker.bundleVersion()
        self.init(
            currentVersion: version,
            currentInfoPlistMtime: mtime,
            currentBundleURL: bundleURL,
            defaults: defaults
        )
    }

    /// Test seam: caller supplies the version + mtime explicitly so the
    /// suite can drive every combination without touching the real
    /// bundle. Updates persistence as a side effect, just like the
    /// production init — tests assert on the final stored values to
    /// verify the rollover behaviour.
    init(
        currentVersion: String,
        currentInfoPlistMtime: Date?,
        currentBundleURL: URL,
        defaults: UserDefaults
    ) {
        self.currentVersion = currentVersion
        self.defaults = defaults

        let prevMtime = defaults.object(forKey: Self.lastSeenMtimeKey) as? Date
        let prevVersion = defaults.string(forKey: Self.lastSeenVersionKey)

        self.failedReplaceDetected = Self.detect(
            previousMtime: prevMtime,
            previousVersion: prevVersion,
            currentMtime: currentInfoPlistMtime,
            currentVersion: currentVersion,
            currentBundleURL: currentBundleURL
        )

        // Persist both halves of the baseline ATOMICALLY: only
        // advance the stored snapshot when we successfully stat'd
        // the current Info.plist. A previous shape unconditionally
        // wrote ``currentVersion`` even when ``currentInfoPlistMtime``
        // was nil — that can produce a stale mixed baseline (old
        // mtime + new version) if the stat fails during a legitimate
        // upgrade, which would then falsely flag a failed Replace
        // on the launch after that (the launch with a successful
        // stat would see fresh new-version mtime against an already
        // stored new version). Persisting both only when we trusted
        // the stat keeps the baseline atomically tied to a mtime we
        // actually observed.
        if let mtime = currentInfoPlistMtime {
            defaults.set(mtime, forKey: Self.lastSeenMtimeKey)
            defaults.set(currentVersion, forKey: Self.lastSeenVersionKey)
        }
    }

    /// Pure detection logic, exposed for testing. The running bundle
    /// must live under ``/Applications``, the mtime must have moved
    /// forward, and the version must be unchanged.
    ///
    /// Edge cases handled explicitly:
    ///   * First launch (no previous values): always returns false —
    ///     we have nothing to compare against, and the persistence
    ///     write at the end of init records the baseline for next time.
    ///   * Missing current mtime (Info.plist unreadable): returns
    ///     false. A bundle we can't stat is a separate failure mode;
    ///     no need to compose two unrelated errors into one banner.
    ///   * Bundle outside /Applications: returns false. Dev builds,
    ///     mounted DMGs, and other copies cannot represent this specific
    ///     failed Finder Replace scenario.
    ///   * Same mtime: the bundle wasn't touched at all → not a failed
    ///     install attempt.
    ///   * Version moved: the install DID take, regardless of mtime
    ///     direction → suppress the banner. Includes downgrade rollbacks.
    nonisolated static func detect(
        previousMtime: Date?,
        previousVersion: String?,
        currentMtime: Date?,
        currentVersion: String,
        currentBundleURL: URL
    ) -> Bool {
        // Only a bundle sitting DIRECTLY in /Applications is a Finder-Replace
        // release install. `hasPrefix("/Applications/")` matched at any depth,
        // so a dev build under a subdirectory
        // (e.g. /Applications/RapidDev/Rapid-MLX Desktop.app) slipped through:
        // rebuilds bump the Info.plist mtime while keeping the same version,
        // which then read as a "failed replace" and flashed the banner. A
        // parent-directory equality check keeps the sibling-prefix exclusion
        // (/ApplicationsBackup/…) and adds the subdirectory one.
        let bundleURL = currentBundleURL.standardizedFileURL
        guard bundleURL.deletingLastPathComponent().path == "/Applications" else {
            return false
        }
        guard let previousMtime, let previousVersion, let currentMtime else {
            return false
        }
        if currentVersion != previousVersion {
            return false
        }
        // Half-second slack absorbs filesystem timestamp jitter
        // between APFS resolution and Date() rounding so a no-op
        // re-launch with the exact same bundle doesn't flag.
        return currentMtime.timeIntervalSince(previousMtime) > 0.5
    }

    /// User clicked the banner's "Dismiss" affordance. Clears the
    /// flag for the current process; persistence already reflects
    /// today's mtime so the next launch won't re-fire unless ANOTHER
    /// Finder Replace happens between now and then.
    func dismiss() {
        failedReplaceDetected = false
    }
}
