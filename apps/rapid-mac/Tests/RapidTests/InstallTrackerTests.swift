import Foundation
import Testing
@testable import Rapid

/// rapid-desktop issue #251 — Finder Replace into ``/Applications/``
/// silently fails when the live app holds files inside its own bundle.
/// ``InstallTracker`` detects the resulting "bundle was touched but
/// the version didn't move" signature on the next launch and drives
/// the ``FailedReplaceBanner``.
///
/// These tests exercise both the pure detection logic and the
/// init-time persistence write so we know the next-launch comparison
/// will see the right baseline regardless of what the current launch
/// flagged.
@MainActor
@Suite("InstallTracker — failed Finder Replace detection (#251)")
struct InstallTrackerTests {
    private let bundleID = "rapid-tests.installtracker.\(UUID().uuidString)"
    private let installedBundleURL = URL(
        fileURLWithPath: "/Applications/Rapid-MLX Desktop.app",
        isDirectory: true
    )

    private func freshDefaults() -> UserDefaults {
        let suite = UserDefaults(suiteName: bundleID)!
        suite.removePersistentDomain(forName: bundleID)
        return suite
    }

    // MARK: - Pure detection logic

    @Test("first launch: nil baseline never flags")
    func firstLaunchNeverFlags() {
        #expect(
            InstallTracker.detect(
                previousMtime: nil,
                previousVersion: nil,
                currentMtime: Date(),
                currentVersion: "0.7.6",
                currentBundleURL: installedBundleURL
            ) == false
        )
    }

    @Test("no current mtime: never flags")
    func missingCurrentMtimeNeverFlags() {
        #expect(
            InstallTracker.detect(
                previousMtime: Date(),
                previousVersion: "0.7.6",
                currentMtime: nil,
                currentVersion: "0.7.6",
                currentBundleURL: installedBundleURL
            ) == false
        )
    }

    @Test("version moved forward: never flags (legitimate upgrade)")
    func upgradeNeverFlags() {
        let prev = Date(timeIntervalSince1970: 1_000_000)
        let now = prev.addingTimeInterval(3_600)
        #expect(
            InstallTracker.detect(
                previousMtime: prev,
                previousVersion: "0.7.5",
                currentMtime: now,
                currentVersion: "0.7.6",
                currentBundleURL: installedBundleURL
            ) == false
        )
    }

    @Test("version moved backward: never flags (downgrade rollback)")
    func downgradeNeverFlags() {
        let prev = Date(timeIntervalSince1970: 1_000_000)
        let now = prev.addingTimeInterval(3_600)
        #expect(
            InstallTracker.detect(
                previousMtime: prev,
                previousVersion: "0.7.6",
                currentMtime: now,
                currentVersion: "0.7.5",
                currentBundleURL: installedBundleURL
            ) == false
        )
    }

    @Test("same version, same mtime: no-op relaunch never flags")
    func unchangedNeverFlags() {
        let frozen = Date(timeIntervalSince1970: 1_000_000)
        #expect(
            InstallTracker.detect(
                previousMtime: frozen,
                previousVersion: "0.7.6",
                currentMtime: frozen,
                currentVersion: "0.7.6",
                currentBundleURL: installedBundleURL
            ) == false
        )
    }

    @Test("sub-jitter mtime drift (≤ 0.5 s): never flags")
    func jitterDoesNotFlag() {
        let prev = Date(timeIntervalSince1970: 1_000_000)
        #expect(
            InstallTracker.detect(
                previousMtime: prev,
                previousVersion: "0.7.6",
                currentMtime: prev.addingTimeInterval(0.4),
                currentVersion: "0.7.6",
                currentBundleURL: installedBundleURL
            ) == false
        )
    }

    @Test("mtime advanced + version unchanged: FLAGS the failed Replace")
    func failedReplaceFlags() {
        let prev = Date(timeIntervalSince1970: 1_000_000)
        let now = prev.addingTimeInterval(120)
        #expect(
            InstallTracker.detect(
                previousMtime: prev,
                previousVersion: "0.7.6",
                currentMtime: now,
                currentVersion: "0.7.6",
                currentBundleURL: installedBundleURL
            ) == true
        )
    }

    @Test(
        "same version + newer mtime outside /Applications: never flags",
        arguments: [
            "/Users/developer/DerivedData/Rapid-MLX Desktop.app",
            "/Volumes/Rapid-MLX/Rapid-MLX Desktop.app",
            "/ApplicationsBackup/Rapid-MLX Desktop.app",
            // Dev build under a /Applications SUBdirectory: rebuilds bump the
            // mtime while the version is unchanged, but this is not the
            // canonical Finder-Replace install and must never flag.
            "/Applications/RapidDev/Rapid-MLX Desktop.app",
        ]
    )
    func nonInstalledBundleNeverFlags(path: String) {
        let prev = Date(timeIntervalSince1970: 1_000_000)
        #expect(
            InstallTracker.detect(
                previousMtime: prev,
                previousVersion: "0.7.6",
                currentMtime: prev.addingTimeInterval(120),
                currentVersion: "0.7.6",
                currentBundleURL: URL(fileURLWithPath: path, isDirectory: true)
            ) == false
        )
    }

    // MARK: - Init-time persistence rollover

    @Test("first launch: writes baseline, never flags, stays clean")
    func firstLaunchSeedsBaselineWithoutFlag() {
        let defaults = freshDefaults()
        let now = Date(timeIntervalSince1970: 2_000_000)
        let tracker = InstallTracker(
            currentVersion: "0.7.6",
            currentInfoPlistMtime: now,
            currentBundleURL: installedBundleURL,
            defaults: defaults
        )
        #expect(tracker.failedReplaceDetected == false)
        #expect(defaults.string(forKey: InstallTracker.lastSeenVersionKey) == "0.7.6")
        let stored = defaults.object(forKey: InstallTracker.lastSeenMtimeKey) as? Date
        #expect(stored == now)
    }

    @Test("failed-Replace launch: flags AND updates baseline so it won't re-fire next launch")
    func failedReplaceFlagsAndRollsBaseline() {
        let defaults = freshDefaults()
        let prevMtime = Date(timeIntervalSince1970: 2_000_000)
        defaults.set(prevMtime, forKey: InstallTracker.lastSeenMtimeKey)
        defaults.set("0.7.6", forKey: InstallTracker.lastSeenVersionKey)

        let touchedMtime = prevMtime.addingTimeInterval(120)
        let tracker = InstallTracker(
            currentVersion: "0.7.6",
            currentInfoPlistMtime: touchedMtime,
            currentBundleURL: installedBundleURL,
            defaults: defaults
        )
        #expect(tracker.failedReplaceDetected == true)
        // Baseline rolls forward so a clean relaunch after the user
        // sees the banner doesn't keep nagging — only ANOTHER failed
        // Replace between dismissal and next launch re-fires.
        let storedMtime = defaults.object(forKey: InstallTracker.lastSeenMtimeKey) as? Date
        #expect(storedMtime == touchedMtime)
        #expect(defaults.string(forKey: InstallTracker.lastSeenVersionKey) == "0.7.6")
    }

    @Test("legit upgrade: never flags, baseline tracks the new version")
    func legitUpgradeRollsVersionForward() {
        let defaults = freshDefaults()
        let prevMtime = Date(timeIntervalSince1970: 2_000_000)
        defaults.set(prevMtime, forKey: InstallTracker.lastSeenMtimeKey)
        defaults.set("0.7.5", forKey: InstallTracker.lastSeenVersionKey)

        let newMtime = prevMtime.addingTimeInterval(3_600)
        let tracker = InstallTracker(
            currentVersion: "0.7.6",
            currentInfoPlistMtime: newMtime,
            currentBundleURL: installedBundleURL,
            defaults: defaults
        )
        #expect(tracker.failedReplaceDetected == false)
        #expect(defaults.string(forKey: InstallTracker.lastSeenVersionKey) == "0.7.6")
        let storedMtime = defaults.object(forKey: InstallTracker.lastSeenMtimeKey) as? Date
        #expect(storedMtime == newMtime)
    }

    @Test("dismiss(): clears the flag without touching persistence")
    func dismissClearsFlag() {
        let defaults = freshDefaults()
        let prevMtime = Date(timeIntervalSince1970: 2_000_000)
        defaults.set(prevMtime, forKey: InstallTracker.lastSeenMtimeKey)
        defaults.set("0.7.6", forKey: InstallTracker.lastSeenVersionKey)

        let touchedMtime = prevMtime.addingTimeInterval(120)
        let tracker = InstallTracker(
            currentVersion: "0.7.6",
            currentInfoPlistMtime: touchedMtime,
            currentBundleURL: installedBundleURL,
            defaults: defaults
        )
        #expect(tracker.failedReplaceDetected == true)
        tracker.dismiss()
        #expect(tracker.failedReplaceDetected == false)
        // Persistence still carries the touched mtime — a no-op relaunch
        // tomorrow stays clean.
        let storedMtime = defaults.object(forKey: InstallTracker.lastSeenMtimeKey) as? Date
        #expect(storedMtime == touchedMtime)
    }

    @Test("missing current mtime: doesn't overwrite stored mtime")
    func missingCurrentMtimePreservesStoredBaseline() {
        let defaults = freshDefaults()
        let prevMtime = Date(timeIntervalSince1970: 2_000_000)
        defaults.set(prevMtime, forKey: InstallTracker.lastSeenMtimeKey)
        defaults.set("0.7.6", forKey: InstallTracker.lastSeenVersionKey)

        let tracker = InstallTracker(
            currentVersion: "0.7.6",
            currentInfoPlistMtime: nil,
            currentBundleURL: installedBundleURL,
            defaults: defaults
        )
        #expect(tracker.failedReplaceDetected == false)
        let storedMtime = defaults.object(forKey: InstallTracker.lastSeenMtimeKey) as? Date
        // Stat'd mtime preserved — overwriting it with nil would mean
        // the NEXT launch starts from a clean slate and a real
        // failed-Replace that happens between now and then wouldn't be
        // caught. Same reasoning the production init applies.
        #expect(storedMtime == prevMtime)
    }

    @Test("missing current mtime DURING upgrade: doesn't roll version forward")
    func missingCurrentMtimeDuringUpgradePreservesBothBaselines() {
        // Codex r1 nit (atomic-baseline): the previous shape wrote
        // ``currentVersion`` unconditionally even when the stat for
        // ``currentInfoPlistMtime`` failed. If the stat fails during
        // a legitimate upgrade (0.7.6 → 0.7.7 with a transient I/O
        // hiccup), the stored baseline becomes (mtime=0.7.6's,
        // version=0.7.7) — a half-state. On the launch AFTER that, a
        // successful stat reads the fresh 0.7.7 mtime, sees the SAME
        // stored 0.7.7 version, and the detector falsely flags a
        // failed Replace. Persisting both halves only when we trusted
        // the mtime keeps the baseline atomically tied to a stat we
        // actually observed.
        let defaults = freshDefaults()
        let prevMtime = Date(timeIntervalSince1970: 2_000_000)
        defaults.set(prevMtime, forKey: InstallTracker.lastSeenMtimeKey)
        defaults.set("0.7.6", forKey: InstallTracker.lastSeenVersionKey)

        let tracker = InstallTracker(
            currentVersion: "0.7.7",
            currentInfoPlistMtime: nil,
            currentBundleURL: installedBundleURL,
            defaults: defaults
        )
        #expect(tracker.failedReplaceDetected == false)
        let storedMtime = defaults.object(forKey: InstallTracker.lastSeenMtimeKey) as? Date
        let storedVersion = defaults.string(forKey: InstallTracker.lastSeenVersionKey)
        #expect(storedMtime == prevMtime)
        #expect(storedVersion == "0.7.6")
    }
}
