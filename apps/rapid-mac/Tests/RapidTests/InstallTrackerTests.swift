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

    // MARK: - Post-upgrade "what's new" notice

    /// Sparkle's silent install means the version moving forward is the only
    /// evidence the user ever gets that something changed. 0.13.1 → 0.14.1
    /// crossed two feature releases and the window came back identical apart
    /// from the version pill.
    @Test("version moved forward: reports what it upgraded from")
    func upgradeReportsPreviousVersion() {
        let defaults = freshDefaults()
        defaults.set(Date(timeIntervalSince1970: 2_000_000), forKey: InstallTracker.lastSeenMtimeKey)
        defaults.set("0.13.1", forKey: InstallTracker.lastSeenVersionKey)

        let tracker = InstallTracker(
            currentVersion: "0.14.1",
            currentInfoPlistMtime: Date(timeIntervalSince1970: 2_003_600),
            currentBundleURL: installedBundleURL,
            defaults: defaults
        )
        #expect(tracker.upgradedFrom == "0.13.1")
        #expect(tracker.failedReplaceDetected == false)
    }

    @Test("first launch, same version, and downgrade: no upgrade notice")
    func noUpgradeNoticeWithoutAForwardMove() {
        let mtime = Date(timeIntervalSince1970: 2_000_000)

        // First ever launch: nothing to have upgraded from.
        let first = freshDefaults()
        #expect(InstallTracker(
            currentVersion: "0.14.1",
            currentInfoPlistMtime: mtime,
            currentBundleURL: installedBundleURL,
            defaults: first
        ).upgradedFrom == nil)

        // Routine relaunch.
        let same = freshDefaults()
        same.set("0.14.1", forKey: InstallTracker.lastSeenVersionKey)
        #expect(InstallTracker(
            currentVersion: "0.14.1",
            currentInfoPlistMtime: mtime,
            currentBundleURL: installedBundleURL,
            defaults: same
        ).upgradedFrom == nil)

        // Deliberate rollback: "Updated to v0.13.1" would read as a bug.
        let back = freshDefaults()
        back.set("0.14.1", forKey: InstallTracker.lastSeenVersionKey)
        #expect(InstallTracker(
            currentVersion: "0.13.1",
            currentInfoPlistMtime: mtime,
            currentBundleURL: installedBundleURL,
            defaults: back
        ).upgradedFrom == nil)
    }

    /// A dev build outside /Applications still upgraded — the location gate
    /// belongs to the failed-Replace signal, not to this one.
    @Test("an upgrade outside /Applications still reports the notice")
    func upgradeNoticeIgnoresBundleLocation() {
        let defaults = freshDefaults()
        defaults.set("0.13.1", forKey: InstallTracker.lastSeenVersionKey)
        let tracker = InstallTracker(
            currentVersion: "0.14.1",
            currentInfoPlistMtime: Date(timeIntervalSince1970: 2_000_000),
            currentBundleURL: URL(fileURLWithPath: "/Users/x/build/Rapid-MLX Desktop.app"),
            defaults: defaults
        )
        #expect(tracker.upgradedFrom == "0.13.1")
    }

    @Test("dismissUpgradeNotice(): clears the notice; the next launch stays clean")
    func dismissUpgradeNoticeIsSticky() {
        let defaults = freshDefaults()
        let mtime = Date(timeIntervalSince1970: 2_000_000)
        defaults.set(mtime, forKey: InstallTracker.lastSeenMtimeKey)
        defaults.set("0.13.1", forKey: InstallTracker.lastSeenVersionKey)

        let tracker = InstallTracker(
            currentVersion: "0.14.1",
            currentInfoPlistMtime: mtime.addingTimeInterval(3_600),
            currentBundleURL: installedBundleURL,
            defaults: defaults
        )
        #expect(tracker.upgradedFrom == "0.13.1")
        tracker.dismissUpgradeNotice()
        #expect(tracker.upgradedFrom == nil)

        // Dismissing recorded the acknowledgement, so the banner cannot come
        // back for a version the user has seen.
        let relaunch = InstallTracker(
            currentVersion: "0.14.1",
            currentInfoPlistMtime: mtime.addingTimeInterval(3_600),
            currentBundleURL: installedBundleURL,
            defaults: defaults
        )
        #expect(relaunch.upgradedFrom == nil)
    }

    @Test("release-notes link is built only for a real version")
    func releaseNotesLinkGrammar() {
        #expect(
            WhatsNewBanner.releaseNotesURL(for: "0.14.1")?.absoluteString
                == "https://github.com/raullenchai/Rapid-MLX/releases/tag/rapid-mac-v0.14.1"
        )
        // No tag exists for these, so no link is offered rather than a 404.
        #expect(WhatsNewBanner.releaseNotesURL(for: "0.14.1-dev") == nil)
        #expect(WhatsNewBanner.releaseNotesURL(for: "dev") == nil)
        #expect(WhatsNewBanner.releaseNotesURL(for: "") == nil)
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
    @Test("The upgrade notice survives a quit before it is read")
    func upgradeNoticeIsStickyUntilAcknowledged() {
        // Adversarial review round 5 (codex, nit): the notice used to be
        // consumed by the launch that detected it, because it compared against
        // `lastSeenVersion` — which has to advance every launch for the
        // failed-Replace baseline. Quitting or crashing before reading it lost
        // it for good, contradicting the sticky-banner contract.
        let defaults = freshDefaults()
        let bundle = installedBundleURL

        // Launch 1: first ever. Nothing to announce, nothing owed.
        #expect(InstallTracker(
            currentVersion: "0.13.1", currentInfoPlistMtime: Date(),
            currentBundleURL: bundle, defaults: defaults).upgradedFrom == nil)
        #expect(defaults.string(forKey: InstallTracker.pendingUpgradeNoticeFromKey) == nil)

        // Launch 2: upgraded, notice shown — and the user quits without
        // touching it.
        let upgraded = InstallTracker(
            currentVersion: "0.14.1", currentInfoPlistMtime: Date(),
            currentBundleURL: bundle, defaults: defaults)
        #expect(upgraded.upgradedFrom == "0.13.1")

        // Launch 3: still pending, still announcing the same origin version.
        let relaunched = InstallTracker(
            currentVersion: "0.14.1", currentInfoPlistMtime: Date(),
            currentBundleURL: bundle, defaults: defaults)
        #expect(relaunched.upgradedFrom == "0.13.1")

        // Dismissing (or opening the notes) is what consumes it.
        relaunched.dismissUpgradeNotice()
        #expect(relaunched.upgradedFrom == nil)
        #expect(defaults.string(forKey: InstallTracker.pendingUpgradeNoticeFromKey) == nil)
        let afterDismiss = InstallTracker(
            currentVersion: "0.14.1", currentInfoPlistMtime: Date(),
            currentBundleURL: bundle, defaults: defaults)
        #expect(afterDismiss.upgradedFrom == nil)
    }

    @Test("Existing users get the notice on the upgrade that ships it")
    func upgradeNoticeMigratesFromLastSeenVersion() {
        // Adversarial review round 7 (codex, blocking): the acknowledgement key
        // does not exist yet for anyone already running the app, and seeding it
        // to the current version swallowed the notice on exactly the upgrade
        // that introduces the feature. It falls back to the last-seen version.
        let defaults = freshDefaults()
        defaults.set("0.14.1", forKey: InstallTracker.lastSeenVersionKey)
        defaults.set(Date(timeIntervalSince1970: 2_000_000), forKey: InstallTracker.lastSeenMtimeKey)
        #expect(defaults.string(forKey: InstallTracker.pendingUpgradeNoticeFromKey) == nil)

        let tracker = InstallTracker(
            currentVersion: "0.14.2",
            currentInfoPlistMtime: Date(timeIntervalSince1970: 2_003_600),
            currentBundleURL: installedBundleURL,
            defaults: defaults
        )
        #expect(tracker.upgradedFrom == "0.14.1")
        // Recorded as owed, so the next launch shows the same notice.
        #expect(defaults.string(forKey: InstallTracker.pendingUpgradeNoticeFromKey) == "0.14.1")
    }

    @Test("A fresh install stays silent, and the upgrade after it does not")
    func freshInstallHasNothingToAnnounce() {
        let defaults = freshDefaults()
        let first = InstallTracker(
            currentVersion: "0.14.1", currentInfoPlistMtime: Date(timeIntervalSince1970: 2_000_000),
            currentBundleURL: installedBundleURL, defaults: defaults)
        #expect(first.upgradedFrom == nil)
        // The failed-Replace baseline the first launch DID write is enough for
        // the next upgrade to compare against.
        let next = InstallTracker(
            currentVersion: "0.14.2", currentInfoPlistMtime: Date(timeIntervalSince1970: 2_003_600),
            currentBundleURL: installedBundleURL, defaults: defaults)
        #expect(next.upgradedFrom == "0.14.1")
    }

    @Test("Both install banners can fire at once; the stale-bundle one wins")
    func upgradeNoticeYieldsToFailedReplace() {
        // Adversarial review round 7 (codex, nit): the two stopped being
        // mutually exclusive when the notice became sticky. "Updated to
        // v0.14.2" over "your update didn't install" contradicts itself.
        let defaults = freshDefaults()
        let mtime = Date(timeIntervalSince1970: 2_000_000)
        defaults.set(mtime, forKey: InstallTracker.lastSeenMtimeKey)
        defaults.set("0.14.1", forKey: InstallTracker.lastSeenVersionKey)

        // Launch 1: upgraded to 0.14.2, notice pending (never dismissed).
        _ = InstallTracker(
            currentVersion: "0.14.2", currentInfoPlistMtime: mtime.addingTimeInterval(60),
            currentBundleURL: installedBundleURL, defaults: defaults)
        // Launch 2: a Finder Replace touched the bundle but the version stuck.
        let tracker = InstallTracker(
            currentVersion: "0.14.2", currentInfoPlistMtime: mtime.addingTimeInterval(3_600),
            currentBundleURL: installedBundleURL, defaults: defaults)
        #expect(tracker.failedReplaceDetected)
        #expect(tracker.upgradedFrom == "0.14.1")
        // Both true, so the view-level precedence is what keeps them from
        // stacking — see `WhatsNewBanner.body`.
    }

    @Test("A rollback cancels an owed notice even from further back")
    func rollbackCancelsAPendingNotice() {
        // Adversarial review round 8 (codex, blocking): the pending notice was
        // kept whenever the current version beat the version it was ABOUT, so
        // 0.13.1 -> 0.15.0 -> rollback to 0.14.0 announced "Updated to
        // v0.14.0" over a rollback.
        let defaults = freshDefaults()
        let mtime = Date(timeIntervalSince1970: 2_000_000)
        defaults.set(mtime, forKey: InstallTracker.lastSeenMtimeKey)
        defaults.set("0.13.1", forKey: InstallTracker.lastSeenVersionKey)

        // Upgrade to 0.15.0; the notice is owed and never read.
        let upgraded = InstallTracker(
            currentVersion: "0.15.0", currentInfoPlistMtime: mtime.addingTimeInterval(60),
            currentBundleURL: installedBundleURL, defaults: defaults)
        #expect(upgraded.upgradedFrom == "0.13.1")

        // Roll back to 0.14.0: still ahead of 0.13.1, but this launch went
        // backwards, so there is nothing to celebrate.
        let rolledBack = InstallTracker(
            currentVersion: "0.14.0", currentInfoPlistMtime: mtime.addingTimeInterval(120),
            currentBundleURL: installedBundleURL, defaults: defaults)
        #expect(rolledBack.upgradedFrom == nil)
        #expect(defaults.string(forKey: InstallTracker.pendingUpgradeNoticeFromKey) == nil)
    }

}
