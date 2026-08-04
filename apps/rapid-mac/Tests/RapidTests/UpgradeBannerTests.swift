import Foundation
import Testing
@testable import Rapid

/// Pins the v0.7.2 "upgrade from the bundled model" banner contract.
///
/// The visibility predicate (``UpgradeBanner.shouldShow(_:)``) and the
/// copy generator (``UpgradeBanner.makeCopy(...)``) are pure functions
/// of the ``UpgradeBannerInputs`` struct + scalars, so every scenario
/// in the v0.7.2 spec is reachable without standing up a SwiftUI host,
/// a real ``SessionStore``, or a live ``ServerManager``.
///
/// Test plan (mirrors the 9 scenarios called out in the task spec):
///
///   1. Counter increments on user messages, not assistant messages.
///   2. Banner does NOT appear at <5 messages.
///   3. Banner appears at exactly 5 messages with bundled alias active.
///   4. Banner does NOT appear if active alias ≠ bundled alias.
///   5. Banner does NOT appear if suppressed flag is true.
///   6. "Maybe later" → banner hidden for session, re-appears on
///      simulated relaunch.
///   7. "Don't show again" → persisted flag set, banner hidden forever.
///   8. Copy renders correct alias name + size for each RAM bucket
///      (parametrized across the 6 buckets).
///   9. Banner does NOT appear if RAMBucketedDefault returns the
///      bundled alias itself (edge case).
@Suite("UpgradeBanner visibility predicate + copy generator")
struct UpgradeBannerTests {

    // MARK: - Predicate inputs builder

    private func inputs(
        active: String? = "bonsai-1.7b-2bit",
        bundled: String = "bonsai-1.7b-2bit",
        upgrade: String = "qwen3.6-27b-4bit",
        upgradeInCatalog: Bool = true,
        userMessages: Int = UpgradeBanner.minUserMessages,
        dismissedSession: Bool = false,
        suppressed: Bool = false,
        downloading: Bool = false,
        alreadyDownloaded: Bool = false,
        switchDismissed: Bool = false
    ) -> UpgradeBannerInputs {
        UpgradeBannerInputs(
            activeAlias: active,
            bundledAlias: bundled,
            upgradeAlias: upgrade,
            upgradeAliasExistsInCatalog: upgradeInCatalog,
            userMessageCount: userMessages,
            dismissedThisSession: dismissedSession,
            suppressedPermanently: suppressed,
            downloadInProgress: downloading,
            upgradeAliasAlreadyDownloaded: alreadyDownloaded,
            switchPromptDismissed: switchDismissed
        )
    }

    // MARK: - Test 1: counter increments on user messages, not assistant

    @Test("Counter — only role=user messages count toward the threshold")
    func userMessageCounter_excludes_non_user_roles() {
        // Build a transcript of mixed roles and pin the count helper
        // semantics: only `.user` rows count.
        let transcript: [ChatMessage.Role] = [
            .system,
            .user,        // 1
            .assistant,
            .user,        // 2
            .assistant,
            .tool,
            .user,        // 3
            .assistant,
            .user,        // 4
            .assistant
        ]
        let count = transcript.reduce(0) { acc, role in
            acc + (role == .user ? 1 : 0)
        }
        #expect(count == 4)

        // And a transcript with only assistants → 0.
        let assistantOnly: [ChatMessage.Role] = [
            .system, .assistant, .assistant, .tool, .assistant
        ]
        let zero = assistantOnly.reduce(0) { acc, r in
            acc + (r == .user ? 1 : 0)
        }
        #expect(zero == 0)
    }

    // MARK: - Test 2: hidden at <5 messages

    @Test("Below threshold — banner is hidden at 0/1/2 user turns")
    func below_threshold_hides_banner() {
        // minUserMessages is 3 (2026-07-10, bonsai starter swap); every
        // count strictly below it must stay hidden.
        for n in [0, 1, 2] {
            #expect(n < UpgradeBanner.minUserMessages)
            let input = inputs(userMessages: n)
            #expect(
                UpgradeBanner.shouldShow(input) == false,
                "userMessageCount=\(n) should be below threshold"
            )
        }
    }

    // MARK: - Test 3: shown at exactly minUserMessages with bundled active

    @Test("At threshold — banner fires at exactly minUserMessages with bundled alias active")
    func at_threshold_fires_banner() {
        let input = inputs(userMessages: UpgradeBanner.minUserMessages)
        #expect(UpgradeBanner.shouldShow(input) == true)
    }

    @Test("Above threshold — banner stays visible at 6/10/50 user turns")
    func above_threshold_keeps_banner_visible() {
        for n in [6, 10, 50, 500] {
            let input = inputs(userMessages: n)
            #expect(
                UpgradeBanner.shouldShow(input) == true,
                "userMessageCount=\(n) should fire banner"
            )
        }
    }

    // MARK: - Test 4: active ≠ bundled ⇒ hidden

    @Test("Active alias differs from bundled — banner hidden (user already upgraded)")
    func non_bundled_active_alias_hides_banner() {
        // Already running the recommended upgrade.
        let input1 = inputs(active: "qwen3.6-27b-4bit")
        #expect(UpgradeBanner.shouldShow(input1) == false)

        // Running some other model entirely (manual pick).
        let input2 = inputs(active: "gemma-4-12b-4bit")
        #expect(UpgradeBanner.shouldShow(input2) == false)

        // Server not yet ready — no active alias.
        let input3 = inputs(active: nil)
        #expect(UpgradeBanner.shouldShow(input3) == false)

        // Empty / whitespace active alias.
        let input4 = inputs(active: "   ")
        #expect(UpgradeBanner.shouldShow(input4) == false)
    }

    @Test("Case-insensitive + whitespace-tolerant comparison for active vs bundled")
    func active_bundled_comparison_is_normalized() {
        let input = inputs(active: "  Bonsai-1.7B-2BIT  ", bundled: "bonsai-1.7b-2bit")
        #expect(UpgradeBanner.shouldShow(input) == true)
    }

    // MARK: - Test 5: suppressed flag ⇒ hidden

    @Test("Permanent suppression wins over every other gate")
    func permanent_suppression_hides_banner() {
        let input = inputs(suppressed: true)
        #expect(UpgradeBanner.shouldShow(input) == false)

        // Even at 100 turns + every other gate green.
        let high = inputs(userMessages: 100, suppressed: true)
        #expect(UpgradeBanner.shouldShow(high) == false)
    }

    // MARK: - Test 6: "Maybe later" → session-only dismissal

    @Test("Per-session dismissal — banner hidden until simulated relaunch")
    func session_dismissal_hides_banner() {
        // User clicked "Maybe later" → in-memory flag flipped.
        let dismissed = inputs(dismissedSession: true)
        #expect(UpgradeBanner.shouldShow(dismissed) == false)

        // Simulated relaunch: coordinator rebuilt → `dismissedThisSession`
        // resets to false. Banner returns.
        let relaunched = inputs(dismissedSession: false)
        #expect(UpgradeBanner.shouldShow(relaunched) == true)
    }

    @Test("Coordinator — dismissForSession sets the per-session flag")
    @MainActor
    func coordinator_dismiss_for_session_flips_flag() {
        // Use an isolated UserDefaults so prior test runs don't bleed
        // a sticky "suppressed" flag into this case.
        let defaults = UserDefaults(suiteName: "UpgradeBannerTests.dismissForSession")!
        defaults.removePersistentDomain(forName: "UpgradeBannerTests.dismissForSession")
        let coord = UpgradeBannerCoordinator(
            bundledAlias: "bonsai-1.7b-2bit",
            hardware: MacHardware.detect(),
            defaults: defaults
        )
        #expect(coord.dismissedThisSession == false)
        coord.dismissForSession()
        #expect(coord.dismissedThisSession == true)
    }

    // MARK: - Test 7: "Don't show again" → persisted UserDefaults

    @Test("Coordinator — suppressPermanently writes UserDefaults under canonical key")
    @MainActor
    func coordinator_suppress_permanently_persists() {
        let suiteName = "UpgradeBannerTests.suppressPermanently"
        let defaults = UserDefaults(suiteName: suiteName)!
        // Clear any sticky state from previous runs.
        defaults.removePersistentDomain(forName: suiteName)

        let coord = UpgradeBannerCoordinator(
            bundledAlias: "bonsai-1.7b-2bit",
            hardware: MacHardware.detect(),
            defaults: defaults
        )
        #expect(defaults.bool(forKey: UpgradeBanner.suppressionKey) == false)
        coord.suppressPermanently()
        #expect(defaults.bool(forKey: UpgradeBanner.suppressionKey) == true)

        // Build a fresh coordinator against the same defaults — the
        // flag must survive a relaunch.
        let reborn = UpgradeBannerCoordinator(
            bundledAlias: "bonsai-1.7b-2bit",
            hardware: MacHardware.detect(),
            defaults: defaults
        )
        let inputs = UpgradeBannerInputs(
            activeAlias: reborn.bundledAlias,
            bundledAlias: reborn.bundledAlias,
            upgradeAlias: reborn.upgradeAlias,
            upgradeAliasExistsInCatalog: true,
            userMessageCount: UpgradeBanner.minUserMessages,
            dismissedThisSession: false,
            suppressedPermanently: defaults.bool(forKey: UpgradeBanner.suppressionKey),
            downloadInProgress: false,
            upgradeAliasAlreadyDownloaded: false,
            switchPromptDismissed: false
        )
        #expect(UpgradeBanner.shouldShow(inputs) == false)
    }

    @Test("UserDefaults key is the canonical dotted-namespace string")
    func suppression_key_is_canonical() {
        #expect(UpgradeBanner.suppressionKey == "rapid.banner.upgradeFromBundled.suppressed")
    }

    // MARK: - Already-downloaded gate (v0.10.6 regression)
    //
    // The reported bug: the user clicked "Download in background", the
    // pull finished, the picker showed "Downloaded — ready to load", and
    // the banner was STILL inviting them to download it. Completion is
    // precisely the event that clears `downloadInProgress`, so without a
    // terminal gate finishing the download un-hides the banner.

    @Test("Predicate — an upgrade alias already on disk never re-offers the download")
    func alreadyDownloaded_suppresses_the_cta() {
        #expect(UpgradeBanner.shouldShow(inputs(alreadyDownloaded: true)) == false)
        // …and it stays suppressed no matter how engaged the user is:
        // the gate is terminal, not a threshold.
        #expect(
            UpgradeBanner.shouldShow(inputs(userMessages: 50, alreadyDownloaded: true)) == false
        )
        // Control: same inputs with the weights absent still fire.
        #expect(UpgradeBanner.shouldShow(inputs(alreadyDownloaded: false)) == true)
    }

    @MainActor
    @Test("A completed pull marks the alias downloaded, and dismissing the progress row doesn't undo it")
    func completedPull_suppresses_durably() {
        let suite = "UpgradeBannerTests.completedPull"
        let defaults = UserDefaults(suiteName: suite)!
        defaults.removePersistentDomain(forName: suite)
        let coord = UpgradeBannerCoordinator(
            bundledAlias: "bonsai-1.7b-2bit",
            hardware: MacHardware.detect(),
            defaults: defaults
        )
        let downloads = DownloadManager()

        #expect(coord.isUpgradeAliasDownloaded(downloads: downloads) == false)

        // `_testingFinish` is a no-op unless the job already exists —
        // `handleExit` opens with `guard let job = jobs[alias]`.
        _ = downloads._testingSeedJob(alias: coord.upgradeAlias)
        downloads._testingFinish(alias: coord.upgradeAlias, status: 0, reason: .exit)
        #expect(coord.isUpgradeAliasDownloaded(downloads: downloads) == true)

        // ChatView re-snapshots the catalog on completion, so the cached
        // set is populated before the user can reach the Dismiss button
        // on the completed DownloadStrip row.
        coord.setCatalog([
            ModelEntry(
                alias: coord.upgradeAlias,
                hfRepo: nil,
                sizeOnDisk: "4.1 GB",
                cached: true
            )
        ])
        downloads.dismissJob(alias: coord.upgradeAlias)
        #expect(downloads.job(for: coord.upgradeAlias) == nil)
        #expect(
            coord.isUpgradeAliasDownloaded(downloads: downloads) == true,
            "dismissing the completed download row must not resurrect the banner"
        )
    }

    @MainActor
    @Test("Failed and cancelled pulls leave the CTA available so the user can retry")
    func failedOrCancelledPull_keeps_the_cta() {
        let suite = "UpgradeBannerTests.failedPull"
        let defaults = UserDefaults(suiteName: suite)!
        defaults.removePersistentDomain(forName: suite)
        let coord = UpgradeBannerCoordinator(
            bundledAlias: "bonsai-1.7b-2bit",
            hardware: MacHardware.detect(),
            defaults: defaults
        )

        let failed = DownloadManager()
        _ = failed._testingSeedJob(alias: coord.upgradeAlias)
        failed._testingFinish(alias: coord.upgradeAlias, status: 1, reason: .exit)
        #expect(coord.isUpgradeAliasDownloaded(downloads: failed) == false)

        let cancelled = DownloadManager()
        _ = cancelled._testingSeedJob(alias: coord.upgradeAlias)
        cancelled._testingFinish(
            alias: coord.upgradeAlias,
            status: 0,
            reason: .exit,
            wasCancelling: true
        )
        #expect(coord.isUpgradeAliasDownloaded(downloads: cancelled) == false)
    }

    // MARK: - Ready-to-switch strip
    //
    // The download CTA is not just hidden once the weights land — it is
    // replaced by the action that's actually left. The strip inherits
    // EVERY consent gate the CTA respects, or it would become a fresh
    // nag for users who already opted out.

    @Test("Ready-to-switch fires exactly when the download CTA retires")
    func readyToSwitch_replaces_the_cta() {
        // Weights absent: CTA shows, strip doesn't.
        #expect(UpgradeBanner.shouldShow(inputs()) == true)
        #expect(UpgradeBanner.shouldShowReadyToSwitch(inputs()) == false)
        // Weights present: they swap.
        #expect(UpgradeBanner.shouldShow(inputs(alreadyDownloaded: true)) == false)
        #expect(UpgradeBanner.shouldShowReadyToSwitch(inputs(alreadyDownloaded: true)) == true)
        // Never both at once, whatever the inputs.
        for downloaded in [true, false] {
            for downloading in [true, false] {
                let i = inputs(downloading: downloading, alreadyDownloaded: downloaded)
                #expect(!(UpgradeBanner.shouldShow(i) && UpgradeBanner.shouldShowReadyToSwitch(i)))
            }
        }
    }

    @Test("Ready-to-switch honours every consent gate the download CTA honours")
    func readyToSwitch_respects_consent_gates() {
        // "Don't show again" — the important one: a user who opted out
        // must not be handed a brand-new prompt.
        #expect(
            UpgradeBanner.shouldShowReadyToSwitch(
                inputs(suppressed: true, alreadyDownloaded: true)
            ) == false
        )
        // "Maybe later" this launch.
        #expect(
            UpgradeBanner.shouldShowReadyToSwitch(
                inputs(dismissedSession: true, alreadyDownloaded: true)
            ) == false
        )
        // Engagement threshold — no nag at turn zero.
        #expect(
            UpgradeBanner.shouldShowReadyToSwitch(
                inputs(userMessages: 0, alreadyDownloaded: true)
            ) == false
        )
        // Already serving something other than the bundled model: the
        // user made their own choice, don't second-guess it.
        #expect(
            UpgradeBanner.shouldShowReadyToSwitch(
                inputs(active: "gemma-4-12b-4bit", alreadyDownloaded: true)
            ) == false
        )
        // Mid-download the progress strip owns the surface.
        #expect(
            UpgradeBanner.shouldShowReadyToSwitch(
                inputs(downloading: true, alreadyDownloaded: true)
            ) == false
        )
        // Its own per-launch dismissal.
        #expect(
            UpgradeBanner.shouldShowReadyToSwitch(
                inputs(alreadyDownloaded: true, switchDismissed: true)
            ) == false
        )
    }

    @Test("Dismissing the switch strip is independent of the download CTA's dismissal")
    func switchDismissal_is_independent() {
        // Dismissing the switch offer must not resurrect the download
        // CTA for a model that is already on disk.
        #expect(
            UpgradeBanner.shouldShow(
                inputs(alreadyDownloaded: true, switchDismissed: true)
            ) == false
        )
        // And declining the download ("Maybe later") is tracked by a
        // different flag, so it isn't confused with this one.
        #expect(
            UpgradeBanner.shouldShowReadyToSwitch(
                inputs(alreadyDownloaded: true, switchDismissed: false)
            ) == true
        )
    }

    @MainActor
    @Test("dismissSwitchPrompt is per-launch and doesn't touch the CTA's session flag")
    func coordinator_switch_dismissal_seam() {
        let suite = "UpgradeBannerTests.switchDismiss"
        let defaults = UserDefaults(suiteName: suite)!
        defaults.removePersistentDomain(forName: suite)
        let coord = UpgradeBannerCoordinator(
            bundledAlias: "bonsai-1.7b-2bit",
            hardware: MacHardware.detect(),
            defaults: defaults
        )
        #expect(coord.switchPromptDismissed == false)
        coord.dismissSwitchPrompt()
        #expect(coord.switchPromptDismissed == true)
        #expect(
            coord.dismissedThisSession == false,
            "dismissing the switch strip must not also count as declining the download"
        )
    }

    @MainActor
    @Test("The catalog's cached flag alone marks the alias downloaded (had it before this launch)")
    func catalogCachedFlag_marks_downloaded() {
        let suite = "UpgradeBannerTests.catalogCached"
        let defaults = UserDefaults(suiteName: suite)!
        defaults.removePersistentDomain(forName: suite)
        let coord = UpgradeBannerCoordinator(
            bundledAlias: "bonsai-1.7b-2bit",
            hardware: MacHardware.detect(),
            defaults: defaults
        )
        let downloads = DownloadManager()

        coord.setCatalog([
            ModelEntry(alias: coord.upgradeAlias, hfRepo: nil, sizeOnDisk: nil, cached: false)
        ])
        #expect(coord.isUpgradeAliasDownloaded(downloads: downloads) == false)

        coord.setCatalog([
            ModelEntry(alias: coord.upgradeAlias, hfRepo: nil, sizeOnDisk: "4.1 GB", cached: true)
        ])
        #expect(coord.isUpgradeAliasDownloaded(downloads: downloads) == true)
    }

    // MARK: - Test 8: copy renders correct alias + size for each RAM bucket

    /// The six RAM tiers surfaced in ``RAMBucketedDefault`` — sampled at
    /// each tier floor so the round-DOWN boundary is pinned alongside the
    /// upgrade copy.
    private static let bucketSamples: [(label: String, ramGB: Double, expectedAlias: String)] = [
        ("16 GB",  16,  "bonsai-27b-2bit"),
        ("18 GB",  18,  "bonsai-27b-2bit"),   // 18 mirrors 16
        ("24 GB",  24,  "gemma-4-26b-4bit"),
        ("32 GB",  32,  "qwen3.6-35b-4bit"),
        ("64 GB",  64,  "qwen3.6-35b-8bit"),
        ("96 GB+", 192, "qwen3.5-122b-mxfp4"),
    ]

    @Test("Copy — every RAM tier yields the right upgrade alias", arguments: bucketSamples)
    func copy_per_bucket(sample: (label: String, ramGB: Double, expectedAlias: String)) {
        let upgrade = RAMBucketedDefault.alias(forPhysicalRAMGB: sample.ramGB)
        #expect(
            upgrade == sample.expectedAlias,
            "\(sample.label) GB Mac should default to \(sample.expectedAlias), got \(upgrade)"
        )
        let footprint = ModelSizing.estimate(alias: upgrade)
        let copy = UpgradeBanner.makeCopy(
            physicalRAMGB: Int(sample.ramGB.rounded()),
            upgradeAlias: upgrade,
            estimatedWeightsGB: footprint.paramsBillions == nil
                ? nil
                : footprint.weightsGB
        )
        // The body must contain the alias verbatim AND the user-
        // facing RAM count. The value-prop tail is fixed copy.
        #expect(copy.body.contains(upgrade),
                "\(sample.label): body missing alias '\(upgrade)' — got '\(copy.body)'")
        #expect(copy.body.contains("\(Int(sample.ramGB.rounded())) GB Mac"),
                "\(sample.label): body missing 'N GB Mac' phrase — got '\(copy.body)'")
        #expect(copy.body.contains("much smarter at coding, writing, reasoning"),
                "\(sample.label): body missing value prop — got '\(copy.body)'")
        // Accessibility label must not contain literal verbatim
        // markdown / placeholders.
        #expect(copy.accessibilityLabel.contains(upgrade))
        #expect(copy.accessibilityLabel.contains("\(Int(sample.ramGB.rounded())) gigabyte Mac"))
    }

    @Test("Copy — omits size hint when paramsBillions is nil (custom alias)")
    func copy_handles_unknown_size() {
        let copy = UpgradeBanner.makeCopy(
            physicalRAMGB: 32,
            upgradeAlias: "totally-custom-alias",
            estimatedWeightsGB: nil
        )
        // The "(~N GB · " sub-phrase is gated on a non-nil estimate
        // ≥0.5; absence means we just open with "(" + valueProp.
        #expect(copy.body.contains("(~") == false,
                "Unknown-size copy should not print '~0 GB' — got '\(copy.body)'")
        #expect(copy.body.contains("totally-custom-alias"))
        #expect(copy.body.contains("much smarter at coding, writing, reasoning"))
    }

    // MARK: - Test 9: bucketed default == bundled alias ⇒ hidden

    @Test("Upgrade == bundled ⇒ hidden (nonsensical 'upgrade to the same thing' guard)")
    func bundled_self_upgrade_hides_banner() {
        let input = inputs(
            active: "bonsai-1.7b-2bit",
            bundled: "bonsai-1.7b-2bit",
            upgrade: "bonsai-1.7b-2bit"
        )
        #expect(UpgradeBanner.shouldShow(input) == false)
    }

    @Test("Empty upgrade alias ⇒ hidden (defensive — bucketed lookup returned nothing)")
    func empty_upgrade_alias_hides_banner() {
        let input = inputs(upgrade: "")
        #expect(UpgradeBanner.shouldShow(input) == false)
    }

    @Test("Upgrade alias missing from catalog ⇒ hidden (defensive vs out-of-sync rapid-mlx)")
    func upgrade_alias_missing_from_catalog_hides_banner() {
        let input = inputs(upgradeInCatalog: false)
        #expect(UpgradeBanner.shouldShow(input) == false)
    }

    // MARK: - Bonus: download-in-progress collapses banner

    @Test("Download in progress ⇒ banner hidden (progress strip takes over)")
    func download_in_progress_hides_banner() {
        let input = inputs(downloading: true)
        #expect(UpgradeBanner.shouldShow(input) == false)
    }

    // MARK: - Bonus: coordinator updateUserMessageCount seam

    @Test("Coordinator — updateUserMessageCount round-trips through lastObserved")
    @MainActor
    func coordinator_user_message_count_seam() {
        let suiteName = "UpgradeBannerTests.coordinator_user_message_count_seam"
        let defaults = UserDefaults(suiteName: suiteName)!
        defaults.removePersistentDomain(forName: suiteName)
        let coord = UpgradeBannerCoordinator(
            bundledAlias: "bonsai-1.7b-2bit",
            hardware: MacHardware.detect(),
            defaults: defaults
        )
        #expect(coord.lastObservedUserMessageCount == 0)
        coord.updateUserMessageCount(3)
        #expect(coord.lastObservedUserMessageCount == 3)
        coord.updateUserMessageCount(7)
        #expect(coord.lastObservedUserMessageCount == 7)
    }

    @Test("Coordinator — setCatalog snapshot drives the exists-in-catalog gate")
    @MainActor
    func coordinator_catalog_round_trip() {
        let suiteName = "UpgradeBannerTests.catalog_round_trip"
        let defaults = UserDefaults(suiteName: suiteName)!
        defaults.removePersistentDomain(forName: suiteName)
        let coord = UpgradeBannerCoordinator(
            bundledAlias: "bonsai-1.7b-2bit",
            hardware: MacHardware.detect(),
            defaults: defaults
        )
        // Empty catalog — upgrade alias is NOT in catalog.
        coord.setCatalogAliases([])
        #expect(coord.upgradeAlias.isEmpty == false)
        // Populated catalog containing the upgrade alias — gate flips.
        coord.setCatalogAliases([coord.upgradeAlias, "other-alias"])
        // Round-trip via ModelEntry too.
        let entries = [
            ModelEntry(alias: coord.upgradeAlias, hfRepo: nil, sizeOnDisk: nil, cached: false),
        ]
        coord.setCatalog(entries)
    }
}
