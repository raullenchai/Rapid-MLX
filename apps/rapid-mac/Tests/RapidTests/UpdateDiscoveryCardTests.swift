import AppKit
import Foundation
import SwiftUI
import Testing
@testable import Rapid

@MainActor
@Suite("Update discovery card presentation")
struct UpdateDiscoveryCardTests {
    @Test("Update strip stays compact in narrow and normal windows, light and dark",
          arguments: [440.0, 720.0, 1024.0], [false, true])
    func compactStripHeight(width: Double, dark: Bool) {
        let release = UpdateChecker.Release(
            schemaVersion: 1, version: "0.16.6", tagName: "v0.16.6",
            htmlURL: "https://rapidmlx.com/desktop", notes: "", publishedAt: "", dmgURL: nil
        )
        for state in [(true, true), (true, false), (false, true)] {
            let host = NSHostingView(rootView: UpdateDiscoveryCard(
                release: release, sparkleEnabled: state.0, sparkleCanCheck: state.1,
                releaseURL: URL(string: release.htmlURL), onUpdate: {}, onDismiss: {}
            ).environment(\.colorScheme, dark ? .dark : .light).frame(width: width))
            let size = host.fittingSize
            #expect(abs(size.width - width) < 0.5)
            #expect(size.height >= 32 && size.height <= 40)
        }
    }

    @Test("A release published while running agrees across card, footer and Settings")
    func refreshedReleaseSurfaces() async throws {
        let clock = UpdateTestClock()
        let feed = UpdateTestFeed()
        let checker = UpdateChecker(currentVersion: "0.16.5", fetcher: { try await feed.fetch() },
                                    checksEnabled: { true }, now: { clock.date })
        await checker.check()
        #expect(checker.availableUpdate == nil)
        await feed.set()
        clock.advance(UpdateRefreshPolicy.refreshInterval + 1)
        await checker.refreshIfNeeded()
        let release = try #require(checker.availableUpdate)
        #expect(release.version == "0.16.6")
        #expect(DesktopVersionPill.resolve(currentVersion: checker.currentVersion,
                                          availableUpdate: release, latest: checker.latest)
                == .updateAvailable(current: "0.16.5", latest: "0.16.6"))
        #expect(SettingsView.resolveAppUpdateStatus(
            currentVersion: checker.currentVersion, availableUpdate: release,
            latest: checker.latest, checking: checker.checking,
            lastCheckedAt: checker.lastCheckedAt, lastError: checker.lastError
        ) == .available(version: "0.16.6"))
        for (dismissed, handedOff, expected) in [
            ("", nil as String?, true), ("0.16.5", nil, true),
            ("0.16.6", nil, false), ("", "0.16.6", false), ("", "0.16.5", true)
        ] {
            #expect(ContentView.shouldPresentUpdateCard(
                releaseVersion: release.version, dismissedVersion: dismissed,
                handedOffVersion: handedOff, onboardingVisible: false,
                blockingOverlayVisible: false, hasAction: true
            ) == expected)
        }
    }

    @Test("A new actionable release is presented")
    func presentsNewRelease() {
        #expect(ContentView.shouldPresentUpdateCard(
            releaseVersion: "0.13.4",
            dismissedVersion: "",
            handedOffVersion: nil,
            onboardingVisible: false,
            blockingOverlayVisible: false,
            hasAction: true
        ))
    }

    @Test("Dismissal applies only to the dismissed version")
    func dismissalIsVersionScoped() {
        #expect(!ContentView.shouldPresentUpdateCard(
            releaseVersion: "0.13.4",
            dismissedVersion: "0.13.4",
            handedOffVersion: nil,
            onboardingVisible: false,
            blockingOverlayVisible: false,
            hasAction: true
        ))
        #expect(ContentView.shouldPresentUpdateCard(
            releaseVersion: "0.13.5",
            dismissedVersion: "0.13.4",
            handedOffVersion: nil,
            onboardingVisible: false,
            blockingOverlayVisible: false,
            hasAction: true
        ))
    }

    @Test("Sparkle hand-off suppresses the duplicate card for this session")
    func handoffSuppressesDuplicateSurface() {
        #expect(!ContentView.shouldPresentUpdateCard(
            releaseVersion: "0.13.4",
            dismissedVersion: "",
            handedOffVersion: "0.13.4",
            onboardingVisible: false,
            blockingOverlayVisible: false,
            hasAction: true
        ))
    }

    @Test("Sparkle hand-off is recorded only after the check is accepted")
    func handoffRequiresAcceptance() {
        var handedOff: String?
        #expect(!ContentView.handOffUpdate(
            version: "0.13.4",
            start: { false },
            onAccepted: { handedOff = $0 }
        ))
        #expect(handedOff == nil)

        #expect(ContentView.handOffUpdate(
            version: "0.13.4",
            start: { true },
            onAccepted: { handedOff = $0 }
        ))
        #expect(handedOff == "0.13.4")
    }

    @Test("Manual download dismisses only after Launch Services accepts the URL")
    func manualDownloadRequiresSuccessfulOpen() throws {
        let url = try #require(URL(string: "https://example.com/releases/0.13.4"))
        var dismissals = 0
        #expect(!UpdateDiscoveryCard.openManualDownload(
            url,
            using: { _ in false },
            onOpened: { dismissals += 1 }
        ))
        #expect(dismissals == 0)

        #expect(UpdateDiscoveryCard.openManualDownload(
            url,
            using: { $0 == url },
            onOpened: { dismissals += 1 }
        ))
        #expect(dismissals == 1)
    }

    @Test("Onboarding and blocking overlays defer presentation")
    func defersForHigherPrioritySurfaces() {
        for state in [(true, false), (false, true)] {
            #expect(!ContentView.shouldPresentUpdateCard(
                releaseVersion: "0.13.4",
                dismissedVersion: "",
                handedOffVersion: nil,
                onboardingVisible: state.0,
                blockingOverlayVisible: state.1,
                hasAction: true
            ))
        }
    }

    @Test("A card without a safe action stays hidden")
    func hidesWithoutAction() {
        #expect(!ContentView.shouldPresentUpdateCard(
            releaseVersion: "0.13.4",
            dismissedVersion: "",
            handedOffVersion: nil,
            onboardingVisible: false,
            blockingOverlayVisible: false,
            hasAction: false
        ))
    }
}
