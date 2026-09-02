import Foundation
import Testing
@testable import Rapid

@MainActor
@Suite("Update discovery card presentation")
struct UpdateDiscoveryCardTests {
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
