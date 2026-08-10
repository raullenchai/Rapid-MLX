import Testing
@testable import Rapid

@MainActor
@Suite("Update install presentation")
struct UpdateInstallPresentationTests {
    private func release(_ version: String) -> UpdateChecker.Release {
        UpdateChecker.Release(
            schemaVersion: 1,
            version: version,
            tagName: "rapid-mac-v\(version)",
            htmlURL: "https://github.com/machinefi/rapid-desktop/releases/tag/rapid-mac-v\(version)",
            notes: "notes",
            publishedAt: "2026-08-10T00:00:00Z",
            dmgURL: "https://dl.rapidmlx.com/rapid-mac/\(version)/rapid-mlx-desktop.dmg"
        )
    }

    @Test("current release renders up-to-date, not an unavailable installer")
    func currentReleaseIsCoherent() {
        let current = release("0.12.8")
        #expect(UpdateInstallView.resolvePresentation(
            availableUpdate: nil,
            latest: current,
            checking: false,
            lastError: nil
        ) == .upToDate)
    }

    @Test("a newer release remains installable")
    func newerReleaseWins() {
        let newer = release("0.12.9")
        #expect(UpdateInstallView.resolvePresentation(
            availableUpdate: newer,
            latest: newer,
            checking: false,
            lastError: nil
        ) == .update(newer))
    }
}
