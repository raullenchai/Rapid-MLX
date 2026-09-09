import Foundation
import Testing
@testable import Rapid

@Suite("About candidate identity")
struct AboutPanelCandidateIdentityTests {
    @Test("required engine attribution is contextualized in About")
    func engineAttribution() {
        #expect(AboutPanel.mtplxAttribution == "Powered by MTPLX")
        #expect(AboutPanel.mtplxURL == "https://github.com/youssofal/mtplx")
    }

    @Test("About renders the required attribution and Privacy does not")
    func engineAttributionPlacement() throws {
        let testsDirectory = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
        let rapidMacRoot = testsDirectory
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let aboutSource = try String(
            contentsOf: rapidMacRoot.appendingPathComponent("Sources/Rapid/AboutPanel.swift"),
            encoding: .utf8
        )
        let settingsSource = try String(
            contentsOf: rapidMacRoot.appendingPathComponent("Sources/Rapid/UI/SettingsView.swift"),
            encoding: .utf8
        )

        #expect(aboutSource.contains("Link(AboutPanel.mtplxAttribution"))
        #expect(aboutSource.contains("destination: URL(string: mtplxURL)!"))
        #expect(aboutSource.contains(".accessibilityIdentifier(\"About.Link.MTPLX\")"))
        #expect(!settingsSource.contains("Powered by MTPLX"))
        #expect(!settingsSource.contains("Settings.Privacy.Link.MTPLX"))
    }

    @Test("release build keeps the stable version line")
    func releaseVersionLine() {
        #expect(
            AboutPanel.versionLine(
                version: "0.13.1",
                build: "166",
                candidateIdentity: nil
            ) == "Version 0.13.1 (166)"
        )
    }

    @Test("candidate build exposes its exact source identity")
    func candidateVersionLine() {
        #expect(
            AboutPanel.versionLine(
                version: "0.13.1",
                build: "166",
                candidateIdentity: "candidate-a6b820cf"
            ) == "Version 0.13.1 (166) · candidate-a6b820cf"
        )
    }
}
