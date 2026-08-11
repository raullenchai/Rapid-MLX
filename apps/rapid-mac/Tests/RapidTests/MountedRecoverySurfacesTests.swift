import Foundation
import Testing
@testable import Rapid

/// #1588 was a reachability defect: every component compiled and had unit
/// tests, but no production view mounted it.  These source-shape pins bridge
/// that exact gap; GoldenFlow drives the interactive controls in the built app.
@Suite("#1588 formerly unmounted recovery surfaces")
struct MountedRecoverySurfacesTests {
    private static var sourceRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Sources/Rapid")
    }

    private func source(_ path: String) throws -> String {
        try String(contentsOf: Self.sourceRoot.appendingPathComponent(path), encoding: .utf8)
    }

    @Test("ContentView mounts update recovery, download progress, logs, and status")
    func contentViewMountsWindowLevelSurfaces() throws {
        let content = try source("UI/ContentView.swift")
        #expect(content.contains("FailedReplaceBanner()"))
        #expect(content.contains("DownloadStrip(downloads: downloads)"))
        #expect(content.contains("LogDrawer(server: server)"))
        #expect(content.contains("ServerStatusPill(state: server.state)"))
    }

    @Test("Every transcript message exposes the cross-block selection sheet")
    func messageRowMountsSelectTextSheet() throws {
        let chat = try source("UI/ChatView.swift")
        #expect(chat.contains("SelectTextSheet(text:"))
        #expect(chat.contains("actionIdentifier(\"SelectText\")"))
    }

    @Test("The obsolete pre-residency switch warning was removed")
    func staleSwitchWarningIsGone() {
        #expect(!FileManager.default.fileExists(
            atPath: Self.sourceRoot.appendingPathComponent("UI/ModelSwitchWarning.swift").path
        ))
    }
}
