import AppKit
import SwiftUI
import Testing
@testable import Rapid

@MainActor
@Suite("Connect Tools session action")
struct ConnectToolsSessionActionTests {
    @Test("Session commands are offered only for supported agent rows")
    func sessionAction() {
        let view = ConnectToolsView(host: "127.0.0.1", port: 8000, bearer: "key",
                                    alias: .constant("model"),
                                    server: ServerManager(testingState: .idle),
                                    downloads: DownloadManager(), onClose: {})
        let ids = ["claude-code", "codex", "hermes", "pi", "opencode", "goose", "qwen-code", "crush", "aider", "cursor", "adapter"]
        let targets = ids.map { IntegrationTarget(id: $0, name: $0, kind: .adapterProfile, configPath: nil) }
        let rows = view.tools(for: targets)
        #expect(rows.count == ids.count)
        for row in rows {
            #expect(row.offersSessionCommand == ["claude-code", "codex", "hermes"].contains(row.id))
        }
        #expect(view.tools(for: []).allSatisfy { !$0.offersSessionCommand })
    }

    @Test("Terminal guidance keeps commands on this Mac")
    func terminalGuidance() {
        #expect(ConnectToolsView.terminalFootnote.contains("terminal on this Mac"))
        #expect(!ConnectToolsView.terminalFootnote.localizedCaseInsensitiveContains("another Mac"))
        #expect(!ConnectToolsView.terminalFootnote.localizedCaseInsensitiveContains("or Mac"))
    }
}
