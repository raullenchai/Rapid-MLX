import Foundation
import SwiftUI
import Testing
@testable import Rapid

@Suite("Experimental Local Search gate", .serialized)
struct LocalSearchFeatureGateTests {
    @Test("Search is opt-in and persists the choice")
    func defaultsOff() throws {
        let suite = "rapid.local-search-gate-tests.\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        #expect(!LocalSearchFeatureConfig.isEnabled(in: defaults))
        defaults.set(true, forKey: LocalSearchFeatureConfig.enabledKey)
        #expect(LocalSearchFeatureConfig.isEnabled(in: defaults))
        defaults.set(false, forKey: LocalSearchFeatureConfig.enabledKey)
        #expect(!LocalSearchFeatureConfig.isEnabled(in: defaults))
    }

    @MainActor
    @Test("Restoring or disabling hidden Search falls back to Chat")
    func restorationFallback() {
        let requested = SidebarSection.harnessRequested(environment: [
            "RAPID_GUI_GOLDEN_MODE": "1", "RAPID_GUI_INITIAL_SECTION": "search"
        ]) ?? .chat
        #expect(ContentView.sectionAfterLocalSearchGateChange(current: requested, enabled: false) == .chat)
        #expect(ContentView.sectionAfterLocalSearchGateChange(current: requested, enabled: true) == .search)
        #expect(ContentView.sectionAfterLocalSearchGateChange(current: .images, enabled: false) == .images)
    }

    @MainActor
    @Test("Full and compact sidebars hide Search by default and expose it only on opt-in")
    func sidebarGate() async throws {
        let chat = ChatViewModel(persistsConversations: false)
        for compact in [false, true] {
            do {
                let stage = GoldenStage(SidebarView(
                    isCompact: compact, selection: .constant(.chat),
                    chat: chat, onNewChat: {}, onSelectConversation: { _ in }
                ).frame(width: compact ? SidebarView.compactWidth : SidebarView.columnIdealWidth, height: 640))
                try await stage.waitForIdentifier("Sidebar.Images")
                #expect(!stage.identifiers().contains("Sidebar.Search"))
                #expect(!stage.identifiers().contains("Sidebar.ExperimentalHeader"))
            }
            do {
                let stage = GoldenStage(SidebarView(
                    isCompact: compact, selection: .constant(.search), localSearchEnabled: true,
                    chat: chat, onNewChat: {}, onSelectConversation: { _ in }
                ).frame(width: compact ? SidebarView.compactWidth : SidebarView.columnIdealWidth, height: 640))
                try await stage.waitForIdentifier("Sidebar.Search")
                if !compact {
                    try await stage.waitForIdentifier("Sidebar.ExperimentalHeader")
                    let ids = stage.identifiers()
                    #expect(try #require(ids.firstIndex(of: "Sidebar.ExperimentalHeader")) < #require(ids.firstIndex(of: "Sidebar.Search")))
                }
            }
        }
    }

    @Test("Production spawn requires both the stored opt-in and runtime capability")
    func spawnGate() throws {
        let suite = "rapid.local-search-spawn-tests.\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        for enabled in [false, true] {
            defaults.set(enabled, forKey: LocalSearchFeatureConfig.enabledKey)
            for supported in [false, true] {
                let capabilities = ServerRuntimeCapabilities.parse(serveHelp:
                    supported ? "--enable-local-search --local-search-dir" : "")
                let args = ServerManager.serveArguments(
                    alias: "qwen3.5-9b-4bit", host: "127.0.0.1", port: 8000,
                    capabilities: capabilities,
                    localSearchEnabled: LocalSearchFeatureConfig.isEnabled(in: defaults),
                    localSearchDirectory: "/tmp/Search Index")
                #expect(args.contains("--enable-local-search") == (enabled && supported))
                #expect(args.contains("--local-search-dir") == (enabled && supported))
                #expect(args.contains("/tmp/Search Index") == (enabled && supported))
                if enabled && supported {
                    #expect(args.suffix(3) == ["--enable-local-search", "--local-search-dir", "/tmp/Search Index"])
                }
            }
        }
    }

    @Test("Search restarts an active model exactly when the spawn flag changes")
    func restartGate() {
        let alias = "qwen3.5-9b-4bit"
        for state: ServerState in [.starting(alias: alias), .ready(alias: alias)] {
            #expect(LocalSearchFeatureConfig.restartAlias(enabled: true, enabledAtSpawn: false, state: state) == alias)
            #expect(LocalSearchFeatureConfig.restartAlias(enabled: false, enabledAtSpawn: true, state: state) == alias)
            #expect(LocalSearchFeatureConfig.restartAlias(enabled: true, enabledAtSpawn: true, state: state) == nil)
            #expect(LocalSearchFeatureConfig.restartAlias(enabled: false, enabledAtSpawn: false, state: state) == nil)
        }
        for state: ServerState in [.missing, .idle, .stopped, .crashed(alias: alias, message: "Exited")] {
            #expect(LocalSearchFeatureConfig.restartAlias(enabled: true, enabledAtSpawn: false, state: state) == nil)
            #expect(LocalSearchFeatureConfig.restartAlias(enabled: false, enabledAtSpawn: true, state: state) == nil)
        }
    }
}
