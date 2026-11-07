import Foundation
import SwiftUI
import Testing
@testable import Rapid

/// Opt-in hardware dogfood. The harness supplies an owned, offline fixture and
/// holds the shared GPU lock for this test's complete server lifetime.
@Suite("Existing checkpoint real native user path", .serialized, TestTimeouts.hangProne) @MainActor
struct OpenedCheckpointRealUserTests {
    @Test(.enabled(if: ProcessInfo.processInfo.environment["RAPID_CHECKPOINT_REAL_PATH"] != nil))
    func registerChatAndForgetWithoutChangingFilesOrDownloadLocation() async throws {
        let env = ProcessInfo.processInfo.environment
        let path = try #require(env["RAPID_CHECKPOINT_REAL_PATH"])
        let binary = URL(fileURLWithPath: try #require(env["RAPID_CHECKPOINT_REAL_BINARY"]))
        let artifacts = URL(fileURLWithPath: try #require(env["RAPID_CHECKPOINT_REAL_ARTIFACTS"]))
        let selected = URL(fileURLWithPath: path)
        let fixture = OpenedCheckpointTests()
        let (prefs, domain) = fixture.defaults(); defer { prefs.removePersistentDomain(forName: domain) }
        let originalFolder = ModelsFolderPreference.storedPath(defaults: prefs)
        let weights = selected.appendingPathComponent("model.safetensors")
        let originalBytes = try weights.resourceValues(forKeys: [.fileSizeKey]).fileSize
        let server = ServerManager(testingState: .stopped, binaryPath: binary, sessionDefaults: prefs)
        let downloads = DownloadManager(binaryPath: binary, binaryLocator: { binary })
        let settings = GoldenStage(ScrollView {
            SettingsModelManagementPanel(checkpointDefaults: prefs, checkpointDirectory: { selected })
                .environment(server).environment(downloads)
        }, size: CGSize(width: 520, height: 720))
        try await settings.waitForIdentifier("Settings.ModelManagement.OpenCheckpoint")
        try settings.press("Settings.ModelManagement.OpenCheckpoint")
        try await settings.wait(for: "real checkpoint registered") { OpenedCheckpoints.paths(defaults: prefs) == [path] }
        try await settings.waitForIdentifierGone("Settings.ModelManagement.CheckingCheckpoint")
        let entries = await ModelCatalog.load(binary: binary, openedPaths: OpenedCheckpoints.paths(defaults: prefs))
        let entry = try #require(entries.first { $0.alias == path })
        #expect(entry.cached && entry.isExternal && entry.hfRepo == nil)
        #expect(downloads.jobs.isEmpty)
        try settings.writeScreenshot(to: artifacts.appendingPathComponent("registered-520.png"))
        let wide = GoldenStage(ScrollView {
            SettingsModelManagementPanel(checkpointDefaults: prefs, checkpointDirectory: { selected })
                .environment(server).environment(downloads)
        }, size: CGSize(width: 900, height: 720))
        try await wide.waitForIdentifier("Settings.ModelManagement.RemoveCheckpoint.\(path)")
        try wide.writeScreenshot(to: artifacts.appendingPathComponent("registered-900.png"))

        let backend = ChatBackendConfig(defaults: prefs); backend.harnessEnabled = false
        let sampling = SamplingConfig(defaults: prefs); sampling.maxTokens = 256
        let model = ChatViewModel(chatBackendConfig: backend, toolDefaults: prefs, sampling: sampling,
            customInstructions: CustomInstructionsConfig(defaults: prefs), server: server, persistsConversations: false)
        // The current composer is the actual launch surface: send calls
        // ensureServing, which must freshly certify the registered path.
        let chat = GoldenStage(CheckpointChatSurface(model: model, server: server, path: path)
            .environment(server).environment(downloads).environment(QuickstartCoordinator()),
            size: CGSize(width: 900, height: 720))
        do {
            try await chat.waitForIdentifier("Readiness.Action")
            try chat.press("Readiness.Action")
            try await chat.wait(for: "real engine ready", timeout: 60) { server.servingAlias == path }
            try await chat.waitForIdentifier(AutosizingTextView.composeAccessibilityIdentifier)
            try chat.setValue("Reply with the single word OK. /no_think", for: AutosizingTextView.composeAccessibilityIdentifier)
            try chat.submitComposer()
            try await chat.wait(for: "real user turn admitted") { model.messages.contains { $0.role == .user } }
            await model._testingWaitForCurrentTurn()
            #expect(model.lastError == nil)
            let answer = model.messages.filter { $0.role == .assistant }.map(\.content).joined(separator: "\n")
            #expect(!answer.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            #expect(server.servingAlias == path)
            #expect(model.messages.last(where: { $0.role == .assistant })?.status == .complete)
            try chat.writeScreenshot(to: artifacts.appendingPathComponent("real-chat-900.png"))
            try JSONSerialization.data(withJSONObject: ["path": path, "answer": answer,
                "external": entry.isExternal, "hf_repo_absent": entry.hfRepo == nil,
                "download_jobs": downloads.jobs.count], options: [.prettyPrinted, .sortedKeys])
                .write(to: artifacts.appendingPathComponent("native-answer.json"))
            try settings.press("Settings.ModelManagement.RemoveCheckpoint.\(path)")
            try await settings.wait(for: "reference forgotten") { OpenedCheckpoints.paths(defaults: prefs).isEmpty }
            #expect(FileManager.default.fileExists(atPath: weights.path))
            #expect(try weights.resourceValues(forKeys: [.fileSizeKey]).fileSize == originalBytes)
            #expect(ModelsFolderPreference.storedPath(defaults: prefs) == originalFolder)
            #expect(downloads.jobs.isEmpty)
        } catch {
            await server.stop()
            throw error
        }
        await server.stop()
        #expect(server.state == .stopped)
        try Data("owned server stopped; reference removed; checkpoint files retained".utf8)
            .write(to: artifacts.appendingPathComponent("native-complete.txt"))
    }
}


/// The current parent wiring: the readiness Load action routes through the
/// shared production adapter; observing the server unlocks the real composer.
private struct CheckpointChatSurface: View {
    let model: ChatViewModel
    let server: ServerManager
    let path: String
    var body: some View {
        ChatView(viewModel: model, server: server, alias: .constant(path),
            readiness: server.isModelResident(path) ? .ready(alias: path) : .needsStart(alias: path),
            onReadinessAction: { action in
                if case .start(let alias) = action {
                    Task { await ReadinessModelStart.perform(server, alias: alias, hfPath: nil) }
                }
            })
    }
}
