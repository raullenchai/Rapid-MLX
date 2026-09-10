import SwiftUI
import Testing
@testable import Rapid

@Suite("Composer resident unload")
struct ComposerResidentUnloadTests {
    @MainActor
    private final class UnloadProbe {
        var calls = 0
    }

    private func residentSnapshot(
        alias: String = "qwen3.5-4b-4bit",
        activeRequests: Int = 0,
        extraModel: Bool = false
    ) -> ModelResidencySnapshot {
        let gib = UInt64(1) << 30
        var models = [
            ResidentModelStatus(
                id: alias,
                modelPath: "mlx-community/\(alias)",
                aliases: [alias],
                modality: "text",
                state: "resident",
                pinned: true,
                primary: true,
                activeRequests: activeRequests,
                estimatedBytes: 6 * gib,
                measuredBytes: nil,
                idleSeconds: 12
            ),
        ]
        if extraModel {
            models.append(
                ResidentModelStatus(
                    id: "secondary-model",
                    modelPath: "test/secondary-model",
                    aliases: [],
                    modality: "image",
                    state: "resident",
                    pinned: false,
                    primary: false,
                    activeRequests: 0,
                    estimatedBytes: 2 * gib,
                    measuredBytes: nil,
                    idleSeconds: 8
                )
            )
        }
        return ModelResidencySnapshot(
            memoryLimitBytes: 14 * gib,
            memoryUsedBytes: extraModel ? 8 * gib : 6 * gib,
            memoryAvailableBytes: extraModel ? 6 * gib : 8 * gib,
            idleTTLSeconds: 900,
            loadsTotal: models.count,
            evictionsTotal: 0,
            models: models,
            audioLanes: []
        )
    }

    @MainActor
    private func stage(
        selectedAlias: String = "qwen3.5-4b-4bit",
        snapshot: ModelResidencySnapshot,
        onUnload: @escaping () async -> Void = {}
    ) -> GoldenStage {
        let server = ServerManager(
            testingState: .ready(alias: "qwen3.5-4b-4bit"),
            residency: snapshot
        )
        let chat = ChatViewModel(persistsConversations: false)
        return GoldenStage(
            ChatView(
                viewModel: chat,
                server: server,
                alias: .constant(selectedAlias),
                readiness: .ready(alias: selectedAlias),
                onUnloadResidentModels: onUnload
            )
            .environment(DownloadManager())
            .environment(QuickstartCoordinator())
            .frame(width: 720, height: 560)
        )
    }

    @MainActor
    @Test("The model-adjacent control unloads an idle selected resident model")
    func idleControlWiring() async throws {
        let probe = UnloadProbe()
        let stage = stage(snapshot: residentSnapshot()) {
            probe.calls += 1
        }

        try await stage.waitForIdentifier("ChatView.Residency.Unload")
        try stage.press("ChatView.Residency.Unload")
        try await stage.wait(for: "the composer unload action") { probe.calls == 1 }
    }

    @MainActor
    @Test("A busy resident pool disables the composer control")
    func busyControlIsDisabled() async throws {
        let stage = stage(snapshot: residentSnapshot(activeRequests: 1))
        try await stage.waitForIdentifier("ChatView.Residency.Unload")
        #expect(throws: GoldenStage.StageError.self) {
            try stage.press("ChatView.Residency.Unload")
        }
    }

    @MainActor
    @Test("The control does not imply an unloaded picker selection is resident")
    func coldSelectionHidesControl() async throws {
        let stage = stage(
            selectedAlias: "different-model",
            snapshot: residentSnapshot()
        )
        try await Task.sleep(for: .milliseconds(100))
        #expect(!stage.tree().contains { $0.id == "ChatView.Residency.Unload" })
    }

    @MainActor
    @Test("A ready legacy server without residency truth does not show zero-byte unload")
    func missingResidencyTruthHidesControl() async throws {
        let stage = stage(snapshot: .empty)
        try await Task.sleep(for: .milliseconds(100))
        #expect(!stage.tree().contains { $0.id == "ChatView.Residency.Unload" })
    }

    @Test("Visible copy names whole-pool scope when more than one model is resident")
    func visibleScopeCopy() {
        #expect(ChatView.composerResidentUnloadTitle(modelCount: 1) == "Unload")
        #expect(ChatView.composerResidentUnloadTitle(modelCount: 2) == "Unload all")
    }
}
