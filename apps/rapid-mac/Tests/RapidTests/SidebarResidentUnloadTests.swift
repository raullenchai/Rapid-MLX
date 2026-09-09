import SwiftUI
import Testing
@testable import Rapid

@Suite("Sidebar resident unload")
struct SidebarResidentUnloadTests {
    @MainActor
    private final class UnloadProbe {
        var calls = 0
    }

    private func residentSnapshot(
        activeRequests: Int = 0,
        includeTextModel: Bool = true,
        audioActiveRequests: Int? = nil
    ) -> ModelResidencySnapshot {
        let gib = UInt64(1) << 30
        return ModelResidencySnapshot(
            memoryLimitBytes: 14 * gib,
            memoryUsedBytes: 6 * gib,
            memoryAvailableBytes: 8 * gib,
            idleTTLSeconds: 900,
            loadsTotal: 1,
            evictionsTotal: 0,
            models: includeTextModel ? [
                ResidentModelStatus(
                    id: "qwen3.5-4b-4bit",
                    modelPath: "mlx-community/qwen3.5-4b-4bit",
                    aliases: ["qwen3.5-4b-4bit"],
                    modality: "text",
                    state: "resident",
                    pinned: true,
                    primary: true,
                    activeRequests: activeRequests,
                    estimatedBytes: 6 * gib,
                    measuredBytes: nil,
                    idleSeconds: 12
                ),
            ] : [],
            audioLanes: audioActiveRequests.map {
                [ResidentAudioLaneStatus(
                    lane: "tts",
                    model: "test/voice",
                    state: "resident",
                    activeRequests: $0
                )]
            } ?? []
        )
    }

    @MainActor
    @Test("The AX eject control invokes unload when idle and rejects a busy press")
    func controlWiring() async throws {
        let chat = ChatViewModel(persistsConversations: false)
        let probe = UnloadProbe()
        let idleServer = ServerManager(
            testingState: .ready(alias: "qwen3.5-4b-4bit"),
            residency: residentSnapshot()
        )
        let idleStage = GoldenStage(
            SidebarView(
                selection: .constant(.chat),
                chat: chat,
                onNewChat: {},
                onSelectConversation: { _ in },
                server: idleServer,
                onUnloadResidentModels: { probe.calls += 1 }
            )
            .frame(width: SidebarView.columnIdealWidth, height: 640)
        )

        try await idleStage.waitForIdentifier("Sidebar.Residency")
        try await idleStage.waitForIdentifier("Sidebar.Residency.Unload")
        try idleStage.press("Sidebar.Residency.Unload")
        try await idleStage.wait(for: "the unload action") { probe.calls == 1 }

        let busyServer = ServerManager(
            testingState: .ready(alias: "qwen3.5-4b-4bit"),
            residency: residentSnapshot(activeRequests: 1)
        )
        let busyStage = GoldenStage(
            SidebarView(
                selection: .constant(.chat),
                chat: chat,
                onNewChat: {},
                onSelectConversation: { _ in },
                server: busyServer,
                onUnloadResidentModels: { probe.calls += 1 }
            )
            .frame(width: SidebarView.columnIdealWidth, height: 640)
        )

        try await busyStage.waitForIdentifier("Sidebar.Residency.Unload")
        #expect(throws: GoldenStage.StageError.self) {
            try busyStage.press("Sidebar.Residency.Unload")
        }
        #expect(probe.calls == 1)

        let busyAudioServer = ServerManager(
            testingState: .ready(alias: "qwen3.5-4b-4bit"),
            residency: residentSnapshot(audioActiveRequests: 1)
        )
        let busyAudioStage = GoldenStage(
            SidebarView(
                selection: .constant(.chat),
                chat: chat,
                onNewChat: {},
                onSelectConversation: { _ in },
                server: busyAudioServer,
                onUnloadResidentModels: { probe.calls += 1 }
            )
            .frame(width: SidebarView.columnIdealWidth, height: 640)
        )

        try await busyAudioStage.waitForIdentifier("Sidebar.Residency.Unload")
        #expect(throws: GoldenStage.StageError.self) {
            try busyAudioStage.press("Sidebar.Residency.Unload")
        }
        #expect(probe.calls == 1)
    }

    @Test("Audio-only residency stays visible and participates in busy state")
    func audioResidencyPresentation() {
        let idle = residentSnapshot(
            includeTextModel: false,
            audioActiveRequests: 0
        )
        let busy = residentSnapshot(
            includeTextModel: false,
            audioActiveRequests: 1
        )

        #expect(SidebarView.hasResidentWorkloads(idle))
        #expect(SidebarView.residentWorkloadCount(idle) == 1)
        #expect(!SidebarView.hasActiveResidentRequests(idle))
        #expect(SidebarView.hasActiveResidentRequests(busy))
    }

    @Test("Unload result copy explains a refused or unavailable stop")
    func resultCopy() {
        #expect(SidebarView.residentUnloadMessage(for: .stopped) == nil)
        #expect(
            SidebarView.residentUnloadMessage(for: .busy)
                == "A request is still using the models. Stop it, then try again."
        )
        #expect(
            SidebarView.residentUnloadMessage(for: .unavailable)
                == "Rapid couldn't confirm that the models were idle. Wait a moment, then try again."
        )
    }

    @Test("Memory summary keeps one shared unit in the narrow sidebar")
    func compactMemorySummary() {
        let gib = UInt64(1) << 30
        #expect(
            SidebarView.memorySummary(
                usedBytes: 6 * gib,
                limitBytes: 204 * gib
            ) == "6 / 204 GB"
        )
    }

    @Test("Unload is available only while the resident pool is idle")
    func disabledState() {
        #expect(!SidebarView.residentUnloadDisabled(
            isOperating: false,
            chatIsStreaming: false,
            hasActiveRequests: false
        ))
        #expect(SidebarView.residentUnloadDisabled(
            isOperating: true,
            chatIsStreaming: false,
            hasActiveRequests: false
        ))
        #expect(SidebarView.residentUnloadDisabled(
            isOperating: false,
            chatIsStreaming: true,
            hasActiveRequests: false
        ))
        #expect(SidebarView.residentUnloadDisabled(
            isOperating: false,
            chatIsStreaming: false,
            hasActiveRequests: true
        ))
    }

    @Test("Accessible copy names the scope and memory released")
    func accessibleCopy() {
        let gib = UInt64(1) << 30
        #expect(
            SidebarView.residentUnloadLabel(
                modelCount: 1,
                memoryUsedBytes: 6 * gib
            ) == "Unload model and free 6 GB"
        )
        #expect(
            SidebarView.residentUnloadLabel(
                modelCount: 2,
                memoryUsedBytes: 10 * gib
            ) == "Unload all models and free 10 GB"
        )
    }

    @Test("Help explains blocked and in-progress states")
    func helpCopy() {
        #expect(
            SidebarView.residentUnloadHelp(
                isOperating: false,
                hasActiveResponse: false,
                enabledLabel: "Unload model and free 6 GB"
            ) == "Unload model and free 6 GB"
        )
        #expect(
            SidebarView.residentUnloadHelp(
                isOperating: false,
                hasActiveResponse: true,
                enabledLabel: "unused"
            ) == "Stop the active response before unloading models"
        )
        #expect(
            SidebarView.residentUnloadHelp(
                isOperating: true,
                hasActiveResponse: true,
                enabledLabel: "unused"
            ) == "Unloading models…"
        )
    }
}
