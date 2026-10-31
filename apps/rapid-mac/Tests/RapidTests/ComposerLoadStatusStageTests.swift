import AppKit
import Observation
import SwiftUI
import Testing
@testable import Rapid

/// The composer's load status on a real, mounted ``ChatView``: one surface
/// driven through download → load → failure → retry → ready, asserting on
/// the accessibility tree a VoiceOver user (and the AX golden flows) see.
@Suite("Composer load status — mounted chat")
@MainActor
struct ComposerLoadStatusStageTests {
    private static let alias = "qwen3.5-4b-4bit"
    private static let composeID = AutosizingTextView.composeAccessibilityIdentifier

    @MainActor
    @Observable
    final class Lifecycle {
        var readiness: ModelReadiness = .needsStart(alias: ComposerLoadStatusStageTests.alias)
        var progress: ModelReadiness.ProgressSnapshot?
        var actions: [ModelReadiness.Action] = []
    }

    private struct Host: View {
        let lifecycle: Lifecycle
        let chat: ChatViewModel
        let server: ServerManager

        var body: some View {
            ChatView(
                viewModel: chat,
                server: server,
                alias: .constant(ComposerLoadStatusStageTests.alias),
                readiness: lifecycle.readiness,
                loadProgress: lifecycle.progress,
                onReadinessAction: { lifecycle.actions.append($0) }
            )
            .environment(server)
            .environment(DownloadManager())
            .environment(QuickstartCoordinator())
            .frame(width: 720, height: 560)
        }
    }

    private func node(_ stage: GoldenStage, _ id: String) -> GoldenStage.Node? {
        stage.tree().first { $0.id == id }
    }

    @Test("A minimal status keeps its title accessible without inventing detail or progress")
    func minimalStatus() async throws {
        let status = ComposerLoadStatus(
            phase: .stopped, title: "Download stopped", percentText: nil,
            detail: nil, etaText: nil, progress: .none, action: nil
        )
        let stage = GoldenStage(VStack {
            ComposerLoadStatusView(status: status)
            ComposerProgressRail(progress: .none, label: status.title)
        })
        try await stage.waitForIdentifier("Composer.LoadStatus")
        #expect(stage.treeText().contains("Download stopped"))
        #expect(!stage.identifiers().contains("Composer.DownloadAction"))
        #expect(!stage.tree().contains { $0.role == "AXProgressIndicator" || $0.role == "AXBusyIndicator" })
    }

    @Test("Reduce Motion keeps the indeterminate segment mounted without a timeline")
    func reducedMotionSegment() async throws {
        let rail = ComposerProgressRail(progress: .indeterminate, label: "Loading")
        let stage = GoldenStage(
            rail.indeterminateSegment(trackWidth: 200, reduceMotion: true)
                .accessibilityIdentifier("ReducedSegment")
        )
        try await stage.waitForIdentifier("ReducedSegment")
        #expect(try #require(stage.frame(of: "ReducedSegment")).width == 60)
    }

    @Test("Blocked Return retains the draft through attention and repeated attempts")
    func blockedReturn() async throws {
        let lifecycle = Lifecycle()
        lifecycle.readiness = .starting(alias: Self.alias, detail: nil)
        let chat = ChatViewModel(persistsConversations: false)
        let server = ServerManager(testingState: .starting(alias: Self.alias))
        let stage = GoldenStage(Host(lifecycle: lifecycle, chat: chat, server: server))
        try await stage.waitForIdentifier(Self.composeID)
        try stage.setValue("Keep my unsent text", for: Self.composeID)
        try stage.submitComposer()
        try await Task.sleep(for: .milliseconds(100))
        try stage.submitComposer()
        try await Task.sleep(for: .milliseconds(1500))
        #expect(chat.composerDraft == "Keep my unsent text")
        #expect(chat.messages.isEmpty)
        #expect(stage.value(of: Self.composeID) == "Keep my unsent text")
        #expect(stage.identifiers().contains("Composer.LoadStatus"))
    }

    @Test("The acceleration badge fits beside the composer status")
    func accelerationBadge() async throws {
        let server = ServerManager(testingState: .ready(alias: Self.alias))
        server.applyActiveModelProfile(ServerModelProfile(
            id: Self.alias,
            speculativeDecoding: .init(
                configured: true, method: "mtp", runtimeState: .active,
                requestFallbackFeatures: []
            )
        ), forAlias: Self.alias)
        let stage = GoldenStage(ChatView(
            viewModel: ChatViewModel(persistsConversations: false), server: server,
            alias: .constant(Self.alias), readiness: .starting(alias: Self.alias, detail: nil)
        ).environment(server).environment(DownloadManager()).environment(QuickstartCoordinator()))
        try await stage.waitForIdentifier("ChatView.SpeculativeDecodingStatus")
        #expect(stage.treeText().contains("MTP ready"))
        let badge = try #require(stage.frame(of: "ChatView.SpeculativeDecodingStatus"))
        let send = try #require(stage.frame(of: "ChatView.SendOrStopButton"))
        #expect(badge.maxX <= send.minX)
    }

    @Test("The development capture matrix mounts the composer at every width and appearance")
    func snapshotMatrix() async throws {
        let server = ServerManager(testingState: .idle)
        var captures: [(AnyView, CGSize, NSAppearance.Name, String)] = []
        DevSnapshot.renderComposerSnapshots(
            chat: ChatViewModel(persistsConversations: false), server: server,
            downloads: DownloadManager(), quickstart: QuickstartCoordinator(), directory: "review"
        ) { captures.append(($0, $1, $2, $3)) }
        #expect(captures.count == 30)
        #expect(Set(captures.map { $0.3 }).count == 30)
        #expect(Set(captures.map { $0.1.width }) == [520, 800, 1240])
        #expect(Set(captures.map { $0.2 }) == [.aqua, .darkAqua])
        for (view, size, appearance, path) in captures {
            let stage = GoldenStage(view.environment(\.colorScheme, appearance == .darkAqua ? .dark : .light), size: size)
            try await stage.waitForIdentifier("Composer.LoadStatus")
            #expect(size.height == 640)
            #expect(stage.identifiers().contains(AutosizingTextView.composeAccessibilityIdentifier))
            let text = stage.treeText()
            if path.contains("downloading") || path.contains("long-alias") {
                let action = try #require(node(stage, "Composer.DownloadAction"))
                #expect(action.text == "Stop download")
                let frame = try #require(stage.frame(of: "Composer.DownloadAction"))
                #expect(frame.minX >= stage.windowFrame.minX && frame.maxX <= stage.windowFrame.maxX)
            } else {
                #expect(!stage.identifiers().contains("Composer.DownloadAction"))
            }
            if path.contains("downloading-unknown-total") {
                #expect(text.contains(DownloadProgress.formatBytes(Int64(1 << 30) * 29 / 10)))
                #expect(!text.contains("%"))
            } else if path.contains("downloading-") {
                #expect(text.contains("59%"))
            } else if path.contains("loading-") {
                #expect(text.contains("Loading the model into memory"))
                #expect(!text.contains("100%"))
            } else if path.contains("failed-") {
                #expect(stage.identifiers().contains("Readiness.Action"))
                #expect(text.contains("exited while loading"))
            } else {
                #expect(path.contains("long-alias"))
                #expect(text.contains(ComposerLayoutBoundsTests.longAlias))
            }
        }
    }

    @Test("Download, load, failure and ready all render in the composer and keep the draft")
    func lifecycleInTheComposer() async throws {
        let lifecycle = Lifecycle()
        let chat = ChatViewModel(persistsConversations: false)
        let server = ServerManager(testingState: .starting(alias: Self.alias))
        let stage = GoldenStage(Host(lifecycle: lifecycle, chat: chat, server: server))
        let gib: Int64 = 1 << 30

        // A decision state: the notice above the card, no composer slot.
        try await stage.waitForIdentifier("Readiness.Action")
        #expect(node(stage, "Composer.LoadStatus") == nil)

        // The user drafts before starting.
        try await stage.waitForIdentifier(Self.composeID)
        try stage.setValue("Summarise my notes", for: Self.composeID)
        #expect(chat.composerDraft == "Summarise my notes")

        // Downloading: determinate progress with real numbers.
        lifecycle.progress = .init(
            activity: .downloading, fraction: 0.59,
            bytesDownloaded: gib * 17 / 10, totalBytes: gib * 29 / 10,
            etaText: "< 1 min left"
        )
        lifecycle.readiness = .downloading(alias: Self.alias, detail: nil, fraction: 0.59)
        try await stage.waitForIdentifier("Composer.LoadStatus")
        let downloadingRail = try #require(node(stage, "Composer.LoadProgress"))
        #expect(downloadingRail.role == "AXProgressIndicator")
        #expect(stage.treeText().contains("\(DownloadProgress.formatBytes(gib * 17 / 10)) of \(DownloadProgress.formatBytes(gib * 29 / 10))"))
        #expect(stage.treeText().contains("59%"))
        // The banner stood down: one description of the moment, not two.
        #expect(node(stage, "Readiness.Action") == nil)
        #expect(!stage.identifiers().contains("Readiness.Band"))
        #expect(stage.value(of: Self.composeID) == "Summarise my notes")

        // Loading a cached model: indeterminate, and the ~1.0 fraction the
        // byte monitor reports must not appear as a percentage.
        lifecycle.progress = .init(
            activity: .loading, fraction: 1.0,
            bytesDownloaded: gib * 29 / 10, totalBytes: gib * 29 / 10
        )
        lifecycle.readiness = .starting(alias: Self.alias, detail: nil)
        try await stage.wait(for: "the loading title") {
            stage.treeText().contains("Loading the model into memory…")
        }
        // AppKit's role for progress with no value: an indeterminate
        // indicator, never a determinate one pinned at some number.
        #expect(node(stage, "Composer.LoadProgress")?.role == "AXBusyIndicator")
        #expect(!stage.treeText().contains("100%"))
        #expect(stage.value(of: Self.composeID) == "Summarise my notes")

        // A failed start: the recovery action lives in the slot, under the
        // identifier the golden flows already address.
        lifecycle.progress = nil
        lifecycle.readiness = .failed(
            alias: Self.alias, message: "The model process exited while loading.",
            action: .retry(alias: Self.alias)
        )
        try await stage.waitForIdentifier("Readiness.Action")
        try await stage.waitForIdentifierGone("Composer.LoadProgress")
        #expect(stage.treeText().contains("The model process exited while loading."))
        try stage.press("Readiness.Action")
        #expect(lifecycle.actions == [.retry(alias: Self.alias)])
        #expect(stage.value(of: Self.composeID) == "Summarise my notes")

        // Retry → loading → ready: the slot closes in place and the draft is
        // exactly what the user typed, still unsent.
        lifecycle.readiness = .starting(alias: Self.alias, detail: nil)
        try await stage.waitForIdentifier("Composer.LoadProgress")
        lifecycle.readiness = .ready(alias: Self.alias)
        try await stage.waitForIdentifierGone("Composer.LoadStatus")
        #expect(node(stage, "Composer.LoadProgress") == nil)
        #expect(node(stage, "Readiness.Action") == nil)
        #expect(stage.value(of: Self.composeID) == "Summarise my notes")
        #expect(chat.composerDraft == "Summarise my notes")
        #expect(chat.messages.isEmpty)
    }
}
