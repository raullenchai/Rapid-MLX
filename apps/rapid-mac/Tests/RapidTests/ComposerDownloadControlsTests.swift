import AppKit
import Darwin
import SwiftUI
import Testing
@testable import Rapid

@Suite("Composer download controls")
@MainActor
struct ComposerDownloadControlsTests {
    private let alias = "qwen3.5-4b-4bit"

    @Test("Only controllable download tasks move out of the global strip")
    func ownership() {
        let running = ModelReadiness.downloading(alias: alias, detail: nil, fraction: 0.3)
        #expect(ComposerLoadStatus.ownsDownload(readiness: running, status: .running))
        #expect(ComposerLoadStatus.ownsDownload(readiness: running, status: .cancelled))
        #expect(!ComposerLoadStatus.ownsDownload(readiness: running, status: nil))
        #expect(!ComposerLoadStatus.ownsDownload(readiness: .starting(alias: alias, detail: nil), status: .running))
        #expect(!ComposerLoadStatus.ownsDownload(readiness: .ready(alias: alias), status: .completed))
        let stopped = ModelReadiness.needsDownload(alias: alias, sizeText: nil)
        #expect(ComposerLoadStatus.ownsDownload(readiness: stopped, status: .cancelled))
        #expect(!ComposerLoadStatus.ownsDownload(readiness: stopped, status: .failed(message: "Offline")))
        let slot = ComposerLoadStatus.resolve(readiness: stopped, progress: nil, downloadStatus: .cancelled)
        #expect(slot?.phase == .stopped)
        #expect(slot?.progress == ComposerLoadStatus.Progress.none)
        #expect(slot?.percentText == nil)
        #expect(ComposerLoadStatus.resolve(readiness: stopped, progress: nil) == nil)
    }

    @Test("Global strip hides only the owned alias and restores its stop and resume controls")
    func globalStrip() async throws {
        let downloads = DownloadManager()
        _ = downloads._testingSeedJob(alias: alias)
        _ = downloads._testingSeedJob(alias: "background-image")
        let hidden = GoldenStage(DownloadStrip(downloads: downloads, composerOwnedAlias: alias))
        try await hidden.waitForIdentifier("DownloadStrip.Cancel.background-image")
        #expect(!hidden.identifiers().contains("DownloadStrip.Cancel.\(alias)"))
        let visible = GoldenStage(DownloadStrip(downloads: downloads))
        try await visible.waitForIdentifier("DownloadStrip.Cancel.\(alias)")
        downloads._testingFinish(alias: alias, status: 15, reason: .uncaughtSignal, wasCancelling: true)
        try await visible.waitForIdentifier("DownloadStrip.Resume.\(alias)")
        #expect(!hidden.identifiers().contains("DownloadStrip.Resume.\(alias)"))
        #expect(downloads.canRetryDownload(alias: alias))
    }

    @Test("Composer shows Stop then Resume without losing the draft; controls fit narrow width")
    func composerControls() async throws {
        let downloads = DownloadManager()
        _ = downloads._testingSeedJob(alias: alias)
        let chat = ChatViewModel(persistsConversations: false)
        chat.composerDraft = "Keep this draft"
        let server = ServerManager(testingState: .idle)
        func surface(_ readiness: ModelReadiness) -> some View {
            ChatView(viewModel: chat, server: server, alias: .constant(alias), readiness: readiness)
                .environment(downloads).environment(server).environment(QuickstartCoordinator())
                .frame(width: 520, height: 640)
        }
        let running = GoldenStage(surface(.downloading(alias: alias, detail: nil, fraction: 0.3)), size: CGSize(width: 520, height: 640))
        try await running.waitForIdentifier("Composer.DownloadAction")
        #expect(running.treeText().contains("Stop download"))
        #expect(running.help(of: "Composer.DownloadAction")?.hasPrefix("Stops the download") == true)
        let action = try #require(running.frame(of: "Composer.DownloadAction"))
        #expect(action.minX >= running.windowFrame.minX && action.maxX <= running.windowFrame.maxX)
        downloads._testingFinish(alias: alias, status: 15, reason: .uncaughtSignal, wasCancelling: true)
        // The job cancels before the parent supplies fresh readiness.
        try await running.wait(for: "stopped copy before readiness catches up") {
            running.treeText().contains("Download stopped")
                && running.treeText().contains("Resume download")
        }
        #expect(!running.identifiers().contains("Composer.LoadProgress"))
        #expect(!running.treeText().contains("Stop download"))
        #expect(!running.treeText().contains("Downloading the model"))
        #expect(running.value(of: AutosizingTextView.composeAccessibilityIdentifier) == "Keep this draft")
        let stopped = GoldenStage(surface(.needsDownload(alias: alias, sizeText: nil)), size: CGSize(width: 520, height: 640))
        try await stopped.waitForIdentifier("Composer.DownloadAction")
        #expect(stopped.treeText().contains("Resume download"))
        #expect(stopped.help(of: "Composer.DownloadAction") == "Resumes the download using available cached data.")
        #expect(!stopped.identifiers().contains("Readiness.Action"))
        #expect(chat.composerDraft == "Keep this draft")
        if let dir = ProcessInfo.processInfo.environment["RAPID_COMPOSER_REVIEW_DIR"] {
            try export(surface(.needsDownload(alias: alias, sizeText: nil)), name: "composer-stopped", directory: dir)
            _ = downloads._testingSeedJob(alias: alias)
            try export(surface(.downloading(alias: alias, detail: nil, fraction: 0.3)), name: "composer-downloading", directory: dir)
            try export(DownloadStrip(downloads: downloads).frame(width: 520, height: 100), name: "global-background-download", directory: dir, height: 100)
        }
        #expect(!stopped.identifiers().contains("Composer.LoadProgress"))
    }
    @Test("Stop and resume use the same manager, preserving job metadata and waiting for cleanup", arguments: [false, true])
    func actualProcessStopAndResume(useGlobalResume: Bool) async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let binary = root.appendingPathComponent("fake-pull")
        try "#!/bin/sh\nexec /bin/sleep 30\n".write(to: binary, atomically: true, encoding: .utf8)
        chmod(binary.path, 0o755)
        let downloads = DownloadManager(
            binaryPath: binary, binaryLocator: { binary }, cacheAccessCheck: { nil }
        )
        defer { downloads.beginShutdown(); downloads.finishShutdown() }
        #expect(downloads.startDownload(alias: alias, hfPath: "test/model", totalBytes: 1024))
        let first = try #require(downloads.job(for: alias))
        // The pull spawns after the async cache-access preflight.
        let spawnDeadline = Date().addingTimeInterval(5)
        while !downloads._testingHasProcess(alias: alias) && Date() < spawnDeadline {
            try await Task.sleep(for: .milliseconds(10))
        }
        #expect(!downloads.canRetryDownload(alias: alias))
        let chat = ChatViewModel(persistsConversations: false)
        let server = ServerManager(testingState: .idle)
        func stage(_ readiness: ModelReadiness) -> GoldenStage {
            GoldenStage(ChatView(viewModel: chat, server: server, alias: .constant(alias), readiness: readiness)
                .environment(downloads).environment(server).environment(QuickstartCoordinator())
                .frame(width: 520, height: 640), size: CGSize(width: 520, height: 640))
        }
        let running = stage(.downloading(alias: alias, detail: nil, fraction: nil))
        try await running.waitForIdentifier("Composer.DownloadAction")
        try running.press("Composer.DownloadAction")
        #expect(first.status == .cancelled)
        #expect(!downloads.canRetryDownload(alias: alias))
        let deadline = Date().addingTimeInterval(5)
        while !downloads.canRetryDownload(alias: alias) && Date() < deadline {
            try await Task.sleep(for: .milliseconds(25))
        }
        #expect(downloads.canRetryDownload(alias: alias))
        let stopped = useGlobalResume
            ? GoldenStage(DownloadStrip(downloads: downloads))
            : stage(.needsDownload(alias: alias, sizeText: nil))
        let resumeID = useGlobalResume ? "DownloadStrip.Resume.\(alias)" : "Composer.DownloadAction"
        try await stopped.waitForIdentifier(resumeID)
        try stopped.press(resumeID)
        let resumed = try #require(downloads.job(for: alias))
        #expect(resumed.instanceID != first.instanceID)
        #expect(resumed.status == .running)
        #expect(resumed.hfPath == first.hfPath)
        #expect(resumed.totalBytes == first.totalBytes)
        #expect(!downloads.canRetryDownload(alias: alias))
    }

    /// Optional visual evidence from the same real views used in the AX checks.
    private func export<V: View>(_ view: V, name: String, directory: String, height: CGFloat = 640) throws {
        try FileManager.default.createDirectory(atPath: directory, withIntermediateDirectories: true)
        let host = NSHostingView(rootView: view.environment(\.colorScheme, .light))
        let size = CGSize(width: 520, height: height)
        let window = NSWindow(contentRect: CGRect(origin: .zero, size: size), styleMask: .borderless, backing: .buffered, defer: false)
        window.appearance = NSAppearance(named: .aqua)
        window.contentView = host
        host.frame = CGRect(origin: .zero, size: size)
        host.layoutSubtreeIfNeeded()
        window.displayIfNeeded()
        RunLoop.current.run(until: Date().addingTimeInterval(0.2))
        host.layoutSubtreeIfNeeded()
        let rep = try #require(host.bitmapImageRepForCachingDisplay(in: host.bounds))
        host.cacheDisplay(in: host.bounds, to: rep)
        let png = try #require(rep.representation(using: .png, properties: [:]))
        try png.write(to: URL(fileURLWithPath: directory).appendingPathComponent(name + ".png"))
    }

}
